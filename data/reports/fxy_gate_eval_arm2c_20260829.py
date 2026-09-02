"""ARM 2c -- copy of ``fxy_gate_eval_arm2b_20260829.py`` with ONLY ``OUT`` and
this header changed.  Re-run through the FIXED serve path
(``featurize.serve_provenance``) with the per-cell calibration artifacts REFIT on
that same fixed path on BOTH sides (``data/models/s1i`` and the arm-2 candidate).

INFORMATIONAL: this is an arm-2c READING, not a re-adjudication of arm 2.  The
arm-2 verdict (ORIGINAL serve path, pre-refit artifacts) is final and unchanged.

Original header follows.

ARM 2 copy of ``fxy_gate_eval_20260829.py`` (that file is untouched).

Adjudicates the arm-2 candidate against prereg
``data/reports/fxy_head_prereg_20260829.md`` **Amendment B** (G2′/G3′, the
serving-proxy yardstick) while still reporting the A.5/A.6 readings.

Read-only.  Loads the candidate ensemble EXACTLY as serving does
(:class:`lpopt.model.model_api.PosValCnnBackend` from the checkpoint dir, patterns
rebuilt from each store row's packed ``pattern`` and ``CaseKey(case_pair, feed)`` --
the same construction :func:`lpopt.curriculum.score_no_regression_cell` uses), calls
``predict_fxy`` for the head, and compares it against the linear prior computed the
way the prereg computed it (train-fold-fitted ``a*f_r + b`` on the MEASURED ``f_r``).

Writes ``data/reports/fxy_gate_eval_20260829.json``.  Promotes nothing, renames
nothing, edits no deck.

    python data/reports/fxy_gate_eval_arm2c_20260829.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from lpopt.data.schema import unpack_pattern                     # noqa: E402
from lpopt.model.model_api import PosValCnnBackend, CaseKey      # noqa: E402
from lpopt.model import ab_paired                                # noqa: E402
from lpopt.search import acquisition as acq                      # noqa: E402

CAND = ROOT / "data/models/20260829_163820"
PREV = ROOT / "data/models/s1i"
STORE = ROOT / "data/store"
SPLIT = ROOT / "data/splits/S1j.json"
OUT = ROOT / "data/reports/fxy_gate_eval_arm2c_20260829.json"

# Amendment A.4 -- the prior the run actually fitted (S1j TRAIN fold).
PRIOR_A, PRIOR_B = 1.216147350612086, -0.24878433827113983
G2_BAR = 0.0463          # A.5  (train-fitted prior's holdout resid sd)
G2B_BAR = 0.0355         # A.5  (same prior's holdout MAE) -- report only
G3_BAR = 0.8944          # A.6  (unweighted mean rho over the 11 >=20-label cells)
G3_MIN_CELL = 20
G4_LO, G4_HI = 0.55, 0.80
# Amendment B.5 -- the BINDING arm-2 bars (serving proxy on s1i, results §8).
G2P_BAR = 0.0767         # B.5 G2′  MAE(f_xy) must be <
G3P_BAR = 0.7263         # B.5 G3′  rho-bar must be >
PROXY_S1I_BIAS = 0.0053  # B.5 report-only reference


def _sp(a, b):
    if len(a) < 3:
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r)


def main() -> int:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    val_ids = set(split["val_ids"] if "val_ids" in split else split["val"])
    df = pd.read_parquet(STORE / "records.parquet").drop_duplicates("record_id")

    sub = df[df["record_id"].isin(val_ids)]
    sub = sub[(sub["converged"] == True) & sub["f_xy"].notna()]   # noqa: E712
    sub = sub.reset_index(drop=True)
    n = len(sub)
    print(f"S1j VAL labelled+converged rows: {n}")

    pats = [unpack_pattern(str(p)) for p in sub["pattern"]]
    cases = [CaseKey(str(p), int(f)) for p, f in zip(sub["case_pair"], sub["feed"])]

    def _load(d):
        return PosValCnnBackend.from_dir(d, store_dir=STORE, library_id="ga80",
                                         device="cpu")

    cand = _load(CAND)
    print("candidate targets:", cand.target_names)

    mu, sigma, source = cand.predict_fxy(pats, cases)
    truth = sub["f_xy"].to_numpy(dtype=float)
    f_r = sub["f_r"].to_numpy(dtype=float)
    prior = PRIOR_A * f_r + PRIOR_B
    proxy = acq.FXY_PROXY_SLOPE * f_r + acq.FXY_PROXY_INTERCEPT

    res: dict = {
        "n_holdout": int(n), "source": source,
        "prior": {"a": PRIOR_A, "b": PRIOR_B},
        "proxy": {"slope": acq.FXY_PROXY_SLOPE,
                  "intercept": acq.FXY_PROXY_INTERCEPT,
                  "resid_sd": acq.FXY_PROXY_RESID_SD,
                  "sigma_k": acq.FXY_PROXY_SIGMA_K},
    }

    def _err(pred, name):
        e = pred - truth
        return {"name": name, "mae": float(np.mean(np.abs(e))),
                "bias": float(np.mean(e)), "resid_sd": float(np.std(e, ddof=1)),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "max_abs": float(np.max(np.abs(e))),
                "p95_abs": float(np.percentile(np.abs(e), 95))}

    res["errors"] = {k: _err(v, k) for k, v in
                     (("HEAD", mu), ("PRIOR", prior), ("PROXY", proxy))}

    # ---- G2 ----------------------------------------------------------------
    mae_head = res["errors"]["HEAD"]["mae"]
    res["G2"] = {"bar": G2_BAR, "measured": mae_head, "pass": mae_head < G2_BAR}
    res["G2b"] = {"bar": G2B_BAR, "measured": mae_head,
                  "strong_form": mae_head < G2B_BAR}

    # ---- G3 ----------------------------------------------------------------
    sub = sub.assign(_cell=sub["case_pair"].astype(str) + "/f"
                     + sub["feed"].astype(int).astype(str))
    cells = sub["_cell"].to_numpy()
    per_cell = []
    head_by_cell, prior_by_cell = {}, {}
    for c in sorted(set(cells)):
        m = cells == c
        if int(m.sum()) < G3_MIN_CELL:
            continue
        rh, rp = _sp(truth[m], mu[m]), _sp(truth[m], prior[m])
        rx = _sp(truth[m], proxy[m])
        head_by_cell[c], prior_by_cell[c] = rh, rp
        per_cell.append({
            "cell": c, "n": int(m.sum()),
            "rho_head": rh, "rho_prior": rp, "rho_proxy": rx,
            "mae_head": float(np.mean(np.abs(mu[m] - truth[m]))),
            "mae_prior": float(np.mean(np.abs(prior[m] - truth[m]))),
            "resid_sd_prior": float(np.std(prior[m] - truth[m], ddof=1)),
        })
    rh_bar = float(np.mean([r["rho_head"] for r in per_cell]))
    rp_bar = float(np.mean([r["rho_prior"] for r in per_cell]))
    nsum = sum(r["n"] for r in per_cell)
    res["G3"] = {"bar": G3_BAR, "n_cells": len(per_cell), "n_rows": nsum,
                 "rho_head_mean": rh_bar, "rho_prior_mean": rp_bar,
                 "rho_head_wmean": float(sum(r["rho_head"] * r["n"] for r in per_cell) / nsum),
                 "delta": rh_bar - rp_bar,
                 "pass": rh_bar >= G3_BAR, "per_cell": per_cell}

    # prereg A.6: |delta| <= 0.05 -> cell-clustered paired BCa CI decides tie
    if abs(rh_bar - rp_bar) <= 0.05:
        pd_ = ab_paired.paired_cell_bootstrap(
            head_by_cell, prior_by_cell, metric="within_cell_spearman_f_xy",
            arm="HEAD", control="PRIOR", higher_is_better=True,
            aggregate="mean", seed=0)
        res["G3"]["paired"] = pd_.to_dict()
        res["G3"]["tie"] = bool(pd_.ci_lo <= 0.0 <= pd_.ci_hi)

    # ---- G4 ----------------------------------------------------------------
    cov = float(np.mean(np.abs(truth - mu) <= sigma))
    res["G4"] = {"lo": G4_LO, "hi": G4_HI, "coverage": cov,
                 "sigma_mean": float(np.mean(sigma)),
                 "sigma_median": float(np.median(sigma)),
                 "sigma_min": float(np.min(sigma)),
                 "sigma_max": float(np.max(sigma)),
                 "pass": G4_LO <= cov <= G4_HI}

    # ---- SERVE-PATH proxy (Amendment B.5 bar) ------------------------------
    # acquisition.fxy_proxy reads prediction.mean[:, 0] -- the CALIBRATED predicted
    # F_r -- so the honest alternative to the head is the proxy on the incumbent's
    # served F_r column, not the proxy on a measured F_r the serving path never sees.
    prev_serve = _load(PREV)
    p_prev = prev_serve.predict(pats, cases)
    p_cand_pred = cand.predict(pats, cases)
    proxy_s1i, proxy_s1i_sd = acq.fxy_proxy(p_prev)
    proxy_cand, proxy_cand_sd = acq.fxy_proxy(p_cand_pred)
    res["errors"]["PROXY_S1I_SERVE"] = _err(proxy_s1i, "PROXY_S1I_SERVE")
    res["errors"]["PROXY_CAND_SERVE"] = _err(proxy_cand, "PROXY_CAND_SERVE")

    def _rho_by_cell(pred):
        out = {}
        for c in sorted(set(cells)):
            m = cells == c
            if int(m.sum()) >= G3_MIN_CELL:
                out[c] = _sp(truth[m], pred[m])
        return out

    proxy_s1i_by_cell = _rho_by_cell(proxy_s1i)
    proxy_cand_by_cell = _rho_by_cell(proxy_cand)
    rho_proxy_s1i = float(np.mean(list(proxy_s1i_by_cell.values())))
    rho_proxy_cand = float(np.mean(list(proxy_cand_by_cell.values())))

    res["serve_path"] = {
        "PROXY_S1I": {"mae": res["errors"]["PROXY_S1I_SERVE"]["mae"],
                      "bias": res["errors"]["PROXY_S1I_SERVE"]["bias"],
                      "rho_mean": rho_proxy_s1i,
                      "sigma_mean": float(np.mean(proxy_s1i_sd)),
                      "coverage68": float(np.mean(np.abs(truth - proxy_s1i) <= proxy_s1i_sd))},
        "PROXY_CAND": {"mae": res["errors"]["PROXY_CAND_SERVE"]["mae"],
                       "bias": res["errors"]["PROXY_CAND_SERVE"]["bias"],
                       "rho_mean": rho_proxy_cand,
                       "sigma_mean": float(np.mean(proxy_cand_sd)),
                       "coverage68": float(np.mean(np.abs(truth - proxy_cand) <= proxy_cand_sd))},
        "HEAD": {"mae": res["errors"]["HEAD"]["mae"],
                 "bias": res["errors"]["HEAD"]["bias"],
                 "rho_mean": rh_bar,
                 "sigma_mean": float(np.mean(sigma)),
                 "coverage68": float(np.mean(np.abs(truth - mu) <= sigma))},
        "per_cell_rho_proxy_s1i": proxy_s1i_by_cell,
    }

    # ---- G2' / G3' (Amendment B.5, BINDING for arm 2) -----------------------
    res["G2p"] = {"bar": G2P_BAR, "measured": mae_head,
                  "pass": bool(mae_head < G2P_BAR),
                  "source": "serving proxy on s1i (results 20260829 sec.8)"}
    d3 = rh_bar - G3P_BAR
    g3p = {"bar": G3P_BAR, "measured": rh_bar, "delta": d3,
           "point_pass": bool(rh_bar > G3P_BAR),
           "rho_proxy_s1i_measured_here": rho_proxy_s1i}
    # B.5 tie rule: |delta| <= 0.05 -> cell-clustered paired BCa CI, tie == FAIL
    if abs(rh_bar - rho_proxy_s1i) <= 0.05:
        pd2 = ab_paired.paired_cell_bootstrap(
            head_by_cell, proxy_s1i_by_cell, metric="within_cell_spearman_f_xy",
            arm="HEAD", control="PROXY_S1I", higher_is_better=True,
            aggregate="mean", seed=0)
        g3p["paired_vs_proxy"] = pd2.to_dict()
        g3p["tie"] = bool(pd2.ci_lo <= 0.0 <= pd2.ci_hi)
        g3p["pass"] = bool(g3p["point_pass"] and not g3p["tie"])
    else:
        g3p["tie"] = False
        g3p["pass"] = g3p["point_pass"]
    res["G3p"] = g3p

    # ---- member sanity: leave-one-out ensembles ----------------------------
    all_members, all_metas = list(cand.members), list(cand.metas)
    seeds = [str(m.get("seed")) for m in all_metas]
    big = [c for c in sorted(set(cells)) if int((cells == c).sum()) >= G3_MIN_CELL]

    def _subset(idx):
        cand.members = [all_members[i] for i in idx]
        cand.metas = [all_metas[i] for i in idx]
        try:
            m2, s2, _ = cand.predict_fxy(pats, cases)
        finally:
            cand.members, cand.metas = all_members, all_metas
        rr = [_sp(truth[cells == c], m2[cells == c]) for c in big]
        return {"mae": float(np.mean(np.abs(m2 - truth))),
                "rho_mean": float(np.mean(rr)),
                "bias": float(np.mean(m2 - truth)),
                "coverage": float(np.mean(np.abs(truth - m2) <= s2))}

    rng = range(len(all_members))
    res["loo"] = {f"drop_{seeds[d]}": _subset([i for i in rng if i != d])
                  for d in rng}
    res["single_member"] = {seeds[i]: _subset([i]) for i in rng}

    # ---- legacy fixed yardstick on the SAME slice, cand vs s1i -------------
    prev = prev_serve
    yard = {}
    p_new = p_cand_pred
    p_old = p_prev
    # surrogate column order
    from lpopt.model.model_api import TARGET_NAMES as SURR
    surr = list(SURR)
    for name, truth_col in (("cyclen", "cyclen"), ("F_r", "f_r"),
                            ("F_q", "f_q"), ("CBC_max", "cbc_max")):
        if truth_col not in sub.columns or name not in surr:
            continue
        col = surr.index(name)
        t = pd.to_numeric(sub[truth_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t)
        if ok.sum() < 3:
            continue
        o, nw = p_old.mean[ok, col], p_new.mean[ok, col]
        rr_o, rr_n = [], []
        for c in sorted(set(cells[ok])):
            msk = cells[ok] == c
            if int(msk.sum()) >= G3_MIN_CELL:
                rr_o.append(_sp(t[ok][msk], o[msk]))
                rr_n.append(_sp(t[ok][msk], nw[msk]))
        yard[name] = {
            "n": int(ok.sum()),
            "rho_global_prev": _sp(t[ok], o), "rho_global_new": _sp(t[ok], nw),
            "rho_cellmean_prev": float(np.mean(rr_o)) if rr_o else float("nan"),
            "rho_cellmean_new": float(np.mean(rr_n)) if rr_n else float("nan"),
            "mae_prev": float(np.mean(np.abs(o - t[ok]))),
            "mae_new": float(np.mean(np.abs(nw - t[ok]))),
        }
    # node_peak via the map head
    if "node_peak" in sub.columns:
        t = pd.to_numeric(sub["node_peak"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t)
        if ok.sum() >= 3:
            pk_o, _, cv_o, _ = prev.predict_map_flatness(pats, cases)
            pk_n, _, cv_n, _ = cand.predict_map_flatness(pats, cases)
            for nm, a, b_ in (("node_peak", pk_o, pk_n),):
                yard[nm] = {
                    "n": int(ok.sum()),
                    "rho_global_prev": _sp(t[ok], a[ok]),
                    "rho_global_new": _sp(t[ok], b_[ok]),
                    "mae_prev": float(np.mean(np.abs(a[ok] - t[ok]))),
                    "mae_new": float(np.mean(np.abs(b_[ok] - t[ok]))),
                }
            if "map_cov" in sub.columns:
                t2 = pd.to_numeric(sub["map_cov"], errors="coerce").to_numpy(dtype=float)
                ok2 = np.isfinite(t2)
                if ok2.sum() >= 3:
                    yard["map_cov"] = {
                        "n": int(ok2.sum()),
                        "rho_global_prev": _sp(t2[ok2], cv_o[ok2]),
                        "rho_global_new": _sp(t2[ok2], cv_n[ok2]),
                        "mae_prev": float(np.mean(np.abs(cv_o[ok2] - t2[ok2]))),
                        "mae_new": float(np.mean(np.abs(cv_n[ok2] - t2[ok2]))),
                    }
    res["legacy_yardstick"] = yard

    # calibration artifacts present on each side (asymmetry check)
    res["artifacts"] = {
        "candidate": sorted(p.name for p in CAND.glob("*.json")),
        "incumbent": sorted(p.name for p in PREV.glob("*.json")),
    }

    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("G3", "legacy_yardstick")}, indent=2)[:4000])
    print("G2p:", res["G2p"])
    print("G3p:", {k: v for k, v in res["G3p"].items() if k != "paired_vs_proxy"})
    print("G3p paired:", res["G3p"].get("paired_vs_proxy"))
    print("SERVE:", json.dumps({k: v for k, v in res["serve_path"].items() if k != "per_cell_rho_proxy_s1i"}, indent=1))
    print("G3:", {k: v for k, v in res["G3"].items() if k != "per_cell"})
    for r in res["G3"]["per_cell"]:
        print("  ", r)
    print("YARD:", json.dumps(yard, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
