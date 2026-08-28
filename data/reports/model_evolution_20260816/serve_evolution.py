"""Serve every champion generation on the SAME frozen 36-cell evaluation surface.

Fixed yardstick: `groups.ab2_frozen_val_by_cell` from data/splits/S1b..S1f.json --
3,207 rows, 36 cells, byte-identical (sha e516cadca1839e65) across all five
curriculum splits, and carved out of S1's *val* fold so the two pre-S1b
generations (both `--split S1`) never trained on it either.

Serving reuses the validated harness of data/models/scaling_20260815/dump_scaling.py:
the SERVE path (PosValCnnBackend.predict / predict_map_flatness) is used throughout,
because
  * the cyclen physics-prior add-back lives there (_ensemble_raw -> predict), and
  * predict_map_flatness is the only definition of predicted node_peak / map_cov.
Each generation is served AS PROMOTED, i.e. with its own calibration files.

Usage:  python serve_evolution.py
Writes: raw_<gen>.npz (per generation), evolution_metrics.csv, evolution_metrics.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent
STORE, SPLITS = str(REPO / "data" / "store"), REPO / "data" / "splits"
MODELS = REPO / "data" / "models"
# Local GPU is SM < 7.5 and this torch build's cuDNN (91900) refuses it, so the
# whole lineage is served on CPU.  3,207 rows x 5 members x 7 generations is well
# inside CPU budget and the numerics are identical.
DEV = "cpu"

from lpopt.data.store import StoreReader                       # noqa: E402
from lpopt.data.schema import unpack_pattern                   # noqa: E402
from lpopt.model.folds import cell_key                         # noqa: E402
from lpopt.model.model_api import PosValCnnBackend             # noqa: E402
from lpopt.vendor.masterrl.domain import CaseKey               # noqa: E402

# champion lineage, in promotion order
GENS = [
    ("20260729_054749", "S1",  "2026-07-29"),
    ("20260810_bu_T",   "S1",  "2026-08-10"),
    ("split_S1b",       "S1b", "2026-08-10"),
    ("s1c",             "S1c", "2026-08-12"),
    ("s1d",             "S1d", "2026-08-14"),
    ("s1e",             "S1e", "2026-08-15"),
    ("s1f",             "S1f", "2026-08-16"),
    ("s1g",             "S1g", "2026-08-16"),
]
TARGETS = ("f_r", "node_peak", "map_cov", "cyclen", "cbc_max", "f_q")
# for P@10%: which end of the ranking the search actually wants
LOWER_BETTER = {"f_r": True, "node_peak": True, "map_cov": True,
                "cyclen": False, "cbc_max": True, "f_q": True}
MIN_GROUP_ROWS = 8            # inherited from split_secondary_readout / scaling prereg
P_AT_FRAC = 0.10


# ------------------------------------------------------------ metric helpers
# (verbatim from data/models/scaling_20260815/score_scaling.py)
def _rank(a):
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    return r


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


def _p_at_frac(pred, true, lower_better=True):
    n = len(pred)
    k = max(4, int(round(P_AT_FRAC * n)))
    if k >= n:
        return np.nan
    s = 1.0 if lower_better else -1.0
    tp = set(np.argsort(s * pred, kind="mergesort")[:k].tolist())
    tt = set(np.argsort(s * true, kind="mergesort")[:k].tolist())
    return len(tp & tt) / k


def per_group(fn, pred, true, groups, min_rows=MIN_GROUP_ROWS):
    out = {}
    ok = np.isfinite(pred) & np.isfinite(true)
    for c in np.unique(groups[ok]):
        m = ok & (groups == c)
        if m.sum() < min_rows:
            continue
        v = fn(pred[m], true[m])
        if np.isfinite(v):
            out[str(c)] = float(v)
    return out


# ------------------------------------------------------------ the eval surface
def load_surface():
    man = json.loads((SPLITS / "S1f.json").read_text(encoding="utf-8"))
    fz = man["groups"]["ab2_frozen_val_by_cell"]
    ids = set()
    for v in fz.values():
        ids |= set(map(str, v))
    d = StoreReader(STORE).records
    d = d[d["record_id"].astype(str).isin(ids)]
    d = d[d["converged"] == True].reset_index(drop=True)      # noqa: E712
    # stable row order so every generation is scored on identical rows/order
    d = d.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    cells = cell_key(d).astype(str).to_numpy()
    # iso group = campaign(cell) x case_pair x feed  (mission spec).  The frozen
    # cells are legacy `ebin_feed` cells and 34/36 of them MIX case_pairs, so the
    # refinement is load-bearing: a pair-mixed rho is partly between-pair scale.
    iso = np.array([f"{c}|{p}|f{f}" for c, p, f in
                    zip(cells, d["case_pair"].astype(str), d["feed"])])
    assert len(ids) == 3207 and len(d) == 3207, (len(ids), len(d))
    assert len(set(cells)) == 36, len(set(cells))
    return d, cells, iso


def contamination_flags():
    """Verify the frozen rows were excluded from EVERY generation's train fold."""
    man = json.loads((SPLITS / "S1b.json").read_text(encoding="utf-8"))
    ids = set()
    for v in man["groups"]["ab2_frozen_val_by_cell"].values():
        ids |= set(map(str, v))
    flags = {}
    for _gen, split, _dt in GENS:
        s = json.loads((SPLITS / f"{split}.json").read_text(encoding="utf-8"))
        tr = set(map(str, s["train_ids"]))
        va = set(map(str, s["val_ids"]))
        flags[split] = {"n_frozen_in_train": len(ids & tr),
                        "n_frozen_in_val": len(ids & va),
                        "clean": len(ids & tr) == 0}
    return flags


# ------------------------------------------------------------------- serving
def serve(run_dir, d, pats, cases):
    b = PosValCnnBackend.from_dir(str(run_dir), store_dir=STORE,
                                  library_id="ga80", device=DEV)
    sp = b.predict(pats, cases)          # 7-col surrogate, calibrated, prior added back
    pk_m, pk_s, cv_m, cv_s = b.predict_map_flatness(pats, cases)
    # surrogate cols: 0 F_r, 1 CBC_max, 2 F_q, 3 cyclen, 4 AO_abs, 5 asm_bu, 6 pin_bu
    mean = np.asarray(sp.mean, float)
    pred = {"f_r": mean[:, 0], "cbc_max": mean[:, 1], "f_q": mean[:, 2],
            "cyclen": mean[:, 3],
            "node_peak": np.asarray(pk_m, float),
            "map_cov": np.asarray(cv_m, float)}
    meta = {"n_members": len(b.metas), "cond_schema": str(b.cond_schema),
            "n_params": int(b.metas[0].get("n_params", 0)),
            "best_epoch": int(b.metas[0].get("best_epoch", -1)),
            "target_names": list(b.target_names),
            "has_cyclen_prior": bool(b._cyclen_prior is not None),
            "cyclen_target_idx": (-1 if b._cyclen_target_idx is None
                                  else int(b._cyclen_target_idx))}
    sig = {"f_r": np.asarray(sp.calibrated_std, float)[:, 0],
           "node_peak": np.asarray(pk_s, float),
           "map_cov": np.asarray(cv_s, float)}
    del b
    return pred, sig, meta


def main():
    t0 = time.time()
    d, cells, iso = load_surface()
    flags = contamination_flags()
    print("frozen surface: %d rows / %d cells / %d iso groups"
          % (len(d), len(set(cells)), len(set(iso))), flush=True)
    print("contamination:", json.dumps(flags), flush=True)

    pats = [unpack_pattern(p) for p in d["pattern"].tolist()]
    cases = [CaseKey(str(cp), int(f)) for cp, f in zip(d["case_pair"], d["feed"])]
    truth = {t: pd.to_numeric(d[t], errors="coerce").to_numpy(float)
             for t in TARGETS if t in d.columns}

    rows, detail = [], {}
    for gen, split, date in GENS:
        rd = MODELS / gen
        if not (rd / "ensemble.json").exists():
            print(f"SKIP {gen}: no ensemble.json", flush=True)
            continue
        try:
            pred, sig, meta = serve(rd, d, pats, cases)
        except Exception as exc:                                  # noqa: BLE001
            print(f"SKIP {gen}: serve failed -> {exc!r}", flush=True)
            continue
        r = {"gen": gen, "split": split, "date": date,
             "n_rows": len(d), "n_iso_groups": len(set(iso)), **meta}
        det = {}
        for t in TARGETS:
            if t not in truth:
                continue
            p, y = pred[t], truth[t]
            g_iso = per_group(spearman, p, y, iso)
            g_cell = per_group(spearman, p, y, cells)
            det[t] = {"iso": g_iso, "cell": g_cell}
            r[f"rho_iso_{t}"] = float(np.median(list(g_iso.values()))) if g_iso else np.nan
            r[f"rho_iso_{t}_n"] = len(g_iso)
            r[f"rho_cell_{t}"] = float(np.median(list(g_cell.values()))) if g_cell else np.nan
            ok = np.isfinite(p) & np.isfinite(y)
            r[f"mae_{t}"] = float(np.mean(np.abs(p[ok] - y[ok]))) if ok.sum() else np.nan
            r[f"n_{t}"] = int(ok.sum())
            if t in ("node_peak", "map_cov"):
                gp = per_group(lambda a, b, _l=LOWER_BETTER[t]: _p_at_frac(a, b, _l),
                               p, y, iso)
                r[f"p10_{t}"] = float(np.median(list(gp.values()))) if gp else np.nan
                det[t]["p10_iso"] = gp
        rows.append(r)
        detail[gen] = det
        np.savez_compressed(OUT / f"raw_{gen}.npz",
                            record_ids=d["record_id"].astype(str).to_numpy(),
                            cells=cells, iso=iso,
                            **{f"pred_{k}": v for k, v in pred.items()},
                            **{f"sigma_{k}": v for k, v in sig.items()},
                            **{f"true_{k}": v for k, v in truth.items()})
        print("  %-18s rho_iso f_r=%.4f  cyclen=%.4f  node_peak=%.4f  (%.0fs)"
              % (gen, r.get("rho_iso_f_r", np.nan), r.get("rho_iso_cyclen", np.nan),
                 r.get("rho_iso_node_peak", np.nan), time.time() - t0), flush=True)

    if not rows:
        raise SystemExit("no generation served successfully")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "evolution_metrics.csv", index=False)
    (OUT / "evolution_metrics.json").write_text(json.dumps(
        {"arms": rows, "contamination": flags, "per_group": detail,
         "config": {"MIN_GROUP_ROWS": MIN_GROUP_ROWS, "P_AT_FRAC": P_AT_FRAC,
                    "surface": "ab2_frozen_val_by_cell", "n_rows": len(d)}},
        indent=1, default=float), encoding="utf-8")
    cols = [c for c in df.columns if c.startswith("rho_iso_") and not c.endswith("_n")]
    print(df[["gen", "split", *cols]].to_string(index=False))
    print("\nwrote %s" % (OUT / "evolution_metrics.csv"))


if __name__ == "__main__":
    main()
