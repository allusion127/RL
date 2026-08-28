"""PREREG-2 §5 adjudication — did the high-Gd anchors clear the boron wall?

Reads the MASTER rows the mesh-v3 anchor deck bought (``campaign ==
meshv3anchor20260817``) out of the canonical store AFTER the merge, and scores
them against the three hypotheses that were registered before the spend:

    H1  the flagship P6656Z1G06N24_P6661Z2G10N24 @ f109 reaches CBC_max <= 1600
    H2  the n_gd 24 -> 22 control gap is 147 +/- 101 ppm
    H3  even if boron passes, F_r stays above Tier-1

    python anchor_readout.py

Reads data/store only; writes anchors_measured.csv + anchor_readout.log under
data/reports/mesh_v3_20260817/.  Runs no MASTER and touches no remote box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
OUT = BASE / "data" / "reports" / "mesh_v3_20260817"

#: ``produce`` stamps each row's ``campaign`` with its STRATUM name, not with the
#: deck-level campaign tag printed in the run header.  Selecting on the header
#: tag returns zero rows; select on the stratum prefix the deck gives its strata.
CAMPAIGN_PREFIX = "mv3_"
FR_LIMIT, FQ_LIMIT, CBC_LIMIT, AO_LIMIT = 1.55, 2.41, 1600.0, 0.30
FLAGSHIP = "P6656Z1G06N24_P6661Z2G10N24"
CONTROL = "P6656Z1G08N20_P6661Z2G10N24"
#: PREREG-2 §3a/§3b, frozen BEFORE the spend.  (pair, feed) -> predicted CBC_max
PRED = {(FLAGSHIP, 109): 1551, (FLAGSHIP, 117): 1641, (FLAGSHIP, 125): 1731,
        (FLAGSHIP, 101): 1461,
        ("P6253Z1G06N24_P6257Z1G06N24", 109): 1413,
        ("P6253Z1G06N24_P6257Z1G06N24", 117): 1503,
        ("P6253Z1G06N24_P6257Z1G06N24", 125): 1592,
        ("P6253Z1G06N24_P6257Z1G06N24", 101): 1323,
        ("P6253Z1G06N24_P6253Z2G10N24", 109): 1349,
        ("P6253Z1G06N24_P6253Z2G10N24", 125): 1529,
        ("P6253Z1G06N24_P6253Z2G06N24", 109): 1347,
        (CONTROL, 109): 1709, (CONTROL, 125): 1888}
#: cells 12-13 of the deck are CALIBRATION, not map cells (PREREG-2 §3b)
CALIBRATION_FEEDS = (101,)
REG_RMS = 101.0                 # ppm, the regression residual the predictions carry
H2_EFFECT, H2_TOL = 147.0, 101.0


def joint_tier(fr: float, cbc: float) -> str:
    if not (np.isfinite(fr) and np.isfinite(cbc)):
        return ""
    for name, a, b in (("tier1", 1.55, 1600.0), ("tier2", 1.65, 1800.0),
                       ("tier3", 1.80, 2200.0)):
        if fr <= a and cbc <= b:
            return name
    return "none"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=None,
                    help="read this records.parquet instead of the canonical "
                         "store (e.g. a read-only kit copy, before the merge)")
    args = ap.parse_args(argv)
    src = Path(args.records) if args.records else BASE / "data/store/records.parquet"

    lines: list[str] = []

    def log(msg="") -> None:
        print(msg, flush=True)
        lines.append(str(msg))

    log(f"source: {src}")
    s = pd.read_parquet(src)
    a = s[s.campaign.astype(str).str.startswith(CAMPAIGN_PREFIX)]
    log(f"strata {CAMPAIGN_PREFIX}*: {len(a)} rows, "
        f"{a.campaign.nunique()} strata")
    if not len(a):
        log("NOTHING MERGED YET — nothing to adjudicate.")
        return 1
    conv = a[(a.valid == True) & (a.converged == True)]          # noqa: E712
    log(f"  converged & valid: {len(conv)}  "
        f"({100 * len(conv) / len(a):.0f} %)   "
        f"strata: {a.stratum.nunique()}")

    # ------------------------------------------------- per-cell measurement -- #
    rows = []
    for (pair, feed), q in conv.groupby(["case_pair", "feed"]):
        fr_min, cbc_min = float(q.f_r.min()), float(q.cbc_max.min())
        # the core that MINIMISES boron is the one H1 is about; report its F_r
        j = q.cbc_max.idxmin()
        pred = PRED.get((pair, int(feed)), np.nan)
        rows.append(dict(
            pair=pair, feed=int(feed), role=("보정" if int(feed) in CALIBRATION_FEEDS
                                             else "지도"),
            n=int(len(q)), e_core=float(q.e_core.mean()),
            cbc_pred=pred, cbc_min=cbc_min, cbc_resid=cbc_min - pred,
            fr_at_cbc_min=float(q.loc[j, "f_r"]),
            cyclen_at_cbc_min=float(q.loc[j, "cyclen"]),
            fr_min=fr_min, fq_min=float(q.f_q.min()),
            ao_max=float(q.ao_abs.abs().max()),
            cyclen_q95=float(q.cyclen.quantile(0.95)),
            n_pass_cbc=int((q.cbc_max <= CBC_LIMIT).sum()),
            n_pass_both=int(((q.cbc_max <= CBC_LIMIT) & (q.f_r <= FR_LIMIT)).sum()),
            joint_tier=joint_tier(fr_min, cbc_min)))
    M = pd.DataFrame(rows).sort_values(["pair", "feed"])
    M.round(4).to_csv(OUT / "anchors_measured.csv", index=False, encoding="utf-8")

    log("\n--- measured vs pre-registered prediction (ppm) ---")
    log(M[["pair", "feed", "role", "n", "e_core", "cbc_pred", "cbc_min",
           "cbc_resid", "fr_at_cbc_min", "fr_min", "n_pass_cbc",
           "joint_tier"]].round(1).to_string(index=False))
    ok = M[M.cbc_pred.notna()]
    if len(ok):
        log(f"\nprediction residual: bias {ok.cbc_resid.mean():+.0f} ppm, "
            f"rms {np.sqrt((ok.cbc_resid ** 2).mean()):.0f} ppm "
            f"(registered rms {REG_RMS:.0f} ppm), n = {len(ok)} cells")
        log(f"  -> the regression is {'CONFIRMED' if np.sqrt((ok.cbc_resid ** 2).mean()) <= 2 * REG_RMS else 'REFUTED'} "
            "at the 2-sigma level of its own residual")

    # ------------------------------------------------------------------ H1 --- #
    log("\n=== H1 — does n_gd = 24 clear the boron wall at high enrichment? ===")
    f = M[(M.pair == FLAGSHIP) & (M.feed == 109)]
    if not len(f):
        log("  flagship @ f109 has no converged rows — UNDECIDED.")
    else:
        r = f.iloc[0]
        v = ("확증 (confirmed)" if r.cbc_min <= 1600 else
             "경계 (marginal, 회귀 잔차 내)" if r.cbc_min <= 1700 else "기각 (refuted)")
        log(f"  flagship {FLAGSHIP} @ f109 (e_core {r.e_core:.3f}, n = {int(r.n)})")
        log(f"    predicted min CBC_max  {r.cbc_pred:.0f} ppm")
        log(f"    MEASURED  min CBC_max  {r.cbc_min:.1f} ppm  "
            f"({r.cbc_resid:+.0f} ppm vs prediction)")
        log(f"    limit 1600 ppm -> {int(r.n_pass_cbc)}/{int(r.n)} cores under the limit")
        log(f"    VERDICT: {v}")

    # ------------------------------------------------------------------ H2 --- #
    log("\n=== H2 — is the Gd-pin effect 147 +/- 101 ppm (2 pins, n_gd 24 vs 22)? ===")
    got = False
    for feed in sorted(set(M[M.pair == CONTROL].feed) & set(M[M.pair == FLAGSHIP].feed)):
        x = M[(M.pair == FLAGSHIP) & (M.feed == feed)].iloc[0]
        y = M[(M.pair == CONTROL) & (M.feed == feed)].iloc[0]
        d = y.cbc_min - x.cbc_min
        got = True
        log(f"  f{feed}: control(n_gd 22) {y.cbc_min:.1f} - flagship(n_gd 24) "
            f"{x.cbc_min:.1f} = {d:+.1f} ppm  "
            f"[{H2_EFFECT - H2_TOL:.0f}…{H2_EFFECT + H2_TOL:.0f} 기대] -> "
            f"{'확인' if abs(d - H2_EFFECT) <= H2_TOL else '회귀 갱신 필요'}")
        log(f"        (e_core {x.e_core:.3f} vs {y.e_core:.3f}; "
            f"per-pin {d / 2:+.1f} ppm, registered -73.5)")
    if not got:
        log("  no feed has both flagship and control rows — UNDECIDED.")

    # ------------------------------------------------------------------ H3 --- #
    log("\n=== H3 — does F_r survive after boron passes? ===")
    p = conv[conv.cbc_max <= CBC_LIMIT]
    log(f"  {len(p)}/{len(conv)} converged anchor cores are under CBC 1600 ppm")
    if len(p):
        log(f"    their F_r: min {p.f_r.min():.4f}, median {p.f_r.median():.4f}")
        log(f"    their F_q: min {p.f_q.min():.4f} (limit {FQ_LIMIT})")
        log(f"    of those, F_r <= 1.55 (Tier-1): {int((p.f_r <= FR_LIMIT).sum())} cores")
    for nm, a_, b_ in (("Tier-1", 1.55, 1600.0), ("Tier-2", 1.65, 1800.0),
                       ("Tier-3", 1.80, 2200.0)):
        n = int(((conv.f_r <= a_) & (conv.cbc_max <= b_)
                 & (conv.f_q <= FQ_LIMIT)
                 & (conv.ao_abs.abs() <= AO_LIMIT)).sum())
        log(f"    JOINT {nm} (F_r<={a_} ∧ CBC<={b_:.0f}): {n} cores")
    log(f"  overall anchor floors: F_r {conv.f_r.min():.4f} · "
        f"CBC {conv.cbc_max.min():.1f} ppm · F_q {conv.f_q.min():.4f} · "
        f"|AO| max {conv.ao_abs.abs().max():.4f}")

    # ------------------------------------------- refit the boron regression -- #
    log("\n--- boron regression refit WITH the new anchors ---")
    try:
        from lpopt.data.fuel_types import FuelLibrary          # noqa: F401
        fu = pd.read_parquet(BASE / "data/store/fuel_types.parquet")
        fp = fu[fu.library_id == "paramA"].set_index("type_id")

        def ngd(pair: str) -> float:
            a_, b_ = pair.split("_", 1)
            try:
                return float((fp.n_gd[a_] + fp.n_gd[b_]) / 2)
            except Exception:                                   # noqa: BLE001
                return np.nan

        hp = s[(s.valid == True) & (s.converged == True)          # noqa: E712
               & (s.e_core >= 5.6) & (s.library_id == "paramA")]
        g = hp.groupby(["case_pair", "feed"]).agg(
            cbc_min=("cbc_max", "min"), e_core=("e_core", "mean")).reset_index()
        g["n_gd"] = [ngd(p_) for p_ in g.case_pair]
        g = g[g.n_gd.notna()]
        X = np.c_[np.ones(len(g)), g.n_gd, g.e_core, g.feed]
        coef, *_ = np.linalg.lstsq(X, g.cbc_min.to_numpy(), rcond=None)
        res = g.cbc_min.to_numpy() - X @ coef
        log(f"  CBC_min ~ {coef[0]:+.0f} {coef[1]:+.1f}*n_gd {coef[2]:+.0f}*e_core "
            f"{coef[3]:+.1f}*feed")
        log(f"  n = {len(g)} cells, R^2 = {1 - res.var() / g.cbc_min.var():.3f}, "
            f"rms = {np.sqrt((res ** 2).mean()):.0f} ppm")
        log("  (PREREG-2 held -73.5 ppm/pin, +454 ppm/wt%, +11.2 ppm/bundle, "
            "rms 101 ppm on 72 cells)")
    except Exception as exc:                                    # noqa: BLE001
        log(f"  refit skipped: {exc}")

    (OUT / "anchor_readout.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwrote {OUT/'anchors_measured.csv'}, {OUT/'anchor_readout.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
