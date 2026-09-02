"""PREREG-2 §5 adjudication — did the high-Gd anchors clear the boron wall?

Reads the MASTER rows the mesh-v3 anchor deck bought (``campaign ==
meshv3anchor20260817``) out of the canonical store AFTER the merge, and scores
them against the three hypotheses that were registered before the spend:

    H1  the flagship P6656Z1G06N24_P6661Z2G10N24 @ f109 reaches CBC_max <= 1600
    H2  the n_gd 24 -> 22 control gap is 147 +/- 101 ppm
    H3  even if boron passes, F_r stays above Tier-1

    python anchor_readout.py
    python anchor_readout.py --deck fpcamp_minfxy_...inp    # read H3 on F_xy

Reads data/store only; writes anchors_measured.csv + anchor_readout.log under
data/reports/mesh_v3_20260817/.  Runs no MASTER and touches no remote box.

THE OBJECTIVE AXIS (design ``data/reports/fxy_switch_design_20260829.md`` §3.5.5).
H1/H2 are boron hypotheses and are axis-free.  H3 — "does the peaking axis survive
after boron passes?" — is the FRONTIER readout, and it is now read on whichever
axis the deck named: F_r (default, unchanged) or, under ``objective = "min_fxy"``,
the store's MEASURED ``f_xy`` against ``f_xy_limit``.  Rows with no measured F_xy
are EXCLUDED and counted rather than passed through, because a "frontier" ranked
over the 8 % of the population that happened to be harvested is not a frontier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import readout_axis as RA                                     # noqa: E402

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


#: The joint (peaking, boron) tier ladder, as F_r caps.  Under the F_xy axis each
#: cap is shifted by the axis's own limit — tier1's peaking cap IS the licensing
#: limit of whichever axis is being read, and the ladder above it keeps the same
#: relative headroom (+0.10 / +0.25) it has always had on F_r.  Inventing an
#: independent F_xy ladder would be a licensing claim this readout has no basis
#: for; scaling the one registered ladder is arithmetic and is labelled as such.
_TIERS_FR = (("tier1", 1.55, 1600.0), ("tier2", 1.65, 1800.0),
             ("tier3", 1.80, 2200.0))


def tiers_for(axis: RA.Axis) -> tuple[tuple[str, float, float], ...]:
    shift = axis.limit - 1.55
    return tuple((name, round(a + shift, 4), b) for name, a, b in _TIERS_FR)


def joint_tier(fr: float, cbc: float,
               tiers: tuple[tuple[str, float, float], ...] = _TIERS_FR) -> str:
    if not (np.isfinite(fr) and np.isfinite(cbc)):
        return ""
    for name, a, b in tiers:
        if fr <= a and cbc <= b:
            return name
    return "none"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=None,
                    help="read this records.parquet instead of the canonical "
                         "store (e.g. a read-only kit copy, before the merge)")
    RA.add_axis_args(ap)
    ap.add_argument("--lambda-check", dest="lambda_check", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="print the mandatory 'cyclen - lambda*axis' frontier "
                         "reading (default: on for the F_xy axis, off for F_r "
                         "so existing min_fr logs stay line-for-line comparable)")
    args = ap.parse_args(argv)
    src = Path(args.records) if args.records else BASE / "data/store/records.parquet"
    axis = RA.axis_from_args(args)
    tiers = tiers_for(axis)

    lines: list[str] = []

    def log(msg="") -> None:
        print(msg, flush=True)
        lines.append(str(msg))

    log(f"source: {src}")
    if axis.is_fxy:
        log(f"axis:   {axis.provenance()}")
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
        # The peaking side of every joint count below is the OBJECTIVE axis.  On
        # F_r these are the same numbers under the same names; on F_xy the cell's
        # unlabelled rows drop out of the axis floor and the joint counts, and
        # `axis_unlabelled` says how many did.
        qa, n_unlab = RA.split_labelled(q, axis)
        av_all = RA.axis_values(q, axis)
        axis_min = float(RA.axis_values(qa, axis).min()) if len(qa) else np.nan
        row = dict(
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
            n_pass_both=int(((q.cbc_max <= CBC_LIMIT)
                             & (av_all <= axis.limit)).sum()),
            joint_tier=joint_tier(axis_min, cbc_min, tiers))
        if axis.is_fxy:
            # Added ONLY off the default axis, so anchors_measured.csv keeps its
            # exact schema for every min_fr readout that already consumes it.
            row.update(axis=axis.label, axis_min=axis_min,
                       axis_at_cbc_min=float(av_all.loc[j]),
                       axis_labelled=int(len(qa)), axis_unlabelled=int(n_unlab))
        rows.append(row)
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
    log(f"\n=== H3 — does {axis.label} survive after boron passes? ===")
    p = conv[conv.cbc_max <= CBC_LIMIT]
    log(f"  {len(p)}/{len(conv)} converged anchor cores are under CBC 1600 ppm")
    p_lab, p_unlab = RA.split_labelled(p, axis)
    note = RA.unlabelled_note(axis, p_unlab, len(p), what="cores")
    if len(p):
        pav = RA.axis_values(p_lab, axis)
        if note:
            log(note)
        if len(p_lab):
            log(f"    their {axis.label}: min {pav.min():.4f}, "
                f"median {pav.median():.4f}")
        log(f"    their F_q: min {p.f_q.min():.4f} (limit {FQ_LIMIT})")
        log(f"    of those, {axis.gate} (Tier-1): "
            f"{int((pav <= axis.limit).sum())} cores")
    conv_lab, conv_unlab = RA.split_labelled(conv, axis)
    cav = RA.axis_values(conv_lab, axis)
    for nm, a_, b_ in (("Tier-1", *tiers[0][1:]), ("Tier-2", *tiers[1][1:]),
                       ("Tier-3", *tiers[2][1:])):
        n = int(((cav <= a_) & (conv_lab.cbc_max <= b_)
                 & (conv_lab.f_q <= FQ_LIMIT)
                 & (conv_lab.ao_abs.abs() <= AO_LIMIT)).sum())
        log(f"    JOINT {nm} ({axis.label}<={a_} ∧ CBC<={b_:.0f}): {n} cores")
    log(f"  overall anchor floors: {axis.label} "
        f"{cav.min() if len(conv_lab) else float('nan'):.4f} · "
        f"CBC {conv.cbc_max.min():.1f} ppm · F_q {conv.f_q.min():.4f} · "
        f"|AO| max {conv.ao_abs.abs().max():.4f}")

    # ------------------------------------------- the mandatory λ reading ----- #
    # Registered rule: a frontier is read on ``cyclen − λ·axis``, never on
    # ``min(axis)`` alone — the axis-only headline was overturned twice.  Design
    # §3.5.5 moves the AXIS of that rule to F_xy and leaves the rule itself in
    # force.  Off by default on F_r so this file's existing logs stay comparable
    # line for line; ``--lambda-check`` turns it on for either axis.
    want_lambda = axis.is_fxy if args.lambda_check is None else args.lambda_check
    if want_lambda:
        log(f"\n=== λ-OBJECTIVE reading (mandatory on every frontier readout) ===")
        log(f"  scalar = cyclen − λ·{axis.label},  λ = {axis.lam:g} EFPD per unit "
            f"{axis.label}")
        best, n_unlab = RA.best_by_lambda(conv, axis)
        if best is None:
            log(f"  NO core carries a measured {axis.label} "
                f"({n_unlab}/{len(conv)} unlabelled) — the λ objective is "
                f"UNDEFINED here and no frontier claim is made.")
        else:
            floor_row = conv_lab.loc[cav.idxmin()] if len(conv_lab) else None
            log(RA.unlabelled_note(axis, n_unlab, len(conv), what="cores")
                or f"    unlabelled: 0/{len(conv)} cores")
            log(f"  λ-BEST      {str(best.record_id)[:12]}  {axis.label} "
                f"{float(RA.axis_values(conv_lab, axis).loc[best.name]):.4f}  "
                f"cyclen {float(best.cyclen):.1f} EFPD  CBC {float(best.cbc_max):.1f}")
            if floor_row is not None:
                same = "SAME core" if floor_row.name == best.name else "DIFFERENT core"
                log(f"  {axis.label}-FLOOR  {str(floor_row.record_id)[:12]}  "
                    f"{axis.label} {float(cav.min()):.4f}  "
                    f"cyclen {float(floor_row.cyclen):.1f} EFPD  "
                    f"CBC {float(floor_row.cbc_max):.1f}   -> {same}")

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
