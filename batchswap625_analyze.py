"""Registered readouts for the batch_swap 625 EFPD-branch wave.

Pre-registration: ``data/reports/batchswap625_wave_prereg_20260815.md`` §4.

§4a  best IN-BAND feasible F_r vs 1.4747 / 1.4749 / 1.4636
§4b  per-parent improving fraction, DEEP-N ONLY (n >= 40), honouring the n=4
     retraction in batchswap_wave_results §3
§4c  lambda-objective comparison vs the r8 record - the number that decides
     whether the result is usable

THE OBJECTIVE AXIS (design ``data/reports/fxy_switch_design_20260829.md`` §3.5.5).
§4a and §4c are the frontier and the lambda readings, and they are now computed on
whichever axis the deck named: F_r (default, unchanged, with this wave's own
registered lambda 400) or MEASURED F_xy under ``objective = "min_fxy"``.  The
three reference marks (1.4747 / 1.4749 / 1.4636) are F_r RECORDS; there is no
F_xy record to compare against, so on the F_xy axis §4c reports the axis and the
lambda arithmetic and REFUSES the comparison rather than subtracting an F_xy from
an F_r (design §3.6: no citing an F_xy reference that does not exist).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import readout_axis as RA                                     # noqa: E402

CAMPAIGN = "batchswap_enum_625_T6T4"
BAND_LO, BAND_HI = 620.0, 645.0
IN_BAND_BEST = 1.4747
R8_RECORD, R8_CYCLEN = 1.4749, 625.459
GA80_INCUMBENT, GA80_CYCLEN = 1.4636, 633.329
LAMBDA = 400.0
DEEP_N = 40


def main(argv=None) -> int:
    import mine_policy_corpus as M
    import ablation_analyze as A

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reports/batchswap625_wave_tables.txt")
    RA.add_axis_args(ap)
    args = ap.parse_args(argv)

    # This wave registered lambda = 400 (batchswap625_wave.py), not the 1000 the
    # min_fr decks carry, so it is pinned here rather than taken from the shared
    # default; on the F_xy axis the deck's own minfxy_lambda applies.
    axis = RA.resolve_axis(objective=args.objective, deck=args.deck) \
        if (args.objective or args.deck) else \
        RA.resolve_axis(limits={"minfr_lambda": LAMBDA})

    out: list[str] = []

    def add(s: str = "") -> None:
        out.append(s)
        print(s)

    if axis.is_fxy:
        add(f"axis: {axis.provenance()}")
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    w = store[store["campaign"] == CAMPAIGN].copy()
    w["feasible"] = M.feasibility(w)
    if axis.is_fxy:
        # ADD the measured F_xy gate to the tri-state program feasibility; F_r
        # stays in it (design §3.5.2).  An unmeasured F_xy stays <NA>, not True:
        # this is the delivery-side reading (§3.5.4), where UNKNOWN is not a pass.
        av = RA.axis_values(w, axis)
        ok = (av <= axis.limit).astype("boolean")
        ok[av.isna()] = pd.NA
        w["feasible"] = w["feasible"] & ok
    conv = w[w["converged"].astype(bool)]
    feas = conv[conv["feasible"].fillna(False).astype(bool)]
    inband = feas[(feas["cyclen"] >= BAND_LO) & (feas["cyclen"] <= BAND_HI)]
    add(f"chains {len(w)}  converged {len(conv)}  feasible {len(feas)}  "
        f"in-band feasible {len(inband)}")
    add(f"failures: {conv.shape[0] and w[~w['converged'].astype(bool)]['failure'].value_counts().to_dict()}")
    add(f"cyclen range of children: {conv['cyclen'].min():.2f} - {conv['cyclen'].max():.2f}"
        f"   (out-of-band children: {int((~conv['cyclen'].between(BAND_LO, BAND_HI)).sum())})")
    add()

    steps = A.build_wave_steps(CAMPAIGN, None)
    smap = steps.set_index("child_record_id")

    # ---- §4a --------------------------------------------------------------- #
    add(f"## §4a  PRIMARY - best IN-BAND feasible {axis.label}")
    add()
    inband, _ = RA.split_labelled(inband, axis)
    # Counted over the CONVERGED population: on the F_xy axis the unmeasured rows
    # are dropped by the tri-state gate above, so this is the only place a reader
    # learns how much of the wave the frontier could not see.
    note = RA.unlabelled_note(axis, RA.split_labelled(conv, axis)[1], len(conv),
                              what="converged boards")
    if note:
        add(note)
    cols = ["record_id", "parent_record_id", "f_r", "f_q", "cbc_max", "ao_abs",
            "cyclen", "node_peak"]
    if axis.is_fxy:
        cols.insert(2, axis.key)
    top = inband.nsmallest(10, axis.key)[cols].copy()
    top["parent_f_r"] = [float(smap.loc[r, "parent_f_r"]) if r in smap.index else np.nan
                         for r in top["record_id"]]
    top["d_f_r"] = top["f_r"] - top["parent_f_r"]
    add(top.to_string(index=False, max_colwidth=14,
                      float_format=lambda v: f"{v:.4f}"))
    add()
    if len(inband):
        b = inband.nsmallest(1, axis.key).iloc[0]
        b_axis = float(b[axis.key])
        add(f"BEST IN-BAND: {b.record_id[:12]}  {axis.label} {b_axis:.4f} "
            f"@ {b.cyclen:.3f} EFPD")
        add(f"  F_q {b.f_q:.4f}  CBC {b.cbc_max:.2f}  |AO| {b.ao_abs:.4f}  "
            f"node_peak {b.node_peak:.4f}  parent {b.parent_record_id[:12]}")
        # The three marks are F_r RECORDS.  On the F_xy axis they are not
        # comparable to the headline above, and the readout says so instead of
        # differencing two different physical quantities.
        if axis.is_fxy:
            add(f"  vs the registered marks 1.4747 / 1.4749 / 1.4636: NOT COMPARED — "
                f"those are F_r records and no {axis.label} record exists for this "
                f"cell (design 20260829 sec. 3.6).  This board's F_r is "
                f"{float(b.f_r):.4f}, carried as the constraint reading only.")
        else:
            for name, mark in (("current in-band best", IN_BAND_BEST),
                               ("r8 campaign record", R8_RECORD),
                               ("ga80 incumbent (stretch)", GA80_INCUMBENT)):
                d = b_axis - mark
                verdict = "BEATS" if d < 0 else "does not beat"
                add(f"  vs {name:<26} {mark:.4f}: {d:+.4f}   {verdict}")
        add()
        # ---- §4c ---------------------------------------------------------- #
        add("## §4c  lambda-objective vs the r8 record (the usability test)")
        add()
        if axis.is_fxy:
            add(f"  scalar = cyclen - lambda*{axis.label}, lambda = {axis.lam:g} "
                f"EFPD per unit {axis.label}; this board scores "
                f"{b.cyclen - axis.lam * b_axis:+.2f} EFPD-eq.")
            add(f"  NO REFERENCE: the r8 / ga80 marks are F_r-and-cyclen pairs, so "
                f"the NET column is undefined on this axis until a measured "
                f"{axis.label} reference core exists.")
        else:
            for name, mark, cyc in (("r8 record", R8_RECORD, R8_CYCLEN),
                                    ("ga80 incumbent", GA80_INCUMBENT, GA80_CYCLEN)):
                dfr, dcy = b_axis - mark, b.cyclen - cyc
                net = -dfr * axis.lam + dcy
                add(f"  vs {name:<16} d_{axis.label} {dfr:+.4f} "
                    f"(worth {-dfr * axis.lam:+.2f} EFPD-eq)"
                    f"  d_cyclen {dcy:+.2f}  ->  NET {net:+.2f} EFPD-eq  "
                    f"{'NEW BOARD WINS' if net > 0 else 'reference preferred'}")
        add()
        if not axis.is_fxy:
            add(f"  boards in-band beating 1.4747: "
                f"{int((inband[axis.key] < IN_BAND_BEST).sum())}"
                f"   beating 1.4636: {int((inband[axis.key] < GA80_INCUMBENT).sum())}")
        else:
            add(f"  boards in-band under the {axis.gate} licensing limit: "
                f"{int((inband[axis.key] <= axis.limit).sum())}/{len(inband)}")
    add()

    # ---- §4b --------------------------------------------------------------- #
    add("## §4b  SECONDARY - per-parent improving fraction (DEEP-N ONLY, n>=40)")
    add()
    s = steps[steps["both_converged"].fillna(False).astype(bool)]
    g = s.groupby("parent_record_id")["d_f_r"]
    tbl = pd.concat([
        g.size().rename("n"), g.min().rename("min"), g.quantile(.25).rename("p25"),
        g.median().rename("p50"), g.mean().rename("mean"), g.max().rename("max"),
        g.apply(lambda x: (x < 0).mean()).rename("frac<0"),
    ], axis=1)
    tbl["parent_f_r"] = [float(s[s.parent_record_id == i]["parent_f_r"].iloc[0])
                         for i in tbl.index]
    tbl.index = [i[:12] for i in tbl.index]
    deep = tbl[tbl["n"] >= DEEP_N].sort_values("parent_f_r")
    shallow = tbl[tbl["n"] < DEEP_N].sort_values("parent_f_r")
    add("QUOTED (n >= 40):")
    add(deep.to_string(float_format=lambda v: f"{v:.4f}"))
    add()
    add(f"NOT QUOTED as rates (n < {DEEP_N}) - carried in the corpus only, per the "
        f"n=4 retraction:")
    add(shallow.to_string(float_format=lambda v: f"{v:.4f}"))
    add()
    imp = s["improved_fr"].astype("boolean")
    rate, n = float(imp.mean()), int(imp.notna().sum())
    se = float(np.sqrt(rate * (1 - rate) / n)) if n else np.nan
    add(f"pooled improving_fr = {rate:.3f} +- {1.96 * se:.3f} (n={n})")
    add(f"  618-branch wave measured 0.052; registered expectation was "
        f"'at or below 0.052' (better parents)")
    add()
    d = s.groupby("fresh_radial_dir")["improved_fr"].agg(
        n="size", improving=lambda x: x.astype("boolean").mean())
    add("by direction:")
    add(d.to_string(float_format=lambda v: f"{v:.3f}"))
    add()

    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n[analyze] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
