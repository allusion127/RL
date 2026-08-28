"""Registered readouts for the batch_swap 625 EFPD-branch wave.

Pre-registration: ``data/reports/batchswap625_wave_prereg_20260815.md`` §4.

§4a  best IN-BAND feasible F_r vs 1.4747 / 1.4749 / 1.4636
§4b  per-parent improving fraction, DEEP-N ONLY (n >= 40), honouring the n=4
     retraction in batchswap_wave_results §3
§4c  lambda-objective comparison vs the r8 record - the number that decides
     whether the result is usable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

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
    args = ap.parse_args(argv)

    out: list[str] = []

    def add(s: str = "") -> None:
        out.append(s)
        print(s)

    store = pd.read_parquet(BASE / "data/store/records.parquet")
    w = store[store["campaign"] == CAMPAIGN].copy()
    w["feasible"] = M.feasibility(w)
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
    add("## §4a  PRIMARY - best IN-BAND feasible F_r")
    add()
    top = inband.nsmallest(10, "f_r")[
        ["record_id", "parent_record_id", "f_r", "f_q", "cbc_max", "ao_abs",
         "cyclen", "node_peak"]].copy()
    top["parent_f_r"] = [float(smap.loc[r, "parent_f_r"]) if r in smap.index else np.nan
                         for r in top["record_id"]]
    top["d_f_r"] = top["f_r"] - top["parent_f_r"]
    add(top.to_string(index=False, max_colwidth=14,
                      float_format=lambda v: f"{v:.4f}"))
    add()
    if len(inband):
        b = inband.nsmallest(1, "f_r").iloc[0]
        add(f"BEST IN-BAND: {b.record_id[:12]}  F_r {b.f_r:.4f} @ {b.cyclen:.3f} EFPD")
        add(f"  F_q {b.f_q:.4f}  CBC {b.cbc_max:.2f}  |AO| {b.ao_abs:.4f}  "
            f"node_peak {b.node_peak:.4f}  parent {b.parent_record_id[:12]}")
        for name, mark in (("current in-band best", IN_BAND_BEST),
                           ("r8 campaign record", R8_RECORD),
                           ("ga80 incumbent (stretch)", GA80_INCUMBENT)):
            d = b.f_r - mark
            verdict = "BEATS" if d < 0 else "does not beat"
            add(f"  vs {name:<26} {mark:.4f}: {d:+.4f}   {verdict}")
        add()
        # ---- §4c ---------------------------------------------------------- #
        add("## §4c  lambda-objective vs the r8 record (the usability test)")
        add()
        for name, mark, cyc in (("r8 record", R8_RECORD, R8_CYCLEN),
                                ("ga80 incumbent", GA80_INCUMBENT, GA80_CYCLEN)):
            dfr, dcy = b.f_r - mark, b.cyclen - cyc
            net = -dfr * LAMBDA + dcy
            add(f"  vs {name:<16} d_F_r {dfr:+.4f} (worth {-dfr * LAMBDA:+.2f} EFPD-eq)"
                f"  d_cyclen {dcy:+.2f}  ->  NET {net:+.2f} EFPD-eq  "
                f"{'NEW BOARD WINS' if net > 0 else 'reference preferred'}")
        add()
        add(f"  boards in-band beating 1.4747: {int((inband['f_r'] < IN_BAND_BEST).sum())}"
            f"   beating 1.4636: {int((inband['f_r'] < GA80_INCUMBENT).sum())}")
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
