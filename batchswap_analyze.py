"""Registered readouts for the batch_swap deep-sample wave.

Pre-registration: ``data/reports/batchswap_wave_prereg_20260815.md`` §4.

§4a  best feasible F_r found, vs the parent, vs the cell record 1.4685, vs the
     ga80 incumbent 1.4636.
§4b  per-parent improvement distribution, and the audit of the ablation wave's
     DIRECTION-BALANCED estimate against this wave's PROPORTIONAL one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

CAMPAIGN = "batchswap_enum_T6T4"
CELL_RECORD_PRIOR = 1.4685
GA80_INCUMBENT = 1.4636
#: Registered in prereg §4b, BEFORE the labels existed.
PREDICTED_RATE = 0.389
NAIVE_BALANCED_RATE = 0.325


def main(argv=None) -> int:
    import mine_policy_corpus as M
    import ablation_analyze as A

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reports/batchswap_wave_tables.txt")
    args = ap.parse_args(argv)

    out: list[str] = []

    def add(s: str = "") -> None:
        out.append(s)
        print(s)

    store = pd.read_parquet(BASE / "data/store/records.parquet")
    wave = store[store["campaign"] == CAMPAIGN].copy()
    wave["feasible"] = M.feasibility(wave)
    conv = wave[wave["converged"].astype(bool)]
    add(f"chains {len(wave)}  converged {len(conv)}  "
        f"feasible {int(wave['feasible'].fillna(False).sum())}")
    add(f"failures: {wave[~wave['converged'].astype(bool)]['failure'].value_counts().to_dict()}")
    add()

    # ---- §4a best feasible ------------------------------------------------ #
    add("## §4a  PRIMARY - best feasible F_r found")
    add()
    feas = conv[conv["feasible"].fillna(False).astype(bool)].sort_values("f_r")
    steps = A.build_wave_steps(CAMPAIGN, None)
    smap = steps.set_index("child_record_id")
    top = feas.head(10)[["record_id", "parent_record_id", "f_r", "f_q",
                         "cbc_max", "ao_abs", "cyclen", "node_peak"]].copy()
    top["parent_f_r"] = [float(smap.loc[r, "parent_f_r"]) if r in smap.index else np.nan
                         for r in top["record_id"]]
    top["d_f_r"] = top["f_r"] - top["parent_f_r"]
    add(top.to_string(index=False, max_colwidth=14,
                      float_format=lambda v: f"{v:.4f}"))
    add()
    if len(feas):
        b = feas.iloc[0]
        add(f"BEST FEASIBLE: {b.record_id[:12]}  F_r {b.f_r:.4f}")
        add(f"  parent {b.parent_record_id[:12]}  F_q {b.f_q:.4f}  "
            f"CBC {b.cbc_max:.2f}  |AO| {b.ao_abs:.4f}  cyclen {b.cyclen:.3f}  "
            f"node_peak {b.node_peak:.4f}")
        add(f"  vs prior cell record {CELL_RECORD_PRIOR:.4f}: "
            f"{b.f_r - CELL_RECORD_PRIOR:+.4f}"
            f"  {'NEW CELL RECORD' if b.f_r < CELL_RECORD_PRIOR else 'no improvement'}")
        add(f"  vs ga80 incumbent   {GA80_INCUMBENT:.4f}: "
            f"{b.f_r - GA80_INCUMBENT:+.4f}"
            f"  {'BEATS INCUMBENT' if b.f_r < GA80_INCUMBENT else 'incumbent stands'}")
    add()

    # ---- §4b distribution + the registered audit -------------------------- #
    add("## §4b  SECONDARY - improving fraction: proportional vs balanced")
    add()
    ok = steps["both_converged"].fillna(False).astype(bool)
    s = steps[ok].copy()
    imp = s["improved_fr"].astype("boolean")
    rate = float(imp.mean())
    n = int(imp.notna().sum())
    se = float(np.sqrt(rate * (1 - rate) / n)) if n else np.nan
    add(f"THIS WAVE (proportional draw, n={n}): improving_fr = {rate:.3f} "
        f"+- {1.96 * se:.3f} (95%)")
    add(f"  registered prediction (reweighted to true mix) : {PREDICTED_RATE:.3f}")
    add(f"  naive balanced pooled value (ablation wave)    : {NAIVE_BALANCED_RATE:.3f}")
    lo, hi = rate - 1.96 * se, rate + 1.96 * se
    add(f"  prediction {'INSIDE' if lo <= PREDICTED_RATE <= hi else 'OUTSIDE'} CI"
        f"   |   naive {'INSIDE' if lo <= NAIVE_BALANCED_RATE <= hi else 'OUTSIDE'} CI")
    add()
    add("by direction (compare ablation wave: inward 0.400, outward 0.250):")
    d = s.groupby("fresh_radial_dir")["improved_fr"].agg(
        n="size", improving=lambda x: x.astype("boolean").mean())
    add(d.to_string(float_format=lambda v: f"{v:.3f}"))
    add()
    add("per-parent d_f_r distribution (negative = improvement on the parent):")
    g = s.groupby("parent_record_id")["d_f_r"]
    tbl = pd.concat([
        g.size().rename("n"), g.min().rename("min"), g.quantile(.25).rename("p25"),
        g.median().rename("p50"), g.mean().rename("mean"), g.max().rename("max"),
        g.apply(lambda x: (x < 0).mean()).rename("frac<0"),
    ], axis=1)
    tbl.index = [i[:12] for i in tbl.index]
    tbl["parent_f_r"] = [float(s[s.parent_record_id.str.startswith(i)]
                               ["parent_f_r"].iloc[0]) for i in tbl.index]
    add(tbl.sort_values("parent_f_r").to_string(
        float_format=lambda v: f"{v:.4f}"))
    add()
    add("AUDIT of the ablation wave's 4-sample balanced estimate "
        "(the two shared parents):")
    prior = pd.read_parquet(BASE / "data/policy/steps.parquet")
    ab = prior[(prior["lineage_source"] == "ablation_paramA")
               & (prior["move_class"] == "batch_swap")]
    for pid in ("1165441c31ea", "188c9a338d9f"):
        a4 = ab[ab["parent_record_id"].str.startswith(pid)]["improved_fr"].astype("boolean")
        now = s[s["parent_record_id"].str.startswith(pid)]["improved_fr"].astype("boolean")
        if not len(now):
            continue
        add(f"  {pid}: balanced 4-sample {a4.mean():.3f} (n={a4.notna().sum()})"
            f"   vs proportional {now.mean():.3f} (n={now.notna().sum()})")
    add()

    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n[analyze] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
