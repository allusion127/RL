"""Readout for the multi-type mesh — the numbers the README quotes.

Three blocks, in the order a reader needs them:

1. **coverage** — how many of the 90 cells even ADMIT a composition-matched
   3- or 4-type case, split by R1 spec.  A delta map is unreadable without it:
   a blank cell because the roster has no third rung and a blank cell because
   grading did nothing are opposite statements.
2. **the delta** — the distribution of ``delta(3-2)`` on the joint-clean F_r
   floor, and the per-cell table.
3. **the 2-type control** — this sweep's k=2 column against the v3 sweep's own
   cells.  Same 90 cells, same 1 200-candidate budget, same gate on four of the
   five axes; the model changed s1g -> s1i and the pin axis joined the gate.
   Any k=2 movement is those two things, and stating its size is what stops a
   reader attributing it to the multi-type machinery.

    python mesh_multitype_readout.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "reports" / "mesh_multitype_20260818"
V3 = BASE / "data" / "reports" / "mesh_v3_20260817"


def main() -> int:
    d = pd.read_csv(OUT / "mesh_multitype.csv")
    lines: list[str] = []

    def log(m: str = "") -> None:
        print(m, flush=True)
        lines.append(str(m))

    d["has3"] = d.case_3.fillna("").astype(str) != ""
    d["has4"] = d.case_4.fillna("").astype(str) != ""
    d["mono3"] = d.get("mono_anchor_3", False)
    log("=== 1. COVERAGE — which cells admit a composition-matched multi-type case ===")
    log(f"cells swept: {len(d)}")
    log(f"  with a 3-type case: {int(d.has3.sum())}"
        f"   (R1 mono-spec {int((d.has3 & d.mono3.astype(bool)).sum())}, "
        f"cross-spec {int((d.has3 & ~d.mono3.astype(bool)).sum())})")
    log(f"  with a 4-type case: {int(d.has4.sum())}   — ALL of them EXTRAPOLATION "
        f"(0 four-type rows in the training store)")
    log(f"  no 3-type ladder in the roster: {int((~d.has3).sum())} cells "
        f"(e-levels {sorted(d[~d.has3].e_target.unique())})")
    log("")
    log("by e-level:")
    cov = d.groupby("e_target").agg(
        n=("cell", "size"), has3=("has3", "sum"), mono3=("mono3", "sum"),
        has4=("has4", "sum"))
    log(cov.to_string())

    log("")
    log("=== 2. THE DELTA — delta(3-2) on the joint-clean F_r floor ===")
    log("(joint-clean = passes CBC, F_q, |AO| and predicted pin <= 78; F_r free)")
    g = d[d.has3].copy()
    dd = g.d_min_f_r_clean_3v2.astype(float)
    fin = dd[np.isfinite(dd)]
    if len(fin):
        log(f"defined in {len(fin)}/{int(d.has3.sum())} cells with a triple "
            f"(the rest have no joint-clean core at one of the two counts)")
        log(f"  mean {fin.mean():+.4f}   median {fin.median():+.4f}   "
            f"best {fin.min():+.4f}   worst {fin.max():+.4f}")
        log(f"  gains beyond the 0.005 dead band: {int((fin < -0.005).sum())}   "
            f"losses beyond it: {int((fin > 0.005).sum())}   "
            f"inside it: {int((fin.abs() <= 0.005).sum())}")
    else:
        log("  NO cell has a defined delta — see §4.3 of the pre-registration.")

    cols = ["cell", "library_id", "pair", "case_3", "mono_anchor_3", "nested_3",
            "min_f_r_clean_2", "min_f_r_clean_3", "d_min_f_r_clean_3v2",
            "clean_pin_2", "clean_pin_3", "n_clean_but_fr_2", "n_clean_but_fr_3"]
    cols = [c for c in cols if c in g.columns]
    log("\nper-cell, best delta first:")
    log(g.sort_values("d_min_f_r_clean_3v2")[cols].to_string(
        index=False, max_colwidth=46, float_format=lambda v: f"{v:.4f}"))

    log("")
    log("=== 3. THE 2-TYPE CONTROL — this sweep's k=2 vs the v3 (s1g) sweep ===")
    v3p = V3 / "mesh_nodes.csv"
    if v3p.exists():
        v3 = pd.read_csv(v3p).set_index("cell")
        j = d.set_index("cell").join(v3[["min_pred_f_r", "n_feasible", "pair"]],
                                     rsuffix="_v3", how="inner")
        delta_fr = (j.min_pred_f_r_2.astype(float) - j.min_pred_f_r.astype(float))
        same_pair = (j.pair == j.pair_v3)
        log(f"cells matched: {len(j)}   same 2-type pair in both: "
            f"{int(same_pair.sum())}/{len(j)}")
        log(f"in-band min predicted F_r, s1i - s1g: mean {delta_fr.mean():+.4f}  "
            f"median {delta_fr.median():+.4f}  sd {delta_fr.std():.4f}")
        log(f"tier-1 feasible cells: v3(s1g, 4 axes) {int((j.n_feasible > 0).sum())}"
            f"  ->  this sweep (s1i, 5 axes) "
            f"{int((j.n_feasible_2.astype(float) > 0).sum())}"
            f"  [4-axis on s1i: "
            f"{int((j.n_feasible_4ax_2.astype(float) > 0).sum())}]")
        log("The 4-axis column is the like-for-like one; the drop to 5 axes is the "
            "pin gate doing work, not a modelling change.")
    else:
        log("  v3 mesh_nodes.csv not found — control skipped.")

    calib = OUT / "calibration_hgd569.json"
    log("")
    log("=== 4. CALIBRATION — the one MEASURED 3-type delta (PREREG §6) ===")
    if calib.exists():
        c = json.load(open(calib, encoding="utf-8"))
        log(f"hgd569/f125   predicted delta {c['delta_predicted']:+.4f}   "
            f"measured delta {c['delta_measured']:+.4f}   "
            f"sign reproduced: {c['sign_reproduced']}")
        if not c["sign_reproduced"]:
            log("  REGISTERED CONSEQUENCE: no confidence is claimed for the anchor "
                "predictions (PREREG §6).")
    else:
        log("  not yet run (calib_multitype_hgd569.py)")

    (OUT / "readout.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT / 'readout.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
