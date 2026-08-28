"""P3b anchor selection — the PRE-REGISTERED rule, executed mechanically.

``PREREG_multitype_mesh_20260818.md`` §4.2 fixed five hard filters and a
three-key sort BEFORE the sweep was read.  This script is that text as code, so
the choice of which cells get 60 MASTER calls contains no discretion exercised
after seeing the numbers.  It prints every cell's verdict — including WHY each
rejected cell was rejected — because a filter that only shows its survivors
cannot be audited.

    python anchor_select_multitype.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "reports" / "mesh_multitype_20260818"

# ---- PREREG §4.2, verbatim ------------------------------------------------- #
EXCLUDE_CELLS = {(5.7, 125)}      # F1 — hgd569/f125, already measured 3 ways
FR_NEAR_GATE = 1.75               # F4 — beyond this the 3-type step cannot reach
MIN_GAIN = -0.005                 # F5 — noise floor on the delta
MAX_ANCHORS = 3


def _fin(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def verdicts(d: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: every filter's outcome and the sort keys."""

    pair_feed = store.groupby(["library_id", "case_pair", "feed"]).size()
    pair_any = store.groupby(["library_id", "case_pair"]).size()

    rows = []
    for r in d.itertuples():
        e, f = round(float(r.e_target), 1), int(r.feed)
        case3 = getattr(r, "case_3", "")
        has3 = isinstance(case3, str) and bool(case3)
        gain = float(getattr(r, "d_min_f_r_clean_3v2", np.nan))
        fr3 = float(getattr(r, "min_f_r_clean_3", np.nan))
        n_here = int(pair_feed.get((r.library_id, r.pair, f), 0))
        n_any = int(pair_any.get((r.library_id, r.pair), 0))
        why = []
        if (e, f) in EXCLUDE_CELLS:
            why.append("F1 already-measured cell")
        if not has3:
            why.append("F2 no composition-matched triple")
        elif int(getattr(r, "nested_3", 0)) != 2:
            why.append(f"F2 triple not nested (nested={int(getattr(r,'nested_3',0))}/2)")
        if has3 and not bool(getattr(r, "mono_anchor_3", False)):
            why.append("F2' triple is R1 cross-spec")
        if n_here == 0 and n_any == 0:
            why.append("F3 no restart asset for this pair at any feed")
        if not _fin(fr3) or fr3 > FR_NEAR_GATE:
            why.append(f"F4 joint-clean F_r floor {fr3:.4f} > {FR_NEAR_GATE}"
                       if _fin(fr3) else "F4 no joint-clean 3-type core")
        if not _fin(gain) or gain > MIN_GAIN:
            why.append(f"F5 gain {gain:+.4f} not below {MIN_GAIN}"
                       if _fin(gain) else "F5 gain undefined")
        rows.append(dict(cell=r.cell, e_target=e, feed=f, library_id=r.library_id,
                         pair=r.pair, case_3=case3 if has3 else "",
                         nested_3=int(getattr(r, "nested_3", 0)),
                         mono_anchor_3=bool(getattr(r, "mono_anchor_3", False)),
                         min_f_r_clean_2=float(getattr(r, "min_f_r_clean_2", np.nan)),
                         min_f_r_clean_3=fr3, gain=gain,
                         n_store_pair_feed=n_here, n_store_pair_any=n_any,
                         passes=not why, reject_reasons="; ".join(why)))
    return pd.DataFrame(rows)


def choose(v: pd.DataFrame) -> pd.DataFrame:
    """PREREG §4.2 sort + the one-cell-per-e-level diversity constraint."""

    ok = v[v.passes].sort_values(
        ["gain", "min_f_r_clean_3", "n_store_pair_feed"],
        ascending=[True, True, False])
    picked, used_e = [], set()
    for r in ok.itertuples():
        if r.e_target in used_e:
            continue
        picked.append(r.Index)
        used_e.add(r.e_target)
        if len(picked) >= MAX_ANCHORS:
            break
    return v.loc[picked]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=str(OUT / "mesh_multitype.csv"))
    args = ap.parse_args()

    d = pd.read_csv(args.nodes)
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    store = store[(store.valid == True) & (store.converged == True)]   # noqa: E712

    v = verdicts(d, store)
    v.to_csv(OUT / "anchor_verdicts.csv", index=False, encoding="utf-8")
    sel = choose(v)
    sel.to_csv(OUT / "anchor_selected.csv", index=False, encoding="utf-8")

    print(f"cells evaluated: {len(v)}   passing all five filters: {int(v.passes.sum())}")
    print("\n-- every cell with a triple, best first by the registered sort --")
    show = v[v.case_3 != ""].sort_values(["gain"])
    cols = ["cell", "pair", "case_3", "nested_3", "mono_anchor_3",
            "min_f_r_clean_2", "min_f_r_clean_3", "gain", "n_store_pair_feed",
            "passes", "reject_reasons"]
    print(show[cols].to_string(index=False, max_colwidth=52))
    print(f"\n-- SELECTED ({len(sel)}) --")
    print(sel[cols[:-1]].to_string(index=False, max_colwidth=52) if len(sel)
          else "NONE — PREREG §4.3 applies: P3b is NOT run.")
    print(f"\nwrote {OUT/'anchor_verdicts.csv'} and {OUT/'anchor_selected.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
