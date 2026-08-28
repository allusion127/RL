"""batch_swap sweep on the 625 EFPD branch - the IN-BAND frontier.

Pre-registration: ``data/reports/batchswap625_wave_prereg_20260815.md``.

WHY: the first batch_swap wave drove F_r to 1.4605 but on the ~618 EFPD branch,
which is OUTSIDE the program band [620, 645].  On the deck's own objective
(minfr_lambda = 400 EFPD per unit F_r) that board LOSES to the r8 record.  The
in-band frontier is a different, unexplored branch.  This wave applies the same
instrument there.

FIX CARRIED FORWARD: the first wave deduplicated candidates within each parent
but not ACROSS parents, and paid 7 duplicate chains (3.2% of budget) because two
of its parents were one swap apart.  Here the dedup is GLOBAL over the whole
candidate pool before allocation.

REUSE: enumerator / annotator / runner / kit builder are ``ablation_wave``'s,
with only the provenance constants rebound.  Neither ``ablation_wave.py`` nor
``batchswap_wave.py`` is edited - both hashes stay valid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import ablation_wave as W                                    # noqa: E402

CAMPAIGN = "batchswap_enum_625_T6T4"
GENERATOR = "batchswap_enum_625"
LINEAGE_SOURCE = "batchswap_enum_625"
W.CAMPAIGN = CAMPAIGN
W.GENERATOR = GENERATOR

PAIR, FEED, LIBRARY = W.PAIR, W.FEED, W.LIBRARY
CELL = W.CELL
SEED = 20260815
BUDGET_CAP = 220

#: The program band, registered as the in-band definition for this wave.
BAND_LO, BAND_HI = 620.0, 645.0
#: Marks the result is read against.
IN_BAND_BEST = 1.4747          # abd38bc5b212, ablation wave
R8_RECORD = 1.4749             # 188c9a338d9f, fpcamp_minfr_T6T4_r8
GA80_INCUMBENT = 1.4636        # deb058c00433, E1_E2 f121 @ 633.33 EFPD
LAMBDA_EFPD = 400.0

#: Registered allocation: the in-band F_r frontier, deepest first.  A node_peak
#: parent is deliberately NOT included - the registered readout is in-band F_r,
#: and the previous wave showed the record comes from the best parents'
#: neighbourhoods, not from the high-improving-fraction weak ones.
ALLOCATION: list[tuple[str, int, str]] = [
    ("abd38bc5b212", 70, "current IN-BAND best 1.4747 @ 625.29"),
    ("188c9a338d9f", 70, "r8 campaign record 1.4749 @ 625.46"),
    ("1ca37638c03c", 40, "1.4750 @ 623.75 - the 623 sub-branch"),
    ("a6d11f78a7f6", 25, "1.4762 @ 625.60"),
    ("195699e488e6", 15, "1.4763 @ 623.74"),
]


def cmd_plan(args) -> int:
    import mine_policy_corpus as M
    import pandas as pd
    from lpopt.data.schema import compute_record_id, unpack_pattern
    from lpopt.search.construct import predicted_e_core
    from lpopt.search.verify import PRODUCE_DECK_KNOBS

    store = pd.read_parquet(BASE / "data/store/records.parquet")
    enr = M.load_enrichment(BASE / "data/store/fuel_types.parquet").get(LIBRARY)
    known = set(store["record_id"].astype(str))
    cell = store[(store["case_pair"] == PAIR) & (store["feed"] == FEED)
                 & (store["library_id"] == LIBRARY)].copy()
    cell["feasible"] = M.feasibility(cell)
    feas = cell[cell["feasible"].fillna(False).astype(bool).to_numpy()]
    inband = feas[(feas["cyclen"] >= BAND_LO) & (feas["cyclen"] <= BAND_HI)]
    print(f"[plan] cell {CELL}: {len(feas)} feasible, {len(inband)} in-band "
          f"[{BAND_LO:.0f},{BAND_HI:.0f}]  in-band best F_r {inband['f_r'].min():.4f}")

    try:
        from lpopt.data.fuel_types import FuelLibrary
        fuel = FuelLibrary.from_parquet(BASE / "data/store/fuel_types.parquet")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[plan] WARNING: fuel table unreadable ({exc})")
        fuel = None

    indexed = cell.set_index("record_id", drop=False)
    # ---- enumerate every parent FIRST, then dedup GLOBALLY ---------------- #
    pools: dict[str, "pd.DataFrame"] = {}
    parents = []
    for prefix, quota, why in ALLOCATION:
        hits = [r for r in indexed["record_id"] if r.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"prefix {prefix!r} matched {len(hits)} rows")
        row = indexed.loc[hits[0]]
        g = M.genome_of(row["pattern"])
        ann = W.annotate(row["pattern"], g, W.enum_batch_swap(g), enr)
        frame = pd.DataFrame(ann).drop_duplicates(subset=["pattern"])
        frame["parent_record_id"] = row["record_id"]
        frame["record_id"] = [
            compute_record_id(p, LIBRARY, PAIR, PRODUCE_DECK_KNOBS)
            for p in frame["pattern"]
        ]
        pools[row["record_id"]] = frame
        parents.append({
            "record_id": row["record_id"], "why": why, "quota": quota,
            "pattern": row["pattern"], "campaign": row["campaign"],
            "f_r": float(row["f_r"]), "node_peak": float(row["node_peak"]),
            "cyclen": float(row["cyclen"]), "cbc_max": float(row["cbc_max"]),
            "f_q": float(row["f_q"]), "neighbourhood": int(len(frame)),
        })

    # GLOBAL dedup: against the store, and against every earlier parent's take.
    # (The first wave deduped only within a parent and paid 7 duplicate chains.)
    claimed: set[str] = set()
    blocks, census = [], []
    for p in parents:
        frame = pools[p["record_id"]]
        in_store = frame["record_id"].isin(known)
        collide = frame["record_id"].isin(claimed)
        avail = frame[~in_store & ~collide]
        picked = avail.loc[_pick_even(avail, p["quota"], "dose")].copy()
        claimed.update(picked["record_id"])
        picked["source"] = "paid"
        for k in ("f_r", "node_peak", "cyclen", "cbc_max"):
            picked[f"parent_{k}"] = p[k]
        picked["e_core"] = ([predicted_e_core(unpack_pattern(x), fuel, LIBRARY)
                             for x in picked["pattern"]] if fuel is not None else None)
        blocks.append(picked)
        p["already_labeled"] = int(in_store.sum())
        p["cross_parent_collisions"] = int(collide.sum())
        p["taken"] = int(len(picked))
        census.append({"parent": p["record_id"][:12], "nbhd": len(frame),
                       "in_store": int(in_store.sum()),
                       "collide": int(collide.sum()), "take": len(picked)})
        print(f"[plan]   {p['record_id'][:12]} F_r {p['f_r']:.4f} @ "
              f"{p['cyclen']:.2f}  nbhd {len(frame)}  in_store "
              f"{int(in_store.sum()):3d}  cross-parent-dup {int(collide.sum()):3d}"
              f"  take {len(picked):3d}")

    cand = pd.concat(blocks, ignore_index=True)
    assert cand["record_id"].is_unique, "global dedup failed"
    if len(cand) > BUDGET_CAP:
        raise SystemExit(f"paid {len(cand)} exceeds cap {BUDGET_CAP}")

    manifest = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell": CELL, "pair": PAIR, "feed": FEED, "library_id": LIBRARY,
        "campaign": CAMPAIGN, "generator": GENERATOR, "seed": SEED,
        "deck_knobs": PRODUCE_DECK_KNOBS, "budget_cap": BUDGET_CAP,
        "band": [BAND_LO, BAND_HI], "move_class": "batch_swap",
        "marks": {"in_band_best": IN_BAND_BEST, "r8_record": R8_RECORD,
                  "ga80_incumbent": GA80_INCUMBENT, "lambda_efpd": LAMBDA_EFPD},
        "n_paid": int(len(cand)), "n_free": 0,
        "parents": parents, "census": census,
        "candidates": json.loads(cand.to_json(orient="records")),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[plan] paid {len(cand)} (cap {BUDGET_CAP}), all record_ids unique "
          f"-> {out}")
    print(f"[plan] direction mix: {cand['fresh_radial_dir'].value_counts().to_dict()}")
    return 0


def _pick_even(frame, k: int, key: str) -> list[int]:
    if len(frame) == 0 or k <= 0:
        return []
    order = frame.sort_values([key, "move_tag"], kind="mergesort").index.tolist()
    if len(order) <= k:
        return order
    pos = np.linspace(0, len(order) - 1, k)
    return [order[int(round(p))] for p in dict.fromkeys(pos.tolist()).keys()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--out", default="data/design/batchswap625_wave_20260815.json")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run")
    p.add_argument("--plan", default="data/design/batchswap625_wave_20260815.json")
    p.add_argument("--package", default="data/design/package")
    p.add_argument("--exe", default="C:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    p.add_argument("--run-dir", default="runs/batchswap_enum_625_T6T4")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--host-reserve", type=int, default=1)
    p.add_argument("--wave-size", type=int, default=16)
    p.add_argument("--max-cycles", type=int, default=16)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--max-chains", type=int, default=0)
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=W.cmd_run)

    p = sub.add_parser("kit")
    p.add_argument("--run-dir", default="runs/batchswap_enum_625_T6T4")
    p.set_defaults(func=W.cmd_kit)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
