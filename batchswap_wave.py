"""batch_swap deep-sample wave - the under-played move class, on the frontier.

Pre-registration: ``data/reports/batchswap_wave_prereg_20260815.md``.

WHY: the 1-move ablation wave found the cell record (F_r 1.4685) with a
``batch_swap`` - a class the campaigns have played 4 times in 19,820 same-cell
corpus moves, because ``_one_move`` reaches it only via ``batch_prob`` (0.15)
split with ``batch_flip``, and a swap additionally needs two fresh units of
DIFFERENT batches.  This wave spends 220 MASTER chains on that class alone.

REUSE, NOT DUPLICATION: the enumerator, annotator, runner and kit builder are
``ablation_wave``'s, imported here.  Only the two provenance constants differ, so
they are rebound on the imported module rather than copied.  ``ablation_wave.py``
itself is NOT edited - its sha256 is pinned in the ablation pre-registration §8
and it is the artefact that ran on 199.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import ablation_wave as W                                    # noqa: E402

CAMPAIGN = "batchswap_enum_T6T4"
GENERATOR = "batchswap_enum"
LINEAGE_SOURCE = "batchswap_enum"
#: Rebound so ``W._build_kit`` stamps THIS wave's provenance onto its rows.
W.CAMPAIGN = CAMPAIGN
W.GENERATOR = GENERATOR

PAIR, FEED, LIBRARY = W.PAIR, W.FEED, W.LIBRARY
CELL = W.CELL
SEED = 20260815
BUDGET_CAP = 220

#: Registered allocation.  One parent's batch_swap neighbourhood is ~224 boards,
#: so the cap buys ONE exhaustive parent or a deep sample across several; the
#: pre-registration §2 explains why the latter was chosen and corrects the
#: "~200 chains for the elite set" estimate that sized this wave.
ALLOCATION: list[tuple[str, int, str]] = [
    ("d84668059508", 70, "the record core (F_r 1.4685) - frontier expansion"),
    ("1165441c31ea", 70, "the parent that produced it; an ablation-wave parent, "
                         "so its 4-sample stratified estimate can be audited"),
    ("a4291805f655", 25, "cell #2 (1.4740), the record's sibling"),
    ("abd38bc5b212", 25, "cell #3 (1.4747), a different basin"),
    ("188c9a338d9f", 20, "r8 top-F_r elite (1.4749); also an ablation parent"),
    ("c6edd01be332", 10, "best node_peak in the cell (1.2742)"),
]


def _pick_even(frame, k: int, key: str) -> list[int]:
    """``k`` evenly spaced RANKS over ``key`` - the ablation wave's rule."""
    if len(frame) == 0 or k <= 0:
        return []
    order = frame.sort_values([key, "move_tag"], kind="mergesort").index.tolist()
    if len(order) <= k:
        return order
    pos = np.linspace(0, len(order) - 1, k)
    return [order[int(round(p))] for p in dict.fromkeys(pos.tolist()).keys()]


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
    print(f"[plan] cell {CELL}: {len(cell)} rows, {len(feas)} feasible, "
          f"best F_r {feas['f_r'].min():.4f}")

    try:
        from lpopt.data.fuel_types import FuelLibrary
        fuel = FuelLibrary.from_parquet(BASE / "data/store/fuel_types.parquet")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[plan] WARNING: fuel table unreadable ({exc}); e_core null")
        fuel = None

    indexed = cell.set_index("record_id", drop=False)
    parents, blocks, census = [], [], []
    for prefix, quota, why in ALLOCATION:
        hits = [r for r in indexed["record_id"] if r.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"parent prefix {prefix!r} matched {len(hits)} rows")
        row = indexed.loc[hits[0]]
        g = M.genome_of(row["pattern"])
        kids = W.enum_batch_swap(g)
        ann = W.annotate(row["pattern"], g, kids, enr)
        frame = pd.DataFrame(ann).drop_duplicates(subset=["pattern"])
        frame["parent_record_id"] = row["record_id"]
        frame["record_id"] = [
            compute_record_id(p, LIBRARY, PAIR, PRODUCE_DECK_KNOBS)
            for p in frame["pattern"]
        ]
        frame["already_labeled"] = frame["record_id"].isin(known)
        fresh_pool = frame[~frame["already_labeled"]]
        # Proportional (UNstratified) even-rank draw over dose: an unbiased
        # subsample of the true neighbourhood, which is exactly the contrast the
        # ablation wave's DIRECTION-BALANCED 4-sample estimate must be audited
        # against (prereg §4b).
        picked = fresh_pool.loc[_pick_even(fresh_pool, quota, "dose")].copy()
        picked["source"] = "paid"
        picked["parent_f_r"] = float(row["f_r"])
        picked["parent_node_peak"] = float(row["node_peak"])
        picked["parent_cyclen"] = float(row["cyclen"])
        picked["parent_cbc_max"] = float(row["cbc_max"])
        picked["e_core"] = ([predicted_e_core(unpack_pattern(p), fuel, LIBRARY)
                             for p in picked["pattern"]] if fuel is not None else None)
        blocks.append(picked)
        parents.append({
            "record_id": row["record_id"], "why": why, "quota": quota,
            "pattern": row["pattern"], "campaign": row["campaign"],
            "f_r": float(row["f_r"]), "node_peak": float(row["node_peak"]),
            "cyclen": float(row["cyclen"]), "cbc_max": float(row["cbc_max"]),
            "f_q": float(row["f_q"]),
            "neighbourhood": int(len(frame)),
            "already_labeled": int(frame["already_labeled"].sum()),
            "coverage": round(quota / len(frame), 4),
        })
        census.append({
            "parent": row["record_id"][:12], "n_total": int(len(frame)),
            "n_free": int(frame["already_labeled"].sum()), "quota": quota,
            **frame["fresh_radial_dir"].value_counts().to_dict()})
        print(f"[plan]   {row['record_id'][:12]} F_r {row['f_r']:.4f}  "
              f"neighbourhood {len(frame):3d}  free {int(frame['already_labeled'].sum()):2d}  "
              f"take {len(picked):3d}  coverage {quota / len(frame):.0%}")

    cand = pd.concat(blocks, ignore_index=True)
    if len(cand) > BUDGET_CAP:
        raise SystemExit(f"paid {len(cand)} exceeds cap {BUDGET_CAP}")
    manifest = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell": CELL, "pair": PAIR, "feed": FEED, "library_id": LIBRARY,
        "campaign": CAMPAIGN, "generator": GENERATOR, "seed": SEED,
        "deck_knobs": PRODUCE_DECK_KNOBS, "budget_cap": BUDGET_CAP,
        "n_paid": int(len(cand)), "n_free": 0,
        "move_class": "batch_swap",
        "parents": parents, "census": census,
        "candidates": json.loads(cand.to_json(orient="records")),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[plan] paid {len(cand)} (cap {BUDGET_CAP}) -> {out}")
    print(f"[plan] direction mix: {cand['fresh_radial_dir'].value_counts().to_dict()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--out", default="data/design/batchswap_wave_20260815.json")
    p.set_defaults(func=cmd_plan)

    # run / kit are ablation_wave's, with this wave's provenance rebound above.
    p = sub.add_parser("run")
    p.add_argument("--plan", default="data/design/batchswap_wave_20260815.json")
    p.add_argument("--package", default="data/design/package")
    p.add_argument("--exe", default="C:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    p.add_argument("--run-dir", default="runs/batchswap_enum_T6T4")
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
    p.add_argument("--run-dir", default="runs/batchswap_enum_T6T4")
    p.set_defaults(func=W.cmd_kit)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
