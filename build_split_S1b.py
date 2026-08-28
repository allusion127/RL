"""Build `data/splits/S1b.json` — S1 grown by the rows that post-date its freeze.

Pre-registered in `data/reports/ab2_addendum_SPLIT_20260810.md`.

WHY THIS IS NOT `make_curriculum_split(...)` RE-RUN
---------------------------------------------------
Two independent hazards make a full regeneration the wrong tool, and both are
recorded in the repo's own history (the 2026-07-18 lesson):

1. **The legacy pool's val carving is an RNG shuffle over ancestry groups**
   (`splits.py` ~line 405: `rng = random.Random(seed); rng.shuffle(candidates)`).
   The candidate list is derived from the store, so adding campaigns changes the
   group inventory, which changes what that shuffle selects -- a regenerated
   split can evict a whole evaluation band that used to be in train, or admit
   one that must not be in val.  So the legacy carving is COPIED, never redrawn.

2. **The curriculum pool's holdout is stable-hash but NOT size-invariant.**
   Per cell it takes the `val_count = round(0.20 * n_conv)` smallest-hash
   converged ids.  A row's *hash* is invariant, but `val_count` grows with the
   cell, so a regeneration on a grown store pulls MORE ids out of the same sorted
   pool.  Existing val ids are very likely preserved (they hold the smallest
   hashes) but nothing GUARANTEES it, and "very likely" is not a property an
   evaluation surface may rest on.

So this script applies the cell-wise 80/20 stable-hash rule **to the increment
only**, on top of frozen S1 assignments.  Growth-invariance then holds by
construction rather than by argument: no pre-existing id can move, because no
pre-existing id is ever reconsidered.

WHAT EACH NEW ROW GETS
----------------------
* **New row in a known curriculum cell** -> the cell-wise 80/20 stable-hash
  holdout, applied to the NEW converged rows of that cell:
  `new_val_count = round(0.20 * n_new_conv)`, take the smallest `_hash01` ids
  (ties broken by record_id, exactly as `make_curriculum_split` does).
  Non-converged new rows go to train, as they do upstream.
* **New row in the legacy pool** (campaign is not a cell id) -> **train**.
  The legacy val fold is a frozen group carving; growing it means redrawing it,
  which is hazard 1.  This is also R3's precedent: every new row went to train
  and val did not grow by a single row.

Run from anywhere:  python 5_RL/build_split_S1b.py [--write]
Without `--write` it is a dry run: it prints every verification and writes
nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from lpopt.model.splits import SplitManifest, _hash01   # noqa: E402

SPLITS = REPO / "data" / "splits"
RECORDS = REPO / "data" / "store" / "records.parquet"
CELL_VAL_FRAC = 0.20                      # == make_curriculum_split's default


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _self_check(parent: str) -> int:
    """ONE runnable check for the `--holdout-new-campaigns` clause (round 7).

    Rebuilding S1b byte-identically is NOT available as a check: S1b was built at
    71,155 store rows and the store has grown since, so a rebuild legitimately
    sees a bigger increment.  What IS invariant, and is what the new clause could
    plausibly break, is the val-growth contract:

        flag OFF -> val grows by 0            (the S1b legacy->train rule)
        flag ON  -> val grows by exactly the new-only 80/20 holdout
        either   -> no parent id changes fold

    ponytail: compares COUNTS and the a/b/c set-containment checks, not the full
    manifest.  Ceiling: it would not catch two new-only campaigns swapping which
    rows they hold out.  Upgrade path: assert the chosen id set equals a
    recomputed `sorted(pool, key=_hash01)[:k]` per campaign.
    """
    import io
    import contextlib

    out = {}
    for flag in (False, True):
        argv = ["--parent", parent, "--name", "_selfcheck"]
        if flag:
            argv.append("--holdout-new-campaigns")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(argv)            # dry run: no --write, nothing is persisted
        text = buf.getvalue()
        if rc != 0:
            print(f"self-check FAIL: build returned {rc} with flag={flag}\n{text}")
            return 1
        line = [ln for ln in text.splitlines() if "new train/val" in ln][0]
        delta_val = int(line.split("+")[-1].rstrip(")"))
        newonly = int([ln for ln in text.splitlines()
                       if "new-only campaign" in ln][0].split(":")[1])
        ok = all("True" in ln for ln in text.splitlines()
                 if ln.strip().startswith(("a_", "b_", "c_")))
        out[flag] = (delta_val, newonly, ok)
        print(f"  flag={str(flag):5s} val_growth={delta_val:<5d} "
              f"new_only_rows={newonly:<5d} a/b/c_hold={ok}")

    off_growth, _, off_ok = out[False]
    on_growth, on_newonly, on_ok = out[True]
    fails = []
    if off_growth != 0:
        fails.append(f"flag OFF grew val by {off_growth}, expected 0")
    if on_newonly and on_growth <= 0:
        fails.append("flag ON saw new-only rows but held none out")
    if on_growth > on_newonly:
        fails.append(f"flag ON held out {on_growth} > {on_newonly} new-only rows")
    if not (off_ok and on_ok):
        fails.append("a/b/c growth-invariance checks did not hold")
    print("\nself-check " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return 0 if not fails else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="S1", help="split to grow FROM")
    ap.add_argument("--name", default="S1b", help="name of the split to build")
    ap.add_argument("--holdout-new-campaigns", action="store_true",
                    help="also apply the cell-wise 80/20 stable-hash holdout to "
                         "campaigns that exist ONLY in the increment (round 7 / "
                         "S1c: without it every brand-new campaign lands wholly "
                         "in train and is unmeasurable)")
    ap.add_argument("--write", action="store_true",
                    help="actually write the split (default: dry run)")
    ap.add_argument("--self-check", action="store_true",
                    help="the ONE runnable check for --holdout-new-campaigns: "
                         "off => val must not grow at all; on => val must grow by "
                         "exactly the new-only holdout, with parent assignments "
                         "unmoved either way")
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check(args.parent)

    SRC = SPLITS / f"{args.parent}.json"
    DST = SPLITS / f"{args.name}.json"

    df = pd.read_parquet(RECORDS)
    old = json.loads(SRC.read_text(encoding="utf-8"))
    g_old = old["groups"]

    train0 = list(map(str, old["train_ids"]))
    val0 = list(map(str, old["val_ids"]))
    train0_set, val0_set = set(train0), set(val0)
    covered = train0_set | val0_set

    rid = df["record_id"].astype(str)
    conv = df["converged"].astype(bool)
    camp = df["campaign"].map(
        lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
    frozen_cells = dict(g_old["curriculum_val_by_cell"])
    # A "known cell" for holdout purposes is a campaign that ALREADY HAS a val
    # holdout in the parent -- i.e. a key of `curriculum_val_by_cell`, not of
    # `groups['cells']`.  The two coincided at S1b and diverged the moment a
    # round promoted a new campaign to a cell: `groups['cells']` is never
    # updated, so from S1c on it lists 36 while the parent really scores 48.
    # Reading the stale key would give a campaign the 80/20 holdout in the round
    # it appears and then silently send its NEXT increment wholly to train.
    #
    # ponytail: `groups['cells']` is deliberately left stale.  Ceiling: it is
    # also what the TRAINER reads for the curriculum sampling-weight cap
    # (`train.py` ~1733), so rewriting it here would change training behaviour,
    # not just fold assignment -- a second change, inadmissible in a data-growth
    # arm.  Upgrade path: promote it in a dedicated arm that A/Bs the weight cap.
    cells = set(map(str, g_old["cells"])) | set(map(str, frozen_cells))
    # The HISTORICAL 3,207-row AB2 surface.  Once a parent carries it, it must be
    # forwarded verbatim -- rebuilding it from the parent's (already grown)
    # curriculum map would silently redefine the surface every round.
    ab2_frozen = dict(g_old.get("ab2_frozen_val_by_cell") or frozen_cells)

    is_new = ~rid.isin(covered)
    new_df = df[is_new]
    new_rid = rid[is_new]
    new_camp = camp[is_new]
    new_conv = conv[is_new]

    # Campaigns that exist ONLY in the increment.  Holding 20% of a brand-new
    # campaign out does NOT touch the legacy group carving (nothing of it is
    # assigned yet), so hazard 1 in the module docstring is not engaged -- which
    # is why this is admissible where growing the EXISTING legacy val fold is not.
    new_only = (set(new_camp) - set(camp[~is_new])) if args.holdout_new_campaigns \
        else set()
    holdout_cells = cells | new_only

    print(f"store rows          : {len(df)}")
    print(f"{args.parent} train/val : {len(train0)} / {len(val0)}")
    print(f"new (in neither)    : {len(new_df)}")
    in_cell = new_camp.isin(holdout_cells)
    print(f"  in a known cell   : {int(new_camp.isin(cells).sum())}")
    print(f"  new-only campaign : {int(new_camp.isin(new_only).sum())}")
    print(f"  legacy -> train   : {int((~in_cell).sum())}")

    # ---- curriculum increment: cell-wise 80/20 stable hash on NEW rows only --- #
    add_val: list[str] = []
    per_cell: dict[str, dict] = {}
    for cell in sorted(set(new_camp[in_cell])):
        m = in_cell & new_camp.eq(cell)
        cell_new_conv = sorted(new_rid[m & new_conv].tolist())
        n_new_conv = len(cell_new_conv)
        n_new_total = int(m.sum())
        if n_new_conv >= 2:
            k = max(1, int(round(CELL_VAL_FRAC * n_new_conv)))
            k = min(k, n_new_conv - 1)          # keep >=1 new converged in train
        else:
            k = 0                                # a lone new converged row: train
        pool = sorted(cell_new_conv, key=lambda r: (_hash01(r), r))
        chosen = sorted(pool[:k])
        add_val.extend(chosen)
        per_cell[cell] = {"new_rows": n_new_total, "new_converged": n_new_conv,
                          "new_val": len(chosen), "new_train": n_new_total - len(chosen)}
        print(f"  cell {cell:<16} new={n_new_total:<5} conv={n_new_conv:<5} "
              f"-> val {len(chosen)}, train {n_new_total - len(chosen)}")

    add_val_set = set(add_val)
    new_train = sorted(set(new_rid) - add_val_set)

    train1 = sorted(train0_set | set(new_train))
    val1 = sorted(val0_set | add_val_set)

    # ---- the four pre-registered verifications ------------------------------- #
    checks: dict = {}
    checks["a_every_old_val_stays_val"] = val0_set <= set(val1)
    checks["b_every_old_train_stays_train"] = train0_set <= set(train1)
    checks["c_only_new_ids_get_fresh_assignments"] = (
        (set(train1) - train0_set) | (set(val1) - val0_set)) == set(new_rid)
    # (d) the frozen 36-cell surface.  The historical AB2 surface is the ID LIST
    # per cell; it must survive verbatim.  Three cells legitimately GAIN rows
    # (that is the arm), so the check is: every frozen cell still exists, its
    # original id list is a subset of the new one in unchanged order, and no
    # original id was dropped or reordered.
    cell_of_new = dict(zip(new_rid, new_camp))
    val_by_cell_new = {c: list(v) for c, v in frozen_cells.items()}
    for r in add_val:
        c = cell_of_new[r]
        # setdefault: a new-only campaign is a cell the parent never had
        val_by_cell_new[c] = sorted(val_by_cell_new.setdefault(c, []) + [r])
    # subset, not equality: a new-only campaign legitimately ADDS a cell.  What
    # may never happen is a parent cell disappearing.
    checks["d_frozen36_cells_all_present"] = set(frozen_cells) <= set(val_by_cell_new)
    checks["d_cells_added"] = sorted(set(val_by_cell_new) - set(frozen_cells))
    checks["d_ab2_surface_forwarded_verbatim"] = (
        ab2_frozen == (g_old.get("ab2_frozen_val_by_cell") or frozen_cells))
    checks["d_ab2_surface_rows"] = sum(len(v) for v in ab2_frozen.values())
    checks["d_frozen36_ids_all_retained"] = all(
        set(frozen_cells[c]) <= set(val_by_cell_new[c]) for c in frozen_cells)
    checks["d_frozen36_original_order_intact"] = all(
        [x for x in val_by_cell_new[c] if x in set(frozen_cells[c])] == list(frozen_cells[c])
        for c in frozen_cells)
    checks["d_frozen36_rows_total"] = sum(len(v) for v in frozen_cells.values())
    checks["d_cells_that_grew"] = sorted(
        c for c in frozen_cells if len(val_by_cell_new[c]) != len(frozen_cells[c]))
    checks["no_train_val_overlap"] = not (set(train1) & set(val1))
    checks["covers_every_store_row"] = set(train1) | set(val1) == set(rid)

    print("\n--- pre-registered verifications ---")
    for k, v in checks.items():
        print(f"  {k:42s} {v}")
    print(f"\nnew train/val       : {len(train1)} / {len(val1)} "
          f"(+{len(train1)-len(train0)} / +{len(val1)-len(val0)})")

    MUST_HOLD = ("a_every_old_val_stays_val", "b_every_old_train_stays_train",
                 "c_only_new_ids_get_fresh_assignments",
                 "d_frozen36_cells_all_present", "d_frozen36_ids_all_retained",
                 "d_ab2_surface_forwarded_verbatim",
                 "d_frozen36_original_order_intact",
                 "no_train_val_overlap", "covers_every_store_row")
    failed = [k for k in MUST_HOLD if not checks[k]]
    if failed:
        print(f"\nREFUSED: verification(s) failed {failed}; nothing written.")
        return 1

    # The copied `groups` carries S1's per-cell converged counts, which are stale
    # for the three cells that grew.  A manifest whose bookkeeping disagrees with
    # its own id lists is a trap for the next reader, so refresh them from the
    # store rather than shipping numbers that no longer describe the split.
    conv_counts = dict(g_old.get("curriculum_conv_counts", {}))
    train_conv_counts = dict(g_old.get("curriculum_train_conv_counts", {}))
    val1_set_tmp = set(val1)
    for cell in per_cell:
        m = camp.eq(cell) & conv
        cell_conv = set(rid[m])
        conv_counts[cell] = len(cell_conv)
        train_conv_counts[cell] = len(cell_conv - val1_set_tmp)

    man = SplitManifest(
        name=args.name, kind="curriculum_group", seed=int(old["seed"]),
        train_ids=train1, val_ids=val1, status="ok",
        predicate={**old.get("predicate", {}), "derived_from": "S1",
                   "increment_rule": "cell-wise 80/20 stable hash on NEW rows only; "
                                     "legacy-pool new rows -> train"},
        groups={
            **g_old,
            "curriculum_val_by_cell": {c: sorted(v) for c, v in val_by_cell_new.items()},
            "curriculum_conv_counts": conv_counts,
            "curriculum_train_conv_counts": train_conv_counts,
            # The historical AB2 scoring surface, forwarded VERBATIM so rounds
            # 1-3 + BU stay comparable however many cells have since grown.
            "ab2_frozen_val_by_cell": {c: list(v) for c, v in ab2_frozen.items()},
            "ab2_frozen_n_rows": sum(len(v) for v in ab2_frozen.values()),
            "derived_from_split": args.parent,
            "derived_from_split_sha256": sha256(SRC),
            "records_sha256": sha256(RECORDS),
            "increment_new_ids": sorted(new_rid),
            "increment_per_cell": per_cell,
            "increment_legacy_to_train": len(new_train) - sum(
                v["new_train"] for v in per_cell.values()),
        },
    )
    if args.write:
        man.to_json(DST)
        print(f"\nwrote {DST}\n  sha256 {sha256(DST)}")
    else:
        print("\n(dry run — pass --write to persist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
