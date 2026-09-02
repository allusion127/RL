"""Quarantine every store row of a campaign whose LABELS are known to be wrong.

``python -m lpopt.tools.quarantine_campaign --campaign <tag> --failure <reason>``
``[--store-dir data/store] [--steps data/policy/steps.parquet]``
``[--backup-dir E:/lpopt_data/5_RL/backups] [--backup-tag hgd569_quarantine]``
``[--unconverge] [--apply]``                     (dry-run is the DEFAULT)

Why this exists
---------------
A harness defect can produce rows that are *well formed and converged* and still
describe a core nobody designed.  The first such case is
``intervention_HGD569_f125`` (``data/reports/hgd569_degeneracy_memo_20260830.md``):
the wave's deck-emission path held a resolver with an empty ``type_id -> alias``
bridge, so ``%LPD_SHF`` carried raw ``type_id``\\ s, MASTER silently resolved the
whole core to one unrelated batch, and 160 chains were labelled +35 EFPD off the
cell's real physics.  MASTER converged on every one of them, so nothing in the
normal QC taxonomy rejects them.

What this does
--------------
* ``records.parquet`` — sets ``valid = False`` and ``failure = <reason>`` on the
  campaign's rows.  **Rows are never deleted.**  ``valid=False`` is the schema's
  own word for "this row is not evidence" (:func:`~..data.store._quality_rank`
  ranks a valid row above an invalid one at equal convergence), and keeping the
  row keeps the ``record_id`` occupied so a corrected re-evaluation UPGRADES it
  in place instead of arriving as a second, competing row.
* ``steps.parquet`` — DROPS the campaign's edges.  A flag column was the first
  choice and the schema does not tolerate one: ``intervention_wave.cmd_corpus``
  refuses to append when ``set(existing.columns) - set(new.columns)`` is
  non-empty, so a ``quarantined`` column added here would break the very re-run
  that repairs the data.  The rows are recoverable from the backup, and the
  corrected re-run re-appends them (``build_steps`` is deterministic).

``--unconverge`` (opt-in, READ THIS)
------------------------------------
``valid=False`` alone does NOT remove a row from the search: measured on this
tree, the elite pools, replay/holdout draws and every model training filter key
on ``converged == True`` and do not look at ``valid`` at all (the one exception
is ``CampaignDriver._replay_rows``).  A quarantined-but-converged row therefore
still seeds elites and still trains surrogates.  ``--unconverge`` additionally
sets ``converged = False``, which is what actually excludes it everywhere.  It is
opt-in because it overwrites an observed fact (MASTER did converge) with a
statement about trust; use it when the labels must leave the search NOW, and
prefer fixing the filters otherwise.

It applies to **every** row of the campaign, including rows a previous run
already quarantined: adding ``--unconverge`` to a campaign that is already
``valid=False`` is the normal way to escalate, so "already quarantined" must
never mean "skipped".  The rows keep their labels and their ``failure`` tag;
only ``converged`` goes to ``False``.

Contract (same as :mod:`.repair_parent_ids` / :mod:`.backfill_flatness`)
------------------------------------------------------------------------
* **Dry-run by default** — ``--apply`` is required to write anything.
* **Backup first** — ``--apply`` copies each file it will touch into
  ``--backup-dir`` as ``<name>.bak_pre_<tag>_<YYYYMMDD>`` and refuses to clobber
  an existing copy of that name.
* **Atomic + order-preserving** — the records edit re-reads the file, keys by
  ``record_id`` (never by position) and goes out through the store's own
  :func:`..data.store._atomic_write` / :func:`..data.store.frame_to_table`.
* **Idempotent** — a second run finds nothing to change and writes nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

from ..data.store import (
    RECORDS_NAME, _atomic_write, ensure_schema_columns, frame_to_table,
)
from ..safelog import configure_stdio

#: Where the programme keeps its pre-mutation safety copies.
DEFAULT_BACKUP_DIR = Path("E:/lpopt_data/5_RL/backups")


def sha256_of(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass
class Plan:
    """What a run would change, per file."""

    records_rows: int = 0
    records_already: int = 0
    records_unconverge: int = 0
    records_pending: int = 0
    steps_rows: int = 0
    steps_total: int = 0


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def pending_mask(df: pd.DataFrame, campaign: str, failure: str,
                 *, unconverge: bool) -> tuple[pd.Series, pd.Series]:
    """``(mask of the campaign's rows, mask of rows not yet in the end state)``.

    The end state is ``valid=False`` + ``failure=<tag>`` and, under
    ``--unconverge``, ``converged=False`` as well.  A row that is ALREADY
    quarantined but still carries ``converged=True`` is therefore *pending*
    under ``--unconverge``: prior quarantine state never excuses a row from the
    converged flip, which is the whole point of the flag.  The dry-run scan and
    the apply path share this one predicate so the count that is printed is
    exactly the count that is written.
    """

    mask = df["campaign"].fillna("").astype(str) == campaign
    pending = mask & (
        df["valid"].fillna(True).astype(bool)
        | (df["failure"].fillna("").astype(str) != failure)
    )
    if unconverge:
        pending = pending | (mask & df["converged"].fillna(False).astype(bool))
    return mask, pending


def scan_records(df: pd.DataFrame, campaign: str, failure: str,
                 *, unconverge: bool = False) -> tuple[pd.Series, Plan]:
    """``(mask of the campaign's rows, plan counts)``."""

    mask, pending = pending_mask(df, campaign, failure, unconverge=unconverge)
    already = mask & (~df["valid"].fillna(True).astype(bool)) & (
        df["failure"].fillna("").astype(str) == failure)
    plan = Plan(
        records_rows=int(mask.sum()),
        records_already=int(already.sum()),
        records_unconverge=int((mask & df["converged"].fillna(False).astype(bool)).sum()),
        records_pending=int(pending.sum()),
    )
    return mask, plan


def quarantine_records(path: Path, campaign: str, failure: str,
                       *, unconverge: bool) -> int:
    """Flip ``valid``/``failure`` (and optionally ``converged``) in place.

    Returns the number of rows actually changed (0 => nothing written).
    """

    current = ensure_schema_columns(pd.read_parquet(path))
    mask, pending = pending_mask(current, campaign, failure, unconverge=unconverge)
    n = int(pending.sum())
    if not n:
        return 0

    current.loc[mask, "valid"] = False
    current.loc[mask, "failure"] = failure
    if unconverge:
        current.loc[mask, "converged"] = False
    table = frame_to_table(current)
    _atomic_write(path, lambda p: pq.write_table(table, p))
    return n


def drop_steps(path: Path, campaign: str) -> int:
    """Drop the campaign's edges from ``steps.parquet``.  Returns rows dropped."""

    steps = pd.read_parquet(path)
    if "campaign" not in steps.columns:
        return 0
    keep = steps["campaign"].fillna("").astype(str) != campaign
    n = int((~keep).sum())
    if not n:
        return 0
    steps.loc[keep].reset_index(drop=True).to_parquet(path, index=False)
    return n


def backup(path: Path, backup_dir: Path, tag: str) -> Path:
    """Copy ``path`` into ``backup_dir`` as ``<name>.bak_pre_<tag>_<date>``."""

    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{path.name}.bak_pre_{tag}_{time.strftime('%Y%m%d')}"
    if dest.exists():
        raise FileExistsError(
            f"backup already exists: {dest} -- move or delete it before re-running "
            "with --apply (refusing to overwrite a prior safety copy)")
    shutil.copy2(path, dest)
    return dest


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    ap = argparse.ArgumentParser(
        prog="python -m lpopt.tools.quarantine_campaign",
        description="Mark a campaign's store rows invalid (never delete them) "
                    "and drop its policy-corpus edges.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--failure", required=True,
                    help="the `failure` tag written on every quarantined row, "
                         "e.g. alias_noop_P6_20260830")
    ap.add_argument("--store-dir", default="data/store", type=Path)
    ap.add_argument("--steps", default="data/policy/steps.parquet", type=Path)
    ap.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR, type=Path)
    ap.add_argument("--backup-tag", default=None,
                    help="backup filename tag (default: derived from --failure)")
    ap.add_argument("--unconverge", action="store_true",
                    help="ALSO set converged=False (the only thing that removes "
                         "the rows from elite pools / model training, which key "
                         "on `converged`, not `valid`)")
    ap.add_argument("--no-steps", action="store_true",
                    help="leave steps.parquet alone")
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default: dry-run, no file touched)")
    args = ap.parse_args(argv)

    records = Path(args.store_dir) / RECORDS_NAME
    if not records.exists():
        print(f"no store at {records}", file=sys.stderr)
        return 2
    tag = args.backup_tag or args.failure.replace(":", "_")

    df = ensure_schema_columns(pd.read_parquet(records))
    mask, plan = scan_records(df, args.campaign, args.failure,
                              unconverge=args.unconverge)
    steps_path = Path(args.steps)
    do_steps = not args.no_steps and steps_path.exists()
    if do_steps:
        steps = pd.read_parquet(steps_path, columns=["campaign"])
        plan.steps_total = len(steps)
        plan.steps_rows = int(
            (steps["campaign"].fillna("").astype(str) == args.campaign).sum())

    print(f"campaign        {args.campaign}")
    print(f"failure tag     {args.failure}")
    print(f"records         {records}  sha256 {sha256_of(records)}")
    print(f"  rows in campaign        {plan.records_rows}")
    print(f"  already quarantined     {plan.records_already}")
    print(f"  converged (would {'clear' if args.unconverge else 'KEEP'}) {plan.records_unconverge}")
    print(f"  rows to change          {plan.records_pending}")
    if do_steps:
        print(f"steps           {steps_path}  sha256 {sha256_of(steps_path)}")
        print(f"  edges to DROP           {plan.steps_rows} of {plan.steps_total}")
    if not plan.records_rows:
        print("\nnothing matches -- no file touched")
        return 0
    if not args.unconverge and plan.records_unconverge:
        print("\nNOTE: valid=False does NOT remove these rows from elite pools or "
              "model training -- those filters key on `converged`.  Pass "
              "--unconverge to exclude them, or exclude the campaign explicitly "
              "downstream.")

    if not plan.records_pending and not (do_steps and plan.steps_rows):
        # Idempotency, and the reason it is checked HERE rather than inside
        # `quarantine_records`: a re-run must not trip the backup-exists guard
        # (which aborts the whole apply) just to discover it had nothing to do.
        print("\nalready in the requested state -- 0 rows to change, no file touched")
        return 0

    if not args.apply:
        print("\nDRY RUN -- pass --apply to write (a backup is taken first)")
        return 0

    print(f"\nbackup -> {backup(records, args.backup_dir, tag)}")
    n = quarantine_records(records, args.campaign, args.failure,
                           unconverge=args.unconverge)
    print(f"records: {n} row(s) changed  sha256 {sha256_of(records)}")
    if do_steps and plan.steps_rows:
        print(f"backup -> {backup(steps_path, args.backup_dir, tag)}")
        dropped = drop_steps(steps_path, args.campaign)
        print(f"steps:   {dropped} row(s) dropped  sha256 {sha256_of(steps_path)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
