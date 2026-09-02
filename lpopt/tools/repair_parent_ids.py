"""Repair dangling ``parent_record_id`` foreign keys in ``records.parquet``.

``python -m lpopt.tools.repair_parent_ids [--store-dir data/store] [--campaign X]``
``[--min-rows 20] [--null-phantom] [--apply]``   (dry-run is the DEFAULT)

The defect
----------
``parent_record_id`` is a store foreign key, but three producers used to stamp it
with the ``record_id`` of an *unverified pool candidate* -- a well-formed 64-hex
:func:`~..data.schema.compute_record_id` preimage of a board that was surrogate-
scored and discarded without ever reaching MASTER, so no such row exists:

* :func:`..search.acquisition.local_search` -- the hill-climb's ``current`` board
  (generator ``local``: 27 of 2,200 children resolved);
* ``CampaignDriver._lean_local_search`` -- same shape;
* ``CampaignDriver`` ``prev_top`` -- the previous wave's *predicted* top
  candidates (generator ``elite``: 864 of 1,219 resolved).

All three are fixed at the source by :func:`..search.verify.lineage_anchor`, so
new rows are correct.  This tool is the ONE-OFF pass over the rows already
written.

What is and is not recoverable
------------------------------
A dangling parent can be repaired only if the parent BOARD exists in the store
under a different key -- e.g. a cross-cell donor whose row lives under its own
``case_pair`` while the child re-keyed it into the child's cell.  That is a real
failure mode of this column's shape, so the tool tests for it first: it indexes
every store row by :func:`..tools.backfill_fxy.digest_of_packed` (the vendor
``Pattern.digest``, the pattern-only 16-hex key that is INVARIANT across cells)
and re-derives the true id.  It only accepts a match whose row is the parent's
true cell -- never a re-key of the child's own cell, which would mint a second
phantom.

Measured on this store 2026-08-29, that path recovers **zero** rows: the dangling
parents are not store boards under any key.  Their patterns are nowhere on disk
either (checked against every ``runs/**/labels.jsonl``, ``waves/**/selection.json``,
``data/produce/ledger.jsonl`` and ``data/campaigns/**``: the ids appear ONLY as
``parent_record_id`` references, never once as a ``record_id``).  The boards were
never evaluated, so there is nothing to point at.

For those the honest repair is to NULL the column -- ``--null-phantom``.  A null
says "this row has no recorded lineage", which is true; the current value says
"this row descends from board X", which is false and which
``mine_policy_corpus.build_steps`` must (and does) drop anyway.  Nulling loses no
information and stops the store asserting something untrue.  It is opt-in and
never the default.

Contract (identical to :mod:`.backfill_flatness`)
-------------------------------------------------
* **Dry-run by default** -- ``--apply`` is required to write anything.
* **Backup first** -- ``--apply`` copies ``records.parquet`` to
  ``records.parquet.bak_pre_parentid_<YYYYMMDD>`` before the write, and refuses
  to clobber an existing backup of the same name.
* **Atomic + order-preserving** -- the edit is applied to a FRESH read keyed by
  ``record_id`` and written through the store's own :func:`..data.store._atomic_write`
  and :func:`..data.store.frame_to_table`, so row order is byte-for-byte
  preserved and a concurrently appended row survives.
* **Idempotent** -- a second run finds nothing to change and writes no file.
* **Never destructive** -- a resolvable parent is never nulled, and a row whose
  parent already resolves is never touched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

from ..data.store import (
    RECORDS_NAME, StoreReader, _atomic_write, ensure_schema_columns, frame_to_table,
)
from ..safelog import configure_stdio
from .backfill_fxy import digest_of_packed


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
@dataclass
class CampaignReport:
    """Per-campaign resolution counts, before and after the repair."""

    campaign: str
    rows: int = 0
    non_null: int = 0
    resolved: int = 0            # already fine
    repaired: int = 0            # re-derived from the pattern digest
    phantom: int = 0             # unrecoverable: parent board never existed
    generators: dict[str, int] = field(default_factory=dict)

    @property
    def resolved_frac(self) -> float:
        return self.resolved / self.non_null if self.non_null else float("nan")

    @property
    def after_frac(self) -> float:
        n = self.non_null
        return (self.resolved + self.repaired) / n if n else float("nan")


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
class _BoardIndex:
    """Every distinct store BOARD, probeable by "what id would cell C give it?".

    Two indexes over the same rows:

    * ``by_digest`` -- :func:`~.backfill_fxy.digest_of_packed` (the vendor
      ``Pattern.digest``, ``sha256(canonical)[:16]``) -> the rows carrying that
      board in ANY cell.  The digest is pattern-ONLY, so it is exactly the key
      that survives a cell change; ``record_id`` cannot answer "is this board in
      the store under a different ``case_pair`` / ``library_id``?" because it
      hashes the cell in.
    * ``rekey(lib, pair)`` -- a lazily built, cached
      ``{compute_record_id(board, lib, pair): digest}`` for ONE child cell.  The
      probe direction has to be this way round: a dangling id is a hash with no
      invertible preimage, so the only test available is to re-key every known
      board into the child's cell and look for the collision.  Built per cell
      (a few dozen) rather than per dangling parent (a few thousand).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        cols = [c for c in ("record_id", "pattern", "case_pair", "library_id",
                            "campaign") if c in df.columns]
        self.by_digest: dict[str, list[dict]] = {}
        for row in df[cols].to_dict("records"):
            packed = row.get("pattern")
            if not isinstance(packed, str) or not packed:
                continue
            self.by_digest.setdefault(digest_of_packed(packed), []).append(row)
        # one representative packed pattern per distinct board
        self._boards = [(d, rows[0]["pattern"]) for d, rows in self.by_digest.items()]
        self._rekey: dict[tuple[str, str], dict[str, str]] = {}

    def rekey(self, lib: str, pair: str) -> dict[str, str]:
        from ..data.schema import compute_record_id
        from ..search.verify import PRODUCE_DECK_KNOBS

        key = (lib, pair)
        cached = self._rekey.get(key)
        if cached is None:
            cached = {
                compute_record_id(packed, lib, pair, PRODUCE_DECK_KNOBS): digest
                for digest, packed in self._boards
            }
            self._rekey[key] = cached
        return cached


def scan(
    df: pd.DataFrame, *, campaigns: list[str] | None, min_rows: int
) -> tuple[dict[str, CampaignReport], dict[str, str | None]]:
    """Classify every non-null ``parent_record_id``; return per-campaign counts
    and the ``{record_id: new_parent_or_None}`` edit map (only rows that change).

    ``new_parent`` is a re-derived TRUE id where one exists, else ``None`` (the
    phantom case -- the caller decides whether to apply the null).
    """

    ids = set(df["record_id"].dropna().astype(str))
    index = _BoardIndex(df)
    reports: dict[str, CampaignReport] = {}
    edits: dict[str, str | None] = {}

    camp_col = df["campaign"].fillna("<none>").astype(str)
    sizes = camp_col.value_counts()
    wanted = {
        c for c, n in sizes.items()
        if n >= min_rows and (campaigns is None or c in campaigns)
    }
    if campaigns is not None:
        wanted |= {c for c in campaigns if c in set(sizes.index)}

    sub = df[camp_col.isin(wanted)]
    for camp, g in sub.groupby(camp_col.loc[sub.index]):
        rep = CampaignReport(campaign=str(camp), rows=len(g))
        for row in g.to_dict("records"):
            parent = row.get("parent_record_id")
            if not isinstance(parent, str) or not parent:
                continue
            rep.non_null += 1
            if parent in ids:
                rep.resolved += 1
                continue
            gen = str(row.get("generator") or "<none>")
            rep.generators[gen] = rep.generators.get(gen, 0) + 1
            new = _rederive(parent, row, index, ids)
            if new is not None:
                rep.repaired += 1
            else:
                rep.phantom += 1
            edits[str(row["record_id"])] = new
        reports[str(camp)] = rep
    return reports, edits


def _rederive(parent: str, child: dict, index: _BoardIndex, ids: set[str]) -> str | None:
    """The TRUE store ``record_id`` of ``child``'s parent board, or ``None``.

    The dangling id is a hash, so the parent's *pattern* cannot be read out of
    it -- there is no preimage to invert.  What CAN be tested is the only shape
    in which a dangling id coexists with a present board: the child re-keyed a
    real parent into its OWN cell.  If so, re-keying some known board into the
    child's cell reproduces ``parent`` exactly, and that board's own row id is
    the answer.  Accepting only a row whose id differs from ``parent`` is what
    keeps this from re-minting the same phantom.
    """

    lib = child.get("library_id")
    pair = child.get("case_pair")
    if not isinstance(lib, str) or not isinstance(pair, str):
        return None
    digest = index.rekey(lib, pair).get(parent)
    if digest is None:
        return None
    for r in index.by_digest.get(digest, ()):
        rid = str(r["record_id"])
        if rid in ids and rid != parent:
            return rid
    return None


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def apply_edits(path: Path, edits: dict[str, str | None]) -> int:
    """Write ``edits`` into ``records.parquet`` in place.  Returns rows changed.

    Re-reads the file so a row appended between the scan and the write survives,
    keys by ``record_id`` (never by position), and goes out through the store's
    own atomic writer.
    """

    if not edits:
        return 0
    current = ensure_schema_columns(pd.read_parquet(path))
    key = current["record_id"].astype(str)
    mapped = key.map(edits)
    touched = key.isin(edits.keys())
    changed = touched & (current["parent_record_id"].astype("object") != mapped)
    n = int(changed.sum())
    if not n:
        return 0
    current.loc[changed, "parent_record_id"] = mapped[changed]
    table = frame_to_table(current)
    _atomic_write(path, lambda p: pq.write_table(table, p))
    return n


def backup(path: Path) -> Path:
    """Copy ``records.parquet`` aside before any write; never clobber."""
    dest = path.with_name(f"{path.name}.bak_pre_parentid_{time.strftime('%Y%m%d')}")
    if dest.exists():
        raise FileExistsError(
            f"backup already exists: {dest} -- move or delete it before re-running "
            "with --apply (refusing to overwrite a prior safety copy)")
    shutil.copy2(path, dest)
    return dest


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def _print(reports: dict[str, CampaignReport], edits: dict[str, str | None],
           *, null_phantom: bool) -> None:
    rows = sorted(reports.values(), key=lambda r: (r.resolved_frac, r.campaign))
    print(f"{'campaign':46s} {'rows':>5s} {'par':>5s} {'ok':>5s} {'fix':>5s} "
          f"{'phantom':>7s} {'before':>7s} {'after':>7s}")
    print("-" * 96)
    for r in rows:
        print(f"{r.campaign:46s} {r.rows:5d} {r.non_null:5d} {r.resolved:5d} "
              f"{r.repaired:5d} {r.phantom:7d} "
              f"{r.resolved_frac:6.1%} {r.after_frac:6.1%}")
    tot_par = sum(r.non_null for r in rows)
    tot_ok = sum(r.resolved for r in rows)
    tot_fix = sum(r.repaired for r in rows)
    tot_ph = sum(r.phantom for r in rows)
    print("-" * 96)
    print(f"{'TOTAL':46s} {sum(r.rows for r in rows):5d} {tot_par:5d} {tot_ok:5d} "
          f"{tot_fix:5d} {tot_ph:7d} "
          f"{tot_ok / tot_par if tot_par else float('nan'):6.1%} "
          f"{(tot_ok + tot_fix) / tot_par if tot_par else float('nan'):6.1%}")

    gens: dict[str, int] = {}
    for r in rows:
        for g, n in r.generators.items():
            gens[g] = gens.get(g, 0) + n
    if gens:
        print("\nunresolved by generator: " +
              ", ".join(f"{g}={n}" for g, n in sorted(gens.items(), key=lambda t: -t[1])))
    n_null = sum(1 for v in edits.values() if v is None)
    print(f"\nedits staged: {tot_fix} re-derived"
          + (f", {n_null} nulled (--null-phantom)" if null_phantom
             else f"; {tot_ph} phantom left UNTOUCHED "
                  "(pass --null-phantom to clear them)"))


def main(argv: list[str] | None = None) -> int:
    # stdio is REDIRECTED to a log file by every launcher, so Windows gives it
    # the ANSI codepage; a non-ASCII banner then raises UnicodeEncodeError and
    # kills the run (incident 2026-08-30).  This module has its own __main__,
    # so it cannot rely on lpopt.cli.main's call.
    configure_stdio()
    ap = argparse.ArgumentParser(
        prog="python -m lpopt.tools.repair_parent_ids",
        description="Repair dangling parent_record_id foreign keys in the store.")
    ap.add_argument("--store-dir", default="data/store", type=Path)
    ap.add_argument("--campaign", action="append", default=None,
                    help="limit to this campaign (repeatable); default: every "
                         "campaign with >= --min-rows rows")
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--null-phantom", action="store_true",
                    help="also NULL the parents that provably never existed "
                         "(no store board under any key, no pattern on disk)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default: dry-run, no file touched)")
    args = ap.parse_args(argv)

    path = Path(args.store_dir) / RECORDS_NAME
    if not path.exists():
        print(f"no store at {path}", file=sys.stderr)
        return 2
    df = StoreReader(args.store_dir).records
    reports, edits = scan(df, campaigns=args.campaign, min_rows=args.min_rows)
    if not args.null_phantom:
        edits = {k: v for k, v in edits.items() if v is not None}
    _print(reports, edits, null_phantom=args.null_phantom)

    if not args.apply:
        print("\nDRY RUN — nothing written.  Re-run with --apply to write.")
        return 0
    if not edits:
        print("\nnothing to change; no file written.")
        return 0
    dest = backup(path)
    print(f"\nbackup: {dest}")
    n = apply_edits(path, edits)
    print(f"wrote {n} repaired rows to {path}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
