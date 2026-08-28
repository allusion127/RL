"""Backfill ``node_peak`` / ``map_cov`` for store rows that already carry a map.

``python -m lpopt.tools.backfill_flatness [--store-dir data/store] [--dry-run]``

The two columns are written at harvest time from now on
(:func:`..search.verify.outcome_to_record`), but ~30k rows were harvested before
the columns existed.  Their maps are still in ``maps.npz``, so the scalars are
recoverable exactly — this pass recomputes them with the ONE canonical definition
(:mod:`..data.flatness`) and writes them back.

Contract
--------
* **Idempotent** — a row whose stored value already equals the recomputed one is
  not touched, and a run that finds nothing to change writes NO file at all.  Run
  it as often as you like.
* **Atomic** — the write goes through the store's own
  :func:`..data.store._atomic_write` (temp file + ``os.replace`` with the bounded
  retry that keeps a Windows reader's file lock from destroying the store) and
  the store's own :func:`..data.store.frame_to_table` schema coercion.
* **Order-preserving** — the values are applied to a FRESH read of
  ``records.parquet`` keyed by ``record_id``, so the row order is byte-for-byte
  what it was and any row a producer appended between our scan and our write
  survives (it simply waits for the next run).  Going through
  ``write_records(append=True)`` instead would move every updated row to the tail
  of the store, which silently changes what ``df.head(n)`` means to every
  downstream consumer.
* **Never destructive** — a ``maps_key`` that is not in ``maps.npz`` (the 13
  dangling keys of program §9 P-5) is COUNTED and left null, never guessed.  A map
  that fails to parse is counted the same way.

Order note (program §11): run the P-5 dangling-key null pass BEFORE this one if
you want the dangling rows' ``maps_key`` cleared; this pass is correct either way
(it just reports them as dangling).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from ..data.flatness import record_flatness
from ..data.schema import LATE_COLUMNS
from ..data.store import (
    RECORDS_NAME, StoreReader, _atomic_write, ensure_schema_columns, frame_to_table,
)

#: Two stored values count as "already correct" within this tolerance.
#:
#: The comparison is NOT float64-vs-float64.  A harvest-time value was computed
#: by :func:`..search.verify.outcome_to_record` from the **float32** EDIT5 array,
#: while this pass recomputes it from the **float16** copy in ``maps.npz`` — so
#: the two agree only to the storage resolution, never to 1e-12.  The old 1e-12
#: atol therefore declared essentially every already-correct row stale and
#: rewrote ~30k rows on pure dtype noise, which is the opposite of the
#: "idempotent, writes nothing when nothing changed" contract above.
#:
#: One float16 ULP is ``2**-10`` RELATIVE (an 11-bit significand).  Measured over
#: 2,913 store maps, re-deriving both scalars from a float32 array that rounds to
#: the stored float16 moves ``node_peak`` by <= 4.6e-4 and ``map_cov`` by
#: <= 2.1e-4 relative — i.e. inside one ULP, with margin.  Anything OUTSIDE it is
#: a real disagreement (a stale value, a changed definition) and is rewritten.
EQUAL_RTOL = 2.0 ** -10
#: Absolute floor, so a scalar at/near zero (the CoV of a perfectly flat map)
#: does not demand infinite relative precision.
EQUAL_ATOL = 1.0e-9


@dataclass
class BackfillReport:
    """What one :func:`backfill` pass found and did."""

    store_dir: str
    n_rows: int = 0                 # rows in the store
    n_with_maps_key: int = 0        # rows claiming a harvested map
    n_populated: int = 0            # rows this pass WROTE a value into
    n_already: int = 0              # rows already carrying the correct value
    n_dangling: int = 0             # maps_key with no entry in maps.npz (P-5)
    n_unreadable: int = 0           # map present but unusable (left null)
    n_without_maps_key: int = 0     # rows with no map at all (left null)
    dry_run: bool = False
    wrote: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"store            {self.store_dir}",
            f"rows             {self.n_rows}",
            f"rows with map    {self.n_with_maps_key}",
            f"  populated      {self.n_populated}",
            f"  already ok     {self.n_already}",
            f"  dangling key   {self.n_dangling}",
            f"  unreadable     {self.n_unreadable}",
            f"rows without map {self.n_without_maps_key}",
            f"wrote            {self.wrote}{'  (dry-run)' if self.dry_run else ''}",
        ]
        return "\n".join(lines)


def _is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / ``pd.NA`` / ``NaT`` — the store's null shapes."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):     # pragma: no cover - non-scalar
        return False


def _needs_write(stored: Any, computed: float | None) -> bool:
    """True when the stored cell does not already hold ``computed``.

    "Already holds" is judged at float16 storage resolution (:data:`EQUAL_RTOL`),
    because the stored value came from the float32 map and ``computed`` comes
    from its float16 copy.
    """
    have = not _is_missing(stored)
    if computed is None:
        return False                      # never overwrite a value with a null
    if not have:
        return True
    try:
        current = float(stored)
    except (TypeError, ValueError):        # pragma: no cover - non-numeric cell
        return True
    if not math.isfinite(current):
        return True
    return not math.isclose(current, computed,
                            rel_tol=EQUAL_RTOL, abs_tol=EQUAL_ATOL)


def backfill(store_dir: str | Path, *, dry_run: bool = False,
             log: Callable[[str], None] | None = None) -> BackfillReport:
    """Compute and persist the flatness columns for every row that has a map."""
    log = log or (lambda m: print(m, flush=True))
    reader = StoreReader(store_dir)
    # Read-only view: nothing below mutates ``df``, only the slice handed to the
    # writer, so the 50k-row frame is never copied whole.
    df = ensure_schema_columns(reader.records)
    rep = BackfillReport(store_dir=str(store_dir), n_rows=int(len(df)), dry_run=dry_run)

    keys = df["maps_key"] if "maps_key" in df.columns else pd.Series([None] * len(df))
    rids = df["record_id"].astype(str).to_numpy(dtype=object)
    stored_peak = df["node_peak"].to_numpy(dtype=object)
    stored_cov = df["map_cov"].to_numpy(dtype=object)

    new_peak: dict[str, float | None] = {}
    new_cov: dict[str, float | None] = {}

    for pos, key in enumerate(keys.to_numpy(dtype=object)):
        if _is_missing(key) or not str(key):
            rep.n_without_maps_key += 1
            continue
        rep.n_with_maps_key += 1
        arr = reader.maps(str(key))
        if arr is None:
            rep.n_dangling += 1
            continue
        peak, cov = record_flatness(arr)
        if peak is None and cov is None:
            rep.n_unreadable += 1
            continue
        if _needs_write(stored_peak[pos], peak) or _needs_write(stored_cov[pos], cov):
            new_peak[rids[pos]] = peak
            new_cov[rids[pos]] = cov
            rep.n_populated += 1
        else:
            rep.n_already += 1

    if not new_peak:
        log(f"[backfill_flatness] nothing to do "
            f"({rep.n_already} row(s) already correct); store untouched")
        return rep

    log(f"[backfill_flatness] {rep.n_populated} row(s) to populate "
        f"({rep.n_already} already correct, {rep.n_dangling} dangling key(s))")
    if dry_run:
        return rep

    # Re-read and patch IN PLACE by record_id: row order survives untouched and a
    # row appended since the scan is carried through (it just waits for the next
    # run).  Then the store's own atomic write.
    path = Path(store_dir) / RECORDS_NAME
    current = ensure_schema_columns(pd.read_parquet(path))
    rid_col = current["record_id"].astype(str)
    for column, values in (("node_peak", new_peak), ("map_cov", new_cov)):
        patched = rid_col.map(values)
        current[column] = patched.where(patched.notna(),
                                        pd.to_numeric(current[column], errors="coerce"))
    table = frame_to_table(current)
    _atomic_write(path, lambda p: pq.write_table(table, p))
    rep.wrote = True
    log(f"[backfill_flatness] wrote {rep.n_populated} value(s); "
        f"store now {len(current)} row(s)")
    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m lpopt.tools.backfill_flatness",
        description=f"backfill {', '.join(LATE_COLUMNS)} from maps.npz")
    ap.add_argument("--store-dir", default="data/store")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args(argv)
    rep = backfill(args.store_dir, dry_run=args.dry_run)
    print(rep.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EQUAL_ATOL", "EQUAL_RTOL", "BackfillReport", "backfill", "main"]
