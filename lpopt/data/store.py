"""Unified store I/O: ``records.parquet`` + ``maps.npz`` (plan 4.2).

* :class:`StoreWriter` — append/dedup by ``record_id``, atomic parquet write via
  a same-directory temp file + ``os.replace``, and a companion ``maps.npz`` of
  float16 EDIT5 map stacks keyed by ``maps_key == record_id``.
* :class:`StoreReader` — load ``records.parquet`` and lazily fetch map stacks.

Atomicity: every on-disk artefact is written to ``<name>.tmp-<pid>`` first and
then ``os.replace``-d onto the final path.  A failure mid-write removes the temp
file and leaves any pre-existing final file untouched (no partial store).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import LATE_COLUMNS, PARQUET_SCHEMA, SCHEMA_COLUMNS, CanonicalRecord

RECORDS_NAME = "records.parquet"
MAPS_NAME = "maps.npz"


# --------------------------------------------------------------------------- #
# atomic write primitive
# --------------------------------------------------------------------------- #
#: Bounded retry for the final rename.  On Windows ``os.replace`` fails with
#: ``PermissionError`` (WinError 5/32) while ANY process holds the destination
#: open — an analysis script doing ``np.load("maps.npz")``, a virus scanner, or
#: the indexer.  That is transient, but it used to propagate out of
#: ``write_maps`` and kill an entire multi-hour production campaign mid-wave
#: (forensic 20260725: the 104 coverage-fill died on wave 3 exactly this way).
#: The data is already safely materialized in the temp file at that point, so
#: retrying the rename is pure upside.
#: 30 attempts with backoff capped at 5 s ~= 135 s of total patience.  A reader
#: holding maps.npz (an A/B watcher polling every 5 min, an analysis script, the
#: virus scanner) can hold it for many seconds; 8 attempts / ~30 s was measured
#: to be too short (forensic 20260725: the 104 fill died twice).  Waiting two
#: minutes is always cheaper than losing a multi-hour campaign.
_REPLACE_ATTEMPTS = 30
_REPLACE_BACKOFF_S = 0.25
_REPLACE_BACKOFF_CAP_S = 5.0


def _atomic_write(final: Path, writer: Callable[[Path], None]) -> None:
    """Run ``writer(tmp)`` then ``os.replace(tmp, final)``; clean up on failure.

    ``writer`` must fully materialize the file at the temp path.  If it raises,
    the temp file is removed and ``final`` is left exactly as it was.

    The rename is retried with backoff on :class:`PermissionError` (see
    :data:`_REPLACE_ATTEMPTS`) so a transient Windows file lock on ``final``
    cannot destroy a production run; the last failure is re-raised if every
    attempt loses.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f"{final.name}.tmp-{os.getpid()}")
    try:
        writer(tmp)
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, final)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(min(_REPLACE_BACKOFF_S * (2 ** attempt),
                               _REPLACE_BACKOFF_CAP_S))
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# --------------------------------------------------------------------------- #
# record <-> table
# --------------------------------------------------------------------------- #
def records_to_frame(records: Iterable[CanonicalRecord]) -> pd.DataFrame:
    """Build a column-ordered DataFrame from :class:`CanonicalRecord`s."""
    rows = [r.to_record() for r in records]
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


def ensure_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing :data:`~.schema.LATE_COLUMNS` as all-null float columns.

    A ``records.parquet`` (or a multi-PC kit) written before a tail column landed
    genuinely does not have it.  That is a legitimate store state, not corruption,
    so every read boundary passes the frame through here instead of raising a
    ``KeyError`` on ``df[SCHEMA_COLUMNS]``.  Only :data:`~.schema.LATE_COLUMNS`
    are auto-filled: a frame missing one of the 36 frozen columns is broken and
    must still fail loudly.
    """
    missing = [c for c in LATE_COLUMNS if c not in df.columns]
    if not missing:
        return df
    out = df.copy()
    for name in missing:
        out[name] = np.full(len(out), np.nan, dtype=np.float64)
    return out


def frame_to_table(df: pd.DataFrame) -> pa.Table:
    """Coerce a records DataFrame to the canonical :data:`PARQUET_SCHEMA`."""
    return pa.Table.from_pandas(
        ensure_schema_columns(df)[SCHEMA_COLUMNS],
        schema=PARQUET_SCHEMA,
        preserve_index=False,
    )


def _quality_rank(df: pd.DataFrame) -> np.ndarray:
    """Per-row "information quality" rank for the UPSERT dedup (higher = better).

    A converged label is always the best evidence for a ``record_id``; among rows
    of equal convergence a *valid* row (an honest non-convergence) outranks an
    *invalid* one (a ``non_finite_flux`` / harness ``error`` with all-None
    targets); and among rows equal on BOTH, a row carrying the harvested flatness
    columns outranks one that does not.  Encoded as
    ``converged*4 + valid*2 + has_flatness`` so convergence dominates:

    * ``7`` converged & valid & mapped   (the richest label; never lose it)
    * ``6`` converged & valid            (the label we must never lose)
    * ``4``/``5`` converged & invalid    (defensive; ``outcome_to_record`` never emits it)
    * ``2``/``3`` non-converged & valid
    * ``0``/``1`` invalid (error / non_finite_flux)

    The flatness bit is the LOW-ORDER term and it decides only exact ties, where
    the previous "keep the last write" rule let a re-write of the same
    ``record_id`` that carried no map silently null out an already-harvested
    ``node_peak`` / ``map_cov`` — the labels the flatness-first objective is
    defined on (program §1.3).  Ties on all three bits still keep the last
    occurrence, so an equal-quality write can still refresh auxiliary fields.
    """
    conv = df["converged"].fillna(False).astype(bool).to_numpy().astype(np.int8)
    valid = df["valid"].fillna(False).astype(bool).to_numpy().astype(np.int8)
    flat = np.zeros(len(df), dtype=np.int8)
    for column in ("node_peak", "map_cov"):
        if column in df.columns:
            present = pd.to_numeric(df[column], errors="coerce").notna()
            flat |= present.to_numpy().astype(np.int8)
    return conv * 4 + valid * 2 + flat


def dedup_upsert(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate ``record_id`` rows keeping the highest-quality one.

    UPSERT semantics (plan 4.2; store is the authoritative record of an LP
    evaluation): for each ``record_id`` the surviving row is the one with the
    greatest :func:`_quality_rank`.  A strictly better incoming row therefore
    REPLACES the stored one (e.g. a converged retry upgrades an earlier
    non-converged/failed row, and a mapped row upgrades an unmapped one); a
    strictly worse one is DISCARDED so a stale or racing write can NEVER downgrade
    a converged label or drop its harvested flatness columns.  Ties (equal
    quality) keep the last occurrence — the pre-existing store behaviour, which
    also lets a later equal-quality write refresh auxiliary fields.

    Rows are returned in stable arrival order (existing rows first, then new).
    """
    if df.empty:
        return df.reset_index(drop=True)
    work = df.reset_index(drop=True)
    rank = _quality_rank(work)
    order = np.arange(len(work), dtype=np.int64)
    # Best row per record_id = max (rank, arrival order): sort ascending and keep
    # the last occurrence, so the highest rank wins and equal ranks keep the
    # latest write.  Collect the winners' original positions and restore order.
    keyed = pd.DataFrame(
        {"rid": work["record_id"].to_numpy(), "rank": rank, "order": order}
    )
    keyed = keyed.sort_values(["rank", "order"], kind="stable")
    # np.sort (copy) — .to_numpy() may hand back a read-only view, on which the
    # in-place .sort() raises "sort array is read-only" (numpy 2.x)
    winners = np.sort(keyed.drop_duplicates("rid", keep="last")["order"].to_numpy())
    return work.iloc[winners].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class StoreWriter:
    """Append/dedup writer for ``records.parquet`` + ``maps.npz``."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.records_path = self.store_dir / RECORDS_NAME
        self.maps_path = self.store_dir / MAPS_NAME

    # -- records ------------------------------------------------------------ #
    def write_records(
        self,
        records: Sequence[CanonicalRecord] | pd.DataFrame,
        *,
        append: bool = True,
    ) -> dict[str, int]:
        """Write records, UPSERT-deduplicating by ``record_id``.

        On a ``record_id`` collision the higher-quality row wins (:func:`dedup_upsert`):
        a converged label upgrades an earlier non-converged/failed one and a
        strictly worse write can never downgrade it.  Returns
        ``{"new": n_incoming, "total": n_after_dedup}``.
        """
        new_df = records if isinstance(records, pd.DataFrame) else records_to_frame(records)
        new_df = ensure_schema_columns(new_df)
        n_new = len(new_df)

        if append and self.records_path.exists():
            existing = ensure_schema_columns(pd.read_parquet(self.records_path))
            new_slice = new_df[SCHEMA_COLUMNS]
            if len(existing) and len(new_slice):
                # Align the incoming dtypes to the persisted store so an all-None
                # column in one frame (e.g. produce rows carry no cbc_boc) does
                # not trip the pandas>=2.1 "all-NA concat" FutureWarning; the
                # result is schema-normalized by frame_to_table regardless.
                new_slice = new_slice.astype(existing.dtypes.to_dict())
            # Existing rows FIRST so a strictly-better incoming row upgrades and an
            # equal-quality one wins the tie (last), while a worse one is dropped.
            combined = pd.concat([existing, new_slice], ignore_index=True)
            combined = dedup_upsert(combined)
        else:
            combined = dedup_upsert(new_df[SCHEMA_COLUMNS])

        table = frame_to_table(combined)
        _atomic_write(self.records_path, lambda p: pq.write_table(table, p))
        return {"new": n_new, "total": len(combined)}

    # -- maps --------------------------------------------------------------- #
    def write_maps(
        self,
        maps: Mapping[str, np.ndarray],
        *,
        append: bool = True,
    ) -> dict[str, int]:
        """Write EDIT5 map stacks (float16) keyed by ``record_id``.

        Returns ``{"new": n_incoming, "total": n_after_merge}``.
        """
        merged: dict[str, np.ndarray] = {}
        if append and self.maps_path.exists():
            with np.load(self.maps_path) as existing:
                for k in existing.files:
                    merged[k] = existing[k]
        for k, arr in maps.items():
            merged[k] = np.asarray(arr, dtype=np.float16)

        def _write(p: Path) -> None:
            np.savez_compressed(p, **merged)
            # np.savez_compressed appends ".npz" unless the name already ends in it.
            if not p.name.endswith(".npz") and p.with_suffix(p.suffix + ".npz").exists():
                os.replace(p.with_suffix(p.suffix + ".npz"), p)

        _atomic_write(self.maps_path, _write)
        return {"new": len(maps), "total": len(merged)}


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
class StoreReader:
    """Read ``records.parquet`` and lazily fetch ``maps.npz`` stacks."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.records_path = self.store_dir / RECORDS_NAME
        self.maps_path = self.store_dir / MAPS_NAME
        self._records: pd.DataFrame | None = None
        self._maps: np.lib.npyio.NpzFile | None = None

    @property
    def records(self) -> pd.DataFrame:
        if self._records is None:
            if not self.records_path.exists():
                raise FileNotFoundError(f"no records store at {self.records_path}")
            # ensure_schema_columns: a store written before a tail column landed
            # still reads back with the full column set (nulls), so consumers can
            # index it unconditionally.
            self._records = ensure_schema_columns(pd.read_parquet(self.records_path))
        return self._records

    @property
    def has_maps(self) -> bool:
        return self.maps_path.exists()

    def _npz(self) -> dict[str, np.ndarray] | None:
        """Materialize maps.npz into memory, RELEASING the file handle immediately.

        ``np.load`` on an .npz is lazy: it keeps the zip handle open until closed.
        On Windows an open read handle makes ``os.replace`` fail, so a long-lived
        reader (an A/B watcher polling every 5 min, a scoring job, an analysis
        script) would block the producer's atomic ``maps.npz`` write and — before
        this — kill the whole campaign (forensic 20260725, the 104 fill died twice
        exactly this way).  Reading eagerly inside a ``with`` block costs one
        decompression but holds the file for milliseconds instead of minutes.
        """
        if self._maps is None and self.maps_path.exists():
            with np.load(self.maps_path) as z:
                self._maps = {k: z[k] for k in z.files}
        return self._maps

    def maps_keys(self) -> set[str]:
        npz = self._npz()
        return set(npz) if npz is not None else set()

    def maps(self, record_id: str) -> np.ndarray | None:
        """Return the ``(4, 9, 9)`` float16 map stack for ``record_id`` (or None)."""
        npz = self._npz()
        if npz is None:
            return None
        return npz.get(record_id)

    def close(self) -> None:
        # The handle is already released by ``_npz``; this just drops the cache.
        self._maps = None
