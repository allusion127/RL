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
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .fuel_types import FuelLibrary, core_enrichment_split
from .schema import LATE_COLUMNS, PARQUET_SCHEMA, SCHEMA_COLUMNS, CanonicalRecord, unpack_pattern

RECORDS_NAME = "records.parquet"
MAPS_NAME = "maps.npz"
FUEL_TYPES_NAME = "fuel_types.parquet"


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
    targets); and among rows equal on BOTH, a row carrying a harvested label
    column outranks one that does not.  Encoded as
    ``converged*8 + valid*4 + has_flatness*2 + has_fxy`` so convergence dominates:

    * ``15`` converged & valid & mapped & F_xy-labelled (the richest; never lose it)
    * ``12`` converged & valid                (the label we must never lose)
    * ``8``-``11`` converged & invalid  (defensive; ``outcome_to_record`` never emits it)
    * ``4``-``7`` non-converged & valid
    * ``0``-``3`` invalid (error / non_finite_flux)

    The two label bits are the LOW-ORDER terms and they decide only exact ties,
    where the previous "keep the last write" rule let a re-write of the same
    ``record_id`` that carried no map silently null out an already-harvested
    ``node_peak`` / ``map_cov`` — the labels the flatness-first objective is
    defined on (program §1.3).  ``f_xy`` / ``f_xya`` get their OWN bit rather than
    joining the flatness OR: they come from a different file (``MAS_OUT``, not
    ``MAS_SUM``) and can be present when the maps are not, so folding them
    together would let a map-only row tie — and therefore overwrite — an
    F_xy-labelled one, which is the exact defect this rank exists to prevent
    (design 20260829 §3.3).  Rows that tie on all four bits are settled by
    :func:`dedup_upsert`'s FIRST-wins rule, i.e. the incumbent survives.
    """
    def _flag(column: str) -> np.ndarray:
        """Null-safe 0/1 flag: every null shape (None, NaN, ``pd.NA``) reads False.

        Masking by hand rather than ``.fillna(False)``: on an OBJECT-dtype column
        (a frame built straight from records, where the nulls are ``None`` /
        ``pd.NA`` rather than a parquet bool column) ``fillna`` silently downcasts
        and emits the pandas>=2.2 FutureWarning.  ``multi_pc._truthy`` is the
        row-at-a-time twin of this and must keep matching it.
        """
        col = df[column]
        values = col.to_numpy(copy=True)
        values[~col.notna().to_numpy()] = False
        return values.astype(bool).astype(np.int8)

    conv = _flag("converged")
    valid = _flag("valid")

    def _any_present(columns: tuple[str, ...]) -> np.ndarray:
        bit = np.zeros(len(df), dtype=np.int8)
        for column in columns:
            if column in df.columns:
                present = pd.to_numeric(df[column], errors="coerce").notna()
                bit |= present.to_numpy().astype(np.int8)
        return bit

    flat = _any_present(("node_peak", "map_cov"))
    fxy = _any_present(("f_xy", "f_xya"))
    return conv * 8 + valid * 4 + flat * 2 + fxy


def dedup_upsert(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate ``record_id`` rows keeping the highest-quality one.

    UPSERT semantics (plan 4.2; store is the authoritative record of an LP
    evaluation): for each ``record_id`` the surviving row is the one with the
    greatest :func:`_quality_rank`.  A strictly better incoming row therefore
    REPLACES the stored one (e.g. a converged retry upgrades an earlier
    non-converged/failed row, and a mapped row upgrades an unmapped one); a
    strictly worse one is DISCARDED so a stale or racing write can NEVER downgrade
    a converged label or drop its harvested flatness columns.

    TIES KEEP THE FIRST OCCURRENCE (incumbent wins).  Callers concatenate
    ``[existing, incoming]``, so on equal quality the row ALREADY IN THE STORE
    survives: local truth is only ever replaced by strictly higher-quality
    evidence.  This is the fix for the merge clobber measured 20260829 — the rank
    covers ``converged`` / ``valid`` / the label bits and NOTHING else, so under
    the old "ties keep the last write" rule every other column (``e_core``,
    ``e_split``, ``parent_record_id``, …) was silently taken from the incoming
    row on a tie.  Merging a remote kit whose store was a stale mirror of the
    local one would therefore have reverted 397 corrected ``e_core`` values and
    un-nulled 1,203 repaired ``parent_record_id``s while the CLI classified every
    one of them as a harmless "duplicate (kept)".  An equal-quality write can no
    longer refresh auxiliary fields; a caller that genuinely wants to overwrite a
    stored row must write a strictly better one (or repair the store in place,
    the way :func:`backfill_e_core` does).

    Rows are returned in stable arrival order (existing rows first, then new).
    """
    if df.empty:
        return df.reset_index(drop=True)
    work = df.reset_index(drop=True)
    rank = _quality_rank(work)
    order = np.arange(len(work), dtype=np.int64)
    # Best row per record_id = max rank, min arrival order.  Sort by rank
    # ASCENDING and order DESCENDING and keep the last occurrence: the highest
    # rank wins and equal ranks keep the EARLIEST (incumbent) write.  Collect the
    # winners' original positions and restore arrival order.
    keyed = pd.DataFrame(
        {"rid": work["record_id"].to_numpy(), "rank": rank, "order": order}
    )
    keyed = keyed.sort_values(["rank", "order"], ascending=[True, False],
                              kind="stable")
    # np.sort (copy) — .to_numpy() may hand back a read-only view, on which the
    # in-place .sort() raises "sort array is read-only" (numpy 2.x)
    winners = np.sort(keyed.drop_duplicates("rid", keep="last")["order"].to_numpy())
    return work.iloc[winners].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# derived enrichment columns (e_core / e_split)
# --------------------------------------------------------------------------- #
#: ``e_core`` / ``e_split`` are DERIVED columns, not free-form metadata: both are
#: a pure function of ``(pattern, library_id)`` through the single shared recipe
#: :func:`~.fuel_types.core_enrichment_split`.  Dataset-A/B extraction has always
#: filled them that way, and inference reconstructs them the same way from a
#: served pattern (``featurize`` / ``cell_calibrate``), which is what makes the
#: train/serve conditioning parity contract hold.
#:
#: The produce/campaign write path, however, used to pass the CaseContext's
#: *nominal* enrichment straight through ``outcome_to_record`` — the planned
#: 50/50 (or 1/N) split value of the case, CONSTANT across a whole campaign and
#: paired with ``e_split=None``.  A realized pattern almost never lands on the
#: nominal split, so those rows carried an ``e_core`` describing a core that was
#: never actually loaded (measured drift up to 0.068 w/o, ~1.4 e_core bins).
#: Normalizing here — at the one choke point every writer funnels through —
#: makes the column self-consistent no matter which caller produced the row.
def derive_enrichment(
    df: pd.DataFrame, fuel_library: "FuelLibrary"
) -> tuple[pd.Series, pd.Series]:
    """Recompute ``(e_core, e_split)`` from each row's ``(pattern, library_id)``.

    Returns two float Series aligned to ``df.index``; an entry is ``NaN`` when the
    recipe cannot resolve the row (unknown library, unresolvable batch id, or a
    fed type with no enrichment) — exactly the ``(None, None)`` fallback
    :func:`~.fuel_types.core_enrichment_split` gives extraction and inference, so
    a caller-supplied value is kept rather than destroyed.

    Results are memoized per ``(library_id, pattern)`` so a produce wave (many
    rows, few distinct patterns) and a whole-store backfill both stay cheap.
    """
    cache: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    cores: list[float] = []
    splits: list[float] = []
    for lib, packed in zip(
        df["library_id"].astype(str), df["pattern"].astype(str), strict=True
    ):
        key = (lib, packed)
        hit = cache.get(key)
        if hit is None:
            try:
                feed = unpack_pattern(packed).batch_feed()
                hit = core_enrichment_split(fuel_library, lib, feed)
            except Exception:
                # A malformed/foreign pattern is a legitimate store state for a
                # synthetic or partial row; never let it fail a records write.
                hit = (None, None)
            cache[key] = hit
        cores.append(np.nan if hit[0] is None else float(hit[0]))
        splits.append(np.nan if hit[1] is None else float(hit[1]))
    return (
        pd.Series(cores, index=df.index, dtype="float64"),
        pd.Series(splits, index=df.index, dtype="float64"),
    )


def _load_fuel_library(store_dir: Path) -> "FuelLibrary | None":
    """Best-effort ``fuel_types.parquet`` load from a store directory."""
    path = Path(store_dir) / FUEL_TYPES_NAME
    if not path.is_file():
        return None
    try:
        return FuelLibrary.from_parquet(path)
    except Exception:
        return None


def _normalize_enrichment(
    df: pd.DataFrame, fuel_library: "FuelLibrary | None"
) -> pd.DataFrame:
    """Overwrite ``e_core``/``e_split`` with the pattern-derived values.

    A row the recipe cannot resolve keeps whatever the caller supplied, so this is
    a strict improvement: it can fill a null and correct a nominal, never blank an
    otherwise-good value.  Returns ``df`` unchanged (same object) when there is
    nothing to derive.
    """
    if fuel_library is None or df.empty:
        return df
    core, split = derive_enrichment(df, fuel_library)
    have = core.notna()
    if not have.any():
        return df
    out = df.copy()
    out.loc[have, "e_core"] = core[have]
    out.loc[have, "e_split"] = split[have]
    return out


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class StoreWriter:
    """Append/dedup writer for ``records.parquet`` + ``maps.npz``."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.records_path = self.store_dir / RECORDS_NAME
        self.maps_path = self.store_dir / MAPS_NAME
        self._fuel: FuelLibrary | None = None
        self._fuel_loaded = False

    @property
    def fuel_library(self) -> "FuelLibrary | None":
        """``fuel_types.parquet`` of this store, loaded once (``None`` if absent)."""
        if not self._fuel_loaded:
            self._fuel = _load_fuel_library(self.store_dir)
            self._fuel_loaded = True
        return self._fuel

    # -- records ------------------------------------------------------------ #
    def write_records(
        self,
        records: Sequence[CanonicalRecord] | pd.DataFrame,
        *,
        append: bool = True,
        derive_enrichment_columns: bool = True,
    ) -> dict[str, int]:
        """Write records, UPSERT-deduplicating by ``record_id``.

        On a ``record_id`` collision the higher-quality row wins (:func:`dedup_upsert`):
        a converged label upgrades an earlier non-converged/failed one, and a
        strictly worse OR EQUAL-quality write can never replace it.  Returns
        ``{"new": n_incoming, "total": n_after_dedup}``.

        ``derive_enrichment_columns`` (default on) re-derives ``e_core``/``e_split``
        from each incoming row's own ``(pattern, library_id)`` via
        :func:`derive_enrichment`, so a caller that passes its case's *nominal*
        enrichment (the produce/campaign path did, with ``e_split=None``) cannot
        stamp a value that does not describe the pattern actually written.  Rows
        the recipe cannot resolve keep the caller's value; pass ``False`` only to
        write a frame verbatim (migrations that must be byte-preserving).
        """
        new_df = records if isinstance(records, pd.DataFrame) else records_to_frame(records)
        new_df = ensure_schema_columns(new_df)
        if derive_enrichment_columns:
            new_df = _normalize_enrichment(new_df, self.fuel_library)
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
            # Existing rows FIRST so a strictly-better incoming row upgrades while
            # an equal-quality one LOSES the tie (dedup_upsert keeps the first
            # occurrence) and a worse one is dropped: local truth is authoritative.
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


# --------------------------------------------------------------------------- #
# e_core / e_split backfill (one-shot repair of nominally-stamped rows)
# --------------------------------------------------------------------------- #
#: Default drift threshold [w/o] for "this row's e_core does not describe its own
#: pattern".  0.005 is half the tightest curriculum ``e_core_band`` step and an
#: order of magnitude above the float noise of the recipe (measured max residual
#: on correctly-written rows: 2.7e-15).
ECORE_BACKFILL_TOL = 0.005


def backfill_e_core(
    store_dir: str | Path,
    *,
    dry_run: bool = True,
    tol: float = ECORE_BACKFILL_TOL,
    backup_suffix: str | None = None,
) -> dict[str, Any]:
    """Re-derive ``e_core``/``e_split`` for rows written with a NOMINAL value.

    DRY RUN BY DEFAULT — call with ``dry_run=False`` to write.

    Rows whose ``e_core`` came from the extractor (Dataset A/B) already agree with
    :func:`~.fuel_types.core_enrichment_split` to float precision and are left
    byte-identical.  What this repairs are the produce/campaign rows that carried
    the case's *nominal* enrichment (the planned 50/50 or 1/N split, constant
    across a campaign, always paired with a null ``e_split``) instead of the value
    implied by the pattern that was actually written; see :func:`derive_enrichment`.

    Returns a report dict::

        {"rows", "resolvable", "null_filled", "corrected", "unchanged",
         "unresolvable", "max_abs_drift", "by_campaign", "applied", "backup"}

    ``corrected`` counts rows whose non-null ``e_core`` moves by more than ``tol``;
    ``by_campaign`` breaks those down by ``(library_id, campaign)``.  A write is
    atomic (temp file + ``os.replace``) and takes a ``.bak_<suffix>`` copy of the
    previous ``records.parquet`` first unless ``backup_suffix`` is ``None``.
    """
    store_dir = Path(store_dir)
    records_path = store_dir / RECORDS_NAME
    if not records_path.is_file():
        raise FileNotFoundError(f"no records store at {records_path}")
    fuel = _load_fuel_library(store_dir)
    if fuel is None:
        raise FileNotFoundError(f"no fuel table at {store_dir / FUEL_TYPES_NAME}")

    df = ensure_schema_columns(pd.read_parquet(records_path))
    core, split = derive_enrichment(df, fuel)
    stored = pd.to_numeric(df["e_core"], errors="coerce")

    resolvable = core.notna()
    null_fill = resolvable & stored.isna()
    drift = (core - stored).abs()
    corrected = resolvable & stored.notna() & (drift > tol)
    touched = resolvable & (null_fill | (drift > tol))

    sub = df.loc[corrected]
    by_campaign = (
        sub.assign(_d=drift[corrected])
        .groupby([sub["library_id"].astype(str), sub["campaign"].astype(str)])
        .agg(n=("_d", "size"), max_abs_drift=("_d", "max"))
        .sort_values("n", ascending=False)
        .to_dict("index")
    )

    report: dict[str, Any] = {
        "rows": int(len(df)),
        "resolvable": int(resolvable.sum()),
        "unresolvable": int((~resolvable).sum()),
        "null_filled": int(null_fill.sum()),
        "corrected": int(corrected.sum()),
        "unchanged": int(len(df) - touched.sum()),
        "max_abs_drift": float(drift[resolvable & stored.notna()].max())
        if (resolvable & stored.notna()).any() else 0.0,
        "by_campaign": {f"{k[0]}|{k[1]}": v for k, v in by_campaign.items()},
        "applied": False,
        "backup": None,
    }
    if dry_run or not touched.any():
        return report

    if backup_suffix:
        backup = records_path.with_name(f"{records_path.name}.bak_{backup_suffix}")
        backup.write_bytes(records_path.read_bytes())
        report["backup"] = str(backup)

    out = df.copy()
    out.loc[resolvable, "e_core"] = core[resolvable]
    out.loc[resolvable, "e_split"] = split[resolvable]
    table = frame_to_table(out)
    _atomic_write(records_path, lambda p: pq.write_table(table, p))
    report["applied"] = True
    return report
