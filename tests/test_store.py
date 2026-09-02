"""Store I/O: schema round-trip, dedup-by-record_id, atomic write, maps.npz."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.data.schema import CanonicalRecord, SCHEMA_COLUMNS, compute_record_id
from lpopt.data.store import StoreReader, StoreWriter, _atomic_write, records_to_frame
from lpopt.vendor.masterrl.domain import SLOTS


def _make_record(
    rid: str,
    cbc: float | None = 1500.0,
    cyclen: float = 680.0,
    *,
    converged: bool = True,
    valid: bool = True,
    failure: str = "",
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=rid,
        dataset="A",
        campaign="sa_2b_cache",
        stratum=None,
        generator=None,
        parent_record_id=None,
        case_pair="B1_C2",
        feed=121,
        n_batches=2,
        depth2_edges=0,
        e_core=5.4,
        e_split=0.1,
        library_id="260624",
        sym_class="rot61",
        pattern="F:B1:0|F:C2:0",
        f_r=1.9,
        f_q=2.4,
        cbc_max=cbc,
        cbc_boc=1480.0,
        cbc_kind="max" if cbc is not None else "boc_only",
        cyclen=cyclen,
        ao_abs=0.05,
        cycle_burnup=27.0,
        discharge_burnup=54.0,
        max_assembly_burnup=67.0,
        max_pin_burnup=71.0,
        eoc_ppm=10.0,
        delta_efpd=0.5,
        n_cycles=11.0,
        converged=converged,
        converged_at_cap=False,
        tolerance_margin=None,
        restart_provenance="mocha_native",
        valid=valid,
        failure=failure,
        maps_key=rid if cbc is not None else None,
    )


def _nonconverged(rid: str, **kw) -> CanonicalRecord:
    """An honest non-convergence: a valid, stored label but converged=False."""
    return _make_record(rid, converged=False, valid=True, failure="", **kw)


def _failed(rid: str, **kw) -> CanonicalRecord:
    """An invalid error label (e.g. non_finite_flux): converged=False, valid=False."""
    return _make_record(rid, converged=False, valid=False,
                        failure="non_finite_flux", **kw)


def test_schema_columns_cover_dataclass() -> None:
    rec = _make_record("abc")
    assert set(rec.to_record()) == set(SCHEMA_COLUMNS)


def test_write_and_read_records(tmp_path: Path) -> None:
    writer = StoreWriter(tmp_path)
    stats = writer.write_records([_make_record("r1"), _make_record("r2")], append=False)
    assert stats == {"new": 2, "total": 2}

    reader = StoreReader(tmp_path)
    df = reader.records
    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df) == 2
    assert df["feed"].tolist() == [121, 121]
    # nullable columns round-trip as NaN/None, not a crash.
    assert df["tolerance_margin"].isna().all()


def test_dedup_same_record_id(tmp_path: Path) -> None:
    """Two EQUAL-quality writes (both converged) collapse to one; the FIRST wins.

    Ties keep the incumbent: the row already in the store is local truth and is
    only ever displaced by strictly higher-quality evidence.
    """
    writer = StoreWriter(tmp_path)
    writer.write_records([_make_record("dup", cyclen=680.0)], append=False)
    stats = writer.write_records([_make_record("dup", cyclen=999.0)], append=True)
    assert stats["total"] == 1

    df = StoreReader(tmp_path).records
    assert len(df) == 1
    assert df.iloc[0]["cyclen"] == pytest.approx(680.0)   # equal quality -> stored row kept


# --------------------------------------------------------------------------- #
# TIE => THE EXISTING ROW SURVIVES (clobber defect, measured 20260829)
#
# The quality rank encodes converged / valid / the two label bits and NOTHING
# else, so two rows can tie and still disagree on every other column.  Under the
# old "ties keep the last write" rule the tie handed those columns to the
# INCOMING row: merging a remote kit whose store was a stale copy of the local
# one would have reverted 397 corrected ``e_core`` values and un-nulled 1,203
# repaired ``parent_record_id``s — all of them reported as "duplicates (kept)".
# --------------------------------------------------------------------------- #
def _corrected(rid: str, *, e_core: float, e_split, parent, **kw) -> CanonicalRecord:
    rec = _make_record(rid, **kw)
    rec.e_core = e_core
    rec.e_split = e_split
    rec.parent_record_id = parent
    return rec


def test_tie_keeps_the_existing_rows_corrected_columns(tmp_path: Path) -> None:
    """A tie must not let a stale row reinstate the values a repair removed."""
    writer = StoreWriter(tmp_path)
    # local truth: e_core corrected by the backfill, parent_record_id repaired to null
    writer.write_records(
        [_corrected("rid", e_core=5.4712, e_split=1.0, parent=None)], append=False)
    # the stale mirror: same record_id, same quality, pre-repair values
    stats = writer.write_records(
        [_corrected("rid", e_core=5.4, e_split=None, parent="bogus_parent")],
        append=True)
    assert stats["total"] == 1

    row = StoreReader(tmp_path).records.iloc[0]
    assert row["e_core"] == pytest.approx(5.4712)      # correction survives
    assert row["e_split"] == pytest.approx(1.0)
    assert pd.isna(row["parent_record_id"])            # repair survives


def test_tie_keeps_the_first_row_within_one_batch(tmp_path: Path) -> None:
    """The rule is positional, not per-call: first occurrence wins inside a batch."""
    stats = StoreWriter(tmp_path).write_records(
        [_make_record("rid", cyclen=680.0), _make_record("rid", cyclen=999.0)],
        append=False,
    )
    assert stats["total"] == 1
    assert StoreReader(tmp_path).records.iloc[0]["cyclen"] == pytest.approx(680.0)


def test_strictly_better_incoming_still_replaces_the_whole_row(tmp_path: Path) -> None:
    """Ties-keep-existing must NOT have blunted the upgrade path.

    (a) converged beats non-converged, (b) an f_xy-labelled row beats a null one,
    and in both cases the winner brings its OWN values for the columns outside
    the rank (the upgrade is a row replacement, not a column-wise merge).
    """
    # (a) converged over non-converged
    writer = StoreWriter(tmp_path)
    writer.write_records(
        [_corrected("rid", e_core=5.4, e_split=None, parent="old", converged=False)],
        append=False)
    writer.write_records(
        [_corrected("rid", e_core=5.9, e_split=0.5, parent="new")], append=True)
    row = StoreReader(tmp_path).records.iloc[0]
    assert bool(row["converged"]) is True
    assert row["e_core"] == pytest.approx(5.9)         # the upgrade's own values
    assert row["parent_record_id"] == "new"

    # (b) f_xy-filled over null, at otherwise identical quality
    b = tmp_path / "b"
    writer_b = StoreWriter(b)
    writer_b.write_records([_make_record("rid", cyclen=680.0)], append=False)
    labelled = {**_make_record("rid", cyclen=999.0).__dict__, "f_xy": 1.431}
    writer_b.write_records([CanonicalRecord(**labelled)], append=True)
    got = StoreReader(b).records.iloc[0]
    assert float(got["f_xy"]) == pytest.approx(1.431)
    assert got["cyclen"] == pytest.approx(999.0)       # strict upgrade -> its row


# --------------------------------------------------------------------------- #
# UPSERT dedup: a converged label is the ground truth and must never be lost.
# Root cause of the 5.5-5.75_f117 149/150 loop: an earlier non-converged row for
# a record_id was kept by the old keep="last" dedup, silently discarding a later
# converged retry of the SAME pattern; the store then showed one fewer converged
# than the ledger counted, driving an off-by-one curriculum relaunch loop.
# --------------------------------------------------------------------------- #
def test_upsert_nonconverged_then_converged(tmp_path: Path) -> None:
    """(i) non-converged then converged retry (same record_id) -> store converged.

    The converged retry is strictly better information, so it upgrades the stored
    row; the store's converged count is then consistent with the retry outcome.
    """
    writer = StoreWriter(tmp_path)
    writer.write_records([_nonconverged("rid", cyclen=100.0)], append=False)
    stats = writer.write_records([_make_record("rid", cyclen=680.0)], append=True)
    assert stats["total"] == 1

    df = StoreReader(tmp_path).records
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["converged"]) is True
    assert bool(row["valid"]) is True
    assert row["cyclen"] == pytest.approx(680.0)          # the converged retry's data
    # store's converged count for this record_id is exactly 1 (no off-by-one)
    assert int((df["converged"] == True).sum()) == 1      # noqa: E712


def test_upsert_converged_then_nonconverged_never_downgrades(tmp_path: Path) -> None:
    """(ii) converged then a later non-converged/failed write keeps CONVERGED.

    A stale or racing worse write must never downgrade a converged label.
    """
    writer = StoreWriter(tmp_path)
    writer.write_records([_make_record("rid", cyclen=680.0)], append=False)
    # a later non-convergence for the same record_id ...
    writer.write_records([_nonconverged("rid", cyclen=111.0)], append=True)
    # ... and a later hard error, in either order, both refused.
    stats = writer.write_records([_failed("rid", cyclen=222.0)], append=True)
    assert stats["total"] == 1

    df = StoreReader(tmp_path).records
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["converged"]) is True                 # never downgraded
    assert row["cyclen"] == pytest.approx(680.0)          # original converged data intact
    assert int((df["converged"] == True).sum()) == 1      # noqa: E712


def test_upsert_plain_duplicate_unchanged(tmp_path: Path) -> None:
    """(iii) an exact duplicate write leaves the store unchanged (one row)."""
    writer = StoreWriter(tmp_path)
    writer.write_records([_make_record("rid", cyclen=680.0)], append=False)
    before = StoreReader(tmp_path).records
    stats = writer.write_records([_make_record("rid", cyclen=680.0)], append=True)
    assert stats["total"] == 1

    after = StoreReader(tmp_path).records
    assert len(after) == 1
    pd.testing.assert_frame_equal(before, after)


def test_upsert_nonconverged_over_error(tmp_path: Path) -> None:
    """A valid non-convergence outranks an invalid error; error never downgrades it."""
    writer = StoreWriter(tmp_path)
    writer.write_records([_failed("rid")], append=False)
    writer.write_records([_nonconverged("rid")], append=True)      # upgrade error -> valid
    writer.write_records([_failed("rid")], append=True)            # refused (worse)

    df = StoreReader(tmp_path).records
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["converged"]) is False
    assert bool(row["valid"]) is True                              # non-convergence kept


def test_upsert_mapped_row_outranks_the_unmapped_one(tmp_path: Path) -> None:
    """The harvested flatness columns are part of a row's information quality.

    ``node_peak`` / ``map_cov`` are what the flat_power objective is DEFINED on
    (program §1.3), so a re-write of the same ``record_id`` that carries no map
    must not null them out under the "ties keep the last write" rule.
    """
    writer = StoreWriter(tmp_path)
    mapped_dict = {**_make_record("rid").__dict__,
                   "node_peak": 1.512, "map_cov": 0.0834}
    writer.write_records([CanonicalRecord(**mapped_dict)], append=False)
    writer.write_records([_make_record("rid")], append=True)     # no map: refused

    df = StoreReader(tmp_path).records
    assert len(df) == 1
    assert float(df.iloc[0]["node_peak"]) == pytest.approx(1.512)

    # and the reverse direction still upgrades: an unmapped stored row is
    # replaced by the mapped one.
    writer2 = StoreWriter(tmp_path / "b")
    writer2.write_records([_make_record("rid")], append=False)
    writer2.write_records([CanonicalRecord(**mapped_dict)], append=True)
    got = StoreReader(tmp_path / "b").records
    assert len(got) == 1
    assert float(got.iloc[0]["map_cov"]) == pytest.approx(0.0834)


def test_upsert_converged_still_beats_a_mapped_nonconverged_row(tmp_path: Path) -> None:
    """The flatness bit is the LOW-order term: convergence still dominates."""
    writer = StoreWriter(tmp_path)
    writer.write_records([_make_record("rid")], append=False)      # converged, no map
    mapped_bad = {**_nonconverged("rid").__dict__,
                  "node_peak": 1.20, "map_cov": 0.05}
    writer.write_records([CanonicalRecord(**mapped_bad)], append=True)

    df = StoreReader(tmp_path).records
    assert len(df) == 1
    assert bool(df.iloc[0]["converged"]) is True
    assert pd.isna(df.iloc[0]["node_peak"])


def test_upsert_within_single_batch(tmp_path: Path) -> None:
    """UPSERT also collapses a converged+nonconverged pair inside ONE write batch."""
    writer = StoreWriter(tmp_path)
    stats = writer.write_records(
        [_make_record("rid", cyclen=680.0), _nonconverged("rid", cyclen=1.0)],
        append=False,
    )
    assert stats["total"] == 1
    df = StoreReader(tmp_path).records
    assert bool(df.iloc[0]["converged"]) is True
    assert df.iloc[0]["cyclen"] == pytest.approx(680.0)


def test_atomic_write_no_partial_on_failure(tmp_path: Path) -> None:
    """A failing writer leaves the pre-existing file untouched and no temp file."""
    final = tmp_path / "records.parquet"
    final.write_bytes(b"ORIGINAL")

    def _boom(p: Path) -> None:
        p.write_bytes(b"PARTIAL")
        raise RuntimeError("simulated failure mid-write")

    with pytest.raises(RuntimeError):
        _atomic_write(final, _boom)

    assert final.read_bytes() == b"ORIGINAL"                 # untouched
    assert not list(tmp_path.glob("records.parquet.tmp-*"))  # temp cleaned up


def test_maps_roundtrip_float16(tmp_path: Path) -> None:
    writer = StoreWriter(tmp_path)
    a = np.arange(4 * 9 * 9, dtype=np.float32).reshape(4, 9, 9)
    a[0, 8, 8] = np.nan
    writer.write_maps({"rid1": a}, append=False)

    reader = StoreReader(tmp_path)
    assert reader.has_maps
    got = reader.maps("rid1")
    assert got is not None
    assert got.shape == (4, 9, 9)
    assert got.dtype == np.float16
    assert np.isnan(got[0, 8, 8])
    assert reader.maps("missing") is None


def test_maps_append_merges(tmp_path: Path) -> None:
    writer = StoreWriter(tmp_path)
    writer.write_maps({"a": np.ones((4, 9, 9), np.float32)}, append=False)
    stats = writer.write_maps({"b": np.zeros((4, 9, 9), np.float32)}, append=True)
    assert stats["total"] == 2
    assert StoreReader(tmp_path).maps_keys() == {"a", "b"}


# --- regression: transient Windows lock must not kill a production run ------ #
# forensic 20260725: the 104 coverage-fill died mid-wave with
# PermissionError [WinError 5] on os.replace(maps.npz.tmp -> maps.npz) because
# another process momentarily held maps.npz open.  The bytes were already
# written; only the rename lost.  _atomic_write now retries with backoff.
def test_atomic_write_retries_transient_permission_error(monkeypatch, tmp_path) -> None:
    import lpopt.data.store as store_mod

    final = tmp_path / "maps.npz"
    final.write_bytes(b"old")
    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:                      # fail twice, then succeed
            raise PermissionError(5, "locked")
        return real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", flaky)
    monkeypatch.setattr(store_mod, "_REPLACE_BACKOFF_S", 0.0)
    store_mod._atomic_write(final, lambda p: p.write_bytes(b"new"))

    assert final.read_bytes() == b"new"         # the write landed
    assert calls["n"] == 3                      # and it took the retries


def test_atomic_write_gives_up_and_cleans_tmp(monkeypatch, tmp_path) -> None:
    import lpopt.data.store as store_mod

    final = tmp_path / "maps.npz"
    final.write_bytes(b"old")
    monkeypatch.setattr(store_mod.os, "replace",
                        lambda s, d: (_ for _ in ()).throw(PermissionError(5, "held")))
    monkeypatch.setattr(store_mod, "_REPLACE_BACKOFF_S", 0.0)
    with pytest.raises(PermissionError):
        store_mod._atomic_write(final, lambda p: p.write_bytes(b"new"))
    assert final.read_bytes() == b"old"         # original untouched
    assert not list(tmp_path.glob("*.tmp-*"))   # temp cleaned up


# --- reader must not hold maps.npz open (the WinError 5 root cause) --------- #
def test_store_reader_releases_maps_handle(tmp_path) -> None:
    """StoreReader.maps must not keep the npz handle open.

    A held read handle is exactly what made ``os.replace(maps.npz.tmp, maps.npz)``
    fail on Windows and kill the 104 coverage-fill twice (forensic 20260725).
    After a read, replacing the file must still succeed.
    """
    import lpopt.data.store as store_mod

    # StoreReader needs a records file to exist alongside the maps
    pd.DataFrame({"record_id": ["r1"]}).to_parquet(tmp_path / "records.parquet")
    maps_path = tmp_path / "maps.npz"
    np.savez_compressed(maps_path, a=np.ones((4, 9, 9), dtype=np.float16))

    reader = store_mod.StoreReader(tmp_path)
    assert reader.maps_keys() == {"a"}
    got = reader.maps("a")
    assert got is not None and got.shape == (4, 9, 9)
    assert reader.maps("missing") is None

    # THE ASSERTION: with the reader still alive and having read, the producer's
    # atomic rename onto maps.npz must succeed (it did not, before this fix).
    replacement = tmp_path / "replacement.npz"
    np.savez_compressed(replacement, a=np.zeros((4, 9, 9), dtype=np.float16))
    os.replace(replacement, maps_path)

    with np.load(maps_path) as z:
        assert float(z["a"].sum()) == 0.0        # the new file really landed


def test_maps_write_failure_skips_wave_not_campaign(monkeypatch, tmp_path) -> None:
    """LAST-RESORT GUARD: a locked maps.npz must cost one wave of maps, not the run.

    Mirrors the ProduceDriver/CampaignDriver guard: write_maps raising
    PermissionError is caught, counted, and execution continues.
    """
    class _Store:
        def write_maps(self, maps, append=True):
            raise PermissionError(5, "held by a reader")

    store = _Store()
    skipped_waves = skipped_records = 0
    wave_maps = {"r1": np.zeros((4, 9, 9)), "r2": np.zeros((4, 9, 9))}

    # the exact guard shape used in produce.py / campaign.py
    try:
        store.write_maps(wave_maps, append=True)
    except (PermissionError, OSError):
        skipped_waves += 1
        skipped_records += len(wave_maps)

    assert skipped_waves == 1 and skipped_records == 2   # counted, not raised


def test_replace_retry_budget_is_generous() -> None:
    """The retry budget must exceed a minute — 8 attempts (~30 s) was too short."""
    import lpopt.data.store as store_mod

    total = sum(min(store_mod._REPLACE_BACKOFF_S * (2 ** i),
                    store_mod._REPLACE_BACKOFF_CAP_S)
                for i in range(store_mod._REPLACE_ATTEMPTS - 1))
    assert store_mod._REPLACE_ATTEMPTS >= 30
    assert total >= 60.0


# --------------------------------------------------------------------------- #
# e_core / e_split are DERIVED, not caller metadata (regression 20260829)
#
# ``outcome_to_record`` takes e_core/e_split as plain kwargs, and the produce /
# campaign path passed its CaseContext's NOMINAL enrichment — the planned 50/50
# (or 1/N) split of the case, constant across a whole campaign, always with
# ``e_split=None``.  A realized pattern almost never lands on the nominal split,
# so 979 store rows advertised a core that was never loaded (max drift 0.068 w/o).
# The writer is the one choke point every producer funnels through, so it
# re-derives both columns from the row's own ``(pattern, library_id)``.
# --------------------------------------------------------------------------- #
def _fuel_parquet(store_dir: Path) -> None:
    """A 2-type ``fuel_types.parquet`` next to the records store."""
    store_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(library_id="260624", type_id="B1", u_avg_enrichment=5.0,
             u_mass_g=100.0, n_gd=None, source_flags=None, axial_zone=None,
             feature_poor=False),
        dict(library_id="260624", type_id="C2", u_avg_enrichment=6.0,
             u_mass_g=200.0, n_gd=None, source_flags=None, axial_zone=None,
             feature_poor=False),
    ]).to_parquet(store_dir / "fuel_types.parquet")


def _pattern_record(rid: str, packed: str, *, e_core, e_split) -> CanonicalRecord:
    rec = _make_record(rid)
    rec.pattern = packed
    rec.e_core = e_core
    rec.e_split = e_split
    return rec


def _full_pattern(head: str, tail: str, k: int = 35) -> str:
    """A real 69-slot pattern: the first ``k`` orbits ``head``, the rest ``tail``.

    ``batch_feed`` weights each slot by its ORBIT MULTIPLICITY, so the feed of
    this pattern is (115, 126) full-core positions, not (35, 34) — the derivation
    has to go through the geometry, which a 2-token toy string never exercises.
    """
    return "|".join([f"F:{head}:0"] * k + [f"F:{tail}:0"] * (len(SLOTS) - k))


#: B1 (e=5.0, m=100) x 115 positions + C2 (e=6.0, m=200) x 126 positions.
_MIXED = _full_pattern("B1", "C2")
_MIXED_MASS_WEIGHTED = (115 * 100 * 5.0 + 126 * 200 * 6.0) / (115 * 100 + 126 * 200)
_MIXED_COUNT_WEIGHTED = (115 * 5.0 + 126 * 6.0) / 241


def test_mixed_fixture_feed_matches_the_orbit_multiplicities() -> None:
    """Guard the fixture itself: if the 69-slot geometry ever changes, the two
    derived-column tests below must fail loudly rather than drift."""
    assert sum(s.multiplicity for s in SLOTS[:35]) == 115
    assert sum(s.multiplicity for s in SLOTS[35:]) == 126


def test_writer_overrides_a_nominal_e_core_with_the_pattern_derived_value(tmp_path
                                                                          ) -> None:
    _fuel_parquet(tmp_path)
    StoreWriter(tmp_path).write_records([
        # the produce path's exact signature: a nominal e_core + a null e_split
        _pattern_record("r1", _MIXED, e_core=5.4, e_split=None),
    ])
    got = StoreReader(tmp_path).records.iloc[0]
    assert got["e_core"] == pytest.approx(_MIXED_MASS_WEIGHTED)
    assert got["e_split"] == pytest.approx(1.0)       # 6.0 - 5.0, filled not null
    # the nominal is gone, and the result is the MASS-weighted mean specifically
    assert abs(got["e_core"] - 5.4) > 1e-3
    assert abs(got["e_core"] - _MIXED_COUNT_WEIGHTED) > 1e-3


def test_writer_leaves_the_caller_value_when_the_recipe_cannot_resolve(tmp_path
                                                                       ) -> None:
    """An unresolvable feed must never BLANK a supplied value — the recipe's
    ``(None, None)`` is a fallback, not a verdict."""
    _fuel_parquet(tmp_path)
    StoreWriter(tmp_path).write_records([
        _pattern_record("r1", _full_pattern("ZZ", "C2"), e_core=5.4, e_split=0.1),
        # a truncated / foreign pattern must not raise out of a records write
        _pattern_record("r2", "F:B1:0|F:C2:0", e_core=5.3, e_split=0.2),
    ])
    rows = StoreReader(tmp_path).records.set_index("record_id")
    assert rows.loc["r1", "e_core"] == pytest.approx(5.4)
    assert rows.loc["r1", "e_split"] == pytest.approx(0.1)
    assert rows.loc["r2", "e_core"] == pytest.approx(5.3)
    assert rows.loc["r2", "e_split"] == pytest.approx(0.2)


def test_writer_without_a_fuel_table_writes_verbatim(tmp_path) -> None:
    """No ``fuel_types.parquet`` (a multi-PC kit, a synthetic fixture) -> the
    derivation is simply unavailable and the frame is written as handed over."""
    writer = StoreWriter(tmp_path)
    assert writer.fuel_library is None
    writer.write_records([_pattern_record("r1", _MIXED, e_core=5.4, e_split=0.1)])
    got = StoreReader(tmp_path).records.iloc[0]
    assert got["e_core"] == pytest.approx(5.4) and got["e_split"] == pytest.approx(0.1)


def test_writer_opt_out_preserves_a_verbatim_frame(tmp_path) -> None:
    _fuel_parquet(tmp_path)
    StoreWriter(tmp_path).write_records(
        [_pattern_record("r1", _MIXED, e_core=5.4, e_split=None)],
        derive_enrichment_columns=False,
    )
    assert StoreReader(tmp_path).records.iloc[0]["e_core"] == pytest.approx(5.4)


def test_backfill_e_core_is_dry_run_by_default(tmp_path) -> None:
    from lpopt.data.store import backfill_e_core

    _fuel_parquet(tmp_path)
    # write the bad rows PAST the writer guard, exactly as the old produce path did
    StoreWriter(tmp_path).write_records(
        [_pattern_record("r1", _MIXED, e_core=5.4, e_split=None),       # nominal
         _pattern_record("r2", _MIXED, e_core=None, e_split=None),      # never filled
         _pattern_record("r3", _MIXED, e_core=_MIXED_MASS_WEIGHTED,     # already right
                         e_split=1.0)],
        derive_enrichment_columns=False,
    )
    report = backfill_e_core(tmp_path)
    assert report["applied"] is False and report["backup"] is None
    assert report["rows"] == 3 and report["resolvable"] == 3
    assert report["corrected"] == 1                 # r1
    assert report["null_filled"] == 1               # r2
    assert report["unchanged"] == 1                 # r3
    # the dry run wrote NOTHING
    assert StoreReader(tmp_path).records.set_index("record_id").loc[
        "r1", "e_core"] == pytest.approx(5.4)

    applied = backfill_e_core(tmp_path, dry_run=False, backup_suffix="t")
    assert applied["applied"] is True
    assert Path(applied["backup"]).is_file()
    rows = StoreReader(tmp_path).records.set_index("record_id")
    for rid in ("r1", "r2", "r3"):
        assert rows.loc[rid, "e_core"] == pytest.approx(_MIXED_MASS_WEIGHTED)
        assert rows.loc[rid, "e_split"] == pytest.approx(1.0)
    # idempotent: a second pass finds nothing left to correct
    assert backfill_e_core(tmp_path)["corrected"] == 0
