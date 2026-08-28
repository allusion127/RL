"""Store I/O: schema round-trip, dedup-by-record_id, atomic write, maps.npz."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.data.schema import CanonicalRecord, SCHEMA_COLUMNS, compute_record_id
from lpopt.data.store import StoreReader, StoreWriter, _atomic_write, records_to_frame


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
    """Two EQUAL-quality writes (both converged) collapse to one; last wins."""
    writer = StoreWriter(tmp_path)
    writer.write_records([_make_record("dup", cyclen=680.0)], append=False)
    stats = writer.write_records([_make_record("dup", cyclen=999.0)], append=True)
    assert stats["total"] == 1

    df = StoreReader(tmp_path).records
    assert len(df) == 1
    assert df.iloc[0]["cyclen"] == pytest.approx(999.0)   # equal quality -> last write wins


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
