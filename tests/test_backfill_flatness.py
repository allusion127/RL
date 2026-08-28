"""The two flatness record columns: schema growth, harvest write, and backfill.

``node_peak`` / ``map_cov`` are the first columns appended to the record schema
after the 36-column freeze, so what is pinned here is as much the *migration
contract* as the values: an old ``records.parquet`` (or an old multi-PC kit) that
predates the columns must still read, still merge, and still write — with nulls,
not a ``KeyError``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.data import flatness as F
from lpopt.data.schema import (
    FROZEN_COLUMNS, LATE_COLUMNS, SCHEMA_COLUMNS, CanonicalRecord,
)
from lpopt.data.store import (
    RECORDS_NAME, StoreReader, StoreWriter, ensure_schema_columns,
)
from lpopt.tools.backfill_flatness import backfill
from lpopt.vendor.masterrl.domain import SLOTS


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _record(rid: str, *, maps_key: str | None = None, **kw) -> CanonicalRecord:
    base = dict(
        record_id=rid, dataset="P", campaign="c", stratum=None, generator="g",
        parent_record_id=None, case_pair="B1_C2", feed=121, n_batches=2,
        depth2_edges=0, e_core=5.4, e_split=0.1, library_id="260624",
        sym_class="rot61", pattern="F:B1:0|F:C2:0", f_r=1.9, f_q=2.4,
        cbc_max=1500.0, cbc_boc=1480.0, cbc_kind="max", cyclen=680.0, ao_abs=0.05,
        cycle_burnup=27.0, discharge_burnup=54.0, max_assembly_burnup=67.0,
        max_pin_burnup=71.0, eoc_ppm=10.0, delta_efpd=0.5, n_cycles=11.0,
        converged=True, converged_at_cap=False, tolerance_margin=None,
        restart_provenance="mocha_native", valid=True, failure="",
        maps_key=maps_key,
    )
    base.update(kw)
    return CanonicalRecord(**base)


def _legacy_stack(vals: np.ndarray) -> np.ndarray:
    """``(4, 9, 9)`` store stack whose BOC channel carries ``vals``."""
    plane = np.full((9, 9), np.nan, dtype=np.float32)
    plane[F.SLOT_ROWS, F.SLOT_COLS] = vals
    other = np.full((9, 9), np.nan, dtype=np.float32)
    return np.stack([plane, other, other, other], axis=0)


def _expected(stack: np.ndarray) -> tuple[float, float]:
    """``(node_peak, map_cov)`` written out longhand, independent of the module.

    The store keeps maps as float16, so the reference is computed on the STORED
    precision — otherwise this would be pinning the round-trip error, not the
    definition.
    """
    sv = np.asarray(stack, dtype=np.float16).astype(np.float64)[0]
    sv = sv[[s.row for s in SLOTS], [s.col for s in SLOTS]]
    w = np.array([s.multiplicity for s in SLOTS], dtype=np.float64)
    mean = float((w * sv).sum() / w.sum())
    sd = float(np.sqrt((w * (sv - mean) ** 2).sum() / w.sum()))
    return float(sv.max()), sd / mean


def _seeded_store(tmp_path: Path, n_mapped: int = 3, n_bare: int = 2):
    """A store with ``n_mapped`` map-carrying rows whose columns are still null."""
    rng = np.random.default_rng(19)
    records, maps, truth = [], {}, {}
    for i in range(n_mapped):
        rid = f"m{i}"
        vals = 1.0 + 0.05 * rng.standard_normal(69)
        maps[rid] = _legacy_stack(vals)
        truth[rid] = _expected(maps[rid])
        records.append(_record(rid, maps_key=rid))
    for i in range(n_bare):
        records.append(_record(f"b{i}"))
    w = StoreWriter(tmp_path)
    w.write_records(records, append=False)
    if maps:
        w.write_maps(maps, append=False)
    return truth


# --------------------------------------------------------------------------- #
# schema growth
# --------------------------------------------------------------------------- #
def test_columns_are_appended_after_the_frozen_prefix() -> None:
    assert len(FROZEN_COLUMNS) == 36
    assert SCHEMA_COLUMNS[:36] == list(FROZEN_COLUMNS)
    assert (SCHEMA_COLUMNS[36:] == list(LATE_COLUMNS)
            == ["node_peak", "map_cov", "max_rod_avg_burnup"])


def test_new_columns_default_to_none() -> None:
    rec = _record("x")
    assert rec.node_peak is None and rec.map_cov is None
    assert set(rec.to_record()) == set(SCHEMA_COLUMNS)


def test_columns_round_trip_through_parquet(tmp_path: Path) -> None:
    StoreWriter(tmp_path).write_records(
        [_record("a", node_peak=1.42, map_cov=0.31), _record("b")], append=False)
    df = StoreReader(tmp_path).records.set_index("record_id")
    assert list(df.columns)[-3:] == ["node_peak", "map_cov", "max_rod_avg_burnup"]
    assert df.loc["a", "node_peak"] == pytest.approx(1.42)
    assert df.loc["a", "map_cov"] == pytest.approx(0.31)
    assert pd.isna(df.loc["b", "node_peak"])


# --------------------------------------------------------------------------- #
# backward compatibility with a pre-column store
# --------------------------------------------------------------------------- #
def _strip_late_columns(store_dir: Path) -> None:
    """Rewrite records.parquet WITHOUT the tail columns (an old store)."""
    p = store_dir / RECORDS_NAME
    df = pd.read_parquet(p).drop(columns=list(LATE_COLUMNS))
    df.to_parquet(p, index=False)


def test_old_store_reads_back_with_null_columns(tmp_path: Path) -> None:
    StoreWriter(tmp_path).write_records([_record("a"), _record("b")], append=False)
    _strip_late_columns(tmp_path)
    assert list(pd.read_parquet(tmp_path / RECORDS_NAME).columns) == list(FROZEN_COLUMNS)

    df = StoreReader(tmp_path).records
    assert list(df.columns) == SCHEMA_COLUMNS
    assert df["node_peak"].isna().all() and df["map_cov"].isna().all()


def test_appending_to_an_old_store_upgrades_it(tmp_path: Path) -> None:
    StoreWriter(tmp_path).write_records([_record("a")], append=False)
    _strip_late_columns(tmp_path)

    StoreWriter(tmp_path).write_records(
        [_record("b", node_peak=1.5, map_cov=0.3)], append=True)
    df = StoreReader(tmp_path).records.set_index("record_id")
    assert len(df) == 2
    assert pd.isna(df.loc["a", "node_peak"])           # pre-existing row: null
    assert df.loc["b", "node_peak"] == pytest.approx(1.5)


def test_ensure_schema_columns_is_a_no_op_when_present(tmp_path: Path) -> None:
    StoreWriter(tmp_path).write_records([_record("a")], append=False)
    df = pd.read_parquet(tmp_path / RECORDS_NAME)
    assert ensure_schema_columns(df) is df                # same object, no copy


def test_ensure_schema_columns_does_not_invent_frozen_columns() -> None:
    df = pd.DataFrame({"record_id": ["a"]})
    out = ensure_schema_columns(df)
    assert set(LATE_COLUMNS) <= set(out.columns)
    assert "f_r" not in out.columns                      # a broken frame stays broken


# --------------------------------------------------------------------------- #
# harvest-time population
# --------------------------------------------------------------------------- #
def test_outcome_to_record_populates_from_the_harvested_map() -> None:
    from lpopt.search.verify import outcome_to_record

    rng = np.random.default_rng(4)
    vals = 1.0 + 0.05 * rng.standard_normal(69)

    class _Outcome:
        status = "converged"
        fom = None
        n_cycles = 11
        tolerance_margin = None
        converged_at_cap = False
        failure = ""
        restart_provenance = "mocha_native"
        maps = _legacy_stack(vals)

        class case_key:      # noqa: N801 - stand-in for the vendor CaseKey
            pair = "B1_C2"
            feed = 121

        class pattern:       # noqa: N801 - stand-in for the vendor Pattern
            @staticmethod
            def canonical() -> str:
                return "F:B1:0|F:C2:0"

    rec = outcome_to_record(_Outcome(), library_id="260624")
    assert rec.node_peak == pytest.approx(float(F.node_peak(vals)[0]))
    assert rec.map_cov == pytest.approx(float(F.map_cov(vals)[0]))

    _Outcome.maps = None
    bare = outcome_to_record(_Outcome(), library_id="260624")
    assert bare.node_peak is None and bare.map_cov is None and bare.maps_key is None


def test_outcome_to_record_survives_a_broken_map() -> None:
    """A map is optional; the F_r/cyclen labels of the same row are not."""
    from lpopt.search.verify import outcome_to_record

    class _Outcome:
        status = "converged"
        fom = None
        n_cycles = 11
        tolerance_margin = None
        converged_at_cap = False
        failure = ""
        restart_provenance = "mocha_native"
        maps = np.zeros((5, 5))            # not a map layout

        class case_key:      # noqa: N801
            pair = "B1_C2"
            feed = 121

        class pattern:       # noqa: N801
            @staticmethod
            def canonical() -> str:
                return "F:B1:0|F:C2:0"

    rec = outcome_to_record(_Outcome(), library_id="260624")
    assert rec.node_peak is None and rec.map_cov is None
    assert rec.maps_key == rec.record_id   # the map is still stored


# --------------------------------------------------------------------------- #
# the backfill
# --------------------------------------------------------------------------- #
def test_backfill_populates_every_mapped_row(tmp_path: Path) -> None:
    truth = _seeded_store(tmp_path)
    rep = backfill(tmp_path, log=lambda m: None)

    assert rep.n_populated == len(truth)
    assert rep.n_with_maps_key == len(truth)
    assert rep.n_without_maps_key == 2
    assert rep.wrote is True

    df = StoreReader(tmp_path).records.set_index("record_id")
    for rid, (peak, cov) in truth.items():
        assert df.loc[rid, "node_peak"] == pytest.approx(peak)
        assert df.loc[rid, "map_cov"] == pytest.approx(cov)
    assert df.loc["b0", "node_peak"] != df.loc["b0", "node_peak"]   # still NaN


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    _seeded_store(tmp_path)
    backfill(tmp_path, log=lambda m: None)
    first = (tmp_path / RECORDS_NAME).read_bytes()

    second = backfill(tmp_path, log=lambda m: None)
    assert second.n_populated == 0
    assert second.n_already == 3
    assert second.wrote is False
    # a no-op run does not even touch the file
    assert (tmp_path / RECORDS_NAME).read_bytes() == first


def test_backfill_works_on_a_store_that_predates_the_columns(tmp_path: Path) -> None:
    truth = _seeded_store(tmp_path)
    _strip_late_columns(tmp_path)

    rep = backfill(tmp_path, log=lambda m: None)
    assert rep.n_populated == len(truth)
    df = StoreReader(tmp_path).records.set_index("record_id")
    for rid, (peak, _cov) in truth.items():
        assert df.loc[rid, "node_peak"] == pytest.approx(peak)


def test_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    _seeded_store(tmp_path)
    before = (tmp_path / RECORDS_NAME).read_bytes()
    rep = backfill(tmp_path, dry_run=True, log=lambda m: None)
    assert rep.n_populated == 3 and rep.wrote is False
    assert (tmp_path / RECORDS_NAME).read_bytes() == before


def test_backfill_counts_a_dangling_maps_key_and_leaves_it_null(tmp_path: Path) -> None:
    """Program §9 P-5: 13 rows point at a maps_key that is not in maps.npz."""
    _seeded_store(tmp_path)
    StoreWriter(tmp_path).write_records(
        [_record("ghost", maps_key="ghost")], append=True)

    rep = backfill(tmp_path, log=lambda m: None)
    assert rep.n_dangling == 1
    df = StoreReader(tmp_path).records.set_index("record_id")
    assert pd.isna(df.loc["ghost", "node_peak"])


def test_backfill_preserves_rows_written_after_its_read(tmp_path: Path) -> None:
    """The patch is applied to a FRESH read, so a concurrent append survives."""
    _seeded_store(tmp_path)
    reader = StoreReader(tmp_path)
    _ = reader.records                       # warm the cached frame
    StoreWriter(tmp_path).write_records([_record("late")], append=True)

    backfill(tmp_path, log=lambda m: None)
    df = StoreReader(tmp_path).records
    assert "late" in set(df["record_id"])
    assert len(df) == 6


def test_backfill_preserves_row_order(tmp_path: Path) -> None:
    """Row ORDER is part of the store's contract.

    Downstream code slices the frame positionally (``head(n)``, the first row that
    carries a map, fold construction).  A backfill that appends its updated rows —
    which is what ``write_records(append=True)`` does on a record_id collision —
    silently changes what every one of those slices means.
    """
    _seeded_store(tmp_path)
    before = StoreReader(tmp_path).records["record_id"].tolist()
    backfill(tmp_path, log=lambda m: None)
    assert StoreReader(tmp_path).records["record_id"].tolist() == before


def test_backfill_leaves_the_frozen_columns_untouched(tmp_path: Path) -> None:
    _seeded_store(tmp_path)
    before = StoreReader(tmp_path).records[list(FROZEN_COLUMNS)]
    backfill(tmp_path, log=lambda m: None)
    after = StoreReader(tmp_path).records[list(FROZEN_COLUMNS)]
    assert before.equals(after)


# --------------------------------------------------------------------------- #
# the "already correct" tolerance
#
# The regression: the comparison used atol 1e-12, but a harvest-time value is
# derived from the FLOAT32 EDIT5 array while this pass recomputes from the
# FLOAT16 copy in maps.npz.  Every already-correct row therefore looked stale and
# ~30k rows were rewritten on pure dtype noise every run.
# --------------------------------------------------------------------------- #
def test_a_float32_derived_value_is_not_rewritten_as_float16_noise(tmp_path: Path) -> None:
    from lpopt.tools.backfill_flatness import EQUAL_RTOL

    _seeded_store(tmp_path, n_mapped=3, n_bare=0)
    reader = StoreReader(tmp_path)
    df = ensure_schema_columns(reader.records)
    # Write, for each row, the value the HARVEST path produced: the same scalar
    # computed from a float32 array that rounds to the stored float16 map.
    rng = np.random.default_rng(3)
    for pos, rid in enumerate(df["record_id"].astype(str)):
        stored = np.asarray(reader.maps(rid))               # float16, as stored
        # one float16 ULP of each slot, in float32 — the exact envelope of
        # float32 values that quantize back to the stored map.
        ulp = np.spacing(np.nan_to_num(stored, nan=np.float16(1.0)))
        jitter = ulp.astype(np.float32) * (rng.random(stored.shape) - 0.5)
        peak, cov = F.record_flatness(
            stored.astype(np.float32) + jitter.astype(np.float32))
        df.loc[df.index[pos], "node_peak"] = peak
        df.loc[df.index[pos], "map_cov"] = cov
    StoreWriter(tmp_path).write_records(df, append=False)
    before = (tmp_path / RECORDS_NAME).read_bytes()

    rep = backfill(tmp_path, log=lambda m: None)
    assert rep.n_populated == 0
    assert rep.n_already == 3
    assert rep.wrote is False
    assert (tmp_path / RECORDS_NAME).read_bytes() == before
    # and the tolerance is the float16 resolution, not a float64 round-trip.
    assert EQUAL_RTOL == pytest.approx(2.0 ** -10)


def test_a_genuinely_stale_value_is_still_rewritten(tmp_path: Path) -> None:
    """The looser tolerance must not become a licence to keep a wrong value."""
    from lpopt.tools.backfill_flatness import EQUAL_RTOL

    _seeded_store(tmp_path, n_mapped=1, n_bare=0)
    reader = StoreReader(tmp_path)
    df = ensure_schema_columns(reader.records)
    true_peak = float(F.record_flatness(reader.maps("m0"))[0])
    df.loc[df.index[0], "node_peak"] = true_peak * (1.0 + 100.0 * EQUAL_RTOL)
    StoreWriter(tmp_path).write_records(df, append=False)

    rep = backfill(tmp_path, log=lambda m: None)
    assert rep.n_populated == 1 and rep.wrote is True
    got = StoreReader(tmp_path).records.set_index("record_id").loc["m0", "node_peak"]
    assert got == pytest.approx(true_peak)


def test_needs_write_treats_a_non_finite_stored_value_as_stale() -> None:
    from lpopt.tools.backfill_flatness import _needs_write

    assert _needs_write(float("inf"), 1.5) is True
    assert _needs_write(float("nan"), 1.5) is True      # NaN is "missing"
    assert _needs_write(1.5, None) is False             # never null out a value
    assert _needs_write(1.5, 1.5 * (1.0 + 2.0 ** -12)) is False


def test_backfill_matches_the_canonical_definition(tmp_path: Path) -> None:
    """The backfilled value is byte-for-byte what the harvest path would write."""
    _seeded_store(tmp_path, n_mapped=2, n_bare=0)
    backfill(tmp_path, log=lambda m: None)
    reader = StoreReader(tmp_path)
    df = reader.records
    for rid, peak, cov in zip(df["record_id"], df["node_peak"], df["map_cov"]):
        direct = F.record_flatness(reader.maps(rid))
        assert peak == pytest.approx(direct[0])
        assert cov == pytest.approx(direct[1])
