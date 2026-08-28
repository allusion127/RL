"""cond_v6b — the burnup-placement arm (`data/reports/ab2_addendum_BU_20260810.md`).

Two things have to be true for the A/B to mean anything, and both are asserted
here rather than asserted in a docstring:

1. **v6 did not move.**  The control arm is a v6 retrain, so if this change
   perturbed a single v6 feature bit, every paired difference measured against
   that control would be measuring the refactor instead of the arm.  The v6
   arithmetic is therefore compared BIT-FOR-BIT against a verbatim
   re-implementation of the pre-change formulas (the same proof strategy round 2
   used for the `_map_quarter` code motion, pre-registration §9).

2. **v6b changes exactly what it claims to.**  The prefix test below encodes the
   same real store rows under both schemas and pins the changed channel set to
   exactly {`nominal_burnup`, `origin_kinf_contrast`, `prior_power`,
   `prior_power_contrast`} plus the six appended ones — an unexpected extra
   channel moving would fail here, not in a four-hour training run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lpopt.data.fuel_types import FuelLibrary
from lpopt.data.schema import unpack_pattern
from lpopt.model import power_prior as pp
from lpopt.model.featurize import (
    CHANNELS_BY_SCHEMA,
    CHANNELS_V6,
    CHANNELS_V6B,
    NOMINAL_CYCLE_BURNUP_MWD_KG,
    _BURNUP_SCALE,
    _R_MAX,
    _REGIME_CYCLE_BURNUP_MWD_KG,
    _V6B_SRC_CHAIN_EXTRA,
    _V6_POWER_SCALE,
    FeatureEncoder,
    RecordInputs,
    regime_cycle_burnup,
    schema_uses_regime_burnup,
)
from lpopt.model.physics_prior import _resolve, assembly_rho
from lpopt.vendor.masterrl.domain import SLOTS

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"
RECORDS = STORE / "records.parquet"
FUEL = STORE / "fuel_types.parquet"

#: Channels cond_v6b is ALLOWED to move inside v6's 52.  Anything else moving is
#: a bug: the regime constant reaches the burn state and the power prior it
#: drives, and nothing else.
_EXPECTED_CHANGED = {
    "nominal_burnup",            # (age-1) * B_regime / 66
    "origin_kinf_contrast",      # k-inf now read at the regime burnup
    "prior_power",               # ... so the diffusion solve moves
    "prior_power_contrast",
}


def _load(n: int = 48):
    if not RECORDS.is_file() or not FUEL.is_file():
        pytest.skip("Dataset-A store not present")
    import pandas as pd

    df = pd.read_parquet(RECORDS)
    fl = FuelLibrary.from_parquet(FUEL)
    return df, fl


def _sample(df, n_per_lib: int = 6):
    """A few rows from every library, so both regime-covered and fallback
    libraries are exercised."""
    out = []
    for lib, grp in df.groupby("library_id"):
        out.append(grp.head(n_per_lib))
    import pandas as pd

    return pd.concat(out).reset_index(drop=True)


@pytest.fixture(scope="module")
def store():
    return _load()


# --------------------------------------------------------------------------- #
# 1) the regime table and its fallback chain
# --------------------------------------------------------------------------- #
def test_regime_table_exact_hits_are_the_measured_constants() -> None:
    """The six numbers of memo §3(0), unrounded and unreordered."""
    assert regime_cycle_burnup("ga80", 101) == pytest.approx(24.94)
    assert regime_cycle_burnup("ga80", 121) == pytest.approx(28.69)
    assert regime_cycle_burnup("ga80", 141) == pytest.approx(31.14)
    assert regime_cycle_burnup("paramA", 101) == pytest.approx(30.47)
    assert regime_cycle_burnup("paramA", 121) == pytest.approx(35.33)
    assert regime_cycle_burnup("paramA", 141) == pytest.approx(36.77)


def test_regime_table_interpolates_in_feed_within_a_library() -> None:
    # feed 111 sits halfway between the 101 and 121 anchors.
    assert regime_cycle_burnup("ga80", 111) == pytest.approx((24.94 + 28.69) / 2)
    assert regime_cycle_burnup("paramA", 131) == pytest.approx((35.33 + 36.77) / 2)
    # every feed the store actually carries is INSIDE the hull, so it is
    # interpolated and never invented
    for feed in (101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141):
        for lib in ("ga80", "paramA"):
            b = regime_cycle_burnup(lib, feed)
            lo = min(_REGIME_CYCLE_BURNUP_MWD_KG[lib].values())
            hi = max(_REGIME_CYCLE_BURNUP_MWD_KG[lib].values())
            assert lo - 1e-9 <= b <= hi + 1e-9


def test_regime_table_falls_back_to_library_mean_outside_the_hull() -> None:
    """No extrapolation: a 3-point table does not license one."""
    ga = _REGIME_CYCLE_BURNUP_MWD_KG["ga80"]
    mean = sum(ga.values()) / len(ga)
    assert regime_cycle_burnup("ga80", 61) == pytest.approx(mean)
    assert regime_cycle_burnup("ga80", 201) == pytest.approx(mean)


def test_regime_table_falls_back_to_the_legacy_constant_for_unknown_libraries() -> None:
    for lib in ("260624", "5.8_5.1", "legacy_a", "CPHA", "unresolved:X"):
        assert regime_cycle_burnup(lib, 121) == NOMINAL_CYCLE_BURNUP_MWD_KG
    # and a nonsense feed never raises -- an exception here would kill a whole
    # training run from inside the encoder over one bad cell
    for bad in (None, float("nan"), float("inf"), float("-inf"), "", "abc", []):
        assert regime_cycle_burnup("ga80", bad) == NOMINAL_CYCLE_BURNUP_MWD_KG


def test_regime_table_reads_no_label() -> None:
    """It is constants in source, keyed on two SAFE input fields."""
    assert set(_REGIME_CYCLE_BURNUP_MWD_KG) == {"ga80", "paramA"}
    for tbl in _REGIME_CYCLE_BURNUP_MWD_KG.values():
        assert set(tbl) == {101, 121, 141}
        assert all(isinstance(v, float) for v in tbl.values())


def test_schema_marker_only_fires_for_v6b() -> None:
    assert schema_uses_regime_burnup("v6b")
    for s in ("v2", "v3", "v4", "v5", "v5_noshape", "v6", "v6_prior", "v6_contrast"):
        assert not schema_uses_regime_burnup(s)
    assert not schema_uses_regime_burnup("nonexistent")


# --------------------------------------------------------------------------- #
# 2) the inventory is append-only
# --------------------------------------------------------------------------- #
def test_v6b_channels_are_append_only_after_v6() -> None:
    assert len(CHANNELS_V6) == 52
    assert CHANNELS_V6B[:52] == CHANNELS_V6
    assert CHANNELS_V6B == CHANNELS_V6 + _V6B_SRC_CHAIN_EXTRA
    assert len(CHANNELS_V6B) == 58
    assert len(set(CHANNELS_V6B)) == len(CHANNELS_V6B)          # unique names
    assert CHANNELS_BY_SCHEMA["v6b"] == CHANNELS_V6B
    # the residual-skip plane the map head reads must not have moved
    assert CHANNELS_V6B.index("prior_power") == CHANNELS_V6.index("prior_power") == 50


def test_v6b_globals_are_identical_to_v6(store) -> None:
    a = FeatureEncoder(cond_schema="v6")
    b = FeatureEncoder(cond_schema="v6b")
    assert a.globals_names == b.globals_names
    assert len(a.globals_names) == 13


def test_v6b_encodes_finite_arrays_of_the_right_shape(store) -> None:
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6b")
    cells, gvec = enc.encode(df.iloc[0], fl)
    assert cells.shape == (58, 19, 19)
    assert cells.dtype == np.float32
    assert gvec.shape == (13,)
    assert np.isfinite(cells).all()
    assert np.isfinite(gvec).all()


# --------------------------------------------------------------------------- #
# 3) v6 DID NOT MOVE — verbatim re-implementation of the pre-change formulas
# --------------------------------------------------------------------------- #
def test_v6_nominal_burnup_is_the_legacy_arithmetic(store) -> None:
    """``(age - 1) * 22.0 / (22.0 * 3)`` — bit-for-bit, on real rows."""
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6")
    ix = enc._ch_index["nominal_burnup"]
    for _, row in _sample(df, 4).iterrows():
        inp = RecordInputs.coerce(row)
        vals = enc.encode_slot_matrix(inp, fl)
        age, _origin, _direct = enc._trace_chain(unpack_pattern(inp.pattern).items)
        want = np.asarray(
            [(age[s.index] - 1) * NOMINAL_CYCLE_BURNUP_MWD_KG
             / (NOMINAL_CYCLE_BURNUP_MWD_KG * 3.0) for s in SLOTS],
            dtype=np.float32)
        assert np.array_equal(vals[ix][[s.index for s in SLOTS]], want)


def test_v6_kinf_quarter_is_the_legacy_arithmetic(store) -> None:
    """``bu = (age-1)*NOMINAL + 0.5*NOMINAL`` — the pre-change body, verbatim."""
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6")
    for _, row in _sample(df, 3).iterrows():
        inp = RecordInputs.coerce(row)
        got = pp.kinf_quarter(inp, fl, encoder=enc)

        items = unpack_pattern(inp.pattern).items
        age, origin, _ = enc._trace_chain(items)
        want = np.full(len(SLOTS), np.nan, dtype=np.float64)
        for slot in SLOTS:
            j = slot.index
            vec = _resolve(fl, items[origin[j]].batch, inp.library_id, None)
            bu = ((age[j] - 1) * NOMINAL_CYCLE_BURNUP_MWD_KG
                  + 0.5 * NOMINAL_CYCLE_BURNUP_MWD_KG)
            rho, _slope = assembly_rho(vec, bu)
            if math.isfinite(rho) and rho < 1.0e5:
                want[j] = 1.0 / (1.0 - rho / 1.0e5)
        finite = np.isfinite(want)
        if not finite.any():
            want[:] = pp.FALLBACK_KINF
        elif not finite.all():
            want[~finite] = float(np.median(want[finite]))
        assert np.array_equal(got, want)


def test_kinf_quarter_batch_default_is_bit_identical_to_per_row(store) -> None:
    df, fl = store
    sub = _sample(df, 3)
    batch = pp.kinf_quarter_batch(sub, fl)
    assert batch.shape == (len(sub), len(SLOTS))
    enc = FeatureEncoder(cond_schema="v6")
    for i, (_, row) in enumerate(sub.iterrows()):
        assert np.array_equal(
            batch[i], pp.kinf_quarter(RecordInputs.coerce(row), fl, encoder=enc))


# --------------------------------------------------------------------------- #
# 4) v6b changes EXACTLY the four channels it claims to, and nothing else
# --------------------------------------------------------------------------- #
def test_v6b_prefix_is_identical_to_v6_for_fallback_libraries(store) -> None:
    """Where the regime table has no entry, B == 22.0, so v6's 52 must be
    bit-identical — only the six appended channels may carry anything."""
    df, fl = store
    fallback = df[~df["library_id"].astype(str).isin(_REGIME_CYCLE_BURNUP_MWD_KG)]
    if not len(fallback):
        pytest.skip("no fallback-library rows in the store")
    a = FeatureEncoder(cond_schema="v6")
    b = FeatureEncoder(cond_schema="v6b")
    for _, row in fallback.head(8).iterrows():
        ca, ga = a.encode(row, fl)
        cb, gb = b.encode(row, fl)
        assert np.array_equal(ca, cb[:52])
        assert np.array_equal(ga, gb)


def test_v6b_moves_only_the_expected_channels_for_regime_libraries(store) -> None:
    df, fl = store
    covered = df[df["library_id"].astype(str).isin(_REGIME_CYCLE_BURNUP_MWD_KG)]
    if not len(covered):
        pytest.skip("no regime-covered rows in the store")
    a = FeatureEncoder(cond_schema="v6")
    b = FeatureEncoder(cond_schema="v6b")
    moved: set[str] = set()
    for _, row in _sample(covered, 4).iterrows():
        ca, ga = a.encode(row, fl)
        cb, gb = b.encode(row, fl)
        assert np.array_equal(ga, gb), "globals must not move"
        for i, name in enumerate(CHANNELS_V6):
            if not np.array_equal(ca[i], cb[i]):
                moved.add(name)
    assert moved <= _EXPECTED_CHANGED, f"unexpected channels moved: {moved - _EXPECTED_CHANGED}"
    # and the burn state MUST actually have moved, or the arm is a no-op
    assert "nominal_burnup" in moved
    assert "prior_power" in moved


def test_v6b_nominal_burnup_carries_the_regime_constant(store) -> None:
    df, fl = store
    covered = df[df["library_id"].astype(str).isin(_REGIME_CYCLE_BURNUP_MWD_KG)]
    if not len(covered):
        pytest.skip("no regime-covered rows in the store")
    enc = FeatureEncoder(cond_schema="v6b")
    ix = enc._ch_index["nominal_burnup"]
    for _, row in _sample(covered, 3).iterrows():
        inp = RecordInputs.coerce(row)
        vals = enc.encode_slot_matrix(inp, fl)
        age, _o, _d = enc._trace_chain(unpack_pattern(inp.pattern).items)
        b = regime_cycle_burnup(inp.library_id, inp.feed)
        assert b != NOMINAL_CYCLE_BURNUP_MWD_KG
        for slot in SLOTS:
            assert vals[ix, slot.index] == pytest.approx(
                (age[slot.index] - 1) * b / _BURNUP_SCALE, rel=1e-6, abs=1e-7)


# --------------------------------------------------------------------------- #
# 5) the source-chain block means what it says
# --------------------------------------------------------------------------- #
def test_src_chain_matches_a_hand_walk_of_the_shuffle_chain(store) -> None:
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6b")
    ix = enc._ch_index
    seen_fresh = seen_a2 = seen_a3 = 0
    # Age-3 slots are only ~6% of a ga80/paramA core and many cores are pure
    # 2-cycle, so the sample has to be wide enough to reach the second-order
    # branch at all.
    for _, row in _sample(df, 30).iterrows():
        inp = RecordInputs.coerce(row)
        vals = enc.encode_slot_matrix(inp, fl)
        items = unpack_pattern(inp.pattern).items
        age, _origin, direct = enc._trace_chain(items)
        b = regime_cycle_burnup(inp.library_id, inp.feed)
        p = (vals[ix["prior_power"]] * _V6_POWER_SCALE + 1.0)
        present = vals[ix["shuffle_source_present"]]

        n_fresh = n_a2 = n_a3 = 0
        for slot in SLOTS:
            j = slot.index
            if present[j] <= 0.0:
                n_fresh += 1
                for ch in _V6B_SRC_CHAIN_EXTRA:
                    assert vals[ix[ch], j] == 0.0, (ch, j)
                continue
            s1 = direct[j]
            assert s1 is not None
            assert vals[ix["src1_present"], j] == 1.0
            assert vals[ix["src1_prior_power"], j] == pytest.approx(
                (p[s1] - 1.0) / _V6_POWER_SCALE, rel=1e-5, abs=1e-6)
            chain = float(p[s1])
            s2 = direct[s1] if present[s1] > 0.0 else None
            if s2 is None:
                n_a2 += 1
                assert vals[ix["src2_present"], j] == 0.0
                assert vals[ix["src2_radius"], j] == 0.0
                assert vals[ix["src2_prior_power"], j] == 0.0
            else:
                n_a3 += 1
                assert vals[ix["src2_present"], j] == 1.0
                assert vals[ix["src2_radius"], j] == pytest.approx(
                    SLOTS[s2].radius / _R_MAX, rel=1e-6)
                chain += float(p[s2])
            assert vals[ix["chain_bu_integral"], j] == pytest.approx(
                b * chain / _BURNUP_SCALE, rel=1e-5, abs=1e-6)
        seen_fresh += n_fresh
        seen_a2 += n_a2
        seen_a3 += n_a3
    # All three residence depths must be exercised SOMEWHERE in the sample -- a
    # single core may legitimately be 2-cycle and carry no age-3 slot, but a
    # sample spanning every library must hit the second-order branch, or this
    # test would silently never check it.
    assert seen_fresh > 0 and seen_a2 > 0 and seen_a3 > 0


def test_src_chain_beats_the_flat_encoding_on_spread(store) -> None:
    """The whole point: power-weighted chain burnup is NOT ``(age-1)*B``.

    If these two channels agreed slot-for-slot the arm would carry no new
    information, so this pins that they measurably disagree.
    """
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6b")
    ix = enc._ch_index
    diffs = []
    for _, row in _sample(df, 4).iterrows():
        vals = enc.encode_slot_matrix(RecordInputs.coerce(row), fl)
        burned = vals[ix["shuffle_source_present"]] > 0.0
        d = (vals[ix["chain_bu_integral"]] - vals[ix["nominal_burnup"]])[burned]
        diffs.append(np.abs(d) * _BURNUP_SCALE)          # back to GWd/tU
    all_d = np.concatenate(diffs)
    assert all_d.mean() > 0.5, "chain burnup is indistinguishable from (age-1)*B"


def test_src_chain_is_absent_from_every_other_schema() -> None:
    for s in ("v3", "v4", "v5", "v6", "v6_prior", "v6_contrast"):
        assert not (set(_V6B_SRC_CHAIN_EXTRA) & set(CHANNELS_BY_SCHEMA[s]))


# --------------------------------------------------------------------------- #
# 6) transpose-augment equivariance still holds for the new block
# --------------------------------------------------------------------------- #
def test_v6b_transpose_augment_is_finite_and_shaped(store) -> None:
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6b")
    row = df.iloc[0]
    cells, gvec = enc.encode(row, fl)
    tc, tg = enc.augment_transpose(cells, gvec, row, fl)
    assert tc.shape == cells.shape
    # Globals are transpose-invariant up to float32 summation order (the
    # fresh-slot weighted means accumulate in a different slot order after the
    # transpose).  That is pre-existing v2..v6 behaviour, not a v6b property.
    np.testing.assert_allclose(tg, gvec, rtol=1e-6, atol=1e-6)
    assert np.isfinite(tc).all()
