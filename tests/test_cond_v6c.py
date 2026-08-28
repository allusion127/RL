"""cond_v6c — the ADF arm (`data/reports/ab2_addendum_ADF_20260810.md`).

Same two obligations as the v6b suite, for the same reason:

1. **v6b did not move.**  The control is the standing champion `split_S1b`, a
   v6b model.  If this change perturbed a single v6b feature bit, every paired
   difference measured against that control would be measuring the refactor.

2. **v6c changes exactly what it claims to** — four appended channels and
   nothing inside v6b's 58.

Plus one obligation specific to this arm: the addendum's §1 finding — that
`origin_adf_corner_g2`, `origin_cr1_worth` and `origin_ff_pin_max` were ALREADY
in v6b — is load-bearing for how the secondary readout is interpreted, so it is
pinned here as a test rather than left as prose that could drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpopt.data.fuel_types import FuelLibrary
from lpopt.data.schema import unpack_pattern
from lpopt.model.featurize import (
    CHANNELS_BY_SCHEMA,
    CHANNELS_V6B,
    CHANNELS_V6C,
    _V6C_ADF_EXTRA,
    _V6C_ADF_SCALED,
    FeatureEncoder,
    RecordInputs,
)
from lpopt.vendor.masterrl.domain import SLOTS

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"
RECORDS = STORE / "records.parquet"
FUEL = STORE / "fuel_types.parquet"

#: The three quantities v6b ALREADY carried.  Pinned because the ADF arm's
#: interpretation depends on it (addendum §1 / §5.6): the control is NOT blind
#: to H2/H4, so the H-family readout is a does-face-ADF-help test, not a
#: can-vs-cannot test.
_ALREADY_IN_V6B = ("origin_adf_corner_g2", "origin_cr1_worth",
                   "origin_ff_pin_max")


def _load():
    if not RECORDS.is_file() or not FUEL.is_file():
        pytest.skip("Dataset-A store not present")
    import pandas as pd

    return pd.read_parquet(RECORDS), FuelLibrary.from_parquet(FUEL)


def _sample(df, n_per_lib: int = 4):
    import pandas as pd

    return pd.concat([g.head(n_per_lib) for _, g in df.groupby("library_id")]
                     ).reset_index(drop=True)


@pytest.fixture(scope="module")
def store():
    return _load()


# --------------------------------------------------------------------------- #
# 1) inventory: append-only, and the §1 finding
# --------------------------------------------------------------------------- #
def test_v6c_is_append_only_after_v6b() -> None:
    assert len(CHANNELS_V6B) == 58
    assert CHANNELS_V6C[:58] == CHANNELS_V6B
    assert CHANNELS_V6C == CHANNELS_V6B + _V6C_ADF_EXTRA
    assert len(CHANNELS_V6C) == 62
    assert len(set(CHANNELS_V6C)) == len(CHANNELS_V6C)
    assert CHANNELS_BY_SCHEMA["v6c"] == CHANNELS_V6C
    # the map head's residual-skip plane must not have moved
    assert CHANNELS_V6C.index("prior_power") == 50


def test_the_three_adf_channels_v6b_already_had() -> None:
    """Addendum §1: the control is NOT structurally blind to H2/H4.

    If a future edit moves these into the v6c block, the secondary readout's
    interpretation changes and this test is where that must be noticed.
    """
    for name in _ALREADY_IN_V6B:
        assert name in CHANNELS_V6B, f"{name} was expected to predate v6c"
        assert name not in _V6C_ADF_EXTRA


def test_v6c_adds_only_the_face_and_corner_g1_adfs() -> None:
    assert _V6C_ADF_EXTRA == ("origin_adf_face_g1", "origin_adf_face_g2",
                              "origin_adf_corner_g1", "origin_adf_present")


def test_v6c_globals_identical_to_v6b() -> None:
    assert (FeatureEncoder(cond_schema="v6c").globals_names
            == FeatureEncoder(cond_schema="v6b").globals_names)


def test_adf_block_absent_from_every_earlier_schema() -> None:
    for s in ("v3", "v4", "v5", "v6", "v6_prior", "v6_contrast", "v6b"):
        assert not (set(_V6C_ADF_EXTRA) & set(CHANNELS_BY_SCHEMA[s]))


# --------------------------------------------------------------------------- #
# 2) v6b did not move
# --------------------------------------------------------------------------- #
def test_v6b_prefix_is_bit_identical_under_v6c(store) -> None:
    df, fl = store
    a = FeatureEncoder(cond_schema="v6b")
    b = FeatureEncoder(cond_schema="v6c")
    for _, row in _sample(df).iterrows():
        ca, ga = a.encode(row, fl)
        cb, gb = b.encode(row, fl)
        assert np.array_equal(ca, cb[:58]), "v6c perturbed a v6b channel"
        assert np.array_equal(ga, gb), "v6c perturbed the globals"


def test_v6c_shape_and_finiteness(store) -> None:
    df, fl = store
    cells, gvec = FeatureEncoder(cond_schema="v6c").encode(df.iloc[0], fl)
    assert cells.shape == (62, 19, 19)
    assert cells.dtype == np.float32
    assert gvec.shape == (13,)
    assert np.isfinite(cells).all() and np.isfinite(gvec).all()


# --------------------------------------------------------------------------- #
# 3) the ADF channels carry the right numbers
# --------------------------------------------------------------------------- #
def test_adf_channels_match_the_origin_vec(store) -> None:
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6c")
    ix = enc._ch_index
    checked = 0
    for _, row in _sample(df, 3).iterrows():
        inp = RecordInputs.coerce(row)
        vals = enc.encode_slot_matrix(inp, fl)
        items = unpack_pattern(inp.pattern).items
        _age, origin, _d = enc._trace_chain(items)
        for slot in SLOTS:
            j = slot.index
            vec = enc._resolve_vec(fl, items[origin[j]].batch, inp.library_id)
            if vec is None:
                continue
            for ch, attr, ref, scale in _V6C_ADF_SCALED:
                raw = getattr(vec, attr, None)
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    assert vals[ix[ch], j] == 0.0
                else:
                    assert vals[ix[ch], j] == pytest.approx(
                        (float(raw) - ref) / scale, rel=1e-6, abs=1e-7)
                    checked += 1
    assert checked > 0, "no ADF value was actually exercised"


def test_presence_gate_is_exact_for_the_whole_block(store) -> None:
    """All six ADF/CR/FF columns share one coverage pattern, so ONE gate is exact."""
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6c")
    ix = enc._ch_index
    seen_present = seen_absent = 0
    for _, row in _sample(df, 4).iterrows():
        inp = RecordInputs.coerce(row)
        vals = enc.encode_slot_matrix(inp, fl)
        items = unpack_pattern(inp.pattern).items
        _a, origin, _d = enc._trace_chain(items)
        for slot in SLOTS:
            j = slot.index
            vec = enc._resolve_vec(fl, items[origin[j]].batch, inp.library_id)
            gate = vals[ix["origin_adf_present"], j]
            assert gate in (0.0, 1.0)
            if gate == 1.0:
                seen_present += 1
                # a present gate implies every scaled channel came from a real value
                assert vec is not None
                for _ch, attr, _r, _s in _V6C_ADF_SCALED:
                    v = getattr(vec, attr, None)
                    assert v is not None and not np.isnan(float(v))
            else:
                seen_absent += 1
                for ch, _a2, _r, _s in _V6C_ADF_SCALED:
                    assert vals[ix[ch], j] == 0.0
    assert seen_present > 0


def test_adf_channels_are_not_dead_and_vary(store) -> None:
    """A constant channel is no feature; a dead one makes the arm meaningless."""
    df, fl = store
    enc = FeatureEncoder(cond_schema="v6c")
    cells, _ = enc.encode_batch(_sample(df, 6), fl)
    for ch, _attr, _ref, _scale in _V6C_ADF_SCALED:
        plane = cells[:, enc.channels.index(ch)]
        assert np.count_nonzero(plane) > 0, f"{ch} is identically zero"
        assert float(plane.std()) > 1e-6, f"{ch} is constant"


def test_adf_channels_separate_the_H_family_treated_pair(store) -> None:
    """H2 vs H4 must differ on the NEW channels, H1 vs H3 must not.

    This is the STEP 0 contrast in feature space: if the encoding cannot tell H2
    from H4, the §5.6 readout is vacuous before a model is even trained.
    """
    _df, fl = store
    enc = FeatureEncoder(cond_schema="v6c")
    got = {}
    for t in ("H1", "H2", "H3", "H4"):
        vec = enc._resolve_vec(fl, t, "ga80")
        if vec is None:
            pytest.skip("ga80 H-family not in the fuel table")
        got[t] = {ch: (float(getattr(vec, attr)) - ref) / scale
                  for ch, attr, ref, scale in _V6C_ADF_SCALED}
    treated = max(abs(got["H2"][c] - got["H4"][c]) for c in got["H2"])
    control = max(abs(got["H1"][c] - got["H3"][c]) for c in got["H1"])
    assert treated > 0.15, f"treated pair barely separates ({treated:.4f})"
    assert control < 0.05, f"negative control pair separates ({control:.4f})"
    assert treated > 5 * control
