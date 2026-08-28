"""cond_schema v5 — the poison-agnostic channel swap, and v4's byte-identity.

v5 replaces the poison-SPECIFIC design channels (``origin_n_gd`` /
``origin_gd_wt`` / ``origin_gd_u_enr`` + the two Gd globals) with the
poison-AGNOSTIC k-conv curve-shape block, so the model keys on absorber
BEHAVIOUR and transfers to IFBA / Er / Dy.  ``v5_noshape`` is the ablation arm
(the removal WITHOUT the replacement), which is what separates "the shape
channels carry the signal" from "dropping Gd helped".

The load-bearing test here is :func:`test_v2_v3_v4_encoding_is_byte_identical`:
v5 is additive only if the existing schemas encode bit-for-bit as they did
before it landed.  The golden digests were captured from the pre-change
featurizer over a fixed 64-row slice of the real store.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from lpopt.data.fuel_types import FuelLibrary
from lpopt.model.featurize import (
    CHANNELS, CHANNELS_BY_SCHEMA, CHANNELS_V4, CHANNELS_V5,
    CHANNELS_V5_NOSHAPE, FeatureEncoder,
)

STORE = "data/store"
#: sha256 of the (cells, globals) float32 buffers for the first 64 record_ids in
#: sorted order, captured from the featurizer BEFORE the v5 work landed.
GOLDEN = {
    "v2": ("937072a2feccc320414df8e611a5f8b620f3f7b1e1402c161f3bab60ce713f72",
           "6557fbd932c608bcb7c3855a9d46382202e4aae47bb45f8f01f369f15f9fa8ae"),
    "v3": ("6b34d8eb12d656398c15444b33c258bd490c59467e527da6165a313eec53475e",
           "f4b12d5f34eaf6b79b69994974e5081992e0580884a5e12c319ec2486089c104"),
    "v4": ("55a0c564c7b5cad9eb16ce53c13c28c8e8340fa27a624557847f531f46cfba0a",
           "2c62132ccb4702fcb2f9c7c7caaf6a4abd2dd27b62950f14b48ee4cace9643e9"),
}
GD_CHANNELS = ("origin_n_gd", "origin_gd_wt", "origin_gd_u_enr")
GD_GLOBALS = ("g_fresh_mean_n_gd", "g_fresh_mean_gd_wt")
SHAPE_CHANNELS = (
    "origin_reactivity_swing", "origin_depletion_slope", "origin_bu_peak",
    "origin_bu_dip", "origin_rho_boc_minus_peak", "origin_kinf_eol50",
    "origin_kconv_monotone", "origin_kconv_present",
)


@pytest.fixture(scope="module")
def fuel():
    return FuelLibrary.from_parquet(f"{STORE}/fuel_types.parquet")


@pytest.fixture(scope="module")
def rows():
    # 골든 다이제스트의 기준 64행은 동결 스냅샷에서 로드한다. 라이브 스토어를
    # 정렬-샘플링하면 캠페인 병합으로 상위 64행 구성이 바뀌어 인코더가 무변경인데도
    # 골든이 깨진다 (2026-07-22 round1c 병합에서 실증).
    return pd.read_parquet("tests/data/v5_golden_rows.parquet")


def _digest(enc, rows, fuel):
    cells, gvec = enc.encode_batch(rows, fuel)
    return (hashlib.sha256(np.ascontiguousarray(cells).tobytes()).hexdigest(),
            hashlib.sha256(np.ascontiguousarray(gvec).tobytes()).hexdigest())


# --------------------------------------------------------------------------- #
# the additive contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("schema", ["v2", "v3", "v4"])
def test_v2_v3_v4_encoding_is_byte_identical(schema, rows, fuel):
    """Registering v5 must not perturb ANY existing schema by a single bit."""
    assert _digest(FeatureEncoder(cond_schema=schema), rows, fuel) == GOLDEN[schema]


def test_v4_channel_and_global_inventory_unchanged():
    assert len(CHANNELS_V4) == 43
    assert CHANNELS_V4[:26] == CHANNELS          # v4 is append-only over v2/v3
    enc = FeatureEncoder(cond_schema="v4")
    assert len(enc.globals_names) == 13
    for name in GD_CHANNELS:
        assert name in enc.channels
    for name in GD_GLOBALS:
        assert name in enc.globals_names


# --------------------------------------------------------------------------- #
# v5 inventory
# --------------------------------------------------------------------------- #
def test_v5_drops_every_poison_specific_channel_and_global():
    enc = FeatureEncoder(cond_schema="v5")
    for name in GD_CHANNELS:
        assert name not in enc.channels, f"{name} must not be a v5 model input"
    for name in GD_GLOBALS:
        assert name not in enc.globals_names
    # nothing Gd-flavoured survives under any spelling
    assert not [c for c in enc.channels if "gd" in c.lower()]
    assert not [g for g in enc.globals_names if "gd" in g.lower()]


def test_v5_adds_the_shape_block_and_its_two_globals():
    enc = FeatureEncoder(cond_schema="v5")
    for name in SHAPE_CHANNELS:
        assert name in enc.channels
    assert "g_fresh_mean_reactivity_swing" in enc.globals_names
    assert "g_fresh_mean_depletion_slope" in enc.globals_names
    # 43 - 3 dropped + 8 shape = 48 channels; 13 - 2 + 2 = 13 globals
    assert len(enc.channels) == 48 == len(CHANNELS_V5)
    assert len(enc.globals_names) == 13


def test_v5_keeps_every_other_v4_channel():
    """Only the three Gd axes may differ; every other v4 channel must survive."""
    kept = set(CHANNELS_V4) - set(GD_CHANNELS)
    assert kept <= set(CHANNELS_V5)


def test_v5_noshape_is_the_ablation_arm():
    enc = FeatureEncoder(cond_schema="v5_noshape")
    for name in GD_CHANNELS:
        assert name not in enc.channels
    for name in SHAPE_CHANNELS:
        assert name not in enc.channels, "the ablation arm must have NO shape block"
    assert len(enc.channels) == 40 == len(CHANNELS_V5_NOSHAPE)
    assert len(enc.globals_names) == 11
    assert tuple(enc.channels) == tuple(c for c in CHANNELS_V4 if c not in GD_CHANNELS)


def test_all_registered_schemas():
    # v6 / v6_contrast / v6_prior joined at the hires bundle (design doc
    # 20260725): v5 + the appended local-contrast and power-prior families.
    # v6b joined at the burnup-placement arm (ab2_addendum_BU_20260810.md):
    # v6 + the regime (library, feed) per-cycle burnup table + the source-chain
    # channels.  v6c joined at the ADF arm (ab2_addendum_ADF_20260810.md):
    # v6b + the face / corner-g1 assembly-discontinuity-factor block.  v7 joined
    # at the 3-fresh-type arm (tripletype_design_20260817.md): v6c's CELLS
    # verbatim + the appended composition-moment GLOBALS.  v8 joined at the
    # 5-fresh-type widening (same doc, addendum): v7's block with the per-type
    # fraction padding widened 3 -> 5.  This closed-set
    # assertion is a deliberate TRIPWIRE -- adding a
    # schema must be a conscious edit here, not something that lands silently.
    assert set(CHANNELS_BY_SCHEMA) == {
        "v2", "v3", "v4", "v5", "v5_noshape", "v6", "v6_contrast", "v6_prior",
        "v6b", "v6c", "v7", "v8"}


def test_unknown_schema_is_rejected():
    with pytest.raises(ValueError, match="unknown cond_schema"):
        FeatureEncoder(cond_schema="v99_not_a_schema")


def test_config_accepts_the_new_schemas(tmp_path):
    """``[model] cond_schema`` validation reads CHANNELS_BY_SCHEMA, so v5 is live."""
    from lpopt.config import _validate_cond_schema
    for schema in ("v4", "v5", "v5_noshape"):
        _validate_cond_schema(schema, tmp_path / "deck.inp")


# --------------------------------------------------------------------------- #
# the shape block actually carries signal
# --------------------------------------------------------------------------- #
def test_shape_channels_are_populated_and_non_degenerate(rows, fuel):
    """A dead (all-zero) channel would silently make the ablation meaningless."""
    enc = FeatureEncoder(cond_schema="v5")
    cells, _ = enc.encode_batch(rows, fuel)
    for name in SHAPE_CHANNELS:
        plane = cells[:, enc.channels.index(name)]
        assert np.count_nonzero(plane) > 0, f"{name} is identically zero"
    # the continuous ones must actually VARY (a constant channel is no feature)
    for name in ("origin_reactivity_swing", "origin_depletion_slope",
                 "origin_bu_peak", "origin_kinf_eol50"):
        plane = cells[:, enc.channels.index(name)]
        assert float(plane.std()) > 1e-6, f"{name} is constant"


def test_shape_channels_are_o1_normalized(rows, fuel):
    """Normalization constants must keep the block O(1), like every other channel."""
    enc = FeatureEncoder(cond_schema="v5")
    cells, _ = enc.encode_batch(rows, fuel)
    for name in SHAPE_CHANNELS:
        plane = cells[:, enc.channels.index(name)]
        assert np.abs(plane).max() < 10.0, f"{name} is not O(1)"


def test_kconv_present_gate_is_confined_to_the_fuel_grid(rows, fuel):
    enc = FeatureEncoder(cond_schema="v5")
    cells, _ = enc.encode_batch(rows, fuel)
    present = cells[:, enc.channels.index("origin_kconv_present")]
    fuel_mask = cells[:, enc.channels.index("fuel_mask")]
    assert np.all(present[fuel_mask == 0] == 0.0)     # never set off the fuel grid
    assert present.sum() > 0                          # and it does fire


def test_kconv_present_gate_distinguishes_harvested_from_absent():
    """The gate must be 0 for an origin with NO harvested curve, and the whole
    shape block must normalize to the 0 sentinel there — otherwise the network
    cannot tell a genuine centered-zero from an unharvested one."""
    from lpopt.data.fuel_types import FuelVec
    enc = FeatureEncoder(cond_schema="v5")
    ix = {n: i for i, n in enumerate(enc.channels)}
    vals = np.zeros((len(enc.channels), 1), dtype=np.float32)

    bare = FuelVec(library_id="L", type_id="T")           # nothing harvested
    enc._fill_v5_shape(vals, ix, 0, bare)
    assert vals[ix["origin_kconv_present"], 0] == 0.0
    for name in SHAPE_CHANNELS:
        assert vals[ix[name], 0] == 0.0

    harvested = FuelVec(library_id="L", type_id="T", kinf_peak=1.1461,
                        bu_peak_gwd=19.0, reactivity_swing_pcm=1046.8,
                        depletion_slope_pcm_per_gwd=-599.85, kinf_eol50=0.9566,
                        kconv_is_monotone=0.0, bu_dip_gwd=7.0,
                        rho_boc_minus_peak_pcm=46.6)
    enc._fill_v5_shape(vals, ix, 0, harvested)
    assert vals[ix["origin_kconv_present"], 0] == 1.0
    # median-centered inputs land at ~0 by construction; the gate is what
    # separates them from the absent case above.
    assert vals[ix["origin_reactivity_swing"], 0] == pytest.approx(1046.8 / 2500)
    assert vals[ix["origin_depletion_slope"], 0] == pytest.approx(
        (-599.85 + 600.0) / 130.0)


def test_monotone_curve_leaves_dip_and_swing_at_the_sentinel():
    """A weak-absorber (monotone) design has no hump: dip/swing are NaN by
    omission and must normalize to 0 while the monotone FLAG carries the state."""
    from lpopt.data.fuel_types import FuelVec
    enc = FeatureEncoder(cond_schema="v5")
    ix = {n: i for i, n in enumerate(enc.channels)}
    vals = np.zeros((len(enc.channels), 1), dtype=np.float32)
    mono = FuelVec(library_id="L", type_id="X0", kinf_peak=1.13,
                   bu_peak_gwd=0.0, reactivity_swing_pcm=float("nan"),
                   bu_dip_gwd=float("nan"), kconv_is_monotone=1.0,
                   depletion_slope_pcm_per_gwd=-470.0, kinf_eol50=0.95)
    enc._fill_v5_shape(vals, ix, 0, mono)
    assert vals[ix["origin_kconv_monotone"], 0] == 1.0
    assert vals[ix["origin_reactivity_swing"], 0] == 0.0
    assert vals[ix["origin_bu_dip"], 0] == 0.0
    assert vals[ix["origin_kconv_present"], 0] == 1.0     # the curve IS harvested


def test_v5_encoding_is_deterministic(rows, fuel):
    a = _digest(FeatureEncoder(cond_schema="v5"), rows, fuel)
    b = _digest(FeatureEncoder(cond_schema="v5"), rows, fuel)
    assert a == b


def test_v5_and_v5_noshape_differ(rows, fuel):
    """Sanity: the ablation must not accidentally encode the same thing."""
    assert (_digest(FeatureEncoder(cond_schema="v5"), rows, fuel)
            != _digest(FeatureEncoder(cond_schema="v5_noshape"), rows, fuel))


def test_v5_cond_norm_carries_a_distinguishing_marker():
    """A v5 checkpoint must never be silently loadable as a v4 one."""
    v4 = FeatureEncoder(cond_schema="v4").cond_norm
    v5 = FeatureEncoder(cond_schema="v5").cond_norm
    assert v4["cond_schema"] == "v4" and v5["cond_schema"] == "v5"
    # the (enrichment, feed, depth-2) envelope is deliberately IDENTICAL
    for key in ("enr_ref", "enr_scale", "feed_center", "feed_center_scale",
                "depth2_scale"):
        assert v4[key] == v5[key]
