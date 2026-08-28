"""Leakage-safe physics featurization of a loading pattern (plan sec. 4.4).

:class:`FeatureEncoder` turns one store record into a pair

* ``cells``   — a ``float32[C, 19, 19]`` stack of per-position channels, and
* ``globals`` — a ``float32[G]`` FiLM conditioning vector.  Its (enrichment,
  feed, depth-2) normalization follows the encoder's ``cond_schema`` — ``cond_v2``
  (fixed feed=121) or the Phase D default ``cond_v3`` (expanded envelope,
  plan sec. 12.4); the two differ only in those centering/scale constants.

The southeast quarter's 69 ``%LPD_SHF`` slots are mirror-expanded to the full
17x17 fuel grid and zero-padded to 19x19; a reflector-ring channel marks the
pad cells adjacent to fuel.

**Leakage rule (plan sec. 4.4, hard):** the encoding depends *only* on
``(pattern, feed, e_core, e_split, case_pair, library_id, sym_class, dataset)``
plus the static ``fuel_types`` table.  It never reads a record's target /
metric / map columns — the burn state is a strict *a-priori* quantity (source
residence age and a nominal per-cycle burnup constant), never the record's own
``cycle_burnup`` or EDIT5 labels.  :class:`RecordInputs` is the enforcement
surface: only the allow-listed fields are ever pulled off a record row, so a
row carrying labels featurizes byte-identically to one with them stripped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.fuel_types import FuelLibrary, FuelVec, core_enrichment_split
from ..data.geometry import cell_of_slot, transpose
from ..data.schema import unpack_pattern
from ..vendor.masterrl.domain import SLOTS, Pattern
from ..vendor.masterrl.ga import _coord_slot


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: Nominal accumulated burnup per completed residence cycle [MWd/kgHM].  This is
#: an *a-priori* physics constant (plan sec. 4.4), deliberately NOT the record's
#: own measured ``cycle_burnup`` — using the label would leak the target.
NOMINAL_CYCLE_BURNUP_MWD_KG: float = 22.0

#: --- cond_v6b: the REGIME per-cycle burnup table ---------------------------- #
#: The single 22.0 constant above is wrong in every regime the store contains.
#: Measured implied constants over 12,174 trajectory cores
#: (``data/reports/kcurve_fusion_memo_20260809.md`` §3(0)): the burned-slot
#: placement error averages 12.18 GWd/tU == 2801 pcm of core-mean reactivity,
#: which is 5.1x the 4-scalar k-curve compression error and exceeds a whole
#: fuel-swap effect in 45.8% of cores.  The error is SYSTEMATIC in
#: ``(library_id, feed)`` and both of those are already model inputs, so it is a
#: pure encoding defect rather than missing information.
#:
#: These are HARD-CODED CONSTANTS, not a fit: nothing here reads a label at
#: encode time, so the table is a-priori by construction and the leakage rule
#: (module docstring) is untouched.  Only a ``cond_schema`` carrying the
#: ``v6b_burnup`` marker consults it; every v2..v6 encoder keeps the flat 22.0
#: and is byte-identical.
_REGIME_CYCLE_BURNUP_MWD_KG: dict[str, dict[int, float]] = {
    "ga80":   {101: 24.94, 121: 28.69, 141: 31.14},
    "paramA": {101: 30.47, 121: 35.33, 141: 36.77},
}


def regime_cycle_burnup(library_id: Any, feed: Any) -> float:
    """A-priori per-cycle burnup [MWd/kgHM] for one ``(library_id, feed)`` regime.

    **Fallback chain, in order** (documented here because the fallbacks decide
    what 54.9% of the store sees):

    1. **Exact hit** on ``(library, feed)`` -- one of the six measured constants.
    2. **Linear interpolation in ``feed`` within the same library.**  The three
       measured anchors (101 / 121 / 141) span the whole feed envelope the store
       uses, so every feed a real record carries is interpolated, never invented.
    3. **That library's mean**, for a feed OUTSIDE the measured hull.  Deliberately
       not an extrapolation: a two-point linear extrapolation off a three-point
       table has no support in the measurement and can run to absurd values at the
       envelope edges, whereas the mean is bounded by what was measured.
    4. **The legacy :data:`NOMINAL_CYCLE_BURNUP_MWD_KG` (22.0)** for a library that
       is not in the table at all (260624 / 5.8_5.1 / legacy_a).  Fitting those
       libraries a constant from the ga80+paramA numbers would be exactly the kind
       of unregistered inference the A/B protocol exists to refuse, so they keep
       today's encoding and only the source-chain channels change for them.
    """
    table = _REGIME_CYCLE_BURNUP_MWD_KG.get(str(library_id))
    if not table:
        return NOMINAL_CYCLE_BURNUP_MWD_KG
    try:
        f = float(feed)
    except (TypeError, ValueError):
        return NOMINAL_CYCLE_BURNUP_MWD_KG
    # isfinite, not just isnan: an inf would reach ``int(f)`` below and raise
    # OverflowError from inside the encoder, killing a training run over a bad
    # cell in one row.  Any non-finite feed falls back to the legacy constant.
    if not math.isfinite(f):
        return NOMINAL_CYCLE_BURNUP_MWD_KG
    feeds = sorted(table)
    exact = table.get(int(f)) if float(int(f)) == f else None
    if exact is not None:
        return float(exact)
    if feeds[0] <= f <= feeds[-1]:
        return float(np.interp(f, np.asarray(feeds, dtype=float),
                               np.asarray([table[k] for k in feeds], dtype=float)))
    return float(sum(table.values()) / len(table))


def schema_uses_regime_burnup(cond_schema: str) -> bool:
    """True when ``cond_schema`` evaluates the burn state on the regime table.

    Exported so the trainer can fit the power prior's ``(M^2, extrap)`` on the
    SAME burn state the encoder will serve on, without importing the marker dict.
    """
    return bool(_COND_NORM.get(str(cond_schema), {}).get("v6b_burnup"))


# --- schema-independent normalization constants (shared by cond_v2 / cond_v3) --
_N_GD_SCALE = 24.0      # max Gd-pin count normalizer
_GD_WT_SCALE = 10.0     # max Gd2O3 wt% normalizer
_ESPLIT_SCALE = 0.5     # pair enrichment-spread normalizer [w/o]
_FEED_SCALE = 241.0     # full-core assembly count
_AGE_SCALE = 3.0        # residence-age normalizer (fresh=1 .. thrice=3)
#: 3-cycle nominal burnup.  PINNED to the legacy ``22.0 * 3`` and deliberately NOT
#: made regime-dependent: it is a fixed normalizer, not a physics quantity.  Two
#: consequences, both wanted -- (a) every v2..v6 encoder is byte-identical, and
#: (b) under cond_v6b the ``nominal_burnup`` channel actually CARRIES the regime
#: constant.  (Had the scale moved with the constant, ``(age-1)*B / (3*B)``
#: would have cancelled it out and the table would have been a silent no-op.)
_BURNUP_SCALE = 22.0 * 3.0

# --- conditioning-schema-dependent normalization constants --------------------
# The (enrichment, feed, depth-2) centering/scale is the ONLY thing that differs
# between cond_v2 (the fixed feed=121 / ~5.4 w/o campaign) and cond_v3 (the
# Phase D expanded envelope: e_core 5.0-6.5, feed 101-141; plan sec. 12.4).
# Everything else in the cell/global inventory is schema-invariant.  The active
# set is selected per :class:`FeatureEncoder` instance from its ``cond_schema``
# and stamped into the training checkpoint so a served pattern re-normalizes
# byte-identically (and a v2 checkpoint keeps loading against the v2 constants).
#
#   cond_v3 (plan sec. 12.4):  e_core (x-5.75)/1.5,  feed (x-121)/20,
#   depth2 (60-2N)/20  (feed 101 -> N=25 -> 10 depth-2 edges; feed 141 -> N=35
#   -> -10, i.e. single-cycle-discharge units).
#
#   cond_v4 (plan sec. 4.4 feature expansion): the (enrichment, feed, depth-2)
#   envelope is UNCHANGED from v3 — v4 only *adds* result-based channels/globals
#   (the reference-depletion k-inf curve + BOC branch/XS/ADF signatures harvested
#   into ``fuel_types``).  Its normalization constants are therefore v3's, plus a
#   marker so a v4 checkpoint is never confused with a v3 one at load time.
_COND_NORM: dict[str, dict[str, float]] = {
    "v2": {
        "enr_ref": 5.4, "enr_scale": 0.6,
        "feed_center": 113.0, "feed_center_scale": 8.0,
        "depth2_scale": 16.0,
    },
    "v3": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
    },
    "v4": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0,          # marker: expanded result-based inventory
    },
    #   cond_v5 (data/reports/kinf_shape_features.md "v5 channel plan"): the
    #   (enrichment, feed, depth-2) envelope is UNCHANGED from v3/v4.  v5 only
    #   swaps the POISON-SPECIFIC design channels (n_gd / gd_wt / gd_u_enr and
    #   the two Gd globals) for the POISON-AGNOSTIC k-conv curve-shape block, so
    #   the model keys on absorber BEHAVIOUR rather than absorber identity and
    #   generalizes to IFBA / Er / Dy.  Markers keep a v5 checkpoint from ever
    #   being confused with a v4 one at load time.
    "v5": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0,
        "v5_features": 1.0,          # marker: poison-agnostic inventory
    },
    #   cond_v5_noshape — the ABLATION arm.  Identical to v5 except the k-conv
    #   shape block is absent, i.e. it is exactly "v4 minus the Gd channels".
    #   Training this arm alongside v5 is what separates "the shape channels
    #   carry the poison signal" from "merely removing Gd helped".
    "v5_noshape": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0,
        "v5_features": 1.0,
        "v5_no_shape": 1.0,          # marker: ablation (shape channels removed)
    },
    # --- cond_v6 (hires bundle): v5 constants + family markers.  The three v6
    # schemas differ ONLY in which appended channel family is active, so they
    # share every normalization constant with v5 and a v5 checkpoint's cond_norm
    # comparison stays meaningful.
    "v6": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0, "v5_features": 1.0,
        "v6_contrast": 1.0, "v6_prior": 1.0,
    },
    "v6_contrast": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0, "v5_features": 1.0,
        "v6_contrast": 1.0,
    },
    "v6_prior": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0, "v5_features": 1.0,
        "v6_prior": 1.0,
    },
    # --- cond_v6b (burnup-placement arm, data/reports/ab2_addendum_BU_20260810.md)
    # v6 + (a) the regime ``(library_id, feed)`` per-cycle burnup table and (b) the
    # appended source-chain channels.  Every v6 normalization constant is
    # inherited unchanged and the two families are APPEND-ONLY after v6's 52
    # channels, so a v6 checkpoint keeps loading and serving on its own 52.  The
    # two markers are separate so a future arm can take one half without the other.
    "v6b": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0, "v5_features": 1.0,
        "v6_contrast": 1.0, "v6_prior": 1.0,
        "v6b_burnup": 1.0,          # marker: regime (library, feed) burnup table
        "v6b_srcchain": 1.0,        # marker: appended source-chain channels
    },
    # --- cond_v6c (ADF arm, data/reports/ab2_addendum_ADF_20260810.md)
    # v6b + the face/corner-g1 ADF block, append-only after v6b's 58 channels.
    # Every v6/v6b normalization constant is inherited unchanged, so a v6b
    # checkpoint's cond_norm comparison stays meaningful and it keeps loading.
    "v6c": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0, "v5_features": 1.0,
        "v6_contrast": 1.0, "v6_prior": 1.0,
        "v6b_burnup": 1.0, "v6b_srcchain": 1.0,
        "v6c_adf": 1.0,             # marker: appended face/corner-g1 ADF block
    },
    # --- cond_v7 (3-fresh-type arm, data/reports/tripletype_design_20260817.md)
    # v6c + the APPENDED composition-moment globals.  Every v6c normalization
    # constant is inherited unchanged and the CELL inventory is v6c's verbatim
    # (a per-slot channel already carries its own origin type, so growing the
    # fresh-type alphabet needs no new channel) -- only the GLOBAL vector grows,
    # 13 -> 18.  The two pair-specific globals (``g_e_split`` = |e_A - e_B|,
    # ``g_split_frac`` = the first member's feed fraction) cannot describe a
    # 3-type feed; v7 appends the per-type fractions padded to 3 plus the
    # feed-weighted enrichment standard deviation, which is what separates
    # {5.0, 5.7} 50/50 from {5.0, 5.35, 5.7} in thirds (same mean, same
    # max-min).  A 2-type record featurizes byte-identically on the v6c prefix.
    "v7": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0, "v5_features": 1.0,
        "v6_contrast": 1.0, "v6_prior": 1.0,
        "v6b_burnup": 1.0, "v6b_srcchain": 1.0,
        "v6c_adf": 1.0,
        "v7_composition": 1.0,      # marker: appended composition-moment globals
    },
    # --- cond_v8 (5-fresh-type arm, data/reports/tripletype_design_20260817.md
    # addendum).  IDENTICAL to v7 except that the per-type fraction block is
    # padded to 5 instead of 3 and ``g_n_fresh_types`` is normalized on 5.  The
    # cells are still v6c's verbatim and the v6c GLOBAL prefix (13) is still
    # untouched, so v8 is the same additive family v7 is -- it simply widens the
    # composition block to the operator-directed 3~5-type mesh.  v7 is left
    # exactly as it is (a v7 retrain is in flight); a v8 run is its own retrain.
    "v8": {
        "enr_ref": 5.75, "enr_scale": 1.5,
        "feed_center": 121.0, "feed_center_scale": 20.0,
        "depth2_scale": 20.0,
        "v4_features": 1.0, "v5_features": 1.0,
        "v6_contrast": 1.0, "v6_prior": 1.0,
        "v6b_burnup": 1.0, "v6b_srcchain": 1.0,
        "v6c_adf": 1.0,
        "v8_composition": 1.0,      # marker: 5-wide composition-moment globals
    },
}
#: Default conditioning schema for a freshly-constructed encoder (Phase D).  v4 is
#: opt-in at retrain time via an explicit ``cond_schema="v4"`` (plan: flip DEFAULT
#: only once a v4 population is harvested + a v4 ensemble is trained).
DEFAULT_COND_SCHEMA = "v3"

# Legacy module-level aliases (cond_v2 values) kept for any external importer;
# the encoder itself reads its instance-level, schema-selected constants.
_ENR_REF = _COND_NORM["v2"]["enr_ref"]
_ENR_SCALE = _COND_NORM["v2"]["enr_scale"]
_FEED_CENTER = _COND_NORM["v2"]["feed_center"]
_FEED_CENTER_SCALE = _COND_NORM["v2"]["feed_center_scale"]
_DEPTH2_SCALE = _COND_NORM["v2"]["depth2_scale"]

# --- cond_v4 result-based-feature normalization (single source of truth) -------
# k-inf-curve + U-mass + BOC pin-form-function centering/scaling.  The k-inf
# points center on criticality (k=1) with a 0.25 spread so a BOC ~1.2 lands ~0.8
# and an EOL ~0.8 lands ~-0.8 (harvested kinf0/10/20/30 in [1.06, 1.26] -> [0.2,
# 1.0], O(1)); bu_k1 (GWd/tU where k crosses 1.0, harvested [38.5, 44.4]) is
# scaled by a nominal 30 -> [1.28, 1.48].  U-mass is per-ASSEMBLY grams (the
# parquet stores 138.1-139.5 g, VERIFIED n=48; NOT the ~450 kg full-core basis);
# FINALIZED to ref=median(138.8) / scale=half-range(0.7) so the O(0.1)-wide
# population spans O(1) instead of the near-dead ~0.2 the initial (139, 6) gave.
_KINF_REF: float = 1.0
_KINF_SCALE: float = 0.25
_BU_K1_SCALE: float = 30.0        # bu_k1 [GWd/tU] -> bu / 30  (no centering)
_FF_REF: float = 1.0              # BOC pin form-function max centers on flat 1.0
_FF_SCALE: float = 0.30           # harvested ff_pin_max [1.10, 1.20] -> [0.34, 0.67]
_U_MASS_REF: float = 138.8        # per-assembly gU: median of parquet [138.1, 139.5]
_U_MASS_SCALE: float = 0.7        # half-range of the harvested population (n=48)

#: (ref, scale) for the BOC branch / macro-XS / ADF signatures — FINALIZED from
#: the harvested population (data/reports/fuel_types_v4_harvest.md, n=84; zone
#: n=48): ref = median, scale ~= 2*std ~= half-range, rounded, so each channel is
#: O(1) over the ACTUAL population.  ONE dict keeps every constant single-source.
#: (For zone_pins the population is strictly bimodal {52, 100}; centering on the
#:  non-occurring median 76 with a 24 half-range maps z1 -> -1.0, z2 -> +1.0, and
#:  the ga80/legacy absent-value to the 0 sentinel — no present/absent collision.)
_V4_SCALES: dict[str, tuple[float, float]] = {
    "boron_worth": (-5.5, 0.3),       # pcm/ppm at BOC   [-5.91, -5.34], med -5.49
    "doppler": (-2.0, 0.2),           # pcm/K at BOC     [-2.13, -1.78], med -1.98
    "mtc_dmod": (105.0, 25.0),        # pcm per 0.01 g/cc [72.2, 122.9], med 104.7
    "cr1_worth": (12400.0, 700.0),    # pcm at BOC   [11456, 12872], med 12414
    "zone_pins": (76.0, 24.0),        # zoning (UO2_2) pin count, bimodal {52, 100}
    "xs_a2": (0.109, 0.005),          # group-2 macro Sigma_a [0.103, 0.114]
    "xs_nf2": (0.152, 0.01),          # group-2 macro nuSigma_f [0.142, 0.161]
    "xs_s12": (0.0167, 0.00025),      # 1->2 scatter [0.01644, 0.01697]
    "adf_corner_g2": (1.236, 0.07),   # group-2 corner ADF [1.169, 1.305]
}

_GRID = 19              # padded grid side
_CORE = 17             # full-core fuel grid side
_CORE_CENTER = 8       # full-core centre index (0-based, in the 17x17 core)
_PAD = 1               # reflector pad ring thickness
_GRID_CENTER = _PAD + _CORE_CENTER   # centre index in the 19x19 grid == 9

_R_MAX = max(slot.radius for slot in SLOTS)   # largest quarter-slot radius


# --------------------------------------------------------------------------- #
# channel / global inventories (order-stable)
# --------------------------------------------------------------------------- #
#: Ordered cell-channel names.  ``encode`` returns ``float32[len(CHANNELS),19,19]``.
CHANNELS: tuple[str, ...] = (
    # masks (2)
    "fuel_mask",
    "reflector_mask",
    # occupancy (2)
    "occ_fresh",
    "occ_burned",
    # origin-type physics traced to the fresh origin of the shuffle chain (9)
    "origin_enrichment",        # (e - enr_ref) / enr_scale  (cond-schema constants)
    "origin_n_gd",              # n_gd / 24
    "origin_gd_wt",             # gd_wt / 10
    "origin_axial_z2",          # 1.0 for z2 axial zoning
    "origin_feature_poor",      # 1.0 when Gd design unknown / type unresolved
    "origin_kinf_present",      # v4: 1.0 when kinf harvested; v2/v3: dormant 0
    "origin_kinf0",             # v4: (k-1)/0.25; v2/v3: dormant 0 (kinf is v4)
    "origin_kinf20",            # v4: (k-1)/0.25; v2/v3: dormant 0
    "origin_bu_k1",             # v4: bu/30;      v2/v3: dormant 0
    # a-priori burn state (2)
    "residence_age",            # age / 3  (1=fresh, 2=once-burned, 3=twice)
    "nominal_burnup",           # (age-1) * 22 / 66  (a-priori, NOT the label)
    # chain-radius sequence (2)
    "chain_source_radius",      # direct-source slot radius / r_max
    "chain_displacement",       # |r_source - r_dest| / r_max
    # shuffle geometry (5)
    "shuffle_source_present",
    "shuffle_src_x",            # source quarter qi / 9
    "shuffle_src_y",            # source quarter qj / 9
    "shuffle_rot1",             # vertical-axis shuffle one-hot
    "shuffle_rot2",             # interior / horizontal shuffle one-hot
    # position (4)
    "pos_radius",               # slot radius / r_max
    "pos_multiplicity",         # orbit multiplicity / 4
    "pos_on_axis",              # 1.0 on either symmetry axis
    "pos_center",               # 1.0 at the core centre
)
#: cond_v4 result-based channels, APPENDED after the v2/v3 26-tuple (append-only,
#: index-stable — the first 26 indices are byte-identical to v2/v3 so a v3
#: checkpoint keeps serving on its 26 channels).  Each is traced to the fresh
#: chain origin exactly like the existing ``origin_*`` channels; a None/NaN
#: underlying value normalizes to 0.0 (presence semantics), and
#: ``origin_lattice_present`` gates whether the lattice-harvested block is real.
_V4_EXTRA: tuple[str, ...] = (
    # design axes not yet featurized in v2/v3 (5)
    "origin_enr_main",          # (e - enr_ref) / enr_scale  (schema envelope)
    "origin_enr_zone",          # (e - enr_ref) / enr_scale
    "origin_gd_u_enr",          # (e - enr_ref) / enr_scale
    "origin_u_mass",            # (m - 138.8) / 0.7   per-assembly gU
    "origin_zone_pins",         # (zone_pin_count - 76) / 24  (bimodal {52,100})
    # reference-depletion k-inf curve completion (2)  [kinf0/kinf20 are v2/v3]
    "origin_kinf10",            # (k - 1.0) / 0.25
    "origin_kinf30",            # (k - 1.0) / 0.25
    # BOC behavioural signatures harvested from .sum/HGC (9)
    "origin_ff_pin_max",        # (f - 1.0) / 0.30   BOC pin form-function max
    "origin_boron_worth",       # _V4_SCALES
    "origin_doppler",           # _V4_SCALES
    "origin_mtc_dmod",          # _V4_SCALES
    "origin_cr1_worth",         # _V4_SCALES
    "origin_xs_a2",             # _V4_SCALES  (group-2 macro Sigma_a)
    "origin_xs_nf2",            # _V4_SCALES  (group-2 macro nuSigma_f)
    "origin_xs_s12",            # _V4_SCALES  (1->2 scatter)
    "origin_adf_corner_g2",     # _V4_SCALES  (group-2 corner ADF)
    # presence gate for the whole lattice-harvested block (1)
    "origin_lattice_present",   # 1.0 when fuel_types.kinf0 is finite
)

#: (channel, FuelVec attr, _V4_SCALES key) for the branch/XS/ADF signatures.
_V4_ORIGIN_SCALED: tuple[tuple[str, str, str], ...] = (
    ("origin_boron_worth", "boron_worth", "boron_worth"),
    ("origin_doppler", "doppler_coef", "doppler"),
    ("origin_mtc_dmod", "mtc_dmod", "mtc_dmod"),
    ("origin_cr1_worth", "cr1_worth", "cr1_worth"),
    ("origin_zone_pins", "zone_pin_count", "zone_pins"),
    ("origin_xs_a2", "xs_a2", "xs_a2"),
    ("origin_xs_nf2", "xs_nf2", "xs_nf2"),
    ("origin_xs_s12", "xs_s12", "xs_s12"),
    ("origin_adf_corner_g2", "adf_corner_g2", "adf_corner_g2"),
)

#: v2/v3 keep the base 26-tuple; v4 appends ``_V4_EXTRA``.  ``FeatureEncoder``
#: selects its active channel list (and index / geometry-slot matrix) by schema.
CHANNELS_V4: tuple[str, ...] = CHANNELS + _V4_EXTRA

# --------------------------------------------------------------------------- #
# cond_v5: the poison-AGNOSTIC channel swap
# --------------------------------------------------------------------------- #
#: Poison-SPECIFIC design channels dropped at v5 (user directive 2026-07-20,
#: data/reports/kinf_shape_features.md).  These encode *which absorber design*
#: an assembly carries (Gd pin count / Gd2O3 wt% / the Gd-pin U enrichment), so a
#: model that keys on them cannot transfer to an IFBA / Er / Dy assembly.  The
#: columns stay in ``fuel_types`` as QC bookkeeping — only the model channel
#: SELECTION drops them.
_V5_DROPPED_CHANNELS: frozenset[str] = frozenset({
    "origin_n_gd", "origin_gd_wt", "origin_gd_u_enr",
})
#: Poison-specific GLOBALS dropped at v5 (the feed-weighted fresh Gd means).
_V5_DROPPED_GLOBALS: frozenset[str] = frozenset({
    "g_fresh_mean_n_gd", "g_fresh_mean_gd_wt",
})

#: The poison-agnostic k-conv curve-shape channels that REPLACE the Gd block.
#: They summarise the absorber holddown -> burnout-release SIGNATURE read off the
#: reference k-inf(BU) depletion curve in reactivity space, so the same
#: descriptors apply to any absorber chemistry.  Traced to the fresh chain origin
#: exactly like every other ``origin_*`` channel; None/NaN -> 0.0 with
#: ``origin_kconv_present`` as the block's presence gate.
_V5_SHAPE_EXTRA: tuple[str, ...] = (
    "origin_reactivity_swing",     # swing / 2500                (holddown release)
    "origin_depletion_slope",      # (slope + 600) / 130          (burnout decay)
    "origin_bu_peak",              # (bu_peak - 19) / 12          (burnout timing)
    "origin_bu_dip",               # (bu_dip - 7) / 7             (trough timing)
    "origin_rho_boc_minus_peak",   # rho_bmp / 2800               (suppression depth)
    "origin_kinf_eol50",           # (k - 0.957) / 0.05           (discharge k)
    "origin_kconv_monotone",       # {0,1} has-no-hump gate       (absorber strength)
    "origin_kconv_present",        # {0,1} presence gate for the block
)

#: (channel, FuelVec attr, ref, scale) for the shape block.  Constants are
#: ``ref = median`` / ``scale ~= robust half-span`` of the harvested population
#: (data/reports/kinf_shape_features.md "Measured distributions", n=92..117), so
#: every channel is O(1) over the ACTUAL population.  ONE tuple keeps them
#: single-source (the physics prior reads the raw columns, not these).
_V5_SHAPE_SCALED: tuple[tuple[str, str, float, float], ...] = (
    ("origin_reactivity_swing", "reactivity_swing_pcm", 0.0, 2500.0),
    ("origin_depletion_slope", "depletion_slope_pcm_per_gwd", -600.0, 130.0),
    ("origin_bu_peak", "bu_peak_gwd", 19.0, 12.0),
    ("origin_bu_dip", "bu_dip_gwd", 7.0, 7.0),
    ("origin_rho_boc_minus_peak", "rho_boc_minus_peak_pcm", 0.0, 2800.0),
    ("origin_kinf_eol50", "kinf_eol50", 0.957, 0.05),
    ("origin_kconv_monotone", "kconv_is_monotone", 0.0, 1.0),
)


def _drop(channels: tuple[str, ...], dropped: frozenset[str]) -> tuple[str, ...]:
    """``channels`` with ``dropped`` removed, order otherwise preserved."""
    return tuple(n for n in channels if n not in dropped)


#: v5 = v4 - the 3 Gd design channels + the 8 curve-shape channels (43 -> 48).
CHANNELS_V5: tuple[str, ...] = (
    _drop(CHANNELS_V4, _V5_DROPPED_CHANNELS) + _V5_SHAPE_EXTRA
)
#: v5_noshape = v4 - the 3 Gd design channels ONLY (43 -> 40): the ablation arm
#: that isolates "the shape channels carry the signal" from "removing Gd helped".
CHANNELS_V5_NOSHAPE: tuple[str, ...] = _drop(CHANNELS_V4, _V5_DROPPED_CHANNELS)

# --------------------------------------------------------------------------- #
# cond_v6: the LOCAL-CONTRAST and POWER-PRIOR channel families (hires bundle)
# --------------------------------------------------------------------------- #
#: Local neighbour-contrast channels (design doc 20260725, arm A4 / report R5).
#: Every existing per-slot channel describes the assembly IN ISOLATION; nothing
#: in the input tells the network how a slot differs from its neighbours.  That
#: difference IS the high-spatial-frequency content the 20260725 report measured
#: the trunk to attenuate (per-mode amplitude correlation 0.897 -> 0.593 with
#: rising wavenumber), so it is supplied directly instead of being reconstructed.
_V6_CONTRAST_EXTRA: tuple[str, ...] = (
    "origin_kinf_contrast",     # (kinf_j - mean over face neighbours) / 0.05
    "origin_age_contrast",      # (age_j - mean over face neighbours) / 2
)
#: Diffusion power-prior channels (design doc 20260725, arm A2 / report R1).
#: ``prior_power`` is the leading-order relative assembly power from
#: :mod:`.power_prior`; ``prior_power_contrast`` is its neighbour contrast.  The
#: prior is a pure function of (pattern, feed, library) + the static fuel table —
#: the SAME leakage-safe input surface every other channel uses.
_V6_PRIOR_EXTRA: tuple[str, ...] = (
    "prior_power",              # (P_j - 1) / 0.4    (P normalized to core mean 1)
    "prior_power_contrast",     # (P_j - neighbour mean P) / 0.4
)
#: Normalization scales for the v6 block (O(1) over the harvested population).
_V6_KINF_CONTRAST_SCALE = 0.05
_V6_AGE_CONTRAST_SCALE = 2.0
_V6_POWER_SCALE = 0.4

#: v6 families, append-only after the v5 48-tuple so every v5 index is stable.
CHANNELS_V6_CONTRAST: tuple[str, ...] = CHANNELS_V5 + _V6_CONTRAST_EXTRA
CHANNELS_V6_PRIOR: tuple[str, ...] = CHANNELS_V5 + _V6_PRIOR_EXTRA
CHANNELS_V6: tuple[str, ...] = CHANNELS_V5 + _V6_CONTRAST_EXTRA + _V6_PRIOR_EXTRA

# --------------------------------------------------------------------------- #
# cond_v6b: the SOURCE-CHAIN burn-state channels
# --------------------------------------------------------------------------- #
#: Where the a-priori burn state actually lives (memo §3(0)).  On an equilibrium
#: core the previous-cycle slot of any assembly is a pure function of the pattern:
#: it is ``direct[s]``, the shuffle source the featurizer already traces, and the
#: cycle before that it sat in ``direct[direct[s]]``.  A slot's accumulated burnup
#: is therefore ``B_regime * sum_k P(src_k)`` -- power integrated over residence,
#: not a flat ``(age-1) * B``.  Regressing the TRUE slot BOC burnup on exactly
#: these quantities reaches out-of-fold |err| 2.011 GWd/tU / R^2 0.952, a 6x
#: reduction on today's constant; using CURRENT-slot features only reaches 7.787,
#: which is why no previous feature round found this.
#:
#: All six are pure functions of ``(pattern, feed, library)`` plus the static fuel
#: table -- the identical leakage-safe surface every other channel uses -- and
#: they are APPENDED after v6's 52 so indices 0..51 and all 13 globals are
#: byte-identical to v6.  A fresh slot has no source and leaves every one at 0,
#: which the two presence gates disambiguate from a genuine centered zero (the
#: ``origin_*_present`` convention).
#:
#: NOTE ``chain_source_radius`` (index 16) ALREADY carries ``radius(direct[s])``,
#: so only the second-order source's radius is new here; adding a duplicate of an
#: existing channel would be a perfectly collinear input.  Likewise the
#: chain-vs-flat burnup RESIDUAL is deliberately absent: it is exactly
#: ``chain_bu_integral - nominal_burnup`` and both are in the inventory.
#: ``chain_bu_integral`` is ``B_regime * sum_k P(src_k)``: burnup is power
#: integrated over residence, and the flat ``(age-1) * B`` throws the power
#: weights away.  Measured against the TRUE slot BOC burnup (``__traj`` plane 1 at
#: step 0) over 13,619 burned slots / 400 trajectory cores, 2026-08-10:
#:
#:     encoding                        mean |err|   median |err|   [GWd/tU]
#:     (age-1) * 22      (v6 today)      12.075        11.844
#:     (age-1) * B_regime               7.874         7.018
#:     B_regime * sum_k P(src_k)        8.188         5.309
#:
#: So the regime constant alone removes 35% of the mean error, and the power
#: weighting more than halves the MEDIAN -- at the cost of a heavier tail,
#: because the leading-order diffusion prior is fit for RANK (within-cell rho
#: 0.75) and its amplitude is inflated (fitted-M^2 P spans 0.13..3.14 where a
#: real assembly spans ~0.4..1.4).  **Both estimators are therefore in the
#: inventory** -- ``nominal_burnup`` carries the flat regime one and
#: ``chain_bu_integral`` the power-weighted one -- and the network is left to
#: blend them rather than being forced onto either.  Read the channel as
#: "power-weighted residence integral on the regime scale", proportional to
#: burnup up to the prior's known amplitude inflation, NOT as calibrated GWd/tU.
#: (Damping the weight would fix the tail but would introduce a free parameter
#: with no pre-registered value, which a pre-registered arm may not carry.)
_V6B_SRC_CHAIN_EXTRA: tuple[str, ...] = (
    "src1_prior_power",     # (P(direct[s]) - 1) / 0.4        last cycle's position
    "src2_prior_power",     # (P(direct[direct[s]]) - 1) / 0.4   two cycles back
    "src2_radius",          # radius(direct[direct[s]]) / r_max
    "chain_bu_integral",    # B_regime * sum_k P(src_k) / _BURNUP_SCALE
    "src1_present",         # {0,1} the direct source resolved
    "src2_present",         # {0,1} the second-order source resolved
)

#: v6b = v6 + the source-chain block (52 -> 58), append-only.
CHANNELS_V6B: tuple[str, ...] = CHANNELS_V6 + _V6B_SRC_CHAIN_EXTRA

# --------------------------------------------------------------------------- #
# cond_v6c: the ASSEMBLY DISCONTINUITY FACTOR channels
# --------------------------------------------------------------------------- #
#: The second feature the STEP 0 MASTER null test named
#: (``kcurve_fusion_memo_20260809.md`` §7, `ab2_addendum_ADF_20260810.md`).
#: H2/H4 are the same assembly under (k-curve + enrichment): 5.5 w/o both, 24 Gd
#: pins both, k-inf matched to <=57 pcm at every stored burnup -- yet they build
#: cores differing by node_peak 0.0072 and F_r 0.0351, at 14x the negative
#: control.  ADF is what separates them, and R^2(adf_face_g2 | 10-column k-curve)
#: = 0.003: it is the ONE harvested descriptor that does not collapse onto the
#: k-curve (M^2 0.998, MTC 0.991, boron 0.991, Doppler 0.993 all do).
#:
#: **Scope, registered:** v6b ALREADY carries ``origin_adf_corner_g2`` (idx 38),
#: ``origin_cr1_worth`` (34) and ``origin_ff_pin_max`` (30) -- all three inherited
#: from v4.  So this block adds only the FACE ADFs and the group-1 corner ADF,
#: and the arm tests what they contribute ON TOP of that existing block.  The
#: control is therefore NOT blind to H2/H4; see the addendum §1 and §5.6.
_V6C_ADF_EXTRA: tuple[str, ...] = (
    "origin_adf_face_g1",       # (x - 1.001) / 0.014
    "origin_adf_face_g2",       # (x - 1.068) / 0.061   <- the H2/H4 discriminator
    "origin_adf_corner_g1",     # (x - 0.968) / 0.029
    "origin_adf_present",       # {0,1} presence gate for the whole ADF block
)

#: ``(channel, FuelVec attr, ref, scale)``.  ``ref = median`` / ``scale ~=
#: half-range`` over the 117 harvested types, the same rule :data:`_V4_SCALES`
#: and :data:`_V5_SHAPE_SCALED` follow, so each channel is O(1) over the ACTUAL
#: population (face_g1 [0.9926, 1.02025], face_g2 [1.02338, 1.14588],
#: corner_g1 [0.95145, 1.00960]).
_V6C_ADF_SCALED: tuple[tuple[str, str, float, float], ...] = (
    ("origin_adf_face_g1", "adf_face_g1", 1.001, 0.014),
    ("origin_adf_face_g2", "adf_face_g2", 1.068, 0.061),
    ("origin_adf_corner_g1", "adf_corner_g1", 0.968, 0.029),
)

#: v6c = v6b + the ADF block (58 -> 62), append-only.
CHANNELS_V6C: tuple[str, ...] = CHANNELS_V6B + _V6C_ADF_EXTRA

CHANNELS_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    "v2": CHANNELS,
    "v3": CHANNELS,
    "v4": CHANNELS_V4,
    "v5": CHANNELS_V5,
    "v5_noshape": CHANNELS_V5_NOSHAPE,
    "v6": CHANNELS_V6,
    "v6_contrast": CHANNELS_V6_CONTRAST,
    "v6_prior": CHANNELS_V6_PRIOR,
    "v6b": CHANNELS_V6B,
    "v6c": CHANNELS_V6C,
    # v7 / v8 grow the GLOBAL vector only — their cell inventory IS v6c's.
    "v7": CHANNELS_V6C,
    "v8": CHANNELS_V6C,
}

#: Largest fresh-type alphabet the composition-moment globals describe (v8).
#: Mirrors :data:`lpopt.search.genome.MAX_FRESH_TYPES`; the per-type fraction
#: block is padded to this width so the global vector has a fixed length whether
#: a case feeds 2, 3, 4 or 5 types (a 2-type record simply lands hard 0.0 in
#: slots 3..5).
MAX_FRESH_TYPES: int = 5

#: Composition-block width per schema marker.  A schema PINS its width forever:
#: the width sets both the number of ``g_type_frac_*`` slots and the divisor of
#: ``g_n_fresh_types``, so changing an existing entry would silently move a
#: trained checkpoint's inputs.  v7 stays at 3 (its retrain is in flight); v8 is
#: the 5-wide block.
_COMPOSITION_WIDTH_BY_FLAG: dict[str, int] = {
    "v7_composition": 3,
    "v8_composition": 5,
}

_CH_INDEX = {name: i for i, name in enumerate(CHANNELS)}
C = len(CHANNELS)

#: Base global-vector names (order-stable).  ``g_dataset_flag`` is dropped when
#: ``include_dataset_flag=False`` (plan: config-gated).
_GLOBALS_FULL: tuple[str, ...] = (
    "g_feed_241",           # feed / 241
    "g_feed_centered",      # (feed - feed_center) / feed_center_scale  (schema)
    "g_e_core",             # (e_core - enr_ref) / enr_scale  (schema)
    "g_e_split",            # |e_A - e_B| / 0.5
    "g_split_frac",         # feed fraction on the lexicographically-first pair member
    "g_depth2_frac",        # (60 - 2N) / depth2_scale  (schema)
    "g_fresh_mean_n_gd",    # feed-weighted fresh mean n_gd / 24
    "g_fresh_mean_gd_wt",   # feed-weighted fresh mean gd_wt / 10
    "g_dataset_flag",       # A=0, B=1  (config-gated)
    "g_sym_class",          # 1.0 for rot61
)

#: cond_v4 global-vector extras, appended AFTER the base globals for v4 only.
#: Multiplicity-weighted fresh-slot means of the (already-normalized) per-slot
#: result channels, mirroring ``g_fresh_mean_n_gd`` — a whole-core summary of the
#: fresh-feed reactivity signature.
_V4_GLOBALS_EXTRA: tuple[str, ...] = (
    "g_fresh_mean_kinf0",       # feed-weighted fresh mean of origin_kinf0 (norm)
    "g_fresh_mean_bu_k1",       # feed-weighted fresh mean of origin_bu_k1 (norm)
    "g_fresh_mean_boron_worth", # feed-weighted fresh mean of origin_boron_worth
)

#: cond_v5 global-vector extras — the poison-agnostic REPLACEMENTS for the two
#: dropped Gd means, built by the identical multiplicity-weighted fresh-slot
#: recipe over the (already-normalized) shape channels.
_V5_GLOBALS_EXTRA: tuple[str, ...] = (
    "g_fresh_mean_reactivity_swing",   # whole-core fresh-feed absorber release
    "g_fresh_mean_depletion_slope",    # whole-core fresh-feed burnout decay
)

#: cond_v7 global extras — the composition MOMENTS of the fresh-type alphabet,
#: appended after every earlier global (index-stable: a v6c checkpoint keeps
#: serving on its own 13).  Together with the globals that already exist they are
#: the full moment set of the fresh feed:
#:
#:   * **mean**      -> ``g_e_core``   (already present; U-mass-weighted)
#:   * **spread**    -> ``g_e_split``  (already present; max - min over the FED
#:                      types, which is exactly ``|e_A - e_B|`` for a pair)
#:   * **fractions** -> ``g_type_frac_1..3`` (NEW; lexicographic member order,
#:                      zero-padded to the schema's composition width)
#:   * **2nd moment**-> ``g_e_type_std`` (NEW; feed-fraction-weighted standard
#:                      deviation of the fed type enrichments)
#:   * **cardinality**-> ``g_n_fresh_types`` (NEW; distinct FED types / 3)
#:
#: ``g_type_frac_1`` reproduces ``g_split_frac`` exactly for a 2-type record, and
#: the std is the channel that distinguishes a graded 3-type feed from the 2-type
#: feed with the same mean and the same max-min.
_V7_GLOBALS_EXTRA: tuple[str, ...] = (
    "g_type_frac_1",
    "g_type_frac_2",
    "g_type_frac_3",
    "g_e_type_std",
    "g_n_fresh_types",
)

#: cond_v8 global extras — v7's composition block widened from 3 to 5 fraction
#: slots (operator directive "3~5종 그물망").  Same moments, same recipe, same
#: lexicographic member order; only the padding width and the ``g_n_fresh_types``
#: divisor change (3 -> 5).  A 2-type record lands 0.0 in slots 3..5 and a 3-type
#: record 0.0 in slots 4..5, so **the whole 2-type and 3-type corpus featurizes
#: under v8 carrying exactly the information it carried under v7** — v8 is a
#: widening, not a re-encoding, which is what lets the ~39k pair records plus any
#: triple records seed a v8 retrain.
_V8_GLOBALS_EXTRA: tuple[str, ...] = (
    "g_type_frac_1",
    "g_type_frac_2",
    "g_type_frac_3",
    "g_type_frac_4",
    "g_type_frac_5",
    "g_e_type_std",
    "g_n_fresh_types",
)
#: ``(global name, source channel)`` for the fresh-slot means, so the global is
#: always the exact weighted mean of a channel that is actually in the inventory.
_FRESH_MEAN_SOURCES: dict[str, str] = {
    "g_fresh_mean_n_gd": "origin_n_gd",
    "g_fresh_mean_gd_wt": "origin_gd_wt",
    "g_fresh_mean_kinf0": "origin_kinf0",
    "g_fresh_mean_bu_k1": "origin_bu_k1",
    "g_fresh_mean_boron_worth": "origin_boron_worth",
    "g_fresh_mean_reactivity_swing": "origin_reactivity_swing",
    "g_fresh_mean_depletion_slope": "origin_depletion_slope",
}


def _norm_opt(value: Any, ref: float, scale: float) -> float:
    """``(value - ref) / scale`` with presence semantics: None/NaN -> 0.0.

    The 0.0 sentinel means "value absent" — paired with a presence gate channel
    (``origin_kinf_present`` / ``origin_lattice_present``) so the network can tell
    a genuine centered-zero from an unharvested one.
    """
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    return (v - ref) / scale


# --------------------------------------------------------------------------- #
# leakage-safe input surface
# --------------------------------------------------------------------------- #
#: Columns the encoder is *allowed* to read.  Anything else (targets, metrics,
#: maps, provenance) is invisible to featurization.
SAFE_INPUT_FIELDS: tuple[str, ...] = (
    "pattern",
    "feed",
    "e_core",
    "e_split",
    "case_pair",
    "library_id",
    "sym_class",
    "dataset",
)

#: The ga80 letter library is Dataset B (free-69 orbit representation); every
#: other library is a Dataset A APR1400 core (rot-61).  A store row carries these
#: two flags explicitly, but a :class:`~..vendor.masterrl.domain.CaseKey` served
#: at inference carries no provenance — so the serving path derives them from the
#: campaign's ``library_id`` here.  (Mirrors ``extract_b.GA80_LIBRARY`` /
#: ``GA_SYM_CLASS`` and ``extract_a`` / ``schema.SYM_CLASS``.)
_GA80_LIBRARY = "ga80"


def library_provenance(library_id: str) -> tuple[str, str]:
    """``(dataset, sym_class)`` implied by a library id for inference featurization.

    ga80 -> ``("B", "free69")``; any Dataset A library -> ``("A", "rot61")``.
    """
    if library_id == _GA80_LIBRARY:
        return "B", "free69"
    return "A", "rot61"


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


@dataclass(frozen=True)
class RecordInputs:
    """The complete leakage-safe input a record contributes to featurization.

    Only :data:`SAFE_INPUT_FIELDS` are ever pulled from a record row via
    :meth:`coerce`; the frozen dataclass rejects any extra keyword (so a target
    column can never sneak in through the constructor).
    """

    pattern: str
    feed: int
    case_pair: str
    library_id: str
    e_core: float | None = None
    e_split: float | None = None
    sym_class: str = "rot61"
    dataset: str = "A"

    @classmethod
    def coerce(cls, row: Any) -> "RecordInputs":
        if isinstance(row, RecordInputs):
            return row

        def _f(key: str) -> float | None:
            v = _row_get(row, key)
            if v is None:
                return None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return None if math.isnan(fv) else fv

        return cls(
            pattern=str(_row_get(row, "pattern")),
            feed=int(_row_get(row, "feed")),
            case_pair=str(_row_get(row, "case_pair")),
            library_id=str(_row_get(row, "library_id")),
            e_core=_f("e_core"),
            e_split=_f("e_split"),
            sym_class=str(_row_get(row, "sym_class", "rot61")),
            dataset=str(_row_get(row, "dataset", "A")),
        )


# --------------------------------------------------------------------------- #
# static geometry (computed once at import)
# --------------------------------------------------------------------------- #
def _mirror_positions(row: int, col: int) -> list[tuple[int, int]]:
    """The (r, c) 19x19 positions a quarter slot mirrors onto (SE/SW/NE/NW)."""
    out = set()
    for dr in (row, -row):
        for dc in (col, -col):
            out.add((_GRID_CENTER + dr, _GRID_CENTER + dc))
    return sorted(out)


def _build_scatter() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    slot_ids: list[int] = []
    fuel = np.zeros((_GRID, _GRID), dtype=np.float32)
    for slot in SLOTS:
        for (r, c) in _mirror_positions(slot.row, slot.col):
            rows.append(r)
            cols.append(c)
            slot_ids.append(slot.index)
            fuel[r, c] = 1.0
    # reflector ring: non-fuel cells with a fuel neighbour (8-connectivity).
    reflector = np.zeros((_GRID, _GRID), dtype=np.float32)
    for r in range(_GRID):
        for c in range(_GRID):
            if fuel[r, c]:
                continue
            hit = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < _GRID and 0 <= cc < _GRID and fuel[rr, cc]:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                reflector[r, c] = 1.0
    return (
        np.asarray(rows, dtype=np.intp),
        np.asarray(cols, dtype=np.intp),
        np.asarray(slot_ids, dtype=np.intp),
        fuel,
        reflector,
    )


_POS_ROWS, _POS_COLS, _POS_SLOT, _FUEL_MASK, _REFLECTOR_MASK = _build_scatter()


def _geometry_slot_values(channels: tuple[str, ...]) -> np.ndarray:
    """Static per-slot channel values (masks + position block), shape (len,69).

    Only the schema-invariant geometry channels (``fuel_mask`` + the position
    block) are non-zero; every appended v4 channel is left zero here (it is a
    physics quantity filled per record).  Because the channel list is append-only
    and index-stable, the first 26 rows are byte-identical across schemas.
    """
    ch_index = {name: i for i, name in enumerate(channels)}
    vals = np.zeros((len(channels), len(SLOTS)), dtype=np.float32)
    vals[ch_index["fuel_mask"], :] = 1.0
    for slot in SLOTS:
        j = slot.index
        vals[ch_index["pos_radius"], j] = slot.radius / _R_MAX
        vals[ch_index["pos_multiplicity"], j] = slot.multiplicity / 4.0
        vals[ch_index["pos_on_axis"], j] = (
            1.0 if slot.orbit_class in ("horizontal_axis", "vertical_axis") else 0.0
        )
        vals[ch_index["pos_center"], j] = 1.0 if slot.orbit_class == "center" else 0.0
    return vals


#: Per-schema geometry-slot matrix (built once at import; the encoder picks its
#: own by ``cond_schema``).  Keyed by channel tuple so v2/v3 share one build.
_GEOM_SLOT_VALS_BY_CHANNELS: dict[tuple[str, ...], np.ndarray] = {
    ch: _geometry_slot_values(ch) for ch in set(CHANNELS_BY_SCHEMA.values())
}
_GEOM_SLOT_VALS = _GEOM_SLOT_VALS_BY_CHANNELS[CHANNELS]        # base (v2/v3)


# --------------------------------------------------------------------------- #
# type-id resolution (tolerant of the B01<->B1 zero-padding quirk)
# --------------------------------------------------------------------------- #
def _depad(type_id: str) -> str:
    """Strip a zero-padded numeric suffix: ``B01 -> B1``, ``X10 -> X10``."""
    i = len(type_id)
    while i > 0 and type_id[i - 1].isdigit():
        i -= 1
    head, tail = type_id[:i], type_id[i:]
    if not tail:
        return type_id
    return f"{head}{int(tail)}"


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class FeatureEncoder:
    """Stateless-per-call physics encoder with a per-(library,type) vec cache."""

    def __init__(self, include_dataset_flag: bool = True,
                 cond_schema: str = DEFAULT_COND_SCHEMA,
                 power_prior: Any | None = None):
        self.include_dataset_flag = bool(include_dataset_flag)
        if cond_schema not in _COND_NORM:
            raise ValueError(
                f"unknown cond_schema {cond_schema!r}; have {sorted(_COND_NORM)}"
            )
        self.cond_schema = str(cond_schema)
        norm = _COND_NORM[self.cond_schema]
        self._enr_ref = float(norm["enr_ref"])
        self._enr_scale = float(norm["enr_scale"])
        self._feed_center = float(norm["feed_center"])
        self._feed_center_scale = float(norm["feed_center_scale"])
        self._depth2_scale = float(norm["depth2_scale"])

        # -- schema-selected cell-channel inventory ----------------------------
        # v4 appends to v2/v3 (index-stable).  v5 / v5_noshape are NOT
        # append-only — they DROP the three Gd design channels — so every write
        # into the channel matrix is guarded by inventory membership rather than
        # by a bare schema flag.  ``_is_v4`` therefore means "the v4 result-based
        # block is active", which is true for v4, v5 and v5_noshape alike.
        # The v6 family is v5 + appended blocks, so it inherits BOTH v5 flags;
        # membership is still read off the marker dict rather than a name test so
        # a future schema cannot silently fall back to the v2/v3 behaviour.
        norm_flags = _COND_NORM[self.cond_schema]
        self._is_v5 = bool(norm_flags.get("v5_features"))
        self._is_v4 = bool(norm_flags.get("v4_features")) or self._is_v5
        #: True for the full v5 and every v6 (the v5 ablation arm drops the block).
        self._has_shape = self._is_v5 and not norm_flags.get("v5_no_shape")
        #: v6 appended families (design doc 20260725, arms A2 / A4).
        self._has_contrast = bool(norm_flags.get("v6_contrast"))
        self._has_power_prior = bool(norm_flags.get("v6_prior"))
        #: v6b halves (ab2_addendum_BU_20260810.md).  ``_has_regime_burnup``
        #: switches the a-priori burn state from the flat 22.0 to the measured
        #: ``(library_id, feed)`` table -- everywhere it is consumed, i.e. the
        #: ``nominal_burnup`` channel AND the burnup the power prior evaluates
        #: each slot's reference k-inf curve at (which builds channels 48/50/51).
        self._has_regime_burnup = bool(norm_flags.get("v6b_burnup"))
        self._has_src_chain = bool(norm_flags.get("v6b_srcchain"))
        #: v6c: the appended face / corner-g1 ADF block (the corner-g2, CR-worth
        #: and pin-form-factor channels are v4's and are present from v4 on).
        self._has_adf = bool(norm_flags.get("v6c_adf"))
        #: v7 / v8: the appended composition-moment globals (graded fresh-type
        #: arm).  A GLOBALS-only family — the cell inventory is v6c's verbatim.
        #: ``_composition_width`` is the per-type fraction block width the ACTIVE
        #: schema pins (v7 -> 3, v8 -> 5); 0 means the family is off.
        self._composition_width: int = next(
            (w for flag, w in _COMPOSITION_WIDTH_BY_FLAG.items()
             if norm_flags.get(flag)), 0)
        self._has_composition = self._composition_width > 0
        self.channels: tuple[str, ...] = CHANNELS_BY_SCHEMA[self.cond_schema]
        self._ch_index: dict[str, int] = {n: i for i, n in enumerate(self.channels)}
        self.n_channels: int = len(self.channels)
        self._geom_slot_vals = _GEOM_SLOT_VALS_BY_CHANNELS[self.channels]
        # The k-inf curve is a cond_v4 feature (module docstring): only a v4 encoder
        # activates + O(1)-normalizes the origin_kinf* channels; v2/v3 keep them
        # dormant.  These transforms are therefore used under v4 only.
        self._kinf_tx = lambda x: (float(x) - _KINF_REF) / _KINF_SCALE
        self._bu_k1_tx = lambda x: float(x) / _BU_K1_SCALE

        base_globals = tuple(
            n for n in _GLOBALS_FULL
            if self.include_dataset_flag or n != "g_dataset_flag"
        )
        if self._is_v5:
            # v5 drops the two poison-specific Gd means; the FULL v5 replaces
            # them with the two curve-shape means (the ablation arm does not).
            base_globals = _drop(base_globals, _V5_DROPPED_GLOBALS)
            extra = _V4_GLOBALS_EXTRA + (_V5_GLOBALS_EXTRA if self._has_shape else ())
            self.globals_names: tuple[str, ...] = base_globals + extra
        else:
            self.globals_names = (
                base_globals + _V4_GLOBALS_EXTRA if self._is_v4 else base_globals
            )
        # v7 / v8 append the composition moments AFTER every earlier global, so
        # the v6c prefix keeps its indices (append-only, like every channel
        # family).  v8 is v7's block widened 3 -> 5 fraction slots.
        if self._has_composition:
            self.globals_names = self.globals_names + (
                _V8_GLOBALS_EXTRA if self._composition_width == 5
                else _V7_GLOBALS_EXTRA)
        self._vec_cache: dict[tuple[str, str], FuelVec | None] = {}
        # ``power_prior`` supplies the fitted (M^2, extrap); ``None`` uses the
        # module defaults.  Only consulted when the active schema turns the
        # ``prior_power*`` family on, so no v2..v5 encoder ever touches it.
        self._power_prior = power_prior

    @property
    def power_prior(self) -> Any | None:
        """The fitted :class:`~.power_prior.PowerPrior`, or ``None`` for defaults."""
        return self._power_prior

    def _fill_v6(self, vals: np.ndarray, ix: dict[str, int],
                 inp: RecordInputs, fuel_library: FuelLibrary,
                 direct: Sequence[int | None] | None = None) -> None:
        """Write the cond_v6 local-contrast / power-prior channels (arms A2, A4).

        Under cond_v6b this also (a) evaluates the k-inf curve at the REGIME
        per-cycle burnup rather than the flat 22.0 -- so channels 48/50/51 carry
        the corrected burn state -- and (b) writes the appended source-chain
        block from the same power map, which needs ``direct`` from the caller's
        chain trace.

        Imported lazily: :mod:`.power_prior` imports this module, so a top-level
        import would be circular (the same pattern :mod:`.physics_prior` uses).
        """
        from . import power_prior as _pp

        # ``None`` keeps the module constant, so every v2..v6 encoder is
        # byte-identical; only v6b passes a regime value.
        bu_per_cycle = (regime_cycle_burnup(inp.library_id, inp.feed)
                        if self._has_regime_burnup else None)
        kinf = _pp.kinf_quarter(inp, fuel_library, encoder=self,
                                vec_cache=self._vec_cache,
                                bu_per_cycle=bu_per_cycle)
        if self._has_contrast:
            vals[ix["origin_kinf_contrast"]] = (
                (kinf - _pp.neighbour_mean(kinf)) / _V6_KINF_CONTRAST_SCALE)
            age_row = vals[ix["residence_age"]] * _AGE_SCALE
            vals[ix["origin_age_contrast"]] = (
                (age_row - _pp.neighbour_mean(age_row)) / _V6_AGE_CONTRAST_SCALE)
        if self._has_power_prior:
            prior = self._power_prior
            kw = ({} if prior is None
                  else {"m2_cm2": prior.m2_cm2, "extrap": prior.extrap})
            p = _pp.power_maps_from_kinf(kinf[None, :], **kw)[0]
            vals[ix["prior_power"]] = (p - 1.0) / _V6_POWER_SCALE
            vals[ix["prior_power_contrast"]] = (
                (p - _pp.neighbour_mean(p)) / _V6_POWER_SCALE)
            if self._has_src_chain and direct is not None:
                self._fill_src_chain(
                    vals, ix, p, direct,
                    NOMINAL_CYCLE_BURNUP_MWD_KG if bu_per_cycle is None
                    else bu_per_cycle)

    def _fill_src_chain(self, vals: np.ndarray, ix: dict[str, int],
                        p: np.ndarray, direct: Sequence[int | None],
                        bu_per_cycle: float) -> None:
        """Write the :data:`_V6B_SRC_CHAIN_EXTRA` block from the prior power map.

        ``p`` is the 69-slot a-priori relative assembly power (core mean 1) that
        already fills channel 50, so the source quantities are a pure gather --
        no second physics solve.

        The guard is read off ``shuffle_source_present`` rather than recomputed,
        so this block fires on EXACTLY the slots ``chain_source_radius`` /
        ``shuffle_src_x`` fire on.  That matters because
        :meth:`_trace_chain` has a defensive branch (a cyclic or dangling shuffle
        source is treated as a fresh root while ``direct`` stays populated); reusing
        the already-written gate keeps the whole chain family consistent on those
        slots instead of inventing a residence history for them.
        """
        present = vals[ix["shuffle_source_present"]]
        for slot in SLOTS:
            j = slot.index
            if present[j] <= 0.0:
                continue                      # fresh / unresolved: leave all zero
            s1 = direct[j]
            if s1 is None:
                continue
            vals[ix["src1_present"], j] = 1.0
            vals[ix["src1_prior_power"], j] = (p[s1] - 1.0) / _V6_POWER_SCALE
            chain_power = float(p[s1])
            s2 = direct[s1] if present[s1] > 0.0 else None
            if s2 is not None:
                vals[ix["src2_present"], j] = 1.0
                vals[ix["src2_prior_power"], j] = (p[s2] - 1.0) / _V6_POWER_SCALE
                vals[ix["src2_radius"], j] = SLOTS[s2].radius / _R_MAX
                chain_power += float(p[s2])
            # Burnup accumulated over residence == power integrated over time.
            # The regime constant is the per-cycle scale; the power weights are
            # what the flat ``(age-1)*B`` encoding throws away.
            vals[ix["chain_bu_integral"], j] = (
                bu_per_cycle * chain_power / _BURNUP_SCALE)

    @property
    def cond_norm(self) -> dict[str, float]:
        """The active schema's (enrichment, feed, depth-2) normalization constants.

        Stamped into the checkpoint meta at train time so the serving path can
        rebuild an identically-normalizing encoder (and reject a v2/v3 mismatch).
        """
        return {
            "cond_schema": self.cond_schema,
            "enr_ref": self._enr_ref,
            "enr_scale": self._enr_scale,
            "feed_center": self._feed_center,
            "feed_center_scale": self._feed_center_scale,
            "depth2_scale": self._depth2_scale,
        }

    # -- fuel-vec resolution ------------------------------------------------ #
    def _resolve_vec(self, fuel_library: FuelLibrary, type_id: str,
                     library_id: str) -> FuelVec | None:
        key = (library_id, type_id)
        if key in self._vec_cache:
            return self._vec_cache[key]
        vec: FuelVec | None
        try:
            vec = fuel_library.get(type_id, library_id)
        except KeyError:
            alt = _depad(type_id)
            vec = None
            if alt != type_id:
                try:
                    vec = fuel_library.get(alt, library_id)
                except KeyError:
                    vec = None
        self._vec_cache[key] = vec
        return vec

    # -- chain tracing ------------------------------------------------------ #
    @staticmethod
    def _trace_chain(items: Sequence) -> tuple[list[int], list[int], list[int | None]]:
        """Per-slot ``(age, origin_slot, direct_source_slot)`` (a-priori only).

        ``age`` is 1 for fresh, 2 for once-burned (source is fresh), 3 for
        twice-burned, resolved by following each shuffle card's source
        coordinate up the chain to its fresh root.  Cycles / dangling sources
        fall back to treating the slot as its own root (defensive).
        """
        n = len(items)
        age = [0] * n
        origin = [0] * n
        direct: list[int | None] = [None] * n
        resolving = [False] * n

        def resolve(s: int) -> None:
            if age[s]:
                return
            item = items[s]
            if item.is_fresh:
                age[s], origin[s], direct[s] = 1, s, None
                return
            src = _coord_slot(item.x, int(item.y))
            direct[s] = src
            if src is None or src == s or resolving[src]:
                age[s], origin[s] = 1, s          # defensive: treat as root
                return
            resolving[s] = True
            resolve(src)
            resolving[s] = False
            age[s] = age[src] + 1
            origin[s] = origin[src]

        for s in range(n):
            resolve(s)
        return age, origin, direct

    # -- v4 result-based origin channels ------------------------------------ #
    def _fill_v4_origin(self, vals: np.ndarray, ix: dict[str, int], j: int,
                        vec: FuelVec | None) -> None:
        """Fill the ``_V4_EXTRA`` origin channels for slot ``j`` from origin ``vec``.

        Every underlying ``fuel_types`` column is read defensively via
        ``getattr(vec, name, None)`` so this imports/runs even before the harvest
        agent's columns land (they are simply absent -> 0.0 with presence 0).
        """
        if vec is None:
            return
        er, es = self._enr_ref, self._enr_scale
        vals[ix["origin_enr_main"], j] = _norm_opt(getattr(vec, "enr_main", None), er, es)
        vals[ix["origin_enr_zone"], j] = _norm_opt(getattr(vec, "enr_zone", None), er, es)
        # POISON-SPECIFIC: present for v4, dropped by v5 (see _V5_DROPPED_CHANNELS).
        if "origin_gd_u_enr" in ix:
            vals[ix["origin_gd_u_enr"], j] = _norm_opt(
                getattr(vec, "gd_u_enr", None), er, es)
        vals[ix["origin_u_mass"], j] = _norm_opt(
            getattr(vec, "u_mass_g", None), _U_MASS_REF, _U_MASS_SCALE)
        vals[ix["origin_kinf10"], j] = _norm_opt(
            getattr(vec, "kinf10", None), _KINF_REF, _KINF_SCALE)
        vals[ix["origin_kinf30"], j] = _norm_opt(
            getattr(vec, "kinf30", None), _KINF_REF, _KINF_SCALE)
        vals[ix["origin_ff_pin_max"], j] = _norm_opt(
            getattr(vec, "ff_pin_max", None), _FF_REF, _FF_SCALE)
        for ch, attr, key in _V4_ORIGIN_SCALED:
            ref, scale = _V4_SCALES[key]
            vals[ix[ch], j] = _norm_opt(getattr(vec, attr, None), ref, scale)
        k0 = getattr(vec, "kinf0", None)
        present = k0 is not None and not (isinstance(k0, float) and math.isnan(k0))
        vals[ix["origin_lattice_present"], j] = 1.0 if present else 0.0

    # -- v5 poison-agnostic curve-shape channels ---------------------------- #
    def _fill_v5_shape(self, vals: np.ndarray, ix: dict[str, int], j: int,
                       vec: FuelVec | None) -> None:
        """Fill the :data:`_V5_SHAPE_EXTRA` channels for slot ``j`` from ``vec``.

        The k-conv shape block is the poison-AGNOSTIC replacement for the dropped
        Gd design channels: it describes the absorber's holddown -> release
        BEHAVIOUR (dip/peak timing, reactivity swing, burnout slope, discharge
        k-inf), not the chemistry that produced it, so it transfers unchanged to
        an IFBA / Er / Dy lattice.  Every column is read defensively; an absent
        or NaN value normalizes to 0.0 and ``origin_kconv_present`` (gated on a
        finite ``kinf_peak``, the column that fills for EVERY harvested curve)
        tells the network a genuine centered-zero from an unharvested one.
        A monotone curve legitimately leaves dip/swing NaN -> 0, which is exactly
        the "no prominent hump" state ``origin_kconv_monotone`` also marks.
        """
        if vec is None:
            return
        for ch, attr, ref, scale in _V5_SHAPE_SCALED:
            vals[ix[ch], j] = _norm_opt(getattr(vec, attr, None), ref, scale)
        peak = getattr(vec, "kinf_peak", None)
        has_curve = peak is not None and not (
            isinstance(peak, float) and math.isnan(peak))
        vals[ix["origin_kconv_present"], j] = 1.0 if has_curve else 0.0

    # -- v6c assembly-discontinuity-factor channels ------------------------- #
    def _fill_v6c_adf(self, vals: np.ndarray, ix: dict[str, int], j: int,
                      vec: FuelVec | None) -> None:
        """Fill :data:`_V6C_ADF_EXTRA` for slot ``j`` from its fresh-origin ``vec``.

        The ADF is the ONE harvested descriptor that does not collapse onto the
        k-inf curve (R^2(adf_face_g2 | 10-column k-curve) = 0.003 on ga80, vs
        0.99+ for M^2 / MTC / boron worth / Doppler), which is why the STEP 0
        MASTER null test could separate two assemblies the k-curve says are
        identical.  Read defensively like every other origin block; an absent or
        NaN column normalizes to 0.0 and ``origin_adf_present`` tells the network
        a genuine centered zero from an unharvested one.

        All six ADF/CR/FF columns share one coverage pattern (117 of 153 types,
        and 100% of the slots on the frozen decision surface), so a single
        presence gate is exact rather than approximate.
        """
        if vec is None:
            return
        for ch, attr, ref, scale in _V6C_ADF_SCALED:
            vals[ix[ch], j] = _norm_opt(getattr(vec, attr, None), ref, scale)
        face = getattr(vec, "adf_face_g2", None)
        present = face is not None and not (
            isinstance(face, float) and math.isnan(face))
        vals[ix["origin_adf_present"], j] = 1.0 if present else 0.0

    # -- slot channel matrix ------------------------------------------------ #
    def encode_slot_matrix(self, record_row: Any,
                           fuel_library: FuelLibrary) -> np.ndarray:
        """Per-slot channel values, shape ``(C, 69)`` (pre-grid-expansion).

        Exposed for spot-check tests (center/origin tracing) without grid
        position arithmetic; :meth:`encode` scatters this onto the 19x19 grid.
        """
        inp = RecordInputs.coerce(record_row)
        pattern = unpack_pattern(inp.pattern)
        items = pattern.items
        age, origin, direct = self._trace_chain(items)

        vals = self._geom_slot_vals.copy()
        ix = self._ch_index
        # A-priori per-cycle burnup.  v2..v6 keep the flat module constant (so
        # they stay byte-identical); v6b reads the measured (library, feed)
        # regime.  ``_BURNUP_SCALE`` is pinned to 66.0 either way, so this is the
        # one place the table can actually move the channel.
        bu_cycle = (regime_cycle_burnup(inp.library_id, inp.feed)
                    if self._has_regime_burnup else NOMINAL_CYCLE_BURNUP_MWD_KG)
        for slot in SLOTS:
            j = slot.index
            item = items[j]
            is_fresh = item.is_fresh
            vals[ix["occ_fresh"], j] = 1.0 if is_fresh else 0.0
            vals[ix["occ_burned"], j] = 0.0 if is_fresh else 1.0

            origin_item = items[origin[j]]
            vec = self._resolve_vec(fuel_library, origin_item.batch, inp.library_id)
            e = None if vec is None else vec.u_avg_enrichment
            if e is not None and not math.isnan(e):
                vals[ix["origin_enrichment"], j] = (e - self._enr_ref) / self._enr_scale
            # POISON-SPECIFIC design axes: in the v2/v3/v4 inventory, DROPPED by
            # v5 (the k-conv shape block below carries the absorber signal in a
            # chemistry-independent form).  Guarded by inventory membership, so
            # the v2/v3/v4 path is unchanged (the names are always present there).
            if "origin_n_gd" in ix and vec is not None and vec.n_gd is not None:
                vals[ix["origin_n_gd"], j] = vec.n_gd / _N_GD_SCALE
            if ("origin_gd_wt" in ix and vec is not None
                    and vec.gd_wt is not None and not math.isnan(vec.gd_wt)):
                vals[ix["origin_gd_wt"], j] = vec.gd_wt / _GD_WT_SCALE
            if vec is not None and vec.axial_zone == "z2":
                vals[ix["origin_axial_z2"], j] = 1.0
            if vec is None or vec.feature_poor:
                vals[ix["origin_feature_poor"], j] = 1.0
            # The k-inf curve is a cond_v4 feature (module docstring): v4 activates
            # + O(1)-normalizes origin_kinf* ((k-1)/0.25, bu/30); v2/v3 keep them
            # DORMANT (0).  Dormant is byte-identical to the pre-harvest training a
            # v2/v3 champion saw (kinf was NaN -> 0), so filling the fuel table's
            # k-inf columns never feeds an existing v2/v3 champion OOD raw k-inf —
            # only a freshly-trained v4 ensemble consumes the harvested curve.
            if (self._is_v4 and vec is not None and vec.kinf0 is not None
                    and not math.isnan(vec.kinf0)):
                vals[ix["origin_kinf_present"], j] = 1.0
                vals[ix["origin_kinf0"], j] = self._kinf_tx(vec.kinf0)
                if vec.kinf20 is not None and not math.isnan(vec.kinf20):
                    vals[ix["origin_kinf20"], j] = self._kinf_tx(vec.kinf20)
                if vec.bu_k1 is not None and not math.isnan(vec.bu_k1):
                    vals[ix["origin_bu_k1"], j] = self._bu_k1_tx(vec.bu_k1)

            # v4-only: the appended result-based origin channels (traced to the
            # same fresh origin ``vec``; None/NaN -> 0 with presence semantics).
            if self._is_v4:
                self._fill_v4_origin(vals, ix, j, vec)
            # v5-only: the poison-agnostic k-conv curve-shape block.
            if self._has_shape:
                self._fill_v5_shape(vals, ix, j, vec)
            # v6c-only: the appended face / corner-g1 ADF block.
            if self._has_adf:
                self._fill_v6c_adf(vals, ix, j, vec)

            a = age[j]
            vals[ix["residence_age"], j] = a / _AGE_SCALE
            vals[ix["nominal_burnup"], j] = (a - 1) * bu_cycle / _BURNUP_SCALE

            if not is_fresh and direct[j] is not None:
                src = direct[j]
                vals[ix["shuffle_source_present"], j] = 1.0
                r_src = SLOTS[src].radius
                vals[ix["chain_source_radius"], j] = r_src / _R_MAX
                vals[ix["chain_displacement"], j] = abs(r_src - slot.radius) / _R_MAX
                src_cell = cell_of_slot(src)
                vals[ix["shuffle_src_x"], j] = src_cell.qi / 9.0
                vals[ix["shuffle_src_y"], j] = src_cell.qj / 9.0
                if item.rotation == 1:
                    vals[ix["shuffle_rot1"], j] = 1.0
                elif item.rotation == 2:
                    vals[ix["shuffle_rot2"], j] = 1.0
        # v6-only: the appended neighbour-contrast / power-prior families.  These
        # are inherently NON-local (they read the whole quarter at once), so they
        # are filled after the per-slot loop rather than inside it.
        if self._has_contrast or self._has_power_prior:
            self._fill_v6(vals, ix, inp, fuel_library, direct)
        return vals

    # -- full encode -------------------------------------------------------- #
    def encode(self, record_row: Any,
               fuel_library: FuelLibrary) -> tuple[np.ndarray, np.ndarray]:
        """Featurize one record into ``(cells[C,19,19], globals[G])`` (float32)."""
        inp = RecordInputs.coerce(record_row)
        slot_vals = self.encode_slot_matrix(inp, fuel_library)

        grid = np.zeros((self.n_channels, _GRID, _GRID), dtype=np.float32)
        grid[:, _POS_ROWS, _POS_COLS] = slot_vals[:, _POS_SLOT]
        grid[self._ch_index["reflector_mask"]] = _REFLECTOR_MASK

        globals_ = self._encode_globals(inp, fuel_library, slot_vals)
        return grid, globals_

    def _encode_globals(self, inp: RecordInputs, fuel_library: FuelLibrary,
                        slot_vals: np.ndarray) -> np.ndarray:
        pattern = unpack_pattern(inp.pattern)
        feed = float(inp.feed)
        n_fresh = (int(inp.feed) - 1) / 4.0

        # Case-member enrichments via FuelLibrary (fall back to e_split column).
        # ``max - min`` over ALL members generalizes the pair's ``|e_A - e_B|``
        # to a 3-type graded case and is arithmetically identical for two.
        members = [m for m in inp.case_pair.split("_") if m]
        member_enr = self._member_enrichments(members, inp.library_id, fuel_library)
        e_split = inp.e_split
        if member_enr is not None and len(member_enr) >= 2:
            e_split = max(member_enr) - min(member_enr)
        if e_split is None or (isinstance(e_split, float) and math.isnan(e_split)):
            e_split = 0.0

        # e_core: prefer the stored input; fall back to the shared feed-average
        # recipe (identical to the value extraction wrote for a store row).
        e_core = inp.e_core
        if e_core is None or (isinstance(e_core, float) and math.isnan(e_core)):
            e_core = self._estimate_e_core(inp, fuel_library, pattern)

        # split fraction on the lexicographically-first pair member.
        batch_feed = pattern.batch_feed()
        total_feed = pattern.feed or 1
        batch_a = sorted(members)[0] if members else None
        split_frac = (batch_feed.get(batch_a, 0) / total_feed) if batch_a else 0.0

        depth2_frac = (60.0 - 2.0 * n_fresh) / self._depth2_scale

        # feed-weighted fresh means from the per-slot matrix.
        ix = self._ch_index
        occ = slot_vals[ix["occ_fresh"]]
        mult = np.asarray([s.multiplicity for s in SLOTS], dtype=np.float32)
        w = occ * mult
        wsum = float(w.sum()) or 1.0

        values = {
            "g_feed_241": feed / _FEED_SCALE,
            "g_feed_centered": (feed - self._feed_center) / self._feed_center_scale,
            "g_e_core": (float(e_core) - self._enr_ref) / self._enr_scale,
            "g_e_split": float(e_split) / _ESPLIT_SCALE,
            "g_split_frac": float(split_frac),
            "g_depth2_frac": depth2_frac,
            "g_dataset_flag": 0.0 if inp.dataset == "A" else 1.0,
            "g_sym_class": 1.0 if inp.sym_class == "rot61" else 0.0,
        }
        # Multiplicity-weighted fresh-slot means of the (already-normalized)
        # per-slot channels.  Driven off :data:`_FRESH_MEAN_SOURCES` so a global
        # is only ever computed from a channel the ACTIVE schema carries — that
        # is what lets v5 drop the two Gd means and add the two shape means
        # without a schema branch.  The arithmetic is identical to the previous
        # hand-written v2/v3/v4 lines, so those schemas are byte-identical.
        for name in self.globals_names:
            src = _FRESH_MEAN_SOURCES.get(name)
            if src is not None:
                values[name] = float((slot_vals[ix[src]] * w).sum()) / wsum
        if self._has_composition:
            values.update(self._composition_globals(
                members, member_enr, batch_feed, total_feed))
        return np.asarray(
            [values[n] for n in self.globals_names], dtype=np.float32
        )

    def _member_enrichments(self, members: Sequence[str], library_id: str,
                            fuel_library: FuelLibrary) -> list[float] | None:
        """Every case member's U-average enrichment, or ``None`` if any is unknown.

        All-or-nothing exactly like :func:`.fuel_types.core_enrichment_split`, so
        a partially-resolvable case falls back to the stored ``e_split`` column
        rather than to a silently truncated alphabet.
        """
        if len(members) < 2:
            return None
        out: list[float] = []
        for name in members:
            vec = self._resolve_vec(fuel_library, name, library_id)
            if vec is None:
                return None
            e = vec.u_avg_enrichment
            if e is None or math.isnan(e):
                return None
            out.append(float(e))
        return out

    def _composition_globals(self, members: Sequence[str],
                             member_enr: Sequence[float] | None,
                             batch_feed: dict[str, int], total_feed: int
                             ) -> dict[str, float]:
        """The cond_v7 / cond_v8 composition moments.

        Members are taken in LEXICOGRAPHIC order — the same order
        ``g_split_frac`` already uses for its "first pair member" — so
        ``g_type_frac_1`` reproduces ``g_split_frac`` exactly for a 2-type record
        and the block is invariant to how a case id happens to be written.
        Fractions are of the FULL-CORE feed (assemblies, multiplicity-weighted by
        ``Pattern.batch_feed``), padded with hard zeros to the active schema's
        composition width (v7 -> 3, v8 -> 5).  A member beyond the width is
        DROPPED rather than folded in, which cannot happen for a well-formed case
        (``case_batches`` caps the alphabet at the same number) but would
        otherwise make the fractions silently not sum to 1.
        """
        width = self._composition_width
        order = sorted(range(len(members)), key=lambda i: members[i])
        total = float(total_feed or 1)
        fracs = [batch_feed.get(members[i], 0) / total for i in order]
        enrs = ([member_enr[i] for i in order]
                if member_enr is not None and len(member_enr) == len(members)
                else None)
        if len(fracs) > width:
            fracs = fracs[:width]
            enrs = enrs[:width] if enrs is not None else None

        out: dict[str, float] = {}
        for k in range(width):
            out[f"g_type_frac_{k + 1}"] = float(fracs[k]) if k < len(fracs) else 0.0

        # Feed-fraction-weighted enrichment std (0 when the types are unknown or
        # only one type is actually fed).  Normalized on the pair-spread scale so
        # it is directly comparable to ``g_e_split``.
        std = 0.0
        if enrs is not None:
            wsum = sum(fracs)
            if wsum > 0.0:
                mean = sum(f * e for f, e in zip(fracs, enrs, strict=True)) / wsum
                var = sum(f * (e - mean) ** 2
                          for f, e in zip(fracs, enrs, strict=True)) / wsum
                std = math.sqrt(max(var, 0.0))
        out["g_e_type_std"] = float(std) / _ESPLIT_SCALE
        out["g_n_fresh_types"] = (
            float(sum(1 for f in fracs if f > 0.0)) / float(width))
        return out

    def _estimate_e_core(self, inp: RecordInputs, fuel_library: FuelLibrary,
                         pattern: Pattern) -> float:
        """Feed-average enrichment from the pattern's fresh batches.

        Delegates to :func:`core_enrichment_split` — the same recipe extraction
        used to fill the store ``e_core`` column — so a served pattern's e_core is
        byte-identical to the stored value for the same core.  Falls back to the
        normalization centre when the feed is unresolvable in ``library_id``.
        """
        e_core, _ = core_enrichment_split(
            fuel_library, inp.library_id, pattern.batch_feed()
        )
        return self._enr_ref if e_core is None else e_core

    # -- batch -------------------------------------------------------------- #
    def encode_batch(self, df, fuel_library: FuelLibrary
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Featurize every row of ``df`` into ``(cells[N,C,19,19], globals[N,G])``.

        Per-(library, type) fuel lookups are cached across the batch; per-pattern
        work runs in a Python loop (fast enough for the full ~39k store).
        """
        n = len(df)
        cells = np.zeros((n, self.n_channels, _GRID, _GRID), dtype=np.float32)
        gvec = np.zeros((n, len(self.globals_names)), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            c, g = self.encode(row, fuel_library)
            cells[i] = c
            gvec[i] = g
        return cells, gvec

    # -- transpose augmentation --------------------------------------------- #
    def augment_transpose(self, cells: np.ndarray, globals_: np.ndarray,
                          record_row: Any, fuel_library: FuelLibrary
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Diagonal-mirror augmentation via re-encoding the transposed pattern.

        Correctness over speed (plan sec. 4.4): the pattern is transposed with
        :func:`geometry.transpose` (which re-normalizes source coordinates and
        rotations) and re-encoded.  ``cells`` / ``globals_`` are accepted for API
        symmetry; the global vector is invariant under transpose.
        """
        inp = RecordInputs.coerce(record_row)
        tpat = transpose(unpack_pattern(inp.pattern))
        tinp = replace(inp, pattern=tpat.canonical())
        return self.encode(tinp, fuel_library)


__all__ = [
    "CHANNELS",
    "CHANNELS_V4",
    "CHANNELS_V5",
    "CHANNELS_V5_NOSHAPE",
    "CHANNELS_V6",
    "CHANNELS_V6B",
    "CHANNELS_V6C",
    "CHANNELS_BY_SCHEMA",
    "DEFAULT_COND_SCHEMA",
    "MAX_FRESH_TYPES",
    "FeatureEncoder",
    "NOMINAL_CYCLE_BURNUP_MWD_KG",
    "RecordInputs",
    "SAFE_INPUT_FIELDS",
    "regime_cycle_burnup",
    "schema_uses_regime_burnup",
]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    import time
    import pandas as pd

    df = pd.read_parquet("data/store/records.parquet")
    fl = FuelLibrary.from_parquet("data/store/fuel_types.parquet")
    enc = FeatureEncoder()
    cells, gvec = enc.encode(df.iloc[0], fl)
    print(f"channels C={C}  cells={cells.shape}  globals G={len(enc.globals_names)}")
    print("CHANNELS:", CHANNELS)
    print("GLOBALS :", enc.globals_names)
    n = min(2000, len(df))
    t0 = time.time()
    enc.encode_batch(df.iloc[:n], fl)
    print(f"encode_batch({n} rows) in {time.time() - t0:.1f}s")
