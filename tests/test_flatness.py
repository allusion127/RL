"""Canonical flatness scalars (flatness-first program 20260725 §1.1).

The load-bearing claims pinned here:

1. the multiplicity-weighted mean of a REAL harvested map is 1.0000, which is
   what makes ``node_peak == nanmax`` identical to F_xy;
2. the unweighted 69-slot mean is NOT 1.0 (so the weighting is load-bearing, not
   cosmetic);
3. NaN slots drop out of both moments instead of poisoning them;
4. every store map layout — ``(4,9,9)`` legacy and ``(n_steps,3,9,9)`` hires —
   resolves to the same BOC power plane;
5. the harvest-time entry point never raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpopt.data import flatness as F

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPS = REPO_ROOT / "data" / "store" / "maps.npz"


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_slot_weights_are_the_multiplicities() -> None:
    assert F.N_SLOTS == 69
    assert F.TOTAL_WEIGHT == 241.0
    counts = {1: 0, 2: 0, 4: 0}
    for w in F.SLOT_WEIGHTS:
        counts[int(w)] += 1
    assert counts == {1: 1, 2: 16, 4: 52}


def test_periphery_and_neighbours_are_consistent() -> None:
    # every slot has at least two in-core face neighbours, and the centre slot
    # (multiplicity 1) is interior
    assert F.NEIGH_VALID.sum(axis=1).min() >= 2
    assert not F.PERIPHERY_MASK[0]
    assert F.PERIPHERY_MASK.any()


# --------------------------------------------------------------------------- #
# the definition
# --------------------------------------------------------------------------- #
def test_flat_map_has_unit_mean_zero_cov_unit_peak() -> None:
    vals = np.ones((1, 69))
    assert F.weighted_mean(vals)[0] == pytest.approx(1.0)
    assert F.node_peak(vals)[0] == pytest.approx(1.0)
    assert F.map_cov(vals)[0] == pytest.approx(0.0)


def test_weighted_mean_differs_from_unweighted() -> None:
    """A field that is high only on the low-multiplicity axis slots."""
    vals = np.ones(69)
    vals[F.SLOT_WEIGHTS < 4] = 2.0
    weighted = F.weighted_mean(vals)[0]
    unweighted = float(np.mean(vals))
    assert weighted == pytest.approx(
        float((F.SLOT_WEIGHTS * vals).sum() / F.TOTAL_WEIGHT))
    assert abs(weighted - unweighted) > 1.0e-3


def test_map_cov_matches_the_written_formula() -> None:
    rng = np.random.default_rng(11)
    vals = 1.0 + 0.1 * rng.standard_normal((5, 69))
    w = F.SLOT_WEIGHTS
    for i in range(5):
        mean = float((w * vals[i]).sum() / w.sum())
        var = float((w * (vals[i] - mean) ** 2).sum() / w.sum())
        assert F.map_cov(vals)[i] == pytest.approx(np.sqrt(var) / mean)
        assert F.node_peak(vals)[i] == pytest.approx(vals[i].max())


def test_scaling_a_map_leaves_cov_invariant_and_scales_peak() -> None:
    rng = np.random.default_rng(3)
    vals = 1.0 + 0.05 * rng.standard_normal((1, 69))
    assert F.map_cov(2.0 * vals)[0] == pytest.approx(F.map_cov(vals)[0])
    assert F.node_peak(2.0 * vals)[0] == pytest.approx(2.0 * F.node_peak(vals)[0])


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #
def test_nan_slots_are_dropped_from_both_moments() -> None:
    rng = np.random.default_rng(7)
    vals = 1.0 + 0.05 * rng.standard_normal(69)
    holed = vals.copy()
    holed[[3, 40]] = np.nan

    keep = np.ones(69, dtype=bool)
    keep[[3, 40]] = False
    w = F.SLOT_WEIGHTS[keep]
    mean = float((w * vals[keep]).sum() / w.sum())
    var = float((w * (vals[keep] - mean) ** 2).sum() / w.sum())

    assert F.weighted_mean(holed)[0] == pytest.approx(mean)
    assert F.map_cov(holed)[0] == pytest.approx(np.sqrt(var) / mean)
    # the peak also ignores the holes (it would otherwise be NaN)
    assert F.node_peak(holed)[0] == pytest.approx(vals[keep].max())


def test_all_nan_map_yields_nan_not_an_exception() -> None:
    vals = np.full((1, 69), np.nan)
    assert np.isnan(F.weighted_mean(vals)[0])
    assert np.isnan(F.node_peak(vals)[0])
    assert np.isnan(F.map_cov(vals)[0])


def test_all_nan_map_reports_empty_scalars() -> None:
    s = F.flatness_scalars(np.full(69, np.nan))
    assert s.node_peak is None and s.map_cov is None and s.p_bar is None


# --------------------------------------------------------------------------- #
# layouts
# --------------------------------------------------------------------------- #
def _plane_from_slots(vals: np.ndarray) -> np.ndarray:
    plane = np.full((9, 9), np.nan)
    plane[F.SLOT_ROWS, F.SLOT_COLS] = vals
    return plane


def test_every_store_layout_resolves_to_the_same_boc_plane() -> None:
    rng = np.random.default_rng(5)
    vals = 1.0 + 0.05 * rng.standard_normal(69)
    plane = _plane_from_slots(vals)
    decoy = np.full((9, 9), 999.0)

    legacy = np.stack([plane, decoy, decoy, decoy], axis=0)          # (4,9,9)
    traj = np.stack(                                                 # (n,3,9,9)
        [np.stack([plane, decoy, decoy], axis=0)]
        + [np.stack([decoy, decoy, decoy], axis=0)] * 3, axis=0)

    for form in (vals, vals[None, :], plane, legacy, traj):
        got = F.slot_values(form)
        assert got.shape[-1] == 69
        np.testing.assert_allclose(got[0], vals)


def test_absent_map_is_none_and_bad_shape_raises() -> None:
    assert F.slot_values(None) is None
    assert F.slot_values(np.zeros((0, 69))) is None
    with pytest.raises(ValueError):
        F.slot_values(np.zeros((5, 5)))
    with pytest.raises(ValueError):
        F.slot_values(np.zeros((2, 2, 2, 2, 2)))
    # the scalar helpers only take [N, 69], never a raw stack
    with pytest.raises(ValueError):
        F.map_cov(np.zeros((4, 9, 9)))


def test_record_flatness_never_raises() -> None:
    for bad in (None, np.zeros((5, 5)), "not a map", object(), np.zeros((0,))):
        assert F.record_flatness(bad) == (None, None)


def test_record_flatness_returns_the_canonical_pair() -> None:
    rng = np.random.default_rng(2)
    vals = 1.0 + 0.05 * rng.standard_normal(69)
    legacy = np.stack([_plane_from_slots(vals)] + [np.full((9, 9), np.nan)] * 3)
    peak, cov = F.record_flatness(legacy)
    assert peak == pytest.approx(F.node_peak(vals)[0])
    assert cov == pytest.approx(F.map_cov(vals)[0])


# --------------------------------------------------------------------------- #
# report-only diagnostics
# --------------------------------------------------------------------------- #
def test_gradient_stats_of_a_flat_map_are_zero() -> None:
    gmax, gp90 = F.gradient_stats(np.ones(69))
    assert gmax[0] == pytest.approx(0.0)
    assert gp90[0] == pytest.approx(0.0)


def test_periphery_and_radial_diagnostics_are_finite() -> None:
    rng = np.random.default_rng(9)
    vals = 1.0 + 0.05 * rng.standard_normal((3, 69))
    assert np.isfinite(F.periphery_mean(vals)).all()
    assert np.isfinite(F.radial_weighted_power(vals)).all()
    # a flat map: every weighted average is 1.0
    assert F.periphery_mean(np.ones(69))[0] == pytest.approx(1.0)
    assert F.radial_weighted_power(np.ones(69))[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# "one definition only" (§1.1)
# --------------------------------------------------------------------------- #
def test_power_prior_uses_the_canonical_cov() -> None:
    from lpopt.model import power_prior

    assert power_prior._cov is F.map_cov


def test_ab_score_uses_the_canonical_scalars() -> None:
    pytest.importorskip("torch")
    from lpopt.model import ab_score

    assert ab_score._map_cov is F.map_cov
    assert ab_score._node_peak is F.node_peak
    # the local unweighted copy is gone, not shadowed
    assert not hasattr(ab_score, "_cov")


# --------------------------------------------------------------------------- #
# the real store (§1.1 measurement)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_maps() -> np.ndarray:
    if not MAPS.is_file():
        pytest.skip("store maps.npz not present")
    with np.load(MAPS) as z:
        keys = [k for k in z.files if not k.endswith(("__traj", "__axial"))][:200]
        if not keys:
            pytest.skip("no legacy map stacks in maps.npz")
        return np.stack([F.slot_values(z[k])[0] for k in keys])


def test_weighted_mean_of_real_maps_is_one(real_maps: np.ndarray) -> None:
    """The map is ALREADY normalized to the core average — under the WEIGHTED mean.

    This is the fact that makes ``node_peak = nanmax`` equal to F_xy, and the
    reason ``map_cov``'s denominator is a physical constant rather than a
    per-record accident.
    """
    means = F.weighted_mean(real_maps)
    assert np.abs(means - 1.0).max() < 1.0e-3


def test_unweighted_mean_of_real_maps_is_not_one(real_maps: np.ndarray) -> None:
    """The negative control: drop the weights and the normalization is gone."""
    plain = np.nanmean(real_maps, axis=1)
    assert np.abs(plain - 1.0).max() > 1.0e-2


def test_node_peak_of_real_maps_equals_nanmax(real_maps: np.ndarray) -> None:
    np.testing.assert_allclose(F.node_peak(real_maps),
                               np.nanmax(real_maps, axis=1))
