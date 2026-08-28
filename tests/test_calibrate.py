"""Calibration: isotonic monotonicity + Platt logistic (M3b)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from lpopt.model.calibrate import (          # noqa: E402
    _fit_isotonic, _fit_platt, apply_calibration, apply_platt,
)
from lpopt.model.dataset_torch import TARGETS      # noqa: E402


def test_isotonic_is_monotone() -> None:
    rng = np.random.default_rng(0)
    sigma = np.abs(rng.normal(1.0, 0.4, 500))
    # error magnitude grows with sigma + noise
    abs_err = 0.8 * sigma + np.abs(rng.normal(0, 0.2, 500))
    curve = _fit_isotonic(sigma, abs_err)
    # build a full calib dict and apply over a sorted sigma grid
    calib = {"targets": list(TARGETS),
             "isotonic": {name: curve for name in TARGETS}}
    grid = np.linspace(sigma.min(), sigma.max(), 200)
    sig5 = np.tile(grid[:, None], (1, len(TARGETS)))
    out = apply_calibration(sig5, calib)
    # non-decreasing in the input sigma for every target
    for k in range(len(TARGETS)):
        diffs = np.diff(out[:, k])
        assert (diffs >= -1e-9).all(), f"target {k} not monotone"


def test_isotonic_degenerate_identity() -> None:
    # too few / no-variance points -> identity passthrough
    curve = _fit_isotonic(np.full(5, 0.5), np.full(5, 0.1))
    calib = {"targets": list(TARGETS),
             "isotonic": {name: curve for name in TARGETS}}
    sig = np.full((3, len(TARGETS)), 0.5)
    out = apply_calibration(sig, calib)
    assert np.isfinite(out).all()


def test_platt_monotone_probability() -> None:
    rng = np.random.default_rng(1)
    logit = rng.normal(0, 2, 400)
    label = (logit + rng.normal(0, 0.5, 400) > 0).astype(int)
    platt = _fit_platt(logit, label.astype(float))
    calib = {"platt": platt}
    grid = np.linspace(-5, 5, 100)
    p = apply_platt(grid, calib)
    assert (np.diff(p) >= -1e-9).all()       # increasing in the logit
    assert (p >= 0).all() and (p <= 1).all()


def test_platt_degenerate_single_class() -> None:
    platt = _fit_platt(np.linspace(-1, 1, 20), np.ones(20))
    assert platt["degenerate"] is True


# --- regression: 8-target model vs 7-target calibration artifact ------------- #
# The champion's ensemble predicts 8 targets (promote_max_asm_bu adds
# ``max_assembly_burnup``) while its calibration.json lists only 7 — the
# freeze-finetune recipe copies the older artifact verbatim.  The original
# ``np.empty_like`` + "loop over the artifact's targets" left the 8th column
# UNINITIALIZED, so served ``max_assembly_burnup`` sigma was raw heap garbage.

_CAL7 = ["f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
         "discharge_burnup", "max_pin_burnup"]
_MODEL8 = _CAL7 + ["max_assembly_burnup"]


def _identity_calib(names):
    curve = {"x": [0.0, 10.0], "y": [0.0, 10.0]}       # identity interpolation
    return {"targets": list(names), "isotonic": {n: curve for n in names}}


def test_uncalibrated_target_keeps_raw_sigma_not_garbage() -> None:
    """A model target with no fitted curve must pass through its RAW sigma."""
    calib = _identity_calib(_CAL7)
    sigma = np.arange(1, 3 * len(_MODEL8) + 1, dtype=float).reshape(3, len(_MODEL8))
    for _ in range(20):                        # repeat: garbage would vary
        out = apply_calibration(sigma, calib, _MODEL8)
        assert np.isfinite(out).all()
        # the unfitted 8th column is the raw sigma, exactly
        assert np.array_equal(out[:, 7], sigma[:, 7])
    # and it is deterministic across calls
    assert np.array_equal(apply_calibration(sigma, calib, _MODEL8),
                          apply_calibration(sigma, calib, _MODEL8))


def test_calibration_maps_by_name_not_position() -> None:
    """A reordered/subset artifact lands on the right columns when names given."""
    # curve doubles sigma, and only for cyclen (index 3 of the model order)
    calib = {"targets": ["cyclen"],
             "isotonic": {"cyclen": {"x": [0.0, 10.0], "y": [0.0, 20.0]}}}
    sigma = np.ones((2, len(_MODEL8)))
    out = apply_calibration(sigma, calib, _MODEL8)
    assert np.allclose(out[:, 3], 2.0)                 # cyclen doubled
    untouched = [c for c in range(len(_MODEL8)) if c != 3]
    assert np.allclose(out[:, untouched], 1.0)         # everything else raw


def test_positional_fallback_preserved_without_names() -> None:
    """Legacy positional behaviour still holds when target_names is omitted."""
    calib = _identity_calib(_CAL7)
    sigma = np.full((2, len(_MODEL8)), 3.0)
    out = apply_calibration(sigma, calib)
    assert np.isfinite(out).all()
    assert np.array_equal(out[:, 7], sigma[:, 7])      # still no garbage
