"""fr_boundary objective (F_r=1.55 licensing-boundary training campaign,
user directive 2026-07-22).

F_r is a PURE minimization objective (never gated); CBC / F_q / |AO| + a predicted
pin-BU screen are the hard gates; a MID-TIER band penalty on the F_r MEAN keeps the
search inside the plausibility window without letting cyclen leak into ordering.
"""

from __future__ import annotations

import numpy as np
import pytest

from lpopt.search import acquisition as acq
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction


# 7-col surrogate mean layout: (F_r, CBC, F_q, cyclen, AO, _, pin_bu).
def _pred(rows, std=None):
    """rows: list of (f_r, cbc, f_q, cyclen, ao[, pin_bu]).  Missing pin_bu -> NaN."""
    n = len(rows)
    m = np.full((n, 7), np.nan)
    for i, r in enumerate(rows):
        m[i, :5] = r[:5]
        if len(r) > 5:
            m[i, 6] = r[5]
    if std is None:
        s = np.zeros((n, 7))
    else:
        s = np.asarray(std, dtype=float)
    return SurrogatePrediction(m, s.copy(), s.copy())


def _spec(**kw):
    return acq.MinFrBoundarySpec(**kw)


# --------------------------------------------------------------------------- #
# (a) cyclen invariance — col 3 mu/sigma must never move the score or rank
# --------------------------------------------------------------------------- #
def test_cyclen_invariance_score_and_rank():
    spec = _spec()
    base = [[1.56, 1400.0, 2.30, 625.0, 0.20, 68.0],
            [1.60, 1400.0, 2.30, 625.0, 0.20, 68.0]]
    fb0 = acq.score_fr_boundary(_pred(base), spec)
    # perturb cyclen mean AND sigma wildly on both rows.
    m = np.full((2, 7), np.nan)
    for i, r in enumerate(base):
        m[i, :5] = r[:5]; m[i, 6] = r[5]
    m[:, 3] = [999.0, -400.0]
    s = np.zeros((2, 7)); s[:, 3] = [50.0, 80.0]
    fb1 = acq.score_fr_boundary(SurrogatePrediction(m, s.copy(), s.copy()), spec)
    assert fb1.total == pytest.approx(fb0.total)
    # rank order (row 0 = lower F_r) preserved and unchanged by cyclen.
    assert fb0.total[0] > fb0.total[1]
    assert fb1.total[0] > fb1.total[1]


# --------------------------------------------------------------------------- #
# (b) F_r monotonicity at fixed sigma — lower in-band F_r ranks higher
# --------------------------------------------------------------------------- #
def test_fr_monotonic_at_fixed_sigma():
    spec = _spec()
    rows = [[1.50, 1400.0, 2.30, 625.0, 0.20, 68.0],
            [1.55, 1400.0, 2.30, 625.0, 0.20, 68.0],
            [1.60, 1400.0, 2.30, 625.0, 0.20, 68.0]]
    fb = acq.score_fr_boundary(_pred(rows), spec)
    assert fb.total[0] > fb.total[1] > fb.total[2]   # strictly minimizing F_r
    assert fb.scalar[0] == pytest.approx(-1.50)


def test_fr_enters_at_ucb():
    spec = _spec(risk_z=0.25)
    m = np.full((1, 7), np.nan); m[:, :5] = [1.55, 1400.0, 2.30, 625.0, 0.20]; m[:, 6] = 68.0
    s = np.zeros((1, 7)); s[:, 0] = 0.4
    fb = acq.score_fr_boundary(SurrogatePrediction(m, s.copy(), s.copy()), spec)
    assert fb.fr_ucb[0] == pytest.approx(1.55 + 0.25 * 0.4)
    assert fb.fr_mean[0] == pytest.approx(1.55)      # band uses the MEAN, not UCB


# --------------------------------------------------------------------------- #
# (c) CBC / F_q / |AO| tier penalty ordering
# --------------------------------------------------------------------------- #
def test_constraint_violation_dominates_lower_fr():
    spec = _spec()
    rows = [
        [1.62, 1400.0, 2.30, 625.0, 0.20, 68.0],   # feasible, F_r 1.62
        [1.50, 1600.0, 2.30, 625.0, 0.20, 68.0],   # CBC 1600 > 1550 -> infeasible
    ]
    fb = acq.score_fr_boundary(_pred(rows), spec)
    assert bool(fb.constraint_ok[0]) and not bool(fb.constraint_ok[1])
    # the lower-F_r candidate has the higher raw scalar but must sink below feasible.
    assert fb.scalar[1] > fb.scalar[0]
    assert fb.total[0] > fb.total[1]


def test_least_infeasible_first_before_any_feasible():
    spec = _spec()
    rows = [
        [1.50, 1560.0, 2.30, 625.0, 0.20, 68.0],   # CBC 1560 (10 over)
        [1.50, 1700.0, 2.30, 625.0, 0.20, 68.0],   # CBC 1700 (150 over)
    ]
    fb = acq.score_fr_boundary(_pred(rows), spec)
    assert not bool(fb.constraint_ok[0]) and not bool(fb.constraint_ok[1])
    assert fb.total[0] > fb.total[1]                 # least-infeasible ranks first


# --------------------------------------------------------------------------- #
# (d) pin-BU finite-mean certification + UCB when sigma finite
# --------------------------------------------------------------------------- #
def test_pin_bu_certified_on_finite_mean():
    spec = _spec(pin_bu_limit=80.0)
    # pin BU present and under limit -> certifies; over limit -> penalized + not ok.
    ok = acq.score_fr_boundary(_pred([[1.55, 1400.0, 2.30, 625.0, 0.20, 68.0]]), spec)
    over = acq.score_fr_boundary(_pred([[1.55, 1400.0, 2.30, 625.0, 0.20, 88.0]]), spec)
    assert bool(ok.constraint_ok[0])
    assert not bool(over.constraint_ok[0])
    assert ok.total[0] > over.total[0]
    # missing pin BU (NaN mean) cannot certify feasibility.
    missing = acq.score_fr_boundary(_pred([[1.55, 1400.0, 2.30, 625.0, 0.20]]), spec)
    assert not bool(missing.constraint_ok[0])


def test_pin_bu_ucb_when_sigma_finite():
    spec = _spec(pin_bu_limit=80.0, risk_z=0.25)
    m = np.full((1, 7), np.nan); m[:, :5] = [1.55, 1400.0, 2.30, 625.0, 0.20]; m[:, 6] = 78.0
    s = np.zeros((1, 7)); s[:, 6] = 40.0            # UCB 78 + 0.25*40 = 88 > 80
    fb = acq.score_fr_boundary(SurrogatePrediction(m, s.copy(), s.copy()), spec)
    assert not bool(fb.constraint_ok[0])            # certified on the +kappa*sigma UCB


# --------------------------------------------------------------------------- #
# (e) band gate — NEAR-band step+slope invariant (the critique fix)
# --------------------------------------------------------------------------- #
def test_band_near_low_below_in_band():
    spec = _spec(band_lo=1.45, band_hi=1.70)
    # 1.44 is just below band_lo; 1.56 is in-band with HIGHER F_r.  The step+slope
    # penalty must still rank the OOD-low 1.44 BELOW the in-band 1.56 (100x fix).
    fb = acq.score_fr_boundary(
        _pred([[1.44, 1400.0, 2.30, 625.0, 0.20, 68.0],
               [1.56, 1400.0, 2.30, 625.0, 0.20, 68.0]]), spec)
    assert fb.total[1] > fb.total[0]
    assert fb.band_penalty[0] > 0.0 and fb.band_penalty[1] == 0.0


def test_band_near_high_below_in_band():
    spec = _spec(band_lo=1.45, band_hi=1.70)
    # 1.71 just above band_hi must rank below in-band 1.69.
    fb = acq.score_fr_boundary(
        _pred([[1.71, 1400.0, 2.30, 625.0, 0.20, 68.0],
               [1.69, 1400.0, 2.30, 625.0, 0.20, 68.0]]), spec)
    assert fb.total[1] > fb.total[0]


def test_band_penalty_uses_mean_not_ucb():
    # High-sigma OOD-low candidate cannot dodge the band penalty via UCB inflation.
    spec = _spec(band_lo=1.45, band_hi=1.70, risk_z=0.25)
    m = np.full((1, 7), np.nan); m[:, :5] = [1.30, 1400.0, 2.30, 625.0, 0.20]; m[:, 6] = 68.0
    s = np.zeros((1, 7)); s[:, 0] = 1.0             # UCB would be 1.55 (in-band)
    fb = acq.score_fr_boundary(SurrogatePrediction(m, s.copy(), s.copy()), spec)
    assert fb.band_penalty[0] > 0.0                 # mean 1.30 is out-of-band -> penalized


# --------------------------------------------------------------------------- #
# (h) band penalty stays BELOW the constraint TIER
# --------------------------------------------------------------------------- #
def test_band_below_constraint_tier():
    spec = _spec()
    # in-band but CBC-violating vs out-of-band but fully feasible: the feasible
    # out-of-band candidate must outrank the in-band constraint violator (band term
    # is two orders below the 1e4 constraint tier).
    fb = acq.score_fr_boundary(
        _pred([[1.56, 1650.0, 2.30, 625.0, 0.20, 68.0],   # in-band, CBC 1650 over
               [1.30, 1400.0, 2.30, 625.0, 0.20, 68.0]]), spec)  # OOB but feasible
    assert not bool(fb.constraint_ok[0]) and bool(fb.constraint_ok[1])
    assert fb.total[1] > fb.total[0]


# --------------------------------------------------------------------------- #
# (f) constraints sentinel: p_feasible ignores F_r
# --------------------------------------------------------------------------- #
def test_constraints_ignore_fr():
    spec = _spec()
    c = acq.make_fr_boundary_constraints(spec)
    assert c.f_r_limit >= 1.0e12
    pred = _pred([[2.50, 1400.0, 2.30, 625.0, 0.20, 68.0]])   # F_r 2.5 (huge)
    pf = acq.p_feasible(pred, c)
    assert pf[0] == pytest.approx(1.0)                        # F_r contributes no gate


# --------------------------------------------------------------------------- #
# (g) NaN-sigma -> no shift, constraint_ok False
# --------------------------------------------------------------------------- #
def test_nan_sigma_no_shift_not_certified():
    spec = _spec(risk_z=0.25)
    m = np.full((1, 7), np.nan); m[:, :5] = [1.55, 1400.0, 2.30, 625.0, 0.20]; m[:, 6] = 68.0
    s = np.full((1, 7), np.nan)                     # every axis unknown
    fb = acq.score_fr_boundary(SurrogatePrediction(m, s.copy(), s.copy()), spec)
    assert fb.fr_ucb[0] == pytest.approx(1.55)      # no UCB shift when sigma NaN
    assert not bool(fb.constraint_ok[0])            # cannot certify with unknown sigma
