"""Task #5 regression: the OPSCREEN model-free FLOOR chain as pure functions.

The published numbers being pinned here come from
``5_RL/opmodel/OPSCREEN.md``, ``5_RL/opmodel/s16_out.txt`` and the fitted
``frmodel2.npz``.  No file in ``opmodel/`` is imported: the point of
``lpopt/design/opscreen_chain.py`` is that the chain runs without it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lpopt.design import opscreen_chain as C


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
def test_node_peak_coefficients_match_published() -> None:
    """OPSCREEN.md:241 quotes 1.4210 - 4.1725*contrast - 3.4862*d_fresh."""
    c0, c1, c2 = C.NODE_PEAK_COEFFS
    assert round(c0, 4) == 1.4210
    assert round(c1, 4) == -4.1725
    assert round(c2, 4) == -3.4862
    assert C.NODE_PEAK_N == 15
    assert C.NODE_PEAK_RMS == pytest.approx(0.036)
    assert C.NODE_PEAK_R2 == pytest.approx(0.866)


def test_two_A_constants_are_distinct_and_labelled() -> None:
    """s16 uses the least-squares A (frmodel2), OPSCREEN.md quotes the ratio
    mean 1.035 +- 0.031.  Conflating them shifts every Fr_fix by ~0.005."""
    assert C.A_FUSION == pytest.approx(1.0319443879524353, abs=1e-12)
    assert round(C.A_RATIO_MEAN, 3) == 1.035
    assert C.A_SIGMA == pytest.approx(0.031)
    assert abs(C.A_RATIO_MEAN - C.A_FUSION) > 3e-3


def test_floor_constants() -> None:
    assert C.FR_FLOOR_A == 1.03
    assert C.FR_FLOOR_NODE_PEAK == 1.2085
    assert C.CONTRAST_MIN == 0.026
    assert C.HUMP_CAL_MAX == 0.0148


# --------------------------------------------------------------------------- #
# (2)  s16_out.txt  FEED 121  [B] #3  reproduction
# --------------------------------------------------------------------------- #
# s16_out.txt:31 (feed 121, block [B], rank 3):
#     cyc 623.5  raw 622.2  CBC 1406  FFhot 1.1208  contr +0.0308
#     hump 0.0056  npk 1.293  Fr_fix 1.495  Fr_flr 1.395  new 2
#       68: u5.50/4.6750 gd8x20 2:2;4:1;6:3 PB/z1
#       53: u5.00/4.2500 gd10x20 1:1;4:1;6:4 PB/z1
S16_B3_FF_HOT = 1.1208
S16_B3_CONTRAST = 0.0308
S16_B3_HUMP = 0.0056
S16_B3_NPK = 1.293
S16_B3_FR_FIX = 1.495
S16_B3_FR_FLR = 1.395


def _d_fresh_for(target_npk: float, contrast: float) -> float:
    """Invert the regression for the d_fresh that the cached screen row had.

    s16_out.txt does not print d_fresh, so it is recovered from the printed
    (contrast, node_peak).  The recovered value must be small, which is the
    independent check that the inversion is not absorbing an error.
    """
    c0, c1, c2 = C.NODE_PEAK_COEFFS
    return (c0 + c1 * contrast - target_npk) / (-c2)


def test_s16_feed121_blockB_rank3_fusion_and_floor_reproduce() -> None:
    """Only the ``f_r_fixed`` / ``f_r_floor`` legs reproduce ``s16_out.txt:31``.

    ``s16_out.txt`` does not print the third regressor, so ``_d_fresh_for``
    back-solves it from the printed ``node_peak``; re-deriving ``node_peak``
    from it is an inversion identity, not a reproduction.  The regression
    coefficients themselves are pinned non-circularly by
    :func:`test_s16_node_peak_coefficients_are_pinned_without_inversion`.
    """
    dfresh = _d_fresh_for(S16_B3_NPK, S16_B3_CONTRAST)
    assert abs(dfresh) < 1e-3, "recovered d_fresh should be ~0 for this row"

    npk = C.node_peak(S16_B3_CONTRAST, dfresh)
    assert round(npk, 3) == S16_B3_NPK      # identity, by construction

    assert round(C.f_r_fixed(npk, S16_B3_FF_HOT), 3) == S16_B3_FR_FIX
    assert round(C.f_r_floor(S16_B3_FF_HOT), 3) == S16_B3_FR_FLR

    res = C.evaluate(contrast=S16_B3_CONTRAST, dfresh=dfresh,
                     ff_hot=S16_B3_FF_HOT)
    assert round(res.node_peak, 3) == S16_B3_NPK
    assert round(res.f_r_fixed, 3) == S16_B3_FR_FIX
    assert round(res.f_r_floor, 3) == S16_B3_FR_FLR
    assert res.contrast_ok is True          # 0.0308 >= 0.026
    assert not C.hump_extrapolates(S16_B3_HUMP)   # block [B] == no extrapolation


def test_s16_node_peak_coefficients_are_pinned_without_inversion() -> None:
    """The non-circular leg: with the third regressor set to zero (the printed
    row's recovered ``d_fresh`` is -1.5e-4), the regression must land on the
    printed ``node_peak`` 1.293 to within the printed precision.  This pins c0
    and c1 against ``s16_out.txt:31`` with nothing back-solved."""
    npk = C.node_peak(S16_B3_CONTRAST, 0.0)
    assert npk == pytest.approx(S16_B3_NPK, abs=1e-3)
    # and it is the coefficients, not a coincidence: perturb c1 and it moves out
    bad = C.node_peak(S16_B3_CONTRAST, 0.0,
                      coeffs=(C.NODE_PEAK_COEFFS[0], C.NODE_PEAK_COEFFS[1] + 1.0,
                              C.NODE_PEAK_COEFFS[2]))
    assert bad != pytest.approx(S16_B3_NPK, abs=1e-3)


def test_contrast_window_conventions_are_pinned() -> None:
    """s09b: 0.5..bc/3 with 20 points.  s16: 0.5..8.0 with 12 points.  Mixing
    the s16 window with the s09b point count is a silent convention error."""
    assert C.CONTRAST_BU_LO == 0.5
    assert C.CONTRAST_NPTS == 20
    assert C.CONTRAST_BU_HI_FIXED == 8.0
    assert C.CONTRAST_NPTS_FIXED == 12


def test_ratio_mean_A_would_not_reproduce_the_table() -> None:
    """Guard against silently swapping A_RATIO_MEAN in for A_FUSION."""
    wrong = C.f_r_fixed(S16_B3_NPK, S16_B3_FF_HOT, a=C.A_RATIO_MEAN)
    assert round(wrong, 3) != S16_B3_FR_FIX


# --------------------------------------------------------------------------- #
# (1)  T3-T6 FF values and the measured arms
# --------------------------------------------------------------------------- #
# OPSCREEN.md:175 -- surrogate ensemble / DeCART %DIST, held out:
#     T3 1.1073/1.1090, T4 1.1409/1.1430, T5 1.1012/1.1020, T6 1.1011/1.1020
T3_T6_FF = {
    "T3": (1.1073, 1.1090),
    "T4": (1.1409, 1.1430),
    "T5": (1.1012, 1.1020),
    "T6": (1.1011, 1.1020),
}


def test_t3_t6_ff_transfer_bias() -> None:
    """The surrogate FF runs 0.0014 low with rms 0.0015 and max 0.0021
    (OPSCREEN.md:176-177).  Those are the G-H4 thresholds, so pin them."""
    d = [s - dec for s, dec in T3_T6_FF.values()]
    assert round(sum(d) / len(d), 4) == -0.0014
    assert round(math.sqrt(sum(x * x for x in d) / len(d)), 4) == 0.0015
    assert round(max(abs(x) for x in d), 4) == 0.0021


def test_t3_t6_floor_chain_on_published_ff() -> None:
    """The floor is a pure function of FF_hot, so the four T rows are exact."""
    for name, (ff_surrogate, ff_decart) in T3_T6_FF.items():
        flr = C.f_r_floor(ff_surrogate)
        assert flr == pytest.approx(1.03 * 1.2085 * ff_surrogate, abs=1e-12)
        # the surrogate's 0.0014 low bias is worth ~0.0017 of F_r floor
        assert abs(C.f_r_floor(ff_decart) - flr) < 0.003, name


# --------------------------------------------------------------------------- #
# (3)  d_fresh is NOT hump   (v2 mis-reading, locked as a regression)
# --------------------------------------------------------------------------- #
def _curve(bu, rho):
    bu = np.asarray(bu, float)
    rho = np.asarray(rho, float)

    def f(x):
        return np.interp(np.asarray(x, float), bu, rho)
    return f


def _gd_pair():
    """Two synthetic rho(BU) curves with a real Gd hump, on the BU >= 0.2 grid."""
    bu = np.array([0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0,
                   20.0, 30.0, 40.0, 50.0, 60.0])
    # hi role: hump peaks near BU 8; lo role: flatter, lower reactivity
    hi = np.array([0.240, 0.243, 0.248, 0.256, 0.268, 0.276, 0.279, 0.277,
                   0.272, 0.256, 0.240, 0.196, 0.150, 0.100, 0.045])
    lo = hi - 0.030
    return _curve(bu, hi), _curve(bu, lo)


def test_d_fresh_and_hump_are_different_quantities() -> None:
    hi, lo = _gd_pair()
    rm = C.mix_rho([hi, lo], [68.0, 53.0])
    bc = 20.0
    df = C.d_fresh(rm, 121, bc)
    hp = C.mixture_hump(rm)

    assert hp > 0.0                     # the Gd overshoot is positive by build
    assert df != pytest.approx(hp, abs=1e-6)
    assert abs(df - hp) > 0.01, (df, hp)


def test_measured_d_fresh_and_published_hump_disagree_in_sign() -> None:
    """The v2 draft read the OPSCREEN ``hump`` column as ``d_fresh``.

    For the T3_T4 pair at feed 121 they have **opposite signs**: the published
    hump is +0.0128 (OPSCREEN.md:276) while the measured d_fresh of the same
    arm (B2) is -0.00655.  Substituting one for the other moves node_peak by
    -3.4862 * 0.01935 = -0.0675, i.e. 1.9x the rms of the whole regression.
    """
    b2 = C.measured_arm("B2")
    hump = C.PUBLISHED_HUMP["T3_T4@121"]
    assert b2["d_fresh"] < 0.0 < hump
    npk_right = C.node_peak(b2["contrast"], b2["d_fresh"])
    npk_wrong = C.node_peak(b2["contrast"], hump)
    assert abs(npk_right - npk_wrong) == pytest.approx(0.0675, abs=5e-4)
    assert abs(npk_right - npk_wrong) > 1.8 * C.NODE_PEAK_RMS
    # NB: on this one arm the wrong substitution happens to land closer to the
    # measured node_peak (the fit carries a +0.046 residual on B2).  That is a
    # coincidence of one row, not evidence -- d_fresh is the regressor the fit
    # was built on (s09b_contrast.py:27), and the substitution is simply a
    # different variable.
    assert C.PUBLISHED_HUMP["T5_T6@121"] != C.measured_arm("B3")["d_fresh"]


def test_hump_ignores_feed_and_d_fresh_does_not() -> None:
    hi, lo = _gd_pair()
    rm = C.mix_rho([hi, lo], [68.0, 53.0])
    assert C.d_fresh(rm, 121, 20.0) != pytest.approx(C.d_fresh(rm, 101, 20.0))
    # mixture_hump has no feed argument at all -- the property under test
    assert "feed" not in C.mixture_hump.__code__.co_varnames


def test_batch_weights_census_sums_to_241() -> None:
    for feed in (101, 117, 121, 125, 141):
        bw = C.batch_weights(feed)
        assert sum(n for n, _ in bw) == C.NSLOT
        assert bw[0] == (float(feed), 0)


# --------------------------------------------------------------------------- #
# contrast + the B3 counterexample
# --------------------------------------------------------------------------- #
def test_role_contrast_sign_and_window() -> None:
    hi, lo = _gd_pair()
    c = C.role_contrast(hi, lo, bu_hi=8.0)
    assert c == pytest.approx(0.030, abs=1e-9)
    assert C.role_contrast(lo, hi, bu_hi=8.0) == pytest.approx(-c, abs=1e-12)


def test_contrast_gate_boundary() -> None:
    assert C.contrast_gate(0.026) is True
    assert C.contrast_gate(0.0259999) is False
    assert C.contrast_gate(-0.00156) is False       # arm B3


def test_b3_counterexample_is_documented_and_true() -> None:
    d = C.b3_counterexample()
    assert d["flatter_is_B3"] is True        # B3 has the lower FF_hot
    assert d["worse_f_r_is_B3"] is True      # ... and the worse measured F_r
    assert d["B3"]["contrast_ok"] is False   # the gate is what catches it
    assert d["B2"]["contrast_ok"] is True
    assert d["B3"]["f_r"] == 1.5795
    assert d["B3"]["ff_hot"] == 1.1020
    # ranking on FF_hot alone inverts the true F_r order -- the whole point
    assert d["B3"]["ff_hot"] < d["B2"]["ff_hot"]
    assert d["B3"]["f_r"] > d["B2"]["f_r"]
    # and the module docstring carries the disclaimer verbatim
    assert "not as a prediction" in C.__doc__
    assert "1.5795" in C.__doc__


def test_floor_disclaimer_is_in_the_function_docstrings() -> None:
    doc = " ".join(C.f_r_floor.__doc__.split())
    assert "best value ever measured" in doc
    assert "not a prediction" in doc.lower()


# --------------------------------------------------------------------------- #
# interval + F_xy
# --------------------------------------------------------------------------- #
def test_chain_returns_an_interval_not_a_point() -> None:
    res = C.evaluate(contrast=0.030, dfresh=0.0, ff_hot=1.12)
    lo, hi = res.f_r_bounds
    assert lo == C.f_r_floor(1.12)
    assert hi == C.f_r_fixed(res.node_peak, 1.12)
    assert res.f_r_bounds == C.f_r_interval(res.node_peak, 1.12)


def test_f_xy_scale() -> None:
    assert C.f_xy(1.5, r=1.0) == pytest.approx(1.5)
    assert C.f_xy(1.5) == pytest.approx(1.5 * C.F_XY_RATIO_MEAN)
    # store-measured spreads, quoted in the retrodiction report
    assert C.F_XY_RATIO_MEAN == pytest.approx(1.0780)
    assert C.F_XY_RATIO_SIGMA == pytest.approx(0.0299)
    assert C.F_XY_RATIO_MEAN_F121 == pytest.approx(1.0693)


def test_error_budget_is_dominated_by_A() -> None:
    b = C.error_budget(1.12)
    assert b["sigma_from_A"] > b["sigma_from_node_peak"]
    assert b["sigma_f_r"] == pytest.approx(
        math.hypot(b["sigma_from_A"], b["sigma_from_node_peak"]))
    # ~0.042 on F_r at FF_hot 1.12 -- an order above the 0.005 decision bar
    assert 0.03 < b["sigma_f_r"] < 0.06


def test_as_table_renders() -> None:
    rows = [C.evaluate(contrast=0.031, dfresh=0.0, ff_hot=1.1208),
            C.evaluate(contrast=-0.002, dfresh=0.017, ff_hot=1.1020)]
    txt = C.as_table(rows)
    assert "Fr_flr" in txt and "OK" in txt and "LOW" in txt
    assert len(txt.splitlines()) == 3
