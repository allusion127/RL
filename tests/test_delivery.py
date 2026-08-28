"""Delivery-candidate selection — program §2.2 / decision D2.

The rule under test is a REJECTION rule as much as a ranking one: the flattest
point is NOT what gets delivered.
"""

from __future__ import annotations

import numpy as np
import pytest

from lpopt.search import delivery as D


def _rows(peaks, f_rs, covs=None):
    covs = covs if covs is not None else [0.30] * len(peaks)
    return [{"record_id": f"r{i}", "node_peak": p, "map_cov": c, "f_r": f}
            for i, (p, c, f) in enumerate(zip(peaks, covs, f_rs))]


# --------------------------------------------------------------------------- #
# compliance margin
# --------------------------------------------------------------------------- #
def test_compliance_margin_is_headroom_to_1p55():
    assert D.LICENSING_FR_LIMIT == 1.55
    assert D.compliance_margin(1.50) == pytest.approx(0.05)
    assert D.compliance_margin(1.60) == pytest.approx(-0.05)
    assert D.compliance_margin(None) is None
    assert D.compliance_margin("x") is None


def test_compliance_margin_applies_the_prediction_bias_correction():
    # a head that UNDER-predicts F_r by 0.04 (bias = -0.04 in pred-minus-actual
    # terms) is corrected by passing the bias to be added back.
    assert D.compliance_margin(1.50, bias=-0.04) == pytest.approx(1.55 - 1.54)


# --------------------------------------------------------------------------- #
# within-cell percentile
# --------------------------------------------------------------------------- #
def test_percentile_is_ascending_so_flat_is_low():
    p = D.within_cell_percentile([1.30, 1.50, 1.70, 1.90])
    assert p[0] < p[1] < p[2] < p[3]
    assert p[0] == pytest.approx(0.0)


def test_percentile_ties_share_an_average_rank():
    p = D.within_cell_percentile([1.5, 1.5, 1.9, 2.0])
    assert p[0] == pytest.approx(p[1])


def test_percentile_ignores_non_finite():
    p = D.within_cell_percentile([1.3, np.nan, 1.9])
    assert np.isnan(p[1]) and np.isfinite(p[0]) and np.isfinite(p[2])


# --------------------------------------------------------------------------- #
# the §2.2 rule
# --------------------------------------------------------------------------- #
def test_the_flattest_point_is_not_delivered():
    """THE rule.  Compliant rows sit at within-cell peak percentile ~0.22, not 0."""
    peaks = np.linspace(1.30, 1.90, 20)
    # give the FLATTEST candidate the best F_r too, so only the band can exclude it
    f_rs = np.linspace(1.50, 1.75, 20)
    rep = D.select_delivery(_rows(peaks, f_rs))
    assert rep.banded
    ids = [c.record_id for c in rep.ranked]
    assert "r0" not in ids                      # the flattest point is excluded
    reasons = {c.record_id: c.reason for c in rep.excluded}
    assert reasons["r0"] == "flatter than the band"
    # ...and it is excluded despite having the LARGEST compliance margin.
    flattest = next(c for c in rep.excluded if c.record_id == "r0")
    assert flattest.compliance_margin == max(
        c.compliance_margin for c in rep.ranked + rep.excluded
        if c.compliance_margin is not None)


def test_band_bounds_are_the_010_040_percentiles():
    peaks = np.linspace(1.30, 1.90, 20)
    rep = D.select_delivery(_rows(peaks, [1.52] * 20))
    kept = {c.record_id for c in rep.ranked}
    # 20 rows -> percentiles 0.00, 0.05, ... 0.95; band [0.10, 0.40] keeps r2..r8
    assert kept == {f"r{i}" for i in range(2, 9)}
    assert all(D.BAND_LO <= c.peak_percentile <= D.BAND_HI for c in rep.ranked)


def test_ranking_inside_the_band_is_by_compliance_margin():
    peaks = np.linspace(1.30, 1.90, 20)
    f_rs = [1.60] * 20
    f_rs[5] = 1.50          # best margin, inside the band
    f_rs[6] = 1.53
    rep = D.select_delivery(_rows(peaks, f_rs))
    assert rep.ranked[0].record_id == "r5"
    assert rep.ranked[1].record_id == "r6"
    assert rep.ranked[0].compliance_margin == pytest.approx(0.05)
    # descending margin throughout
    margins = [c.compliance_margin for c in rep.ranked]
    assert margins == sorted(margins, reverse=True)


def test_a_candidate_exactly_on_the_limit_sorts_by_value_not_last():
    """margin == 0.0 is a real value; `margin or -inf` would bury it."""
    peaks = np.linspace(1.30, 1.90, 20)
    f_rs = [1.60] * 20
    f_rs[5] = 1.55          # margin exactly 0.0, inside the band
    rep = D.select_delivery(_rows(peaks, f_rs))
    assert rep.ranked[0].record_id == "r5"
    assert rep.ranked[0].compliance_margin == pytest.approx(0.0)


def test_rows_without_a_flatness_label_or_fr_are_excluded_by_name():
    rows = _rows([1.4, 1.5, 1.6, 1.7, 1.8], [1.5, 1.5, 1.5, 1.5, 1.5])
    rows[2]["node_peak"] = None
    rows[3]["f_r"] = None
    rep = D.select_delivery(rows)
    reasons = {c.record_id: c.reason for c in rep.excluded}
    assert reasons["r2"] == "no flatness label"
    assert reasons["r3"] == "no F_r"


def test_too_few_rows_disables_the_band_and_says_so():
    rep = D.select_delivery(_rows([1.4, 1.5, 1.6], [1.50, 1.52, 1.54]))
    assert not rep.banded
    assert len(rep.ranked) == 3                 # nothing dropped for percentile
    assert rep.ranked[0].record_id == "r0"      # ranked purely by margin


def test_top_k_truncates_ranked_without_calling_it_excluded():
    peaks = np.linspace(1.30, 1.90, 20)
    rep = D.select_delivery(_rows(peaks, np.linspace(1.75, 1.50, 20)), top_k=2)
    assert len(rep.ranked) == 2
    assert all(c.reason != "in band" for c in rep.excluded)


def test_report_is_json_serializable_and_counts_everything():
    peaks = np.linspace(1.30, 1.90, 12)
    rep = D.select_delivery(_rows(peaks, [1.52] * 12))
    d = rep.as_dict()
    assert d["n_rows"] == 12 and d["n_scored"] == 12
    assert len(d["ranked"]) + len(d["excluded"]) == 12
    import json
    json.dumps(d)


def test_empty_input_is_an_empty_report():
    rep = D.select_delivery([])
    assert rep.ranked == [] and rep.excluded == [] and rep.n_rows == 0
