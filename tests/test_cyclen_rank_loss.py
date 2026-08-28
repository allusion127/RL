"""Within-cell cyclen margin-rank loss + its (feed, e_core-bin, dataset) cell
codes (RL forensic 20260720).

The loss decouples the WITHIN-cell cyclen ordering signal (which the honest
no-regression gate scores) from the global z-scale that compresses within-cell
target spread to ~0.36 z — deep in Huber's quadratic regime where the ranking
gradient is weakest.  These tests pin: the loss is scale-free in the target and
only sign-sensitive, ignores cross-cell / sub-noise / invalid / unresolved-cell
pairs, is one-sided (only wrong or under-margin orderings pay), and that the
cell codes mirror the sampler's weighting cells with a -1 escape for NaN e_core.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.model.dataset_torch import cyclen_cell_codes                 # noqa: E402
from lpopt.model.train import cyclen_rank_loss                          # noqa: E402


def _t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


# --------------------------------------------------------------------------- #
# cyclen_rank_loss — pure behavior
# --------------------------------------------------------------------------- #
def test_perfect_within_cell_ranking_beyond_margin_is_zero():
    # one cell, prediction order matches truth order with a > margin z-gap.
    mu = _t([2.0, 1.0, 0.0])
    raw = _t([600.0, 590.0, 580.0])          # gaps 10 EFPD >> min_gap
    valid = _t([1, 1, 1])
    cell = _t([0, 0, 0], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    assert float(loss) == pytest.approx(0.0)


def test_anti_ranking_incurs_positive_loss():
    # prediction order is the exact reverse of truth -> every pair is penalized.
    mu = _t([0.0, 1.0, 2.0])
    raw = _t([600.0, 590.0, 580.0])          # truth: row0 > row1 > row2
    valid = _t([1, 1, 1])
    cell = _t([0, 0, 0], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    assert float(loss) > 0.5


def test_scale_free_in_target_magnitude():
    # scaling the RAW cyclen gaps (as a growing tstd would compress the z-gaps)
    # must NOT change the loss — it depends only on the sign of the raw gap.
    mu = _t([0.0, 0.5, 0.2])
    valid = _t([1, 1, 1])
    cell = _t([0, 0, 0], torch.long)
    a = cyclen_rank_loss(mu, _t([600., 590., 580.]), valid, cell,
                         margin=0.1, min_gap_efpd=2.0)
    b = cyclen_rank_loss(mu, _t([560., 500., 440.]), valid, cell,
                         margin=0.1, min_gap_efpd=2.0)
    assert float(a) == pytest.approx(float(b))


def test_cross_cell_pairs_are_ignored():
    # two singleton cells -> no same-cell pair -> zero regardless of prediction.
    mu = _t([5.0, -5.0])
    raw = _t([600.0, 500.0])
    valid = _t([1, 1])
    cell = _t([0, 1], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    assert float(loss) == pytest.approx(0.0)


def test_sub_min_gap_pairs_are_ignored():
    # raw gap 1 EFPD < min_gap 2 -> ordering is convergence noise, no pair formed.
    mu = _t([0.0, 1.0])                       # would be "wrong" if it counted
    raw = _t([600.0, 600.9])
    valid = _t([1, 1])
    cell = _t([0, 0], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    assert float(loss) == pytest.approx(0.0)


def test_invalid_and_unresolved_cell_rows_excluded():
    # row1 invalid (mask 0), row2 has cell -1 (NaN e_core) -> only row0 usable ->
    # < 2 usable rows -> zero.
    mu = _t([0.0, 9.0, -9.0])
    raw = _t([600.0, 550.0, 650.0])
    valid = _t([1, 0, 1])
    cell = _t([0, 0, -1], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    assert float(loss) == pytest.approx(0.0)


def test_one_sided_correct_but_within_margin_pays_only_the_deficit():
    # correctly ordered (mu0 > mu1) but only by 0.03 < margin 0.1 -> pays 0.07.
    mu = _t([0.53, 0.50])
    raw = _t([600.0, 580.0])
    valid = _t([1, 1])
    cell = _t([0, 0], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    assert float(loss) == pytest.approx(0.07, abs=1e-5)


def test_each_ordering_counted_once():
    # 3 rows, one cell, all gaps real: 3 ordered pairs (0>1,0>2,1>2); an
    # all-equal prediction pays exactly `margin` on each -> mean == margin.
    mu = _t([0.0, 0.0, 0.0])
    raw = _t([600.0, 590.0, 580.0])
    valid = _t([1, 1, 1])
    cell = _t([0, 0, 0], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    assert float(loss) == pytest.approx(0.1, abs=1e-6)


# --------------------------------------------------------------------------- #
# cyclen_cell_codes
# --------------------------------------------------------------------------- #
def test_cell_codes_group_by_feed_ebin_dataset():
    df = pd.DataFrame({
        "feed": [121, 121, 121, 117],
        "e_core": [5.41, 5.42, 5.55, 5.41],     # bins 5.40, 5.40, 5.55, 5.40
        "dataset": ["A", "A", "A", "A"],
    })
    codes = cyclen_cell_codes(df, e_core_bin_width=0.05)
    assert codes[0] == codes[1]                 # same feed+bin+dataset
    assert codes[2] != codes[0]                 # different e_core bin
    assert codes[3] != codes[0]                 # different feed
    assert codes.dtype == np.int64


def test_cell_codes_dataset_axis_separates():
    df = pd.DataFrame({
        "feed": [121, 121],
        "e_core": [5.41, 5.41],
        "dataset": ["A", "P"],                  # same feed+bin, different provenance
    })
    codes = cyclen_cell_codes(df)
    assert codes[0] != codes[1]


def test_cell_codes_nan_e_core_is_minus_one():
    df = pd.DataFrame({
        "feed": [121, 121],
        "e_core": [np.nan, 5.41],
        "dataset": ["P", "P"],
    })
    codes = cyclen_cell_codes(df)
    assert codes[0] == -1
    assert codes[1] >= 0


def test_cell_codes_empty():
    codes = cyclen_cell_codes(pd.DataFrame({"feed": [], "e_core": [], "dataset": []}))
    assert codes.shape == (0,)
    assert codes.dtype == np.int64


# --------------------------------------------------------------------------- #
# gradient sanity — the loss actually pushes a mis-ranked pair apart
# --------------------------------------------------------------------------- #
def test_gradient_pushes_misranked_pair_in_the_right_direction():
    mu = torch.tensor([0.0, 0.0], requires_grad=True)
    raw = _t([600.0, 580.0])                    # row0 should rank above row1
    valid = _t([1, 1])
    cell = _t([0, 0], torch.long)
    loss = cyclen_rank_loss(mu, raw, valid, cell, margin=0.1, min_gap_efpd=2.0)
    loss.backward()
    # descending the gradient raises mu[0] and lowers mu[1].
    assert mu.grad[0] < 0
    assert mu.grad[1] > 0
