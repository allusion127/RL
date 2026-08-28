"""Guards for the v2 move-proposal policy.

The v2 results report rests on five claims that a comment cannot enforce: the
target is the clipped expected improvement and not a relabelled binary, the gate
fold is current-era and lineage-clean, the era reweighting says what it claims,
v2's feature vector strictly contains v1's, and ``regret@8`` measures what its
name says.  Each is a test here.

The reactivity covariate has its own guard because it is the whole point of the
round: if ``fresh_enr_mass`` is not conserved by the conserving operators, it is
not the coordinate the post-mortem asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lpopt.policy.data import FORBIDDEN_COLUMNS, scalar_features
from lpopt.policy.v2 import (
    CURRENT_ERA_LIBRARIES, NEW_SCALARS, TARGET_CLIP, build_splits_v2,
    era_weights, load_universe_v2, scalar_features_v2, targets,
)

STEPS = "data/policy/steps.parquet"
#: Operators that conserve the total fresh reactivity exactly (ablation §2c).
CONSERVING = ("rewire_swap", "batch_swap")


@pytest.fixture(scope="module")
def universe() -> pd.DataFrame:
    try:
        return load_universe_v2(STEPS)
    except FileNotFoundError:                       # corpus not present locally
        pytest.skip(f"{STEPS} missing")


@pytest.fixture(scope="module")
def fold(universe: pd.DataFrame) -> pd.Series:
    return build_splits_v2(universe)


# --------------------------------------------------------------------------- #
# the covariate
# --------------------------------------------------------------------------- #
def test_fresh_enr_mass_is_in_the_schema_and_conserved_by_conserving_moves(
        universe: pd.DataFrame) -> None:
    for side in ("parent", "child", "d"):
        assert f"{side}_fresh_enr_mass" in universe.columns
    d = universe["d_fresh_enr_mass"].to_numpy(float)
    assert np.isfinite(d).all()

    keep = universe["move_class"].isin(CONSERVING).to_numpy()
    assert keep.sum() > 100
    assert np.abs(d[keep]).max() < 1e-9, "a conserving operator moved the mass"

    flip = (universe["move_class"] == "batch_flip").to_numpy()
    assert np.abs(d[flip]).max() > 1.0, "batch_flip must move the mass"


# --------------------------------------------------------------------------- #
# the target
# --------------------------------------------------------------------------- #
def test_target_is_clipped_expected_improvement_not_a_binary(
        universe: pd.DataFrame) -> None:
    y, mask = targets(universe)
    assert y.min() >= 0.0 and y.max() <= 1.0

    d = universe["d_f_r"].to_numpy(float)
    ok = mask[:, 0] > 0
    # worsening moves are exactly zero; improving moves are strictly positive
    assert (y[ok & (d >= 0), 0] == 0.0).all()
    assert (y[ok & (d < 0), 0] > 0.0).all()
    # and it is NOT a two-valued label: the whole point is magnitude resolution
    interior = y[ok & (d < 0), 0]
    assert len(np.unique(np.round(interior, 4))) > 20

    # the clip is where it is registered
    worst = -d[ok].min()
    assert worst > TARGET_CLIP["fr"], "test corpus lacks a saturating row"
    assert y[ok, 0].max() == pytest.approx(1.0)


def test_target_matches_its_formula_row_for_row(universe: pd.DataFrame) -> None:
    y, mask = targets(universe)
    c = TARGET_CLIP["fr"]
    want = np.clip(-universe["d_f_r"].to_numpy(float), 0.0, c) / c
    ok = mask[:, 0] > 0
    assert np.allclose(y[ok, 0], want[ok], atol=1e-6)


def test_unconverged_rows_are_masked_not_scored_zero(
        universe: pd.DataFrame) -> None:
    _, mask = targets(universe)
    both = universe["both_converged"].fillna(False).to_numpy(bool)
    assert (mask[~both] == 0.0).all()


# --------------------------------------------------------------------------- #
# the split
# --------------------------------------------------------------------------- #
def test_gate_is_current_era_only_and_deterministic(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    assert fold.equals(build_splits_v2(universe))
    era = universe["library_id"].isin(CURRENT_ERA_LIBRARIES)
    assert (universe.loc[fold == "gate_cur", "library_id"]
            .isin(CURRENT_ERA_LIBRARIES).all())
    assert not (fold[~era] == "gate_cur").any()
    # and it is a real split, not everything or nothing
    assert 0.2 < (fold[era] == "gate_cur").mean() < 0.8


def test_no_current_era_lineage_component_straddles_gate_and_train(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    from lpopt.policy.data import _components

    cur = universe[universe["era_current"]]
    comp = pd.Series(_components(cur), index=cur.index)
    side = fold.loc[cur.index]
    per_component = pd.crosstab(comp, side == "gate_cur")
    if per_component.shape[1] == 2:
        straddling = ((per_component[True] > 0) & (per_component[False] > 0))
        assert not straddling.any(), "a lineage component is on both sides"


def test_gate_carries_enough_parents_for_the_deployment_metric(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    from lpopt.policy.train_v2 import REGRET_MIN_CANDIDATES

    gate = universe[fold == "gate_cur"]
    counts = gate[gate["improved_fr"].notna()].groupby("parent_record_id").size()
    assert int((counts >= REGRET_MIN_CANDIDATES).sum()) >= 5


# --------------------------------------------------------------------------- #
# the reweighting
# --------------------------------------------------------------------------- #
def test_era_weighting_equalises_the_two_eras(universe: pd.DataFrame,
                                              fold: pd.Series) -> None:
    w = era_weights(universe, fold)
    train = (fold == "train").to_numpy()
    era = universe["era_current"].to_numpy(bool)
    assert (w[~era] == 1.0).all()
    mass_cur = w[train & era].sum()
    mass_leg = w[train & ~era].sum()
    # the cap may bind; it must never overshoot, and must land near parity
    assert 0.45 <= mass_cur / (mass_cur + mass_leg) <= 0.55


# --------------------------------------------------------------------------- #
# the features
# --------------------------------------------------------------------------- #
def test_v2_features_strictly_contain_v1s(universe: pd.DataFrame) -> None:
    base, v1_names = scalar_features(universe)
    x, names = scalar_features_v2(universe)
    assert set(names) == set(v1_names) | set(NEW_SCALARS)
    for i, n in enumerate(v1_names):
        assert np.array_equal(x[:, names.index(n)], base[:, i]), n


def test_no_outcome_or_provenance_in_v2_features(universe: pd.DataFrame) -> None:
    _, names = scalar_features_v2(universe)
    assert not (set(names) & FORBIDDEN_COLUMNS)
    assert not any(n.startswith("child_") for n in names)
    # the era IS allowed (a cell attribute); the lineage source is NOT
    assert "era_current" in names
    assert "lineage_source" not in names


# --------------------------------------------------------------------------- #
# the new metric
# --------------------------------------------------------------------------- #
def test_regret_at_k_is_zero_for_an_oracle_and_maximal_for_an_antioracle() -> None:
    from lpopt.policy.train_v2 import regret_at_k

    rng = np.random.default_rng(0)
    gain = rng.normal(size=40)
    parents = np.array(["p"] * 20 + ["q"] * 20)
    oracle, _, keys = regret_at_k(gain, gain, parents, k=8, min_candidates=10)
    assert len(keys) == 2
    assert np.allclose(oracle, 0.0)

    anti, _, _ = regret_at_k(-gain, gain, parents, k=8, min_candidates=10)
    assert (anti > 0).all()


def test_regret_at_k_skips_parents_that_cannot_make_a_selection() -> None:
    from lpopt.policy.train_v2 import regret_at_k

    gain = np.arange(16, dtype=float)
    parents = np.array(["small"] * 8 + ["big"] * 8)
    _, _, keys = regret_at_k(gain, gain, parents, k=8, min_candidates=10)
    assert len(keys) == 0, "a parent with exactly k candidates gives free zero regret"


def test_normalized_regret_is_bounded(universe: pd.DataFrame,
                                      fold: pd.Series) -> None:
    from lpopt.policy.train_v2 import regret_at_k

    gate = universe[fold == "gate_cur"]
    ok = gate["both_converged"].fillna(False).to_numpy(bool) & gate["d_f_r"].notna()
    g = gate[ok]
    gain = -g["d_f_r"].to_numpy(float)
    rng = np.random.default_rng(1)
    _, norm, keys = regret_at_k(rng.random(len(g)), gain,
                                g["parent_record_id"].to_numpy())
    assert len(keys) >= 5
    finite = norm[np.isfinite(norm)]
    assert (finite >= 0).all() and (finite <= 1).all()
