"""Guards for the v1 move-proposal policy: leakage, splits, shapes, metrics.

These four claims are what the results report rests on, so they are tests and
not comments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lpopt.policy.data import (
    FORBIDDEN_COLUMNS, HELDOUT_CELL, HELDOUT_ERA_LIBRARIES, HELDOUT_LIBRARY,
    build_splits, load_universe, scalar_features,
)
from lpopt.policy.train import auc, parent_blocked_auc, precision_at_k

STEPS = "data/policy/steps.parquet"


@pytest.fixture(scope="module")
def universe() -> pd.DataFrame:
    try:
        return load_universe(STEPS)
    except FileNotFoundError:                       # corpus not present locally
        pytest.skip(f"{STEPS} missing")


def test_universe_is_same_cell_and_labeled(universe: pd.DataFrame) -> None:
    assert not universe["cross_cell"].any()
    assert (universe["improved_fr"].notna()
            | universe["improved_flat"].notna()).all()
    # sa_unknown is a FEATURE value and must survive the universe filter
    assert (universe["move_class"] == "sa_unknown").sum() > 0


def test_no_outcome_or_provenance_in_features(universe: pd.DataFrame) -> None:
    _, names = scalar_features(universe)
    assert not (set(names) & FORBIDDEN_COLUMNS)
    # the d_* ring descriptors ARE allowed: they are functions of the child
    # PATTERN, which the move determines, not of the child's evaluated outcome.
    assert "d_fresh_share_periph" in names
    assert not any(n.startswith("child_") for n in names)


def test_split_holds_out_whole_families_and_is_deterministic(
        universe: pd.DataFrame) -> None:
    fold = build_splits(universe, seed=20260815)
    assert fold.equals(build_splits(universe, seed=20260815))

    era = universe["library_id"].isin(HELDOUT_ERA_LIBRARIES)
    assert (fold[era] == "heldout_era").all()
    assert not (fold[~era] == "heldout_era").any()
    assert (fold[universe["cell"] == HELDOUT_CELL] == "heldout_cell").all()
    lib = (universe["library_id"] == HELDOUT_LIBRARY) & ~era
    assert (fold[lib] == "heldout_lib").all()


def test_no_lineage_component_straddles_train_and_test(
        universe: pd.DataFrame) -> None:
    """The whole point of the grouped split: no board links the two folds."""
    fold = build_splits(universe, seed=20260815)
    boards = {}
    for name in ("train", "test"):
        sub = universe[fold == name]
        boards[name] = set(sub["parent_record_id"]) | set(sub["child_record_id"])
    assert not (boards["train"] & boards["test"])


def test_transpose_leaves_the_scalar_features_alone(
        universe: pd.DataFrame) -> None:
    """Augmentation copies every column verbatim — so features must be equal."""
    sub = universe.head(64)
    a, _ = scalar_features(sub)
    mirrored = sub.copy()
    from lpopt.data.geometry import transpose
    from lpopt.data.schema import pack_pattern, unpack_pattern
    for side in ("parent", "child"):
        mirrored[f"{side}_pattern"] = [
            pack_pattern(transpose(unpack_pattern(p)))
            for p in sub[f"{side}_pattern"]]
    b, _ = scalar_features(mirrored)
    np.testing.assert_allclose(a, b)


def test_auc_matches_a_brute_force_pair_count() -> None:
    rng = np.random.default_rng(0)
    s, y = rng.random(60), (rng.random(60) > 0.5).astype(float)
    pos, neg = s[y > 0], s[y == 0]
    brute = ((pos[:, None] > neg[None, :]).sum()
             + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert auc(s, y) == pytest.approx(brute)
    assert np.isnan(auc(s, np.ones(60)))            # one class only


def test_auc_handles_ties_as_half_credit() -> None:
    assert auc(np.zeros(10), np.r_[np.ones(5), np.zeros(5)]) == pytest.approx(0.5)


def test_parent_blocked_auc_ignores_single_class_parents() -> None:
    scores = np.array([0.9, 0.1, 0.5, 0.4])
    labels = np.array([1.0, 0.0, 1.0, 1.0])
    parents = np.array(["a", "a", "b", "b"])
    value, n_pairs = parent_blocked_auc(scores, labels, parents)
    assert n_pairs == 1 and value == pytest.approx(1.0)   # parent b contributes 0


def test_precision_at_k_is_paired_and_uses_the_tiebreak() -> None:
    y = np.r_[np.ones(50), np.zeros(50)]
    perfect = np.r_[np.ones(50), np.zeros(50)]
    draws = np.tile(np.arange(100), (3, 1))
    tie = np.random.default_rng(0).random((3, 100))
    assert precision_at_k(perfect, y, draws=draws, tiebreak=tie, k=10).mean() == 1.0
    # an all-constant scorer must fall back to the base rate, not to array order
    flat = precision_at_k(np.zeros(100), y, draws=draws, tiebreak=tie, k=50)
    assert 0.2 < flat.mean() < 0.8
