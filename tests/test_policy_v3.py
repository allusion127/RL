"""Guards for the v3 move-proposal policy.

The v3 pre-registration (``data/reports/policy_v3_prereg_20260831.md``) makes
claims a comment cannot enforce, and each one is a test here — the v3 answer to
v2's thirteen guards:

* the ``fxy`` target is the CONSTRAINT-GATED clipped expected improvement, and
  the gate is the registered one (§2a/§2b, reproduced row for row);
* the split is clean — the held-out cell reaches no other fold, no lineage
  component straddles the gate, and the alternation is REVERSED (§3a);
* v3's feature vector strictly CONTAINS v2's, which in turn contains v1's (§4d);
* the Gd descriptors obey their conservation law: an operator that moves no
  fresh assembly moves no Gd mass, and the diagonal mirror moves none either —
  which is what makes the transpose augmentation still label-preserving;
* the coverage flag is real: ``gd_wt`` is missing for every ga80 type, so
  ``gdwt_present`` must be 0 there and the mass must be zeroed, and that is the
  registered first suspect if the transfer bar fails (§7);
* ``regret@4-of-8`` and NDCG@4 measure what their names say;
* the weighting says what it claims — every parent equal in the ``fxy`` head,
  the two eras equal overall;
* serving reproduces the training features for the new columns, and the v3
  loader refuses a checkpoint from another feature contract.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mine_policy_corpus as M
from lpopt.policy.data import FORBIDDEN_COLUMNS, scalar_features
from lpopt.policy.v2 import (
    NEW_SCALARS, POLICY_SCHEMA_V2, scalar_features_v2, targets,
)
from lpopt.policy.v3 import (
    CYCLEN_TOL, F_R_LIMIT, HEADS_V3, INTERVENTIONAL_SOURCES_V3, NEW_SCALARS_V3,
    POLICY_SCHEMA_V3, PROSPECTIVE_CELL, STEPS_V3, TARGET_CLIP_V3,
    build_splits_v3, fxy_feasible, load_universe_v3, parent_weights,
    provenance_v3, scalar_features_v3, split_summary_v3, targets_v3, weights_v3,
)

STEPS = STEPS_V3
FUEL_TYPES = Path("data/store/fuel_types.parquet")
STORE = Path("data/store/records.parquet")

#: The realized §3b table.  Registered numbers, so a change to the universe rule,
#: the alternation or the val draw shows up as a failing test and not as a quiet
#: re-definition of the fold the gate is computed on.
FOLD_ROWS = {"train": 17848, "val": 1982, "gate_cur": 1137,
             "prospective_cell": 167}
#: Current-era rows carrying an F_xy reading, and the registered §2b effect of
#: the two constraint terms on them.
N_FXY_CURRENT = 1309
N_FXY_POSITIVE = 184
MEAN_GATED_GAIN = 0.00274


@pytest.fixture(scope="module")
def universe() -> pd.DataFrame:
    try:
        return load_universe_v3(STEPS)
    except FileNotFoundError:                       # v3 corpus not built locally
        pytest.skip(f"{STEPS} missing "
                    f"(python policy_v2_corpus.py backfill-v3 --apply)")


@pytest.fixture(scope="module")
def fold(universe: pd.DataFrame) -> pd.Series:
    return build_splits_v3(universe)


# --------------------------------------------------------------------------- #
# 1. the corpus
# --------------------------------------------------------------------------- #
def test_v3_corpus_carries_the_twenty_two_new_columns(universe: pd.DataFrame) -> None:
    assert len(M.V3_SCHEMA_COLUMNS) == 22
    missing = [c for c in M.V3_SCHEMA_COLUMNS if c not in universe.columns]
    assert not missing, missing
    # and the v2 corpus's own columns are all still there, unrenamed
    for name in M.PHYSICS:
        for side in ("parent", "child", "d"):
            assert f"{side}_{name}" in universe.columns


def test_the_five_intervention_waves_are_registered_as_interventional(
        universe: pd.DataFrame) -> None:
    tags = set(universe["lineage_source"].unique())
    waves = {t for t in tags if t.startswith("intervention_")}
    assert len(waves) == 5
    assert waves <= set(INTERVENTIONAL_SOURCES_V3)
    cur = universe[universe["era_current"].to_numpy(bool)]
    # prereg §1b: 48.9% of current-era same-cell rows are interventional
    assert 0.45 < cur["interventional"].mean() < 0.55


def test_gd_coverage_is_what_the_prereg_measured(universe: pd.DataFrame) -> None:
    """§4c: ``n_gd`` / ``kinf0`` 100% on the live era, ``gd_wt`` paramA-only."""
    cur = universe[universe["era_current"].to_numpy(bool)]
    for col in ("fresh_gd_mass", "fresh_kinf0_mass"):
        ok = (np.isfinite(cur[f"parent_{col}"].to_numpy(float))
              & np.isfinite(cur[f"child_{col}"].to_numpy(float)))
        assert ok.all(), f"{col} unresolved on {int((~ok).sum())} current-era rows"

    gdwt = np.isfinite(cur["d_fresh_gdwt_mass"].to_numpy(float))
    ga80 = (cur["library_id"] == "ga80").to_numpy()
    assert not gdwt[ga80].any(), "gd_wt is absent for every ga80 type (§4c)"
    assert gdwt[~ga80].all(), "gd_wt is present for every paramA type (§4c)"
    # the held-out cell is ga80, which is why this is the registered first
    # suspect for a transfer failure (§7)
    assert PROSPECTIVE_CELL.endswith("ga80")


# --------------------------------------------------------------------------- #
# 2. the Gd descriptors — conservation laws
# --------------------------------------------------------------------------- #
def test_the_gd_descriptors_obey_their_conservation_laws(
        universe: pd.DataFrame) -> None:
    """Measured, and each law says which operator the descriptor can see.

    * a REWIRE moves no fresh assembly, so it moves no Gd quantity at all;
    * a BATCH_SWAP permutes two fresh labels between equal-multiplicity slots,
      so every Gd MASS is conserved exactly and only the radial moments move —
      the same shape v2 documented for ``fresh_enr_mass``;
    * ``fresh_relocate`` / ``batch_flip`` change which fresh types are loaded and
      move the masses.

    The middle law has a consequence the results report has to carry: on the
    ``HGD569_f125`` pair, whose two types share ``n_gd`` = 24 and differ only in
    ``gd_wt``, a ``batch_swap`` moves NONE of the six §4a descriptors.
    """
    mass = ("fresh_gd_mass", "fresh_gdwt_mass", "fresh_kinf0_mass")
    moment = ("fresh_gd_r_center", "fresh_gd_share_periph",
              "fresh_kinf0_r_center")
    klass = universe["move_class"].to_numpy()

    rewire = np.isin(klass, ("rewire_swap", "rewire_multi"))
    assert rewire.sum() > 100
    for name in (*mass, *moment):
        d = universe[f"d_{name}"].to_numpy(float)
        keep = rewire & np.isfinite(d)
        assert np.abs(d[keep]).max() == 0.0, f"a rewire moved {name}"

    swap = klass == "batch_swap"
    assert swap.sum() > 100
    for name in mass:
        d = universe[f"d_{name}"].to_numpy(float)
        keep = swap & np.isfinite(d)
        assert np.abs(d[keep]).max() < 1e-9, f"batch_swap moved {name}"
    moved = [np.nanmax(np.abs(universe.loc[swap, f"d_{n}"].to_numpy(float)))
             for n in moment]
    assert max(moved) > 0.0, "batch_swap must move a radial moment"

    for name in mass:
        d = universe[f"d_{name}"].to_numpy(float)
        keep = np.isin(klass, ("fresh_relocate", "batch_flip")) & np.isfinite(d)
        assert np.abs(d[keep]).max() > 0.0, f"{name} is blind to a fresh reload"


def test_the_hgd569_pair_is_invisible_to_the_registered_gd_descriptors(
        universe: pd.DataFrame) -> None:
    """A MEASURED limit of the registered feature set, asserted so it is not lost.

    Prereg §4c expects ``gd_wt`` to be what separates ``HGD569_f125``'s
    ``batch_swap`` moves.  It does not: the registered ``gd_wt`` feature is the
    MASS ``d_fresh_gdwt_mass``, and a label swap between equal-multiplicity slots
    conserves it exactly, while the ``n_gd``-weighted quantities are blind
    because that pair is 24/24.  Five of the six §4a descriptors and
    ``fresh_gd_contrast`` are identically zero on those 39 rows; what is left is
    the kinf0 first moment, at |d| <= 0.0017 against the 0.35 an ``E1E2``
    relocation moves.  Not fixed here — §4d fixes the twelve features and the
    round is pre-registered — but recorded, because it is the honest reading of a
    v3 win or loss on that slice (§7).
    """
    rows = universe[universe["lineage_source"] == "intervention_HGD569_f125"]
    swap = rows[rows["move_class"] == "batch_swap"]
    if swap.empty:
        pytest.skip("HGD569 wave not in this corpus")
    for name in M.GD_PHYSICS:
        d = np.nanmax(np.abs(swap[f"d_{name}"].to_numpy(float)))
        # 3e-14 on a 137-unit kinf0 mass is float cancellation, not a signal
        if name == "fresh_kinf0_r_center":
            assert 0.0 < d < 0.002, name
        else:
            assert d < 1e-9, name
    assert (swap["fresh_gd_contrast"].to_numpy(float) == 0.0).all()


def test_gd_descriptors_survive_the_diagonal_mirror(universe: pd.DataFrame) -> None:
    """The transpose augmentation must stay label-preserving for the new columns.

    ``transpose`` maps every slot to one of EQUAL radius and EQUAL orbit
    multiplicity, so a multiplicity-weighted radial moment cannot move.  If it
    did, the p=0.5 mirror in ``PolicySteps`` would be injecting noise into the
    very axis §4 adds.
    """
    from lpopt.data.geometry import transpose
    from lpopt.data.schema import pack_pattern, unpack_pattern

    table = M.load_fuel_table(FUEL_TYPES)
    rows = universe[universe["era_current"].to_numpy(bool)].head(60)
    for pat, lib in zip(rows["child_pattern"], rows["library_id"], strict=True):
        types = table.get(str(lib))
        before = M.gd_profile(pat, types)
        after = M.gd_profile(pack_pattern(transpose(unpack_pattern(pat))), types)
        for name in M.GD_PHYSICS:
            a, b = before[name], after[name]
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == pytest.approx(b, rel=0, abs=1e-9), name


def test_fresh_gd_contrast_is_zero_exactly_when_no_fresh_identity_changes(
        universe: pd.DataFrame) -> None:
    changed = universe["n_fresh_type_changed"].to_numpy(float)
    contrast = universe["fresh_gd_contrast"].to_numpy(float)
    complete = universe["gd_table_complete"].to_numpy(bool)
    assert (contrast[complete & (changed == 0)] == 0.0).all()
    # a batch_swap in the E1E2 family exchanges a 20-rod and a 24-rod type
    e1e2 = universe["cell"].str.startswith("E1_E2").to_numpy()
    swap = (universe["move_class"] == "batch_swap").to_numpy()
    live = e1e2 & swap & (changed > 0)
    if live.any():
        assert contrast[live].max() >= 4.0


# --------------------------------------------------------------------------- #
# 3. the target
# --------------------------------------------------------------------------- #
def test_fxy_target_matches_its_formula_row_for_row(universe: pd.DataFrame) -> None:
    y, mask = targets_v3(universe)
    assert y.shape[1] == len(HEADS_V3) == 3
    assert y.min() >= 0.0 and y.max() <= 1.0

    c = TARGET_CLIP_V3["fxy"]
    gain = np.clip(-universe["d_f_xy"].to_numpy(float), 0.0, None)
    want = np.clip(np.where(fxy_feasible(universe), gain, 0.0), 0.0, c) / c
    ok = mask[:, 2] > 0
    assert np.allclose(y[ok, 2], want[ok], atol=1e-6, equal_nan=False)

    # the fr / flat heads are v2's, byte for byte
    y2, m2 = targets(universe)
    assert np.array_equal(y[:, :2], y2) and np.array_equal(mask[:, :2], m2)


def test_fxy_mask_is_the_reading_and_convergence_not_the_gate(
        universe: pd.DataFrame) -> None:
    _, mask = targets_v3(universe)
    want = (universe["d_f_xy"].notna().to_numpy()
            & universe["both_converged"].fillna(False).to_numpy(bool))
    assert np.array_equal(mask[:, 2] > 0, want)
    # an infeasible row is MASKED IN with target 0 — it is a real observation of
    # "no usable improvement", not a missing one
    y, _ = targets_v3(universe)
    infeasible = want & ~fxy_feasible(universe)
    if infeasible.any():
        assert (y[infeasible, 2] == 0.0).all()


def test_the_registered_constraint_gate_reproduces_prereg_2b(
        universe: pd.DataFrame) -> None:
    """230 raw improvements -> 184 positives; mean gated gain 0.00274."""
    cur = universe[universe["era_current"].to_numpy(bool)
                   & universe["d_f_xy"].notna().to_numpy()]
    assert len(cur) == N_FXY_CURRENT
    raw = np.clip(-cur["d_f_xy"].to_numpy(float), 0.0, None)
    gated = np.where(fxy_feasible(cur), raw, 0.0)
    assert int((gated > 0).sum()) == N_FXY_POSITIVE
    assert gated.mean() == pytest.approx(MEAN_GATED_GAIN, abs=5e-6)

    # the two terms are the registered ones and neither is doing nothing
    d_cyclen = cur["d_cyclen"].to_numpy(float)
    assert int((d_cyclen < -CYCLEN_TOL).sum()) == 70
    assert int((cur["child_f_r"].to_numpy(float) <= F_R_LIMIT).sum()) == 374


def test_clip_saturates_only_the_one_registered_cell(
        universe: pd.DataFrame) -> None:
    """§2c: the four improvements above 0.060 are all ``HGD569_f125``."""
    inter = universe[universe["interventional"].to_numpy(bool)]
    y, mask = targets_v3(inter)
    sat = (y[:, 2] >= 1.0) & (mask[:, 2] > 0)
    assert 0 < sat.sum() <= 6
    cells = set(inter.loc[sat, "cell"])
    assert len(cells) == 1 and "f125" in next(iter(cells))


# --------------------------------------------------------------------------- #
# 4. the splits
# --------------------------------------------------------------------------- #
def test_folds_partition_the_universe_at_the_registered_sizes(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    assert fold.value_counts().to_dict() == FOLD_ROWS
    assert len(universe) == sum(FOLD_ROWS.values()) == 21134


def test_the_held_out_cell_reaches_no_other_fold(universe: pd.DataFrame,
                                                 fold: pd.Series) -> None:
    hold = (universe["cell"] == PROSPECTIVE_CELL).to_numpy()
    assert hold.sum() == FOLD_ROWS["prospective_cell"]
    assert (fold[hold] == "prospective_cell").all()
    assert (fold[~hold] != "prospective_cell").all()
    # and no board of the held-out cell appears as a parent or child elsewhere
    ids = set(universe.loc[hold, "parent_record_id"]) | set(
        universe.loc[hold, "child_record_id"])
    other = universe[~hold]
    assert not (set(other["parent_record_id"]) & ids)
    assert not (set(other["child_record_id"]) & ids)


def test_no_current_era_lineage_component_straddles_gate_and_train(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    from lpopt.policy.data import _components

    cur = universe[universe["era_current"].to_numpy(bool)
                   & (fold != "prospective_cell").to_numpy()]
    comp = pd.Series(_components(cur), index=cur.index)
    for key, group in comp.groupby(comp):
        assert fold.loc[group.index].nunique() == 1, key


def test_the_alternation_is_reversed_relative_to_v2(universe: pd.DataFrame,
                                                    fold: pd.Series) -> None:
    """Rank 0 (the largest F_xy component) goes to TRAIN, not to the gate.

    Declared in §3a on label counts alone.  With v2's direction the ``fxy`` head
    would train on 488 rows; the test asserts the realized numbers so a silent
    flip back cannot happen.
    """
    _, mask = targets_v3(universe)
    fxy = mask[:, 2] > 0
    n_train = int((fxy & (fold == "train").to_numpy()).sum())
    n_gate = int((fxy & (fold == "gate_cur").to_numpy()).sum())
    assert (n_train, n_gate) == (530, 540)


def test_gate_and_transfer_folds_carry_the_parents_the_metric_needs(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    summary = split_summary_v3(universe, fold).set_index("fold")
    assert int(summary.loc["gate_cur", "n_parents_ge8"]) == 38
    assert int(summary.loc["prospective_cell", "n_parents_ge8"]) == 19
    assert int(summary.loc["gate_cur", "y_fxy_pos"]) == 65
    assert int(summary.loc["prospective_cell", "y_fxy_pos"]) == 30
    # legacy rows never reach the gate
    assert int(summary.loc["gate_cur", "n_current"]) == FOLD_ROWS["gate_cur"]


# --------------------------------------------------------------------------- #
# 5. the weighting
# --------------------------------------------------------------------------- #
def test_weights_are_per_head_and_the_parent_term_is_fxy_only(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    w = weights_v3(universe, fold)
    assert w.shape == (len(universe), len(HEADS_V3))
    assert np.array_equal(w[:, 0], w[:, 1]), "fr and flat carry the era weight only"

    era = universe["era_current"].to_numpy(bool)
    assert w[era, 0][0] == pytest.approx(13.06, abs=0.01)
    assert (w[~era, 0] == 1.0).all()
    # the two eras carry the same training loss mass, which is the claim
    train = (fold == "train").to_numpy()
    assert (w[train & era, 0].sum() == pytest.approx(w[train & ~era, 0].sum(),
                                                     rel=1e-6))


def test_every_parent_carries_the_same_fxy_weight(universe: pd.DataFrame,
                                                  fold: pd.Series) -> None:
    pw = parent_weights(universe, fold)
    _, mask = targets_v3(universe)
    fxy = mask[:, 2] > 0
    assert (pw[~fxy] == 1.0).all(), "the parent term is scoped to the fxy head"

    frame = pd.DataFrame({"p": universe["parent_record_id"], "w": pw})[fxy]
    totals = frame.groupby("p")["w"].sum()
    assert totals.max() == pytest.approx(totals.min(), rel=1e-6)
    assert totals.iloc[0] == pytest.approx(3.605, abs=0.01)


# --------------------------------------------------------------------------- #
# 6. the features
# --------------------------------------------------------------------------- #
def test_v3_features_strictly_contain_v2s_which_contain_v1s(
        universe: pd.DataFrame) -> None:
    v1, v1_names = scalar_features(universe)
    v2, v2_names = scalar_features_v2(universe)
    x, names = scalar_features_v3(universe)

    assert len(names) == 51 and len(v2_names) == 39 and len(v1_names) == 36
    assert set(names) == set(v2_names) | set(NEW_SCALARS_V3)
    assert set(v2_names) == set(v1_names) | set(NEW_SCALARS)
    for i, n in enumerate(v2_names):
        assert np.array_equal(x[:, names.index(n)], v2[:, i]), n
    for i, n in enumerate(v1_names):
        assert np.array_equal(x[:, names.index(n)], v1[:, i]), n


def test_no_outcome_or_provenance_or_diagnostic_in_v3_features(
        universe: pd.DataFrame) -> None:
    _, names = scalar_features_v3(universe)
    assert not (set(names) & FORBIDDEN_COLUMNS)
    assert not any(n.startswith("child_") for n in names)
    assert "gd_table_complete" not in names, "a diagnostic, not a feature (§9)"
    assert "lineage_source" not in names
    assert "era_current" in names


def test_absent_gd_wt_is_zeroed_and_flagged(universe: pd.DataFrame) -> None:
    x, names = scalar_features_v3(universe)
    present = x[:, names.index("gdwt_present")]
    mass = x[:, names.index("d_fresh_gdwt_mass")]
    assert set(np.unique(present)) <= {0.0, 1.0}
    assert (mass[present == 0.0] == 0.0).all()
    assert np.isfinite(x).all(), "a NaN reached the conditioning vector"

    ga80 = (universe["library_id"] == "ga80").to_numpy()
    cur = universe["era_current"].to_numpy(bool)
    assert (present[ga80 & cur] == 0.0).all()


# --------------------------------------------------------------------------- #
# 7. the metrics
# --------------------------------------------------------------------------- #
def test_regret_at_4_of_8_is_zero_for_an_oracle_and_positive_for_an_antioracle(
) -> None:
    from lpopt.policy.train_v2 import regret_at_k
    from lpopt.policy.train_v3 import PROBE_K, REGRET_MIN_CANDIDATES

    assert (PROBE_K, REGRET_MIN_CANDIDATES) == (4, 8)
    rng = np.random.default_rng(0)
    gain = rng.normal(size=16)
    parents = np.array(["p"] * 8 + ["q"] * 8)
    oracle, _, keys = regret_at_k(gain, gain, parents, k=PROBE_K,
                                  min_candidates=REGRET_MIN_CANDIDATES)
    assert len(keys) == 2 and np.allclose(oracle, 0.0)
    anti, _, _ = regret_at_k(-gain, gain, parents, k=PROBE_K,
                             min_candidates=REGRET_MIN_CANDIDATES)
    assert (anti > 0).all()


def test_regret_at_8_would_be_identically_zero_on_this_corpus() -> None:
    """§1f — the reason k moved to 4 is the data's shape, not a softer bar."""
    from lpopt.policy.train_v2 import regret_at_k

    gain = np.arange(8, dtype=float)
    parents = np.array(["p"] * 8)
    absolute, _, keys = regret_at_k(gain[::-1], gain, parents, k=8,
                                    min_candidates=8)
    assert len(keys) == 1 and absolute[0] == 0.0


def test_ndcg_at_4_is_one_for_an_oracle_and_lower_for_an_antioracle() -> None:
    from lpopt.policy.train_v3 import ndcg_at_k

    gain = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    parents = np.array(["p"] * 8)
    best, keys = ndcg_at_k(gain, gain, parents)
    assert len(keys) == 1 and best[0] == pytest.approx(1.0)
    worst, _ = ndcg_at_k(-gain, gain, parents)
    assert 0.0 <= worst[0] < best[0]


# --------------------------------------------------------------------------- #
# 8. provenance and serving
# --------------------------------------------------------------------------- #
def test_provenance_v3_is_the_store_truth_for_every_servable_library() -> None:
    """§8-A: the v3 corpus is featurized with what a SERVED pattern will be.

    Re-derived from the live store rather than trusted: every library's campaign
    rows carry ``sym_class="rot61"`` and the store's own ``dataset``, and
    ``provenance_v3`` must answer exactly that.  The 574 ``free69`` rows are the
    historical ``extract_b`` ga80 harvest, which no serve path can ever produce.
    """
    if not STORE.is_file():
        pytest.skip(f"{STORE} missing")
    store = pd.read_parquet(STORE, columns=["library_id", "dataset", "sym_class"])
    census = store.groupby(["library_id", "dataset", "sym_class"]).size()
    for (lib, dataset, sym_class), n in census.items():
        got = provenance_v3(str(lib))
        # Only A / not-A reaches the encoder (g_dataset_flag), which is the
        # equivalence ``build_pattern_cache`` enforces row by row.
        assert (got[0] == "A") == (str(dataset) == "A"), (lib, dataset, got)
        if sym_class == "free69":
            # the historical extract_b ga80 harvest; no served pattern can be
            # written this way, so serve_provenance must NOT reproduce it
            assert n < 0.01 * len(store)
            assert got[1] == "rot61"
        else:
            assert got[1] == str(sym_class), (lib, sym_class, got)


def test_v3_pattern_cache_uses_a_different_provenance_than_v2(
        universe: pd.DataFrame) -> None:
    from lpopt.policy.data import corpus_provenance

    assert provenance_v3("ga80") == ("P", "rot61")
    assert corpus_provenance("ga80") == ("P", "free69")   # what v2 must keep
    assert provenance_v3("paramA") == corpus_provenance("paramA") == ("P", "rot61")


def _serving_frame(row: pd.Series):
    """A one-child move frame built through ``MoveScorerV3.move_frame``."""
    from lpopt.data.schema import unpack_pattern
    from lpopt.policy.scorer import MoveScorerV3

    scorer = MoveScorerV3(
        members=[], encoder=types.SimpleNamespace(n_channels=1),
        fuel=None, enrichment=M.load_enrichment(FUEL_TYPES),
        delta_channels=[], scalar_names=[], fuel_types=FUEL_TYPES)
    ctx = types.SimpleNamespace(feed=int(row["feed"]), pair=str(row["case_pair"]),
                                library_id=str(row["library_id"]))
    parent = (M.genome_of(row["parent_pattern"]),
              unpack_pattern(row["parent_pattern"]))
    child = (M.genome_of(row["child_pattern"]),
             unpack_pattern(row["child_pattern"]))
    return scorer.move_frame(parent, [child], ctx)


def test_serving_reproduces_the_corpus_gd_columns_row_for_row(
        universe: pd.DataFrame) -> None:
    """Train/serve parity for the 22 new columns, without a checkpoint.

    The whole point of ``move_frame`` is that a corpus row and a proposal-time
    row for the same (parent, child) pair are the SAME row.  Anything else and
    the model is served a feature vector it was never trained on — the failure
    the 2026-08-29 provenance forensic found the hard way.
    """
    rows = universe[universe["era_current"].to_numpy(bool)
                    & universe["interventional"].to_numpy(bool)]
    assert len(rows) > 10
    for _, row in rows.head(6).iterrows():
        frame = _serving_frame(row)
        for col in M.V3_SCHEMA_COLUMNS:
            served, mined = frame[col].iloc[0], row[col]
            if isinstance(mined, (bool, np.bool_)):
                assert bool(served) == bool(mined), col
            elif np.isnan(float(mined)):
                assert np.isnan(float(served)), col
            else:
                assert float(served) == pytest.approx(float(mined), abs=1e-9), col


def test_serving_renders_the_full_51_scalar_vector(universe: pd.DataFrame) -> None:
    row = universe[universe["era_current"].to_numpy(bool)].iloc[0]
    frame = _serving_frame(row)
    served, names = scalar_features_v3(frame)
    corpus, corpus_names = scalar_features_v3(universe.loc[[row.name]])
    assert names == corpus_names and len(names) == 51
    assert np.allclose(served[0], corpus[0], atol=1e-6)


# --------------------------------------------------------------------------- #
# 9. the serving stamp
# --------------------------------------------------------------------------- #
def _meta(**over: object) -> dict:
    meta = {"cond_schema": "v6b", "policy_schema": POLICY_SCHEMA_V3,
            "policy_version": "v3", "era_libraries": ["ga80", "paramA"],
            "scalar_names": ["a"], "net_config": {"n_heads": 3}}
    meta.update(over)
    return meta


def test_v3_loader_accepts_only_its_own_contract() -> None:
    from lpopt.policy.scorer import MoveScorerV3

    MoveScorerV3._check_meta(_meta(), Path("cnn_seed1"))
    for bad in (_meta(policy_schema=POLICY_SCHEMA_V2),
                _meta(policy_version="v2"),
                _meta(era_libraries=["ga80"]),
                _meta(net_config={"n_heads": 2}),
                _meta(scalar_names=[])):
        with pytest.raises(ValueError):
            MoveScorerV3._check_meta(bad, Path("cnn_seed1"))


def test_the_three_serving_paths_are_distinct_and_registered() -> None:
    from lpopt.policy.scorer import HEAD_INDEX_V3, SCORERS
    from lpopt.search.construct import POLICY_MODES

    assert set(SCORERS) == {"v1", "v2", "v3"}
    assert SCORERS["v3"].HEADS == HEADS_V3
    assert HEAD_INDEX_V3["fxy"] == 2
    # the deck values, and the head each one ranks on
    assert POLICY_MODES["v3"] == ("v3", "fxy")
    assert POLICY_MODES["shadow_v3"] == ("shadow_v3", "fxy")


def test_the_launcher_command_is_the_registered_one() -> None:
    """§8b, verbatim except the declared corpus-path deviation."""
    import train_policy_v3 as L

    args = L.main.__globals__["argparse"].Namespace(
        seeds=5, base_seed=20260831, epochs=120, patience=15, batch_size=256,
        lr="1e-3", weight_decay="1e-4", width=112, n_blocks=6, protocol="revB",
        num_workers=8, extra="")
    cmd = L.train_args(args)
    for token in ("--steps data/policy/steps_v3.parquet",
                  "--fuel-types data/store/fuel_types.parquet",
                  "--cache data/policy/_feature_cache_v3.npz",
                  "--v2-baseline data/design/policy_v3_v2_baseline.csv",
                  '--holdout-cell "N1_N2/f113/ga80"',
                  "--seeds 5", "--base-seed 20260831", "--epochs 120",
                  "--patience 15", "--batch-size 256", "--lr 1e-3",
                  "--weight-decay 1e-4", "--width 112", "--n-blocks 6",
                  "--protocol revB", "--device auto", "--num-workers 8"):
        assert token in cmd, token


def test_the_blind_v2_baseline_exists_and_covers_the_eval_folds(
        universe: pd.DataFrame, fold: pd.Series) -> None:
    """§5c: emitted and hashed BEFORE the corpus changed and before any weight."""
    path = Path("data/design/policy_v3_v2_baseline.csv")
    if not path.is_file():
        pytest.skip(f"{path} missing")
    csv = pd.read_csv(path)
    assert {"child_record_id", "p_improve_fr", "p_improve_flat"} <= set(csv.columns)
    evaluated = universe[fold.isin(("gate_cur", "val", "prospective_cell")
                                   ).to_numpy()]
    assert set(evaluated["child_record_id"]) == set(csv["child_record_id"])
    assert csv["p_improve_fr"].between(0.0, 1.0).all()


def test_gate_thresholds_are_the_registered_ones() -> None:
    from lpopt.policy import train_v3 as T

    assert T.GATE_AUC == 0.65 and T.GATE_AUC_CI_LO == 0.50
    assert T.TRANSFER_AUC == 0.60
    assert T.BASELINES == ("random", "class_freq", "periph", "gd_rule",
                           "policy_v2")
    # a PASS needs BOTH clauses; there is no partial credit
    empty = T.gate_verdict({"folds": {}})
    assert empty["PASS"] is False and empty["transfer_bar"]["PASS"] is False


def test_the_v2_artefacts_this_round_depends_on_are_untouched() -> None:
    """The v3 corpus is a NEW file; v2's corpus and checkpoints keep their bytes."""
    from lpopt.policy.data import corpus_fingerprint

    v2_steps = Path("data/policy/steps.parquet")
    meta = Path("data/models/policy_v2/cnn_seed20260817/meta.json")
    if not (v2_steps.is_file() and meta.is_file()):
        pytest.skip("v2 corpus or checkpoints missing")
    assert Path(STEPS) != v2_steps
    want = json.loads(meta.read_text()).get("corpus_sha256", "")
    have = {corpus_fingerprint(p)
            for p in sorted(Path("data/policy").glob("steps.parquet*"))}
    assert want in have, ("no steps.parquet snapshot matches the shipped v2 "
                          "checkpoints' corpus_sha256 any more")


# --------------------------------------------------------------------------- #
# 10. the training machinery (no checkpoint, no real training run)
# --------------------------------------------------------------------------- #
def _toy_cache(frame: pd.DataFrame):
    """A stand-in :class:`PatternCache` over the frame's real patterns.

    The board CONTENT is irrelevant to the guard below — what is being tested is
    that three heads, three masks and a per-head weight matrix survive the
    collate, the loss and the evaluator — so the encoder is not run.
    """
    from lpopt.policy.data import PatternCache

    pats = sorted(set(frame["parent_pattern"]) | set(frame["child_pattern"]))
    index = {p: i for i, p in enumerate(pats)}
    rng = np.random.default_rng(0)
    return PatternCache(
        index=index,
        slots=rng.random((len(pats), 4, 69)).astype(np.float16),
        globals_=rng.random((len(pats), 3)).astype(np.float32),
        mirror=np.arange(len(pats), dtype=np.int32),
        channels=[f"c{i}" for i in range(4)],
        globals_names=[f"g{i}" for i in range(3)])


def test_the_three_head_training_step_runs_end_to_end(universe: pd.DataFrame,
                                                      fold: pd.Series) -> None:
    pytest.importorskip("torch")
    from lpopt.policy.train_v3 import evaluate_fold, gate_verdict, train_one
    from lpopt.policy.v3 import PolicyStepsV3

    sub = universe[fold.isin(("gate_cur", "prospective_cell")).to_numpy()
                   ].head(240).reset_index(drop=True)
    cache = _toy_cache(sub)
    scalars, names = scalar_features_v3(sub)
    weights = weights_v3(sub, pd.Series("train", index=sub.index))
    sets = {name: PolicyStepsV3(sub, cache, scalars, weights,
                                delta_channels=[0, 1], augment=(name == "train"),
                                seed=1)
            for name in ("train", "val")}
    assert sets["train"].labels.shape == (len(sub), 3)
    assert sets["train"].weights.shape == (len(sub), 3)
    assert sets["train"].n_cond == len(names) + 3

    model, meta = train_one(20260831, sets=sets, device="cpu", epochs=1,
                            batch_size=64, lr=1e-3, weight_decay=1e-4,
                            patience=1, width=8, n_blocks=1, num_workers=0)
    assert meta["net_config"]["n_heads"] == 3
    assert meta["heads"] == list(HEADS_V3)

    rng = np.random.default_rng(0)
    probs = rng.random((len(sub), 3))
    out = evaluate_fold("gate_cur", sub, probs, sub, None, gd_sign=-1.0,
                        rng=rng, n_boot=32)
    assert set(HEADS_V3) <= set(out)
    verdict = gate_verdict({"folds": {"gate_cur": out}})
    assert set(verdict["regret_beats"]) == set(
        ("random", "class_freq", "periph", "gd_rule", "policy_v2"))
    assert isinstance(verdict["PASS"], bool)


# --------------------------------------------------------------------------- #
# quarantine leakage into the corpus edge universe (track C5)
#
# `build_steps` filtered NOTHING: `lineage_edges` keys only on
# `parent_record_id.notna()` plus membership in the store, so a `valid=False`
# row (the 2026-08-30 HGD569 alias quarantine — a board whose deck named a raw
# type id and therefore evaluated a core that was never loaded) became a policy
# training edge, as parent AND as child.  Masking downstream on `both_converged`
# does not help: that column is about CONVERGENCE, and a quarantined row can be
# `converged=True`.  The drop has to happen BEFORE the edge set is built.
# --------------------------------------------------------------------------- #
_C5_PAIR = "A1_A2"


def _c5_board(seed: int) -> str:
    import random as _random

    from lpopt.data.schema import pack_pattern as _pack
    from lpopt.search.genome import random_genome as _rg
    return _pack(_rg(_random.Random(seed), _C5_PAIR, 30).to_pattern())


def _c5_row(rid: str, parent: str | None, packed: str, **over) -> dict:
    row = {
        "record_id": rid, "parent_record_id": parent, "pattern": packed,
        "case_pair": _C5_PAIR, "feed": 121, "library_id": "testlib",
        "campaign": "unit", "dataset": "P", "generator": "unit",
        "converged": True, "valid": True, "f_r": 1.5, "f_q": 2.0,
        "cbc_max": 1200.0, "ao_abs": 0.05, "cyclen": 500.0,
        "node_peak": 1.3, "map_cov": 0.2,
    }
    row.update(over)
    return row


def test_build_steps_drops_quarantined_rows_from_the_edge_universe() -> None:
    p, c, q = (_c5_board(31), _c5_board(32), _c5_board(33))
    store = pd.DataFrame([
        _c5_row("p0", None, p),
        _c5_row("c0", "p0", c),
        # a quarantined CHILD of the good parent, and a good child of the
        # quarantined row — both edges must vanish, in both directions.
        _c5_row("q0", "p0", q, valid=False, failure="alias_noop_P6_20260830"),
        _c5_row("c1", "q0", _c5_board(34)),
    ])
    steps = M.build_steps(store, {}, {})
    pairs = set(zip(steps["parent_record_id"], steps["child_record_id"]))
    assert pairs == {("p0", "c0")}, (
        "a quarantined row still trains the move-proposal policy: it is a "
        "converged=True row, so both_converged masks nothing")
    assert "q0" not in set(steps["parent_record_id"]) | set(steps["child_record_id"])


def test_build_steps_keeps_nonconverged_edges_and_marks_them() -> None:
    """Deliberate asymmetry: a FAILED child is signal; a FAKE child is not.

    `both_converged` exists to carry the failed half of the move-proposal
    corpus, so the quarantine filter must key on `valid` alone and leave the
    non-converged edges exactly where they were.
    """
    p, c = _c5_board(41), _c5_board(42)
    store = pd.DataFrame([
        _c5_row("p0", None, p),
        _c5_row("c0", "p0", c, converged=False, f_r=None, cyclen=None),
    ])
    steps = M.build_steps(store, {}, {})
    assert len(steps) == 1
    assert bool(steps["both_converged"].fillna(False).iloc[0]) is False


def test_build_steps_is_unchanged_when_nothing_is_quarantined() -> None:
    """Byte-identical contract: `valid` all-True, or absent, changes nothing."""
    rows = [_c5_row("p0", None, _c5_board(51)),
            _c5_row("c0", "p0", _c5_board(52))]
    with_col = M.build_steps(pd.DataFrame(rows), {}, {})
    without = M.build_steps(pd.DataFrame(rows).drop(columns=["valid"]), {}, {})
    assert len(with_col) == len(without) == 1
    shared = [c for c in with_col.columns if c in without.columns]
    pd.testing.assert_frame_equal(with_col[shared], without[shared])


def test_build_elites_drops_quarantined_rows_from_the_good_states() -> None:
    """The sibling artefact of steps.parquet, and the worse leak of the two.

    `feasibility()` gates on `converged` ALONE, and the rank is F_r-ASCENDING,
    so a quarantined row with the best F_r is not merely admitted to
    elites.parquet — it is rank 1 of its cell, i.e. the first board the
    constructor policy is told to imitate.
    """
    store = pd.DataFrame([
        _c5_row("g0", None, _c5_board(61), f_r=1.50),
        _c5_row("q0", None, _c5_board(62), f_r=1.20,
                valid=False, failure="alias_noop_P6_20260830"),
    ])
    elites = M.build_elites(store, {}, k=8)
    assert set(elites["record_id"]) == {"g0"}, (
        "elites.parquet is the 'good states' imitation target and a read-only "
        "input of the sha-pinned ablation wave; the quarantined board's best "
        "F_r ranks it first in its cell")


def test_build_elites_is_unchanged_when_nothing_is_quarantined() -> None:
    """Byte-identical contract: `valid` all-True, or absent, changes nothing."""
    frame = pd.DataFrame([_c5_row("g0", None, _c5_board(63), f_r=1.50),
                          _c5_row("g1", None, _c5_board(64), f_r=1.40)])
    with_col = M.build_elites(frame, {}, k=8).reset_index(drop=True)
    without = M.build_elites(frame.drop(columns=["valid"]), {},
                             k=8).reset_index(drop=True)
    pd.testing.assert_frame_equal(with_col, without)
