"""Corpus, target, splits and features for the **v3** move-proposal policy.

Pre-registered in ``data/reports/policy_v3_prereg_20260831.md``, written before
any v3 weight existed.  Three measurements make v2 obsolete and each one is a
concrete difference in this file:

1. **The objective changed.**  The campaigns run ``min_fxy`` (F_xy first, a
   cyclen band, an F_r constraint) and v2's two heads are ``d_f_r`` /
   ``d_node_peak``.  v3 adds a third head, :data:`HEADS_V3` ``fxy``, and that
   head carries the gate.
2. **"F_xy is a monotone transform of F_r" was REJECTED.**  The r1 wave measured
   the F_xy:F_r transfer coefficient at 1.22-1.42 on the enrichment-radius
   families and 0.55-0.73 on the Gd/lattice families — a 1.9-2.6x split, i.e.
   using an F_r ranker as an F_xy ranker is a registered error.  So the shipped
   v2 ensemble is a BASELINE here, not an initialisation.
3. **A feature deficit was established.**  Interventions with
   ``d_fresh_enr_r_center == 0`` move F_xy by +0.0712 (20/20, p=1.9e-6).  The
   corpus had no Gd/lattice coordinate; :func:`scalar_features_v3` adds twelve.

Nothing in ``lpopt/policy/{data,net,train,train_v2,v2}.py`` changes behaviour:
v1 and v2 are registered baselines for this round, their checkpoints must keep
loading, and ``tests/test_policy_prior.py`` asserts serving still reproduces
their training probabilities.  v3 is additive.

Provenance
----------
:func:`provenance_v3` is :func:`~..model.featurize.serve_provenance`, NOT
``data.corpus_provenance``.  That is the whole of prereg §8-A: the v2 corpus was
built with ``sym_class`` from ``library_provenance``, so every ga80 board trained
at ``g_sym_class`` 0.0 while the store says ``"rot61"`` — train and serve were
consistently wrong and the shipped v2 checkpoint has to keep being fed the wrong
value.  v3 is mined fresh, so it takes the truth: ``serve_provenance`` is what a
SERVED pattern will be featurized with, and train/serve identity is the
acceptance criterion.

Note what "the store row's ``sym_class``" resolves to, because the prereg phrases
the fix that way: the store carries ``"rot61"`` on all 35,289 campaign rows and
all 38,854 Dataset-A rows, and ``"free69"`` on exactly 574 (0.77%) — the
historical ``extract_b`` ga80 harvest, 65 of which reach this corpus.  A SERVED
pattern has no store row and will be written ``"rot61"`` unconditionally, so
reading the row would re-open the train/serve gap for those 65 rows instead of
closing it.  ``serve_provenance`` is the store truth for everything the policy
can ever be asked to score, and ``tests/test_policy_v3.py`` re-derives that from
the live store rather than trusting this paragraph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data import FORBIDDEN_COLUMNS, PatternCache, PolicySteps, _components
from .v2 import (
    CURRENT_ERA_LIBRARIES, EVAL_LABEL, INTERVENTIONAL_SOURCES, TARGET_CLIP,
    era_weights, scalar_features_v2, targets,
)

#: The v3 corpus.  A SEPARATE file from ``data/policy/steps.parquet`` on
#: purpose: the shipped ``data/models/policy_v2`` members stamp the v2 corpus's
#: sha256 and the serving-parity tests re-read it, so the v2 corpus keeps its
#: bytes.  Built by ``python policy_v2_corpus.py backfill-v3 --apply``.
STEPS_V3 = "data/policy/steps_v3.parquet"

# --------------------------------------------------------------------------- #
# serving contract
# --------------------------------------------------------------------------- #
#: Stamped into every v3 member's ``meta.json`` and REQUIRED by
#: :class:`lpopt.policy.scorer.MoveScorerV3` at load.  It names the serving
#: contract — v2's 39 scalars plus :data:`NEW_SCALARS_V3`, over the ``v6b`` board
#: encoding, read as a within-parent ranker with THREE logits — and nothing else.
POLICY_SCHEMA_V3 = "policy_move_v3"

#: Three heads.  ``fr`` and ``flat`` inherit v2's definitions VERBATIM (prereg
#: §2a) so the trunk keeps its 21k / 18k rows of supervision; ``fxy`` is new and
#: is the only head the gate reads.
HEADS_V3: tuple[str, ...] = ("fr", "flat", "fxy")

#: v2's interventional tags plus the five r1 intervention waves.  These are the
#: rows produced by fixing a parent and enumerating moves off it, so they are
#: the only current-era rows with enough candidates per parent to answer the
#: within-parent question.  48.9% of current-era same-cell rows are now
#: interventional (281/1,006 in the v2 round).
INTERVENTIONAL_SOURCES_V3: tuple[str, ...] = (
    *INTERVENTIONAL_SOURCES,
    "intervention_T6T4_f121", "intervention_E1E2_f109", "intervention_E1E2_f121",
    "intervention_N1N2_f113", "intervention_HGD569_f125",
)

# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
#: Clip per head.  ``fr`` / ``flat`` are v2's registered values.
#:
#: ``fxy`` = 0.060 is a DECLARED amendment of v2's rule and the reason is
#: written here rather than discovered later: v2's rule ("the smallest 5e-3
#: rounded value that saturates no current-era interventional improvement")
#: gives 0.145, and that value is made by a SINGLE cell — the 125 interventional
#: improvements are p50 0.0083 / p90 0.0280 / p99 0.0857 and the four above 0.06
#: are all ``HGD569_f125`` (0.0605, 0.0669, 0.0917, 0.1432).  At 0.145, 96.8% of
#: improvements compress below y = 0.2 and the clip loses the magnitude
#: resolution it exists for.  Registered rule: the smallest 5e-3 rounded value
#: that saturates no improvement in FOUR of the five interventional cells.
TARGET_CLIP_V3: dict[str, float] = {**TARGET_CLIP, "fxy": 0.060}

#: Feasibility gate on the ``fxy`` target (prereg §2b).  ``d_cyclen`` must not
#: fall more than this; the threshold is the 5% quantile (-5.154) of ``d_cyclen``
#: over the current-era F_xy rows, rounded to 5e-1.  ``in_cyclen_band_child`` is
#: known on only 63 of 1,309 such rows (4.8%), so band membership itself cannot
#: be used.
CYCLEN_TOL = 5.0
#: The program's F_r limit.  A HARD gate (``child_f_r <= 1.55``) keeps only 374
#: of 1,307 rows because 71% of parents are already above it, so the registered
#: rule is "do not cross the limit, or if the parent already crossed it, do not
#: get worse": ``child_f_r <= max(parent_f_r, F_R_LIMIT)``.
F_R_LIMIT = 1.55

#: FOM whose decrease is the improvement, per head.
TARGET_DELTA_V3: dict[str, str] = {
    "fr": "d_f_r", "flat": "d_node_peak", "fxy": "d_f_xy"}
#: Binary label kept for EVALUATION only — v3 is scored on ``improved_fxy``,
#: never on its own training target, the same discipline that scored v2 on v1's
#: label.
EVAL_LABEL_V3: dict[str, str] = {**EVAL_LABEL, "fxy": "improved_fxy"}


def fxy_feasible(steps: pd.DataFrame) -> np.ndarray:
    """The two registered constraint terms of the ``fxy`` target, as one mask.

    ``d_cyclen >= -CYCLEN_TOL`` AND ``child_f_r <= max(parent_f_r, F_R_LIMIT)``.
    A missing constraint reading is NOT feasible: an unharvested cyclen or F_r
    means the constraint was not shown to hold, and the target this feeds is an
    EXPECTED improvement, so an unverified row must contribute zero rather than
    an optimistic gain.
    """
    d_cyclen = pd.to_numeric(steps["d_cyclen"], errors="coerce").to_numpy(float)
    child_fr = pd.to_numeric(steps["child_f_r"], errors="coerce").to_numpy(float)
    parent_fr = pd.to_numeric(steps["parent_f_r"], errors="coerce").to_numpy(float)
    both = steps["both_converged"].fillna(False).to_numpy(bool)
    headroom = np.fmax(parent_fr, F_R_LIMIT)
    # A NaN comparison is False in numpy, which is exactly the rule above: an
    # unharvested cyclen or F_r reading is not a satisfied constraint.
    with np.errstate(invalid="ignore"):
        ok = (d_cyclen >= -CYCLEN_TOL) & (child_fr <= headroom)
    return both & ok


def targets_v3(steps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(y[N, 3] in [0, 1], mask[N, 3])`` — prereg §2a.

    ``fr`` / ``flat`` are :func:`lpopt.policy.v2.targets` verbatim.  ``fxy`` is::

        gain_fxy = max(0, -d_f_xy)
        feasible = both_converged AND d_cyclen >= -5.0 AND child_f_r <= max(parent_f_r, 1.55)
        y_fxy    = min(gain_fxy * feasible, 0.060) / 0.060

    and its mask is ``d_f_xy.notna() & both_converged``.  Measured on the 1,309
    current-era F_xy rows: 230 raw improvements become 184 positives at mean
    0.00274 (the ungated numbers are 230 / 0.00335; the rejected alternative
    ``d_f_r <= 0`` would leave 136 / 0.00245).
    """
    n = len(steps)
    y = np.zeros((n, len(HEADS_V3)), np.float32)
    mask = np.zeros((n, len(HEADS_V3)), np.float32)
    y[:, :2], mask[:, :2] = targets(steps)

    both = steps["both_converged"].fillna(False).to_numpy(bool)
    delta = pd.to_numeric(steps[TARGET_DELTA_V3["fxy"]],
                          errors="coerce").to_numpy(float)
    gain = np.clip(-np.nan_to_num(delta, nan=0.0), 0.0, None)
    gain = np.where(fxy_feasible(steps), gain, 0.0)
    clip = TARGET_CLIP_V3["fxy"]
    y[:, 2] = (np.clip(gain, 0.0, clip) / clip).astype(np.float32)
    mask[:, 2] = (both & np.isfinite(delta)).astype(np.float32)
    return y, mask


def eval_labels_v3(steps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(y[N, 3] binary, mask[N, 3])`` — the GATE label, for evaluation only."""
    n = len(steps)
    y = np.zeros((n, len(HEADS_V3)), np.float32)
    mask = np.zeros((n, len(HEADS_V3)), np.float32)
    for h, head in enumerate(HEADS_V3):
        raw = steps[EVAL_LABEL_V3[head]]
        mask[:, h] = raw.notna().to_numpy(np.float32)
        y[:, h] = raw.fillna(False).astype(bool).to_numpy(np.float32)
    return y, mask


# --------------------------------------------------------------------------- #
# universe and splits
# --------------------------------------------------------------------------- #
#: The held-out cell of prereg §3d.  Registered BEFORE any weight existed and
#: for reasons that are all structural: its 160 interventional rows are complete
#: and direction-balanced (19 parents with >= 8 candidates, 13 mixed-label), it
#: is the one cell where the wave's Gd hypothesis did NOT hold (F_xy p=0.503) so
#: an over-fitted Gd feature shows up here, and it is a ``twice_plus`` supplier
#: so the burn-state stratum is exercised.  ``E1_E2/f121/ga80`` was rejected
#: because the prospective A/B/C runs in it; ``HGD569_f125`` because its labels
#: died and were revived by the r1 alias defect.
PROSPECTIVE_CELL = "N1_N2/f113/ga80"

FOLDS_V3: tuple[str, ...] = ("train", "val", "gate_cur", "prospective_cell")


def load_universe_v3(path: str | Path = STEPS_V3) -> pd.DataFrame:
    """Same-cell rows carrying at least one of the THREE labels.

    v2's universe was "``improved_fr`` or ``improved_flat`` is known"; v3 admits
    a row that only has ``d_f_xy`` as well, because that row supervises the head
    the gate reads.  Realized 21,134 rows on the 2026-08-31 corpus.
    """
    steps = pd.read_parquet(path)
    same_cell = steps[~steps["cross_cell"].astype(bool)].copy()
    has_label = (same_cell["improved_fr"].notna()
                 | same_cell["improved_flat"].notna()
                 | same_cell["d_f_xy"].notna())
    out = same_cell[has_label].reset_index(drop=True)
    out["move_class"] = out["move_class"].astype(str)
    out["era_current"] = out["library_id"].isin(CURRENT_ERA_LIBRARIES).astype(bool)
    out["interventional"] = out["lineage_source"].isin(
        INTERVENTIONAL_SOURCES_V3).astype(bool)
    return out


def build_splits_v3(steps: pd.DataFrame, *, seed: int = 20260831,
                    val_frac: float = 0.10,
                    holdout_cell: str | None = PROSPECTIVE_CELL) -> pd.Series:
    """Assign every row to one of :data:`FOLDS_V3`.  Deterministic in ``seed``.

    Four folds, not v2's three:

    * ``prospective_cell`` — the WHOLE of ``holdout_cell``, removed before any
      other rule runs.  It enters neither training nor the gate and is opened
      ONCE, after the gate numbers exist (prereg §6).
    * ``gate_cur`` — the remaining current-era lineage connected components,
      ranked by their ``d_f_xy`` LABEL COUNT (descending, ties by component key)
      and assigned alternately.  No RNG and no outcome is read.
    * ``val`` — 10% of the remaining components, drawn independently inside each
      era, used for early stopping and nothing else.
    * ``train`` — everything else, legacy included.

    **The alternation is reversed relative to v2, and that is declared here.**
    v2 gave rank 0 to the gate and split the current era 1,006 gate / 815 train.
    F_xy labels are 40x scarcer than ``improved_fr`` (1,309 vs 21,132); keeping
    v2's direction leaves the ``fxy`` head 488 training rows with 61 positives,
    reversing it gives 532 / 79 against a gate of 540 and still leaves 38 gate
    parents with >= 8 candidates.  The choice was made on LABEL COUNTS, which are
    known before any outcome is modelled, not on a result.
    """
    fold = pd.Series("train", index=steps.index, dtype=object)
    if holdout_cell:
        fold[steps["cell"] == holdout_cell] = "prospective_cell"

    era = steps["era_current"].to_numpy(bool)
    cur = steps[(fold == "train").to_numpy() & era]
    if len(cur):
        comp = pd.Series(_components(cur), index=cur.index)
        labelled = cur["d_f_xy"].notna()
        size = comp[labelled].value_counts().reindex(comp.unique(), fill_value=0)
        order = sorted(size.index, key=lambda k: (-int(size[k]), str(k)))
        gate_keys = set(order[1::2])            # rank 0 -> train (reversed)
        fold[cur.index[comp.isin(gate_keys).to_numpy()]] = "gate_cur"

    rng = np.random.default_rng(seed)
    pool = (fold == "train").to_numpy()
    for flag in (False, True):                  # legacy first, then current
        idx = steps.index[pool & (era == flag)]
        if not len(idx):
            continue
        groups = _components(steps.loc[idx])
        keys, counts = np.unique(groups, return_counts=True)
        pick = rng.permutation(len(keys))
        target, total, taken = val_frac * len(idx), 0, set()
        for j in pick:
            if total + counts[j] <= target:
                taken.add(keys[j])
                total += int(counts[j])
        if taken:
            fold.loc[idx[np.isin(groups, list(taken))]] = "val"
    return fold


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def parent_weights(steps: pd.DataFrame, fold: pd.Series) -> np.ndarray:
    """``mean_candidates_per_parent / n_candidates(parent)`` — EVERY PARENT EQUAL.

    Only the ``fxy`` head is weighted this way (prereg §3c).  The object being
    trained is a within-parent ranker and the consumer picks k moves off ONE
    parent, so a parent with 8 interventional candidates must not out-vote eight
    observational parents with one candidate each.  There is no knob: the
    statement is "every parent carries the same weight" and the constant is the
    realized mean on the training fold, **530/147 = 3.605**.  (The prereg's 3.62
    is 532/147: it counted the rows carrying a ``d_f_xy`` READING, and two of
    them did not converge, so the head's own mask drops them.  The masked count
    is the one the loss actually sees and is the one used here.)

    Candidates are counted over the F_xy-masked rows of the whole universe; a
    parent never straddles a fold, because every fold boundary is a lineage
    connected component and a parent's candidates share its component.
    """
    mask = targets_v3(steps)[1][:, 2] > 0
    parents = steps["parent_record_id"].astype(str).to_numpy()
    counts = pd.Series(parents[mask]).value_counts()
    n = pd.Series(parents).map(counts).to_numpy(float)

    train = (fold == "train").to_numpy() & mask
    scale = (train.sum() / max(len(set(parents[train])), 1)) if train.any() else 1.0
    out = np.where(mask & np.isfinite(n) & (n > 0), scale / np.where(n > 0, n, 1.0),
                   1.0)
    return out.astype(np.float32)


def weights_v3(steps: pd.DataFrame, fold: pd.Series, *,
               cap: float = 20.0) -> np.ndarray:
    """``w[N, 3] = w_era x w_parent`` — per HEAD, because ``w_parent`` is fxy-only.

    ``w_era`` is v2's, unchanged: legacy 1.0, current era
    ``n_train_legacy / n_train_current`` capped at 20, so the two eras carry
    exactly half the loss mass each.  Realized 16,579/1,269 = 13.06 — the cap
    does not bind this round (it did in v2, at 20.3).
    """
    w_era = era_weights(steps, fold, cap=cap)
    w = np.repeat(w_era[:, None], len(HEADS_V3), axis=1)
    w[:, 2] *= parent_weights(steps, fold)
    return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
#: Scales for the twelve additions, all measured on the current-era same-cell
#: rows of the 2026-08-31 corpus and rounded, in the spirit of v2's
#: ``FLIP_MASS_UNIT`` / ``PARENT_MASS_CENTER``.  They exist so every input to
#: FiLM is O(1); none of them is fitted and none is tuned.
GD_MASS_UNIT = 100.0            # d_fresh_gd_mass p05/p95 = -64 / +80
GD_MASS_CENTER, GD_MASS_SCALE = 2600.0, 250.0     # parent median 2600, sd 219
GDWT_MASS_UNIT = 1000.0         # d_fresh_gdwt_mass sd 485, max|d| 5120
KINF0_MASS_CENTER, KINF0_MASS_SCALE = 137.0, 10.0  # parent median 137.1, sd 9.6
#: ``|n_gd(E1) - n_gd(E2)| = |20 - 24| = 4``: one live interventional swap, so
#: ``fresh_gd_contrast`` reads "how many E1E2-swap-equivalents of Gd contrast".
GD_CONTRAST_UNIT = 4.0

#: The twelve columns v3 adds to v2's 39.  Named so the results report and the
#: checkpoint metadata agree on exactly what changed.
NEW_SCALARS_V3: tuple[str, ...] = (
    "d_fresh_gd_mass", "parent_fresh_gd_mass", "d_fresh_gd_r_center",
    "d_fresh_gd_share_periph", "d_fresh_gdwt_mass", "gdwt_present",
    "d_fresh_kinf0_mass", "parent_fresh_kinf0_mass", "d_fresh_kinf0_r_center",
    "n_fresh_type_changed", "fresh_type_multiset_changed", "fresh_gd_contrast",
)

#: The length of the v3 serving contract.  Stated as a constant rather than left
#: implicit because the v3.1 stamp is only allowed onto a checkpoint whose vector
#: is :data:`N_SCALARS_V31` long, and the two numbers have to be comparable in
#: one place (``train_v3.featurize_round``).
N_SCALARS_V3 = 51

#: Columns the v3 feature frame needs and v2's does not.
REQUIRED_V3_COLUMNS: tuple[str, ...] = (
    "parent_fresh_gd_mass", "child_fresh_gd_mass", "d_fresh_gd_mass",
    "d_fresh_gd_r_center", "d_fresh_gd_share_periph",
    "parent_fresh_gdwt_mass", "child_fresh_gdwt_mass", "d_fresh_gdwt_mass",
    "parent_fresh_kinf0_mass", "d_fresh_kinf0_mass", "d_fresh_kinf0_r_center",
    "n_fresh_type_changed", "fresh_type_multiset_changed", "fresh_gd_contrast",
)


def scalar_features_v3(steps: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """v2's 39 move/context scalars plus the twelve v3 additions -> 51.

    ``scalar_features_v2`` is CALLED, not copied — as it in turn calls v1's
    ``scalar_features`` — so v1's 36 and v2's 39 stay bit-identical inside v3's
    51 and the three models differ by the additions, the target and the folds,
    by nothing else that could explain a gate difference.

    ``gdwt_present`` is the ``swap_span`` treatment from v1 (prereg §4c): every
    ga80 fuel type is missing ``gd_wt``, so the mass is zeroed and the flag keeps
    "absent" distinguishable from "zero".  That missingness is also the
    registered first suspect if the transfer bar fails, because the held-out
    cell is ga80 (prereg §7).

    ``gd_table_complete`` is NOT here.  It is a diagnostic column, and a model
    that could read "this row's fuel table is complete" would be reading a
    library identity by the back door.
    """
    missing = [c for c in REQUIRED_V3_COLUMNS if c not in steps.columns]
    if missing:
        raise KeyError(
            f"the v3 Gd columns are absent from this frame: {missing}.  Build "
            f"the v3 corpus with `python policy_v2_corpus.py backfill-v3 "
            f"--apply` and load it from {STEPS_V3!r}")

    base, names = scalar_features_v2(steps)

    def num(col: str, fill: float = 0.0) -> np.ndarray:
        return np.nan_to_num(
            pd.to_numeric(steps[col], errors="coerce").to_numpy(float), nan=fill)

    gdwt_present = (
        np.isfinite(pd.to_numeric(steps["parent_fresh_gdwt_mass"],
                                  errors="coerce").to_numpy(float))
        & np.isfinite(pd.to_numeric(steps["child_fresh_gdwt_mass"],
                                    errors="coerce").to_numpy(float)))

    extra = {
        "d_fresh_gd_mass": num("d_fresh_gd_mass") / GD_MASS_UNIT,
        "parent_fresh_gd_mass": (
            num("parent_fresh_gd_mass", GD_MASS_CENTER) - GD_MASS_CENTER
        ) / GD_MASS_SCALE,
        "d_fresh_gd_r_center": num("d_fresh_gd_r_center"),
        "d_fresh_gd_share_periph": num("d_fresh_gd_share_periph"),
        "d_fresh_gdwt_mass": np.where(
            gdwt_present, num("d_fresh_gdwt_mass"), 0.0) / GDWT_MASS_UNIT,
        "gdwt_present": gdwt_present.astype(float),
        "d_fresh_kinf0_mass": num("d_fresh_kinf0_mass"),
        "parent_fresh_kinf0_mass": (
            num("parent_fresh_kinf0_mass", KINF0_MASS_CENTER) - KINF0_MASS_CENTER
        ) / KINF0_MASS_SCALE,
        "d_fresh_kinf0_r_center": num("d_fresh_kinf0_r_center"),
        "n_fresh_type_changed": num("n_fresh_type_changed") / 10.0,
        "fresh_type_multiset_changed": steps[
            "fresh_type_multiset_changed"].fillna(False).astype(bool).to_numpy(float),
        "fresh_gd_contrast": num("fresh_gd_contrast") / GD_CONTRAST_UNIT,
    }
    bad = FORBIDDEN_COLUMNS & set(extra)
    if bad:                                   # structural, not a style check
        raise AssertionError(f"outcome/provenance leaked into features: {sorted(bad)}")

    order = sorted([*names, *extra])
    lookup = {n: base[:, i] for i, n in enumerate(names)}
    lookup.update({k: v.astype(np.float32) for k, v in extra.items()})
    return np.stack([lookup[n] for n in order], axis=1).astype(np.float32), order


# --------------------------------------------------------------------------- #
# provenance / pattern cache
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# v3.1 — the pure-code deltas of policy_v31_prereg_20260831_DRAFT.md
#
# Everything below is ADDITIVE and is reached only through a ``*_v31`` name or
# a flag that defaults off.  The v3 objects above keep their bytes because the
# shipped ``data/models/policy_v3`` members stamp them and the v3 results
# document is judged on them, the same relationship v3 has to v2.
# --------------------------------------------------------------------------- #
#: The v3.1 corpus (111 columns).  ``mine_policy_corpus.py --v31 --apply``.
STEPS_V31 = "data/policy/steps_v31.parquet"

#: The v3.1 serving contract: v3's 51 scalars plus :data:`NEW_SCALARS_V31`, over
#: the same ``v6b`` board encoding, read as a within-parent ranker with three
#: logits of which ``fxy`` is served through a monotone Platt map (§5c).  A
#: checkpoint stamped ``POLICY_SCHEMA_V3`` is a DIFFERENT contract and must not
#: load here, and vice versa.
POLICY_SCHEMA_V31 = "policy_move_v31"

#: The two burnt sub-lattice columns of prereg v3.1 §4c, and the ONLY two.  The
#: 65-column non-conserving Gd/lattice family of §4a was measured and REJECTED
#: (leave-one-cell-out ridge: +RAD null, +ABS null, +DSP harmful at
#: [-0.1382, -0.0259] pb-AUC, +NC-all harmful on regret), and ``rew_cor_r2`` /
#: ``rew_flux_*`` fail the STRUCTURAL half of the §4c selection rule -- they are
#: conserving moments, expressible as ``sum mult*g(r)*X`` -- so they are excluded
#: regardless of their readouts.  Selecting on structure rather than on a p-value
#: is what stops this round repeating v3's H3/Gd error.
NEW_SCALARS_V31: tuple[str, ...] = ("burnt_absmov", "burnt_absmov_r")

#: Corpus columns the v3.1 feature frame needs and v3's does not.  The two
#: diagnostics (``burnt_slots_moved``, ``burnt_token_complete``) are deliberately
#: NOT here: they are read by the census and by tests, never by the model, the
#: same standing ``gd_table_complete`` has in v3.
REQUIRED_V31_COLUMNS: tuple[str, ...] = NEW_SCALARS_V31

#: The length of the v3.1 serving contract.  A checkpoint may carry
#: :data:`POLICY_SCHEMA_V31` only if its ``scalar_names`` is this long and
#: contains both of :data:`NEW_SCALARS_V31`; ``train_v3.featurize_round`` refuses
#: the run otherwise.  The reason is a measured near-miss rather than tidiness:
#: the first cut of this track stamped ``policy_move_v31`` onto a 51-name vector
#: because ``main`` still called :func:`scalar_features_v3`, and
#: ``MoveScorerV31`` would then have rendered 53 names against a 51-name
#: checkpoint and refused to serve the round's own ensemble.
N_SCALARS_V31 = N_SCALARS_V3 + len(NEW_SCALARS_V31)

#: Feature scales for the two additions.  REGISTERED CONSTANTS, not fitted, and
#: shared by every fold — so they cannot leak a fold's distribution the way a
#: per-fold normalizer would (§10-4).
#:
#: The registered rule is v3's own: the current-era same-cell p05/p95, rounded to
#: the next 1-2-5 step, exactly as ``GD_MASS_UNIT = 100`` came from tails
#: -64/+80 and ``GDWT_MASS_UNIT = 1000`` from sd 485.  Measured once, on
#: ``steps_v3.parquet`` (28,889 rows, 2,813 current-era same-cell), by
#: ``python mine_policy_corpus.py --v31`` — which prints these two lines so the
#: number can be re-derived rather than believed::
#:
#:     burnt_absmov     p05/p50/p95 = -2.2493 +0.0000 +3.2677   max|.| 9.801
#:     burnt_absmov_r   p05/p50/p95 = -1.6299 +0.0000 +2.5420   max|.| 6.965
#:
#: Both tails round to 5.0.  The value is FROZEN here: re-measuring on
#: ``steps_v31.parquet`` (the same rows plus ~100 r2 edges) is a check, not a
#: licence to move the constant, because moving it changes every feature.
BURNT_ABSMOV_UNIT = 5.0
BURNT_ABSMOV_R_UNIT = 5.0


def scalar_features_v31(steps: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """v3's 51 move/context scalars plus the two v3.1 additions -> 53.

    :func:`scalar_features_v3` is CALLED, not copied — as it in turn calls v2's
    and v1's — so v1's 36, v2's 39 and v3's 51 stay bit-identical inside v3.1's
    53 (``tests/test_policy_v31.py`` asserts the columns element for element),
    and the two models differ by the two additions, the folds and the stage-2
    branch, by nothing else that could explain a gate difference.

    The leakage guard is inherited rather than re-stated: ``FORBIDDEN_COLUMNS``
    already covers the outcome and provenance columns, both additions are
    functions of ``(parent_pattern, child_pattern)`` alone and so are known at
    proposal time, and the assertion below re-checks the additions by name.
    """
    missing = [c for c in REQUIRED_V31_COLUMNS if c not in steps.columns]
    if missing:
        raise KeyError(
            f"the v3.1 burnt columns are absent from this frame: {missing}.  "
            f"Build the v3.1 corpus with `python mine_policy_corpus.py --v31 "
            f"--apply` and load it from {STEPS_V31!r}")

    base, names = scalar_features_v3(steps)

    def num(col: str) -> np.ndarray:
        return np.nan_to_num(
            pd.to_numeric(steps[col], errors="coerce").to_numpy(float), nan=0.0)

    extra = {
        "burnt_absmov": num("burnt_absmov") / BURNT_ABSMOV_UNIT,
        "burnt_absmov_r": num("burnt_absmov_r") / BURNT_ABSMOV_R_UNIT,
    }
    bad = FORBIDDEN_COLUMNS & set(extra)
    if bad:                                   # structural, not a style check
        raise AssertionError(f"outcome/provenance leaked into features: {sorted(bad)}")
    if set(extra) != set(NEW_SCALARS_V31):
        raise AssertionError(
            f"v3.1 adds exactly {sorted(NEW_SCALARS_V31)} (§4c selection rule); "
            f"this build adds {sorted(extra)}")

    order = sorted([*names, *extra])
    lookup = {n: base[:, i] for i, n in enumerate(names)}
    lookup.update({k: v.astype(np.float32) for k, v in extra.items()})
    return np.stack([lookup[n] for n in order], axis=1).astype(np.float32), order


# --------------------------------------------------------------------------- #
# v3.1 splits — K-fold component-blocked cross-fit (§3a / §3d)
# --------------------------------------------------------------------------- #
#: The v3.1 held-out cell.  Chosen by the four REGISTERED difficulty conditions
#: of §3c — >= 20 parents with >= 8 F_xy candidates, live share >= 0.70,
#: parent-max gain median >= 0.10 clip units, permutation-null MDE <= 0.20 —
#: which read LABELS only and no scorer at all.  ``E1_E2/f109/ga80`` is the one
#: qualifying cell (20 / 0.80 / 0.239 / 0.160); v3's ``N1_N2/f113/ga80`` fails
#: three of the four, which is why its FAIL supported "this cell cannot be used
#: for transfer" and not "v3 does not transfer" (results §5-R2).
PROSPECTIVE_CELL_V31 = "E1_E2/f109/ga80"

#: The registered number of cross-fit folds.
XFIT_K = 5

FOLDS_V31: tuple[str, ...] = ("train", "val", "pool", "prospective_cell")


def load_universe_v31(path: str | Path = STEPS_V31) -> pd.DataFrame:
    """:func:`load_universe_v3` over the v3.1 corpus, with the columns checked."""
    out = load_universe_v3(path)
    missing = [c for c in REQUIRED_V31_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"{path} is not a v3.1 corpus: {missing} absent")
    return out


#: The v3.1 ``val`` fraction, and NOT v3's 0.10.  The two rounds draw val from
#: different bases — v3 drew it from what the gate had already declined, v3.1
#: draws it from the whole non-holdout frame BEFORE the blocks — so carrying
#: v3's number over spends the gate pool twice as hard as v3 did.  A DECLARED
#: DEVIATION from §3d's "pool 잔여 성분의 10%", which must be recorded as such in
#: the results document's §1 before any v3.1 weight exists; it is written down
#: here because it moves the judgement set, and the measurement it rests on is
#: printed rather than asserted.  Registered holdout ``E1_E2/f109/ga80``, k = 5,
#: base seed 20260903, as ``(pool rows, pool F_xy, pool parents >= 8, pool
#: live)`` and the val fold's F_xy row count.  BOTH corpora are given because
#: they differ and the run reads the SECOND one::
#:
#:     steps_v3.parquet   (28,889 rows -- NOT what the round reads)
#:       val_frac 0.100 -> pool (2298, 1005, 67, 35)   val 142 F_xy
#:       val_frac 0.050 -> pool (2426, 1080, 75, 40)   val  67 F_xy
#:     steps_v31.parquet  (28,970 rows -- the corpus the round reads)
#:       val_frac 0.100 -> pool (2371, 1073, 66, 35)   val 155 F_xy
#:       val_frac 0.050 -> pool (2503, 1155, 72, 39)   val  73 F_xy  <- registered
#:
#: §3a's "79 / 41" is the pool BEFORE val is carved and §3d registers "val ~71
#: F_xy rows"; on the corpus the round reads, 0.05 meets the second (73) and
#: buys 72 / 39, while the unexamined 0.10 default would have put the gate pool
#: at 35 live and the val fold at 155 rows.  The fraction is over ROWS, not over
#: components — §3d's wording says components — and that is the second half of
#: the declared deviation.  ``emit_crossfit_splits`` guards the realized live
#: count so a later knob cannot spend it again silently.
VAL_FRAC_V31 = 0.05

#: The registered floor on the cross-fit pool's live parent count with the
#: registered holdout, §6a clause 2B's power is computed on it — and it is
#: derived on ``steps_v31.parquet``, the corpus the round actually reads, NOT on
#: ``steps_v3.parquet``.  The two differ: the r2 campaign's 81 edges enlarge the
#: current era, the row-proportional val draw therefore takes one more live
#: parent (val 4 -> 7 parents >= 8, 1 -> 2 live) and the pool realizes 39 rather
#: than the 40 measured on the v3 frame.  Clause 2B's power must be restated on
#: 39, not on the prereg's headline 41.
#:
#: The floor deliberately carries NO margin: it is a change detector, not a
#: slack budget.  Every knob that draws rows out of the pool spends the round's
#: only judgement set, so the intended behaviour is that ANY movement below the
#: measured value refuses the assignment and forces a declared deviation, rather
#: than being absorbed silently by a floor set comfortably low.
MIN_POOL_LIVE_V31 = 39


def build_splits_v31(steps: pd.DataFrame, *, seed: int = 20260903,
                     k: int = XFIT_K, val_frac: float = VAL_FRAC_V31,
                     holdout_cell: str | None = PROSPECTIVE_CELL_V31,
                     ) -> pd.DataFrame:
    """``DataFrame[fold, xfit_fold]`` — the §3a cross-fit, deterministic in ``seed``.

    v3 spent its current era on ONE alternating gate/train split and realized 38
    >= 8-candidate parents of which 23 were live.  Scoring the whole current era
    out-of-fold instead buys 79/41 BEFORE ``val`` is carved and a REALIZED
    72 / 39 after it (:data:`VAL_FRAC_V31` carries the measured table, on both
    corpora), at a cost of zero MASTER calls, and that is
    what makes the §3c cell replacement affordable at all: a single split with
    ``E1_E2/f109`` held out leaves 16 live parents, which decides nothing.  The
    two are registered as ONE change for that reason.

    Four labels, and an integer block for the pool:

    * ``prospective_cell`` — the WHOLE of ``holdout_cell``, removed before any
      other rule runs, in NO cross-fit fold (``xfit_fold = -1``), opened once
      after the gate numbers exist.
    * ``val`` — :data:`VAL_FRAC_V31` of the remaining components, drawn
      independently inside each era (see that constant for the measured table
      that fixes the fraction, and for why it is not v3's 0.10).
      ``xfit_fold = -1``: v3.1 selects lambda on ``val`` (§2c), so a ``val``
      parent inside the gate pool would let selection and judgement read the same
      rows.  That exclusion is a NEW discipline relative to v3 and is registered.
    * ``pool`` — the rest of the current era, its lineage connected components
      dealt into ``k`` blocks.  Each row's ``xfit_fold`` is the block that scores
      it OUT OF FOLD; it trains in the other ``k - 1``.
    * ``train`` — everything else, legacy included, ``xfit_fold = -1``, in EVERY
      fold's training set.

    The blocking is on lineage COMPONENTS, not parents, so v3's leakage rule is
    inherited unchanged: no board reachable from a training board is scored.
    Components are computed ONCE on the whole frame and reused by the val draw
    and by the block deal alike, so a chain that crosses the era boundary is one
    component everywhere rather than two inside the per-era subsets.
    Components are ranked by their ``d_f_xy`` LABEL COUNT (descending, ties by
    component key) and dealt ``rank % k`` — the exact generalization of v3's
    ``rank % 2`` alternation, with no RNG and no outcome read, so the K-fold
    allocation is decided by counts that are known before anything is modelled.
    """
    if k < 2:
        raise ValueError(f"cross-fit needs at least 2 folds, got k={k}")
    fold = pd.Series("train", index=steps.index, dtype=object)
    xfit = pd.Series(-1, index=steps.index, dtype=int)
    if holdout_cell:
        fold[steps["cell"] == holdout_cell] = "prospective_cell"

    # ``val`` FIRST, unlike v3.  v3 drew val out of what the gate had already
    # declined; here the pool is the judgement set, so val has to be carved out
    # before the blocks are dealt or the exclusion above is not enforceable.
    era = steps["era_current"].to_numpy(bool)
    rng = np.random.default_rng(seed)
    # Components are taken ONCE, over the WHOLE frame, and every rule below
    # reads that one labelling.  Deriving them per era subset instead is a
    # leakage hole with a name: a lineage chain that crosses the era boundary
    # (a legacy parent with a current-era child) is TWO components inside the
    # subsets and ONE component on the frame, so the legacy draw could take its
    # legacy half into ``val`` while its current-era half stayed in the gate
    # pool -- the exact straddle §3d exists to forbid, invisible to any check
    # that re-derives components the same way the split did.  Whole components
    # leave the pool, both eras of them.
    comp_all = pd.Series(_components(steps), index=steps.index)
    for flag in (False, True):                  # legacy first, then current
        avail = (fold == "train").to_numpy()
        idx = steps.index[avail & (era == flag)]
        if not len(idx):
            continue
        groups = comp_all.loc[idx].to_numpy()
        keys, counts = np.unique(groups, return_counts=True)
        pick = rng.permutation(len(keys))
        target, total, taken = val_frac * len(idx), 0, set()
        for j in pick:
            if total + counts[j] <= target:
                taken.add(keys[j])
                total += int(counts[j])
        if taken:
            sel = comp_all.isin(taken).to_numpy() & avail
            fold.loc[steps.index[sel]] = "val"

    cur = steps[(fold == "train").to_numpy() & era]
    if len(cur):
        comp = comp_all.loc[cur.index]
        labelled = cur["d_f_xy"].notna()
        size = comp[labelled].value_counts().reindex(comp.unique(), fill_value=0)
        order = sorted(size.index, key=lambda key: (-int(size[key]), str(key)))
        block = {key: rank % k for rank, key in enumerate(order)}
        fold[cur.index] = "pool"
        xfit[cur.index] = comp.map(block).to_numpy(int)
    return pd.DataFrame({"fold": fold, "xfit_fold": xfit})


def xfit_indices(splits: pd.DataFrame, block: int) -> dict[str, np.ndarray]:
    """``{"train", "val", "eval"}`` row positions for one cross-fit block.

    ``eval`` is the block's out-of-fold rows and is scored by a member that never
    saw them; ``train`` is every other pool block plus the legacy ``train`` fold;
    ``val`` is the shared early-stopping / lambda-selection fold and is in
    neither.  The held-out cell is in none of the three.
    """
    fold = splits["fold"].to_numpy()
    xf = splits["xfit_fold"].to_numpy()
    pool = fold == "pool"
    return {
        "train": np.flatnonzero((fold == "train") | (pool & (xf != block))),
        "val": np.flatnonzero(fold == "val"),
        "eval": np.flatnonzero(pool & (xf == block)),
    }


def calib_index(splits: pd.DataFrame) -> np.ndarray:
    """The ``calib`` fold of §3d — every out-of-fold prediction, and nothing else.

    The Platt map of §5c is fitted HERE and never on ``val`` (71 rows, whose
    fitted mean overshoots the base rate 0.160 vs 0.071 and worsens binned ECE
    0.0521 -> 0.0858) and never on the gate pool's own labels in-fold.
    """
    return np.flatnonzero(splits["fold"].to_numpy() == "pool")


def split_summary_v31(steps: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """:func:`split_summary_v3` over the v3.1 labels, plus the per-block census."""
    out = split_summary_v3(steps, splits["fold"])
    order = [f for f in FOLDS_V31 if f in set(out["fold"])]
    out = out.set_index("fold").reindex(order).reset_index()
    rows = []
    for block in sorted(set(splits.loc[splits["fold"] == "pool", "xfit_fold"])):
        idx = xfit_indices(splits, int(block))
        rows.append({"xfit_fold": int(block),
                     "n_train": int(len(idx["train"])),
                     "n_eval": int(len(idx["eval"])),
                     "n_eval_fxy": int(
                         targets_v3(steps.iloc[idx["eval"]])[1][:, 2].sum())})
    out.attrs["xfit"] = rows
    return out


# --------------------------------------------------------------------------- #
# v3.1 serving — the monotone Platt map and its parity assertion (§5c / §9a-H(g))
# --------------------------------------------------------------------------- #
def platt_serve(logits: np.ndarray, *, a: float, b: float) -> np.ndarray:
    """``sigmoid(a * z + b)`` — the served ``fxy`` probability of §5c.

    v3's *probability* p90-p10 on the gate fold is 0.0324 while its *logit*
    p90-p10 is 9.66 and its within-parent logit spread is 8.64, against
    ``policy_v2``'s 0.160 / 4.92 / 3.75.  The collapsed scale is therefore a
    SERVING artefact of a sigmoid on a 15% base rate and not a ranking defect —
    v3 separates candidates inside a parent MORE widely than v2 does — so the fix
    is a map on the logit, and it is required to be MONOTONE (``a > 0``) so that
    no ranking statistic can move: the same ordering, re-scaled.  Clause 4 gates
    the resulting width, not the ranking.
    """
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"the Platt slope must be positive and finite to keep "
                         f"the ordering; got a={a!r}")
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def fit_platt(logits: np.ndarray, y: np.ndarray, *, iters: int = 200,
              tol: float = 1e-10) -> tuple[float, float]:
    """``(a, b)`` for :func:`platt_serve`, fitted by damped Newton with ``a > 0``.

    The slope is carried as ``a = exp(alpha)`` rather than clipped after the
    fact, so monotonicity is a property of the PARAMETRIZATION and not of the
    data: there is no sample on which this can return a map that reorders the
    candidates, which is what §5c requires of the calibration and what
    :func:`assert_serving_parity_v31`'s clause 2 re-checks at serving time.

    Deterministic and dependency-free (no scipy): the objective is the
    two-parameter logistic log-loss, whose gradient and Hessian in
    ``(alpha, b)`` are closed forms, and the step is halved until the loss
    actually decreases.  A constant label column is REFUSED rather than fitted —
    it drives ``b`` to +-inf and the resulting map is not servable.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    if z.shape != t.shape:
        raise ValueError(f"{z.shape} logits against {t.shape} labels")
    if z.size == 0:
        raise ValueError("the Platt map cannot be fitted on an empty fold")
    if not np.isfinite(z).all() or not np.isfinite(t).all():
        raise ValueError("non-finite logits or labels reached the Platt fit")
    if t.min() == t.max():
        raise ValueError(
            f"the calibration fold carries a single label value ({t.min()}); "
            f"the fitted intercept is unbounded and the map is not servable")

    def loss(al: float, b: float) -> float:
        u = np.exp(al) * z + b
        return float(np.sum(np.logaddexp(0.0, u) - t * u))

    al, b = 0.0, 0.0
    prev = loss(al, b)
    for _ in range(iters):
        u = np.exp(al) * z + b
        p = 1.0 / (1.0 + np.exp(-u))
        w = p * (1.0 - p)
        d = p - t
        s = np.exp(al) * z                       # d u / d alpha
        g = np.array([float(d @ s), float(d.sum())])
        h = np.array([[float(w @ (s * s) + d @ s), float(w @ s)],
                      [float(w @ s), float(w.sum())]])
        h[0, 0] += 1e-9
        h[1, 1] += 1e-9
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:            # fall back to a gradient step
            step = g / max(float(np.abs(g).max()), 1.0)
        scale = 1.0
        for _ in range(40):
            cand = loss(al - scale * step[0], b - scale * step[1])
            if np.isfinite(cand) and cand <= prev:
                break
            scale *= 0.5
        else:
            break
        al, b = al - scale * step[0], b - scale * step[1]
        if prev - cand < tol:
            prev = cand
            break
        prev = cand
    a = float(np.exp(al))
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"the Platt fit did not converge to a positive slope "
                         f"(a={a!r}); the map would reorder the candidates")
    return a, float(b)


def fit_platt_v31(logits: np.ndarray, y: np.ndarray,
                  splits: pd.DataFrame) -> dict[str, float]:
    """§5c: fit the served ``fxy`` map on the ``calib`` fold and NOTHING else.

    The fold is taken from :func:`calib_index`, not from the caller, and that is
    the whole guard: ``val`` is where lambda is selected (§2c) and the holdout
    cell is opened once after the gate exists, so neither may reach the map.
    ``val``'s own numbers say why it is not merely a discipline — 71 rows whose
    fitted mean overshoots the base rate (0.160 against 0.071) and whose binned
    ECE worsens 0.0521 -> 0.0858.

    Returns the map and the census of the rows it was fitted on, so a reader of
    ``metrics.json`` can check the fold size without re-deriving the split.
    """
    idx = calib_index(splits)
    z = np.asarray(logits, dtype=np.float64).ravel()
    if z.size != len(splits):
        raise ValueError(f"the Platt fit needs one logit per corpus row: got "
                         f"{z.size} for {len(splits)} rows")
    a, b = fit_platt(z[idx], np.asarray(y, dtype=np.float64).ravel()[idx])
    return {"a": a, "b": b, "n_calib": int(len(idx)),
            "calib_base_rate": float(
                np.asarray(y, dtype=np.float64).ravel()[idx].mean())}


def assert_serving_parity_v31(train_logits: np.ndarray, serve_logits: np.ndarray,
                              *, a: float = 1.0, b: float = 0.0,
                              atol: float = 1e-5,
                              meta: dict | None = None) -> dict[str, float]:
    """The §9a-H(g) hook: serving must reproduce training, logit for logit.

    Four assertions, in the order they can fail:

    0. the checkpoint being served carries the v3.1 stamp.  The error text below
       has always CLAIMED this check ("a v3.1 checkpoint is only servable if its
       ``policy_move_v31`` feature contract is reproduced exactly") while
       comparing nothing but logits, and a v3 checkpoint scored through the v3.1
       serving path reproduces itself perfectly — so parity would have passed on
       precisely the mis-stamped artefact this hook exists to catch.  Pass the
       checkpoint's ``meta.json`` as ``meta`` and the claim becomes a test;
       ``meta=None`` keeps the logit-only form for the callers that have no
       checkpoint (the unit guards, and a parity check between two paths of the
       same run);
    1. the serving path's LOGITS reproduce the training path's within ``atol``
       — this is the featurization/contract check, and it is done on logits
       rather than probabilities because a sigmoid on a 15% base rate compresses
       a real disagreement into the fourth decimal (§5c);
    2. the Platt map preserves the ordering exactly, so no ranking statistic in
       §5a can be moved by the calibration;
    3. the served probabilities agree.

    Returns the realized spread readouts (``p90 - p10`` on both scales, and the
    max absolute logit disagreement) so clause 4 is computed from the same call
    that proves parity rather than from a second, drifting implementation.
    """
    if meta is not None:
        stamp = meta.get("policy_schema")
        if stamp != POLICY_SCHEMA_V31:
            raise AssertionError(
                f"this checkpoint is stamped {stamp!r}, not {POLICY_SCHEMA_V31!r}; "
                f"the v3.1 serving path must refuse it rather than score it "
                f"(prereg §9a-H(g))")
        names = meta.get("scalar_names")
        if names is not None and len(names) != N_SCALARS_V31:
            raise AssertionError(
                f"a {POLICY_SCHEMA_V31!r} checkpoint carries {N_SCALARS_V31} "
                f"scalars; this one carries {len(names)}, so the stamp and the "
                f"feature contract disagree")
    t = np.asarray(train_logits, dtype=np.float64).ravel()
    s = np.asarray(serve_logits, dtype=np.float64).ravel()
    if t.shape != s.shape:
        raise AssertionError(f"serving scored {s.shape} rows, training {t.shape}")
    gap = float(np.max(np.abs(t - s))) if t.size else 0.0
    if gap > atol:
        raise AssertionError(
            f"serving does not reproduce training: max |dz| = {gap:.3e} > {atol:.1e}."
            f"  A v3.1 checkpoint is only servable if its {POLICY_SCHEMA_V31!r} "
            f"feature contract is reproduced exactly (prereg §9a-H(g))")
    p_train = platt_serve(t, a=a, b=b)
    p_serve = platt_serve(s, a=a, b=b)
    if t.size:
        rank_t = np.argsort(np.argsort(t, kind="stable"), kind="stable")
        rank_p = np.argsort(np.argsort(p_serve, kind="stable"), kind="stable")
        if not np.array_equal(rank_t, rank_p):
            raise AssertionError(
                "the Platt map reordered the candidates; it must be monotone so "
                "that clause 4 cannot flatter any ranking statistic (§5c)")
        if not np.allclose(p_train, p_serve, atol=atol):
            raise AssertionError("served probabilities disagree with training")
    def _spread(v: np.ndarray) -> float:
        return float(np.percentile(v, 90) - np.percentile(v, 10)) if v.size else 0.0
    return {"max_abs_logit_gap": gap, "logit_p90_p10": _spread(t),
            "prob_p90_p10": _spread(p_serve)}


def provenance_v3(library_id: str) -> tuple[str, str]:
    """``(dataset, sym_class)`` the v3 corpus and the v3 serve path both use.

    This IS :func:`~..model.featurize.serve_provenance` — the collapse prereg
    §8-A registers.  ``data.corpus_provenance`` exists only to keep feeding the
    shipped v2 checkpoint the ``sym_class`` its corpus was (wrongly) built with;
    v3's corpus is re-featurized here, so it takes the store truth on both
    halves and the two functions stop being able to drift because there is only
    one of them left on this path.
    """
    from ..model.featurize import serve_provenance

    return serve_provenance(str(library_id))


def build_pattern_cache_v3(steps: pd.DataFrame, **kwargs) -> PatternCache:
    """:func:`~.data.build_pattern_cache` under :func:`provenance_v3`."""
    from .data import build_pattern_cache

    return build_pattern_cache(steps, provenance=provenance_v3, **kwargs)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class PolicyStepsV3(PolicySteps):
    """v1's tensor construction, v3's three targets and a per-HEAD loss weight."""

    def __init__(self, steps: pd.DataFrame, cache: PatternCache,
                 scalars: np.ndarray, weights: np.ndarray, *,
                 delta_channels: Sequence[int], augment: bool = False,
                 seed: int = 0):
        super().__init__(steps, cache, scalars, delta_channels=delta_channels,
                         augment=augment, seed=seed)
        self.labels, self.mask = targets_v3(steps)
        w = np.asarray(weights, np.float32)
        if w.ndim == 1:                       # tolerate an era-only weight
            w = np.repeat(w[:, None], len(HEADS_V3), axis=1)
        if w.shape != self.labels.shape:
            raise ValueError(f"weights {w.shape} do not match targets "
                             f"{self.labels.shape}")
        self.weights = w

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        item = super().__getitem__(i)
        item["w"] = self.weights[i]
        return item


def split_summary_v3(steps: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """The prereg §3b table, rendered from the data rather than transcribed."""
    rows = []
    for name, group in steps.groupby(fold, sort=False):
        y, m = targets_v3(group)
        fxy = m[:, 2] > 0
        parents = group["parent_record_id"].astype(str).to_numpy()
        counts = pd.Series(parents[fxy]).value_counts()
        entry = {
            "fold": name, "n_steps": len(group),
            "n_current": int(group["era_current"].sum()),
            "n_interventional": int(group["interventional"].sum()),
            "n_cells": group["cell"].nunique(),
            "n_parents": group["parent_record_id"].nunique(),
            "n_fxy": int(fxy.sum()),
            "n_fxy_interventional": int((fxy & group["interventional"]
                                         .to_numpy(bool)).sum()),
            "n_fxy_parents": int(len(counts)),
            "n_parents_ge8": int((counts >= 8).sum()),
            "y_fxy_pos": int((y[:, 2] > 0).sum()),
        }
        for h, head in enumerate(HEADS_V3):
            keep = m[:, h] > 0
            entry[f"n_{head}"] = int(keep.sum())
            col = group[EVAL_LABEL_V3[head]]
            entry[f"base_{head}"] = (float(col.mean()) if col.notna().any()
                                     else float("nan"))
        rows.append(entry)
    out = pd.DataFrame(rows).set_index("fold")
    return out.reindex([f for f in FOLDS_V3 if f in out.index]).reset_index()


__all__ = [
    "CYCLEN_TOL", "EVAL_LABEL_V3", "FOLDS_V3", "F_R_LIMIT", "HEADS_V3",
    "INTERVENTIONAL_SOURCES_V3", "NEW_SCALARS_V3", "POLICY_SCHEMA_V3",
    "PROSPECTIVE_CELL", "PolicyStepsV3", "STEPS_V3", "TARGET_CLIP_V3",
    "build_pattern_cache_v3", "build_splits_v3", "eval_labels_v3",
    "fxy_feasible", "load_universe_v3", "parent_weights", "provenance_v3",
    "scalar_features_v3", "split_summary_v3", "targets_v3", "weights_v3",
    # ---- v3.1 (additive; nothing above changes behaviour) ---------------- #
    "BURNT_ABSMOV_R_UNIT", "BURNT_ABSMOV_UNIT", "FOLDS_V31", "NEW_SCALARS_V31",
    "POLICY_SCHEMA_V31", "PROSPECTIVE_CELL_V31", "REQUIRED_V31_COLUMNS",
    "STEPS_V31", "XFIT_K", "assert_serving_parity_v31", "build_splits_v31",
    "calib_index", "load_universe_v31", "platt_serve", "scalar_features_v31",
    "split_summary_v31", "xfit_indices",
]
