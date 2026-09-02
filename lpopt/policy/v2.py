"""Corpus, target, splits and features for the **v2** move-proposal policy.

Everything here is a correction the v1 post-mortem prescribed
(``data/reports/ablation_wave_results_20260815.md`` section 9), and each one is
pre-registered in ``data/reports/policy_v2_prereg_20260817.md``:

1. **Current-era data and a current-era gate.**  v1 passed on the legacy SA
   corpus and, tested prospectively on balanced ga80/paramA moves, landed at
   parent-blocked AUC 0.492 — chance.  v2 trains on both eras with the era as an
   input and the current era reweighted to half the loss mass, and it is judged
   ONLY on held-out current-era lineage components.

2. **The reactivity covariate.**  ``d_fresh_enr_mass`` (and the parent's level)
   enter the conditioning vector.  v1's largest single commitment was a 0.64
   probability gap favouring outward ``batch_flip``, of which 0 of 10 improved:
   it was reading a reactivity/batch-label proxy as if it were the radial axis.
   ``fresh_enr_mass`` is exactly the coordinate that separates the two — it is
   identically conserved by ``rewire_swap`` / ``batch_swap`` / most
   ``fresh_relocate`` and moved only by the operators that change how much fresh
   reactivity is in the core.

3. **The target is no longer the improving FRACTION.**  See :func:`targets`.

Nothing in ``lpopt/policy/{data,net,train}.py`` is modified.  v1 is a registered
BASELINE for this round, its checkpoints must keep loading, and
``tests/test_policy_prior.py`` asserts that serving reproduces its training
probabilities — so v2 is additive and v1's feature layout is frozen.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .data import (
    HEADS, PatternCache, PolicySteps, _components, load_universe,
    scalar_features,
)

# --------------------------------------------------------------------------- #
# serving contract
# --------------------------------------------------------------------------- #
#: Stamped into every v2 member's ``meta.json`` and REQUIRED by
#: :class:`lpopt.policy.scorer.MoveScorerV2` at load.  It names the serving
#: contract — v1's 36 move/context scalars plus :data:`NEW_SCALARS`, over the
#: ``v6b`` board encoding, read as a within-parent RANKER — and nothing else.
#: Bump it whenever a change would make an older checkpoint's conditioning
#: vector mean something different, so the old checkpoint fails LOUDLY at load
#: instead of being silently mis-fed at score time.
POLICY_SCHEMA_V2 = "policy_move_v2"

# --------------------------------------------------------------------------- #
# eras
# --------------------------------------------------------------------------- #
#: The live operating point: the libraries the running program actually loads.
CURRENT_ERA_LIBRARIES: tuple[str, ...] = ("ga80", "paramA")
#: Lineage tags whose rows are VERIFIED single moves off a fixed parent, produced
#: by the interventional waves rather than by an optimiser that chose them.  They
#: are the only current-era rows with enough candidates per parent to answer the
#: within-parent question, so they carry the gate.
INTERVENTIONAL_SOURCES: tuple[str, ...] = (
    "ablation_paramA", "batchswap_enum", "batchswap_enum_625",
)

# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
#: Clip on the improvement magnitude, per head, in FOM units.
#:
#: Registered rule: the smallest 5e-3-rounded value that leaves EVERY improvement
#: in the current-era interventional set unsaturated.  Measured on the corpus of
#: 2026-08-17: the largest ``-d_f_r`` over the 577 verified single moves is
#: 0.0265 and the largest ``-d_node_peak`` is 0.0341.
#:
#: The clip is what makes an expected-improvement target readable.  Pooled over
#: the whole corpus the improvement magnitude spans four orders — p50 0.019,
#: p99 1.63, max 2.18 — and essentially all of that spread is PARENT DIFFICULTY:
#: a 2.18 "improvement" is a rescue off a broken board, not a good move.  An
#: unclipped regression would put nearly all of its gradient on those rows and
#: would learn the same board-difficulty signal the pooled AUC is already
#: confounded by.  Clipped at the elite band, the target keeps full magnitude
#: resolution exactly where the gate and the deployment consumer live and
#: saturates the difficulty tail to a constant.
TARGET_CLIP: dict[str, float] = {"fr": 0.030, "flat": 0.035}
#: FOM whose decrease is the improvement, per head.
TARGET_DELTA: dict[str, str] = {"fr": "d_f_r", "flat": "d_node_peak"}
#: Binary label kept for EVALUATION only — the v1 gate metric, so the comparison
#: against v1 is on v1's own terms.  It is never a training target in v2.
EVAL_LABEL: dict[str, str] = {"fr": "improved_fr", "flat": "improved_flat"}


def targets(steps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(y[N, 2] in [0, 1], mask[N, 2])`` — normalized clipped expected improvement.

    ``y = min(max(0, -d_FOM), c) / c``.

    **Why not the improving fraction.**  ``improved_fr`` is magnitude-blind: it
    scores a +1e-4 nudge off an elite and a −2.18 rescue off a broken board
    identically, and the ablation wave measured what that costs — inward
    ``fresh_relocate`` improves 14.3% of the time while outward improves 3.6%, so
    the fraction ranks inward first, yet inward's mean ``d_f_r`` is +0.539
    against outward's +0.100.  The fraction prefers the lottery.

    **Why not a "reliable improver" band either.**  That was the other candidate
    and the data rejects it.  Every improvement in the current-era interventional
    set already has |d_f_r| ≤ 0.0265, so a band at or above 0.03 is *identical*
    to ``improved_fr`` on the very rows the gate is decided on — a relabelling
    that changes nothing.  Tightening it to 0.01 does change something, and in
    the wrong direction: it turns 12 of the 42 real improvements into negatives,
    including the largest ones, which are precisely the moves ``regret@8`` is
    asking the policy to find.  A band is a binary, and a binary cannot express
    "this move improves by 0.026 and that one by 0.001".

    **Why expected improvement specifically.**  The consumer is an 8-call probe
    that keeps its best board.  Maximising the best of k draws is what E[max(0,
    −Δ)] is the matched acquisition for; the improving fraction is the
    probability-of-improvement acquisition, which is the magnitude-blind one.
    The deployment metric this round adds (``regret@8``) is the empirical form of
    the same quantity, so target and metric are the same object measured twice.
    """
    n = len(steps)
    y = np.zeros((n, len(HEADS)), np.float32)
    mask = np.zeros((n, len(HEADS)), np.float32)
    both = steps["both_converged"].fillna(False).to_numpy(bool)
    for h, head in enumerate(HEADS):
        delta = steps[TARGET_DELTA[head]].to_numpy(np.float64)
        ok = both & np.isfinite(delta)
        gain = np.clip(-np.nan_to_num(delta, nan=0.0), 0.0, TARGET_CLIP[head])
        y[:, h] = (gain / TARGET_CLIP[head]).astype(np.float32)
        mask[:, h] = ok.astype(np.float32)
    return y, mask


def eval_labels(steps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(y[N, 2] binary, mask[N, 2])`` — the v1 gate label, for evaluation only."""
    n = len(steps)
    y = np.zeros((n, len(HEADS)), np.float32)
    mask = np.zeros((n, len(HEADS)), np.float32)
    for h, head in enumerate(HEADS):
        raw = steps[EVAL_LABEL[head]]
        mask[:, h] = raw.notna().to_numpy(np.float32)
        y[:, h] = raw.fillna(False).astype(bool).to_numpy(np.float32)
    return y, mask


# --------------------------------------------------------------------------- #
# universe and splits
# --------------------------------------------------------------------------- #
def load_universe_v2(path: str) -> pd.DataFrame:
    """v1's universe, plus the era flag.  Same-cell, at least one head labelled."""
    steps = load_universe(path)
    steps["era_current"] = steps["library_id"].isin(
        CURRENT_ERA_LIBRARIES).astype(bool)
    steps["interventional"] = steps["lineage_source"].isin(
        INTERVENTIONAL_SOURCES).astype(bool)
    return steps


def build_splits_v2(steps: pd.DataFrame, *, seed: int = 20260817,
                    val_frac: float = 0.10) -> pd.Series:
    """Assign every row to ``train`` / ``val`` / ``gate_cur``.  Deterministic.

    **The gate fold.**  Lineage connected components are computed WITHIN the
    current era; the components are ranked by labelled-row count (descending,
    ties broken by the component key so the order is total and deterministic) and
    assigned ALTERNATELY — rank 0 to the gate, rank 1 to train, rank 2 to the
    gate, and so on.  No RNG and no label is used: component size is known before
    any outcome is read.

    Alternation rather than a random draw because the quantity that matters is
    scarce and lumpy.  Only 17 current-era components carry ≥ 12 labelled
    candidates, and ``regret@8`` can only be evaluated on a parent with enough
    candidates to make a top-8 a real selection.  A random 50/50 draw over 544
    components would routinely put most of the high-fan-out mass on one side; a
    size-ranked alternation splits it as evenly as a deterministic rule can.

    **The train/val split.**  ``val`` is 10% of the remaining components, drawn
    independently inside each era so both eras are represented, and is used for
    early stopping and nothing else.  Legacy rows are never in the gate.
    """
    fold = pd.Series("train", index=steps.index, dtype=object)
    era = steps["era_current"].to_numpy(bool)

    cur = steps[era]
    comp = pd.Series(_components(cur), index=cur.index)
    labelled = cur["improved_fr"].notna() | cur["improved_flat"].notna()
    size = comp[labelled].value_counts()
    size = size.reindex(comp.unique(), fill_value=0)
    order = sorted(size.index, key=lambda k: (-int(size[k]), str(k)))
    gate_keys = set(order[0::2])
    fold[cur.index[comp.isin(gate_keys).to_numpy()]] = "gate_cur"

    rng = np.random.default_rng(seed)
    pool = (fold == "train").to_numpy()
    for flag in (False, True):                       # legacy first, then current
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


def era_weights(steps: pd.DataFrame, fold: pd.Series, *,
                cap: float = 20.0) -> np.ndarray:
    """Per-row loss weight: current-era rows carry HALF the training loss mass.

    ``w = n_train_legacy / n_train_current`` on the current era and 1.0 on the
    legacy era, capped.  Parameter-free by construction — there is no weight to
    tune, only the registered statement "the two eras count equally".
    """
    train = (fold == "train").to_numpy()
    era = steps["era_current"].to_numpy(bool)
    n_cur = int((train & era).sum())
    n_leg = int((train & ~era).sum())
    w_cur = min(float(cap), (n_leg / n_cur) if n_cur else 1.0)
    return np.where(era, w_cur, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
#: Fresh-enrichment mass of a single ``batch_flip`` at the current era — the
#: natural unit for the reactivity delta, so the feature reads "how many
#: batch-flip-equivalents of fresh reactivity did this move add or remove".
#: Measured (``ablation_wave_results_20260815`` section 2c): max |d| = 1.2017.
FLIP_MASS_UNIT = 1.2
#: Centre and scale of the parent's absolute fresh-enrichment mass over the
#: same-cell corpus (median 653.4, sd 23.7), rounded.
PARENT_MASS_CENTER, PARENT_MASS_SCALE = 653.0, 25.0

#: The three columns v2 adds to v1's 36.  Named so the results report and the
#: checkpoint metadata agree on exactly what changed.
NEW_SCALARS: tuple[str, ...] = (
    "d_fresh_enr_mass", "parent_fresh_enr_mass", "era_current")


def scalar_features_v2(steps: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """v1's 36 move/context scalars plus the three v2 additions.

    ``scalar_features`` is CALLED, not copied: v1's vector must stay bit-identical
    inside v2's so the two models differ by the additions and by the target, and
    by nothing else that could explain a difference in the gate.
    """
    base, names = scalar_features(steps)
    extra = {
        "d_fresh_enr_mass": np.nan_to_num(
            steps["d_fresh_enr_mass"].to_numpy(np.float64), nan=0.0
        ) / FLIP_MASS_UNIT,
        "parent_fresh_enr_mass": (
            np.nan_to_num(steps["parent_fresh_enr_mass"].to_numpy(np.float64),
                          nan=PARENT_MASS_CENTER)
            - PARENT_MASS_CENTER) / PARENT_MASS_SCALE,
        # Era is a CELL attribute, known before any move is proposed (the deck
        # names the library), so it is not provenance in the sense v1 excluded.
        # ``lineage_source`` remains excluded: that IS provenance.
        "era_current": steps["era_current"].to_numpy(np.float64),
    }
    order = sorted([*names, *extra])
    lookup = {n: base[:, i] for i, n in enumerate(names)}
    lookup.update({k: v.astype(np.float32) for k, v in extra.items()})
    return np.stack([lookup[n] for n in order], axis=1).astype(np.float32), order


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class PolicyStepsV2(PolicySteps):
    """v1's tensor construction, with the regression target and a loss weight."""

    def __init__(self, steps: pd.DataFrame, cache: PatternCache,
                 scalars: np.ndarray, weights: np.ndarray, *,
                 delta_channels: Sequence[int], augment: bool = False,
                 seed: int = 0):
        super().__init__(steps, cache, scalars, delta_channels=delta_channels,
                         augment=augment, seed=seed)
        self.labels, self.mask = targets(steps)
        self.weights = np.asarray(weights, np.float32)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        item = super().__getitem__(i)
        item["w"] = np.float32(self.weights[i])
        return item


def split_summary_v2(steps: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """Rows / cells / parents / era mix / target mass per fold — the prereg table."""
    rows = []
    for name, group in steps.groupby(fold, sort=False):
        y, m = targets(group)
        entry = {
            "fold": name, "n_steps": len(group),
            "n_current": int(group["era_current"].sum()),
            "n_interventional": int(group["interventional"].sum()),
            "n_cells": group["cell"].nunique(),
            "n_parents": group["parent_record_id"].nunique(),
        }
        for h, head in enumerate(HEADS):
            keep = m[:, h] > 0
            entry[f"n_{head}"] = int(keep.sum())
            entry[f"base_{head}"] = float(
                group[EVAL_LABEL[head]].mean()) if keep.any() else float("nan")
            entry[f"mean_y_{head}"] = float(y[keep, h].mean()) if keep.any() else float("nan")
        rows.append(entry)
    order = ["train", "val", "gate_cur"]
    out = pd.DataFrame(rows).set_index("fold")
    return out.reindex([o for o in order if o in out.index]).reset_index()
