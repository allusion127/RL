"""Train and judge the **v3** move-proposal policy.

    python -m lpopt.policy.train_v3 --out-dir runs/policy_v3 --seeds 5

The protocol is fixed by ``data/reports/policy_v3_prereg_20260831.md``, written
before any v3 weight existed.  Four things differ from v2 and this module is the
only place they are written down:

* a THIRD head, ``fxy``, whose target is the constraint-gated normalized clipped
  expected improvement of the campaigns' actual objective (``v3.targets_v3``);
* FOUR folds — a whole unseen cell (``prospective_cell``) is removed before any
  other rule runs and is opened once, after the gate numbers exist;
* the loss weight is per HEAD, ``w_era x w_parent``, because the parent-equal
  weighting is an ``fxy``-only statement (prereg §3c);
* the consumer metric is ``regret@4-of-8`` and the baseline set gains ``gd_rule``
  (the single new descriptor, used as a rule) and ``policy_v2`` (the "F_xy is a
  monotone transform of F_r" hypothesis, at ranking level).

``--emit-v2-baseline PATH`` runs the SHIPPED v2 ensemble over the v3 evaluation
folds with v2's own features and writes its probabilities to a CSV.  It is run
BEFORE any v3 weight exists and BEFORE the corpus is re-mined, so the baseline is
blind, exactly as v2's blind v1 baseline was.

``precision@32 of 256`` is REPORTED and NOT gated this round; the reason is
pre-registered (§5e) and rests on v2's measurement that the pooled metric is
confounded by parent difficulty, not on any v3 number.

v3.1
----
``policy_v31_prereg_20260831_DRAFT.md`` adds a SECOND stage — trunk frozen,
``fr`` / ``flat`` frozen, a new ``fxy`` branch trained with
``BCE + lambda * listwise-CE`` against a raw-gain teacher — behind ``--stage2 on``.
Every v3.1 flag defaults off and :func:`assert_v3_path_untouched` refuses a
half-enabled run, so with no flags this module is the v3 protocol it was, bit
for bit, and ``metrics.json`` records which of the two ran under ``"v31"``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    COND_SCHEMA, MOVE_CLASSES, PatternCache, PolicySteps, build_pattern_cache,
    corpus_fingerprint, pick_delta_channels,
)
from .net import PolicyNet, PolicyNetConfig, count_parameters
from .train import (
    BATCH_CANDIDATES, N_BOOTSTRAP, auc, calibration, parent_blocked_auc,
    precision_at_k, _boot_ci,
)
from .train_v2 import (
    _TorchSteps, _delta, _paired_parent_bootstrap, _per_parent_blocked,
    _predict, _val_spearman, regret_at_k,
)
from .v2 import (
    CURRENT_ERA_LIBRARIES, POLICY_SCHEMA_V2, scalar_features_v2,
)
from .v3 import (
    EVAL_LABEL_V3, HEADS_V3, MIN_POOL_LIVE_V31, NEW_SCALARS_V3,
    NEW_SCALARS_V31, N_SCALARS_V3, N_SCALARS_V31, POLICY_SCHEMA_V3,
    POLICY_SCHEMA_V31, PROSPECTIVE_CELL, PROSPECTIVE_CELL_V31,
    REQUIRED_V31_COLUMNS,
    PolicyStepsV3, STEPS_V3, STEPS_V31, TARGET_CLIP_V3, build_pattern_cache_v3,
    build_splits_v3, build_splits_v31, calib_index, fxy_feasible,
    load_universe_v3, load_universe_v31, provenance_v3, scalar_features_v3,
    scalar_features_v31, split_summary_v3, targets_v3, weights_v3,
    xfit_indices,
)

#: The consumer spends 4 of the 8 moves an interventional parent offers.
PROBE_K = 4
#: A parent needs at least this many F_xy-labelled candidates for the selection
#: to BE a selection.  8, not v2's 10, and the reason is the data's structure
#: (prereg §1f): the interventional waves computed exactly 8 moves per parent, so
#: ``regret@8`` is identically zero on them, and only 6 parents in the whole
#: corpus have >= 10 candidates.  At 8 the sample is 38 gate parents against
#: v2's 9.
REGRET_MIN_CANDIDATES = 8
#: Listwise report metric.
NDCG_K = 4

BASELINES: tuple[str, ...] = (
    "random", "class_freq", "periph", "gd_rule", "policy_v2")

#: The registered gate thresholds (§6).  Here so ``metrics.json`` carries the
#: bar it was judged against and a later reader cannot mistake a moved bar for a
#: passed one.
GATE_AUC = 0.65
GATE_AUC_CI_LO = 0.50
TRANSFER_AUC = 0.60
#: v2's ``fr`` head on its own gate fold.  A v3 ``fr`` head more than 0.05 below
#: this is a REGRESSION and is reported as one (§6).
V2_FR_PARENT_BLOCKED = 0.728
FR_REGRESSION_TOL = 0.05


# --------------------------------------------------------------------------- #
# listwise metric
# --------------------------------------------------------------------------- #
def ndcg_at_k(scores: np.ndarray, gain: np.ndarray, parents: np.ndarray, *,
              k: int = NDCG_K, min_candidates: int = REGRET_MIN_CANDIDATES,
              reps: int = 256, seed: int = 20260831
              ) -> tuple[np.ndarray, np.ndarray]:
    """Per-parent NDCG@k of the scorer's ordering.  Higher is better.

    The listwise companion to :func:`~.train_v2.regret_at_k`: regret asks only
    "was the best move inside the k you picked", NDCG asks whether the whole
    top-k ORDER tracks the gains.  Reported, never gated — the consumer keeps
    the best of its k calls and does not care in what order they ran.

    Ties are broken uniformly at random and averaged over ``reps`` draws, the
    same treatment :func:`regret_at_k` gives them and for the same reason: the
    frequency baselines have a handful of distinct values.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(parents, kind="mergesort")
    p_sorted = parents[order]
    bounds = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    out, keys = [], []
    discount = 1.0 / np.log2(np.arange(k) + 2.0)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[a:b]
        if len(idx) < min_candidates:
            continue
        g, s = gain[idx], scores[idx]
        g = np.clip(g, 0.0, None)                      # a loss contributes no gain
        ideal = float((np.sort(g)[::-1][:k] * discount[:min(k, len(g))]).sum())
        if ideal <= 0.0:
            continue
        got = np.empty(reps)
        for r in range(reps):
            take = np.lexsort((rng.random(len(s)), -s))[:k]
            got[r] = float((g[take] * discount[:len(take)]).sum())
        out.append(got.mean() / ideal)
        keys.append(p_sorted[a])
    return np.array(out), np.array(keys, dtype=object)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def gd_rule_sign(train: pd.DataFrame) -> float:
    """The sign of the ``gd_rule`` baseline, fixed ONCE on the training fold.

    The registered rule is ``-d_fresh_gd_mass`` (prereg §5c) and the sign is
    fitted the way the other three baselines are fitted: on ``train`` only,
    against ``improved_fxy``, before any evaluation fold is read.  Returned
    rather than hidden so ``metrics.json`` records which way it came out.
    """
    col = pd.to_numeric(train["d_fresh_gd_mass"], errors="coerce").to_numpy(float)
    lab = train[EVAL_LABEL_V3["fxy"]]
    keep = np.isfinite(col) & lab.notna().to_numpy()
    if keep.sum() < 2:
        return -1.0
    y = lab[keep].astype(bool).to_numpy()
    if y.all() or not y.any():
        return -1.0
    return -1.0 if auc(-col[keep], y) >= 0.5 else 1.0


def baseline_scores_v3(frame: pd.DataFrame, train: pd.DataFrame, head: str,
                       v2: pd.DataFrame | None, *, gd_sign: float,
                       seed: int = 20260831) -> dict[str, np.ndarray]:
    """The five pre-registered baselines.  The first four are fitted on TRAIN only.

    ``policy_v2`` is not fitted at all: it is the shipped v2 ensemble's ``fr``
    head, read from the blind CSV emitted before v3 existed.  It is in the set
    because it IS the hypothesis "F_xy is a monotone transform of F_r" — rejected
    at effect level by the r1 wave's transfer coefficients, untested at ranking
    level until here.
    """
    col = EVAL_LABEL_V3[head]
    rng = np.random.default_rng(seed)
    lab = train[col]
    prior = float(lab.mean()) if lab.notna().any() else 0.5
    # A move class can be entirely UNLABELLED on this head — ``improved_fxy`` is
    # known on 1,309 rows against ``improved_fr``'s 21,132 — and a groupby mean
    # over an all-NA boolean group is pd.NA, not a number.  Such a class falls
    # back to the fold's base rate, which is what "we have never seen this class
    # improve or fail here" means.
    by_class = {str(k): (float(v) if pd.notna(v) else prior)
                for k, v in train.groupby("move_class")[col].mean().items()}
    klass = frame["move_class"].astype(str).to_numpy()

    out = {
        "random": rng.random(len(frame)),
        "class_freq": np.array([float(by_class.get(k, prior)) for k in klass]),
        "periph": frame["d_fresh_share_periph"].to_numpy(np.float64),
        "gd_rule": gd_sign * pd.to_numeric(
            frame["d_fresh_gd_mass"], errors="coerce").to_numpy(np.float64),
    }
    if v2 is not None:
        table = v2.set_index("child_record_id")["p_improve_fr"]
        out["policy_v2"] = frame["child_record_id"].map(table).to_numpy(np.float64)
    else:
        out["policy_v2"] = np.full(len(frame), np.nan)
    return out


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def masked_bce_soft(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                    w: torch.Tensor) -> torch.Tensor:
    """Head-masked BCE against the soft target, weighted PER HEAD — ``revB``.

    ``w`` is ``[B, 3]`` here where v2's was ``[B]``: ``w_parent`` applies to the
    ``fxy`` head only (prereg §3c), so the weight cannot be a row scalar.  The
    loss itself is v2's declared deviation, carried verbatim: cross-entropy is a
    proper scoring rule for a target that is a number in [0, 1] and its gradient
    through the sigmoid is ``sigma(z) - y``, so the model cannot stall by
    saturating the way the pre-registered Huber did (v2 results §2a — five seeds
    stopped at epoch 0).  Re-registering that Huber would be knowingly
    reproducing a measured failure.
    """
    loss = nn.functional.binary_cross_entropy_with_logits(pred, y, reduction="none")
    weight = m * w
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


# --------------------------------------------------------------------------- #
# v3.1 stage 2 — frozen trunk + a new fxy branch + a listwise term (§2b)
#
# EVERYTHING in this section is reached only from ``--stage2 on``, which is NOT
# the default.  With the flag off ``main`` runs the v3 path it always ran, and
# ``assert_v3_path_untouched`` is called unconditionally to say so in the
# manifest rather than in a comment.
# --------------------------------------------------------------------------- #
#: Softmax temperature of the listwise teacher.  It is the REGISTERED F_xy clip
#: constant ``TARGET_CLIP_V3["fxy"]`` re-used, so stage 2 introduces no new free
#: constant (§2b) and a change to the clip cannot silently desynchronize the two.
TEACHER_TEMP = TARGET_CLIP_V3["fxy"]
#: Uniform smoothing of the teacher, over the FEASIBLE candidates only.
TEACHER_EPS = 0.10
#: The registered lambda grid.  Selected on ``val`` 3-head mean Spearman ALONE
#: (§2c); the gate pool, the cross-fit folds and the prospective cell are not
#: read.  The pre-registered expectation is that 0 wins — the 2-seed pilot had
#: 0.4557 / 0.4465 / 0.4445 — and that outcome ships v3.1 without the listwise
#: term and closes the line with a measurement, which is a legitimate result.
LAMBDA_GRID: tuple[float, ...] = (0.0, 0.3, 1.0)
#: Stage 2's own optimizer settings (§9b).
STAGE2_LR = 1e-4
STAGE2_EPOCHS = 40
STAGE2_PATIENCE = 10


def listwise_teacher(steps: pd.DataFrame, *, temp: float = TEACHER_TEMP,
                     eps: float = TEACHER_EPS) -> np.ndarray:
    """Per-row teacher mass ``q`` over each parent's candidates — RAW gain.

    ``q_i ∝ exp(u_i / T) · feasible_i`` with ``u = −d_f_xy``, normalized inside
    the parent and smoothed uniformly over that parent's FEASIBLE candidates::

        q = (1 − eps) · softmax_feasible(u / T) + eps / n_feasible

    Rows outside a usable group get 0 and contribute nothing.

    **Why the teacher is raw gain while the gate is the registered gain, and why
    that is not a mismatch anyone should quietly fix.**  A registered-gain
    teacher discards every parent whose ``y_fxy`` sums to zero, which takes the
    training groups from 57 parents / 368 rows to 35 / 232, and the 2-seed pilot
    measured ``fxy`` pb-AUC collapsing 0.8371 -> 0.7029 (full net) / 0.6869
    (head-only) when the two are aligned.  The listwise term only has to induce
    an ORDER; the clip and the feasibility gate are what DEFINE the metric.  The
    infeasible rows keep ``q = 0`` but stay in the student's softmax denominator
    (:func:`listwise_ce`), which is how "do not rank an infeasible move highly"
    reaches the gradient at all.
    """
    if not (0.0 <= eps < 1.0):
        raise ValueError(f"teacher smoothing must be in [0, 1), got {eps!r}")
    if temp <= 0.0:
        raise ValueError(f"teacher temperature must be positive, got {temp!r}")
    u = -pd.to_numeric(steps["d_f_xy"], errors="coerce").to_numpy(float)
    mask = targets_v3(steps)[1][:, 2] > 0
    feasible = fxy_feasible(steps) & mask & np.isfinite(u)
    parents = steps["parent_record_id"].astype(str).to_numpy()

    q = np.zeros(len(steps), float)
    order = np.argsort(parents, kind="mergesort")
    p_sorted = parents[order]
    bounds = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[a:b]
        if mask[idx].sum() < 2:                 # not a within-parent question
            continue
        live = idx[feasible[idx]]
        if not len(live):                       # nothing rankable in this parent
            continue
        z = u[live] / temp
        w = np.exp(z - z.max())
        q[live] = (1.0 - eps) * (w / w.sum()) + eps / len(live)
    return q


def listwise_groups(steps: pd.DataFrame, q: np.ndarray) -> list[np.ndarray]:
    """Parents with >= 2 F_xy-labelled candidates and some teacher mass (§2b)."""
    mask = targets_v3(steps)[1][:, 2] > 0
    parents = steps["parent_record_id"].astype(str).to_numpy()
    out: list[np.ndarray] = []
    for key in pd.unique(parents):
        idx = np.flatnonzero((parents == key) & mask)
        if len(idx) >= 2 and q[idx].sum() > 0.0:
            out.append(idx)
    return out


def listwise_ce(logits: torch.Tensor, q: torch.Tensor,
                sizes: Sequence[int]) -> torch.Tensor:
    """``(1/|G|) Σ_p Σ_i −q_i log softmax_p(z)_i`` over the packed groups.

    ``logits`` and ``q`` are the concatenation of the groups in ``sizes`` order.
    The softmax denominator runs over EVERY candidate of the parent, infeasible
    ones included, while ``q`` is zero there — that asymmetry is the point (§2b).
    """
    total = logits.new_zeros(())
    off = 0
    for n in sizes:
        z, w = logits[off:off + n], q[off:off + n]
        total = total - (w * torch.log_softmax(z, dim=0)).sum()
        off += n
    return total / max(len(sizes), 1)


class Stage2FxyBranch(nn.Module):
    """Frozen v3 net + a NEW ``fxy`` branch on the same pooled representation.

    The branch is ``2W + n_cond -> 256 -> 256 -> 1``, i.e. the head's own
    geometry, and its three layers are initialised FROM the frozen head with the
    last layer taking that head's ``fxy`` row.  So at epoch 0 the branch
    reproduces stage 1's ``fxy`` logit exactly (:func:`assert_stage2_init_is_stage1`)
    and stage 2 starts where stage 1 stopped rather than from noise.

    **Why two stages at all — the measured reason.**  A listwise term on the full
    net destroys the ``fr`` head: the 2-seed pilot read ``fr`` 0.7808 at
    lambda = 0, 0.6664 at 0.3 and 0.6900 at 1.0, and 0.6664 is below the
    registered regression floor 0.678.  With the trunk and the ``fr``/``flat``
    outputs frozen, clause 5 is satisfied STRUCTURALLY — those two logits are the
    same tensor stage 1 produced — instead of being satisfied by luck.
    """

    def __init__(self, base: PolicyNet):
        super().__init__()
        if base.arm != "cnn":
            raise ValueError("stage 2 is registered for the cnn arm only")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

        cfg = base.config
        self.fxy_index = HEADS_V3.index("fxy")
        self.branch = nn.Sequential(
            nn.Linear(2 * cfg.width + cfg.n_cond, cfg.head_hidden), nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden), nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        src = [m for m in base.head if isinstance(m, nn.Linear)]
        dst = [m for m in self.branch if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for a, b in zip(src[:-1], dst[:-1], strict=True):
                b.weight.copy_(a.weight)
                b.bias.copy_(a.bias)
            dst[-1].weight.copy_(src[-1].weight[self.fxy_index:self.fxy_index + 1])
            dst[-1].bias.copy_(src[-1].bias[self.fxy_index:self.fxy_index + 1])

    def trunk(self, cells: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """The ``[B, 2W + n_cond]`` pooled representation ``PolicyNet.forward`` builds.

        Recomputed here rather than returned by ``PolicyNet``: ``lpopt/policy/net.py``
        is a v1/v2/v3 serving contract and stage 2 must not need an edit to it.
        The pooling is the same three lines, and
        :func:`assert_stage2_init_is_stage1` is what proves the copy did not
        drift — it compares this path's logits against ``base(cells, cond)``.
        """
        base = self.base
        fuel_mask = cells[:, 0:1]
        h = base.stem(cells)
        for b, block in enumerate(base.blocks):
            h = block(h)
            if str(b) in base.films:
                h = base.films[str(b)](h, cond)
        denom = fuel_mask.sum(dim=(2, 3)).clamp_min(1.0)
        mean = (h * fuel_mask).sum(dim=(2, 3)) / denom
        masked = h.masked_fill(fuel_mask == 0, torch.finfo(h.dtype).min)
        peak = masked.amax(dim=(2, 3))
        return torch.cat([mean, peak, cond], dim=1)

    def forward(self, cells: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``[B, 3]`` — ``fr`` / ``flat`` from the FROZEN head, ``fxy`` from the branch.

        The trunk is evaluated ONCE and fed to both the frozen head and the
        branch.  It used to be evaluated twice (``self.base(...)`` then
        ``self.trunk(...)``), which doubled the convolutional cost of every
        stage-2 batch for an identical result — ``PolicyNet.forward`` on the
        ``cnn`` arm IS ``self.head(trunk(...))``, and
        :func:`assert_stage2_init_is_stage1` is what keeps that equality honest
        by comparing this path against ``base(cells, cond)`` directly.  The
        trunk is frozen, so computing it under ``no_grad`` costs the branch no
        gradient: the branch's own weights are the only parameters on the path.
        """
        with torch.no_grad():
            pooled = self.trunk(cells, cond)
            frozen = self.base.head(pooled)
        z = self.branch(pooled)
        out = frozen.clone()
        out[:, self.fxy_index] = z[:, 0]
        return out


def assert_stage2_init_is_stage1(model: Stage2FxyBranch, cells: torch.Tensor,
                                 cond: torch.Tensor, *, atol: float = 1e-5) -> None:
    """§9a-H(b): at initialisation stage 2 IS stage 1, on all three heads.

    ``fr`` / ``flat`` must be bit-identical for the life of stage 2 (they are the
    frozen tensor), and ``fxy`` must match at epoch 0 because the branch was
    seeded from the frozen head's ``fxy`` row.  A failure here is an
    implementation defect and clause 5 must not be lowered to accommodate it.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        got, want = model(cells, cond), model.base(cells, cond)
    model.train(was_training)
    if not torch.equal(got[:, :2], want[:, :2]):
        raise AssertionError("stage 2 changed the frozen fr/flat logits")
    gap = float((got[:, 2] - want[:, 2]).abs().max()) if len(got) else 0.0
    if gap > atol:
        raise AssertionError(
            f"the stage-2 branch was not seeded from the frozen fxy row: "
            f"max |dz| = {gap:.3e} > {atol:.1e}")


def train_stage2(base: PolicyNet, *, sets: dict[str, PolicyStepsV3],
                 frames: dict[str, pd.DataFrame], device: str, lam: float,
                 epochs: int = STAGE2_EPOCHS, lr: float = STAGE2_LR,
                 patience: int = STAGE2_PATIENCE, batch_size: int = 256,
                 teacher: str = "raw", temp: float = TEACHER_TEMP,
                 eps: float = TEACHER_EPS, seed: int = 20260903,
                 ) -> tuple[Stage2FxyBranch, dict[str, Any]]:
    """Train the ``fxy`` branch with ``BCE + lam * listwise-CE``.  Nothing else moves.

    Batches are whole PARENTS (§9b) because the listwise term is defined inside
    a parent; a row-shuffled batch would split groups across steps and the
    softmax would be taken over an accidental subset.
    """
    if teacher != "raw":
        raise ValueError(
            "the registered teacher is the RAW gain (§2b); the registered-gain "
            "teacher is a measured failure (fxy pb-AUC 0.8371 -> 0.7029) and is "
            "not offered here")
    torch.manual_seed(seed)
    model = Stage2FxyBranch(base).to(device)
    train_set, train_frame = sets["train"], frames["train"]
    probe = _collate(train_set, np.arange(min(8, len(train_set))), device)
    assert_stage2_init_is_stage1(model, probe["cells"], probe["cond"])

    q_np = listwise_teacher(train_frame, temp=temp, eps=eps)
    groups = listwise_groups(train_frame, q_np)
    fxy_rows = np.flatnonzero(targets_v3(train_frame)[1][:, 2] > 0)
    opt = torch.optim.AdamW(model.branch.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    best, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        model.base.eval()                       # frozen: no BN/dropout drift
        tot, nb = 0.0, 0
        for rows in _parent_batches(groups, fxy_rows, batch_size, rng):
            batch = _collate(train_set, rows, device)
            pred = model(batch["cells"], batch["cond"])
            loss = masked_bce_soft(pred, batch["y"], batch["m"], batch["w"])
            if lam > 0.0:
                packed, sizes = [], []
                for g in groups:
                    hit = np.flatnonzero(np.isin(rows, g))
                    if len(hit) == len(g) and len(g) >= 2:
                        packed.append(hit)
                        sizes.append(len(hit))
                if sizes:
                    take = torch.as_tensor(np.concatenate(packed), device=device)
                    qt = torch.as_tensor(
                        q_np[rows][np.concatenate(packed)], dtype=pred.dtype,
                        device=device)
                    loss = loss + lam * listwise_ce(
                        pred[take, HEADS_V3.index("fxy")], qt, sizes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.branch.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1

        model.eval()
        vloader = DataLoader(_TorchSteps(sets["val"]), batch_size=batch_size,
                             shuffle=False, num_workers=0)
        rho = _val_spearman(_predict(model, vloader, device),
                            sets["val"].labels, sets["val"].mask)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1),
                        "val_spearman": rho})
        if rho > best + 1e-7:
            best, best_epoch, stale = rho, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.branch.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.branch.load_state_dict(best_state)
    return model, {"stage": 2, "lam": float(lam), "teacher": teacher,
                   "teacher_temp": temp, "teacher_eps": eps,
                   "n_groups": len(groups), "n_fxy_rows": int(len(fxy_rows)),
                   "best_epoch": best_epoch, "best_val_spearman": best,
                   "history": history}


def select_stage2_lambda(scores: dict[float, dict[int, float]]) -> float:
    """The ONE listwise weight the ensemble carries (§9d), from val alone.

    ``scores`` is ``{lambda: {seed: best_val_spearman}}``.  The pick is the
    lambda with the highest SEED-MEAN val Spearman, ties broken toward the
    smaller lambda (the smaller lambda is the smaller deviation from v3, so a
    tie must not buy a listwise term).

    Per-seed selection is the defect this replaces.  Choosing inside the seed
    loop lets members of one ensemble carry different lambdas, and the gate then
    averages logits produced under different objectives while ``metrics.json``
    reports a single ``stage2_lam_selected`` per member -- there would be no one
    lambda that the gated artefact is.  §9d registers that exactly one selected
    lambda reaches the gate, and the selection reads ``val`` only: the pool, the
    blocks and the prospective cell are not touched here.
    """
    if not scores:
        raise ValueError("no lambda was scored; the grid was empty")
    means: dict[float, float] = {}
    for lam, per_seed in scores.items():
        if not per_seed:
            raise ValueError(f"lambda {lam} was scored on no seed")
        vals = [float(v) for v in per_seed.values()]
        if not all(np.isfinite(vals)):
            raise ValueError(f"lambda {lam} produced a non-finite val score: "
                             f"{per_seed}")
        means[float(lam)] = float(np.mean(vals))
    seeds = {tuple(sorted(d)) for d in scores.values()}
    if len(seeds) != 1:
        raise ValueError(f"the lambda grid was not scored on the same seeds: "
                         f"{sorted(seeds)}")
    return min(means, key=lambda lam: (-means[lam], lam))


def _parent_batches(groups: list[np.ndarray], extra: np.ndarray, batch_size: int,
                    rng: np.random.Generator) -> list[np.ndarray]:
    """Whole groups packed to ``batch_size``, then the ungrouped fxy rows.

    Every row reaches exactly one batch per epoch.  The ungrouped rows used to be
    re-permuted INSIDE the chunk loop, which resampled them instead of shuffling
    them: on the real corpus 1,000 loose rows produced 1,000 emitted rows of
    which 679 were distinct, 265 appeared twice and 321 never appeared, so each
    stage-2 epoch trained on a random two thirds of the singleton-parent F_xy
    rows.  The lambda = 0 arm would then not have been "v3's loss on the same
    rows" and the §9d two-arm decomposition would have compared two different
    training sets.  One permutation, taken before the loop.
    """
    loose = np.setdiff1d(extra, np.concatenate(groups) if groups else extra[:0])
    order = rng.permutation(len(groups))
    out, cur = [], []
    for j in order:
        if cur and sum(len(c) for c in cur) + len(groups[j]) > batch_size:
            out.append(np.concatenate(cur))
            cur = []
        cur.append(groups[j])
    if cur:
        out.append(np.concatenate(cur))
    shuffled = rng.permutation(loose)
    for a in range(0, len(shuffled), batch_size):
        out.append(shuffled[a:a + batch_size])
    return [b for b in out if len(b)]


def _collate(inner: PolicyStepsV3, rows: np.ndarray,
             device: str) -> dict[str, torch.Tensor]:
    """Stack ``inner[i]`` for ``rows`` — the group batcher's own collate."""
    items = [inner[int(i)] for i in rows]
    return {k: torch.as_tensor(np.stack([it[k] for it in items])).to(device)
            for k in items[0]}


def delta_d_status(splits_path: str) -> dict[str, Any]:
    """Is **prereg delta D** on disk, and is it pointed at the FROZEN assignment?

    Delta D is ``lpopt.policy.metrics_v31`` (prereg §5d / §9a-D): the module that
    refits ``random`` / ``class_freq`` / ``periph`` / ``gd_rule`` inside each
    block's train fold and stitches the out-of-fold score column from the K fits.
    Until it exists there is no way to compute a cross-fit gate number at all,
    which is why :func:`assert_v3_path_untouched` refuses ``--stage2 on``.

    Two conditions, and both are checked rather than asserted, because "the
    module is importable" alone would let a training run be gated on an
    assignment nobody hashed:

    * the module imports (it is optional at import time here so that a v3 run in
      an environment without it is unaffected), and
    * ``--splits`` names a file whose sha256 is
      :data:`~.metrics_v31.SPLITS_V31_SHA256`, the STEP 0-a emission of freeze
      stamp §S0.5.  The realized fold table, the pool's 39 live parents and
      clause 2B's 70% power are all statements about those bytes.
    """
    try:
        from . import metrics_v31 as _m31
    except Exception as exc:                    # pragma: no cover - env specific
        return {"available": False,
                "reason": f"lpopt.policy.metrics_v31 does not import ({exc})"}
    if not splits_path:
        return {"available": False, "module": _m31.__name__,
                "reason": "--splits was not given, so the gate would be "
                          "computed on a re-derived assignment rather than on "
                          "the hashed STEP 0-a emission (freeze stamp §S0.5)"}
    if not _m31.splits_fingerprint_ok(splits_path):
        return {"available": False, "module": _m31.__name__,
                "splits": splits_path,
                "reason": f"{splits_path} is not the registered cross-fit "
                          f"assignment {_m31.SPLITS_V31_SHA256}"}
    return {"available": True, "module": _m31.__name__, "splits": splits_path,
            "splits_sha256": _m31.SPLITS_V31_SHA256}


def assert_v3_path_untouched(args: argparse.Namespace) -> dict[str, Any]:
    """Return the v3.1 flag state and REFUSE a half-enabled run.

    The contract this round is that ``lpopt.policy.train_v3`` still runs v3 bit
    for bit when the v3.1 flags are at their defaults, and that no flag can be
    set in a way that LOOKS like it changed the protocol without changing it.
    Three rules:

    * ``--lam-grid`` away from :data:`LAMBDA_GRID` without ``--stage2 on`` is
      refused.  It would read as a listwise run in the log and be a v3 run.
      The ONE exception is §9d's arm ii, whose registered form is
      ``--no-burnt --lam-grid 0 --stage2 off``: there the zero is the
      declaration that the control carries no listwise term, and only the
      single value ``0`` is admitted.
    * ``--no-burnt`` (§9d arm ii) is refused WITH ``--stage2 on``.  The control
      is a v3 refit on the v3.1 corpus, so its vector is 51 names long and must
      keep the v3 serving contract; stage 2 would stamp
      :data:`POLICY_SCHEMA_V31` on it.  It is stamped ``burnt: off``.
    * ``--xfit-k`` is a standalone EMISSION mode (:func:`emit_crossfit_splits`),
      not a training knob, and says so; it is allowed alone and is stamped —
      but only WITH ``--holdout-cell`` set to :data:`PROSPECTIVE_CELL_V31`.
      ``--holdout-cell`` defaults to v3's cell, and an emission that forgets the
      flag would write ``splits_v31.csv`` on the wrong cell with the gate-pool
      floor disabled (that guard fires only on the registered holdout, because
      on any other frame the number it compares is a different quantity).  The
      ambiguity is refused rather than papered over with a second default.
    * ``--stage2 on`` WITHOUT ``--xfit-k`` is refused outright and stays refused
      for the life of the round.  ``--stage2 on --xfit-k K`` was refused too
      until **prereg delta D** landed; with :func:`delta_d_status` reporting the
      metric module importable AND ``--splits`` hashing to the frozen STEP 0-a
      assignment, the flag stops meaning "emit the split" and means "consume the
      frozen one", and the run is allowed.  Nothing else about the refusal
      moves: without ``--splits``, or against different bytes, or without the
      module, it is the same SystemExit it always was.  WITHOUT ``--xfit-k`` the
      run trains and
      would be gated on :func:`build_splits_v3`'s single alternating split,
      which with the v3.1 holdout realizes 37 parents with >= 8 F_xy candidates
      of which 16 are live -- verbatim the row prereg §3a marks REJECTED and
      says decides nothing -- while stamping the result ``policy_version =
      "v31"``; §3a registers the cross-fit and the cell replacement as ONE
      change, and this shape takes the second without the first.  WITH
      ``--xfit-k`` the emission branch returns 0 before a single weight is
      trained, so ``run.sh`` touches ``DONE`` on a run that produced no model.
      Both are the "looks like a v3.1 run and is a different run" class this
      function exists to refuse, so both are refused rather than ranked.  What
      unblocks them is prereg delta D: §5d refits ``random`` / ``class_freq`` /
      ``periph`` / ``gd_rule`` inside EACH block's train fold, so the
      out-of-fold score column is stitched from K baseline fits, and that
      stitching lives in the metric module, which this track does not own.
      Until then the v3.1 FEATURE path is still built and still checked
      (:func:`featurize_round`), so the contract is ready and only the fold
      arithmetic is missing.
    """
    on = {"stage2": args.stage2 == "on",
          "lam_grid": tuple(args.lam_grid) != LAMBDA_GRID,
          "xfit_k": int(args.xfit_k) > 0,
          "no_burnt": bool(getattr(args, "no_burnt", False))}
    delta_d = delta_d_status(getattr(args, "splits", "") or "")
    if on["no_burnt"] and on["stage2"]:
        raise SystemExit(
            f"--no-burnt is prereg §9d's arm ii and §9d registers it as "
            f"`--no-burnt --lam-grid 0 --stage2 off`: the control is the v3 "
            f"refit on the v3.1 corpus, so stage 2 is OFF by construction.  "
            f"With --stage2 on the run would train the fxy branch on a "
            f"{N_SCALARS_V3}-scalar vector and stamp it {POLICY_SCHEMA_V31!r}, "
            f"which is the mis-stamp featurize_round refuses.")
    if on["lam_grid"] and not on["stage2"]:
        # §9d's arm ii is written `--lam-grid 0 --stage2 off`, and the ZERO is
        # the point: it says in the log that the control carries no listwise
        # term.  That single value is admitted under --no-burnt and nothing
        # else is -- a non-zero grid on a stage-2-off run is still the "logs as
        # a listwise run, is a v3 run" shape this refusal exists for.
        if not (on["no_burnt"] and tuple(args.lam_grid) == (0.0,)):
            raise SystemExit(
                "--lam-grid was set without --stage2 on.  The listwise term "
                "only exists inside stage 2, so this would log as a listwise "
                "run and be a v3 run; it is refused rather than silently "
                "ignored (§9a).  §9d's arm ii is the one exception and it is "
                "exactly `--no-burnt --lam-grid 0 --stage2 off`.")
    if on["stage2"] and not on["xfit_k"]:
        raise SystemExit(
            "--stage2 on without --xfit-k would train on v3's SINGLE "
            "alternating split and stamp the checkpoint v3.1.  With the v3.1 "
            "holdout that split REALIZES 37 parents with >= 8 F_xy candidates "
            "of which 16 are live, against the cross-fit pool's realized "
            "72 / 39 on steps_v31.parquet; it is "
            "the arm prereg §3a registers as REJECTED, and §3a registers the "
            "cross-fit and the cell replacement as ONE change.  Blocked until "
            "the out-of-fold loop lands with prereg delta D (the metric module "
            "refits the four baselines inside each block, §5d).")
    if on["xfit_k"] and args.holdout_cell != PROSPECTIVE_CELL_V31:
        raise SystemExit(
            f"--xfit-k is the v3.1 split emission and §3c registers its held-out "
            f"cell as {PROSPECTIVE_CELL_V31!r}, but --holdout-cell is "
            f"{args.holdout_cell!r}; the flag defaults to v3's cell "
            f"({PROSPECTIVE_CELL!r}), so an emission that forgets it writes a "
            f"splits_v31.csv on "
            f"the WRONG cell AND, because emit_crossfit_splits judges the "
            f"realized live count only on the registered holdout, with the gate "
            f"pool floor silently disabled.  The ambiguity is refused rather "
            f"than resolved by a default: pass --holdout-cell "
            f'"{PROSPECTIVE_CELL_V31}" explicitly so runs/<ts>/run.sh records '
            f"what was actually held out.")
    if on["stage2"] and on["xfit_k"] and not delta_d["available"]:
        raise SystemExit(
            f"--stage2 on --xfit-k {int(args.xfit_k)} needs prereg delta D and "
            f"the FROZEN cross-fit assignment, and this run has neither in "
            f"place: {delta_d['reason']}.  Delta D is "
            f"lpopt/policy/metrics_v31.py (§5d refits the four baselines inside "
            f"each block's train fold and stitches the out-of-fold score column "
            f"from the K fits); the assignment is data/policy/v31_split/"
            f"splits_v31.csv, sha256 recorded in the STEP 0 freeze stamp §S0.5. "
            f"Without --splits the flag is still the split EMISSION step, which "
            f"returns before a single weight is trained -- run.sh would touch "
            f"DONE on a run with no model, which is the shape this function "
            f"exists to refuse.  Pass "
            f"--splits data/policy/v31_split/splits_v31.csv.")
    stamp = {"enabled": on["stage2"], "version": "v31" if on["stage2"] else "v3",
            "lam_grid": list(args.lam_grid), "xfit_k": int(args.xfit_k),
            "teacher": args.teacher, "teacher_temp": args.teacher_temp,
            "teacher_eps": args.teacher_eps,
            # ``--xfit-k`` with a verified ``--splits`` and ``--stage2 on`` is
            # the CONSUMPTION of the frozen assignment; alone it is still the
            # emission.  The two are distinguished here so ``main`` branches on
            # a recorded fact rather than re-deriving the condition.
            "crossfit": bool(on["stage2"] and on["xfit_k"]
                             and delta_d["available"]),
            "delta_d": delta_d}
    # Written ONLY for the arm that turns the columns off, so a default v3 run
    # and arm i keep the stamp they already had byte for byte; the key's
    # presence is itself the declaration that this is the §9d control.
    if on["no_burnt"]:
        stamp["burnt"] = "off"
    return stamp


def load_round(path: str | Path, *, v31: bool) -> pd.DataFrame:
    """The corpus for the round: v3's 107 columns, or v3.1's 111."""
    return load_universe_v31(path) if v31 else load_universe_v3(path)


def featurize_round(steps: pd.DataFrame, *, v31: bool, no_burnt: bool = False
                    ) -> tuple[np.ndarray, list[str], list[str]]:
    """``(scalars, names, new_scalars)`` for the round, with the STAMP enforced.

    The whole point of this function is that it is impossible to reach the
    serving stamp and the feature vector through different doors.  The first cut
    of this track branched on the flag when writing ``policy_schema`` but not
    when featurizing, so ``--stage2 on`` produced a checkpoint stamped
    ``policy_move_v31`` carrying v3's 51 names: the two burnt columns reached no
    model, prereg §4d's "51 -> 53" was realized nowhere runnable, and
    ``MoveScorerV31`` would have rendered 53 names against a 51-name checkpoint
    and refused to serve the round's own ensemble.  Here the vector and the
    stamp come out of one call, and the length is asserted both ways: a v3.1
    round must carry both of :data:`NEW_SCALARS_V31` and be
    :data:`N_SCALARS_V31` long, and a v3 round must carry NEITHER.

    ``no_burnt`` is prereg §9d's **arm ii**, and it is a THIRD declared variant
    rather than a way of sneaking a v3 vector past the v3.1 door.  The control
    the decomposition needs is "v3 refit on the SAME corpus", so the frame is
    the 111-column v3.1 corpus and the vector is v3's 51 — which is exactly the
    combination the ``v31=False`` branch below cannot distinguish from a plain
    v3 run, because ``scalar_features_v3`` never names the burnt columns and so
    never "leaks" them.  Declaring it explicitly is what makes the arm legible:
    the corpus is REQUIRED to carry :data:`REQUIRED_V31_COLUMNS` (a ``--no-burnt``
    run against ``steps_v3.parquet`` is not a control, it is the v3 round again
    under a different name and is refused), the length is asserted to be
    :data:`N_SCALARS_V3`, and ``assert_v3_path_untouched`` stamps ``burnt: off``
    so ``metrics.json`` separates arm ii from arm i and from v3 without anyone
    having to diff ``scalar_names``.  ``--no-burnt`` with ``--stage2 on`` is
    refused there for the same reason the mis-stamp above is: a 51-name vector
    must never carry :data:`POLICY_SCHEMA_V31`.
    """
    if no_burnt:
        if v31:                                  # unreachable from the CLI
            raise SystemExit(
                f"--no-burnt drops the two burnt columns, so the vector is "
                f"{N_SCALARS_V3} names long and cannot be stamped "
                f"{POLICY_SCHEMA_V31!r}; this is the same mis-stamp "
                f"featurize_round exists to refuse.")
        missing = [c for c in REQUIRED_V31_COLUMNS if c not in steps.columns]
        if missing:
            raise SystemExit(
                f"--no-burnt is prereg §9d's arm ii -- the v3 refit on the "
                f"v3.1 CORPUS with {sorted(NEW_SCALARS_V31)} dropped -- but "
                f"this frame does not carry {missing}, so dropping them "
                f"decides nothing and the run would be the v3 round again "
                f"under another name.  Point --steps at the v3.1 corpus "
                f"({STEPS_V31!r}).")
        scalars, names = scalar_features_v3(steps)
        leaked = sorted(set(NEW_SCALARS_V31) & set(names))
        if leaked or len(names) != N_SCALARS_V3:
            raise SystemExit(
                f"the §9d control must be v3's {N_SCALARS_V3}-scalar contract; "
                f"this one has {len(names)}"
                + (f" and still carries {leaked}" if leaked else "") + ".")
        return scalars, names, list(NEW_SCALARS_V3)
    if v31:
        scalars, names = scalar_features_v31(steps)
        new = [*NEW_SCALARS_V3, *NEW_SCALARS_V31]
        missing = sorted(set(NEW_SCALARS_V31) - set(names))
        if missing or len(names) != N_SCALARS_V31:
            raise SystemExit(
                f"a run stamped {POLICY_SCHEMA_V31!r} must carry the v3.1 "
                f"feature contract: {N_SCALARS_V31} scalars including "
                f"{sorted(NEW_SCALARS_V31)}.  This one has {len(names)}"
                + (f" and is missing {missing}" if missing else "")
                + ".  Point --steps at the v3.1 corpus "
                  "(`python mine_policy_corpus.py --v31 --apply`).")
    else:
        scalars, names = scalar_features_v3(steps)
        new = list(NEW_SCALARS_V3)
        leaked = sorted(set(NEW_SCALARS_V31) & set(names))
        if leaked or len(names) != N_SCALARS_V3:
            raise SystemExit(
                f"a run stamped {POLICY_SCHEMA_V3!r} must be v3's "
                f"{N_SCALARS_V3}-scalar contract; this one has {len(names)}"
                + (f" and carries the v3.1 columns {leaked}" if leaked else "")
                + ".  A v3 stamp on a v3.1 vector is the same mis-stamp as the "
                  "converse and is refused for the same reason.")
    return scalars, names, new


def emit_crossfit_splits(steps: pd.DataFrame, out_dir: Path, *, k: int,
                         seed: int, holdout_cell: str | None) -> dict[str, Any]:
    """Write the §3a cross-fit assignment and its census.  Trains nothing.

    This is an EMISSION mode, deliberately, and the reason is a scope statement
    rather than a shortcut.  Cross-fit changes what "the train fold" means: §5d
    fits ``random`` / ``class_freq`` / ``periph`` / ``gd_rule`` on **each block's**
    train fold, so an out-of-fold score column is stitched from K baseline fits
    and the metric module (prereg §9a-D) is where that stitching belongs.  What
    this function owns is the ASSIGNMENT — component-blocked, deterministic, the
    held-out cell in no block, ``val`` in no block — and the census that says
    what it bought.  Emitting it as a file means the assignment the metrics are
    computed on is a hashable artefact and not a re-derivation.

    Returns the census and writes ``splits_v31.csv`` / ``xfit_census.json``.
    """
    splits = build_splits_v31(steps, seed=seed, k=k, holdout_cell=holdout_cell)
    mask = targets_v3(steps)[1][:, 2] > 0
    parents = steps["parent_record_id"].astype(str).to_numpy()
    y = targets_v3(steps)[0][:, 2]

    def census(idx: np.ndarray) -> dict[str, int]:
        keep = idx[mask[idx]]
        counts = pd.Series(parents[keep]).value_counts()
        big = set(counts.index[counts >= REGRET_MIN_CANDIDATES])
        live = {p for p in big if y[keep[parents[keep] == p]].sum() > 0}
        return {"n_rows": int(len(idx)), "n_fxy": int(len(keep)),
                "n_parents_ge8": len(big), "n_live": len(live)}

    out: dict[str, Any] = {
        "k": k, "seed": seed, "holdout_cell": holdout_cell,
        "folds": {name: census(np.flatnonzero(
            (splits["fold"] == name).to_numpy()))
            for name in sorted(set(splits["fold"]))},
        "blocks": [{"block": b, **{part: census(idx)
                                   for part, idx in xfit_indices(splits, b).items()}}
                   for b in range(k)],
        "calib_rows": int(len(calib_index(splits))),
    }
    # The gate pool is the round's only judgement set and every knob that draws
    # rows out of it (``val_frac`` above all) spends §6a clause 2B's power.  With
    # the REGISTERED holdout the realized live count is checked against the
    # registered floor here, where the assignment is written, so a later default
    # change shows up as a refusal rather than as a quietly weaker gate.  On any
    # other holdout this is a re-derivation on a different frame and is not
    # judged.
    live = out["folds"].get("pool", {}).get("n_live", 0)
    if holdout_cell == PROSPECTIVE_CELL_V31 and live < MIN_POOL_LIVE_V31:
        raise SystemExit(
            f"the cross-fit pool realizes {live} live parents against the "
            f"registered floor {MIN_POOL_LIVE_V31} (72 parents >= 8 / 39 live "
            f"on steps_v31.parquet at val_frac 0.05).  Something upstream is "
            f"spending the gate -- the "
            f"val fraction, the holdout, or the corpus -- and clause 2B's power "
            f"is computed on this number, so the assignment is refused rather "
            f"than written.")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = splits.copy()
    frame.insert(0, "child_record_id", steps["child_record_id"].to_numpy())
    frame.to_csv(out_dir / "splits_v31.csv", index=False)
    (out_dir / "xfit_census.json").write_text(json.dumps(out, indent=2,
                                                         sort_keys=True))
    return out


def _predict_logits(model: nn.Module, loader: DataLoader,
                    device: str) -> np.ndarray:
    """``_predict`` without the sigmoid.

    §5c is the reason this exists: v3's probability p90-p10 on the gate fold is
    0.0324 while its LOGIT p90-p10 is 9.66, so the scale question — and the
    Platt map that answers it — is asked on the logit.  Re-deriving the logit by
    inverting a float32 sigmoid loses the tails exactly where clause 4 measures.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            cells = batch["cells"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            out.append(model(cells, cond).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_one(seed: int, *, sets: dict[str, PolicyStepsV3], device: str,
              epochs: int, batch_size: int, lr: float, weight_decay: float,
              patience: int, width: int, n_blocks: int, num_workers: int,
              protocol: str = "revB") -> tuple[PolicyNet, dict[str, Any]]:
    """Train one member.  ``protocol`` exists only to stamp the checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    train_set = sets["train"]
    cfg = PolicyNetConfig(arm="cnn", in_channels=train_set.n_channels,
                          n_cond=train_set.n_cond, width=width, n_blocks=n_blocks,
                          n_heads=len(HEADS_V3))
    model = PolicyNet(cfg).to(device)

    loaders = {
        name: DataLoader(_TorchSteps(s), batch_size=batch_size,
                         shuffle=(name == "train"), drop_last=False,
                         num_workers=num_workers,
                         pin_memory=device.startswith("cuda"),
                         persistent_workers=bool(num_workers))
        for name, s in sets.items()
    }
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in loaders["train"]:
            cells = batch["cells"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            loss = masked_bce_soft(model(cells, cond), batch["y"].to(device),
                                   batch["m"].to(device), batch["w"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()

        model.eval()
        vtot, vw = 0.0, 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                cells = batch["cells"].to(device)
                cond = batch["cond"].to(device)
                m, w = batch["m"].to(device), batch["w"].to(device)
                l = masked_bce_soft(model(cells, cond), batch["y"].to(device), m, w)
                mass = float((m * w).sum())
                vtot += float(l) * mass
                vw += mass
        val_loss = vtot / max(vw, 1e-6)
        # Early stopping is the UNWEIGHTED mean of the three heads' val Spearman:
        # the object is a ranker, and the era weighting that is right for the
        # training objective turns a small current-era slice into half the
        # criterion, which is what froze v2's Run A at epoch 0.
        rho = _val_spearman(_predict(model, loaders["val"], device),
                            sets["val"].labels, sets["val"].mask)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1),
                        "val_loss": val_loss, "val_spearman": rho})
        if rho > best + 1e-7:
            best, best_epoch, stale = rho, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        if epoch % 10 == 0:
            print(f"  [seed {seed}] epoch {epoch:3d} loss={tot / max(nb, 1):.5f} "
                  f"val_loss={val_loss:.5f} val_rho={rho:+.4f} "
                  f"(best {best:+.5f} @ {best_epoch})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    chosen = history[best_epoch] if 0 <= best_epoch < len(history) else {}
    return model, {"arm": "cnn", "seed": seed, "best_epoch": best_epoch,
                   "protocol": protocol, "heads": list(HEADS_V3),
                   "best_val_loss": chosen.get("val_loss", float("nan")),
                   "best_val_spearman": chosen.get("val_spearman", float("nan")),
                   "n_params": count_parameters(model),
                   "net_config": dict(cfg.__dict__), "history": history}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_fold(name: str, frame: pd.DataFrame, probs: np.ndarray,
                  train: pd.DataFrame, v2: pd.DataFrame | None, *,
                  gd_sign: float, rng: np.random.Generator,
                  n_boot: int = N_BOOTSTRAP) -> dict[str, Any]:
    """Every pre-registered number for one fold, all three heads."""
    out: dict[str, Any] = {"fold": name, "n_steps": int(len(frame)),
                           "n_cells": int(frame["cell"].nunique()),
                           "n_current": int(frame["era_current"].sum())}
    parents = frame["parent_record_id"].astype(str).to_numpy()
    scorers = ("policy", *BASELINES)

    for h, head in enumerate(HEADS_V3):
        col = EVAL_LABEL_V3[head]
        keep = frame[col].notna().to_numpy()
        sub = frame[keep]
        y = sub[col].astype(bool).to_numpy().astype(np.float64)
        base = baseline_scores_v3(sub, train, head, v2, gd_sign=gd_sign)
        score = {"policy": probs[keep, h], **base}
        # A baseline with missing values cannot be ranked; NaN sorts last under
        # every ordering here, so it is replaced by the worst finite value rather
        # than being silently favoured.
        for k, v in score.items():
            if np.isnan(v).any():
                finite = v[np.isfinite(v)]
                floor = (float(finite.min()) - 1.0) if finite.size else 0.0
                score[k] = np.where(np.isfinite(v), v, floor)

        entry: dict[str, Any] = {
            "n_labeled": int(len(y)),
            "base_rate": float(y.mean()) if len(y) else float("nan"),
            "auc": {k: auc(score[k], y) for k in scorers},
        }
        if len(y) > 1 and 0 < y.sum() < len(y):
            boot = np.array([auc(score["policy"][i], y[i]) for i in
                             (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
            m, lo, hi = _boot_ci(boot[~np.isnan(boot)])
            entry["auc_ci"] = {"mean": m, "lo": lo, "hi": hi}

        # ---- M5: pooled precision@32 of 256 — REPORTED, NOT GATED (§5e) ---- #
        if len(y) >= BATCH_CANDIDATES:
            draws = np.array([rng.choice(len(y), BATCH_CANDIDATES, replace=False)
                              for _ in range(n_boot)])
            tiebreak = rng.random((n_boot, BATCH_CANDIDATES))
            p = {k: precision_at_k(score[k], y, draws=draws, tiebreak=tiebreak)
                 for k in scorers}
            entry["precision_at_32"] = {
                k: dict(zip(("mean", "lo", "hi"), _boot_ci(p[k]))) for k in scorers}
            entry["precision_delta"] = {}
            for b in BASELINES:
                d_m, d_lo, d_hi = _boot_ci(p["policy"] - p[b])
                entry["precision_delta"][b] = {
                    "mean": d_m, "lo": d_lo, "hi": d_hi, "beats": bool(d_lo > 0.0)}
        else:
            entry["precision_at_32"] = None

        # ---- M1: parent-blocked AUC, paired parent bootstrap --------------- #
        pb, npairs = parent_blocked_auc(score["policy"], y, parents[keep])
        entry["parent_blocked_auc"] = {
            "n_pairs": npairs,
            **{k: parent_blocked_auc(score[k], y, parents[keep])[0] for k in scorers},
        }
        per_parent = _per_parent_blocked(score, y, parents[keep], scorers)
        if per_parent is not None:
            summary, boots = _paired_parent_bootstrap(per_parent)
            entry["parent_blocked_auc_ci"] = summary
            entry["n_mixed_parents"] = int(len(per_parent["policy"]))
            entry["parent_blocked_delta"] = {
                b: _delta(boots["policy"] - boots[b],
                          float(np.nanmean(per_parent["policy"]
                                           - per_parent[b]))) for b in BASELINES}

        # ---- M2 / M4: regret@4-of-8 and NDCG@4 ----------------------------- #
        gain_col = {"fr": "d_f_r", "flat": "d_node_peak", "fxy": "d_f_xy"}[head]
        gain = -pd.to_numeric(frame[gain_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(gain) & frame["both_converged"].fillna(False).to_numpy(bool)
        if ok.sum():
            allp = {"policy": probs[:, h],
                    **baseline_scores_v3(frame, train, head, v2, gd_sign=gd_sign)}
            reg: dict[str, np.ndarray] = {}
            nreg: dict[str, np.ndarray] = {}
            ndcg: dict[str, np.ndarray] = {}
            for k in scorers:
                s = allp[k]
                if not np.isfinite(s).all():
                    finite = s[np.isfinite(s)]
                    floor = (float(finite.min()) - 1.0) if finite.size else 0.0
                    s = np.where(np.isfinite(s), s, floor)
                a, nn_, _keys = regret_at_k(
                    s[ok], gain[ok], parents[ok], k=PROBE_K,
                    min_candidates=REGRET_MIN_CANDIDATES, seed=20260831)
                reg[k], nreg[k] = a, nn_
                ndcg[k] = ndcg_at_k(s[ok], gain[ok], parents[ok])[0]
            if len(reg["policy"]):
                summary, boots = _paired_parent_bootstrap(reg)
                nsummary, _ = _paired_parent_bootstrap(nreg)
                entry["regret_at_4_of_8"] = summary
                entry["regret_at_4_of_8_normalized"] = nsummary
                # LOWER is better, so the improvement is (baseline - policy).
                entry["regret_delta"] = {
                    b: _delta(boots[b] - boots["policy"],
                              float(np.nanmean(reg[b] - reg["policy"])))
                    for b in BASELINES}
                entry["beats_all_baselines_regret"] = bool(
                    all(entry["regret_delta"][b]["beats"] for b in BASELINES))
            if len(ndcg["policy"]):
                nd, ndboots = _paired_parent_bootstrap(ndcg)
                entry["ndcg_at_4"] = nd
                entry["ndcg_delta"] = {
                    b: _delta(ndboots["policy"] - ndboots[b],
                              float(np.nanmean(ndcg["policy"] - ndcg[b])))
                    for b in BASELINES}
            # v2 comparability: the OLD metric on the 6 parents that support it.
            allp8 = {k: allp[k] for k in scorers}
            reg8 = {}
            for k in scorers:
                s = allp8[k]
                s = np.where(np.isfinite(s), s,
                             (np.nanmin(s[np.isfinite(s)]) - 1.0
                              if np.isfinite(s).any() else 0.0))
                reg8[k] = regret_at_k(s[ok], gain[ok], parents[ok], k=8,
                                      min_candidates=10, seed=20260831)[0]
            if len(reg8["policy"]):
                entry["regret_at_8_legacy"] = _paired_parent_bootstrap(reg8)[0]

        entry["calibration"] = calibration(probs[keep, h], y) if len(y) else None
        entry["target_rmse"] = _target_rmse(frame, probs, h)
        out[head] = entry
    return out


def _target_rmse(frame: pd.DataFrame, probs: np.ndarray, h: int) -> float:
    """RMSE against the v3 training target — reported, never gated."""
    y, m = targets_v3(frame)
    keep = m[:, h] > 0
    if not keep.any():
        return float("nan")
    return float(np.sqrt(np.mean((probs[keep, h] - y[keep, h]) ** 2)))


def gate_verdict(results: dict[str, Any]) -> dict[str, Any]:
    """The §6 gate and the §6 transfer bar, computed from the fold results.

    Both clauses must hold for a PASS.  There is no partial credit and no
    post-hoc metric substitution: if one clause passes it is reported as one
    clause passing.
    """
    g = results["folds"].get("gate_cur", {}).get("fxy", {})
    pb = g.get("parent_blocked_auc", {})
    pbci = g.get("parent_blocked_auc_ci", {}).get("policy", {})
    delta = g.get("regret_delta", {})
    gate = {
        "pb_auc": float(pb.get("policy", float("nan"))),
        "pb_auc_ci_lo": float(pbci.get("lo", float("nan"))),
        "n_mixed_parents": g.get("n_mixed_parents"),
        "clause_1_pb_auc": bool(pb.get("policy", 0.0) >= GATE_AUC
                                and pbci.get("lo", 0.0) > GATE_AUC_CI_LO),
        "clause_2_regret_beats_all": bool(
            delta and all(delta.get(b, {}).get("beats") for b in BASELINES)),
        "regret_beats": {b: bool(delta.get(b, {}).get("beats"))
                         for b in BASELINES},
    }
    gate["PASS"] = bool(gate["clause_1_pb_auc"] and gate["clause_2_regret_beats_all"])

    t = results["folds"].get("prospective_cell", {}).get("fxy", {})
    t_pb = t.get("parent_blocked_auc", {}).get("policy", float("nan"))
    t_delta = t.get("regret_delta", {})
    transfer = {
        "pb_auc": float(t_pb),
        "clause_1_pb_auc": bool(t_pb >= TRANSFER_AUC),
        "clause_2_regret": bool(
            t_delta and all(t_delta.get(b, {}).get("beats")
                            for b in ("class_freq", "policy_v2"))),
    }
    transfer["PASS"] = bool(transfer["clause_1_pb_auc"] and transfer["clause_2_regret"])

    fr = results["folds"].get("gate_cur", {}).get("fr", {})
    fr_pb = fr.get("parent_blocked_auc", {}).get("policy", float("nan"))
    return {
        **gate, "transfer_bar": transfer,
        "fr_head_parent_blocked": float(fr_pb),
        "fr_head_regression": bool(
            np.isfinite(fr_pb) and fr_pb < V2_FR_PARENT_BLOCKED - FR_REGRESSION_TOL),
    }


# --------------------------------------------------------------------------- #
# the blind v2 baseline
# --------------------------------------------------------------------------- #
def emit_v2_baseline(steps: pd.DataFrame, cache: PatternCache, out_path: Path,
                     model_dir: Path, device: str = "cpu") -> pd.DataFrame:
    """Score every row with the SHIPPED v2 ensemble and write the CSV.

    v2's OWN feature layout is used — :func:`lpopt.policy.v2.scalar_features_v2`
    and the delta channels named in v2's ``meta.json`` — and the caller must pass
    a cache built with v2's provenance (``data.corpus_provenance``), because the
    shipped checkpoint learned ga80 at ``g_sym_class`` 0.0 and reproducing its
    training scores means feeding it that.  So this is the model that was gated
    in August, not a re-implementation of it.

    The v2 serving stamp is ENFORCED here (prereg §5c): a checkpoint that is not
    ``policy_move_v2`` over ``[ga80, paramA]`` is not the baseline this round
    registered and must not be silently scored as if it were.
    """
    dirs = sorted(d for d in model_dir.glob("cnn_seed*")
                  if (d / "model.pt").is_file())
    if not dirs:
        raise SystemExit(f"no v2 checkpoints under {model_dir}")
    meta0 = json.loads((dirs[0] / "meta.json").read_text())
    stamped = (meta0.get("policy_schema"), str(meta0.get("policy_version", "")),
               tuple(meta0.get("era_libraries") or ()))
    wanted = (POLICY_SCHEMA_V2, "v2", CURRENT_ERA_LIBRARIES)
    if stamped != wanted:
        raise SystemExit(f"{model_dir} is stamped {stamped!r}, not {wanted!r}; "
                         f"this is not the registered v2 baseline")

    index = {c: i for i, c in enumerate(cache.channels)}
    delta = [index[c] for c in meta0["delta_channels"]]
    scalars, names = scalar_features_v2(steps)
    if names != meta0["scalar_names"]:
        raise SystemExit("v2 scalar layout drifted; v2 cannot be scored as a "
                         "baseline")

    data = PolicySteps(steps, cache, scalars, delta_channels=delta, augment=False)
    loader = DataLoader(_TorchSteps(data), batch_size=256, shuffle=False)
    total = np.zeros((len(steps), 2))
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        net = PolicyNet(PolicyNetConfig(**meta["net_config"]))
        net.load_state_dict(torch.load(d / "model.pt", map_location="cpu",
                                       weights_only=True))
        net.eval().to(device)
        total += _predict(net, loader, device)
    total /= len(dirs)

    frame = pd.DataFrame({
        "parent_record_id": steps["parent_record_id"].to_numpy(),
        "child_record_id": steps["child_record_id"].to_numpy(),
        "cell": steps["cell"].to_numpy(),
        "p_improve_fr": total[:, 0], "p_improve_flat": total[:, 1],
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"[v2-baseline] {len(frame)} rows from {len(dirs)} members "
          f"-> {out_path}", flush=True)
    return frame


# --------------------------------------------------------------------------- #
# the cross-fit round (§3a) — one stage-1 + stage-2 fit per (block, seed)
# --------------------------------------------------------------------------- #
def train_crossfit_v31(steps: pd.DataFrame, splits: pd.DataFrame, *,
                       args: argparse.Namespace, cache: PatternCache,
                       delta: Sequence[int], scalars: np.ndarray,
                       device: str, v2: pd.DataFrame | None, out_dir: Path,
                       ) -> dict[str, Any]:
    """Train the K x seeds ensemble and stitch the out-of-fold ``fxy`` logits.

    The shape §3a registers, and the two things that make it a cross-fit rather
    than K unrelated runs:

    * a row's OUT-OF-FOLD logit is produced by the members of the block that
      never trained on it — ``xfit_indices`` decides that, and the assignment is
      the hashed STEP 0-a file, not a re-derivation;
    * ``lambda`` is selected ONCE for the whole round on ``val`` mean Spearman
      alone (§2c / §9d), pooled over blocks and seeds.  ``val`` is the same
      component set in every block and is in no block's eval fold, so selection
      and judgement never read the same row (§3d).

    The held-out cell is scored by the FULL ensemble, which is legitimate for the
    same reason the stitch is: the cell is in no block, so every member is out of
    fold for it.

    ``weights_v3`` is recomputed per block, because ``w_era`` is defined against
    "the training fold" and under cross-fit that fold is block-dependent; using
    the legacy-only ``train`` label would put 13x the era weight on rows the
    block actually trains on.
    """
    lam_grid = [float(x) for x in args.lam_grid]
    fold_col = splits["fold"].to_numpy()
    blocks = sorted({int(b) for b in splits.loc[splits["fold"] == "pool",
                                                "xfit_fold"]})
    n_rows = len(steps)
    oof = np.full(n_rows, np.nan)
    hold = np.flatnonzero(fold_col == "prospective_cell")
    hold_stack: list[np.ndarray] = []
    val_scores: dict[float, dict[str, float]] = {lam: {} for lam in lam_grid}
    per_block: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []

    def make(idx: np.ndarray, weights: np.ndarray, augment: bool,
             seed: int) -> PolicyStepsV3:
        return PolicyStepsV3(steps.iloc[idx], cache, scalars[idx], weights[idx],
                             delta_channels=list(delta), augment=augment,
                             seed=seed)

    for b in blocks:
        idx = xfit_indices(splits, b)
        block_fold = pd.Series(
            np.where(np.isin(np.arange(n_rows), idx["train"]), "train",
                     np.where(np.isin(np.arange(n_rows), idx["val"]), "val",
                              "gate_cur")), index=steps.index)
        weights = weights_v3(steps, block_fold)
        trained: list[tuple[int, nn.Module, dict[str, Any],
                            dict[str, PolicyStepsV3]]] = []
        for k in range(args.seeds):
            seed = args.base_seed + k
            sets = {"train": make(idx["train"], weights, True, seed),
                    "val": make(idx["val"], weights, False, seed)}
            model, meta = train_one(
                seed, sets=sets, device=device, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, patience=args.patience,
                width=args.width, n_blocks=args.n_blocks,
                num_workers=args.num_workers, protocol=args.protocol)
            trained.append((seed, model, meta, sets))
            print(f"  [block {b} seed {seed}] stage1 "
                  f"val_rho={meta['best_val_spearman']:+.4f}", flush=True)

        grid: dict[float, dict[int, tuple[nn.Module, dict[str, Any]]]] = {}
        for lam in lam_grid:
            for seed, model, _m, sets in trained:
                cand, s2 = train_stage2(
                    model, sets=sets,
                    frames={"train": steps.iloc[idx["train"]]}, device=device,
                    lam=lam, epochs=args.stage2_epochs, lr=args.stage2_lr,
                    patience=args.stage2_patience, batch_size=args.batch_size,
                    teacher=args.teacher, temp=args.teacher_temp,
                    eps=args.teacher_eps, seed=seed)
                grid.setdefault(lam, {})[seed] = (cand, s2)
                val_scores[lam][f"b{b}_s{seed}"] = float(
                    s2["best_val_spearman"])
                print(f"  [block {b} seed {seed}] stage2 lam={lam} "
                      f"val_rho={s2['best_val_spearman']:+.4f}", flush=True)
        per_block.append({"block": int(b), "n_train": int(len(idx["train"])),
                          "n_eval": int(len(idx["eval"])),
                          "grid": {str(lam): {str(s): v[1]["best_val_spearman"]
                                              for s, v in d.items()}
                                   for lam, d in grid.items()}})
        per_block[-1]["_grid"] = grid
        per_block[-1]["_idx"] = idx

    # ---- §2c / §9d: ONE lambda, from val alone, for the whole round -------- #
    lam_selected = select_stage2_lambda(val_scores)
    print(f"=== stage2 selected lam={lam_selected} on val mean Spearman ALONE "
          f"over {len(blocks)} blocks x {args.seeds} seeds (§2c) ===", flush=True)

    for entry in per_block:
        b, grid, idx = entry["block"], entry.pop("_grid"), entry.pop("_idx")
        eval_stack, hold_block = [], []
        for seed in sorted(grid[lam_selected]):
            served = grid[lam_selected][seed][0]
            member_dir = out_dir / f"cnn_block{b}_seed{seed}"
            member_dir.mkdir(parents=True, exist_ok=True)
            torch.save(served.base.state_dict(), member_dir / "model.pt")
            torch.save(served.branch.state_dict(), member_dir / "fxy_branch.pt")
            meta = grid[lam_selected][seed][1]
            (member_dir / "meta.json").write_text(json.dumps(
                {k: v for k, v in meta.items() if k != "history"}
                | {"block": int(b), "seed": int(seed),
                   "policy_schema": POLICY_SCHEMA_V31, "policy_version": "v31",
                   "stage2_lam_selected": float(lam_selected)},
                indent=2, sort_keys=True))
            members.append({"block": int(b), "seed": int(seed),
                            "lam": float(lam_selected),
                            "val_spearman": float(meta["best_val_spearman"])})
            weights = np.ones((len(steps), len(HEADS_V3)), np.float32)
            for name, rows, sink in (("eval", idx["eval"], eval_stack),
                                     ("hold", hold, hold_block)):
                if not len(rows):
                    continue
                loader = DataLoader(
                    _TorchSteps(make(rows, weights, False, seed)),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
                sink.append(_predict_logits(served, loader, device)[
                    :, HEADS_V3.index("fxy")])
                _ = name
        if eval_stack:
            oof[idx["eval"]] = np.mean(np.stack(eval_stack), axis=0)
        if hold_block:
            hold_stack.append(np.mean(np.stack(hold_block), axis=0))

    if hold_stack and len(hold):
        oof[hold] = np.mean(np.stack(hold_stack), axis=0)
    np.savez_compressed(out_dir / "logits_oof.npz", fxy=oof,
                        fold=fold_col.astype(str))
    return {"lam_selected": float(lam_selected),
            "val_spearman": {str(lam): d for lam, d in val_scores.items()},
            "blocks": per_block, "members": members, "logits_fxy": oof}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m lpopt.policy.train_v3")
    ap.add_argument("--steps", default=STEPS_V3)
    ap.add_argument("--fuel-types", default="data/store/fuel_types.parquet")
    ap.add_argument("--cache", default="data/policy/_feature_cache_v3.npz")
    ap.add_argument("--out-dir", default="runs/policy_v3")
    ap.add_argument("--v2-baseline", default="data/design/policy_v3_v2_baseline.csv")
    ap.add_argument("--v2-model-dir", default="data/models/policy_v2")
    ap.add_argument("--emit-v2-baseline", default="")
    ap.add_argument("--holdout-cell", default=PROSPECTIVE_CELL)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=20260831)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--protocol", default="revB", choices=("revB",),
                    help="the loss carried over from v2's declared deviation; "
                         "the pre-registered Huber is NOT offered here because "
                         "it is a measured failure (v2 results §2a)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--device", default="auto")
    # ---- v3.1 (policy_v31_prereg_20260831_DRAFT.md).  ALL DEFAULT OFF: with
    # these untouched this module runs the v3 protocol bit for bit, which
    # ``assert_v3_path_untouched`` enforces rather than merely claims. -------- #
    ap.add_argument("--stage2", default="off", choices=("off", "on"),
                    help="train the frozen-trunk fxy branch of prereg v3.1 §2b "
                         "after stage 1; off runs v3 unchanged")
    ap.add_argument("--lam-grid", type=lambda s: tuple(
        float(x) for x in str(s).split(",") if x != ""), default=LAMBDA_GRID,
        help="listwise weights; selected on val mean Spearman ALONE (§2c)")
    ap.add_argument("--no-burnt", action="store_true",
                    help="prereg §9d arm ii: refit v3 on the v3.1 CORPUS with "
                         f"the two burnt columns {list(NEW_SCALARS_V31)} "
                         "dropped (53 -> 51).  Registered as `--no-burnt "
                         "--lam-grid 0 --stage2 off`; the run stamps "
                         "`burnt: off` and keeps the v3 serving contract")
    ap.add_argument("--stage2-lr", type=float, default=STAGE2_LR)
    ap.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--stage2-patience", type=int, default=STAGE2_PATIENCE)
    ap.add_argument("--teacher", default="raw", choices=("raw",),
                    help="the registered teacher gain; the registered-gain "
                         "teacher is a measured failure (§2b) and is not offered")
    ap.add_argument("--teacher-temp", type=float, default=TEACHER_TEMP)
    ap.add_argument("--teacher-eps", type=float, default=TEACHER_EPS)
    ap.add_argument("--xfit-k", type=int, default=0,
                    help="K for the component-blocked cross-fit of §3a; 0 keeps "
                         "v3's single alternating split")
    ap.add_argument("--splits", default="",
                    help="the FROZEN splits_v31.csv of STEP 0-a.  With "
                         "--stage2 on --xfit-k it turns the emission flag into "
                         "the consumption of a hashed assignment; the sha256 "
                         "must be the one the freeze stamp §S0.5 registered")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--tables":
        print(render_tables(json.loads(Path(argv[1]).read_text())))
        return 0

    args = _parser().parse_args(argv)
    # Before anything is read or written: either every v3.1 flag is off and this
    # is a v3 run, or stage 2 is on and the run says so (prereg §9a).
    v31 = assert_v3_path_untouched(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device} torch={torch.__version__}", flush=True)

    holdout = args.holdout_cell or None
    # arm ii reads the v3.1 corpus too -- that is what makes it a control and
    # not the v3 round again -- so the column check runs for it as well.
    steps = load_round(args.steps, v31=v31["enabled"] or args.no_burnt)
    fold = build_splits_v3(steps, seed=args.base_seed, holdout_cell=holdout)
    summary = split_summary_v3(steps, fold)
    print("=== splits ===\n" + summary.to_string(index=False), flush=True)

    if v31["xfit_k"] > 0 and not v31["crossfit"]:
        census = emit_crossfit_splits(
            steps, Path(args.out_dir), k=v31["xfit_k"], seed=args.base_seed,
            holdout_cell=holdout)
        print("=== cross-fit splits ===\n"
              + json.dumps(census, indent=2), flush=True)
        if not args.emit_v2_baseline:
            print(f"wrote {Path(args.out_dir) / 'splits_v31.csv'}", flush=True)
            return 0

    if args.emit_v2_baseline:
        # Only the EVALUATION folds need a v2 score, and featurizing the whole
        # corpus to produce a few thousand numbers would cost an hour of encoder
        # calls for nothing.  The cache is built over the subset and thrown away
        # — and it is built with v2's OWN provenance, because the point is to
        # reproduce the shipped checkpoint's numbers.
        emit_fold = fold
        if v31["xfit_k"] > 0:
            # ---- DECLARED DEVIATION (prereg v3.1 §3b) -------------------- #
            # Cross-fit scores the WHOLE current era out of fold, so the 3,286
            # rows the shipped `policy_v3_v2_baseline.csv` covers are no longer
            # the evaluation set and the blind baseline has to be re-emitted
            # over the pool.  It does not weaken the blind: the v2 weights were
            # frozen before v3 existed and this scoring is deterministic, so
            # re-running it cannot see a v3.1 number.  The discipline that makes
            # that checkable is ORDER — this file is written, and its sha256
            # recorded, while no v3.1 weight exists — and a reviewer may still
            # call it a weakened blind.  Both halves are registered.
            emit_fold = build_splits_v31(
                steps, seed=args.base_seed, k=v31["xfit_k"],
                holdout_cell=holdout)["fold"].replace({"pool": "gate_cur"})
            print("=== re-emitting the blind v2 baseline over the cross-fit "
                  "pool (§3b, a DECLARED deviation) ===", flush=True)
        sub = steps[emit_fold.isin(("gate_cur", "val", "prospective_cell"))
                    ].reset_index(drop=True)
        print(f"=== blind v2 baseline over {len(sub)} eval rows, "
              f"{sub['cell'].nunique()} cells ===", flush=True)
        small = build_pattern_cache(sub, fuel_types=args.fuel_types)
        emit_v2_baseline(sub, small, Path(args.emit_v2_baseline),
                         Path(args.v2_model_dir), device="cpu")
        return 0

    cache_path = Path(args.cache)
    if cache_path.is_file():
        print(f"=== loading feature cache {cache_path} ===", flush=True)
        cache = PatternCache.load(cache_path)
    else:
        print("=== building feature cache (once, TRUE provenance) ===", flush=True)
        cache = build_pattern_cache_v3(steps, fuel_types=args.fuel_types)
        cache.save(cache_path)

    v2 = (pd.read_csv(args.v2_baseline) if Path(args.v2_baseline).is_file()
          else None)
    if v2 is None:
        print(f"WARNING: {args.v2_baseline} missing; the policy_v2 baseline "
              f"column will be NaN and its comparison void", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_frame = steps[fold == "train"]
    delta = pick_delta_channels(train_frame, cache, seed=args.base_seed)
    print(f"=== delta channels: {len(delta)} of {cache.slots.shape[1]} ===",
          flush=True)

    scalars, scalar_names, new_scalars = featurize_round(
        steps, v31=v31["enabled"], no_burnt=args.no_burnt)

    # ---- the v3.1 cross-fit round (§3a), on the FROZEN assignment ---------- #
    if v31["crossfit"]:
        from . import metrics_v31 as m31
        splits = m31.load_splits(args.splits)
        m31.assert_splits_align(steps, splits)
        legacy = np.flatnonzero((splits["fold"] == "train").to_numpy())
        delta_x = pick_delta_channels(steps.iloc[legacy], cache,
                                      seed=args.base_seed)
        print(f"=== cross-fit: splits {args.splits} "
              f"sha256 {v31['delta_d']['splits_sha256']}; "
              f"delta channels {len(delta_x)} picked on the legacy train fold "
              f"({len(legacy)} rows, in EVERY block) ===", flush=True)
        xf = train_crossfit_v31(steps, splits, args=args, cache=cache,
                                delta=delta_x, scalars=scalars, device=device,
                                v2=v2, out_dir=out_dir)
        report = m31.gate_report_v31(
            steps, splits, xf["logits_fxy"], v2=v2, gate_auc=GATE_AUC,
            gate_auc_ci_lo=GATE_AUC_CI_LO, transfer_auc=TRANSFER_AUC,
            seed=args.base_seed)
        manifest = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "corpus_sha256": corpus_fingerprint(args.steps),
            "splits_sha256": v31["delta_d"]["splits_sha256"],
            "cond_schema": COND_SCHEMA, "move_classes": list(MOVE_CLASSES),
            "heads": list(HEADS_V3), "scalar_names": scalar_names,
            "new_scalars": new_scalars, "target_clip": TARGET_CLIP_V3,
            "probe_k": PROBE_K, "ndcg_k": NDCG_K,
            "regret_min_candidates": REGRET_MIN_CANDIDATES,
            "holdout_cell": holdout, "protocol": args.protocol,
            "delta_channels": [cache.channels[i] for i in delta_x],
            "gate_thresholds": {"auc": GATE_AUC, "auc_ci_lo": GATE_AUC_CI_LO,
                                "transfer_auc": TRANSFER_AUC,
                                "ni_margin": m31.NI_MARGIN,
                                "cell_min_live": m31.CELL_MIN_LIVE,
                                "serving_spread_min": m31.SERVING_SPREAD_MIN},
            "v31": v31, "stage2_lam_selected": xf["lam_selected"],
            "stage2_val_spearman": xf["val_spearman"],
            "blocks": xf["blocks"], "members": xf["members"],
            "gate": report, "args": vars(args), "device": device,
            "torch": torch.__version__, "wall_seconds": time.time() - t0,
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=float))
        print("=== gate (prereg delta D) ===\n" + m31.render_gate(report),
              flush=True)
        print(f"wrote {out_dir / 'metrics.json'} in {time.time() - t0:.1f}s",
              flush=True)
        return 0

    weights = weights_v3(steps, fold)
    gd_sign = gd_rule_sign(train_frame)
    cur = steps["era_current"].to_numpy(bool)
    print(f"=== scalars: {len(scalar_names)} (v3 adds {list(NEW_SCALARS_V3)}"
          + (f", v3.1 adds {list(NEW_SCALARS_V31)}" if v31["enabled"] else "")
          + (f"; §9d arm ii: {list(NEW_SCALARS_V31)} DROPPED"
             if args.no_burnt else "")
          + "); "
          f"current-era loss weight {weights[cur, 0][0]:.2f}; "
          f"gd_rule sign {gd_sign:+.0f} ===", flush=True)
    folds = {name: steps.index[fold == name].to_numpy() for name in fold.unique()}

    def make(name: str, augment: bool, seed: int) -> PolicyStepsV3:
        idx = folds[name]
        return PolicyStepsV3(steps.loc[idx], cache, scalars[idx], weights[idx],
                             delta_channels=delta, augment=augment, seed=seed)

    eval_folds = [f for f in ("gate_cur", "prospective_cell", "val")
                  if f in folds]
    manifest: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_sha256": corpus_fingerprint(args.steps),
        "cond_schema": COND_SCHEMA, "move_classes": list(MOVE_CLASSES),
        "heads": list(HEADS_V3),
        "scalar_names": scalar_names, "new_scalars": new_scalars,
        "target_clip": TARGET_CLIP_V3,
        "probe_k": PROBE_K, "regret_min_candidates": REGRET_MIN_CANDIDATES,
        "ndcg_k": NDCG_K, "gd_rule_sign": float(gd_sign),
        "holdout_cell": holdout, "protocol": args.protocol,
        "provenance": provenance_v3.__name__,
        "delta_channels": [cache.channels[i] for i in delta],
        "split_summary": summary.to_dict("records"),
        "current_era_loss_weight": float(weights[cur, 0][0]),
        "gate_thresholds": {"auc": GATE_AUC, "auc_ci_lo": GATE_AUC_CI_LO,
                            "transfer_auc": TRANSFER_AUC},
        "v31": v31,
        "args": vars(args), "device": device, "torch": torch.__version__,
    }

    members: list[dict[str, Any]] = []
    per_fold: dict[str, list[np.ndarray]] = {n: [] for n in eval_folds}
    # Stage 1 for every seed FIRST.  Nothing about a member's stage-1 weights
    # depends on lambda, and the selection below has to see every seed's score
    # before it can pick one lambda for the whole ensemble (§9d), so the two
    # phases are separated rather than nested.  ``train_one`` and ``make`` are
    # both seeded, so a member's stage-1 result is exactly what the single-loop
    # version produced.
    trained: list[tuple[int, nn.Module, dict[str, Any], dict[str, Any]]] = []
    for k in range(args.seeds):
        seed = args.base_seed + k
        sets = {"train": make("train", True, seed), "val": make("val", False, seed)}
        model, meta = train_one(
            seed, sets=sets, device=device, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            width=args.width, n_blocks=args.n_blocks,
            num_workers=args.num_workers, protocol=args.protocol)
        trained.append((seed, model, meta, sets))

    # Stage 2, the whole grid on every seed, then ONE lambda for the ensemble.
    grid: dict[float, dict[int, tuple[nn.Module, dict[str, Any]]]] = {}
    lam_selected: float | None = None
    if v31["enabled"]:
        for lam in args.lam_grid:
            for seed, model, _meta, sets in trained:
                cand, s2 = train_stage2(
                    model, sets=sets,
                    frames={"train": steps.loc[folds["train"]]}, device=device,
                    lam=float(lam), epochs=args.stage2_epochs,
                    lr=args.stage2_lr, patience=args.stage2_patience,
                    batch_size=args.batch_size, teacher=args.teacher,
                    temp=args.teacher_temp, eps=args.teacher_eps, seed=seed)
                grid.setdefault(float(lam), {})[seed] = (cand, s2)
                print(f"  [seed {seed}] stage2 lam={lam} "
                      f"val_rho={s2['best_val_spearman']:+.4f}", flush=True)
        # lambda is chosen on ``val`` mean Spearman ALONE (§2c): the gate pool,
        # the cross-fit blocks and the prospective cell are not read.
        lam_selected = select_stage2_lambda(
            {lam: {seed: v[1]["best_val_spearman"] for seed, v in d.items()}
             for lam, d in grid.items()})
        manifest["stage2_lam_selected"] = lam_selected
        manifest["stage2_val_spearman"] = {
            str(lam): {str(seed): v[1]["best_val_spearman"]
                       for seed, v in d.items()} for lam, d in grid.items()}
        print(f"=== stage2 selected lam={lam_selected} on val, for EVERY member "
              f"(§9d) ===", flush=True)

    for seed, model, meta, _sets in trained:
        served: nn.Module = model
        if v31["enabled"]:
            served, s2 = grid[float(lam_selected)][seed]
            meta = {**meta, "stage2": {k: v for k, v in s2.items()
                                       if k != "history"},
                    "stage2_lam_selected": float(lam_selected),
                    "stage2_val_spearman": {
                        str(lam): d[seed][1]["best_val_spearman"]
                        for lam, d in grid.items()}}
        member_dir = out_dir / f"cnn_seed{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), member_dir / "model.pt")
        if v31["enabled"]:
            # The stage-1 checkpoint above is the trunk and the fr/flat head and
            # is saved unchanged; the branch is a SEPARATE file, so a v3.1 member
            # is provably "v3 plus a branch" on disk and not a re-fit of v3.
            torch.save(served.branch.state_dict(), member_dir / "fxy_branch.pt")
        (member_dir / "meta.json").write_text(json.dumps(
            {**meta, "cond_schema": COND_SCHEMA,
             "delta_channels": [cache.channels[i] for i in delta],
             "scalar_names": scalar_names,
             "policy_version": v31["version"],
             "protocol": args.protocol,
             # The serving stamp the loader refuses to load without.
             "policy_schema": (POLICY_SCHEMA_V31 if v31["enabled"]
                               else POLICY_SCHEMA_V3),
             "era_libraries": list(CURRENT_ERA_LIBRARIES),
             "corpus_sha256": manifest["corpus_sha256"],
             "target_clip": TARGET_CLIP_V3}, indent=2, sort_keys=True))
        members.append({k2: v for k2, v in meta.items() if k2 != "history"})
        print(f"  [seed {seed}] params={meta['n_params']:,} "
              f"val_rho={meta['best_val_spearman']:+.4f} @{meta['best_epoch']}",
              flush=True)
        for name in eval_folds:
            loader = DataLoader(_TorchSteps(make(name, False, seed)),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
            per_fold[name].append(_predict(served, loader, device))

    print("=== evaluating ===", flush=True)
    rng = np.random.default_rng(args.base_seed)
    results: dict[str, Any] = {"members": members, "folds": {}}
    for name in eval_folds:
        ens = np.mean(np.stack(per_fold[name]), axis=0)
        results["folds"][name] = evaluate_fold(
            name, steps.loc[folds[name]], ens, train_frame, v2,
            gd_sign=gd_sign, rng=rng, n_boot=args.n_bootstrap)
    np.savez_compressed(out_dir / "probs.npz",
                        **{n: np.stack(per_fold[n]) for n in eval_folds})

    manifest["gate"] = gate_verdict(results)
    manifest["results"] = results
    manifest["wall_seconds"] = time.time() - t0
    (out_dir / "metrics.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=float))
    print("=== gate ===\n" + json.dumps(manifest["gate"], indent=2), flush=True)
    print(f"wrote {out_dir / 'metrics.json'} in {time.time() - t0:.1f}s",
          flush=True)
    return 0


# --------------------------------------------------------------------------- #
# report tables (rendered from metrics.json so the report cannot drift from it)
# --------------------------------------------------------------------------- #
def render_tables(metrics: dict[str, Any]) -> str:
    scorers = ("policy", *BASELINES)
    out: list[str] = []
    res = metrics["results"]
    rhos = [m.get("best_val_spearman", float("nan")) for m in res["members"]]
    out.append(f"\n{len(res['members'])} seeds, "
               f"{res['members'][0]['n_params']:,} params, 3 heads "
               f"{metrics.get('heads')}, protocol "
               f"`{res['members'][0].get('protocol', 'revB')}`, val Spearman "
               f"{np.nanmean(rhos):+.4f} (min {np.nanmin(rhos):+.4f}, max "
               f"{np.nanmax(rhos):+.4f}), stop epochs "
               f"{[m['best_epoch'] for m in res['members']]}\n")

    out.append("\n**AUC on `improved_*`** (5-seed ensemble)\n")
    out.append("| fold | head | n | base | " + " | ".join(scorers)
               + " | policy 95% CI |")
    out.append("|" + "---|" * (5 + len(scorers)))
    for f, fr in res["folds"].items():
        for head in metrics["heads"]:
            e = fr[head]
            ci = e.get("auc_ci")
            cis = f"[{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else "n/a"
            out.append(f"| {f} | {head} | {e['n_labeled']} | {e['base_rate']:.3f} | "
                       + " | ".join(f"{e['auc'][s]:.3f}" for s in scorers)
                       + f" | {cis} |")

    out.append("\n**M1 parent-blocked AUC** — moves ranked WITHIN a parent "
               "(paired parent bootstrap)\n")
    out.append("| fold | head | n_pairs | " + " | ".join(scorers)
               + " | beats " + " | beats ".join(BASELINES) + " |")
    out.append("|" + "---|" * (3 + len(scorers) + len(BASELINES)))
    for f, fr in res["folds"].items():
        for head in metrics["heads"]:
            e = fr[head]
            pb = e["parent_blocked_auc"]
            d = e.get("parent_blocked_delta", {})
            beats = " | ".join(
                (("**yes**" if d[b]["beats"] else "no")
                 + f" ({d[b]['mean']:+.3f} [{d[b]['lo']:+.3f}, {d[b]['hi']:+.3f}])")
                if b in d else "n/a" for b in BASELINES)
            out.append(f"| {f} | {head} | {pb['n_pairs']} | "
                       + " | ".join(f"{pb[s]:.3f}" for s in scorers)
                       + f" | {beats} |")

    out.append("\n**M2 regret@4-of-8** — of 4 moves proposed off a parent with "
               "8 labelled candidates, how much reachable gain is missed "
               "(LOWER is better)\n")
    out.append("| fold | head | n_parents | " + " | ".join(scorers)
               + " | beats " + " | beats ".join(BASELINES) + " |")
    out.append("|" + "---|" * (3 + len(scorers) + len(BASELINES)))
    for f, fr in res["folds"].items():
        for head in metrics["heads"]:
            e = fr[head]
            r = e.get("regret_at_4_of_8")
            if not r:
                continue
            d = e["regret_delta"]
            out.append(f"| {f} | {head} | {r['policy']['n_parents']} | "
                       + " | ".join(f"{r[s]['mean']:.4f}" for s in scorers) + " | "
                       + " | ".join(
                           ("**yes**" if d[b]["beats"] else "no")
                           + f" ({d[b]['mean']:+.4f} [{d[b]['lo']:+.4f}, "
                             f"{d[b]['hi']:+.4f}])" for b in BASELINES) + " |")

    out.append("\n**M4 within-parent NDCG@4** (reported)\n")
    out.append("| fold | head | n_parents | " + " | ".join(scorers) + " |")
    out.append("|" + "---|" * (3 + len(scorers)))
    for f, fr in res["folds"].items():
        for head in metrics["heads"]:
            n = fr[head].get("ndcg_at_4")
            if n:
                out.append(f"| {f} | {head} | {n['policy']['n_parents']} | "
                           + " | ".join(f"{n[s]['mean']:.3f}" for s in scorers)
                           + " |")

    out.append("\n**M5 precision@32 of 256** — REPORTED, NOT GATED (prereg §5e)\n")
    out.append("| fold | head | " + " | ".join(scorers) + " |")
    out.append("|" + "---|" * (2 + len(scorers)))
    for f, fr in res["folds"].items():
        for head in metrics["heads"]:
            p = fr[head].get("precision_at_32")
            if p:
                out.append(f"| {f} | {head} | "
                           + " | ".join(f"{p[s]['mean']:.3f}" for s in scorers)
                           + " |")

    out.append("\n**M6 calibration and target fit** (reported, not gated)\n")
    out.append("| fold | head | Brier | ECE | target RMSE |")
    out.append("|---|---|---|---|---|")
    for f, fr in res["folds"].items():
        for head in metrics["heads"]:
            c = fr[head].get("calibration")
            if c:
                out.append(f"| {f} | {head} | {c['brier']:.4f} | {c['ece']:.4f} | "
                           f"{fr[head]['target_rmse']:.4f} |")

    out.append("\n**Gate** (§6)\n```json\n"
               + json.dumps(metrics.get("gate", {}), indent=2) + "\n```")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
