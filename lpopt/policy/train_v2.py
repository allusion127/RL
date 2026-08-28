"""Train and judge the **v2** move-proposal policy.

    python -m lpopt.policy.train_v2 --out-dir runs/policy_v2 --seeds 5

The protocol is fixed by ``data/reports/policy_v2_prereg_20260817.md``.  Three
things differ from v1 and this module is the only place they are written down:

* the training target is the normalized clipped EXPECTED IMPROVEMENT
  (``lpopt.policy.v2.targets``) under a Huber loss, not the improving fraction;
* training spans both eras with the era as an input and the current era
  reweighted to half the loss mass, while the GATE is a held-out current-era
  fold and nothing else;
* the baseline set gains **policy v1**, and the metric set gains **regret@8** —
  the deployment-shaped number, because the consumer (``autoeng``'s PROBE stage)
  spends exactly 8 MASTER calls and keeps its best board.

``--emit-v1-baseline PATH`` runs the v1 ensemble over the corpus and writes its
probabilities to a CSV.  It is run BEFORE any v2 weight exists, so the baseline
is blind, exactly as the ablation wave's registered prospective test was.
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
from torch.utils.data import DataLoader, Dataset

from .data import (
    COND_SCHEMA, HEADS, MOVE_CLASSES, PatternCache, PolicySteps,
    build_pattern_cache, corpus_fingerprint, pick_delta_channels,
    scalar_features,
)
from .net import ARMS, PolicyNet, PolicyNetConfig, count_parameters
from .train import (
    BATCH_CANDIDATES, N_BOOTSTRAP, TOP_K, auc, calibration, parent_blocked_auc,
    precision_at_k, _boot_ci,
)
from .v2 import (
    EVAL_LABEL, NEW_SCALARS, PolicyStepsV2, TARGET_CLIP, build_splits_v2,
    era_weights, load_universe_v2, scalar_features_v2, split_summary_v2,
    targets,
)

#: Probe size of ``autoeng``'s PROBE stage (``Target.probe_budget = 8``).  The
#: deployment metric is shaped to it deliberately.
PROBE_K = 8
#: A parent needs strictly more candidates than the probe spends, or every
#: scorer trivially achieves zero regret.  10 keeps 9 gate parents.
REGRET_MIN_CANDIDATES = 10
#: Huber transition point on the [0, 1] normalized-gain scale.  Residuals below
#: it are quadratic; the few rows at the clip are linear.
HUBER_DELTA = 0.2

BASELINES: tuple[str, ...] = ("random", "class_freq", "periph", "policy_v1")


# --------------------------------------------------------------------------- #
# the new metric
# --------------------------------------------------------------------------- #
def regret_at_k(scores: np.ndarray, gain: np.ndarray, parents: np.ndarray, *,
                k: int = PROBE_K, min_candidates: int = REGRET_MIN_CANDIDATES,
                reps: int = 256, seed: int = 20260817
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-parent regret of a top-``k`` selection.  Lower is better.

    ``gain`` is ``-d_FOM`` (positive = improvement), so the oracle takes its max.
    For each eligible parent: rank the candidates by ``scores``, take the top
    ``k``, and report ``max(gain over all) - max(gain over the k picked)`` — how
    much of the reachable improvement an 8-call probe would have missed.  Ties in
    the score are broken uniformly at random and averaged over ``reps`` draws,
    because the frequency baselines have only a handful of distinct values and an
    array-order tiebreak would flatter or punish them arbitrarily.

    Returns ``(absolute_regret, normalized_regret, parent_keys)``, one entry per
    eligible parent.  The normalized form divides by the parent's own
    ``max(gain) - min(gain)`` spread so a mean over parents of different
    difficulty is a meaningful number; it is NaN for a parent whose candidates
    all share one gain.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(parents, kind="mergesort")
    p_sorted = parents[order]
    bounds = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    abs_r, norm_r, keys = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[a:b]
        if len(idx) < min_candidates:
            continue
        g = gain[idx]
        s = scores[idx]
        best, worst = float(g.max()), float(g.min())
        picked = np.empty(reps)
        for r in range(reps):
            take = np.lexsort((rng.random(len(s)), -s))[:k]
            picked[r] = g[take].max()
        got = float(picked.mean())
        abs_r.append(best - got)
        norm_r.append((best - got) / (best - worst) if best > worst else np.nan)
        keys.append(p_sorted[a])
    return np.array(abs_r), np.array(norm_r), np.array(keys, dtype=object)


def _paired_parent_bootstrap(per_parent: dict[str, np.ndarray], *,
                             reps: int = 4000, seed: int = 20260817
                             ) -> tuple[dict[str, dict[str, float]],
                                        dict[str, np.ndarray]]:
    """Bootstrap the mean of each scorer's per-parent statistic, resampling PARENTS.

    Every scorer is resampled with the SAME parent draw, so the differences are
    paired.  The analysis unit is the parent because that is the unit the
    deployment metric is defined on and the unit the gate fold was split on.
    """
    names = list(per_parent)
    n = len(per_parent[names[0]])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(reps, n)) if n else np.zeros((reps, 0), int)
    boots = {k: np.nanmean(per_parent[k][draws], axis=1) for k in names}
    out: dict[str, dict[str, float]] = {}
    for k in names:
        m, lo, hi = _boot_ci(boots[k])
        out[k] = {"mean": float(np.nanmean(per_parent[k])), "lo": lo, "hi": hi,
                  "n_parents": int(n)}
    return out, boots


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def baseline_scores_v2(frame: pd.DataFrame, train: pd.DataFrame, head: str,
                       v1: pd.DataFrame | None, seed: int = 20260817
                       ) -> dict[str, np.ndarray]:
    """The four pre-registered baselines, all fitted on TRAIN only.

    ``policy_v1`` is not fitted here at all: it is the shipped v1 ensemble's
    probability, read from the blind CSV emitted before v2 existed.
    """
    col = EVAL_LABEL[head]
    rng = np.random.default_rng(seed)
    lab = train[col]
    prior = float(lab.mean()) if lab.notna().any() else 0.5
    by_class = train.groupby("move_class")[col].mean().to_dict()
    klass = frame["move_class"].astype(str).to_numpy()

    out = {
        "random": rng.random(len(frame)),
        "class_freq": np.array([float(by_class.get(k, prior)) for k in klass]),
        "periph": frame["d_fresh_share_periph"].to_numpy(np.float64),
    }
    v1_col = f"p_improve_{head}"
    if v1 is not None:
        table = v1.set_index("child_record_id")[v1_col]
        out["policy_v1"] = frame["child_record_id"].map(table).to_numpy(np.float64)
    else:
        out["policy_v1"] = np.full(len(frame), np.nan)
    return out


# --------------------------------------------------------------------------- #
# torch plumbing
# --------------------------------------------------------------------------- #
class _TorchSteps(Dataset):
    def __init__(self, inner: PolicySteps):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(np.ascontiguousarray(v))
                for k, v in self.inner[i].items()}


def _predict(model: PolicyNet, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            cells = batch["cells"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            out.append(torch.sigmoid(model(cells, cond)).cpu().numpy())
    return np.concatenate(out, axis=0)


def _masked_huber(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                  w: torch.Tensor) -> torch.Tensor:
    """Era-weighted, head-masked Huber on the SIGMOID output — protocol ``registered``.

    This is the pre-registered loss (prereg §2d) and it is kept so Run A stays
    reproducible from this file.  **It is gradient-starved on this target and
    should not be used again.**  With ~84% of the target mass at zero the linear
    Huber regime pushes every logit down until the sigmoid saturates, at which
    point ``dL/dz = huber'(r)·σ'(z) → 0`` and learning stalls: measured locally,
    the prediction mean falls 0.115 → 0.045 over four epochs while the spread
    collapses, and the val loss is minimised at epoch 0 for every seed.
    """
    loss = nn.functional.smooth_l1_loss(
        torch.sigmoid(pred), y, beta=HUBER_DELTA, reduction="none")
    weight = m * w.unsqueeze(1)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def _masked_bce_soft(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                     w: torch.Tensor) -> torch.Tensor:
    """Era-weighted, head-masked BCE against the SOFT target — protocol ``revB``.

    Same target, same [0, 1] output, same serving contract; only the distance
    function changes.  Cross-entropy is a proper scoring rule for a target that
    is itself a number in [0, 1], and its gradient through the sigmoid is exactly
    ``σ(z) − y`` — the ``σ'(z)`` factor that kills :func:`_masked_huber` cancels,
    so the model cannot stall by saturating.
    """
    loss = nn.functional.binary_cross_entropy_with_logits(pred, y, reduction="none")
    weight = m * w.unsqueeze(1)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


LOSSES = {"registered": _masked_huber, "revB": _masked_bce_soft}


def _val_spearman(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Mean over heads of Spearman(prediction, target) on the labelled val rows.

    The early-stopping criterion for protocol ``revB``.  A rank statistic because
    the object being trained is a RANKER (prereg §2d and ``scorer.py``'s own
    contract), and unweighted over the whole fold because the era weighting —
    correct for the training objective — turns a 90-row current-era slice into
    half the criterion, which is what froze Run A at epoch 0.
    """
    out = []
    for h in range(probs.shape[1]):
        keep = mask[:, h] > 0
        if keep.sum() < 2:
            continue
        a = pd.Series(probs[keep, h]).rank().to_numpy()
        b = pd.Series(y[keep, h]).rank().to_numpy()
        if a.std() == 0 or b.std() == 0:
            out.append(0.0)
            continue
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(out)) if out else 0.0


def train_one(seed: int, *, sets: dict[str, PolicyStepsV2], device: str,
              epochs: int, batch_size: int, lr: float, weight_decay: float,
              patience: int, width: int, n_blocks: int, num_workers: int,
              protocol: str = "revB") -> tuple[PolicyNet, dict[str, Any]]:
    """Train one member.

    ``protocol='registered'``  the pre-registered loss (Huber on the sigmoid) and
        the pre-registered stopping rule (era-weighted val loss).  Run A.
    ``protocol='revB'``  the declared deviation: BCE against the same soft target,
        early-stopped on the unweighted val Spearman.  Run B.

    Both are kept in one function so the two runs differ by one argument and
    nothing else — same data, same splits, same schedule, same seeds.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    train_set = sets["train"]
    cfg = PolicyNetConfig(arm="cnn", in_channels=train_set.n_channels,
                          n_cond=train_set.n_cond, width=width, n_blocks=n_blocks)
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
    criterion = LOSSES[protocol]
    # 'registered' minimises the val loss; 'revB' maximises the val Spearman.
    sign = -1.0 if protocol == "registered" else 1.0

    best, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in loaders["train"]:
            cells = batch["cells"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            loss = criterion(model(cells, cond),
                             batch["y"].to(device), batch["m"].to(device),
                             batch["w"].to(device))
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
                l = criterion(model(cells, cond), batch["y"].to(device), m, w)
                mass = float((m * w.unsqueeze(1)).sum())
                vtot += float(l) * mass
                vw += mass
        val_loss = vtot / max(vw, 1e-6)
        rho = _val_spearman(_predict(model, loaders["val"], device),
                            sets["val"].labels, sets["val"].mask)
        score = sign * val_loss if protocol == "registered" else rho
        history.append({"epoch": epoch, "loss": tot / max(nb, 1),
                        "val_loss": val_loss, "val_spearman": rho})
        if score > best + 1e-7:
            best, best_epoch, stale = score, epoch, 0
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
                   "protocol": protocol,
                   "best_val_loss": chosen.get("val_loss", float("nan")),
                   "best_val_spearman": chosen.get("val_spearman", float("nan")),
                   "n_params": count_parameters(model),
                   "net_config": dict(cfg.__dict__), "history": history}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_fold(name: str, frame: pd.DataFrame, probs: np.ndarray,
                  train: pd.DataFrame, v1: pd.DataFrame | None, *,
                  rng: np.random.Generator, n_boot: int = N_BOOTSTRAP
                  ) -> dict[str, Any]:
    """Every pre-registered number for one fold, both heads."""
    out: dict[str, Any] = {"fold": name, "n_steps": int(len(frame)),
                           "n_cells": int(frame["cell"].nunique()),
                           "n_current": int(frame["era_current"].sum())}
    parents = frame["parent_record_id"].astype(str).to_numpy()
    scorers = ("policy", *BASELINES)

    for h, head in enumerate(HEADS):
        col = EVAL_LABEL[head]
        keep = frame[col].notna().to_numpy()
        sub = frame[keep]
        y = sub[col].astype(bool).to_numpy().astype(np.float64)
        base = baseline_scores_v2(sub, train, head, v1)
        score = {"policy": probs[keep, h], **base}
        # A baseline with missing values cannot be ranked; NaN sorts last under
        # every ordering here, so it is replaced by the worst finite value rather
        # than being silently favoured.
        for k, v in score.items():
            if np.isnan(v).any():
                score[k] = np.where(np.isnan(v), np.nanmin(v) - 1.0, v)

        entry: dict[str, Any] = {
            "n_labeled": int(len(y)), "base_rate": float(y.mean()),
            "auc": {k: auc(score[k], y) for k in scorers},
        }
        if len(y) > 1 and 0 < y.sum() < len(y):
            boot = np.array([auc(score["policy"][i], y[i]) for i in
                             (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
            m, lo, hi = _boot_ci(boot[~np.isnan(boot)])
            entry["auc_ci"] = {"mean": m, "lo": lo, "hi": hi}

        # ---- deployment metric, v1's shape, for comparability -------------- #
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
            entry["beats_all_baselines_p32"] = bool(
                all(entry["precision_delta"][b]["beats"] for b in BASELINES))
        else:
            entry["precision_at_32"] = None
            entry["beats_all_baselines_p32"] = None

        # ---- parent-blocked AUC, with a PAIRED parent bootstrap ------------ #
        pb, npairs = parent_blocked_auc(score["policy"], y, parents[keep])
        entry["parent_blocked_auc"] = {
            "n_pairs": npairs,
            **{k: parent_blocked_auc(score[k], y, parents[keep])[0] for k in scorers},
        }
        per_parent = _per_parent_blocked(score, y, parents[keep], scorers)
        if per_parent is not None:
            summary, boots = _paired_parent_bootstrap(per_parent)
            entry["parent_blocked_auc_ci"] = summary
            entry["parent_blocked_delta"] = {
                b: _delta(boots["policy"] - boots[b],
                          float(np.nanmean(per_parent["policy"]
                                           - per_parent[b]))) for b in BASELINES}

        # ---- regret@8 — the deployment-shaped metric ----------------------- #
        gain = -frame[("d_f_r" if head == "fr" else "d_node_peak")].to_numpy(float)
        ok = np.isfinite(gain) & frame["both_converged"].fillna(False).to_numpy(bool)
        if ok.sum():
            reg: dict[str, np.ndarray] = {}
            nreg: dict[str, np.ndarray] = {}
            allp = {"policy": probs[:, h], **baseline_scores_v2(frame, train, head, v1)}
            for k in scorers:
                s = allp[k]
                s = np.where(np.isnan(s), np.nanmin(s) - 1.0, s) if np.isnan(s).any() else s
                a, nn_, keys = regret_at_k(s[ok], gain[ok], parents[ok])
                reg[k], nreg[k] = a, nn_
            if len(reg["policy"]):
                summary, boots = _paired_parent_bootstrap(reg)
                nsummary, nboots = _paired_parent_bootstrap(nreg)
                entry["regret_at_8"] = summary
                entry["regret_at_8_normalized"] = nsummary
                # LOWER is better, so the improvement is (baseline - policy).
                entry["regret_delta"] = {
                    b: _delta(boots[b] - boots["policy"],
                              float(np.nanmean(reg[b] - reg["policy"])))
                    for b in BASELINES}
                entry["beats_all_baselines_regret"] = bool(
                    all(entry["regret_delta"][b]["beats"] for b in BASELINES))
        entry["calibration"] = calibration(probs[keep, h], y)
        entry["target_rmse"] = _target_rmse(frame, probs, h, head)
        out[head] = entry
    return out


def _delta(boot: np.ndarray, point: float) -> dict[str, float]:
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"mean": point, "lo": lo, "hi": hi, "beats": bool(lo > 0.0)}


def _target_rmse(frame: pd.DataFrame, probs: np.ndarray, h: int,
                 head: str) -> float:
    """RMSE against the v2 training target — reported, never gated."""
    y, m = targets(frame)
    keep = m[:, h] > 0
    if not keep.any():
        return float("nan")
    return float(np.sqrt(np.mean((probs[keep, h] - y[keep, h]) ** 2)))


def _per_parent_blocked(score: dict[str, np.ndarray], y: np.ndarray,
                        parents: np.ndarray, scorers: Sequence[str]):
    """Per-parent concordance, one row per parent that has both label classes."""
    order = np.argsort(parents, kind="mergesort")
    p_sorted = parents[order]
    bounds = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    rows: dict[str, list[float]] = {k: [] for k in scorers}
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[a:b]
        lab = y[idx].astype(bool)
        if not lab.any() or lab.all():
            continue
        for k in scorers:
            s = score[k][idx]
            d = s[lab][:, None] - s[~lab][None, :]
            rows[k].append(float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size))
    if not rows[scorers[0]]:
        return None
    return {k: np.array(v) for k, v in rows.items()}


# --------------------------------------------------------------------------- #
# the blind v1 baseline
# --------------------------------------------------------------------------- #
def emit_v1_baseline(steps: pd.DataFrame, cache: PatternCache, out_path: Path,
                     model_dir: Path, device: str = "cpu") -> pd.DataFrame:
    """Score every row with the shipped v1 ensemble and write the CSV.

    v1's OWN feature layout is used — ``lpopt.policy.data.scalar_features`` and
    the delta channels named in v1's ``meta.json`` — so this is the model that
    was gated in August, not a re-implementation of it.
    """
    dirs = sorted(d for d in model_dir.glob("cnn_seed*") if (d / "model.pt").is_file())
    if not dirs:
        raise SystemExit(f"no v1 checkpoints under {model_dir}")
    meta0 = json.loads((dirs[0] / "meta.json").read_text())
    index = {c: i for i, c in enumerate(cache.channels)}
    delta = [index[c] for c in meta0["delta_channels"]]

    scalars, names = scalar_features(steps)
    if names != meta0["scalar_names"]:
        raise SystemExit("v1 scalar layout drifted; v1 cannot be scored as a baseline")
    data = PolicySteps(steps, cache, scalars, delta_channels=delta, augment=False)
    loader = DataLoader(_TorchSteps(data), batch_size=256, shuffle=False)

    total = np.zeros((len(steps), len(HEADS)))
    for d in dirs:
        net = PolicyNet(PolicyNetConfig(**json.loads((d / "meta.json").read_text())
                                        ["net_config"]))
        net.load_state_dict(torch.load(d / "model.pt", map_location="cpu",
                                       weights_only=True))
        net.eval().to(device)
        total += _predict(net, loader, device)
    total /= len(dirs)

    frame = pd.DataFrame({
        "parent_record_id": steps["parent_record_id"].to_numpy(),
        "child_record_id": steps["child_record_id"].to_numpy(),
        "p_improve_fr": total[:, 0], "p_improve_flat": total[:, 1],
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"[v1-baseline] {len(frame)} rows from {len(dirs)} members -> {out_path}",
          flush=True)
    return frame


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--tables":
        print(render_tables(json.loads(Path(argv[1]).read_text())))
        return 0

    ap = argparse.ArgumentParser(prog="python -m lpopt.policy.train_v2")
    ap.add_argument("--steps", default="data/policy/steps.parquet")
    ap.add_argument("--fuel-types", default="data/store/fuel_types.parquet")
    ap.add_argument("--cache", default="data/policy/_feature_cache_v2.npz")
    ap.add_argument("--out-dir", default="runs/policy_v2")
    ap.add_argument("--v1-baseline", default="data/design/policy_v2_v1_baseline.csv")
    ap.add_argument("--v1-model-dir", default="data/models/policy_v1")
    ap.add_argument("--emit-v1-baseline", default="")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=20260817)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--protocol", default="revB", choices=sorted(LOSSES),
                    help="'registered' = the pre-registered Huber/val-loss "
                         "protocol (Run A); 'revB' = the declared deviation")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device} torch={torch.__version__}", flush=True)

    steps = load_universe_v2(args.steps)
    fold = build_splits_v2(steps, seed=args.base_seed)
    summary = split_summary_v2(steps, fold)
    print("=== splits ===\n" + summary.to_string(index=False), flush=True)

    if args.emit_v1_baseline:
        # Only the EVALUATION folds need a v1 score, and featurizing the whole
        # 20k-row corpus to produce 3k numbers would cost ~13 minutes of encoder
        # calls for nothing.  The cache is built over the subset and thrown away.
        sub = steps[fold.isin(("gate_cur", "val"))].reset_index(drop=True)
        print(f"=== v1 baseline over {len(sub)} eval rows ===", flush=True)
        small = build_pattern_cache(sub, fuel_types=args.fuel_types)
        emit_v1_baseline(sub, small, Path(args.emit_v1_baseline),
                         Path(args.v1_model_dir), device="cpu")
        return 0

    cache_path = Path(args.cache)
    if cache_path.is_file():
        print(f"=== loading feature cache {cache_path} ===", flush=True)
        cache = PatternCache.load(cache_path)
    else:
        print("=== building feature cache (once) ===", flush=True)
        cache = build_pattern_cache(steps, fuel_types=args.fuel_types)
        cache.save(cache_path)

    v1 = pd.read_csv(args.v1_baseline) if Path(args.v1_baseline).is_file() else None
    if v1 is None:
        print(f"WARNING: {args.v1_baseline} missing; the v1 baseline column will "
              f"be NaN and its comparison void", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_frame = steps[fold == "train"]
    delta = pick_delta_channels(train_frame, cache, seed=args.base_seed)
    print(f"=== delta channels: {len(delta)} of {cache.slots.shape[1]} ===",
          flush=True)

    scalars, scalar_names = scalar_features_v2(steps)
    weights = era_weights(steps, fold)
    print(f"=== scalars: {len(scalar_names)} (v2 adds {list(NEW_SCALARS)}); "
          f"current-era loss weight {weights[steps['era_current'].to_numpy()][0]:.2f} "
          f"===", flush=True)
    folds = {name: steps.index[fold == name].to_numpy() for name in fold.unique()}

    def make(name: str, augment: bool, seed: int) -> PolicyStepsV2:
        idx = folds[name]
        return PolicyStepsV2(steps.loc[idx], cache, scalars[idx], weights[idx],
                             delta_channels=delta, augment=augment, seed=seed)

    eval_folds = [f for f in ("gate_cur", "val") if f in folds]
    manifest: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_sha256": corpus_fingerprint(args.steps),
        "cond_schema": COND_SCHEMA, "move_classes": list(MOVE_CLASSES),
        "scalar_names": scalar_names, "new_scalars": list(NEW_SCALARS),
        "target_clip": TARGET_CLIP, "huber_delta": HUBER_DELTA,
        "probe_k": PROBE_K, "regret_min_candidates": REGRET_MIN_CANDIDATES,
        "protocol": args.protocol,
        "delta_channels": [cache.channels[i] for i in delta],
        "split_summary": summary.to_dict("records"),
        "current_era_loss_weight": float(
            weights[steps["era_current"].to_numpy()][0]),
        "args": vars(args), "device": device, "torch": torch.__version__,
    }

    members: list[dict[str, Any]] = []
    per_fold: dict[str, list[np.ndarray]] = {n: [] for n in eval_folds}
    for k in range(args.seeds):
        seed = args.base_seed + k
        sets = {"train": make("train", True, seed), "val": make("val", False, seed)}
        model, meta = train_one(
            seed, sets=sets, device=device, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            width=args.width, n_blocks=args.n_blocks,
            num_workers=args.num_workers, protocol=args.protocol)
        member_dir = out_dir / f"cnn_seed{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), member_dir / "model.pt")
        (member_dir / "meta.json").write_text(json.dumps(
            {**meta, "cond_schema": COND_SCHEMA,
             "delta_channels": [cache.channels[i] for i in delta],
             "scalar_names": scalar_names, "policy_version": "v2",
             "protocol": args.protocol,
             "target_clip": TARGET_CLIP}, indent=2, sort_keys=True))
        members.append({k2: v for k2, v in meta.items() if k2 != "history"})
        print(f"  [seed {seed}] params={meta['n_params']:,} "
              f"val_loss={meta['best_val_loss']:.5f} "
              f"val_rho={meta['best_val_spearman']:+.4f} @{meta['best_epoch']}",
              flush=True)
        for name in eval_folds:
            loader = DataLoader(_TorchSteps(make(name, False, seed)),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
            per_fold[name].append(_predict(model, loader, device))

    print("=== evaluating ===", flush=True)
    rng = np.random.default_rng(args.base_seed)
    results: dict[str, Any] = {"members": members, "folds": {}}
    for name in eval_folds:
        ens = np.mean(np.stack(per_fold[name]), axis=0)
        results["folds"][name] = evaluate_fold(
            name, steps.loc[folds[name]], ens, train_frame, v1,
            rng=rng, n_boot=args.n_bootstrap)
    np.savez_compressed(out_dir / "probs.npz",
                        **{n: np.stack(per_fold[n]) for n in eval_folds})

    g = results["folds"].get("gate_cur", {}).get("fr", {})
    pb = g.get("parent_blocked_auc", {})
    pbci = g.get("parent_blocked_auc_ci", {}).get("policy", {})
    gate = {
        "pb_auc": float(pb.get("policy", float("nan"))),
        "pb_auc_ci_lo": float(pbci.get("lo", float("nan"))),
        "clause_1_pb_auc": bool(pb.get("policy", 0) >= 0.65
                                and pbci.get("lo", 0) > 0.50),
        "clause_2_p32_beats_all": bool(g.get("beats_all_baselines_p32") is True),
    }
    gate["PASS"] = bool(gate["clause_1_pb_auc"] and gate["clause_2_p32_beats_all"])
    gate["recommendation_regret"] = bool(
        g.get("regret_delta", {}).get("random", {}).get("beats") is True
        and g.get("regret_delta", {}).get("policy_v1", {}).get("beats") is True)
    manifest["gate"] = gate
    manifest["results"] = results
    manifest["wall_seconds"] = time.time() - t0
    (out_dir / "metrics.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=float))
    print("=== gate ===\n" + json.dumps(gate, indent=2), flush=True)
    print(f"wrote {out_dir / 'metrics.json'} in {time.time() - t0:.1f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# report tables (rendered from metrics.json so the report cannot drift from it)
# --------------------------------------------------------------------------- #
def render_tables(metrics: dict[str, Any]) -> str:
    scorers = ("policy", *BASELINES)
    out: list[str] = []
    res = metrics["results"]
    losses = [m["best_val_loss"] for m in res["members"]]
    rhos = [m.get("best_val_spearman", float("nan")) for m in res["members"]]
    rho_txt = ("n/a" if not np.isfinite(rhos).any()
               else f"{np.nanmean(rhos):+.4f} "
                    f"(min {np.nanmin(rhos):+.4f}, max {np.nanmax(rhos):+.4f})")
    out.append(f"\n{len(res['members'])} seeds, {res['members'][0]['n_params']:,} "
               f"params, protocol "
               f"`{res['members'][0].get('protocol', 'registered')}`, "
               f"val loss {np.mean(losses):.5f}, val Spearman {rho_txt}, "
               f"stop epochs {[m['best_epoch'] for m in res['members']]}\n")

    out.append("\n**AUC on `improved_*`** (5-seed ensemble)\n")
    out.append("| fold | head | n | base | " + " | ".join(scorers) + " | policy 95% CI |")
    out.append("|" + "---|" * (5 + len(scorers)))
    for f, fr in res["folds"].items():
        for head in HEADS:
            e = fr[head]
            ci = e.get("auc_ci")
            cis = f"[{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else "n/a"
            out.append(f"| {f} | {head} | {e['n_labeled']} | {e['base_rate']:.3f} | "
                       + " | ".join(f"{e['auc'][s]:.3f}" for s in scorers)
                       + f" | {cis} |")

    out.append("\n**Parent-blocked AUC** — moves ranked WITHIN a parent "
               "(paired parent bootstrap)\n")
    out.append("| fold | head | n_pairs | " + " | ".join(scorers)
               + " | beats " + " | beats ".join(BASELINES) + " |")
    out.append("|" + "---|" * (3 + len(scorers) + len(BASELINES)))
    for f, fr in res["folds"].items():
        for head in HEADS:
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

    out.append("\n**precision@32 of 256** (paired bootstrap)\n")
    out.append("| fold | head | " + " | ".join(scorers)
               + " | beats " + " | beats ".join(BASELINES) + " |")
    out.append("|" + "---|" * (2 + len(scorers) + len(BASELINES)))
    for f, fr in res["folds"].items():
        for head in HEADS:
            e = fr[head]
            p = e.get("precision_at_32")
            if not p:
                continue
            d = e["precision_delta"]
            out.append(f"| {f} | {head} | "
                       + " | ".join(f"{p[s]['mean']:.3f}" for s in scorers) + " | "
                       + " | ".join(
                           ("**yes**" if d[b]["beats"] else "no")
                           + f" ({d[b]['mean']:+.3f} [{d[b]['lo']:+.3f}, "
                             f"{d[b]['hi']:+.3f}])" for b in BASELINES) + " |")

    out.append("\n**regret@8** — of 8 proposed moves, how much of the reachable "
               "improvement is missed (LOWER is better)\n")
    out.append("| fold | head | n_parents | " + " | ".join(scorers)
               + " | beats " + " | beats ".join(BASELINES) + " |")
    out.append("|" + "---|" * (3 + len(scorers) + len(BASELINES)))
    for f, fr in res["folds"].items():
        for head in HEADS:
            e = fr[head]
            r = e.get("regret_at_8")
            if not r:
                continue
            d = e["regret_delta"]
            out.append(f"| {f} | {head} | {r['policy']['n_parents']} | "
                       + " | ".join(f"{r[s]['mean']:.4f}" for s in scorers) + " | "
                       + " | ".join(
                           ("**yes**" if d[b]["beats"] else "no")
                           + f" ({d[b]['mean']:+.4f} [{d[b]['lo']:+.4f}, "
                             f"{d[b]['hi']:+.4f}])" for b in BASELINES) + " |")

    out.append("\n**regret@8, normalized by each parent's own gain spread**\n")
    out.append("| fold | head | " + " | ".join(scorers) + " |")
    out.append("|" + "---|" * (2 + len(scorers)))
    for f, fr in res["folds"].items():
        for head in HEADS:
            r = fr[head].get("regret_at_8_normalized")
            if r:
                out.append(f"| {f} | {head} | "
                           + " | ".join(f"{r[s]['mean']:.3f}" for s in scorers) + " |")

    out.append("\n**Calibration and target fit** (reported, not gated)\n")
    out.append("| fold | head | Brier | ECE | target RMSE |")
    out.append("|---|---|---|---|---|")
    for f, fr in res["folds"].items():
        for head in HEADS:
            c = fr[head]["calibration"]
            out.append(f"| {f} | {head} | {c['brier']:.4f} | {c['ece']:.4f} | "
                       f"{fr[head]['target_rmse']:.4f} |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
