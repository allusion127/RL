"""Train and judge the v1 move-proposal policy.

    python -m lpopt.policy.train --out-dir runs/policy_v1 --arms cnn,mlp --seeds 5

Everything the gate needs is computed here and written to ``metrics.json``:
AUC per head, the deployment metric (precision@32 out of a 256-candidate batch)
against the three pre-registered baselines with PAIRED bootstrap CIs, the
parent-blocked AUC, calibration, and the three held-out-family readouts.

The protocol is fixed by ``data/reports/policy_v1_prereg_20260815.md`` and this
module is deliberately the only place it is written down in code.
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
    build_pattern_cache, build_splits, corpus_fingerprint, load_universe,
    pick_delta_channels, scalar_features, split_summary,
)
from .net import ARMS, PolicyNet, PolicyNetConfig, count_parameters

#: Deployment metric shape (prereg section 4).  The brief asked for 256
#: candidates PER PARENT; the corpus cannot supply that — the busiest parent has
#: 42 children and only one parent has 32 — so a batch is drawn from the
#: evaluation fold rather than from one parent, and the strict within-parent
#: question is answered separately by the parent-blocked AUC.
BATCH_CANDIDATES = 256
TOP_K = 32
N_BOOTSTRAP = 2000

BASELINES: tuple[str, ...] = ("random", "class_freq", "periph")
#: Reported, NOT gated — a per-(cell, class) frequency table is a much stronger
#: baseline than the three the brief pre-registered, and folding it into the
#: gate after the fact would be moving the goalposts.
ADVISORY_BASELINES: tuple[str, ...] = ("cell_class_freq",)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by rank sum; ties get their mean rank.  NaN if one class only."""
    y = labels.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0                                     # average ranks within tie runs
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def parent_blocked_auc(scores: np.ndarray, labels: np.ndarray,
                       parents: np.ndarray) -> tuple[float, int]:
    """Fraction of concordant (improving, non-improving) pairs SHARING a parent.

    This is the strict generator-prior question — "of the moves available from
    THIS board, does the policy rank the good ones first?" — with the parent's
    own difficulty differenced out.  It uses every parent that has both classes,
    so it needs no minimum-children threshold.
    """
    conc = ties = total = 0
    order = np.argsort(parents, kind="mergesort")
    p_sorted = parents[order]
    bounds = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[a:b]
        y = labels[idx].astype(bool)
        if not y.any() or y.all():
            continue
        pos, neg = scores[idx][y], scores[idx][~y]
        diff = pos[:, None] - neg[None, :]
        conc += int((diff > 0).sum())
        ties += int((diff == 0).sum())
        total += diff.size
    if total == 0:
        return float("nan"), 0
    return (conc + 0.5 * ties) / total, total


def precision_at_k(scores: np.ndarray, labels: np.ndarray, *,
                   draws: np.ndarray, tiebreak: np.ndarray,
                   k: int = TOP_K) -> np.ndarray:
    """Per-replicate precision@k.  ``draws[r]`` is one 256-candidate batch.

    ``tiebreak`` is drawn once per replicate and shared by every scorer, so the
    frequency baselines (which have only a handful of distinct values) are
    broken uniformly at random rather than by array order, and the comparison
    stays paired.
    """
    out = np.empty(len(draws), np.float64)
    for r, idx in enumerate(draws):
        s = scores[idx]
        order = np.lexsort((tiebreak[r], -s))
        out[r] = labels[idx][order[:k]].mean()
    return out


def calibration(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 10) -> dict[str, Any]:
    """Brier, ECE and the reliability table on equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        conf, freq, n = probs[m].mean(), labels[m].mean(), int(m.sum())
        ece += n / len(probs) * abs(conf - freq)
        rows.append({"bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
                     "n": n, "mean_pred": float(conf), "observed": float(freq)})
    return {"brier": float(np.mean((probs - labels) ** 2)),
            "ece": float(ece), "bins": rows}


def _boot_ci(x: np.ndarray) -> tuple[float, float, float]:
    return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def baseline_scores(frame: pd.DataFrame, train: pd.DataFrame, head: str,
                    seed: int = 20260815) -> dict[str, np.ndarray]:
    """The three pre-registered baselines (plus the advisory one).

    All three are fitted on TRAIN only, exactly like the policy.
    """
    col = "improved_fr" if head == "fr" else "improved_flat"
    rng = np.random.default_rng(seed)
    lab = train[col]
    prior = float(lab.mean()) if lab.notna().any() else 0.5

    by_class = train.groupby("move_class")[col].mean().to_dict()
    by_cell_class = train.groupby(["cell", "move_class"])[col].mean().to_dict()

    klass = frame["move_class"].astype(str).to_numpy()
    cells = frame["cell"].astype(str).to_numpy()
    return {
        "random": rng.random(len(frame)),
        "class_freq": np.array(
            [float(by_class.get(k, prior)) for k in klass], np.float64),
        # the engineer's rule of thumb, as a continuous ranker: "push fresh out"
        "periph": frame["d_fresh_share_periph"].to_numpy(np.float64),
        "cell_class_freq": np.array(
            [float(by_cell_class.get((c, k), by_class.get(k, prior)))
             for c, k in zip(cells, klass, strict=True)], np.float64),
    }


# --------------------------------------------------------------------------- #
# torch plumbing
# --------------------------------------------------------------------------- #
class _TorchSteps(Dataset):
    def __init__(self, inner: PolicySteps):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(np.ascontiguousarray(v))
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


def train_one(arm: str, seed: int, *, sets: dict[str, PolicySteps],
              device: str, epochs: int, batch_size: int, lr: float,
              weight_decay: float, patience: int, width: int, n_blocks: int,
              num_workers: int) -> tuple[PolicyNet, dict[str, Any]]:
    """Train one member; early-stop on the mean val AUC over the two heads."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    train_set = sets["train"]
    cfg = PolicyNetConfig(arm=arm, in_channels=train_set.n_channels,
                          n_cond=train_set.n_cond, width=width,
                          n_blocks=n_blocks)
    model = PolicyNet(cfg).to(device)

    loaders = {
        name: DataLoader(_TorchSteps(s), batch_size=batch_size,
                         shuffle=(name == "train"), drop_last=False,
                         num_workers=num_workers,
                         pin_memory=(device.startswith("cuda")),
                         persistent_workers=bool(num_workers))
        for name, s in sets.items()
    }
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    val_y = sets["val"].labels
    val_m = sets["val"].mask
    best, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in loaders["train"]:
            cells = batch["cells"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            m = batch["m"].to(device, non_blocking=True)
            logits = model(cells, cond)
            # masked BCE: a row missing the flatness label trains the F_r head
            # only, and contributes nothing to the flatness gradient.
            loss = (bce(logits, y) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()

        probs = _predict(model, loaders["val"], device)
        aucs = [auc(probs[val_m[:, h] > 0, h], val_y[val_m[:, h] > 0, h])
                for h in range(len(HEADS))]
        score = float(np.nanmean(aucs))
        history.append({"epoch": epoch, "loss": tot / max(nb, 1),
                        **{f"val_auc_{h}": a for h, a in zip(HEADS, aucs, strict=True)}})
        if score > best + 1e-5:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        if epoch % 10 == 0:
            print(f"  [{arm} seed {seed}] epoch {epoch:3d} loss={tot / max(nb, 1):.4f} "
                  f"val_auc={score:.4f} (best {best:.4f} @ {best_epoch})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"arm": arm, "seed": seed, "best_epoch": best_epoch,
            "best_val_auc": best, "n_params": count_parameters(model),
            "net_config": dict(cfg.__dict__), "history": history}
    return model, meta


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_fold(name: str, frame: pd.DataFrame, probs: np.ndarray,
                  train: pd.DataFrame, *, rng: np.random.Generator,
                  n_boot: int = N_BOOTSTRAP) -> dict[str, Any]:
    """Every pre-registered number for one evaluation fold, both heads."""
    out: dict[str, Any] = {"fold": name, "n_steps": int(len(frame)),
                           "n_cells": int(frame["cell"].nunique())}
    parents = frame["parent_record_id"].astype(str).to_numpy()

    for h, head in enumerate(HEADS):
        col = "improved_fr" if head == "fr" else "improved_flat"
        keep = frame[col].notna().to_numpy()
        sub = frame[keep]
        y = sub[col].astype(bool).to_numpy().astype(np.float64)
        pol = probs[keep, h]
        base = baseline_scores(sub, train, head)
        entry: dict[str, Any] = {
            "n_labeled": int(len(y)), "base_rate": float(y.mean()),
            "auc": {"policy": auc(pol, y),
                    **{b: auc(base[b], y) for b in (*BASELINES, *ADVISORY_BASELINES)}},
        }

        # bootstrap AUC CI for the policy (resample rows with replacement)
        if len(y) > 1 and 0 < y.sum() < len(y):
            boot = np.empty(n_boot)
            for r in range(n_boot):
                idx = rng.integers(0, len(y), len(y))
                boot[r] = auc(pol[idx], y[idx])
            m, lo, hi = _boot_ci(boot[~np.isnan(boot)])
            entry["auc_ci"] = {"mean": m, "lo": lo, "hi": hi}

        # --- deployment metric ------------------------------------------- #
        if len(y) >= BATCH_CANDIDATES:
            draws = np.array([rng.choice(len(y), BATCH_CANDIDATES, replace=False)
                              for _ in range(n_boot)])
            tiebreak = rng.random((n_boot, BATCH_CANDIDATES))
            p_pol = precision_at_k(pol, y, draws=draws, tiebreak=tiebreak)
            prec = {"policy": dict(zip(("mean", "lo", "hi"), _boot_ci(p_pol)))}
            deltas = {}
            for b in (*BASELINES, *ADVISORY_BASELINES):
                p_b = precision_at_k(base[b], y, draws=draws, tiebreak=tiebreak)
                prec[b] = dict(zip(("mean", "lo", "hi"), _boot_ci(p_b)))
                d_m, d_lo, d_hi = _boot_ci(p_pol - p_b)   # PAIRED difference
                deltas[b] = {"mean": d_m, "lo": d_lo, "hi": d_hi,
                             "beats": bool(d_lo > 0.0)}
            entry["precision_at_32"] = prec
            entry["precision_delta"] = deltas
            entry["gate_beats_all_three"] = bool(
                all(deltas[b]["beats"] for b in BASELINES))
        else:
            entry["precision_at_32"] = None
            entry["gate_beats_all_three"] = None

        pb, npairs = parent_blocked_auc(pol, y, parents[keep])
        entry["parent_blocked_auc"] = {"policy": pb, "n_pairs": npairs}
        for b in BASELINES:
            entry["parent_blocked_auc"][b] = parent_blocked_auc(
                base[b], y, parents[keep])[0]
        entry["calibration"] = calibration(pol, y)
        out[head] = entry
    return out


# --------------------------------------------------------------------------- #
# report tables (rendered from metrics.json so the report cannot drift from it)
# --------------------------------------------------------------------------- #
def render_tables(metrics: dict[str, Any]) -> str:
    """Markdown tables for ``data/reports/policy_v1_results_*.md``."""
    scorers = ("policy", *BASELINES, *ADVISORY_BASELINES)
    out: list[str] = []

    for arm, res in metrics["results"].items():
        out.append(f"\n### arm `{arm}`\n")
        aucs = [m["best_val_auc"] for m in res["members"]]
        out.append(f"{len(res['members'])} seeds, "
                   f"{res['members'][0]['n_params']:,} params, "
                   f"best val AUC {np.mean(aucs):.4f} "
                   f"(min {np.min(aucs):.4f}, max {np.max(aucs):.4f}), "
                   f"stop epochs {[m['best_epoch'] for m in res['members']]}\n")

        out.append("\n**AUC** (5-seed ensemble)\n")
        out.append("| fold | head | n | base | " + " | ".join(scorers)
                   + " | policy 95% CI |")
        out.append("|" + "---|" * (5 + len(scorers) + 1))
        for fold, fr in res["folds"].items():
            for head in HEADS:
                e = fr[head]
                ci = e.get("auc_ci")
                cis = f"[{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else "n/a"
                cells = " | ".join(f"{e['auc'][s]:.3f}" for s in scorers)
                out.append(f"| {fold} | {head} | {e['n_labeled']} | "
                           f"{e['base_rate']:.3f} | {cells} | {cis} |")

        out.append("\n**precision@32 of 256** (paired bootstrap; "
                   "`beats` = 95% CI of the paired difference excludes 0)\n")
        out.append("| fold | head | " + " | ".join(scorers)
                   + " | beats random | beats class_freq | beats periph |")
        out.append("|" + "---|" * (2 + len(scorers) + 3))
        for fold, fr in res["folds"].items():
            for head in HEADS:
                e = fr[head]
                p = e.get("precision_at_32")
                if not p:
                    out.append(f"| {fold} | {head} | "
                               + " | ".join(["n/a"] * len(scorers))
                               + " | n/a | n/a | n/a |")
                    continue
                d = e["precision_delta"]
                cells = " | ".join(f"{p[s]['mean']:.3f}" for s in scorers)
                beats = " | ".join(
                    ("**yes**" if d[b]["beats"] else "no")
                    + f" ({d[b]['mean']:+.3f} [{d[b]['lo']:+.3f}, {d[b]['hi']:+.3f}])"
                    for b in BASELINES)
                out.append(f"| {fold} | {head} | {cells} | {beats} |")

        out.append("\n**Parent-blocked AUC** — moves ranked WITHIN a parent\n")
        out.append("| fold | head | n_pairs | policy | "
                   + " | ".join(BASELINES) + " |")
        out.append("|" + "---|" * (3 + 1 + len(BASELINES)))
        for fold, fr in res["folds"].items():
            for head in HEADS:
                pb = fr[head]["parent_blocked_auc"]
                base = " | ".join(f"{pb[b]:.3f}" for b in BASELINES)
                out.append(f"| {fold} | {head} | {pb['n_pairs']} | "
                           f"{pb['policy']:.3f} | {base} |")

        out.append("\n**Calibration**\n")
        out.append("| fold | head | Brier | ECE |")
        out.append("|---|---|---|---|")
        for fold, fr in res["folds"].items():
            for head in HEADS:
                c = fr[head]["calibration"]
                out.append(f"| {fold} | {head} | {c['brier']:.4f} | {c['ece']:.4f} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--tables":
        print(render_tables(json.loads(Path(argv[1]).read_text())))
        return 0

    ap = argparse.ArgumentParser(prog="python -m lpopt.policy.train")
    ap.add_argument("--steps", default="data/policy/steps.parquet")
    ap.add_argument("--fuel-types", default="data/store/fuel_types.parquet")
    ap.add_argument("--cache", default="data/policy/_feature_cache.npz")
    ap.add_argument("--out-dir", default="runs/policy_v1")
    ap.add_argument("--arms", default="cnn,mlp")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=20260815)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"device={device} torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()}", flush=True)

    steps = load_universe(args.steps)
    fold = build_splits(steps, seed=args.base_seed)
    summary = split_summary(steps, fold)
    print("=== splits ===\n" + summary.to_string(index=False), flush=True)

    cache_path = Path(args.cache)
    if cache_path.is_file():
        print(f"=== loading feature cache {cache_path} ===", flush=True)
        cache = PatternCache.load(cache_path)
    else:
        print("=== building feature cache (once) ===", flush=True)
        tc = time.time()
        cache = build_pattern_cache(steps, fuel_types=args.fuel_types)
        cache.save(cache_path)
        print(f"=== cache: {cache.slots.shape} in {time.time() - tc:.1f}s ===",
              flush=True)

    train_frame = steps[fold == "train"]
    delta = pick_delta_channels(train_frame, cache, seed=args.base_seed)
    print(f"=== delta channels: {len(delta)} of {cache.slots.shape[1]} "
          f"({[cache.channels[i] for i in delta]}) ===", flush=True)

    scalars, scalar_names = scalar_features(steps)
    folds = {name: steps.index[fold == name].to_numpy()
             for name in fold.unique()}

    def make(name: str, augment: bool, seed: int) -> PolicySteps:
        idx = folds[name]
        return PolicySteps(steps.loc[idx], cache, scalars[idx],
                           delta_channels=delta, augment=augment, seed=seed)

    eval_folds = [f for f in ("test", "heldout_cell", "heldout_lib",
                              "heldout_era") if f in folds]

    manifest: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_sha256": corpus_fingerprint(args.steps),
        "cond_schema": COND_SCHEMA, "move_classes": list(MOVE_CLASSES),
        "scalar_names": scalar_names,
        "delta_channels": [cache.channels[i] for i in delta],
        "split_summary": summary.to_dict("records"),
        "args": vars(args), "device": device,
        "torch": torch.__version__,
    }

    probs_by_arm: dict[str, dict[str, list[np.ndarray]]] = {}
    members: list[dict[str, Any]] = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}")
        probs_by_arm[arm] = {name: [] for name in eval_folds}
        for k in range(args.seeds):
            seed = args.base_seed + k
            sets = {"train": make("train", True, seed), "val": make("val", False, seed)}
            model, meta = train_one(
                arm, seed, sets=sets, device=device, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, patience=args.patience,
                width=args.width, n_blocks=args.n_blocks,
                num_workers=args.num_workers)
            member_dir = out_dir / f"{arm}_seed{seed}"
            member_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), member_dir / "model.pt")
            (member_dir / "meta.json").write_text(
                json.dumps({**meta, "cond_schema": COND_SCHEMA,
                            "delta_channels": [cache.channels[i] for i in delta],
                            "scalar_names": scalar_names},
                           indent=2, sort_keys=True))
            members.append({k2: v for k2, v in meta.items() if k2 != "history"})
            print(f"  [{arm} seed {seed}] params={meta['n_params']:,} "
                  f"best_val_auc={meta['best_val_auc']:.4f} "
                  f"@epoch {meta['best_epoch']}", flush=True)

            for name in eval_folds:
                loader = DataLoader(_TorchSteps(make(name, False, seed)),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
                probs_by_arm[arm][name].append(_predict(model, loader, device))

    print("=== evaluating ===", flush=True)
    results: dict[str, Any] = {}
    for arm, per_fold in probs_by_arm.items():
        rng = np.random.default_rng(args.base_seed)
        results[arm] = {"members": [m for m in members if m["arm"] == arm],
                        "folds": {}}
        for name in eval_folds:
            ens = np.mean(np.stack(per_fold[name]), axis=0)
            results[arm]["folds"][name] = evaluate_fold(
                name, steps.loc[folds[name]], ens, train_frame,
                rng=rng, n_boot=args.n_bootstrap)
            # per-seed spread on the primary fold only (cheap, honest)
            if name == "test":
                frame = steps.loc[folds[name]]
                per_seed = []
                for p in per_fold[name]:
                    row = {}
                    for h, head in enumerate(HEADS):
                        col = frame["improved_fr" if head == "fr"
                                    else "improved_flat"]
                        keep = col.notna().to_numpy()
                        row[head] = auc(
                            p[keep, h],
                            col[keep].astype(bool).to_numpy().astype(float))
                    per_seed.append(row)
                results[arm]["folds"][name]["per_seed_auc"] = per_seed
        np.savez_compressed(out_dir / f"probs_{arm}.npz",
                            **{name: np.stack(per_fold[name])
                               for name in eval_folds})

    gate = {}
    for arm, res in results.items():
        t = res["folds"].get("test", {})
        gate[arm] = {
            head: bool(t.get(head, {}).get("gate_beats_all_three") is True
                       and t.get(head, {}).get("auc_ci", {}).get("lo", 0) > 0.5)
            for head in HEADS
        }
        gate[arm]["PASS"] = bool(all(gate[arm][h] for h in HEADS))
    manifest["gate"] = gate
    manifest["results"] = results
    manifest["wall_seconds"] = time.time() - t0

    (out_dir / "metrics.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=float))
    print("=== gate ===\n" + json.dumps(gate, indent=2), flush=True)
    print(f"wrote {out_dir / 'metrics.json'} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
