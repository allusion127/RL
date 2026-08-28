"""Pre-registered A/B scoring for the hires arms (design doc 20260725 section 6).

Scores one or more trained model directories against the metrics fixed BEFORE the
arms were launched, and accumulates them into ``data/reports/hires_ab_results.json``.

What it measures, and why these and not accuracy alone
------------------------------------------------------
The diagnosed bottleneck is not error magnitude but **effective resolution**: how
small a true difference the model can still order correctly inside one design
cell.  Report 20260725 section 3.2 measured node_peak's within-cell MAE (0.145) to
EXCEED the within-cell spread it must resolve (0.141) -- at that point a lower MAE
is not evidence of a better search model.  So the primary metric is

    Delta75 / within-cell SD

the true gap at which pairwise ordering first reaches 75% accuracy, expressed in
units of the signal actually present in a cell.  Below 1.0 the model resolves
inside-cell structure; above 1.0 it does not.

Secondary metrics are conventional within-cell rho / MAE.  Two auxiliary families
target the specific arm hypotheses: the map FFT band table (arms A1/A3 claim the
high-wavenumber attenuation shrinks) and the predicted/actual SD ratio (arm A2
claims the shrinkage toward the cell mean relaxes).

Everything primary is computed on **fold C only** -- folds A and B were consumed
by best-epoch selection, so they are reference columns, never decision inputs.

Sidesteps model_api by design
-----------------------------
Inference here rebuilds the encoder and net straight from each member's
``meta.json``.  That keeps this module off the concurrently-edited serving path,
and it is also the only route that scores a cond_v6 arm correctly today: the
serving encoder does not yet read ``meta["power_prior"]``, so it would build the
``prior_power`` channel with default constants instead of the fitted ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.fuel_types import FuelLibrary
from ..data.store import StoreReader
from .ab_paired import PairedDiff, paired_cell_bootstrap, paired_from_arrays
from .folds import (
    FOLD_NAMES, MIN_CELL_ROWS, UNCONTAMINATED_FOLD, FoldFrame, assign_folds,
    fold_frame, summarize_folds,
)
from .splits import SplitManifest

DEFAULT_STORE = "data/store"
DEFAULT_SPLITS = "data/splits"
DEFAULT_SPLIT = "S1"
#: Where the accumulated arm scores live.
DEFAULT_RESULTS = "data/reports/hires_ab_results.json"
#: The incumbent every arm is gated against.
DEFAULT_CHAMPION = "data/models/20260724_213535"

# --------------------------------------------------------------------------- #
# pre-registered binning — DO NOT tune after launch
# --------------------------------------------------------------------------- #
#: Delta bin edges per target, verbatim from report 20260725 section 3.1.  Frozen:
#: changing them after the arms ran would turn a pre-registered test into a
#: post-hoc one.
DELTA_BINS: dict[str, tuple[float, ...]] = {
    "cyclen":         (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    "f_r":            (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, math.inf),
    "f_q":            (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, math.inf),
    "cbc_max":        (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, math.inf),
    "max_assembly_burnup": (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, math.inf),
    "node_peak":      (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, math.inf),
    "map_cov":        (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, math.inf),
}
#: The four targets the decision rule reads.
PRIMARY_TARGETS: tuple[str, ...] = ("node_peak", "map_cov", "f_r", "cyclen")
#: Sign-hit level defining the effective resolution.
RESOLUTION_LEVEL = 0.75
#: Radial-wavenumber bands for the map spectrum (report section 3.6).
SPECTRAL_BANDS: tuple[tuple[float, float], ...] = (
    (0.00, 0.13), (0.13, 0.25), (0.25, 0.36), (0.36, 0.47), (0.47, 1.00))
#: Deterministic cap on pairs per cell (fold B has cells of ~2k rows == 2M pairs).
MAX_PAIRS_PER_CELL = 400_000
#: Section 8.2 M6: the extended no-regression gate.  The draft's default
#: ``("cyclen", "f_r")`` cannot see a per-cell MAP collapse at all, which is the
#: exact failure a flatness-first switch makes possible.
EXTENDED_GATE_TARGETS: tuple[str, ...] = ("cyclen", "f_r", "map_cov", "node_peak")


# --------------------------------------------------------------------------- #
# small statistics helpers (no scipy dependency in the hot loop)
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    ra, rb = _rankdata(a[ok]), _rankdata(b[ok])
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _cell_groups(cells: np.ndarray, min_rows: int = MIN_CELL_ROWS) -> list[np.ndarray]:
    groups = []
    for c in pd.unique(cells):
        if c == "":
            continue
        idx = np.flatnonzero(cells == c)
        if len(idx) >= min_rows:
            groups.append(idx)
    return groups


def cluster_bootstrap_ci(per_cell: Sequence[float], reps: int = 400,
                         seed: int = 0) -> tuple[float, float]:
    """95% CI of ONE arm's MEDIAN over cells, resampling CELLS (not rows).

    Rows inside a cell are not independent draws for a within-cell statistic, so
    a row bootstrap would understate the interval.

    **This is a descriptive interval, not a comparison.**  Two of these read side
    by side say nothing about the difference between the arms: they share the
    cell effects that dominate their width, so they overlap even when the paired
    difference is decisive, and they can separate on a shared shift that is not a
    difference at all.  Any A/B judgement must use
    :func:`~.ab_paired.paired_cell_bootstrap`, which resamples the same cells and
    returns a CI on ``arm - control``.  Re-exported here so the two live next to
    each other and the wrong one is harder to reach for by accident.
    """
    vals = np.asarray([v for v in per_cell if np.isfinite(v)], dtype=float)
    # reps <= 0 is the caller opting out (e.g. small provenance slices), not an error
    if len(vals) < 2 or int(reps) < 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(vals[rng.integers(0, len(vals), size=(reps, len(vals)))], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


# --------------------------------------------------------------------------- #
# metric 1 (PRIMARY): effective resolution
# --------------------------------------------------------------------------- #
def _pairs(idx: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(idx)
    total = n * (n - 1) // 2
    if total <= MAX_PAIRS_PER_CELL:
        i, j = np.triu_indices(n, k=1)
        return idx[i], idx[j]
    a = rng.integers(0, n, size=MAX_PAIRS_PER_CELL)
    b = rng.integers(0, n, size=MAX_PAIRS_PER_CELL)
    keep = a != b
    return idx[a[keep]], idx[b[keep]]


def resolution_curve(pred: np.ndarray, true: np.ndarray, cells: np.ndarray,
                     bins: Sequence[float], *, seed: int = 0,
                     min_rows: int = MIN_CELL_ROWS) -> dict[str, Any]:
    """Pairwise sign-hit rate per |true difference| bin, plus Delta75.

    Only WITHIN-cell pairs are formed: across cells the ordering is dominated by
    design-level scale, which the model gets right trivially and which the search
    never has to decide.
    """
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ai_all, bi_all = [], []
    for idx in _cell_groups(cells, min_rows):
        a, b = _pairs(idx, rng)
        ai_all.append(a)
        bi_all.append(b)
    if not ai_all:
        return {"bins": [], "n_pairs": 0, "delta75": None, "curve": []}
    ai = np.concatenate(ai_all)
    bi = np.concatenate(bi_all)
    dt = true[ai] - true[bi]
    dp = pred[ai] - pred[bi]
    ok = np.isfinite(dt) & np.isfinite(dp) & (dt != 0.0)
    dt, dp = dt[ok], dp[ok]
    adt = np.abs(dt)
    hit = (dt > 0) == (dp > 0)

    curve = []
    delta75 = None
    edges = list(bins)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (adt >= lo) & (adt < hi)
        n = int(sel.sum())
        rate = float(hit[sel].mean()) if n else float("nan")
        curve.append({"lo": lo, "hi": (None if math.isinf(hi) else hi),
                      "n": n, "hit": rate})
        # Delta75 == the LOWER edge of the first bin reaching the level (the gap
        # size at which ordering becomes reliable).
        if delta75 is None and n >= 30 and np.isfinite(rate) and rate >= RESOLUTION_LEVEL:
            delta75 = lo
    return {"bins": edges, "n_pairs": int(len(dt)), "delta75": delta75,
            "curve": curve}


def within_cell_sd(true: np.ndarray, cells: np.ndarray,
                   min_rows: int = MIN_CELL_ROWS) -> float:
    """Median over cells of the within-cell standard deviation of the TRUTH."""
    true = np.asarray(true, dtype=float)
    sds = [float(np.nanstd(true[idx])) for idx in _cell_groups(cells, min_rows)]
    sds = [s for s in sds if np.isfinite(s) and s > 0]
    return float(np.median(sds)) if sds else float("nan")


def effective_resolution(pred: np.ndarray, true: np.ndarray, cells: np.ndarray,
                         target: str, *, seed: int = 0) -> dict[str, Any]:
    """The primary metric: Delta75 and Delta75 / within-cell SD."""
    bins = DELTA_BINS.get(target)
    if bins is None:
        return {"target": target, "unsupported": True}
    curve = resolution_curve(pred, true, cells, bins, seed=seed)
    sd = within_cell_sd(true, cells)
    d75 = curve["delta75"]
    ratio = (float(d75) / sd) if (d75 is not None and np.isfinite(sd) and sd > 0) else None
    low = curve["curve"][0] if curve["curve"] else {}
    return {
        "target": target,
        "delta75": d75,
        "within_cell_sd": sd,
        "delta75_over_sd": ratio,
        "lowest_bin_hit": low.get("hit"),
        "lowest_bin_n": low.get("n"),
        "n_pairs": curve["n_pairs"],
        "curve": curve["curve"],
    }


# --------------------------------------------------------------------------- #
# metric 2 (SECONDARY): within-cell rho / MAE, and metric 4: SD ratio
# --------------------------------------------------------------------------- #
def within_cell_stats(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
                      bootstrap: int = 400, seed: int = 0,
                      min_rows: int = MIN_CELL_ROWS) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    groups = _cell_groups(cells, min_rows)
    rhos, maes, sd_ratios = [], [], []
    for idx in groups:
        # A cell whose truth column is entirely NaN (a target this dataset never
        # labelled) contributes nothing -- skip it rather than emit NaN slices.
        fin = np.isfinite(pred[idx]) & np.isfinite(true[idx])
        if int(fin.sum()) < 2:
            continue
        sel = idx[fin]
        rhos.append(spearman(pred[sel], true[sel]))
        maes.append(float(np.mean(np.abs(pred[sel] - true[sel]))))
        st, sp = float(np.std(true[sel])), float(np.std(pred[sel]))
        if np.isfinite(st) and st > 0:
            sd_ratios.append(sp / st)
    lo, hi = cluster_bootstrap_ci(rhos, reps=bootstrap, seed=seed)
    ok = np.isfinite(pred) & np.isfinite(true)
    finite_rhos = [r for r in rhos if np.isfinite(r)]
    return {
        "n_cells": len(rhos),
        "median_rho": float(np.median(finite_rhos)) if finite_rhos else float("nan"),
        "rho_ci95": [lo, hi],
        "median_cell_mae": float(np.median(maes)) if maes else float("nan"),
        # Ratio of predicted to actual within-cell spread.  < 1 means the model
        # shrinks toward the cell mean -- the compression arm A2 targets.
        "sd_ratio": float(np.median(sd_ratios)) if sd_ratios else float("nan"),
        "global_rho": spearman(pred, true),
        "mae": float(np.mean(np.abs(pred[ok] - true[ok]))) if ok.any() else float("nan"),
        "bias": float(np.mean((pred - true)[ok])) if ok.any() else float("nan"),
        "n": int(ok.sum()),
    }


# --------------------------------------------------------------------------- #
# metric 3 (AUXILIARY): map spatial spectrum
# --------------------------------------------------------------------------- #
def map_spectrum(pred_maps: np.ndarray, true_maps: np.ndarray) -> list[dict[str, Any]]:
    """Band table of predicted/actual power ratio and per-mode amplitude rho.

    Same protocol as report 20260725 section 3.6: keep the 69 valid slots, remove
    the slot mean (DC carries no shape), share one window between prediction and
    truth, 2-D FFT, bin by radial wavenumber.  Only RATIOS and correlations are
    interpreted -- the window makes absolute power meaningless.
    """
    p = np.asarray(pred_maps, dtype=float)
    t = np.asarray(true_maps, dtype=float)
    if p.ndim != 3 or p.shape != t.shape or p.shape[0] == 0:
        return []
    side = p.shape[-1]
    valid = np.isfinite(t) & np.isfinite(p)
    pf = np.where(valid, p, 0.0)
    tf = np.where(valid, t, 0.0)
    n = valid.reshape(len(p), -1).sum(axis=1).clip(min=1)
    pf = np.where(valid, pf - (pf.reshape(len(p), -1).sum(1) / n)[:, None, None], 0.0)
    tf = np.where(valid, tf - (tf.reshape(len(t), -1).sum(1) / n)[:, None, None], 0.0)
    fp = np.fft.fft2(pf, axes=(-2, -1))
    ft = np.fft.fft2(tf, axes=(-2, -1))

    k = np.fft.fftfreq(side)
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    total = float(np.mean(np.sum(np.abs(ft[:, kr > 0]) ** 2, axis=1))) or 1.0
    out = []
    for lo, hi in SPECTRAL_BANDS:
        m = (kr >= lo) & (kr < hi)
        if lo == 0.0:
            m = m & ~((k[:, None] == 0) & (k[None, :] == 0))
        if not m.any():
            continue
        pt = float(np.mean(np.sum(np.abs(ft[:, m]) ** 2, axis=1)))
        pp = float(np.mean(np.sum(np.abs(fp[:, m]) ** 2, axis=1)))
        out.append({
            "band": [lo, hi], "n_modes": int(m.sum()),
            "true_power_frac": pt / total,
            "power_ratio": (pp / pt) if pt > 0 else float("nan"),
            "mode_rho": spearman(np.abs(ft[:, m]).ravel(), np.abs(fp[:, m]).ravel()),
        })
    return out


# --------------------------------------------------------------------------- #
# metric 5 (TERTIARY): honest no-regression gate
# --------------------------------------------------------------------------- #
def no_regression_gate(new_pred: dict[str, np.ndarray],
                       old_pred: dict[str, np.ndarray],
                       truth: dict[str, np.ndarray], cells: np.ndarray, *,
                       targets: Sequence[str] = ("cyclen", "f_r"),
                       epsilon: float = 0.134,
                       min_rows: int = 30,
                       fr_guarded: bool | None = None) -> dict[str, Any]:
    """Per (cell, target) within-cell Spearman drop vs the incumbent.

    Mirrors ``gate_retrain*.json``: a candidate must not lose more than
    ``epsilon`` of within-cell ranking skill in ANY sufficiently-populated cell.
    An average that improves while one cell collapses is not an improvement --
    that cell is somebody's actual reload.

    **This is a PROXY, not the promotion gate.**  It runs on fold C with this
    module's cell keys; the authoritative gate is ``lpopt gate-promote``, which
    scores the curriculum's own ``val_by_cell`` / ``done_cells`` (72 checks) and
    additionally runs the legacy high-cyclen tail gate.  ``epsilon`` defaults to
    the 0.134 the production gate actually used (a family-wise 5% bar), because
    an artificially strict proxy manufactures false FAILs -- the first run of this
    harness used 0.05 and reported "no winner" for a slate whose best arm in fact
    passed the real gate with a 3x margin.

    **F_r DEFERRAL (user decision 2026-07-26).**  A proxy for the promotion gate
    must proxy the gate's SEMANTICS, not just its arithmetic, and the default
    family here (``cyclen``, ``f_r``) let ``f_r`` withhold promotion on exactly
    the corpus that motivated the deferral.  ``f_r`` is therefore still scored,
    still reported per (cell, target) and named in ``note``, but it does not enter
    ``worst_drop`` and cannot flip ``pass`` -- resolved through the SAME switch
    (:func:`..config.fr_guard_enforced`, ``[curriculum]
    gate_noreg_fr_guard_enabled``) that re-arms every other promotion surface.
    Every check carries ``enforced`` so a reader can see which ones had teeth.
    """
    from ..config import FR_GUARD_KNOB, fr_guard_enforced
    guarded = fr_guard_enforced(fr_guarded)
    report_only_names = () if guarded else ("f_r",)
    checks = []
    for idx in _cell_groups(cells, min_rows):
        cell = str(cells[idx[0]])
        for tgt in targets:
            if tgt not in new_pred or tgt not in old_pred or tgt not in truth:
                continue
            old = spearman(old_pred[tgt][idx], truth[tgt][idx])
            new = spearman(new_pred[tgt][idx], truth[tgt][idx])
            if not (np.isfinite(old) and np.isfinite(new)):
                continue
            checks.append({"cell": cell, "target": tgt, "n": int(len(idx)),
                           "enforced": tgt not in report_only_names,
                           "old_spearman": old, "new_spearman": new,
                           "drop": float(old - new)})
    worst = max((c["drop"] for c in checks if c["enforced"]), default=0.0)
    worst_any = max((c["drop"] for c in checks), default=0.0)
    scored = [t for t in targets if any(c["target"] == t for c in checks)]
    out = {"pass": bool(worst <= epsilon), "epsilon": epsilon,
           "worst_drop": float(worst), "worst_drop_any_axis": float(worst_any),
           "n_checks": sum(1 for c in checks if c["enforced"]),
           "guarded_targets": [t for t in scored if t not in report_only_names],
           "report_only_targets": [t for t in scored if t in report_only_names],
           "scored_targets": list(scored),
           "fr_guard": {"target": "f_r", "enforced": bool(guarded),
                        "knob": FR_GUARD_KNOB},
           "checks": sorted(checks, key=lambda c: -c["drop"])[:20]}
    ro = out["report_only_targets"]
    if ro:
        ro_worst = max(c["drop"] for c in checks if not c["enforced"])
        out["note"] = (
            "REPORT-ONLY axes (scored, NOT enforced): " + ", ".join(ro)
            + f" — their drop cannot fail this gate (worst {ro_worst:.4f} vs eps "
              f"{float(epsilon):.4f}). A pass does NOT mean " + "/".join(ro)
            + " was verified regression-free. "
              f"Set {FR_GUARD_KNOB} = true to enforce.")
    return out


__all__ = [
    "DELTA_BINS", "EXTENDED_GATE_TARGETS", "PRIMARY_TARGETS", "PairedDiff",
    "RESOLUTION_LEVEL", "SPECTRAL_BANDS", "cluster_bootstrap_ci",
    "effective_resolution", "map_spectrum", "no_regression_gate",
    "paired_cell_bootstrap", "paired_from_arrays", "resolution_curve", "spearman",
    "within_cell_sd", "within_cell_stats",
]
