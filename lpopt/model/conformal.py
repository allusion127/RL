"""Per-target split-conformal prediction intervals (plan sec. 4.4 addendum).

The champion's calibrated σ (:mod:`lpopt.model.calibrate`) is a *marginal* isotonic
map fit on the S1-val ensemble residuals; it is not a finite-sample coverage
guarantee, and on the honest per-cell holdouts its z-scores drift away from 1 for
several targets (F_q/F_r/cyclen are under-dispersed, CBC_max/pin-burnup over-).
This module adds a **split-conformal** wrapper that turns the served prediction
into an interval with a distribution-free marginal coverage guarantee on
exchangeable data.

It is **additive and report-only**: nothing here changes :meth:`predict` outputs
or any campaign consumer.  The fit runs on the champion's *honest per-cell
holdout* rows — the ``curriculum_val_by_cell`` record ids that were held out of the
champion's training (the same rows the honest no-regression gate scores) — so the
nonconformity scores measure the model's true out-of-sample error.

Design (mirrors :mod:`lpopt.model.cell_calibrate`):

* **Nonconformity.**  Per target the score is ``|pred - actual|`` (``abs``) or its
  σ-normalized form ``|pred - actual| / calibrated_σ`` (``norm``).  The score type
  is chosen **per target** at fit time by a 2-fold cell-split validation: the type
  whose held-out-cell coverage is valid (≥ nominal within tolerance) with the
  smaller interval width wins; ``norm`` adapts per row, ``abs`` is a flat band.

* **Quantile.**  ``q_hat`` is the finite-sample-corrected split-conformal quantile
  ``sorted(scores)[ceil((n+1)(1-α))-1]`` at ``α ∈ {0.10, 0.32}`` (90% / 68%).

* **Cells.**  Scores are grouped by the SERVE-recipe ``(feed, e_core-bin)`` key
  (:func:`cell_calibrate.cyclen_cell_key`), keyed identically at fit and serve so
  the lookup is parity-correct.  The bin width is 0.25 EFPD-enrichment (one
  curriculum band) so each honest-holdout cell carries ~30 rows; a cell (per
  target) with ``n >= min_cell`` gets its own ``q_hat``, else a **global**
  per-target fallback (pooled over all holdout rows) is served.

* **Serve.**  :meth:`PosValCnnBackend.predict_interval` centers the interval on the
  served :meth:`predict` mean (already cyclen-/pin-calibrated) and adds the fitted
  half-width — a pure additive accessor.

The artifact (``conformal.json``) is persisted next to the checkpoints as plain
JSON arrays/scalars (no pickled objects), exactly like ``calibration.json`` and
``cell_calibration.json``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cell_calibrate import cyclen_cell_key, _atomic_write_json, _finite

#: filename of the persisted artifact inside a model dir.
CONFORMAL_NAME = "conformal.json"
#: schema tag stamped into the artifact (bump on a breaking format change).
CONFORMAL_SCHEMA = "split_conformal_v1"
#: e_core bin width for the conformal cells — ONE curriculum band (0.25), so each
#: honest-holdout cell carries ~30 rows (>= ``DEFAULT_MIN_CELL``).  Deliberately
#: coarser than the cyclen-calibration 0.05 bin, whose cells would be too small to
#: fit a stable per-cell quantile on the holdout.
DEFAULT_BIN_WIDTH = 0.25
#: minimum finite (pred, actual) pairs in a (cell, target) to fit a per-cell
#: quantile; below this the per-target global fallback is served.
DEFAULT_MIN_CELL = 20
#: miscoverage levels — 90% and 68% two-sided intervals.
DEFAULT_ALPHAS: tuple[float, ...] = (0.10, 0.32)
#: coverage tolerance (finite-sample slack) when judging a score type "valid".
_COV_TOL = 0.02

#: conformal target NAME -> 7-column surrogate index (:data:`surrogate.TARGET_NAMES`
#: order: 0 F_r, 1 CBC_max, 2 F_q, 3 cyclen, 4 AO_abs, 5 max_assembly_burnup,
#: 6 max_pin_burnup).  ``max_assembly_burnup`` / ``discharge_burnup`` are excluded
#: (no served truth on the honest holdout — discharge is all-NaN there).
CONFORMAL_TARGETS: tuple[tuple[str, int], ...] = (
    ("f_r", 0),
    ("cbc_max", 1),
    ("f_q", 2),
    ("cyclen", 3),
    ("ao_abs", 4),
    ("max_pin_burnup", 6),
)


# --------------------------------------------------------------------------- #
# pure quantile / score primitives (unit-testable without an ensemble)
# --------------------------------------------------------------------------- #
def _akey(alpha: float) -> str:
    """Canonical JSON key for a miscoverage level (``0.1`` -> ``"0.1"``)."""
    return f"{float(alpha):g}"


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal quantile of nonconformity ``scores`` at level ``alpha``.

    Returns ``sorted(finite scores)[ceil((n+1)(1-alpha)) - 1]`` — the finite-sample
    correction that makes the marginal coverage ``>= 1 - alpha`` on exchangeable
    data.  When ``ceil((n+1)(1-alpha)) > n`` (too few points to certify the level)
    the quantile is ``+inf`` (the interval covers everything — a valid, if vacuous,
    conformal bound).  Empty input also yields ``+inf``.
    """
    s = np.sort(np.asarray([v for v in scores if math.isfinite(v)], dtype=float))
    n = s.size
    if n == 0:
        return math.inf
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if k > n:
        return math.inf
    return float(s[k - 1])


def nonconformity(pred: np.ndarray, actual: np.ndarray, sigma: np.ndarray | None,
                  kind: str) -> np.ndarray:
    """Per-row nonconformity score array for one target.

    ``abs`` -> ``|pred - actual|``; ``norm`` -> ``|pred - actual| / sigma`` (NaN
    where ``sigma <= 0`` or non-finite, so those rows drop out of the fit).
    """
    ae = np.abs(np.asarray(pred, dtype=float) - np.asarray(actual, dtype=float))
    if kind == "abs":
        return ae
    if kind == "norm":
        if sigma is None:
            return np.full_like(ae, np.nan)
        sig = np.asarray(sigma, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(sig > 0, ae / sig, np.nan)
        return out
    raise ValueError(f"unknown score kind {kind!r}")


def _fit_cell_qs(scores: np.ndarray, keys: Sequence[str], alphas: Sequence[float],
                 min_cell: int) -> tuple[dict[str, dict], dict[str, float], int]:
    """Per-cell + global conformal quantiles for one target's score vector.

    Returns ``(cells, global_q, n_total)`` where ``cells`` maps
    ``cell_key -> {"n": int, "q": {alpha_key: q_hat}}`` for every cell with
    ``>= min_cell`` finite scores, ``global_q`` maps ``alpha_key -> q_hat`` over ALL
    finite scores (the fallback), and ``n_total`` is the finite-score count.
    """
    scores = np.asarray(scores, dtype=float)
    keys = list(keys)
    finite = np.isfinite(scores)
    n_total = int(finite.sum())
    global_q = {_akey(a): conformal_quantile(scores[finite], a) for a in alphas}
    by_cell: dict[str, list[float]] = {}
    for s, k, ok in zip(scores, keys, finite):
        if ok:
            by_cell.setdefault(k, []).append(float(s))
    cells: dict[str, dict] = {}
    for k, sc in sorted(by_cell.items()):
        if len(sc) < int(min_cell):
            continue
        cells[k] = {"n": len(sc),
                    "q": {_akey(a): conformal_quantile(sc, a) for a in alphas}}
    return cells, global_q, n_total


def _lookup_q(entry: Mapping[str, Any], cell_key: str, alpha: float) -> tuple[float, bool]:
    """``(q_hat, from_cell)`` for one target entry + cell key at ``alpha``.

    Uses the per-cell quantile when the cell was fitted for this alpha, else the
    per-target global fallback (``from_cell=False``).
    """
    ak = _akey(alpha)
    cell = entry.get("cells", {}).get(cell_key)
    if cell is not None and ak in cell.get("q", {}):
        return float(cell["q"][ak]), True
    return float(entry.get("global", {}).get(ak, math.inf)), False


def halfwidths(entry: Mapping[str, Any], cell_keys: Sequence[str],
               sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-row interval half-widths + ``from_cell`` flags for one target.

    ``abs`` -> the (cell/global) ``q_hat`` directly; ``norm`` -> ``q_hat * sigma``.
    """
    kind = entry.get("score_type", "abs")
    sigma = np.asarray(sigma, dtype=float)
    n = len(cell_keys)
    hw = np.empty(n, dtype=float)
    from_cell = np.zeros(n, dtype=bool)
    for i, k in enumerate(cell_keys):
        q, fc = _lookup_q(entry, k, alpha)
        from_cell[i] = fc
        hw[i] = q if kind == "abs" else q * sigma[i]
    return hw, from_cell


# --------------------------------------------------------------------------- #
# 2-fold cell-split validation + per-target score-type selection
# --------------------------------------------------------------------------- #
def _cell_folds(cells: Sequence[str], k: int, seed: int) -> list[set[str]]:
    """Deterministic k-way partition of the unique cell keys."""
    uniq = sorted(set(cells))
    idx = np.arange(len(uniq))
    np.random.default_rng(seed).shuffle(idx)
    return [{uniq[i] for i in idx[f::k]} for f in range(k)]


def kfold_cell_coverage(pred: np.ndarray, actual: np.ndarray, sigma: np.ndarray,
                        keys: Sequence[str], kind: str, alphas: Sequence[float],
                        *, min_cell: int, k: int = 2, seed: int = 0
                        ) -> dict[str, dict[str, float]] | None:
    """Held-out-cell coverage + mean width for one target under score ``kind``.

    Fits per-cell + global quantiles on ``k-1`` folds of the cells and scores the
    held-out fold's rows: because the held-out cells are absent from the fit set,
    they exercise the **global fallback** — exactly the "unseen cell" path whose
    generalization is the thing worth validating (per-cell coverage on a fitted
    cell is conservative by construction).  Coverage is pooled row-weighted over
    the folds.  Returns ``{"cov": {ak: cov}, "width": {ak: mean width}, "n": rows}``
    or ``None`` when the ``kind`` is unusable (no finite scores — e.g. ``norm`` with
    degenerate sigma).
    """
    pred = np.asarray(pred, float); actual = np.asarray(actual, float)
    sigma = np.asarray(sigma, float); keys = list(keys)
    sc = nonconformity(pred, actual, sigma, kind)
    truth_ok = np.isfinite(pred) & np.isfinite(actual)
    if not np.isfinite(sc[truth_ok]).any():
        return None
    folds = _cell_folds(keys, k, seed)
    hits = {_akey(a): 0.0 for a in alphas}
    widths = {_akey(a): 0.0 for a in alphas}
    n_eval = 0
    for f in range(k):
        eval_cells = folds[f]
        fit_mask = np.array([kk not in eval_cells for kk in keys]) & np.isfinite(sc)
        cells_q, global_q, _ = _fit_cell_qs(sc[fit_mask],
                                            [kk for kk, m in zip(keys, fit_mask) if m],
                                            alphas, min_cell)
        entry = {"score_type": kind, "cells": cells_q, "global": global_q}
        ev = np.array([kk in eval_cells for kk in keys]) & truth_ok
        if not ev.any():
            continue
        err = np.abs(pred[ev] - actual[ev])
        ev_keys = [kk for kk, m in zip(keys, ev) if m]
        for a in alphas:
            hw, _fc = halfwidths(entry, ev_keys, sigma[ev], a)
            hits[_akey(a)] += float(np.sum(err <= hw))
            widths[_akey(a)] += float(np.sum(2.0 * hw))
        n_eval += int(ev.sum())
    if n_eval == 0:
        return None
    return {
        "cov": {ak: hits[ak] / n_eval for ak in hits},
        "width": {ak: widths[ak] / n_eval for ak in widths},
        "n": n_eval,
    }


def _score_type_valid(metrics: Mapping[str, Any], alphas: Sequence[float]) -> bool:
    cov = metrics["cov"]
    return all(cov[_akey(a)] >= (1.0 - a) - _COV_TOL for a in alphas)


def select_score_type(val_abs: Mapping | None, val_norm: Mapping | None,
                      alphas: Sequence[float]) -> str:
    """Pick ``abs`` vs ``norm`` for one target from their validation metrics.

    Prefer a *valid* type (held-out coverage ≥ nominal within tolerance at every
    alpha); among valid types pick the tighter one (smaller width at the primary
    alpha); when neither is valid pick the higher primary-alpha coverage.  ``abs``
    is the default when ``norm`` is unavailable.
    """
    a0 = _akey(alphas[0])
    cands = [("abs", val_abs), ("norm", val_norm)]
    avail = [(k, m) for k, m in cands if m is not None]
    if not avail:
        return "abs"
    valids = [(k, m) for k, m in avail if _score_type_valid(m, alphas)]
    if valids:
        return min(valids, key=lambda km: km[1]["width"][a0])[0]
    return max(avail, key=lambda km: km[1]["cov"][a0])[0]


# --------------------------------------------------------------------------- #
# load / apply
# --------------------------------------------------------------------------- #
def load_conformal(path: str | Path) -> dict:
    """Load a ``conformal.json`` artifact (full dict incl. metadata)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conformal_targets(artifact: Mapping[str, Any] | None) -> dict[str, dict]:
    """The ``{target_name: entry}`` map from an artifact (empty when absent)."""
    if not artifact:
        return {}
    pt = artifact.get("per_target")
    return dict(pt) if isinstance(pt, Mapping) else {}


def interval_arrays(mean: np.ndarray, sigma: np.ndarray, cell_keys: Sequence[str],
                    artifact: Mapping[str, Any], alpha: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``(lower, upper, halfwidth, from_cell)`` in the 7-column surrogate
    layout for a served batch.

    ``mean`` / ``sigma`` are ``[N, 7]`` (``predict().mean`` and
    ``predict().calibrated_std``).  Columns without a fitted conformal target
    (``max_assembly_burnup``, and any target the artifact omits) stay NaN
    (``from_cell`` False), so a consumer reads a finite bound only where the
    interval is real.  Purely additive — the caller's arrays are never mutated.
    """
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n, ncol = mean.shape
    lower = np.full((n, ncol), np.nan)
    upper = np.full((n, ncol), np.nan)
    hw_out = np.full((n, ncol), np.nan)
    from_cell = np.zeros((n, ncol), dtype=bool)
    per_target = conformal_targets(artifact)
    for name, col in CONFORMAL_TARGETS:
        entry = per_target.get(name)
        if entry is None:
            continue
        hw, fc = halfwidths(entry, cell_keys, sigma[:, col], alpha)
        m = mean[:, col]
        good = np.isfinite(m) & np.isfinite(hw)
        hw_out[good, col] = hw[good]
        lower[good, col] = m[good] - hw[good]
        upper[good, col] = m[good] + hw[good]
        from_cell[:, col] = fc & good
    return lower, upper, hw_out, from_cell


# --------------------------------------------------------------------------- #
# fit driver (serve-path scoring on the honest per-cell holdout)
# --------------------------------------------------------------------------- #
def fit_conformal(
    model_dir: str | Path,
    store_dir: str | Path,
    splits_dir: str | Path,
    *,
    split: str | None = None,
    library_id: str = "ga80",
    bin_width: float = DEFAULT_BIN_WIDTH,
    min_cell: int = DEFAULT_MIN_CELL,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    device: str = "cpu",
    batch_size: int = 512,
    val_folds: int = 2,
    val_seed: int = 0,
    write: bool = True,
    out_name: str = CONFORMAL_NAME,
) -> dict:
    """Fit + persist per-target split-conformal intervals for a champion.

    Scores the champion's HONEST per-cell holdout rows (the split's
    ``curriculum_val_by_cell`` record ids — held out of the champion's training)
    through the serve path, computes per-target nonconformity, chooses ``abs`` vs
    ``norm`` per target via a ``val_folds``-way cell-split coverage validation, and
    fits per-(feed, e_core-bin) cell quantiles (``n >= min_cell``) with a per-target
    global fallback at every ``alpha``.  Returns the artifact dict; when ``write`` it
    is also atomically written to ``model_dir/out_name``.

    Leakage guarantee: only ``curriculum_val_by_cell`` ids enter the fit — the exact
    rows the champion never trained on (asserted below).
    """
    from ..data.schema import unpack_pattern  # noqa: F401  (import validated here)
    from ..data.store import StoreReader
    from ..vendor.masterrl.domain import CaseKey  # noqa: F401
    from .model_api import PosValCnnBackend
    from .splits import SplitManifest

    model_dir = Path(model_dir)
    reader = StoreReader(store_dir)
    df = reader.records
    indexed = df.drop_duplicates("record_id").set_index("record_id")

    # -- resolve the split the champion trained on ------------------------- #
    if split is None:
        metas = sorted(model_dir.glob("member_*/meta.json"))
        split = "S1"
        if metas:
            split = str(json.loads(metas[0].read_text(encoding="utf-8"))
                        .get("split", "S1"))
    manifest = SplitManifest.from_json(Path(splits_dir) / f"{split}.json")
    val_by_cell = manifest.groups.get("curriculum_val_by_cell", {})
    val_ids = [rid for ids in val_by_cell.values() for rid in ids
               if rid in indexed.index]
    train_ids = set(manifest.record_ids("train"))

    sub = indexed.loc[val_ids].reset_index()
    sub = sub[sub["converged"] == True].reset_index(drop=True)  # noqa: E712

    # -- serve-path backend (conformal bin width installed for keying) ----- #
    backend = PosValCnnBackend.from_dir(
        model_dir, store_dir=store_dir, library_id=library_id, device=device)
    backend.conformal_bin_width = float(bin_width)

    mean, sigma, keys = _score_holdout(backend, sub, unpack_pattern, CaseKey,
                                       bin_width, batch_size)

    per_target: dict[str, dict] = {}
    for name, col in CONFORMAL_TARGETS:
        actual = np.asarray([_finite(v) for v in sub[name]], dtype=float)
        pred = mean[:, col]
        sig = sigma[:, col]
        # per-target score-type choice from held-out-cell validation
        val_abs = kfold_cell_coverage(pred, actual, sig, keys, "abs", alphas,
                                       min_cell=min_cell, k=val_folds, seed=val_seed)
        val_norm = kfold_cell_coverage(pred, actual, sig, keys, "norm", alphas,
                                       min_cell=min_cell, k=val_folds, seed=val_seed)
        kind = select_score_type(val_abs, val_norm, alphas)
        # final fit on ALL holdout rows with the chosen score type
        sc = nonconformity(pred, actual, sig, kind)
        cells_q, global_q, n_used = _fit_cell_qs(sc, keys, alphas, min_cell)
        per_target[name] = {
            "surrogate_col": col,
            "score_type": kind,
            "n": n_used,
            "n_cells": len(cells_q),
            "global": global_q,
            "cells": cells_q,
            "validation": {
                "selected": kind,
                "abs": val_abs,
                "norm": val_norm,
            },
        }

    artifact = {
        "schema": CONFORMAL_SCHEMA,
        "model_dir": str(model_dir),
        "split": split,
        "library_id": library_id,
        "bin_width": float(bin_width),
        "min_cell": int(min_cell),
        "alphas": [float(a) for a in alphas],
        "val_folds": int(val_folds),
        "targets": [n for n, _ in CONFORMAL_TARGETS],
        "n_holdout": int(len(sub)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_target": per_target,
    }
    # leakage guard: the fit must ONLY have touched honest holdout rows.
    assert not (set(sub["record_id"].astype(str)) & train_ids), \
        "conformal fit set intersects the champion's train fold"

    if write:
        _atomic_write_json(model_dir / out_name, artifact)
    return artifact


def _score_holdout(backend: Any, sub: Any, unpack_pattern, CaseKey,
                   bin_width: float, batch_size: int):
    """Serve-path ``predict`` over holdout store rows -> ``(mean, sigma, keys)``.

    Predictions come through the exact SERVE path (:meth:`PosValCnnBackend.predict`,
    with whatever cyclen/pin calibration the champion ships), so the intervals are
    centered on the served mean.  Cell keys use the serve-recipe e_core
    (:meth:`PosValCnnBackend.cyclen_e_core`, the EFFECTIVE library — corrected
    2026-07-29 debug-panel; it previously resolved against the CONFIGURED library,
    which sent every rerouted pattern, e.g. paramA under a ga80 campaign, into the
    ``ebin=None`` cell) at the conformal bin width — byte-identical to the
    serve-time :meth:`predict_interval` key, which is the only property that matters
    here and which the change preserves because both sides call the same method.
    """
    pats = [unpack_pattern(str(p)) for p in sub["pattern"].tolist()]
    cases = [CaseKey(str(pr), int(fd)) for pr, fd in zip(sub["case_pair"], sub["feed"])]
    n = len(pats)
    mean = np.empty((n, 7)); sigma = np.empty((n, 7))
    for s in range(0, n, int(batch_size)):
        pred = backend.predict(pats[s:s + batch_size], cases[s:s + batch_size], 0.0)
        mean[s:s + batch_size] = pred.mean
        sigma[s:s + batch_size] = pred.calibrated_std
    keys: list[str] = []
    for pat, case in zip(pats, cases):
        e_core, _ = backend.cyclen_e_core(pat)
        keys.append(cyclen_cell_key(int(case.feed), e_core, float(bin_width)))
    return mean, sigma, keys


# --------------------------------------------------------------------------- #
# coverage report (validation deliverable)
# --------------------------------------------------------------------------- #
def coverage_table(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the fit's held-out-cell validation into a per-target coverage table.

    One row per target: the SELECTED score type and its 2-fold held-out-cell
    coverage + mean interval width at each alpha (plus the rejected type's coverage
    for the audit trail).
    """
    alphas = artifact.get("alphas", DEFAULT_ALPHAS)
    rows: list[dict[str, Any]] = []
    for name, entry in conformal_targets(artifact).items():
        val = entry.get("validation", {})
        sel = entry.get("score_type")
        m = val.get(sel) or {}
        row: dict[str, Any] = {"target": name, "score_type": sel,
                               "n_cells": entry.get("n_cells"), "n": entry.get("n")}
        for a in alphas:
            ak = _akey(a)
            row[f"cov@{int(round((1 - a) * 100))}"] = (m.get("cov", {}) or {}).get(ak)
            row[f"width@{int(round((1 - a) * 100))}"] = (m.get("width", {}) or {}).get(ak)
        rows.append(row)
    return rows


__all__ = [
    "CONFORMAL_NAME",
    "CONFORMAL_SCHEMA",
    "CONFORMAL_TARGETS",
    "DEFAULT_ALPHAS",
    "DEFAULT_BIN_WIDTH",
    "DEFAULT_MIN_CELL",
    "conformal_quantile",
    "conformal_targets",
    "coverage_table",
    "fit_conformal",
    "halfwidths",
    "interval_arrays",
    "kfold_cell_coverage",
    "load_conformal",
    "nonconformity",
    "select_score_type",
]
