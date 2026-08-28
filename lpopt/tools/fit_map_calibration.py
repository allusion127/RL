"""Fit ``data/store/map_calibration.json`` — the map-head level calibration.

``python -m lpopt.tools.fit_map_calibration [--model-dir ...] [--dry-run]``

Program §2.1 makes this artifact a **precondition** for running the flatness
objective.  Without it the F_r safety gate's bias correction is inert (the gate
holds flat at 1.70) and the acquisition consumes the map head's raw levels,
whose fold-C optimism was measured at −0.147 (``node_peak``) / −0.058
(``map_cov``) — while the only pessimism available to it is
``risk_z x ensemble spread``, an epistemic disagreement statistic that cannot
express a bias every member shares.

What is measured, and on which rows
-----------------------------------
For every eligible row we compare the champion's SERVE-path prediction against
the stored MASTER label, per (feed, e_core-bin) cell:

===========  ==========================  ==================================
target       predicted by                compared against
===========  ==========================  ==================================
``node_peak``  ``predict_map_flatness``    the ``node_peak`` record column
``map_cov``    ``predict_map_flatness``    the ``map_cov`` record column
``f_r``        ``predict``, column 0       the ``f_r`` record column
===========  ==========================  ==================================

``node_peak`` / ``map_cov`` are re-derived from the CANONICAL definition
(:mod:`..data.flatness`, multiplicity-weighted) inside
``predict_map_flatness`` — there is no second copy of the formula here, and the
stored label columns were written by the same module's :func:`record_flatness`.
``f_r`` is served exactly as the campaign serves it (i.e. WITH the champion's
own ``f_r_calibration.json`` applied), because the D1 gate is applied to that
served number and its residual is what the gate must guard.

**The slice is fold C** (:mod:`..model.folds`): converged + valid store rows in
NEITHER the champion's frozen ``train_ids`` NOR its ``val_ids``, i.e. rows that
provably did not exist when the split was written.  This is the only
uncontaminated fold — folds A and B were both consumed by best-epoch selection
and sigma fitting, so a bias measured there is optimistic by construction, and
train rows are the ones ``f_r_calibration.json`` was itself fit on.  Measuring a
*level* correction on a contaminated fold is how a calibration ends up
certifying its own training error.

Rows are further required to carry BOTH flatness columns and a finite ``f_r``,
and are predicted under **their own** ``library_id`` (a foreign-library row
resolves its fed types against the wrong roster, so both its prediction and its
e_core bin would be meaningless).  ``--production-only`` drops model-PROPOSED
rows (``generator``/``campaign`` starting ``alsearch``); the default keeps them
and records the proposal share per cell, because fold C is ~1/3 proposed and
dropping them costs most cells their row floor.

What is written
---------------
``<store-dir>/map_calibration.json`` — schema, provenance (model dir + id +
member fingerprint, split, fold, slice, row/cell counts, date), a ``global``
pooled fallback block, and one entry per cell with >= ``--min-rows`` rows:

* ``bias``        ``median(pred - actual)`` — NEGATIVE means optimistic;
* ``sigma``       robust (MAD-scaled) SD of that residual;
* ``sigma_ens``   RMS of the per-row ensemble spread the model reported;
* ``sigma_extra`` ``sqrt(max(0, sigma^2 - sigma_ens^2))`` — the dispersion the
  ensemble does NOT have, which is what the UCB is missing;
* ``fr_bias`` / ``fr_sigma`` — the derived D1 gate keys (see
  :mod:`..data.map_calibration` for the sign convention and the never-loosen
  clamp).

The fit is deterministic: medians and MADs over a stably-ordered row set, no
RNG, no bootstrap.  Re-running on an unchanged store + champion reproduces the
file byte for byte apart from the ``fitted_at`` stamp.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.map_calibration import (
    ARTIFACT_NAME, ARTIFACT_SCHEMA, GATE_K, MIN_CELL_ROWS, TARGETS,
    FINGERPRINT_SCHEMA,
    TargetCalibration, gate_shift, model_fingerprint, model_id,
)
from ..data.store import StoreReader
from ..model.cell_calibrate import DEFAULT_BIN_WIDTH, cyclen_cell_key

#: Default champion (cond_schema v6) — the serving pointer of program §P0-C.
DEFAULT_MODEL_DIR = "data/models/20260725_063351"
#: Fold the fit is allowed to read.  Fold C is the only uncontaminated slice.
HONEST_FOLD = "C"
#: Surrogate column of ``f_r`` in the 7-column layout.
FR_COL = 0
#: 1 / Phi^-1(0.75) — MAD -> SD for a Gaussian.
_MAD_TO_SD = 1.4826


class MapCalibrationFitError(RuntimeError):
    """The honest slice cannot support a fit (refuse rather than emit a stub)."""


# --------------------------------------------------------------------------- #
# the pure statistics (no torch, no store — unit testable on arrays)
# --------------------------------------------------------------------------- #
def robust_sd(residual: np.ndarray) -> float:
    """MAD-scaled SD of ``residual`` (0.0 for < 2 finite values).

    Robust rather than the plain SD because a handful of non-converged-adjacent
    outliers in a 12-row cell would otherwise set the pessimism for the whole
    cell.  The plain SD is reported alongside it so the difference is visible.
    """
    r = np.asarray(residual, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    mad = float(np.median(np.abs(r - float(np.median(r)))))
    return float(_MAD_TO_SD * mad)


def fit_target(pred: Sequence[float], actual: Sequence[float],
               sigma_ens: Sequence[float] | None = None, *,
               min_rows: int = MIN_CELL_ROWS) -> dict[str, Any] | None:
    """One (cell, target) calibration, or ``None`` below the row floor.

    ``bias`` is ``median(pred - actual)``; ``sigma`` is :func:`robust_sd` of the
    same residual; ``sigma_ens`` is the RMS of the model's own per-row spread;
    ``sigma_extra = sqrt(max(0, sigma^2 - sigma_ens^2))`` is the dispersion the
    ensemble is missing.  ``sigma_extra`` is clamped at 0, so a cell where the
    ensemble is already over-dispersed produces an exactly inert correction
    rather than a negative variance.

    Deterministic — order-independent medians, no resampling.
    """
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    n0 = min(p.size, a.size)
    p, a = p[:n0], a[:n0]
    s = (np.full(n0, np.nan) if sigma_ens is None
         else np.asarray(sigma_ens, dtype=float)[:n0])
    finite = np.isfinite(p) & np.isfinite(a)
    if int(finite.sum()) < int(min_rows):
        return None
    p, a, s = p[finite], a[finite], s[finite]
    resid = p - a
    bias = float(np.median(resid))
    sigma = robust_sd(resid)
    s_ok = s[np.isfinite(s)]
    sigma_ens_rms = float(np.sqrt(np.mean(s_ok * s_ok))) if s_ok.size else 0.0
    sigma_extra = float(np.sqrt(max(0.0, sigma * sigma - sigma_ens_rms ** 2)))
    mae_before = float(np.median(np.abs(resid)))
    mae_after = float(np.median(np.abs(resid - bias)))
    out = TargetCalibration(bias=bias, sigma=sigma, sigma_ens=sigma_ens_rms,
                            sigma_extra=sigma_extra, n=int(p.size)).as_dict()
    out.update({
        "sd": round(float(np.std(resid, ddof=1)) if p.size > 1 else 0.0, 6),
        "mae_before": round(mae_before, 6),
        "mae_after": round(mae_after, 6),
        "pred_median": round(float(np.median(p)), 6),
        "actual_median": round(float(np.median(a)), 6),
    })
    return out


def gate_keys(f_r_entry: dict[str, Any] | None) -> dict[str, float]:
    """The derived D1 gate keys for one cell/global block (empty when no ``f_r``)."""
    if not f_r_entry:
        return {}
    return {"fr_bias": round(gate_shift(f_r_entry.get("bias")), 6),
            "fr_sigma": round(float(f_r_entry.get("sigma_extra") or 0.0), 6)}


# --------------------------------------------------------------------------- #
# the honest slice
# --------------------------------------------------------------------------- #
@dataclass
class SliceReport:
    """Why each row was kept or dropped — printed and embedded in the artifact."""

    n_store: int = 0
    n_fold: int = 0
    n_converged_valid: int = 0
    n_labelled: int = 0
    n_proposal: int = 0
    libraries: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"n_store_rows": self.n_store, "n_fold_rows": self.n_fold,
                "n_converged_valid": self.n_converged_valid,
                "n_used": self.n_labelled, "n_proposal": self.n_proposal,
                "libraries": dict(sorted(self.libraries.items()))}


def honest_slice(records: pd.DataFrame, manifest: Any, *,
                 fold: str = HONEST_FOLD, production_only: bool = False,
                 bin_width: float = DEFAULT_BIN_WIDTH
                 ) -> tuple[pd.DataFrame, SliceReport]:
    """Fold-``fold`` rows eligible for the fit, plus the attrition report.

    Eligibility: the fold predicate (see :mod:`..model.folds`), ``converged``,
    ``valid``, finite ``node_peak`` / ``map_cov`` / ``f_r``, a resolvable
    ``(feed, e_core)`` cell key, and a known ``library_id``.  ``production_only``
    additionally drops model-proposed rows.
    """
    from ..model import folds as _folds

    rep = SliceReport(n_store=int(len(records)))
    labels = _folds.assign_folds(records, manifest)
    sub = records.loc[(labels == fold).to_numpy()].reset_index(drop=True)
    rep.n_fold = int(len(sub))
    for col in ("converged", "valid"):
        if col in sub.columns:
            sub = sub[sub[col].fillna(False).astype(bool)]
    sub = sub.reset_index(drop=True)
    rep.n_converged_valid = int(len(sub))
    for col in ("node_peak", "map_cov", "f_r"):
        if col not in sub.columns:
            return sub.iloc[0:0], rep
        sub = sub[pd.to_numeric(sub[col], errors="coerce").notna()]
    sub = sub.reset_index(drop=True)
    if not len(sub):
        return sub, rep

    proposal = _folds.proposal_mask(sub)
    rep.n_proposal = int(proposal.sum())
    if production_only:
        sub = sub.loc[~proposal].reset_index(drop=True)

    feed = pd.to_numeric(sub["feed"], errors="coerce").to_numpy(dtype=float)
    ecore = pd.to_numeric(sub["e_core"], errors="coerce").to_numpy(dtype=float)
    keys = [
        cyclen_cell_key(int(f), float(e) if math.isfinite(e) else None, bin_width)
        if math.isfinite(f) else ""
        for f, e in zip(feed, ecore)
    ]
    sub = sub.assign(_cell=keys)
    sub = sub[sub["_cell"] != ""].reset_index(drop=True)
    if "library_id" in sub.columns:
        sub = sub[sub["library_id"].notna()].reset_index(drop=True)
        rep.libraries = {str(k): int(v)
                         for k, v in sub["library_id"].astype(str)
                         .value_counts().items()}
    sub = sub.assign(_proposal=_folds.proposal_mask(sub))
    rep.n_labelled = int(len(sub))
    rep.n_proposal = int(sub["_proposal"].sum())
    return sub, rep


# --------------------------------------------------------------------------- #
# serve-path predictions
# --------------------------------------------------------------------------- #
def predict_slice(slice_df: pd.DataFrame, model_dir: str | Path,
                  store_dir: str | Path, *, device: str = "cpu",
                  batch_size: int = 128, log: Any = print) -> dict[str, np.ndarray]:
    """Serve-path predictions for every row, grouped by the row's OWN library.

    Returns ``{target: (mean[N], sigma[N])}`` flattened into
    ``{f"{target}_mean", f"{target}_sigma"}`` arrays aligned with ``slice_df``.
    One backend is built per distinct ``library_id`` because the encoder resolves
    a pattern's fed assemblies against that library's roster — predicting a
    ``paramA`` row through a ``ga80`` backend is not a worse prediction, it is a
    different physical question.
    """
    from ..data.schema import unpack_pattern
    from ..model.model_api import PosValCnnBackend
    from ..vendor.masterrl.domain import CaseKey

    n = len(slice_df)
    out = {f"{t}_{k}": np.full(n, np.nan)
           for t in TARGETS for k in ("mean", "sigma")}
    if not n:
        return out
    libs = (slice_df["library_id"].astype(str).to_numpy()
            if "library_id" in slice_df.columns else np.array(["ga80"] * n))
    patterns = [unpack_pattern(str(p)) for p in slice_df["pattern"]]
    cases = [CaseKey(pair=str(cp), feed=int(f))
             for cp, f in zip(slice_df["case_pair"], slice_df["feed"])]

    for lib in sorted(set(libs.tolist())):
        idx = np.flatnonzero(libs == lib)
        log(f"[fit_map_calibration] library {lib}: {idx.size} rows")
        backend = PosValCnnBackend.from_dir(
            model_dir, store_dir=store_dir, library_id=str(lib), device=device)
        for start in range(0, idx.size, int(batch_size)):
            take = idx[start:start + int(batch_size)]
            bp = [patterns[i] for i in take]
            bc = [cases[i] for i in take]
            pred = backend.predict(bp, bc, 0.0)
            out["f_r_mean"][take] = np.asarray(pred.mean, dtype=float)[:, FR_COL]
            out["f_r_sigma"][take] = np.asarray(
                pred.calibrated_std, dtype=float)[:, FR_COL]
            pk_m, pk_s, cv_m, cv_s = backend.predict_map_flatness(bp, bc, 0.0)
            out["node_peak_mean"][take] = np.asarray(pk_m, dtype=float)
            out["node_peak_sigma"][take] = np.asarray(pk_s, dtype=float)
            out["map_cov_mean"][take] = np.asarray(cv_m, dtype=float)
            out["map_cov_sigma"][take] = np.asarray(cv_s, dtype=float)
            if (start // int(batch_size)) % 4 == 0:
                log(f"[fit_map_calibration]   {min(start + int(batch_size), idx.size)}"
                    f"/{idx.size}")
    return out


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #
def build_artifact(slice_df: pd.DataFrame, preds: dict[str, np.ndarray], *,
                   model_dir: str | Path, store_dir: str | Path,
                   split: str, fold: str, slice_report: SliceReport,
                   min_rows: int = MIN_CELL_ROWS,
                   bin_width: float = DEFAULT_BIN_WIDTH,
                   production_only: bool = False) -> dict[str, Any]:
    """Assemble the full ``map_calibration.json`` document (pure, no I/O)."""
    cells_out: dict[str, dict[str, Any]] = {}
    n_seen = 0
    if len(slice_df):
        order = sorted(set(slice_df["_cell"].astype(str).tolist()))
        cell_col = slice_df["_cell"].astype(str).to_numpy()
        for cell in order:
            n_seen += 1
            idx = np.flatnonzero(cell_col == cell)
            entry: dict[str, Any] = {}
            for target in TARGETS:
                fit = fit_target(preds[f"{target}_mean"][idx],
                                 pd.to_numeric(slice_df[target], errors="coerce")
                                 .to_numpy(dtype=float)[idx],
                                 preds[f"{target}_sigma"][idx],
                                 min_rows=min_rows)
                if fit is not None:
                    entry[target] = fit
            if not entry:
                continue
            entry["n"] = int(idx.size)
            entry["n_proposal"] = int(np.asarray(
                slice_df["_proposal"].to_numpy(dtype=bool))[idx].sum())
            if "library_id" in slice_df.columns:
                libs = slice_df["library_id"].astype(str).to_numpy()[idx]
                entry["libraries"] = {str(k): int(v) for k, v in
                                      pd.Series(libs).value_counts().items()}
            entry.update(gate_keys(entry.get("f_r")))
            cells_out[cell] = entry

    # -- global fallback: the SAME statistic pooled over every eligible row --- #
    global_out: dict[str, Any] = {}
    for target in TARGETS:
        fit = fit_target(preds[f"{target}_mean"],
                         pd.to_numeric(slice_df[target], errors="coerce")
                         .to_numpy(dtype=float) if len(slice_df) else np.zeros(0),
                         preds[f"{target}_sigma"], min_rows=min_rows)
        if fit is not None:
            global_out[target] = fit
    global_out.update(gate_keys(global_out.get("f_r")))

    return {
        "schema": ARTIFACT_SCHEMA,
        "bias_convention": ("bias = median(pred - actual); NEGATIVE means the "
                            "head UNDER-predicts (optimistic). De-bias by "
                            "SUBTRACTING it: corrected = pred - bias."),
        "sigma_convention": ("sigma = MAD-scaled SD of (pred - actual); "
                            "sigma_ens = RMS of the model's own ensemble spread; "
                            "sigma_extra = sqrt(max(0, sigma^2 - sigma_ens^2)) is "
                            "the dispersion the ensemble does NOT have. The "
                            "calibrated UCB sigma is "
                            "sqrt(sigma_ens_row^2 + sigma_extra^2)."),
        "gate": {
            "rule": "F_r safety gate = fr_limit - fr_bias - k*fr_sigma (D1)",
            "k": GATE_K,
            "fr_bias": "max(0, -f_r.bias) — the correction may only TIGHTEN",
            "fr_sigma": "f_r.sigma_extra",
        },
        "fit": {
            "model_dir": str(model_dir),
            "model_id": model_id(model_dir),
            "model_fingerprint": model_fingerprint(model_dir),
            # which fingerprint scheme produced it — v2 covers the members'
            # ``model.pt`` weights, not only their metadata.
            "model_fingerprint_schema": FINGERPRINT_SCHEMA,
            "store_dir": str(store_dir),
            "split": str(split),
            "fold": str(fold),
            "slice": (f"fold {fold} (in neither {split} train_ids nor val_ids), "
                      "converged & valid, finite node_peak/map_cov/f_r"
                      + (", production rows only" if production_only else "")),
            "production_only": bool(production_only),
            "cell_key": f"cyclen_cell_key(feed, e_core, bin_width={bin_width})",
            "min_cell_rows": int(min_rows),
            "targets": list(TARGETS),
            "n_cells_seen": int(n_seen),
            "n_cells_fitted": len(cells_out),
            "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **slice_report.as_dict(),
        },
        "global": global_out,
        "cells": dict(sorted(cells_out.items())),
    }


def fit_map_calibration(model_dir: str | Path = DEFAULT_MODEL_DIR,
                        store_dir: str | Path = "data/store",
                        splits_dir: str | Path = "data/splits", *,
                        split: str | None = None,
                        fold: str = HONEST_FOLD,
                        min_rows: int = MIN_CELL_ROWS,
                        bin_width: float = DEFAULT_BIN_WIDTH,
                        device: str = "cpu",
                        batch_size: int = 128,
                        production_only: bool = False,
                        dry_run: bool = False,
                        log: Any = print) -> dict[str, Any]:
    """Measure + (unless ``dry_run``) write ``<store_dir>/map_calibration.json``."""
    from ..model.splits import SplitManifest

    model_dir = Path(model_dir)
    store = Path(store_dir)
    if split is None:
        metas = sorted(model_dir.glob("member_*/meta.json"))
        split = "S1"
        if metas:
            split = str(json.loads(metas[0].read_text(encoding="utf-8"))
                        .get("split", "S1"))
    manifest = SplitManifest.from_json(Path(splits_dir) / f"{split}.json")

    records = StoreReader(store).records
    sl, rep = honest_slice(records, manifest, fold=fold,
                           production_only=production_only, bin_width=bin_width)
    log(f"[fit_map_calibration] champion {model_dir} (split {split}) | fold "
        f"{fold}: {rep.n_fold} rows -> {rep.n_labelled} usable "
        f"({rep.n_proposal} model-proposed) | libraries {rep.libraries}")
    if not len(sl):
        raise MapCalibrationFitError(
            f"fold {fold} of split {split} has no row carrying node_peak, map_cov "
            "and f_r; nothing can be calibrated (run the flatness backfill first)")

    preds = predict_slice(sl, model_dir, store, device=device,
                          batch_size=batch_size, log=log)
    doc = build_artifact(sl, preds, model_dir=model_dir, store_dir=store,
                         split=split, fold=fold, slice_report=rep,
                         min_rows=min_rows, bin_width=bin_width,
                         production_only=production_only)

    g = doc["global"]
    for target in TARGETS:
        e = g.get(target)
        if e:
            log(f"[fit_map_calibration] GLOBAL {target:<10} bias {e['bias']:+.4f} "
                f"sigma {e['sigma']:.4f} (ens {e['sigma_ens']:.4f}, extra "
                f"{e['sigma_extra']:.4f}) n={e['n']}")
    log(f"[fit_map_calibration] gate keys: fr_bias {g.get('fr_bias')} "
        f"fr_sigma {g.get('fr_sigma')} -> global gate 1.70 - "
        f"{g.get('fr_bias', 0.0)} - {GATE_K}*{g.get('fr_sigma', 0.0)}")
    log(f"[fit_map_calibration] {doc['fit']['n_cells_fitted']} cells fitted of "
        f"{doc['fit']['n_cells_seen']} seen (floor {min_rows})")

    if dry_run:
        log("[fit_map_calibration] --dry-run: nothing written")
        return doc
    store.mkdir(parents=True, exist_ok=True)
    tmp = store / f"{ARTIFACT_NAME}.tmp"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(store / ARTIFACT_NAME)
    log(f"[fit_map_calibration] -> {store / ARTIFACT_NAME}")
    return doc


def main(argv: Any = None) -> int:  # pragma: no cover - CLI wrapper
    ap = argparse.ArgumentParser(prog="python -m lpopt.tools.fit_map_calibration")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--store-dir", default="data/store")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--split", default=None)
    ap.add_argument("--fold", default=HONEST_FOLD, choices=("A", "B", "C"))
    ap.add_argument("--min-rows", type=int, default=MIN_CELL_ROWS)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--production-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    fit_map_calibration(args.model_dir, args.store_dir, args.splits_dir,
                        split=args.split, fold=args.fold, min_rows=args.min_rows,
                        batch_size=args.batch_size, device=args.device,
                        production_only=bool(args.production_only),
                        dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
