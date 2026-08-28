"""MASTER-verified debug panel: score a champion in NEUTRONICS UNITS.

``lpopt debug-panel score --model-dir <dir> [--input lpopt.inp] [--out <json>]``

Why this exists (user directive 2026-07-29, "앞으로 모델 구축할 때 MASTER와
디버깅하면서 진행해"): every model build must be checkable against MASTER-verified
truth in the units the licensing engineer argues in — EFPD, ppm, F_r — not in the
loss/Spearman space the trainer optimizes.  A composite validation score of 0.76
and a within-case Spearman of 0.88 are both perfectly compatible with a cbc_max
that is 42 ppm off and a cyclen that is 4.4 EFPD off, which is what the champion
``data/models/20260729_054749`` actually was when this module was written.

What it measures, per target
----------------------------
* ``n`` / ``mae`` / ``bias`` (mean signed ``pred - truth``) / ``max_abs_err`` —
  the accuracy block.  ``bias`` is separated from ``mae`` deliberately: a residual
  that is mostly bias is CORRECTABLE (that is what the per-cell affine
  calibrations in :mod:`..model.cell_calibrate` do), while one that is mostly
  scatter is not, and the two demand different work.
* ``frac_abs_z_gt2`` — the fraction of rows whose ``|pred - truth| / sigma``
  exceeds 2, using the model's OWN served sigma (``calibrated_std`` for the
  surrogate columns, the map-head spread for the flatness scalars).  This is the
  OOD-OVERCONFIDENCE DETECTOR.  A well-calibrated sigma puts ~5% of rows above
  |z|=2; the champion's map head put a single blind OOD case at z = −12.8, which
  no accuracy metric can surface because the MEAN was merely wrong, not absurd —
  it was the SIGMA that was absurd.
* ``verdict`` — ``PASS``/``FAIL`` of ``mae <= tolerance``.
* ``by_library`` — the same n / MAE / bias split by the SERVE library each row
  resolves to.  Added 2026-07-29 after a whole-slice number hid a total failure:
  the champion's cbc_max bias read +25.6 ppm overall while it was +1.5 for ga80
  and +49.0 for paramA, because the calibration covered ga80 only.  An aggregate
  over two regimes is the average of a solved problem and an untouched one, and
  it looks like partial progress on both.

Scoring goes through :meth:`..model.model_api.PosValCnnBackend.predict`, i.e. the
exact SERVE path including every per-cell calibration the checkpoint ships, and
:meth:`predict_map_flatness` for the two flatness scalars, which is the same
surface :func:`..curriculum._map_head_flatness` reads.  What is scored is
therefore what the campaign actually consumes, not a raw head.

Report-only, ALWAYS exit 0
--------------------------
This command can never block anything.  That mirrors the 2026-07-26 warn-don't-
block gate decision, and for the same reason: a report that can stop a pipeline
acquires a constituency for making its thresholds meaningless.  The tolerances
here are meant to be argued with, and the JSON artifact is meant to be diffed
across builds.

Panel membership comes from the store's ``campaign`` column (default globs
``debug_panel*`` / ``democase*``).  Seeding the panel is a MASTER production run
and is deliberately OUT OF SCOPE here — this module reads verified rows, it never
launches MASTER.
"""

from __future__ import annotations

import fnmatch
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

#: Store campaign globs that define panel membership by default.
DEFAULT_PANEL_CAMPAIGNS: tuple[str, ...] = ("debug_panel*", "democase*")

#: Per-target NEUTRONICS tolerances the verdict is taken against.
#:
#: These are engineering quantities, not statistics: cyclen 3 EFPD is roughly the
#: MASTER equilibrium convergence noise floor (~2 EFPD) plus a margin; cbc_max
#: 20 ppm is ~1.3% of the 1550 ppm ``cbc_limit`` and well inside a boron-worth
#: reading; F_r 0.05 is ~3% of the 1.55 licensing limit; node_peak 0.05 and
#: map_cov 0.02 are the flatness-objective scales the search actually resolves.
#: Overridable per key from a deck's ``[debug_panel] tolerances`` table.
DEFAULT_TOLERANCES: dict[str, float] = {
    "cyclen": 3.0,          # EFPD
    "cbc_max": 20.0,        # ppm
    "f_r": 0.05,            # -
    "f_q": 0.08,            # -
    "ao_abs": 0.010,        # -
    "node_peak": 0.05,      # - (F_xy)
    "map_cov": 0.02,        # -
}

#: Physical unit of each target, for the console table only.
TARGET_UNITS: dict[str, str] = {
    "cyclen": "EFPD", "cbc_max": "ppm", "f_r": "-", "f_q": "-",
    "ao_abs": "-", "node_peak": "-", "map_cov": "-",
}

#: Targets served through the 7-column surrogate layout, with their column index.
#: Mirrors ``model_api._TARGET_TO_SURROGATE_COL`` — kept as an explicit local map
#: because the panel scores a FIXED inventory (adding a target here is a decision
#: about what the panel promises, not a consequence of a checkpoint's head width).
SURROGATE_COLS: dict[str, int] = {
    "f_r": 0, "cbc_max": 1, "f_q": 2, "cyclen": 3, "ao_abs": 4,
}
#: Targets served through the map head (``predict_map_flatness``).
FLATNESS_TARGETS: tuple[str, ...] = ("node_peak", "map_cov")

#: Full scoring inventory, in report order.
PANEL_TARGETS: tuple[str, ...] = (
    "cyclen", "cbc_max", "f_r", "f_q", "ao_abs", "node_peak", "map_cov")

#: |z| above which a row counts as a sigma miss (2 sigma ~= 95% nominal coverage).
Z_OUTLIER = 2.0

ARTIFACT_SCHEMA = "debug_panel_score_v1"


# --------------------------------------------------------------------------- #
# panel membership
# --------------------------------------------------------------------------- #
def campaign_matches(campaign: Any, globs: Sequence[str]) -> bool:
    """True when ``campaign`` fnmatches any glob (None/NaN never matches)."""
    if campaign is None or (isinstance(campaign, float) and math.isnan(campaign)):
        return False
    text = str(campaign)
    return any(fnmatch.fnmatch(text, str(g)) for g in globs)


def panel_frame(records: pd.DataFrame,
                globs: Sequence[str] = DEFAULT_PANEL_CAMPAIGNS,
                *, converged_only: bool = True) -> pd.DataFrame:
    """The panel slice of a store frame: campaign-glob members that converged.

    A non-converged row carries no usable truth (its targets are the state the
    solver stopped in, not an equilibrium), so it is dropped rather than scored
    against — otherwise the panel would report a model error for a MASTER
    non-convergence.  ``converged_only=False`` is available for forensics.
    """
    if records is None or not len(records):
        return records if isinstance(records, pd.DataFrame) else pd.DataFrame()
    camp = records["campaign"] if "campaign" in records.columns else None
    if camp is None:
        return records.iloc[0:0]
    keep = np.asarray([campaign_matches(c, globs) for c in camp], dtype=bool)
    if converged_only and "converged" in records.columns:
        keep = keep & records["converged"].fillna(False).astype(bool).to_numpy()
    return records[keep]


# --------------------------------------------------------------------------- #
# pure scoring math
# --------------------------------------------------------------------------- #
def _truth_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """The panel truth vector for ``name`` (all-NaN when the column is absent).

    ``cbc_max`` additionally censors ``cbc_kind == "boc_only"`` rows: a BOC-only
    boron reading is not the EDIT2 MAXIMUM the head predicts, so scoring against
    it would report a model error for a label-definition mismatch.  This is the
    same censoring :func:`..model.train._valid_target_values` applies at train
    time, restated here so the panel can never disagree with training about what
    a cbc_max label IS.
    """
    n = len(frame)
    if name not in frame.columns:
        return np.full(n, np.nan)
    vals = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if name == "cbc_max" and "cbc_kind" in frame.columns:
        boc = frame["cbc_kind"].astype(str).to_numpy() == "boc_only"
        vals = np.where(boc, np.nan, vals)
    return vals


def _split_by_library(pred: np.ndarray, truth: np.ndarray,
                      libraries: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """``{library: {n, mae, bias, max_abs_err}}`` over the finite rows.

    A per-library view is the difference between "the model is 25 ppm biased" and
    "the model is fine on one regime and 49 ppm out on the other" — the second is
    actionable and the first is not.
    """
    libs = np.asarray([str(x) for x in libraries], dtype=object)
    ok = np.isfinite(pred) & np.isfinite(truth)
    out: dict[str, dict[str, Any]] = {}
    for lib in sorted(set(libs[ok].tolist())):
        sel = ok & (libs == lib)
        err = pred[sel] - truth[sel]
        out[lib] = {
            "n": int(err.size),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "max_abs_err": float(np.max(np.abs(err))),
        }
    return out


def score_target(pred: Sequence[float], truth: Sequence[float],
                 sigma: Sequence[float] | None = None,
                 *, tolerance: float | None = None,
                 unit: str = "",
                 libraries: Sequence[Any] | None = None) -> dict[str, Any]:
    """Accuracy + sigma-coverage stats for ONE target (pure; no model, no store).

    Rows are scored only where BOTH ``pred`` and ``truth`` are finite — a target
    the checkpoint does not predict (NaN column) or a row missing the label simply
    does not contribute, and ``n`` says so.  The sigma block is computed over the
    further subset with a finite POSITIVE sigma, reported separately as ``n_sigma``
    so "no coverage stat" is visibly distinct from "coverage stat of zero misses".
    """
    p = np.asarray(pred, dtype=float).reshape(-1)
    t = np.asarray(truth, dtype=float).reshape(-1)
    if p.size != t.size:
        raise ValueError(f"pred/truth length mismatch: {p.size} vs {t.size}")
    ok = np.isfinite(p) & np.isfinite(t)
    err = p[ok] - t[ok]
    n = int(err.size)
    out: dict[str, Any] = {
        "n": n,
        "unit": unit,
        "tolerance": None if tolerance is None else float(tolerance),
        "mae": None, "bias": None, "max_abs_err": None, "rmse": None,
        "bias_share": None,
        "n_sigma": 0, "frac_abs_z_gt2": None, "mean_abs_z": None,
        "max_abs_z": None, "min_signed_z": None,
        "by_library": {},
        "verdict": "NO DATA",
    }
    if libraries is not None and n:
        if len(libraries) != p.size:
            raise ValueError(
                f"libraries length mismatch: {len(libraries)} vs {p.size}")
        out["by_library"] = _split_by_library(p, t, libraries)
    if n:
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        out["mae"] = mae
        out["bias"] = bias
        out["max_abs_err"] = float(np.max(np.abs(err)))
        out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        # How much of the error is a correctable shift?  |bias|/MAE == 1 means a
        # pure offset (a per-cell affine calibration removes it); ~0 means pure
        # scatter (only a better model helps).
        out["bias_share"] = float(abs(bias) / mae) if mae > 0 else 0.0
        if tolerance is not None:
            out["verdict"] = "PASS" if mae <= float(tolerance) else "FAIL"
        else:
            out["verdict"] = "NO TOL"

    if sigma is not None and n:
        s = np.asarray(sigma, dtype=float).reshape(-1)
        if s.size != p.size:
            raise ValueError(f"sigma length mismatch: {s.size} vs {p.size}")
        s_ok = s[ok]
        good = np.isfinite(s_ok) & (s_ok > 0.0)
        if good.any():
            z = err[good] / s_ok[good]
            out["n_sigma"] = int(z.size)
            out["frac_abs_z_gt2"] = float(np.mean(np.abs(z) > Z_OUTLIER))
            out["mean_abs_z"] = float(np.mean(np.abs(z)))
            out["max_abs_z"] = float(np.max(np.abs(z)))
            out["min_signed_z"] = float(np.min(z))
    return out


def score_frame(frame: pd.DataFrame,
                pred: Mapping[str, Sequence[float]],
                sigma: Mapping[str, Sequence[float]] | None = None,
                *, tolerances: Mapping[str, float] | None = None,
                targets: Sequence[str] = PANEL_TARGETS,
                libraries: Sequence[Any] | None = None) -> dict[str, Any]:
    """Score every panel target of a frame against supplied predictions.

    Split out from :func:`score_panel` so the MATH is testable without a torch
    checkpoint: the caller owns the forward pass, this owns the statistics and the
    verdicts.  A target absent from ``pred`` is reported with ``n=0`` and verdict
    ``NO DATA`` rather than omitted — a silently missing row in the table is how a
    head stops being watched.
    """
    tol = dict(DEFAULT_TOLERANCES)
    tol.update({str(k): float(v) for k, v in (tolerances or {}).items()})
    n_rows = len(frame)
    per_target: dict[str, Any] = {}
    for name in targets:
        p = pred.get(name)
        if p is None:
            p = np.full(n_rows, np.nan)
        s = None if sigma is None else sigma.get(name)
        per_target[name] = score_target(
            p, _truth_column(frame, name), s,
            tolerance=tol.get(name), unit=TARGET_UNITS.get(name, ""),
            libraries=libraries)
    failed = [k for k, v in per_target.items() if v["verdict"] == "FAIL"]
    passed = [k for k, v in per_target.items() if v["verdict"] == "PASS"]
    unscored = [k for k, v in per_target.items()
                if v["verdict"] not in ("PASS", "FAIL")]
    return {
        "n_panel_rows": int(n_rows),
        "libraries": (sorted(set(str(x) for x in libraries))
                      if libraries is not None else []),
        "tolerances": tol,
        "targets": per_target,
        "passed": passed,
        "failed": failed,
        "unscored": unscored,
        # NOT a gate: a convenience roll-up for a report line.  Nothing in this
        # module or the CLI reads it to decide an exit code.
        "all_within_tolerance": (not failed) and bool(passed),
    }


# --------------------------------------------------------------------------- #
# model forward
# --------------------------------------------------------------------------- #
def predict_panel(model: Any, frame: pd.DataFrame, *, batch_size: int = 512
                  ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """``(pred, sigma, notes)`` for the panel rows, through the SERVE path.

    Surrogate targets come from ``model.predict`` (means + ``calibrated_std``),
    the flatness pair from ``model.predict_map_flatness`` (means + across-member
    spread, already OOD-floored by the backend).  A model without a map head
    yields NaN flatness columns and a note, never an exception: the panel must
    still report the five scalar targets for a checkpoint whose map head is
    missing or broken.
    """
    from ..data.schema import unpack_pattern
    from ..vendor.masterrl.domain import CaseKey

    notes: list[str] = []
    n = len(frame)
    pred: dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in PANEL_TARGETS}
    sigma: dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in PANEL_TARGETS}
    if not n:
        return pred, sigma, notes

    patterns = [unpack_pattern(str(p)) for p in frame["pattern"]]
    cases = [CaseKey(pair=str(cp), feed=int(f))
             for cp, f in zip(frame["case_pair"], frame["feed"])]

    flat_fn = getattr(model, "predict_map_flatness", None)
    flat_failed = False
    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        bp, bc = patterns[start:stop], cases[start:stop]
        out = model.predict(bp, bc, 0.0)
        mean = np.asarray(out.mean, dtype=float)
        std = np.asarray(out.calibrated_std, dtype=float)
        for name, col in SURROGATE_COLS.items():
            pred[name][start:stop] = mean[:, col]
            sigma[name][start:stop] = std[:, col]
        if callable(flat_fn) and not flat_failed:
            try:
                pk_m, pk_s, cv_m, cv_s = flat_fn(bp, bc)
            except (AttributeError, IndexError, KeyError, RuntimeError,
                    TypeError, ValueError) as exc:
                flat_failed = True
                notes.append(f"map head failed ({type(exc).__name__}: {exc}); "
                             "node_peak/map_cov unscored")
                # Blank the WHOLE column, not just this batch: a flatness stat
                # computed over the rows that happened to precede the failure is
                # a silently biased sample, which is worse than no number.
                for name in FLATNESS_TARGETS:
                    pred[name][:] = np.nan
                    sigma[name][:] = np.nan
            else:
                pred["node_peak"][start:stop] = np.asarray(pk_m, dtype=float)
                sigma["node_peak"][start:stop] = np.asarray(pk_s, dtype=float)
                pred["map_cov"][start:stop] = np.asarray(cv_m, dtype=float)
                sigma["map_cov"][start:stop] = np.asarray(cv_s, dtype=float)
    if not callable(flat_fn):
        notes.append("model exposes no predict_map_flatness; "
                     "node_peak/map_cov unscored")
    return pred, sigma, notes


def panel_libraries(model: Any, frame: pd.DataFrame) -> list[str]:
    """The SERVE library each panel row resolves to, for the per-library split.

    Uses ``model.serve_library`` (the effective-library rerouting the featurizer
    and every calibration key already use) so the breakdown groups rows the way
    the model actually treats them — NOT by the stored ``library_id``, which can
    differ for a pattern the serve path cannot disambiguate.  Falls back to the
    stored column for a model without the accessor.
    """
    fn = getattr(model, "serve_library", None)
    stored = (frame["library_id"].astype(str).tolist()
              if "library_id" in frame.columns else ["?"] * len(frame))
    if not callable(fn) or "pattern" not in frame.columns:
        return stored
    from ..data.schema import unpack_pattern
    out: list[str] = []
    for packed, fallback in zip(frame["pattern"], stored):
        try:
            out.append(str(fn(unpack_pattern(str(packed)))))
        except (AttributeError, KeyError, TypeError, ValueError):
            out.append(fallback)
    return out


def score_panel(model: Any, frame: pd.DataFrame, *,
                tolerances: Mapping[str, float] | None = None,
                batch_size: int = 512) -> dict[str, Any]:
    """Forward the panel through ``model`` and score it (the whole measurement)."""
    pred, sigma, notes = predict_panel(model, frame, batch_size=batch_size)
    report = score_frame(frame, pred, sigma, tolerances=tolerances,
                         libraries=panel_libraries(model, frame))
    if notes:
        report["notes"] = notes
    return report


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}g}"


def format_report(report: Mapping[str, Any], *, by_library: bool = True) -> str:
    """The console table: one row per target, then its per-serve-library sub-rows.

    The sub-rows are indented and carry only n / MAE / bias — the columns that
    answer "is this target broken for one regime and fine for another?".  They are
    suppressed when the panel has a single library (nothing to compare) or when no
    library information was supplied.
    """
    header = ("target", "unit", "n", "MAE", "tol", "bias", "bias%",
              "max|err|", "n_sig", "|z|>2", "max|z|", "verdict")
    rows = [header]
    for name in PANEL_TARGETS:
        t = report["targets"].get(name)
        if t is None:
            continue
        share = t.get("bias_share")
        rows.append((
            name,
            str(t.get("unit", "")),
            str(t["n"]),
            _fmt(t["mae"]),
            _fmt(t["tolerance"]),
            _fmt(t["bias"]),
            "-" if share is None else f"{100.0 * share:.0f}%",
            _fmt(t["max_abs_err"]),
            str(t["n_sigma"]),
            "-" if t["frac_abs_z_gt2"] is None else f"{100.0 * t['frac_abs_z_gt2']:.0f}%",
            _fmt(t["max_abs_z"], 3),
            t["verdict"],
        ))
        libs = t.get("by_library") or {}
        if by_library and len(libs) > 1:
            for lib, st in sorted(libs.items()):
                rows.append((f"  └ {lib}", "", str(st["n"]), _fmt(st["mae"]), "",
                             _fmt(st["bias"]), "", _fmt(st["max_abs_err"]),
                             "", "", "", ""))
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[j])
                               for j, cell in enumerate(row)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    for note in report.get("notes", ()):
        lines.append(f"[note] {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI body
# --------------------------------------------------------------------------- #
def run_score(model_dir: str | Path, *,
              store_dir: str | Path = "data/store",
              library_id: str = "ga80",
              device: str = "cpu",
              campaigns: Sequence[str] = DEFAULT_PANEL_CAMPAIGNS,
              tolerances: Mapping[str, float] | None = None,
              out_path: str | Path | None = None,
              batch_size: int = 512,
              records: pd.DataFrame | None = None,
              model: Any = None) -> dict[str, Any]:
    """Load the champion, score the panel, write the JSON artifact, return it.

    ``records`` / ``model`` are injection points for tests; production passes
    neither and this reads the store + loads the checkpoint itself.
    """
    from ..data.store import StoreReader

    model_dir = Path(model_dir)
    if records is None:
        records = StoreReader(store_dir).records
    frame = panel_frame(records, campaigns)

    if model is None and len(frame):
        from ..model.model_api import PosValCnnBackend
        model = PosValCnnBackend.from_dir(
            model_dir, store_dir=store_dir, library_id=library_id, device=device)

    if len(frame) and model is not None:
        report = score_panel(model, frame, tolerances=tolerances,
                             batch_size=batch_size)
    else:
        report = score_frame(frame, {}, None, tolerances=tolerances)
        if not len(frame):
            report.setdefault("notes", []).append(
                "panel is EMPTY: no store row matches campaigns "
                f"{list(campaigns)} — seed it with a MASTER production run "
                "(out of scope for this command, which never runs MASTER)")

    report.update({
        "schema": ARTIFACT_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(model_dir),
        "store_dir": str(store_dir),
        "library_id": library_id,
        "campaigns": list(campaigns),
        "report_only": True,
    })
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str),
                       encoding="utf-8")
        tmp.replace(p)
    return report


__all__ = [
    "ARTIFACT_SCHEMA",
    "DEFAULT_PANEL_CAMPAIGNS",
    "DEFAULT_TOLERANCES",
    "FLATNESS_TARGETS",
    "PANEL_TARGETS",
    "SURROGATE_COLS",
    "TARGET_UNITS",
    "Z_OUTLIER",
    "campaign_matches",
    "format_report",
    "panel_libraries",
    "panel_frame",
    "predict_panel",
    "run_score",
    "score_frame",
    "score_panel",
    "score_target",
]
