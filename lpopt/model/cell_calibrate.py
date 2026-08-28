"""Per-cell cyclen bias correction (two-stage, plan sec. 4.4 / 12.5 addendum).

The champion's cyclen head carries a **uniform per-cell bias** (the model over-
predicts cyclen by ~+6.6..+14.7 EFPD on honest per-cell holdouts) even though it
trains on ~80% of each cell: the globally z-scored Huber/NLL loss is dominated by
the ~38k feed-121 corpus, which shrinks the cell-conditional means toward the
Dataset-A regime.  The bias is (nearly) a constant shift per (feed, e_core-bin)
cell — bias-centering the per-cell MAE collapses it to the MASTER convergence
noise floor (~2 EFPD).  ``user_criteria`` gates candidates by a
``|cyclen - target| <= tol`` band, so a +15 EFPD shift walks the whole band off
target and *silently loses recall* in screening (it never causes a wrong
acceptance — the top-K is MASTER-verified — only misses).

Two complementary correctors live here:

**Stage 1 — serve-side per-cell affine calibration** (:func:`fit_cell_affine`).
A robust ``cyclen_cal = a * pred + b`` is fit per (feed, e_core-bin) cell from the
champion's OWN split-manifest **train** rows (never holdout rows — the fit must
not touch the honest eval fold).  It is persisted as ``cell_calibration.json`` in
the model dir and applied inside :meth:`PosValCnnBackend.predict` to the cyclen
column only.  Because the fit sees only train rows and the model already trained
on them, this is not "training on the test set": it corrects a *known systematic
shift* that survives on train rows too, and it is measured/applied identically for
every champion (the no-regression gate stays fair — see
:meth:`PosValCnnBackend.predict`).

**Stage 2 — campaign running bias corrector** (:class:`CampaignBiasCorrector`).
For a cell that has NO fitted calibration (the new-cell exploration case), a
running per-cell cyclen bias is accumulated from THIS campaign's MASTER-verified
chains — ``bias_hat = n/(n+prior) * median(pred - actual)`` — and subtracted from
subsequent screening/deepen cyclen predictions for that cell.  It is inert
(``bias_hat == 0``) until a verified label lands, shrinks toward 0 with a prior,
and never touches a cell that Stage 1 already covers (no double correction).

Stage 1 is TARGET-AGNOSTIC and is instantiated three times, one artifact file per
corrected column: :func:`fit_cell_affine` (cyclen, surrogate col 3),
:func:`fit_cell_affine_fr` (F_r, col 0) and :func:`fit_cell_affine_cbc`
(CBC_max, col 1 — added 2026-07-29 debug-panel).  The three files are
independent, touch disjoint columns, and each is a pure no-op when absent, so a
checkpoint that ships only some of them loads and serves unchanged.

Cell key parity: both stages key a cell exactly the way the training sampler keys
its inverse-sqrt weighting cells (:func:`dataset_torch.compute_cell_weights`) —
``floor(e_core / bin_width) * bin_width`` — but WITHOUT the dataset axis, because
at serve time the dataset is fixed by the campaign library and the key must be a
pure function of ``CaseKey.feed`` + the pattern's serve-recipe ``e_core`` (the
same recipe extraction used, so the served bin equals the stored bin).

That parity is the whole mechanism, and it broke silently once (forensic
2026-07-29 debug-panel): the serve recipe resolved ``e_core`` against the
CONFIGURED library while the featurizer resolved it against the EFFECTIVE one, so
a paramA pattern served through a ga80 campaign keyed into ``ebin=None`` — a cell
no fit populates — and received no correction on ANY of the three targets, for
50.9% of the curriculum-val slice.  Both sides now call
:meth:`PosValCnnBackend.cyclen_e_core`, which resolves the EFFECTIVE library, and
the fit admits exactly the rows for which that resolution round-trips
(:func:`serve_parity_mask`).  A row whose cell is unfitted falls back to its own
LIBRARY's pooled shift (``global_by_library``), so a thinly-sampled cell no longer
means an uncorrected global bias.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: filename of the persisted Stage-1 artifact inside a model dir.
CELL_CALIB_NAME = "cell_calibration.json"
#: schema tag stamped into the artifact (bump on a breaking format change).
CELL_CALIB_SCHEMA = "cell_cyclen_affine_v1"
#: filename + schema of the F_r sibling artifact (plan Task A — same machinery as
#: the cyclen calibration, a SEPARATE file so the cyclen schema is untouched).
FR_CALIB_NAME = "f_r_calibration.json"
FR_CALIB_SCHEMA = "cell_f_r_affine_v1"
#: filename + schema of the CBC_max sibling artifact (2026-07-29 debug-panel).
#: Measured on the curriculum holdout of champion 20260729_054749: cbc_max MAE
#: 42.4 ppm of which 36% is PER-GROUP BIAS (campaign|case_pair|feed), with a
#: GLOBAL +27 ppm over-prediction and per-group bias up to +113 ppm — the same
#: signature the cyclen head had, so the same per-cell affine machinery applies.
#: A SEPARATE file so neither the cyclen nor the F_r artifact schema is touched.
CBC_CALIB_NAME = "cbc_calibration.json"
CBC_CALIB_SCHEMA = "cell_cbc_max_affine_v1"
#: filename + schema of the F_q and |AO| siblings (2026-07-29 "all targets").
#: Measured on the same curriculum-val holdout: F_q MAE 0.250 with bias −0.178
#: (71% of the error is a pure shift — the largest bias share of any scalar), and
#: ao_abs MAE 0.0060 with bias −0.0010 (17%).  The ao_abs gain is small in MAE but
#: it removes ~70% of the residual bias, and |AO| is a licensing-reported quantity
#: where a systematic offset is worth removing even when the scatter dominates.
FQ_CALIB_NAME = "f_q_calibration.json"
FQ_CALIB_SCHEMA = "cell_f_q_affine_v1"
AO_CALIB_NAME = "ao_abs_calibration.json"
AO_CALIB_SCHEMA = "cell_ao_abs_affine_v1"
#: filename + schema of the FLATNESS artifact.  Unlike the five scalar files this
#: holds BOTH map-head axes, because they come from one ``predict_map_flatness``
#: forward and splitting them would double the (expensive) fit for no benefit.
#: Its ``cells`` / ``global_by_library`` are therefore keyed by TARGET first — see
#: :func:`flatness_cells`.  INTERCEPT-ONLY by construction (see
#: :func:`fit_flatness_calibration`).
FLAT_CALIB_NAME = "flatness_calibration.json"
FLAT_CALIB_SCHEMA = "cell_flatness_intercept_v1"
#: The two map-head axes the flatness artifact corrects, in report order.
FLATNESS_TARGETS: tuple[str, ...] = ("node_peak", "map_cov")
#: e_core bin width — must equal ``compute_cell_weights``' ``e_core_bin_width``.
DEFAULT_BIN_WIDTH = 0.05
#: minimum labelled train rows in a cell to fit ANY correction.
DEFAULT_MIN_ROWS = 30
#: below this, the slope is under-identified -> intercept-only (a=1, b=-bias).
DEFAULT_SLOPE_MIN_ROWS = 50
#: cyclen lives in surrogate/target column 3 (both layouts agree — see model_api).
CYCLEN_COL = 3
#: F_r lives in surrogate/target column 0 (both layouts agree — see model_api).
FR_COL = 0
#: CBC_max lives in SURROGATE column 1 (``_TARGET_TO_SURROGATE_COL["cbc_max"]``).
#: NOTE the two layouts do NOT agree here — the dataset/target order is
#: ``(f_r, f_q, cbc_max, ...)`` while the surrogate order is
#: ``(F_r, CBC_max, F_q, ...)`` — and this constant is the SURROGATE one, because
#: the fit and the serve hook both read ``predict().mean``, which is the 7-column
#: surrogate layout.  (CYCLEN_COL/FR_COL are only ambiguity-free by coincidence.)
CBC_COL = 1
#: F_q lives in SURROGATE column 2 and |AO| in SURROGATE column 4 (same caveat as
#: CBC_COL above: these are surrogate indices, not dataset-target indices).
FQ_COL = 2
AO_COL = 4
#: shrinkage prior weight for the Stage-2 running corrector.
DEFAULT_PRIOR_WEIGHT = 4.0
#: OOF median-abs-error an affine slope must beat intercept-only by to be chosen —
#: a conservative margin so the uniform-bias default wins on noisy cells.  It is a
#: TARGET-SCALE quantity: the cyclen default is in EFPD (target ~600), the F_r fit
#: passes an F_r-scale margin (target ~1.5-4.7) via :data:`_AFFINE_MARGIN_FR`.
_AFFINE_MARGIN_EFPD = 0.25
#: F_r-scale OOF margin (F_r ranges ~1.5-4.7; an EFPD-scale 0.25 would never let a
#: real slope win).  ~0.4% of a typical F_r — small enough to admit a genuine
#: slope distortion, large enough that the parsimonious per-cell shift wins on noise.
_AFFINE_MARGIN_FR = 0.006
#: CBC-scale OOF margin [ppm].  cbc_max ranges ~1200-2300 ppm with a measured MAE
#: of 42 ppm, so the EFPD-scale 0.25 would let noise win the affine-vs-intercept
#: contest on every cell.  2 ppm ~= 5% of the current MAE: small enough to admit a
#: genuine slope distortion, large enough that the parsimonious per-cell shift
#: (which is what a 36%-bias-share residual actually is) wins on noise.
_AFFINE_MARGIN_CBC = 2.0
#: F_q-scale OOF margin.  F_q ranges ~2.1-4.9 with a measured MAE of 0.25, so this
#: is ~3% of the MAE — the same "small enough to admit a real slope, large enough
#: that noise cannot win" calibration as the F_r margin, at F_q's scale.
_AFFINE_MARGIN_FQ = 0.008
#: |AO|-scale OOF margin.  ao_abs is O(0.05) with a measured MAE of 0.006; 2e-4 is
#: ~3% of that MAE.  A margin borrowed from any other target would be either
#: thousands of times too large (EFPD) or so small that noise picks the slope.
_AFFINE_MARGIN_AO = 2.0e-4
#: plausible robust-slope range; a fit outside this is rejected as unstable.
_SLOPE_LO, _SLOPE_HI = 0.2, 3.0


# --------------------------------------------------------------------------- #
# cell key (shared by both stages + the serve path)
# --------------------------------------------------------------------------- #
def cyclen_cell_key(feed: Any, e_core: float | None,
                    bin_width: float = DEFAULT_BIN_WIDTH) -> str:
    """``(feed, e_core-bin)`` calibration-cell key, mirroring the weighting cells.

    ``floor(e_core / bin_width) * bin_width`` rounded to 4 dp (identical to
    :func:`dataset_torch.compute_cell_weights`), minus the dataset axis so the key
    is derivable at serve time from ``CaseKey.feed`` + the pattern's serve-recipe
    ``e_core``.  A missing / non-finite ``e_core`` yields the ``ebin=None`` cell
    (rows whose fed types are unresolvable share one bin, fit + serve alike).
    """
    f = int(feed)
    if e_core is None or (isinstance(e_core, float) and not math.isfinite(e_core)):
        return f"feed={f}|ebin=None"
    ebin = round(math.floor(float(e_core) / bin_width) * bin_width, 4)
    return f"feed={f}|ebin={ebin}"


# --------------------------------------------------------------------------- #
# Stage 1: per-cell affine fit
# --------------------------------------------------------------------------- #
def _robust_affine(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float] | None:
    """Robust ``actual ~ a*pred + b`` via Huber; ``None`` if degenerate/unavailable."""
    if float(np.ptp(pred)) < 1e-9:
        return None
    try:
        from sklearn.linear_model import HuberRegressor

        h = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=300)
        h.fit(pred.reshape(-1, 1), actual)
        a = float(h.coef_[0])
        b = float(h.intercept_)
    except Exception:      # pragma: no cover - sklearn edge / non-convergence
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return a, b


def _median_abs_err(a: float, b: float, pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.median(np.abs((a * pred + b) - actual)))


def _kfold_indices(n: int, k: int, seed: int = 0) -> list[np.ndarray]:
    """Deterministic k-fold index partition (shuffled by a fixed rng)."""
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


def _crossfit_choice(pred: np.ndarray, actual: np.ndarray, *, k: int = 5,
                     affine_margin: float = _AFFINE_MARGIN_EFPD
                     ) -> tuple[str, float, float]:
    """Pick ``intercept`` vs ``affine`` by out-of-fold stability.

    Returns ``(estimator, oof_intercept_mae, oof_affine_mae)``.  ``affine`` is
    chosen only when EVERY fold's robust slope is finite and inside
    ``[_SLOPE_LO, _SLOPE_HI]`` (stable) AND its pooled OOF median-abs-error beats
    intercept-only by at least ``affine_margin`` (target-scale — EFPD for cyclen,
    F_r-scale for F_r) — otherwise the uniform per-cell shift (a=1) wins, matching
    the measured near-constant bias.
    """
    n = pred.size
    k = max(2, min(k, n))
    folds = _kfold_indices(n, k)
    err_int: list[float] = []
    err_aff: list[float] = []
    affine_stable = True
    for held in folds:
        mask = np.ones(n, dtype=bool)
        mask[held] = False
        p_tr, a_tr = pred[mask], actual[mask]
        p_te, a_te = pred[held], actual[held]
        if p_te.size == 0 or p_tr.size < 2:
            continue
        b_int = float(np.median(a_tr - p_tr))
        err_int.append(_median_abs_err(1.0, b_int, p_te, a_te))
        aff = _robust_affine(p_tr, a_tr)
        if aff is None or not (_SLOPE_LO <= aff[0] <= _SLOPE_HI):
            affine_stable = False
            continue
        err_aff.append(_median_abs_err(aff[0], aff[1], p_te, a_te))
    oof_int = float(np.mean(err_int)) if err_int else float("inf")
    oof_aff = (float(np.mean(err_aff))
               if (affine_stable and len(err_aff) == len(err_int) and err_aff)
               else float("inf"))
    if affine_stable and oof_aff < oof_int - affine_margin:
        return "affine", oof_int, oof_aff
    return "intercept", oof_int, oof_aff


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of ``values`` under non-negative ``weights``.

    The smallest ``v`` whose cumulative weight (values sorted ascending) reaches
    half the total weight.  Falls back to the plain median when all weights are 0.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")
    tot = float(w.sum())
    if tot <= 0.0:
        return float(np.median(v))
    order = np.argsort(v, kind="stable")
    vs, ws = v[order], w[order]
    cum = np.cumsum(ws)
    k = int(np.searchsorted(cum, 0.5 * tot, side="left"))
    return float(vs[min(k, vs.size - 1)])


def fit_affine_cell(pred: Sequence[float], actual: Sequence[float], *,
                    min_rows: int = DEFAULT_MIN_ROWS,
                    slope_min_rows: int = DEFAULT_SLOPE_MIN_ROWS,
                    affine_margin: float = _AFFINE_MARGIN_EFPD,
                    low_weight_thresh: float | None = None,
                    low_weight: float = 1.0,
                    conformal_offset: float = 0.0) -> dict | None:
    """Fit one cell's correction ``target_cal = a*pred + b`` (cyclen or F_r).

    Default is the robust intercept-only shift ``a=1, b=median(actual-pred)`` (=
    ``-median_bias``).  When ``n >= slope_min_rows`` a robust affine slope is
    tried and adopted ONLY if a 5-fold cross-fit proves it both stable and better
    than intercept-only by ``affine_margin`` (target-scale — EFPD for cyclen,
    F_r-scale for F_r; see :func:`_crossfit_choice`).  Returns ``None`` when the
    cell has fewer than ``min_rows`` finite pairs.  The chosen ``a`` is always > 0
    (intercept a=1, or affine a in ``[_SLOPE_LO, _SLOPE_HI]``), so the map is
    strictly increasing and preserves within-cell ranking (gate-neutral).

    Boundary conservatization (F_r, forensic parity_round1c_20260722): when
    ``low_weight_thresh`` is set, the intercept uses a WEIGHTED median of the
    residuals with rows whose ``actual <= low_weight_thresh`` up-weighted by
    ``low_weight`` — so the shift is fit where the search actually operates (the
    low-F_r boundary), not the crowded high-F_r bulk.  ``conformal_offset`` is a
    fixed additive shift ADDED to ``b`` (for F_r, a positive value pushes the
    calibrated prediction UP = conservative, offsetting the champion's measured
    global non-conservative bias of −0.084).  Both default to a no-op, so the
    cyclen path and any caller that omits them is byte-identical.
    """
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    finite = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[finite], actual[finite]
    n = int(pred.size)
    if n < int(min_rows):
        return None

    residual = pred - actual                               # +ve => over-predicts
    if low_weight_thresh is not None and low_weight != 1.0:
        w = np.where(actual <= float(low_weight_thresh), float(low_weight), 1.0)
        median_bias = _weighted_median(residual, w)
    else:
        median_bias = float(np.median(residual))
    a, b, estimator = 1.0, -median_bias, "intercept"
    oof_int = oof_aff = None
    if n >= int(slope_min_rows):
        choice, oof_int, oof_aff = _crossfit_choice(pred, actual, affine_margin=affine_margin)
        if choice == "affine":
            aff = _robust_affine(pred, actual)
            if aff is not None and _SLOPE_LO <= aff[0] <= _SLOPE_HI:
                a, b, estimator = float(aff[0]), float(aff[1]), "affine"

    # Conservatism margin: shift the calibrated prediction by a fixed offset (F_r:
    # +offset => higher predicted F_r => the search under-proposes over-limit LPs).
    b = float(b) + float(conformal_offset)

    mae_before = float(np.median(np.abs(pred - actual)))
    mae_after = _median_abs_err(a, b, pred, actual)
    out = {
        "a": float(a),
        "b": float(b),
        "n": n,
        "estimator": estimator,
        "median_bias": round(float(median_bias), 4),
        "conformal_offset": round(float(conformal_offset), 4),
        "mae_before": round(mae_before, 4),
        "mae_after": round(mae_after, 4),
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if oof_int is not None and math.isfinite(oof_int):
        out["oof_intercept_mae"] = round(oof_int, 4)
    if oof_aff is not None and math.isfinite(oof_aff):
        out["oof_affine_mae"] = round(oof_aff, 4)
    return out


def serve_parity_mask(row_libraries: Sequence[Any],
                      serve_libraries: Sequence[Any]) -> np.ndarray:
    """Rows whose SERVE-resolved library equals the library they were produced under.

    This replaces the old blanket ``library_id == "ga80"`` fit filter and is the
    load-bearing invariant behind it, stated directly: a calibration cell is only
    meaningful when the ``(feed, e_core-bin)`` key computed at FIT time is the key
    the SERVE path will compute for the same pattern.  The serve path resolves
    e_core against :meth:`PosValCnnBackend._effective_library`, so the fit must
    admit exactly the rows for which that resolution round-trips.

    Measured 2026-07-29 on the real fuel table: ``ga80`` and ``paramA`` patterns
    round-trip (their fresh types live in exactly one roster), while ``260624`` and
    ``5.8_5.1`` patterns resolve to ``ga80`` — their batch labels collide with the
    ga80 roster and a pattern alone cannot disambiguate them.  Admitting those rows
    would key them by a ga80-resolved e_core that is not their own, contaminating
    the ga80 cells.  Stating the rule as an invariant instead of a library allowlist
    means a newly added library is admitted or rejected on its own merits with no
    code change.
    """
    a = np.asarray([str(x) for x in row_libraries], dtype=object)
    b = np.asarray([str(x) for x in serve_libraries], dtype=object)
    return a == b


def fit_row_mask(records: "Any", train_ids: set[str],
                 library_id: str | Sequence[str] | None = None,
                 target_col: str = "cyclen") -> np.ndarray:
    """Boolean mask of the rows eligible for the calibration fit.

    A row enters the fit iff its ``record_id`` is a **train** id of the champion's
    split (never a holdout/``val`` id — that is the whole leakage guarantee),
    AND it converged, AND it carries a finite ``target_col`` label (``cyclen``,
    ``f_r`` or ``cbc_max``), AND (when ``library_id`` is given) it was produced
    under that SERVE library.

    ``cbc_max`` additionally drops ``cbc_kind == "boc_only"`` rows, mirroring
    :func:`train._valid_target_values` / :meth:`dataset_torch.PosValDataset._targets`
    exactly: a BOC-only row's boron is not the EDIT2 MAXIMUM the head predicts, so
    calibrating against it would fit the shift to the wrong quantity.  Today those
    rows also carry a NULL ``cbc_max`` (so the finite filter already drops all
    11,259 of them), but the guard is written explicitly because the label being
    absent is a property of the current extractor, not of the contract.

    ``library_id`` accepts one id or a sequence of them.  It is a coarse filter
    kept for direct callers and tests; the fit path passes ``None`` and applies
    :func:`serve_parity_mask` instead, which states the same invariant exactly
    (see that function — the blanket ``"ga80"`` filter it replaced silently
    excluded every paramA row, leaving 50.9% of the curriculum-val slice with no
    calibration at all, forensic 2026-07-29 debug-panel).

    Kept as a small pure function so the "holdout ids never enter the fit"
    invariant is unit testable without loading a 6 MB ensemble.
    """
    rid = records["record_id"].astype(str)
    conv = (records["converged"].astype(bool).to_numpy()
            if "converged" in records.columns else np.ones(len(records), dtype=bool))
    label = np.asarray([_finite(v) for v in records[target_col]], dtype=float)
    mask = rid.isin(train_ids).to_numpy() & conv & np.isfinite(label)
    if target_col == "cbc_max" and "cbc_kind" in records.columns:
        mask = mask & (records["cbc_kind"].astype(str).to_numpy() != "boc_only")
    if library_id is not None and "library_id" in records.columns:
        wanted = ({str(library_id)} if isinstance(library_id, str)
                  else {str(x) for x in library_id})
        mask = mask & records["library_id"].astype(str).isin(wanted).to_numpy()
    return mask


@dataclass
class _FitRows:
    """The rows a calibration fit is allowed to see, plus their serve cell keys.

    Extracted so the scalar fit (:func:`_fit_cell_affine_target`) and the flatness
    fit (:func:`fit_flatness_calibration`) share ONE implementation of row
    admission and cell keying.  That sharing is the point: the 2026-07-29 forensic
    was a fit/serve key divergence, and a second copy of this logic is a second
    chance to reintroduce exactly that bug.
    """

    backend: Any
    sub: Any                       # the admitted store rows (DataFrame)
    patterns: list
    cases: list
    keys: list[str]                # serve-recipe (feed, e_core-bin) cell key
    row_libs: np.ndarray
    dropped_libs: dict[str, int]
    val_ids: set
    split: str

    @classmethod
    def build(cls, model_dir, store_dir, splits_dir, *, split, device,
              library_id, target_col, bin_width) -> "_FitRows":
        from ..data.schema import unpack_pattern
        from ..data.store import StoreReader
        from ..vendor.masterrl.domain import CaseKey
        from .model_api import PosValCnnBackend
        from .splits import SplitManifest

        model_dir = Path(model_dir)
        records = StoreReader(store_dir).records

        # -- resolve the split the champion trained on --------------------- #
        if split is None:
            metas = sorted(model_dir.glob("member_*/meta.json"))
            split = "S1"
            if metas:
                split = str(json.loads(metas[0].read_text(encoding="utf-8"))
                            .get("split", "S1"))
        manifest = SplitManifest.from_json(Path(splits_dir) / f"{split}.json")
        train_ids = set(manifest.record_ids("train"))
        val_ids = set(manifest.record_ids("val"))

        # -- serve-path backend with EVERY calibration disabled for the fit - #
        # A calibration must be fit on the RAW head prediction; disabling every
        # hook makes the fit independent of whichever sibling artifact may already
        # sit in the model dir (order-independent refit).
        backend = PosValCnnBackend.from_dir(
            model_dir, store_dir=store_dir, library_id=library_id, device=device)
        for flag in ("apply_cell_calibration", "apply_fr_calibration",
                     "apply_cbc_calibration", "apply_fq_calibration",
                     "apply_ao_calibration", "apply_flatness_calibration"):
            if hasattr(backend, flag):
                setattr(backend, flag, False)

        # -- labelled train rows (converged + finite target), stored order -- #
        # No library allowlist: admission is serve parity (see serve_parity_mask),
        # so paramA is no longer silently dropped and 260624 is still rejected.
        df = records.copy()
        df["_rid"] = df["record_id"].astype(str)
        keep = fit_row_mask(df, train_ids, library_id=None, target_col=target_col)
        cand = df[keep]
        cand_patterns = [unpack_pattern(str(p)) for p in cand["pattern"]]
        serve_libs = [backend.serve_library(p) for p in cand_patterns]
        parity = serve_parity_mask(cand["library_id"].astype(str).tolist(), serve_libs)
        dropped: dict[str, int] = {}
        for lib, ok in zip(cand["library_id"].astype(str).tolist(), parity):
            if not ok:
                dropped[lib] = dropped.get(lib, 0) + 1

        sub = cand[parity]
        patterns = [p for p, ok in zip(cand_patterns, parity) if ok]
        cases = [CaseKey(pair=str(cp), feed=int(f))
                 for cp, f in zip(sub["case_pair"], sub["feed"])]
        keys = [cyclen_cell_key(c.feed, backend.cyclen_e_core(p)[0], bin_width)
                for p, c in zip(patterns, cases)]
        return cls(backend, sub, patterns, cases, keys,
                   sub["library_id"].astype(str).to_numpy(), dropped,
                   val_ids, split)


def _fit_cells_and_globals(preds: np.ndarray, actual: np.ndarray,
                           keys: Sequence[str], row_libs: np.ndarray, *,
                           min_rows: int, slope_min_rows: int,
                           affine_margin: float,
                           low_weight_thresh: float | None = None,
                           low_weight: float = 1.0,
                           conformal_offset: float = 0.0
                           ) -> tuple[dict, dict, dict, dict]:
    """``(cells, skipped, mixed_library_cells, global_by_library)`` for one target.

    Shared by the scalar and flatness fits.  ``slope_min_rows`` greater than the
    row count forces INTERCEPT-ONLY, which is how the flatness fit guarantees its
    rank-preservation property (a=1 => strictly increasing => within-cell order
    untouched).
    """
    by_cell: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        by_cell.setdefault(key, []).append(i)

    cells: dict[str, dict] = {}
    skipped: dict[str, int] = {}
    # A cell whose rows come from more than one library is RECORDED, not dropped:
    # the key is (feed, e_core-bin) with no library axis, so two libraries sharing
    # a bin would be blended into one shift.  Today they cannot (ga80 spans e_core
    # 5.00-5.50, paramA 5.79-6.36 — disjoint), so this stays empty; if it ever
    # fills, the fix is to put the library INTO the cell key, which is a schema
    # break and therefore a deliberate decision rather than a silent blend.
    mixed: dict[str, list[str]] = {}
    for key, rows in sorted(by_cell.items()):
        idx = np.asarray(rows, dtype=int)
        libs = sorted(set(row_libs[idx].tolist()))
        if len(libs) > 1:
            mixed[key] = libs
        fit = fit_affine_cell(preds[idx], actual[idx],
                              min_rows=min_rows, slope_min_rows=slope_min_rows,
                              affine_margin=affine_margin,
                              low_weight_thresh=low_weight_thresh,
                              low_weight=low_weight,
                              conformal_offset=conformal_offset)
        if fit is None:
            skipped[key] = int(idx.size)
        else:
            cells[key] = fit

    # -- per-library global fallback --------------------------------------- #
    # The shift for a row whose cell missed ``min_rows``.  Fitted per LIBRARY, not
    # pooled, because the bias is library-dependent (measured cbc_max curriculum-val
    # bias: ga80 +9.2 ppm, paramA +49.0 ppm — one pooled median is wrong for both).
    # ``min_rows`` is reused as the admission bar so a library with a handful of
    # rows cannot mint a confident global shift.  ALWAYS intercept-only: a global
    # slope across a whole library is a far stronger claim than a global offset.
    globals_by_lib: dict[str, dict] = {}
    for lib in sorted(set(row_libs.tolist())):
        sel = np.flatnonzero(row_libs == lib)
        gfit = fit_affine_cell(preds[sel], actual[sel],
                               min_rows=min_rows,
                               slope_min_rows=len(sel) + 1,   # intercept ONLY
                               affine_margin=affine_margin,
                               low_weight_thresh=low_weight_thresh,
                               low_weight=low_weight,
                               conformal_offset=conformal_offset)
        if gfit is not None:
            globals_by_lib[lib] = gfit
    return cells, skipped, mixed, globals_by_lib


def _fit_cell_affine_target(
    model_dir: str | Path,
    store_dir: str | Path,
    splits_dir: str | Path,
    *,
    target_col_name: str,
    surrogate_col: int,
    col_key_name: str,
    schema: str,
    target_label: str,
    out_name: str,
    affine_margin: float,
    split: str | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    slope_min_rows: int = DEFAULT_SLOPE_MIN_ROWS,
    bin_width: float = DEFAULT_BIN_WIDTH,
    device: str = "cpu",
    library_id: str = "ga80",
    batch_size: int = 1024,
    write: bool = True,
    low_weight_thresh: float | None = None,
    low_weight: float = 1.0,
    conformal_offset: float = 0.0,
) -> dict:
    """Shared per-cell affine-calibration fit for one target (cyclen / F_r / CBC).

    Predictions are produced through the exact SERVE path
    (:meth:`PosValCnnBackend.predict` with EVERY cell calibration disabled), on
    the champion's split-manifest **train** rows only — never the honest holdout
    (``val``) rows — so the artifact corrects the shift the model actually emits
    at serve time without leaking the eval fold.  Rows are grouped by
    :func:`cyclen_cell_key` (feed + serve-recipe e_core bin — target-independent);
    a cell with ``>= min_rows`` labelled rows gets a correction, and every other
    row falls back to its LIBRARY's pooled shift (``global_by_library``).

    **Row admission (changed 2026-07-29 debug-panel).**  Rows are no longer
    filtered to ``library_id == "ga80"``.  That filter excluded every ``paramA``
    row, so half of the curriculum-val slice (1,361 of 2,676 rows) could not be
    corrected at all and the artifact's cells spanned only e_core 5.00-5.50 while
    paramA lives at 5.79-6.36.  Admission is now the invariant the filter was a
    proxy for — :func:`serve_parity_mask`, "the serve path resolves this pattern
    back to the library it was produced under" — which admits ga80 AND paramA and
    still rejects the ambiguous ``260624`` / ``5.8_5.1`` rosters.  ``library_id``
    remains the backend's SERVE library and is unchanged: the fit must measure the
    prediction the campaign will actually emit.

    ``split`` defaults to the value in the members' ``meta.json`` (the split the
    champion trained on, e.g. ``S1``).  Returns the full artifact dict; when
    ``write`` it is also atomically written to ``model_dir/out_name``.

    ``model_dir`` is coerced to a :class:`~pathlib.Path` up front.  It used not to
    be, and the only place that mattered was the ``model_dir / out_name`` join at
    the very END of the fit: a ``str`` caller (``curriculum._fit_cell_calibrations``
    passes whatever ``_retrain_local_full`` returned, which is a ``str``) did the
    entire multi-minute serve-path fit and then died on
    ``TypeError: unsupported operand type(s) for /: 'str' and 'str'`` — inside
    ``train.fit_cell_calibrations``'s per-target ``except Exception``, which
    printed one ``WARNING:`` line and moved on.  The result was a model dir
    silently missing calibration artifacts.  Coercing here is the root fix; the
    loud reporting in :func:`lpopt.model.train.fit_cell_calibrations` is the
    second line of defence.
    """
    model_dir = Path(model_dir)
    fit = _FitRows.build(model_dir, store_dir, splits_dir, split=split,
                         device=device, library_id=library_id,
                         target_col=target_col_name, bin_width=bin_width)
    backend, sub, patterns, cases = fit.backend, fit.sub, fit.patterns, fit.cases
    keys, row_libs, dropped_libs = fit.keys, fit.row_libs, fit.dropped_libs
    val_ids, split = fit.val_ids, fit.split
    actual = np.asarray([_finite(v) for v in sub[target_col_name]], dtype=float)

    # -- batched serve-path prediction ------------------------------------- #
    preds = np.empty(len(patterns), dtype=float)
    for start in range(0, len(patterns), int(batch_size)):
        stop = start + int(batch_size)
        bp, bc = patterns[start:stop], cases[start:stop]
        if not bp:
            continue
        pred = backend.predict(bp, bc, 0.0)
        preds[start:stop] = np.asarray(pred.mean, dtype=float)[:, surrogate_col]

    cells, skipped, mixed, globals_by_lib = _fit_cells_and_globals(
        preds, actual, keys, row_libs,
        min_rows=min_rows, slope_min_rows=slope_min_rows,
        affine_margin=affine_margin, low_weight_thresh=low_weight_thresh,
        low_weight=low_weight, conformal_offset=conformal_offset)

    artifact = {
        "schema": schema,
        "target": target_label,
        col_key_name: surrogate_col,
        "model_dir": str(model_dir),
        "split": split,
        "bin_width": float(bin_width),
        "min_rows": int(min_rows),
        "slope_min_rows": int(slope_min_rows),
        "library_id": library_id,
        "fit_libraries": sorted(set(row_libs.tolist())),
        "dropped_serve_parity": dropped_libs,
        "n_train_labelled": int(len(patterns)),
        "n_cells_fitted": len(cells),
        "n_cells_skipped": len(skipped),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cells": cells,
        "skipped_cells": skipped,
        "mixed_library_cells": mixed,
        "global_by_library": globals_by_lib,
    }
    # leakage guard: the fit must never have touched a holdout row.
    assert not (set(sub["_rid"]) & val_ids), \
        "cell-calibration fit set intersects the honest holdout (val) fold"

    if write:
        _atomic_write_json(model_dir / out_name, artifact)
    return artifact


def fit_cell_affine(
    model_dir: str | Path,
    store_dir: str | Path,
    splits_dir: str | Path,
    *,
    split: str | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    slope_min_rows: int = DEFAULT_SLOPE_MIN_ROWS,
    bin_width: float = DEFAULT_BIN_WIDTH,
    device: str = "cpu",
    library_id: str = "ga80",
    batch_size: int = 1024,
    write: bool = True,
    out_name: str = CELL_CALIB_NAME,
) -> dict:
    """Fit + persist the per-cell **cyclen** affine calibration for a champion.

    Thin wrapper over :func:`_fit_cell_affine_target` (see it for the leakage-safe
    train-only fit + cell keying); writes ``cell_calibration.json`` (surrogate
    column 3) and corrects the champion's uniform per-cell cyclen over-prediction.
    """
    return _fit_cell_affine_target(
        model_dir, store_dir, splits_dir,
        target_col_name="cyclen", surrogate_col=CYCLEN_COL,
        col_key_name="cyclen_col", schema=CELL_CALIB_SCHEMA,
        target_label="cyclen", out_name=out_name,
        affine_margin=_AFFINE_MARGIN_EFPD,
        split=split, min_rows=min_rows, slope_min_rows=slope_min_rows,
        bin_width=bin_width, device=device, library_id=library_id,
        batch_size=batch_size, write=write,
    )


#: Boundary conservatization defaults for the F_r calibration (forensic
#: parity_round1c_20260722.md): fit the per-cell shift where the search operates
#: (F_r <= 1.7) by up-weighting those rows, and offer a conformal offset to
#: counter the champion's measured global non-conservative bias (−0.084).
_FR_LOW_WEIGHT_THRESH = 1.7
_FR_LOW_WEIGHT = 3.0
_FR_CONFORMAL_OFFSET = 0.0        # opt-in; retrain/CLI may pass +0.084 to fully offset


def fit_cell_affine_fr(
    model_dir: str | Path,
    store_dir: str | Path,
    splits_dir: str | Path,
    *,
    split: str | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    slope_min_rows: int = DEFAULT_SLOPE_MIN_ROWS,
    bin_width: float = DEFAULT_BIN_WIDTH,
    device: str = "cpu",
    library_id: str = "ga80",
    batch_size: int = 1024,
    write: bool = True,
    out_name: str = FR_CALIB_NAME,
    low_weight_thresh: float | None = _FR_LOW_WEIGHT_THRESH,
    low_weight: float = _FR_LOW_WEIGHT,
    conformal_offset: float = _FR_CONFORMAL_OFFSET,
) -> dict:
    """Fit + persist the per-cell **F_r** affine calibration for a champion.

    Mirror of :func:`fit_cell_affine` for surrogate column 0 (F_r): the champion
    under-predicts F_r by a near-uniform per-cell shift (non-conservative vs the
    ``F_r <= 1.55`` feasibility limit), which the intercept-only correction
    removes.  Writes a SEPARATE ``f_r_calibration.json`` (the cyclen artifact
    schema is untouched).  Applied at serve time to ``predict().mean[:, 0]`` only.

    Boundary conservatization (default ON, forensic parity_round1c_20260722): the
    intercept is a F_r<=``low_weight_thresh``-up-weighted median, so the shift is
    fit in the low-F_r boundary the 1.55 search queries (not the crowded high-F_r
    bulk), and ``conformal_offset`` optionally adds a fixed conservative margin to
    offset the champion's global −0.084 under-prediction.  Pass
    ``low_weight_thresh=None`` to recover the plain unweighted fit.
    """
    return _fit_cell_affine_target(
        model_dir, store_dir, splits_dir,
        target_col_name="f_r", surrogate_col=FR_COL,
        col_key_name="f_r_col", schema=FR_CALIB_SCHEMA,
        target_label="f_r", out_name=out_name,
        affine_margin=_AFFINE_MARGIN_FR,
        split=split, min_rows=min_rows, slope_min_rows=slope_min_rows,
        bin_width=bin_width, device=device, library_id=library_id,
        batch_size=batch_size, write=write,
        low_weight_thresh=low_weight_thresh, low_weight=low_weight,
        conformal_offset=conformal_offset,
    )


def fit_cell_affine_cbc(
    model_dir: str | Path,
    store_dir: str | Path,
    splits_dir: str | Path,
    *,
    split: str | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    slope_min_rows: int = DEFAULT_SLOPE_MIN_ROWS,
    bin_width: float = DEFAULT_BIN_WIDTH,
    device: str = "cpu",
    library_id: str = "ga80",
    batch_size: int = 1024,
    write: bool = True,
    out_name: str = CBC_CALIB_NAME,
) -> dict:
    """Fit + persist the per-cell **CBC_max** affine calibration for a champion.

    Mirror of :func:`fit_cell_affine` for SURROGATE column 1 (CBC_max).  Measured
    2026-07-29 on the curriculum holdout of ``data/models/20260729_054749``: MAE
    42.4 ppm, 36% of it per-group BIAS (campaign|case_pair|feed), global +27 ppm
    over-prediction, per-group up to +113 ppm — i.e. the same near-constant
    per-cell shift the cyclen head carries, which the intercept-only correction
    removes.  Writes a SEPARATE ``cbc_calibration.json``; applied at serve time to
    ``predict().mean[:, 1]`` only.

    NO conservatism knobs (unlike the F_r fit): the reward stack treats CBC_max as
    a lower-is-better CONSTRAINT (``cbc_limit`` 1550 ppm), and an over-predicting
    model is already the conservative side of that limit.  The point of this
    artifact is ACCURACY (the MASTER-verified debug panel's 20 ppm tolerance), so
    it corrects the shift symmetrically and adds no margin; a deliberate
    conservative offset would have to be argued for separately.
    """
    return _fit_cell_affine_target(
        model_dir, store_dir, splits_dir,
        target_col_name="cbc_max", surrogate_col=CBC_COL,
        col_key_name="cbc_max_col", schema=CBC_CALIB_SCHEMA,
        target_label="cbc_max", out_name=out_name,
        affine_margin=_AFFINE_MARGIN_CBC,
        split=split, min_rows=min_rows, slope_min_rows=slope_min_rows,
        bin_width=bin_width, device=device, library_id=library_id,
        batch_size=batch_size, write=write,
    )


def fit_cell_affine_fq(
    model_dir: str | Path, store_dir: str | Path, splits_dir: str | Path, *,
    split: str | None = None, min_rows: int = DEFAULT_MIN_ROWS,
    slope_min_rows: int = DEFAULT_SLOPE_MIN_ROWS,
    bin_width: float = DEFAULT_BIN_WIDTH, device: str = "cpu",
    library_id: str = "ga80", batch_size: int = 1024, write: bool = True,
    out_name: str = FQ_CALIB_NAME,
) -> dict:
    """Fit + persist the per-cell **F_q** affine calibration (surrogate column 2).

    F_q carries the largest bias SHARE of any scalar target: curriculum-val MAE
    0.250 with a −0.178 mean bias, i.e. 71% of the error is a uniform
    UNDER-prediction.  Like F_r that is the non-conservative direction against the
    F_q <= 2.41 limit, so removing it also tightens the constraint the search sees.

    No conservatism knobs: unlike F_r there is no measured boundary-vs-bulk split
    to weight for, so the plain per-cell shift is fit and an explicit margin, if
    one is ever wanted, should be argued for on its own.
    """
    return _fit_cell_affine_target(
        model_dir, store_dir, splits_dir,
        target_col_name="f_q", surrogate_col=FQ_COL,
        col_key_name="f_q_col", schema=FQ_CALIB_SCHEMA,
        target_label="f_q", out_name=out_name,
        affine_margin=_AFFINE_MARGIN_FQ,
        split=split, min_rows=min_rows, slope_min_rows=slope_min_rows,
        bin_width=bin_width, device=device, library_id=library_id,
        batch_size=batch_size, write=write,
    )


def fit_cell_affine_ao(
    model_dir: str | Path, store_dir: str | Path, splits_dir: str | Path, *,
    split: str | None = None, min_rows: int = DEFAULT_MIN_ROWS,
    slope_min_rows: int = DEFAULT_SLOPE_MIN_ROWS,
    bin_width: float = DEFAULT_BIN_WIDTH, device: str = "cpu",
    library_id: str = "ga80", batch_size: int = 1024, write: bool = True,
    out_name: str = AO_CALIB_NAME,
) -> dict:
    """Fit + persist the per-cell **|AO|** affine calibration (surrogate column 4).

    The smallest of the five scalar corrections — curriculum-val MAE 0.0060 with a
    −0.0010 bias, so only 17% of the error is a shift and the cross-fitted MAE gain
    is ~3%.  It is fitted anyway because it removes ~70% of that residual bias for
    the same one-line serve cost, and |AO| is a licensing-reported axis where a
    known systematic offset should not be left in on the grounds that the scatter
    is larger.  ``ao_abs`` is a magnitude (>= 0); the per-cell shift can in
    principle push a near-zero prediction negative, which is physically meaningless
    but harmless for the |AO| <= 0.30 constraint (it can only read as MORE
    feasible, never less, and the top-K is MASTER-verified).
    """
    return _fit_cell_affine_target(
        model_dir, store_dir, splits_dir,
        target_col_name="ao_abs", surrogate_col=AO_COL,
        col_key_name="ao_abs_col", schema=AO_CALIB_SCHEMA,
        target_label="ao_abs", out_name=out_name,
        affine_margin=_AFFINE_MARGIN_AO,
        split=split, min_rows=min_rows, slope_min_rows=slope_min_rows,
        bin_width=bin_width, device=device, library_id=library_id,
        batch_size=batch_size, write=write,
    )


# --------------------------------------------------------------------------- #
# flatness (map-head) calibration — INTERCEPT-ONLY, both axes in one artifact
# --------------------------------------------------------------------------- #
def fit_flatness_calibration(
    model_dir: str | Path, store_dir: str | Path, splits_dir: str | Path, *,
    split: str | None = None, min_rows: int = DEFAULT_MIN_ROWS,
    bin_width: float = DEFAULT_BIN_WIDTH, device: str = "cpu",
    library_id: str = "ga80", batch_size: int = 512, write: bool = True,
    out_name: str = FLAT_CALIB_NAME,
) -> dict:
    """Fit + persist the per-cell **node_peak / map_cov** INTERCEPT-only calibration.

    The map head is optimistic about both flatness axes: curriculum-val mean bias
    −0.0462 on ``node_peak`` (38% of a 0.121 MAE) and −0.0272 on ``map_cov`` (84%
    of a 0.033 MAE).  The objective consumes these as LEVELS, so an offset walks
    the whole flat-power ranking's scale.

    **INTERCEPT-ONLY, structurally** (``slope_min_rows`` is forced above the row
    count, so :func:`fit_affine_cell` can never adopt a slope).  Two reasons, and
    the first is the load-bearing one:

    1. ``a == 1`` makes the correction a pure translation, so WITHIN a calibration
       cell the order of candidates is bit-identical before and after.  The honest
       no-regression gate ranks within cells, so the correction cannot manufacture
       (or hide) a ranking change there.  NOTE the honest scope: a CURRICULUM cell
       (campaign, e.g. ``5-5.25_f117``) spans several 0.05-wide calibration cells,
       and two rows in different bins get different shifts, so ranks CAN move
       across bins — exactly as they already can for the four scalar calibrations.
       The guarantee proved here is per calibration cell.
    2. Both axes are bounded, low-variance statistics of a 69-slot map; a fitted
       slope on such a quantity is far more likely to be sampling noise than a real
       distortion, and the measured bias share says a shift is what is there.

    Both axes live in ONE artifact because they come from a single
    ``predict_map_flatness`` forward — splitting them would double the fit cost for
    no benefit.  ``cells`` and ``global_by_library`` are therefore keyed by TARGET
    first (see :func:`flatness_cells`).  Rows without a ``node_peak`` / ``map_cov``
    label (the map was not harvested) simply do not contribute to that axis.
    """
    model_dir = Path(model_dir)
    fit = _FitRows.build(model_dir, store_dir, splits_dir, split=split,
                         device=device, library_id=library_id,
                         target_col="node_peak", bin_width=bin_width)
    n = len(fit.patterns)
    peak = np.full(n, np.nan)
    cov = np.full(n, np.nan)
    for start in range(0, n, int(batch_size)):
        stop = start + int(batch_size)
        bp, bc = fit.patterns[start:stop], fit.cases[start:stop]
        if not bp:
            continue
        pk_m, _pk_s, cv_m, _cv_s = fit.backend.predict_map_flatness(bp, bc)
        peak[start:stop] = np.asarray(pk_m, dtype=float)
        cov[start:stop] = np.asarray(cv_m, dtype=float)

    preds = {"node_peak": peak, "map_cov": cov}
    cells: dict[str, dict] = {}
    skipped: dict[str, dict] = {}
    mixed: dict[str, dict] = {}
    gmap: dict[str, dict] = {}
    for tgt in FLATNESS_TARGETS:
        actual = np.asarray([_finite(v) for v in fit.sub[tgt]], dtype=float)
        c, s, m, g = _fit_cells_and_globals(
            preds[tgt], actual, fit.keys, fit.row_libs,
            min_rows=min_rows,
            slope_min_rows=n + 1,          # INTERCEPT-ONLY (see the docstring)
            affine_margin=float("inf"))    # …belt and braces: no slope can win
        cells[tgt], skipped[tgt], mixed[tgt], gmap[tgt] = c, s, m, g

    artifact = {
        "schema": FLAT_CALIB_SCHEMA,
        "targets": list(FLATNESS_TARGETS),
        "estimator": "intercept",
        "model_dir": str(model_dir),
        "split": fit.split,
        "bin_width": float(bin_width),
        "min_rows": int(min_rows),
        "library_id": library_id,
        "fit_libraries": sorted(set(fit.row_libs.tolist())),
        "dropped_serve_parity": fit.dropped_libs,
        "n_train_labelled": {t: int(np.isfinite(preds[t]).sum())
                             for t in FLATNESS_TARGETS},
        "n_cells_fitted": {t: len(cells[t]) for t in FLATNESS_TARGETS},
        "n_cells_skipped": {t: len(skipped[t]) for t in FLATNESS_TARGETS},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cells": cells,
        "skipped_cells": skipped,
        "mixed_library_cells": mixed,
        "global_by_library": gmap,
    }
    assert not (set(fit.sub["_rid"]) & fit.val_ids), \
        "flatness-calibration fit set intersects the honest holdout (val) fold"
    if write:
        _atomic_write_json(Path(model_dir) / out_name, artifact)
    return artifact


def flatness_cells(calib: Mapping[str, Any] | None, target: str) -> dict[str, dict]:
    """The ``{cell_key: {a, b, ...}}`` map for ONE flatness axis (empty if absent)."""
    if not calib:
        return {}
    cells = (calib.get("cells") or {}).get(target)
    return dict(cells) if isinstance(cells, Mapping) else {}


def flatness_global_by_library(calib: Mapping[str, Any] | None,
                               target: str) -> dict[str, dict]:
    """The per-library fallback map for ONE flatness axis (empty if absent)."""
    if not calib:
        return {}
    g = (calib.get("global_by_library") or {}).get(target)
    return dict(g) if isinstance(g, Mapping) else {}


def _finite(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _atomic_write_json(path: str | Path, obj: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


# --------------------------------------------------------------------------- #
# Stage 1: serve-side load + apply
# --------------------------------------------------------------------------- #
def load_cell_calibration(path: str | Path) -> dict:
    """Load a ``cell_calibration.json`` / ``f_r_calibration.json`` artifact (full
    dict, incl. metadata).  Both files share the ``{schema, cells: {...}}`` shape."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


#: F_r artifacts share the exact JSON shape, so the loader is the same.
load_fr_calibration = load_cell_calibration
#: …and so do the CBC_max / F_q / |AO| artifacts (identical JSON shape), and the
#: flatness artifact (a different SHAPE — target-keyed cells — but plain JSON, so
#: the same loader; :func:`flatness_cells` is what knows about the shape).
load_cbc_calibration = load_cell_calibration
load_fq_calibration = load_cell_calibration
load_ao_calibration = load_cell_calibration
load_flatness_calibration = load_cell_calibration


def calibration_cells(calib: Mapping[str, Any] | None) -> dict[str, dict]:
    """The ``{cell_key: {a, b, ...}}`` map from an artifact (empty when absent)."""
    if not calib:
        return {}
    cells = calib.get("cells")
    return dict(cells) if isinstance(cells, Mapping) else {}


def global_by_library(calib: Mapping[str, Any] | None) -> dict[str, dict]:
    """The ``{library_id: {a, b, ...}}`` per-library fallback map (empty when absent).

    Absent on every artifact fitted before 2026-07-29, which is the backward-compat
    contract: no key -> empty map -> :func:`apply_affine_calibration` behaves
    exactly as it did before the fallback existed.
    """
    if not calib:
        return {}
    g = calib.get("global_by_library")
    return dict(g) if isinstance(g, Mapping) else {}


def apply_affine_calibration(values: np.ndarray, cell_keys: Sequence[str],
                             cells: Mapping[str, Any],
                             *, globals_by_lib: Mapping[str, Any] | None = None,
                             libraries: Sequence[str] | None = None) -> np.ndarray:
    """Return ``values`` with ``a*x + b`` applied for each fitted cell (else id).

    Target-agnostic (cyclen / F_r / CBC_max): pure/vectorized; NaNs and unfitted
    cells pass through unchanged.  Operates on a copy — the caller's array is never
    mutated.  Because every fitted ``a`` is > 0 (see :func:`fit_affine_cell`), the
    map is strictly increasing, so within-cell ranking is preserved (gate-neutral).

    **Per-library global fallback** (2026-07-29 debug-panel).  A cell that missed
    ``min_rows`` used to fall through UNCORRECTED, which let the whole global bias
    component survive purely because a cell was thinly sampled — the measured
    champion carried a +27 ppm global cbc_max bias while 17 of its 74 cells were
    skipped.  When ``globals_by_lib`` and ``libraries`` are supplied, such a row is
    corrected by its OWN library's pooled shift instead.

    The fallback is keyed by LIBRARY, not pooled globally, because the bias is
    strongly library-dependent: on the curriculum-val slice the champion's cbc_max
    bias is +9.2 ppm for ga80 and +49.0 ppm for paramA.  One pooled median would be
    wrong by ~20 ppm for both, i.e. worse than no fallback for the larger group.

    A row whose library has NO fitted global entry is left UNCORRECTED (identity).
    That is deliberate for out-of-distribution / unknown-provenance rows: the
    central forensic finding here is that cbc_max bias varies by provenance by
    100-400 ppm (Dataset A ``mocha_native`` vs Dataset P MASTER-native at matched
    cells), so extrapolating a fitted regime's median onto an unseen regime would
    transfer the estimate along exactly the axis it varies most on.  An uncorrected
    row is visibly wrong in the debug panel; a confidently mis-corrected one is not.
    """
    out = np.array(values, dtype=float, copy=True)
    gmap = dict(globals_by_lib or {})
    if not cells and not gmap:
        return out
    for i, key in enumerate(cell_keys):
        x = out[i]
        if not math.isfinite(x):
            continue
        params = cells.get(key)
        if params is None:
            if not gmap or libraries is None:
                continue
            params = gmap.get(str(libraries[i]))
            if params is None:
                continue
        out[i] = float(params["a"]) * x + float(params["b"])
    return out


#: Back-compat alias — the apply is target-agnostic (kept for the cyclen callers).
apply_cyclen_calibration = apply_affine_calibration


# --------------------------------------------------------------------------- #
# Stage 2: campaign running bias corrector
# --------------------------------------------------------------------------- #
class CampaignBiasCorrector:
    """Running per-cell cyclen bias for cells with NO fitted Stage-1 calibration.

    Accumulates ``delta = pred - actual`` from THIS campaign's MASTER-verified
    chains per (feed, e_core-bin) cell and returns a shrunk estimate
    ``bias_hat = n/(n+prior_weight) * median(delta)`` that subsequent
    screening/deepen predictions subtract from their cyclen mean.  A cell already
    covered by Stage 1 is never observed and always returns ``0`` (no double
    correction).  Fully JSON round-trippable for campaign resume.
    """

    def __init__(self, *, prior_weight: float = DEFAULT_PRIOR_WEIGHT,
                 bin_width: float = DEFAULT_BIN_WIDTH,
                 fitted_cells: Sequence[str] | None = None) -> None:
        self.prior_weight = float(prior_weight)
        self.bin_width = float(bin_width)
        self.fitted_cells: set[str] = set(fitted_cells or ())
        self._deltas: dict[str, list[float]] = {}

    # -- keying ------------------------------------------------------------ #
    def key(self, feed: Any, e_core: float | None) -> str:
        return cyclen_cell_key(feed, e_core, self.bin_width)

    # -- observe ----------------------------------------------------------- #
    def observe(self, cell_key: str, pred_cyclen: float, actual_cyclen: float) -> bool:
        """Record one verified ``pred - actual`` delta for ``cell_key``.

        No-op (returns ``False``) when the cell already has a Stage-1 calibration
        or either value is non-finite — so a fitted cell is never double-counted
        and a failed/censored verification never poisons the estimate.
        """
        if cell_key in self.fitted_cells:
            return False
        p, a = _finite(pred_cyclen), _finite(actual_cyclen)
        if not (math.isfinite(p) and math.isfinite(a)):
            return False
        self._deltas.setdefault(cell_key, []).append(float(p - a))
        return True

    def observe_feed_ecore(self, feed: Any, e_core: float | None,
                           pred_cyclen: float, actual_cyclen: float) -> bool:
        return self.observe(self.key(feed, e_core), pred_cyclen, actual_cyclen)

    # -- estimate ---------------------------------------------------------- #
    def bias(self, cell_key: str) -> float:
        """Shrunk cyclen bias estimate for ``cell_key`` (0 for fitted/empty cells)."""
        if cell_key in self.fitted_cells:
            return 0.0
        deltas = self._deltas.get(cell_key)
        if not deltas:
            return 0.0
        n = len(deltas)
        shrink = n / (n + self.prior_weight)
        return float(shrink * float(np.median(deltas)))

    def bias_feed_ecore(self, feed: Any, e_core: float | None) -> float:
        return self.bias(self.key(feed, e_core))

    def correct(self, cell_key: str, pred_cyclen: float) -> float:
        """``pred_cyclen - bias(cell_key)`` (identity for fitted/empty cells)."""
        return float(pred_cyclen) - self.bias(cell_key)

    def n_obs(self, cell_key: str) -> int:
        return len(self._deltas.get(cell_key, ()))

    @property
    def active(self) -> bool:
        """True once any non-fitted cell has at least one observation."""
        return any(self._deltas.get(k) for k in self._deltas
                   if k not in self.fitted_cells)

    # -- persistence (resume round-trip) ----------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "campaign_cyclen_bias_v1",
            "prior_weight": self.prior_weight,
            "bin_width": self.bin_width,
            "fitted_cells": sorted(self.fitted_cells),
            "deltas": {k: list(v) for k, v in self._deltas.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CampaignBiasCorrector":
        obj = cls(
            prior_weight=float(d.get("prior_weight", DEFAULT_PRIOR_WEIGHT)),
            bin_width=float(d.get("bin_width", DEFAULT_BIN_WIDTH)),
            fitted_cells=d.get("fitted_cells", ()),
        )
        for k, v in (d.get("deltas") or {}).items():
            obj._deltas[str(k)] = [float(x) for x in v]
        return obj

    def save(self, path: str | Path) -> Path:
        return _atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "CampaignBiasCorrector":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "AO_CALIB_NAME",
    "AO_CALIB_SCHEMA",
    "AO_COL",
    "CBC_CALIB_NAME",
    "CBC_CALIB_SCHEMA",
    "CBC_COL",
    "CELL_CALIB_NAME",
    "CELL_CALIB_SCHEMA",
    "CYCLEN_COL",
    "FLATNESS_TARGETS",
    "FLAT_CALIB_NAME",
    "FLAT_CALIB_SCHEMA",
    "FQ_CALIB_NAME",
    "FQ_CALIB_SCHEMA",
    "FQ_COL",
    "FR_CALIB_NAME",
    "FR_CALIB_SCHEMA",
    "FR_COL",
    "CampaignBiasCorrector",
    "apply_affine_calibration",
    "apply_cyclen_calibration",
    "calibration_cells",
    "cyclen_cell_key",
    "fit_affine_cell",
    "fit_cell_affine",
    "fit_cell_affine_ao",
    "fit_cell_affine_cbc",
    "fit_cell_affine_fq",
    "fit_cell_affine_fr",
    "fit_flatness_calibration",
    "fit_row_mask",
    "flatness_cells",
    "flatness_global_by_library",
    "global_by_library",
    "load_ao_calibration",
    "load_cbc_calibration",
    "load_cell_calibration",
    "load_flatness_calibration",
    "load_fq_calibration",
    "load_fr_calibration",
    "serve_parity_mask",
]
