"""Position-value model interface (plan sec. 4.5).

:class:`PositionValueModel` is the ``Protocol`` the campaign consumes; its
``predict`` returns the vendor :class:`SurrogatePrediction` 7-column layout so
``reward.RewardModel`` is reused unmodified.  :class:`PosValCnnBackend` implements
it over the PosValNet deep ensemble.

**Column contract (verified against ``surrogate.TARGET_NAMES``):**
``(F_r, CBC_max, F_q, cyclen, AO_abs, max_assembly_burnup, max_pin_burnup)``.

Phase D (plan sec. 12.4) promotes ``discharge_burnup`` and ``max_pin_burnup`` to
first-class network targets — dataset order becomes ``(f_r, f_q, cbc_max, cyclen,
ao_abs, discharge_burnup, max_pin_burnup)``.  The 7 columns are filled as:

* the five standard axes reorder into surrogate columns ``[0, 2, 1, 3, 4]``
  (``f_r→F_r``, ``cbc_max→CBC_max``, ``f_q→F_q``, ``cyclen→cyclen``,
  ``ao_abs→AO_abs``);
* ``max_pin_burnup`` is an **exact-match** into surrogate column 6
  (``max_pin_burnup``) — the reward stack's optional pin-burnup constraint axis
  now carries a real model, so a licensing limit gates it honestly;
* surrogate column 5 (``max_assembly_burnup``) stays **NaN**.  Our
  ``discharge_burnup`` target is the AVERAGE discharge burnup — a *target-distance*
  axis (like cyclen), NOT the vendor's MAX-assembly-burnup lower-is-better
  CONSTRAINT.  ``reward.RewardModel`` treats column 5 as a constraint that, when
  its limit is enabled, penalizes values above it; routing an average discharge
  burnup through that column would corrupt the assembly-burnup gate.  So we take
  the least-invasive correct option: leave column 5 unknown (reward keeps
  consuming predictions unchanged) and expose ``discharge_burnup`` through a
  dedicated :meth:`PosValCnnBackend.predict_extra` accessor (:class:`ExtraPrediction`),
  which the ``user_criteria`` acquisition reads for its discharge-burnup target.

A cond_v2 checkpoint predicts only the first five targets; its column-6/5 stay
NaN and ``predict_extra`` returns NaN for ``discharge_burnup`` — the encoder and
mapping are rebuilt from the checkpoint's own ``cond_schema`` / ``target_names``,
and a mixed-schema ensemble is rejected.

``sklearn_fallback`` (vendor ``SurrogateEnsemble``) is deferred to M4 (noted).
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
import torch

from ..data.fuel_types import FuelLibrary, core_enrichment_split, resolve_type_id
from ..vendor.masterrl.domain import CaseKey, Pattern
from ..vendor.masterrl.surrogate import SurrogatePrediction, TARGET_NAMES
from .calibrate import CALIB_NAME, apply_calibration, apply_platt, load_calibration
from .cell_calibrate import (
    AO_CALIB_NAME, CBC_CALIB_NAME, CELL_CALIB_NAME, DEFAULT_BIN_WIDTH,
    FLATNESS_TARGETS, FLAT_CALIB_NAME, FQ_CALIB_NAME, FR_CALIB_NAME,
    apply_affine_calibration, apply_cyclen_calibration, calibration_cells,
    cyclen_cell_key, flatness_cells, flatness_global_by_library,
    global_by_library, load_ao_calibration, load_cbc_calibration,
    load_cell_calibration, load_flatness_calibration, load_fq_calibration,
    load_fr_calibration,
)
from .conformal import (
    CONFORMAL_NAME, DEFAULT_BIN_WIDTH as CONFORMAL_BIN_WIDTH,
    conformal_targets, interval_arrays, load_conformal,
)
from .dataset_torch import TARGETS
from .featurize import FeatureEncoder, RecordInputs, library_provenance
from .pinbu_physics import (
    PINBU_PHYSICS_NAME, PINBU_SURROGATE_COL, PinBuPhysicsEstimator,
    load_pinbu_physics,
)
from .net import PosValNet, PosValNetConfig
from .ood_guard import (
    DEFAULT_MARGIN as _OOD_MARGIN,
    feature_ood_vecs,
    format_ood_warning,
    population_envelope_from_library,
)
from .train import (
    TrainConfig, convergence_loss, load_member, map_loss, norm_from_meta,
    regression_loss, save_member,
)

#: dataset target NAME -> surrogate ``TARGET_NAMES`` column, for the
#: :class:`SurrogatePrediction` fill.  ``discharge_burnup`` is deliberately
#: ABSENT: it is NOT the vendor's ``max_assembly_burnup`` (column 5) constraint
#: axis (see the module docstring) and is served via :meth:`predict_extra`.
#: surrogate cols: 0 F_r, 1 CBC_max, 2 F_q, 3 cyclen, 4 AO_abs,
#: 5 max_assembly_burnup, 6 max_pin_burnup.
#: ``max_assembly_burnup`` maps to surrogate column 5 — its EXACT vendor
#: counterpart (the MAX-assembly-burnup lower-is-better constraint).  It appears
#: in ``target_names`` only for a checkpoint trained with ``promote_max_asm_bu``;
#: for every existing checkpoint the name is absent, ``_to_surrogate`` skips it,
#: and column 5 stays NaN exactly as before.  Note this is NOT our
#: ``discharge_burnup`` target (core AVERAGE, a target-distance axis), which
#: remains deliberately excluded from the surrogate layout — routing an average
#: through the constraint column would corrupt the assembly-burnup gate.
_TARGET_TO_SURROGATE_COL: dict[str, int] = {
    "f_r": 0,
    "cbc_max": 1,
    "f_q": 2,
    "cyclen": 3,
    "ao_abs": 4,
    "max_assembly_burnup": 5,
    "max_pin_burnup": 6,
}
#: Dataset targets exposed via :meth:`predict_extra` instead of the 7-column
#: surrogate layout (NaN-filled for a checkpoint that does not predict them).
_EXTRA_TARGET_NAMES: tuple[str, ...] = ("discharge_burnup",)
#: The 5-target cond_v2 dataset order (default for a checkpoint lacking
#: ``target_names`` in its meta, and for the bare ``_to_surrogate`` contract).
_LEGACY_TARGET_NAMES: tuple[str, ...] = ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs")
_N_SURROGATE = len(TARGET_NAMES)             # 7
#: surrogate column the per-cell affine cyclen calibration writes (== index 3).
_CYCLEN_SURROGATE_COL = _TARGET_TO_SURROGATE_COL["cyclen"]
#: surrogate column the per-cell affine F_r calibration writes (== index 0).
_FR_SURROGATE_COL = _TARGET_TO_SURROGATE_COL["f_r"]
#: surrogate column the per-cell affine CBC_max calibration writes (== index 1).
#: NOTE this is the SURROGATE index, not the dataset-target index (cbc_max is
#: target 2 in the ``(f_r, f_q, cbc_max, ...)`` dataset order) — the calibration
#: hook operates on ``predict().mean``, which is the 7-column surrogate layout.
_CBC_SURROGATE_COL = _TARGET_TO_SURROGATE_COL["cbc_max"]
#: surrogate columns the per-cell F_q / |AO| calibrations write (== 2 and 4).
_FQ_SURROGATE_COL = _TARGET_TO_SURROGATE_COL["f_q"]
_AO_SURROGATE_COL = _TARGET_TO_SURROGATE_COL["ao_abs"]
_BACKEND_MANIFEST = "backend.json"
#: Default OOD sigma floors for :meth:`PosValCnnBackend.predict_map_flatness`
#: (2026-07-29 debug-panel).  A single blind out-of-distribution case measured
#: err/sigma of −12.8 (node_peak) and −7.5 (map_cov) against the map head's
#: flatness spread: the across-member spread is an EPISTEMIC statistic of an
#: ensemble whose members agree strongly off-distribution, so it under-states the
#: true error by an order of magnitude exactly where it matters.  These floors are
#: ~half the observed OOD error, i.e. deliberately still optimistic but no longer
#: absurd.  Overridable per checkpoint — see :meth:`_resolve_flatness_sigma_floor`.
_DEFAULT_FLATNESS_SIGMA_FLOOR: dict[str, float] = {"node_peak": 0.06, "map_cov": 0.02}
#: Per-channel training-population feature envelope sidecar (review sec. 4b), written
#: next to the champion so the serve-time OOD guard uses the frozen train-time range.
_FEATURE_OOD_NAME = "feature_ood.json"


class EncoderChannelMismatch(ValueError):
    """The encoder's cell-channel inventory does not match the checkpoint's net.

    Raised when a member's ``in_channels`` (or the checkpoint's recorded
    ``channels`` list) disagrees with the active encoder — a cond-schema /
    feature-schema change (e.g. v3 -> v4) that a fine-tune cannot bridge because
    the stem's input width is fixed.  A ``ValueError`` subclass so it satisfies
    the existing "reject a mismatched load" contract, but a distinct type so the
    curriculum's retrain guard can catch exactly this case and force a full
    retrain instead of a fine-tune.
    """


@runtime_checkable
class PositionValueModel(Protocol):
    """The interface the campaign consumes (plan sec. 4.5)."""

    def predict(self, patterns: Sequence[Pattern], case: CaseKey,
                cell: float) -> SurrogatePrediction: ...

    def predict_convergence(self, patterns: Sequence[Pattern], case: CaseKey,
                            cell: float) -> np.ndarray: ...

    def position_values(self, pattern: Pattern, case: CaseKey,
                        cell: float) -> None: ...

    def finetune(self, new: Any, replay: Any, epochs: int, seed: int) -> dict: ...

    def save(self, path: str | Path) -> Path: ...


# --------------------------------------------------------------------------- #
# dataset-order -> 7-column surrogate scatter (by target name)
# --------------------------------------------------------------------------- #
def _to_surrogate(cols: np.ndarray,
                  target_names: Sequence[str] = _LEGACY_TARGET_NAMES,
                  *, fill: float = np.nan) -> np.ndarray:
    """Scatter a dataset-order ``[N, len(target_names)]`` matrix into ``[N,7]``.

    Each column is placed at :data:`_TARGET_TO_SURROGATE_COL` for its name; a
    target with no surrogate slot (``discharge_burnup``) is skipped, so its
    surrogate column stays ``fill`` (NaN).  ``target_names`` defaults to the
    5-target cond_v2 order, so a bare ``_to_surrogate(cols5)`` keeps the
    historical contract (burnup columns 5/6 NaN).
    """
    cols = np.asarray(cols, dtype=float)
    n = cols.shape[0]
    out = np.full((n, _N_SURROGATE), fill, dtype=float)
    for k, name in enumerate(target_names):
        col = _TARGET_TO_SURROGATE_COL.get(name)
        if col is not None:
            out[:, col] = cols[:, k]
    return out


@dataclass(frozen=True)
class ExtraPrediction:
    """Ensemble prediction for the non-surrogate targets (plan sec. 12.4/12.5).

    ``mean`` / ``epistemic_std`` / ``calibrated_std`` are ``[N, K]`` in
    ``names`` order (currently just ``discharge_burnup``).  A checkpoint that
    does not predict a given extra target fills its column with NaN — the same
    "no model for this axis" sentinel the surrogate layout uses.
    """

    names: tuple[str, ...]
    mean: np.ndarray
    epistemic_std: np.ndarray
    calibrated_std: np.ndarray

    def column(self, name: str) -> int:
        return self.names.index(name)

    def value(self, name: str) -> np.ndarray:
        """The ``mean`` column for ``name`` (raises if the name is unknown)."""
        return self.mean[:, self.column(name)]


@dataclass(frozen=True)
class QuantileSurrogatePrediction(SurrogatePrediction):
    """A :class:`SurrogatePrediction` carrying ADDITIVE quantile outputs.

    Subclassing (rather than mutating) is what keeps the vendor contract intact:
    ``mean`` / ``epistemic_std`` / ``calibrated_std`` are still the validated
    ``[N, 7]`` arrays in :data:`TARGET_NAMES` order, ``__post_init__`` still runs,
    and ``isinstance(pred, SurrogatePrediction)`` still holds — so
    ``reward.RewardModel`` and every other consumer of the 7-column layout is
    byte-identical whether or not the champion has quantile heads.

    ``quantiles`` is ``[N, K, Q]`` in ``quantile_targets`` x ``quantile_levels``
    order, in the SAME raw units as ``mean`` (de-normalized, and with the cyclen
    physics prior already added back when one is active).  Only a champion
    trained with ``quantile_heads`` returns this type; otherwise ``predict``
    returns a plain :class:`SurrogatePrediction`.
    """

    quantiles: np.ndarray = None                      # type: ignore[assignment]
    quantile_targets: tuple[str, ...] = ()
    quantile_levels: tuple[float, ...] = ()

    def band(self, name: str, lo: float = 0.10, hi: float = 0.90
             ) -> tuple[np.ndarray, np.ndarray]:
        """``(q_lo, q_hi)`` columns for a target name (raises if not fitted)."""
        t = self.quantile_targets.index(name)
        return (self.quantiles[:, t, self.quantile_levels.index(lo)],
                self.quantiles[:, t, self.quantile_levels.index(hi)])


@dataclass(frozen=True)
class IntervalPrediction:
    """Split-conformal prediction interval for a served batch (report-only).

    ``lower`` / ``upper`` / ``halfwidth`` are ``[N, 7]`` in :data:`TARGET_NAMES`
    order and centered on the served :meth:`PosValCnnBackend.predict` mean; a column
    with no fitted conformal target (``max_assembly_burnup``, or a target the
    artifact omits) — or a row whose mean/σ is NaN — is left NaN, so a finite bound
    marks a real interval.  ``from_cell`` flags where a per-cell quantile (vs the
    per-target global fallback) was used.  ``available`` is False when the champion
    ships no ``conformal.json`` (all-NaN intervals).  ``coverage`` is the nominal
    ``1 - alpha``.
    """

    alpha: float
    coverage: float
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    halfwidth: np.ndarray
    from_cell: np.ndarray
    available: bool

    def column(self, name: str) -> int:
        return self.names.index(name)

    def bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` columns for a target name."""
        c = self.column(name)
        return self.lower[:, c], self.upper[:, c]


class PosValCnnBackend:
    """Deep-ensemble PosValNet backend implementing :class:`PositionValueModel`."""

    def __init__(
        self,
        members: Sequence[PosValNet],
        metas: Sequence[dict],
        *,
        fuel: FuelLibrary,
        library_id: str = "ga80",
        calibration: dict | None = None,
        cell_calibration: dict | None = None,
        apply_cell_calibration: bool = True,
        fr_calibration: dict | None = None,
        apply_fr_calibration: bool = True,
        cbc_calibration: dict | None = None,
        apply_cbc_calibration: bool = True,
        fq_calibration: dict | None = None,
        apply_fq_calibration: bool = True,
        ao_calibration: dict | None = None,
        apply_ao_calibration: bool = True,
        flatness_calibration: dict | None = None,
        apply_flatness_calibration: bool = True,
        flatness_sigma_floor: dict | None = None,
        pinbu_physics: dict | None = None,
        apply_pinbu_physics: bool = True,
        conformal: dict | None = None,
        device: str | torch.device = "cpu",
        encoder: FeatureEncoder | None = None,
        feature_ood_envelope: dict | None = None,
    ):
        if not members:
            raise ValueError("PosValCnnBackend needs at least one member")
        self.members = list(members)
        self.metas = list(metas)
        self.fuel = fuel
        self.library_id = library_id
        self.calibration = calibration
        # -- per-cell cyclen affine calibration (plan 4.4, cell_calibrate.py) --
        # A serve-side ``cyclen_cal = a*pred + b`` correcting the champion's
        # uniform per-(feed,e_core-bin) cyclen over-prediction.  ``cell_calibration``
        # is the full artifact; ``_cell_cyclen_cells`` is its {cell_key:{a,b,..}}
        # map, applied to the cyclen column of predict() means when the request's
        # cell resolves to a fitted cell (sigma + other columns untouched).  The
        # flag disables it without dropping the loaded artifact.
        self.cell_calibration = cell_calibration
        self._cell_cyclen_cells = calibration_cells(cell_calibration)
        self._cell_cyclen_global = global_by_library(cell_calibration)
        self.apply_cell_calibration = bool(apply_cell_calibration)
        self.cell_bin_width = float(
            (cell_calibration or {}).get("bin_width", DEFAULT_BIN_WIDTH)
        )
        # -- per-cell F_r affine calibration (plan Task A, cell_calibrate.py) --
        # The exact sibling of the cyclen hook for surrogate column 0 (F_r): the
        # champion under-predicts F_r by a near-uniform per-(feed,e_core-bin) shift
        # (non-conservative vs the F_r <= 1.55 feasibility limit).  ``fr_calibration``
        # is the full ``f_r_calibration.json`` artifact; ``_fr_cells`` is its
        # {cell_key:{a,b,..}} map, applied ONLY to the F_r column of predict() means
        # for a request whose cell was fitted (sigma + every other column untouched).
        # CRITICAL: when the artifact is ABSENT (``_fr_cells == {}``) predict() is
        # byte-identical to today (the guard short-circuits); and every fitted a>0,
        # so the map is monotone and within-cell ranking (the honest gate's Spearman)
        # is unchanged.
        self.fr_calibration = fr_calibration
        self._fr_cells = calibration_cells(fr_calibration)
        self._fr_global = global_by_library(fr_calibration)
        self.apply_fr_calibration = bool(apply_fr_calibration)
        self.fr_bin_width = float(
            (fr_calibration or {}).get("bin_width", DEFAULT_BIN_WIDTH)
        )
        # -- per-cell CBC_max affine calibration (2026-07-29 debug-panel) ------
        # The third instance of the identical hook, for surrogate column 1.  The
        # champion over-predicts cbc_max by a near-uniform per-(feed,e_core-bin)
        # shift (measured global +27 ppm, per-group up to +113 ppm, 36% of the
        # 42.4 ppm MAE), which walks the MASTER-verified debug panel's 20 ppm
        # neutronics tolerance on bias alone.  ``cbc_calibration`` is the full
        # ``cbc_calibration.json``; ``_cbc_cells`` is its {cell_key:{a,b,..}} map,
        # applied ONLY to the CBC_max column of predict() means for a request whose
        # cell was fitted (sigma + every other column untouched).
        # BACKWARD COMPATIBILITY: a checkpoint WITHOUT the artifact yields
        # ``_cbc_cells == {}`` and the guard in predict() short-circuits, so its
        # predictions are byte-identical to before this hook existed.  Every fitted
        # a > 0, so the map is monotone and within-cell ranking is unchanged.
        self.cbc_calibration = cbc_calibration
        self._cbc_cells = calibration_cells(cbc_calibration)
        self._cbc_global = global_by_library(cbc_calibration)
        self.apply_cbc_calibration = bool(apply_cbc_calibration)
        self.cbc_bin_width = float(
            (cbc_calibration or {}).get("bin_width", DEFAULT_BIN_WIDTH)
        )
        # -- per-cell F_q + |AO| affine calibration (2026-07-29 all-targets) ----
        # The 4th and 5th instances of the identical hook, for surrogate columns 2
        # and 4.  F_q is the biggest remaining bias share of any scalar (71% of a
        # 0.250 MAE, UNDER-predicting by 0.178 — the non-conservative direction
        # against the 2.41 limit); |AO| is the smallest (17% of 0.0060) and is
        # corrected for bias hygiene on a licensing-reported axis rather than for
        # MAE.  Absent artifact -> empty maps -> predict() byte-identical.
        self.fq_calibration = fq_calibration
        self._fq_cells = calibration_cells(fq_calibration)
        self._fq_global = global_by_library(fq_calibration)
        self.apply_fq_calibration = bool(apply_fq_calibration)
        self.fq_bin_width = float(
            (fq_calibration or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        self.ao_calibration = ao_calibration
        self._ao_cells = calibration_cells(ao_calibration)
        self._ao_global = global_by_library(ao_calibration)
        self.apply_ao_calibration = bool(apply_ao_calibration)
        self.ao_bin_width = float(
            (ao_calibration or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        # -- per-cell INTERCEPT-ONLY flatness calibration (map head) ------------
        # node_peak / map_cov are consumed as LEVELS by the flatness-first
        # objective and the map head is optimistic about both (measured bias
        # -0.0462 / -0.0272).  ONE artifact holds both axes (one forward fits
        # them), so its maps are keyed by target first.  a == 1 by construction,
        # which is what makes the correction rank-preserving inside a calibration
        # cell — see :meth:`predict_map_flatness`.
        self.flatness_calibration = flatness_calibration
        self._flat_cells = {t: flatness_cells(flatness_calibration, t)
                            for t in FLATNESS_TARGETS}
        self._flat_global = {t: flatness_global_by_library(flatness_calibration, t)
                             for t in FLATNESS_TARGETS}
        self.apply_flatness_calibration = bool(apply_flatness_calibration)
        self.flatness_bin_width = float(
            (flatness_calibration or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        # -- OOD sigma floor for the map-head flatness spread ------------------
        # Resolved once at construction: this argument (which is how ``from_dir``
        # passes a ``backend.json`` override) beats the members' meta, which beats
        # the module defaults — see :meth:`_resolve_flatness_sigma_floor`.
        self.flatness_sigma_floor = self._resolve_flatness_sigma_floor(
            flatness_sigma_floor)
        self.apply_flatness_sigma_floor = True
        # -- serve-side physics pin-burnup estimator (pinbu_physics.py) --------
        # When a ``pinbu_physics.json`` is loaded and enabled, predict() overrides
        # the served max_pin_burnup mean (surrogate col 6) with a per-type-physics
        # magnitude estimate reconstructed from the (strong) served cyclen; the raw
        # head weights + sigma are untouched, and a champion with no artifact is a
        # pure no-op (deployed champion unaffected).  Gate parity mirrors the cyclen
        # calibration: each champion loads its OWN artifact, so a gate comparison is
        # symmetric (raw-vs-raw or physics-vs-physics).
        self.pinbu_physics = pinbu_physics
        self.apply_pinbu_physics = bool(apply_pinbu_physics)
        self._pinbu = (
            PinBuPhysicsEstimator.from_artifact(pinbu_physics, self.fuel)
            if pinbu_physics else None
        )
        # -- split-conformal prediction intervals (conformal.py) ---------------
        # A REPORT-ONLY additive wrapper: ``conformal`` is the full ``conformal.json``
        # artifact (per-target split-conformal quantiles fit on the honest per-cell
        # holdout).  It NEVER touches predict()/predict_extra/predict_convergence —
        # :meth:`predict_interval` is the only reader — so every campaign consumer is
        # byte-identical whether or not a champion ships one.  ``conformal_bin_width``
        # keys a served pattern's (feed, e_core-bin) cell for the interval lookup,
        # identical to the fit-time key.
        self.conformal = conformal
        self._conformal_targets = conformal_targets(conformal)
        self.conformal_bin_width = float(
            (conformal or {}).get("bin_width", CONFORMAL_BIN_WIDTH)
        )
        self.device = torch.device(device)

        # Conditioning schema + target order come from the checkpoint meta so a
        # served pattern re-normalizes exactly as it was trained.  A mixed-schema
        # or mixed-target ensemble is rejected (the reject/flag half of the plan
        # sec. 6.2 v2/v3 compat rule); an absent meta field defaults to the
        # cond_v2 / 5-target legacy so old checkpoints keep loading.
        schemas = {str(m.get("cond_schema", "v2")) for m in self.metas}
        if len(schemas) != 1:
            raise ValueError(
                f"ensemble mixes cond_schema {sorted(schemas)}; retrain uniformly"
            )
        self.cond_schema = schemas.pop()
        target_name_sets = {
            tuple(m.get("target_names", _LEGACY_TARGET_NAMES)) for m in self.metas
        }
        if len(target_name_sets) != 1:
            raise ValueError("ensemble members disagree on target_names")
        self.target_names: tuple[str, ...] = target_name_sets.pop()

        # cond_v6 power-map prior: the ``prior_power`` channel is a function of the
        # two FITTED constants (M^2, extrapolation), so serving must rebuild the
        # encoder with the checkpoint's own values.  Without this the encoder falls
        # back to the module defaults and the served ``prior_power`` channel is not
        # the one the network trained on — silent, and invisible in the 7-column
        # output.  Handled exactly like cond_schema: one value per ensemble, a
        # mismatch is a hard error, and a checkpoint without the field (every
        # v2..v5 model) yields ``None`` and the untouched legacy path.
        self.power_prior = None
        prior_sigs = {
            (round(float(pp["m2_cm2"]), 6), round(float(pp["extrap"]), 6))
            for pp in (m.get("power_prior") or {} for m in self.metas)
            if pp.get("schema") and pp.get("m2_cm2") is not None
        }
        if len(prior_sigs) > 1:
            raise ValueError(
                f"ensemble mixes power_prior constants {sorted(prior_sigs)}; "
                "retrain uniformly")
        if prior_sigs:
            from .power_prior import PowerPrior
            m2, extrap = prior_sigs.pop()
            self.power_prior = PowerPrior(m2_cm2=m2, extrap=extrap)

        if encoder is not None:
            if getattr(encoder, "cond_schema", self.cond_schema) != self.cond_schema:
                raise ValueError(
                    f"supplied encoder cond_schema {encoder.cond_schema!r} != "
                    f"checkpoint cond_schema {self.cond_schema!r}"
                )
            self.encoder = encoder
        else:
            self.encoder = FeatureEncoder(cond_schema=self.cond_schema,
                                          power_prior=self.power_prior)

        # -- load-time cell-channel parity (the F2 fix) -------------------------
        # The encoder's channel inventory must match BOTH the network stem width
        # (net.in_channels) and, when the checkpoint recorded it, the frozen
        # ``channels`` list — otherwise a served pattern is featurized to a width
        # the weights never saw (silent garbage).  A distinct exception type lets
        # the retrain guard force a full retrain on a schema change.
        self._assert_channel_parity()

        tmean, tstd = norm_from_meta(self.metas[0])
        self.tmean = np.asarray(tmean, dtype=np.float64)
        self.tstd = np.asarray(tstd, dtype=np.float64)
        if not (len(self.tmean) == len(self.tstd) == len(self.target_names)):
            raise ValueError(
                "target_zscore length does not match target_names "
                f"({len(self.tmean)} vs {len(self.target_names)})"
            )
        for m in self.members:
            m.to(self.device).eval()

        # -- remote-screening state (plan 4.7) — off unless a screener is attached.
        # ``_remote`` is a callable ``(backend, patterns, cases) -> (mu_z, log_sigma,
        # conv_logit)`` that offloads a batch to the GPU box; ``_screen_cache`` is a
        # per-session memo keyed by ``(canonical pattern, pair, feed)`` so the 3x
        # predict/convergence/extra passes over one pool collapse to a single
        # compute (and a prewarmed batch is served locally to the per-cell scorer).
        self._remote = None
        self._remote_min = 0
        self._remote_log = None
        self._screen_cache: dict[tuple[str, str, int],
                                 tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

        # -- cyclen physics-prior residual head (physics_prior.py) --------------
        # A v5 checkpoint trained with ``cyclen_physics_prior`` predicts the
        # RESIDUAL ``cyclen - prior``; the prior's two fitted scalars live in the
        # checkpoint meta, so serving reconstructs it from the pattern and adds it
        # back — ``predict()`` therefore always returns ABSOLUTE cyclen, and every
        # downstream consumer (reward, acquisition, gates, cell calibration) is
        # unaware the head is a residual head.  Every existing checkpoint has no
        # such meta block, so ``_cyclen_prior is None`` and the add-back is skipped
        # entirely (byte-identical predict).
        self._cyclen_prior = None
        prior_meta = self.metas[0].get("cyclen_physics_prior")
        if isinstance(prior_meta, dict) and prior_meta.get("enabled"):
            priors = {json.dumps(m.get("cyclen_physics_prior"), sort_keys=True)
                      for m in self.metas}
            if len(priors) != 1:
                raise ValueError(
                    "ensemble members disagree on cyclen_physics_prior; "
                    "retrain uniformly")
            from .physics_prior import CyclenPhysicsPrior
            self._cyclen_prior = CyclenPhysicsPrior.from_dict(prior_meta)
        self._cyclen_target_idx = (
            self.target_names.index("cyclen")
            if "cyclen" in self.target_names else None
        )

        # -- pinball quantile heads (net.PosValNet quantile_head) ---------------
        # Additive: a champion without them keeps returning a plain
        # SurrogatePrediction from predict().
        q_meta = self.metas[0].get("quantile_heads") or {}
        self.quantile_targets: tuple[str, ...] = (
            tuple(q_meta.get("targets", ())) if q_meta.get("enabled") else ()
        )
        self.quantile_levels: tuple[float, ...] = (
            tuple(float(x) for x in q_meta.get("levels", ()))
            if q_meta.get("enabled") else ()
        )

        # -- axial profile head (net.PosValNet axial_head, decision D10) --------
        # Additive, exactly like the quantile heads above: a champion without one
        # has no ``axial_head`` meta key, ``has_axial()`` is False and every
        # existing predict path is untouched.  The basis is a checkpoint artifact
        # (fit on the member's TRAIN fold), so it is read from meta rather than
        # re-fit at serve time.
        a_meta = self.metas[0].get("axial_head") or {}
        self._axial_basis = None
        if a_meta.get("enabled") and a_meta.get("basis"):
            from ..data.axial import AxialBasis
            self._axial_basis = AxialBasis.from_dict(a_meta["basis"])

        # -- serve-time feature/geometry OOD guard (review sec. 4b) -------------
        # A per-channel training-population z-envelope.  When the checkpoint shipped
        # a ``feature_ood.json`` sidecar it is the frozen train-time range; otherwise
        # it is computed lazily from the served fuel population on first use.  Purely
        # advisory — the guard NEVER changes a prediction (see feature_ood_types).
        self._feature_ood_envelope = feature_ood_envelope

    def _resolve_flatness_sigma_floor(self, override: dict | None) -> dict[str, float]:
        """Per-model ``{"node_peak": s, "map_cov": s}`` OOD floor for the map head.

        Resolution order, most-specific first:

        1. ``override`` — the constructor / :meth:`set_flatness_sigma_floor` argument
           (also how ``from_dir`` passes a ``backend.json`` ``flatness_sigma_floor``);
        2. member meta ``flatness_residual_sd`` — the residual SD of the flatness
           scalars measured on that member's OWN validation fold.  This is the
           RIGHT number and the reason the key is read first: it is the model's
           measured error, not a guess.  No checkpoint persists it today; training
           can start writing it without any change here.  Taken as the MAX over
           members (a floor must cover the worst member, not the average);
        3. member meta ``flatness_sigma_floor`` — an explicitly stamped floor;
        4. :data:`_DEFAULT_FLATNESS_SIGMA_FLOOR` (node_peak 0.06 / map_cov 0.02).

        A missing or non-finite entry falls through per channel, so a source that
        carries only ``node_peak`` still gets the default ``map_cov``.

        FUTURE (A/B arm, not implemented): a distance-aware calibration — inflate
        sigma by how far the served pattern sits outside the training feature
        envelope (:meth:`feature_ood_types` already measures that distance) instead
        of a flat per-model constant.  A constant floor cannot distinguish a
        marginally-novel core from a wildly novel one; it is deliberately the
        minimal honest fix for a spread that was off by 12x, not the final answer.
        """
        def _clean(block: Any, name: str) -> float | None:
            try:
                val = float(block[name])
            except (KeyError, TypeError, ValueError, IndexError):
                return None
            return val if math.isfinite(val) and val >= 0.0 else None

        out = dict(_DEFAULT_FLATNESS_SIGMA_FLOOR)
        # meta keys, lowest priority first; within one key the MAX across members
        # wins (a floor must cover the worst member, not the average).
        for meta_key in ("flatness_sigma_floor", "flatness_residual_sd"):
            for name in list(out):
                vals = [v for v in (_clean(m.get(meta_key) or {}, name)
                                    for m in self.metas) if v is not None]
                if vals:
                    out[name] = max(vals)
        for name in list(out):
            val = _clean(override or {}, name)
            if val is not None:
                out[name] = val
        return out

    def set_flatness_sigma_floor(self, floor: dict | None) -> None:
        """Re-resolve the map-head OOD sigma floor (``None`` -> back to defaults).

        Runtime hook for the A/B harness and tests; mirrors ``set_*_calibration``.
        """
        self.flatness_sigma_floor = self._resolve_flatness_sigma_floor(floor)

    def _assert_channel_parity(self) -> None:
        """Reject an encoder whose channel inventory disagrees with the members.

        Guards two edges: (1) every member's ``net.in_channels`` must equal the
        encoder's channel count, and (2) when a member's meta froze a ``channels``
        list, it must equal the encoder's channels exactly (order-stable).
        """
        enc_channels = list(getattr(self.encoder, "channels", ()))
        n_enc = len(enc_channels)
        for meta, model in zip(self.metas, self.members):
            in_ch = int(getattr(model.config, "in_channels", n_enc))
            if in_ch != n_enc:
                raise EncoderChannelMismatch(
                    f"member (seed {meta.get('seed')}) net in_channels {in_ch} != "
                    f"encoder channel count {n_enc} (cond_schema {self.cond_schema!r}); "
                    f"a feature-schema change requires a full retrain, not a load"
                )
            meta_channels = meta.get("channels")
            if meta_channels is not None and enc_channels and \
                    list(meta_channels) != enc_channels:
                raise EncoderChannelMismatch(
                    f"member (seed {meta.get('seed')}) meta 'channels' "
                    f"(len {len(meta_channels)}) != encoder channels (len {n_enc}) "
                    f"for cond_schema {self.cond_schema!r}; retrain uniformly"
                )

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_dir(cls, ensemble_dir: str | Path, *,
                 store_dir: str | Path = "data/store",
                 library_id: str = "ga80",
                 device: str | torch.device = "cpu") -> "PosValCnnBackend":
        d = Path(ensemble_dir)
        member_dirs = sorted(d.glob("member_*"))
        if not member_dirs:
            raise FileNotFoundError(f"no member_* checkpoints under {d}")
        members, metas = [], []
        for md in member_dirs:
            model, meta = load_member(md, device)
            members.append(model)
            metas.append(meta)
        calib = None
        cpath = d / CALIB_NAME
        if cpath.is_file():
            calib = load_calibration(cpath)
        cell_calib = None
        cell_cpath = d / CELL_CALIB_NAME
        if cell_cpath.is_file():
            cell_calib = load_cell_calibration(cell_cpath)
        fr_calib = None
        fr_cpath = d / FR_CALIB_NAME
        if fr_cpath.is_file():
            fr_calib = load_fr_calibration(fr_cpath)
        # CBC_max per-cell calibration: ABSENT on every pre-2026-07-29 checkpoint,
        # which is exactly the backward-compat contract — ``None`` here means the
        # backend's cbc hook is a no-op and predict() is unchanged.
        cbc_calib = None
        cbc_cpath = d / CBC_CALIB_NAME
        if cbc_cpath.is_file():
            cbc_calib = load_cbc_calibration(cbc_cpath)
        # F_q / |AO| / flatness: absent on every pre-2026-07-29 checkpoint, which
        # is the backward-compat contract (None -> the hook is a no-op).
        fq_calib = None
        if (d / FQ_CALIB_NAME).is_file():
            fq_calib = load_fq_calibration(d / FQ_CALIB_NAME)
        ao_calib = None
        if (d / AO_CALIB_NAME).is_file():
            ao_calib = load_ao_calibration(d / AO_CALIB_NAME)
        flat_calib = None
        if (d / FLAT_CALIB_NAME).is_file():
            flat_calib = load_flatness_calibration(d / FLAT_CALIB_NAME)
        pinbu = None
        pinbu_path = d / PINBU_PHYSICS_NAME
        if pinbu_path.is_file():
            pinbu = load_pinbu_physics(pinbu_path)
        conformal = None
        conformal_path = d / CONFORMAL_NAME
        if conformal_path.is_file():
            conformal = load_conformal(conformal_path)
        feature_ood = None
        ood_path = d / _FEATURE_OOD_NAME
        if ood_path.is_file():
            try:
                feature_ood = json.loads(ood_path.read_text()).get("envelope")
            except (OSError, ValueError):
                feature_ood = None
        sigma_floor = None
        manifest = d / _BACKEND_MANIFEST
        if manifest.is_file():
            mani = json.loads(manifest.read_text())
            library_id = mani.get("library_id", library_id)
            # OPTIONAL per-checkpoint override of the map-head OOD sigma floor
            # (2026-07-29 debug-panel).  Absent on every existing backend.json, in
            # which case the member meta / module defaults decide.
            floor = mani.get("flatness_sigma_floor")
            if isinstance(floor, dict):
                sigma_floor = floor
            # cross-check the manifest's frozen channel/global lists against the
            # members' metas (both written by the same training encoder); a
            # disagreement means the checkpoint dir was assembled inconsistently.
            _cross_check_manifest(mani, metas)
        fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
        return cls(members, metas, fuel=fuel, library_id=library_id,
                   calibration=calib, cell_calibration=cell_calib,
                   fr_calibration=fr_calib, cbc_calibration=cbc_calib,
                   fq_calibration=fq_calib, ao_calibration=ao_calib,
                   flatness_calibration=flat_calib,
                   flatness_sigma_floor=sigma_floor,
                   pinbu_physics=pinbu, conformal=conformal, device=device,
                   feature_ood_envelope=feature_ood)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "PosValCnnBackend":
        return cls.from_dir(path, **kwargs)

    # -- serve-time library resolution (train/serve parity, plan 6.2) ------- #
    def _rosters(self) -> dict[str, set]:
        """``{library_id: {type_id, ...}}`` for every library in the fuel table.

        Cached lazily; used to detect when a served pattern's fresh fuel types are
        absent from the configured serving library (a train/serve parity break).
        """
        cache = getattr(self, "_roster_cache", None)
        if cache is None:
            cache = {L: set(self.fuel.types(L)) for L in self.fuel.libraries()}
            self._roster_cache = cache
        return cache

    def _effective_library(self, pattern: Pattern) -> str:
        """The library whose roster actually carries this pattern's fresh types.

        The configured ``self.library_id`` is used whenever it EXACTLY contains
        every fresh batch type — the production fast path (a ga80 campaign serving
        ga80 patterns always hits this, byte-identical to the pre-fix behaviour).

        When it does NOT (e.g. a Dataset-A holdout row whose types live in the
        ``5.8_5.1`` library scored through a ga80/260624 backend), the fresh types
        are matched against every library's roster and, when they all live in
        exactly ONE other library, that library is used so the featurizer sees the
        SAME fuel physics training did.  This is the fix for the v4 tail collapse:
        without it a v4 model's active lattice channels (kinf/u_mass/branch) silently
        collapse to the absent-sentinel 0 for an unresolved type, so a high-reactivity
        assembly is served as low-reactivity and cyclen under-predicts by ~40-50 EFPD.

        AMBIGUOUS patterns (fresh types present in >1 library with different physics
        — e.g. the ``5.8_5.1`` / ``CPHA`` overlap) or UNRESOLVED ones keep the
        configured library: the pattern alone cannot disambiguate the physics, so
        provenance-carrying callers must featurize from the store row's own
        ``library_id`` (see :meth:`predict_rows_raw`), and :meth:`unresolved_fresh_types`
        surfaces the silent-degradation risk for the campaign's provenance-less path.
        """
        batches = list(pattern.batch_feed().keys())
        if not batches:
            return self.library_id
        rosters = self._rosters()
        configured = rosters.get(self.library_id, set())
        if all(b in configured for b in batches):
            return self.library_id
        cands = [L for L in sorted(rosters)
                 if L != self.library_id and all(b in rosters[L] for b in batches)]
        if len(cands) == 1:
            return cands[0]
        return self.library_id

    def unresolved_fresh_types(self, pattern: Pattern) -> list[str]:
        """Fresh batch types this pattern carries that resolve in NO library at the
        served (effective) library — the serve-parity break signal.

        For a v4 checkpoint these types featurize with a zeroed lattice block
        (``origin_lattice_present=0``), which the model never saw at train time for
        a high-reactivity assembly, so its cyclen prediction is untrustworthy.  The
        campaign / eval harness can call this to WARN instead of silently emitting a
        catastrophic under-prediction.
        """
        lib = self._effective_library(pattern)
        return [b for b in pattern.batch_feed()
                if resolve_type_id(self.fuel, lib, b) is None]

    # -- serve-time feature / geometry OOD guard (review sec. 4b) ----------- #
    def feature_ood_envelope(self) -> dict:
        """The per-channel training-population feature z-envelope.

        Returns the checkpoint's frozen ``feature_ood.json`` range when one was
        loaded, otherwise a lazily-computed (and cached) envelope over the FULL
        training fuel population (every library in the table — the population the
        ensemble was featurized against).  Freezing it at train time (the sidecar)
        is what makes the guard honest: a served geometry variant added to the fuel
        table later must NOT be allowed to stretch its own envelope.
        """
        env = self._feature_ood_envelope
        if env is None:
            env = population_envelope_from_library(self.fuel, None)
            self._feature_ood_envelope = env
        return env

    def _fresh_vecs(self, pattern: Pattern) -> dict[str, Any]:
        """``{type_id: FuelVec|None}`` for a pattern's fresh batch types (effective
        library), the input surface the feature-OOD guard reads."""
        lib = self._effective_library(pattern)
        out: dict[str, Any] = {}
        for b in pattern.batch_feed():
            tid = resolve_type_id(self.fuel, lib, b)
            if tid is None:
                out[b] = None
                continue
            try:
                out[b] = self.fuel.get(tid, lib)
            except KeyError:
                out[b] = None
        return out

    def feature_ood_types(self, pattern: Pattern, *, margin: float = _OOD_MARGIN
                          ) -> dict[str, list[tuple[str, float]]]:
        """Fresh types whose harvested features fall OUTSIDE the training envelope.

        Mirrors :meth:`unresolved_fresh_types`: a *warning surface* (returns the
        offending ``{type_id: [(channel, z), ...]}``), NEVER a hard fail and NEVER a
        change to any prediction.  A pin-pitch / pin-radius geometry variant trips
        it via ``u_mass`` (radius canary), ``xs_s12`` (spectral pitch canary), and
        the direct geometry channels; a pattern of pure training types is clean by
        construction (its features sit inside the population range).
        """
        return feature_ood_vecs(self._fresh_vecs(pattern),
                                self.feature_ood_envelope(), margin=margin)

    def feature_ood_report(self, patterns: Sequence[Pattern],
                           cases: Sequence[CaseKey] | CaseKey | None = None,
                           *, margin: float = _OOD_MARGIN) -> dict[str, list[tuple[str, float]]]:
        """Union of :meth:`feature_ood_types` over a batch of patterns.

        The single call the campaign / report uses to surface a warning block
        (``cases`` is accepted for signature symmetry with ``predict`` and unused —
        the feature-OOD signal depends only on the pattern's fresh types).
        """
        agg: dict[str, list[tuple[str, float]]] = {}
        for pat in patterns:
            for tid, offenders in self.feature_ood_types(pat, margin=margin).items():
                if tid not in agg:
                    agg[tid] = offenders
        return agg

    def feature_ood_warning(self, patterns: Sequence[Pattern],
                            *, margin: float = _OOD_MARGIN) -> str:
        """A one-line human warning for a batch (``""`` when nothing is OOD)."""
        return format_ood_warning(self.feature_ood_report(patterns, margin=margin))

    # -- featurization ------------------------------------------------------ #
    def _record_inputs(self, pattern: Pattern, case: CaseKey) -> RecordInputs:
        """Reconstruct the full leakage-safe :class:`RecordInputs` for a served
        ``(pattern, case)`` — everything a training store row carried, recomputed
        from the pattern + campaign library so the encoding matches training.

        A :class:`CaseKey` carries no provenance, so ``dataset`` / ``sym_class``
        are derived from the *effective* ``library_id`` (:meth:`_effective_library`,
        which reroutes to the library that actually carries the pattern's fresh types
        when the configured one does not — ga80 -> Dataset B / free69, else Dataset A
        / rot61) rather than left at the RecordInputs defaults (``"A"`` / ``"rot61"``),
        which biased ga80 inference onto the Dataset-A regime.  ``e_core`` / ``e_split``
        use the same feed-average recipe as extraction so the served conditioning
        equals the stored value.
        """
        lib = self._effective_library(pattern)
        dataset, sym_class = library_provenance(lib)
        e_core, e_split = core_enrichment_split(
            self.fuel, lib, pattern.batch_feed()
        )
        return RecordInputs(
            pattern=pattern.canonical(),
            feed=int(case.feed),
            case_pair=case.pair,
            library_id=lib,
            e_core=e_core,
            e_split=e_split,
            sym_class=sym_class,
            dataset=dataset,
        )

    # -- provenance-correct row scoring (gate tail guard, plan 12.3) -------- #
    @torch.no_grad()
    def predict_rows_raw(self, rows: Any) -> np.ndarray:
        """7-column raw ensemble means for store rows, featurized from each row's OWN
        ``library_id`` provenance (``RecordInputs.coerce``) — the train/serve-parity
        scoring path.

        Unlike :meth:`predict` (which must reconstruct provenance from a
        provenance-less :class:`CaseKey` and cannot disambiguate a type shared by
        several libraries), this honours the store row's recorded library, so a
        Dataset-A ``5.8_5.1`` tail row is scored against the exact fuel physics it
        trained on.  The per-cell cyclen calibration is deliberately NOT applied
        (base-model skill), so the honest no-regression tail gate compares raw
        ensemble outputs symmetrically across champions.
        """
        frame = _as_frame(rows)
        if not len(frame):
            return np.zeros((0, _N_SURROGATE))
        cells, gvecs = [], []
        for _, row in frame.iterrows():
            c, g = self.encoder.encode(RecordInputs.coerce(row), self.fuel)
            cells.append(c)
            gvecs.append(g)
        cells_t = torch.from_numpy(np.ascontiguousarray(np.stack(cells))).to(self.device)
        g_t = torch.from_numpy(np.ascontiguousarray(np.stack(gvecs))).to(self.device)
        mus = []
        for m in self.members:
            mus.append(m(cells_t, g_t)["mu"].float().cpu().numpy())
        members_raw = np.stack(mus) * self.tstd[None, None, :] + self.tmean[None, None, :]
        mean_t = members_raw.mean(axis=0)
        # Residual head -> add the physics prior back here too, so the tail /
        # no-regression gates compare ABSOLUTE cyclen across arms.
        prior_vals = self._cyclen_prior_rows(frame)
        if prior_vals is not None:
            mean_t = mean_t.copy()
            mean_t[:, self._cyclen_target_idx] += prior_vals
        return _to_surrogate(mean_t, self.target_names)

    # -- per-cell cyclen calibration keying --------------------------------- #
    def serve_library(self, pattern: Pattern) -> str:
        """The library this pattern is actually FEATURIZED against at serve time.

        A public alias for :meth:`_effective_library`, exposed so the calibration
        fit can admit exactly the rows whose serve-time resolution round-trips
        (:func:`cell_calibrate.serve_parity_mask`).
        """
        return self._effective_library(pattern)

    def cyclen_e_core(self, pattern: Pattern) -> tuple[float | None, float | None]:
        """Serve-recipe ``(e_core, e_split)`` for a pattern (mirrors _record_inputs).

        Exposed so the calibration FIT keys a row exactly as :meth:`predict` will
        at serve time (identical ``core_enrichment_split`` recipe -> identical bin).

        Resolves against the EFFECTIVE library (2026-07-29 debug-panel), which is
        what :meth:`_record_inputs` already featurizes against — previously this
        used the CONFIGURED ``self.library_id``, so the two disagreed for any
        pattern the campaign reroutes.  The consequence was silent and total: a
        paramA pattern served through a ga80-configured backend resolved e_core to
        ``None`` (its fresh types are absent from the ga80 roster), so every paramA
        request keyed into the ``ebin=None`` cell, which no fit ever populates.  All
        1,361 paramA rows of the curriculum-val slice therefore received ZERO
        calibration on all three targets.

        Byte-identical for a ga80 pattern under a ga80 backend (``_effective_library``
        short-circuits to the configured library), and identical for any pattern the
        rerouting cannot disambiguate — so an existing artifact keeps resolving
        exactly as before, and the change only reaches patterns that were resolving
        to a meaningless bin.
        """
        return core_enrichment_split(
            self.fuel, self._effective_library(pattern), pattern.batch_feed())

    def _calib_cell_keys(self, patterns: Sequence[Pattern],
                         cases: Sequence[CaseKey], bin_width: float) -> list[str]:
        """Serve-recipe ``(feed, e_core-bin)`` keys at one artifact's bin width.

        The ONE implementation of calibration keying — every per-target wrapper
        below delegates here, so no two calibrations can ever key a pattern
        differently (which is the failure mode the 2026-07-29 forensic found
        between the fit and serve sides).
        """
        keys: list[str] = []
        for pat, case in zip(patterns, cases):
            e_core, _ = self.cyclen_e_core(pat)
            keys.append(cyclen_cell_key(int(case.feed), e_core, bin_width))
        return keys

    def _cyclen_cell_keys(self, patterns: Sequence[Pattern],
                          cases: Sequence[CaseKey]) -> list[str]:
        return self._calib_cell_keys(patterns, cases, self.cell_bin_width)

    def fitted_cyclen_cells(self) -> set[str]:
        """Cell keys that carry a fitted cyclen calibration (empty when disabled).

        The campaign's Stage-2 running corrector reads this to avoid double-
        correcting a cell Stage-1 already covers.
        """
        if not self.apply_cell_calibration:
            return set()
        return set(self._cell_cyclen_cells)

    def set_cell_calibration(self, artifact: dict | None,
                             *, enabled: bool = True) -> None:
        """Install (or clear with ``None``) the per-cell cyclen calibration.

        Recomputes the ``{cell_key: {a, b, ...}}`` map + bin width from the
        artifact and toggles application — the runtime hook the fit/refit path and
        tests use without reconstructing the backend.
        """
        self.cell_calibration = artifact
        self._cell_cyclen_cells = calibration_cells(artifact)
        self._cell_cyclen_global = global_by_library(artifact)
        self.cell_bin_width = float((artifact or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        self.apply_cell_calibration = bool(enabled)

    # -- per-cell F_r calibration keying (mirror of the cyclen hooks) ------- #
    def _fr_cell_keys(self, patterns: Sequence[Pattern],
                      cases: Sequence[CaseKey]) -> list[str]:
        return self._calib_cell_keys(patterns, cases, self.fr_bin_width)

    def fitted_fr_cells(self) -> set[str]:
        """Cell keys that carry a fitted F_r calibration (empty when disabled)."""
        if not self.apply_fr_calibration:
            return set()
        return set(self._fr_cells)

    def set_fr_calibration(self, artifact: dict | None,
                           *, enabled: bool = True) -> None:
        """Install (or clear with ``None``) the per-cell F_r calibration.

        Recomputes the ``{cell_key: {a, b, ...}}`` map + bin width from the
        artifact and toggles application — the runtime hook the fit/refit path and
        tests use without reconstructing the backend.  Applies to
        ``predict().mean[:, 0]`` only (never sigma, never another column).
        """
        self.fr_calibration = artifact
        self._fr_cells = calibration_cells(artifact)
        self._fr_global = global_by_library(artifact)
        self.fr_bin_width = float((artifact or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        self.apply_fr_calibration = bool(enabled)

    # -- per-cell CBC_max calibration keying (mirror of the cyclen hooks) ---- #
    def _cbc_cell_keys(self, patterns: Sequence[Pattern],
                       cases: Sequence[CaseKey]) -> list[str]:
        return self._calib_cell_keys(patterns, cases, self.cbc_bin_width)

    def _fq_cell_keys(self, patterns: Sequence[Pattern],
                      cases: Sequence[CaseKey]) -> list[str]:
        return self._calib_cell_keys(patterns, cases, self.fq_bin_width)

    def _ao_cell_keys(self, patterns: Sequence[Pattern],
                      cases: Sequence[CaseKey]) -> list[str]:
        return self._calib_cell_keys(patterns, cases, self.ao_bin_width)

    def _flatness_cell_keys(self, patterns: Sequence[Pattern],
                            cases: Sequence[CaseKey]) -> list[str]:
        return self._calib_cell_keys(patterns, cases, self.flatness_bin_width)

    def fitted_fq_cells(self) -> set[str]:
        """Cell keys that carry a fitted F_q calibration (empty when disabled)."""
        return set(self._fq_cells) if self.apply_fq_calibration else set()

    def fitted_ao_cells(self) -> set[str]:
        """Cell keys that carry a fitted |AO| calibration (empty when disabled)."""
        return set(self._ao_cells) if self.apply_ao_calibration else set()

    def fitted_flatness_cells(self, target: str) -> set[str]:
        """Cell keys carrying a fitted flatness shift for one map axis."""
        if not self.apply_flatness_calibration:
            return set()
        return set(self._flat_cells.get(target, {}))

    def set_fq_calibration(self, artifact: dict | None, *, enabled: bool = True) -> None:
        """Install (or clear with ``None``) the per-cell F_q calibration.

        Applies to ``predict().mean[:, 2]`` only (never sigma, never another
        column) — the runtime hook the fit/refit path and tests use.
        """
        self.fq_calibration = artifact
        self._fq_cells = calibration_cells(artifact)
        self._fq_global = global_by_library(artifact)
        self.fq_bin_width = float((artifact or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        self.apply_fq_calibration = bool(enabled)

    def set_ao_calibration(self, artifact: dict | None, *, enabled: bool = True) -> None:
        """Install (or clear with ``None``) the per-cell |AO| calibration.

        Applies to ``predict().mean[:, 4]`` only.
        """
        self.ao_calibration = artifact
        self._ao_cells = calibration_cells(artifact)
        self._ao_global = global_by_library(artifact)
        self.ao_bin_width = float((artifact or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        self.apply_ao_calibration = bool(enabled)

    def set_flatness_calibration(self, artifact: dict | None,
                                 *, enabled: bool = True) -> None:
        """Install (or clear with ``None``) the INTERCEPT-only flatness calibration.

        Affects :meth:`predict_map_flatness` MEANS only — never the sigmas, never
        the 7-column :meth:`predict` output.
        """
        self.flatness_calibration = artifact
        self._flat_cells = {t: flatness_cells(artifact, t) for t in FLATNESS_TARGETS}
        self._flat_global = {t: flatness_global_by_library(artifact, t)
                             for t in FLATNESS_TARGETS}
        self.flatness_bin_width = float(
            (artifact or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        self.apply_flatness_calibration = bool(enabled)

    def fitted_cbc_cells(self) -> set[str]:
        """Cell keys that carry a fitted CBC_max calibration (empty when disabled)."""
        if not self.apply_cbc_calibration:
            return set()
        return set(self._cbc_cells)

    def set_cbc_calibration(self, artifact: dict | None,
                            *, enabled: bool = True) -> None:
        """Install (or clear with ``None``) the per-cell CBC_max calibration.

        Recomputes the ``{cell_key: {a, b, ...}}`` map + bin width from the
        artifact and toggles application — the runtime hook the fit/refit path and
        tests use without reconstructing the backend.  Applies to
        ``predict().mean[:, 1]`` only (never sigma, never another column).
        """
        self.cbc_calibration = artifact
        self._cbc_cells = calibration_cells(artifact)
        self._cbc_global = global_by_library(artifact)
        self.cbc_bin_width = float((artifact or {}).get("bin_width", DEFAULT_BIN_WIDTH))
        self.apply_cbc_calibration = bool(enabled)

    def set_pinbu_physics(self, artifact: dict | None, *, enabled: bool = True) -> None:
        """Install (or clear with ``None``) the physics pin-burnup estimator.

        Rebuilds the :class:`PinBuPhysicsEstimator` from the artifact + fuel table
        and toggles application — the runtime hook the fit path and tests use
        without reconstructing the backend.
        """
        self.pinbu_physics = artifact
        self._pinbu = (
            PinBuPhysicsEstimator.from_artifact(artifact, self.fuel)
            if artifact else None
        )
        self.apply_pinbu_physics = bool(enabled)

    # -- split-conformal intervals (report-only, conformal.py) -------------- #
    def set_conformal(self, artifact: dict | None) -> None:
        """Install (or clear with ``None``) the split-conformal interval artifact.

        Rebuilds the ``{target: entry}`` map + bin width — the runtime hook the fit
        path and tests use without reconstructing the backend.  Report-only: this
        never affects :meth:`predict`.
        """
        self.conformal = artifact
        self._conformal_targets = conformal_targets(artifact)
        self.conformal_bin_width = float(
            (artifact or {}).get("bin_width", CONFORMAL_BIN_WIDTH)
        )

    def has_conformal(self) -> bool:
        """True when a conformal artifact with at least one fitted target is loaded."""
        return bool(self._conformal_targets)

    def _conformal_cell_keys(self, patterns: Sequence[Pattern],
                             cases: Sequence[CaseKey]) -> list[str]:
        """Serve-recipe ``(feed, e_core-bin)`` keys at the CONFORMAL bin width.

        Identical recipe to :meth:`_cyclen_cell_keys` but with
        ``conformal_bin_width`` (0.25) — so a served pattern lands in the same cell
        the conformal fit keyed it into.
        """
        keys: list[str] = []
        for pat, case in zip(patterns, cases):
            e_core, _ = self.cyclen_e_core(pat)
            keys.append(cyclen_cell_key(int(case.feed), e_core, self.conformal_bin_width))
        return keys

    def _pinbu_column(self, patterns: Sequence[Pattern], cases: Sequence[CaseKey],
                      cyclen: np.ndarray, raw_pin: np.ndarray) -> np.ndarray:
        """Physics pin-burnup override for the served max_pin_burnup column.

        Reconstructs each candidate's pin burnup from the (already cell-calibrated)
        served ``cyclen`` and its fresh-type peaking-ratio curve; a non-finite
        estimate (e.g. an unresolved fuel type) falls back to the raw head value,
        so the column never degrades below the deployed behaviour.
        """
        out = np.array(raw_pin, dtype=float, copy=True)
        for i, (pat, case) in enumerate(zip(patterns, cases)):
            est = self._pinbu.estimate(pat.batch_feed(), int(case.feed), float(cyclen[i]))
            if math.isfinite(est):
                out[i] = est
        return out

    # -- cyclen physics prior (serve-side reconstruction) ------------------- #
    def _cyclen_prior_values(self, patterns: Sequence[Pattern],
                             cases: Sequence[CaseKey]) -> np.ndarray | None:
        """Per-pattern cyclen physics prior [EFPD], or ``None`` when inactive.

        Reconstructed from the SAME leakage-safe inputs the featurizer builds
        (:meth:`_record_inputs`, which resolves the effective library), so the
        served prior equals the one training subtracted for an identical core.
        """
        if self._cyclen_prior is None or self._cyclen_target_idx is None:
            return None
        rows = [self._record_inputs(p, c) for p, c in zip(patterns, cases)]
        return np.asarray(self._cyclen_prior.for_rows(rows, self.fuel), dtype=float)

    def _cyclen_prior_rows(self, frame: Any) -> np.ndarray | None:
        """Prior for store rows featurized from their OWN ``library_id``."""
        if self._cyclen_prior is None or self._cyclen_target_idx is None:
            return None
        rows = [RecordInputs.coerce(r) for _, r in frame.iterrows()]
        return np.asarray(self._cyclen_prior.for_rows(rows, self.fuel), dtype=float)

    # -- quantile heads ----------------------------------------------------- #
    def has_quantiles(self) -> bool:
        """True when this champion's members carry fitted pinball quantile heads."""
        return bool(self.quantile_targets and self.quantile_levels)

    @torch.no_grad()
    def _quantile_raw(self, patterns: Sequence[Pattern], cases: Sequence[CaseKey],
                      prior_vals: np.ndarray | None) -> np.ndarray:
        """Ensemble-mean quantiles in RAW units, ``[N, K, Q]``.

        Runs its own local forward rather than widening the ``_raw_forward``
        tuple, so the remote-screening wire protocol (and its fallback logic) is
        untouched.  Each quantile is de-normalized with its target's own z-score
        constants and, under residual learning, gets the same prior added back as
        the mean — so the band brackets ABSOLUTE cyclen.
        """
        cells_t, g_t = self._encode_batch(patterns, cases)
        acc = []
        for m in self.members:
            out = m(cells_t, g_t)
            acc.append(out["quantiles"].float().cpu().numpy())
        q_z = np.stack(acc).mean(axis=0)                       # [N, K, Q]
        idx = [self.target_names.index(n) for n in self.quantile_targets]
        raw = q_z * self.tstd[idx][None, :, None] + self.tmean[idx][None, :, None]
        if prior_vals is not None and "cyclen" in self.quantile_targets:
            k = self.quantile_targets.index("cyclen")
            raw[:, k, :] += prior_vals[:, None]
        return raw

    # -- axial profile head -------------------------------------------------- #
    def has_axial(self) -> bool:
        """True when this champion's members carry a fitted axial profile head."""
        return self._axial_basis is not None

    @property
    def axial_anchors(self) -> tuple[str, ...]:
        """Burnup anchors the axial head predicts, in output order."""
        return () if self._axial_basis is None else tuple(self._axial_basis.anchors)

    @torch.no_grad()
    def predict_axial(self, patterns: Sequence[Pattern],
                      case: CaseKey | Sequence[CaseKey]) -> dict[str, np.ndarray]:
        """Predicted axial power profiles + the scalars derived FROM them.

        Returns ``{"profile": [N, A, 25], "f_z": [N, A], "ao": [N, A],
        "asi": [N, A], "saddle_depth": [N, A], "anchors": (...)}``.

        The scalars are computed from the emitted profile by
        :mod:`lpopt.data.axial`, not by separate heads — so ``f_z`` and ``ao``
        can never disagree with the shape they are supposed to summarise, and a
        saddle is distinguishable from a double hump (which ``|AO|``, a first
        moment, structurally cannot do).

        Raises ``RuntimeError`` on a champion without the head, so a caller can
        never silently receive a fabricated profile.
        """
        from ..data.axial import derived_metrics

        if self._axial_basis is None:
            raise RuntimeError(
                "this champion has no axial head (meta 'axial_head' absent or "
                "disabled); retrain with --axial-head to serve axial profiles")
        cases, _ = self._broadcast(patterns, case, 0.0)
        cells_t, g_t = self._encode_batch(patterns, cases)
        acc = [m(cells_t, g_t)["axial"].float().cpu().numpy() for m in self.members]
        z = np.stack(acc).mean(axis=0)                       # [N, A, K]
        profile = self._axial_basis.z_decode(z)              # [N, A, P]
        out: dict[str, Any] = {"profile": profile,
                               "anchors": tuple(self._axial_basis.anchors)}
        out.update(derived_metrics(profile))
        return out

    def _encode_batch(self, patterns: Sequence[Pattern],
                      cases: Sequence[CaseKey]) -> tuple[torch.Tensor, torch.Tensor]:
        cells, gvecs = [], []
        for pat, case in zip(patterns, cases):
            c, g = self.encoder.encode(self._record_inputs(pat, case), self.fuel)
            cells.append(c)
            gvecs.append(g)
        cells_t = torch.from_numpy(np.ascontiguousarray(np.stack(cells)))
        g_t = torch.from_numpy(np.ascontiguousarray(np.stack(gvecs)))
        return cells_t.to(self.device), g_t.to(self.device)

    @staticmethod
    def _broadcast(patterns, case, cell):
        n = len(patterns)
        cases = case if isinstance(case, (list, tuple)) else [case] * n
        cells = cell if isinstance(cell, (list, tuple, np.ndarray)) else [cell] * n
        return list(cases), list(cells)

    @torch.no_grad()
    def _raw_forward_local(self, patterns: Sequence[Pattern],
                           cases: Sequence[CaseKey]):
        """Encode + run every ensemble member locally (the compute primitive).

        Returns raw ``(mu_z[M,N,T], log_sigma[M,N,T], conv_logit[M,N])`` in
        model-target (z) space.  This is the exact body the remote GPU entry
        (:func:`lpopt.model.remote_infer.run_request`) executes, so a remote
        round-trip on the same device yields identical arrays.
        """
        cells_t, g_t = self._encode_batch(patterns, cases)
        mu_z, log_sigma, conv_logit = [], [], []
        for m in self.members:
            out = m(cells_t, g_t)
            mu_z.append(out["mu"].float().cpu().numpy())
            log_sigma.append(out["log_sigma"].float().cpu().numpy())
            logit = torch.logit(torch.sigmoid(out["conv_logit"]).clamp(1e-6, 1 - 1e-6))
            conv_logit.append(logit.float().cpu().numpy())
        return (np.stack(mu_z), np.stack(log_sigma), np.stack(conv_logit))

    def _raw_forward(self, patterns: Sequence[Pattern], cases: Sequence[CaseKey]):
        """Dispatch the raw ensemble forward through remote screening + cache.

        Default (no screener, no cache) is the unchanged local compute — a plain
        delegation to :meth:`_raw_forward_local`, so the campaign's existing
        behaviour is byte-identical.  With a remote screener and/or a session
        cache attached (during a lean screen), the routed path memoizes per item
        and offloads large miss-batches to the GPU box, falling back to local CPU
        on ANY remote failure.
        """
        if self._remote is None and self._screen_cache is None:
            return self._raw_forward_local(patterns, cases)
        return self._raw_forward_routed(list(patterns), list(cases))

    # -- remote screening: routing + session cache (plan 4.7) --------------- #
    def _compute_missing(self, patterns: Sequence[Pattern],
                         cases: Sequence[CaseKey]):
        """Compute a miss-batch: remote GPU when large enough, else local CPU.

        The campaign must NEVER hard-fail because the server is unreachable, so
        ANY remote exception (or a ``None`` return) falls through to the local
        CPU path with a single log line.
        """
        if self._remote is not None and len(patterns) >= self._remote_min:
            try:
                out = self._remote(self, patterns, cases)
                if out is not None:
                    return out
                self._log_remote(
                    "remote screening returned no data; local CPU fallback")
            except Exception as exc:  # noqa: BLE001 — never abort the campaign
                self._log_remote(
                    f"remote screening failed ({type(exc).__name__}: {exc}); "
                    "local CPU fallback")
        return self._raw_forward_local(patterns, cases)

    def _raw_forward_routed(self, patterns: list[Pattern], cases: list[CaseKey]):
        n = len(patterns)
        if n == 0:
            t = len(self.target_names)
            m = len(self.members)
            return (np.zeros((m, 0, t)), np.zeros((m, 0, t)), np.zeros((m, 0)))
        cache = self._screen_cache
        if cache is None:                      # remote attached but no memo
            return self._compute_missing(patterns, cases)

        keys = [(p.canonical(), c.pair, int(c.feed))
                for p, c in zip(patterns, cases)]
        miss_idx = [i for i, k in enumerate(keys) if k not in cache]
        if miss_idx:
            mu_m, ls_m, cl_m = self._compute_missing(
                [patterns[i] for i in miss_idx], [cases[i] for i in miss_idx])
            for j, i in enumerate(miss_idx):
                cache[keys[i]] = (mu_m[:, j, :], ls_m[:, j, :], cl_m[:, j])
        mu = np.stack([cache[k][0] for k in keys], axis=1)
        ls = np.stack([cache[k][1] for k in keys], axis=1)
        cl = np.stack([cache[k][2] for k in keys], axis=1)
        return mu, ls, cl

    def _log_remote(self, msg: str) -> None:
        if self._remote_log is not None:
            self._remote_log(f"[remote_screening] {msg}")

    def enable_remote_screening(self, remote_fn, *, min_predictions: int = 5000,
                                log=None) -> None:
        """Route large screening miss-batches through ``remote_fn`` + memoize.

        ``remote_fn(backend, patterns, cases) -> (mu_z, log_sigma, conv_logit)``
        offloads to the GPU box; ``min_predictions`` gates it (smaller batches
        stay on local CPU — SSH overhead is not worth it).  A fresh session cache
        is installed so the per-cell scorer's repeat predict/convergence/extra
        passes over a prewarmed pool are pure memo hits.
        """
        self._remote = remote_fn
        self._remote_min = int(min_predictions)
        self._remote_log = log
        self._screen_cache = {}

    def disable_remote_screening(self) -> None:
        """Detach the screener + drop the session cache (e.g. after the screen)."""
        self._remote = None
        self._remote_min = 0
        self._remote_log = None
        self._screen_cache = None

    def prewarm(self, patterns: Sequence[Pattern],
                cases: Sequence[CaseKey]) -> None:
        """Predict a whole batch once (routed) to fill the session cache.

        The campaign calls this with every pattern its per-cell screen will
        subsequently score, so the bulk inference is one GPU round-trip and the
        per-cell :meth:`predict` calls are served from the memo.  Deduplicates by
        cache key so repeated patterns cost one compute.  No-op without a cache.
        """
        if self._screen_cache is None:
            return
        patterns = list(patterns)
        cases = list(cases)
        seen: set[tuple[str, str, int]] = set()
        uniq_p: list[Pattern] = []
        uniq_c: list[CaseKey] = []
        for p, c in zip(patterns, cases):
            k = (p.canonical(), c.pair, int(c.feed))
            if k in seen or k in self._screen_cache:
                continue
            seen.add(k)
            uniq_p.append(p)
            uniq_c.append(c)
        if uniq_p:
            self._raw_forward_routed(uniq_p, uniq_c)

    def _ensemble_raw(self, patterns: Sequence[Pattern], cases: Sequence[CaseKey]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw-space ``(mean, epistemic, calibrated)`` ``[N,T]`` in model-target order.

        ``T`` is the checkpoint's target count (5 for cond_v2, 7 for cond_v3);
        calibration is applied in that same order (the calibration file stores
        the identical ``target_names``).
        """
        mu_z, log_sigma, _ = self._raw_forward(patterns, cases)   # [M,N,T]
        members_raw = mu_z * self.tstd[None, None, :] + self.tmean[None, None, :]
        mean_t = members_raw.mean(axis=0)                         # [N,T]
        epistemic_t = members_raw.std(axis=0)
        # Physics-prior residual head: add the prior back so ``mean_t`` is
        # ABSOLUTE cyclen.  The prior is deterministic given the pattern, so it
        # shifts the mean only — the epistemic spread across members and the
        # aleatoric sigma below are unaffected, which is exactly right.
        prior_vals = self._cyclen_prior_values(patterns, cases)
        if prior_vals is not None:
            mean_t = mean_t.copy()
            mean_t[:, self._cyclen_target_idx] += prior_vals
        alea_raw = np.exp(log_sigma) * self.tstd[None, None, :]
        total_t = np.sqrt((alea_raw ** 2).mean(axis=0) + epistemic_t ** 2)
        calibrated_t = (
            apply_calibration(total_t, self.calibration, self.target_names)
            if self.calibration else total_t
        )
        return mean_t, epistemic_t, calibrated_t

    def _scalar_calibration_hooks(self) -> list[tuple]:
        """``[(surrogate_col, enabled, cells, global_by_lib, bin_width), ...]``.

        The single registry of per-cell scalar calibrations, read by
        :meth:`predict`.  Each entry keeps its own named public attributes
        (``fq_calibration`` / ``apply_fq_calibration`` / ``set_fq_calibration``…)
        so nothing about the external surface is table-driven — only the apply
        loop is, which is where five hand-copied blocks would otherwise drift.
        """
        return [
            (_CYCLEN_SURROGATE_COL, self.apply_cell_calibration,
             self._cell_cyclen_cells, self._cell_cyclen_global, self.cell_bin_width),
            (_FR_SURROGATE_COL, self.apply_fr_calibration,
             self._fr_cells, self._fr_global, self.fr_bin_width),
            (_CBC_SURROGATE_COL, self.apply_cbc_calibration,
             self._cbc_cells, self._cbc_global, self.cbc_bin_width),
            (_FQ_SURROGATE_COL, self.apply_fq_calibration,
             self._fq_cells, self._fq_global, self.fq_bin_width),
            (_AO_SURROGATE_COL, self.apply_ao_calibration,
             self._ao_cells, self._ao_global, self.ao_bin_width),
        ]

    # -- Protocol ----------------------------------------------------------- #
    def predict(self, patterns: Sequence[Pattern], case: CaseKey,
                cell: float = 0.0) -> SurrogatePrediction:
        """7-column ensemble prediction (``cell`` retained for API compat, unused).

        ``max_pin_burnup`` fills surrogate column 6 (exact match); column 5
        (``max_assembly_burnup``) stays NaN — ``discharge_burnup`` is served via
        :meth:`predict_extra`, never through the assembly-burnup constraint axis.

        When a ``cell_calibration.json`` is loaded and enabled, the **cyclen
        column (index 3)** of the means is passed through the fitted per-cell
        affine ``a*pred + b`` for any request whose (feed, e_core-bin) cell was
        fitted (unfitted cells + all other columns + sigma are untouched).  This
        corrects the champion's uniform per-cell cyclen over-prediction.  A
        Four siblings do the identical thing on disjoint columns:
        ``f_r_calibration.json`` (**index 0**, correcting the uniform per-cell F_r
        UNDER-prediction), ``cbc_calibration.json`` (**index 1**, the CBC_max
        OVER-prediction), ``f_q_calibration.json`` (**index 2**, the F_q
        UNDER-prediction — 71% of its error is that one shift) and
        ``ao_abs_calibration.json`` (**index 4**).  Every one carries the same
        guarantees: an ABSENT artifact leaves predict() byte-identical, each fitted
        ``a`` is > 0 so the map is monotone and within-cell ranking is unchanged,
        and sigma is never touched.  A row whose cell was not fitted falls back to
        its own serve LIBRARY's pooled shift when the artifact carries one.
        (``node_peak`` / ``map_cov`` are NOT here — they are map-head scalars and
        are corrected in :meth:`predict_map_flatness`.)

        Gate parity: the no-regression gate scores every champion through this
        same ``predict`` path, and each champion loads its OWN
        ``cell_calibration.json`` (or none) — so the calibration is applied
        symmetrically to both sides of a gate comparison and cannot contaminate
        it (an old champion without an artifact is compared raw-vs-raw; a new one
        with an artifact is compared calibrated-vs-calibrated, which is exactly
        the served behaviour being gated).
        """
        patterns = list(patterns)
        if not patterns:
            empty = np.zeros((0, _N_SURROGATE))
            return SurrogatePrediction(empty, empty.copy(), empty.copy())
        cases, _ = self._broadcast(patterns, case, cell)
        mean_t, epistemic_t, calibrated_t = self._ensemble_raw(patterns, cases)
        mean = _to_surrogate(mean_t, self.target_names)
        epistemic = _to_surrogate(epistemic_t, self.target_names)
        calibrated = _to_surrogate(calibrated_t, self.target_names)
        # The five per-cell scalar corrections, on DISJOINT columns.  Driven off
        # one table rather than five copies of the same five lines: the copies
        # drifted once already (the cbc hook shipped before the effective-library
        # key fix reached it), and a table cannot drift.  Order is irrelevant
        # between them — but the whole block must precede the pin-burnup override,
        # which reads the already-calibrated cyclen.
        hooks = self._scalar_calibration_hooks()
        # Per-row serve library keys the per-library global fallback (for a row
        # whose cell missed min_rows).  Resolved once, and only when some enabled
        # artifact actually carries a fallback map.
        libs = None
        if any(enabled and gmap for _c, enabled, cells, gmap, _bw in hooks):
            libs = [self.serve_library(p) for p in patterns]
        for col, enabled, cells, gmap, bin_width in hooks:
            if not enabled or not (cells or gmap):
                continue
            keys = self._calib_cell_keys(patterns, cases, bin_width)
            mean[:, col] = apply_affine_calibration(
                mean[:, col], keys, cells, globals_by_lib=gmap, libraries=libs)
        # physics pin-burnup override (uses the already-calibrated served cyclen);
        # raw head sigma (epistemic/calibrated) is preserved.
        if self.apply_pinbu_physics and self._pinbu is not None:
            mean[:, PINBU_SURROGATE_COL] = self._pinbu_column(
                patterns, cases, mean[:, _CYCLEN_SURROGATE_COL],
                mean[:, PINBU_SURROGATE_COL],
            )
        if self.has_quantiles():
            # ADDITIVE: a SurrogatePrediction subclass, so the 7-column contract
            # and every existing consumer are unchanged.
            q = self._quantile_raw(patterns, cases,
                                   self._cyclen_prior_values(patterns, cases))
            return QuantileSurrogatePrediction(
                mean, epistemic, calibrated,
                quantiles=q, quantile_targets=self.quantile_targets,
                quantile_levels=self.quantile_levels)
        return SurrogatePrediction(mean, epistemic, calibrated)

    def _map_norm(self, meta: dict, channel: int) -> tuple[float, float]:
        """``(mean, std)`` that de-normalizes ONE member's map channel.

        Each member stamps its own ``map_zscore`` at train time, so the ensemble
        must be de-normalized MEMBER BY MEMBER (a shared constant would be wrong
        for any ensemble whose members saw different training folds).  A
        checkpoint without the field is treated as already-physical — the honest
        reading of "this head stored no normalization".
        """
        mz = meta.get("map_zscore") or {}
        try:
            mu = float(np.asarray(mz["mean"], dtype=float)[int(channel)])
            sd = float(np.asarray(mz["std"], dtype=float)[int(channel)])
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.0, 1.0
        return (mu, sd) if math.isfinite(mu) and math.isfinite(sd) and sd > 0 else (0.0, 1.0)

    def predict_map_flatness(self, patterns: Sequence[Pattern], case: CaseKey,
                             cell: float = 0.0, *, channel: int = 0
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(peak_mean, peak_std, cov_mean, cov_std)`` in PHYSICAL units.

        The flatness-first objective (program §1.2) consumes ``node_peak``
        (primary) and ``map_cov`` (secondary) as LEVELS, not ranks, so this
        de-normalizes every member's map head with that member's own
        ``map_zscore``, gathers the 69 quarter-core slots, and applies the ONE
        canonical definition (:mod:`..data.flatness` — multiplicity weighted).

        Convention: **mean-of-stat**, i.e. the scalar is computed per member and
        THEN averaged, so the reported ``std`` is the across-member spread of the
        scalar itself (which is what the UCB ``mean + risk_z*std`` needs).  The
        alternative (stat-of-the-mean-map) is optimistic by Jensen and carries no
        usable spread; program §6.1 has it queued for measurement, and this
        docstring is where the chosen convention is recorded.

        ADDITIVE: a fresh forward that touches no existing head and not the
        7-column contract.

        **Per-cell INTERCEPT-only calibration** (2026-07-29 all-targets).  When a
        ``flatness_calibration.json`` is loaded and enabled, each MEAN is shifted by
        its cell's fitted ``b`` (or its serve library's pooled ``b`` when the cell
        was not fitted), correcting the map head's measured optimism — curriculum-val
        bias −0.0462 on node_peak and −0.0272 on map_cov, the latter 84% of that
        axis's whole error.  The two ``std`` arrays are NOT touched.

        The correction is a pure TRANSLATION (``a == 1`` by construction, see
        :func:`cell_calibrate.fit_flatness_calibration`), so within one calibration
        cell the candidate ORDER is bit-identical before and after and the honest
        no-regression gate's within-cell Spearman cannot move.  Honest scope: a
        CURRICULUM cell spans several 0.05-wide calibration cells, and rows in
        different bins receive different shifts, so ranks can still move across
        bins — the same property the four scalar calibrations already have.  An
        absent artifact is a pure no-op.

        **OOD sigma floor** (2026-07-29 debug-panel): the two returned ``std``
        arrays are floored at :attr:`flatness_sigma_floor` (defaults node_peak 0.06
        / map_cov 0.02, resolved per checkpoint by
        :meth:`_resolve_flatness_sigma_floor`).  The unfloored spread is the
        across-member disagreement of an ensemble whose members were trained on the
        same corpus, so on a genuinely out-of-distribution core they agree with each
        other and are confidently wrong together: one blind OOD case measured
        err/sigma of −12.8 (node_peak) and −7.5 (map_cov).  Since the flatness UCB
        (``mean + risk_z*std``) and every coverage statistic read this number, an
        order-of-magnitude-too-small sigma is not a reporting nuisance — it makes
        the acquisition treat an unknown core as a known one.  The floor is a
        constant, i.e. deliberately crude: a distance-aware inflation keyed on
        :meth:`feature_ood_types` is the queued A/B arm, NOT this change.  Set
        ``apply_flatness_sigma_floor = False`` to recover the raw spread.
        """
        from ..data.flatness import SLOT_COLS, SLOT_ROWS, flatness_pair

        patterns = list(patterns)
        if not patterns:
            z = np.zeros(0)
            return z, z.copy(), z.copy(), z.copy()
        cases, _ = self._broadcast(patterns, case, cell)
        cells_t, g_t = self._encode_batch(patterns, cases)
        peaks: list[np.ndarray] = []
        covs: list[np.ndarray] = []
        with torch.no_grad():
            for m, meta in zip(self.members, self.metas):
                out = m(cells_t, g_t)
                plane = out["map"][:, int(channel)].float().cpu().numpy()  # [N,9,9]
                mu, sd = self._map_norm(meta, channel)
                slots = plane[:, SLOT_ROWS, SLOT_COLS] * sd + mu           # [N,69]
                pk, cv = flatness_pair(slots)
                peaks.append(pk)
                covs.append(cv)
        pk_stack = np.stack(peaks)                          # [M, N]
        cv_stack = np.stack(covs)
        pk_mean, cv_mean = pk_stack.mean(axis=0), cv_stack.mean(axis=0)
        pk_sd, cv_sd = pk_stack.std(axis=0), cv_stack.std(axis=0)
        # per-cell INTERCEPT-only shift on the MEANS only (never the spread), on
        # the SE-quadrant power channel the flatness scalars are defined on.  A
        # request for another channel is a diagnostic read of a different physical
        # quantity, so the boc_power-fitted shift must not be applied to it.
        if (self.apply_flatness_calibration and int(channel) == 0
                and (any(self._flat_cells.values())
                     or any(self._flat_global.values()))):
            keys = self._flatness_cell_keys(patterns, cases)
            libs = ([self.serve_library(p) for p in patterns]
                    if any(self._flat_global.values()) else None)
            pk_mean = apply_affine_calibration(
                pk_mean, keys, self._flat_cells.get("node_peak", {}),
                globals_by_lib=self._flat_global.get("node_peak", {}), libraries=libs)
            cv_mean = apply_affine_calibration(
                cv_mean, keys, self._flat_cells.get("map_cov", {}),
                globals_by_lib=self._flat_global.get("map_cov", {}), libraries=libs)
        if self.apply_flatness_sigma_floor:
            floor = self.flatness_sigma_floor
            pk_sd = np.maximum(pk_sd, float(floor.get("node_peak", 0.0)))
            cv_sd = np.maximum(cv_sd, float(floor.get("map_cov", 0.0)))
        return pk_mean, pk_sd, cv_mean, cv_sd

    def predict_map_peak(self, patterns: Sequence[Pattern], case: CaseKey,
                         cell: float = 0.0, *, channel: int = 0
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble NODE-POWER map-peak ``(mean[N], epistemic_std[N])``.

        RE-POINTED (program §13) at :meth:`predict_map_flatness` so there is ONE
        computation of the peak: this returns its first two outputs, i.e. the
        multiplicity-weighted ``node_peak`` in PHYSICAL units (== F_xy) rather
        than the head's z-space max the previous implementation returned.  The
        z-space max was monotone with the physical peak and therefore fine for a
        pure ranking, but the flatness objective consumes LEVELS (a scale, a bias
        correction, a record column to compare against), and a z-space number
        cannot be compared with the ``node_peak`` column MASTER labels write.
        """
        peak_mean, peak_std, _cov_mean, _cov_std = self.predict_map_flatness(
            patterns, case, cell, channel=channel)
        return peak_mean, peak_std

    def predict_extra(self, patterns: Sequence[Pattern], case: CaseKey,
                      cell: float = 0.0) -> ExtraPrediction:
        """Ensemble prediction for the non-surrogate targets (``discharge_burnup``).

        Returns a NaN column for any extra target this checkpoint does not
        predict (e.g. a cond_v2 5-target model), so callers get a stable schema.
        """
        patterns = list(patterns)
        names = _EXTRA_TARGET_NAMES
        if not patterns:
            empty = np.full((0, len(names)), np.nan)
            return ExtraPrediction(names, empty, empty.copy(), empty.copy())
        cases, _ = self._broadcast(patterns, case, cell)
        mean_t, epistemic_t, calibrated_t = self._ensemble_raw(patterns, cases)
        n = mean_t.shape[0]
        idx = {name: k for k, name in enumerate(self.target_names)}

        def _col(mat: np.ndarray, name: str) -> np.ndarray:
            k = idx.get(name)
            return mat[:, k] if k is not None else np.full(n, np.nan)

        mean = np.column_stack([_col(mean_t, nm) for nm in names])
        epistemic = np.column_stack([_col(epistemic_t, nm) for nm in names])
        calibrated = np.column_stack([_col(calibrated_t, nm) for nm in names])
        return ExtraPrediction(names, mean, epistemic, calibrated)

    def predict_convergence(self, patterns: Sequence[Pattern], case: CaseKey,
                            cell: float = 0.0) -> np.ndarray:
        patterns = list(patterns)
        if not patterns:
            return np.zeros(0, dtype=float)
        cases, _ = self._broadcast(patterns, case, cell)
        _, _, conv_logit = self._raw_forward(patterns, cases)     # [M,N]
        mean_logit = conv_logit.mean(axis=0)
        if self.calibration is not None:
            return apply_platt(mean_logit, self.calibration)
        return 1.0 / (1.0 + np.exp(-mean_logit))

    # -- report-only split-conformal intervals (conformal.py) --------------- #
    def predict_interval(self, patterns: Sequence[Pattern], case: CaseKey,
                         cell: float = 0.0, *, alpha: float = 0.10
                         ) -> IntervalPrediction:
        """Per-target split-conformal interval for a served batch (REPORT-ONLY).

        Centers the interval on the served :meth:`predict` mean (already cyclen-/
        pin-calibrated) and adds the fitted half-width for the request's
        ``(feed, e_core-bin)`` cell (per-cell quantile when the cell was fitted, else
        the per-target global fallback) at miscoverage ``alpha`` (0.10 -> 90%,
        0.32 -> 68% — the fitted levels).  This method is the ONLY reader of the
        conformal artifact and changes NO other output: a caller that never asks for
        an interval sees byte-identical behaviour.

        A champion without a ``conformal.json`` returns an ``available=False``
        all-NaN interval (still a valid, callable accessor).
        """
        patterns = list(patterns)
        names = tuple(TARGET_NAMES)
        coverage = 1.0 - float(alpha)
        if not patterns:
            empty = np.zeros((0, _N_SURROGATE))
            return IntervalPrediction(
                float(alpha), coverage, names, empty, empty.copy(), empty.copy(),
                np.zeros((0, _N_SURROGATE), dtype=bool), self.has_conformal())
        cases, _ = self._broadcast(patterns, case, cell)
        pred = self.predict(patterns, cases)          # served mean + calibrated_std
        if not self._conformal_targets:
            nan = np.full((len(patterns), _N_SURROGATE), np.nan)
            return IntervalPrediction(
                float(alpha), coverage, names, nan, nan.copy(), nan.copy(),
                np.zeros((len(patterns), _N_SURROGATE), dtype=bool), False)
        keys = self._conformal_cell_keys(patterns, cases)
        lower, upper, hw, from_cell = interval_arrays(
            pred.mean, pred.calibrated_std, keys, self.conformal, float(alpha))
        return IntervalPrediction(
            float(alpha), coverage, names, lower, upper, hw, from_cell, True)

    def position_values(self, pattern: Pattern, case: CaseKey,
                        cell: float = 0.0) -> None:
        """v1: attribution deferred (plan sec. 4.4 lists it as a later item)."""
        return None

    # -- fine-tune (local CPU wave update, plan sec. 4.7) ------------------- #
    def finetune(self, new: Any, replay: Any, epochs: int = 3,
                 seed: int = 0, *, lr: float = 1.0e-4) -> dict:
        """Fine-tune every member on ``new`` + ``replay`` store rows (local CPU).

        Raises :class:`EncoderChannelMismatch` when the loaded members' stem width
        no longer matches the active encoder (a cond-schema/feature-schema change,
        e.g. a v3 champion under a v4 encoder): fine-tuning cannot change the input
        width, so a full retrain is required.  The curriculum's retrain guard
        catches this and switches to a from-scratch (remote_full) retrain.
        """
        n_enc = len(getattr(self.encoder, "channels", ()))
        bad = sorted({int(getattr(m.config, "in_channels", n_enc)) for m in self.members
                      if int(getattr(m.config, "in_channels", n_enc)) != n_enc})
        if bad:
            raise EncoderChannelMismatch(
                f"cannot fine-tune: member in_channels {bad} != encoder channel "
                f"count {n_enc} (cond_schema {self.cond_schema!r}); full retrain required"
            )
        # Weights are about to change; any session memo is now stale.
        if self._screen_cache is not None:
            self._screen_cache.clear()
        new_df = _as_frame(new)
        replay_df = _as_frame(replay)
        # Align replay dtypes to the fresh rows so an all-None column in one
        # frame does not trip the pandas>=2.1 all-NA concat FutureWarning.
        if len(new_df) and len(replay_df):
            shared = [c for c in new_df.columns if c in replay_df.columns]
            replay_df = replay_df.copy()
            new_df = new_df.copy()
            for col in shared:
                if replay_df[col].isna().all() or new_df[col].isna().all():
                    replay_df[col] = replay_df[col].astype(object)
                    new_df[col] = new_df[col].astype(object)
        combined = pd.concat([new_df, replay_df], ignore_index=True)
        n_new, n_replay = len(new_df), len(replay_df)
        if combined.empty:
            return {"wall_seconds": 0.0, "n_new": 0, "n_replay": 0, "epochs": epochs}

        cells, gvec, y, ymask, clabel, cmask = self._featurize_rows(combined)
        # Residual head: the fine-tune must optimize the SAME quantity training
        # did, so subtract the physics prior from the cyclen label here too.
        prior_vals = self._cyclen_prior_rows(combined)
        if prior_vals is not None:
            y = y.clone()
            y[:, self._cyclen_target_idx] -= torch.as_tensor(
                prior_vals, dtype=y.dtype, device=y.device)
        cfg = TrainConfig()
        tmean = torch.as_tensor(self.tmean, dtype=torch.float32, device=self.device)
        tstd = torch.as_tensor(self.tstd, dtype=torch.float32, device=self.device)
        n = cells.shape[0]
        batch = min(256, n)
        t0 = time.time()
        for m in self.members:
            torch.manual_seed(seed)
            m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=cfg.weight_decay)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for s0 in range(0, n, batch):
                    idx = perm[s0:s0 + batch]
                    out = m(cells[idx], gvec[idx])
                    z_t = (y[idx] - tmean) / tstd
                    loss = regression_loss(out["mu"], out["log_sigma"], z_t, ymask[idx],
                                           use_nll=True, beta=cfg.beta_nll,
                                           delta=cfg.huber_delta)
                    loss = loss + cfg.conv_weight * convergence_loss(
                        out["conv_logit"], clabel[idx], cmask[idx])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            m.eval()
        return {"wall_seconds": round(time.time() - t0, 2),
                "n_new": n_new, "n_replay": n_replay, "epochs": epochs,
                "n_members": len(self.members)}

    def _featurize_rows(self, df: pd.DataFrame):
        # Target inventory follows the CHECKPOINT (7 targets today, 8 under
        # ``promote_max_asm_bu``) so the fine-tune label matrix always matches the
        # head width.  For every existing checkpoint this equals ``TARGETS``.
        names = self.target_names
        cells, gvec = [], []
        y = np.full((len(df), len(names)), np.nan, dtype=np.float32)
        ymask = np.zeros((len(df), len(names)), dtype=np.float32)
        clabel = np.zeros(len(df), dtype=np.float32)
        cmask = np.zeros(len(df), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            c, g = self.encoder.encode(RecordInputs.coerce(row), self.fuel)
            cells.append(c)
            gvec.append(g)
            converged = bool(row.get("converged", True))
            clabel[i] = 1.0 if converged else 0.0
            cmask[i] = 0.0 if bool(row.get("converged_at_cap", False)) else 1.0
            boc = str(row.get("cbc_kind", "")) == "boc_only"
            for k, name in enumerate(names):
                v = row.get(name)
                fv = float(v) if v is not None and not pd.isna(v) else float("nan")
                y[i, k] = fv
                ok = converged and np.isfinite(fv) and not (name == "cbc_max" and boc)
                ymask[i, k] = 1.0 if ok else 0.0
        cells_t = torch.from_numpy(np.ascontiguousarray(np.stack(cells))).to(self.device)
        g_t = torch.from_numpy(np.ascontiguousarray(np.stack(gvec))).to(self.device)
        return (cells_t, g_t,
                torch.from_numpy(y).to(self.device),
                torch.from_numpy(ymask).to(self.device),
                torch.from_numpy(clabel).to(self.device),
                torch.from_numpy(cmask).to(self.device))

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for meta, model in zip(self.metas, self.members):
            save_member(out / f"member_{meta['seed']}", model, meta)
        if self.calibration is not None:
            (out / CALIB_NAME).write_text(
                json.dumps(self.calibration, indent=2, sort_keys=True),
                encoding="utf-8")
        # Freeze the training-population feature envelope next to the champion so the
        # serve-time OOD guard uses the exact train-time range (review sec. 4b).
        (out / _FEATURE_OOD_NAME).write_text(
            json.dumps({"margin": _OOD_MARGIN, "library_id": self.library_id,
                        "envelope": self.feature_ood_envelope()},
                       indent=2, sort_keys=True),
            encoding="utf-8")
        (out / _BACKEND_MANIFEST).write_text(
            json.dumps({"backend": "posval_cnn", "library_id": self.library_id,
                        "n_members": len(self.members),
                        "cond_schema": self.cond_schema,
                        # Round-trip the resolved map-head OOD sigma floor so a
                        # re-load reproduces the served spread exactly (an older
                        # manifest simply lacks the key -> defaults, unchanged).
                        "flatness_sigma_floor": dict(self.flatness_sigma_floor),
                        "target_names": list(self.target_names),
                        "channels": list(getattr(self.encoder, "channels", ())),
                        "globals": list(getattr(self.encoder, "globals_names", ()))},
                       indent=2, sort_keys=True),
            encoding="utf-8")
        return out


def _cross_check_manifest(manifest: dict, metas: Sequence[dict]) -> None:
    """Reject a checkpoint whose ``backend.json`` channels/globals contradict the
    members' metas (both are frozen at train time, so they must agree)."""
    if not metas:
        return
    meta0 = metas[0]
    for key in ("channels", "globals"):
        man = manifest.get(key)
        met = meta0.get(key)
        if man is not None and met is not None and list(man) != list(met):
            raise EncoderChannelMismatch(
                f"backend.json {key!r} (len {len(man)}) disagrees with member "
                f"meta {key!r} (len {len(met)}); the checkpoint dir is inconsistent"
            )


def _as_frame(rows: Any) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows
    if rows is None:
        return pd.DataFrame()
    return pd.DataFrame(list(rows))


__all__ = [
    "PositionValueModel", "PosValCnnBackend", "ExtraPrediction",
    "IntervalPrediction", "QuantileSurrogatePrediction", "TARGET_NAMES",
    "EncoderChannelMismatch",
]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble_dir")
    args = ap.parse_args()
    backend = PosValCnnBackend.from_dir(args.ensemble_dir)
    print("loaded", len(backend.members), "members; library", backend.library_id)
