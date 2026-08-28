"""Ensemble surrogate trained from the packaged GA-Surrogate warm seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .domain import CaseKey, FOM, Pattern, PatternRecord
from .features import FeatureEncoder


TARGET_NAMES: tuple[str, ...] = (
    "F_r",
    "CBC_max",
    "F_q",
    "cyclen",
    "AO_abs",
    "max_assembly_burnup",
    "max_pin_burnup",
)
PRIMARY_TARGET_COUNT = 4

# Upper bound on cached prediction rows.  On overflow the cache is cleared
# wholesale (simple and safe); a warm PPO run revisits far fewer than this.
_PREDICTION_CACHE_CAP = 100_000
# Minimum number of trees kept when subsampling for the epistemic estimate.
_MIN_STD_TREES = 8
# Below this many rows a forest prediction runs single-threaded: the joblib
# fan-out across a 24-thread pool costs more than the trees on tiny batches.
_PARALLEL_PREDICT_MIN_ROWS = 128


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Spearman rank correlation of one holdout (NaN when undefined)."""

    if len(truth) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(truth, prediction)[0])
    except Exception:  # pragma: no cover - scipy is a sklearn dependency
        return float("nan")


@dataclass(frozen=True)
class SurrogatePrediction:
    """Mean and uncertainty arrays in ``TARGET_NAMES`` order."""

    mean: np.ndarray
    epistemic_std: np.ndarray
    calibrated_std: np.ndarray

    def __post_init__(self) -> None:
        for value in (self.mean, self.epistemic_std, self.calibrated_std):
            if value.ndim != 2 or value.shape[1] != len(TARGET_NAMES):
                raise ValueError(f"prediction has invalid shape {value.shape}")

    def row(self, index: int) -> dict[str, dict[str, float]]:
        # ``known`` marks whether a real per-target model backs this value: a
        # NaN calibrated std is the "no labels, prediction is only the case
        # baseline" sentinel and must never be mistaken for zero uncertainty.
        return {
            name: {
                "mean": float(self.mean[index, column]),
                "epistemic_std": float(self.epistemic_std[index, column]),
                "calibrated_std": float(self.calibrated_std[index, column]),
                "known": bool(math.isfinite(float(self.calibrated_std[index, column]))),
            }
            for column, name in enumerate(TARGET_NAMES)
        }

    def mean_fom(self, index: int) -> FOM:
        values = self.mean[index]

        # An unknown axis (NaN calibrated std: no per-target model, the mean
        # is only the case baseline) must stay unknown in the FOM.
        # ``is_fom_feasible`` treats the missing value as a violation, so a
        # baseline 0.0 can never certify feasibility in surrogate dry-runs.
        def known(column: int) -> bool:
            return bool(math.isfinite(float(self.calibrated_std[index, column])))

        return FOM(
            f_r=float(values[0]),
            cbc_max=float(values[1]),
            f_q=float(values[2]),
            cyclen=float(values[3]),
            ao_min=-float(values[4]) if known(4) else None,
            ao_max=float(values[4]) if known(4) else None,
            max_burnup=float(values[5]) if known(5) else None,
            max_pin_burnup=float(values[6]) if known(6) else None,
        )


@dataclass(frozen=True)
class TargetQuality:
    """Actionable per-target fit statistics for the surrogate quality gate."""

    name: str
    label_count: int
    holdout_count: int
    r2: float
    mae: float
    baseline_mae: float
    skill: float          # 1 - mae/baseline_mae, NaN if baseline invalid
    coverage_1s: float    # frac(|truth-pred| <= calibrated_std), NaN if holdout small
    trainable: bool
    passed: bool


@dataclass(frozen=True)
class SurrogateQualityReport:
    """Gate verdict consumed by journal/flow before any MASTER budget is spent.

    The verdict statistic is skill (1 - mae/baseline_mae) against the case-mean
    baseline; R² on the tiny holdouts is retained for information only.

    ``mode`` semantics:
      * ``halt``      — F_r or cyclen is trainable and evaluable yet provably
        no better than the case-mean baseline (skill <= min_skill_halt).
        Spending MASTER budget on this model is not defensible.
      * ``explore``   — quality is unproven (label/holdout shortage on any
        primary target or an enabled constraint axis) or a secondary primary
        target (CBC_max, F_q) fails.  More MASTER data is the cure, so the
        acquisition switches to exploration instead of stopping.
      * ``objective`` — every gate passed; normal exploitation is licensed.
    """

    gate_passed: bool
    mode: str             # "objective" | "explore" | "halt"
    per_target: dict[str, TargetQuality]
    reasons: list[str]
    # Every threshold the verdict was judged against (audit trail, N-6).
    thresholds: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "gate_passed": self.gate_passed,
            "mode": self.mode,
            "per_target": {
                name: asdict(quality) for name, quality in self.per_target.items()
            },
            "reasons": list(self.reasons),
            "thresholds": dict(self.thresholds),
        }


class SurrogateEnsemble:
    """Bootstrap Extra-Trees ensemble with case-stratified holdout calibration."""

    def __init__(
        self,
        *,
        n_estimators: int = 256,
        min_samples_leaf: int = 2,
        max_features: float = 0.7,
        random_seed: int = 20260711,
        n_jobs: int = -1,
        std_tree_stride: int = 4,
        validation_fraction: float = 0.2,
        gate_min_skill: float = 0.10,
        gate_holdout_repeats: int = 5,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = float(max_features)
        self.random_seed = int(random_seed)
        self.n_jobs = int(n_jobs)
        # Sticky holdout fraction: ``fit`` defaults to this value, so wave
        # refits (which call ``fit(records)`` without kwargs) keep the deck's
        # validation_fraction instead of silently reverting to 0.2 (N-3).
        self.validation_fraction = float(validation_fraction)
        # Quality-gate thresholds consumed by ``quality_report``/``fit`` when
        # their keyword arguments are left at None (deck-driven, F-07).
        self.gate_min_skill = float(gate_min_skill)
        self.gate_holdout_repeats = max(1, int(gate_holdout_repeats))
        # Trees are strided when estimating epistemic std; stride 1 reproduces
        # the exact per-tree spread, larger strides trade a second-order error
        # for a large speed-up (see :meth:`predict`).
        self.std_tree_stride = max(1, int(std_tree_stride))
        # Per-row prediction cache keyed by (pattern.canonical(), case, cell).
        self._prediction_cache: dict[
            tuple[str, CaseKey, float],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self.encoder: FeatureEncoder | None = None
        self.models: tuple[ExtraTreesRegressor | None, ...] | None = None
        self.target_mean: np.ndarray | None = None
        self.target_scale: np.ndarray | None = None
        self.case_target_mean: dict[CaseKey, np.ndarray] | None = None
        self.residual_rmse: np.ndarray | None = None
        self.validation_metrics: dict[str, dict[str, Any]] = {}
        # Holdout truth/prediction pairs retained for parity/calibration
        # figures; keyed by target name.
        self.holdout_predictions: dict[str, dict[str, list[float]]] = {}
        self.training_size = 0
        # Usable (finite, converged) label count per target column.
        self.label_counts: np.ndarray | None = None
        # Whether a per-target residual model exists (>= 2 usable labels).
        self.target_known: tuple[bool, ...] | None = None
        # Per-record converged flags plus the optional convergence classifier
        # fitted on them; non-converged last-iterate values are censored from
        # every regression target and only inform this classifier.
        self.convergence_labels: np.ndarray | None = None
        self.convergence_model: ExtraTreesClassifier | None = None
        # Training-data provenance stamped by the flow when the model is
        # fitted: sha256 of the seed store's manifest.csv bytes plus the store
        # path.  The flow refuses to reuse a model whose fingerprint differs
        # from the current store unless [surrogate] allow_foreign_model.
        self.manifest_sha256: str | None = None
        self.package_root: str | None = None

    @staticmethod
    def _target_matrix(records: Sequence[PatternRecord]) -> np.ndarray:
        return np.asarray(
            [
                (
                    record.fom.f_r,
                    record.fom.cbc_max,
                    record.fom.f_q,
                    record.fom.cyclen,
                    np.nan if record.fom.ao_abs is None else record.fom.ao_abs,
                    (
                        np.nan
                        if record.fom.max_assembly_burnup is None
                        else record.fom.max_assembly_burnup
                    ),
                    (
                        np.nan
                        if record.fom.max_pin_burnup is None
                        else record.fom.max_pin_burnup
                    ),
                )
                for record in records
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _stratified_holdout(
        records: Sequence[PatternRecord], fraction: float, random_seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_seed)
        by_case: dict[CaseKey, list[int]] = {}
        for index, record in enumerate(records):
            by_case.setdefault(record.case, []).append(index)
        test: list[int] = []
        for indices in by_case.values():
            shuffled = np.asarray(indices, dtype=int)
            rng.shuffle(shuffled)
            count = min(len(indices) - 1, max(1, int(round(len(indices) * fraction))))
            test.extend(int(value) for value in shuffled[:count])
        test_indices = np.asarray(sorted(test), dtype=int)
        train_indices = np.setdiff1d(np.arange(len(records), dtype=int), test_indices)
        return train_indices, test_indices

    def _new_model(
        self, *, target_index: int, n_estimators: int | None = None
    ) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=n_estimators or self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=False,
            random_state=self.random_seed + 1009 * target_index,
            n_jobs=self.n_jobs,
        )

    def _prediction_row_cache(
        self,
    ) -> dict[tuple[str, CaseKey, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return the per-row cache, lazily creating it for legacy pickles."""

        cache = getattr(self, "_prediction_cache", None)
        if cache is None:
            cache = {}
            self._prediction_cache = cache
        return cache

    def _std_tree_stride(self) -> int:
        """Return the epistemic tree stride, defaulting for legacy pickles."""

        return max(1, int(getattr(self, "std_tree_stride", 4)))

    def fit(
        self,
        records: Sequence[PatternRecord],
        *,
        validation_fraction: float | None = None,
        holdout_repeats: int | None = None,
    ) -> "SurrogateEnsemble":
        # ``None`` inherits the instance settings, so wave refits that call
        # ``fit(records)`` keep the configured holdout fraction/repeats
        # instead of reverting to hard-coded defaults (N-3).
        fraction = (
            float(getattr(self, "validation_fraction", 0.2))
            if validation_fraction is None
            else float(validation_fraction)
        )
        repeats = max(
            1,
            int(getattr(self, "gate_holdout_repeats", 5))
            if holdout_repeats is None
            else int(holdout_repeats),
        )
        # A refit changes the mapping from inputs to predictions, so any cached
        # rows from an earlier fit must never be served afterwards.
        self._prediction_row_cache().clear()
        if len(records) < 8:
            raise ValueError("at least eight evaluated patterns are required")
        cases = tuple(record.case for record in records)
        self.encoder = FeatureEncoder(cases)
        x = self.encoder.transform(
            [record.pattern for record in records],
            cases,
            [record.cell for record in records],
        )
        y = self._target_matrix(records)
        # Non-converged equilibria report last-iterate values, not physics;
        # censor them from every regression target.  The pattern itself stays
        # informative through the convergence classifier fitted below.
        nonconverged = np.asarray(
            [not record.fom.converged for record in records], dtype=bool
        )
        y[nonconverged, :] = np.nan
        self.convergence_labels = ~nonconverged
        self.label_counts = np.asarray(
            [int(np.isfinite(y[:, column]).sum()) for column in range(len(TARGET_NAMES))],
            dtype=int,
        )
        # Repeated case-stratified holdouts (F-07): repeat 0 reproduces the
        # historical single split exactly; further repeats decorrelate the
        # skill estimate from one lucky/unlucky partition.
        primary_splits = [
            self._stratified_holdout(records, fraction, self.random_seed + 9973 * r)
            for r in range(repeats)
        ]
        defaults = np.zeros(len(TARGET_NAMES), dtype=float)
        self.target_mean = np.asarray(
            [
                float(np.mean(y[np.isfinite(y[:, column]), column]))
                if np.any(np.isfinite(y[:, column]))
                else defaults[column]
                for column in range(len(TARGET_NAMES))
            ]
        )
        self.target_scale = np.asarray(
            [
                float(np.std(y[np.isfinite(y[:, column]), column]))
                if np.any(np.isfinite(y[:, column]))
                else 1.0
                for column in range(len(TARGET_NAMES))
            ]
        )
        self.target_scale[self.target_scale < 1.0e-12] = 1.0
        self.case_target_mean = {}
        for case in set(cases):
            values = np.empty(len(TARGET_NAMES), dtype=float)
            case_indices = np.asarray(
                [index for index, record in enumerate(records) if record.case == case],
                dtype=int,
            )
            for column in range(len(TARGET_NAMES)):
                available = case_indices[np.isfinite(y[case_indices, column])]
                values[column] = (
                    float(np.mean(y[available, column]))
                    if len(available)
                    else self.target_mean[column]
                )
            self.case_target_mean[case] = values

        residual_rmse = np.zeros(len(TARGET_NAMES), dtype=float)
        metrics: dict[str, dict[str, Any]] = {}
        holdout: dict[str, dict[str, list[float]]] = {}
        fitted_models: list[ExtraTreesRegressor | None] = []
        case_set = set(cases)
        rng = np.random.default_rng(self.random_seed + 4241)

        def _holdout_evaluation(
            column: int,
            validation_train: np.ndarray,
            validation_test: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
            """Fit one fold model and return (truth, prediction, test_base)."""

            if len(validation_train) < 2 or len(validation_test) < 1:
                return None
            fold_case_mean: dict[CaseKey, float] = {}
            for case in case_set:
                matching = [
                    index
                    for index in validation_train
                    if records[int(index)].case == case
                ]
                fold_case_mean[case] = (
                    float(np.mean(y[matching, column]))
                    if matching
                    else float(np.mean(y[validation_train, column]))
                )
            train_base = np.asarray(
                [fold_case_mean[records[int(index)].case] for index in validation_train]
            )
            test_base = np.asarray(
                [fold_case_mean[records[int(index)].case] for index in validation_test]
            )
            validation_model = self._new_model(
                target_index=column, n_estimators=min(self.n_estimators, 192)
            )
            validation_model.fit(
                x[validation_train], y[validation_train, column] - train_base
            )
            prediction = validation_model.predict(x[validation_test]) + test_base
            truth = y[validation_test, column]
            return truth, prediction, test_base

        for column, name in enumerate(TARGET_NAMES):
            available = np.flatnonzero(np.isfinite(y[:, column]))

            def _validation_split(repeat: int) -> tuple[np.ndarray, np.ndarray]:
                if column < PRIMARY_TARGET_COUNT:
                    train_indices, test_indices = primary_splits[repeat]
                    return (
                        train_indices[np.isfinite(y[train_indices, column])],
                        test_indices[np.isfinite(y[test_indices, column])],
                    )
                if len(available) >= 8:
                    shuffled = available.copy()
                    # Repeat 0 consumes the shared stream exactly as the
                    # historical single-split code did; later repeats use
                    # dedicated deterministic streams (seed + 9973*r).
                    repeat_rng = (
                        rng
                        if repeat == 0
                        else np.random.default_rng(
                            self.random_seed + 4241 + 9973 * repeat + 101 * column
                        )
                    )
                    repeat_rng.shuffle(shuffled)
                    test_count = max(1, int(round(len(shuffled) * fraction)))
                    return shuffled[test_count:], shuffled[:test_count]
                empty = np.asarray([], dtype=int)
                return empty, empty

            skill_values: list[float] = []
            for repeat in range(repeats):
                validation_train, validation_test = _validation_split(repeat)
                outcome = _holdout_evaluation(column, validation_train, validation_test)
                if outcome is None:
                    skill_values.append(float("nan"))
                    if repeat == 0:
                        metrics[name] = {
                            "mae": float("nan"),
                            "rmse": float("nan"),
                            "r2": float("nan"),
                            "spearman": float("nan"),
                            "case_mean_baseline_mae": float("nan"),
                            "holdout_count": 0.0,
                        }
                    continue
                truth, prediction, test_base = outcome
                repeat_mae = float(mean_absolute_error(truth, prediction))
                repeat_baseline = float(mean_absolute_error(truth, test_base))
                skill_values.append(
                    1.0 - repeat_mae / repeat_baseline
                    if repeat_baseline > 0.0
                    else float("nan")
                )
                if repeat != 0:
                    continue
                residual_rmse[column] = float(
                    math.sqrt(mean_squared_error(truth, prediction))
                )
                holdout[name] = {
                    "truth": [float(value) for value in truth],
                    "prediction": [float(value) for value in prediction],
                    "case": [records[int(index)].case.label for index in validation_test],
                    # Homoscedastic v1: the calibration constant applied at
                    # predict time, retained per point for coverage checks.
                    "calibrated_std": [float(residual_rmse[column])] * len(truth),
                }
                metrics[name] = {
                    "mae": repeat_mae,
                    "rmse": residual_rmse[column],
                    "r2": (
                        float(r2_score(truth, prediction))
                        if len(validation_test) >= 2
                        else float("nan")
                    ),
                    "spearman": _spearman(truth, prediction),
                    "case_mean_baseline_mae": repeat_baseline,
                    "holdout_count": float(len(validation_test)),
                }
            finite_skills = [value for value in skill_values if math.isfinite(value)]
            metrics[name].update(
                {
                    "skill_values": [float(value) for value in skill_values],
                    "skill_median": (
                        float(np.median(finite_skills)) if finite_skills else float("nan")
                    ),
                    "skill_q25": (
                        float(np.percentile(finite_skills, 25.0))
                        if finite_skills
                        else float("nan")
                    ),
                    "skill_min": (
                        float(np.min(finite_skills)) if finite_skills else float("nan")
                    ),
                }
            )

            if len(available) >= 2:
                full_base = np.asarray(
                    [self.case_target_mean[records[int(index)].case][column] for index in available]
                )
                model = self._new_model(target_index=column)
                model.fit(x[available], y[available, column] - full_base)
                fitted_models.append(model)
            else:
                fitted_models.append(None)
        self.residual_rmse = residual_rmse
        self.validation_metrics = metrics
        self.holdout_predictions = holdout
        self.models = tuple(fitted_models)
        self.target_known = tuple(model is not None for model in fitted_models)
        # Convergence classifier: fitted on ALL records (censored rows
        # included) whenever both classes are present.
        if bool(nonconverged.any()) and not bool(nonconverged.all()):
            classifier = ExtraTreesClassifier(
                n_estimators=128,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=False,
                random_state=self.random_seed + 1009 * len(TARGET_NAMES),
                n_jobs=self.n_jobs,
            )
            classifier.fit(x, ~nonconverged)
            self.convergence_model = classifier
        else:
            self.convergence_model = None
        self.training_size = len(records)
        return self

    def _require_fitted(
        self,
    ) -> tuple[
        FeatureEncoder,
        tuple[ExtraTreesRegressor | None, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[CaseKey, np.ndarray],
    ]:
        if (
            self.encoder is None
            or self.models is None
            or self.target_mean is None
            or self.target_scale is None
            or self.residual_rmse is None
            or self.case_target_mean is None
        ):
            raise RuntimeError("surrogate has not been fitted")
        return (
            self.encoder,
            self.models,
            self.target_mean,
            self.target_scale,
            self.residual_rmse,
            self.case_target_mean,
        )

    def predict(
        self,
        patterns: Sequence[Pattern],
        cases: Sequence[CaseKey],
        cells: Sequence[float],
    ) -> SurrogatePrediction:
        (
            encoder,
            models,
            target_mean,
            target_scale,
            residual_rmse,
            case_target_mean,
        ) = self._require_fitted()
        if not (len(patterns) == len(cases) == len(cells)):
            raise ValueError("patterns, cases and cells must have equal length")
        count = len(patterns)
        if count == 0:
            empty = np.zeros((0, len(TARGET_NAMES)), dtype=float)
            return SurrogatePrediction(empty, empty.copy(), empty.copy())

        cache = self._prediction_row_cache()
        keys = [
            (pattern.canonical(), case, cell)
            for pattern, case, cell in zip(patterns, cases, cells, strict=True)
        ]
        mean_rows: list[np.ndarray | None] = [None] * count
        epistemic_rows: list[np.ndarray | None] = [None] * count
        calibrated_rows: list[np.ndarray | None] = [None] * count
        pending: list[int] = []
        for row, key in enumerate(keys):
            cached = cache.get(key)
            if cached is None:
                pending.append(row)
            else:
                mean_rows[row], epistemic_rows[row], calibrated_rows[row] = cached

        if pending:
            x = encoder.transform(
                [patterns[i] for i in pending],
                [cases[i] for i in pending],
                [cells[i] for i in pending],
            )
            stride = self._std_tree_stride()
            # sklearn's per-tree Python wrappers (input validation + config
            # propagation) dominate on the small batches the PPO env sends
            # every step; those go straight to the Cython predictors.  Bulk
            # sweeps keep the forest's parallel path.
            sequential = len(pending) < _PARALLEL_PREDICT_MIN_ROWS
            x32 = np.ascontiguousarray(x, dtype=np.float32)
            mean_columns: list[np.ndarray] = []
            std_columns: list[np.ndarray] = []
            for model in models:
                if model is None:
                    # No labels for this target: the mean stays at the case
                    # baseline and the NaN std is the explicit "unknown"
                    # marker (never zero-certainty).
                    mean_columns.append(np.zeros(len(pending), dtype=float))
                    std_columns.append(np.full(len(pending), np.nan))
                    continue
                estimators = model.estimators_
                if sequential:
                    outputs = np.empty(
                        (len(estimators), x32.shape[0]), dtype=np.float64
                    )
                    try:
                        for index, tree in enumerate(estimators):
                            outputs[index] = tree.tree_.predict(x32)[:, 0]
                    except AttributeError:
                        for index, tree in enumerate(estimators):
                            outputs[index] = tree.predict(x, check_input=False)
                    # Exact all-tree mean (fixed summation order, deterministic).
                    mean_columns.append(outputs.mean(axis=0))
                    subset_outputs = outputs[::stride]
                    if subset_outputs.shape[0] < _MIN_STD_TREES:
                        subset_outputs = outputs
                    std_columns.append(subset_outputs.std(axis=0, ddof=0))
                    continue
                # Mean over the whole forest is a single C-parallel call and is
                # statistically identical to averaging the per-tree outputs.
                mean_columns.append(model.predict(x))
                subset = estimators[::stride]
                if len(subset) < _MIN_STD_TREES:
                    subset = estimators
                tree_predictions = np.stack([tree.predict(x) for tree in subset], axis=0)
                std_columns.append(tree_predictions.std(axis=0, ddof=0))
            residual_mean = np.column_stack(mean_columns)
            baseline = np.stack([case_target_mean[cases[i]] for i in pending])
            computed_mean = residual_mean + baseline
            computed_epistemic = np.column_stack(std_columns)
            computed_calibrated = np.sqrt(
                computed_epistemic**2 + residual_rmse[None, :] ** 2
            )
            for offset, row in enumerate(pending):
                mean_row = computed_mean[offset]
                epistemic_row = computed_epistemic[offset]
                calibrated_row = computed_calibrated[offset]
                mean_rows[row] = mean_row
                epistemic_rows[row] = epistemic_row
                calibrated_rows[row] = calibrated_row
                if len(cache) >= _PREDICTION_CACHE_CAP:
                    cache.clear()
                cache[keys[row]] = (mean_row, epistemic_row, calibrated_row)

        # ``np.stack`` allocates a fresh contiguous array, so the returned
        # prediction never aliases the cached rows (callers may mutate it).
        mean = np.stack(mean_rows, axis=0)
        epistemic_std = np.stack(epistemic_rows, axis=0)
        calibrated_std = np.stack(calibrated_rows, axis=0)
        return SurrogatePrediction(mean, epistemic_std, calibrated_std)

    def predict_one(self, pattern: Pattern, case: CaseKey, cell: float) -> SurrogatePrediction:
        return self.predict([pattern], [case], [cell])

    def predict_convergence_probability(
        self,
        patterns: Sequence[Pattern],
        cases: Sequence[CaseKey],
        cells: Sequence[float],
    ) -> np.ndarray:
        """P(equilibrium converges) per pattern; 1.0 without a classifier."""

        encoder, *_ = self._require_fitted()
        model = self.convergence_model
        if model is None:
            return np.ones(len(patterns), dtype=float)
        x = encoder.transform(list(patterns), list(cases), list(cells))
        column = int(np.flatnonzero(model.classes_ == True)[0])  # noqa: E712
        return np.asarray(model.predict_proba(x)[:, column], dtype=float)

    def quality_report(
        self,
        *,
        constraints: "ConstraintConfig | None" = None,
        min_labels_primary: int = 16,
        min_labels_constraint: int = 8,
        min_holdout: int = 3,
        min_skill_halt: float = 0.0,
        min_skill_objective: float | None = None,
        coverage_band: tuple[float, float] = (0.45, 0.95),
        coverage_min_holdout: int = 8,
    ) -> SurrogateQualityReport:
        """Judge each target by skill vs the case-mean baseline, then gate.

        Two distinct thresholds (F-07): ``halt`` requires PROVEN incompetence
        of F_r or cyclen (trainable + evaluable yet skill <=
        ``min_skill_halt``); promotion to ``objective`` requires skill >
        ``min_skill_objective`` on every required axis.  Every data-shortage
        failure — and any failing ENABLED constraint axis (AO always; burnups
        when their limit is active) — maps to ``explore`` because more MASTER
        data is the remedy; a constraint axis can never halt the search (its
        labels only grow through MASTER waves).

        Invariant (pinned by tests): with required = the four primary targets
        plus every enabled constraint axis,
        ``gate_passed == (mode == "objective") == all(required passed)``.
        """

        if self.label_counts is None or not self.validation_metrics:
            raise RuntimeError("surrogate has not been fitted")
        if constraints is None:
            from .reward import ConstraintConfig

            constraints = ConstraintConfig()
        objective_threshold = (
            float(getattr(self, "gate_min_skill", 0.10))
            if min_skill_objective is None
            else float(min_skill_objective)
        )
        per_target: dict[str, TargetQuality] = {}
        reasons: list[str] = []
        halt = False
        explore = False
        constraint_axis_enabled = {
            "AO_abs": True,
            "max_assembly_burnup": bool(constraints.assembly_burnup_enabled),
            "max_pin_burnup": bool(constraints.pin_burnup_enabled),
        }
        for column, name in enumerate(TARGET_NAMES):
            primary = column < PRIMARY_TARGET_COUNT
            label_count = int(self.label_counts[column])
            stats = self.validation_metrics.get(name, {})
            holdout_count = int(stats.get("holdout_count", 0.0))
            mae = float(stats.get("mae", float("nan")))
            baseline_mae = float(stats.get("case_mean_baseline_mae", float("nan")))
            r2 = float(stats.get("r2", float("nan")))
            skill_single = (
                1.0 - mae / baseline_mae
                if math.isfinite(baseline_mae) and baseline_mae > 0.0
                else float("nan")
            )
            # Judged statistic: the repeated-holdout median when available;
            # single-split fits (legacy models, stubs) fall back to their one
            # skill estimate for both statistics.
            skill = float(stats.get("skill_median", skill_single))
            skill_q25 = float(stats.get("skill_q25", skill))
            coverage = float("nan")
            holdout = self.holdout_predictions.get(name)
            if holdout_count >= coverage_min_holdout and holdout is not None:
                sigma = holdout.get("calibrated_std")
                if sigma is not None:
                    truth = np.asarray(holdout["truth"], dtype=float)
                    predicted = np.asarray(holdout["prediction"], dtype=float)
                    coverage = float(
                        np.mean(
                            np.abs(truth - predicted)
                            <= np.asarray(sigma, dtype=float)
                        )
                    )
            minimum_labels = min_labels_primary if primary else min_labels_constraint
            trainable = label_count >= minimum_labels
            evaluable = holdout_count >= min_holdout
            skillful = math.isfinite(skill) and skill > objective_threshold
            robust = math.isfinite(skill_q25) and skill_q25 > min_skill_halt
            covered = holdout_count < coverage_min_holdout or (
                math.isfinite(coverage)
                and coverage_band[0] <= coverage <= coverage_band[1]
            )
            passed = trainable and evaluable and skillful and robust and covered
            per_target[name] = TargetQuality(
                name=name,
                label_count=label_count,
                holdout_count=holdout_count,
                r2=r2,
                mae=mae,
                baseline_mae=baseline_mae,
                skill=skill,
                coverage_1s=coverage,
                trainable=trainable,
                passed=passed,
            )
            failing: list[str] = []
            if not trainable:
                failing.append(f"label_count {label_count} < {minimum_labels}")
            if not evaluable:
                failing.append(f"holdout_count {holdout_count} < {min_holdout}")
            if not skillful:
                failing.append(f"skill {skill:.3f} <= {objective_threshold:.3f}")
            if skillful and not robust:
                failing.append(f"skill_q25 {skill_q25:.3f} <= {min_skill_halt:.3f}")
            if not covered:
                failing.append(
                    f"coverage_1s {coverage:.3f} outside "
                    f"[{coverage_band[0]:.2f}, {coverage_band[1]:.2f}]"
                )
            proven_incompetent = (
                trainable
                and evaluable
                and math.isfinite(skill)
                and skill <= min_skill_halt
            )
            if name in ("F_r", "cyclen"):
                if proven_incompetent:
                    halt = True
                    reasons.append(f"{name}: proven no-skill ({'; '.join(failing)})")
                elif not passed:
                    explore = True
                    reasons.append(f"{name}: unproven quality ({'; '.join(failing)})")
            elif primary:
                if not passed:
                    explore = True
                    reasons.append(f"{name}: {'; '.join(failing)}")
            elif constraint_axis_enabled.get(name, False) and not passed:
                # An enabled constraint axis that fails the gate (missing
                # labels OR trainable-yet-unskillful) demotes exploitation to
                # exploration; it never halts (F-04).
                explore = True
                reasons.append(
                    f"{name}: enabled constraint axis failed quality gate "
                    f"({'; '.join(failing)})"
                )
        mode = "halt" if halt else ("explore" if explore else "objective")
        thresholds: dict[str, Any] = {
            "min_labels_primary": int(min_labels_primary),
            "min_labels_constraint": int(min_labels_constraint),
            "min_holdout": int(min_holdout),
            "min_skill_halt": float(min_skill_halt),
            "min_skill_objective": objective_threshold,
            "coverage_band": [float(coverage_band[0]), float(coverage_band[1])],
            "coverage_min_holdout": int(coverage_min_holdout),
            "holdout_repeats": int(getattr(self, "gate_holdout_repeats", 5)),
        }
        return SurrogateQualityReport(
            gate_passed=(mode == "objective"),
            mode=mode,
            per_target=per_target,
            reasons=reasons,
            thresholds=thresholds,
        )

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateEnsemble":
        value = joblib.load(Path(path))
        if not isinstance(value, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        _, models, target_mean, target_scale, residual_rmse, _ = value._require_fitted()
        expected = len(TARGET_NAMES)
        if not (
            len(models) == expected
            and target_mean.shape == (expected,)
            and target_scale.shape == (expected,)
            and residual_rmse.shape == (expected,)
        ):
            raise ValueError(
                f"{path} uses an obsolete surrogate target schema; retraining is required"
            )
        if (
            not hasattr(value, "convergence_model")
            or getattr(value, "label_counts", None) is None
        ):
            value._migrate_legacy_schema()
        return value

    def _migrate_legacy_schema(self) -> None:
        """Backfill attributes missing from pre-censoring pickles.

        Legacy models trained before the censoring/quality-gate schema keep
        their regression heads but carry no convergence classifier and no
        per-target label counts.  The backfill is conservative: unknown
        label counts read as zero, so ``quality_report`` can only judge such
        a model ``explore`` (never ``objective``/``halt`` on unproven data),
        and the absent classifier means P(converge) = 1.0 exactly as the
        original schema behaved.
        """

        self.convergence_labels = getattr(self, "convergence_labels", None)
        self.convergence_model = getattr(self, "convergence_model", None)
        self.validation_fraction = float(getattr(self, "validation_fraction", 0.2))
        self.gate_min_skill = float(getattr(self, "gate_min_skill", 0.10))
        self.gate_holdout_repeats = int(getattr(self, "gate_holdout_repeats", 5))
        if getattr(self, "label_counts", None) is None:
            self.label_counts = np.zeros(len(TARGET_NAMES), dtype=int)
        if getattr(self, "target_known", None) is None:
            self.target_known = tuple(model is not None for model in self.models)
        self.manifest_sha256 = getattr(self, "manifest_sha256", None)
        self.package_root = getattr(self, "package_root", None)
        # Coverage checks expect the homoscedastic per-point sigma column.
        for column, name in enumerate(TARGET_NAMES):
            holdout = self.holdout_predictions.get(name)
            if holdout is not None and "calibrated_std" not in holdout:
                holdout["calibrated_std"] = [
                    float(self.residual_rmse[column])
                ] * len(holdout.get("truth", []))
        self.legacy_schema = True

    def report(self) -> dict[str, object]:
        (
            _,
            models,
            target_mean,
            target_scale,
            residual_rmse,
            case_target_mean,
        ) = self._require_fitted()
        return {
            "training_size": self.training_size,
            "manifest_sha256": getattr(self, "manifest_sha256", None),
            "package_root": getattr(self, "package_root", None),
            "n_estimators_per_target": [
                0 if model is None else len(model.estimators_) for model in models
            ],
            "target_names": list(TARGET_NAMES),
            "target_mean": target_mean.tolist(),
            "target_scale": target_scale.tolist(),
            "calibration_rmse": residual_rmse.tolist(),
            "case_target_mean": {
                case.label: values.tolist() for case, values in sorted(case_target_mean.items())
            },
            "validation": self.validation_metrics,
        }

    def write_report(self, path: str | Path) -> Path:
        from .jsonio import dumps_strict

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dumps_strict(self.report()), encoding="utf-8")
        return target
