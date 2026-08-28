"""Constraint-aware multi-objective reward and acquisition functions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

import numpy as np

from .domain import CaseKey, FOM, PatternRecord
from .surrogate import SurrogatePrediction


# A deliberately huge test/default limit disables the corresponding burnup
# gate.  Real runs activate it by supplying an engineering limit below this
# sentinel; a missing PPI/SUM value then becomes a constraint violation.
DISABLED_BURNUP_LIMIT = 1.0e8


# Objective-mode vocabulary: "trade_off" is the historical EFPD↔CBC
# compromise; "target_cycle" pins the verified cycle length to a fixed target
# window and minimizes Max CBC without compromise inside that window.
OBJECTIVE_MODES = ("trade_off", "target_cycle")


@dataclass(frozen=True)
class ConstraintConfig:
    """Physics limits, objective weights, and smooth penalty scales."""

    f_r_limit: float = 1.55
    cbc_limit: float = 1550.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30
    max_assembly_burnup_limit: float = 1.0e9
    max_pin_burnup_limit: float = 1.0e9
    # Explicit screening-only opt-out: a production deck (MASTER active mode)
    # may leave the burnup gates at the disable sentinel only when this is
    # true; the run-level licensing status is then capped at ``pending``.
    allow_disabled_burnup_gate: bool = False

    f_r_width: float = 0.01
    cbc_width: float = 25.0
    f_q_width: float = 0.05
    ao_width: float = 0.02
    assembly_burnup_width: float = 1.0
    pin_burnup_width: float = 1.0

    cycle_weight: float = 0.50
    f_r_objective_weight: float = 0.30
    cbc_objective_weight: float = 0.20
    risk_z: float = 0.25
    penalty_weight: float = 4.0
    feasible_bonus: float = 0.25
    # Probability credited to a constraint axis the surrogate cannot predict
    # at all (NaN std).  0.5 keeps "unknown" strictly worse than "predicted
    # safe" without the old certainty-of-safety fabrication.
    unknown_axis_probability: float = 0.5

    # Objective mode (W3): "trade_off" (default, historical behavior) or
    # "target_cycle" (verified cyclen pinned to cycle_target_efpd within
    # ±cycle_tolerance_efpd; Max CBC minimized inside the window).  The window
    # judges the MASTER-verified equilibrium cyclen, never the manifest value.
    objective_mode: str = "trade_off"
    cycle_target_efpd: float | None = None
    cycle_tolerance_efpd: float = 2.0

    @property
    def is_target_mode(self) -> bool:
        # getattr: legacy pickled/stubbed instances predate objective_mode.
        return getattr(self, "objective_mode", "trade_off") == "target_cycle"

    def objective_weights(self) -> np.ndarray:
        mode = getattr(self, "objective_mode", "trade_off")
        if mode not in OBJECTIVE_MODES:
            raise ValueError(
                f"unknown objective mode {mode!r}; "
                f"valid modes: {', '.join(OBJECTIVE_MODES)}"
            )
        if self.is_target_mode:
            if self.cycle_target_efpd is None or not math.isfinite(
                self.cycle_target_efpd
            ) or self.cycle_target_efpd <= 0.0:
                raise ValueError(
                    "target_cycle mode requires a finite positive cycle_target_efpd"
                )
            if not math.isfinite(self.cycle_tolerance_efpd) or (
                self.cycle_tolerance_efpd <= 0.0
            ):
                raise ValueError(
                    "target_cycle mode requires a positive cycle_tolerance_efpd"
                )
        weights = np.asarray(
            [self.cycle_weight, self.f_r_objective_weight, self.cbc_objective_weight],
            dtype=float,
        )
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("multi-objective weights must be non-negative with positive sum")
        return weights / weights.sum()

    @property
    def assembly_burnup_enabled(self) -> bool:
        return bool(
            math.isfinite(self.max_assembly_burnup_limit)
            and self.max_assembly_burnup_limit < DISABLED_BURNUP_LIMIT
        )

    @property
    def pin_burnup_enabled(self) -> bool:
        return bool(
            math.isfinite(self.max_pin_burnup_limit)
            and self.max_pin_burnup_limit < DISABLED_BURNUP_LIMIT
        )


def is_fom_feasible(fom: FOM, constraints: ConstraintConfig) -> bool:
    """Return the authoritative physical feasibility decision for one FOM."""

    if not (
        fom.converged
        and fom.f_r <= constraints.f_r_limit
        and fom.cbc_max <= constraints.cbc_limit
        and fom.f_q <= constraints.f_q_limit
    ):
        return False
    if fom.ao_abs is None or fom.ao_abs > constraints.ao_abs_limit:
        return False
    if constraints.assembly_burnup_enabled:
        if (
            fom.max_assembly_burnup is None
            or fom.max_assembly_burnup > constraints.max_assembly_burnup_limit
        ):
            return False
    if constraints.pin_burnup_enabled:
        if (
            fom.max_pin_burnup is None
            or fom.max_pin_burnup > constraints.max_pin_burnup_limit
        ):
            return False
    return True


def cycle_target_distance(cyclen: float, constraints: ConstraintConfig) -> float:
    """Continuous distance |cyclen − target| the target mode optimizes on."""

    if constraints.cycle_target_efpd is None:
        raise ValueError(
            "cycle_target_distance requires constraints.cycle_target_efpd"
        )
    return abs(float(cyclen) - float(constraints.cycle_target_efpd))


def is_cycle_on_target(fom: FOM, constraints: ConstraintConfig) -> bool:
    """Window membership: converged AND |cyclen − target| ≤ tolerance.

    Judged on the verified equilibrium cyclen only — orthogonal to (and never
    combined into) the licensing verdict.
    """

    return bool(
        fom.converged
        and cycle_target_distance(fom.cyclen, constraints)
        <= constraints.cycle_tolerance_efpd
    )


@dataclass(frozen=True)
class CaseRewardScale:
    reference_cycle: float
    cycle_scale: float
    reference_f_r: float
    f_r_scale: float
    reference_cbc: float
    cbc_scale: float


@dataclass(frozen=True)
class ScoreBatch:
    total: np.ndarray
    objective: np.ndarray
    penalty: np.ndarray
    feasible: np.ndarray
    conservative: np.ndarray
    # All columns are maximized: cycle gain, F_r reduction, CBC reduction.
    objectives: np.ndarray


class RewardModel:
    """Score predictions using constrained Cycle/F_r/CBC objectives."""

    def __init__(
        self,
        scales: Mapping[CaseKey, CaseRewardScale],
        constraints: ConstraintConfig | None = None,
    ) -> None:
        self.scales = dict(scales)
        self.constraints = constraints or ConstraintConfig()
        self.constraints.objective_weights()  # validate once

    @staticmethod
    def _spread(values: np.ndarray, minimum: float) -> float:
        return max(
            minimum,
            float(values.std()),
            float(values.max() - values.min()) / 4.0,
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[PatternRecord],
        constraints: ConstraintConfig | None = None,
    ) -> "RewardModel":
        config = constraints or ConstraintConfig()
        grouped: dict[CaseKey, list[PatternRecord]] = {}
        for record in records:
            grouped.setdefault(record.case, []).append(record)
        scales: dict[CaseKey, CaseRewardScale] = {}
        for case, case_records in grouped.items():
            # Non-converged last-iterate values must not define the reward
            # scale/reference statistics (their inflated cyclen would poison
            # reference_cycle); fall back only if nothing converged.
            converged = [record for record in case_records if record.fom.converged]
            case_records = converged or case_records
            feasible = [record for record in case_records if is_fom_feasible(record.fom, config)]
            reference_pool = feasible or case_records
            all_cycles = np.asarray([record.fom.cyclen for record in case_records], dtype=float)
            all_f_r = np.asarray([record.fom.f_r for record in case_records], dtype=float)
            all_cbc = np.asarray([record.fom.cbc_max for record in case_records], dtype=float)
            if config.is_target_mode:
                # Target mode: distances to the target define both the
                # reference and the normalization scale, so a basin far from
                # the target still yields a well-conditioned pull.
                reference_cycle = float(config.cycle_target_efpd)
                cycle_scale = cls._spread(
                    np.abs(all_cycles - reference_cycle), 1.0
                )
            else:
                reference_cycle = max(
                    record.fom.cyclen for record in reference_pool
                )
                cycle_scale = cls._spread(all_cycles, 1.0)
            scales[case] = CaseRewardScale(
                reference_cycle=reference_cycle,
                cycle_scale=cycle_scale,
                reference_f_r=float(np.median([record.fom.f_r for record in reference_pool])),
                f_r_scale=cls._spread(all_f_r, 0.01),
                reference_cbc=float(
                    np.median([record.fom.cbc_max for record in reference_pool])
                ),
                cbc_scale=cls._spread(all_cbc, 25.0),
            )
        return cls(scales, config)

    def score(
        self,
        prediction: SurrogatePrediction,
        cases: Sequence[CaseKey],
        *,
        stage: Literal["feasibility", "objective"] = "objective",
        calibrated: bool = False,
    ) -> ScoreBatch:
        if prediction.mean.shape[0] != len(cases):
            raise ValueError("prediction row count and case count differ")
        config = self.constraints
        std = prediction.calibrated_std if calibrated else prediction.epistemic_std
        # NaN std marks an axis the surrogate cannot predict at all: no
        # conservative shift, no penalty, and — crucially — no feasibility
        # credit is derived from it.
        known = np.isfinite(std)
        shift = np.where(known, std, 0.0)
        conservative = prediction.mean.copy()
        # Lower-is-better quantities use an upper confidence bound; cycle
        # length uses a lower confidence bound.
        conservative[:, 0:3] += config.risk_z * shift[:, 0:3]
        conservative[:, 3] -= config.risk_z * shift[:, 3]
        conservative[:, 4:7] += config.risk_z * shift[:, 4:7]

        excess_columns = [
            (0, config.f_r_limit, config.f_r_width),
            (1, config.cbc_limit, config.cbc_width),
            (2, config.f_q_limit, config.f_q_width),
            (4, config.ao_abs_limit, config.ao_width),
        ]
        if config.assembly_burnup_enabled:
            excess_columns.append(
                (5, config.max_assembly_burnup_limit, config.assembly_burnup_width)
            )
        if config.pin_burnup_enabled:
            excess_columns.append(
                (6, config.max_pin_burnup_limit, config.pin_burnup_width)
            )
        excesses = [
            np.where(
                known[:, column],
                np.maximum(0.0, conservative[:, column] - limit) / width,
                0.0,
            )
            for column, limit, width in excess_columns
        ]
        penalty = np.sum(np.square(np.stack(excesses, axis=1)), axis=1)
        # An unknown enforced axis can never certify feasibility.
        required_known = known[:, [column for column, _, _ in excess_columns]]
        feasible = (penalty <= 1.0e-12) & required_known.all(axis=1)

        if config.is_target_mode:
            # Distance-based pull (never a window probability): the cycle
            # column is −max(0, d_UCB − tol)/scale with d_UCB the
            # uncertainty-inflated distance, so it is exactly 0 inside the
            # window (CBC/F_r weights then dominate automatically) and grows
            # linearly, symmetrically, outside it.
            target = float(config.cycle_target_efpd)
            tolerance = float(config.cycle_tolerance_efpd)
            d_ucb = (
                np.abs(prediction.mean[:, 3] - target)
                + config.risk_z * shift[:, 3]
            )
            cycle_terms = -np.maximum(0.0, d_ucb - tolerance)
            objectives = np.asarray(
                [
                    (
                        cycle_terms[row] / self.scales[case].cycle_scale,
                        (self.scales[case].reference_f_r - conservative[row, 0])
                        / self.scales[case].f_r_scale,
                        (self.scales[case].reference_cbc - conservative[row, 1])
                        / self.scales[case].cbc_scale,
                    )
                    for row, case in enumerate(cases)
                ],
                dtype=float,
            )
        else:
            objectives = np.asarray(
                [
                    (
                        (conservative[row, 3] - self.scales[case].reference_cycle)
                        / self.scales[case].cycle_scale,
                        (self.scales[case].reference_f_r - conservative[row, 0])
                        / self.scales[case].f_r_scale,
                        (self.scales[case].reference_cbc - conservative[row, 1])
                        / self.scales[case].cbc_scale,
                    )
                    for row, case in enumerate(cases)
                ],
                dtype=float,
            )
        objective = objectives @ config.objective_weights()
        if stage == "feasibility":
            total = -penalty
        elif stage == "objective":
            total = (
                objective
                - config.penalty_weight * penalty
                + config.feasible_bonus * feasible.astype(float)
            )
        else:
            raise ValueError(f"unknown reward stage: {stage}")
        return ScoreBatch(total, objective, penalty, feasible, conservative, objectives)

    @staticmethod
    def _normal_upper_probability(
        mean: np.ndarray,
        std: np.ndarray,
        limit: float,
        *,
        unknown_probability: float = 0.5,
    ) -> np.ndarray:
        result = np.empty_like(mean, dtype=float)
        for index, (mu, sigma) in enumerate(zip(mean, std, strict=True)):
            if not math.isfinite(sigma):
                # No model for this axis: neither safe nor unsafe.
                result[index] = unknown_probability
            elif sigma <= 1.0e-12:
                result[index] = float(mu <= limit)
            else:
                z = (limit - mu) / (sigma * math.sqrt(2.0))
                result[index] = 0.5 * (1.0 + math.erf(z))
        return result

    def feasible_probability(self, prediction: SurrogatePrediction) -> np.ndarray:
        config = self.constraints
        mean = prediction.mean
        std = prediction.calibrated_std
        unknown = config.unknown_axis_probability
        probability = (
            self._normal_upper_probability(
                mean[:, 0], std[:, 0], config.f_r_limit, unknown_probability=unknown
            )
            * self._normal_upper_probability(
                mean[:, 1], std[:, 1], config.cbc_limit, unknown_probability=unknown
            )
            * self._normal_upper_probability(
                mean[:, 2], std[:, 2], config.f_q_limit, unknown_probability=unknown
            )
            * self._normal_upper_probability(
                mean[:, 4], std[:, 4], config.ao_abs_limit, unknown_probability=unknown
            )
        )
        if config.assembly_burnup_enabled:
            probability *= self._normal_upper_probability(
                mean[:, 5],
                std[:, 5],
                config.max_assembly_burnup_limit,
                unknown_probability=unknown,
            )
        if config.pin_burnup_enabled:
            probability *= self._normal_upper_probability(
                mean[:, 6],
                std[:, 6],
                config.max_pin_burnup_limit,
                unknown_probability=unknown,
            )
        return probability

    def cycle_window_probability(
        self, prediction: SurrogatePrediction
    ) -> np.ndarray:
        """Two-sided Φ probability of the cyclen window — DIAGNOSTIC ONLY.

        This value is recorded for auditing and must NEVER be multiplied into
        the acquisition or score: with the target ~26–34 EFPD from the basin
        and σ ≈ 1.4 it is exactly 0 in float64, which would starve the entire
        search (design principle 1).
        """

        config = self.constraints
        if config.cycle_target_efpd is None:
            raise ValueError(
                "cycle_window_probability requires constraints.cycle_target_efpd"
            )
        target = float(config.cycle_target_efpd)
        tolerance = float(config.cycle_tolerance_efpd)
        lower, upper = target - tolerance, target + tolerance
        mean = prediction.mean[:, 3]
        std = prediction.calibrated_std[:, 3]
        result = np.empty_like(mean, dtype=float)
        for index, (mu, sigma) in enumerate(zip(mean, std, strict=True)):
            if not math.isfinite(sigma):
                result[index] = config.unknown_axis_probability
            elif sigma <= 1.0e-12:
                result[index] = float(lower <= mu <= upper)
            else:
                scale = sigma * math.sqrt(2.0)
                result[index] = 0.5 * (
                    math.erf((upper - mu) / scale) - math.erf((lower - mu) / scale)
                )
        return result

    def acquisition(
        self,
        prediction: SurrogatePrediction,
        cases: Sequence[CaseKey],
        *,
        exploration_weight: float = 0.15,
        incumbent_cycles: Sequence[float] | None = None,
        incumbent_cycle_distances: Sequence[float] | None = None,
        convergence_probability: np.ndarray | None = None,
        constraint_uncertainty_weight: float = 0.5,
    ) -> np.ndarray:
        """Constrained multi-objective expected gain used for MASTER selection."""

        probability = self.feasible_probability(prediction)
        if convergence_probability is not None:
            convergence = np.asarray(convergence_probability, dtype=float)
            if convergence.shape != probability.shape:
                raise ValueError(
                    "convergence_probability and prediction rows must have equal length"
                )
            probability = probability * convergence
        if incumbent_cycles is not None and len(incumbent_cycles) != len(cases):
            raise ValueError("incumbent_cycles and cases must have equal length")
        if incumbent_cycle_distances is not None and len(
            incumbent_cycle_distances
        ) != len(cases):
            raise ValueError(
                "incumbent_cycle_distances and cases must have equal length"
            )
        score = self.score(prediction, cases, calibrated=True)
        gain = np.maximum(0.0, score.objective)
        # A primary target with no fitted model (non-converged censoring can
        # leave < 2 usable labels) carries NaN std; like the constraint-axis
        # nansum below it contributes zero exploration instead of turning
        # every candidate's acquisition into NaN.
        primary_std = np.nan_to_num(prediction.epistemic_std, nan=0.0)
        uncertainty = np.asarray(
            [
                np.dot(
                    self.constraints.objective_weights(),
                    (
                        primary_std[row, 3] / self.scales[case].cycle_scale,
                        primary_std[row, 0] / self.scales[case].f_r_scale,
                        primary_std[row, 1] / self.scales[case].cbc_scale,
                    ),
                )
                for row, case in enumerate(cases)
            ]
        )
        # Constraint axes carry their own reducible uncertainty (F_q, AO,
        # enabled burnups); without this term the acquisition never explores
        # the very axes the feasibility gate depends on.  NaN std (unknown
        # axis, no model yet) contributes nothing here — exploration on an
        # axis without any labels cannot be directed by the surrogate.
        config = self.constraints
        constraint_columns = [
            (2, config.f_q_width),
            (4, config.ao_width),
        ]
        if config.assembly_burnup_enabled:
            constraint_columns.append((5, config.assembly_burnup_width))
        if config.pin_burnup_enabled:
            constraint_columns.append((6, config.pin_burnup_width))
        columns = [column for column, _ in constraint_columns]
        widths = np.asarray([width for _, width in constraint_columns], dtype=float)
        constraint_uncertainty = np.nansum(
            prediction.epistemic_std[:, columns] / widths[None, :], axis=1
        )
        uncertainty = uncertainty + constraint_uncertainty_weight * constraint_uncertainty
        # Preserve the former cycle-incumbent behavior as an additional gain,
        # while Fr/CBC improvements are supplied by the multi-objective utility.
        if incumbent_cycles is not None:
            cycle_gain = np.asarray(
                [
                    max(0.0, prediction.mean[row, 3] - float(incumbent_cycles[row]))
                    / self.scales[case].cycle_scale
                    for row, case in enumerate(cases)
                ]
            )
            gain = np.maximum(gain, self.constraints.objective_weights()[0] * cycle_gain)
        # Target mode (W3): expected distance reduction against the incumbent's
        # distance, using an optimistic (LCB) candidate distance.  The window
        # probability is deliberately NOT part of this product — the feasible
        # probability above covers only the safety axes, so far-from-target
        # candidates keep a positive, distance-monotone acquisition (no
        # starvation collapse when P(window) underflows to 0).
        if incumbent_cycle_distances is not None:
            if config.cycle_target_efpd is None:
                raise ValueError(
                    "incumbent_cycle_distances requires constraints.cycle_target_efpd"
                )
            target = float(config.cycle_target_efpd)
            calibrated = np.nan_to_num(prediction.calibrated_std[:, 3], nan=0.0)
            d_lcb = np.maximum(
                0.0,
                np.abs(prediction.mean[:, 3] - target)
                - config.risk_z * calibrated,
            )
            distance_gain = np.asarray(
                [
                    max(0.0, float(incumbent_cycle_distances[row]) - d_lcb[row])
                    / self.scales[case].cycle_scale
                    for row, case in enumerate(cases)
                ]
            )
            gain = np.maximum(
                gain, self.constraints.objective_weights()[0] * distance_gain
            )
        return probability * (gain + exploration_weight * uncertainty)
