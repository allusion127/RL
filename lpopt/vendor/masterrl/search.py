"""Hand-written shim of the snapshot ``master_rl/search.py``.

This is NOT a byte-identical copy. The upstream ``search.py`` drags heavy
dependencies (``env``, ``ppo``, ``multiobjective``, ``cea_proxy`` — all outside
the pinned 11-file vendor set). Only the light evaluation-contract block is
needed here because ``parallel.py`` imports ``EvaluationResult`` and
``PatternEvaluator`` from this module.

The class bodies below are reproduced VERBATIM from the pinned snapshot
``search.py`` lines 34-89 (``EvaluationResult`` + ``PatternEvaluator`` Protocol +
``SurrogateEvaluator`` + ``EquilibriumEvaluator``), together with only the
imports that block requires. See ``VENDOR_MANIFEST.json`` for the source path and
hashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .dataset import CaseData
from .domain import FOM, Pattern
from .surrogate import SurrogateEnsemble


@dataclass(frozen=True)
class EvaluationResult:
    fom: FOM
    raw_master_calls: int = 0
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class PatternEvaluator(Protocol):
    def evaluate(self, case: CaseData, pattern: Pattern) -> EvaluationResult:
        """Perform one candidate-level evaluation."""


class SurrogateEvaluator:
    """Cheap evaluator for dry runs and tests; never a final verification."""

    def __init__(self, surrogate: SurrogateEnsemble) -> None:
        self.surrogate = surrogate

    def evaluate(self, case: CaseData, pattern: Pattern) -> EvaluationResult:
        prediction = self.surrogate.predict_one(pattern, case.key, case.cell)
        return EvaluationResult(
            fom=prediction.mean_fom(0),
            raw_master_calls=0,
            metadata={"mode": "surrogate", "uncertainty": prediction.row(0)},
        )


class EquilibriumEvaluator:
    """Adapter from :class:`EquilibriumRunner` to the active-search contract."""

    def __init__(self, equilibrium_runner: Any) -> None:
        self.runner = equilibrium_runner

    def evaluate(self, case: CaseData, pattern: Pattern) -> EvaluationResult:
        result = self.runner.run(case, pattern)
        return EvaluationResult(
            fom=replace(result.fom, converged=bool(result.converged)),
            raw_master_calls=int(result.master_process_calls),
            metadata={
                "mode": "equilibrium_master",
                "converged": bool(result.converged),
                "n_cycles": int(result.n_cycles),
                "comparisons": [
                    {
                        "previous_cycle": comparison.previous_cycle,
                        "current_cycle": comparison.current_cycle,
                        "deltas": dict(comparison.deltas),
                        "within_tolerance": comparison.within_tolerance,
                    }
                    for comparison in result.comparisons
                ],
                "tolerances": result.tolerances.as_dict(),
            },
        )
