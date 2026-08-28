"""Learned move-proposal policy (stage 2 of the pivot to an engineer-like optimizer).

Unlike :mod:`lpopt.model`, which predicts the FOMs of a *board*, this package
scores a *move*: given a parent loading pattern, a candidate edit, and the cell
context, how likely is the child to beat the parent on F_r and on node flatness?

v1 has exactly two heads.  The leakage-arbitration (cyclen / CBC) head is out of
scope by design — ``data/reports/policy_corpus_20260815.md`` section 4g shows the
two lineage eras disagree on the SIGN of the cycle-length response to outward
fresh loading, and nothing observational can settle it.
"""

from .data import (
    HELDOUT_CELL,
    HELDOUT_ERA_LIBRARIES,
    HELDOUT_LIBRARY,
    MOVE_CLASSES,
    PolicySteps,
    build_pattern_cache,
    build_splits,
    load_universe,
    scalar_features,
)
from .net import PolicyNet, PolicyNetConfig

__all__ = [
    "HELDOUT_CELL", "HELDOUT_ERA_LIBRARIES", "HELDOUT_LIBRARY", "MOVE_CLASSES",
    "PolicySteps", "PolicyNet", "PolicyNetConfig", "build_pattern_cache",
    "build_splits", "load_universe", "scalar_features",
]
