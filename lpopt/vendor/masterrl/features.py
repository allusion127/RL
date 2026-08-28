"""Physics-aware, case-conditioned features for the surrogate and policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .domain import CaseKey, Pattern, SLOTS, X_INDEX


def _neighbor_pairs() -> tuple[tuple[int, int], ...]:
    by_coordinate = {(slot.row, slot.col): slot.index for slot in SLOTS}
    pairs: list[tuple[int, int]] = []
    for slot in SLOTS:
        for coordinate in ((slot.row + 1, slot.col), (slot.row, slot.col + 1)):
            neighbor = by_coordinate.get(coordinate)
            if neighbor is not None:
                pairs.append((slot.index, neighbor))
    return tuple(pairs)


NEIGHBOR_PAIRS = _neighbor_pairs()

# Upper bound on memoized encodings; cleared wholesale on overflow.
_FEATURE_CACHE_CAP = 65_536


@dataclass(frozen=True)
class FeatureEncoder:
    """Encode one pattern without relying on case-specific token vocabularies.

    Fresh batches are represented relative to ``pair`` (first/second) and
    shuffled assemblies by their source coordinates and rotation.  This lets
    the 502 patterns share statistical strength across enrichment/feed cases.
    """

    cases: tuple[CaseKey, ...]

    def __init__(self, cases: Iterable[CaseKey]) -> None:
        unique = tuple(sorted(set(cases)))
        if not unique:
            raise ValueError("FeatureEncoder requires at least one case")
        object.__setattr__(self, "cases", unique)
        object.__setattr__(self, "_case_index", {case: i for i, case in enumerate(unique)})
        # ``transform_one`` is pure in (pattern, case, cell); memoize its output.
        object.__setattr__(self, "_transform_cache", {})

    @property
    def per_slot_dim(self) -> int:
        return 11

    @property
    def aggregate_dim(self) -> int:
        return 19

    @property
    def output_dim(self) -> int:
        return len(SLOTS) * self.per_slot_dim + self.aggregate_dim + len(self.cases) + 2

    def _memo(self) -> dict[tuple[str, CaseKey, float], np.ndarray]:
        """Return the encoding memo, lazily creating it for legacy pickles."""

        cache = getattr(self, "_transform_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_transform_cache", cache)
        return cache

    def transform_one(self, pattern: Pattern, case: CaseKey, cell: float) -> np.ndarray:
        cache = self._memo()
        memo_key = (pattern.canonical(), case, cell)
        cached = cache.get(memo_key)
        if cached is not None:
            return cached
        try:
            case_index = self._case_index[case]
        except KeyError as error:
            raise KeyError(f"case {case.label} was not fitted by this encoder") from error
        batches = case.pair.split("_")
        if len(batches) != 2:
            raise ValueError(f"case pair must contain two fresh batches: {case.pair}")

        features: list[float] = []
        fresh_mask = np.zeros(len(SLOTS), dtype=bool)
        batch_masks = [np.zeros(len(SLOTS), dtype=bool) for _ in range(2)]
        burned_source_radius: list[float] = []
        burned_displacement: list[float] = []
        max_radius = max(slot.radius for slot in SLOTS)

        for slot, item in zip(SLOTS, pattern.items, strict=True):
            destination_x = slot.col / 8.0
            destination_y = slot.row / 8.0
            destination_radius = slot.radius / max_radius
            if item.is_fresh:
                assert item.batch is not None
                if item.batch not in batches:
                    raise ValueError(f"batch {item.batch} is outside case pair {case.pair}")
                batch_index = batches.index(item.batch)
                fresh0 = float(batch_index == 0)
                fresh1 = float(batch_index == 1)
                burned = 0.0
                source_x = source_y = rotation1 = rotation2 = 0.0
                source_radius = displacement = 0.0
                fresh_mask[slot.index] = True
                batch_masks[batch_index][slot.index] = True
            else:
                assert item.x is not None and item.y is not None
                fresh0 = fresh1 = 0.0
                burned = 1.0
                # J/10 is the quarter-core origin in the packaged APRQ decks.
                source_x = (X_INDEX[item.x] - X_INDEX["J"]) / 8.0
                source_y = (item.y - 10) / 8.0
                rotation1 = float(item.rotation == 1)
                rotation2 = float(item.rotation == 2)
                source_radius = math.hypot(source_x, source_y) / math.sqrt(2.0)
                displacement = math.hypot(
                    source_x - destination_x, source_y - destination_y
                ) / math.sqrt(2.0)
                burned_source_radius.append(source_radius)
                burned_displacement.append(displacement)
            features.extend(
                (
                    fresh0,
                    fresh1,
                    burned,
                    source_x,
                    source_y,
                    rotation1,
                    rotation2,
                    source_radius,
                    displacement,
                    destination_radius * fresh0,
                    destination_radius * fresh1,
                )
            )

        weights = np.asarray([slot.multiplicity for slot in SLOTS], dtype=float)
        radii = np.asarray([slot.radius / max_radius for slot in SLOTS], dtype=float)
        weighted_feed = float(weights[fresh_mask].sum())
        batch_feeds = [float(weights[mask].sum()) for mask in batch_masks]

        def weighted_radial(mask: np.ndarray) -> tuple[float, float]:
            if not mask.any():
                return 0.0, 0.0
            selected_weights = weights[mask]
            selected_radii = radii[mask]
            mean = float(np.average(selected_radii, weights=selected_weights))
            variance = float(np.average((selected_radii - mean) ** 2, weights=selected_weights))
            return mean, math.sqrt(max(variance, 0.0))

        all_fresh_radial = weighted_radial(fresh_mask)
        batch_radial = [weighted_radial(mask) for mask in batch_masks]
        same_fresh_edges = sum(
            fresh_mask[first] and fresh_mask[second] for first, second in NEIGHBOR_PAIRS
        ) / max(len(NEIGHBOR_PAIRS), 1)
        mixed_batch_edges = sum(
            fresh_mask[first]
            and fresh_mask[second]
            and batch_masks[0][first] != batch_masks[0][second]
            for first, second in NEIGHBOR_PAIRS
        ) / max(len(NEIGHBOR_PAIRS), 1)
        radial_bins = []
        for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            mask = fresh_mask & (radii >= low) & (radii < high)
            radial_bins.append(float(weights[mask].sum()) / 241.0)

        src = np.asarray(burned_source_radius or [0.0], dtype=float)
        disp = np.asarray(burned_displacement or [0.0], dtype=float)
        aggregate = [
            weighted_feed / 241.0,
            batch_feeds[0] / 241.0,
            batch_feeds[1] / 241.0,
            all_fresh_radial[0],
            all_fresh_radial[1],
            batch_radial[0][0],
            batch_radial[0][1],
            batch_radial[1][0],
            batch_radial[1][1],
            same_fresh_edges,
            mixed_batch_edges,
            *radial_bins,
            float(src.mean()),
            float(src.std()),
            float(disp.mean()),
            float(disp.std()),
        ]
        if len(aggregate) != self.aggregate_dim:
            raise AssertionError("aggregate feature contract changed")
        features.extend(aggregate)
        case_one_hot = [0.0] * len(self.cases)
        case_one_hot[case_index] = 1.0
        features.extend(case_one_hot)
        # Cell is nominal enrichment metadata; centre only, do not infer assay.
        features.extend(((float(cell) - 5.25) / 0.25, (case.feed - 119.0) / 4.0))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.output_dim,):
            raise AssertionError(f"feature shape {result.shape}, expected {(self.output_dim,)}")
        # Freeze the cached array: ``transform`` stacks (and thus copies) it, so
        # returning the shared instance is safe as long as it is never mutated.
        result.flags.writeable = False
        if len(cache) >= _FEATURE_CACHE_CAP:
            cache.clear()
        cache[memo_key] = result
        return result

    def transform(
        self,
        patterns: Sequence[Pattern],
        cases: Sequence[CaseKey],
        cells: Sequence[float],
    ) -> np.ndarray:
        if not (len(patterns) == len(cases) == len(cells)):
            raise ValueError("patterns, cases and cells must have equal length")
        return np.stack(
            [
                self.transform_one(pattern, case, cell)
                for pattern, case, cell in zip(patterns, cases, cells, strict=True)
            ],
            axis=0,
        )
