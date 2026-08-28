"""GA stage: full-recompute loading-pattern search on the orbit-unit genome.

The genome layer encodes what every archived feasible core actually is
(empirically verified over the packaged seeds):

* the quarter core has **60 movable orbit units** — 52 interior slots plus 8
  axis twin pairs (the horizontal-axis slot at ``(0, k)`` and the vertical-axis
  slot at ``(k, 0)`` are one unit whose two arms always share one state) — and
  the centre slot, which is always fresh and never a shuffle source;
* a feed-121 pattern holds fresh fuel on exactly 30 units (1 + 4·30 = 121) and
  wires the remaining 30 burned units to the 30 fresh units by a **perfect
  matching**: each shuffle card's source ``(x, y)`` is a slot that carries a
  FRESH card in the same pattern (depth-1, two-batch management), every fresh
  unit is consumed exactly once, and axis units are always referenced through
  their horizontal representative (never a vertical-axis coordinate);
* fresh cards use rotation 0, vertical-axis shuffle arms rotation 1, all other
  shuffle cards rotation 2.

Random ``Pattern.swap`` chains break this wiring (they re-seat cards without
re-connecting sources, producing source-graph loops = permanently
non-equilibrium cores), so the GA never scrambles a scaffold: every operator
below is **closed** over the invariants, and :meth:`OrbitGenome.from_pattern`
rejects anything outside the intended space with :class:`GenomeError`.

Note: the packaged feed-117 seeds are 3-batch cores (two depth-2 shuffle
edges each); they are deliberately outside this genome and are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import md5
import math
from pathlib import Path
import random
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from .dataset import CaseData, _merge_physical_records
from .domain import (
    CaseKey,
    FOM,
    FuelItem,
    Pattern,
    PatternRecord,
    ROW_LENGTHS,
    SLOTS,
    X_INDEX,
    X_LABELS,
)
from .equilibrium import deck_cycle
from .jsonio import write_json_atomic
from .master import replace_lpd_shf
from .reward import ConstraintConfig, is_fom_feasible
from .surrogate import _spearman


class GenomeError(ValueError):
    """A pattern outside the depth-1 orbit-unit design space."""


class GAStageError(RuntimeError):
    """A GA stage asset/configuration failure."""


OrbitUnit = int  # 0..59: 52 interior units + 8 axis twin pairs


# --------------------------------------------------------------------------
# Orbit-unit geometry
# --------------------------------------------------------------------------

_ROW_OFFSETS: tuple[int, ...] = tuple(
    sum(ROW_LENGTHS[:row]) for row in range(len(ROW_LENGTHS))
)
_CENTER_X = X_LABELS[8]  # "J": the full-core centre column
_CENTER_Y = 9


def _slot_index(row: int, col: int) -> int:
    return _ROW_OFFSETS[row] + col


def _slot_source_coord(slot_index: int) -> tuple[str, int]:
    """Full-core (x, y) of a quarter slot (southeast quarter convention)."""

    slot = SLOTS[slot_index]
    return (X_LABELS[8 + slot.col], 9 + slot.row)


@dataclass(frozen=True, slots=True)
class OrbitUnitDef:
    """One movable orbit unit: its slots and its source representative."""

    unit: OrbitUnit
    kind: Literal["interior", "axis_pair"]
    slots: tuple[int, ...]        # interior: (slot,); axis: (horizontal, vertical)
    source: tuple[str, int]       # representative full-core (x, y)
    radius: float


def _build_units() -> tuple[OrbitUnitDef, ...]:
    units: list[OrbitUnitDef] = []
    for slot in SLOTS:
        if slot.orbit_class != "interior":
            continue
        units.append(
            OrbitUnitDef(
                unit=len(units),
                kind="interior",
                slots=(slot.index,),
                source=_slot_source_coord(slot.index),
                radius=slot.radius,
            )
        )
    for k in range(1, 9):
        horizontal = _slot_index(0, k)
        vertical = _slot_index(k, 0)
        units.append(
            OrbitUnitDef(
                unit=len(units),
                kind="axis_pair",
                slots=(horizontal, vertical),
                source=_slot_source_coord(horizontal),
                radius=SLOTS[horizontal].radius,
            )
        )
    if len(units) != 60:
        raise AssertionError("expected exactly 60 movable orbit units")
    return tuple(units)


ORBIT_UNITS: tuple[OrbitUnitDef, ...] = _build_units()
MOVABLE_UNIT_COUNT = len(ORBIT_UNITS)
FRESH_UNIT_COUNT = 30  # feed 121 = 1 (centre) + 4 * 30
_SOURCE_TO_UNIT: Mapping[tuple[str, int], OrbitUnit] = {
    unit.source: unit.unit for unit in ORBIT_UNITS
}
_SLOT_TO_UNIT: Mapping[int, OrbitUnit] = {
    slot: unit.unit for unit in ORBIT_UNITS for slot in unit.slots
}


def _shuffle_rotation(slot_index: int) -> int:
    return 1 if SLOTS[slot_index].orbit_class == "vertical_axis" else 2


def _coord_slot(x: str, y: int) -> int | None:
    """Quarter slot index for a full-core (x, y), or None when outside."""

    if x not in X_INDEX:
        return None
    col = X_INDEX[x] - 8
    row = int(y) - 9
    if row < 0 or row >= len(ROW_LENGTHS) or col < 0 or col >= ROW_LENGTHS[row]:
        return None
    return _slot_index(row, col)


# --------------------------------------------------------------------------
# Genome
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitGenome:
    """Feed-121 quarter core as fresh-unit set + burned←fresh wiring."""

    fresh: tuple[tuple[OrbitUnit, str], ...]        # 30 (unit, batch), sorted
    wiring: tuple[tuple[OrbitUnit, OrbitUnit], ...]  # 30 (burned ← fresh), sorted
    center_batch: str

    def validate(self) -> None:
        if not self.center_batch or not isinstance(self.center_batch, str):
            raise GenomeError("centre batch must be a non-empty batch name")
        fresh_units = [unit for unit, _ in self.fresh]
        burned_units = [unit for unit, _ in self.wiring]
        sources = [source for _, source in self.wiring]
        for unit in (*fresh_units, *burned_units, *sources):
            if not (0 <= unit < MOVABLE_UNIT_COUNT):
                raise GenomeError(f"orbit unit {unit} out of range")
        if list(fresh_units) != sorted(set(fresh_units)):
            raise GenomeError("fresh units must be strictly sorted and unique")
        if list(burned_units) != sorted(set(burned_units)):
            raise GenomeError("burned units must be strictly sorted and unique")
        for unit, batch in self.fresh:
            if not batch or not isinstance(batch, str):
                raise GenomeError(f"fresh unit {unit} has an invalid batch name")
        fresh_set = set(fresh_units)
        burned_set = set(burned_units)
        if fresh_set & burned_set:
            raise GenomeError("a unit cannot be both fresh and burned")
        if fresh_set | burned_set != set(range(MOVABLE_UNIT_COUNT)):
            raise GenomeError("fresh and burned units must cover all 60 units")
        if len(self.fresh) != FRESH_UNIT_COUNT:
            raise GenomeError(
                f"feed-121 genome requires exactly {FRESH_UNIT_COUNT} fresh "
                f"units, found {len(self.fresh)}"
            )
        if sorted(sources) != sorted(fresh_set):
            raise GenomeError(
                "wiring must be a perfect matching: every fresh unit consumed "
                "exactly once"
            )

    def to_pattern(self) -> Pattern:
        """Compile to a 69-card pattern under the seed conventions."""

        self.validate()
        items: list[FuelItem | None] = [None] * len(SLOTS)
        items[0] = FuelItem(kind="fresh", batch=self.center_batch)
        for unit, batch in self.fresh:
            for slot in ORBIT_UNITS[unit].slots:
                items[slot] = FuelItem(kind="fresh", batch=batch)
        for burned, source in self.wiring:
            x, y = ORBIT_UNITS[source].source
            for slot in ORBIT_UNITS[burned].slots:
                items[slot] = FuelItem(
                    kind="shuffle",
                    restart=1,
                    x=x,
                    y=y,
                    rotation=_shuffle_rotation(slot),
                )
        if any(item is None for item in items):
            raise AssertionError("orbit units failed to cover every slot")
        return Pattern(tuple(items))  # type: ignore[arg-type]

    @classmethod
    def from_pattern(cls, pattern: Pattern) -> "OrbitGenome":
        """Parse a pattern, rejecting anything outside the design space."""

        centre = pattern.items[0]
        if not centre.is_fresh:
            raise GenomeError("the centre slot must hold a fresh assembly")
        if centre.rotation != 0:
            raise GenomeError("the centre fresh card must use rotation 0")

        fresh: list[tuple[OrbitUnit, str]] = []
        wiring_raw: list[tuple[OrbitUnit, tuple[str, int]]] = []
        for unit in ORBIT_UNITS:
            arms = [pattern.items[slot] for slot in unit.slots]
            states = {arm.is_fresh for arm in arms}
            if len(states) != 1:
                raise GenomeError(
                    f"axis twin pair {unit.unit} has inconsistent arms "
                    "(one fresh, one shuffled)"
                )
            if arms[0].is_fresh:
                batches = {arm.batch for arm in arms}
                if len(batches) != 1:
                    raise GenomeError(
                        f"axis twin pair {unit.unit} arms carry different "
                        f"batches {sorted(batches)}"  # type: ignore[type-var]
                    )
                for arm in arms:
                    if arm.rotation != 0:
                        raise GenomeError(
                            f"fresh card on unit {unit.unit} must use rotation 0"
                        )
                assert arms[0].batch is not None
                fresh.append((unit.unit, arms[0].batch))
                continue
            sources = {(arm.restart, arm.x, arm.y) for arm in arms}
            if len(sources) != 1:
                raise GenomeError(
                    f"axis twin pair {unit.unit} arms shuffle from different "
                    f"sources {sorted(sources)}"  # type: ignore[type-var]
                )
            for slot, arm in zip(unit.slots, arms, strict=True):
                expected = _shuffle_rotation(slot)
                if arm.rotation != expected:
                    raise GenomeError(
                        f"shuffle card at slot {slot} uses rotation "
                        f"{arm.rotation}; the seed convention is {expected}"
                    )
            restart, x, y = next(iter(sources))
            if restart != 1:
                raise GenomeError(
                    f"unit {unit.unit} shuffles from restart {restart}; the "
                    "single-restart equilibrium decks require restart 1"
                )
            assert x is not None and y is not None
            wiring_raw.append((unit.unit, (x, int(y))))

        wiring: list[tuple[OrbitUnit, OrbitUnit]] = []
        for burned, (x, y) in wiring_raw:
            slot = _coord_slot(x, y)
            if slot is None:
                raise GenomeError(
                    f"unit {burned} sources ({x}, {y}) outside the quarter core"
                )
            orbit_class = SLOTS[slot].orbit_class
            if orbit_class == "center":
                raise GenomeError(
                    f"unit {burned} sources the centre slot; the centre is "
                    "fresh and never a shuffle source"
                )
            if orbit_class == "vertical_axis":
                raise GenomeError(
                    f"unit {burned} sources the vertical-axis coordinate "
                    f"({x}, {y}); axis units are always referenced through "
                    "their horizontal representative"
                )
            source_unit = _SLOT_TO_UNIT[slot]
            wiring.append((burned, source_unit))

        fresh_units = {unit for unit, _ in fresh}
        consumption: dict[OrbitUnit, int] = {}
        for burned, source_unit in wiring:
            if source_unit not in fresh_units:
                raise GenomeError(
                    f"unit {burned} sources unit {source_unit}, which is not "
                    "fresh in the same pattern (shuffle depth != 1)"
                )
            consumption[source_unit] = consumption.get(source_unit, 0) + 1
        over = sorted(unit for unit, count in consumption.items() if count != 1)
        unconsumed = sorted(fresh_units - set(consumption))
        if over or unconsumed:
            raise GenomeError(
                "wiring is not a perfect matching: units consumed more than "
                f"once {over}, fresh units never consumed {unconsumed}"
            )

        assert centre.batch is not None
        genome = cls(
            fresh=tuple(sorted(fresh)),
            wiring=tuple(sorted(wiring)),
            center_batch=centre.batch,
        )
        genome.validate()
        return genome


def _pair_batches(pair: str) -> tuple[str, str]:
    parts = pair.split("_")
    if len(parts) != 2 or not all(parts):
        raise GenomeError(f"pair {pair!r} must be 'TYPEA_TYPEB'")
    return parts[0], parts[1]


def random_genome(rng: random.Random, pair: str) -> OrbitGenome:
    """Uniform random feed-121 genome: 30 fresh units, random matching."""

    batches = _pair_batches(pair)
    fresh_units = sorted(rng.sample(range(MOVABLE_UNIT_COUNT), FRESH_UNIT_COUNT))
    fresh = tuple((unit, rng.choice(batches)) for unit in fresh_units)
    burned = [
        unit for unit in range(MOVABLE_UNIT_COUNT) if unit not in set(fresh_units)
    ]
    sources = list(fresh_units)
    rng.shuffle(sources)
    wiring = tuple(sorted(zip(burned, sources, strict=True)))
    return OrbitGenome(fresh=fresh, wiring=wiring, center_batch=batches[0])


def _deterministic_genome(
    fresh_units: Sequence[OrbitUnit], batches: tuple[str, str]
) -> OrbitGenome:
    """Deterministic batches + radius-matched wiring for heuristic starts."""

    fresh_sorted = sorted(fresh_units)
    by_radius = sorted(fresh_sorted, key=lambda u: (ORBIT_UNITS[u].radius, u))
    batch_of = {unit: batches[index % 2] for index, unit in enumerate(by_radius)}
    fresh = tuple((unit, batch_of[unit]) for unit in fresh_sorted)
    burned = sorted(set(range(MOVABLE_UNIT_COUNT)) - set(fresh_sorted))
    burned_by_radius = sorted(burned, key=lambda u: (ORBIT_UNITS[u].radius, u))
    # Inner burned positions inherit from outer fresh positions and vice
    # versa — the classic in-out reload motion.
    sources = list(reversed(by_radius))
    wiring = tuple(sorted(zip(burned_by_radius, sources, strict=True)))
    return OrbitGenome(fresh=fresh, wiring=wiring, center_batch=batches[0])


def heuristic_genomes(pair: str) -> list[OrbitGenome]:
    """Structural (ring / checkerboard / radial zoning) initial genomes.

    These are geometric priors, not warm seeds: they carry no measured labels
    and are MASTER-confirmed like every other candidate.
    """

    batches = _pair_batches(pair)
    by_radius = sorted(
        range(MOVABLE_UNIT_COUNT), key=lambda u: (ORBIT_UNITS[u].radius, u)
    )

    # Ring pattern: alternating radius ranks (fresh/burned annuli).
    rings = [unit for index, unit in enumerate(by_radius) if index % 2 == 0]

    # Checkerboard: (row + col) parity of the unit's primary slot, trimmed or
    # topped up in radius order to exactly 30 fresh units.
    checker = [
        unit.unit
        for unit in ORBIT_UNITS
        if (SLOTS[unit.slots[0]].row + SLOTS[unit.slots[0]].col) % 2 == 0
    ]
    if len(checker) > FRESH_UNIT_COUNT:
        keep = sorted(checker, key=lambda u: (ORBIT_UNITS[u].radius, u))
        checker = keep[:FRESH_UNIT_COUNT]
    else:
        missing = [unit for unit in by_radius if unit not in set(checker)]
        checker = checker + missing[: FRESH_UNIT_COUNT - len(checker)]

    # Radial zoning: fresh fuel on the 30 outermost units (low-leakage-style
    # peripheral feed).
    radial = by_radius[-FRESH_UNIT_COUNT:]

    return [
        _deterministic_genome(rings, batches),
        _deterministic_genome(checker, batches),
        _deterministic_genome(radial, batches),
    ]


# --------------------------------------------------------------------------
# Mutation (closed operators)
# --------------------------------------------------------------------------


def sample_move_count(rng: random.Random, geometric_p: float) -> int:
    """1 + Geometric(p) with a hard cap to keep children local."""

    if not (0.0 < geometric_p <= 1.0):
        raise ValueError("geometric_p must be in (0, 1]")
    count = 1
    while count < 8 and rng.random() > geometric_p:
        count += 1
    return count


def _rewire_swap(genome: OrbitGenome, rng: random.Random) -> OrbitGenome:
    """Exchange the sources of two burned units."""

    wiring = list(genome.wiring)
    first, second = rng.sample(range(len(wiring)), 2)
    (b1, s1), (b2, s2) = wiring[first], wiring[second]
    wiring[first], wiring[second] = (b1, s2), (b2, s1)
    return replace(genome, wiring=tuple(sorted(wiring)))


def _fresh_relocate(genome: OrbitGenome, rng: random.Random) -> OrbitGenome:
    """Swap the fresh/burned roles of one fresh unit A and one burned unit B.

    A's consumer re-targets to B (now fresh, inheriting A's batch) and A
    inherits B's old source — a three-card edit that preserves the matching.
    """

    fresh = dict(genome.fresh)
    wiring = dict(genome.wiring)
    a = rng.choice(sorted(fresh))
    b = rng.choice(sorted(wiring))
    batch_a = fresh.pop(a)
    source_b = wiring.pop(b)
    fresh[b] = batch_a
    if source_b == a:
        # B consumed A directly: the edge simply reverses.
        wiring[a] = b
    else:
        consumer = next(unit for unit, source in genome.wiring if source == a)
        wiring[consumer] = b
        wiring[a] = source_b
    return OrbitGenome(
        fresh=tuple(sorted(fresh.items())),
        wiring=tuple(sorted(wiring.items())),
        center_batch=genome.center_batch,
    )


def _batch_flip(
    genome: OrbitGenome, rng: random.Random, batches: Sequence[str]
) -> OrbitGenome:
    """Flip one fresh unit's (or the centre's) batch label."""

    if len(batches) < 2:
        return _rewire_swap(genome, rng)
    targets: list[Any] = sorted(unit for unit, _ in genome.fresh) + ["center"]
    target = rng.choice(targets)
    if target == "center":
        others = [batch for batch in batches if batch != genome.center_batch]
        return replace(genome, center_batch=rng.choice(others))
    fresh = dict(genome.fresh)
    others = [batch for batch in batches if batch != fresh[target]]
    fresh[target] = rng.choice(others)
    return replace(genome, fresh=tuple(sorted(fresh.items())))


def _batch_swap(
    genome: OrbitGenome, rng: random.Random, batches: Sequence[str]
) -> OrbitGenome:
    """Exchange the batch labels of two fresh units with different batches."""

    groups: dict[str, list[OrbitUnit]] = {}
    for unit, batch in genome.fresh:
        groups.setdefault(batch, []).append(unit)
    if len(groups) < 2:
        return _batch_flip(genome, rng, batches)
    names = sorted(groups)
    first, second = rng.sample(names, 2)
    u1 = rng.choice(sorted(groups[first]))
    u2 = rng.choice(sorted(groups[second]))
    fresh = dict(genome.fresh)
    fresh[u1], fresh[u2] = fresh[u2], fresh[u1]
    return replace(genome, fresh=tuple(sorted(fresh.items())))


def _one_move(
    genome: OrbitGenome,
    rng: random.Random,
    fresh_relocate_prob: float,
    batch_prob: float,
    batches: Sequence[str],
) -> OrbitGenome:
    draw = rng.random()
    if draw < fresh_relocate_prob:
        return _fresh_relocate(genome, rng)
    if draw < fresh_relocate_prob + batch_prob:
        if rng.random() < 0.5:
            return _batch_flip(genome, rng, batches)
        return _batch_swap(genome, rng, batches)
    return _rewire_swap(genome, rng)


def mutate(
    genome: OrbitGenome,
    rng: random.Random,
    n_moves: int,
    *,
    fresh_relocate_prob: float = 0.35,
    batch_prob: float = 0.15,
    batches: Sequence[str] | None = None,
) -> OrbitGenome:
    """Apply ``n_moves`` closed operators; guaranteed to change the genome."""

    if n_moves < 1:
        raise ValueError("n_moves must be >= 1")
    if batches is None:
        batches = sorted(
            {batch for _, batch in genome.fresh} | {genome.center_batch}
        )
    child = genome
    for _ in range(n_moves):
        child = _one_move(child, rng, fresh_relocate_prob, batch_prob, batches)
    guard = 0
    while child == genome:
        # A move sequence can cancel itself; a mutation must never be a no-op.
        child = _one_move(child, rng, fresh_relocate_prob, batch_prob, batches)
        guard += 1
        if guard > 32:
            raise AssertionError("mutation failed to change the genome")
    return child


# --------------------------------------------------------------------------
# Fitness
# --------------------------------------------------------------------------


_PENALTY_WEIGHT = 1.0e6
_CBC_TIEBREAK_WEIGHT = 1.0e-3
# Trade-off z-scales: fixed characteristic spreads (ranking only).
_TRADE_CYCLE_SCALE = 10.0   # EFPD
_TRADE_F_R_SCALE = 0.05
_TRADE_CBC_SCALE = 50.0     # ppm


@dataclass(frozen=True)
class GAFitness:
    """Scale-separated lexicographic fitness over MASTER-verified FOMs.

    ``penalty`` is the normalized squared constraint excess (ConstraintConfig
    widths) plus 1.0 per unknown-but-enforced axis and 1.0 for a
    non-converged chain; its 1e6 weight makes any violation dominate every
    objective difference.  ``target_cycle`` then optimizes the **continuous**
    distance |cyclen − target| (never a window hinge) with Max CBC as the
    strictly-inferior tie-break.
    """

    mode: Literal["trade_off", "target_cycle"]
    constraints: ConstraintConfig
    target_efpd: float = 625.0

    def __post_init__(self) -> None:
        if self.mode not in ("trade_off", "target_cycle"):
            raise ValueError(f"unknown GA fitness mode {self.mode!r}")
        if self.mode == "target_cycle" and (
            not math.isfinite(self.target_efpd) or self.target_efpd <= 0.0
        ):
            raise ValueError(
                "target_cycle fitness requires a finite positive target_efpd"
            )

    def penalty(self, fom: FOM) -> float:
        config = self.constraints
        axes: list[tuple[float | None, float, float]] = [
            (fom.f_r, config.f_r_limit, config.f_r_width),
            (fom.cbc_max, config.cbc_limit, config.cbc_width),
            (fom.f_q, config.f_q_limit, config.f_q_width),
            (fom.ao_abs, config.ao_abs_limit, config.ao_width),
        ]
        if config.assembly_burnup_enabled:
            axes.append(
                (
                    fom.max_assembly_burnup,
                    config.max_assembly_burnup_limit,
                    config.assembly_burnup_width,
                )
            )
        if config.pin_burnup_enabled:
            axes.append(
                (
                    fom.max_pin_burnup,
                    config.max_pin_burnup_limit,
                    config.pin_burnup_width,
                )
            )
        total = 0.0
        for value, limit, width in axes:
            if value is None:
                # An unknown enforced axis can never certify feasibility.
                total += 1.0
                continue
            total += (max(0.0, float(value) - limit) / width) ** 2
        if not getattr(fom, "converged", True):
            total += 1.0
        return total

    def cyclen_gap(self, fom: FOM) -> float | None:
        if self.mode != "target_cycle":
            return None
        return abs(float(fom.cyclen) - float(self.target_efpd))

    def scalar(self, fom: FOM) -> float:
        p = self.penalty(fom)
        if self.mode == "target_cycle":
            return (
                -_PENALTY_WEIGHT * p
                - abs(float(fom.cyclen) - float(self.target_efpd))
                - _CBC_TIEBREAK_WEIGHT * float(fom.cbc_max)
            )
        weights = self.constraints.objective_weights()
        value = (
            weights[0] * (float(fom.cyclen) / _TRADE_CYCLE_SCALE)
            - weights[1] * (float(fom.f_r) / _TRADE_F_R_SCALE)
            - weights[2] * (float(fom.cbc_max) / _TRADE_CBC_SCALE)
        )
        return -_PENALTY_WEIGHT * p + float(value)

    def constraint_feasible(self, fom: FOM) -> bool:
        """Constraint feasibility of the values, ignoring convergence.

        This is the GA archive's ``feasible`` flag (ga_eval parity); the
        separate ``eq_ok`` flag carries convergence so ``require_eq_ok``
        ingest policies stay meaningful.
        """

        return is_fom_feasible(replace(fom, converged=True), self.constraints)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GASearchConfig:
    """[ga_search] deck section (mutation-only μ+λ, no crossover)."""

    enabled: bool = False
    mode: str = "trade_off"
    target_efpd: float = 625.0
    population: int = 24            # μ: MASTER-confirmed elites only
    offspring: int = 60             # λ per generation; surrogate ranks within it
    confirm_per_generation: int = 8
    master_budget: int = 160        # confirmed evaluations per case, bootstrap included
    bootstrap: int = 16
    explore_confirm_fraction: float = 0.5  # random confirm share pre-objective; floor 0.25 after
    stagnation_patience: int = 6    # generations without best-fitness improvement
    geometric_p: float = 0.5
    fresh_relocate_prob: float = 0.35
    batch_prob: float = 0.15
    tournament: int = 3
    archive_dir: str = ""           # empty -> <run_dir>/ga_archive
    # Rank-model warm start: train the within-generation ranker on prior
    # MASTER labels from the skeleton-source package (fp_v12 parity — the
    # original GA never ranked with a cold model).  Population, elites and
    # the archive stay from-scratch; the labels only inform ranking.
    surrogate_warm_labels: bool = False


# --------------------------------------------------------------------------
# Case assets (skeleton only — the GA stage never reads manifest seeds)
# --------------------------------------------------------------------------


def _read_deck_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decoding is total")


@dataclass(frozen=True)
class GACaseAssets:
    """Everything one GA case needs: deck skeleton, base restart, cell."""

    case: CaseData
    template_text: str
    cycle: int                      # deck's %JOB_IDE cycle (base cycle + 1)
    base_restart_name: str
    source_root: Path
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)

    @classmethod
    def load(
        cls,
        source_root: str | Path,
        pair: str,
        feed: int,
        cell: float,
        *,
        constraints: ConstraintConfig | None = None,
    ) -> "GACaseAssets":
        if int(feed) != 1 + 4 * FRESH_UNIT_COUNT:
            raise GAStageError(
                f"the orbit-unit GA genome supports feed 121 only, got {feed}"
            )
        root = Path(source_root).resolve()
        key = CaseKey(pair, int(feed))
        base_dir = root / "bases" / key.folder
        restarts = sorted(base_dir.glob("MAS_RST.*")) if base_dir.is_dir() else []
        if len(restarts) != 1:
            raise GAStageError(
                f"case {key.label} requires exactly one {base_dir}/MAS_RST.*; "
                f"found {len(restarts)}"
            )
        decks = sorted((root / "cores" / key.folder).glob("*/MAS_INP_cy*.inp"))
        if not decks:
            raise GAStageError(
                f"case {key.label} has no template deck under "
                f"{root / 'cores' / key.folder}"
            )
        # Any seed deck is a valid skeleton; %LPD_SHF is always replaced.
        template = decks[0]
        text = _read_deck_text(template)
        if restarts[0].name not in text:
            raise GAStageError(
                f"template deck {template} does not reference the base "
                f"restart {restarts[0].name}"
            )
        case = CaseData(
            key=key,
            cell=float(cell),
            records=(),
            template_path=template,
            restart_path=restarts[0],
        )
        return cls(
            case=case,
            template_text=text,
            cycle=deck_cycle(text),
            base_restart_name=restarts[0].name,
            source_root=root,
            constraints=constraints or ConstraintConfig(),
        )

    def cycle1_deck(self, pattern: Pattern) -> str:
        """The first reload deck for this pattern (cycle-1 archive deck).

        Archiving only this deck keeps ``ingest._select_final_deck`` pointed
        at a deck that references the case's base restart; archiving the whole
        chain would promote a max-cycle deck whose restart the store lacks.
        """

        return replace_lpd_shf(self.template_text, pattern.to_shf())


# --------------------------------------------------------------------------
# Archive (ga_eval-compatible)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GAEvaluation:
    """One MASTER-confirmed candidate label."""

    fom: FOM
    ncyc: int
    eq_ok: bool
    feasible: bool                  # constraint feasibility, convergence aside
    fitness: float
    penalty: float
    raw_master_calls: int = 0


def archive_candidate(
    archive_root: str | Path,
    base_key: str,
    pattern: Pattern,
    result: GAEvaluation,
    cycle1_deck_text: str,
    base_restart_name: str,
    extras: Mapping[str, Any],
) -> Path:
    """Write one ``ga_eval``-compatible archive entry (cycle-1 deck ONLY)."""

    shf = pattern.to_shf()
    seed_id = md5(shf.encode("utf-8")).hexdigest()[:12]
    seed_dir = Path(archive_root) / base_key / seed_id
    seed_dir.mkdir(parents=True, exist_ok=True)
    fom = result.fom
    meta = {
        "base_restart": base_restart_name,
        "shf": shf,
        "F_r": fom.f_r,
        "CBC_max": fom.cbc_max,
        "F_q": fom.f_q,
        "AO_min": fom.ao_min,
        "AO_max": fom.ao_max,
        "cyclen": fom.cyclen,
        "ncyc": result.ncyc,
        "eq_ok": result.eq_ok,
        "feasible": result.feasible,
        "extras": dict(extras),
    }
    warning = write_json_atomic(seed_dir / "meta.json", meta)
    if warning is not None:
        raise GAStageError(f"GA archive meta write degraded: {warning}")
    cycle = deck_cycle(cycle1_deck_text)
    (seed_dir / f"MAS_INP_cy{cycle:02d}.inp").write_text(
        cycle1_deck_text, encoding="utf-8", newline="\n"
    )
    return seed_dir


# --------------------------------------------------------------------------
# Search loop
# --------------------------------------------------------------------------


@dataclass
class GACaseResult:
    case_label: str
    mode: str
    stop_reason: str
    generations: int
    confirmed: int
    budget: int
    budget_used: int
    failures: int
    archived: int
    feasible: int
    duplicates_blocked: int
    raw_master_calls: int
    best: dict[str, Any] | None
    best_fitness: float | None
    spearman: list[dict[str, Any]]
    archive_root: str
    algorithm: str = "ga"
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self, *, include_events: bool = False) -> dict[str, Any]:
        payload = {
            "case": self.case_label,
            "algorithm": self.algorithm,
            "mode": self.mode,
            "stop_reason": self.stop_reason,
            "generations": self.generations,
            "confirmed": self.confirmed,
            "budget": self.budget,
            "budget_used": self.budget_used,
            "failures": self.failures,
            "archived": self.archived,
            "feasible": self.feasible,
            "duplicates_blocked": self.duplicates_blocked,
            "raw_master_calls": self.raw_master_calls,
            "best": self.best,
            "best_fitness": self.best_fitness,
            "spearman": list(self.spearman),
            "archive_root": self.archive_root,
        }
        if include_events:
            payload["events"] = list(self.events)
        return payload


@dataclass
class _Confirmed:
    genome: OrbitGenome | None
    pattern: Pattern
    evaluation: GAEvaluation
    generation: int
    parent_digest: str | None
    n_moves: int


class _BudgetedRun:
    """Shared ledger/label/archive/stagnation bookkeeping for both runners."""

    def __init__(
        self,
        assets: GACaseAssets,
        cfg: GASearchConfig,
        evaluator: Any,
        rng: random.Random,
        on_event: Callable[[dict[str, Any]], None] | None,
        archive_root: str | Path,
        rng_seed: int | None,
        algorithm: str,
    ) -> None:
        self.assets = assets
        self.cfg = cfg
        self.evaluator = evaluator
        self.rng = rng
        self.on_event = on_event
        self.archive_root = Path(archive_root)
        self.rng_seed = rng_seed
        self.algorithm = algorithm
        target = cfg.target_efpd
        if (
            cfg.mode == "target_cycle"
            and assets.constraints.cycle_target_efpd is not None
        ):
            target = float(assets.constraints.cycle_target_efpd)
        self.fitness = GAFitness(
            mode=cfg.mode,  # type: ignore[arg-type]
            constraints=assets.constraints,
            target_efpd=target,
        )
        self.batches = _pair_batches(assets.case.key.pair)
        self.ledger: dict[str, str] = {}   # canonical -> digest
        self.confirmed: list[_Confirmed] = []
        self.events: list[dict[str, Any]] = []
        self.spearman_rows: list[dict[str, Any]] = []
        self.budget_used = 0
        self.failures = 0
        self.archived = 0
        self.feasible_count = 0
        self.duplicates_blocked = 0
        self.raw_master_calls = 0
        self.best_fitness: float | None = None
        self.best: _Confirmed | None = None
        self.stall = 0

    # ------------------------------------------------------------- plumbing

    def emit(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)
        if self.on_event is not None:
            self.on_event(payload)

    def claim(self, pattern: Pattern) -> bool:
        """Register a canonical in the run ledger; False when already seen."""

        canonical = pattern.canonical()
        if canonical in self.ledger:
            self.duplicates_blocked += 1
            return False
        self.ledger[canonical] = pattern.digest
        return True

    @property
    def budget_left(self) -> int:
        return self.cfg.master_budget - self.budget_used

    def evaluate_wave(
        self, entries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """MASTER-confirm a batch; every outcome (fail included) is a label."""

        patterns = [entry["pattern"] for entry in entries]
        many = getattr(self.evaluator, "evaluate_many", None)
        if many is not None:
            outcomes = many(self.assets.case, patterns)
        else:
            outcomes = []
            for pattern in patterns:
                try:
                    outcomes.append(self.evaluator.evaluate(self.assets.case, pattern))
                except Exception as error:  # noqa: BLE001 — in-place failure
                    outcomes.append(error)
        rows: list[dict[str, Any]] = []
        for entry, outcome in zip(entries, outcomes, strict=True):
            self.budget_used += 1
            pattern = entry["pattern"]
            if isinstance(outcome, Exception):
                self.failures += 1
                rows.append(
                    {
                        **entry,
                        "digest": pattern.digest,
                        "error": f"{type(outcome).__name__}: {outcome}",
                    }
                )
                continue
            fom = outcome.fom
            metadata = getattr(outcome, "metadata", {}) or {}
            evaluation = GAEvaluation(
                fom=fom,
                ncyc=int(metadata.get("n_cycles", 0) or 0),
                eq_ok=bool(fom.converged),
                feasible=self.fitness.constraint_feasible(fom),
                fitness=self.fitness.scalar(fom),
                penalty=self.fitness.penalty(fom),
                raw_master_calls=int(getattr(outcome, "raw_master_calls", 0) or 0),
            )
            self.raw_master_calls += evaluation.raw_master_calls
            confirmed = _Confirmed(
                genome=entry.get("genome"),
                pattern=pattern,
                evaluation=evaluation,
                generation=int(entry.get("generation", 0)),
                parent_digest=entry.get("parent_digest"),
                n_moves=int(entry.get("n_moves", 0)),
            )
            self.confirmed.append(confirmed)
            if is_fom_feasible(fom, self.assets.constraints):
                self.feasible_count += 1
            if evaluation.feasible:
                self.archive(confirmed)
            rows.append(
                {
                    **entry,
                    "digest": pattern.digest,
                    "fitness": evaluation.fitness,
                    "penalty": evaluation.penalty,
                    "feasible": evaluation.feasible,
                    "eq_ok": evaluation.eq_ok,
                    "fom": fom.as_dict(),
                }
            )
        return rows

    def archive(self, confirmed: _Confirmed) -> Path:
        evaluation = confirmed.evaluation
        extras = {
            "algorithm": self.algorithm,
            "generation": confirmed.generation,
            "parent_digest": confirmed.parent_digest,
            "n_moves": confirmed.n_moves,
            "mode": self.cfg.mode,
            "fitness": evaluation.fitness,
            "cyclen_gap": self.fitness.cyclen_gap(evaluation.fom),
            "seed_rng": self.rng_seed,
        }
        path = archive_candidate(
            self.archive_root,
            self.assets.case.key.folder,
            confirmed.pattern,
            evaluation,
            self.assets.cycle1_deck(confirmed.pattern),
            self.assets.base_restart_name,
            extras,
        )
        self.archived += 1
        return path

    def track_best(self) -> bool:
        """Update the best penalized fitness; True when it improved."""

        if not self.confirmed:
            return False
        best = max(self.confirmed, key=lambda c: c.evaluation.fitness)
        improved = (
            self.best_fitness is None
            or best.evaluation.fitness > self.best_fitness + 1.0e-9
        )
        if improved:
            self.best_fitness = best.evaluation.fitness
            self.best = best
        return improved

    def note_generation(self, improved: bool) -> None:
        self.stall = 0 if improved else self.stall + 1

    def best_payload(self) -> dict[str, Any] | None:
        if self.best is None:
            return None
        evaluation = self.best.evaluation
        return {
            "digest": self.best.pattern.digest,
            "generation": self.best.generation,
            "parent_digest": self.best.parent_digest,
            "fitness": evaluation.fitness,
            "penalty": evaluation.penalty,
            "feasible": evaluation.feasible,
            "eq_ok": evaluation.eq_ok,
            "cyclen_gap": self.fitness.cyclen_gap(evaluation.fom),
            "fom": evaluation.fom.as_dict(),
        }

    def result(self, stop_reason: str, generations: int) -> GACaseResult:
        return GACaseResult(
            case_label=self.assets.case.key.label,
            mode=self.cfg.mode,
            stop_reason=stop_reason,
            generations=generations,
            confirmed=len(self.confirmed),
            budget=self.cfg.master_budget,
            budget_used=self.budget_used,
            failures=self.failures,
            archived=self.archived,
            feasible=self.feasible_count,
            duplicates_blocked=self.duplicates_blocked,
            raw_master_calls=self.raw_master_calls,
            best=self.best_payload(),
            best_fitness=self.best_fitness,
            spearman=self.spearman_rows,
            archive_root=str(self.archive_root),
            algorithm=self.algorithm,
            events=self.events,
        )


def _predicted_foms(
    model: Any, patterns: Sequence[Pattern], key: CaseKey, cell: float
) -> list[FOM] | None:
    """Rank-only surrogate probe; any failure degrades to random ranking."""

    predict = getattr(model, "predict", None)
    if model is None or predict is None:
        return None
    try:
        prediction = predict(list(patterns), [key] * len(patterns), [cell] * len(patterns))
        return [prediction.mean_fom(index) for index in range(len(patterns))]
    except Exception:  # noqa: BLE001 — surrogate quality must never abort GA
        return None


def _gate_mode(model: Any, constraints: ConstraintConfig) -> str:
    """objective/explore verdict for the confirm-batch mix (stub-friendly)."""

    if model is None:
        return "explore"
    mode = getattr(model, "gate_mode", None)
    if mode is not None:
        return str(mode)
    probe = getattr(model, "quality_report", None)
    if probe is None:
        return "explore"
    try:
        return str(probe(constraints=constraints).mode)
    except Exception:  # noqa: BLE001
        return "explore"


def _training_records(run: _BudgetedRun) -> list[PatternRecord]:
    """Every confirmed label — failures-of-constraints included — as records."""

    assets = run.assets
    records: list[PatternRecord] = []
    for confirmed in run.confirmed:
        records.append(
            PatternRecord(
                case=assets.case.key,
                cell=assets.case.cell,
                seed_id=confirmed.pattern.digest,
                pattern=confirmed.pattern,
                fom=confirmed.evaluation.fom,
                ncyc=max(1, confirmed.evaluation.ncyc),
                deck_path=assets.case.template_path,
                shf_path=assets.case.template_path,
            )
        )
    return records


def run_ga_case(
    assets: GACaseAssets,
    cfg: GASearchConfig,
    evaluator: Any,
    surrogate_factory: Callable[[Sequence[PatternRecord]], Any] | None,
    rng: random.Random,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    *,
    archive_root: str | Path,
    rng_seed: int | None = None,
    warm_records: Sequence[PatternRecord] = (),
) -> GACaseResult:
    """Mutation-only (μ+λ) GA with within-generation surrogate ranking.

    μ is drawn from MASTER-confirmed elites only; the surrogate is refit each
    generation on every confirmed label (failed-constraint labels included)
    and used strictly to rank the current offspring.  The confirm batch mixes
    top-ranked and random offspring (always-on control group + Spearman audit
    sample), duplicates are blocked by a run-global canonical ledger, and the
    search stops on exact budget exhaustion or best-penalized-fitness
    stagnation — never on archive growth.
    """

    run = _BudgetedRun(
        assets, cfg, evaluator, rng, on_event, archive_root, rng_seed, "ga"
    )
    pair = assets.case.key.pair

    # ---------------------------------------------------------- bootstrap
    bootstrap_entries: list[dict[str, Any]] = []
    for genome in heuristic_genomes(pair):
        if len(bootstrap_entries) >= min(cfg.bootstrap, run.budget_left):
            break
        pattern = genome.to_pattern()
        if run.claim(pattern):
            bootstrap_entries.append(
                {
                    "genome": genome,
                    "pattern": pattern,
                    "generation": 0,
                    "parent_digest": None,
                    "n_moves": 0,
                    "selected": "heuristic",
                }
            )
    guard = 0
    while len(bootstrap_entries) < min(cfg.bootstrap, run.budget_left):
        genome = random_genome(rng, pair)
        pattern = genome.to_pattern()
        if run.claim(pattern):
            bootstrap_entries.append(
                {
                    "genome": genome,
                    "pattern": pattern,
                    "generation": 0,
                    "parent_digest": None,
                    "n_moves": 0,
                    "selected": "random",
                }
            )
        guard += 1
        if guard > cfg.bootstrap * 1000:
            raise AssertionError("bootstrap failed to draw unique genomes")
    rows = run.evaluate_wave(bootstrap_entries)
    improved = run.track_best()
    run.emit(
        {
            "type": "bootstrap",
            "generation": 0,
            "batch": [
                {**{k: v for k, v in row.items() if k not in ("genome", "pattern")}, "shf": row["pattern"].to_shf()}
                for row in rows
            ],
            "confirmed_total": len(run.confirmed),
            "archived_total": run.archived,
            "best_fitness": run.best_fitness,
            "duplicates_blocked": run.duplicates_blocked,
        }
    )

    # -------------------------------------------------------- generations
    stop_reason = "budget_exhausted"
    generation = 0
    objective_generations = 0
    while run.budget_left > 0:
        generation += 1

        # Surrogate refit on ALL confirmed labels (non-feasible included)
        # plus any prior warm labels; this run's confirmations win on a
        # canonical collision so the ranker always tracks the freshest label.
        model = None
        if surrogate_factory is not None and (
            len(run.confirmed) + len(warm_records) >= 2
        ):
            try:
                model = surrogate_factory(
                    _merge_physical_records(warm_records, _training_records(run))
                )
            except Exception:  # noqa: BLE001 — degrade to random ranking
                model = None
        gate = _gate_mode(model, assets.constraints)
        if gate == "objective":
            objective_generations += 1
            # Deck-relative floor: the permanent random control arm never
            # shrinks below half the configured explore fraction.
            explore_fraction = max(
                cfg.explore_confirm_fraction / 2.0,
                cfg.explore_confirm_fraction * (0.5 ** objective_generations),
            )
        else:
            explore_fraction = cfg.explore_confirm_fraction

        # Parents: tournament over MASTER-confirmed elites only.
        elites = sorted(
            [c for c in run.confirmed if c.genome is not None],
            key=lambda c: c.evaluation.fitness,
            reverse=True,
        )[: cfg.population]

        offspring: list[dict[str, Any]] = []
        attempts = 0
        while len(offspring) < cfg.offspring and attempts < cfg.offspring * 30:
            attempts += 1
            if elites:
                contenders = [
                    elites[rng.randrange(len(elites))]
                    for _ in range(max(1, min(cfg.tournament, len(elites))))
                ]
                parent = max(contenders, key=lambda c: c.evaluation.fitness)
                n_moves = sample_move_count(rng, cfg.geometric_p)
                child = mutate(
                    parent.genome,  # type: ignore[arg-type]
                    rng,
                    n_moves,
                    fresh_relocate_prob=cfg.fresh_relocate_prob,
                    batch_prob=cfg.batch_prob,
                    batches=run.batches,
                )
                parent_digest = parent.pattern.digest
            else:
                # No confirmed parent at all (every bootstrap chain failed):
                # keep proposing honest random genomes instead of stopping.
                child = random_genome(rng, pair)
                n_moves = 0
                parent_digest = None
            pattern = child.to_pattern()
            if not run.claim(pattern):
                continue
            offspring.append(
                {
                    "genome": child,
                    "pattern": pattern,
                    "generation": generation,
                    "parent_digest": parent_digest,
                    "n_moves": n_moves,
                }
            )
        if not offspring:
            stop_reason = "no_new_candidates"
            break

        # Within-generation surrogate ranking (rank only, never a label).
        predicted = _predicted_foms(
            model,
            [entry["pattern"] for entry in offspring],
            assets.case.key,
            assets.case.cell,
        )
        feasibility_stage = run.feasible_count == 0
        if predicted is not None:
            for entry, fom in zip(offspring, predicted, strict=True):
                entry["predicted_score"] = (
                    -run.fitness.penalty(fom)
                    if feasibility_stage
                    else run.fitness.scalar(fom)
                )
            ranked = sorted(
                offspring,
                key=lambda e: (-e["predicted_score"], e["pattern"].digest),
            )
        else:
            ranked = list(offspring)
            rng.shuffle(ranked)
            for entry in ranked:
                entry["predicted_score"] = None

        confirm = min(cfg.confirm_per_generation, run.budget_left)
        n_explore = min(confirm, int(round(confirm * explore_fraction)))
        top = ranked[: confirm - n_explore]
        for entry in top:
            entry["selected"] = "objective"
        remaining = ranked[confirm - n_explore :]
        explore_picks = (
            rng.sample(remaining, min(n_explore, len(remaining)))
            if remaining
            else []
        )
        for entry in explore_picks:
            entry["selected"] = "explore"
        batch = top + explore_picks

        rows = run.evaluate_wave(batch)

        # Spearman audit: predicted score vs MASTER fitness over the batch.
        spearman = float("nan")
        scored = [
            row
            for row in rows
            if row.get("predicted_score") is not None and "fitness" in row
        ]
        if len(scored) >= 2:
            spearman = _spearman(
                np.asarray([row["fitness"] for row in scored], dtype=float),
                np.asarray([row["predicted_score"] for row in scored], dtype=float),
            )
        spearman_value = None if math.isnan(spearman) else float(spearman)
        run.spearman_rows.append(
            {
                "generation": generation,
                "spearman": spearman_value,
                "batch_size": len(scored),
            }
        )

        improved = run.track_best()
        run.note_generation(improved)
        run.emit(
            {
                "type": "generation",
                "generation": generation,
                "gate_mode": gate,
                "explore_fraction": explore_fraction,
                "spearman": spearman_value,
                "batch": [
                    {**{k: v for k, v in row.items() if k not in ("genome", "pattern")}, "shf": row["pattern"].to_shf()}
                    for row in rows
                ],
                "confirmed_total": len(run.confirmed),
                "archived_total": run.archived,
                "best_fitness": run.best_fitness,
                "stall": run.stall,
                "duplicates_blocked": run.duplicates_blocked,
            }
        )
        if run.budget_left <= 0:
            stop_reason = "budget_exhausted"
            break
        if run.stall >= cfg.stagnation_patience:
            stop_reason = "stagnation"
            break

    return run.result(stop_reason, generation)


def run_random_baseline_case(
    assets: GACaseAssets,
    cfg: GASearchConfig,
    evaluator: Any,
    rng: random.Random,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    *,
    archive_root: str | Path,
    rng_seed: int | None = None,
) -> GACaseResult:
    """Equal-budget pure random-search control (no surrogate, no mutation)."""

    run = _BudgetedRun(
        assets, cfg, evaluator, rng, on_event, archive_root, rng_seed, "random_baseline"
    )
    pair = assets.case.key.pair
    stop_reason = "budget_exhausted"
    generation = 0
    while run.budget_left > 0:
        generation += 1
        batch_size = min(cfg.confirm_per_generation, run.budget_left)
        entries: list[dict[str, Any]] = []
        guard = 0
        while len(entries) < batch_size:
            genome = random_genome(rng, pair)
            pattern = genome.to_pattern()
            if run.claim(pattern):
                entries.append(
                    {
                        "genome": genome,
                        "pattern": pattern,
                        "generation": generation,
                        "parent_digest": None,
                        "n_moves": 0,
                        "selected": "random",
                        "predicted_score": None,
                    }
                )
            guard += 1
            if guard > batch_size * 1000:
                stop_reason = "no_new_candidates"
                break
        if not entries:
            break
        rows = run.evaluate_wave(entries)
        improved = run.track_best()
        run.note_generation(improved)
        run.spearman_rows.append(
            {"generation": generation, "spearman": None, "batch_size": 0}
        )
        run.emit(
            {
                "type": "generation",
                "generation": generation,
                "gate_mode": None,
                "explore_fraction": 1.0,
                "spearman": None,
                "batch": [
                    {**{k: v for k, v in row.items() if k not in ("genome", "pattern")}, "shf": row["pattern"].to_shf()}
                    for row in rows
                ],
                "confirmed_total": len(run.confirmed),
                "archived_total": run.archived,
                "best_fitness": run.best_fitness,
                "stall": run.stall,
                "duplicates_blocked": run.duplicates_blocked,
            }
        )
        if run.budget_left <= 0:
            stop_reason = "budget_exhausted"
            break
        if run.stall >= cfg.stagnation_patience:
            stop_reason = "stagnation"
            break
    return run.result(stop_reason, generation)


__all__ = [
    "FRESH_UNIT_COUNT",
    "GACaseAssets",
    "GACaseResult",
    "GAEvaluation",
    "GAFitness",
    "GASearchConfig",
    "GAStageError",
    "GenomeError",
    "MOVABLE_UNIT_COUNT",
    "ORBIT_UNITS",
    "OrbitGenome",
    "OrbitUnit",
    "archive_candidate",
    "heuristic_genomes",
    "mutate",
    "random_genome",
    "run_ga_case",
    "run_random_baseline_case",
    "sample_move_count",
]
