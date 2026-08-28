"""Feed-general orbit-unit genome (``GeneralOrbitGenome``).

The vendored :class:`lpopt.vendor.masterrl.ga.OrbitGenome` encodes a single
feed level (feed 121 = 1 + 4*30) as a *depth-1* perfect matching: 30 fresh
orbit units and 30 burned units, each burned unit shuffled from a fresh unit.
That is the two-batch equilibrium core.

``GeneralOrbitGenome`` lifts that scaffold to **any reachable feed on the
1 + 4*N grid** by allowing a *depth-2* source chain: a burned unit may shuffle
from another burned unit (a twice-burned assembly), so the genome describes a
mixed two/three-batch core.  The invariants (plan sec. 6.1):

* the 60 movable orbit units split into ``N`` fresh units and ``60 - N`` burned
  units (the centre slot is always fresh and never a source, exactly as in the
  vendor);
* every source unit is consumed at most once, and the source graph is an
  acyclic in-forest rooted at fresh units with chain depth
  ``<= max_shuffle_depth`` (default 2);
* strict mode (default) requires every fresh unit to be consumed.  This forces
  the identities

      feed        = 1 + 4*N
      depth2 edges = 60 - 2*N          (N <= 30)

  For ``N > 30`` there are not enough burned units to consume every fresh unit;
  exactly ``2*N - 60`` fresh units are discharged after a single cycle.  That is
  economically wasteful, so it is gated behind
  ``allow_single_cycle_discharge`` and produces 0 depth-2 edges.

The geometry (``ORBIT_UNITS``, representative source coordinates, rotation
conventions) is imported verbatim from the vendor so that a feed-121 pattern
round-trips byte-identically through this class.

**Fresh-type alphabet (2026-08-17).**  The vendor ``_pair_batches`` hard-codes a
two-type feed (``"TYPEA_TYPEB"``).  Commercial reloads grade the radial
reactivity with three or more fresh types, so :func:`case_batches` accepts an
``"A_B"``, ``"A_B_C"``, ... up to a :data:`MAX_FRESH_TYPES`-member case string
and every batch operator below is written over an arbitrary-length alphabet.  A
two-member case string returns exactly the vendor's 2-tuple and no operator
changes its random draws, so **every 2-type path is byte-identical** (pinned by
``tests/test_triple_type.py``).  The genome's structural invariants (the 1 + 4N
feed grid, strict consumption, the depth cap) are batch-label-blind and are
therefore untouched by the alphabet growth.

**3 -> 5 types (2026-08-17, same day).**  The cap was raised to five (the
operator directive "3~5종 그물망").  Nothing but the cap and the operator
docstrings moved: every routine below already ranged over ``batches`` rather
than over a hard-coded ``(a, b)``/``(a, b, c)``, so the 4- and 5-type paths are
the 3-type paths with a longer alphabet.  :func:`graded_morph` additionally
accepts an explicit ``donor``/``target`` so a caller can convert units between
*any* ordered pair of the alphabet; its default draw is unchanged, which is what
keeps the 3-type move stream identical too.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Sequence

from ..vendor.masterrl.domain import FuelItem, Pattern, SLOTS
from ..vendor.masterrl.ga import (
    FRESH_UNIT_COUNT,
    GenomeError,
    MOVABLE_UNIT_COUNT,
    ORBIT_UNITS,
    OrbitUnit,
    _coord_slot,
    _pair_batches,
    _shuffle_rotation,
    _SLOT_TO_UNIT,
    sample_move_count,
)


# --------------------------------------------------------------------------
# Fresh-type alphabet
# --------------------------------------------------------------------------

#: Largest fresh-type alphabet a case string may declare.  Two is the historical
#: pair; three is the graded-reactivity case (commercial practice); five is the
#: operator-directed ceiling for the graded mesh ("3~5종 그물망").  The cap is a
#: guard against a mis-parsed case id, not a physics limit — the genome, the
#: %LPD_SHF encoding and the store schema are all alphabet-agnostic.  Raising it
#: further needs only this constant plus the cond-schema fraction block width
#: (:data:`lpopt.model.featurize.MAX_FRESH_TYPES`), which is a *retrain*.
MAX_FRESH_TYPES = 5


def case_batches(case: str) -> tuple[str, ...]:
    """Fresh-type alphabet of a case id: ``"A_B"``, ``"A_B_C"``, ... up to 5.

    A two-member case returns exactly what the vendor ``_pair_batches`` returns
    (same order, same tuple), so every 2-type caller is unchanged.  Three to
    :data:`MAX_FRESH_TYPES` members are accepted for a graded core.  Duplicate
    members are rejected: a repeated type would silently shrink the effective
    alphabet of ``_batch_flip`` and make ``case_pair`` ambiguous against the
    store.
    """

    parts = case.split("_")
    if len(parts) == 2 and all(parts):
        return _pair_batches(case)          # vendor path, byte-identical
    if not (2 <= len(parts) <= MAX_FRESH_TYPES) or not all(parts):
        raise GenomeError(
            f"case {case!r} must be 'TYPEA_TYPEB[_TYPEC...]' "
            f"(2..{MAX_FRESH_TYPES} non-empty members)"
        )
    if len(set(parts)) != len(parts):
        raise GenomeError(f"case {case!r} repeats a fresh type")
    return tuple(parts)


# --------------------------------------------------------------------------
# Feed / batch arithmetic identities (plan sec. 1-2)
# --------------------------------------------------------------------------


def feed_from_fresh_units(n_fresh: int) -> int:
    """Reachable full-core feed for ``n_fresh`` fresh orbit units: ``1 + 4N``."""

    return 1 + 4 * int(n_fresh)


def fresh_units_from_feed(feed: int) -> int:
    """Inverse of :func:`feed_from_fresh_units`; rejects off-grid feeds."""

    if (int(feed) - 1) % 4 != 0:
        raise GenomeError(f"feed {feed} is not on the 1 + 4N grid")
    n = (int(feed) - 1) // 4
    if not (0 <= n <= MOVABLE_UNIT_COUNT):
        raise GenomeError(f"feed {feed} implies {n} fresh units, out of range")
    return n


def depth2_edges_for_fresh_units(n_fresh: int) -> int:
    """Strict depth-2 orbit-unit edge count for ``N <= 30``: ``60 - 2N``."""

    return 60 - 2 * int(n_fresh)


def once_burned_count(feed: int) -> int:
    """Full-core once-burned assembly count: ``F - 1``."""

    return int(feed) - 1


def twice_burned_count(feed: int) -> int:
    """Full-core twice-burned assembly count: ``242 - 2F``."""

    return 242 - 2 * int(feed)


# --------------------------------------------------------------------------
# Genome
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralOrbitGenome:
    """A reachable-feed quarter core as fresh-unit set + burned source chains."""

    fresh: tuple[tuple[OrbitUnit, str], ...]         # N (unit, batch), sorted
    wiring: tuple[tuple[OrbitUnit, OrbitUnit], ...]  # (burned <- source), sorted
    center_batch: str
    max_shuffle_depth: int = 2
    allow_single_cycle_discharge: bool = False

    # --------------------------------------------------------------- derived

    @property
    def n_fresh(self) -> int:
        return len(self.fresh)

    @property
    def feed(self) -> int:
        return feed_from_fresh_units(len(self.fresh))

    @property
    def fresh_units(self) -> frozenset[OrbitUnit]:
        return frozenset(unit for unit, _ in self.fresh)

    @property
    def burned_units(self) -> frozenset[OrbitUnit]:
        return frozenset(unit for unit, _ in self.wiring)

    @property
    def depth2_edge_count(self) -> int:
        """Number of burned units that shuffle from another burned unit."""

        fresh_set = self.fresh_units
        return sum(1 for _, source in self.wiring if source not in fresh_set)

    @property
    def unconsumed_fresh_units(self) -> frozenset[OrbitUnit]:
        consumed = {source for _, source in self.wiring}
        return frozenset(self.fresh_units - consumed)

    @property
    def batch_counts(self) -> dict[str, int]:
        """``{batch: fresh-unit count}`` (the centre slot is NOT an orbit unit).

        The radial-grading readout of a genome: a 2-type core has two entries, a
        graded 3-type core three.  Unit counts, not assemblies — multiply by the
        unit multiplicity for the full-core feed split.
        """

        counts: dict[str, int] = {}
        for _, batch in self.fresh:
            counts[batch] = counts.get(batch, 0) + 1
        return counts

    # ------------------------------------------------------------- geometry

    def _depths(self) -> dict[OrbitUnit, int]:
        """Depth of every burned unit (edges to its fresh root).

        Raises :class:`GenomeError` on a cycle or a burned unit with no source.
        Fresh units are roots at depth 0 and omitted from the result.
        """

        src = dict(self.wiring)
        fresh_set = self.fresh_units
        depth: dict[OrbitUnit, int] = {}
        for start in src:
            chain: list[OrbitUnit] = []
            node = start
            while node not in depth:
                if node in fresh_set:
                    depth[node] = 0
                    break
                if node not in src:
                    raise GenomeError(f"burned unit {node} has no source")
                if node in chain:
                    raise GenomeError(
                        f"source chain contains a cycle at unit {node}"
                    )
                chain.append(node)
                node = src[node]
            base = depth[node]
            for member in reversed(chain):
                base += 1
                depth[member] = base
        return {unit: d for unit, d in depth.items() if unit not in fresh_set}

    # ------------------------------------------------------------- validate

    def validate(self) -> None:
        if not self.center_batch or not isinstance(self.center_batch, str):
            raise GenomeError("centre batch must be a non-empty batch name")
        if self.max_shuffle_depth < 1:
            raise GenomeError("max_shuffle_depth must be >= 1")

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
            raise GenomeError(
                "fresh and burned units must cover all 60 movable units"
            )
        if not fresh_set:
            raise GenomeError("a core needs at least one fresh unit")

        # Every source is consumed at most once.
        if len(sources) != len(set(sources)):
            raise GenomeError(
                "a source unit is consumed by more than one burned unit"
            )

        # Acyclic in-forest rooted at fresh units, chain depth within the cap.
        depths = self._depths()
        max_depth = max(depths.values(), default=0)
        if max_depth > self.max_shuffle_depth:
            raise GenomeError(
                f"source chain depth {max_depth} exceeds the cap "
                f"{self.max_shuffle_depth}"
            )

        # Strict consumption / feed identities.
        n = len(self.fresh)
        consumed = set(sources)
        unconsumed_fresh = fresh_set - consumed
        depth2 = sum(1 for source in sources if source in burned_set)
        if n <= FRESH_UNIT_COUNT:
            if unconsumed_fresh:
                raise GenomeError(
                    "strict genome requires every fresh unit consumed; "
                    f"{sorted(unconsumed_fresh)} are not"
                )
            expected = depth2_edges_for_fresh_units(n)
            if depth2 != expected:
                raise GenomeError(
                    f"feed {self.feed} (N={n}) requires exactly {expected} "
                    f"depth-2 edges, found {depth2}"
                )
        else:
            if not self.allow_single_cycle_discharge:
                raise GenomeError(
                    f"feed {self.feed} (N={n} > {FRESH_UNIT_COUNT}) discharges "
                    "fuel after one cycle; set allow_single_cycle_discharge"
                )
            expected_unconsumed = 2 * n - MOVABLE_UNIT_COUNT
            if len(unconsumed_fresh) != expected_unconsumed:
                raise GenomeError(
                    f"N={n} > {FRESH_UNIT_COUNT} requires exactly "
                    f"{expected_unconsumed} unconsumed fresh units, found "
                    f"{len(unconsumed_fresh)}"
                )
            if depth2 != 0:
                raise GenomeError(
                    "single-cycle-discharge genome must have 0 depth-2 edges"
                )

    # ------------------------------------------------------------- compile

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
            raise GenomeError("orbit units failed to cover every slot")
        return Pattern(tuple(items))  # type: ignore[arg-type]

    @classmethod
    def from_pattern(
        cls,
        pattern: Pattern,
        *,
        max_shuffle_depth: int = 2,
        allow_single_cycle_discharge: bool = False,
    ) -> "GeneralOrbitGenome":
        """Parse a pattern, resolving depth-1 and depth-2 source chains.

        This replaces the vendor's depth-1-only rejection with chain
        resolution: a shuffle card's source may point at a burned unit's
        representative coordinate.  Acyclicity, the depth cap and the strict
        consumption invariant are enforced by :meth:`validate`.
        """

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
            arm_sources = {(arm.restart, arm.x, arm.y) for arm in arms}
            if len(arm_sources) != 1:
                raise GenomeError(
                    f"axis twin pair {unit.unit} arms shuffle from different "
                    f"sources {sorted(arm_sources)}"  # type: ignore[type-var]
                )
            for slot, arm in zip(unit.slots, arms, strict=True):
                expected = _shuffle_rotation(slot)
                if arm.rotation != expected:
                    raise GenomeError(
                        f"shuffle card at slot {slot} uses rotation "
                        f"{arm.rotation}; the seed convention is {expected}"
                    )
            restart, x, y = next(iter(arm_sources))
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
                    f"({x}, {y}); axis units are referenced through their "
                    "horizontal representative"
                )
            wiring.append((burned, _SLOT_TO_UNIT[slot]))

        assert centre.batch is not None
        genome = cls(
            fresh=tuple(sorted(fresh)),
            wiring=tuple(sorted(wiring)),
            center_batch=centre.batch,
            max_shuffle_depth=max_shuffle_depth,
            allow_single_cycle_discharge=allow_single_cycle_discharge,
        )
        genome.validate()
        return genome


# --------------------------------------------------------------------------
# Mutation (closed operators)
# --------------------------------------------------------------------------


def _finalize(
    fresh: dict[OrbitUnit, str],
    wiring: dict[OrbitUnit, OrbitUnit],
    template: GeneralOrbitGenome,
) -> GeneralOrbitGenome | None:
    """Build a candidate from edit dicts; ``None`` when it fails validate."""

    candidate = replace(
        template,
        fresh=tuple(sorted(fresh.items())),
        wiring=tuple(sorted(wiring.items())),
    )
    try:
        candidate.validate()
    except GenomeError:
        return None
    return candidate


def _consumer_map(genome: GeneralOrbitGenome) -> dict[OrbitUnit, OrbitUnit]:
    """source unit -> the (unique) burned unit that consumes it."""

    return {source: burned for burned, source in genome.wiring}


def _rewire_swap(genome: GeneralOrbitGenome, rng: random.Random) -> GeneralOrbitGenome:
    """Exchange the sources of two burned units (any feed)."""

    if len(genome.wiring) < 2:
        return genome
    wiring = list(genome.wiring)
    for _ in range(8):
        i, j = rng.sample(range(len(wiring)), 2)
        (b1, s1), (b2, s2) = wiring[i], wiring[j]
        if s1 == s2:
            continue
        edit = dict(genome.wiring)
        edit[b1], edit[b2] = s2, s1
        candidate = _finalize(dict(genome.fresh), edit, genome)
        if candidate is not None and candidate != genome:
            return candidate
    return genome


def _fresh_relocate(
    genome: GeneralOrbitGenome, rng: random.Random
) -> GeneralOrbitGenome:
    """Swap the fresh/burned roles of one fresh unit A and one burned unit B.

    Generalises the vendor operator: it handles an *unconsumed* fresh A (feed
    > 121) and a *consumed* burned B (depth-2 chains) without the vendor's
    ``next(...)`` StopIteration crash, by swapping the consumer of A onto B and
    the consumer of B onto A.  Every candidate is gated by validate().
    """

    if not genome.fresh or not genome.wiring:
        return genome
    fresh_batches = dict(genome.fresh)
    source_of = dict(genome.wiring)
    consumer = _consumer_map(genome)
    fresh_units = sorted(fresh_batches)
    burned_units = sorted(source_of)

    for _ in range(12):
        a = rng.choice(fresh_units)
        b = rng.choice(burned_units)
        source_b = source_of[b]

        new_fresh = dict(fresh_batches)
        batch_a = new_fresh.pop(a)
        new_fresh[b] = batch_a

        new_wiring = dict(genome.wiring)
        del new_wiring[b]  # B becomes fresh: drop its out-edge.

        if source_b == a:
            # B shuffled from A directly; the edge simply reverses.
            new_wiring[a] = b
        else:
            c = consumer.get(a)  # A's consumer (may be None when unconsumed)
            d = consumer.get(b)  # B's consumer (may be None)
            new_wiring[a] = source_b  # A inherits B's freed source.
            if c is not None:
                new_wiring[c] = b     # A's consumer now consumes fresh B.
            if d is not None:
                new_wiring[d] = a     # B's consumer now consumes burned A.

        candidate = _finalize(new_fresh, new_wiring, genome)
        if candidate is not None and candidate != genome:
            return candidate
    return genome


def _batch_flip(
    genome: GeneralOrbitGenome, rng: random.Random, batches: Sequence[str]
) -> GeneralOrbitGenome:
    """Flip one fresh unit's (or the centre's) batch label."""

    if len(batches) < 2:
        return genome
    targets: list[object] = sorted(unit for unit, _ in genome.fresh)
    targets.append("center")
    target = rng.choice(targets)
    if target == "center":
        others = [batch for batch in batches if batch != genome.center_batch]
        if not others:
            return genome
        return replace(genome, center_batch=rng.choice(others))
    fresh = dict(genome.fresh)
    others = [batch for batch in batches if batch != fresh[target]]
    if not others:
        return genome
    fresh[target] = rng.choice(others)
    return replace(genome, fresh=tuple(sorted(fresh.items())))


def _batch_swap(
    genome: GeneralOrbitGenome, rng: random.Random, batches: Sequence[str]
) -> GeneralOrbitGenome:
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


def graded_morph(
    genome: GeneralOrbitGenome,
    rng: random.Random,
    batches: Sequence[str],
    *,
    fraction: float = 0.34,
    donor: str | None = None,
    target: str | None = None,
) -> GeneralOrbitGenome:
    """Convert a radial slice of one fresh type's units to another type.

    This is the cold-start bridge from a 2-type elite into the graded (3..5-type)
    space: an optimized ``A/B`` core is already radially graded, so the cheapest
    way to seed an ``A/B/C[/D/E]`` campaign is to re-label a contiguous-in-radius
    slice of the donor's units to the under-represented type rather than to
    re-randomize the whole board (which throws away the elite's structure).

    The operator converts between **any ordered pair of the alphabet**.  Pass
    ``donor`` / ``target`` to name that pair explicitly (a directed re-grade —
    e.g. shave the hot type onto the mid type of a 5-type mesh); leave them
    ``None`` for the default draw, which is what ``mutate`` uses:

    * the donor is the *most* populated batch that has >= 2 units;
    * the target is the *least* populated member of ``batches`` (an absent type
      counts as zero, so a 2-type parent morphs straight onto the third type, a
      3-type onto the fourth, and so on — repeated morphs walk the alphabet up);
    * the converted slice is the ``ceil(fraction * donor units)`` donor units at
      one radial extreme (inner or outer, drawn), which keeps the re-label a
      *grading* move instead of a scatter.

    The default draw is unchanged from the 3-type build, so the 3-type move
    stream is identical under the widened cap.

    Batch labels are structurally inert (:meth:`GeneralOrbitGenome.validate`
    never reads them beyond the non-empty check), so the feed, the wiring and the
    depth-2 count are preserved exactly.  Returns the genome unchanged when the
    alphabet has fewer than 3 members (a pair has nowhere to grade *to*, which is
    what keeps the 2-type move stream byte-identical), when a named donor/target
    is not usable, or when no donor slice exists.
    """

    if len(batches) < 3 or not genome.fresh:
        return genome
    counts = genome.batch_counts
    if donor is None:
        donors = [b for b, n in counts.items() if n >= 2]
        if not donors:
            return genome
        donor = max(sorted(donors), key=lambda b: counts[b])
    elif counts.get(donor, 0) < 2:
        return genome                       # named donor cannot spare a unit
    if target is None:
        target = min(sorted(batches), key=lambda b: counts.get(b, 0))
    elif target not in batches:
        return genome
    if target == donor:
        return genome

    units = sorted(unit for unit, batch in genome.fresh if batch == donor)
    by_radius = sorted(units, key=lambda u: (ORBIT_UNITS[u].radius, u))
    k = max(1, min(len(units) - 1, math.ceil(float(fraction) * len(units))))
    slice_ = by_radius[:k] if rng.random() < 0.5 else by_radius[-k:]

    fresh = dict(genome.fresh)
    for unit in slice_:
        fresh[unit] = target
    candidate = replace(genome, fresh=tuple(sorted(fresh.items())))
    return candidate if candidate != genome else genome


def _remove_fresh_strict(
    genome: GeneralOrbitGenome, rng: random.Random
) -> GeneralOrbitGenome | None:
    """N -> N-1 (feed -4) for a strict core: reform 3 length-2 chains into 2.

    A length-2 chain is a fresh unit consumed by an unconsumed depth-1 tail.
    Taking three such chains ``(f,b)`` and re-seating one fresh unit as the
    twice-burned tail of another chain adds exactly two depth-2 edges.
    """

    consumer = _consumer_map(genome)
    consumed = set(consumer)
    fresh_batches = dict(genome.fresh)
    length2 = [
        (f, consumer[f])
        for f in fresh_batches
        if f in consumer and consumer[f] not in consumed
    ]
    if len(length2) < 3:
        return None
    (fa, ba), (fb, bb), (fc, bc) = rng.sample(length2, 3)

    new_fresh = dict(fresh_batches)
    del new_fresh[fc]
    new_wiring = dict(genome.wiring)
    new_wiring[fc] = ba   # fc (now burned) shuffles from ba -> depth 2
    new_wiring[bc] = bb   # bc re-seats onto bb -> depth 2
    return _finalize(new_fresh, new_wiring, genome)


def _remove_fresh_discharge(
    genome: GeneralOrbitGenome, rng: random.Random
) -> GeneralOrbitGenome | None:
    """N -> N-1 for a discharge core (N > 30): merge two discharged units."""

    consumed = {source for _, source in genome.wiring}
    fresh_batches = dict(genome.fresh)
    unconsumed = [unit for unit in fresh_batches if unit not in consumed]
    if len(unconsumed) < 2:
        return None
    a, g = rng.sample(unconsumed, 2)
    new_fresh = dict(fresh_batches)
    del new_fresh[a]
    new_wiring = dict(genome.wiring)
    new_wiring[a] = g   # a (now burned) shuffles from the still-fresh g
    return _finalize(new_fresh, new_wiring, genome)


def _remove_fresh_unit(
    genome: GeneralOrbitGenome, rng: random.Random
) -> GeneralOrbitGenome:
    """Delete one fresh unit (feed -4), keeping the core valid and strict."""

    if len(genome.fresh) <= 1:
        return genome
    if len(genome.fresh) <= FRESH_UNIT_COUNT:
        candidate = _remove_fresh_strict(genome, rng)
    else:
        candidate = _remove_fresh_discharge(genome, rng)
    if candidate is not None and candidate != genome:
        return candidate
    return genome


def _add_fresh_strict(
    genome: GeneralOrbitGenome, rng: random.Random, batches: Sequence[str]
) -> GeneralOrbitGenome | None:
    """N -> N+1 (feed +4) for a strict core: split 2 length-3 chains into 3.

    Promote the twice-burned tail of one length-3 chain to fresh and re-seat
    the twice-burned tail of another chain onto it; both moves remove a depth-2
    edge, restoring strict consumption at the higher feed.
    """

    fresh_set = genome.fresh_units
    source_of = dict(genome.wiring)
    consumer = _consumer_map(genome)
    consumed = set(consumer)
    length3 = [
        (tail, source_of[tail])
        for tail, source in genome.wiring
        if source not in fresh_set and tail not in consumed
    ]
    if len(length3) < 2:
        return None
    (t1, m1), (t2, m2) = rng.sample(length3, 2)
    if len({t1, m1, t2, m2}) < 4:
        return None

    new_fresh = dict(genome.fresh)
    new_fresh[t1] = rng.choice(list(batches)) if batches else genome.center_batch
    new_wiring = dict(genome.wiring)
    del new_wiring[t1]    # t1 promoted to fresh: drop its out-edge
    new_wiring[t2] = t1   # t2 re-seats onto the now-fresh t1 -> depth 1
    return _finalize(new_fresh, new_wiring, genome)


def _add_fresh_discharge(
    genome: GeneralOrbitGenome, rng: random.Random, batches: Sequence[str]
) -> GeneralOrbitGenome | None:
    """N -> N+1 for a discharge core: promote a depth-1 tail to fresh."""

    if not genome.allow_single_cycle_discharge:
        return None
    fresh_set = genome.fresh_units
    consumed = {source for _, source in genome.wiring}
    tails = [
        burned
        for burned, source in genome.wiring
        if source in fresh_set and burned not in consumed
    ]
    if not tails:
        return None
    b = rng.choice(tails)
    new_fresh = dict(genome.fresh)
    new_fresh[b] = rng.choice(list(batches)) if batches else genome.center_batch
    new_wiring = dict(genome.wiring)
    del new_wiring[b]   # b promoted to fresh; its old source becomes unconsumed
    return _finalize(new_fresh, new_wiring, genome)


def _add_fresh_unit(
    genome: GeneralOrbitGenome, rng: random.Random, batches: Sequence[str]
) -> GeneralOrbitGenome:
    """Add one fresh unit (feed +4), keeping the core valid and strict."""

    if len(genome.fresh) >= MOVABLE_UNIT_COUNT:
        return genome
    candidate: GeneralOrbitGenome | None = None
    if len(genome.fresh) < FRESH_UNIT_COUNT:
        candidate = _add_fresh_strict(genome, rng, batches)
    if candidate is None:
        candidate = _add_fresh_discharge(genome, rng, batches)
    if candidate is not None and candidate != genome:
        return candidate
    return genome


def _one_move(
    genome: GeneralOrbitGenome,
    rng: random.Random,
    fresh_relocate_prob: float,
    batch_prob: float,
    feed_move_prob: float,
    batches: Sequence[str],
) -> GeneralOrbitGenome:
    draw = rng.random()
    if draw < fresh_relocate_prob:
        return _fresh_relocate(genome, rng)
    if draw < fresh_relocate_prob + batch_prob:
        pick = rng.random()
        if pick < 0.5:
            return _batch_flip(genome, rng, batches)
        # The graded morph exists only for a >=3-type alphabet, and it takes the
        # UPPER half of the swap band.  A 2-type alphabet therefore draws exactly
        # one rng.random() and dispatches flip/swap on the same 0.5 threshold as
        # before — the 2-type move stream is byte-identical.
        if len(batches) >= 3 and pick >= 0.75:
            return graded_morph(genome, rng, batches)
        return _batch_swap(genome, rng, batches)
    if draw < fresh_relocate_prob + batch_prob + feed_move_prob:
        if rng.random() < 0.5:
            return _remove_fresh_unit(genome, rng)
        return _add_fresh_unit(genome, rng, batches)
    return _rewire_swap(genome, rng)


def mutate(
    genome: GeneralOrbitGenome,
    rng: random.Random,
    n_moves: int,
    *,
    fresh_relocate_prob: float = 0.30,
    batch_prob: float = 0.15,
    feed_move_prob: float = 0.15,
    batches: Sequence[str] | None = None,
) -> GeneralOrbitGenome:
    """Apply ``n_moves`` closed operators; guaranteed to change the genome."""

    if n_moves < 1:
        raise ValueError("n_moves must be >= 1")
    if batches is None:
        batches = sorted(
            {batch for _, batch in genome.fresh} | {genome.center_batch}
        )
    child = genome
    for _ in range(n_moves):
        child = _one_move(
            child, rng, fresh_relocate_prob, batch_prob, feed_move_prob, batches
        )
    guard = 0
    while child == genome:
        child = _one_move(
            child, rng, fresh_relocate_prob, batch_prob, feed_move_prob, batches
        )
        guard += 1
        if guard > 64:
            # A move sequence can cancel itself; force a structural move.
            child = _rewire_swap(genome, rng)
            if child == genome:
                raise GenomeError("mutation failed to change the genome")
            break
    return child


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------


def random_genome(
    rng: random.Random,
    pair: str,
    n_fresh_units: int = FRESH_UNIT_COUNT,
    *,
    max_shuffle_depth: int = 2,
    allow_single_cycle_discharge: bool = False,
) -> GeneralOrbitGenome:
    """Uniform random valid genome with ``n_fresh_units`` fresh units.

    Sources are assigned in residence layers (the MOCHA ``random_rule``
    concept): a depth-1 layer perfectly matches burned units to fresh roots,
    and a depth-2 layer seats the remaining burned units onto depth-1 units so
    every chain terminates at a fresh root within the depth cap.  At N=30 this
    reduces to the vendor's feed-121 perfect matching.

    ``pair`` is a case id over 2..:data:`MAX_FRESH_TYPES` fresh types
    (:func:`case_batches`); fresh units draw uniformly from that alphabet.  A
    2-type case draws from the same 2-tuple as before, so its RNG stream is
    unchanged.
    """

    batches = case_batches(pair)
    n = int(n_fresh_units)
    if not (1 <= n <= MOVABLE_UNIT_COUNT):
        raise GenomeError(f"n_fresh_units {n} out of range 1..{MOVABLE_UNIT_COUNT}")
    if n <= FRESH_UNIT_COUNT and (MOVABLE_UNIT_COUNT - 2 * n) > n:
        raise GenomeError(
            f"N={n} needs {MOVABLE_UNIT_COUNT - 2 * n} depth-2 units but only "
            f"{n} depth-1 units exist; unreachable within depth 2 (need N>=20)"
        )
    discharge = allow_single_cycle_discharge or n > FRESH_UNIT_COUNT

    fresh_units = sorted(rng.sample(range(MOVABLE_UNIT_COUNT), n))
    fresh_set = set(fresh_units)
    burned = [unit for unit in range(MOVABLE_UNIT_COUNT) if unit not in fresh_set]
    rng.shuffle(burned)

    wiring: dict[OrbitUnit, OrbitUnit] = {}
    if n <= FRESH_UNIT_COUNT:
        # Depth-1 layer: perfect matching burned -> fresh (N edges).
        depth1 = burned[:n]
        depth2 = burned[n:]  # 60 - 2N units
        roots = list(fresh_units)
        rng.shuffle(roots)
        for burned_unit, root in zip(depth1, roots, strict=True):
            wiring[burned_unit] = root
        # Depth-2 layer: seat onto distinct depth-1 units.
        pool = list(depth1)
        rng.shuffle(pool)
        for burned_unit, mid in zip(depth2, pool, strict=False):
            wiring[burned_unit] = mid
    else:
        # Discharge: every burned unit shuffles from a distinct fresh root;
        # the surplus 2N-60 fresh units are single-cycle discharged.
        roots = list(fresh_units)
        rng.shuffle(roots)
        for burned_unit, root in zip(burned, roots, strict=False):
            wiring[burned_unit] = root

    fresh = tuple((unit, rng.choice(batches)) for unit in fresh_units)
    genome = GeneralOrbitGenome(
        fresh=fresh,
        wiring=tuple(sorted(wiring.items())),
        center_batch=batches[0],
        max_shuffle_depth=max_shuffle_depth,
        allow_single_cycle_discharge=discharge,
    )
    genome.validate()
    return genome


__all__ = [
    "FRESH_UNIT_COUNT",
    "GeneralOrbitGenome",
    "GenomeError",
    "MAX_FRESH_TYPES",
    "MOVABLE_UNIT_COUNT",
    "ORBIT_UNITS",
    "OrbitUnit",
    "case_batches",
    "depth2_edges_for_fresh_units",
    "feed_from_fresh_units",
    "graded_morph",
    "fresh_units_from_feed",
    "mutate",
    "once_burned_count",
    "random_genome",
    "sample_move_count",
    "twice_burned_count",
]
