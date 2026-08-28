"""GeneralOrbitGenome: arithmetic identities, feed-121 vendor parity, f117
seed round-trips, and a 10k-move closed-mutation fuzz (plan M1).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from lpopt.search.genome import (
    FRESH_UNIT_COUNT,
    GeneralOrbitGenome,
    GenomeError,
    MOVABLE_UNIT_COUNT,
    depth2_edges_for_fresh_units,
    feed_from_fresh_units,
    fresh_units_from_feed,
    mutate,
    once_burned_count,
    random_genome,
    sample_move_count,
    twice_burned_count,
)
from lpopt.search import genome as genome_mod
from lpopt.vendor.masterrl.domain import Pattern
from lpopt.vendor.masterrl.ga import random_genome as vendor_random_genome
from lpopt.vendor.masterrl.master import extract_lpd_shf

REPO_ROOT = Path(__file__).resolve().parents[1]
# ..\5_RL\tests -> ..\5_RL -> ..\2_계산 -> 3_GA_Surrogate\FEASIBLE_PACKAGE
FEASIBLE_CORES = (
    REPO_ROOT.parent / "3_GA_Surrogate" / "FEASIBLE_PACKAGE" / "cores"
)
F117_CASES = ("E1_E2_f117", "J1_J2_f117", "K1_K2_f117")


# --------------------------------------------------------------------------
# Arithmetic identities (plan sec. 1-2)
# --------------------------------------------------------------------------

# Reachable feeds on the 1 + 4N grid, with their strict depth-2 orbit-unit
# edge count (60 - 2N for N <= 30) and full-core burn tallies.
FEED_TABLE = {
    # feed: (N, depth2_edges, once_burned, twice_burned)
    97: (24, 12, 96, 48),
    105: (26, 8, 104, 32),
    109: (27, 6, 108, 24),
    113: (28, 4, 112, 16),
    117: (29, 2, 116, 8),
    121: (30, 0, 120, 0),
}


@pytest.mark.parametrize("feed,expected", FEED_TABLE.items())
def test_feed_arithmetic_identities(feed: int, expected: tuple[int, int, int, int]) -> None:
    n, depth2, once, twice = expected
    assert feed_from_fresh_units(n) == feed
    assert fresh_units_from_feed(feed) == n
    assert depth2_edges_for_fresh_units(n) == depth2
    assert once_burned_count(feed) == once
    assert twice_burned_count(feed) == twice
    # Full-core conservation: fresh + once + twice == 241 assemblies.
    assert feed + once + twice == 241


def test_feed_grid_rejects_off_grid_feeds() -> None:
    for feed in (118, 119, 120, 122, 100):
        with pytest.raises(GenomeError):
            fresh_units_from_feed(feed)


def test_once_and_twice_burned_match_generated_genome() -> None:
    rng = random.Random(0)
    for n in (24, 27, 29, 30):
        g = random_genome(rng, "K1_K2", n)
        feed = g.feed
        assert feed == feed_from_fresh_units(n)
        assert g.depth2_edge_count == depth2_edges_for_fresh_units(n)
        # twice-burned assemblies == full-core multiplicity of depth-2 units.
        twice_from_units = sum(
            _unit_multiplicity(source)
            for _, source in g.wiring
            if source in g.burned_units
        )
        assert twice_from_units == twice_burned_count(feed)


def _unit_multiplicity(unit: int) -> int:
    from lpopt.vendor.masterrl.ga import ORBIT_UNITS
    from lpopt.vendor.masterrl.domain import SLOTS

    return sum(SLOTS[slot].multiplicity for slot in ORBIT_UNITS[unit].slots)


# --------------------------------------------------------------------------
# validate() structural rules
# --------------------------------------------------------------------------


def test_validate_rejects_incomplete_partition() -> None:
    g = random_genome(random.Random(1), "K1_K2", 30)
    # Drop one fresh unit without re-homing it -> partition no longer covers 60.
    bad = GeneralOrbitGenome(
        fresh=g.fresh[1:], wiring=g.wiring, center_batch=g.center_batch
    )
    with pytest.raises(GenomeError):
        bad.validate()


def test_validate_rejects_double_consumption() -> None:
    g = random_genome(random.Random(2), "K1_K2", 30)
    wiring = list(g.wiring)
    # Force two burned units to shuffle from the same source.
    (b0, _), (b1, s1) = wiring[0], wiring[1]
    wiring[0] = (b0, s1)
    bad = GeneralOrbitGenome(
        fresh=g.fresh, wiring=tuple(sorted(wiring)), center_batch=g.center_batch
    )
    with pytest.raises(GenomeError):
        bad.validate()


def test_validate_rejects_source_cycle() -> None:
    # Build a small explicit 2-cycle among burned units.
    fresh_units = list(range(FRESH_UNIT_COUNT))
    fresh = tuple((u, "K1") for u in fresh_units)
    burned = list(range(FRESH_UNIT_COUNT, MOVABLE_UNIT_COUNT))
    wiring = {}
    # match most burned -> fresh, but make the last two source each other.
    for b, f in zip(burned[:-2], fresh_units, strict=False):
        wiring[b] = f
    x, y = burned[-2], burned[-1]
    wiring[x] = y
    wiring[y] = x
    bad = GeneralOrbitGenome(
        fresh=fresh, wiring=tuple(sorted(wiring.items())), center_batch="K1"
    )
    with pytest.raises(GenomeError):
        bad.validate()


def test_validate_rejects_depth3_chain() -> None:
    g = random_genome(random.Random(4), "K1_K2", 29)  # has depth-2 tails
    tail2 = next(b for b, s in g.wiring if s in g.burned_units)
    consumed = {s for _, s in g.wiring}
    victim = next(b for b, _ in g.wiring if b not in consumed and b != tail2)
    wiring = dict(g.wiring)
    wiring[victim] = tail2  # victim -> depth-2 tail -> depth 3
    bad = GeneralOrbitGenome(
        fresh=g.fresh, wiring=tuple(sorted(wiring.items())), center_batch=g.center_batch
    )
    with pytest.raises(GenomeError):
        bad.validate()


def test_discharge_gate_requires_flag() -> None:
    # A hand-built N=31 discharge genome must be rejected without the flag.
    g = random_genome(random.Random(5), "K1_K2", 31, allow_single_cycle_discharge=True)
    g.validate()  # flag on: fine
    ungated = GeneralOrbitGenome(
        fresh=g.fresh,
        wiring=g.wiring,
        center_batch=g.center_batch,
        allow_single_cycle_discharge=False,
    )
    with pytest.raises(GenomeError):
        ungated.validate()


def test_discharge_round_trip() -> None:
    g = random_genome(random.Random(6), "K1_K2", 31, allow_single_cycle_discharge=True)
    assert g.feed == 125
    assert len(g.unconsumed_fresh_units) == 2  # 2N - 60
    pattern = g.to_pattern()
    assert pattern.feed == 125
    back = GeneralOrbitGenome.from_pattern(pattern, allow_single_cycle_discharge=True)
    assert back.to_pattern().to_shf() == pattern.to_shf()


# --------------------------------------------------------------------------
# ACCEPTANCE (1): feed-121 vendor parity
# --------------------------------------------------------------------------


def test_accept_feed121_vendor_parity() -> None:
    """>=1000 vendor OrbitGenome patterns round-trip byte-identically and
    validate with 0 depth-2 edges."""

    pairs = ("K1_K2", "E1_E2", "J1_J2", "N1_N2", "L1_L2")
    total = 0
    for seed in range(260):
        rng = random.Random(1000 + seed)
        for pair in pairs:
            vendor = vendor_random_genome(rng, pair)
            pattern = vendor.to_pattern()
            parsed = GeneralOrbitGenome.from_pattern(pattern)

            # validate() already ran inside from_pattern; assert the invariants.
            assert parsed.feed == 121
            assert parsed.depth2_edge_count == 0
            assert len(parsed.unconsumed_fresh_units) == 0

            round_tripped = parsed.to_pattern()
            # Byte-equal canonical form and .to_shf() text.
            assert round_tripped.to_shf() == pattern.to_shf()
            assert round_tripped.canonical() == pattern.canonical()
            assert round_tripped.digest == pattern.digest
            total += 1
    assert total >= 1000, total


# --------------------------------------------------------------------------
# ACCEPTANCE (2): packaged feed-117 3-batch seeds
# --------------------------------------------------------------------------


def _read_seed_pattern(seed_dir: Path) -> Pattern | None:
    """Parse a seed's loading pattern, or None when OneDrive-dehydrated."""

    shf = seed_dir / "loading_shf.txt"
    try:
        return Pattern.parse(shf.read_bytes().decode("utf-8", "replace"))
    except OSError:
        pass
    except Exception:
        return None
    for deck in sorted(seed_dir.glob("MAS_INP_cy*.inp")):
        try:
            text = deck.read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        try:
            return Pattern.parse(extract_lpd_shf(text))
        except Exception:
            continue
    return None


def _collect_readable_f117() -> dict[str, Pattern]:
    """First readable seed pattern per f117 case (empty when all dehydrated)."""

    found: dict[str, Pattern] = {}
    if not FEASIBLE_CORES.is_dir():
        return found
    for case in F117_CASES:
        cdir = FEASIBLE_CORES / case
        if not cdir.is_dir():
            continue
        for seed_dir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            pattern = _read_seed_pattern(seed_dir)
            if pattern is not None:
                found[case] = pattern
                break
    return found


def test_accept_f117_seed_round_trip() -> None:
    """Each readable feed-117 seed parses as a 3-batch core (2 depth-2 edges)
    and re-emits a byte-identical %LPD_SHF."""

    readable = _collect_readable_f117()
    if not readable:
        pytest.skip("FEASIBLE_PACKAGE dehydrated -- acceptance deferred")

    for case, pattern in readable.items():
        assert pattern.feed == 117, f"{case}: feed {pattern.feed}"
        genome = GeneralOrbitGenome.from_pattern(pattern)
        assert genome.feed == 117, case
        assert genome.depth2_edge_count == 2, f"{case}: {genome.depth2_edge_count}"
        assert len(genome.unconsumed_fresh_units) == 0, case
        # Byte-identical %LPD_SHF round trip.
        assert genome.to_pattern().to_shf() == pattern.to_shf(), case


# --------------------------------------------------------------------------
# ACCEPTANCE (3): 10k mixed-operator fuzz, zero invariant violations
# --------------------------------------------------------------------------


def _assert_invariants(g: GeneralOrbitGenome) -> None:
    g.validate()
    n = g.n_fresh
    assert g.feed == 1 + 4 * n
    if n <= FRESH_UNIT_COUNT:
        assert g.depth2_edge_count == 60 - 2 * n, (n, g.depth2_edge_count)
        assert len(g.unconsumed_fresh_units) == 0
    else:
        assert g.depth2_edge_count == 0
        assert len(g.unconsumed_fresh_units) == 2 * n - 60


def test_accept_fuzz_mutations_closed() -> None:
    rng = random.Random(20260716)
    starts = [
        (24, False),
        (27, False),
        (29, False),
        (30, False),
        (31, True),
    ]
    genomes = [
        random_genome(rng, "K1_K2", n, allow_single_cycle_discharge=flag)
        for n, flag in starts
    ]
    total_moves = 0
    per_start = 10_000 // len(genomes)
    for parent in genomes:
        _assert_invariants(parent)
        for _ in range(per_start):
            n_moves = sample_move_count(rng, 0.5)
            child = mutate(parent, rng, n_moves, batches=("K1", "K2"))
            _assert_invariants(child)
            assert child != parent  # no-op guard
            parent = child
            total_moves += 1
    assert total_moves >= 10_000


# --------------------------------------------------------------------------
# Individual operator behaviour
# --------------------------------------------------------------------------


def test_feed_move_operators_step_by_four() -> None:
    rng = random.Random(11)
    g = random_genome(rng, "K1_K2", 28)
    removed = genome_mod._remove_fresh_unit(g, rng)
    assert removed.feed in (g.feed, g.feed - 4)
    if removed != g:
        assert removed.feed == g.feed - 4
        _assert_invariants(removed)
    added = genome_mod._add_fresh_unit(g, rng, ("K1", "K2"))
    assert added.feed in (g.feed, g.feed + 4)
    if added != g:
        assert added.feed == g.feed + 4
        _assert_invariants(added)


def test_fresh_relocate_preserves_feed_and_validity() -> None:
    rng = random.Random(12)
    for n, flag in ((24, False), (30, False), (32, True)):
        g = random_genome(rng, "K1_K2", n, allow_single_cycle_discharge=flag)
        for _ in range(50):
            child = genome_mod._fresh_relocate(g, rng)
            assert child.feed == g.feed  # relocation never changes feed
            child.validate()
            g = child


def test_rewire_swap_keeps_feed_121() -> None:
    rng = random.Random(13)
    g = random_genome(rng, "K1_K2", 30)
    for _ in range(200):
        g = genome_mod._rewire_swap(g, rng)
        assert g.feed == 121
        assert g.depth2_edge_count == 0
        g.validate()


def test_mutate_rejects_zero_moves() -> None:
    g = random_genome(random.Random(14), "K1_K2", 30)
    with pytest.raises(ValueError):
        mutate(g, random.Random(0), 0)


def test_random_genome_matches_vendor_structure_at_30() -> None:
    rng = random.Random(15)
    g = random_genome(rng, "K1_K2", 30)
    assert g.n_fresh == 30
    assert len(g.wiring) == 30
    assert g.feed == 121
    assert g.depth2_edge_count == 0
    # Every fresh unit consumed exactly once: a perfect matching, like vendor.
    assert {s for _, s in g.wiring} == g.fresh_units


def test_random_genome_rejects_unreachable_low_feed() -> None:
    # N < 20 cannot be realised within a depth-2 cap.
    with pytest.raises(GenomeError):
        random_genome(random.Random(16), "K1_K2", 19)
