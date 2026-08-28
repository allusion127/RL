"""``[search] elite_seed_cases`` — the 3-fresh-type cold-start donor knob.

A graded case ``A_B_C`` matches ZERO store rows by ``case_pair``, so
``CampaignDriver._store_elites`` returns an empty parent set and the
``graded_morph`` operator (which re-labels a radial slice of a 2-type parent onto
the third type) is never reachable from a campaign pool.  The knob names DONOR
case ids whose converged rows join the elite-mutation parent set.

Registered claims, one test each:

1. knob unset -> the parent set is byte-identical to today's (the triple case
   gets nothing, the pair case gets exactly its own rows);
2. knob set   -> the donor pair's converged rows become parents of the TRIPLE
   campaign;
3. the donors do NOT leak into ``_holdout_rows`` (the wave fine-tune gate scores
   this case's own labels or nothing at all);
4. non-converged donor rows are excluded, and naming the campaign's own case is a
   no-op (no double-counting);
5. end-to-end: ``build_pool`` over the donor parents actually reaches the third
   fresh type — i.e. the cold start works, not just the plumbing.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from lpopt.config import (
    AcquisitionConfig, CaseConfig, DataConfig, ExtractConfig, FlowConfig, FuelConfig,
    LpoptConfig, MasterConfig, ModelConfig, ProduceConfig, RemoteConfig, SearchConfig,
    VerifyConfig,
)
from lpopt.data.schema import (
    SYM_CLASS, CanonicalRecord, compute_record_id, pack_pattern,
)
from lpopt.data.store import StoreWriter
from lpopt.search.campaign import CampaignDriver
from lpopt.search.construct import CAMPAIGN_DECK_KNOBS, CaseContext, build_pool
from lpopt.search.genome import random_genome
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction

HOT = "P6253Z1G06N24"
MID = "P6253Z2G08N16"
COLD = "P6253Z2G10N24"
PAIR = f"{HOT}_{COLD}"
TRIPLE = f"{HOT}_{MID}_{COLD}"
FEED = 125
N_FRESH = (FEED - 1) // 4


class _FakeModel:
    def predict(self, patterns, case, cell=0.0):
        patterns = list(patterns)
        n = len(patterns)
        mean = np.tile([1.7, 1500.0, 2.1, 730.0, 0.03, 70.0, 77.0], (n, 1))
        std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, 1.0, 1.0], (n, 1))
        return SurrogatePrediction(mean, std.copy(), std.copy())

    def predict_convergence(self, patterns, case, cell=0.0):
        return np.ones(len(list(patterns)), dtype=float)

    def position_values(self, pattern, case, cell=0.0):
        return None


def _pair_pattern(seed: int):
    return random_genome(random.Random(seed), PAIR, N_FRESH).to_pattern()


def _record(pattern, *, f_r: float, converged: bool = True) -> CanonicalRecord:
    rid = compute_record_id(pattern.canonical(), "paramA", PAIR, CAMPAIGN_DECK_KNOBS)
    return CanonicalRecord(
        record_id=rid, dataset="P", campaign="fpcamp_minfr_hgd569_f125",
        stratum="min_fr", generator="local", parent_record_id=None,
        case_pair=PAIR, feed=FEED, n_batches=2, depth2_edges=0,
        e_core=5.694, e_split=0.184, library_id="paramA", sym_class=SYM_CLASS,
        pattern=pack_pattern(pattern),
        f_r=f_r, f_q=2.03, cbc_max=1565.0, cbc_boc=None, cbc_kind="max",
        cyclen=730.9, ao_abs=0.027, cycle_burnup=None, discharge_burnup=None,
        max_assembly_burnup=None, max_pin_burnup=None, eoc_ppm=None, delta_efpd=None,
        n_cycles=11.0, converged=converged, converged_at_cap=False,
        tolerance_margin=0.2, restart_provenance="pair_ecore:X", valid=True,
        failure="", maps_key=None,
    )


def _cfg(tmp_path: Path, store_dir: Path, pair: str,
         donors: tuple[str, ...]) -> LpoptConfig:
    deck = tmp_path / "lpopt.inp"
    deck.write_text("# fake\n", encoding="utf-8")
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(), data=DataConfig(),
        case=CaseConfig(mode="fixed", pair=pair, feed=FEED),
        fuel=FuelConfig(), extract=ExtractConfig(), produce=ProduceConfig(),
        search=SearchConfig(elite_seed_cases=donors, elite_top_k=32),
        acquisition=AcquisitionConfig(budget=8, objective="min_fr_max_cycle"),
        model=ModelConfig(store_dir=str(store_dir), library_id="paramA"),
        source_path=deck,
    )


def _driver(tmp_path: Path, records, pair: str,
            donors: tuple[str, ...] = ()) -> CampaignDriver:
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    StoreWriter(store_dir).write_records(list(records))
    cfg = _cfg(tmp_path, store_dir, pair, donors)
    return CampaignDriver(
        cfg, _FakeModel(), dry_run=True, run_dir=tmp_path / f"run_{pair}_{len(donors)}",
        progress=False, log=lambda m: None, budget=8,
    )


@pytest.fixture(scope="module")
def rows():
    return [_record(_pair_pattern(s), f_r=1.60 + 0.01 * s) for s in range(6)]


# --------------------------------------------------------------------------- #
# 1) default OFF — nothing changes for anybody
# --------------------------------------------------------------------------- #
def test_knob_defaults_off_and_triple_gets_nothing(tmp_path: Path, rows) -> None:
    assert list(SearchConfig().elite_seed_cases) == []
    triple = _driver(tmp_path / "a", rows, TRIPLE)
    assert triple._elite_seed_rows() == []
    assert triple._store_elites() == [], (
        "without the knob a graded 3-type case matches no store row and its elite "
        "pool is empty — this is the defect the knob exists to fix"
    )
    # the 2-type case keeps seeing exactly its own rows, knob or no knob.
    pair = _driver(tmp_path / "b", rows, PAIR)
    assert len({rid for rid, _ in pair._store_elites()}) == len(rows)


# --------------------------------------------------------------------------- #
# 2) knob ON — the donor pair's rows become the triple's parents
# --------------------------------------------------------------------------- #
def test_donor_rows_seed_the_triple(tmp_path: Path, rows) -> None:
    drv = _driver(tmp_path, rows, TRIPLE, donors=(PAIR,))
    elites = drv._store_elites()
    assert {rid for rid, _ in elites} == {r.record_id for r in rows}
    # they arrive as real 2-type boards: every batch label is inside the triple
    # alphabet, so `mutate(batches=<triple>)` can re-label without rejection.
    for _, pat in elites:
        labels = {b for _, b in pat.assignments()} if hasattr(pat, "assignments") \
            else set(pat.batch_feed())
        assert labels <= {HOT, MID, COLD}
        assert MID not in labels, "the donor is a 2-type parent by construction"


# --------------------------------------------------------------------------- #
# 3) donors are parents ONLY — the wave gate's holdout must not absorb them
# --------------------------------------------------------------------------- #
def test_donors_do_not_leak_into_the_holdout(tmp_path: Path, rows) -> None:
    off = _driver(tmp_path / "off", rows, TRIPLE)
    on = _driver(tmp_path / "on", rows, TRIPLE, donors=(PAIR,))
    assert off._holdout_rows() == []
    assert on._holdout_rows() == [], (
        "a donor row is a label of a DIFFERENT case; letting it into the online "
        "fine-tune holdout would silently redefine what that gate measures"
    )
    assert on._case_store_rows(converged=True) == []


# --------------------------------------------------------------------------- #
# 4) non-converged donors excluded; self-naming is a no-op
# --------------------------------------------------------------------------- #
def test_nonconverged_excluded_and_self_naming_is_noop(tmp_path: Path, rows) -> None:
    mixed = list(rows) + [_record(_pair_pattern(99), f_r=9.9, converged=False)]
    drv = _driver(tmp_path / "x", mixed, TRIPLE, donors=(PAIR,))
    assert {rid for rid, _ in drv._store_elites()} == {r.record_id for r in rows}

    # naming the campaign's own case must not double-count its rows.
    own = _driver(tmp_path / "y", rows, PAIR, donors=(PAIR,))
    ids = [rid for rid, _ in own._store_elites()]
    assert len(ids) == len(set(ids)) == len(rows)


# --------------------------------------------------------------------------- #
# 5) end-to-end: the cold start actually REACHES the third type
# --------------------------------------------------------------------------- #
def test_pool_from_donor_parents_reaches_the_third_type(tmp_path: Path, rows) -> None:
    drv = _driver(tmp_path, rows, TRIPLE, donors=(PAIR,))
    ctx = CaseContext(pair=TRIPLE, feed=FEED, library_id="paramA")
    assert ctx.batches == (HOT, MID, COLD)
    pool = build_pool(
        ctx, _FakeModel(), drv._store_elites(), set(), random.Random(5695),
        drv.cfg, wave_index=0, size=400,
    )
    assert pool, "the donor parents must produce a non-empty pool"
    elite_children = [c for c in pool if c.origin == "elite"]
    assert elite_children, "no elite-mutation children were generated from the donors"
    with_mid = [c for c in elite_children if MID in set(c.genome.batch_counts)]
    assert with_mid, (
        "graded_morph never fired: not one elite child carries the third fresh "
        "type, so the 3-type cold start would start from random boards"
    )
    # structural preservation is graded_morph's contract — the morphed children
    # keep the parent's feed exactly.
    assert all(c.genome.n_fresh == N_FRESH for c in with_mid)


# --------------------------------------------------------------------------- #
# 6) `[search] require_all_fresh_types` — the graded-budget guard
#
# A 3-type alphabet legally admits boards that feed only two of its members, and
# such a board is the SAME PHYSICAL CORE as a 2-type board of that sub-alphabet.
# Measured on the real donors at live pool settings, 59.5% of a wave-0 pool was
# 2-type, and an out-of-distribution model ranked those FIRST — so the campaign
# would have spent its MASTER budget re-measuring cores the 2-type campaign at
# the same cell already labelled.  The flag makes the budget buy graded cores.
# --------------------------------------------------------------------------- #
def _pool(drv, ctx, wave=0, size=600):
    return build_pool(ctx, _FakeModel(), drv._store_elites(), set(),
                      random.Random(5695), drv.cfg, wave_index=wave, size=size)


def test_guard_defaults_off_and_pool_is_mixed(tmp_path: Path, rows) -> None:
    from lpopt.config import SearchConfig as _SC
    assert _SC().require_all_fresh_types is False
    drv = _driver(tmp_path, rows, TRIPLE, donors=(PAIR,))
    ctx = CaseContext(pair=TRIPLE, feed=FEED, library_id="paramA")
    assert ctx.require_all_batches is False
    pool = _pool(drv, ctx)
    two = [c for c in pool if MID not in c.genome.batch_counts]
    assert two, ("with the guard off the pool must still admit 2-type boards — "
                 "that is the behaviour the guard exists to change")


def test_guard_on_admits_only_full_alphabet_boards(tmp_path: Path, rows) -> None:
    drv = _driver(tmp_path, rows, TRIPLE, donors=(PAIR,))
    ctx = CaseContext(pair=TRIPLE, feed=FEED, library_id="paramA",
                      require_all_batches=True)
    pool = _pool(drv, ctx)
    assert pool, "the guard must not empty the pool"
    for c in pool:
        counts = c.genome.batch_counts
        assert all(counts.get(b, 0) > 0 for b in (HOT, MID, COLD)), (
            f"guard admitted a board missing a fresh type: {dict(counts)}")
    # feed is still exactly preserved — the guard filters, it does not rewrite.
    assert all(c.genome.n_fresh == N_FRESH for c in pool)


def test_guard_is_a_noop_for_a_two_type_case(tmp_path: Path, rows) -> None:
    """A pair board always feeds both members, so the flag must change nothing."""
    drv = _driver(tmp_path, rows, PAIR)
    off = CaseContext(pair=PAIR, feed=FEED, library_id="paramA")
    on = CaseContext(pair=PAIR, feed=FEED, library_id="paramA",
                     require_all_batches=True)
    a = [c.record_id for c in _pool(drv, off)]
    b = [c.record_id for c in _pool(drv, on)]
    assert a == b, "the guard perturbed a 2-type pool (rng draw order must match)"


def test_guard_reaches_the_context_from_the_deck(tmp_path: Path, rows) -> None:
    """The knob is plumbed deck -> SearchConfig -> CaseContext, not just present."""
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    StoreWriter(store_dir).write_records(list(rows))
    cfg = _cfg(tmp_path, store_dir, TRIPLE, (PAIR,))
    object.__setattr__(cfg.search, "require_all_fresh_types", True)
    drv = CampaignDriver(cfg, _FakeModel(), dry_run=True,
                         run_dir=tmp_path / "run_plumb", progress=False,
                         log=lambda m: None, budget=8)
    assert drv._case_context().require_all_batches is True
