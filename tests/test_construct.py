"""Candidate-pool construction (plan sec. 4.6): proportions, feed pin, dedup."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from lpopt.config import load_config, LpoptConfig, SearchConfig, AcquisitionConfig, ModelConfig
from lpopt.config import (
    FlowConfig, RemoteConfig, MasterConfig, VerifyConfig, DataConfig, CaseConfig,
    FuelConfig, ExtractConfig, ProduceConfig,
)
from lpopt.search.construct import CaseContext, build_pool, candidate_record_id
from lpopt.search.genome import random_genome


def _cfg() -> LpoptConfig:
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(), data=DataConfig(), case=CaseConfig(pair="K1_K2", feed=121),
        fuel=FuelConfig(), extract=ExtractConfig(), produce=ProduceConfig(),
        search=SearchConfig(), acquisition=AcquisitionConfig(), model=ModelConfig(),
    )


def _ctx() -> CaseContext:
    return CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)


def test_pool_feed_pinned_and_unique():
    cfg = _cfg()
    ctx = _ctx()
    rng = random.Random(0)
    elites = [(f"e{i}", random_genome(rng, "K1_K2", 30).to_pattern()) for i in range(6)]
    pool = build_pool(ctx, None, elites, set(), rng, cfg, wave_index=0, size=300)
    assert len(pool) == 300
    # every candidate stays on the fixed campaign feed (121) — no feed drift.
    assert all(c.pattern.feed == 121 for c in pool)
    # record_ids are unique within the pool.
    assert len({c.record_id for c in pool}) == 300
    # record_id is the schema preimage of the pattern.
    for c in pool[:20]:
        assert c.record_id == candidate_record_id(c.pattern, ctx)


def test_pool_proportions_with_parents():
    cfg = _cfg()
    ctx = _ctx()
    rng = random.Random(1)
    elites = [(f"e{i}", random_genome(rng, "K1_K2", 30).to_pattern()) for i in range(8)]
    pool = build_pool(ctx, None, elites, set(), rng, cfg, wave_index=0, size=400)
    counts = Counter(c.origin for c in pool)
    # elite ~60%, guided ~30%, diversity ~10% (top-up may add a few random).
    assert counts["elite"] >= 0.5 * 400
    assert counts["guided"] >= 0.2 * 400
    # elites record their parent lineage.
    assert all(c.parent_record_id is not None for c in pool if c.origin == "elite")


def test_pool_dedups_against_ledger():
    cfg = _cfg()
    ctx = _ctx()
    rng = random.Random(2)
    # pre-seed the ledger with some candidate record_ids; they must never recur.
    seed_rng = random.Random(99)
    banned = set()
    for _ in range(30):
        pat = random_genome(seed_rng, "K1_K2", 30).to_pattern()
        banned.add(candidate_record_id(pat, ctx))
    pool = build_pool(ctx, None, [], banned, rng, cfg, wave_index=0, size=200)
    assert banned.isdisjoint({c.record_id for c in pool})


# --------------------------------------------------------------------------- #
# refinement 2: neighbourhood spread (round-robin parents + near-miss n_moves=1)
# --------------------------------------------------------------------------- #
def test_elite_children_round_robin_across_all_parents():
    """Elite children spread round-robin: every parent is drawn, and no parent
    gets a 3rd child before all parents have had 2 (the pilot's parent-
    concentration failure)."""

    cfg = _cfg()
    ctx = _ctx()
    rng = random.Random(5)
    elites = [(f"e{i}", random_genome(rng, "K1_K2", 30).to_pattern()) for i in range(6)]
    pool = build_pool(ctx, None, elites, set(), rng, cfg, wave_index=0, size=120)
    counts = Counter(c.parent_record_id for c in pool if c.origin == "elite")
    assert set(counts) == {f"e{i}" for i in range(6)}        # every parent used
    # round-robin keeps parents within one child of each other -> no parent
    # reaches a 3rd child while another is still on its 1st.
    assert max(counts.values()) - min(counts.values()) <= 1


def test_near_miss_parents_bias_small_moves():
    """Near-miss parents join the elite arm and mutate with an n_moves=1 (small
    trust-region) bias, so their children sit closer than the default n_moves=2
    elite children."""

    cfg = _cfg()
    ctx = _ctx()
    base = random_genome(random.Random(3), "K1_K2", 30).to_pattern()

    nm_pool = build_pool(
        ctx, None, [], set(), random.Random(7), cfg, wave_index=0,
        near_miss_parents=[("nm", base)], size=60,
    )
    nm_children = [c for c in nm_pool if c.origin == "elite" and c.parent_record_id == "nm"]
    assert nm_children                                        # near-miss became a parent

    el_pool = build_pool(
        ctx, None, [("nm", base)], set(), random.Random(7), cfg, wave_index=0, size=60,
    )
    el_children = [c for c in el_pool if c.origin == "elite" and c.parent_record_id == "nm"]
    nm_ham = np.mean([c.pattern.hamming(base) for c in nm_children])
    el_ham = np.mean([c.pattern.hamming(base) for c in el_children])
    assert nm_ham < el_ham                                    # 1-move stays closer than 2-move


def test_near_miss_parent_deduped_against_elites():
    """A board that is both a near-miss and an elite is a single parent (no
    double-weighting); it keeps the tighter near-miss n_moves=1 bias."""

    cfg = _cfg()
    ctx = _ctx()
    shared = random_genome(random.Random(9), "K1_K2", 30).to_pattern()
    other = random_genome(random.Random(10), "K1_K2", 30).to_pattern()
    pool = build_pool(
        ctx, None, [("shared", shared), ("other", other)], set(), random.Random(2),
        cfg, wave_index=0, near_miss_parents=[("shared", shared)], size=90,
    )
    # both parents produce children; 'shared' is not silently dropped by dedup.
    parents_used = {c.parent_record_id for c in pool if c.origin == "elite"}
    assert {"shared", "other"} <= parents_used
