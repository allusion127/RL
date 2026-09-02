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
import pytest

from lpopt.search import construct
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


# --------------------------------------------------------------------------- #
# policy_prior serving modes at the elite-mutation step (review section 6.2)
#
# These use a STUB ensemble injected at ``lpopt.policy.scorer.get_scorer``, so
# they exercise the real dispatch, the real shadow bookkeeping and the real
# strict/fallback branches without torch or a checkpoint.
# --------------------------------------------------------------------------- #
class _StubEnsemble:
    def __init__(self, version: str) -> None:
        self.version = version
        self.calls = 0

    def score(self, parent, children, ctx):  # noqa: ANN001, ANN201
        self.calls += 1
        n = len(children)
        col = np.linspace(0.9, 0.1, n) if n > 1 else np.array([0.5])
        return np.stack([col, col[::-1]], axis=1)


def _acq_cfg(**acq) -> LpoptConfig:
    cfg = _cfg()
    cfg.acquisition = AcquisitionConfig(**acq)
    return cfg


def _elites(n: int = 6):
    rng = random.Random(0)
    return [(f"e{i}", random_genome(rng, "K1_K2", 30).to_pattern()) for i in range(n)]


def _stub_loader(monkeypatch, scorer=None, seen=None):
    """Intercept ``get_scorer`` and record how the dispatcher called it."""

    def _fake(model_dir=None, *, version="v1", **kw):
        if seen is not None:
            seen.append({"model_dir": str(model_dir), "version": version, **kw})
        return _StubEnsemble(version) if scorer is None else scorer

    monkeypatch.setattr("lpopt.policy.scorer.get_scorer", _fake)


@pytest.mark.parametrize("mode,version,dir_key", [
    ("fr", "v1", "policy_prior_model_dir"),
    ("both", "v1", "policy_prior_model_dir"),
    ("v1", "v1", "policy_prior_model_dir"),
    ("v2", "v2", "policy_prior_model_dir_v2"),
    ("shadow_v2", "v2", "policy_prior_model_dir_v2"),
])
def test_mode_dispatch_picks_the_version_and_the_directory(
        monkeypatch, mode: str, version: str, dir_key: str) -> None:
    seen: list[dict] = []
    _stub_loader(monkeypatch, seen=seen)
    cfg = _acq_cfg(policy_prior=mode)
    prior = construct._policy_prior(cfg)

    assert seen and seen[0]["version"] == version
    assert seen[0]["model_dir"] == getattr(cfg.acquisition, dir_key)
    assert prior.mode == mode and prior.version == version
    # shadow watches; every other on-mode selects
    assert (prior.shadow is not None) is (mode == "shadow_v2")
    assert (prior.scorer is not None) is (mode != "shadow_v2")
    assert prior.as_meta() == {"policy_mode": mode, "policy_version": version,
                               "policy_fallback": False}


def test_off_mode_loads_nothing_and_reports_nothing(monkeypatch) -> None:
    seen: list[dict] = []
    _stub_loader(monkeypatch, seen=seen)
    prior = construct._policy_prior(_acq_cfg(policy_prior="off"))
    assert not seen                       # no checkpoint touched
    assert prior.as_meta() == {"policy_mode": "off", "policy_version": "",
                               "policy_fallback": False}


def test_v2_mode_uses_the_v2_temperature(monkeypatch) -> None:
    """v1's tau on v2's score scale is a near-uniform softmax — a null arm."""

    taus: list[float] = []
    stub = _StubEnsemble("v2")
    _stub_loader(monkeypatch, scorer=stub)
    real_pick = construct._policy_pick
    monkeypatch.setattr(
        construct, "_policy_pick",
        lambda s, h, c, p, cands, rng, tau: (taus.append(tau)
                                             or real_pick(s, h, c, p, cands, rng, tau)))
    build_pool(_ctx(), None, _elites(), set(), random.Random(4),
               _acq_cfg(policy_prior="v2", policy_prior_random_floor=0.0,
                        policy_prior_candidates=4),
               wave_index=0, size=80)
    assert taus and set(taus) == {0.08}


def test_shadow_mode_builds_the_off_pool_and_only_records(monkeypatch) -> None:
    """The prospective A/B arm: v2 watches the wave, it does not steer it.

    Same seed, same parents: the candidate list must be identical to ``off``
    down to the record_id order, because a shadow arm that perturbs the pool is
    not a control for anything.
    """

    stub = _StubEnsemble("v2")
    _stub_loader(monkeypatch, scorer=stub)

    off_meta: dict = {}
    off = build_pool(_ctx(), None, _elites(), set(), random.Random(5),
                     _acq_cfg(policy_prior="off"), wave_index=0, size=150,
                     meta=off_meta)
    shadow_meta: dict = {}
    shadow = build_pool(_ctx(), None, _elites(), set(), random.Random(5),
                        _acq_cfg(policy_prior="shadow_v2"), wave_index=0,
                        size=150, meta=shadow_meta)

    assert [c.record_id for c in shadow] == [c.record_id for c in off]
    assert stub.calls > 0                              # v2 really scored
    scores = shadow_meta["policy_shadow_scores"]
    elite = [c.record_id for c in shadow if c.origin == "elite"]
    assert elite and set(scores) == set(elite)         # every elite child, and only those
    assert all(len(v) == 2 and all(np.isfinite(v)) for v in scores.values())
    assert shadow_meta["policy_mode"] == "shadow_v2"
    assert shadow_meta["policy_version"] == "v2"
    assert shadow_meta["policy_fallback"] is False
    # 'off' says so plainly rather than by omission
    assert off_meta == {"policy_mode": "off", "policy_version": "",
                        "policy_fallback": False}


def test_strict_mode_raises_when_the_policy_will_not_load(tmp_path) -> None:
    """Fail-closed (review section 6.12): production stops, it does not degrade."""

    cfg = _acq_cfg(policy_prior="v2", policy_prior_strict=True,
                   policy_prior_model_dir_v2=str(tmp_path / "absent"))
    with pytest.raises(Exception) as exc:
        build_pool(_ctx(), None, _elites(), set(), random.Random(4), cfg,
                   wave_index=0, size=60)
    assert not isinstance(exc.value, AssertionError)


def test_non_strict_fallback_is_recorded_in_the_wave_metadata(
        tmp_path, monkeypatch, capsys) -> None:
    """The fail-OPEN path must still be unmistakable in the readout.

    A random-mutation wave labelled ``policy_prior = "v2"`` in the deck is the
    exact fail-open confusion P1-05 names; ``policy_fallback`` is what a readout
    checks so it can never count this wave as a policy arm.
    """

    monkeypatch.setattr(construct, "_POLICY_WARNED", False)
    cfg = _acq_cfg(policy_prior="v2",
                   policy_prior_model_dir_v2=str(tmp_path / "absent"))
    meta: dict = {}
    pool = build_pool(_ctx(), None, _elites(), set(), random.Random(4), cfg,
                      wave_index=0, size=60, meta=meta)

    assert len(pool) == 60                             # the wave still runs
    assert meta == {"policy_mode": "v2", "policy_version": "",
                    "policy_fallback": True}
    assert "policy_shadow_scores" not in meta
    assert "WARNING" in capsys.readouterr().out


def test_meta_is_optional_and_the_pool_is_unchanged_by_it(monkeypatch) -> None:
    """Every existing caller passes no ``meta``; that must stay a no-op."""

    stub = _StubEnsemble("v2")
    _stub_loader(monkeypatch, scorer=stub)
    cfg = _acq_cfg(policy_prior="shadow_v2")
    a = build_pool(_ctx(), None, _elites(), set(), random.Random(6), cfg,
                   wave_index=0, size=100)
    b = build_pool(_ctx(), None, _elites(), set(), random.Random(6), cfg,
                   wave_index=0, size=100, meta={})
    assert [c.record_id for c in a] == [c.record_id for c in b]


def test_an_unknown_mode_is_not_silently_treated_as_off() -> None:
    """A typo in a hand-built config must not resolve to a control arm."""

    with pytest.raises(ValueError, match="policy_prior"):
        construct._policy_prior(_acq_cfg(policy_prior="v2_fr"))
