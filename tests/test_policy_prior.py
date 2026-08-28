"""The learned move prior at the ``build_pool`` mutation step.

The first test is the one that matters most: with ``[acquisition] policy_prior``
at its default ``"off"``, the pool must be byte-identical to the pool the
pre-change code built — same candidates, same order, same rng consumption.  The
rest exercise the flag-on machinery with a STUB scorer (so they need neither
torch nor the checkpoints), plus two guards on the real ensemble that skip when
``data/models/policy_v1`` is absent.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lpopt.config import (
    AcquisitionConfig, CaseConfig, ConfigError, DataConfig, ExtractConfig,
    FlowConfig, FuelConfig, LpoptConfig, MasterConfig, ModelConfig,
    ProduceConfig, RemoteConfig, SearchConfig, VerifyConfig,
)
from lpopt.search import construct
from lpopt.search.construct import CaseContext, build_pool
from lpopt.search.genome import random_genome

MODEL_DIR = Path("data/models/policy_v1")
STEPS = Path("data/policy/steps.parquet")
FUEL_TYPES = Path("data/store/fuel_types.parquet")


def _cfg(**acq: object) -> LpoptConfig:
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(), data=DataConfig(),
        case=CaseConfig(pair="K1_K2", feed=121), fuel=FuelConfig(),
        extract=ExtractConfig(), produce=ProduceConfig(), search=SearchConfig(),
        acquisition=AcquisitionConfig(**acq), model=ModelConfig(),
    )


def _ctx() -> CaseContext:
    return CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)


def _elites(n: int = 6) -> list[tuple[str, object]]:
    rng = random.Random(0)
    return [(f"e{i}", random_genome(rng, "K1_K2", 30).to_pattern()) for i in range(n)]


def _pool_ids(cfg: object, seed: int = 4, size: int = 120) -> list[str]:
    pool = build_pool(_ctx(), None, _elites(), set(), random.Random(seed), cfg,
                      wave_index=0, size=size)
    return [c.record_id for c in pool]


class _StubScorer:
    """Scores candidates by a fixed descending ramp; counts its calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def score(self, parent, children, ctx) -> np.ndarray:  # noqa: ANN001
        self.calls += 1
        self.batch_sizes.append(len(children))
        n = len(children)
        col = np.linspace(0.95, 0.05, n) if n > 1 else np.array([0.5])
        return np.stack([col, col[::-1]], axis=1)


# --------------------------------------------------------------------------- #
# 1. THE REGRESSION GUARD: flag off == the pool the previous code built
# --------------------------------------------------------------------------- #
def test_flag_off_pool_is_identical_to_a_config_without_the_knob() -> None:
    """Default ``policy_prior="off"`` must not perturb the pool by one draw.

    The control is a config whose ``acquisition`` has NONE of the policy fields
    — the exact attribute surface ``build_pool`` saw before this feature existed.
    If the new code consumed a single extra rng draw, or loaded a model, or
    reordered an admission, these two lists would diverge.
    """

    legacy = _cfg()
    legacy.acquisition = SimpleNamespace()          # pre-change attribute surface
    assert _pool_ids(_cfg()) == _pool_ids(legacy)


def test_flag_off_is_deterministic_for_a_fixed_seed() -> None:
    assert _pool_ids(_cfg(), seed=11) == _pool_ids(_cfg(), seed=11)


def test_flag_off_never_touches_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """No checkpoint is loaded and no score is taken when the flag is off."""

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("policy_prior=off must not load a scorer")

    monkeypatch.setattr("lpopt.policy.scorer.get_scorer", _boom)
    assert len(_pool_ids(_cfg())) == 120


# --------------------------------------------------------------------------- #
# 2. flag-on smoke, with a stub scorer (no torch, no checkpoints)
# --------------------------------------------------------------------------- #
def _with_stub(monkeypatch: pytest.MonkeyPatch, mode: str = "both") -> _StubScorer:
    stub = _StubScorer()
    monkeypatch.setattr(construct, "_policy_prior", lambda cfg: (stub, mode))
    return stub


def test_policy_on_keeps_the_pool_valid_and_deduplicated(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _with_stub(monkeypatch)
    cfg = _cfg(policy_prior="both", policy_prior_candidates=6,
               policy_prior_random_floor=0.2)
    pool = build_pool(_ctx(), None, _elites(), set(), random.Random(4), cfg,
                      wave_index=0, size=120)

    assert len(pool) == 120
    assert len({c.record_id for c in pool}) == 120       # dedup holds
    assert all(c.pattern.feed == 121 for c in pool)      # feed still pinned
    assert all(c.record_id == construct.candidate_record_id(c.pattern, _ctx())
               for c in pool[:20])
    elite = [c for c in pool if c.origin == "elite"]
    assert elite and all(c.parent_record_id is not None for c in elite)
    assert stub.calls > 0                                # the prior actually fired
    assert max(stub.batch_sizes) <= 6                    # honours the candidate cap


def test_policy_on_respects_the_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scored candidates are filtered against the ledger BEFORE they are ranked,
    so a banned board can never be the softmax pick."""

    _with_stub(monkeypatch)
    ctx = _ctx()
    seed_rng = random.Random(99)
    banned = {construct.candidate_record_id(
        random_genome(seed_rng, "K1_K2", 30).to_pattern(), ctx) for _ in range(30)}
    cfg = _cfg(policy_prior="fr", policy_prior_candidates=6)
    pool = build_pool(ctx, None, _elites(), banned, random.Random(2), cfg,
                      wave_index=0, size=100)
    assert banned.isdisjoint({c.record_id for c in pool})


def test_random_mutation_floor_is_honoured(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The floor is the rail that stops the pool going policy-degenerate."""

    stub = _with_stub(monkeypatch)
    all_floor = _cfg(policy_prior="both", policy_prior_random_floor=1.0)
    build_pool(_ctx(), None, _elites(), set(), random.Random(4), all_floor,
               wave_index=0, size=120)
    assert stub.calls == 0                     # every slot unscored

    stub = _with_stub(monkeypatch)
    no_floor = _cfg(policy_prior="both", policy_prior_random_floor=0.0,
                    policy_prior_candidates=4)
    build_pool(_ctx(), None, _elites(), set(), random.Random(4), no_floor,
               wave_index=0, size=120)
    scored_all = stub.calls

    stub = _with_stub(monkeypatch)
    half = _cfg(policy_prior="both", policy_prior_random_floor=0.5,
                policy_prior_candidates=4)
    build_pool(_ctx(), None, _elites(), set(), random.Random(4), half,
               wave_index=0, size=120)
    # a 50% floor scores materially fewer slots than a 0% floor
    assert 0 < stub.calls < scored_all


# --------------------------------------------------------------------------- #
# 3. the softmax pick itself
# --------------------------------------------------------------------------- #
def _picks(temperature: float, draws: int = 400) -> list[int]:
    stub = _StubScorer()
    parent = random_genome(random.Random(1), "K1_K2", 30)
    cands = [(parent, parent.to_pattern(), f"r{i}") for i in range(8)]
    rng = random.Random(7)
    return [construct._policy_pick(stub, "fr", _ctx(), parent, cands, rng,
                                   temperature) for _ in range(draws)]


def test_pick_samples_rather_than_argmaxes() -> None:
    """The report's safety rail: a good candidate is preferred, not guaranteed."""

    counts = np.bincount(_picks(0.25), minlength=8)
    assert counts[0] > counts[-1]              # the 0.95 candidate wins more often
    assert counts[-1] > 0                      # ...but the 0.05 one still appears
    assert (counts > 0).sum() >= 6             # the neighbourhood is not collapsed


def test_low_temperature_collapses_toward_argmax() -> None:
    counts = np.bincount(_picks(0.01), minlength=8)
    assert counts[0] > 0.9 * counts.sum()


def test_pick_falls_back_to_a_uniform_draw_when_scoring_fails() -> None:
    """A scorer failure degrades to random mutation; it never aborts the pool."""

    class _Broken:
        def score(self, *_a: object, **_k: object) -> np.ndarray:
            raise RuntimeError("checkpoint went away")

    parent = random_genome(random.Random(1), "K1_K2", 30)
    cands = [(parent, parent.to_pattern(), f"r{i}") for i in range(5)]
    rng = random.Random(3)
    picked = [construct._policy_pick(_Broken(), "fr", _ctx(), parent, cands, rng, 0.25)
              for _ in range(50)]
    assert set(picked) <= set(range(5)) and len(set(picked)) > 1


def test_unloadable_checkpoints_degrade_to_off(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    """A deck asking for a prior that will not load must SAY so, not run mute."""

    monkeypatch.setattr(construct, "_POLICY_WARNED", False)
    cfg = _cfg(policy_prior="both", policy_prior_model_dir=str(tmp_path / "nope"))
    scorer, mode = construct._policy_prior(cfg)
    assert scorer is None and mode == "off"
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 4. deck validation
# --------------------------------------------------------------------------- #
def test_deck_rejects_an_unknown_policy_prior(tmp_path: Path) -> None:
    from lpopt.config import load_config

    deck = tmp_path / "d.inp"
    deck.write_text('[case]\npair = "K1_K2"\n\n[acquisition]\n'
                    'policy_prior = "argmax"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="policy_prior"):
        load_config(deck)


def test_deck_accepts_every_documented_mode(tmp_path: Path) -> None:
    from lpopt.config import load_config

    for mode in ("off", "fr", "flat", "both"):
        deck = tmp_path / f"{mode}.inp"
        deck.write_text(f'[case]\npair = "K1_K2"\n\n[acquisition]\n'
                        f'policy_prior = "{mode}"\n', encoding="utf-8")
        assert load_config(deck).acquisition.policy_prior == mode


# --------------------------------------------------------------------------- #
# 5. the real ensemble (skipped when the checkpoints are not present)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_scorer():  # noqa: ANN201
    pytest.importorskip("torch")
    if not (MODEL_DIR / "cnn_seed20260815" / "model.pt").is_file():
        pytest.skip(f"{MODEL_DIR} checkpoints missing")
    if not FUEL_TYPES.is_file():
        pytest.skip(f"{FUEL_TYPES} missing")
    from lpopt.policy.scorer import MoveScorer

    return MoveScorer.load(MODEL_DIR, fuel_types=FUEL_TYPES, device="cpu",
                           n_threads=4)


def test_scorer_loads_five_members_with_one_feature_layout(real_scorer) -> None:  # noqa: ANN001
    from lpopt.policy.data import HEADS

    assert len(real_scorer.members) == 5
    assert len(real_scorer.scalar_names) == 36
    # the board tensor the scorer builds must be the one the net was trained on
    cfg = real_scorer.members[0].config
    assert real_scorer._grid_shape[0] == cfg.in_channels
    assert cfg.n_cond == len(real_scorer.scalar_names) + 13
    assert cfg.n_heads == len(HEADS)


def test_scorer_forward_on_freshly_mutated_children(real_scorer) -> None:  # noqa: ANN001
    from lpopt.search.genome import mutate

    ctx = _ctx()
    rng = random.Random(0)
    parent = random_genome(rng, "K1_K2", 30)
    kids = []
    while len(kids) < 4:
        child = mutate(parent, rng, 2, feed_move_prob=0.0, batches=ctx.batches)
        kids.append((child, child.to_pattern()))

    probs = real_scorer.score((parent, parent.to_pattern()), kids, ctx)
    assert probs.shape == (4, 2)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    assert real_scorer.score((parent, parent.to_pattern()), [], ctx).shape == (0, 2)


def test_serving_reproduces_the_training_run_probabilities(real_scorer) -> None:  # noqa: ANN001
    """The claim the whole integration rests on: these are the SAME features.

    Four rows of the ``heldout_cell`` fold are re-scored through the serving path
    and compared to ``probs_cnn.npz``, which the training run wrote from the
    cached corpus features.  Any drift in a descriptor, a channel order or the
    globals would move these numbers well beyond float noise.
    """

    if not STEPS.is_file() or not (MODEL_DIR / "probs_cnn.npz").is_file():
        pytest.skip("policy corpus / probs artifact missing")
    from lpopt.data.schema import unpack_pattern
    from lpopt.policy.data import build_splits, load_universe
    from lpopt.policy.scorer import _corpus

    m = _corpus()
    steps = load_universe(STEPS)
    fold = build_splits(steps, seed=20260815)
    sub = steps[fold == "heldout_cell"].reset_index(drop=True)
    reference = np.load(MODEL_DIR / "probs_cnn.npz")["heldout_cell"].mean(axis=0)

    for i in range(4):
        row = sub.iloc[i]
        ctx = CaseContext(pair=str(row["case_pair"]), feed=int(row["feed"]),
                          library_id=str(row["library_id"]))
        parent = (m.genome_of(row["parent_pattern"]),
                  unpack_pattern(row["parent_pattern"]))
        child = [(m.genome_of(row["child_pattern"]),
                  unpack_pattern(row["child_pattern"]))]
        np.testing.assert_allclose(
            real_scorer.score(parent, child, ctx)[0], reference[i], atol=5e-3)
