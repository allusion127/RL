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
    prior = construct.PolicyPrior(mode=mode, head=mode, scorer=stub, version="v1")
    monkeypatch.setattr(construct, "_policy_prior", lambda cfg: prior)
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
    prior = construct._policy_prior(cfg)
    assert prior.scorer is None and prior.shadow is None
    # ...and the readout must be able to tell this apart from policy-on: the
    # mode is still the deck's, but the version is empty and fallback is set.
    assert prior.as_meta() == {"policy_mode": "both", "policy_version": "",
                               "policy_fallback": True}
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

    for mode in ("off", "fr", "flat", "both", "v1", "v2", "shadow_v2"):
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


# --------------------------------------------------------------------------- #
# 6. v2: the schema stamp, the refusal, and train/serve parity
# --------------------------------------------------------------------------- #
MODEL_DIR_V2 = Path("data/models/policy_v2")


def _v2_meta() -> dict:
    """A shipped v2 member's meta, as the loader will see it."""

    import json

    member = MODEL_DIR_V2 / "cnn_seed20260817" / "meta.json"
    if not member.is_file():
        pytest.skip(f"{member} missing")
    return json.loads(member.read_text())


def test_shipped_v2_checkpoints_carry_the_serving_stamp() -> None:
    """schema / version / era / feature list — the four the review asks for."""

    from lpopt.policy.v2 import CURRENT_ERA_LIBRARIES, POLICY_SCHEMA_V2

    meta = _v2_meta()
    assert meta["policy_schema"] == POLICY_SCHEMA_V2
    assert meta["policy_version"] == "v2"
    assert tuple(meta["era_libraries"]) == CURRENT_ERA_LIBRARIES
    assert len(meta["scalar_names"]) == 39      # v1's 36 + v2's 3
    assert meta["delta_channels"]


@pytest.mark.parametrize("mutation", [
    {"policy_schema": "policy_move_v3"},        # a future feature contract
    {"policy_schema": None},                    # an unstamped (pre-A-1) checkpoint
    {"policy_version": "v1"},                   # a v1 checkpoint in a v2 dir
    {"era_libraries": ["ga80"]},                # a different era definition
    {"scalar_names": []},                       # no feature list at all
])
def test_v2_loader_refuses_a_mismatched_stamp(mutation: dict) -> None:
    """A hard error, never a warning.

    The reason it must be hard: ``_policy_pick`` swallows every exception raised
    at score time, so a checkpoint that reaches ``score()`` with the wrong
    feature contract degrades to a uniform random draw — an A/B whose treatment
    arm is its own control.  The refusal has to happen at LOAD.
    """

    from lpopt.policy.scorer import MoveScorerV2

    meta = {**_v2_meta(), **mutation}
    if meta.get("policy_schema") is None:
        meta.pop("policy_schema", None)
    with pytest.raises(ValueError):
        MoveScorerV2._check_meta(meta, Path("cnn_seed20260817"))


def test_v2_loader_accepts_the_shipped_stamp() -> None:
    from lpopt.policy.scorer import MoveScorerV2

    MoveScorerV2._check_meta(_v2_meta(), Path("cnn_seed20260817"))


def test_v2_loader_refuses_the_v1_checkpoint_directory() -> None:
    """The whole-loader form of the refusal: v1 members are not stamped for v2."""

    pytest.importorskip("torch")
    if not (MODEL_DIR / "cnn_seed20260815" / "model.pt").is_file():
        pytest.skip(f"{MODEL_DIR} checkpoints missing")
    from lpopt.policy.scorer import MoveScorerV2

    with pytest.raises(ValueError, match="feature contract"):
        MoveScorerV2.load(MODEL_DIR, fuel_types=FUEL_TYPES, device="cpu")


def test_get_scorer_is_fail_open_by_default_and_fail_closed_when_strict(
        tmp_path: Path) -> None:
    """``strict`` is the production switch: the same bad path, two behaviours."""

    from lpopt.policy.scorer import get_scorer

    nowhere = tmp_path / "nope"
    assert get_scorer(nowhere, version="v2") is None
    with pytest.raises(Exception):
        get_scorer(nowhere, version="v2", strict=True)
    # the strict raise must survive the negative cache the non-strict call left
    with pytest.raises(Exception):
        get_scorer(nowhere, version="v2", strict=True)


# --------------------------------------------------------------------------- #
# 6a. the real v2 ensemble (skipped when the checkpoints are not present)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_scorer_v2():  # noqa: ANN201
    pytest.importorskip("torch")
    if not (MODEL_DIR_V2 / "cnn_seed20260817" / "model.pt").is_file():
        pytest.skip(f"{MODEL_DIR_V2} checkpoints missing")
    if not FUEL_TYPES.is_file():
        pytest.skip(f"{FUEL_TYPES} missing")
    from lpopt.policy.scorer import MoveScorerV2

    return MoveScorerV2.load(fuel_types=FUEL_TYPES, device="cpu", n_threads=4)


def test_v2_scorer_loads_five_members_with_v2s_feature_layout(real_scorer_v2) -> None:  # noqa: ANN001
    from lpopt.policy.data import HEADS

    assert real_scorer_v2.version == "v2"
    assert len(real_scorer_v2.members) == 5
    assert len(real_scorer_v2.scalar_names) == 39       # v1 serves 36
    cfg = real_scorer_v2.members[0].config
    assert real_scorer_v2._grid_shape[0] == cfg.in_channels
    assert cfg.n_cond == len(real_scorer_v2.scalar_names) + 13
    assert cfg.n_heads == len(HEADS)


def test_v2_scorer_forward_on_freshly_mutated_children(real_scorer_v2) -> None:  # noqa: ANN001
    from lpopt.search.genome import mutate

    ctx = _ctx()
    rng = random.Random(0)
    parent = random_genome(rng, "K1_K2", 30)
    kids = []
    while len(kids) < 4:
        child = mutate(parent, rng, 2, feed_move_prob=0.0, batches=ctx.batches)
        kids.append((child, child.to_pattern()))

    probs = real_scorer_v2.score((parent, parent.to_pattern()), kids, ctx)
    assert probs.shape == (4, 2)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()


def _training_corpus_snapshot() -> "Path | None":
    """The steps parquet whose sha256 the v2 members were trained on.

    ``data/policy/steps.parquet`` GROWS between rounds and ``build_splits_v2``
    is a function of the whole frame — so re-scoring "row i of gate_cur" against
    a probs artifact written from a different corpus compares two different
    moves, and would pass or fail for no reason.  The snapshot is selected by
    fingerprint, and the test skips rather than guesses when none matches.
    """

    from lpopt.policy.data import corpus_fingerprint

    want = _v2_meta().get("corpus_sha256", "")
    if not want:
        return None
    for path in sorted(Path("data/policy").glob("steps.parquet*")):
        if corpus_fingerprint(path) == want:
            return path
    return None


def test_v2_serving_reproduces_the_training_run_scores(real_scorer_v2) -> None:  # noqa: ANN001
    """Train/serve parity for v2 — the claim the whole A/B rests on.

    v1's version of this test is ``test_serving_reproduces_the_training_run_
    probabilities``.  This one re-scores four ``gate_cur`` rows through the
    serving path and compares them to ``probs.npz``, which the training run
    wrote from the cached corpus features.  Any drift in the three v2 additions
    (``d_fresh_enr_mass``, ``parent_fresh_enr_mass``, ``era_current``), in the
    sorted name order ``scalar_features_v2`` imposes, or in a board channel
    would move these numbers far beyond float noise.
    """

    if not (MODEL_DIR_V2 / "probs.npz").is_file():
        pytest.skip("policy v2 probs artifact missing")
    snapshot = _training_corpus_snapshot()
    if snapshot is None:
        pytest.skip("no steps parquet matches the checkpoints' corpus_sha256")

    from lpopt.data.schema import unpack_pattern
    from lpopt.policy.scorer import _corpus
    from lpopt.policy.v2 import build_splits_v2, load_universe_v2

    m = _corpus()
    steps = load_universe_v2(str(snapshot))
    fold = build_splits_v2(steps, seed=20260817)
    sub = steps[fold == "gate_cur"].reset_index(drop=True)
    reference = np.load(MODEL_DIR_V2 / "probs.npz")["gate_cur"].mean(axis=0)
    assert len(sub) == len(reference)          # the split reproduced row-for-row

    for i in range(4):
        row = sub.iloc[i]
        ctx = CaseContext(pair=str(row["case_pair"]), feed=int(row["feed"]),
                          library_id=str(row["library_id"]))
        parent = (m.genome_of(row["parent_pattern"]),
                  unpack_pattern(row["parent_pattern"]))
        child = [(m.genome_of(row["child_pattern"]),
                  unpack_pattern(row["child_pattern"]))]
        np.testing.assert_allclose(
            real_scorer_v2.score(parent, child, ctx)[0], reference[i], atol=5e-3)


# --------------------------------------------------------------------------- #
# 6b. train/serve provenance parity (2026-08-29 forensic)
# --------------------------------------------------------------------------- #
def _corpus_snapshot_or_skip() -> Path:
    snap = _training_corpus_snapshot()
    if snap is None:
        pytest.skip("no steps parquet matches the checkpoints' corpus_sha256")
    return snap


def _era_libraries() -> tuple[str, ...]:
    from lpopt.policy.v2 import CURRENT_ERA_LIBRARIES

    return CURRENT_ERA_LIBRARIES


def test_corpus_provenance_reproduces_every_corpus_rows_dataset() -> None:
    """``corpus_provenance`` is the serve-side reconstruction of a column the
    corpus took from the store.  It must reproduce it for EVERY row.

    Only A/not-A reaches the encoder (``g_dataset_flag``), so that is the
    equivalence.  Before the 2026-08-29 fix the serve path used
    ``featurize.library_provenance``, which predates ``dataset="P"`` and calls
    paramA a Dataset-A library — inverting the flag on every paramA proposal.
    """

    import pandas as pd

    from lpopt.model.featurize import library_provenance
    from lpopt.policy.data import corpus_provenance

    if not STEPS.is_file():
        pytest.skip(f"{STEPS} missing")
    steps = pd.read_parquet(STEPS, columns=["library_id", "dataset"])
    census = steps.groupby(["library_id", "dataset"]).size()
    assert len(census), "empty corpus"
    for lib, ds in census.index:
        assert (str(ds) == "A") == (corpus_provenance(str(lib))[0] == "A"), \
            f"{lib}: corpus rows carry dataset={ds!r}, serving derives " \
            f"{corpus_provenance(str(lib))[0]!r}"

    # ...and this is precisely what the OLD serve map got wrong, on a live library.
    assert library_provenance("paramA")[0] == "A"          # the historical answer
    assert corpus_provenance("paramA")[0] == "P"           # what the corpus rows say
    # the sym_class half is knowingly still the corpus's (wrong) map: the shipped
    # v2 checkpoint trained on it, so serving must keep feeding it.  Re-mine first.
    assert corpus_provenance("ga80")[1] == library_provenance("ga80")[1] == "free69"


def test_policy_serve_row_featurization_parity(real_scorer_v2) -> None:  # noqa: ANN001
    """HARD GATE, the policy analogue of
    ``tests/test_model_api.py::test_serve_row_featurization_parity``.

    On real corpus rows spanning BOTH live libraries, the board the serve path
    builds (``MoveScorer._board``, from a ``CaseContext``) must be the board the
    training pattern cache built (``build_pattern_cache``, from the step row's own
    store ``dataset``) — every slot value and every conditioning global.

    Slots and every non-``g_fresh_mean_*`` global — the two provenance ones
    (``g_dataset_flag`` / ``g_sym_class``) included — are asserted BYTE-EQUAL:
    they are categorical or exactly reproducible, and a mismatch is the defect
    this gate exists for (paramA's ``g_dataset_flag`` was 0.0 at serve against
    1.0 in training).

    The five ``g_fresh_mean_*`` globals are compared at 1e-3 instead.  They are
    reductions OVER the slot matrix, and ``build_pattern_cache`` reduces the
    full-precision slots while ``_board`` reduces the float16-rounded ones the
    network actually trained on (the cache stores float16 and widens at collate).
    That is a known ~3e-4 asymmetry in the *reduction*, present since v1 and
    unrelated to provenance; it is what leaves the score-level residual at ~2e-4
    rather than 0.
    """

    from lpopt.data.schema import unpack_pattern
    from lpopt.policy.data import build_pattern_cache

    snapshot = _corpus_snapshot_or_skip()
    from lpopt.policy.v2 import load_universe_v2

    steps = load_universe_v2(str(snapshot))
    frames = []
    for lib in _era_libraries():
        k = steps[steps["library_id"] == lib]
        if len(k) < 6:
            pytest.skip(f"fewer than 6 usable {lib} corpus rows available")
        frames.append(k.head(6))
    import pandas as pd

    rows = pd.concat(frames).reset_index(drop=True)
    assert set(rows["library_id"]) == set(_era_libraries())
    assert set(rows["dataset"]) == {"P"}          # the campaign-era rows served

    cache = build_pattern_cache(rows, fuel_types=FUEL_TYPES, progress=False)
    names = list(real_scorer_v2.encoder.globals_names)
    assert {"g_dataset_flag", "g_sym_class"} <= set(names)
    exact = [i for i, n in enumerate(names) if not n.startswith("g_fresh_mean_")]
    fuzzy = [i for i, n in enumerate(names) if n.startswith("g_fresh_mean_")]
    assert exact and fuzzy

    for _, row in rows.iterrows():
        ctx = CaseContext(pair=str(row["case_pair"]), feed=int(row["feed"]),
                          library_id=str(row["library_id"]))
        for side in ("parent", "child"):
            pat = unpack_pattern(str(row[f"{side}_pattern"]))
            real_scorer_v2._slots.clear()
            real_scorer_v2._globals.clear()
            s_serve, g_serve = real_scorer_v2._board(pat, ctx)
            i = cache.index[pat.canonical()]
            s_train = cache.slots[i].astype(np.float32)
            g_train = cache.globals_[i]
            assert np.array_equal(s_serve, s_train), \
                f"{row['library_id']} {side} slots differ"
            msg = (f"{row['library_id']} {side} conditioning globals differ: "
                   f"serve {g_serve[exact]} vs train {g_train[exact]} "
                   f"(names {[names[j] for j in exact]})")
            assert np.array_equal(g_serve[exact], g_train[exact]), msg
            np.testing.assert_allclose(g_serve[fuzzy], g_train[fuzzy], atol=1e-3)


def test_v2_serving_reproduces_the_training_probabilities_on_both_libraries(
        real_scorer_v2) -> None:  # noqa: ANN001
    """The score-level form of the gate, on BOTH libraries.

    ``test_v2_serving_reproduces_the_training_run_scores`` only samples rows 0-3
    of ``gate_cur``, which are all ga80 — the library the old provenance map
    happened to get right (``("B", "free69")`` and ``("P", "free69")`` are the
    same two encoder inputs).  It therefore passed throughout the defect.  This
    one takes the first 8 rows of EACH live library; before the fix the paramA
    rows were off by up to 0.087 absolute, seventeen times the 5e-3 tolerance.
    """

    if not (MODEL_DIR_V2 / "probs.npz").is_file():
        pytest.skip("policy v2 probs artifact missing")
    snapshot = _corpus_snapshot_or_skip()

    from lpopt.data.schema import unpack_pattern
    from lpopt.policy.scorer import _corpus
    from lpopt.policy.v2 import build_splits_v2, load_universe_v2

    m = _corpus()
    steps = load_universe_v2(str(snapshot))
    fold = build_splits_v2(steps, seed=20260817)
    sub = steps[fold == "gate_cur"].reset_index(drop=True)
    reference = np.load(MODEL_DIR_V2 / "probs.npz")["gate_cur"].mean(axis=0)
    assert len(sub) == len(reference)

    seen = set()
    for lib in _era_libraries():
        idx = [i for i in range(len(sub))
               if str(sub.iloc[i]["library_id"]) == lib][:8]
        if len(idx) < 8:
            pytest.skip(f"fewer than 8 gate_cur {lib} rows")
        seen.add(lib)
        for i in idx:
            row = sub.iloc[i]
            ctx = CaseContext(pair=str(row["case_pair"]), feed=int(row["feed"]),
                              library_id=lib)
            parent = (m.genome_of(row["parent_pattern"]),
                      unpack_pattern(row["parent_pattern"]))
            child = [(m.genome_of(row["child_pattern"]),
                      unpack_pattern(row["child_pattern"]))]
            np.testing.assert_allclose(
                real_scorer_v2.score(parent, child, ctx)[0], reference[i],
                atol=5e-3, err_msg=f"{lib} gate_cur row {i}")
    assert seen == set(_era_libraries())
