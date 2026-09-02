"""WaveVerifier + StubEvaluator: two-wave run with an injected failure exercises
the full converged / nonconverged / error taxonomy, and outcome->record
conversion carries ``dataset="P"`` + ``restart_provenance`` (plan 4.6 / M2.5).
"""

from __future__ import annotations

import contextlib
import random
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from lpopt.search import verify as verify_mod
from lpopt.search.assets import ResolvedAssets
from lpopt.search.genome import random_genome
from lpopt.search.stub import StubEvaluator
from lpopt.search.verify import (
    PHYSICS_KILL_FAILURES,
    WatchdogMasterRunner,
    WaveEntry,
    WaveOutcome,
    WaveVerifier,
    classify_outcome,
    outcome_to_record,
)
from lpopt.vendor.masterrl.domain import CaseKey
from lpopt.vendor.masterrl.master import MasterRunError
from lpopt.vendor.masterrl.parallel import CoreLayout

VALID_RELOAD_DECK = (
    "%JOB_TYP\n"
    "        1       stead                               # irrst=1 (restart)\n"
    "        MAS_RST.SEED.01\n"
    "        xsl     MAS_XSL\n"
    "        hff     MAS_HFF\n"
    "        out     MAS_OUT\n"
    "        sum     MAS_SUM\n"
    "%JOB_IDE\n"
    "        APRQ    12\n"
    "%GEN_DIM\n"
    "        10      10      27      83      85          # nx, ny, nz, nbatch, ncomp\n"
    "%LPD_SHF\n"
    "        F K1  0,\n"
    "%EXE_DEP                                            # BOC\n"
    "        0.0\n"
    "%EDT_OPT                                            # EOC restart write\n"
    "        1\n"
    "%END\n"
)


def _unique_patterns(rng: random.Random, pair: str, n_fresh: int, count: int) -> list:
    seen: dict[str, object] = {}
    while len(seen) < count:
        pat = random_genome(rng, pair, n_fresh).to_pattern()
        seen[pat.digest] = pat
    return list(seen.values())


def _resolved(case_key: CaseKey, provenance: str) -> ResolvedAssets:
    return ResolvedAssets(
        case_key=case_key,
        restart_path=None,
        template_deck_path=None,
        fallback_level=0,
        restart_provenance=provenance,
    )


def test_two_wave_taxonomy_and_records(tmp_path: Path) -> None:
    rng = random.Random(11)
    patterns = _unique_patterns(rng, "K1_K2", 30, 8)

    fail_digest = patterns[0].digest       # -> error (evaluator raises)
    nonconv_digest = patterns[3].digest    # -> nonconverged FOM
    stub = StubEvaluator(
        fail_prefixes=[fail_digest],
        nonconverge_prefixes=[nonconv_digest],
    )
    verifier = WaveVerifier(
        run_dir=tmp_path,
        evaluator_factory=lambda worker_id, cpu_core: stub,
        workers=4,
        max_cycles=14,
    )

    case_key = CaseKey("K1_K2", 121)
    provenance = "native:MAS_RST.APRQ_11_0652.86"
    entries = [
        WaveEntry(
            pattern=pat,
            case_key=case_key,
            resolved_assets=_resolved(case_key, provenance),
            meta={"stratum": "s0", "generator": "random"},
        )
        for pat in patterns
    ]

    # Two waves of four.
    outcomes = verifier.evaluate_wave(entries[:4]) + verifier.evaluate_wave(entries[4:])
    assert len(outcomes) == 8

    statuses = {o.status for o in outcomes}
    assert statuses == {"converged", "nonconverged", "error"}

    by_digest = {o.pattern.digest: o for o in outcomes}
    assert by_digest[fail_digest].status == "error"
    assert by_digest[fail_digest].fom is None
    assert by_digest[fail_digest].failure  # non-empty message
    assert by_digest[nonconv_digest].status == "nonconverged"

    # Determinism: re-running yields identical FOMs for the converged rows.
    again = verifier.evaluate_wave(entries[:4])
    for a in again:
        if a.status == "converged":
            assert a.fom.f_r == by_digest[a.pattern.digest].fom.f_r

    # Every outcome -> a P record carrying provenance; error rows are invalid.
    for outcome in outcomes:
        record = outcome_to_record(
            outcome,
            library_id="ga80",
            stratum="s0",
            generator="random",
            campaign="test",
        )
        assert record.dataset == "P"
        assert record.restart_provenance == provenance
        assert record.feed == 121
        assert record.n_batches == 2 and record.depth2_edges == 0
        if outcome.status == "error":
            assert record.valid is False
            assert record.converged is False
            assert record.f_r is None and record.cyclen is None
        else:
            assert record.valid is True
            assert record.f_r is not None
            assert record.converged == (outcome.status == "converged")


# --------------------------------------------------------------------------- #
# four-way QC classification: a non_finite_flux kill is an HONEST NEGATIVE, not a
# harness fault (plan 5.4 / 4.6 정직 회계).  classify_outcome refines the three-way
# ``status`` so the produce driver never halts a legitimately NaN-heavy stratum.
# --------------------------------------------------------------------------- #
def _outcome(status: str, *, failure: str = "", fom=None) -> WaveOutcome:
    pat = random_genome(random.Random(4), "K1_K2", 29).to_pattern()  # feed 117
    return WaveOutcome(
        status=status,
        fom=fom,
        n_cycles=0,
        tolerance_margin=None,
        wall_s=0.0,
        restart_provenance="native:MAS_RST.X",
        failure=failure,
        converged_at_cap=False,
        case_key=CaseKey("K1_K2", 117),
        pattern=pat,
    )


def test_classify_outcome_four_way_taxonomy() -> None:
    assert classify_outcome(_outcome("converged")) == "converged"
    assert classify_outcome(_outcome("nonconverged")) == "nonconverged"
    # A watchdog physics kill is an honest negative, never a harness fault.
    assert classify_outcome(_outcome("error", failure="non_finite_flux")) == "nonfinite"
    # Any other error IS a harness fault (staging / deck / resolver / genome).
    assert classify_outcome(
        _outcome("error", failure="RuntimeError: stub injected failure")
    ) == "harness_error"
    assert classify_outcome(
        _outcome("error", failure="DeckValidationError: missing %EXE_DEP")
    ) == "harness_error"
    assert "non_finite_flux" in PHYSICS_KILL_FAILURES


def test_nonfinite_outcome_record_shape() -> None:
    """A non_finite_flux outcome -> an invalid P row whose convergence label is
    False and whose failure is retained, so it feeds the convergence classifier
    exactly like a plain nonconverged negative (plan 5.4, requirement 3)."""

    nonfinite = outcome_to_record(
        _outcome("error", failure="non_finite_flux"), library_id="ga80"
    )
    assert nonfinite.converged is False
    assert nonfinite.failure == "non_finite_flux"
    assert nonfinite.valid is False            # invalid *as a converged label*
    assert nonfinite.f_r is None and nonfinite.cyclen is None
    # Its convergence label matches a plain nonconverged negative's.
    nonconv = outcome_to_record(_outcome("nonconverged"), library_id="ga80")
    assert nonconv.converged is False and nonfinite.converged is False


# --------------------------------------------------------------------------- #
# wave fan-out concurrency (M2.5): a wave of K entries must run min(K, workers)
# chains truly in parallel, each on its own worker/evaluator.  Guards against a
# regression that serialises the ThreadPoolExecutor fan-out (e.g. a wrong
# ``max_workers``, per-case grouping, a shared lock, or evaluator-index reuse).
# --------------------------------------------------------------------------- #
class _ConcurrencyMeter:
    """Shared atomic counter recording peak concurrent entries + workers seen."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.max_concurrent = 0
        self.workers_seen: set[int] = set()


class _BarrierStub:
    """Instrumented evaluator: records concurrency and blocks on a barrier.

    ``barrier`` has ``parties == expected concurrency``; every entry that reaches
    :meth:`evaluate` waits there.  If the fan-out serialises (fewer than
    ``parties`` entries run at once) the barrier trips its timeout and raises,
    which the verifier turns into an ``error`` outcome — so the test *fails
    loudly* instead of deadlocking.
    """

    _BARRIER_TIMEOUT = 15.0

    def __init__(self, worker_id: int, meter: _ConcurrencyMeter, barrier: threading.Barrier) -> None:
        self.worker_id = worker_id
        self.meter = meter
        self.barrier = barrier
        self._inner = StubEvaluator()

    def evaluate(self, case, pattern):
        m = self.meter
        with m._lock:
            m.current += 1
            m.max_concurrent = max(m.max_concurrent, m.current)
            m.workers_seen.add(self.worker_id)
        try:
            # Blocks until `parties` entries are concurrently in flight; a
            # serialised fan-out never fills the barrier and this raises.
            self.barrier.wait(timeout=self._BARRIER_TIMEOUT)
            return self._inner.evaluate(case, pattern)
        finally:
            with m._lock:
                m.current -= 1


def _measure_wave_concurrency(
    tmp_path: Path, *, k: int, workers: int, overall_timeout: float = 30.0
) -> tuple[_ConcurrencyMeter, list]:
    rng = random.Random(1000 * k + workers)
    patterns = _unique_patterns(rng, "K1_K2", 30, k)
    meter = _ConcurrencyMeter()
    expected = min(k, workers)
    barrier = threading.Barrier(expected)
    verifier = WaveVerifier(
        run_dir=tmp_path,
        evaluator_factory=lambda worker_id, cpu_core: _BarrierStub(worker_id, meter, barrier),
        workers=workers,
    )
    case_key = CaseKey("K1_K2", 121)
    entries = [
        WaveEntry(pat, case_key, _resolved(case_key, "native:MAS_RST.X"), {"i": i})
        for i, pat in enumerate(patterns)
    ]

    box: dict[str, list] = {}
    worker = threading.Thread(target=lambda: box.__setitem__("out", verifier.evaluate_wave(entries)))
    worker.start()
    worker.join(overall_timeout)
    assert not worker.is_alive(), (
        f"evaluate_wave did not finish within {overall_timeout}s "
        f"(K={k}, workers={workers}) — fan-out deadlock/serialisation"
    )
    return meter, box["out"]


def test_wave_fanout_runs_k_equals_workers_concurrently(tmp_path: Path) -> None:
    """K=8 entries, workers=8 -> exactly 8 chains run at once, one per worker."""

    meter, outcomes = _measure_wave_concurrency(tmp_path, k=8, workers=8)
    assert len(outcomes) == 8
    # No entry became an "error": the 8-party barrier filled, proving 8-way.
    assert all(o.status != "error" for o in outcomes)
    assert meter.max_concurrent == 8
    assert meter.workers_seen == set(range(8))


def test_wave_fanout_runs_min_k_workers_concurrently(tmp_path: Path) -> None:
    """K=3 entries, workers=8 -> min(K, workers)=3 chains run concurrently."""

    meter, outcomes = _measure_wave_concurrency(tmp_path, k=3, workers=8)
    assert len(outcomes) == 3
    assert all(o.status != "error" for o in outcomes)
    assert meter.max_concurrent == 3
    assert meter.workers_seen == set(range(3))


def test_wave_fanout_runs_23_way_concurrently(tmp_path: Path) -> None:
    """use_all_cores gives 23 workers on the 8P+16E box (host_reserve=1): a wave
    of 23 must run all 23 chains at once, one per worker.  Guards the all-cores
    wave-sizing against any lingering hardcoded-8 fan-out (plan directive)."""

    meter, outcomes = _measure_wave_concurrency(
        tmp_path, k=23, workers=23, overall_timeout=45.0
    )
    assert len(outcomes) == 23
    assert all(o.status != "error" for o in outcomes)
    assert meter.max_concurrent == 23
    assert meter.workers_seen == set(range(23))


def test_feed_general_record_batch_shape(tmp_path: Path) -> None:
    """A feed-117 (3-batch) outcome records depth2_edges=2, n_batches=3."""

    rng = random.Random(5)
    pat = random_genome(rng, "K1_K2", 29).to_pattern()  # feed 117
    assert pat.feed == 117
    stub = StubEvaluator()
    verifier = WaveVerifier(
        run_dir=tmp_path,
        evaluator_factory=lambda worker_id, cpu_core: stub,
        workers=2,
    )
    case_key = CaseKey("K1_K2", 117)
    entry = WaveEntry(pat, case_key, _resolved(case_key, "pair_feed:MAS_RST.F117"))
    (outcome,) = verifier.evaluate_wave([entry])
    record = outcome_to_record(outcome, library_id="ga80")
    assert record.feed == 117
    assert record.n_batches == 3
    assert record.depth2_edges == 2
    # feed-scaled cyclen is shorter than the feed-121 band midpoint.
    assert outcome.fom.cyclen < 720.0


# --------------------------------------------------------------------------- #
# per-entry prep/validation failure is isolated to that entry (plan 5.4)
# --------------------------------------------------------------------------- #
def _write_deck(root: Path, name: str, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    p.write_text(text, encoding="utf-8")
    return p


def test_wave_prep_failure_isolated_and_others_staged(tmp_path: Path) -> None:
    """A wave of 8 with staging: one bad-deck entry -> that entry errors, the
    other 7 stage + evaluate normally (one bad candidate must not sink the wave).
    """

    rng = random.Random(717)
    patterns = _unique_patterns(rng, "K1_K2", 29, 8)  # feed 117 -> 9-line SHF

    good_template = _write_deck(tmp_path / "good", "MAS_INP_cy12.inp", VALID_RELOAD_DECK)
    bad_text = VALID_RELOAD_DECK.replace(
        "%EXE_DEP                                            # BOC\n        0.0\n", ""
    )
    bad_template = _write_deck(tmp_path / "bad", "MAS_INP_cy12.inp", bad_text)
    restart = tmp_path / "MAS_RST.SEED.01"
    restart.write_bytes(b"restart")

    def _resolved_real(template: Path) -> ResolvedAssets:
        return ResolvedAssets(
            case_key=CaseKey("K1_K2", 117),
            restart_path=restart,
            template_deck_path=template,
            fallback_level=0,
            restart_provenance="native:MAS_RST.SEED.01",
        )

    bad_index = 3
    entries = []
    for i, pat in enumerate(patterns):
        template = bad_template if i == bad_index else good_template
        entries.append(
            WaveEntry(pat, CaseKey("K1_K2", 117), _resolved_real(template), {"i": i})
        )

    stub = StubEvaluator()
    verifier = WaveVerifier(
        run_dir=tmp_path / "run",
        evaluator_factory=lambda worker_id, cpu_core: stub,
        workers=8,
        stage_decks=True,   # exercise real deck prep + sanity gate
    )

    outcomes = verifier.evaluate_wave(entries)
    assert len(outcomes) == 8

    errors = [o for o in outcomes if o.status == "error"]
    good = [o for o in outcomes if o.status != "error"]
    assert len(errors) == 1
    assert len(good) == 7
    # The one error is the bad-deck entry, labelled by the sanity gate.
    assert errors[0].pattern.digest == patterns[bad_index].digest
    assert "EXE_DEP" in errors[0].failure
    # The 7 good entries were staged (their prepared decks exist on disk).
    staged = list((tmp_path / "run" / "produce_cases").glob("K1_K2_f117/*/MAS_INP_cy*.inp"))
    assert len(staged) == 7


def test_same_pattern_different_restart_stage_distinctly(tmp_path: Path) -> None:
    """The SAME pattern against two restarts stages to two distinct decks, each
    referencing its own restart (cross-restart legs must not race on one file).
    """

    template = _write_deck(tmp_path / "tpl", "MAS_INP_cy12.inp", VALID_RELOAD_DECK)
    rst_a = tmp_path / "MAS_RST.SEED.01"
    rst_a.write_bytes(b"a")
    rst_b = tmp_path / "MAS_RST.SEED.02"
    rst_b.write_bytes(b"b")

    pat = random_genome(random.Random(1), "K1_K2", 29).to_pattern()

    def _entry(restart: Path, prov: str) -> WaveEntry:
        return WaveEntry(
            pat,
            CaseKey("K1_K2", 117),
            ResolvedAssets(
                case_key=CaseKey("K1_K2", 117),
                restart_path=restart,
                template_deck_path=template,
                fallback_level=0,
                restart_provenance=prov,
            ),
        )

    verifier = WaveVerifier(
        run_dir=tmp_path / "run",
        evaluator_factory=lambda worker_id, cpu_core: StubEvaluator(),
        workers=2,
        stage_decks=True,
    )
    outcomes = verifier.evaluate_wave(
        [_entry(rst_a, "native:MAS_RST.SEED.01"), _entry(rst_b, "pair_feed:MAS_RST.SEED.02")]
    )
    assert all(o.status != "error" for o in outcomes)

    decks = sorted((tmp_path / "run" / "produce_cases").glob("K1_K2_f117/*/MAS_INP_cy*.inp"))
    assert len(decks) == 2, "same pattern + 2 restarts must stage 2 distinct decks"
    refs = {d.parent.name.split("__")[-1]: d.read_text(encoding="utf-8") for d in decks}
    assert "MAS_RST.SEED.01" in refs and "MAS_RST.SEED.02" in refs
    # Each staged deck references exactly its own restart (no cross-contamination).
    assert "MAS_RST.SEED.01" in refs["MAS_RST.SEED.01"]
    assert "MAS_RST.SEED.02" not in refs["MAS_RST.SEED.01"]
    assert "MAS_RST.SEED.02" in refs["MAS_RST.SEED.02"]
    assert "MAS_RST.SEED.01" not in refs["MAS_RST.SEED.02"]


# --------------------------------------------------------------------------- #
# NaN watchdog: a diverging MASTER is killed, not left to burn the timeout
# --------------------------------------------------------------------------- #
_FAKE_NAN_MASTER = """\
import time
with open("MAS_OUT", "a", encoding="ascii") as fh:
    while True:
        fh.write("MGOUTER   11   20       NaN          NaN         NaN\\n")
        fh.write("****   0.000      NaN      NaN         NaN\\n")
        fh.flush()
        time.sleep(0.01)
"""


def test_nan_watchdog_kills_diverging_master(tmp_path: Path) -> None:
    """A fake MASTER that emits NaN forever is killed by the watchdog well
    before the (generous) timeout, and a NONFINITE_FLUX sentinel is dropped.
    """

    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "MAS_XSL").write_bytes(b"xsl")
    (pkg / "lib" / "MAS_HFF").write_bytes(b"hff")
    restart = tmp_path / "MAS_RST.FAKE.01"
    restart.write_bytes(b"r")
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake = tmp_path / "fake_nan_master.py"
    fake.write_text(_FAKE_NAN_MASTER, encoding="utf-8")

    deck = (
        "%JOB_TYP\n        1       stead\n        MAS_RST.FAKE.01\n"
        "%LPD_SHF\n        F K1  0,\n%END\n"
    )

    runner = WatchdogMasterRunner(
        pkg,
        [sys.executable, str(fake)],
        work_root=tmp_path / "work",
        timeout=60.0,           # generous -> only the watchdog can end this fast
        cpu_core=None,
        nan_poll_s=0.3,
        nan_streak=6,
    )

    start = time.perf_counter()
    with pytest.raises(MasterRunError):
        runner.run(case_dir, deck_text=deck, restart_path=restart, use_cache=False)
    elapsed = time.perf_counter() - start

    assert elapsed < 20.0, f"watchdog did not kill the run promptly ({elapsed:.1f}s)"
    sentinels = list((tmp_path / "work").glob("*/NONFINITE_FLUX"))
    assert sentinels, "watchdog left no NONFINITE_FLUX sentinel"


def test_nan_watchdog_classifies_outcome(tmp_path: Path) -> None:
    """End-to-end: a diverging entry surfaces as error / failure=non_finite_flux."""

    from lpopt.vendor.masterrl.equilibrium import EquilibriumRunner
    from lpopt.vendor.masterrl.search import EquilibriumEvaluator

    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "MAS_XSL").write_bytes(b"xsl")
    (pkg / "lib" / "MAS_HFF").write_bytes(b"hff")
    (pkg / "bases" / "K1_K2_f117").mkdir(parents=True)
    restart = pkg / "bases" / "K1_K2_f117" / "MAS_RST.SEED.01"
    restart.write_bytes(b"r")
    template = _write_deck(tmp_path / "tpl", "MAS_INP_cy12.inp", VALID_RELOAD_DECK)

    fake = tmp_path / "fake_nan_master.py"
    fake.write_text(_FAKE_NAN_MASTER, encoding="utf-8")

    def factory(worker_id: int, cpu_core):
        master = WatchdogMasterRunner(
            pkg,
            [sys.executable, str(fake)],
            work_root=tmp_path / "work" / f"worker_{worker_id:02d}",
            cache_dir=tmp_path / "cache" / f"worker_{worker_id:02d}",
            timeout=60.0,
            cpu_core=None,
            nan_poll_s=0.3,
            nan_streak=6,
        )
        return EquilibriumEvaluator(EquilibriumRunner(master, max_cycles=2, consecutive=2))

    verifier = WaveVerifier(
        run_dir=tmp_path / "run",
        evaluator_factory=factory,
        workers=1,
        stage_decks=True,
    )
    pat = random_genome(random.Random(9), "K1_K2", 29).to_pattern()
    entry = WaveEntry(
        pat,
        CaseKey("K1_K2", 117),
        ResolvedAssets(
            case_key=CaseKey("K1_K2", 117),
            restart_path=restart,
            template_deck_path=template,
            fallback_level=0,
            restart_provenance="native:MAS_RST.SEED.01",
        ),
    )
    (outcome,) = verifier.evaluate_wave([entry])
    assert outcome.status == "error"
    assert outcome.failure == "non_finite_flux"
    # QC counts it as an honest negative (nonfinite), not a harness fault.
    assert classify_outcome(outcome) == "nonfinite"


# --------------------------------------------------------------------------- #
# all-cores worker/core policy (DIRECTIVE: CPU 100% for MASTER data production)
# P-cores FIRST then E-cores, 1:1 pinned, minus host_reserve; explicit workers
# still caps; legacy (P-only) path unchanged.  Layout is patched so the assertions
# are deterministic on any host (here: 4 P + 6 E = 10 logical).
# --------------------------------------------------------------------------- #
_FAKE_LAYOUT = CoreLayout(performance=(0, 1, 2, 3), efficiency=(4, 5, 6, 7, 8, 9))


@pytest.fixture()
def _patched_layout(monkeypatch: pytest.MonkeyPatch) -> CoreLayout:
    monkeypatch.setattr(verify_mod, "detect_core_layout", lambda: _FAKE_LAYOUT)
    return _FAKE_LAYOUT


def test_all_cores_assignment_p_then_e_reserve(tmp_path: Path, _patched_layout) -> None:
    """use_all_cores fills every logical core P-FIRST then E, minus host_reserve;
    each worker is pinned 1:1 and tagged with its CPU class."""

    v = WaveVerifier(run_dir=tmp_path, use_all_cores=True, host_reserve=1)
    # 10 logical - 1 reserved = 9 workers: the 4 P-cores first, then 5 E-cores.
    assert v.n_workers == 9
    assert v.worker_cores == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert v.worker_core_class == ["P", "P", "P", "P", "E", "E", "E", "E", "E"]
    # the host retreats to exactly the reserved core.
    assert v.host_cores == (9,)


def test_all_cores_host_reserve_two(tmp_path: Path, _patched_layout) -> None:
    v = WaveVerifier(run_dir=tmp_path, use_all_cores=True, host_reserve=2)
    assert v.n_workers == 8
    assert v.worker_cores == [0, 1, 2, 3, 4, 5, 6, 7]
    assert set(v.host_cores) == {8, 9}


def test_all_cores_explicit_workers_caps(tmp_path: Path, _patched_layout) -> None:
    """An explicit ``workers`` still caps; idle pool cores join the host window."""

    v = WaveVerifier(run_dir=tmp_path, use_all_cores=True, host_reserve=1, workers=5)
    assert v.n_workers == 5
    assert v.worker_cores == [0, 1, 2, 3, 4]         # P-cores first, then 1 E-core
    assert v.worker_core_class == ["P", "P", "P", "P", "E"]
    # host = reserved(9) + the idle pool cores the cap left free (5..8).
    assert set(v.host_cores) == {5, 6, 7, 8, 9}


def test_legacy_p_only_unchanged(tmp_path: Path, _patched_layout) -> None:
    """use_all_cores=False keeps the historic P-cores-only policy (host on E)."""

    v = WaveVerifier(run_dir=tmp_path, use_all_cores=False)
    assert v.n_workers == 4
    assert v.worker_cores == [0, 1, 2, 3]
    assert v.worker_core_class == ["P", "P", "P", "P"]
    assert set(v.host_cores) == {4, 5, 6, 7, 8, 9}
    # explicit cap in legacy mode.
    v2 = WaveVerifier(run_dir=tmp_path, use_all_cores=False, workers=2)
    assert v2.n_workers == 2
    assert v2.worker_cores == [0, 1]


def test_outcome_core_class_tagging(
    tmp_path: Path, _patched_layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With assign_cores=True (the curriculum's real-factory path), each outcome
    carries the CPU class of the worker that ran it — P for P-core workers, E for
    E-core workers — so wall-time stats can separate the two (accounting only)."""

    # Don't touch this test process's real affinity while exercising the wave.
    monkeypatch.setattr(
        verify_mod, "host_affinity", lambda cores: contextlib.nullcontext(False)
    )
    stub = StubEvaluator()
    v = WaveVerifier(
        run_dir=tmp_path,
        evaluator_factory=lambda worker_id, cpu_core: stub,
        assign_cores=True,          # real injected factory wants P/E assignment
        use_all_cores=True,
        host_reserve=1,
    )
    # 10 logical - 1 = 9 workers, but we only send 5 entries -> first 5 cores:
    # cores [0,1,2,3,4] -> classes [P,P,P,P,E].
    assert v.worker_core_class[:5] == ["P", "P", "P", "P", "E"]

    patterns = _unique_patterns(random.Random(7), "K1_K2", 30, 5)
    case_key = CaseKey("K1_K2", 121)
    entries = [
        WaveEntry(pat, case_key, _resolved(case_key, "native:MAS_RST.X"))
        for pat in patterns
    ]
    outcomes = v.evaluate_wave(entries)
    assert [o.core_class for o in outcomes] == ["P", "P", "P", "P", "E"]


def test_stub_outcomes_have_blank_core_class(tmp_path: Path) -> None:
    """A plain stub run (no injected assign_cores) does no pinning, so outcomes
    carry an empty core_class — the neutral 'unpinned' tag."""

    stub = StubEvaluator()
    v = WaveVerifier(
        run_dir=tmp_path,
        evaluator_factory=lambda worker_id, cpu_core: stub,
        workers=3,
    )
    assert v.worker_core_class == ["", "", ""]
    patterns = _unique_patterns(random.Random(3), "K1_K2", 30, 3)
    case_key = CaseKey("K1_K2", 121)
    entries = [WaveEntry(p, case_key, _resolved(case_key, "native:X")) for p in patterns]
    outcomes = v.evaluate_wave(entries)
    assert all(o.core_class == "" for o in outcomes)


# --------------------------------------------------------------------------- #
# EDIT5 map harvesting into the campaign store (forensic 20260723)
# --------------------------------------------------------------------------- #
def _mas_sum_dir() -> Path:
    hits = list(Path("runs").glob("*/master_work/*/*/MAS_SUM"))
    return hits[0].parent if hits else None


def test_harvest_maps_forces_keep_success():
    v = WaveVerifier(run_dir=Path("."), harvest_maps=True)
    assert v.harvest_maps is True and v.keep_success is True
    # default: no harvest, keep_success stays False (byte-identical).
    v0 = WaveVerifier(run_dir=Path("."))
    assert v0.harvest_maps is False and v0.keep_success is False


def test_extract_maps_from_real_mas_sum_and_graceful_none():
    import types
    import numpy as np
    d = _mas_sum_dir()
    if d is None:
        pytest.skip("no MAS_SUM fixture under runs/")
    res = types.SimpleNamespace(cycles=[types.SimpleNamespace(work_dir=d)],
                                retained_work_dirs=())
    maps = WaveVerifier._extract_maps(res)
    assert maps is not None and maps.shape == (4, 9, 9)
    assert np.isfinite(maps).any()
    # missing dir / bad result -> None, never raises.
    bad = types.SimpleNamespace(cycles=[types.SimpleNamespace(work_dir=Path("nope"))],
                                retained_work_dirs=())
    assert WaveVerifier._extract_maps(bad) is None
    assert WaveVerifier._extract_maps(types.SimpleNamespace()) is None


def test_outcome_to_record_maps_key_set_iff_maps_present():
    import numpy as np
    rng = random.Random(3)
    pat = random_genome(rng, "K1_K2", 30).to_pattern()
    base = dict(status="converged", fom=None, n_cycles=3, tolerance_margin=0.1,
               wall_s=1.0, restart_provenance="native:x", failure="",
               converged_at_cap=False, case_key=CaseKey("K1_K2", 121), pattern=pat)
    with_maps = WaveOutcome(**base, maps=np.zeros((4, 9, 9), dtype=np.float32))
    rec = outcome_to_record(with_maps, library_id="ga80")
    assert rec.maps_key == rec.record_id           # maps present -> keyed
    without = WaveOutcome(**base)                   # maps defaults None
    assert outcome_to_record(without, library_id="ga80").maps_key is None


def test_harvesting_evaluator_puts_maps_in_metadata_and_flows_to_maps_key():
    # Integration: raw EquilibriumResult (with .cycles -> real MAS_SUM) -> the
    # harvesting evaluator attaches maps to metadata -> outcome_to_record sets
    # maps_key (the vendor EvaluationResult drops .cycles, so this is the ONLY
    # place the map survives — forensic 20260723).
    import types
    import numpy as np
    from dataclasses import replace as _rp
    from lpopt.search.verify import HarvestingEquilibriumEvaluator
    d = _mas_sum_dir()
    if d is None:
        pytest.skip("no MAS_SUM fixture under runs/")

    # a minimal dataclass FOM stand-in that supports dataclasses.replace.
    from lpopt.vendor.masterrl.domain import FOM
    import dataclasses as _dc
    fom_fields = {f.name: (0.0 if f.type in ("float", "float | None") else None)
                  for f in _dc.fields(FOM)}
    fom_fields["converged"] = True
    fom = FOM(**{k: v for k, v in fom_fields.items()})

    tol = types.SimpleNamespace(as_dict=lambda: {})
    raw = types.SimpleNamespace(fom=fom, converged=True, master_process_calls=3,
                                n_cycles=12, comparisons=[], tolerances=tol,
                                cycles=[types.SimpleNamespace(work_dir=d)],
                                retained_work_dirs=())

    class _Runner:
        def run(self, case, pattern):
            return raw

    ev = HarvestingEquilibriumEvaluator(_Runner())
    ev.runner = _Runner()
    result = ev.evaluate(None, None)
    assert "maps" in result.metadata
    assert result.metadata["maps"].shape == (4, 9, 9)
    assert result.metadata["converged"] is True and result.metadata["mode"] == "equilibrium_master"

    # a non-converged raw result harvests nothing.
    raw_nc = types.SimpleNamespace(fom=_rp(fom, converged=False), converged=False,
                                   master_process_calls=1, n_cycles=14, comparisons=[],
                                   tolerances=tol,
                                   cycles=[types.SimpleNamespace(work_dir=d)],
                                   retained_work_dirs=())

    class _RunnerNC:
        def run(self, case, pattern):
            return raw_nc
    ev2 = HarvestingEquilibriumEvaluator(_RunnerNC())
    ev2.runner = _RunnerNC()
    assert "maps" not in ev2.evaluate(None, None).metadata


# --------------------------------------------------------------------------- #
# F_xy harvest (design 20260829 §3.2-B): the final cycle's MAS_OUT is the ONLY
# source of FXYP, and it is on disk for free at exactly the same moment as the
# EDIT5 map.  Same never-abort contract as ``maps``.
# --------------------------------------------------------------------------- #
def _mas_out_dir() -> Path:
    hits = list(Path("runs").glob("*/master/master_work/*/*/MAS_OUT"))
    return hits[0].parent if hits else None


def test_fxy_harvest_from_real_mas_out_and_graceful_none(tmp_path: Path) -> None:
    import types
    from lpopt.search.verify import _fxy_from_equilibrium_result
    d = _mas_out_dir()
    if d is None:
        pytest.skip("no MAS_OUT fixture under runs/")
    res = types.SimpleNamespace(cycles=[types.SimpleNamespace(work_dir=d)],
                                retained_work_dirs=())
    peaks = _fxy_from_equilibrium_result(res)
    assert peaks is not None and peaks.sane and 1.0 <= peaks.f_xy <= 3.0

    # missing dir / shapeless result -> None, never raises.
    bad = types.SimpleNamespace(cycles=[types.SimpleNamespace(work_dir=tmp_path / "no")],
                                retained_work_dirs=())
    assert _fxy_from_equilibrium_result(bad) is None
    assert _fxy_from_equilibrium_result(types.SimpleNamespace()) is None

    # a physics-killed dir is refused even though its MAS_OUT parses.
    killed = tmp_path / "killed"
    killed.mkdir()
    shutil.copy2(d / "MAS_OUT", killed / "MAS_OUT")
    (killed / "NONFINITE_FLUX").write_text("", encoding="ascii")
    dead = types.SimpleNamespace(cycles=[types.SimpleNamespace(work_dir=killed)],
                                 retained_work_dirs=())
    assert _fxy_from_equilibrium_result(dead) is None


def test_fxy_flows_metadata_to_outcome_to_record_columns() -> None:
    import types
    from dataclasses import replace as _rp
    from lpopt.search.verify import HarvestingEquilibriumEvaluator
    d = _mas_out_dir()
    if d is None:
        pytest.skip("no MAS_OUT fixture under runs/")

    from lpopt.vendor.masterrl.domain import FOM
    import dataclasses as _dc
    fom_fields = {f.name: (0.0 if f.type in ("float", "float | None") else None)
                  for f in _dc.fields(FOM)}
    fom_fields["converged"] = True
    fom = FOM(**fom_fields)
    tol = types.SimpleNamespace(as_dict=lambda: {})
    raw = types.SimpleNamespace(fom=fom, converged=True, master_process_calls=3,
                                n_cycles=11, comparisons=[], tolerances=tol,
                                cycles=[types.SimpleNamespace(work_dir=d)],
                                retained_work_dirs=())

    class _Runner:
        def run(self, case, pattern):
            return raw

    result = HarvestingEquilibriumEvaluator(_Runner()).evaluate(None, None)
    assert "fxy" in result.metadata
    peaks = result.metadata["fxy"]
    assert peaks.f_xy is not None and peaks.f_xya is not None

    # ... and it lands in the record columns (same atomic write as f_r/cyclen).
    rng = random.Random(11)
    pat = random_genome(rng, "K1_K2", 30).to_pattern()
    base = dict(status="converged", fom=None, n_cycles=11, tolerance_margin=0.1,
                wall_s=1.0, restart_provenance="native:x", failure="",
                converged_at_cap=False, case_key=CaseKey("K1_K2", 121), pattern=pat)
    rec = outcome_to_record(WaveOutcome(**base, fxy=peaks), library_id="ga80")
    assert rec.f_xy == peaks.f_xy and rec.f_xya == peaks.f_xya
    # no harvest -> honest nulls, never a substituted F_r.
    bare = outcome_to_record(WaveOutcome(**base), library_id="ga80")
    assert bare.f_xy is None and bare.f_xya is None

    # a NON-converged chain harvests nothing (its FXYP is not an equilibrium value)
    raw_nc = types.SimpleNamespace(fom=_rp(fom, converged=False), converged=False,
                                   master_process_calls=1, n_cycles=14,
                                   comparisons=[], tolerances=tol,
                                   cycles=[types.SimpleNamespace(work_dir=d)],
                                   retained_work_dirs=())

    class _RunnerNC:
        def run(self, case, pattern):
            return raw_nc
    assert "fxy" not in HarvestingEquilibriumEvaluator(_RunnerNC()).evaluate(
        None, None).metadata
