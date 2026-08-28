"""produce driver: dry-run store/ledger consistency, per-stratum counts,
crash-safe kill-resume (no dup, no loss), idempotent re-run, and the real CLI
end-to-end dry-run (plan 5.4 / M2.5 acceptance 2).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
import json
import random
from pathlib import Path

import pandas as pd
import pytest

from lpopt.config import (
    CaseConfig,
    DataConfig,
    ExtractConfig,
    FlowConfig,
    FuelConfig,
    LpoptConfig,
    MasterConfig,
    ProduceConfig,
    RemoteConfig,
    StratumConfig,
    VerifyConfig,
    load_config,
)
from lpopt.data.store import StoreReader
from lpopt.search.assets import CaseAssetResolver
from lpopt.search.produce import Ledger, ProduceDriver, _StratumState, _normalize_mix
from lpopt.search.stub import StubEvaluator
from lpopt.search.verify import WaveOutcome
from lpopt.vendor.masterrl.domain import CaseKey

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_package(tmp_path: Path) -> Path:
    """A minimal MASTER package: one native K1_K2 restart + a readable deck."""

    pkg = tmp_path / "pkg"
    base = pkg / "bases" / "K1_K2"
    base.mkdir(parents=True)
    (base / "MAS_RST.NATIVE.01").write_bytes(b"rst")
    core = pkg / "cores" / "K1_K2" / "s1"
    core.mkdir(parents=True)
    (core / "MAS_INP_cy01.inp").write_text(
        "dummy\n%LPD_SHF\n F K1  0,\n%END\n", encoding="utf-8"
    )
    return pkg


def _make_cfg(tmp_path: Path, pkg: Path, *, seed: int = 3, workers: int = 6) -> LpoptConfig:
    strata = [
        StratumConfig(name="s_f121", pairs=["K1_K2"], feed=121, n_target=10,
                      generators={"random": 1.0}, priority=100),
        StratumConfig(name="s_f117", pairs=["K1_K2"], feed=117, n_target=10,
                      generators={"random": 0.7, "heuristic": 0.3}, priority=100),
        StratumConfig(name="s_f113", pairs=["K1_K2"], feed=113, n_target=10,
                      generators={"random": 1.0}, priority=90),
    ]
    return LpoptConfig(
        flow=FlowConfig(random_seed=seed),
        remote=RemoteConfig(),
        master=MasterConfig(),
        verify=VerifyConfig(package_root=str(pkg)),
        data=DataConfig(),
        case=CaseConfig(),
        fuel=FuelConfig(),
        extract=ExtractConfig(),
        produce=ProduceConfig(
            campaign="test", workers=workers,
            template_fallbacks=[], strata=strata,
        ),
        source_path=tmp_path / "lpopt.inp",
    )


def _driver(tmp_path: Path, cfg: LpoptConfig, tag: str) -> ProduceDriver:
    return ProduceDriver(
        cfg,
        dry_run=True,
        run_dir=tmp_path / f"run_{tag}",
        store_dir=tmp_path / "store",
        ledger_path=tmp_path / "ledger.jsonl",
        progress=False,
    )


def _ledger_terminal_counts(path: Path) -> tuple[int, int]:
    done = err = 0
    for row in Ledger.replay(path):
        if row.get("status") == "done":
            done += 1
        elif row.get("status") == "error":
            err += 1
    return done, err


# --------------------------------------------------------------------------- #
# consistency
# --------------------------------------------------------------------------- #
def test_dry_run_store_ledger_consistency(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    cfg = _make_cfg(tmp_path, pkg)
    summary = _driver(tmp_path, cfg, "a").run()

    assert summary.chains == 30
    assert summary.converged == 30
    assert summary.nonconverged == 0 and summary.errors == 0

    df = StoreReader(tmp_path / "store").records
    p = df[df["dataset"] == "P"]
    assert len(p) == 30
    assert p["record_id"].nunique() == 30, "duplicate record_ids in store"
    # feed 121 resolves natively; feed 117/113 have no exact folder, so they fall
    # to level-2 (same pair, nearest feed 121) — both off the one native restart.
    assert all(prov.endswith("MAS_RST.NATIVE.01") for prov in p["restart_provenance"])
    assert set(prov.split(":")[0] for prov in p["restart_provenance"]) == {"native", "pair_feed"}
    assert dict(p.groupby("stratum")["record_id"].count()) == {
        "s_f121": 10, "s_f117": 10, "s_f113": 10,
    }
    # feed-general shape wrote through.
    assert dict(p.groupby("feed")["record_id"].count()) == {121: 10, 117: 10, 113: 10}

    done, err = _ledger_terminal_counts(tmp_path / "ledger.jsonl")
    assert done == 30 and err == 0


# --------------------------------------------------------------------------- #
# crash-safe resume
# --------------------------------------------------------------------------- #
def test_kill_resume_no_dup_no_loss(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)

    # First invocation: partial (12 chains), then simulate a torn ledger tail.
    cfg1 = _make_cfg(tmp_path, pkg, seed=1)
    summary1 = _driver(tmp_path, cfg1, "part").run(max_chains=12)
    assert summary1.chains == 12

    df1 = StoreReader(tmp_path / "store").records
    done_before = set(df1[df1["dataset"] == "P"]["record_id"])
    assert len(done_before) == 12

    ledger_path = tmp_path / "ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert lines, "ledger should have entries after a partial run"
    # Simulate kill -9: keep complete lines, leave the last one torn mid-JSON.
    kept = lines[:-1]
    torn_tail = lines[-1][: len(lines[-1]) // 2]
    ledger_path.write_text(
        "\n".join(kept) + (f"\n{torn_tail}" if torn_tail else ""), encoding="utf-8"
    )

    # Resume (fresh process => different seed) to completion.
    cfg2 = _make_cfg(tmp_path, pkg, seed=999)
    summary2 = _driver(tmp_path, cfg2, "resume").run()

    df2 = StoreReader(tmp_path / "store").records
    p = df2[df2["dataset"] == "P"]
    # No duplicates, exactly the target, and every pre-kill label preserved.
    assert p["record_id"].nunique() == len(p)
    assert dict(p.groupby("stratum")["record_id"].count()) == {
        "s_f121": 10, "s_f117": 10, "s_f113": 10,
    }
    assert len(p) == 30
    assert done_before <= set(p["record_id"]), "a pre-kill done label was lost"


def test_idempotent_rerun(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    cfg = _make_cfg(tmp_path, pkg, seed=7)
    first = _driver(tmp_path, cfg, "first").run()
    assert first.chains == 30

    df_first = StoreReader(tmp_path / "store").records
    ids_first = set(df_first[df_first["dataset"] == "P"]["record_id"])

    # A second run over the same store/ledger has nothing left to do.
    cfg2 = _make_cfg(tmp_path, pkg, seed=8)
    second = _driver(tmp_path, cfg2, "second").run()
    assert second.chains == 0

    df_second = StoreReader(tmp_path / "store").records
    ids_second = set(df_second[df_second["dataset"] == "P"]["record_id"])
    assert ids_first == ids_second, "idempotent re-run mutated the store"


# --------------------------------------------------------------------------- #
# config parsing + real CLI end-to-end
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# generator feed invariant: every candidate carries the stratum's N (plan 6.1)
# --------------------------------------------------------------------------- #
def _single_stratum_cfg(
    tmp_path: Path, pkg: Path, strat: StratumConfig, *, seed: int = 3
) -> LpoptConfig:
    return LpoptConfig(
        flow=FlowConfig(random_seed=seed),
        remote=RemoteConfig(),
        master=MasterConfig(),
        verify=VerifyConfig(package_root=str(pkg)),
        data=DataConfig(),
        case=CaseConfig(),
        fuel=FuelConfig(),
        extract=ExtractConfig(),
        produce=ProduceConfig(
            campaign="test", workers=4, template_fallbacks=[], strata=[strat]
        ),
        source_path=tmp_path / "lpopt.inp",
    )


@pytest.mark.parametrize("generator", ["random", "heuristic", "elite_perturb"])
@pytest.mark.parametrize("feed,n_fresh", [(109, 27), (117, 29), (121, 30), (125, 31)])
def test_generator_respects_stratum_feed(
    tmp_path: Path, generator: str, feed: int, n_fresh: int
) -> None:
    """Each generator x N in {27,29,30,31} x 50 draws -> every pattern's weighted
    feed equals the stratum feed (mutate/elite feed-morph must not drift N)."""

    pkg = _make_package(tmp_path)

    # Populate the store with converged K1_K2 elites so elite_perturb exercises
    # its from_pattern -> morph -> mutate path (not the no-elite random fallback).
    seed_cfg = _single_stratum_cfg(
        tmp_path,
        pkg,
        StratumConfig(name="seed_f121", pairs=["K1_K2"], feed=121, n_target=24,
                      generators={"random": 1.0}, priority=100),
    )
    _driver(tmp_path, seed_cfg, "seed").run()

    discharge = n_fresh > 30
    strat = StratumConfig(
        name=f"{generator}_{feed}", pairs=["K1_K2"], feed=feed, n_target=50,
        generators={generator: 1.0}, priority=100,
        allow_single_cycle_discharge=discharge,
    )
    cfg = _single_stratum_cfg(tmp_path, pkg, strat)
    driver = ProduceDriver(
        cfg, dry_run=True, run_dir=tmp_path / f"run_{generator}_{feed}",
        store_dir=tmp_path / "store", ledger_path=tmp_path / "ledger.jsonl",
        progress=False,
    )

    state = _StratumState(cfg=strat)
    state.effective_n_target = 50
    state.effective_generators = _normalize_mix(strat.generators)

    rng = random.Random(20260716)
    feeds: list[int] = []
    for _ in range(50):
        drawn = driver._generate(state, rng)
        if drawn is None:
            continue
        pattern, _gen, _parent, _split, _pair = drawn
        feeds.append(pattern.feed)

    assert len(feeds) >= 40, f"{generator} feed {feed}: too few valid draws ({len(feeds)})"
    off = sorted({f for f in feeds if f != feed})
    assert not off, f"{generator} feed {feed}: off-feed patterns produced: {off}"


def test_real_deck_produce_section_parses() -> None:
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    assert len(cfg.produce.strata) == 8
    names = {s.name for s in cfg.produce.strata}
    assert "p0_feed105" in names and "p0_feed125_discharge" in names
    discharge = next(s for s in cfg.produce.strata if s.name == "p0_feed125_discharge")
    assert discharge.feed == 125 and discharge.allow_single_cycle_discharge is True
    # The all-cores directive: the deck drives CPU 100% for production.
    assert cfg.produce.use_all_cores is True
    assert cfg.produce.workers == 0            # 0 => auto (fill the core pool)
    assert cfg.produce.host_reserve == 1


def test_core_policy_defaults() -> None:
    """Data PRODUCTION defaults to all cores; OPTIMIZE ([master]) stays P-only."""

    p = ProduceConfig()
    assert p.use_all_cores is True and p.workers == 0 and p.host_reserve == 1
    m = MasterConfig()
    assert m.use_all_cores is False and m.workers == 0 and m.host_reserve == 1


class _FixedWorkerVerifier:
    """Fake verifier exposing a fixed ``n_workers`` (no MASTER): proves the driver
    sizes waves to the verifier's ACTUAL worker count, not the [produce] knob."""

    def __init__(self, n_workers: int) -> None:
        self.n_workers = int(n_workers)
        self.cases_dir = Path("does_not_exist_produce_cases")

    def evaluate_wave(self, entries):
        return []


def test_wave_size_follows_verifier_worker_count(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    cfg = _make_cfg(tmp_path, pkg, workers=8)   # [produce] workers says 8...
    driver = ProduceDriver(
        cfg, dry_run=True, run_dir=tmp_path / "run",
        store_dir=tmp_path / "store", ledger_path=tmp_path / "ledger.jsonl",
        verifier=_FixedWorkerVerifier(23), progress=False,
    )
    # ...but the all-cores verifier reports 23 workers, so waves fill to 23.
    assert driver.workers == 23


def test_core_class_accounting_and_ledger(tmp_path: Path) -> None:
    """Every chain is tagged with a CPU class in the summary AND the ledger row
    (accounting only).  A dry-run/stub chain is unpinned -> '?' / blank + wall_s."""

    pkg = _make_package(tmp_path)
    cfg = _make_cfg(tmp_path, pkg)
    summary = _driver(tmp_path, cfg, "cc").run()

    # summary counters cover every chain and reconcile with the total.
    assert sum(summary.core_class_chains.values()) == summary.chains == 30
    assert set(summary.core_class_chains) == {"?"}      # stub chains are unpinned
    assert "?" in summary.core_class_wall_s

    # the durable ledger carries core_class + wall_s on every terminal row.
    done_rows = [r for r in Ledger.replay(tmp_path / "ledger.jsonl")
                 if r.get("status") == "done"]
    assert len(done_rows) == 30
    assert all("core_class" in r and "wall_s" in r for r in done_rows)


def test_cli_end_to_end_dry_run(capsys) -> None:
    from lpopt.cli import main

    rc = main(
        ["produce", "--input", str(REPO_ROOT / "lpopt.inp"), "--dry-run", "--max-chains", "50"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "produce summary" in out
    assert "RESULT: OK" in out
    assert "50 chains" in out

    # Locate and remove the run-scoped store the CLI created (no repo pollution).
    run_dir = None
    for line in out.splitlines():
        if line.strip().startswith("ledger:"):
            run_dir = Path(line.split("ledger:", 1)[1].strip()).parent
    assert run_dir is not None and run_dir.exists()
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# QC four-way counters: a non_finite_flux watchdog kill is an HONEST NEGATIVE
# label (plan 5.4 / 4.6 정직 회계), so it is counted separately and NEVER halts a
# stratum — only a genuine harness fault (>10%) halts.  This is the fix for the
# live P0 regression where low-feed strata halted at 20-40% "exceptions" that
# were really non_finite_flux (p0_f117_seed 38%, p0_feed109 21%).
# --------------------------------------------------------------------------- #
class _ClassInjectingVerifier:
    """Fake ``WaveVerifier``: turns a deterministic ``per_ten`` of every 10 wave
    entries into ``error`` outcomes carrying ``failure`` (a nonfinite physics
    kill or a harness fault), the rest into converged StubEvaluator FOMs.  Drives
    the produce driver's QC counting + HALT/advisory rules with no live MASTER.
    """

    def __init__(self, *, failure: str, per_ten: int) -> None:
        self.failure = failure
        self.per_ten = int(per_ten)
        self._stub = StubEvaluator()
        self._n = 0
        # _maybe_purge checks verifier.cases_dir.exists(); a nonexistent path no-ops.
        self.cases_dir = Path("does_not_exist_produce_cases")

    def evaluate_wave(self, entries):
        outcomes = []
        for entry in entries:
            inject = (self._n % 10) < self.per_ten
            self._n += 1
            prov = entry.resolved_assets.restart_provenance
            if inject:
                outcomes.append(
                    WaveOutcome(
                        status="error", fom=None, n_cycles=0, tolerance_margin=None,
                        wall_s=0.0, restart_provenance=prov, failure=self.failure,
                        converged_at_cap=False, case_key=entry.case_key,
                        pattern=entry.pattern, meta=dict(entry.meta),
                    )
                )
            else:
                feed = entry.case_key.feed
                fom = self._stub.fom_for(entry.pattern.digest, feed)
                outcomes.append(
                    WaveOutcome(
                        status="converged", fom=fom,
                        n_cycles=self._stub.n_cycles_for(entry.pattern.digest, feed),
                        tolerance_margin=0.5, wall_s=0.0, restart_provenance=prov,
                        failure="", converged_at_cap=False, case_key=entry.case_key,
                        pattern=entry.pattern, meta=dict(entry.meta),
                    )
                )
        return outcomes


def _run_injected(
    tmp_path: Path, *, failure: str, per_ten: int, n_target: int,
    workers: int = 8, feed: int = 117, seed: int = 5,
) -> tuple[object, list[str]]:
    pkg = _make_package(tmp_path)
    strat = StratumConfig(
        name="p0_lowfeed", pairs=["K1_K2"], feed=feed, n_target=n_target,
        generators={"random": 1.0}, priority=100,
    )
    cfg = LpoptConfig(
        flow=FlowConfig(random_seed=seed),
        remote=RemoteConfig(),
        master=MasterConfig(),
        verify=VerifyConfig(package_root=str(pkg)),
        data=DataConfig(),
        case=CaseConfig(),
        fuel=FuelConfig(),
        extract=ExtractConfig(),
        produce=ProduceConfig(
            campaign="test", workers=workers, template_fallbacks=[], strata=[strat]
        ),
        source_path=tmp_path / "lpopt.inp",
    )
    logs: list[str] = []
    driver = ProduceDriver(
        cfg,
        dry_run=True,
        run_dir=tmp_path / "run",
        store_dir=tmp_path / "store",
        ledger_path=tmp_path / "ledger.jsonl",
        verifier=_ClassInjectingVerifier(failure=failure, per_ten=per_ten),
        progress=False,
        log=logs.append,
    )
    return driver.run(), logs


def _counters_sum_to_chains(summary) -> None:
    assert (
        summary.converged + summary.nonconverged + summary.nonfinite + summary.harness_error
        == summary.chains
    )
    assert summary.errors == summary.nonfinite + summary.harness_error
    for row in summary.strata:
        assert (
            row["converged"] + row["nonconverged"] + row["nonfinite"] + row["harness_error"]
            == row["attempts"]
        )


def test_injected_nonfinite_does_not_halt(tmp_path: Path) -> None:
    """30% non_finite_flux kills must NOT halt the stratum: they are honest
    negatives, so the stratum runs to its converged n_target."""

    summary, logs = _run_injected(
        tmp_path, failure="non_finite_flux", per_ten=3, n_target=12, workers=6
    )
    row = summary.strata[0]
    assert row["stalled"] is False
    assert row["nonfinite"] > 0, "the injected non_finite_flux kills were not counted"
    assert row["harness_error"] == 0
    assert row["produced"] == 12 and row["converged"] == 12
    # Not one HALT / harness-error WARNING was emitted.
    assert not any("HALT stratum" in line for line in logs)
    assert not any("harness-error rate" in line for line in logs)
    # Store row shape: nonfinite chains are invalid P rows with converged=False.
    df = StoreReader(tmp_path / "store").records
    p = df[df["dataset"] == "P"]
    nonfin = p[p["failure"] == "non_finite_flux"]
    assert len(nonfin) == row["nonfinite"]
    assert (nonfin["converged"] == False).all()  # noqa: E712
    assert (nonfin["valid"] == False).all()       # noqa: E712
    _counters_sum_to_chains(summary)


def test_injected_harness_error_halts(tmp_path: Path) -> None:
    """>10% genuine harness faults DO halt the stratum (deck/asset inspection)."""

    summary, logs = _run_injected(
        tmp_path, failure="DeckValidationError: missing %EXE_DEP",
        per_ten=3, n_target=100, workers=4,
    )
    row = summary.strata[0]
    assert row["stalled"] is True
    assert row["harness_error"] > 0
    assert row["nonfinite"] == 0
    assert row["produced"] < 100, "a halted stratum must not reach n_target"
    halt_lines = [line for line in logs if "HALT stratum" in line]
    assert halt_lines, "harness-error HALT WARNING was not emitted"
    assert "harness-error rate" in halt_lines[0]
    _counters_sum_to_chains(summary)


class _NonconvergedInjectingVerifier:
    """Fake ``WaveVerifier``: a deterministic ``per_ten`` of every 10 wave entries
    become HONEST NON-CONVERGENCES (status="nonconverged": valid label,
    converged=False), the rest converged StubEvaluator FOMs.  No errors.  Drives
    the produce driver's CONVERGED-target counting past the non-converged labels.
    """

    def __init__(self, *, per_ten: int) -> None:
        self.per_ten = int(per_ten)
        self._stub = StubEvaluator()
        self._n = 0
        self.cases_dir = Path("does_not_exist_produce_cases")

    def evaluate_wave(self, entries):
        outcomes = []
        for entry in entries:
            nonconv = (self._n % 10) < self.per_ten
            self._n += 1
            prov = entry.resolved_assets.restart_provenance
            if nonconv:
                outcomes.append(
                    WaveOutcome(
                        status="nonconverged", fom=None, n_cycles=14,
                        tolerance_margin=1.7, wall_s=0.0, restart_provenance=prov,
                        failure="", converged_at_cap=False, case_key=entry.case_key,
                        pattern=entry.pattern, meta=dict(entry.meta),
                    )
                )
            else:
                feed = entry.case_key.feed
                fom = self._stub.fom_for(entry.pattern.digest, feed)
                outcomes.append(
                    WaveOutcome(
                        status="converged", fom=fom,
                        n_cycles=self._stub.n_cycles_for(entry.pattern.digest, feed),
                        tolerance_margin=0.5, wall_s=0.0, restart_provenance=prov,
                        failure="", converged_at_cap=False, case_key=entry.case_key,
                        pattern=entry.pattern, meta=dict(entry.meta),
                    )
                )
        return outcomes


def _nonconv_driver(tmp_path, cfg, tag, *, per_ten):
    return ProduceDriver(
        cfg, dry_run=True, run_dir=tmp_path / f"run_{tag}",
        store_dir=tmp_path / "store", ledger_path=tmp_path / "ledger.jsonl",
        verifier=_NonconvergedInjectingVerifier(per_ten=per_ten), progress=False,
    )


def test_resume_counter_equals_store_converged_count(tmp_path: Path) -> None:
    """The produce ``produced`` counter is the STORE's converged count — even when
    honest non-convergences are stored — so a cell can never declare "done" one
    (or more) converged labels short of the curriculum's converged target (the
    5.5-5.75_f117 off-by-one relaunch loop)."""

    pkg = _make_package(tmp_path)
    # one stratum, converged target 12, ~30% honest non-convergences injected.
    strat = StratumConfig(name="s_f117", pairs=["K1_K2"], feed=117, n_target=12,
                          generators={"random": 1.0}, priority=100)
    cfg = _single_stratum_cfg(tmp_path, pkg, strat, seed=11)
    summary = _nonconv_driver(tmp_path, cfg, "a", per_ten=3).run()

    # produce ran PAST the non-converged labels to reach 12 CONVERGED, not 12 valid.
    assert summary.converged == 12
    assert summary.nonconverged > 0
    assert summary.chains > 12                      # extra chains for the negatives

    df = StoreReader(tmp_path / "store").records
    p = df[df["dataset"] == "P"]
    store_converged = int((p["converged"] == True).sum())          # noqa: E712
    assert store_converged == 12                     # store == target, no off-by-one
    # honest negatives are preserved as valid, converged=False labels.
    negatives = p[p["converged"] == False]           # noqa: E712
    assert len(negatives) == summary.nonconverged
    assert (negatives["valid"] == True).all()        # noqa: E712

    # A fresh resume reconstructs produced == store converged count -> nothing left
    # to do (the "resumed N/target" the curriculum gates on agrees with the store).
    cfg2 = _single_stratum_cfg(tmp_path, pkg, strat, seed=77)
    resumed = _nonconv_driver(tmp_path, cfg2, "b", per_ten=3)
    states = {strat.name: _StratumState(cfg=strat)}
    states[strat.name].effective_n_target = 12
    dedup, _dups = resumed._reconstruct(states)
    assert states[strat.name].produced == store_converged == 12
    assert states[strat.name].remaining == 0
    summary2 = resumed.run()
    assert summary2.chains == 0                       # idempotent: target already met


def test_rebalance_advisory_fires_without_halt(tmp_path: Path) -> None:
    """A NaN-heavy stratum (>50% nonfinite after 100 chains) logs the rebalancing
    advisory and shifts the generator mix, but keeps producing (no halt)."""

    summary, logs = _run_injected(
        tmp_path, failure="non_finite_flux", per_ten=6, n_target=50, workers=8
    )
    row = summary.strata[0]
    assert row["stalled"] is False
    assert row["nonfinite"] > 0
    assert row["produced"] == 50
    advisory = [line for line in logs if "rebalancing" in line]
    assert advisory, "rebalancing advisory WARNING was not emitted"
    assert "heuristic-centred" in advisory[0]
    assert not any("HALT stratum" in line for line in logs)
    _counters_sum_to_chains(summary)


# --------------------------------------------------------------------------- #
# level-1 promoted-restart cache (feed-grid pathfinder 20260815 §4)
#
# ``CaseAssetResolver.promote()`` had NO caller anywhere in lpopt, so the ladder's
# self-improving tier was never populated and every chain in a new (pair, feed)
# cell re-ran off the cross-feed level-2 restart.  On the f129 column that seed is
# far enough from the cell's equilibrium that ~25-33% of chains die
# non_finite_flux while the SAME deck converges from a cell-native restart.
# --------------------------------------------------------------------------- #
class _PromotionVerifier:
    """Fake ``WaveVerifier`` whose converged outcomes carry ``eq_provenance``.

    Mirrors what ``HarvestingEquilibriumEvaluator`` attaches for a live chain —
    the converged final cycle's ``{deck, restart, work_dir}``, the restart a real
    file on disk.  A feed in ``nonconverged_feeds`` never converges and, exactly
    as ``WaveVerifier._result_to_outcome`` does, carries no provenance at all.
    """

    def __init__(self, work_root: Path, nonconverged_feeds: set[int] = frozenset()) -> None:
        self.work_root = Path(work_root)
        self.nonconverged_feeds = set(nonconverged_feeds)
        self._stub = StubEvaluator()
        self._n = 0
        # _maybe_purge checks verifier.cases_dir.exists(); a nonexistent path no-ops.
        self.cases_dir = Path("does_not_exist_produce_cases")

    def _final_restart(self, entry) -> Path:
        """One converged chain's final cycle dir + the MAS_RST.* it generated."""

        self._n += 1
        work_dir = self.work_root / f"cycle_{self._n:03d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        restart = work_dir / f"MAS_RST.EQ_{entry.case_key.feed}.{self._n:02d}"
        restart.write_bytes(f"equilibrium-restart-{self._n}".encode("ascii"))
        return restart

    def evaluate_wave(self, entries):
        outcomes = []
        for entry in entries:
            feed = entry.case_key.feed
            converged = feed not in self.nonconverged_feeds
            fom = dc_replace(
                self._stub.fom_for(entry.pattern.digest, feed), converged=converged
            )
            restart = self._final_restart(entry) if converged else None
            outcomes.append(
                WaveOutcome(
                    status="converged" if converged else "nonconverged",
                    fom=fom,
                    n_cycles=self._stub.n_cycles_for(entry.pattern.digest, feed),
                    tolerance_margin=0.5 if converged else 1.0,
                    wall_s=0.0,
                    restart_provenance=entry.resolved_assets.restart_provenance,
                    failure="", converged_at_cap=False,
                    case_key=entry.case_key, pattern=entry.pattern,
                    meta=dict(entry.meta),
                    eq_provenance=None if restart is None else {
                        "deck": str(restart.parent / "MAS_INP"),
                        "restart": str(restart),
                        "work_dir": str(restart.parent),
                    },
                )
            )
        return outcomes


def test_converged_fallback_chain_promotes_cell_to_level1(tmp_path: Path) -> None:
    """A converged chain in a level->=2 cell seeds the promoted cache, the next
    chain in that cell resolves at level 1, and a cell that never converges is
    never promoted."""

    pkg = _make_package(tmp_path)                 # one native K1_K2 restart @ f121
    promoted = tmp_path / "promoted"
    strata = [
        # converges -> the cell must be promoted off its first converged chain
        StratumConfig(name="s_f125", pairs=["K1_K2"], feed=125, n_target=10,
                      generators={"random": 1.0}, priority=100,
                      allow_single_cycle_discharge=True),
        # never converges -> nothing may be promoted for it
        StratumConfig(name="s_f109", pairs=["K1_K2"], feed=109, n_target=10,
                      generators={"random": 1.0}, priority=90),
    ]
    cfg = LpoptConfig(
        flow=FlowConfig(random_seed=11),
        remote=RemoteConfig(),
        master=MasterConfig(),
        verify=VerifyConfig(package_root=str(pkg)),
        data=DataConfig(),
        case=CaseConfig(),
        fuel=FuelConfig(),
        extract=ExtractConfig(),
        produce=ProduceConfig(
            campaign="test", workers=6, template_fallbacks=[],
            promoted_root=str(promoted), strata=strata,
        ),
        source_path=tmp_path / "lpopt.inp",
    )

    resolver = CaseAssetResolver(pkg, promoted_root=promoted)
    f125, f109 = CaseKey("K1_K2", 125), CaseKey("K1_K2", 109)
    # Precondition: both cells have only the pair's CROSS-FEED f121 restart.
    assert resolver.resolve(f125).fallback_level == 2
    assert resolver.resolve(f109).fallback_level == 2

    driver = ProduceDriver(
        cfg, dry_run=True, run_dir=tmp_path / "run_promote",
        store_dir=tmp_path / "store", ledger_path=tmp_path / "ledger.jsonl",
        resolver=resolver,
        verifier=_PromotionVerifier(tmp_path / "work", nonconverged_feeds={109}),
        progress=False,
    )
    # f125 (priority 100) fills first in waves of 6 -> 6 + 4 chains; the cap then
    # leaves f109 exactly one all-nonconverged wave.
    summary = driver.run(max_chains=16)
    assert summary.chains == 16
    assert summary.converged == 10 and summary.nonconverged == 6

    # 1. the converged fallback chain promoted the cell (exactly one restart,
    #    atomically -- no temp file left behind).
    dest_dir = promoted / "K1_K2_f125"
    restarts = list(dest_dir.glob("MAS_RST.*"))
    assert len(restarts) == 1, f"expected one promoted restart, got {restarts}"
    assert not list(dest_dir.glob(".*tmp*"))

    # 2. the cell now resolves at LEVEL 1, and the later chains actually ran off
    #    it: the stratum's store rows carry BOTH provenances.
    resolved = resolver.resolve(f125)
    assert resolved.fallback_level == 1
    assert resolved.restart_path == restarts[0]
    df = StoreReader(tmp_path / "store").records
    provs = set(df[df["stratum"] == "s_f125"]["restart_provenance"])
    assert any(p.startswith("pair_feed:") for p in provs), provs
    assert any(p.startswith("promoted:") for p in provs), provs

    # 3. a cell whose chains never converge is NEVER promoted.
    assert not (promoted / "K1_K2_f109").exists()
    assert resolver.resolve(f109).fallback_level == 2
