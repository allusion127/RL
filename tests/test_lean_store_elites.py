"""Bug B regression: the lean screen must inject the store's verified elites.

Evidence (runs/20260718_025717): the lean screen ranked ~800 cells from random /
heuristic candidates only, so K1_K2 — which holds STORE-VERIFIED feasible LPs
matching the criteria — was never deepened and its verified elites never seeded
any pool.  This module seeds a store with a converged, criteria-feasible K1_K2
row and asserts the fix:

1. the store's verified rows are injected as elite parents for the pair (and as
   prediction-only registry candidates), and their record_ids are ledger-deduped
   so they are never re-verified;
2. small-move MUTATION CHILDREN of those elites are generated (NEW, verifiable
   candidates near the known optimum);
3. the "known verified LPs in store matching the criteria" report table lists the
   store LP so the user sees existing solutions even when the new wave finds none.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.config import (
    AcquisitionConfig, CaseConfig, CriteriaConfig, DataConfig, ExtractConfig,
    FlowConfig, FuelConfig, LpoptConfig, MasterConfig, ModelConfig, ProduceConfig,
    RemoteConfig, SearchConfig, VerifyConfig,
)
from lpopt.data.fuel_types import FuelLibrary
from lpopt.data.schema import (
    SYM_CLASS, CanonicalRecord, compute_record_id, pack_pattern,
)
from lpopt.data.store import StoreWriter
from lpopt.search.campaign import (
    UserCriteriaDriver, cell_target_e_core, known_verified_lps_table_md,
)
from lpopt.search.construct import CAMPAIGN_DECK_KNOBS, build_pair_universe
from lpopt.search.genome import random_genome
from lpopt.search.stub import StubEvaluator

# Reuse the fake surrogate + fuel doubles from the user_criteria test module.
from test_user_criteria import FakeModel, _fuel, _TIERS


def _k1k2_pattern(seed: int = 5):
    return random_genome(random.Random(seed), "K1_K2", 30).to_pattern()


def _feasible_record(pattern, *, f_r=1.5444, cyclen=652.6, cbc=1326.0) -> CanonicalRecord:
    """A converged, criteria-feasible K1_K2 store row (the M5-style LP)."""

    rid = compute_record_id(pattern.canonical(), "ga80", "K1_K2", CAMPAIGN_DECK_KNOBS)
    return CanonicalRecord(
        record_id=rid, dataset="P", campaign="20260717_073757", stratum="user_criteria",
        generator="local", parent_record_id=None, case_pair="K1_K2", feed=121,
        n_batches=2, depth2_edges=0, e_core=5.2, e_split=0.0, library_id="ga80",
        sym_class=SYM_CLASS, pattern=pack_pattern(pattern),
        f_r=f_r, f_q=2.10, cbc_max=cbc, cbc_boc=None, cbc_kind="max",
        cyclen=cyclen, ao_abs=0.12, cycle_burnup=None, discharge_burnup=None,
        max_assembly_burnup=None, max_pin_burnup=40.0, eoc_ppm=None, delta_efpd=None,
        n_cycles=12.0, converged=True, converged_at_cap=False, tolerance_margin=0.2,
        restart_provenance="native:MAS_RST.APRQ_11_0652.86", valid=True, failure="",
        maps_key=None,
    )


def _seed_store(tmp_path: Path, records) -> Path:
    store_dir = tmp_path / "main_store"
    store_dir.mkdir()
    StoreWriter(store_dir).write_records(list(records))
    return store_dir


def _cfg(store_dir: Path, tmp_path: Path, **crit) -> LpoptConfig:
    deck = tmp_path / "lpopt.inp"
    deck.write_text("# fake\n", encoding="utf-8")
    crit_kwargs = dict(
        search_mode="lean", e_core_target=5.2, e_core_tol=0.05,
        cyclen_target=652.0, cyclen_tol=10.0,
        # M5 LP: F_r 1.5444 <= 1.55, cbc 1326 <= 1550, f_q 2.10 <= 2.41, ao 0.12 <= 0.30.
        f_r_limit=1.55, f_q_limit=2.41, cbc_limit=1550.0, asi_abs_limit=0.30,
        lean_deep_cells=4, lean_pool_per_cell=120, lean_top_k=6,
        lean_per_pair_cap=3, lean_hamming_min=4, screen_pool_per_cell=8,
        lean_store_elites_per_cell=8, post_verify_topk=0,
    )
    crit_kwargs.update(crit)
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(), data=DataConfig(),
        case=CaseConfig(mode="user_criteria", feed=121),
        fuel=FuelConfig(), extract=ExtractConfig(), produce=ProduceConfig(),
        search=SearchConfig(), acquisition=AcquisitionConfig(budget=20),
        model=ModelConfig(store_dir=str(store_dir), library_id="ga80"),
        criteria=CriteriaConfig(**crit_kwargs), source_path=deck,
    )


def _driver(tmp_path: Path, records, **crit) -> UserCriteriaDriver:
    store_dir = _seed_store(tmp_path, records)
    cfg = _cfg(store_dir, tmp_path, **crit)
    return UserCriteriaDriver(
        cfg, FakeModel(), lambda w, c: StubEvaluator(), dry_run=True,
        run_dir=tmp_path / "run", fuel_library=_fuel(_TIERS), progress=False, budget=20,
    )


def _k1k2_cell(driver: UserCriteriaDriver):
    universe = build_pair_universe(driver.fuel, "ga80", 5.2, 0.05)
    return next(c for c in universe if c.pair == "K1_K2")


# --------------------------------------------------------------------------- #
# 1) store elites are injected for the pair + ledger-deduped (never re-verified)
# --------------------------------------------------------------------------- #
def test_store_elites_injected_for_pair(tmp_path: Path) -> None:
    pat = _k1k2_pattern()
    rec = _feasible_record(pat)
    driver = _driver(tmp_path, [rec])

    seeds = driver._store_elite_seeds("K1_K2")
    assert seeds, "the converged, in-band K1_K2 store row must seed an elite parent"
    assert rec.record_id in {rid for rid, _ in seeds}
    # ledger dedup: the verified store row's record_id is registered so it is never
    # regenerated by build_pool nor re-verified by the wave.
    assert rec.record_id in driver.ledger_ids

    cell = _k1k2_cell(driver)
    ctx = driver._cell_context("K1_K2", cell_target_e_core(cell, driver.criteria))
    injected = driver._inject_store_elites(cell, ctx, seeds)
    assert injected, "the store elite must be injected as a prediction-only candidate"
    assert all(lc.verified for lc in injected)
    assert rec.record_id in {lc.candidate.record_id for lc in injected}
    # a pair with NO store row injects nothing.
    assert driver._store_elite_seeds("A1_N1") == []


# --------------------------------------------------------------------------- #
# 2) small-move mutation CHILDREN of the store elite are generated (verifiable)
# --------------------------------------------------------------------------- #
def test_store_elite_mutation_children_generated(tmp_path: Path) -> None:
    pat = _k1k2_pattern()
    rec = _feasible_record(pat)
    driver = _driver(tmp_path, [rec])

    seeds = driver._store_elite_seeds("K1_K2")
    cell = _k1k2_cell(driver)
    ctx, scored = driver._score_cell(
        cell, int(driver.criteria.screen_pool_per_cell), wave_index=0, elites=seeds
    )
    assert scored is not None
    elite_children = [
        c for c in scored.candidates
        if c.origin == "elite" and c.parent_record_id == rec.record_id
    ]
    assert elite_children, "build_pool must generate mutation children of the store elite"
    # the children are NEW candidates (distinct record_id from the verified elite).
    assert all(c.record_id != rec.record_id for c in elite_children)


# --------------------------------------------------------------------------- #
# 2b) NaN == missing == None (decision 2026-07-31) — the INVERTED twin
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("form", [None, float("nan")])
@pytest.mark.parametrize("axis", ["f_r", "cbc_max", "f_q", "ao_abs",
                                  "max_pin_burnup", "cyclen"])
def test_criteria_gates_reject_a_missing_value_in_both_forms(
    tmp_path: Path, axis: str, form
) -> None:
    """``_is_criteria_feasible`` REJECTS a missing gate value — and a parquet
    ``NaN`` is missing, exactly like an in-memory ``None``.

    Before the fix this predicate compared ``float(nan) > limit``, which is
    ``False``, so a NaN silently PASSED a hard user-set gate that a ``None``
    rejected — the inverted twin of the ``is_feasible`` pin-BU hole.
    """

    pat = _k1k2_pattern()
    rec = _feasible_record(pat)
    # pin_bu_limit is report-only by DEFAULT (an unset gate is skipped entirely),
    # so it must be SET for this axis to be a gate at all.
    driver = _driver(tmp_path, [rec], pin_bu_limit=80.0)

    row = rec.to_record()
    ok, _ = driver._is_criteria_feasible(row)
    assert ok is True, "the baseline row must be criteria-feasible"

    ok_missing, _ = driver._is_criteria_feasible({**row, axis: form})
    assert ok_missing is False, f"{axis}={form!r} must NOT pass a SET gate"


def test_criteria_feasibility_unchanged_for_finite_values(tmp_path: Path) -> None:
    """No behaviour change for MEASURED values — only "missing" moved."""

    pat = _k1k2_pattern()
    rec = _feasible_record(pat)
    driver = _driver(tmp_path, [rec], pin_bu_limit=80.0)
    row = rec.to_record()

    assert driver._is_criteria_feasible({**row, "max_pin_burnup": 79.0})[0] is True
    assert driver._is_criteria_feasible({**row, "max_pin_burnup": 81.0})[0] is False
    assert driver._is_criteria_feasible({**row, "f_r": 1.5499})[0] is True
    assert driver._is_criteria_feasible({**row, "f_r": 1.5501})[0] is False


# --------------------------------------------------------------------------- #
# 3) known-verified-LPs report table (unit) + wired into the driver report
# --------------------------------------------------------------------------- #
def test_known_verified_lps_table_md_unit() -> None:
    empty = known_verified_lps_table_md([])
    assert any("Known verified LPs in store" in line for line in empty)
    assert any("No converged store LP" in line for line in empty)

    rows = [
        {"case_pair": "K1_K2", "record_id": "abc123def456ghi789", "f_r": 1.5444,
         "f_q": 2.10, "cbc_max": 1326.0, "cyclen": 652.6, "e_core": 5.2, "n_cycles": 12.0,
         "disch_est": 49.4, "disch_in_band": False},
        {"case_pair": "K1_K2", "record_id": "zzz", "f_r": 1.60,
         "f_q": 2.20, "cbc_max": 1400.0, "cyclen": 650.0, "e_core": 5.19, "n_cycles": 12.0,
         "disch_est": 54.0, "disch_in_band": True},
    ]
    md = "\n".join(known_verified_lps_table_md(rows))
    assert "**2**" in md
    assert "K1_K2" in md
    assert "1.544" in md              # best-F_r row rendered (M5-style LP listed)
    assert "652.6" in md
    # discharge is a post-hoc estimate, annotated (never a hard gate that hides a
    # cyclen-feasible verified LP).
    assert "disch(est)" in md
    assert "est. oob" in md           # 49.4 out of the discharge band, still listed
    assert "in-band" in md


def test_driver_known_verified_store_lps_and_report(tmp_path: Path) -> None:
    pat = _k1k2_pattern()
    rec = _feasible_record(pat)
    driver = _driver(tmp_path, [rec])

    matched = driver._known_verified_store_lps()
    assert [r["record_id"] for r in matched] == [rec.record_id]

    result = driver.run()
    report = (driver.run_dir / "report.md").read_text(encoding="utf-8")
    assert "Known verified LPs in store matching the criteria" in report
    assert rec.record_id[:16] in report
    # K1_K2's screen value reflects the injected elite (finite, not screened out).
    assert "K1_K2" in driver.cell_runs
    assert np.isfinite(driver.cell_runs["K1_K2"].stat.screen_value)

    # the verified store elite itself is NEVER re-verified (predictions only).
    labels_path = driver.run_dir / "labels.jsonl"
    if labels_path.exists() and labels_path.read_text().strip():
        labels = [json.loads(l) for l in labels_path.read_text().splitlines()]
        assert rec.record_id not in {l["record_id"] for l in labels}
    assert result.status in ("complete", "no_feasible")
