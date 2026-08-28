"""FREE-SEARCH ``user_criteria`` mode (plan sec. 6.2 / 12.5).

Covers the mode's five acceptance surfaces without the real CNN:

* pair-universe builder correctness at e_core 5.2 +/- 0.05 (includes K1_K2 and
  J1_N1-style cross pairs, excludes pure-A(5.0) mono and C/D-high pairs; mono
  handling);
* the split-as-inner-variable e_core-band candidate screen;
* the outer racing allocation (activate / race-eliminate / softmax) on synthetic
  cell predictions;
* the user_criteria exploit ranking (= score_user_criteria total) + discharge
  band via ``predict_extra``;
* an end-to-end dry-run campaign that completes and writes its report.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.config import (
    AcquisitionConfig, CaseConfig, ConfigError, CriteriaConfig, DataConfig,
    ExtractConfig, FlowConfig, FuelConfig, LpoptConfig, MasterConfig, ModelConfig,
    ProduceConfig, RemoteConfig, SearchConfig, VerifyConfig,
)
from lpopt.data.fuel_types import FuelLibrary
from lpopt.data.schema import unpack_pattern
from lpopt.search import acquisition as acq
from lpopt.search.campaign import UserCriteriaDriver, run_campaign
from lpopt.search.construct import (
    CaseContext, build_pair_universe, build_pool, e_core_in_band,
    predicted_e_core, screen_e_core_band,
)
from lpopt.search.stub import StubEvaluator
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _u(d: str, salt: str) -> float:
    return int(hashlib.sha256(f"{d}:{salt}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class FakeModel:
    """Deterministic torch-free surrogate double (7-column)."""

    def _row(self, p):
        d = p.canonical()
        f_r = 1.45 + 0.30 * _u(d, "fr")
        return [f_r, 1400 + 180 * _u(d, "cbc"), f_r * 1.4,
                600 + 55 * _u(d, "cy"), 0.10 + 0.20 * _u(d, "ao"),
                np.nan, 45.0 + 10 * _u(d, "pb")]

    def predict(self, patterns, case, cell=0.0):
        patterns = list(patterns)
        if not patterns:
            e = np.zeros((0, 7))
            return SurrogatePrediction(e, e.copy(), e.copy())
        m = np.array([self._row(p) for p in patterns], float)
        s = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, 1.0], (len(patterns), 1))
        return SurrogatePrediction(m, s.copy(), s.copy())

    def predict_convergence(self, patterns, case, cell=0.0):
        return np.ones(len(list(patterns)), float)

    def position_values(self, *a, **k):
        return None


class _Extra:
    def __init__(self, names, mean, std):
        self.names = names
        self.mean = mean
        self.epistemic_std = std
        self.calibrated_std = std


class FakeModelDischarge(FakeModel):
    """FakeModel + a ``discharge_burnup`` extra target keyed on the pattern digest."""

    def predict_extra(self, patterns, case, cell=0.0):
        patterns = list(patterns)
        mean = np.array([[30.0 + 30.0 * _u(p.canonical(), "dis")] for p in patterns], float)
        std = np.full_like(mean, 0.5)
        return _Extra(("discharge_burnup",), mean, std)


_FUEL_COLS = [
    "library_id", "type_id", "u_avg_enrichment", "enr_main", "enr_zone", "u_mass_g",
    "n_gd", "gd_wt", "gd_u_enr", "axial_zone", "source_flags", "feature_poor",
    "kinf0", "kinf10", "kinf20", "kinf30", "bu_k1",
]


def _fuel(tiers: dict[str, float], library_id: str = "ga80") -> FuelLibrary:
    rows = []
    for t, e in tiers.items():
        row = {c: np.nan for c in _FUEL_COLS}
        row.update(library_id=library_id, type_id=t, u_avg_enrichment=e, enr_main=e,
                   axial_zone=None, source_flags=None, feature_poor=False)
        rows.append(row)
    return FuelLibrary(pd.DataFrame(rows, columns=_FUEL_COLS))


# ga80-like enrichment tiers spanning the low->high span (A=5.0 ... D=6.5).
_TIERS = {"A1": 5.0, "J1": 5.1, "K1": 5.2, "K2": 5.2, "L1": 5.3,
          "N1": 5.4, "C1": 6.0, "D1": 6.5}


# --------------------------------------------------------------------------- #
# 1. universe builder
# --------------------------------------------------------------------------- #
def test_universe_at_5p2_includes_cross_excludes_high_and_pureA():
    fuel = _fuel(_TIERS)
    cells = build_pair_universe(fuel, "ga80", 5.2, 0.05)
    by = {c.pair: c for c in cells}

    def inc(a, b):
        return by[f"{a}_{b}"].included

    # K1_K2 (both 5.2) and the J1_N1-style cross pair are reachable.
    assert inc("K1", "K2") is True
    assert inc("J1", "N1") is True
    assert inc("A1", "N1") is True                # low+high bracketing 5.2
    # mono handling: K1_K1 (5.2) in band, pure-A A1_A1 (5.0) out of band.
    assert inc("K1", "K1") is True
    assert by["K1_K1"].mono is True
    assert inc("A1", "A1") is False
    assert by["A1_A1"].mono is True
    # C/D-high pairs are unreachable (well above the band).
    assert inc("C1", "D1") is False
    assert inc("A1", "D1") is False
    # every excluded cell records a reason; every included one does not.
    for c in cells:
        assert (c.reason == "") is c.included


def test_universe_reach_interval_is_split_extremes_and_mono_is_point():
    fuel = _fuel(_TIERS)
    cells = {c.pair: c for c in build_pair_universe(fuel, "ga80", 5.2, 0.05)}
    jn = cells["J1_N1"]                            # 5.1 & 5.4, split in [0.2, 0.8]
    assert jn.e_lo == pytest.approx(5.16, abs=1e-6)
    assert jn.e_hi == pytest.approx(5.34, abs=1e-6)
    assert cells["K1_K1"].e_lo == pytest.approx(5.2) == cells["K1_K1"].e_hi


def test_universe_allow_mono_false_drops_mono_cells():
    fuel = _fuel(_TIERS)
    cells = build_pair_universe(fuel, "ga80", 5.2, 0.05, allow_mono=False)
    assert all(not c.mono for c in cells)
    assert "K1_K1" not in {c.pair for c in cells}


# --------------------------------------------------------------------------- #
# 2. e_core band screen (split as inner variable)
# --------------------------------------------------------------------------- #
def test_e_core_in_band_edges():
    assert e_core_in_band(5.2, 5.2, 0.05)
    assert e_core_in_band(5.15, 5.2, 0.05)
    assert e_core_in_band(5.25, 5.2, 0.05)
    assert not e_core_in_band(5.10, 5.2, 0.05)
    assert not e_core_in_band(None, 5.2, 0.05)


def test_screen_e_core_band_filters_pattern_composition():
    fuel = _fuel(_TIERS)
    rng = __import__("random").Random(0)
    # mono K1: every pattern is e_core 5.2 -> all in band.
    ctx = CaseContext(pair="K1_K1", feed=121, library_id="ga80", e_core=5.2)
    pats = [c.pattern for c in build_pool(ctx, None, [], set(), rng, _mini_cfg(),
                                          wave_index=0, size=8)]
    mask = screen_e_core_band(pats, fuel, "ga80", 5.2, 0.05)
    assert mask.all()
    assert all(abs(predicted_e_core(p, fuel, "ga80") - 5.2) < 1e-9 for p in pats)

    # A pure-D mono cell (6.5) is entirely out of the 5.2 band.
    fuel_d = _fuel({"D1": 6.5})
    ctxd = CaseContext(pair="D1_D1", feed=121, library_id="ga80", e_core=6.5)
    patsd = [c.pattern for c in build_pool(ctxd, None, [], set(), rng, _mini_cfg(),
                                           wave_index=0, size=6)]
    assert not screen_e_core_band(patsd, fuel_d, "ga80", 5.2, 0.05).any()


def test_screen_permissive_without_fuel():
    fuel = _fuel(_TIERS)
    rng = __import__("random").Random(1)
    ctx = CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)
    pats = [c.pattern for c in build_pool(ctx, None, [], set(), rng, _mini_cfg(),
                                          wave_index=0, size=4)]
    assert screen_e_core_band(pats, None, "ga80", 5.2, 0.05).all()


# --------------------------------------------------------------------------- #
# 3. outer racing allocation (synthetic)
# --------------------------------------------------------------------------- #
def test_outer_activate_top_k_by_screen_value():
    cells = [acq.OuterCellStat(f"C{i}", screen_value=v)
             for i, v in enumerate([10.0, 9.0, 1.0, -2.0, -5.0])]
    activated = acq.outer_activate(cells, 3)
    assert activated == ["C0", "C1", "C2"]
    assert [c.active for c in cells] == [True, True, True, False, False]


def test_outer_race_eliminates_dominated_keeps_min():
    cells = [acq.OuterCellStat(f"C{i}", screen_value=v)
             for i, v in enumerate([10.0, 9.5, 5.0, 1.0])]
    acq.outer_activate(cells, 4)
    cells[0].samples = [10.1, 10.0]; cells[0].n_verify = 2
    cells[1].samples = [9.4, 9.6]; cells[1].n_verify = 2
    cells[2].samples = [5.1, 4.9]; cells[2].n_verify = 2
    cells[3].samples = [1.1, 0.9]; cells[3].n_verify = 2
    elim = acq.outer_race(cells, z=1.0, prior_sigma=1.0, min_keep=3)
    assert "C3" in elim                            # clearly worst -> eliminated
    assert sum(c.active and not c.eliminated for c in cells) == 3   # min_keep held


def test_outer_softmax_alloc_floor_on_best_and_sums():
    cells = [acq.OuterCellStat(f"C{i}", screen_value=v, active=True)
             for i, v in enumerate([10.0, 6.0, 2.0])]
    for c, s in zip(cells, ([10.0], [6.0], [2.0])):
        c.samples = s
    alloc = acq.outer_softmax_alloc(cells, slots=12, temperature=1.0, exploit_floor=5)
    assert sum(alloc.values()) == 12
    assert alloc["C0"] >= 5                         # exploit floor on the best cell
    assert alloc["C0"] == max(alloc.values())


# --------------------------------------------------------------------------- #
# 4. user_criteria exploit ranking + discharge band
# --------------------------------------------------------------------------- #
def _score_cell(model, spec, fuel, seed=0, size=24, pair="J1_N1"):
    rng = __import__("random").Random(seed)
    ctx = CaseContext(pair=pair, feed=121, library_id="ga80", e_core=5.2)
    pool = build_pool(ctx, model, [], set(), rng, _mini_cfg(), wave_index=0, size=size)
    scored = acq.score_pool_user_criteria(
        model, ctx, pool, spec, fuel=fuel, library_id="ga80",
        e_core_target=5.2, e_core_tol=0.05,
    )
    return pool, scored


def test_exploit_rank_is_criteria_total_and_region_is_band():
    fuel = _fuel(_TIERS)
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=100.0)  # loose band
    pool, scored = _score_cell(FakeModel(), spec, fuel)
    assert scored.rank.shape[0] == len(pool)
    # rank == exploit == score_user_criteria.total on in-region rows; -inf outside.
    assert np.array_equal(scored.rank, scored.exploit)
    out = ~scored.in_region
    assert np.all(np.isneginf(scored.exploit[out])) if out.any() else True
    # in-region rows carry the pattern's in-band e_core.
    for i in np.flatnonzero(scored.in_region):
        assert e_core_in_band(predicted_e_core(pool[i].pattern, fuel, "ga80"), 5.2, 0.05)


def test_pfeas_gate_uses_set_criteria_limits():
    fuel = _fuel(_TIERS)
    # a tight F_r limit below every prediction (min 1.45) -> p_feas collapses.
    tight = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=100.0, f_r_limit=1.0)
    _, s_tight = _score_cell(FakeModel(), tight, fuel)
    loose = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=100.0, f_r_limit=3.0)
    _, s_loose = _score_cell(FakeModel(), loose, fuel)
    reg = s_tight.in_region
    assert reg.any()
    assert s_tight.p_feas[reg].max() < s_loose.p_feas[reg].max()


def test_discharge_band_penalises_out_of_band_via_predict_extra():
    fuel = _fuel(_TIERS)
    # discharge target far outside the fake's [30, 60] range -> every candidate
    # out of the discharge band -> criteria totals dominated by the band penalty.
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=100.0,
                            target_discharge_burnup=200.0, discharge_tolerance=1.0)
    _, scored = _score_cell(FakeModelDischarge(), spec, fuel)
    reg = scored.in_region
    assert reg.any()
    # band-dominated totals are hugely negative (tiered penalty), never ~ -F_r.
    assert scored.exploit[reg].max() < -1.0e6


# --------------------------------------------------------------------------- #
# 5. dry-run campaign
# --------------------------------------------------------------------------- #
def _mini_cfg(
    budget: int = 20, tmp: Path | None = None, *,
    search_mode: str = "lean", **crit_overrides,
) -> LpoptConfig:
    deck = (tmp / "lpopt.inp") if tmp else None
    if deck:
        deck.write_text("# fake\n", encoding="utf-8")
    crit_kwargs = dict(
        search_mode=search_mode,
        e_core_target=5.2, e_core_tol=0.05, cyclen_target=620.0, cyclen_tol=40.0,
        outer_max_cells=4, outer_screen_budget=8, outer_target_cells=2,
        outer_exploit_floor=3, outer_verify_per_wave=2, screen_pool_per_cell=6,
        post_verify_topk=0,
        # keep the lean deepen light for the StubEvaluator tests.
        lean_deep_cells=4, lean_pool_per_cell=120, lean_top_k=6,
        lean_per_pair_cap=3, lean_hamming_min=4,
    )
    crit_kwargs.update(crit_overrides)
    crit = CriteriaConfig(**crit_kwargs)
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(), data=DataConfig(),
        case=CaseConfig(mode="user_criteria", feed=121),
        fuel=FuelConfig(), extract=ExtractConfig(), produce=ProduceConfig(),
        search=SearchConfig(), acquisition=AcquisitionConfig(budget=budget),
        model=ModelConfig(), criteria=crit, source_path=deck,
    )


def test_dry_run_user_criteria_campaign_completes(tmp_path):
    cfg = _mini_cfg(budget=20, tmp=tmp_path)
    run_dir = tmp_path / "run"
    stub = StubEvaluator()
    result = run_campaign(
        cfg, FakeModel(), lambda w, c: stub, dry_run=True, run_dir=run_dir,
        fuel_library=_fuel(_TIERS), progress=False, budget=20,
    )
    assert result.status in ("complete", "no_feasible")
    assert result.budget_spent <= 20
    assert result.budget_spent > 0
    labels = [json.loads(l) for l in (run_dir / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == result.budget_spent
    assert len({l["record_id"] for l in labels}) == len(labels)   # no double-spend
    assert all("criteria_total" in l and "e_core" in l for l in labels)
    # every verified board's fresh feed is in the e_core band (screen enforced).
    assert all(abs(l["e_core"] - 5.2) <= 0.05 + 1e-9 for l in labels if l["e_core"] is not None)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Pair universe" in report
    assert "Racing timeline" in report
    assert "Best LP (all 7 constraint axes" in report
    assert "calls-to-first-feasible" in report


def test_dry_run_budget_scales_and_status_json(tmp_path):
    cfg = _mini_cfg(budget=12, tmp=tmp_path)
    run_dir = tmp_path / "run"
    stub = StubEvaluator()
    result = run_campaign(
        cfg, FakeModel(), lambda w, c: stub, dry_run=True, run_dir=run_dir,
        fuel_library=_fuel(_TIERS), progress=False, budget=12,
    )
    assert result.budget_spent <= 12
    status = json.loads((run_dir / "status.json").read_text())
    assert status["mode"] == "user_criteria"
    assert status["universe"] >= 1


# --------------------------------------------------------------------------- #
# 6. config validation
# --------------------------------------------------------------------------- #
def test_case_user_criteria_needs_no_pair():
    CaseConfig(mode="user_criteria", feed=121).validate()   # no pair -> ok


def test_criteria_config_validate_rejects_bad_split():
    with pytest.raises(ConfigError):
        CriteriaConfig(split_range=[0.8, 0.2]).validate()
    with pytest.raises(ConfigError):
        CriteriaConfig(e_core_tol=-0.1).validate()


# --------------------------------------------------------------------------- #
# 7. lean (predict-then-verify) mode
# --------------------------------------------------------------------------- #
def test_config_default_search_mode_is_lean():
    assert CriteriaConfig().search_mode == "lean"


def test_criteria_config_rejects_bad_search_mode():
    with pytest.raises(ConfigError):
        CriteriaConfig(search_mode="turbo").validate()


def test_lean_single_verify_wave_topk_diversity_and_cap(tmp_path):
    """Lean flow: ONE batched verification wave of the diverse, per-pair-capped
    global top-K predicted candidates (the core predict-then-verify promise)."""

    cfg = _mini_cfg(budget=30, tmp=tmp_path, search_mode="lean",
                    lean_top_k=6, lean_per_pair_cap=3, lean_hamming_min=4,
                    lean_deep_cells=4)
    run_dir = tmp_path / "run"
    stub = StubEvaluator()
    driver = UserCriteriaDriver(
        cfg, FakeModel(), lambda w, c: stub, dry_run=True, run_dir=run_dir,
        fuel_library=_fuel(_TIERS), progress=False, budget=30,
    )
    calls = {"n": 0, "sizes": []}
    orig = driver.verifier.evaluate_wave

    def counting(entries):
        entries = list(entries)
        calls["n"] += 1
        calls["sizes"].append(len(entries))
        return orig(entries)

    driver.verifier.evaluate_wave = counting
    result = driver.run()

    assert result.status in ("complete", "no_feasible")
    assert calls["n"] == 1                              # a SINGLE verification wave
    labels = [json.loads(l) for l in (run_dir / "labels.jsonl").read_text().splitlines()]
    assert all(l["phase"] == "lean_r1" for l in labels)
    assert 0 < len(labels) <= 6                         # top-K respected
    assert len(labels) == calls["sizes"][0]            # all K in the one wave
    assert result.budget_spent == len(labels)

    per_pair = Counter(l["pair"] for l in labels)
    assert max(per_pair.values()) <= 3                  # per-pair cap held

    pats = [unpack_pattern(l["record"]["pattern"]) for l in labels]
    for i in range(len(pats)):
        for j in range(i + 1, len(pats)):
            assert pats[i].hamming(pats[j]) >= 4        # pairwise diversity floor

    # honest predicted-vs-actual precision table + wall-time in the report.
    assert len(driver.lean_rows) == len(labels)
    assert all("pred" in r and "actual" in r for r in driver.lean_rows)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Predicted vs actual (lean top-K)" in report
    assert "Wall-time breakdown" in report


def test_lean_every_verified_board_is_in_e_core_band(tmp_path):
    cfg = _mini_cfg(budget=20, tmp=tmp_path, search_mode="lean")
    run_dir = tmp_path / "run"
    stub = StubEvaluator()
    result = run_campaign(
        cfg, FakeModel(), lambda w, c: stub, dry_run=True, run_dir=run_dir,
        fuel_library=_fuel(_TIERS), progress=False, budget=20,
    )
    labels = [json.loads(l) for l in (run_dir / "labels.jsonl").read_text().splitlines()]
    assert labels
    assert len({l["record_id"] for l in labels}) == len(labels)   # no double-spend
    assert all(abs(l["e_core"] - 5.2) <= 0.05 + 1e-9
               for l in labels if l["e_core"] is not None)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["search_mode"] == "lean"
    assert "screen_seconds" in status and "verify_seconds" in status


def test_active_path_batches_multi_cell_into_one_wave(tmp_path):
    """Active (racing) path stays selectable and batches every active cell's
    verify slots into ONE full-width wave (never per-cell 2-entry mini-waves)."""

    cfg = _mini_cfg(budget=12, tmp=tmp_path, search_mode="active")
    run_dir = tmp_path / "run"
    stub = StubEvaluator()
    driver = UserCriteriaDriver(
        cfg, FakeModel(), lambda w, c: stub, dry_run=True, run_dir=run_dir,
        fuel_library=_fuel(_TIERS), progress=False, budget=12,
    )
    sizes: list[int] = []
    orig = driver.verifier.evaluate_wave

    def counting(entries):
        entries = list(entries)
        sizes.append(len(entries))
        return orig(entries)

    driver.verifier.evaluate_wave = counting
    result = driver.run()

    assert result.status in ("complete", "no_feasible")
    assert result.budget_spent <= 12
    phases = {ev["phase"] for ev in result.wave_reports}
    assert "screen" in phases and "exploit" in phases       # racing timeline intact
    # a racing wave batched > outer_verify_per_wave entries across cells.
    assert sizes and max(sizes) > cfg.criteria.outer_verify_per_wave


def test_lean_second_round_fires_only_when_no_feasible(tmp_path):
    """The opt-in second round runs a single extra batched wave iff no verified
    candidate met the criteria bands (default OFF keeps the flow truly lean)."""

    # F_r limit below every stub output -> nothing meets the bands.
    cfg = _mini_cfg(budget=30, tmp=tmp_path, search_mode="lean",
                    lean_top_k=4, lean_second_round=True, f_r_limit=1.0)
    run_dir = tmp_path / "run"
    stub = StubEvaluator()
    driver = UserCriteriaDriver(
        cfg, FakeModel(), lambda w, c: stub, dry_run=True, run_dir=run_dir,
        fuel_library=_fuel(_TIERS), progress=False, budget=30,
    )
    calls = {"n": 0}
    orig = driver.verifier.evaluate_wave

    def counting(entries):
        calls["n"] += 1
        return orig(list(entries))

    driver.verifier.evaluate_wave = counting
    driver.run()

    assert driver.best is None                      # tight F_r limit -> no feasible
    assert calls["n"] == 2                           # first round + one second round
    assert {r["round"] for r in driver.lean_rows} == {"r1", "r2"}
