"""Guided-search campaign end-to-end with a StubEvaluator + a fast FakeModel.

Covers the plan sec. 4.6 acceptance items that do not need the real CNN:
full-budget 12-waves + reserve, resume with no budget double-spend, and the
budget=0 proposals-only mode — all against the deterministic StubEvaluator.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from lpopt.config import (
    AcquisitionConfig, CaseConfig, DataConfig, ExtractConfig, FlowConfig, FuelConfig,
    LpoptConfig, MasterConfig, ModelConfig, ProduceConfig, RemoteConfig, SearchConfig,
    VerifyConfig,
)
from lpopt.search.campaign import run_campaign
from lpopt.search.stub import StubEvaluator
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction

_STORE = Path(__file__).resolve().parents[1] / "data" / "store"


def _u(digest: str, salt: str) -> float:
    raw = hashlib.sha256(f"{digest}:{salt}".encode()).hexdigest()
    return int(raw[:8], 16) / 0xFFFFFFFF


class FakeModel:
    """Deterministic, torch-free PositionValueModel double (per-pattern FOM)."""

    def _row(self, pattern):
        d = pattern.canonical()
        f_r = 1.45 + 0.30 * _u(d, "fr")
        return [f_r, 1400 + 180 * _u(d, "cbc"), f_r * 1.4,
                600 + 55 * _u(d, "cy"), 0.10 + 0.20 * _u(d, "ao"), np.nan, np.nan]

    def predict(self, patterns, case, cell=0.0):
        patterns = list(patterns)
        if not patterns:
            empty = np.zeros((0, 7))
            return SurrogatePrediction(empty, empty.copy(), empty.copy())
        mean = np.array([self._row(p) for p in patterns], dtype=float)
        std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (len(patterns), 1))
        return SurrogatePrediction(mean, std.copy(), std.copy())

    def predict_convergence(self, patterns, case, cell=0.0):
        return np.ones(len(patterns), dtype=float)

    def position_values(self, pattern, case, cell=0.0):
        return None

    def finetune(self, new, replay, epochs=3, seed=0):
        return {"refit": False, "n_new": len(list(new))}

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "fake.json").write_text("{}", encoding="utf-8")
        return p


def _cfg(tmp_path: Path, budget: int) -> LpoptConfig:
    deck = tmp_path / "lpopt.inp"
    deck.write_text("# fake deck\n", encoding="utf-8")
    acq = AcquisitionConfig(budget=budget, gate_skill_halt=-2.0)
    model = ModelConfig(store_dir=str(_STORE), model_dir=str(_STORE))
    cfg = LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(), data=DataConfig(), case=CaseConfig(pair="K1_K2", feed=121),
        fuel=FuelConfig(), extract=ExtractConfig(), produce=ProduceConfig(),
        search=SearchConfig(), acquisition=acq, model=model, source_path=deck,
    )
    return cfg


def _factory():
    stub = StubEvaluator()
    return lambda worker_id, cpu_core: stub


def _labels(run_dir: Path) -> list[dict]:
    path = run_dir / "labels.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_full_budget_twelve_waves_plus_reserve(tmp_path):
    cfg = _cfg(tmp_path, budget=100)
    run_dir = tmp_path / "run"
    result = run_campaign(
        cfg, FakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        backend_factory=lambda ckpt: FakeModel(), early_stop=False, progress=False,
    )
    assert result.status == "complete"
    assert result.waves == 13                      # 12 waves x 8 + 1 reserve x 4
    assert result.budget_spent == 100
    labels = _labels(run_dir)
    assert len(labels) == 100
    assert len({l["record_id"] for l in labels}) == 100    # no duplicate evaluations
    # last wave is the exploit-only reserve wave.
    last = json.loads((run_dir / "waves" / "wave_12" / "selection.json").read_text("utf-8"))
    assert all(s["slot"] == "exploit" for s in last["selection"])
    # report + figures generated, GA-600 overlay present.
    assert (run_dir / "report.md").exists()
    assert (run_dir / "figures" / "budget_curve.png").exists()
    assert (run_dir / "figures" / "parity.png").exists()


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_resume_no_budget_double_spend(tmp_path):
    run_dir = tmp_path / "run"
    # first invocation: stop after 3 waves (simulates a kill), state persisted.
    cfg = _cfg(tmp_path, budget=48)
    r1 = run_campaign(
        cfg, FakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        backend_factory=lambda ckpt: FakeModel(), early_stop=False, max_waves=3,
        progress=False,
    )
    assert r1.status == "paused"
    assert r1.budget_spent == 24
    first_ids = {l["record_id"] for l in _labels(run_dir)}
    assert len(first_ids) == 24

    # resume: continues from wave 3 to full budget, no double-spend.
    cfg2 = _cfg(tmp_path, budget=48)
    r2 = run_campaign(
        cfg2, FakeModel(), _factory(), dry_run=True, run_dir=run_dir, resume=True,
        backend_factory=lambda ckpt: FakeModel(), early_stop=False, progress=False,
    )
    assert r2.status == "complete"
    assert r2.budget_spent == 48
    labels = _labels(run_dir)
    assert len(labels) == 48
    assert len({l["record_id"] for l in labels}) == 48     # every eval unique
    assert first_ids.issubset({l["record_id"] for l in labels})


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_budget_zero_proposals_only(tmp_path):
    cfg = _cfg(tmp_path, budget=0)
    run_dir = tmp_path / "run"
    result = run_campaign(
        cfg, FakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        budget=0, progress=False,
    )
    assert result.status == "proposals_only"
    assert len(result.proposals) == 16
    assert (run_dir / "proposals.json").exists()
    assert not (run_dir / "labels.jsonl").exists()          # no evaluation happened


def test_feed_range_mode_raises_not_implemented(tmp_path):
    cfg = _cfg(tmp_path, budget=8)
    cfg.case.mode = "feed_range"
    with pytest.raises(NotImplementedError):
        run_campaign(cfg, FakeModel(), _factory(), dry_run=True, run_dir=tmp_path / "r")


# --------------------------------------------------------------------------- #
# max_cycle_min_fr objective mode (additive; user directive 2026-07-21)
# --------------------------------------------------------------------------- #
def _run(tmp_path: Path, budget: int, objective: str | None):
    """Dry-run a campaign; ``objective`` None keeps the deck default (target_cycle)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(tmp_path, budget)
    if objective is not None:
        cfg.acquisition.objective = objective
    run_dir = tmp_path / "run"
    result = run_campaign(
        cfg, FakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        backend_factory=lambda ckpt: FakeModel(), early_stop=False, progress=False,
    )
    return result, _labels(run_dir)


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_objective_flag_off_is_byte_identical(tmp_path):
    # The additive knob's default value ("target_cycle") must be byte-identical to
    # setting it EXPLICITLY: same labels, same slots, same order, same status.
    _, lbl_default = _run(tmp_path / "a", 40, None)
    _, lbl_explicit = _run(tmp_path / "b", 40, "target_cycle")
    key = lambda rows: [(r["wave"], r["slot"], r["record_id"], r["status"]) for r in rows]
    assert key(lbl_default) == key(lbl_explicit)


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_max_cycle_min_fr_runs_and_tracks_objective(tmp_path):
    result, labels = _run(tmp_path / "mc", 40, "max_cycle_min_fr")
    assert result.status in ("complete", "stalled")
    assert len(labels) == result.budget_spent
    if result.best is not None:
        b = result.best
        # best objective is the verified scalar cyclen - λ·F_r (λ default 100).
        assert b["objective"] == pytest.approx(float(b["cyclen"]) - 100.0 * float(b["f_r"]))
        assert b["distance"] is None                      # no target window in this mode


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_max_cycle_feasibility_drops_the_fr_gate(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    # a converged row that PASSES F_q/CBC/|AO| but has F_r well above 1.55.
    row = {"converged": True, "f_r": 1.90, "cbc_max": 1500.0, "f_q": 2.30,
           "ao_abs": 0.20, "cyclen": 730.0}

    (tmp_path / "t").mkdir(parents=True, exist_ok=True)
    cfg_t = _cfg(tmp_path / "t", 8)
    drv_t = CampaignDriver(cfg_t, FakeModel(), _factory(), dry_run=True,
                           run_dir=tmp_path / "t" / "run", progress=False)
    assert drv_t.objective == "target_cycle"
    assert drv_t._is_feasible(row) is False               # F_r 1.90 > 1.55 gate

    (tmp_path / "m").mkdir(parents=True, exist_ok=True)
    cfg_m = _cfg(tmp_path / "m", 8)
    cfg_m.acquisition.objective = "max_cycle_min_fr"
    drv_m = CampaignDriver(cfg_m, FakeModel(), _factory(), dry_run=True,
                           run_dir=tmp_path / "m" / "run", progress=False)
    assert drv_m.objective == "max_cycle_min_fr"
    assert drv_m._is_feasible(row) is True                # F_r ungated -> feasible
    assert drv_m._campaign_objective(row) == pytest.approx(730.0 - 100.0 * 1.90)
    assert drv_m.max_cycle_spec is not None and drv_m.max_cycle_spec.lam == 100.0


# --------------------------------------------------------------------------- #
# min_fr_max_cycle objective mode (F_r primary + hard gate; user 2026-07-22)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_min_fr_runs_and_reports_best_overall(tmp_path):
    result, labels = _run(tmp_path / "mf", 40, "min_fr_max_cycle")
    assert result.status in ("complete", "stalled")
    assert len(labels) == result.budget_spent
    # best_overall (honesty channel) is populated whenever anything converged.
    assert result.best_overall is not None
    ov = result.best_overall
    assert ov["f_r"] is not None and ov["f_r_margin_to_limit"] is not None
    # status.json records best_feasible (may be null) + best_overall separately.
    status = json.loads((tmp_path / "mf" / "run" / "status.json").read_text("utf-8"))
    assert "best_overall" in status and "best_feasible" in status
    assert status["best_feasible"] == status["best"]     # 'best' == feasible best
    if result.best is None:
        assert status["best_feasible"] is None            # honest no-feasible path


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_min_fr_bookkeeping_and_honesty(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    (tmp_path / "d").mkdir(parents=True, exist_ok=True)
    cfg = _cfg(tmp_path / "d", 8)
    cfg.acquisition.objective = "min_fr_max_cycle"
    drv = CampaignDriver(cfg, FakeModel(), _factory(), dry_run=True,
                         run_dir=tmp_path / "d" / "run", progress=False)
    assert drv.objective == "min_fr_max_cycle" and drv.min_fr_spec.lam_fr == 1000.0

    # F_r 1.62 > 1.55: INFEASIBLE even though F_q/CBC/|AO| pass (F_r rejoins the gate).
    over = {"converged": True, "f_r": 1.62, "cbc_max": 1500.0, "f_q": 2.30,
            "ao_abs": 0.20, "cyclen": 745.0, "record_id": "r_over"}
    assert drv._is_feasible(over) is False
    drv._maybe_update_overall(over)
    assert drv.best is None                               # violator never promoted
    assert drv.best_overall is not None
    assert drv.best_overall["f_r_margin_to_limit"] == pytest.approx(1.55 - 1.62)
    assert drv.best_overall["feasible"] is False

    # a genuinely feasible row (F_r 1.53) IS feasible and updates best.
    good = {"converged": True, "f_r": 1.53, "cbc_max": 1490.0, "f_q": 2.10,
            "ao_abs": 0.10, "cyclen": 690.0, "record_id": "r_good"}
    assert drv._is_feasible(good) is True
    # lexicographic objective: lower F_r wins regardless of cyclen.
    assert drv._campaign_objective(good) > drv._campaign_objective(over)
    hi_cy = dict(over, cyclen=900.0)                      # same F_r, +155 EFPD
    assert drv._campaign_objective(good) > drv._campaign_objective(hi_cy)  # F_r dominates


# --------------------------------------------------------------------------- #
# fr_boundary objective mode (F_r=1.55 boundary training campaign; user 2026-07-22)
# --------------------------------------------------------------------------- #
def _frb_driver(tmp_path: Path, *, store_dir: Path | None = None, live: bool = False):
    from lpopt.search.campaign import CampaignDriver

    sub = tmp_path
    sub.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(sub, 8)
    cfg.acquisition.objective = "fr_boundary"
    if store_dir is not None:
        cfg.model.store_dir = str(store_dir)
    factory = None if live else _factory()
    return CampaignDriver(
        cfg, FakeModel(), factory, dry_run=True,
        run_dir=sub / "run", progress=False,
    )


def test_fr_boundary_spec_from_knobs(tmp_path):
    drv = _frb_driver(tmp_path / "spec")
    assert drv.objective == "fr_boundary"
    spec = drv.fr_boundary_spec
    assert spec is not None
    assert spec.cbc_limit == pytest.approx(drv.acq.cbc_limit)
    assert spec.f_q_limit == pytest.approx(drv.acq.f_q_limit)
    assert spec.ao_abs_limit == pytest.approx(drv.acq.ao_abs_limit)
    assert spec.pin_bu_limit == pytest.approx(80.0)
    assert spec.band_lo == pytest.approx(1.45) and spec.band_hi == pytest.approx(1.70)
    # NO f_r_limit / cyclen / lam fields (cyclen must not leak into ordering).
    assert not hasattr(spec, "f_r_limit")
    assert not hasattr(spec, "cyclen_lo") and not hasattr(spec, "lam_fr")


def test_fr_boundary_feasibility_no_fr_no_cyclen_gate(tmp_path):
    drv = _frb_driver(tmp_path / "feas")
    # F_r 1.90 far above 1.55 is STILL feasible (F_r is a pure objective, ungated).
    row = {"converged": True, "f_r": 1.90, "cbc_max": 1500.0, "f_q": 2.30,
           "ao_abs": 0.20, "cyclen": 730.0, "max_pin_burnup": 68.0}
    assert drv._is_feasible(row) is True
    # cyclen far outside any band is irrelevant (recorded, never gated).
    assert drv._is_feasible(dict(row, cyclen=100.0)) is True
    # a converged row with NO cyclen at all is still feasible.
    no_cy = {"converged": True, "f_r": 1.60, "cbc_max": 1500.0, "f_q": 2.30,
             "ao_abs": 0.20, "max_pin_burnup": 68.0}
    assert drv._is_feasible(no_cy) is True
    # CBC / F_q / |AO| DO gate.
    assert drv._is_feasible(dict(row, cbc_max=1600.0)) is False
    assert drv._is_feasible(dict(row, f_q=2.50)) is False
    assert drv._is_feasible(dict(row, ao_abs=0.40)) is False


def test_fr_boundary_pin_bu_is_none_tolerant(tmp_path):
    drv = _frb_driver(tmp_path / "pin")
    base = {"converged": True, "f_r": 1.60, "cbc_max": 1500.0, "f_q": 2.30,
            "ao_abs": 0.20, "cyclen": 620.0}
    # missing max_pin_burnup PASSES (None-tolerant — MASTER adjudicates).
    assert drv._is_feasible(base) is True
    assert drv._is_feasible(dict(base, max_pin_burnup=None)) is True
    # present-and-under passes; present-and-over fails.
    assert drv._is_feasible(dict(base, max_pin_burnup=78.0)) is True
    assert drv._is_feasible(dict(base, max_pin_burnup=88.0)) is False


def test_fr_boundary_objective_is_minus_fr_reachable_with_cyclen_none(tmp_path):
    drv = _frb_driver(tmp_path / "obj")
    row = {"converged": True, "f_r": 1.58, "cbc_max": 1500.0, "f_q": 2.30,
           "ao_abs": 0.20, "cyclen": 625.0}
    assert drv._campaign_objective(row) == pytest.approx(-1.58)
    # reachable (finite) even when cyclen is absent — placed BEFORE the cyclen guard.
    no_cy = {"converged": True, "f_r": 1.62, "cbc_max": 1500.0, "f_q": 2.30, "ao_abs": 0.20}
    assert drv._campaign_objective(no_cy) == pytest.approx(-1.62)
    # lower F_r is a strictly higher objective.
    assert drv._campaign_objective(row) > drv._campaign_objective(no_cy)
    # F_r absent -> -inf (invisible), never a crash.
    assert drv._campaign_objective({"converged": True, "cyclen": 625.0}) == float("-inf")


def test_fr_boundary_strict_restart_live_evaluator(tmp_path):
    # live path (no stub evaluator_factory) -> strict_restart is ON (no cross-pair
    # fallback ever); the stub path keeps the graceful fallback.
    live = _frb_driver(tmp_path / "live", live=True)
    assert live._resolver().strict_restart is True
    stub = _frb_driver(tmp_path / "stub", live=False)
    assert stub._resolver().strict_restart is False


def test_fr_boundary_best_dict_distance_none_and_keys(tmp_path):
    drv = _frb_driver(tmp_path / "best")
    row = {"converged": True, "record_id": "rZ", "f_r": 1.57, "cbc_max": 1500.0,
           "f_q": 2.30, "ao_abs": 0.20, "cyclen": 630.0, "max_pin_burnup": 70.0,
           "pattern": None}
    bd = drv._best_dict(row, drv._campaign_objective(row))
    assert bd["distance"] is None                     # no target window in this mode
    # exactly the keys the outer race consumes must be present and populated.
    assert bd["f_r"] == pytest.approx(1.57)
    assert bd["max_pin_burnup"] == pytest.approx(70.0)
    assert bd["feasible"] is True
    assert "f_r_margin_to_limit" in bd and "cyclen" in bd
    # _maybe_update_best routes fr_boundary through _best_dict (no missing-key path).
    drv._maybe_update_best(row, None)
    assert drv.best is not None
    assert set(("f_r", "max_pin_burnup", "feasible", "distance")) <= set(drv.best)
    assert drv.best["distance"] is None


def test_fr_boundary_elites_fr_ascending_feasible_first(tmp_path):
    import random as _random
    from lpopt.data.schema import SYM_CLASS, CanonicalRecord, compute_record_id, pack_pattern
    from lpopt.data.store import StoreWriter
    from lpopt.search.construct import CAMPAIGN_DECK_KNOBS
    from lpopt.search.genome import random_genome

    def _rec(seed, f_r, *, cbc=1400.0, feasible=True):
        pat = random_genome(_random.Random(seed), "K1_K2", 30).to_pattern()
        rid = compute_record_id(pat.canonical(), "ga80", "K1_K2", CAMPAIGN_DECK_KNOBS)
        return rid, f_r, CanonicalRecord(
            record_id=rid, dataset="P", campaign="c", stratum="frB",
            generator="local", parent_record_id=None, case_pair="K1_K2", feed=121,
            n_batches=2, depth2_edges=0, e_core=5.2, e_split=0.0, library_id="ga80",
            sym_class=SYM_CLASS, pattern=pack_pattern(pat),
            f_r=f_r, f_q=2.10, cbc_max=cbc, cbc_boc=None, cbc_kind="max",
            cyclen=650.0, ao_abs=0.12, cycle_burnup=None, discharge_burnup=None,
            max_assembly_burnup=None, max_pin_burnup=40.0, eoc_ppm=None, delta_efpd=None,
            n_cycles=12.0, converged=True, converged_at_cap=False, tolerance_margin=0.2,
            restart_provenance="native:MAS_RST", valid=True, failure="", maps_key=None,
        )

    # three feasible rows at F_r 1.70 / 1.50 / 1.60 + one INFEASIBLE (CBC 1650).
    r_hi = _rec(11, 1.70); r_lo = _rec(12, 1.50); r_mid = _rec(13, 1.60)
    r_bad = _rec(14, 1.45, cbc=1650.0)
    store_dir = tmp_path / "estore"
    store_dir.mkdir(parents=True, exist_ok=True)
    StoreWriter(store_dir).write_records([r[2] for r in (r_hi, r_lo, r_mid, r_bad)])

    drv = _frb_driver(tmp_path / "elt", store_dir=store_dir)
    fr_by_rid = {rid: fr for rid, fr, _ in (r_hi, r_lo, r_mid, r_bad)}
    elites = drv._store_elites()
    ordered = [rid for rid, _ in elites if rid in fr_by_rid]
    frs = [fr_by_rid[rid] for rid in ordered]
    # feasible rows come first, ascending in F_r; the CBC-infeasible row ranks last.
    assert frs[:3] == [1.50, 1.60, 1.70]
    assert frs[-1] == 1.45                            # infeasible backfill after feasible
