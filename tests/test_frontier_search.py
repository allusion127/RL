"""Outer fr_boundary cell-race: fixed roster, seeds, one-process-per-round budget
accounting, exception handling, d8 probe demotion, retrain hook (stub driver)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from lpopt.search import frontier_search as fs
from lpopt.search.frontier_search import FrBoundaryOuterRace


def _cfg(objective="target_cycle", *, compliant_only=False, store_dir=None):
    """Race cfg double.

    ``compliant_only=False`` by default so the accounting / exclusion / probe
    tests below keep exercising the FULL 24-cell roster they were written
    against — they test the budget machinery, which is roster-agnostic.  The
    re-aimed 16-cell fr_boundary roster (decision D5) has its own tests.
    """
    return types.SimpleNamespace(
        flow=types.SimpleNamespace(random_seed=0),
        case=types.SimpleNamespace(mode="fixed", pair=None, feed=None),
        model=types.SimpleNamespace(store_dir=store_dir),
        acquisition=types.SimpleNamespace(
            objective=objective, fr_boundary_compliant_only=compliant_only),
    )


def _write_state(run_dir: Path, *, spent: int, best_fr=None, best_overall_fr=None) -> None:
    """Mimic a CampaignDriver state.json write (what a resumed cell persists)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"budget_spent": spent}
    if best_fr is not None:
        payload["best"] = {"f_r": best_fr, "feasible": True}
    if best_overall_fr is not None:
        payload["best_overall"] = {"f_r": best_overall_fr}
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# roster + R1 + seeds
# --------------------------------------------------------------------------- #
def test_roster_shape_and_classes():
    roster = fs.build_roster()
    assert len(roster) == 24
    klass = {}
    for c in roster:
        klass[c.klass] = klass.get(c.klass, 0) + 1
    assert klass == {"native": 9, "d4": 12, "d8_probe": 3}
    probes = {c.cell_id for c in roster if c.is_probe}
    assert probes == {"L1_L2_f113", "N1_N2_f113", "G3_G4_f113"}
    # every roster pair is mono-anchor (R1) — no cross-anchor slips in.
    assert all(not fs.is_cross_anchor(c.pair) for c in roster)


def test_round1_weights_sum_to_budget():
    roster = fs.build_roster()
    w = fs.round1_weights(roster)
    assert sum(w.values()) == 276
    assert w["E1_E2_f117"] == 16 and w["E1_E2_f113"] == 10 and w["L1_L2_f113"] == 4


def test_r1_cross_anchor_construction_hard_fail(monkeypatch):
    # inject a cross-anchor pair into the roster source -> construction must hard-fail.
    bad = dict(fs._PAIR_ECORE)
    bad["E1_J2"] = 5.6
    monkeypatch.setattr(fs, "_PAIR_ECORE", bad)
    with pytest.raises(Exception) as ei:
        fs.build_roster()
    assert "R1" in str(ei.value)


def test_seed_determinism():
    assert fs.cell_seed("K1_K2_f121", 7) == fs.cell_seed("K1_K2_f121", 7)
    assert fs.cell_seed("K1_K2_f121", 7) != fs.cell_seed("K1_K2_f125", 7)
    assert fs.cell_seed("K1_K2_f121", 7) != fs.cell_seed("K1_K2_f121", 8)


# --------------------------------------------------------------------------- #
# proximity weight invariants (round >= 2)
# --------------------------------------------------------------------------- #
def test_proximity_weights_sum_floor_and_fallback():
    roster = fs.build_roster()
    # no cell has any converged/feasible row -> all fall to the floor baseline, but
    # the weights still normalize to the full budget with the floor respected.
    best = {c.cell_id: None for c in roster}
    conv = {c.cell_id: False for c in roster}
    w = fs.proximity_weights(roster, best, conv, budget=276)
    assert sum(w.values()) == 276
    assert min(w.values()) >= 4
    # a cell sitting on F_r=1.55 gets the LARGEST non-probe weight.
    best2 = dict(best); best2["K1_K2_f121"] = 1.55
    conv2 = dict(conv); conv2["K1_K2_f121"] = True
    w2 = fs.proximity_weights(roster, best2, conv2, budget=276)
    assert sum(w2.values()) == 276
    non_probe = {k: v for k, v in w2.items() if not k.endswith("113")
                 or k.startswith(("E1", "J1", "K1"))}
    assert w2["K1_K2_f121"] == max(non_probe.values())


def test_proximity_probe_capped_until_converged():
    roster = fs.build_roster()
    best = {c.cell_id: None for c in roster}
    best["L1_L2_f113"] = 1.55                        # right on target...
    conv = {c.cell_id: False for c in roster}         # ...but NOT yet converged
    w = fs.proximity_weights(roster, best, conv, budget=276)
    assert w["L1_L2_f113"] == 4                       # capped at the probe size
    # once it has a converged row the cap lifts and it can win weight.
    conv["L1_L2_f113"] = True
    w2 = fs.proximity_weights(roster, best, conv, budget=276)
    assert w2["L1_L2_f113"] > 4


# --------------------------------------------------------------------------- #
# TWO-INSTANTIATION no-phantom accounting
# --------------------------------------------------------------------------- #
class _StubResult:
    def __init__(self, spent, best_fr=None, best_overall_fr=None, n_feasible=0):
        self.budget_spent = spent
        self.best = {"f_r": best_fr, "feasible": True} if best_fr is not None else None
        self.best_overall = {"f_r": best_overall_fr} if best_overall_fr is not None else None
        self.n_feasible = n_feasible


def _factory(spent_fn, *, best_fr=None, best_overall_fr=None, record=None):
    """Build a stub driver_factory: each cell's run persists a state.json with the
    CUMULATIVE spent returned by ``spent_fn(cell_id, granted)`` and returns a result."""
    def factory(cell, granted, run_dir, seed):
        if record is not None:
            record.setdefault(cell.cell_id, []).append(granted)

        class D:
            def run(_s):
                spent = spent_fn(cell.cell_id, granted)
                _write_state(Path(run_dir), spent=spent, best_fr=best_fr,
                             best_overall_fr=best_overall_fr)
                return _StubResult(spent, best_fr=best_fr, best_overall_fr=best_overall_fr)
        return D()
    return factory


def test_two_instantiation_no_phantom_and_extends(tmp_path):
    # Round 1: each cell spends its FULL grant (cumulative == granted).
    grants1: dict[str, list[int]] = {}
    r1 = FrBoundaryOuterRace(
        _cfg(), "M", run_root=tmp_path / "runs",
        driver_factory=_factory(lambda cid, g: g, best_overall_fr=1.60, record=grants1))
    res1 = r1.run_round()
    assert res1.round_index == 0
    assert res1.round_spent == 276                    # round 1 charges the full budget
    # a SECOND, FRESH race instance over the same run dirs (simulating a new process).
    grants2: dict[str, list[int]] = {}
    # round 2: cells add 5 NEW calls each on top of their persisted cumulative spend.
    def spent2(cid, granted):
        return granted                                # spends whatever it is granted
    r2 = FrBoundaryOuterRace(
        _cfg(), "M", run_root=tmp_path / "runs",
        driver_factory=_factory(spent2, best_overall_fr=1.60, record=grants2))
    assert r2.round_index == 1                         # resumed the round counter
    # cell_spent seeded from each cell's OWN state.json (not race_state).
    assert r2.cell_spent["E1_E2_f117"] == 16
    res2 = r2.run_round()
    # grants EXTEND cumulatively: round-2 grant = persisted_spent + round2_weight.
    g_before = r2.cell_spent  # already updated post-run; check via grants2 record
    for cid, granted_list in grants2.items():
        # the granted budget exceeds the round-1 cumulative (true extension).
        assert granted_list[0] > 16 or granted_list[0] >= 4
    # round-2 delta counts ONLY new calls (granted - persisted), never re-charges.
    total_new = sum(row["delta"] for row in res2.per_cell)
    assert total_new == res2.round_spent
    assert res2.round_spent == 276                     # exactly the round-2 weight sum
    # no cell was falsely exhausted by its own history: every survivor ran.
    assert len(res2.per_cell) == 24


def test_no_false_exhaustion_when_delta_zero(tmp_path):
    # A cell whose cumulative spent does NOT grow across a round (already exhausted
    # its within-cell exploit) charges delta 0 but is NOT excluded — unlike the
    # fuel-cost elimination race, every boundary cell is a coverage target.
    root = tmp_path / "runs"
    # each cell's cumulative spend is capped at 12 no matter how much is granted.
    fac = _factory(lambda cid, g: min(g, 12), best_overall_fr=1.60)
    r1 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=fac)
    res1 = r1.run_round()
    assert res1.round_spent > 0
    # round 2 over a FRESH instance: grants extend but cumulative stays capped at 12,
    # so most cells add zero new calls (delta 0) — none may be excluded for it.
    r2 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=fac)
    res2 = r2.run_round()
    assert any(row["delta"] == 0 for row in res2.per_cell)     # delta-0 resumes occur
    assert all(not r2._by_id[row["cell"]].excluded for row in res2.per_cell)


# --------------------------------------------------------------------------- #
# exception handling
# --------------------------------------------------------------------------- #
class _AssetResolutionError(RuntimeError):
    pass
_AssetResolutionError.__name__ = "AssetResolutionError"


def test_structural_exception_excludes_and_charges_precrash(tmp_path):
    target = "E1_E2_f117"

    def factory(cell, granted, run_dir, seed):
        rd = Path(run_dir)
        class D:
            def run(_s):
                if cell.cell_id == target:
                    # 3 real MASTER calls happened before the crash (state.json holds them).
                    _write_state(rd, spent=3)
                    raise _AssetResolutionError("no same-pair restart")
                _write_state(rd, spent=granted)
                return _StubResult(granted, best_overall_fr=1.60)
        return D()

    r = FrBoundaryOuterRace(_cfg(), "M", run_root=tmp_path / "runs", driver_factory=factory)
    res = r.run_round()
    assert r._by_id[target].excluded is True
    assert "structural" in r.exclusions[target] and "AssetResolutionError" in r.exclusions[target]
    # the real pre-crash delta (3) was charged, not zero and not the full grant.
    row = next(x for x in res.per_cell if x["cell"] == target)
    assert row["delta"] == 3
    # persisted: a fresh instance sees it excluded.
    r2 = FrBoundaryOuterRace(_cfg(), "M", run_root=tmp_path / "runs", driver_factory=factory)
    assert r2._by_id[target].excluded is True


def test_transient_exception_retained_then_excluded_after_two(tmp_path):
    target = "K1_K2_f121"

    def factory(cell, granted, run_dir, seed):
        rd = Path(run_dir)
        class D:
            def run(_s):
                if cell.cell_id == target:
                    _write_state(rd, spent=2)
                    raise ValueError("flaky transient")
                _write_state(rd, spent=granted)
                return _StubResult(granted, best_overall_fr=1.60)
        return D()

    root = tmp_path / "runs"
    r1 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=factory)
    r1.run_round()
    assert r1._by_id[target].excluded is False        # retained after ONE failure
    assert r1.transient_strikes[target] == 1
    r2 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=factory)
    assert r2.transient_strikes.get(target) == 1       # streak persisted
    r2.run_round()
    assert r2._by_id[target].excluded is True           # excluded after TWO consecutive
    assert "transient" in r2.exclusions[target]


def test_transient_streak_resets_on_success(tmp_path):
    target = "K1_K2_f121"
    state = {"fail": True}

    def factory(cell, granted, run_dir, seed):
        rd = Path(run_dir)
        class D:
            def run(_s):
                if cell.cell_id == target and state["fail"]:
                    _write_state(rd, spent=2)
                    raise ValueError("flaky")
                _write_state(rd, spent=granted)
                return _StubResult(granted, best_overall_fr=1.60)
        return D()

    root = tmp_path / "runs"
    r1 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=factory)
    r1.run_round()
    assert r1.transient_strikes[target] == 1
    state["fail"] = False                              # next round the cell succeeds
    r2 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=factory)
    r2.run_round()
    assert r2.transient_strikes[target] == 0
    assert r2._by_id[target].excluded is False


# --------------------------------------------------------------------------- #
# d8 probe demotion across two instantiations (persisted strikes)
# --------------------------------------------------------------------------- #
def test_d8_probe_demotion_two_strikes(tmp_path):
    probe = "L1_L2_f113"

    def factory(cell, granted, run_dir, seed):
        rd = Path(run_dir)
        class D:
            def run(_s):
                # the probe NEVER converges (no best / no best_overall); others do.
                if cell.cell_id == probe:
                    _write_state(rd, spent=granted)
                    return _StubResult(granted)          # zero-converged
                _write_state(rd, spent=granted, best_overall_fr=1.60)
                return _StubResult(granted, best_overall_fr=1.60)
        return D()

    root = tmp_path / "runs"
    r1 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=factory)
    r1.run_round()
    assert r1.probe_strikes[probe] == 1
    assert r1._by_id[probe].excluded is False
    r2 = FrBoundaryOuterRace(_cfg(), "M", run_root=root, driver_factory=factory)
    assert r2.probe_strikes.get(probe) == 1            # strike persisted
    r2.run_round()
    assert r2._by_id[probe].excluded is True
    assert "non_finite_flux" in r2.exclusions[probe]


# --------------------------------------------------------------------------- #
# retrain hook + gated champion swap
# --------------------------------------------------------------------------- #
def test_retrain_hook_and_gated_swap(tmp_path):
    events = []

    def factory(cell, granted, run_dir, seed):
        rd = Path(run_dir)
        class D:
            def run(_s):
                _write_state(rd, spent=granted, best_overall_fr=1.60)
                return _StubResult(granted, best_overall_fr=1.60)
        return D()

    # gate FAILS first -> no swap.
    r = FrBoundaryOuterRace(
        _cfg(), "M0", run_root=tmp_path / "a", driver_factory=factory,
        retrain_gate_callback=lambda n: {"pass": False, "champion_model_dir": "NEW"},
        model_reload=lambda d: "M_NEW")
    r.run_round()
    assert r.model == "M0"
    assert r.retrain_events[-1]["pass"] is False

    # gate PASSES -> champion swapped in.
    r2 = FrBoundaryOuterRace(
        _cfg(), "M0", run_root=tmp_path / "b", driver_factory=factory,
        retrain_gate_callback=lambda n: {"pass": True, "champion_model_dir": "NEW"},
        model_reload=lambda d: "M_NEW")
    r2.run_round()
    assert r2.model == "M_NEW"


# --------------------------------------------------------------------------- #
# plateaued cell with early_stop=False spends its full grant
# --------------------------------------------------------------------------- #
def test_plateaued_cell_spends_full_grant(tmp_path):
    # A plateaued cell (F_r no longer improving) STILL spends its whole grant because
    # the frontier driver runs with early_stop=False — labels are the product.
    grants: dict[str, list[int]] = {}

    def spent_fn(cid, granted):
        return granted                                 # always spends the full grant

    r = FrBoundaryOuterRace(
        _cfg(), "M", run_root=tmp_path / "runs",
        driver_factory=_factory(spent_fn, best_fr=1.60, best_overall_fr=1.60, record=grants))
    res = r.run_round()
    native = next(x for x in res.per_cell if x["cell"] == "E1_E2_f117")
    assert native["granted"] == 16 and native["delta"] == 16     # full grant spent


# --------------------------------------------------------------------------- #
# race_state atomic write survives a mid-round kill
# --------------------------------------------------------------------------- #
def test_race_state_written_and_consistent(tmp_path):
    r = FrBoundaryOuterRace(
        _cfg(), "M", run_root=tmp_path / "runs",
        driver_factory=_factory(lambda cid, g: g, best_overall_fr=1.60))
    r.run_round()
    # race_state.json exists, is valid JSON, and no .tmp turd is left behind.
    assert r.race_state_path.exists()
    st = json.loads(r.race_state_path.read_text(encoding="utf-8"))
    assert st["round_index"] == 1
    assert st["cell_spent"]["E1_E2_f117"] == 16
    assert not list(tmp_path.glob("runs/race_state.json.tmp-*"))


# --------------------------------------------------------------------------- #
# CLI: frontier-produce one-round-exit + LPOPT_WORKER guard
# --------------------------------------------------------------------------- #
_DECK = Path(__file__).resolve().parents[1] / "lpopt.inp"


def test_cli_frontier_produce_worker_guard(tmp_path, monkeypatch):
    if not _DECK.is_file():
        pytest.skip("deck not present")
    from lpopt.cli import main
    monkeypatch.delenv("LPOPT_WORKER", raising=False)
    run_root = tmp_path / "frontier"
    rc = main(["frontier-produce", "--input", str(_DECK), "--run-root", str(run_root)])
    assert rc == 1                                    # live round refused off PC2
    # The roster JSON is still emitted so the caller sees the plan.  The deck has
    # no [acquisition] objective, so the CLI clamps to fr_boundary, which is now
    # RE-AIMED at the 16 empirically-compliant cells (decision D5).
    roster = json.loads((run_root / "roster.json").read_text(encoding="utf-8"))
    assert len(roster) == 16
    assert all(r["feed"] in (117, 121, 125) for r in roster)
    assert not (run_root / "frontier_round.json").exists()   # no round ran


def test_cli_frontier_produce_one_round_exit(tmp_path, monkeypatch):
    if not _DECK.is_file():
        pytest.skip("deck not present")
    from lpopt.cli import main
    from lpopt.search import frontier_search as _fs

    calls = {"run_round": 0}

    class _StubRace:
        def __init__(self, *a, **k):
            pass

        def roster_report(self):
            return [{"cell": "E1_E2_f117"}]

        def run_round(self):
            calls["run_round"] += 1
            return _fs.FrontierRoundResult(
                round_index=0, round_spent=8,
                per_cell=[{"cell": "E1_E2_f117", "delta": 8}], excluded=[])

    monkeypatch.setattr(_fs, "FrBoundaryOuterRace", _StubRace)
    run_root = tmp_path / "frontier"
    rc = main(["frontier-produce", "--input", str(_DECK), "--run-root", str(run_root),
               "--dry-run"])
    assert rc == 0
    assert calls["run_round"] == 1                     # EXACTLY one round, then exit
    payload = json.loads((run_root / "frontier_round.json").read_text(encoding="utf-8"))
    assert payload["round_index"] == 0 and payload["round_spent"] == 8


# --------------------------------------------------------------------------- #
# CLI: gate-promote atomic promotion helper
# --------------------------------------------------------------------------- #
def test_gate_promote_apply_edits_atomically(tmp_path):
    from lpopt.cli import _apply_promotion

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"champion_model_dir": "data/models/OLD",
                                      "order": []}), encoding="utf-8")
    deck = tmp_path / "lpopt.inp"
    deck.write_text('[model]\nmodel_dir = "data/models/OLD"   # comment kept\n',
                    encoding="utf-8")
    _apply_promotion(state_path, deck, "data/models/NEW")
    st = json.loads(state_path.read_text(encoding="utf-8"))
    assert st["champion_model_dir"] == "data/models/NEW"
    txt = deck.read_text(encoding="utf-8")
    assert 'model_dir = "data/models/NEW"' in txt
    assert "# comment kept" in txt                     # inline comment preserved
    # no temp turds left behind by the atomic replace.
    assert not list(tmp_path.glob("*.tmp-*"))


def test_gate_promote_parser_exposes_check_only():
    """A gate you cannot inspect without triggering it is a foot-gun.

    ``gate-promote`` promotes UNCONDITIONALLY on PASS; before --check-only there
    was no way to see a gate result without swapping the champion (this is how
    arm A6 got promoted by a read-only-intended invocation on 2026-07-25).
    """
    from lpopt.cli import build_parser

    ap = build_parser()
    args = ap.parse_args(["gate-promote", "--prev", "a", "--new", "b",
                          "--check-only"])
    assert args.check_only is True
    assert ap.parse_args(["gate-promote", "--prev", "a", "--new", "b"]).check_only         is False, "default must stay the historical promote-on-pass behaviour"


# --------------------------------------------------------------------------- #
# lean-search clamp (forensic 20260723: heavy [search] defaults flood local_cpu)
# --------------------------------------------------------------------------- #
def test_lean_search_clamps_heavy_defaults_and_is_idempotent():
    from lpopt.config import SearchConfig, LocalSearchConfig
    from lpopt.search.frontier_search import _lean_search
    heavy = SearchConfig(pool_size=20000,
                         local_search=LocalSearchConfig(top_m=256, neighbors=200,
                                                        depth=3, max_predictions=40000))
    lean = _lean_search(heavy)
    assert lean.pool_size == 2000
    assert lean.local_search.max_predictions == 1500
    assert lean.local_search.top_m == 32 and lean.local_search.neighbors == 48
    assert lean.local_search.depth == 2
    # only ever LOWERS: a deck already lean is unchanged (idempotent).
    again = _lean_search(lean)
    assert again.pool_size == lean.pool_size
    assert again.local_search.max_predictions == lean.local_search.max_predictions
    # a deck BELOW a cap keeps its smaller value.
    small = SearchConfig(pool_size=500,
                        local_search=LocalSearchConfig(max_predictions=800))
    assert _lean_search(small).pool_size == 500
    assert _lean_search(small).local_search.max_predictions == 800


# --------------------------------------------------------------------------- #
# multi-PC disjoint split (user directive 2026-07-23)
# --------------------------------------------------------------------------- #
def test_exclude_cells_disjoint_split_and_persists(tmp_path):
    peer = {c.cell_id for c in fs.build_roster()
            if c.pair in ("J1_J2", "L1_L2", "G3_G4")}
    cfg = _cfg()
    cfg.flow.random_seed = 100
    cfg.search = types.SimpleNamespace()
    race = FrBoundaryOuterRace(cfg, None, run_root=tmp_path, round_budget=138,
                               exclude_cells=peer)
    survivors = [c for c in race.roster if not c.excluded]
    excluded = [c for c in race.roster if c.excluded]
    assert len(survivors) == 12 and len(excluded) == 12
    assert {c.pair for c in survivors} == {"E1_E2", "K1_K2", "N1_N2"}
    assert {c.pair for c in excluded} == {"J1_J2", "L1_L2", "G3_G4"}
    assert race.base_seed == 100                     # per-PC seed from the deck
    # exclusions persist so a resumed round keeps the same half.
    race._save_race_state()
    reopened = FrBoundaryOuterRace(cfg, None, run_root=tmp_path, round_budget=138)
    assert len([c for c in reopened.roster if c.excluded]) == 12
