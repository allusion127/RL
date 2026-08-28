"""Outer race under the flatness program: coverage-deficit budget (decision D7),
the re-aimed fr_boundary roster (decision D5), and race_state key migration.

The regression this file exists for: ``proximity_weights`` allocated the outer
budget by ``|best_F_r - 1.55|`` for EVERY objective, including the flat_power
campaign whose inner objective contains no F_r at all.
"""

from __future__ import annotations

import json
import types

import pytest

from lpopt.search import frontier_search as fs
from lpopt.search.frontier_search import FrBoundaryOuterRace

from test_frontier_search import _cfg


# --------------------------------------------------------------------------- #
# D5 — fr_boundary RE-AIMED at the 16 empirically-compliant cells (program §5)
# --------------------------------------------------------------------------- #
def test_compliant_roster_is_the_16_cells_of_section_5():
    full = fs.build_roster()
    comp = fs.build_roster(compliant_only=True)
    assert len(full) == 24 and len(comp) == 16
    # feed 113 disappears entirely: no 113 cell holds a compliant row.
    assert {c.feed for c in comp} == {117, 121, 125}
    # the two (e_core, feed) combinations program §5 shows EMPTY are the ones
    # missing from the 5.0-5.5 x {117,121,125} block.
    missing = {(c.e_core, c.feed) for c in full} - {(c.e_core, c.feed) for c in comp}
    assert (5.4, 117) in missing and (5.5, 117) in missing
    assert all(fs.is_compliant_cell(c.e_core, c.feed) for c in comp)
    assert not fs.is_compliant_cell(5.0, 113)
    # a strict SUBSET: the re-aim removes cells, it never invents them.
    assert {c.cell_id for c in comp} < {c.cell_id for c in full}


def test_fr_boundary_race_defaults_to_the_compliant_roster(tmp_path):
    race = FrBoundaryOuterRace(_cfg("fr_boundary", compliant_only=True),
                               None, run_root=tmp_path / "a")
    assert race.objective == "fr_boundary" and race.compliant_only
    assert len(race.roster) == 16
    # the deck can still ask for the original 24 (reproducibility escape hatch).
    full = FrBoundaryOuterRace(_cfg("fr_boundary", compliant_only=False),
                               None, run_root=tmp_path / "b")
    assert len(full.roster) == 24


def test_flat_power_race_keeps_the_full_roster(tmp_path):
    """The re-aim is an fr_boundary decision; flat_power still covers the grid."""
    race = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path)
    assert race.objective == "flat_power"
    assert race.compliant_only is False and len(race.roster) == 24


def test_unknown_objective_is_clamped_to_fr_boundary(tmp_path):
    race = FrBoundaryOuterRace(_cfg("target_cycle"), None, run_root=tmp_path)
    assert race.objective == "fr_boundary"


# --------------------------------------------------------------------------- #
# D7 — flat_power allocates by PURE map-coverage deficit
# --------------------------------------------------------------------------- #
def test_coverage_weights_follow_the_deficit_and_sum_to_budget():
    roster = fs.build_roster()
    counts = {c.cell_id: 30 for c in roster}
    counts["E1_E2_f121"] = 55        # nearly covered -> small deficit
    counts["J1_J2_f121"] = 0         # empty          -> full deficit
    w = fs.coverage_weights(roster, counts, budget=276, target=60)
    assert sum(w.values()) == 276
    assert w["J1_J2_f121"] > w["E1_E2_f121"]
    assert min(w.values()) >= min(fs.FLOOR_WEIGHT, fs.PROBE_SIZE)


def test_coverage_weights_read_only_a_count():
    """D7's point: no model-derived quantity can enter this allocation."""
    roster = fs.build_roster()
    counts = {c.cell_id: 10 for c in roster}
    a = fs.coverage_weights(roster, counts, budget=276)
    b = fs.coverage_weights(roster, dict(counts), budget=276)
    assert a == b
    # equal coverage -> equal weight, so a heavily-sampled cell gains no
    # advantage (no rich-get-richer).
    non_probe = [a[c.cell_id] for c in roster if not c.is_probe]
    assert max(non_probe) - min(non_probe) <= 1      # largest-remainder only


def test_coverage_weights_split_evenly_once_every_cell_is_covered():
    roster = fs.build_roster()
    counts = {c.cell_id: 999 for c in roster}
    w = fs.coverage_weights(roster, counts, budget=276, target=60)
    assert sum(w.values()) == 276
    vals = sorted(w.values())
    assert vals[-1] - vals[0] <= 1


def test_coverage_weights_cap_an_unmapped_probe_cell():
    roster = fs.build_roster()
    counts = {c.cell_id: (0 if c.is_probe else 30) for c in roster}
    w = fs.coverage_weights(roster, counts, budget=276, target=60)
    for c in roster:
        if c.is_probe:
            assert w[c.cell_id] <= fs.PROBE_SIZE
    assert sum(w.values()) == 276


def test_coverage_weights_respect_exclusions_and_tiny_budgets():
    roster = fs.build_roster()
    roster[0].excluded = True
    counts = {c.cell_id: 0 for c in roster}
    w = fs.coverage_weights(roster, counts, budget=276)
    assert roster[0].cell_id not in w
    tiny = fs.coverage_weights(roster, counts, budget=4)
    assert all(v <= fs.FLOOR_WEIGHT for v in tiny.values())
    assert fs.coverage_weights([], {}, budget=276) == {}


def test_weights_for_round_dispatches_on_objective(tmp_path):
    seen = {}

    def _counts(cells):
        seen["called"] = True
        return {c.cell_id: 0 for c in cells}

    flat = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path / "f",
                               map_counts=_counts, log=lambda m: None)
    w = flat.weights_for_round(1)
    assert seen.get("called") and sum(w.values()) == fs.ROUND_BUDGET

    # fr_boundary still uses F_r proximity — its objective IS F_r.
    frb = FrBoundaryOuterRace(_cfg("fr_boundary", compliant_only=False), None,
                              run_root=tmp_path / "b", log=lambda m: None)
    frb.best_fr_feasible = {c.cell_id: 1.56 for c in frb.roster}
    frb.best_fr_feasible[frb.roster[0].cell_id] = 2.50     # far from the boundary
    frb.best_fr_converged = dict(frb.best_fr_feasible)
    w2 = frb.weights_for_round(1)
    assert sum(w2.values()) == fs.ROUND_BUDGET
    assert w2[frb.roster[0].cell_id] < max(w2.values())


def test_flat_power_allocation_ignores_best_fr_entirely(tmp_path):
    """Same coverage, wildly different best_F_r -> IDENTICAL allocation."""
    def _counts(cells):
        return {c.cell_id: 20 for c in cells}

    race = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path,
                               map_counts=_counts, log=lambda m: None)
    base = race.weights_for_round(1)
    race.best_fr_feasible = {c.cell_id: 1.55 for c in race.roster}
    race.best_fr_converged = {c.cell_id: 1.55 for c in race.roster}
    assert race.weights_for_round(1) == base


def test_round1_weights_are_unchanged_for_both_objectives(tmp_path):
    flat = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path / "f",
                               log=lambda m: None)
    assert flat.weights_for_round(0) == fs.round1_weights(flat.roster)


def test_map_count_read_failure_is_not_fatal(tmp_path):
    def _boom(cells):
        raise RuntimeError("store locked")

    logs: list[str] = []
    race = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path,
                               map_counts=_boom, log=logs.append)
    w = race.weights_for_round(1)
    assert sum(w.values()) == fs.ROUND_BUDGET
    assert any("map-count read failed" in m for m in logs)


def test_store_map_counts_is_safe_without_a_store(tmp_path):
    roster = fs.build_roster()
    assert fs.store_map_counts(None, roster) == {c.cell_id: 0 for c in roster}
    assert fs.store_map_counts(tmp_path / "nope", roster) == {
        c.cell_id: 0 for c in roster}


# --------------------------------------------------------------------------- #
# D7 — race_state key migration (explicit, logged, never silent)
# --------------------------------------------------------------------------- #
def test_objective_change_resets_scoped_keys_and_logs_loudly(tmp_path):
    logs: list[str] = []
    frb = FrBoundaryOuterRace(_cfg("fr_boundary", compliant_only=False), None,
                              run_root=tmp_path, log=logs.append)
    frb.best_fr_feasible = {"E1_E2_f121": 1.52}
    frb.best_fr_converged = {"E1_E2_f121": 1.52}
    frb.cell_spent = {"E1_E2_f121": 40}
    frb.probe_strikes = {"L1_L2_f113": 1}
    frb.exclusions = {"G3_G4_f113": "structural"}
    frb.round_index = 3
    frb._save_race_state()

    logs.clear()
    flat = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path,
                               log=logs.append)
    joined = "\n".join(logs)
    assert "MIGRATION" in joined and "fr_boundary" in joined and "flat_power" in joined
    # objective-SCOPED keys reset...
    assert flat.best_fr_feasible == {} and flat.best_fr_converged == {}
    assert flat.map_counts == {}
    # ...objective-INDEPENDENT accounting kept (losing it would re-charge spend).
    assert flat.round_index == 3
    assert flat.probe_strikes.get("L1_L2_f113") == 1
    assert "G3_G4_f113" in flat.exclusions
    # the reset is auditable in the persisted state.
    flat._save_race_state()
    st = json.loads((tmp_path / "race_state.json").read_text(encoding="utf-8"))
    assert st["objective"] == "flat_power"
    assert st["migrations"] and st["migrations"][0]["from_objective"] == "fr_boundary"


def test_same_objective_reload_does_not_migrate(tmp_path):
    logs: list[str] = []
    a = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path,
                            log=logs.append)
    a.best_fr_converged = {"E1_E2_f121": 1.60}
    a.map_counts = {"E1_E2_f121": 12}
    a._save_race_state()
    logs.clear()
    b = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path,
                            log=logs.append)
    assert "MIGRATION" not in "\n".join(logs)
    assert b.map_counts == {"E1_E2_f121": 12}
    assert b.migrations == []


def test_pre_v2_state_without_an_objective_key_is_named_not_guessed(tmp_path):
    (tmp_path / "race_state.json").write_text(json.dumps({
        "round_index": 2, "cell_spent": {"E1_E2_f121": 16},
        "best_fr_feasible": {"E1_E2_f121": 1.52},
    }), encoding="utf-8")
    logs: list[str] = []
    race = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path,
                               log=logs.append)
    joined = "\n".join(logs)
    assert "pre-v2" in joined and "MIGRATION" in joined
    assert race.best_fr_feasible == {}          # reset: an fr_boundary-only key
    assert race.round_index == 2                 # accounting kept


def test_roster_change_is_logged_with_the_off_roster_cells(tmp_path):
    logs: list[str] = []
    full = FrBoundaryOuterRace(_cfg("fr_boundary", compliant_only=False), None,
                               run_root=tmp_path, log=logs.append)
    full.cell_spent = {c.cell_id: 8 for c in full.roster}
    full._save_race_state()
    logs.clear()
    comp = FrBoundaryOuterRace(_cfg("fr_boundary", compliant_only=True), None,
                               run_root=tmp_path, log=logs.append)
    joined = "\n".join(logs)
    assert "MIGRATION" in joined and "full24" in joined and "compliant16" in joined
    assert any("f113" in m for m in logs)
    assert comp.migrations and comp.migrations[0]["off_roster_cells"]


def test_roster_report_carries_the_objective_and_coverage(tmp_path):
    race = FrBoundaryOuterRace(_cfg("flat_power"), None, run_root=tmp_path,
                               map_counts=lambda cells: {c.cell_id: 7 for c in cells},
                               log=lambda m: None)
    race.refresh_map_counts()
    rep = race.roster_report()
    assert len(rep) == 24
    assert all(r["objective"] == "flat_power" for r in rep)
    assert all(r["n_mapped"] == 7 for r in rep)
    json.dumps(rep)
