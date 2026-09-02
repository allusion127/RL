"""autoeng driver — unit tests.

Three things are load-bearing and are tested against a RecordingRunner (the
StubEvaluator pattern: one injected seam, no MASTER, no ssh, no writes outside
tmp_path):

1. the generated deck really is the f113 recipe (every non-overridden knob equal),
   it LOADS through the real strict deck loader, and the prereg is written before
   the campaign step in the plan;
2. the state log is append-only and a kill -9 costs exactly one step, with the
   MASTER-call ledger surviving the crash;
3. the human gates fire — new_assembly halts, a failed promotion gate halts, and
   a forbidden box is refused before any command runs.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import autoeng as ae  # noqa: E402

PARENT_DECK = ROOT / ae.DEFAULT_PARENT_DECK
STORE = ROOT / "data" / "store" / "records.parquet"
FRONTIER = ROOT / "data" / "reports" / "dbx_frontier_table.csv"
CHAMP_STATE = ROOT / "data" / "curriculum" / "state.json"

needs_repo = pytest.mark.skipif(
    not (PARENT_DECK.exists() and STORE.exists() and CHAMP_STATE.exists()),
    reason="needs the repo's parent deck + store + curriculum state",
)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
class RecordingRunner:
    """Records every command instead of running it; returns a scripted rc."""

    def __init__(self, rcs: dict[str, int] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.rcs = rcs or {}

    def __call__(self, argv, *, cwd, timeout=3600):
        self.calls.append(list(argv))
        for key, rc in self.rcs.items():
            if key in " ".join(argv):
                return ae.RunResult(rc, stdout="0\nDONE\n")
        return ae.RunResult(0, stdout="0\nDONE\n")


def _cfg(tmp_path: Path, targets, **kw) -> ae.AutoengConfig:
    cfg = ae.AutoengConfig(root=ROOT, run_id="t_" + tmp_path.name, targets=tuple(targets), **kw)
    # keep every write inside tmp_path
    cfg.notebook = str(tmp_path / "AUTOENG_LOG.md")
    object.__setattr__(cfg, "_tmp", tmp_path)
    cfg.__class__.run_dir = property(lambda self, _t=tmp_path: _t / "run")   # type: ignore[assignment]
    return cfg


@pytest.fixture(autouse=True)
def _restore_run_dir():
    original = ae.AutoengConfig.run_dir
    yield
    ae.AutoengConfig.run_dir = original


# --------------------------------------------------------------------------- #
# 1. config
# --------------------------------------------------------------------------- #
def test_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text('[autoeng]\nrun_id = "x"\nopen_budgetz = 3\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        ae.load_autoeng_config(p)


def test_config_rejects_unknown_target_key(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text('[[targets]]\npair = "N1_N2"\nfeed = 109\nlibary = "ga80"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        ae.load_autoeng_config(p)


def test_fleet_guard_refuses_forbidden_box(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text('[fleet]\ncampaign_host = "USER@HOST_181"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="FLEET GUARD"):
        ae.load_autoeng_config(p)


def test_guard_argv_refuses_production_box(tmp_path):
    cfg = _cfg(tmp_path, [])
    with pytest.raises(RuntimeError, match="FLEET GUARD"):
        ae.guard_argv(cfg, ["scp", "x", "USER@HOST_198:/kit/"])
    ae.guard_argv(cfg, ["scp", "x", "USER@HOST_199:/kit/"])       # allowed


def test_shipped_config_loads_and_is_guarded():
    cfg = ae.load_autoeng_config(ROOT / "autoeng.toml")
    assert cfg.targets and all(t.library == "ga80" for t in cfg.targets)
    assert "HOST_181" in cfg.fleet.forbidden
    assert set(cfg.pause_for_approval) == {
        "new_assembly", "retrain_promote_fail", "budget_exceeded"}


# --------------------------------------------------------------------------- #
# 2. derived knobs
# --------------------------------------------------------------------------- #
def test_seed_is_deterministic_and_never_a_spent_one():
    a = ae.derive_seed("N1_N2", 109)
    assert a == ae.derive_seed("N1_N2", 109)
    assert a != ae.derive_seed("N1_N2", 117)
    for used in ae.USED_SEEDS:
        assert ae.derive_seed("X_Y", 101, used=[used, a]) not in (used, a)


def test_near_miss_sits_below_the_store_floor_and_above_the_reachable_floor():
    # the f113 numbers: store floor 1.7243, bias-corrected mesh floor 1.5960.
    k = ae.derive_deck_knobs({"pair": "N1_N2", "feed": 113, "store_f_r_floor": 1.7243,
                              "db_f_r_min": 1.5374, "corrected_floor": 1.5960,
                              "store_cyclen_min": 617.5, "store_cyclen_max": 651.0,
                              "db_best_efpd": 659.7})
    assert 1.5960 < k["near_miss_f_r"] < 1.7243
    assert k["near_miss_f_r"] == pytest.approx(1.66)          # human chose 1.65
    assert k["cycle_target_efpd"] == pytest.approx(659.7)     # the DB-truth EFPD
    # the tolerance must actually SPAN the cell, which the hand-written 30.0 did not
    assert k["cycle_target_efpd"] - k["cycle_tolerance_efpd"] <= 617.5


def test_near_miss_when_we_already_match_the_db():
    # L3_L4/f121: store floor 1.5217 vs DB 1.5215 — already on the frontier, so the
    # bound must land just UNDER our own floor, not above it.
    k = ae.derive_deck_knobs({"pair": "L3_L4", "feed": 121, "store_f_r_floor": 1.5217,
                              "db_f_r_min": 1.5215, "corrected_floor": None})
    assert k["near_miss_f_r"] < 1.5217


def test_next_arm_sequence():
    assert ae.next_arm("data/models/s1g") == ("s1h", "S1h", "S1g")
    assert ae.next_arm("data/models/s1f") == ("s1g", "S1g", "S1f")
    with pytest.raises(ValueError):
        ae.next_arm("data/models/20260721_105824")


# --------------------------------------------------------------------------- #
# 3. deck generation — the recipe must be CARRIED, not retyped
# --------------------------------------------------------------------------- #
@needs_repo
def test_generated_deck_is_the_parent_recipe_apart_from_the_cell_knobs(tmp_path):
    cfg = _cfg(tmp_path, [])
    target = ae.Target(pair="N1_N2", feed=109)
    marks = ae.measure_marks(cfg, target)
    text, overrides = ae.build_deck(cfg, target, marks, "data/models/s1g")

    parent = tomllib.loads(PARENT_DECK.read_text(encoding="utf-8"))
    child = tomllib.loads(text)

    for sec in ae.DROP_SECTIONS:
        assert sec not in child, f"[{sec}] must be dropped (recorded defect fix)"
    changed = set(overrides)
    for sec, table in parent.items():
        if sec in ae.DROP_SECTIONS:
            continue
        assert sec in child
        for key, val in table.items():
            if isinstance(val, dict):
                assert child[sec][key] == val, f"[{sec}.{key}] sub-table drifted"
                continue
            if f"{sec}.{key}" in changed:
                continue
            assert child[sec][key] == val, f"[{sec}] {key} drifted from the parent recipe"
    # and the overrides really landed
    assert child["case"]["pair"] == "N1_N2" and child["case"]["feed"] == 109
    assert child["model"]["model_dir"] == "data/models/s1g"
    assert child["acquisition"]["budget"] == 100
    # the objective and the lambda that MAKE it the f113 recipe are untouched
    assert child["acquisition"]["objective"] == "min_fr_max_cycle"
    assert child["acquisition"]["minfr_lambda"] == 1000.0


@needs_repo
def test_generated_deck_loads_through_the_real_strict_loader(tmp_path):
    from lpopt.config import load_config

    cfg = _cfg(tmp_path, [])
    target = ae.Target(pair="N1_N2", feed=109)
    marks = ae.measure_marks(cfg, target)
    text, _ = ae.build_deck(cfg, target, marks, "data/models/s1g")
    p = tmp_path / "gen.inp"
    p.write_text(text, encoding="utf-8")
    loaded = load_config(p)          # raises on any unknown/retired key
    assert loaded.case.pair == "N1_N2" and loaded.case.feed == 109
    assert loaded.acquisition.objective == "min_fr_max_cycle"
    assert loaded.search.near_miss_f_r == pytest.approx(marks["near_miss_f_r"])


@needs_repo
def test_prereg_header_pins_every_mark(tmp_path):
    cfg = _cfg(tmp_path, [])
    target = ae.Target(pair="N1_N2", feed=109)
    marks = ae.measure_marks(cfg, target)
    text, _ = ae.build_deck(cfg, target, marks, "data/models/s1g")
    header = text.split("\n[flow]")[0]
    assert "PRE-REGISTRATION" in header
    for token in ("THE MARKS", "OUR STORE'S FLOOR", "DB TRUTH",
                  "THE MODEL'S OWN PREDICTED FLOOR", "SUCCESS", "NULL"):
        assert token in header, f"prereg lost the {token!r} block"
    assert f"{marks['store_f_r_floor']:.4f}" in header
    assert f"{marks['db_f_r_min']:.4f}" in header
    # the pin-burnup cliff at this cell must be registered IN ADVANCE
    assert "PIN-BURNUP CLIFF" in header


def test_toml_emitter_refuses_array_of_tables():
    with pytest.raises(TypeError, match="array-of-tables"):
        ae._toml_dumps({"produce": {"strata": [{"name": "a"}]}})


def test_toml_emitter_roundtrips_scalars_and_subtables():
    data = {"flow": {"title": 'a "quoted" b', "random_seed": 7},
            "search": {"pool_size": 2500, "trust_region": {"enabled": True, "band": 0.1},
                       "fallbacks": []}}
    assert tomllib.loads(ae._toml_dumps(data)) == data


# --------------------------------------------------------------------------- #
# 4. the plan
# --------------------------------------------------------------------------- #
@needs_repo
def test_plan_orders_stages_and_puts_the_prereg_before_any_master_call(tmp_path):
    cfg = _cfg(tmp_path, [])
    steps = ae.plan_cell(cfg, ae.Target(pair="N1_N2", feed=109), champion="data/models/s1g",
                         arm="s1h", split="S1h", parent_split="S1g")
    names = [s.name for s in steps]
    assert names[:3] == ["precheck", "prereg", "arm_scripts"]
    first_master = min(i for i, s in enumerate(steps) if s.master_calls)
    assert names.index("prereg") < first_master
    assert [s.stage for s in steps] == sorted(s.stage for s in steps)  # stages monotone
    assert sum(s.master_calls for s in steps) == 108                   # 8 probe + 100 open


@needs_repo
def test_plan_budget_and_fleet_allocation(tmp_path):
    cfg = _cfg(tmp_path, [])
    steps = ae.plan_cell(cfg, ae.Target(pair="N1_N2", feed=109), champion="data/models/s1g",
                         arm="s1h", split="S1h", parent_split="S1g")
    by = {s.name: s for s in steps}
    # every MASTER call is on the campaign box; training is on 238; nothing else.
    assert {s.where for s in steps if s.master_calls} == {"199"}
    assert by["train_launch"].where == "238"
    assert "--split S1h" in " ".join(by["train_launch"].argv)
    assert by["mesh_recompute"].where == "199"
    for s in steps:
        ae.guard_argv(cfg, s.argv)                 # no step may address 181/198
        ae.guard_argv(cfg, s.poll_argv)


@needs_repo
def test_map_update_is_skipped_when_the_champion_did_not_change(tmp_path):
    cfg = _cfg(tmp_path, [])
    steps = ae.plan_cell(cfg, ae.Target(pair="N1_N2", feed=109), champion="data/models/s1g",
                         arm="s1h", split="S1h", parent_split="S1g")
    mesh = [s for s in steps if s.stage == "5-map"]
    assert mesh and all(s.skip_if == "gate_failed" for s in mesh)


def test_new_assembly_target_plans_a_precheck_then_a_gate(tmp_path):
    cfg = _cfg(tmp_path, [])
    steps = ae.plan_cell(cfg, ae.Target(new_assembly="config/newfa.yaml"),
                         champion="data/models/s1g")
    assert len(steps) == 1
    assert steps[0].gate == "new_assembly" and steps[0].master_calls == 0


# --------------------------------------------------------------------------- #
# 5. ordering
# --------------------------------------------------------------------------- #
def test_ordering_is_transfer_aware_then_db_frontier(tmp_path):
    cfg = _cfg(tmp_path, [])
    t109 = ae.Target(pair="N1_N2", feed=109)
    t117 = ae.Target(pair="K5_K6", feed=117)
    t121 = ae.Target(pair="L3_L4", feed=121)
    tnew = ae.Target(new_assembly="config/newfa.yaml")
    # N1_N2/f113 is opened -> its same-pair neighbour is nearest and must lead.
    got = ae.order_targets(cfg, [t121, t117, tnew, t109], opened=["N1_N2_f113"])
    assert got[0].cell_id == "N1_N2_f109"
    assert got[-1].is_new_assembly           # a halting target must never block cells
    # with nothing opened, the DB frontier decides (L3_L4 1.5215 < K5_K6 1.5229).
    got2 = ae.order_targets(cfg, [t117, t121], opened=[])
    assert [t.pair for t in got2] == ["L3_L4", "K5_K6"]


@needs_repo
def test_opened_cells_read_from_the_store_include_the_hand_run_f113():
    cfg = ae.AutoengConfig(root=ROOT)
    opened = ae.opened_cells_from_store(cfg)
    assert "N1_N2_f113" in opened          # 41 feasible cores, opened by hand 2026-08-16
    assert "N1_N2_f109" not in opened      # zero feasible — the cell autoeng targets


# --------------------------------------------------------------------------- #
# 6. state log — append-only, kill -9 resumable, every MASTER call accounted
# --------------------------------------------------------------------------- #
def test_state_log_is_append_only_and_survives_a_torn_tail(tmp_path):
    p = tmp_path / "state.jsonl"
    s = ae.StateLog(p)
    s.append("step_start", cell="c", step="a")
    s.append("step_done", cell="c", step="a", master_calls=8)
    s.append("step_start", cell="c", step="b")
    before = p.read_text(encoding="utf-8")
    # simulate kill -9 mid-write: a truncated final line.
    p.write_text(before + '{"seq": 3, "kind": "step_do', encoding="utf-8")
    s2 = ae.StateLog(p)
    assert len(s2.events) == 3
    assert s2.step_status("c", "a") == "done"
    assert s2.step_status("c", "b") == "started"          # NOT done -> will be re-run
    assert s2.master_calls() == 8                          # ledger survived
    # and appending after the torn tail never rewrites history
    s2.append("step_done", cell="c", step="b", master_calls=100)
    assert p.read_text(encoding="utf-8").startswith(before)
    assert ae.StateLog(p).master_calls() == 108


def test_completed_steps_are_skipped_on_resume(tmp_path):
    cfg = _cfg(tmp_path, [])
    runner = RecordingRunner()
    eng = ae.AutoEngineer(cfg, runner=runner, log=lambda s: None)
    step = ae.Step("merge", "3-harvest", "local", "x", argv=("echo", "hi"))
    t = ae.Target(pair="N1_N2", feed=109)
    assert eng.execute_step(t, step)
    assert len(runner.calls) == 1
    eng2 = ae.AutoEngineer(cfg, runner=runner, log=lambda s: None)
    assert eng2.execute_step(t, step)
    assert len(runner.calls) == 1                    # not re-run


def test_failed_step_is_recorded_and_stops_the_cell(tmp_path):
    cfg = _cfg(tmp_path, [])
    runner = RecordingRunner(rcs={"boom": 3})
    eng = ae.AutoEngineer(cfg, runner=runner, log=lambda s: None)
    step = ae.Step("merge", "3-harvest", "local", "x", argv=("boom",))
    assert eng.execute_step(ae.Target(pair="N1_N2", feed=109), step) is False
    assert eng.state.step_status("N1_N2_f109", "merge") == "failed"
    assert eng.state.master_calls() == 0             # a failed step charges nothing


def test_master_budget_cap_raises_the_gate(tmp_path):
    cfg = _cfg(tmp_path, [], master_budget_total=50)
    eng = ae.AutoEngineer(cfg, runner=RecordingRunner(), log=lambda s: None)
    step = ae.Step("open_launch", "2-open", "199", "x", argv=("true",), master_calls=100)
    assert eng.execute_step(ae.Target(pair="N1_N2", feed=109), step) is False
    kinds = [e["kind"] for e in eng.state.events]
    assert "gate_pause" in kinds
    assert eng.state.master_calls() == 0             # never spent


# --------------------------------------------------------------------------- #
# 7. human gates
# --------------------------------------------------------------------------- #
def test_retrain_gate_fail_pauses_and_records_the_registered_fallback(tmp_path):
    cfg = _cfg(tmp_path, [])
    eng = ae.AutoEngineer(cfg, runner=RecordingRunner(), log=lambda s: None)
    gp = tmp_path / "gate_s1h_checkonly.json"
    gp.write_text(json.dumps({"pass": False}), encoding="utf-8")
    cfg.p = lambda rel, _g=gp: _g if "gate_" in str(rel) else ae.AutoengConfig.p(cfg, rel)
    eng.ctx["arm"] = "s1h"
    step = ae.Step("gate_check", "4-retrain", "local", "x", argv=("true",),
                   gate="retrain_promote_fail")
    assert eng.execute_step(ae.Target(pair="N1_N2", feed=109), step) is False
    assert eng.ctx["gate_failed"] is True
    pause = [e for e in eng.state.events if e["kind"] == "gate_pause"]
    assert pause and "폴백" in pause[0]["detail"]


def test_retrain_gate_pass_promotes(tmp_path):
    cfg = _cfg(tmp_path, [])
    eng = ae.AutoEngineer(cfg, runner=RecordingRunner(), log=lambda s: None)
    gp = tmp_path / "gate_s1h_checkonly.json"
    gp.write_text(json.dumps({"pass": True}), encoding="utf-8")
    cfg.p = lambda rel, _g=gp: _g if "gate_" in str(rel) else ae.AutoengConfig.p(cfg, rel)
    eng.ctx["arm"] = "s1h"
    step = ae.Step("gate_check", "4-retrain", "local", "x", argv=("true",),
                   gate="retrain_promote_fail")
    assert eng.execute_step(ae.Target(pair="N1_N2", feed=109), step) is True
    assert eng.ctx.get("gate_failed") is False


def test_skip_if_suppresses_the_step(tmp_path):
    cfg = _cfg(tmp_path, [])
    runner = RecordingRunner()
    eng = ae.AutoEngineer(cfg, runner=runner, log=lambda s: None)
    eng.ctx["gate_failed"] = True
    step = ae.Step("mesh_recompute", "5-map", "199", "x", argv=("true",),
                   skip_if="gate_failed")
    assert eng.execute_step(ae.Target(pair="N1_N2", feed=109), step)
    assert runner.calls == []
    assert eng.state.step_status("N1_N2_f109", "mesh_recompute") == "skipped"


# --------------------------------------------------------------------------- #
# 8. dry run executes and writes NOTHING
# --------------------------------------------------------------------------- #
@needs_repo
def test_dry_run_never_calls_the_runner_and_writes_no_cell_files(tmp_path):
    cfg = ae.load_autoeng_config(ROOT / "autoeng.toml")
    cfg.run_id = "t_dryrun_" + tmp_path.name
    runner = RecordingRunner()
    eng = ae.AutoEngineer(cfg, dry_run=True, runner=runner, log=lambda s: None)
    plan = eng.plan_only()
    assert runner.calls == []
    assert not cfg.run_dir.exists()
    assert not any(s.name == "open_launch" and s.where != "199" for _, ss in plan for s in ss)
    assert sum(len(ss) for _, ss in plan) == 30 * len(plan)


def test_dry_run_refuses_to_execute_a_step(tmp_path):
    cfg = _cfg(tmp_path, [])
    runner = RecordingRunner()
    eng = ae.AutoEngineer(cfg, dry_run=True, runner=runner, log=lambda s: None)
    step = ae.Step("merge", "3-harvest", "local", "x", argv=("echo", "hi"))
    with pytest.raises(RuntimeError, match="dry-run"):
        eng.execute_step(ae.Target(pair="N1_N2", feed=109), step)
    assert runner.calls == []


@needs_repo
def test_dry_run_plan_matches_the_shipped_targets():
    cfg = ae.load_autoeng_config(ROOT / "autoeng.toml")
    eng = ae.AutoEngineer(cfg, dry_run=True, log=lambda s: None)
    plan = eng.plan_only()
    assert [t.cell_id for t, _ in plan][0] == "N1_N2_f109"      # transfer-nearest first
    total = sum(s.master_calls for _, ss in plan for s in ss)
    assert total == 108 * len(cfg.targets)
    assert total <= cfg.master_budget_total


# --------------------------------------------------------------------------- #
# 9. generated scripts carry the launcher's proven gates
# --------------------------------------------------------------------------- #
def test_launch_script_carries_busy_hash_and_precondition_gates(tmp_path):
    cfg = _cfg(tmp_path, [])
    t = ae.Target(pair="N1_N2", feed=109)
    ps1 = ae.render_launch_ps1(cfg, t, "d.inp", "tag", "abc123", "data/models/s1g",
                               fresh_run_dir="runs/tag")
    assert "REFUSED: box busy" in ps1
    assert "ABC123" in ps1 and "deck sha256 mismatch" in ps1
    assert "ensemble.json" in ps1 and "records.parquet" in ps1
    assert "Invoke-CimMethod" in ps1 and "schtasks" not in ps1
    assert "Remove-Item (Join-Path $k 'runs/tag')" in ps1     # fresh run dir


@needs_repo
def test_probe_script_uses_its_own_state_dir(tmp_path):
    cfg = _cfg(tmp_path, [])
    t = ae.Target(pair="N1_N2", feed=109, probe_budget=8)
    src = ae.render_probe_script(cfg, t, {"e_core": 5.4}, "data/models/s1g", "d.inp")
    compile(src, "probe.py", "exec")                       # it must at least parse
    assert "_phase_blind_probe" in src
    assert "probe_size = 8" in src.replace("drv.curr.", "")
    assert "autoeng_probe/N1_N2_f109" in src
    # the real curriculum state must never be the driver's state_dir
    assert 'state_dir=STATE' in src
    body = src.split('"""', 2)[-1]             # ignore the docstring
    assert "data/curriculum" not in body


# --------------------------------------------------------------------------- #
# 7. the objective axis (F_xy switch, design fxy_switch_design_20260829 §3.5.5)
#
# The load-bearing property is NEGATIVE: moving the readouts onto a resolvable
# axis must not have moved a single number on the F_r path.  Two tests hold that
# (`..._marks_are_the_same_numbers...`, `..._byte_identical...`); the rest check
# that the F_xy path gates, excludes and refuses the way the design says.
# --------------------------------------------------------------------------- #
import readout_axis as RA  # noqa: E402


def test_default_axis_is_the_literals_the_readouts_used_to_inline():
    a = RA.resolve_axis()
    assert (a.key, a.label, a.limit, a.lam) == ("f_r", "F_r", 1.55, 1000.0)
    assert a is RA.F_R_AXIS
    assert RA.resolve_axis(objective="min_fr_max_cycle").limit == 1.55
    # every non-fxy objective keeps the F_r axis, including flat_power, which
    # carries an F_xy GATE but is not an F_xy objective
    for obj in ("target_cycle", "max_cycle_min_fr", "min_fuel_cost",
                "fr_boundary", "flat_power"):
        assert RA.resolve_axis(objective=obj).key == "f_r"
    b = RA.resolve_axis(objective="min_fxy")
    assert (b.key, b.label, b.limit) == ("f_xy", "F_xy", 1.65)
    assert b.gate == "F_xy <= 1.65"


def test_axis_limits_agree_with_lpopt():
    RA._check_against_lpopt()          # raises if the readouts would gate elsewhere


def test_axis_resolves_from_a_deck(tmp_path):
    p = tmp_path / "d.inp"
    p.write_text('[acquisition]\nobjective = "min_fxy"\nf_xy_limit = 1.60\n'
                 'minfxy_lambda = 750.0\n', encoding="utf-8")
    a = RA.resolve_axis(deck=p)
    assert a.key == "f_xy" and a.limit == 1.60 and a.lam == 750.0
    q = tmp_path / "e.inp"
    q.write_text('[acquisition]\nobjective = "min_fr_max_cycle"\n', encoding="utf-8")
    assert RA.resolve_axis(deck=q).key == "f_r"
    r = tmp_path / "f.inp"
    r.write_text("[acquisition]\nbudget = 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names no objective"):
        RA.resolve_axis(deck=r)


def test_unlabelled_rows_are_excluded_and_counted_never_silently_dropped():
    import numpy as np
    import pandas as pd

    axis = RA.resolve_axis(objective="min_fxy")
    df = pd.DataFrame({"cyclen": [600.0, 610.0, 620.0, 630.0],
                       "f_r": [1.50, 1.52, 1.48, 1.49],
                       "f_xy": [1.60, np.nan, 1.70, np.nan]})
    kept, n_unlab = RA.split_labelled(df, axis)
    assert len(kept) == 2 and n_unlab == 2
    note = RA.unlabelled_note(axis, n_unlab, len(df), what="cores")
    assert "2/4" in note and "EXCLUDED" in note
    # the F_r axis drops nothing here and prints NOTHING - that silence is what
    # keeps every min_fr readout byte-identical
    kept_fr, n_fr = RA.split_labelled(df, RA.F_R_AXIS)
    assert len(kept_fr) == 4 and n_fr == 0
    assert RA.unlabelled_note(RA.F_R_AXIS, 0, 4) == ""
    # an ABSENT column is unlabelled, not a KeyError
    _, n_missing = RA.split_labelled(df.drop(columns=["f_xy"]), axis)
    assert n_missing == 4


def test_lambda_objective_can_disagree_with_the_axis_floor():
    import pandas as pd

    axis = RA.resolve_axis(objective="min_fxy")
    # lambda = 1000 EFPD per unit F_xy, so +40 EFPD buys at most +0.040 F_xy.
    # A 0.060 penalty is too dear and the axis FLOOR wins...
    df = pd.DataFrame({"record_id": ["floor", "lam"], "cyclen": [600.0, 640.0],
                       "f_xy": [1.600, 1.660]})
    best, n_unlab = RA.best_by_lambda(df, axis)
    assert n_unlab == 0 and best.record_id == "floor"
    # ...but a 0.010 penalty is not, and the longer core wins even though it is
    # NOT the flattest - the exact reversal the registered rule exists to catch
    df.loc[1, "f_xy"] = 1.610
    best, _ = RA.best_by_lambda(df, axis)
    assert best.record_id == "lam"
    # nothing labelled -> no frontier claim, and NOT a fallback to F_r
    empty = df.assign(f_xy=[float("nan")] * 2)
    best, n_unlab = RA.best_by_lambda(empty, axis)
    assert best is None and n_unlab == 2


def test_autoeng_config_validates_the_axis(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text('[autoeng]\nobjective_axis = "f_q"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="objective_axis"):
        ae.load_autoeng_config(p)
    p.write_text('[autoeng]\nobjective_axis = "f_xy"\n', encoding="utf-8")
    assert ae.load_autoeng_config(p).axis.key == "f_xy"


def test_shipped_config_is_on_the_f_r_axis_and_documents_the_key():
    cfg = ae.load_autoeng_config(ROOT / "autoeng.toml")
    assert cfg.objective_axis == "f_r"
    assert cfg.axis is RA.F_R_AXIS          # the SHARED default object
    toml_text = (ROOT / "autoeng.toml").read_text(encoding="utf-8")
    assert "objective_axis" in toml_text and "f_xy" in toml_text


@needs_repo
def test_f_r_axis_marks_are_the_same_numbers_under_the_new_names(tmp_path):
    """The axis marks must EQUAL the F_r marks on the default axis."""
    cfg = _cfg(tmp_path, [])
    marks = ae.measure_marks(cfg, ae.Target(pair="N1_N2", feed=109))
    assert marks["axis"] == "f_r" and marks["axis_limit"] == 1.55
    assert marks["store_axis_floor"] == marks["store_f_r_floor"]
    assert marks["store_axis_p10"] == marks["store_f_r_p10"]
    assert marks["elite_parent_axis_min"] == marks["elite_parent_f_r_min"]
    assert marks["store_axis_unlabelled"] == 0


@needs_repo
def test_f_r_prereg_header_is_byte_identical_to_the_pre_switch_text(tmp_path):
    """Golden lines: the axis substitution must render the SAME bytes on F_r."""
    cfg = _cfg(tmp_path, [])
    target = ae.Target(pair="N1_N2", feed=109)
    marks = ae.measure_marks(cfg, target)
    text, _ = ae.build_deck(cfg, target, marks, "data/models/s1g")
    header = text.split("\n[flow]")[0]
    assert ("#   PURE F_r MINIMISATION UNDER SAFETY GATES ONLY.  No cycle-length "
            "target and") in header
    assert (f"#      F_r floor {marks['store_f_r_floor']:.4f}   "
            f"p10 {marks['store_f_r_p10']:.4f}") in header
    assert f"#      best parent F_r {marks['elite_parent_f_r_min']:.4f}." in header
    assert ("#   PRIMARY   a MASTER-verified constraint-feasible core "
            "(F_r <= 1.55, CBC <= 1600," in header)
    assert "#             F_q <= 2.41, |AO| <= 0.30) at this cell." in header
    assert ("#   [constraints] DROPPED: inert on min_fr_max_cycle "
            "(fpcamp_N1N2_f113_results_20260816.md, defect 2)." in header)
    assert "PURE F_r min, NO cycle band" in text
    # and NOTHING from the F_xy path leaked in
    for token in ("F_xy", "AXIS:", "UNLABELLED"):
        assert token not in header


@needs_repo
def test_f_xy_axis_refuses_a_min_fr_parent_deck(tmp_path):
    cfg = _cfg(tmp_path, [], objective_axis="f_xy")
    target = ae.Target(pair="N1_N2", feed=109)
    marks = ae.measure_marks(cfg, target)
    with pytest.raises(ValueError, match="needs a parent deck with objective"):
        ae.build_deck(cfg, target, marks, "data/models/s1g")


@needs_repo
def test_f_xy_axis_marks_exclude_the_unmeasured_rows_and_say_so(tmp_path):
    cfg = _cfg(tmp_path, [], objective_axis="f_xy")
    marks = ae.measure_marks(cfg, ae.Target(pair="N1_N2", feed=109))
    assert marks["axis"] == "f_xy" and marks["axis_limit"] == 1.65
    assert (marks["store_axis_labelled"] + marks["store_axis_unlabelled"]
            == marks["store_converged"])
    # F_r is KEPT as a constraint mark (design 3.5.2), not replaced
    assert marks["store_f_r_floor"] is not None


@needs_repo
def test_f_xy_opened_gate_can_only_shrink_the_opened_set(tmp_path):
    """UNKNOWN F_xy is not OPENED - the delivery side of the 3.5.4 split."""
    fr = ae.opened_cells_from_store(_cfg(tmp_path, []))
    fxy = ae.opened_cells_from_store(_cfg(tmp_path, [], objective_axis="f_xy"))
    assert set(fxy) <= set(fr)


def test_anchor_readout_h3_follows_the_deck_axis():
    import anchor_readout as AR

    assert AR.tiers_for(RA.F_R_AXIS) == AR._TIERS_FR
    shifted = AR.tiers_for(RA.resolve_axis(objective="min_fxy"))
    assert [t[1] for t in shifted] == [1.65, 1.75, 1.90]   # ladder keeps its headroom
    assert [t[2] for t in shifted] == [1600.0, 1800.0, 2200.0]
    # the F_r ladder is untouched, so the existing anchor_readout.log stays valid
    assert AR.joint_tier(1.54, 1500.0) == "tier1"
    assert AR.joint_tier(1.60, 1500.0) == "tier2"
    assert AR.joint_tier(float("nan"), 1500.0) == ""


def test_producers_refuse_the_fxy_axis_rather_than_relabel_f_r(monkeypatch):
    """scoping_mesh / mesh_multitype_readout are F_r producers - see their docs.

    Refusing is the deliverable here: both write or read F_r-named columns, so a
    silent switch would put F_r numbers under an F_xy heading, which is the one
    claim design 20260829 sec. 3.6 singles out as forbidden.
    """
    import mesh_multitype_readout as MMR
    import scoping_mesh as SM

    monkeypatch.setattr(sys, "argv", ["scoping_mesh", "--objective", "min_fxy"])
    with pytest.raises(SystemExit, match="F_r producer"):
        SM.main()
    with pytest.raises(SystemExit, match="no F_xy column"):
        MMR.main(["--objective", "min_fxy"])
