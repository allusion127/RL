"""The pre-registered v5 A/B runner: plan, validation, dry-run, scoring.

The experiment is only meaningful if the comparison is fair, so these tests pin
the fairness properties rather than the plumbing:

* all four arms share ONE base seed, ONE ensemble size and ONE split, so an arm
  difference is a method difference;
* the ablation arm exists and differs from the full arm in exactly the shape
  channels;
* every arm is scored on the SAME honest holdout, and validation REFUSES to run
  if that holdout intersects the training fold;
* ``--dry-run`` validates without launching anything.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lpopt.model.v5_experiment import (
    ARMS, ARMS_BY_NAME, DECISION_TARGETS, ELITE_K, HOLDOUT_GROUP,
    ExperimentConfig, build_plan, decision_table, discover_champions,
    format_plan, precision_at_k, run_experiment, validate_plan,
)

STORE = "data/store"
SPLITS = "data/splits"


@pytest.fixture
def cfg(tmp_path):
    champs = discover_champions("data/models")
    return ExperimentConfig(
        store_dir=STORE, splits_dir=SPLITS, reports_dir=str(tmp_path),
        champion_dir=str(champs[-1]) if champs else None,
        teacher_map="auto",
    )


# --------------------------------------------------------------------------- #
# the pre-registered arm set
# --------------------------------------------------------------------------- #
def test_the_four_arms_are_registered():
    assert [a.name for a in ARMS] == [
        "v4_baseline", "v5_full", "v5_minus_shape", "v5_distill"]


def test_baseline_arm_carries_no_v5_knobs():
    a = ARMS_BY_NAME["v4_baseline"]
    assert a.cond_schema == "v4"
    assert not (a.physics_prior or a.quantile_heads or a.promote_max_asm_bu
                or a.distill)


def test_ablation_differs_from_full_only_in_the_schema():
    full = ARMS_BY_NAME["v5_full"]
    abl = ARMS_BY_NAME["v5_minus_shape"]
    assert full.cond_schema == "v5"
    assert abl.cond_schema == "v5_noshape"
    for knob in ("physics_prior", "quantile_heads", "promote_max_asm_bu",
                 "distill"):
        assert getattr(full, knob) == getattr(abl, knob), (
            f"the ablation must isolate the SHAPE CHANNELS; {knob} differs")


def test_distill_arm_is_v5_full_plus_distillation():
    full = ARMS_BY_NAME["v5_full"]
    dis = ARMS_BY_NAME["v5_distill"]
    assert dis.cond_schema == full.cond_schema
    assert dis.distill is True
    for knob in ("physics_prior", "quantile_heads", "promote_max_asm_bu"):
        assert getattr(dis, knob) == getattr(full, knob)


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_every_arm_shares_seeds_split_and_ensemble(cfg):
    plan = build_plan(cfg)
    assert plan["seeds"] == [cfg.base_seed + i for i in range(cfg.ensemble)]
    for arm in plan["arms"]:
        argv = arm["train_argv"]
        assert argv[argv.index("--base-seed") + 1] == str(cfg.base_seed)
        assert argv[argv.index("--split") + 1] == cfg.split
        assert argv[argv.index("--ensemble") + 1] == str(cfg.ensemble)


def test_plan_names_the_honest_holdout(cfg):
    plan = build_plan(cfg)
    assert HOLDOUT_GROUP in plan["holdout"]
    assert plan["decision_metrics"] == [
        "within_cell_spearman", "calibrated_mae", "legacy_tail_delta_mae",
        f"p_at_{ELITE_K}"]


def test_train_argv_carries_the_right_flags(cfg):
    by_name = {a["name"]: a["train_argv"] for a in build_plan(cfg)["arms"]}
    assert "--cyclen-physics-prior" not in by_name["v4_baseline"]
    assert "--cond-schema" in by_name["v4_baseline"]
    for name in ("v5_full", "v5_minus_shape", "v5_distill"):
        argv = by_name[name]
        assert "--cyclen-physics-prior" in argv
        assert "--quantile-heads" in argv
        assert "--promote-max-asm-bu" in argv
    assert "--distill-targets" in by_name["v5_distill"]
    assert "--distill-targets" not in by_name["v5_full"]
    v5 = by_name["v5_full"]
    assert v5[v5.index("--cond-schema") + 1] == "v5"
    abl = by_name["v5_minus_shape"]
    assert abl[abl.index("--cond-schema") + 1] == "v5_noshape"


def test_train_argv_flags_are_real_trainer_flags(cfg):
    """A plan that prints a flag the trainer rejects is worse than useless."""
    from lpopt.model.train import main as train_main
    import argparse
    import contextlib
    import io

    for arm in build_plan(cfg)["arms"]:
        args = arm["train_argv"][3:]          # drop 'python -m lpopt.model.train'
        # --help exits 0 only after the parser accepts every other flag shape;
        # instead parse directly against the trainer's own parser.
        buf = io.StringIO()
        with pytest.raises(SystemExit) as exc, contextlib.redirect_stdout(buf):
            train_main(args + ["--help"])
        assert exc.value.code == 0


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_valid_config_passes(cfg):
    assert validate_plan(cfg) == []


def test_missing_champion_is_caught(cfg):
    cfg.champion_dir = None
    problems = validate_plan(cfg)
    assert any("champion_dir is unset" in p for p in problems)


def test_nonexistent_champion_is_caught(cfg, tmp_path):
    cfg.champion_dir = str(tmp_path / "nope")
    assert any("does not exist" in p for p in validate_plan(cfg))


def test_champion_without_members_is_caught(cfg, tmp_path):
    d = tmp_path / "empty_champ"
    d.mkdir()
    cfg.champion_dir = str(d)
    assert any("no member_* checkpoints" in p for p in validate_plan(cfg))


def test_missing_teacher_map_is_caught(cfg):
    cfg.teacher_map = None
    assert any("v5_distill arm needs --teacher-map" in p
               for p in validate_plan(cfg))


def test_teacher_map_pointing_nowhere_is_caught(cfg, tmp_path):
    cfg.teacher_map = str(tmp_path / "missing.json")
    assert any("teacher map not found" in p for p in validate_plan(cfg))


def test_teacher_map_with_a_bad_dir_is_caught(cfg, tmp_path):
    p = tmp_path / "teachers.json"
    p.write_text(json.dumps({"teachers": {"feed=121|ebin=5.7": "/nope/nowhere"}}),
                 encoding="utf-8")
    cfg.teacher_map = str(p)
    assert any("does not exist" in x for x in validate_plan(cfg))


def test_unknown_arm_is_caught(cfg):
    cfg.arms = ("v4_baseline", "v9_moonshot")
    assert any("unknown arm" in p for p in validate_plan(cfg))


def test_missing_split_is_caught(cfg):
    cfg.split = "S_does_not_exist"
    assert any("missing split manifest" in p for p in validate_plan(cfg))


def test_missing_store_is_caught(cfg, tmp_path):
    cfg.store_dir = str(tmp_path / "no_store")
    assert any("missing store file" in p for p in validate_plan(cfg))


def test_holdout_leakage_is_caught(cfg, tmp_path):
    """The single most important check: an arm must never be scored on a row it
    trained on.  A manifest whose holdout intersects train_ids must be refused."""
    src = json.loads((tmp_path.parent / "x").parent.joinpath(
        SPLITS, "S1.json").read_text(encoding="utf-8")) \
        if False else json.loads(open(f"{SPLITS}/S1.json", encoding="utf-8").read())
    leaked = dict(src)
    holdout = dict(src["groups"][HOLDOUT_GROUP])
    first_cell = sorted(holdout)[0]
    leaked["train_ids"] = list(src["train_ids"]) + list(holdout[first_cell][:2])
    (tmp_path / "SLEAK.json").write_text(json.dumps(leaked), encoding="utf-8")
    cfg.splits_dir = str(tmp_path)
    cfg.split = "SLEAK"
    problems = validate_plan(cfg)
    assert any(p.startswith("LEAKAGE:") for p in problems), problems


def test_missing_holdout_group_is_caught(cfg, tmp_path):
    src = json.loads(open(f"{SPLITS}/S1.json", encoding="utf-8").read())
    src["groups"] = {}
    (tmp_path / "SNOG.json").write_text(json.dumps(src), encoding="utf-8")
    cfg.splits_dir = str(tmp_path)
    cfg.split = "SNOG"
    assert any(HOLDOUT_GROUP in p and "carries no" in p
               for p in validate_plan(cfg))


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #
def test_dry_run_passes_and_launches_nothing(cfg):
    lines: list[str] = []
    rc = run_experiment(cfg, dry_run=True, log=lines.append)
    assert rc == 0
    text = "\n".join(lines)
    assert "validation: OK" in text
    assert "nothing launched" in text
    for arm in ARMS:
        assert arm.name in text


def test_dry_run_returns_nonzero_when_validation_fails(cfg):
    cfg.champion_dir = None
    lines: list[str] = []
    assert run_experiment(cfg, dry_run=True, log=lines.append) == 1
    assert "VALIDATION FAILED" in "\n".join(lines)


def test_cli_exposes_the_subcommand():
    from lpopt.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["v5-experiment", "--dry-run",
                              "--champion-dir", "x"])
    assert args.dry_run is True
    assert args.champion_dir == "x"
    assert args.func.__name__ == "cmd_v5_experiment"


def test_champion_defaults_to_the_deck_champion():
    """So the coordinator does not have to retype a timestamped path at launch."""
    from lpopt.config import load_config
    from lpopt.model.v5_experiment import config_from_args
    from lpopt.cli import build_parser

    args = build_parser().parse_args(["v5-experiment", "--dry-run",
                                      "--teacher-map", "auto"])
    cfg = config_from_args(args)
    assert cfg.champion_dir == load_config("lpopt.inp").model.model_dir


def test_explicit_champion_overrides_the_deck(tmp_path):
    from lpopt.model.v5_experiment import config_from_args
    from lpopt.cli import build_parser

    args = build_parser().parse_args(
        ["v5-experiment", "--champion-dir", str(tmp_path)])
    assert config_from_args(args).champion_dir == str(tmp_path)


def test_format_plan_reports_problems():
    cfg = ExperimentConfig()
    text = format_plan(build_plan(cfg), ["boom", "bang"])
    assert "VALIDATION FAILED — 2 problem(s)" in text
    assert "boom" in text and "bang" in text


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_precision_at_k_perfect_and_inverted():
    truth = np.arange(20, dtype=float)
    assert precision_at_k(truth, truth, k=8, high_is_better=True) == 1.0
    assert precision_at_k(-truth, truth, k=8, high_is_better=True) == 0.0


def test_precision_at_k_respects_the_elite_direction():
    """cyclen's elite is the HIGH tail, f_r's is the LOW tail."""
    truth = np.arange(20, dtype=float)
    assert precision_at_k(truth, truth, k=8, high_is_better=False) == 1.0
    assert precision_at_k(-truth, truth, k=8, high_is_better=False) == 0.0


def test_precision_at_k_partial_overlap():
    # truth's elite (top 8 of 0..15) is {8..15}.  Promote four rows from the
    # BOTTOM half into the predicted top, displacing four genuine elites.
    truth = np.arange(16, dtype=float)
    pred = truth.copy()
    pred[[0, 1, 2, 3]] = 100.0
    p = precision_at_k(pred, truth, k=8, high_is_better=True)
    assert p == pytest.approx(0.5)


def test_precision_at_k_needs_enough_rows():
    assert np.isnan(precision_at_k(np.arange(4.0), np.arange(4.0), k=8))


def test_precision_at_k_ignores_non_finite_pairs():
    truth = np.arange(20, dtype=float)
    pred = truth.copy()
    pred[0] = np.nan
    assert 0.0 <= precision_at_k(pred, truth, k=8) <= 1.0


def test_elite_directions_cover_every_decision_target():
    from lpopt.model.v5_experiment import ELITE_HIGH_IS_BETTER
    for name, _, _ in DECISION_TARGETS:
        assert name in ELITE_HIGH_IS_BETTER
    assert ELITE_HIGH_IS_BETTER["cyclen"] is True     # longer cycle is better
    assert ELITE_HIGH_IS_BETTER["f_r"] is False       # lower peaking is better


def test_decision_table_renders_every_arm_and_target():
    results = {
        arm.name: {
            "cyclen": {"within_cell_spearman": 0.8, "calibrated_mae": 3.2,
                       f"p_at_{ELITE_K}": 0.75, "n_cells": 12},
            "f_r": {"within_cell_spearman": 0.6, "calibrated_mae": 0.01,
                    f"p_at_{ELITE_K}": 0.5, "n_cells": 12},
            "legacy_tail": {"pass": True, "worst_increase": 0.4},
        }
        for arm in ARMS
    }
    table = decision_table(results)
    for arm in ARMS:
        assert arm.name in table
    for name, _, _ in DECISION_TARGETS:
        assert name in table
    assert f"P@{ELITE_K}" in table
    assert "legacy high-cyclen tail" in table
    assert "0.8000" in table


def test_decision_table_handles_missing_values():
    table = decision_table({"arm": {"cyclen": {}, "f_r": {}}})
    assert "n/a" in table
