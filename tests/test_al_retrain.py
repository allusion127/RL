"""Champion-faithful AL retrain harness (forensic 20260722).

Pins that the harness recovers the champion's recipe from its checkpoint and
composes a retrain invocation that (a) reproduces every recipe switch the plain
`remote train` dropped (width/distill/prior/quantile/promote) and (b) folds in the
model-backlog boundary-F_r improvements — so retrain #3 no longer regresses the
mid-band the honest gate scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.model.al_retrain import (
    champion_recipe, recipe_to_train_args, remote_invocation, plan_al_retrain,
)


def _fake_champion(tmp_path: Path, **tc_over) -> Path:
    d = tmp_path / "champ"
    (d / "member_A").mkdir(parents=True)
    tc = {
        "epochs": 150, "width": 160, "n_blocks": 6, "head_hidden": 256,
        "cyclen_physics_prior": True, "quantile_heads": True, "quantile_weight": 0.2,
        "quantile_targets": ["f_r", "cyclen"], "distill_weight": 0.3,
        "distill_min_match_frac": 0.5, "distill_targets": "data/models/_v5_distill_soft.npz",
        "promote_max_asm_bu": True, "auto_fit_cell_calibration": True,
        "cyclen_rank_weight": 0.1, "num_workers": 8, "split": "S1",
    }
    tc.update(tc_over)
    meta = {"train_config": tc, "cond_schema": "v5", "seed": 20260716,
            "net_config": {"width": tc["width"], "n_blocks": tc["n_blocks"],
                           "head_hidden": tc["head_hidden"], "n_targets": 8},
            "target_names": ["f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                             "discharge_burnup", "max_pin_burnup", "max_assembly_burnup"]}
    (d / "member_A" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "ensemble.json").write_text(json.dumps(
        {"members": ["member_A"], "n_members": 5, "split": "S1",
         "base_seed": 20260716}), encoding="utf-8")
    return d


def test_recipe_recovers_v5_distill_w160(tmp_path):
    r = champion_recipe(_fake_champion(tmp_path))
    assert r["width"] == 160 and r["n_blocks"] == 6 and r["n_members"] == 5
    assert r["split"] == "S1"
    assert r["cyclen_physics_prior"] and r["quantile_heads"]
    assert r["quantile_weight"] == 0.2 and r["distill_weight"] == 0.3
    assert r["promote_max_asm_bu"] and r["auto_fit_cell_calibration"]


def test_train_args_reproduce_recipe_and_add_fr_rank(tmp_path):
    r = champion_recipe(_fake_champion(tmp_path))
    args = recipe_to_train_args(r, distill_cache="data/models/_v5_distill_soft.npz")
    s = " ".join(args)
    # champion recipe reproduced
    assert "--width 160" in s and "--n-blocks 6" in s
    assert "--ensemble 5" in s and "--split S1" in s
    assert "--cyclen-physics-prior" in s
    assert "--quantile-heads" in s and "--quantile-weight 0.2" in s
    assert "--promote-max-asm-bu" in s
    assert "--distill-targets data/models/_v5_distill_soft.npz" in s
    assert "--distill-weight 0.3" in s and "--distill-min-match-frac 0.5" in s
    # [1] boundary improvement folded in; map-consistency off by default
    assert "--f-r-rank-weight 0.1" in s
    assert "--map-fr-consistency-weight" not in s
    # out-dir left to the remote wrapper
    assert "--out-dir" not in s


def test_train_args_respect_switched_off_recipe(tmp_path):
    # a champion trained WITHOUT distill/prior/quantile/promote + auto-cal OFF.
    d = _fake_champion(tmp_path, distill_weight=0.0, cyclen_physics_prior=False,
                       quantile_heads=False, promote_max_asm_bu=False,
                       auto_fit_cell_calibration=False)
    r = champion_recipe(d)
    args = recipe_to_train_args(r, distill_cache="c.npz", map_fr_consistency_weight=0.1)
    s = " ".join(args)
    assert "--cyclen-physics-prior" not in s
    assert "--quantile-heads" not in s
    assert "--promote-max-asm-bu" not in s
    assert "--distill-targets" not in s          # distill_weight 0 -> no teacher
    assert "--no-auto-cell-calibration" in s
    assert "--map-fr-consistency-weight 0.1" in s


def test_remote_invocation_forwards_after_stop_token(tmp_path):
    r = champion_recipe(_fake_champion(tmp_path))
    args = recipe_to_train_args(r, distill_cache="c.npz")
    cmd = remote_invocation(args, input_deck="lpopt.inp")
    # Wrapper options MUST precede the sub-command: lpopt.remote has two
    # positionals (cmd, train_args nargs="*") and argparse fills positionals
    # greedily, so "train --input X -- ..." silently empties train_args and the
    # forwarded flags come back as "unrecognized arguments" (seen 2026-07-25).
    assert cmd.startswith("python -m lpopt.remote --input lpopt.inp train -- ")
    # ensemble/split ride in the forwarded args (they replace the wrapper default)
    assert "--ensemble 5" in cmd and "--split S1" in cmd


def test_remote_invocation_actually_parses(tmp_path):
    """The composed string must survive the REAL lpopt.remote parser.

    Regression for the 2026-07-25 A/B launch failure: ``lpopt.remote`` declares
    two positionals (``cmd``, then ``train_args`` with ``nargs="*"``).  argparse
    fills positionals greedily at the first opportunity, so an optional placed
    BETWEEN them -- ``train --input X -- ...`` -- matches BOTH against the single
    slot ahead of ``--input``, leaves ``train_args`` empty, and every forwarded
    flag comes back as ``unrecognized arguments``.  Asserting on the string shape
    alone (as this file previously did) cannot catch that; only parsing can.
    """
    import shlex

    import lpopt.remote as remote_mod

    r = champion_recipe(_fake_champion(tmp_path))
    cmd = remote_invocation(recipe_to_train_args(r, distill_cache="c.npz"))
    argv = shlex.split(cmd)[3:]          # drop "python -m lpopt.remote"

    sentinel = RuntimeError("parsed")

    def _boom(_path):
        raise sentinel

    monkey = remote_mod.RemoteSettings.from_input
    remote_mod.RemoteSettings.from_input = staticmethod(_boom)
    try:
        with pytest.raises(RuntimeError, match="parsed"):
            remote_mod.main(argv)        # SystemExit here == an argparse rejection
        # and the forwarded recipe must arrive INTACT, not swallowed
        ap_argv = argv[argv.index("--") + 1:]
        assert "--ensemble" in ap_argv and "--width" in ap_argv
    finally:
        remote_mod.RemoteSettings.from_input = monkey


def test_plan_includes_teacher_refresh_and_gate(tmp_path):
    d = _fake_champion(tmp_path)
    plan = plan_al_retrain(str(d), distill_cache="data/models/_v5_distill_soft.npz")
    assert plan["distill_teacher_refresh"]["enabled"] is True
    assert plan["distill_teacher_refresh"]["teacher"] == str(d)
    # steps: refresh -> push -> remote train ; gate documented
    assert any("refresh_distill_cache" in s for s in plan["steps"])
    # Wrapper options precede the sub-command, so the command name is no longer
    # adjacent to "remote" (see the remote_invocation docstring for why).
    assert any(s.startswith("python -m lpopt.remote ") and s.rstrip().endswith("push")
               for s in plan["steps"])
    assert any(s.startswith("python -m lpopt.remote ") and " train -- " in s
               for s in plan["steps"])
    assert "gate_promote.py" in plan["gate_after"]


def test_plan_no_distill_champion_skips_refresh(tmp_path):
    d = _fake_champion(tmp_path, distill_weight=0.0)
    plan = plan_al_retrain(str(d))
    assert plan["distill_teacher_refresh"]["enabled"] is False
    assert not any("refresh_distill_cache" in s for s in plan["steps"])


# --------------------------------------------------------------------------- #
# the hires recipe must survive a retrain (arm A6, champion 2026-07-25)
# --------------------------------------------------------------------------- #
def _hires_champion(tmp_path):
    d = _fake_champion(tmp_path)
    meta_p = d / "member_0" / "meta.json"
    if not meta_p.exists():
        meta_p = next(d.glob("member_*/meta.json"))
    meta = json.loads(meta_p.read_text())
    meta["cond_schema"] = "v6"
    meta.setdefault("net_config", {}).update(
        {"width": 224, "n_blocks": 8, "head_hidden": 384,
         "map_head_mode": "multiscale", "map_prior_channel": 50})
    meta.setdefault("train_config", {}).update(
        {"map_head_mode": "multiscale", "map_prior_residual": True,
         "map_spectral_weight": 0.3, "map_peak_weight": 2.0,
         "width": 224, "n_blocks": 8, "head_hidden": 384})
    meta_p.write_text(json.dumps(meta))
    return d


def test_recipe_captures_the_hires_structure(tmp_path):
    r = champion_recipe(_hires_champion(tmp_path))
    assert r["cond_schema"] == "v6"
    assert r["map_head_mode"] == "multiscale"
    assert r["map_prior_residual"] is True
    assert r["map_spectral_weight"] == 0.3
    assert r["head_hidden"] == 384


def test_retrain_args_reproduce_the_v6_champion(tmp_path):
    """A v6 champion must not be silently rebuilt at the trainer's v3 default.

    Regression for 2026-07-25: cond_schema and head_hidden were read into the
    recipe but never emitted, and the hires flags were not read at all, so an AL
    retrain from arm A6 would have produced a v3-schema, linear-map-head model
    wearing the champion's name.
    """
    args = recipe_to_train_args(champion_recipe(_hires_champion(tmp_path)),
                                distill_cache="c.npz")
    s = " ".join(args)
    assert "--cond-schema v6" in s
    assert "--head-hidden 384" in s
    assert "--map-decoder multiscale" in s
    assert "--map-prior-residual" in s
    assert "--map-spectral-weight 0.3" in s
    assert "--map-peak-weight 2.0" in s


def test_pre_hires_champion_emits_no_hires_flags(tmp_path):
    """A v5 champion keeps the legacy invocation exactly."""
    args = recipe_to_train_args(champion_recipe(_fake_champion(tmp_path)),
                                distill_cache="c.npz")
    s = " ".join(args)
    assert "--map-decoder" not in s
    assert "--map-prior-residual" not in s
    assert "--map-spectral-weight" not in s
    assert "--axial-head" not in s


# --------------------------------------------------------------------------- #
# the axial head must survive a retrain too (decision D10)
# --------------------------------------------------------------------------- #
def _axial_champion(tmp_path):
    d = _hires_champion(tmp_path)
    meta_p = next(d.glob("member_*/meta.json"))
    meta = json.loads(meta_p.read_text())
    meta["net_config"].update({"n_axial_anchors": 2, "n_axial_modes": 6})
    meta["train_config"].update({"axial_head": True, "axial_rank": 6,
                                 "axial_weight": 0.2})
    meta["axial_head"] = {"enabled": True, "anchors": ["boc", "eoc"], "rank": 6}
    meta_p.write_text(json.dumps(meta))
    return d


def test_recipe_captures_the_axial_head(tmp_path):
    r = champion_recipe(_axial_champion(tmp_path))
    assert r["axial_head"] is True
    assert r["axial_rank"] == 6
    assert r["axial_weight"] == 0.2


def test_retrain_args_reproduce_the_axial_champion(tmp_path):
    args = recipe_to_train_args(champion_recipe(_axial_champion(tmp_path)),
                                distill_cache="c.npz")
    s = " ".join(args)
    assert "--axial-head" in s
    assert "--axial-rank 6" in s
    assert "--axial-weight 0.2" in s


def test_recipe_reads_the_axial_head_off_net_config_alone(tmp_path):
    """A checkpoint whose train_config predates the flag still yields the head.

    ``net_config`` is what the WEIGHTS were built with, so it wins.
    """
    d = _axial_champion(tmp_path)
    meta_p = next(d.glob("member_*/meta.json"))
    meta = json.loads(meta_p.read_text())
    meta["train_config"].pop("axial_head")
    meta["train_config"].pop("axial_rank")
    meta_p.write_text(json.dumps(meta))
    r = champion_recipe(d)
    assert r["axial_head"] is True and r["axial_rank"] == 6
