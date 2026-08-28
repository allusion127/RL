"""Freeze-and-finetune training mode (``--init-from`` / ``--freeze-trunk-cyclen``).

This is a nuclear-fuel-loading surrogate, so the binding requirement is that the
frozen CYCLEN prediction stays byte-identical to the champion while F_r and the
node_peak / map heads adapt.  The tests pin, at ``atol=0``:

* ``--init-from`` loads the champion's per-member weights (a strict, fail-loud
  state_dict load; a net-config mismatch raises).
* one optimizer step under ``--freeze-trunk-cyclen`` leaves the shared trunk,
  the convergence head, and every CYCLEN output row EXACTLY where they were,
  while ``map_head`` and the non-cyclen (f_r) rows do move.
* the frozen net's CYCLEN ``mu`` on a fixed input is bit-identical before and
  after training (the whole cyclen path is frozen).
* the champion's per-cell cyclen calibration is copied verbatim, never re-fit.

The freeze contract is checked at the unit level (through the REAL
``_apply_freeze_trunk_cyclen`` / ``_build_member_optim`` / ``_step_member``
functions, driven by tiny random stubs) with a deliberately LARGE weight decay,
so any decoupled-weight-decay leak onto the frozen cyclen rows would surface.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.model.cell_calibrate import CELL_CALIB_NAME          # noqa: E402
from lpopt.model.physics_prior import PRIOR_NAME                 # noqa: E402
from lpopt.model.dataset_torch import TARGETS                    # noqa: E402
from lpopt.model.net import PosValNet, PosValNetConfig           # noqa: E402
from lpopt.model.train import (                                  # noqa: E402
    TrainConfig, _Norm, _apply_freeze_trunk_cyclen, _build_member_optim,
    _cyclen_quantile_rows, _load_champion_member_states, _step_member,
    fit_cell_calibrations,
)

STORE = "data/store"
N_CH, N_G = 43, 13
_CY = TARGETS.index("cyclen")          # 3
_FR = TARGETS.index("f_r")             # 0
_Q_NAMES = ("f_r", "cyclen")           # champion quantile-target order


# --------------------------------------------------------------------------- #
# tiny stubs (mirroring tests/test_v5_training_integration.py)
# --------------------------------------------------------------------------- #
def _net(seed: int = 11) -> PosValNet:
    torch.manual_seed(seed)
    return PosValNet(PosValNetConfig(
        in_channels=N_CH, n_globals=N_G, n_targets=7,
        n_quantile_targets=2, n_quantiles=3))


def _member(*, freeze: bool, wd: float = 0.1, lr: float = 1e-3, seed: int = 11):
    """A minimal ``_MemberState`` stand-in wired for ``_step_member``."""
    net = _net(seed)
    handles = _apply_freeze_trunk_cyclen(net, _Q_NAMES) if freeze else []
    optim = _build_member_optim(net, lr=lr, weight_decay=wd, freeze=freeze)
    m = types.SimpleNamespace()
    m.model = net
    m.fwd = net
    m.optim = optim
    m.norm = _Norm(np.full(7, 600.0), np.full(7, 5.0), np.zeros(4), np.ones(4),
                   torch.device("cpu"), _CY)
    m.use_nll = True
    m.use_prior = False
    m.q_idx = torch.tensor([_FR, _CY], dtype=torch.long)
    m.q_names = _Q_NAMES
    m.running = 0.0
    m.n_batches = 0
    m.freeze_handles = handles
    return m


def _batch(n: int = 12, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    cells = torch.randn(n, N_CH, 19, 19, generator=g)
    cells[:, 0] = 1.0                                       # fuel mask
    return {
        "cells": cells,
        "globals": torch.randn(n, N_G, generator=g),
        "targets": torch.randn(n, 7, generator=g) * 5.0 + 600.0,
        "target_mask": torch.ones(n, 7),
        "conv_label": torch.ones(n),
        "conv_mask": torch.ones(n),
        "maps": torch.randn(n, 4, 9, 9, generator=g),
        "maps_mask": torch.ones(n, 4, 9, 9),
        "cyclen_cell": torch.zeros(n, dtype=torch.long),
    }


def _cyclen_mu(net: PosValNet, batch: dict) -> torch.Tensor:
    was_training = net.training
    net.eval()
    with torch.no_grad():
        out = net(batch["cells"], batch["globals"])
    if was_training:
        net.train()
    return out["mu"][:, _CY].clone()


def _make_champion(root: Path, seeds, net_cfg: PosValNetConfig) -> Path:
    d = root / "champ"
    d.mkdir()
    names = []
    for s in seeds:
        md = d / f"member_{s}"
        md.mkdir()
        torch.manual_seed(int(s))
        torch.save(PosValNet(net_cfg).state_dict(), md / "model.pt")
        names.append(md.name)
    (d / "ensemble.json").write_text(
        json.dumps({"members": names, "n_members": len(names)}), encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# (1) --init-from loads champion weights (strict, fail-loud)
# --------------------------------------------------------------------------- #
def test_init_from_loads_champion_weights(tmp_path):
    net_cfg = PosValNetConfig(in_channels=N_CH, n_globals=N_G, n_targets=7,
                              n_quantile_targets=2, n_quantiles=3)
    champ = _make_champion(tmp_path, [100, 101], net_cfg)

    states = _load_champion_member_states(champ)
    assert len(states) == 2
    # ensemble.json order is preserved (member 100 then 101, distinct inits).
    assert not torch.equal(states[0]["mu_head.weight"], states[1]["mu_head.weight"])

    torch.manual_seed(999)
    net = PosValNet(net_cfg)                                # a DIFFERENT init
    assert not torch.equal(net.mu_head.weight, states[0]["mu_head.weight"])

    net.load_state_dict(states[0], strict=True)            # the init-from step
    torch.testing.assert_close(net.mu_head.weight, states[0]["mu_head.weight"],
                               rtol=0, atol=0)
    torch.testing.assert_close(net.stem[0].weight, states[0]["stem.0.weight"],
                               rtol=0, atol=0)


def test_init_from_strict_mismatch_fails_loudly(tmp_path):
    champ = _make_champion(
        tmp_path, [100],
        PosValNetConfig(in_channels=N_CH, n_globals=N_G, n_targets=7))
    states = _load_champion_member_states(champ)
    # a different trunk width cannot strict-load the champion state_dict.
    wide = PosValNet(PosValNetConfig(in_channels=N_CH, n_globals=N_G,
                                     n_targets=7, width=128))
    with pytest.raises(RuntimeError):
        wide.load_state_dict(states[0], strict=True)


def test_fewer_champion_members_reuse_member_zero(tmp_path):
    champ = _make_champion(
        tmp_path, [100],
        PosValNetConfig(in_channels=N_CH, n_globals=N_G, n_targets=7))
    states = _load_champion_member_states(champ)
    assert len(states) == 1
    # the train_ensemble mapping picks champion member 0 for every position.
    picked = [states[min(i, len(states) - 1)] for i in range(5)]
    assert all(p is states[0] for p in picked)


def test_missing_champion_members_raise(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_champion_member_states(empty)


# --------------------------------------------------------------------------- #
# (2)/(3) freeze contract: trunk + cyclen frozen, f_r + map trained
# --------------------------------------------------------------------------- #
def test_cyclen_quantile_rows_target_the_cyclen_block():
    net = _net()
    # q_names = (f_r, cyclen), n_quantiles = 3  ->  cyclen block is rows [3, 4, 5].
    assert _cyclen_quantile_rows(net, _Q_NAMES) == [3, 4, 5]
    # no quantile head or cyclen absent -> no rows.
    plain = PosValNet(PosValNetConfig(in_channels=N_CH, n_globals=N_G, n_targets=7))
    assert _cyclen_quantile_rows(plain, _Q_NAMES) == []
    assert _cyclen_quantile_rows(net, ("f_r",)) == []


def test_freeze_optim_excludes_trunk_and_zeros_head_decay():
    m = _member(freeze=True, wd=1e-4)
    net = m.model
    for name in ("stem", "blocks", "films", "head_trunk", "conv_head"):
        mod = getattr(net, name)
        assert all(not p.requires_grad for p in mod.parameters()), name
    # map head + output heads stay trainable.
    for name in ("map_head", "mu_head", "log_sigma_head", "quantile_head"):
        assert all(p.requires_grad for p in getattr(net, name).parameters()), name
    # two optimizer groups: decay (map_head) and no-decay (mu/log_sigma/quantile).
    groups = m.optim.param_groups
    assert len(groups) == 2
    assert sorted(g["weight_decay"] for g in groups) == [0.0, 1e-4]
    # the frozen trunk params never enter the optimizer.
    opt_ids = {id(p) for g in groups for p in g["params"]}
    for name in ("stem", "blocks", "films", "head_trunk", "conv_head"):
        for p in getattr(net, name).parameters():
            assert id(p) not in opt_ids


def test_freeze_step_freezes_trunk_and_cyclen_but_trains_fr_and_map():
    # Large weight decay: if decoupled WD leaked onto the frozen cyclen rows,
    # the atol=0 assertions below would fail.
    m = _member(freeze=True, wd=0.1, lr=1e-3)
    net = m.model
    cfg = TrainConfig()
    fixed = _batch(seed=5)

    frozen = {
        "stem": net.stem[0].weight,
        "block0_conv1": net.blocks[0].conv1.weight,
        "block5_conv2": net.blocks[5].conv2.weight,
        "head_trunk0": net.head_trunk[0].weight,
        "head_trunk2": net.head_trunk[2].weight,
        "conv_w": net.conv_head.weight,
        "conv_b": net.conv_head.bias,
        "mu_cy_w": net.mu_head.weight[_CY],
        "mu_cy_b": net.mu_head.bias[_CY:_CY + 1],
        "ls_cy_w": net.log_sigma_head.weight[_CY],
        "ls_cy_b": net.log_sigma_head.bias[_CY:_CY + 1],
        "q_cy_w": net.quantile_head.weight[3:6],
        "q_cy_b": net.quantile_head.bias[3:6],
    }
    trained = {
        "map_w": net.map_head.weight,
        "mu_fr_w": net.mu_head.weight[_FR],
        "ls_fr_w": net.log_sigma_head.weight[_FR],
        "q_fr_w": net.quantile_head.weight[0:3],
    }
    before_frozen = {k: v.detach().clone() for k, v in frozen.items()}
    before_trained = {k: v.detach().clone() for k, v in trained.items()}
    cy_before = _cyclen_mu(net, fixed)

    net.train()
    for s in range(3):
        _step_member(m, _batch(seed=100 + s), cfg,
                     use_amp=False, device=torch.device("cpu"))

    # --- frozen: byte-identical (atol 0) --------------------------------------
    for k, ref in before_frozen.items():
        torch.testing.assert_close(frozen[k], ref, rtol=0, atol=0,
                                   msg=f"{k} moved but should be frozen")
    # --- trained: actually moved ---------------------------------------------
    for k, ref in before_trained.items():
        assert not torch.equal(trained[k], ref), f"{k} did not train"

    # --- the frozen net's cyclen mu on a fixed input is bit-identical ---------
    cy_after = _cyclen_mu(net, fixed)
    torch.testing.assert_close(cy_after, cy_before, rtol=0, atol=0)
    assert m.n_batches == 3 and np.isfinite(m.running)


def test_freeze_requires_init_state():
    """Freezing randomly-inited cyclen rows is refused (needs champion weights)."""
    from lpopt.model.train import _train_members
    with pytest.raises(ValueError, match="requires init_from"):
        _train_members(
            [1], train_ds=None, val_ds=None,
            cfg=TrainConfig(freeze_trunk_cyclen=True), device="cpu",
            globals_names=["g"] * N_G, reader=None, eff_batch=8,
            lr=1e-3, lr_final=1e-4, warm=1, resident=False, compile_flag=False,
            n_channels=N_CH, verbose=False, init_states=None)


# --------------------------------------------------------------------------- #
# (calibration) champion per-cell cyclen calibration is copied verbatim
# --------------------------------------------------------------------------- #
def test_freeze_copies_champion_cyclen_calibration(tmp_path):
    champ = tmp_path / "champ"
    champ.mkdir()
    payload = {
        "bin_width": 0.05,
        "schema": "cell_calib_v1",
        "cells": {"feed=101|ebin=5.0": {
            "a": 1.2360769441777237, "b": -138.78619375647736,
            "estimator": "affine", "n": 168}},
    }
    (champ / CELL_CALIB_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    model_dir = tmp_path / "new"
    model_dir.mkdir()
    cfg = TrainConfig()
    cfg.freeze_trunk_cyclen = True
    cfg.init_from = str(champ)
    # split pinned so the (swallowed) F_r fit does not glob a missing meta.json;
    # it fails fast on the empty model dir and is logged, not raised.
    out = fit_cell_calibrations(model_dir, split="S1", cfg=cfg, device="cpu",
                                store_dir=STORE)

    assert (model_dir / CELL_CALIB_NAME).is_file()
    copied = json.loads((model_dir / CELL_CALIB_NAME).read_text(encoding="utf-8"))
    assert copied == payload                                # verbatim
    assert out["cyclen"]["copied_from_champion"].endswith(CELL_CALIB_NAME)


# --------------------------------------------------------------------------- #
# (physics prior) champion cyclen physics prior is copied byte-identical
# --------------------------------------------------------------------------- #
def _write_champion_prior(champ: Path) -> bytes:
    """Write a champion prior file with LF endings (as a real champion has);
    return its raw bytes for a byte-identity assertion."""
    raw = (
        b'{\n'
        b'  "alpha": 10.169506849134203,\n'
        b'  "beta": 605.9212261303663,\n'
        b'  "fallback_cyclen": 683.6650142716411,\n'
        b'  "n_fit": 39589,\n'
        b'  "pearson": 0.8613997568084025,\n'
        b'  "rho_leak": 3500.0,\n'
        b'  "schema": "cyclen_physics_prior_v1",\n'
        b'  "spearman": 0.8631479959631229,\n'
        b'  "split": "S1"\n'
        b'}'
    )
    (champ / PRIOR_NAME).write_bytes(raw)
    return raw


def test_freeze_copies_champion_cyclen_physics_prior(tmp_path):
    """Freeze + prior ON: the served prior must be the champion's, byte-identical.

    Re-fitting on a grown store would drift the served cyclen (= residual + prior)
    and change within-cell cyclen ranking; the frozen residual head was trained
    against the champion's prior, so the prior is COPIED verbatim (never re-fit).
    """
    champ = tmp_path / "champ"
    champ.mkdir()
    prior_bytes = _write_champion_prior(champ)
    # cell_calibration.json is also copied in freeze mode; provide it so the
    # (swallowed) copy does not merely log a warning.
    (champ / CELL_CALIB_NAME).write_text(
        json.dumps({"bin_width": 0.05, "schema": "cell_calib_v1", "cells": {}},
                   indent=2, sort_keys=True), encoding="utf-8")

    model_dir = tmp_path / "new"
    model_dir.mkdir()
    cfg = TrainConfig()
    cfg.freeze_trunk_cyclen = True
    cfg.init_from = str(champ)
    cfg.cyclen_physics_prior = True
    out = fit_cell_calibrations(model_dir, split="S1", cfg=cfg, device="cpu",
                                store_dir=STORE)

    dst = model_dir / PRIOR_NAME
    assert dst.is_file()
    assert dst.read_bytes() == prior_bytes                  # BYTE-identical
    assert out["cyclen_physics_prior"]["copied_from_champion"].endswith(PRIOR_NAME)


def test_flag_off_does_not_copy_cyclen_physics_prior(tmp_path):
    """Prior flag OFF (the from-scratch path): never copy — the prior is re-fit.

    Also covers freeze OFF: the champion prior must only be copied when BOTH the
    freeze mode and the ``--cyclen-physics-prior`` flag are on.
    """
    champ = tmp_path / "champ"
    champ.mkdir()
    _write_champion_prior(champ)

    for freeze, prior_on in ((True, False), (False, True)):
        model_dir = tmp_path / f"new_{int(freeze)}_{int(prior_on)}"
        model_dir.mkdir()
        cfg = TrainConfig()
        cfg.freeze_trunk_cyclen = freeze
        cfg.init_from = str(champ) if freeze else None
        cfg.cyclen_physics_prior = prior_on
        out = fit_cell_calibrations(model_dir, split="S1", cfg=cfg, device="cpu",
                                    store_dir=STORE)
        assert not (model_dir / PRIOR_NAME).exists()        # never copied
        assert out["cyclen_physics_prior"] is None


def test_flags_off_optim_is_the_legacy_single_group():
    """Freeze OFF must build the pre-flag optimizer (one group, all params)."""
    net = _net()
    opt = _build_member_optim(net, lr=3e-4, weight_decay=1e-4, freeze=False)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 1e-4
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in net.parameters())
    assert all(p.requires_grad for p in net.parameters())   # nothing frozen
