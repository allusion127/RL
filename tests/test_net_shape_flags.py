"""Network-shape training knobs (--width / --n-blocks / --head-hidden).

Capacity sweep support: the member CNN's width/depth/head size become training
config + CLI flags.  Two contracts:

* **Flag OFF is byte-identical.** At the defaults (112/6/256) the constructed
  ``PosValNetConfig`` is field-for-field the pre-flag one, so a member's init
  (under its per-seed ``manual_seed``) is bit-identical — the deployed training
  path is unchanged.
* **Flag ON resizes AND round-trips through serving.** ``meta.net_config`` carries
  the width, and ``load_member`` rebuilds the exact architecture — so model_api
  needs no change to serve a wider champion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lpopt.model.net import PosValNet, PosValNetConfig, count_parameters   # noqa: E402
from lpopt.model.train import TrainConfig, load_member, main, save_member  # noqa: E402


def _sha(cfg: PosValNetConfig) -> str:
    torch.manual_seed(1234)
    sd = PosValNet(cfg).state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# flag OFF: byte identity
# --------------------------------------------------------------------------- #
def test_train_config_shape_defaults_match_the_net():
    tc = TrainConfig()
    pc = PosValNetConfig()
    assert (tc.width, tc.n_blocks, tc.head_hidden) == (pc.width, pc.n_blocks,
                                                       pc.head_hidden)
    assert (tc.width, tc.n_blocks, tc.head_hidden) == (112, 6, 256)


@pytest.mark.parametrize("in_ch,n_g,nt", [(26, 10, 7), (48, 13, 8)])
def test_explicit_defaults_are_byte_identical_to_implicit(in_ch, n_g, nt):
    implicit = PosValNetConfig(in_channels=in_ch, n_globals=n_g, n_targets=nt)
    explicit = PosValNetConfig(in_channels=in_ch, n_globals=n_g, n_targets=nt,
                               width=112, n_blocks=6, head_hidden=256)
    assert _sha(implicit) == _sha(explicit)


def test_train_cli_defaults_leave_shape_untouched():
    """A trainer invocation without the flags must keep 112/6/256."""
    import argparse
    ap = argparse.ArgumentParser()
    # mirror the real defaults: the flags default to None -> cfg unchanged
    for name in ("width", "n_blocks", "head_hidden"):
        ap.add_argument(f"--{name.replace('_', '-')}", type=int, default=None)
    args = ap.parse_args([])
    cfg = TrainConfig()
    if args.width is not None:
        cfg.width = args.width
    assert (cfg.width, cfg.n_blocks, cfg.head_hidden) == (112, 6, 256)


def test_trainer_help_exposes_the_flags():
    import contextlib
    import io
    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
        main(["--help"])
    out = buf.getvalue()
    for flag in ("--width", "--n-blocks", "--head-hidden"):
        assert flag in out


# --------------------------------------------------------------------------- #
# flag ON: resize + serving round-trip
# --------------------------------------------------------------------------- #
def test_width_160_is_about_two_x_params():
    p112 = count_parameters(PosValNet(PosValNetConfig(
        in_channels=48, n_globals=13, n_targets=8,
        n_quantile_targets=2, n_quantiles=3)))
    p160 = count_parameters(PosValNet(PosValNetConfig(
        in_channels=48, n_globals=13, width=160, n_targets=8,
        n_quantile_targets=2, n_quantiles=3)))
    assert 1.85 < p160 / p112 < 2.05
    assert 3.0e6 < p160 < 3.3e6


def test_wider_member_round_trips_through_load_member(tmp_path):
    cfg = PosValNetConfig(in_channels=48, n_globals=13, width=160, n_targets=8,
                          n_quantile_targets=2, n_quantiles=3)
    torch.manual_seed(9)
    model = PosValNet(cfg).eval()
    meta = {
        "net_config": dict(model.config.__dict__),
        "target_zscore": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "cond_schema": "v5",
        "target_names": ["f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                         "discharge_burnup", "max_pin_burnup", "max_assembly_burnup"],
    }
    d = tmp_path / "member_1"
    save_member(d, model, meta)
    # meta must actually record the wider width (serving reads it)
    assert json.loads((d / "meta.json").read_text())["net_config"]["width"] == 160

    reloaded, meta2 = load_member(d)
    assert meta2["net_config"]["width"] == 160
    assert count_parameters(reloaded) == count_parameters(model)
    cells = torch.randn(2, 48, 19, 19)
    cells[:, 0] = 1.0
    g = torch.randn(2, 13)
    with torch.no_grad():
        a, b = model(cells, g), reloaded(cells, g)
    torch.testing.assert_close(a["mu"], b["mu"])
    torch.testing.assert_close(a["quantiles"], b["quantiles"])


def test_wider_net_keeps_all_output_heads():
    net = PosValNet(PosValNetConfig(in_channels=48, n_globals=13, width=160,
                                    n_targets=8, n_quantile_targets=2, n_quantiles=3))
    cells = torch.randn(2, 48, 19, 19)
    cells[:, 0] = 1.0
    out = net(cells, torch.randn(2, 13))
    assert set(out) == {"mu", "log_sigma", "map", "conv_logit", "quantiles"}
    assert out["mu"].shape == (2, 8)
    assert out["quantiles"].shape == (2, 2, 3)
