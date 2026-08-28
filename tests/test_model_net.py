"""PosValNet shape/param-count + checkpoint save/load round-trip identity (M3b)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.model.net import (          # noqa: E402
    PosValNet, PosValNetConfig, build_member, count_parameters,
)
from lpopt.model.train import load_member, save_member       # noqa: E402


def _fake_batch(cfg: PosValNetConfig, n: int = 3):
    cells = torch.randn(n, cfg.in_channels, 19, 19)
    cells[:, 0] = (torch.rand(n, 19, 19) > 0.3).float()       # fuel mask
    g = torch.randn(n, cfg.n_globals)
    return cells, g


def test_forward_shapes() -> None:
    net = PosValNet()
    cfg = net.config
    cells, g = _fake_batch(cfg, 4)
    out = net(cells, g)
    # Phase D: 7 targets (adds discharge_burnup, max_pin_burnup) — plan sec. 12.4.
    assert out["mu"].shape == (4, 7)
    assert out["log_sigma"].shape == (4, 7)
    assert out["map"].shape == (4, 4, 9, 9)
    assert out["conv_logit"].shape == (4,)
    assert net.config.n_targets == 7


def test_legacy_5target_config_rebuilds() -> None:
    # A cond_v2 checkpoint stores n_targets=5; the net must rebuild at that width
    # (backward-compat load, plan sec. 6.2).
    net = PosValNet(PosValNetConfig(n_targets=5))
    cells, g = _fake_batch(net.config, 4)
    out = net(cells, g)
    assert out["mu"].shape == (4, 5)
    assert out["log_sigma"].shape == (4, 5)


def test_v4_wide_input_config_shapes_and_band() -> None:
    # cond_v4 widens the stem input (43 channels) + globals (13); the head still
    # predicts 7 targets and the member stays inside the 1.0M-2.5M param band.
    from lpopt.model.featurize import CHANNELS_V4
    cfg = PosValNetConfig(in_channels=len(CHANNELS_V4), n_globals=13)
    net = PosValNet(cfg)
    cells, g = _fake_batch(cfg, 3)
    assert cells.shape[1] == len(CHANNELS_V4) == 43
    assert g.shape[1] == 13
    out = net(cells, g)
    assert out["mu"].shape == (3, 7)
    assert out["log_sigma"].shape == (3, 7)
    assert out["map"].shape == (3, 4, 9, 9)
    n = count_parameters(net)
    assert 1_000_000 <= n <= 2_500_000, f"{n} outside 1.0M-2.5M band"


def test_v4_checkpoint_roundtrip_identity(tmp_path) -> None:
    # a 43-channel/13-global v4 member round-trips through meta.json + model.pt
    from lpopt.model.featurize import CHANNELS_V4
    cfg = PosValNetConfig(in_channels=len(CHANNELS_V4), n_globals=13)
    net = PosValNet(cfg).eval()
    cells, g = _fake_batch(cfg, 4)
    with torch.no_grad():
        ref = net(cells, g)
    meta = {"net_config": cfg.__dict__, "cond_schema": "v4",
            "channels": list(CHANNELS_V4),
            "target_zscore": {"mean": [0.0] * 7, "std": [1.0] * 7},
            "seed": 0, "versions": {"torch": torch.__version__}}
    save_member(tmp_path / "member_0", net, meta)
    loaded, meta2 = load_member(tmp_path / "member_0", "cpu")
    assert meta2["net_config"]["in_channels"] == 43
    with torch.no_grad():
        got = loaded(cells, g)
    for key in ("mu", "log_sigma", "map", "conv_logit"):
        assert torch.allclose(ref[key], got[key], atol=1e-6), key


def test_param_count_in_band() -> None:
    n = count_parameters(PosValNet())
    assert 1_000_000 <= n <= 2_500_000, f"{n} outside 1.0M-2.5M band"
    # build_member asserts the band internally
    assert count_parameters(build_member()) == n


def test_param_count_band_enforced() -> None:
    # a deliberately tiny width underfills the band -> build_member must reject.
    with pytest.raises(AssertionError):
        build_member(width=16)


def test_checkpoint_roundtrip_identity(tmp_path) -> None:
    cfg = PosValNetConfig()
    net = PosValNet(cfg).eval()
    cells, g = _fake_batch(cfg, 5)
    with torch.no_grad():
        ref = net(cells, g)

    meta = {
        "net_config": cfg.__dict__,
        "target_zscore": {"mean": [1.0, 2.0, 3.0, 4.0, 5.0],
                          "std": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "seed": 0,
        "versions": {"torch": torch.__version__},
    }
    save_member(tmp_path / "member_0", net, meta)
    # state_dict-only checkpoint: must load with weights_only=True (no pickled class)
    state = torch.load(tmp_path / "member_0" / "model.pt", weights_only=True)
    assert isinstance(state, dict)

    loaded, meta2 = load_member(tmp_path / "member_0", "cpu")
    assert meta2["net_config"] == cfg.__dict__
    with torch.no_grad():
        got = loaded(cells, g)
    for key in ("mu", "log_sigma", "map", "conv_logit"):
        assert torch.allclose(ref[key], got[key], atol=1e-6), key


def test_meta_is_pure_json(tmp_path) -> None:
    import json
    cfg = PosValNetConfig()
    net = PosValNet(cfg)
    meta = {"net_config": cfg.__dict__, "target_zscore": {"mean": [0]*5, "std": [1]*5}}
    save_member(tmp_path / "m", net, meta)
    # meta.json round-trips as plain JSON (no custom objects)
    round = json.loads((tmp_path / "m" / "meta.json").read_text(encoding="utf-8"))
    assert round["net_config"]["width"] == cfg.width
