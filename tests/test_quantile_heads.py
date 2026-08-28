"""Pinball-loss quantile heads (q10/q50/q90) for f_r and cyclen.

Two contracts:

* **Flag OFF is a no-op.** The network's module set, parameter count, state_dict
  and ``forward`` output keys must be bit-identical to the pre-v5 net (golden
  digests captured before the change), so every deployed checkpoint keeps
  loading and serving unchanged.
* **Flag ON produces calibrated quantiles.** Minimizing the pinball loss must
  drive q10/q90 to the true 10th/90th percentiles — verified by fitting a
  synthetic distribution and measuring that the q10-q90 band covers ~80% of it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.model.net import (                                       # noqa: E402
    PosValNet, PosValNetConfig, count_parameters,
)
from lpopt.model.train import TrainConfig, pinball_loss             # noqa: E402

#: (state_dict sha, forward-output sha) captured BEFORE the quantile head landed.
GOLDEN_NET = {
    (26, 10): ("e98b35a8e6843749760a94ee4dd9f3cbc733950d2842036a21965105716af7a1",
               "c2cbcc6c76d751d5e63ba6ed786bfd1c5d9e0f0334a7fb8400fd3f47f17ba750"),
    (43, 13): ("72a94de17475f39c9895083f1c2f9b515198e462537898c7668543d0a04d109a",
               "c92d19b9cbc7eb22380b3d1bf81f9bad00e55919b676c700a533d9f56c8d64fb"),
}
LEGACY_KEYS = {"conv_logit", "log_sigma", "map", "mu"}


def _net_digest(cfg: PosValNetConfig) -> tuple[str, str]:
    torch.manual_seed(1234)
    net = PosValNet(cfg)
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    torch.manual_seed(7)
    cells = torch.randn(3, cfg.in_channels, 19, 19)
    cells[:, 0] = 1.0
    g = torch.randn(3, cfg.n_globals)
    net.eval()
    with torch.no_grad():
        out = net(cells, g)
    oh = hashlib.sha256()
    for k in sorted(out):
        oh.update(k.encode())
        oh.update(out[k].numpy().tobytes())
    return h.hexdigest(), oh.hexdigest()


# --------------------------------------------------------------------------- #
# flag OFF: byte-identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("in_ch,n_g", [(26, 10), (43, 13)])
def test_disabled_net_is_byte_identical(in_ch, n_g):
    cfg = PosValNetConfig(in_channels=in_ch, n_globals=n_g)
    assert _net_digest(cfg) == GOLDEN_NET[(in_ch, n_g)]


def test_disabled_net_registers_no_quantile_module():
    net = PosValNet(PosValNetConfig(in_channels=43, n_globals=13))
    assert not net.has_quantiles
    assert not hasattr(net, "quantile_head")
    assert not any("quantile" in k for k in net.state_dict())


def test_disabled_forward_emits_only_the_legacy_keys():
    net = PosValNet(PosValNetConfig(in_channels=43, n_globals=13))
    cells = torch.randn(2, 43, 19, 19)
    cells[:, 0] = 1.0
    assert set(net(cells, torch.randn(2, 13))) == LEGACY_KEYS


def test_train_config_defaults_leave_quantiles_off():
    cfg = TrainConfig()
    assert cfg.quantile_heads is False
    assert cfg.quantile_levels == (0.10, 0.50, 0.90)
    assert cfg.quantile_targets == ("f_r", "cyclen")


def test_legacy_meta_without_quantile_keys_still_builds():
    """``PosValNetConfig(**meta['net_config'])`` for a pre-v5 meta must work."""
    legacy = {"in_channels": 43, "n_globals": 13, "width": 112, "n_blocks": 6,
              "groups": 8, "film_every": 2, "head_hidden": 256, "n_targets": 7,
              "n_map_channels": 4}
    net = PosValNet(PosValNetConfig(**legacy))
    assert not net.has_quantiles
    assert count_parameters(net) == 1_614_131


# --------------------------------------------------------------------------- #
# flag ON: shapes + the pinball loss
# --------------------------------------------------------------------------- #
def test_enabled_net_emits_the_quantile_block():
    net = PosValNet(PosValNetConfig(in_channels=48, n_globals=13, n_targets=8,
                                    n_quantile_targets=2, n_quantiles=3))
    cells = torch.randn(4, 48, 19, 19)
    cells[:, 0] = 1.0
    out = net(cells, torch.randn(4, 13))
    assert set(out) == LEGACY_KEYS | {"quantiles"}
    assert out["quantiles"].shape == (4, 2, 3)
    assert out["mu"].shape == (4, 8)          # the mean head is untouched


def test_quantile_head_stays_inside_the_param_band():
    from lpopt.model.net import build_member
    build_member(PosValNetConfig(in_channels=48, n_globals=13, n_targets=8,
                                 n_quantile_targets=2, n_quantiles=3))


def test_pinball_is_zero_on_an_exact_fit():
    y = torch.tensor([[1.0, 2.0]])
    q = y.unsqueeze(-1).repeat(1, 1, 3)
    loss = pinball_loss(q, y, torch.ones_like(y), (0.1, 0.5, 0.9))
    assert float(loss) == pytest.approx(0.0)


def test_pinball_is_asymmetric_in_the_right_direction():
    """For tau=0.9 under-prediction must cost 9x what over-prediction costs."""
    y = torch.tensor([[0.0]])
    mask = torch.ones_like(y)
    under = pinball_loss(torch.tensor([[[-1.0]]]), y, mask, (0.9,))
    over = pinball_loss(torch.tensor([[[1.0]]]), y, mask, (0.9,))
    assert float(under) == pytest.approx(0.9)
    assert float(over) == pytest.approx(0.1)
    # and the mirror image at tau=0.1
    assert float(pinball_loss(torch.tensor([[[-1.0]]]), y, mask, (0.1,))) \
        == pytest.approx(0.1)


def test_pinball_ignores_masked_entries():
    y = torch.tensor([[1.0, 99.0]])
    q = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])          # second target censored
    assert float(pinball_loss(q, y, mask, (0.1, 0.5, 0.9))) == pytest.approx(0.0)


def test_pinball_is_zero_when_nothing_is_valid():
    y = torch.tensor([[1.0]])
    q = torch.tensor([[[5.0, 5.0, 5.0]]])
    assert float(pinball_loss(q, y, torch.zeros_like(y), (0.1, 0.5, 0.9))) == 0.0


def test_pinball_tolerates_nan_under_a_zero_mask():
    """Masked rows carry NaN raw targets; they must not poison the mean."""
    y = torch.tensor([[float("nan")]])
    q = torch.tensor([[[0.0, 0.0, 0.0]]])
    assert float(pinball_loss(q, y, torch.zeros_like(y), (0.5,))) == 0.0


# --------------------------------------------------------------------------- #
# coverage: the metric that actually matters
# --------------------------------------------------------------------------- #
def test_q10_q90_band_covers_about_eighty_percent_on_a_synthetic_fit():
    """Fit three free scalars against a known distribution by pinball loss; the
    recovered q10/q90 must bracket ~80% of the samples (the nominal coverage)."""
    torch.manual_seed(0)
    n = 8000
    y = torch.randn(n, 1) * 3.0 + 10.0                 # N(10, 3)
    levels = (0.10, 0.50, 0.90)
    q = torch.zeros(1, 1, 3, requires_grad=True)
    opt = torch.optim.Adam([q], lr=0.2)
    mask = torch.ones(n, 1)
    for _ in range(600):
        opt.zero_grad()
        loss = pinball_loss(q.expand(n, 1, 3), y, mask, levels)
        loss.backward()
        opt.step()
    lo, mid, hi = (float(v) for v in q.detach()[0, 0])
    assert lo < mid < hi                                # ordered quantiles
    assert lo == pytest.approx(10.0 - 1.2816 * 3.0, abs=0.35)
    assert mid == pytest.approx(10.0, abs=0.25)
    assert hi == pytest.approx(10.0 + 1.2816 * 3.0, abs=0.35)
    covered = float(((y >= lo) & (y <= hi)).float().mean())
    assert 0.75 < covered < 0.85, covered


# --------------------------------------------------------------------------- #
# the prediction object stays vendor-compatible
# --------------------------------------------------------------------------- #
def test_quantile_prediction_is_a_surrogate_prediction():
    from lpopt.model.model_api import QuantileSurrogatePrediction
    from lpopt.vendor.masterrl.surrogate import SurrogatePrediction, TARGET_NAMES

    mean = np.zeros((4, len(TARGET_NAMES)))
    p = QuantileSurrogatePrediction(
        mean, mean.copy(), mean.copy(),
        quantiles=np.zeros((4, 2, 3)),
        quantile_targets=("f_r", "cyclen"),
        quantile_levels=(0.1, 0.5, 0.9))
    # the vendor contract: still a SurrogatePrediction, still 7 columns
    assert isinstance(p, SurrogatePrediction)
    assert p.mean.shape[1] == len(TARGET_NAMES) == 7
    assert p.row(0)["cyclen"]["mean"] == 0.0
    assert p.mean_fom(0).f_r == 0.0
    # and the additive part is reachable
    lo, hi = p.band("cyclen")
    assert lo.shape == hi.shape == (4,)


def test_quantile_prediction_rejects_a_wrong_width():
    from lpopt.model.model_api import QuantileSurrogatePrediction
    bad = np.zeros((4, 5))
    with pytest.raises(ValueError, match="invalid shape"):
        QuantileSurrogatePrediction(bad, bad.copy(), bad.copy())
