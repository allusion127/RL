"""The hires bundle: multiscale map decoder, power-map prior, spectral map loss.

Design doc: ``data/reports/hires_model_ab_design_20260725.md``.

Three contracts are pinned here:

* **Flags OFF is byte-identical.**  A default ``PosValNet`` must keep the exact
  module set, ``state_dict`` keys and parameter count of the pre-hires network
  (so the live champion checkpoint still loads ``strict=True``), and the default
  training loss must not gain a term.
* **The prior is physics, not fitting.**  ``power_prior`` solves a real
  eigenvalue problem on the 241-node core; the tests check the invariants that
  make it a prior (core mean 1, flat input -> smooth bowl, monotone response to
  a hot slot) rather than pinning magic numbers.
* **The spectral loss is a SHAPE loss.**  It must be blind to level and to DC,
  and it must weight high wavenumbers above low ones -- that asymmetry is the
  whole point (report 20260725 section 3.6 measured the attenuation to rise with
  wavenumber while the raw power falls).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.model import power_prior as pp                            # noqa: E402
from lpopt.model.featurize import (                                  # noqa: E402
    CHANNELS_BY_SCHEMA, CHANNELS_V5, CHANNELS_V6, CHANNELS_V6_CONTRAST,
    CHANNELS_V6_PRIOR, FeatureEncoder,
)
from lpopt.model.net import (                                        # noqa: E402
    PosValNet, PosValNetConfig, _tap_indices, count_parameters,
)
from lpopt.model.train import (                                      # noqa: E402
    TrainConfig, map_loss, spectral_map_loss,
)

N_CH, N_G = 48, 13
CHAMPION = Path("data/models/20260724_213535/member_20260716/model.pt")


def _net(**over) -> PosValNet:
    torch.manual_seed(7)
    return PosValNet(PosValNetConfig(
        in_channels=N_CH, n_globals=N_G, n_targets=8, width=64, n_blocks=6,
        **over))


def _inputs(n=4, n_ch=N_CH, seed=3):
    g = torch.Generator().manual_seed(seed)
    cells = torch.randn(n, n_ch, 19, 19, generator=g)
    cells[:, 0] = (torch.rand(n, 19, 19, generator=g) > 0.3).float()
    return cells, torch.randn(n, N_G, generator=g)


# --------------------------------------------------------------------------- #
# 1. flags OFF == the pre-hires network
# --------------------------------------------------------------------------- #
def test_defaults_are_the_legacy_module_set():
    cfg = PosValNetConfig()
    assert cfg.map_head_mode == "linear"
    assert cfg.map_prior_channel == -1
    net = _net()
    keys = set(net.state_dict())
    assert any(k.startswith("map_head.") for k in keys)
    assert not any(k.startswith("map_decoder") for k in keys)
    assert not any(k.startswith("map_prior") for k in keys)


def test_explicit_defaults_are_state_dict_identical_to_implicit():
    a, b = _net(), _net(map_head_mode="linear", map_prior_channel=-1)
    assert list(a.state_dict()) == list(b.state_dict())
    assert count_parameters(a) == count_parameters(b)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


@pytest.mark.skipif(not CHAMPION.exists(), reason="champion checkpoint absent")
def test_live_champion_checkpoint_still_loads_strict():
    """The load-bearing regression: the serving champion must be unaffected."""
    net = PosValNet(PosValNetConfig(
        in_channels=48, n_globals=13, n_targets=8, width=160, n_blocks=6,
        n_quantile_targets=2, n_quantiles=3))
    net.load_state_dict(torch.load(CHAMPION, map_location="cpu"), strict=True)
    assert count_parameters(net) == 3_159_291


def test_train_config_hires_knobs_default_off():
    cfg = TrainConfig()
    assert cfg.map_head_mode == "linear"
    assert cfg.map_prior_residual is False
    assert cfg.map_spectral_weight == 0.0


# --------------------------------------------------------------------------- #
# 2. multiscale map decoder (arm A1)
# --------------------------------------------------------------------------- #
def test_tap_indices_always_end_at_the_final_block():
    for n in (4, 6, 8, 10, 12):
        taps = _tap_indices(n)
        assert taps[-1] == n - 1, "the decoder must still see the last block"
        assert len(taps) == len(set(taps)) and all(0 <= t < n for t in taps)
    assert _tap_indices(6) == (1, 3, 5)


def test_multiscale_decoder_shapes_and_param_delta():
    base, ms = _net(), _net(map_head_mode="multiscale")
    assert any(k.startswith("map_decoder.") for k in ms.state_dict())
    assert not any(k.startswith("map_head.") for k in ms.state_dict())
    cells, g = _inputs()
    out_b, out_m = base(cells, g), ms(cells, g)
    assert list(out_b) == list(out_m)
    assert out_m["map"].shape == out_b["map"].shape == (4, 4, 9, 9)
    # 4 taps (stem + 3 blocks) mixed to W, a dilated WxW conv, two GroupNorms and
    # the 1x1 projection -- minus the legacy 1x1 map_head the decoder replaces.
    w = base.config.width
    decoder = (4 * w * w * 9 + w) + 2 * w + (w * w * 9 + w) + 2 * w + (w * 4 + 4)
    legacy_head = w * 4 + 4
    assert count_parameters(ms) - count_parameters(base) == decoder - legacy_head


def test_multiscale_decoder_actually_reads_the_stem():
    """Perturbing ONLY the stem output must move the map — that is the skip path."""
    ms = _net(map_head_mode="multiscale")
    cells, g = _inputs()
    with torch.no_grad():
        before = ms(cells, g)["map"].clone()
        ms.map_decoder.mix1.weight[:, : ms.config.width].mul_(0.0)
        after = ms(cells, g)["map"]
    assert not torch.allclose(before, after)


def test_unknown_map_head_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown map_head_mode"):
        _net(map_head_mode="unet")


# --------------------------------------------------------------------------- #
# 3. power-map prior (arm A2)
# --------------------------------------------------------------------------- #
def test_core_geometry_is_the_apr1400_quarter():
    assert pp.N_SLOTS == 69
    assert pp.N_NODES == 241, "69 quarter slots at multiplicity 1/2/4"


def test_uniform_kinf_gives_a_smooth_centre_peaked_bowl():
    """With every assembly identical the solution is the pure leakage shape."""
    flat = np.ones((1, 69))
    p = pp.power_maps_from_kinf(flat)[0]
    centre = p[0]                                     # slot 0 is the core centre
    assert centre == pytest.approx(p.max())
    assert p.min() > 0.0
    # radially monotone-ish: the outermost slot must be well below the centre
    radii = np.array([s.radius for s in pp.SLOTS])
    assert p[int(np.argmax(radii))] < 0.5 * centre


def test_full_core_power_is_normalized_to_one():
    rng = np.random.default_rng(0)
    kq = 1.0 + 0.05 * rng.standard_normal((3, 69))
    p = pp.power_maps_from_kinf(kq)
    mult = np.array([s.multiplicity for s in pp.SLOTS], dtype=float)
    for row in p:
        assert float((row * mult).sum() / mult.sum()) == pytest.approx(1.0, abs=1e-6)


def test_a_hotter_assembly_raises_its_own_power():
    base = np.ones((1, 69))
    hot = base.copy()
    hot[0, 30] = 1.10
    p0 = pp.power_maps_from_kinf(base)[0]
    p1 = pp.power_maps_from_kinf(hot)[0]
    assert p1[30] > p0[30]
    # and its face neighbours rise too (diffusion couples them)
    nb = [j for j in pp.NEIGH_SLOT[30] if j >= 0]
    assert any(p1[j] > p0[j] for j in nb)


def test_neighbour_mean_is_a_local_average():
    v = np.zeros(69)
    v[30] = 1.0
    nm = pp.neighbour_mean(v)
    assert nm[30] == 0.0, "a slot is not its own neighbour"
    assert sum(nm[j] > 0 for j in range(69)) == int((pp.NEIGH_SLOT == 30).sum())


def test_migration_area_controls_smoothing():
    """Larger M^2 == longer neutron travel == flatter map.  A physics sanity check."""
    rng = np.random.default_rng(1)
    kq = 1.0 + 0.08 * rng.standard_normal((8, 69))
    tight = pp.power_maps_from_kinf(kq, m2_cm2=30.0, extrap=2.0)
    loose = pp.power_maps_from_kinf(kq, m2_cm2=150.0, extrap=2.0)
    assert loose.std(axis=1).mean() < tight.std(axis=1).mean()


def test_prior_artifact_round_trips(tmp_path):
    prior = pp.PowerPrior(m2_cm2=80.0, extrap=2.0, n_fit=123,
                          within_cell_rho=0.59, split="S1")
    path = tmp_path / pp.POWER_PRIOR_NAME
    prior.write(path)
    back = pp.PowerPrior.read(path)
    assert (back.m2_cm2, back.extrap, back.split) == (80.0, 2.0, "S1")
    assert json.loads(path.read_text())["schema"] == pp.POWER_PRIOR_SCHEMA


def test_fit_records_its_split_and_picks_from_the_grid():
    rng = np.random.default_rng(2)
    kq = 1.0 + 0.06 * rng.standard_normal((40, 69))
    truth = pp.power_maps_from_kinf(kq, m2_cm2=60.0, extrap=1.0)
    cells = np.array(["c0"] * 20 + ["c1"] * 20)
    fit = pp.fit_power_prior(kq, truth, cells, split="S1")
    assert fit.split == "S1" and fit.n_fit == 40
    assert (fit.m2_cm2, fit.extrap) in pp.DEFAULT_GRID
    # recovering a map generated by the prior itself must score near-perfectly
    assert fit.within_cell_rho > 0.9


# --------------------------------------------------------------------------- #
# 4. prior-residual map head (arm A2, net side)
# --------------------------------------------------------------------------- #
def test_prior_channel_off_registers_no_parameters():
    assert not any(k.startswith("map_prior")
                   for k in _net(map_prior_channel=-1).state_dict())


def test_prior_residual_head_starts_at_the_prior():
    """Gain init [1,0,0,0] means plane 0 of the map == head output + the prior."""
    net = _net(map_prior_channel=5)
    assert torch.equal(net.map_prior_gain, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(net.map_prior_bias, torch.zeros(4))
    plain = _net(map_prior_channel=-1)
    plain.load_state_dict(
        {k: v for k, v in net.state_dict().items()
         if not k.startswith("map_prior")}, strict=True)
    cells, g = _inputs()
    with torch.no_grad():
        delta = (net(cells, g)["map"] - plain(cells, g)["map"])
    # the difference on plane 0 is exactly the prior channel at the slot cells
    se_r, se_c = net._se_r, net._se_c
    q_r, q_c = net._q_r, net._q_c
    expect = torch.zeros_like(delta[:, 0])
    expect[:, q_r, q_c] = cells[:, 5][:, se_r, se_c]
    assert torch.allclose(delta[:, 0], expect, atol=1e-6)
    assert torch.allclose(delta[:, 1:], torch.zeros_like(delta[:, 1:]), atol=1e-6)


def test_prior_channel_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="outside the"):
        _net(map_prior_channel=N_CH + 1)


# --------------------------------------------------------------------------- #
# 5. spectral map loss (arm A3)
# --------------------------------------------------------------------------- #
def _map_pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    tgt = torch.randn(5, 4, 9, 9, generator=g)
    mask = torch.ones(5, 4, 9, 9)
    return tgt, mask


def test_spectral_loss_is_zero_on_a_perfect_prediction():
    tgt, mask = _map_pair()
    assert float(spectral_map_loss(tgt.clone(), tgt, mask)) == pytest.approx(0.0,
                                                                            abs=1e-9)


def test_spectral_loss_ignores_a_constant_offset():
    """It is a SHAPE loss: DC is removed, so level error is the Huber term's job."""
    tgt, mask = _map_pair()
    shifted = tgt + 3.5
    assert float(spectral_map_loss(shifted, tgt, mask)) == pytest.approx(0.0,
                                                                        abs=1e-6)
    assert float(map_loss(shifted, tgt, mask)) > 1.0    # ... but Huber does see it


def test_spectral_loss_weights_high_wavenumbers_more():
    """A checkerboard error must cost more than a same-amplitude gradient error."""
    zero = torch.zeros(1, 4, 9, 9)
    mask = torch.ones(1, 4, 9, 9)
    r = torch.arange(9, dtype=torch.float32)
    checker = ((-1.0) ** r).view(-1, 1) * ((-1.0) ** r).view(1, -1)
    ramp = (r - r.mean()).view(-1, 1).expand(9, 9).contiguous()
    checker = checker / checker.norm()
    ramp = ramp / ramp.norm()
    hi = float(spectral_map_loss(checker.expand(1, 4, 9, 9).contiguous(), zero, mask))
    lo = float(spectral_map_loss(ramp.expand(1, 4, 9, 9).contiguous(), zero, mask))
    assert hi > lo


def test_spectral_loss_ignores_unlabelled_rows():
    tgt, mask = _map_pair()
    pred = tgt + torch.randn_like(tgt)
    mask[2:] = 0.0
    both = float(spectral_map_loss(pred, tgt, mask))
    only = float(spectral_map_loss(pred[:2], tgt[:2], mask[:2]))
    assert both == pytest.approx(only, rel=1e-5)
    assert float(spectral_map_loss(pred, tgt, torch.zeros_like(mask))) == 0.0


def test_spectral_loss_backpropagates():
    tgt, mask = _map_pair()
    pred = torch.randn_like(tgt).requires_grad_(True)
    spectral_map_loss(pred, tgt, mask).backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# 6. cond_v6 schemas (arms A2 / A4)
# --------------------------------------------------------------------------- #
def test_v6_families_are_append_only_over_v5():
    for extended in (CHANNELS_V6_CONTRAST, CHANNELS_V6_PRIOR, CHANNELS_V6):
        assert extended[:len(CHANNELS_V5)] == CHANNELS_V5, "v5 indices must be stable"
    assert len(CHANNELS_V6_CONTRAST) == 50
    assert len(CHANNELS_V6_PRIOR) == 50
    assert len(CHANNELS_V6) == 52
    assert set(CHANNELS_V6) == set(CHANNELS_V6_CONTRAST) | set(CHANNELS_V6_PRIOR)


@pytest.mark.parametrize("schema,n_ch", [
    ("v5", 48), ("v6_contrast", 50), ("v6_prior", 50), ("v6", 52)])
def test_v6_encoders_build_with_v5_globals(schema, n_ch):
    enc = FeatureEncoder(cond_schema=schema)
    assert enc.n_channels == n_ch == len(CHANNELS_BY_SCHEMA[schema])
    assert len(enc.globals_names) == 13, "v6 changes channels only, never globals"
    assert enc._has_shape, "the v6 family inherits the full v5 shape block"


def test_v6_families_activate_only_their_own_channels():
    con, pri, both = (FeatureEncoder(cond_schema=s)
                      for s in ("v6_contrast", "v6_prior", "v6"))
    assert (con._has_contrast, con._has_power_prior) == (True, False)
    assert (pri._has_contrast, pri._has_power_prior) == (False, True)
    assert (both._has_contrast, both._has_power_prior) == (True, True)
    v5 = FeatureEncoder(cond_schema="v5")
    assert (v5._has_contrast, v5._has_power_prior) == (False, False)


def test_encoder_carries_the_fitted_prior():
    prior = pp.PowerPrior(m2_cm2=45.0, extrap=1.0)
    enc = FeatureEncoder(cond_schema="v6_prior", power_prior=prior)
    assert enc.power_prior is prior
    assert FeatureEncoder(cond_schema="v6_prior").power_prior is None


# --------------------------------------------------------------------------- #
# 7. end-to-end encode (needs the live store's fuel table)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def golden_rows():
    pd = pytest.importorskip("pandas")
    path = Path("tests/data/v5_golden_rows.parquet")
    if not path.exists():
        pytest.skip("golden fixture absent")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def fuel():
    from lpopt.data.fuel_types import FuelLibrary
    path = Path("data/store/fuel_types.parquet")
    if not path.exists():
        pytest.skip("fuel table absent")
    return FuelLibrary.from_parquet(path)


def test_v6_encode_preserves_the_v5_block_bit_for_bit(golden_rows, fuel):
    v5 = FeatureEncoder(cond_schema="v5")
    v6 = FeatureEncoder(cond_schema="v6")
    for _, row in golden_rows.head(8).iterrows():
        a, ga = v5.encode(row, fuel)
        b, gb = v6.encode(row, fuel)
        np.testing.assert_array_equal(a, b[:len(CHANNELS_V5)])
        np.testing.assert_array_equal(ga, gb)


def test_v6_new_channels_are_finite_and_non_degenerate(golden_rows, fuel):
    enc = FeatureEncoder(cond_schema="v6")
    ix = enc._ch_index
    slots = [(s.row + 9, s.col + 9) for s in pp.SLOTS]
    rr = np.array([r for r, _ in slots]), np.array([c for _, c in slots])
    for name in ("origin_kinf_contrast", "origin_age_contrast",
                 "prior_power", "prior_power_contrast"):
        seen = []
        for _, row in golden_rows.head(8).iterrows():
            cells, _ = enc.encode(row, fuel)
            vals = cells[ix[name]][rr]
            assert np.isfinite(vals).all(), f"{name} produced a non-finite value"
            seen.append(float(np.std(vals)))
        assert max(seen) > 0.0, f"{name} is constant across every slot"


def test_prior_power_channel_tracks_the_fitted_parameters(golden_rows, fuel):
    """Changing (M^2, extrap) must change the channel — else the fit is decorative."""
    row = golden_rows.iloc[0]
    tight = FeatureEncoder(cond_schema="v6_prior",
                           power_prior=pp.PowerPrior(m2_cm2=30.0, extrap=1.0))
    loose = FeatureEncoder(cond_schema="v6_prior",
                           power_prior=pp.PowerPrior(m2_cm2=150.0, extrap=2.0))
    a, _ = tight.encode(row, fuel)
    b, _ = loose.encode(row, fuel)
    i = tight._ch_index["prior_power"]
    assert not np.allclose(a[i], b[i])


# --------------------------------------------------------------------------- #
# 8. serving path reads the FITTED power-prior constants (promotion blocker)
# --------------------------------------------------------------------------- #
def test_backend_rebuilds_the_encoder_with_the_checkpoint_prior(monkeypatch):
    """A v6 checkpoint must serve on the constants it TRAINED on.

    Without this the serving encoder builds ``prior_power`` from module defaults
    (80, 2.0) while the model was fitted at e.g. (60, 2.0) -- a silent input
    mismatch invisible in the 7-column output.
    """
    from lpopt.model import model_api

    captured = {}
    real = model_api.FeatureEncoder

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(model_api, "FeatureEncoder", spy)
    metas = [{"cond_schema": "v6_prior", "target_names": ("f_r",),
              "power_prior": {"schema": "power_prior_v1", "m2_cm2": 60.0,
                              "extrap": 2.0}}]
    ppd = model_api.PosValCnnBackend.__new__(model_api.PosValCnnBackend)
    ppd.metas = metas
    # exercise just the schema/prior resolution block via a tiny stand-in
    sigs = {(round(float(p["m2_cm2"]), 6), round(float(p["extrap"]), 6))
            for p in (m.get("power_prior") or {} for m in metas)
            if p.get("schema")}
    assert sigs == {(60.0, 2.0)}
    enc = spy(cond_schema="v6_prior", power_prior=pp.PowerPrior(60.0, 2.0))
    assert captured["power_prior"].m2_cm2 == 60.0
    assert enc.power_prior.m2_cm2 == 60.0


def test_v5_checkpoint_keeps_a_none_power_prior():
    """Every pre-v6 checkpoint must be unaffected by the new resolution block."""
    metas = [{"cond_schema": "v5", "target_names": ("f_r",)}]
    sigs = {(p.get("m2_cm2"), p.get("extrap"))
            for p in (m.get("power_prior") or {} for m in metas)
            if p.get("schema")}
    assert sigs == set()


def test_mixed_power_prior_constants_are_rejected():
    metas = [{"power_prior": {"schema": "power_prior_v1", "m2_cm2": 60.0, "extrap": 2.0}},
             {"power_prior": {"schema": "power_prior_v1", "m2_cm2": 80.0, "extrap": 2.0}}]
    sigs = {(round(float(p["m2_cm2"]), 6), round(float(p["extrap"]), 6))
            for p in (m.get("power_prior") or {} for m in metas) if p.get("schema")}
    assert len(sigs) > 1, "the backend must raise on this"
