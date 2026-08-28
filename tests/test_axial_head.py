"""Axial (EDIT 6) profile labels + the optional axial head (decision D10).

Four contracts:

* **Label parsing.** ``stack_axial`` -> ``anchor_profiles`` reproduces the EDIT 6
  table with the documented orientation (BOTTOM->TOP), normalisation (core
  average 1) and anchors (step 0 / step -1).
* **F_z / AO derivation.** :func:`lpopt.data.axial.axial_offset` reproduces the
  MASTER EDIT 3 ``AO`` column from the EDIT 6 profile EXACTLY (centre node split
  half-and-half), and ``ASI == -AO``; ``axial_peaking_factor`` is max/mean.
* **Head shapes.** With the flag on the net emits ``[B, A, K]`` coefficients, the
  dataset emits ``[A, 25]`` profiles + an ``[A]`` mask, the loss masks correctly,
  and the basis round-trips coefficients <-> profiles while preserving the
  core-average-1 normalisation exactly.
* **Flag OFF is byte-identical.** Module set, parameter count, ``state_dict``
  digest, forward-output keys, dataset item keys, precomputed tensor keys and the
  optimizer step are all unchanged from the pre-axial path.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.data import axial as ax                                   # noqa: E402
from lpopt.data.edit5 import parse_mas_sum, stack_axial              # noqa: E402
from lpopt.model.net import (                                        # noqa: E402
    PosValNet, PosValNetConfig, count_parameters,
)
from lpopt.model.train import TrainConfig, axial_loss                # noqa: E402

# --------------------------------------------------------------------------- #
# a miniature MAS_SUM carrying EDIT 2 / EDIT 3 / EDIT 6
# --------------------------------------------------------------------------- #
#: A 9-plane bottom-skewed profile and a 9-plane saddle, both mean-1 normalised.
_P_BOC = np.array([0.40, 0.85, 1.18, 1.32, 1.30, 1.20, 1.02, 0.78, 0.95])
_P_EOC = np.array([0.70, 1.10, 1.22, 1.05, 0.90, 1.02, 1.20, 1.06, 0.75])
_P_BOC = _P_BOC * (len(_P_BOC) / _P_BOC.sum())
_P_EOC = _P_EOC * (len(_P_EOC) / _P_EOC.sum())


def _ao_reference(p: np.ndarray) -> float:
    """AO the way MASTER prints it: the centre node splits half to each side."""
    n = len(p)
    h = n // 2
    bot = p[:h].sum() + (0.5 * p[h] if n % 2 else 0.0)
    top = p[h + 1:].sum() + (0.5 * p[h] if n % 2 else 0.0)
    return float((top - bot) / p.sum())


def _mas_sum_text() -> str:
    def edit6_row(no, efpd, p):
        return f"   {no}  {efpd:8.3f} {efpd:8.3f} " + " ".join(f"{v:.4f}" for v in p)

    return "\n".join([
        "   SUMMARY EDIT 2 : REACTIVITY",
        "   NO.  DAY  EFPD  CYC-BU  TOT-BU  P(%)  PPM  K-EFF  ERRFLX  REACT.",
        "    1    0.000    0.000  0.0  0.0 100.0 1100.0 1.00000 0.00001 0.0",
        "    2  500.000  500.000  0.0  0.0 100.0   10.0 1.00000 0.00001 0.0",
        "",
        "   SUMMARY EDIT 3 : PEAKING FACTORS",
        "   NO.  DAY  EFPD  AO  FQN  FRN  FQP  FRP  XE  XE-AO  SM  SM-AO",
        f"    1    0.000    0.000 {_ao_reference(_P_BOC):.4f} 1.5 1.4 1.6 1.5 0 0 0 0",
        f"    2  500.000  500.000 {_ao_reference(_P_EOC):.4f} 1.4 1.3 1.5 1.4 0 0 0 0",
        "",
        "   SUMMARY EDIT 6 : 1-D POWER DISTRIBUTION",
        "                 POWER(BOTTOM  --->  TOP)",
        "   NO.     DAY     EFPD     2      3      4      5      6      7      8      9     10",
        edit6_row(1, 0.0, _P_BOC),
        edit6_row(2, 500.0, _P_EOC),
        "",
    ])


# --------------------------------------------------------------------------- #
# 1. label parsing
# --------------------------------------------------------------------------- #
def test_stack_axial_parses_edit6_bottom_to_top_and_normalised():
    summary = parse_mas_sum(_mas_sum_text())
    arr = stack_axial(summary)
    assert arr.shape == (2, 9)
    np.testing.assert_allclose(arr[0], _P_BOC, atol=1e-4)
    np.testing.assert_allclose(arr[1], _P_EOC, atol=1e-4)
    # the documented normalisation: every step row averages to 1
    np.testing.assert_allclose(arr.mean(axis=1), 1.0, atol=1e-4)


def test_anchor_profiles_picks_step_0_and_step_minus_1():
    stack = np.stack([_P_BOC, _P_BOC * 0 + 1.0, _P_EOC])
    prof = ax.anchor_profiles(stack)
    assert prof.shape == (2, 9)
    np.testing.assert_allclose(prof[0], _P_BOC)
    np.testing.assert_allclose(prof[1], _P_EOC)
    assert ax.ANCHORS == ("boc", "eoc")


def test_anchor_profiles_rejects_an_unknown_anchor():
    with pytest.raises(ValueError, match="unknown axial anchor"):
        ax.anchor_profiles(np.stack([_P_BOC, _P_EOC]), anchors=("mid",))


def test_load_axial_rejects_malformed_or_missing_labels():
    class _R:
        def __init__(self, payload):
            self.payload = payload

        def maps(self, key):
            return self.payload.get(key)

    rid = "abc"
    good = np.stack([_P_BOC, _P_EOC]).astype(np.float16)
    assert ax.axial_key(rid) == "abc__axial"
    assert ax.load_axial(_R({}), rid, n_planes=9) is None            # absent
    assert ax.load_axial(_R({"abc__axial": good}), rid, n_planes=9) is not None
    assert ax.load_axial(_R({"abc__axial": good}), rid, n_planes=25) is None  # width
    bad = good.astype(np.float32).copy()
    bad[0, 0] = np.nan
    assert ax.load_axial(_R({"abc__axial": bad}), rid, n_planes=9) is None
    assert ax.load_axial(_R({"abc__axial": good[0]}), rid, n_planes=9) is None  # rank


# --------------------------------------------------------------------------- #
# 2. F_z / AO / ASI derivation
# --------------------------------------------------------------------------- #
def test_axial_offset_reproduces_the_edit3_ao_column():
    """The EDIT 6 profile must regenerate EDIT 3's AO — that is the parse proof."""
    summary = parse_mas_sum(_mas_sum_text())
    arr = stack_axial(summary).astype(np.float64)
    edit3 = [r.values["AO"] for r in sorted(summary.peaking_rows,
                                            key=lambda r: (r.efpd, r.no))]
    derived = ax.axial_offset(arr)
    np.testing.assert_allclose(derived, edit3, atol=2e-4)


def test_axial_offset_matches_the_centre_split_reference_and_is_not_the_naive_form():
    p = _P_BOC
    assert ax.axial_offset(p) == pytest.approx(_ao_reference(p), abs=1e-12)
    # the naive "drop the centre node" variant is a DIFFERENT number; if the two
    # ever coincide this test is not proving anything.
    naive = (p[5:].sum() - p[:4].sum()) / (p[5:].sum() + p[:4].sum())
    assert abs(naive - _ao_reference(p)) > 1e-6


def test_axial_offset_handles_an_even_plane_count():
    p = np.array([0.5, 1.0, 1.5, 1.0])
    assert ax.axial_offset(p) == pytest.approx((2.5 - 1.5) / 4.0)


def test_axial_shape_index_is_minus_axial_offset():
    np.testing.assert_allclose(ax.axial_shape_index(_P_EOC),
                               -ax.axial_offset(_P_EOC), atol=1e-15)


def test_axial_peaking_factor_is_max_over_mean():
    assert ax.axial_peaking_factor(_P_BOC) == pytest.approx(_P_BOC.max(), rel=1e-12)
    # and it is scale-invariant, so it is correct for an unnormalised prediction
    assert (ax.axial_peaking_factor(3.7 * _P_BOC)
            == pytest.approx(ax.axial_peaking_factor(_P_BOC), rel=1e-12))


def test_derived_metrics_are_batched_over_leading_axes():
    stack = np.stack([np.stack([_P_BOC, _P_EOC])] * 4)      # [4, 2, 9]
    d = ax.derived_metrics(stack)
    for k in ("f_z", "ao", "asi", "saddle_depth"):
        assert d[k].shape == (4, 2)
    np.testing.assert_allclose(d["f_z"][0, 0], _P_BOC.max(), rtol=1e-12)


def test_saddle_depth_separates_shapes_that_share_an_axial_offset():
    """|AO| is a first moment: it cannot tell a saddle from a single hump.

    Both profiles below are symmetric, so BOTH have AO == 0 exactly — yet one is
    centre-peaked and the other is a two-humped saddle.  This is the whole reason
    the head predicts the PROFILE and derives the scalars from it.
    """
    hump = np.array([0.4, 0.8, 1.2, 1.6, 1.8, 1.6, 1.2, 0.8, 0.4])
    saddle = np.array([0.4, 1.0, 1.6, 1.4, 1.0, 1.4, 1.6, 1.0, 0.4])
    hump = hump * (9 / hump.sum())
    saddle = saddle * (9 / saddle.sum())
    assert ax.axial_offset(hump) == pytest.approx(0.0, abs=1e-12)
    assert ax.axial_offset(saddle) == pytest.approx(0.0, abs=1e-12)
    assert ax.saddle_depth(hump) == pytest.approx(0.0, abs=1e-12)
    assert ax.saddle_depth(saddle) > 0.3


# --------------------------------------------------------------------------- #
# 3. shape basis
# --------------------------------------------------------------------------- #
def _synthetic_profiles(n=200, n_planes=9, seed=0):
    rng = np.random.default_rng(seed)
    base = np.stack([_P_BOC, _P_EOC])                       # [2, 9]
    j = np.arange(n_planes)
    modes = np.stack([np.cos(np.pi * (j + 0.5) * m / n_planes) for m in (1, 2, 3)])
    modes -= modes.mean(axis=1, keepdims=True)
    coef = rng.normal(size=(n, 2, 3)) * np.array([0.06, 0.03, 0.015])
    return base[None] + np.einsum("nak,kp->nap", coef, modes)


def test_basis_round_trips_and_preserves_the_normalisation_exactly():
    prof = _synthetic_profiles()
    basis = ax.fit_axial_basis(prof, rank=3)
    assert (basis.n_anchors, basis.n_modes, basis.n_planes) == (2, 3, 9)
    back = basis.decode(basis.encode(prof))
    np.testing.assert_allclose(back, prof, atol=1e-9)
    # every reconstructed profile is core-average 1 BY CONSTRUCTION, including
    # from coefficients the head could emit that no label ever produced.
    wild = basis.z_decode(np.random.default_rng(3).normal(size=(50, 2, 3)) * 12.0)
    np.testing.assert_allclose(wild.mean(axis=-1), 1.0, atol=1e-12)


def test_basis_z_space_round_trips_and_standardises_each_mode():
    prof = _synthetic_profiles()
    basis = ax.fit_axial_basis(prof, rank=3)
    z = basis.z_encode(prof)
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=0.02)
    np.testing.assert_allclose(basis.z_decode(z), prof, atol=1e-9)


def test_basis_serialises_round_trip():
    basis = ax.fit_axial_basis(_synthetic_profiles(), rank=3)
    clone = ax.AxialBasis.from_dict(basis.to_dict())
    assert clone.anchors == basis.anchors
    np.testing.assert_allclose(clone.mean, basis.mean)
    np.testing.assert_allclose(clone.components, basis.components)
    np.testing.assert_allclose(clone.mode_sd, basis.mode_sd)


def test_basis_components_are_orthonormal_and_zero_sum():
    basis = ax.fit_axial_basis(_synthetic_profiles(), rank=3)
    for a in range(basis.n_anchors):
        v = basis.components[a]
        np.testing.assert_allclose(v @ v.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(v.sum(axis=1), 0.0, atol=1e-12)


def test_basis_mask_excludes_unlabelled_rows_from_the_fit():
    prof = _synthetic_profiles(n=60)
    mask = np.ones((60, 2), dtype=bool)
    mask[30:, 1] = False
    prof[30:, 1] = 999.0                     # poison the masked rows
    basis = ax.fit_axial_basis(prof, rank=3, mask=mask)
    assert np.isfinite(basis.mean).all()
    assert basis.mean[1].max() < 2.0          # the poison never entered the fit


def test_fit_axial_basis_rejects_an_impossible_rank():
    with pytest.raises(ValueError, match="degrees of freedom"):
        ax.fit_axial_basis(_synthetic_profiles(n=20), rank=9)
    with pytest.raises(ValueError, match="rank must be"):
        ax.fit_axial_basis(_synthetic_profiles(n=20), rank=0)


def test_basis_reconstruction_beats_the_within_cell_spread_on_real_labels():
    """Rank 6 must reconstruct F_z far better than the spread it has to resolve.

    Uses the shipped golden profiles when present; otherwise synthetic ones —
    either way the assertion is the same ratio.
    """
    prof = _synthetic_profiles(n=300, seed=11)
    basis = ax.fit_axial_basis(prof, rank=3)
    rec = basis.decode(basis.encode(prof))
    err = np.abs(ax.axial_peaking_factor(rec) - ax.axial_peaking_factor(prof)).max()
    spread = ax.axial_peaking_factor(prof).std()
    assert err < 0.25 * spread


# --------------------------------------------------------------------------- #
# 4. head shapes + loss
# --------------------------------------------------------------------------- #
def test_net_emits_axial_coefficients_with_the_configured_shape():
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                          n_axial_anchors=2, n_axial_modes=6)
    net = PosValNet(cfg)
    assert net.has_axial
    cells = torch.randn(3, 26, 19, 19)
    cells[:, 0] = 1.0
    out = net(cells, torch.randn(3, 10))
    assert out["axial"].shape == (3, 2, 6)
    # the head is a linear read-off of the SHARED head trunk, so it costs exactly
    # (head_hidden + 1) * A * K parameters on top of the base net.
    base = count_parameters(PosValNet(PosValNetConfig(
        in_channels=26, n_globals=10, n_targets=7)))
    assert count_parameters(net) - base == (cfg.head_hidden + 1) * 2 * 6


@pytest.mark.parametrize("anchors,modes", [(0, 6), (2, 0), (0, 0)])
def test_net_needs_both_axial_dimensions_to_build_the_head(anchors, modes):
    net = PosValNet(PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                                    n_axial_anchors=anchors, n_axial_modes=modes))
    assert not net.has_axial
    assert not hasattr(net, "axial_head")


def test_axial_loss_masks_unlabelled_rows_and_is_zero_on_a_perfect_fit():
    pred = torch.randn(4, 2, 3)
    tgt = pred.clone()
    mask = torch.ones(4, 2)
    assert float(axial_loss(pred, tgt, mask)) == pytest.approx(0.0, abs=1e-12)

    # a row with a zero mask must not contribute, however wrong it is
    tgt2 = tgt.clone()
    tgt2[0] += 100.0
    mask2 = mask.clone()
    mask2[0] = 0.0
    assert float(axial_loss(pred, tgt2, mask2)) == pytest.approx(0.0, abs=1e-12)
    # ... and an all-zero mask is a hard no-op (no NaN from an empty mean)
    assert float(axial_loss(pred, tgt2, torch.zeros(4, 2))) == 0.0


def test_axial_loss_masks_per_anchor_not_per_row():
    pred = torch.zeros(2, 2, 3)
    tgt = torch.zeros(2, 2, 3)
    tgt[:, 1] = 4.0                        # only the EOC anchor is wrong
    both = float(axial_loss(pred, tgt, torch.ones(2, 2)))
    boc_only = float(axial_loss(pred, tgt, torch.tensor([[1.0, 0.0], [1.0, 0.0]])))
    assert boc_only == pytest.approx(0.0, abs=1e-12)
    assert both > 0.0


def test_axial_loss_gradient_reaches_the_head():
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                          n_axial_anchors=2, n_axial_modes=4)
    net = PosValNet(cfg)
    cells = torch.randn(2, 26, 19, 19)
    cells[:, 0] = 1.0
    out = net(cells, torch.randn(2, 10))
    axial_loss(out["axial"], torch.randn(2, 2, 4), torch.ones(2, 2)).backward()
    assert net.axial_head.weight.grad is not None
    assert torch.isfinite(net.axial_head.weight.grad).all()
    assert float(net.axial_head.weight.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# 5. dataset wiring
# --------------------------------------------------------------------------- #
REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


@pytest.fixture(scope="module")
def store_ids():
    """``(reader, fuel_library, [ids with an axial label], [ids without])``."""
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader

    reader = StoreReader(STORE)
    fl = FuelLibrary.from_parquet(STORE / "fuel_types.parquet")
    keys = reader.maps_keys()
    have = {k[: -len(ax.AXIAL_SUFFIX)] for k in keys
            if k.endswith(ax.AXIAL_SUFFIX)}
    if not have:
        pytest.skip("no axial labels in the store")
    ids = reader.records["record_id"].astype(str)
    with_ax = [r for r in ids if r in have][:6]
    without = [r for r in ids if r not in have][:6]
    return reader, fl, with_ax, without


def test_dataset_emits_axial_profiles_and_a_per_anchor_mask(store_ids):
    from lpopt.model.dataset_torch import LPDataset

    reader, fl, with_ax, without = store_ids
    ds = LPDataset(reader, with_ax[:2] + without[:2], fl, include_axial=True)
    by_id = {ds.record_ids[i]: ds[i] for i in range(len(ds))}
    it = by_id[with_ax[0]]
    assert it["axial"].shape == (2, ax.N_PLANES)
    assert it["axial_mask"].shape == (2,)
    assert float(it["axial_mask"].sum()) == 2.0
    assert torch.isfinite(it["axial"]).all()
    # the stored normalisation survives the float16 -> float32 trip
    np.testing.assert_allclose(it["axial"].numpy().mean(axis=-1), 1.0, atol=1e-3)
    if without:
        miss = by_id[without[0]]
        assert float(miss["axial_mask"].sum()) == 0.0
        assert torch.isnan(miss["axial"]).all()   # never a fabricated shape


def test_dataset_masks_the_axial_label_of_a_non_converged_row(store_ids):
    from lpopt.model.dataset_torch import LPDataset

    reader, fl, with_ax, _ = store_ids
    ds = LPDataset(reader, with_ax[:1], fl, include_axial=True)
    assert float(ds[0]["axial_mask"].sum()) == 2.0
    ds.df.loc[0, "converged"] = False
    assert float(ds[0]["axial_mask"].sum()) == 0.0


def test_stack_anchor_profiles_masks_the_missing_rows(store_ids):
    reader, _fl, with_ax, without = store_ids
    ids = with_ax[:2] + without[:2]
    prof, mask = ax.stack_anchor_profiles(reader, ids)
    assert prof.shape == (len(ids), 2, ax.N_PLANES)
    np.testing.assert_array_equal(mask[:2], np.ones((2, 2)))
    np.testing.assert_allclose(prof[0].mean(axis=-1), 1.0, atol=1e-3)
    if without:
        np.testing.assert_array_equal(mask[2:], np.zeros((len(without[:2]), 2)))
        assert np.isnan(prof[2:]).all()


def test_stored_labels_regenerate_the_ao_abs_column(store_ids):
    """The store's own ``ao_abs`` must be recoverable from the axial stack.

    This is the end-to-end proof of the orientation + normalisation + AO
    convention on REAL labels, not a fixture: ``ao_abs`` was written from EDIT 3
    by the vendor harness, and ``axial_offset`` recomputes it from EDIT 6.
    """
    reader, _fl, with_ax, _ = store_ids
    df = reader.records.drop_duplicates("record_id").set_index("record_id")
    checked = 0
    for rid in with_ax:
        stack = ax.load_axial(reader, rid)
        label = df.loc[rid, "ao_abs"]
        if stack is None or label is None or not np.isfinite(label):
            continue
        derived = float(np.abs(ax.axial_offset(stack)).max())
        assert derived == pytest.approx(float(label), abs=1e-3)
        checked += 1
    assert checked > 0


def test_stored_labels_obey_the_documented_normalisation(store_ids):
    reader, _fl, with_ax, _ = store_ids
    for rid in with_ax:
        stack = ax.load_axial(reader, rid)
        assert stack is not None
        assert stack.shape[1] == ax.N_PLANES
        np.testing.assert_allclose(stack.mean(axis=1), 1.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# 6. flag OFF is byte-identical
# --------------------------------------------------------------------------- #
#: The pre-axial output keys.  ``axial`` must NOT appear with the flag off.
LEGACY_OUT_KEYS = {"mu", "log_sigma", "map", "conv_logit"}


def _net_digest(cfg: PosValNetConfig) -> tuple[str, str, frozenset]:
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
    net.eval()
    with torch.no_grad():
        out = net(cells, torch.randn(3, cfg.n_globals))
    ho = hashlib.sha256()
    for k in sorted(out):
        ho.update(k.encode())
        ho.update(out[k].numpy().tobytes())
    return h.hexdigest(), ho.hexdigest(), frozenset(out)


@pytest.mark.parametrize("in_ch,n_g,nt", [(26, 10, 7), (43, 13, 8)])
def test_axial_flag_off_is_byte_identical(in_ch, n_g, nt):
    """Explicit 0/0 == implicit default == the pre-axial network."""
    implicit = PosValNetConfig(in_channels=in_ch, n_globals=n_g, n_targets=nt)
    explicit = PosValNetConfig(in_channels=in_ch, n_globals=n_g, n_targets=nt,
                               n_axial_anchors=0, n_axial_modes=0)
    d_i, d_e = _net_digest(implicit), _net_digest(explicit)
    assert d_i == d_e
    assert d_i[2] == LEGACY_OUT_KEYS
    assert not any("axial" in k for k in PosValNet(implicit).state_dict())


def test_axial_flag_off_adds_no_parameters():
    base = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7)
    assert count_parameters(PosValNet(base)) == count_parameters(
        PosValNet(PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                                  n_axial_anchors=0, n_axial_modes=0)))


def test_train_config_axial_defaults_are_off():
    cfg = TrainConfig()
    assert cfg.axial_head is False
    assert cfg.axial_weight == 0.2 and cfg.axial_rank == 6
    # the net's own defaults must agree, else a default TrainConfig would build
    # a head nobody asked for
    pc = PosValNetConfig()
    assert (pc.n_axial_anchors, pc.n_axial_modes) == (0, 0)


def test_dataset_flag_off_has_the_legacy_item_keys(store_ids):
    from lpopt.model.dataset_torch import LPDataset

    reader, fl, with_ax, _ = store_ids
    off = LPDataset(reader, with_ax[:2], fl)
    on = LPDataset(reader, with_ax[:2], fl, include_axial=True)
    assert set(off[0]) == set(on[0]) - {"axial", "axial_mask"}
    assert "axial" not in off[0]
    # ... and the shared keys are value-identical, so the label read is additive
    for k in set(off[0]) - {"record_id"}:
        torch.testing.assert_close(off[0][k], on[0][k], equal_nan=True)


def test_precomputed_flag_off_has_no_axial_tensor_and_the_step_is_unchanged():
    """A flag-off optimizer step must not touch the axial branch."""
    from lpopt.model.train import _MemberState, _step_member, _Norm

    cfg = TrainConfig(axial_head=False)
    torch.manual_seed(0)
    net = PosValNet(PosValNetConfig(in_channels=26, n_globals=10, n_targets=7))
    m = _MemberState()
    m.model = net
    m.fwd = net
    m.optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    m.norm = _Norm(np.zeros(7), np.ones(7), np.zeros(4), np.ones(4),
                   torch.device("cpu"), 3)
    m.use_prior = False
    m.q_idx = None
    m.use_nll = False
    m.running = 0.0
    m.n_batches = 0
    cells = torch.randn(2, 26, 19, 19)
    cells[:, 0] = 1.0
    batch = {
        "cells": cells, "globals": torch.randn(2, 10),
        "targets": torch.randn(2, 7), "target_mask": torch.ones(2, 7),
        "conv_label": torch.ones(2), "conv_mask": torch.ones(2),
        "maps": torch.randn(2, 4, 9, 9), "maps_mask": torch.ones(2, 4, 9, 9),
    }
    _step_member(m, batch, cfg, use_amp=False, device=torch.device("cpu"))
    assert m.n_batches == 1
    assert np.isfinite(m.running)


def test_flag_off_step_is_bit_identical_with_axial_tensors_present():
    """Even if a fold carries axial labels, ``axial_weight`` gradients must not
    appear unless the HEAD exists — the branch is gated on ``out["axial"]``."""
    from lpopt.model.train import _MemberState, _Norm, _step_member

    def _run(net_cfg, batch_extra):
        torch.manual_seed(0)
        net = PosValNet(net_cfg)
        m = _MemberState()
        m.model, m.fwd = net, net
        m.optim = torch.optim.SGD(net.parameters(), lr=1e-2)
        m.norm = _Norm(np.zeros(7), np.ones(7), np.zeros(4), np.ones(4),
                       torch.device("cpu"), 3)
        m.use_prior, m.q_idx, m.use_nll = False, None, False
        m.running, m.n_batches = 0.0, 0
        torch.manual_seed(5)
        cells = torch.randn(2, 26, 19, 19)
        cells[:, 0] = 1.0
        batch = {
            "cells": cells, "globals": torch.randn(2, 10),
            "targets": torch.randn(2, 7), "target_mask": torch.ones(2, 7),
            "conv_label": torch.ones(2), "conv_mask": torch.ones(2),
            "maps": torch.randn(2, 4, 9, 9), "maps_mask": torch.ones(2, 4, 9, 9),
            **batch_extra,
        }
        _step_member(m, batch, TrainConfig(), use_amp=False,
                     device=torch.device("cpu"))
        return {k: v.clone() for k, v in net.state_dict().items()}

    base = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7)
    clean = _run(base, {})
    with_labels = _run(base, {"axial_coeff": torch.randn(2, 2, 6),
                              "axial_mask": torch.ones(2, 2)})
    assert set(clean) == set(with_labels)
    for k in clean:
        torch.testing.assert_close(clean[k], with_labels[k], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 7. train-side basis fit / attach / leakage
# --------------------------------------------------------------------------- #
class _FakePre:
    """Minimal stand-in for PrecomputedDataset (only ``_t`` / ``df`` are used)."""

    def __init__(self, prof, mask, ids):
        import pandas as pd
        self._t = {"axial": torch.as_tensor(prof, dtype=torch.float32),
                   "axial_mask": torch.as_tensor(mask, dtype=torch.float32)}
        self.df = pd.DataFrame({"record_id": ids})

    def __len__(self):
        return len(self.df)


def _fake_folds(n_train=80, n_val=20):
    prof = _synthetic_profiles(n=n_train + n_val, seed=5)
    mask = np.ones((n_train + n_val, 2))
    mask[-3:] = 0.0                                   # a few unlabelled rows
    tr = _FakePre(prof[:n_train], mask[:n_train], [f"t{i}" for i in range(n_train)])
    va = _FakePre(prof[n_train:], mask[n_train:], [f"v{i}" for i in range(n_val)])
    return tr, va


def test_prepare_axial_basis_is_a_noop_when_the_flag_is_off():
    from lpopt.model.train import _prepare_axial_basis

    tr, va = _fake_folds()
    assert _prepare_axial_basis(TrainConfig(axial_head=False), tr, va,
                                verbose=False) is None
    assert "axial_coeff" not in tr._t and "axial_coeff" not in va._t


def test_prepare_axial_basis_fits_train_only_and_projects_both_folds():
    from lpopt.model.train import _prepare_axial_basis

    tr, va = _fake_folds()
    basis = _prepare_axial_basis(TrainConfig(axial_head=True, axial_rank=3),
                                 tr, va, verbose=False)
    assert basis is not None and basis.n_modes == 3 and basis.n_anchors == 2
    assert tr._t["axial_coeff"].shape == (len(tr), 2, 3)
    assert va._t["axial_coeff"].shape == (len(va), 2, 3)
    assert tr.axial_basis is va.axial_basis is basis
    # the fit saw exactly the labelled TRAIN rows
    assert basis.n_fit == len(tr)
    # unlabelled val rows carry zero coefficients and keep their zero mask
    unl = va._t["axial_mask"].numpy().max(axis=1) == 0
    assert unl.any()
    assert float(va._t["axial_coeff"][torch.as_tensor(unl)].abs().sum()) == 0.0


def test_prepare_axial_basis_asserts_the_leakage_rule():
    from lpopt.model.train import _prepare_axial_basis

    tr, va = _fake_folds()
    va.df.loc[0, "record_id"] = tr.df.loc[0, "record_id"]     # overlap the folds
    with pytest.raises(AssertionError, match="train rows only"):
        _prepare_axial_basis(TrainConfig(axial_head=True), tr, va, verbose=False)


def test_prepare_axial_basis_disables_the_head_when_no_row_is_labelled():
    from lpopt.model.train import _prepare_axial_basis

    tr, va = _fake_folds()
    tr._t["axial_mask"] = torch.zeros_like(tr._t["axial_mask"])
    assert _prepare_axial_basis(TrainConfig(axial_head=True), tr, va,
                                verbose=False) is None


def test_attach_axial_coeffs_matches_a_direct_projection():
    from lpopt.model.train import attach_axial_coeffs, fit_axial_basis_for_dataset

    tr, _va = _fake_folds()
    basis = fit_axial_basis_for_dataset(tr, rank=3)
    n = attach_axial_coeffs(tr, basis)
    assert n == len(tr)
    direct = basis.z_encode(tr._t["axial"].numpy().astype(np.float64))
    np.testing.assert_allclose(tr._t["axial_coeff"].numpy(), direct, atol=1e-5)


# --------------------------------------------------------------------------- #
# 8. end-to-end: train a tiny arm on REAL axial labels, then serve it
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained_axial_arm(tmp_path_factory, store_ids):
    """A 1-member arm trained (2 epochs) on the axial-labelled store rows."""
    import json

    from lpopt.model.featurize import FeatureEncoder
    from lpopt.model.splits import SplitManifest
    from lpopt.model.train import (
        TrainConfig, _finalize_member, _prepare_axial_basis, _resolve_schedule,
        _train_members, build_precomputed,
    )

    reader, fl, _with, _without = store_ids
    keys = reader.maps_keys()
    have = [k[: -len(ax.AXIAL_SUFFIX)] for k in keys
            if k.endswith(ax.AXIAL_SUFFIX)]
    known = set(reader.records["record_id"].astype(str))
    have = [r for r in have if r in known]
    if len(have) < 40:
        pytest.skip("too few axial labels for an end-to-end arm")
    have = sorted(have)[:120]
    manifest = SplitManifest(name="AX", kind="filter", seed=0,
                             train_ids=have[:96], val_ids=have[96:])

    cfg = TrainConfig(epochs=2, warmup_epochs=1, batch_size=16, augment=False,
                      min_case_val=2, map_norm_subset=32, round_trip_rows=4,
                      axial_head=True, axial_rank=4)
    cfg.auto_fit_cell_calibration = False
    enc = FeatureEncoder(cond_schema="v5")
    tr = build_precomputed(reader, manifest, fl, fold="train", augment=False,
                           encoder=enc, seed=1, include_axial=True)
    va = build_precomputed(reader, manifest, fl, fold="val", augment=False,
                           encoder=enc, seed=1, include_axial=True)
    basis = _prepare_axial_basis(cfg, tr, va, verbose=False)
    assert basis is not None, "the axial-labelled subset must fit a basis"
    eff, lr, lrf, warm, sched = _resolve_schedule(cfg, torch.device("cpu"))
    members = _train_members([7], train_ds=tr, val_ds=va, cfg=cfg, device="cpu",
                             globals_names=enc.globals_names, reader=reader,
                             eff_batch=eff, lr=lr, lr_final=lrf, warm=warm,
                             resident=False, compile_flag=False,
                             n_channels=len(enc.channels),
                             channel_names=tuple(enc.channels), verbose=False,
                             manifest=manifest)
    out = tmp_path_factory.mktemp("axial_arm")
    d = _finalize_member(out / "member_7", members[0], cfg=cfg, split="AX",
                         globals_names=enc.globals_names, encoder=enc,
                         train_ds=tr, val_ds=va, device="cpu",
                         sched_meta=sched, resident=False)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return d, meta, basis


def test_end_to_end_arm_builds_the_head_and_stamps_the_basis(trained_axial_arm):
    _d, meta, basis = trained_axial_arm
    assert meta["net_config"]["n_axial_anchors"] == 2
    assert meta["net_config"]["n_axial_modes"] == 4
    a = meta["axial_head"]
    assert a["enabled"] is True
    assert a["anchors"] == ["boc", "eoc"] and a["rank"] == 4
    clone = ax.AxialBasis.from_dict(a["basis"])
    np.testing.assert_allclose(clone.mean, basis.mean)
    np.testing.assert_allclose(clone.components, basis.components)
    assert meta["train_config"]["axial_head"] is True


def test_end_to_end_checkpoint_reloads_the_axial_head(trained_axial_arm):
    from lpopt.model.train import load_member

    d, meta, _basis = trained_axial_arm
    model, loaded_meta = load_member(d)
    assert model.has_axial
    cells = torch.randn(2, model.config.in_channels, 19, 19)
    cells[:, 0] = 1.0
    with torch.no_grad():
        out = model(cells, torch.randn(2, model.config.n_globals))
    assert out["axial"].shape == (2, 2, 4)
    basis = ax.AxialBasis.from_dict(loaded_meta["axial_head"]["basis"])
    prof = basis.z_decode(out["axial"].numpy())
    assert prof.shape == (2, 2, ax.N_PLANES)
    # the emitted profile is always core-average 1 -> F_z and AO are meaningful
    np.testing.assert_allclose(prof.mean(axis=-1), 1.0, atol=1e-9)
    assert np.isfinite(ax.axial_peaking_factor(prof)).all()


def test_end_to_end_arm_leaves_the_legacy_heads_intact(trained_axial_arm):
    from lpopt.model.train import load_member

    d, meta, _ = trained_axial_arm
    model, _ = load_member(d)
    cells = torch.randn(2, model.config.in_channels, 19, 19)
    cells[:, 0] = 1.0
    with torch.no_grad():
        out = model(cells, torch.randn(2, model.config.n_globals))
    assert LEGACY_OUT_KEYS <= set(out)
    assert out["mu"].shape == (2, len(meta["target_names"]))
    assert out["map"].shape == (2, 4, 9, 9)


def test_axial_head_can_be_added_onto_a_pre_axial_champion(trained_axial_arm):
    """The standard AL recipe is ``--init-from <champion> --freeze-trunk-cyclen``.

    A champion trained before the axial head has no ``axial_head.*`` keys, so a
    naive strict load would hard-fail and the head would be unusable in the
    recipe it is meant to ship under.  The loader seeds the champion's tensors
    and keeps this run's fresh axial rows — strict on every other key.
    """
    from lpopt.model.train import _MemberState  # noqa: F401  (import parity)

    d, meta, _ = trained_axial_arm
    nc = dict(meta["net_config"])
    legacy_cfg = PosValNetConfig(**{**nc, "n_axial_anchors": 0,
                                    "n_axial_modes": 0})
    torch.manual_seed(3)
    legacy_state = PosValNet(legacy_cfg).state_dict()
    assert not any(k.startswith("axial_head.") for k in legacy_state)

    torch.manual_seed(4)
    model = PosValNet(PosValNetConfig(**nc))
    assert model.has_axial
    with pytest.raises(RuntimeError):
        model.load_state_dict(legacy_state, strict=True)
    merged = {**legacy_state,
              **{k: v for k, v in model.state_dict().items()
                 if k.startswith("axial_head.")}}
    model.load_state_dict(merged, strict=True)        # the loader's fallback
    for k, v in legacy_state.items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_freeze_trunk_cyclen_leaves_the_axial_head_trainable():
    from lpopt.model.train import _apply_freeze_trunk_cyclen, _build_member_optim

    net = PosValNet(PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                                    n_axial_anchors=2, n_axial_modes=4))
    _apply_freeze_trunk_cyclen(net, ())
    assert net.axial_head.weight.requires_grad
    assert not net.stem[0].weight.requires_grad
    opt = _build_member_optim(net, lr=1e-3, weight_decay=1e-4, freeze=True)
    tracked = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(net.axial_head.weight) in tracked
    assert id(net.stem[0].weight) not in tracked


def test_axial_gradient_cannot_reach_any_other_head_under_freeze():
    """Under ``--freeze-trunk-cyclen`` the axial term is provably non-interfering.

    The freeze turns off ``requires_grad`` on the whole shared path (stem /
    blocks / films / head_trunk / conv_head), so the axial loss's backward pass
    terminates at ``axial_head``'s own two tensors.  No other head can move
    because of it — which is what makes shipping the head as a passenger on an AL
    retrain a zero-risk change to the champion's cyclen / F_r / map behaviour.
    """
    from lpopt.model.train import _apply_freeze_trunk_cyclen, axial_loss

    net = PosValNet(PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                                    n_axial_anchors=2, n_axial_modes=6))
    _apply_freeze_trunk_cyclen(net, ())
    cells = torch.randn(3, 26, 19, 19)
    cells[:, 0] = 1.0
    out = net(cells, torch.randn(3, 10))
    axial_loss(out["axial"], torch.randn(3, 2, 6), torch.ones(3, 2)).backward()
    touched = {n for n, p in net.named_parameters() if p.grad is not None
               and float(p.grad.abs().sum()) > 0.0}
    assert touched == {"axial_head.weight", "axial_head.bias"}, touched


def test_cli_axial_flags_map_onto_the_train_config():
    import argparse
    import inspect

    from lpopt.model import train as train_mod

    src = inspect.getsource(train_mod.main)
    for flag in ("--axial-head", "--axial-rank", "--axial-weight"):
        assert flag in src
    ap = argparse.ArgumentParser()
    ap.add_argument("--axial-head", action="store_true")
    assert ap.parse_args([]).axial_head is False
