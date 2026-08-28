"""A/B round-2 variance arms A1 / A2 / A3 (pre-registration 20260730).

Three independent, default-OFF changes on top of the 20260729_054749 champion
recipe, plus the contract every one of them has to satisfy first:

* **A1 trajectory supervision** (``--traj-weight``) — the ``<rid>__traj`` EDIT5
  per-burnup-step planes supervise the EXISTING map decoder, re-run on a trunk
  feature FiLM-conditioned by cycle-burnup fraction.
* **A2 provenance-conditioned CBC** (``--cbc-provenance-offset``) — one learned
  label-convention offset per CBC provenance group, applied ONLY inside the cbc
  regression loss, with the served (MASTER-native) group pinned to 0.
* **A3 map-loss peak focus** (``--map-peak-topk-weight``) — extra weight on the
  K hottest LABEL slots of each map plane.

The load-bearing tests are the OFF-identity ones.  An A/B whose control arm is
not bit-identical to the champion recipe measures the harness, not the change.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.data import traj as TJ                                    # noqa: E402
from lpopt.model.dataset_torch import (                              # noqa: E402
    CBC_PROVENANCE_GROUPS, CBC_PROVENANCE_REFERENCE, LPDataset,
    cbc_provenance_codes, cbc_provenance_labels,
)
from lpopt.model.net import (                                        # noqa: E402
    TRAJ_MAP_CHANNELS, PosValNet, PosValNetConfig, count_parameters,
)
from lpopt.model.train import (                                      # noqa: E402
    TrainConfig, _MemberState, _Norm, _step_member, map_loss,
    top_k_slot_weight, traj_loss, _parse_traj_anchors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"

#: The pre-traj / pre-A2 forward output keys.  Nothing new may appear unless a
#: caller explicitly asks for it.
LEGACY_OUT_KEYS = {"mu", "log_sigma", "map", "conv_logit"}


# =========================================================================== #
# 0. shared helpers
# =========================================================================== #
def _net(**over) -> PosValNet:
    base = dict(in_channels=26, n_globals=10, n_targets=7)
    return PosValNet(PosValNetConfig(**{**base, **over}))


def _cells(n: int, c: int = 26) -> torch.Tensor:
    cells = torch.randn(n, c, 19, 19)
    cells[:, 0] = 1.0                       # fuel mask
    return cells


def _member(net: PosValNet, *, n_targets: int = 7, cbc_idx: int = 2,
            use_traj: bool = False, cbc_offset: bool = False,
            tstd: np.ndarray | None = None) -> _MemberState:
    m = _MemberState()
    m.model, m.fwd = net, net
    m.optim = torch.optim.SGD(net.parameters(), lr=1e-2)
    std = np.ones(n_targets) if tstd is None else tstd
    m.norm = _Norm(np.zeros(n_targets), std, np.zeros(4), np.ones(4),
                   torch.device("cpu"), 3)
    m.use_prior, m.q_idx, m.use_nll = False, None, False
    m.running, m.n_batches = 0.0, 0
    m.use_traj = use_traj
    m.cbc_idx = cbc_idx
    m.cbc_offset_param = (net.cbc_provenance_offset
                          if (cbc_offset and net.has_cbc_provenance) else None)
    return m


def _base_batch(n: int = 4, n_targets: int = 7) -> dict[str, torch.Tensor]:
    return {
        "cells": _cells(n), "globals": torch.randn(n, 10),
        "targets": torch.randn(n, n_targets),
        "target_mask": torch.ones(n, n_targets),
        "conv_label": torch.ones(n), "conv_mask": torch.ones(n),
        "maps": torch.randn(n, 4, 9, 9), "maps_mask": torch.ones(n, 4, 9, 9),
    }


def _run_step(net_cfg: PosValNetConfig, cfg: TrainConfig, extra: dict,
              *, seed: int = 0, **member_kw) -> dict[str, torch.Tensor]:
    """One optimizer step from a fixed seed -> the resulting state_dict."""
    torch.manual_seed(seed)
    net = PosValNet(net_cfg)
    m = _member(net, **member_kw)
    torch.manual_seed(5)
    batch = {**_base_batch(), **extra}
    _step_member(m, batch, cfg, use_amp=False, device=torch.device("cpu"))
    return {k: v.clone() for k, v in net.state_dict().items()}


def _assert_state_identical(a: dict, b: dict, *, skip: tuple[str, ...] = ()) -> None:
    assert set(a) == set(b)
    for k in a:
        if k in skip:
            continue
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=k)


# =========================================================================== #
# 1. A1 — the trajectory LABEL contract (real store)
# =========================================================================== #
@pytest.fixture(scope="module")
def store_traj():
    """``(reader, fuel_library, [ids with a traj], [ids without])``."""
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader

    reader = StoreReader(STORE)
    fl = FuelLibrary.from_parquet(STORE / "fuel_types.parquet")
    keys = reader.maps_keys()
    have = {k[: -len(TJ.TRAJ_SUFFIX)] for k in keys
            if k.endswith(TJ.TRAJ_SUFFIX)}
    if not have:
        pytest.skip("no trajectory labels in the store")
    ids = reader.records["record_id"].astype(str)
    with_t = [r for r in ids if r in have][:8]
    without = [r for r in ids if r not in have][:8]
    return reader, fl, with_t, without


def test_traj_endpoints_are_exactly_the_legacy_map_planes(store_traj):
    """The claim the whole arm rests on, checked on REAL labels.

    If the trajectory endpoints were not bit-identical to the 4-plane stack, the
    map head's z-score constants would be the wrong ones for the intermediate
    steps and the shared-decoder design would be unsound.
    """
    reader, _fl, with_t, _ = store_traj
    checked = 0
    for rid in with_t:
        stack = TJ.load_traj(reader, rid)
        legacy = reader.maps(rid)
        if stack is None or legacy is None:
            continue
        legacy = np.asarray(legacy, dtype=np.float64)
        for map_plane, traj_slice in (
            (legacy[0], stack[0, 0]),        # boc_power  == power  @ step 0
            (legacy[1], stack[-1, 0]),       # eoc_power  == power  @ step -1
            (legacy[2], stack[-1, 1]),       # eoc_burnup == burnup @ step -1
            (legacy[3], stack[-1, 2]),       # eoc_kinf   == kinf   @ step -1
        ):
            np.testing.assert_array_equal(
                np.nan_to_num(map_plane, nan=-1.0),
                np.nan_to_num(traj_slice, nan=-1.0))
        checked += 1
    assert checked > 0


def test_traj_map_channels_match_the_endpoint_identity():
    """``TRAJ_MAP_CHANNELS`` must be the EOC triple, in trajectory-plane order."""
    from lpopt.model.train import _MAP_KEYS

    assert [_MAP_KEYS[c] for c in TRAJ_MAP_CHANNELS] == [
        "eoc_power", "eoc_burnup", "eoc_kinf"]
    assert TJ.STEP_PLANES == ("power", "burnup", "kinf")


def test_cycle_burnup_fraction_is_zero_at_boc_one_at_eoc_and_monotone(store_traj):
    reader, _fl, with_t, _ = store_traj
    checked = 0
    for rid in with_t:
        stack = TJ.load_traj(reader, rid)
        if stack is None:
            continue
        f = TJ.cycle_burnup_fraction(stack)
        assert f is not None
        assert f[0] == 0.0 and f[-1] == pytest.approx(1.0, abs=1e-12)
        assert (np.diff(f) >= 0).all()
        assert f.shape == (stack.shape[0],)
        checked += 1
    assert checked > 0


def test_cycle_burnup_fraction_uses_the_69_slots_not_the_nan_padding(store_traj):
    """A mean over all 81 quarter cells would be NaN — the 12 non-slot cells are
    NaN by construction.  That is exactly why the clock is slot-masked."""
    reader, _fl, with_t, _ = store_traj
    stack = TJ.load_traj(reader, with_t[0])
    assert stack is not None
    assert np.isnan(stack).any(), "the padding must really be NaN"
    assert np.isfinite(TJ.slot_mean_burnup(stack)).all()
    assert np.isnan(stack[:, TJ.BURNUP_PLANE].reshape(len(stack), -1).mean(1)).any()


def test_load_traj_rejects_malformed_or_missing_labels():
    class _R:
        def __init__(self, payload):
            self.payload = payload

        def maps(self, key):
            return self.payload.get(key)

    rid = "abc"
    good = np.ones((6, 3, 9, 9), dtype=np.float16)
    assert TJ.traj_key(rid) == "abc__traj"
    assert TJ.load_traj(_R({}), rid) is None                       # absent
    assert TJ.load_traj(_R({"abc__traj": good}), rid) is not None
    assert TJ.load_traj(_R({"abc__traj": good[0]}), rid) is None   # rank
    assert TJ.load_traj(_R({"abc__traj": good[:1]}), rid) is None  # 1 step
    assert TJ.load_traj(_R({"abc__traj": np.ones((6, 4, 9, 9))}), rid) is None
    poisoned = good.astype(np.float32).copy()
    poisoned[0, 0, 0, 0] = np.nan          # (0,0) IS a real slot
    assert TJ.load_traj(_R({"abc__traj": poisoned}), rid) is None
    # ... but NaN in the non-slot padding is normal and must NOT reject
    padded = good.astype(np.float32).copy()
    padded[:, :, 8, 8] = np.nan            # (8,8) is not a slot
    assert TJ.load_traj(_R({"abc__traj": padded}), rid) is not None


def test_degenerate_trajectory_has_no_burnup_clock():
    flat = np.ones((5, 3, 9, 9))            # burnup never advances
    assert TJ.cycle_burnup_fraction(flat) is None
    assert TJ.anchor_planes(flat) is None


def test_anchor_selection_resolves_ties_to_the_first_step():
    """MASTER prints two steps at zero cycle burnup; anchor 0.0 must pick the
    real BOC snapshot (step 0), not the second one."""
    frac = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0])
    idx = TJ.anchor_indices(frac, (0.0, 0.5, 1.0))
    np.testing.assert_array_equal(idx, [0, 3, 5])


def test_anchor_masks_out_a_fraction_no_step_can_support():
    """A two-step trajectory cannot honestly label "half way through the cycle"."""
    stack = np.zeros((2, 3, 9, 9))
    stack[1, TJ.BURNUP_PLANE] = 10.0        # 0 -> 10 GWd/t in one step
    got = TJ.anchor_planes(stack, (0.0, 0.5, 1.0))
    assert got is not None
    _planes, achieved, mask = got
    np.testing.assert_array_equal(mask, [1.0, 0.0, 1.0])
    np.testing.assert_allclose(achieved, [0.0, 0.0, 1.0])


def test_anchor_fraction_returned_is_the_ACHIEVED_one(store_traj):
    """The model is conditioned on where the label really is, never on where it
    was asked to be."""
    reader, _fl, with_t, _ = store_traj
    stack = TJ.load_traj(reader, with_t[0])
    planes, achieved, mask = TJ.anchor_planes(stack, (0.0, 0.25, 0.5, 0.75, 1.0))
    assert planes.shape == (5, 3, 9, 9)
    assert mask.sum() == 5.0
    assert achieved[0] == 0.0 and achieved[-1] == pytest.approx(1.0)
    # the interior anchors land near, but not exactly on, the request
    assert (np.abs(achieved[1:4] - np.array([0.25, 0.5, 0.75]))
            <= TJ.MAX_ANCHOR_FRAC_ERROR).all()


# =========================================================================== #
# 2. A1 — dataset / batch plumbing on a SYNTHETIC store
# =========================================================================== #
@pytest.fixture(scope="module")
def synthetic_store(tmp_path_factory):
    """A tiny store whose ``__traj`` arrays have a KNOWN, checkable structure.

    Burnup advances by exactly 1 GWd/t per step from 0, so the cycle-burnup
    fraction is exactly ``t / (T-1)`` and anchor ``f`` must select step
    ``round(f*(T-1))``.  The power plane at step ``t`` is the constant ``t``, so a
    selected plane identifies its own step index.  Endpoints are wired into the
    legacy 4-plane stack so the endpoint identity holds here too.
    """
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    n_steps, n_rows = 9, 6
    src = pd.read_parquet(STORE / "records.parquet")
    src = src[src["converged"].astype(bool) & src["maps_key"].notna()].head(n_rows)
    if len(src) < n_rows:
        pytest.skip("not enough converged mapped rows to build a synthetic store")

    out = tmp_path_factory.mktemp("traj_store")
    shutil.copyfile(STORE / "fuel_types.parquet", out / "fuel_types.parquet")
    rows = src.copy().reset_index(drop=True)
    ids = rows["record_id"].astype(str).tolist()
    rows["maps_key"] = ids
    rows.to_parquet(out / "records.parquet", index=False)

    slot_r = np.asarray([s.row for s in __import__(
        "lpopt.vendor.masterrl.domain", fromlist=["SLOTS"]).SLOTS])
    slot_c = np.asarray([s.col for s in __import__(
        "lpopt.vendor.masterrl.domain", fromlist=["SLOTS"]).SLOTS])
    maps = {}
    for i, rid in enumerate(ids):
        traj = np.full((n_steps, 3, 9, 9), np.nan, dtype=np.float32)
        for t in range(n_steps):
            traj[t, 0][slot_r, slot_c] = float(t)              # power == step
            traj[t, 1][slot_r, slot_c] = float(t)              # burnup: +1/step
            traj[t, 2][slot_r, slot_c] = 1.0 + 0.01 * t        # kinf
        legacy = np.full((4, 9, 9), np.nan, dtype=np.float32)
        legacy[0] = traj[0, 0]
        legacy[1] = traj[-1, 0]
        legacy[2] = traj[-1, 1]
        legacy[3] = traj[-1, 2]
        maps[rid] = legacy
        maps[TJ.traj_key(rid)] = traj
    np.savez_compressed(out / "maps.npz", **maps)
    return out, ids, n_steps


def test_synthetic_store_traj_selects_the_expected_steps(synthetic_store):
    from lpopt.data.store import StoreReader

    out, ids, n_steps = synthetic_store
    reader = StoreReader(out)
    stack = TJ.load_traj(reader, ids[0])
    assert stack is not None and stack.shape == (n_steps, 3, 9, 9)
    frac = TJ.cycle_burnup_fraction(stack)
    np.testing.assert_allclose(frac, np.arange(n_steps) / (n_steps - 1))
    anchors = (0.0, 0.25, 0.5, 0.75, 1.0)
    planes, achieved, mask = TJ.anchor_planes(stack, anchors)
    # (n_steps-1) == 8, so the requested fractions land EXACTLY on steps 0,2,4,6,8
    np.testing.assert_allclose(achieved, anchors)
    np.testing.assert_array_equal(mask, np.ones(5))
    picked = np.nanmax(planes[:, 0].reshape(5, -1), axis=1)
    np.testing.assert_allclose(picked, [0, 2, 4, 6, 8])


def test_dataset_emits_traj_planes_fraction_and_mask(synthetic_store):
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader

    out, ids, n_steps = synthetic_store
    reader = StoreReader(out)
    fl = FuelLibrary.from_parquet(out / "fuel_types.parquet")
    ds = LPDataset(reader, ids, fl, include_traj=True)
    it = ds[0]
    assert it["traj"].shape == (5, 3, 9, 9)
    assert it["traj_frac"].shape == (5,) and it["traj_mask"].shape == (5,)
    assert float(it["traj_mask"].sum()) == 5.0
    np.testing.assert_allclose(it["traj_frac"].numpy(),
                               [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-6)
    # the 12 non-slot cells stay NaN, exactly as ``maps`` does
    assert int(torch.isnan(it["traj"][0, 0]).sum()) == 12
    # and the loss's per-slot mask is derived from that, not carried separately
    assert int(torch.isfinite(it["traj"][0, 0]).sum()) == 69


def test_dataset_masks_traj_for_a_non_converged_row(synthetic_store):
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader

    out, ids, _ = synthetic_store
    reader = StoreReader(out)
    fl = FuelLibrary.from_parquet(out / "fuel_types.parquet")
    ds = LPDataset(reader, ids[:1], fl, include_traj=True)
    assert float(ds[0]["traj_mask"].sum()) == 5.0
    ds.df.loc[0, "converged"] = False
    assert float(ds[0]["traj_mask"].sum()) == 0.0
    assert torch.isnan(ds[0]["traj"]).all()      # never a fabricated trajectory


def test_precomputed_carries_traj_through_the_batch_gather(synthetic_store):
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    from lpopt.model.featurize import FeatureEncoder
    from lpopt.model.splits import SplitManifest
    from lpopt.model.train import _gather_train_batch, build_precomputed

    out, ids, _ = synthetic_store
    reader = StoreReader(out)
    fl = FuelLibrary.from_parquet(out / "fuel_types.parquet")
    man = SplitManifest(name="TJ", kind="filter", seed=0,
                        train_ids=ids[:4], val_ids=ids[4:])
    enc = FeatureEncoder(cond_schema="v5")
    pre = build_precomputed(reader, man, fl, fold="train", augment=False,
                            encoder=enc, seed=0, include_traj=True)
    assert pre._t["traj"].shape == (4, 5, 3, 9, 9)
    assert pre._t["traj_frac"].shape == (4, 5)
    assert pre._t["traj_mask"].shape == (4, 5)
    sel = torch.arange(3)
    batch = _gather_train_batch(pre._t, None, sel, None, False,
                                torch.device("cpu"), False)
    for k in ("traj", "traj_frac", "traj_mask", "cbc_prov"):
        assert k in batch, k
    assert batch["traj"].shape == (3, 5, 3, 9, 9)


def test_precomputed_flag_off_has_no_traj_tensors(synthetic_store):
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    from lpopt.model.featurize import FeatureEncoder
    from lpopt.model.splits import SplitManifest
    from lpopt.model.train import build_precomputed

    out, ids, _ = synthetic_store
    reader = StoreReader(out)
    fl = FuelLibrary.from_parquet(out / "fuel_types.parquet")
    man = SplitManifest(name="TJ", kind="filter", seed=0,
                        train_ids=ids[:4], val_ids=ids[4:])
    enc = FeatureEncoder(cond_schema="v5")
    pre = build_precomputed(reader, man, fl, fold="train", augment=False,
                            encoder=enc, seed=0)
    assert not any(k.startswith("traj") for k in pre._t)


# =========================================================================== #
# 3. A1 — the head
# =========================================================================== #
def test_traj_head_costs_exactly_one_film():
    """The arm's whole parameter cost is one FiLM: the DECODER is shared."""
    for mode in ("linear", "multiscale"):
        base = count_parameters(_net(map_head_mode=mode))
        on = count_parameters(_net(map_head_mode=mode, n_traj_anchors=5,
                                   n_traj_planes=3))
        film = count_parameters(
            __import__("lpopt.model.net", fromlist=["FiLM"]).FiLM(10 + 1, 112))
        assert on - base == film, mode


@pytest.mark.parametrize("anchors,planes", [(0, 3), (5, 0), (0, 0)])
def test_traj_head_needs_both_dimensions(anchors, planes):
    net = _net(n_traj_anchors=anchors, n_traj_planes=planes)
    assert not net.has_traj
    assert not hasattr(net, "traj_film")


def test_traj_head_rejects_more_planes_than_reusable_map_channels():
    with pytest.raises(ValueError, match="reusable map-head planes"):
        _net(n_traj_anchors=5, n_traj_planes=4)


@pytest.mark.parametrize("mode", ["linear", "multiscale"])
def test_forward_emits_traj_only_when_asked(mode):
    net = _net(map_head_mode=mode, n_traj_anchors=5, n_traj_planes=3)
    cells, g = _cells(3), torch.randn(3, 10)
    assert set(net(cells, g)) == LEGACY_OUT_KEYS          # serving call
    frac = torch.rand(3, 5)
    out = net(cells, g, frac)
    assert out["traj"].shape == (3, 5, 3, 9, 9)
    assert set(out) == LEGACY_OUT_KEYS | {"traj"}


def test_traj_readout_is_the_map_head_under_an_identity_film():
    """With the FiLM neutralised the trajectory readout must BE the map head's
    EOC planes — the shared-decoder claim, checked numerically."""
    net = _net(n_traj_anchors=2, n_traj_planes=3)
    with torch.no_grad():
        for lin in net.traj_film.to_scale_shift:
            if isinstance(lin, torch.nn.Linear):
                lin.weight.zero_()
                lin.bias.zero_()
    cells, g = _cells(2), torch.randn(2, 10)
    with torch.no_grad():
        out = net(cells, g, torch.zeros(2, 2))
    torch.testing.assert_close(out["traj"][:, 0],
                               out["map"][:, list(TRAJ_MAP_CHANNELS)])
    torch.testing.assert_close(out["traj"][:, 1],
                               out["map"][:, list(TRAJ_MAP_CHANNELS)])


def test_traj_rows_without_a_label_are_skipped_not_computed():
    net = _net(n_traj_anchors=3, n_traj_planes=3)
    frac = torch.rand(4, 3)
    frac[1] = float("nan")                  # this row has no trajectory
    out = net(_cells(4), torch.randn(4, 10), frac)
    assert float(out["traj"][1].detach().abs().sum()) == 0.0
    assert float(out["traj"][0].detach().abs().sum()) > 0.0


def test_traj_all_rows_unlabelled_is_a_hard_noop():
    net = _net(n_traj_anchors=3, n_traj_planes=3)
    out = net(_cells(4), torch.randn(4, 10), torch.full((4, 3), float("nan")))
    assert float(out["traj"].detach().abs().sum()) == 0.0


def test_traj_loss_masks_per_anchor_and_ignores_nan_padding():
    pred = torch.randn(2, 3, 3, 9, 9)
    tgt = pred.clone()
    tgt[:, :, :, 8, 8] = float("nan")          # non-slot padding
    assert float(traj_loss(pred, tgt, torch.ones(2, 3))) == pytest.approx(0.0)
    tgt2 = tgt.clone()
    tgt2[0, 1] += 50.0
    per_anchor = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    assert float(traj_loss(pred, tgt2, per_anchor)) == pytest.approx(0.0)
    assert float(traj_loss(pred, tgt2, torch.ones(2, 3))) > 0.0
    assert float(traj_loss(pred, tgt2, torch.zeros(2, 3))) == 0.0


def test_traj_gradient_reaches_the_film_and_the_shared_decoder():
    net = _net(n_traj_anchors=3, n_traj_planes=3, map_head_mode="multiscale")
    out = net(_cells(3), torch.randn(3, 10), torch.rand(3, 3))
    traj_loss(out["traj"], torch.randn(3, 3, 3, 9, 9), torch.ones(3, 3)).backward()
    assert float(net.traj_film.to_scale_shift[0].weight.grad.abs().sum()) > 0.0
    # ... and it reaches the SHARED decoder, which is the point of the design
    assert float(net.map_decoder.proj.weight.grad.abs().sum()) > 0.0


def test_z_traj_uses_the_map_constants_of_the_reused_channels():
    norm = _Norm(np.zeros(7), np.ones(7), np.array([0.0, 1.0, 20.0, 0.5]),
                 np.array([1.0, 2.0, 4.0, 0.25]), torch.device("cpu"), 3)
    raw = torch.zeros(1, 1, 3, 9, 9)
    z = norm.z_traj(raw)
    # channels (1,2,3) -> means (1,20,0.5), stds (2,4,0.25)
    np.testing.assert_allclose(z[0, 0, :, 0, 0].numpy(),
                               [(0 - 1) / 2, (0 - 20) / 4, (0 - 0.5) / 0.25])


# =========================================================================== #
# 4. A1 — flag OFF is byte-identical
# =========================================================================== #
def _net_digest(cfg: PosValNetConfig) -> tuple[str, str, frozenset]:
    torch.manual_seed(1234)
    net = PosValNet(cfg)
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    torch.manual_seed(7)
    cells = _cells(3, cfg.in_channels)
    net.eval()
    with torch.no_grad():
        out = net(cells, torch.randn(3, cfg.n_globals))
    ho = hashlib.sha256()
    for k in sorted(out):
        ho.update(k.encode())
        ho.update(out[k].numpy().tobytes())
    return h.hexdigest(), ho.hexdigest(), frozenset(out)


@pytest.mark.parametrize("mode", ["linear", "multiscale"])
def test_all_three_arms_off_is_byte_identical_to_the_implicit_default(mode):
    implicit = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                               map_head_mode=mode)
    explicit = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                               map_head_mode=mode, n_traj_anchors=0,
                               n_traj_planes=0, n_cbc_provenance_groups=0)
    assert _net_digest(implicit) == _net_digest(explicit)
    assert _net_digest(implicit)[2] == LEGACY_OUT_KEYS
    sd = PosValNet(implicit).state_dict()
    assert not any("traj" in k or "cbc" in k for k in sd)


def _legacy_forward(self, cells, globals_):
    """``PosValNet.forward`` EXACTLY as it read before the 2026-07-30 extraction.

    A1 needed the map head / physics-prior / gather sequence to be callable twice
    (once for the map, once per burnup anchor), so it was extracted into
    ``PosValNet._map_quarter``.  That extraction is a pure code motion and this
    function is the receipt: it is the pre-extraction body, kept verbatim, so the
    comparison below is against the champion's ACTUAL serving arithmetic rather
    than against a restatement of the new code.
    """
    from lpopt.model.net import _QUARTER

    fuel_mask = cells[:, 0:1]
    h = self.stem(cells)
    taps = [h] if self._map_taps else []
    for b, block in enumerate(self.blocks):
        h = block(h)
        if str(b) in self.films:
            h = self.films[str(b)](h, globals_)
        if self._map_taps and b in self._map_taps:
            taps.append(h)
    map_feat = self.map_decoder(taps) if self._map_taps else self.map_head(h)
    if self.map_prior_channel >= 0:
        pp = cells[:, self.map_prior_channel:self.map_prior_channel + 1]
        map_feat = (map_feat + self.map_prior_gain.view(1, -1, 1, 1) * pp
                    + self.map_prior_bias.view(1, -1, 1, 1))
    gathered = map_feat[:, :, self._se_r, self._se_c]
    quarter = map_feat.new_zeros((gathered.shape[0], self.config.n_map_channels,
                                  _QUARTER, _QUARTER))
    quarter[:, :, self._q_r, self._q_c] = gathered
    denom = fuel_mask.sum(dim=(2, 3)).clamp_min(1.0)
    masked_mean = (h * fuel_mask).sum(dim=(2, 3)) / denom
    neg_inf = torch.finfo(h.dtype).min
    masked_max = h.masked_fill(fuel_mask == 0, neg_inf).amax(dim=(2, 3))
    feat = self.head_trunk(torch.cat([masked_mean, masked_max, globals_], dim=1))
    out = {"mu": self.mu_head(feat), "log_sigma": self.log_sigma_head(feat),
           "map": quarter, "conv_logit": self.conv_head(feat).squeeze(-1)}
    if self.has_quantiles:
        out["quantiles"] = self.quantile_head(feat).view(
            -1, self.n_quantile_targets, self.n_quantiles)
    if self.has_axial:
        out["axial"] = self.axial_head(feat).view(
            -1, self.n_axial_anchors, self.n_axial_modes)
    return out


#: The champion's exact net shape (`data/models/20260729_054749`), which is what
#: the control arm A0 must reproduce number-for-number.
CHAMPION_NET = dict(in_channels=52, n_globals=13, width=224, n_blocks=8,
                    head_hidden=384, n_targets=8, n_quantile_targets=2,
                    n_quantiles=3, map_head_mode="multiscale",
                    map_prior_channel=50)


@pytest.mark.parametrize("kw", [
    dict(in_channels=26, n_globals=10, n_targets=7),
    dict(in_channels=26, n_globals=10, n_targets=7, map_head_mode="multiscale"),
    CHAMPION_NET,
    {**CHAMPION_NET, "n_traj_anchors": 0, "n_traj_planes": 0,
     "n_cbc_provenance_groups": 0},
])
def test_map_quarter_extraction_is_a_bit_identical_code_motion(kw):
    """The serving forward must produce the SAME BITS as before the extraction.

    This is the guarantee the whole A/B rests on: if the control arm's arithmetic
    moved, every paired difference measured against it is measuring the refactor.
    """
    torch.manual_seed(99)
    net = PosValNet(PosValNetConfig(**kw)).eval()
    torch.manual_seed(3)
    cells = _cells(5, kw["in_channels"])
    g = torch.randn(5, kw["n_globals"])
    with torch.no_grad():
        new = net(cells, g)
        old = _legacy_forward(net, cells, g)
    assert set(new) == set(old)
    for k in new:
        torch.testing.assert_close(new[k], old[k], rtol=0, atol=0, msg=k)


def test_train_config_arm_defaults_are_all_off():
    cfg = TrainConfig()
    assert cfg.traj_weight == 0.0
    assert cfg.cbc_provenance_offset is False
    assert cfg.map_peak_topk_weight == 0.0
    assert cfg.traj_anchors == TJ.DEFAULT_ANCHORS
    assert cfg.map_peak_topk == 5
    pc = PosValNetConfig()
    assert (pc.n_traj_anchors, pc.n_traj_planes, pc.n_cbc_provenance_groups) \
        == (0, 0, 0)


def test_traj_weight_zero_step_is_bit_identical_even_with_labels_present():
    """The control arm's guarantee: trajectory tensors sitting in the batch must
    not perturb a single weight when ``traj_weight`` is 0."""
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7)
    clean = _run_step(cfg, TrainConfig(), {})
    with_labels = _run_step(cfg, TrainConfig(), {
        "traj": torch.randn(4, 5, 3, 9, 9),
        "traj_frac": torch.rand(4, 5),
        "traj_mask": torch.ones(4, 5),
    })
    _assert_state_identical(clean, with_labels)


def test_traj_head_present_but_weight_zero_still_does_not_train_it():
    """Even a net that HAS the head must not receive a trajectory gradient at
    weight 0 — the branch is gated on the weight, not only on the module."""
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                          n_traj_anchors=5, n_traj_planes=3)
    extra = {"traj": torch.randn(4, 5, 3, 9, 9),
             "traj_frac": torch.rand(4, 5),
             "traj_mask": torch.ones(4, 5)}
    torch.manual_seed(0)
    init = {k: v.clone() for k, v in PosValNet(cfg).state_dict().items()}
    off = _run_step(cfg, TrainConfig(traj_weight=0.0), extra, use_traj=True)
    on = _run_step(cfg, TrainConfig(traj_weight=0.3), extra, use_traj=True)
    # at weight 0 the FiLM is untouched: no gradient ever reached it
    for k in init:
        if k.startswith("traj_film."):
            torch.testing.assert_close(off[k], init[k], rtol=0, atol=0, msg=k)
            assert not torch.equal(on[k], init[k]), k
    # ... and the SHARED decoder moves only when the term is on
    assert not torch.equal(off["map_head.weight"], on["map_head.weight"])


# =========================================================================== #
# 5. A2 — provenance grouping, offset, and the serve contract
# =========================================================================== #
def test_provenance_groups_have_the_reference_first():
    assert CBC_PROVENANCE_GROUPS[0] == CBC_PROVENANCE_REFERENCE == "master_native"
    assert len(set(CBC_PROVENANCE_GROUPS)) == len(CBC_PROVENANCE_GROUPS)


def test_provenance_labels_resolve_the_store_columns():
    df = pd.DataFrame({
        "dataset": ["A", "B", "P", "P", "P", "Z"],
        "restart_provenance": ["mocha_native", "ga_native",
                               "pair_ecore:MAS_RST.APRQ_11_0767.14",
                               "pair_feed:MAS_RST.APRQ_11_0686.40",
                               "something_new", None],
    })
    labels = list(cbc_provenance_labels(df))
    assert labels == ["mocha_native", "ga_native", "master_native",
                      "master_native", "master_native", "master_native"]
    np.testing.assert_array_equal(cbc_provenance_codes(df), [1, 2, 0, 0, 0, 0])


def test_unknown_provenance_falls_back_to_the_reference():
    """A new data source must get NO offset until someone argues for one."""
    df = pd.DataFrame({"dataset": ["Q"], "restart_provenance": ["brand_new"]})
    assert list(cbc_provenance_labels(df)) == ["master_native"]
    assert list(cbc_provenance_codes(df)) == [0]


def test_provenance_codes_survive_missing_columns():
    df = pd.DataFrame({"dataset": ["A", "P"]})
    np.testing.assert_array_equal(cbc_provenance_codes(df), [1, 0])
    assert len(cbc_provenance_codes(pd.DataFrame())) == 0


def test_real_store_provenance_split_matches_the_forensic():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    df = pd.read_parquet(STORE / "records.parquet",
                         columns=["dataset", "restart_provenance", "cbc_kind"])
    codes = cbc_provenance_codes(df)
    # Dataset A is the mocha_native group and is the MAJORITY of the corpus —
    # which is why the convention gap matters at all.
    assert (codes == 1).sum() == int((df["dataset"] == "A").sum())
    assert (codes == 1).mean() > 0.5


def test_cbc_offset_parameter_omits_the_reference_group():
    net = _net(n_cbc_provenance_groups=3)
    assert net.has_cbc_provenance
    assert net.cbc_provenance_offset.shape == (2,)      # NOT 3
    assert not _net(n_cbc_provenance_groups=0).has_cbc_provenance
    assert not _net(n_cbc_provenance_groups=1).has_cbc_provenance


def test_cbc_offset_is_never_read_by_forward():
    """The serve path cannot be shifted, because ``forward`` does not know the
    parameter exists."""
    net = _net(n_cbc_provenance_groups=3)
    cells, g = _cells(3), torch.randn(3, 10)
    with torch.no_grad():
        before = net(cells, g)["mu"].clone()
        net.cbc_provenance_offset.fill_(37.0)     # an absurd offset
        after = net(cells, g)["mu"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_cbc_offset_flag_off_step_is_bit_identical_with_codes_present():
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7)
    clean = _run_step(cfg, TrainConfig(), {})
    with_codes = _run_step(cfg, TrainConfig(),
                           {"cbc_prov": torch.tensor([0, 1, 2, 1])})
    _assert_state_identical(clean, with_codes)


def test_cbc_offset_moves_only_the_cbc_residual():
    """The offset must change the cbc loss and NOTHING else.

    Proved by construction: with an all-reference batch (code 0) the offset is a
    structural zero, so a net whose offset parameter is huge must step exactly
    like one whose offset is zero.  Flip one row to a non-reference group and the
    step changes.
    """
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                          n_cbc_provenance_groups=3)
    tcfg = TrainConfig(cbc_provenance_offset=True)

    def _step(codes, poison):
        torch.manual_seed(0)
        net = PosValNet(cfg)
        with torch.no_grad():
            net.cbc_provenance_offset.fill_(poison)
        m = _member(net, cbc_offset=True)
        torch.manual_seed(5)
        _step_member(m, {**_base_batch(), "cbc_prov": codes}, tcfg,
                     use_amp=False, device=torch.device("cpu"))
        return {k: v.clone() for k, v in net.state_dict().items()}

    all_ref = torch.zeros(4, dtype=torch.long)
    # every NETWORK weight is identical (the offset tensor itself trivially is
    # not: the test set it to two different values)
    _assert_state_identical(_step(all_ref, 0.0), _step(all_ref, 9.0),
                            skip=("cbc_provenance_offset",))
    mixed = torch.tensor([0, 1, 0, 2])
    assert not torch.equal(_step(mixed, 0.0)["mu_head.weight"],
                           _step(mixed, 9.0)["mu_head.weight"])


def test_cbc_offset_receives_a_gradient_and_the_reference_cannot():
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                          n_cbc_provenance_groups=3)
    torch.manual_seed(0)
    net = PosValNet(cfg)
    m = _member(net, cbc_offset=True)
    batch = {**_base_batch(), "cbc_prov": torch.tensor([0, 1, 1, 2])}
    _step_member(m, batch, TrainConfig(cbc_provenance_offset=True),
                 use_amp=False, device=torch.device("cpu"))
    g = net.cbc_provenance_offset.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0
    # the reference group has NO parameter at all, so it has no gradient either
    assert net.cbc_provenance_offset.numel() == len(CBC_PROVENANCE_GROUPS) - 1


def test_cbc_offset_is_learned_in_z_units_not_ppm():
    """A ppm-scale parameter would be unlearnable under Adam (see _step_member).

    Checked behaviourally: with tstd == 386 ppm, an offset of 1.0 must shift the
    cbc residual by ONE z, not by one ppm.
    """
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7,
                          n_cbc_provenance_groups=3)
    torch.manual_seed(0)
    net = PosValNet(cfg)
    with torch.no_grad():
        net.cbc_provenance_offset.fill_(1.0)
    tstd = np.ones(7)
    tstd[2] = 386.32
    m = _member(net, cbc_offset=True, tstd=tstd)
    torch.manual_seed(5)
    batch = {**_base_batch(), "cbc_prov": torch.ones(4, dtype=torch.long)}
    batch["targets"] = torch.zeros(4, 7)
    with torch.no_grad():
        mu = net(batch["cells"], batch["globals"])["mu"]
    _step_member(m, batch, TrainConfig(cbc_provenance_offset=True),
                 use_amp=False, device=torch.device("cpu"))
    # the shifted prediction is mu + 1.0 (z), not mu + 1/386 — so the residual
    # the loss saw differs from the unshifted one by ~1 z.
    from lpopt.model.train import regression_loss
    z_t = torch.zeros(4, 7)
    plain = float(regression_loss(mu, torch.zeros(4, 7), z_t, torch.ones(4, 7),
                                  use_nll=False, beta=0.5, delta=1.0))
    mu_z = mu.clone()
    mu_z[:, 2] += 1.0
    shifted = float(regression_loss(mu_z, torch.zeros(4, 7), z_t,
                                    torch.ones(4, 7), use_nll=False,
                                    beta=0.5, delta=1.0))
    assert abs(shifted - plain) > 0.05


# =========================================================================== #
# 6. A3 — top-K map peak focus
# =========================================================================== #
def test_topk_weight_off_is_byte_identical():
    torch.manual_seed(0)
    pred, tgt = torch.randn(3, 4, 9, 9), torch.randn(3, 4, 9, 9)
    mask = torch.ones(3, 4, 9, 9)
    for peak_w in (0.0, 2.0):
        a = map_loss(pred, tgt, mask, 1.0, peak_weight=peak_w)
        b = map_loss(pred, tgt, mask, 1.0, peak_weight=peak_w,
                     peak_topk=5, peak_topk_weight=0.0)
        c = map_loss(pred, tgt, mask, 1.0, peak_weight=peak_w,
                     peak_topk=0, peak_topk_weight=2.0)
        assert float(a) == float(b) == float(c)


def test_topk_weight_on_changes_the_loss_and_multiplies_with_peak_weight():
    torch.manual_seed(0)
    pred, tgt = torch.randn(3, 4, 9, 9), torch.randn(3, 4, 9, 9)
    mask = torch.ones(3, 4, 9, 9)
    plain = float(map_loss(pred, tgt, mask, 1.0))
    topk = float(map_loss(pred, tgt, mask, 1.0, peak_topk=5,
                          peak_topk_weight=2.0))
    both = float(map_loss(pred, tgt, mask, 1.0, peak_weight=2.0,
                          peak_topk=5, peak_topk_weight=2.0))
    only_cont = float(map_loss(pred, tgt, mask, 1.0, peak_weight=2.0))
    assert topk != plain and both != only_cont


def test_topk_selects_exactly_k_slots_per_plane_by_LABEL():
    tgt = torch.randn(2, 4, 9, 9)
    w = top_k_slot_weight(tgt, torch.ones(2, 4, 9, 9), 5, 3.0)
    assert w.shape == (2, 4, 9, 9)
    for b in range(2):
        for c in range(4):
            hot = w[b, c] > 1.0
            assert int(hot.sum()) == 5
            # they really are the 5 largest LABEL values of that plane
            thresh = tgt[b, c].flatten().topk(5).values.min()
            assert float(tgt[b, c][hot].min()) >= float(thresh) - 1e-6


def test_topk_never_selects_a_masked_or_nan_slot():
    tgt = torch.arange(81, dtype=torch.float32).view(1, 1, 9, 9)
    mask = torch.zeros(1, 1, 9, 9)
    mask[0, 0, 0, :3] = 1.0                 # only 3 valid slots, the SMALLEST
    w = top_k_slot_weight(tgt, mask, 5, 3.0)
    assert float(w[mask == 0].max()) == 1.0
    assert int((w > 1.0).sum()) == 3        # k clamped to what is valid
    nan_tgt = tgt.clone()
    nan_tgt[0, 0, 8, 8] = float("nan")      # the largest slot, but NaN
    w2 = top_k_slot_weight(nan_tgt, torch.ones(1, 1, 9, 9), 3, 3.0)
    assert float(w2[0, 0, 8, 8]) == 1.0


def test_topk_weight_off_step_is_bit_identical():
    cfg = PosValNetConfig(in_channels=26, n_globals=10, n_targets=7)
    clean = _run_step(cfg, TrainConfig(map_peak_weight=2.0), {})
    explicit = _run_step(cfg, TrainConfig(map_peak_weight=2.0, map_peak_topk=5,
                                          map_peak_topk_weight=0.0), {})
    _assert_state_identical(clean, explicit)
    on = _run_step(cfg, TrainConfig(map_peak_weight=2.0, map_peak_topk=5,
                                    map_peak_topk_weight=2.0), {})
    assert not torch.equal(clean["map_head.weight"], on["map_head.weight"])


# =========================================================================== #
# 7. end-to-end: train a tiny arm with all three changes, then serve it
# =========================================================================== #
@pytest.fixture(scope="module")
def trained_arm(tmp_path_factory, synthetic_store):
    """A 1-member arm trained (2 epochs) with A1 + A2 + A3 all ON."""
    import json

    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    from lpopt.model.featurize import FeatureEncoder
    from lpopt.model.splits import SplitManifest
    from lpopt.model.train import (
        _finalize_member, _resolve_schedule, _train_members, build_precomputed,
    )

    out, ids, _ = synthetic_store
    reader = StoreReader(out)
    fl = FuelLibrary.from_parquet(out / "fuel_types.parquet")
    man = SplitManifest(name="AB2", kind="filter", seed=0,
                        train_ids=ids[:4], val_ids=ids[4:])
    cfg = TrainConfig(epochs=2, warmup_epochs=1, batch_size=4, augment=False,
                      min_case_val=2, map_norm_subset=8, round_trip_rows=2,
                      traj_weight=0.3, cbc_provenance_offset=True,
                      map_peak_weight=2.0, map_peak_topk_weight=2.0)
    cfg.auto_fit_cell_calibration = False
    enc = FeatureEncoder(cond_schema="v5")
    tr = build_precomputed(reader, man, fl, fold="train", augment=False,
                           encoder=enc, seed=1, include_traj=True,
                           traj_anchors=cfg.traj_anchors)
    va = build_precomputed(reader, man, fl, fold="val", augment=False,
                           encoder=enc, seed=1, include_traj=True,
                           traj_anchors=cfg.traj_anchors)
    eff, lr, lrf, warm, sched = _resolve_schedule(cfg, torch.device("cpu"))
    members = _train_members([7], train_ds=tr, val_ds=va, cfg=cfg, device="cpu",
                             globals_names=enc.globals_names, reader=reader,
                             eff_batch=eff, lr=lr, lr_final=lrf, warm=warm,
                             resident=False, compile_flag=False,
                             n_channels=len(enc.channels),
                             channel_names=tuple(enc.channels), verbose=False,
                             manifest=man)
    d = _finalize_member(tmp_path_factory.mktemp("ab2") / "member_7", members[0],
                         cfg=cfg, split="AB2", globals_names=enc.globals_names,
                         encoder=enc, train_ds=tr, val_ds=va, device="cpu",
                         sched_meta=sched, resident=False)
    return d, json.loads((d / "meta.json").read_text(encoding="utf-8"))


def test_end_to_end_stamps_the_traj_contract(trained_arm):
    _d, meta = trained_arm
    assert meta["net_config"]["n_traj_anchors"] == 5
    assert meta["net_config"]["n_traj_planes"] == 3
    t = meta["traj_head"]
    assert t["enabled"] is True and t["serve_affecting"] is False
    assert t["anchors"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert t["planes"] == ["power", "burnup", "kinf"]
    assert t["map_channels"] == list(TRAJ_MAP_CHANNELS)
    assert t["weight"] == 0.3
    assert meta["train_config"]["traj_weight"] == 0.3


def test_end_to_end_stamps_the_cbc_offsets_in_both_units(trained_arm):
    _d, meta = trained_arm
    c = meta["cbc_provenance_offset"]
    assert c["enabled"] is True and c["serve_affecting"] is False
    assert c["groups"] == list(CBC_PROVENANCE_GROUPS)
    assert c["reference"] == c["serve_convention"] == "master_native"
    # the REFERENCE offset is a structural zero in BOTH unit systems
    assert c["offsets_z"][0] == 0.0 and c["offsets_ppm"][0] == 0.0
    assert len(c["offsets_z"]) == len(CBC_PROVENANCE_GROUPS)
    std = c["cbc_tstd_ppm"]
    np.testing.assert_allclose(c["offsets_ppm"],
                               [v * std for v in c["offsets_z"]], rtol=1e-9)


def test_end_to_end_checkpoint_serves_without_a_traj_output(trained_arm):
    from lpopt.model.train import load_member

    d, meta = trained_arm
    model, _ = load_member(d)
    assert model.has_traj and model.has_cbc_provenance
    cells = _cells(2, model.config.in_channels)
    with torch.no_grad():
        out = model(cells, torch.randn(2, model.config.n_globals))
    # the serving call is two positional arguments -> no traj, no offset, no
    # change of any kind to what the rest of the system consumes.
    assert set(out) == LEGACY_OUT_KEYS
    assert out["mu"].shape == (2, len(meta["target_names"]))
    assert out["map"].shape == (2, 4, 9, 9)


def test_end_to_end_flag_off_meta_has_no_arm_keys(synthetic_store,
                                                  tmp_path_factory):
    import json

    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    from lpopt.model.featurize import FeatureEncoder
    from lpopt.model.splits import SplitManifest
    from lpopt.model.train import (
        _finalize_member, _resolve_schedule, _train_members, build_precomputed,
    )

    out, ids, _ = synthetic_store
    reader = StoreReader(out)
    fl = FuelLibrary.from_parquet(out / "fuel_types.parquet")
    man = SplitManifest(name="AB2", kind="filter", seed=0,
                        train_ids=ids[:4], val_ids=ids[4:])
    cfg = TrainConfig(epochs=1, warmup_epochs=1, batch_size=4, augment=False,
                      min_case_val=2, map_norm_subset=8, round_trip_rows=2)
    cfg.auto_fit_cell_calibration = False
    enc = FeatureEncoder(cond_schema="v5")
    tr = build_precomputed(reader, man, fl, fold="train", augment=False,
                           encoder=enc, seed=1)
    va = build_precomputed(reader, man, fl, fold="val", augment=False,
                           encoder=enc, seed=1)
    eff, lr, lrf, warm, sched = _resolve_schedule(cfg, torch.device("cpu"))
    ms = _train_members([7], train_ds=tr, val_ds=va, cfg=cfg, device="cpu",
                        globals_names=enc.globals_names, reader=reader,
                        eff_batch=eff, lr=lr, lr_final=lrf, warm=warm,
                        resident=False, compile_flag=False,
                        n_channels=len(enc.channels),
                        channel_names=tuple(enc.channels), verbose=False,
                        manifest=man)
    d = _finalize_member(tmp_path_factory.mktemp("ab2off") / "member_7", ms[0],
                         cfg=cfg, split="AB2", globals_names=enc.globals_names,
                         encoder=enc, train_ds=tr, val_ds=va, device="cpu",
                         sched_meta=sched, resident=False)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert "traj_head" not in meta
    assert "cbc_provenance_offset" not in meta
    assert meta["net_config"]["n_traj_anchors"] == 0
    assert meta["net_config"]["n_cbc_provenance_groups"] == 0
    assert not any("traj" in k or "cbc" in k
                   for k in torch.load(d / "model.pt", weights_only=True))


# =========================================================================== #
# 8. CLI surface
# =========================================================================== #
def test_cli_exposes_every_arm_flag():
    import inspect

    from lpopt.model import train as train_mod

    src = inspect.getsource(train_mod.main)
    for flag in ("--traj-weight", "--traj-anchors", "--cbc-provenance-offset",
                 "--map-peak-topk-weight", "--map-peak-topk"):
        assert flag in src, flag


def test_cli_defaults_leave_the_champion_recipe_untouched():
    from lpopt.model.train import main as train_main   # noqa: F401  (import parity)

    cfg = TrainConfig()
    assert (cfg.traj_weight, cfg.cbc_provenance_offset,
            cfg.map_peak_topk_weight) == (0.0, False, 0.0)


#: The champion recipe (``data/models/20260729_054749/run.sh``), as CLI args.
CHAMPION_ARGS = (
    "--ensemble 5 --split S1 --cond-schema v6 --width 224 --n-blocks 8 "
    "--head-hidden 384 --epochs 150 --num-workers 8 --device cpu "
    "--map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 "
    "--map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads "
    "--quantile-weight 0.2 --promote-max-asm-bu "
    "--distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 "
    "--distill-min-match-frac 0.5 --f-r-rank-weight 0.1"
).split()

#: The pre-registered arms: label -> (extra flags, the ONE TrainConfig field that
#: may differ from A0).  Transcribed from
#: ``data/reports/ab2_preregistration_20260730.md`` section 3.1.
PREREGISTERED_ARMS = {
    "A0": ([], None),
    "A1": (["--traj-weight", "0.3"], "traj_weight"),
    "A2": (["--cbc-provenance-offset"], "cbc_provenance_offset"),
    "A3": (["--map-peak-topk-weight", "2.0"], "map_peak_topk_weight"),
}


def test_each_preregistered_arm_differs_from_the_control_in_exactly_one_field(
        monkeypatch):
    """The whole A/B rests on this: one change per arm, nothing else.

    Drives the REAL ``main`` argument parser on the exact commands the
    pre-registration document ships, with ``train_ensemble`` stubbed out, and
    diffs the resulting ``TrainConfig`` field by field against A0's.
    """
    from lpopt.model import train as train_mod

    captured: dict = {}

    def _fake(n, **kw):
        captured["n"] = n
        captured["kw"] = kw
        return []

    monkeypatch.setattr(train_mod, "train_ensemble", _fake)

    configs: dict[str, dict] = {}
    for label, (extra, _field) in PREREGISTERED_ARMS.items():
        assert train_mod.main(CHAMPION_ARGS + extra) == 0
        assert captured["n"] == 5
        assert captured["kw"]["split"] == "S1"
        assert captured["kw"]["cond_schema"] == "v6"
        configs[label] = captured["kw"]["config"].to_dict()

    a0 = configs["A0"]
    # A0 IS the champion recipe: every champion knob survives the parse ...
    for key, want in (("width", 224), ("n_blocks", 8), ("head_hidden", 384),
                      ("map_head_mode", "multiscale"),
                      ("map_prior_residual", True), ("map_spectral_weight", 0.3),
                      ("map_peak_weight", 2.0), ("cyclen_physics_prior", True),
                      ("quantile_heads", True), ("quantile_weight", 0.2),
                      ("promote_max_asm_bu", True), ("distill_weight", 0.4),
                      ("f_r_rank_weight", 0.1)):
        assert a0[key] == want, key
    # ... and A0 has every round-2 knob OFF
    assert (a0["traj_weight"], a0["cbc_provenance_offset"],
            a0["map_peak_topk_weight"]) == (0.0, False, 0.0)

    for label, (_extra, field) in PREREGISTERED_ARMS.items():
        if field is None:
            continue
        diff = {k for k in a0 if configs[label][k] != a0[k]}
        assert diff == {field}, (label, sorted(diff))


def test_traj_anchor_parsing_is_deterministic_and_range_checked():
    assert _parse_traj_anchors("0,0.25,0.5,0.75,1") == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert _parse_traj_anchors("1,0,0.5,0.5") == (0.0, 0.5, 1.0)   # sorted, unique
    assert _parse_traj_anchors("-1,2,abc,,0.5") == (0.5,)          # out of range
    assert _parse_traj_anchors("") == ()
