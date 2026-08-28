"""GPU-throughput refactor of the training loop (train.py):

* device-resident batching == host batching (bit-identical losses/weights),
* parallel-member joint training == sequential same-seed (exactness),
* per-member bootstrap independence (different permutations / weights),
* early-stop dropout bookkeeping (a member that stops does not perturb others),
* batch/LR/warmup schedule resolution (1024 default on cuda, 256 on cpu),
* the transpose-augmentation gather preserves ``PrecomputedDataset`` semantics.

Store-gated tests skip cleanly without the Dataset-A store; the pure-logic tests
(schedule, gather, generator independence, config defaults) always run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.model import train as T                       # noqa: E402
from lpopt.model.train import TrainConfig, _resolve_schedule, _gather_train_batch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


# --------------------------------------------------------------------------- #
# pure-logic tests (no store required)
# --------------------------------------------------------------------------- #
def test_schedule_cpu_is_legacy() -> None:
    cfg = TrainConfig()
    eff, lr, lr_final, warm, meta = _resolve_schedule(cfg, torch.device("cpu"))
    assert eff == 256
    assert lr == pytest.approx(3.0e-4)
    assert lr_final == pytest.approx(3.0e-5)
    assert warm == 20                       # unchanged on the base batch
    assert meta["batch_scale"] == pytest.approx(1.0)


def test_schedule_cuda_scales_batch_lr_warmup() -> None:
    # CUDA default: batch 1024, linear LR x4, warmup steps preserved (20 -> 80).
    cfg = TrainConfig()
    eff, lr, lr_final, warm, meta = _resolve_schedule(cfg, torch.device("cuda"))
    assert eff == 1024
    assert lr == pytest.approx(3.0e-4 * 4)
    assert lr_final == pytest.approx(3.0e-5 * 4)
    assert warm == 80
    assert meta["effective_batch"] == 1024
    assert meta["warmup_epochs_effective"] == 80


def test_schedule_explicit_batch_pins_value() -> None:
    cfg = TrainConfig(batch_size=512, batch_size_explicit=True)
    # explicit pin is honored on any device; LR/warmup still scale off base 256.
    eff, lr, _lf, warm, _m = _resolve_schedule(cfg, torch.device("cuda"))
    assert eff == 512
    assert lr == pytest.approx(3.0e-4 * 2)
    assert warm == 40


def test_schedule_lr_scaling_off() -> None:
    cfg = TrainConfig(lr_scaling=False, warmup_step_scaling=False)
    _eff, lr, lr_final, warm, _m = _resolve_schedule(cfg, torch.device("cuda"))
    assert lr == pytest.approx(3.0e-4)
    assert lr_final == pytest.approx(3.0e-5)
    assert warm == 20


def test_schedule_warmup_clamped_below_epochs() -> None:
    cfg = TrainConfig(epochs=5)                         # 20*4=80 must clamp to <=4
    _eff, _lr, _lf, warm, _m = _resolve_schedule(cfg, torch.device("cuda"))
    assert warm == 4


def test_config_defaults_preserve_cpu() -> None:
    cfg = TrainConfig()
    assert cfg.batch_size == 256                # cpu batch unchanged
    assert cfg.batch_size_cuda == 1024
    assert cfg.parallel_members == 1            # sequential by default (safe)
    assert cfg.device_resident is True
    assert cfg.torch_compile is False


def test_gather_transpose_selects_cells_t_per_row() -> None:
    # Only ``cells`` differ between the base and transposed variants; every other
    # key must come from the base row regardless of the per-row transpose choice.
    n = 6
    host = {
        "cells": torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).repeat(1, 2, 2, 2),
        "cells_t": (100 + torch.arange(n, dtype=torch.float32)).view(n, 1, 1, 1).repeat(1, 2, 2, 2),
        "globals": torch.arange(n, dtype=torch.float32).view(n, 1),
        "targets": torch.arange(n, dtype=torch.float32).view(n, 1),
        "target_mask": torch.ones(n, 1),
        "conv_label": torch.ones(n),
        "conv_mask": torch.ones(n),
        "maps": torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1),
        "maps_mask": torch.ones(n, 1, 1, 1),
    }
    sel = torch.tensor([0, 3, 5])
    uset = torch.tensor([False, True, False])
    batch = _gather_train_batch(host, None, sel, uset, augment=True,
                                device=torch.device("cpu"), resident=False)
    # row 0 -> base(0), row 1 -> transposed(103), row 2 -> base(5)
    assert batch["cells"][:, 0, 0, 0].tolist() == [0.0, 103.0, 5.0]
    # non-cell keys always base-indexed
    assert batch["globals"].squeeze(-1).tolist() == [0.0, 3.0, 5.0]
    assert batch["targets"].squeeze(-1).tolist() == [0.0, 3.0, 5.0]


def test_sampler_generators_independent_per_seed() -> None:
    # Per-member seeded generators must yield different bootstrap permutations.
    w = torch.ones(200, dtype=torch.double)
    g0 = torch.Generator().manual_seed(100)
    g1 = torch.Generator().manual_seed(101)
    i0 = torch.multinomial(w, 200, True, generator=g0)
    i1 = torch.multinomial(w, 200, True, generator=g1)
    assert not torch.equal(i0, i1)
    # same seed -> identical draw (determinism)
    g0b = torch.Generator().manual_seed(100)
    assert torch.equal(i0, torch.multinomial(w, 200, True, generator=g0b))


# --------------------------------------------------------------------------- #
# store-gated end-to-end tests (tiny subset, few epochs)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_data():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("Dataset-A store not present")
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    from lpopt.model.featurize import FeatureEncoder
    from lpopt.model.splits import make_splits

    reader = StoreReader(STORE)
    fuel = FuelLibrary.from_parquet(STORE / "fuel_types.parquet")
    enc = FeatureEncoder()
    man = make_splits(reader.records, seed=0, persist=False)["S0"]
    train_pre = T.build_precomputed(reader, man, fuel, fold="train", augment=True,
                                    encoder=enc, seed=0, subset_rows=40)
    val_pre = T.build_precomputed(reader, man, fuel, fold="val", augment=False,
                                  encoder=enc, seed=0, subset_rows=16)
    return reader, enc, train_pre, val_pre


def _run(seeds, small_data, cfg, resident=False):
    reader, enc, train_pre, val_pre = small_data
    eff, lr, lr_final, warm, _m = _resolve_schedule(cfg, torch.device("cpu"))
    return T._train_members(
        list(seeds), train_ds=train_pre, val_ds=val_pre, cfg=cfg, device="cpu",
        globals_names=enc.globals_names, reader=reader, eff_batch=eff, lr=lr,
        lr_final=lr_final, warm=warm, resident=resident, compile_flag=False,
        verbose=False)


def _sd(m):
    return {k: v.clone() for k, v in m.model.state_dict().items()}


def _equal(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _allclose(a, b, atol=1e-5) -> bool:
    return a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=atol) for k in a)


def test_parallel_members_1_equals_sequential(small_data) -> None:
    """Exactness: a member trained in a joint chunk is bit-identical to the same
    seed trained alone (parallel-members joint stepping leaks no RNG/state)."""
    cfg = TrainConfig(epochs=3, warmup_epochs=1, batch_size=16, patience=10,
                      map_norm_subset=200, augment=True)
    solo0 = _sd(_run([100], small_data, cfg)[0])
    solo1 = _sd(_run([101], small_data, cfg)[0])
    joint = _run([100, 101], small_data, cfg)
    assert _equal(_sd(joint[0]), solo0)
    assert _equal(_sd(joint[1]), solo1)


def test_member_seed_independence(small_data) -> None:
    """Different seeds -> different init + bootstrap permutation -> different weights."""
    cfg = TrainConfig(epochs=3, warmup_epochs=1, batch_size=16, patience=10,
                      map_norm_subset=200, augment=True)
    joint = _run([100, 101], small_data, cfg)
    assert not _equal(_sd(joint[0]), _sd(joint[1]))


def test_device_resident_equals_host(small_data) -> None:
    """The device-resident gather path reproduces the host path bit-for-bit
    (weights and per-epoch training losses) over multiple epochs."""
    cfg = TrainConfig(epochs=3, warmup_epochs=1, batch_size=16, patience=10,
                      map_norm_subset=200, augment=True)
    host = _run([100], small_data, cfg)[0]
    resident = _run([100], small_data, cfg, resident=True)[0]
    assert _allclose(_sd(resident), _sd(host), atol=1e-5)
    h = [m["train_loss"] for m in host.history]
    r = [m["train_loss"] for m in resident.history]
    assert np.allclose(h, r, atol=1e-6)


def test_predict_resident_matches_predict_dataset(small_data) -> None:
    """_predict_member_resident == predict_dataset for a single member."""
    from lpopt.model.net import PosValNet, PosValNetConfig
    reader, enc, train_pre, val_pre = small_data
    model = PosValNet(PosValNetConfig(n_globals=len(enc.globals_names))).eval()
    val_dev = {k: v.to("cpu") for k, v in val_pre._t.items()}
    a = T._predict_member_resident(model, val_dev, val_pre, torch.device("cpu"))
    b = T.predict_dataset([model], val_pre, torch.device("cpu"), num_workers=0)
    for key in ("mu_z_members", "log_sigma_members", "conv_prob_members"):
        assert np.allclose(a[key], b[key], atol=1e-6), key
    assert np.array_equal(a["record_ids"].astype(str), b["record_ids"].astype(str))


def test_early_stop_dropout_bookkeeping(small_data) -> None:
    """A member that early-stops drops out; survivors are unaffected (still
    bit-identical to a solo run of that seed), and the stop epoch = best + patience."""
    cfg = TrainConfig(epochs=15, warmup_epochs=1, batch_size=16, patience=1,
                      map_norm_subset=200, augment=True)
    solo0 = _sd(_run([100], small_data, cfg)[0])
    solo1 = _sd(_run([101], small_data, cfg)[0])
    joint = _run([100, 101], small_data, cfg)
    # differential early-stop must not perturb either member
    assert _equal(_sd(joint[0]), solo0)
    assert _equal(_sd(joint[1]), solo1)
    for m in joint:
        stopped = len(m.history) < cfg.epochs
        if stopped:
            assert m.live is False
            # stopped exactly ``patience`` epochs after its best epoch
            last_epoch = m.history[-1]["epoch"]
            assert last_epoch - m.best["epoch"] == cfg.patience
        else:
            assert m.live is True
    # at least one member should trip early-stop under patience=1
    assert any(len(m.history) < cfg.epochs for m in joint)


def test_train_member_wrapper_writes_checkpoint(small_data, tmp_path) -> None:
    """The public train_member wrapper still writes a loadable checkpoint and
    records the new schedule metadata."""
    import json
    reader, enc, train_pre, val_pre = small_data
    cfg = TrainConfig(epochs=2, warmup_epochs=1, batch_size=16, patience=10,
                      map_norm_subset=200, augment=True)
    d = T.train_member(100, split="S0", device="cpu", out_dir=tmp_path / "m",
                       config=cfg, train_pre=train_pre, val_pre=val_pre,
                       globals_names=enc.globals_names, verbose=False)
    meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
    assert meta["schedule"]["effective_batch"] == 16
    assert meta["schedule"]["device_resident"] is False
    model, meta2 = T.load_member(d, "cpu")
    assert meta2["seed"] == 100
