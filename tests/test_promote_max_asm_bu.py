"""Promoting ``max_assembly_burnup`` from advisory to a first-class target.

Today surrogate column 5 (the vendor's MAX-assembly-burnup CONSTRAINT axis) is
always NaN: our ``discharge_burnup`` target is the core AVERAGE, a different
physical quantity, and routing it through that column would corrupt the
assembly-burnup gate.  Behind ``promote_max_asm_bu`` the global head grows by one
output that regresses the real ``max_assembly_burnup`` label, masked wherever it
is absent, and column 5 starts carrying a model.

The flag-off half of every test below is the regression guard: 7 targets, the
same indices, column 5 still NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.data.fuel_types import FuelLibrary                       # noqa: E402
from lpopt.data.store import StoreReader                            # noqa: E402
from lpopt.model.dataset_torch import (                             # noqa: E402
    LPDataset, TARGETS, TARGETS_WITH_ASM_BU, targets_for,
)
from lpopt.model.model_api import _TARGET_TO_SURROGATE_COL, _to_surrogate  # noqa: E402
from lpopt.model.net import PosValNet, PosValNetConfig              # noqa: E402
from lpopt.model.train import TrainConfig, compute_target_norm      # noqa: E402
from lpopt.vendor.masterrl.surrogate import TARGET_NAMES            # noqa: E402

STORE = "data/store"


@pytest.fixture(scope="module")
def store_bits():
    reader = StoreReader(STORE)
    fuel = FuelLibrary.from_parquet(f"{STORE}/fuel_types.parquet")
    ids = reader.records["record_id"].astype(str).tolist()[:200]
    return reader, fuel, ids


# --------------------------------------------------------------------------- #
# target inventory
# --------------------------------------------------------------------------- #
def test_flag_off_keeps_the_seven_target_inventory():
    assert targets_for(False) == TARGETS
    assert len(TARGETS) == 7
    assert "max_assembly_burnup" not in TARGETS


def test_promotion_appends_and_never_reorders():
    """Appending (not inserting) is what keeps cyclen at index 3 — the index the
    rank loss, the cell calibration and the pin-BU physics all key on."""
    assert targets_for(True) == TARGETS_WITH_ASM_BU
    assert TARGETS_WITH_ASM_BU[:7] == TARGETS
    assert TARGETS_WITH_ASM_BU[7] == "max_assembly_burnup"
    assert TARGETS_WITH_ASM_BU.index("cyclen") == TARGETS.index("cyclen") == 3
    assert TARGETS_WITH_ASM_BU.index("f_r") == TARGETS.index("f_r") == 0


def test_train_config_defaults_to_unpromoted():
    assert TrainConfig().promote_max_asm_bu is False


# --------------------------------------------------------------------------- #
# dataset widths + masking
# --------------------------------------------------------------------------- #
def test_dataset_width_follows_the_flag(store_bits):
    reader, fuel, ids = store_bits
    off = LPDataset(reader, ids, fuel, fold="train")
    on = LPDataset(reader, ids, fuel, fold="train", promote_max_asm_bu=True)
    assert off[0]["targets"].shape == (7,)
    assert off[0]["target_mask"].shape == (7,)
    assert on[0]["targets"].shape == (8,)
    assert on[0]["target_mask"].shape == (8,)


def test_first_seven_targets_are_unchanged_by_promotion(store_bits):
    reader, fuel, ids = store_bits
    off = LPDataset(reader, ids, fuel, fold="train")
    on = LPDataset(reader, ids, fuel, fold="train", promote_max_asm_bu=True)
    for i in range(min(40, len(off))):
        a, b = off[i], on[i]
        # equal_nan: an unlabelled target is NaN, and NaN != NaN would make this
        # pass or fail on whichever rows the live store currently holds rather
        # than on whether promotion perturbed the first seven columns.
        torch.testing.assert_close(a["targets"], b["targets"][:7], equal_nan=True)
        torch.testing.assert_close(a["target_mask"], b["target_mask"][:7])
        torch.testing.assert_close(a["cells"], b["cells"])


def test_absent_labels_are_masked(store_bits):
    """The store carries max_assembly_burnup for ~94% of rows; the rest must be
    masked out rather than trained against a NaN."""
    reader, fuel, ids = store_bits
    ds = LPDataset(reader, ids, fuel, fold="train", promote_max_asm_bu=True)
    seen_valid = seen_masked = False
    for i in range(len(ds)):
        item = ds[i]
        row = ds.df.iloc[i]
        raw = row.get("max_assembly_burnup")
        finite = raw is not None and np.isfinite(float(raw))
        converged = bool(row["converged"])
        expect = 1.0 if (finite and converged) else 0.0
        assert float(item["target_mask"][7]) == expect
        seen_valid |= expect == 1.0
        seen_masked |= expect == 0.0
    assert seen_valid, "no promoted labels in the sample — test is vacuous"


def test_masked_rows_are_excluded_from_the_z_score(store_bits):
    reader, fuel, ids = store_bits
    df = reader.records[reader.records["record_id"].astype(str).isin(ids)]
    mean, std = compute_target_norm(df, TARGETS_WITH_ASM_BU)
    assert len(mean) == len(std) == 8
    vals = pd.to_numeric(df["max_assembly_burnup"], errors="coerce").to_numpy(float)
    ok = df["converged"].astype(bool).to_numpy() & np.isfinite(vals)
    assert mean[7] == pytest.approx(float(vals[ok].mean()))
    assert std[7] > 0.0
    # the first seven constants must be identical to the unpromoted call
    m7, s7 = compute_target_norm(df)
    np.testing.assert_allclose(mean[:7], m7)
    np.testing.assert_allclose(std[:7], s7)


# --------------------------------------------------------------------------- #
# the head + the surrogate column
# --------------------------------------------------------------------------- #
def test_global_head_grows_by_exactly_one_output():
    off = PosValNet(PosValNetConfig(in_channels=48, n_globals=13, n_targets=7))
    on = PosValNet(PosValNetConfig(in_channels=48, n_globals=13, n_targets=8))
    assert off.mu_head.out_features == 7
    assert on.mu_head.out_features == 8
    assert on.log_sigma_head.out_features == 8


def test_column_five_stays_nan_for_a_seven_target_checkpoint():
    cols = np.arange(7, dtype=float).reshape(1, 7)
    out = _to_surrogate(cols, TARGETS)
    assert np.isnan(out[0, 5]), "unpromoted checkpoints must leave column 5 unknown"
    assert out[0, 3] == cols[0, TARGETS.index("cyclen")]


def test_promoted_checkpoint_fills_column_five():
    cols = np.arange(8, dtype=float).reshape(1, 8)
    out = _to_surrogate(cols, TARGETS_WITH_ASM_BU)
    assert out[0, 5] == cols[0, 7]
    # every other column keeps its meaning
    for name, scol in _TARGET_TO_SURROGATE_COL.items():
        if name in TARGETS_WITH_ASM_BU:
            assert out[0, scol] == cols[0, TARGETS_WITH_ASM_BU.index(name)]


def test_discharge_burnup_never_reaches_the_constraint_column():
    """The whole reason column 5 was NaN: discharge_burnup is an AVERAGE, not the
    vendor's MAX-assembly constraint, and must never be routed there."""
    assert "discharge_burnup" not in _TARGET_TO_SURROGATE_COL
    assert _TARGET_TO_SURROGATE_COL["max_assembly_burnup"] == 5
    assert TARGET_NAMES[5] == "max_assembly_burnup"
