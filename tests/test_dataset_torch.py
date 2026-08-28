"""Torch dataset (plan sec. 4.4): item schema, raw targets + convergence/boc/map
masking, transpose augmentation, DataLoader collation, and the inverse-sqrt
(feed, e_core-bin, dataset) cell-weighting.  Skips cleanly without torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.data.fuel_types import FuelLibrary          # noqa: E402
from lpopt.data.store import StoreReader                # noqa: E402
from lpopt.model.dataset_torch import (                 # noqa: E402
    LPDataset,
    TARGETS,
    compute_cell_weights,
)
from lpopt.model.splits import make_splits              # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


def _fixtures():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("Dataset-A store not present")
    reader = StoreReader(STORE)
    fl = FuelLibrary.from_parquet(STORE / "fuel_types.parquet")
    return reader, fl


@pytest.fixture(scope="module")
def fixtures():
    return _fixtures()


def _small_ids(reader, n=200):
    return reader.records["record_id"].astype(str).head(n).tolist()


# --------------------------------------------------------------------------- #
# item schema
# --------------------------------------------------------------------------- #
def test_item_schema_shapes_and_dtypes(fixtures) -> None:
    reader, fl = fixtures
    ds = LPDataset(reader, _small_ids(reader), fl)
    item = ds[0]
    assert item["cells"].shape == (26, 19, 19)
    assert item["cells"].dtype == torch.float32
    assert item["globals"].ndim == 1
    assert item["targets"].shape == (len(TARGETS),)
    assert item["target_mask"].shape == (len(TARGETS),)
    assert item["maps"].shape == (4, 9, 9)
    assert item["maps_mask"].shape == (4, 9, 9)
    assert item["conv_label"].ndim == 0
    assert item["conv_mask"].ndim == 0
    assert isinstance(item["record_id"], str)


def test_targets_are_raw_not_normalized(fixtures) -> None:
    reader, fl = fixtures
    ids = _small_ids(reader, 8)
    ds = LPDataset(reader, ids, fl)
    df = reader.records.set_index("record_id")
    rid = ds.record_ids[0]
    item = ds[0]
    for k, name in enumerate(TARGETS):
        expected = float(df.loc[rid, name])
        if item["target_mask"][k] > 0:
            assert item["targets"][k].item() == pytest.approx(expected, rel=1e-5)


def test_targets_include_burnup_axes() -> None:
    # Phase D promoted the two burnup axes to first-class targets (plan sec. 12.4).
    assert TARGETS == ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                       "discharge_burnup", "max_pin_burnup")


def test_per_target_nan_censoring(fixtures) -> None:
    """A converged Dataset B/P row with NaN burnup labels is censored on those
    TWO axes only — the other five still train (per-target mask, plan sec. 12.4)."""
    reader, fl = fixtures
    df = reader.records
    di = TARGETS.index("discharge_burnup")
    pi = TARGETS.index("max_pin_burnup")
    fri = TARGETS.index("f_r")
    cyi = TARGETS.index("cyclen")
    # a converged row whose burnup labels are NaN (Dataset B/P rows carry none)
    burnup = pd.to_numeric(df["max_pin_burnup"], errors="coerce")
    conv = df["converged"].astype(bool)
    sel = df[conv & burnup.isna() & df["f_r"].notna() & df["cyclen"].notna()]
    if sel.empty:
        pytest.skip("no converged row with NaN burnup labels")
    ds = LPDataset(reader, sel["record_id"].astype(str).head(1).tolist(), fl)
    item = ds[0]
    # burnup axes censored...
    assert item["target_mask"][di].item() == 0.0
    assert item["target_mask"][pi].item() == 0.0
    # ...while the standard axes with finite labels still train.
    assert item["target_mask"][fri].item() == 1.0
    assert item["target_mask"][cyi].item() == 1.0


def _a_burnup_ids(df, n=1):
    a = df[(df["dataset"] == "A") & df["converged"].astype(bool)
           & pd.to_numeric(df["max_pin_burnup"], errors="coerce").notna()
           & pd.to_numeric(df["discharge_burnup"], errors="coerce").notna()
           & df["f_r"].notna() & df["cyclen"].notna()]
    return a["record_id"].astype(str).head(n).tolist()


def test_dataset_a_pin_label_censored_by_default(fixtures) -> None:
    """Forensic censoring (default ON): a converged Dataset-A row's
    ``max_pin_burnup`` label is masked — it is a MOCHA-cache surrogate, not a real
    pin calc (data/reports/pinbu_forensics.md) — while every OTHER target,
    including ``discharge_burnup``, stays trainable (unchanged)."""
    reader, fl = fixtures
    ids = _a_burnup_ids(reader.records)
    if not ids:
        pytest.skip("no Dataset-A row with burnup labels")
    ds = LPDataset(reader, ids, fl)                     # default: censoring ON
    item = ds[0]
    assert item["target_mask"][TARGETS.index("max_pin_burnup")].item() == 0.0
    for name in ("f_r", "cyclen", "discharge_burnup"):
        assert item["target_mask"][TARGETS.index(name)].item() == 1.0


def test_censor_flag_off_restores_a_pin_label(fixtures) -> None:
    """``censor_dataset_a_pin_labels=False`` restores the legacy behavior: a
    converged Dataset-A row trains its ``max_pin_burnup`` label (mask 1)."""
    reader, fl = fixtures
    ids = _a_burnup_ids(reader.records)
    if not ids:
        pytest.skip("no Dataset-A row with burnup labels")
    ds = LPDataset(reader, ids, fl, censor_dataset_a_pin_labels=False)
    item = ds[0]
    assert item["target_mask"][TARGETS.index("discharge_burnup")].item() == 1.0
    assert item["target_mask"][TARGETS.index("max_pin_burnup")].item() == 1.0


def test_dataset_p_pin_label_kept_under_censoring(fixtures) -> None:
    """Censoring targets Dataset A ONLY: a converged Dataset-P row carrying a real
    (MAS_PPI) pin label keeps its ``max_pin_burnup`` mask under the default."""
    reader, fl = fixtures
    df = reader.records
    p = df[(df["dataset"] == "P") & df["converged"].astype(bool)
           & pd.to_numeric(df["max_pin_burnup"], errors="coerce").notna()]
    if p.empty:
        pytest.skip("no converged Dataset-P row with a pin label")
    ds = LPDataset(reader, p["record_id"].astype(str).head(1).tolist(), fl)
    item = ds[0]
    assert item["target_mask"][TARGETS.index("max_pin_burnup")].item() == 1.0


def test_boc_only_masks_cbc_max(fixtures) -> None:
    reader, fl = fixtures
    df = reader.records
    boc = df[df["cbc_kind"] == "boc_only"].head(1)
    if boc.empty:
        pytest.skip("no boc_only records")
    ds = LPDataset(reader, boc["record_id"].astype(str).tolist(), fl)
    item = ds[0]
    ci = TARGETS.index("cbc_max")
    assert item["target_mask"][ci].item() == 0.0


def test_maps_absent_is_nan_filled_and_masked(fixtures) -> None:
    reader, fl = fixtures
    df = reader.records
    absent = df[df["maps_key"].isna()].head(1)
    if absent.empty:
        pytest.skip("every record has maps")
    ds = LPDataset(reader, absent["record_id"].astype(str).tolist(), fl)
    item = ds[0]
    assert item["maps_mask"].sum().item() == 0.0
    assert torch.isnan(item["maps"]).all()


def test_maps_present_mask_marks_finite_cells(fixtures) -> None:
    reader, fl = fixtures
    df = reader.records
    present = df[df["maps_key"].notna()].head(1)
    if present.empty:
        pytest.skip("no records carry maps")
    ds = LPDataset(reader, present["record_id"].astype(str).tolist(), fl)
    item = ds[0]
    mask = item["maps_mask"].bool()
    # the mask is exactly the finite cells of the (NaN-cornered) quarter map
    assert torch.equal(mask, torch.isfinite(item["maps"]))
    assert mask.sum().item() > 0


def test_conv_mask_unknown_when_converged_at_cap(fixtures) -> None:
    reader, fl = fixtures
    df = reader.records
    ds = LPDataset(reader, _small_ids(reader, 4), fl)
    item = ds[0]
    row = df.set_index("record_id").loc[ds.record_ids[0]]
    expected_mask = 0.0 if bool(row["converged_at_cap"]) else 1.0
    assert item["conv_mask"].item() == expected_mask
    assert item["conv_label"].item() == (1.0 if bool(row["converged"]) else 0.0)


# --------------------------------------------------------------------------- #
# augmentation + collation
# --------------------------------------------------------------------------- #
def test_augment_preserves_target_and_map(fixtures) -> None:
    reader, fl = fixtures
    ids = _small_ids(reader, 20)
    plain = LPDataset(reader, ids, fl, augment=False)
    aug = LPDataset(reader, ids, fl, augment=True, seed=1)
    # augmentation only touches (cells, globals); labels/masks are untouched.
    # NaN-aware: an unlabelled target is stored as NaN and NaN != NaN, so a plain
    # torch.equal here fails or passes purely on which rows the live store
    # happens to hold.  "Identical including where both are NaN" is the contract.
    for k in range(5):
        a, b = plain[k], aug[k]
        torch.testing.assert_close(a["targets"], b["targets"], equal_nan=True,
                                   rtol=0, atol=0)
        assert torch.equal(a["target_mask"], b["target_mask"])
        assert a["globals"].shape == b["globals"].shape


def test_dataloader_collates_batch(fixtures) -> None:
    reader, fl = fixtures
    from torch.utils.data import DataLoader

    ds = LPDataset(reader, _small_ids(reader, 32), fl)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert batch["cells"].shape == (8, 26, 19, 19)
    assert batch["globals"].shape[0] == 8
    assert batch["targets"].shape == (8, len(TARGETS))
    assert batch["maps"].shape == (8, 4, 9, 9)
    assert len(batch["record_id"]) == 8


def test_fold_selection_matches_manifest(fixtures) -> None:
    reader, fl = fixtures
    manifests = make_splits(reader.records, seed=3, persist=False)
    s0 = manifests["S0"]
    train_ds = LPDataset(reader, s0, fl, fold="train")
    val_ds = LPDataset(reader, s0, fl, fold="val")
    assert len(train_ds) == s0.n_train
    assert len(val_ds) == s0.n_val
    assert set(train_ds.record_ids).isdisjoint(set(val_ds.record_ids))


# --------------------------------------------------------------------------- #
# cell weighting
# --------------------------------------------------------------------------- #
def test_compute_cell_weights_inverse_sqrt_capped(fixtures) -> None:
    reader, _ = fixtures
    df = reader.records
    weights, summary = compute_cell_weights(df, cap=8.0)
    assert weights.shape == (len(df),)
    assert weights.min() >= 1.0 - 1e-6
    assert weights.max() <= 8.0 + 1e-6
    assert summary["n_cells"] >= 1
    assert summary["n_rows"] == len(df)
    # rarer cells carry strictly more weight than the most common cell (=1.0)
    assert weights.max() > weights.min()
    # effective mass reported per cell
    assert all("effective_mass" in v for v in summary["cells"].values())


def test_cell_weights_cap_binds_on_skew() -> None:
    import pandas as pd

    # one common cell (100 rows) + one ultra-rare cell (1 row): 1/sqrt ratio = 10,
    # so the cap at 8.0 must bind.
    rows = [{"feed": 121, "e_core": 5.40, "dataset": "A"} for _ in range(100)]
    rows.append({"feed": 105, "e_core": 5.10, "dataset": "A"})
    df = pd.DataFrame(rows)
    weights, summary = compute_cell_weights(df, cap=8.0)
    assert weights.max() == pytest.approx(8.0)
    assert summary["cap_hits"] == 1


def test_cell_weights_curriculum_cap_override() -> None:
    """A higher ``curriculum_cap`` un-caps ONLY the curriculum-cell rows
    (``dataset=='P'`` with a ``campaign`` in ``curriculum_campaigns``); the legacy
    corpus — including non-curriculum Dataset-P rows — keeps the global cap."""
    import pandas as pd

    # common legacy cell (100 A rows -> normalizes to 1.0), plus two rare 1-row
    # cells with the SAME 1/sqrt ratio of 10 relative to the common cell:
    #   - a curriculum cell (dataset P, campaign is a known cell) and
    #   - a legacy P0 pathfinder row (dataset P, campaign NOT a curriculum cell).
    rows = [{"feed": 121, "e_core": 5.40, "dataset": "A", "campaign": "A::cache"}
            for _ in range(100)]
    rows.append({"feed": 117, "e_core": 5.10, "dataset": "P",
                 "campaign": "5-5.25_f117"})       # curriculum cell
    rows.append({"feed": 117, "e_core": 5.12, "dataset": "P",
                 "campaign": "P0_pathfinder"})     # legacy P (NOT a cell)
    df = pd.DataFrame(rows)
    curr = df["campaign"].eq("5-5.25_f117").to_numpy()
    p0 = df["campaign"].eq("P0_pathfinder").to_numpy()

    # baseline: both rare rows want weight 10 but the global cap binds at 8.
    w0, s0 = compute_cell_weights(df, cap=8.0)
    assert w0[curr][0] == pytest.approx(8.0)
    assert w0[p0][0] == pytest.approx(8.0)
    assert s0["cap_hits"] == 2
    assert s0["n_curriculum_rows"] == 0
    assert s0["curriculum_cap"] is None

    # override cap 16: the curriculum row un-caps to its true weight 10 (< 16),
    # the legacy P0 row stays capped at the global 8.
    w1, s1 = compute_cell_weights(
        df, cap=8.0,
        curriculum_campaigns=["5-5.25_f117", "5.25-5.5_f117"], curriculum_cap=16.0)
    assert w1[curr][0] == pytest.approx(10.0)      # curriculum row un-capped
    assert w1[p0][0] == pytest.approx(8.0)         # legacy P row still capped
    assert w1.max() == pytest.approx(10.0)
    assert s1["curriculum_cap"] == 16.0
    assert s1["n_curriculum_rows"] == 1
    assert s1["curriculum_cap_hits"] == 0          # 10 < 16, so the raised cap never binds


def test_cell_weights_curriculum_cap_binds_when_high_enough() -> None:
    """If a curriculum cell is rare enough that even the raised cap binds, it is
    counted in ``curriculum_cap_hits`` and the legacy cap is untouched."""
    import pandas as pd

    # 400 common rows -> a 1-row curriculum cell wants 1/sqrt ratio = 20 > 16.
    rows = [{"feed": 121, "e_core": 5.40, "dataset": "A", "campaign": "A::cache"}
            for _ in range(400)]
    rows.append({"feed": 117, "e_core": 5.10, "dataset": "P",
                 "campaign": "5-5.25_f117"})
    df = pd.DataFrame(rows)
    curr = df["campaign"].eq("5-5.25_f117").to_numpy()
    w, s = compute_cell_weights(
        df, cap=8.0, curriculum_campaigns=["5-5.25_f117"], curriculum_cap=16.0)
    assert w[curr][0] == pytest.approx(16.0)        # raised cap binds
    assert s["curriculum_cap_hits"] == 1
    assert s["cap_hits"] == 0                        # nothing hit the legacy 8.0
