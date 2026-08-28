"""Two-stage per-cell cyclen bias correction (lpopt.model.cell_calibrate).

Stage 1 — serve-side per-cell affine calibration:
  * synthetic biased predictions -> recovered (a, b) [intercept + affine];
  * the ``min_rows`` gate + the intercept-vs-affine cross-fit choice;
  * the (feed, e_core-bin) cell-key recipe == the weighting cells (minus dataset);
  * no-contamination: the fit-row selector never admits a holdout id, and only
    serve-library converged+labelled rows;
  * serve-side application on a real backend: calibrated vs raw on a fitted cell,
    identity on an unfitted cell, sigma + non-cyclen columns untouched, disable
    flag, and ``from_dir`` load round-trip.

Stage 2 — campaign running bias corrector:
  * shrinkage math (n/(n+prior) * median), fitted-cell skip, NaN guard, resume
    round-trip, and its effect threaded through score_(pool_)user_criteria.

CBC_max calibration (2026-07-29 debug-panel) — the third instance of the same
Stage-1 machinery, on surrogate column 1:
  * ppm-scale fit (intercept + affine at the CBC OOF margin);
  * the ``cbc_kind == "boc_only"`` fit-row censor;
  * serve-side application, disjointness from the cyclen/F_r hooks, and the
    BACKWARD-COMPAT contract: a checkpoint with NO ``cbc_calibration.json``
    loads and predicts byte-identically.

Map-head OOD sigma floor (same forensic): the floor is applied to the flatness
spread, resolved from the backend manifest / member meta when present, and falls
back to the documented defaults when neither exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.model.cell_calibrate import (
    CampaignBiasCorrector, apply_cyclen_calibration, calibration_cells,
    cyclen_cell_key, fit_affine_cell, fit_row_mask, load_cell_calibration,
)
from lpopt.model.dataset_torch import compute_cell_weights

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


# --------------------------------------------------------------------------- #
# Stage 1 — affine fit (pure, torch-free)
# --------------------------------------------------------------------------- #
def test_fit_recovers_uniform_bias() -> None:
    rng = np.random.default_rng(0)
    actual = rng.normal(625.0, 10.0, 240)
    pred = actual + 8.5 + rng.normal(0.0, 2.0, 240)          # +8.5 EFPD over-predict
    fit = fit_affine_cell(pred, actual)
    assert fit is not None
    assert fit["estimator"] == "intercept"                   # uniform shift wins
    assert fit["a"] == 1.0
    # b == -median_bias centres the prediction on the actual
    assert fit["b"] == pytest.approx(-8.5, abs=1.0)
    assert fit["median_bias"] == pytest.approx(8.5, abs=1.0)
    assert fit["mae_after"] < fit["mae_before"]              # correction helps
    cal = apply_cyclen_calibration(pred, ["k"] * pred.size, {"k": fit})
    assert abs(np.median(cal - actual)) < 1.0               # residual bias removed


def test_fit_recovers_affine_slope() -> None:
    rng = np.random.default_rng(1)
    actual = rng.normal(625.0, 15.0, 400)
    pred = 1.3 * actual - 180.0 + rng.normal(0.0, 1.5, 400)   # genuine slope distortion
    fit = fit_affine_cell(pred, actual)
    assert fit is not None
    assert fit["estimator"] == "affine"                      # cross-fit picks slope
    cal = fit["a"] * pred + fit["b"]
    assert float(np.median(np.abs(cal - actual))) < 1.5      # recovers the map
    assert fit["oof_affine_mae"] < fit["oof_intercept_mae"]


def test_fit_min_rows_gate() -> None:
    rng = np.random.default_rng(2)
    pred = rng.normal(600.0, 5.0, 10)
    assert fit_affine_cell(pred, pred + 5.0, min_rows=30) is None
    # exactly at the threshold it fits
    pred30 = rng.normal(600.0, 5.0, 30)
    assert fit_affine_cell(pred30, pred30 + 5.0, min_rows=30) is not None


def test_fit_prefers_intercept_when_slope_near_one() -> None:
    # the champion's real regime: a near-constant per-cell shift with little
    # independent prediction noise -> slope ~= 1, so the cross-fit margin is not
    # met and the parsimonious intercept-only correction is chosen (19/20 real
    # cells land here).  (A genuinely attenuated pred = actual + big-noise would
    # legitimately pick affine — that is regression-to-the-mean, not overfitting.)
    rng = np.random.default_rng(3)
    actual = rng.normal(625.0, 12.0, 150)
    pred = actual + 10.0 + rng.normal(0.0, 1.5, 150)
    fit = fit_affine_cell(pred, actual, slope_min_rows=50)
    assert fit["estimator"] == "intercept"
    assert fit["b"] == pytest.approx(-10.0, abs=1.0)


def test_fit_drops_nan_pairs() -> None:
    pred = np.array([610.0, 620.0, np.nan, 630.0] * 20)
    actual = np.array([600.0, 610.0, 615.0, np.nan] * 20)
    fit = fit_affine_cell(pred, actual, min_rows=10)
    assert fit is not None
    assert fit["n"] == 40                                     # only the finite pairs


# --------------------------------------------------------------------------- #
# cell key parity with the weighting cells
# --------------------------------------------------------------------------- #
def test_cell_key_recipe_matches_weighting_cells() -> None:
    # same floor(e/w)*w rounding as compute_cell_weights, minus the dataset axis.
    df = pd.DataFrame({
        "feed": [121, 121, 117, 109],
        "e_core": [5.42, 5.399, 5.0, np.nan],
        "dataset": ["B", "B", "P", "P"],
    })
    _, summary = compute_cell_weights(df, e_core_bin_width=0.05)
    weight_bins = set()
    for label in summary["cells"]:
        parts = dict(kv.split("=", 1) for kv in label.split("|"))
        weight_bins.add((int(parts["feed"]), parts["ebin"]))
    for feed, e in zip(df["feed"], df["e_core"]):
        key = cyclen_cell_key(feed, None if pd.isna(e) else e)
        f, ebin = key.split("|")
        assert f == f"feed={feed}"
        assert (feed, ebin.split("=", 1)[1]) in weight_bins  # identical bin string

    assert cyclen_cell_key(121, 5.42) == "feed=121|ebin=5.4"
    assert cyclen_cell_key(121, 5.399) == "feed=121|ebin=5.35"
    assert cyclen_cell_key(117, None) == "feed=117|ebin=None"
    assert cyclen_cell_key(117, float("nan")) == "feed=117|ebin=None"


# --------------------------------------------------------------------------- #
# no-contamination: the fit-row selector
# --------------------------------------------------------------------------- #
def test_fit_row_mask_never_admits_holdout() -> None:
    records = pd.DataFrame({
        "record_id": ["t1", "t2", "v1", "t3", "t4", "t5"],
        "converged": [True, True, True, False, True, True],
        "cyclen": [620.0, 630.0, 625.0, 610.0, np.nan, 640.0],
        "library_id": ["ga80", "ga80", "ga80", "ga80", "ga80", "260624"],
    })
    train_ids = {"t1", "t2", "t3", "t4", "t5"}
    val_ids = {"v1"}
    mask = fit_row_mask(records, train_ids, library_id="ga80")
    chosen = set(records.loc[mask, "record_id"])
    # holdout id never enters the fit (the leakage guarantee)
    assert not (chosen & val_ids)
    # t3 dropped (non-converged), t4 dropped (NaN cyclen), t5 dropped (foreign lib)
    assert chosen == {"t1", "t2"}


def test_fit_row_mask_no_library_filter() -> None:
    records = pd.DataFrame({
        "record_id": ["a", "b"],
        "converged": [True, True],
        "cyclen": [620.0, 630.0],
        "library_id": ["ga80", "260624"],
    })
    mask = fit_row_mask(records, {"a", "b"}, library_id=None)
    assert set(records.loc[mask, "record_id"]) == {"a", "b"}   # both kept


# --------------------------------------------------------------------------- #
# vectorized apply helper
# --------------------------------------------------------------------------- #
def test_apply_cyclen_calibration_fitted_and_identity() -> None:
    cyclen = np.array([700.0, 650.0, np.nan, 680.0])
    keys = ["fit", "none", "fit", "none"]
    cells = {"fit": {"a": 0.5, "b": 100.0}}
    src = cyclen.copy()
    out = apply_cyclen_calibration(cyclen, keys, cells)
    assert out[0] == pytest.approx(0.5 * 700.0 + 100.0)        # fitted cell affine
    assert out[1] == 650.0                                     # unfitted -> identity
    assert np.isnan(out[2])                                    # NaN passthrough
    assert out[3] == 680.0
    assert np.array_equal(cyclen, src, equal_nan=True)         # input not mutated
    # no cells -> pure identity copy
    assert np.array_equal(apply_cyclen_calibration(cyclen, keys, {}), src, equal_nan=True)


def test_calibration_cells_accessor() -> None:
    assert calibration_cells(None) == {}
    assert calibration_cells({"cells": {"k": {"a": 1.0, "b": 0.0}}}) == {"k": {"a": 1.0, "b": 0.0}}
    assert calibration_cells({"no_cells": 1}) == {}


# --------------------------------------------------------------------------- #
# Task A — F_r calibration: pure fit + apply (torch-free)
# --------------------------------------------------------------------------- #
from lpopt.model.cell_calibrate import (            # noqa: E402
    _AFFINE_MARGIN_FR, apply_affine_calibration,
)


def test_fr_fit_recovers_uniform_underbias() -> None:
    # The champion UNDER-predicts F_r by a near-constant per-cell shift (pred <
    # actual by ~0.25), the opposite sign of the cyclen over-prediction.  The
    # intercept-only shift b = -median(pred-actual) = +0.25 recovers it.
    rng = np.random.default_rng(10)
    actual = rng.normal(1.75, 0.06, 240)                    # F_r-scale target
    pred = actual - 0.25 + rng.normal(0.0, 0.01, 240)       # under-predicts by 0.25
    fit = fit_affine_cell(pred, actual, affine_margin=_AFFINE_MARGIN_FR)
    assert fit is not None
    assert fit["estimator"] == "intercept"                  # uniform shift wins
    assert fit["a"] == 1.0                                   # a>0 -> monotone
    assert fit["b"] == pytest.approx(0.25, abs=0.03)         # +0.25 correction
    assert fit["median_bias"] == pytest.approx(-0.25, abs=0.03)
    assert fit["mae_after"] < fit["mae_before"]
    cal = apply_affine_calibration(pred, ["k"] * pred.size, {"k": fit})
    assert abs(np.median(cal - actual)) < 0.03              # residual bias removed


def test_fr_fit_recovers_affine_slope_at_fr_margin() -> None:
    # A genuine slope distortion at F_r scale is still detectable with the smaller
    # F_r-scale OOF margin (the cyclen EFPD margin 0.25 would swamp it).
    rng = np.random.default_rng(11)
    actual = rng.normal(2.2, 0.35, 400)
    pred = 1.3 * actual - 0.66 + rng.normal(0.0, 0.03, 400)
    fit = fit_affine_cell(pred, actual, affine_margin=_AFFINE_MARGIN_FR)
    assert fit is not None
    assert fit["estimator"] == "affine"
    assert 0.2 <= fit["a"] <= 3.0                            # in-range positive slope
    cal = fit["a"] * pred + fit["b"]
    assert float(np.median(np.abs(cal - actual))) < 0.05


def test_apply_affine_calibration_alias_and_monotone() -> None:
    # apply_cyclen_calibration is the same object as apply_affine_calibration.
    assert apply_cyclen_calibration is apply_affine_calibration
    # per-cell distinct a>0 preserves WITHIN-cell ranking (gate neutrality).
    raw = np.array([1.60, 1.90, 1.55, 2.30, 1.72, 2.05])
    keys = ["cellX", "cellY", "cellX", "cellY", "cellX", "cellY"]
    cells = {"cellX": {"a": 0.5, "b": 0.9}, "cellY": {"a": 2.0, "b": -1.0}}
    cal = apply_affine_calibration(raw, keys, cells)
    for cell in ("cellX", "cellY"):
        idx = [i for i, k in enumerate(keys) if k == cell]
        r, c = raw[idx], cal[idx]
        # strictly-increasing map -> identical within-cell ranking
        assert np.array_equal(np.argsort(r, kind="stable"),
                              np.argsort(c, kind="stable"))


# --------------------------------------------------------------------------- #
# Stage 1 — serve-side application on a real backend
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")

from lpopt.model.featurize import CHANNELS_V4, FeatureEncoder      # noqa: E402
from lpopt.model.model_api import PosValCnnBackend                 # noqa: E402
from lpopt.model.net import PosValNet, PosValNetConfig             # noqa: E402
from lpopt.model.train import save_member                          # noqa: E402
from lpopt.data.schema import unpack_pattern                       # noqa: E402
from lpopt.data.store import StoreReader                           # noqa: E402
from lpopt.vendor.masterrl.domain import CaseKey                   # noqa: E402

_ZMEAN = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0]
_ZSTD = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0]
_TARGETS = ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
            "discharge_burnup", "max_pin_burnup")


def _make_v4_ensemble(tmp: Path, n: int = 2) -> Path:
    ens = tmp / "ens"
    globals_names = list(FeatureEncoder(cond_schema="v4").globals_names)
    cfg = PosValNetConfig(in_channels=len(CHANNELS_V4), n_globals=len(globals_names))
    for i in range(n):
        seed = 500 + i
        net = PosValNet(cfg)
        meta = {
            "net_config": cfg.__dict__, "cond_schema": "v4",
            "channels": list(CHANNELS_V4), "globals": globals_names,
            "target_names": list(_TARGETS),
            "target_zscore": {"mean": _ZMEAN, "std": _ZSTD},
            "seed": seed, "split": "S1", "versions": {"torch": torch.__version__},
        }
        save_member(ens / f"member_{seed}", net, meta)
    return ens


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path_factory.mktemp("cellcalib"), n=2)
    return PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")


@pytest.fixture(scope="module")
def sample_case(backend):
    reader = StoreReader(STORE)
    row = reader.records[reader.records["library_id"].astype(str) == "ga80"].iloc[0]
    pat = unpack_pattern(str(row["pattern"]))
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    return pat, case


def test_backend_applies_affine_on_fitted_cell(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_cell_calibration = False
    raw = backend.predict([pat], case, 0.0)
    raw_cy = float(raw.mean[0, 3])
    key = backend._cyclen_cell_keys([pat], [case])[0]

    a, b = 0.5, 123.0
    backend.set_cell_calibration(
        {"bin_width": 0.05, "cells": {key: {"a": a, "b": b, "n": 99}}}, enabled=True
    )
    cal = backend.predict([pat], case, 0.0)
    # cyclen column follows a*raw + b …
    assert float(cal.mean[0, 3]) == pytest.approx(a * raw_cy + b, abs=1e-4)
    # … and NOTHING else moves: other mean columns + sigma are byte-identical.
    for col in (0, 1, 2, 4, 5, 6):
        assert np.array_equal(cal.mean[:, col], raw.mean[:, col], equal_nan=True)
    assert np.array_equal(cal.epistemic_std, raw.epistemic_std, equal_nan=True)
    assert np.array_equal(cal.calibrated_std, raw.calibrated_std, equal_nan=True)
    backend.set_cell_calibration(None)


def test_backend_identity_on_unfitted_cell(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_cell_calibration = False
    raw_cy = float(backend.predict([pat], case, 0.0).mean[0, 3])
    backend.set_cell_calibration(
        {"bin_width": 0.05,
         "cells": {"feed=999|ebin=9.9": {"a": 0.0, "b": 0.0, "n": 50}}},
        enabled=True,
    )
    cal_cy = float(backend.predict([pat], case, 0.0).mean[0, 3])
    assert cal_cy == pytest.approx(raw_cy, abs=1e-4)          # cell not fitted -> id
    backend.set_cell_calibration(None)


def test_backend_disable_flag(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_cell_calibration = False
    raw_cy = float(backend.predict([pat], case, 0.0).mean[0, 3])
    key = backend._cyclen_cell_keys([pat], [case])[0]
    backend.set_cell_calibration(
        {"bin_width": 0.05, "cells": {key: {"a": 1.0, "b": 500.0}}}, enabled=False
    )
    assert float(backend.predict([pat], case, 0.0).mean[0, 3]) == pytest.approx(raw_cy, abs=1e-4)
    assert backend.fitted_cyclen_cells() == set()            # disabled -> reports none
    backend.set_cell_calibration(None)


def test_from_dir_loads_and_applies(tmp_path) -> None:
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path, n=2)
    reader = StoreReader(STORE)
    row = reader.records[reader.records["library_id"].astype(str) == "ga80"].iloc[0]
    pat = unpack_pattern(str(row["pattern"]))
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))

    raw_backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    raw_cy = float(raw_backend.predict([pat], case, 0.0).mean[0, 3])
    key = raw_backend._cyclen_cell_keys([pat], [case])[0]

    (ens / "cell_calibration.json").write_text(json.dumps({
        "schema": "cell_cyclen_affine_v1", "bin_width": 0.05,
        "cells": {key: {"a": 1.0, "b": -15.0, "n": 80, "estimator": "intercept"}},
    }), encoding="utf-8")
    loaded = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert key in loaded.fitted_cyclen_cells()
    assert float(loaded.predict([pat], case, 0.0).mean[0, 3]) == pytest.approx(raw_cy - 15.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# Task A — F_r calibration: serve-side application on a real backend
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_batch(backend):
    """~24 ga80 patterns (they cluster into one F_r cell — see the fit probe)."""
    reader = StoreReader(STORE)
    sub = reader.records[reader.records["library_id"].astype(str) == "ga80"].head(24)
    pats = [unpack_pattern(str(p)) for p in sub["pattern"]]
    cases = [CaseKey(pair=str(cp), feed=int(f))
             for cp, f in zip(sub["case_pair"], sub["feed"])]
    return pats, cases


def test_fr_absent_artifact_is_byte_identical(backend, sample_case) -> None:
    # INVARIANT (i): with NO F_r artifact loaded, predict() is byte-identical
    # whether the apply flag is on or off — the deployed champion is unaffected.
    pat, case = sample_case
    backend.set_fr_calibration(None)                          # ensure absent
    backend.apply_fr_calibration = False
    off = backend.predict([pat], case, 0.0)
    backend.apply_fr_calibration = True                       # flag on, but no cells
    on = backend.predict([pat], case, 0.0)
    assert np.array_equal(off.mean, on.mean, equal_nan=True)
    assert np.array_equal(off.epistemic_std, on.epistemic_std, equal_nan=True)
    assert np.array_equal(off.calibrated_std, on.calibrated_std, equal_nan=True)
    assert backend.fitted_fr_cells() == set()                 # nothing fitted


def test_backend_applies_fr_affine_on_fitted_cell(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_fr_calibration = False
    raw = backend.predict([pat], case, 0.0)
    raw_fr = float(raw.mean[0, 0])
    key = backend._fr_cell_keys([pat], [case])[0]

    a, b = 0.8, 0.33
    backend.set_fr_calibration(
        {"bin_width": 0.05, "cells": {key: {"a": a, "b": b, "n": 99}}}, enabled=True
    )
    cal = backend.predict([pat], case, 0.0)
    # F_r column (0) follows a*raw + b …
    assert float(cal.mean[0, 0]) == pytest.approx(a * raw_fr + b, abs=1e-4)
    # … and NOTHING else moves: other mean columns + sigma are byte-identical.
    for col in (1, 2, 3, 4, 5, 6):
        assert np.array_equal(cal.mean[:, col], raw.mean[:, col], equal_nan=True)
    assert np.array_equal(cal.epistemic_std, raw.epistemic_std, equal_nan=True)
    assert np.array_equal(cal.calibrated_std, raw.calibrated_std, equal_nan=True)
    backend.set_fr_calibration(None)


def test_backend_fr_identity_on_unfitted_cell(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_fr_calibration = False
    raw_fr = float(backend.predict([pat], case, 0.0).mean[0, 0])
    backend.set_fr_calibration(
        {"bin_width": 0.05,
         "cells": {"feed=999|ebin=9.9": {"a": 0.0, "b": 0.0, "n": 50}}},
        enabled=True,
    )
    cal_fr = float(backend.predict([pat], case, 0.0).mean[0, 0])
    assert cal_fr == pytest.approx(raw_fr, abs=1e-4)           # cell not fitted -> id
    backend.set_fr_calibration(None)


def test_backend_fr_disable_flag(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_fr_calibration = False
    raw_fr = float(backend.predict([pat], case, 0.0).mean[0, 0])
    key = backend._fr_cell_keys([pat], [case])[0]
    backend.set_fr_calibration(
        {"bin_width": 0.05, "cells": {key: {"a": 1.0, "b": 0.5}}}, enabled=False
    )
    assert float(backend.predict([pat], case, 0.0).mean[0, 0]) == pytest.approx(raw_fr, abs=1e-4)
    assert backend.fitted_fr_cells() == set()                 # disabled -> reports none
    backend.set_fr_calibration(None)


def test_fr_and_cyclen_calibration_touch_disjoint_columns(backend, sample_case) -> None:
    # The two hooks are independent: F_r moves only col 0, cyclen only col 3.
    pat, case = sample_case
    backend.apply_fr_calibration = False
    backend.apply_cell_calibration = False
    raw = backend.predict([pat], case, 0.0)
    fr_key = backend._fr_cell_keys([pat], [case])[0]
    cy_key = backend._cyclen_cell_keys([pat], [case])[0]
    backend.set_fr_calibration({"bin_width": 0.05, "cells": {fr_key: {"a": 1.0, "b": 0.2}}})
    backend.set_cell_calibration({"bin_width": 0.05, "cells": {cy_key: {"a": 1.0, "b": -7.0}}})
    cal = backend.predict([pat], case, 0.0)
    assert float(cal.mean[0, 0]) == pytest.approx(float(raw.mean[0, 0]) + 0.2, abs=1e-4)
    assert float(cal.mean[0, 3]) == pytest.approx(float(raw.mean[0, 3]) - 7.0, abs=1e-4)
    for col in (1, 2, 4, 5, 6):
        assert np.array_equal(cal.mean[:, col], raw.mean[:, col], equal_nan=True)
    backend.set_fr_calibration(None)
    backend.set_cell_calibration(None)


def test_fr_calibration_gate_neutral_within_cell(backend, sample_batch) -> None:
    # INVARIANT (ii): per-cell affine (a>0) is strictly increasing, so the honest
    # gate's WITHIN-CELL ranking (Spearman) is identical before/after calibration.
    pats, cases = sample_batch
    backend.apply_fr_calibration = False
    raw = np.asarray(backend.predict(pats, cases, 0.0).mean[:, 0], dtype=float)
    keys = backend._fr_cell_keys(pats, cases)

    # each distinct cell gets its OWN positive-slope affine (a>0, varied b).
    distinct = sorted(set(keys))
    cells = {k: {"a": 0.5 + 0.4 * (i % 3), "b": 0.1 * (i - 1)}
             for i, k in enumerate(distinct)}
    backend.set_fr_calibration({"bin_width": 0.05, "cells": cells}, enabled=True)
    cal = np.asarray(backend.predict(pats, cases, 0.0).mean[:, 0], dtype=float)

    # at least one cell must have >=2 members for the ranking check to bite.
    from collections import Counter
    counts = Counter(keys)
    assert max(counts.values()) >= 2
    checked_multi = False
    for cell in distinct:
        idx = [i for i, k in enumerate(keys) if k == cell]
        if len(idx) >= 2:
            checked_multi = True
        r, c = raw[idx], cal[idx]
        # within-cell ranking unchanged (strictly increasing map)
        assert np.array_equal(np.argsort(r, kind="stable"),
                              np.argsort(c, kind="stable"))
        # spearman == 1.0 within the cell (monotone), computed rank-free
        if len(idx) >= 2:
            rr = np.argsort(np.argsort(r))
            cc = np.argsort(np.argsort(c))
            assert np.array_equal(rr, cc)
    assert checked_multi
    backend.set_fr_calibration(None)


# --------------------------------------------------------------------------- #
# Stage 2 — campaign running bias corrector
# --------------------------------------------------------------------------- #
def test_corrector_shrinkage_math() -> None:
    bc = CampaignBiasCorrector(prior_weight=4.0)
    key = bc.key(109, 5.02)
    assert bc.bias(key) == 0.0 and not bc.active               # inert until observed
    bc.observe(key, 660.0, 650.0)                              # delta +10, n=1
    assert bc.bias(key) == pytest.approx(1 / 5 * 10.0)         # shrink 1/(1+4)
    assert bc.active
    for d in (12.0, 8.0, 11.0, 9.0):                           # -> n=5, median 10
        bc.observe(key, 650.0 + d, 650.0)
    assert bc.bias(key) == pytest.approx(5 / 9 * 10.0)
    assert bc.correct(key, 660.0) == pytest.approx(660.0 - 5 / 9 * 10.0)
    assert bc.n_obs(key) == 5


def test_corrector_converges_to_median_with_growth() -> None:
    bc = CampaignBiasCorrector(prior_weight=4.0)
    key = bc.key(117, 5.5)
    for _ in range(200):
        bc.observe(key, 615.0, 600.0)                          # constant +15 delta
    # shrink -> 200/204 ~= 0.98, so bias approaches the raw median (15)
    assert bc.bias(key) == pytest.approx(15.0, abs=0.5)


def test_corrector_skips_fitted_cells() -> None:
    bc = CampaignBiasCorrector(fitted_cells={"feed=117|ebin=5.0"})
    fitted = "feed=117|ebin=5.0"
    assert bc.observe(fitted, 700.0, 620.0) is False           # never observes
    assert bc.bias(fitted) == 0.0                              # never biases
    assert bc.correct(fitted, 700.0) == 700.0
    assert not bc.active


def test_corrector_nan_guard() -> None:
    bc = CampaignBiasCorrector()
    key = bc.key(109, 5.0)
    assert bc.observe(key, float("nan"), 600.0) is False
    assert bc.observe(key, 610.0, float("nan")) is False
    assert bc.observe(key, None, 600.0) is False
    assert bc.n_obs(key) == 0 and bc.bias(key) == 0.0


def test_corrector_resume_roundtrip(tmp_path) -> None:
    bc = CampaignBiasCorrector(prior_weight=3.0, fitted_cells={"feed=121|ebin=5.4"})
    k1, k2 = bc.key(109, 5.5), bc.key(117, 5.05)
    for d in (10.0, 14.0, 9.0):
        bc.observe(k1, 600.0 + d, 600.0)
    bc.observe(k2, 605.0, 600.0)
    path = tmp_path / "cyclen_bias.json"
    bc.save(path)

    restored = CampaignBiasCorrector.load(path)
    assert restored.prior_weight == 3.0
    assert restored.fitted_cells == {"feed=121|ebin=5.4"}
    assert restored.bias(k1) == pytest.approx(bc.bias(k1))
    assert restored.bias(k2) == pytest.approx(bc.bias(k2))
    assert restored.n_obs(k1) == 3
    # dict round-trip is stable through json
    d = json.loads(json.dumps(bc.to_dict()))
    again = CampaignBiasCorrector.from_dict(d)
    assert again.bias(k1) == pytest.approx(bc.bias(k1))


# --------------------------------------------------------------------------- #
# Stage 2 — effect threaded through the user_criteria scoring
# --------------------------------------------------------------------------- #
from lpopt.search import acquisition as acq                         # noqa: E402
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction     # noqa: E402


def _spec(target=625.0, tol=5.0):
    return acq.CriteriaSpec(
        target_cyclen=target, target_discharge_burnup=None,
        cyclen_tolerance=tol, discharge_tolerance=0.0, risk_z=0.0,
    )


def _pred_one(cyclen: float) -> SurrogatePrediction:
    mean = np.array([[1.5, 1400.0, 2.1, cyclen, 0.1, np.nan, 45.0]])
    sig = np.array([[0.02, 15.0, 0.05, 0.0, 0.02, np.nan, 1.0]])
    return SurrogatePrediction(mean, sig.copy(), sig.copy())


def test_score_user_criteria_cyclen_bias_moves_band() -> None:
    spec = _spec(target=625.0, tol=5.0)
    # raw cyclen 640 is 15 above target -> out of the +/-5 band.
    pred = _pred_one(640.0)
    base = acq.score_user_criteria(pred, spec)
    assert base.cyclen_distance[0] > 0.0                       # out of band, penalized
    # subtract a +15 EFPD over-prediction estimate -> corrected 625 is on target.
    corrected = acq.score_user_criteria(pred, spec, cyclen_bias=15.0)
    assert corrected.cyclen_distance[0] == pytest.approx(0.0)  # now in band
    # None default is byte-identical to the pre-corrector call.
    same = acq.score_user_criteria(pred, spec, cyclen_bias=None)
    assert same.cyclen_distance[0] == base.cyclen_distance[0]


def test_score_user_criteria_bias_per_candidate_array() -> None:
    spec = _spec(target=625.0, tol=3.0)
    mean = np.array([
        [1.5, 1400.0, 2.1, 636.0, 0.1, np.nan, 45.0],
        [1.5, 1400.0, 2.1, 636.0, 0.1, np.nan, 45.0],
    ])
    sig = np.tile([0.02, 15.0, 0.05, 0.0, 0.02, np.nan, 1.0], (2, 1))
    pred = SurrogatePrediction(mean, sig.copy(), sig.copy())
    # correct only the first candidate's +11 bias; the second stays raw (out of band)
    uc = acq.score_user_criteria(pred, spec, cyclen_bias=np.array([11.0, 0.0]))
    assert uc.cyclen_distance[0] == pytest.approx(0.0)         # 636-11=625 in band
    assert uc.cyclen_distance[1] > 0.0                         # 636 still out of band


# --------------------------------------------------------------------------- #
# F_r boundary conservatization: low-F_r weighting + conformal offset
# (parity_round1c_20260722 backlog [1](b))
# --------------------------------------------------------------------------- #
def test_fr_low_weight_shifts_intercept_toward_boundary_rows():
    from lpopt.model.cell_calibrate import _weighted_median
    # bulk rows (f_r ~2.5) are calibrated fine; boundary rows (f_r ~1.6) are
    # UNDER-predicted by 0.05.  Plain median (bulk dominates) barely corrects;
    # up-weighting the boundary rows moves b to fix them.
    pred = [1.60] * 8 + [2.50] * 16
    actual = [1.65] * 8 + [2.50] * 16          # only low rows biased (+0.05)
    plain = fit_affine_cell(pred, actual, min_rows=5)
    weighted = fit_affine_cell(pred, actual, min_rows=5,
                               low_weight_thresh=1.7, low_weight=5.0)
    assert weighted["b"] > plain["b"]          # shift up toward the +0.05 boundary
    assert weighted["b"] == pytest.approx(0.05, abs=1e-6)
    # weighted-median helper sanity.
    assert _weighted_median(np.array([1., 2., 3.]), np.array([1., 1., 10.])) == 3.0


def test_fr_conformal_offset_adds_conservatism_and_is_recorded():
    pred = [1.60] * 10
    actual = [1.60] * 10                        # no bias
    base = fit_affine_cell(pred, actual, min_rows=5, low_weight_thresh=1.7, low_weight=3.0)
    off = fit_affine_cell(pred, actual, min_rows=5, low_weight_thresh=1.7,
                          low_weight=3.0, conformal_offset=0.084)
    assert off["b"] - base["b"] == pytest.approx(0.084)
    assert off["conformal_offset"] == pytest.approx(0.084)
    # a positive offset makes the calibrated F_r prediction HIGHER (conservative).
    assert 1.0 * pred[0] + off["b"] > 1.0 * pred[0] + base["b"]


def test_fr_defaults_are_noop_when_weighting_disabled():
    # low_weight_thresh=None (or low_weight=1) + zero offset == plain fit.
    pred = list(np.linspace(1.5, 2.6, 20))
    actual = [p - 0.03 for p in pred]
    plain = fit_affine_cell(pred, actual, min_rows=5)
    off = fit_affine_cell(pred, actual, min_rows=5, low_weight_thresh=None,
                          low_weight=1.0, conformal_offset=0.0)
    assert off["b"] == pytest.approx(plain["b"])


# --------------------------------------------------------------------------- #
# CBC_max calibration (2026-07-29 debug-panel): pure fit + row mask
# --------------------------------------------------------------------------- #
from lpopt.model.cell_calibrate import (            # noqa: E402
    CBC_CALIB_NAME, CBC_CALIB_SCHEMA, CBC_COL, _AFFINE_MARGIN_CBC,
    load_cbc_calibration,
)


def test_cbc_fit_recovers_uniform_overbias() -> None:
    # Measured on the champion's curriculum holdout: cbc_max is OVER-predicted by
    # a near-constant per-cell shift (global +27 ppm, per-group up to +113), which
    # the intercept-only correction removes.  b = -median(pred-actual) = -27.
    rng = np.random.default_rng(20)
    actual = rng.normal(1450.0, 60.0, 240)                 # ppm-scale target
    pred = actual + 27.0 + rng.normal(0.0, 8.0, 240)
    fit = fit_affine_cell(pred, actual, affine_margin=_AFFINE_MARGIN_CBC)
    assert fit is not None
    assert fit["estimator"] == "intercept"                 # uniform shift wins
    assert fit["a"] == 1.0                                 # a>0 -> monotone
    assert fit["b"] == pytest.approx(-27.0, abs=3.0)
    assert fit["median_bias"] == pytest.approx(27.0, abs=3.0)
    assert fit["mae_after"] < fit["mae_before"]
    cal = apply_affine_calibration(pred, ["k"] * pred.size, {"k": fit})
    assert abs(np.median(cal - actual)) < 3.0              # residual bias removed


def test_cbc_fit_recovers_affine_slope_at_cbc_margin() -> None:
    # A genuine slope distortion at ppm scale is still detectable with the 2 ppm
    # OOF margin (the F_r margin 0.006 would let pure noise win every cell).
    rng = np.random.default_rng(21)
    actual = rng.normal(1600.0, 200.0, 400)
    pred = 1.25 * actual - 380.0 + rng.normal(0.0, 10.0, 400)
    fit = fit_affine_cell(pred, actual, affine_margin=_AFFINE_MARGIN_CBC)
    assert fit is not None
    assert fit["estimator"] == "affine"
    assert 0.2 <= fit["a"] <= 3.0
    assert float(np.median(np.abs(fit["a"] * pred + fit["b"] - actual))) < 15.0


def test_cbc_fit_row_mask_censors_boc_only() -> None:
    # A BOC-only boron reading is not the EDIT2 MAXIMUM the head predicts, so it
    # must never enter the cbc fit — the same censor training applies.
    records = pd.DataFrame({
        "record_id": ["t1", "t2", "t3", "t4"],
        "converged": [True, True, True, True],
        "cbc_max": [1400.0, 1450.0, 1380.0, 1500.0],
        "cyclen": [620.0, 630.0, 625.0, 640.0],
        "cbc_kind": ["max", "boc_only", "max", "boc_only"],
        "library_id": ["ga80"] * 4,
    })
    train = {"t1", "t2", "t3", "t4"}
    cbc = set(records.loc[fit_row_mask(records, train, library_id="ga80",
                                       target_col="cbc_max"), "record_id"])
    assert cbc == {"t1", "t3"}
    # …and the censor is cbc-only: the cyclen fit still sees every row.
    cyc = set(records.loc[fit_row_mask(records, train, library_id="ga80",
                                       target_col="cyclen"), "record_id"])
    assert cyc == {"t1", "t2", "t3", "t4"}


def test_cbc_surrogate_column_is_one() -> None:
    # CBC_max is target index 2 in the DATASET order but column 1 in the SURROGATE
    # layout; the calibration operates on predict().mean, hence 1.  Pinned because
    # an off-by-one here would silently calibrate F_q instead.
    from lpopt.model.model_api import _CBC_SURROGATE_COL, _TARGET_TO_SURROGATE_COL
    assert CBC_COL == 1 == _CBC_SURROGATE_COL == _TARGET_TO_SURROGATE_COL["cbc_max"]


# --------------------------------------------------------------------------- #
# CBC_max calibration: serve-side application on a real backend
# --------------------------------------------------------------------------- #
def test_cbc_absent_artifact_is_byte_identical(backend, sample_case) -> None:
    # BACKWARD COMPATIBILITY: a checkpoint whose member meta / model dir carries
    # NO cbc calibration must load and predict exactly as it did before the hook
    # existed — the apply flag alone can change nothing.
    pat, case = sample_case
    backend.set_cbc_calibration(None)                         # ensure absent
    backend.apply_cbc_calibration = False
    off = backend.predict([pat], case, 0.0)
    backend.apply_cbc_calibration = True                      # flag on, no cells
    on = backend.predict([pat], case, 0.0)
    assert np.array_equal(off.mean, on.mean, equal_nan=True)
    assert np.array_equal(off.epistemic_std, on.epistemic_std, equal_nan=True)
    assert np.array_equal(off.calibrated_std, on.calibrated_std, equal_nan=True)
    assert backend.fitted_cbc_cells() == set()
    assert backend.cbc_calibration is None


def test_backend_applies_cbc_affine_on_fitted_cell(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_cbc_calibration = False
    raw = backend.predict([pat], case, 0.0)
    raw_cbc = float(raw.mean[0, CBC_COL])
    key = backend._cbc_cell_keys([pat], [case])[0]

    a, b = 0.9, -27.0
    backend.set_cbc_calibration(
        {"bin_width": 0.05, "cells": {key: {"a": a, "b": b, "n": 99}}}, enabled=True
    )
    cal = backend.predict([pat], case, 0.0)
    assert float(cal.mean[0, CBC_COL]) == pytest.approx(a * raw_cbc + b, abs=1e-3)
    # … and NOTHING else moves: other mean columns + sigma are byte-identical.
    for col in (0, 2, 3, 4, 5, 6):
        assert np.array_equal(cal.mean[:, col], raw.mean[:, col], equal_nan=True)
    assert np.array_equal(cal.epistemic_std, raw.epistemic_std, equal_nan=True)
    assert np.array_equal(cal.calibrated_std, raw.calibrated_std, equal_nan=True)
    backend.set_cbc_calibration(None)


def test_backend_cbc_identity_on_unfitted_cell(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_cbc_calibration = False
    raw_cbc = float(backend.predict([pat], case, 0.0).mean[0, CBC_COL])
    backend.set_cbc_calibration(
        {"bin_width": 0.05,
         "cells": {"feed=999|ebin=9.9": {"a": 0.0, "b": 0.0, "n": 50}}},
        enabled=True,
    )
    cal_cbc = float(backend.predict([pat], case, 0.0).mean[0, CBC_COL])
    assert cal_cbc == pytest.approx(raw_cbc, abs=1e-3)
    backend.set_cbc_calibration(None)


def test_backend_cbc_disable_flag(backend, sample_case) -> None:
    pat, case = sample_case
    backend.apply_cbc_calibration = False
    raw_cbc = float(backend.predict([pat], case, 0.0).mean[0, CBC_COL])
    key = backend._cbc_cell_keys([pat], [case])[0]
    backend.set_cbc_calibration(
        {"bin_width": 0.05, "cells": {key: {"a": 1.0, "b": -100.0}}}, enabled=False
    )
    assert float(backend.predict([pat], case, 0.0).mean[0, CBC_COL]) == pytest.approx(
        raw_cbc, abs=1e-3)
    assert backend.fitted_cbc_cells() == set()               # disabled -> none
    backend.set_cbc_calibration(None)


def test_three_calibrations_touch_disjoint_columns(backend, sample_case) -> None:
    # F_r moves only col 0, cbc_max only col 1, cyclen only col 3.
    pat, case = sample_case
    backend.apply_fr_calibration = False
    backend.apply_cell_calibration = False
    backend.apply_cbc_calibration = False
    raw = backend.predict([pat], case, 0.0)
    fr_key = backend._fr_cell_keys([pat], [case])[0]
    cy_key = backend._cyclen_cell_keys([pat], [case])[0]
    cbc_key = backend._cbc_cell_keys([pat], [case])[0]
    backend.set_fr_calibration({"bin_width": 0.05, "cells": {fr_key: {"a": 1.0, "b": 0.2}}})
    backend.set_cell_calibration({"bin_width": 0.05, "cells": {cy_key: {"a": 1.0, "b": -7.0}}})
    backend.set_cbc_calibration({"bin_width": 0.05, "cells": {cbc_key: {"a": 1.0, "b": -27.0}}})
    cal = backend.predict([pat], case, 0.0)
    assert float(cal.mean[0, 0]) == pytest.approx(float(raw.mean[0, 0]) + 0.2, abs=1e-4)
    assert float(cal.mean[0, 1]) == pytest.approx(float(raw.mean[0, 1]) - 27.0, abs=1e-3)
    assert float(cal.mean[0, 3]) == pytest.approx(float(raw.mean[0, 3]) - 7.0, abs=1e-4)
    for col in (2, 4, 5, 6):
        assert np.array_equal(cal.mean[:, col], raw.mean[:, col], equal_nan=True)
    backend.set_fr_calibration(None)
    backend.set_cell_calibration(None)
    backend.set_cbc_calibration(None)


def test_cbc_calibration_gate_neutral_within_cell(backend, sample_batch) -> None:
    # Every fitted a>0, so the map is strictly increasing and the honest gate's
    # WITHIN-CELL ranking (Spearman) is identical before/after calibration.
    pats, cases = sample_batch
    backend.apply_cbc_calibration = False
    raw = np.asarray(backend.predict(pats, cases, 0.0).mean[:, CBC_COL], dtype=float)
    keys = backend._cbc_cell_keys(pats, cases)
    distinct = sorted(set(keys))
    cells = {k: {"a": 0.6 + 0.3 * (i % 3), "b": -20.0 * (i - 1)}
             for i, k in enumerate(distinct)}
    backend.set_cbc_calibration({"bin_width": 0.05, "cells": cells}, enabled=True)
    cal = np.asarray(backend.predict(pats, cases, 0.0).mean[:, CBC_COL], dtype=float)
    for cell in distinct:
        idx = [i for i, k in enumerate(keys) if k == cell]
        assert np.array_equal(np.argsort(raw[idx], kind="stable"),
                              np.argsort(cal[idx], kind="stable"))
    backend.set_cbc_calibration(None)


def test_cbc_from_dir_roundtrip_and_backward_compat(tmp_path) -> None:
    """Write -> from_dir -> apply, and the SAME dir without the artifact."""
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path, n=2)
    reader = StoreReader(STORE)
    row = reader.records[reader.records["library_id"].astype(str) == "ga80"].iloc[0]
    pat = unpack_pattern(str(row["pattern"]))
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))

    # (1) backward compat: no cbc_calibration.json in the dir at all.
    bare = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert bare.cbc_calibration is None
    assert bare.fitted_cbc_cells() == set()
    raw_cbc = float(bare.predict([pat], case, 0.0).mean[0, CBC_COL])
    key = bare._cbc_cell_keys([pat], [case])[0]

    # (2) round trip: the artifact the fit writes is the artifact from_dir reads.
    artifact = {
        "schema": CBC_CALIB_SCHEMA, "target": "cbc_max", "cbc_max_col": CBC_COL,
        "bin_width": 0.05,
        "cells": {key: {"a": 1.0, "b": -27.0, "n": 80, "estimator": "intercept"}},
    }
    (ens / CBC_CALIB_NAME).write_text(json.dumps(artifact), encoding="utf-8")
    assert load_cbc_calibration(ens / CBC_CALIB_NAME) == artifact
    loaded = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert key in loaded.fitted_cbc_cells()
    assert float(loaded.predict([pat], case, 0.0).mean[0, CBC_COL]) == pytest.approx(
        raw_cbc - 27.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Map-head OOD sigma floor (2026-07-29 debug-panel)
# --------------------------------------------------------------------------- #
from lpopt.model.model_api import (                 # noqa: E402
    _DEFAULT_FLATNESS_SIGMA_FLOOR,
)


# --------------------------------------------------------------------------- #
# F_q + |AO| per-cell calibration (2026-07-29 all-targets): the 4th and 5th
# instances of the identical hook, on surrogate columns 2 and 4.
# --------------------------------------------------------------------------- #
from lpopt.model.cell_calibrate import (          # noqa: E402
    AO_CALIB_NAME, AO_COL, FLAT_CALIB_NAME, FLAT_CALIB_SCHEMA, FLATNESS_TARGETS,
    FQ_CALIB_NAME, FQ_COL, _AFFINE_MARGIN_AO, _AFFINE_MARGIN_FQ,
    flatness_cells, flatness_global_by_library,
)


def test_fq_ao_surrogate_columns() -> None:
    from lpopt.model.model_api import (
        _AO_SURROGATE_COL, _FQ_SURROGATE_COL, _TARGET_TO_SURROGATE_COL)
    assert FQ_COL == 2 == _FQ_SURROGATE_COL == _TARGET_TO_SURROGATE_COL["f_q"]
    assert AO_COL == 4 == _AO_SURROGATE_COL == _TARGET_TO_SURROGATE_COL["ao_abs"]


def test_fq_fit_recovers_the_measured_underbias() -> None:
    # Curriculum-val: F_q MAE 0.250 with bias -0.178 — 71% of the error is one
    # uniform UNDER-prediction, the non-conservative direction vs the 2.41 limit.
    rng = np.random.default_rng(30)
    actual = rng.normal(2.75, 0.30, 240)
    pred = actual - 0.178 + rng.normal(0.0, 0.05, 240)
    fit = fit_affine_cell(pred, actual, affine_margin=_AFFINE_MARGIN_FQ)
    assert fit["estimator"] == "intercept" and fit["a"] == 1.0
    assert fit["b"] == pytest.approx(0.178, abs=0.02)          # pushes F_q UP
    cal = apply_affine_calibration(pred, ["k"] * pred.size, {"k": fit})
    assert abs(np.median(cal - actual)) < 0.02


def test_ao_fit_recovers_a_tiny_bias_at_its_own_scale() -> None:
    # ao_abs is O(0.05) with MAE 0.0060: a margin borrowed from any other target
    # would be either thousands of times too large or so small noise wins.
    rng = np.random.default_rng(31)
    actual = np.abs(rng.normal(0.05, 0.02, 240))
    pred = actual - 0.0010 + rng.normal(0.0, 0.0005, 240)
    fit = fit_affine_cell(pred, actual, affine_margin=_AFFINE_MARGIN_AO)
    assert fit["estimator"] == "intercept"
    assert fit["b"] == pytest.approx(0.0010, abs=3e-4)


@pytest.mark.parametrize("setter,col,delta", [
    ("set_fq_calibration", FQ_COL, 0.18),
    ("set_ao_calibration", AO_COL, -0.002),
])
def test_backend_applies_fq_ao_on_fitted_cell_only(backend, sample_case,
                                                   setter, col, delta) -> None:
    pat, case = sample_case
    apply_flag = {"set_fq_calibration": "apply_fq_calibration",
                  "set_ao_calibration": "apply_ao_calibration"}[setter]
    setattr(backend, apply_flag, False)
    raw = backend.predict([pat], case, 0.0)
    raw_v = float(raw.mean[0, col])
    key = backend._cbc_cell_keys([pat], [case])[0]      # same recipe, same width
    getattr(backend, setter)(
        {"bin_width": 0.05, "cells": {key: {"a": 1.0, "b": delta, "n": 99}}})
    cal = backend.predict([pat], case, 0.0)
    assert float(cal.mean[0, col]) == pytest.approx(raw_v + delta, abs=1e-5)
    for other in (0, 1, 2, 3, 4, 5, 6):
        if other == col:
            continue
        assert np.array_equal(cal.mean[:, other], raw.mean[:, other], equal_nan=True)
    assert np.array_equal(cal.calibrated_std, raw.calibrated_std, equal_nan=True)
    getattr(backend, setter)(None)


@pytest.mark.parametrize("setter,flag", [
    ("set_fq_calibration", "apply_fq_calibration"),
    ("set_ao_calibration", "apply_ao_calibration"),
])
def test_fq_ao_absent_artifact_is_byte_identical(backend, sample_case,
                                                 setter, flag) -> None:
    pat, case = sample_case
    getattr(backend, setter)(None)
    setattr(backend, flag, False)
    off = backend.predict([pat], case, 0.0)
    setattr(backend, flag, True)
    on = backend.predict([pat], case, 0.0)
    assert np.array_equal(off.mean, on.mean, equal_nan=True)
    assert np.array_equal(off.calibrated_std, on.calibrated_std, equal_nan=True)


def test_all_five_scalar_hooks_touch_disjoint_columns(backend, sample_case) -> None:
    pat, case = sample_case
    for flag in ("apply_cell_calibration", "apply_fr_calibration",
                 "apply_cbc_calibration", "apply_fq_calibration",
                 "apply_ao_calibration"):
        setattr(backend, flag, False)
    raw = backend.predict([pat], case, 0.0)
    key = backend._cbc_cell_keys([pat], [case])[0]
    shifts = {0: 0.2, 1: -27.0, 2: 0.18, 3: -7.0, 4: -0.002}
    backend.set_fr_calibration({"bin_width": .05, "cells": {key: {"a": 1., "b": shifts[0]}}})
    backend.set_cbc_calibration({"bin_width": .05, "cells": {key: {"a": 1., "b": shifts[1]}}})
    backend.set_fq_calibration({"bin_width": .05, "cells": {key: {"a": 1., "b": shifts[2]}}})
    backend.set_cell_calibration({"bin_width": .05, "cells": {key: {"a": 1., "b": shifts[3]}}})
    backend.set_ao_calibration({"bin_width": .05, "cells": {key: {"a": 1., "b": shifts[4]}}})
    cal = backend.predict([pat], case, 0.0)
    for col, d in shifts.items():
        assert float(cal.mean[0, col]) == pytest.approx(
            float(raw.mean[0, col]) + d, abs=1e-4), f"column {col}"
    for col in (5, 6):
        assert np.array_equal(cal.mean[:, col], raw.mean[:, col], equal_nan=True)
    for s in ("set_fr_calibration", "set_cbc_calibration", "set_fq_calibration",
              "set_cell_calibration", "set_ao_calibration"):
        getattr(backend, s)(None)


def test_fq_ao_from_dir_roundtrip(tmp_path) -> None:
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path, n=2)
    reader = StoreReader(STORE)
    row = reader.records[reader.records["library_id"].astype(str) == "ga80"].iloc[0]
    pat = unpack_pattern(str(row["pattern"]))
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    bare = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert bare.fq_calibration is None and bare.ao_calibration is None
    raw = bare.predict([pat], case, 0.0).mean[0]
    key = bare._cbc_cell_keys([pat], [case])[0]
    (ens / FQ_CALIB_NAME).write_text(json.dumps(
        {"bin_width": 0.05, "cells": {key: {"a": 1.0, "b": 0.18}}}), encoding="utf-8")
    (ens / AO_CALIB_NAME).write_text(json.dumps(
        {"bin_width": 0.05, "cells": {key: {"a": 1.0, "b": -0.002}}}), encoding="utf-8")
    loaded = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert key in loaded.fitted_fq_cells() and key in loaded.fitted_ao_cells()
    cal = loaded.predict([pat], case, 0.0).mean[0]
    assert float(cal[FQ_COL]) == pytest.approx(float(raw[FQ_COL]) + 0.18, abs=1e-5)
    assert float(cal[AO_COL]) == pytest.approx(float(raw[AO_COL]) - 0.002, abs=1e-5)


# --------------------------------------------------------------------------- #
# Flatness calibration: INTERCEPT-ONLY on the map-head means
# --------------------------------------------------------------------------- #
def test_flatness_accessors_are_target_keyed() -> None:
    art = {"cells": {"node_peak": {"k": {"a": 1.0, "b": 0.04}}, "map_cov": {}},
           "global_by_library": {"map_cov": {"ga80": {"a": 1.0, "b": 0.02}}}}
    assert flatness_cells(art, "node_peak") == {"k": {"a": 1.0, "b": 0.04}}
    assert flatness_cells(art, "map_cov") == {}
    assert flatness_cells(None, "node_peak") == {}
    assert flatness_global_by_library(art, "map_cov") == {"ga80": {"a": 1.0, "b": 0.02}}
    assert flatness_global_by_library(art, "node_peak") == {}


def test_flatness_absent_artifact_is_byte_identical(backend, sample_batch) -> None:
    pats, cases = sample_batch
    backend.set_flatness_calibration(None)
    backend.apply_flatness_calibration = False
    off = backend.predict_map_flatness(pats, cases)
    backend.apply_flatness_calibration = True
    on = backend.predict_map_flatness(pats, cases)
    for a, b in zip(off, on):
        assert np.array_equal(a, b, equal_nan=True)


def test_flatness_shifts_means_only_never_sigma(backend, sample_batch) -> None:
    pats, cases = sample_batch
    backend.set_flatness_calibration(None)
    pk0, pks0, cv0, cvs0 = backend.predict_map_flatness(pats, cases)
    keys = backend._flatness_cell_keys(pats, cases)
    d_pk, d_cv = 0.0462, 0.0272                    # the measured optimism
    backend.set_flatness_calibration({
        "bin_width": 0.05,
        "cells": {"node_peak": {k: {"a": 1.0, "b": d_pk} for k in set(keys)},
                  "map_cov": {k: {"a": 1.0, "b": d_cv} for k in set(keys)}},
    })
    pk1, pks1, cv1, cvs1 = backend.predict_map_flatness(pats, cases)
    assert pk1 == pytest.approx(pk0 + d_pk)
    assert cv1 == pytest.approx(cv0 + d_cv)
    assert np.array_equal(pks0, pks1) and np.array_equal(cvs0, cvs1)   # sigma fixed
    # predict()'s 7 columns are a different head and must not move at all.
    backend.set_flatness_calibration(None)


def test_flatness_calibration_does_not_touch_predict(backend, sample_case) -> None:
    pat, case = sample_case
    backend.set_flatness_calibration(None)
    before = backend.predict([pat], case, 0.0)
    key = backend._flatness_cell_keys([pat], [case])[0]
    backend.set_flatness_calibration({
        "bin_width": 0.05,
        "cells": {"node_peak": {key: {"a": 1.0, "b": 5.0}},
                  "map_cov": {key: {"a": 1.0, "b": 5.0}}}})
    after = backend.predict([pat], case, 0.0)
    assert np.array_equal(before.mean, after.mean, equal_nan=True)
    backend.set_flatness_calibration(None)


def test_flatness_calibration_preserves_within_cell_ranks(backend, sample_batch) -> None:
    """THE constraint: intercept-only is rank-preserving inside a calibration cell.

    The honest no-regression gate ranks candidates within a cell (Spearman), so a
    correction that could reorder them would make a gate PASS/FAIL an artifact of
    the calibration rather than of model skill.  ``a == 1`` makes the map a pure
    translation, so the order is bit-identical — asserted here on the real backend
    for both axes, per calibration cell, with distinct per-cell shifts.
    """
    pats, cases = sample_batch
    backend.set_flatness_calibration(None)
    pk0, _s, cv0, _s2 = backend.predict_map_flatness(pats, cases)
    keys = np.asarray(backend._flatness_cell_keys(pats, cases), dtype=object)
    distinct = sorted(set(keys.tolist()))
    # a DIFFERENT shift per cell, so the test cannot pass by the shifts agreeing
    backend.set_flatness_calibration({
        "bin_width": 0.05,
        "cells": {
            "node_peak": {k: {"a": 1.0, "b": 0.02 * (i + 1)} for i, k in enumerate(distinct)},
            "map_cov": {k: {"a": 1.0, "b": -0.01 * (i + 1)} for i, k in enumerate(distinct)},
        }})
    pk1, _s3, cv1, _s4 = backend.predict_map_flatness(pats, cases)
    checked = 0
    for cell in distinct:
        idx = np.flatnonzero(keys == cell)
        if idx.size < 2:
            continue
        checked += 1
        for before, after in ((pk0, pk1), (cv0, cv1)):
            b, a = before[idx], after[idx]
            # identical ORDER …
            assert np.array_equal(np.argsort(b, kind="stable"),
                                  np.argsort(a, kind="stable"))
            # … and identical RANKS, which is what Spearman consumes: rho == 1.
            assert np.array_equal(np.argsort(np.argsort(b)), np.argsort(np.argsort(a)))
            from scipy.stats import spearmanr
            assert float(spearmanr(b, a)[0]) == pytest.approx(1.0)
    assert checked, "no calibration cell had >= 2 members; test proved nothing"
    backend.set_flatness_calibration(None)


def test_flatness_gate_surface_sees_the_same_ranks(backend, sample_batch) -> None:
    """The gate reads the map head through curriculum._map_head_flatness."""
    from lpopt.curriculum import _map_head_flatness

    pats, cases = sample_batch
    backend.set_flatness_calibration(None)
    before, r0 = _map_head_flatness(backend, pats, cases)
    keys = np.asarray(backend._flatness_cell_keys(pats, cases), dtype=object)
    distinct = sorted(set(keys.tolist()))
    backend.set_flatness_calibration({
        "bin_width": 0.05,
        "cells": {t: {k: {"a": 1.0, "b": 0.03 * (i + 1)}
                      for i, k in enumerate(distinct)} for t in FLATNESS_TARGETS}})
    after, r1 = _map_head_flatness(backend, pats, cases)
    assert r0 is None and r1 is None
    for t in FLATNESS_TARGETS:
        for cell in distinct:
            idx = np.flatnonzero(keys == cell)
            if idx.size < 2:
                continue
            assert np.array_equal(np.argsort(np.argsort(before[t][idx])),
                                  np.argsort(np.argsort(after[t][idx])))
    backend.set_flatness_calibration(None)


def test_flatness_global_fallback_and_unknown_library(backend, paramA_case) -> None:
    pat, case, _row = paramA_case
    backend.set_flatness_calibration(None)
    pk0 = float(backend.predict_map_flatness([pat], case)[0][0])
    backend.set_flatness_calibration({
        "bin_width": 0.05, "cells": {"node_peak": {}, "map_cov": {}},
        "global_by_library": {"node_peak": {"paramA": {"a": 1.0, "b": 0.05}}}})
    assert float(backend.predict_map_flatness([pat], case)[0][0]) == \
        pytest.approx(pk0 + 0.05, abs=1e-5)
    # unknown library -> identity (never extrapolate a bias across provenance)
    backend.set_flatness_calibration({
        "bin_width": 0.05, "cells": {},
        "global_by_library": {"node_peak": {"nope": {"a": 1.0, "b": 0.05}}}})
    assert float(backend.predict_map_flatness([pat], case)[0][0]) == \
        pytest.approx(pk0, abs=1e-5)
    backend.set_flatness_calibration(None)


def test_flatness_fit_is_structurally_intercept_only(tmp_path, monkeypatch) -> None:
    """The fit must never emit a slope — that is what the rank proof rests on."""
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    from lpopt.model import cell_calibrate as cc
    from lpopt.model.splits import SplitManifest

    allrows = StoreReader(STORE).records
    ok = (allrows["converged"].astype(bool) & allrows["node_peak"].notna()
          & allrows["map_cov"].notna())
    parts = [allrows[ok & (allrows["library_id"].astype(str) == lib)].head(40)
             for lib in ("ga80", "paramA")]
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) < 20:
        pytest.skip("not enough flatness-labelled rows")
    train_ids = [str(r) for r in frame["record_id"]]

    class _Reader:
        def __init__(self, *a, **k):
            self.records = frame

    monkeypatch.setattr("lpopt.data.store.StoreReader", _Reader)
    monkeypatch.setattr(SplitManifest, "from_json", classmethod(
        lambda cls, p: SplitManifest(name="T", kind="filter", seed=0,
                                     train_ids=train_ids, val_ids=[])))
    ens = _make_v4_ensemble(tmp_path, n=1)
    art = cc.fit_flatness_calibration(ens, STORE, tmp_path, split="T",
                                      min_rows=5, device="cpu", write=False)
    assert art["schema"] == FLAT_CALIB_SCHEMA
    assert art["estimator"] == "intercept"
    assert set(art["cells"]) == set(FLATNESS_TARGETS)
    fitted = 0
    for tgt in FLATNESS_TARGETS:
        for params in art["cells"][tgt].values():
            fitted += 1
            assert params["a"] == 1.0                 # NEVER a slope
            assert params["estimator"] == "intercept"
        for params in art["global_by_library"][tgt].values():
            assert params["a"] == 1.0
    assert fitted, "no flatness cell fitted"
    # paramA is admitted here too (same serve-parity admission as the scalars)
    assert set(art["fit_libraries"]) <= {"ga80", "paramA"}


def test_sigma_floor_defaults_when_manifest_and_meta_are_absent(backend) -> None:
    # Neither backend.json nor member meta carries a floor -> the documented
    # defaults (node_peak 0.06 / map_cov 0.02, ~half the observed OOD error).
    assert backend.flatness_sigma_floor == _DEFAULT_FLATNESS_SIGMA_FLOOR
    assert backend.flatness_sigma_floor["node_peak"] == pytest.approx(0.06)
    assert backend.flatness_sigma_floor["map_cov"] == pytest.approx(0.02)


def test_sigma_floor_is_applied_to_the_flatness_spread(backend, sample_batch) -> None:
    pats, cases = sample_batch
    backend.apply_flatness_sigma_floor = False
    _pk_m, pk_s_raw, _cv_m, cv_s_raw = backend.predict_map_flatness(pats, cases)
    backend.apply_flatness_sigma_floor = True
    pk_m, pk_s, cv_m, cv_s = backend.predict_map_flatness(pats, cases)

    floor = backend.flatness_sigma_floor
    assert np.all(pk_s >= floor["node_peak"] - 1e-12)
    assert np.all(cv_s >= floor["map_cov"] - 1e-12)
    assert pk_s == pytest.approx(np.maximum(pk_s_raw, floor["node_peak"]))
    assert cv_s == pytest.approx(np.maximum(cv_s_raw, floor["map_cov"]))
    # The floor must actually BITE somewhere, otherwise this test proves nothing.
    # A 2-member random-weight ensemble happens to disagree a lot, so raise the
    # floor above that spread and check every element is clamped — this is the
    # OOD regime the floor exists for (members that agree, wrongly).
    big_pk = float(np.max(pk_s_raw)) * 2.0 + 1.0
    big_cv = float(np.max(cv_s_raw)) * 2.0 + 1.0
    backend.set_flatness_sigma_floor({"node_peak": big_pk, "map_cov": big_cv})
    _m, pk_hi, _m2, cv_hi = backend.predict_map_flatness(pats, cases)
    assert pk_hi == pytest.approx(np.full_like(pk_s_raw, big_pk))
    assert cv_hi == pytest.approx(np.full_like(cv_s_raw, big_cv))
    backend.set_flatness_sigma_floor(None)
    # the MEANS are untouched: this is an uncertainty fix, not a prediction change.
    assert pk_m == pytest.approx(backend.predict_map_peak(pats, cases)[0])


def test_sigma_floor_from_backend_manifest(tmp_path) -> None:
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path, n=2)
    (ens / "backend.json").write_text(json.dumps({
        "backend": "posval_cnn", "library_id": "ga80",
        "flatness_sigma_floor": {"node_peak": 0.11, "map_cov": 0.033},
    }), encoding="utf-8")
    b = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert b.flatness_sigma_floor == {"node_peak": 0.11, "map_cov": 0.033}
    # a partial manifest floor falls through per channel to the default
    (ens / "backend.json").write_text(json.dumps({
        "backend": "posval_cnn", "library_id": "ga80",
        "flatness_sigma_floor": {"node_peak": 0.2},
    }), encoding="utf-8")
    b2 = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert b2.flatness_sigma_floor["node_peak"] == pytest.approx(0.2)
    assert b2.flatness_sigma_floor["map_cov"] == pytest.approx(
        _DEFAULT_FLATNESS_SIGMA_FLOOR["map_cov"])


def test_sigma_floor_prefers_measured_val_fold_residual_sd(tmp_path) -> None:
    # A member that PERSISTS its own validation-fold residual SD wins over the
    # blanket default — that is the measured number, not a guess.  Taken as the
    # MAX across members (a floor must cover the worst member).
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path, n=2)
    metas = sorted(ens.glob("member_*/meta.json"))
    for path, sd in zip(metas, (0.08, 0.13)):
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["flatness_residual_sd"] = {"node_peak": sd, "map_cov": 0.5 * sd}
        path.write_text(json.dumps(meta), encoding="utf-8")
    b = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert b.flatness_sigma_floor["node_peak"] == pytest.approx(0.13)
    assert b.flatness_sigma_floor["map_cov"] == pytest.approx(0.065)
    # an explicit runtime override still wins over everything
    b.set_flatness_sigma_floor({"node_peak": 0.5})
    assert b.flatness_sigma_floor["node_peak"] == pytest.approx(0.5)
    assert b.flatness_sigma_floor["map_cov"] == pytest.approx(0.065)
    # …and clearing it re-resolves back to the meta value (never to a stale one)
    b.set_flatness_sigma_floor(None)
    assert b.flatness_sigma_floor["node_peak"] == pytest.approx(0.13)


# --------------------------------------------------------------------------- #
# paramA coverage: serve-parity admission + effective-library cell keying
# (forensic 2026-07-29 debug-panel: the ga80-only fit + configured-library keying
#  left 1,361 of the 2,676 curriculum-val rows with ZERO correction on all three
#  targets, which is why the cbc artifact moved val MAE only 42.32 -> 40.86 ppm)
# --------------------------------------------------------------------------- #
from lpopt.model.cell_calibrate import (           # noqa: E402
    global_by_library, serve_parity_mask,
)


def test_serve_parity_mask_admits_roundtrip_libraries_only() -> None:
    rows = ["ga80", "paramA", "260624", "5.8_5.1", "legacy_a"]
    # what PosValCnnBackend.serve_library resolves each pattern to (measured on the
    # real fuel table): ga80/paramA/legacy_a round-trip; 260624 and 5.8_5.1 share
    # batch labels with the ga80 roster and collapse onto it.
    serve = ["ga80", "paramA", "ga80", "ga80", "legacy_a"]
    m = serve_parity_mask(rows, serve)
    assert list(m) == [True, True, False, False, True]
    # the old blanket "library_id == ga80" filter kept only the FIRST row — which
    # is exactly the bug: paramA and legacy_a were dropped from every fit.
    assert m.sum() == 3


def test_fit_row_mask_accepts_a_library_sequence() -> None:
    records = pd.DataFrame({
        "record_id": ["a", "b", "c"],
        "converged": [True, True, True],
        "cyclen": [620.0, 630.0, 640.0],
        "library_id": ["ga80", "paramA", "260624"],
    })
    ids = {"a", "b", "c"}
    assert set(records.loc[fit_row_mask(records, ids, library_id="ga80"),
                           "record_id"]) == {"a"}
    assert set(records.loc[fit_row_mask(records, ids, library_id=["ga80", "paramA"]),
                           "record_id"]) == {"a", "b"}
    assert set(records.loc[fit_row_mask(records, ids, library_id=None),
                           "record_id"]) == {"a", "b", "c"}


def test_global_by_library_accessor() -> None:
    assert global_by_library(None) == {}
    assert global_by_library({"cells": {}}) == {}          # absent -> no fallback
    art = {"global_by_library": {"ga80": {"a": 1.0, "b": -9.0}}}
    assert global_by_library(art) == {"ga80": {"a": 1.0, "b": -9.0}}


def test_apply_global_fallback_is_per_library_and_cell_wins() -> None:
    vals = np.array([1500.0, 1500.0, 1500.0, 1500.0, np.nan])
    keys = ["fitted", "missing", "missing", "missing", "missing"]
    libs = ["ga80", "ga80", "paramA", "unseen_lib", "ga80"]
    cells = {"fitted": {"a": 1.0, "b": -100.0}}
    gmap = {"ga80": {"a": 1.0, "b": -9.0}, "paramA": {"a": 1.0, "b": -49.0}}
    out = apply_affine_calibration(vals, keys, cells,
                                   globals_by_lib=gmap, libraries=libs)
    assert out[0] == pytest.approx(1400.0)      # per-cell fit WINS over the global
    assert out[1] == pytest.approx(1491.0)      # ga80 global
    assert out[2] == pytest.approx(1451.0)      # paramA global (its own, not pooled)
    assert out[3] == pytest.approx(1500.0)      # unknown library -> identity
    assert np.isnan(out[4])                     # NaN still passes through


def test_apply_without_a_global_map_is_byte_identical() -> None:
    # BACKWARD COMPAT: an artifact predating the fallback carries no
    # ``global_by_library``, so the apply must behave exactly as it always did.
    vals = np.array([1500.0, 1480.0, 1520.0])
    keys = ["fitted", "missing", "missing"]
    cells = {"fitted": {"a": 1.0, "b": -20.0}}
    base = apply_affine_calibration(vals, keys, cells)
    same = apply_affine_calibration(vals, keys, cells, globals_by_lib={},
                                    libraries=["ga80"] * 3)
    also = apply_affine_calibration(vals, keys, cells, globals_by_lib=None,
                                    libraries=None)
    assert np.array_equal(base, same) and np.array_equal(base, also)
    assert base[1] == 1480.0 and base[2] == 1520.0        # unfitted -> identity


@pytest.fixture(scope="module")
def paramA_case(backend):
    """A real paramA pattern — the regime the ga80-only fit could never reach."""
    reader = StoreReader(STORE)
    sub = reader.records[reader.records["library_id"].astype(str) == "paramA"]
    if not len(sub):
        pytest.skip("no paramA rows in the store")
    row = sub.iloc[0]
    return unpack_pattern(str(row["pattern"])), \
        CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"])), row


def test_serve_library_reroutes_paramA(backend, paramA_case, sample_case) -> None:
    pat, _case, _row = paramA_case
    # the backend is configured for ga80, yet a paramA pattern must resolve to the
    # library that actually carries its fresh types — the same rerouting
    # _record_inputs already does for featurization.
    assert backend.serve_library(pat) == "paramA"
    assert backend.serve_library(sample_case[0]) == "ga80"      # fast path intact


def test_cyclen_e_core_keys_paramA_into_a_real_bin(backend, paramA_case) -> None:
    pat, case, row = paramA_case
    e_core, _ = backend.cyclen_e_core(pat)
    # THE regression: this used to resolve against the CONFIGURED ga80 roster and
    # return None, so every paramA request keyed into "ebin=None" — a cell no fit
    # can ever populate, hence zero correction for half the curriculum-val slice.
    assert e_core is not None and np.isfinite(float(e_core))
    assert float(e_core) == pytest.approx(float(row["e_core"]), abs=1e-6)
    for keys in (backend._cbc_cell_keys([pat], [case]),
                 backend._cyclen_cell_keys([pat], [case]),
                 backend._fr_cell_keys([pat], [case])):
        assert not keys[0].endswith("ebin=None")


def test_ga80_cell_keys_are_unchanged_by_the_rerouting(backend, sample_batch) -> None:
    # A ga80 pattern under a ga80 backend takes the _effective_library fast path,
    # so every existing artifact keeps resolving exactly as it did.
    pats, cases = sample_batch
    for pat, case in zip(pats, cases):
        e_core, _ = backend.cyclen_e_core(pat)
        assert cyclen_cell_key(int(case.feed), e_core, 0.05) == \
            backend._cbc_cell_keys([pat], [case])[0]
        assert backend.serve_library(pat) == "ga80"


def test_backend_global_fallback_corrects_an_unfitted_cell(backend, paramA_case
                                                           ) -> None:
    pat, case, _row = paramA_case
    backend.apply_cbc_calibration = False
    raw = float(backend.predict([pat], case, 0.0).mean[0, CBC_COL])
    # artifact with a DIFFERENT cell fitted + a paramA global term: the row's own
    # cell is unfitted, so the library global must catch it.
    backend.set_cbc_calibration({
        "bin_width": 0.05,
        "cells": {"feed=999|ebin=9.9": {"a": 1.0, "b": -500.0}},
        "global_by_library": {"paramA": {"a": 1.0, "b": -49.0, "n": 400}},
    }, enabled=True)
    assert float(backend.predict([pat], case, 0.0).mean[0, CBC_COL]) == \
        pytest.approx(raw - 49.0, abs=1e-3)
    # a library with no global entry is left alone (never extrapolate a bias
    # across provenance — it varies by 100-400 ppm along exactly that axis).
    backend.set_cbc_calibration({
        "bin_width": 0.05, "cells": {},
        "global_by_library": {"some_other_lib": {"a": 1.0, "b": -49.0}},
    }, enabled=True)
    assert float(backend.predict([pat], case, 0.0).mean[0, CBC_COL]) == \
        pytest.approx(raw, abs=1e-3)
    backend.set_cbc_calibration(None)


def test_per_cell_fit_beats_the_global_on_the_real_backend(backend, sample_case
                                                           ) -> None:
    pat, case = sample_case
    backend.apply_cbc_calibration = False
    raw = float(backend.predict([pat], case, 0.0).mean[0, CBC_COL])
    key = backend._cbc_cell_keys([pat], [case])[0]
    backend.set_cbc_calibration({
        "bin_width": 0.05,
        "cells": {key: {"a": 1.0, "b": -27.0}},
        "global_by_library": {"ga80": {"a": 1.0, "b": -500.0}},
    }, enabled=True)
    assert float(backend.predict([pat], case, 0.0).mean[0, CBC_COL]) == \
        pytest.approx(raw - 27.0, abs=1e-3)          # cell wins, global unused
    backend.set_cbc_calibration(None)


def test_fit_admits_paramA_and_writes_a_per_library_global(tmp_path, monkeypatch
                                                           ) -> None:
    """End-to-end fit on a small REAL slice: the artifact must now cover paramA.

    Runs the actual :func:`_fit_cell_affine_target` (real featurization, real
    forward, real cell keying) over a few dozen store rows, with the store reader
    and split manifest stubbed so it costs a second instead of an hour.
    """
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    from lpopt.model import cell_calibrate as cc
    from lpopt.model.splits import SplitManifest

    allrows = StoreReader(STORE).records
    ok = allrows["converged"].astype(bool) & allrows["cbc_max"].notna()
    parts = [allrows[ok & (allrows["library_id"].astype(str) == lib)].head(n)
             for lib, n in (("ga80", 40), ("paramA", 40), ("260624", 40))]
    frame = pd.concat(parts, ignore_index=True)
    train_ids = [str(r) for r in frame["record_id"]]

    class _Reader:
        def __init__(self, *a, **k):
            self.records = frame

    monkeypatch.setattr("lpopt.data.store.StoreReader", _Reader)
    monkeypatch.setattr(SplitManifest, "from_json", classmethod(
        lambda cls, p: SplitManifest(name="T", kind="filter", seed=0,
                                     train_ids=train_ids, val_ids=[])))

    ens = _make_v4_ensemble(tmp_path, n=1)
    art = cc._fit_cell_affine_target(
        ens, STORE, tmp_path, target_col_name="cbc_max", surrogate_col=CBC_COL,
        col_key_name="cbc_max_col", schema=CBC_CALIB_SCHEMA, target_label="cbc_max",
        out_name=CBC_CALIB_NAME, affine_margin=cc._AFFINE_MARGIN_CBC,
        split="T", min_rows=5, device="cpu", library_id="ga80", write=False)

    # paramA is IN (this is the whole fix) and 260624 is OUT (its patterns resolve
    # to the ga80 roster, so its e_core bin would be someone else's).
    assert set(art["fit_libraries"]) == {"ga80", "paramA"}
    assert art["dropped_serve_parity"].get("260624") == 40
    assert art["n_train_labelled"] == 80
    # per-library global fallback exists for both admitted libraries…
    assert set(art["global_by_library"]) == {"ga80", "paramA"}
    for g in art["global_by_library"].values():
        assert g["a"] == 1.0 and g["estimator"] == "intercept"   # never a slope
    # …and paramA cells land in paramA's own e_core range, never in ebin=None.
    assert art["cells"], "no cell fitted"
    assert not any(k.endswith("ebin=None") for k in art["cells"])
    ebins = {float(k.split("ebin=")[1]) for k in art["cells"]}
    assert max(ebins) > 5.5, f"no paramA-range cell fitted: {sorted(ebins)}"
    # disjoint e_core ranges today -> no cell may blend two libraries.
    assert art["mixed_library_cells"] == {}


def test_sigma_floor_ignores_garbage_values(tmp_path) -> None:
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path, n=1)
    (ens / "backend.json").write_text(json.dumps({
        "backend": "posval_cnn", "library_id": "ga80",
        "flatness_sigma_floor": {"node_peak": "nonsense", "map_cov": -1.0},
    }), encoding="utf-8")
    b = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    # a negative or unparseable floor is not a floor; fall back, never crash.
    assert b.flatness_sigma_floor == _DEFAULT_FLATNESS_SIGMA_FLOOR
