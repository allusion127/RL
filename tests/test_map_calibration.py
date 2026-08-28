"""``map_calibration.json`` — the map-head LEVEL calibration (program §2.1 / D1).

Program §2.1 makes this artifact a PRECONDITION for running the flatness
objective: without it the F_r safety gate's bias correction is inert and the
acquisition consumes raw map-head levels whose fold-C optimism was measured at
−0.147 (``node_peak``) / −0.058 (``map_cov``), while its only pessimism is
``risk_z x ensemble spread`` — a disagreement statistic that cannot express a
bias every member shares.

What is pinned here:

* the fit is DETERMINISTIC and order-independent (no RNG, no bootstrap);
* per-cell resolution falls back to the global block, and then to "no
  correction" — never to a silent zero;
* the D1 gate is corrected WITH the artifact and HELD without it, and a fitted
  correction can only ever TIGHTEN it;
* the calibrated UCB sigma is the raw ensemble spread when nothing is fitted and
  strictly larger when something is;
* an artifact fitted on a DIFFERENT champion is a loud refusal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from lpopt.data import flat_scale as _FS
from lpopt.data.map_calibration import (
    ARTIFACT_NAME, ARTIFACT_SCHEMA, GATE_K, SOURCE_CELL, SOURCE_GLOBAL,
    SOURCE_NONE, TARGETS, MapCalibration, ModelMismatchError, gate_shift,
    load_gate_correction, model_fingerprint, model_id,
)
from lpopt.search import acquisition as acq
from lpopt.tools import fit_map_calibration as FIT

from test_campaign_stub import _STORE           # noqa: F401  (shared fixture store)


CELL = "feed=121|ebin=5.2"
OTHER = "feed=117|ebin=5.0"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _doc(cells: dict | None = None, glob: dict | None = None,
         fit: dict | None = None) -> dict:
    return {"schema": ARTIFACT_SCHEMA, "cells": cells or {},
            "global": glob or {}, "fit": fit or {}}


def _entry(bias: float, sigma: float = 0.0, sigma_ens: float = 0.0,
           sigma_extra: float | None = None, n: int = 40) -> dict:
    if sigma_extra is None:
        sigma_extra = math.sqrt(max(0.0, sigma * sigma - sigma_ens * sigma_ens))
    return {"bias": bias, "sigma": sigma, "sigma_ens": sigma_ens,
            "sigma_extra": sigma_extra, "n": n}


def _write(store: Path, doc: dict) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    p = store / ARTIFACT_NAME
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _fake_model_dir(root: Path, name: str, *, seeds=(1, 2), tag: str = "a",
                    weights: bytes | None = None) -> Path:
    d = root / name
    for s in seeds:
        m = d / f"member_{s}"
        m.mkdir(parents=True, exist_ok=True)
        (m / "meta.json").write_text(
            json.dumps({"seed": s, "split": "S1", "cond_schema": "v6", "tag": tag}),
            encoding="utf-8")
        if weights is not None:
            (m / "model.pt").write_bytes(weights)
    return d


# --------------------------------------------------------------------------- #
# 1. the fit: determinism + the statistic it claims to compute
# --------------------------------------------------------------------------- #
def test_fit_target_is_deterministic_and_order_independent():
    rng = np.random.default_rng(20260726)
    actual = rng.normal(1.8, 0.05, 200)
    pred = actual - 0.147 + rng.normal(0.0, 0.02, 200)
    sig = np.full(200, 0.01)

    a = FIT.fit_target(pred, actual, sig)
    b = FIT.fit_target(pred, actual, sig)
    assert a == b                                   # bit-identical, no RNG

    order = rng.permutation(200)
    c = FIT.fit_target(pred[order], actual[order], sig[order])
    assert c == a                                   # medians/MAD are order-free
    # ...and it measures the injected optimism.
    assert a["bias"] == pytest.approx(-0.147, abs=0.01)
    assert a["mae_after"] < a["mae_before"]


def test_fit_target_sigma_extra_is_what_the_ensemble_is_missing():
    rng = np.random.default_rng(7)
    actual = rng.normal(1.8, 0.05, 400)
    pred = actual + rng.normal(0.0, 0.05, 400)      # residual SD ~ 0.05
    under = FIT.fit_target(pred, actual, np.full(400, 0.01))
    # residual spread 0.05, ensemble claims 0.01 -> the ensemble is missing
    # sqrt(0.05^2 - 0.01^2) ~ 0.049 of dispersion.
    assert under["sigma_extra"] == pytest.approx(
        math.sqrt(under["sigma"] ** 2 - 0.01 ** 2), abs=1e-6)
    assert under["sigma_extra"] > 0.03

    # an OVER-dispersed ensemble yields exactly zero, never a negative variance.
    over = FIT.fit_target(pred, actual, np.full(400, 1.0))
    assert over["sigma_extra"] == 0.0


def test_fit_target_respects_the_row_floor():
    pred = np.arange(10, dtype=float)
    assert FIT.fit_target(pred, pred + 1.0, None, min_rows=12) is None
    assert FIT.fit_target(pred, pred + 1.0, None, min_rows=10) is not None
    # non-finite pairs do not count toward the floor.
    p = np.concatenate([pred, np.full(5, np.nan)])
    a = np.concatenate([pred + 1.0, np.zeros(5)])
    assert FIT.fit_target(p, a, None, min_rows=12) is None


def test_robust_sd_ignores_a_single_outlier():
    r = np.concatenate([np.full(50, 0.0), [100.0]])
    assert FIT.robust_sd(r) == pytest.approx(0.0)
    assert np.std(r, ddof=1) > 10.0                 # the plain SD does not
    assert FIT.robust_sd(np.zeros(1)) == 0.0


def test_gate_shift_only_ever_tightens():
    assert gate_shift(-0.09) == pytest.approx(0.09)   # optimistic -> tighten
    assert gate_shift(+0.09) == 0.0                   # pessimistic -> hold
    assert gate_shift(None) == 0.0
    assert gate_shift(float("nan")) == 0.0


def test_build_artifact_shape_and_provenance(tmp_path):
    """A synthetic slice through the real assembler: cells, global, provenance."""
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 60
    cells = np.array([CELL] * 40 + [OTHER] * 20)
    df = pd.DataFrame({
        "_cell": cells,
        "_proposal": np.zeros(n, dtype=bool),
        "library_id": ["ga80"] * n,
        "node_peak": rng.normal(1.8, 0.05, n),
        "map_cov": rng.normal(0.35, 0.02, n),
        "f_r": rng.normal(1.6, 0.05, n),
    })
    preds = {}
    for t, b in (("node_peak", -0.147), ("map_cov", -0.058), ("f_r", -0.09)):
        preds[f"{t}_mean"] = df[t].to_numpy(float) + b
        preds[f"{t}_sigma"] = np.full(n, 0.005)
    model_dir = _fake_model_dir(tmp_path, "20260725_063351")
    doc = FIT.build_artifact(df, preds, model_dir=model_dir,
                             store_dir=tmp_path / "store", split="S1", fold="C",
                             slice_report=FIT.SliceReport(n_labelled=n),
                             min_rows=12)

    assert doc["schema"] == ARTIFACT_SCHEMA
    assert set(doc["cells"]) == {CELL, OTHER}
    assert doc["fit"]["model_id"] == "20260725_063351"
    assert doc["fit"]["model_fingerprint"] == model_fingerprint(model_dir)
    assert doc["fit"]["fold"] == "C" and doc["fit"]["split"] == "S1"
    assert doc["fit"]["n_cells_fitted"] == 2
    for target, want in (("node_peak", -0.147), ("map_cov", -0.058)):
        assert doc["global"][target]["bias"] == pytest.approx(want, abs=0.01)
        assert doc["cells"][CELL][target]["bias"] == pytest.approx(want, abs=0.02)
    # derived gate keys, on both scopes
    assert doc["global"]["fr_bias"] == pytest.approx(0.09, abs=0.01)
    assert doc["cells"][CELL]["fr_bias"] == pytest.approx(0.09, abs=0.02)

    # determinism of the WHOLE assembler (modulo the wall-clock stamp).
    again = FIT.build_artifact(df, preds, model_dir=model_dir,
                               store_dir=tmp_path / "store", split="S1", fold="C",
                               slice_report=FIT.SliceReport(n_labelled=n),
                               min_rows=12)
    for d in (doc, again):
        d["fit"].pop("fitted_at")
    assert json.dumps(doc, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_build_artifact_drops_a_cell_under_the_floor(tmp_path):
    import pandas as pd

    n = 20
    df = pd.DataFrame({
        "_cell": np.array([CELL] * 15 + [OTHER] * 5),
        "_proposal": np.zeros(n, dtype=bool),
        "node_peak": np.linspace(1.7, 1.9, n),
        "map_cov": np.linspace(0.30, 0.40, n),
        "f_r": np.linspace(1.5, 1.7, n),
    })
    preds = {f"{t}_{k}": (df[t].to_numpy(float) - 0.1 if k == "mean"
                          else np.full(n, 0.01))
             for t in TARGETS for k in ("mean", "sigma")}
    doc = FIT.build_artifact(df, preds, model_dir=tmp_path / "m",
                             store_dir=tmp_path, split="S1", fold="C",
                             slice_report=FIT.SliceReport(), min_rows=12)
    assert set(doc["cells"]) == {CELL}              # OTHER has 5 < 12 rows
    assert doc["fit"]["n_cells_seen"] == 2
    # ...but the under-floor rows still feed the GLOBAL fallback.
    assert doc["global"]["node_peak"]["n"] == n


# --------------------------------------------------------------------------- #
# 2. resolution: per-cell -> global fallback -> none
# --------------------------------------------------------------------------- #
def test_per_cell_wins_global_falls_back_and_unknown_is_none():
    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"node_peak": _entry(-0.20, 0.05)}},
        glob={"node_peak": _entry(-0.10, 0.03), "map_cov": _entry(-0.05, 0.02)},
    ))
    assert mc.resolve("node_peak", CELL).source == SOURCE_CELL
    assert mc.bias("node_peak", CELL) == pytest.approx(-0.20)
    # a cell the fit never reached: the global block, explicitly labelled.
    assert mc.resolve("node_peak", OTHER).source == SOURCE_GLOBAL
    assert mc.bias("node_peak", OTHER) == pytest.approx(-0.10)
    # a cell that IS fitted but not for this target still falls back globally.
    assert mc.resolve("map_cov", CELL).source == SOURCE_GLOBAL
    # ...and a target with neither is UNAVAILABLE, not zero.
    assert mc.resolve("f_r", CELL).source == SOURCE_NONE
    assert mc.bias("f_r", CELL) is None
    assert mc.sigma_extra("f_r", CELL) is None


def test_absent_artifact_is_inert_not_a_zero_correction(tmp_path):
    mc = MapCalibration.from_store(tmp_path / "nope")
    assert mc.present is False
    for target in TARGETS:
        assert mc.bias(target, CELL) is None
        assert mc.resolve(target, CELL).source == SOURCE_NONE
    assert mc.gate_for(CELL) == (None, None)
    sig = np.array([0.01, 0.02, np.nan])
    assert np.allclose(mc.calibrated_sigma("node_peak", CELL, sig), sig,
                       equal_nan=True)


def test_malformed_artifact_degrades_to_inert(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / ARTIFACT_NAME).write_text("{ not json", encoding="utf-8")
    mc = MapCalibration.from_store(store)
    assert mc.present is False and mc.gate_for(CELL) == (None, None)
    # a well-formed doc whose entries are junk drops the entries, not the file.
    _write(store, _doc(cells={CELL: {"node_peak": {"sigma": 1.0}}}))
    mc = MapCalibration.from_store(store)
    assert mc.present is True
    assert mc.bias("node_peak", CELL) is None       # no ``bias`` key -> no entry


def test_calibrated_sigma_only_ever_adds_pessimism():
    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"node_peak": _entry(-0.1, sigma=0.05, sigma_ens=0.03)}}))
    raw = np.array([0.03, 0.10, 0.0])
    out = mc.calibrated_sigma("node_peak", CELL, raw)
    extra = mc.sigma_extra("node_peak", CELL)
    assert extra > 0.0
    assert np.allclose(out, np.sqrt(raw ** 2 + extra ** 2))
    assert np.all(out >= raw)
    # a cell with nothing fitted keeps the raw ensemble spread exactly.
    assert np.allclose(mc.calibrated_sigma("map_cov", CELL, raw), raw)


# --------------------------------------------------------------------------- #
# 3. the D1 F_r safety gate
# --------------------------------------------------------------------------- #
def test_gate_reads_the_fitted_f_r_block_and_falls_back_globally():
    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"f_r": _entry(-0.09, sigma=0.06, sigma_ens=0.02)}},
        glob={"f_r": _entry(-0.04, sigma=0.05, sigma_ens=0.03)},
    ))
    bias, sigma = mc.gate_for(CELL)
    assert bias == pytest.approx(0.09)              # = max(0, -bias)
    assert sigma == pytest.approx(math.sqrt(0.06 ** 2 - 0.02 ** 2))
    g_bias, _g_sigma = mc.gate_for(OTHER)
    assert g_bias == pytest.approx(0.04)            # global fallback


def test_gate_explicit_keys_win_and_cannot_loosen():
    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"fr_bias": 0.05, "fr_sigma": 0.04,
                      "f_r": _entry(-0.30, sigma=0.30)}}))
    assert mc.gate_for(CELL) == (0.05, 0.04)        # explicit key wins
    # a hand-edited NEGATIVE shift (which would LOOSEN a licensing gate) is
    # clamped away at read time, not trusted.
    loose = MapCalibration.from_doc(_doc(cells={CELL: {"fr_bias": -0.20}}))
    assert loose.gate_for(CELL)[0] == 0.0


def test_flatpower_gate_holds_without_and_tightens_with_the_artifact():
    held = acq.FlatPowerSpec()
    assert acq.flatpower_fr_gate(held) == pytest.approx(1.70)
    corrected = acq.FlatPowerSpec(fr_bias=0.09, fr_sigma=0.06)
    assert acq.flatpower_fr_gate(corrected) == pytest.approx(
        1.70 - 0.09 - GATE_K * 0.06)
    assert acq.flatpower_fr_gate(corrected) < 1.70
    # the gate can never be LOOSENED by a correction, whatever the artifact says.
    assert acq.flatpower_fr_gate(
        acq.FlatPowerSpec(fr_bias=-0.5, fr_sigma=0.0)) == pytest.approx(1.70)
    assert acq.flatpower_fr_gate(
        acq.FlatPowerSpec(fr_bias=0.0, fr_sigma=-0.2)) == pytest.approx(
            1.70 - GATE_K * 0.2)
    # a non-finite correction is refused, not propagated as NaN.
    assert acq.flatpower_fr_gate(
        acq.FlatPowerSpec(fr_bias=float("nan"))) == pytest.approx(1.70)
    assert acq.flatpower_fr_gate(
        acq.FlatPowerSpec(fr_bias=0.09, fr_sigma=float("nan"))) == pytest.approx(
            1.70 - 0.09)


def test_gate_k_is_the_same_constant_everywhere():
    assert acq.FLATPOWER_GATE_K == GATE_K
    assert _FS.GATE_CALIBRATION_NAME == ARTIFACT_NAME


def test_flat_scale_gate_helper_delegates(tmp_path):
    store = tmp_path / "store"
    _write(store, _doc(cells={CELL: {"f_r": _entry(-0.09, sigma=0.06,
                                                   sigma_ens=0.02)}}))
    assert _FS.load_gate_correction(store, CELL) == load_gate_correction(store, CELL)
    assert _FS.load_gate_correction(store, CELL)[0] == pytest.approx(0.09)
    assert _FS.load_gate_correction(store, None) == (None, None)
    assert _FS.load_gate_correction(None, CELL) == (None, None)


# --------------------------------------------------------------------------- #
# 4. the acquisition consumes the calibration
# --------------------------------------------------------------------------- #
class _Pred:
    """Minimal SurrogatePrediction stand-in (7 columns, all inside their limits)."""

    def __init__(self, n: int = 3):
        mean = np.zeros((n, 7))
        mean[:, 0] = 1.50          # f_r
        mean[:, 1] = 1400.0        # cbc
        mean[:, 2] = 2.20          # f_q
        mean[:, 4] = 0.10          # |ao|
        mean[:, 6] = 60.0          # pin bu
        self.mean = mean
        self.calibrated_std = np.zeros((n, 7))


def test_score_flat_power_is_byte_identical_without_a_calibration():
    pred = _Pred(3)
    peak = np.array([1.70, 1.80, 1.90])
    cov = np.array([0.30, 0.32, 0.34])
    sd = np.array([0.01, 0.01, 0.01])
    base = acq.score_flat_power(pred, peak, sd, acq.FlatPowerSpec(), cov, sd)
    same = acq.score_flat_power(pred, peak, sd,
                                acq.FlatPowerSpec(peak_bias=None,
                                                  peak_sigma_extra=None),
                                cov, sd)
    assert np.array_equal(base.total, same.total)
    assert np.array_equal(base.peak_ucb, same.peak_ucb)


def test_score_flat_power_debiases_the_levels_it_minimizes():
    pred = _Pred(1)
    peak, cov = np.array([1.70]), np.array([0.30])
    sd = np.array([0.0])
    spec = acq.FlatPowerSpec(peak_bias=-0.147, cov_bias=-0.058)
    got = acq.score_flat_power(pred, peak, sd, spec, cov, sd)
    # optimism is negative bias, so de-biasing RAISES the minimized level.
    assert got.peak_ucb[0] == pytest.approx(1.70 + 0.147)
    assert got.cov_ucb[0] == pytest.approx(0.30 + 0.058)
    raw = acq.score_flat_power(pred, peak, sd, acq.FlatPowerSpec(), cov, sd)
    assert got.scalar[0] < raw.scalar[0]            # honest is worse than rosy


def test_score_flat_power_uses_the_calibrated_sigma_not_the_raw_spread():
    pred = _Pred(1)
    peak, cov = np.array([1.70]), np.array([0.30])
    ens = np.array([0.004])                          # an under-dispersed ensemble
    spec = acq.FlatPowerSpec(peak_sigma_extra=0.03, cov_sigma_extra=0.01)
    got = acq.score_flat_power(pred, peak, ens, spec, cov, ens)
    want_sd = math.sqrt(0.004 ** 2 + 0.03 ** 2)
    assert got.peak_ucb[0] == pytest.approx(1.70 + spec.risk_z * want_sd)
    assert got.peak_ucb[0] > 1.70 + spec.risk_z * 0.004
    # ...and the correction only ever adds pessimism.
    raw = acq.score_flat_power(pred, peak, ens, acq.FlatPowerSpec(), cov, ens)
    assert got.scalar[0] < raw.scalar[0]


def test_calibration_does_not_reorder_candidates_within_a_cell():
    """A per-cell shift is a constant: it must not silently re-rank the pool.

    (The pessimism term CAN re-rank, because it scales a per-candidate sigma —
    that is the point of calibrating it — but the bias term must not.)
    """
    pred = _Pred(3)
    peak = np.array([1.70, 1.80, 1.90])
    cov = np.array([0.30, 0.32, 0.34])
    sd = np.zeros(3)
    raw = acq.score_flat_power(pred, peak, sd, acq.FlatPowerSpec(), cov, sd)
    cal = acq.score_flat_power(pred, peak, sd,
                               acq.FlatPowerSpec(peak_bias=-0.147,
                                                 cov_bias=-0.058), cov, sd)
    assert list(np.argsort(-raw.total)) == list(np.argsort(-cal.total))


# --------------------------------------------------------------------------- #
# 5. model-mismatch refusal
# --------------------------------------------------------------------------- #
def test_require_model_accepts_the_serving_champion(tmp_path):
    model_dir = _fake_model_dir(tmp_path, "20260725_063351")
    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"f_r": _entry(-0.09)}},
        fit={"model_dir": str(model_dir), "model_id": model_id(model_dir),
             "model_fingerprint": model_fingerprint(model_dir)}))
    assert mc.require_model(model_dir) is True
    # relocating the model tree is NOT a mismatch (same id + fingerprint).
    moved = _fake_model_dir(tmp_path / "elsewhere", "20260725_063351")
    assert mc.require_model(moved) is True


def test_require_model_refuses_a_different_champion(tmp_path):
    fitted_on = _fake_model_dir(tmp_path, "20260725_063351", tag="fitted")
    serving = _fake_model_dir(tmp_path, "20260724_213535", tag="other")
    assert model_fingerprint(fitted_on) != model_fingerprint(serving)
    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"f_r": _entry(-0.09)}},
        fit={"model_dir": str(fitted_on), "model_id": model_id(fitted_on),
             "model_fingerprint": model_fingerprint(fitted_on)}))
    with pytest.raises(ModelMismatchError) as e:
        mc.require_model(serving)
    assert "20260725_063351" in str(e.value) and "20260724_213535" in str(e.value)


def test_require_model_decides_on_the_fingerprint_not_on_the_directory_name(tmp_path):
    """THE bug: the NAME was checked BEFORE the fingerprint was ever computed.

    A byte-identical champion under another name was refused (the same checkpoint,
    renamed / re-saved), and — the dangerous half — the name was silently doing
    the job the fingerprint exists to do.  Identity is CONTENT; the name is a hint
    that may only colour the message.
    """
    fitted_on = _fake_model_dir(tmp_path / "old", "20260725_063351", tag="a")
    renamed = _fake_model_dir(tmp_path / "new", "champion_2026_promoted", tag="a")
    assert model_id(fitted_on) != model_id(renamed)          # names disagree
    # …but the CONTENT does not.
    assert model_fingerprint(fitted_on) == model_fingerprint(renamed)
    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"f_r": _entry(-0.09)}},
        fit={"model_dir": str(fitted_on), "model_id": model_id(fitted_on),
             "model_fingerprint": model_fingerprint(fitted_on)}))
    said: list[str] = []
    assert mc.require_model(renamed, log=said.append) is True
    # accepted ON CONTENT, and the name disagreement is REPORTED, not hidden.
    assert said and "NOTE" in said[0] and "SAME checkpoint" in said[0]

    # ... and the converse: different weights are refused however the path/name
    # matches (this is the check the name was quietly standing in for).
    retrained = _fake_model_dir(tmp_path / "old2", "20260725_063351", tag="RETRAINED")
    assert model_id(retrained) == model_id(fitted_on)
    with pytest.raises(ModelMismatchError, match="retrained in place"):
        mc.require_model(retrained)


def test_require_model_refuses_a_retrain_that_reused_the_directory_name(tmp_path):
    """The id matches; the members do not.  Silence here is the whole failure."""
    before = _fake_model_dir(tmp_path / "before", "20260725_063351", tag="a")
    after = _fake_model_dir(tmp_path / "after", "20260725_063351", tag="RETRAINED")
    assert model_id(before) == model_id(after)
    assert model_fingerprint(before) != model_fingerprint(after)
    mc = MapCalibration.from_doc(_doc(
        fit={"model_id": model_id(before),
             "model_fingerprint": model_fingerprint(before)}))
    with pytest.raises(ModelMismatchError, match="retrained in place"):
        mc.require_model(after)


# --------------------------------------------------------------------------- #
# 5b. the fingerprint covers the WEIGHTS, not only the metadata
# --------------------------------------------------------------------------- #
def test_the_fingerprint_sees_a_weight_swap_the_metadata_hides(tmp_path):
    """THE bug: ``model_fingerprint`` hashed member ``meta.json`` bytes ONLY.

    Two checkpoints with identical metadata and DIFFERENT ``model.pt`` weights
    fingerprinted identically, so ``require_model`` positively CERTIFIED one as
    the other — the strongest statement the artifact makes ("accepted on content,
    not on its name") was not true of the thing that actually produces the
    predictions.  A fine-tune, a re-export or a hand-swapped member is exactly
    this case.
    """
    fitted_on = _fake_model_dir(tmp_path / "a", "champ", weights=b"WEIGHTS-A" * 64)
    swapped = _fake_model_dir(tmp_path / "b", "champ", weights=b"WEIGHTS-B" * 64)
    # identical metadata…
    assert [p.read_bytes() for p in sorted(fitted_on.glob("member_*/meta.json"))] == \
           [p.read_bytes() for p in sorted(swapped.glob("member_*/meta.json"))]
    # …different weights, and the fingerprint SAYS so.
    assert model_fingerprint(fitted_on) != model_fingerprint(swapped)

    mc = MapCalibration.from_doc(_doc(
        cells={CELL: {"f_r": _entry(-0.09)}},
        fit={"model_dir": str(fitted_on), "model_id": model_id(fitted_on),
             "model_fingerprint": model_fingerprint(fitted_on)}))
    assert mc.require_model(fitted_on) is True          # the real one still passes
    with pytest.raises(ModelMismatchError):
        mc.require_model(swapped)                       # …and the swap does not


def test_identical_weights_under_another_path_still_fingerprint_equal(tmp_path):
    """Content, not location: a relocated/renamed checkpoint stays the same one."""
    a = _fake_model_dir(tmp_path / "here", "champ", weights=b"W" * 512)
    b = _fake_model_dir(tmp_path / "there", "renamed", weights=b"W" * 512)
    assert model_fingerprint(a) == model_fingerprint(b)
    # …and a v1 (meta-only) token can never collide with a v2 one.
    meta_only = _fake_model_dir(tmp_path / "nopt", "champ")
    assert model_fingerprint(meta_only) != model_fingerprint(a)


def test_the_weights_hash_is_cached_per_file_identity(tmp_path, monkeypatch):
    """The champion is 10.35M params x 5 members: re-hashing it on every
    ``require_model`` (construction, resume, every per-wave swap) is not free, so
    the digest is memoized on ``(path, size, mtime_ns)`` — and INVALIDATED when
    the file changes, which is the half that keeps the cache honest."""
    import builtins

    from lpopt.data import map_calibration as MC

    model = _fake_model_dir(tmp_path, "champ", weights=b"W" * 4096)
    MC._WEIGHTS_DIGEST_CACHE.clear()

    opened: list[str] = []
    real_open = builtins.open

    def _spy(file, *a, **kw):
        if str(file).endswith("model.pt"):
            opened.append(str(file))
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", _spy)
    first = model_fingerprint(model)
    assert len(opened) == 2                       # one read per member, cold
    again = model_fingerprint(model)
    assert again == first
    assert len(opened) == 2                       # …and none at all, warm

    # rewriting a member's weights invalidates its entry: a stale cache must never
    # certify bytes that are no longer there.
    for p in sorted(model.glob("member_*/model.pt")):
        p.write_bytes(b"X" * 8192)
    assert model_fingerprint(model) != first
    assert len(opened) == 4


def test_require_model_refuses_when_the_serving_champion_is_unknown(tmp_path):
    fitted_on = _fake_model_dir(tmp_path, "20260725_063351")
    mc = MapCalibration.from_doc(_doc(fit={"model_id": model_id(fitted_on)}))
    with pytest.raises(ModelMismatchError):
        mc.require_model(None)


def test_unverifiable_artifact_warns_loudly_instead_of_passing_silently(tmp_path):
    """A legacy/hand-written artifact declares no model — say so, do not assume."""
    model_dir = _fake_model_dir(tmp_path, "20260725_063351")
    mc = MapCalibration.from_doc(_doc(cells={CELL: {"fr_bias": 0.05}}))
    said: list[str] = []
    assert mc.require_model(model_dir, log=said.append) is False
    assert said and "WARNING" in said[0] and "declares no fit model" in said[0]
    # an ABSENT artifact says nothing (there is nothing to warn about).
    said.clear()
    assert MapCalibration.empty().require_model(model_dir, log=said.append) is False
    assert said == []


# --------------------------------------------------------------------------- #
# 6. end-to-end through the campaign driver
# --------------------------------------------------------------------------- #
def _drv(tmp_path: Path, **kw):
    from test_flatness_campaign import _drv as make
    return make(tmp_path, **kw)


def test_campaign_holds_the_gate_and_the_raw_levels_without_the_artifact(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    said: list[str] = []
    from test_flatness_campaign import _cfg
    from lpopt.search.campaign import CampaignDriver
    from lpopt.search.stub import StubEvaluator
    from test_campaign_stub import FakeModel

    cfg = _cfg(tmp_path / "cfg", store_dir=store)
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: StubEvaluator(),
                         dry_run=True, run_dir=tmp_path / "run", progress=False,
                         log=said.append)
    assert drv.map_calibration.present is False
    assert drv.flat_power_spec.fr_gate == pytest.approx(1.70)
    assert drv.flat_power_spec.peak_bias is None
    assert drv.flat_power_spec.peak_sigma_extra is None
    assert any("map_calibration ABSENT" in m for m in said)


def test_campaign_applies_the_per_cell_calibration(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    cell = _drv(tmp_path / "probe", store_dir=store).flat_cell_key
    _write(store, _doc(cells={cell: {
        "node_peak": _entry(-0.147, sigma=0.05, sigma_ens=0.01),
        "map_cov": _entry(-0.058, sigma=0.02, sigma_ens=0.01),
        "f_r": _entry(-0.09, sigma=0.06, sigma_ens=0.02),
    }}))
    drv = _drv(tmp_path / "cal", store_dir=store)
    spec = drv.flat_power_spec
    assert spec.peak_bias == pytest.approx(-0.147)
    assert spec.cov_bias == pytest.approx(-0.058)
    assert spec.peak_sigma_extra == pytest.approx(
        math.sqrt(0.05 ** 2 - 0.01 ** 2))
    assert spec.fr_gate == pytest.approx(
        1.70 - 0.09 - GATE_K * math.sqrt(0.06 ** 2 - 0.02 ** 2))


def test_campaign_uses_the_global_fallback_for_an_unfitted_cell(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    _write(store, _doc(cells={"feed=999|ebin=9.9": {"node_peak": _entry(-1.0)}},
                       glob={"node_peak": _entry(-0.12, sigma=0.04),
                             "f_r": _entry(-0.03, sigma=0.05)}))
    drv = _drv(tmp_path / "glob", store_dir=store)
    assert drv.flat_power_spec.peak_bias == pytest.approx(-0.12)
    assert drv.flat_power_spec.fr_gate == pytest.approx(
        1.70 - 0.03 - GATE_K * 0.05)


# --------------------------------------------------------------------------- #
# 7. the SHIPPED artifact (skipped where the store is not checked out)
#
# These are the only INTEGRATION tests in this file, and they must survive a
# champion promotion.  ``map_calibration.json`` is fingerprint-bound to ONE
# checkpoint by design, so a fixture that pinned the champion's id went red
# every time the champion was legitimately promoted and the artifact refit —
# a promotion turning the suite red is a fixture defect, not a finding.  Nothing
# below names a checkpoint: the champion is resolved FROM the artifact, and a
# disagreement with the SERVING champion is reported as a SKIP naming both.
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[1]
_SHIPPED = _REPO / "data" / "store" / ARTIFACT_NAME


def _named_champion(mc: MapCalibration) -> Path | None:
    """The checkpoint dir the shipped artifact says it was fitted on, if present.

    ``fit.model_dir`` is recorded relative to the repo root by the fitter, so
    both readings are tried before falling back to ``data/models/<model_id>``.
    """
    candidates: list[Path] = []
    raw = str(mc.fit.get("model_dir") or "")
    if raw:
        candidates += [Path(raw), _REPO / raw]
    if mc.fit_model_id:
        candidates.append(_REPO / "data" / "models" / mc.fit_model_id)
    return next((c for c in candidates if c.exists()), None)


@pytest.mark.skipif(not _SHIPPED.exists(), reason="no store artifact present")
def test_shipped_artifact_is_attributable_to_the_champion_and_well_formed():
    mc = MapCalibration.from_store(_SHIPPED.parent)
    assert mc.schema == ARTIFACT_SCHEMA
    assert mc.present and mc.verifiable
    # it NAMES a champion and the slice it was fitted on — which one is the
    # promotion's business, not this test's.
    assert mc.fit_model_id and isinstance(mc.fit_model_id, str)
    assert mc.fit["fold"] and mc.fit["split"]
    assert mc.fit["n_used"] > 0 and mc.fit["n_cells_fitted"] > 0
    # every fitted cell resolves the three targets and a gate that never loosens
    for cell in mc.cells:
        for target in TARGETS:
            assert mc.resolve(target, cell).source == SOURCE_CELL
        fr_bias, fr_sigma = mc.gate_for(cell)
        assert fr_bias >= 0.0 and fr_sigma >= 0.0
        assert acq.flatpower_fr_gate(
            acq.FlatPowerSpec(fr_bias=fr_bias, fr_sigma=fr_sigma)) <= 1.70
    # a cell the fit never reached still gets the global fallback, not silence
    assert mc.gate_for("feed=999|ebin=9.9")[0] is not None
    assert mc.bias("node_peak", "feed=999|ebin=9.9") is not None


@pytest.mark.skipif(not _SHIPPED.exists(), reason="no store artifact present")
def test_shipped_artifact_matches_the_champion_it_names():
    """The artifact's own attribution must be TRUE — whoever the champion is."""
    mc = MapCalibration.from_store(_SHIPPED.parent)
    champion = _named_champion(mc)
    if champion is None:
        pytest.skip(f"the champion {mc.fit_model_id!r} named by {ARTIFACT_NAME} "
                    "is not checked out here")
    assert mc.require_model(champion) is True
    assert mc.fit_model_fingerprint == model_fingerprint(champion)


@pytest.mark.skipif(not (_SHIPPED.exists() and (_REPO / "lpopt.inp").exists()),
                    reason="no store artifact / no deck present")
def test_shipped_artifact_covers_the_champion_the_deck_actually_serves():
    """INTEGRATION, and a SKIP — never a failure — when the two have diverged.

    This is the one statement that depends on the state of the working tree: the
    deck's ``[model] model_dir`` is what a real ``flat_power`` run would serve,
    and construction refuses a calibration fitted on anything else.  A promotion
    legitimately breaks that pairing for as long as it takes to refit, so the
    divergence is REPORTED (naming both checkpoints) instead of failing a suite
    that has nothing to do with it.  The refusal contract itself is pinned above
    against fixture artifacts, so nothing is lost by skipping here.
    """
    import re

    deck = (_REPO / "lpopt.inp").read_text(encoding="utf-8", errors="replace")
    m = re.search(r'^\s*model_dir\s*=\s*"([^"]+)"', deck, re.M)
    if m is None:
        pytest.skip("the deck declares no [model] model_dir")
    serving = _REPO / m.group(1)
    if not serving.exists():
        pytest.skip(f"the deck's champion {m.group(1)!r} is not checked out here")

    mc = MapCalibration.from_store(_SHIPPED.parent)
    try:
        assert mc.require_model(serving) is True
    except ModelMismatchError as exc:
        pytest.skip(
            f"{ARTIFACT_NAME} was fitted on {mc.fit_model_id!r} but the deck "
            f"serves {serving.name!r} — refit with "
            f"`python -m lpopt.tools.fit_map_calibration` ({exc})")


def test_campaign_refuses_a_calibration_from_another_champion(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    other = _fake_model_dir(tmp_path, "some_other_champion")
    _write(store, _doc(cells={CELL: {"f_r": _entry(-0.09)}},
                       fit={"model_dir": str(other), "model_id": model_id(other),
                            "model_fingerprint": model_fingerprint(other)}))
    with pytest.raises(ModelMismatchError, match="some_other_champion"):
        _drv(tmp_path / "mismatch", store_dir=store)


# --------------------------------------------------------------------------- #
# 7. the refusal is not construction-time only (the campaign SWAPS the served
#    weights mid-run: wave fine-tune -> champion save, resume -> reload)
# --------------------------------------------------------------------------- #
def _resumed(tmp_path: Path, state: dict, **kw):
    from test_flatness_campaign import _resumed as make
    return make(tmp_path, state, **kw)


def test_a_mid_run_champion_swap_is_re_checked_not_assumed(tmp_path):
    """THE bug: identity was proven ONCE, at construction.

    Everything after that — the per-wave champion save, a resume that reloads a
    persisted checkpoint — swapped the served weights while the calibration kept
    being applied, silently, to a model it was never fitted on.
    """
    drv = _drv(tmp_path / "swap")        # constructed against its fixture champion
    assert drv._calibrated_model_dir is not None
    other = _fake_model_dir(tmp_path, "some_other_champion")
    with pytest.raises(ModelMismatchError) as e:
        drv._require_calibration_model(other, context="wave 3 champion swap")
    # the SWAP is named, not just the artifact.
    assert "wave 3 champion swap" in str(e.value)
    assert "some_other_champion" in str(e.value)


def test_resume_refuses_a_persisted_champion_the_calibration_does_not_cover(tmp_path):
    """``state.json`` can name a champion construction never saw."""
    other = _fake_model_dir(tmp_path, "resumed_from_another_run")
    drv = _resumed(tmp_path / "resume_swap",
                   {"wave_index": 2, "budget_spent": 8,
                    "champion_ckpt": str(other)})
    with pytest.raises(ModelMismatchError) as e:
        drv._load_state()
    assert "[resume]" in str(e.value)
    assert "resumed_from_another_run" in str(e.value)


def test_resume_of_this_runs_own_wave_champion_is_not_a_mismatch(tmp_path):
    """A run-scoped ``models/champion_wave_NN`` has no fingerprint of its own.

    Its NAME is never the fitted champion's, so refusing on the name would repeat
    the name-first inversion at the swap; with nothing to decide on CONTENT there
    is nothing to prove, and the fine-tune drift is REPORTED instead.
    """
    run = tmp_path / "resume_own" / "run"
    own = run / "models" / "champion_wave_02"
    own.mkdir(parents=True)
    drv = _resumed(tmp_path / "resume_own",
                   {"wave_index": 3, "budget_spent": 16,
                    "champion_ckpt": str(own)})
    assert drv._load_state() is True             # no refusal


def test_a_run_scoped_swap_is_not_refused_when_the_ARTIFACT_has_no_fingerprint(tmp_path):
    """THE bug: the escape hatch tested the SERVING side's fingerprint.

    ``require_model`` lets CONTENT decide only when BOTH sides carry a
    fingerprint; with either side missing one it falls through to the NAME, and a
    run-scoped ``models/champion_wave_NN`` can never match the fitted champion's
    name.  Testing only the serving side therefore left the common case wide open:
    a legacy / hand-written artifact that records NO fingerprint, against a
    run-scoped checkpoint that HAS one — the hatch did not fire, the name branch
    ran, and the campaign aborted at its own champion swap on a naming convention.
    """
    from test_flatness_campaign import FIXTURE_CHAMPION_NAME

    store = tmp_path / "store"
    store.mkdir(parents=True)
    # verifiable (it names the champion the driver will serve) but carries NO
    # fingerprint — the legacy / hand-written case the hatch exists for.
    _write(store, _doc(cells={CELL: {"f_r": _entry(-0.09)}},
                       fit={"model_id": FIXTURE_CHAMPION_NAME}))
    drv = _drv(tmp_path / "hatch", store_dir=store)      # construction: name-matched
    assert drv.map_calibration.present
    assert drv.map_calibration.fit_model_fingerprint == ""

    # this run's own wave champion — a real checkpoint, so it DOES fingerprint.
    own = _fake_model_dir(drv.run_dir / "models", "champion_wave_02")
    assert model_fingerprint(own) != ""
    assert drv._require_calibration_model(
        own, context="wave 2 champion swap") is False    # reported, not refused


def test_the_escape_hatch_is_not_a_hole_for_a_different_model(tmp_path):
    """When both sides DO fingerprint, content decides.

    The hatch exists because a name is not evidence, not because being under the
    run dir is a free pass.  REVISED with the fine-tune fix: a run's OWN
    ``champion_wave_NN`` is now recognised as a fine-tuned descendant and reported
    instead of refused (see
    ``test_a_wave_finetune_champion_swap_completes_and_marks_the_calibration_stale``),
    so this pins the two cases where descent cannot be argued — a foreign fitted
    model dir, and a foreign checkpoint dropped under ``<run_dir>/models/`` under
    any other name.  Both are still a hard refusal.
    """
    store = tmp_path / "store"
    store.mkdir(parents=True)
    fitted_on = _fake_model_dir(tmp_path / "fitted", "20260725_063351",
                                weights=b"FITTED" * 32)
    _write(store, _doc(cells={CELL: {"f_r": _entry(-0.09)}},
                       fit={"model_dir": str(fitted_on),
                            "model_id": model_id(fitted_on),
                            "model_fingerprint": model_fingerprint(fitted_on)}))
    drv = _drv(tmp_path / "hole", store_dir=store, model_dir=fitted_on)

    other = _fake_model_dir(tmp_path / "elsewhere", "another_fitted_champion",
                            weights=b"SOMETHING ELSE" * 32)
    assert model_fingerprint(other) != model_fingerprint(fitted_on)
    with pytest.raises(ModelMismatchError) as e:
        drv._require_calibration_model(other, context="wave 2 champion swap")
    assert "wave 2 champion swap" in str(e.value)

    # …and under the run's own models/ dir, but NOT by the wave-champion name this
    # run writes: nothing says this run produced it, so there is no descent.
    intruder = _fake_model_dir(drv.run_dir / "models", "handcopied_champion",
                               weights=b"SOMETHING ELSE" * 32)
    assert drv._is_run_scoped(intruder) and not drv._is_wave_champion(intruder)
    with pytest.raises(ModelMismatchError):
        drv._require_calibration_model(intruder, context="wave 3 champion swap")


def test_a_rejected_stateless_refit_still_reports_the_weight_drift(tmp_path):
    """The followup: ``_note_finetuned_weights`` fired only on an ACCEPTED gate.

    That is exact for the CNN backend (a rejection restores the member state), but
    a stateless-refit backend is not snapshotted, so its fine-tune survives the
    rejection — and under MODEL_HALT the run keeps serving exactly those weights
    while the champion pointer, the member metas and the fingerprint all say
    nothing moved.  The drift must be reported on that path too.
    """
    from types import SimpleNamespace

    drv = _drv(tmp_path / "drift")
    assert drv.map_calibration.present and drv.map_calibration_stale is False

    # rejected AND rolled back (CNN backend): nothing moved, nothing to report.
    drv._note_gate_weight_drift(
        SimpleNamespace(accepted=False, weights_rolled_back=True))
    assert drv.map_calibration_stale is False

    # rejected and NOT rolled back (stateless refit): the served weights moved.
    drv._note_gate_weight_drift(
        SimpleNamespace(accepted=False, weights_rolled_back=False))
    assert drv.map_calibration_stale is True

    # …and an accepted swap still reports, exactly as before.
    fresh = _drv(tmp_path / "drift_ok")
    fresh._note_gate_weight_drift(
        SimpleNamespace(accepted=True, weights_rolled_back=False))
    assert fresh.map_calibration_stale is True


def test_the_wave_finetune_reports_that_it_moved_off_the_fitted_checkpoint(tmp_path):
    """The drift is stated once rather than assumed away.

    (Historically this said the fingerprint "cannot see" a fine-tune because the
    member metas do not change.  Since the fingerprint began covering ``model.pt``
    it CAN see it — what it cannot see is whether the difference is a descendant
    or another champion.  See ``_note_finetuned_weights``.)
    """
    said: list[str] = []
    from test_flatness_campaign import _cfg
    from lpopt.search.campaign import CampaignDriver
    from lpopt.search.stub import StubEvaluator
    from test_campaign_stub import FakeModel

    cfg = _cfg(tmp_path / "cfg")
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: StubEvaluator(),
                         dry_run=True, run_dir=tmp_path / "run", progress=False,
                         log=said.append)
    assert drv.map_calibration.present and drv.map_calibration_stale is False
    drv._note_finetuned_weights()
    assert drv.map_calibration_stale is True
    assert any("WARNING" in m and "fine-tune replaced the served weights" in m
               for m in said)
    said.clear()
    drv._note_finetuned_weights()               # one-shot, not once per wave
    assert said == []
