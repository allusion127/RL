"""flat_power campaign driver — FLATNESS-NATIVE (program 20260725 §1.2/§1.3/§2.1).

The regression this file exists for: ``_campaign_objective`` used to score
``flat_power`` rows as ``-f_r*1e3 - f_q``, i.e. the node-peak campaign ranked its
own MASTER labels by the F_r scalar the user retired from the objective.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from lpopt.config import (
    AcquisitionConfig, CaseConfig, DataConfig, ExtractConfig, FlowConfig, FuelConfig,
    LpoptConfig, MasterConfig, ModelConfig, ProduceConfig, RemoteConfig, SearchConfig,
    VerifyConfig,
)
from lpopt.data.flat_scale import ARTIFACT_NAME, SCALAR_VERSION, FlatScale
from lpopt.data.map_calibration import (
    ARTIFACT_NAME as MAP_CALIBRATION_NAME,
    ARTIFACT_SCHEMA as MAP_CALIBRATION_SCHEMA,
    model_fingerprint, model_id,
)
from lpopt.search.campaign import CampaignDriver, MapHarvestAbort
from lpopt.search.stub import StubEvaluator

from test_campaign_stub import FakeModel, _STORE


# --------------------------------------------------------------------------- #
# the fixture champion + the calibration fitted on it
#
# THE DESIGN FLAW THIS REPLACES: these fixtures used to serve the LIVE
# ``data/store`` against a HARD-CODED ``data/models/<champion id>``.
# ``map_calibration.json`` is fingerprint-bound to exactly one checkpoint by
# design (``MapCalibration.require_model`` raises ``ModelMismatchError`` on any
# other), so every legitimate champion promotion + refit turned this file red
# without a single line of production code changing — the suite was measuring
# which checkpoint happened to be champion today.
#
# Nothing below reads the live store or any real checkpoint: every driver gets
# its OWN store carrying its OWN artifact, fitted (BY FINGERPRINT, so the
# ``require_model`` contract is exercised for real, not bypassed) on its OWN
# fixture ensemble dir.  The one test that genuinely wants the shipped records
# copies them in and still brings its own calibration.
# --------------------------------------------------------------------------- #

#: Directory NAME of the fixture champion.  Deliberately not any real
#: ``data/models/*`` id — a real id is exactly what went stale.
FIXTURE_CHAMPION_NAME = "fixture_champion"


def fixture_champion(root: Path, name: str = FIXTURE_CHAMPION_NAME) -> Path:
    """A minimal two-member ensemble dir that ``model_fingerprint`` can read.

    Deterministic bytes -> a deterministic fingerprint, so the artifact written
    by :func:`fixture_store` verifies against it on CONTENT.  Nothing ever loads
    it (a ``FakeModel`` is injected); only its name and its member bytes are
    read.  It lives outside ``<run_dir>/models`` so it is never mistaken for a
    run-scoped wave champion.
    """
    d = root / "_fixture_models" / name
    for seed in (1, 2):
        member = d / f"member_{seed}"
        member.mkdir(parents=True, exist_ok=True)
        (member / "meta.json").write_text(
            json.dumps({"seed": seed, "split": "S1", "cond_schema": "v6"}),
            encoding="utf-8")
        (member / "model.pt").write_bytes(b"FIXTURE-CHAMPION-WEIGHTS\n" * 8)
    return d


def _cal_entry(bias: float, sigma: float, sigma_ens: float, n: int) -> dict:
    return {"bias": bias, "sigma": sigma, "sigma_ens": sigma_ens,
            "sigma_extra": 0.0, "n": n}


#: The fixture artifact's numbers — a faithful miniature of a real fit.  The
#: fake deck declares no ``e_core``, so the campaign cell is
#: ``feed=121|ebin=None`` and the GLOBAL block is the scope that actually
#: resolves (exactly as it did against the shipped artifact); the per-cell block
#: is present so the artifact is not a degenerate one.  ``fr_bias`` 0.0171 keeps
#: the D1 gate at 1.683 — corrected, and still above every F_r these fixtures
#: feed it.
_FIXTURE_GLOBAL = {
    "node_peak": _cal_entry(-0.0175, 0.0191, 0.0602, 157),
    "map_cov": _cal_entry(-0.0054, 0.0061, 0.0200, 157),
    "f_r": _cal_entry(-0.0171, 0.0157, 0.2145, 157),
    "fr_bias": 0.0171, "fr_sigma": 0.0,
}
_FIXTURE_CELLS = {
    "feed=121|ebin=5.2": {
        "node_peak": _cal_entry(-0.0290, 0.0107, 0.0600, 31),
        "map_cov": _cal_entry(-0.0123, 0.0040, 0.0200, 31),
        "f_r": _cal_entry(-0.0291, 0.0153, 0.2161, 31),
        "fr_bias": 0.0291, "fr_sigma": 0.0, "n": 31,
    },
}


def fixture_calibration_doc(model_dir: Path) -> dict:
    """A schema-complete ``map_calibration.json`` attributed to ``model_dir``."""
    return {
        "schema": MAP_CALIBRATION_SCHEMA,
        "cells": {k: dict(v) for k, v in _FIXTURE_CELLS.items()},
        "global": dict(_FIXTURE_GLOBAL),
        "fit": {
            "model_dir": str(model_dir),
            "model_id": model_id(model_dir),
            "model_fingerprint": model_fingerprint(model_dir),
            "split": "S1", "fold": "C", "n_used": 157,
            "n_cells_seen": 1, "n_cells_fitted": 1, "min_cell_rows": 12,
        },
    }


def fixture_store(root: Path, model_dir: Path) -> Path:
    """A store dir whose ONLY content is the calibration fitted on ``model_dir``.

    No ``flat_scale.json``: the shipped one's global block is byte-equal to
    ``flat_scale``'s own defaults for this campaign's cell, so omitting it keeps
    the normalizers these tests see unchanged while removing one more live
    dependency.
    """
    store = root / "_fixture_store"
    store.mkdir(parents=True, exist_ok=True)
    (store / MAP_CALIBRATION_NAME).write_text(
        json.dumps(fixture_calibration_doc(model_dir)), encoding="utf-8")
    return store


def _cfg(tmp_path: Path, *, budget: int = 8, harvest: bool = True,
         store_dir: Path | None = None, model_dir: Path | None = None,
         **acq_kw) -> LpoptConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    deck = tmp_path / "lpopt.inp"
    deck.write_text("# fake deck\n", encoding="utf-8")
    acq = AcquisitionConfig(budget=budget, gate_skill_halt=-2.0,
                            objective="flat_power", **acq_kw)
    champion = Path(model_dir) if model_dir is not None else fixture_champion(tmp_path)
    sd = Path(store_dir) if store_dir is not None else fixture_store(tmp_path, champion)
    model = ModelConfig(store_dir=str(sd), model_dir=str(champion))
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(harvest_maps=harvest), data=DataConfig(),
        case=CaseConfig(pair="K1_K2", feed=121), fuel=FuelConfig(),
        extract=ExtractConfig(), produce=ProduceConfig(), search=SearchConfig(),
        acquisition=acq, model=model, source_path=deck,
    )


def _drv(tmp_path: Path, **kw) -> CampaignDriver:
    cfg = _cfg(tmp_path, **kw)
    stub = StubEvaluator()
    return CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                          run_dir=tmp_path / "run", progress=False,
                          log=lambda m: None)


def _bare_store(tmp_path: Path) -> Path:
    """An EMPTY store dir — no ``map_calibration.json``, no ``flat_scale.json``.

    Tests that pin the *uncorrected* branch (the gate HELD at 1.70, the raw
    ensemble UCB) must not read a store that carries a calibration — neither the
    live ``data/store`` (where the real ``map_calibration.json`` turned "1.70"
    into the tightened per-cell gate, so those assertions were measuring the
    artifact rather than the branch) nor :func:`fixture_store`, which now ships
    one on purpose.
    """
    store = tmp_path / "bare_store"
    store.mkdir(parents=True, exist_ok=True)
    return store


def _row(peak=None, cov=None, f_r=1.60, **kw):
    row = {"converged": True, "f_r": f_r, "cbc_max": 1500.0, "f_q": 2.30,
           "ao_abs": 0.20, "cyclen": 625.0, "record_id": "r0",
           "node_peak": peak, "map_cov": cov}
    row.update(kw)
    return row


# --------------------------------------------------------------------------- #
# §1.3 — the verified-row objective is flatness, NOT F_r
# --------------------------------------------------------------------------- #
def test_objective_reads_the_record_flatness_columns(tmp_path):
    drv = _drv(tmp_path / "obj")
    peak_s, cov_s = drv.flat_peak_scale, drv.flat_cov_scale
    got = drv._campaign_objective(_row(peak=1.60, cov=0.32))
    assert got == pytest.approx(-(1.60 / peak_s + 0.5 * 0.32 / cov_s))


def test_objective_no_longer_ranks_by_f_r(tmp_path):
    """THE bug: two rows with identical flatness and different F_r must TIE."""
    drv = _drv(tmp_path / "fr")
    a = drv._campaign_objective(_row(peak=1.60, cov=0.32, f_r=1.50))
    b = drv._campaign_objective(_row(peak=1.60, cov=0.32, f_r=1.68))
    assert a == pytest.approx(b)
    # and F_q, the old tie-break, is equally irrelevant now.
    c = drv._campaign_objective(_row(peak=1.60, cov=0.32, f_q=2.10))
    assert a == pytest.approx(c)
    # the FLATTER row wins even though its F_r is worse — the old scalar
    # (-f_r*1e3 - f_q) ranked these the other way round.
    flat_high_fr = drv._campaign_objective(_row(peak=1.45, cov=0.30, f_r=1.68))
    peaky_low_fr = drv._campaign_objective(_row(peak=1.75, cov=0.40, f_r=1.50))
    assert flat_high_fr > peaky_low_fr


def test_objective_node_peak_is_primary(tmp_path):
    drv = _drv(tmp_path / "prim")
    sp, sc = drv.flat_peak_scale, drv.flat_cov_scale
    better_peak = drv._campaign_objective(_row(peak=1.60 - sp, cov=0.32 + sc))
    better_cov = drv._campaign_objective(_row(peak=1.60, cov=0.32))
    assert better_peak > better_cov


def test_missing_map_is_minus_inf_and_flagged_as_no_label(tmp_path):
    drv = _drv(tmp_path / "miss")
    row = _row(peak=None, cov=None)
    assert drv._campaign_objective(row) == float("-inf")
    assert drv.has_flat_label(row) is False
    assert drv.has_flat_label(_row(peak=1.6)) is True
    # a peak with no cov keeps the primary term (not -inf).
    assert math.isfinite(drv._campaign_objective(_row(peak=1.6, cov=None)))


def test_objective_survives_a_missing_cyclen(tmp_path):
    drv = _drv(tmp_path / "nocy")
    row = _row(peak=1.6, cov=0.3)
    row.pop("cyclen")
    assert math.isfinite(drv._campaign_objective(row))


# --------------------------------------------------------------------------- #
# §1.2 / D4 — per-cell normalization
# --------------------------------------------------------------------------- #
def test_per_cell_scales_are_resolved_for_the_campaign_cell(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    (store / ARTIFACT_NAME).write_text(json.dumps({
        "global": {"peak_scale": 0.40, "cov_scale": 0.08},
        "cells": {"feed=121|ebin=5.2": {"n": 50, "peak_scale": 0.10,
                                        "cov_scale": 0.04}},
    }), encoding="utf-8")
    drv = _drv(tmp_path / "cell", store_dir=store)
    # the campaign's cell key is derived from its own (feed, e_core).
    assert drv.flat_cell_key.startswith("feed=121|ebin=")
    if drv.flat_cell_key == "feed=121|ebin=5.2":
        assert drv.flat_peak_scale == pytest.approx(0.10)
        assert drv.flat_cov_scale == pytest.approx(0.04)
    else:                                    # unfitted cell -> global fallback
        assert drv.flat_peak_scale == pytest.approx(0.40)
        assert drv.flat_cov_scale == pytest.approx(0.08)
    # the spec the acquisition uses carries the SAME numbers as the objective.
    assert drv.flat_power_spec.peak_scale == pytest.approx(drv.flat_peak_scale)
    assert drv.flat_power_spec.cov_scale == pytest.approx(drv.flat_cov_scale)
    assert drv.flat_power_spec.w_cov == pytest.approx(drv.flat_w_cov)


def test_acquisition_and_verified_objective_are_the_same_function(tmp_path):
    """One definition: the pool scalar and the label scalar must agree.

    Measured on an uncalibrated store on purpose.  With a ``map_calibration.json``
    the acquisition scalar de-biases the model's PREDICTED levels while
    ``_campaign_objective`` reads MASTER LABELS, which need no correction — so the
    two agreeing exactly is a statement about the FORMULA, and it has to be made
    where no level correction is in play.
    """
    from lpopt.search import acquisition as acq
    from lpopt.vendor.masterrl.surrogate import SurrogatePrediction

    drv = _drv(tmp_path / "same", store_dir=_bare_store(tmp_path))
    spec = drv.flat_power_spec
    assert spec.peak_bias is None and spec.cov_bias is None
    mean = np.array([[1.50, 1400.0, 2.30, 625.0, 0.20, np.nan, 68.0]])
    std = np.zeros((1, 7))
    fp = acq.score_flat_power(SurrogatePrediction(mean, std.copy(), std.copy()),
                              np.array([1.60]), np.zeros(1), spec,
                              np.array([0.32]), np.zeros(1))
    assert fp.scalar[0] == pytest.approx(
        drv._campaign_objective(_row(peak=1.60, cov=0.32)))


def test_scale_overrides_pin_the_normalizers(tmp_path):
    drv = _drv(tmp_path / "ovr", flatpower_peak_scale=0.5, flatpower_cov_scale=0.1,
               flatpower_w_cov=0.25)
    assert drv.flat_peak_scale == pytest.approx(0.5)
    assert drv.flat_cov_scale == pytest.approx(0.1)
    assert drv._campaign_objective(_row(peak=1.6, cov=0.32)) == pytest.approx(
        -(1.6 / 0.5 + 0.25 * 0.32 / 0.1))


def test_per_cell_can_be_switched_off(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    (store / ARTIFACT_NAME).write_text(json.dumps({
        "global": {"peak_scale": 0.40, "cov_scale": 0.08},
        "cells": {"feed=121|ebin=5.2": {"n": 50, "peak_scale": 0.10,
                                        "cov_scale": 0.04}},
    }), encoding="utf-8")
    drv = _drv(tmp_path / "nc", store_dir=store, flatpower_per_cell_scale=False)
    assert drv.flat_peak_scale == pytest.approx(0.40)
    assert drv.flat_cov_scale == pytest.approx(0.08)


# --------------------------------------------------------------------------- #
# §2.1 / D1 — F_r is a SAFETY GATE
# --------------------------------------------------------------------------- #
def test_fr_safety_gate_holds_at_1p70_and_still_screens(tmp_path):
    # bare store => no map_calibration.json => the D1 correction is unavailable
    # and the gate HOLDS, which is the branch this test exists for.
    drv = _drv(tmp_path / "gate", store_dir=_bare_store(tmp_path))
    assert drv.flat_power_spec.fr_gate == pytest.approx(1.70)
    assert drv._is_feasible(_row(peak=1.6, cov=0.3, f_r=1.68)) is True
    assert drv._is_feasible(_row(peak=1.6, cov=0.3, f_r=1.72)) is False
    # ...but it does not ORDER two passing rows.
    a = drv._campaign_objective(_row(peak=1.6, cov=0.3, f_r=1.51))
    b = drv._campaign_objective(_row(peak=1.6, cov=0.3, f_r=1.69))
    assert a == pytest.approx(b)


def test_fr_gate_uses_the_bias_correction_when_the_artifact_exists(tmp_path):
    """D1: bias-corrected when AVAILABLE — and the availability test is strict."""
    from lpopt.data.flat_scale import GATE_CALIBRATION_NAME, load_gate_correction

    store = tmp_path / "store"
    store.mkdir(parents=True)
    # no artifact at all -> hold.
    assert load_gate_correction(store, "feed=121|ebin=5.2") == (None, None)
    (store / GATE_CALIBRATION_NAME).write_text(json.dumps({
        "cells": {"feed=121|ebin=5.2": {"fr_bias": 0.04, "fr_sigma": 0.02},
                  "feed=117|ebin=5.0": {"node_peak_bias": -0.147}},
    }), encoding="utf-8")
    assert load_gate_correction(store, "feed=121|ebin=5.2") == (0.04, 0.02)
    # a cell whose entry has no fr_bias is NOT a correction -> the gate holds.
    assert load_gate_correction(store, "feed=117|ebin=5.0") == (None, None)
    assert load_gate_correction(store, "feed=999|ebin=9.9") == (None, None)
    assert load_gate_correction(None, "feed=121|ebin=5.2") == (None, None)
    # unparseable artifact -> hold, never crash.
    (store / GATE_CALIBRATION_NAME).write_text("{ nope", encoding="utf-8")
    assert load_gate_correction(store, "feed=121|ebin=5.2") == (None, None)


def test_campaign_wires_the_gate_correction_for_its_own_cell(tmp_path):
    from lpopt.data.flat_scale import GATE_CALIBRATION_NAME

    store = tmp_path / "store"
    store.mkdir(parents=True)
    probe = _drv(tmp_path / "probe0", store_dir=store)
    cell = probe.flat_cell_key
    (store / GATE_CALIBRATION_NAME).write_text(json.dumps({
        "cells": {cell: {"fr_bias": 0.05, "fr_sigma": 0.04}},
    }), encoding="utf-8")
    drv = _drv(tmp_path / "probe1", store_dir=store)
    assert drv.flat_power_spec.fr_gate == pytest.approx(1.70 - 0.05 - 0.5 * 0.04)
    # the TIGHTENED gate is what feasibility actually applies.
    assert drv._is_feasible(_row(peak=1.6, f_r=1.64)) is False
    assert drv._is_feasible(_row(peak=1.6, f_r=1.62)) is True


def test_fr_gate_is_configurable_and_pin_bu_still_screens(tmp_path):
    drv = _drv(tmp_path / "gate2", store_dir=_bare_store(tmp_path),
               flatpower_fr_limit=1.60)
    assert drv.flat_power_spec.fr_gate == pytest.approx(1.60)
    assert drv._is_feasible(_row(peak=1.6, f_r=1.58)) is True
    assert drv._is_feasible(_row(peak=1.6, f_r=1.62)) is False
    base = _row(peak=1.6, f_r=1.55)
    assert drv._is_feasible(dict(base, max_pin_burnup=78.0)) is True
    assert drv._is_feasible(dict(base, max_pin_burnup=88.0)) is False
    assert drv._is_feasible(base) is True             # None-tolerant


# --------------------------------------------------------------------------- #
# §2.2 / D2 — _best_dict reports against the RIGHT limits
# --------------------------------------------------------------------------- #
def test_best_dict_margin_is_against_the_mode_gate_not_1p55(tmp_path):
    drv = _drv(tmp_path / "best", store_dir=_bare_store(tmp_path))
    row = _row(peak=1.60, cov=0.32, f_r=1.62, max_pin_burnup=70.0, pattern=None)
    bd = drv._best_dict(row, drv._campaign_objective(row))
    # the gate this mode APPLIES is 1.70 — reporting -0.07 "over 1.55" described
    # a limit flat_power never enforced.
    assert bd["f_r_limit_applied"] == pytest.approx(1.70)
    assert bd["f_r_margin_to_limit"] == pytest.approx(1.70 - 1.62)
    # the LICENSING number lives in its own column (§2.2, decision D2).
    assert bd["compliance_limit"] == pytest.approx(1.55)
    assert bd["compliance_margin"] == pytest.approx(1.55 - 1.62)
    assert bd["node_peak"] == pytest.approx(1.60)
    assert bd["map_cov"] == pytest.approx(0.32)
    assert bd["distance"] is None


def test_other_modes_keep_reporting_against_acq_f_r_limit(tmp_path):
    """Regression pin: only flat_power's applied limit changed."""
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / "mfr", harvest=False)
    cfg.acquisition.objective = "min_fr_max_cycle"
    stub = StubEvaluator()
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                         run_dir=tmp_path / "mfr" / "run", progress=False,
                         log=lambda m: None)
    row = {"converged": True, "f_r": 1.62, "cbc_max": 1500.0, "f_q": 2.30,
           "ao_abs": 0.20, "cyclen": 745.0, "record_id": "r"}
    bd = drv._best_dict(row, drv._campaign_objective(row))
    assert bd["f_r_limit_applied"] == pytest.approx(1.55)
    assert bd["f_r_margin_to_limit"] == pytest.approx(1.55 - 1.62)


def test_best_dict_routes_flat_power_without_a_cyclen(tmp_path):
    drv = _drv(tmp_path / "nocy2")
    row = _row(peak=1.55, cov=0.30, pattern=None)
    row.pop("cyclen")
    drv._maybe_update_best(row, None)          # must not raise on |cyclen-target|
    assert drv.best is not None and drv.best["distance"] is None
    assert drv.best["node_peak"] == pytest.approx(1.55)


def test_delivery_artifact_applies_the_band_and_excludes_the_flattest(tmp_path):
    drv = _drv(tmp_path / "deliver")
    peaks = np.linspace(1.30, 1.68, 12)
    for i, p in enumerate(peaks):
        drv.campaign_rows.append(_row(peak=float(p), cov=0.30,
                                      f_r=1.50 + 0.01 * i, record_id=f"d{i}",
                                      max_pin_burnup=70.0))
    payload = drv._write_delivery()
    assert payload is not None and payload["banded"]
    ranked = [c["record_id"] for c in payload["ranked"]]
    assert "d0" not in ranked                         # the flattest is NOT delivered
    excl = {c["record_id"]: c["reason"] for c in payload["excluded"]}
    assert excl["d0"] == "flatter than the band"
    # ranked by compliance margin to 1.55, descending.
    margins = [c["compliance_margin"] for c in payload["ranked"]]
    assert margins == sorted(margins, reverse=True)
    written = json.loads((drv.run_dir / "delivery.json").read_text("utf-8"))
    assert written["compliance_limit"] == pytest.approx(1.55)
    assert "downstream" in written["note"]


def test_delivery_artifact_is_flat_power_only(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / "nod", harvest=False)
    cfg.acquisition.objective = "fr_boundary"
    stub = StubEvaluator()
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                         run_dir=tmp_path / "nod" / "run", progress=False,
                         log=lambda m: None)
    assert drv._write_delivery() is None
    assert not (drv.run_dir / "delivery.json").exists()


# --------------------------------------------------------------------------- #
# §1.3 — map harvest rate: "no label" != "worse"
# --------------------------------------------------------------------------- #
def test_harvest_maps_is_asserted_at_construction(tmp_path):
    with pytest.raises(ValueError, match="harvest_maps"):
        _drv(tmp_path / "noharvest", harvest=False)


def test_map_harvest_rate_counts_converged_rows_only(tmp_path):
    drv = _drv(tmp_path / "rate")
    rows = [_row(peak=1.6), _row(peak=None), _row(peak=None, converged=False)]
    assert drv._map_harvest_rate(rows) == pytest.approx(0.5)
    assert drv._map_harvest_rate([]) is None
    assert drv._map_harvest_rate([_row(peak=None, converged=False)]) is None


def test_low_harvest_hard_aborts(tmp_path):
    drv = _drv(tmp_path / "abort", flatpower_min_map_harvest=0.5)
    drv._check_map_harvest(0.75)                       # fine
    with pytest.raises(MapHarvestAbort):
        drv._check_map_harvest(0.25)
    # threshold 0 disables the abort entirely.
    drv2 = _drv(tmp_path / "abort0", flatpower_min_map_harvest=0.0)
    drv2._check_map_harvest(0.0)


def test_harvest_abort_never_fires_for_other_objectives(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / "frb", harvest=False)
    cfg.acquisition.objective = "fr_boundary"
    stub = StubEvaluator()
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                         run_dir=tmp_path / "frb" / "run", progress=False,
                         log=lambda m: None)
    drv._check_map_harvest(0.0)                        # no raise


def test_unjudgeable_wave_is_not_a_no_improvement_wave(tmp_path):
    drv = _drv(tmp_path / "judge")
    labelled = [_row(peak=1.6), _row(peak=None)]
    unlabelled = [_row(peak=None), _row(peak=None)]
    assert drv._wave_judgeable(labelled) is True
    assert drv._wave_judgeable(unlabelled) is False
    # nothing converged at all IS a real non-result (judgeable).
    assert drv._wave_judgeable([_row(peak=None, converged=False)]) is True


def test_other_objectives_are_always_judgeable(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / "j2", harvest=False)
    cfg.acquisition.objective = "fr_boundary"
    stub = StubEvaluator()
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                         run_dir=tmp_path / "j2" / "run", progress=False,
                         log=lambda m: None)
    assert drv._wave_judgeable([_row(peak=None), _row(peak=None)]) is True


# --------------------------------------------------------------------------- #
# §10 STOP — elite-mutation parent seeding is OBJECTIVE-aware, not F_r-aware
#
# The regression: ``_near_miss_parents`` seeded the tight (n_moves=1) local-search
# arm by ``f_r <= near_miss_f_r`` in EVERY mode, so a flat_power campaign aimed
# its most exploitative moves at low-F_r parents.
# --------------------------------------------------------------------------- #
def _patterned_rows(n: int = 6):
    """``n`` converged rows with real patterns, flatness DESCENDING with F_r."""
    import random

    from lpopt.search.genome import random_genome

    rng = random.Random(7)
    rows = []
    for i in range(n):
        pat = random_genome(rng, "K1_K2", 30).to_pattern()
        rows.append(_row(peak=1.30 + 0.05 * i,          # row 0 is the FLATTEST
                         cov=0.10 + 0.01 * i,
                         f_r=1.68 - 0.03 * i,           # row 0 has the WORST F_r
                         record_id=f"n{i}",
                         pattern=pat.canonical()))
    return rows


def test_flat_power_seeds_near_miss_parents_by_flatness_not_f_r(tmp_path):
    drv = _drv(tmp_path / "nm")
    rows = _patterned_rows()
    drv.campaign_rows.extend(rows)
    parents = drv._near_miss_parents()
    assert parents, "flat_power must still seed the tight local-search arm"
    ids = [rid for rid, _pat in parents]
    # flattest FIRST; the F_r rule would have ordered these exactly backwards and
    # excluded n0/n1 entirely (f_r 1.68 / 1.65 > the 1.60 bound).
    assert ids[0] == "n0"
    assert ids == [f"n{i}" for i in range(len(rows))]
    assert set(ids) >= {"n0", "n1"}


def test_other_objectives_keep_the_f_r_near_miss_rule(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / "nmfr", harvest=False)
    cfg.acquisition.objective = "fr_boundary"
    stub = StubEvaluator()
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                         run_dir=tmp_path / "nmfr" / "run", progress=False,
                         log=lambda m: None)
    drv.campaign_rows.extend(_patterned_rows())
    ids = [rid for rid, _pat in drv._near_miss_parents()]
    # unchanged: only rows at/below near_miss_f_r (1.60) qualify.
    assert ids == ["n3", "n4", "n5"]


def test_flat_power_near_miss_excludes_unlabelled_rows_and_honours_top_k(tmp_path):
    drv = _drv(tmp_path / "nmk")
    drv.search.near_miss_top_k = 2
    rows = _patterned_rows()
    rows[1]["node_peak"] = None                    # NO objective value (§1.3)
    drv.campaign_rows.extend(rows)
    ids = [rid for rid, _pat in drv._near_miss_parents()]
    assert ids == ["n0", "n2"]


def test_near_miss_off_switch_still_disables_the_arm_under_flat_power(tmp_path):
    drv = _drv(tmp_path / "nmoff")
    drv.search.near_miss_f_r = 0.0
    drv.campaign_rows.extend(_patterned_rows())
    assert drv._near_miss_parents() == []


# --------------------------------------------------------------------------- #
# §1.2 / D4 — a persisted flat_power objective carries its NORMALIZER identity
# --------------------------------------------------------------------------- #
def test_best_records_the_scale_identity_it_was_computed_in(tmp_path):
    drv = _drv(tmp_path / "id")
    drv._maybe_update_best(_row(peak=1.55, cov=0.30), None)
    ident = drv.best["objective_scale"]
    assert ident["version"] == SCALAR_VERSION
    assert ident["peak_scale"] == pytest.approx(drv.flat_peak_scale)
    assert ident["cov_scale"] == pytest.approx(drv.flat_cov_scale)
    assert ident["w_cov"] == pytest.approx(drv.flat_w_cov)
    drv._save_state()
    state = json.loads((drv.run_dir / "state.json").read_text("utf-8"))
    assert state["flat_scale"] == ident


def test_other_objectives_record_no_scale(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / "noid", harvest=False)
    cfg.acquisition.objective = "fr_boundary"
    stub = StubEvaluator()
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                         run_dir=tmp_path / "noid" / "run", progress=False,
                         log=lambda m: None)
    assert drv.flat_scale_id is None
    drv._maybe_update_best(_row(peak=None, cov=None), None)
    assert drv.best["objective_scale"] is None


def _resumed(tmp_path, state: dict, **kw):
    """A driver resumed against a hand-written ``state.json``."""
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "state.json").write_text(json.dumps(state), encoding="utf-8")
    cfg = _cfg(tmp_path, **kw)
    stub = StubEvaluator()
    return CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                          run_dir=run, resume=True, progress=False,
                          log=lambda m: None)


def test_resume_migrates_a_best_written_under_a_different_normalizer(tmp_path):
    """A refit of flat_scale.json silently redefines a stored objective value."""
    stale = {
        "wave_index": 3, "budget_spent": 24,
        "best": {"record_id": "b0", "objective": -999.0,
                 "node_peak": 1.55, "map_cov": 0.30},
        "flat_scale": {"version": SCALAR_VERSION, "peak_scale": 0.111,
                       "cov_scale": 0.222, "w_cov": 0.5},
    }
    drv = _resumed(tmp_path / "mig", stale)
    drv._load_state()
    expected = -(1.55 / drv.flat_peak_scale + drv.flat_w_cov * 0.30 / drv.flat_cov_scale)
    assert drv.best["objective"] == pytest.approx(expected)
    assert drv.best["objective_scale"] == drv.flat_scale_id


def test_resume_leaves_a_matching_normalizer_untouched(tmp_path):
    drv0 = _drv(tmp_path / "same0")
    ident = drv0.flat_scale_id
    state = {
        "wave_index": 1, "budget_spent": 8,
        "best": {"record_id": "b0", "objective": -1.25,
                 "node_peak": 1.55, "map_cov": 0.30},
        "flat_scale": ident,
    }
    drv = _resumed(tmp_path / "same", state)
    drv._load_state()
    assert drv.best["objective"] == pytest.approx(-1.25)     # NOT recomputed


def test_resume_refuses_an_unmigratable_stale_best(tmp_path):
    state = {
        "wave_index": 3, "budget_spent": 24,
        "best": {"record_id": "b0", "objective": -999.0},   # no node_peak label
        "flat_scale": {"version": SCALAR_VERSION, "peak_scale": 0.111,
                       "cov_scale": 0.222, "w_cov": 0.5},
    }
    drv = _resumed(tmp_path / "refuse", state)
    with pytest.raises(ValueError, match="resume refused"):
        drv._load_state()


def test_resume_of_a_state_with_no_recorded_scale_is_treated_as_a_mismatch(tmp_path):
    """Silence is not proof: a pre-identity state.json takes the migrate path."""
    state = {
        "wave_index": 1, "budget_spent": 8,
        "best": {"record_id": "b0", "objective": -999.0,
                 "node_peak": 1.55, "map_cov": 0.30},
    }
    drv = _resumed(tmp_path / "silent", state)
    drv._load_state()
    assert drv.best["objective"] != pytest.approx(-999.0)
    assert drv.best["objective_scale"] == drv.flat_scale_id


# --------------------------------------------------------------------------- #
# D9 — the SDM/MTC licensing MASTER budget is PERSISTED, never re-spent
# --------------------------------------------------------------------------- #
def test_post_verify_accounting_is_persisted_and_resumed(tmp_path):
    drv = _drv(tmp_path / "pv")
    drv.post_verify_calls = 7
    drv.post_verify_violators = ["rid_a"]
    drv.post_verify_done = True
    drv._save_state()
    state = json.loads((drv.run_dir / "state.json").read_text("utf-8"))
    assert state["post_verify_calls"] == 7
    assert state["post_verify_done"] is True

    resumed = _resumed(tmp_path / "pv2", state)
    resumed._load_state()
    assert resumed.post_verify_calls == 7
    assert resumed.post_verify_violators == ["rid_a"]
    assert resumed.post_verify_done is True


def test_post_verify_gate_does_not_re_spend_on_a_resumed_run(tmp_path):
    from lpopt.config import ConstraintsConfig

    class _Boom:
        """Any branch execution here would be a licensing double-spend."""

        def run(self, *a, **kw):
            raise AssertionError("the D9 gate re-ran and re-spent MASTER")

    drv = _drv(tmp_path / "pv3")
    drv.cfg.constraints = ConstraintsConfig(mtc_enable=True, post_verify_top_k=2)
    # an injected executor bypasses the dry-run short-circuit, so ONLY the
    # already-ran guard can stop the gate here.
    drv.post_verify_executor = _Boom()
    drv.post_verify_calls = 5
    drv.post_verify_done = True
    logs: list[str] = []
    drv._log = logs.append

    drv._maybe_post_verify({"ranked": [{"record_id": "r0"}]})
    assert drv.post_verify_calls == 5
    assert any("already ran" in m for m in logs)


def test_flat_power_holdout_prefers_rows_that_carry_a_flatness_label(tmp_path):
    """The panel's primaries are node_peak/map_cov — an unmapped holdout is blind."""
    drv = _drv(tmp_path / "hold")
    rows = [_row(peak=1.40 + 0.01 * i, record_id=f"h{i}") for i in range(4)]
    rows += [_row(peak=None, record_id=f"u{i}") for i in range(4)]
    drv._case_store_rows = lambda converged=True: list(rows)
    got = drv._holdout_rows()
    assert got and all(drv.has_flat_label(r) for r in got)

    # fewer than two mapped rows: fall back, and SAY the panel will be blind.
    logs: list[str] = []
    drv._log = logs.append
    drv._case_store_rows = lambda converged=True: [_row(peak=None, record_id="u0"),
                                                   _row(peak=None, record_id="u1")]
    assert len(drv._holdout_rows()) == 2
    assert any("node_peak label" in m for m in logs)


def test_other_objectives_keep_the_f_r_cyclen_holdout_rule(tmp_path):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / "hold2", harvest=False)
    cfg.acquisition.objective = "fr_boundary"
    stub = StubEvaluator()
    drv = CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                         run_dir=tmp_path / "hold2" / "run", progress=False,
                         log=lambda m: None)
    rows = [_row(peak=None, record_id=f"u{i}") for i in range(3)]
    drv._case_store_rows = lambda converged=True: list(rows)
    assert len(drv._holdout_rows()) == 3       # unmapped rows are fine here


# --------------------------------------------------------------------------- #
# §10 STOP — the retired F_r-steered production modes announce themselves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("objective", ["max_cycle_min_fr", "min_fr_max_cycle"])
def test_retired_production_objectives_log_a_deprecation(tmp_path, objective):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / objective, harvest=False)
    cfg.acquisition.objective = objective
    logs: list[str] = []
    stub = StubEvaluator()
    CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                   run_dir=tmp_path / objective / "run", progress=False,
                   log=logs.append)
    assert any("DEPRECATED" in m and objective in m for m in logs)


@pytest.mark.parametrize("objective", ["flat_power", "fr_boundary"])
def test_live_objectives_do_not_log_a_deprecation(tmp_path, objective):
    from lpopt.search.campaign import CampaignDriver

    cfg = _cfg(tmp_path / objective, harvest=(objective == "flat_power"))
    cfg.acquisition.objective = objective
    logs: list[str] = []
    stub = StubEvaluator()
    CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                   run_dir=tmp_path / objective / "run", progress=False,
                   log=logs.append)
    assert not any("DEPRECATED" in m for m in logs)


# --------------------------------------------------------------------------- #
# end-to-end (stub MASTER)
# --------------------------------------------------------------------------- #
def _records_store(tmp_path: Path) -> Path:
    """The fixture store PLUS the shipped record tables, copied in.

    The end-to-end run wants real rows to seed its pool — it does not want the
    live ``map_calibration.json`` that sits beside them, because that artifact
    belongs to whichever checkpoint is champion today.  Copying (never linking)
    the two tables keeps the real-data value of this test while its calibration
    stays the fixture one, so a promotion cannot reach it.
    """
    store = fixture_store(tmp_path, fixture_champion(tmp_path))
    for name in ("records.parquet", "fuel_types.parquet"):
        src = _STORE / name
        if src.exists():
            shutil.copy2(src, store / name)
    return store


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_flat_power_campaign_runs_end_to_end(tmp_path):
    from lpopt.search.campaign import run_campaign

    cfg = _cfg(tmp_path / "e2e", budget=8, store_dir=_records_store(tmp_path / "e2e"))
    stub = StubEvaluator()
    result = run_campaign(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                          run_dir=tmp_path / "e2e" / "run",
                          backend_factory=lambda ckpt: FakeModel(),
                          early_stop=False, progress=False)
    # the StubEvaluator harvests no maps, so the run must END on the harvest
    # abort rather than silently ranking nothing — that is the §1.3 contract.
    assert result.status in ("map_harvest_abort", "complete", "stalled")
    if result.status == "map_harvest_abort":
        assert result.budget_spent > 0            # labels were still committed
        status = json.loads(
            (tmp_path / "e2e" / "run" / "status.json").read_text("utf-8"))
        assert status["status"] == "map_harvest_abort"


# --------------------------------------------------------------------------- #
# wave error surfacing — "100 calls, only symptom conv=0"
# --------------------------------------------------------------------------- #
def test_wave_errors_are_counted_and_the_first_one_is_named(tmp_path, monkeypatch):
    """An all-error wave must say so, and say WHY, on the progress line.

    ``status="error"`` (a chain that never ran: staging refused, deck rejected,
    NaN flux) used to be indistinguishable from ``nonconverged`` — both reported
    only ``conv=0``, which reads as "the search is hard" instead of "nothing
    ran".  A clean wave's line stays byte-identical (ECC audit).
    """
    from lpopt.search import campaign as C
    from lpopt.search.verify import WaveOutcome

    said: list[str] = []
    cfg = _cfg(tmp_path / "err", flatpower_min_map_harvest=0.0)
    stub = StubEvaluator()
    drv = C.CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                           run_dir=tmp_path / "err" / "run", progress=True,
                           log=said.append)

    clean = drv._run_wave(4, reserve=False)
    assert clean.errors == 0
    line = [m for m in said if m.startswith("[optimize] wave")][-1]
    assert " err=" not in line                    # zero-error line unchanged

    def _all_errors(entries):
        return [WaveOutcome(status="error", fom=None, n_cycles=0,
                            tolerance_margin=None, wall_s=0.0,
                            restart_provenance="", failure="stage_failed",
                            converged_at_cap=False, case_key=e.case_key,
                            pattern=e.pattern, meta=dict(e.meta))
                for e in entries]

    monkeypatch.setattr(drv.verifier, "evaluate_wave", _all_errors)
    report = drv._run_wave(4, reserve=False)
    assert report.errors == 4 and report.converged == 0
    line = [m for m in said if m.startswith("[optimize] wave")][-1]
    assert " err=4 [stage_failed]" in line


def test_champion_save_and_reload_failures_are_logged_not_swallowed(tmp_path):
    """Both swallowed-exception paths around the champion pointer now speak.

    ``_save_champion``: the gate ACCEPTED the challenger, so the in-memory model
    is already the new champion while the served pointer silently stays on the
    old checkpoint.  ``_load_state``: a failed reload silently resumes on the
    construction-time model, discarding every wave of fine-tuning.  Both remain
    non-fatal by design (ponytail ceiling) — but never silent (ECC audit).
    """
    from lpopt.search import campaign as C

    said: list[str] = []
    cfg = _cfg(tmp_path / "ptr", flatpower_min_map_harvest=0.0)
    stub = StubEvaluator()

    def _boom(ckpt):
        raise OSError("checkpoint unreadable")

    drv = C.CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                           run_dir=tmp_path / "ptr" / "run", progress=False,
                           log=said.append, resume=True, backend_factory=_boom)

    class _UnsaveableModel:
        def save(self, out):
            raise OSError("disk full")

    drv.model = _UnsaveableModel()
    stale = drv.champion_ckpt
    assert str(drv._save_champion()) == stale        # stale pointer still returned
    assert any("champion save FAILED" in m and "disk full" in m for m in said)

    said.clear()
    drv._save_state()
    assert drv._load_state() is True                 # resume still succeeds
    assert any("champion reload FAILED" in m and "checkpoint unreadable" in m
               for m in said)
