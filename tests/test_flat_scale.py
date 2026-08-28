"""Flatness objective normalization scales (program 20260725 §1.2, decision D4).

The point of these tests is not that the arithmetic works — it is that the
DECLARED weight ratio (node_peak 1.0 : map_cov 0.5) is the ratio the objective
actually realizes, and that the cost of the global-constant fallback is measured
rather than assumed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lpopt.data.flat_scale import (
    ARTIFACT_NAME, SCALAR_VERSION, CellScale, DEFAULT_COV_SCALE,
    DEFAULT_PEAK_SCALE, DEFAULT_W_COV, FlatScale, identity_matches,
)
from lpopt.tools.fit_flat_scale import evaluation_slice, fit_cells, fit_flat_scale


# --------------------------------------------------------------------------- #
# the scalar
# --------------------------------------------------------------------------- #
def test_scalar_is_negated_weighted_z_sum():
    fs = FlatScale(peak_scale=0.4, cov_scale=0.08)
    got = fs.scalar_one(1.6, 0.32, w_cov=0.5)
    assert got == pytest.approx(-(1.6 / 0.4 + 0.5 * 0.32 / 0.08))


def test_scalar_ranks_flatter_higher():
    fs = FlatScale(peak_scale=0.4, cov_scale=0.08)
    flat = fs.scalar_one(1.50, 0.30)
    peaky = fs.scalar_one(1.70, 0.30)
    assert flat > peaky
    rough = fs.scalar_one(1.50, 0.40)
    assert flat > rough                      # cov still shapes, at half weight


def test_peak_is_primary_one_sd_each_way():
    """One within-cell SD of peak must beat one SD of cov at w_cov = 0.5."""
    fs = FlatScale(peak_scale=0.4, cov_scale=0.08)
    better_peak = fs.scalar_one(1.60 - 0.40, 0.30 + 0.08)
    better_cov = fs.scalar_one(1.60, 0.30)
    assert better_peak > better_cov
    assert (better_peak - better_cov) == pytest.approx(1.0 - 0.5)


def test_missing_peak_is_minus_inf_missing_cov_drops_the_secondary_term():
    fs = FlatScale(peak_scale=0.4, cov_scale=0.08)
    assert fs.scalar_one(None, 0.30) == -np.inf
    assert fs.scalar_one(float("nan"), 0.30) == -np.inf
    assert fs.scalar_one(1.6, None) == pytest.approx(-1.6 / 0.4)
    assert fs.scalar_one(1.6, float("nan")) == pytest.approx(-1.6 / 0.4)


def test_scalar_is_vectorized_and_matches_scalar_one():
    fs = FlatScale(peak_scale=0.4, cov_scale=0.08)
    peaks = np.array([1.4, 1.6, np.nan])
    covs = np.array([0.30, np.nan, 0.30])
    vec = fs.scalar(peaks, covs)
    assert vec[0] == pytest.approx(fs.scalar_one(1.4, 0.30))
    assert vec[1] == pytest.approx(fs.scalar_one(1.6, None))
    assert vec[2] == -np.inf


# --------------------------------------------------------------------------- #
# per-cell resolution
# --------------------------------------------------------------------------- #
def _doc():
    return {
        "global": {"peak_scale": 0.40, "cov_scale": 0.08},
        "cells": {
            "feed=121|ebin=5.4": {"n": 100, "peak_scale": 0.20, "cov_scale": 0.06},
            "feed=117|ebin=5.0": {"n": 40, "peak_scale": 0.60, "cov_scale": 0.05},
        },
    }


def test_per_cell_scales_are_used_when_fitted_and_global_otherwise():
    fs = FlatScale.from_artifact(_doc())
    assert fs.scales_for("feed=121|ebin=5.4") == (0.20, 0.06)
    assert fs.scales_for("feed=999|ebin=9.9") == (0.40, 0.08)
    assert fs.scales_for(None) == (0.40, 0.08)
    assert fs.has_cell("feed=121|ebin=5.4") and not fs.has_cell("nope")


def test_per_cell_off_pins_every_cell_to_the_global_constants():
    fs = FlatScale.from_artifact(_doc(), per_cell=False)
    assert fs.scales_for("feed=121|ebin=5.4") == (0.40, 0.08)
    assert not fs.has_cell("feed=121|ebin=5.4")


def test_per_cell_normalization_makes_the_declared_weight_exact():
    """D4, the whole reason per-cell is the default.

    In SD units the secondary term's realized weight must equal the DECLARED
    w_cov in every fitted cell.  Under global constants it does not.
    """
    fs = FlatScale.from_artifact(_doc())
    for cell, (sd_p, sd_c) in {
        "feed=121|ebin=5.4": (0.20, 0.06),
        "feed=117|ebin=5.0": (0.60, 0.05),
    }.items():
        base = fs.scalar_one(1.6, 0.30, cell_key=cell)
        one_sd_peak = base - fs.scalar_one(1.6 + sd_p, 0.30, cell_key=cell)
        one_sd_cov = base - fs.scalar_one(1.6, 0.30 + sd_c, cell_key=cell)
        assert one_sd_peak == pytest.approx(1.0)
        assert one_sd_cov == pytest.approx(DEFAULT_W_COV)

    # global constants: the SAME two cells realize DIFFERENT secondary weights.
    fg = FlatScale.from_artifact(_doc(), per_cell=False)
    realized = []
    for cell, (sd_p, sd_c) in {
        "feed=121|ebin=5.4": (0.20, 0.06),
        "feed=117|ebin=5.0": (0.60, 0.05),
    }.items():
        base = fg.scalar_one(1.6, 0.30, cell_key=cell)
        p = base - fg.scalar_one(1.6 + sd_p, 0.30, cell_key=cell)
        c = base - fg.scalar_one(1.6, 0.30 + sd_c, cell_key=cell)
        realized.append(c / p)
    assert realized[0] != pytest.approx(realized[1])
    assert max(realized) / min(realized) > 3.0


def test_realized_w_cov_report_is_exact_under_per_cell_and_spread_under_global():
    fs = FlatScale.from_artifact(_doc())
    r = fs.realized_w_cov()
    assert r["per_cell"] and r["min"] == r["max"] == pytest.approx(DEFAULT_W_COV)
    fg = FlatScale.from_artifact(_doc(), per_cell=False)
    rg = fg.realized_w_cov()
    assert rg["min"] < DEFAULT_W_COV < rg["max"]
    assert rg["spread"] > 3.0


def test_absent_artifact_falls_back_to_the_measured_module_defaults(tmp_path):
    fs = FlatScale.from_store(tmp_path)
    assert fs.scales_for() == (DEFAULT_PEAK_SCALE, DEFAULT_COV_SCALE)
    assert "absent" in fs.source
    assert FlatScale.from_store(None).cells == {}


def test_malformed_artifact_entries_are_skipped_not_trusted(tmp_path):
    (tmp_path / ARTIFACT_NAME).write_text(json.dumps({
        "global": {"peak_scale": 0.4, "cov_scale": 0.08},
        "cells": {
            "ok": {"n": 10, "peak_scale": 0.2, "cov_scale": 0.05},
            "zero": {"n": 10, "peak_scale": 0.0, "cov_scale": 0.05},
            "nan": {"n": 10, "peak_scale": "x", "cov_scale": 0.05},
            "notadict": 5,
        },
    }), encoding="utf-8")
    fs = FlatScale.from_store(tmp_path)
    assert set(fs.cells) == {"ok"}
    assert fs.scales_for("zero") == (0.4, 0.08)      # degenerate -> global


def test_unreadable_artifact_is_not_fatal(tmp_path):
    (tmp_path / ARTIFACT_NAME).write_text("{ not json", encoding="utf-8")
    fs = FlatScale.from_store(tmp_path)
    assert fs.scales_for() == (DEFAULT_PEAK_SCALE, DEFAULT_COV_SCALE)


# --------------------------------------------------------------------------- #
# the fitter
# --------------------------------------------------------------------------- #
def _frame(cells: dict[str, tuple[int, float, float]]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for (feed, ebin), (n, mu_p, sd_p) in cells.items():
        peaks = rng.normal(mu_p, sd_p, n)
        covs = rng.normal(0.30, sd_p / 4.0, n)
        for p, c in zip(peaks, covs):
            rows.append({"converged": True, "valid": True, "feed": feed,
                         "e_core": ebin + 0.01, "node_peak": p, "map_cov": c})
    return pd.DataFrame(rows)


def test_evaluation_slice_keeps_only_labelled_converged_rows():
    df = pd.DataFrame([
        {"converged": True, "valid": True, "feed": 121, "e_core": 5.42,
         "node_peak": 1.6, "map_cov": 0.3},
        {"converged": False, "valid": True, "feed": 121, "e_core": 5.42,
         "node_peak": 1.6, "map_cov": 0.3},
        {"converged": True, "valid": True, "feed": 121, "e_core": 5.42,
         "node_peak": None, "map_cov": 0.3},
    ])
    sl = evaluation_slice(df)
    assert len(sl) == 1 and sl["cell"].iloc[0] == "feed=121|ebin=5.4"


def test_fit_cells_respects_the_row_floor():
    df = _frame({(121, 5.4): (30, 1.6, 0.2), (117, 5.0): (4, 1.4, 0.1)})
    cells, seen = fit_cells(evaluation_slice(df), min_rows=8)
    assert seen == 2 and len(cells) == 1
    assert cells[0].cell == "feed=121|ebin=5.4"
    assert cells[0].peak_scale == pytest.approx(0.2, rel=0.4)


def test_fit_writes_the_artifact_and_reports_the_realized_weight(tmp_path):
    df = _frame({(121, 5.4): (60, 1.6, 0.20), (117, 5.0): (60, 1.4, 0.60)})
    store = tmp_path / "store"
    store.mkdir()

    class _Reader:
        records = df

    import lpopt.tools.fit_flat_scale as M
    orig = M.StoreReader
    M.StoreReader = lambda _d: _Reader()
    try:
        rep = fit_flat_scale(store, min_rows=8, log=lambda m: None)
    finally:
        M.StoreReader = orig
    assert rep.wrote and rep.n_cells_fitted == 2
    doc = json.loads((store / ARTIFACT_NAME).read_text(encoding="utf-8"))
    assert set(doc["cells"]) == {"feed=121|ebin=5.4", "feed=117|ebin=5.0"}
    # the honesty number is present and shows the global-constant spread.
    r = doc["realized_w_cov_with_global_scales"]
    assert r["declared"] == 0.5 and r["spread"] > 1.0
    # and the artifact round-trips into a FlatScale that resolves both cells.
    fs = FlatScale.from_store(store)
    assert fs.has_cell("feed=121|ebin=5.4") and fs.has_cell("feed=117|ebin=5.0")


def test_fit_dry_run_writes_nothing(tmp_path):
    df = _frame({(121, 5.4): (30, 1.6, 0.2)})
    store = tmp_path / "store"
    store.mkdir()

    class _Reader:
        records = df

    import lpopt.tools.fit_flat_scale as M
    orig = M.StoreReader
    M.StoreReader = lambda _d: _Reader()
    try:
        rep = fit_flat_scale(store, min_rows=8, dry_run=True, log=lambda m: None)
    finally:
        M.StoreReader = orig
    assert not rep.wrote and not (store / ARTIFACT_NAME).exists()


def test_empty_slice_keeps_the_module_defaults(tmp_path):
    store = tmp_path / "store"
    store.mkdir()

    class _Reader:
        records = pd.DataFrame(columns=["converged", "valid", "feed", "e_core",
                                        "node_peak", "map_cov"])

    import lpopt.tools.fit_flat_scale as M
    orig = M.StoreReader
    M.StoreReader = lambda _d: _Reader()
    try:
        rep = fit_flat_scale(store, log=lambda m: None)
    finally:
        M.StoreReader = orig
    assert not rep.wrote
    assert rep.peak_scale == DEFAULT_PEAK_SCALE and rep.cov_scale == DEFAULT_COV_SCALE


# --------------------------------------------------------------------------- #
# the live artifact (the numbers the objective actually ships with)
# --------------------------------------------------------------------------- #
def test_module_defaults_are_the_measured_weighted_scales_not_the_draft():
    # The draft's 0.23 / 0.065 came from the UNWEIGHTED definition on a corpus
    # that was 87% two mega-cells; §1.2 says they must not be carried forward.
    assert DEFAULT_PEAK_SCALE != pytest.approx(0.23)
    assert DEFAULT_COV_SCALE != pytest.approx(0.065)
    assert 0.2 < DEFAULT_PEAK_SCALE < 0.6
    assert 0.05 < DEFAULT_COV_SCALE < 0.12


# --------------------------------------------------------------------------- #
# scale IDENTITY — a stored objective value must say what units it is in
#
# A flat_power ``best["objective"]`` is a number in the units of the cell's
# normalizer.  Re-fitting flat_scale.json silently redefines it, and nothing in
# the persisted state said so; this is what makes the redefinition detectable.
# --------------------------------------------------------------------------- #
def test_identity_carries_the_scalar_version_and_the_applied_scales():
    fs = FlatScale(peak_scale=0.37, cov_scale=0.081,
                   cells={"c": CellScale("c", 50, 0.10, 0.04)})
    ident = fs.identity(w_cov=0.5, cell_key="c")
    assert ident["version"] == SCALAR_VERSION
    assert ident["peak_scale"] == pytest.approx(0.10)     # the CELL's scales
    assert ident["cov_scale"] == pytest.approx(0.04)
    assert ident["w_cov"] == pytest.approx(0.5)
    assert ident["fitted"] is True
    # an unfitted cell reports the global fallback, and says it fell back.
    other = fs.identity(w_cov=0.5, cell_key="nope")
    assert other["peak_scale"] == pytest.approx(0.37) and other["fitted"] is False


def test_identity_round_trips_through_json():
    fs = FlatScale(peak_scale=0.370285, cov_scale=0.081127)
    ident = fs.identity()
    assert identity_matches(json.loads(json.dumps(ident)), ident)


def test_identity_matches_detects_a_refit():
    base = FlatScale(peak_scale=0.37, cov_scale=0.081).identity()
    refit = FlatScale(peak_scale=0.39, cov_scale=0.081).identity()
    assert identity_matches(base, base) is True
    assert identity_matches(base, refit) is False
    # a changed w_cov is just as much a redefinition as a changed scale.
    assert identity_matches(base, FlatScale(peak_scale=0.37, cov_scale=0.081)
                            .identity(w_cov=0.25)) is False


def test_identity_matches_is_null_intolerant():
    """Silence is not proof of sameness — an absent identity is a MISMATCH."""
    ident = FlatScale().identity()
    assert identity_matches(None, ident) is False
    assert identity_matches({}, ident) is False
    assert identity_matches({"peak_scale": ident["peak_scale"]}, ident) is False
    assert identity_matches({**ident, "version": "other"}, ident) is False
