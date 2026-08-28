"""MASTER-verified debug panel (lpopt.tools.debug_panel + the CLI subcommand).

Covers, in the order the module is used:

* panel MEMBERSHIP — campaign globs, the converged filter, an empty panel;
* score() MATH on a synthetic frame — MAE / bias / max|err| / bias-share, and the
  err/sigma coverage block that is the OOD-overconfidence detector;
* TOLERANCE VERDICTS — pass/fail either side of the neutronics thresholds, and
  deck overrides;
* MISSING-FLATNESS rows — a model with no map head, a map head that raises, and
  NaN truth columns must all degrade to "unscored", never to an exception;
* the ``cbc_kind == "boc_only"`` truth censor (a BOC boron reading is not the
  EDIT2 maximum the head predicts);
* the CLI: report-only, ALWAYS exit 0, even with a target far out of tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.tools import debug_panel as dp

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


# --------------------------------------------------------------------------- #
# panel membership
# --------------------------------------------------------------------------- #
def _records(campaigns, converged=None) -> pd.DataFrame:
    n = len(campaigns)
    return pd.DataFrame({
        "record_id": [f"r{i}" for i in range(n)],
        "campaign": campaigns,
        "converged": [True] * n if converged is None else converged,
        "cyclen": np.linspace(620.0, 640.0, n),
    })


def test_campaign_globs_select_the_panel() -> None:
    df = _records(["debug_panel_v1", "democase_A", "5-5.25_f101",
                   "debug_panelXYZ", None, "predebug_panel"])
    keep = dp.panel_frame(df)
    assert set(keep["campaign"]) == {"debug_panel_v1", "democase_A", "debug_panelXYZ"}
    # a None/NaN campaign never matches (it is provenance-less, not a panel row)
    assert dp.campaign_matches(None, dp.DEFAULT_PANEL_CAMPAIGNS) is False
    assert dp.campaign_matches(float("nan"), dp.DEFAULT_PANEL_CAMPAIGNS) is False
    # custom globs replace the defaults wholesale
    assert set(dp.panel_frame(df, ["5-5.25*"])["campaign"]) == {"5-5.25_f101"}


def test_panel_drops_non_converged_rows() -> None:
    df = _records(["debug_panel_a", "debug_panel_b"], converged=[True, False])
    assert list(dp.panel_frame(df)["record_id"]) == ["r0"]
    # forensics escape hatch keeps them
    assert len(dp.panel_frame(df, converged_only=False)) == 2


def test_empty_panel_is_not_an_error() -> None:
    df = _records(["5-5.25_f101", "P0_pathfinder"])
    frame = dp.panel_frame(df)
    assert len(frame) == 0
    report = dp.score_frame(frame, {}, None)
    assert report["n_panel_rows"] == 0
    assert report["passed"] == [] and report["failed"] == []
    assert set(report["unscored"]) == set(dp.PANEL_TARGETS)
    assert report["all_within_tolerance"] is False      # nothing proven, not "pass"


# --------------------------------------------------------------------------- #
# score() math on a synthetic frame
# --------------------------------------------------------------------------- #
def test_score_target_accuracy_math() -> None:
    truth = np.array([600.0, 610.0, 620.0, 630.0])
    pred = truth + np.array([2.0, -4.0, 6.0, 0.0])
    out = dp.score_target(pred, truth, tolerance=3.0, unit="EFPD")
    assert out["n"] == 4
    assert out["mae"] == pytest.approx((2 + 4 + 6 + 0) / 4)
    assert out["bias"] == pytest.approx((2 - 4 + 6 + 0) / 4)
    assert out["max_abs_err"] == pytest.approx(6.0)
    assert out["rmse"] == pytest.approx(np.sqrt((4 + 16 + 36) / 4))
    assert out["bias_share"] == pytest.approx(abs(out["bias"]) / out["mae"])
    assert out["unit"] == "EFPD"


def test_score_target_ignores_rows_without_finite_pred_or_truth() -> None:
    truth = np.array([600.0, np.nan, 620.0, 630.0])
    pred = np.array([605.0, 610.0, np.nan, 630.0])
    out = dp.score_target(pred, truth, tolerance=10.0)
    assert out["n"] == 2                                  # only rows 0 and 3
    assert out["mae"] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="length mismatch"):
        dp.score_target([1.0, 2.0], [1.0])


def test_score_target_bias_share_flags_a_correctable_shift() -> None:
    # a PURE offset: 100% bias share -> a per-cell affine calibration removes it.
    truth = np.linspace(1400.0, 1500.0, 40)
    out = dp.score_target(truth + 27.0, truth, tolerance=20.0)
    assert out["bias_share"] == pytest.approx(1.0)
    assert out["verdict"] == "FAIL"                       # 27 ppm > 20 ppm
    # pure symmetric scatter: bias ~ 0, share ~ 0 -> calibration cannot help.
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 30.0, 4000)
    scatter = dp.score_target(truth.mean() + noise, np.full(4000, truth.mean()),
                              tolerance=20.0)
    assert scatter["bias_share"] < 0.05


# --------------------------------------------------------------------------- #
# err/sigma coverage — the OOD-overconfidence detector
# --------------------------------------------------------------------------- #
def test_sigma_coverage_detects_overconfidence() -> None:
    truth = np.zeros(100)
    pred = np.zeros(100)
    pred[:10] = 5.0                                       # 10 rows 5 units off
    sigma = np.full(100, 1.0)                             # claims +/-1
    out = dp.score_target(pred, truth, sigma, tolerance=10.0)
    assert out["n_sigma"] == 100
    assert out["frac_abs_z_gt2"] == pytest.approx(0.10)
    assert out["max_abs_z"] == pytest.approx(5.0)
    assert out["mean_abs_z"] == pytest.approx(0.5)
    # a signed minimum surfaces the direction of the worst miss (the blind OOD
    # case was err/sigma = -12.8, i.e. a huge UNDER-prediction sold as certain).
    under = dp.score_target(-np.ones(4) * 12.8, np.zeros(4), np.ones(4))
    assert under["min_signed_z"] == pytest.approx(-12.8)


def test_sigma_block_absent_is_distinct_from_zero_misses() -> None:
    truth, pred = np.zeros(5), np.zeros(5)
    none = dp.score_target(pred, truth, None, tolerance=1.0)
    assert none["n_sigma"] == 0 and none["frac_abs_z_gt2"] is None
    # non-positive / non-finite sigmas are dropped, not treated as zero-width
    bad = dp.score_target(pred, truth, np.array([0.0, -1.0, np.nan, np.inf, 2.0]),
                          tolerance=1.0)
    assert bad["n_sigma"] == 1
    assert bad["frac_abs_z_gt2"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# tolerance verdicts
# --------------------------------------------------------------------------- #
def _panel_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": [f"p{i}" for i in range(n)],
        "campaign": ["debug_panel_v1"] * n,
        "converged": [True] * n,
        "cyclen": np.linspace(620.0, 630.0, n),
        "cbc_max": np.linspace(1400.0, 1450.0, n),
        "cbc_kind": ["max"] * n,
        "f_r": np.linspace(1.50, 1.60, n),
        "f_q": np.linspace(2.10, 2.20, n),
        "ao_abs": np.linspace(0.05, 0.08, n),
        "node_peak": np.linspace(1.30, 1.40, n),
        "map_cov": np.linspace(0.20, 0.24, n),
    })


def test_verdicts_against_the_neutronics_defaults() -> None:
    frame = _panel_frame()
    truth = {k: frame[k].to_numpy(dtype=float) for k in dp.PANEL_TARGETS}
    pred = {
        "cyclen": truth["cyclen"] + 1.0,        # 1.0 <= 3.0 EFPD -> PASS
        "cbc_max": truth["cbc_max"] + 42.0,     # 42 > 20 ppm     -> FAIL
        "f_r": truth["f_r"] - 0.01,             # PASS
        "f_q": truth["f_q"] + 0.20,             # 0.20 > 0.08     -> FAIL
        "ao_abs": truth["ao_abs"] + 0.002,      # PASS
        "node_peak": truth["node_peak"] + 0.01,  # PASS
        "map_cov": truth["map_cov"] + 0.05,     # 0.05 > 0.02     -> FAIL
    }
    report = dp.score_frame(frame, pred)
    assert set(report["failed"]) == {"cbc_max", "f_q", "map_cov"}
    assert set(report["passed"]) == {"cyclen", "f_r", "ao_abs", "node_peak"}
    assert report["unscored"] == []
    assert report["all_within_tolerance"] is False
    assert report["tolerances"]["cbc_max"] == pytest.approx(20.0)
    assert report["targets"]["cbc_max"]["mae"] == pytest.approx(42.0)


def test_tolerance_override_from_the_deck_table_flips_a_verdict() -> None:
    frame = _panel_frame()
    pred = {"cyclen": frame["cyclen"].to_numpy() + 4.0}
    assert dp.score_frame(frame, pred)["targets"]["cyclen"]["verdict"] == "FAIL"
    loose = dp.score_frame(frame, pred, tolerances={"cyclen": 5.0})
    assert loose["targets"]["cyclen"]["verdict"] == "PASS"
    assert loose["tolerances"]["cyclen"] == pytest.approx(5.0)
    # an override touches ONLY its own key
    assert loose["tolerances"]["cbc_max"] == pytest.approx(
        dp.DEFAULT_TOLERANCES["cbc_max"])


def test_every_panel_target_is_reported_even_when_unpredicted() -> None:
    frame = _panel_frame()
    report = dp.score_frame(frame, {"cyclen": frame["cyclen"].to_numpy()})
    # a head that emits nothing must still occupy a row in the table, otherwise it
    # silently stops being watched.
    assert set(report["targets"]) == set(dp.PANEL_TARGETS)
    assert report["targets"]["f_q"]["verdict"] == "NO DATA"
    assert report["targets"]["f_q"]["n"] == 0
    assert "f_q" in report["unscored"]


def test_cbc_boc_only_rows_are_censored_from_truth() -> None:
    frame = _panel_frame(4)
    frame.loc[[1, 3], "cbc_kind"] = "boc_only"
    pred = {"cbc_max": frame["cbc_max"].to_numpy(dtype=float)}
    out = dp.score_frame(frame, pred)["targets"]["cbc_max"]
    assert out["n"] == 2                       # the two boc_only rows never score
    assert out["mae"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# missing flatness / missing map head
# --------------------------------------------------------------------------- #
class _StubPred:
    def __init__(self, mean, std):
        self.mean = mean
        self.calibrated_std = std


class _StubModel:
    """Minimal duck-typed backend: constant surrogate columns, optional map head."""

    def __init__(self, *, mean_row, std_row, flat=None, flat_raises=False):
        self.mean_row = np.asarray(mean_row, dtype=float)
        self.std_row = np.asarray(std_row, dtype=float)
        self.flat = flat
        self.flat_raises = flat_raises
        if flat is not None or flat_raises:
            self.predict_map_flatness = self._flat        # attached only when real

    def predict(self, patterns, cases, cell=0.0):
        n = len(patterns)
        return _StubPred(np.tile(self.mean_row, (n, 1)),
                         np.tile(self.std_row, (n, 1)))

    def _flat(self, patterns, cases, cell=0.0):
        if self.flat_raises:
            raise RuntimeError("map head is missing from this checkpoint")
        n = len(patterns)
        return tuple(np.full(n, v) for v in self.flat)


def _stub_patterns(frame):
    """Attach real packed patterns/cases so predict_panel can decode them."""
    from lpopt.data.store import StoreReader
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    reader = StoreReader(STORE)
    src = reader.records[reader.records["library_id"].astype(str) == "ga80"].head(len(frame))
    if len(src) < len(frame):
        pytest.skip("not enough ga80 rows for the stub panel")
    out = frame.copy()
    out["pattern"] = list(src["pattern"].astype(str))
    out["case_pair"] = list(src["case_pair"].astype(str))
    out["feed"] = list(src["feed"].astype(int))
    return out


def test_model_without_map_head_leaves_flatness_unscored() -> None:
    frame = _stub_patterns(_panel_frame(3))
    model = _StubModel(mean_row=[1.55, 1400.0, 2.15, 625.0, 0.06, np.nan, 45.0],
                       std_row=[0.02, 15.0, 0.05, 5.0, 0.01, np.nan, 1.0])
    assert not hasattr(model, "predict_map_flatness")
    report = dp.score_panel(model, frame)
    for name in dp.FLATNESS_TARGETS:
        assert report["targets"][name]["n"] == 0
        assert report["targets"][name]["verdict"] == "NO DATA"
    assert any("no predict_map_flatness" in n for n in report["notes"])
    # the five scalar targets are still fully scored
    assert report["targets"]["cyclen"]["n"] == 3


def test_map_head_that_raises_is_reported_not_propagated() -> None:
    frame = _stub_patterns(_panel_frame(3))
    model = _StubModel(mean_row=[1.55, 1400.0, 2.15, 625.0, 0.06, np.nan, 45.0],
                       std_row=[0.02, 15.0, 0.05, 5.0, 0.01, np.nan, 1.0],
                       flat_raises=True)
    report = dp.score_panel(model, frame)          # must not raise
    assert report["targets"]["node_peak"]["verdict"] == "NO DATA"
    assert any("map head failed" in n for n in report["notes"])


def test_missing_truth_column_is_unscored_not_zero() -> None:
    frame = _stub_patterns(_panel_frame(3)).drop(columns=["map_cov"])
    model = _StubModel(mean_row=[1.55, 1400.0, 2.15, 625.0, 0.06, np.nan, 45.0],
                       std_row=[0.02, 15.0, 0.05, 5.0, 0.01, np.nan, 1.0],
                       flat=(1.35, 0.02, 0.22, 0.004))
    report = dp.score_panel(model, frame)
    assert report["targets"]["map_cov"]["n"] == 0          # no truth -> no score
    assert report["targets"]["node_peak"]["n"] == 3        # its truth is present


def test_predict_panel_reads_the_serve_columns_and_sigma() -> None:
    frame = _stub_patterns(_panel_frame(3))
    model = _StubModel(mean_row=[1.55, 1400.0, 2.15, 625.0, 0.06, np.nan, 45.0],
                       std_row=[0.02, 15.0, 0.05, 5.0, 0.01, np.nan, 1.0],
                       flat=(1.35, 0.06, 0.22, 0.02))
    pred, sigma, notes = dp.predict_panel(model, frame)
    assert notes == []
    # surrogate column map: f_r 0, cbc_max 1, f_q 2, cyclen 3, ao_abs 4
    assert pred["f_r"] == pytest.approx(np.full(3, 1.55))
    assert pred["cbc_max"] == pytest.approx(np.full(3, 1400.0))
    assert pred["cyclen"] == pytest.approx(np.full(3, 625.0))
    assert sigma["cbc_max"] == pytest.approx(np.full(3, 15.0))
    assert pred["node_peak"] == pytest.approx(np.full(3, 1.35))
    assert sigma["map_cov"] == pytest.approx(np.full(3, 0.02))


def test_batching_does_not_change_the_result() -> None:
    frame = _stub_patterns(_panel_frame(5))
    model = _StubModel(mean_row=[1.55, 1400.0, 2.15, 625.0, 0.06, np.nan, 45.0],
                       std_row=[0.02, 15.0, 0.05, 5.0, 0.01, np.nan, 1.0],
                       flat=(1.35, 0.06, 0.22, 0.02))
    one = dp.score_panel(model, frame, batch_size=1)
    big = dp.score_panel(model, frame, batch_size=512)
    assert one["targets"] == big["targets"]


# --------------------------------------------------------------------------- #
# rendering + artifact
# --------------------------------------------------------------------------- #
def test_format_report_lists_every_target_and_the_notes() -> None:
    frame = _panel_frame()
    report = dp.score_frame(frame, {"cyclen": frame["cyclen"].to_numpy() + 9.0})
    report["notes"] = ["map head failed (RuntimeError: boom)"]
    text = dp.format_report(report)
    for name in dp.PANEL_TARGETS:
        assert name in text
    assert "FAIL" in text and "NO DATA" in text
    assert "[note] map head failed" in text


# --------------------------------------------------------------------------- #
# per-library breakdown (2026-07-29): an aggregate over two regimes is the
# average of a solved problem and an untouched one, and hid the paramA hole.
# --------------------------------------------------------------------------- #
def test_per_library_breakdown_splits_a_hidden_bias() -> None:
    # Reproduces the shape of the real finding: ga80 solved (+1.5), paramA not
    # (+49), aggregate a meaningless +25 that looks like partial progress.
    frame = _panel_frame(4)
    truth = frame["cbc_max"].to_numpy(dtype=float)
    pred = {"cbc_max": truth + np.array([1.5, 1.5, 49.0, 49.0])}
    libs = ["ga80", "ga80", "paramA", "paramA"]
    t = dp.score_frame(frame, pred, libraries=libs)["targets"]["cbc_max"]
    assert t["bias"] == pytest.approx(25.25)                 # the misleading total
    assert set(t["by_library"]) == {"ga80", "paramA"}
    assert t["by_library"]["ga80"]["bias"] == pytest.approx(1.5)
    assert t["by_library"]["ga80"]["n"] == 2
    assert t["by_library"]["paramA"]["bias"] == pytest.approx(49.0)
    assert t["by_library"]["paramA"]["mae"] == pytest.approx(49.0)
    assert t["by_library"]["paramA"]["max_abs_err"] == pytest.approx(49.0)


def test_per_library_breakdown_is_absent_without_libraries() -> None:
    frame = _panel_frame(3)
    t = dp.score_frame(frame, {"cyclen": frame["cyclen"].to_numpy()})["targets"]
    assert t["cyclen"]["by_library"] == {}          # opt-in, never fabricated
    with pytest.raises(ValueError, match="libraries length mismatch"):
        dp.score_target([1.0, 2.0], [1.0, 2.0], libraries=["ga80"])


def test_per_library_breakdown_skips_rows_without_truth() -> None:
    frame = _panel_frame(4)
    frame.loc[[1, 3], "cbc_kind"] = "boc_only"     # censored -> not in any library
    pred = {"cbc_max": frame["cbc_max"].to_numpy(dtype=float) + 10.0}
    t = dp.score_frame(frame, pred,
                       libraries=["ga80", "ga80", "paramA", "paramA"]
                       )["targets"]["cbc_max"]
    assert t["n"] == 2
    assert {k: v["n"] for k, v in t["by_library"].items()} == {"ga80": 1, "paramA": 1}


def test_format_report_renders_library_subrows() -> None:
    frame = _panel_frame(4)
    truth = frame["cbc_max"].to_numpy(dtype=float)
    report = dp.score_frame(frame, {"cbc_max": truth + np.array([1., 1., 49., 49.])},
                            libraries=["ga80", "ga80", "paramA", "paramA"])
    text = dp.format_report(report)
    assert "└ ga80" in text and "└ paramA" in text
    # a single-library panel has nothing to compare, so no sub-rows are emitted
    solo = dp.score_frame(frame, {"cbc_max": truth}, libraries=["ga80"] * 4)
    assert "└" not in dp.format_report(solo)
    assert "└" not in dp.format_report(report, by_library=False)


def test_panel_libraries_prefers_the_models_serve_resolution() -> None:
    frame = _stub_patterns(_panel_frame(3))
    frame["library_id"] = ["stored_a", "stored_b", "stored_c"]

    class _M:
        def serve_library(self, pattern):
            return "resolved"

    # the model's effective-library rerouting wins over the stored column…
    assert dp.panel_libraries(_M(), frame) == ["resolved"] * 3

    # …and a model without the accessor falls back to what the store recorded.
    class _Plain:
        pass

    assert dp.panel_libraries(_Plain(), frame) == ["stored_a", "stored_b", "stored_c"]


def test_score_panel_attaches_the_library_split() -> None:
    frame = _stub_patterns(_panel_frame(3))
    model = _StubModel(mean_row=[1.55, 1400.0, 2.15, 625.0, 0.06, np.nan, 45.0],
                       std_row=[0.02, 15.0, 0.05, 5.0, 0.01, np.nan, 1.0],
                       flat=(1.35, 0.06, 0.22, 0.02))
    report = dp.score_panel(model, frame)
    assert report["libraries"]                       # recorded on the report
    assert report["targets"]["cbc_max"]["by_library"]


def test_run_score_writes_the_artifact_and_never_raises(tmp_path) -> None:
    records = _panel_frame(3)
    records["campaign"] = "democase_smoke"
    out = tmp_path / "panel.json"
    # model=None + an empty panel: no checkpoint is loaded at all.
    report = dp.run_score("data/models/does_not_exist",
                          records=records.iloc[0:0], out_path=out)
    assert report["schema"] == dp.ARTIFACT_SCHEMA
    assert report["report_only"] is True
    assert any("panel is EMPTY" in n for n in report["notes"])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["campaigns"] == list(dp.DEFAULT_PANEL_CAMPAIGNS)
    assert saved["n_panel_rows"] == 0


def test_run_score_with_an_injected_model(tmp_path) -> None:
    frame = _stub_patterns(_panel_frame(3))
    frame["campaign"] = "debug_panel_smoke"
    model = _StubModel(mean_row=[1.55, 1400.0, 2.15, 625.0, 0.06, np.nan, 45.0],
                       std_row=[0.02, 15.0, 0.05, 5.0, 0.01, np.nan, 1.0],
                       flat=(1.35, 0.06, 0.22, 0.02))
    out = tmp_path / "panel.json"
    report = dp.run_score("model_dir_unused", records=frame, model=model,
                          out_path=out)
    assert report["n_panel_rows"] == 3
    assert report["targets"]["cbc_max"]["n"] == 3
    assert json.loads(out.read_text(encoding="utf-8"))["model_dir"] == "model_dir_unused"


# --------------------------------------------------------------------------- #
# CLI: report-only, ALWAYS exit 0
# --------------------------------------------------------------------------- #
def test_cli_exits_zero_even_when_out_of_tolerance(tmp_path, monkeypatch, capsys) -> None:
    from lpopt import cli

    deck = REPO_ROOT / "lpopt.inp"
    if not deck.is_file():
        pytest.skip("reference deck not present")

    def _fake_run_score(model_dir, **kw):
        report = {
            "schema": dp.ARTIFACT_SCHEMA, "n_panel_rows": 7,
            "targets": {n: dp.score_target([1.0], [0.0], [0.1], tolerance=0.5,
                                           unit=dp.TARGET_UNITS.get(n, ""))
                        for n in dp.PANEL_TARGETS},
            "passed": [], "failed": list(dp.PANEL_TARGETS), "unscored": [],
            "tolerances": dict(dp.DEFAULT_TOLERANCES),
        }
        return report

    monkeypatch.setattr(dp, "run_score", _fake_run_score)
    rc = cli.main(["debug-panel", "score", "--input", str(deck),
                   "--model-dir", "data/models/20260729_054749",
                   "--out", str(tmp_path / "p.json")])
    assert rc == 0                                   # NEVER blocks
    assert "OUT OF TOLERANCE" in capsys.readouterr().out


def test_cli_exits_zero_when_the_scorer_explodes(tmp_path, monkeypatch, capsys) -> None:
    from lpopt import cli

    deck = REPO_ROOT / "lpopt.inp"
    if not deck.is_file():
        pytest.skip("reference deck not present")

    def _boom(model_dir, **kw):
        raise FileNotFoundError("no member_* checkpoints")

    monkeypatch.setattr(dp, "run_score", _boom)
    rc = cli.main(["debug-panel", "score", "--input", str(deck),
                   "--model-dir", str(tmp_path / "nope")])
    assert rc == 0
    assert "debug-panel score failed" in capsys.readouterr().out


def test_deck_accepts_a_debug_panel_table(tmp_path) -> None:
    from lpopt.config import load_config

    deck = tmp_path / "d.inp"
    deck.write_text(
        "[debug_panel]\n"
        'campaigns = ["debug_panel*", "democase*", "mycal_*"]\n'
        "tolerances = { cbc_max = 15.0, cyclen = 2.5 }\n",
        encoding="utf-8")
    cfg = load_config(deck)
    assert cfg.debug_panel.campaigns[-1] == "mycal_*"
    assert cfg.debug_panel.tolerances == {"cbc_max": 15.0, "cyclen": 2.5}
    # an absent [debug_panel] table still yields the documented defaults
    bare = tmp_path / "bare.inp"
    bare.write_text("[flow]\ntitle = 'x'\n", encoding="utf-8")
    assert load_config(bare).debug_panel.campaigns == list(dp.DEFAULT_PANEL_CAMPAIGNS)
