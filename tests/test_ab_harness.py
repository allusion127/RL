"""The A/B judging harness: fold definitions, metrics, arm identification.

These pin the things a scoring bug would silently corrupt:

* **fold C really is "produced after the split"** -- a set difference against a
  frozen manifest, with no overlap against train or val.  If this drifts, every
  "uncontaminated" number in every report is wrong.
* **Delta75 measures what it claims** -- verified against synthetic predictors
  whose true resolution is known by construction, not against a stored constant.
* **The arm identifier cannot mistake a historical run for a campaign arm** --
  the failure that would silently overwrite an arm's row with a 2026-07-21 model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.model import ab_eval as M
from lpopt.model import ab_watch as W
from lpopt.model.ab_score import render_markdown, update_results
from lpopt.model.folds import (
    FOLD_NAMES, assign_folds, cell_key, fold_a_ids, fold_frame, proposal_mask,
    summarize_folds,
)
from lpopt.model.splits import SplitManifest


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _frame(n=40, start=0):
    return pd.DataFrame({
        "record_id": [f"r{i:03d}" for i in range(start, start + n)],
        "dataset": ["A"] * n,
        "campaign": [None] * n,
        "generator": ["prod"] * n,
        "feed": [121] * n,
        "e_core": [5.26] * n,
        "converged": [True] * n,
        "maps_key": [None] * n,
        "cyclen": np.linspace(600, 700, n),
        "f_r": np.linspace(1.5, 1.9, n),
    })


def _manifest(train_ids, val_ids, curr_val=()):
    return SplitManifest(
        name="T", kind="curriculum_group", seed=0,
        train_ids=list(train_ids), val_ids=list(val_ids),
        groups={"curriculum_val_by_cell": {"c0": list(curr_val)}})


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def test_fold_c_is_exactly_what_the_frozen_manifest_never_saw():
    df = _frame(40)
    ids = df["record_id"].tolist()
    man = _manifest(ids[:20], ids[20:30], curr_val=ids[20:25])
    lab = assign_folds(df, man)
    assert set(lab[:20]) == {"train"}
    assert set(lab[20:25]) == {"A"}
    assert set(lab[25:30]) == {"B"}
    # rows 30..39 are in NEITHER frozen list -> produced after the split
    assert set(lab[30:]) == {"C"}
    fc = set(df["record_id"][lab == "C"])
    assert not (fc & set(man.train_ids)), "fold C must not touch train"
    assert not (fc & set(man.val_ids)), "fold C must not touch val"


def test_folds_partition_without_overlap_or_silent_merge():
    df = _frame(40)
    ids = df["record_id"].tolist()
    man = _manifest(ids[:20], ids[20:30], curr_val=ids[20:25])
    lab = assign_folds(df, man)
    counts = lab.value_counts().to_dict()
    assert sum(counts.values()) == len(df)
    assert set(counts) <= {"A", "B", "C", "train", ""}


def test_non_legacy_val_rows_are_dropped_not_folded_into_b():
    df = _frame(10)
    df.loc[5:, "feed"] = 133                # not the legacy regime
    ids = df["record_id"].tolist()
    man = _manifest(ids[:2], ids[2:], curr_val=())
    lab = assign_folds(df, man)
    assert set(lab[2:5]) == {"B"}
    assert set(lab[5:]) == {""}, "a non-legacy val row is unlabelled, never fold B"


def test_fold_a_ids_unions_every_cell():
    man = _manifest([], [], curr_val=[])
    man.groups["curriculum_val_by_cell"] = {"c0": ["a", "b"], "c1": ["b", "c"]}
    assert fold_a_ids(man) == {"a", "b", "c"}


def test_cell_key_prefers_campaign_then_bins():
    df = _frame(4)
    df.loc[0, "campaign"] = "alsearch_K1_K2_f121"
    df.loc[1, "e_core"] = 5.24
    keys = cell_key(df).tolist()
    assert keys[0] == "alsearch_K1_K2_f121"
    assert keys[1] == "ebin5.20_f121"
    assert keys[2] == "ebin5.25_f121"


def test_cell_key_is_blank_when_unbinnable():
    df = _frame(2)
    df.loc[0, "e_core"] = np.nan
    assert cell_key(df).tolist()[0] == ""


def test_proposal_mask_flags_model_proposed_rows():
    df = _frame(3)
    df.loc[0, "generator"] = "alsearch_v5"
    df.loc[1, "campaign"] = "alsearch_G3_G4"
    m = proposal_mask(df)
    assert m.tolist() == [True, True, False]


def test_fold_frame_drops_unconverged():
    df = _frame(20)
    df.loc[10:, "converged"] = False
    ids = df["record_id"].tolist()
    man = _manifest(ids[:5], [], curr_val=())
    ff = fold_frame(df, man, "C")
    assert len(ff) == 5                       # rows 5..9 only
    assert ff.df["converged"].all()


def test_summarize_folds_marks_only_c_uncontaminated():
    df = _frame(30)
    ids = df["record_id"].tolist()
    s = summarize_folds(df, _manifest(ids[:10], ids[10:20], curr_val=ids[10:15]))
    assert s["C"]["uncontaminated"] is True
    assert s["A"]["uncontaminated"] is False and s["B"]["uncontaminated"] is False
    assert s["n_store_rows"] == 30


# --------------------------------------------------------------------------- #
# metrics: effective resolution
# --------------------------------------------------------------------------- #
def test_spearman_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    assert M.spearman(a, b) == pytest.approx(float(sp.spearmanr(a, b)[0]), abs=1e-9)


def test_perfect_predictor_resolves_the_smallest_bin():
    rng = np.random.default_rng(1)
    true = rng.normal(scale=10.0, size=300)
    cells = np.array(["c"] * 300)
    r = M.effective_resolution(true.copy(), true, cells, "cyclen")
    assert r["delta75"] == 0.0, "a perfect predictor resolves from the first bin"
    assert r["lowest_bin_hit"] == pytest.approx(1.0)


def test_pure_noise_predictor_never_resolves():
    rng = np.random.default_rng(2)
    true = rng.normal(scale=10.0, size=300)
    noise = rng.normal(size=300)
    r = M.effective_resolution(noise, true, np.array(["c"] * 300), "cyclen")
    assert r["delta75"] is None
    assert r["lowest_bin_hit"] == pytest.approx(0.5, abs=0.08)


def test_delta75_rises_as_the_predictor_gets_noisier():
    """The metric must ORDER models by resolution, which is its entire job."""
    rng = np.random.default_rng(3)
    true = rng.normal(scale=10.0, size=400)
    cells = np.array(["c"] * 400)
    prev = -1.0
    seen = []
    for sigma in (0.5, 3.0, 8.0):
        pred = true + rng.normal(scale=sigma, size=400)
        d = M.effective_resolution(pred, true, cells, "cyclen")["delta75"]
        seen.append(d if d is not None else float("inf"))
    assert seen[0] <= seen[1] <= seen[2]
    assert seen[0] < seen[2], "a much noisier model must need a bigger gap"


def test_delta75_is_the_lower_edge_of_the_first_qualifying_bin():
    curve = M.resolution_curve(
        np.arange(200.0), np.arange(200.0), np.array(["c"] * 200),
        M.DELTA_BINS["cyclen"])
    first = next(b for b in curve["curve"]
                 if b["n"] >= 30 and b["hit"] >= M.RESOLUTION_LEVEL)
    assert curve["delta75"] == first["lo"]


def test_resolution_uses_only_within_cell_pairs():
    """Between-cell ordering is free skill; it must not inflate the metric."""
    true = np.concatenate([np.zeros(30), np.ones(30) * 100.0])
    pred = true.copy()                       # perfect BETWEEN cells...
    rng = np.random.default_rng(4)
    true = true + rng.normal(scale=1.0, size=60)   # ... random WITHIN
    cells = np.array(["a"] * 30 + ["b"] * 30)
    r = M.effective_resolution(pred, true, cells, "cyclen")
    assert r["lowest_bin_hit"] == pytest.approx(0.5, abs=0.15)


def test_within_cell_sd_is_the_median_over_cells():
    true = np.concatenate([np.zeros(10), np.arange(10.0)])
    cells = np.array(["a"] * 10 + ["b"] * 10)
    # cell "a" has zero spread and is excluded; only "b" contributes
    assert M.within_cell_sd(true, cells) == pytest.approx(np.std(np.arange(10.0)))


def test_cells_below_the_minimum_are_ignored():
    true = np.arange(20.0)
    cells = np.array(["a"] * 4 + ["b"] * 16)
    groups = M._cell_groups(cells)
    assert len(groups) == 1 and len(groups[0]) == 16


# --------------------------------------------------------------------------- #
# metrics: within-cell stats, SD ratio, spectrum, gate
# --------------------------------------------------------------------------- #
def test_sd_ratio_detects_shrinkage_toward_the_cell_mean():
    rng = np.random.default_rng(5)
    true = rng.normal(size=200)
    cells = np.array(["c"] * 200)
    shrunk = true * 0.5                      # the compression arm A2 targets
    assert M.within_cell_stats(shrunk, true, cells, bootstrap=0)["sd_ratio"] == \
        pytest.approx(0.5, abs=0.02)
    assert M.within_cell_stats(true, true, cells, bootstrap=0)["sd_ratio"] == \
        pytest.approx(1.0, abs=1e-9)


def test_within_cell_stats_survives_an_all_nan_target():
    true = np.full(20, np.nan)
    stats = M.within_cell_stats(np.arange(20.0), true, np.array(["c"] * 20),
                                bootstrap=0)
    assert stats["n_cells"] == 0 and np.isnan(stats["median_rho"])


def test_cluster_bootstrap_ci_brackets_the_median():
    vals = list(np.linspace(0.5, 0.9, 25))
    lo, hi = M.cluster_bootstrap_ci(vals, reps=500)
    assert lo <= float(np.median(vals)) <= hi


def test_cluster_bootstrap_ci_opts_out_cleanly():
    assert all(np.isnan(v) for v in M.cluster_bootstrap_ci([0.1, 0.2], reps=0))


def test_map_spectrum_is_perfect_on_an_exact_copy():
    rng = np.random.default_rng(6)
    t = rng.normal(size=(12, 9, 9))
    bands = M.map_spectrum(t.copy(), t)
    assert len(bands) == len(M.SPECTRAL_BANDS)
    for b in bands:
        assert b["power_ratio"] == pytest.approx(1.0, rel=1e-6)
        assert b["mode_rho"] == pytest.approx(1.0, abs=1e-6)


def test_map_spectrum_sees_high_wavenumber_attenuation():
    """A low-pass-filtered prediction must show a falling power ratio -- the
    exact signature arms A1/A3 claim to reduce."""
    rng = np.random.default_rng(7)
    t = rng.normal(size=(64, 9, 9))
    ft = np.fft.fft2(t, axes=(-2, -1))
    k = np.fft.fftfreq(9)
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    p = np.real(np.fft.ifft2(ft * np.exp(-6.0 * kr ** 2), axes=(-2, -1)))
    bands = M.map_spectrum(p, t)
    ratios = [b["power_ratio"] for b in bands]
    assert ratios[0] > ratios[-1], "low-pass must attenuate the top band most"
    assert ratios[-1] < 0.9


def test_map_spectrum_ignores_a_constant_offset():
    rng = np.random.default_rng(8)
    t = rng.normal(size=(8, 9, 9))
    for b in M.map_spectrum(t + 5.0, t):
        assert b["power_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_gate_fails_on_a_single_collapsed_cell():
    rng = np.random.default_rng(9)
    n = 60
    cells = np.array(["good"] * 30 + ["bad"] * 30)
    truth = {"cyclen": rng.normal(size=n)}
    old = {"cyclen": truth["cyclen"] + rng.normal(scale=0.1, size=n)}
    new = {"cyclen": old["cyclen"].copy()}
    new["cyclen"][30:] = rng.normal(size=30)          # one cell destroyed
    g = M.no_regression_gate(new, old, truth, cells, targets=("cyclen",),
                             min_rows=8)
    assert g["pass"] is False
    assert g["checks"][0]["cell"] == "bad"


def test_gate_passes_when_nothing_regresses():
    rng = np.random.default_rng(10)
    cells = np.array(["a"] * 40)
    truth = {"cyclen": rng.normal(size=40)}
    old = {"cyclen": truth["cyclen"] + rng.normal(scale=0.3, size=40)}
    g = M.no_regression_gate({"cyclen": truth["cyclen"]}, old, truth, cells,
                             targets=("cyclen",), min_rows=8)
    assert g["pass"] is True and g["worst_drop"] <= 0.0


# --------------------------------------------------------------------------- #
# arm identification
# --------------------------------------------------------------------------- #
_COMMON = ("--ensemble 5 --split S1 --epochs 150 --num-workers 8 --device auto "
           "--parallel-members 5 --cyclen-physics-prior --quantile-heads "
           "--quantile-weight 0.2 --promote-max-asm-bu "
           "--distill-targets data/models/_v5_distill_soft.npz "
           "--distill-weight 0.4 --distill-min-match-frac 0.5 "
           "--f-r-rank-weight 0.1 --cyclen-rank-weight 0.25 --map-peak-weight 2.0")
_TAILS = {
    "B1": "--cond-schema v5 --width 160 --n-blocks 6",
    "A1": "--cond-schema v5 --width 160 --n-blocks 6 --map-decoder multiscale",
    "A2": "--cond-schema v6_prior --width 160 --n-blocks 6 --map-prior-residual",
    "A3": "--cond-schema v5 --width 160 --n-blocks 6 --map-spectral-weight 0.3",
    "A4": "--cond-schema v6_contrast --width 160 --n-blocks 6",
    "A5": "--cond-schema v5 --width 256 --n-blocks 10 --head-hidden 512",
    "A6": ("--cond-schema v6 --width 224 --n-blocks 8 --head-hidden 384 "
           "--map-decoder multiscale --map-prior-residual "
           "--map-spectral-weight 0.3"),
}


def _run_sh(tail: str, extra: str = "") -> str:
    return (
        '#!/bin/bash\nset -o pipefail\nRUN="$HOME/lpopt_ws/runs/X"\n'
        f'CUDA_VISIBLE_DEVICES=0 $HOME/lpopt_ws/venv/bin/python -m lpopt.model.train '
        f'{_COMMON} {tail} {extra} --out-dir "runs/X" > "$RUN/train.log" 2>&1\n'
        'RC=$?\n')


@pytest.mark.parametrize("label", sorted(_TAILS))
def test_every_arm_is_identified_from_its_own_run_sh(label):
    assert W.identify_arm(_run_sh(_TAILS[label]), ts="20260725_120000") == label


def test_arm_signatures_cover_the_launcher_table():
    assert set(W.ARM_SIGNATURES) == set(_TAILS)


def test_arms_are_mutually_exclusive():
    for label, tail in _TAILS.items():
        matches = [lab for lab in W.ARM_SIGNATURES
                   if W.identify_arm(_run_sh(tail), ts="20260725_120000") == lab]
        assert matches == [label], f"{label} is ambiguous: {matches}"


def test_frozen_trunk_run_is_not_mistaken_for_an_arm():
    """20260724_213535's shape matches B1; its freeze flag must exclude it."""
    text = _run_sh(_TAILS["B1"],
                   "--init-from data/models/20260721_105824 --freeze-trunk-cyclen")
    assert W.identify_arm(text, ts="20260725_120000") is None


def test_pre_campaign_timestamp_is_rejected():
    assert W.identify_arm(_run_sh(_TAILS["B1"]), ts="20260721_105824") is None
    assert W.identify_arm(_run_sh(_TAILS["B1"]), ts="20260725_060441") == "B1"


def test_run_without_the_common_recipe_is_not_an_arm():
    """20260721_105824 is v5/160/6 too, but predates --map-peak-weight."""
    lean = ("CUDA_VISIBLE_DEVICES=0 python -m lpopt.model.train "
            "--ensemble 5 --split S1 --cond-schema v5 --width 160 --n-blocks 6 "
            '--out-dir "runs/X" > "$RUN/train.log" 2>&1')
    assert W.parse_train_args(lean)["map_peak_weight"] == 0.0
    assert W.identify_arm(lean, ts="20260725_120000") is None


def test_parse_train_args_reads_every_hires_flag():
    got = W.parse_train_args(_run_sh(_TAILS["A6"]))
    assert got["cond_schema"] == "v6" and got["width"] == 224
    assert got["n_blocks"] == 8 and got["head_hidden"] == 384
    assert got["map_decoder"] == "multiscale"
    assert got["map_prior_residual"] is True
    assert got["map_spectral_weight"] == 0.3
    assert got["map_peak_weight"] == 2.0


# --------------------------------------------------------------------------- #
# results accumulation + report
# --------------------------------------------------------------------------- #
def _entry(label, d75=1.0, rho=0.7):
    return {
        "label": label, "model_dir": f"data/models/{label}",
        "n_params_total": 3_159_291,
        "folds": {"C": {
            "name": "new_unseen", "n": 100, "n_cells": 10,
            "resolution": {t: {"delta75_over_sd": d75} for t in M.PRIMARY_TARGETS},
            "accuracy": {t: {"median_rho": rho, "sd_ratio": 0.5}
                         for t in M.PRIMARY_TARGETS},
            "map_spectrum": [{"band": [0.47, 1.0], "power_ratio": 0.8,
                              "mode_rho": 0.6}],
            "gate": {"pass": True, "worst_drop": 0.01},
        }},
    }


def test_update_results_accumulates_and_replaces(tmp_path):
    p = tmp_path / "res.json"
    update_results(_entry("B1"), p)
    doc = update_results(_entry("A2"), p)
    assert set(doc["arms"]) == {"B1", "A2"}
    doc = update_results(_entry("B1", d75=0.5), p)
    assert set(doc["arms"]) == {"B1", "A2"}, "re-scoring must replace, not append"
    assert doc["arms"]["B1"]["folds"]["C"]["resolution"]["f_r"]["delta75_over_sd"] == 0.5


def test_update_results_survives_a_corrupt_file(tmp_path):
    p = tmp_path / "res.json"
    p.write_text("{not json", encoding="utf-8")
    assert set(update_results(_entry("B1"), p)["arms"]) == {"B1"}


def test_markdown_lists_arms_in_design_order():
    doc = {"updated_at": "now",
           "arms": {"A5": _entry("A5"), "B1": _entry("B1"), "A2": _entry("A2")}}
    md = render_markdown(doc)
    assert md.index("| B1 ") < md.index("| A2 ") < md.index("| A5 ")
    assert "Δ₇₅/셀내SD" in md and "PASS" in md


def test_markdown_renders_missing_metrics_as_dashes():
    doc = {"updated_at": "now", "arms": {"B1": {"label": "B1", "folds": {}}}}
    assert "—" in render_markdown(doc)


# --------------------------------------------------------------------------- #
# the pre-registered decision rule
# --------------------------------------------------------------------------- #
from lpopt.model import ab_decide as D          # noqa: E402


def _paired(point, half=0.05, control=D.CONTROL, method="bca"):
    """A paired-CI record shaped like ``flat_ab.paired_block()`` output."""
    return {"control": control, "verdict": "promote", "metrics": {
        D.PAIRED_PRIMARY: {"metric": D.PAIRED_PRIMARY, "point": point,
                           "ci_lo": None if point is None else point - half,
                           "ci_hi": None if point is None else point + half,
                           "method": method, "n_cells": 29}}}


def _arm(label, res: dict, rho=0.75, gate_pass=True, schema="v5", d=None,
         paired="auto"):
    """One scored arm. ``res`` maps target -> Delta75/SD (lower == finer).

    ``paired`` defaults to interval evidence derived from the arm's node_peak
    resolution so the fixtures exercise a coherent arm (a finer arm also has a
    real paired gain).  Pass an explicit dict to decouple the two -- that is how
    the "point favours A, interval does not" case is built -- or ``None`` to
    model an arm that was never put through the paired apparatus.
    """
    if paired == "auto":
        np_res = res.get("node_peak")
        paired = None if np_res is None else _paired(round(_FLAT_BASE - np_res, 6))
    entry = {
        "label": label, "model_dir": d or f"data/models/{label}",
        "cond_schema": schema,
        "folds": {"C": {
            "resolution": {t: {"delta75_over_sd": v} for t, v in res.items()},
            "accuracy": {t: {"median_rho": rho} for t in M.PRIMARY_TARGETS},
            "gate": {"pass": gate_pass, "worst_drop": 0.0},
        }},
    }
    if paired is not None:
        entry["paired"] = {"C": paired}
    return entry


#: node_peak Delta75/SD of the control fixture below.
_FLAT_BASE = 1.40


_FLAT = {"node_peak": 1.40, "map_cov": 1.00, "f_r": 0.50, "cyclen": 0.88}


def _doc(*arms):
    return {"arms": {a["label"]: a for a in arms}}


def _full(extra):
    """A complete arm set: control + all six arms, with `extra` overriding."""
    arms = [_arm("B0", _FLAT), _arm("B1", _FLAT)]
    for lab in ("A1", "A2", "A3", "A4", "A5", "A6"):
        arms.append(extra.get(lab) or _arm(lab, _FLAT))
    return _doc(*arms)


def test_blocked_without_the_control():
    v = D.evaluate_arms(_doc(_arm("A2", _FLAT)))
    assert v["verdict"] == "blocked" and "B1" in v["reason"]


def test_incomplete_reports_the_leader_without_deciding():
    v = D.evaluate_arms(_doc(_arm("B1", _FLAT), _arm("B0", _FLAT),
                             _arm("A2", {**_FLAT, "node_peak": 1.0})))
    assert v["verdict"] == "incomplete"
    assert v["leader"] == "A2"
    assert set(v["missing"]) == {"A1", "A3", "A4", "A5", "A6"}


def test_winner_needs_the_full_primary_improvement():
    """0.10 of improvement is real but below the pre-registered 0.15 bar."""
    v = _full({"A2": _arm("A2", {**_FLAT, "node_peak": 1.30})})
    assert D.evaluate_arms(v)["verdict"] == "no_winner"
    v = _full({"A2": _arm("A2", {**_FLAT, "node_peak": 1.20})})
    out = D.evaluate_arms(v)
    assert out["verdict"] == "winner" and out["winner"] == "A2"


def test_a_regression_elsewhere_disqualifies():
    """Improving node_peak while losing cyclen is not a win."""
    arm = _arm("A2", {"node_peak": 1.0, "map_cov": 1.0, "f_r": 0.50,
                      "cyclen": 0.99})          # cyclen worse by 0.11 > 0.05
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_primary"] is False
    assert out["verdict"] == "no_winner"


def test_gate_failure_disqualifies_even_with_the_best_metric():
    arm = _arm("A2", {**_FLAT, "node_peak": 0.9}, gate_pass=False)
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_primary"] is True
    assert out["verdicts"]["A2"]["eligible"] is False
    assert out["verdict"] == "no_winner"


def test_secondary_rho_drop_disqualifies():
    arm = _arm("A2", {**_FLAT, "node_peak": 0.9}, rho=0.60)   # B0 is 0.75
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_secondary"] is False
    assert out["verdict"] == "no_winner"


def test_near_tie_escalates_instead_of_guessing():
    doc = _full({"A2": _arm("A2", {**_FLAT, "node_peak": 1.00}),
                 "A6": _arm("A6", {**_FLAT, "node_peak": 1.02})})
    out = D.evaluate_arms(doc)
    assert out["verdict"] == "escalate"
    assert set(out["candidates"]) == {"A2", "A6"}


def test_clear_margin_picks_the_winner():
    doc = _full({"A2": _arm("A2", {**_FLAT, "node_peak": 0.80}),
                 "A6": _arm("A6", {**_FLAT, "node_peak": 1.10})})
    out = D.evaluate_arms(doc)
    assert out["verdict"] == "winner" and out["winner"] == "A2"


def test_capacity_clause_is_reported_either_way():
    lose = D.evaluate_arms(_full({}))["capacity_clause"]
    assert lose["null_rejected"] is False and "무효" in lose["conclusion"]
    win = D.evaluate_arms(
        _full({"A5": _arm("A5", {**_FLAT, "node_peak": 1.0})}))["capacity_clause"]
    assert win["null_rejected"] is True and "기각" in win["conclusion"]


def test_v6_winner_raises_the_serving_blocker():
    doc = _full({"A2": _arm("A2", {**_FLAT, "node_peak": 0.8}, schema="v6_prior")})
    out = D.evaluate_arms(doc)
    assert out["winner"] == "A2" and "power_prior" in out["serving_blocker"]
    doc = _full({"A1": _arm("A1", {**_FLAT, "node_peak": 0.8}, schema="v5")})
    assert "serving_blocker" not in D.evaluate_arms(doc)


def test_unreached_delta75_does_not_silently_count_as_improvement():
    arm = _arm("A2", {**_FLAT, "node_peak": None})
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_primary"] is False
    assert any("Delta75" in n for n in out["verdicts"]["A2"]["notes"])


# --------------------------------------------------------------------------- #
# interval evidence is mandatory (program sections 8.3 / 8.5)
# --------------------------------------------------------------------------- #
def test_a_point_lead_without_an_interval_cannot_promote():
    """Best Delta75/SD on the slate, no paired record -> not eligible."""
    arm = _arm("A2", {**_FLAT, "node_peak": 0.80}, paired=None)
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_primary"] is True
    assert out["verdicts"]["A2"]["passes_paired"] is False
    assert out["verdicts"]["A2"]["eligible"] is False
    assert out["verdict"] != "winner"
    assert any("paired" in n for n in out["verdicts"]["A2"]["notes"])


def test_point_estimate_favours_the_arm_but_the_interval_does_not():
    """The core refusal: +0.60 of Delta75/SD, paired CI straddling the null."""
    arm = _arm("A2", {**_FLAT, "node_peak": 0.80},
               paired=_paired(0.02, half=0.05))          # CI [-0.03, +0.07]
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_primary"] is True
    assert out["verdicts"]["A2"]["passes_paired"] is False
    assert out["verdict"] == "escalate", "an underpowered lead is escalated"
    assert out["candidates"] == ["A2"]
    assert "점추정" in out["reason"] and "CI" in out["reason"]


def test_a_slate_with_no_paired_evidence_at_all_is_blocked_not_no_winner():
    doc = _doc(*[_arm(lab, _FLAT, paired=None)
                 for lab in ("B0", "B1", "A1", "A2", "A3", "A4", "A5", "A6")])
    out = D.evaluate_arms(doc)
    assert out["verdict"] == "blocked"
    assert "paired" in out["reason"]


def test_paired_evidence_against_the_wrong_control_is_refused():
    """Section 8.4: only ``arm - control`` attributes the loss change."""
    arm = _arm("A2", {**_FLAT, "node_peak": 0.80},
               paired=_paired(0.60, control="B0"))
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_paired"] is False
    assert out["verdict"] != "winner"
    assert any("B0" in n for n in out["verdicts"]["A2"]["notes"])


def test_too_few_paired_cells_establishes_nothing():
    arm = _arm("A2", {**_FLAT, "node_peak": 0.80},
               paired=_paired(0.60, method="insufficient"))
    out = D.evaluate_arms(_full({"A2": arm}))
    assert out["verdicts"]["A2"]["passes_paired"] is False
    assert out["verdict"] != "winner"


def test_a_degenerate_paired_record_cannot_promote():
    """THE bug: a point-mass resample reads as the most decisive CI on the slate.

    ``paired_cell_bootstrap`` returns ``ci_lo == ci_hi == point`` with ``se = 0``
    when every paired cell moved by exactly the same amount, and the promotion
    test is ``ci_lo > 0`` -- so a resample that saw ONE distinct value cleared it
    outright, and nothing downstream distinguished that from a genuine
    zero-width win.
    """
    arm = _arm("A2", {**_FLAT, "node_peak": 0.80},
               paired=_paired(0.60, half=0.0, method="degenerate"))
    out = D.evaluate_arms(_full({"A2": arm}))
    v = out["verdicts"]["A2"]
    assert v["paired_lo"] == v["paired_hi"] == pytest.approx(0.60)  # the trap …
    assert v["passes_primary"] is True                             # … and a strong arm
    assert v["passes_paired"] is False, "a point mass cannot promote"
    assert v["eligible"] is False
    assert out["verdict"] != "winner"
    assert any("point mass" in n for n in v["notes"])
    # the finding is 'nothing was measured', so it escalates rather than being
    # recorded as 'no effect'.
    assert out["verdict"] == "escalate" and out["candidates"] == ["A2"]


def test_overlapping_paired_intervals_are_a_tie_even_on_different_points():
    doc = _full({"A2": _arm("A2", {**_FLAT, "node_peak": 0.80},
                            paired=_paired(0.60, half=0.30)),
                 "A6": _arm("A6", {**_FLAT, "node_peak": 1.10},
                            paired=_paired(0.30, half=0.20))})
    out = D.evaluate_arms(doc)
    assert out["verdict"] == "escalate"
    assert set(out["candidates"]) == {"A2", "A6"}


def test_promotion_needs_both_pre_registered_rules_to_agree():
    doc = _full({"A2": _arm("A2", {**_FLAT, "node_peak": 0.80})})
    v = D.evaluate_arms(doc)
    assert v["verdict"] == "winner"
    ok, _ = D.promotion_allowed(v, doc)
    assert ok is True
    doc_hold = dict(doc, flat_slate={"verdict": "hold", "reason": "M0 unresolved"})
    ok, why = D.promotion_allowed(v, doc_hold)
    assert ok is False and "hold" in why
    doc_other = dict(doc, flat_slate={"verdict": "promote", "winner": "A6"})
    ok, why = D.promotion_allowed(v, doc_other)
    assert ok is False and "A6" in why


def test_render_verdict_is_printable_for_every_outcome():
    for doc in (_doc(_arm("A2", _FLAT)), _full({}),
                _full({"A2": _arm("A2", {**_FLAT, "node_peak": 0.8})})):
        text = D.render_verdict(D.evaluate_arms(doc))
        assert "verdict:" in text


# --------------------------------------------------------------------------- #
# curriculum threads the hires structure to BOTH retrain paths
# --------------------------------------------------------------------------- #
def _driver_stub(**model_over):
    """A stand-in exposing just what the two builders read."""
    import lpopt.curriculum as C
    from lpopt.config import load_config

    cfg = load_config("lpopt.inp")
    for k, v in model_over.items():
        setattr(cfg.model, k, v)

    class D:
        pass
    d = D()
    d.cfg, d.curr = cfg, cfg.curriculum
    d._v5_train_config = C.CurriculumDriver._v5_train_config.__get__(d)
    d._v5_train_flags = C.CurriculumDriver._v5_train_flags.__get__(d)
    return d


def test_curriculum_local_and_remote_agree_on_the_map_structure():
    """The two retrain paths must build the SAME network, or a local retrain and
    a remote retrain of the same deck silently differ."""
    d = _driver_stub(map_head_mode="multiscale", map_prior_residual=True,
                     map_spectral_weight=0.3)
    tc = d._v5_train_config()
    flags = " ".join(d._v5_train_flags())
    assert (tc.map_head_mode, tc.map_prior_residual, tc.map_spectral_weight) == \
        ("multiscale", True, 0.3)
    assert "--map-decoder multiscale" in flags
    assert "--map-prior-residual" in flags
    assert "--map-spectral-weight 0.3" in flags


def test_curriculum_defaults_emit_no_hires_flags():
    d = _driver_stub(map_head_mode="linear", map_prior_residual=False,
                     map_spectral_weight=0.0)
    tc = d._v5_train_config()
    assert (tc.map_head_mode, tc.map_prior_residual, tc.map_spectral_weight) == \
        ("linear", False, 0.0)
    assert "--map-" not in " ".join(d._v5_train_flags())


def test_live_deck_pairs_v6_with_a_usable_map_head():
    """The deck must never ship a v6 schema behind the linear map head: the extra
    channels would be paid for and unreadable (worse than staying on v5)."""
    from lpopt.config import load_config

    m = load_config("lpopt.inp").model
    if str(m.cond_schema).startswith("v6"):
        assert m.map_head_mode == "multiscale" or m.map_prior_residual, (
            "cond_schema v6 requires a map path that can read the new channels")


def test_no_regression_gate_scores_f_r_without_letting_it_veto():
    """LOW (4): ``no_regression_gate``'s default family put ``f_r`` on equal
    footing with ``cyclen``, so the proxy gate withheld promotion on exactly the
    axis the user deferred.  Scored, reported, no veto — under the same switch.
    """
    import numpy as np
    from lpopt.model import ab_eval as M

    cells = np.array(["c0"] * 40 + ["c1"] * 40)
    truth = {"cyclen": np.arange(80.0), "f_r": np.arange(80.0)}
    old = {"cyclen": np.arange(80.0), "f_r": np.arange(80.0)}
    new = {"cyclen": np.arange(80.0), "f_r": -np.arange(80.0)}   # f_r rank inverted

    g = M.no_regression_gate(new, old, truth, cells)
    fr = [c for c in g["checks"] if c["target"] == "f_r"]
    assert fr and fr[0]["drop"] > 1.5            # SCORED, and a total collapse
    assert all(c["enforced"] is False for c in fr)
    assert g["pass"] is True                     # …and it cannot veto
    assert g["worst_drop"] <= g["epsilon"]
    assert g["worst_drop_any_axis"] > 1.5
    assert g["guarded_targets"] == ["cyclen"]
    assert g["report_only_targets"] == ["f_r"]
    assert "REPORT-ONLY" in g["note"] and "f_r" in g["note"]

    on = M.no_regression_gate(new, old, truth, cells, fr_guarded=True)
    assert on["pass"] is False
    assert on["guarded_targets"] == ["cyclen", "f_r"]
    assert all(c["enforced"] is True for c in on["checks"] if c["target"] == "f_r")


def test_no_regression_gate_still_fails_on_a_cyclen_collapse():
    import numpy as np
    from lpopt.model import ab_eval as M

    cells = np.array(["c0"] * 40 + ["c1"] * 40)
    truth = {"cyclen": np.arange(80.0), "f_r": np.arange(80.0)}
    old = {"cyclen": np.arange(80.0), "f_r": np.arange(80.0)}
    new = {"cyclen": -np.arange(80.0), "f_r": np.arange(80.0)}
    assert M.no_regression_gate(new, old, truth, cells)["pass"] is False


# --------------------------------------------------------------------------- #
# F_r promotion-gate deferral: ab_decide's TWO point vetoes (HIGH-1)
#
# ``evaluate_arms`` withholds promotion on an F_r drop in two independent places,
# and both were hardcoded with no path for the one switch to reach them:
#
#   (1) the PRIMARY rule's regression veto ranges over
#       ``ab_eval.PRIMARY_TARGETS``, which contains ``f_r`` -- so an F_r
#       Delta75/SD regression fails ``passes_primary``;
#   (2) the SECONDARY rule ranges over the literal ``("cyclen", "f_r")`` -- so an
#       F_r within-cell rho drop fails ``passes_secondary``.
#
# Both must resolve through ``config.fr_guard_enforced`` like every other
# promotion surface: report-only by default, enforcing when the switch is set.
# --------------------------------------------------------------------------- #
def _arm_per_target_rho(label, res: dict, rho_by_target: dict, gate_pass=True):
    """Like ``_arm`` but with a DIFFERENT median_rho per target, so an f_r-only
    rho drop can be built without also dropping cyclen."""
    a = _arm(label, res, gate_pass=gate_pass)
    a["folds"]["C"]["accuracy"] = {
        t: {"median_rho": rho_by_target.get(t, 0.75)} for t in M.PRIMARY_TARGETS}
    return a


def test_an_f_r_only_delta75_regression_does_not_veto_by_default():
    """Veto (1): the primary rule's ``worst_regression`` must not count ``f_r``
    while the guard is deferred, and must count it again when it is armed."""
    arm = _arm("A2", {"node_peak": 0.90, "map_cov": 1.00,
                      "f_r": 0.61,          # worse than the control's 0.50 by 0.11
                      "cyclen": 0.88})
    doc = _full({"A2": arm})

    out = D.evaluate_arms(doc)
    v = out["verdicts"]["A2"]
    assert v["passes_primary"] is True, "f_r regression vetoed a promotion"
    assert out["verdict"] == "winner" and out["winner"] == "A2"
    # scored and named, never silently dropped
    assert any("f_r" in n for n in v["notes"])
    assert out["fr_guard"]["enforced"] is False
    assert out["fr_guard"]["knob"] == "[curriculum] gate_noreg_fr_guard_enabled"
    assert v["worst_regression_any_axis"] == pytest.approx(0.11)

    on = D.evaluate_arms(doc, fr_guarded=True)
    assert on["verdicts"]["A2"]["passes_primary"] is False
    assert on["verdict"] == "no_winner"
    assert on["fr_guard"]["enforced"] is True


def test_an_f_r_only_secondary_rho_drop_does_not_veto_by_default():
    """Veto (2): the ``("cyclen", "f_r")`` within-cell rho rule."""
    arm = _arm_per_target_rho("A2", {**_FLAT, "node_peak": 0.90},
                              {"f_r": 0.60})       # incumbent B0 is 0.75
    doc = _full({"A2": arm})

    out = D.evaluate_arms(doc)
    v = out["verdicts"]["A2"]
    assert v["passes_secondary"] is True, "f_r rho drop vetoed a promotion"
    assert out["verdict"] == "winner" and out["winner"] == "A2"
    assert any("f_r" in n and "REPORT-ONLY" in n for n in v["notes"])

    on = D.evaluate_arms(doc, fr_guarded=True)
    assert on["verdicts"]["A2"]["passes_secondary"] is False
    assert on["verdict"] == "no_winner"


def test_a_cyclen_regression_still_vetoes_while_f_r_is_deferred():
    """The deferral is ONE axis wide: cyclen keeps both its vetoes."""
    prim = _arm("A2", {"node_peak": 0.90, "map_cov": 1.00, "f_r": 0.50,
                       "cyclen": 0.99})            # cyclen worse by 0.11
    assert D.evaluate_arms(_full({"A2": prim}))["verdicts"]["A2"][
        "passes_primary"] is False
    sec = _arm_per_target_rho("A2", {**_FLAT, "node_peak": 0.90},
                              {"cyclen": 0.60})
    assert D.evaluate_arms(_full({"A2": sec}))["verdicts"]["A2"][
        "passes_secondary"] is False


def test_ab_decide_resolves_the_switch_off_the_same_deck(tmp_path):
    """The switch must REACH ab_decide from a deck, not only from a kwarg."""
    deck = tmp_path / "on.inp"
    deck.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = true\n",
                    encoding="utf-8")
    arm = _arm("A2", {"node_peak": 0.90, "map_cov": 1.00, "f_r": 0.61,
                      "cyclen": 0.88})
    doc = _full({"A2": arm})
    from lpopt.config import fr_guard_from_deck
    pol = fr_guard_from_deck(deck)
    assert pol["enforced"] is True and pol["source"] == "deck"
    out = D.evaluate_arms(doc, fr_guarded=pol["enforced"])
    assert out["verdict"] == "no_winner"


# --------------------------------------------------------------------------- #
# MEDIUM: the one switch must reach the OFFLINE A/B
# --------------------------------------------------------------------------- #
def test_score_flatness_ab_is_actually_handed_the_policy(tmp_path, monkeypatch):
    """``score_flatness_ab(fr_guarded=...)`` existed but no caller ever passed it
    and no path read a deck, so flipping the knob re-armed the curriculum gates
    while the offline A/B stayed deferred -- the split-brain the single-switch
    design exists to prevent."""
    from lpopt.model import ab_score as S

    deck = tmp_path / "on.inp"
    deck.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = true\n",
                    encoding="utf-8")

    seen: dict = {}

    def _fake(arms, **kw):
        seen.update(kw)
        seen["arms"] = dict(arms)
        return {"schema": "flat_ab_slate_v1", "arms": [], "judgements": {}}

    monkeypatch.setattr(S, "score_flatness_ab", _fake)
    monkeypatch.setattr(S.FA, "render_slate", lambda slate: "")
    monkeypatch.setattr(S, "merge_flat_slate", lambda slate, path: slate)

    rc = S.main(["--flat-ab", "--arm", "B1=x", "--arm", "A2=y",
                 "--deck", str(deck), "--results", str(tmp_path / "r.json")])
    assert rc == 0
    # the CLI hands the A/B a deck to resolve the switch from …
    assert seen["deck"] == str(deck), "no caller ever gave the A/B a deck to read"
    assert seen["fr_guarded"] is None, "no CLI override was asked for"
    # … and that deck resolves to ENFORCED, which is what the judgement then sees.
    from lpopt.config import fr_guard_from_deck
    assert fr_guard_from_deck(seen["deck"])["enforced"] is True


def test_the_offline_ab_cli_can_override_the_deck(tmp_path, monkeypatch):
    """An explicit ``--fr-guarded`` beats the deck (and ``--no-fr-guarded``)."""
    from lpopt.model import ab_score as S

    deck = tmp_path / "off.inp"
    deck.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = false\n",
                    encoding="utf-8")
    seen: dict = {}
    monkeypatch.setattr(S, "score_flatness_ab",
                        lambda arms, **kw: (seen.update(kw) or
                                            {"schema": "flat_ab_slate_v1"}))
    monkeypatch.setattr(S.FA, "render_slate", lambda slate: "")
    monkeypatch.setattr(S, "merge_flat_slate", lambda slate, path: slate)

    S.main(["--flat-ab", "--arm", "B1=x", "--fr-guarded", "--deck", str(deck),
            "--results", str(tmp_path / "r.json")])
    assert seen["fr_guarded"] is True
    S.main(["--flat-ab", "--arm", "B1=x", "--no-fr-guarded", "--deck", str(deck),
            "--results", str(tmp_path / "r.json")])
    assert seen["fr_guarded"] is False


def test_offline_ab_default_is_the_documented_default_not_a_second_copy(tmp_path):
    """No deck -> the SAME dataclass default, and the artifact says so."""
    from lpopt.config import CurriculumConfig, fr_guard_from_deck

    pol = fr_guard_from_deck(tmp_path / "absent.inp")
    assert pol["source"] == "default"
    assert pol["enforced"] is CurriculumConfig().gate_noreg_fr_guard_enabled
    assert "deck_error" in pol

    off = tmp_path / "off.inp"
    off.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = false\n",
                   encoding="utf-8")
    assert fr_guard_from_deck(off) == {
        "enforced": False, "source": "deck",
        "knob": "[curriculum] gate_noreg_fr_guard_enabled", "deck": str(off)}
    # an explicit override still wins (tests pin a branch without a file)
    assert fr_guard_from_deck(off, fr_guarded=True)["enforced"] is True


def test_one_deck_setting_flips_every_promotion_surface(tmp_path, monkeypatch):
    """The pin the single-switch design is FOR: one setting, every surface."""
    import numpy as np
    from lpopt.config import fr_guard_from_deck
    from lpopt.curriculum import enforced_noreg_targets
    from lpopt.model import ab_eval as ME
    from lpopt.model import flat_ab as FL

    deck = tmp_path / "on.inp"
    deck.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = true\n",
                    encoding="utf-8")
    on = fr_guard_from_deck(deck)["enforced"]
    assert on is True

    # 1. curriculum no-regression gate
    assert "f_r" in enforced_noreg_targets(fr_guarded=on)
    # 2. offline A/B proxy gate
    cells = np.array(["c0"] * 40 + ["c1"] * 40)
    truth = {"cyclen": np.arange(80.0), "f_r": np.arange(80.0)}
    old = {"cyclen": np.arange(80.0), "f_r": np.arange(80.0)}
    new = {"cyclen": np.arange(80.0), "f_r": -np.arange(80.0)}
    assert ME.no_regression_gate(new, old, truth, cells,
                                 fr_guarded=on)["pass"] is False
    # 3. flatness A/B condition 5
    assert FL._fr_guard_enforced(on) is True
    # 4. ab_decide's two point vetoes
    arm = _arm("A2", {"node_peak": 0.90, "map_cov": 1.00, "f_r": 0.61,
                      "cyclen": 0.88})
    assert D.evaluate_arms(_full({"A2": arm}),
                           fr_guarded=on)["verdict"] == "no_winner"
    # 5. the fold-C gate ``ab_score`` writes and ``ab_decide`` reads as
    #    ``passes_gate`` -- the surface the switch used not to reach at all
    S, champ = _stub_score_arm_world(monkeypatch, tmp_path)
    scored = S.score_arm("A2", tmp_path / "arm", folds=("C",), bootstrap=0,
                         cache_dir=None, champion_dir=champ, verbose=False,
                         deck=deck)
    assert scored["folds"]["C"]["gate"]["fr_guard"]["enforced"] is True
    assert scored["folds"]["C"]["gate"]["pass"] is False


# --------------------------------------------------------------------------- #
# HIGH: the fold-C no-regression gate is the FIFTH promotion surface, and it was
# the one the switch never reached.
#
# ``ab_score.score_arm`` built ``folds[C].gate`` by calling
# ``ab_eval.no_regression_gate`` with no ``fr_guarded`` and no deck, so the gate
# resolved to the dataclass DEFAULT however the deck was set.  ``ab_decide`` then
# consumed that gate verbatim as ``passes_gate`` -- one of the four legs of
# promotion eligibility -- and, unlike ``flat_ab.judge_arm``, never checked the
# gate's own recorded setting against the setting it was judging at.  So flipping
# ``[curriculum] gate_noreg_fr_guard_enabled`` would re-arm every other surface
# and leave this one deferred, silently.
# --------------------------------------------------------------------------- #
def _stub_score_arm_world(monkeypatch, tmp_path):
    """A real run of ``score_arm`` down to the fold-C gate, everything else stubbed.

    The gate itself is the genuine :func:`ab_eval.no_regression_gate`, because
    the setting that gate resolves to is exactly what is under test.  The arm's
    ``f_r`` is anti-ranked against truth while the champion's is perfectly
    ranked, so the F_r axis alone decides: DEFERRED it passes, ARMED it fails.
    """
    from lpopt.model import ab_score as S
    from lpopt.model.folds import FoldFrame

    n = 80
    truth_vec = np.arange(float(n))
    df = pd.DataFrame({
        "record_id": [f"r{i:03d}" for i in range(n)],
        "converged": [True] * n,
        "cyclen": truth_vec,
        "f_r": truth_vec,
    })
    cells = np.array(["c0"] * 40 + ["c1"] * 40)
    ff = FoldFrame(fold="C", name="fold C", df=df, cells=cells,
                   is_proposal=np.zeros(n, dtype=bool))

    class _Arm:
        def __init__(self, sign):
            self.sign = sign
            self.meta = {"target_names": ["cyclen", "f_r"], "cond_schema": "v5",
                         "net_config": {}, "train_config": {}}

        @classmethod
        def load(cls, d, device="cpu"):
            # the champion ranks f_r perfectly; the arm inverts it
            return cls(1.0 if "champ" in str(d) else -1.0)

        def encode(self, df_, fuel, cache_dir=None, verbose=False):
            return None, None

        def predict(self, cells_t, gvecs):
            return np.stack([truth_vec, self.sign * truth_vec], axis=1), \
                np.zeros((n, 69))

        def add_cyclen_prior(self, mu, df_, fuel):
            return mu

    champ_dir = tmp_path / "champ"
    champ_dir.mkdir()

    class _Reader:
        records = df

        @staticmethod
        def maps(k):
            return None

    monkeypatch.setattr(S, "ArmModel", _Arm)
    monkeypatch.setattr(S, "StoreReader", lambda d: _Reader())
    monkeypatch.setattr(S.FuelLibrary, "from_parquet", staticmethod(lambda p: None))
    monkeypatch.setattr(S.SplitManifest, "from_json", staticmethod(lambda p: None))
    monkeypatch.setattr(S, "fold_frame", lambda *a, **k: ff)
    monkeypatch.setattr(S, "true_maps",
                        lambda r, d: (np.zeros(n, dtype=bool), np.zeros((0, 69))))
    monkeypatch.setattr(S.M, "effective_resolution", lambda *a, **k: {})
    monkeypatch.setattr(S.M, "within_cell_stats", lambda *a, **k: {})
    return S, str(champ_dir)


def test_score_arm_resolves_the_fold_c_gate_off_the_deck(tmp_path, monkeypatch):
    """One deck setting must re-arm the fold-C gate too.

    Before the fix ``score_arm`` took neither a deck nor an ``fr_guarded``, so
    the ON deck below still produced a gate reporting ``enforced: false`` that
    still PASSED an arm whose f_r ranking had collapsed.
    """
    S, champ = _stub_score_arm_world(monkeypatch, tmp_path)
    on = tmp_path / "on.inp"
    on.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = true\n",
                  encoding="utf-8")
    off = tmp_path / "off.inp"
    off.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = false\n",
                   encoding="utf-8")

    armed = S.score_arm("A2", tmp_path / "arm", folds=("C",), bootstrap=0,
                        cache_dir=None, champion_dir=champ, verbose=False,
                        deck=on)
    gate = armed["folds"]["C"]["gate"]
    assert gate["fr_guard"]["enforced"] is True, (
        "the deck says ENFORCED but score_arm's gate resolved to the default")
    assert "f_r" in gate["guarded_targets"]
    assert gate["pass"] is False, "an armed F_r collapse still passed the gate"
    assert armed["fr_guard_policy"]["source"] == "deck"
    assert armed["fr_guard_policy"]["enforced"] is True

    deferred = S.score_arm("A2", tmp_path / "arm", folds=("C",), bootstrap=0,
                           cache_dir=None, champion_dir=champ, verbose=False,
                           deck=off)
    dg = deferred["folds"]["C"]["gate"]
    assert dg["fr_guard"]["enforced"] is False
    assert dg["pass"] is True, "today's setting must still let this arm through"
    assert deferred["fr_guard_policy"]["enforced"] is False


def test_score_arm_cli_override_beats_the_deck(tmp_path, monkeypatch):
    """``--fr-guarded`` pins the branch without editing a deck (as flat-ab's does)."""
    S, champ = _stub_score_arm_world(monkeypatch, tmp_path)
    off = tmp_path / "off.inp"
    off.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = false\n",
                   encoding="utf-8")
    e = S.score_arm("A2", tmp_path / "arm", folds=("C",), bootstrap=0,
                    cache_dir=None, champion_dir=champ, verbose=False,
                    deck=off, fr_guarded=True)
    assert e["fr_guard_policy"]["source"] == "explicit"
    assert e["folds"]["C"]["gate"]["fr_guard"]["enforced"] is True
    assert e["folds"]["C"]["gate"]["pass"] is False


def test_the_ab_score_cli_hands_score_arm_the_deck(tmp_path, monkeypatch):
    """The per-arm path must read the same ``--deck`` the flat-ab path does."""
    from lpopt.model import ab_score as S

    seen: dict = {}
    monkeypatch.setattr(S, "score_arm",
                        lambda label, d, **kw: (seen.update(kw) or
                                                {"label": label, "folds": {}}))
    monkeypatch.setattr(S, "update_results", lambda entry, path: {})
    deck = tmp_path / "on.inp"
    deck.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = true\n",
                    encoding="utf-8")
    S.main(["--arm", "A2=x", "--deck", str(deck), "--markdown", "",
            "--results", str(tmp_path / "r.json")])
    assert seen["deck"] == str(deck), "the per-arm CLI never passed a deck"
    assert seen["fr_guarded"] is None


def test_ab_decide_reports_a_gate_computed_at_the_other_setting(tmp_path):
    """A split-brain must be REPORTED, not silently consumed.

    ``flat_ab.judge_arm`` already refuses to mix a gate computed at one setting
    with a judgement made at the other; ``ab_decide`` consumed ``gate['pass']``
    verbatim, so a stale-setting gate would be laundered into ``passes_gate``
    with nothing in the artifact to show it.
    """
    arm = _arm("A2", {**_FLAT, "node_peak": 0.90})
    # a gate that PASSED, but that was computed with the guard DEFERRED
    arm["folds"]["C"]["gate"] = {
        "pass": True, "worst_drop": 0.0,
        "fr_guard": {"target": "f_r", "enforced": False,
                     "knob": "[curriculum] gate_noreg_fr_guard_enabled"}}
    doc = _full({"A2": arm})

    # judged at the SAME setting: no complaint, the arm wins on its merits
    same = D.evaluate_arms(doc, fr_guarded=False)
    assert same["verdicts"]["A2"]["passes_gate"] is True
    assert same["verdict"] == "winner"

    # judged with the guard ARMED: the gate is a mismatched input
    mixed = D.evaluate_arms(doc, fr_guarded=True)
    v = mixed["verdicts"]["A2"]
    assert v["passes_gate"] is False, "a stale-setting gate was laundered into a pass"
    assert v["eligible"] is False
    assert any("gate_noreg_fr_guard_enabled" in n and "DEFERRED" in n
               for n in v["notes"]), v["notes"]
    assert mixed["verdict"] == "no_winner"


def test_a_gate_with_no_fr_guard_block_is_not_a_mismatch():
    """Legacy artifacts predate the block; absence is not evidence of conflict."""
    arm = _arm("A2", {**_FLAT, "node_peak": 0.90})
    assert "fr_guard" not in arm["folds"]["C"]["gate"]
    out = D.evaluate_arms(_full({"A2": arm}), fr_guarded=True)
    assert out["verdicts"]["A2"]["passes_gate"] is True


# --------------------------------------------------------------------------- #
# MEDIUM: the default deck must not depend on where the command was typed
# --------------------------------------------------------------------------- #
def test_the_default_deck_is_not_resolved_against_the_cwd(tmp_path, monkeypatch):
    """``FR_GUARD_DEFAULT_DECK = 'lpopt.inp'`` was a bare relative path, so the
    offline A/B silently fell back to the deferred default whenever it ran from
    anywhere but the repo root -- the resolved POLICY depended on the shell's
    working directory."""
    import lpopt
    from lpopt.config import FR_GUARD_DEFAULT_DECK, fr_guard_from_deck

    p = Path(FR_GUARD_DEFAULT_DECK)
    assert p.is_absolute(), (
        f"{FR_GUARD_DEFAULT_DECK!r} is relative; it resolves against the cwd")
    assert p.name == "lpopt.inp"
    assert p.parent == Path(lpopt.__file__).resolve().parent.parent

    # the resolution is identical from two different working directories
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    home = fr_guard_from_deck()
    monkeypatch.chdir(elsewhere)
    away = fr_guard_from_deck()
    assert away == home, "the F_r-guard policy moved with the working directory"


def test_a_missing_default_deck_is_loud_not_silent(tmp_path):
    """If the deck is unreadable the fallback must NAME it, so 'deferred because
    nothing was read' never reads as 'deferred because the deck says so'."""
    from lpopt.config import fr_guard_from_deck

    pol = fr_guard_from_deck(tmp_path / "nope.inp")
    assert pol["source"] == "default"
    assert "deck_error" in pol and "nope.inp" in pol["deck_error"]
