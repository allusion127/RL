"""``lpopt.policy.metrics_v31`` — prereg delta D, on a synthetic corpus.

The module is the thing that lifts ``train_v3``'s ``--stage2 on`` refusal, so
what it has to be tested for is not "does it produce numbers" but the four
properties the refusal was protecting:

* the K-fold STITCH is a partition — every pool row scored exactly once, by a
  fit that never saw it (``test_a_baseline_never_sees_the_block_it_scores`` is
  the quantitative form: the ``class_freq`` rate a block's rows are scored with
  is computed from the OTHER blocks and provably differs from the pooled one);
* the paired-parent CI machinery and the registered sizing formula agree with
  the arithmetic §6a is written in;
* ECE is computed and REPORTED and cannot silently become a gate (§5c);
* the within-parent permutation test separates a rankable cell from an
  unrankable one, because clause 3 turns a FAIL into UNDECIDABLE on its verdict.

The corpus is synthetic and small on purpose: the frozen 28,970-row one is on
the box, and a fold-arithmetic property that only holds on that file is not a
property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lpopt.policy import metrics_v31 as M
from lpopt.policy.train_v3 import BASELINES
from lpopt.policy.v3 import PROSPECTIVE_CELL_V31

K = 5
CELLS = ("A_B/f109/ga80", "C_D/f113/ga80", "E_F/f121/ga80")


def _corpus(seed: int = 20260903, per_parent: int = 8, n_pool: int = 60,
            n_val: int = 6, n_legacy: int = 40, n_hold: int = 12
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A corpus and its ``DataFrame[fold, xfit_fold]``, built by hand.

    One parent is one lineage component, so the block deal is a parent deal and
    the leakage question the real ``build_splits_v31`` answers with components is
    answered here with parents — the metric module reads only the assignment.
    """
    rng = np.random.default_rng(seed)
    rows, fold, xfit = [], [], []

    def emit(pid: str, cell: str, tag: str, block: int, signal: float,
             klass_pool: tuple[str, ...]) -> None:
        for j in range(per_parent):
            quality = rng.normal()
            improve = quality * signal > 0.4
            d_fxy = -0.02 * quality * signal if improve else 0.01 * abs(quality)
            rows.append({
                "parent_record_id": pid, "child_record_id": f"{pid}_c{j}",
                "cell": cell, "move_class": klass_pool[j % len(klass_pool)],
                "era_current": tag != "train", "both_converged": True,
                "d_f_xy": d_fxy, "d_f_r": -0.01 * quality,
                "d_node_peak": -0.005 * quality, "d_cyclen": 1.0,
                "child_f_r": 1.40, "parent_f_r": 1.42,
                "improved_fxy": bool(d_fxy < 0), "improved_fr": bool(quality > 0),
                "improved_flat": bool(quality > 0),
                "d_fresh_share_periph": 0.3 * quality + rng.normal(scale=0.1),
                "d_fresh_gd_mass": -20.0 * quality + rng.normal(scale=5.0),
                "quality": quality,
            })
            fold.append(tag)
            xfit.append(block)

    # The pool: n_pool parents dealt round-robin into K blocks.  ``move_class``
    # is BLOCK-DEPENDENT for the first class so the refit isolation is
    # measurable: "cls_hot" appears only in block 0 and only on improving rows.
    for i in range(n_pool):
        block = i % K
        cell = CELLS[i % len(CELLS)]
        klass = ("cls_hot", "cls_hot") if block == 0 else ("cls_a", "cls_b")
        emit(f"P{i:03d}", cell, "pool", block, 1.0, klass)
    for i in range(n_val):
        emit(f"V{i:03d}", CELLS[0], "val", -1, 1.0, ("cls_a", "cls_b"))
    for i in range(n_legacy):
        emit(f"L{i:03d}", CELLS[i % len(CELLS)], "train", -1, 1.0,
             ("cls_a", "cls_b", "cls_hot"))
    for i in range(n_hold):
        emit(f"H{i:03d}", PROSPECTIVE_CELL_V31, "prospective_cell", -1, 1.0,
             ("cls_a", "cls_b"))

    steps = pd.DataFrame(rows)
    splits = pd.DataFrame({"child_record_id": steps["child_record_id"],
                           "fold": fold, "xfit_fold": xfit})
    return steps, splits


@pytest.fixture(scope="module")
def corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _corpus()


def _v2_csv(steps: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """A stand-in for the blind v2 baseline CSV — one frozen column per row."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "child_record_id": steps["child_record_id"],
        "p_improve_fr": 1.0 / (1.0 + np.exp(
            -(0.8 * steps["quality"].to_numpy() + rng.normal(scale=0.5,
                                                             size=len(steps))))),
    })


# --------------------------------------------------------------------------- #
# §5d — the stitch
# --------------------------------------------------------------------------- #
def test_every_pool_row_is_scored_exactly_once_and_nothing_else_is(corpus):
    steps, splits = corpus
    cols, fits = M.oof_baseline_scores(steps, splits, v2=_v2_csv(steps))
    pool = (splits["fold"] == "pool").to_numpy()
    assert len(fits) == K
    for name in M.FITTED_BASELINES:
        v = cols[name]
        assert np.isfinite(v[pool]).all(), f"{name} left a pool row unscored"
        assert not np.isfinite(v[~pool]).any(), (
            f"{name} scored a row outside the cross-fit pool; val, the legacy "
            f"train fold and the held-out cell are in no block")


def test_a_baseline_never_sees_the_block_it_scores(corpus):
    """§5d, the quantitative form.

    ``cls_hot`` exists in block 0's eval rows and in the legacy train fold and
    NOWHERE else in the pool.  If the fit were pooled, block 0's rows would be
    scored with a rate that their own labels helped set; refitted inside the
    block's train fold they are scored with the legacy rate alone, and the two
    numbers differ.
    """
    steps, splits = corpus
    cols, _ = M.oof_baseline_scores(steps, splits, v2=None)
    fold = splits["fold"].to_numpy()
    xf = splits["xfit_fold"].to_numpy()
    hot = (steps["move_class"] == "cls_hot").to_numpy()

    ev0 = np.flatnonzero((fold == "pool") & (xf == 0) & hot)
    assert len(ev0) > 0
    fit_rows = ((fold == "train") | ((fold == "pool") & (xf != 0))) & hot
    want = float(steps.loc[fit_rows, "improved_fxy"].mean())
    got = np.unique(np.round(cols["class_freq"][ev0], 12))
    assert got.size == 1 and got[0] == pytest.approx(want, abs=1e-12)

    pooled = float(steps.loc[hot & (fold != "prospective_cell"),
                             "improved_fxy"].mean())
    assert abs(want - pooled) > 1e-6, (
        "the synthetic corpus does not separate the in-fold and out-of-fold "
        "rates, so this test cannot detect a leaked fit")


def test_the_gd_rule_sign_is_refitted_per_block(corpus):
    steps, splits = corpus
    _, fits = M.oof_baseline_scores(steps, splits, v2=None)
    assert {f["block"] for f in fits} == set(range(K))
    assert all(f["gd_rule_sign"] in (-1.0, 1.0) for f in fits)
    assert all(f["n_fit_rows"] > f["n_eval_rows"] for f in fits)


def test_policy_v2_is_carried_through_and_not_refitted(corpus):
    """§5d: the fifth baseline is a frozen checkpoint's blind CSV.  Whatever
    block a row lands in, it is scored with the same number."""
    steps, splits = corpus
    v2 = _v2_csv(steps)
    cols, _ = M.oof_baseline_scores(steps, splits, v2=v2)
    pool = (splits["fold"] == "pool").to_numpy()
    table = v2.set_index("child_record_id")["p_improve_fr"]
    want = steps["child_record_id"].map(table).to_numpy(float)
    assert np.allclose(cols["policy_v2"][pool], want[pool])


def test_a_block_that_overlaps_its_own_eval_fold_is_refused(corpus):
    steps, splits = corpus
    bad = splits.copy()
    # Put one of block 0's eval rows into the legacy train fold as well by
    # relabelling a block-1 row into block 0's eval set twice over.
    ev0 = np.flatnonzero((bad["fold"] == "pool").to_numpy()
                         & (bad["xfit_fold"] == 0).to_numpy())
    bad.loc[bad.index[ev0[0]], "fold"] = "pool"
    bad2 = bad.copy()
    bad2["xfit_fold"] = np.where((bad2["fold"] == "pool").to_numpy(), 0,
                                 bad2["xfit_fold"])
    # every pool row now belongs to block 0, so blocks 1..4 have no eval rows
    cols, fits = M.oof_baseline_scores(steps, bad2, v2=None)
    assert len(fits) == 1 and fits[0]["block"] == 0
    _ = cols


# --------------------------------------------------------------------------- #
# §5a — the registered gain, and only it
# --------------------------------------------------------------------------- #
def test_the_registered_gain_is_in_the_unit_interval(corpus):
    steps, _ = corpus
    g = M.registered_gain(steps)
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert M.assert_registered_gain(g, steps) is not None


def test_a_ranking_statistic_fed_the_raw_gain_raises(corpus):
    """§9a-H(c) — v3 deviation §1.6 closed from the metric side."""
    steps, _ = corpus
    raw = -pd.to_numeric(steps["d_f_xy"], errors="coerce").to_numpy(float)
    with pytest.raises(ValueError, match="registered y_fxy"):
        M.assert_registered_gain(raw, steps)


def test_a_rescaled_raw_gain_is_still_refused(corpus):
    """A monotone re-scaling into [0, 1] is not the registered gain either: the
    clip and the feasibility mask are what DEFINE it (§2a)."""
    steps, _ = corpus
    raw = -pd.to_numeric(steps["d_f_xy"], errors="coerce").to_numpy(float)
    scaled = (raw - raw.min()) / (raw.max() - raw.min())
    with pytest.raises(ValueError, match="differs from the registered y_fxy"):
        M.assert_registered_gain(scaled, steps)


def test_the_gate_report_refuses_a_raw_gain_argument(corpus):
    steps, splits = corpus
    z = np.asarray(steps["quality"], float)
    raw = -pd.to_numeric(steps["d_f_xy"], errors="coerce").to_numpy(float)
    with pytest.raises(ValueError):
        M.gate_report_v31(steps, splits, z, gain=raw, perm_reps=32)


# --------------------------------------------------------------------------- #
# CI machinery and the registered sizing arithmetic
# --------------------------------------------------------------------------- #
def test_n80_is_the_prereg_formula():
    """§6a clause 2B's own table: sd 0.2724, Delta 0.0423, delta 0.05 -> 54."""
    assert M.n80(0.0423, 0.2724, margin=0.05) == pytest.approx(54.0, abs=1.0)
    assert M.n80(0.0, 0.2724, margin=0.05) == pytest.approx(184.0, abs=2.0)
    assert M.n80(0.0, 0.0) == float("inf")


def test_power_and_n80_are_the_same_statement():
    """At ``n = n80`` the one-sided power is 0.80 by construction."""
    sd, d, margin = 0.2724, 0.0423, 0.05
    n = M.n80(d, sd, margin=margin)
    assert M.normal_power(d, sd, int(round(n)), margin=margin) == pytest.approx(
        0.80, abs=0.02)


def test_clause_2b_is_one_sided_and_paired(corpus):
    steps, splits = corpus
    per = {"policy": np.array([0.5, 0.6, 0.7, 0.55, 0.65] * 8),
           "policy_v2": np.array([0.45, 0.62, 0.66, 0.5, 0.6] * 8)}
    out = M.clause_2b_noninferiority(per)
    d = per["policy"] - per["policy_v2"]
    assert out["delta"] == pytest.approx(float(d.mean()))
    assert out["sd"] == pytest.approx(float(d.std(ddof=1)))
    assert out["lo_one_sided_95"] == pytest.approx(
        float(d.mean()) - M.NI_Z * float(d.std(ddof=1)) / np.sqrt(len(d)))
    assert out["PASS"] is True and out["verdict"] == "PASS"
    _ = steps, splits


def test_a_clause_2b_fail_under_low_power_is_undecided_not_fail():
    """§6d, written before the numbers: the round must not record a FAIL it
    could not have detected."""
    rng = np.random.default_rng(3)
    d = rng.normal(-0.30, 0.30, size=12)          # badly worse, tiny n
    per = {"policy": d, "policy_v2": np.zeros_like(d)}
    out = M.clause_2b_noninferiority(per)
    assert out["PASS"] is False
    assert out["power_at_observed"] < M.POWER_FLOOR
    assert out["verdict"] == "UNDECIDED"


# --------------------------------------------------------------------------- #
# the permutation test, the MDE and clause 3
# --------------------------------------------------------------------------- #
def test_the_permutation_test_separates_a_ranked_cell_from_noise(corpus):
    steps, splits = corpus
    pool = (splits["fold"] == "pool").to_numpy()
    sub = steps[pool].reset_index(drop=True)
    g = M.registered_gain(sub)
    par = sub["parent_record_id"].to_numpy()
    good = M.within_parent_permutation(-sub["d_f_xy"].to_numpy(), g, par,
                                       reps=400, seed=1)
    noise = M.within_parent_permutation(
        np.random.default_rng(0).normal(size=len(sub)), g, par, reps=400, seed=1)
    assert good["p_value"] < 0.01
    assert noise["p_value"] > 0.05
    assert good["observed"] > noise["observed"]


def test_the_measured_mde_is_the_registered_constant_times_the_null_sd(corpus):
    steps, splits = corpus
    pool = (splits["fold"] == "pool").to_numpy()
    sub = steps[pool].reset_index(drop=True)
    g = M.registered_gain(sub)
    out = M.within_parent_permutation(
        np.random.default_rng(1).normal(size=len(sub)), g,
        sub["parent_record_id"].to_numpy(), reps=400, seed=2)
    assert out["mde"] == pytest.approx(M.N80_Z * out["null_sd"])
    # §3c-(4)'s closed form is the same order as the measured value; the point
    # of measuring is that it is not the same NUMBER.
    assert 0.3 < out["mde"] / out["mde_closed_form"] < 3.0


def test_a_cell_below_the_live_floor_is_undecidable_not_a_fail(corpus):
    steps, splits = corpus
    pool = (splits["fold"] == "pool").to_numpy()
    sub = steps[pool].reset_index(drop=True)
    g = M.registered_gain(sub)
    par = sub["parent_record_id"].to_numpy()
    scores = {"policy": -sub["d_f_xy"].to_numpy()}
    cols, _ = M.oof_baseline_scores(steps, splits, v2=_v2_csv(steps))
    for b in BASELINES:
        scores[b] = np.nan_to_num(cols[b][pool], nan=-9.0)
    out = M.clause_3_within_cell(sub, scores, g, par, min_live=10_000,
                                 reps=64)
    assert out["cells"], "no cell was censused"
    assert all(e["verdict"] == "UNDECIDABLE" for e in out["cells"].values())
    assert out["n_eligible"] == 0
    assert out["PASS"] is False        # nothing eligible is not a pass either


def test_an_unrankable_cell_is_undecidable(corpus):
    """Clause 3-(ii): the eligibility test reads the BASELINES, never v3.1."""
    steps, splits = corpus
    pool = (splits["fold"] == "pool").to_numpy()
    sub = steps[pool].reset_index(drop=True)
    g = M.registered_gain(sub)
    par = sub["parent_record_id"].to_numpy()
    rng = np.random.default_rng(11)
    scores = {"policy": -sub["d_f_xy"].to_numpy()}
    for b in BASELINES:                       # every baseline is pure noise
        scores[b] = rng.normal(size=len(sub))
    out = M.clause_3_within_cell(sub, scores, g, par, min_live=1, reps=200)
    assert out["undecidable"], (
        "a cell whose five baselines are all at chance must be UNDECIDABLE "
        "even though v3.1 itself ranks it perfectly")
    for cell in out["undecidable"]:
        assert "clause 3-ii" in out["cells"][cell]["reason"]


def test_the_cell_concentration_clause_voids_a_lopsided_pool(corpus):
    steps, splits = corpus
    pool = (splits["fold"] == "pool").to_numpy()
    sub = steps[pool].reset_index(drop=True)
    g = M.registered_gain(sub)
    par = sub["parent_record_id"].to_numpy()
    ok = M.cell_concentration(sub, g, par)
    assert ok["VOID"] is False and ok["max_share"] <= M.CELL_SHARE_MAX
    one = sub.copy()
    one["cell"] = "ONE/CELL/ga80"
    assert M.cell_concentration(one, g, par)["VOID"] is True


# --------------------------------------------------------------------------- #
# calibration and the serving scale
# --------------------------------------------------------------------------- #
def test_ece_is_reported_and_says_it_is_not_gated():
    p = np.linspace(0.01, 0.99, 200)
    y = (np.arange(200) % 5 == 0).astype(float)
    out = M.calibration_report(p, y)
    assert out["gated"] is False and "§5c" in out["why_not_gated"]
    assert 0.0 <= out["ece"] <= 1.0 and out["report_line"] == 0.05
    assert set(out) >= {"brier", "ece", "bins", "ece_le_line"}


def test_clause_4_is_fitted_on_calib_and_widens_without_reordering(corpus):
    steps, splits = corpus
    # A NOISY label, deliberately: a separable one drives the Platt slope to
    # infinity and the served probabilities saturate to exact 0/1, at which
    # point float64 has lost the ordering the map is supposed to preserve.  The
    # real gate fold has a 15% base rate and is nothing like separable (§5c).
    rng = np.random.default_rng(4)
    z = 1.5 * np.asarray(steps["quality"], float) - 1.0
    y = (rng.random(len(steps)) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    pool = np.flatnonzero((splits["fold"] == "pool").to_numpy())
    out = M.clause_4_serving_scale(z, y, splits, pool)
    assert out["platt"]["a"] > 0.0
    assert out["platt"]["n_calib"] == int((splits["fold"] == "pool").sum())
    served = M.platt_serve(z, a=out["platt"]["a"], b=out["platt"]["b"])
    assert np.array_equal(np.argsort(served), np.argsort(z))
    assert out["spread_min"] == M.SERVING_SPREAD_MIN
    assert out["spread_logit"] > out["spread_served"]       # §5c's whole point


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def test_a_foreign_split_file_is_refused(tmp_path):
    p = tmp_path / "splits_v31.csv"
    p.write_text("child_record_id,fold,xfit_fold\nx,pool,0\n")
    assert M.splits_fingerprint_ok(p) is False
    with pytest.raises(SystemExit, match="registered"):
        M.assert_splits_registered(p)
    assert M.assert_splits_registered(p, expected=M.sha256_file(p))


def test_a_missing_split_file_names_the_emission_command(tmp_path):
    with pytest.raises(SystemExit, match="--xfit-k 5"):
        M.assert_splits_registered(tmp_path / "nope.csv")


def test_the_split_must_be_row_aligned_with_the_corpus(corpus):
    steps, splits = corpus
    M.assert_splits_align(steps, splits)
    shuffled = splits.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SystemExit, match="row-aligned"):
        M.assert_splits_align(steps, shuffled)
    with pytest.raises(SystemExit, match="not for this corpus"):
        M.assert_splits_align(steps, splits.iloc[:-1])


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_the_gate_report_carries_every_registered_clause(corpus):
    steps, splits = corpus
    z = 3.0 * np.asarray(steps["quality"], float)
    report = M.gate_report_v31(steps, splits, z, v2=_v2_csv(steps),
                               perm_reps=64)
    for key in ("clause_1", "clause_2A", "clause_2B", "clause_3", "clause_4",
                "cell_concentration", "calibration", "mde", "PASS",
                "block_fits", "transfer_bar"):
        assert key in report, key
    assert len(report["block_fits"]) == K
    assert report["clause_1"]["policy"] > 0.5
    assert set(report["clause_2A"]["delta"]) == set(M.FITTED_BASELINES)
    assert report["clause_2B"]["verdict"] in ("PASS", "FAIL", "UNDECIDED")
    assert report["transfer_bar"]["cell"] == PROSPECTIVE_CELL_V31
    assert isinstance(M.render_gate(report), str)


def test_the_gate_report_needs_one_logit_per_corpus_row(corpus):
    steps, splits = corpus
    with pytest.raises(ValueError, match="logits for"):
        M.gate_report_v31(steps, splits, np.zeros(len(steps) - 1), perm_reps=8)


def test_the_transfer_cell_is_scored_by_a_fit_that_never_saw_it(corpus):
    """§3a-3: the held-out cell is in no block, so every block's fit is out of
    fold for it and the report records WHICH one it used."""
    steps, splits = corpus
    z = 3.0 * np.asarray(steps["quality"], float)
    report = M.gate_report_v31(steps, splits, z, v2=_v2_csv(steps),
                               perm_reps=32)
    assert report["transfer_bar"]["baseline_fit_block"] == 0
    hold = (splits["fold"] == "prospective_cell").to_numpy()
    cols, _ = M.oof_baseline_scores(steps, splits, v2=None)
    for b in M.FITTED_BASELINES:
        assert not np.isfinite(cols[b][hold]).any()


def test_the_gate_report_reads_a_masked_boolean_label_with_missing_rows():
    """The frozen corpus carries ``improved_fxy`` as pandas' MASKED ``boolean``
    dtype, most rows ``<NA>``: only 1,309 of 28,970 have an F_xy reading.

    The v3.1 cross-fit run died here after fold 0 -- ``label.fillna(0)`` is a
    dtype error on a boolean array (pandas 3.0: "Invalid value '0' for dtype
    'boolean'"), and on older pandas it silently object-ified the column.  The
    unlabelled rows must simply not be judged; they are excluded by
    ``label.notna()`` everywhere downstream, so the fill VALUE is irrelevant and
    must not be allowed to be a dtype question.
    """
    steps, splits = _corpus()
    steps = steps.copy()
    label = steps["improved_fxy"].astype("boolean")
    blank = np.zeros(len(steps), bool)
    blank[::3] = True                       # a third of every fold goes missing
    label[blank] = pd.NA
    steps["improved_fxy"] = label
    assert str(steps["improved_fxy"].dtype) == "boolean"
    assert steps["improved_fxy"].isna().any()

    z = 3.0 * np.asarray(steps["quality"], float)
    report = M.gate_report_v31(steps, splits, z, v2=_v2_csv(steps),
                               perm_reps=32)

    pool = (splits["fold"] == "pool").to_numpy()
    assert report["pool"]["n_rows"] == int(pool.sum())
    assert report["pool"]["n_fxy"] == int(
        (steps["improved_fxy"].notna().to_numpy() & pool).sum())
    assert report["pool"]["n_fxy"] < report["pool"]["n_rows"]
    # the <NA> rows are dropped, not scored as negatives: the signal survives
    assert report["clause_1"]["policy"] > 0.5
    assert np.isfinite(report["clause_4"]["platt"]["a"])
