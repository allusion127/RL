"""The flatness-first A/B judging apparatus — program sections 7.1 through 8.6.

The load-bearing test in this file is
:func:`test_point_estimate_favours_a_but_the_paired_ci_refuses_to_promote`: an
arm that wins on every point estimate and whose paired interval includes the
null must NOT promote.  That is the whole reason the apparatus exists, and it is
built here from row-level predictions rather than from hand-written summary
numbers, so it exercises the real metric -> paired -> rule chain.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.model import ab_paired as P
from lpopt.model import c2_slice as C2
from lpopt.model import flat_ab as FA
from lpopt.model import flat_metrics as FM
from lpopt.model.splits import SplitManifest

REPS = 400


# =========================================================================== #
# paired inference (section 8.3)
# =========================================================================== #
def test_norm_ppf_inverts_norm_cdf():
    for p in (0.001, 0.01, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999):
        assert abs(P.norm_cdf(P.norm_ppf(p)) - p) < 1e-12


def test_norm_ppf_matches_known_quantiles():
    assert abs(P.norm_ppf(0.975) - 1.959963985) < 1e-8
    assert abs(P.norm_ppf(0.025) + 1.959963985) < 1e-8


def _cells(n, start=0):
    return [f"cell{i:02d}" for i in range(start, start + n)]


def test_uniform_gain_is_established():
    keys = _cells(12)
    arm = {k: 0.10 + 0.01 * i for i, k in enumerate(keys)}
    ctl = {k: 0.20 + 0.01 * i for i, k in enumerate(keys)}
    d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                higher_is_better=False, reps=REPS)
    assert d.point > 0                       # lower is better -> arm gains 0.10
    assert d.establishes_gain()
    assert d.method in ("bca", "degenerate")


def test_the_ci_is_on_the_difference_not_on_two_medians():
    """A shared cell effect swamps single-arm CIs and cancels in the paired one.

    This is the concrete reason ``cluster_bootstrap_ci`` cannot be used for a
    comparison: both arms' marginal intervals here are enormous and overlap
    almost entirely, while the difference is nailed down.
    """
    from lpopt.model.ab_eval import cluster_bootstrap_ci
    rng = np.random.default_rng(0)
    keys = _cells(20)
    shared = {k: float(v) for k, v in zip(keys, rng.normal(0.5, 0.30, len(keys)))}
    ctl = dict(shared)
    arm = {k: v + 0.02 for k, v in shared.items()}

    a_lo, a_hi = cluster_bootstrap_ci(list(arm.values()), reps=REPS)
    c_lo, c_hi = cluster_bootstrap_ci(list(ctl.values()), reps=REPS)
    assert a_lo < c_hi and c_lo < a_hi, "single-arm intervals overlap heavily"

    d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                reps=REPS)
    assert d.establishes_gain()
    assert abs(d.point - 0.02) < 1e-9


def test_a_point_lead_with_a_straddling_interval_establishes_nothing():
    # 7 cells favour the arm, 5 favour the control by more -> median > 0 but the
    # resampled median is negative far more often than 2.5% of the time.
    arm = {f"g{i}": 0.010 + 0.001 * i for i in range(7)}
    arm.update({f"b{i}": -0.040 - 0.002 * i for i in range(5)})
    ctl = {k: 0.0 for k in arm}
    d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                reps=2000, seed=3)
    assert d.point > 0
    assert not d.establishes_gain()
    assert d.straddles_null()
    assert d.favours_arm_on_points_only()


def test_harm_upper_is_the_negated_gain_lower_bound():
    arm = {k: -0.001 * i for i, k in enumerate(_cells(15))}
    ctl = {k: 0.0 for k in arm}
    d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                reps=REPS)
    assert d.harm_upper == pytest.approx(-d.ci_lo)
    assert d.bounds_harm(1.0) and not d.bounds_harm(0.0)


def test_too_few_clusters_refuses_to_judge():
    d = P.paired_cell_bootstrap({"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 0.0},
                                metric="m", arm="A", control="B", reps=REPS)
    assert d.method == "insufficient"
    assert not d.establishes_gain()
    assert not d.bounds_harm(10.0), "an absent interval cannot bound a harm either"
    assert "paired cells" in " ".join(d.notes)


def test_bca_degrades_to_percentile_below_the_cluster_floor():
    keys = _cells(4)
    arm = {k: 0.5 + 0.1 * i for i, k in enumerate(keys)}
    ctl = {k: 0.0 for k in keys}
    d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                reps=REPS)
    assert d.method == "percentile"
    assert any("clusters" in n for n in d.notes)


def test_an_identical_shift_in_every_cell_is_a_point_mass():
    keys = _cells(10)
    d = P.paired_cell_bootstrap({k: 1.0 for k in keys}, {k: 0.5 for k in keys},
                                metric="m", arm="A", control="B", reps=REPS)
    assert d.method == "degenerate"
    assert d.ci_lo == d.ci_hi == pytest.approx(0.5)
    # the collapsed interval is REPORTED in full ...
    assert d.point == pytest.approx(0.5) and d.se == 0.0


def test_a_degenerate_resample_establishes_nothing_and_cannot_promote():
    """THE bug: ``ci_lo == ci_hi == point`` with ``se = 0`` read as a decisive win.

    Nothing downstream could tell a point-mass resample apart from a genuine
    zero-width CI that excluded the null -- and arithmetically it clears
    ``ci_lo > 0`` more comfortably than any real interval, so a resample that saw
    ONE distinct value was the strongest evidence in the slate.
    """
    keys = _cells(10)
    d = P.paired_cell_bootstrap({k: 1.0 for k in keys}, {k: 0.5 for k in keys},
                                metric="m", arm="A", control="B", reps=REPS)
    assert d.method == "degenerate" and d.degenerate is True
    assert d.measured is False
    assert not d.establishes_gain(), "a point mass is not a gain"
    assert not d.bounds_harm(10.0), "and it cannot bound a harm either"
    assert d.straddles_null(), "so it must route to a human, like every other " \
                               "comparison that established nothing"
    # the status the flatness rule reads is the SAME refusal as 'insufficient'.
    assert FA._gain_status(d) == "unresolved"
    assert FA._harm_status(d, 0.02) == "unresolved"
    assert "point mass" in FA._explain(d, "unresolved")
    # and the JSON a consumer reads says so without having to parse ``method``.
    doc = d.to_dict()
    assert doc["degenerate"] is True and doc["measured"] is False
    assert doc["establishes_gain"] is False


def test_a_degenerate_primary_leaves_condition_one_unresolved():
    """The propagation, not just the predicate: the RULE must not read it as a win."""
    keys = _cells(8)
    d = P.paired_cell_bootstrap({k: 1.0 for k in keys}, {k: 0.5 for k in keys},
                                metric=FA.PRIMARY_METRICS[0], arm="A2",
                                control="B1", reps=REPS)
    status = {"established": "pass", "established_worse": "fail",
              "straddles_null": "unresolved", "unresolved": "unresolved"}
    assert status[FA._gain_status(d)] == "unresolved"     # never 'pass'


def test_only_cells_present_in_both_arms_are_paired():
    arm = {k: 1.0 for k in _cells(10)}
    ctl = {k: 0.0 for k in _cells(6)}
    d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                reps=REPS)
    assert d.n_cells == 6 and d.n_dropped == 4


def test_paired_bootstrap_is_deterministic_and_order_free():
    keys = _cells(14)
    rng = np.random.default_rng(1)
    arm = {k: float(v) for k, v in zip(keys, rng.normal(1.0, 0.1, len(keys)))}
    ctl = {k: float(v) for k, v in zip(keys, rng.normal(0.9, 0.1, len(keys)))}
    a = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                reps=REPS, seed=7)
    b = P.paired_cell_bootstrap(dict(reversed(list(arm.items()))),
                                dict(reversed(list(ctl.items()))),
                                metric="m", arm="A", control="B",
                                reps=REPS, seed=7)
    assert (a.point, a.ci_lo, a.ci_hi) == (b.point, b.ci_lo, b.ci_hi)


def test_mde_is_pre_disclosed_from_the_paired_se():
    rng = np.random.default_rng(2)
    keys = _cells(20)
    arm = {k: float(v) for k, v in zip(keys, rng.normal(0.5, 0.05, len(keys)))}
    ctl = {k: 0.5 for k in keys}
    d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                reps=2000)
    assert d.mde() > 0
    assert d.mde() > d.se, "MDE at 80% power exceeds one SE"


def test_bca_intervals_cover_the_truth_at_about_the_nominal_rate():
    """A coverage simulation, because an interval that is merely *computed* is not
    an interval.  20 clusters, known gain, 150 replications."""
    rng = np.random.default_rng(0)
    trials, covered = 150, 0
    true_gain = 0.05
    for t in range(trials):
        g = rng.normal(true_gain, 0.08, 20)
        arm = {f"c{i}": float(v) for i, v in enumerate(g)}
        ctl = {f"c{i}": 0.0 for i in range(20)}
        d = P.paired_cell_bootstrap(arm, ctl, metric="m", arm="A", control="B",
                                    reps=600, seed=t, aggregate="mean")
        covered += int(d.ci_lo <= true_gain <= d.ci_hi)
    rate = covered / trials
    assert 0.88 <= rate <= 0.99, f"nominal 95%, measured {rate:.3f}"


def test_paired_from_arrays_requires_aligned_rows():
    with pytest.raises(ValueError, match="pairing"):
        P.paired_from_arrays(np.zeros(4), np.zeros(3), np.array(["a"] * 3),
                             metric="m", arm="A", control="B")


def test_paired_from_arrays_aggregates_rows_then_pairs_cells():
    cells = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10)
    ctl = np.ones(40)
    arm = ctl - 0.1
    d = P.paired_from_arrays(arm, ctl, cells, metric="m", arm="A", control="B",
                             higher_is_better=False, reps=REPS)
    assert d.n_cells == 4 and d.point == pytest.approx(0.1)


# =========================================================================== #
# the metric set (section 8.2)
# =========================================================================== #
def _cellular(rng, n_cells=6, per=30, sd=0.05):
    cells, true = [], []
    for c in range(n_cells):
        cells += [f"c{c:02d}"] * per
        true += list(1.20 + 0.04 * c + rng.normal(0, sd, per))
    return np.asarray(cells), np.asarray(true, dtype=float)


def test_regret_is_zero_for_a_perfect_predictor_and_positive_for_a_blind_one():
    rng = np.random.default_rng(0)
    cells, true = _cellular(rng, n_cells=12, per=60)
    perfect = FM.regret_at_k(true, true, cells, k=8)
    assert perfect and all(v == 0.0 for v in perfect.values())
    blind = FM.regret_at_k(rng.normal(0, 1, len(true)), true, cells, k=8)
    assert np.mean(list(blind.values())) > 0
    assert sum(v > 0 for v in blind.values()) > len(blind) // 2


def test_regret_is_zero_inflated_so_the_registry_aggregates_it_with_the_mean():
    """Pins the reason M0 does not use the median (see ab_paired's docstring)."""
    rng = np.random.default_rng(11)
    cells, true = _cellular(rng, n_cells=12, per=60)
    noisy = FM.regret_at_k(true + rng.normal(0, 0.06, len(true)), true, cells, k=8)
    assert np.median(list(noisy.values())) == 0.0, (
        "a competent model's per-cell regret median is 0 by construction")
    assert np.mean(list(noisy.values())) > 0
    assert FM.METRICS_BY_KEY["M0_regret8_node_peak"].aggregate == "mean"
    assert FM.METRICS_BY_KEY["M2_flat_tercile_rho_node_peak"].aggregate == "median"


def test_regret_is_in_physical_units():
    cells = np.array(["c0"] * 12)
    true = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1])
    # a predictor that ranks the cell exactly backwards keeps only the worst 8
    r = FM.regret_at_k(-true, true, cells, k=8)
    assert r["c0"] == pytest.approx(1.4 - 1.0)


def test_normalized_p_at_k_is_one_for_chance_and_larger_for_skill():
    rng = np.random.default_rng(3)
    cells, true = _cellular(rng, n_cells=8, per=32)
    perfect = FM.normalized_precision_at_k(true, true, cells, k=8)
    assert all(v == pytest.approx(32 / 8) for v in perfect.values())
    blind = FM.normalized_precision_at_k(rng.normal(0, 1, len(true)), true, cells,
                                         k=8)
    assert 0.0 <= np.median(list(blind.values())) <= 2.0


def test_flat_tercile_rho_sees_skill_the_whole_cell_statistic_hides():
    """A predictor that orders the cell but scrambles the flat end."""
    rng = np.random.default_rng(4)
    cells, true = _cellular(rng, n_cells=6, per=30)
    pred = true.copy()
    for c in sorted(set(cells.tolist())):
        idx = np.flatnonzero(cells == c)
        cut = np.quantile(true[idx], 1.0 / 3.0)
        flat = idx[true[idx] <= cut]
        # a permutation WITHIN the flat tercile: the cell-wide ordering is
        # untouched (those rows are still the lowest), only the decisions the
        # search actually makes are destroyed
        pred[flat] = rng.permutation(true[flat])
    whole = FM.cell_rho(pred, true, cells)
    tercile = FM.flat_tercile_rho(pred, true, cells)
    assert np.median(list(whole.values())) > 0.90
    assert np.median(list(tercile.values())) < 0.5


def test_sign_hit_band_only_counts_the_operating_scale():
    cells = np.array(["c0"] * 40)
    rng = np.random.default_rng(5)
    true = np.sort(rng.uniform(0, 1, 40))
    # correct on far pairs, inverted on near ones
    pred = true + np.where(np.arange(40) % 2 == 0, 0.30, -0.30) * 0.0
    pred = true.copy()
    pred[::2] += 0.05
    pred[1::2] -= 0.05
    near = FM.sign_hit_band(pred, true, cells, band=0.02, min_pairs=5)
    wide = FM.sign_hit_band(pred, true, cells, band=10.0, min_pairs=5)
    assert near["c0"] < wide["c0"]


def test_every_registered_metric_declares_a_direction_and_a_role():
    for m in FM.PRE_REGISTERED_METRICS:
        assert m.role in ("primary", "support", "harm", "report")
        assert isinstance(m.higher_is_better, bool)
        assert m.key in FM.METRICS_BY_KEY
    assert set(FA.HARM_MARGINS) <= set(FM.METRICS_BY_KEY)
    for key in FA.PRIMARY_METRICS:
        assert FM.METRICS_BY_KEY[key].role == "primary"


# =========================================================================== #
# the arena: the control is not optional (section 8.4)
# =========================================================================== #
def _arena(spec, *, n_cells=12, per=60, seed=0, control="B1", incumbent=None,
           production=None):
    """Build an arena from a per-cell {arm: 'exact'|sigma} specification.

    ``spec`` maps an arm label to either a float (prediction noise, same in every
    cell) or a list of per-cell values, so a fixture can make an arm win some
    cells and lose others deterministically.
    """
    rng = np.random.default_rng(seed)
    cells, true = _cellular(rng, n_cells=n_cells, per=per)
    truth = {"node_peak": true,
             "map_cov": true * 0.05,
             "f_r": true * 1.1,
             "f_q": true * 1.2,
             "cyclen": true * 300.0}
    preds: dict[str, dict[str, np.ndarray]] = {}
    order = sorted(set(cells.tolist()))
    for arm, sigmas in spec.items():
        if not isinstance(sigmas, (list, tuple)):
            sigmas = [float(sigmas)] * n_cells
        noise = np.zeros(len(true))
        for c, s in zip(order, sigmas):
            idx = np.flatnonzero(cells == c)
            noise[idx] = rng.normal(0, float(s), len(idx)) if s > 0 else 0.0
        preds[arm] = {t: v + noise * (1.0 if t == "node_peak" else
                                      (0.05 if t == "map_cov" else 1.0))
                      for t, v in truth.items()}
    return FA.FlatArena(
        cells=cells, truth=truth, preds=preds, control=control,
        incumbent=incumbent, frozen_cells=tuple(order),
        production=(np.ones(len(true), dtype=bool) if production is None
                    else production),
        provenance={"schema": "test"})


def test_an_arena_without_a_control_cannot_be_built():
    with pytest.raises(FA.ControlMissingError, match=r"8\.4"):
        _arena({"A": 0.02})                        # no B1 in the arm set


def test_an_empty_control_label_is_refused():
    with pytest.raises(FA.ControlMissingError):
        FA.FlatArena(cells=np.array(["c"]), truth={"node_peak": np.array([1.0])},
                     preds={"A": {"node_peak": np.array([1.0])}}, control="")


def test_arms_must_be_scored_on_the_same_rows():
    with pytest.raises(ValueError, match="paired"):
        FA.FlatArena(cells=np.array(["c", "c"]),
                     truth={"node_peak": np.array([1.0, 1.1])},
                     preds={"B1": {"node_peak": np.array([1.0, 1.1])},
                            "A": {"node_peak": np.array([1.0])}},
                     control="B1")


def test_judging_the_control_against_itself_is_refused():
    arena = _arena({"B1": 0.05, "A": 0.02})
    with pytest.raises(ValueError):
        FA.judge_arm(arena, "B1", reps=REPS)


# =========================================================================== #
# the decision rule (sections 8.5 / 8.6)
# =========================================================================== #
_GATE = {"pass": True, "worst_drop": 0.0, "epsilon": 0.134, "n_checks": 40}


def test_a_uniformly_better_arm_promotes():
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=11)
    j = FA.judge_arm(arena, "A", reps=REPS, gate=_GATE)
    assert j["verdict"] == FA.PROMOTE, j["reason"]
    assert all(c["status"] == "pass" for c in j["conditions"])
    assert j["falsification"]["triggered"] is False


def test_point_estimate_favours_a_but_the_paired_ci_refuses_to_promote():
    """THE refusal.  A wins 7 cells outright and loses 5 outright.

    The median paired gain is positive -- on a point comparison A is "better" --
    but a resample of 12 cells lands on a losing median far more often than 2.5%
    of the time, so the interval includes the null.  The apparatus must escalate,
    not promote.
    """
    n = 12
    a_sig = [0.0] * 7 + [0.20] * 5          # exact where A wins, blind where it loses
    b_sig = [0.20] * 7 + [0.0] * 5
    arena = _arena({"A": a_sig, "B1": b_sig}, n_cells=n, seed=21)

    spec = FM.METRICS_BY_KEY[FA.PRIMARY_METRICS[0]]
    d = FA.paired_metric(arena, "A", spec, reps=2000, seed=5)
    assert d.point > 0, "fixture premise: the point estimate favours A"
    assert not d.establishes_gain(), "fixture premise: the interval does not"
    assert d.favours_arm_on_points_only()

    j = FA.judge_arm(arena, "A", reps=2000, seed=5, gate=_GATE)
    assert j["verdict"] == FA.ESCALATE
    assert j["verdict"] != FA.PROMOTE
    assert "does not exclude the null" in j["reason"]
    assert "human" in j["reason"]
    cond1 = [c for c in j["conditions"] if c["id"] == "1"][0]
    assert cond1["status"] == "unresolved"
    assert cond1["evidence"] == "straddles_null"


def test_an_established_loss_is_rejected_outright():
    arena = _arena({"B1": 0.0, "A": 0.08}, seed=13)
    j = FA.judge_arm(arena, "A", reps=REPS, gate=_GATE)
    assert j["verdict"] == FA.REJECT
    assert any(c["status"] == "fail" for c in j["conditions"])


def test_the_falsification_condition_is_recorded_whether_or_not_it_fires():
    win = FA.judge_arm(_arena({"B1": 0.06, "A": 0.0}, seed=11), "A", reps=REPS,
                       gate=_GATE)
    lose = FA.judge_arm(_arena({"B1": 0.0, "A": 0.0}, seed=11), "A", reps=REPS,
                        gate=_GATE)
    for j in (win, lose):
        assert j["falsification"]["condition"]
        assert "map_cov_weight" in j["falsification"]["consequence"]
        assert "triggered" in j["falsification"]
    assert lose["falsification"]["triggered"] is True
    assert win["falsification"]["triggered"] is False


def test_a_missing_extended_gate_cannot_pass_condition_four():
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=11)
    j = FA.judge_arm(arena, "A", reps=REPS, gate=None)
    c4 = [c for c in j["conditions"] if c["id"] == "4"][0]
    assert c4["status"] == "unresolved"
    assert j["verdict"] != FA.PROMOTE


def test_a_failed_extended_gate_rejects_however_good_the_primary():
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=11)
    j = FA.judge_arm(arena, "A", reps=REPS,
                     gate={"pass": False, "worst_drop": 0.4, "epsilon": 0.134})
    assert j["verdict"] == FA.REJECT


def test_no_production_stratum_leaves_condition_six_unresolved():
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=11,
                   production=np.zeros(12 * 60, dtype=bool))
    j = FA.judge_arm(arena, "A", reps=REPS, gate=_GATE)
    c6 = [c for c in j["conditions"] if c["id"] == "6"][0]
    assert c6["status"] == "unresolved"
    assert "proposed" in c6["detail"]
    assert j["verdict"] != FA.PROMOTE


def test_delta75_is_reported_and_marked_as_not_deciding():
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=11)
    rep = FA.reported_effective_resolution(arena, "A")
    assert rep["decides"] is False
    assert "node_peak" in rep
    j = FA.judge_arm(arena, "A", reps=REPS, gate=_GATE, reported=rep)
    assert j["reported_not_deciding"]["decides"] is False


# =========================================================================== #
# the slate (sections 8.4 / 8.5)
# =========================================================================== #
def test_the_slate_separates_the_data_effect_from_the_loss_effect():
    arena = _arena({"B0": 0.10, "B1": 0.06, "A": 0.0}, seed=17, incumbent="B0")
    slate = FA.judge_all(arena, reps=REPS, gates={"A": _GATE})
    tw = slate["three_way"]
    assert tw["incumbent"] == "B0" and tw["control"] == "B1"
    key = FA.PRIMARY_METRICS[0]
    assert key in tw["metrics"]
    assert tw["metrics"][key]["control_minus_incumbent"]["arm"] == "B1"
    assert tw["metrics"][key]["mde80_from_this_se"] is not None
    assert "LOSS effect" in tw["metrics"][key]["reading"]


def test_without_an_incumbent_the_slate_says_it_can_attribute_nothing():
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=17)
    slate = FA.judge_all(arena, reps=REPS, gates={"A": _GATE})
    assert slate["three_way"]["metrics"] == {}
    assert "attributes nothing" in slate["three_way"]["note"]


def test_two_co_qualifiers_that_cannot_be_separated_escalate():
    arena = _arena({"B1": 0.08, "A": 0.0, "A2": 0.0}, seed=19)
    slate = FA.judge_all(arena, reps=REPS,
                         gates={"A": _GATE, "A2": _GATE})
    assert slate["verdict"] == FA.ESCALATE
    assert set(slate["candidates"]) == {"A", "A2"}
    assert "human" in slate["reason"]
    assert slate["head_to_head"]["separated"] is False


def test_the_slate_renders_for_every_outcome():
    for spec, gates in ((({"B1": 0.06, "A": 0.0}), {"A": _GATE}),
                        (({"B1": 0.0, "A": 0.06}), {"A": _GATE}),
                        (({"B1": 0.0, "A": 0.0}), {})):
        slate = FA.judge_all(_arena(spec, seed=23), reps=REPS, gates=gates)
        text = FA.render_slate(slate)
        assert "verdict :" in text
        json.dumps(slate)                    # the artifact must be serializable


def test_paired_block_carries_the_control_label_for_re_checking():
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=11)
    block = FA.paired_block(FA.judge_arm(arena, "A", reps=REPS, gate=_GATE))
    assert block["control"] == "B1"
    assert FA.PRIMARY_METRICS[0] in block["metrics"]
    assert block["conditions"] and block["falsification"]["condition"]


# =========================================================================== #
# the C2 slice and the stale-split refusal (section 7.1)
# =========================================================================== #
def _store(n_train=40, n_new=40):
    rows = []
    for i in range(n_train):
        rows.append({"record_id": f"t{i:03d}", "dataset": "A", "campaign": None,
                     "generator": "production", "parent_record_id": None,
                     "case_pair": f"pair{i%8}", "feed": 121,
                     "e_core": 5.50 + 0.001 * i, "pattern": f"pat_t{i}",
                     "converged": True, "node_peak": 1.4, "map_cov": 0.05})
    for i in range(n_new):
        rows.append({"record_id": f"n{i:03d}", "dataset": "P", "campaign": None,
                     "generator": "production", "parent_record_id": None,
                     "case_pair": f"newpair{i%8}", "feed": 121,
                     "e_core": 5.50 + 0.001 * i, "pattern": f"pat_n{i}",
                     "converged": True, "node_peak": 1.4, "map_cov": 0.05})
    return pd.DataFrame(rows)


def _fresh_manifest(df, train_ids, val_ids):
    return SplitManifest(
        name="S2", kind="flat_cell_holdout", seed=0,
        train_ids=list(train_ids), val_ids=list(val_ids), status="ok",
        predicate={"program": C2.FLATNESS_PROGRAM},
        groups={"store_fingerprint": C2.store_fingerprint(df)})


def test_a_split_that_does_not_declare_the_program_is_stale():
    df = _store()
    m = SplitManifest(name="S2", kind="leave_pair", seed=0,
                      train_ids=[f"t{i:03d}" for i in range(40)],
                      val_ids=[], status="ok")
    audit = C2.audit_split(m, df)
    assert audit["stale"] is True
    assert any("program" in r for r in audit["reasons"])
    assert any("outside the manifest" in r for r in audit["reasons"])
    assert C2.REGEN_CMD in C2.render_audit(audit)


def test_build_c2_refuses_on_a_stale_split_and_names_the_remedy():
    df = _store()
    m = SplitManifest(name="S2", kind="leave_pair", seed=0,
                      train_ids=[f"t{i:03d}" for i in range(40)], val_ids=[])
    with pytest.raises(C2.SplitStaleError) as exc:
        C2.build_c2(df, m)
    assert "STALE" in str(exc.value)
    assert C2.REGEN_CMD in str(exc.value)


def test_allow_stale_still_stamps_the_slice_as_stale():
    df = _store()
    m = SplitManifest(name="S2", kind="leave_pair", seed=0,
                      train_ids=[f"t{i:03d}" for i in range(40)], val_ids=[])
    c2 = C2.build_c2(df, m, allow_stale=True, min_cell_rows=2)
    assert c2.provenance["stale_split"] is True
    assert c2.audit["stale"] is True
    assert c2.provenance["source"] == "foldC"


def test_c2_drops_lineage_symmetry_and_duplicate_rows():
    df = _store(n_train=40, n_new=0)
    extra = [
        # a child of a training row: fold C by construction, contaminated in fact
        {"record_id": "x_child", "dataset": "P", "campaign": None,
         "generator": "production", "parent_record_id": "t000",
         "case_pair": "cleanA", "feed": 121, "e_core": 5.55,
         "pattern": "pat_child", "converged": True, "node_peak": 1.4,
         "map_cov": 0.05},
        # a grandchild -- the lineage rule must be transitive
        {"record_id": "x_grand", "dataset": "P", "campaign": None,
         "generator": "production", "parent_record_id": "x_child",
         "case_pair": "cleanB", "feed": 121, "e_core": 5.55,
         "pattern": "pat_grand", "converged": True, "node_peak": 1.4,
         "map_cov": 0.05},
        # the transpose partner of a training row
        {"record_id": "x_sym", "dataset": "P", "campaign": None,
         "generator": "production", "parent_record_id": None,
         "case_pair": "pair0", "feed": 121, "e_core": 5.55,
         "pattern": "pat_sym", "converged": True, "node_peak": 1.4,
         "map_cov": 0.05},
        # an exact repeat of a training pattern
        {"record_id": "x_dup", "dataset": "P", "campaign": None,
         "generator": "production", "parent_record_id": None,
         "case_pair": "cleanC", "feed": 121, "e_core": 5.55,
         "pattern": "pat_t5", "converged": True, "node_peak": 1.4,
         "map_cov": 0.05},
        # unlabelled: no flatness target
        {"record_id": "x_nolabel", "dataset": "P", "campaign": None,
         "generator": "production", "parent_record_id": None,
         "case_pair": "cleanD", "feed": 121, "e_core": 5.55,
         "pattern": "pat_nolabel", "converged": True, "node_peak": None,
         "map_cov": None},
        # clean
        {"record_id": "x_ok", "dataset": "P", "campaign": None,
         "generator": "production", "parent_record_id": None,
         "case_pair": "cleanE", "feed": 121, "e_core": 5.55,
         "pattern": "pat_ok", "converged": True, "node_peak": 1.4,
         "map_cov": 0.05},
    ]
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    m = _fresh_manifest(df, [f"t{i:03d}" for i in range(40)],
                        [r["record_id"] for r in extra])
    c2 = C2.build_c2(df, m, min_cell_rows=1)
    kept = set(c2.record_ids)
    assert kept == {"x_ok"}
    d = c2.provenance["dropped"]
    assert d["lineage"] == 2, "the grandchild must be caught transitively"
    assert d["symmetry_pair"] == 1
    assert d["duplicate_pattern"] == 1
    assert d["missing_label"] == 1
    assert c2.provenance["source"] == "val"
    assert c2.provenance["n_source"] == 6


def test_a_program_split_judges_its_val_fold_and_a_legacy_one_judges_fold_c():
    """The judging rows come from different folds for different split kinds."""
    df = _store(n_train=20, n_new=20)
    train = [f"t{i:03d}" for i in range(20)]
    val = [f"n{i:03d}" for i in range(20)]
    prog = C2.build_c2(df, _fresh_manifest(df, train, val), min_cell_rows=1)
    assert prog.provenance["source"] == "val"
    assert set(prog.record_ids) == set(val)

    legacy = SplitManifest(name="S2", kind="leave_pair", seed=0,
                           train_ids=train, val_ids=[], status="ok")
    old = C2.build_c2(df, legacy, allow_stale=True, min_cell_rows=1)
    assert old.provenance["source"] == "foldC"
    assert set(old.record_ids) == set(val), "fold C is the complement here"


def test_c2_records_the_provenance_stratum_and_freezes_its_cells():
    df = _store(n_train=20, n_new=0)
    rows = []
    for i in range(20):
        rows.append({"record_id": f"p{i:03d}", "dataset": "P",
                     "campaign": "alsearch_x" if i % 2 else None,
                     "generator": "alsearch_x" if i % 2 else "production",
                     "parent_record_id": None, "case_pair": f"q{i}", "feed": 121,
                     "e_core": 5.60, "pattern": f"pp{i}", "converged": True,
                     "node_peak": 1.4, "map_cov": 0.05})
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    m = _fresh_manifest(df, [f"t{i:03d}" for i in range(20)],
                        [r["record_id"] for r in rows])
    c2 = C2.build_c2(df, m, min_cell_rows=4)
    assert c2.provenance["n_proposed"] > 0 and c2.provenance["n_production"] > 0
    assert 0.0 < c2.provenance["production_frac"] < 1.0
    assert c2.frozen_cells and len(c2.provenance["cell_manifest_sha1"]) == 16
    assert c2.production_mask().sum() == c2.provenance["n_production"] + 0


def test_mark_stale_preserves_the_ids_but_not_the_healthy_look(tmp_path):
    df = _store()
    m = SplitManifest(name="S2", kind="leave_pair", seed=0,
                      train_ids=[f"t{i:03d}" for i in range(40)], val_ids=[])
    p = tmp_path / "S2.json"
    m.to_json(p)
    audit = C2.audit_split(m, df)
    back = C2.mark_stale(p, audit)
    assert back.status == "stale"
    assert back.train_ids == m.train_ids and back.val_ids == m.val_ids
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["status"] == "stale"
    assert on_disk["groups"]["staleness_audit"]["reasons"]


# =========================================================================== #
# regeneration (section 7.2)
# =========================================================================== #
def test_the_regenerated_split_declares_the_program_and_holds_out_whole_cells():
    rows = []
    for c in range(10):
        for i in range(12):
            rows.append({"record_id": f"r{c:02d}_{i:02d}", "dataset": "P",
                         "campaign": None, "generator": "production",
                         "parent_record_id": None, "case_pair": None,
                         "feed": 121, "e_core": 5.40 + 0.05 * c,
                         "pattern": f"p{c}_{i}", "converged": True,
                         "node_peak": 1.4, "map_cov": 0.05})
    df = pd.DataFrame(rows)
    m = C2.make_flat_cell_split(df, seed=0, e_core_holdout_from=5.75,
                                min_cell_rows=4)
    assert m.predicate["program"] == C2.FLATNESS_PROGRAM
    assert C2.audit_split(m, df)["stale"] is False

    from lpopt.model.folds import cell_key
    cell_of = dict(zip(df["record_id"].astype(str), cell_key(df).astype(str)))
    train_cells = {cell_of[i] for i in m.train_ids}
    val_cells = {cell_of[i] for i in m.val_ids}
    assert train_cells and val_cells
    assert not (train_cells & val_cells), "no cell may straddle the holdout"
    assert m.groups["extrapolation_cells"], "high e_core cells are held out"


def test_cell_membership_is_invariant_to_adding_rows():
    def frame(per):
        rows = []
        for c in range(8):
            for i in range(per):
                rows.append({"record_id": f"r{c:02d}_{i:03d}", "dataset": "P",
                             "campaign": None, "generator": "production",
                             "parent_record_id": None, "case_pair": None,
                             "feed": 121, "e_core": 5.40 + 0.05 * c,
                             "pattern": f"p{c}_{i}", "converged": True,
                             "node_peak": 1.4, "map_cov": 0.05})
        return pd.DataFrame(rows)

    a = C2.make_flat_cell_split(frame(10), seed=0, min_cell_rows=4)
    b = C2.make_flat_cell_split(frame(25), seed=0, min_cell_rows=4)
    assert a.groups["in_domain_val_cells"] == b.groups["in_domain_val_cells"]
    assert a.groups["extrapolation_cells"] == b.groups["extrapolation_cells"]


# =========================================================================== #
# the shipped artifact
# =========================================================================== #
_SPLIT = Path("data/splits/S2.json")
_STORE = Path("data/store/records.parquet")


@pytest.mark.skipif(not (_SPLIT.exists() and _STORE.exists()),
                    reason="needs the real store and split")
def test_the_shipped_S2_is_flagged_stale_against_the_live_store():
    """Regression pin: the manifest the flatness A/B would reach for is not it.

    S2.json is the legacy leave-pair-out split.  It is well-formed, so nothing
    but an explicit audit can tell -- which is why this is a test and not a
    comment.
    """
    df = pd.read_parquet(_STORE)
    audit = C2.audit_split(SplitManifest.from_json(_SPLIT), df)
    assert audit["stale"] is True
    assert audit["declared_program"] is None
    assert audit["n_store_rows_outside_manifest"] > 0
    sidecar = _SPLIT.with_suffix(".audit.json")
    assert sidecar.exists(), (
        "the staleness verdict must be recorded next to the manifest; run "
        "python -m lpopt.tools.audit_c2_split --invalidate")


# =========================================================================== #
# F_r DEFERRAL (user decision 2026-07-26) — the offline A/B is a promotion
# surface too, and it re-arms from the SAME switch.
# =========================================================================== #
def _fr_broken_arena(seed=11):
    """A uniformly better arm whose ``f_r`` rank skill has been destroyed."""
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=seed)
    rng = np.random.default_rng(3)
    arena.preds["A"]["f_r"] = rng.normal(0.0, 1.0, len(arena.cells))
    return arena


def test_an_f_r_only_rank_collapse_no_longer_withholds_arm_promotion():
    """M5 (``M5_cell_rho_f_r``) let F_r veto a promotion for the same reason the
    curriculum gate did — and on the same corpus.  Report-only by default."""
    j = FA.judge_arm(_fr_broken_arena(), "A", reps=REPS, gate=_GATE)

    c5 = [c for c in j["conditions"] if c["id"] == "5"][0]
    assert c5["metrics"]["M5_cell_rho_f_r"]["enforced"] is False
    assert c5["metrics"]["M5_cell_rho_f_r"]["status"] == "violated"   # still SCORED
    assert c5["report_only"] == ["M5_cell_rho_f_r"]
    assert c5["status"] != "fail"
    assert "f_r" in c5["detail"] and "not enforced" in c5["detail"].lower()
    assert j["verdict"] == FA.PROMOTE, j["reason"]
    assert j["fr_guard"]["enforced"] is False
    assert j["fr_guard"]["knob"] == "[curriculum] gate_noreg_fr_guard_enabled"


def test_the_same_switch_re_arms_the_offline_ab_f_r_condition():
    j = FA.judge_arm(_fr_broken_arena(), "A", reps=REPS, gate=_GATE,
                     fr_guarded=True)

    c5 = [c for c in j["conditions"] if c["id"] == "5"][0]
    assert c5["metrics"]["M5_cell_rho_f_r"]["enforced"] is True
    assert c5["status"] == "fail"
    assert j["verdict"] == FA.REJECT
    assert j["fr_guard"]["enforced"] is True


def test_f_q_and_cyclen_keep_their_M5_teeth():
    """The deferral is one axis wide here too."""
    arena = _arena({"B1": 0.06, "A": 0.0}, seed=11)
    rng = np.random.default_rng(4)
    arena.preds["A"]["f_q"] = rng.normal(0.0, 1.0, len(arena.cells))
    j = FA.judge_arm(arena, "A", reps=REPS, gate=_GATE)
    assert j["verdict"] == FA.REJECT
