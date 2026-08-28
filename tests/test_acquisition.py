"""Acquisition math, trust-region gate, and wave composition (plan sec. 4.6 / 6.2)."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest
from scipy.stats import norm

from lpopt.config import TrustRegionConfig
from lpopt.search import acquisition as acq
from lpopt.search.construct import Candidate, CaseContext, candidate_record_id
from lpopt.search.genome import mutate, random_genome
from lpopt.vendor.masterrl.reward import ConstraintConfig
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction


def _pred(mean, std):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return SurrogatePrediction(mean, std.copy(), std.copy())


# --------------------------------------------------------------------------- #
# p_feasible vs scipy norm.cdf
# --------------------------------------------------------------------------- #
def test_p_feasible_matches_scipy_normcdf():
    c = ConstraintConfig(f_r_limit=1.55, cbc_limit=1550.0, f_q_limit=2.41, ao_abs_limit=0.30)
    # 7-col surrogate layout: 0 F_r, 1 CBC, 2 F_q, 3 cyclen, 4 AO, 5/6 burnups.
    mean = np.array([
        [1.50, 1500.0, 2.30, 620.0, 0.20, np.nan, np.nan],
        [1.60, 1600.0, 2.50, 610.0, 0.35, np.nan, np.nan],
    ])
    std = np.array([
        [0.02, 20.0, 0.05, 3.0, 0.02, np.nan, np.nan],
        [0.01, 10.0, 0.03, 2.0, 0.01, np.nan, np.nan],
    ])
    got = acq.p_feasible(_pred(mean, std), c)

    def ref(row):
        p = 1.0
        for col, lim in ((0, c.f_r_limit), (1, c.cbc_limit), (2, c.f_q_limit), (4, c.ao_abs_limit)):
            p *= float(norm.cdf((lim - mean[row, col]) / std[row, col]))
        return p

    assert got == pytest.approx([ref(0), ref(1)], rel=1e-9)


def test_p_feasible_unknown_axis_and_convergence():
    c = ConstraintConfig()
    mean = np.array([[1.50, 1500.0, 2.30, 620.0, 0.20, np.nan, np.nan]])
    std = np.array([[np.nan, 20.0, 0.05, 3.0, 0.02, np.nan, np.nan]])  # F_r unknown
    p = acq.p_feasible(_pred(mean, std), c)
    # unknown F_r axis contributes exactly unknown_axis_probability (0.5).
    ref = c.unknown_axis_probability
    for col, lim in ((1, c.cbc_limit), (2, c.f_q_limit), (4, c.ao_abs_limit)):
        ref *= float(norm.cdf((lim - mean[0, col]) / std[0, col]))
    assert p[0] == pytest.approx(ref, rel=1e-9)
    # convergence probability multiplies through.
    p2 = acq.p_feasible(_pred(mean, std), c, convergence=[0.5])
    assert p2[0] == pytest.approx(0.5 * p[0], rel=1e-9)


def test_p_feasible_zero_std_indicator():
    c = ConstraintConfig()
    mean = np.array([[1.50, 1500.0, 2.30, 620.0, 0.20, np.nan, np.nan]])
    std = np.array([[0.0, 20.0, 0.05, 3.0, 0.02, np.nan, np.nan]])  # F_r certain, safe
    assert acq.p_feasible(_pred(mean, std), c)[0] > 0.0
    mean_bad = mean.copy(); mean_bad[0, 0] = 1.90  # F_r certainly over limit
    assert acq.p_feasible(_pred(mean_bad, std), c)[0] == 0.0


# --------------------------------------------------------------------------- #
# trust region (plan sec. 6.2)
# --------------------------------------------------------------------------- #
def _ctx():
    return CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)


def test_trust_region_hard_gate_offgrid():
    tr = acq.TrustRegion(
        TrustRegionConfig(), supported=set(), campaign_feed=121, campaign_e_core=5.2
    )
    # campaign cell is always in-region; a synthetic feed-117 candidate is not.
    assert tr.in_region(121, 5.2) is True
    assert tr.in_region(117, 5.2) is False
    # feed 117 is one reachable step (=4) from 121 -> frontier, with sigma inflation.
    assert tr.is_frontier(117, 5.2) is True
    assert tr.sigma_scale(117, 5.2) == pytest.approx(1.5)
    assert tr.sigma_scale(121, 5.2) == pytest.approx(1.0)


def test_trust_region_growth_by_labels():
    cfg = TrustRegionConfig(promote_after=16)
    tr = acq.TrustRegion(cfg, supported=set(), campaign_feed=121, campaign_e_core=5.2)
    assert tr.in_region(125, 5.2) is False
    for _ in range(15):
        tr.observe(125, 5.2)
    assert tr.in_region(125, 5.2) is False   # not yet promoted
    tr.observe(125, 5.2)                      # 16th label promotes the bin
    assert tr.in_region(125, 5.2) is True


class _ConstModel:
    """Predicts a fixed near-feasible FOM for any pattern (7-col layout)."""

    def predict(self, patterns, case, cell=0.0):
        n = len(patterns)
        mean = np.tile([1.52, 1500.0, 2.30, 620.0, 0.20, np.nan, np.nan], (n, 1))
        std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (n, 1))
        return SurrogatePrediction(mean.astype(float), std.copy(), std.copy())

    def predict_convergence(self, patterns, case, cell=0.0):
        return np.ones(len(patterns))


def test_score_pool_hard_gates_offgrid_candidate():
    ctx = _ctx()
    rng = random.Random(0)
    model = _ConstModel()
    constraints = ConstraintConfig(objective_mode="target_cycle", cycle_target_efpd=625.0)
    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    # one on-grid feed-121 candidate and one synthetic off-grid feed-117 candidate.
    g121 = random_genome(rng, "K1_K2", 30)
    g117 = random_genome(rng, "K1_K2", 29)
    c121 = Candidate(g121.to_pattern(), g121, "random", None,
                     candidate_record_id(g121.to_pattern(), ctx), 5.2)
    c117 = Candidate(g117.to_pattern(), g117, "random", None,
                     candidate_record_id(g117.to_pattern(), ctx), 5.2)
    boot = model.predict([c121.pattern], ctx.case_key, 5.2)
    rm = acq.build_reward_model(ctx, [c121.pattern], boot, constraints)
    scored = acq.score_pool(model, ctx, [c121, c117], rm, constraints, tr)
    assert scored.in_region.tolist() == [True, False]
    assert scored.p_feas[0] > 0.0
    assert scored.p_feas[1] == 0.0            # hard-gated off-grid candidate
    assert scored.acq[1] == 0.0
    # out-of-region candidate is -inf on the exploit score (sorts last / never
    # adopted); the in-region candidate carries a finite exploit score.
    assert scored.exploit[1] == -np.inf
    assert np.isfinite(scored.exploit[0])


# --------------------------------------------------------------------------- #
# wave composition (5/2/1 + Hamming)
# --------------------------------------------------------------------------- #
def _synthetic_scored(n=60, seed=1):
    rng = random.Random(seed)
    ctx = _ctx()
    cands, patterns = [], []
    seen = set()
    while len(cands) < n:
        g = random_genome(rng, "K1_K2", 30)
        pat = g.to_pattern()
        if pat.digest in seen:
            continue
        seen.add(pat.digest)
        rid = candidate_record_id(pat, ctx)
        cands.append(Candidate(pat, g, "random", None, rid, 5.2))
        patterns.append(pat)
    mean = np.tile([1.52, 1500.0, 2.30, 620.0, 0.20, np.nan, np.nan], (n, 1)).astype(float)
    std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (n, 1)).astype(float)
    return acq.ScoredPool(
        candidates=cands, mean=mean, epistemic=std.copy(), calibrated=std.copy(),
        conv=np.ones(n), p_feas=np.linspace(0.05, 0.4, n),
        acq=np.linspace(0.1, 2.0, n)[::-1].copy(),
        raw_epi=np.linspace(0.01, 0.1, n), in_region=np.ones(n, dtype=bool),
        # exploit score ascending with index (independent of raw_epi/acq gradients).
        exploit=np.linspace(-2.0, 2.0, n),
    )


def test_compose_wave_5_2_1_and_hamming():
    scored = _synthetic_scored()
    rng = random.Random(7)
    slots = acq.compose_wave(
        scored, [], rng, size=8, n_exploit=5, n_explore=2, n_control=1,
        tau=0.3, hamming_min=4,
    )
    assert len(slots) == 8
    assert Counter(s.slot for s in slots) == {"exploit": 5, "explore": 2, "control": 1}
    pats = [scored.candidates[s.index].pattern for s in slots]
    idx = list(range(len(pats)))
    assert min(pats[i].hamming(pats[j]) for i in idx for j in idx if i < j) >= 4


def test_compose_wave_reserve_exploit_only():
    scored = _synthetic_scored()
    rng = random.Random(3)
    slots = acq.compose_wave(
        scored, [], rng, size=4, n_exploit=4, n_explore=0, n_control=0,
        tau=0.3, hamming_min=4,
    )
    assert len(slots) == 4
    assert all(s.slot == "exploit" for s in slots)


# --------------------------------------------------------------------------- #
# corrected exploit vs explore ranking (the M5-pilot regression)
# --------------------------------------------------------------------------- #
class _TwoClassModel:
    """A 'good' feasible-basin FOM (low F_r, low σ) for ``good`` digests and a
    'bad' OOD FOM (high F_r, huge σ) for everything else."""

    def __init__(self, good_digests):
        self.good = set(good_digests)

    def predict(self, patterns, case, cell=0.0):
        means, stds = [], []
        for p in patterns:
            if p.digest in self.good:               # obviously-good candidate
                means.append([1.50, 1480.0, 2.30, 640.0, 0.15, np.nan, np.nan])
                stds.append([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan])
            else:                                    # bad, huge-σ candidate
                means.append([1.90, 1600.0, 2.55, 692.0, 0.20, np.nan, np.nan])
                stds.append([0.60, 420.0, 0.80, 45.0, 0.10, np.nan, np.nan])
        m = np.asarray(means, dtype=float)
        s = np.asarray(stds, dtype=float)
        return SurrogatePrediction(m, s.copy(), s.copy())

    def predict_convergence(self, patterns, case, cell=0.0):
        return np.ones(len(patterns))


def _two_class_pool(n_good=12, n_bad=12, seed=5):
    """Distinct feed-121 K1_K2 candidates split into good/bad classes."""

    rng = random.Random(seed)
    ctx = _ctx()
    cands, good = [], set()
    seen = set()
    while len(cands) < n_good + n_bad:
        g = random_genome(rng, "K1_K2", 30)
        pat = g.to_pattern()
        if pat.digest in seen:
            continue
        seen.add(pat.digest)
        rid = candidate_record_id(pat, ctx)
        cands.append(Candidate(pat, g, "random", None, rid, 5.2))
        if len(cands) <= n_good:
            good.add(pat.digest)
    return ctx, cands, good


def _score_two_class(have_feasible: bool):
    ctx, cands, good = _two_class_pool()
    model = _TwoClassModel(good)
    constraints = ConstraintConfig(
        objective_mode="target_cycle", cycle_target_efpd=625.0, cycle_tolerance_efpd=2.0
    )
    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    boot = model.predict([c.pattern for c in cands], ctx.case_key, 5.2)
    rm = acq.build_reward_model(ctx, [c.pattern for c in cands], boot, constraints)
    scored = acq.score_pool(
        model, ctx, cands, rm, constraints, tr, have_feasible=have_feasible
    )
    good_idx = [i for i, c in enumerate(cands) if c.pattern.digest in good]
    bad_idx = [i for i, c in enumerate(cands) if c.pattern.digest not in good]
    return scored, good, good_idx, bad_idx


def test_exploit_score_prefers_good_over_high_sigma():
    """The obviously-good candidate outranks the bad high-σ one on the exploit
    score (both feasibility-first and objective phase), and the reverse holds on
    raw epistemic σ — exactly the two axes exploit and explore rank on."""

    for have_feasible in (False, True):
        scored, good, good_idx, bad_idx = _score_two_class(have_feasible)
        assert min(scored.exploit[good_idx]) > max(scored.exploit[bad_idx])
        # raw epistemic σ is inverted: the bad huge-σ class dominates exploration.
        assert min(scored.raw_epi[bad_idx]) > max(scored.raw_epi[good_idx])


def _planted_pool(n=24, n_good=12, seed=8):
    """A ScoredPool with the exploit score and raw-σ deliberately anti-correlated:
    the first ``n_good`` rows have HIGH exploit / LOW σ (the feasible basin), the
    rest LOW exploit / HIGH σ (the uncertain OOD tail).  Both classes clear the
    ``p_feas >= τ/2`` explore floor so the *ranking key* — not the gate — decides
    each slot."""

    rng = random.Random(seed)
    ctx = _ctx()
    cands, seen = [], set()
    while len(cands) < n:
        g = random_genome(rng, "K1_K2", 30)
        pat = g.to_pattern()
        if pat.digest in seen:
            continue
        seen.add(pat.digest)
        cands.append(Candidate(pat, g, "random", None, candidate_record_id(pat, ctx), 5.2))
    good = np.arange(n) < n_good
    mean = np.tile([1.52, 1500.0, 2.30, 640.0, 0.20, np.nan, np.nan], (n, 1)).astype(float)
    mean[~good, 0] = 1.90                       # OOD tail predicts a high F_r
    std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (n, 1)).astype(float)
    scored = acq.ScoredPool(
        candidates=cands, mean=mean, epistemic=std.copy(), calibrated=std.copy(),
        conv=np.ones(n),
        p_feas=np.full(n, 0.40),                # all clear τ and τ/2
        acq=np.where(good, 0.05, 0.5),          # acq (wrongly) favours the OOD tail
        raw_epi=np.where(good, 0.1, 5.0),       # OOD tail dominates exploration
        in_region=np.ones(n, dtype=bool),
        exploit=np.where(good, 1.0, -1.0),      # exploit favours the feasible basin
    )
    return scored, good


def test_compose_wave_exploit_by_score_explore_by_sigma():
    """Exploit slots follow the exploit score (feasible basin); explore slots
    follow raw σ (OOD tail) — never the exploration acquisition.  Had exploit
    still ranked on ``acq`` (which here favours the OOD tail) the exploit picks
    would be the high-σ tail: exactly the M5-pilot failure this guards against."""

    scored, good = _planted_pool()
    rng = random.Random(4)
    slots = acq.compose_wave(
        scored, [], rng, size=8, n_exploit=5, n_explore=2, n_control=1,
        tau=0.30, hamming_min=1,
    )
    exploit_idx = [s.index for s in slots if s.slot == "exploit"]
    explore_idx = [s.index for s in slots if s.slot == "explore"]
    assert len(exploit_idx) == 5 and len(explore_idx) == 2
    # exploit -> the high-exploit-score feasible basin (low predicted F_r) ...
    assert all(good[i] for i in exploit_idx)
    assert max(scored.mean[exploit_idx, 0]) < 1.6
    # ... explore -> the high-σ OOD tail (the only slot that rewards uncertainty).
    assert all(not good[i] for i in explore_idx)


def test_local_search_climbs_exploit_not_uncertainty():
    """Local search adopts neighbours that improve the exploit score; a bad
    huge-σ seed must not be hill-climbed to an even higher-σ neighbour just
    because its exploration acquisition is larger."""

    scored, good, good_idx, bad_idx = _score_two_class(have_feasible=False)
    ctx, cands, gd = _two_class_pool()  # fresh identical layout for the model
    model = _TwoClassModel(good)
    constraints = ConstraintConfig(
        objective_mode="target_cycle", cycle_target_efpd=625.0, cycle_tolerance_efpd=2.0
    )
    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    boot = model.predict([c.pattern for c in scored.candidates], ctx.case_key, 5.2)
    rm = acq.build_reward_model(ctx, [c.pattern for c in scored.candidates], boot, constraints)

    from lpopt.config import LocalSearchConfig

    cfg_ls = LocalSearchConfig(top_m=8, neighbors=6, depth=2, max_predictions=200, n_moves=1)
    refined = acq.local_search(
        model, ctx, scored, rm, constraints, tr, cfg_ls, random.Random(1),
        set(), have_feasible=False,
    )
    # any local child that was adopted must not be worse than its seed on the
    # exploit score, and the best exploit score never decreases.
    assert np.max(refined.exploit) >= np.max(scored.exploit) - 1e-9


# --------------------------------------------------------------------------- #
# refinement 1: risk-averse constraint tie-break (feasibility-margin LCB)
# --------------------------------------------------------------------------- #
def test_feasibility_margin_prefers_safer_and_penalizes_uncertainty():
    c = ConstraintConfig(
        f_r_limit=1.55, cbc_limit=1550.0, f_q_limit=2.41, ao_abs_limit=0.30, risk_z=0.25
    )
    # A sits further below the binding F_r limit than B (both same σ).
    mean = np.array([
        [1.540, 1400.0, 2.00, 640.0, 0.05, np.nan, np.nan],
        [1.545, 1400.0, 2.00, 640.0, 0.05, np.nan, np.nan],
    ])
    std = np.tile([0.01, 10.0, 0.05, 3.0, 0.02, np.nan, np.nan], (2, 1))
    m = acq.feasibility_margin(_pred(mean, std), c)
    assert m[0] > m[1]                              # safer candidate -> larger margin
    # the margin is an LCB: inflating σ on the binding axis shrinks it.
    hi = std.copy(); hi[0, 0] = 0.20
    m_lo = acq.feasibility_margin(_pred(mean[:1], std[:1]), c)[0]
    m_hi = acq.feasibility_margin(_pred(mean[:1], hi[:1]), c)[0]
    assert m_hi < m_lo


def test_rank_tiebreak_breaks_penalty_ties_by_margin():
    """Among exploit look-alikes (identical score within ε) the larger predicted
    feasibility margin wins — the pilot's boundary-basin degeneracy fix."""

    exploit = np.array([-0.46, -0.46, -0.46, -0.46, -0.46])
    margin = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    rank = acq.rank_with_tiebreak(exploit, margin, tie_epsilon=0.1)
    assert int(np.argmax(rank)) == int(np.argmax(margin))       # index 3
    assert list(np.argsort(-rank)) == list(np.argsort(-margin))


def test_rank_tiebreak_keeps_exploit_primary_between_buckets():
    """A real exploit gap (beyond ε) dominates the margin tie-break."""

    exploit = np.array([1.0, 0.0])
    margin = np.array([0.0, 100.0])            # huge margin but worse exploit
    rank = acq.rank_with_tiebreak(exploit, margin, tie_epsilon=0.1)
    assert int(np.argmax(rank)) == 0


def test_rank_tiebreak_disabled_is_pure_exploit():
    exploit = np.array([-0.46, -0.50, -0.30])
    margin = np.array([0.0, 5.0, -1.0])
    assert np.array_equal(acq.rank_with_tiebreak(exploit, margin, 0.0), exploit)


def _flat_pool(cands, exploit, margin):
    """A ScoredPool whose exploit scores are (near) tied so the margin decides."""

    n = len(cands)
    mean = np.tile([1.52, 1500.0, 2.30, 640.0, 0.20, np.nan, np.nan], (n, 1)).astype(float)
    std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (n, 1)).astype(float)
    return acq.ScoredPool(
        candidates=cands, mean=mean, epistemic=std.copy(), calibrated=std.copy(),
        conv=np.ones(n), p_feas=np.full(n, 0.40), acq=np.zeros(n),
        raw_epi=np.zeros(n), in_region=np.ones(n, dtype=bool),
        exploit=np.asarray(exploit, float),
        margin=np.asarray(margin, float),
        rank=acq.rank_with_tiebreak(np.asarray(exploit, float), np.asarray(margin, float), 0.1),
    )


def test_compose_wave_exploit_tiebreak_takes_largest_margin():
    """Exploit slots among score-ties are filled largest-margin first."""

    rng = random.Random(21)
    ctx = _ctx()
    cands, seen = [], set()
    while len(cands) < 8:
        g = random_genome(rng, "K1_K2", 30)
        pat = g.to_pattern()
        if pat.digest in seen:
            continue
        seen.add(pat.digest)
        cands.append(Candidate(pat, g, "local", None, candidate_record_id(pat, ctx), 5.2))
    exploit = np.full(8, -0.46)                # penalty-tied
    margin = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    scored = _flat_pool(cands, exploit, margin)
    slots = acq.compose_wave(
        scored, [], random.Random(1), size=5, n_exploit=5, n_explore=0, n_control=0,
        tau=0.30, hamming_min=1,
    )
    chosen = [s.index for s in slots if s.slot == "exploit"]
    # the five largest-margin candidates are selected (order-independent).
    assert set(chosen) == set(np.argsort(-margin)[:5].tolist())


# --------------------------------------------------------------------------- #
# refinement 3: Hamming-vs-verified hard filter for exploit slots
# --------------------------------------------------------------------------- #
def test_compose_wave_exploit_excludes_near_verified_repeat():
    """A 1-move near-repeat of an already-verified board is barred from the
    exploit slot even when it out-ranks the alternatives (default-on floor)."""

    rng = random.Random(33)
    ctx = _ctx()
    g0 = random_genome(rng, "K1_K2", 30)
    p0 = g0.to_pattern()                        # the verified board
    near = near_g = None
    while near is None:
        child = mutate(g0, rng, 1, feed_move_prob=0.0, batches=ctx.batches)
        cand = child.to_pattern()
        if 0 < cand.hamming(p0) <= 4:
            near, near_g = cand, child
    far = far_g = None
    while far is None:
        g = random_genome(rng, "K1_K2", 30)
        if g.to_pattern().hamming(p0) >= 8:
            far, far_g = g.to_pattern(), g
    thr = near.hamming(p0) + 1
    assert near.hamming(p0) < thr <= far.hamming(p0)
    cands = [
        Candidate(near, near_g, "local", None, candidate_record_id(near, ctx), 5.2),
        Candidate(far, far_g, "local", None, candidate_record_id(far, ctx), 5.2),
    ]
    # near out-ranks far on the exploit score.
    scored = _flat_pool(cands, np.array([1.0, 0.0]), np.array([0.0, 0.0]))

    off = acq.compose_wave(
        scored, [p0], random.Random(1), size=1, n_exploit=1, n_explore=0, n_control=0,
        tau=0.3, hamming_min=1, exploit_verified_hamming=0,
    )
    assert cands[off[0].index].pattern.digest == near.digest      # filter off -> near

    on = acq.compose_wave(
        scored, [p0], random.Random(1), size=1, n_exploit=1, n_explore=0, n_control=0,
        tau=0.3, hamming_min=1, exploit_verified_hamming=thr,
    )
    assert cands[on[0].index].pattern.digest == far.digest         # filter on -> far


# --------------------------------------------------------------------------- #
# user-criteria scoring (plan sec. 12.5 — Phase E; both user addenda)
# --------------------------------------------------------------------------- #
# 7-col surrogate row: 0 F_r, 1 CBC, 2 F_q, 3 cyclen, 4 AO/|ASI|, 5/6 burnups.
def _crow(f_r, cbc, f_q, cyclen, ao, pin=np.nan):
    return [f_r, cbc, f_q, cyclen, ao, np.nan, pin]


def test_criteria_unset_limits_do_not_gate():
    # All gated limits None -> report-only: an absurd F_r/CBC is never penalized.
    spec = acq.CriteriaSpec(
        target_cyclen=625.0, cyclen_tolerance=5.0,
        f_r_limit=None, cbc_limit=None, f_q_limit=None,
        asi_abs_limit=None, pin_bu_limit=None,
    )
    mean = np.array([_crow(9.9, 9999.0, 9.9, 625.0, 0.9)])
    std = np.array([_crow(0.02, 20.0, 0.05, 2.0, 0.02)])
    s = acq.score_user_criteria(_pred(mean, std), spec)
    assert s.gated_axes == ()                       # nothing gates
    assert s.constraint_penalty[0] == 0.0
    assert bool(s.on_target[0]) and bool(s.feasible[0])   # in band, no constraints


def test_criteria_asi_gates_like_ao():
    # asi_abs_limit constrains the ao_abs axis (col 4): |ASI| == |AO| numerically.
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=5.0,
                            f_r_limit=None, cbc_limit=None, f_q_limit=None,
                            asi_abs_limit=0.30, pin_bu_limit=None)
    mean = np.array([_crow(1.5, 1400.0, 2.3, 625.0, 0.20),
                     _crow(1.5, 1400.0, 2.3, 625.0, 0.40)])
    std = np.array([_crow(0.02, 20.0, 0.05, 2.0, 0.01),
                    _crow(0.02, 20.0, 0.05, 2.0, 0.01)])
    s = acq.score_user_criteria(_pred(mean, std), spec)
    assert "asi_abs_limit" in s.gated_axes
    # excess matches the reward ao gate on col 4 (UCB = mu + risk_z*sigma).
    kappa = spec.risk_z
    ex0 = max(0.0, 0.20 + kappa * 0.01 - 0.30) / 0.02
    ex1 = max(0.0, 0.40 + kappa * 0.01 - 0.30) / 0.02
    assert s.constraint_penalty[0] == pytest.approx(ex0 ** 2)     # == 0 (under limit)
    assert s.constraint_penalty[1] == pytest.approx(ex1 ** 2)     # > 0 (over limit)
    assert bool(s.feasible[0]) and not bool(s.feasible[1])
    assert s.total[0] > s.total[1]


def test_criteria_mtc_sdm_never_scored():
    # MTC/SDM are post-verification axes: setting them must not change any score.
    base = dict(target_cyclen=625.0, cyclen_tolerance=5.0, pin_bu_limit=70.0)
    spec_none = acq.CriteriaSpec(**base, mtc_limit=None, sdm_limit=None)
    spec_set = acq.CriteriaSpec(**base, mtc_limit=-5.0, sdm_limit=5000.0)
    mean = np.array([_crow(1.52, 1500.0, 2.35, 625.0, 0.20, pin=69.0)])
    std = np.array([_crow(0.02, 20.0, 0.05, 2.0, 0.01, pin=0.5)])
    a = acq.score_user_criteria(_pred(mean, std), spec_none)
    b = acq.score_user_criteria(_pred(mean, std), spec_set)
    np.testing.assert_allclose(a.total, b.total)
    np.testing.assert_allclose(a.constraint_penalty, b.constraint_penalty)
    assert a.gated_axes == b.gated_axes
    assert "mtc_limit" not in a.gated_axes and "sdm_limit" not in a.gated_axes


def test_criteria_matching_outranks_mismatch():
    # A candidate on both targets outranks one whose cyclen is out of band.
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=2.0,
                            target_discharge_burnup=53.0, discharge_tolerance=0.5,
                            f_r_limit=None, cbc_limit=None, f_q_limit=None,
                            asi_abs_limit=None)
    mean = np.array([_crow(1.50, 1400.0, 2.3, 625.0, 0.2),     # on target
                     _crow(1.45, 1400.0, 2.3, 660.0, 0.2)])    # cyclen far off
    std = np.array([_crow(0.02, 20.0, 0.05, 1.0, 0.01),
                    _crow(0.02, 20.0, 0.05, 1.0, 0.01)])
    s = acq.score_user_criteria(_pred(mean, std), spec,
                                discharge_mean=[53.1, 53.1], discharge_std=[0.1, 0.1])
    assert bool(s.on_target[0]) and not bool(s.on_target[1])
    # matching candidate wins even though the mismatch has the lower F_r.
    assert s.total[0] > s.total[1]


def test_criteria_minimize_fr_within_band():
    # Among two in-band candidates, lower F_r UCB wins regardless of cbc/cyclen.
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=2.0,
                            f_r_limit=None, cbc_limit=None, f_q_limit=None,
                            asi_abs_limit=None)
    mean = np.array([_crow(1.48, 1500.0, 2.3, 626.0, 0.2),     # lower F_r, worse cbc
                     _crow(1.52, 1400.0, 2.3, 624.0, 0.2)])    # higher F_r, better cbc
    std = np.array([_crow(0.02, 20.0, 0.05, 1.0, 0.01),
                    _crow(0.02, 20.0, 0.05, 1.0, 0.01)])
    s = acq.score_user_criteria(_pred(mean, std), spec)
    assert bool(s.on_target[0]) and bool(s.on_target[1])
    assert s.fr_ucb[0] < s.fr_ucb[1]
    assert s.total[0] > s.total[1]        # lower F_r UCB wins, cbc/cyclen ignored


def test_criteria_out_of_band_never_outranks_in_band():
    # An out-of-band candidate with a much lower F_r still loses to an in-band one.
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=2.0,
                            f_r_limit=None, cbc_limit=None, f_q_limit=None,
                            asi_abs_limit=None)
    mean = np.array([_crow(1.70, 1400.0, 2.3, 625.0, 0.2),     # in band, high F_r
                     _crow(1.30, 1400.0, 2.3, 700.0, 0.2)])    # out of band, low F_r
    std = np.array([_crow(0.02, 20.0, 0.05, 1.0, 0.01),
                    _crow(0.02, 20.0, 0.05, 1.0, 0.01)])
    s = acq.score_user_criteria(_pred(mean, std), spec)
    assert bool(s.on_target[0]) and not bool(s.on_target[1])
    assert s.total[0] > s.total[1]


def test_criteria_pin_bu_limit_penalizes_and_fr_gate_dual_role():
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=5.0,
                            f_r_limit=1.55, cbc_limit=None, f_q_limit=None,
                            asi_abs_limit=None, pin_bu_limit=70.0)
    # row0: pin under limit, F_r under limit -> feasible
    # row1: pin over limit                  -> penalized + infeasible
    # row2: F_r over its limit (dual role gate) -> penalized + infeasible
    mean = np.array([_crow(1.50, 1400.0, 2.3, 625.0, 0.2, pin=68.0),
                     _crow(1.50, 1400.0, 2.3, 625.0, 0.2, pin=78.0),
                     _crow(1.62, 1400.0, 2.3, 625.0, 0.2, pin=68.0)])
    std = np.array([_crow(0.02, 20.0, 0.05, 2.0, 0.01, pin=0.5),
                    _crow(0.02, 20.0, 0.05, 2.0, 0.01, pin=0.5),
                    _crow(0.02, 20.0, 0.05, 2.0, 0.01, pin=0.5)])
    s = acq.score_user_criteria(_pred(mean, std), spec)
    assert "pin_bu_limit" in s.gated_axes and "f_r_limit" in s.gated_axes
    assert bool(s.feasible[0])
    assert s.constraint_penalty[1] > 0 and not bool(s.feasible[1])   # pin BU gate
    assert s.constraint_penalty[2] > 0 and not bool(s.feasible[2])   # F_r gate
    assert s.total[0] > s.total[1] and s.total[0] > s.total[2]


def test_criteria_unknown_axis_cannot_certify_feasible():
    # A SET limit on an axis the model can't predict (NaN std, e.g. pin BU on a
    # v2 checkpoint) never certifies feasibility (mirrors reward semantics).
    spec = acq.CriteriaSpec(target_cyclen=625.0, cyclen_tolerance=5.0,
                            f_r_limit=None, cbc_limit=None, f_q_limit=None,
                            asi_abs_limit=None, pin_bu_limit=70.0)
    mean = np.array([_crow(1.5, 1400.0, 2.3, 625.0, 0.2, pin=68.0)])
    std = np.array([_crow(0.02, 20.0, 0.05, 2.0, 0.01, pin=np.nan)])  # pin unknown
    s = acq.score_user_criteria(_pred(mean, std), spec)
    assert s.constraint_penalty[0] == 0.0        # no penalty from an unknown axis
    assert not bool(s.feasible[0])               # but cannot be certified feasible


# --------------------------------------------------------------------------- #
# max_cycle_min_fr objective (user directive 2026-07-21)
# --------------------------------------------------------------------------- #
def _mc_pred(rows):
    """7-col surrogate mean (F_r, CBC, F_q, cyclen, AO, +2 NaN burnups), zero std."""
    m = np.full((len(rows), 7), np.nan)
    m[:, :5] = np.asarray(rows, dtype=float)
    z = np.zeros((len(rows), 7))
    return SurrogatePrediction(m, z.copy(), z.copy())


def test_max_cycle_scalar_is_cyclen_lcb_minus_lambda_fr_ucb():
    # zero std: scalar == cyclen - lam*F_r
    pred = _mc_pred([[1.80, 1500.0, 2.30, 700.0, 0.20]])
    spec = acq.MaxCycleSpec(lam=100.0, risk_z=0.25)
    mc = acq.score_max_cycle_min_fr(pred, spec)
    assert mc.scalar[0] == pytest.approx(700.0 - 100.0 * 1.80)
    assert mc.cyclen_lcb[0] == pytest.approx(700.0)
    assert mc.fr_ucb[0] == pytest.approx(1.80)
    assert bool(mc.constraint_ok[0])
    # with std, cyclen enters at LCB and F_r at UCB (conservative both ways).
    m = np.full((1, 7), np.nan); m[:, :5] = [1.80, 1500.0, 2.30, 700.0, 0.20]
    s = np.zeros((1, 7)); s[:, 0] = 0.1; s[:, 3] = 4.0
    mc2 = acq.score_max_cycle_min_fr(SurrogatePrediction(m, s.copy(), s.copy()), spec)
    assert mc2.cyclen_lcb[0] == pytest.approx(700.0 - 0.25 * 4.0)
    assert mc2.fr_ucb[0] == pytest.approx(1.80 + 0.25 * 0.1)
    assert mc2.scalar[0] < mc.scalar[0]


def test_max_cycle_constraint_violation_dominates():
    # A: feasible; B: CBC over the limit -> must sink below A despite higher cyclen.
    pred = _mc_pred([
        [1.87, 1532.0, 2.30, 732.8, 0.20],   # feasible
        [1.80, 1600.0, 2.30, 745.0, 0.20],   # CBC 1600 > 1550 -> infeasible
    ])
    spec = acq.MaxCycleSpec(lam=100.0, risk_z=0.25)
    mc = acq.score_max_cycle_min_fr(pred, spec)
    assert bool(mc.constraint_ok[0]) and not bool(mc.constraint_ok[1])
    assert mc.total[0] > mc.total[1]                 # feasible beats infeasible
    assert mc.scalar[1] > mc.scalar[0]               # ...even though its raw scalar is higher


def test_max_cycle_lambda_calibration_10efpd_vs_0p1_fr():
    # λ=100: a 10 EFPD cyclen gain trades against a 0.1 F_r reduction (equal scalar).
    spec = acq.MaxCycleSpec(lam=100.0, risk_z=0.0)
    pred = _mc_pred([
        [1.80, 1500.0, 2.30, 720.0, 0.20],   # higher F_r, +10 EFPD
        [1.70, 1500.0, 2.30, 710.0, 0.20],   # lower F_r by 0.1, -10 EFPD
    ])
    mc = acq.score_max_cycle_min_fr(pred, spec)
    assert mc.scalar[0] == pytest.approx(mc.scalar[1])   # exactly balanced at λ=100
    # a slightly bigger F_r drop wins; a slightly bigger cyclen gain wins.
    assert acq.score_max_cycle_min_fr(
        _mc_pred([[1.69, 1500.0, 2.30, 710.0, 0.20]]), spec
    ).scalar[0] > mc.scalar[0]


def test_max_cycle_fr_is_ungated():
    # F_r far above the target-mode 1.55 gate is STILL constraint-feasible here.
    c = acq.make_maxcycle_constraints(acq.MaxCycleSpec())
    assert c.f_r_limit >= 1.0e12
    pred = _mc_pred([[2.50, 1500.0, 2.30, 700.0, 0.20]])   # F_r 2.5 (huge)
    pf = acq.p_feasible(pred, c)
    assert pf[0] == pytest.approx(1.0)                     # F_r contributes no gate
    assert bool(acq.score_max_cycle_min_fr(pred, acq.MaxCycleSpec()).constraint_ok[0])


def test_score_pool_max_cycle_gates_region_and_shapes():
    ctx = CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)
    rng = random.Random(7)
    cands = []
    for _ in range(6):
        g = random_genome(rng, ctx.pair, ctx.n_fresh)
        pat = g.to_pattern()
        cands.append(Candidate(pat, g, "random", None, candidate_record_id(pat, ctx), 5.2))

    class _M:
        def predict(self, patterns, case, cell=0.0):
            n = len(list(patterns))
            m = np.full((n, 7), np.nan)
            m[:, :5] = np.tile([1.8, 1500.0, 2.3, 700.0, 0.2], (n, 1))
            z = np.zeros((n, 7))
            return SurrogatePrediction(m, z.copy(), z.copy())

    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    scored = acq.score_pool_max_cycle(_M(), ctx, cands, acq.MaxCycleSpec(), tr, tie_epsilon=0.1)
    assert len(scored) == 6
    assert scored.exploit.shape == (6,) and scored.rank.shape == (6,)
    assert np.all(scored.in_region)                       # campaign bin always in-region
    assert np.all(np.isfinite(scored.exploit))
    # out-of-region candidate is hard-gated to -inf.
    tr2 = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=999, campaign_e_core=9.9)
    scored2 = acq.score_pool_max_cycle(_M(), ctx, cands, acq.MaxCycleSpec(), tr2)
    assert np.all(~np.isfinite(scored2.exploit))          # feed 121 not supported -> -inf


# --------------------------------------------------------------------------- #
# min_fr_max_cycle objective (user directive 2026-07-22 — revised hierarchy)
# --------------------------------------------------------------------------- #
def test_min_fr_scalar_and_fr_strict_dominance():
    spec = acq.MinFrSpec(lam_fr=1000.0, risk_z=0.25)
    pred = _mc_pred([[1.60, 1500.0, 2.0, 700.0, 0.05]])
    mf = acq.score_min_fr_max_cycle(pred, spec)
    assert mf.scalar[0] == pytest.approx(700.0 - 1000.0 * 1.60)
    # λ_Fr=1000: a 0.01 F_r reduction is worth exactly 10 EFPD in the scalar.
    a = acq.score_min_fr_max_cycle(_mc_pred([[1.60, 1500.0, 2.0, 700.0, 0.05]]), spec).scalar[0]
    b = acq.score_min_fr_max_cycle(_mc_pred([[1.59, 1500.0, 2.0, 690.0, 0.05]]), spec).scalar[0]
    assert a == pytest.approx(b)                          # 0.01 F_r == 10 EFPD
    # F_r STRICTLY dominates the cell's ~30 EFPD cyclen spread: 0.02 lower F_r beats
    # even a +30 EFPD candidate.
    lo_fr = acq.score_min_fr_max_cycle(_mc_pred([[1.60, 1500.0, 2.0, 700.0, 0.05]]), spec)
    hi_cy = acq.score_min_fr_max_cycle(_mc_pred([[1.62, 1500.0, 2.0, 730.0, 0.05]]), spec)
    assert lo_fr.total[0] > hi_cy.total[0]


def _mc_pred_pin(rows, pin):
    """:func:`_mc_pred` with the col-6 pin-BU mean filled in (the gated axis)."""
    pred = _mc_pred(rows)
    pred.mean[:, 6] = float(pin)
    return pred


def test_min_fr_gates_all_four_including_fr():
    spec = acq.MinFrSpec()
    # F_r 1.60 > 1.55 with F_q/CBC/AO fine -> INFEASIBLE (F_r rejoins the gate).
    over_fr = _mc_pred_pin([[1.60, 1500.0, 2.30, 700.0, 0.20]], 70.0)
    assert not bool(acq.score_min_fr_max_cycle(over_fr, spec).constraint_ok[0])
    # all four within limits (and pin BU under the gate) -> feasible.
    ok = _mc_pred_pin([[1.54, 1500.0, 2.30, 690.0, 0.20]], 70.0)
    assert bool(acq.score_min_fr_max_cycle(ok, spec).constraint_ok[0])
    # make_minfr_constraints gates F_r at 1.55 (p_feasible reflects it).
    c = acq.make_minfr_constraints(spec)
    assert c.f_r_limit == pytest.approx(1.55)
    assert acq.p_feasible(over_fr, c)[0] < 0.5            # F_r factor pulls it down


def test_min_fr_gates_predicted_pin_burnup():
    """min_fr_max_cycle screens the PREDICTED pin BU (col 6) — the axis that made
    both closed min_fr campaigns report over-limit cores as feasible
    (data/reports/fpcamp_E1E2_f109_results_20260817.md §7)."""
    spec = acq.MinFrSpec(risk_z=0.0)
    assert spec.pin_bu_limit == pytest.approx(78.0)       # model-margin default

    row = [1.50, 1500.0, 2.30, 690.0, 0.20]               # clean on all four axes
    under = acq.score_min_fr_max_cycle(_mc_pred_pin([row], 77.0), spec)
    over = acq.score_min_fr_max_cycle(_mc_pred_pin([row], 83.2), spec)  # f109 winner

    assert bool(under.constraint_ok[0])                   # 77.0 <= 78.0 -> feasible
    assert not bool(over.constraint_ok[0])                # 83.2 > 78.0 -> REJECTED
    assert over.constraint_penalty[0] == pytest.approx((83.2 - 78.0) ** 2)
    assert under.constraint_penalty[0] == pytest.approx(0.0)
    # the gate has teeth: the violation is TIER-dominant, so an over-limit core can
    # never outrank a feasible one no matter how much better its F_r/cyclen scalar.
    better_fr = acq.score_min_fr_max_cycle(_mc_pred_pin([[1.40, 1500.0, 2.30, 760.0, 0.20]],
                                                        83.2), spec)
    assert better_fr.scalar[0] > under.scalar[0]
    assert better_fr.total[0] < under.total[0]
    # the knob is configurable: at the licensing number 83.2 is still out, 79.0 in.
    at80 = acq.MinFrSpec(risk_z=0.0, pin_bu_limit=80.0)
    assert bool(acq.score_min_fr_max_cycle(_mc_pred_pin([row], 79.0), at80).constraint_ok[0])
    assert not bool(acq.score_min_fr_max_cycle(_mc_pred_pin([row], 83.2), at80).constraint_ok[0])
    # an UNPREDICTED pin BU (NaN col 6) is never silently called feasible.
    assert not bool(acq.score_min_fr_max_cycle(_mc_pred([row]), spec).constraint_ok[0])
    # …but it does not distort the ranking (penalty contribution stays 0).
    assert acq.score_min_fr_max_cycle(_mc_pred([row]), spec).constraint_penalty[0] == \
        pytest.approx(0.0)


def test_min_fr_feasible_outranks_infeasible_and_cyclen_tiebreak():
    spec = acq.MinFrSpec(lam_fr=1000.0, risk_z=0.0)
    pred = _mc_pred([
        [1.62, 1500.0, 2.0, 745.0, 0.05],   # A infeasible
        [1.62, 1500.0, 2.0, 760.0, 0.05],   # B same F_r, +15 EFPD
        [1.54, 1490.0, 1.95, 690.0, 0.05],  # C feasible (F_r<=1.55)
    ])
    mf = acq.score_min_fr_max_cycle(pred, spec)
    assert int(np.argmax(mf.total)) == 2                 # feasible C wins outright
    assert mf.total[1] > mf.total[0]                     # equal F_r -> higher cyclen tie-break


def test_score_pool_min_fr_shapes_and_region():
    ctx = CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)
    rng = random.Random(11)
    cands = []
    for _ in range(5):
        g = random_genome(rng, ctx.pair, ctx.n_fresh)
        pat = g.to_pattern()
        cands.append(Candidate(pat, g, "random", None, candidate_record_id(pat, ctx), 5.2))

    class _M:
        def predict(self, patterns, case, cell=0.0):
            n = len(list(patterns))
            m = np.full((n, 7), np.nan)
            m[:, :5] = np.tile([1.62, 1500.0, 2.0, 730.0, 0.05], (n, 1))
            z = np.zeros((n, 7))
            return SurrogatePrediction(m, z.copy(), z.copy())

    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    scored = acq.score_pool_min_fr(_M(), ctx, cands, acq.MinFrSpec(), tr, tie_epsilon=0.1)
    assert len(scored) == 5 and scored.exploit.shape == (5,)
    assert np.all(scored.in_region)
    # F_r 1.62 > 1.55 -> all infeasible -> exploit heavily negative (penalty-dominated).
    assert np.all(scored.exploit < 0)


# --------------------------------------------------------------------------- #
# min_fuel_cost objective (user directive 2026-07-21 — minimize fresh fuel cost)
# --------------------------------------------------------------------------- #
def _fc_pred(rows, stds=None):
    """7-col surrogate mean (F_r,CBC,F_q,cyclen,AO,asm_bu,pin_bu); default zero std."""
    m = np.asarray(rows, dtype=float)
    assert m.shape[1] == 7
    z = np.zeros_like(m) if stds is None else np.asarray(stds, dtype=float)
    return SurrogatePrediction(m, z.copy(), z.copy())


def _fcrow(fr, cbc, fq, cy, ao, pin):
    return [fr, cbc, fq, cy, ao, np.nan, pin]


def test_fuel_cost_scalar_is_neg_fe_minus_lambda_fr_ucb():
    spec = acq.MinFuelCostSpec(lam_fr=20.0, risk_z=0.25)
    # zero std: scalar == -FE - lam*F_r
    pred = _fc_pred([_fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 68.0)])
    s = acq.score_min_fuel_cost(pred, spec, np.array([700.0]))
    assert s.scalar[0] == pytest.approx(-700.0 - 20.0 * 1.50)
    assert bool(s.constraint_ok[0]) and s.constraint_penalty[0] == pytest.approx(0.0)
    # with std, F_r enters at its UCB (conservative).
    std = np.array([_fcrow(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)])
    s2 = acq.score_min_fuel_cost(_fc_pred(
        [_fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 68.0)], std), spec, np.array([700.0]))
    assert s2.fr_ucb[0] == pytest.approx(1.50 + 0.25 * 0.1)


def test_fuel_cost_lower_fe_wins_and_fr_tiebreak_within_equal_fe():
    spec = acq.MinFuelCostSpec(lam_fr=20.0, risk_z=0.0)
    pred = _fc_pred([_fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 68.0)] * 2)
    # different FE, same F_r -> lower FE wins (FE primary).
    s = acq.score_min_fuel_cost(pred, spec, np.array([680.0, 720.0]))
    assert s.total[0] > s.total[1]
    # equal FE, different F_r -> lower F_r wins (subordinate tie-break).
    p2 = _fc_pred([_fcrow(1.48, 1400.0, 2.30, 625.0, 0.20, 68.0),
                   _fcrow(1.52, 1400.0, 2.30, 625.0, 0.20, 68.0)])
    s2 = acq.score_min_fuel_cost(p2, spec, np.array([700.0, 700.0]))
    assert s2.total[0] > s2.total[1]


def test_fuel_cost_cyclen_band_is_two_sided_and_dominates():
    spec = acq.MinFuelCostSpec(lam_fr=20.0, risk_z=0.0)
    # in-band (625) feasible; below 615 and above 635 both infeasible + penalized.
    pred = _fc_pred([_fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 68.0),   # feasible
                     _fcrow(1.50, 1400.0, 2.30, 600.0, 0.20, 68.0),   # cyclen < 615
                     _fcrow(1.50, 1400.0, 2.30, 650.0, 0.20, 68.0)])  # cyclen > 635
    s = acq.score_min_fuel_cost(pred, spec, np.array([700.0, 650.0, 650.0]))
    assert bool(s.constraint_ok[0])
    assert not bool(s.constraint_ok[1]) and not bool(s.constraint_ok[2])
    # feasible in-band beats both out-of-band even though they have LOWER FE.
    assert s.total[0] > s.total[1] and s.total[0] > s.total[2]


def test_fuel_cost_pin_bu_point_estimate_gates_without_sigma():
    spec = acq.MinFuelCostSpec(risk_z=0.25)
    # pin BU is a physics POINT estimate: NaN std must still gate (>80 infeasible),
    # and a finite point estimate <=80 certifies feasibility (no sigma required).
    pred = _fc_pred([_fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 78.0),
                     _fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 85.0)],
                    stds=[_fcrow(0.02, 20.0, 0.05, 2.0, 0.01, np.nan)] * 2)
    s = acq.score_min_fuel_cost(pred, spec, np.array([700.0, 700.0]))
    assert bool(s.constraint_ok[0])                       # 78 <= 80, NaN std OK
    assert not bool(s.constraint_ok[1])                   # 85 > 80 -> gated
    assert s.constraint_penalty[1] == pytest.approx(25.0)  # (85-80)^2, no kappa shift


def test_fuel_cost_pin_bu_trained_head_uses_ucb_shift():
    spec = acq.MinFuelCostSpec(risk_z=0.25)
    # trained pin head: finite σ -> gate on mean + κσ (conservative).
    # pin 78 σ 8 -> ucb 80.0 (borderline feasible); pin 79 σ 8 -> ucb 81 > 80 gated.
    pred = _fc_pred([_fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 78.0),
                     _fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 79.0)],
                    stds=[_fcrow(0.02, 20.0, 0.05, 2.0, 0.01, 8.0)] * 2)
    s = acq.score_min_fuel_cost(pred, spec, np.array([700.0, 700.0]))
    assert bool(s.constraint_ok[0])                       # ucb 80.0 <= 80
    assert not bool(s.constraint_ok[1])                   # ucb 81.0 > 80 -> gated
    assert s.constraint_penalty[1] == pytest.approx(1.0)  # (81-80)^2, κσ shift applied


def test_fuel_cost_unresolvable_fe_sorts_last():
    spec = acq.MinFuelCostSpec()
    pred = _fc_pred([_fcrow(1.50, 1400.0, 2.30, 625.0, 0.20, 68.0)] * 2)
    s = acq.score_min_fuel_cost(pred, spec, np.array([700.0, np.nan]))
    assert np.isfinite(s.total[0]) and not np.isfinite(s.total[1])
    assert s.scalar[1] == -np.inf


def test_fresh_fuel_charge_and_array_ga80_count_weighted():
    import os
    from lpopt.data.fuel_types import FuelLibrary, fresh_fuel_charge
    fp = "data/store/fuel_types.parquet"
    if not os.path.exists(fp):
        pytest.skip("fuel_types.parquet not present")
    fuel = FuelLibrary.from_parquet(fp)
    ctx = CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)
    rng = random.Random(3)
    cands = []
    for _ in range(4):
        g = random_genome(rng, ctx.pair, ctx.n_fresh)
        pat = g.to_pattern()
        cands.append(Candidate(pat, g, "random", None, candidate_record_id(pat, ctx), 5.2))
    charge, mw = fresh_fuel_charge(fuel, "ga80", cands[0].pattern.batch_feed())
    assert charge is not None and charge > 0.0
    assert mw is False                                    # ga80 has no u_mass_g
    fe, run_mw = acq.fuel_charge_array(fuel, "ga80", cands)
    assert fe.shape == (4,) and np.all(np.isfinite(fe)) and run_mw is False
    # None fuel -> all-NaN FE (degenerate, never raises).
    fe0, _ = acq.fuel_charge_array(None, "ga80", cands)
    assert np.all(np.isnan(fe0))


def test_score_pool_min_fuel_cost_shapes_and_region():
    import os
    from lpopt.data.fuel_types import FuelLibrary
    fp = "data/store/fuel_types.parquet"
    fuel = FuelLibrary.from_parquet(fp) if os.path.exists(fp) else None
    ctx = CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)
    rng = random.Random(5)
    cands = []
    for _ in range(6):
        g = random_genome(rng, ctx.pair, ctx.n_fresh)
        pat = g.to_pattern()
        cands.append(Candidate(pat, g, "random", None, candidate_record_id(pat, ctx), 5.2))

    class _M:
        def predict(self, patterns, case, cell=0.0):
            n = len(list(patterns))
            m = np.full((n, 7), np.nan)
            m[:, :5] = np.tile([1.50, 1400.0, 2.30, 625.0, 0.20], (n, 1))
            m[:, 6] = 68.0
            z = np.zeros((n, 7)) + 0.5
            return SurrogatePrediction(m, z.copy(), z.copy())

    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    scored = acq.score_pool_min_fuel_cost(
        _M(), ctx, cands, acq.MinFuelCostSpec(), tr, tie_epsilon=0.1,
        fuel=fuel, library_id="ga80")
    assert len(scored) == 6 and scored.exploit.shape == (6,)
    assert np.all(scored.in_region)
    if fuel is not None:
        assert np.all(np.isfinite(scored.exploit))        # feasible + resolvable FE
    # out-of-region candidate is hard-gated to -inf.
    tr2 = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=999, campaign_e_core=9.9)
    scored2 = acq.score_pool_min_fuel_cost(
        _M(), ctx, cands, acq.MinFuelCostSpec(), tr2, fuel=fuel, library_id="ga80")
    assert np.all(~np.isfinite(scored2.exploit))


# --------------------------------------------------------------------------- #
# flat_power objective — FLATNESS-NATIVE (program 20260725 §1.2 / §2.1)
# --------------------------------------------------------------------------- #
def test_flat_power_scalar_is_weighted_peak_plus_cov_ucb():
    spec = acq.FlatPowerSpec(risk_z=0.25, w_cov=0.5,
                             peak_scale=0.4, cov_scale=0.08)
    pred = _fc_pred([_fcrow(1.5, 1400.0, 2.30, 600.0, 0.2, 68.0),
                     _fcrow(1.5, 1400.0, 2.30, 999.0, 0.2, 68.0)])
    fp = acq.score_flat_power(pred, np.array([1.4, 1.4]), np.array([0.1, 0.1]),
                              spec, np.array([0.30, 0.30]), np.array([0.02, 0.02]))
    # cyclen is record-only: identical flatness -> identical score.
    assert fp.total[0] == pytest.approx(fp.total[1])
    # the scalar is EXACTLY -(z_peak + w_cov*z_cov) on the UCBs.
    peak_ucb = 1.4 + 0.25 * 0.1
    cov_ucb = 0.30 + 0.25 * 0.02
    assert fp.peak_ucb[0] == pytest.approx(peak_ucb)
    assert fp.cov_ucb[0] == pytest.approx(cov_ucb)
    assert fp.scalar[0] == pytest.approx(-(peak_ucb / 0.4 + 0.5 * cov_ucb / 0.08))


def test_flat_power_node_peak_is_primary_and_cov_secondary():
    """The C1 weight inversion: peak dominates cov at the DECLARED 1 : 0.5."""
    spec = acq.FlatPowerSpec(risk_z=0.0, w_cov=0.5, peak_scale=0.4, cov_scale=0.08)
    pred = _fc_pred([_fcrow(1.5, 1400.0, 2.30, 625.0, 0.2, 68.0)] * 2)
    # candidate 0: one within-cell SD flatter on PEAK, one SD worse on COV.
    fp = acq.score_flat_power(pred, np.array([1.4 - 0.4, 1.4]), np.zeros(2), spec,
                              np.array([0.30 + 0.08, 0.30]), np.zeros(2))
    # 1.0 SD of peak beats 1.0 SD of cov, because cov is weighted 0.5.
    assert fp.total[0] > fp.total[1]
    # and the realized trade is exactly the declared ratio in SD units.
    assert (fp.z_peak[1] - fp.z_peak[0]) == pytest.approx(1.0)
    assert 0.5 * (fp.z_cov[0] - fp.z_cov[1]) == pytest.approx(0.5)


def test_flat_power_missing_cov_keeps_the_primary_term():
    spec = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08)
    pred = _fc_pred([_fcrow(1.5, 1400.0, 2.30, 625.0, 0.2, 68.0)] * 2)
    fp = acq.score_flat_power(pred, np.array([1.2, 1.6]), np.zeros(2), spec)
    assert np.all(np.isnan(fp.z_cov))
    assert fp.scalar[0] == pytest.approx(-1.2 / 0.4)     # cov term dropped, not 0/0
    assert fp.total[0] > fp.total[1]                     # still ranks on the peak
    # a missing PEAK is fatal (no ranking on cov alone).
    fp2 = acq.score_flat_power(pred, np.array([np.nan, 1.6]), np.zeros(2), spec,
                               np.array([0.1, 0.5]), np.zeros(2))
    assert fp2.total[0] == -np.inf


def test_flat_power_fr_is_a_binary_safety_gate_not_a_graded_penalty():
    """F_r may VETO a candidate; it may never ORDER two candidates (§2.1)."""
    spec = acq.FlatPowerSpec(risk_z=0.0, fr_limit=1.70, peak_scale=0.4, cov_scale=0.08)
    # F_r is NOT in the graded tier list at all.
    assert all(col != 0 for col, _a, _w in acq._FLATPOWER_MODEL_AXES)
    pred = _fc_pred([_fcrow(1.65, 1400.0, 2.30, 625.0, 0.2, 68.0),   # passes
                     _fcrow(1.72, 1400.0, 2.30, 625.0, 0.2, 68.0),   # 0.02 over
                     _fcrow(2.40, 1400.0, 2.30, 625.0, 0.2, 68.0)])  # far over
    fp = acq.score_flat_power(pred, np.full(3, 1.4), np.zeros(3), spec,
                              np.full(3, 0.30), np.zeros(3))
    assert not fp.fr_gate_violated[0] and bool(fp.constraint_ok[0])
    assert fp.fr_gate_violated[1] and fp.fr_gate_violated[2]
    assert not bool(fp.constraint_ok[1])
    # the veto dominates the objective...
    assert fp.total[0] > fp.total[1]
    # ...but it does NOT grade: 1.72 and 2.40 are penalized IDENTICALLY.  Under
    # the old width-0.01 tier a 0.02 overshoot alone cost 4e4.
    assert fp.total[1] == pytest.approx(fp.total[2])
    assert (fp.total[0] - fp.total[1]) == pytest.approx(acq._FLATPOWER_CONSTRAINT_TIER)


def test_flat_power_fr_gate_holds_at_1p70_without_a_bias_correction():
    """Decision D1: bias-corrected form when available, else HOLD at 1.70."""
    assert acq.FlatPowerSpec().fr_gate == pytest.approx(1.70)
    assert acq.FlatPowerSpec(fr_bias=None, fr_sigma=0.3).fr_gate == pytest.approx(1.70)
    # with a correction: 1.70 - bias - 0.5*sigma
    s = acq.FlatPowerSpec(fr_bias=0.04, fr_sigma=0.02)
    assert s.fr_gate == pytest.approx(1.70 - 0.04 - 0.5 * 0.02)
    assert acq.flatpower_fr_gate(s) == pytest.approx(s.fr_gate)
    # an OPTIMISTIC map head (negative bias would loosen) still tightens correctly
    # by sign convention: bias is subtracted as given.
    assert acq.FlatPowerSpec(fr_bias=0.10).fr_gate == pytest.approx(1.60)


def test_flat_power_other_hard_axes_still_grade():
    spec = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08)
    pred = _fc_pred([_fcrow(1.5, 1400.0, 2.30, 625.0, 0.2, 68.0),
                     _fcrow(1.5, 1400.0, 2.30, 625.0, 0.2, 85.0),   # pin BU > 80
                     _fcrow(1.5, 1400.0, 2.60, 625.0, 0.2, 68.0)])  # F_q > 2.41
    fp = acq.score_flat_power(pred, np.full(3, 1.4), np.zeros(3), spec,
                              np.full(3, 0.30), np.zeros(3))
    assert bool(fp.constraint_ok[0])
    assert not bool(fp.constraint_ok[1]) and not bool(fp.constraint_ok[2])
    assert fp.total[0] > fp.total[1] and fp.total[0] > fp.total[2]


def test_flat_power_constraints_retire_fr_from_pfeas_and_margin():
    """F_r must not enter p_feasible OR the margin tie-break in this mode (§2.1)."""
    spec = acq.FlatPowerSpec()
    c = acq.make_flatpower_constraints(spec)
    assert c.f_r_limit == 1.0e12                     # the retirement sentinel
    # a candidate wildly over the F_r limit is NOT penalized by p_feas / margin;
    # both see only CBC / F_q / AO.
    mean = np.array([[3.00, 1400.0, 2.30, 625.0, 0.20, np.nan, 68.0],
                     [1.40, 1400.0, 2.30, 625.0, 0.20, np.nan, 68.0]])
    std = np.array([[0.02, 20.0, 0.05, 3.0, 0.02, np.nan, 1.0]] * 2)
    pf = acq.p_feasible(_pred(mean, std), c)
    assert pf[0] == pytest.approx(pf[1], rel=1e-9)
    mg = acq.feasibility_margin(_pred(mean, std), c)
    assert mg[0] == pytest.approx(mg[1], rel=1e-9)


def _flat_model(peaks, covs=None, *, flatness_api=True):
    class _M:
        def predict(self, patterns, case, cell=0.0):
            n = len(list(patterns))
            m = np.full((n, 7), np.nan)
            m[:, :5] = np.tile([1.60, 1400.0, 2.30, 625.0, 0.20], (n, 1))
            m[:, 6] = 68.0
            z = np.zeros((n, 7)) + 0.01
            return SurrogatePrediction(m, z.copy(), z.copy())

    if flatness_api:
        def predict_map_flatness(self, patterns, case, cell=0.0):
            n = len(list(patterns))
            return (np.asarray(peaks, float)[:n], np.full(n, 0.01),
                    np.asarray(covs, float)[:n], np.full(n, 0.001))
        _M.predict_map_flatness = predict_map_flatness
    else:
        def predict_map_peak(self, patterns, case, cell=0.0):
            n = len(list(patterns))
            return np.asarray(peaks, float)[:n], np.full(n, 0.01)
        _M.predict_map_peak = predict_map_peak
    return _M()


def _flat_cands(n=5):
    ctx = CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)
    rng = random.Random(9)
    cands = []
    for _ in range(n):
        g = random_genome(rng, ctx.pair, ctx.n_fresh)
        pat = g.to_pattern()
        cands.append(Candidate(pat, g, "random", None, candidate_record_id(pat, ctx), 5.2))
    return ctx, cands


def test_score_pool_flat_power_uses_the_flatness_api_and_region():
    ctx, cands = _flat_cands(5)
    model = _flat_model(np.linspace(1.30, 1.70, 5), np.linspace(0.30, 0.40, 5))
    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    scored = acq.score_pool_flat_power(model, ctx, cands, acq.FlatPowerSpec(), tr,
                                       tie_epsilon=0.1)
    assert len(scored) == 5 and np.all(scored.in_region)
    assert np.all(np.isfinite(scored.exploit))
    assert np.argmax(scored.exploit) == 0             # flattest peak+cov wins
    # out-of-region is still a hard -inf.
    tr2 = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=999, campaign_e_core=9.9)
    s2 = acq.score_pool_flat_power(model, ctx, cands, acq.FlatPowerSpec(), tr2)
    assert np.all(~np.isfinite(s2.exploit))


def test_score_pool_flat_power_falls_back_to_peak_only_backend():
    ctx, cands = _flat_cands(4)
    model = _flat_model(np.linspace(1.30, 1.60, 4), flatness_api=False)
    tr = acq.TrustRegion(TrustRegionConfig(), set(), campaign_feed=121, campaign_e_core=5.2)
    scored = acq.score_pool_flat_power(model, ctx, cands, acq.FlatPowerSpec(), tr)
    assert np.all(np.isfinite(scored.exploit))
    assert np.argmax(scored.exploit) == 0


def test_predict_flatness_without_any_map_head_is_all_nan():
    ctx, cands = _flat_cands(3)

    class _Bare:
        def predict(self, patterns, case, cell=0.0):
            n = len(list(patterns))
            return SurrogatePrediction(np.zeros((n, 7)), np.zeros((n, 7)), np.zeros((n, 7)))

    pk_m, pk_s, cv_m, cv_s = acq.predict_flatness(
        _Bare(), [c.pattern for c in cands], ctx)
    for arr in (pk_m, pk_s, cv_m, cv_s):
        assert np.all(np.isnan(arr))
