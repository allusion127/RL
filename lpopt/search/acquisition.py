"""Acquisition, trust region, local search, wave composition (plan sec. 4.6 / 6.2).

The scoring stack is deliberately the vendor one: :func:`build_reward_model`
constructs a :class:`~lpopt.vendor.masterrl.reward.RewardModel` in ``target_cycle``
mode (625 EFPD, ``risk_z=0.25``) whose ``score`` / ``acquisition`` consume the
backend's :class:`SurrogatePrediction` unchanged, so the risk-adjusted objective
is never reimplemented here.

:func:`p_feasible` is the explicit ``Π Φ((limit−μ)/σ_total)`` gate over
F_r / CBC / F_q / AO (× convergence probability), computed with
:func:`scipy.stats.norm.cdf` (its unit test asserts equality against a direct
``norm.cdf`` reference).

:class:`TrustRegion` gates candidates to the store-supported ``(feed, e_core)``
bins — a **hard zero** outside, growth only from measured labels — and inflates
σ on frontier bins.  In the v1 fixed feed-121 campaign the region is trivially
satisfied, but it is always active and unit-tested with off-grid candidates.

:func:`compose_wave` builds the 5 exploit / 2 explore / 1 control wave with a
pairwise-Hamming ≥ 4 diversity filter (also against the verified set) and the
τ schedule (0.3 → quantile-based once a feasible label exists).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from ..data import flat_scale as _FS
from ..data import map_calibration as _MC
from ..vendor.masterrl.domain import CaseKey, Pattern, PatternRecord
from ..vendor.masterrl.reward import ConstraintConfig, RewardModel
from ..vendor.masterrl.surrogate import SurrogatePrediction
from .construct import Candidate, CaseContext, screen_e_core_band
from .genome import GenomeError, mutate
from .rule_metrics import rule_penalty as _rule_penalty

# surrogate columns: 0 F_r, 1 CBC_max, 2 F_q, 3 cyclen, 4 AO_abs, 5/6 burnups.
_PRED_AXES = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# constraints + p_feasible
# --------------------------------------------------------------------------- #
def make_constraints(acq: Any) -> ConstraintConfig:
    """Build the vendor :class:`ConstraintConfig` from ``[acquisition]`` (target mode)."""

    return ConstraintConfig(
        f_r_limit=float(acq.f_r_limit),
        cbc_limit=float(acq.cbc_limit),
        f_q_limit=float(acq.f_q_limit),
        ao_abs_limit=float(acq.ao_abs_limit),
        risk_z=float(acq.risk_z),
        objective_mode="target_cycle",
        cycle_target_efpd=float(acq.cycle_target_efpd),
        cycle_tolerance_efpd=float(acq.cycle_tolerance_efpd),
    )


def p_feasible(
    prediction: SurrogatePrediction,
    constraints: ConstraintConfig,
    convergence: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """``Π Φ((limit−μ)/σ_total)`` over F_r / CBC / F_q / AO (× convergence prob).

    ``σ_total`` is the calibrated std.  An axis with a non-finite std (no model,
    plan sec. 4.5) contributes ``unknown_axis_probability``; a (near-)zero std
    collapses to the hard indicator ``μ ≤ limit`` — matching the vendor
    ``feasible_probability`` semantics, which uses ``erf`` == ``norm.cdf``.
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    columns = [
        (0, constraints.f_r_limit),
        (1, constraints.cbc_limit),
        (2, constraints.f_q_limit),
        (4, constraints.ao_abs_limit),
    ]
    prob = np.ones(n, dtype=float)
    for column, limit in columns:
        mu = mean[:, column]
        sd = std[:, column]
        col = np.empty(n, dtype=float)
        known = np.isfinite(sd)
        col[~known] = constraints.unknown_axis_probability
        positive = known & (sd > 1.0e-12)
        col[positive] = norm.cdf((limit - mu[positive]) / sd[positive])
        degenerate = known & (sd <= 1.0e-12)
        col[degenerate] = (mu[degenerate] <= limit).astype(float)
        prob *= col
    if convergence is not None:
        prob = prob * np.asarray(convergence, dtype=float)
    return prob


def raw_epistemic(prediction: SurrogatePrediction) -> np.ndarray:
    """Per-candidate raw epistemic σ summary (max over the predicted axes)."""

    epi = np.asarray(prediction.epistemic_std, dtype=float)[:, list(_PRED_AXES)]
    return np.nanmax(np.where(np.isfinite(epi), epi, 0.0), axis=1)


# constraint columns + their reward-model penalty widths (the natural per-axis
# scale for the feasibility margin — identical to the reward penalty scaling).
_MARGIN_AXES = (
    (0, "f_r_limit", "f_r_width"),
    (1, "cbc_limit", "cbc_width"),
    (2, "f_q_limit", "f_q_width"),
    (4, "ao_abs_limit", "ao_width"),
)


def feasibility_margin(
    prediction: SurrogatePrediction, constraints: ConstraintConfig
) -> np.ndarray:
    """Risk-adjusted feasibility MARGIN LCB (plan sec. 4.6 tie-break key).

    For each safety axis the risk-adjusted upper bound is ``μ_c + κ σ_c`` with
    ``κ = risk_z``; its normalized excess over the limit is
    ``(μ_c + κ σ_c − limit_c) / width_c``.  The candidate margin is the negated
    **worst** (max) axis excess: **higher is safer** (further below every limit
    with uncertainty accounted).  An axis the surrogate cannot predict (NaN σ)
    contributes its mean only (no shift); a fully-unknown axis set yields ``-inf``.
    This is the secondary ranking key that separates constraint-look-alikes whose
    exploit scores tie — exactly the boundary-basin degeneracy the model cannot
    resolve on the primary score alone.
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    kappa = float(constraints.risk_z)
    worst = np.full(n, -np.inf, dtype=float)
    for column, limit_attr, width_attr in _MARGIN_AXES:
        mu = mean[:, column]
        sd = std[:, column]
        shift = np.where(np.isfinite(sd), kappa * sd, 0.0)
        limit = float(getattr(constraints, limit_attr))
        width = float(getattr(constraints, width_attr))
        excess = (mu + shift - limit) / width
        worst = np.maximum(worst, np.where(np.isfinite(mu), excess, -np.inf))
    # margin = -worst-excess (higher is safer); -inf worst (all axes NaN) -> -inf.
    return np.where(np.isfinite(worst), -worst, -np.inf)


def rank_with_tiebreak(
    exploit: np.ndarray, margin: np.ndarray, tie_epsilon: float
) -> np.ndarray:
    """Composite ranking scalar: exploit primary, feasibility margin tie-break.

    Candidates are bucketed by ``round(exploit / tie_epsilon)`` so scores within
    the ``tie_epsilon`` band share a bucket; the within-bucket order is the
    fractional rank of ``margin`` (in ``[0, 1)`` so it never crosses a bucket
    boundary — the primary exploit ordering is preserved between buckets).
    ``tie_epsilon <= 0`` returns the exploit score unchanged (fallback). Rows with
    a non-finite exploit (out-of-region ``−inf``) are preserved so they sort last.
    """

    exploit = np.asarray(exploit, dtype=float)
    margin = np.asarray(margin, dtype=float)
    if not np.isfinite(tie_epsilon) or tie_epsilon <= 0.0 or exploit.size == 0:
        return exploit.copy()
    n = exploit.size
    bucket = np.round(exploit / float(tie_epsilon))
    # fractional rank of margin in [0, 1): higher margin -> larger fraction.
    keyed = np.where(np.isfinite(margin), margin, -np.inf)
    frac = np.argsort(np.argsort(keyed, kind="stable"), kind="stable") / float(n)
    composite = bucket + frac
    return np.where(np.isfinite(exploit), composite, exploit)


# --------------------------------------------------------------------------- #
# max_cycle_min_fr objective (user directive 2026-07-21)
# --------------------------------------------------------------------------- #
#: A gated-constraint violation must dominate the (cyclen − λ·F_r) trade so a
#: candidate that breaks F_q / CBC / |AO| can never outrank a constraint-feasible
#: one — a hierarchical HARD gate that still degrades gracefully (ranks the least-
#: infeasible first) when NO candidate is predicted-feasible yet.  Sized well above
#: the ~O(10²) EFPD scalar spread: a single-axis excess of one penalty-width
#: contributes ``TIER`` (10⁴), swamping any within-feasible cyclen/F_r difference.
_MAXCYCLE_CONSTRAINT_TIER = 1.0e4

#: Gated safety axes for max_cycle_min_fr — F_r (col 0) is DELIBERATELY absent
#: (it is the minimization objective, not a gate).  (surrogate column, limit attr,
#: penalty width) with the same per-axis widths the reward/criteria scorers use.
_MAXCYCLE_GATED_AXES: tuple[tuple[int, str, float], ...] = (
    (1, "cbc_limit", 25.0),
    (2, "f_q_limit", 0.05),
    (4, "ao_abs_limit", 0.02),
)


@dataclass(frozen=True)
class MaxCycleSpec:
    """Objective + gated limits for the ``max_cycle_min_fr`` campaign mode.

    MAXIMIZE cyclen, MINIMIZE F_r subject to F_q / CBC / |AO| (F_r is NOT gated).
    ``lam`` (λ) is the EFPD-per-unit-F_r trade in the exploit scalar
    ``cyclen_LCB − λ·F_r_UCB``; ``risk_z`` is the κ shift applied as an LCB on
    cyclen and a UCB on F_r and on every gated axis (conservative both ways).
    """

    lam: float = 100.0
    risk_z: float = 0.25
    cbc_limit: float = 1550.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30


@dataclass
class MaxCycleScore:
    """Result of :func:`score_max_cycle_min_fr` (all arrays length N).

    ``total`` is the single exploit ranking scalar (higher is better): the
    ``scalar`` cyclen/F_r trade, dominated hierarchically by any gated-constraint
    violation.  ``constraint_ok`` is the predicted-feasibility HARD screen
    (F_q / CBC / |AO| at +risk_z·σ), the campaign's model-side analogue of the
    verified constraint-feasibility bookkeeping (F_r excluded).
    """

    total: np.ndarray            # scalar − TIER·constraint_penalty (the rank key)
    scalar: np.ndarray           # cyclen_LCB − λ·F_r_UCB (the objective trade)
    cyclen_lcb: np.ndarray       # μ_cy − κσ_cy
    fr_ucb: np.ndarray           # μ_Fr + κσ_Fr
    constraint_penalty: np.ndarray   # Σ squared gated-axis UCB excess
    constraint_ok: np.ndarray    # predicted-feasible at +κσ (F_q/CBC/|AO|), bool


def make_maxcycle_constraints(spec: MaxCycleSpec) -> ConstraintConfig:
    """Vendor :class:`ConstraintConfig` for the p_feasible gate + feasibility
    margin in max_cycle_min_fr mode.

    F_r is left UNGATED (``f_r_limit`` set to a large sentinel) so the F_r Φ-factor
    in :func:`p_feasible` collapses to ≈ 1 and only F_q / CBC / |AO| gate.  The
    objective mode is irrelevant here (``trade_off``, unused — ranking is done by
    :func:`score_max_cycle_min_fr`).
    """

    big = 1.0e12
    return ConstraintConfig(
        f_r_limit=big,                       # F_r is an OBJECTIVE, never a gate
        cbc_limit=float(spec.cbc_limit),
        f_q_limit=float(spec.f_q_limit),
        ao_abs_limit=float(spec.ao_abs_limit),
        risk_z=float(spec.risk_z),
        objective_mode="trade_off",
    )


def score_max_cycle_min_fr(
    prediction: SurrogatePrediction, spec: MaxCycleSpec
) -> MaxCycleScore:
    """Score candidates for max_cycle_min_fr (higher ``total`` is better).

    Objective (risk-adjusted, on the *calibrated* std):

    * cyclen at its LCB ``μ_cy − κσ_cy`` (a conservative floor on the gain),
    * F_r at its UCB ``μ_Fr + κσ_Fr`` (a conservative ceiling on the peaking),
    * scalarized as ``scalar = cyclen_LCB − λ·F_r_UCB``.

    HARD constraint screen (hierarchical, dominant): F_q / CBC / |AO| each enter
    as a UCB-shifted upper-bound excess ``max(0, μ_c + κσ_c − limit)/width``; the
    summed-square penalty is multiplied by :data:`_MAXCYCLE_CONSTRAINT_TIER` and
    SUBTRACTED, so any predicted violation sinks a candidate below every
    constraint-feasible one while still ranking the least-infeasible first (no
    starvation before the first feasible label).  An axis the surrogate cannot
    predict (NaN σ) contributes no shift and no penalty but also cannot certify
    feasibility (``constraint_ok`` stays False for it).
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    kappa = float(spec.risk_z)
    shift = np.where(np.isfinite(std), kappa * std, 0.0)

    cyclen_lcb = mean[:, 3] - shift[:, 3]
    fr_ucb = mean[:, 0] + shift[:, 0]
    scalar = cyclen_lcb - float(spec.lam) * fr_ucb

    penalty = np.zeros(n, dtype=float)
    constraint_ok = np.ones(n, dtype=bool)
    for column, attr, width in _MAXCYCLE_GATED_AXES:
        limit = float(getattr(spec, attr))
        mu_c = mean[:, column]
        sd_c = std[:, column]
        known = np.isfinite(sd_c)
        ucb = mu_c + np.where(known, kappa * sd_c, 0.0)
        excess = np.maximum(0.0, ucb - limit) / float(width)
        penalty = penalty + excess ** 2
        constraint_ok = constraint_ok & known & (excess <= 1.0e-12)

    total = scalar - _MAXCYCLE_CONSTRAINT_TIER * penalty
    return MaxCycleScore(
        total=total, scalar=scalar, cyclen_lcb=cyclen_lcb, fr_ucb=fr_ucb,
        constraint_penalty=penalty, constraint_ok=constraint_ok,
    )


def score_pool_max_cycle(
    model: Any,
    ctx: CaseContext,
    candidates: Sequence[Candidate],
    spec: MaxCycleSpec,
    trust_region: "TrustRegion",
    *,
    tie_epsilon: float = 0.0,
) -> ScoredPool:
    """Score a candidate pool for the ``max_cycle_min_fr`` mode.

    Structurally identical to :func:`score_pool` (predict → trust-region hard gate
    → frontier σ-inflation → p_feas / exploit / margin / rank), but:

    * ``exploit`` is :func:`score_max_cycle_min_fr` ``total`` (cyclen LCB − λ·F_r
      UCB, dominated by the F_q/CBC/|AO| screen), NOT the target-cycle reward;
    * ``p_feas`` gates only F_q / CBC / |AO| (F_r ungated, :func:`make_maxcycle_constraints`);
    * ``margin`` / ``raw_epi`` / ``in_region`` are computed exactly as in
      :func:`score_pool`, so the wave composer consumes this pool unchanged.
    """

    candidates = list(candidates)
    patterns = [c.pattern for c in candidates]
    n = len(patterns)
    if n == 0:
        empty = np.zeros((0, 7))
        z = np.zeros(0)
        return ScoredPool(
            candidates=[], mean=empty, epistemic=empty.copy(), calibrated=empty.copy(),
            conv=z, p_feas=z.copy(), acq=z.copy(), raw_epi=z.copy(),
            in_region=np.zeros(0, dtype=bool), exploit=z.copy(),
            margin=z.copy(), rank=z.copy(),
        )

    prediction = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    conv = _safe_convergence(model, patterns, ctx)

    # Frontier σ-inflation + trust-region hard gate (identical to score_pool).
    calibrated = np.asarray(prediction.calibrated_std, dtype=float).copy()
    region = np.ones(n, dtype=bool)
    for i, cand in enumerate(candidates):
        feed = cand.pattern.feed
        ec = cand.e_core if cand.e_core is not None else ctx.e_core
        region[i] = trust_region.in_region(feed, ec)
        scale = trust_region.sigma_scale(feed, ec)
        if scale != 1.0:
            calibrated[i] = calibrated[i] * scale
    inflated = SurrogatePrediction(prediction.mean, prediction.epistemic_std, calibrated)

    constraints = make_maxcycle_constraints(spec)
    pf = np.where(region, p_feasible(inflated, constraints, convergence=conv), 0.0)

    mc = score_max_cycle_min_fr(inflated, spec)
    exploit = np.where(region, mc.total, -np.inf)
    margin = np.where(region, feasibility_margin(inflated, constraints), -np.inf)
    rank = rank_with_tiebreak(exploit, margin, tie_epsilon)

    return ScoredPool(
        candidates=candidates,
        mean=np.asarray(prediction.mean, dtype=float),
        epistemic=np.asarray(prediction.epistemic_std, dtype=float),
        calibrated=calibrated,
        conv=np.asarray(conv, dtype=float),
        p_feas=np.asarray(pf, dtype=float),
        acq=np.where(region, pf, 0.0),
        raw_epi=raw_epistemic(inflated),
        in_region=region,
        exploit=np.asarray(exploit, dtype=float),
        margin=np.asarray(margin, dtype=float),
        rank=np.asarray(rank, dtype=float),
    )


# --------------------------------------------------------------------------- #
# min_fr_max_cycle objective (user directive 2026-07-22 — revised hierarchy)
# --------------------------------------------------------------------------- #
#: Gated safety axes for min_fr_max_cycle — F_r (col 0) REJOINS the constraint set
#: (it is BOTH the primary objective AND a hard limit).  (surrogate column, limit
#: attr, penalty width); widths match the reward/criteria scorers.
_MINFR_GATED_AXES: tuple[tuple[int, str, float], ...] = (
    (0, "f_r_limit", 0.01),
    (1, "cbc_limit", 25.0),
    (2, "f_q_limit", 0.05),
    (4, "ao_abs_limit", 0.02),
)
#: predicted max-pin-burnup column + certification (mirrors min_fuel_cost /
#: fr_boundary col 6).  ADDED 2026-08-17: this mode alone screened NOTHING on pin
#: burnup, and both min_fr campaigns (N1_N2/f113, E1_E2/f109) reported 100% of
#: their "feasible" cores over the 80 GWd/tU LEU+ limit on the validated
#: prediction — undeliverable cores called feasible
#: (data/reports/fpcamp_E1E2_f109_results_20260817.md §7).
_MINFR_PINBU_COL = 6


@dataclass(frozen=True)
class MinFrSpec:
    """Objective + gated limits for the ``min_fr_max_cycle`` campaign mode.

    MINIMIZE F_r (primary) with cyclen maximization the secondary tie-break, subject
    to F_r <= ``f_r_limit`` AND F_q / CBC / |AO| (all four gated) AND the predicted
    ``max_pin_burnup <= pin_bu_limit``.  ``lam_fr`` (λ_Fr) sizes F_r to strictly
    dominate cyclen in the exploit scalar ``cyclen_LCB − λ_Fr·F_r_UCB``
    (default 1000: a 0.01 F_r reduction == 10 EFPD).

    ``pin_bu_limit`` defaults to **78.0**, not the 80.0 the other pin-gated modes
    use.  The LEU+ licensing figure is 80; the 2.0 GWd/tU haircut is MODEL margin,
    sized from the in-cell validation of the s1g pin head at E1_E2/f109 (n=33,
    MAE 1.84, bias −1.39 — it UNDER-predicts, so a core predicted at 80.0 is
    expected at ~81.4).  Gating the prediction at 78.0 keeps the corrected
    expectation under the real limit by roughly one MAE.  Raise it to 80.0 to gate
    on the licensing number directly (deck knob ``minfr_pin_bu_limit``).
    """

    lam_fr: float = 1000.0
    risk_z: float = 0.25
    f_r_limit: float = 1.55
    cbc_limit: float = 1550.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30
    pin_bu_limit: float = 78.0


@dataclass
class MinFrScore:
    """Result of :func:`score_min_fr_max_cycle` (all arrays length N).

    ``total`` is the exploit ranking scalar (higher is better): the ``scalar``
    F_r/cyclen trade dominated hierarchically by any of the FIVE gated-constraint
    violations (F_r / F_q / CBC / |AO| / predicted pin BU).  ``constraint_ok`` is
    the predicted all-five feasibility screen at +risk_z·σ.
    """

    total: np.ndarray            # scalar − TIER·constraint_penalty (the rank key)
    scalar: np.ndarray           # cyclen_LCB − λ_Fr·F_r_UCB (F_r strictly dominant)
    cyclen_lcb: np.ndarray       # μ_cy − κσ_cy
    fr_ucb: np.ndarray           # μ_Fr + κσ_Fr
    constraint_penalty: np.ndarray   # Σ squared gated-axis excess (incl. F_r + pin)
    constraint_ok: np.ndarray    # predicted-feasible (F_r/F_q/CBC/|AO|/pin BU), bool


def make_minfr_constraints(spec: MinFrSpec) -> ConstraintConfig:
    """Vendor :class:`ConstraintConfig` for the p_feasible gate + feasibility margin
    in min_fr_max_cycle mode — F_r is GATED at ``f_r_limit`` (all four axes gate)."""

    return ConstraintConfig(
        f_r_limit=float(spec.f_r_limit),
        cbc_limit=float(spec.cbc_limit),
        f_q_limit=float(spec.f_q_limit),
        ao_abs_limit=float(spec.ao_abs_limit),
        risk_z=float(spec.risk_z),
        objective_mode="trade_off",
    )


def score_min_fr_max_cycle(
    prediction: SurrogatePrediction, spec: MinFrSpec
) -> MinFrScore:
    """Score candidates for min_fr_max_cycle (higher ``total`` is better).

    Objective (risk-adjusted, on the *calibrated* std): ``scalar = cyclen_LCB −
    λ_Fr·F_r_UCB`` with cyclen at its LCB and F_r at its UCB (conservative both ways).
    With ``λ_Fr = 1000`` F_r strictly dominates — a 0.01 F_r reduction outweighs a
    10 EFPD cyclen gain, so the whole ~30 EFPD cyclen spread only orders candidates
    of near-identical F_r.

    HARD screen (hierarchical, dominant): F_r / F_q / CBC / |AO| each enter as a
    UCB-shifted upper-bound excess ``max(0, μ_c + κσ_c − limit)/width``, and the
    predicted ``max_pin_burnup`` (col 6) as ``max(0, pin_UCB − pin_bu_limit)``; the
    summed-square penalty × :data:`_MAXCYCLE_CONSTRAINT_TIER` is subtracted, so any
    predicted violation (including F_r > 1.55) sinks below every feasible candidate,
    least-infeasible first (no starvation before the first feasible label — which,
    empirically at this cell, may never arrive).
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    kappa = float(spec.risk_z)
    shift = np.where(np.isfinite(std), kappa * std, 0.0)

    cyclen_lcb = mean[:, 3] - shift[:, 3]
    fr_ucb = mean[:, 0] + shift[:, 0]
    scalar = cyclen_lcb - float(spec.lam_fr) * fr_ucb

    penalty = np.zeros(n, dtype=float)
    constraint_ok = np.ones(n, dtype=bool)
    for column, attr, width in _MINFR_GATED_AXES:
        limit = float(getattr(spec, attr))
        mu_c = mean[:, column]
        sd_c = std[:, column]
        known = np.isfinite(sd_c)
        ucb = mu_c + np.where(known, kappa * sd_c, 0.0)
        excess = np.maximum(0.0, ucb - limit) / float(width)
        penalty = penalty + excess ** 2
        constraint_ok = constraint_ok & known & (excess <= 1.0e-12)

    # -- predicted pin burnup (col 6) — copied verbatim from score_min_fuel_cost --
    # A trained pin head carries a finite σ → gate on its +κσ UCB (conservative,
    # like the other model axes).  The physics-estimator override is a σ-free POINT
    # estimate → gate on the mean.  Certification needs only a finite MEAN (MASTER
    # adjudicates the verified value).  A model with NO pin head predicts NaN here:
    # the penalty stays 0 (ranking unchanged) but ``constraint_ok`` goes False, so
    # an unscreened core is never SILENTLY called feasible — which is exactly the
    # failure this gate closes.
    pin_mu = mean[:, _MINFR_PINBU_COL]
    pin_sd = std[:, _MINFR_PINBU_COL]
    pin_known = np.isfinite(pin_mu)
    pin_ucb = pin_mu + np.where(np.isfinite(pin_sd), kappa * pin_sd, 0.0)
    pin_excess = np.where(pin_known,
                          np.maximum(0.0, pin_ucb - float(spec.pin_bu_limit)), 0.0)
    penalty = penalty + pin_excess ** 2
    constraint_ok = constraint_ok & pin_known & (pin_excess <= 1.0e-12)

    total = scalar - _MAXCYCLE_CONSTRAINT_TIER * penalty
    return MinFrScore(
        total=total, scalar=scalar, cyclen_lcb=cyclen_lcb, fr_ucb=fr_ucb,
        constraint_penalty=penalty, constraint_ok=constraint_ok,
    )


def score_pool_min_fr(
    model: Any,
    ctx: CaseContext,
    candidates: Sequence[Candidate],
    spec: MinFrSpec,
    trust_region: "TrustRegion",
    *,
    tie_epsilon: float = 0.0,
) -> ScoredPool:
    """Score a candidate pool for the ``min_fr_max_cycle`` mode.

    Structurally identical to :func:`score_pool_max_cycle`, but the exploit score is
    :func:`score_min_fr_max_cycle` ``total`` (F_r strictly dominant) and ``p_feas`` /
    ``margin`` gate ALL FOUR constraints including F_r (:func:`make_minfr_constraints`).
    """

    candidates = list(candidates)
    patterns = [c.pattern for c in candidates]
    n = len(patterns)
    if n == 0:
        empty = np.zeros((0, 7))
        z = np.zeros(0)
        return ScoredPool(
            candidates=[], mean=empty, epistemic=empty.copy(), calibrated=empty.copy(),
            conv=z, p_feas=z.copy(), acq=z.copy(), raw_epi=z.copy(),
            in_region=np.zeros(0, dtype=bool), exploit=z.copy(),
            margin=z.copy(), rank=z.copy(),
        )

    prediction = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    conv = _safe_convergence(model, patterns, ctx)

    calibrated = np.asarray(prediction.calibrated_std, dtype=float).copy()
    region = np.ones(n, dtype=bool)
    for i, cand in enumerate(candidates):
        feed = cand.pattern.feed
        ec = cand.e_core if cand.e_core is not None else ctx.e_core
        region[i] = trust_region.in_region(feed, ec)
        scale = trust_region.sigma_scale(feed, ec)
        if scale != 1.0:
            calibrated[i] = calibrated[i] * scale
    inflated = SurrogatePrediction(prediction.mean, prediction.epistemic_std, calibrated)

    constraints = make_minfr_constraints(spec)
    pf = np.where(region, p_feasible(inflated, constraints, convergence=conv), 0.0)

    mf = score_min_fr_max_cycle(inflated, spec)
    exploit = np.where(region, mf.total, -np.inf)
    margin = np.where(region, feasibility_margin(inflated, constraints), -np.inf)
    rank = rank_with_tiebreak(exploit, margin, tie_epsilon)

    return ScoredPool(
        candidates=candidates,
        mean=np.asarray(prediction.mean, dtype=float),
        epistemic=np.asarray(prediction.epistemic_std, dtype=float),
        calibrated=calibrated,
        conv=np.asarray(conv, dtype=float),
        p_feas=np.asarray(pf, dtype=float),
        acq=np.where(region, pf, 0.0),
        raw_epi=raw_epistemic(inflated),
        in_region=region,
        exploit=np.asarray(exploit, dtype=float),
        margin=np.asarray(margin, dtype=float),
        rank=np.asarray(rank, dtype=float),
    )


# --------------------------------------------------------------------------- #
# min_fuel_cost objective (user directive 2026-07-21 — minimize fresh fuel cost)
# --------------------------------------------------------------------------- #
#: A gated-constraint violation must dominate the fuel-cost / F_r trade so a
#: candidate breaking ANY of the six hard constraints (cyclen band both edges,
#: F_r, F_q, CBC, |AO|, predicted pin BU) can never outrank a constraint-feasible
#: one — hierarchical HARD gate that still ranks the least-infeasible first before
#: a feasible label exists.  Sized well above the FE scalar spread: the ga80
#: count-weighted FE ~500-900 [pos·w/o] and true-mass FE ~O(10³) [g U-235], so a
#: single-axis one-width excess (TIER = 1e6) swamps any within-feasible FE/F_r gap.
_FUELCOST_CONSTRAINT_TIER = 1.0e6

#: Finite-σ MODEL gated axes for min_fuel_cost — (surrogate column, spec limit
#: field, penalty width).  Same per-axis widths the reward/criteria scorers use.
#: F_r (col 0) is BOTH a hard limit here AND the secondary tie-break objective
#: (dual role, like min_fr_max_cycle).  The cyclen band (col 3, two-sided) and the
#: physics pin-BU axis (col 6, point estimate) are handled specially below.
_FUELCOST_MODEL_AXES: tuple[tuple[int, str, float], ...] = (
    (0, "f_r_limit", 0.01),
    (1, "cbc_limit", 25.0),
    (2, "f_q_limit", 0.05),
    (4, "ao_abs_limit", 0.02),
)
#: cyclen-band penalty column + width (EFPD); two-sided [cyclen_lo, cyclen_hi].
_FUELCOST_CYCLEN_COL = 3
#: predicted max-pin-burnup column (physics estimator override of model.predict) +
#: width (GWd/MTU).  Certified on the finite POINT estimate (no σ required) — the
#: pin-BU estimator is deterministic and MASTER does the final adjudication.
_FUELCOST_PINBU_COL = 6


@dataclass(frozen=True)
class MinFuelCostSpec:
    """Objective + hard limits for the ``min_fuel_cost`` campaign mode.

    MINIMIZE the fresh fuel-economics metric ``FE`` (total fresh U-235 charge,
    :func:`lpopt.data.fuel_types.fresh_fuel_charge`) as the PRIMARY objective, with
    F_r as the SECONDARY tie-break, subject to ALL SIX hard constraints:
    ``cyclen ∈ [cyclen_lo, cyclen_hi]`` (both edges), ``F_r ≤ f_r_limit``,
    ``F_q ≤ f_q_limit``, ``CBC ≤ cbc_limit``, ``|AO| ≤ ao_abs_limit``, and the
    predicted ``max_pin_burnup ≤ pin_bu_limit`` (LEU+ 80 GWd/MTU).

    Exploit scalar (risk-adjusted): ``scalar = −FE − λ_Fr·F_r_UCB``.  ``FE`` is
    position-invariant (it depends only on the fresh-assembly composition, not the
    LP layout), so within a fixed (feed, e_core) cell every LP shares the same FE
    and F_r decides — while ACROSS cells FE dominates.  ``lam_fr`` (default 20.0)
    is therefore a subordinate tie-break: a 0.1 F_r reduction is worth 2 FE units
    (ga80 [pos·w/o]), far below one feed step (≈4 fresh positions × enrichment ≈
    22 units) so FE never loses to F_r across cells, yet F_r fully orders the
    within-cell FE ties.  ``risk_z`` shifts F_r to its UCB (conservative) and the
    cyclen band / constraints to their risk-adjusted bounds.
    """

    lam_fr: float = 20.0
    risk_z: float = 0.25
    cyclen_lo: float = 615.0
    cyclen_hi: float = 635.0
    cyclen_width: float = 10.0
    f_r_limit: float = 1.55
    cbc_limit: float = 1550.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30
    pin_bu_limit: float = 80.0


@dataclass
class MinFuelCostScore:
    """Result of :func:`score_min_fuel_cost` (all arrays length N).

    ``total`` is the exploit ranking scalar (higher is better): the ``scalar``
    (−FE − λ_Fr·F_r_UCB) fuel-cost/F_r trade dominated hierarchically by ANY of the
    six gated-constraint violations.  ``constraint_ok`` is the predicted all-six
    feasibility screen (cyclen band + F_r/F_q/CBC/|AO| at +risk_z·σ + pin BU point
    estimate).  ``fe`` is the per-candidate fuel-economics metric (NaN where the
    fresh composition is unresolvable → ``scalar``/``total`` = −inf).
    """

    total: np.ndarray            # scalar − TIER·constraint_penalty (the rank key)
    scalar: np.ndarray           # −FE − λ_Fr·F_r_UCB (FE primary, F_r tie-break)
    fe: np.ndarray               # fresh U-235 charge (lower is better); NaN if N/A
    fr_ucb: np.ndarray           # μ_Fr + κσ_Fr
    cyclen_lcb: np.ndarray       # μ_cy − κσ_cy (lower band edge test)
    cyclen_ucb: np.ndarray       # μ_cy + κσ_cy (upper band edge test)
    constraint_penalty: np.ndarray   # Σ squared gated-axis excess (all six)
    constraint_ok: np.ndarray    # predicted-feasible over all six axes, bool


def make_minfuelcost_constraints(spec: MinFuelCostSpec) -> ConstraintConfig:
    """Vendor :class:`ConstraintConfig` for the p_feasible gate + feasibility margin
    in min_fuel_cost mode — F_r / CBC / F_q / |AO| gated at their limits (the
    ``p_feasible`` gate covers only these four columns; the cyclen band and pin-BU
    axis are enforced by :func:`score_min_fuel_cost`'s hierarchical penalty)."""

    return ConstraintConfig(
        f_r_limit=float(spec.f_r_limit),
        cbc_limit=float(spec.cbc_limit),
        f_q_limit=float(spec.f_q_limit),
        ao_abs_limit=float(spec.ao_abs_limit),
        risk_z=float(spec.risk_z),
        objective_mode="trade_off",
    )


def score_min_fuel_cost(
    prediction: SurrogatePrediction, spec: MinFuelCostSpec, fe: np.ndarray
) -> MinFuelCostScore:
    """Score candidates for min_fuel_cost (higher ``total`` is better).

    Objective: ``scalar = −FE − λ_Fr·F_r_UCB`` with F_r at its UCB (conservative).
    HARD screen (hierarchical, dominant): each of the SIX constraints enters as a
    normalized excess whose summed square × :data:`_FUELCOST_CONSTRAINT_TIER` is
    subtracted, so any predicted violation sinks below every feasible candidate,
    least-infeasible first (no starvation before the first feasible label):

    * cyclen band (col 3, two-sided): ``max(0, cyclen_lo − cy_LCB)`` and
      ``max(0, cy_UCB − cyclen_hi)`` (LCB/UCB conservative on both edges);
    * F_r / CBC / F_q / |AO| (cols 0/1/2/4): UCB upper-bound excess (finite-σ);
    * predicted max pin burnup (col 6): the physics point-estimate excess
      ``max(0, μ_pin − pin_bu_limit)`` — certified on the finite mean (no σ), since
      the estimator is deterministic and MASTER adjudicates the verified value.

    ``fe`` is the per-candidate fuel-economics metric; a non-finite ``fe`` (feed
    composition unresolvable) yields ``scalar = total = −inf`` so it sorts last.
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    kappa = float(spec.risk_z)
    fe = np.asarray(fe, dtype=float).reshape(-1)
    shift = np.where(np.isfinite(std), kappa * std, 0.0)

    fr_ucb = mean[:, 0] + shift[:, 0]
    scalar = np.where(np.isfinite(fe), -fe - float(spec.lam_fr) * fr_ucb, -np.inf)

    penalty = np.zeros(n, dtype=float)
    constraint_ok = np.ones(n, dtype=bool)

    # -- cyclen band (two-sided, col 3) ------------------------------------- #
    cy_mu = mean[:, _FUELCOST_CYCLEN_COL]
    cy_sd = std[:, _FUELCOST_CYCLEN_COL]
    cy_known = np.isfinite(cy_sd)
    cy_shift = np.where(cy_known, kappa * cy_sd, 0.0)
    cyclen_lcb = cy_mu - cy_shift
    cyclen_ucb = cy_mu + cy_shift
    w_cy = float(spec.cyclen_width)
    excess_lo = np.maximum(0.0, float(spec.cyclen_lo) - cyclen_lcb) / w_cy
    excess_hi = np.maximum(0.0, cyclen_ucb - float(spec.cyclen_hi)) / w_cy
    penalty = penalty + excess_lo ** 2 + excess_hi ** 2
    constraint_ok = constraint_ok & cy_known & (excess_lo <= 1.0e-12) & (excess_hi <= 1.0e-12)

    # -- finite-σ model axes (F_r / CBC / F_q / |AO|) ----------------------- #
    for column, attr, width in _FUELCOST_MODEL_AXES:
        limit = float(getattr(spec, attr))
        mu_c = mean[:, column]
        sd_c = std[:, column]
        known = np.isfinite(sd_c)
        ucb = mu_c + np.where(known, kappa * sd_c, 0.0)
        excess = np.maximum(0.0, ucb - limit) / float(width)
        penalty = penalty + excess ** 2
        constraint_ok = constraint_ok & known & (excess <= 1.0e-12)

    # -- predicted pin burnup (col 6) --------------------------------------- #
    # A trained pin head carries a finite σ → gate on its +κσ UCB (conservative,
    # like the other model axes).  The physics-estimator override is a σ-free POINT
    # estimate → gate on the mean.  Certification needs only a finite MEAN (the
    # point estimate is a valid screen; MASTER adjudicates the verified value).
    pin_mu = mean[:, _FUELCOST_PINBU_COL]
    pin_sd = std[:, _FUELCOST_PINBU_COL]
    pin_known = np.isfinite(pin_mu)
    pin_ucb = pin_mu + np.where(np.isfinite(pin_sd), kappa * pin_sd, 0.0)
    pin_excess = np.where(pin_known,
                          np.maximum(0.0, pin_ucb - float(spec.pin_bu_limit)), 0.0)
    penalty = penalty + pin_excess ** 2
    constraint_ok = constraint_ok & pin_known & (pin_excess <= 1.0e-12)

    total = np.where(np.isfinite(scalar), scalar - _FUELCOST_CONSTRAINT_TIER * penalty, -np.inf)
    return MinFuelCostScore(
        total=total, scalar=scalar, fe=fe, fr_ucb=fr_ucb,
        cyclen_lcb=cyclen_lcb, cyclen_ucb=cyclen_ucb,
        constraint_penalty=penalty, constraint_ok=constraint_ok,
    )


def fuel_charge_array(
    fuel: Any, library_id: str, candidates: Sequence[Candidate]
) -> tuple[np.ndarray, bool]:
    """Per-candidate fresh fuel-economics metric ``FE`` (see
    :func:`lpopt.data.fuel_types.fresh_fuel_charge`).

    Returns ``(fe, mass_weighted)`` — ``fe`` is an ``(N,)`` float array (NaN where
    the fresh composition is unresolvable or ``fuel`` is None); ``mass_weighted``
    is True only when EVERY candidate's every fed type carried a ``u_mass_g`` (true
    grams-U-235 scale), False for the ga80 count-weighted [pos·w/o] proxy.  A
    single count-weighted candidate makes the whole run count-weighted, keeping the
    metric on ONE scale for cross-candidate/cross-cell comparison.
    """

    from ..data.fuel_types import fresh_fuel_charge

    n = len(candidates)
    fe = np.full(n, np.nan, dtype=float)
    if fuel is None:
        return fe, False
    all_mass = True
    for i, cand in enumerate(candidates):
        charge, mass_weighted = fresh_fuel_charge(fuel, library_id, cand.pattern.batch_feed())
        if charge is not None:
            fe[i] = charge
        if not mass_weighted:
            all_mass = False
    return fe, bool(all_mass)


def score_pool_min_fuel_cost(
    model: Any,
    ctx: CaseContext,
    candidates: Sequence[Candidate],
    spec: MinFuelCostSpec,
    trust_region: "TrustRegion",
    *,
    tie_epsilon: float = 0.0,
    fuel: Any = None,
    library_id: str | None = None,
) -> ScoredPool:
    """Score a candidate pool for the ``min_fuel_cost`` mode.

    Structurally identical to :func:`score_pool_min_fr` (same trust-region σ
    inflation + p_feas gate + margin tie-break contract), but the exploit score is
    :func:`score_min_fuel_cost` ``total`` (fuel-cost minimization with F_r
    tie-break and all six hard constraints).  ``fuel`` / ``library_id`` supply the
    fresh-composition FE metric; when ``fuel`` is None every FE is NaN and the pool
    ranks by the constraint penalty alone (degenerate — callers pass the library).
    """

    candidates = list(candidates)
    patterns = [c.pattern for c in candidates]
    n = len(patterns)
    lib = library_id or ctx.library_id
    if n == 0:
        empty = np.zeros((0, 7))
        z = np.zeros(0)
        return ScoredPool(
            candidates=[], mean=empty, epistemic=empty.copy(), calibrated=empty.copy(),
            conv=z, p_feas=z.copy(), acq=z.copy(), raw_epi=z.copy(),
            in_region=np.zeros(0, dtype=bool), exploit=z.copy(),
            margin=z.copy(), rank=z.copy(),
        )

    prediction = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    conv = _safe_convergence(model, patterns, ctx)

    calibrated = np.asarray(prediction.calibrated_std, dtype=float).copy()
    region = np.ones(n, dtype=bool)
    for i, cand in enumerate(candidates):
        feed = cand.pattern.feed
        ec = cand.e_core if cand.e_core is not None else ctx.e_core
        region[i] = trust_region.in_region(feed, ec)
        scale = trust_region.sigma_scale(feed, ec)
        if scale != 1.0:
            calibrated[i] = calibrated[i] * scale
    inflated = SurrogatePrediction(prediction.mean, prediction.epistemic_std, calibrated)

    constraints = make_minfuelcost_constraints(spec)
    pf = np.where(region, p_feasible(inflated, constraints, convergence=conv), 0.0)

    fe, _ = fuel_charge_array(fuel, lib, candidates)
    fc = score_min_fuel_cost(inflated, spec, fe)
    exploit = np.where(region, fc.total, -np.inf)
    margin = np.where(region, feasibility_margin(inflated, constraints), -np.inf)
    rank = rank_with_tiebreak(exploit, margin, tie_epsilon)

    return ScoredPool(
        candidates=candidates,
        mean=np.asarray(prediction.mean, dtype=float),
        epistemic=np.asarray(prediction.epistemic_std, dtype=float),
        calibrated=calibrated,
        conv=np.asarray(conv, dtype=float),
        p_feas=np.asarray(pf, dtype=float),
        acq=np.where(region, pf, 0.0),
        raw_epi=raw_epistemic(inflated),
        in_region=region,
        exploit=np.asarray(exploit, dtype=float),
        margin=np.asarray(margin, dtype=float),
        rank=np.asarray(rank, dtype=float),
    )


# --------------------------------------------------------------------------- #
# fr_boundary objective (fr_boundary campaign, user directive 2026-07-22 —
# F_r=1.55 licensing-boundary training-data campaign)
# --------------------------------------------------------------------------- #
#: Band-shaping penalty coefficient for fr_boundary (MID-TIER — below the hard-gate
#: TIER, above the in-band F_r scalar spread).  Sized to 100 (not 10) so an
#: out-of-band predicted F_r provably sinks below the WHOLE in-band range: the net
#: slope below band is ``1 − 100 = −99`` per unit F_r, so any candidate more than
#: (band_hi − band_lo)/99 ≈ 0.0026 outside the band drops below every in-band
#: candidate (band width 0.25).  A 10× coefficient left predicted-F_r 1.44 able to
#: outrank in-band 1.56 (OOD-low mode-collapse) — 100× closes that hole.
_FRBOUNDARY_BAND_COEFF = 100.0

#: Finite-σ MODEL gated axes for fr_boundary — CBC / F_q / |AO| ONLY (F_r is a PURE
#: objective, NEVER gated; identical to :data:`_MAXCYCLE_GATED_AXES`).
_FRBOUNDARY_GATED_AXES: tuple[tuple[int, str, float], ...] = _MAXCYCLE_GATED_AXES
#: predicted max-pin-burnup column + certification (mirrors min_fuel_cost col 6).
_FRBOUNDARY_PINBU_COL = 6


@dataclass(frozen=True)
class MinFrBoundarySpec:
    """Objective + gated limits for the ``fr_boundary`` campaign mode.

    MINIMIZE F_r as a PURE objective (F_r is NOT a hard constraint — no
    ``f_r_limit`` field) while biasing candidates toward the F_r=1.55 licensing
    boundary via a MID-TIER band-shaping penalty on the predicted F_r MEAN.
    Subject to F_q / CBC / |AO| (gated at their limits) and a predicted
    ``max_pin_burnup ≤ pin_bu_limit`` screen; cyclen is recorded but NEVER gated
    (no cyclen / λ fields so cyclen cannot leak into the ranking).

    ``band_lo`` / ``band_hi`` bound the plausibility window on the model's F_r
    location estimate: a predicted F_r mean outside it is pushed below the whole
    in-band range so the search does not mode-collapse onto OOD-low fantasy F_r.
    """

    risk_z: float = 0.25
    cbc_limit: float = 1550.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30
    pin_bu_limit: float = 80.0
    band_lo: float = 1.45
    band_hi: float = 1.70


@dataclass
class MinFrBoundaryScore:
    """Result of :func:`score_fr_boundary` (all arrays length N).

    ``total`` is the exploit ranking scalar (higher is better): the ``scalar``
    (−F_r UCB) dominated hierarchically by any CBC / F_q / |AO| / pin-BU violation,
    then shaped by the MID-TIER out-of-band penalty on the F_r MEAN.
    ``constraint_ok`` is the predicted CBC/F_q/|AO|+pin feasibility screen at
    +risk_z·σ (F_r is EXCLUDED — it is an objective, not a constraint).
    """

    total: np.ndarray            # scalar − TIER·penalty − band_penalty (rank key)
    scalar: np.ndarray           # −(μ_Fr + κσ_Fr)  (minimize F_r)
    fr_ucb: np.ndarray           # μ_Fr + κσ_Fr
    fr_mean: np.ndarray          # μ_Fr (the band term uses the MEAN, never the UCB)
    band_penalty: np.ndarray     # _FRBOUNDARY_BAND_COEFF · out-of-band excess (μ_Fr)
    constraint_penalty: np.ndarray   # Σ squared gated-axis excess (CBC/F_q/|AO|/pin)
    constraint_ok: np.ndarray    # predicted-feasible over CBC/F_q/|AO|+pin, bool


def make_fr_boundary_constraints(spec: MinFrBoundarySpec) -> ConstraintConfig:
    """Vendor :class:`ConstraintConfig` for the p_feasible gate + feasibility
    margin in fr_boundary mode.

    F_r is left UNGATED (``f_r_limit`` = large sentinel) — it is a PURE objective,
    so :func:`p_feasible` / :func:`feasibility_margin` gate ONLY F_q / CBC / |AO|
    (identical to :func:`make_maxcycle_constraints`).  No cyclen leakage.
    """

    big = 1.0e12
    return ConstraintConfig(
        f_r_limit=big,                       # F_r is an OBJECTIVE, never a gate
        cbc_limit=float(spec.cbc_limit),
        f_q_limit=float(spec.f_q_limit),
        ao_abs_limit=float(spec.ao_abs_limit),
        risk_z=float(spec.risk_z),
        objective_mode="trade_off",
    )


def score_fr_boundary(
    prediction: SurrogatePrediction, spec: MinFrBoundarySpec
) -> MinFrBoundaryScore:
    """Score candidates for fr_boundary (higher ``total`` is better).

    Objective (risk-adjusted): ``scalar = −(μ_Fr + κσ_Fr)`` — MINIMIZE F_r at its
    UCB (conservative).  HARD screen (hierarchical, dominant, × 1e4): F_q / CBC /
    |AO| UCB excess + the predicted pin-BU point excess, so any predicted violation
    sinks below every feasible candidate, least-infeasible first (no starvation
    before the first feasible label).  MID-TIER band shaping:

        band_penalty = 100 · max(0, band_lo − μ_Fr, μ_Fr − band_hi)

    subtracted from ``total``.  The band term deliberately uses the F_r MEAN, NOT
    the UCB: the band is an OOD/plausibility filter on the model's *location*
    estimate, so risk-inflating it with +σ would let high-σ OOD-low candidates dodge
    the low-side penalty.  The 100× coefficient (see :data:`_FRBOUNDARY_BAND_COEFF`)
    makes the net below-band slope −99/unit so any candidate >~0.0026 outside the
    band sinks below the entire in-band F_r range — closing the OOD-low
    mode-collapse hole a 10× coefficient left open.  An axis the surrogate cannot
    predict (NaN σ) contributes no shift and no penalty but cannot certify
    feasibility (``constraint_ok`` stays False for it).
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    kappa = float(spec.risk_z)
    shift = np.where(np.isfinite(std), kappa * std, 0.0)

    fr_mean = mean[:, 0]
    fr_ucb = fr_mean + shift[:, 0]
    scalar = -fr_ucb

    penalty = np.zeros(n, dtype=float)
    constraint_ok = np.ones(n, dtype=bool)

    # -- finite-σ model axes (CBC / F_q / |AO|) — F_r EXCLUDED ---------------- #
    for column, attr, width in _FRBOUNDARY_GATED_AXES:
        limit = float(getattr(spec, attr))
        mu_c = mean[:, column]
        sd_c = std[:, column]
        known = np.isfinite(sd_c)
        ucb = mu_c + np.where(known, kappa * sd_c, 0.0)
        excess = np.maximum(0.0, ucb - limit) / float(width)
        penalty = penalty + excess ** 2
        constraint_ok = constraint_ok & known & (excess <= 1.0e-12)

    # -- predicted pin burnup (col 6) — copied verbatim from score_min_fuel_cost
    # A trained pin head carries a finite σ → gate on its +κσ UCB (conservative);
    # the physics-estimator override is a σ-free POINT estimate → gate on the mean.
    # Certification needs only a finite MEAN (MASTER adjudicates the verified value).
    pin_mu = mean[:, _FRBOUNDARY_PINBU_COL]
    pin_sd = std[:, _FRBOUNDARY_PINBU_COL]
    pin_known = np.isfinite(pin_mu)
    pin_ucb = pin_mu + np.where(np.isfinite(pin_sd), kappa * pin_sd, 0.0)
    pin_excess = np.where(pin_known,
                          np.maximum(0.0, pin_ucb - float(spec.pin_bu_limit)), 0.0)
    penalty = penalty + pin_excess ** 2
    constraint_ok = constraint_ok & pin_known & (pin_excess <= 1.0e-12)

    # -- MID-TIER band shaping on the F_r MEAN (OOD/plausibility filter) ------ #
    band_excess = np.maximum.reduce([
        np.zeros(n, dtype=float),
        float(spec.band_lo) - fr_mean,
        fr_mean - float(spec.band_hi),
    ])
    band_penalty = _FRBOUNDARY_BAND_COEFF * band_excess

    total = scalar - _MAXCYCLE_CONSTRAINT_TIER * penalty - band_penalty
    return MinFrBoundaryScore(
        total=total, scalar=scalar, fr_ucb=fr_ucb, fr_mean=fr_mean,
        band_penalty=band_penalty,
        constraint_penalty=penalty, constraint_ok=constraint_ok,
    )


def score_pool_fr_boundary(
    model: Any,
    ctx: CaseContext,
    candidates: Sequence[Candidate],
    spec: MinFrBoundarySpec,
    trust_region: "TrustRegion",
    *,
    tie_epsilon: float = 0.0,
) -> ScoredPool:
    """Score a candidate pool for the ``fr_boundary`` mode.

    Structurally identical to :func:`score_pool_min_fr` (same trust-region σ
    inflation + p_feas gate + margin tie-break contract), but the exploit score is
    :func:`score_fr_boundary` ``total`` (pure F_r minimization + band shaping) and
    ``p_feas`` / ``margin`` gate ONLY F_q / CBC / |AO| (F_r ungated,
    :func:`make_fr_boundary_constraints`).
    """

    candidates = list(candidates)
    patterns = [c.pattern for c in candidates]
    n = len(patterns)
    if n == 0:
        empty = np.zeros((0, 7))
        z = np.zeros(0)
        return ScoredPool(
            candidates=[], mean=empty, epistemic=empty.copy(), calibrated=empty.copy(),
            conv=z, p_feas=z.copy(), acq=z.copy(), raw_epi=z.copy(),
            in_region=np.zeros(0, dtype=bool), exploit=z.copy(),
            margin=z.copy(), rank=z.copy(),
        )

    prediction = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    conv = _safe_convergence(model, patterns, ctx)

    calibrated = np.asarray(prediction.calibrated_std, dtype=float).copy()
    region = np.ones(n, dtype=bool)
    for i, cand in enumerate(candidates):
        feed = cand.pattern.feed
        ec = cand.e_core if cand.e_core is not None else ctx.e_core
        region[i] = trust_region.in_region(feed, ec)
        scale = trust_region.sigma_scale(feed, ec)
        if scale != 1.0:
            calibrated[i] = calibrated[i] * scale
    inflated = SurrogatePrediction(prediction.mean, prediction.epistemic_std, calibrated)

    constraints = make_fr_boundary_constraints(spec)
    pf = np.where(region, p_feasible(inflated, constraints, convergence=conv), 0.0)

    fb = score_fr_boundary(inflated, spec)
    exploit = np.where(region, fb.total, -np.inf)
    margin = np.where(region, feasibility_margin(inflated, constraints), -np.inf)
    rank = rank_with_tiebreak(exploit, margin, tie_epsilon)

    return ScoredPool(
        candidates=candidates,
        mean=np.asarray(prediction.mean, dtype=float),
        epistemic=np.asarray(prediction.epistemic_std, dtype=float),
        calibrated=calibrated,
        conv=np.asarray(conv, dtype=float),
        p_feas=np.asarray(pf, dtype=float),
        acq=np.where(region, pf, 0.0),
        raw_epi=raw_epistemic(inflated),
        in_region=region,
        exploit=np.asarray(exploit, dtype=float),
        margin=np.asarray(margin, dtype=float),
        rank=np.asarray(rank, dtype=float),
    )


# --------------------------------------------------------------------------- #
# flat_power objective — FLATNESS-NATIVE (program 20260725 §1.2 / §1.3 / §2.1)
# --------------------------------------------------------------------------- #
#: A gated-constraint violation must dominate the flatness objective so a
#: candidate breaking a hard limit can never outrank a feasible one.  The scalar
#: is now O(1)-to-O(10) in z units (peak/PEAK_SCALE ~ 4, cov term ~ 2), so a
#: one-width excess contributing TIER = 1e4 is still overwhelming.
_FLATPOWER_CONSTRAINT_TIER = 1.0e4

#: Finite-σ MODEL gated axes for flat_power — (surrogate column, spec limit field,
#: penalty width).
#:
#: **F_r (column 0) is DELIBERATELY ABSENT.**  It used to sit here with width 0.01,
#: which made a 0.02 overshoot worth 4e4 — four times the whole constraint tier —
#: so the "flatness" mode was in practice ranking by F_r, exactly the thing the
#: user retired from the objective.  F_r is now a BINARY SAFETY GATE
#: (:func:`flatpower_fr_gate` / ``fr_gate_violated`` below): it can veto a
#: candidate, but it can no longer grade one.
#:
#: cyclen (col 3) is absent as record-only (same contract as fr_boundary); pin BU
#: (col 6) is a point estimate, handled separately below.
_FLATPOWER_MODEL_AXES: tuple[tuple[int, str, float], ...] = (
    (1, "cbc_limit", 25.0),
    (2, "f_q_limit", 0.05),
    (4, "ao_abs_limit", 0.02),
)
_FLATPOWER_FR_COL = 0
_FLATPOWER_PINBU_COL = 6

#: Pessimism multiplier on the per-cell map-head sigma in the bias-corrected
#: safety gate ``1.70 − bias_cell − k·sigma_cell`` (program §2.1, decision D1).
#: Sourced from :mod:`..data.map_calibration` so the artifact that ships the
#: ``fr_sigma`` and the gate that consumes it can never disagree about ``k``.
FLATPOWER_GATE_K = _MC.GATE_K


@dataclass(frozen=True)
class FlatPowerSpec:
    """Objective + hard limits for the FLATNESS-NATIVE ``flat_power`` mode.

    The objective is the program §1.2 acquisition scalar

    .. code-block:: text

        z_peak = peak_UCB / peak_scale                 # PRIMARY   (weight 1.0)
        z_cov  = cov_UCB  / cov_scale                  # SECONDARY (weight w_cov)
        scalar = -( z_peak + w_cov * z_cov )
        peak_UCB = peak_mean + risk_z * peak_std
        cov_UCB  = cov_mean  + risk_z * cov_std

    with ``node_peak`` PRIMARY because within a cell rho(node_peak, F_r) = 0.983
    and the partial rho of ``map_cov`` given ``node_peak`` is only 0.107 — the
    licensing-relevant signal rides on the peak.  ``peak_scale`` / ``cov_scale``
    come from :class:`..data.flat_scale.FlatScale` (per-cell by default) so the
    declared 1 : ``w_cov`` ratio is what each cell actually realizes.

    **F_r is a SAFETY GATE, not an objective and not a graded penalty.**
    ``fr_limit`` (1.70, decision D1) vetoes a candidate whose F_r UCB exceeds it;
    when a per-cell map-head bias correction is available, ``fr_bias`` /
    ``fr_sigma`` tighten it to ``1.70 − bias − 0.5·sigma`` (:attr:`fr_gate`).
    Without that correction the gate HOLDS at 1.70 — relaxing it while the map
    head is optimistic (fold C node_peak bias −0.147) would amplify the winner's
    curse, which is why the draft's 1.75 was rejected.

    The remaining hard set is unchanged: ``F_q``, ``CBC``, ``|AO|`` and predicted
    ``max_pin_burnup ≤ pin_bu_limit`` (LEU+ 80).  ``cyclen`` is record-only.
    """

    risk_z: float = 0.25
    #: SECONDARY-term weight (program §1.2 default, decision D4).
    w_cov: float = _FS.DEFAULT_W_COV
    #: Normalizers; a campaign overrides these with its cell's fitted values.
    peak_scale: float = _FS.DEFAULT_PEAK_SCALE
    cov_scale: float = _FS.DEFAULT_COV_SCALE
    #: F_r SAFETY GATE (D1) and its optional per-cell bias correction.
    fr_limit: float = 1.70
    fr_bias: float | None = None
    fr_sigma: float | None = None
    #: Map-head LEVEL calibration (``map_calibration.json``, program §2.1).  Each
    #: is ``median(pred - actual)`` for that target, so the de-biased level is
    #: ``mean - bias``; ``*_sigma_extra`` is the dispersion the ensemble spread
    #: does not carry, folded into the UCB as
    #: ``sqrt(sigma_ens^2 + sigma_extra^2)``.  All four default to ``None``, which
    #: is byte-identical to the raw ensemble behaviour.
    peak_bias: float | None = None
    peak_sigma_extra: float | None = None
    cov_bias: float | None = None
    cov_sigma_extra: float | None = None
    cbc_limit: float = 1550.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30
    pin_bu_limit: float = 80.0
    #: ENGINEERING-RULE SOFT PENALTY (optional; ``None``/empty/all-zero == the
    #: byte-identical previous behaviour).  ``{metric: weight}`` over the
    #: VALIDATED arrangement metrics of :mod:`.rule_metrics`
    #: (:data:`.rule_metrics.VALIDATED_PENALTY_METRICS` = rm1 / rm1i / rm2 /
    #: rm2i), subtracted from the exploit ``total`` in the SAME units as the
    #: normalized flatness scalar.  It is a soft shaping term by construction: it
    #: reorders near-ties, it can never veto, and it is NOT part of the hard
    #: constraint tier — the source report's own lesson is that promoting a
    #: loading heuristic to a hard constraint truncates the search space.
    #: RM3 / RM4 / RM5 / RM6 are refused by :func:`.rule_metrics.rule_penalty`
    #: (report-only; see that module's docstring for the measured evidence).
    rule_penalty_weights: Mapping[str, float] | None = None

    @property
    def fr_gate(self) -> float:
        """The effective F_r safety gate (decision D1).

        ``1.70 − bias_cell − k·sigma_cell`` when a map-head bias correction is
        available for the cell, else the unmodified ``fr_limit`` (1.70 held).
        """
        return flatpower_fr_gate(self)


def flatpower_fr_gate(spec: "FlatPowerSpec") -> float:
    """Effective F_r safety gate for ``spec`` (program §2.1, decision D1).

    Free function so the rule has one implementation and a test can pin the
    "no correction available -> hold 1.70" branch explicitly.

    The result is CLAMPED at ``fr_limit``: a fitted correction may TIGHTEN a
    licensing-adjacent safety gate but may never loosen it.  Loosening on the
    strength of a fitted bias is precisely the draft's rejected 1.75 move, and a
    hand-edited or stale artifact must not be able to reintroduce it.
    """
    base = float(spec.fr_limit)
    if spec.fr_bias is None:
        return base
    sigma = float(spec.fr_sigma) if spec.fr_sigma is not None else 0.0
    if not math.isfinite(sigma):
        sigma = 0.0
    bias = float(spec.fr_bias)
    if not math.isfinite(bias):
        return base
    return min(base, base - bias - FLATPOWER_GATE_K * abs(sigma))


def _debias(values: np.ndarray, bias: float | None) -> np.ndarray:
    """``values - bias`` with the map-head level calibration (no-op when absent).

    ``bias`` is ``median(pred - actual)`` (:mod:`..data.map_calibration`), so a
    NEGATIVE bias — the champion's optimism, −0.147 on ``node_peak`` — RAISES the
    level the objective minimizes.  Non-finite values pass through untouched.
    """
    if bias is None or not math.isfinite(float(bias)) or float(bias) == 0.0:
        return values
    return values - float(bias)


def _inflate(sigma: np.ndarray, sigma_extra: float | None) -> np.ndarray:
    """``sqrt(sigma^2 + sigma_extra^2)`` — the CALIBRATED UCB spread.

    The raw ensemble spread is an epistemic disagreement statistic: when every
    member shares an extrapolation bias it is small and the UCB's pessimism
    vanishes exactly where it is needed (program §2.1).  ``sigma_extra`` is the
    residual dispersion the ensemble did not have on the honest slice, so the
    combination can only ADD pessimism and reduces to the identity when no
    calibration is available.
    """
    if (sigma_extra is None or not math.isfinite(float(sigma_extra))
            or float(sigma_extra) <= 0.0):
        return sigma
    return np.sqrt(sigma * sigma + float(sigma_extra) ** 2)


@dataclass
class FlatPowerScore:
    """Result of :func:`score_flat_power` (all arrays length N).

    ``total`` is the exploit ranking scalar (higher is better): ``scalar`` (the
    negated weighted flatness z-sum) dominated hierarchically by the graded
    constraint penalty AND by the binary F_r safety-gate veto.
    """

    total: np.ndarray            # scalar − TIER·(penalty + fr_gate_violated)
    scalar: np.ndarray           # −( z_peak + w_cov·z_cov )
    z_peak: np.ndarray           # peak_UCB / peak_scale (PRIMARY)
    z_cov: np.ndarray            # cov_UCB / cov_scale   (SECONDARY, NaN if absent)
    peak_ucb: np.ndarray         # node_peak mean + κ·std  (physical units)
    cov_ucb: np.ndarray          # map_cov  mean + κ·std   (physical units)
    fr_ucb: np.ndarray           # F_r mean + κ·σ (gate input, NOT the objective)
    fr_gate_violated: np.ndarray     # bool — the SAFETY veto
    constraint_penalty: np.ndarray   # Σ squared gated-axis excess (F_r EXCLUDED)
    constraint_ok: np.ndarray    # predicted-feasible over every hard axis, bool
    #: SOFT engineering-rule shaping term already subtracted from ``total``
    #: (all-zero unless ``spec.rule_penalty_weights`` is set AND the caller
    #: supplied the patterns).  Reported so a run can show what it cost.
    rule_penalty: np.ndarray | None = None


def make_flatpower_constraints(spec: FlatPowerSpec) -> ConstraintConfig:
    """Vendor :class:`ConstraintConfig` for the p_feasible gate + feasibility
    margin in flat_power mode.

    ``f_r_limit`` is the RETIRED sentinel ``1e12`` (the same device
    :func:`make_maxcycle_constraints` uses for an ungated axis).  That removes F_r
    from BOTH downstream users in one place: its Φ-factor in :func:`p_feasible`
    collapses to ≈ 1, and its normalized excess in :func:`feasibility_margin`
    becomes ≈ −1e12 so it can never be the worst axis and can never drive the
    tie-break.  Program §2.2: the tie-break that F_r used to own is replaced at
    the DELIVERY stage by ``compliance_margin`` (:mod:`.delivery`), not inside the
    search objective.  The safety veto lives in :func:`score_flat_power`.
    """

    return ConstraintConfig(
        f_r_limit=1.0e12,                    # F_r RETIRED from p_feas / margin
        cbc_limit=float(spec.cbc_limit),
        f_q_limit=float(spec.f_q_limit),
        ao_abs_limit=float(spec.ao_abs_limit),
        risk_z=float(spec.risk_z),
        objective_mode="trade_off",
    )


def score_flat_power(
    prediction: SurrogatePrediction,
    peak_mean: np.ndarray,
    peak_std: np.ndarray,
    spec: FlatPowerSpec,
    cov_mean: np.ndarray | None = None,
    cov_std: np.ndarray | None = None,
    patterns: Sequence[Pattern] | None = None,
) -> FlatPowerScore:
    """Score candidates for flat_power (higher ``total`` is better).

    Objective (program §1.2)::

        scalar = −( peak_UCB/peak_scale + w_cov · cov_UCB/cov_scale )

    both terms UCB-conservatized at ``risk_z`` (0.25) — assume the peak and the
    spread are as high as the model's uncertainty allows.  ``node_peak`` is the
    PRIMARY term at weight 1.0; ``map_cov`` is the SECONDARY shaping term at
    ``w_cov`` (0.5).  A candidate with no predicted peak scores ``−inf``; a
    candidate with a peak but no CoV keeps the primary term and drops the
    secondary one (weight 0), rather than being discarded.

    HARD screen (hierarchical, dominant):

    * graded — F_q / CBC / |AO| UCB excesses and the pin-BU point estimate, summed
      squared × :data:`_FLATPOWER_CONSTRAINT_TIER`;
    * binary — the F_r SAFETY GATE (:attr:`FlatPowerSpec.fr_gate`).  A violation
      subtracts exactly one TIER: it vetoes, it does not grade.  F_r no longer
      appears in the graded tier list at all, so the mode can never rank its
      candidates by F_r.

    cyclen is never read (record-only contract).

    LEVEL CALIBRATION (program §2.1).  ``node_peak`` / ``map_cov`` are consumed
    as physical levels, and the map head is optimistic about both (fold C bias
    −0.147 / −0.058).  When ``spec`` carries the ``map_calibration.json``
    numbers, each mean is de-biased (``mean - bias``) and each ensemble sigma is
    replaced by the calibrated ``sqrt(sigma_ens^2 + sigma_extra^2)`` BEFORE the
    UCB is formed — the raw spread cannot express a bias every member shares, so
    without this the ``risk_z`` pessimism is smallest exactly where the head is
    furthest from its training distribution.  With no calibration the arithmetic
    is byte-identical to the raw-ensemble form.

    ENGINEERING-RULE SOFT PENALTY (optional, default OFF).  When ``spec`` carries
    ``rule_penalty_weights`` AND ``patterns`` is supplied, the weighted sum of the
    VALIDATED arrangement metrics of :mod:`.rule_metrics` (fresh-fresh face /
    diagonal adjacency, whole-core and inboard) is SUBTRACTED from ``total``.  It
    sits OUTSIDE the constraint tier by design: it is a preference, not a screen,
    so it can reorder near-ties but can never make a candidate infeasible and can
    never veto.  RM3 / RM4 / RM5 / RM6 are refused as weights (report-only — see
    :mod:`.rule_metrics`).  With no weights (the default) the penalty is an exact
    zero vector and the arithmetic is byte-identical to the previous form.
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    kappa = float(spec.risk_z)

    pk_m = _debias(np.asarray(peak_mean, dtype=float).reshape(-1), spec.peak_bias)
    pk_s = _inflate(np.asarray(peak_std, dtype=float).reshape(-1),
                    spec.peak_sigma_extra)
    peak_ucb = pk_m + kappa * np.where(np.isfinite(pk_s), pk_s, 0.0)
    if cov_mean is None:
        cv_m = np.full(n, np.nan)
        cv_s = np.full(n, np.nan)
    else:
        cv_m = _debias(np.asarray(cov_mean, dtype=float).reshape(-1), spec.cov_bias)
        cv_s = _inflate(
            np.full(n, np.nan) if cov_std is None
            else np.asarray(cov_std, dtype=float).reshape(-1),
            spec.cov_sigma_extra)
    cov_ucb = cv_m + kappa * np.where(np.isfinite(cv_s), cv_s, 0.0)

    z_peak = peak_ucb / float(spec.peak_scale)
    z_cov = cov_ucb / float(spec.cov_scale)
    peak_ok = np.isfinite(z_peak)
    cov_term = np.where(np.isfinite(z_cov), float(spec.w_cov) * z_cov, 0.0)
    scalar = np.where(peak_ok,
                      -(np.where(peak_ok, z_peak, 0.0) + cov_term), -np.inf)

    penalty = np.zeros(n, dtype=float)
    constraint_ok = np.ones(n, dtype=bool)
    for column, attr, width in _FLATPOWER_MODEL_AXES:
        limit = float(getattr(spec, attr))
        mu_c = mean[:, column]
        sd_c = std[:, column]
        known = np.isfinite(sd_c)
        ucb = mu_c + np.where(known, kappa * sd_c, 0.0)
        excess = np.maximum(0.0, ucb - limit) / float(width)
        penalty = penalty + excess ** 2
        constraint_ok = constraint_ok & known & (excess <= 1.0e-12)
    # predicted pin burnup (col 6): trained head UCB if σ finite, else point est.
    pin_mu = mean[:, _FLATPOWER_PINBU_COL]
    pin_sd = std[:, _FLATPOWER_PINBU_COL]
    pin_known = np.isfinite(pin_mu)
    pin_ucb = pin_mu + np.where(np.isfinite(pin_sd), kappa * pin_sd, 0.0)
    pin_excess = np.where(pin_known, np.maximum(0.0, pin_ucb - float(spec.pin_bu_limit)), 0.0)
    penalty = penalty + pin_excess ** 2
    constraint_ok = constraint_ok & pin_known & (pin_excess <= 1.0e-12)

    # F_r SAFETY GATE — binary veto at the (optionally bias-corrected) limit.
    gate = flatpower_fr_gate(spec)
    fr_mu = mean[:, _FLATPOWER_FR_COL]
    fr_sd = std[:, _FLATPOWER_FR_COL]
    fr_ucb = fr_mu + np.where(np.isfinite(fr_sd), kappa * fr_sd, 0.0)
    fr_violated = np.isfinite(fr_ucb) & (fr_ucb > gate)
    constraint_ok = constraint_ok & np.isfinite(fr_ucb) & ~fr_violated

    total = np.where(
        np.isfinite(scalar),
        scalar - _FLATPOWER_CONSTRAINT_TIER * (penalty + fr_violated.astype(float)),
        -np.inf,
    )
    # SOFT engineering-rule shaping, outside the constraint tier.  Absent weights
    # (or absent patterns) leave ``total`` untouched — not "minus zero", untouched.
    rule_pen: np.ndarray | None = None
    if spec.rule_penalty_weights and patterns is not None:
        rule_pen = _rule_penalty(list(patterns), spec.rule_penalty_weights)
        if rule_pen.shape[0] != n:
            raise ValueError(
                f"score_flat_power: {rule_pen.shape[0]} patterns for {n} candidates")
        total = np.where(np.isfinite(total), total - rule_pen, total)
    return FlatPowerScore(
        total=total, scalar=scalar, z_peak=z_peak, z_cov=z_cov,
        peak_ucb=peak_ucb, cov_ucb=cov_ucb, fr_ucb=fr_ucb,
        fr_gate_violated=fr_violated,
        constraint_penalty=penalty, constraint_ok=constraint_ok,
        rule_penalty=rule_pen,
    )


def predict_flatness(model: Any, patterns: Sequence[Pattern], ctx: CaseContext
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(peak_mean, peak_std, cov_mean, cov_std)`` from whatever the backend has.

    Prefers :meth:`PosValCnnBackend.predict_map_flatness` (physical-unit peak AND
    CoV); falls back to ``predict_map_peak`` (peak only, CoV NaN -> the secondary
    term drops); a backend with neither yields all-NaN, so every exploit score is
    ``−inf`` and the caller sees an empty pool rather than a silent F_r ranking.
    """
    n = len(patterns)
    nan = np.full(n, np.nan)
    if hasattr(model, "predict_map_flatness"):
        pk_m, pk_s, cv_m, cv_s = model.predict_map_flatness(
            patterns, ctx.case_key, ctx.e_core or 0.0)
        return (np.asarray(pk_m, dtype=float), np.asarray(pk_s, dtype=float),
                np.asarray(cv_m, dtype=float), np.asarray(cv_s, dtype=float))
    if hasattr(model, "predict_map_peak"):
        pk_m, pk_s = model.predict_map_peak(patterns, ctx.case_key, ctx.e_core or 0.0)
        return (np.asarray(pk_m, dtype=float), np.asarray(pk_s, dtype=float),
                nan, nan.copy())
    return nan, nan.copy(), nan.copy(), nan.copy()


def score_pool_flat_power(
    model: Any,
    ctx: CaseContext,
    candidates: Sequence[Candidate],
    spec: FlatPowerSpec,
    trust_region: "TrustRegion",
    *,
    tie_epsilon: float = 0.0,
) -> ScoredPool:
    """Score a candidate pool for the FLATNESS-NATIVE ``flat_power`` mode.

    Structurally identical to :func:`score_pool_min_fr` (trust-region σ inflation +
    p_feas gate + margin tie-break), but the exploit score is
    :func:`score_flat_power` ``total`` — the weighted ``node_peak`` / ``map_cov``
    minimization from :func:`predict_flatness`.  The p_feas gate and the margin
    tie-break run against :func:`make_flatpower_constraints`, whose ``f_r_limit``
    is the retired 1e12 sentinel, so **F_r contributes to neither** in this mode.
    """

    candidates = list(candidates)
    patterns = [c.pattern for c in candidates]
    n = len(patterns)
    if n == 0:
        empty = np.zeros((0, 7))
        z = np.zeros(0)
        return ScoredPool(
            candidates=[], mean=empty, epistemic=empty.copy(), calibrated=empty.copy(),
            conv=z, p_feas=z.copy(), acq=z.copy(), raw_epi=z.copy(),
            in_region=np.zeros(0, dtype=bool), exploit=z.copy(),
            margin=z.copy(), rank=z.copy(),
        )

    prediction = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    conv = _safe_convergence(model, patterns, ctx)
    pk_mean, pk_std, cv_mean, cv_std = predict_flatness(model, patterns, ctx)

    calibrated = np.asarray(prediction.calibrated_std, dtype=float).copy()
    region = np.ones(n, dtype=bool)
    for i, cand in enumerate(candidates):
        feed = cand.pattern.feed
        ec = cand.e_core if cand.e_core is not None else ctx.e_core
        region[i] = trust_region.in_region(feed, ec)
        scale = trust_region.sigma_scale(feed, ec)
        if scale != 1.0:
            calibrated[i] = calibrated[i] * scale
    inflated = SurrogatePrediction(prediction.mean, prediction.epistemic_std, calibrated)

    constraints = make_flatpower_constraints(spec)
    pf = np.where(region, p_feasible(inflated, constraints, convergence=conv), 0.0)

    fp = score_flat_power(inflated, pk_mean, pk_std, spec, cv_mean, cv_std,
                          patterns=patterns)
    exploit = np.where(region, fp.total, -np.inf)
    margin = np.where(region, feasibility_margin(inflated, constraints), -np.inf)
    rank = rank_with_tiebreak(exploit, margin, tie_epsilon)

    return ScoredPool(
        candidates=candidates,
        mean=np.asarray(prediction.mean, dtype=float),
        epistemic=np.asarray(prediction.epistemic_std, dtype=float),
        calibrated=calibrated,
        conv=np.asarray(conv, dtype=float),
        p_feas=np.asarray(pf, dtype=float),
        acq=np.where(region, pf, 0.0),
        raw_epi=raw_epistemic(inflated),
        in_region=region,
        exploit=np.asarray(exploit, dtype=float),
        margin=np.asarray(margin, dtype=float),
        rank=np.asarray(rank, dtype=float),
    )


# --------------------------------------------------------------------------- #
# reward model
# --------------------------------------------------------------------------- #
def build_reward_model(
    ctx: CaseContext,
    patterns: Sequence[Pattern],
    prediction: SurrogatePrediction,
    constraints: ConstraintConfig,
) -> RewardModel:
    """Construct a case-scaled :class:`RewardModel` from bootstrap predictions.

    The reward scales (reference cycle/F_r/CBC + spreads) are derived from a
    bootstrap pool's predicted FOMs so the target-cycle objective is
    well-conditioned before any MASTER label exists.
    """

    case = ctx.case_key
    records: list[PatternRecord] = []
    for index in range(prediction.mean.shape[0]):
        fom = prediction.mean_fom(index)
        pattern = patterns[index] if index < len(patterns) else patterns[0]
        records.append(
            PatternRecord(
                case=case,
                cell=float(ctx.e_core or 0.0),
                seed_id=f"boot{index}",
                pattern=pattern,
                fom=fom,
                ncyc=1,
                deck_path=Path("."),
                shf_path=Path("."),
            )
        )
    return RewardModel.from_records(records, constraints)


# --------------------------------------------------------------------------- #
# trust region (plan sec. 6.2)
# --------------------------------------------------------------------------- #
def _e_bin(e_core: float | None, band: float) -> int | None:
    if e_core is None or (isinstance(e_core, float) and math.isnan(e_core)):
        return None
    return int(round(float(e_core) / band))


class TrustRegion:
    """Support-grid gate over ``(feed, e_core)`` bins (plan sec. 6.2)."""

    def __init__(
        self,
        cfg: Any,
        supported: set[tuple[int, int | None]],
        *,
        campaign_feed: int,
        campaign_e_core: float | None,
    ) -> None:
        self.cfg = cfg
        self.band = float(cfg.e_core_band)
        self.feed_step = int(cfg.feed_step)
        self.enabled = bool(cfg.enabled)
        self.promote_after = int(cfg.promote_after)
        self.inflation = float(cfg.frontier_sigma_inflation)
        self.supported = set(supported)
        # The campaign's own cell is always in-region (fixed-case, plan sec. 6.2).
        self.supported.add((int(campaign_feed), _e_bin(campaign_e_core, self.band)))
        self._counts: Counter[tuple[int, int | None]] = Counter()

    @property
    def supported_feeds(self) -> set[int]:
        return {feed for feed, _ in self.supported}

    def bin_of(self, feed: int, e_core: float | None) -> tuple[int, int | None]:
        return (int(feed), _e_bin(e_core, self.band))

    def in_region(self, feed: int, e_core: float | None) -> bool:
        if not self.enabled:
            return True
        key = self.bin_of(feed, e_core)
        if key in self.supported:
            return True
        # e_core unknown: admit if the feed itself is supported at any e_core.
        if key[1] is None and key[0] in self.supported_feeds:
            return True
        return False

    def is_frontier(self, feed: int, e_core: float | None) -> bool:
        """One reachable feed step (or one e_core band) from a supported bin."""

        if self.in_region(feed, e_core):
            return False
        ebin = _e_bin(e_core, self.band)
        for sfeed, sebin in self.supported:
            if abs(int(feed) - sfeed) <= self.feed_step and (
                ebin is None or sebin is None or abs(ebin - sebin) <= 1
            ):
                return True
        return False

    def sigma_scale(self, feed: int, e_core: float | None) -> float:
        return self.inflation if self.is_frontier(feed, e_core) else 1.0

    def observe(self, feed: int, e_core: float | None) -> None:
        """Fold a verified label into the region; promote a bin at ``promote_after``."""

        key = self.bin_of(feed, e_core)
        self._counts[key] += 1
        if self._counts[key] >= self.promote_after:
            self.supported.add(key)

    @classmethod
    def from_store(
        cls,
        store_dir: str | Path,
        cfg: Any,
        ctx: CaseContext,
    ) -> "TrustRegion":
        """Seed supported bins from ``records.parquet`` (bins with ≥ n_min labels)."""

        from ..data.store import StoreReader

        band = float(cfg.e_core_band)
        n_min = int(cfg.n_min)
        supported: set[tuple[int, int | None]] = set()
        try:
            df = StoreReader(store_dir).records
        except (FileNotFoundError, OSError):
            df = None
        if df is not None and len(df):
            counts: Counter[tuple[int, int | None]] = Counter()
            for feed, e_core in zip(df["feed"].tolist(), df["e_core"].tolist()):
                counts[(int(feed), _e_bin(e_core, band))] += 1
            supported = {key for key, count in counts.items() if count >= n_min}
        return cls(
            cfg, supported,
            campaign_feed=ctx.feed, campaign_e_core=ctx.e_core,
        )


# --------------------------------------------------------------------------- #
# scored pool
# --------------------------------------------------------------------------- #
@dataclass
class ScoredPool:
    candidates: list[Candidate]
    mean: np.ndarray
    epistemic: np.ndarray
    calibrated: np.ndarray
    conv: np.ndarray
    p_feas: np.ndarray
    acq: np.ndarray
    raw_epi: np.ndarray
    in_region: np.ndarray
    #: Risk-adjusted **exploit** ranking score (plan sec. 4.6): the LCB of the
    #: target-cycle objective (UCB penalty on the constraint axes), with **no**
    #: epistemic bonus — this is the ONLY score exploit slots and local search
    #: rank on.  Before any feasible label exists it is the feasibility-stage
    #: ``−penalty`` (feasibility-first); afterwards the objective-stage total.
    #: Out-of-region candidates are ``−inf`` (they sort last / never adopted).
    exploit: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Risk-adjusted feasibility MARGIN LCB (higher = safer); the exploit
    #: tie-break key (plan sec. 4.6).  Out-of-region rows are ``−inf``.
    margin: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Composite exploit ranking scalar (exploit primary, ``margin`` tie-break
    #: within ``tie_epsilon``): the key EVERY exploit ranking sorts on.  Equal to
    #: ``exploit`` when the tie-break is disabled.  ``exploit`` keeps its pure
    #: value for improvement thresholds and reporting.
    rank: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        # Directly-constructed pools (tests, external callers) may omit the
        # tie-break fields: default ``margin`` to 0 and ``rank`` to the pure
        # exploit score, so every consumer can sort on ``rank`` unconditionally.
        n = len(self.candidates)
        if np.size(self.margin) != n:
            self.margin = np.zeros(n, dtype=float)
        if np.size(self.rank) != n:
            self.rank = np.asarray(self.exploit, dtype=float).copy()

    def __len__(self) -> int:
        return len(self.candidates)


def exploit_score(
    reward_model: RewardModel,
    prediction: SurrogatePrediction,
    cases: Sequence[CaseKey],
    region: np.ndarray,
    *,
    have_feasible: bool,
) -> np.ndarray:
    """Risk-adjusted exploit ranking score (plan sec. 4.6) — no epistemic term.

    Feasibility-first (``have_feasible`` False): the feasibility-stage score
    ``−penalty`` (drive to the feasible region).  After the first verified
    feasible label: the objective-stage total (risk-adjusted target-cycle
    objective − penalty + feasible bonus).  Both use the *calibrated* std, so
    the constraint axes enter at their UCB and the cycle distance at its
    conservative bound — the exploit rule optimizes the pessimistic objective,
    never the exploration-inflated acquisition.  Out-of-region rows are
    ``−inf``.
    """

    stage = "objective" if have_feasible else "feasibility"
    batch = reward_model.score(prediction, cases, stage=stage, calibrated=True)
    total = np.asarray(batch.total, dtype=float)
    return np.where(np.asarray(region, dtype=bool), total, -np.inf)


def score_pool(
    model: Any,
    ctx: CaseContext,
    candidates: Sequence[Candidate],
    reward_model: RewardModel,
    constraints: ConstraintConfig,
    trust_region: TrustRegion,
    *,
    incumbent_distance: float | None = None,
    have_feasible: bool = False,
    tie_epsilon: float = 0.0,
) -> ScoredPool:
    """Predict + score a candidate pool; apply the trust-region hard gate.

    ``have_feasible`` selects the exploit-score stage (feasibility-first before
    the first verified feasible label, objective mode afterwards); it does not
    affect ``acq`` / ``p_feas`` / ``raw_epi``.  ``tie_epsilon`` (> 0) enables the
    feasibility-margin tie-break folded into the ``rank`` ordering scalar.
    """

    patterns = [c.pattern for c in candidates]
    n = len(patterns)
    prediction = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    conv = _safe_convergence(model, patterns, ctx)

    # Frontier σ-inflation: inflate calibrated std on frontier bins (conservative).
    calibrated = np.asarray(prediction.calibrated_std, dtype=float).copy()
    region = np.ones(n, dtype=bool)
    for i, cand in enumerate(candidates):
        feed = cand.pattern.feed
        ec = cand.e_core if cand.e_core is not None else ctx.e_core
        region[i] = trust_region.in_region(feed, ec)
        scale = trust_region.sigma_scale(feed, ec)
        if scale != 1.0:
            calibrated[i] = calibrated[i] * scale
    inflated = SurrogatePrediction(prediction.mean, prediction.epistemic_std, calibrated)

    pf = p_feasible(inflated, constraints, convergence=conv)
    # Hard gate: candidates outside the trust region can never be feasible.
    pf = np.where(region, pf, 0.0)

    cases = [ctx.case_key] * n
    inc = None if incumbent_distance is None else [float(incumbent_distance)] * n
    acq = reward_model.acquisition(
        inflated, cases, convergence_probability=conv, incumbent_cycle_distances=inc,
    )
    acq = np.where(region, acq, 0.0)
    exploit = np.asarray(
        exploit_score(reward_model, inflated, cases, region, have_feasible=have_feasible),
        dtype=float,
    )
    # feasibility-margin tie-break key (uses the σ-inflated calibrated std, like
    # the exploit score); out-of-region rows are -inf so they never win a tie.
    margin = np.where(region, feasibility_margin(inflated, constraints), -np.inf)
    rank = rank_with_tiebreak(exploit, margin, tie_epsilon)

    return ScoredPool(
        candidates=list(candidates),
        mean=np.asarray(prediction.mean, dtype=float),
        epistemic=np.asarray(prediction.epistemic_std, dtype=float),
        calibrated=calibrated,
        conv=np.asarray(conv, dtype=float),
        p_feas=pf,
        acq=np.asarray(acq, dtype=float),
        raw_epi=raw_epistemic(inflated),
        in_region=region,
        exploit=exploit,
        margin=np.asarray(margin, dtype=float),
        rank=np.asarray(rank, dtype=float),
    )


def _safe_convergence(model: Any, patterns: Sequence[Pattern], ctx: CaseContext) -> np.ndarray:
    predict_conv = getattr(model, "predict_convergence", None)
    if predict_conv is None:
        return np.ones(len(patterns), dtype=float)
    try:
        conv = np.asarray(predict_conv(patterns, ctx.case_key, ctx.e_core or 0.0), dtype=float)
    except Exception:  # noqa: BLE001
        return np.ones(len(patterns), dtype=float)
    return np.clip(np.nan_to_num(conv, nan=1.0), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# local search refinement (plan sec. 4.6)
# --------------------------------------------------------------------------- #
def local_search(
    model: Any,
    ctx: CaseContext,
    scored: ScoredPool,
    reward_model: RewardModel,
    constraints: ConstraintConfig,
    trust_region: TrustRegion,
    cfg_ls: Any,
    rng: random.Random,
    ledger_ids: set[str],
    *,
    incumbent_distance: float | None = None,
    have_feasible: bool = False,
    tie_epsilon: float = 0.0,
    score_fn: "Any | None" = None,
) -> ScoredPool:
    """First-improvement hill climb over the top-M candidates by exploit score.

    Each seed generates ≤ ``neighbors`` genome-neighbours (one/two-move
    mutations, feed pinned); the batch is surrogate-scored and the best
    strictly-improving neighbour is adopted, up to ``depth`` steps or the
    ``max_predictions`` budget cap.  The improvement criterion is the
    **exploit score** (the constrained, risk-adjusted objective — plan sec. 4.6),
    NOT the exploration-weighted acquisition: hill-climbing on ``acq`` walks
    candidates toward high-σ OOD regions (the M5-pilot failure mode); climbing
    on the exploit score refines them toward the feasible / on-target basin.
    Improved candidates are appended to the scored pool.

    ``score_fn`` (optional) overrides how a neighbour sub-pool is scored: a callable
    ``neighbours -> ScoredPool``.  When ``None`` (the default, byte-identical to the
    pre-change behaviour) it is the target-cycle :func:`score_pool` built from the
    passed ``reward_model`` / ``constraints`` / ``have_feasible`` / ``incumbent_distance``.
    The ``max_cycle_min_fr`` campaign passes a :func:`score_pool_max_cycle` closure
    so the hill-climb refines on the cyclen/F_r objective, not the target reward.
    """

    from .construct import candidate_record_id

    budget = int(cfg_ls.max_predictions)
    if budget <= 0 or len(scored) == 0:
        return scored

    if score_fn is None:
        def score_fn(neighbours: Sequence[Candidate]) -> ScoredPool:
            return score_pool(
                model, ctx, neighbours, reward_model, constraints, trust_region,
                incumbent_distance=incumbent_distance, have_feasible=have_feasible,
                tie_epsilon=tie_epsilon,
            )
    order = np.argsort(-scored.rank)
    seeds = [int(i) for i in order[: int(cfg_ls.top_m)] if scored.in_region[int(i)]]
    seen: set[str] = set(c.record_id for c in scored.candidates) | set(ledger_ids)
    spent = 0

    new_cands: list[Candidate] = []
    new_rows: list[tuple] = []

    for seed in seeds:
        if spent >= budget:
            break
        current = scored.candidates[seed]
        current_score = float(scored.exploit[seed])
        for _ in range(int(cfg_ls.depth)):
            if spent >= budget:
                break
            neighbours: list[Candidate] = []
            for _ in range(int(cfg_ls.neighbors)):
                try:
                    child = mutate(
                        current.genome, rng, max(1, int(cfg_ls.n_moves)),
                        feed_move_prob=0.0, batches=ctx.batches,
                    )
                    pattern = child.to_pattern()
                except GenomeError:
                    continue
                # Same guard as build_pool._admit: the hill-climb must not walk a
                # graded board back down to a smaller alphabet (no-op unless
                # ctx.require_all_batches).
                if not ctx.uses_full_alphabet(child):
                    continue
                rid = candidate_record_id(pattern, ctx)
                if rid in seen:
                    continue
                seen.add(rid)
                neighbours.append(
                    Candidate(pattern, child, "local", current.record_id, rid, ctx.e_core)
                )
                if spent + len(neighbours) >= budget:
                    break
            if not neighbours:
                break
            sub = score_fn(neighbours)
            spent += len(neighbours)
            # Pick the best neighbour on the composite rank (margin tie-break),
            # but the ADOPTION criterion stays the pure exploit score so a purely
            # margin-driven "improvement" never causes an endless hill-climb.
            best = int(np.argmax(sub.rank))
            if float(sub.exploit[best]) > current_score + 1.0e-9:
                current = neighbours[best]
                current_score = float(sub.exploit[best])
                new_cands.append(current)
                new_rows.append(
                    (
                        sub.mean[best], sub.epistemic[best], sub.calibrated[best],
                        float(sub.conv[best]), float(sub.p_feas[best]),
                        float(sub.acq[best]), float(sub.raw_epi[best]),
                        bool(sub.in_region[best]), float(sub.exploit[best]),
                        float(sub.margin[best]), float(sub.rank[best]),
                    )
                )
            else:
                break

    if not new_cands:
        return scored
    return _extend(scored, new_cands, new_rows)


def _extend(scored: ScoredPool, cands: list[Candidate], rows: list) -> ScoredPool:
    mean = np.vstack([scored.mean, np.stack([r[0] for r in rows])])
    epi = np.vstack([scored.epistemic, np.stack([r[1] for r in rows])])
    cal = np.vstack([scored.calibrated, np.stack([r[2] for r in rows])])
    conv = np.concatenate([scored.conv, np.asarray([r[3] for r in rows])])
    pf = np.concatenate([scored.p_feas, np.asarray([r[4] for r in rows])])
    acq = np.concatenate([scored.acq, np.asarray([r[5] for r in rows])])
    raw = np.concatenate([scored.raw_epi, np.asarray([r[6] for r in rows])])
    region = np.concatenate([scored.in_region, np.asarray([r[7] for r in rows], dtype=bool)])
    exploit = np.concatenate([scored.exploit, np.asarray([r[8] for r in rows])])
    margin = np.concatenate([scored.margin, np.asarray([r[9] for r in rows])])
    rank = np.concatenate([scored.rank, np.asarray([r[10] for r in rows])])
    return ScoredPool(
        candidates=scored.candidates + cands,
        mean=mean, epistemic=epi, calibrated=cal, conv=conv,
        p_feas=pf, acq=acq, raw_epi=raw, in_region=region, exploit=exploit,
        margin=margin, rank=rank,
    )


# --------------------------------------------------------------------------- #
# wave composition (plan sec. 4.6)
# --------------------------------------------------------------------------- #
@dataclass
class WaveSlot:
    index: int
    slot: str        # exploit | explore | control


def tau_schedule(
    scored: ScoredPool, tau0: float, *, have_feasible: bool
) -> float:
    """τ schedule: fixed ``tau0`` until a feasible label exists, then quantile."""

    if not have_feasible:
        return float(tau0)
    gated = scored.p_feas[scored.in_region]
    if gated.size == 0:
        return float(tau0)
    return float(min(0.95, max(tau0, np.quantile(gated, 0.75))))


def compose_wave(
    scored: ScoredPool,
    verified_patterns: Sequence[Pattern],
    rng: random.Random,
    *,
    size: int,
    n_exploit: int,
    n_explore: int,
    n_control: int,
    tau: float,
    hamming_min: int,
    exploit_verified_hamming: int = 0,
) -> list[WaveSlot]:
    """Compose the exploit/explore/control wave with the Hamming diversity filter.

    ``exploit_verified_hamming`` (> 0) adds a HARD floor for exploit slots: an
    exploit pick must stay ≥ that Hamming distance from EVERY verified pattern —
    applied even in the relaxed (thin-pool) fallback, so the surrogate's
    indiscriminable near-repeats of an already-verified board never waste budget.
    """

    size = int(size)
    counts = _slot_counts(size, n_exploit, n_explore, n_control)
    selected: list[WaveSlot] = []
    picked: set[int] = set()
    verified_list: list[Pattern] = list(verified_patterns)
    chosen_patterns: list[Pattern] = list(verified_patterns)

    def _far_enough(idx: int) -> bool:
        pat = scored.candidates[idx].pattern
        return all(pat.hamming(other) >= hamming_min for other in chosen_patterns)

    _verif_cache: dict[int, bool] = {}

    def _clears_verified(idx: int, threshold: int) -> bool:
        if threshold <= 0 or not verified_list:
            return True
        cached = _verif_cache.get(idx)
        if cached is None:
            pat = scored.candidates[idx].pattern
            cached = all(pat.hamming(v) >= threshold for v in verified_list)
            _verif_cache[idx] = cached
        return cached

    region = np.flatnonzero(scored.in_region)
    if region.size == 0:                       # nothing verifiable this wave
        return []

    def _take(
        order: Sequence[int], need: int, slot: str, *, enforce: bool, hard_verified: int = 0
    ) -> None:
        for idx in order:
            if need <= 0:
                break
            idx = int(idx)
            if idx in picked:
                continue
            if hard_verified and not _clears_verified(idx, hard_verified):
                continue                       # hard floor: never relaxed
            if enforce and not _far_enough(idx):
                continue
            selected.append(WaveSlot(idx, slot))
            picked.add(idx)
            chosen_patterns.append(scored.candidates[idx].pattern)
            need -= 1
        if need > 0 and enforce:               # relax Hamming if the pool is thin
            _take(order, need, slot, enforce=False, hard_verified=hard_verified)

    # exploit (plan sec. 4.6): among the candidates passing the τ gate, ranked by
    # the risk-adjusted EXPLOIT rank (constrained objective LCB with the
    # feasibility-margin tie-break, no epistemic bonus); the rest of the region —
    # the feasibility-first fallback when nothing passes τ — ranked by the SAME
    # key, never by the exploration acquisition.  Ranking exploit on ``acq`` was
    # the M5-pilot bug: the raw-σ term made high-σ OOD candidates win every slot.
    by_exploit = region[np.argsort(-scored.rank[region])]
    gate_hi = scored.p_feas[by_exploit] >= tau
    exploit_order = [*by_exploit[gate_hi], *by_exploit[~gate_hi]]
    _take(
        exploit_order, counts["exploit"], "exploit",
        enforce=True, hard_verified=int(exploit_verified_hamming),
    )

    # explore: max raw epistemic σ, preferring p_feas >= tau/2.  This is the ONLY
    # slot that rewards uncertainty.
    by_epi = region[np.argsort(-scored.raw_epi[region])]
    gate_lo = scored.p_feas[by_epi] >= tau / 2.0
    explore_order = [*by_epi[gate_lo], *by_epi[~gate_lo]]
    _take(explore_order, counts["explore"], "explore", enforce=True)

    # control: uniform random from the in-region (ungated) pool.
    control_pool = [int(i) for i in region if int(i) not in picked]
    rng.shuffle(control_pool)
    _take(control_pool, counts["control"], "control", enforce=True)

    # backfill any shortfall from the best remaining region candidates by exploit
    # rank (the hard verified floor still applies to these exploit slots).
    shortfall = size - len(selected)
    if shortfall > 0:
        _take(
            [int(i) for i in by_exploit], shortfall, "exploit",
            enforce=False, hard_verified=int(exploit_verified_hamming),
        )

    return selected[:size]


def _slot_counts(size: int, n_exploit: int, n_explore: int, n_control: int) -> dict[str, int]:
    total = n_exploit + n_explore + n_control
    if total == size:
        return {"exploit": n_exploit, "explore": n_explore, "control": n_control}
    # Scale to a short (reserve) wave, keeping exploit dominant.
    if size <= 0:
        return {"exploit": 0, "explore": 0, "control": 0}
    control = 1 if size >= 4 and n_control > 0 else 0
    explore = 1 if size >= 3 and n_explore > 0 else 0
    exploit = max(0, size - control - explore)
    return {"exploit": exploit, "explore": explore, "control": control}


# --------------------------------------------------------------------------- #
# user-criteria scoring (plan sec. 12.5 — Phase E; wiring deferred to the CLI
# owner, this module only exposes the spec + scoring function for later use)
# --------------------------------------------------------------------------- #
#: A cyclen/discharge band violation must dominate every within-band term.
_CRITERIA_BAND_TIER = 1.0e8
#: A design-limit (gated constraint) violation must dominate the F_r objective.
_CRITERIA_CONSTRAINT_TIER = 1.0e4
#: Real-time GATED constraint axes: (surrogate column, CriteriaSpec limit field,
#: penalty width).  ``asi_abs_limit`` maps to the AO_abs column (4): ASI ≡ −AO by
#: sign convention, so |ASI| == |AO| numerically and the model's ao_abs axis is
#: exactly what an |ASI| limit constrains.  ``pin_bu_limit`` maps to column 6
#: (max_pin_burnup).  A limit left at ``None`` is report-only (never gated).
_CRITERIA_GATED_AXES: tuple[tuple[int, str, float], ...] = (
    (0, "f_r_limit", 0.01),
    (1, "cbc_limit", 25.0),
    (2, "f_q_limit", 0.05),
    (4, "asi_abs_limit", 0.02),
    (6, "pin_bu_limit", 1.0),
)


@dataclass(frozen=True)
class CriteriaSpec:
    """User-specified search criteria for the plan sec. 12.5 ``user_criteria`` mode.

    Two TARGET axes (band criteria, both model-predicted):

    * ``target_cyclen``            — desired cycle length [EFPD]
    * ``target_discharge_burnup``  — desired average discharge burnup [MWd/kgU]

    with tolerance windows ``cyclen_tolerance`` / ``discharge_tolerance``.  A
    candidate is *on target* when the (risk-adjusted) deviation of BOTH lies
    within its window.  Neither is a minimization axis — inside its band it is
    neutral, outside it is penalized (dominant, hierarchical).

    Primary OBJECTIVE (unchanged project convention): among on-target,
    criteria-satisfying candidates, **minimize F_r** — ranked on the F_r upper
    confidence bound ``μ_Fr + risk_z·σ_Fr`` so uncertain candidates never look
    better than they are.  F_r therefore plays a DUAL role: an optional
    ``f_r_limit`` GATES feasibility (like MOCHA's fr limit) while F_r remains the
    within-region minimization objective — the gate and the objective compose
    without double-counting (the limit only zeroes/penalizes; the ranking uses
    the UCB).

    Real-time GATED constraint axes (model-predicted; ``None`` == report-only, no
    gating).  Every SET limit is enforced as an upper bound at its UCB by
    :func:`score_user_criteria`:

    * ``f_r_limit``     — pin peaking F_r            (col 0; also the objective)
    * ``cbc_limit``     — max critical boron          (col 1)
    * ``f_q_limit``     — heat-flux peaking F_q       (col 2)
    * ``asi_abs_limit`` — max |ASI| axial shape index (col 4; |ASI| == |AO|,
      constrains the model's ao_abs axis; user-facing name/sign use ASI)
    * ``pin_bu_limit``  — max pin burnup              (col 6; default report-only,
      plan sec. 12.5 "한계 설정 가능, 기본 보고만")

    Post-verification constraint axes — **NOT model-predicted, NEVER scored here**:

    * ``mtc_limit`` — moderator temperature coefficient
    * ``sdm_limit`` — shutdown margin

    These are enforced by a separate post-verification stage
    (%EXE_RHO / %EXE_ROD branch calculations on the top-K converged candidates —
    a later ``lpopt/search/sdm_mtc.py`` port of 2_LP MOCHA's sdm_mtc harness).
    :func:`score_user_criteria` explicitly EXCLUDES them; the surrogate has no
    physical basis to predict MTC/SDM.  They are carried here only so the spec is
    the single source of truth for a campaign's user limits.

    ``e_core`` is not a field: the target core-average enrichment is realized by
    the pair/split candidate filter upstream, not by this scoring function.
    """

    target_cyclen: float
    target_discharge_burnup: float | None = None
    cyclen_tolerance: float = 2.0
    discharge_tolerance: float = 0.5
    # real-time gated axes (None == report-only)
    f_r_limit: float | None = 1.55
    cbc_limit: float | None = 1550.0
    f_q_limit: float | None = 2.41
    asi_abs_limit: float | None = 0.30
    pin_bu_limit: float | None = None
    # post-verification axes — carried, never scored here
    mtc_limit: float | None = None
    sdm_limit: float | None = None
    # risk aversion: UCB shift on the F_r objective, the target distances, and
    # every gated constraint axis (κ in μ + κσ).
    risk_z: float = 0.25
    # band-distance normalizers (only affect the graded part of an already
    # dominant out-of-band penalty; ranking within-band is pure F_r).
    cyclen_scale: float = 10.0
    discharge_scale: float = 1.0

    def gated_axes(self) -> tuple[str, ...]:
        """Names of the constraint axes that are SET (and therefore gate)."""
        return tuple(attr for _, attr, _ in _CRITERIA_GATED_AXES
                     if getattr(self, attr) is not None)


@dataclass
class UserCriteriaScore:
    """Result of :func:`score_user_criteria` (all arrays length N).

    ``total`` is the single ranking scalar (higher is better): the F_r objective
    dominated, hierarchically, first by any cyclen/discharge band violation and
    then by any gated design-limit violation.  The component arrays are exposed
    so a future campaign wiring can rank lexicographically if it prefers.
    """

    total: np.ndarray
    objective: np.ndarray          # −(μ_Fr + κσ_Fr): the minimize-F_r objective
    fr_ucb: np.ndarray             # μ_Fr + κσ_Fr
    band_penalty: np.ndarray       # 0 inside both bands, > 0 outside (dominant)
    constraint_penalty: np.ndarray # Σ squared gated-axis excess
    feasible: np.ndarray           # on_target ∧ constraints ok ∧ (converged)
    on_target: np.ndarray          # both target deviations within tolerance
    cyclen_distance: np.ndarray    # risk-adjusted |μ_cy − target| beyond tol
    discharge_distance: np.ndarray # risk-adjusted |μ_dis − target| beyond tol (NaN if absent)
    gated_axes: tuple[str, ...]


def score_user_criteria(
    prediction: SurrogatePrediction,
    spec: CriteriaSpec,
    *,
    discharge_mean: Sequence[float] | np.ndarray | None = None,
    discharge_std: Sequence[float] | np.ndarray | None = None,
    convergence: Sequence[float] | np.ndarray | None = None,
    cyclen_bias: float | Sequence[float] | np.ndarray | None = None,
) -> UserCriteriaScore:
    """Score candidates for the user-criteria mode (plan sec. 12.5, both addenda).

    Hierarchy (dominant first): (1) cyclen tolerance band, (2) discharge_burnup
    tolerance band — both criteria, penalized only OUTSIDE the window; (3) inside
    both bands, minimize F_r on its UCB.  Gated design limits (``f_r_limit``,
    ``cbc_limit``, ``f_q_limit``, ``asi_abs_limit``, ``pin_bu_limit``) enter as
    UCB-shifted upper-bound penalties for whichever are SET; MTC/SDM are never
    read here.  ``discharge_*`` come from ``PosValCnnBackend.predict_extra`` (the
    discharge axis is NOT in the 7-column surrogate); when absent the discharge
    criterion is treated as satisfied (neutral).

    ``cyclen_bias`` (Stage-2 running corrector, ``cell_calibrate``) is subtracted
    from the cyclen mean *before* the band test only — a scalar or per-candidate
    array of estimated over-prediction [EFPD].  It affects the cyclen band term
    (item 1) exclusively; every other axis and the reported ``prediction`` are
    untouched.  ``None`` (the default) is byte-identical to the pre-corrector
    behaviour, so a cell with no accumulated bias screens exactly as before.
    """

    mean = np.asarray(prediction.mean, dtype=float)
    std = np.asarray(prediction.calibrated_std, dtype=float)
    n = mean.shape[0]
    kappa = float(spec.risk_z)

    def _ucb_dev(mu: np.ndarray, sd: np.ndarray, target: float, tol: float) -> np.ndarray:
        shift = np.where(np.isfinite(sd), kappa * sd, 0.0)
        return np.maximum(0.0, np.abs(mu - target) + shift - tol)

    # -- (1) cyclen band (col 3), optionally bias-corrected ---------------- #
    cy_mu = mean[:, 3]
    if cyclen_bias is not None:
        cy_mu = cy_mu - np.asarray(cyclen_bias, dtype=float)
    d_cy = _ucb_dev(cy_mu, std[:, 3], float(spec.target_cyclen),
                    float(spec.cyclen_tolerance))

    # -- (2) discharge band (predict_extra; neutral when absent) ----------- #
    if discharge_mean is not None and spec.target_discharge_burnup is not None:
        dmu = np.asarray(discharge_mean, dtype=float)
        dsd = (np.zeros(n) if discharge_std is None
               else np.asarray(discharge_std, dtype=float))
        d_dis = _ucb_dev(dmu, dsd, float(spec.target_discharge_burnup),
                         float(spec.discharge_tolerance))
    else:
        d_dis = np.zeros(n, dtype=float)

    on_target = (d_cy <= 0.0) & (d_dis <= 0.0)
    band_penalty = np.where(
        on_target, 0.0,
        1.0 + d_cy / float(spec.cyclen_scale) + d_dis / float(spec.discharge_scale),
    )

    # -- (3) gated design-limit constraints (UCB upper bounds) ------------- #
    constraint_penalty = np.zeros(n, dtype=float)
    known_ok = np.ones(n, dtype=bool)
    for column, attr, width in _CRITERIA_GATED_AXES:
        limit = getattr(spec, attr)
        if limit is None:
            continue                       # report-only axis: never gates
        mu_c = mean[:, column]
        sd_c = std[:, column]
        known = np.isfinite(sd_c)
        excess = np.where(
            known,
            np.maximum(0.0, (mu_c + kappa * sd_c - float(limit))) / float(width),
            0.0,                            # unknown axis: no penalty…
        )
        constraint_penalty = constraint_penalty + excess ** 2
        known_ok = known_ok & known         # …but it cannot certify feasibility

    # -- objective: minimize F_r on its UCB (col 0) ------------------------ #
    fr_ucb = mean[:, 0] + np.where(np.isfinite(std[:, 0]), kappa * std[:, 0], 0.0)
    objective = -fr_ucb

    constraint_feasible = (constraint_penalty <= 1.0e-12) & known_ok
    feasible = on_target & constraint_feasible
    if convergence is not None:
        feasible = feasible & (np.asarray(convergence, dtype=float) >= 0.5)

    total = (
        objective
        - _CRITERIA_BAND_TIER * band_penalty
        - _CRITERIA_CONSTRAINT_TIER * constraint_penalty
    )
    return UserCriteriaScore(
        total=total,
        objective=objective,
        fr_ucb=fr_ucb,
        band_penalty=band_penalty,
        constraint_penalty=constraint_penalty,
        feasible=feasible,
        on_target=on_target,
        cyclen_distance=d_cy,
        discharge_distance=(d_dis if (discharge_mean is not None
                                      and spec.target_discharge_burnup is not None)
                            else np.full(n, np.nan)),
        gated_axes=spec.gated_axes(),
    )


# --------------------------------------------------------------------------- #
# user_criteria mode: pool scoring (exploit ranking = score_user_criteria) and
# the outer racing allocation over the pair universe (plan sec. 6.2 / 12.5).
# --------------------------------------------------------------------------- #
def make_criteria_constraints(spec: CriteriaSpec) -> ConstraintConfig:
    """Vendor :class:`ConstraintConfig` from the CriteriaSpec SET limits.

    The ``p_feasible`` gate reads F_r / CBC / F_q / |ASI|(==AO) at their limits;
    an axis left ``None`` in the spec (report-only) is mapped to a large limit so
    it never gates.  Ranking is done by :func:`score_user_criteria`, so the reward
    objective mode is irrelevant here (``trade_off``, unused).
    """

    big = 1.0e12
    return ConstraintConfig(
        f_r_limit=float(spec.f_r_limit) if spec.f_r_limit is not None else big,
        cbc_limit=float(spec.cbc_limit) if spec.cbc_limit is not None else big,
        f_q_limit=float(spec.f_q_limit) if spec.f_q_limit is not None else big,
        ao_abs_limit=float(spec.asi_abs_limit) if spec.asi_abs_limit is not None else big,
        risk_z=float(spec.risk_z),
        objective_mode="trade_off",
    )


def _extra_discharge(
    model: Any, patterns: Sequence[Pattern], ctx: CaseContext, spec: CriteriaSpec
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(mean, calibrated_std)`` of the ``discharge_burnup`` extra target, or
    ``(None, None)`` when the spec sets no discharge target or the backend has no
    ``predict_extra`` / discharge column (score treats absent discharge as neutral)."""

    if spec.target_discharge_burnup is None or not hasattr(model, "predict_extra"):
        return None, None
    try:
        extra = model.predict_extra(patterns, ctx.case_key, ctx.e_core or 0.0)
        names = list(extra.names)
        if "discharge_burnup" not in names:
            return None, None
        k = names.index("discharge_burnup")
        mean = np.asarray(extra.mean, dtype=float)[:, k]
        std = np.asarray(extra.calibrated_std, dtype=float)[:, k]
        if not np.any(np.isfinite(mean)):
            return None, None
        return mean, std
    except Exception:  # noqa: BLE001 — discharge is optional; never abort scoring
        return None, None


def score_pool_user_criteria(
    model: Any,
    ctx: CaseContext,
    candidates: Sequence[Candidate],
    spec: CriteriaSpec,
    *,
    fuel: Any = None,
    library_id: str | None = None,
    e_core_target: float | None = None,
    e_core_tol: float = 0.05,
    cyclen_bias: float | Sequence[float] | np.ndarray | None = None,
) -> ScoredPool:
    """Score a candidate pool for the ``user_criteria`` mode (plan sec. 12.5).

    Differences from :func:`score_pool` (target_cycle path):

    * ``in_region`` is the e_core-band screen — a candidate whose fresh-composition
      ``e_core`` (shared recipe) is out of ``e_core_target +/- e_core_tol`` is
      excluded from every slot (split-as-inner-variable screen, item 2);
    * the exploit ``rank`` is :func:`score_user_criteria` ``total`` (hierarchical
      cyclen band -> discharge band -> min-F_r UCB), NOT the target_cycle reward;
    * ``p_feas`` gates on the CriteriaSpec SET constraint limits (item 3);
    * ``raw_epi`` (explore) and control are unchanged.

    Consumed unchanged by :func:`compose_wave` (exploit on ``rank``, explore on
    ``raw_epi``, τ gate on ``p_feas``, region on ``in_region``).
    """

    candidates = list(candidates)
    patterns = [c.pattern for c in candidates]
    n = len(patterns)
    lib = library_id or ctx.library_id
    if n == 0:
        empty = np.zeros((0, 7))
        z = np.zeros(0)
        return ScoredPool(
            candidates=[], mean=empty, epistemic=empty.copy(), calibrated=empty.copy(),
            conv=z, p_feas=z.copy(), acq=z.copy(), raw_epi=z.copy(),
            in_region=np.zeros(0, dtype=bool), exploit=z.copy(),
            margin=z.copy(), rank=z.copy(),
        )

    prediction = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    conv = _safe_convergence(model, patterns, ctx)
    d_mean, d_std = _extra_discharge(model, patterns, ctx, spec)

    if fuel is not None and e_core_target is not None:
        region = screen_e_core_band(patterns, fuel, lib, float(e_core_target), float(e_core_tol))
    else:
        region = np.ones(n, dtype=bool)
    region = np.asarray(region, dtype=bool)

    constraints = make_criteria_constraints(spec)
    pf = np.where(region, p_feasible(prediction, constraints, convergence=conv), 0.0)

    uc = score_user_criteria(
        prediction, spec, discharge_mean=d_mean, discharge_std=d_std,
        convergence=conv, cyclen_bias=cyclen_bias,
    )
    exploit = np.where(region, uc.total, -np.inf)
    margin = np.where(region, feasibility_margin(prediction, constraints), -np.inf)

    return ScoredPool(
        candidates=candidates,
        mean=np.asarray(prediction.mean, dtype=float),
        epistemic=np.asarray(prediction.epistemic_std, dtype=float),
        calibrated=np.asarray(prediction.calibrated_std, dtype=float),
        conv=np.asarray(conv, dtype=float),
        p_feas=np.asarray(pf, dtype=float),
        acq=np.where(region, pf, 0.0),
        raw_epi=raw_epistemic(prediction),
        in_region=region,
        exploit=np.asarray(exploit, dtype=float),
        margin=np.asarray(margin, dtype=float),
        rank=np.asarray(exploit, dtype=float),
    )


# ---- outer racing allocation over the pair universe ----------------------- #
@dataclass
class OuterCellStat:
    """Per-cell (pair) racing state for the outer allocation (plan sec. 6.2).

    ``screen_value`` is the wave-0 surrogate-only virtual-screen quality (best
    :func:`score_user_criteria` total over the cell's screen pool); ``samples`` are
    the verified candidates' criteria totals accumulated during racing.  Both are
    on the same (criteria-total) scale, so the screen prior and verified labels
    combine into one mean/spread used for UCB/LCB racing.  Higher value = better
    cell (F_r minimization is encoded as −F_r_UCB inside the criteria total).
    """

    cell_id: str
    screen_value: float
    samples: list[float] = field(default_factory=list)
    active: bool = False
    eliminated: bool = False
    n_verify: int = 0
    best_fr_ucb: float = float("inf")
    best_feasible: bool = False

    def _mean_spread(self, prior_sigma: float) -> tuple[float, float, int]:
        vals = [self.screen_value, *self.samples]
        vals = [v for v in vals if math.isfinite(v)]
        n_eff = 1 + len(self.samples)
        if not vals:
            return float("-inf"), float(prior_sigma), n_eff
        mean = float(np.mean(vals))
        spread = float(np.std(vals)) if len(vals) >= 2 else float(prior_sigma)
        spread = max(spread, 1.0e-9)
        return mean, spread, n_eff

    def value(self) -> float:
        """Representative cell quality (mean of screen prior + verified samples)."""
        mean, _, _ = self._mean_spread(0.0)
        return mean

    def ucb(self, z: float, prior_sigma: float) -> float:
        mean, spread, n_eff = self._mean_spread(prior_sigma)
        return mean + float(z) * spread / math.sqrt(n_eff)

    def lcb(self, z: float, prior_sigma: float) -> float:
        mean, spread, n_eff = self._mean_spread(prior_sigma)
        return mean - float(z) * spread / math.sqrt(n_eff)


def outer_activate(cells: Sequence[OuterCellStat], outer_max_cells: int) -> list[str]:
    """Activate the top ``outer_max_cells`` cells by wave-0 screen value.

    Sets ``active`` on the winners (clears it on the rest) and returns the
    activated cell ids (highest screen value first)."""

    order = sorted(cells, key=lambda c: (-c.screen_value, c.cell_id))
    activated: list[str] = []
    for k, cell in enumerate(order):
        cell.active = k < int(outer_max_cells)
        if cell.active:
            activated.append(cell.cell_id)
    return activated


def outer_race(
    cells: Sequence[OuterCellStat], *, z: float, prior_sigma: float, min_keep: int
) -> list[str]:
    """Eliminate active cells whose UCB < max active LCB (racing, plan sec. 6.2).

    A cell can be the best only if its optimistic bound (UCB) reaches the best
    pessimistic bound (max LCB) among survivors; those that cannot are eliminated
    worst-first, never dropping below ``min_keep`` survivors.  Returns the ids
    eliminated this call.
    """

    active = [c for c in cells if c.active and not c.eliminated]
    if len(active) <= int(min_keep):
        return []
    threshold = max(c.lcb(z, prior_sigma) for c in active)
    losers = sorted(
        (c for c in active if c.ucb(z, prior_sigma) < threshold - 1.0e-12),
        key=lambda c: c.ucb(z, prior_sigma),
    )
    eliminated: list[str] = []
    survivors = len(active)
    for cell in losers:
        if survivors <= int(min_keep):
            break
        cell.eliminated = True
        cell.active = False
        eliminated.append(cell.cell_id)
        survivors -= 1
    return eliminated


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer apportionment of ``total`` across ``weights`` (largest remainder)."""

    w = np.asarray(weights, dtype=float)
    if total <= 0 or w.sum() <= 0:
        return np.zeros(w.shape[0], dtype=int)
    raw = w / w.sum() * float(total)
    base = np.floor(raw).astype(int)
    rem = int(total - base.sum())
    if rem > 0:
        for i in np.argsort(-(raw - base))[:rem]:
            base[int(i)] += 1
    return base


def outer_softmax_alloc(
    cells: Sequence[OuterCellStat], slots: int, *,
    temperature: float, exploit_floor: int,
) -> dict[str, int]:
    """Allocate ``slots`` verify slots across surviving cells (plan sec. 6.2).

    The best surviving cell (highest value) gets a hard ``exploit_floor`` first;
    the remainder is apportioned by a softmax over cell values at ``temperature``
    (largest-remainder rounding).  Only ``active`` non-eliminated cells receive
    slots.
    """

    survivors = [c for c in cells if c.active and not c.eliminated]
    if not survivors or int(slots) <= 0:
        return {}
    survivors_sorted = sorted(survivors, key=lambda c: (-c.value(), c.cell_id))
    alloc: dict[str, int] = {c.cell_id: 0 for c in survivors}

    slots = int(slots)
    best = survivors_sorted[0]
    floor = min(int(exploit_floor), slots)
    alloc[best.cell_id] += floor
    remaining = slots - floor
    if remaining > 0:
        vals = np.array([c.value() for c in survivors_sorted], dtype=float)
        finite = vals[np.isfinite(vals)]
        base = finite.max() if finite.size else 0.0
        vals = np.where(np.isfinite(vals), vals, base - 1.0e6)
        temp = max(float(temperature), 1.0e-6)
        shifted = (vals - vals.max()) / temp
        weights = np.exp(shifted)
        counts = _largest_remainder(weights, remaining)
        for cell, extra in zip(survivors_sorted, counts):
            alloc[cell.cell_id] += int(extra)
    return alloc


__all__ = [
    "TrustRegion",
    "ScoredPool",
    "WaveSlot",
    "CriteriaSpec",
    "UserCriteriaScore",
    "MaxCycleSpec",
    "MaxCycleScore",
    "MinFrSpec",
    "MinFrScore",
    "MinFuelCostSpec",
    "MinFuelCostScore",
    "MinFrBoundarySpec",
    "MinFrBoundaryScore",
    "FlatPowerSpec",
    "FlatPowerScore",
    "FLATPOWER_GATE_K",
    "flatpower_fr_gate",
    "predict_flatness",
    "OuterCellStat",
    "build_reward_model",
    "make_maxcycle_constraints",
    "make_minfr_constraints",
    "make_minfuelcost_constraints",
    "make_fr_boundary_constraints",
    "make_flatpower_constraints",
    "score_max_cycle_min_fr",
    "score_min_fr_max_cycle",
    "score_min_fuel_cost",
    "score_fr_boundary",
    "score_flat_power",
    "fuel_charge_array",
    "score_pool_max_cycle",
    "score_pool_min_fr",
    "score_pool_min_fuel_cost",
    "score_pool_fr_boundary",
    "score_pool_flat_power",
    "compose_wave",
    "exploit_score",
    "feasibility_margin",
    "local_search",
    "make_constraints",
    "make_criteria_constraints",
    "outer_activate",
    "outer_race",
    "outer_softmax_alloc",
    "p_feasible",
    "rank_with_tiebreak",
    "raw_epistemic",
    "score_pool",
    "score_pool_user_criteria",
    "score_user_criteria",
    "tau_schedule",
]
