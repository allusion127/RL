"""Paired, cell-clustered bootstrap inference — flatness-first program section 8.3.

Why this module exists
----------------------
Every arm in the flatness A/B is evaluated on the **same rows** of the same C2
slice.  A comparison of two statistics computed on identical rows is a *paired*
comparison, and the paired difference has far smaller variance than either arm's
marginal statistic: the cell-to-cell variation that dominates a single-arm
interval cancels exactly.  :func:`~.ab_eval.cluster_bootstrap_ci` returns a CI on
one arm's median and nothing else, so reading two of them side by side is not an
inference about the difference — overlapping single-arm CIs are routinely
compatible with a decisive paired difference, and disjoint ones can be produced
by a shared cell effect that is not a difference at all.

So the estimand here is the **difference**, never the pair of levels:

    theta = aggregate over cells of  ( stat_arm(cell) - stat_control(cell) )

sign-flipped where lower is better, so ``theta > 0`` always means "arm beats
control".  The resampling unit is the **cell**, not the row: rows inside one
design cell are not independent draws for a within-cell statistic (they share a
design, a library and a generator), so a row bootstrap understates the interval.

Median or mean, chosen per metric
---------------------------------
``aggregate`` defaults to the median, matching the rest of the harness and
resisting the one pathological cell.  It is **wrong for regret**, and the metric
registry says so explicitly: top-8 regret is a non-negative, zero-inflated loss
-- a competent model catches the flattest candidate in most cells, so the median
per-cell regret is exactly 0 for both arms and the median paired difference is
identically 0 no matter how much regret the arm removes from the cells where it
is not 0.  Measured on a 12-cell / 60-row synthetic at within-cell rho ~ 0.6,
regret is zero in 6 of 12 cells.  The expected loss per cell -- the mean -- is
what the search actually pays, so M0 aggregates with the mean and every ranking
statistic keeps the median.

BCa, not percentile
-------------------
Section 8.3 asks for BCa because the C2 slice has ~29 clusters and the
statistic is a median of a skewed per-cell difference.  At that cluster count the
percentile interval is anti-conservative: it inherits the bootstrap
distribution's median bias and ignores the dependence of the standard error on
the parameter.  BCa corrects both — ``z0`` from the fraction of replicates below
the point estimate, ``a`` from the jackknife over cells.

When BCa cannot be formed honestly (too few clusters for a jackknife, a
degenerate bootstrap distribution, a non-finite acceleration) the interval falls
back to percentile **and says so** in ``method`` and ``notes``.  It never
silently pretends to be BCa.

Reading the interval
--------------------
``PairedDiff`` deliberately exposes two asymmetric predicates, matching section
8.3's rule that a gain and a harm are judged with different tails:

* **gain** — :meth:`PairedDiff.establishes_gain`, ``ci_lo > 0``.  A point
  estimate is never enough.
* **harm** — :meth:`PairedDiff.bounds_harm`, ``-ci_lo < margin``.  The harm's
  upper bound is the negated *lower* bound of the gain, so one interval answers
  both questions and they can never disagree.

An interval that straddles the null answers neither: it is not evidence of
equivalence, and :meth:`straddles_null` exists so callers can route that case to
a human instead of resolving it.

A degenerate resample is not a decisive win
-------------------------------------------
When every paired cell moved by exactly the same amount the resampling
distribution is a point mass and the interval collapses to ``ci_lo == ci_hi ==
theta_hat`` with ``se = 0``.  Read as an interval that is a CI of zero width
excluding the null — i.e. the most decisive result the apparatus can produce —
which is the opposite of what it is: the bootstrap saw one distinct value and
could not express any uncertainty at all (n=3 identical cells and n=300 identical
cells produce the same "interval").  So ``degenerate`` joins ``insufficient`` in
:data:`NO_EVIDENCE_METHODS`: it is reported in full, and it fails every gain test
AND every harm test, so nothing downstream can promote on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Two-sided level for every pre-registered interval.
DEFAULT_ALPHA = 0.05
#: Bootstrap resamples.  2000 is the smallest count at which the 2.5%/97.5%
#: percentiles of a BCa-adjusted interval are stable to ~0.001 for our n.
DEFAULT_REPS = 2000
#: Fewer clusters than this and the jackknife acceleration is meaningless, so
#: BCa degrades to percentile (announced in ``notes``).
MIN_CELLS_BCA = 6
#: Fewer clusters than this and NO interval is reported: the comparison is
#: returned as ``method="insufficient"`` with an infinite interval, which fails
#: every gain test AND every harm test.  Refusing to judge is the safe default.
MIN_CELLS = 3
#: Power used for the pre-disclosed minimum detectable effect (section 8.3).
MDE_POWER = 0.80
#: ``method`` values that carry NO usable interval evidence.  ``insufficient`` has
#: too few clusters to form one; ``degenerate`` formed one of zero width out of a
#: point-mass resample, which is not the same thing as having measured a decisive
#: effect.  Both fail every gain AND every harm test — refusing to judge is the
#: safe default, and it must be the SAME refusal in both cases.
NO_EVIDENCE_METHODS = frozenset({"insufficient", "degenerate"})


# --------------------------------------------------------------------------- #
# normal quantiles without scipy
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    """Standard normal CDF (exact to double precision via ``erfc``)."""
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


# Acklam's rational approximation coefficients.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Standard normal quantile.

    Acklam's rational approximation plus one Halley step against :func:`norm_cdf`,
    which takes the relative error below 1e-15 -- well past what a 2000-replicate
    bootstrap percentile can resolve.  Written out rather than pulled from scipy
    because this module is imported by the decision path and must not acquire a
    heavyweight optional dependency.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        return float("-inf") if p <= 0.0 else float("inf")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


# --------------------------------------------------------------------------- #
# the result object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairedDiff:
    """A CI on ``arm - control``, in GAIN orientation (positive == arm better).

    ``point``/``ci_lo``/``ci_hi`` are already sign-flipped for metrics where a
    lower raw value is better (regret, MAE, |bias|), so every consumer reads one
    orientation and no call site has to remember a metric's direction.
    """

    metric: str
    arm: str
    control: str
    higher_is_better: bool
    point: float
    ci_lo: float
    ci_hi: float
    se: float
    method: str                 # bca | percentile | degenerate | insufficient
    aggregate: str              # median | mean (see the module docstring)
    reps: int
    seed: int
    alpha: float
    n_cells: int
    n_dropped: int
    arm_level: float
    control_level: float
    notes: tuple[str, ...] = ()

    # -- decision predicates ------------------------------------------------- #
    @property
    def measured(self) -> bool:
        """Did this comparison produce usable interval evidence at all?

        ``False`` for :data:`NO_EVIDENCE_METHODS`.  Every predicate below routes
        through it, so a degenerate point-mass resample cannot be read as the
        zero-width CI of a decisive effect (see the module docstring).
        """
        return self.method not in NO_EVIDENCE_METHODS

    @property
    def degenerate(self) -> bool:
        """The resampling distribution was a point mass (zero-width interval)."""
        return self.method == "degenerate"

    def establishes_gain(self) -> bool:
        """The gain CI excludes the null (section 8.3: point estimates cannot)."""
        return bool(self.measured and np.isfinite(self.ci_lo) and self.ci_lo > 0.0)

    @property
    def harm_upper(self) -> float:
        """Upper confidence bound on the HARM (= negated gain lower bound)."""
        return float(-self.ci_lo) if np.isfinite(self.ci_lo) else float("inf")

    def bounds_harm(self, margin: float) -> bool:
        """Non-inferiority: the harm's upper bound sits below ``margin``."""
        return bool(self.measured and np.isfinite(self.ci_lo)
                    and self.harm_upper < float(margin))

    def straddles_null(self) -> bool:
        """Neither a gain nor a bounded harm -- the case that must be escalated."""
        if not (self.measured and np.isfinite(self.ci_lo) and np.isfinite(self.ci_hi)):
            return True
        return bool(self.ci_lo <= 0.0 <= self.ci_hi)

    def favours_arm_on_points_only(self) -> bool:
        """Point estimate says 'arm', the interval does not.  The trap this
        apparatus exists to catch."""
        return bool(np.isfinite(self.point) and self.point > 0.0
                    and not self.establishes_gain())

    def mde(self, power: float = MDE_POWER) -> float:
        """Minimum detectable effect at ``power`` for this paired SE (section 8.3).

        Pre-disclosing the MDE is what stops "no significant difference" from
        being read as "no difference" when the design could never have seen one.
        """
        if not np.isfinite(self.se) or self.se <= 0:
            return float("nan")
        return float((norm_ppf(1.0 - self.alpha / 2.0) + norm_ppf(power)) * self.se)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "metric": self.metric, "arm": self.arm, "control": self.control,
            "higher_is_better": self.higher_is_better,
            "point": _j(self.point), "ci_lo": _j(self.ci_lo), "ci_hi": _j(self.ci_hi),
            "se": _j(self.se), "method": self.method,
            # carried explicitly so a consumer reading the JSON (rather than the
            # object) cannot mistake a zero-width point-mass interval for a
            # decisive one -- ``method`` alone was routinely not looked at.
            "measured": self.measured, "degenerate": self.degenerate,
            "aggregate": self.aggregate, "reps": self.reps,
            "seed": self.seed, "alpha": self.alpha, "n_cells": self.n_cells,
            "n_dropped": self.n_dropped,
            "arm_level": _j(self.arm_level), "control_level": _j(self.control_level),
            "establishes_gain": self.establishes_gain(),
            "harm_upper": _j(self.harm_upper),
            "straddles_null": self.straddles_null(),
            "mde80": _j(self.mde()),
            "notes": list(self.notes),
        }
        return d


def _j(v: float) -> float | None:
    """JSON-safe float (``inf``/``nan`` become ``None``)."""
    f = float(v)
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- #
# the bootstrap
# --------------------------------------------------------------------------- #
#: Cell-level aggregates the paired statistic may use.
AGGREGATES = {"median": np.median, "mean": np.mean}


def _agg(x: np.ndarray, how: str = "median") -> float:
    fn = AGGREGATES.get(how)
    if fn is None:
        raise ValueError(f"unknown aggregate {how!r}; use one of {sorted(AGGREGATES)}")
    return float(fn(x)) if len(x) else float("nan")


def _bca_bounds(theta_hat: float, boot: np.ndarray, jack: np.ndarray,
                alpha: float) -> tuple[float, float, str, list[str]]:
    """BCa percentile positions, or a percentile fallback with the reason."""
    notes: list[str] = []
    reps = len(boot)
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0

    below = float(np.sum(boot < theta_hat) + 0.5 * np.sum(boot == theta_hat))
    frac = below / reps
    if not (0.0 < frac < 1.0):
        notes.append("bootstrap distribution entirely on one side of the point "
                     "estimate; BCa bias correction undefined -> percentile")
        return (float(np.percentile(boot, 100 * lo_q)),
                float(np.percentile(boot, 100 * hi_q)), "percentile", notes)
    z0 = norm_ppf(frac)

    jbar = float(np.mean(jack))
    diff = jbar - jack
    denom = 6.0 * float(np.sum(diff ** 2)) ** 1.5
    if denom <= 0 or not math.isfinite(denom):
        notes.append("jackknife has zero spread; acceleration undefined -> "
                     "bias-corrected only (a=0)")
        a = 0.0
    else:
        a = float(np.sum(diff ** 3)) / denom
    if not math.isfinite(a):
        notes.append("non-finite acceleration -> percentile")
        return (float(np.percentile(boot, 100 * lo_q)),
                float(np.percentile(boot, 100 * hi_q)), "percentile", notes)

    out = []
    for q in (lo_q, hi_q):
        z = norm_ppf(q)
        num = z0 + z
        den = 1.0 - a * num
        if den <= 0 or not math.isfinite(den):
            notes.append("BCa adjustment diverged -> percentile")
            return (float(np.percentile(boot, 100 * lo_q)),
                    float(np.percentile(boot, 100 * hi_q)), "percentile", notes)
        adj = norm_cdf(z0 + num / den)
        # keep the requested quantile inside what `reps` replicates can express
        adj = min(max(adj, 1.0 / reps), 1.0 - 1.0 / reps)
        out.append(float(np.percentile(boot, 100 * adj)))
    return out[0], out[1], "bca", notes


def paired_cell_bootstrap(
    arm_by_cell: Mapping[str, float],
    control_by_cell: Mapping[str, float],
    *,
    metric: str,
    arm: str,
    control: str,
    higher_is_better: bool = True,
    reps: int = DEFAULT_REPS,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    method: str = "bca",
    aggregate: str = "median",
) -> PairedDiff:
    """CI on ``arm - control`` over CELL-clustered resamples of the same cells.

    ``arm_by_cell`` / ``control_by_cell`` map a cell key to that cell's value of
    one metric.  Only cells present (and finite) in BOTH maps are used — that
    intersection is the pairing, and its size is reported as ``n_cells`` with the
    dropped count alongside, so a comparison silently thinned to a handful of
    cells is visible rather than merely narrow.
    """
    keys = sorted(set(arm_by_cell) & set(control_by_cell))
    pairs = [(k, float(arm_by_cell[k]), float(control_by_cell[k])) for k in keys]
    pairs = [(k, a, c) for k, a, c in pairs if math.isfinite(a) and math.isfinite(c)]
    n_dropped = (len(set(arm_by_cell) | set(control_by_cell)) - len(pairs))

    sign = 1.0 if higher_is_better else -1.0
    gains = np.asarray([sign * (a - c) for _, a, c in pairs], dtype=float)
    arm_level = _agg(np.asarray([a for _, a, _ in pairs], dtype=float), aggregate)
    ctl_level = _agg(np.asarray([c for _, _, c in pairs], dtype=float), aggregate)
    n = len(gains)

    base = dict(metric=metric, arm=arm, control=control,
                higher_is_better=higher_is_better, reps=int(reps), seed=int(seed),
                alpha=float(alpha), n_cells=n, n_dropped=int(n_dropped),
                arm_level=arm_level, control_level=ctl_level,
                aggregate=str(aggregate))

    if n < MIN_CELLS or int(reps) < 1:
        return PairedDiff(point=_agg(gains, aggregate), ci_lo=float("-inf"),
                          ci_hi=float("inf"), se=float("nan"),
                          method="insufficient", notes=(
                              f"only {n} paired cells (need >= {MIN_CELLS}); "
                              "no interval is reported, so this comparison can "
                              "neither establish a gain nor bound a harm",), **base)

    theta_hat = _agg(gains, aggregate)
    if float(np.ptp(gains)) == 0.0:
        # Every cell moved by exactly the same amount: the bootstrap is a point
        # mass.  Report the collapsed interval rather than manufacturing width --
        # but as ``degenerate``, which carries NO evidence (NO_EVIDENCE_METHODS):
        # ci_lo == ci_hi == theta_hat with se = 0 is arithmetically the most
        # decisive interval expressible, and nothing downstream could tell it
        # apart from a genuine zero-width win.
        return PairedDiff(point=theta_hat, ci_lo=theta_hat, ci_hi=theta_hat,
                          se=0.0, method="degenerate",
                          notes=("every paired cell has an identical difference; "
                                 "the resampling distribution is a point mass, so "
                                 "the zero-width interval expresses no uncertainty "
                                 "and establishes neither a gain nor a bounded "
                                 "harm",),
                          **base)

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(reps), n))
    boot = AGGREGATES[aggregate](gains[idx], axis=1)
    se = float(np.std(boot, ddof=1))

    notes: list[str] = []
    use = method
    if use == "bca" and n < MIN_CELLS_BCA:
        use = "percentile"
        notes.append(f"{n} clusters < {MIN_CELLS_BCA}: jackknife acceleration is "
                     "not trustworthy, reporting the percentile interval")
    if use == "bca":
        jack = np.asarray(
            [_agg(np.delete(gains, i), aggregate) for i in range(n)], dtype=float)
        lo, hi, used, extra = _bca_bounds(theta_hat, boot, jack, alpha)
        notes.extend(extra)
    else:
        lo = float(np.percentile(boot, 100 * alpha / 2.0))
        hi = float(np.percentile(boot, 100 * (1.0 - alpha / 2.0)))
        used = "percentile"
    return PairedDiff(point=theta_hat, ci_lo=lo, ci_hi=hi, se=se, method=used,
                      notes=tuple(notes), **base)


def paired_from_arrays(arm_values: np.ndarray, control_values: np.ndarray,
                       cells: np.ndarray, *, row_aggregate=np.mean,
                       **kw) -> PairedDiff:
    """Convenience wrapper: aggregate per-row values into per-cell values first.

    Used for row-level quantities (an absolute error, say) where the per-cell
    statistic is a simple aggregate.  Ranking statistics must NOT come through
    here -- they are computed per cell by :mod:`.flat_metrics`.
    """
    arm = np.asarray(arm_values, dtype=float)
    control = np.asarray(control_values, dtype=float)
    cells = np.asarray(cells)
    if not (len(arm) == len(control) == len(cells)):
        raise ValueError("paired_from_arrays needs arm, control and cells aligned "
                         "row-for-row (that alignment IS the pairing)")
    a_by, c_by = {}, {}
    for cell in dict.fromkeys(cells.tolist()):
        if cell == "":
            continue
        sel = cells == cell
        ok = sel & np.isfinite(arm) & np.isfinite(control)
        if not ok.any():
            continue
        a_by[str(cell)] = float(row_aggregate(arm[ok]))
        c_by[str(cell)] = float(row_aggregate(control[ok]))
    return paired_cell_bootstrap(a_by, c_by, **kw)


__all__ = [
    "AGGREGATES", "DEFAULT_ALPHA", "DEFAULT_REPS", "MDE_POWER", "MIN_CELLS", "MIN_CELLS_BCA",
    "NO_EVIDENCE_METHODS", "PairedDiff", "norm_cdf", "norm_ppf",
    "paired_cell_bootstrap", "paired_from_arrays",
]
