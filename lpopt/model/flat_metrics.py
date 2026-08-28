"""The flatness-centred pre-registered metric set — program section 8.2.

Every function here returns ``{cell_key: value}``: **one number per design
cell**, never a pooled scalar.  That shape is not cosmetic — the cell is the
resampling unit of the paired bootstrap (section 8.3), so a metric that collapses
to a single number cannot be judged by the apparatus at all.

Why these metrics replace ``Delta75/SD``
---------------------------------------
Section 8.1 measured the draft's primary statistic to be unusable as a decision
input: ``Delta75`` is the lower edge of the first bin reaching 75% sign-hit, so it
takes six distinct values over the whole slate — B1/A2/A3 all scored exactly
1.409 and A1/A4/A6 all scored exactly 0.705.  A statistic with six attainable
values cannot order seven arms; the A6 promotion it produced was a tiebreak
wearing a metric's clothes.  ``Delta75/SD`` is still computed and reported
(:func:`~.ab_eval.effective_resolution`), but it decides nothing.

What decides instead is **regret in physical units** (:func:`regret_at_k`): of
the k candidates the model would actually hand to the search, how much
``node_peak`` is left on the table against the cell's oracle.  That is the loss
the program exists to reduce, it is continuous, and it is invariant to any
binning choice.  The ranking metrics (M1-M4) support it; they are strongly
mutually correlated (section 8.3 "중복성 인지"), which is why they support rather
than triangulate.

Direction convention
--------------------
Each metric declares ``higher_is_better`` in its :class:`MetricSpec`; the paired
layer sign-flips so that a positive difference always means "arm beats control".
Nothing downstream re-derives a direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .ab_eval import DELTA_BINS, spearman
from .folds import MIN_CELL_ROWS

#: Operating-scale bands (section 8.1: the sibling-candidate ``map_cov`` SD is
#: 0.0103, so the decision-relevant differences live at |delta| <= 0.02).
OPERATING_BANDS: tuple[float, ...] = (0.02, 0.01)
#: Delivery slate size — P@8 is "the eight flattest the model would propose".
DEFAULT_K = 8
#: Quantile defining the flat tercile.
FLAT_TERCILE = 1.0 / 3.0
#: Truncation caps for the band-equal-weight AUC (section 8.2 M4).
AUC_CAPS: dict[str, float] = {"node_peak": 0.05, "map_cov": 0.02}
#: Deterministic cap on within-cell pairs (cells are small, but a curriculum
#: campaign cell can carry thousands of rows).
MAX_PAIRS_PER_CELL = 200_000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _groups(cells: np.ndarray, min_rows: int) -> list[tuple[str, np.ndarray]]:
    """``(cell_key, row_index)`` for every cell with enough rows, sorted."""
    cells = np.asarray(cells)
    out: list[tuple[str, np.ndarray]] = []
    for c in sorted({str(x) for x in cells.tolist() if str(x) != ""}):
        idx = np.flatnonzero(cells.astype(str) == c)
        if len(idx) >= min_rows:
            out.append((c, idx))
    return out


def _finite(pred: np.ndarray, true: np.ndarray, idx: np.ndarray) -> np.ndarray:
    ok = np.isfinite(pred[idx]) & np.isfinite(true[idx])
    return idx[ok]


def _within_cell_pairs(n: int, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray]:
    total = n * (n - 1) // 2
    if total <= MAX_PAIRS_PER_CELL:
        return np.triu_indices(n, k=1)
    a = rng.integers(0, n, size=MAX_PAIRS_PER_CELL)
    b = rng.integers(0, n, size=MAX_PAIRS_PER_CELL)
    keep = a != b
    return a[keep], b[keep]


# --------------------------------------------------------------------------- #
# M0 — regret, in physical units (PRIMARY)
# --------------------------------------------------------------------------- #
def regret_at_k(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
                k: int = DEFAULT_K, min_rows: int = MIN_CELL_ROWS,
                lower_is_flatter: bool = True) -> dict[str, float]:
    """Per cell: TRUE value of the best of the model's top-k, minus the oracle.

    This is the quantity the program is trying to reduce.  The search asks the
    model for a shortlist and pays the true flatness of whatever it picks; the
    oracle is the flattest row that was actually in the cell.  Regret is >= 0, is
    zero exactly when the model's shortlist contains the true best, and is in the
    physical units of ``node_peak`` -- so a 0.01 improvement means 0.01 of peak,
    not 0.01 of an arbitrary index.

    Cells with ``n <= k`` are skipped: there the shortlist is the whole cell and
    the regret is identically zero for every arm, which would dilute the paired
    difference with structural zeros.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out: dict[str, float] = {}
    for cell, idx in _groups(cells, max(min_rows, int(k) + 1)):
        sel = _finite(pred, true, idx)
        if len(sel) <= int(k):
            continue
        p = pred[sel] if lower_is_flatter else -pred[sel]
        t = true[sel] if lower_is_flatter else -true[sel]
        order = np.argsort(p, kind="stable")[:int(k)]
        out[cell] = float(np.min(t[order]) - np.min(t))
    return out


# --------------------------------------------------------------------------- #
# M1 — operating-scale sign hit
# --------------------------------------------------------------------------- #
def sign_hit_band(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
                  band: float, min_rows: int = MIN_CELL_ROWS,
                  seed: int = 0, min_pairs: int = 20) -> dict[str, float]:
    """Per cell: sign-hit rate restricted to pairs with ``0 < |dtrue| <= band``.

    The pooled sign-hit rate is dominated by easy pairs -- section 8.1 measured
    44% of ``map_cov`` pairs sitting in the ``|delta| >= 0.1`` band at a 0.977 hit
    rate, while the two bands that matter carry 7.6% of the weight.  Restricting
    to the operating band is what makes the statistic answer the question the
    search asks.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    rng = np.random.default_rng(int(seed))
    out: dict[str, float] = {}
    for cell, idx in _groups(cells, min_rows):
        sel = _finite(pred, true, idx)
        if len(sel) < 3:
            continue
        ai, bi = _within_cell_pairs(len(sel), rng)
        dt = true[sel][ai] - true[sel][bi]
        dp = pred[sel][ai] - pred[sel][bi]
        keep = (np.abs(dt) > 0) & (np.abs(dt) <= float(band))
        if int(keep.sum()) < min_pairs:
            continue
        out[cell] = float(((dt[keep] > 0) == (dp[keep] > 0)).mean())
    return out


# --------------------------------------------------------------------------- #
# M2 — flat-tercile within-cell Spearman
# --------------------------------------------------------------------------- #
def flat_tercile_rho(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
                     q: float = FLAT_TERCILE, min_rows: int = MIN_CELL_ROWS,
                     min_tercile_rows: int = 4,
                     lower_is_flatter: bool = True) -> dict[str, float]:
    """Per cell: Spearman(pred, true) among the flattest tercile of TRUE rows.

    Ranking skill averaged over a whole cell is not the skill the search
    consumes: once the campaign has converged on the flat end, every remaining
    decision is *inside* the flat tercile, where the spread is smallest and the
    model is weakest.  Conditioning on the true tercile is deliberate and is
    stated in the pre-registration -- it makes this a conditional estimand, not a
    biased estimate of the unconditional one.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out: dict[str, float] = {}
    for cell, idx in _groups(cells, min_rows):
        sel = _finite(pred, true, idx)
        if len(sel) < max(min_tercile_rows, 3):
            continue
        t = true[sel]
        cut = float(np.quantile(t, q if lower_is_flatter else 1.0 - q))
        flat = t <= cut if lower_is_flatter else t >= cut
        if int(flat.sum()) < min_tercile_rows:
            continue
        r = spearman(pred[sel][flat], t[flat])
        if math.isfinite(r):
            out[cell] = float(r)
    return out


# --------------------------------------------------------------------------- #
# M3 — normalized top-k recovery ("P@8-flattest")
# --------------------------------------------------------------------------- #
def normalized_precision_at_k(pred: np.ndarray, true: np.ndarray,
                              cells: np.ndarray, *, k: int = DEFAULT_K,
                              min_rows: int = MIN_CELL_ROWS,
                              lower_is_flatter: bool = True) -> dict[str, float]:
    """Per cell: ``P@k / (k/n)`` -- top-k recovery over the chance rate.

    Raw ``P@k`` is not comparable across cells because ``k/n`` (the value a coin
    flip attains) differs per cell; normalizing by it makes 1.0 mean "no better
    than random" in every cell, so a median over cells is meaningful.  Section
    8.2 specifies the normalized form explicitly.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out: dict[str, float] = {}
    kk = int(k)
    for cell, idx in _groups(cells, max(min_rows, kk + 1)):
        sel = _finite(pred, true, idx)
        n = len(sel)
        if n <= kk:
            continue
        p = pred[sel] if lower_is_flatter else -pred[sel]
        t = true[sel] if lower_is_flatter else -true[sel]
        top_p = set(np.argsort(p, kind="stable")[:kk].tolist())
        top_t = set(np.argsort(t, kind="stable")[:kk].tolist())
        chance = kk / float(n)
        out[cell] = float((len(top_p & top_t) / kk) / chance)
    return out


# --------------------------------------------------------------------------- #
# M4 — band-equal-weight truncated AUC
# --------------------------------------------------------------------------- #
def truncated_band_auc(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
                       bin_target: str, cap: float | None = None,
                       min_rows: int = MIN_CELL_ROWS, seed: int = 0,
                       min_bin_pairs: int = 10) -> dict[str, float]:
    """Per cell: mean over |dtrue| bins (EQUAL weight) of the sign-hit rate.

    Equal weight per bin, and truncation at the operating cap, is what stops the
    statistic degenerating into the pooled concordance -- section 8.1 showed that
    a pair-weighted average over ``DELTA_BINS`` is algebraically
    ``(Kendall tau_b + 1)/2`` and is therefore decided by the easy bins.
    """
    edges = [e for e in DELTA_BINS.get(bin_target, ()) if math.isfinite(e)]
    limit = float(cap if cap is not None else AUC_CAPS.get(bin_target, math.inf))
    edges = [e for e in edges if e <= limit]
    if len(edges) < 2:
        return {}
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    rng = np.random.default_rng(int(seed))
    out: dict[str, float] = {}
    for cell, idx in _groups(cells, min_rows):
        sel = _finite(pred, true, idx)
        if len(sel) < 3:
            continue
        ai, bi = _within_cell_pairs(len(sel), rng)
        dt = true[sel][ai] - true[sel][bi]
        dp = pred[sel][ai] - pred[sel][bi]
        adt = np.abs(dt)
        hit = (dt > 0) == (dp > 0)
        rates = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (adt >= lo) & (adt < hi) & (adt > 0)
            if int(m.sum()) >= min_bin_pairs:
                rates.append(float(hit[m].mean()))
        if rates:
            out[cell] = float(np.mean(rates))
    return out


# --------------------------------------------------------------------------- #
# M5 / M6 / M7 — non-inferiority quantities
# --------------------------------------------------------------------------- #
def cell_rho(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
             min_rows: int = MIN_CELL_ROWS) -> dict[str, float]:
    """Per cell: plain within-cell Spearman (the M5 non-inferiority quantity)."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out: dict[str, float] = {}
    for cell, idx in _groups(cells, min_rows):
        sel = _finite(pred, true, idx)
        if len(sel) < 3:
            continue
        r = spearman(pred[sel], true[sel])
        if math.isfinite(r):
            out[cell] = float(r)
    return out


def cell_mae(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
             min_rows: int = MIN_CELL_ROWS) -> dict[str, float]:
    """Per cell: MAE.  M7 exists because the acquisition consumes LEVELS.

    A model can rank a cell perfectly and still be useless to the UCB, which
    de-biases and adds a sigma to a physical level.  Ranking metrics are blind to
    that, so absolute accuracy is a separate gate rather than a footnote.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out: dict[str, float] = {}
    for cell, idx in _groups(cells, min_rows):
        sel = _finite(pred, true, idx)
        if len(sel) < 2:
            continue
        out[cell] = float(np.mean(np.abs(pred[sel] - true[sel])))
    return out


def cell_abs_bias(pred: np.ndarray, true: np.ndarray, cells: np.ndarray, *,
                  min_rows: int = MIN_CELL_ROWS) -> dict[str, float]:
    """Per cell: ``|mean(pred - true)|`` -- the level error the gate correction
    has to absorb."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out: dict[str, float] = {}
    for cell, idx in _groups(cells, min_rows):
        sel = _finite(pred, true, idx)
        if len(sel) < 2:
            continue
        out[cell] = float(abs(np.mean(pred[sel] - true[sel])))
    return out


# --------------------------------------------------------------------------- #
# the frozen registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricSpec:
    """One pre-registered metric.  Frozen: editing after launch converts the
    pre-registered test into a post-hoc one."""

    key: str
    family: str                 # M0 .. M7
    target: str
    label: str
    higher_is_better: bool
    role: str                   # primary | support | harm | report
    fn: Callable[..., dict[str, float]]
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: How the paired layer aggregates this metric OVER cells.  Median
    #: everywhere except regret: regret is a non-negative, zero-inflated loss
    #: whose per-cell median is 0 for any competent model (top-8 of a ~26-row
    #: cell is a wide net), so a median-of-differences is structurally 0 and
    #: cannot see the improvement.  The mean is the expected loss per cell,
    #: which is what the search actually pays.
    aggregate: str = "median"

    def per_cell(self, pred: np.ndarray, true: np.ndarray,
                 cells: np.ndarray) -> dict[str, float]:
        return self.fn(pred, true, cells, **self.kwargs)


def _spec(key, family, target, label, hib, role, fn, aggregate="median",
          **kw) -> MetricSpec:
    return MetricSpec(key=key, family=family, target=target, label=label,
                      higher_is_better=hib, role=role, fn=fn, kwargs=dict(kw),
                      aggregate=aggregate)


#: Section 8.2, transcribed.  ``primary`` metrics must ESTABLISH a gain;
#: ``harm`` metrics must BOUND a harm; ``support``/``report`` are printed.
PRE_REGISTERED_METRICS: tuple[MetricSpec, ...] = (
    # M0 — regret (the go/no-go axis)
    _spec("M0_regret8_node_peak", "M0", "node_peak",
          "top-8 regret (node_peak, physical)", False, "primary",
          regret_at_k, aggregate="mean", k=8),
    _spec("M0_regret1_node_peak", "M0", "node_peak",
          "top-1 regret (node_peak, physical)", False, "support",
          regret_at_k, aggregate="mean", k=1),
    _spec("M0_regret8_map_cov", "M0", "map_cov",
          "top-8 regret (map_cov, physical)", False, "support",
          regret_at_k, aggregate="mean", k=8),
    # M1 — operating-scale sign hit
    _spec("M1_signhit_0.02_node_peak", "M1", "node_peak",
          "sign-hit |d|<=0.02 (node_peak)", True, "primary",
          sign_hit_band, band=0.02),
    _spec("M1_signhit_0.01_node_peak", "M1", "node_peak",
          "sign-hit |d|<=0.01 (node_peak)", True, "support",
          sign_hit_band, band=0.01),
    _spec("M1_signhit_0.02_map_cov", "M1", "map_cov",
          "sign-hit |d|<=0.02 (map_cov)", True, "support",
          sign_hit_band, band=0.02),
    # M2 — flat-tercile skill
    _spec("M2_flat_tercile_rho_node_peak", "M2", "node_peak",
          "flat-tercile within-cell rho (node_peak)", True, "harm",
          flat_tercile_rho),
    _spec("M2_flat_tercile_rho_map_cov", "M2", "map_cov",
          "flat-tercile within-cell rho (map_cov)", True, "harm",
          flat_tercile_rho),
    # M3 — normalized P@8
    _spec("M3_norm_p_at_8_node_peak", "M3", "node_peak",
          "P@8-flattest / chance (node_peak)", True, "harm",
          normalized_precision_at_k, k=8),
    _spec("M3_norm_p_at_8_map_cov", "M3", "map_cov",
          "P@8-flattest / chance (map_cov)", True, "harm",
          normalized_precision_at_k, k=8),
    # M4 — truncated band-equal AUC (reported)
    _spec("M4_trunc_auc_node_peak", "M4", "node_peak",
          "band-equal AUC truncated at 0.05 (node_peak)", True, "report",
          truncated_band_auc, bin_target="node_peak"),
    _spec("M4_trunc_auc_map_cov", "M4", "map_cov",
          "band-equal AUC truncated at 0.02 (map_cov)", True, "report",
          truncated_band_auc, bin_target="map_cov"),
    # M5 — secondary non-regression
    _spec("M5_cell_rho_f_r", "M5", "f_r", "within-cell rho (f_r)", True, "harm",
          cell_rho),
    _spec("M5_cell_rho_f_q", "M5", "f_q", "within-cell rho (f_q)", True, "harm",
          cell_rho),
    _spec("M5_cell_rho_cyclen", "M5", "cyclen", "within-cell rho (cyclen)", True,
          "harm", cell_rho),
    # M7 — absolute accuracy (what the acquisition consumes)
    _spec("M7_cell_mae_node_peak", "M7", "node_peak", "within-cell MAE (node_peak)",
          False, "harm", cell_mae),
    _spec("M7_cell_mae_map_cov", "M7", "map_cov", "within-cell MAE (map_cov)",
          False, "harm", cell_mae),
    _spec("M7_abs_bias_node_peak", "M7", "node_peak", "|within-cell bias| (node_peak)",
          False, "harm", cell_abs_bias),
)

METRICS_BY_KEY: dict[str, MetricSpec] = {m.key: m for m in PRE_REGISTERED_METRICS}

#: --- A/B round-2 VARIANCE arms: the target axes (pre-registered 20260730) ----
#: ``data/reports/ab2_preregistration_20260730.md`` registers three arms whose
#: target axes are per-cell MAE on scalar targets the flatness registry above
#: does not carry (the flatness program's own axes are node_peak / map_cov).
#:
#: These live in a SEPARATE tuple, deliberately: :data:`PRE_REGISTERED_METRICS`
#: is the frozen scoring surface of the flatness program and every arm ever
#: judged against it was judged on exactly those keys.  Appending to it — even a
#: ``report``-role metric — would change that surface after the fact.  The round-2
#: axes reuse the same primitive (:func:`cell_mae`), the same per-cell clustering
#: and the same paired machinery (``ab_paired.paired_cell_bootstrap``), so the two
#: sets are directly comparable without being the same object.
#:
#: ``higher_is_better=False`` on all four: they are errors.  ``median`` aggregate,
#: because unlike regret a per-cell MAE is not zero-inflated.
AB2_TARGET_METRICS: tuple[MetricSpec, ...] = (
    _spec("T_cell_mae_cyclen", "T", "cyclen", "within-cell MAE (cyclen, EFPD)",
          False, "primary", cell_mae),
    _spec("T_cell_mae_cbc_max", "T", "cbc_max", "within-cell MAE (cbc_max, ppm)",
          False, "primary", cell_mae),
    _spec("T_cell_mae_f_q", "T", "f_q", "within-cell MAE (f_q)",
          False, "primary", cell_mae),
    _spec("T_cell_mae_node_peak", "T", "node_peak", "within-cell MAE (node_peak)",
          False, "primary", cell_mae),
    _spec("T_abs_bias_cbc_max", "T", "cbc_max", "|within-cell bias| (cbc_max, ppm)",
          False, "support", cell_abs_bias),
)

AB2_METRICS_BY_KEY: dict[str, MetricSpec] = {m.key: m for m in AB2_TARGET_METRICS}

#: Which target axes each round-2 arm must IMPROVE (CI lower bound > 0 on the
#: sign-flipped paired difference vs the A0 control).  Transcribed from the
#: pre-registration document so an artifact carries the rule it was judged under;
#: an arm listed here with more than one axis must improve EVERY listed axis.
AB2_ARM_TARGET_AXES: dict[str, tuple[str, ...]] = {
    "A1": ("T_cell_mae_cyclen", "T_cell_mae_cbc_max"),
    "A2": ("T_cell_mae_cbc_max",),
    "A3": ("T_cell_mae_node_peak", "T_cell_mae_f_q"),
}


def metrics_for(targets: Sequence[str]) -> tuple[MetricSpec, ...]:
    """The subset of the registry whose target is available in a slice."""
    have = set(targets)
    return tuple(m for m in PRE_REGISTERED_METRICS if m.target in have)


__all__ = [
    "AUC_CAPS", "DEFAULT_K", "FLAT_TERCILE", "METRICS_BY_KEY", "MetricSpec",
    "AB2_ARM_TARGET_AXES", "AB2_METRICS_BY_KEY", "AB2_TARGET_METRICS",
    "OPERATING_BANDS", "PRE_REGISTERED_METRICS", "cell_abs_bias", "cell_mae",
    "cell_rho", "flat_tercile_rho", "metrics_for", "normalized_precision_at_k",
    "regret_at_k", "sign_hit_band", "truncated_band_auc",
]
