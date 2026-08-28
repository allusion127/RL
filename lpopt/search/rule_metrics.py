"""Engineering-rule metrics RM1-RM6 (public-literature PWR loading rules).

Every function here is a **pure** ``pattern -> float`` (or ``map -> float``)
arrangement statistic.  None of them is a constraint and none of them may ever
become one: the source report's own lesson (the McFLOP / "Ring-of-Fire" case) is
that promoting a loading HEURISTIC to a hard constraint truncates the search
space and removes the optimum along with the bad patterns.  They are adopted
here as (a) an OPTIONAL soft penalty in :func:`..search.acquisition.score_flat_power`
(default weight 0.0 -> byte-identical), (b) an OPTIONAL generator bias in
:mod:`.produce` (default off), and (c) a report axis.

GEOMETRY (the measured study's M1/M2/M4, verified independently)
---------------------------------------------------------------
Adjacency is evaluated in the **mirror-expanded 17x17 full core**, not on the
raw 9x9 quarter, so a slot sitting on a symmetry axis sees its own mirror image
— which is the physically correct neighbour, because the assembly across the
axis IS that assembly.  Face neighbours are :data:`..data.flatness.NEIGH_SLOT`
(the same table the flatness gradient diagnostic uses); the diagonal twin
:data:`DIAG_SLOT` is built here with the same construction and the four
``(+-1, +-1)`` offsets.

Counts are **multiplicity-weighted full-core** counts::

    RM1 = 0.5 * sum_i w_i * #{face nb j of i : fresh_i and fresh_j}

with ``w = SLOT_WEIGHTS`` (1 / 2 / 4).  This is exact, not an approximation:
every symmetry image of a slot has a congruent neighbourhood, so the
south-east representative times ``w_i`` reproduces the true ordered-pair count
over the full 241-assembly core, and the ``0.5`` turns ordered pairs into
unordered physical pairs.  (The identity ``0.5 * sum_i w_i * deg_i ==
full-core unordered pair count`` was brute-force checked against an independent
17x17 enumeration over random fresh sets: 0 mismatches.)

"Periphery" is :data:`..data.flatness.PERIPHERY_MASK` — a slot with at least one
face looking OUT of the fuel region, i.e. exactly the outermost ring (13 of the
69 quarter slots == 48 of the 241 full-core assemblies).

MEASURED EVIDENCE (this store, this cell definition)
----------------------------------------------------
Within-cell Spearman rho against ``node_peak`` (cell = ``campaign |
case_pair | feed``; 218 cells, 42,050 converged rows, 33,543 carrying a
flatness label; within-cell ``e_core`` std median 0.0049, so only ARRANGEMENT
varies inside a cell).  Size-weighted mean rho [IQR across cells]; the sign
convention is *lower ``node_peak`` is better*, so a POSITIVE rho means "more of
this metric is worse".  Each function's docstring repeats its own row.

Two facts govern which of these were adopted as a penalty:

* the causal carrier is **inboard fresh clustering** (``inboard=True``), not
  peripheral fresh per se — ``partial(RM1i | RM4) -> node_peak = +0.134`` while
  ``partial(RM4 | RM1i) -> node_peak = -0.045`` (null), and RM1i ~ RM4 is
  ``-0.715``, i.e. the two are largely ONE axis read from opposite ends;
* the peaking effect of RM1i is fully **mediated** by peripheral power share
  (``partial(RM1i | RM5) -> node_peak = -0.069``), which is why RM5 is the
  mechanism/report variable and never a candidate score.

ADOPTED as soft-penalty candidates: :func:`rm_fresh_face_adjacency` (R-03, both
variants) and :func:`rm_fresh_diag_adjacency` (R-04, both variants).  See
:data:`VALIDATED_PENALTY_METRICS`.

NOT adopted (report-only), and why:

* **RM3** :func:`rm_reactivity_mismatch` — zero-order rho ``+0.016`` with 47.7%
  of cells negative, i.e. null; and it is ``rho = -0.885`` collinear with RM1,
  the largest magnitude in the whole metric matrix.  RM3 is a noisily negated
  RM1, not an independent axis, so it can never earn its own penalty slot.
* **RM4** :func:`rm_fresh_periphery` — the L-01 low-leakage doctrine's
  prediction is CONFIRMED on the mechanism (rho vs RM5 peripheral power share
  ``+0.397``) but its apparent ``-0.189`` against ``node_peak`` is entirely the
  RM1i axis read backwards; conditioned on RM1i it is null (``-0.045``), and a
  "prefer low RM4" selection arm made ``node_peak`` WORSE by ``+0.064``.
  Report axis only.
* **RM6** :func:`rm_checkerboard_degree` — ``-0.008`` with 50.7% of cells
  negative; indistinguishable from noise.
* **RM5** :func:`rm_peripheral_power_share` — ``-0.744`` vs ``node_peak``, but
  RM5 and ``node_peak`` are computed from the SAME harvested BOC map, so that
  number is an identity diagnostic and NOT predictive skill.  Report axis (L-03)
  and mechanism variable only; it is not a function of the pattern and so cannot
  be a candidate score at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..data.flatness import (
    N_SLOTS,
    NEIGH_SLOT,
    NEIGH_VALID,
    PERIPHERY_MASK,
    SLOT_WEIGHTS,
    slot_values,
)
from ..vendor.masterrl.domain import SLOTS

__all__ = [
    "BURN_FACTOR",
    "DIAG_SLOT",
    "DIAG_VALID",
    "PENALTY_METRIC_FUNCS",
    "VALIDATED_PENALTY_METRICS",
    "enrichment_by_batch",
    "fresh_mask",
    "reactivity_index",
    "rm_checkerboard_degree",
    "rm_fresh_diag_adjacency",
    "rm_fresh_face_adjacency",
    "rm_fresh_periphery",
    "rm_peripheral_power_share",
    "rm_reactivity_mismatch",
    "rule_penalty",
]

_CORE = 17
_CORE_CENTER = 8


def _diagonal_gather() -> tuple[np.ndarray, np.ndarray]:
    """``(diag_slot[69, 4], in_core[69, 4])`` diagonal neighbours, FULL core.

    Built exactly like :func:`..data.flatness._neighbour_gather` but with the
    four ``(+-1, +-1)`` offsets, so the axis-mirror behaviour (M1) is identical.
    """
    mask = np.zeros((_CORE, _CORE), dtype=bool)
    slot_of = np.full((_CORE, _CORE), -1, dtype=np.intp)
    for slot in SLOTS:
        for dr in {slot.row, -slot.row}:
            for dc in {slot.col, -slot.col}:
                mask[_CORE_CENTER + dr, _CORE_CENTER + dc] = True
                slot_of[_CORE_CENTER + dr, _CORE_CENTER + dc] = slot.index
    neigh = np.full((N_SLOTS, 4), -1, dtype=np.intp)
    for slot in SLOTS:
        r, c = _CORE_CENTER + slot.row, _CORE_CENTER + slot.col
        for k, (dr, dc) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            rr, cc = r + dr, c + dc
            if 0 <= rr < _CORE and 0 <= cc < _CORE and mask[rr, cc]:
                neigh[slot.index, k] = slot_of[rr, cc]
    return neigh, neigh >= 0


#: Diagonal analogue of :data:`..data.flatness.NEIGH_SLOT` (R-04).
DIAG_SLOT, DIAG_VALID = _diagonal_gather()

#: RM3 burned-reactivity proxy by residence age (1 = fresh).  The task offered
#: ``enrichment*0.6`` once-burned / ``*0.4`` twice-burned; ages >= 4 continue the
#: same monotone decay.  Documented choice, not a fitted quantity.
BURN_FACTOR: dict[int, float] = {1: 1.0, 2: 0.6, 3: 0.4, 4: 0.3, 5: 0.25}
_BURN_FLOOR = 0.2

#: The metrics that survived the within-cell study as SOFT-PENALTY candidates.
#: Anything outside this set is report-only by decision, and
#: :func:`rule_penalty` refuses it rather than silently applying an unvalidated
#: (or, for RM4, actively counter-productive) shaping term.
VALIDATED_PENALTY_METRICS: tuple[str, ...] = ("rm1", "rm1i", "rm2", "rm2i")


# --------------------------------------------------------------------------- #
# pattern -> per-slot primitives
# --------------------------------------------------------------------------- #
def _items(pattern: Any) -> Sequence[Any]:
    items = getattr(pattern, "items", None)
    if items is None:
        raise TypeError(f"expected a vendor Pattern, got {type(pattern)!r}")
    if len(items) != N_SLOTS:
        raise ValueError(f"pattern must carry {N_SLOTS} slots, got {len(items)}")
    return items


def fresh_mask(pattern: Any) -> np.ndarray:
    """``bool[69]`` — which quarter slots hold a FRESH assembly."""
    return np.array([bool(item.is_fresh) for item in _items(pattern)], dtype=bool)


def _pair_count(flag: np.ndarray, neigh: np.ndarray, valid: np.ndarray,
                *, inboard: bool) -> float:
    """``0.5 * sum_i w_i * #{nb j : flag_i and flag_j}`` (unordered full-core pairs).

    ``inboard`` additionally requires BOTH slots to be non-peripheral.
    """
    keep = valid & flag[:, None] & flag[np.where(valid, neigh, 0)]
    if inboard:
        inb = ~PERIPHERY_MASK
        keep = keep & inb[:, None] & inb[np.where(valid, neigh, 0)]
    return float(0.5 * (SLOT_WEIGHTS * keep.sum(axis=1)).sum())


def enrichment_by_batch(library: Any, library_id: str) -> dict[str, float]:
    """``{batch_id: u_avg_enrichment}`` for one fuel library.

    Convenience for :func:`rm_reactivity_mismatch`; the library is a
    :class:`..data.fuel_types.FuelLibrary`.  Types with no enrichment recorded
    are omitted (the caller's default then applies).
    """
    out: dict[str, float] = {}
    for type_id in library.types(library_id):
        try:
            vec = library.get(type_id, library_id)
        except KeyError:  # pragma: no cover - defensive
            continue
        enr = getattr(vec, "u_avg_enrichment", None)
        if enr is not None and np.isfinite(float(enr)):
            out[str(type_id)] = float(enr)
    return out


def reactivity_index(pattern: Any,
                     enrichment: Mapping[str, float] | None = None,
                     *, default_enrichment: float = 1.0) -> np.ndarray:
    """``RI[69]`` — the RM3 reactivity proxy of each quarter slot.

    A burned assembly is traced up its shuffle chain to its FRESH ORIGIN (the
    same walk as :meth:`..model.featurize.PatternEncoder._trace_chain`), and

    ``RI = u_avg_enrichment(origin batch) * BURN_FACTOR[age]``

    with ``age`` 1 for fresh, 2 once-burned, 3 twice-burned, ... .  ``enrichment``
    maps a fresh batch id to its ``u_avg_enrichment`` (see
    :func:`enrichment_by_batch`); ``None`` gives every batch
    ``default_enrichment``, which reduces RI to the pure residence-age proxy.
    """
    from ..vendor.masterrl.ga import _coord_slot

    items = _items(pattern)
    n = len(items)
    age = [0] * n
    origin = [0] * n
    resolving = [False] * n

    def resolve(s: int) -> None:
        if age[s]:
            return
        item = items[s]
        if item.is_fresh:
            age[s], origin[s] = 1, s
            return
        src = _coord_slot(item.x, int(item.y))
        if src is None or src == s or resolving[src]:
            age[s], origin[s] = 1, s          # defensive: treat as its own root
            return
        resolving[s] = True
        resolve(src)
        resolving[s] = False
        age[s] = age[src] + 1
        origin[s] = origin[src]

    for s in range(n):
        resolve(s)

    ri = np.empty(n, dtype=np.float64)
    for s in range(n):
        batch = getattr(items[origin[s]], "batch", None)
        enr = default_enrichment
        if enrichment is not None and batch is not None:
            enr = float(enrichment.get(str(batch), default_enrichment))
        ri[s] = enr * BURN_FACTOR.get(age[s], _BURN_FLOOR)
    return ri


# --------------------------------------------------------------------------- #
# RM1 / RM2 — fresh-fresh adjacency (R-03 / R-04)
# --------------------------------------------------------------------------- #
def rm_fresh_face_adjacency(pattern: Any, *, inboard: bool = False) -> float:
    """RM1 — fresh-fresh FACE adjacency pair count (rule **R-03**).

    Multiplicity-weighted count of UNORDERED full-core face pairs in which both
    assemblies are fresh.  R-03 says fewer fresh face neighbours means lower
    local peaking, i.e. a POSITIVE correlation with ``node_peak`` is the
    predicted sign.

    MEASURED within-cell Spearman rho (218-cell study; positive = higher metric
    is worse)::

        variant              node_peak   map_cov     f_q      f_r     cyclen
        RM1  (whole core)      +0.085     +0.071   +0.088   +0.079   -0.041
        RM1i (inboard=True)    +0.235     +0.295   +0.282   +0.254   -0.056

    The whole-core form is weak and noisy — the IQR ``[-0.083, +0.137]`` straddles
    zero and 44.2% of cells carry a negative rho — but the sign is right and it
    survives out-of-sample against the surrogate's own error (residual rho
    ``+0.114``, p = 5.6e-05, on the S1 holdout).  ``inboard=True`` requires BOTH
    slots to be non-peripheral and is much stronger (``+0.235``, 26.7% of cells
    negative, IQR ``[-0.019, +0.268]``); the study's partial correlations
    identify it as the causal carrier.  Prefer it when adopting one variant.
    """
    return _pair_count(fresh_mask(pattern), NEIGH_SLOT, NEIGH_VALID,
                       inboard=inboard)


def rm_fresh_diag_adjacency(pattern: Any, *, inboard: bool = False) -> float:
    """RM2 — fresh-fresh DIAGONAL adjacency pair count (rule **R-04**).

    Same construction as :func:`rm_fresh_face_adjacency` over the four diagonal
    neighbours.  R-04 is the report's weaker companion to R-03, and the data
    agree that it is weaker on ``node_peak``::

        variant              node_peak   map_cov     f_q      f_r     cyclen
        RM2  (whole core)      +0.076     +0.116   +0.126   +0.112   -0.053
        RM2i (inboard=True)    +0.172     +0.263   +0.251   +0.221   -0.076

    Note RM2 tracks ``map_cov`` / ``f_q`` better than it tracks ``node_peak``,
    and RM2i ~ RM2 is ``+0.754`` while RM2i ~ RM4 is ``-0.720``, so the inboard
    variant again rides the inboard-clustering axis.
    """
    return _pair_count(fresh_mask(pattern), DIAG_SLOT, DIAG_VALID,
                       inboard=inboard)


# --------------------------------------------------------------------------- #
# RM3 — neighbour reactivity mismatch (REPORT-ONLY: refuted as an own axis)
# --------------------------------------------------------------------------- #
def rm_reactivity_mismatch(pattern: Any,
                           enrichment: Mapping[str, float] | None = None,
                           *, default_enrichment: float = 1.0) -> float:
    """RM3 — ``M_face = 0.5 * sum_i w_i * sum_{face j} |RI_i - RI_j|``.

    ``RI`` is :func:`reactivity_index` (fresh-origin enrichment times a
    residence-age burn factor, :data:`BURN_FACTOR`).  The source report warns
    that blindly MINIMIZING mismatch throws away the power-sharing benefit of
    deliberately mixing reactivities, so the sign was left an empirical question.

    MEASURED::

        node_peak   map_cov      f_q      f_r     cyclen
          +0.016     +0.069    +0.010   +0.016   +0.069

    i.e. **null** — 47.7% of cells carry a negative rho and only 8.3% reach
    p<0.05 on ``node_peak``.  **REPORT-ONLY.**  It is furthermore ``rho = -0.885``
    collinear with RM1 (the largest magnitude in the metric matrix), so its
    apparent conditional signal (``partial(RM3|RM1) -> node_peak = +0.182``
    against a zero-order ``+0.017``) is textbook suppression, not an independent
    lever.  RM3 and RM1 must never be adopted as two separate penalties.
    """
    ri = reactivity_index(pattern, enrichment,
                          default_enrichment=default_enrichment)
    gathered = ri[np.where(NEIGH_VALID, NEIGH_SLOT, 0)]
    diff = np.where(NEIGH_VALID, np.abs(gathered - ri[:, None]), 0.0)
    return float(0.5 * (SLOT_WEIGHTS * diff.sum(axis=1)).sum())


# --------------------------------------------------------------------------- #
# RM4 — fresh in the periphery (REPORT-ONLY: mechanism confirmed, lever refuted)
# --------------------------------------------------------------------------- #
def rm_fresh_periphery(pattern: Any) -> float:
    """RM4 — multiplicity-weighted count of FRESH assemblies on the outer ring.

    Low-leakage doctrine **L-01** ("burned fuel belongs at the periphery")
    predicts that more peripheral fresh RAISES peripheral power share and costs
    economy.  The mechanism half is CONFIRMED: rho(RM4, RM5) = ``+0.397``, the
    strongest mechanism coupling of any metric here.

    MEASURED against the flatness labels::

        node_peak   map_cov      f_q      f_r     cyclen
          -0.189     -0.285    -0.222   -0.205   -0.105

    **REPORT-ONLY, deliberately not a penalty.**  That negative rho is the RM1i
    inboard-clustering axis read backwards (RM1i ~ RM4 = ``-0.715``): conditioned
    on RM1i, RM4 is null (``partial(RM4|RM1i) -> node_peak = -0.045``), whereas
    RM1i survives conditioning on RM4 (``+0.134``).  Acting on RM4 directly is
    worse than doing nothing — the "prefer LOW RM4" selection arm moved mean
    ``node_peak`` by ``+0.064`` against a cell mean of 1.711, the worst arm in
    the study.  The economy half of L-01 was not tested here (no economics label).
    """
    return float((SLOT_WEIGHTS * fresh_mask(pattern) * PERIPHERY_MASK).sum())


# --------------------------------------------------------------------------- #
# RM5 — peripheral power share (REPORT AXIS L-03; same-map circular, never a score)
# --------------------------------------------------------------------------- #
def rm_peripheral_power_share(maps: Any) -> float:
    """RM5 — outer-ring share of total core power, from the BOC power map.

    ``sum(w_i p_i over the periphery) / sum(w_i p_i over all)``, NaN slots
    dropped from both sums.  Accepts every map layout
    :func:`..data.flatness.slot_values` accepts (69-vector, ``(9, 9)``, the
    legacy ``(4, 9, 9)`` stack with BOC power at channel 0, the trajectory
    stack).  Returns ``nan`` for an absent or unusable map.

    Vessel-fluence proxy for rule **L-03**: radial flattening buys its peak
    reduction by raising the periphery, which raises RPV beltline fast flux and
    baffle heating.

    MEASURED::

        node_peak   map_cov      f_q      f_r     cyclen
          -0.744     -0.933    -0.728   -0.726   -0.033

    **CIRCULARITY WARNING — REPORT AXIS ONLY.**  RM5 and ``node_peak`` /
    ``map_cov`` / ``f_q`` / ``f_r`` are derived from the SAME harvested BOC map,
    so those four numbers are an identity diagnostic, not predictive skill.  RM5
    is also not a function of the pattern, so it cannot be a candidate score at
    all.  It is the MECHANISM variable of the study: the peaking effect of
    inboard fresh clustering is fully mediated by it
    (``partial(RM1i|RM5) -> node_peak = -0.069``).
    """
    try:
        vals = slot_values(maps)
    except (ValueError, TypeError):
        return float("nan")
    if vals is None or vals.shape[0] < 1:
        return float("nan")
    v = vals[0]
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan")
    w = np.where(ok, SLOT_WEIGHTS, 0.0)
    total = float((w * np.where(ok, v, 0.0)).sum())
    if not np.isfinite(total) or total == 0.0:
        return float("nan")
    edge = float((w * PERIPHERY_MASK * np.where(ok, v, 0.0)).sum())
    return edge / total


# --------------------------------------------------------------------------- #
# RM6 — checkerboard degree (REPORT-ONLY: refuted, indistinguishable from noise)
# --------------------------------------------------------------------------- #
def rm_checkerboard_degree(pattern: Any) -> float:
    """RM6 — weighted fraction of FRESH assemblies with NO fresh face neighbour.

    1.0 is a perfect checkerboard (no fresh assembly touches another face-wise),
    0.0 means every fresh assembly has at least one fresh face neighbour.
    Returns ``nan`` when the pattern carries no fresh assembly.

    MEASURED::

        node_peak   map_cov      f_q      f_r     cyclen
          -0.008     +0.000    +0.023   +0.020   -0.052

    **REFUTED / REPORT-ONLY.**  50.7% of cells carry a negative rho and only 7.8%
    reach p<0.05; this is noise.  It is also ``-0.664`` collinear with RM1 and
    ``+0.631`` with RM3, so it carries no information those do not.
    """
    fresh = fresh_mask(pattern)
    if not fresh.any():
        return float("nan")
    nb_fresh = (NEIGH_VALID & fresh[np.where(NEIGH_VALID, NEIGH_SLOT, 0)])
    isolated = fresh & ~nb_fresh.any(axis=1)
    denom = float((SLOT_WEIGHTS * fresh).sum())
    return float((SLOT_WEIGHTS * isolated).sum() / denom)


# --------------------------------------------------------------------------- #
# the soft-penalty adapter
# --------------------------------------------------------------------------- #
#: Name -> ``pattern -> float`` for the VALIDATED penalty metrics only.
PENALTY_METRIC_FUNCS = {
    "rm1": lambda p: rm_fresh_face_adjacency(p, inboard=False),
    "rm1i": lambda p: rm_fresh_face_adjacency(p, inboard=True),
    "rm2": lambda p: rm_fresh_diag_adjacency(p, inboard=False),
    "rm2i": lambda p: rm_fresh_diag_adjacency(p, inboard=True),
}


def rule_penalty(patterns: Sequence[Any],
                 weights: Mapping[str, float] | None) -> np.ndarray:
    """``penalty[N] = sum_m weight_m * metric_m(pattern)`` (>= 0, never NaN).

    A SOFT shaping term: the caller SUBTRACTS it from a higher-is-better score,
    so a positive weight expresses "fewer fresh-fresh adjacencies, please".  It
    can reorder near-ties; it can never veto, and it is not a constraint.

    ``weights`` may name only :data:`VALIDATED_PENALTY_METRICS`; an unknown or a
    deliberately-not-adopted key (``rm3`` / ``rm4`` / ``rm5`` / ``rm6``) raises,
    because silently applying a refuted heuristic as a shaping term is exactly
    the failure mode the source report warns about — RM4 in particular made
    ``node_peak`` WORSE in the measured selection arms.

    ``None``, an empty mapping, or all-zero weights return an exact zero vector,
    which is what makes the default path byte-identical.
    """
    n = len(patterns)
    if not weights:
        return np.zeros(n, dtype=float)
    active = {}
    for name, weight in weights.items():
        key = str(name).lower()
        if key not in PENALTY_METRIC_FUNCS:
            raise ValueError(
                f"rule_penalty: {name!r} is not an adopted penalty metric; "
                f"valid keys are {VALIDATED_PENALTY_METRICS}.  RM3/RM4/RM5/RM6 "
                f"were measured and are REPORT-ONLY (RM3 is -0.885 collinear "
                f"with RM1, RM4 is null once conditioned on RM1i and its "
                f"selection arm made node_peak worse, RM6 is noise, RM5 is "
                f"same-map circular and is not a function of the pattern)."
            )
        w = float(weight)
        if w != 0.0:
            active[key] = w
    if not active:
        return np.zeros(n, dtype=float)
    out = np.zeros(n, dtype=float)
    for i, pattern in enumerate(patterns):
        total = 0.0
        for key, w in active.items():
            total += w * PENALTY_METRIC_FUNCS[key](pattern)
        out[i] = total
    return out
