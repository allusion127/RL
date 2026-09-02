"""Canonical core-flatness scalars from a harvested EDIT5 assembly map.

This module is the **single definition** of ``node_peak`` and ``map_cov``
(flatness-first program 20260725 §1.1).  Everything that needs either scalar —
the harvest path (:mod:`..search.verify`), the record backfill
(:mod:`..tools.backfill_flatness`), A/B scoring (:mod:`..model.ab_score`), the
power-map prior fit (:mod:`..model.power_prior`) — imports it from here.  A
second local copy is what produced the two incompatible numbers the program
report had to reconcile, so there is exactly one.

The definition (multiplicity-WEIGHTED)
--------------------------------------
A record's map is a 69-slot **quarter** core.  Each quarter slot stands for
``SLOTS[i].multiplicity`` physical assemblies (52 slots x4, 16 slots x2, the
centre slot x1; ``sum == 241``, the full APR1400 core), so a plain 69-slot mean
is not the core average::

    w_i       = SLOTS[i].multiplicity
    p_bar     = sum(w_i p_i) / sum(w_i)                    # == 1.0000 in practice
    node_peak = max_i p_i          # BOC assembly radial peak (assembly 2-D)
    map_cov   = sqrt(sum(w_i (p_i - p_bar)^2) / sum(w_i)) / p_bar

NOT F_xy
--------
``node_peak`` is the **BOC assembly radial peak (assembly-level 2-D, the FRA
family)**.  It is **NOT MASTER's FXYP (pin planar)**, which is the ``f_xy``
record column parsed from ``MAS_OUT`` by :mod:`.fxy`.  Two different physical
quantities: ``node_peak`` never resolves pins and is read at BOC only, while
FXYP is a within-plane PIN maximum over every depletion step.  Measured
corr(node_peak, f_xy) is 0.735-0.854 with residual sd 0.057-0.064 — the same
order as the 1.65 gate itself — so ``node_peak`` may NOT be used as an F_xy
surrogate in any gate (design 20260829 §1.1 / §3.4.4 / §3.10).  The earlier
"== F_xy" claim in this docstring was a naming collision, corrected 2026-08-29.

The weighting is not cosmetic.  Measured over ``maps.npz`` the weighted mean is
1.0000 (0.9999-1.0001) — i.e. the harvested map is ALREADY normalized to the
core average, so ``node_peak = nanmax`` is exactly the assembly radial peaking
factor.  The UNweighted 69-slot mean has median 1.0233 and ranges
0.983-1.088, which means an unweighted CoV divides every record by a different,
record-dependent denominator and cannot be reported as a physical quantity.

Map layouts accepted
--------------------
* ``(69,)``            — slot values already gathered;
* ``(N, 69)``          — a batch of the above;
* ``(9, 9)``           — one SE-quadrant plane (NaN off-slot);
* ``(4, 9, 9)``        — the legacy :data:`.edit5.MAP_KEYS` stack, BOC
  assembly power is channel 0;
* ``(n_steps, 3, 9, 9)`` — the high-resolution trajectory stack
  (:func:`.edit5.stack_step_maps`); BOC is step 0, ``power`` is channel 0.

NaN slots are dropped from BOTH the mean and the variance (their weight is
dropped with them), so a partially parsed map still yields an honest scalar; a
map with no finite slot yields NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..vendor.masterrl.domain import SLOTS

#: Number of independent quarter-core slots (== map cells a record carries).
N_SLOTS = len(SLOTS)

#: Row / column of each quarter slot inside the ``(9, 9)`` SE-quadrant plane.
SLOT_ROWS = np.array([s.row for s in SLOTS], dtype=np.intp)
SLOT_COLS = np.array([s.col for s in SLOTS], dtype=np.intp)

#: ``w_i`` — full-core assemblies represented by each quarter slot (4x52, 2x16,
#: 1x1).  This is the whole point of the module; see the header.
SLOT_WEIGHTS = np.array([s.multiplicity for s in SLOTS], dtype=np.float64)

#: ``sum(w_i)`` == 241, the APR1400 full-core assembly count.
TOTAL_WEIGHT = float(SLOT_WEIGHTS.sum())

#: Radius (in assembly pitches) of each quarter slot from the core centre.
SLOT_RADIUS = np.array([s.radius for s in SLOTS], dtype=np.float64)

#: BOC assembly power lives at channel 0 of the legacy ``(4, 9, 9)`` stack and at
#: ``[step 0][channel 0]`` of the ``(n_steps, 3, 9, 9)`` trajectory stack.
BOC_CHANNEL = 0
BOC_STEP = 0

#: Guard against a silently mis-shaped ``p_bar`` divisor.
_MIN_MEAN = 1.0e-12

if int(SLOT_WEIGHTS.sum()) != 241:  # pragma: no cover - vendor geometry guard
    raise AssertionError("quarter-slot multiplicities do not sum to 241")


# --------------------------------------------------------------------------- #
# geometry for the report-only diagnostics
# --------------------------------------------------------------------------- #
_CORE = 17
_CORE_CENTER = 8


def _core_geometry() -> tuple[np.ndarray, np.ndarray]:
    """``(fuel_mask, slot_of)`` for the mirror-expanded 17x17 full core."""
    mask = np.zeros((_CORE, _CORE), dtype=bool)
    slot_of = np.full((_CORE, _CORE), -1, dtype=np.intp)
    for slot in SLOTS:
        for dr in {slot.row, -slot.row}:
            for dc in {slot.col, -slot.col}:
                mask[_CORE_CENTER + dr, _CORE_CENTER + dc] = True
                slot_of[_CORE_CENTER + dr, _CORE_CENTER + dc] = slot.index
    return mask, slot_of


_CORE_MASK, _SLOT_OF = _core_geometry()


def _neighbour_gather() -> tuple[np.ndarray, np.ndarray]:
    """``(neigh_slot[69, 4], in_core[69, 4])`` face neighbours in the FULL core.

    A slot on a symmetry axis sees its own mirror image, which is physically
    right — the assembly across the axis IS that assembly.  Out-of-core faces are
    ``-1`` and masked out.
    """
    neigh = np.full((N_SLOTS, 4), -1, dtype=np.intp)
    for slot in SLOTS:
        r, c = _CORE_CENTER + slot.row, _CORE_CENTER + slot.col
        for k, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            rr, cc = r + dr, c + dc
            if 0 <= rr < _CORE and 0 <= cc < _CORE and _CORE_MASK[rr, cc]:
                neigh[slot.index, k] = _SLOT_OF[rr, cc]
    return neigh, neigh >= 0


NEIGH_SLOT, NEIGH_VALID = _neighbour_gather()

#: Peripheral slots — a face that looks OUT of the fuel region.  These are the
#: assemblies whose power drives RPV beltline fast fluence and baffle heating,
#: which radial flattening raises (program §2.3, report-only).
PERIPHERY_MASK = ~NEIGH_VALID.all(axis=1)


# --------------------------------------------------------------------------- #
# shape normalization
# --------------------------------------------------------------------------- #
def slot_values(maps: Any) -> np.ndarray | None:
    """Gather BOC assembly power as ``[N, 69]`` float64 (``None`` if absent).

    Accepts every layout the store carries (see the module header).  ``None`` in
    -> ``None`` out; an unrecognised shape raises :class:`ValueError` (the caller
    that must never raise is :func:`record_flatness`).
    """
    if maps is None:
        return None
    arr = np.asarray(maps, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] != N_SLOTS:
            raise ValueError(f"1-D map must have {N_SLOTS} slots, got {arr.shape}")
        return arr[None, :]
    if arr.ndim == 2:
        if arr.shape[1] == N_SLOTS:
            return arr
        if arr.shape == (9, 9):
            return arr[None, SLOT_ROWS, SLOT_COLS]
        raise ValueError(f"unrecognised 2-D map shape {arr.shape}")
    if arr.ndim == 3:
        # legacy (4, 9, 9) MAP_KEYS stack — BOC assembly power is channel 0
        if arr.shape[1:] != (9, 9):
            raise ValueError(f"unrecognised 3-D map shape {arr.shape}")
        return arr[BOC_CHANNEL][None, SLOT_ROWS, SLOT_COLS]
    if arr.ndim == 4:
        # (n_steps, 3, 9, 9) trajectory — BOC is the first step, power channel 0
        if arr.shape[2:] != (9, 9):
            raise ValueError(f"unrecognised 4-D map shape {arr.shape}")
        return arr[BOC_STEP, BOC_CHANNEL][None, SLOT_ROWS, SLOT_COLS]
    raise ValueError(f"unrecognised map ndim {arr.ndim}")


# --------------------------------------------------------------------------- #
# the canonical scalars (vectorized over [N, 69])
# --------------------------------------------------------------------------- #
def _masked(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(v[N,69], finite[N,69], w[N,69])`` with NaN slots carrying zero weight."""
    v = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if v.ndim != 2 or v.shape[1] != N_SLOTS:
        raise ValueError(f"slot values must be [N, {N_SLOTS}], got {v.shape}")
    ok = np.isfinite(v)
    return v, ok, np.where(ok, SLOT_WEIGHTS, 0.0)


def weighted_mean(values: np.ndarray) -> np.ndarray:
    """``p_bar[N]`` — multiplicity-weighted core mean over the finite slots."""
    v, ok, w = _masked(values)
    wsum = w.sum(axis=1)
    out = np.full(v.shape[0], np.nan, dtype=np.float64)
    good = wsum > 0.0
    out[good] = (w * np.where(ok, v, 0.0)).sum(axis=1)[good] / wsum[good]
    return out


def node_peak(values: np.ndarray) -> np.ndarray:
    """``node_peak[N] = max_i p_i`` — the BOC ASSEMBLY radial peaking factor.

    Assembly-level 2-D (FRA family), NOT MASTER's FXYP (pin planar) — see the
    module docstring; the pin-planar value is the ``f_xy`` column (:mod:`.fxy`).

    The maximum is a per-assembly extreme, so the multiplicity weights do not
    enter it; they only fix the mean the map is normalized to (1.0000), which is
    what makes the raw max a peaking FACTOR at all.
    """
    v, ok, _w = _masked(values)
    any_ok = ok.any(axis=1)
    out = np.full(v.shape[0], np.nan, dtype=np.float64)
    out[any_ok] = np.where(ok, v, -np.inf).max(axis=1)[any_ok]
    return out


def map_cov(values: np.ndarray) -> np.ndarray:
    """``map_cov[N]`` — multiplicity-weighted CoV of the assembly power map."""
    v, ok, w = _masked(values)
    wsum = w.sum(axis=1)
    out = np.full(v.shape[0], np.nan, dtype=np.float64)
    good = wsum > 0.0
    if not good.any():
        return out
    mean = weighted_mean(v)
    dev = np.where(ok, v - mean[:, None], 0.0)
    var = (w * dev * dev).sum(axis=1)
    use = good & (np.abs(mean) > _MIN_MEAN)
    out[use] = np.sqrt(var[use] / wsum[use]) / mean[use]
    return out


def flatness_pair(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(node_peak[N], map_cov[N])`` — the two objective scalars in one pass."""
    return node_peak(values), map_cov(values)


# --------------------------------------------------------------------------- #
# report-only diagnostics (program §2.3 / §4.1 — NEVER in the objective scalar)
# --------------------------------------------------------------------------- #
def adjacent_gradient(values: np.ndarray) -> np.ndarray:
    """``|p_i - p_j|`` over every full-core face pair, as ``[N, 69, 4]``.

    Kept here because §4.1 MEASURED the gradient term and rejected it for the
    objective (partial rho with F_r given ``node_peak`` is 0.043 vs 0.107 for
    ``map_cov``); it stays as a five-line reporting diagnostic so the decision can
    be revisited when pin-level labels exist.  Out-of-core faces are NaN.
    """
    v, _ok, _w = _masked(values)
    gathered = v[:, NEIGH_SLOT]                       # [N, 69, 4]
    grad = np.abs(gathered - v[:, :, None])
    return np.where(NEIGH_VALID[None, :, :], grad, np.nan)


def gradient_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(max[N], p90[N])`` of :func:`adjacent_gradient` (report-only)."""
    ag = adjacent_gradient(values)
    g = ag.reshape(ag.shape[0], -1)
    out_max = np.full(g.shape[0], np.nan, dtype=np.float64)
    out_p90 = np.full(g.shape[0], np.nan, dtype=np.float64)
    for i, row in enumerate(g):
        finite = row[np.isfinite(row)]
        if finite.size:
            out_max[i] = finite.max()
            out_p90[i] = np.percentile(finite, 90.0)
    return out_max, out_p90


def periphery_mean(values: np.ndarray) -> np.ndarray:
    """Weighted mean power of the OUTERMOST ring (report-only, program §2.3).

    Radial flattening buys its peak reduction by raising the periphery, which
    raises RPV beltline fast flux and baffle heating.  Free from the map we
    already harvest, so it is recorded rather than discovered later.
    """
    v, ok, w = _masked(values)
    wm = np.where(PERIPHERY_MASK[None, :], w, 0.0)
    wsum = wm.sum(axis=1)
    out = np.full(v.shape[0], np.nan, dtype=np.float64)
    good = wsum > 0.0
    out[good] = (wm * np.where(ok, v, 0.0)).sum(axis=1)[good] / wsum[good]
    return out


def radial_weighted_power(values: np.ndarray) -> np.ndarray:
    """``sum(w_i r_i p_i) / sum(w_i r_i)`` — radius-weighted power (leakage proxy)."""
    v, ok, w = _masked(values)
    wr = w * SLOT_RADIUS
    wsum = wr.sum(axis=1)
    out = np.full(v.shape[0], np.nan, dtype=np.float64)
    good = wsum > 0.0
    out[good] = (wr * np.where(ok, v, 0.0)).sum(axis=1)[good] / wsum[good]
    return out


# --------------------------------------------------------------------------- #
# single-record entry points
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlatnessScalars:
    """The canonical scalars of ONE map (``None`` when the map is unusable)."""

    node_peak: float | None
    map_cov: float | None
    p_bar: float | None
    n_finite: int

    def as_columns(self) -> dict[str, float | None]:
        """The two record columns, in schema order."""
        return {"node_peak": self.node_peak, "map_cov": self.map_cov}


#: The scalars of an absent / unusable map.
EMPTY = FlatnessScalars(node_peak=None, map_cov=None, p_bar=None, n_finite=0)


def _clean(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def flatness_scalars(maps: Any) -> FlatnessScalars:
    """:class:`FlatnessScalars` of one map stack (any accepted layout).

    Strict about the layout — an unrecognised shape raises.  Use
    :func:`record_flatness` on the harvest path, where nothing may raise.
    """
    vals = slot_values(maps)
    if vals is None:
        return EMPTY
    if vals.shape[0] != 1:
        raise ValueError(f"expected a single map, got a batch of {vals.shape[0]}")
    peak, cov = flatness_pair(vals)
    mean = weighted_mean(vals)
    return FlatnessScalars(
        node_peak=_clean(peak[0]),
        map_cov=_clean(cov[0]),
        p_bar=_clean(mean[0]),
        n_finite=int(np.isfinite(vals[0]).sum()),
    )


def record_flatness(maps: Any) -> tuple[float | None, float | None]:
    """``(node_peak, map_cov)`` for the record columns — NEVER raises.

    The harvest path calls this on whatever the EDIT5 parse produced.  A missing,
    empty, mis-shaped or all-NaN map yields ``(None, None)``: a flatness label is
    optional, the F_r / cyclen labels of the same record are not, and a wave must
    never die because a map came back odd.
    """
    try:
        s = flatness_scalars(maps)
    except Exception:  # noqa: BLE001 - the columns are optional, the wave is not
        return (None, None)
    return (s.node_peak, s.map_cov)


__all__ = [
    "BOC_CHANNEL",
    "BOC_STEP",
    "EMPTY",
    "FlatnessScalars",
    "NEIGH_SLOT",
    "NEIGH_VALID",
    "N_SLOTS",
    "PERIPHERY_MASK",
    "SLOT_COLS",
    "SLOT_RADIUS",
    "SLOT_ROWS",
    "SLOT_WEIGHTS",
    "TOTAL_WEIGHT",
    "adjacent_gradient",
    "flatness_pair",
    "flatness_scalars",
    "gradient_stats",
    "map_cov",
    "node_peak",
    "periphery_mean",
    "radial_weighted_power",
    "record_flatness",
    "slot_values",
    "weighted_mean",
]
