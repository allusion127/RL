"""EDIT5 burnup-TRAJECTORY labels: access, burnup coordinate, anchor selection.

The high-resolution harvest (:func:`lpopt.search.verify._hires_from_equilibrium_result`)
stores EVERY EDIT5 burnup step under ``<record_id>__traj`` in ``maps.npz`` as
``float16[n_steps, 3, 9, 9]``, planes in :data:`lpopt.data.edit5.STEP_MAP_KEYS`
order ``(power, burnup, kinf)``.  The legacy 4-plane ``<record_id>`` stack keeps
only the two endpoints of that trajectory.  This module is the single place that
knows what the trajectory numbers *mean*, so the model, the tests and any
analysis read the same contract — the same role :mod:`lpopt.data.axial` plays for
the EDIT6 axial stack.

Measured contract (10,586 records, ``data/store/maps.npz`` @ 2026-07-30)
------------------------------------------------------------------------
* **The endpoints ARE the legacy map stack, exactly.**  Verified bit-for-bit on
  stored records:

  ==========================  ============================
  legacy ``<rid>`` plane      trajectory slice
  ==========================  ============================
  ``boc_power``  (index 0)    ``traj[0,  0]``
  ``eoc_power``  (index 1)    ``traj[-1, 0]``
  ``eoc_burnup`` (index 2)    ``traj[-1, 1]``
  ``eoc_kinf``   (index 3)    ``traj[-1, 2]``
  ==========================  ============================

  This is what makes the trajectory an *extension* of the existing map
  supervision rather than a second, differently-scaled label source: the map
  head's own z-score constants are the right ones for the intermediate steps too.
* **The step axis has NO stored EFPD** (same purge that motivated
  :mod:`lpopt.data.axial`'s two-anchor design).  But the trajectory carries its
  own burnup coordinate: the ``burnup`` plane is CUMULATIVE assembly burnup
  (fresh feed enters at 0, twice-burnt assemblies start at tens of GWd/t), so the
  slot-mean burnup is a monotone, deck-family-independent clock.
  :func:`cycle_burnup_fraction` normalises it to ``0`` at BOC and ``1`` at EOC —
  the *fraction of this cycle's burnup already accumulated*.  Nothing here needs
  an EFPD, and nothing assumes the EDIT5 ladder is uniform (it is not).
* **Step 0 and step 1 usually share a burnup** (MASTER prints a 0-EFPD step and
  an equilibrium-xenon step at the same burnup).  Anchor selection therefore
  resolves ties to the FIRST matching step, so anchor ``0.0`` is always the BOC
  snapshot that equals ``boc_power``.
* Only the 69 quarter-core SLOT positions carry data; the remaining 12 cells of
  the 9x9 quarter are **NaN** (the same layout ``edit5._quadrant`` writes, and
  the same convention the legacy 4-plane stack uses — which is why
  :meth:`lpopt.model.dataset_torch.LPDataset._maps` builds its mask with
  ``isfinite``).  Validity is therefore checked, and the burnup coordinate
  computed, over the SLOT positions only: a mean over all 81 cells would be NaN,
  and a finiteness check over all 81 cells would reject every record.

Why intermediate steps at all
-----------------------------
``cyclen`` is the ENDPOINT of the boron let-down trajectory: the cycle ends when
the critical boron concentration reaches zero.  A model supervised only at BOC
and EOC sees the two ends of that curve and must infer the whole path, which is
exactly where the residual cyclen / CBC variance lives.  Supervising the map head
at intermediate burnup fractions puts a label on the path itself.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..vendor.masterrl.domain import SLOTS

#: npz key suffix the high-resolution harvest writes.
TRAJ_SUFFIX = "__traj"

#: Per-step planes, in stored order (``lpopt.data.edit5.STEP_MAP_KEYS``).
STEP_PLANES: tuple[str, ...] = ("power", "burnup", "kinf")

#: Number of stored planes per step.
N_PLANES = len(STEP_PLANES)

#: Side of the 9x9 quarter map (``edit5._quadrant`` layout).
QUARTER = 9

#: Index of the cumulative-burnup plane (the burnup clock).
BURNUP_PLANE = STEP_PLANES.index("burnup")

#: Default cycle-burnup-fraction anchors for the trajectory supervision.
#:
#: ``0.0`` and ``1.0`` are deliberately included even though the legacy 4-plane
#: map already supervises those two states: they are the two fractions whose
#: label is *known independently* to equal an existing map plane (see the module
#: docstring table), so they anchor the burnup-fraction conditioning to the
#: supervision the champion already has.  Without them the conditioning input
#: would be free to mean anything.  The three interior fractions are the new
#: information.
DEFAULT_ANCHORS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: An anchor is MASKED OUT when the nearest available step's achieved fraction is
#: further than this from the requested one.  A trajectory with a coarse ladder
#: (or a truncated one) therefore contributes only the anchors it can honestly
#: support, instead of silently labelling "half way through the cycle" with a BOC
#: snapshot.  Real ladders have ~25-33 steps, i.e. a spacing of ~0.03-0.04, so
#: this rejects nothing on a healthy record.
MAX_ANCHOR_FRAC_ERROR = 0.10

#: Row/col of the 69 real quarter slots, for the slot-mean burnup clock.
_SLOT_ROWS = np.asarray([s.row for s in SLOTS], dtype=np.intp)
_SLOT_COLS = np.asarray([s.col for s in SLOTS], dtype=np.intp)


def traj_key(record_id: str) -> str:
    """npz key holding a record's EDIT5 burnup trajectory."""
    return f"{record_id}{TRAJ_SUFFIX}"


# --------------------------------------------------------------------------- #
# label access
# --------------------------------------------------------------------------- #
def load_traj(reader: Any, record_id: str) -> np.ndarray | None:
    """``(n_steps, 3, 9, 9)`` float64 trajectory for a record, or ``None``.

    Returns ``None`` — never raises — when the record has no trajectory label,
    the stored array has the wrong rank/shape, fewer than two steps, or a
    non-finite value **at a real slot**.  The 12 non-slot cells of the quarter
    are NaN by construction and are NOT a defect (module docstring).  Callers
    treat a missing trajectory exactly like a missing map label: mask it out and
    keep training on the other supervision.
    """
    arr = reader.maps(traj_key(str(record_id)))
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 4 or a.shape[1:] != (N_PLANES, QUARTER, QUARTER):
        return None
    if a.shape[0] < 2:
        return None
    if not np.isfinite(a[:, :, _SLOT_ROWS, _SLOT_COLS]).all():
        return None
    return a


def slot_mean_burnup(stack: np.ndarray) -> np.ndarray:
    """``(n_steps,)`` core-average (69-slot mean) cumulative assembly burnup.

    The mean is over the 69 REAL quarter slots; the 12 NaN cells of the 9x9
    quarter are excluded, so this is a core-average burnup — and finite, which a
    mean over all 81 cells would not be.
    """
    s = np.asarray(stack, dtype=np.float64)
    return s[:, BURNUP_PLANE][:, _SLOT_ROWS, _SLOT_COLS].mean(axis=1)


def cycle_burnup_fraction(stack: np.ndarray) -> np.ndarray | None:
    """``(n_steps,)`` fraction of THIS cycle's burnup accumulated, or ``None``.

    ``f_t = (b_t - b_0) / (b_T - b_0)`` on the slot-mean cumulative burnup, so
    ``f`` is ``0`` at BOC and ``1`` at EOC by construction, whatever prior-cycle
    burnup the reload carries and whatever the deck's EFPD ladder is.  The result
    is made non-decreasing (a float16 storage wobble can otherwise produce a
    1e-4 dip) and clipped to ``[0, 1]``.

    ``None`` when the cycle accumulates no burnup at all (``b_T <= b_0``), which
    is a degenerate/truncated trajectory rather than a usable label.
    """
    b = slot_mean_burnup(stack)
    span = float(b[-1] - b[0])
    if not np.isfinite(span) or span <= 0.0:
        return None
    f = (b - b[0]) / span
    return np.clip(np.maximum.accumulate(f), 0.0, 1.0)


def anchor_indices(fractions: np.ndarray,
                   anchors: Sequence[float] = DEFAULT_ANCHORS) -> np.ndarray:
    """``(A,)`` step index nearest to each requested cycle-burnup fraction.

    Ties resolve to the FIRST matching step (``argmin`` semantics), so anchor
    ``0.0`` picks the true BOC snapshot even though MASTER usually prints two
    steps at zero cycle burnup.
    """
    f = np.asarray(fractions, dtype=np.float64)
    want = np.asarray(list(anchors), dtype=np.float64)
    return np.abs(f[None, :] - want[:, None]).argmin(axis=1).astype(np.intp)


def anchor_planes(
    stack: np.ndarray,
    anchors: Sequence[float] = DEFAULT_ANCHORS,
    *,
    max_frac_error: float = MAX_ANCHOR_FRAC_ERROR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``((A,3,9,9) planes, (A,) achieved fractions, (A,) mask)`` — or ``None``.

    The returned fraction is the step's ACHIEVED cycle-burnup fraction, not the
    requested one: that is what the model is conditioned on, so the conditioning
    input never claims a burnup state the label does not hold.  An anchor whose
    nearest step is further than ``max_frac_error`` away is masked out (its
    planes are still returned, from the nearest step, but contribute no loss).

    The planes keep their NaN non-slot cells, exactly as the legacy 4-plane stack
    does, so a consumer derives the per-slot validity mask with ``isfinite`` —
    the identical rule ``LPDataset._maps`` uses.

    ``None`` when the trajectory has no usable burnup coordinate.
    """
    s = np.asarray(stack, dtype=np.float64)
    frac = cycle_burnup_fraction(s)
    if frac is None:
        return None
    idx = anchor_indices(frac, anchors)
    achieved = frac[idx]
    want = np.asarray(list(anchors), dtype=np.float64)
    mask = (np.abs(achieved - want) <= float(max_frac_error)).astype(np.float64)
    return s[idx], achieved, mask


def stack_anchor_traj(
    reader: Any,
    record_ids: Iterable[str],
    *,
    anchors: Sequence[float] = DEFAULT_ANCHORS,
    max_frac_error: float = MAX_ANCHOR_FRAC_ERROR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``((N,A,3,9,9) planes, (N,A) fractions, (N,A) mask)`` for record ids.

    Records with no (or unusable) trajectory get all-NaN planes, their requested
    fractions, and a zero mask — mirroring
    :meth:`lpopt.model.dataset_torch.LPDataset._maps`.
    """
    ids = [str(r) for r in record_ids]
    want = np.asarray(list(anchors), dtype=np.float64)
    a_n = len(want)
    planes = np.full((len(ids), a_n, N_PLANES, QUARTER, QUARTER), np.nan)
    fracs = np.tile(want, (len(ids), 1))
    mask = np.zeros((len(ids), a_n), dtype=np.float64)
    for i, rid in enumerate(ids):
        stack = load_traj(reader, rid)
        if stack is None:
            continue
        got = anchor_planes(stack, anchors, max_frac_error=max_frac_error)
        if got is None:
            continue
        planes[i], fracs[i], mask[i] = got
    return planes, fracs, mask


__all__ = [
    "BURNUP_PLANE",
    "DEFAULT_ANCHORS",
    "MAX_ANCHOR_FRAC_ERROR",
    "N_PLANES",
    "QUARTER",
    "STEP_PLANES",
    "TRAJ_SUFFIX",
    "anchor_indices",
    "anchor_planes",
    "cycle_burnup_fraction",
    "load_traj",
    "slot_mean_burnup",
    "stack_anchor_traj",
    "traj_key",
]
