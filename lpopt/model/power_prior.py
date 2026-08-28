"""Coarse-mesh diffusion POWER-MAP prior (hires bundle, design doc 20260725 §3).

Where :mod:`.physics_prior` gives a zero-dimensional reactivity balance for the
scalar ``cyclen``, this module gives the leading-order **spatial** answer: a
one-group coarse-mesh diffusion eigenvalue solve on the 241-assembly APR1400
core, driven by the same harvested reference k-inf(BU) curves.  Its output is a
69-slot relative assembly power map on exactly the grid ``edit5._quadrant``
writes the ``boc_power`` labels on, so ``label = prior + residual`` is an exact
round-trip and the network only has to learn what the leading-order solve
cannot express.

Why this and not more width
---------------------------
``data/reports/cyclen_nodepeak_resolution_20260725.md`` §3.6 measured the CNN's
spectral signature: the per-mode amplitude correlation falls monotonically
0.897 -> 0.593 as the spatial wavenumber rises, and the predicted/actual map
power ratio decays 1.00 -> 0.71 to Nyquist.  The design-doc probe measured this
prior's signature on the same protocol and found the **inverse**: its per-mode
correlation RISES 0.545 -> 0.861 with wavenumber, and its Nyquist power ratio is
0.862.  That is physically obvious in hindsight — the prior's high-wavenumber
content comes straight from the local k-inf contrast the shuffle creates, while
its low-wavenumber error comes from the crude lumped leakage.  The two error
structures are complementary, so the decomposition puts each component where it
is strong.  This changes what the parameters are asked to represent; it is not
the same lever as making the trunk wider.

The formula
-----------
For each of the 69 quarter slots ``j``, trace the shuffle chain to its fresh
origin (exactly as :class:`~.featurize.FeatureEncoder` and
:mod:`.physics_prior` do), read that origin's harvested curve and evaluate

    rho_j = assembly_rho(vec_j, (age_j - 1) * NOMINAL + NOMINAL / 2)   [pcm]
    kinf_j = 1 / (1 - rho_j / 1e5)

Mirror-expand the quarter to the full 17x17 core (241 fuel nodes) and solve the
one-group diffusion eigenvalue problem in the standard finite-difference form

    A phi = (1 / k) * (h^2 / M^2) * kinf * phi

with ``h`` the assembly pitch, ``A`` the 5-point Laplacian plus ``h^2 / M^2``
plus an extrapolated-boundary term ``1 / extrap`` on every face that looks out
of the fuel region.  The relative assembly power is ``P_j ~ kinf_j * phi_j``,
normalized to a full-core mean of 1 and folded back to the quarter.

Assumptions (all deliberate, all leading-order — the residual learns the rest)
1. **One group, uniform D and Sigma_a.**  Only ``M^2 = D / Sigma_a`` survives,
   and it is one global constant.  Spectral effects, reflector detail and the
   thermal/fast split are left to the residual.
2. **A-priori burn state.**  ``age_j`` and the nominal per-cycle burnup, never
   the record's own labels — the same leakage-safe input surface the featurizer
   uses, so the prior is computable for an unlabelled served pattern.
3. **Radial only.**  EDIT5 labels are assembly-wise with no axial dimension
   (design doc §1), so a radial prior matches the label resolution exactly.
4. **Two global fitted scalars.**  ``(M^2, extrap)`` are fit ONCE on the train
   split.  They are NOT per-cell, so the prior cannot launder per-cell label
   information into the model — within-cell metrics stay honest.
5. **Graceful degradation.**  A slot whose origin resolves to no harvested curve
   takes the population-median k-inf; a pattern where NO slot resolves returns
   the flat map (all ones), so the prior is always finite.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data.flatness import map_cov as _cov
from ..data.schema import unpack_pattern
from ..data.fuel_types import FuelLibrary
from ..vendor.masterrl.domain import SLOTS
from .featurize import NOMINAL_CYCLE_BURNUP_MWD_KG, RecordInputs
from .physics_prior import _resolve, _shared_encoder, assembly_rho

#: Artifact file name written next to the ensemble.
POWER_PRIOR_NAME = "power_prior.json"
#: Schema tag stored in the artifact.
POWER_PRIOR_SCHEMA = "power_prior_v1"

#: APR1400 assembly pitch [cm].
PITCH_CM = 20.78
#: Default migration area [cm^2] (design-doc grid sweep optimum).
DEFAULT_M2_CM2 = 80.0
#: Default extrapolated-boundary distance in node widths.
DEFAULT_EXTRAP = 2.0
#: Fallback k-inf for a slot whose origin type has no harvested curve.
FALLBACK_KINF = 1.0

_CORE = 17
_CORE_CENTER = 8
#: Number of quarter slots (== number of map cells the label carries).
N_SLOTS = len(SLOTS)

_QROW = np.array([s.row for s in SLOTS], dtype=np.intp)
_QCOL = np.array([s.col for s in SLOTS], dtype=np.intp)


def _core_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fuel_mask, slot_of, node_index)`` for the mirror-expanded 17x17 core."""
    mask = np.zeros((_CORE, _CORE), dtype=bool)
    slot_of = np.full((_CORE, _CORE), -1, dtype=np.intp)
    for slot in SLOTS:
        for dr in {slot.row, -slot.row}:
            for dc in {slot.col, -slot.col}:
                mask[_CORE_CENTER + dr, _CORE_CENTER + dc] = True
                slot_of[_CORE_CENTER + dr, _CORE_CENTER + dc] = slot.index
    node_index = np.full((_CORE, _CORE), -1, dtype=np.intp)
    node_index[mask] = np.arange(int(mask.sum()))
    return mask, slot_of, node_index


CORE_MASK, SLOT_OF, NODE_INDEX = _core_geometry()
#: 241 for the APR1400 (69 quarter slots at multiplicity 1/2/4).
N_NODES = int(CORE_MASK.sum())
#: Per-node quarter-slot id, in node order.
NODE_SLOT = SLOT_OF[CORE_MASK]

_OP_CACHE: dict[tuple[float, float], tuple[np.ndarray, float]] = {}


def _operator(m2_cm2: float, extrap: float) -> tuple[np.ndarray, float]:
    """``(A^-1, h^2 / M^2)`` for the one-group finite-difference operator."""
    key = (round(float(m2_cm2), 6), round(float(extrap), 6))
    hit = _OP_CACHE.get(key)
    if hit is not None:
        return hit
    gamma = PITCH_CM ** 2 / max(float(m2_cm2), 1.0e-6)
    ex = max(float(extrap), 1.0e-3)
    a = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for r in range(_CORE):
        for c in range(_CORE):
            if not CORE_MASK[r, c]:
                continue
            i = int(NODE_INDEX[r, c])
            diag = gamma
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < _CORE and 0 <= cc < _CORE and CORE_MASK[rr, cc]:
                    a[i, int(NODE_INDEX[rr, cc])] -= 1.0
                    diag += 1.0
                else:
                    diag += 1.0 / ex          # extrapolated boundary
            a[i, i] = diag
    out = (np.linalg.inv(a), gamma)
    if len(_OP_CACHE) < 64:
        _OP_CACHE[key] = out
    return out


def power_maps_from_kinf(kinf_q: np.ndarray, *, m2_cm2: float = DEFAULT_M2_CM2,
                         extrap: float = DEFAULT_EXTRAP,
                         max_iter: int = 300, tol: float = 1.0e-11) -> np.ndarray:
    """Batched power iteration.  ``[N, 69]`` k-inf -> ``[N, 69]`` relative power.

    Vectorised over the batch (one dense solve reused for every row), so the
    whole 50k-row store costs a handful of GEMMs.
    """
    kq = np.atleast_2d(np.asarray(kinf_q, dtype=np.float64))
    ainv, gamma = _operator(m2_cm2, extrap)
    kf = kq[:, NODE_SLOT].T                      # [N_NODES, N]
    phi = np.ones_like(kf)
    for _ in range(int(max_iter)):
        nxt = ainv @ (gamma * kf * phi)
        nxt /= nxt.mean(axis=0, keepdims=True)
        if float(np.max(np.abs(nxt - phi))) < tol:
            phi = nxt
            break
        phi = nxt
    p = kf * phi
    p /= p.mean(axis=0, keepdims=True)           # full-core mean == 1
    # Fold back to the quarter: every mirror image of a slot carries the same
    # value, so taking the first occurrence is exact (no averaging).
    out = np.ones((kq.shape[0], N_SLOTS), dtype=np.float64)
    first = np.zeros(N_SLOTS, dtype=np.intp)
    for j in range(N_SLOTS):
        first[j] = int(np.flatnonzero(NODE_SLOT == j)[0])
    out[:, :] = p[first, :].T
    return out


def _neighbour_gather() -> tuple[np.ndarray, np.ndarray]:
    """``(neigh_slot[69,4], neigh_count[69])`` over the mirror-expanded core.

    For each quarter slot, the quarter ids of its four full-core face neighbours
    (a slot on a symmetry axis sees its own mirror image, which is correct — the
    physical neighbour across the axis IS that assembly).  Out-of-core faces are
    marked ``-1`` and excluded from the mean.
    """
    neigh = np.full((N_SLOTS, 4), -1, dtype=np.intp)
    for slot in SLOTS:
        r, c = _CORE_CENTER + slot.row, _CORE_CENTER + slot.col
        for k, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            rr, cc = r + dr, c + dc
            if 0 <= rr < _CORE and 0 <= cc < _CORE and CORE_MASK[rr, cc]:
                neigh[slot.index, k] = SLOT_OF[rr, cc]
    return neigh, (neigh >= 0).sum(axis=1).astype(np.float64)


NEIGH_SLOT, NEIGH_COUNT = _neighbour_gather()


def neighbour_mean(values_q: np.ndarray) -> np.ndarray:
    """Face-neighbour mean of a ``[..., 69]`` per-slot field, same shape out.

    This is the local-contrast operator behind the ``*_contrast`` channels: a
    slot minus this quantity is exactly the high-spatial-frequency part of the
    field, which is the band the 20260725 report measured the CNN to attenuate.
    """
    v = np.asarray(values_q, dtype=np.float64)
    gathered = v[..., NEIGH_SLOT]                       # [..., 69, 4]
    valid = (NEIGH_SLOT >= 0).astype(np.float64)
    total = (gathered * valid).sum(axis=-1)
    return total / np.maximum(NEIGH_COUNT, 1.0)


def kinf_quarter(inputs: RecordInputs, fuel: FuelLibrary, *,
                 encoder: Any | None = None,
                 vec_cache: dict | None = None,
                 bu_per_cycle: float | None = None) -> np.ndarray:
    """Per-slot a-priori k-inf ``[69]`` for one pattern (pure function of inputs).

    ``bu_per_cycle`` overrides the flat :data:`~.featurize.NOMINAL_CYCLE_BURNUP_MWD_KG`
    with the a-priori per-cycle burnup of this record's ``(library_id, feed)``
    regime (:func:`~.featurize.regime_cycle_burnup`, cond_v6b).  ``None`` -- the
    default and what every v2..v6 caller passes -- reproduces the legacy arithmetic
    bit-for-bit, which is what keeps a v6 encoder byte-identical.

    The override is still a-priori: it is a hard-coded constant selected by two
    fields that are already model inputs, so nothing here reads a label.
    """
    enc = encoder or _shared_encoder()
    items = unpack_pattern(inputs.pattern).items
    age, origin, _ = enc._trace_chain(items)
    b_cycle = (NOMINAL_CYCLE_BURNUP_MWD_KG if bu_per_cycle is None
               else float(bu_per_cycle))
    out = np.full(N_SLOTS, np.nan, dtype=np.float64)
    for slot in SLOTS:
        j = slot.index
        vec = _resolve(fuel, items[origin[j]].batch, inputs.library_id, vec_cache)
        bu = (age[j] - 1) * b_cycle + 0.5 * b_cycle
        rho, _slope = assembly_rho(vec, bu)
        if math.isfinite(rho) and rho < 1.0e5:
            out[j] = 1.0 / (1.0 - rho / 1.0e5)
    finite = np.isfinite(out)
    if not finite.any():
        out[:] = FALLBACK_KINF
    elif not finite.all():
        out[~finite] = float(np.median(out[finite]))
    return out


def kinf_quarter_batch(rows: Any, fuel: FuelLibrary, *,
                       regime_burnup: bool = False) -> np.ndarray:
    """``[N, 69]`` a-priori k-inf for every row of a store frame.

    ``regime_burnup=True`` evaluates each row at its own
    :func:`~.featurize.regime_cycle_burnup`, which is what a cond_v6b run needs so
    that ``(M^2, extrap)`` is FIT on the same burn state the encoder will SERVE on.
    Default ``False`` is the legacy arithmetic, bit-for-bit.
    """
    from .featurize import regime_cycle_burnup

    enc = _shared_encoder()
    cache: dict = {}
    it = rows.iterrows() if hasattr(rows, "iterrows") else enumerate(rows)
    out = []
    for _, row in it:
        inp = RecordInputs.coerce(row)
        out.append(kinf_quarter(
            inp, fuel, encoder=enc, vec_cache=cache,
            bu_per_cycle=(regime_cycle_burnup(inp.library_id, inp.feed)
                          if regime_burnup else None)))
    return (np.stack(out) if out
            else np.zeros((0, N_SLOTS), dtype=np.float64))


# --------------------------------------------------------------------------- #
# fitted artifact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PowerPrior:
    """The fitted ``(M^2, extrap)`` pair — two GLOBAL scalars, never per-cell."""

    m2_cm2: float = DEFAULT_M2_CM2
    extrap: float = DEFAULT_EXTRAP
    n_fit: int = 0
    within_cell_rho: float = float("nan")
    split: str | None = None
    schema: str = POWER_PRIOR_SCHEMA
    grid: tuple = field(default=())

    def maps_for_kinf(self, kinf_q: np.ndarray) -> np.ndarray:
        return power_maps_from_kinf(kinf_q, m2_cm2=self.m2_cm2, extrap=self.extrap)

    def maps_for_rows(self, rows: Any, fuel: FuelLibrary) -> np.ndarray:
        return self.maps_for_kinf(kinf_quarter_batch(rows, fuel))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "m2_cm2": float(self.m2_cm2),
            "extrap": float(self.extrap), "n_fit": int(self.n_fit),
            "within_cell_rho": float(self.within_cell_rho),
            "split": self.split, "pitch_cm": PITCH_CM,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PowerPrior":
        return cls(
            m2_cm2=float(payload.get("m2_cm2", DEFAULT_M2_CM2)),
            extrap=float(payload.get("extrap", DEFAULT_EXTRAP)),
            n_fit=int(payload.get("n_fit", 0)),
            within_cell_rho=float(payload.get("within_cell_rho", float("nan"))),
            split=payload.get("split"),
            schema=str(payload.get("schema", POWER_PRIOR_SCHEMA)))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "PowerPrior":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


#: Grid searched by :func:`fit_power_prior` (design-doc sweep envelope).
DEFAULT_GRID: tuple[tuple[float, float], ...] = tuple(
    (m2, ex) for m2 in (30.0, 45.0, 60.0, 80.0, 110.0, 150.0)
    for ex in (0.5, 1.0, 2.0))


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def fit_power_prior(kinf_q: np.ndarray, true_maps: np.ndarray, cells: np.ndarray,
                    *, split: str | None = None,
                    grid: tuple[tuple[float, float], ...] = DEFAULT_GRID,
                    min_cell: int = 8) -> PowerPrior:
    """Pick ``(M^2, extrap)`` by median WITHIN-CELL rank correlation of map CoV.

    ``kinf_q`` / ``true_maps`` are ``[N, 69]`` and ``cells`` is ``[N]`` cell keys.
    The caller must pass **train-split rows only** — this function cannot know
    the split, exactly like :func:`~.physics_prior.fit_cyclen_prior`.

    Fitting on CoV rather than peak is deliberate: the 20260725 report measured
    CoV to be the better-conditioned flatness descriptor (within-cell rho 0.804
    vs 0.650 for peak) because it is a second moment over 69 slots rather than an
    extreme-value statistic.

    The CoV is the canonical multiplicity-weighted one (:mod:`..data.flatness`);
    the fit is a within-cell RANK correlation, whose weighted/unweighted rank
    agreement was measured at 0.988, so this is a definition unification rather
    than a change of what the grid search optimizes.
    """
    kq = np.asarray(kinf_q, dtype=np.float64)
    tm = np.asarray(true_maps, dtype=np.float64)
    cl = np.asarray(cells)
    true_cov = _cov(tm)
    ok = np.isfinite(true_cov)
    if int(ok.sum()) < max(min_cell, 3):
        return PowerPrior(n_fit=int(ok.sum()), split=split)
    kq, tm, cl, true_cov = kq[ok], tm[ok], cl[ok], true_cov[ok]
    groups = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    groups = [g for g in groups if len(g) >= int(min_cell)]

    best: tuple[float, float, float] = (-np.inf, DEFAULT_M2_CM2, DEFAULT_EXTRAP)
    for m2, ex in grid:
        pcov = _cov(power_maps_from_kinf(kq, m2_cm2=m2, extrap=ex))
        if groups:
            rhos = [_spearman(pcov[g], true_cov[g]) for g in groups]
            score = float(np.nanmedian(rhos)) if np.isfinite(rhos).any() else -np.inf
        else:                       # no cell reaches min_cell: fall back to global
            score = _spearman(pcov, true_cov)
        if np.isfinite(score) and score > best[0]:
            best = (score, float(m2), float(ex))
    return PowerPrior(m2_cm2=best[1], extrap=best[2], n_fit=int(len(true_cov)),
                      within_cell_rho=float(best[0]), split=split)


__all__ = [
    "POWER_PRIOR_NAME",
    "POWER_PRIOR_SCHEMA",
    "DEFAULT_M2_CM2",
    "DEFAULT_EXTRAP",
    "DEFAULT_GRID",
    "N_SLOTS",
    "N_NODES",
    "PowerPrior",
    "fit_power_prior",
    "kinf_quarter",
    "kinf_quarter_batch",
    "neighbour_mean",
    "power_maps_from_kinf",
]
