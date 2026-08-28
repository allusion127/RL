"""Axial (EDIT 6) power-shape labels: parsing, derived scalars, shape basis.

The high-resolution harvest (:func:`lpopt.search.verify._hires_from_equilibrium_result`)
stores the EDIT 6 1-D power distribution under ``<record_id>__axial`` in
``maps.npz`` as ``float16[n_steps, 25]``.  This module is the single place that
knows what those numbers *mean*, so the model, the tests and any analysis all
read the same contract.

Measured contract (894 records, ``data/store/maps.npz`` @ 2026-07-25)
---------------------------------------------------------------------
* **Orientation is BOTTOM -> TOP.**  Not an assumption: ``MAS_SUM`` prints
  ``POWER(BOTTOM  --->  TOP)`` directly above the EDIT 6 table, and the columns
  are nodes 2..26 — the 25 *fuel* nodes, with nodes 1 and 27 (the axial
  reflectors) excluded.  ``profile[0]`` is the bottom-most fuel node.
* **Normalisation is core-average = 1.**  Every step row sums to exactly
  ``n_planes`` (max deviation over all 894 x ~27 rows: 1.95e-4, which is the
  float16 storage quantum).  So a plane value IS the axial power peaking of
  that plane, and :func:`axial_peaking_factor` is just ``max(profile)``.
* **The physical quantity is relative nodal power**, one value per axial node,
  per burnup step.
* **The step axis has NO stored EFPD.**  The burnup ladder is deck-family
  dependent (the ga80 ``fill_5*`` campaigns and the paramA ``fill_6.25-6.5``
  campaigns disagree by one step at the same cycle length), so a step index
  cannot be turned into an EFPD without the raw ``MAS_SUM`` — which the purge
  deletes.  Only the two *index-addressable* anchors are unambiguous:
  ``profile[0]`` (BOC) and ``profile[-1]`` (EOC).  That is why :data:`ANCHORS`
  has exactly those two, and it costs almost nothing: the cycle-maximum F_z is
  attained at step 0 or 1 on 94.6% of records, and ``max_step(F_z) - F_z(BOC)``
  has mean 0.0013 / p95 0.0033 (vs a within-cell F_z spread of ~0.020).

Axial offset
------------
:func:`axial_offset` reproduces the MASTER EDIT 3 ``AO`` column **exactly** from
the EDIT 6 profile: the centre node is split half-and-half between the two
halves.  Verified per-step against ``MAS_SUM`` EDIT 3 on stored runs — max
absolute deviation 6e-5 (float16 quantum); the naive "drop the centre node"
variant is wrong by up to 3e-3.  ``ASI == -AO`` by the convention already used
in :mod:`lpopt.search.acquisition`.

Why the profile and not just |AO|
---------------------------------
|AO| is a first moment and cannot distinguish a saddle from a double hump.
Measured on the same 894 records: the EOC saddle depth (peak minus centre-node
power) correlates with ``|AO|`` at only **-0.58** — and with the *wrong* sign —
while it correlates with ``F_z`` at **+0.985**.  The profile is the primitive;
|AO| is a lossy projection of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

#: npz key suffix the high-resolution harvest writes.
AXIAL_SUFFIX = "__axial"

#: Number of axial fuel nodes MASTER prints in EDIT 6 for the APR1400 deck.
#: The parser (:func:`lpopt.data.edit5.stack_axial`) reads the width from the
#: data, so this is the *expected* width, never a hard-coded slice.
N_PLANES = 25

#: Index-addressable burnup anchors, in head-output order.  See the module
#: docstring for why these two and only these two.
ANCHORS: tuple[str, ...] = ("boc", "eoc")


def axial_key(record_id: str) -> str:
    """npz key holding a record's axial stack."""
    return f"{record_id}{AXIAL_SUFFIX}"


# --------------------------------------------------------------------------- #
# label access
# --------------------------------------------------------------------------- #
def load_axial(reader: Any, record_id: str,
               n_planes: int = N_PLANES) -> np.ndarray | None:
    """``(n_steps, n_planes)`` float64 stack for a record, or ``None``.

    Returns ``None`` — never raises — when the record has no axial label, the
    stored array has the wrong rank/width, or any value is non-finite.  Callers
    treat a missing axial label exactly like a missing map label: mask it out and
    keep training on the other supervision.
    """
    arr = reader.maps(axial_key(str(record_id)))
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] != int(n_planes):
        return None
    if not np.isfinite(a).all():
        return None
    return a


def anchor_profiles(stack: np.ndarray,
                    anchors: Sequence[str] = ANCHORS) -> np.ndarray:
    """``(len(anchors), n_planes)`` profiles at the named burnup anchors.

    ``"boc"`` is step 0 and ``"eoc"`` is step -1 — the only two step positions
    that mean the same thing across deck families (module docstring).
    """
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 2:
        raise ValueError(f"axial stack must be 2-D, got shape {stack.shape}")
    out = []
    for name in anchors:
        if name == "boc":
            out.append(stack[0])
        elif name == "eoc":
            out.append(stack[-1])
        else:
            raise ValueError(
                f"unknown axial anchor {name!r}; have {ANCHORS}")
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------- #
# derived scalars (pure functions of a profile; work on any trailing-axis shape)
# --------------------------------------------------------------------------- #
def axial_peaking_factor(profile: np.ndarray) -> np.ndarray:
    """F_z — axial peaking factor: ``max(plane) / mean(plane)``.

    Stored profiles are already core-average-normalised, so this is ``max()`` to
    within the float16 quantum; the explicit division makes the function correct
    for a *predicted* profile too, where the normalisation is only structural.
    """
    p = np.asarray(profile, dtype=np.float64)
    return p.max(axis=-1) / p.mean(axis=-1)


def axial_offset(profile: np.ndarray) -> np.ndarray:
    """AO — axial offset, MASTER EDIT 3 convention (centre node split in half).

    ``AO = (P_top - P_bottom) / P_total`` where, for an odd plane count, the
    centre node contributes half to each side.  Reproduces the EDIT 3 ``AO``
    column from the EDIT 6 profile to 6e-5 (see module docstring).
    """
    p = np.asarray(profile, dtype=np.float64)
    n = p.shape[-1]
    h = n // 2
    bot = p[..., :h].sum(axis=-1)
    top = p[..., n - h:].sum(axis=-1)          # odd n -> skips the centre node
    return (top - bot) / p.sum(axis=-1)


def axial_shape_index(profile: np.ndarray) -> np.ndarray:
    """ASI — axial shape index.  ``ASI == -AO`` (acquisition.py convention)."""
    return -axial_offset(profile)


def saddle_depth(profile: np.ndarray) -> np.ndarray:
    """Peak-minus-centre power: the saddle/double-hump depth |AO| cannot see.

    Zero for a single-humped (centre-peaked) profile, positive and growing as the
    shape flattens into a saddle and then into two separated humps.
    """
    p = np.asarray(profile, dtype=np.float64)
    centre = p[..., p.shape[-1] // 2]
    return p.max(axis=-1) - centre


def derived_metrics(profile: np.ndarray) -> dict[str, np.ndarray]:
    """All derived axial scalars of a profile, by name."""
    return {
        "f_z": axial_peaking_factor(profile),
        "ao": axial_offset(profile),
        "asi": axial_shape_index(profile),
        "saddle_depth": saddle_depth(profile),
    }


# --------------------------------------------------------------------------- #
# shape basis
# --------------------------------------------------------------------------- #
#: Default basis rank.  Measured leave-one-CAMPAIGN-out on the 894 stored
#: records: rank 6 reconstructs a held-out campaign's profiles to <= 2.1e-3 RMS
#: per plane (BOC) / 4.4e-4 (EOC), with a worst-case F_z reconstruction error of
#: 9.3e-3 — half the within-cell F_z spread (0.019), so the basis is not the
#: accuracy bottleneck.  Rank 4 is ~2x worse; a *fixed* analytic (DCT) basis is
#: 3-10x worse at the same rank because the axial shape is dominated by its
#: strong end-effect profile, which no low-order cosine series captures.
DEFAULT_RANK = 6


# ``eq=False``: the fields are numpy arrays, so a generated ``__eq__`` would
# return an array (ambiguous truth value) and the frozen ``__hash__`` would raise.
@dataclass(frozen=True, eq=False)
class AxialBasis:
    """Per-anchor mean profile + orthonormal shape components + mode scales.

    ``decode`` is *exactly* mean-preserving: the mean profile is renormalised to
    ``sum == n_planes`` and every component is projected onto the zero-sum
    subspace, so any reconstructed profile has core-average power exactly 1 by
    construction — the head cannot emit a profile that violates the
    normalisation the label obeys.
    """

    anchors: tuple[str, ...]
    mean: np.ndarray              # (A, P)
    components: np.ndarray        # (A, K, P)
    mode_sd: np.ndarray           # (A, K) train-fold sd of each mode's coeff
    n_fit: int = 0

    @property
    def n_anchors(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.components.shape[1])

    @property
    def n_planes(self) -> int:
        return int(self.mean.shape[1])

    # -- coefficient <-> profile ------------------------------------------- #
    def encode(self, profiles: np.ndarray) -> np.ndarray:
        """``(..., A, P)`` profiles -> ``(..., A, K)`` raw coefficients."""
        p = np.asarray(profiles, dtype=np.float64)
        d = p - self.mean
        return np.einsum("...ap,akp->...ak", d, self.components)

    def decode(self, coeff: np.ndarray) -> np.ndarray:
        """``(..., A, K)`` raw coefficients -> ``(..., A, P)`` profiles."""
        c = np.asarray(coeff, dtype=np.float64)
        return self.mean + np.einsum("...ak,akp->...ap", c, self.components)

    def z_encode(self, profiles: np.ndarray) -> np.ndarray:
        """Profiles -> per-mode standardised coefficients (the training target).

        Standardising per mode is the axial analogue of the band weighting in
        :func:`lpopt.model.train.spectral_map_loss`: mode 0 carries ~80% of the
        shape variance, so an unstandardised loss would put essentially no
        gradient on the higher modes — which are exactly the saddle/double-hump
        structure the profile head exists to resolve.
        """
        return self.encode(profiles) / self.mode_sd

    def z_decode(self, z_coeff: np.ndarray) -> np.ndarray:
        """Standardised coefficients -> profiles."""
        return self.decode(np.asarray(z_coeff, dtype=np.float64) * self.mode_sd)

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "anchors": list(self.anchors),
            "mean": np.asarray(self.mean, dtype=float).tolist(),
            "components": np.asarray(self.components, dtype=float).tolist(),
            "mode_sd": np.asarray(self.mode_sd, dtype=float).tolist(),
            "n_fit": int(self.n_fit),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AxialBasis":
        return cls(
            anchors=tuple(d["anchors"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            components=np.asarray(d["components"], dtype=np.float64),
            mode_sd=np.asarray(d["mode_sd"], dtype=np.float64),
            n_fit=int(d.get("n_fit", 0)),
        )


def _zero_sum_orthonormal(v: np.ndarray) -> np.ndarray:
    """Project rows onto the zero-sum subspace and re-orthonormalise (Gram-Schmidt).

    The SVD components of mean-1-normalised profiles already lie (to float error)
    in the zero-sum subspace; making that exact is what buys ``decode`` its
    normalisation guarantee.
    """
    out = []
    for row in np.asarray(v, dtype=np.float64):
        w = row - row.mean()
        for prev in out:
            w = w - float(w @ prev) * prev
        norm = float(np.linalg.norm(w))
        if norm <= 1e-12:
            continue
        out.append(w / norm)
    return np.asarray(out, dtype=np.float64)


def fit_axial_basis(
    profiles: np.ndarray,
    *,
    rank: int = DEFAULT_RANK,
    anchors: Sequence[str] = ANCHORS,
    mask: np.ndarray | None = None,
) -> AxialBasis:
    """Fit the per-anchor PCA shape basis from ``(N, A, P)`` labelled profiles.

    ``mask`` is ``(N, A)``; only rows with a present label at an anchor enter that
    anchor's fit.  **Leakage rule:** callers must pass TRAIN-fold rows only — the
    basis is a label-derived artifact, exactly like the cyclen physics prior and
    the diffusion power prior, and both of those are fit train-only for the same
    reason.

    An anchor with too few labelled rows to support ``rank`` modes falls back to
    the modes it *can* support, zero-padded, so the head width stays fixed and a
    thin arm degrades instead of crashing.
    """
    p = np.asarray(profiles, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError(f"profiles must be (N, A, P), got shape {p.shape}")
    n_rows, n_anchor, n_plane = p.shape
    if n_anchor != len(anchors):
        raise ValueError(
            f"profiles has {n_anchor} anchors but anchors={tuple(anchors)}")
    k = int(rank)
    if k < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if k > n_plane - 1:
        raise ValueError(
            f"rank {k} exceeds the {n_plane - 1} shape degrees of freedom of a "
            f"{n_plane}-plane mean-normalised profile")
    m = (np.ones((n_rows, n_anchor), dtype=bool) if mask is None
         else np.asarray(mask, dtype=bool))

    mean = np.zeros((n_anchor, n_plane), dtype=np.float64)
    comps = np.zeros((n_anchor, k, n_plane), dtype=np.float64)
    sd = np.ones((n_anchor, k), dtype=np.float64)
    n_fit = 0
    for a in range(n_anchor):
        rows = p[m[:, a], a]
        n_fit = max(n_fit, int(rows.shape[0]))
        if rows.shape[0] == 0:
            mean[a] = 1.0
            continue
        mu = rows.mean(axis=0)
        # exact normalisation (the corpus mean is 1 to ~2e-6 already)
        mean[a] = mu * (n_plane / mu.sum())
        x = rows - mean[a]
        if rows.shape[0] < 2:
            continue
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        v = _zero_sum_orthonormal(vt[:k])
        comps[a, :v.shape[0]] = v
        c = x @ comps[a].T
        s = c.std(axis=0)
        sd[a] = np.where(s > 1e-9, s, 1.0)
    return AxialBasis(anchors=tuple(anchors), mean=mean, components=comps,
                      mode_sd=sd, n_fit=int(n_fit))


def stack_anchor_profiles(
    reader: Any,
    record_ids: Iterable[str],
    *,
    anchors: Sequence[str] = ANCHORS,
    n_planes: int = N_PLANES,
) -> tuple[np.ndarray, np.ndarray]:
    """``((N, A, P) profiles, (N, A) mask)`` for a list of record ids.

    Records with no (or malformed) axial label get an all-NaN profile and a zero
    mask, mirroring :meth:`lpopt.model.dataset_torch.LPDataset._maps`.
    """
    ids = [str(r) for r in record_ids]
    a_n = len(anchors)
    out = np.full((len(ids), a_n, n_planes), np.nan, dtype=np.float64)
    mask = np.zeros((len(ids), a_n), dtype=np.float64)
    for i, rid in enumerate(ids):
        stack = load_axial(reader, rid, n_planes=n_planes)
        if stack is None:
            continue
        out[i] = anchor_profiles(stack, anchors)
        mask[i] = 1.0
    return out, mask


__all__ = [
    "ANCHORS",
    "AXIAL_SUFFIX",
    "AxialBasis",
    "DEFAULT_RANK",
    "N_PLANES",
    "anchor_profiles",
    "axial_key",
    "axial_offset",
    "axial_peaking_factor",
    "axial_shape_index",
    "derived_metrics",
    "fit_axial_basis",
    "load_axial",
    "saddle_depth",
    "stack_anchor_profiles",
]
