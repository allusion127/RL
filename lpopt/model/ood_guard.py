"""Serve-time feature / geometry out-of-distribution guard (review sec. 4b).

The champion is trained on a fuel population whose per-type physics features live
on a narrow enrichment x Gd x zoning manifold and whose pin-cell geometry is
strictly CONSTANT (pin pitch 1.285, assembly pitch 20.7772, pellet 0.4096, clad
0.4178/0.4750).  A future pin-pitch / pin-radius optimization moves a served fuel
type OFF that manifold — and, per the readiness review sec. 2.3, the ensemble's
own epistemic variance is BLIND to off-manifold bias (all members trained on the
same manifold agree with each other while being *jointly* wrong).  The z-range
guard, not ensemble variance, is therefore the front line.

This module computes, per feature channel, the z-value ``z = (value - ref) /
scale`` using the SAME normalization constants the v4 featurizer uses
(:mod:`lpopt.model.featurize`), records the training population's ``[z_min,
z_max]`` envelope per channel, and flags any served fuel type whose z on any
channel falls outside ``[z_min - margin, z_max + margin]``.  It mirrors
``PosValCnnBackend.unresolved_fresh_types`` exactly: a *warning surface* that
returns the offending types, NOT a hard fail — predictions are never changed.

Two OOD regimes (review sec. 2.3) are both caught:

* **radius axis** — ``u_mass_g`` is a near-exact ``r_pellet^2`` proxy with a 0.7 g
  population half-range on a 138.8 g mean, so even +0.5 % radius blows its z past
  the envelope (``z ~= +2``).  Loud by construction.
* **pitch axis** — pure pitch leaves ``u_mass`` nominal, so the *spectral* channels
  carry it: ``xs_s12`` (down-scatter ~ moderator volume; scale only 1.5 % of ref)
  and the direct geometry channels ``pin_pitch`` / ``v_mod_over_v_fuel``.

The geometry channels (``pin_pitch`` ... ``v_mod_over_v_fuel``) are a v5 addition:
the training population is constant on them, so their population envelope is the
degenerate ``[0, 0]`` and their ``scale`` is a physically-motivated normalizer
(not a fitted half-range).  Including them as direct channels CLOSES the
"silent-but-wrong small-pitch" band the review flagged as the dangerous case
(sec. 2.3) — at the cost of the guard firing on the admissible +0.5..1 % pitch
edge, which is the safe conservative direction (those are exactly the variants the
sec. 4c DeCART blind-probe must certify before they enter the optimizer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..data.fuel_types import FuelVec
from .featurize import (
    _BU_K1_SCALE,
    _COND_NORM,
    _FF_REF,
    _FF_SCALE,
    _KINF_REF,
    _KINF_SCALE,
    _U_MASS_REF,
    _U_MASS_SCALE,
    _V4_SCALES,
)

#: Default two-sided margin (in z units) added to the population envelope before a
#: channel is deemed OOD.  ~0.5 keeps a hair of the population's own boundary noise
#: from tripping the guard while still catching a genuine off-manifold move.
DEFAULT_MARGIN: float = 0.5

#: enrichment (ref, scale) — the v4/v3 conditioning envelope (featurize _COND_NORM).
_ENR_REF: float = _COND_NORM["v4"]["enr_ref"]
_ENR_SCALE: float = _COND_NORM["v4"]["enr_scale"]

#: v5 geometry-channel normalizers.  The training population is CONSTANT on these
#: (envelope [0, 0]); ``scale`` is a physically-motivated normalizer chosen so a
#: variant lands O(1) outside the envelope for an inadmissible/large move while the
#: admissible +0.5 % pitch edge sits near the margin.  ``asm_pitch`` is the frozen
#: assembly envelope — a tight scale makes any drift a hard tripwire (it must never
#: move; the deck editor asserts it, this is the serve-side backstop).
_GEOM_SCALES: dict[str, tuple[float, float]] = {
    "pin_pitch": (1.285, 0.02),
    "asm_pitch": (20.7772, 0.005),
    "r_pellet": (0.4096, 0.004),
    "r_clad_in": (0.4178, 0.004),
    "r_clad_out": (0.4750, 0.005),
    "p_over_d": (1.3526, 0.02),
    "v_mod_over_v_fuel": (1.788, 0.05),
}


@dataclass(frozen=True)
class OodChannel:
    """One guarded feature channel: ``(name, FuelVec attr, ref, scale)``."""

    name: str
    attr: str
    ref: float
    scale: float

    def z(self, vec: FuelVec) -> float | None:
        """``(value - ref)/scale`` for this channel of ``vec`` (None if absent)."""
        v = getattr(vec, self.attr, None)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(fv) or self.scale == 0.0:
            return None
        return (fv - self.ref) / self.scale


def _build_channels() -> tuple[OodChannel, ...]:
    """The ordered guarded-channel inventory (physics v4 + geometry v5)."""
    chans: list[OodChannel] = [
        OodChannel("u_mass", "u_mass_g", _U_MASS_REF, _U_MASS_SCALE),
        OodChannel("kinf0", "kinf0", _KINF_REF, _KINF_SCALE),
        OodChannel("kinf10", "kinf10", _KINF_REF, _KINF_SCALE),
        OodChannel("kinf20", "kinf20", _KINF_REF, _KINF_SCALE),
        OodChannel("kinf30", "kinf30", _KINF_REF, _KINF_SCALE),
        OodChannel("bu_k1", "bu_k1", 0.0, _BU_K1_SCALE),
        OodChannel("ff_pin_max", "ff_pin_max", _FF_REF, _FF_SCALE),
        OodChannel("enr_main", "enr_main", _ENR_REF, _ENR_SCALE),
        OodChannel("enr_zone", "enr_zone", _ENR_REF, _ENR_SCALE),
        OodChannel("gd_u_enr", "gd_u_enr", _ENR_REF, _ENR_SCALE),
    ]
    # branch / macro-XS / ADF / zone signatures — the SAME (attr, _V4_SCALES key)
    # pairs the v4 featurizer scales its origin_* channels with.
    _scaled = (
        ("boron_worth", "boron_worth", "boron_worth"),
        ("doppler", "doppler_coef", "doppler"),
        ("mtc_dmod", "mtc_dmod", "mtc_dmod"),
        ("cr1_worth", "cr1_worth", "cr1_worth"),
        ("zone_pins", "zone_pin_count", "zone_pins"),
        ("xs_a2", "xs_a2", "xs_a2"),
        ("xs_nf2", "xs_nf2", "xs_nf2"),
        ("xs_s12", "xs_s12", "xs_s12"),
        ("adf_corner_g2", "adf_corner_g2", "adf_corner_g2"),
    )
    for name, attr, key in _scaled:
        ref, scale = _V4_SCALES[key]
        chans.append(OodChannel(name, attr, ref, scale))
    # v5 pin-cell geometry channels.
    for name, (ref, scale) in _GEOM_SCALES.items():
        chans.append(OodChannel(name, name, ref, scale))
    return tuple(chans)


#: The guarded-channel inventory (built once at import).
OOD_CHANNELS: tuple[OodChannel, ...] = _build_channels()
_CHANNEL_BY_NAME: dict[str, OodChannel] = {c.name: c for c in OOD_CHANNELS}


# --------------------------------------------------------------------------- #
# population envelope
# --------------------------------------------------------------------------- #
def population_envelope(vecs: Iterable[FuelVec]) -> dict[str, list[float]]:
    """``{channel: [z_min, z_max]}`` over a fuel population.

    Only vecs carrying a finite value for a channel contribute to that channel's
    envelope.  A channel that no vec carries (or that is strictly constant, e.g. the
    geometry channels on today's population) yields ``[0.0, 0.0]``.
    """
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for vec in vecs:
        for chan in OOD_CHANNELS:
            z = chan.z(vec)
            if z is None:
                continue
            if chan.name not in lo or z < lo[chan.name]:
                lo[chan.name] = z
            if chan.name not in hi or z > hi[chan.name]:
                hi[chan.name] = z
    env: dict[str, list[float]] = {}
    for chan in OOD_CHANNELS:
        env[chan.name] = [lo.get(chan.name, 0.0), hi.get(chan.name, 0.0)]
    return env


def population_envelope_from_library(fuel: Any,
                                     library_ids: Sequence[str] | None = None
                                     ) -> dict[str, list[float]]:
    """Population envelope over a :class:`~lpopt.data.fuel_types.FuelLibrary`.

    ``library_ids`` restricts the population (default: every library in the table)
    — a caller can scope the envelope to the training library so a served type from
    a *different* physics library still trips the guard.
    """
    df = fuel.frame
    if library_ids is not None:
        df = df[df["library_id"].isin(list(library_ids))]
    vecs: list[FuelVec] = []
    for _, row in df.iterrows():
        try:
            vecs.append(fuel.get(str(row["type_id"]), str(row["library_id"])))
        except KeyError:
            continue
    return population_envelope(vecs)


# --------------------------------------------------------------------------- #
# per-type OOD check
# --------------------------------------------------------------------------- #
def vec_ood_channels(vec: FuelVec, envelope: Mapping[str, Sequence[float]],
                     *, margin: float = DEFAULT_MARGIN
                     ) -> list[tuple[str, float]]:
    """The ``(channel, z)`` pairs on which ``vec`` falls outside the envelope.

    A channel is OOD when its z is below ``z_min - margin`` or above ``z_max +
    margin``.  Channels absent from ``vec`` (None/NaN) never fire.  Returned in
    :data:`OOD_CHANNELS` order (most-fragile physics canaries first).
    """
    out: list[tuple[str, float]] = []
    for chan in OOD_CHANNELS:
        z = chan.z(vec)
        if z is None:
            continue
        band = envelope.get(chan.name)
        if not band:
            continue
        z_min, z_max = float(band[0]), float(band[1])
        if z < z_min - margin or z > z_max + margin:
            out.append((chan.name, z))
    return out


def feature_ood_vecs(vecs_by_type: Mapping[str, FuelVec | None],
                     envelope: Mapping[str, Sequence[float]],
                     *, margin: float = DEFAULT_MARGIN
                     ) -> dict[str, list[tuple[str, float]]]:
    """``{type_id: [(channel, z), ...]}`` for every type with >=1 OOD channel.

    A ``None`` vec (unresolved type) is skipped here — that is the separate
    ``unresolved_fresh_types`` signal, not a feature-OOD one.
    """
    flagged: dict[str, list[tuple[str, float]]] = {}
    for tid, vec in vecs_by_type.items():
        if vec is None:
            continue
        offenders = vec_ood_channels(vec, envelope, margin=margin)
        if offenders:
            flagged[tid] = offenders
    return flagged


def format_ood_warning(flagged: Mapping[str, Sequence[tuple[str, float]]]) -> str:
    """One-line human warning for a ``feature_ood_vecs`` result (``""`` if clean)."""
    if not flagged:
        return ""
    parts: list[str] = []
    for tid in sorted(flagged):
        worst = max(flagged[tid], key=lambda cz: abs(cz[1]))
        parts.append(f"{tid} [{worst[0]} z={worst[1]:+.1f}]")
    return ("geometry/spectrum OOD — prediction unvalidated for: "
            + ", ".join(parts))


__all__ = [
    "DEFAULT_MARGIN",
    "OOD_CHANNELS",
    "OodChannel",
    "feature_ood_vecs",
    "format_ood_warning",
    "population_envelope",
    "population_envelope_from_library",
    "vec_ood_channels",
]
