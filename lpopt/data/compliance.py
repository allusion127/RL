"""Assembly-design compliance utilities (user rules R1-R3, 2026-07-22).

This module ships the R1-R3 compliance machinery for the F_r=1.55 boundary training
campaign AND the contract for future (Phase A) parametric type generation.  It is
deliberately data-light: the ga80 letter-type library lost its ``enr_main`` /
``enr_zone`` bookkeeping for all 70 rows (44/80 types have no recoverable
enrichment), so for those types compliance is the honest terminal state
``'unknown'`` — they are KEPT for training-label production (this campaign makes
training data, not final designs) but reported as ``'unknown'`` in any real-design
search roster.

Rules
-----
* **R1** (one enrichment spec per pattern; cross-anchor pairs banned):
  :func:`is_cross_anchor` returns True for a pair whose two family anchors differ
  (e.g. ``E1_J2`` -> anchors ``E`` != ``J``).  The frontier roster builder and the
  PC2 kit exporter both hard-fail on any cross-anchor pair, citing R1.
* **R2** (octant / 1/8 pin symmetry):
  :func:`is_octant_symmetric` checks a flat 16x16 %DIST pin map is invariant under
  the full dihedral group D4 (transpose + horizontal mirror generate it).
  :func:`audit_fuel_types` records ``octant_symmetry`` and ``zone_ratio`` flags in
  ``{pass, fail, unknown}`` per type, keyed by type id, alongside a
  ``compliance_source``.
* **R3** (1/4 rotational core symmetry): structurally enforced by the orbit units
  (:class:`~lpopt.search.genome.GeneralOrbitGenome`) — verification only, no
  mechanics here (a test asserts orbit-unit placement preserves quarter-core
  symmetry).

Phase A contract
----------------
:func:`enforce_new_type` is the single entry point future parametric type
generation MUST call: it HARD-enforces ``enr_zone = 0.85 * enr_main`` and
octant-symmetric pin placement, raising :class:`ComplianceError` on any violation
it cannot normalize.  Keeping the rule in the repo (not session memory) is the
whole point of shipping it here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ComplianceError",
    "ZONE_RATIO_TARGET",
    "ZONE_RATIO_TOL",
    "OCTANT_TOL",
    "TypeCompliance",
    "family_anchor",
    "is_cross_anchor",
    "assert_mono_anchor",
    "is_octant_symmetric",
    "octant_symmetry_flag",
    "zone_ratio_flag",
    "audit_fuel_types",
    "enforce_new_type",
]


class ComplianceError(ValueError):
    """Raised when a proposed type / roster violates a HARD compliance rule."""


#: R1 axial-zoning ratio: zoning pins run at 0.85 x the main-pin enrichment.
ZONE_RATIO_TARGET = 0.85
#: Tolerance on the measured enr_zone/enr_main ratio before a type is flagged fail.
ZONE_RATIO_TOL = 0.03
#: Absolute tolerance on %DIST octant-symmetry residuals (relative pin power).
OCTANT_TOL = 1.0e-3


# --------------------------------------------------------------------------- #
# R1 — family anchors / cross-anchor detection
# --------------------------------------------------------------------------- #
def family_anchor(half: str) -> str:
    """The family-anchor letters of one pair half (``"E1"`` -> ``"E"``).

    The leading alphabetic run identifies the enrichment family; the trailing
    index selects the axial-split sibling within it.  ``""`` for a half with no
    leading letters (defensive — treated as its own anchor by the caller).
    """
    m = re.match(r"^([A-Za-z]+)", str(half).strip())
    return m.group(1).upper() if m else ""


def is_cross_anchor(pair: str) -> bool:
    """True when ``pair`` couples two DIFFERENT enrichment families (R1 ban).

    A pair is written ``<half>_<half>`` (e.g. ``E1_E2``).  Mono-anchor pairs share
    a family letter (``E1_E2`` -> ``E``/``E``); a cross-anchor pair (``E1_J2`` ->
    ``E``/``J``) mixes enrichment specs across feed assemblies and is banned from
    the roster.  A malformed pair (not exactly two halves) is treated as
    cross-anchor (fails safe — the caller hard-rejects it).
    """
    parts = str(pair).split("_")
    if len(parts) != 2:
        return True
    return family_anchor(parts[0]) != family_anchor(parts[1])


def assert_mono_anchor(pairs: Iterable[str]) -> None:
    """Hard-fail (R1) if ANY pair in ``pairs`` is cross-anchor.

    Used by the frontier roster builder and the PC2 kit exporter so a cross-anchor
    pair can never enter the campaign — the rule is enforced structurally, not by
    convention.
    """
    bad = sorted({p for p in pairs if is_cross_anchor(p)})
    if bad:
        raise ComplianceError(
            f"R1 violation: cross-anchor pair(s) {bad} are banned from the roster "
            f"(a loading pattern must use ONE enrichment spec across all feed "
            f"assemblies; mono-anchor same-family pairs only)"
        )


# --------------------------------------------------------------------------- #
# R2 — octant (1/8) pin-map symmetry
# --------------------------------------------------------------------------- #
def is_octant_symmetric(flat_map: Sequence[float], n: int = 16,
                        tol: float = OCTANT_TOL) -> bool:
    """True when the ``n x n`` pin map (row-major flat) obeys 1/8 octant symmetry.

    Octant symmetry is invariance under the square's dihedral group D4; the
    transpose (main-diagonal reflection) and the horizontal mirror generate the
    whole group, so a map invariant under BOTH (within ``tol``) is octant-symmetric.
    Guide-tube / instrument zeros participate like any other position.  Returns
    False on a wrong-length or non-finite map (an unusable map is never certified).
    """
    arr = np.asarray(list(flat_map), dtype=float)
    if arr.size != n * n or not np.all(np.isfinite(arr)):
        return False
    m = arr.reshape(n, n)
    if np.max(np.abs(m - m.T)) > tol:                 # main-diagonal reflection
        return False
    if np.max(np.abs(m - m[:, ::-1])) > tol:          # horizontal mirror
        return False
    return True


def octant_symmetry_flag(flat_map: Sequence[float] | None, n: int = 16,
                         tol: float = OCTANT_TOL) -> str:
    """``'pass'`` / ``'fail'`` / ``'unknown'`` for a type's %DIST pin map.

    ``None`` or an empty map (no preserved HGC) -> ``'unknown'``; a well-formed map
    is certified pass/fail by :func:`is_octant_symmetric`.
    """
    if flat_map is None or len(flat_map) == 0:
        return "unknown"
    arr = np.asarray(list(flat_map), dtype=float)
    if arr.size != n * n:
        return "unknown"
    return "pass" if is_octant_symmetric(arr, n=n, tol=tol) else "fail"


# --------------------------------------------------------------------------- #
# R1 — enr_zone = 0.85 x enr_main zoning ratio
# --------------------------------------------------------------------------- #
def zone_ratio_flag(enr_main: float | None, enr_zone: float | None,
                    target: float = ZONE_RATIO_TARGET,
                    tol: float = ZONE_RATIO_TOL) -> str:
    """``'pass'`` / ``'fail'`` / ``'unknown'`` for the zoning-enrichment ratio.

    ``'unknown'`` when either enrichment is missing / NaN / non-positive (the ga80
    all-NaN case).  Otherwise ``'pass'`` iff ``|enr_zone/enr_main - target| <= tol``.
    """
    def _num(x: Any) -> float | None:
        if x is None:
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    a = _num(enr_main)
    z = _num(enr_zone)
    if a is None or z is None or a <= 0.0:
        return "unknown"
    return "pass" if abs(z / a - float(target)) <= float(tol) else "fail"


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
@dataclass
class TypeCompliance:
    """Per-type R1/R2 compliance record (a sidecar row, keyed by ``type_id``)."""

    type_id: str
    library_id: str
    octant_symmetry: str        # pass | fail | unknown
    zone_ratio: str             # pass | fail | unknown
    compliance_source: str      # e.g. "hgc%dist", "enr", "hgc%dist+enr", "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id, "library_id": self.library_id,
            "octant_symmetry": self.octant_symmetry, "zone_ratio": self.zone_ratio,
            "compliance_source": self.compliance_source,
        }


def _compliance_source(has_map: bool, has_enr: bool) -> str:
    parts = []
    if has_map:
        parts.append("hgc%dist")
    if has_enr:
        parts.append("enr")
    return "+".join(parts) if parts else "none"


def audit_types(
    records: Iterable[Mapping[str, Any]], *, n: int = 16, tol: float = OCTANT_TOL,
    zone_target: float = ZONE_RATIO_TARGET, zone_tol: float = ZONE_RATIO_TOL,
) -> list[TypeCompliance]:
    """Audit an iterable of type records into :class:`TypeCompliance` rows.

    Each record is a mapping with ``type_id`` (+ optional ``library_id``,
    ``enr_main``, ``enr_zone``, and ``dist_map`` — a flat 16x16 %DIST pin map or
    None).  This is the pure, testable core of :func:`audit_fuel_types`; it needs
    no live HGC files (the caller resolves ``dist_map`` however it likes).
    """
    out: list[TypeCompliance] = []
    for rec in records:
        dist = rec.get("dist_map")
        has_map = dist is not None and len(dist) > 0
        enr_main = rec.get("enr_main")
        enr_zone = rec.get("enr_zone")
        zr = zone_ratio_flag(enr_main, enr_zone, zone_target, zone_tol)
        out.append(TypeCompliance(
            type_id=str(rec.get("type_id")),
            library_id=str(rec.get("library_id", "ga80")),
            octant_symmetry=octant_symmetry_flag(dist, n=n, tol=tol),
            zone_ratio=zr,
            compliance_source=_compliance_source(has_map, zr != "unknown"),
        ))
    return out


def audit_fuel_types(
    fuel: Any = None, *, library_id: str = "ga80",
    hgc_maps: Mapping[str, Sequence[float]] | None = None,
    n: int = 16, tol: float = OCTANT_TOL,
) -> list[TypeCompliance]:
    """Audit every ``library_id`` type of a :class:`FuelLibrary` for R1/R2.

    ``hgc_maps`` (optional) maps ``type_id`` -> a flat 16x16 %DIST pin map for the
    types with a preserved HGC; a type absent from it (or with no map) audits as
    octant ``'unknown'``.  ``enr_main`` / ``enr_zone`` come from the fuel vecs and
    are ``'unknown'`` for the all-NaN ga80 rows.  Returns the sidecar rows; the CLI
    writes them to a report JSON (a sidecar keyed by type id, NOT a mutation of the
    shipped ``fuel_types.parquet`` consumed by older code paths).
    """
    hgc_maps = hgc_maps or {}
    records: list[dict[str, Any]] = []
    if fuel is not None:
        for tid in fuel.types(library_id):
            try:
                vec = fuel.get(tid, library_id)
            except KeyError:
                continue
            records.append({
                "type_id": tid, "library_id": library_id,
                "enr_main": getattr(vec, "enr_main", None),
                "enr_zone": getattr(vec, "enr_zone", None),
                "dist_map": hgc_maps.get(tid),
            })
    else:
        for tid, mp in hgc_maps.items():
            records.append({"type_id": tid, "library_id": library_id, "dist_map": mp})
    return audit_types(records, n=n, tol=tol)


# --------------------------------------------------------------------------- #
# Phase A contract — hard enforcement for future parametric type generation
# --------------------------------------------------------------------------- #
def enforce_new_type(spec: Mapping[str, Any], *, n: int = 16,
                     tol: float = OCTANT_TOL) -> dict[str, Any]:
    """Return a compliance-normalized copy of a NEW parametric type spec (Phase A).

    HARD-enforces the two generative rules the ga80 legacy library could only be
    audited against:

    * **enr_zone = 0.85 x enr_main** — if ``enr_zone`` is absent it is DERIVED; if
      present it must already equal ``0.85*enr_main`` within :data:`ZONE_RATIO_TOL`
      or :class:`ComplianceError` is raised (the generator must not ship an
      off-ratio zoning enrichment).
    * **octant-symmetric pin placement** — ``pin_map`` (flat ``n*n``), when given,
      must be octant-symmetric or :class:`ComplianceError` is raised.

    ``spec`` requires ``enr_main`` (> 0).  The returned dict is a shallow copy with
    ``enr_zone`` filled in; the input is never mutated.  This is the single entry
    point future generation MUST call so the rule lives in the repo, not in memory.
    """
    out = dict(spec)
    try:
        enr_main = float(out["enr_main"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComplianceError("enforce_new_type: enr_main is required and numeric") from exc
    if not math.isfinite(enr_main) or enr_main <= 0.0:
        raise ComplianceError(f"enforce_new_type: enr_main must be > 0 (got {enr_main})")

    target_zone = ZONE_RATIO_TARGET * enr_main
    if out.get("enr_zone") is None:
        out["enr_zone"] = target_zone
    else:
        enr_zone = float(out["enr_zone"])
        if zone_ratio_flag(enr_main, enr_zone) != "pass":
            raise ComplianceError(
                f"R1 violation: enr_zone={enr_zone} must equal 0.85*enr_main="
                f"{target_zone:.4f} (+/-{ZONE_RATIO_TOL}); parametric zoning "
                f"enrichment is fixed at 0.85 of the main-pin enrichment"
            )

    pin_map = out.get("pin_map")
    if pin_map is not None and not is_octant_symmetric(pin_map, n=n, tol=tol):
        raise ComplianceError(
            "R2 violation: pin_map is not octant (1/8) symmetric; parametric pin "
            "placement must obey octant symmetry"
        )
    return out
