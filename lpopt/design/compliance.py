"""Design-path adapter for the R1/R2 compliance contract (task #16).

:func:`lpopt.data.compliance.enforce_new_type` is the "single entry point future
parametric type generation MUST call" — and until now it had **no production
caller at all** (only tests).  Left unwired it fails open twice over, and both
failures are silent:

* ``enr_zone`` is OPTIONAL.  When a caller omits it, ``enforce_new_type`` does not
  reject the spec — it **fills in** ``0.85 * enr_main``
  (``lpopt/data/compliance.py:308-311``).  A design screened at an 0.92 zoning
  ratio would therefore sail through and then be realized at 0.85: the surrogate
  screen that chose it would have been run on a different lattice than the one
  built.  This adapter always passes ``enr_zone`` explicitly.
* ``pin_map`` is OPTIONAL.  Omitted, the R2 octant check is skipped entirely
  (``compliance.py:321-322``).  This adapter passes the full 16x16 map from
  :func:`lpopt.design.lattice.octant_to_full` — not because the gate demands it,
  but because we choose to certify the map we actually authored.

Recorded, not fixed (out of scope here): ``spec.DESIGN_GRID["ratio"]`` offers
``{0.85, 0.92}`` while ``ZONE_RATIO_TARGET`` is ``0.85 +/- 0.03``, so 0.92 is
outside the compliance window and a live 0.92 type exists (``P6257Z2G08N16``,
6.2/5.7 = 0.919).  ``DESIGN_GRID`` is an LHS sampling grid, not a validator
(``FuelDesign.__post_init__`` only checks ``0 < e2 <= e1``).  With the adapter
wired, an attempt to REGENERATE a 0.92 type fails; the shipped rows are untouched.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..data.compliance import (
    OCTANT_TOL,
    ZONE_RATIO_TARGET,
    ZONE_RATIO_TOL,
    ComplianceError,
    enforce_new_type,
)
from .spec import FuelDesign


def design_spec(design: FuelDesign, *,
                pin_map: Sequence[float] | None = None,
                extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``enforce_new_type`` spec for a :class:`FuelDesign`.

    ``e1`` is the main-pin enrichment and ``e2`` the edge-zoning enrichment, so the
    mapping is ``enr_main = e1`` / ``enr_zone = e2``.  ``enr_zone`` is ALWAYS
    present in the returned spec (that is the whole point — see the module
    docstring); ``pin_map`` is included only when the caller has one.
    """
    spec: dict[str, Any] = {
        "type_id": design.type_id,
        "enr_main": float(design.e1),
        "enr_zone": float(design.e2),
        "gd_wt": float(design.gd_wt),
        "n_gd": int(design.n_gd),
        "zoning_variant": design.zoning_variant,
    }
    if design.gd_positions is not None:
        spec["gd_positions"] = design.gd_layout
    if pin_map is not None:
        spec["pin_map"] = list(pin_map)
    if extra:
        spec.update(extra)
    return spec


def enforce_design(design: FuelDesign, *,
                   pin_map: Sequence[float] | None = None,
                   extra: Mapping[str, Any] | None = None,
                   n: int = 16, tol: float = OCTANT_TOL) -> dict[str, Any]:
    """Hard-enforce R1/R2 on a design about to be realized.

    Raises :class:`~lpopt.data.compliance.ComplianceError` when the zoning ratio is
    off ``0.85 +/- 0.03`` (an 0.92 design is rejected here, not normalized) or when
    ``pin_map`` is not octant-symmetric.  Returns the normalized spec.
    """
    return enforce_new_type(design_spec(design, pin_map=pin_map, extra=extra),
                            n=n, tol=tol)


__all__ = [
    "OCTANT_TOL",
    "ZONE_RATIO_TARGET",
    "ZONE_RATIO_TOL",
    "ComplianceError",
    "design_spec",
    "enforce_design",
]
