"""Normalization scales for the flatness-first objective (program 20260725 §1.2).

The objective scalar of the ``flat_power`` mode is

.. code-block:: text

    z_peak = node_peak / PEAK_SCALE          # PRIMARY   (weight 1.0)
    z_cov  = map_cov   / COV_SCALE           # SECONDARY (weight w_cov = 0.5)
    scalar = -( z_peak + w_cov * z_cov )

so ``PEAK_SCALE`` / ``COV_SCALE`` are what make the declared 1 : 0.5 weight ratio
mean anything.  This module owns both, plus the artifact that carries the
measured per-cell values, and it is imported by BOTH consumers of the scalar —
:mod:`..search.acquisition` (predicted candidates) and
:mod:`..search.campaign` (verified record columns) — so the two can never drift.

Why per-cell normalization is the DEFAULT (program §1.2 / decision D4)
---------------------------------------------------------------------
Declaring ``w_cov = 0.5`` only fixes the weight ratio if both terms are divided
by a scale in the SAME units of "how much this cell's candidates actually vary".
They are not, with fixed global constants: measured over the store's mapped
converged rows (63 cells, n >= 8) the within-cell ``SD_cov / SD_peak`` ratio
spans 0.111 - 0.549, a factor of **4.95**.  Feeding those cells a single global
pair of constants makes the REALIZED secondary weight range over
0.25 - 1.25 — i.e. in the worst cell ``map_cov`` is weighted 2.5x MORE than
declared, and in another 2x less.  Dividing each cell by its own within-cell SD
makes the realized weight exactly ``w_cov`` in every cell, which is the only
setting under which the declared ratio is honest.

The global constants remain as the fallback for a cell the artifact never fitted
(a fresh feed / e_core bin), and :meth:`FlatScale.realized_w_cov` reports the
distribution above so the cost of that fallback is never invisible.

The scales are SDs of the multiplicity-weighted scalars of
:mod:`.flatness` — the draft's 0.23 / 0.065 came from the unweighted definition
on a corpus that was 87% two mega-cells and are NOT carried forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from collections.abc import Mapping as _MappingABC
from typing import Any, Iterable, Mapping

import numpy as np

from . import map_calibration as _MC

#: Artifact file name, read from the STORE dir (it is a property of the label
#: corpus, not of any model checkpoint).
ARTIFACT_NAME = "flat_scale.json"
ARTIFACT_SCHEMA = "flat_scale_v1"

#: Version of the SCALAR DEFINITION itself (``-(z_peak + w_cov*z_cov)``).  Bump
#: it when the formula changes shape — a stored objective value produced by a
#: different formula cannot be migrated by rescaling and must be refused.
SCALAR_VERSION = "flat_scalar_v1"

#: Two recorded scale identities count as the same normalizer within this
#: RELATIVE tolerance.  The identity is round-tripped through JSON at 9 decimals,
#: so anything looser than a float round-trip is a genuine refit.
IDENTITY_RTOL = 1.0e-9

#: Global fallback scales — the MEDIAN within-cell SD over the store's mapped
#: converged rows under the multiplicity-weighted definition (63 cells, n >= 8,
#: measured 2026-07-26 by ``python -m lpopt.tools.fit_flat_scale``).
DEFAULT_PEAK_SCALE = 0.370285
DEFAULT_COV_SCALE = 0.081127

#: Declared secondary weight (program §1.2 C1: node_peak PRIMARY, map_cov
#: SECONDARY).  The weight inversion versus the draft is the whole point: within
#: a cell rho(node_peak, F_r) = 0.983 and map_cov adds partial rho 0.107 once
#: node_peak is known, so peak carries the licensing-relevant signal.
DEFAULT_W_COV = 0.5

#: UCB pessimism used by both the acquisition scalar and the reported scales.
DEFAULT_RISK_Z = 0.25

#: A cell needs this many mapped rows before its own SD is trusted.
MIN_CELL_ROWS = 8

#: A within-cell SD below this is a degenerate (collapsed) cell — fall back to
#: the global scale rather than dividing by ~0 and inventing a huge z.
_MIN_SCALE = 1.0e-6


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass(frozen=True)
class CellScale:
    """One fitted cell's within-cell SDs (the per-cell normalizers)."""

    cell: str
    n: int
    peak_scale: float
    cov_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {"n": int(self.n), "peak_scale": float(self.peak_scale),
                "cov_scale": float(self.cov_scale)}


@dataclass(frozen=True)
class FlatScale:
    """Resolved normalization for the flatness objective.

    ``per_cell`` (default TRUE, program §1.2 / D4) uses a cell's own within-cell
    SDs when the artifact fitted it, and the global constants otherwise.  Setting
    it False pins every cell to the global constants — kept because it is the
    only way to reproduce a fixed-constant run, never because it is honest.
    """

    peak_scale: float = DEFAULT_PEAK_SCALE
    cov_scale: float = DEFAULT_COV_SCALE
    cells: Mapping[str, CellScale] = field(default_factory=dict)
    per_cell: bool = True
    source: str = "module defaults"

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_artifact(cls, doc: Mapping[str, Any] | None, *,
                      per_cell: bool = True,
                      source: str = "artifact") -> "FlatScale":
        """Build from a parsed ``flat_scale.json`` (``None`` -> module defaults)."""
        if not doc:
            return cls(per_cell=per_cell)
        g = doc.get("global") or {}
        peak = _finite(g.get("peak_scale")) or DEFAULT_PEAK_SCALE
        cov = _finite(g.get("cov_scale")) or DEFAULT_COV_SCALE
        cells: dict[str, CellScale] = {}
        for key, entry in (doc.get("cells") or {}).items():
            if not isinstance(entry, _MappingABC):
                continue
            p = _finite(entry.get("peak_scale"))
            c = _finite(entry.get("cov_scale"))
            if p is None or c is None or p < _MIN_SCALE or c < _MIN_SCALE:
                continue
            cells[str(key)] = CellScale(str(key), int(entry.get("n", 0) or 0), p, c)
        return cls(peak_scale=peak, cov_scale=cov, cells=cells,
                   per_cell=bool(per_cell), source=source)

    @classmethod
    def from_store(cls, store_dir: str | Path | None, *,
                   per_cell: bool = True) -> "FlatScale":
        """Load ``<store_dir>/flat_scale.json``; module defaults when absent.

        An absent / unreadable / malformed artifact is NOT an error: the mode
        still runs on the global constants, and :attr:`source` says so.
        """
        if store_dir is None:
            return cls(per_cell=per_cell)
        path = Path(store_dir) / ARTIFACT_NAME
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return cls(per_cell=per_cell, source=f"module defaults ({path} absent)")
        return cls.from_artifact(doc, per_cell=per_cell, source=str(path))

    # -- resolution --------------------------------------------------------- #
    def has_cell(self, cell_key: str | None) -> bool:
        return bool(self.per_cell and cell_key and cell_key in self.cells)

    def scales_for(self, cell_key: str | None = None) -> tuple[float, float]:
        """``(peak_scale, cov_scale)`` for a cell (global when unfitted)."""
        if self.has_cell(cell_key):
            cs = self.cells[str(cell_key)]
            return float(cs.peak_scale), float(cs.cov_scale)
        return float(self.peak_scale), float(self.cov_scale)

    def identity(self, *, w_cov: float = DEFAULT_W_COV,
                 cell_key: str | None = None,
                 cell: str | None = None) -> dict[str, Any]:
        """The scalar's IDENTITY — what a stored objective value means.

        A ``flat_power`` objective value is ``-(peak/peak_scale + w_cov*cov/
        cov_scale)``, i.e. a number in the units of THIS normalizer.  Re-fitting
        ``flat_scale.json`` (or flipping ``per_cell`` / a deck override) silently
        redefines every persisted ``best["objective"]`` — the old number stops
        being comparable with the new ones, and nothing in the file says so.

        This dict is what gets written alongside the value so a reader can tell.
        Compare two of them with :func:`identity_matches`; ``cell`` overrides the
        reported cell label when the caller resolved the scales elsewhere (e.g.
        after folding in a deck override).
        """
        peak, cov = self.scales_for(cell_key)
        return {
            "version": SCALAR_VERSION,
            "schema": ARTIFACT_SCHEMA,
            "cell": cell if cell is not None else cell_key,
            "peak_scale": round(float(peak), 9),
            "cov_scale": round(float(cov), 9),
            "w_cov": round(float(w_cov), 9),
            "per_cell": bool(self.per_cell),
            "fitted": self.has_cell(cell_key),
            "source": self.source,
        }

    def describe(self, cell_key: str | None = None) -> dict[str, Any]:
        """What this scalar is actually normalized by (for the run log/report)."""
        peak, cov = self.scales_for(cell_key)
        return {
            "cell": cell_key,
            "per_cell": bool(self.per_cell),
            "fitted": self.has_cell(cell_key),
            "peak_scale": round(peak, 6),
            "cov_scale": round(cov, 6),
            "source": self.source,
            "n_cells": len(self.cells),
        }

    # -- the scalar (ONE definition, shared by acquisition + campaign) ------- #
    def z(self, peak: Any, cov: Any, cell_key: str | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
        """``(z_peak, z_cov)`` arrays; a non-finite input stays NaN."""
        ps, cs = self.scales_for(cell_key)
        p = np.asarray(peak, dtype=float)
        c = np.asarray(cov, dtype=float)
        return p / ps, c / cs

    def scalar(self, peak: Any, cov: Any, *, w_cov: float = DEFAULT_W_COV,
               cell_key: str | None = None) -> np.ndarray:
        """``-( z_peak + w_cov * z_cov )`` — higher is flatter (program §1.2/§1.3).

        ``node_peak`` is the PRIMARY term and is required: a NaN peak yields
        ``-inf`` (an unusable candidate / an unlabelled row), never a peak-free
        ranking on ``map_cov`` alone.  A NaN ``map_cov`` drops the SECONDARY term
        only (contributes 0), so a map that produced a peak but no CoV still
        ranks on the term that carries the licensing signal.
        """
        z_peak, z_cov = self.z(peak, cov, cell_key)
        peak_ok = np.isfinite(z_peak)
        cov_term = np.where(np.isfinite(z_cov), float(w_cov) * z_cov, 0.0)
        out = -(np.where(peak_ok, z_peak, 0.0) + cov_term)
        return np.where(peak_ok, out, -np.inf)

    def scalar_one(self, peak: Any, cov: Any, *, w_cov: float = DEFAULT_W_COV,
                   cell_key: str | None = None) -> float:
        """Scalar of ONE (peak, cov) pair — ``-inf`` when the peak is missing."""
        p = _finite(peak)
        if p is None:
            return float("-inf")
        return float(self.scalar(np.array([p]), np.array([_finite(cov) or np.nan]),
                                 w_cov=w_cov, cell_key=cell_key)[0])

    # -- honesty reporting -------------------------------------------------- #
    def realized_w_cov(self, *, w_cov: float = DEFAULT_W_COV,
                       cells: Iterable[CellScale] | None = None
                       ) -> dict[str, Any]:
        """Realized secondary weight per cell, in within-cell SD units.

        With ``per_cell`` normalization every cell realizes EXACTLY ``w_cov`` (the
        declared value), because each term is divided by that cell's own SD.
        With the global constants the realized weight is
        ``w_cov * (SD_cov_cell / COV_SCALE) / (SD_peak_cell / PEAK_SCALE)`` and
        varies by the SD-ratio spread — this is the number program §1.2 demands be
        written down rather than assumed.
        """
        pool = list(cells if cells is not None else self.cells.values())
        out: dict[str, Any] = {"declared": float(w_cov), "n_cells": len(pool),
                               "per_cell": bool(self.per_cell)}
        if not pool:
            return out
        if self.per_cell:
            out.update({"min": float(w_cov), "median": float(w_cov),
                        "max": float(w_cov), "spread": 1.0,
                        "note": "per-cell normalization: realized == declared in "
                                "every fitted cell"})
            return out
        r = np.array([
            float(w_cov) * (c.cov_scale / self.cov_scale)
            / (c.peak_scale / self.peak_scale)
            for c in pool if c.peak_scale > _MIN_SCALE
        ], dtype=float)
        r = r[np.isfinite(r)]
        if not r.size:
            return out
        out.update({
            "min": float(r.min()), "p25": float(np.percentile(r, 25)),
            "median": float(np.median(r)), "p75": float(np.percentile(r, 75)),
            "max": float(r.max()),
            "spread": float(r.max() / r.min()) if r.min() > 0 else float("inf"),
            "note": "global constants: the declared weight is NOT what any "
                    "individual cell realizes",
        })
        return out


def identity_matches(stored: Mapping[str, Any] | None,
                     current: Mapping[str, Any] | None) -> bool:
    """True when a stored objective value is still in ``current``'s units.

    Deliberately strict and NULL-INTOLERANT: a missing / partial / non-mapping
    identity means "we cannot prove these are the same normalizer", which is a
    mismatch.  A state file written before the identity existed therefore reads
    as a mismatch and takes the reader's migrate-or-refuse path, rather than
    being trusted because it is silent.
    """
    if not isinstance(stored, _MappingABC) or not isinstance(current, _MappingABC):
        return False
    if str(stored.get("version") or "") != str(current.get("version") or ""):
        return False
    for key in ("peak_scale", "cov_scale", "w_cov"):
        a = _finite(stored.get(key))
        b = _finite(current.get(key))
        if a is None or b is None:
            return False
        if not math.isclose(a, b, rel_tol=IDENTITY_RTOL, abs_tol=0.0):
            return False
    return True


# --------------------------------------------------------------------------- #
# F_r safety-gate bias correction (program §2.1, decision D1)
# --------------------------------------------------------------------------- #
#: Per-cell map-head calibration artifact, produced by
#: :mod:`..tools.fit_map_calibration` and read from the STORE dir.  Only two of
#: its keys concern the safety gate; the schema itself lives in
#: :mod:`.map_calibration`, and this name is pinned equal to it by a test.
GATE_CALIBRATION_NAME = _MC.ARTIFACT_NAME


def load_gate_correction(store_dir: str | Path | None, cell_key: str | None
                         ) -> tuple[float | None, float | None]:
    """``(fr_bias, fr_sigma)`` for the D1 gate, or ``(None, None)``.

    Decision D1: the F_r safety gate is ``1.70 − bias_cell − 0.5·sigma_cell``
    *when the map-head bias correction is available*, and otherwise HOLDS at
    1.70.  This function is the "is it available" test, and it is deliberately
    strict: an absent file, an unparseable file, or a cell the artifact never
    fitted **and** no global fallback all return ``(None, None)`` so the gate
    holds.  Relaxing a licensing-adjacent gate on a guess is exactly the failure
    the program's §2.1 rejects (the draft's 1.75 loosened the gate while the map
    head carried a −0.147 node_peak optimism bias, which compounds the winner's
    curse instead of correcting it).

    Thin delegate to :func:`.map_calibration.load_gate_correction` — kept here
    because the acquisition/campaign call site imports it from this module, and
    because the gate rule belongs next to the scalar it guards.  The artifact
    contract (both the explicit gate keys and the fitted ``f_r`` block they are
    derived from) is documented in :mod:`.map_calibration`.
    """
    return _MC.load_gate_correction(store_dir, cell_key)


__all__ = [
    "ARTIFACT_NAME",
    "ARTIFACT_SCHEMA",
    "CellScale",
    "DEFAULT_COV_SCALE",
    "DEFAULT_PEAK_SCALE",
    "DEFAULT_RISK_Z",
    "DEFAULT_W_COV",
    "FlatScale",
    "GATE_CALIBRATION_NAME",
    "IDENTITY_RTOL",
    "MIN_CELL_ROWS",
    "SCALAR_VERSION",
    "identity_matches",
    "load_gate_correction",
]
