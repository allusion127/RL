"""Delivery-candidate selection — the flatness program's §2.2 rule (decision D2).

**This is NOT part of the search objective.**  The search objective is flatness
(``node_peak`` primary, ``map_cov`` secondary) and contains no F_r.  This module
runs strictly downstream of it, on rows that already exist, to answer a different
question: *of the flat patterns we produced, which one do we actually hand over
for licensing review?*  Decision D2 approved using predicted F_r margin as the
ranking key at THIS stage and only at this stage.

The rule (program §2.2)
-----------------------
::

    1) restrict to the cell's FLAT BAND: node_peak within-cell percentile 0.10-0.40
    2) inside the band, rank by compliance_margin = 1.55 - (bias-corrected F_r)
    3) the top candidates go to SDM / MTC / axial confirmation (outside the loop)

Step 1 is the load-bearing one and it is a REJECTION rule: **the flattest point is
not what gets delivered.**  Measured over the store, the within-cell flatness
percentile of rows that actually meet F_r <= 1.55 has median 0.22 for
``node_peak`` and 0.33 for ``map_cov`` — compliant patterns live in the *lower
band*, not at the extreme.  The store's minimum F_r is 1.508 against a 1.55
limit, i.e. the entire fleet-wide headroom is 0.042, so walking to the flatness
extreme spends margin the design does not have.  A percentile floor of 0.10 makes
that explicit: a candidate flatter than 10% of its cell is EXCLUDED, by name, not
by accident.

Everything here is pure (rows in, rows out) so the rule is unit-testable without
a model, a store or a campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: The licensing limit F_r must respect.  It is a REPORTING / DELIVERY constant
#: here — never a search objective and never a feasibility definition inside the
#: flatness campaign (program §10 STOP list).
LICENSING_FR_LIMIT = 1.55

#: The delivery flat band, as within-cell ``node_peak`` percentiles (program §2.2).
#: The floor is the "do not deliver the flattest point" rule.
BAND_LO = 0.10
BAND_HI = 0.40

#: A cell needs this many candidates before its percentile band means anything.
MIN_BAND_ROWS = 5


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def compliance_margin(f_r: Any, *, bias: float = 0.0,
                      limit: float = LICENSING_FR_LIMIT) -> float | None:
    """``limit − (f_r − bias)`` — headroom to the licensing limit (>= 0 compliant).

    ``bias`` is the per-cell F_r prediction bias (predicted minus actual) when the
    input is a PREDICTION; pass 0 for a MASTER-verified F_r, which needs no
    correction.  ``None`` in -> ``None`` out, never a fabricated margin.
    """
    v = _num(f_r)
    if v is None:
        return None
    b = _num(bias) or 0.0
    return float(limit) - (v - b)


def within_cell_percentile(values: Sequence[float]) -> np.ndarray:
    """Fractional rank in ``[0, 1)`` of each finite value (NaN -> NaN).

    Ascending, so a LOW ``node_peak`` (flat) gets a LOW percentile — which is what
    the band bounds are written against.  Ties take their average rank so a cell
    of identical candidates is not silently split by input order.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan, dtype=float)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n == 0:
        return out
    vals = v[ok]
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # average rank within tie groups
    uniq, inverse = np.unique(vals, return_inverse=True)
    for k in range(uniq.size):
        m = inverse == k
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    out[ok] = ranks / float(n)
    return out


@dataclass(frozen=True)
class DeliveryCandidate:
    """One ranked delivery candidate (a view over the source row)."""

    record_id: str | None
    node_peak: float | None
    map_cov: float | None
    f_r: float | None
    compliance_margin: float | None
    peak_percentile: float | None
    in_band: bool
    reason: str
    row: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "node_peak": self.node_peak,
            "map_cov": self.map_cov,
            "f_r": self.f_r,
            "compliance_margin": self.compliance_margin,
            "peak_percentile": self.peak_percentile,
            "in_band": self.in_band,
            "reason": self.reason,
        }


@dataclass
class DeliveryReport:
    """The outcome of one :func:`select_delivery` pass."""

    ranked: list[DeliveryCandidate]      # in-band, best compliance margin first
    excluded: list[DeliveryCandidate]    # why each dropped out
    n_rows: int = 0
    n_scored: int = 0
    band: tuple[float, float] = (BAND_LO, BAND_HI)
    banded: bool = True                  # False -> too few rows, band not applied

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows, "n_scored": self.n_scored,
            "band": list(self.band), "banded": self.banded,
            "ranked": [c.as_dict() for c in self.ranked],
            "excluded": [c.as_dict() for c in self.excluded],
        }


def select_delivery(
    rows: Iterable[Mapping[str, Any]],
    *,
    band: tuple[float, float] = (BAND_LO, BAND_HI),
    fr_bias: float = 0.0,
    limit: float = LICENSING_FR_LIMIT,
    peak_key: str = "node_peak",
    cov_key: str = "map_cov",
    fr_key: str = "f_r",
    min_band_rows: int = MIN_BAND_ROWS,
    top_k: int | None = None,
) -> DeliveryReport:
    """Rank delivery candidates by program §2.2 (decision D2).

    ``rows`` are candidates from ONE cell (the percentile band is within-cell by
    construction — a campaign is fixed-cell, and a multi-cell caller groups first).
    ``fr_bias`` is that cell's F_r bias correction when ``fr_key`` holds a
    PREDICTION; leave it 0 for verified rows.

    Returns every candidate, partitioned: ``ranked`` are in-band and ordered by
    descending :func:`compliance_margin` (largest licensing headroom first),
    ``excluded`` carry the reason they dropped out — ``"flatter than the band"``
    (the §2.2 rule that the flattest point is not delivered), ``"less flat than
    the band"``, ``"no flatness label"``, ``"no F_r"``.

    Fewer than ``min_band_rows`` scorable rows: the band is NOT applied (a
    percentile over 3 rows is noise), ``banded`` is False, and everything scorable
    is ranked by margin.  That is a reported degradation, not a silent one.
    """

    rows = [dict(r) for r in rows]
    lo, hi = float(band[0]), float(band[1])
    peaks = np.array([_num(r.get(peak_key)) if _num(r.get(peak_key)) is not None
                      else np.nan for r in rows], dtype=float)
    pct = within_cell_percentile(peaks)
    n_scored = int(np.isfinite(peaks).sum())
    banded = n_scored >= int(min_band_rows)

    ranked: list[DeliveryCandidate] = []
    excluded: list[DeliveryCandidate] = []
    for i, row in enumerate(rows):
        peak = _num(row.get(peak_key))
        cov = _num(row.get(cov_key))
        f_r = _num(row.get(fr_key))
        margin = compliance_margin(f_r, bias=fr_bias, limit=limit)
        p = float(pct[i]) if np.isfinite(pct[i]) else None
        rid = row.get("record_id")
        rid = None if rid is None else str(rid)

        def _mk(in_band: bool, reason: str) -> DeliveryCandidate:
            return DeliveryCandidate(
                record_id=rid, node_peak=peak, map_cov=cov, f_r=f_r,
                compliance_margin=margin, peak_percentile=p,
                in_band=in_band, reason=reason, row=row,
            )

        if peak is None:
            excluded.append(_mk(False, "no flatness label"))
            continue
        if margin is None:
            excluded.append(_mk(False, "no F_r"))
            continue
        if banded and p is not None:
            if p < lo:
                # THE §2.2 RULE: the flattest point is not what gets delivered.
                excluded.append(_mk(False, "flatter than the band"))
                continue
            if p > hi:
                excluded.append(_mk(False, "less flat than the band"))
                continue
        ranked.append(_mk(True, "in band"))

    # NOTE the explicit None test: ``margin or -inf`` would send a candidate
    # sitting EXACTLY on the 1.55 limit (margin 0.0) to the bottom of the list.
    ranked.sort(key=lambda c: (
        -(c.compliance_margin if c.compliance_margin is not None else -np.inf),
        str(c.record_id or "")))
    if top_k is not None:
        # a truncated tail is NOT "excluded" — it passed every rule, it just did
        # not make the cut; ``excluded`` stays reserved for rule failures.
        ranked = ranked[: int(top_k)]
    return DeliveryReport(ranked=ranked, excluded=excluded, n_rows=len(rows),
                          n_scored=n_scored, band=(lo, hi), banded=banded)


__all__ = [
    "BAND_HI",
    "BAND_LO",
    "DeliveryCandidate",
    "DeliveryReport",
    "LICENSING_FR_LIMIT",
    "MIN_BAND_ROWS",
    "compliance_margin",
    "select_delivery",
    "within_cell_percentile",
]
