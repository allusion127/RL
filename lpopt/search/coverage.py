"""Support-bin coverage (KPI **A3**) — one definition of "in distribution".

``active_frontier_loop_spec_20260903.md`` §4d A3 defines coverage as the
``TrustRegion`` support set ``S``: the ``(feed, e_core-bin)`` pairs carrying at
least ``promote_after`` verified labels — the bins
:meth:`lpopt.search.acquisition.TrustRegion.observe` would have promoted.  A
cell is *in distribution* when its own bin is in ``S``.

This module exists because the mesh readout (``scoping_mesh.py``) shipped
``in_distribution = (feed == 121)``, a CONSTANT that answers a different
question (which feed the study happened to be centred on) and therefore makes
the A3 KPI unmeasurable.  Nothing here reads a model, so the coverage number is
recomputable at any time from ``records.parquet`` alone.

The bin arithmetic is imported from :mod:`lpopt.search.acquisition` rather than
restated, so the coverage set and the trust region can never drift apart.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .acquisition import _e_bin as _acq_e_bin

#: The shipped ``[trust_region]`` defaults (``lpopt/config.py``), repeated only
#: as the signature default of a stand-alone readout that has no deck.
DEFAULT_E_CORE_BAND = 0.05
DEFAULT_PROMOTE_AFTER = 16

Bin = tuple[int, "int | None"]


def e_bin(e_core: float | None, band: float = DEFAULT_E_CORE_BAND) -> int | None:
    """The trust region's own e_core bin index (``None`` for an unknown e_core)."""

    return _acq_e_bin(e_core, float(band))


def bin_counts(
    records: Any,
    *,
    e_core_band: float = DEFAULT_E_CORE_BAND,
) -> Counter[Bin]:
    """``{(feed, e-bin): n_labels}`` over a records frame."""

    counts: Counter[Bin] = Counter()
    if records is None or not len(records):
        return counts
    feeds = records["feed"].tolist()
    e_cores = records["e_core"].tolist()
    for feed, e_core in zip(feeds, e_cores):
        try:
            key = (int(feed), e_bin(e_core, e_core_band))
        except (TypeError, ValueError):
            continue
        counts[key] += 1
    return counts


def support_bins(
    store_dir: str | Path,
    *,
    e_core_band: float = DEFAULT_E_CORE_BAND,
    promote_after: int = DEFAULT_PROMOTE_AFTER,
) -> tuple[set[Bin], Counter[Bin]]:
    """``(S, counts)`` for a record store — ``S`` = bins with ≥ ``promote_after``."""

    from ..data.store import StoreReader

    counts = bin_counts(StoreReader(store_dir).records, e_core_band=e_core_band)
    return ({key for key, n in counts.items() if n >= int(promote_after)}, counts)


def in_distribution(
    feed: int | None,
    e_core: float | None,
    supported: set[Bin],
    *,
    e_core_band: float = DEFAULT_E_CORE_BAND,
) -> bool:
    """Is ``(feed, e_core)``'s bin a supported one?

    ``e_core`` unknown (``None``) admits on the feed alone — the same
    concession :meth:`TrustRegion.in_region` makes, so a row that never recorded
    its enrichment is not silently declared out of distribution.
    """

    if feed is None:
        return False
    key = (int(feed), e_bin(e_core, e_core_band))
    if key in supported:
        return True
    if key[1] is None:
        return any(sfeed == key[0] for sfeed, _ in supported)
    return False


__all__ = [
    "DEFAULT_E_CORE_BAND", "DEFAULT_PROMOTE_AFTER",
    "bin_counts", "e_bin", "in_distribution", "support_bins",
]
