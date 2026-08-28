"""Fuel-design spec + MASTER alias registry + LHS grid sampling (plan 12.1).

A :class:`FuelDesign` is the five design axes of plan section 12.1:

    e1              main UO2 enrichment            [w/o]   {5.0,5.4,5.8,6.2,6.6}
    e2              zoning UO2_2 enrichment        [w/o]   e2/e1 in {0.85,0.92}
    zoning_variant  edge-zoning pin arrangement    z1|z2   (template family)
    gd_wt           Gd2O3 content of the Gd pins   [wt%]   {6,8,10}
    n_gd            number of Gd (UO2G) pins       count   {12,16,20,24}

``type_id`` is a stable, human-readable descriptor (``P<e1x10><e2x10>Z<z>G<gd>N<n>``).
MASTER, however, keys cross-section sets by a 5-character COMP name ``FA_<xx>``
(verified against the ga80 ``MAS_XSL``: every set name is exactly ``FA_`` + a
letter + a digit).  A :class:`DesignRegistry` therefore assigns each design a
stable **2-character alias** (letter+digit, ``R`` avoided so it never collides
with the reflector batch ids ``R2/R3/R4``); the alias is the DeCART product name,
the HGC stem, the ``FA_<alias>`` COMP/XS-set name, and the MASTER batch id.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# design grid (plan 12.1)
# --------------------------------------------------------------------------- #
DESIGN_GRID: dict[str, list] = {
    "e1": [5.0, 5.4, 5.8, 6.2, 6.6],
    "ratio": [0.85, 0.92],          # e2 / e1
    "zoning_variant": ["z1", "z2"],
    "gd_wt": [6.0, 8.0, 10.0],
    "n_gd": [12, 16, 20, 24],
}

#: Coarse enrichment used by the UO2G (Gd carrier) pin — fixed at 4.0 w/o U-235
#: per the reference lattices; not a design axis.
GD_CARRIER_ENR: float = 4.0

_VALID_Z = ("z1", "z2")


@dataclass(frozen=True)
class FuelDesign:
    """One parametric assembly design (the five plan-12.1 axes)."""

    e1: float
    e2: float
    zoning_variant: str
    gd_wt: float
    n_gd: int

    def __post_init__(self) -> None:
        if self.zoning_variant not in _VALID_Z:
            raise ValueError(
                f"zoning_variant must be one of {_VALID_Z}, got {self.zoning_variant!r}"
            )
        if not (0.0 < self.e2 <= self.e1):
            raise ValueError(
                f"require 0 < e2 ({self.e2}) <= e1 ({self.e1}) for corner zoning"
            )
        if self.n_gd <= 0 or self.n_gd % 4 != 0:
            # Gd pins live on 8-fold octant positions; the reference lattices use
            # multiples of 4 (12/16/20/24).
            raise ValueError(f"n_gd must be a positive multiple of 4, got {self.n_gd}")
        if self.gd_wt <= 0.0:
            raise ValueError(f"gd_wt must be positive, got {self.gd_wt}")

    # -- identity ----------------------------------------------------------- #
    @property
    def e1x10(self) -> int:
        return int(round(self.e1 * 10))

    @property
    def e2x10(self) -> int:
        return int(round(self.e2 * 10))

    @property
    def z_digit(self) -> str:
        return self.zoning_variant[-1]          # "1" | "2"

    @property
    def type_id(self) -> str:
        """Stable descriptive id, e.g. ``P5849Z1G08N16`` (plan 12.1)."""
        return (
            f"P{self.e1x10:02d}{self.e2x10:02d}"
            f"Z{self.z_digit}G{int(round(self.gd_wt)):02d}N{self.n_gd:02d}"
        )

    @property
    def key(self) -> tuple[int, int, str, int, int]:
        """Hashable identity over the discretized axes (dedup key)."""
        return (self.e1x10, self.e2x10, self.zoning_variant,
                int(round(self.gd_wt)), self.n_gd)

    @property
    def ratio(self) -> float:
        return self.e2 / self.e1

    def as_dict(self) -> dict:
        return {
            "type_id": self.type_id,
            "e1": self.e1,
            "e2": self.e2,
            "zoning_variant": self.zoning_variant,
            "gd_wt": self.gd_wt,
            "n_gd": self.n_gd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FuelDesign":
        return cls(
            e1=float(d["e1"]),
            e2=float(d["e2"]),
            zoning_variant=str(d["zoning_variant"]),
            gd_wt=float(d["gd_wt"]),
            n_gd=int(d["n_gd"]),
        )


# --------------------------------------------------------------------------- #
# anchors (plan 12.1: "keep existing anchors")
# --------------------------------------------------------------------------- #
#: Canonical designs that mirror existing hardware (5.8/5.1 corner-zoned lattices
#: already in the 5.8_5.1 / 260624 libraries), so a produced HGC can be cross-
#: checked against a preserved FA_*.out.  e2=5.1 does not sit on the strict
#: e2/e1 ratio grid — anchors are explicit, not grid-sampled.
ANCHOR_DESIGNS: list[FuelDesign] = [
    FuelDesign(5.8, 5.1, "z1", 6.0, 12),
    FuelDesign(5.8, 5.1, "z1", 8.0, 16),
    FuelDesign(5.8, 5.1, "z2", 10.0, 20),
    FuelDesign(5.8, 5.1, "z1", 8.0, 24),
]


# --------------------------------------------------------------------------- #
# alias registry
# --------------------------------------------------------------------------- #
#: First-letter pool for the 2-char MASTER alias, preferring ``P..Z`` (reads as a
#: paramA type) then falling back to ``A..O``.  ``R`` is excluded so an alias
#: never collides with the reflector batch ids R2/R3/R4.  25 letters x 10 digits
#: = 250 aliases, enough for the full 240-point grid plus anchors.
_ALIAS_LETTERS = "PQSTUVWXYZABCDEFGHIJKLMNO"
_ALIAS_DIGITS = "0123456789"


def _alias_pool() -> list[str]:
    return [f"{a}{d}" for a in _ALIAS_LETTERS for d in _ALIAS_DIGITS]


class DesignRegistry:
    """Stable, persisted ``type_id <-> 2-char MASTER alias`` map.

    A design keeps its alias for the life of the registry file, so re-running the
    chain (or adding new designs) never renames an existing MASTER set.  New
    designs take the next free alias from :func:`_alias_pool`.
    """

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._by_type: dict[str, str] = dict(mapping or {})
        self._by_alias: dict[str, str] = {a: t for t, a in self._by_type.items()}
        if len(self._by_alias) != len(self._by_type):
            raise ValueError("registry has duplicate aliases")

    # -- persistence -------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> "DesignRegistry":
        p = Path(path)
        if not p.is_file():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data.get("aliases", data))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"aliases": self._by_type}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(p)

    # -- assignment --------------------------------------------------------- #
    def alias(self, design: "FuelDesign | str") -> str:
        """Return the alias for a design, assigning a new one if unseen."""
        type_id = design if isinstance(design, str) else design.type_id
        existing = self._by_type.get(type_id)
        if existing is not None:
            return existing
        for cand in _alias_pool():
            if cand not in self._by_alias:
                self._by_type[type_id] = cand
                self._by_alias[cand] = type_id
                return cand
        raise ValueError("alias pool exhausted (increase _ALIAS_LETTERS)")

    def register_all(self, designs: list["FuelDesign"]) -> dict[str, str]:
        """Assign aliases to every design (in order) and return type_id->alias."""
        for d in designs:
            self.alias(d)
        return {d.type_id: self._by_type[d.type_id] for d in designs}

    def type_id_of(self, alias: str) -> str | None:
        return self._by_alias.get(alias)

    def hgc_name(self, design: "FuelDesign | str") -> str:
        """The renamed DeCART product / MASTER COMP stem, e.g. ``FA_P0``."""
        return f"FA_{self.alias(design)}"

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._by_type)

    def __len__(self) -> int:
        return len(self._by_type)


# --------------------------------------------------------------------------- #
# LHS grid sampling
# --------------------------------------------------------------------------- #
def _balanced_column(levels: list, n: int, rng: random.Random) -> list:
    """A length-``n`` sequence that visits each level as evenly as possible,
    then shuffles — the per-axis Latin-hypercube marginal."""
    seq = [levels[i % len(levels)] for i in range(n)]
    rng.shuffle(seq)
    return seq


def lhs_grid(n: int, seed: int = 0, *, include_anchors: bool = True) -> list[FuelDesign]:
    """Latin-hypercube sample of the plan-12.1 design grid.

    Each of the five axes is stratified independently (balanced column then
    shuffle), so every level of every axis appears with near-uniform marginal
    frequency.  :data:`ANCHOR_DESIGNS` are included first (when
    ``include_anchors``) and always retained.  Duplicates (including anchor
    collisions) are dropped and back-filled by an exhaustive shuffled sweep of
    the full 240-point grid, so exactly ``min(n, 240)`` distinct designs return.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)

    designs: list[FuelDesign] = []
    seen: set = set()
    if include_anchors:
        for d in ANCHOR_DESIGNS:
            if d.key not in seen:
                designs.append(d)
                seen.add(d.key)
    designs = designs[:n]
    seen = {d.key for d in designs}

    remaining = n - len(designs)
    if remaining > 0:
        cols = {
            axis: _balanced_column(levels, remaining, rng)
            for axis, levels in DESIGN_GRID.items()
        }
        for i in range(remaining):
            e1 = cols["e1"][i]
            e2 = round(e1 * cols["ratio"][i], 2)
            d = FuelDesign(e1, e2, cols["zoning_variant"][i], cols["gd_wt"][i], cols["n_gd"][i])
            if d.key in seen:
                continue
            designs.append(d)
            seen.add(d.key)

    # back-fill any shortfall from dedup by sweeping the full grid in a
    # deterministic shuffled order.
    if len(designs) < n:
        full: list[FuelDesign] = []
        for e1 in DESIGN_GRID["e1"]:
            for ratio in DESIGN_GRID["ratio"]:
                for z in DESIGN_GRID["zoning_variant"]:
                    for gd in DESIGN_GRID["gd_wt"]:
                        for ng in DESIGN_GRID["n_gd"]:
                            full.append(FuelDesign(e1, round(e1 * ratio, 2), z, gd, ng))
        rng.shuffle(full)
        for d in full:
            if len(designs) >= n:
                break
            if d.key not in seen:
                designs.append(d)
                seen.add(d.key)

    return designs[:n]


__all__ = [
    "ANCHOR_DESIGNS",
    "DESIGN_GRID",
    "GD_CARRIER_ENR",
    "DesignRegistry",
    "FuelDesign",
    "lhs_grid",
]
