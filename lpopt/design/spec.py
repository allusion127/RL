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

import hashlib
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

#: Rows of the DeCART ``assembly`` octant triangle (lengths 1..8 for a 16x16 FA).
OCTANT_ROWS = 8


# --------------------------------------------------------------------------- #
# Gd octant layout ("1:1;4:1;6:4") <-> tuple of (row, col) octant positions
# --------------------------------------------------------------------------- #
def parse_gd_positions(value) -> tuple[tuple[int, int], ...]:
    """Normalize a Gd octant layout to a sorted tuple of ``(row, col)`` pairs.

    Accepts the ``designs.json`` / plan wire format ``"1:1;4:1;6:4"`` (the same
    grammar as ``6_DeCART_Surrogate/surrogate/features.py:70``) or any iterable of
    2-sequences.  ``None`` / ``""`` -> ``()``.  Positions are returned sorted so a
    layout has ONE canonical spelling (the dedup key and the layout tag both
    depend on that).
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        pairs = []
        for part in text.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                r, c = part.split(":")
                pairs.append((int(r), int(c)))
            except ValueError as exc:
                raise ValueError(
                    f"malformed gd_positions token {part!r} (want '<row>:<col>')"
                ) from exc
    else:
        pairs = []
        for item in value:
            try:
                r, c = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed gd_positions entry {item!r} (want a (row, col) pair)"
                ) from exc
            pairs.append((int(r), int(c)))
    return tuple(sorted(pairs))


def format_gd_positions(positions) -> str:
    """The wire spelling ``"1:1;4:1;6:4"`` of a normalized layout (``""`` if empty)."""
    return ";".join(f"{i}:{j}" for i, j in parse_gd_positions(positions))


def gd_multiplicity(positions) -> int:
    """Full-map Gd pin count of an octant layout: diagonal x4, off-diagonal x8.

    The octant -> quarter -> full expansion truth used by the surrogate feature
    builder (``6_DeCART_Surrogate/surrogate/features.py:85-99``): an octant cell on
    the main diagonal has 4 images in the 16x16 map, an off-diagonal cell has 8.
    """
    return sum(4 if i == j else 8 for i, j in parse_gd_positions(positions))


@dataclass(frozen=True)
class FuelDesign:
    """One parametric assembly design (the five plan-12.1 axes).

    ``gd_positions`` is the OPTIONAL sixth (non-grid) attribute: the Gd octant
    layout this design is realized with.  It defaults to ``None`` = "whatever the
    frozen 0_APR1400 template for ``(gd_wt, n_gd, zoning_variant)`` carries", which
    is exactly what every pre-existing design record means, so every existing
    ``type_id`` / ``key`` / ``as_dict()`` stays byte-identical.  A design that
    names its own layout (an authored on-demand assembly) carries it here.
    """

    e1: float
    e2: float
    zoning_variant: str
    gd_wt: float
    n_gd: int
    gd_positions: tuple[tuple[int, int], ...] | None = None

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
        if self.gd_positions is not None:
            positions = parse_gd_positions(self.gd_positions)
            if not positions:
                raise ValueError("gd_positions is empty; use None for 'template layout'")
            if len(set(positions)) != len(positions):
                raise ValueError(f"gd_positions has duplicate cells: {positions}")
            for i, j in positions:
                if not (0 <= j <= i < OCTANT_ROWS):
                    raise ValueError(
                        f"gd_positions cell {i}:{j} is outside the {OCTANT_ROWS}-row "
                        f"octant triangle (require 0 <= col <= row < {OCTANT_ROWS})"
                    )
            realized = gd_multiplicity(positions)
            if realized != self.n_gd:
                raise ValueError(
                    f"gd_positions {format_gd_positions(positions)} realizes {realized} "
                    f"Gd pins (diagonal x4 / off-diagonal x8) but n_gd is {self.n_gd}"
                )
            object.__setattr__(self, "gd_positions", positions)

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
    def gd_layout(self) -> str | None:
        """The wire spelling of :attr:`gd_positions` (``None`` = template layout)."""
        if self.gd_positions is None:
            return None
        return format_gd_positions(self.gd_positions)

    @property
    def layout_tag(self) -> str | None:
        """``"L" + sha1(layout)[:3]`` — the disambiguator for a NAMED layout.

        ``None`` when the design uses its template's frozen layout.  It is NOT
        appended to :attr:`type_id` automatically: the 37 shipped paramA ids (T3/T5/T6
        already carry OPEN layouts and no tag) must stay byte-identical, and the
        slice-Z prereg registers ``P5547Z1G08N20`` untagged.  Use
        :attr:`type_id_tagged` explicitly when two layouts of the same
        ``(e1, e2, z, gd_wt, n_gd)`` must coexist as distinct MASTER types.
        """
        layout = self.gd_layout
        if not layout:
            return None
        return "L" + hashlib.sha1(layout.encode("ascii")).hexdigest()[:3]

    @property
    def type_id_tagged(self) -> str:
        """:attr:`type_id` with :attr:`layout_tag` appended when a layout is named."""
        tag = self.layout_tag
        return self.type_id if tag is None else f"{self.type_id}{tag}"

    @property
    def key(self) -> tuple:
        """Hashable identity over the discretized axes (dedup key).

        A design with no named layout keeps the historical 5-tuple, so every
        existing dedup / registry path is byte-identical.  A design that NAMES a Gd
        layout appends its wire spelling as a sixth element: two layouts over the
        same five axes are genuinely different lattices (different FF, different
        HGC) and must not collapse onto one another.
        """
        base = (self.e1x10, self.e2x10, self.zoning_variant,
                int(round(self.gd_wt)), self.n_gd)
        if self.gd_positions is None:
            return base
        return base + (self.gd_layout,)

    @property
    def ratio(self) -> float:
        return self.e2 / self.e1

    def as_dict(self) -> dict:
        rec = {
            "type_id": self.type_id,
            "e1": self.e1,
            "e2": self.e2,
            "zoning_variant": self.zoning_variant,
            "gd_wt": self.gd_wt,
            "n_gd": self.n_gd,
        }
        if self.gd_positions is not None:
            rec["gd_positions"] = self.gd_layout
        return rec

    @classmethod
    def from_dict(cls, d: dict) -> "FuelDesign":
        return cls(
            e1=float(d["e1"]),
            e2=float(d["e2"]),
            zoning_variant=str(d["zoning_variant"]),
            gd_wt=float(d["gd_wt"]),
            n_gd=int(d["n_gd"]),
            gd_positions=(parse_gd_positions(d["gd_positions"])
                          if d.get("gd_positions") else None),
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

    def __init__(self, mapping: dict[str, str] | None = None,
                 designs: dict[str, dict] | None = None) -> None:
        self._by_type: dict[str, str] = dict(mapping or {})
        self._by_alias: dict[str, str] = {a: t for t, a in self._by_type.items()}
        if len(self._by_alias) != len(self._by_type):
            raise ValueError("registry has duplicate aliases")
        #: type_id -> the design record it was assigned for.  Populated by
        #: :meth:`alias` for every design in THIS process, and reloaded from disk
        #: for the layout-bearing designs (see :meth:`save`).  This is what makes
        #: the same ``type_id`` with a different design tuple / Gd layout a hard
        #: error instead of a silent alias reuse (R23).
        self._designs: dict[str, dict] = {
            str(t): dict(rec) for t, rec in (designs or {}).items()
        }

    # -- persistence -------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path, *,
             designs_manifest: str | Path | None = None) -> "DesignRegistry":
        """Load ``registry.json``, hydrating the design records that guard R23.

        The live ``registry.json`` is a bare ``{"aliases": ...}`` document, so
        ``_designs`` would be empty after a load and :meth:`_check_design` would
        wave through a NEW lattice that happens to quantize onto a shipped
        ``type_id`` — the guard would only ever fire inside one process.  The
        sibling ``designs.json`` (the package manifest) carries the records, so it
        is read when present.  ``designs_manifest`` overrides that sibling;
        ``False``-y paths and a missing file simply leave ``_designs`` as before.
        Nothing here changes what :meth:`save` writes.
        """
        p = Path(path)
        if not p.is_file():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return cls(data)
        designs = data.get("designs")
        reg = cls(data.get("aliases", data),
                  designs if isinstance(designs, dict) else None)
        if designs_manifest is None:
            cand = p.with_name("designs.json")
            designs_manifest = cand if cand.is_file() else None
        if designs_manifest:
            reg._hydrate_from_manifest(designs_manifest)
        return reg

    def _hydrate_from_manifest(self, manifest: str | Path) -> None:
        """Seed ``_designs`` from a ``designs.json`` package manifest.

        Records are normalized through ``FuelDesign.from_dict(...).as_dict()`` so
        the manifest's extra columns (alias, gd_u_enr, lat1600_id, provenance, …)
        cannot make :meth:`_check_design`'s equality compare spuriously fail.  An
        in-process record already recorded for a ``type_id`` wins (``setdefault``).
        """
        doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
        for rec in doc.get("designs", []):
            try:
                norm = FuelDesign.from_dict(rec).as_dict()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"designs manifest {manifest} has an unusable record "
                    f"{rec.get('type_id', rec)!r}: {exc}"
                ) from exc
            self._designs.setdefault(str(norm["type_id"]), norm)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Only layout-bearing designs are persisted: a registry that has never
        # seen a named Gd layout writes exactly the historical {"aliases": ...}
        # document, byte for byte.  Readers take only "aliases"
        # (search/assets.py:551), so the extra section is inert for them.
        payload: dict = {"aliases": self._by_type}
        layout_recs = {t: rec for t, rec in self._designs.items()
                       if rec.get("gd_positions")}
        if layout_recs:
            payload["designs"] = layout_recs
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(p)

    # -- assignment --------------------------------------------------------- #
    def _check_design(self, design: "FuelDesign") -> None:
        """Hard-fail when ``type_id`` is already bound to a DIFFERENT design.

        ``type_id`` quantizes the enrichments to 0.1 w/o and says nothing at all
        about the Gd layout, so two genuinely different lattices can spell the same
        id (``e2`` 4.6750 vs 4.70 both fold to ``47``; the frozen and the open
        layout of one ``(gd_wt, n_gd, z)`` are indistinguishable).  Reusing an alias
        across them would ship one lattice's cross sections under the other's name.
        """
        rec = design.as_dict()
        prev = self._designs.get(design.type_id)
        if prev is None:
            self._designs[design.type_id] = rec
            return
        if prev != rec:
            diff = sorted(k for k in set(prev) | set(rec)
                          if prev.get(k) != rec.get(k))
            raise ValueError(
                f"registry conflict: type_id {design.type_id!r} is already bound to "
                f"a different design (differs on {diff}: recorded {prev}, requested "
                f"{rec}); give the new lattice a distinct type_id (see "
                f"FuelDesign.type_id_tagged) before assigning an alias"
            )

    def alias(self, design: "FuelDesign | str") -> str:
        """Return the alias for a design, assigning a new one if unseen."""
        if not isinstance(design, str):
            self._check_design(design)
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

    def design_of(self, type_id: str) -> dict | None:
        """The design record this registry has bound to ``type_id`` (``None`` if none)."""
        rec = self._designs.get(str(type_id))
        return dict(rec) if rec is not None else None

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
    "OCTANT_ROWS",
    "DesignRegistry",
    "FuelDesign",
    "format_gd_positions",
    "gd_multiplicity",
    "lhs_grid",
    "parse_gd_positions",
]
