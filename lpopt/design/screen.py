"""The DeCART lattice-surrogate bridge for on-demand assembly design (task #4/#6).

Four jobs, none of which existed before:

1. **``FuelDesign`` -> surrogate input row.**  :func:`design_to_row` maps the
   lpopt five-axis design plus a Gd layout onto the physical-unit dict the
   surrogate's ``predict_cases`` wants (``SURROGATE_USAGE.md`` §7.2).

2. **The ``z1 <-> PB`` / ``z2 <-> PA`` mapping, asserted at import.**  See
   :data:`Z_TO_PATTERN`.  This is *counter-intuitive* and there is no error and
   no warning if you get it wrong -- MOCHA's default is ``surrogate_pattern =
   "PA"`` (``2_LP/MOCHA/config.py:349-358``), so screening a ``z1`` design with
   the default silently evaluates the wrong lattice family.

3. **A fuel-type -> design catalog loader** that reads
   ``data/design/package/designs.json`` rather than hard-coding a table
   (:func:`load_design_catalog`).

4. **Diversity / dedup / role-pair contrast gating** as pure functions
   (task #6): :func:`dedup_exact`, :func:`dedup_near`, :func:`greedy_maxmin`,
   :func:`role_pair_gate`.

**Bounds are enforced; the grid is not.**  ``T3``/``T4`` were screened at
``du = 0.75`` -- off the surrogate's 0.1 ``du`` grid -- and then passed the
DeCART cross-check at < 100 pcm (``opmodel/OPSCREEN.md:170-172``).  Extrapolating
past a *bound*, by contrast, collapses: ``n_gd = 28`` moved BOC k by
**+1,750 pcm** (``SURROGATE_USAGE.md:143``).  So :func:`validate_bounds` refuses
out-of-bounds designs and accepts off-grid ones.  The surrogate's own
``validate_design`` cannot be reused for selection because its ``_snap_to_grid``
(``6_DeCART_Surrogate/surrogate/predict.py:109-121``) rejects off-grid values,
which would retroactively disqualify T3/T4.

**Graceful degradation.**  The surrogate checkpoints are an optional, external
asset.  Importing this module never requires them; :func:`predict` raises
:class:`NotAvailable` with the precise blocker when they are missing.  A missing
or invalid prediction is *advisory* -- it must not be turned into a hard reject.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# 1.  zoning  <->  surrogate pattern family                    (R17, closed)
# --------------------------------------------------------------------------- #
# The surrogate's two zoning families differ ONLY in octant row 7 (the last row
# of the 1/8 lower triangle), ``6_DeCART_Surrogate/surrogate/predict.py:67-71``:
#
#     ZONING_COMMON     = {(1,0), (3,2), (4,2), (5,3), (5,4)}
#     PA = ZONING_COMMON | {(7, c) for c in range(8)}     -> row 7 all zoning
#     PB = ZONING_COMMON | {(7, 6), (7, 7)}               -> row 7 zoning only at 6,7
#
# The lpopt template decks put the *opposite* labels on those two rows:
#
#     0_APR1400/5.8_5.1/FA/IGD_16/6_16_z1/dec_FA_A01.inp:97   "1 1 1 1 1 1 2 2"
#     0_APR1400/5.8_5.1/FA/IGD_16/6_16_z2/dec_FA_A02.inp:97   "2 2 2 2 2 2 2 2"
#
# cell id 2 == UO2_2 (the zoning enrichment).  So z1's row 7 carries zoning at
# columns 6 and 7 only  ->  z1 IS PB, and z2's row 7 is entirely zoning
# ->  z2 IS PA.  **This is the reverse of what the names suggest.**
_OCTANT_ROW7: dict[str, tuple[int, ...]] = {
    # zoning_variant -> octant row 7 cell ids, verbatim from the template decks
    "z1": (1, 1, 1, 1, 1, 1, 2, 2),   # dec_FA_A01.inp:97
    "z2": (2, 2, 2, 2, 2, 2, 2, 2),   # dec_FA_A02.inp:97
}

#: Guide-tube / instrument cells that no design may occupy
#: (``predict.py:66`` ``FIXED``).
FIXED_CELLS: dict[tuple[int, int], int] = {
    (0, 0): 9, (3, 3): 6, (4, 3): 8, (4, 4): 9,
}

#: Zoning cells shared by both families (``predict.py:67``).
ZONING_COMMON: frozenset[tuple[int, int]] = frozenset(
    {(1, 0), (3, 2), (4, 2), (5, 3), (5, 4)}
)

#: ``predict.py:68-71``.
ZONING_BY_PATTERN: dict[str, frozenset[tuple[int, int]]] = {
    "PA": ZONING_COMMON | frozenset((7, c) for c in range(8)),
    "PB": ZONING_COMMON | frozenset({(7, 6), (7, 7)}),
}

#: The 10 admissible Gd (UO2G) octant positions (``SURROGATE_USAGE.md`` §2 B,
#: ``predict.py:52`` ``GD_CANDIDATES``).  Diagonal positions carry 4 pins in the
#: full assembly, off-diagonal 8.
GD_CANDIDATES: tuple[tuple[int, int], ...] = (
    (1, 1), (2, 0), (2, 2), (3, 1), (4, 1), (5, 1), (5, 2), (5, 5), (6, 3), (6, 4),
)

#: Minimum Chebyshev separation between two Gd positions (``predict.py:117``).
MIN_CHEB: int = 2


def _row7_pattern(row7: Sequence[int]) -> str:
    """Which surrogate family does this octant row 7 encode?"""
    zoned = frozenset((7, c) for c, v in enumerate(row7) if int(v) == 2)
    for name, cells in ZONING_BY_PATTERN.items():
        if zoned == frozenset(p for p in cells if p[0] == 7):
            return name
    raise ValueError(f"octant row 7 {tuple(row7)} matches no zoning family")


#: ``zoning_variant -> surrogate pattern``.  Derived, not asserted by hand.
Z_TO_PATTERN: dict[str, str] = {z: _row7_pattern(r) for z, r in _OCTANT_ROW7.items()}

# Import-time guard.  If a future edit flips this mapping, every screen run is
# silently evaluating the wrong lattice family, so fail loudly at import.  This
# is an unconditional ``raise``, not an ``assert``: under ``python -O`` an
# assert is compiled out and the guard would vanish exactly when it is needed.
if Z_TO_PATTERN != {"z1": "PB", "z2": "PA"}:
    raise RuntimeError(
        "z1<->PB / z2<->PA mapping broken: got "
        f"{Z_TO_PATTERN!r}; see dec_FA_A01.inp:97 and dec_FA_A02.inp:97")

#: Reverse map, for reading a surrogate manifest back into lpopt terms.
PATTERN_TO_Z: dict[str, str] = {v: k for k, v in Z_TO_PATTERN.items()}


def pattern_for(zoning_variant: str) -> str:
    """``"z1" -> "PB"``, ``"z2" -> "PA"``.

    Use this everywhere instead of a literal.  Passing ``"PA"`` for a ``z1``
    design produces no error and no warning from the surrogate -- it just
    evaluates a different assembly.
    """
    try:
        return Z_TO_PATTERN[str(zoning_variant)]
    except KeyError:
        raise ValueError(
            f"unknown zoning_variant {zoning_variant!r}; "
            f"expected one of {sorted(Z_TO_PATTERN)}"
        ) from None


def verify_zoning_against_templates(apr1400_root: str | Path,
                                    line_no: int = 97) -> dict[str, str]:
    """Re-derive :data:`Z_TO_PATTERN` from the two template decks on disk.

    Returns the derived mapping; raises :class:`RuntimeError` if it disagrees
    with the module constant.  ``apr1400_root`` is the ``0_APR1400`` directory.
    Raises :class:`FileNotFoundError` when the templates are not staged (they
    are not present on HOST_238), so callers/tests should skip in that case.
    """
    root = Path(apr1400_root)
    decks = {
        "z1": root / "5.8_5.1" / "FA" / "IGD_16" / "6_16_z1" / "dec_FA_A01.inp",
        "z2": root / "5.8_5.1" / "FA" / "IGD_16" / "6_16_z2" / "dec_FA_A02.inp",
    }
    derived: dict[str, str] = {}
    for z, deck in decks.items():
        if not deck.is_file():
            raise FileNotFoundError(str(deck))
        lines = deck.read_text(encoding="ascii", errors="strict").splitlines()
        row = [int(tok) for tok in lines[line_no - 1].split()]
        if len(row) != 8:
            raise ValueError(
                f"{deck}:{line_no} is not the 8-wide octant row: {row!r}")
        derived[z] = _row7_pattern(row)
    if derived != Z_TO_PATTERN:
        raise RuntimeError(
            f"template decks say {derived!r} but Z_TO_PATTERN is "
            f"{Z_TO_PATTERN!r}")
    return derived


# --------------------------------------------------------------------------- #
# 2.  bounds (enforced) vs grid (not enforced)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SurrogateBounds:
    """The surrogate's **valid design space** (``SURROGATE_USAGE.md`` §1).

    Bounds only.  The step sizes quoted alongside them in the usage note are
    the *sampling* grid of the training set, not a validity condition -- see
    the module docstring.
    """

    u_high: tuple[float, float] = (5.00, 7.00)
    du: tuple[float, float] = (0.40, 0.80)
    gd_u: tuple[float, float] = (3.00, 4.30)
    #: The three template Gd families lpopt owns decks for.  The surrogate
    #: itself also accepts 7 and 9 (``predict.py:113`` ``GD_WT_VALUES``), but
    #: lpopt has no base deck for them, so they are closed for round 1.
    gd_wt: tuple[int, ...] = (6, 8, 10)
    #: ``SURROGATE_USAGE.md:143`` -- "n_gd in {12,16,20,24} only is trusted"
    #: (n_gd = 28 measured BOC k +1,750 pcm).  The low-Gd extension
    #: {0, 4, 8} exists in the model but has zero production history here.
    n_gd: tuple[int, ...] = (12, 16, 20, 24)
    min_cheb: int = MIN_CHEB


BOUNDS = SurrogateBounds()

#: Fixed Gd-carrier U enrichment (``lpopt.design.spec.GD_CARRIER_ENR``).
GD_CARRIER_ENR: float = 4.0


def parse_gd_positions(text: str | Iterable[tuple[int, int]]
                       ) -> tuple[tuple[int, int], ...]:
    """``"1:1;4:1;6:4"`` -> ``((1,1),(4,1),(6,4))``.  Idempotent on tuples."""
    if not isinstance(text, str):
        return tuple((int(r), int(c)) for r, c in text)
    out: list[tuple[int, int]] = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        r, c = part.split(":")
        out.append((int(r), int(c)))
    return tuple(out)


def format_gd_positions(positions: Iterable[tuple[int, int]]) -> str:
    """``((1,1),(4,1))`` -> ``"1:1;4:1"`` (the surrogate manifest format)."""
    return ";".join(f"{r}:{c}" for r, c in positions)


def pin_multiplicity(pos: tuple[int, int]) -> int:
    """4 pins for a diagonal octant cell, 8 otherwise (``predict.py:80-81``)."""
    return 4 if pos[0] == pos[1] else 8


def cheb_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def validate_layout(positions: Iterable[tuple[int, int]],
                    n_gd: int,
                    *, min_cheb: int = MIN_CHEB) -> list[str]:
    """Layout-only checks: candidate set, duplicates, Chebyshev, pin census."""
    pos = tuple(positions)
    errs: list[str] = []
    bad = [p for p in pos if p not in GD_CANDIDATES]
    if bad:
        errs.append(f"gd_positions {bad} not in the 10 candidate positions")
    if len(set(pos)) != len(pos):
        errs.append("gd_positions contains duplicates")
    for a, b in combinations(pos, 2):
        d = cheb_distance(a, b)
        if d < min_cheb:
            errs.append(f"gd_positions {a} and {b} violate Chebyshev >= "
                        f"{min_cheb} (got {d})")
    total = sum(pin_multiplicity(p) for p in pos)
    if total != int(n_gd):
        errs.append(f"n_gd={n_gd} does not match the summed placement "
                    f"multiplicity {total} (4 if r==c else 8 per position)")
    return errs


def validate_bounds(row: dict, *, bounds: SurrogateBounds = BOUNDS,
                    require_assembly_rows: bool = True) -> list[str]:
    """Bounds-only validation of a surrogate input row.  Returns error strings.

    Deliberately **does not** check the sampling grid: ``du = 0.75`` (T3/T4)
    must pass, ``du = 0.825`` (the OPSCREEN draft's ``u5.50/4.6750``) must not.

    It *does* check ``assembly_rows``, because bypassing ``validate_design``
    also bypasses the reconstruction the surrogate relies on; see
    :func:`design_to_row`.  Set ``require_assembly_rows=False`` only to check
    the numeric axes of a partially built row.
    """
    errs: list[str] = []
    for key in ("u_high", "du", "gd_u"):
        lo, hi = getattr(bounds, key)
        try:
            v = float(row[key])
        except (KeyError, TypeError, ValueError):
            errs.append(f"missing or non-numeric {key}")
            continue
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            errs.append(f"{key}={v:g} outside bounds [{lo:g}, {hi:g}]")
    for key in ("gd_wt", "n_gd"):
        allowed = getattr(bounds, key)
        try:
            v = float(row[key])
        except (KeyError, TypeError, ValueError):
            errs.append(f"missing or non-numeric {key}")
            continue
        if abs(v - round(v)) > 1e-9 or int(round(v)) not in allowed:
            errs.append(f"{key}={row[key]!r} must be one of {allowed}")
    if "pattern" in row and row["pattern"] not in ZONING_BY_PATTERN:
        errs.append(f"pattern={row['pattern']!r} not in "
                    f"{sorted(ZONING_BY_PATTERN)}")
    if not errs:
        errs.extend(validate_layout(parse_gd_positions(row.get("gd_positions", "")),
                                    int(round(float(row["n_gd"]))),
                                    min_cheb=bounds.min_cheb))
    if not errs and require_assembly_rows:
        # features.py:145 and :341 subscript this key with no fallback.
        want = build_assembly_rows(row["pattern"],
                                   parse_gd_positions(row.get("gd_positions", "")))
        have = row.get("assembly_rows")
        if not have:
            errs.append(
                "missing assembly_rows -- the surrogate feature builders "
                "require it (features.py:145,341); build it with "
                "build_assembly_rows(pattern, gd_positions)")
        elif str(have) != want:
            errs.append(
                "assembly_rows disagrees with (pattern, gd_positions): "
                f"got {have!r}, expected {want!r}")
    return errs


# --------------------------------------------------------------------------- #
# 3.  FuelDesign -> surrogate row
# --------------------------------------------------------------------------- #
def build_assembly_rows(pattern: str,
                        gd_positions: Iterable[tuple[int, int]]) -> str:
    """The 1/8 lower-triangle map as the manifest ``"v v|v v v|..."`` string.

    Port of ``predict.py:83-102`` so a caller can hand the surrogate an explicit
    map (and so tests can assert the octant without the surrogate installed).
    """
    gd = set(parse_gd_positions(gd_positions))
    zoning = ZONING_BY_PATTERN[pattern]
    rows: list[str] = []
    for r in range(8):
        row: list[int] = []
        for c in range(r + 1):
            pos = (r, c)
            if pos in FIXED_CELLS:
                row.append(FIXED_CELLS[pos])
            elif pos in gd:
                row.append(3)
            elif pos in zoning:
                row.append(2)
            else:
                row.append(1)
        rows.append(" ".join(str(v) for v in row))
    return "|".join(rows)


def design_to_row(design: Any,
                  gd_positions: str | Iterable[tuple[int, int]] | None = None,
                  *, gd_u: float | None = None) -> dict:
    """``FuelDesign`` (or an equivalent mapping) -> surrogate input dict.

    ``SURROGATE_USAGE.md`` §7.2 schema, physical units, no normalisation.
    ``design`` may be a :class:`lpopt.design.spec.FuelDesign` or any object /
    mapping exposing ``e1``, ``e2``, ``zoning_variant``, ``gd_wt``, ``n_gd``
    (and optionally ``gd_positions``).

    The ``pattern`` field is filled from :func:`pattern_for` -- never from a
    caller-supplied literal, and never from a default.

    ``assembly_rows`` is emitted **unconditionally**.  Both surrogate feature
    builders subscript it without a fallback (``features.py:145`` in
    ``build_case_features``, ``features.py:341`` in ``build_batch_features``),
    and the reconstruction promised by ``SURROGATE_USAGE.md`` §7.2 lives inside
    ``predict.py:245 validate_design`` -- the function this module deliberately
    bypasses in order to admit off-grid ``du``.  A row without the key raises
    ``KeyError`` inside the surrogate after torch and the checkpoints have
    loaded, so it is built here where it is cheap and pure.
    """
    if isinstance(design, dict):
        get = design.get
    else:
        def get(k, default=None):
            return getattr(design, k, default)

    e1 = float(get("e1"))
    e2 = float(get("e2"))
    z = str(get("zoning_variant"))
    positions = gd_positions if gd_positions is not None else get("gd_positions")
    if positions is None:
        raise ValueError(
            "gd_positions is required: the Gd layout drives the pin map, so a "
            "design tuple alone does not determine the surrogate input "
            "(SURROGATE_USAGE.md sec 7.2)")
    pos = parse_gd_positions(positions)
    row = {
        "u_high": e1,
        "du": round(e1 - e2, 10),
        "u_low": e2,
        "gd_u": float(GD_CARRIER_ENR if gd_u is None else gd_u),
        "gd_wt": int(round(float(get("gd_wt")))),
        "n_gd": int(round(float(get("n_gd")))),
        "gd_positions": format_gd_positions(pos),
        "pattern": pattern_for(z),
        "zoning": z,
    }
    row["assembly_rows"] = build_assembly_rows(row["pattern"], pos)
    return row


# --------------------------------------------------------------------------- #
# 4.  fuel-type -> design catalog
# --------------------------------------------------------------------------- #
@dataclass
class CatalogEntry:
    type_id: str
    e1: float
    e2: float
    zoning_variant: str
    gd_wt: float
    n_gd: int
    alias: str | None = None
    gd_u_enr: float = GD_CARRIER_ENR
    gd_positions: tuple[tuple[int, int], ...] | None = None
    extra: dict = field(default_factory=dict)

    @property
    def pattern(self) -> str:
        return pattern_for(self.zoning_variant)

    def to_row(self, **kw) -> dict:
        if self.gd_positions is None:
            raise NotAvailable(
                f"{self.type_id}: catalog row carries no gd_positions, so no "
                "surrogate row can be built (task #13 promotes the field to "
                "required for on-demand types)")
        return design_to_row(self, self.gd_positions, gd_u=self.gd_u_enr, **kw)


@dataclass
class Catalog:
    library_id: str
    entries: dict[str, CatalogEntry]
    warnings: list[str] = field(default_factory=list)
    source: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, type_id: str) -> CatalogEntry:
        return self.entries[type_id]

    def by_alias(self, alias: str) -> CatalogEntry:
        for e in self.entries.values():
            if e.alias == alias:
                return e
        raise KeyError(alias)

    @property
    def with_layout(self) -> list[CatalogEntry]:
        return [e for e in self.entries.values() if e.gd_positions is not None]


_CATALOG_KNOWN = {"type_id", "e1", "e2", "zoning_variant", "gd_wt", "n_gd",
                  "alias", "gd_u_enr", "gd_positions"}


def load_design_catalog(designs_json: str | Path) -> Catalog:
    """Read the fuel-type -> design catalog from ``designs.json``.

    The catalog is **data, not code**: the (C) adapter has been dead in
    production because ``config_apr1400.yaml`` carries no
    ``surrogate_fuel_catalog`` key at all, so nothing could cross the bridge.
    Rows without ``gd_positions`` (33 of the 37 live paramA rows) are loaded but
    recorded in :attr:`Catalog.warnings` -- they can be ranked, not screened.
    """
    p = Path(designs_json)
    data = json.loads(p.read_text(encoding="utf-8"))
    entries: dict[str, CatalogEntry] = {}
    warnings: list[str] = []
    for d in data.get("designs", []):
        tid = str(d["type_id"])
        gp = d.get("gd_positions")
        pos = parse_gd_positions(gp) if gp else None
        if pos is None:
            warnings.append(
                f"{tid}: no gd_positions in the catalog -- surrogate screening "
                "unavailable for this type")
        entry = CatalogEntry(
            type_id=tid,
            e1=float(d["e1"]),
            e2=float(d["e2"]),
            zoning_variant=str(d["zoning_variant"]),
            gd_wt=float(d["gd_wt"]),
            n_gd=int(d["n_gd"]),
            alias=d.get("alias"),
            gd_u_enr=float(d.get("gd_u_enr", GD_CARRIER_ENR)),
            gd_positions=pos,
            extra={k: v for k, v in d.items() if k not in _CATALOG_KNOWN},
        )
        if tid in entries:
            raise ValueError(f"duplicate type_id in catalog: {tid}")
        entries[tid] = entry
    return Catalog(library_id=str(data.get("library_id", "")),
                   entries=entries, warnings=warnings, source=str(p))


# --------------------------------------------------------------------------- #
# 5.  round-1 enumeration
# --------------------------------------------------------------------------- #
def enumerate_gd_layouts(n_gd: int | None = None,
                         *, min_cheb: int = MIN_CHEB
                         ) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    """Every admissible Gd layout, keyed by ``n_gd``.

    A layout is a subset of :data:`GD_CANDIDATES` whose summed pin multiplicity
    equals ``n_gd`` and whose members are pairwise Chebyshev >= ``min_cheb``
    apart.  ``SURROGATE_USAGE.md`` §7.5 records the answer independently:
    **89 valid layouts** -- n12: 18, n16: 24, n20: 28, n24: 19 -- all of which
    appear in the training set.
    """
    wanted = BOUNDS.n_gd if n_gd is None else (int(n_gd),)
    out: dict[int, list[tuple[tuple[int, int], ...]]] = {n: [] for n in wanted}
    max_k = max(wanted) // 4 + 1
    for k in range(1, min(len(GD_CANDIDATES), max_k) + 1):
        for combo in combinations(GD_CANDIDATES, k):
            total = sum(pin_multiplicity(p) for p in combo)
            if total not in out:
                continue
            if any(cheb_distance(a, b) < min_cheb
                   for a, b in combinations(combo, 2)):
                continue
            out[total].append(combo)
    return {n: sorted(v) for n, v in out.items()}


def _hundredths(x: float) -> int:
    return int(round(x * 100.0))


@dataclass(frozen=True)
class EnumerationResult:
    count: int
    n_u_high: int
    n_enrichment_pairs: int
    n_gd_wt: int
    n_layout_pairs: int
    n_zoning: int
    enrichment_pairs: tuple[tuple[float, float], ...]
    layouts_by_n_gd: dict[int, int]

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "n_u_high": self.n_u_high,
            "n_enrichment_pairs": self.n_enrichment_pairs,
            "n_gd_wt": self.n_gd_wt,
            "n_layout_pairs": self.n_layout_pairs,
            "n_zoning": self.n_zoning,
            "layouts_by_n_gd": dict(self.layouts_by_n_gd),
            "enrichment_pairs": [list(p) for p in self.enrichment_pairs],
        }


def enumerate_round1(*,
                     u_lo: float = 5.00, u_hi: float = 5.50,
                     u_step: float = 0.05,
                     ratio_center: float = 0.85, ratio_tol: float = 0.03,
                     du_lo: float = 0.40, du_hi: float = 0.80,
                     e2_step: float = 0.05,
                     gd_wt: Sequence[int] = BOUNDS.gd_wt,
                     n_gd: Sequence[int] = BOUNDS.n_gd,
                     zoning: Sequence[str] = ("z1", "z2"),
                     ) -> EnumerationResult:
    """Count the round-1 candidate window, deterministically.

    The frozen rule (slice-Z pre-registration section 1.3):

        ``u_high in [5.00, 5.50]`` on a 0.05 step, ``e2`` on the 0.05 grid
        satisfying **both** ``ratio = e2/u_high in [0.82, 0.88]`` and
        ``du = u_high - e2 in [0.40, 0.80]``, times ``gd_wt in {6,8,10}``,
        times the ``(n_gd, layout)`` pairs, times ``z in {z1, z2}``;
        ``gd_u`` fixed at 4.0.

    Arithmetic is done in integer hundredths so the count does not depend on
    binary floating point.  The pre-registration explicitly forbids quoting the
    v2 figure of 3,738 here: that number assumed ``ratio = 0.85`` exactly.
    """
    u_cents = list(range(_hundredths(u_lo), _hundredths(u_hi) + 1,
                         _hundredths(u_step)))
    e2_cent_step = _hundredths(e2_step)
    r_lo, r_hi = ratio_center - ratio_tol, ratio_center + ratio_tol
    pairs: list[tuple[float, float]] = []
    for uc in u_cents:
        u = uc / 100.0
        lo = max(r_lo * u, u - du_hi)
        hi = min(r_hi * u, u - du_lo)
        first = math.ceil((lo * 100.0 - 1e-6) / e2_cent_step) * e2_cent_step
        for ec in range(int(first), _hundredths(hi) + 1, e2_cent_step):
            e2 = ec / 100.0
            # re-check on the exact hundredths so boundary points are decided
            # by the printed number, not by 1e-16 of float noise
            if not (r_lo - 1e-12 <= e2 / u <= r_hi + 1e-12):
                continue
            if not (du_lo - 1e-9 <= u - e2 <= du_hi + 1e-9):
                continue
            pairs.append((u, e2))

    layouts = enumerate_gd_layouts()
    layouts_by_n = {n: len(layouts.get(n, ())) for n in n_gd}
    n_layout_pairs = sum(layouts_by_n.values())
    count = len(pairs) * len(gd_wt) * n_layout_pairs * len(zoning)
    return EnumerationResult(
        count=count,
        n_u_high=len(u_cents),
        n_enrichment_pairs=len(pairs),
        n_gd_wt=len(gd_wt),
        n_layout_pairs=n_layout_pairs,
        n_zoning=len(zoning),
        enrichment_pairs=tuple(pairs),
        layouts_by_n_gd=layouts_by_n,
    )


# --------------------------------------------------------------------------- #
# 6.  the surrogate itself (optional dependency)
# --------------------------------------------------------------------------- #
class NotAvailable(RuntimeError):
    """The DeCART lattice surrogate is not usable from here.

    Carries the precise blocker (missing tree, missing checkpoint, missing
    torch, ...).  Callers must treat this as *advisory*: a design is not
    rejected because the surrogate could not be reached.
    """


#: Where the surrogate tree may live, in priority order.  ``None`` entries are
#: skipped.  A caller can always pass an explicit root.
DEFAULT_SURROGATE_ROOTS: tuple[str, ...] = (
    "~/lattice_surrogate/kpin_pa",              # HOST_238, USER-readable stage
    "/home/USER2/lattice_surrogate/kpin_pa",    # HOST_238, USER2-owned original
    "../6_DeCART_Surrogate",                    # workspace checkout
)


def find_surrogate_root(root: str | Path | None = None,
                        *, search: Sequence[str] = DEFAULT_SURROGATE_ROOTS,
                        base: str | Path | None = None) -> Path:
    """Locate a usable surrogate tree, or raise :class:`NotAvailable`.

    "Usable" == ``surrogate/predict.py`` and ``surrogate_runs/ens_peak`` and a
    ``dataset_*/bu_grid.npy`` are all present (``Engines.__init__``,
    ``predict.py:899-910``, needs all three).
    """
    tried: list[str] = []
    cands: list[Path] = []
    if root is not None:
        cands.append(Path(root).expanduser())
    else:
        anchor = Path(base) if base is not None else Path(__file__).resolve().parents[2]
        for s in search:
            p = Path(s).expanduser()
            cands.append(p if p.is_absolute() else (anchor / p))
    for c in cands:
        # All three checks are computed before the report, so the blocker
        # message names every missing path, not just the first one.
        missing = [rel for rel in ("surrogate/predict.py", "surrogate_runs/ens_peak")
                   if not (c / rel).exists()]
        if not list(c.glob("surrogate_runs/dataset_*/bu_grid.npy")):
            missing.append("surrogate_runs/dataset_*/bu_grid.npy")
        if not missing:
            return c.resolve()
        tried.append(f"{c} (missing: {', '.join(missing) or 'nothing'})")
    raise NotAvailable(
        "no usable DeCART lattice-surrogate tree found; tried:\n  "
        + "\n  ".join(tried or ["<no candidates>"]))


def _with_assembly_rows(row: dict) -> dict:
    """Return ``row`` guaranteed to carry ``assembly_rows``, without mutating it.

    Rows reach the bridge from a catalog or a CSV as well as from
    :func:`design_to_row`, so the key is filled here too.  A row whose
    ``pattern``/``gd_positions`` are unusable is returned untouched and left to
    :func:`validate_bounds` to report.
    """
    if row.get("assembly_rows"):
        return row
    try:
        rows = build_assembly_rows(row["pattern"],
                                   parse_gd_positions(row.get("gd_positions", "")))
    except (KeyError, ValueError):
        return row
    return dict(row, assembly_rows=rows)


class SurrogateBridge:
    """Lazy handle on the surrogate's ``Engines`` / ``predict_cases``.

    Construction never imports torch and never touches the checkpoints, so this
    class is safe to build in any process.  The first :meth:`predict` performs
    the load; failures surface as :class:`NotAvailable`.
    """

    def __init__(self, root: str | Path | None = None, *, device: str = "cpu",
                 pinmap_ens: str = "auto", pinmap_glob: str = "film_s*",
                 dataset: str | None = None) -> None:
        self._root_arg = root
        self.device = device
        self.pinmap_ens = pinmap_ens
        self.pinmap_glob = pinmap_glob
        self._dataset = dataset
        self._root: Path | None = None
        self._module: Any | None = None
        self._engines: Any | None = None
        self._error: str | None = None

    # -- introspection ------------------------------------------------------ #
    @property
    def available(self) -> bool:
        try:
            self.root
        except NotAvailable:
            return False
        return True

    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = find_surrogate_root(self._root_arg)
        return self._root

    @property
    def load_error(self) -> str | None:
        """The reason the surrogate is unusable, or ``None``."""
        try:
            self._load_module()
        except NotAvailable as exc:
            return str(exc)
        return self._error

    # -- loading ------------------------------------------------------------ #
    def _load_module(self) -> Any:
        if self._module is not None:
            return self._module
        path = self.root / "surrogate" / "predict.py"
        import_dir = str(path.parent)
        old_path = list(sys.path)
        try:
            if import_dir not in sys.path:
                sys.path.insert(0, import_dir)
            spec = importlib.util.spec_from_file_location(
                "lpopt_decart_lattice_predict", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:                       # optional dependency
            self._error = f"{type(exc).__name__}: {exc}"
            raise NotAvailable(
                f"cannot import {path}: {self._error}") from exc
        finally:
            sys.path[:] = old_path
        self._module = module
        return module

    def _dataset_dir(self) -> Path:
        if self._dataset:
            return Path(self._dataset)
        hits = sorted(self.root.glob("surrogate_runs/dataset_*/bu_grid.npy"))
        if not hits:
            raise NotAvailable(
                f"no surrogate_runs/dataset_*/bu_grid.npy under {self.root}")
        return hits[-1].parent

    def engines(self) -> Any:
        """Load (once) and return the surrogate ``Engines``."""
        if self._engines is not None:
            return self._engines
        module = self._load_module()
        from types import SimpleNamespace
        args = SimpleNamespace(
            device=self.device,
            dataset=str(self._dataset_dir()),
            runs_root=str(self.root / "surrogate_runs"),
            pinmap_glob=self.pinmap_glob,
            pinmap_ens=self.pinmap_ens,
        )
        try:
            self._engines = module.Engines(args)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            raise NotAvailable(
                f"cannot build surrogate Engines from {self.root}: "
                f"{self._error}") from exc
        return self._engines

    # -- prediction --------------------------------------------------------- #
    def predict(self, rows: Sequence[dict], *, fast: bool = False) -> dict:
        """Batched prediction over validated surrogate rows.

        ``rows`` must already carry ``pattern`` from :func:`pattern_for`.  Each
        row is bounds-checked first (:func:`validate_bounds`); a bounds
        violation is a ``ValueError``, because that is a caller bug, whereas a
        missing surrogate is :class:`NotAvailable`.
        """
        rows = [_with_assembly_rows(r) for r in rows]
        for i, r in enumerate(rows):
            errs = validate_bounds(r)
            if errs:
                raise ValueError(f"row {i}: " + "; ".join(errs))
        module = self._load_module()
        eng = self.engines()
        return module.predict_cases(eng, rows, fast=fast)

    def predict_one(self, row: dict) -> dict:
        row = _with_assembly_rows(row)
        module = self._load_module()
        eng = self.engines()
        errs = validate_bounds(row)
        if errs:
            raise ValueError("; ".join(errs))
        return module.predict_case(eng, row)


def predict(rows: Sequence[dict], *, root: str | Path | None = None,
            device: str = "cpu", fast: bool = False,
            bridge: SurrogateBridge | None = None) -> dict:
    """One-shot convenience wrapper.  Raises :class:`NotAvailable` if the
    surrogate cannot be reached -- never silently degrades to a fabricated
    prediction."""
    b = bridge if bridge is not None else SurrogateBridge(root, device=device)
    return b.predict(rows, fast=fast)


def screen_features(pred: dict, index: int | None = None) -> dict:
    """Pull the three screen-relevant scalars out of a ``predict_cases`` result.

    ``ff`` is ``peak_max`` -- the surrogate's FF (all 236 fuel pins, Gd and
    zoning pins included, assembly average = 1.0; ``SURROGATE_USAGE.md`` §7.3).
    It runs **0.0014 low** vs DeCART ``%DIST`` (``OPSCREEN.md:175-177``).
    """
    def _get(key):
        v = pred[key]
        return v if index is None else v[index]
    return {
        "ff": float(_get("peak_max")),
        "ff_bu": float(_get("peak_max_bu")),
        "k_bu0": float(_get("k_bu0")),
        "crossing_bu": float(_get("crossing_bu")),
    }


# --------------------------------------------------------------------------- #
# 7.  diversity / dedup / role-pair contrast gate            (task #6)
# --------------------------------------------------------------------------- #
def design_key(row: dict) -> tuple:
    """Exact-match dedup key: the full design tuple **including the layout**.

    The layout is part of the identity -- the same ``(gd_wt, n_gd, z)`` with a
    different Gd arrangement is a different assembly (OPSCREEN section 8: the
    frozen layout reaches FF 1.1657, the open ``1:1;4:1;6:4`` layout 1.1208).
    """
    return (
        round(float(row["u_high"]), 6),
        round(float(row.get("u_low", float(row["u_high"]) - float(row["du"]))), 6),
        str(row.get("pattern", "")),
        int(round(float(row["gd_wt"]))),
        int(round(float(row["n_gd"]))),
        format_gd_positions(parse_gd_positions(row.get("gd_positions", ""))),
    )


def dedup_exact(rows: Sequence[dict]) -> list[int]:
    """Indices of rows that survive exact design-tuple dedup, first wins."""
    seen: set[tuple] = set()
    keep: list[int] = []
    for i, r in enumerate(rows):
        k = design_key(r)
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    return keep


def z_scale(descriptors: Sequence[Sequence[float]]) -> list[float]:
    """Per-channel scale for the descriptor space: the **sample** sd (``ddof=1``)
    of the candidate set, with degenerate channels (sd == 0) given scale 1.0 so
    they contribute nothing rather than dividing by zero.  This mirrors how
    ``ood_guard`` collapses a degenerate ``[0, 0]`` envelope.

    This is a **fallback**.  The scale registered for task #6 is the
    ``ood_guard`` population envelope (:func:`ood_scale`); pass it explicitly as
    ``scale=`` to :func:`dedup_near` / :func:`greedy_maxmin` / :func:`select_pairs`
    when a library is available, otherwise the metric drifts with the candidate
    set from round to round."""
    if not descriptors:
        return []
    n = len(descriptors[0])
    out: list[float] = []
    for j in range(n):
        col = [float(d[j]) for d in descriptors]
        mu = sum(col) / len(col)
        var = sum((v - mu) ** 2 for v in col) / max(len(col) - 1, 1)
        sd = math.sqrt(var)
        out.append(sd if sd > 1e-12 else 1.0)
    return out


def ood_scale(fuel: Any, channels: Sequence[str],
              *, library_ids: Sequence[str] | None = None) -> list[float]:
    """The **registered** diversity scale (task #6): the ``ood_guard`` envelope.

    ``fuel`` is a :class:`lpopt.data.fuel_types.FuelLibrary`; ``channels`` names
    the descriptor channels in the same order as the descriptor vectors handed to
    :func:`dedup_near` / :func:`greedy_maxmin`.  Each channel's scale is the
    width ``z_max - z_min`` of the population envelope, so the metric is fixed by
    the library rather than by whichever candidates happen to be in this round.
    A degenerate (zero-width or unknown) channel gets scale 1.0, matching
    :func:`z_scale` and ``ood_guard``'s own ``[0, 0]`` collapse.

    Imported lazily: ``screen`` must stay importable with no model stack present.
    """
    from ..model.ood_guard import population_envelope_from_library
    env = population_envelope_from_library(fuel, library_ids)
    out: list[float] = []
    for name in channels:
        lo, hi = env.get(str(name), (0.0, 0.0))
        w = float(hi) - float(lo)
        out.append(w if w > 1e-12 else 1.0)
    return out


def _zdist(a: Sequence[float], b: Sequence[float], scale: Sequence[float]) -> float:
    return math.sqrt(sum(((float(x) - float(y)) / s) ** 2
                         for x, y, s in zip(a, b, scale)))


def dedup_near(descriptors: Sequence[Sequence[float]],
               *, tol: float = 0.25,
               scale: Sequence[float] | None = None,
               order: Sequence[int] | None = None) -> list[int]:
    """Fold near-duplicates in **descriptor** space (z-normalised distance < tol).

    Deterministic: candidates are visited in ``order`` (default: input order)
    and a candidate is dropped when it falls within ``tol`` of an already-kept
    one.  Distance is Euclidean on z-scaled channels; ``tol = 0.25`` is a
    quarter of a population sd, the value registered in task #6.
    """
    if not descriptors:
        return []
    sc = list(scale) if scale is not None else z_scale(descriptors)
    idx = list(order) if order is not None else list(range(len(descriptors)))
    keep: list[int] = []
    for i in idx:
        if all(_zdist(descriptors[i], descriptors[j], sc) >= tol for j in keep):
            keep.append(i)
    return keep


def greedy_maxmin(descriptors: Sequence[Sequence[float]], k: int,
                  *, scale: Sequence[float] | None = None,
                  seed_index: int | None = None) -> list[int]:
    """Deterministic greedy max-min diversity selection in descriptor space.

    Diversity is taken in the **descriptor** space, not the design space: two
    designs can differ in every axis and still be the same lattice as far as
    the operating point is concerned.  Ties are broken by the smaller index, so
    the same input always yields the same output.  The seed is the point
    closest to the centroid (or ``seed_index`` when given).
    """
    n = len(descriptors)
    if n == 0 or k <= 0:
        return []
    k = min(int(k), n)
    sc = list(scale) if scale is not None else z_scale(descriptors)
    if seed_index is None:
        dim = len(descriptors[0])
        centroid = [sum(float(d[j]) for d in descriptors) / n for j in range(dim)]
        seed_index = min(range(n),
                         key=lambda i: (_zdist(descriptors[i], centroid, sc), i))
    picked = [int(seed_index)]
    dmin = [_zdist(descriptors[i], descriptors[picked[0]], sc) for i in range(n)]
    while len(picked) < k:
        cand = max((i for i in range(n) if i not in picked),
                   key=lambda i: (dmin[i], -i))
        picked.append(cand)
        for i in range(n):
            d = _zdist(descriptors[i], descriptors[cand], sc)
            if d < dmin[i]:
                dmin[i] = d
    return picked


@dataclass(frozen=True)
class PairVerdict:
    i: int
    j: int
    contrast: float
    ok: bool
    reason: str = ""


def role_pair_gate(pairs: Sequence[tuple[int, int]],
                   contrasts: Sequence[float],
                   *, contrast_min: float | None = None) -> list[PairVerdict]:
    """Gate ``(68-slot, 53-slot)`` role pairs on role contrast.

    Candidates are selected as **pairs**, never singly: the operating point is a
    property of the pair, and the arms with contrast ~ 0 scattered over
    ``node_peak`` 1.387-1.551 / ``F_r`` 1.559-1.818 (``OPSCREEN.md:235-239``).
    The threshold defaults to
    :data:`lpopt.design.opscreen_chain.CONTRAST_MIN` (0.026).
    """
    from .opscreen_chain import CONTRAST_MIN, contrast_gate
    thr = CONTRAST_MIN if contrast_min is None else float(contrast_min)
    out: list[PairVerdict] = []
    for (i, j), c in zip(pairs, contrasts):
        ok = contrast_gate(c, thr)
        out.append(PairVerdict(
            i=int(i), j=int(j), contrast=float(c), ok=ok,
            reason="" if ok else f"contrast {float(c):+.4f} < {thr:.3f}"))
    return out


def select_pairs(descriptors: Sequence[Sequence[float]],
                 pairs: Sequence[tuple[int, int]],
                 contrasts: Sequence[float],
                 k: int,
                 *, contrast_min: float | None = None,
                 near_tol: float = 0.25) -> list[tuple[int, int]]:
    """End-to-end deterministic pair selection: contrast gate -> near-dedup on
    the pair's concatenated descriptors -> greedy max-min to ``k`` pairs."""
    verdicts = role_pair_gate(pairs, contrasts, contrast_min=contrast_min)
    alive = [v for v in verdicts if v.ok]
    if not alive:
        return []
    pair_desc = [list(descriptors[v.i]) + list(descriptors[v.j]) for v in alive]
    sc = z_scale(pair_desc)
    kept = dedup_near(pair_desc, tol=near_tol, scale=sc)
    sub = [pair_desc[i] for i in kept]
    chosen = greedy_maxmin(sub, k, scale=sc)
    return [(alive[kept[c]].i, alive[kept[c]].j) for c in sorted(chosen)]


# --------------------------------------------------------------------------- #
# 8.  CLI
# --------------------------------------------------------------------------- #
def _cmd_enumerate(args: argparse.Namespace) -> int:
    res = enumerate_round1()
    print(json.dumps(res.as_dict(), indent=2))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    b = SurrogateBridge(args.root, device=args.device)
    try:
        root = b.root
    except NotAvailable as exc:
        print(json.dumps({"available": False, "blocker": str(exc)}, indent=2))
        return 1
    err = b.load_error
    print(json.dumps({"available": err is None, "root": str(root),
                      "blocker": err}, indent=2))
    return 0 if err is None else 1


def _cmd_catalog(args: argparse.Namespace) -> int:
    cat = load_design_catalog(args.designs_json)
    print(json.dumps({
        "library_id": cat.library_id,
        "n_types": len(cat),
        "n_with_layout": len(cat.with_layout),
        "warnings": cat.warnings,
    }, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lpopt-design-screen",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enumerate", help="count the round-1 candidate window")
    e.set_defaults(func=_cmd_enumerate)

    p = sub.add_parser("probe", help="report surrogate availability / blocker")
    p.add_argument("--root", default=None)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=_cmd_probe)

    c = sub.add_parser("catalog", help="summarize designs.json")
    c.add_argument("designs_json")
    c.set_defaults(func=_cmd_catalog)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
