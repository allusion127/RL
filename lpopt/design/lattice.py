"""DeCART2D lattice deck generation + (parallel) runner (plan 12.1).

Template families live under two roots:

    <apr1400>/5.8_5.1/FA/IGD_{12,16,20}/{gd}_{n}_z{1,2}/dec_FA_*.inp
    <apr1400>/260624/FA/IGD_24/{gd}_{n}_z{1,2}/dec_FA_*.inp

Selecting a template by ``(gd_wt, n_gd, zoning_variant)`` fixes the Gd-pin count
and the edge-zoning arrangement (both encoded in the assembly pin map).  On that
path only three numeric MATERIAL edits differ per design: the ``UO2`` 92235 (e1),
the ``UO2_2`` 92235 (e2), and the ``UO2G`` ``6408`` Gd2O3 wt% — :func:`edit_dec_text`
leaves the pin map byte-identical, which is what the 37 shipped paramA types were
built with and must keep being built with.

An ON-DEMAND design additionally AUTHORS its own Gd pin map: :func:`author_gd_layout`
moves the ``UO2G`` cell ids inside the 1/8 octant triangle under census /
guide-tube / edge-zoning guards, :func:`author_template` drops the result into a
per-design ``dec_FA_<type_id>.inp`` that :func:`resolve_template` prefers, and
:func:`write_authored_deck` enforces the R1/R2 compliance contract on the way out.
Without it an on-demand design is confined to the frozen layouts, which top out
below the incumbent (OPSCREEN sec. 8).

The DeCART product ``<CASEID>_0101.HGC`` is renamed to ``FA_<alias>.HGC`` (the ga80
5-char COMP-name convention); the companions ``<CASEID>.out`` (MASS(g) inventory)
and ``<CASEID>.sum`` (reference k-inf curve + BOC coefficients) become
``FA_<alias>.out`` / ``FA_<alias>.sum``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .._proc import no_window_flags
from .spec import (
    DesignRegistry,
    FuelDesign,
    format_gd_positions,
    gd_multiplicity,
    parse_gd_positions,
)

DEFAULT_DECART_EXE = r"D:\DeCART_MASTER\BIN\decart2d1.1m5omp.exe"

#: The SERIAL DeCART2D build.  ``DEFAULT_DECART_EXE`` is the OpenMP build and
#: needs ``libiomp5md.dll`` beside it; neither HOST_181 nor HOST_199 ships that
#: dll, so :func:`resolve_decart_exe` falls back to this one there (slice-Z
#: prereg sec. 3.3).
DECART_SERIAL_EXE = r"D:\DeCART_MASTER\BIN\decart2d1.1m5.exe"
#: The Intel OpenMP runtime the ``*omp.exe`` build links against.
OMP_RUNTIME_DLL = "libiomp5md.dll"

#: SHA-256 pre-checks carried over from the proven 181 queue recipe
#: (``2_LP/artifacts/run_decart_eq_xesm_queue_181.ps1:5-7``).  Compared
#: case-insensitively; the queue script prints them upper-case.
DECART_SERIAL_EXE_SHA256 = (
    "5F0F10F10BD4CC6546173C266DA3FDE72BDF1A09A191C59629FF7B4B0AF006CE"
)
DECART_XS_LIB = r"D:\DeCART_MASTER\LIB\DML-E71N047G018-PV01-cr08.BIN"
DECART_XS_LIB_SHA256 = (
    "AEF86EEBFB8B6398D0A45164C70E0FB04FCB5066546A12A3BBAB9106AF64E377"
)

#: Xenon treatment every paramA / ga80 lattice was produced with (assumption A3).
#: Mixing Xe modes inside one ``MAS_XSL`` has never been validated, so an authored
#: on-demand deck must keep the base deck's ``xenon TR`` card verbatim.
XENON_MODE = "TR"

#: (library subtree, IGD groups) to search for a ``{gd}_{n}_z{z}`` template dir.
_TEMPLATE_ROOTS = (
    ("5.8_5.1/FA", (12, 16, 20)),
    ("260624/FA", (24,)),
)


class LatticeError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# template resolution
# --------------------------------------------------------------------------- #
def _dir_name(design: FuelDesign) -> str:
    return f"{int(round(design.gd_wt))}_{design.n_gd}_{design.zoning_variant}"


def template_subtree(n_gd: int) -> str:
    """The library subtree that owns ``IGD_<n_gd>`` templates.

    Covers every supported ``n_gd`` (12/16/20/24 — the surrogate's validated range,
    ``SURROGATE_USAGE.md:143``), so an authoring driver never has to hard-code the
    16/24 pair and die on ``n_gd = 20``.
    """
    for subtree, groups in _TEMPLATE_ROOTS:
        if n_gd in groups:
            return subtree
    raise LatticeError(
        f"no template subtree rule for n_gd={n_gd}; supported: "
        + ", ".join(str(g) for _s, gs in _TEMPLATE_ROOTS for g in gs)
    )


def template_dir(design: FuelDesign, root: str | Path) -> Path:
    """``<root>/<subtree>/IGD_<n>/<gd>_<n>_<z>`` — where a template for ``design`` lives."""
    return (Path(root) / template_subtree(design.n_gd)
            / f"IGD_{design.n_gd}" / _dir_name(design))


def authored_deck_name(design: FuelDesign) -> str:
    """``dec_FA_<type_id>.inp`` — the UNIQUE file name of an authored template.

    A template directory is keyed by ``(gd_wt, n_gd, zoning_variant)`` ONLY, the
    frozen tree names its deck after the reference case (``dec_FA_B03.inp``), and
    :func:`resolve_template` took ``sorted(glob(...))[0]`` — so two authored decks
    dropped into one directory shadowed each other (R23).  Keying the file by
    ``type_id`` (which :func:`resolve_template` now prefers over the alphabetic
    first) lets every design resolve to its own deck.  The spelling is the one the
    slice-Z pre-registration freezes: ``dec_FA_P5547Z1G08N20.inp``.

    Residual, guarded elsewhere: two layouts over IDENTICAL five axes still spell
    one ``type_id``.  :meth:`DesignRegistry.alias` refuses that collision, and
    :attr:`FuelDesign.type_id_tagged` is the explicit way out.
    """
    return f"dec_FA_{design.type_id}.inp"


def _pick_deck(cand: Path, design: FuelDesign) -> Path | None:
    """The deck to use inside a template directory: this design's authored deck
    if it is there, else the historical ``sorted(glob)[0]``."""
    if not cand.is_dir():
        return None
    mine = cand / authored_deck_name(design)
    if mine.is_file():
        return mine
    decks = sorted(cand.glob("dec_FA_*.inp"))
    return decks[0] if decks else None


def resolve_template(design: FuelDesign, apr1400_root: str | Path, *,
                     template_root: str | Path | None = None) -> Path:
    """Locate the ``dec_FA_*.inp`` template for a design's (gd_wt, n_gd, z).

    ``template_root``, when given, is searched FIRST (and it alone, if it has a
    match): that is the authored ``templates_lat1600/`` tree, whose per-design
    ``dec_FA_<type_id>.inp`` decks carry the open Gd layouts.  Within any candidate
    directory the design's own :func:`authored_deck_name` wins over the alphabetic
    first deck, so two layouts of one ``(gd_wt, n_gd, z)`` never shadow each other.
    """
    roots = [Path(template_root)] if template_root is not None else []
    roots.append(Path(apr1400_root))
    want = _dir_name(design)
    for root in roots:
        for subtree, groups in _TEMPLATE_ROOTS:
            if design.n_gd not in groups:
                continue
            deck = _pick_deck(root / subtree / f"IGD_{design.n_gd}" / want, design)
            if deck is not None:
                return deck
        # broad fallback: scan every IGD_* for the directory name.
        for subtree, _groups in _TEMPLATE_ROOTS:
            for igd in sorted((root / subtree).glob("IGD_*")):
                deck = _pick_deck(igd / want, design)
                if deck is not None:
                    return deck
    raise LatticeError(
        f"no dec_FA template for design {design.type_id} (dir {want!r}) under "
        + ", ".join(str(r) for r in roots)
    )


# --------------------------------------------------------------------------- #
# deck editing
# --------------------------------------------------------------------------- #
def _read_text_flex(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _caseid_of(text: str) -> str:
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0].upper() == "CASEID":
            return toks[1]
    raise LatticeError("template has no CASEID line")


def _sub_after(line: str, marker: str, value: str) -> str:
    """Replace the numeric token that follows ``marker`` on a line, keeping the
    exact surrounding whitespace."""
    pat = re.compile(rf"(\b{re.escape(marker)}\s+)([+-]?\d+(?:\.\d+)?)")
    new, n = pat.subn(rf"\g<1>{value}", line, count=1)
    if n == 0:
        raise LatticeError(f"marker {marker!r} not found on line: {line!r}")
    return new


def _fmt_enr(value: float) -> str:
    """Format an enrichment/wt%% compactly but always with a decimal point
    (Fortran list-directed reads are happiest with an explicit real)."""
    s = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s or "0.0"


def edit_dec_text(template_text: str, design: FuelDesign, new_caseid: str) -> str:
    """Return the DeCART deck text for ``design`` from a template.

    Edits: rename the CASEID token everywhere it appears; set ``UO2`` 92235=e1,
    ``UO2_2`` 92235=e2, and the ``UO2G`` ``6408``=gd_wt.  The pin map, Gd carrier
    enrichment (4.0), and all geometry stay byte-identical to the template.
    """
    old_caseid = _caseid_of(template_text)
    out_lines: list[str] = []
    in_gd = False
    for raw in template_text.splitlines():
        line = raw
        stripped = line.strip()
        if stripped.startswith("mixture UO2_2"):
            line = _sub_after(line, "92235", _fmt_enr(design.e2))
            in_gd = False
        elif stripped.startswith("mixture UO2G"):
            in_gd = True                        # 6408 sits on the next line
        elif stripped.startswith("mixture UO2"):
            line = _sub_after(line, "92235", _fmt_enr(design.e1))
            in_gd = False
        elif in_gd and "6408" in line.split():
            line = _sub_after(line, "6408", _fmt_enr(design.gd_wt))
            in_gd = False
        elif in_gd and stripped.startswith("mixture"):
            in_gd = False
        out_lines.append(line)

    text = "\n".join(out_lines)
    if template_text.endswith("\n"):
        text += "\n"
    # rename the CASEID token (whole word) everywhere: CASEID line, assembly,
    # rad_conf, and the standalone FA name line.
    text = re.sub(rf"(?<![\w]){re.escape(old_caseid)}(?![\w])", new_caseid, text)
    return text


# --------------------------------------------------------------------------- #
# Gd pin-map authoring (task #1) — promoted from realize_lat1600.author_template.
#
# The ``assembly`` card is followed by the 1/8 octant triangle: 8 rows of 1..8
# cell ids, row 0 being the assembly CENTRE.  Cell ids (verified against
# 0_APR1400/5.8_5.1/FA/IGD_20/8_20_z1/dec_FA_B03.inp:89-97): 1 = UO2 main fuel,
# 2 = UO2_2 edge zoning, 3 = UO2G (Gd), 6-9 = the four quarters of a guide /
# instrument tube.  ``edit_dec_text`` edits three numeric tokens and CANNOT move a
# Gd pin; authoring the %DIST-bearing map is what makes an on-demand layout
# reachable at all (OPSCREEN sec. 8: the frozen layouts top out at FF 1.1657,
# worse than the ga80 incumbent 1.1390).
# --------------------------------------------------------------------------- #
UO2_CELL_ID = 1
ZONE_CELL_ID = 2
GD_CELL_ID = 3
#: The four guide/instrument-tube quarter cells (cellgeo 3-6 in every reference deck).
GUIDE_TUBE_CELL_IDS = frozenset({6, 7, 8, 9})
#: Their frozen octant seats — one central tube (0,0) plus (3,3)/(4,3)/(4,4),
#: which expand to 5 tubes x 4 cells = the 20 non-fuel cells of a 16x16 CE FA
#: (``6_DeCART_Surrogate/surrogate/features.py:32``: 256 - 20 = 236 fuel pins).
GUIDE_TUBE_OCTANT = frozenset({(0, 0), (3, 3), (4, 3), (4, 4)})
OCTANT_ROWS = 8
FULL_MAP_N = 16


def parse_octant_triangle(text: str) -> tuple[list[str], list[list[int]], list[int]]:
    """Split a deck; return ``(lines, 8-row octant triangle, line indices)``.

    The triangle is the all-numeric block right after the ``assembly`` card and
    must have rows of length 1..8.  Raises :class:`LatticeError` on any other shape
    (the reference implementation ``realize_lat1600._triangle`` raised SystemExit —
    unusable from library code).
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if toks and toks[0].lower() == "assembly":
            start = i
            break
    if start is None:
        raise LatticeError("deck has no 'assembly' card")
    rows: list[list[int]] = []
    idx: list[int] = []
    for j in range(start + 1, len(lines)):
        toks = lines[j].split()
        if toks and all(t.isdigit() for t in toks):
            rows.append([int(t) for t in toks])
            idx.append(j)
            if len(rows) == OCTANT_ROWS:
                break
        elif rows:
            break
    if [len(r) for r in rows] != list(range(1, OCTANT_ROWS + 1)):
        raise LatticeError(
            f"assembly octant triangle is not rows 1..{OCTANT_ROWS}: "
            f"{[len(r) for r in rows]}"
        )
    return lines, rows, idx


def octant_census(rows: list[list[int]], want) -> set[tuple[int, int]]:
    """The octant cells whose id is ``want`` (an id or a set of ids)."""
    wanted = want if isinstance(want, (set, frozenset)) else {want}
    return {(i, j) for i, r in enumerate(rows) for j, v in enumerate(r) if v in wanted}


def octant_to_full(rows: list[list[int]], n: int = FULL_MAP_N) -> list[int]:
    """Expand the octant triangle to the FLAT row-major ``n x n`` assembly map.

    The expander the chain never had: :func:`lpopt.data.compliance.is_octant_symmetric`
    consumes a flattened full map, so R2 could not be checked on an authored deck.
    Uses the surrogate's octant -> quarter -> full truth verbatim
    (``6_DeCART_Surrogate/surrogate/features.py:80-99``): ``quarter[i][j] =
    rows[max(i,j)][min(i,j)]`` and the quarter is mirrored about both centre lines,
    i.e. octant ring ``i`` seats full rows ``n//2 + i`` and ``n//2 - 1 - i``.  Row 0
    is therefore the assembly CENTRE, not the edge.

    The result is D4-invariant by construction, so it always passes R2 — which is
    the point: it certifies that the AUTHORED map is the one being checked.
    """
    half = n // 2
    if half != OCTANT_ROWS or len(rows) != OCTANT_ROWS:
        raise LatticeError(
            f"octant_to_full needs an {OCTANT_ROWS}-row triangle for n={n}, "
            f"got {len(rows)} rows"
        )
    flat = [0] * (n * n)
    for i in range(half):
        for j in range(half):
            v = rows[max(i, j)][min(i, j)]
            for r in (half + i, half - 1 - i):
                for c in (half + j, half - 1 - j):
                    flat[r * n + c] = v
    return flat


def _assert_gd_separation(rows: list[list[int]]) -> None:
    """No two Gd pins are 8-neighbours in the EXPANDED full map (thermal shadowing).

    Adjacency has to be judged on the full 16x16 map, not on the octant: two pins
    that are far apart inside the triangle can land side by side across a mirror
    line.  :class:`LatticeError` names the offending pair.
    """
    flat = octant_to_full(rows)
    gd = [(r, c) for r in range(FULL_MAP_N) for c in range(FULL_MAP_N)
          if flat[r * FULL_MAP_N + c] == GD_CELL_ID]
    for a in range(len(gd)):
        for b in range(a + 1, len(gd)):
            (r1, c1), (r2, c2) = gd[a], gd[b]
            if max(abs(r1 - r2), abs(c1 - c2)) < 2:
                raise LatticeError(
                    f"Gd pins adjacent in the full map at {gd[a]}/{gd[b]} "
                    f"(Chebyshev separation >= 2 required)"
                )


def author_gd_layout(base_deck_text: str, gd_positions, n_gd: int) -> str:
    """Return ``base_deck_text`` with the Gd (``UO2G``) cells moved to ``gd_positions``.

    ONLY the octant-triangle Gd ids move; materials, geometry, the edge-zoning
    arrangement and every OPTION/DEPL/BRANCH block stay byte-identical
    (``edit_dec_text`` sets e1/e2/gd_wt numerically afterwards).  Hard guards, all
    :class:`LatticeError`:

    * the layout's full-map multiplicity (diagonal x4, off-diagonal x8) equals ``n_gd``;
    * the base deck's guide tubes sit exactly on :data:`GUIDE_TUBE_OCTANT`;
    * no target cell lands on a guide tube or an edge-zoning (``UO2_2``) cell;
    * after the move the zoning and guide-tube censuses are unchanged and the Gd
      census is exactly the requested layout;
    * no two Gd pins are 8-neighbours in the EXPANDED full map (Chebyshev >= 2).
    """
    target = set(parse_gd_positions(gd_positions))
    if not target:
        raise LatticeError("author_gd_layout: gd_positions is empty")
    realized = gd_multiplicity(target)
    if realized != n_gd:
        raise LatticeError(
            f"layout {format_gd_positions(target)} realizes {realized} Gd pins, "
            f"design wants n_gd={n_gd}"
        )
    lines, rows, idx = parse_octant_triangle(base_deck_text)
    gt_before = octant_census(rows, GUIDE_TUBE_CELL_IDS)
    if gt_before != set(GUIDE_TUBE_OCTANT):
        raise LatticeError(
            f"base template guide tubes at {sorted(gt_before)}, expected "
            f"{sorted(GUIDE_TUBE_OCTANT)}"
        )
    zone_before = octant_census(rows, ZONE_CELL_ID)

    new = [r[:] for r in rows]
    for (i, j) in octant_census(rows, GD_CELL_ID):
        new[i][j] = UO2_CELL_ID
    for (i, j) in sorted(target):
        if not (0 <= j <= i < OCTANT_ROWS):
            raise LatticeError(f"Gd target {i}:{j} is outside the octant triangle")
        if new[i][j] != UO2_CELL_ID:
            raise LatticeError(
                f"Gd target {i}:{j} would overwrite cell id {new[i][j]} "
                f"(guide tube or edge zoning) -- illegal layout"
            )
        new[i][j] = GD_CELL_ID

    if octant_census(new, ZONE_CELL_ID) != zone_before:
        raise LatticeError("zoning census changed -- authoring bug")
    if octant_census(new, GUIDE_TUBE_CELL_IDS) != set(GUIDE_TUBE_OCTANT):
        raise LatticeError("guide-tube census changed -- authoring bug")
    if octant_census(new, GD_CELL_ID) != target:
        raise LatticeError("Gd census != target -- authoring bug")
    _assert_gd_separation(new)

    for k, li in enumerate(idx):
        lines[li] = "  ".join(str(v) for v in new[k])
    text = "\n".join(lines)
    if base_deck_text.endswith("\n"):
        text += "\n"
    return text


# --------------------------------------------------------------------------- #
# xenon treatment (task #3)
# --------------------------------------------------------------------------- #
_XENON_RE = re.compile(r"(?im)^\s*xenon\s+(\S+)\s*$")


def xenon_mode(text: str) -> str | None:
    """The deck's ``xenon`` OPTION token (``"TR"`` in every reference deck)."""
    m = _XENON_RE.search(text)
    return m.group(1).upper() if m else None


def assert_xenon_mode(text: str, expected: str = XENON_MODE) -> str:
    """Hard-fail unless the deck carries ``xenon <expected>`` (assumption A3).

    All 37 paramA and 80 ga80 lattices were produced with ``xenon TR``; one
    ``MAS_XSL`` holding COMPs built under different Xe treatments has never been
    validated, so an authored on-demand deck must inherit the base deck's card.
    """
    got = xenon_mode(text)
    if got is None:
        raise LatticeError(
            f"deck has no 'xenon' OPTION card; on-demand lattices require "
            f"'xenon {expected}' (assumption A3)"
        )
    if got != expected.upper():
        raise LatticeError(
            f"deck declares 'xenon {got}' but the paramA library is built entirely "
            f"with 'xenon {expected}'; mixing Xe treatments in one MAS_XSL is "
            f"unvalidated (assumption A3)"
        )
    return got


# --------------------------------------------------------------------------- #
# GEOM editing (pin pitch + fuel-pin radii) — review sec. 3.3 / 4c.
#
# The MATERIAL editor above keeps ALL geometry byte-identical.  A pin-pitch /
# pin-radius optimization additionally needs to move the FIRST ``pitch`` token
# (pin pitch) and the ``cellgeo 1`` / ``cellgeo 2`` leading radii (r_pellet,
# r_clad_in, r_clad_out — the normal-fuel and IGD fuel pins).  It must NEVER touch
# the SECOND ``pitch`` token (assembly pitch 20.7772 == coredeck CoreParams.wide,
# the frozen envelope that keeps the MASTER core model valid) nor the guide-tube
# ``cellgeo 3-6``.  ``edit_dec_geom_text`` enforces exactly that with a hard guard.
# --------------------------------------------------------------------------- #
_NUM = r"[+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?"
_PITCH_T1_RE = re.compile(rf"(?im)^(\s*pitch\s+)({_NUM})(\s+)({_NUM})")
_CELLGEO_R3_RE = re.compile(
    rf"(?im)^(\s*cellgeo\s+[12]\s+)({_NUM})(\s+)({_NUM})(\s+)({_NUM})")
_CELLGEO_ANY_RE = re.compile(rf"(?im)^\s*cellgeo\s+(\d+)\b.*$")
_ASM_PITCH_TOL = 1.0e-6


def _fmt_geom(value: float) -> str:
    """Compact real with an explicit decimal point (Fortran list-directed read)."""
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s or "0.0"


def _geom_cellgeo_lines(text: str, ids: set[int]) -> list[str]:
    """The verbatim ``cellgeo`` lines whose id is in ``ids`` (frozen-check anchor)."""
    return [m.group(0) for m in _CELLGEO_ANY_RE.finditer(text)
            if int(m.group(1)) in ids]


def edit_dec_geom_text(
    template_text: str,
    *,
    pin_pitch: float | None = None,
    r_pellet: float | None = None,
    r_clad_in: float | None = None,
    r_clad_out: float | None = None,
) -> str:
    """Return deck text with the pin pitch and/or fuel-pin radii edited.

    Edits the first ``pitch`` token (pin pitch) and the three leading radii of the
    fuel pins ``cellgeo 1`` and ``cellgeo 2`` (kept identical across the two, as in
    every reference deck).  Any argument left ``None`` is not touched.  After
    editing it HARD-ASSERTS the invariants the MASTER core model depends on:

    * the second ``pitch`` token (assembly pitch) is byte-identical to the template;
    * the guide-tube ``cellgeo 3-6`` lines are byte-identical to the template.

    Raises :class:`LatticeError` if the deck has no ``pitch``/``cellgeo 1`` card or
    if an edit would violate either invariant.
    """
    orig_asm = None
    m = _PITCH_T1_RE.search(template_text)
    if m is None:
        raise LatticeError("deck has no editable 'pitch <pin> <asm>' card")
    orig_asm = float(m.group(4))
    if not _CELLGEO_R3_RE.search(template_text):
        raise LatticeError("deck has no 'cellgeo 1/2 <r1> <r2> <r3>' fuel-pin card")
    frozen_before = _geom_cellgeo_lines(template_text, {3, 4, 5, 6})

    text = template_text
    if pin_pitch is not None:
        text = _PITCH_T1_RE.sub(
            lambda mm: f"{mm.group(1)}{_fmt_geom(pin_pitch)}{mm.group(3)}{mm.group(4)}",
            text, count=1,
        )
    if r_pellet is not None or r_clad_in is not None or r_clad_out is not None:
        def _sub_radii(mm: re.Match) -> str:
            r1 = _fmt_geom(r_pellet) if r_pellet is not None else mm.group(2)
            r2 = _fmt_geom(r_clad_in) if r_clad_in is not None else mm.group(4)
            r3 = _fmt_geom(r_clad_out) if r_clad_out is not None else mm.group(6)
            return (f"{mm.group(1)}{r1}{mm.group(3)}{r2}{mm.group(5)}{r3}")
        text = _CELLGEO_R3_RE.sub(_sub_radii, text)

    # -- hard guard: the frozen envelope + guide tubes are untouched -----------
    m2 = _PITCH_T1_RE.search(text)
    if m2 is None or abs(float(m2.group(4)) - orig_asm) > _ASM_PITCH_TOL:
        raise LatticeError(
            "geometry edit changed the assembly-pitch token (frozen envelope); "
            "only the FIRST pitch token (pin pitch) may move"
        )
    frozen_after = _geom_cellgeo_lines(text, {3, 4, 5, 6})
    if frozen_after != frozen_before:
        raise LatticeError("geometry edit changed a guide-tube cellgeo 3-6 card (frozen)")
    return text


def write_dec_deck(design: FuelDesign, out_dir: str | Path, registry: DesignRegistry,
                   apr1400_root: str | Path, *,
                   template_root: str | Path | None = None) -> Path:
    """Write ``dec_FA_<alias>.inp`` for ``design`` into ``out_dir``.

    ``template_root`` (the authored ``templates_lat1600/`` tree) is searched before
    ``apr1400_root``; omitted, the frozen tree resolves exactly as before.

    When ``design.gd_positions`` names a layout the resolved deck's Gd census MUST
    match it.  ``edit_dec_text`` edits three numeric tokens and by construction
    cannot move a Gd pin, so without this guard a layout-bearing design silently
    realizes whatever layout the frozen template happens to carry and the HGC is
    then recorded in ``designs.json`` under a layout its deck does not have.  A
    layout-free design (the whole 37-type library) takes exactly the old path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alias = registry.alias(design)
    caseid = f"FA_{alias}"
    template = resolve_template(design, apr1400_root, template_root=template_root)
    raw = _read_text_flex(template)
    if design.gd_positions is not None:
        _lines, _rows, _idx = parse_octant_triangle(raw)
        got = octant_census(_rows, GD_CELL_ID)
        if got != set(design.gd_positions):
            raise LatticeError(
                f"design {design.type_id} names Gd layout "
                f"{format_gd_positions(design.gd_positions)} but the resolved "
                f"template {template} carries {format_gd_positions(got)}; author it "
                f"first (write_authored_deck / author_template) and pass "
                f"template_root=<templates_lat1600 tree>"
            )
    text = edit_dec_text(raw, design, caseid)
    deck_path = out / f"dec_FA_{alias}.inp"
    deck_path.write_text(text, encoding="utf-8")
    return deck_path


def author_template(design: FuelDesign, apr1400_root: str | Path,
                    template_root: str | Path) -> Path:
    """Author ``<template_root>/…/dec_FA_<type_id>.inp`` for ``design.gd_positions``.

    Resolves the FROZEN base deck for the design's own ``(gd_wt, n_gd, z)`` — the
    family convention ties the UO2G carrier DENSITY to ``gd_wt`` (6 -> 10.01,
    8 -> 9.95, 10 -> 9.88 g/cc) and ``edit_dec_text`` never edits a density, so a
    per-n_gd base would silently realize the lattice with the wrong carrier density
    — moves the Gd cells with :func:`author_gd_layout`, and re-asserts the xenon
    card.  Deterministic and idempotent: same inputs, byte-identical output.
    """
    if design.gd_positions is None:
        raise LatticeError(
            f"author_template: design {design.type_id} names no gd_positions "
            f"(nothing to author; use resolve_template for the frozen layout)"
        )
    base = resolve_template(design, apr1400_root)
    base_text = _read_text_flex(base)
    assert_xenon_mode(base_text)
    text = author_gd_layout(base_text, design.gd_positions, design.n_gd)
    assert_xenon_mode(text)
    out_dir = template_dir(design, template_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / authored_deck_name(design)
    dst.write_text(text, encoding="utf-8")
    return dst


def write_authored_deck(design: FuelDesign, out_dir: str | Path,
                        registry: DesignRegistry, apr1400_root: str | Path,
                        template_root: str | Path, *,
                        enforce_compliance: bool = True) -> Path:
    """Author the template, ENFORCE compliance, then write ``dec_FA_<alias>.inp``.

    The production path for an on-demand assembly, and the only production caller
    of :func:`lpopt.data.compliance.enforce_new_type` (task #16): it passes
    ``enr_main`` **and** ``enr_zone`` (never letting the 0.85 default be filled in
    silently over a surrogate screen run at a different ratio) plus the FULL 16x16
    pin map from :func:`octant_to_full`, so both R1 and R2 are actually exercised.
    """
    deck_template = author_template(design, apr1400_root, template_root)
    text = _read_text_flex(deck_template)
    if enforce_compliance:
        from .compliance import enforce_design      # local: avoids an import cycle

        _lines, rows, _idx = parse_octant_triangle(text)
        try:
            enforce_design(design, pin_map=octant_to_full(rows))
        except Exception:
            # A rejected design must not leave an authored deck behind: it would
            # sit in the template tree where resolve_template(template_root=...)
            # prefers it over the frozen deck on every later call.
            deck_template.unlink(missing_ok=True)
            raise
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alias = registry.alias(design)
    deck_path = out / f"dec_FA_{alias}.inp"
    deck_path.write_text(edit_dec_text(text, design, f"FA_{alias}"), encoding="utf-8")
    return deck_path


# --------------------------------------------------------------------------- #
# exe / XS-library preflight (task #2) — the two verified properties lifted from
# 2_LP/artifacts/run_decart_eq_xesm_queue_181.ps1 (the queue itself is NOT adopted;
# lattice.run_batch stays the runner of record).
# --------------------------------------------------------------------------- #
_NXFILE_RE = re.compile(r"(?im)^(\s*nxfile\s+)(\S.*?)(\s*)$")


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Lower-case SHA-256 of a file's bytes (streamed; the XS BIN is ~1 GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected: str, *, what: str = "file") -> str:
    """Return the file's SHA-256, raising :class:`LatticeError` on absence/mismatch."""
    p = Path(path)
    if not p.is_file():
        raise LatticeError(f"{what} not found: {p}")
    got = sha256_file(p)
    if got.lower() != str(expected).strip().lower():
        raise LatticeError(
            f"{what} SHA-256 mismatch for {p}: expected {str(expected).lower()}, "
            f"got {got}"
        )
    return got


def nxfile_of(text: str) -> str | None:
    """The deck's ``nxfile`` XS-library path token (``None`` when the card is absent)."""
    m = _NXFILE_RE.search(text)
    return m.group(2).strip() if m else None


def rewrite_nxfile(text: str, xs_path: str | Path) -> str:
    """Point the deck's ``nxfile`` card at ``xs_path`` (whitespace preserved).

    A staged deck carries the AUTHORING host's XS path; the run host's copy lives
    somewhere else.  Raises :class:`LatticeError` when the deck has no ``nxfile``
    card at all — DeCART would otherwise start and fail deep inside the run.
    """
    if _NXFILE_RE.search(text) is None:
        raise LatticeError(
            "deck has no 'nxfile <path>' XSEC card; refusing to launch DeCART "
            "against an unknown cross-section library"
        )
    return _NXFILE_RE.sub(
        lambda m: f"{m.group(1)}{xs_path}{m.group(3)}", text, count=1)


def resolve_decart_exe(exe: str | Path = DEFAULT_DECART_EXE, *,
                       serial_exe: str | Path = DECART_SERIAL_EXE) -> Path:
    """The DeCART executable actually runnable here (serial fallback).

    :data:`DEFAULT_DECART_EXE` is the OpenMP build and needs
    :data:`OMP_RUNTIME_DLL` beside it.  That dll exists only on the local
    workstation, so on a run host the omp exe starts and dies with a loader error;
    this falls back to the SERIAL build instead.  Raises :class:`LatticeError` when
    neither is present.
    """
    exe_p = Path(exe)
    if exe_p.is_file():
        needs_omp = "omp" in exe_p.stem.lower()
        if not needs_omp or (exe_p.parent / OMP_RUNTIME_DLL).is_file():
            return exe_p
    fallback = Path(serial_exe)
    if fallback.is_file():
        return fallback
    raise LatticeError(
        f"no runnable DeCART executable: {exe_p} is missing or needs "
        f"{OMP_RUNTIME_DLL} beside it, and the serial fallback {fallback} is absent"
    )


@dataclass
class DecartPreflight:
    """The verified launch context for one DeCART case."""

    exe: Path
    xs_lib: Path
    exe_sha256: str | None = None
    xs_sha256: str | None = None
    deck_text: str | None = None
    serial: bool = False

    @property
    def env(self) -> dict[str, str]:
        """Process environment forcing one thread per case (queue recipe ``:16``).

        Unconditional, as in the ported recipe: a no-op for the serial build, and
        the cap that actually matters for the omp build under
        ``run_batch(max_parallel=N)``.
        """
        return {**os.environ, "OMP_NUM_THREADS": "1"}


#: Sentinel for "use the pinned digest, but only for the pinned artefact".
_PINNED = "<pinned>"


def preflight_decart(deck_text: str, *, exe: str | Path = DEFAULT_DECART_EXE,
                     xs_lib: str | Path = DECART_XS_LIB,
                     exe_sha256: str | None = _PINNED,
                     xs_sha256: str | None = _PINNED,
                     serial_exe: str | Path = DECART_SERIAL_EXE,
                     rewrite: bool = True) -> DecartPreflight:
    """Fail fast BEFORE a 1-2 h DeCART case on a wrong binary / library / deck.

    Resolves the executable (serial fallback), checks the XS library exists,
    compares both against pinned SHA-256 digests, and rewrites the deck's
    ``nxfile`` card to the resolved library.  Every failure is a
    :class:`LatticeError` raised before anything is launched.

    ``exe_sha256`` / ``xs_sha256`` default to the PINNED digests
    (:data:`DECART_SERIAL_EXE_SHA256` / :data:`DECART_XS_LIB_SHA256`), which are
    applied only to the artefacts they actually pin — the serial exe at
    :data:`DECART_SERIAL_EXE` and the library at :data:`DECART_XS_LIB` — so the
    default path really verifies the pinned binaries while any other exe/library
    (a test double, a re-sited install) is passed through unchecked.  Pass an
    explicit digest to check such a path too, or ``None`` to opt out.
    """
    resolved = resolve_decart_exe(exe, serial_exe=serial_exe)
    xs = Path(xs_lib)
    if not xs.is_file():
        raise LatticeError(f"DeCART cross-section library not found: {xs}")
    if exe_sha256 is _PINNED and resolved != Path(DECART_SERIAL_EXE):
        exe_sha256 = None
    elif exe_sha256 is _PINNED:
        exe_sha256 = DECART_SERIAL_EXE_SHA256
    if xs_sha256 is _PINNED and xs != Path(DECART_XS_LIB):
        xs_sha256 = None
    elif xs_sha256 is _PINNED:
        xs_sha256 = DECART_XS_LIB_SHA256
    exe_digest = (verify_sha256(resolved, exe_sha256, what="DeCART exe")
                  if exe_sha256 else None)
    xs_digest = (verify_sha256(xs, xs_sha256, what="DeCART XS library")
                 if xs_sha256 else None)
    text = rewrite_nxfile(deck_text, xs) if rewrite else deck_text
    return DecartPreflight(
        exe=resolved, xs_lib=xs, exe_sha256=exe_digest, xs_sha256=xs_digest,
        deck_text=text, serial="omp" not in resolved.stem.lower(),
    )


# --------------------------------------------------------------------------- #
# DeCART runner
# --------------------------------------------------------------------------- #
@dataclass
class DecartRun:
    """A launched DeCART2D run (detached) plus its bookkeeping."""

    design: FuelDesign
    alias: str
    work_dir: Path
    caseid: str
    fa_name: str
    process: subprocess.Popen | None = None
    started: float = 0.0
    wall_s: float | None = None
    returncode: int | None = None
    hgc_path: Path | None = None
    out_path: Path | None = None
    sum_path: Path | None = None
    #: The deck DeCART was launched from (``dec_FA_<alias>.inp``), NOT the
    #: ``decart.inp`` working copy — :func:`harvest` deletes that one, so it is the
    #: only byte-auditable record of the pin map that produced the HGC (#13b).
    deck_path: Path | None = None
    #: SHA-256 of the deck bytes DeCART actually consumed (queue recipe ``:63``).
    #: Recorded at launch so a product can be tied to its exact input later.
    input_sha256: str | None = None
    error: str | None = None

    @property
    def raw_hgc(self) -> Path:
        return self.work_dir / f"{self.caseid}_0101.HGC"

    def poll(self) -> bool:
        """True once the process has exited; records wall + returncode."""
        if self.process is None:
            return True
        rc = self.process.poll()
        if rc is None:
            return False
        if self.wall_s is None:
            self.wall_s = time.monotonic() - self.started
            self.returncode = rc
        return True


def _stage_deck(deck_path: Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(deck_path, work_dir / "decart.inp")


def launch_decart(deck_path: str | Path, work_dir: str | Path, design: FuelDesign,
                  alias: str, exe: str | Path = DEFAULT_DECART_EXE,
                  env: dict[str, str] | None = None) -> DecartRun:
    """Copy the deck to ``decart.inp`` and launch DeCART2D detached.

    ``env=None`` inherits this process's environment (historical behaviour); a
    :class:`DecartPreflight` hands in ``OMP_NUM_THREADS=1`` here.
    """
    deck_path = Path(deck_path)
    work_dir = Path(work_dir)
    caseid = _caseid_of(_read_text_flex(deck_path))
    run = DecartRun(design=design, alias=alias, work_dir=work_dir,
                    caseid=caseid, fa_name=f"FA_{alias}", deck_path=deck_path)
    if run.raw_hgc.exists():
        run.raw_hgc.unlink()                    # a stale product masks a failed run
    _stage_deck(deck_path, work_dir)
    run.input_sha256 = sha256_file(work_dir / "decart.inp")
    log = open(work_dir / "decart.stdout", "wb")
    run.started = time.monotonic()
    run.process = subprocess.Popen(
        [str(exe)], cwd=str(work_dir), stdout=log, stderr=subprocess.STDOUT,
        env=env, **no_window_flags(),
    )
    return run


def harvest(run: DecartRun) -> DecartRun:
    """Rename the DeCART products to ``FA_<alias>.HGC`` / ``.out`` / ``.sum``.

    The ``.sum`` companion (task #13b) is PRESERVED, not discarded: it carries the
    reference k-inf curve and the BOC coefficients the cond_v4 harvest reads, and
    ``data/design/package/hgc/`` holds 37 ``.HGC`` + 37 ``.out`` and not one
    ``.sum`` precisely because this function used to drop it.  ``decart.inp`` (the
    working copy) is still removed; ``run.deck_path`` — the authored
    ``dec_FA_<alias>.inp`` — is never touched, and is what :func:`stage_hgc` copies.
    """
    wd = run.work_dir
    raw = run.raw_hgc
    if not raw.is_file():
        run.error = f"no product {raw.name} (rc={run.returncode})"
        return run
    hgc = wd / f"{run.fa_name}.HGC"
    if hgc.exists():
        hgc.unlink()
    raw.rename(hgc)
    run.hgc_path = hgc

    # DeCART writes <caseid>.out; fall back to decart.out / any fresh *.out.
    out_src = None
    for cand in (wd / f"{run.caseid}.out", wd / "decart.out"):
        if cand.is_file():
            out_src = cand
            break
    if out_src is None:
        outs = [p for p in wd.glob("*.out") if p.name != f"{run.fa_name}.out"]
        out_src = outs[0] if outs else None
    if out_src is not None:
        out_dst = wd / f"{run.fa_name}.out"
        if out_dst != out_src:
            if out_dst.exists():
                out_dst.unlink()
            shutil.copyfile(out_src, out_dst)
        run.out_path = out_dst

    # DeCART writes <caseid>.sum next to the .out; keep it as FA_<alias>.sum.
    sum_src = None
    for cand in (wd / f"{run.caseid}.sum", wd / "decart.sum"):
        if cand.is_file():
            sum_src = cand
            break
    if sum_src is None:
        sums = [p for p in wd.glob("*.sum") if p.name != f"{run.fa_name}.sum"]
        sum_src = sums[0] if sums else None
    if sum_src is not None:
        sum_dst = wd / f"{run.fa_name}.sum"
        if sum_dst != sum_src:
            if sum_dst.exists():
                sum_dst.unlink()
            shutil.copyfile(sum_src, sum_dst)
        run.sum_path = sum_dst
    elif (wd / f"{run.fa_name}.sum").is_file():
        run.sum_path = wd / f"{run.fa_name}.sum"

    (wd / "decart.inp").unlink(missing_ok=True)
    return run


def run_decart(deck_path: str | Path, work_dir: str | Path, design: FuelDesign,
               alias: str, *, exe: str | Path = DEFAULT_DECART_EXE,
               detached: bool = False, timeout_s: float = 5400.0) -> DecartRun:
    """Run DeCART2D on one deck.

    ``detached=True`` returns immediately with a live :class:`DecartRun` (poll
    ``run.poll()`` then :func:`harvest`).  ``detached=False`` blocks up to
    ``timeout_s``, harvests, and returns the finished run.
    """
    run = launch_decart(deck_path, work_dir, design, alias, exe=exe)
    if detached:
        return run
    try:
        run.returncode = run.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        run.process.kill()
        run.process.wait()
        run.error = f"DeCART timed out after {timeout_s:g}s"
        run.wall_s = time.monotonic() - run.started
        return run
    run.wall_s = time.monotonic() - run.started
    return harvest(run)


def _hgc_looks_valid(hgc_path: Path, n_gd: int | None = None) -> bool:
    """Structural check that a staged ``FA_<alias>.HGC`` is a COMPLETE DeCART
    product (the idempotent-skip guard for :func:`run_batch`).

    Delegates to the registered HGC gates (task #11,
    :mod:`lpopt.design.hgc_gates`) rather than re-implementing them: G-H1c
    (liveness + a reference curve whose burnup axis rises and whose k-inf does not
    rise after the burnout peak).  A truncated product from a crashed run
    therefore fails the skip guard and is re-run instead of being silently reused.

    ``n_gd`` additionally applies G-H2 (the BOC Gd-pin census equals the requested
    count).  :func:`_completed_run` supplies it ONLY for a design that names its
    own Gd layout — an authored on-demand assembly, where a census mismatch means
    the reused product belongs to a different pin map.  For a frozen-template
    design the census is a property of the template, and demanding it here would
    change which pre-existing products a resumed batch accepts.

    G-H1 (the 334-state census) and G-H1b (the exact byte size) are NOT applied
    here: they gate a FINISHED product for DELIVERY, and a skip guard that
    demanded them would re-run every legitimately partial or differently-sized
    artefact this guard has always accepted.  :func:`gate_products` applies the
    full G-H1/G-H1b/G-H1c/G-H2 set to the products of a batch instead.
    """
    try:
        if not hgc_path.is_file() or hgc_path.stat().st_size < 256:
            return False
        text = _read_text_flex(hgc_path)
    except OSError:
        return False
    from .hgc_gates import FAIL, gate_h1c_validity, gate_h2_gd_census
    if gate_h1c_validity(text, size_bytes=hgc_path.stat().st_size).status == FAIL:
        return False
    if n_gd is not None and gate_h2_gd_census(text, int(n_gd)).status == FAIL:
        return False
    return True


def _completed_run(design: FuelDesign, alias: str, wd: Path) -> DecartRun | None:
    """Reuse a previously produced run: return a finished :class:`DecartRun` when
    ``wd`` already holds a valid ``FA_<alias>.HGC`` **and** its ``FA_<alias>.out``
    companion (both needed by the library build + fuel_types ingest); else None."""
    hgc = wd / f"FA_{alias}.HGC"
    out = wd / f"FA_{alias}.out"
    # G-H2 only for an AUTHORED layout (see _hgc_looks_valid): a frozen-template
    # design keeps the historical liveness-only guard, byte for byte.
    gate_n_gd = design.n_gd if design.gd_positions is not None else None
    if not (_hgc_looks_valid(hgc, n_gd=gate_n_gd) and out.is_file()):
        return None
    run = DecartRun(design=design, alias=alias, work_dir=wd,
                    caseid=f"FA_{alias}", fa_name=f"FA_{alias}")
    run.hgc_path = hgc
    run.out_path = out
    # Companions a resumed batch must still hand to stage_hgc (#13b).
    sm = wd / f"FA_{alias}.sum"
    if sm.is_file():
        run.sum_path = sm
    deck = wd / f"dec_FA_{alias}.inp"
    if deck.is_file():
        run.deck_path = deck
    run.wall_s = 0.0
    run.returncode = 0
    return run


# --------------------------------------------------------------------------- #
# the ported 181 queue recipe (task #10) — the PROPERTIES, not the queue
# --------------------------------------------------------------------------- #
#: The image name the shared 181 host is gated on.  The queue script counted THIS
#: process, not its own workers: the cap is a property of the HOST, so it must
#: not scale with ``max_parallel``.
DECART_PROCESS_NAME = "decart2d1.1m5.exe"
#: Host-wide concurrent-DeCART ceiling (queue recipe ``:32``: "< 2").
HOST_PROCESS_LIMIT = 2
#: The line the DeCART driver prints on a clean finish (queue recipe ``:57``).
SUCCESS_MARKER = "JOB FINISHED"
#: The manifest a gated batch writes beside its work root.
BATCH_MANIFEST_NAME = "manifest.json"
#: The case list a gated batch reads (task #10 (1)).
WAVE_FILE_NAME = "design_wave.json"


def decart_process_count(name: str = DECART_PROCESS_NAME) -> int:
    """Live count of ``name`` processes on THIS host (Windows ``tasklist``).

    Raises :class:`LatticeError` when the count cannot be taken — a gate that
    silently reports 0 is worse than no gate, because it would let a batch pile
    onto a host that is already saturated.
    """
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=60, **no_window_flags())
    except (OSError, subprocess.SubprocessError) as exc:
        raise LatticeError(f"cannot count {name} processes: {exc}") from exc
    if proc.returncode != 0:
        raise LatticeError(
            f"cannot count {name} processes: tasklist rc={proc.returncode}")
    needle = '"' + name.lower() + '"'
    return sum(1 for line in proc.stdout.splitlines()
               if line.strip().lower().startswith(needle))


@dataclass
class BatchOptions:
    """The queue-recipe properties :func:`run_batch` can additionally enforce.

    Every field is OFF / permissive by default, so ``run_batch(...)`` with no
    ``options`` launches exactly what it launched before this existed.  The
    registered slice-Z recipe is :meth:`recipe_181`.
    """

    #: Resolve + hash the exe and the XS library and rewrite the deck's ``nxfile``
    #: card before anything is launched (:func:`preflight_decart`).
    preflight: bool = False
    xs_lib: str | Path = DECART_XS_LIB
    exe_sha256: str | None = _PINNED
    xs_sha256: str | None = _PINNED
    #: Refuse to harvest a case whose stdout never printed :data:`SUCCESS_MARKER`.
    require_success_marker: bool = False
    success_marker: str = SUCCESS_MARKER
    #: Host-wide process gate: hold launches while ``process_count_fn() >=
    #: host_process_limit``.  ``None`` = no gate.  The limit is deliberately NOT
    #: ``max_parallel``: raising throughput must not widen a shared-host ceiling.
    process_count_fn: Callable[[], int] | None = None
    host_process_limit: int = HOST_PROCESS_LIMIT
    #: Write ``manifest.json`` (with the per-case ``design`` block) under out_root.
    manifest: bool = False
    #: Carried into the manifest verbatim (wave id, prereg tag, predicted FF/k...).
    wave_meta: dict = field(default_factory=dict)

    @classmethod
    def recipe_181(cls, **kw) -> "BatchOptions":
        """The proven HOST_181 recipe: preflight + pinned digests + the serial
        environment + the host process gate + the completion marker + a manifest."""
        base = dict(preflight=True, require_success_marker=True, manifest=True,
                    process_count_fn=decart_process_count)
        base.update(kw)
        return cls(**base)


def designs_from_wave(path: str | Path) -> tuple[list[FuelDesign], dict]:
    """Read a ``design_wave.json`` case list (task #10 (1)).

    Wire format::

        {"wave": "slice_Z", "cases": [
            {"e1": 5.5, "e2": 4.7, "zoning_variant": "z1", "gd_wt": 8.0,
             "n_gd": 20, "gd_positions": "1:1;4:1;6:4",
             "predicted": {"ff": 1.1208, "kinf_boc": 1.2345}}]}

    Returns ``(designs, meta)``; ``meta`` carries everything that is not a
    :class:`FuelDesign` field — notably ``predicted`` per case, which the manifest
    records so a product can be held against what the screener promised.  Raises
    :class:`LatticeError` on a malformed or empty wave.
    """
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LatticeError(f"cannot read design wave {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LatticeError(f"malformed design wave {p}: {exc}") from exc
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise LatticeError(f"design wave {p} carries no 'cases' list")

    designs: list[FuelDesign] = []
    predicted: dict[str, dict] = {}
    required = ("e1", "e2", "zoning_variant", "gd_wt", "n_gd")
    for k, case in enumerate(cases):
        if not isinstance(case, dict):
            raise LatticeError(f"design wave {p} case {k} is not an object")
        missing = [f for f in required if f not in case]
        if missing:
            raise LatticeError(f"design wave {p} case {k} is missing {missing}")
        gd_pos = case.get("gd_positions")
        try:
            design = FuelDesign(
                e1=float(case["e1"]), e2=float(case["e2"]),
                zoning_variant=str(case["zoning_variant"]),
                gd_wt=float(case["gd_wt"]), n_gd=int(case["n_gd"]),
                gd_positions=(parse_gd_positions(gd_pos) if gd_pos else None),
            )
        except (TypeError, ValueError) as exc:
            raise LatticeError(f"design wave {p} case {k}: {exc}") from exc
        designs.append(design)
        pred = case.get("predicted")
        if isinstance(pred, dict):
            predicted[design.type_id_tagged] = dict(pred)

    meta = {kk: vv for kk, vv in payload.items() if kk != "cases"}
    meta["source"] = str(p)
    meta["predicted"] = predicted
    return designs, meta


def _design_block(design: FuelDesign, alias: str, meta: dict) -> dict:
    """The ``design`` record a manifest row carries (task #10 (1))."""
    predicted = (meta.get("predicted") or {}).get(design.type_id_tagged, {})
    return {
        "alias": alias,
        "type_id": design.type_id,
        "type_id_tagged": design.type_id_tagged,
        "e1": design.e1, "e2": design.e2,
        "zoning_variant": design.zoning_variant,
        "gd_wt": design.gd_wt, "n_gd": design.n_gd,
        "gd_positions": design.gd_layout,
        "predicted": dict(predicted),
    }


def write_batch_manifest(path: str | Path, runs: list[DecartRun], *,
                         options: "BatchOptions | None" = None,
                         exe: str | Path | None = None,
                         max_parallel: int | None = None,
                         timeout_s: float | None = None) -> Path:
    """One ``manifest.json`` for a batch: the recipe plus a row per case.

    Each row carries the ``design`` block (design tuple + ``gd_positions`` +
    predicted FF/k), the product paths, the wall time, and the ``input_sha256``
    receipt of the deck bytes DeCART consumed.
    """
    opts = options or BatchOptions()
    out = Path(path)
    rows = []
    for run in runs:
        rows.append({
            "design": _design_block(run.design, run.alias, opts.wave_meta),
            "work_dir": str(run.work_dir),
            "deck": str(run.deck_path) if run.deck_path else None,
            "input_sha256": run.input_sha256,
            "hgc": str(run.hgc_path) if run.hgc_path else None,
            "out": str(run.out_path) if run.out_path else None,
            "sum": str(run.sum_path) if run.sum_path else None,
            "wall_s": run.wall_s,
            "returncode": run.returncode,
            "error": run.error,
        })
    payload = {
        "schema": "lpopt.design.lattice/batch-manifest/1",
        "wave": {k: v for k, v in opts.wave_meta.items() if k != "predicted"},
        "recipe": {
            "executable": str(exe) if exe is not None else None,
            "xs_library": str(opts.xs_lib),
            "max_parallel": max_parallel,
            "timeout_s": timeout_s,
            "omp_num_threads": 1 if opts.preflight else None,
            "success_marker": (opts.success_marker
                               if opts.require_success_marker else None),
            "host_process_limit": (opts.host_process_limit
                                   if opts.process_count_fn else None),
        },
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def gate_products(runs: list[DecartRun], *, screen_kinf=None, screen_ff=None) -> dict:
    """Run the registered HGC gates (task #11) over a batch's products.

    Returns ``{alias: {"verdict": ..., "gates": [...]}}``.  This is the DELIVERY
    verdict — the full G-H1 / G-H1b / G-H1c / G-H2 set, plus G-H4 when a screen
    series is supplied — as opposed to :func:`_hgc_looks_valid`, which is only the
    cheap re-run guard.
    """
    from .hgc_gates import run_gates_for_file, verdict as fold_verdict
    report: dict = {}
    for run in runs:
        if run.hgc_path is None:
            report[run.alias] = {
                "verdict": "FAIL",
                "gates": [{"gate": "G-H1c", "status": "FAIL",
                           "detail": run.error or "no product", "metrics": {}}]}
            continue
        results = run_gates_for_file(run.hgc_path, n_gd=run.design.n_gd,
                                     screen_kinf=screen_kinf, screen_ff=screen_ff)
        report[run.alias] = {"verdict": fold_verdict(results),
                             "gates": [r.as_dict() for r in results]}
    return report


def _stdout_has_marker(work_dir: Path, marker: str) -> bool:
    """True when the case's captured stdout carries the completion marker."""
    try:
        text = (work_dir / "decart.stdout").read_text(encoding="utf-8",
                                                      errors="replace")
    except OSError:
        return False
    return marker in text


def run_batch(designs: list[FuelDesign], out_root: str | Path, registry: DesignRegistry,
              apr1400_root: str | Path, *, exe: str | Path = DEFAULT_DECART_EXE,
              max_parallel: int = 4, poll_s: float = 15.0,
              timeout_s: float = 5400.0,
              options: "BatchOptions | None" = None) -> list[DecartRun]:
    """Generate decks and run DeCART2D concurrently (bounded by ``max_parallel``).

    Idempotent: a design whose ``FA_<alias>.HGC`` (+ ``.out``) already exists under
    ``out_root/<alias>`` and parses is reused as-is (no DeCART re-launch), so a
    pre-generated band or a crash-resumed batch does not recompute finished
    lattices.  Measures wall time per run.  Returns the finished
    :class:`DecartRun` list in the same order as ``designs``.

    ``options`` (task #10) additionally enforces the properties ported from the
    proven 181 queue script: the exe / XS SHA-256 preflight with its ``nxfile``
    rewrite, ``OMP_NUM_THREADS=1``, a host-wide ``decart2d1.1m5.exe`` process gate
    that does NOT scale with ``max_parallel``, the ``JOB FINISHED`` completion
    marker, and a ``manifest.json`` carrying each case's design block.  ``None``
    (the default) reproduces the historical launches byte for byte.
    """
    opts = options or BatchOptions()
    out_root = Path(out_root)
    pending = list(designs)
    active: list[DecartRun] = []
    done: dict[str, DecartRun] = {}
    launch_exe: str | Path = exe
    launch_env: dict[str, str] | None = None

    while pending or active:
        while pending and len(active) < max_parallel:
            if (opts.process_count_fn is not None and active
                    and int(opts.process_count_fn()) >= int(opts.host_process_limit)):
                # Host-wide ceiling (queue recipe ":32"), deliberately independent
                # of max_parallel: another user's DeCART runs count too.  Only
                # consulted with a launch already in flight, so a batch can never
                # deadlock waiting on its own zero-progress state.
                break
            design = pending.pop(0)
            alias = registry.alias(design)
            wd = out_root / alias
            cached = _completed_run(design, alias, wd)
            if cached is not None:
                done[alias] = cached          # idempotent skip: reuse the product
                continue
            deck = write_dec_deck(design, wd, registry, apr1400_root)
            if opts.preflight:
                pf = preflight_decart(_read_text_flex(deck), exe=exe,
                                      xs_lib=opts.xs_lib,
                                      exe_sha256=opts.exe_sha256,
                                      xs_sha256=opts.xs_sha256)
                if pf.deck_text is not None:
                    deck.write_text(pf.deck_text, encoding="utf-8")
                launch_exe, launch_env = pf.exe, pf.env
            if launch_env is None:
                # byte-identical to the historical call (no ``env`` kwarg at all)
                active.append(launch_decart(deck, wd, design, alias, exe=launch_exe))
            else:
                active.append(launch_decart(deck, wd, design, alias, exe=launch_exe,
                                            env=launch_env))

        time.sleep(poll_s)
        still: list[DecartRun] = []
        for run in active:
            timed_out = (time.monotonic() - run.started) > timeout_s
            if run.poll():
                if (opts.require_success_marker
                        and not _stdout_has_marker(run.work_dir,
                                                   opts.success_marker)):
                    # NOT harvested: a product from a run that never announced a
                    # clean finish must not be renamed into the delivery name.
                    run.error = (f"DeCART stdout never printed "
                                 f"{opts.success_marker!r} (rc={run.returncode})")
                    done[run.alias] = run
                    continue
                harvest(run)
                done[run.alias] = run
            elif timed_out:
                run.process.kill()
                run.process.wait()
                run.wall_s = time.monotonic() - run.started
                run.error = f"DeCART timed out after {timeout_s:g}s"
                done[run.alias] = run
            else:
                still.append(run)
        active = still

    runs = [done[registry.alias(d)] for d in designs]
    if opts.manifest:
        write_batch_manifest(out_root / BATCH_MANIFEST_NAME, runs, options=opts,
                             exe=exe, max_parallel=max_parallel,
                             timeout_s=timeout_s)
    return runs


__all__ = [
    "BATCH_MANIFEST_NAME",
    "DECART_PROCESS_NAME",
    "DECART_SERIAL_EXE",
    "DECART_SERIAL_EXE_SHA256",
    "DECART_XS_LIB",
    "DECART_XS_LIB_SHA256",
    "DEFAULT_DECART_EXE",
    "FULL_MAP_N",
    "GD_CELL_ID",
    "GUIDE_TUBE_CELL_IDS",
    "GUIDE_TUBE_OCTANT",
    "HOST_PROCESS_LIMIT",
    "OCTANT_ROWS",
    "OMP_RUNTIME_DLL",
    "SUCCESS_MARKER",
    "UO2_CELL_ID",
    "WAVE_FILE_NAME",
    "XENON_MODE",
    "ZONE_CELL_ID",
    "BatchOptions",
    "DecartPreflight",
    "DecartRun",
    "LatticeError",
    "assert_xenon_mode",
    "author_gd_layout",
    "author_template",
    "authored_deck_name",
    "edit_dec_geom_text",
    "decart_process_count",
    "designs_from_wave",
    "edit_dec_text",
    "gate_products",
    "harvest",
    "launch_decart",
    "nxfile_of",
    "octant_census",
    "octant_to_full",
    "parse_octant_triangle",
    "preflight_decart",
    "resolve_decart_exe",
    "resolve_template",
    "rewrite_nxfile",
    "run_batch",
    "run_decart",
    "sha256_file",
    "template_dir",
    "template_subtree",
    "verify_sha256",
    "write_authored_deck",
    "write_batch_manifest",
    "write_dec_deck",
    "xenon_mode",
]
