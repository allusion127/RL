"""MASTER4 summary (``MAS_SUM``) EDIT2 / EDIT5 parser.

Ported from ``2_LP/MOCHA/master_sum.py`` (``parse_mas_sum`` / ``_parse_edit5`` /
``MasterSummary``); the parsing logic is copied verbatim and only the imports,
the text-reader (Windows/Korean cp949 fallback), the public dataclass/accessor
names (``Summary`` with ``.reactivity_rows`` / ``.edit5_maps``) and the
SE-quadrant map extraction were adapted.  Do not edit the upstream file.

Summary structure (verified against APR1400 samples):

    SUMMARY EDIT 2 : REACTIVITY    per-step NO. DAY EFPD CYC-BU TOT-BU P(%) PPM ...
    SUMMARY EDIT 3 : AO/PEAK       per-step AO FQN FRN FQP FRP ...
    SUMMARY EDIT 5 : ASSEMBLY      per-step 2-D maps batch/power/burnup/k-inf

The CBC convention used by 3_GA (``cbc_max`` = max PPM over all EDIT2 steps) is
provided by :func:`cbc_max`; the MOCHA convention (``boc_ppm`` = EDIT2 row 0) is
:func:`cbc_boc`.  Gd cores peak mid-cycle, so ``cbc_max >= cbc_boc`` always.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..vendor.masterrl.domain import ROW_LENGTHS, SLOTS, X_LABELS

# --------------------------------------------------------------------------- #
# SE-quadrant map geometry
# --------------------------------------------------------------------------- #
#: EDIT5 full-core label of quarter slot (row, col): X = J.. (col+8), Y = row+9.
#: Matches lpopt.data.geometry.label_of_cell (centre slot -> ("J", 9)).
_MAP_SIDE = len(ROW_LENGTHS)                                     # 9
_SLOT_LABELS: list[tuple[int, int, str, str]] = [
    (slot.row, slot.col, X_LABELS[slot.col + 8], str(slot.row + 9)) for slot in SLOTS
]

#: Fixed order of the four harvested maps (also the stack axis order).
MAP_KEYS: tuple[str, ...] = ("boc_power", "eoc_power", "eoc_burnup", "eoc_kinf")

#: Per-burnup-step planes kept by :func:`stack_step_maps` (``batch`` is a string
#: label, not a number, so it is not stackable).  The legacy 4-plane
#: :data:`MAP_KEYS` stack keeps BOC/EOC only; this is the FULL trajectory.
STEP_MAP_KEYS: tuple[str, ...] = ("power", "burnup", "kinf")


# --------------------------------------------------------------------------- #
# dataclasses (ported)
# --------------------------------------------------------------------------- #
@dataclass
class StepRow:
    no: int
    day: float
    efpd: float
    values: dict[str, float]


@dataclass
class AssemblyMaps:
    """One EDIT-5 snapshot: maps keyed by full-core ``(col_letter, row_number)``."""

    no: int
    day: float
    efpd: float
    batch: dict[tuple[str, str], str] = field(default_factory=dict)
    power: dict[tuple[str, str], float] = field(default_factory=dict)
    burnup: dict[tuple[str, str], float] = field(default_factory=dict)
    kinf: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass
class Summary:
    """Parsed ``MAS_SUM``: EDIT2 reactivity rows + EDIT5 per-step assembly maps."""

    reactivity_rows: list[StepRow] = field(default_factory=list)   # EDIT 2
    peaking_rows: list[StepRow] = field(default_factory=list)      # EDIT 3
    edit5_maps: list[AssemblyMaps] = field(default_factory=list)   # EDIT 5
    #: EDIT 6 axial power shape, one row per burnup step, planes BOTTOM->TOP.
    #: Free (``_split_sections`` already carried section 6); see
    #: :func:`stack_axial`.
    axial_rows: list[StepRow] = field(default_factory=list)        # EDIT 6

    # ---- EOC accessors (max-EFPD step; ties broken by NO.) ---------------- #
    @property
    def eoc_reactivity(self) -> StepRow:
        return max(self.reactivity_rows, key=lambda r: (r.efpd, r.no))

    @property
    def boc_assembly(self) -> AssemblyMaps:
        return min(self.edit5_maps, key=lambda a: (a.efpd, a.no))

    @property
    def eoc_assembly(self) -> AssemblyMaps:
        return max(self.edit5_maps, key=lambda a: (a.efpd, a.no))

    @property
    def cycle_length_efpd(self) -> float:
        return self.eoc_reactivity.efpd


# --------------------------------------------------------------------------- #
# text reader (Windows/Korean cp949 fallback)
# --------------------------------------------------------------------------- #
def _read_text_flex(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("ascii", errors="replace")


def _looks_like_existing_file(s: str) -> bool:
    """``True`` when the string plausibly names a file that exists.

    Parsers here accept either a path or the file's text, so a text argument
    also reaches this probe.  Multi-line or long strings are text by
    construction, and ``Path.is_file()`` still raises for names the OS rejects
    outright (Linux ENAMETOOLONG, errno 36, is not among the errors pathlib
    swallows), so guard the probe rather than let a stray argument crash it.
    """
    if "\n" in s or "\x00" in s or not 0 < len(s) < 500:
        return False
    try:
        return Path(s).is_file()
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# parsing (ported verbatim from master_sum.py)
# --------------------------------------------------------------------------- #
_EDIT_RE = re.compile(r"^\s*SUMMARY EDIT (\d+)\s*:")


def _split_sections(text: str) -> dict[int, list[str]]:
    sections: dict[int, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        m = _EDIT_RE.match(line)
        if m:
            current = sections.setdefault(int(m.group(1)), [])
            continue
        if current is not None:
            current.append(line)
    return sections


def _parse_step_table(lines: list[str], columns: list[str]) -> list[StepRow]:
    """Parse an EDIT 2/3 style table: 'NO. DAY EFPD <columns...>'."""
    rows: list[StepRow] = []
    for line in lines:
        toks = line.split()
        if len(toks) < 3 + len(columns) or not toks[0].isdigit():
            continue
        try:
            no = int(toks[0])
            day = float(toks[1])
            efpd = float(toks[2])
            vals = [float(t) for t in toks[3:3 + len(columns)]]
        except ValueError:
            continue
        rows.append(StepRow(no, day, efpd, dict(zip(columns, vals))))
    return rows


def _parse_edit6(lines: list[str]) -> list[StepRow]:
    """Parse EDIT 6 (axial power shape): ``NO. DAY EFPD <plane values...>``.

    The plane count is read from the data (nominally 25, columns 2..26 of the
    printed table) rather than hard-coded, so a deck with a different axial mesh
    still parses.  Planes are stored BOTTOM->TOP as ``p00..p<n-1>``.  Rows whose
    numeric tail is ragged (a truncated final line) are skipped, never guessed.
    """
    rows: list[StepRow] = []
    for line in lines:
        toks = line.split()
        if len(toks) < 4 or not toks[0].isdigit():
            continue
        try:
            no = int(toks[0])
            day = float(toks[1])
            efpd = float(toks[2])
            vals = [float(t) for t in toks[3:]]
        except ValueError:
            continue
        if not vals:
            continue
        rows.append(StepRow(no, day, efpd,
                            {f"p{i:02d}": v for i, v in enumerate(vals)}))
    return rows


_EDIT5_STEP_RE = re.compile(
    r"NO\.\s*=\s*(\d+)\s+DAY\s*=\s*([0-9.Ee+-]+)\s+EFPD\s*=\s*([0-9.Ee+-]+)"
)


def _parse_edit5(lines: list[str]) -> list[AssemblyMaps]:
    steps: list[AssemblyMaps] = []
    i = 0
    col_positions: list[tuple[str, int]] = []   # (letter, char_pos)
    while i < len(lines):
        line = lines[i]
        m = _EDIT5_STEP_RE.search(line)
        if m:
            steps.append(AssemblyMaps(int(m.group(1)), float(m.group(2)), float(m.group(3))))
            i += 1
            continue
        if "Y\\X" in line:
            col_positions = [
                (t.group(0), t.start()) for t in re.finditer(r"[A-Z]+", line)
                if t.group(0) not in ("Y", "X", "RE")
            ]
            i += 1
            continue
        toks = line.split()
        if steps and col_positions and toks and toks[0].isdigit() and i + 3 < len(lines):
            row_no = toks[0]
            block = [line] + lines[i + 1:i + 4]
            maps = steps[-1]
            for kind, raw in zip(("batch", "power", "burnup", "kinf"), block):
                matches = list(re.finditer(r"\S+", raw))
                if kind == "batch":
                    matches = matches[1:]   # drop the leading row number
                for t in matches:
                    center = (t.start() + t.end() - 1) / 2.0
                    letter = min(col_positions, key=lambda c: abs(c[1] - center))[0]
                    key = (letter, row_no)
                    if kind == "batch":
                        maps.batch[key] = t.group(0)
                    else:
                        try:
                            getattr(maps, kind)[key] = float(t.group(0))
                        except ValueError:
                            pass
            i += 4
            continue
        i += 1
    return steps


def parse_mas_sum(path_or_text: str | Path) -> Summary:
    """Parse a ``MAS_SUM`` file (or its text) into a :class:`Summary`."""
    if isinstance(path_or_text, Path) or _looks_like_existing_file(str(path_or_text)):
        text = _read_text_flex(Path(path_or_text))
    else:
        text = str(path_or_text)

    sections = _split_sections(text)
    summary = Summary()
    if 2 in sections:
        summary.reactivity_rows = _parse_step_table(
            sections[2],
            ["CYC-BU", "TOT-BU", "P(%)", "PPM", "K-EFF", "ERRFLX", "REACT."],
        )
    if 3 in sections:
        summary.peaking_rows = _parse_step_table(
            sections[3],
            ["AO", "FQN", "FRN", "FQP", "FRP", "XE", "XE-AO", "SM", "SM-AO"],
        )
    if 5 in sections:
        summary.edit5_maps = _parse_edit5(sections[5])
    if 6 in sections:
        summary.axial_rows = _parse_edit6(sections[6])
    if not summary.reactivity_rows:
        raise ValueError("MAS_SUM parse failure: EDIT 2 (REACTIVITY) not found")
    return summary


# --------------------------------------------------------------------------- #
# CBC helpers
# --------------------------------------------------------------------------- #
def cbc_boc(summary: Summary) -> float:
    """MOCHA CBC: soluble boron at BOC (EDIT2 row 0) [ppm]."""
    return float(summary.reactivity_rows[0].values["PPM"])


def cbc_max(summary: Summary) -> float:
    """3_GA CBC: maximum soluble boron over all EDIT2 steps [ppm].

    Gd-bearing cores peak mid-cycle, so this is the conservative constraint
    quantity (``cbc_max >= cbc_boc``).
    """
    return float(max(r.values["PPM"] for r in summary.reactivity_rows))


# --------------------------------------------------------------------------- #
# SE-quadrant map extraction
# --------------------------------------------------------------------------- #
def _quadrant(cell_map: dict[tuple[str, str], float]) -> np.ndarray:
    """Fold an EDIT5 label map into a 9x9 float32 SE-quadrant (NaN padding)."""
    arr = np.full((_MAP_SIDE, _MAP_SIDE), np.nan, dtype=np.float32)
    for row, col, letter, ynum in _SLOT_LABELS:
        val = cell_map.get((letter, ynum))
        if val is not None:
            arr[row, col] = val
    return arr


def extract_maps(summary: Summary) -> dict[str, np.ndarray]:
    """Extract the four multitask maps as (9, 9) float32 SE-quadrant arrays.

    Keys (:data:`MAP_KEYS` order): ``boc_power`` (BOC assembly power),
    ``eoc_power`` (EOC assembly power), ``eoc_burnup`` (EOC assembly burnup),
    ``eoc_kinf`` (EOC assembly k-infinite).  Non-fuel / off-quadrant cells are
    NaN.  Requires EDIT5 to be present (raises otherwise).
    """
    if not summary.edit5_maps:
        raise ValueError("MAS_SUM has no EDIT5 assembly maps")
    boc = summary.boc_assembly
    eoc = summary.eoc_assembly
    return {
        "boc_power": _quadrant(boc.power),
        "eoc_power": _quadrant(eoc.power),
        "eoc_burnup": _quadrant(eoc.burnup),
        "eoc_kinf": _quadrant(eoc.kinf),
    }


def stack_maps(summary: Summary) -> np.ndarray:
    """Stack :func:`extract_maps` into a single ``(4, 9, 9)`` float32 array."""
    maps = extract_maps(summary)
    return np.stack([maps[k] for k in MAP_KEYS], axis=0)


def stack_step_maps(summary: Summary) -> np.ndarray:
    """EVERY EDIT5 burnup step as ``(n_steps, 3, 9, 9)`` float32.

    ``parse_mas_sum`` already parses all ~30 steps x 69 assemblies x 4 quantities
    into memory; the legacy :func:`stack_maps` keeps only the BOC/EOC planes and
    discards the other ~28 steps.  This returns the FULL trajectory at **zero
    extra parsing cost** (planes in :data:`STEP_MAP_KEYS` order), turning the
    single-scalar cyclen/CBC supervision into a burnup-trajectory supervision.

    Steps are ordered by ``(efpd, no)`` — the same key the BOC/EOC accessors use,
    so ``result[0]`` is the BOC snapshot and ``result[-1]`` the EOC one.
    """
    if not summary.edit5_maps:
        raise ValueError("MAS_SUM has no EDIT5 assembly maps")
    steps = sorted(summary.edit5_maps, key=lambda a: (a.efpd, a.no))
    return np.stack(
        [np.stack([_quadrant(getattr(s, k)) for k in STEP_MAP_KEYS], axis=0)
         for s in steps],
        axis=0,
    )


def stack_axial(summary: Summary) -> np.ndarray:
    """EDIT 6 axial power shape as ``(n_steps, n_planes)`` float32, BOTTOM->TOP.

    Section 6 was already being split out and thrown away; this is the ~10-line
    harvest.  Gives ``ao_abs`` a 25-dimensional structural parent instead of a
    single derived scalar.  Ragged rows are NaN-padded to the widest row so the
    array is always rectangular.
    """
    if not summary.axial_rows:
        raise ValueError("MAS_SUM has no EDIT6 axial rows")
    rows = sorted(summary.axial_rows, key=lambda r: (r.efpd, r.no))
    width = max(len(r.values) for r in rows)
    arr = np.full((len(rows), width), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for j in range(width):
            v = r.values.get(f"p{j:02d}")
            if v is not None:
                arr[i, j] = v
    return arr
