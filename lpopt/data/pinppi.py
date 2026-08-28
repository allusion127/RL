"""HZ-weighted rod-average pin burnup reduction from MAS_PPI (M2 close, 2026-08-20).

WHY.  ``lpopt/vendor/masterrl/burnup.py:parse_ppi_max_pin_burnup`` extracts only
the NODE peak -- the single highest axial-layer pin burnup value, and only
within the ONE assembly SUMMARY EDIT 5 names as the max-assembly-burnup
location.  The licensing question this module exists for is a DIFFERENT
reduction of the SAME 3-D array: the axial (HZ-weighted) mean burnup PER PIN,
maximised over pins -- and, because the rod-average peak need not fall in the
same assembly as the node peak, maximised over EVERY assembly block in the
file, not just the EDIT5 one.

FORMAT (Section 2 of MASTER4.0_UM_rev01.txt:12884-12922; verified against REAL
MAS_PPI bytes 2026-08-20, ``runs/pinbu_wave_keep_f113pin5/`` -- five box-199
chains run with ``[master] keep_success = true``, not the manual's sample
file):

    Record 1  "FANAME <X> <Y> <NZC> <HZ_1> <HZ_2> ... <HZ_NZC>"
              HZ(:) are the NZC axial FUEL-plane heights (bottom-to-top).
              Confirmed on real bytes: NZC=25 fuel planes, HZ uniform 15.24 cm
              (ZMESH's 27-entry core mesh minus the 2 non-fuel reflector rows).
    Record 9  "PIN 3-D BURNUP DISTRIBUTION (BOTTOM TO TOP)  (I - NX, J - NY)"
              NZC blocks of NPIN rows x NPIN columns, bottom-to-top layer
              order -- the exact grid ``burnup.parse_ppi_max_pin_burnup``
              already walks for its node-peak reduction.  BPIN layer index k
              (0-based, bottom-to-top) aligns 1:1 with ``HZ[k]``: both records
              are restricted to the same NZC fuel planes, so no reflector
              offset is needed for the axial weighting (unlike
              ``burnup.py``'s own +1 MASTER-layer-numbering offset, which is
              cosmetic labelling only, not a data alignment).

VALIDATION GATE (parser correctness; must pass before this module's rod-average
output is trusted).  This module's OWN raw (unweighted) 3-D max, computed by
scanning every assembly block in the file, must equal
``burnup.parse_ppi_max_pin_burnup``'s max EXACTLY on the SAME real bytes --
see ``tests/test_pinppi.py::test_gate_a_reproduces_node_peak`` (real-byte
fixture, not synthetic).  Confirmed 2026-08-20 on all 5 f113-PIN cores: exact
bit-for-bit match (84.202, 83.849, 84.172, 82.983, 84.242 GWd/tU).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterator, Sequence

from ..vendor.masterrl.burnup import _PPI_BLOCK, _PPI_PIN_BURNUP
from ..vendor.masterrl.master import SummaryParseError

__all__ = [
    "AssemblyRodAverage",
    "CoreRodAveragePeak",
    "parse_ppi_assembly_rod_average",
    "parse_ppi_core_rod_average_peak",
]


@dataclass(frozen=True, slots=True)
class AssemblyRodAverage:
    """One assembly block's raw (node) 3-D max plus its HZ-weighted rod-average
    peak -- both computed from the SAME BPIN(:,:,:) grid so the gate (a)
    cross-check and the gate (b) reduction can never read different numbers.
    """

    assembly_x: str
    assembly_y: int
    nzc: int
    npin: int
    hz: tuple[float, ...]
    raw_max: float
    raw_max_layer: int
    raw_max_i: int
    raw_max_j: int
    rod_avg_max: float
    rod_avg_max_i: int
    rod_avg_max_j: int

    @property
    def assembly_location(self) -> str:
        return f"{self.assembly_x}{self.assembly_y}"


@dataclass(frozen=True, slots=True)
class CoreRodAveragePeak:
    """Core-wide (all assembly blocks in one MAS_PPI file) peaks: the raw node
    max (gate a target) and the TRUE HZ-weighted rod-average max (what this
    module exists to compute), each with its own peak location.
    """

    raw_max: float
    raw_max_location: str
    rod_avg_max: float
    rod_avg_max_location: str
    n_assemblies: int


def _parse_hz_line(block_match: "re.Match[str]") -> list[float]:
    """Extract HZ(:) -- the NZC axial fuel-plane heights that trail NZC on the
    Record-1 FANAME data line itself: ``FANAME <X> <Y> <NZC> <HZ_1> ... <HZ_NZC>``.
    """
    text = block_match.string
    nzc = int(block_match.group("nzc"))
    line_end = text.find("\n", block_match.end())
    if line_end == -1:
        line_end = len(text)
    tokens = text[block_match.end() : line_end].split()
    if len(tokens) != nzc:
        raise SummaryParseError(
            f"PPI assembly {block_match.group('x').upper()}{block_match.group('y')} "
            f"HZ(:) line has {len(tokens)} values, expected NZC={nzc}"
        )
    try:
        values = [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
    except ValueError as exc:
        raise SummaryParseError(
            "PPI HZ(:) line contains a non-numeric value"
        ) from exc
    if not all(math.isfinite(v) and v > 0.0 for v in values):
        raise SummaryParseError(
            "PPI HZ(:) line contains a non-finite or non-positive value"
        )
    return values


def _pin_burnup_grid(
    text: str, block: str, block_match: "re.Match[str]"
) -> tuple[int, list[list[float]]]:
    """Return ``(npin, numeric_lines)`` for one assembly block's BPIN(:,:,:)
    grid.  Mirrors ``burnup.parse_ppi_max_pin_burnup``'s own row-scan (NPIN
    from the file header before this block, rows walked until
    ``NZC*NPIN`` numeric rows of NPIN values are collected) so the two
    parsers can never disagree on which numbers are pin burnups.
    """
    npin_matches = re.findall(
        r"^\s*NPIN\s*:\s*(\d+)", text[: block_match.start()], flags=re.MULTILINE
    )
    if not npin_matches:
        raise SummaryParseError("PPI NPIN header was not found")
    npin = int(npin_matches[-1])
    nzc = int(block_match.group("nzc"))
    section = _PPI_PIN_BURNUP.search(block)
    if section is None:
        raise SummaryParseError(
            f"PPI assembly {block_match.group('x').upper()}"
            f"{block_match.group('y')} has no pin burnup section"
        )
    expected_lines = nzc * npin
    numeric_lines: list[list[float]] = []
    for raw_line in block[section.end() :].splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) != npin:
            if numeric_lines:
                break
            continue
        try:
            values = [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
        except ValueError:
            if numeric_lines:
                break
            continue
        if not all(math.isfinite(v) for v in values):
            raise SummaryParseError("PPI pin burnup section contains non-finite data")
        numeric_lines.append(values)
        if len(numeric_lines) == expected_lines:
            break
    if len(numeric_lines) != expected_lines:
        raise SummaryParseError(
            f"PPI pin burnup section has {len(numeric_lines)} rows, "
            f"expected {expected_lines}"
        )
    return npin, numeric_lines


def _reduce_grid(
    numeric_lines: Sequence[Sequence[float]], npin: int, hz: Sequence[float]
) -> tuple[float, int, int, int, float, int, int]:
    """Walk the flat ``nzc*npin`` row list once, accumulating BOTH the plain
    (unweighted) 3-D max -- the gate (a) value -- and the HZ-weighted axial
    sum per pin, whose max (after dividing by total HZ) is the rod-average
    peak.  One pass so gate (a) and gate (b) are provably the same read of
    the same bytes.
    """
    total_hz = math.fsum(hz)
    weighted_sum = [[0.0] * npin for _ in range(npin)]  # [j-1][i-1]
    raw_max = -math.inf
    raw_layer = raw_i = raw_j = 0
    for row_index, values in enumerate(numeric_lines):
        layer = row_index // npin  # 0-based fuel-plane index -> hz[layer]
        pin_j = row_index % npin + 1
        w = hz[layer]
        row_acc = weighted_sum[pin_j - 1]
        for column, value in enumerate(values, start=1):
            row_acc[column - 1] += value * w
            if value > raw_max:
                raw_max = value
                raw_layer = layer + 2  # match burnup.py's MASTER layer 2..NZ-1 labelling
                raw_i = column
                raw_j = pin_j
    rod_avg_max = -math.inf
    ra_i = ra_j = 0
    for j in range(npin):
        for i in range(npin):
            mean = weighted_sum[j][i] / total_hz
            if mean > rod_avg_max:
                rod_avg_max = mean
                ra_j = j + 1
                ra_i = i + 1
    return raw_max, raw_layer, raw_i, raw_j, rod_avg_max, ra_i, ra_j


def _iter_assembly_blocks(
    text: str,
) -> Iterator[tuple[str, "re.Match[str]"]]:
    """Yield ``(block_text, block_match)`` for every assembly block in a
    MAS_PPI file, splitting on the SAME ``_PPI_BLOCK`` matches
    ``burnup.parse_ppi_max_pin_burnup`` uses, so block boundaries can never
    disagree between the two parsers.
    """
    blocks = list(_PPI_BLOCK.finditer(text))
    if not blocks:
        raise SummaryParseError("PPI file has no FANAME assembly blocks")
    for index, match in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
        yield text[match.start() : end], match


def parse_ppi_assembly_rod_average(
    text: str, block: str, block_match: "re.Match[str]"
) -> AssemblyRodAverage:
    """Reduce one assembly's BPIN(:,:,:) grid to its raw 3-D max (gate a) and
    HZ-weighted rod-average max (gate b) in a single pass.
    """
    x = block_match.group("x").upper()
    y = int(block_match.group("y"))
    nzc = int(block_match.group("nzc"))
    hz = _parse_hz_line(block_match)
    npin, numeric_lines = _pin_burnup_grid(text, block, block_match)
    raw_max, raw_layer, raw_i, raw_j, rod_avg_max, ra_i, ra_j = _reduce_grid(
        numeric_lines, npin, hz
    )
    return AssemblyRodAverage(
        assembly_x=x,
        assembly_y=y,
        nzc=nzc,
        npin=npin,
        hz=tuple(hz),
        raw_max=raw_max,
        raw_max_layer=raw_layer,
        raw_max_i=raw_i,
        raw_max_j=raw_j,
        rod_avg_max=rod_avg_max,
        rod_avg_max_i=ra_i,
        rod_avg_max_j=ra_j,
    )


def parse_ppi_core_rod_average_peak(text: str) -> CoreRodAveragePeak:
    """Scan EVERY assembly block in one MAS_PPI file and return the core-wide
    raw node max (must equal ``burnup.parse_ppi_max_pin_burnup`` at the
    SUMMARY EDIT5 max-assembly location -- gate a) and the core-wide
    HZ-weighted rod-average max (the TRUE rod-average peak this module exists
    to compute -- gate b).  The rod-average peak is maximised over ALL
    assemblies, not just the node-peak one: the two peaks are not guaranteed
    to coincide.
    """
    best_raw: AssemblyRodAverage | None = None
    best_rod: AssemblyRodAverage | None = None
    n = 0
    for block, match in _iter_assembly_blocks(text):
        n += 1
        asm = parse_ppi_assembly_rod_average(text, block, match)
        if best_raw is None or asm.raw_max > best_raw.raw_max:
            best_raw = asm
        if best_rod is None or asm.rod_avg_max > best_rod.rod_avg_max:
            best_rod = asm
    assert best_raw is not None and best_rod is not None  # _iter_assembly_blocks guards n==0
    return CoreRodAveragePeak(
        raw_max=best_raw.raw_max,
        raw_max_location=(
            f"{best_raw.assembly_location}/z{best_raw.raw_max_layer}"
            f"/i{best_raw.raw_max_i}/j{best_raw.raw_max_j}"
        ),
        rod_avg_max=best_rod.rod_avg_max,
        rod_avg_max_location=(
            f"{best_rod.assembly_location}/i{best_rod.rod_avg_max_i}"
            f"/j{best_rod.rod_avg_max_j}"
        ),
        n_assemblies=n,
    )
