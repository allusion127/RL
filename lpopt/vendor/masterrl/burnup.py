"""MASTER assembly and pin burnup parsing plus PPI deck control."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

from .master import DeckFormatError, SummaryParseError


@dataclass(frozen=True, slots=True)
class AssemblyBurnupPeak:
    value: float
    x: str
    y: int

    @property
    def location(self) -> str:
        return f"{self.x}{self.y}"


@dataclass(frozen=True, slots=True)
class PinBurnupPeak:
    value: float
    assembly_x: str
    assembly_y: int
    axial_layer: int
    pin_i: int
    pin_j: int

    @property
    def assembly_location(self) -> str:
        return f"{self.assembly_x}{self.assembly_y}"

    @property
    def location(self) -> str:
        return (
            f"{self.assembly_location}/z{self.axial_layer}/"
            f"i{self.pin_i}/j{self.pin_j}"
        )


_EDIT5 = re.compile(r"SUMMARY\s+EDIT\s+5\s*:\s*ASSEMBLY", re.IGNORECASE)
_NEXT_EDIT = re.compile(r"SUMMARY\s+EDIT\s+6\s*:", re.IGNORECASE)
_STATE = re.compile(
    r"^\s*NO\.\s*=\s*(?P<no>\d+).*?EFPD\s*=\s*(?P<efpd>[+-]?[\d.]+)",
    re.IGNORECASE,
)
_GRID_HEADER = re.compile(r"^\s*Y\\X\s+(?P<x>.+)$", re.IGNORECASE)
_ROW = re.compile(r"^\s*(?P<y>\d+)\s+")


def _finite_floats(line: str, *, expected: int, context: str) -> list[float]:
    tokens = line.split()
    if len(tokens) != expected:
        raise SummaryParseError(
            f"{context} expected {expected} numeric entries, found {len(tokens)}"
        )
    try:
        values = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
    except ValueError as error:
        raise SummaryParseError(f"{context} contains a non-numeric value") from error
    if not all(math.isfinite(value) for value in values):
        raise SummaryParseError(f"{context} contains a non-finite value")
    return values


def parse_summary_max_assembly_burnup(text: str) -> AssemblyBurnupPeak | None:
    """Parse the maximum final-state assembly burnup from SUMMARY EDIT 5.

    Older synthetic unit-test summaries may omit EDIT 5; those return ``None``.
    A present but malformed EDIT 5 is rejected rather than silently ignored.
    """

    match = _EDIT5.search(text)
    if match is None:
        return None
    end_match = _NEXT_EDIT.search(text, match.end())
    section = text[match.end() : end_match.start() if end_match else len(text)]
    lines = section.splitlines()
    state_indices = [index for index, line in enumerate(lines) if _STATE.match(line)]
    if not state_indices:
        raise SummaryParseError("SUMMARY EDIT 5 contains no state blocks")
    start = state_indices[-1]
    end = len(lines)
    header_index = None
    x_labels: list[str] = []
    for index in range(start + 1, end):
        header = _GRID_HEADER.match(lines[index])
        if header:
            header_index = index
            x_labels = header.group("x").split()
            break
    if header_index is None or not x_labels:
        raise SummaryParseError("SUMMARY EDIT 5 final grid header was not found")

    peak: AssemblyBurnupPeak | None = None
    index = header_index + 1
    while index < end:
        row = _ROW.match(lines[index])
        if row is None:
            index += 1
            continue
        y = int(row.group("y"))
        batch_tokens = lines[index].split()[1:]
        if not batch_tokens or len(batch_tokens) > len(x_labels):
            raise SummaryParseError(
                f"SUMMARY EDIT 5 row {y} has {len(batch_tokens)} assemblies, "
                f"maximum {len(x_labels)}"
            )
        if index + 3 >= end:
            raise SummaryParseError(f"SUMMARY EDIT 5 row {y} is truncated")
        # Row layout: batch IDs, power, burnup, k-infinite.
        burnup = _finite_floats(
            lines[index + 2],
            expected=len(batch_tokens),
            context=f"SUMMARY EDIT 5 row {y} burnup",
        )
        for x, value in zip(x_labels[: len(batch_tokens)], burnup, strict=True):
            if peak is None or value > peak.value:
                peak = AssemblyBurnupPeak(value=value, x=x, y=y)
        index += 4
    if peak is None:
        raise SummaryParseError("SUMMARY EDIT 5 final state contains no assembly rows")
    return peak


_PPI_BLOCK = re.compile(
    r"^\s*FANAME\s+(?P<x>[A-Z]+)\s+(?P<y>\d+)\s+(?P<nzc>\d+)\s+",
    re.MULTILINE,
)
_PPI_PIN_BURNUP = re.compile(
    r"^\s*PIN\s+3-D\s+BURNUP\s+DISTRIBUTION.*$", re.IGNORECASE | re.MULTILINE
)


def parse_ppi_max_pin_burnup(
    text: str,
    assembly_x: str,
    assembly_y: int,
) -> PinBurnupPeak:
    """Return the maximum 3-D pin burnup in one PPI assembly block."""

    blocks = list(_PPI_BLOCK.finditer(text))
    target = next(
        (
            (index, match)
            for index, match in enumerate(blocks)
            if match.group("x").upper() == assembly_x.upper()
            and int(match.group("y")) == assembly_y
        ),
        None,
    )
    if target is None:
        raise SummaryParseError(
            f"PPI has no assembly block for {assembly_x.upper()}{assembly_y}"
        )
    block_index, block_match = target
    block_end = blocks[block_index + 1].start() if block_index + 1 < len(blocks) else len(text)
    block = text[block_match.start() : block_end]
    section = _PPI_PIN_BURNUP.search(block)
    if section is None:
        raise SummaryParseError(
            f"PPI assembly {assembly_x.upper()}{assembly_y} has no pin burnup section"
        )
    header = text[: block_match.start()]
    npin_matches = re.findall(r"^\s*NPIN\s*:\s*(\d+)", header, flags=re.MULTILINE)
    if not npin_matches:
        raise SummaryParseError("PPI NPIN header was not found")
    npin = int(npin_matches[-1])
    nzc = int(block_match.group("nzc"))
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
            values = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
        except ValueError:
            if numeric_lines:
                break
            continue
        if not all(math.isfinite(value) for value in values):
            raise SummaryParseError("PPI pin burnup section contains non-finite data")
        numeric_lines.append(values)
        if len(numeric_lines) == expected_lines:
            break
    if len(numeric_lines) != expected_lines:
        raise SummaryParseError(
            f"PPI pin burnup section has {len(numeric_lines)} rows, "
            f"expected {expected_lines}"
        )
    maximum = -math.inf
    peak_layer = peak_i = peak_j = 0
    for row_index, values in enumerate(numeric_lines):
        layer = row_index // npin + 1
        pin_j = row_index % npin + 1
        for column, value in enumerate(values, start=1):
            if value > maximum:
                maximum = value
                peak_layer = layer + 1  # PPI fuel layers map to MASTER layers 2..NZ-1.
                peak_i = column
                peak_j = pin_j
    return PinBurnupPeak(
        value=maximum,
        assembly_x=assembly_x.upper(),
        assembly_y=assembly_y,
        axial_layer=peak_layer,
        pin_i=peak_i,
        pin_j=peak_j,
    )


_EDT_OPT_HEADER = re.compile(r"^\s*%EDT_OPT\b", re.IGNORECASE)
_CARD_HEADER = re.compile(r"^\s*%\S+")


def enable_ppi_output(deck: str) -> str:
    """Set ``ipin=1`` on the last EDT_OPT card, preserving other flags."""

    lines = deck.splitlines(keepends=True)
    headers = [index for index, line in enumerate(lines) if _EDT_OPT_HEADER.match(line)]
    if not headers:
        raise DeckFormatError("deck has no %EDT_OPT card for PPI output")
    header = headers[-1]
    data_index = None
    for index in range(header + 1, len(lines)):
        if _CARD_HEADER.match(lines[index]):
            break
        stripped = lines[index].split("#", 1)[0].strip()
        if stripped and stripped != "/":
            data_index = index
            break
    if data_index is None:
        raise DeckFormatError("last %EDT_OPT card has no data line")
    line = lines[data_index]
    eol = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
    body = line[: -len(eol)] if eol else line
    data, marker, comment = body.partition("#")
    tokens = data.split()
    if len(tokens) < 3 or any(re.fullmatch(r"[+-]?\d+", token) is None for token in tokens[:3]):
        raise DeckFormatError("last %EDT_OPT data line is malformed")
    tokens[2] = "1"
    indent = data[: len(data) - len(data.lstrip())]
    rebuilt = indent + "       ".join(tokens)
    if marker:
        rebuilt += " " + marker + comment
    lines[data_index] = rebuilt + eol
    return "".join(lines)


__all__ = [
    "AssemblyBurnupPeak",
    "PinBurnupPeak",
    "enable_ppi_output",
    "parse_ppi_max_pin_burnup",
    "parse_summary_max_assembly_burnup",
]
