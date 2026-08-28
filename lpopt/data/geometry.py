"""Canonical quarter-core geometry: MOCHA rot61 cache keys <-> mirror-69 patterns.

Two independent representations of the same APR1400 southeast quarter core are
reconciled here:

* **MOCHA rot61** — the 61 independent optimization cells (1 centre + 8 east
  arm + 52 interior) as used by ``2_LP/MOCHA`` and serialized in the Dataset A
  ``sa_2b_cache`` records.  A cell is a :class:`QuarterCell` ``(qi, qj)``,
  1-based with ``(1, 1)`` the core centre; the south arm is the rotational
  image of the east arm and therefore dependent.

* **mirror-69** — the vendored ``lpopt.vendor.masterrl.domain.Pattern``: 69
  ``%LPD_SHF`` cards over :data:`SLOTS` with orbit multiplicities 1/2/4 summing
  to 241.  ``Slot(row=qj-1, col=qi-1)``; MASTER shuffle labels satisfy
  ``qi = X_INDEX[X] - 7`` and ``qj = Y - 8`` (centre ``(J, 9)``).  This is the
  ``validate_quarter_conventions`` canonical target: fresh rotation 0,
  vertical-axis shuffle rotation 1, all other shuffles rotation 2.

The ``(y - 10)`` idiom of ``vendor/masterrl/features.py`` is a known off-by-one
bug and is never used here; the source-coordinate convention matches the
correct one in ``vendor/masterrl/ga.py`` (``_slot_source_coord``/``_coord_slot``).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence, Union

from ..vendor.masterrl.domain import (
    FuelItem,
    Pattern,
    SLOTS,
    X_INDEX,
    X_LABELS,
)

_SLOT_INDEX_BY_RC: dict[tuple[int, int], int] = {
    (slot.row, slot.col): slot.index for slot in SLOTS
}

# Full-core column offset of the quarter origin: quarter (qi, qj) sources map to
# MASTER label (X_LABELS[7 + qi], 8 + qj); centre (1, 1) -> ("J", 9).
_X_OFFSET = 7
_Y_OFFSET = 8


# --------------------------------------------------------------------------- #
# Canonical coordinate layer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class QuarterCell:
    """One MOCHA quarter cell (1-based; ``(1, 1)`` is the core centre)."""

    qi: int
    qj: int

    @property
    def kind(self) -> str:
        if self.qi == 1 and self.qj == 1:
            return "center"
        if self.qj == 1:
            return "east_arm"
        if self.qi == 1:
            return "south_arm"
        return "interior"


def slot_index_of(cell: QuarterCell) -> int:
    """Vendor :data:`SLOTS` index of a quarter cell."""
    return _SLOT_INDEX_BY_RC[(cell.qj - 1, cell.qi - 1)]


def cell_of_slot(slot: int) -> QuarterCell:
    """Quarter cell of a vendor slot index."""
    s = SLOTS[slot]
    return QuarterCell(s.col + 1, s.row + 1)


def cell_of_label(x: str, y: int) -> QuarterCell:
    """MASTER shuffle-card label ``(X, Y)`` -> quarter cell."""
    return QuarterCell(X_INDEX[x] - _X_OFFSET, int(y) - _Y_OFFSET)


def label_of_cell(cell: QuarterCell) -> tuple[str, int]:
    """Quarter cell -> MASTER shuffle-card label ``(X, Y)``."""
    return X_LABELS[cell.qi + _X_OFFSET], cell.qj + _Y_OFFSET


# --------------------------------------------------------------------------- #
# rot61 cache-key entry specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Fresh:
    batch: str
    rot: int = 0


@dataclass(frozen=True, slots=True)
class Burned:
    source: QuarterCell
    rot: int = 2


Spec = Union[Fresh, Burned]

_FRESH_RE = re.compile(r"^F:(?P<batch>[^r]+?)(?:r(?P<rot>\d+))?$")
_BURNED_RE = re.compile(r"^B:\((?P<qi>\d+),(?P<qj>\d+)\)(?:r(?P<rot>\d+))?$")


def parse_spec(spec: str) -> Spec:
    """Parse one cache-key spec; accepts stale rotation-less fresh entries."""
    fresh = _FRESH_RE.match(spec)
    if fresh:
        return Fresh(fresh.group("batch"), int(fresh.group("rot") or 0))
    burned = _BURNED_RE.match(spec)
    if burned:
        return Burned(
            QuarterCell(int(burned.group("qi")), int(burned.group("qj"))),
            int(burned.group("rot") or 2),
        )
    raise ValueError(f"unparseable cache-key spec: {spec!r}")


def format_spec(spec: Spec) -> str:
    """Serialize a spec in the current ``sa_2b_cache`` form."""
    if isinstance(spec, Fresh):
        return f"F:{spec.batch}r{spec.rot}"
    return f"B:({spec.source.qi},{spec.source.qj})r{spec.rot}"


# --------------------------------------------------------------------------- #
# rot61 <-> mirror-69
# --------------------------------------------------------------------------- #
def _fuel_item(spec: Spec, *, vertical: bool) -> FuelItem:
    if isinstance(spec, Fresh):
        return FuelItem(kind="fresh", batch=spec.batch)
    x, y = label_of_cell(spec.source)
    rotation = (spec.rot - 1) % 4 if vertical else spec.rot
    return FuelItem(kind="shuffle", restart=1, x=x, y=y, rotation=rotation)


def to_canonical_from_cache_key(key: Sequence[Sequence[Any]]) -> Pattern:
    """Expand a rot61 cache-key record into a canonical mirror-69 pattern.

    South-arm (vertical-axis) cards mirror their east-arm partner per MOCHA
    ``expand_quarter``: the shared source coordinate is kept and the burned
    rotation becomes ``(rot - 1) % 4`` (so the east-arm 2 becomes 1); fresh
    cards keep rotation 0.
    """
    rule: dict[tuple[int, int], Spec] = {
        (int(qi), int(qj)): parse_spec(spec) for qi, qj, spec in key
    }
    items: list[FuelItem] = []
    for slot in SLOTS:
        cell = cell_of_slot(slot.index)
        if slot.orbit_class == "vertical_axis":
            spec = rule[(cell.qj, cell.qi)]
            items.append(_fuel_item(spec, vertical=True))
        else:
            spec = rule[(cell.qi, cell.qj)]
            items.append(_fuel_item(spec, vertical=False))
    pattern = Pattern(tuple(items))
    pattern.validate_quarter_conventions()
    return pattern


def to_cache_key(pattern: Pattern) -> list[list[Any]]:
    """Collapse a mirror-69 pattern into its rot61 cache-key record."""
    key: list[list[Any]] = []
    for slot, item in zip(SLOTS, pattern.items, strict=True):
        if slot.orbit_class == "vertical_axis":
            continue
        cell = cell_of_slot(slot.index)
        if item.is_fresh:
            assert item.batch is not None
            spec: Spec = Fresh(item.batch, item.rotation)
        else:
            assert item.x is not None and item.y is not None
            spec = Burned(cell_of_label(item.x, item.y), item.rotation)
        key.append([cell.qi, cell.qj, format_spec(spec)])
    return key


def feed_of_key(key: Sequence[Sequence[Any]]) -> int:
    """Full-core feed of a rot61 record: centre +1, every other fresh +4."""
    total = 0
    for qi, qj, spec in key:
        if str(spec).startswith("F:"):
            total += 1 if (int(qi), int(qj)) == (1, 1) else 4
    return total


def to_canonical_from_shf(text: str) -> Pattern:
    """Parse a ``%LPD_SHF`` body into a mirror-69 pattern.

    Accepts either a bare 9-line body or a full deck, in which case the
    ``%LPD_SHF`` block is extracted up to the next section marker.
    """
    lines = text.splitlines()
    if any(line.lstrip().startswith("%LPD_SHF") for line in lines):
        body: list[str] = []
        grabbing = False
        for line in lines:
            if line.lstrip().startswith("%LPD_SHF"):
                grabbing = True
                continue
            if grabbing:
                if line.lstrip().startswith("%"):
                    break
                body.append(line)
    else:
        body = [line for line in lines if not line.lstrip().startswith("%")]
    return Pattern.parse("\n".join(body))


# --------------------------------------------------------------------------- #
# Diagonal-mirror data augmentation (plan section 4.4)
# --------------------------------------------------------------------------- #
def _transposed_index(slot_index: int) -> int:
    slot = SLOTS[slot_index]
    return _SLOT_INDEX_BY_RC[(slot.col, slot.row)]


def _source_slot(item: FuelItem) -> int:
    assert item.x is not None and item.y is not None
    return _SLOT_INDEX_BY_RC[(int(item.y) - 9, X_INDEX[item.x] - 8)]


def transpose(pattern: Pattern) -> Pattern:
    """Diagonal-mirror (qi<->qj) reflection of a canonical pattern.

    Swaps the horizontal/vertical axis-twin roles, transposes every shuffle
    source coordinate, renormalizes a source landing on the vertical axis back
    to its horizontal representative, and re-asserts the canonical conventions.
    The reflection rotation ``(-rot) % 4`` coincides with the positionally
    canonical rotation for interior/centre cards and is superseded on the axis
    twins (vertical 1 / horizontal 2), so rotations are written canonically per
    destination slot.  It is an involution and preserves ``feed``.
    """
    items: list[FuelItem | None] = [None] * len(SLOTS)
    for slot, item in zip(SLOTS, pattern.items, strict=True):
        dest = _transposed_index(slot.index)
        if item.is_fresh:
            items[dest] = FuelItem(kind="fresh", batch=item.batch)
            continue
        source = _transposed_index(_source_slot(item))
        if SLOTS[source].orbit_class == "vertical_axis":
            source = _transposed_index(source)
        x = X_LABELS[8 + SLOTS[source].col]
        y = 9 + SLOTS[source].row
        rotation = 1 if SLOTS[dest].orbit_class == "vertical_axis" else 2
        items[dest] = FuelItem(
            kind="shuffle", restart=item.restart, x=x, y=y, rotation=rotation
        )
    result = Pattern(tuple(items))  # type: ignore[arg-type]
    result.validate_quarter_conventions()
    return result
