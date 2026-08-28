"""Domain objects and exact ``%LPD_SHF`` parsing/formatting.

The packaged MASTER decks use a southeast quarter-core representation with 69
decision slots.  The corresponding full-core orbit multiplicities are 1 for
the centre, 2 on either symmetry axis, and 4 in the interior.  Keeping these
multiplicities explicit is essential: ``feed`` is a physical full-core count,
not the number of fresh cards in the 69-entry file.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import re
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence


ROW_LENGTHS: tuple[int, ...] = (9, 9, 9, 9, 8, 8, 7, 6, 4)
X_LABELS: tuple[str, ...] = tuple("A B C D E F G H J K L M N P R S T".split())
X_INDEX = {label: index for index, label in enumerate(X_LABELS)}


@dataclass(frozen=True, slots=True)
class Slot:
    """One independent quarter-core slot."""

    index: int
    row: int
    col: int
    multiplicity: int
    orbit_class: Literal["center", "horizontal_axis", "vertical_axis", "interior"]
    radius: float


def _build_slots() -> tuple[Slot, ...]:
    slots: list[Slot] = []
    for row, length in enumerate(ROW_LENGTHS):
        for col in range(length):
            if row == 0 and col == 0:
                multiplicity = 1
                orbit_class: Literal[
                    "center", "horizontal_axis", "vertical_axis", "interior"
                ] = "center"
            elif row == 0 or col == 0:
                multiplicity = 2
                orbit_class = "horizontal_axis" if row == 0 else "vertical_axis"
            else:
                multiplicity = 4
                orbit_class = "interior"
            slots.append(
                Slot(
                    index=len(slots),
                    row=row,
                    col=col,
                    multiplicity=multiplicity,
                    orbit_class=orbit_class,
                    radius=math.hypot(row, col),
                )
            )
    if len(slots) != 69 or sum(slot.multiplicity for slot in slots) != 241:
        raise AssertionError("APR1400 quarter-core geometry is inconsistent")
    return tuple(slots)


SLOTS: tuple[Slot, ...] = _build_slots()
MOVABLE_SLOT_INDICES: tuple[int, ...] = tuple(
    slot.index for slot in SLOTS if slot.multiplicity != 1
)


@dataclass(frozen=True, slots=True)
class FuelItem:
    """A fresh batch card or a shuffled assembly-orbit card."""

    kind: Literal["fresh", "shuffle"]
    batch: str | None = None
    restart: int | None = None
    x: str | None = None
    y: int | None = None
    rotation: int = 0

    def __post_init__(self) -> None:
        if self.kind == "fresh":
            if not self.batch or any(v is not None for v in (self.restart, self.x, self.y)):
                raise ValueError("fresh item requires only a batch name")
        elif self.kind == "shuffle":
            if self.batch is not None or self.restart is None or self.x is None or self.y is None:
                raise ValueError("shuffle item requires restart, x and y")
            if self.x not in X_INDEX:
                raise ValueError(f"unsupported MASTER X coordinate: {self.x!r}")
        else:
            raise ValueError(f"unknown fuel item kind: {self.kind!r}")
        if self.rotation not in (0, 1, 2, 3):
            raise ValueError(f"invalid rotation: {self.rotation}")

    @property
    def is_fresh(self) -> bool:
        return self.kind == "fresh"

    def to_card(self) -> str:
        if self.is_fresh:
            return f"F {self.batch:<2}  {self.rotation}"
        return f"{self.restart} {self.x} {self.y:>2} {self.rotation}"

    def canonical(self) -> str:
        if self.is_fresh:
            return f"F:{self.batch}:{self.rotation}"
        return f"S:{self.restart}:{self.x}:{self.y}:{self.rotation}"


_FRESH_RE = re.compile(r"^F\s+(?P<batch>\S+)\s+(?P<rotation>\d+)$", re.IGNORECASE)
_SHUFFLE_RE = re.compile(
    r"^(?P<restart>\d+)\s+(?P<x>[A-Z])\s+(?P<y>\d+)\s+(?P<rotation>\d+)$",
    re.IGNORECASE,
)


def parse_fuel_item(card: str) -> FuelItem:
    """Parse one semi-free-format MASTER shuffle card."""

    normalized = card.strip()
    match = _FRESH_RE.fullmatch(normalized)
    if match:
        return FuelItem(
            kind="fresh",
            batch=match.group("batch").upper(),
            rotation=int(match.group("rotation")),
        )
    match = _SHUFFLE_RE.fullmatch(normalized)
    if match:
        return FuelItem(
            kind="shuffle",
            restart=int(match.group("restart")),
            x=match.group("x").upper(),
            y=int(match.group("y")),
            rotation=int(match.group("rotation")),
        )
    raise ValueError(f"invalid LPD_SHF card: {card!r}")


@dataclass(frozen=True, slots=True)
class Pattern:
    """An immutable 69-entry quarter-core loading pattern."""

    items: tuple[FuelItem, ...]

    def __post_init__(self) -> None:
        if len(self.items) != len(SLOTS):
            raise ValueError(f"expected 69 LPD_SHF entries, received {len(self.items)}")

    @classmethod
    def parse(cls, text: str) -> "Pattern":
        cards = [part.strip() for part in text.replace("\r", "").split(",") if part.strip()]
        return cls(tuple(parse_fuel_item(card) for card in cards))

    @classmethod
    def from_file(cls, path: str | Path) -> "Pattern":
        return cls.parse(Path(path).read_text(encoding="utf-8"))

    @property
    def feed(self) -> int:
        return sum(
            slot.multiplicity
            for slot, item in zip(SLOTS, self.items, strict=True)
            if item.is_fresh
        )

    @property
    def fresh_card_count(self) -> int:
        return sum(item.is_fresh for item in self.items)

    def batch_feed(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for slot, item in zip(SLOTS, self.items, strict=True):
            if item.is_fresh:
                assert item.batch is not None
                counts[item.batch] = counts.get(item.batch, 0) + slot.multiplicity
        return counts

    def canonical(self) -> str:
        return "|".join(item.canonical() for item in self.items)

    @property
    def digest(self) -> str:
        return sha256(self.canonical().encode("ascii")).hexdigest()[:16]

    def to_shf(self, *, indent: str = "        ", final_newline: bool = False) -> str:
        lines: list[str] = []
        offset = 0
        for length in ROW_LENGTHS:
            cards = [item.to_card() for item in self.items[offset : offset + length]]
            lines.append(indent + ", ".join(cards) + ",")
            offset += length
        text = "\n".join(lines)
        return text + ("\n" if final_newline else "")

    def swap(self, first: int, second: int, *, enforce_orbit: bool = True) -> "Pattern":
        if not (0 <= first < len(self.items) and 0 <= second < len(self.items)):
            raise IndexError("slot index out of range")
        if enforce_orbit and SLOTS[first].orbit_class != SLOTS[second].orbit_class:
            raise ValueError("swap would change the physical feed/inventory symmetry orbit")
        if first == second:
            return self
        items = list(self.items)
        items[first], items[second] = items[second], items[first]
        return Pattern(tuple(items))

    def hamming(self, other: "Pattern") -> int:
        return sum(a != b for a, b in zip(self.items, other.items, strict=True))

    def validate_case(self, pair: str, feed: int) -> None:
        allowed_batches = set(pair.split("_"))
        actual_batches = {item.batch for item in self.items if item.is_fresh}
        if actual_batches - allowed_batches:
            raise ValueError(
                f"pattern uses batches {sorted(actual_batches)} outside pair {pair}"
            )
        if self.feed != feed:
            raise ValueError(f"weighted feed is {self.feed}, manifest requires {feed}")

    def validate_quarter_conventions(self) -> None:
        """Validate the symmetry/rotation conventions observed in all seeds."""

        if not self.items[0].is_fresh:
            raise ValueError("the quarter-core centre slot must be fresh")
        shuffled: list[str] = []
        for slot, item in zip(SLOTS, self.items, strict=True):
            if item.is_fresh:
                if item.rotation != 0:
                    raise ValueError(f"fresh item at slot {slot.index} must have rotation 0")
                continue
            expected_rotation = 1 if slot.orbit_class == "vertical_axis" else 2
            if item.rotation != expected_rotation:
                raise ValueError(
                    f"shuffled item at slot {slot.index} has rotation {item.rotation}; "
                    f"expected {expected_rotation} for {slot.orbit_class}"
                )
            shuffled.append(item.canonical())
        if len(shuffled) != len(set(shuffled)):
            raise ValueError("duplicate full shuffle card (restart,x,y,rotation)")


def parse_loading_shf(text: str) -> Pattern:
    return Pattern.parse(text)


def format_loading_shf(pattern: Pattern, *, final_newline: bool = False) -> str:
    return pattern.to_shf(final_newline=final_newline)


@dataclass(frozen=True, order=True, slots=True)
class CaseKey:
    pair: str
    feed: int

    @property
    def folder(self) -> str:
        return self.pair if self.feed == 121 else f"{self.pair}_f{self.feed}"

    @property
    def label(self) -> str:
        return f"{self.pair}/feed-{self.feed}"


@dataclass(frozen=True, slots=True)
class FOM:
    """Figures of merit used by the reward and constraints."""

    f_r: float
    cbc_max: float
    f_q: float
    cyclen: float
    ao_min: float | None = None
    ao_max: float | None = None
    # ``max_burnup`` is retained as the assembly-burnup field for backward
    # compatibility with existing run JSON files.
    max_burnup: float | None = None
    max_pin_burnup: float | None = None
    max_burnup_assembly: str | None = None
    max_burnup_pin: str | None = None
    converged: bool = True

    @property
    def max_assembly_burnup(self) -> float | None:
        return self.max_burnup

    @property
    def ao_abs(self) -> float | None:
        if self.ao_min is None or self.ao_max is None:
            return None
        return max(abs(self.ao_min), abs(self.ao_max))

    def as_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "F_r": self.f_r,
            "CBC_max": self.cbc_max,
            "F_q": self.f_q,
            "cyclen": self.cyclen,
            "AO_min": self.ao_min,
            "AO_max": self.ao_max,
            "max_burnup": self.max_burnup,
            "max_assembly_burnup": self.max_burnup,
            "max_pin_burnup": self.max_pin_burnup,
            "max_burnup_assembly": self.max_burnup_assembly,
            "max_burnup_pin": self.max_burnup_pin,
            "converged": self.converged,
        }


@dataclass(frozen=True, slots=True)
class PatternRecord:
    case: CaseKey
    cell: float
    seed_id: str
    pattern: Pattern
    fom: FOM
    ncyc: int
    deck_path: Path
    shf_path: Path


def iter_rows(pattern: Pattern) -> Iterator[tuple[FuelItem, ...]]:
    offset = 0
    for length in ROW_LENGTHS:
        yield pattern.items[offset : offset + length]
        offset += length
