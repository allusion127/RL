"""Canonical geometry: rot61 <-> mirror-69 expansion, round-trips, transpose.

The ground-truth oracle is the set of MOCHA-authored final-cycle ``MAS_INP``
``%LPD_SHF`` decks under ``2_LP/0_Case/runs/*/cases/*/cy*``: every one is the
expansion MOCHA itself produced from a rot61 rule, so matching them pins the
south-arm mirror, source-coordinate and rotation conventions exactly.
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

import pytest

from lpopt.data.geometry import (
    Burned,
    Fresh,
    QuarterCell,
    cell_of_label,
    cell_of_slot,
    feed_of_key,
    format_spec,
    label_of_cell,
    parse_spec,
    slot_index_of,
    to_cache_key,
    to_canonical_from_cache_key,
    to_canonical_from_shf,
    transpose,
)
from lpopt.vendor.masterrl import ga
from lpopt.vendor.masterrl.domain import SLOTS, X_INDEX, Pattern

_SLOT_BY_RC = {(slot.row, slot.col): slot for slot in SLOTS}

_CASE_ROOT = Path(__file__).resolve().parents[2] / "2_LP" / "0_Case"
_CACHE = _CASE_ROOT / "sa_2b_cache.jsonl"
_RUNS = _CASE_ROOT / "runs"


# --------------------------------------------------------------------------- #
# Data fixtures (skip cleanly when the local Dataset A tree is absent)
# --------------------------------------------------------------------------- #
def _shf_body(text: str) -> str:
    body: list[str] = []
    grab = False
    for line in text.splitlines():
        if line.strip().startswith("%LPD_SHF"):
            grab = True
            continue
        if grab:
            if line.strip().startswith("%"):
                break
            body.append(line)
    return "\n".join(body)


def _latest_run() -> Path:
    runs = sorted(p for p in _RUNS.glob("*") if (p / "cases").is_dir())
    if not runs:
        pytest.skip("no Dataset A run directories present")
    return runs[-1]


def _iter_case_decks(limit: int):
    run = _latest_run()
    out = []
    for case_dir in sorted((run / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        decks = sorted(case_dir.glob("cy*/MAS_INP"))
        if not decks:
            continue
        try:
            pattern = Pattern.parse(
                _shf_body(decks[-1].read_text(encoding="utf-8", errors="replace"))
            )
        except Exception:
            continue
        out.append((case_dir.name, pattern))
        if len(out) >= limit:
            break
    return out


def _iter_cache_records(limit: int):
    if not _CACHE.is_file():
        pytest.skip("sa_2b_cache.jsonl not present")
    out = []
    with _CACHE.open(encoding="utf-8") as handle:
        handle.readline()  # fingerprint header, not a record
        for line in handle:
            record = json.loads(line)
            if "key" not in record:
                continue
            out.append(record)
            if len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------- #
# Coordinate layer
# --------------------------------------------------------------------------- #
def test_quarter_cell_kinds() -> None:
    assert QuarterCell(1, 1).kind == "center"
    assert QuarterCell(5, 1).kind == "east_arm"
    assert QuarterCell(1, 5).kind == "south_arm"
    assert QuarterCell(3, 4).kind == "interior"


def test_center_label_convention() -> None:
    assert label_of_cell(QuarterCell(1, 1)) == ("J", 9)
    assert cell_of_label("J", 9) == QuarterCell(1, 1)


def test_label_round_trip_and_no_y_minus_10_bug() -> None:
    for slot in SLOTS:
        cell = cell_of_slot(slot.index)
        x, y = label_of_cell(cell)
        assert cell_of_label(x, y) == cell
        # correct convention: qj = Y - 8 (NOT the features.py Y - 10 idiom)
        assert cell.qj == y - 8
        assert cell.qi == X_INDEX[x] - 7


def test_slot_cell_bijection_and_orbit_correspondence() -> None:
    equivalence = {
        "center": "center",
        "east_arm": "horizontal_axis",
        "south_arm": "vertical_axis",
        "interior": "interior",
    }
    for slot in SLOTS:
        cell = cell_of_slot(slot.index)
        assert slot_index_of(cell) == slot.index
        assert equivalence[cell.kind] == slot.orbit_class


def test_source_coord_matches_vendor_ga() -> None:
    # geometry's source encoding must equal ga._slot_source_coord (the correct
    # vendor convention), never features.py's buggy (y - 10) encoding.
    for slot in SLOTS:
        cell = cell_of_slot(slot.index)
        assert label_of_cell(cell) == ga._slot_source_coord(slot.index)


# --------------------------------------------------------------------------- #
# Spec parsing (modern + stale forms)
# --------------------------------------------------------------------------- #
def test_parse_spec_modern_and_stale() -> None:
    assert parse_spec("F:B1r0") == Fresh("B1", 0)
    assert parse_spec("F:C2r0") == Fresh("C2", 0)
    assert parse_spec("F:A03") == Fresh("A03", 0)       # stale: no rotation suffix
    assert parse_spec("F:A01r0") == Fresh("A01", 0)      # 3-char legacy batch
    assert parse_spec("B:(6,5)r2") == Burned(QuarterCell(6, 5), 2)
    assert parse_spec("B:(2,2)") == Burned(QuarterCell(2, 2), 2)  # default rot 2


def test_format_spec_round_trip() -> None:
    for spec in ("F:B1r0", "B:(6,5)r2", "F:A03"):
        parsed = parse_spec(spec)
        assert parse_spec(format_spec(parsed)) == parsed


# --------------------------------------------------------------------------- #
# rot61 -> mirror-69 against real cache records
# --------------------------------------------------------------------------- #
def test_cache_key_expansion_valid_and_feed() -> None:
    records = _iter_cache_records(500)
    assert len(records) >= 20
    for record in records:
        key = record["key"]
        pattern = to_canonical_from_cache_key(key)
        pattern.validate_quarter_conventions()  # canonical
        assert feed_of_key(key) == pattern.feed


def test_cache_key_fixed_point() -> None:
    def normalize(key):
        return sorted((int(a), int(b), format_spec(parse_spec(s))) for a, b, s in key)

    for record in _iter_cache_records(500):
        key = record["key"]
        first = to_canonical_from_cache_key(key)
        round_key = to_cache_key(first)
        second = to_canonical_from_cache_key(round_key)
        assert first.items == second.items
        assert normalize(key) == normalize(round_key)


# --------------------------------------------------------------------------- #
# Ground truth: MOCHA-authored MAS_INP decks
# --------------------------------------------------------------------------- #
def test_deck_round_trip_both_directions() -> None:
    decks = _iter_case_decks(40)
    assert len(decks) >= 20
    for _name, pattern in decks:
        pattern.validate_quarter_conventions()
        key = to_cache_key(pattern)
        assert len(key) == 61
        assert to_canonical_from_cache_key(key).items == pattern.items


def test_cache_key_reproduces_real_decks_exactly() -> None:
    decks = _iter_case_decks(600)
    index = {pattern.canonical(): name for name, pattern in decks}
    if not index:
        pytest.skip("no case decks to match against")
    with _CACHE.open(encoding="utf-8") as handle:
        tail = deque(handle, maxlen=8000)
    matches = 0
    for line in tail:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = record.get("key")
        if key is None:
            continue
        canonical = to_canonical_from_cache_key(key).canonical()
        if canonical in index:
            matches += 1
    assert matches >= 20, f"only {matches} cache keys matched real decks"


def test_to_canonical_from_shf_matches_parse() -> None:
    decks = _iter_case_decks(5)
    run = _latest_run()
    for name, pattern in decks:
        raw = next((run / "cases" / name).glob("cy*/MAS_INP")).read_text(
            encoding="utf-8", errors="replace"
        )
        assert to_canonical_from_shf(raw).items == pattern.items


# --------------------------------------------------------------------------- #
# transpose (diagonal-mirror augmentation, plan section 4.4)
# --------------------------------------------------------------------------- #
def test_transpose_involution_on_cache_patterns() -> None:
    records = _iter_cache_records(500)
    assert len(records) >= 500
    for record in records:
        pattern = to_canonical_from_cache_key(record["key"])
        mirror = transpose(pattern)
        mirror.validate_quarter_conventions()
        assert transpose(mirror).items == pattern.items
        assert mirror.feed == pattern.feed


def test_transpose_involution_on_random_genomes() -> None:
    rng = random.Random(20260716)
    for _ in range(100):
        pattern = ga.random_genome(rng, "B1_C2").to_pattern()
        mirror = transpose(pattern)
        mirror.validate_quarter_conventions()
        assert transpose(mirror).items == pattern.items
        assert mirror.feed == pattern.feed


def test_transpose_is_reflection_not_rotation() -> None:
    # A shuffle source (qi, qj) must map to (qj, qi): a reflection is its own
    # inverse, whereas a 90-degree rotation idiom would break the involution.
    rng = random.Random(1)
    pattern = ga.random_genome(rng, "B1_C2").to_pattern()
    mirror = transpose(pattern)
    changed = False
    for slot, item in zip(SLOTS, pattern.items):
        if item.is_fresh:
            continue
        source = cell_of_label(item.x, item.y)
        dest = _SLOT_BY_RC[(slot.col, slot.row)]
        mirrored = mirror.items[dest.index]
        expected = QuarterCell(source.qj, source.qi)
        if expected.kind == "south_arm":  # renormalized to horizontal rep
            expected = QuarterCell(expected.qj, expected.qi)
        assert cell_of_label(mirrored.x, mirrored.y) == expected
        if source.qi != source.qj:
            changed = True
    assert changed  # transpose actually moves off-diagonal sources
