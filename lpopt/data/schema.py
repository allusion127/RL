"""Unified data schema for the lpopt store (plan section 4.2).

One :class:`CanonicalRecord` is one unique loading-pattern evaluation.  The
identity key is

    record_id = sha256(canonical_pattern_string | library_id | case_pair
                       | deck_knobs_string)                       [full 64-hex]

where ``canonical_pattern_string`` is the vendor :meth:`Pattern.canonical`
serialization (69 cell tokens joined by ``|``) and ``deck_knobs_string`` is a
stable repr of the deck knobs that influence the physics but are not captured by
the pattern (max residence, etc.).  Dataset A is produced by the MOCHA SA/GA
harness with fixed deck knobs, so its ``deck_knobs_string`` is the constant
``"mocha_default"``.

The vendor :attr:`Pattern.digest` (16-hex, pattern-only) is deliberately *not*
used for cross-library dedup: the same 69-card pattern evaluated against two
different fuel libraries is two different physical experiments, and their
``Pattern.digest`` collides while their ``record_id`` does not (see
``tests/test_extract_a.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Any

import pyarrow as pa

from ..vendor.masterrl.domain import Pattern, parse_fuel_item

#: Constant deck-knob signature for Dataset A (MOCHA-native, fixed knobs).
MOCHA_DECK_KNOBS = "mocha_default"

#: Symmetry class of every quarter-core record.
SYM_CLASS = "rot61"


# --------------------------------------------------------------------------- #
# record_id
# --------------------------------------------------------------------------- #
def compute_record_id(
    canonical_pattern: str,
    library_id: str,
    case_pair: str,
    deck_knobs: str = MOCHA_DECK_KNOBS,
) -> str:
    """Full 64-hex sha256 identity of a unique LP evaluation (plan 4.2).

    ``canonical_pattern`` is ``Pattern.canonical()``.  Including ``library_id``
    (and the pair / deck knobs) is what separates the *same* pattern evaluated
    against *different* fuel libraries — a distinction the vendor 16-hex
    ``Pattern.digest`` cannot make.
    """
    payload = f"{canonical_pattern}|{library_id}|{case_pair}|{deck_knobs}"
    return sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# pattern (de)serialization — the "69-cell packed" column
# --------------------------------------------------------------------------- #
def pack_pattern(pattern: Pattern) -> str:
    """Serialize a 69-card pattern to its canonical packed string (lossless)."""
    return pattern.canonical()


def unpack_pattern(packed: str) -> Pattern:
    """Rebuild a :class:`Pattern` from a :func:`pack_pattern` string."""
    items = []
    for token in packed.split("|"):
        parts = token.split(":")
        if parts[0] == "F":
            items.append(parse_fuel_item(f"F {parts[1]} {parts[2]}"))
        elif parts[0] == "S":
            # S:restart:x:y:rot -> "restart x y rot"
            items.append(parse_fuel_item(f"{parts[1]} {parts[2]} {parts[3]} {parts[4]}"))
        else:  # pragma: no cover - defensive
            raise ValueError(f"unparseable packed cell token: {token!r}")
    return Pattern(tuple(items))


# --------------------------------------------------------------------------- #
# record dataclass
# --------------------------------------------------------------------------- #
@dataclass
class CanonicalRecord:
    """One row of ``records.parquet`` (plan 4.2 column list)."""

    record_id: str
    dataset: str                      # "A" | "B" | "P"
    campaign: str | None              # source cache/run tag (provenance)
    stratum: str | None               # produce stratum (None for A/B)
    generator: str | None             # generator id (None for A)
    parent_record_id: str | None      # lineage (None for A)
    case_pair: str                    # e.g. "B1_C2" (sorted, fresh types incl. centre)
    feed: int
    n_batches: int
    depth2_edges: int
    e_core: float | None              # mass-weighted core-average enrichment [w/o]
    e_split: float | None             # |e_a - e_b| enrichment spread of the pair [w/o]
    library_id: str
    sym_class: str
    pattern: str                      # 69-cell packed (Pattern.canonical())
    # ---- targets ----
    f_r: float | None                 # max_frp (pin radial peaking)
    f_q: float | None                 # max_fqp (pin total peaking)
    cbc_max: float | None             # max EDIT2 PPM (harvested; NaN until then)
    cbc_boc: float | None             # boc_ppm (EDIT2 row 0)
    cbc_kind: str                     # "boc_only" | "max"
    cyclen: float | None              # cycle_length_efpd
    ao_abs: float | None              # max_abs_ao
    # ---- auxiliary targets ----
    cycle_burnup: float | None
    discharge_burnup: float | None
    max_assembly_burnup: float | None
    max_pin_burnup: float | None
    eoc_ppm: float | None
    delta_efpd: float | None
    n_cycles: float | None
    # ---- status / provenance ----
    converged: bool
    converged_at_cap: bool            # "unknown" for A (kept False)
    tolerance_margin: float | None    # NaN for A
    restart_provenance: str           # "mocha_native"
    valid: bool
    failure: str
    maps_key: str | None              # record_id when EDIT5 maps harvested, else None
    # ---- map-derived flatness scalars (appended 2026-07-26, see LATE_COLUMNS) ----
    #: ``max_i p_i`` over the 69 quarter slots of the BOC assembly-power map
    #: (== F_xy).  ``None`` when the record carries no map.
    node_peak: float | None = None
    #: Multiplicity-WEIGHTED CoV of the same map (:mod:`.flatness` §1.1).
    map_cov: float | None = None
    #: TRUE rod-average pin burnup peak: HZ-weighted axial mean per pin,
    #: maximised over pins and over EVERY assembly in the MAS_PPI file
    #: (``lpopt.data.pinppi.parse_ppi_core_rod_average_peak``). Distinct from
    #: ``max_pin_burnup`` (a single-node peak) -- appended 2026-08-20 for the
    #: M2 rod-average measurement; ``None`` until a keep_success=true replay
    #: measures it (see data/reports/pinbu_rodavg_true_20260820.md).
    max_rod_avg_burnup: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


#: Ordered column list of ``records.parquet``.
SCHEMA_COLUMNS: list[str] = [f.name for f in fields(CanonicalRecord)]

#: Columns APPENDED after the original 36-column freeze.
#:
#: The record schema is otherwise frozen: several readers depend on the exact
#: 36-column prefix and its order, so growth is **append-only at the tail** and
#: every new column is nullable with a ``None`` default.  A parquet written
#: before a column landed simply does not have it, which is a legitimate store
#: state — :func:`.store.ensure_schema_columns` back-fills the missing
#: names with nulls at every read boundary, so an old ``records.parquet`` (or an
#: old multi-PC kit) still merges and still loads.  Never insert in the middle,
#: never reorder, never make one non-nullable.
LATE_COLUMNS: tuple[str, ...] = ("node_peak", "map_cov", "max_rod_avg_burnup")

#: The 36-column prefix that predates :data:`LATE_COLUMNS` (position-dependent
#: readers key on this).
FROZEN_COLUMNS: tuple[str, ...] = tuple(
    c for c in SCHEMA_COLUMNS if c not in LATE_COLUMNS
)

#: pyarrow schema (nullable everywhere; explicit types for a stable parquet).
PARQUET_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("dataset", pa.string()),
        ("campaign", pa.string()),
        ("stratum", pa.string()),
        ("generator", pa.string()),
        ("parent_record_id", pa.string()),
        ("case_pair", pa.string()),
        ("feed", pa.int32()),
        ("n_batches", pa.int32()),
        ("depth2_edges", pa.int32()),
        ("e_core", pa.float64()),
        ("e_split", pa.float64()),
        ("library_id", pa.string()),
        ("sym_class", pa.string()),
        ("pattern", pa.string()),
        ("f_r", pa.float64()),
        ("f_q", pa.float64()),
        ("cbc_max", pa.float64()),
        ("cbc_boc", pa.float64()),
        ("cbc_kind", pa.string()),
        ("cyclen", pa.float64()),
        ("ao_abs", pa.float64()),
        ("cycle_burnup", pa.float64()),
        ("discharge_burnup", pa.float64()),
        ("max_assembly_burnup", pa.float64()),
        ("max_pin_burnup", pa.float64()),
        ("eoc_ppm", pa.float64()),
        ("delta_efpd", pa.float64()),
        ("n_cycles", pa.float64()),
        ("converged", pa.bool_()),
        ("converged_at_cap", pa.bool_()),
        ("tolerance_margin", pa.float64()),
        ("restart_provenance", pa.string()),
        ("valid", pa.bool_()),
        ("failure", pa.string()),
        ("maps_key", pa.string()),
        # ---- LATE_COLUMNS (append-only tail) ----
        ("node_peak", pa.float64()),
        ("map_cov", pa.float64()),
        ("max_rod_avg_burnup", pa.float64()),
    ]
)

# Guard: the dataclass field order and the pyarrow schema must stay in lockstep.
assert SCHEMA_COLUMNS == [f.name for f in PARQUET_SCHEMA], (
    "CanonicalRecord fields and PARQUET_SCHEMA columns are out of sync"
)
# Guard: schema growth is append-only at the tail (see LATE_COLUMNS).
assert SCHEMA_COLUMNS[:len(FROZEN_COLUMNS)] == list(FROZEN_COLUMNS), (
    "LATE_COLUMNS must be appended AFTER the frozen 36-column prefix"
)
assert len(FROZEN_COLUMNS) == 36, "the frozen record prefix is 36 columns"
