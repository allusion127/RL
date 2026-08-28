"""Target-leakage acceptance (plan M2/M3): featurization must depend ONLY on
``(pattern, feed, e_core/e_split, case_pair, library_id, sym_class, dataset)``
and the static fuel table — never on a record's target / metric / map columns.

Two guarantees are pinned:

1. **Byte-identity** — encoding a full labelled row and the same row with every
   label/metric/map column dropped produce bit-for-bit identical arrays.
2. **Signature isolation** — the encoder can only ever *see* the safe fields:
   :class:`RecordInputs` carries no target attribute and its constructor rejects
   a target keyword.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpopt.data.fuel_types import FuelLibrary
from lpopt.model.featurize import FeatureEncoder, RecordInputs, SAFE_INPUT_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"
RECORDS = STORE / "records.parquet"
FUEL = STORE / "fuel_types.parquet"

#: Every column that is a target / metric / map / status / provenance field —
#: the encoder must never read any of these.
LABEL_COLUMNS = [
    "f_r", "f_q", "cbc_max", "cbc_boc", "cbc_kind", "cyclen", "ao_abs",
    "cycle_burnup", "discharge_burnup", "max_assembly_burnup", "max_pin_burnup",
    "eoc_ppm", "delta_efpd", "n_cycles", "converged", "converged_at_cap",
    "tolerance_margin", "restart_provenance", "valid", "failure", "maps_key",
    # map-derived flatness scalars (data.flatness) — a target, never an input
    "node_peak", "map_cov",
]


def _load(n: int = 64):
    if not RECORDS.is_file() or not FUEL.is_file():
        pytest.skip("Dataset-A store not present")
    import pandas as pd

    df = pd.read_parquet(RECORDS).iloc[:n].copy()
    fl = FuelLibrary.from_parquet(FUEL)
    return df, fl


# --------------------------------------------------------------------------- #
# 1) byte-identity with labels present vs stripped
# --------------------------------------------------------------------------- #
#: Every schema whose leakage-safety is asserted here.  The default (v3) is the
#: historical guarantee; ``v6`` is the champion's; ``v6b`` adds the regime
#: ``(library_id, feed)`` burnup table and the source-chain channels
#: (``data/reports/ab2_addendum_BU_20260810.md``) and must clear the SAME bar --
#: its new inputs are hard-coded constants selected by fields that are already in
#: :data:`SAFE_INPUT_FIELDS`, so no label may reach a feature bit.
#: ``v7`` adds the composition-moment globals, which read ``case_pair`` (already
#: a safe field) and ``Pattern.batch_feed`` (already the pattern) -- so it must
#: clear the SAME bar and is listed here rather than exempted.  ``v8`` is that
#: same block widened to a 5-type alphabet: identical inputs, wider padding, so
#: it clears the bar for the identical reason and is listed too.
LEAKAGE_SCHEMAS = ("v3", "v6", "v6b", "v6c", "v7", "v8")


@pytest.mark.parametrize("schema", LEAKAGE_SCHEMAS)
def test_encoding_byte_identical_with_labels_dropped(schema: str) -> None:
    df, fl = _load()
    stripped = df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])

    full_cells, full_g = FeatureEncoder(cond_schema=schema).encode_batch(df, fl)
    strip_cells, strip_g = FeatureEncoder(cond_schema=schema).encode_batch(
        stripped, fl)

    assert np.array_equal(full_cells, strip_cells)
    assert np.array_equal(full_g, strip_g)


@pytest.mark.parametrize("schema", LEAKAGE_SCHEMAS)
def test_encoding_ignores_corrupted_label_columns(schema: str) -> None:
    """Poisoning the labels must not perturb a single feature bit."""
    df, fl = _load()
    poisoned = df.copy()
    for col in ("f_r", "cbc_max", "cyclen", "ao_abs"):
        poisoned[col] = -999.0
    poisoned["converged"] = ~poisoned["converged"].astype(bool)
    # The burn state is the axis cond_v6b changes, so poison the burnup labels
    # explicitly: a regime table that ever consulted a record's OWN cycle_burnup
    # instead of the a-priori constant would be caught right here.
    for col in ("cycle_burnup", "discharge_burnup", "max_assembly_burnup"):
        if col in poisoned.columns:
            poisoned[col] = -999.0

    base_cells, base_g = FeatureEncoder(cond_schema=schema).encode_batch(df, fl)
    pois_cells, pois_g = FeatureEncoder(cond_schema=schema).encode_batch(
        poisoned, fl)
    assert np.array_equal(base_cells, pois_cells)
    assert np.array_equal(base_g, pois_g)


# --------------------------------------------------------------------------- #
# 2) signature isolation (negative control)
# --------------------------------------------------------------------------- #
def test_safe_fields_disjoint_from_labels() -> None:
    assert set(SAFE_INPUT_FIELDS).isdisjoint(set(LABEL_COLUMNS))


def test_record_inputs_has_no_target_attribute() -> None:
    fields = set(RecordInputs.__dataclass_fields__)
    assert fields == set(SAFE_INPUT_FIELDS)
    for label in LABEL_COLUMNS:
        assert not hasattr(RecordInputs(
            pattern="F:B1:0", feed=121, case_pair="B1_C2", library_id="260624"
        ), label)


def test_record_inputs_constructor_rejects_target_kwarg() -> None:
    with pytest.raises(TypeError):
        RecordInputs(
            pattern="F:B1:0", feed=121, case_pair="B1_C2",
            library_id="260624", cyclen=999.0,     # a target must not be accepted
        )


@pytest.mark.parametrize("schema", LEAKAGE_SCHEMAS)
def test_coerce_reads_only_safe_keys(schema: str) -> None:
    """A mapping whose ONLY safe keys are present encodes the same as a full row."""
    df, fl = _load(4)
    row = df.iloc[0]
    minimal = {k: row[k] for k in SAFE_INPUT_FIELDS}
    enc = FeatureEncoder(cond_schema=schema)
    c_full, g_full = enc.encode(row, fl)
    c_min, g_min = enc.encode(minimal, fl)
    assert np.array_equal(c_full, c_min)
    assert np.array_equal(g_full, g_min)
