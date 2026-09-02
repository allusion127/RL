"""Physics featurization (plan sec. 4.4): shapes, channel inventory, known-value
spot checks, shuffle-chain origin tracing, transpose-augment equivariance, and
batch/throughput smokes over the real Dataset-A store."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from lpopt.data.fuel_types import FuelLibrary, core_enrichment_split
from lpopt.data.schema import SYM_CLASS, unpack_pattern
from lpopt.model.featurize import (
    SERVE_DATASET,
    SERVE_SYM_CLASS,
    _DATASET_A_LIBRARIES,
    serve_provenance,
    CHANNELS,
    CHANNELS_BY_SCHEMA,
    CHANNELS_V4,
    FeatureEncoder,
    NOMINAL_CYCLE_BURNUP_MWD_KG,
    RecordInputs,
    _FF_REF,
    _FF_SCALE,
    _U_MASS_REF,
    _U_MASS_SCALE,
    _V4_EXTRA,
    _V4_SCALES,
    library_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]          # 5_RL
STORE = REPO_ROOT / "data" / "store"
RECORDS = STORE / "records.parquet"
FUEL = STORE / "fuel_types.parquet"

_CH = {name: i for i, name in enumerate(CHANNELS)}
_CH4 = {name: i for i, name in enumerate(CHANNELS_V4)}


def _load(n: int | None = None):
    if not RECORDS.is_file() or not FUEL.is_file():
        pytest.skip("Dataset-A store not present")
    import pandas as pd

    df = pd.read_parquet(RECORDS)
    if n is not None:
        df = df.iloc[:n].copy()
    fl = FuelLibrary.from_parquet(FUEL)
    return df, fl


@pytest.fixture(scope="module")
def store():
    return _load()


# --------------------------------------------------------------------------- #
# shapes / dtypes / inventory
# --------------------------------------------------------------------------- #
def test_shapes_dtypes_and_inventory(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    cells, globals_ = enc.encode(df.iloc[0], fl)
    assert cells.shape == (len(CHANNELS), 19, 19)
    assert cells.dtype == np.float32
    assert globals_.shape == (len(enc.globals_names),)
    assert globals_.dtype == np.float32
    assert len(CHANNELS) == 26
    assert not np.isnan(cells).any()
    assert not np.isnan(globals_).any()


@pytest.mark.parametrize(
    "schema,n_channels,n_extra_globals",
    [("v2", 26, 0), ("v3", 26, 0), ("v4", 26 + len(_V4_EXTRA), 3)],
)
def test_encode_shapes_by_schema(store, schema, n_channels, n_extra_globals) -> None:
    """The channel inventory is schema-selected: v2/v3 keep the 26-tuple, v4
    appends _V4_EXTRA (index-stable) and 3 globals; all finite for every schema."""
    df, fl = store
    enc = FeatureEncoder(cond_schema=schema)
    assert enc.n_channels == n_channels
    assert len(enc.channels) == n_channels
    cells, globals_ = enc.encode(df.iloc[0], fl)
    assert cells.shape == (n_channels, 19, 19)
    assert cells.dtype == np.float32
    assert globals_.shape == (len(enc.globals_names),)
    assert not np.isnan(cells).any()
    assert not np.isnan(globals_).any()
    base = FeatureEncoder(cond_schema="v3")
    assert len(enc.globals_names) == len(base.globals_names) + n_extra_globals


def test_v4_channel_inventory_is_append_only() -> None:
    """v4 = the 26 base channels then _V4_EXTRA — the first 26 indices are byte-
    identical to v2/v3 so a v3 checkpoint keeps serving on its 26 channels."""
    assert len(CHANNELS) == 26
    assert CHANNELS_BY_SCHEMA["v2"] == CHANNELS
    assert CHANNELS_BY_SCHEMA["v3"] == CHANNELS
    assert CHANNELS_V4 == CHANNELS + _V4_EXTRA
    assert CHANNELS_V4[:26] == CHANNELS
    assert len(CHANNELS_V4) == 26 + len(_V4_EXTRA)
    assert len(set(CHANNELS_V4)) == len(CHANNELS_V4)      # unique names
    # the append includes the presence gate + the harvested signatures
    assert "origin_lattice_present" in _V4_EXTRA
    assert {"origin_enr_main", "origin_kinf10", "origin_boron_worth",
            "origin_adf_corner_g2"} <= set(_V4_EXTRA)


def test_channels_globals_lengths_match_arrays(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    cells, globals_ = enc.encode(df.iloc[3], fl)
    assert cells.shape[0] == len(CHANNELS)
    assert globals_.shape[0] == len(enc.globals_names)
    # names are unique and order-stable
    assert len(set(CHANNELS)) == len(CHANNELS)
    assert len(set(enc.globals_names)) == len(enc.globals_names)


def test_library_provenance_mapping() -> None:
    # ga80 letter library is Dataset B (free-69 orbit); everything else is A/rot61.
    # This is the HISTORICAL-extractor map and is deliberately frozen; the SERVE
    # path uses ``serve_provenance`` (below) instead.
    assert library_provenance("ga80") == ("B", "free69")
    for lib in ("260624", "5.8_5.1", "CPHA", "legacy_a", "unresolved:X"):
        assert library_provenance(lib) == ("A", "rot61")


def test_serve_provenance_is_the_campaign_write_stamp() -> None:
    """The serve path must stamp what ``verify.outcome_to_record`` will write."""
    assert SERVE_DATASET == "P" and SERVE_SYM_CLASS == SYM_CLASS == "rot61"
    # every campaign library (and any library added later) -> ("P", "rot61")
    for lib in ("ga80", "paramA", "CPHA", "some_future_library"):
        assert serve_provenance(lib) == ("P", "rot61")
    # the extract_a libraries keep their Dataset-A answer, byte-identical
    for lib in ("260624", "5.8_5.1", "legacy_a"):
        assert serve_provenance(lib) == ("A", "rot61")
        assert serve_provenance(lib)[0] == library_provenance(lib)[0]
    # the two flips this fixes (2026-08-29 train/serve forensic)
    assert library_provenance("ga80")[1] != serve_provenance("ga80")[1]
    assert library_provenance("paramA")[0] != serve_provenance("paramA")[0]


def test_dataset_a_libraries_matches_the_live_store() -> None:
    """``_DATASET_A_LIBRARIES`` is a hard-coded set; re-derive it from the store so
    it cannot go stale.  The invariant is that NO library mixes ``dataset == "A"``
    with a non-``"A"`` dataset, so the split is exactly a partition of libraries."""
    pd = pytest.importorskip("pandas")
    if not RECORDS.is_file():
        pytest.skip("store not present")
    df = pd.read_parquet(RECORDS, columns=["library_id", "dataset", "sym_class"])
    by_lib = df.groupby("library_id")["dataset"].agg(lambda s: set(s.astype(str)))
    a_libs, other = set(), set()
    for lib, kinds in by_lib.items():
        assert not (kinds == {"A"}) ^ ("A" in kinds), \
            f"library {lib!r} MIXES Dataset-A and campaign rows: {sorted(kinds)}"
        (a_libs if kinds == {"A"} else other).add(str(lib))
    assert a_libs <= _DATASET_A_LIBRARIES, \
        f"store has Dataset-A libraries not in the constant: {sorted(a_libs - _DATASET_A_LIBRARIES)}"
    assert not (other & _DATASET_A_LIBRARIES), \
        f"constant claims campaign libraries are Dataset A: {sorted(other & _DATASET_A_LIBRARIES)}"
    # and 'rot61' is what a campaign row carries — the free69 label is confined to
    # the historical extract_b ga80 harvest and never to a campaign row.
    assert set(df[df["dataset"] == "P"]["sym_class"].astype(str)) == {SYM_CLASS}
    assert set(df[df["sym_class"] != SYM_CLASS]["dataset"].astype(str)) <= {"B"}


def test_core_enrichment_split_reproduces_stored_e_core(store) -> None:
    """The shared feed-average recipe recomputes the *stored* e_core / e_split
    byte-for-byte from the pattern alone (train/serve conditioning parity).

    Extraction filled the store ``e_core`` column via this recipe; inference
    reconstructs it from ``(pattern, library_id)`` with the same function, so a
    served pattern conditions on exactly the value training saw.
    """
    df, fl = store
    checked = 0
    # cover every library present, not just Dataset A's dominant one.
    for lib, grp in df[df["e_core"].notna()].groupby("library_id"):
        for _, row in grp.head(25).iterrows():
            pat = unpack_pattern(str(row["pattern"]))
            e_core, e_split = core_enrichment_split(fl, str(lib), pat.batch_feed())
            assert e_core is not None, f"{lib}: helper returned None for stored e_core"
            assert e_core == pytest.approx(float(row["e_core"]), abs=1e-9)
            if row["e_split"] is not None and not np.isnan(row["e_split"]):
                assert e_split == pytest.approx(float(row["e_split"]), abs=1e-9)
            checked += 1
    assert checked > 0


def test_dataset_flag_gate_changes_global_length(store) -> None:
    df, fl = store
    with_flag = FeatureEncoder(include_dataset_flag=True)
    without = FeatureEncoder(include_dataset_flag=False)
    assert len(with_flag.globals_names) == len(without.globals_names) + 1
    assert "g_dataset_flag" in with_flag.globals_names
    assert "g_dataset_flag" not in without.globals_names
    _, g_on = with_flag.encode(df.iloc[0], fl)
    _, g_off = without.encode(df.iloc[0], fl)
    assert g_on.shape[0] == g_off.shape[0] + 1


# --------------------------------------------------------------------------- #
# known-value spot checks
# --------------------------------------------------------------------------- #
def test_center_cell_is_fresh_every_record(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    for k in range(0, min(len(df), 400)):
        cells, _ = enc.encode(df.iloc[k], fl)
        assert cells[_CH["occ_fresh"], 9, 9] == 1.0
        assert cells[_CH["pos_center"], 9, 9] == 1.0


def test_b1_260624_fresh_enrichment_channel(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    # find a record whose centre (slot 0) is a fresh B1 on library 260624
    hit = None
    for _, row in df.iterrows():
        if str(row["library_id"]) == "260624" and unpack_pattern(
            row["pattern"]
        ).items[0].batch == "B1":
            hit = row
            break
    if hit is None:
        pytest.skip("no 260624 B1-centre record")
    b1 = fl.get("B1", "260624")
    # cond_v3 default normalization (plan sec. 12.4): (e - 5.75) / 1.5.
    expected = (b1.u_avg_enrichment - enc._enr_ref) / enc._enr_scale
    sm = enc.encode_slot_matrix(hit, fl)
    assert enc.cond_schema == "v3"
    assert sm[_CH["origin_enrichment"], 0] == pytest.approx(expected, abs=1e-5)
    assert expected == pytest.approx((5.5023 - 5.75) / 1.5, abs=2e-3)
    # a v2 encoder reproduces the legacy (e - 5.4) / 0.6 channel byte-for-byte.
    enc_v2 = FeatureEncoder(cond_schema="v2")
    sm_v2 = enc_v2.encode_slot_matrix(hit, fl)
    assert sm_v2[_CH["origin_enrichment"], 0] == pytest.approx(
        (b1.u_avg_enrichment - 5.4) / 0.6, abs=1e-5)


def test_origin_tracing_burned_equals_fresh_origin(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    checked = 0
    for k in range(min(len(df), 60)):
        row = df.iloc[k]
        sm = enc.encode_slot_matrix(row, fl)
        items = unpack_pattern(row["pattern"]).items
        age, origin, _ = enc._trace_chain(items)
        for j in range(len(items)):
            if items[j].is_fresh:
                continue
            oj = origin[j]
            assert items[oj].is_fresh
            # the burned cell's origin-physics equals its fresh origin's
            for ch in ("origin_enrichment", "origin_n_gd", "origin_gd_wt"):
                assert sm[_CH[ch], j] == pytest.approx(sm[_CH[ch], oj], abs=1e-6)
            # residence age is a-priori and >= 2 for a burned cell
            assert age[j] >= 2
            assert sm[_CH["residence_age"], j] == pytest.approx(age[j] / 3.0, abs=1e-6)
            assert sm[_CH["nominal_burnup"], j] == pytest.approx(
                (age[j] - 1) * NOMINAL_CYCLE_BURNUP_MWD_KG / (3 * NOMINAL_CYCLE_BURNUP_MWD_KG),
                abs=1e-6,
            )
            checked += 1
        if checked > 50:
            break
    assert checked > 0


def test_kinf_channels_dormant_for_v2v3_active_for_v4(store) -> None:
    """The k-inf curve is a cond_v4 feature: v2/v3 keep origin_kinf* DORMANT (0)
    even against the harvested store (byte-identical to the pre-harvest training a
    v2/v3 champion saw, so it is never fed OOD raw k-inf), while a v4 encoder
    activates the same channels."""
    df, fl = store
    enc = FeatureEncoder()                       # default v3
    cells, _ = enc.encode(df.iloc[0], fl)
    assert cells[_CH["origin_kinf_present"]].sum() == 0.0
    assert cells[_CH["origin_kinf0"]].sum() == 0.0
    assert cells[_CH["origin_kinf20"]].sum() == 0.0
    assert cells[_CH["origin_bu_k1"]].sum() == 0.0
    # v4 activates them (df.iloc[0] is a harvested library row).
    enc4 = FeatureEncoder(cond_schema="v4")
    cells4, _ = enc4.encode(df.iloc[0], fl)
    assert cells4[_CH4["origin_kinf_present"]].sum() > 0.0
    assert cells4[_CH4["origin_kinf0"]].sum() != 0.0


# --------------------------------------------------------------------------- #
# cond_v4 result-based expansion
# --------------------------------------------------------------------------- #
def _b1_260624_record(df):
    for _, row in df.iterrows():
        if str(row["library_id"]) == "260624" and unpack_pattern(
            row["pattern"]
        ).items[0].batch == "B1":
            return row
    return None


def test_v4_kinf_normalization_o1_regression(store) -> None:
    """When fuel_types k-inf columns are FILLED, v4 O(1)-normalizes the k-inf curve
    ((k-1)/0.25, bu/30) while v2/v3 keep those channels DORMANT (0) — byte-
    identical to the pre-harvest training a v2/v3 champion saw, so enriching the
    fuel table never feeds an existing v2/v3 champion OOD raw k-inf."""
    df, _fl = store
    hit = _b1_260624_record(df)
    if hit is None:
        pytest.skip("no 260624 B1-centre record")
    # fresh (isolated) library so the in-memory k-inf fill never leaks to peers.
    fl = FuelLibrary.from_parquet(FUEL)
    b1 = fl.get("B1", "260624")
    b1.kinf0, b1.kinf10, b1.kinf20, b1.kinf30, b1.bu_k1 = 1.20, 1.12, 1.05, 0.98, 45.0

    e4 = FeatureEncoder(cond_schema="v4")
    sm4 = e4.encode_slot_matrix(hit, fl)
    assert sm4[_CH4["origin_kinf0"], 0] == pytest.approx((1.20 - 1.0) / 0.25)   # 0.8
    assert sm4[_CH4["origin_kinf20"], 0] == pytest.approx((1.05 - 1.0) / 0.25)  # 0.2
    assert sm4[_CH4["origin_kinf10"], 0] == pytest.approx((1.12 - 1.0) / 0.25)
    assert sm4[_CH4["origin_kinf30"], 0] == pytest.approx((0.98 - 1.0) / 0.25)
    assert sm4[_CH4["origin_bu_k1"], 0] == pytest.approx(45.0 / 30.0)           # 1.5
    assert sm4[_CH4["origin_kinf_present"], 0] == 1.0
    assert sm4[_CH4["origin_lattice_present"], 0] == 1.0

    # v2/v3 keep the k-inf curve DORMANT (0) even when the columns are filled —
    # byte-identical to the pre-harvest training a v2/v3 champion saw, so it is
    # never fed OOD raw k-inf when the fuel table is enriched.
    e3 = FeatureEncoder(cond_schema="v3")
    sm3 = e3.encode_slot_matrix(hit, fl)
    assert sm3[_CH["origin_kinf0"], 0] == 0.0
    assert sm3[_CH["origin_kinf20"], 0] == 0.0
    assert sm3[_CH["origin_bu_k1"], 0] == 0.0
    assert sm3[_CH["origin_kinf_present"], 0] == 0.0
    e2 = FeatureEncoder(cond_schema="v2")
    sm2 = e2.encode_slot_matrix(hit, fl)
    assert sm2[_CH["origin_kinf0"], 0] == 0.0
    assert sm2[_CH["origin_kinf_present"], 0] == 0.0


def test_v4_design_origin_channels_normalized(store) -> None:
    """The v4 design-axis + signature channels read the harvested fuel_types
    columns with the v3 enrichment envelope (enr) and the finalized population
    constants (u_mass / _V4_SCALES).  260624 is fully harvested, so every
    signature is finite/non-zero and the lattice-present gate is lit."""
    df, fl = store
    hit = _b1_260624_record(df)
    if hit is None:
        pytest.skip("no 260624 B1-centre record")
    enc = FeatureEncoder(cond_schema="v4")
    b1 = fl.get("B1", "260624")
    sm = enc.encode_slot_matrix(hit, fl)
    assert sm[_CH4["origin_enr_main"], 0] == pytest.approx(
        (b1.enr_main - enc._enr_ref) / enc._enr_scale, abs=1e-6)
    assert sm[_CH4["origin_enr_zone"], 0] == pytest.approx(
        (b1.enr_zone - enc._enr_ref) / enc._enr_scale, abs=1e-6)
    assert sm[_CH4["origin_u_mass"], 0] == pytest.approx(
        (b1.u_mass_g - _U_MASS_REF) / _U_MASS_SCALE, abs=1e-6)
    # harvested signatures land at their finalized-constant normalization.
    br, bs = _V4_SCALES["boron_worth"]
    assert sm[_CH4["origin_boron_worth"], 0] == pytest.approx(
        (b1.boron_worth - br) / bs, abs=1e-6)
    ar, as_ = _V4_SCALES["adf_corner_g2"]
    assert sm[_CH4["origin_adf_corner_g2"], 0] == pytest.approx(
        (b1.adf_corner_g2 - ar) / as_, abs=1e-6)
    zr, zs = _V4_SCALES["zone_pins"]
    assert sm[_CH4["origin_zone_pins"], 0] == pytest.approx(
        (b1.zone_pin_count - zr) / zs, abs=1e-6)          # 52 pins -> -1.0
    assert sm[_CH4["origin_ff_pin_max"], 0] == pytest.approx(
        (b1.ff_pin_max - _FF_REF) / _FF_SCALE, abs=1e-6)
    # every harvested signature is non-zero and the lattice-present gate is lit.
    for ch in ("origin_boron_worth", "origin_xs_a2", "origin_adf_corner_g2",
               "origin_ff_pin_max", "origin_zone_pins"):
        assert sm[_CH4[ch], 0] != 0.0
    assert sm[_CH4["origin_lattice_present"], 0] == 1.0


def test_v4_origin_tracing_extension(store) -> None:
    """Every v4-extra origin channel of a burned cell equals its fresh origin's
    (the same a-priori tracing as the existing origin_* channels)."""
    df, fl = store
    enc = FeatureEncoder(cond_schema="v4")
    v4_origin = [n for n in _V4_EXTRA if n != "origin_lattice_present"]
    checked = 0
    for k in range(min(len(df), 60)):
        row = df.iloc[k]
        sm = enc.encode_slot_matrix(row, fl)
        items = unpack_pattern(row["pattern"]).items
        _age, origin, _ = enc._trace_chain(items)
        for j in range(len(items)):
            if items[j].is_fresh:
                continue
            oj = origin[j]
            for ch in v4_origin:
                assert sm[_CH4[ch], j] == pytest.approx(sm[_CH4[ch], oj], abs=1e-6)
            checked += 1
        if checked > 40:
            break
    assert checked > 0


# --------------------------------------------------------------------------- #
# transpose augmentation (plan sec. 4.4) — globals invariant, per-channel
# multiset preserved (source-coordinate and rotation channels swap as pairs)
# --------------------------------------------------------------------------- #
_SWAP_PAIRS = (("shuffle_src_x", "shuffle_src_y"), ("shuffle_rot1", "shuffle_rot2"))
_SWAP_NAMES = {n for pair in _SWAP_PAIRS for n in pair}


def test_transpose_augment_equivariance(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    for k in range(min(len(df), 120)):
        row = df.iloc[k]
        cells, g0 = enc.encode(row, fl)
        t_cells, g1 = enc.augment_transpose(cells, g0, row, fl)
        # 1) global vector is invariant under the diagonal mirror
        assert np.allclose(g0, g1, atol=1e-6)
        # 2) every non-swap channel keeps its sorted multiset (position permuted)
        for name, i in _CH.items():
            if name in _SWAP_NAMES:
                continue
            assert np.allclose(
                np.sort(cells[i].ravel()), np.sort(t_cells[i].ravel())
            ), f"channel {name} multiset changed under transpose"
        # 3) the source-coord / rotation channels swap: union multiset preserved
        for a, b in _SWAP_PAIRS:
            u0 = np.sort(np.concatenate([cells[_CH[a]].ravel(), cells[_CH[b]].ravel()]))
            u1 = np.sort(np.concatenate([t_cells[_CH[a]].ravel(), t_cells[_CH[b]].ravel()]))
            assert np.allclose(u0, u1), f"swap-pair {a}/{b} union changed"


def test_v4_transpose_augment_equivariance(store) -> None:
    """The appended v4 channels auto-pass transpose equivariance: none are
    direction-paired, so each keeps its sorted multiset (the swap pairs are
    unchanged from v2/v3), and the v4 globals (whole-core fresh means) stay
    invariant under the diagonal mirror."""
    df, fl = store
    enc = FeatureEncoder(cond_schema="v4")
    for k in range(min(len(df), 80)):
        row = df.iloc[k]
        cells, g0 = enc.encode(row, fl)
        t_cells, g1 = enc.augment_transpose(cells, g0, row, fl)
        assert np.allclose(g0, g1, atol=1e-6)         # 13 globals incl. v4 extras
        for name, i in _CH4.items():
            if name in _SWAP_NAMES:
                continue
            assert np.allclose(
                np.sort(cells[i].ravel()), np.sort(t_cells[i].ravel())
            ), f"v4 channel {name} multiset changed under transpose"
        for a, b in _SWAP_PAIRS:
            u0 = np.sort(np.concatenate([cells[_CH4[a]].ravel(), cells[_CH4[b]].ravel()]))
            u1 = np.sort(np.concatenate([t_cells[_CH4[a]].ravel(), t_cells[_CH4[b]].ravel()]))
            assert np.allclose(u0, u1), f"swap-pair {a}/{b} union changed"


def test_transpose_is_an_involution_on_encoding(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    row = df.iloc[1]
    cells, g = enc.encode(row, fl)
    once_c, once_g = enc.augment_transpose(cells, g, row, fl)
    # transpose(transpose(pattern)) == pattern -> re-encoding twice returns origin
    from lpopt.data.geometry import transpose

    tt = transpose(transpose(unpack_pattern(row["pattern"])))
    assert tt.items == unpack_pattern(row["pattern"]).items


# --------------------------------------------------------------------------- #
# batch + throughput
# --------------------------------------------------------------------------- #
def test_batch_encode_200_rows(store) -> None:
    df, fl = store
    enc = FeatureEncoder()
    sub = df.iloc[:200]
    cells, globals_ = enc.encode_batch(sub, fl)
    assert cells.shape == (200, len(CHANNELS), 19, 19)
    assert globals_.shape == (200, len(enc.globals_names))
    # row 0 of the batch matches a single encode
    c0, g0 = enc.encode(sub.iloc[0], fl)
    assert np.array_equal(cells[0], c0)
    assert np.array_equal(globals_[0], g0)


def test_throughput_2000_rows_under_60s() -> None:
    df, fl = _load(2000)
    enc = FeatureEncoder()
    t0 = time.time()
    cells, _ = enc.encode_batch(df, fl)
    dt = time.time() - t0
    assert cells.shape[0] == len(df)
    assert dt < 60.0, f"2000-row featurization took {dt:.1f}s (budget 60s)"


def test_record_inputs_only_reads_safe_fields(store) -> None:
    df, fl = store
    # a RecordInputs coerced from a full row exposes only the safe fields
    inp = RecordInputs.coerce(df.iloc[0])
    assert not hasattr(inp, "cyclen")
    assert not hasattr(inp, "f_r")
    # coercing an already-coerced input is a no-op
    assert RecordInputs.coerce(inp) is inp
