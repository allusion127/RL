"""Graded (3..5-fresh-type) loading patterns — ``data/reports/tripletype_design_20260817.md``.

The machinery below grew from a 2-type *pair* (``A_B``) to a 3-type *graded*
case (``A_B_C``), and then — same day, operator directive "3~5종 그물망" — to a
5-type mesh (``A_B_C_D_E``).  Three obligations, and this suite exists to pin
all of them:

1. **The 2-type paths did not move.**  Every shipped record, every trained
   checkpoint and the standing champion are 2-type.  If the alphabet growth
   perturbed a single genome draw or a single feature bit, every comparison
   against that history would be measuring the refactor.  The genome and
   featurizer identities are pinned as *byte* tests (sha256 of a fixed-seed move
   transcript / of the encoded arrays), captured against the pre-change code.

2. **The 3-type paths did not move either.**  The 3 -> 5 widening raised a cap
   and added optional ``donor``/``target`` arguments to ``graded_morph``; the
   default draw is untouched.  The 3-type move transcript is pinned the same way
   (verified against a verbatim reconstruction of the 3-type build).

3. **3, 4 and 5 types actually work end to end** — the genome carries them, the
   ``%LPD_SHF`` round-trips them, the resolver ladder scores them, and the
   cond_v7 (3-wide) / cond_v8 (5-wide) globals describe them.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pytest

from lpopt.data.fuel_types import (
    FuelLibrary,
    case_e_core,
    mix_e_core,
    pair_e_core,
)
from lpopt.design.coredeck import build_cycle1_deck
from lpopt.model.featurize import (
    CHANNELS_BY_SCHEMA,
    CHANNELS_V6C,
    MAX_FRESH_TYPES,
    FeatureEncoder,
    _V7_GLOBALS_EXTRA,
    _V8_GLOBALS_EXTRA,
)
from lpopt.search.construct import CaseContext
from lpopt.search.genome import (
    MAX_FRESH_TYPES as GENOME_MAX_FRESH_TYPES,
    GeneralOrbitGenome,
    GenomeError,
    case_batches,
    graded_morph,
    mutate,
    random_genome,
)
from lpopt.vendor.masterrl.domain import Pattern
from lpopt.vendor.masterrl.ga import _pair_batches, sample_move_count

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"
RECORDS = STORE / "records.parquet"
FUEL = STORE / "fuel_types.parquet"

TRIPLE = ("AA", "AB", "AC")
TRIPLE_CASE = "AA_AB_AC"
QUAD = ("AA", "AB", "AC", "AD")
QUAD_CASE = "AA_AB_AC_AD"
PENTA = ("AA", "AB", "AC", "AD", "AE")
PENTA_CASE = "AA_AB_AC_AD_AE"
#: Composition-block width each schema PINS (v7 was shipped at 3 and must stay
#: there — a retrain is in flight against it; v8 is the 5-wide widening).
COMPOSITION_WIDTH = {"v7": 3, "v8": 5}


def _load():
    if not RECORDS.is_file() or not FUEL.is_file():
        pytest.skip("Dataset-A store not present")
    import pandas as pd

    return pd.read_parquet(RECORDS), FuelLibrary.from_parquet(FUEL)


@pytest.fixture(scope="module")
def store():
    return _load()


def _sample(df, n_per_lib: int = 6):
    import pandas as pd

    return pd.concat([g.head(n_per_lib) for _, g in df.groupby("library_id")]
                     ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1) the alphabet
# --------------------------------------------------------------------------- #
def test_case_batches_is_the_vendor_pair_for_two_members() -> None:
    for pair in ("K1_K2", "E1_E2", "N1_N2", "T6_T4", "A01_B01"):
        assert case_batches(pair) == _pair_batches(pair)


def test_case_batches_accepts_a_triple() -> None:
    assert case_batches(TRIPLE_CASE) == TRIPLE
    assert case_batches("G06N24_G10N24_G08N24") == (
        "G06N24", "G10N24", "G08N24")


def test_case_batches_accepts_four_and_five_members() -> None:
    """The operator-directed mesh is 3~5 types; the cap is 5."""
    assert GENOME_MAX_FRESH_TYPES == 5
    assert case_batches(QUAD_CASE) == QUAD
    assert case_batches(PENTA_CASE) == PENTA


@pytest.mark.parametrize(
    "bad",
    ["K1", "K1_", "_K2", "K1_K2_K3_K4_K5_K6", "K1_K2_K1", "K1__K3",
     "K1_K2_K3_K4_K1"],
)
def test_case_batches_rejects_malformed_cases(bad: str) -> None:
    with pytest.raises(GenomeError):
        case_batches(bad)


def test_case_batches_rejects_one_past_the_cap() -> None:
    """The cap is enforced at exactly MAX_FRESH_TYPES + 1, not somewhere near."""
    ok = "_".join(f"T{i}" for i in range(GENOME_MAX_FRESH_TYPES))
    assert len(case_batches(ok)) == GENOME_MAX_FRESH_TYPES
    with pytest.raises(GenomeError):
        case_batches(ok + f"_T{GENOME_MAX_FRESH_TYPES}")


# --------------------------------------------------------------------------- #
# 2) 2-type REGRESSION IDENTITY (byte)
# --------------------------------------------------------------------------- #
#: sha256 over a fixed-seed transcript of 2-type ``random_genome`` +
#: ``mutate`` canonical patterns, captured against the code as it stood BEFORE
#: the 3-type alphabet landed (2026-08-17).  A change here means a 2-type
#: campaign would no longer reproduce its own history -- never "just re-pin" it.
_GENOME_2TYPE_SHA256 = (
    "76a4149a3cc4be9bc2a3a67e5809bcf9347b0b6db50d18e14008e1dd2149996c")


def test_two_type_move_stream_is_byte_identical() -> None:
    h = hashlib.sha256()
    rng = random.Random(20260817)
    for pair in ("K1_K2", "E1_E2", "N1_N2"):
        for n in (24, 27, 29, 30):
            g = random_genome(rng, pair, n)
            h.update(g.to_pattern().canonical().encode())
            for _ in range(60):
                g = mutate(g, rng, sample_move_count(rng, 0.5),
                           batches=("K1", "K2"))
                h.update(g.to_pattern().canonical().encode())
    assert h.hexdigest() == _GENOME_2TYPE_SHA256


def test_graded_morph_is_a_no_op_for_a_two_type_alphabet() -> None:
    """The new operator must be unreachable from a pair — that is what keeps the
    2-type move stream above byte-identical.  Naming a donor/target explicitly
    must NOT open a back door into the pair path either."""
    g = random_genome(random.Random(2), "K1_K2", 29)
    for seed in range(50):
        assert graded_morph(g, random.Random(seed), ("K1", "K2")) is g
    assert graded_morph(g, random.Random(0), ("K1", "K2"),
                        donor="K1", target="K2") is g


# --------------------------------------------------------------------------- #
# 2b) 3-type REGRESSION IDENTITY (byte) — the 3 -> 5 widening moved nothing
# --------------------------------------------------------------------------- #
#: sha256 over a fixed-seed transcript of 3-type ``random_genome`` + ``mutate``
#: (which reaches ``graded_morph`` through ``_one_move``) plus 40 direct
#: default-draw ``graded_morph`` calls.  Captured by running this transcript
#: against a VERBATIM reconstruction of the 3-type ``graded_morph`` (the build
#: as it stood before ``donor``/``target`` and the cap raise) and confirming the
#: widened code reproduces it bit for bit.  A change here means a 3-type
#: campaign would no longer reproduce its own history.
_GENOME_3TYPE_SHA256 = (
    "a40d6886f536bd0f1ceb2c9f15fdcbf4ea8ffc9afc29cc5cb956f6ad8e148feb")


def test_three_type_move_stream_is_byte_identical() -> None:
    h = hashlib.sha256()
    rng = random.Random(20260817)
    for n in (24, 27, 29, 30):
        g = random_genome(rng, TRIPLE_CASE, n)
        h.update(g.to_pattern().canonical().encode())
        for _ in range(120):
            g = mutate(g, rng, sample_move_count(rng, 0.5), batches=TRIPLE)
            h.update(g.to_pattern().canonical().encode())
    for seed in range(40):
        parent = random_genome(random.Random(seed), "AA_AB", 29)
        child = graded_morph(parent, random.Random(seed), TRIPLE)
        h.update(child.to_pattern().canonical().encode())
    assert h.hexdigest() == _GENOME_3TYPE_SHA256


# --------------------------------------------------------------------------- #
# 3) 3-type genome invariants + fuzz
# --------------------------------------------------------------------------- #
def _assert_invariants(g: GeneralOrbitGenome) -> None:
    g.validate()
    n = g.n_fresh
    assert g.feed == 1 + 4 * n                      # feed arithmetic UNCHANGED
    if n <= 30:
        assert g.depth2_edge_count == 60 - 2 * n
        assert not g.unconsumed_fresh_units
    else:
        assert g.depth2_edge_count == 0
        assert len(g.unconsumed_fresh_units) == 2 * n - 60
    assert sum(g.batch_counts.values()) == n


@pytest.mark.parametrize("n", [24, 27, 29, 30])
def test_random_triple_genome_invariants(n: int) -> None:
    g = random_genome(random.Random(100 + n), TRIPLE_CASE, n)
    _assert_invariants(g)
    assert set(g.batch_counts) <= set(TRIPLE)


def test_triple_fuzz_mutations_closed() -> None:
    """5k mixed-operator moves over a 3-type alphabet, zero invariant violations."""
    rng = random.Random(20260817)
    starts = [(24, False), (27, False), (29, False), (30, False), (31, True)]
    genomes = [
        random_genome(rng, TRIPLE_CASE, n, allow_single_cycle_discharge=flag)
        for n, flag in starts
    ]
    total = 0
    per_start = 5_000 // len(genomes)
    for parent in genomes:
        _assert_invariants(parent)
        for _ in range(per_start):
            child = mutate(parent, rng, sample_move_count(rng, 0.5),
                           batches=TRIPLE)
            _assert_invariants(child)
            assert child != parent
            parent = child
            total += 1
    assert total >= 5_000


def test_fuzz_reaches_all_three_types() -> None:
    """The alphabet is not decorative: a mutation chain must actually populate
    the third type (otherwise the campaign would search a 2-type subspace)."""
    rng = random.Random(31337)
    g = random_genome(rng, TRIPLE_CASE, 29)
    seen: set[str] = set(g.batch_counts)
    for _ in range(300):
        g = mutate(g, rng, sample_move_count(rng, 0.5), batches=TRIPLE)
        seen |= set(g.batch_counts)
    assert seen == set(TRIPLE)


# --------------------------------------------------------------------------- #
# 4) graded morph (the 2-type -> 3-type elite-seeding path)
# --------------------------------------------------------------------------- #
def test_graded_morph_seeds_the_third_type_without_touching_structure() -> None:
    parent = random_genome(random.Random(5), "AA_AB", 29)
    assert set(parent.batch_counts) == {"AA", "AB"}
    child = graded_morph(parent, random.Random(1), TRIPLE)

    assert "AC" in child.batch_counts, "third type was not seeded"
    # Structure is untouched — a re-label is not a reload change.
    assert child.feed == parent.feed
    assert child.wiring == parent.wiring
    assert child.fresh_units == parent.fresh_units
    assert child.depth2_edge_count == parent.depth2_edge_count
    assert sum(child.batch_counts.values()) == sum(parent.batch_counts.values())
    child.validate()


def test_graded_morph_converts_a_radial_slice_of_one_donor() -> None:
    """The converted units come from ONE donor batch and are contiguous in
    radius — the move is a grading edit, not a scatter."""
    from lpopt.vendor.masterrl.ga import ORBIT_UNITS

    parent = random_genome(random.Random(9), "AA_AB", 30)
    child = graded_morph(parent, random.Random(4), TRIPLE)
    before = dict(parent.fresh)
    moved = [u for u, b in child.fresh if before[u] != b]
    assert moved, "graded morph did nothing"
    assert len({before[u] for u in moved}) == 1, "more than one donor batch"

    donor = before[moved[0]]
    donor_units = sorted((u for u, b in parent.fresh if b == donor),
                         key=lambda u: (ORBIT_UNITS[u].radius, u))
    k = len(moved)
    ranked = sorted(moved, key=lambda u: (ORBIT_UNITS[u].radius, u))
    assert ranked in (donor_units[:k], donor_units[-k:])


def test_graded_morph_leaves_at_least_one_donor_unit() -> None:
    """It must never empty a batch — that would collapse back to 2 types."""
    g = random_genome(random.Random(6), "AA_AB", 29)
    for seed in range(30):
        child = graded_morph(g, random.Random(seed), TRIPLE)
        assert all(n >= 1 for n in child.batch_counts.values())


def test_mutate_reaches_the_third_type_from_a_two_type_elite() -> None:
    """The campaign's actual cold-start path: a 2-type store elite mutated under
    a 3-type alphabet must *reliably* pick up the third type.  If it did not, a
    3-type campaign seeded from 2-type elites would quietly search the 2-type
    subspace forever."""
    hits = 0
    for seed in range(40):
        rng = random.Random(seed)
        g = random_genome(rng, "AA_AB", 29)
        for _ in range(40):
            g = mutate(g, rng, 1, batches=TRIPLE)
            if "AC" in g.batch_counts:
                hits += 1
                break
    assert hits >= 35, f"only {hits}/40 elite seeds reached the third type"


def test_repeated_graded_morphs_converge_toward_a_balanced_split() -> None:
    g = random_genome(random.Random(8), "AA_AB", 30)
    rng = random.Random(0)
    for _ in range(12):
        g = graded_morph(g, rng, TRIPLE)
        g.validate()
    counts = g.batch_counts
    assert set(counts) == set(TRIPLE)
    assert max(counts.values()) - min(counts.values()) <= 6, counts


# --------------------------------------------------------------------------- #
# 5) %LPD_SHF round trip with a synthetic 3-type pattern
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [27, 29, 30])
def test_triple_pattern_shf_round_trip(n: int) -> None:
    g = random_genome(random.Random(200 + n), TRIPLE_CASE, n)
    assert len(g.batch_counts) == 3, g.batch_counts

    pattern = g.to_pattern()
    shf = pattern.to_shf()
    reparsed = Pattern.parse(shf)

    assert reparsed.to_shf() == shf
    assert reparsed.canonical() == pattern.canonical()
    assert reparsed.digest == pattern.digest
    assert set(reparsed.batch_feed()) == set(TRIPLE)
    assert sum(reparsed.batch_feed().values()) == reparsed.feed == 1 + 4 * n

    back = GeneralOrbitGenome.from_pattern(reparsed)
    assert back == g
    assert back.to_pattern().to_shf() == shf


def test_triple_pattern_passes_vendor_case_validation() -> None:
    g = random_genome(random.Random(77), TRIPLE_CASE, 29)
    g.to_pattern().validate_case(TRIPLE_CASE, 117)   # raises on a batch mismatch


def test_case_context_exposes_three_batches() -> None:
    ctx = CaseContext(pair=TRIPLE_CASE, feed=117)
    assert ctx.batches == TRIPLE
    assert ctx.resolved_center == "AA"
    assert ctx.n_fresh == 29
    assert ctx.case_key.folder == f"{TRIPLE_CASE}_f117"


# --------------------------------------------------------------------------- #
# 6) composition-weighted e_core (the resolver's level-3 ladder rung)
# --------------------------------------------------------------------------- #
def test_mix_e_core_reduces_to_pair_e_core(store) -> None:
    _, fl = store
    a, b = fl.get("K1", "ga80"), fl.get("K2", "ga80")
    assert mix_e_core([a, b]) == pair_e_core(a, b, 0.5)
    for split in (0.25, 0.4, 0.75):
        assert mix_e_core([a, b], [split, 1.0 - split]) == pair_e_core(a, b, split)


def test_case_e_core_of_a_triple_is_between_its_extremes(store) -> None:
    _, fl = store
    members = ["K1", "K2", "J1"]
    enrs = [fl.get(m, "ga80").u_avg_enrichment for m in members]
    e = case_e_core(fl, members, "ga80")
    assert min(enrs) <= e <= max(enrs)
    # An equal 3-way split is NOT the 2-member answer — the third type moves it.
    assert not math.isclose(e, pair_e_core(fl.get("K1", "ga80"),
                                           fl.get("K2", "ga80"), 0.5))


def test_resolver_scores_a_triple_case(store, tmp_path) -> None:
    """A triple cell must produce a finite e_core so the level-3 (pair_ecore)
    rung can rank restarts for it instead of falling through to neutral."""
    from lpopt.search.assets import CaseAssetResolver

    _, fl = store
    res = CaseAssetResolver(tmp_path, tmp_path / "promoted",
                            fuel_library=fl, library_id="ga80")
    e_pair = res._pair_e_core("K1_K2")
    e_triple = res._pair_e_core("K1_K2_J1")
    assert e_pair is not None and e_triple is not None
    assert e_triple == case_e_core(fl, ["K1", "K2", "J1"], "ga80")
    # 4- and 5-member cells must score on the SAME rung, not fall to neutral.
    for members in (["K1", "K2", "J1", "J2"], ["K1", "K2", "J1", "J2", "E1"]):
        e = res._pair_e_core("_".join(members))
        assert e is not None
        assert e == case_e_core(fl, members, "ga80")
    # ... and one past the cap still refuses to score.
    assert res._pair_e_core("K1_K2_J1_J2_E1_E2") is None


def test_alias_case_key_maps_every_triple_member(store, tmp_path) -> None:
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.vendor.masterrl.domain import CaseKey

    _, fl = store
    res = CaseAssetResolver(tmp_path, tmp_path / "promoted", fuel_library=fl,
                            registry_aliases={"T1": "AA", "T2": "AB", "T3": "AC"})
    assert res.alias_case_key(CaseKey("T1_T2_T3", 117)) == CaseKey("AA_AB_AC", 117)
    assert res.alias_case_key(CaseKey("T1_T2", 121)) == CaseKey("AA_AB", 121)


# --------------------------------------------------------------------------- #
# 7) cond_v7: additive globals, 2-type featurization identity
# --------------------------------------------------------------------------- #
def test_v7_cell_inventory_is_v6c_verbatim() -> None:
    assert CHANNELS_BY_SCHEMA["v7"] == CHANNELS_V6C
    assert len(CHANNELS_BY_SCHEMA["v7"]) == 62


def test_v7_globals_are_append_only_after_v6c() -> None:
    g6 = FeatureEncoder(cond_schema="v6c").globals_names
    g7 = FeatureEncoder(cond_schema="v7").globals_names
    assert g7[:len(g6)] == g6
    assert g7 == g6 + _V7_GLOBALS_EXTRA
    assert len(g7) == 18
    assert len(set(g7)) == len(g7)


def test_composition_block_absent_from_every_earlier_schema() -> None:
    for schema in ("v3", "v4", "v5", "v6", "v6_prior", "v6_contrast",
                   "v6b", "v6c"):
        names = set(FeatureEncoder(cond_schema=schema).globals_names)
        assert not (names & set(_V7_GLOBALS_EXTRA))


#: sha256 of the v6c cells / globals over a fixed store sample, captured against
#: the pre-change code.  This is the leakage-style byte test for the 2-type
#: featurization: adding the composition globals must not move ONE bit of what
#: the standing champion sees.
_FEAT_V6C_CELLS_SHA256 = (
    "a7580feb67eb0d7555969a55d9c900814f6e4cf86211ee65269f21586061624e")
_FEAT_V6C_GLOBALS_SHA256 = (
    "016b518047f04afc16c2a01ed2318e1b286cbfbce846819902bad8c0950bcd11")

#: The 30 rows the two hashes above were frozen over — ``_sample(df)`` (6 per
#: ``library_id``) of the LIVE store as it stood when they were captured —
#: snapshotted to a fixture holding only :data:`SAFE_INPUT_FIELDS` (plus
#: ``record_id`` for identification).  Reading the live store here would make the
#: hashes move every time the store gains rows, which says nothing about
#: featurization; the guard is "byte-identical output for a FIXED input", so the
#: input is pinned.  Reconstructed from
#: ``records.parquet.bak_pre_ecore_backfill_20260829`` (the newest backup that
#: still reproduces the frozen globals hash — the e_core backfill that followed
#: it rewrote ``e_core`` on some of these rows).
_FEAT_V6C_SAMPLE = Path(__file__).resolve().parent / "data" / "v6c_featurization_sample.parquet"
_FEAT_V6C_SAMPLE_ROWS = 30
#: sha256 over "\n".join(record_id) of the fixture, so a silent re-generation
#: (different rows, different order) fails loudly instead of moving the hashes.
_FEAT_V6C_SAMPLE_IDS_SHA256 = (
    "5e37bd8cea0090b2a089f7865ad7082761051cf07a27a2452517c7a4957d3965")


def test_v6c_featurization_is_byte_identical() -> None:
    if not FUEL.is_file():
        pytest.skip("Dataset-A store not present")
    import pandas as pd

    sample = pd.read_parquet(_FEAT_V6C_SAMPLE)
    assert len(sample) == _FEAT_V6C_SAMPLE_ROWS
    assert hashlib.sha256(
        "\n".join(sample["record_id"].astype(str)).encode()
    ).hexdigest() == _FEAT_V6C_SAMPLE_IDS_SHA256

    fl = FuelLibrary.from_parquet(FUEL)
    cells, gvec = FeatureEncoder(cond_schema="v6c").encode_batch(sample, fl)
    assert hashlib.sha256(
        np.ascontiguousarray(cells).tobytes()).hexdigest() == _FEAT_V6C_CELLS_SHA256
    assert hashlib.sha256(
        np.ascontiguousarray(gvec).tobytes()).hexdigest() == _FEAT_V6C_GLOBALS_SHA256


def test_two_type_records_featurize_identically_under_v7(store) -> None:
    df, fl = store
    a = FeatureEncoder(cond_schema="v6c")
    b = FeatureEncoder(cond_schema="v7")
    n6 = len(a.globals_names)
    for _, row in _sample(df).iterrows():
        ca, ga = a.encode(row, fl)
        cb, gb = b.encode(row, fl)
        assert np.array_equal(ca, cb), "v7 perturbed a v6c channel"
        assert np.array_equal(ga, gb[:n6]), "v7 perturbed a v6c global"


@pytest.mark.parametrize("schema", ["v7", "v8"])
def test_composition_globals_reduce_to_the_pair_values(store, schema: str) -> None:
    """For a 2-type record the appended block must AGREE with the pair globals it
    generalizes — that is what makes v7/v8 a superset rather than a rewrite."""
    df, fl = store
    width = COMPOSITION_WIDTH[schema]
    enc = FeatureEncoder(cond_schema=schema)
    for _, row in _sample(df, 4).iterrows():
        gv = dict(zip(enc.globals_names, enc.encode(row, fl)[1]))
        assert gv["g_type_frac_1"] == pytest.approx(gv["g_split_frac"])
        assert gv["g_type_frac_1"] + gv["g_type_frac_2"] == pytest.approx(1.0)
        for k in range(3, width + 1):
            assert gv[f"g_type_frac_{k}"] == 0.0
        assert 0.0 < gv["g_n_fresh_types"] <= 2.0 / width + 1e-6


def _graded_record(case: str, n: int = 29, *, seed: int = 4242,
                   n_types: int | None = None) -> dict:
    """A synthetic store row for an ``n``-fresh-unit core over ``case``."""
    g = random_genome(random.Random(seed), case, n)
    if n_types is not None:
        assert len(g.batch_counts) == n_types, g.batch_counts
    return {
        "pattern": g.to_pattern().canonical(),
        "feed": 1 + 4 * n,
        "case_pair": case,
        "library_id": "ga80",
        "e_core": None,
        "e_split": None,
        "sym_class": "free69",
        "dataset": "B",
    }


def _triple_record(n: int = 29, case: str = "K1_K2_J1") -> dict:
    g = random_genome(random.Random(4242), case, n)
    assert len(g.batch_counts) == 3
    return {
        "pattern": g.to_pattern().canonical(),
        "feed": 1 + 4 * n,
        "case_pair": case,
        "library_id": "ga80",
        "e_core": None,
        "e_split": None,
        "sym_class": "free69",
        "dataset": "B",
    }


def test_v7_encodes_a_three_type_record(store) -> None:
    _, fl = store
    enc = FeatureEncoder(cond_schema="v7")
    cells, gvec = enc.encode(_triple_record(), fl)
    assert cells.shape == (62, 19, 19)
    assert gvec.shape == (18,)
    assert np.isfinite(cells).all() and np.isfinite(gvec).all()

    gv = dict(zip(enc.globals_names, gvec))
    fracs = [gv[f"g_type_frac_{k}"] for k in (1, 2, 3)]
    assert all(f > 0.0 for f in fracs), fracs
    assert sum(fracs) == pytest.approx(1.0)
    assert gv["g_n_fresh_types"] == pytest.approx(1.0)   # 3 / 3
    assert gv["g_e_type_std"] > 0.0


def test_grading_moves_the_std_but_not_the_spread(store) -> None:
    """The discriminating property of the new block: a 3-type feed and a 2-type
    feed can share ``e_split`` (max-min) while their composition std differs."""
    _, fl = store
    enc = FeatureEncoder(cond_schema="v7")

    triple = enc.encode(_triple_record(case="K1_K2_J1"), fl)[1]
    gv3 = dict(zip(enc.globals_names, triple))

    pair_rec = _triple_record(case="K1_K2_J1")
    g = random_genome(random.Random(4242), "K1_J1", 29)
    pair_rec["pattern"] = g.to_pattern().canonical()
    pair_rec["case_pair"] = "K1_J1"
    gv2 = dict(zip(enc.globals_names, enc.encode(pair_rec, fl)[1]))

    # Same extreme types -> same max-min spread ...
    assert gv3["g_e_split"] == pytest.approx(gv2["g_e_split"])
    # ... but the intermediate type pulls the second moment down.
    assert gv3["g_e_type_std"] < gv2["g_e_type_std"]


# --------------------------------------------------------------------------- #
# 8) coredeck bootstrap map
# --------------------------------------------------------------------------- #
#: sha256 of the 2-type cycle-1 bootstrap deck, captured pre-change.
_CYCLE1_2TYPE_SHA256 = (
    "086434d4b8a839227e2cc2a46334711d5a2bc72dfc4e728c997b5453c83e26b9")


def test_two_type_cycle1_deck_is_byte_identical() -> None:
    text = build_cycle1_deck(["AA", "AB", "AC"], ("AA", "AB"))
    assert hashlib.sha256(text.encode()).hexdigest() == _CYCLE1_2TYPE_SHA256


def test_three_type_cycle1_deck_feeds_all_three() -> None:
    text = build_cycle1_deck(["AA", "AB", "AC"], TRIPLE)
    body = text.split("%LPD_BCH")[1].split("%LPD_B&C")[0]
    tokens = body.split()
    for alias in TRIPLE:
        assert alias in tokens, f"{alias} missing from the fresh-core map"
    # The deck must still parse as the same shape as the 2-type one.
    two = build_cycle1_deck(["AA", "AB", "AC"], ("AA", "AB"))
    assert len(text.splitlines()) == len(two.splitlines())


def test_cycle1_deck_rejects_an_out_of_range_alphabet() -> None:
    with pytest.raises(ValueError):
        build_cycle1_deck(["AA", "AB"], ("AA",))
    with pytest.raises(ValueError):
        build_cycle1_deck(list(PENTA) + ["AF"],
                          tuple(PENTA) + ("AF",))       # 6 > the cap


@pytest.mark.parametrize("alphabet", [QUAD, PENTA])
def test_cycle1_deck_feeds_every_member_of_a_wide_alphabet(alphabet) -> None:
    """The X1 zone round-robins, so no member of a 4/5-type bootstrap core may
    be missing (a dropped member would seed a core the case id lies about)."""
    aliases = list(PENTA)
    text = build_cycle1_deck(aliases, alphabet)
    tokens = text.split("%LPD_BCH")[1].split("%LPD_B&C")[0].split()
    for alias in alphabet:
        assert alias in tokens, f"{alias} missing from the fresh-core map"
    for absent in set(PENTA) - set(alphabet):
        assert absent not in tokens, f"{absent} leaked into the map"
    # Same shape as the 2-type deck over the same alias roster.
    assert (len(text.splitlines())
            == len(build_cycle1_deck(aliases, ("AA", "AB")).splitlines()))


# --------------------------------------------------------------------------- #
# 9) 4- and 5-type alphabets (operator directive "3~5종 그물망")
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alphabet", [QUAD, PENTA])
@pytest.mark.parametrize("n", [24, 27, 29, 30])
def test_wide_random_genome_invariants(alphabet, n: int) -> None:
    case = "_".join(alphabet)
    g = random_genome(random.Random(700 + n + len(alphabet)), case, n)
    _assert_invariants(g)
    assert set(g.batch_counts) <= set(alphabet)


@pytest.mark.parametrize("alphabet", [QUAD, PENTA])
def test_wide_fuzz_mutations_closed(alphabet) -> None:
    """5k mixed-operator moves over a 4/5-type alphabet, zero invariant
    violations — the feed arithmetic (1+4N, 60-2N, strict consumption) is
    batch-label-blind and must stay exactly so as the alphabet grows."""
    case = "_".join(alphabet)
    rng = random.Random(20260817 + len(alphabet))
    starts = [(24, False), (27, False), (29, False), (30, False), (31, True)]
    genomes = [
        random_genome(rng, case, n, allow_single_cycle_discharge=flag)
        for n, flag in starts
    ]
    total = 0
    per_start = 5_000 // len(genomes)
    for parent in genomes:
        _assert_invariants(parent)
        for _ in range(per_start):
            child = mutate(parent, rng, sample_move_count(rng, 0.5),
                           batches=alphabet)
            _assert_invariants(child)
            assert child != parent
            parent = child
            total += 1
    assert total >= 5_000


@pytest.mark.parametrize("alphabet", [QUAD, PENTA])
def test_fuzz_reaches_every_type_of_a_wide_alphabet(alphabet) -> None:
    """A 4/5-type campaign must not quietly search a narrower subspace."""
    case = "_".join(alphabet)
    rng = random.Random(4242 + len(alphabet))
    g = random_genome(rng, case, 29)
    seen: set[str] = set(g.batch_counts)
    for _ in range(400):
        g = mutate(g, rng, sample_move_count(rng, 0.5), batches=alphabet)
        seen |= set(g.batch_counts)
    assert seen == set(alphabet)


@pytest.mark.parametrize("n", [27, 29, 30])
def test_penta_pattern_shf_round_trip(n: int) -> None:
    """The %LPD_SHF encodes a type per POSITION, so a 5-type core must round-trip
    with no deck change at all (canonical / digest / from_pattern all agree)."""
    g = random_genome(random.Random(500 + n), PENTA_CASE, n)
    assert len(g.batch_counts) == 5, g.batch_counts

    pattern = g.to_pattern()
    shf = pattern.to_shf()
    reparsed = Pattern.parse(shf)

    assert reparsed.to_shf() == shf
    assert reparsed.canonical() == pattern.canonical()
    assert reparsed.digest == pattern.digest
    assert set(reparsed.batch_feed()) == set(PENTA)
    assert sum(reparsed.batch_feed().values()) == reparsed.feed == 1 + 4 * n

    back = GeneralOrbitGenome.from_pattern(reparsed)
    assert back == g
    assert back.to_pattern().to_shf() == shf


def test_penta_pattern_passes_vendor_case_validation() -> None:
    g = random_genome(random.Random(88), PENTA_CASE, 29)
    g.to_pattern().validate_case(PENTA_CASE, 117)


@pytest.mark.parametrize("case,alphabet",
                         [(QUAD_CASE, QUAD), (PENTA_CASE, PENTA)])
def test_case_context_exposes_a_wide_alphabet(case: str, alphabet) -> None:
    ctx = CaseContext(pair=case, feed=117)
    assert ctx.batches == alphabet
    assert ctx.resolved_center == alphabet[0]
    assert ctx.n_fresh == 29
    assert ctx.case_key.folder == f"{case}_f117"


def test_mix_e_core_handles_a_five_way_composition(store) -> None:
    _, fl = store
    members = ["K1", "K2", "J1", "J2", "E1"]
    enrs = [fl.get(m, "ga80").u_avg_enrichment for m in members]
    e = case_e_core(fl, members, "ga80")
    assert min(enrs) <= e <= max(enrs)
    # A skewed composition must move it off the equal-split answer.
    skew = case_e_core(fl, members, "ga80", [0.6, 0.1, 0.1, 0.1, 0.1])
    assert not math.isclose(e, skew)


def test_alias_case_key_maps_every_member_of_a_wide_case(tmp_path) -> None:
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.vendor.masterrl.domain import CaseKey

    res = CaseAssetResolver(
        tmp_path, tmp_path / "promoted",
        registry_aliases={"T1": "AA", "T2": "AB", "T3": "AC",
                          "T4": "AD", "T5": "AE"})
    assert res.alias_case_key(CaseKey("T1_T2_T3_T4_T5", 117)) == CaseKey(
        PENTA_CASE, 117)
    assert res.alias_case_key(CaseKey("T1_T2_T3_T4", 117)) == CaseKey(
        QUAD_CASE, 117)
    # One past the cap is left ALONE (fails safe, no half-mapped case id).
    six = CaseKey("T1_T2_T3_T4_T5_T1", 117)
    assert res.alias_case_key(six) == six


# --------------------------------------------------------------------------- #
# 10) graded_morph over a wide alphabet — reachability + any-pair conversion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alphabet", [QUAD, PENTA])
def test_repeated_graded_morphs_walk_the_whole_alphabet(alphabet) -> None:
    """The default draw targets the LEAST populated member, so repeated morphs
    from a 2-type elite must fill every absent type in turn — that is the
    cold-start path for a 4/5-type campaign."""
    g = random_genome(random.Random(11), "AA_AB", 30)
    rng = random.Random(3)
    for _ in range(24):
        g = graded_morph(g, rng, alphabet)
        g.validate()
    assert set(g.batch_counts) == set(alphabet)
    assert all(v >= 1 for v in g.batch_counts.values())


@pytest.mark.parametrize("alphabet", [TRIPLE, QUAD, PENTA])
def test_graded_morph_preserves_structure_on_a_wide_alphabet(alphabet) -> None:
    case = "_".join(alphabet)
    parent = random_genome(random.Random(13), case, 29)
    child = graded_morph(parent, random.Random(2), alphabet)
    assert child.feed == parent.feed
    assert child.wiring == parent.wiring
    assert child.fresh_units == parent.fresh_units
    assert child.depth2_edge_count == parent.depth2_edge_count
    assert sum(child.batch_counts.values()) == sum(parent.batch_counts.values())
    child.validate()


def test_graded_morph_converts_between_any_named_pair() -> None:
    """The generalized operator: name ANY ordered pair of the alphabet and it
    moves a radial slice from the first onto the second, leaving every other
    type alone."""
    from lpopt.vendor.masterrl.ga import ORBIT_UNITS

    g = random_genome(random.Random(17), PENTA_CASE, 30)
    counts = g.batch_counts
    pairs = [(d, t) for d in PENTA for t in PENTA
             if d != t and counts.get(d, 0) >= 2]
    assert len(pairs) >= 12, counts
    for donor, target in pairs:
        child = graded_morph(g, random.Random(7), PENTA,
                             donor=donor, target=target)
        before = dict(g.fresh)
        moved = [u for u, b in child.fresh if before[u] != b]
        assert moved, f"{donor}->{target} did nothing"
        assert {before[u] for u in moved} == {donor}
        assert {b for u, b in child.fresh if before[u] != b} == {target}
        # untouched types keep their exact unit sets
        for other in set(PENTA) - {donor, target}:
            assert ({u for u, b in child.fresh if b == other}
                    == {u for u, b in g.fresh if b == other})
        # the slice is contiguous in radius and never empties the donor
        donor_units = sorted((u for u, b in g.fresh if b == donor),
                             key=lambda u: (ORBIT_UNITS[u].radius, u))
        k = len(moved)
        assert k < len(donor_units)
        ranked = sorted(moved, key=lambda u: (ORBIT_UNITS[u].radius, u))
        assert ranked in (donor_units[:k], donor_units[-k:])


def test_graded_morph_refuses_an_unusable_named_pair() -> None:
    g = random_genome(random.Random(19), "AA_AB", 29)
    # donor absent from the genome -> nothing to convert
    assert graded_morph(g, random.Random(0), PENTA, donor="AE",
                        target="AC") is g
    # target outside the declared alphabet -> refuse rather than invent a type
    assert graded_morph(g, random.Random(0), PENTA, donor="AA",
                        target="ZZ") is g


# --------------------------------------------------------------------------- #
# 11) cond_v8 — the 5-wide composition block
# --------------------------------------------------------------------------- #
def test_v8_cell_inventory_is_v6c_verbatim() -> None:
    assert CHANNELS_BY_SCHEMA["v8"] == CHANNELS_V6C
    assert len(CHANNELS_BY_SCHEMA["v8"]) == 62


def test_v8_globals_are_v6c_plus_the_five_wide_block() -> None:
    g6 = FeatureEncoder(cond_schema="v6c").globals_names
    g8 = FeatureEncoder(cond_schema="v8").globals_names
    assert g8[:len(g6)] == g6                  # v6c prefix is index-stable
    assert g8 == g6 + _V8_GLOBALS_EXTRA
    assert len(g8) == 20
    assert len(set(g8)) == len(g8)
    assert _V8_GLOBALS_EXTRA == (
        "g_type_frac_1", "g_type_frac_2", "g_type_frac_3",
        "g_type_frac_4", "g_type_frac_5",
        "g_e_type_std", "g_n_fresh_types")
    # v8 carries every name v7 carries (a widening, not a rename).
    assert set(_V7_GLOBALS_EXTRA) <= set(_V8_GLOBALS_EXTRA)


def test_v7_is_untouched_by_the_widening() -> None:
    """A v7 retrain is in flight: its block must stay 3 wide and its
    ``g_n_fresh_types`` must stay normalized on 3."""
    enc = FeatureEncoder(cond_schema="v7")
    assert len(enc.globals_names) == 18
    assert "g_type_frac_4" not in enc.globals_names
    assert enc._composition_width == 3
    assert FeatureEncoder(cond_schema="v8")._composition_width == 5
    assert MAX_FRESH_TYPES == 5


def test_composition_block_absent_from_every_pre_v7_schema() -> None:
    for schema in ("v3", "v4", "v5", "v6", "v6_prior", "v6_contrast",
                   "v6b", "v6c"):
        names = set(FeatureEncoder(cond_schema=schema).globals_names)
        assert not (names & set(_V8_GLOBALS_EXTRA))


def test_two_type_records_featurize_identically_under_v8(store) -> None:
    """The v6c cells and the v6c global prefix must not move one bit — that is
    what lets the ~39k shipped pair records seed a v8 retrain."""
    df, fl = store
    a = FeatureEncoder(cond_schema="v6c")
    b = FeatureEncoder(cond_schema="v8")
    n6 = len(a.globals_names)
    for _, row in _sample(df).iterrows():
        ca, ga = a.encode(row, fl)
        cb, gb = b.encode(row, fl)
        assert np.array_equal(ca, cb), "v8 perturbed a v6c channel"
        assert np.array_equal(ga, gb[:n6]), "v8 perturbed a v6c global"


@pytest.mark.parametrize("case,n_types",
                         [("K1_K2", 2), ("K1_K2_J1", 3)])
def test_v8_carries_the_same_information_as_v7_for_narrow_cases(
        store, case: str, n_types: int) -> None:
    """The obligation that makes v8 additive: a 2-type AND a 3-type record must
    featurize under v8 with exactly the information they had under v7 — same
    cells, same v6c globals, same fractions 1..3, same std; only the padding
    (frac_4/5 = 0) and the ``g_n_fresh_types`` divisor differ."""
    _, fl = store
    rec = _graded_record(case, n_types=n_types)
    e7, e8 = FeatureEncoder(cond_schema="v7"), FeatureEncoder(cond_schema="v8")
    c7, g7 = e7.encode(rec, fl)
    c8, g8 = e8.encode(rec, fl)
    assert np.array_equal(c7, c8)

    v7 = dict(zip(e7.globals_names, g7))
    v8 = dict(zip(e8.globals_names, g8))
    for name in e7.globals_names:
        if name == "g_n_fresh_types":
            continue                      # divisor differs by construction
        assert v8[name] == pytest.approx(v7[name]), name
    assert v8["g_type_frac_4"] == 0.0
    assert v8["g_type_frac_5"] == 0.0
    # cardinality survives the re-normalization exactly.
    assert (v8["g_n_fresh_types"] * 5.0
            == pytest.approx(v7["g_n_fresh_types"] * 3.0))
    assert v8["g_n_fresh_types"] == pytest.approx(n_types / 5.0)


@pytest.mark.parametrize("case,alphabet",
                         [(QUAD_CASE, QUAD), (PENTA_CASE, PENTA)])
def test_v8_encodes_a_wide_record(store, case: str, alphabet) -> None:
    _, fl = store
    # The synthetic alphabet must resolve in the library, so map onto real ids.
    real = {"AA": "K1", "AB": "K2", "AC": "J1", "AD": "J2", "AE": "E1"}
    real_case = "_".join(real[a] for a in alphabet)
    enc = FeatureEncoder(cond_schema="v8")
    cells, gvec = enc.encode(
        _graded_record(real_case, n_types=len(alphabet)), fl)
    assert cells.shape == (62, 19, 19)
    assert gvec.shape == (20,)
    assert np.isfinite(cells).all() and np.isfinite(gvec).all()

    gv = dict(zip(enc.globals_names, gvec))
    fracs = [gv[f"g_type_frac_{k}"] for k in range(1, 6)]
    assert all(f > 0.0 for f in fracs[:len(alphabet)]), fracs
    assert all(f == 0.0 for f in fracs[len(alphabet):]), fracs
    assert sum(fracs) == pytest.approx(1.0)
    assert gv["g_n_fresh_types"] == pytest.approx(len(alphabet) / 5.0)
    assert gv["g_e_type_std"] > 0.0


def test_v8_std_separates_a_five_type_mesh_from_its_two_type_hull(store) -> None:
    """The discriminating channel at 5 types, same argument as at 3: the mesh and
    the pair share ``e_split`` (max-min over the FED types) but not the second
    moment."""
    _, fl = store
    enc = FeatureEncoder(cond_schema="v8")
    mesh = dict(zip(enc.globals_names, enc.encode(
        _graded_record("K1_K2_J1_J2_E1", n_types=5), fl)[1]))

    enrs = {m: fl.get(m, "ga80").u_avg_enrichment
            for m in ("K1", "K2", "J1", "J2", "E1")}
    lo = min(enrs, key=lambda m: enrs[m])
    hi = max(enrs, key=lambda m: enrs[m])
    hull_case = "_".join(sorted((lo, hi)))
    hull = dict(zip(enc.globals_names, enc.encode(
        _graded_record(hull_case, n_types=2), fl)[1]))

    assert mesh["g_e_split"] == pytest.approx(hull["g_e_split"])
    assert mesh["g_e_type_std"] < hull["g_e_type_std"]
    assert mesh["g_n_fresh_types"] > hull["g_n_fresh_types"]
