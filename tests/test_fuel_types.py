"""Physics fuel-feature table (plan 4.3): MASS parser, HGC Gd heuristic,
(library_id, type_id) collisions, manual anchors, FuelLibrary accessors, and
pair helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from lpopt.config import load_config
from lpopt.data.fuel_types import (
    FuelLibrary,
    FuelPaths,
    FuelVec,
    GA80_TYPE_IDS,
    build_fuel_table,
    core_enrichment_split,
    count_gd_pins_from_hgc,
    fuel_paths_from_config,
    load_manual_anchors,
    pair_e_core,
    parse_fa_mass_out,
)

REPO_ROOT = Path(__file__).resolve().parents[1]          # 5_RL
DECK = REPO_ROOT / "lpopt.inp"
APR1400 = (REPO_ROOT / ".." / "0_APR1400").resolve()
MOCHA_CONFIG = (REPO_ROOT / ".." / "2_LP" / "MOCHA" / "config.py").resolve()
FA_B1_HGC = APR1400 / "260624" / "hgc" / "FA_B1.HGC"


def _find_fa_out(type_id: str, subtree: str) -> Path:
    root = APR1400 / subtree
    hits = list(root.rglob(f"FA_{type_id}.out"))
    assert hits, f"FA_{type_id}.out not found under {root}"
    return hits[0]


@pytest.fixture(scope="module")
def lib() -> FuelLibrary:
    """Build the whole table once (no persistence side effect)."""
    cfg = load_config(DECK)
    return FuelLibrary.build(cfg, persist=False)


# --------------------------------------------------------------------------- #
# source 1: FA_*.out MASS(g) parser vs the reference MOCHA parser
# --------------------------------------------------------------------------- #
def test_mass_parser_b1_range() -> None:
    fa = _find_fa_out("B1", "260624/FA")
    got = parse_fa_mass_out(fa)
    assert 5.3 <= got["u_avg_enrichment"] <= 5.6
    assert got["u_mass_g"] > 0.0


def test_mass_parser_matches_mocha() -> None:
    """Our ported parser reproduces MOCHA's parse_decart_fresh_inventory."""
    if not MOCHA_CONFIG.exists():
        pytest.skip(f"MOCHA config not present: {MOCHA_CONFIG}")
    spec = importlib.util.spec_from_file_location("mocha_config_ref", str(MOCHA_CONFIG))
    mc = importlib.util.module_from_spec(spec)
    sys.modules["mocha_config_ref"] = mc          # dataclass __module__ resolution
    try:
        spec.loader.exec_module(mc)
    except Exception as exc:                        # pragma: no cover - env dependent
        pytest.skip(f"could not import MOCHA config: {exc}")

    fa = _find_fa_out("B1", "260624/FA")
    ref = mc.parse_decart_fresh_inventory(fa)
    got = parse_fa_mass_out(fa)
    assert got["u235_g"] == pytest.approx(ref["u235_g"])
    assert got["u238_g"] == pytest.approx(ref["u238_g"])
    assert got["u_avg_enrichment"] == pytest.approx(ref["enrichment"])


def test_mass_parser_feeds_table(lib: FuelLibrary) -> None:
    b1 = lib.get("B1", "260624")
    assert b1.u_avg_enrichment == pytest.approx(parse_fa_mass_out(
        _find_fa_out("B1", "260624/FA"))["u_avg_enrichment"])
    assert "fa_mass_out" in b1.source_flags
    assert b1.enr_main == pytest.approx(5.8)
    assert b1.enr_zone == pytest.approx(5.1)
    assert b1.gd_u_enr == pytest.approx(4.0)
    assert b1.n_gd == 20                            # IGD_20
    assert b1.axial_zone == "z1"
    assert b1.feature_poor is False


# --------------------------------------------------------------------------- #
# source 3: HGC %DIST Gd-pin heuristic (plan M1 acceptance)
# --------------------------------------------------------------------------- #
def test_hgc_gd_count_b1_is_20() -> None:
    if not FA_B1_HGC.exists():
        pytest.skip(f"hydrated HGC not present: {FA_B1_HGC}")
    assert count_gd_pins_from_hgc(FA_B1_HGC) == 20


# --------------------------------------------------------------------------- #
# (library_id, type_id) collision — the whole reason library_id is mandatory
# --------------------------------------------------------------------------- #
def test_library_type_collision(lib: FuelLibrary) -> None:
    b1_260624 = lib.get("B1", "260624")
    b1_ga80 = lib.get("B1", "ga80")
    # same type id, different libraries -> different physical designs
    assert b1_260624.type_id == b1_ga80.type_id == "B1"
    assert b1_260624.library_id != b1_ga80.library_id
    assert b1_260624.u_avg_enrichment != b1_ga80.u_avg_enrichment
    # 260624 B1 is fully featured; ga80 B1 is a feature-poor manual anchor
    assert b1_260624.feature_poor is False
    assert b1_ga80.feature_poor is True
    assert b1_ga80.n_gd is None


# --------------------------------------------------------------------------- #
# source 5: manual anchors — yaml loads, F/M excluded, all 80 ids accounted
# --------------------------------------------------------------------------- #
def test_manual_yaml_loads_and_excludes_fm() -> None:
    manual = REPO_ROOT / "config" / "fuel_types_manual.yaml"
    anchors, excluded = load_manual_anchors(manual)
    assert anchors["A"] == 5.0 and anchors["E"] == 5.0
    assert anchors["J"] == 5.1 and anchors["K"] == 5.2
    assert anchors["L"] == 5.3 and anchors["N"] == 5.4
    assert anchors["B"] == 5.5 and anchors["G"] == 5.5 and anchors["H"] == 5.5
    assert anchors["C"] == 6.0 and anchors["D"] == 6.5
    assert "F" in excluded and "M" in excluded
    assert "F" not in anchors and "M" not in anchors


def test_all_80_ga80_ids_accounted(lib: FuelLibrary) -> None:
    assert len(GA80_TYPE_IDS) == 80
    ga80_rows = set(lib.types("ga80"))
    excluded_ids = {t for t in GA80_TYPE_IDS if t[0] in ("F", "M")}
    assert len(excluded_ids) == 10                  # F1-F6 + M1-M4
    # no excluded type ever gets a row
    assert not any(t[0] in ("F", "M") for t in ga80_rows)
    # parsed(HGC, currently 0) + manual(70) + excluded(10) == full roster
    assert ga80_rows | excluded_ids == set(GA80_TYPE_IDS)
    assert len(ga80_rows) == 70


def test_ga80_anchor_values(lib: FuelLibrary) -> None:
    assert lib.get("K3", "ga80").u_avg_enrichment == pytest.approx(5.2)
    assert lib.get("D8", "ga80").u_avg_enrichment == pytest.approx(6.5)
    assert lib.library_enrichment_range("ga80") == (5.0, 6.5)


# --------------------------------------------------------------------------- #
# 5.8_5.1 A0x <-> X byte-identical alias rows (both emitted, cross-referenced)
# --------------------------------------------------------------------------- #
def test_5851_alias_rows(lib: FuelLibrary) -> None:
    a01 = lib.get("A01", "5.8_5.1")
    x2 = lib.get("X2", "5.8_5.1")
    assert a01.u_avg_enrichment == pytest.approx(x2.u_avg_enrichment)
    assert a01.u_mass_g == pytest.approx(x2.u_mass_g)
    assert "alias:X2" in a01.source_flags
    assert "alias_of:A01" in x2.source_flags


# --------------------------------------------------------------------------- #
# FuelLibrary.get — hard KeyError with a helpful message
# --------------------------------------------------------------------------- #
def test_get_unknown_raises_keyerror(lib: FuelLibrary) -> None:
    with pytest.raises(KeyError) as exc:
        lib.get("ZZ9", "nope_library")
    msg = str(exc.value)
    assert "ZZ9" in msg and "nope_library" in msg


def test_get_unknown_hints_other_library(lib: FuelLibrary) -> None:
    # B1 exists (in 260624 / ga80) but not in CPHA -> message names the holders
    with pytest.raises(KeyError) as exc:
        lib.get("B1", "CPHA")
    assert "260624" in str(exc.value) or "ga80" in str(exc.value)


# --------------------------------------------------------------------------- #
# pair_e_core — mass-weighted split
# --------------------------------------------------------------------------- #
def test_pair_e_core_mass_weighted() -> None:
    va = FuelVec("t", "A", u_avg_enrichment=5.0, u_mass_g=100.0)
    vb = FuelVec("t", "B", u_avg_enrichment=6.0, u_mass_g=200.0)
    # (0.5*100*5 + 0.5*200*6) / (0.5*100 + 0.5*200) = 850/150
    assert pair_e_core(va, vb, 0.5) == pytest.approx(850.0 / 150.0)
    # endpoints
    assert pair_e_core(va, vb, 1.0) == pytest.approx(5.0)
    assert pair_e_core(va, vb, 0.0) == pytest.approx(6.0)
    # heavier B mass pulls the mass-weighted mean above the count-weighted 5.5
    assert pair_e_core(va, vb, 0.5) > 5.5


def test_pair_e_core_count_fallback_when_mass_unknown() -> None:
    va = FuelVec("t", "A", u_avg_enrichment=5.0)      # u_mass_g None
    vb = FuelVec("t", "B", u_avg_enrichment=6.0)
    assert pair_e_core(va, vb, 0.5) == pytest.approx(5.5)


def test_pair_e_core_via_library(lib: FuelLibrary) -> None:
    got = lib.pair_e_core("B1", "C1", 0.5, library_id="260624")
    b1 = lib.get("B1", "260624")
    c1 = lib.get("C1", "260624")
    wa, wb = 0.5 * b1.u_mass_g, 0.5 * c1.u_mass_g
    expected = (wa * b1.u_avg_enrichment + wb * c1.u_avg_enrichment) / (wa + wb)
    assert got == pytest.approx(expected)


def test_pair_e_core_bad_split() -> None:
    va = FuelVec("t", "A", u_avg_enrichment=5.0, u_mass_g=100.0)
    vb = FuelVec("t", "B", u_avg_enrichment=6.0, u_mass_g=200.0)
    with pytest.raises(ValueError):
        pair_e_core(va, vb, 1.5)


# --------------------------------------------------------------------------- #
# build/persist + schema + legacy
# --------------------------------------------------------------------------- #
def test_build_persists_parquet_and_roundtrips(tmp_path) -> None:
    cfg = load_config(DECK)
    base = fuel_paths_from_config(cfg)
    store = tmp_path / "fuel_types.parquet"
    paths = FuelPaths(base.apr1400_root, base.ga80_hgc, base.manual_yaml, store)
    df = build_fuel_table(paths, persist=True)
    assert store.exists()
    reloaded = FuelLibrary.from_parquet(store)
    assert set(reloaded.libraries()) == set(df["library_id"].unique())
    # list-column + Int64 survive the round trip
    b1 = reloaded.get("B1", "260624")
    assert b1.n_gd == 20
    assert "fa_mass_out" in b1.source_flags


def test_legacy_rows(lib: FuelLibrary) -> None:
    a0 = lib.get("A0", "legacy_a")
    a1 = lib.get("A1", "legacy_a")
    assert a0.u_avg_enrichment == pytest.approx(5.6345)
    assert a1.u_avg_enrichment == pytest.approx(5.6375)
    assert a0.n_gd == 16 and a1.n_gd == 12
    assert a0.source_flags == ["mocha_hardcoded"]


def test_kinf_columns_nan_when_absent_finite_when_harvested(lib: FuelLibrary) -> None:
    # cond_v4: a source-backed lattice type carries a finite, in-range k-inf
    # curve; a type with no .sum/.HGC source keeps NaN.
    b1 = lib.get("B1", "260624")                      # .sum-harvested
    assert b1.kinf0 is not None and 1.10 <= b1.kinf0 <= 1.30
    assert b1.kinf30 is not None and b1.kinf30 < b1.kinf20   # post-hump decline
    assert b1.bu_k1 is not None and 10.0 <= b1.bu_k1 <= 60.0
    assert b1.ff_pin_max is not None and 1.05 <= b1.ff_pin_max <= 1.25

    a0 = lib.get("A0", "legacy_a")                    # no source -> NaN
    for v in (a0.kinf0, a0.kinf10, a0.kinf20, a0.kinf30, a0.bu_k1):
        assert v is None or math.isnan(v)

    b1_ga80 = lib.get("B1", "ga80")                   # ga80 B has no HGC -> NaN
    for v in (b1_ga80.kinf0, b1_ga80.bu_k1, b1_ga80.xs_a2):
        assert v is None or math.isnan(v)


def test_config_fuel_defaults() -> None:
    cfg = load_config(DECK)
    assert cfg.fuel.apr1400_root == "../0_APR1400"
    assert cfg.fuel.store == "data/store/fuel_types.parquet"


# --------------------------------------------------------------------------- #
# core_enrichment_split — THE derived-column formula (regression 20260829)
#
# ``e_core`` is a pure function of the REALIZED feed, never of the case's planned
# split.  These pin the recipe on a synthetic library so the contract survives any
# future fuel-table refresh:
#
#   * U-mass weighted when EVERY fed type has a known ``u_mass_g``;
#   * count weighted when ANY fed type is missing one (all-or-nothing);
#   * ``(None, None)`` — never a partial answer — when a fed type is unresolvable
#     or carries no enrichment, so extraction and inference fall back identically;
#   * and it is NOT ``pair_e_core(a, b, 0.5)``.  The produce/campaign write path
#     stamped that NOMINAL equal-split value into ``records.e_core`` (paired with a
#     null ``e_split``), which is how 979 store rows came to advertise a core that
#     was never loaded — up to 0.068 w/o away from their own pattern, ~1.4 curriculum
#     e_core bins.
# --------------------------------------------------------------------------- #
_SYNTH_COLUMNS = dict(n_gd=None, source_flags=None, axial_zone=None,
                      feature_poor=False)


def _synth_library(*types: tuple[str, float | None, float | None]) -> FuelLibrary:
    """A one-library FuelLibrary from ``(type_id, enrichment, u_mass_g)`` triples."""
    rows = [dict(library_id="synth", type_id=tid, u_avg_enrichment=enr,
                 u_mass_g=mass, **_SYNTH_COLUMNS) for tid, enr, mass in types]
    return FuelLibrary(pd.DataFrame(rows))


def test_core_enrichment_split_is_u_mass_weighted_on_the_realized_feed() -> None:
    lib = _synth_library(("A", 5.0, 100.0), ("B", 6.0, 200.0))
    e_core, e_split = core_enrichment_split(lib, "synth", {"A": 69, "B": 52})
    # (69*100*5 + 52*200*6) / (69*100 + 52*200) = 96900 / 17300
    assert e_core == pytest.approx(96900.0 / 17300.0)
    assert e_split == pytest.approx(1.0)                    # max(e) - min(e)
    # ... and NOT the count-weighted mean, nor the nominal 50/50 pair value.
    assert abs(e_core - 657.0 / 121.0) > 1e-3               # count weighted
    assert abs(e_core - pair_e_core(lib.get("A", "synth"),
                                    lib.get("B", "synth"), 0.5)) > 1e-3


def test_core_enrichment_split_tracks_the_split_not_the_nominal() -> None:
    """Two feeds of the SAME case give two different e_core — exactly the property
    the nominal write path destroyed by stamping one constant across a campaign."""
    lib = _synth_library(("A", 5.0, 100.0), ("B", 6.0, 200.0))
    a, _ = core_enrichment_split(lib, "synth", {"A": 69, "B": 52})
    b, _ = core_enrichment_split(lib, "synth", {"A": 53, "B": 68})
    assert abs(a - b) > 1e-6
    assert b > a                                            # more of the rich type
    # the endpoints are the pure single-type values, with a zero spread
    only_a, spread = core_enrichment_split(lib, "synth", {"A": 121})
    assert only_a == pytest.approx(5.0) and spread == pytest.approx(0.0)


def test_core_enrichment_split_count_weights_when_any_mass_unknown() -> None:
    """All-or-nothing: ONE missing u_mass_g demotes the whole feed to count
    weighting (the ga80 letter library, whose masses are all NaN)."""
    lib = _synth_library(("A", 5.0, 100.0), ("B", 6.0, None))
    e_core, e_split = core_enrichment_split(lib, "synth", {"A": 69, "B": 52})
    assert e_core == pytest.approx((69 * 5.0 + 52 * 6.0) / 121.0)
    assert e_split == pytest.approx(1.0)


def test_core_enrichment_split_three_types() -> None:
    lib = _synth_library(("A", 5.0, 100.0), ("B", 6.0, 200.0), ("C", 5.5, 150.0))
    e_core, e_split = core_enrichment_split(lib, "synth", {"A": 65, "B": 56, "C": 4})
    num = 65 * 100 * 5.0 + 56 * 200 * 6.0 + 4 * 150 * 5.5
    den = 65 * 100 + 56 * 200 + 4 * 150
    assert e_core == pytest.approx(num / den)
    assert e_split == pytest.approx(1.0)                    # 6.0 - 5.0, not a stdev


def test_core_enrichment_split_returns_a_none_pair_not_a_partial() -> None:
    lib = _synth_library(("A", 5.0, 100.0), ("B", 6.0, 200.0), ("N", None, 100.0))
    assert core_enrichment_split(lib, "synth", {"A": 69, "Z": 52}) == (None, None)
    assert core_enrichment_split(lib, "other", {"A": 121}) == (None, None)
    assert core_enrichment_split(lib, "synth", {"A": 69, "N": 52}) == (None, None)
    assert core_enrichment_split(lib, "synth", {}) == (None, None)


def test_core_enrichment_split_zero_padding_tolerance() -> None:
    """``resolve_type_id`` normalization is part of the recipe: a ``C01`` batch
    name in a pattern still finds the roster's ``C1``."""
    lib = _synth_library(("C1", 5.0, 100.0))
    assert core_enrichment_split(lib, "synth", {"C01": 121})[0] == pytest.approx(5.0)
