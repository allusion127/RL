"""Pin-cell geometry harvest (review sec. 4a): dec inp GEOM parser, additive
columns wired into the build, and the byte-identical additive augment."""

from __future__ import annotations

from pathlib import Path

import math
import pandas as pd
import pytest

from lpopt.config import load_config
from lpopt.data.fuel_types import (
    FuelPaths,
    GEOM_COLUMNS,
    NOMINAL_ASM_PITCH,
    augment_fuel_table_geometry,
    build_fuel_table,
    fuel_paths_from_config,
    geom_derived,
    parse_dec_geom,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DECK = REPO_ROOT / "lpopt.inp"
SAMPLE_DEC = (REPO_ROOT / "data" / "design" / "curriculum_work" / "5.75-6_f109"
              / "P7" / "dec_FA_P7.inp")


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def test_parse_dec_geom_real_deck() -> None:
    if not SAMPLE_DEC.is_file():
        pytest.skip(f"sample dec deck not present: {SAMPLE_DEC}")
    g = parse_dec_geom(SAMPLE_DEC)
    assert g["pin_pitch"] == pytest.approx(1.285)
    assert g["asm_pitch"] == pytest.approx(NOMINAL_ASM_PITCH)      # 20.7772
    assert g["r_pellet"] == pytest.approx(0.4096)
    assert g["r_clad_in"] == pytest.approx(0.4178)
    assert g["r_clad_out"] == pytest.approx(0.4750)
    assert g["p_over_d"] == pytest.approx(1.285 / (2 * 0.4750), rel=1e-6)
    assert g["v_mod_over_v_fuel"] == pytest.approx(1.788, abs=1e-2)


def test_geom_derived_nominal() -> None:
    d = geom_derived(1.285, 0.4096, 0.4750)
    assert d["p_over_d"] == pytest.approx(1.35263, abs=1e-4)
    assert d["v_mod_over_v_fuel"] == pytest.approx(1.788, abs=1e-2)
    # degenerate inputs -> empty
    assert geom_derived(0.0, 0.4, 0.4) == {}


def test_parse_dec_geom_missing_returns_empty(tmp_path) -> None:
    # a MATERIAL-only deck (no GEOM card) parses to {} without raising.
    p = tmp_path / "no_geom.inp"
    p.write_text("MATERIAL\n mixture UO2 2 10 626 / 92235 5.8\n", encoding="utf-8")
    assert parse_dec_geom(p) == {}


# --------------------------------------------------------------------------- #
# build wiring
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built() -> pd.DataFrame:
    if not DECK.is_file():
        pytest.skip("lpopt.inp not present")
    cfg = load_config(DECK)
    return build_fuel_table(cfg, persist=False)


def test_build_adds_geometry_columns(built) -> None:
    for c in GEOM_COLUMNS:
        assert c in built.columns
    # lattice libraries (dec inp present) carry the constant geometry.
    import numpy as np
    lat = built[built["library_id"] == "260624"]
    assert len(lat) > 0
    assert lat["pin_pitch"].notna().all()
    assert np.allclose(lat["pin_pitch"], 1.285)
    assert np.allclose(lat["asm_pitch"], NOMINAL_ASM_PITCH)
    assert np.allclose(lat["r_pellet"], 0.4096)


def test_ga80_legacy_geometry_nan(built) -> None:
    # ga80 / legacy ship no dec inp -> NaN geometry (same contract as zone_pin_count).
    for lib in ("ga80", "legacy_a"):
        sub = built[built["library_id"] == lib]
        if len(sub):
            assert sub["pin_pitch"].isna().all()
            assert sub["asm_pitch"].isna().all()


# --------------------------------------------------------------------------- #
# additive augment: existing values byte-identical, only geometry appended
# --------------------------------------------------------------------------- #
def test_augment_geometry_byte_identical(tmp_path, built) -> None:
    cfg = load_config(DECK)
    paths = fuel_paths_from_config(cfg)
    # simulate a PRE-geometry store: drop the geometry columns and persist.
    pre = built.drop(columns=list(GEOM_COLUMNS))
    store = tmp_path / "fuel_types.parquet"
    pre.to_parquet(store, index=False)

    # the on-disk pre-geometry table (the exact bytes a live reader would see).
    existing_disk = pd.read_parquet(store)

    sp = FuelPaths(apr1400_root=paths.apr1400_root, ga80_hgc=paths.ga80_hgc,
                   manual_yaml=paths.manual_yaml, store=store,
                   paramA_root=paths.paramA_root)
    merged = augment_fuel_table_geometry(sp, persist=True)

    # row count preserved, geometry columns appended.
    assert len(merged) == len(existing_disk)
    for c in GEOM_COLUMNS:
        assert c in merged.columns

    # every pre-existing on-disk column is byte-identical (source_flags is a list
    # column -> normalize list-likes; NaN==NaN via a sentinel).
    def _norm(x):
        if isinstance(x, float) and math.isnan(x):
            return "__nan__"
        if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
            return list(x)
        return x

    for c in existing_disk.columns:
        a = [_norm(x) for x in existing_disk[c].tolist()]
        b = [_norm(x) for x in merged[c].tolist()]
        assert a == b, c


def test_augment_geometry_idempotent(tmp_path, built) -> None:
    cfg = load_config(DECK)
    paths = fuel_paths_from_config(cfg)
    store = tmp_path / "fuel_types.parquet"
    built.drop(columns=list(GEOM_COLUMNS)).to_parquet(store, index=False)
    sp = FuelPaths(apr1400_root=paths.apr1400_root, ga80_hgc=paths.ga80_hgc,
                   manual_yaml=paths.manual_yaml, store=store,
                   paramA_root=paths.paramA_root)
    m1 = augment_fuel_table_geometry(sp, persist=True)
    m2 = augment_fuel_table_geometry(sp, persist=True)
    assert m1.shape == m2.shape
    assert m1["pin_pitch"].fillna(-1).tolist() == m2["pin_pitch"].fillna(-1).tolist()
