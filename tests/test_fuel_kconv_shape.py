"""k-conv (k-inf vs burnup) CURVE-SHAPE feature harvest: the pure-curve
``kconv_curve_shape`` reducer, the ``.sum`` / ``.HGC`` parser fill (shared
``_curve_and_coeffs`` path), .sum-vs-HGC shape parity, the built-table wiring,
and the byte-identical additive augment.

These are the POISON-AGNOSTIC burnable-absorber signatures: the assertions are
physical-range checks against REAL DeCART products (a strong-Gd 10wt%/20pin
lattice, a weak-Gd 6wt%/12pin lattice, a ga80 surrogate HGC) rather than pinned
magic numbers, so they survive a library rebuild.  Every real-file check degrades
to ``pytest.skip`` when its source is not hydrated locally."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from lpopt.config import load_config
from lpopt.data.fuel_types import (
    FuelLibrary,
    FuelPaths,
    KCONV_SHAPE_COLUMNS,
    augment_fuel_table_kinf_shape,
    build_fuel_table,
    fuel_paths_from_config,
    kconv_curve_shape,
    parse_fa_sum,
    parse_hgc_full,
)

REPO_ROOT = Path(__file__).resolve().parents[1]                 # 5_RL
DECK = REPO_ROOT / "lpopt.inp"
APR1400 = (REPO_ROOT / ".." / "0_APR1400").resolve()
GA80_HGC = (REPO_ROOT / ".." / "3_GA_Surrogate" / "FEASIBLE_PACKAGE" / "hgc").resolve()


# --------------------------------------------------------------------------- #
# sample-file locators (skip when the source is not hydrated)
# --------------------------------------------------------------------------- #
def _find_lattice_sum(gd_dir: str) -> Path:
    """First ``FA_*.sum`` inside a lattice dir named ``gd_dir`` (e.g. ``10_20_z1``)."""
    for lib in ("5.8_5.1", "260624", "CPHA"):
        root = APR1400 / lib / "FA"
        if not root.is_dir():
            continue
        for d in sorted(root.rglob(gd_dir)):
            sums = sorted(d.glob("FA_*.sum"))
            if sums:
                return sums[0]
    pytest.skip(f"no {gd_dir} .sum hydrated under {APR1400}")


# --------------------------------------------------------------------------- #
# pure-curve reducer — synthetic curves exercise both branches deterministically
# --------------------------------------------------------------------------- #
def test_kconv_curve_shape_hump() -> None:
    # dip at BU=12 (k=1.130), Gd-burnout hump peak at BU=24 (k=1.155), then decline.
    xs = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 40.0, 50.0, 60.0]
    ys = [1.20, 1.15, 1.14, 1.135, 1.130, 1.140, 1.150, 1.155, 1.150, 1.10, 1.05, 1.00]
    d = kconv_curve_shape(xs, ys)
    assert d["kconv_is_monotone"] == 0.0
    assert d["bu_dip_gwd"] == 12.0 and d["kinf_dip"] == pytest.approx(1.130)
    assert d["bu_peak_gwd"] == 24.0 and d["kinf_peak"] == pytest.approx(1.155)
    assert d["kinf_dip"] < d["kinf_peak"]
    assert d["reactivity_swing_pcm"] > 0.0
    assert d["depletion_slope_pcm_per_gwd"] < 0.0             # burnout decay
    assert d["kinf_eol50"] == pytest.approx(1.05)             # exact grid point


def test_kconv_curve_shape_monotone() -> None:
    # strictly declining after the BU=0 xenon-free spike -> no hump.
    xs = [0.0, 0.2, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    ys = [1.26, 1.22, 1.21, 1.205, 1.19, 1.17, 1.13, 1.09, 1.04, 0.99, 0.94]
    d = kconv_curve_shape(xs, ys)
    assert d["kconv_is_monotone"] == 1.0
    assert "kinf_dip" not in d and "bu_dip_gwd" not in d      # NaN by omission
    assert "reactivity_swing_pcm" not in d
    assert d["kinf_peak"] == pytest.approx(1.26)              # degenerates to BU=0
    assert d["bu_peak_gwd"] == 0.0
    assert d["rho_boc_minus_peak_pcm"] == 0.0                 # peak IS the BOC point
    assert d["depletion_slope_pcm_per_gwd"] < 0.0            # still a valid decay
    assert d["kinf_eol50"] == pytest.approx(0.99)


def test_kconv_curve_shape_degenerate() -> None:
    assert kconv_curve_shape([0.0, 10.0], [1.1, 1.0]) == {}   # < 3 points
    assert kconv_curve_shape([], []) == {}


# --------------------------------------------------------------------------- #
# real .sum — strong-Gd 10wt%/20pin family carries the full dip->hump signature
# --------------------------------------------------------------------------- #
def test_strong_gd_10_20_shape() -> None:
    d = parse_fa_sum(_find_lattice_sum("10_20_z1"))
    assert d["kconv_is_monotone"] == 0.0
    assert d["kinf_dip"] < d["kinf_peak"]                     # dip below the hump
    assert 5.0 <= d["bu_dip_gwd"] <= 20.0
    assert 15.0 <= d["bu_peak_gwd"] <= 30.0
    assert d["reactivity_swing_pcm"] > 0.0                    # Gd release is positive
    assert d["depletion_slope_pcm_per_gwd"] < 0.0
    assert 0.85 <= d["kinf_eol50"] <= 1.05


# --------------------------------------------------------------------------- #
# real .sum — weak-Gd 6wt%/12pin family is near-monotone; handled gracefully
# --------------------------------------------------------------------------- #
def test_weak_gd_6_12_monotone_graceful() -> None:
    d = parse_fa_sum(_find_lattice_sum("6_12_z1"))
    assert d["kconv_is_monotone"] == 1.0
    assert "kinf_dip" not in d                                # no hump -> NaN dip
    assert "reactivity_swing_pcm" not in d                    # undefined without a dip
    # peak degenerates to BU=0 == kinf0, and the burnout slope is still valid.
    assert d["bu_peak_gwd"] == 0.0
    assert d["kinf_peak"] == pytest.approx(d["kinf0"])
    assert d["depletion_slope_pcm_per_gwd"] < 0.0
    assert 0.85 <= d["kinf_eol50"] <= 1.05


# --------------------------------------------------------------------------- #
# ga80 surrogate HGC (HGC-only library) — full curve harvested from %TITL states
# --------------------------------------------------------------------------- #
def test_ga80_hgc_shape() -> None:
    hgc = GA80_HGC / "FA_A2.HGC"                              # strong-Gd surrogate
    if not hgc.exists():
        pytest.skip(f"ga80 HGC not hydrated: {hgc}")
    d = parse_hgc_full(hgc)
    assert "kinf_peak" in d and "kinf_eol50" in d
    assert d["depletion_slope_pcm_per_gwd"] < 0.0
    assert 0.85 <= d["kinf_eol50"] <= 1.05
    if d["kconv_is_monotone"] == 0.0:
        assert d["kinf_dip"] < d["kinf_peak"]
        assert d["reactivity_swing_pcm"] > 0.0


# --------------------------------------------------------------------------- #
# .sum vs .HGC shape parity — a lattice type carrying both products agrees
# --------------------------------------------------------------------------- #
def test_sum_hgc_shape_parity() -> None:
    sum_path = _find_lattice_sum("10_20_z1")                  # has a Gd hump
    stem = sum_path.name[: -len(".sum")]
    hgc = sum_path.with_suffix(".HGC")
    if not hgc.is_file():
        variants = sorted(sum_path.parent.glob(f"{stem}_*.HGC"))
        if not variants:
            pytest.skip(f"no sibling HGC beside {sum_path}")
        hgc = variants[0]
    s = parse_fa_sum(sum_path)
    h = parse_hgc_full(hgc)
    assert s["kconv_is_monotone"] == h["kconv_is_monotone"]
    assert s["bu_dip_gwd"] == h["bu_dip_gwd"]                 # same curve -> same node
    assert s["bu_peak_gwd"] == h["bu_peak_gwd"]
    assert abs(s["kinf_dip"] - h["kinf_dip"]) < 1.0e-4
    assert abs(s["kinf_peak"] - h["kinf_peak"]) < 1.0e-4
    assert abs(s["reactivity_swing_pcm"] - h["reactivity_swing_pcm"]) < 5.0
    assert abs(s["depletion_slope_pcm_per_gwd"]
               - h["depletion_slope_pcm_per_gwd"]) < 1.0


# --------------------------------------------------------------------------- #
# built table wiring — columns present, lattice filled, legacy NaN
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built() -> pd.DataFrame:
    if not DECK.is_file():
        pytest.skip("lpopt.inp not present")
    return build_fuel_table(load_config(DECK), persist=False)


def test_build_adds_kconv_shape_columns(built) -> None:
    for c in KCONV_SHAPE_COLUMNS:
        assert c in built.columns
    lat = built[built["library_id"] == "260624"]
    assert len(lat) > 0
    # every lattice type has a reference curve -> a peak + a burnout slope + eol50.
    assert lat["kinf_peak"].notna().all()
    assert lat["depletion_slope_pcm_per_gwd"].notna().all()
    assert lat["kinf_eol50"].notna().all()
    assert (lat["depletion_slope_pcm_per_gwd"] < 0.0).all()   # always a decay


def test_kconv_monotone_flag_matches_dip_nan(built) -> None:
    """Wherever a curve is harvested: hump <-> finite dip, monotone <-> NaN dip."""
    harv = built[built["kinf_peak"].notna()]
    assert len(harv) > 0
    hump = harv[harv["kconv_is_monotone"] == 0.0]
    mono = harv[harv["kconv_is_monotone"] == 1.0]
    assert hump["kinf_dip"].notna().all()
    assert hump["reactivity_swing_pcm"].notna().all()
    assert (hump["reactivity_swing_pcm"] > 0.0).all()
    assert mono["kinf_dip"].isna().all()


def test_legacy_kconv_nan(built) -> None:
    sub = built[built["library_id"] == "legacy_a"]
    if len(sub):
        for c in KCONV_SHAPE_COLUMNS:
            assert sub[c].isna().all()                        # no source curve


def test_ga80_hgc_type_kconv_filled() -> None:
    lib = FuelLibrary.build(load_config(DECK), persist=False)
    a2 = lib.get("A2", "ga80")                                # A2 has a hydrated HGC
    if a2.kinf_peak is None or math.isnan(a2.kinf_peak):
        pytest.skip("ga80 A2 HGC not hydrated in this checkout")
    assert a2.kinf_eol50 is not None and 0.85 <= a2.kinf_eol50 <= 1.05
    assert a2.depletion_slope_pcm_per_gwd is not None
    assert a2.depletion_slope_pcm_per_gwd < 0.0


# --------------------------------------------------------------------------- #
# additive augment: existing values byte-identical, only shape columns appended
# --------------------------------------------------------------------------- #
def _norm(x):
    if isinstance(x, float) and math.isnan(x):
        return "__nan__"
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return list(x)
    return x


def test_augment_kconv_byte_identical(tmp_path, built) -> None:
    cfg = load_config(DECK)
    paths = fuel_paths_from_config(cfg)
    pre = built.drop(columns=list(KCONV_SHAPE_COLUMNS))
    store = tmp_path / "fuel_types.parquet"
    pre.to_parquet(store, index=False)
    existing_disk = pd.read_parquet(store)

    sp = FuelPaths(apr1400_root=paths.apr1400_root, ga80_hgc=paths.ga80_hgc,
                   manual_yaml=paths.manual_yaml, store=store,
                   paramA_root=paths.paramA_root)
    merged = augment_fuel_table_kinf_shape(sp, persist=True)

    assert len(merged) == len(existing_disk)
    for c in KCONV_SHAPE_COLUMNS:
        assert c in merged.columns
    for c in existing_disk.columns:
        a = [_norm(x) for x in existing_disk[c].tolist()]
        b = [_norm(x) for x in merged[c].tolist()]
        assert a == b, c


def test_augment_kconv_idempotent(tmp_path, built) -> None:
    cfg = load_config(DECK)
    paths = fuel_paths_from_config(cfg)
    store = tmp_path / "fuel_types.parquet"
    built.drop(columns=list(KCONV_SHAPE_COLUMNS)).to_parquet(store, index=False)
    sp = FuelPaths(apr1400_root=paths.apr1400_root, ga80_hgc=paths.ga80_hgc,
                   manual_yaml=paths.manual_yaml, store=store,
                   paramA_root=paths.paramA_root)
    m1 = augment_fuel_table_kinf_shape(sp, persist=True)
    m2 = augment_fuel_table_kinf_shape(sp, persist=True)
    assert m1.shape == m2.shape
    assert (m1["reactivity_swing_pcm"].fillna(-1).tolist()
            == m2["reactivity_swing_pcm"].fillna(-1).tolist())
    assert (m1["kconv_is_monotone"].fillna(-1).tolist()
            == m2["kconv_is_monotone"].fillna(-1).tolist())
