"""cond_v4 physics harvest (plan 4.3 feature expansion): the ``.sum`` / ``.HGC``
parsers, the ``dec_FA_*.inp`` zoning census, .sum-vs-HGC cross-parity, and the
built-table fill for the ga80 HGC-only library.

The assertions are physical-range checks against REAL DeCART products (a 5.8_5.1
X-series lattice, a 260624 lattice, and a ga80 surrogate HGC) rather than pinned
magic numbers, so they survive a library rebuild.  Every check degrades to a
``pytest.skip`` when its source file is not hydrated locally."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lpopt.config import load_config
from lpopt.data.fuel_types import (
    FuelLibrary,
    parse_dec_zone_census,
    parse_fa_sum,
    parse_hgc_boc_xs_adf,
    parse_hgc_full,
)

REPO_ROOT = Path(__file__).resolve().parents[1]                 # 5_RL
DECK = REPO_ROOT / "lpopt.inp"
APR1400 = (REPO_ROOT / ".." / "0_APR1400").resolve()
GA80_HGC = (REPO_ROOT / ".." / "3_GA_Surrogate" / "FEASIBLE_PACKAGE" / "hgc").resolve()


# --------------------------------------------------------------------------- #
# sample-file locators (skip when the source is not hydrated)
# --------------------------------------------------------------------------- #
def _find(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        pytest.skip(f"no {pattern} under {root}")
    return hits[0]


def _dir_n_gd(inp_dir_name: str) -> int | None:
    # lattice dir name is "{gd_wt}_{n_gd}_z{1,2}"
    parts = inp_dir_name.split("_")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return None


def _post_peak_monotone(vals: list[float]) -> bool:
    """k-inf is non-increasing after the (Gd-burnout) hump peak."""
    peak = vals.index(max(vals))
    return all(vals[i] >= vals[i + 1] for i in range(peak, len(vals) - 1))


# --------------------------------------------------------------------------- #
# .sum parser — 5.8_5.1 X8 and 260624 B1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("type_id,subtree", [("X8", "5.8_5.1/FA"), ("B1", "260624/FA")])
def test_parse_fa_sum_ranges(type_id: str, subtree: str) -> None:
    sum_path = _find(APR1400 / subtree, f"FA_{type_id}.sum")
    d = parse_fa_sum(sum_path)

    assert 1.10 <= d["kinf0"] <= 1.30
    curve = [d["kinf0"], d["kinf10"], d["kinf20"], d["kinf30"]]
    assert _post_peak_monotone(curve)
    assert d["kinf30"] < d["kinf20"]                       # post-hump decline
    assert 10.0 <= d["bu_k1"] <= 60.0
    assert 1.05 <= d["ff_pin_max"] <= 1.25
    assert -15.0 <= d["boron_worth"] <= -3.0
    assert -6.0 <= d["doppler_coef"] <= -0.5
    assert 5000.0 <= d["cr1_worth"] <= 60000.0
    assert d["mtc_dmod"] > 0.0                              # denser moderator -> +rho


# --------------------------------------------------------------------------- #
# HGC parser — ga80 surrogate (HGC-only library)
# --------------------------------------------------------------------------- #
def test_parse_hgc_full_ga80_ranges() -> None:
    hgc = GA80_HGC / "FA_E3.HGC"
    if not hgc.exists():
        pytest.skip(f"ga80 HGC not hydrated: {hgc}")
    d = parse_hgc_full(hgc)

    assert 1.10 <= d["kinf0"] <= 1.30
    assert _post_peak_monotone([d["kinf0"], d["kinf10"], d["kinf20"], d["kinf30"]])
    assert 10.0 <= d["bu_k1"] <= 60.0
    assert 1.05 <= d["ff_pin_max"] <= 1.25
    assert -15.0 <= d["boron_worth"] <= -3.0
    assert -6.0 <= d["doppler_coef"] <= -0.5
    assert 5000.0 <= d["cr1_worth"] <= 60000.0
    # BOC 2-group macro cross sections + ADF (HGC-only columns).
    assert 0.10 <= d["xs_a2"] <= 0.13
    assert 0.10 <= d["xs_nf2"] <= 0.20
    assert 0.0 < d["xs_s12"] < 0.05
    assert 1.0 <= d["adf_corner_g2"] <= 1.35
    assert d["xs_d1"] > d["xs_d2"] > 0.0                    # fast diffusion > thermal


# --------------------------------------------------------------------------- #
# dec_FA_*.inp zoning census reproduces the directory n_gd on every lattice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("type_id,subtree", [("X8", "5.8_5.1/FA"), ("B1", "260624/FA")])
def test_zone_census_matches_dir_n_gd(type_id: str, subtree: str) -> None:
    sum_path = _find(APR1400 / subtree, f"FA_{type_id}.sum")
    dec = sorted(sum_path.parent.glob("dec_FA_*.inp"))
    if not dec:
        pytest.skip(f"no dec inp beside {sum_path}")
    census = parse_dec_zone_census(dec[0])
    assert census, "census returned empty"
    assert census["n_gd_census"] == _dir_n_gd(sum_path.parent.name)
    # APR1400 16x16 axial-zone-1 layout carries 52 zoning (UO2_2) pins.
    assert census["zone_pin_count"] > 0


def test_zone_census_all_lattice_dirs() -> None:
    """Every hydrated lattice dir: octant census n_gd == directory n_gd."""
    checked = 0
    for lib in ("5.8_5.1", "260624", "CPHA"):
        root = APR1400 / lib / "FA"
        for dec in root.rglob("dec_FA_*.inp"):
            dir_n_gd = _dir_n_gd(dec.parent.name)
            if dir_n_gd is None:
                continue
            census = parse_dec_zone_census(dec)
            if not census:
                continue
            assert census["n_gd_census"] == dir_n_gd, f"{dec}: {census} vs {dir_n_gd}"
            checked += 1
    if checked == 0:
        pytest.skip("no hydrated lattice dec inp files")
    assert checked >= 12


# --------------------------------------------------------------------------- #
# .sum vs .HGC cross-parity — a lattice type carries both products
# --------------------------------------------------------------------------- #
def test_sum_hgc_cross_parity() -> None:
    sum_path = _find(APR1400 / "5.8_5.1/FA", "FA_X8.sum")
    hgc_path = sum_path.with_suffix(".HGC")
    if not hgc_path.exists():
        pytest.skip(f"no sibling HGC: {hgc_path}")
    s = parse_fa_sum(sum_path)
    h = parse_hgc_full(hgc_path)
    # identical record shape -> the reference k-inf and pin peaking agree.
    assert abs(s["kinf0"] - h["kinf0"]) < 1.0e-4
    assert abs(s["ff_pin_max"] - h["ff_pin_max"]) < 0.01


def test_hgc_boc_and_full_agree_on_xs() -> None:
    """The light BOC reader and the full parser return the same xs/adf/ff."""
    sum_path = _find(APR1400 / "5.8_5.1/FA", "FA_X8.sum")
    hgc_path = sum_path.with_suffix(".HGC")
    if not hgc_path.exists():
        pytest.skip(f"no sibling HGC: {hgc_path}")
    light = parse_hgc_boc_xs_adf(hgc_path)
    full = parse_hgc_full(hgc_path)
    for key in ("xs_d1", "xs_a2", "xs_nf2", "xs_s12", "adf_corner_g2", "ff_pin_max"):
        assert light[key] == pytest.approx(full[key])


# --------------------------------------------------------------------------- #
# built table: ga80 HGC-only library gets kinf + branch + xs + adf + ff filled
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def lib() -> FuelLibrary:
    return FuelLibrary.build(load_config(DECK), persist=False)


def test_ga80_hgc_types_filled(lib: FuelLibrary) -> None:
    e3 = lib.get("E3", "ga80")                             # E3 has a hydrated HGC
    if e3.kinf0 is None or math.isnan(e3.kinf0):
        pytest.skip("ga80 E3 HGC not hydrated in this checkout")
    assert 1.10 <= e3.kinf0 <= 1.30
    assert e3.xs_a2 is not None and 0.10 <= e3.xs_a2 <= 0.13
    assert e3.adf_corner_g2 is not None and e3.adf_corner_g2 > 1.0
    assert e3.cr1_worth is not None and e3.cr1_worth > 5000.0
    assert e3.ff_pin_max is not None and 1.05 <= e3.ff_pin_max <= 1.25
    # ga80 ships no dec inp -> zoning census stays NaN.
    assert e3.zone_pin_count is None or math.isnan(e3.zone_pin_count)


def test_lattice_type_fully_harvested(lib: FuelLibrary) -> None:
    b1 = lib.get("B1", "260624")
    assert b1.zone_pin_count is not None and b1.zone_pin_count > 0
    assert b1.xs_nf2 is not None and 0.10 <= b1.xs_nf2 <= 0.20
    assert b1.boron_worth is not None and -15.0 <= b1.boron_worth <= -3.0
    assert "fa_sum" in b1.source_flags
    assert "hgc_xs" in b1.source_flags
    assert "dec_census" in b1.source_flags


def test_state_point_suffixed_hgc_is_harvested(lib: FuelLibrary) -> None:
    # 5.8_5.1 B-series ships FA_B0x_0101.HGC (state-point suffix), not FA_B0x.HGC;
    # the harvester must still find it and fill the HGC-only xs/adf columns.
    b01 = lib.get("B01", "5.8_5.1")
    if b01.kinf0 is None or math.isnan(b01.kinf0):
        pytest.skip("5.8_5.1 B01 not hydrated")
    assert b01.xs_d1 is not None and not math.isnan(b01.xs_d1)
    assert b01.adf_corner_g2 is not None and b01.adf_corner_g2 > 1.0
    assert "hgc_xs" in b01.source_flags
