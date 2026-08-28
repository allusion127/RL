"""R1-R3 assembly-design compliance utilities (user rules 2026-07-22)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from lpopt.data import compliance as comp
from lpopt.search.genome import random_genome
from lpopt.vendor.masterrl.ga import ORBIT_UNITS


# --------------------------------------------------------------------------- #
# R2 — octant (1/8) pin-map symmetry
# --------------------------------------------------------------------------- #
def _octant_symmetric_map(n=16, seed=0):
    """A random map folded into full octant (D4) symmetry."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = np.minimum(m, m.T)                 # impose transpose symmetry
    m = np.minimum(m, m[:, ::-1])          # impose horizontal-mirror symmetry
    m = np.minimum(m, m.T)                 # re-close under the generator
    m = np.minimum(m, m[:, ::-1])
    # iterate to a fixed point of both folds.
    for _ in range(4):
        m = np.minimum(m, m.T)
        m = np.minimum(m, m[:, ::-1])
    return m.reshape(-1).tolist()


def test_octant_symmetric_pass():
    flat = _octant_symmetric_map(seed=1)
    assert comp.is_octant_symmetric(flat, n=16)
    assert comp.octant_symmetry_flag(flat, n=16) == "pass"


def test_octant_asymmetric_fail():
    m = np.zeros((16, 16))
    m[0, 15] = 1.0                          # breaks both transpose and mirror
    flat = m.reshape(-1).tolist()
    assert not comp.is_octant_symmetric(flat, n=16)
    assert comp.octant_symmetry_flag(flat, n=16) == "fail"


def test_octant_wrong_size_or_missing_is_unknown():
    assert comp.octant_symmetry_flag(None) == "unknown"
    assert comp.octant_symmetry_flag([]) == "unknown"
    assert comp.octant_symmetry_flag([1.0, 2.0, 3.0]) == "unknown"     # not 256
    # a wrong-length map cannot be certified True.
    assert not comp.is_octant_symmetric([1.0, 2.0, 3.0], n=16)
    # non-finite entries never certify.
    bad = _octant_symmetric_map(seed=2); bad[0] = float("nan")
    assert not comp.is_octant_symmetric(bad, n=16)


# --------------------------------------------------------------------------- #
# R1 — enr_zone = 0.85 x enr_main + cross-anchor detection
# --------------------------------------------------------------------------- #
def test_zone_ratio_flag_tolerance():
    assert comp.zone_ratio_flag(5.0, 4.25) == "pass"          # exactly 0.85
    assert comp.zone_ratio_flag(5.0, 4.30) == "pass"          # within 0.03
    assert comp.zone_ratio_flag(5.0, 3.50) == "fail"          # 0.70 far off
    # NaN / None / non-positive enrichment -> unknown (the all-NaN ga80 case).
    assert comp.zone_ratio_flag(None, 4.25) == "unknown"
    assert comp.zone_ratio_flag(5.0, None) == "unknown"
    assert comp.zone_ratio_flag(float("nan"), float("nan")) == "unknown"
    assert comp.zone_ratio_flag(0.0, 0.0) == "unknown"


def test_family_anchor_and_cross_anchor():
    assert comp.family_anchor("E1") == "E"
    assert comp.family_anchor("G3") == "G"
    # mono-anchor same-family pairs (the whole roster) are NOT cross-anchor.
    for pair in ("E1_E2", "J1_J2", "K1_K2", "L1_L2", "N1_N2", "G3_G4"):
        assert comp.is_cross_anchor(pair) is False
    # different family letters ARE cross-anchor (banned).
    assert comp.is_cross_anchor("E1_J2") is True
    assert comp.is_cross_anchor("G3_N4") is True
    # malformed -> fails safe as cross-anchor.
    assert comp.is_cross_anchor("E1") is True


def test_assert_mono_anchor_hard_fails_on_cross():
    comp.assert_mono_anchor(["E1_E2", "J1_J2"])               # all mono -> ok
    with pytest.raises(comp.ComplianceError, match="R1"):
        comp.assert_mono_anchor(["E1_E2", "E1_J2"])


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def test_audit_types_flags_and_source():
    good_map = _octant_symmetric_map(seed=3)
    bad = np.zeros((16, 16)); bad[0, 15] = 1.0
    recs = [
        {"type_id": "T_ok", "enr_main": 5.0, "enr_zone": 4.25, "dist_map": good_map},
        {"type_id": "T_asym", "enr_main": None, "enr_zone": None,
         "dist_map": bad.reshape(-1).tolist()},
        {"type_id": "T_nan", "enr_main": float("nan"), "enr_zone": float("nan"),
         "dist_map": None},
    ]
    rows = {r.type_id: r for r in comp.audit_types(recs)}
    assert rows["T_ok"].octant_symmetry == "pass"
    assert rows["T_ok"].zone_ratio == "pass"
    assert rows["T_ok"].compliance_source == "hgc%dist+enr"
    assert rows["T_asym"].octant_symmetry == "fail"
    assert rows["T_asym"].zone_ratio == "unknown"       # all-NaN ga80 enrichment
    assert rows["T_asym"].compliance_source == "hgc%dist"
    assert rows["T_nan"].octant_symmetry == "unknown"   # no preserved HGC
    assert rows["T_nan"].compliance_source == "none"


def test_audit_fuel_types_over_hgc_maps():
    maps = {"A": _octant_symmetric_map(seed=4)}
    rows = comp.audit_fuel_types(fuel=None, hgc_maps=maps)
    assert len(rows) == 1 and rows[0].type_id == "A"
    assert rows[0].octant_symmetry == "pass"
    assert rows[0].zone_ratio == "unknown"              # no enrichment supplied


# --------------------------------------------------------------------------- #
# Phase A contract — enforce_new_type
# --------------------------------------------------------------------------- #
def test_enforce_new_type_derives_zone_and_checks_octant():
    out = comp.enforce_new_type({"enr_main": 5.2})
    assert out["enr_zone"] == pytest.approx(0.85 * 5.2)
    # a supplied off-ratio zone is a hard R1 error.
    with pytest.raises(comp.ComplianceError, match="R1"):
        comp.enforce_new_type({"enr_main": 5.0, "enr_zone": 3.0})
    # a non-octant pin_map is a hard R2 error.
    bad = np.zeros((16, 16)); bad[0, 15] = 1.0
    with pytest.raises(comp.ComplianceError, match="R2"):
        comp.enforce_new_type({"enr_main": 5.0, "pin_map": bad.reshape(-1).tolist()})
    # an octant map + derived zone passes.
    ok = comp.enforce_new_type(
        {"enr_main": 5.0, "pin_map": _octant_symmetric_map(seed=5)})
    assert ok["enr_zone"] == pytest.approx(4.25)
    # missing / non-positive enr_main is rejected.
    with pytest.raises(comp.ComplianceError):
        comp.enforce_new_type({"enr_zone": 4.0})
    with pytest.raises(comp.ComplianceError):
        comp.enforce_new_type({"enr_main": 0.0})


# --------------------------------------------------------------------------- #
# R3 — orbit-unit placement preserves 1/4 rotational core symmetry (assert-only)
# --------------------------------------------------------------------------- #
def test_r3_orbit_units_preserve_quarter_core_symmetry():
    # A genome compiles the quarter core by placing IDENTICAL fuel content on every
    # slot of an orbit unit (the rotational-equivalence class).  So the axis-twin
    # arms of every unit hold the same fuel, and the pattern round-trips through the
    # orbit structure unchanged — the quarter-core rotational symmetry is structural.
    for seed in (1, 7, 42):
        genome = random_genome(random.Random(seed), "K1_K2", 30)
        pattern = genome.to_pattern()
        for unit in ORBIT_UNITS:
            arms = [pattern.items[s] for s in unit.slots]
            kinds = {a.is_fresh for a in arms}
            assert len(kinds) == 1                        # arms share a fuel state
            if arms[0].is_fresh:
                assert len({a.batch for a in arms}) == 1  # same fresh batch
            else:
                assert len({(a.restart, a.x, a.y) for a in arms}) == 1  # same source
        # round-trip: the orbit structure (hence symmetry) is preserved exactly.
        from lpopt.search.genome import GeneralOrbitGenome
        assert GeneralOrbitGenome.from_pattern(pattern) == genome


# --------------------------------------------------------------------------- #
# CLI: compliance-audit output shape
# --------------------------------------------------------------------------- #
def test_cli_compliance_audit_output_shape(tmp_path):
    import json
    from pathlib import Path

    deck = Path(__file__).resolve().parents[1] / "lpopt.inp"
    fuel = Path(__file__).resolve().parents[1] / "data" / "store" / "fuel_types.parquet"
    if not deck.is_file() or not fuel.is_file():
        pytest.skip("deck / fuel table not present")
    from lpopt.cli import main
    out = tmp_path / "audit.json"
    rc = main(["compliance-audit", "--input", str(deck), "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) >= {"library_id", "n_types", "octant_symmetry",
                           "zone_ratio", "types"}
    assert report["n_types"] == len(report["types"])
    # every per-type row carries the four compliance flags + key.
    for row in report["types"]:
        assert set(row) == {"type_id", "library_id", "octant_symmetry",
                            "zone_ratio", "compliance_source"}
        assert row["octant_symmetry"] in {"pass", "fail", "unknown"}
        assert row["zone_ratio"] in {"pass", "fail", "unknown"}
    # all-NaN ga80 enrichment -> every zone_ratio flag is 'unknown' (the audit fact).
    assert set(report["zone_ratio"]) == {"unknown"}
