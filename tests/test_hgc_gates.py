"""HGC product gates G-H1 / G-H1b / G-H1c / G-H2 / G-H4 (task #11).

Everything runs off a synthetic HGC built to the frozen golden-deck contract
(design v2 §5.2): 334 ``%TITL`` states = DEPL 62 + BRANCH 16x17, four per-state
tags, one trailing ``%FINE``, a 16x16 ``%DIST`` map carrying ``n_gd`` Gd pins and
4 guide tubes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lpopt.design import hgc_gates as hg

BRANCH_LABELS = (
    ["BORON VARIATION", "TFUEL VARIATION"]
    + [f"DMOD{i} VARIATION" for i in range(1, 7)]
    + ["CR1 REFERENCE", "CR1 VARIATION BOR"]
    + [f"CR1 VARIATION DMOD{i}" for i in range(1, 7)]
)
assert len(BRANCH_LABELS) == hg.N_BRANCHES

#: Guide-tube positions read exactly 0.000 (octant seeds, mirrored to 4 corners).
GUIDE = {(3, 3), (3, 12), (12, 3), (12, 12)}

# The frozen golden deck's depletion grid, verbatim:
# ``burnup 0.0 0.2 0.5 1 -45/1.0 -80/2.5`` -> 3 + 45 + 14 = 62 points.  This is
# byte-for-byte the grid the real products carry (verified S4-B, 2026-09-03),
# so a fixture-only BU range can never mask a window/threshold error again.
DEPL_BURNUPS = ([0.0, 0.2, 0.5] + [float(i) for i in range(1, 46)]
                + [47.5 + 2.5 * i for i in range(14)])
assert len(DEPL_BURNUPS) == hg.N_DEPL_STATES
BRANCH_BURNUPS = [0.0, 0.2, 0.5, 1, 3, 5, 7, 10, 15, 20,
                  25, 30, 40, 50, 60, 70, 80]
assert len(BRANCH_BURNUPS) == hg.N_BRANCH_POINTS


def _kinf(bu: float) -> float:
    """Gd-bearing shape: rises to a burnout peak near 10 GWd/tU, then decays."""
    return 1.25 + 0.03 * min(bu, 10.0) / 10.0 - 0.004 * max(bu - 10.0, 0.0)


def _dist_block(n_gd: int, ff_peak: float) -> list[str]:
    """A 16x16 BOC pin-power map: ``n_gd`` Gd pins, 4 guide tubes, peak ``ff_peak``."""
    grid = [[1.0] * 16 for _ in range(16)]
    for r, c in GUIDE:
        grid[r][c] = 0.0
    placed = 0
    for r in range(16):
        for c in range(16):
            if placed >= n_gd:
                break
            if (r, c) in GUIDE or (r, c) == (0, 0):
                continue
            grid[r][c] = 0.35
            placed += 1
    grid[0][0] = ff_peak
    return ["%DIST"] + [" " + " ".join(f"{v:.3f}" for v in row) for row in grid]


def _state_block(case: str, bu: float, kinf: float, *, n_gd: int,
                 ff_peak: float) -> list[str]:
    return [
        "%TITL",
        "=" * 80,
        "",
        f" CASE :: {case}",
        "           2           1          16 2.07772E+01 1.28500E+00 4.31692E+02",
        f" 3.89482E+01 {bu:.5E} {kinf:.5E} 1.25247E+00 4.41786E-03 9.00000E+02",
        " 5.80900E+02 5.00000E+02 0.00000E+00 1.50000E+02 7.08823E-01 1.11379E+03",
        "=" * 80,
        *_dist_block(n_gd, ff_peak),
        "%MACX",
        " 1.4 0.009 0.0 0.006 0.0 0.02",
        " 0.4 0.070 0.0 0.100 0.0 0.00",
        " 0.0 0.02",
        "%MICX",
        " 1.0",
        "%ADFT",
        " 1.02",
        " 1.05",
        " 1.01",
        " 1.03",
    ]


def build_hgc(*, n_gd: int = 20, ff: dict[float, float] | None = None,
              kinf: dict[float, float] | None = None,
              n_depl: int = hg.N_DEPL_STATES,
              branches: int = hg.N_BRANCHES,
              fine: int = 1) -> str:
    """Synthetic HGC text; ``ff`` / ``kinf`` override the reference-curve values."""
    ff = ff or {}
    kinf = kinf or {}
    lines: list[str] = []
    for bu in DEPL_BURNUPS[:n_depl]:
        lines += _state_block(hg.CASE_REFERENCE, bu, kinf.get(bu, _kinf(bu)),
                              n_gd=n_gd, ff_peak=ff.get(bu, 1.086))
    for label in BRANCH_LABELS[:branches]:
        for bu in BRANCH_BURNUPS:
            lines += _state_block(label, bu, _kinf(bu), n_gd=n_gd, ff_peak=1.086)
    lines += ["%FINE"] * fine
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# G-H1 — structure
# --------------------------------------------------------------------------- #
def test_structure_census_of_a_good_product() -> None:
    st = hg.parse_structure(build_hgc())
    assert st["titl"] == hg.N_TITL_EXPECTED == 334
    assert st["fine"] == 1
    assert all(st["tags"][t] == 334 for t in hg.STATE_TAGS)
    assert st["cases"][hg.CASE_REFERENCE] == 62
    assert len(st["cases"]) == 17                       # 1 reference + 16 branches


def test_gate_h1_passes_a_complete_product() -> None:
    res = hg.gate_h1_structure(build_hgc())
    assert res.status == hg.PASS and res.ok
    assert res.metrics["titl"] == 334


def test_gate_h1_fails_a_truncated_product() -> None:
    text = build_hgc()
    truncated = text[: int(len(text) * 0.7)]
    res = hg.gate_h1_structure(truncated)
    assert res.status == hg.FAIL and not res.ok
    assert "%TITL" in res.detail


def test_gate_h1_fails_a_missing_branch() -> None:
    """A missing branch is never tolerated (risk R11: CRD1* only lives there)."""
    res = hg.gate_h1_structure(build_hgc(branches=hg.N_BRANCHES - 1))
    assert res.status == hg.FAIL
    assert "branch labels 15 != 16" in res.detail


def test_gate_h1_fails_a_missing_fine_terminator() -> None:
    res = hg.gate_h1_structure(build_hgc(fine=0))
    assert res.status == hg.FAIL and "%FINE 0 != 1" in res.detail


# --------------------------------------------------------------------------- #
# G-H1b — size, with the ABSTAIN rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_gd", hg.HGC_SIZE_GATED_N_GD)
def test_gate_h1b_exact_size_for_the_measured_n_gd_set(n_gd: int) -> None:
    assert hg.gate_h1b_size(hg.HGC_SIZE_BYTES, n_gd).status == hg.PASS
    off = hg.gate_h1b_size(hg.HGC_SIZE_BYTES - 1, n_gd)
    assert off.status == hg.FAIL and str(hg.HGC_SIZE_BYTES) in off.detail


@pytest.mark.parametrize("n_gd", [0, 4, 8])
def test_gate_h1b_abstains_outside_the_measured_set(n_gd: int) -> None:
    """n_gd=0 measures 6,867,567 B for reasons unknown; {4,8} were never made."""
    res = hg.gate_h1b_size(hg.HGC_SIZE_BYTES_NO_GD, n_gd)
    assert res.status == hg.ABSTAIN
    assert not res.ok                       # an ABSTAIN is not a pass
    assert "manual review" in res.detail
    assert res.metrics["expected_bytes"] is None


def test_abstain_folds_to_abstain_but_a_fail_dominates() -> None:
    ok = hg.GateResult("a", hg.PASS, "")
    ab = hg.GateResult("b", hg.ABSTAIN, "")
    bad = hg.GateResult("c", hg.FAIL, "")
    assert hg.verdict([ok, ok]) == hg.PASS
    assert hg.verdict([ok, ab]) == hg.ABSTAIN
    assert hg.verdict([ok, ab, bad]) == hg.FAIL
    assert hg.verdict([]) == hg.ABSTAIN


# --------------------------------------------------------------------------- #
# G-H1c — validity + per-BU sanity
# --------------------------------------------------------------------------- #
def test_gate_h1c_passes_and_reports_the_curve() -> None:
    res = hg.gate_h1c_validity(build_hgc(), size_bytes=hg.HGC_SIZE_BYTES)
    assert res.status == hg.PASS
    assert res.metrics["n_reference_points"] == 62
    assert res.metrics["burnup_min"] == 0.0
    assert res.metrics["kinf_peak_burnup"] == pytest.approx(10.0)


def test_gate_h1c_fails_an_empty_or_stub_product() -> None:
    assert hg.gate_h1c_validity("", size_bytes=0).status == hg.FAIL
    assert hg.gate_h1c_validity("%TITL\n CASE :: REFERENCE CASE\n",
                                size_bytes=64).status == hg.FAIL


def test_gate_h1c_fails_a_non_monotonic_reference_curve() -> None:
    """A k-inf rise after the Gd burnout peak is corruption, not physics."""
    bu = 40.0                               # past the Gd-burnout window
    assert bu > hg.GD_BURNOUT_BU_MAX
    res = hg.gate_h1c_validity(build_hgc(kinf={bu: _kinf(bu) + 0.02}))
    assert res.status == hg.FAIL and "after the Gd-burnout peak" in res.detail


# --- AMENDMENT 2026-09-03 (S4-B): the peak is the Gd-burnout local max ------ #
def test_burnout_peak_is_the_local_max_not_the_global_max() -> None:
    """The registered rule: last significant local max at BU <= 25."""
    bus = [0.0, 0.2, 0.5, 1.0] + [1.0 + i for i in range(1, 60)]
    # BOC is the global maximum; the Gd-burnout peak is a LOWER local max at 18.
    kin = []
    for b in bus:
        if b <= 1.0:
            kin.append(1.1500 - 0.0300 * b)
        elif b <= 18.0:
            kin.append(1.1200 + 0.00150 * (b - 1.0))
        else:
            kin.append(max(1.1200 + 0.00150 * 17.0 - 0.00600 * (b - 18.0), 0.62))
    assert kin[0] == max(kin)                       # BOC is the global max
    peak = hg.burnout_peak_index(bus, kin)
    assert bus[peak] == pytest.approx(18.0)         # ... but not the burnout peak


def test_burnout_peak_falls_back_to_boc_without_a_gd_hump() -> None:
    bus = [0.0, 0.2, 0.5] + [1.0 + 0.5 * i for i in range(59)]
    kin = [1.25 - 0.004 * b for b in bus]           # monotone decay, no hump
    assert hg.burnout_peak_index(bus, kin) == 0


def test_burnout_peak_ignores_sub_tolerance_ripple() -> None:
    bus = [0.0, 0.2, 0.5] + [1.0 + 0.5 * i for i in range(59)]
    kin = [1.25 - 0.004 * b for b in bus]
    kin[30] += 5.0e-5                               # ripple far below K_MONOTONE_TOL
    assert hg.burnout_peak_index(bus, kin) == 0


def test_gate_h1c_passes_a_high_gd_curve_whose_global_max_is_boc() -> None:
    """Regression for the S4-B defect: 5 of 6 approved n_gd=20 library HGCs
    failed here because the global max (BOC) was read as the burnout peak."""
    curve = {}
    for b in DEPL_BURNUPS:
        if b <= 1.0:
            curve[b] = 1.1500 - 0.0300 * b
        elif b <= 18.0:
            curve[b] = 1.1200 + 0.00150 * (b - 1.0)
        else:
            curve[b] = max(1.1200 + 0.00150 * 17.0 - 0.00600 * (b - 18.0), 0.62)
    res = hg.gate_h1c_validity(build_hgc(kinf=curve))
    assert res.status == hg.PASS
    assert res.metrics["kinf_peak_burnup"] == pytest.approx(18.0)
    assert res.metrics["kinf_global_max_burnup"] == pytest.approx(0.0)


def test_gate_h1c_still_fails_a_genuine_post_burnout_rise_on_that_curve() -> None:
    """The amended gate must not be toothless: the same high-Gd shape with a
    real k-inf rise after the Gd window is still a FAIL."""
    curve = {}
    for b in DEPL_BURNUPS:
        if b <= 1.0:
            curve[b] = 1.1500 - 0.0300 * b
        elif b <= 18.0:
            curve[b] = 1.1200 + 0.00150 * (b - 1.0)
        else:
            curve[b] = max(1.1200 + 0.00150 * 17.0 - 0.00600 * (b - 18.0), 0.62)
    bu = 40.0
    assert bu > hg.GD_BURNOUT_BU_MAX
    curve[bu] = curve[bu] + 0.02
    res = hg.gate_h1c_validity(build_hgc(kinf=curve))
    assert res.status == hg.FAIL and "after the Gd-burnout peak" in res.detail


def test_reference_curve_shape() -> None:
    points = hg.reference_curve(build_hgc())
    assert len(points) == 62
    assert [p.burnup for p in points] == pytest.approx(DEPL_BURNUPS)
    assert points[0].ff_pin_max == pytest.approx(1.086)


# --------------------------------------------------------------------------- #
# G-H2 — Gd census
# --------------------------------------------------------------------------- #
def test_gate_h2_census_20_of_20() -> None:
    res = hg.gate_h2_gd_census(build_hgc(n_gd=20), 20)
    assert res.status == hg.PASS and res.metrics == {"counted": 20, "requested": 20}


def test_gate_h2_fails_a_census_mismatch() -> None:
    res = hg.gate_h2_gd_census(build_hgc(n_gd=16), 20)
    assert res.status == hg.FAIL and res.metrics["counted"] == 16


def test_text_census_agrees_with_fuel_types(tmp_path: Path) -> None:
    """The text-domain twin must not drift from ``count_gd_pins_from_hgc``."""
    pytest.importorskip("pandas")
    from lpopt.data.fuel_types import count_gd_pins_from_hgc

    text = build_hgc(n_gd=24)
    path = tmp_path / "FA_Z1.HGC"
    path.write_text(text, encoding="utf-8")
    assert count_gd_pins_from_hgc(path) == hg.count_gd_pins_from_text(text) == 24


# --------------------------------------------------------------------------- #
# G-H4 — screen regression (AMENDMENT 2026-09-03: |d rho| <= 350 pcm / 0.0021 FF)
# --------------------------------------------------------------------------- #
def _screen(points, *, dk: float = 0.0, dff: float = 0.0, at: float | None = None):
    kinf = {}
    ff = {}
    for p in points:
        hit = (at is None or p.burnup == at)
        kinf[p.burnup] = p.kinf + (dk if hit else 0.0)
        ff[p.burnup] = (p.ff_pin_max or 0.0) + (dff if hit else 0.0)
    return kinf, ff


def _drho_pcm(k: float, dk: float) -> float:
    return abs((1.0 - 1.0 / (k + dk)) - (1.0 - 1.0 / k)) * 1.0e5


def test_gate_h4_measures_reactivity_not_k() -> None:
    """AMENDMENT: the registered quantity is |d rho| pcm (rho = 1 - 1/k), the
    quantity opmodel/s02_surrogate_vs_decart.py differences — NOT |dk| * 1e5."""
    points = hg.reference_curve(build_hgc())
    at = 20.0
    k0 = next(p.kinf for p in points if p.burnup == at)
    kinf, ff = _screen(points, dk=1.5e-3, at=at)
    res = hg.gate_h4_screen_regression(points, kinf, ff)
    assert res.metrics["k_metric"] == "abs_drho_pcm"
    assert res.metrics["max_drho_pcm"] == pytest.approx(_drho_pcm(k0, 1.5e-3), abs=0.2)
    # the superseded |dk| * 1e5 convention would have read 150.0 pcm here
    assert res.metrics["max_drho_pcm"] < 150.0


def test_gate_h4_bar_is_the_amended_350_pcm() -> None:
    assert hg.G_H4_K_TOL_PCM == 350.0
    assert hg.G_H4_K_TOL_PCM_SUPERSEDED == 100.0


def test_gate_h4_passes_inside_the_holdout_thresholds() -> None:
    points = hg.reference_curve(build_hgc())
    kinf, ff = _screen(points, dk=3.0e-3, dff=0.002)
    res = hg.gate_h4_screen_regression(points, kinf, ff)
    assert res.status == hg.PASS
    assert 150.0 < res.metrics["max_drho_pcm"] <= hg.G_H4_K_TOL_PCM
    assert res.metrics["n_kinf_compared"] == 61          # BU = 0.0 excluded


def test_gate_h4_passes_what_the_superseded_100_pcm_bar_rejected() -> None:
    """T7 measured 199.3 pcm |dk| under the old wiring; the holdout it was
    gated against measured 155.6-383.9 pcm on the same harness."""
    points = hg.reference_curve(build_hgc())
    kinf, ff = _screen(points, dk=2.5e-3, at=20.0)
    res = hg.gate_h4_screen_regression(points, kinf, ff)
    assert res.status == hg.PASS
    assert res.metrics["max_drho_pcm"] > hg.G_H4_K_TOL_PCM_SUPERSEDED


def test_gate_h4_fails_a_k_regression() -> None:
    points = hg.reference_curve(build_hgc())
    at = 30.0                                # the softest point on the fixture
    k0 = next(p.kinf for p in points if p.burnup == at)
    kinf, ff = _screen(points, dk=6.0e-3, at=at)
    assert _drho_pcm(k0, 6.0e-3) > hg.G_H4_K_TOL_PCM
    res = hg.gate_h4_screen_regression(points, kinf, ff)
    assert res.status == hg.FAIL
    assert "drho" in res.detail and "BU=30" in res.detail


def test_gate_h4_fails_an_ff_regression() -> None:
    points = hg.reference_curve(build_hgc())
    kinf, ff = _screen(points, dff=0.003, at=5.0)
    res = hg.gate_h4_screen_regression(points, kinf, ff)
    assert res.status == hg.FAIL and "dFF" in res.detail


def test_gate_h4_ignores_deviation_below_bu_min() -> None:
    """The gate is a BU >= 0.2 statement; the fresh point is out of scope."""
    points = hg.reference_curve(build_hgc())
    kinf, ff = _screen(points, dk=0.5, dff=0.5, at=0.0)
    assert hg.gate_h4_screen_regression(points, kinf, ff).status == hg.PASS


def test_gate_h4_abstains_when_nothing_is_comparable() -> None:
    points = hg.reference_curve(build_hgc())
    res = hg.gate_h4_screen_regression(points, {999.0: 1.0})
    assert res.status == hg.ABSTAIN and not res.ok


def test_gate_h4_accepts_pair_sequences_and_refuses_junk() -> None:
    points = hg.reference_curve(build_hgc())
    pairs = [(p.burnup, p.kinf) for p in points]
    assert hg.gate_h4_screen_regression(points, pairs).status == hg.PASS
    with pytest.raises(hg.HgcGateError):
        hg.gate_h4_screen_regression(points, [(1.0, 2.0, 3.0)])


# --------------------------------------------------------------------------- #
# aggregate + file entry point
# --------------------------------------------------------------------------- #
def test_run_gates_all_pass_and_g_h4_is_skipped_not_vacuous() -> None:
    text = build_hgc(n_gd=20)
    results = hg.run_gates(text, n_gd=20, size_bytes=hg.HGC_SIZE_BYTES)
    assert [r.gate for r in results] == ["G-H1", "G-H1b", "G-H1c", "G-H2"]
    assert hg.verdict(results) == hg.PASS

    points = hg.reference_curve(text)
    kinf, ff = _screen(points)
    full = hg.run_gates(text, n_gd=20, size_bytes=hg.HGC_SIZE_BYTES,
                        screen_kinf=kinf, screen_ff=ff)
    assert [r.gate for r in full][-1] == "G-H4"
    assert hg.verdict(full) == hg.PASS


def test_run_gates_for_file(tmp_path: Path) -> None:
    path = tmp_path / "FA_Z1.HGC"
    path.write_text(build_hgc(n_gd=20), encoding="utf-8")
    results = hg.run_gates_for_file(path, n_gd=20)
    # the fixture is not 7,395,955 B, so the size gate is the only failure
    assert {r.gate: r.status for r in results}["G-H1b"] == hg.FAIL
    assert {r.gate: r.status for r in results}["G-H1"] == hg.PASS
    assert {r.gate: r.status for r in results}["G-H2"] == hg.PASS

    missing = hg.run_gates_for_file(tmp_path / "nope.HGC", n_gd=20)
    assert hg.verdict(missing) == hg.FAIL
