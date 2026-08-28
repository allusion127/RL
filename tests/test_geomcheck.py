"""Geometry-validation protocol (review sec. 4c): GEOM deck edit + frozen-envelope
guard, admissible variant grid, acceptance-band scoring, and the DeCART-less
dry-run end-to-end + CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lpopt.config import load_config
from lpopt.data.fuel_types import NOMINAL_ASM_PITCH, parse_dec_geom
from lpopt.design.lattice import LatticeError, edit_dec_geom_text, resolve_template
from lpopt.design.spec import ANCHOR_DESIGNS, FuelDesign
from lpopt.design.geomcheck import (
    ACCEPTANCE_BANDS,
    PITCH_CEIL_FRAC,
    GeomVariant,
    geom_variant_grid,
    run_geom_validation,
    score_variant,
)
from lpopt.vendor.masterrl.domain import FOM

REPO_ROOT = Path(__file__).resolve().parents[1]
DECK = REPO_ROOT / "lpopt.inp"
APR1400 = (REPO_ROOT / ".." / "0_APR1400").resolve()
SAMPLE_DEC = (REPO_ROOT / "data" / "design" / "curriculum_work" / "5.75-6_f109"
              / "P7" / "dec_FA_P7.inp")


def _sample_text() -> str:
    if not SAMPLE_DEC.is_file():
        pytest.skip(f"sample dec deck not present: {SAMPLE_DEC}")
    return SAMPLE_DEC.read_text(encoding="utf-8", errors="replace")


def _parse_text(text: str, tmp_path: Path) -> dict:
    p = tmp_path / "v.inp"
    p.write_text(text, encoding="utf-8")
    return parse_dec_geom(p)


# --------------------------------------------------------------------------- #
# GEOM deck editing + frozen-envelope guard
# --------------------------------------------------------------------------- #
def test_edit_geom_pitch_and_radii(tmp_path) -> None:
    text = _sample_text()
    new = edit_dec_geom_text(text, pin_pitch=1.285 * 1.005, r_pellet=0.4096 * 1.01,
                             r_clad_in=0.4178 * 1.01, r_clad_out=0.4750 * 1.01)
    g = _parse_text(new, tmp_path)
    assert g["pin_pitch"] == pytest.approx(1.285 * 1.005, rel=1e-5)
    assert g["r_pellet"] == pytest.approx(0.4096 * 1.01, rel=1e-5)
    assert g["r_clad_out"] == pytest.approx(0.4750 * 1.01, rel=1e-5)
    # frozen envelope: assembly pitch byte-identical.
    assert g["asm_pitch"] == pytest.approx(NOMINAL_ASM_PITCH)


def test_edit_geom_only_pitch_keeps_radii(tmp_path) -> None:
    text = _sample_text()
    new = edit_dec_geom_text(text, pin_pitch=1.30)
    g = _parse_text(new, tmp_path)
    assert g["pin_pitch"] == pytest.approx(1.30)
    assert g["r_pellet"] == pytest.approx(0.4096)          # untouched
    assert g["asm_pitch"] == pytest.approx(NOMINAL_ASM_PITCH)


def test_edit_geom_guide_tubes_preserved(tmp_path) -> None:
    text = _sample_text()
    new = edit_dec_geom_text(text, pin_pitch=1.29, r_pellet=0.41)
    orig_gt = [ln for ln in text.splitlines() if ln.strip().startswith("cellgeo 3")
               or ln.strip().startswith("cellgeo 4")
               or ln.strip().startswith("cellgeo 5")
               or ln.strip().startswith("cellgeo 6")]
    new_gt = [ln for ln in new.splitlines() if ln.strip().startswith("cellgeo 3")
              or ln.strip().startswith("cellgeo 4")
              or ln.strip().startswith("cellgeo 5")
              or ln.strip().startswith("cellgeo 6")]
    assert orig_gt == new_gt and len(orig_gt) == 4


def test_edit_geom_no_pitch_card_raises() -> None:
    with pytest.raises(LatticeError):
        edit_dec_geom_text("MATERIAL\n mixture UO2 2 10 626 / 92235 5.8\n", pin_pitch=1.3)


# --------------------------------------------------------------------------- #
# variant grid admissibility
# --------------------------------------------------------------------------- #
def test_variant_grid_drops_inadmissible_pitch() -> None:
    anchors = ANCHOR_DESIGNS[:1]
    g = geom_variant_grid([-0.03, 0.0, 0.03], [0.0], anchors)
    fracs = sorted(v.pitch_frac for v in g)
    assert 0.03 not in fracs                    # +3% exceeds the +1.06% ceiling
    assert -0.03 in fracs and 0.0 in fracs
    assert PITCH_CEIL_FRAC == pytest.approx((NOMINAL_ASM_PITCH / 16) / 1.285 - 1.0)


def test_variant_admissibility_reason() -> None:
    # a pitch above the ceiling is inadmissible; a huge radius closes the clad gap.
    v_pitch = GeomVariant("P0", ANCHOR_DESIGNS[0], pitch_frac=0.05, radius_frac=0.0)
    assert v_pitch.admissibility() is not None
    v_rad = GeomVariant("P0", ANCHOR_DESIGNS[0], pitch_frac=0.0, radius_frac=0.40)
    assert v_rad.admissibility() is not None
    v_ok = GeomVariant("P0", ANCHOR_DESIGNS[0], pitch_frac=0.005, radius_frac=0.01)
    assert v_ok.admissibility() is None


# --------------------------------------------------------------------------- #
# acceptance-band scoring
# --------------------------------------------------------------------------- #
def _fom(f_r, cbc, cyclen) -> FOM:
    return FOM(f_r=f_r, cbc_max=cbc, f_q=f_r * 1.2, cyclen=cyclen, converged=True)


def test_score_variant_pass_when_pred_tracks_truth() -> None:
    n = 8
    rng = np.random.default_rng(0)
    truth_fr = 1.4 + rng.random(n)
    truth_cy = 600 + 100 * rng.random(n)
    truth_cbc = 900 + 400 * rng.random(n)
    foms = [_fom(truth_fr[i], truth_cbc[i], truth_cy[i]) for i in range(n)]
    pred = np.full((n, 7), np.nan)
    pred[:, 0] = truth_fr + 0.01               # tight, monotone -> high Spearman
    pred[:, 1] = truth_cbc + 5.0
    pred[:, 3] = truth_cy + 2.0
    score = score_variant(pred, foms)
    assert score["verdict"] == "PASS"
    assert score["per_target"]["f_r"]["pass"]


def test_score_variant_fail_on_bad_predictions() -> None:
    n = 8
    foms = [_fom(1.4 + 0.1 * i, 1000 + 10 * i, 600 + 5 * i) for i in range(n)]
    pred = np.full((n, 7), np.nan)
    # anti-correlated f_r -> Spearman below the 0.70 band.
    pred[:, 0] = np.array([1.4 + 0.1 * (n - i) for i in range(n)])
    pred[:, 1] = np.array([1000.0 for _ in range(n)])       # constant -> Spearman None
    pred[:, 3] = np.array([600.0 + 200 * i for i in range(n)])  # huge MAE
    score = score_variant(pred, foms)
    assert score["verdict"] == "FAIL"


def test_acceptance_bands_match_review() -> None:
    assert ACCEPTANCE_BANDS["f_r"] == (0.70, 0.5)
    assert ACCEPTANCE_BANDS["cyclen"] == (0.60, 15.0)
    assert ACCEPTANCE_BANDS["cbc_max"] == (0.60, 50.0)


# --------------------------------------------------------------------------- #
# dry-run end-to-end + CLI
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cfg():
    if not DECK.is_file():
        pytest.skip("lpopt.inp not present")
    return load_config(DECK)


def _templates_available() -> bool:
    try:
        resolve_template(ANCHOR_DESIGNS[0], APR1400)
        return True
    except Exception:
        return False


def test_dry_run_e2e(tmp_path, cfg) -> None:
    if not _templates_available():
        pytest.skip("APR1400 dec templates not present")
    result = run_geom_validation(
        cfg, pitch_fracs=[-0.01, 0.0, 0.005], radius_fracs=[0.0, 0.01],
        anchors=ANCHOR_DESIGNS[:1], feed=121, probe_size=8, dry_run=True,
        scratch_dir=tmp_path / "gc", seed=1, log=lambda m: None,
    )
    assert result.overall_verdict == "PASS"
    assert result.n_admissible == 6
    # verdict + side table written to the scratch dir.
    assert result.verdict_path.is_file()
    assert (result.scratch_dir / "verdict.md").is_file()
    assert result.side_table_path.is_file()

    # every generated variant deck keeps the assembly-pitch envelope frozen.
    for deck in (result.scratch_dir / "decks").rglob("dec_*.inp"):
        g = parse_dec_geom(deck)
        assert g["asm_pitch"] == pytest.approx(NOMINAL_ASM_PITCH)

    # the nominal-geometry variant (pitch 0, radius 0) is NOT OOD; varied ones are.
    entries = {v["type_id"]: v for v in result.variants}
    nominal = [v for v in result.variants
               if v.get("pitch_frac") == 0.0 and v.get("radius_frac") == 0.0]
    assert nominal and not nominal[0]["ood_channels"]
    assert any(v.get("ood_channels") for v in result.variants)
    # verdict.json is well-formed.
    payload = json.loads(result.verdict_path.read_text())
    assert payload["overall_verdict"] == "PASS"
    assert payload["pitch_ceiling_frac"] == pytest.approx(PITCH_CEIL_FRAC)


def test_cli_geom_validate_dry_run(tmp_path) -> None:
    if not DECK.is_file() or not _templates_available():
        pytest.skip("deck / templates not present")
    from lpopt.cli import main
    rc = main([
        "geom-validate", "--input", str(DECK),
        "--pitch-grid=-1,0", "--radius-grid=0,1", "--anchors", "1",
        "--probe-size", "6", "--dry-run", "--scratch", str(tmp_path / "cli"),
    ])
    assert rc == 0
    assert (tmp_path / "cli" / "verdict.md").is_file()
