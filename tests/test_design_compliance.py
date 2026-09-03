"""``enforce_new_type`` gets a production caller at last (task #16).

Until now the Phase-A compliance contract had zero production callers, and it
fails OPEN in two directions that this suite pins:

* an omitted ``enr_zone`` is silently OVERWRITTEN with ``0.85 * enr_main``
  (``lpopt/data/compliance.py:308-311``) — not rejected.  A screen run at an 0.92
  zoning ratio would pass and then be realized at 0.85.
* an omitted ``pin_map`` skips the R2 octant check entirely
  (``compliance.py:321-322``).

``lpopt.design.compliance.enforce_design`` therefore always passes BOTH
enrichments and the full 16x16 map, and ``lattice.write_authored_deck`` calls it
on the only path that realizes an on-demand assembly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lpopt.data.compliance import (
    ZONE_RATIO_TARGET,
    ComplianceError,
    enforce_new_type,
    is_octant_symmetric,
)
from lpopt.design.compliance import design_spec, enforce_design
from lpopt.design.lattice import (
    GD_CELL_ID,
    LatticeError,
    author_gd_layout,
    octant_census,
    octant_to_full,
    parse_octant_triangle,
    write_authored_deck,
)
from lpopt.design.spec import DesignRegistry, FuelDesign

# The synthetic IGD_20 / 8_20_z1 deck fixture (pytest puts ``tests/`` on sys.path).
from test_design_lattice_author import (  # noqa: E402
    OPEN_N20,
    TEMPLATE,
    _write_frozen_tree,
)

#: Slice Z1': 5.50 / 4.70 -> ratio 0.854545, inside 0.85 +/- 0.03.
COMPLIANT = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions=OPEN_N20)
#: The live 0.92-ratio shape (P6257Z2G08N16 is 6.2/5.7 = 0.919) — outside the window.
#: Kept on z1 so it resolves the same template dir and dies on R1, not on lookup.
OFF_RATIO = FuelDesign(6.2, 5.7, "z1", 8.0, 20, gd_positions=OPEN_N20)


def _full_map(layout: str = OPEN_N20) -> list[int]:
    _l, rows, _i = parse_octant_triangle(author_gd_layout(TEMPLATE, layout, 20))
    return octant_to_full(rows)


# --------------------------------------------------------------------------- #
# the adapter always hands over enr_main AND enr_zone AND the full map
# --------------------------------------------------------------------------- #
def test_design_spec_never_omits_enr_zone() -> None:
    spec = design_spec(COMPLIANT)
    assert spec["enr_main"] == 5.5
    assert spec["enr_zone"] == 4.70          # explicit, never left for the default
    assert spec["gd_positions"] == OPEN_N20
    assert "pin_map" not in spec
    spec2 = design_spec(COMPLIANT, pin_map=_full_map())
    assert len(spec2["pin_map"]) == 256


def test_enforce_design_passes_a_compliant_design() -> None:
    out = enforce_design(COMPLIANT, pin_map=_full_map())
    assert out["enr_main"] == 5.5 and out["enr_zone"] == 4.70
    assert abs(out["enr_zone"] / out["enr_main"] - ZONE_RATIO_TARGET) < 0.03


def test_enforce_design_rejects_the_092_ratio() -> None:
    """0.92 is OUTSIDE the 0.85 +/- 0.03 window: reject, never normalize."""
    assert abs(OFF_RATIO.ratio - 0.919) < 1e-3
    with pytest.raises(ComplianceError, match="R1 violation"):
        enforce_design(OFF_RATIO, pin_map=_full_map())
    # and specifically NOT by quietly rewriting enr_zone to 0.85 * enr_main
    with pytest.raises(ComplianceError):
        enforce_design(OFF_RATIO)


def test_enforce_design_rejects_an_asymmetric_map() -> None:
    bad = _full_map()
    bad[0] = 99                               # break the D4 invariance in one cell
    assert not is_octant_symmetric(bad, n=16)
    with pytest.raises(ComplianceError, match="R2 violation"):
        enforce_design(COMPLIANT, pin_map=bad)


# --------------------------------------------------------------------------- #
# the two fail-open behaviours of the underlying contract, pinned as regressions
# --------------------------------------------------------------------------- #
def test_missing_enr_zone_is_overwritten_not_rejected() -> None:
    """Regression on the DEFAULT-FILL behaviour (compliance.py:308-311).

    This is exactly why the adapter must pass ``enr_zone`` itself: an 0.92 spec
    that simply omits it does not fail — it comes back silently re-zoned to 0.85.
    """
    out = enforce_new_type({"enr_main": 6.2})
    assert out["enr_zone"] == pytest.approx(0.85 * 6.2)


def test_missing_pin_map_skips_the_r2_check() -> None:
    """Regression on the OPTIONAL pin_map (compliance.py:321-322)."""
    assert enforce_new_type({"enr_main": 5.5, "enr_zone": 4.675})["enr_zone"] == 4.675
    asym = [0.0] * 256
    asym[0] = 1.0
    with pytest.raises(ComplianceError, match="R2 violation"):
        enforce_new_type({"enr_main": 5.5, "enr_zone": 4.675, "pin_map": asym})


def test_enr_zone_present_and_off_ratio_raises() -> None:
    with pytest.raises(ComplianceError, match="R1 violation"):
        enforce_new_type({"enr_main": 6.2, "enr_zone": 5.7})


# --------------------------------------------------------------------------- #
# wiring: the on-demand deck path is the production caller
# --------------------------------------------------------------------------- #
def test_write_authored_deck_enforces_compliance(tmp_path: Path) -> None:
    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    reg = DesignRegistry()
    deck = write_authored_deck(COMPLIANT, tmp_path / "work", reg, apr,
                               tmp_path / "templates")
    _l, rows, _i = parse_octant_triangle(deck.read_text(encoding="utf-8"))
    assert len(octant_census(rows, GD_CELL_ID)) == 3

    with pytest.raises(ComplianceError, match="R1 violation"):
        write_authored_deck(OFF_RATIO, tmp_path / "work2", DesignRegistry(), apr,
                            tmp_path / "templates2")


def test_authoring_still_guards_before_compliance(tmp_path: Path) -> None:
    """A layout/census fault is a LatticeError from the authoring guards, not R1/R2."""
    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    bad = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions="1:1;4:1;7:6")
    with pytest.raises(LatticeError, match="guide tube or edge zoning"):
        write_authored_deck(bad, tmp_path / "w", DesignRegistry(), apr, tmp_path / "t")


def test_a_rejected_design_leaves_no_authored_deck(tmp_path: Path) -> None:
    """Review fix: the template tree must not keep the deck of a refused design.

    ``author_template`` writes before ``enforce_design`` runs, so without the
    cleanup an R1-rejected lattice leaves ``dec_FA_<type_id>.inp`` behind, where
    ``resolve_template(..., template_root=)`` PREFERS it on every later call.
    """
    from lpopt.design.lattice import authored_deck_name, template_dir

    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    troot = tmp_path / "templates"
    with pytest.raises(ComplianceError, match="R1 violation"):
        write_authored_deck(OFF_RATIO, tmp_path / "work", DesignRegistry(), apr, troot)
    assert not (template_dir(OFF_RATIO, troot) / authored_deck_name(OFF_RATIO)).exists()
