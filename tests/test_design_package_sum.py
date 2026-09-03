"""``.sum`` + authored-deck staging and the ``harvest`` deck-survival rule (#13b).

``data/design/package/hgc/`` holds 37 ``.HGC`` + 37 ``.out`` and **nothing else**:
``lattice.harvest`` produced only those two and ``package.stage_hgc`` copied only
those two, so the cond_v4 harvest had no ``FA_<alias>.sum`` to read and
``ingest_fuel_types`` left ``zone_pin_count`` NaN for want of a staged
``dec_FA_*.inp``.  Both channels now exist, and the deck that is staged is the
AUTHORED one — never the working directory's ``decart.inp``, which ``harvest``
deletes.
"""

from __future__ import annotations

from pathlib import Path

from lpopt.design.lattice import DecartRun, harvest
from lpopt.design.package import DesignSource, stage_hgc, write_designs_manifest
from lpopt.design.spec import DesignRegistry, FuelDesign

DESIGN = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions="1:1;4:1;6:4")
HGC_BODY = "%TITL x\nCASE :: 1\n%DIST\n" + ("0.0 " * 64 + "\n") * 8


def _make_run(wd: Path, alias: str = "T7") -> DecartRun:
    """A finished DeCART working directory: raw HGC + .out + .sum + the authored deck."""
    wd.mkdir(parents=True, exist_ok=True)
    caseid = f"FA_{alias}"
    (wd / f"{caseid}_0101.HGC").write_text(HGC_BODY, encoding="utf-8")
    (wd / f"{caseid}.out").write_text("MASS(g)\n", encoding="utf-8")
    (wd / f"{caseid}.sum").write_text("EDIT1 CASE :: REF\n", encoding="utf-8")
    deck = wd / f"dec_FA_{alias}.inp"
    deck.write_text("CASEID FA_T7\n assembly FA_T7 45 1\n", encoding="utf-8")
    (wd / "decart.inp").write_text("staged copy\n", encoding="utf-8")
    return DecartRun(design=DESIGN, alias=alias, work_dir=wd, caseid=caseid,
                     fa_name=caseid, deck_path=deck, returncode=0)


# --------------------------------------------------------------------------- #
# harvest
# --------------------------------------------------------------------------- #
def test_harvest_keeps_the_sum_and_the_authored_deck(tmp_path: Path) -> None:
    wd = tmp_path / "T7"
    run = harvest(_make_run(wd))
    assert run.hgc_path == wd / "FA_T7.HGC"
    assert run.out_path == wd / "FA_T7.out"
    assert run.sum_path == wd / "FA_T7.sum" and run.sum_path.is_file()
    assert run.sum_path.read_text(encoding="utf-8") == "EDIT1 CASE :: REF\n"
    # the working copy is still removed, the AUTHORED deck still exists (regression)
    assert not (wd / "decart.inp").exists()
    assert run.deck_path.is_file()


def test_harvest_without_a_sum_is_not_an_error(tmp_path: Path) -> None:
    wd = tmp_path / "T8"
    run = _make_run(wd, "T8")
    (wd / f"{run.caseid}.sum").unlink()
    run = harvest(run)
    assert run.sum_path is None
    assert run.hgc_path.is_file() and run.out_path.is_file()


# --------------------------------------------------------------------------- #
# stage_hgc
# --------------------------------------------------------------------------- #
def test_stage_hgc_stages_sum_and_deck(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    run = harvest(_make_run(tmp_path / "work" / "T7"))
    src = DesignSource(design=DESIGN, alias=run.alias, hgc_path=run.hgc_path,
                       out_path=run.out_path, sum_path=run.sum_path,
                       deck_path=run.deck_path)
    staged = stage_hgc(pkg, [src])
    hgc_dir = pkg / "hgc"
    assert staged == [hgc_dir / "FA_T7.HGC"]
    for name in ("FA_T7.HGC", "FA_T7.out", "FA_T7.sum", "dec_FA_T7.inp"):
        assert (hgc_dir / name).is_file(), name
    assert (hgc_dir / "dec_FA_T7.inp").read_text(encoding="utf-8") == \
        run.deck_path.read_text(encoding="utf-8")


def test_stage_hgc_counts_after_a_two_type_wave(tmp_path: Path) -> None:
    """N new sources -> N ``.sum`` and N ``dec_FA_*.inp`` beside the N HGCs."""
    pkg = tmp_path / "pkg"
    sources = []
    for alias in ("T7", "T8"):
        run = harvest(_make_run(tmp_path / "work" / alias, alias))
        sources.append(DesignSource(design=DESIGN, alias=alias,
                                    hgc_path=run.hgc_path, out_path=run.out_path,
                                    sum_path=run.sum_path, deck_path=run.deck_path))
    stage_hgc(pkg, sources)
    hgc_dir = pkg / "hgc"
    assert len(list(hgc_dir.glob("FA_*.HGC"))) == 2
    assert len(list(hgc_dir.glob("FA_*.sum"))) == 2
    assert len(list(hgc_dir.glob("dec_FA_*.inp"))) == 2


def test_stage_hgc_without_companions_is_unchanged(tmp_path: Path) -> None:
    """The historical 2-file contract still holds for a source with no companions."""
    pkg = tmp_path / "pkg"
    run = harvest(_make_run(tmp_path / "work" / "T7"))
    src = DesignSource(design=DESIGN, alias="T7", hgc_path=run.hgc_path,
                       out_path=run.out_path)
    stage_hgc(pkg, [src])
    names = sorted(p.name for p in (pkg / "hgc").iterdir())
    assert names == ["FA_T7.HGC", "FA_T7.out"]


def test_manifest_records_the_layout(tmp_path: Path) -> None:
    """``designs.json`` carries ``gd_positions`` for an authored type."""
    import json

    pkg = tmp_path / "pkg"
    reg = DesignRegistry()
    src = DesignSource(design=DESIGN, alias=reg.alias(DESIGN),
                       hgc_path=tmp_path / "unused.HGC")
    manifest = write_designs_manifest(pkg, [src], reg)
    rec = json.loads(manifest.read_text(encoding="utf-8"))["designs"][0]
    assert rec["type_id"] == "P5547Z1G08N20"
    assert rec["gd_positions"] == "1:1;4:1;6:4"
    assert rec["gd_u_enr"] == 4.0


# --------------------------------------------------------------------------- #
# gd_positions index convention (review fix)
# --------------------------------------------------------------------------- #
def test_gd_positions_are_zero_indexed_and_canonical() -> None:
    """``2:0`` is a legal octant cell — the manifest guard used to reject it.

    ``spec.parse_gd_positions`` (and the deck's own assembly triangle, whose row 0
    is the assembly centre) is 0-indexed, and the surrogate's layout catalogue is
    full of layouts that start at column 0.  Such a design authored, passed
    compliance and would have run DeCART, only to die at manifest time.
    """
    import pytest

    from lpopt.design.package import (
        DesignManifestError,
        normalize_gd_positions,
        parse_gd_positions as pkg_parse,
    )
    from lpopt.design.spec import parse_gd_positions as spec_parse

    layout = "2:0;2:2;5:2;5:5"
    assert normalize_gd_positions(layout) == layout
    assert pkg_parse(layout) == ((2, 0), (2, 2), (5, 2), (5, 5)) == spec_parse(layout)
    # one canonical spelling: sorted, same as spec's
    assert normalize_gd_positions("5:5;2:0;5:2;2:2") == layout
    assert normalize_gd_positions([(5, 5), (2, 0), (5, 2), (2, 2)]) == layout
    # still outside the triangle / off the map / duplicated -> rejected
    for bad in ("1:5", "8:0", "-1:0", "2:0;2:0"):
        with pytest.raises(DesignManifestError):
            normalize_gd_positions(bad)


def test_manifest_accepts_a_column_zero_layout(tmp_path: Path) -> None:
    import json

    design = FuelDesign(5.5, 4.70, "z1", 8.0, 24, gd_positions="2:0;2:2;5:2;5:5")
    reg = DesignRegistry()
    src = DesignSource(design=design, alias=reg.alias(design),
                       hgc_path=tmp_path / "unused.HGC")
    manifest = write_designs_manifest(tmp_path / "pkg", [src], reg)
    rec = json.loads(manifest.read_text(encoding="utf-8"))["designs"][0]
    assert rec["gd_positions"] == "2:0;2:2;5:2;5:5"
