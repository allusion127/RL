"""Gd pin-map AUTHORING + DeCART preflight + xenon pinning (tasks #1, #2, #3).

The production chain used to edit three numeric tokens and nothing else
(``lattice.edit_dec_text``), so an on-demand design was stuck with whatever Gd
layout the frozen ``0_APR1400`` template for its ``(gd_wt, n_gd, z)`` happened to
carry.  These tests pin the promoted authoring path:

* ``author_gd_layout`` moves ONLY the Gd cell ids, under census / guide-tube /
  edge-zoning guards, and leaves the rest of the deck byte-identical;
* ``octant_to_full`` expands the 8-row octant triangle into the flat 16x16 map
  ``compliance.is_octant_symmetric`` consumes;
* the pre-existing ``edit_dec_text`` behaviour is unchanged when no layout is
  given (byte-identical decks — the whole 37-type library depends on it);
* ``n_gd = 20`` resolves a subtree instead of dying (the slice-Z value);
* the exe / XS-library SHA-256 preflight and the ``nxfile`` rewrite fail fast;
* the authored deck keeps ``xenon TR``.

The template fixture below is SYNTHESIZED (not a copy of a vendor deck): a
minimal but structurally faithful DeCART2D input with the real IGD_20 ``8_20_z1``
octant triangle, whose frozen Gd layout is ``2:2;5:2;6:4`` (n_gd 20).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lpopt.data.compliance import is_octant_symmetric
from lpopt.design.lattice import (
    DECART_SERIAL_EXE_SHA256,
    GD_CELL_ID,
    GUIDE_TUBE_CELL_IDS,
    GUIDE_TUBE_OCTANT,
    LatticeError,
    OMP_RUNTIME_DLL,
    XENON_MODE,
    assert_xenon_mode,
    author_gd_layout,
    author_template,
    authored_deck_name,
    edit_dec_text,
    nxfile_of,
    octant_census,
    octant_to_full,
    parse_octant_triangle,
    preflight_decart,
    resolve_decart_exe,
    resolve_template,
    rewrite_nxfile,
    sha256_file,
    template_dir,
    template_subtree,
    verify_sha256,
    write_authored_deck,
    xenon_mode,
)
from lpopt.design.spec import (
    DesignRegistry,
    FuelDesign,
    format_gd_positions,
    gd_multiplicity,
    parse_gd_positions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _package_designs() -> Path | None:
    """The paramA ``designs.json`` to round-trip, or ``None`` to skip.

    ``LPOPT_PARAMA_DESIGNS`` points the round-trip at any manifest copy.  The
    production package is used only when it is ALREADY staged here: this suite
    must never be a reason to create ``data/design/package/`` on a host, because a
    directory holding a lone ``designs.json`` advertises a registry-less paramA
    package to ``search/verify.py``.
    """
    import os

    env = os.environ.get("LPOPT_PARAMA_DESIGNS")
    if env:
        return Path(env)
    pkg = REPO_ROOT / "data" / "design" / "package"
    manifest = pkg / "designs.json"
    if manifest.is_file() and (pkg / "registry.json").is_file():
        return manifest
    return None

#: The frozen IGD_20 / 8_20_z1 layout and the open one the slice registers.
FROZEN_N20 = "2:2;5:2;6:4"
OPEN_N20 = "1:1;4:1;6:4"

TEMPLATE = """\
CASEID FA_B03

STATE
 th_cond 0.04338 295.5 15.0

MATERIAL
 mixture UO2    2 10.212   626.85 / 92235 5.8
 mixture UO2_2  2 10.212   626.85 / 92235 5.1
 mixture UO2G    2 9.95    626.85 / 92235 4.0
                                    6408  8.0
 mixture COO    0 -1.0     307.75
GEOM
 npins 16
 pitch 1.285 20.7772
 cellgeo 1  0.4096 0.4178 0.4750 / 5 1 1
 cellgeo 2  0.4096 0.4178 0.4750 / 10 1 1
 cell 1 1 / UO2   AIR CLADF COO
 cell 2 1 / UO2_2 AIR CLADF COO
 cell 3 2 / UO2G AIR CLADF COO
 assembly FA_B03 45 1
9
2  1
1  1  3
1  1  2  6
1  1  2  8  9
1  1  3  2  2  1
1  1  1  1  3  1  1
1  1  1  1  1  1  2  2

 rad_conf 45 CENT 1 1
 FA_B03
 albedo 0.0

OPTION
 ray 0.02 8 4 2
 critical T
 boron 500
 xenon TR
 cmfd T

XSEC
 nxfile D:\\DeCART_MASTER\\LIB\\DML-E71N047G018-PV01-cr08.BIN
 lib_type 0 0
.
"""

#: sha256 of ``edit_dec_text(TEMPLATE, FuelDesign(5.0, 4.25, "z1", 10.0, 20), "FA_T9")``.
#: ``edit_dec_text``'s body is byte-for-byte the one the 37 shipped types were built
#: with; this pins its output so a future refactor of the authoring path cannot
#: drift the legacy (no-layout) deck by a single character.
EDIT_DEC_TEXT_SHA256 = (
    "85db4631b58dd35b92144f115e83b09e855708ff7061c07c9388d46ac539d87c"
)

#: An 8 wt% / n_gd 20 / z1 design realized with the OPEN layout (slice Z1').
DESIGN_OPEN = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions=OPEN_N20)
#: Same directory, different enrichments AND a different layout (coexistence).
DESIGN_FROZEN = FuelDesign(5.0, 4.25, "z1", 8.0, 20, gd_positions=FROZEN_N20)


def _write_frozen_tree(root: Path) -> Path:
    """A stand-in 0_APR1400 tree holding the synthetic 8_20_z1 base deck."""
    d = root / "5.8_5.1" / "FA" / "IGD_20" / "8_20_z1"
    d.mkdir(parents=True, exist_ok=True)
    deck = d / "dec_FA_B03.inp"
    deck.write_text(TEMPLATE, encoding="utf-8")
    return deck


# --------------------------------------------------------------------------- #
# spec: layout parsing / census
# --------------------------------------------------------------------------- #
def test_layout_roundtrip_and_multiplicity() -> None:
    pos = parse_gd_positions(OPEN_N20)
    assert pos == ((1, 1), (4, 1), (6, 4))
    assert format_gd_positions(pos) == OPEN_N20
    # diagonal x4 + off-diagonal x8 == 20
    assert gd_multiplicity(pos) == 20
    assert gd_multiplicity(FROZEN_N20) == 20
    # unsorted / iterable input normalizes to the same canonical spelling
    assert parse_gd_positions([(6, 4), (1, 1), (4, 1)]) == pos


def test_design_rejects_layout_census_mismatch() -> None:
    with pytest.raises(ValueError, match="realizes"):
        FuelDesign(5.5, 4.70, "z1", 8.0, 16, gd_positions=OPEN_N20)   # 20 != 16


def test_design_rejects_out_of_triangle_cell() -> None:
    with pytest.raises(ValueError, match="octant triangle"):
        FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions="1:1;1:4;6:4")


def test_design_without_layout_is_byte_identical() -> None:
    """The 37 shipped types carry no layout: id, key and record must not move."""
    d = FuelDesign(5.8, 5.1, "z1", 6.0, 12)
    assert d.gd_positions is None
    assert d.type_id == "P5851Z1G06N12"
    assert d.key == (58, 51, "z1", 6, 12)             # historical 5-tuple
    assert d.layout_tag is None and d.type_id_tagged == d.type_id
    assert "gd_positions" not in d.as_dict()


def test_layout_extends_key_and_record() -> None:
    a = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions=OPEN_N20)
    b = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions=FROZEN_N20)
    assert a.type_id == b.type_id == "P5547Z1G08N20"   # slice-Z registered id
    assert a.key != b.key                              # but NOT the same lattice
    assert a.as_dict()["gd_positions"] == OPEN_N20
    assert a.layout_tag and a.layout_tag != b.layout_tag
    assert a.type_id_tagged.startswith(a.type_id + "L")


def test_packaged_37_type_ids_roundtrip_byte_identical() -> None:
    """Regression: every shipped designs.json record reproduces its own id/record."""
    manifest = _package_designs()
    if manifest is None or not manifest.is_file():
        pytest.skip("paramA package not staged on this host "
                    "(set LPOPT_PARAMA_DESIGNS to point at a manifest copy)")
    import json

    records = json.loads(manifest.read_text(encoding="utf-8"))["designs"]
    assert len(records) == 37
    for rec in records:
        d = FuelDesign.from_dict(rec)
        assert d.type_id == rec["type_id"]
        got = d.as_dict()
        for k, v in got.items():
            assert rec[k] == v, f"{rec['type_id']}: {k} {rec.get(k)!r} != {v!r}"
        # the four lat1600 types keep their layout string verbatim
        assert got.get("gd_positions") == rec.get("gd_positions")


# --------------------------------------------------------------------------- #
# task #1 — author_gd_layout guards
# --------------------------------------------------------------------------- #
def test_author_moves_only_gd_cells() -> None:
    out = author_gd_layout(TEMPLATE, OPEN_N20, 20)
    _l0, rows0, _i0 = parse_octant_triangle(TEMPLATE)
    _l1, rows1, _i1 = parse_octant_triangle(out)
    assert octant_census(rows0, GD_CELL_ID) == {(2, 2), (5, 2), (6, 4)}
    assert octant_census(rows1, GD_CELL_ID) == {(1, 1), (4, 1), (6, 4)}
    # guide tubes and edge zoning are untouched
    assert octant_census(rows1, GUIDE_TUBE_CELL_IDS) == set(GUIDE_TUBE_OCTANT)
    assert octant_census(rows1, 2) == octant_census(rows0, 2)
    # every line outside the triangle is byte-identical
    a, b = TEMPLATE.splitlines(), out.splitlines()
    assert len(a) == len(b)
    moved = {i for i in range(len(a)) if a[i] != b[i]}
    assert moved <= set(_i0), f"authoring touched non-triangle lines {moved - set(_i0)}"


def test_author_census_matches_n_gd() -> None:
    for layout, n in ((OPEN_N20, 20), (FROZEN_N20, 20)):
        _l, rows, _i = parse_octant_triangle(author_gd_layout(TEMPLATE, layout, n))
        assert gd_multiplicity(octant_census(rows, GD_CELL_ID)) == n


def test_author_rejects_wrong_census() -> None:
    with pytest.raises(LatticeError, match="realizes"):
        author_gd_layout(TEMPLATE, OPEN_N20, 16)


def test_author_refuses_guide_tube_and_zoning_cells() -> None:
    #: (4,3) is a guide tube; (7,6) carries the edge-zoning id 2 in this template.
    with pytest.raises(LatticeError, match="guide tube or edge zoning"):
        author_gd_layout(TEMPLATE, "1:1;4:3;6:4", 20)
    with pytest.raises(LatticeError, match="guide tube or edge zoning"):
        author_gd_layout(TEMPLATE, "1:1;4:1;7:6", 20)


def test_author_refuses_template_with_moved_guide_tubes() -> None:
    broken = TEMPLATE.replace("1  1  2  6\n", "1  1  6  2\n")
    with pytest.raises(LatticeError, match="guide tubes at"):
        author_gd_layout(broken, OPEN_N20, 20)


def test_author_output_is_ascii_without_bom() -> None:
    out = author_gd_layout(TEMPLATE, OPEN_N20, 20)
    raw = out.encode("ascii")                     # raises if any non-ASCII slipped in
    assert not raw.startswith(b"\xef\xbb\xbf")


def test_author_is_deterministic_and_idempotent() -> None:
    once = author_gd_layout(TEMPLATE, OPEN_N20, 20)
    assert once == author_gd_layout(TEMPLATE, OPEN_N20, 20)
    assert once == author_gd_layout(once, OPEN_N20, 20)


def test_gd_pins_keep_chebyshev_separation() -> None:
    """No two Gd pins are 8-neighbours in the expanded full map (thermal shadowing)."""
    _l, rows, _i = parse_octant_triangle(author_gd_layout(TEMPLATE, OPEN_N20, 20))
    flat = octant_to_full(rows)
    gd = {(r, c) for r in range(16) for c in range(16) if flat[r * 16 + c] == GD_CELL_ID}
    assert len(gd) == 20
    for (r, c) in gd:
        for (r2, c2) in gd:
            if (r, c) == (r2, c2):
                continue
            assert max(abs(r - r2), abs(c - c2)) >= 2, f"Gd pins adjacent at {(r,c)}/{(r2,c2)}"


# --------------------------------------------------------------------------- #
# task #1 — octant_to_full
# --------------------------------------------------------------------------- #
def test_octant_to_full_is_octant_symmetric() -> None:
    _l, rows, _i = parse_octant_triangle(author_gd_layout(TEMPLATE, OPEN_N20, 20))
    flat = octant_to_full(rows)
    assert len(flat) == 256
    assert is_octant_symmetric(flat, n=16, tol=1e-3)


def test_octant_to_full_reproduces_the_ce16_non_fuel_census() -> None:
    """5 tubes x 4 cells = 20 non-fuel positions, i.e. 236 fuel pins."""
    _l, rows, _i = parse_octant_triangle(TEMPLATE)
    flat = octant_to_full(rows)
    assert sum(1 for v in flat if v in GUIDE_TUBE_CELL_IDS) == 20
    assert sum(1 for v in flat if v not in GUIDE_TUBE_CELL_IDS) == 236
    # octant row 0 is the assembly CENTRE (surrogate features.py:88-98 convention)
    assert {flat[7 * 16 + 7], flat[7 * 16 + 8],
            flat[8 * 16 + 7], flat[8 * 16 + 8]} <= GUIDE_TUBE_CELL_IDS


def test_octant_to_full_rejects_a_wrong_triangle() -> None:
    with pytest.raises(LatticeError, match="8-row triangle"):
        octant_to_full([[1], [1, 1]])


# --------------------------------------------------------------------------- #
# task #1 — edit_dec_text is unchanged when no layout is authored
# --------------------------------------------------------------------------- #
def test_edit_dec_text_leaves_the_pin_map_byte_identical() -> None:
    d = FuelDesign(5.0, 4.25, "z1", 10.0, 20)
    out = edit_dec_text(TEMPLATE, d, "FA_T9")
    _l0, rows0, idx0 = parse_octant_triangle(TEMPLATE)
    _l1, rows1, idx1 = parse_octant_triangle(out)
    assert rows1 == rows0 and idx1 == idx0
    a, b = TEMPLATE.splitlines(), out.splitlines()
    changed = {i for i in range(len(a)) if a[i] != b[i]}
    for i in changed:
        assert i not in idx0                        # never a triangle line
    assert " 92235 5.0" in out and " 92235 4.25" in out and " 6408  10.0" in out
    assert "FA_T9" in out and "FA_B03" not in out


def test_edit_dec_text_output_is_frozen() -> None:
    """A pinned digest of the legacy edit: no refactor may drift the 37-type path."""
    d = FuelDesign(5.0, 4.25, "z1", 10.0, 20)
    once = edit_dec_text(TEMPLATE, d, "FA_T9")
    assert hashlib.sha256(once.encode("utf-8")).hexdigest() == EDIT_DEC_TEXT_SHA256
    # the AUTHORED variant of the same design differs ONLY inside the triangle
    authored = edit_dec_text(author_gd_layout(TEMPLATE, OPEN_N20, 20), d, "FA_T9")
    _l, _rows, idx = parse_octant_triangle(once)
    a, b = once.splitlines(), authored.splitlines()
    assert {i for i in range(len(a)) if a[i] != b[i]} <= set(idx)


# --------------------------------------------------------------------------- #
# task #1 — template resolution: n_gd 20 works, two layouts coexist
# --------------------------------------------------------------------------- #
def test_template_subtree_covers_every_supported_n_gd() -> None:
    assert template_subtree(20) == "5.8_5.1/FA"       # the slice-Z value
    for n in (12, 16, 20):
        assert template_subtree(n) == "5.8_5.1/FA"
    assert template_subtree(24) == "260624/FA"
    with pytest.raises(LatticeError, match="no template subtree rule"):
        template_subtree(28)


def test_authored_deck_name_is_the_registered_spelling() -> None:
    assert authored_deck_name(DESIGN_OPEN) == "dec_FA_P5547Z1G08N20.inp"


def test_two_layouts_coexist_in_one_template_dir(tmp_path: Path) -> None:
    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    troot = tmp_path / "templates"
    p_open = author_template(DESIGN_OPEN, apr, troot)
    p_frozen = author_template(DESIGN_FROZEN, apr, troot)
    assert p_open.parent == p_frozen.parent == template_dir(DESIGN_OPEN, troot)
    assert p_open != p_frozen
    # each design resolves to ITS OWN deck, not sorted(glob)[0]
    assert resolve_template(DESIGN_OPEN, apr, template_root=troot) == p_open
    assert resolve_template(DESIGN_FROZEN, apr, template_root=troot) == p_frozen
    for path, layout in ((p_open, OPEN_N20), (p_frozen, FROZEN_N20)):
        _l, rows, _i = parse_octant_triangle(path.read_text(encoding="utf-8"))
        assert format_gd_positions(octant_census(rows, GD_CELL_ID)) == layout


def test_resolve_template_default_path_is_unchanged(tmp_path: Path) -> None:
    apr = tmp_path / "apr"
    frozen = _write_frozen_tree(apr)
    plain = FuelDesign(5.0, 4.25, "z1", 8.0, 20)      # no layout, no override
    assert resolve_template(plain, apr) == frozen
    assert resolve_template(DESIGN_OPEN, apr) == frozen


def test_author_template_needs_a_layout(tmp_path: Path) -> None:
    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    with pytest.raises(LatticeError, match="names no gd_positions"):
        author_template(FuelDesign(5.0, 4.25, "z1", 8.0, 20), apr, tmp_path / "t")


# --------------------------------------------------------------------------- #
# task #1 — registry guard
# --------------------------------------------------------------------------- #
def test_registry_refuses_same_type_id_with_a_different_layout() -> None:
    reg = DesignRegistry()
    assert reg.alias(DESIGN_OPEN) == "P0"
    clash = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions=FROZEN_N20)
    assert clash.type_id == DESIGN_OPEN.type_id
    with pytest.raises(ValueError, match="already bound to a different design"):
        reg.alias(clash)


def test_registry_refuses_same_type_id_with_a_different_e2() -> None:
    """type_id quantizes e2 to 0.1 w/o: 4.6750 and 4.70 both spell '47'."""
    reg = DesignRegistry()
    reg.alias(FuelDesign(5.5, 4.70, "z1", 8.0, 20))
    with pytest.raises(ValueError, match="differs on \\['e2'\\]"):
        reg.alias(FuelDesign(5.5, 4.675, "z1", 8.0, 20))


def test_registry_reassignment_is_stable_and_persists(tmp_path: Path) -> None:
    reg = DesignRegistry()
    a = reg.alias(DESIGN_OPEN)
    assert reg.alias(DESIGN_OPEN) == a                # idempotent
    path = tmp_path / "registry.json"
    reg.save(path)
    back = DesignRegistry.load(path)
    assert back.alias(DESIGN_OPEN) == a
    assert back.design_of(DESIGN_OPEN.type_id)["gd_positions"] == OPEN_N20
    with pytest.raises(ValueError, match="already bound"):
        back.alias(FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions=FROZEN_N20))


def test_registry_without_layouts_writes_the_historical_document(tmp_path: Path) -> None:
    """Byte-identical persistence for every pre-existing path."""
    reg = DesignRegistry()
    for d in (FuelDesign(5.8, 5.1, "z1", 6.0, 12), FuelDesign(5.8, 5.1, "z1", 8.0, 16)):
        reg.alias(d)
    path = tmp_path / "registry.json"
    reg.save(path)
    body = path.read_text(encoding="utf-8")
    assert '"designs"' not in body
    assert body.startswith('{\n  "aliases": {')


# --------------------------------------------------------------------------- #
# task #3 — xenon TR
# --------------------------------------------------------------------------- #
def test_authored_deck_keeps_the_xenon_card() -> None:
    assert xenon_mode(TEMPLATE) == XENON_MODE == "TR"
    out = author_gd_layout(TEMPLATE, OPEN_N20, 20)
    assert xenon_mode(out) == "TR"
    assert assert_xenon_mode(out) == "TR"
    xen = [ln for ln in out.splitlines() if ln.strip().startswith("xenon")]
    assert xen == [ln for ln in TEMPLATE.splitlines() if ln.strip().startswith("xenon")]


def test_xenon_guard_rejects_eq_and_absent_cards() -> None:
    with pytest.raises(LatticeError, match="mixing Xe treatments"):
        assert_xenon_mode(TEMPLATE.replace(" xenon TR", " xenon EQ"))
    with pytest.raises(LatticeError, match="no 'xenon' OPTION card"):
        assert_xenon_mode(TEMPLATE.replace(" xenon TR\n", ""))


def test_author_template_asserts_xenon(tmp_path: Path) -> None:
    apr = tmp_path / "apr"
    deck = _write_frozen_tree(apr)
    deck.write_text(TEMPLATE.replace(" xenon TR", " xenon EQ"), encoding="utf-8")
    with pytest.raises(LatticeError, match="xenon"):
        author_template(DESIGN_OPEN, apr, tmp_path / "t")


# --------------------------------------------------------------------------- #
# task #2 — SHA-256 preflight / nxfile rewrite / serial fallback
# --------------------------------------------------------------------------- #
def _fake(path: Path, body: bytes = b"decart") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_sha256_and_verify(tmp_path: Path) -> None:
    p = _fake(tmp_path / "bin" / "decart2d1.1m5.exe")
    digest = sha256_file(p)
    assert digest == hashlib.sha256(b"decart").hexdigest()
    assert verify_sha256(p, digest.upper()) == digest      # case-insensitive
    with pytest.raises(LatticeError, match="SHA-256 mismatch"):
        verify_sha256(p, DECART_SERIAL_EXE_SHA256)
    with pytest.raises(LatticeError, match="not found"):
        verify_sha256(tmp_path / "nope.exe", digest)


def test_nxfile_rewrite_and_fail_fast(tmp_path: Path) -> None:
    assert nxfile_of(TEMPLATE).endswith("DML-E71N047G018-PV01-cr08.BIN")
    out = rewrite_nxfile(TEMPLATE, r"C:\lib\XS.BIN")
    assert nxfile_of(out) == r"C:\lib\XS.BIN"
    assert " nxfile " in out                                # whitespace preserved
    assert len(out.splitlines()) == len(TEMPLATE.splitlines())
    with pytest.raises(LatticeError, match="no 'nxfile"):
        rewrite_nxfile(TEMPLATE.replace(" nxfile ", " nofile "), r"C:\lib\XS.BIN")


def test_preflight_falls_back_to_serial_without_the_omp_dll(tmp_path: Path) -> None:
    omp = _fake(tmp_path / "BIN" / "decart2d1.1m5omp.exe")
    serial = _fake(tmp_path / "BIN" / "decart2d1.1m5.exe", b"serial")
    xs = _fake(tmp_path / "LIB" / "XS.BIN", b"xs")
    assert resolve_decart_exe(omp, serial_exe=serial) == serial
    pf = preflight_decart(TEMPLATE, exe=omp, xs_lib=xs, serial_exe=serial)
    assert pf.exe == serial and pf.serial is True
    assert pf.env["OMP_NUM_THREADS"] == "1"
    assert nxfile_of(pf.deck_text) == str(xs)
    # with the dll present the omp build is kept
    _fake(omp.parent / OMP_RUNTIME_DLL, b"dll")
    assert resolve_decart_exe(omp, serial_exe=serial) == omp
    assert preflight_decart(TEMPLATE, exe=omp, xs_lib=xs, serial_exe=serial).serial is False


def test_preflight_raises_on_hash_mismatch_and_missing_library(tmp_path: Path) -> None:
    serial = _fake(tmp_path / "BIN" / "decart2d1.1m5.exe", b"serial")
    xs = _fake(tmp_path / "LIB" / "XS.BIN", b"xs")
    with pytest.raises(LatticeError, match="DeCART exe SHA-256 mismatch"):
        preflight_decart(TEMPLATE, exe=serial, xs_lib=xs, serial_exe=serial,
                         exe_sha256=DECART_SERIAL_EXE_SHA256)
    with pytest.raises(LatticeError, match="XS library SHA-256 mismatch"):
        preflight_decart(TEMPLATE, exe=serial, xs_lib=xs, serial_exe=serial,
                         xs_sha256=DECART_SERIAL_EXE_SHA256)
    with pytest.raises(LatticeError, match="cross-section library not found"):
        preflight_decart(TEMPLATE, exe=serial, xs_lib=tmp_path / "gone.BIN",
                         serial_exe=serial)
    with pytest.raises(LatticeError, match="no runnable DeCART executable"):
        preflight_decart(TEMPLATE, exe=tmp_path / "gone.exe", xs_lib=xs,
                         serial_exe=tmp_path / "gone2.exe")
    # matching digests pass and are reported back
    pf = preflight_decart(TEMPLATE, exe=serial, xs_lib=xs, serial_exe=serial,
                          exe_sha256=sha256_file(serial), xs_sha256=sha256_file(xs))
    assert pf.exe_sha256 == sha256_file(serial) and pf.xs_sha256 == sha256_file(xs)


# --------------------------------------------------------------------------- #
# end to end: the authored production deck
# --------------------------------------------------------------------------- #
def test_write_authored_deck_carries_the_open_layout(tmp_path: Path) -> None:
    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    reg = DesignRegistry()
    deck = write_authored_deck(DESIGN_OPEN, tmp_path / "work", reg, apr,
                               tmp_path / "templates")
    text = deck.read_text(encoding="utf-8")
    assert deck.name == f"dec_FA_{reg.alias(DESIGN_OPEN)}.inp"
    _l, rows, _i = parse_octant_triangle(text)
    assert format_gd_positions(octant_census(rows, GD_CELL_ID)) == OPEN_N20
    assert is_octant_symmetric(octant_to_full(rows), n=16)
    assert " 92235 5.5" in text and " 92235 4.7" in text and " 6408  8.0" in text
    assert xenon_mode(text) == "TR"


# --------------------------------------------------------------------------- #
# review fixes
# --------------------------------------------------------------------------- #
def test_author_rejects_adjacent_gd_pins() -> None:
    """Chebyshev >= 2 is a GUARD, not just a property of the fixture layout.

    ``4:1`` and ``5:1`` are two octant rows apart inside the triangle but land
    side by side in the expanded 16x16 map, which is where thermal shadowing
    actually happens.
    """
    with pytest.raises(LatticeError, match="adjacent in the full map"):
        author_gd_layout(TEMPLATE, "1:1;4:1;5:1", 20)
    # ... and the accepted layouts still author
    assert author_gd_layout(TEMPLATE, OPEN_N20, 20)


def test_write_dec_deck_refuses_a_frozen_template_for_a_named_layout(tmp_path: Path) -> None:
    """A layout-bearing design must never be realized on the FROZEN pin map.

    ``edit_dec_text`` cannot move a Gd pin, so before this guard
    ``write_dec_deck``/``run_batch`` silently produced the frozen layout while
    ``designs.json`` recorded the named one.
    """
    from lpopt.design.lattice import write_dec_deck

    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    reg = DesignRegistry()
    with pytest.raises(LatticeError, match="names Gd layout"):
        write_dec_deck(DESIGN_OPEN, tmp_path / "work", reg, apr)
    # with the authored tree threaded through, the realized deck carries the layout
    troot = tmp_path / "templates"
    author_template(DESIGN_OPEN, apr, troot)
    deck = write_dec_deck(DESIGN_OPEN, tmp_path / "work", reg, apr,
                          template_root=troot)
    _l, rows, _i = parse_octant_triangle(deck.read_text(encoding="utf-8"))
    assert format_gd_positions(octant_census(rows, GD_CELL_ID)) == OPEN_N20


def test_write_dec_deck_is_unchanged_for_a_layout_free_design(tmp_path: Path) -> None:
    """The whole 37-type library takes exactly the old path (no layout, no guard)."""
    from lpopt.design.lattice import write_dec_deck

    apr = tmp_path / "apr"
    _write_frozen_tree(apr)
    plain = FuelDesign(5.5, 4.70, "z1", 8.0, 20)
    deck = write_dec_deck(plain, tmp_path / "work", DesignRegistry(), apr)
    _l, rows, _i = parse_octant_triangle(deck.read_text(encoding="utf-8"))
    assert format_gd_positions(octant_census(rows, GD_CELL_ID)) == FROZEN_N20


def test_preflight_pins_digests_by_default_only_for_the_pinned_artefacts(
        tmp_path: Path) -> None:
    """Default digests are live, but apply to the artefacts they actually pin.

    A test double / re-sited install at another path is passed through unchecked
    (there is no pinned digest for it); an explicit digest still checks it.
    """
    serial = _fake(tmp_path / "BIN" / "decart2d1.1m5.exe", b"serial")
    xs = _fake(tmp_path / "LIB" / "XS.BIN", b"xs")
    pf = preflight_decart(TEMPLATE, exe=serial, xs_lib=xs, serial_exe=serial)
    assert pf.exe_sha256 is None and pf.xs_sha256 is None
    with pytest.raises(LatticeError, match="DeCART exe SHA-256 mismatch"):
        preflight_decart(TEMPLATE, exe=serial, xs_lib=xs, serial_exe=serial,
                         exe_sha256=DECART_SERIAL_EXE_SHA256)


def test_preflight_caps_threads_on_both_branches(tmp_path: Path) -> None:
    """``OMP_NUM_THREADS=1`` is unconditional (the omp build is the one that needs it)."""
    omp = _fake(tmp_path / "BIN" / "decart2d1.1m5omp.exe")
    serial = _fake(tmp_path / "BIN" / "decart2d1.1m5.exe", b"serial")
    xs = _fake(tmp_path / "LIB" / "XS.BIN", b"xs")
    _fake(omp.parent / OMP_RUNTIME_DLL, b"dll")
    pf = preflight_decart(TEMPLATE, exe=omp, xs_lib=xs, serial_exe=serial)
    assert pf.serial is False
    assert pf.env["OMP_NUM_THREADS"] == "1"


def test_registry_guard_survives_a_save_load_roundtrip(tmp_path: Path) -> None:
    """R23: the alias guard must hold ACROSS processes, not only within one.

    ``registry.json`` is a bare ``{"aliases": ...}`` document, so ``_designs`` was
    empty after a load and a NEW lattice quantizing onto a shipped ``type_id``
    silently inherited its alias (and would ship its cross sections under that
    MASTER COMP name).  ``load`` now hydrates from the sibling ``designs.json``.
    """
    import json

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    reg = DesignRegistry()
    alias = reg.alias(DESIGN_OPEN)
    reg.save(pkg / "registry.json")
    pkg.joinpath("designs.json").write_text(
        json.dumps({"library_id": "paramA",
                    "designs": [dict(DESIGN_OPEN.as_dict(), alias=alias,
                                     gd_u_enr=4.0, provenance="test")]}),
        encoding="utf-8")

    back = DesignRegistry.load(pkg / "registry.json")
    assert back.alias(DESIGN_OPEN) == alias                 # not over-tight
    other = FuelDesign(5.5, 4.70, "z1", 8.0, 20, gd_positions=FROZEN_N20)
    assert other.type_id == DESIGN_OPEN.type_id             # same quantized id
    with pytest.raises(ValueError, match="already bound to a different design"):
        back.alias(other)


def test_registry_load_without_a_manifest_is_unchanged(tmp_path: Path) -> None:
    reg = DesignRegistry()
    reg.alias(FuelDesign(5.5, 4.70, "z1", 8.0, 20))
    reg.save(tmp_path / "registry.json")
    assert (tmp_path / "registry.json").read_text(encoding="utf-8").count("designs") == 0
    back = DesignRegistry.load(tmp_path / "registry.json")
    assert back.mapping == reg.mapping
    assert back.design_of("P5547Z1G08N20") is None
