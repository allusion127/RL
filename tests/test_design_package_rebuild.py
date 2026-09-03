"""Package rebuild steps: pre-rebuild snapshot, ``designs.json`` schema, and the
``cores/`` regeneration the v1/v2 drafts omitted (tasks #12, #13, #13c).

Three defects are pinned here:

1. **#12** ``build_master_library`` keeps exactly ONE ``.bak`` generation and the
   next rebuild unlinks it (``library.py:99-105``), so a second rebuild destroys
   the only rollback.  A rebuild must therefore be gated on an external snapshot
   whose hashes still describe the package.
2. **#13** ``gd_positions`` was optional (4 of 37 rows carried it) while
   ``type_id`` quantizes ``e2`` to 0.1 w/o — two authored types can collide on
   the id, and only the pin map tells them apart.  New authored rows must carry
   it; old manifests must still load.
3. **#13c** adding types moves ``%GEN_DIM`` (``nbatch``/``ncomp`` = ``3+N``/
   ``5+N``), which makes every ``cores/`` template stale.  ``_resolve_template``
   uses a disk template with no dimension check while ``validate_reload_deck``
   refuses a mismatched one, so an unregenerated package hard-fails every
   existing pair before Popen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.design.coredeck import library_dims
from lpopt.design.library import (
    MANIFEST_NAME,
    SNAPSHOT_MEMBERS,
    SnapshotError,
    build_master_library,
    package_hashes,
    require_snapshot,
    sha256_file,
    snapshot_package,
    verify_snapshot,
)
from lpopt.design.package import (
    DESIGN_OPTIONAL_FIELDS,
    DesignManifestError,
    DesignSource,
    StaleBasesError,
    assemble_package,
    core_template_paths,
    design_record,
    load_designs_manifest,
    normalize_gd_positions,
    parse_gd_positions,
    regenerate_core_templates,
    stale_base_restarts,
    write_core_template,
    write_designs_manifest,
)
from lpopt.design.spec import DesignRegistry, FuelDesign
from lpopt.search.assets import validate_reload_deck

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
DESIGNS = [
    FuelDesign(5.0, 4.25, "z1", 10.0, 16),
    FuelDesign(5.0, 4.25, "z1", 8.0, 24),
]
#: two fake (pair, feed) cells, as ``cores/<folder>/<seed>/MAS_INP_cyNN.inp``
PAIRS = (("T3_T4", 121, "seed", 12), ("T5_T6", 117, "bootstrap", 2))


def _sources(registry: DesignRegistry, tmp_path: Path) -> list[DesignSource]:
    out = []
    for d in DESIGNS:
        alias = registry.alias(d)
        hgc = tmp_path / f"FA_{alias}.HGC"
        hgc.write_text("hgc", encoding="utf-8")
        out.append(DesignSource(design=d, alias=alias, hgc_path=hgc))
    return out


def _regen(pkg: Path, aliases, **kw):
    """``regenerate_core_templates`` with the two OPERATOR acknowledgements set.

    Both are deliberately non-default in production (a forgotten ``synth_root``
    used to purge nothing and report success; a regenerated package over
    old-library ``bases/`` restarts validates and is still wrong), so every test
    that is not about those two gates states them explicitly here.
    """
    kw.setdefault("purge_synth", False)
    kw.setdefault("accept_stale_bases", True)
    return regenerate_core_templates(pkg, aliases, **kw)


def _package(tmp_path: Path, aliases: list[str]) -> Path:
    """A package with ``lib/``, ``bases/``, and two ``cores/`` templates."""
    pkg = tmp_path / "package"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "MAS_XSL").write_text("COMP FA_%s\n" % aliases[0], encoding="utf-8")
    (pkg / "lib" / "MAS_HFF").write_text("hff\n", encoding="utf-8")
    (pkg / "registry.json").write_text(json.dumps({"aliases": {}}), encoding="utf-8")
    (pkg / "designs.json").write_text(json.dumps({"designs": []}), encoding="utf-8")
    for pair, feed, seed_id, cycle in PAIRS:
        base = pkg / "bases" / (pair if feed == 121 else f"{pair}_f{feed}")
        base.mkdir(parents=True, exist_ok=True)
        (base / "MAS_RST.SEED.01").write_bytes(b"rst")
        write_core_template(pkg, pair, feed, aliases, "MAS_RST.SEED.01",
                            seed_id=seed_id, cycle=cycle)
    return pkg


# --------------------------------------------------------------------------- #
# #12 — pre-rebuild snapshot
# --------------------------------------------------------------------------- #
def test_snapshot_captures_every_rebuild_critical_member(tmp_path: Path) -> None:
    pkg = _package(tmp_path, ["P0", "P1"])
    snap = snapshot_package(pkg, tmp_path / "archive", tag="t1")

    assert snap.archive_path.is_file()
    assert snap.manifest_path.name == MANIFEST_NAME
    doc = json.loads(snap.manifest_path.read_text(encoding="utf-8"))
    assert doc["members"] == list(SNAPSHOT_MEMBERS)
    assert doc["archive_sha256"] == sha256_file(snap.archive_path)
    # lib products + both restarts + both templates + the two identity files
    assert "lib/MAS_XSL" in doc["files"] and "lib/MAS_HFF" in doc["files"]
    assert "registry.json" in doc["files"] and "designs.json" in doc["files"]
    assert sum(1 for k in doc["files"] if k.startswith("cores/")) == len(PAIRS)
    assert doc["files"] == package_hashes(pkg)


def test_rebuild_gate_passes_only_while_the_snapshot_matches(tmp_path: Path) -> None:
    """THE #12 regression: the gate must let a rebuild proceed only when the
    snapshot exists AND still hashes the package it is the rollback for."""
    pkg = _package(tmp_path, ["P0", "P1"])
    snap = snapshot_package(pkg, tmp_path / "archive", tag="t1")

    assert verify_snapshot(pkg, snap.snapshot_dir) == []
    require_snapshot(pkg, snap.snapshot_dir)             # no raise

    (pkg / "lib" / "MAS_XSL").write_text("COMP FA_P0\nCOMP FA_P2\n", encoding="utf-8")
    problems = verify_snapshot(pkg, snap.snapshot_dir)
    assert any("lib/MAS_XSL" in p for p in problems), problems
    with pytest.raises(SnapshotError, match="lib/MAS_XSL"):
        require_snapshot(pkg, snap.snapshot_dir)


def test_missing_snapshot_and_missing_file_are_both_refused(tmp_path: Path) -> None:
    pkg = _package(tmp_path, ["P0", "P1"])
    with pytest.raises(SnapshotError):
        require_snapshot(pkg, tmp_path / "archive" / "never-taken")

    snap = snapshot_package(pkg, tmp_path / "archive", tag="t1")
    (pkg / "lib" / "MAS_HFF").unlink()
    assert any("missing from the package" in p
               for p in verify_snapshot(pkg, snap.snapshot_dir))


def test_a_tag_is_an_immutable_rollback_point(tmp_path: Path) -> None:
    pkg = _package(tmp_path, ["P0", "P1"])
    snapshot_package(pkg, tmp_path / "archive", tag="t1")
    with pytest.raises(SnapshotError, match="already exists"):
        snapshot_package(pkg, tmp_path / "archive", tag="t1")


# --------------------------------------------------------------------------- #
# #13 — designs.json schema
# --------------------------------------------------------------------------- #
def test_plain_manifest_is_byte_identical_to_the_pre_schema_writer(tmp_path: Path) -> None:
    """An existing caller (no authored fields) must emit exactly what it did."""
    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    path = write_designs_manifest(tmp_path / "pkg", sources, registry)

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["library_id"] == "paramA"
    assert doc["designs"] == [
        {"type_id": d.type_id, "e1": d.e1, "e2": d.e2,
         "zoning_variant": d.zoning_variant, "gd_wt": d.gd_wt, "n_gd": d.n_gd,
         "alias": s.alias, "gd_u_enr": 4.0}
        for d, s in zip(DESIGNS, sources)
    ]


def test_authored_record_carries_the_new_fields(tmp_path: Path) -> None:
    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    extras = {
        DESIGNS[0].type_id: {
            "gd_positions": [(1, 1), (5, 2), (5, 5)],
            "layout": "8_20_z1",
            "base_template": "templates_lat1600/6_16_z1/dec_FA_A01.inp",
            "xenon_mode": "TR",
            "density": 10.4,
            "provenance": "on_demand_slice_Z",
            "screen_pattern": "PB",
            "screen_k0": 1.234,
            "hgc_sha256": "0" * 64,
        },
        DESIGNS[1].type_id: {"gd_positions": "1:1;4:1;5:5;6:3",
                             "provenance": "on_demand_slice_Z"},
    }
    path = write_designs_manifest(tmp_path / "pkg", sources, registry, extras=extras)
    rows = json.loads(path.read_text(encoding="utf-8"))["designs"]

    assert rows[0]["gd_positions"] == "1:1;5:2;5:5"       # normalized from pairs
    assert rows[0]["screen_pattern"] == "PB"             # z1 screens as PB
    assert rows[0]["xenon_mode"] == "TR"
    assert rows[1]["gd_positions"] == "1:1;4:1;5:5;6:3"
    # e2 keeps its EXACT value even though type_id quantizes it to 0.1 w/o
    assert rows[0]["e2"] == DESIGNS[0].e2
    # unset optional fields are omitted, never emitted as nulls
    assert "deck_sha256" not in rows[1]


def test_authored_row_without_gd_positions_raises(tmp_path: Path) -> None:
    """THE #13 regression: a new authored row MUST carry the pin map."""
    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    extras = {DESIGNS[0].type_id: {"provenance": "on_demand_slice_Z"}}
    with pytest.raises(DesignManifestError, match="gd_positions"):
        write_designs_manifest(tmp_path / "pkg", sources, registry, extras=extras)

    # ... and the explicit gate applies to an unauthored row too.
    with pytest.raises(DesignManifestError, match="gd_positions"):
        design_record(sources[0], require_gd_positions=True)


def test_old_manifest_loads_with_missing_fields_as_none(tmp_path: Path) -> None:
    """Backward compatibility: the 37-row manifest (4 rows with gd_positions)."""
    manifest = tmp_path / "designs.json"
    manifest.write_text(json.dumps({
        "library_id": "paramA",
        "designs": [
            {"type_id": "P5042Z1G10N16", "e1": 5.0, "e2": 4.25,
             "zoning_variant": "z1", "gd_wt": 10.0, "n_gd": 16, "alias": "T3",
             "gd_u_enr": 4.0, "gd_positions": "1:1;5:2;5:5"},
            {"type_id": "P5042Z1G08N24", "e1": 5.0, "e2": 4.25,
             "zoning_variant": "z1", "gd_wt": 8.0, "n_gd": 24, "alias": "T4",
             "gd_u_enr": 4.0},
        ],
    }), encoding="utf-8")

    doc = load_designs_manifest(manifest)
    assert doc["library_id"] == "paramA"
    for row in doc["designs"]:
        for field_name in DESIGN_OPTIONAL_FIELDS:
            assert field_name in row
    assert doc["designs"][0]["gd_positions"] == "1:1;5:2;5:5"
    assert doc["designs"][1]["gd_positions"] is None      # missing -> None, no raise
    assert doc["designs"][1]["screen_pattern"] is None


def test_gd_positions_round_trip_and_validation() -> None:
    """Octant cells are 0-INDEXED, like every other module in the chain
    (``spec.parse_gd_positions``, ``lattice.GUIDE_TUBE_OCTANT``,
    ``screen.GD_CANDIDATES`` — which contains ``(2, 0)``), and the spelling is
    SORTED, so it agrees with ``FuelDesign.gd_layout``/``layout_tag``."""
    assert normalize_gd_positions("2:0;2:2;4:1") == "2:0;2:2;4:1"
    assert normalize_gd_positions([(4, 1), (2, 0), (2, 2)]) == "2:0;2:2;4:1"
    assert normalize_gd_positions("1:1;5:2") == "1:1;5:2"
    assert parse_gd_positions("5:2;1:1") == ((1, 1), (5, 2))
    assert parse_gd_positions(None) == ()
    # col > row is outside the octant triangle; so is row >= 8.  A zero index is
    # NOT an error (that was the 1-indexed defect: it rejected 26 of the 89
    # admissible layouts, P5547Z1G08N20 among them).
    for bad in ("", "1-1", "0:1", [(0, 1)], [(8, 0)], []):
        with pytest.raises(DesignManifestError):
            normalize_gd_positions(bad)


def test_every_admissible_gd_layout_survives_the_manifest(tmp_path: Path) -> None:
    """THE F1 regression: each layout ``author_gd_layout``/``FuelDesign`` accepts
    must round-trip design -> record -> manifest -> ``FuelDesign.from_dict``."""
    from lpopt.design.screen import enumerate_gd_layouts

    by_n = enumerate_gd_layouts()
    layouts = [(n, lay) for n, lays in by_n.items() for lay in lays]
    assert len(layouts) == 89                     # the census, design v2 L339
    with_zero = [lay for _, lay in layouts if any(c == 0 for _, c in lay)]
    assert len(with_zero) == 26                   # the family 1-indexing rejected

    hgc = tmp_path / "x.HGC"
    hgc.write_text("hgc", encoding="utf-8")
    for n_gd, lay in layouts:
        design = FuelDesign(5.5, 4.7, "z1", 8.0, n_gd, gd_positions=lay)
        rec = design_record(
            DesignSource(design=design, alias="A0", hgc_path=hgc),
            extra={"provenance": "slice_Z"})
        assert rec["gd_positions"] == design.gd_layout
        assert parse_gd_positions(rec["gd_positions"]) == design.gd_positions


def test_screen_pattern_must_match_the_zoning_variant(tmp_path: Path) -> None:
    """z1 screens as PB; writing PA is silently a different assembly."""
    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    extra = {"gd_positions": "1:1;5:2;5:5", "screen_pattern": "PA"}
    with pytest.raises(DesignManifestError, match="screen_pattern"):
        design_record(sources[0], extra=extra)
    ok = design_record(sources[0], extra={**extra, "screen_pattern": "PB"})
    assert ok["screen_pattern"] == "PB"


def test_screening_only_row_still_needs_the_pin_map(tmp_path: Path) -> None:
    """The extras prereg line 354 writes (screen_ff/screen_k0/hgc_sha256) are
    authored markers too — the old hand-listed tuple let them through."""
    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    for marker in ("screen_ff", "screen_k0", "screen_crossing_bu",
                   "decart_wall_s", "hgc_sha256"):
        with pytest.raises(DesignManifestError, match="gd_positions"):
            design_record(sources[0], extra={marker: 1.0})


def test_incumbent_lat1600_provenance_round_trips(tmp_path: Path) -> None:
    """The SHIPPED manifest's four authored rows carry lat1600_id/role; the
    closed schema must neither refuse nor silently drop them."""
    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    extras = {DESIGNS[0].type_id: {
        "gd_positions": "1:1;5:2;5:5",
        "lat1600_id": "Y1",
        "lat1600_role": "E1-role reactivity-matched (68 fresh slots), FF_ens 1.1073",
        "provenance": "realize_lat1600 2026-08-11",
    }}
    path = write_designs_manifest(tmp_path / "pkg", sources, registry, extras=extras)
    rows = json.loads(path.read_text(encoding="utf-8"))["designs"]
    assert rows[0]["lat1600_id"] == "Y1"
    assert rows[0]["lat1600_role"].startswith("E1-role")
    # ...and pure provenance does NOT by itself make a row "authored"
    plain = design_record(sources[1], extra={"lat1600_id": "Y2"})
    assert plain["lat1600_id"] == "Y2" and "gd_positions" not in plain
    assert load_designs_manifest(path)["designs"][1]["lat1600_id"] is None


def test_unknown_manifest_field_is_refused(tmp_path: Path) -> None:
    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    with pytest.raises(DesignManifestError, match="unknown"):
        design_record(sources[0], extra={"screen_fff": 1.0})


# --------------------------------------------------------------------------- #
# #13c — cores/ regeneration
# --------------------------------------------------------------------------- #
def test_dry_run_lists_every_stale_template_and_writes_nothing(tmp_path: Path) -> None:
    old = ["P0", "P1"]
    pkg = _package(tmp_path, old)
    before = {p: p.read_bytes() for p in core_template_paths(pkg)}
    assert len(before) == len(PAIRS)

    new = old + ["Z1", "Z2"]
    report = _regen(pkg, new, dry_run=True)

    assert report.dry_run and report.new_dims == library_dims(len(new))
    assert report.new_dims == (3 + len(new), 5 + len(new))
    assert len(report.templates) == len(PAIRS)
    assert all(t.old_dims == library_dims(len(old)) for t in report.templates)
    assert len(report.stale) == len(PAIRS)
    assert sorted(report.folders) == ["T3_T4", "T5_T6_f117"]
    assert not any(t.written for t in report.templates)
    assert {p: p.read_bytes() for p in core_template_paths(pkg)} == before


def test_regeneration_makes_every_existing_pair_validate_again(tmp_path: Path) -> None:
    """THE #13c regression: with N -> N+2 the old templates fail
    ``validate_reload_deck`` on ``%GEN_DIM``, and regeneration fixes exactly
    that while keeping folder, seed id, cycle and restart basename."""
    old = ["P0", "P1"]
    pkg = _package(tmp_path, old)
    new = old + ["Z1", "Z2"]
    new_dims = library_dims(len(new))

    from lpopt.search.assets import DeckValidationError
    for path in core_template_paths(pkg):
        with pytest.raises(DeckValidationError, match="GEN_DIM"):
            validate_reload_deck(path.read_text(encoding="utf-8"),
                                 "MAS_RST.SEED.01", expected_dims=new_dims)

    report = _regen(pkg, new)

    assert all(t.written for t in report.templates)
    paths = core_template_paths(pkg)
    assert len(paths) == len(PAIRS)                      # no template gained/lost
    assert {(p.parent.parent.name, p.parent.name, p.name) for p in paths} == {
        (pair if feed == 121 else f"{pair}_f{feed}", seed, f"MAS_INP_cy{cycle:02d}.inp")
        for pair, feed, seed, cycle in PAIRS
    }
    for path in paths:
        deck = path.read_text(encoding="utf-8")
        validate_reload_deck(deck, "MAS_RST.SEED.01", expected_dims=new_dims)
        assert "MAS_RST.SEED.01" in deck
        for alias in new:
            assert f"FA_{alias}" in deck


def test_gen_dim_for_39_types_is_the_registered_pair(tmp_path: Path) -> None:
    """The registered slice-Z number: 37 + 2 types -> ``10 10 27 42 44``."""
    from lpopt.design.package import _deck_gen_dim

    assert library_dims(39) == (42, 44)
    pkg = _package(tmp_path, [f"A{i}" for i in range(37)])
    roster = [f"A{i}" for i in range(37)] + ["Z1", "Z2"]
    report = _regen(pkg, roster)
    assert report.new_dims == (42, 44)
    for path in core_template_paths(pkg):
        deck = path.read_text(encoding="utf-8")
        assert _deck_gen_dim(deck) == (42, 44)
        assert "10      10      27      42       44" in deck


def test_snapshot_can_be_a_zip(tmp_path: Path) -> None:
    pkg = _package(tmp_path, ["P0", "P1"])
    snap = snapshot_package(pkg, tmp_path / "archive", tag="z1",
                            archive_format="zip")
    assert snap.archive_path.suffix == ".zip"
    assert verify_snapshot(pkg, snap.snapshot_dir) == []


def test_regeneration_purges_the_synth_deck_cache(tmp_path: Path) -> None:
    pkg = _package(tmp_path, ["P0", "P1"])
    synth = tmp_path / "synth_decks"
    (synth / "P0_P1").mkdir(parents=True)
    cached = synth / "P0_P1" / "MAS_INP_cy12.inp"
    cached.write_text("stale deck with the old %GEN_DIM\n", encoding="utf-8")

    dry = _regen(pkg, ["P0", "P1", "Z1", "Z2"], dry_run=True,
                 synth_root=synth, purge_synth=True)
    assert dry.purged == [cached] and cached.is_file()

    report = _regen(pkg, ["P0", "P1", "Z1", "Z2"],
                    synth_root=synth, purge_synth=True)
    assert report.purged == [cached]
    assert synth.is_dir() and not any(synth.rglob("*"))


def test_regeneration_is_idempotent(tmp_path: Path) -> None:
    pkg = _package(tmp_path, ["P0", "P1"])
    new = ["P0", "P1", "Z1", "Z2"]
    _regen(pkg, new)
    first = {p: p.read_bytes() for p in core_template_paths(pkg)}
    report = _regen(pkg, new)
    assert not report.stale
    assert {p: p.read_bytes() for p in core_template_paths(pkg)} == first


def test_empty_roster_is_refused(tmp_path: Path) -> None:
    pkg = _package(tmp_path, ["P0", "P1"])
    with pytest.raises(ValueError, match="non-empty roster"):
        _regen(pkg, [])


# --------------------------------------------------------------------------- #
# #13c item 3 — bases/ restarts are keyed to the OLD library
# --------------------------------------------------------------------------- #
def test_regeneration_refuses_to_leave_stale_bases_behind(tmp_path: Path) -> None:
    """Regenerating cores/ makes every template pass ``validate_reload_deck``
    again while it still reads a restart produced against the OLD library — the
    one hard stop that would have caught the stale package.  So the restarts are
    enumerated and the run refuses until the operator acknowledges them."""
    pkg = _package(tmp_path, ["P0", "P1"])
    new = ["P0", "P1", "Z1", "Z2"]
    before = {p: p.read_bytes() for p in core_template_paths(pkg)}

    assert len(stale_base_restarts(pkg)) == len(PAIRS)
    with pytest.raises(StaleBasesError, match="re-bootstrap") as excinfo:
        regenerate_core_templates(pkg, new, purge_synth=False)
    # nothing was written: the refusal happens before the first template
    assert {p: p.read_bytes() for p in core_template_paths(pkg)} == before
    report = excinfo.value.report
    assert sorted(report.stale_base_folders) == ["T3_T4", "T5_T6_f117"]
    assert len(report.stale_bases) == len(PAIRS)

    # a dry run always LISTS them (that is the review affordance) ...
    dry = regenerate_core_templates(pkg, new, dry_run=True, purge_synth=False)
    assert len(dry.stale_bases) == len(PAIRS)
    assert dry.as_dict()["stale_base_folders"] == ["T3_T4", "T5_T6_f117"]

    # ... and with the acknowledgement the regeneration proceeds and still says
    # which folders must be re-bootstrapped.
    done = regenerate_core_templates(pkg, new, purge_synth=False,
                                     accept_stale_bases=True)
    assert all(t.written for t in done.templates)
    assert len(done.stale_bases) == len(PAIRS)

    # a package with no bases/ needs no acknowledgement
    for p in stale_base_restarts(pkg):
        p.unlink()
    clean = regenerate_core_templates(pkg, new, purge_synth=False)
    assert clean.stale_bases == [] and all(t.written for t in clean.templates)


def test_purge_synth_without_a_root_is_refused(tmp_path: Path) -> None:
    """The old default purged NOTHING and reported success (``purged == []``),
    which is indistinguishable from 'the cache was already empty'."""
    pkg = _package(tmp_path, ["P0", "P1"])
    with pytest.raises(ValueError, match="synth_root"):
        regenerate_core_templates(pkg, ["P0", "P1", "Z1", "Z2"],
                                  accept_stale_bases=True)


# --------------------------------------------------------------------------- #
# #13c — the regeneration against the SHIPPED template (not one we wrote)
# --------------------------------------------------------------------------- #
FROZEN_TEMPLATE = (Path(__file__).parent / "fixtures"
                   / "core_template_T5_T6_bootstrap_cy02.inp")
#: the 37-alias roster the shipped template carries, in %LPD_HFF order
FROZEN_ALIASES = [
    "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
    "Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9",
    "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "T0", "T1", "T2", "T3", "T4", "T5", "T6",
]


def _cards(deck: str) -> list[tuple[str, tuple[str, ...]]]:
    """The deck as ``[(card, body lines), ...]`` in file order.

    Block-wise, not line-index-wise: a roster that grows by two entries shifts
    every later line, so an index diff would call the whole tail 'changed'.
    """
    out: list[tuple[str, list[str]]] = []
    for line in deck.splitlines():
        if line.startswith("%"):
            out.append((line.split()[0], []))
        elif out:
            out[-1][1].append(line)
    return [(card, tuple(body)) for card, body in out]


def _changed_cards(old: str, new: str) -> set[str]:
    """Cards whose body differs (or whose card sequence differs at all)."""
    a, b = _cards(old), _cards(new)
    assert [c for c, _ in a] == [c for c, _ in b], "the card SEQUENCE changed"
    return {c for (c, ba), (_, bb) in zip(a, b) if ba != bb}


def test_shipped_template_regenerates_to_the_39_type_dims(tmp_path: Path) -> None:
    """THE #13c evidence, on the SHIPPED deck rather than on one this module's
    own writer produced: ``cores/T5_T6/bootstrap/MAS_INP_cy02.inp`` as it ships
    (%GEN_DIM ``40 42`` = 37 types) fails ``validate_reload_deck`` at the
    slice-Z dims, regenerates to ``42 44``, and NOTHING outside the four roster
    cards changes."""
    frozen = FROZEN_TEMPLATE.read_text(encoding="utf-8")
    assert len(FROZEN_ALIASES) == 37, FROZEN_ALIASES
    assert "10      10      27      40       42" in frozen
    restart = "MAS_RST.APRQ_01_0981.02"
    assert restart in frozen

    pkg = tmp_path / "package"
    seed = pkg / "cores" / "T5_T6" / "bootstrap"
    seed.mkdir(parents=True)
    path = seed / "MAS_INP_cy02.inp"
    path.write_text(frozen, encoding="utf-8")

    from lpopt.search.assets import DeckValidationError

    # it is VALID at its own dims ...
    validate_reload_deck(frozen, restart, expected_dims=library_dims(37))
    # ... and hard-fails at the 39-type dims the roster change forces
    with pytest.raises(DeckValidationError, match="GEN_DIM"):
        validate_reload_deck(frozen, restart, expected_dims=(42, 44))

    roster = FROZEN_ALIASES + ["Z1", "Z2"]
    report = regenerate_core_templates(pkg, roster, purge_synth=False)
    assert report.new_dims == (42, 44)
    assert report.stale_bases == []
    assert [t.restart_basename for t in report.templates] == [restart]
    assert [t.old_dims for t in report.templates] == [(40, 42)]

    after = path.read_text(encoding="utf-8")
    validate_reload_deck(after, restart, expected_dims=(42, 44))
    assert "10      10      27      42       44" in after
    assert _changed_cards(frozen, after) <= {
        "%GEN_DIM", "%LPD_B&C", "%LPD_C&X", "%LPD_HFF"}


def test_regenerating_the_shipped_template_at_its_own_roster_is_a_no_op() -> None:
    """The regeneration writer reproduces the shipped deck: with the SAME 37
    aliases the rebuilt text is the frozen text (modulo line endings)."""
    from lpopt.design.coredeck import build_reload_deck

    frozen = FROZEN_TEMPLATE.read_text(encoding="utf-8")
    rebuilt = build_reload_deck(FROZEN_ALIASES, "MAS_RST.APRQ_01_0981.02", 2)
    assert rebuilt.replace("\r\n", "\n") == frozen.replace("\r\n", "\n")


# --------------------------------------------------------------------------- #
# #12 — the gate is ON the rebuild path, and the archive is checked
# --------------------------------------------------------------------------- #
def test_rebuild_is_gated_before_the_only_bak_is_destroyed(tmp_path: Path) -> None:
    """``build_master_library`` keeps ONE ``.bak`` and the next rebuild unlinks
    it, so the snapshot check must fire BEFORE the rename."""
    pkg = _package(tmp_path, ["P0", "P1"])
    snap = snapshot_package(pkg, tmp_path / "archive", tag="t1")
    (pkg / "lib" / "MAS_XSL").write_text("COMP FA_P0\nCOMP FA_P9\n", encoding="utf-8")
    live = (pkg / "lib" / "MAS_XSL").read_bytes()

    with pytest.raises(SnapshotError):
        build_master_library([], pkg / "lib", mas_ref=tmp_path / "MAS_REF",
                             prolog_exe=tmp_path / "prolog41m4.exe",
                             totalbatcher_exe=tmp_path / "TotalBatcher4.exe",
                             snapshot_dir=snap.snapshot_dir)
    assert (pkg / "lib" / "MAS_XSL").read_bytes() == live
    assert not (pkg / "lib" / "MAS_XSL.bak").exists()


def test_assemble_package_checks_the_snapshot_before_rewriting_the_manifest(
        tmp_path: Path) -> None:
    """designs.json/registry.json are themselves snapshot members, so a gate
    placed after the manifest rewrite could never pass."""
    pkg = _package(tmp_path, ["P0", "P1"])
    snap = snapshot_package(pkg, tmp_path / "archive", tag="t1")
    (pkg / "lib" / "MAS_HFF").write_text("mutated\n", encoding="utf-8")
    manifest_before = (pkg / "designs.json").read_bytes()

    registry = DesignRegistry()
    sources = _sources(registry, tmp_path)
    with pytest.raises(SnapshotError):
        assemble_package(pkg, sources, registry, tmp_path / "apr1400",
                         snapshot_dir=snap.snapshot_dir)
    assert (pkg / "designs.json").read_bytes() == manifest_before


def test_snapshot_proves_the_archive_holds_the_recorded_files(tmp_path: Path) -> None:
    """The hashes are taken before staging, so "the manifest verifies" must not
    be the only claim: the archive is read back and checked for every file."""
    import tarfile

    pkg = _package(tmp_path, ["P0", "P1"])
    snap = snapshot_package(pkg, tmp_path / "archive", tag="t1")
    doc = json.loads(snap.manifest_path.read_text(encoding="utf-8"))
    assert set(doc["files"]) <= set(doc["archive_members"])
    assert verify_snapshot(pkg, snap.snapshot_dir) == []

    # an archive that no longer holds a recorded member is refused, even though
    # its own sha256 is re-recorded (i.e. "the archive is intact" is not enough)
    stripped = snap.archive_path
    extract = snap.snapshot_dir / "_re"
    with tarfile.open(stripped) as tf:
        keep = [m for m in tf.getmembers()
                if m.name.lstrip("./") not in ("lib/MAS_HFF",)]
        tf.extractall(extract, members=keep)
    stripped.unlink()
    with tarfile.open(stripped, "w:gz") as tf:
        for p in sorted(extract.rglob("*")):
            tf.add(p, arcname=p.relative_to(extract).as_posix())
    doc["archive_sha256"] = sha256_file(stripped)
    snap.manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    problems = verify_snapshot(pkg, snap.snapshot_dir)
    assert any("MAS_HFF" in p for p in problems), problems
    with pytest.raises(SnapshotError):
        require_snapshot(pkg, snap.snapshot_dir)
