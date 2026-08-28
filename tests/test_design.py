"""Parametric fuel-design chain (plan section 12, Phase A).

Fast, MASTER/DeCART-free tests: spec grid + LHS marginals, alias-registry
uniqueness/stability/persistence, DeCART deck editing against the real
templates, coredeck round-trips through the vendor harness (replace_lpd_shf /
advance_cycle_deck / validate_reload_deck), and the paramA fuel_types ingest.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from lpopt.design.coredeck import (
    build_cycle1_deck,
    build_reload_deck,
    library_dims,
    placeholder_shf,
)
from lpopt.design.lattice import edit_dec_text, resolve_template
from lpopt.design.spec import (
    ANCHOR_DESIGNS,
    DESIGN_GRID,
    DesignRegistry,
    FuelDesign,
    lhs_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
APR1400 = (REPO_ROOT / ".." / "0_APR1400").resolve()


# --------------------------------------------------------------------------- #
# spec: FuelDesign identity + validation
# --------------------------------------------------------------------------- #
def test_fueldesign_type_id_and_key() -> None:
    d = FuelDesign(5.8, 5.1, "z1", 6.0, 12)
    assert d.type_id == "P5851Z1G06N12"
    assert d.key == (58, 51, "z1", 6, 12)
    d2 = FuelDesign(6.2, round(6.2 * 0.92, 2), "z2", 10.0, 20)
    assert d2.type_id == "P6257Z2G10N20"


def test_fueldesign_validation() -> None:
    with pytest.raises(ValueError):
        FuelDesign(5.8, 6.0, "z1", 6.0, 12)          # e2 > e1
    with pytest.raises(ValueError):
        FuelDesign(5.8, 5.1, "z3", 6.0, 12)          # bad zoning
    with pytest.raises(ValueError):
        FuelDesign(5.8, 5.1, "z1", 6.0, 13)          # n_gd not mult of 4
    with pytest.raises(ValueError):
        FuelDesign(5.8, 5.1, "z1", 0.0, 12)          # gd_wt <= 0


def test_fueldesign_roundtrip_dict() -> None:
    d = FuelDesign(6.6, 5.61, "z1", 8.0, 24)
    assert FuelDesign.from_dict(d.as_dict()).key == d.key


# --------------------------------------------------------------------------- #
# spec: LHS grid
# --------------------------------------------------------------------------- #
def test_lhs_grid_distinct_and_anchors() -> None:
    g = lhs_grid(96, seed=1)
    assert len(g) == 96
    assert len({x.key for x in g}) == 96              # all distinct
    for anchor in ANCHOR_DESIGNS:
        assert anchor.key in {x.key for x in g}       # anchors retained


def test_lhs_grid_marginal_coverage() -> None:
    g = lhs_grid(96, seed=2)
    for axis in ("e1", "n_gd"):
        seen = {getattr(x, axis) for x in g}
        assert seen == set(DESIGN_GRID[axis])          # every level appears
    # each e1 level appears a non-trivial number of times (LHS balance)
    counts = collections.Counter(x.e1 for x in g)
    assert min(counts.values()) >= 10


def test_lhs_grid_full_240_unique() -> None:
    g = lhs_grid(240, seed=3)
    assert len({x.key for x in g}) == 240             # whole grid enumerated


def test_lhs_grid_deterministic() -> None:
    assert [x.key for x in lhs_grid(40, seed=7)] == [x.key for x in lhs_grid(40, seed=7)]


# --------------------------------------------------------------------------- #
# alias registry
# --------------------------------------------------------------------------- #
def test_alias_uniqueness_full_grid() -> None:
    reg = DesignRegistry()
    m = reg.register_all(lhs_grid(240, seed=4))
    assert len(set(m.values())) == len(m) == 240       # every alias unique
    # aliases never start with 'R' (reflector batch collision) and are 2 chars
    for alias in m.values():
        assert len(alias) == 2 and alias[0] != "R"


def test_alias_stability_and_persistence(tmp_path) -> None:
    reg = DesignRegistry()
    d1 = FuelDesign(5.0, 4.25, "z1", 8.0, 16)
    d2 = FuelDesign(6.6, 5.61, "z1", 8.0, 24)
    a1, a2 = reg.alias(d1), reg.alias(d2)
    assert reg.alias(d1) == a1                          # stable within session
    path = tmp_path / "registry.json"
    reg.save(path)
    reg2 = DesignRegistry.load(path)
    assert reg2.alias(d1) == a1 and reg2.alias(d2) == a2  # stable across reload
    # a new design gets a fresh, non-colliding alias
    d3 = FuelDesign(5.4, 4.59, "z2", 6.0, 12)
    a3 = reg2.alias(d3)
    assert a3 not in (a1, a2)


def test_registry_hgc_name() -> None:
    reg = DesignRegistry()
    d = FuelDesign(5.8, 5.1, "z1", 6.0, 12)
    assert reg.hgc_name(d) == f"FA_{reg.alias(d)}"
    assert len(reg.hgc_name(d)) == 5                   # FA_ + 2-char = 5 (COMP name)


# --------------------------------------------------------------------------- #
# lattice: template resolution + deck edits (no DeCART run)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not APR1400.is_dir(), reason="0_APR1400 templates absent")
def test_resolve_template_families() -> None:
    for n_gd, sub in ((12, "5.8_5.1"), (16, "5.8_5.1"), (20, "5.8_5.1"), (24, "260624")):
        d = FuelDesign(5.8, 5.1, "z1", 8.0, n_gd)
        tmpl = resolve_template(d, APR1400)
        assert tmpl.is_file()
        assert tmpl.parent.name == f"8_{n_gd}_z1"
        assert sub in str(tmpl)


@pytest.mark.skipif(not APR1400.is_dir(), reason="0_APR1400 templates absent")
def test_edit_dec_text_material() -> None:
    d = FuelDesign(6.6, 5.61, "z1", 8.0, 24)
    tmpl = resolve_template(d, APR1400)
    text = tmpl.read_text(errors="replace")
    out = edit_dec_text(text, d, "FA_ZZ")
    lines = out.splitlines()
    uo2 = [l for l in lines if l.strip().startswith("mixture UO2 ")][0]
    uo2_2 = [l for l in lines if l.strip().startswith("mixture UO2_2")][0]
    gd = [l for l in lines if l.strip().startswith("6408")][0]
    assert "92235 6.6" in uo2
    assert "92235 5.61" in uo2_2
    assert "6408  8.0" in gd or "6408 8.0" in gd
    assert out.splitlines()[0].split()[1] == "FA_ZZ"     # CASEID renamed
    assert "FA_ZZ" in [l for l in lines if l.strip().startswith("assembly")][0]


# --------------------------------------------------------------------------- #
# coredeck: harness round-trips (the M2.5-style byte-limited-diff test)
# --------------------------------------------------------------------------- #
def _aliases(n: int) -> list[str]:
    reg = DesignRegistry()
    return [reg.alias(d) for d in lhs_grid(n, seed=9)]


def test_reload_deck_validates_and_roundtrips() -> None:
    from lpopt.vendor.masterrl.master import extract_lpd_shf, replace_lpd_shf
    from lpopt.vendor.masterrl.equilibrium import advance_cycle_deck, deck_cycle
    from lpopt.search.assets import validate_reload_deck

    aliases = _aliases(4)
    deck = build_reload_deck(aliases, "MAS_RST.paramA", 12)
    dims = library_dims(len(aliases))
    assert dims == (7, 9)
    validate_reload_deck(deck, "MAS_RST.paramA", expected_dims=dims)   # no raise
    # replace(extract) is a byte-for-byte identity (M2.5 contract)
    assert replace_lpd_shf(deck, extract_lpd_shf(deck)) == deck
    # replacing with a fresh placeholder keeps exactly one %LPD_SHF
    d2 = replace_lpd_shf(deck, placeholder_shf(aliases[1]))
    assert d2.count("%LPD_SHF") == 1
    # advance_cycle_deck rewrites only 3 fields, SHF untouched
    adv = advance_cycle_deck(d2, "MAS_RST.NEXT", 13)
    assert deck_cycle(adv) == 13
    assert extract_lpd_shf(adv) == extract_lpd_shf(d2)
    changed = [(a, b) for a, b in zip(d2.splitlines(), adv.splitlines()) if a != b]
    assert len(changed) <= 3                                   # byte-limited diff


def test_reload_deck_dims_scale_with_types() -> None:
    for n in (2, 4, 10, 24):
        aliases = _aliases(n)
        assert library_dims(n) == (3 + n, 5 + n)
        deck = build_reload_deck(aliases, "MAS_RST.paramA", 12)
        assert f"{3 + n}       {5 + n}" in deck               # GEN_DIM nbatch ncomp


def test_cycle1_deck_is_fresh_core() -> None:
    aliases = _aliases(4)
    cy1 = build_cycle1_deck(aliases, (aliases[0], aliases[1]))
    assert "%LPD_BCH" in cy1 and "%LPD_SHF" not in cy1
    assert "irrst" in cy1                                      # %JOB_TYP 0 (no restart)
    # every fuel alias declared in the composition tables
    for a in aliases:
        assert f"FA_{a}" in cy1


def test_cycle1_cap_default_is_byte_identical() -> None:
    """The cap is opt-in: omitting it must reproduce the natural-EOC deck exactly."""
    aliases = _aliases(4)
    pair = (aliases[0], aliases[1])
    assert build_cycle1_deck(aliases, pair) == build_cycle1_deck(
        aliases, pair, cap_efpd=None)


def test_cycle1_cap_replaces_natural_eoc_with_fixed_ramp() -> None:
    aliases = _aliases(4)
    pair = (aliases[0], aliases[1])
    natural = build_cycle1_deck(aliases, pair)
    capped = build_cycle1_deck(aliases, pair, cap_efpd=597.7)

    # the adaptive natural-EOC search is gone; the only surviving boron
    # directive is the BOC %EXE_STD critical-boron search every cycle needs
    def _directives(deck: str, token: str) -> int:
        return sum(1 for line in deck.splitlines()
                   if not line.lstrip().startswith("#")
                   and token in line.split("#", 1)[0])

    assert "boron   10" in natural and "boron   10" not in capped
    assert _directives(natural, "boron") == 2          # EXE_STD + the EOC target
    assert _directives(capped, "boron") == 1           # EXE_STD only
    assert _directives(natural, "-30") == 1            # adaptive step ...
    assert _directives(capped, "-30") == 0             # ... removed when capped

    # the restart is still written at the end of the (now capped) cycle
    assert capped.rstrip().endswith("END")
    assert capped.count("%EDT_OPT") == natural.count("%EDT_OPT")
    assert "keff    tr      tr" in capped               # final %EXE_STD kept

    # depletion adds up to the cap: initial_steps (0+5+10+15 = 30) + the ramp
    from lpopt.design.coredeck import DEFAULT_CORE

    steps = []
    lines = capped.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("%EXE_DEP"):
            steps.append(float(lines[i + 1].split()[0]))
    assert sum(steps) == pytest.approx(597.7, abs=1e-6)
    assert steps[: len(DEFAULT_CORE.initial_steps)] == list(
        DEFAULT_CORE.initial_steps)
    # ramp steps never exceed |adaptive_step|
    assert max(steps[len(DEFAULT_CORE.initial_steps):]) <= abs(
        DEFAULT_CORE.adaptive_step)


def test_cycle1_cap_rejects_cap_below_initial_steps() -> None:
    aliases = _aliases(4)
    pair = (aliases[0], aliases[1])
    with pytest.raises(ValueError, match="must exceed the initial_steps"):
        build_cycle1_deck(aliases, pair, cap_efpd=10.0)


def test_reload_deck_never_capped() -> None:
    """Only cy1 is capped; reload cycles must still find their natural EOC."""
    aliases = _aliases(4)
    deck = build_reload_deck(aliases, "MAS_RST.paramA", 2)
    assert "boron   10" in deck and "-30" in deck


def test_make_band_restart_plumbs_cy1_cap(tmp_path, monkeypatch) -> None:
    """The cap reaches the cy1 deck through make_band_restart (no MASTER)."""
    from lpopt.design import bootstrap as B

    lib = tmp_path / "lib"
    lib.mkdir(parents=True)
    (lib / "MAS_XSL").write_text("COMP FA_P0\nCOMP FA_P1\n", encoding="utf-8")
    (lib / "MAS_HFF").write_text("hff\n", encoding="utf-8")

    seen: dict[str, str] = {}

    def _fake_run_cycle1(deck, xsl, hff, exe, work_dir, *, timeout_s=3600.0):
        seen["deck"] = deck
        raise B.BootstrapError("stop before MASTER")

    monkeypatch.setattr(B, "run_cycle1", _fake_run_cycle1)

    import random

    res = B.make_band_restart(tmp_path, "P0_P1", 121, random.Random(0),
                              cy1_cap_efpd=597.7)
    # make_band_restart funnels exceptions into result.error
    assert res.error is not None and "stop before MASTER" in res.error
    assert "boron   10" not in seen["deck"]
    assert "597.7" in seen["deck"] or "CAPPED at 597.7" in seen["deck"]

    seen.clear()
    B.make_band_restart(tmp_path, "P0_P1", 121, random.Random(0))
    assert "boron   10" in seen["deck"]                 # default unchanged


def test_bootstrap_purges_only_a_CONVERGED_chain(tmp_path, monkeypatch) -> None:
    """A chain that ran clean but never settled is a FAILED bootstrap.

    Its work dir is the only record of why it never settled, so the finally-block
    purge must skip it exactly as it skips the exception path (ECC audit).
    """
    import random

    from lpopt.design import bootstrap as B
    from lpopt.vendor.masterrl.domain import CaseKey

    lib = tmp_path / "lib"
    lib.mkdir(parents=True)
    (lib / "MAS_XSL").write_text("COMP FA_P0\nCOMP FA_P1\n", encoding="utf-8")
    (lib / "MAS_HFF").write_text("hff\n", encoding="utf-8")

    def _fake_run_cycle1(deck, xsl, hff, exe, work_dir, *, timeout_s=3600.0):
        work_dir.mkdir(parents=True, exist_ok=True)
        rst = work_dir / "MAS_RST.CY1.01"
        rst.write_bytes(b"rst")
        return rst

    class _FakeEq:
        converged_at_cap = False
        n_cycles = 3
        tolerance_margin = 0.5
        fom = None
        cycles = ()

        def __init__(self, converged):
            self.converged = converged

    def _runner_cls(converged):
        class _FakeRunner:
            def __init__(self, *a, **kw):
                pass

            def run(self, case_data, pattern):
                return _FakeEq(converged)

        return _FakeRunner

    monkeypatch.setattr(B, "run_cycle1", _fake_run_cycle1)
    monkeypatch.setattr(B, "MasterRunner", lambda *a, **kw: object())
    work = tmp_path / "bootstrap_work" / CaseKey("P0_P1", 121).folder

    monkeypatch.setattr(B, "PurgingEquilibriumRunner", _runner_cls(False))
    res = B.make_band_restart(tmp_path, "P0_P1", 121, random.Random(0))
    assert res.error is None and res.converged is False
    assert work.is_dir(), "a non-converged chain must keep its work dir"

    monkeypatch.setattr(B, "PurgingEquilibriumRunner", _runner_cls(True))
    res = B.make_band_restart(tmp_path, "P0_P1", 121, random.Random(0))
    assert res.converged is True
    assert not work.exists(), "a converged chain still purges (unchanged)"


def test_placeholder_shf_is_nine_lines() -> None:
    body = placeholder_shf("P0")
    lines = [l for l in body.splitlines() if l.strip()]
    assert len(lines) == 9
    assert not any(l.lstrip().startswith("%") for l in lines)


# --------------------------------------------------------------------------- #
# paramA fuel_types ingest
# --------------------------------------------------------------------------- #
def test_paramA_rows_axes(tmp_path) -> None:
    from lpopt.data.fuel_types import paramA_rows

    designs = [
        {"type_id": "P5042Z1G08N16", "alias": "P0", "e1": 5.0, "e2": 4.25,
         "zoning_variant": "z1", "gd_wt": 8.0, "n_gd": 16},
        {"type_id": "P6656Z1G08N24", "alias": "P3", "e1": 6.6, "e2": 5.61,
         "zoning_variant": "z1", "gd_wt": 8.0, "n_gd": 24},
    ]
    (tmp_path / "designs.json").write_text(
        json.dumps({"library_id": "paramA", "designs": designs}), encoding="utf-8")
    rows = paramA_rows(tmp_path)
    assert len(rows) == 2
    by = {r.type_id: r for r in rows}
    assert by["P5042Z1G08N16"].enr_main == 5.0
    assert by["P5042Z1G08N16"].enr_zone == 4.25
    assert by["P5042Z1G08N16"].n_gd == 16
    assert by["P6656Z1G08N24"].gd_wt == 8.0
    assert by["P6656Z1G08N24"].axial_zone == "z1"
    for r in rows:
        assert r.feature_poor is False
        assert r.library_id == "paramA"
        assert any(f.startswith("alias:") for f in r.source_flags)


def test_paramA_missing_out_is_skipped(tmp_path) -> None:
    from lpopt.data.fuel_types import paramA_rows

    (tmp_path / "designs.json").write_text(json.dumps({"designs": [
        {"type_id": "P5042Z1G08N16", "alias": "P0", "e1": 5.0, "e2": 4.25,
         "zoning_variant": "z1", "gd_wt": 8.0, "n_gd": 16}]}), encoding="utf-8")
    rows = paramA_rows(tmp_path)
    # no FA_P0.out present -> MASS unknown, but axes + feature_poor still set
    assert rows[0].u_avg_enrichment is None
    assert rows[0].feature_poor is False


def test_build_fuel_table_paramA_additive(tmp_path) -> None:
    """paramA rows join the table without disturbing the other libraries."""
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelPaths, build_fuel_table, fuel_paths_from_config

    cfg = load_config(REPO_ROOT / "lpopt.inp")
    base = fuel_paths_from_config(cfg)
    if not base.apr1400_root.is_dir():
        pytest.skip("0_APR1400 absent")
    paramA_dir = tmp_path / "paramA"
    paramA_dir.mkdir()
    (paramA_dir / "designs.json").write_text(json.dumps({"designs": [
        {"type_id": "P5851Z1G06N12", "alias": "P1", "e1": 5.8, "e2": 5.1,
         "zoning_variant": "z1", "gd_wt": 6.0, "n_gd": 12}]}), encoding="utf-8")

    without = build_fuel_table(base, persist=False)
    paths = FuelPaths(base.apr1400_root, base.ga80_hgc, base.manual_yaml,
                      tmp_path / "ft.parquet", paramA_root=paramA_dir)
    withp = build_fuel_table(paths, persist=False)
    assert "paramA" not in set(without["library_id"])
    assert "paramA" in set(withp["library_id"])
    # other libraries are untouched
    assert len(withp[withp["library_id"] != "paramA"]) == len(without)
    para = withp[withp["library_id"] == "paramA"]
    assert len(para) == 1
    assert bool(para["feature_poor"].iloc[0]) is False


# --------------------------------------------------------------------------- #
# run_batch idempotency (skip already-produced FA_<alias>.HGC — DeCART-free)
# --------------------------------------------------------------------------- #
_VALID_HGC = (
    "%TITL\n"
    " CASE :: REFERENCE CASE\n"
    " 1\n"
    " 1.0 0.0 1.35 1.30 0.0 900.0\n"
    " 305.0 700.0 0 155.0 0.74 1.0\n"
    "%DIST\n"
    + ("1.000 " * 16 + "\n") * 16
    + "% padding to clear the 256-byte floor " + "x" * 256 + "\n"
)


class _FakeProc:
    """A DeCART process stub that has already exited successfully."""

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_run_batch_idempotent_skips_completed(tmp_path, monkeypatch) -> None:
    """A design whose FA_<alias>.HGC (+ .out) already exists and parses is reused
    without re-launching DeCART; the rest are launched, results stay in order."""
    from lpopt.design import lattice as L
    from lpopt.design.spec import DesignRegistry, FuelDesign

    reg = DesignRegistry()
    d0 = FuelDesign(6.2, 5.70, "z2", 8.0, 16)
    d1 = FuelDesign(5.8, 4.93, "z1", 8.0, 16)
    a0, a1 = reg.alias(d0), reg.alias(d1)

    out_root = tmp_path / "work"
    # Pre-stage a COMPLETE, valid product for d0 only.
    wd0 = out_root / a0
    wd0.mkdir(parents=True)
    (wd0 / f"FA_{a0}.HGC").write_text(_VALID_HGC, encoding="utf-8")
    (wd0 / f"FA_{a0}.out").write_text("fresh inventory placeholder\n", encoding="utf-8")

    launched: list[str] = []

    def fake_launch(deck_path, work_dir, design, alias, exe=L.DEFAULT_DECART_EXE):
        launched.append(alias)
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        # DeCART writes <caseid>_0101.HGC (harvest renames it) + a .out.
        (wd / f"FA_{alias}_0101.HGC").write_text(_VALID_HGC, encoding="utf-8")
        (wd / f"FA_{alias}.out").write_text("fresh inventory placeholder\n", encoding="utf-8")
        run = L.DecartRun(design=design, alias=alias, work_dir=wd,
                          caseid=f"FA_{alias}", fa_name=f"FA_{alias}")
        run.process = _FakeProc()
        run.started = __import__("time").monotonic()
        return run

    monkeypatch.setattr(L, "launch_decart", fake_launch)
    monkeypatch.setattr(L, "write_dec_deck",
                        lambda design, wd, registry, apr: Path(wd) / "deck.inp")

    runs = L.run_batch([d0, d1], out_root, reg, tmp_path / "apr",
                       max_parallel=5, poll_s=0.001, timeout_s=30.0)

    # d0 was cached -> skipped; only d1 launched DeCART.
    assert launched == [a1]
    # order preserved, both usable products.
    assert [r.alias for r in runs] == [a0, a1]
    assert all(r.hgc_path is not None and r.hgc_path.is_file() for r in runs)
    assert all(r.out_path is not None and r.out_path.is_file() for r in runs)
    # the cached run reports zero wall time (no recompute).
    assert runs[0].wall_s == 0.0


def test_hgc_looks_valid_rejects_truncated(tmp_path) -> None:
    """The idempotency guard treats an empty/truncated HGC as NOT complete."""
    from lpopt.design.lattice import _hgc_looks_valid

    good = tmp_path / "FA_P0.HGC"
    good.write_text(_VALID_HGC, encoding="utf-8")
    assert _hgc_looks_valid(good) is True

    empty = tmp_path / "FA_P1.HGC"
    empty.write_text("", encoding="utf-8")
    assert _hgc_looks_valid(empty) is False

    partial = tmp_path / "FA_P2.HGC"
    partial.write_text("%TITL\n garbage without case or dist\n" + "y" * 300,
                       encoding="utf-8")
    assert _hgc_looks_valid(partial) is False

    assert _hgc_looks_valid(tmp_path / "missing.HGC") is False
