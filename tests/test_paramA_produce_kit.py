"""paramA per-library routing for the PLAIN produce path + the multi-PC kit.

The curriculum already routes paramA cells to the design package; these tests
pin the SAME routing on (a) ``ProduceDriver``'s default resolver/verifier (a kit
PC runs ``lpopt produce``, never the curriculum) and (b) the exported kit deck +
bundled design package, so an assigned paramA band cell produces against the
design package (its own bases/cores/lib + registry alias bridge + %GEN_DIM dims)
rather than the ga80 FEASIBLE_PACKAGE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lpopt.config import (
    CaseConfig,
    CurriculumConfig,
    DataConfig,
    DesignConfig,
    ExtractConfig,
    FlowConfig,
    FuelConfig,
    LpoptConfig,
    MasterConfig,
    ModelConfig,
    ProduceConfig,
    RemoteConfig,
    StratumConfig,
    VerifyConfig,
    load_config,
)
from lpopt.design.coredeck import library_dims
from lpopt.multi_pc import KitError, band_library, export_produce_kit
from lpopt.search.assets import LIBRARY_DIMS
from lpopt.search.produce import ProduceDriver
from lpopt.search.resolver import is_paramA_library

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_paramA_pkg(base: Path) -> Path:
    """Minimal assembled paramA design package: registry alias bridge, a lib/
    MAS_XSL COMP roster (3 sets -> dims), one seed base restart + a core deck."""
    pkg = base / "design_pkg"
    (pkg / "bases" / "P0_P1").mkdir(parents=True, exist_ok=True)
    (pkg / "bases" / "P0_P1" / "MAS_RST.SEED.02").write_bytes(b"seed")
    (pkg / "cores" / "P0_P1" / "s1").mkdir(parents=True, exist_ok=True)
    (pkg / "cores" / "P0_P1" / "s1" / "MAS_INP_cy01.inp").write_text(
        "dummy\n%LPD_SHF\n F P0  0,\n%END\n", encoding="utf-8")
    (pkg / "hgc").mkdir(parents=True, exist_ok=True)
    (pkg / "registry.json").write_text(
        json.dumps({"aliases": {"P5849X": "P0", "P6257X": "P1", "P6253X": "P2"}}),
        encoding="utf-8")
    (pkg / "designs.json").write_text(json.dumps({"designs": []}), encoding="utf-8")
    lib = pkg / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    lib.joinpath("MAS_XSL").write_text(
        "COMP FA_P0  x\nCOMP FA_P1  x\nCOMP FA_P2  x\n", encoding="utf-8")
    lib.joinpath("MAS_HFF").write_text("hff\n", encoding="utf-8")
    return pkg


def _make_ga80_pkg(base: Path) -> Path:
    pkg = base / "ga80_pkg"
    (pkg / "bases" / "K1_K2").mkdir(parents=True, exist_ok=True)
    (pkg / "bases" / "K1_K2" / "MAS_RST.NATIVE.01").write_bytes(b"rst")
    return pkg


def _cfg(tmp_path: Path, pkg: Path, strata: list[StratumConfig], *,
         design_pkg: Path | None = None, store_dir: Path | None = None,
         cell_pairs: dict | None = None) -> LpoptConfig:
    return LpoptConfig(
        flow=FlowConfig(random_seed=3),
        remote=RemoteConfig(),
        master=MasterConfig(executable="master.exe"),
        verify=VerifyConfig(package_root=str(pkg) if pkg else None),
        data=DataConfig(),
        case=CaseConfig(),
        fuel=FuelConfig(),
        extract=ExtractConfig(),
        produce=ProduceConfig(campaign="multipc_kit", workers=4,
                              template_fallbacks=[], strata=strata),
        design=DesignConfig(package_root=str(design_pkg) if design_pkg else None),
        model=ModelConfig(store_dir=str(store_dir) if store_dir else str(tmp_path / "store")),
        curriculum=CurriculumConfig(cell_pairs=cell_pairs or {}),
        source_path=tmp_path / "lpopt.inp",
    )


class _FakeLib:
    """Just enough for export (fuel frame write) + pair selection override."""
    frame = pd.DataFrame({"type_id": ["P5849X"], "library_id": ["paramA"],
                          "u_avg_enrichment": [5.85]})


# --------------------------------------------------------------------------- #
# band -> library routing (pure)
# --------------------------------------------------------------------------- #
def test_band_library_routes_by_band() -> None:
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    assert band_library(cfg, [5.75, 6.0]) == "paramA"
    assert band_library(cfg, [6.0, 6.25]) == "paramA"
    assert band_library(cfg, [5.25, 5.5]) == "ga80"
    assert band_library(cfg, [5.0, 5.25]) == "ga80"
    cfg.curriculum.band_libraries = {"5.25-5.5": "paramA"}
    assert band_library(cfg, [5.25, 5.5]) == "paramA"


# --------------------------------------------------------------------------- #
# produce driver resolver / verifier routing
# --------------------------------------------------------------------------- #
def _paramA_strat() -> StratumConfig:
    return StratumConfig(name="5.75-6_f109", campaign="5.75-6_f109",
                         library="paramA", pairs=["P5849X_P6257X"], feed=109,
                         n_target=8, generators={"random": 1.0}, priority=100)


def test_produce_default_resolver_routes_paramA(tmp_path: Path) -> None:
    pkg = _make_paramA_pkg(tmp_path)
    cfg = _cfg(tmp_path, None, [_paramA_strat()], design_pkg=pkg)
    driver = ProduceDriver(cfg, dry_run=True, run_dir=tmp_path / "run",
                           store_dir=tmp_path / "store",
                           ledger_path=tmp_path / "ledger.jsonl", progress=False)
    r = driver.resolver
    assert r.package_root == pkg
    assert r.library_id == "paramA"
    assert r.template_fallbacks == ()                      # ga80 decks suppressed
    assert r.type_to_alias == {"P5849X": "P0", "P6257X": "P1", "P6253X": "P2"}
    assert r.library_dims == library_dims(3)               # 3 COMP FA_* -> (6, 8)


def test_produce_default_resolver_ga80_unchanged(tmp_path: Path) -> None:
    pkg = _make_ga80_pkg(tmp_path)
    strat = StratumConfig(name="5.25-5.5_f101", campaign="5.25-5.5_f101",
                          library="ga80", pairs=["K1_K2"], feed=121, n_target=4,
                          generators={"random": 1.0}, priority=100)
    cfg = _cfg(tmp_path, pkg, [strat])
    driver = ProduceDriver(cfg, dry_run=True, run_dir=tmp_path / "run",
                           store_dir=tmp_path / "store",
                           ledger_path=tmp_path / "ledger.jsonl", progress=False)
    r = driver.resolver
    assert r.package_root == pkg
    assert r.library_dims == LIBRARY_DIMS
    assert r.type_to_alias == {}


def test_produce_live_verifier_paramA_dims_and_root(tmp_path: Path) -> None:
    pkg = _make_paramA_pkg(tmp_path)
    cfg = _cfg(tmp_path, None, [_paramA_strat()], design_pkg=pkg)
    driver = ProduceDriver(cfg, dry_run=False, run_dir=tmp_path / "run",
                           store_dir=tmp_path / "store",
                           ledger_path=tmp_path / "ledger.jsonl", progress=False)
    assert driver.verifier.package_root == pkg
    assert driver.verifier.library_dims == library_dims(3)


# --------------------------------------------------------------------------- #
# kit export: bundles the design package + paramA deck knobs
# --------------------------------------------------------------------------- #
def test_export_kit_paramA_bundles_design_package(tmp_path: Path) -> None:
    pkg = _make_paramA_pkg(tmp_path)
    store = tmp_path / "store"                    # empty -> export writes fuel frame
    cfg = _cfg(tmp_path, None, [], design_pkg=pkg, store_dir=store,
               cell_pairs={"5.75-6_f109": ["P5849X_P6257X"]})
    out = tmp_path / "kit"
    result = export_produce_kit(cfg, ["5.75-6_f109"], out, n_target=150,
                                fuel_library=_FakeLib(), log=lambda m: None)

    parsed = load_config(result.deck_path)
    assert [s.name for s in parsed.produce.strata] == ["5.75-6_f109"]
    strat = parsed.produce.strata[0]
    assert strat.library == "paramA"
    assert list(strat.pairs) == ["P5849X_P6257X"]
    # deck routes to the bundled design package, NOT FEASIBLE_PACKAGE
    assert parsed.design.package_root == "data/design/package"
    assert parsed.verify.package_root is None
    assert is_paramA_library(parsed, "paramA")
    # the design package was copied into the kit (self-contained)
    kit_pkg = out / "data" / "design" / "package"
    assert (kit_pkg / "registry.json").is_file()
    assert (kit_pkg / "lib" / "MAS_XSL").is_file()
    assert (kit_pkg / "bases" / "P0_P1" / "MAS_RST.SEED.02").is_file()
    readme = (out / "KIT_README.md").read_text(encoding="utf-8")
    assert "data/design/package" in readme


def test_is_paramA_library_survives_curriculum_library_paramA(tmp_path: Path) -> None:
    """The verdict is ``library_id == paramA_library`` and nothing else.

    The retired ``lib_id != cfg.curriculum.library`` clause is vacuous while the
    deck says ``curriculum.library = ga80``, but it made this predicate ALWAYS
    false — silently routing every paramA cell back to the ga80 package — as soon
    as an operator set ``curriculum.library = "paramA"`` (ECC audit).
    """
    pkg = _make_paramA_pkg(tmp_path)
    cfg = _cfg(tmp_path, None, [], design_pkg=pkg, store_dir=tmp_path / "store",
               cell_pairs={"5.75-6_f109": ["P5849X_P6257X"]})
    assert is_paramA_library(cfg, "paramA")
    assert not is_paramA_library(cfg, "ga80")

    cfg.curriculum.library = "paramA"               # the trap configuration
    assert is_paramA_library(cfg, "paramA")
    assert not is_paramA_library(cfg, "ga80")


def test_export_kit_paramA_missing_package_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"                   # does not exist
    cfg = _cfg(tmp_path, None, [], design_pkg=missing, store_dir=tmp_path / "store",
               cell_pairs={"5.75-6_f109": ["P5849X_P6257X"]})
    with pytest.raises(KitError):
        export_produce_kit(cfg, ["5.75-6_f109"], tmp_path / "kit",
                           fuel_library=_FakeLib(), log=lambda m: None)
