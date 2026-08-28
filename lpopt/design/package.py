"""Assemble a paramA MASTER package in FEASIBLE_PACKAGE layout (plan 12.1).

Target layout (what ``[verify] package_root`` and ``MasterRunner`` require):

    <pkg>/
      designs.json                     design records (fuel_types ingest source)
      registry.json                    type_id <-> alias map
      hgc/FA_<alias>.HGC + .out        DeCART products (fuel_types MASS source)
      lib/MAS_XSL, lib/MAS_HFF         TotalBatcher library
      bases/<folder>/MAS_RST.*         band-seed restarts (written by bootstrap)
      cores/<folder>/<id>/MAS_INP_cyNN.inp   reload template (replace_lpd_shf-ready)

``ingest_fuel_types`` registers the new types into ``fuel_types.parquet`` with
``library_id="paramA"`` and full physics (``feature_poor=False``).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..data.fuel_types import FuelPaths, build_fuel_table, fuel_paths_from_config
from .coredeck import DEFAULT_CORE, CoreParams, build_reload_deck
from .library import LibraryBuild, build_master_library, default_tool_paths
from .spec import DesignRegistry, FuelDesign


@dataclass
class DesignSource:
    """A produced lattice: its design, alias, and harvested product files."""

    design: FuelDesign
    alias: str
    hgc_path: Path
    out_path: Path | None = None


def write_designs_manifest(pkg_dir: str | Path, sources: list[DesignSource],
                           registry: DesignRegistry) -> Path:
    """Write ``designs.json`` (fuel_types ingest source) + ``registry.json``."""
    pkg = Path(pkg_dir)
    pkg.mkdir(parents=True, exist_ok=True)
    records = []
    for s in sources:
        rec = s.design.as_dict()
        rec["alias"] = s.alias
        rec["gd_u_enr"] = 4.0
        records.append(rec)
    manifest = pkg / "designs.json"
    manifest.write_text(
        json.dumps({"library_id": "paramA", "designs": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    registry.save(pkg / "registry.json")
    return manifest


def stage_hgc(pkg_dir: str | Path, sources: list[DesignSource]) -> list[Path]:
    """Copy each ``FA_<alias>.HGC`` (+ ``.out``) into ``<pkg>/hgc/``."""
    hgc_dir = Path(pkg_dir) / "hgc"
    hgc_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for s in sources:
        dst = hgc_dir / f"FA_{s.alias}.HGC"
        if Path(s.hgc_path).resolve() != dst.resolve():
            shutil.copyfile(s.hgc_path, dst)
        staged.append(dst)
        if s.out_path is not None and Path(s.out_path).is_file():
            out_dst = hgc_dir / f"FA_{s.alias}.out"
            if Path(s.out_path).resolve() != out_dst.resolve():
                shutil.copyfile(s.out_path, out_dst)
    return staged


def build_library_from_sources(pkg_dir: str | Path, sources: list[DesignSource],
                               apr1400_root: str | Path, *,
                               tools: dict | None = None) -> LibraryBuild:
    """Stage HGCs and run TotalBatcher into ``<pkg>/lib/``."""
    pkg = Path(pkg_dir)
    hgc_paths = stage_hgc(pkg, sources)
    tp = tools or default_tool_paths(apr1400_root)
    return build_master_library(
        hgc_paths, pkg / "lib",
        mas_ref=tp["mas_ref"], prolog_exe=tp["prolog_exe"],
        totalbatcher_exe=tp["totalbatcher_exe"], library_id="paramA",
    )


def write_core_template(pkg_dir: str | Path, pair: str, feed: int,
                        aliases: list[str], restart_basename: str, *,
                        seed_id: str = "seed", cycle: int = 12,
                        core: CoreParams = DEFAULT_CORE) -> Path:
    """Write a reload template ``cores/<folder>/<seed_id>/MAS_INP_cyNN.inp``.

    ``restart_basename`` is the base restart the deck references; the harness
    ``prepare_cycle1_deck`` rewrites it to the resolved restart at eval time, and
    ``replace_lpd_shf`` overwrites the placeholder loading.
    """
    from ..vendor.masterrl.domain import CaseKey

    folder = CaseKey(pair, int(feed)).folder
    seed_dir = Path(pkg_dir) / "cores" / folder / seed_id
    seed_dir.mkdir(parents=True, exist_ok=True)
    deck = build_reload_deck(aliases, restart_basename, cycle, core=core)
    path = seed_dir / f"MAS_INP_cy{cycle:02d}.inp"
    path.write_text(deck, encoding="utf-8")
    return path


def ingest_fuel_types(pkg_dir: str | Path, *, cfg=None, base_paths: FuelPaths | None = None,
                      store_path: str | Path | None = None):
    """Merge paramA rows into ``fuel_types.parquet`` (full physics).

    Uses ``build_fuel_table`` with ``paramA_root`` set to ``pkg_dir`` (its
    ``designs.json`` + ``hgc/FA_*.out`` drive the ingest).  Either ``cfg`` (an
    LpoptConfig) or ``base_paths`` supplies the other five libraries' locations.
    Returns the rebuilt DataFrame.

    The paramA rows produced by ``paramA_rows`` now also auto-fill the cond_v4
    physics columns from the staged ``hgc/FA_<alias>.sum`` / ``.HGC`` (reference
    k-inf curve + BOC reactivity coefficients + 2-group xs / ADF / pin-power
    peaking); ``zone_pin_count`` stays NaN unless a ``dec_FA_*.inp`` is staged.
    """
    pkg = Path(pkg_dir)
    if base_paths is None:
        if cfg is None:
            raise ValueError("ingest_fuel_types needs cfg or base_paths")
        base_paths = fuel_paths_from_config(cfg)
    store = Path(store_path) if store_path is not None else base_paths.store
    paths = FuelPaths(
        apr1400_root=base_paths.apr1400_root,
        ga80_hgc=base_paths.ga80_hgc,
        manual_yaml=base_paths.manual_yaml,
        store=store,
        paramA_root=pkg,
    )
    return build_fuel_table(paths, persist=True)


def assemble_package(pkg_dir: str | Path, sources: list[DesignSource],
                     registry: DesignRegistry, apr1400_root: str | Path, *,
                     tools: dict | None = None) -> LibraryBuild:
    """One call: manifest + registry + hgc staging + TotalBatcher library.

    Bootstrap (bases/) and core templates (cores/) are added by the caller once
    seeds exist, since they need an assembled ``lib/`` first.
    """
    write_designs_manifest(pkg_dir, sources, registry)
    return build_library_from_sources(pkg_dir, sources, apr1400_root, tools=tools)


__all__ = [
    "DesignSource",
    "assemble_package",
    "build_library_from_sources",
    "ingest_fuel_types",
    "stage_hgc",
    "write_core_template",
    "write_designs_manifest",
]
