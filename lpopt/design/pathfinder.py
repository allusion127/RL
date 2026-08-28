"""End-to-end pathfinder acceptance gate (plan 12.1).

Runs the whole chain on 4 grid-spanning designs and reports the measurements the
plan asks for: DeCART parallel timing, the anchor cross-check, the TotalBatcher
COMP count (+ optional ceiling probe), the bootstrap cycle count, and one
WaveVerifier evaluation through the existing produce harness.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..data.fuel_types import count_gd_pins_from_hgc, parse_fa_mass_out
from .lattice import launch_decart, harvest, resolve_template, write_dec_deck
from .library import build_master_library, default_tool_paths
from .package import (
    DesignSource,
    ingest_fuel_types,
    write_core_template,
    write_designs_manifest,
    stage_hgc,
)
from .spec import DesignRegistry, FuelDesign

#: 4 designs spanning grid corners + center (plan 12.1 pathfinder).
PATHFINDER_DESIGNS: list[FuelDesign] = [
    FuelDesign(5.0, round(5.0 * 0.85, 2), "z1", 8.0, 16),   # low-enrichment corner
    FuelDesign(5.8, 5.1, "z1", 6.0, 12),                    # anchor (existing hardware)
    FuelDesign(6.2, round(6.2 * 0.92, 2), "z2", 10.0, 20),  # high-e / z2 / gd10
    FuelDesign(6.6, round(6.6 * 0.85, 2), "z1", 8.0, 24),   # max-enrichment / n24
]


@dataclass
class PathfinderResult:
    ok: bool = False
    decart_walls: dict = field(default_factory=dict)
    decart_parallel_wall: float | None = None
    anchor_check: dict = field(default_factory=dict)
    library: dict = field(default_factory=dict)
    comp_ceiling: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    verify: dict = field(default_factory=dict)
    fuel_ingest: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def report(self) -> str:
        return json.dumps({
            "ok": self.ok,
            "decart_walls_s": self.decart_walls,
            "decart_parallel_wall_s": self.decart_parallel_wall,
            "anchor_check": self.anchor_check,
            "library": self.library,
            "comp_ceiling": self.comp_ceiling,
            "bootstrap": self.bootstrap,
            "verify": self.verify,
            "fuel_ingest": self.fuel_ingest,
            "notes": self.notes,
        }, indent=2, default=str)


# --------------------------------------------------------------------------- #
# anchor cross-check
# --------------------------------------------------------------------------- #
def cross_check_anchor(design: FuelDesign, out_path: Path, hgc_path: Path,
                       apr1400_root: Path) -> dict:
    """Compare the anchor's MASS enrichment + Gd count vs its template sibling.

    The anchor (5.8/5.1, gd6, n12) is numerically identical to the 5.8_5.1
    X-series template it edits, so the produced inventory should reproduce the
    reference within 1e-3 (enrichment) and exactly (Gd pin count).
    """
    got = parse_fa_mass_out(out_path)
    template = resolve_template(design, apr1400_root)
    ref_out = None
    for cand in template.parent.glob("FA_*.out"):
        ref_out = cand
        break
    ref = parse_fa_mass_out(ref_out) if ref_out is not None else None
    n_gd_hgc = None
    try:
        n_gd_hgc = count_gd_pins_from_hgc(hgc_path)
    except (OSError, ValueError):
        pass
    d = {
        "design": design.type_id,
        "u_avg_enrichment": got["u_avg_enrichment"],
        "u_mass_g": got["u_mass_g"],
        "reference_out": str(ref_out) if ref_out else None,
        "ref_u_avg_enrichment": ref["u_avg_enrichment"] if ref else None,
        "enr_abs_diff": (abs(got["u_avg_enrichment"] - ref["u_avg_enrichment"])
                         if ref else None),
        "enr_match_1e-3": (abs(got["u_avg_enrichment"] - ref["u_avg_enrichment"]) < 1e-3
                           if ref else None),
        "n_gd_hgc": n_gd_hgc,
        "n_gd_expected": design.n_gd,
        "n_gd_exact": (n_gd_hgc == design.n_gd) if n_gd_hgc is not None else None,
    }
    return d


# --------------------------------------------------------------------------- #
# COMP-ceiling probe
# --------------------------------------------------------------------------- #
def comp_ceiling_probe(new_hgcs: list[Path], ga80_hgc_dir: Path, out_dir: Path,
                       apr1400_root: Path) -> dict:
    """Batch the 4 new + all ga80 HGCs through TotalBatcher and report acceptance.

    Cheap probe of the MASTER COMP ceiling (plan 12.1): if TotalBatcher accepts
    40+ sets it reports the total; a failure is captured (not raised) so the
    pathfinder still passes on the core 4-type library.
    """
    ga = sorted(Path(ga80_hgc_dir).glob("FA_*.HGC")) if Path(ga80_hgc_dir).is_dir() else []
    all_hgcs = list(new_hgcs) + ga
    tp = default_tool_paths(apr1400_root)
    try:
        build = build_master_library(
            all_hgcs, out_dir, mas_ref=tp["mas_ref"], prolog_exe=tp["prolog_exe"],
            totalbatcher_exe=tp["totalbatcher_exe"], library_id="ceiling_probe")
        return {"attempted": len(all_hgcs), "comp_count": build.comp_count,
                "ncomp": build.ncomp, "accepted": True}
    except Exception as exc:                                # noqa: BLE001
        return {"attempted": len(all_hgcs), "accepted": False,
                "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_pathfinder(cfg, work_dir: str | Path, registry: DesignRegistry, *,
                   apr1400_root: Path, decart_exe: str, master_exe: str | None,
                   max_parallel: int = 4, skip_decart: bool = False,
                   do_bootstrap: bool = True, do_verify: bool = True,
                   do_ceiling: bool = True) -> PathfinderResult:
    """The full 4-type acceptance gate; see module docstring."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    designs = PATHFINDER_DESIGNS
    for d in designs:
        registry.alias(d)
    registry.save(work / "registry.json")
    res = PathfinderResult()

    # -- 1. DeCART (concurrent) or reuse -----------------------------------
    runs = {}
    for d in designs:
        alias = registry.alias(d)
        wd = work / "work" / alias
        deck = write_dec_deck(d, wd, registry, apr1400_root)
        runs[alias] = (d, wd, deck)

    if not skip_decart:
        t0 = time.monotonic()
        live = []
        pending = list(designs)
        active = {}
        while pending or active:
            while pending and len(active) < max_parallel:
                d = pending.pop(0)
                alias = registry.alias(d)
                _, wd, deck = runs[alias]
                active[alias] = launch_decart(deck, wd, d, alias, exe=decart_exe)
            time.sleep(15)
            for alias in list(active):
                r = active[alias]
                if r.poll():
                    harvest(r)
                    res.decart_walls[alias] = r.wall_s
                    live.append(r)
                    del active[alias]
        res.decart_parallel_wall = time.monotonic() - t0

    # collect harvested products
    sources: list[DesignSource] = []
    for d in designs:
        alias = registry.alias(d)
        _, wd, _ = runs[alias]
        hgc = wd / f"FA_{alias}.HGC"
        out = wd / f"FA_{alias}.out"
        if not hgc.is_file():
            res.notes.append(f"missing HGC for {alias} ({d.type_id})")
            return res
        sources.append(DesignSource(d, alias, hgc, out if out.is_file() else None))

    # -- 2. anchor cross-check ---------------------------------------------
    anchor = designs[1]                                     # 5.8/5.1 anchor
    a_alias = registry.alias(anchor)
    _, a_wd, _ = runs[a_alias]
    res.anchor_check = cross_check_anchor(
        anchor, a_wd / f"FA_{a_alias}.out", a_wd / f"FA_{a_alias}.HGC", apr1400_root)

    # -- 3. assemble package + build library -------------------------------
    pkg = work / "package"
    write_designs_manifest(pkg, sources, registry)
    staged = stage_hgc(pkg, sources)
    tp = default_tool_paths(apr1400_root)
    build = build_master_library(staged, pkg / "lib", mas_ref=tp["mas_ref"],
                                 prolog_exe=tp["prolog_exe"],
                                 totalbatcher_exe=tp["totalbatcher_exe"],
                                 library_id="paramA")
    res.library = {"comp_count": build.comp_count, "refl_count": build.refl_count,
                   "ncomp": build.ncomp, "sets": build.set_names}

    # -- 3b. COMP-ceiling probe (4 new + ga80) -----------------------------
    if do_ceiling:
        ga80_hgc = apr1400_root / ".." / "3_GA_Surrogate" / "FEASIBLE_PACKAGE" / "hgc"
        res.comp_ceiling = comp_ceiling_probe(
            staged, ga80_hgc.resolve(), work / "ceiling_probe", apr1400_root)

    # -- 4. bootstrap one case (2 new types, feed 121) ---------------------
    aliases = [registry.alias(d) for d in designs]
    pair = f"{aliases[0]}_{aliases[2]}"                     # low-e + high-e new types
    if do_bootstrap and master_exe:
        from .bootstrap import make_band_restart
        b = make_band_restart(pkg, pair, 121, random.Random(0), aliases=aliases,
                              exe=master_exe,
                              max_cycles=cfg.design.bootstrap_max_cycles,
                              enable_pin_burnup=cfg.design.enable_pin_burnup)
        res.bootstrap = b.summary()

        # -- 5. WaveVerifier eval through the existing harness -------------
        if do_verify and b.restart_path is not None and b.converged:
            res.verify = _verify_one(cfg, pkg, pair, 121, aliases, b.restart_path,
                                     master_exe, work / "verify")

    # -- 6. fuel_types ingest ---------------------------------------------
    try:
        from ..data.fuel_types import FuelLibrary
        store_path = Path(cfg.design.store_dir) / "fuel_types_paramA.parquet"
        df = ingest_fuel_types(pkg, cfg=cfg, store_path=store_path)
        lib = FuelLibrary(df)
        paramA = lib.frame[lib.frame["library_id"] == "paramA"]
        res.fuel_ingest = {
            "n_paramA_rows": int(len(paramA)),
            "feature_poor": int(paramA["feature_poor"].sum()),
            "type_ids": paramA["type_id"].tolist(),
        }
    except Exception as exc:                                # noqa: BLE001
        res.fuel_ingest = {"error": f"{type(exc).__name__}: {exc}"}

    res.ok = bool(res.library.get("comp_count") == len(designs)
                  and res.anchor_check.get("enr_match_1e-3") is not False
                  and res.fuel_ingest.get("n_paramA_rows", 0) >= len(designs))
    return res


def _verify_one(cfg, pkg: Path, pair: str, feed: int, aliases: list[str],
                base_restart: Path, master_exe: str, run_dir: Path) -> dict:
    """Run one WaveVerifier evaluation through the existing harness."""
    from ..search.verify import WaveVerifier, WaveEntry
    from ..search.assets import CaseAssetResolver
    from ..search.genome import fresh_units_from_feed, random_genome
    from ..vendor.masterrl.domain import CaseKey
    from .coredeck import library_dims

    # a reload template the resolver can find for this case
    write_core_template(pkg, pair, feed, aliases, base_restart.name, cycle=12)

    resolver = CaseAssetResolver(pkg, library_id="paramA")
    ck = CaseKey(pair, feed)
    resolved = resolver.resolve(ck)
    pattern = random_genome(random.Random(1), pair, fresh_units_from_feed(feed)).to_pattern()
    dims = library_dims(len(aliases))
    verifier = WaveVerifier(run_dir=run_dir, package_root=pkg, executable=master_exe,
                            workers=1, max_cycles=cfg.design.bootstrap_max_cycles,
                            library_dims=dims)
    entry = WaveEntry(pattern=pattern, case_key=ck, resolved_assets=resolved,
                      meta={"library_id": "paramA"})
    outcomes = verifier.evaluate_wave([entry])
    o = outcomes[0]
    return {
        "status": o.status,
        "n_cycles": o.n_cycles,
        "wall_s": o.wall_s,
        "fom": o.fom.as_dict() if o.fom is not None else None,
        "restart_provenance": o.restart_provenance,
        "failure": o.failure,
    }


__all__ = ["PATHFINDER_DESIGNS", "PathfinderResult", "cross_check_anchor",
           "comp_ceiling_probe", "run_pathfinder"]
