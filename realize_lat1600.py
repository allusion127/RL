"""realize_lat1600.py -- realize the lat1600 flat-lattice set (Y1-Y4) into the
production paramA package (5_RL/data/design/package).

PROGRAM DECISION (user, 2026-08-11): CBC feasibility limit 1550 -> 1600 ppm for
new work + new low-FF lattices with OPEN Gd layouts.  The four designs come from
the CPU-surrogate screen at the 1600 gate (scratchpad lat1600/chosen.json):

    id  u_high  u_low   gd_wt n_gd  Gd 1/8 layout        FF_ens  role
    Y1  5.00    4.2500  10    16    1:1;5:2;5:5          1.1073  E1-role matched
    Y2  5.00    4.2500   8    24    1:1;4:1;5:5;6:3      1.1409  E2-role matched (hot)
    Y3  5.25    4.4625   6    16    1:1;5:2;5:5          1.1012  flat anchor (E3 layout)
    Y4  5.30    4.5050   6    16    1:1;5:2;5:5          1.1011  flat extreme @1600

The chain (lpopt.design) has NO code that writes Gd pin maps -- lattice.edit_dec_text
edits only UO2/UO2_2 92235 and UO2G 6408 (lattice.py:114-148), and resolve_template
(lattice.py:49-71) keys the FROZEN 0_APR1400 template tree by (gd_wt, n_gd, z),
whose layouts are the WRONG ones (flat_assembly_fr_plan_20260802.md sec. 1.3: the
frozen layouts cannot beat ga80).  This driver therefore first AUTHORS a custom
template tree (5_RL/templates_lat1600/, same subtree shape resolve_template
expects) by moving the Gd cell ids inside the octant map of the plan's T1-1 base
templates, with hard guards (guide tubes frozen at 1/8 (0,0),(3,3),(4,3),(4,4);
zoning cells untouched; census == target; n_gd exact), then runs the chain's own
functions end to end, mirroring curriculum._generate_band_designs
(lpopt/curriculum.py:1495-1596):

    1. author templates          (this file; deterministic, idempotent)
    2. DesignRegistry.load(package/registry.json) + alias assign + save
                                 (spec.py:154-219; NEVER the divergent
                                  data/design/registry.json store registry)
    3. run_batch 4-way DeCART    (lattice.py:409-455, idempotent-skip, ~13 min)
       + Gd census check on the produced HGC (count_gd_pins_from_hgc)
    4. UNION merge old 33 + new  (curriculum.py:1557-1577 logic; the stale-HGC
                                  guard in build_master_library REFUSES partials)
    5. assemble_package          (package.py:142-151: designs.json + registry.json
                                  + hgc staging + TotalBatcher lib rebuild;
                                  MAS_XSL/MAS_HFF keep ONE .bak generation --
                                  snapshot lib/ FIRST, hence --snapshot-ok)
    6. layout provenance         (gd_positions key added to the new designs.json
                                  records; paramA_rows reads by key -> tolerant)
    7. ingest_fuel_types         (package.py:112-139) -> data/design/
                                  fuel_types_paramA.parquet (pathfinder.py:233-234
                                  convention; data/store is NOT touched)

Aliases are auto-assigned by the package registry (next free after P0..T2 = 33
types: expected T3,T4,T5,T6 in Y1..Y4 order -- but ALWAYS read them from this
driver's output, never assume).  Expected result: ncomp 42 (5 REFL + 37 COMP).

Usage (issuer runs this; DeCART2D exists on box 104 only):

    python realize_lat1600.py --dry-run              # author + verify templates,
                                                     # print aliases; no DeCART,
                                                     # no package mutation
    python realize_lat1600.py --snapshot-ok          # the real run (~15-20 min)
    python realize_lat1600.py --designs Y3,Y4 --snapshot-ok
                                                     # ncomp-40 fallback (plan 4.3)

Never run by the agent -- DeCART/TotalBatcher execution belongs to the issuer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from lpopt.data.fuel_types import FuelPaths, count_gd_pins_from_hgc  # noqa: E402
from lpopt.design.lattice import (  # noqa: E402
    DEFAULT_DECART_EXE,
    _read_text_flex,
    resolve_template,
    run_batch,
)
from lpopt.design.package import (  # noqa: E402
    DesignSource,
    assemble_package,
    ingest_fuel_types,
)
from lpopt.design.spec import DesignRegistry, FuelDesign  # noqa: E402

# --------------------------------------------------------------------------- #
# fixed paths (box 104 workspace layout)
# --------------------------------------------------------------------------- #
APR1400 = (BASE / ".." / "0_APR1400").resolve()
PKG = BASE / "data" / "design" / "package"
WORK = BASE / "data" / "design" / "work"           # cmd_design_run convention
TEMPLATE_ROOT = BASE / "templates_lat1600"          # custom-layout dec_FA tree
GA80_HGC = (BASE / ".." / "3_GA_Surrogate" / "FEASIBLE_PACKAGE" / "hgc").resolve()
MANUAL_YAML = BASE / "config" / "fuel_types_manual.yaml"
FUEL_STORE = BASE / "data" / "design" / "fuel_types_paramA.parquet"
RESULT_JSON = BASE / "data" / "design" / "realize_lat1600_result.json"

# --------------------------------------------------------------------------- #
# the chosen designs (scratchpad lat1600/chosen.json, 2026-08-11)
# --------------------------------------------------------------------------- #
LAYOUT_N16 = ((1, 1), (5, 2), (5, 5))               # = ga80 E3's proven layout
LAYOUT_N24 = ((1, 1), (4, 1), (5, 5), (6, 3))       # X2-style open layout

#: id -> (FuelDesign, 1/8 Gd layout, role note).  e2 floats are EXACT (never let
#: anything re-derive them from the 1-decimal type_id -- plan hazard 4.8).
CHOSEN: dict[str, tuple[FuelDesign, tuple, str]] = {
    "Y1": (FuelDesign(5.0, 4.25, "z1", 10.0, 16), LAYOUT_N16,
           "E1-role reactivity-matched (68 fresh slots), FF_ens 1.1073"),
    "Y2": (FuelDesign(5.0, 4.25, "z1", 8.0, 24), LAYOUT_N24,
           "E2-role reactivity-matched (hot, 53 slots), FF_ens 1.1409"),
    "Y3": (FuelDesign(5.25, 4.4625, "z1", 6.0, 16), LAYOUT_N16,
           "flat anchor, E3 layout (surrogate-truth check), FF_ens 1.1012"),
    "Y4": (FuelDesign(5.3, 4.505, "z1", 6.0, 16), LAYOUT_N16,
           "flat extreme at the 1600 gate, FF_ens 1.1011"),
}

# octant-map cell ids (verified against 0_APR1400/5.8_5.1/FA/IGD_16/8_16_z1/
# dec_FA_A03.inp:80-97 -- cell 1 UO2, 2 UO2_2 zoning, 3 UO2G, 6-9 guide tubes)
UO2_ID, ZON_ID, GD_ID = 1, 2, 3
GT_IDS = {6, 7, 8, 9}
GT_POSITIONS = {(0, 0), (3, 3), (4, 3), (4, 4)}     # frozen (chosen.json contract)

SNAPSHOT_MSG = """\
[STOP] library rebuild not armed.  assemble_package rotates MAS_XSL/MAS_HFF into
their SINGLE .bak generation (library.py:95-101) -- a rebuild without a snapshot
destroys the only rollback copy.  Take the snapshot, then re-run with --snapshot-ok:

  Copy-Item data\\design\\package\\lib data\\design\\package\\lib.snap_20260811 -Recurse
  Copy-Item data\\design\\package\\designs.json data\\design\\package\\designs.json.snap_20260811
  Copy-Item data\\design\\package\\registry.json data\\design\\package\\registry.json.snap_20260811
  Copy-Item data\\design\\fuel_types_paramA.parquet data\\design\\fuel_types_paramA.parquet.snap_20260811
"""


# --------------------------------------------------------------------------- #
# octant-map parsing / authoring
# --------------------------------------------------------------------------- #
def _triangle(text: str) -> tuple[list[str], list[list[int]], list[int]]:
    """Split deck text; return (lines, 8-row lower-triangle, line indices)."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if toks and toks[0].lower() == "assembly":
            start = i
            break
    if start is None:
        raise SystemExit("[ERROR] deck has no 'assembly' card")
    rows: list[list[int]] = []
    idx: list[int] = []
    for j in range(start + 1, len(lines)):
        toks = lines[j].split()
        if toks and all(t.isdigit() for t in toks):
            rows.append([int(t) for t in toks])
            idx.append(j)
            if len(rows) == 8:
                break
        elif rows:
            break
    if [len(r) for r in rows] != list(range(1, 9)):
        raise SystemExit(
            f"[ERROR] assembly octant triangle is not rows 1..8: {[len(r) for r in rows]}")
    return lines, rows, idx


def _census(rows: list[list[int]], want) -> set[tuple[int, int]]:
    wanted = want if isinstance(want, set) else {want}
    return {(i, j) for i, r in enumerate(rows) for j, v in enumerate(r) if v in wanted}


def _n_gd_of(positions) -> int:
    """1/8 -> full-map multiplicity: diagonal 4x, off-diagonal 8x
    (octant->quarter->full truth: 6_DeCART_Surrogate/surrogate/features.py:85-99)."""
    return sum(4 if i == j else 8 for i, j in positions)


def _layout_str(positions) -> str:
    return ";".join(f"{i}:{j}" for i, j in sorted(positions))


def author_template(base_deck: Path, layout, n_gd: int, out_dir: Path) -> Path:
    """Copy ``base_deck`` with ONLY the Gd cell ids moved to ``layout``.

    Hard guards: guide tubes byte-frozen at GT_POSITIONS, zoning (cell 2) census
    unchanged, final Gd census == layout, multiplicity == n_gd.  Everything else
    (materials, geometry, zoning arrangement, OPTION/DEPL/BRANCH blocks) is
    byte-identical; lattice.edit_dec_text later sets e1/e2/gd_wt numerically.
    """
    text = _read_text_flex(base_deck)
    lines, rows, idx = _triangle(text)
    target = {tuple(p) for p in layout}
    if _n_gd_of(target) != n_gd:
        raise SystemExit(f"[ERROR] layout {_layout_str(target)} realizes "
                         f"{_n_gd_of(target)} Gd pins, design wants {n_gd}")
    gt_before = _census(rows, GT_IDS)
    if gt_before != GT_POSITIONS:
        raise SystemExit(f"[ERROR] base template guide tubes at {sorted(gt_before)}, "
                         f"expected {sorted(GT_POSITIONS)}: {base_deck}")
    zon_before = _census(rows, ZON_ID)

    new = [r[:] for r in rows]
    for (i, j) in _census(rows, GD_ID):
        new[i][j] = UO2_ID
    for (i, j) in target:
        if new[i][j] != UO2_ID:
            raise SystemExit(f"[ERROR] Gd target {i}:{j} would overwrite cell id "
                             f"{new[i][j]} (guide tube or zoning) -- illegal layout")
        new[i][j] = GD_ID

    if _census(new, ZON_ID) != zon_before:
        raise SystemExit("[ERROR] zoning census changed -- authoring bug")
    if _census(new, GT_IDS) != GT_POSITIONS:
        raise SystemExit("[ERROR] guide-tube census changed -- authoring bug")
    if _census(new, GD_ID) != target:
        raise SystemExit("[ERROR] Gd census != target -- authoring bug")

    for k, li in enumerate(idx):
        lines[li] = "  ".join(str(v) for v in new[k])
    out_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "dec_FA_lat1600.inp"
    dst.write_text(out_text, encoding="utf-8")
    return dst


def build_template_tree(selected: dict) -> dict:
    """Author every template dir resolve_template needs for ``selected``.

    Dir names follow lattice._dir_name ``{gd}_{n}_{z}`` under the subtree
    resolve_template searches for that n_gd (lattice.py:32-35): n16 under
    5.8_5.1/FA/IGD_16, n24 under 260624/FA/IGD_24.

    The base is resolved PER DESIGN with the design's OWN gd_wt (adversarial
    review 2026-08-11): the family convention ties the UO2G mixture DENSITY to
    gd_wt (6 -> 10.01, 8 -> 9.95, 10 -> 9.88 g/cc, verified on disk in both
    IGD_16 and IGD_24) and ``edit_dec_text`` never edits density -- a fixed
    base per n_gd would silently realize every lattice with the wrong carrier
    density (~0.6-0.7% off, past the 0.5% threshold predict.py:1069-1074 warns
    shifts Gd burnout timing) while every census/GT/zoning guard still passes.
    The Gd MAP is byte-identical across gd_wt within an (n_gd, z) family, so
    resolving the correctly-densified base and moving pins is safe.
    """
    made: dict[str, dict] = {}
    for yid, (d, layout, _role) in selected.items():
        if d.n_gd not in (16, 24):
            raise SystemExit(f"[ERROR] no base template rule for n_gd={d.n_gd}")
        dirname = f"{int(round(d.gd_wt))}_{d.n_gd}_{d.zoning_variant}"
        subtree = "260624/FA" if d.n_gd == 24 else "5.8_5.1/FA"
        out_dir = TEMPLATE_ROOT / subtree / f"IGD_{d.n_gd}" / dirname
        base = resolve_template(
            FuelDesign(5.8, 5.1, d.zoning_variant, d.gd_wt, d.n_gd), APR1400)
        dst = author_template(base, layout, d.n_gd, out_dir)
        made.setdefault(dirname, {
            "base_template": str(base),
            "authored_deck": str(dst),
            "gd_layout_octant": _layout_str({tuple(p) for p in layout}),
            "serves": [],
        })["serves"].append(f"{yid} ({d.type_id})")
    return made


# --------------------------------------------------------------------------- #
# main chain
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--designs", default="Y1,Y2,Y3,Y4",
                    help="comma list among Y1,Y2,Y3,Y4 (fallback: 'Y3,Y4' for the "
                         "2-type ncomp-40 build, plan 4.3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="author+verify templates and print the alias plan; "
                         "no DeCART, no package mutation")
    ap.add_argument("--snapshot-ok", action="store_true",
                    help="confirm the lib/+manifests snapshot exists (the rebuild "
                         "rotates the SINGLE MAS_XSL/MAS_HFF .bak generation)")
    ap.add_argument("--decart-exe", default=DEFAULT_DECART_EXE)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=5400.0)
    args = ap.parse_args()

    ids = [s.strip() for s in args.designs.split(",") if s.strip()]
    unknown = [i for i in ids if i not in CHOSEN]
    if unknown or not ids:
        raise SystemExit(f"[ERROR] --designs must pick from {sorted(CHOSEN)}, got {ids}")
    selected = {i: CHOSEN[i] for i in ids}

    for p, what in ((PKG / "registry.json", "package registry"),
                    (PKG / "designs.json", "package manifest"),
                    (PKG / "lib" / "MAS_XSL", "package library"),
                    (APR1400, "0_APR1400 template tree")):
        if not p.exists():
            raise SystemExit(f"[ERROR] {what} missing: {p}")

    # -- 1. author the custom-layout templates ------------------------------ #
    made = build_template_tree(selected)
    print("[templates]")
    print(json.dumps(made, indent=2))

    # -- 2. package registry (NEVER data/design/registry.json -- diverged) --- #
    registry = DesignRegistry.load(PKG / "registry.json")
    n_before = len(registry)
    aliases = {yid: registry.alias(d) for yid, (d, _l, _r) in selected.items()}
    print(f"[registry] {n_before} existing types; assigned: "
          + ", ".join(f"{y}->{a}" for y, a in aliases.items()))

    packaged = json.loads((PKG / "designs.json").read_text(encoding="utf-8"))
    packaged_recs = packaged.get("designs", [])
    n_union = len({r.get("alias") for r in packaged_recs
                   if (PKG / "hgc" / f"FA_{r.get('alias')}.HGC").is_file()}
                  | set(aliases.values()))
    print(f"[plan] union library: {n_union} fuel COMP -> expected ncomp {n_union + 5}")

    if args.dry_run:
        print("[dry-run] no DeCART launched, package untouched (registry NOT saved).")
        return
    if not args.snapshot_ok:
        raise SystemExit(SNAPSHOT_MSG)

    registry.save(PKG / "registry.json")        # crash-safe (curriculum.py:1536)

    # -- 3. DeCART 4-way (idempotent-skip on valid FA_<alias>.HGC) ----------- #
    designs = [selected[i][0] for i in ids]
    t0 = time.monotonic()
    runs = run_batch(designs, WORK, registry, TEMPLATE_ROOT, exe=args.decart_exe,
                     max_parallel=args.max_parallel, timeout_s=args.timeout)
    wall = time.monotonic() - t0
    bad = [r for r in runs if r.hgc_path is None]
    for r in runs:
        state = f"OK {r.hgc_path.name}" if r.hgc_path else f"FAIL {r.error}"
        print(f"[decart] {r.alias}  wall={r.wall_s and f'{r.wall_s:.0f}s'}  {state}")
    if bad:
        raise SystemExit(f"[ERROR] {len(bad)} DeCART run(s) failed -- library NOT rebuilt")

    # Gd census on the PRODUCT (the authoritative layout is the HGC %DIST --
    # plan hazard 4.8); deck-level positions were already guard-checked above.
    for r in runs:
        n = count_gd_pins_from_hgc(r.hgc_path)
        if n != r.design.n_gd:
            raise SystemExit(f"[ERROR] {r.alias} HGC Gd census {n} != design "
                             f"n_gd {r.design.n_gd} -- wrong pin map, DO NOT ship")
        print(f"[census] {r.alias} HGC Gd pins = {n} (expected {r.design.n_gd}) OK")

    # -- 4. UNION merge: new sources win, packaged fill in ------------------- #
    # (curriculum.py:1557-1577: build_master_library's stale-HGC guard refuses a
    #  request that omits an HGC already staged in lib/.)
    by_alias: dict[str, DesignSource] = {
        r.alias: DesignSource(design=r.design, alias=r.alias,
                              hgc_path=r.hgc_path, out_path=r.out_path)
        for r in runs}
    for rec in packaged_recs:
        al = rec.get("alias")
        if not al or al in by_alias:
            continue
        hgc = PKG / "hgc" / f"FA_{al}.HGC"
        if not hgc.is_file():
            continue
        try:
            prev = FuelDesign.from_dict(rec)
        except (KeyError, ValueError, TypeError):
            continue
        out = PKG / "hgc" / f"FA_{al}.out"
        by_alias[al] = DesignSource(design=prev, alias=al, hgc_path=hgc,
                                    out_path=out if out.is_file() else None)
    sources = list(by_alias.values())
    print(f"[union] rebuilding over {len(sources)} designs "
          f"({len(sources) - len(runs)} packaged + {len(runs)} new)")

    # -- 5. manifest + registry + hgc staging + TotalBatcher rebuild --------- #
    build = assemble_package(PKG, sources, registry, APR1400)
    missing = [f"FA_{a}" for a in aliases.values()
               if f"FA_{a}" not in build.set_names]
    if missing:
        raise SystemExit(f"[ERROR] rebuilt library is missing {missing}")
    print(f"[library] {build.comp_count} COMP + {build.refl_count} REFL "
          f"-> ncomp {build.ncomp}  ({build.xsl_path})")

    # -- 6. layout provenance into designs.json (extra keys are tolerated by
    #       paramA_rows / FuelDesign.from_dict readers) ----------------------- #
    doc = json.loads((PKG / "designs.json").read_text(encoding="utf-8"))
    by_new_alias = {aliases[y]: (y, selected[y]) for y in selected}
    for rec in doc.get("designs", []):
        hit = by_new_alias.get(rec.get("alias"))
        if hit is None:
            continue
        yid, (d, layout, role) = hit
        rec["gd_positions"] = _layout_str({tuple(p) for p in layout})
        rec["lat1600_id"] = yid
        rec["lat1600_role"] = role
        rec["provenance"] = ("realize_lat1600 2026-08-11; open Gd layout (NOT the "
                             "frozen template layout for this (gd,n,z)); CBC gate 1600 ppm")
    (PKG / "designs.json").write_text(json.dumps(doc, indent=2) + "\n",
                                      encoding="utf-8")
    print("[manifest] gd_positions provenance attached to new records")

    # -- 7. fuel_types re-ingest (pathfinder.py:233-234 convention; data/store
    #       is NOT written by this driver) ----------------------------------- #
    base_paths = FuelPaths(apr1400_root=APR1400, ga80_hgc=GA80_HGC,
                           manual_yaml=MANUAL_YAML, store=FUEL_STORE)
    df = ingest_fuel_types(PKG, base_paths=base_paths, store_path=FUEL_STORE)
    paramA = df[df["library_id"] == "paramA"]
    new_type_ids = [selected[y][0].type_id for y in selected]
    got = paramA[paramA["type_id"].isin(new_type_ids)]
    print(f"[ingest] paramA rows total={len(paramA)}  new={len(got)}/{len(new_type_ids)}"
          f"  feature_poor(new)={int(got['feature_poor'].sum()) if len(got) else '?'}"
          f"  -> {FUEL_STORE}")
    if len(got) != len(new_type_ids):
        raise SystemExit(f"[ERROR] ingest missing new types: "
                         f"{sorted(set(new_type_ids) - set(got['type_id']))}")

    result = {
        "date": "2026-08-11",
        "designs": {y: {"type_id": selected[y][0].type_id,
                        "alias": aliases[y],
                        "gd_layout_octant": _layout_str(
                            {tuple(p) for p in selected[y][1]}),
                        "role": selected[y][2]} for y in selected},
        "pairs": {"matched": None, "flat": None},
        "library": {"ncomp": build.ncomp, "comp_count": build.comp_count,
                    "sets": build.set_names},
        "decart_wall_s": wall,
        "templates": made,
        "fuel_store": str(FUEL_STORE),
    }
    if "Y1" in aliases and "Y2" in aliases:
        result["pairs"]["matched"] = f"{aliases['Y1']}_{aliases['Y2']}"
    if "Y3" in aliases and "Y4" in aliases:
        result["pairs"]["flat"] = f"{aliases['Y3']}_{aliases['Y4']}"
    RESULT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[result] {RESULT_JSON}")
    print(json.dumps(result["designs"], indent=2))
    print(f"[pairs] matched={result['pairs']['matched']}  flat={result['pairs']['flat']}")
    print("[next] EVERY existing restart is now stale (ncomp shift). Bootstrap order:")
    print("       1) the matched new pair (= the ncomp-42 MASTER smoke test)")
    print("       2) the flat new pair    3) Q1_Q2    4) Q7_Q8")
    print("       via: python -m lpopt design bootstrap --input design_lat1600_104.inp"
          " --pair <PAIR> --feed 121")


if __name__ == "__main__":
    main()
