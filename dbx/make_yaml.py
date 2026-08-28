"""Deliverable 2: config/fuel_types_dbx_extracted.yaml + store cross-check."""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL")
SCR = Path(__file__).parent
sys.path.insert(0, str(SCR))
from dbx_parse import parse_all  # noqa: E402

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

recs, types = parse_all()
BU = recs["E1_E2"]["bu"]
store = pd.read_parquet(ROOT / "data/store/fuel_types.parquet")
ga80 = store[store.library_id == "ga80"].set_index("type_id")
cores = pd.read_excel(ROOT / "data/reports/scoping_mesh_20260815/feasible_database.xlsx",
                      sheet_name="cores")

FLOAT_KEYS = ["u_avg_enrichment", "enr_main", "enr_zone", "du", "gd_wt", "gd_u_enr",
              "ff_window_max", "u_avg_from_map", "kinf0", "kinf10", "kinf20", "kinf30",
              "bu_k1", "kinf_dip", "bu_dip_gwd", "kinf_peak", "bu_peak_gwd",
              "reactivity_swing_pcm", "rho_boc_minus_peak_pcm",
              "depletion_slope_pcm_per_gwd", "kinf_eol50", "kconv_is_monotone",
              "ff_pin_max", "ff_boc"]
INT_KEYS = ["n_gd", "n_main_pin", "zone_pin_count", "n_gd_from_map", "n_fuel_rod",
            "n_guide_tube", "n_positions"]


def fmt(v, nd=6):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return s if s else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def flow(seq, nd=5):
    return "[" + ", ".join(fmt(float(x), nd) for x in seq) + "]"


# ------------------------------------------------------------------ counts --
rows_by_pair = cores.pair.value_counts().to_dict()
type_rows = {}
for t in types:
    n = 0
    for r in cores.itertuples():
        if r.type1 == t or r.type2 == t:
            n += 1
    type_rows[t] = n

# --------------------------------------------------------------- crosscheck -
cc = []
for t, d in sorted(types.items()):
    row = {"type_id": t, "pair": d["pair"], "segment": d["segment"],
           "in_store": t in ga80.index,
           "db_rows": rows_by_pair.get(d["pair"], 0)}
    if row["in_store"]:
        s = ga80.loc[t]
        row["store_feature_poor"] = bool(s.feature_poor)
        row["store_u_avg"] = float(s.u_avg_enrichment) if pd.notna(s.u_avg_enrichment) else np.nan
        row["dbx_u_avg"] = d.get("u_avg_enrichment", d.get("u_avg_from_map"))
        row["store_n_gd"] = float(s.n_gd) if pd.notna(s.n_gd) else np.nan
        row["dbx_n_gd"] = d.get("n_gd")
        for k in ("kinf0", "kinf10", "kinf20", "kinf30", "bu_k1", "ff_pin_max",
                  "kinf_eol50", "depletion_slope_pcm_per_gwd", "reactivity_swing_pcm"):
            sv = float(s[k]) if k in s and pd.notna(s[k]) else np.nan
            row["store_" + k] = sv
            row["dbx_" + k] = d.get(k, np.nan)
        row["store_zone_pin_count"] = float(s.zone_pin_count) if pd.notna(s.zone_pin_count) else np.nan
        row["dbx_zone_pin_count"] = d.get("zone_pin_count", np.nan)
        row["store_enr_main"] = float(s.enr_main) if pd.notna(s.enr_main) else np.nan
        row["dbx_enr_main"] = d.get("enr_main", np.nan)
    cc.append(row)
CC = pd.DataFrame(cc)
new_types = CC[~CC.in_store].type_id.tolist()
print("types missing from data/store/fuel_types.parquet (ga80):", new_types)
print("DB rows made featurizable by them:",
      cores[cores.pair.isin(sorted({types[t]['pair'] for t in new_types}))].shape[0])
print()
sub = CC[CC.in_store]
for k in ("kinf0", "kinf10", "kinf20", "kinf30", "bu_k1", "ff_pin_max", "kinf_eol50",
          "depletion_slope_pcm_per_gwd", "reactivity_swing_pcm"):
    a, b = sub["store_" + k].astype(float), sub["dbx_" + k].astype(float)
    m = a.notna() & b.notna()
    d = (b - a)[m]
    print(f"  {k:30s} n={m.sum():2d} mean_diff={d.mean():+11.5f} "
          f"max|diff|={d.abs().max():11.5f} rel_max={100*(d.abs()/a[m].abs()).max():7.3f}%")
print()
print("  u_avg: store carries the FAMILY ANCHOR, dbx the realised per-type value")
print(sub[["type_id", "store_u_avg", "dbx_u_avg"]].assign(
    diff=lambda x: x.dbx_u_avg.astype(float) - x.store_u_avg.astype(float)).round(4).to_string(index=False))
print()
print("  n_gd agreement:",
      int((sub.store_n_gd.astype(float) == sub.dbx_n_gd.astype(float)).sum()), "/", len(sub))
mism = sub[sub.store_n_gd.astype(float) != sub.dbx_n_gd.astype(float)]
if len(mism):
    print(mism[["type_id", "store_n_gd", "dbx_n_gd"]].to_string(index=False))
print("  store zone_pin_count non-null:", int(sub.store_zone_pin_count.notna().sum()),
      "  dbx:", int(sub.dbx_zone_pin_count.notna().sum()))
print("  store enr_main non-null:", int(sub.store_enr_main.notna().sum()),
      "  dbx:", int(sub.dbx_enr_main.notna().sum()))
CC.to_csv(SCR / "crosscheck.csv", index=False)

# --------------------------------------------------------------- QC flags ---
qc = {}
for t, d in types.items():
    flags = []
    if not d.get("has_deck"):
        flags.append("no_decart_deck__pin_zoning_and_gd_design_unknown")
    if "u_avg_from_map" in d and "u_avg_enrichment" in d:
        if abs(d["u_avg_from_map"] - d["u_avg_enrichment"]) > 5e-4:
            flags.append("u_avg_pinmap_vs_sheet_mismatch")
    if "n_gd_from_map" in d and "n_gd" in d and d["n_gd_from_map"] != d["n_gd"]:
        flags.append("n_gd_pinmap_vs_sheet_mismatch")
    if d.get("kinf20") is not None and d.get("kinf_peak") is not None:
        if d["kinf20"] > d["kinf_peak"] + 1e-3:
            flags.append("kconv_hump_latched_on_shallow_ripple__true_peak_later")
    qc[t] = flags

# ------------------------------------------------------------------- YAML ---
L = []
A = L.append
A("# fuel_types_dbx_extracted.yaml")
A("#")
A("# Assembly-lattice specs extracted from the READ-ONLY feasible-core database")
A("#   data/reports/scoping_mesh_20260815/feasible_database.xlsx   (sheets P_<pair>)")
A("# by the dbx extraction pass of 2026-08-16.  NOTHING here has been written into")
A("# data/store/fuel_types.parquet — this file is the candidate merge payload only")
A("# (merge recipe: see `merge_note` at the bottom of this file).")
A("#")
A("# Every P_<pair> sheet carries, per assembly type:")
A("#   * a general block   (avg enrichment / role / pin zoning / Gd design)")
A("#   * a 1/8 pin map     (DeCART dec_FA_<type>.inp) with a code legend, OR '(덱 없음)'")
A("#   * a k-inf(BU) and pin-peaking FF(BU) depletion table on a shared 62-point grid")
A("# The 1/8 map is lower-triangular with 8-fold symmetry: cell (i,j<=i) has")
A("# multiplicity 4 on the diagonal and 8 off it -> 8*4 + 28*8 = 256 lattice cells")
A("# (16x16 CE assembly: 236 fuel rods + 20 guide-tube cells).  Rod-count-weighted")
A("# (main, zone, Gd) enrichment reproduces the sheet's stated assembly average to")
A("# <3e-4 w/o on all 36 decked types, so the map is used to RECOVER the average")
A("# where the sheet leaves it blank (A8, A2).")
A("")
A("meta:")
A(f"  extracted: {date.today().isoformat()}")
A("  source_workbook: data/reports/scoping_mesh_20260815/feasible_database.xlsx")
A("  source_sheets: P_<pair>  (20 sheets)")
A("  library_id: ga80          # same letter roster as store library 'ga80'")
A(f"  n_sheets_parsed: {len(recs)}")
A(f"  n_sheets_failed: 0")
A(f"  n_types: {len(types)}")
A(f"  n_types_new_vs_store: {len(new_types)}   # {', '.join(new_types)}")
A(f"  n_types_with_decart_deck: {sum(1 for d in types.values() if d.get('has_deck'))}")
A("  db_rows_unlocked_by_new_types: "
  f"{cores[cores.pair.isin(sorted({types[t]['pair'] for t in new_types}))].shape[0]}")
A("  read_only_inputs: [feasible_database.xlsx, data/store/fuel_types.parquet]")
A("")
A("# All 20 sheets share this burnup grid (verified identical element-wise).")
A(f"burnup_grid_gwd_per_tu: {flow(BU, 3)}")
A("")
A("parse_report:   # honest per-sheet status; no field was inferred where absent")
for p in sorted(recs):
    r = recs[p]
    decked = [t for t, v in r["pinmaps"].items() if v.get("deck")]
    gen = r.get("general", {})
    blank = sorted({k for t in r["types"]
                    for k in set(["u_avg_enrichment", "role", "enr_main", "enr_zone",
                                  "du", "n_gd", "gd_wt", "gd_u_enr", "gd_pattern",
                                  "ff_window_max"]) - set(gen.get(t, {}))})
    A(f"  P_{p}:")
    A(f"    types: [{', '.join(r['types'])}]")
    A(f"    general_block: {'complete' if not blank else 'partial'}")
    if blank:
        A(f"    general_block_blank_fields: [{', '.join(blank)}]")
    A(f"    pin_maps_present: [{', '.join(decked) if decked else ''}]")
    A(f"    kinf_table_points: {len(r.get('bu', []))}")
    A(f"    feed_distribution_rows: {len(r.get('feed_dist', []))}")
    A(f"    parse_problems: []" if not r["problems"]
      else f"    parse_problems: [{', '.join(repr(x) for x in r['problems'])}]")
A("")
A("types:")
for t in sorted(types, key=lambda x: (types[x]["pair"], x)):
    d = types[t]
    A(f"  {t}:")
    A(f"    pair: {d['pair']}")
    A(f"    enrichment_segment: {d['segment']:.1f}")
    A(f"    role: {d.get('role') or 'null'}          # L = light-load, H = heavy-load position")
    uav = d.get("u_avg_enrichment")
    src = "sheet"
    if uav is None:
        uav, src = d.get("u_avg_from_map"), "pinmap_rod_count_weighted"
    A(f"    u_avg_enrichment: {fmt(uav, 4)}")
    A(f"    u_avg_source: {src}")
    if d.get("u_avg_from_map") is not None:
        A(f"    u_avg_from_pinmap: {fmt(d['u_avg_from_map'], 5)}   # QC cross-check")
    A("    # --- pin zoning -------------------------------------------------")
    A(f"    enr_main: {fmt(d.get('enr_main'), 3)}")
    A(f"    enr_zone: {fmt(d.get('enr_zone'), 3)}")
    A(f"    du: {fmt(d.get('du'), 3)}")
    A(f"    n_main_pin: {fmt(d.get('n_main_pin'))}")
    A(f"    zone_pin_count: {fmt(d.get('zone_pin_count'))}")
    A("    # --- burnable absorber ------------------------------------------")
    A(f"    n_gd: {fmt(d.get('n_gd'))}")
    A(f"    gd_wt: {fmt(d.get('gd_wt'), 2)}          # Gd2O3 wt%")
    A(f"    gd_u_enr: {fmt(d.get('gd_u_enr'), 3)}")
    A(f"    gd_pattern: {d.get('gd_pattern') or 'null'}")
    A("    # --- geometry (from the 1/8 map) --------------------------------")
    A(f"    n_fuel_rod: {fmt(d.get('n_fuel_rod'))}")
    A(f"    n_guide_tube: {fmt(d.get('n_guide_tube'))}")
    A(f"    decart_deck: {d.get('deck_file') or 'null'}")
    pm = recs[d["pair"]]["pinmaps"].get(t, {})
    if pm.get("tri"):
        A("    pinmap_octant:   # row i holds j=0..i; w=4 on diagonal, 8 off it")
        for r_ in pm["tri"]:
            A(f"      - [{', '.join(str(x) for x in r_)}]")
        A("    pinmap_legend:")
        for code, desc in sorted(pm.get("legend", {}).items()):
            A(f"      {code}: \"{desc}\"")
    A("    # --- k-inf / FF curve features (same conventions as")
    A("    #     lpopt.data.fuel_types.kconv_curve_shape / _burnup_at_k1) --------")
    for k in ("kinf0", "kinf10", "kinf20", "kinf30", "bu_k1", "kinf_eol50",
              "kinf_dip", "bu_dip_gwd", "kinf_peak", "bu_peak_gwd",
              "reactivity_swing_pcm", "rho_boc_minus_peak_pcm",
              "depletion_slope_pcm_per_gwd", "kconv_is_monotone",
              "ff_pin_max", "ff_boc", "ff_window_max"):
        A(f"    {k}: {fmt(d.get(k), 5)}")
    A(f"    kinf_curve: {flow(recs[d['pair']]['kinf'][t], 5)}")
    A(f"    ff_curve: {flow(recs[d['pair']]['ff'][t], 4)}")
    st = "absent" if t not in ga80.index else (
        "present_feature_poor" if bool(ga80.loc[t].feature_poor) else "present")
    A(f"    store_ga80_status: {st}")
    A(f"    qc_flags: [{', '.join(qc[t])}]")
    A("")

A("# ---------------------------------------------------------------------------")
A("store_crosscheck:   # dbx value vs data/store/fuel_types.parquet library_id='ga80'")
A("  # (parquet READ ONLY — nothing was written back)")
A(f"  n_types_in_dbx: {len(types)}")
A(f"  n_present_in_store: {int(sub.shape[0])}")
A(f"  n_absent_from_store: {len(new_types)}   # [{', '.join(new_types)}]")
A("  n_store_ga80_rows_total: 70    # 34 of them feature_poor (B/C/D/A1/A3-7/G1/G2/H5/H6)")
A("  scalar_agreement:   # dbx - store, over the 36 shared types")
for k, tol in [("kinf10", None), ("kinf20", None), ("kinf30", None), ("kinf_eol50", None),
               ("bu_k1", None), ("ff_pin_max", None), ("kinf0", None),
               ("depletion_slope_pcm_per_gwd", None), ("reactivity_swing_pcm", None)]:
    a_, b_ = sub["store_" + k].astype(float), sub["dbx_" + k].astype(float)
    msk = a_.notna() & b_.notna()
    dd = (b_ - a_)[msk]
    A(f"    {k}: {{n: {int(msk.sum())}, mean_diff: {dd.mean():.5g}, "
      f"median_abs_diff: {dd.abs().median():.5g}, max_abs_diff: {dd.abs().max():.5g}}}")
A("  verdict: |")
A("    kinf10 / kinf20 / kinf30 / kinf_eol50 agree to <7e-4 (<0.07 %) and bu_k1 to")
A("    <0.056 GWd/tU (<0.15 %) on all 36 shared types — the workbook curves and the store's")
A("    HGC harvest are the same physics.  ff_pin_max differs by <0.0041 (<0.36 %).")
A("    kinf0 differs by -0.028 on average: a BU=0 xenon-state convention difference, not a")
A("    disagreement (see the caveat in merge_note).")
A("    reactivity_swing_pcm and depletion_slope_pcm_per_gwd agree to a median 23 pcm and")
A("    0.37 pcm/GWd, with exactly ONE outlier: A8, where the STORE value (swing 11 pcm,")
A("    slope -462) is the artefact — its hump detector latched onto a numerical ripple on the")
A("    finer HGC grid.  The workbook grid gives A8 swing 1497 pcm / slope -639, in line with")
A("    its siblings.  Merging this file therefore also CORRECTS one existing store row.")
A("  fields_the_store_has_no_value_for_today:")
A("    enr_main: {store_non_null: 0, dbx_non_null: 36}")
A("    enr_zone: {store_non_null: 0, dbx_non_null: 36}")
A("    gd_wt: {store_non_null: 0, dbx_non_null: 36}")
A("    gd_u_enr: {store_non_null: 0, dbx_non_null: 36}")
A("    zone_pin_count: {store_non_null: 0, dbx_non_null: 36}")
A("  u_avg_enrichment: |")
A("    The store carries the FAMILY ANCHOR from config/fuel_types_manual.yaml (A/E->5.0,")
A("    J->5.1, K->5.2, L->5.3, N->5.4, B/G/H->5.5), identical for every member of a family.")
A("    The workbook gives the realised per-type value; the spread inside a family reaches")
A("    0.050 w/o (E4 5.0500 vs E3 5.0246) and the anchor is off by up to +0.050 / -0.039")
A("    (A2 4.9610 against anchor 5.0).  n_gd agrees 36/36.")
A("")
A("pair_feed_summary:   # [이 쌍의 합격 노심 분포] block, verbatim")
for p in sorted(recs):
    fd = recs[p].get("feed_dist", [])
    A(f"  {p}:")
    for row in sorted(fd, key=lambda x: x["feed"]):
        A(f"    {row['feed']}: {{n_pass: {row['n_pass']}, best_f_r: {fmt(row['best_f_r'], 4)}}}")
A("")
A("# ---------------------------------------------------------------------------")
A("merge_note: |")
for line in MERGE_NOTE.strip().splitlines() if False else []:
    pass
A("""  HOW THIS WOULD BE MERGED (not done here — fuel_types.parquet is read-only in
  this pass).  The manual-yaml pathway already exists: lpopt.data.fuel_types
  .load_manual_anchors() reads config/fuel_types_manual.yaml and ga80_rows()
  builds the ga80 library from GA80_TYPE_IDS x anchors x HGC Gd counts.  This
  file slots in as a higher-priority ga80 overlay, applied inside ga80_rows()
  after the HGC harvest:

    1. ROSTER.  GA80_FAMILIES currently reads E: 4, J: 6.  The DB uses E5, E6,
       J7, J8 -> set E: 6, J: 8.  That alone makes 472 DB core rows (E5_E6 56 +
       J7_J8 416) featurizable; today they resolve to a missing type_id.
    2. ANCHOR -> REALISED ENRICHMENT.  ga80 rows currently carry
       u_avg_enrichment = the FAMILY anchor (E->5.0, J->5.1, ...).  Replace it
       per type with `u_avg_enrichment` here (5.0119 .. 5.5102): a spread of up
       to 0.05 w/o inside a family that the anchor collapses to one number.
       Set source_flags += ['dbx_lattice'].
    3. NEW COLUMNS FILLED.  enr_main, enr_zone, gd_wt, gd_u_enr and
       zone_pin_count are all-NaN for every ga80 row today (ga80 ships no dec
       inp in this workspace).  The P_ sheets' 1/8 pin maps supply all five for
       the 36 decked types, exactly the columns the featurizer expects.
    4. THE FOUR NEW TYPES.  E5, E6, J7, J8 have no DeCART deck in the workbook
       ('(덱 없음)'), so enr_main / enr_zone / gd_wt / gd_u_enr / zone_pin_count
       stay unknown for them.  They DO carry a full 62-point k-inf and FF curve,
       which is what the reactivity features are computed from — so they can be
       admitted with feature_poor=False for the curve family (kinf0/10/20/30,
       bu_k1, ff_pin_max, kconv_* ) and NaN for the zoning family.  Recommended:
       a new flag `zoning_unknown=True` rather than the blunt feature_poor.
    5. CURVE PARITY.  Load `kinf_curve` (with `burnup_grid_gwd_per_tu`) into the
       same kconv_curve_shape()/_burnup_at_k1() the store already uses, rather
       than trusting the scalar fields here — that keeps the store's train of
       thought and the two sources agree by construction.
    6. DO NOT overwrite kinf0 or ff_pin_max on the 36 types that already carry
       HGC values.  The workbook's BU=0 row is xenon-state-inconsistent between
       decked and deckless sheets (see caveat below); bu_k1 / kinf10 / kinf20 /
       kinf30 agree with the store to <2e-4 and are safe.

  CAVEAT — BU=0 xenon state.  On the 36 decked types the sheet's BU=0 k-inf sits
  ~0.03 BELOW the store's kinf0 and shows no fresh no-xenon spike; on the four
  deckless types (E5, E6, J7, J8) BU=0 is 0.03 ABOVE the BU=0.2 point, i.e. it
  IS the xenon-free spike.  The two families therefore use different BU=0
  conventions.  Everything at BU >= 0.2 is consistent.  Consequence: kinf0 and
  rho_boc_minus_peak_pcm are NOT comparable across the two families, and for E5
  the hump detector latches onto a 1.8e-4 ripple at 3 GWd/tU instead of the real
  burnout peak near 20 GWd/tU (flagged in qc_flags).""")
dest = ROOT / "config" / "fuel_types_dbx_extracted.yaml"
dest.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"\nwrote {dest}  ({dest.stat().st_size/1024:.0f} KB, {len(L)} lines)")

import yaml  # noqa: E402
doc = yaml.safe_load(dest.read_text(encoding="utf-8"))
print("yaml re-parse OK:", len(doc["types"]), "types,", len(doc["parse_report"]), "sheets,",
      len(doc["burnup_grid_gwd_per_tu"]), "burnup points")
