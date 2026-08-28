"""Fixed-loading-pattern F_r attribution experiment (flat-assembly hypothesis).

ONE loading pattern (the flattest feasible core in the store, record
b0ff11ef16de..., pair E1_E2, feed 121, ga80) is evaluated with SEVERAL FUEL SETS.
Only the two fresh batch identities change; every shuffle card, the feed (121),
the symmetry class and the equilibrium protocol are byte-identical across arms.
Any F_r difference is therefore attributable to the assembly pin-power form
function FF, which is exactly the quantity the DeCART surrogate predicts.

Usage (MASTER is expensive -- the issuer schedules these, not the agent):

    python fr_arms.py --list
    python fr_arms.py --arm A0 --package ../3_GA_Surrogate/FEASIBLE_PACKAGE \
        --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe --run-dir runs/fr_arms --workers 12

Each arm is one equilibrium chain (<= max-cycles MASTER runs).  Results are
appended to <run-dir>/fr_arms_results.jsonl -- nothing is written to data/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

# record_id prefix of the flattest feasible core in the store
FLATTEST = "b0ff11ef16de"

# arm -> (pair, {original fresh batch -> substituted fresh batch})
# A0 is the control: identity substitution, must reproduce F_r 1.5207.
ARMS = {
    # ---- tier 0: ga80 / FEASIBLE_PACKAGE, no DeCART, no library rebuild ----
    "A0": ("E1_E2", {"E1": "E1", "E2": "E2"}),   # control
    "A1": ("E3_E4", {"E1": "E3", "E2": "E4"}),   # flatter fuel   (FF 1.101/1.139)
    "A2": ("A8_A2", {"E1": "A8", "E2": "A2"}),   # peakier fuel   (FF 1.157/1.178)
    # ---- tier 1: paramA package, requires the 4 new lattices + bootstraps ----
    "B0": ("Q1_Q2", {"E1": "Q1", "E2": "Q2"}),   # paramA control (FF 1.122/1.174)
    "B1": ("Q7_Q8", {"E1": "Q7", "E2": "Q8"}),   # paramA peaky   (FF 1.205/1.209)
    "B2": ("T3_T4", {"E1": "T3", "E2": "T4"}),   # lat1600 matched pair (FF 1.107/1.141) — BLOCKED until T3_T4 bootstrap
    "B3": ("T5_T6", {"E1": "T5", "E2": "T6"}),   # lat1600 flat pair (FF 1.1012/1.1011), CBC gate 1600
    # ---- tier 0b: MORE ga80 dose points, same FEASIBLE_PACKAGE, zero confound ----
    # FF = measured DeCART pin-power form factor at BU=0 (HGC %DIST, = the BU
    # index of a FRESH slot).  All six resolve at fallback_level 0 with native
    # restarts.  ``validate_case`` forbids fresh batches outside the pair, so
    # every arm draws both fresh tokens from ONE base pair.
    # C1-C4 keep the control's role assignment (flat/high-k type -> E1-role,
    # peaky/low-k type -> E2-role); C5/C6 load ONE type into all 121 fresh
    # slots and are read as their own 2-point series, not against C1-C4.
    "C1": ("H3_H4", {"E1": "H3", "E2": "H4"}),   # FF 1.117/1.171 -> F_r 1.5458
    "C2": ("K5_K6", {"E1": "K5", "E2": "K6"}),   # FF 1.118/1.149 -> F_r 1.5168
    "C3": ("L3_L4", {"E1": "L3", "E2": "L4"}),   # FF 1.115/1.145 -> F_r 1.5115
    "C4": ("J1_J2", {"E1": "J1", "E2": "J2"}),   # FF 1.113/1.146 -> F_r 1.5128
    "C5": ("E3_E4", {"E1": "E3", "E2": "E3"}),   # FF 1.101/1.101 -> F_r 1.4534
    "C6": ("A8_A2", {"E1": "A2", "E2": "A2"}),   # FF 1.178/1.178 -> F_r 1.5551
    # ---- STEP 0 of kcurve_fusion_memo_20260809.md: the H-family NULL TEST ----
    # Pre-registered 2026-08-09, BEFORE the chains ran.  Tests whether the
    # (k-inf curve, enrichment) pair is a SUFFICIENT assembly statistic.
    #
    # H2 and H4 are the same assembly under that hypothesis: enrichment 5.5 w/o
    # both, 24 Gd rods both, and their k-inf curves agree to <= 57 pcm at every
    # stored burnup (kinf0 1.12022/1.11951, kinf10 1.09236/1.09170,
    # kinf20 1.11293/1.11336, kinf30 1.08469/1.08409, bu_k1 43.04/42.93,
    # kinf_eol50 0.95703/0.95629).  What DOES differ is the pin-level shape:
    # adf_face_g2 1.08941 vs 1.12447 (+3.2%), adf_corner_g2 +0.0245,
    # ff_pin_max 1.143 vs 1.171.
    # H1/H3 match on the k-curve AND on the ADF (d adf_face_g2 = 0.0003,
    # d ff_pin_max = 0.002), so they separate "the ADF is what matters" from
    # "any two types differ".
    #
    # Single type in all 121 fresh slots (the C5/C6 construction) so the
    # E1/E2 role assignment cannot contribute.
    # Decision rule, threshold 0.005 (the F3 resolution floor; the A0 control
    # reproduced to 0.0000, so MASTER is deterministic here):
    #   REFUTES sufficiency : |d node_peak(D1,D2)| > 0.005 AND |d(D3,D4)| <= 0.005
    #   SUPPORTS            : both |d| <= 0.005
    #   VOID                : the negative control fires, |d(D3,D4)| > 0.005
    # Null prediction under sufficiency: scaling the C5/C6 dose response
    # (d kinf0 ~ 6800 pcm -> d node_peak 0.164) puts H2-H4's 57 pcm at
    # d node_peak ~ 0.0014 -- below the threshold.
    "D1": ("H1_H2", {"E1": "H2", "E2": "H2"}),   # treated      FF 1.143
    "D2": ("H3_H4", {"E1": "H4", "E2": "H4"}),   # treated      FF 1.171
    "D3": ("H1_H2", {"E1": "H1", "E2": "H1"}),   # neg control  FF 1.115
    "D4": ("H3_H4", {"E1": "H3", "E2": "H3"}),   # neg control  FF 1.117
}

#: Portable copy of the reference pattern so a fleet worker (whose kit ships no
#: records.parquet) can run the arms without the store.
PATTERN_FILE = BASE / "fr_arms_pattern.txt"


#: record_id prefix of the lowest-F_r feasible core in the store.  It is a
#: DIFFERENT pattern from FLATTEST and a strictly better one on F_r
#: (1.4636 vs 1.5207) while being LESS flat nodally (node_peak 1.2620 vs
#: 1.2085) -- the two objectives trade off inside the LP space, so the fuel
#: lever has to be measured on both operating points, not just the flat one.
MIN_FR = "deb058c00433"

REFERENCES = {"flat": (FLATTEST, PATTERN_FILE),
              "minfr": (MIN_FR, BASE / "fr_arms_minfr_pattern.txt")}


def load_pattern(reference: str = "flat"):
    import sys
    sys.path.insert(0, str(BASE))
    from lpopt.data.schema import unpack_pattern
    prefix, pattern_file = REFERENCES[reference]
    store = BASE / "data/store/records.parquet"
    if store.is_file():
        df = pd.read_parquet(store)
        row = df[df.record_id.str.startswith(prefix)].iloc[0]
        return unpack_pattern(row["pattern"]), row
    packed = pattern_file.read_text(encoding="utf-8").strip()
    known = {"flat": {"f_r": 1.5207, "node_peak": 1.2085, "cbc_max": 1330.49,
                      "cyclen": 632.276},
             "minfr": {"f_r": 1.4636, "node_peak": 1.2620, "cbc_max": 1326.69,
                       "cyclen": 633.329}}[reference]
    row = pd.Series({"record_id": prefix, "case_pair": "E1_E2", "feed": 121,
                     "pattern": packed, **known})
    return unpack_pattern(packed), row


def substitute(pattern, mapping):
    items = []
    for it in pattern.items:
        if it.is_fresh and it.batch in mapping:
            it = dataclasses.replace(it, batch=mapping[it.batch])
        items.append(it)
    return type(pattern)(tuple(items))


def library_dims(package_root: Path) -> tuple[int, int]:
    xsl = (package_root / "lib" / "MAS_XSL").read_text(errors="replace")
    comp = sum(1 for ln in xsl.splitlines() if ln.startswith("COMP "))
    refl = sum(1 for ln in xsl.splitlines() if ln.startswith("REFL "))
    return (comp + 3, comp + refl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=None, choices=sorted(ARMS))
    ap.add_argument("--package", default="../3_GA_Surrogate/FEASIBLE_PACKAGE")
    ap.add_argument("--exe", default="C:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    ap.add_argument("--run-dir", default="runs/fr_arms")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cycles", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--reference", default="flat", choices=sorted(REFERENCES),
                    help="which stored core supplies the fixed loading pattern: "
                         "'flat' = the flattest feasible core (node_peak 1.2085), "
                         "'minfr' = the lowest-F_r feasible core (F_r 1.4636)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="run an arm even when its pair resolves at "
                         "fallback_level != 0 (the restart is NOT the pair's own, "
                         "so the arm is a confounded fuel measurement)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(BASE))
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.search.verify import WaveVerifier, WaveEntry
    from lpopt.vendor.masterrl.domain import CaseKey

    pattern, row = load_pattern(args.reference)
    pkg = (BASE / args.package).resolve()
    dims = library_dims(pkg)

    if args.list:
        print(f"reference record : {row.record_id[:12]}  pair={row.case_pair} "
              f"feed={row.feed} F_r={row.f_r} node_peak={row.node_peak:.4f} "
              f"CBC={row.cbc_max} cyclen={row.cyclen}")
        print(f"package          : {pkg}  library_dims(nbatch,ncomp)={dims}")
        for arm, (pair, mp) in ARMS.items():
            pat = substitute(pattern, mp)
            print(f"  {arm:3s} pair={pair:8s} feed={pat.feed:4d} "
                  f"batches={pat.batch_feed()}")
        return

    arms = args.arm or ["A0"]
    resolver = CaseAssetResolver(pkg, library_dims=dims)
    verifier = WaveVerifier(
        run_dir=BASE / args.run_dir, package_root=pkg, executable=args.exe,
        workers=args.workers, timeout=args.timeout, max_cycles=args.max_cycles,
        consecutive=2, library_dims=dims, harvest_maps=True,
    )
    entries = []
    for arm in arms:
        pair, mp = ARMS[arm]
        key = CaseKey(pair, int(row.feed))
        assets = resolver.resolve(key)
        print(f"{arm}: {key.label} fallback_level={assets.fallback_level} "
              f"restart={assets.restart_provenance}")
        if assets.fallback_level != 0:
            print("!" * 78)
            print(f"!! WARNING: arm {arm} pair {pair} resolves at "
                  f"fallback_level={assets.fallback_level} ({assets.kind}).")
            if assets.fallback_level < 0:
                print("!! NOTHING resolved -- this pair has no restart and/or no template")
                print("!! deck in the package.  The chain would die at staging.")
            else:
                print("!! The restart is NOT this pair's own.  A fallback restart carries a")
                print("!! different burnt-fuel history into the chain and is a CONFOUND for")
                print("!! the fuel measurement: the whole point of these arms is that ONLY")
                print("!! the fresh batch identities differ across them.")
            if args.allow_fallback:
                print(f"!! arm {arm} RUN ANYWAY (--allow-fallback) -- do NOT report it "
                      f"as a clean fuel-lever measurement.")
            else:
                print(f"!! arm {arm} SKIPPED (pass --allow-fallback to run it anyway).")
            print("!" * 78)
            if not args.allow_fallback:
                continue
        pat = substitute(pattern, mp)
        # Cheap tripwires on the ONE thing that must survive the swap: only batch
        # NAMES may change.  ponytail: the per-batch multiset is NOT compared
        # (fr_transfer.substitute_checked does) because the C5/C6/D-family arms
        # deliberately collapse two batches into one; upgrade path is comparing
        # against the collapse-aware expected dict.
        if pat.feed != pattern.feed or (sum(pat.batch_feed().values())
                                        != sum(pattern.batch_feed().values())):
            raise SystemExit(f"{arm}: substitution changed the feed/batch-multiset size "
                             f"{pattern.feed}/{pattern.batch_feed()} -> "
                             f"{pat.feed}/{pat.batch_feed()} -- mapping hit a wrong role?")
        entries.append(WaveEntry(pat, key, assets,
                                 {"arm": arm, "reference": row.record_id}))
    if not entries:
        raise SystemExit("no runnable arms (all skipped at fallback_level != 0); "
                         "add the missing assets or pass --allow-fallback")

    t0 = time.time()
    outcomes = verifier.evaluate_wave(entries)
    out = Path(BASE / args.run_dir) / "fr_arms_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for oc in outcomes:
            rec = {"arm": oc.meta.get("arm"), "reference": args.reference,
                   "reference_record": oc.meta.get("reference"),
                   "pair": oc.case_key.pair,
                   "feed": oc.case_key.feed, "status": oc.status,
                   "n_cycles": oc.n_cycles, "wall_s": oc.wall_s,
                   "restart_provenance": oc.restart_provenance,
                   "failure": oc.failure,
                   "fom": oc.fom.as_dict() if oc.fom else None}
            if oc.maps is not None:
                # The canonical flatness scalars, NOT a bare np.max: the harvested
                # quarter-core plane carries NaN in every off-slot cell, so a plain
                # max returns NaN (observed 2026-08-02, first tier-0 sweep lost
                # node_peak on all four chains).  lpopt.data.flatness is the single
                # definition shared by the harvest path, the A/B scorer and the
                # promotion gate, so the number here is comparable to the store's.
                # slot_values() already normalizes every accepted layout
                # ((69,), (9,9), (4,9,9), (n,3,9,9)) to [N, 69]; feeding it a
                # hand-added batch axis would raise on ndim 5.
                from lpopt.data.flatness import map_cov, node_peak, slot_values
                arr = np.asarray(oc.maps, dtype=float)
                sv = slot_values(arr)
                rec["node_peak"] = float(node_peak(sv)[0])
                rec["map_cov"] = float(map_cov(sv)[0])
                # Persist the plane itself -- the verifier purges the case outputs
                # after harvest, so an unsaved map is an unrepeatable measurement.
                # A re-run of the same arm must NOT overwrite the earlier plane:
                # that silently replaces the measurement a published number was
                # read from.  Primary naming is unchanged (kit builders read
                # map_<arm>.npy); a repeat lands at map_<arm>_<k>.npy.
                dst = out.parent / f"map_{oc.meta.get('arm')}.npy"
                if dst.exists():
                    k = 1
                    while (out.parent / f"{dst.stem}_{k}.npy").exists():
                        k += 1
                    new = out.parent / f"{dst.stem}_{k}.npy"
                    print(f"[fr_arms] WARNING {dst.name} already exists; this run's "
                          f"map saved as {new.name} (the earlier plane is kept)")
                    dst = new
                np.save(dst, arr)
            fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec))
    print(f"total wall {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
