"""RULE-CONSTRUCT ACID TEST -- THE MASTER RUNNER.  DO NOT LAUNCH CASUALLY.

THE QUESTION (the program's north star, 2026-08-11)
    Patterns built FROM THE VALIDATED RULES ALONE -- no surrogate in the
    construction loop, no search campaign, zero MASTER calls spent building --
    must approach what the 100-call campaigns found in this cell.  This script
    spends exactly 8 MASTER chains measuring that.

THE CANDIDATES
    8 constructions frozen by ``rule_construct.py`` + the champion referee
    (scratchpad ``rules/acid_batch/candidates.json``): 4 per profile
    (minfr / flat) = top-2 by prediction + 2 diverse mid-rank honesty probes.
    The champion (data/models/split_S1b, v6b) RANKED finished constructions;
    it BUILT nothing.

PRE-REGISTERED SUCCESS CRITERIA -- fixed BEFORE any chain runs; do not move
    Cell records (data/store, ga80|E1_E2|f121, 100-call campaigns):
        F_r record        1.4636      flat record node_peak   1.1899
    SUCCESS(minfr) : min realized F_r over the 4 minfr candidates <= 1.479
                     (record + 0.015) with cbc_max <= 1600, f_q <= 2.41,
                     620 <= cyclen <= 645.
    SUCCESS(flat)  : min realized node_peak over the 4 flat candidates
                     <= 1.205 (record + 0.015) with f_r <= 1.55 and the same
                     cbc/f_q/cyclen gates.
    Either SUCCESS means: the explicit rules capture most of what 100-call
    search finds on that objective.  BOTH failing is a REAL result: the rules
    (as weighted today) do not construct; report plainly.
    SECONDARY (report-only, cannot change the verdict): Spearman(predicted,
    realized) over the 8 -- the honesty probes exist to make this readable.

VALIDITY GATES (all must hold or the verdict is VOID)
    V1 all 8 chains converged
    V2 all at fallback_level 0, restart native:MAS_RST.APRQ_11_0635.19
    V3 the 8 patterns pairwise distinct
    V4 none already in the store (re-checked read-only)
    V5 candidates.json sha256 quoted in the verdict

Usage -- MASTER is expensive; the ISSUER launches this, not the agent::

    python rule_acid_run.py --dry-run          # resolve + plan, NO MASTER
    python rule_acid_run.py                    # the 8 chains (via run_rule_acid.bat)
    python rule_acid_run.py --score-only       # re-print the verdict table

WRITES: ``runs/rule_acid/rule_acid_results.jsonl`` + per-chain map ``.npy``.
NEVER writes to ``data/store/`` or ``data/models/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent

#: The frozen batch staged by score_and_stage.py (scratchpad rules dir).
DEFAULT_CANDIDATES = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude"
    r"\USER"
    r"\8888f052-fa4d-46f0-a439-ef3441b3b061\scratchpad\rules\acid_batch"
    r"\candidates.json")

RESULTS_NAME = "rule_acid_results.jsonl"

#: Pre-registered bars (records + 0.015).  These numbers do not move.
BAR_MINFR_FR = 1.479
BAR_FLAT_NP = 1.205
RECORD_FR = 1.4636
RECORD_NP = 1.1899
#: Program constraints for the bar (cbc uses the NEW 1600 limit; the in-store
#: comparison rows were produced under 1550 -- state this when reporting).
GATES = {"cbc_max": 1600.0, "f_q": 2.41, "cyclen_lo": 620.0,
         "cyclen_hi": 645.0, "f_r": 1.55}

EXPECTED_RESTART = "native:MAS_RST.APRQ_11_0635.19"


def load_candidates(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise SystemExit(f"frozen candidates not found: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    cands = payload.get("candidates") or []
    if not cands:
        raise SystemExit(f"{path} holds no candidates")
    return payload, hashlib.sha256(raw).hexdigest()


def load_results(path: Path) -> list[dict]:
    out: list[dict] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def spearman(x, y) -> float:
    if len(x) < 3 or len(x) != len(y):
        return float("nan")

    def ranks(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
        i = 0
        srt = a[order]
        while i < len(a):
            j = i
            while j + 1 < len(a) and srt[j + 1] == srt[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(x) - np.mean(ranks(x)), ranks(y) - np.mean(ranks(y))
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def check_novelty(payload: dict) -> dict:
    import pandas as pd

    df = pd.read_parquet(BASE / "data/store/records.parquet",
                         columns=["record_id", "pattern"])
    pats = set(df["pattern"].astype(str))
    rids = set(df["record_id"].astype(str))
    hits = sum(1 for c in payload["candidates"]
               if c["pattern"] in pats or str(c["record_id"]) in rids)
    return {"store_rows": int(len(df)), "hits_any": hits, "all_novel": hits == 0}


def _gates_ok(fom: dict, *, need_fr: bool) -> bool:
    try:
        ok = (fom["CBC_max"] <= GATES["cbc_max"] and fom["F_q"] <= GATES["f_q"]
              and GATES["cyclen_lo"] <= fom["cyclen"] <= GATES["cyclen_hi"])
        if need_fr:
            ok = ok and fom["F_r"] <= GATES["f_r"]
        return bool(ok)
    except (KeyError, TypeError):
        return False


def report(payload: dict, cand_sha: str, results: list[dict],
           novelty: dict | None) -> None:
    cands = payload["candidates"]
    by_rid = {str(r["record_id"]): r for r in results}
    print("\n" + "=" * 100)
    print("RULE-CONSTRUCT ACID TEST -- PRE-REGISTERED SCORING")
    print("(success bars fixed in this file's header and run_rule_acid.bat "
          "BEFORE any chain ran)")
    print("=" * 100)
    print(f"candidates.json sha256 : {cand_sha}   <- gate V5: quote in verdict")
    print(f"cell                   : {payload['case']['library_id']} | "
          f"{payload['case']['pair']} | feed {payload['case']['feed']}")
    print(f"constructor            : {payload.get('constructor')}")
    print(f"referee model          : {payload.get('model_dir')} (built nothing)")

    print("\n" + "-" * 100)
    print(f"{'rk':>3} {'origin':<18}{'record_id':<14}{'pred':>8}{'REAL F_r':>9}"
          f"{'node_pk':>9}{'cyclen':>9}{'CBC':>9}{'F_q':>8}  status")
    print("-" * 100)
    for c in cands:
        r = by_rid.get(str(c["record_id"]))
        prof = c["profile"]
        pred_obj = c["pred"]["f_r"] if prof == "minfr" else c["pred"]["node_peak"]
        if r is None:
            print(f"{c['predicted_rank']:>3} {c['origin']:<18}"
                  f"{c['record_id'][:12]:<14}{pred_obj:>8.4f}"
                  f"{'--':>9}{'--':>9}{'--':>9}{'--':>9}{'--':>8}  PENDING")
            continue
        fom = r.get("fom") or {}
        npk = r.get("node_peak")

        def _fmt(v, prec, width):
            return f"{'--':>{width}}" if v is None else f"{float(v):>{width}.{prec}f}"

        print(f"{c['predicted_rank']:>3} {c['origin']:<18}"
              f"{c['record_id'][:12]:<14}{pred_obj:>8.4f}"
              f"{_fmt(fom.get('F_r'), 4, 9)}"
              f"{_fmt(npk, 4, 9)}"
              f"{_fmt(fom.get('cyclen'), 2, 9)}"
              f"{_fmt(fom.get('CBC_max'), 1, 9)}"
              f"{_fmt(fom.get('F_q'), 4, 8)}"
              f"  {r.get('status')}")

    # validity gates
    have = [(c, by_rid.get(str(c["record_id"]))) for c in cands]
    have = [(c, r) for c, r in have if r is not None]
    conv = [(c, r) for c, r in have if r.get("status") == "converged"
            and (r.get("fom") or {}).get("F_r") is not None]
    v1 = len(conv) == len(cands)
    fb = {int(r.get("fallback_level", -1)) for _, r in have}
    prov = {str(r.get("restart_provenance")) for _, r in have}
    v2 = bool(have) and fb == {0} and prov == {EXPECTED_RESTART}
    v3 = len({c["pattern"] for c in cands}) == len(cands)
    v4 = None if novelty is None else bool(novelty.get("all_novel"))
    print("\nVALIDITY GATES: "
          f"V1 {'PASS' if v1 else 'FAIL'} ({len(conv)}/{len(cands)} converged)  "
          f"V2 {'PASS' if v2 else ('FAIL' if have else 'n/a')}  "
          f"V3 {'PASS' if v3 else 'FAIL'}  "
          f"V4 {'PASS' if v4 else ('FAIL' if v4 is False else 'not re-checked')}  "
          "V5 sha256 above")

    if not conv:
        print("\nno converged label yet -- nothing to score")
        return

    print("\n" + "=" * 100)
    print("THE PRE-REGISTERED VERDICT")
    print("=" * 100)
    verdicts = {}
    for prof, bar, rec_val, key in (("minfr", BAR_MINFR_FR, RECORD_FR, "F_r"),
                                    ("flat", BAR_FLAT_NP, RECORD_NP, "node_peak")):
        rows = [(c, r) for c, r in conv if c["profile"] == prof]
        vals = []
        for c, r in rows:
            fom = r["fom"]
            val = fom["F_r"] if key == "F_r" else r.get("node_peak")
            if val is None:
                continue
            feasible = _gates_ok(fom, need_fr=(prof == "flat"))
            vals.append((float(val), feasible, c))
        if not vals:
            verdicts[prof] = "VOID (no labels)"
            continue
        feas_vals = [v for v, ok, _ in vals if ok]
        best = min(feas_vals) if feas_vals else float("nan")
        best_any = min(v for v, _, _ in vals)
        success = bool(feas_vals) and best <= bar
        verdicts[prof] = "SUCCESS" if success else "FAIL"
        print(f"  [{prof}] best feasible {key} = "
              f"{best if feas_vals else float('nan'):.4f}  "
              f"(best ignoring gates {best_any:.4f})   bar {bar:.4f}  "
              f"record {rec_val:.4f}   -> {verdicts[prof]}")
        print(f"          gap to record: "
              f"{(best - rec_val) if feas_vals else float('nan'):+.4f}   "
              f"({len(feas_vals)}/{len(vals)} candidates pass the program "
              f"constraints, cbc<={GATES['cbc_max']:.0f})")

    gates_ok = v1 and v2 and v3 and (v4 is not False)
    if not gates_ok:
        print("\nVERDICT: VOID (a validity gate failed) -- the only repair is "
              "re-running failed chains from the UNCHANGED candidates.json")
    else:
        print(f"\nVERDICT: minfr {verdicts.get('minfr', '--')} | "
              f"flat {verdicts.get('flat', '--')}")
        print("  SUCCESS means: rule-only construction reaches within 0.015 of "
              "what 100-call search found on that objective.")

    # secondary, report-only
    for prof, pkey, rkey in (("minfr", "f_r", "F_r"), ("flat", "node_peak", None)):
        rows = [(c, r) for c, r in conv if c["profile"] == prof]
        if len(rows) >= 3:
            pred = [c["pred"][pkey] for c, _ in rows]
            real = [(r["fom"]["F_r"] if prof == "minfr" else r.get("node_peak"))
                    for _, r in rows]
            if all(v is not None for v in real):
                print(f"  secondary [{prof}] Spearman(pred, real) over "
                      f"{len(rows)} = {spearman(pred, real):+.3f}  "
                      "[report-only; honesty probes make this readable]")


def library_dims(package_root: Path) -> tuple[int, int]:
    xsl = (package_root / "lib" / "MAS_XSL").read_text(errors="replace")
    comp = sum(1 for ln in xsl.splitlines() if ln.startswith("COMP "))
    refl = sum(1 for ln in xsl.splitlines() if ln.startswith("REFL "))
    return (comp + 3, comp + refl)


def main() -> int:
    ap = argparse.ArgumentParser(description="verify the 8 rule-built cores "
                                             "against the pre-registered bar")
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--run-dir", default="runs/rule_acid")
    ap.add_argument("--package", default="../3_GA_Surrogate/FEASIBLE_PACKAGE")
    ap.add_argument("--exe", default="D:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-cycles", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(BASE))
    run_dir = (BASE / args.run_dir).resolve()
    results_path = run_dir / RESULTS_NAME
    payload, cand_sha = load_candidates(Path(args.candidates))
    cands = payload["candidates"]
    results = load_results(results_path)
    done = {str(r["record_id"]) for r in results if r.get("record_id")}

    novelty = None
    try:
        novelty = check_novelty(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] novelty re-check unavailable: {exc}")

    if args.score_only:
        report(payload, cand_sha, results, novelty)
        return 0

    from lpopt.data.schema import unpack_pattern
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.vendor.masterrl.domain import CaseKey

    pkg = (BASE / args.package).resolve()
    if not pkg.is_dir():
        raise SystemExit(f"package not found: {pkg}")
    dims = library_dims(pkg)
    pair = str(payload["case"]["pair"])
    feed = int(payload["case"]["feed"])
    key = CaseKey(pair, feed)
    resolver = CaseAssetResolver(pkg, library_dims=dims)
    assets = resolver.resolve(key)

    print("=" * 100)
    print("RULE-CONSTRUCT ACID TEST -- MASTER RUNNER")
    print("=" * 100)
    print(f"candidates : {args.candidates}  n={len(cands)}")
    print(f"sha256     : {cand_sha}")
    print(f"cell       : {payload['case']['library_id']} | {pair} | feed {feed}")
    print(f"package    : {pkg}  dims={dims}")
    print(f"assets     : fallback_level={assets.fallback_level} "
          f"restart={assets.restart_provenance}")
    if assets.fallback_level != 0 or assets.restart_provenance != EXPECTED_RESTART:
        print("!! GATE V2 CANNOT PASS: not the cell's own native restart -- fix "
              "the package before spending chains")
        if not args.dry_run:
            return 2
    if novelty is not None and not novelty["all_novel"]:
        print(f"!! GATE V4 FAILS: {novelty['hits_any']} candidates already in "
              "the store")
        if not args.dry_run:
            return 2

    entries = []
    for c in cands:
        rid = str(c["record_id"])
        pattern = unpack_pattern(c["pattern"])
        pattern.validate_case(pair, feed)
        if rid not in done:
            entries.append((c, pattern))
    if done:
        print(f"resume: {len(done)}/{len(cands)} already in {results_path.name}")

    if args.dry_run:
        try:
            from lpopt.data.flatness import map_cov, node_peak, slot_values  # noqa: F401
            from lpopt.search.verify import WaveEntry, WaveVerifier  # noqa: F401
            probe = WaveVerifier(
                run_dir=run_dir / "_dryrun_probe", package_root=pkg,
                executable=args.exe, workers=args.workers, timeout=args.timeout,
                max_cycles=args.max_cycles, consecutive=2, library_dims=dims,
                harvest_maps=True)
            print(f"import probe : OK (n_workers={probe.n_workers})")
            print(f"MASTER exe   : {args.exe}  "
                  f"present={Path(str(args.exe)).is_file()}")
        except Exception as exc:  # noqa: BLE001
            print(f"import probe : FAILED -- {type(exc).__name__}: {exc}")
            return 3
        print(f"\nDRY RUN -- no MASTER launched.  {len(entries)} chain(s) "
              f"would run into {run_dir}")
        report(payload, cand_sha, results, novelty)
        return 0

    if not entries:
        print("nothing left to run")
        report(payload, cand_sha, results, novelty)
        return 0

    from lpopt.data.flatness import map_cov, node_peak, slot_values
    from lpopt.search.verify import WaveEntry, WaveVerifier

    verifier = WaveVerifier(
        run_dir=run_dir, package_root=pkg, executable=args.exe,
        workers=args.workers, timeout=args.timeout, max_cycles=args.max_cycles,
        consecutive=2, library_dims=dims, harvest_maps=True)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wave = [
        WaveEntry(pat, key, assets,
                  {"record_id": str(c["record_id"]),
                   "predicted_rank": int(c["predicted_rank"]),
                   "origin": c.get("origin"), "profile": c.get("profile")})
        for c, pat in entries
    ]
    print(f"\n>> wave: {len(wave)} chain(s), {args.workers} workers")
    outcomes = verifier.evaluate_wave(wave)
    with results_path.open("a", encoding="utf-8") as fh:
        for oc in outcomes:
            rid = str(oc.meta.get("record_id"))
            rank = int(oc.meta.get("predicted_rank"))
            rec = {
                "record_id": rid,
                "predicted_rank": rank,
                "origin": oc.meta.get("origin"),
                "profile": oc.meta.get("profile"),
                "pair": pair, "feed": feed,
                "library_id": payload["case"]["library_id"],
                "status": oc.status, "n_cycles": oc.n_cycles,
                "wall_s": oc.wall_s,
                "restart_provenance": oc.restart_provenance,
                "fallback_level": assets.fallback_level,
                "converged_at_cap": bool(oc.converged_at_cap),
                "tolerance_margin": oc.tolerance_margin,
                "failure": oc.failure,
                "fom": oc.fom.as_dict() if oc.fom else None,
                "candidates_sha256": cand_sha,
            }
            if oc.maps is not None:
                arr = np.asarray(oc.maps, dtype=float)
                sv = slot_values(arr)
                rec["node_peak"] = float(node_peak(sv)[0])
                rec["map_cov"] = float(map_cov(sv)[0])
                np.save(run_dir / f"map_rank{rank:02d}_{rid[:12]}.npy", arr)
            else:
                rec["node_peak"] = None
                rec["map_cov"] = None
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            fr = (rec["fom"] or {}).get("F_r")
            print(f"   rank {rank:>2} {rid[:12]}  {rec['status']:<13}"
                  f"REAL F_r {'--' if fr is None else f'{fr:.4f}'}  "
                  f"({rec['wall_s']:.0f}s)")

    print(f"\ntotal wall {time.time() - t0:.1f}s -> {results_path}")
    report(payload, cand_sha, load_results(results_path), novelty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
