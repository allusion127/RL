"""Regenerate FULL cy1 -> equilibrium MASTER chains for STORED loading patterns.

WHY.  The fleet-wide harvest into ``2_LP/LOW_Fr_MASTER_result/`` returned 14,180
cases but only **2** full ``cy01 -> equilibrium`` sequences: every campaign
harness purges its case dirs after harvesting the FOM, so what survives on disk
is almost always the final cycle alone.  The equilibrium-cycle surrogate needs
the per-cycle sequence, not the endpoint.  MASTER is deterministic (proven by
the ``fr_arms`` A0 control reproducing F_r to 0.0000), and the store holds
72,585 evaluated patterns -- so the sequences can simply be REGENERATED with
full retention instead of being recovered.

WHAT.  For each selected store record this driver rebuilds

    cy01  fresh core (%LPD_BCH, no restart, capped)   <- shared per (pair, feed)
    cy02  reload with THE RECORD'S OWN pattern (%LPD_SHF)
    cy03 .. cyNN  same pattern, chained restarts, until the five-FOM
                  successive comparison settles (EquilibriumRunner)

and copies EVERY cycle's outputs to
``2_LP/LOW_Fr_MASTER_result/regen/<record_id[:12]>/cyNN/``.

Nothing here reinvents deck synthesis or chaining: ``lpopt.design.coredeck``
builds both decks, ``lpopt.design.bootstrap.run_cycle1`` runs the fresh core,
and the vendor ``EquilibriumRunner`` (with ``keep_success=True``, i.e. the
NON-purging variant) drives the reload chain.  This module only (a) selects
records, (b) injects the stored pattern in place of ``make_band_restart``'s
``random_genome``, (c) harvests per-cycle files, and (d) compares the
recomputed final-cycle FOMs against the stored labels.

Deck fidelity is not assumed: ``build_cycle1_deck`` and
``build_reload_deck + replace_lpd_shf(pattern.to_shf())`` reproduce the retained
historical bootstrap decks (``package/bootstrap_work/T5_T6_f101/``) BYTE-FOR-BYTE
-- see ``tests/test_regen_chain.py``.

DETERMINISM CAVEAT (read the manifest with this in mind).  The stored labels came
from RESTART-SEEDED chains (each cell's ``bases/<folder>/MAS_RST.*`` band seed);
this driver bootstraps from cy1 instead.  The two agree only if the equilibrium
is unique.  A converged chain whose FOMs miss the tolerances is recorded as
``match_ok=False`` WITH the deltas and is NOT an error -- that discrepancy
measures equilibrium non-uniqueness and is itself a result.

cy1 CAP.  An all-fresh cy1 runs far longer than any equilibrium cycle, which
hands cy02 a carryover ~2x too deep and (observed) diverges to NaN.  The cap that
each cell's OWN historical bootstrap used is recoverable from the restart
basename inside ``package/cores/<folder>/bootstrap/MAS_INP_cy02.inp``
(``MAS_RST.APRQ_01_0620.00`` -> 620.00 EFPD for T6_T4), so that is the default:
the regenerated cy1 is the same cy1 the cell's band seed came from.

USAGE

    # pilot (2 records, sequential)
    python regen_chain.py --record-id 9b9fabe8 --record-id 1637c21e

    # fleet scale-up for one cell
    python regen_chain.py --pair T6_T4 --feed 121 --top-n 50

Read-only on ``data/store``.  Work dirs live under ``runs/regen_pilot/`` and are
purged per record ON SUCCESS ONLY (a failed chain's work dir is its only
evidence -- the same rule ``make_band_restart`` follows).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

#: Local (box 104) MASTER.  The 199 decks reference ``C:/Users/USER/...``; a past
#: local run failed with err=8 on every chain because it inherited that path.
DEFAULT_EXE = "D:/DeCART_MASTER/BIN/master4.0m4_r1.exe"
DEFAULT_PACKAGE = "data/design/package"
DEFAULT_OUT_ROOT = BASE.parent / "2_LP" / "LOW_Fr_MASTER_result" / "regen"
DEFAULT_RUN_DIR = BASE / "runs" / "regen_pilot"

#: Determinism gate: stored label vs recomputed final cycle (task spec).
TOL_CYCLEN, TOL_F_R, TOL_CBC = 0.5, 2.0e-3, 2.0

#: Staged library inputs -- 14 MB each, identical in every cycle dir, already in
#: the package.  Copying them 11x per chain would be ~3 GB of duplication.
_SKIP_NAMES = {"MAS_XSL", "MAS_HFF"}

MANIFEST_FIELDS = [
    "record_id", "pair", "feed", "library_id", "n_cycles",
    "stored_f_r", "stored_cyclen", "stored_cbc_max",
    "regen_f_r", "regen_cyclen", "regen_cbc_max",
    "d_f_r", "d_cyclen", "d_cbc_max", "match_ok",
    "converged", "wall_seconds", "cy1_cap_efpd", "stored_n_cycles",
    "out_dir", "error",
]

_RST_RE = re.compile(r"^MAS_RST\.[A-Za-z0-9]+_(\d+)_(\d+)\.(\d+)$")


# --------------------------------------------------------------------------- #
# cy1 cap: recover the value the cell's own historical bootstrap used
# --------------------------------------------------------------------------- #
def historical_cy1_cap(pkg: Path, folder: str) -> float | None:
    """EFPD encoded in the cy1 restart that ``cores/<folder>/bootstrap`` reads.

    The bootstrap cy02 template names its input restart, and MASTER encodes the
    cycle's end-of-cycle EFPD in that basename, so the cap is recoverable from
    an artifact the cell already carries.  ``None`` when the template is absent.

    CAVEAT: an UNCAPPED (natural-EOC) cy1 leaves the same kind of basename, and
    the two are indistinguishable from the name alone.  A suspiciously long value
    (T5_T6's 981.02 EFPD == an all-fresh natural EOC, ~1.5x any equilibrium
    cycle) means that cell bootstrapped uncapped -- pass ``--cy1-cap-efpd none``
    there, or the fixed-step ramp will retrace that burnup on a different
    depletion path.  The manifest records the value actually used.
    """
    template = pkg / "cores" / folder / "bootstrap" / "MAS_INP_cy02.inp"
    if not template.is_file():
        return None
    for line in template.read_text(errors="replace").splitlines():
        m = _RST_RE.match(line.split("#")[0].strip())
        if m and int(m.group(1)) == 1:
            return float(f"{int(m.group(2))}.{m.group(3)}")
    return None


# --------------------------------------------------------------------------- #
# record selection (READ-ONLY on data/store)
# --------------------------------------------------------------------------- #
def select_records(store_parquet: Path, args) -> "list":
    import pandas as pd

    df = pd.read_parquet(store_parquet)
    if args.record_id:
        rows = []
        for rid in args.record_id:
            hit = df[df.record_id.str.startswith(rid)]
            if hit.empty:
                raise SystemExit(f"no store record starts with {rid!r}")
            if len(hit) > 1:
                raise SystemExit(
                    f"{rid!r} is ambiguous ({len(hit)} records); use more characters")
            rows.append(hit.iloc[0])
        return rows

    if not args.pair:
        raise SystemExit("pass --record-id (repeatable) or --pair [--feed] [--top-n]")
    sub = df[df.case_pair == args.pair]
    if args.feed is not None:
        sub = sub[sub.feed == args.feed]
    if args.library_id:
        sub = sub[sub.library_id == args.library_id]
    # A label that never converged is not a determinism reference.
    sub = sub[sub.valid.fillna(False) & sub.converged.fillna(False) & sub.f_r.notna()]
    if sub.empty:
        raise SystemExit(f"no valid converged store rows for pair={args.pair} "
                         f"feed={args.feed} library={args.library_id}")
    sub = sub.sort_values("f_r").head(args.top_n)
    return [sub.iloc[i] for i in range(len(sub))]


# --------------------------------------------------------------------------- #
# per-cycle harvest
# --------------------------------------------------------------------------- #
def harvest_cycle(src: Path, dst: Path, *, keep_rst: bool) -> list[str]:
    """Copy one cycle's outputs; return the MAS_RST.* basenames it produced.

    The restart binaries are NAMED but (by default) not copied -- the same policy
    the LOW_Fr_MASTER_result collection used (~96.5 GB fleet-wide, opaque state).
    """
    dst.mkdir(parents=True, exist_ok=True)
    restarts: list[str] = []
    for path in sorted(src.iterdir()):
        if not path.is_file() or path.name in _SKIP_NAMES:
            continue
        if path.name.upper().startswith("MAS_RST."):
            restarts.append(path.name)
            if not keep_rst:
                continue
        shutil.copy2(path, dst / path.name)
    return restarts


def write_cycle_meta(dst: Path, *, cycle: int, kind: str, metrics, restarts,
                     work_dir: Path) -> None:
    (dst / "_meta_cycle.json").write_text(json.dumps({
        "cycle": cycle,
        "kind": kind,                       # "cy1_fresh_core" | "reload"
        "metrics": metrics.as_dict() if metrics is not None else None,
        "restart_files": restarts,          # named even when not retained
        "source_work_dir": str(work_dir),
    }, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# one chain
# --------------------------------------------------------------------------- #
def regen_one(row, *, pkg: Path, exe: str, out_root: Path, run_dir: Path,
              aliases: list[str], cap_override, args) -> dict:
    from lpopt.data.schema import unpack_pattern
    from lpopt.design.bootstrap import run_cycle1
    from lpopt.design.coredeck import build_cycle1_deck, build_reload_deck
    from lpopt.vendor.masterrl.dataset import CaseData
    from lpopt.vendor.masterrl.domain import CaseKey
    from lpopt.vendor.masterrl.equilibrium import (
        EquilibriumRunner, EquilibriumTolerances)
    from lpopt.vendor.masterrl.master import MasterRunner

    rid = str(row["record_id"])
    short = rid[:12]
    pair, feed = str(row["case_pair"]), int(row["feed"])
    key = CaseKey(pair, feed)
    a, _, b = pair.partition("_")

    out_dir = out_root / short
    work_root = run_dir / short
    result = {f: "" for f in MANIFEST_FIELDS}
    result.update(record_id=rid, pair=pair, feed=feed,
                  library_id=str(row.get("library_id", "")),
                  stored_f_r=row.get("f_r"), stored_cyclen=row.get("cyclen"),
                  stored_cbc_max=row.get("cbc_max"),
                  stored_n_cycles=row.get("n_cycles"),
                  out_dir=str(out_dir), n_cycles=0, match_ok=False,
                  converged=False, error="")

    cap = cap_override if cap_override != "auto" else historical_cy1_cap(pkg, key.folder)
    result["cy1_cap_efpd"] = "" if cap is None else cap
    pattern = unpack_pattern(str(row["pattern"]))
    pattern.validate_case(pair, feed)
    pattern.validate_quarter_conventions()

    print(f"\n=== {short}  {key.label}  lib={result['library_id']}  "
          f"cy1_cap={cap}  stored F_r={row.get('f_r')} cyclen={row.get('cyclen')} "
          f"CBC={row.get('cbc_max')}")
    if args.dry_run:
        print(f"    [dry-run] -> {out_dir}")
        return result

    t0 = time.monotonic()
    ok = False
    try:
        # ---- cy1: shared per (pair, feed, cap); run once, reuse ------------- #
        cy1_dir = run_dir / "cy1" / f"{key.folder}_cap{'nat' if cap is None else f'{cap:g}'}"
        stamp = cy1_dir / "_cy1_ok.json"
        if stamp.is_file():
            cy1_rst = cy1_dir / json.loads(stamp.read_text())["restart"]
            print(f"    cy1 reused: {cy1_rst.name}")
        else:
            shutil.rmtree(cy1_dir, ignore_errors=True)
            deck = build_cycle1_deck(aliases, (a, b), cap_efpd=cap)
            print("    cy1 running (fresh core)...", flush=True)
            t_cy1 = time.monotonic()
            cy1_rst = run_cycle1(deck, pkg / "lib" / "MAS_XSL", pkg / "lib" / "MAS_HFF",
                                 exe, cy1_dir, timeout_s=args.timeout)
            stamp.write_text(json.dumps({"restart": cy1_rst.name}) + "\n")
            print(f"    cy1 done: {cy1_rst.name}  ({time.monotonic()-t_cy1:.0f}s)")

        # ---- cy02 reload template referencing that cy1 restart -------------- #
        seed_dir = work_root / "seed"
        seed_dir.mkdir(parents=True, exist_ok=True)
        template = seed_dir / "MAS_INP_cy02.inp"
        template.write_text(build_reload_deck(aliases, cy1_rst.name, 2),
                            encoding="utf-8")

        # ---- chain to equilibrium, retaining EVERY cycle -------------------- #
        master = MasterRunner(pkg, str(exe), work_root=work_root / "master",
                              timeout=args.timeout, keep_success=True)
        runner = EquilibriumRunner(master, max_cycles=args.max_cycles,
                                   consecutive=args.consecutive,
                                   tolerances=EquilibriumTolerances(),
                                   keep_success=True,
                                   enable_pin_burnup=args.enable_pin_burnup)
        case = CaseData(key=key, cell=0.0, records=(),
                        template_path=template, restart_path=cy1_rst)
        print(f"    chaining reload cycles (max {args.max_cycles})...", flush=True)
        eq = runner.run(case, pattern)

        # ---- harvest ------------------------------------------------------- #
        shutil.rmtree(out_dir, ignore_errors=True)
        rst = harvest_cycle(cy1_dir, out_dir / "cy01", keep_rst=args.keep_rst)
        write_cycle_meta(out_dir / "cy01", cycle=1, kind="cy1_fresh_core",
                         metrics=None, restarts=rst, work_dir=cy1_dir)
        for i, cyc in enumerate(eq.cycles, start=2):
            dst = out_dir / f"cy{i:02d}"
            rst = harvest_cycle(cyc.work_dir, dst, keep_rst=args.keep_rst)
            write_cycle_meta(dst, cycle=cyc.cycle, kind="reload",
                             metrics=cyc.metrics, restarts=rst,
                             work_dir=cyc.work_dir)

        fom = eq.fom
        d_fr = abs(float(fom.f_r) - float(row["f_r"]))
        d_cl = abs(float(fom.cyclen) - float(row["cyclen"]))
        d_cb = abs(float(fom.cbc_max) - float(row["cbc_max"]))
        match_ok = bool(d_cl <= TOL_CYCLEN and d_fr <= TOL_F_R and d_cb <= TOL_CBC)
        result.update(n_cycles=1 + eq.n_cycles, converged=bool(eq.converged),
                      regen_f_r=fom.f_r, regen_cyclen=fom.cyclen,
                      regen_cbc_max=fom.cbc_max,
                      d_f_r=round(d_fr, 6), d_cyclen=round(d_cl, 4),
                      d_cbc_max=round(d_cb, 3), match_ok=match_ok)

        (out_dir / "_meta_chain.json").write_text(json.dumps({
            "record_id": rid, "pair": pair, "feed": feed,
            "library_id": result["library_id"],
            "package": str(pkg), "executable": str(exe),
            "cy1_cap_efpd": cap, "pattern": str(row["pattern"]),
            "n_cycles": result["n_cycles"], "converged": bool(eq.converged),
            "converged_at_cap": bool(eq.converged_at_cap),
            "tolerance_margin": eq.tolerance_margin,
            "per_cycle_fom": [c.metrics.as_dict() for c in eq.cycles],
            "stored": {"f_r": row.get("f_r"), "cyclen": row.get("cyclen"),
                       "cbc_max": row.get("cbc_max"),
                       "n_cycles": row.get("n_cycles"),
                       "restart_provenance": row.get("restart_provenance"),
                       "campaign": row.get("campaign")},
            "regen": fom.as_dict(),
            "deltas": {"f_r": d_fr, "cyclen": d_cl, "cbc_max": d_cb},
            "tolerances": {"cyclen": TOL_CYCLEN, "f_r": TOL_F_R, "cbc_max": TOL_CBC},
            "match_ok": match_ok,
        }, indent=2, default=str) + "\n", encoding="utf-8")

        print(f"    {result['n_cycles']} cycles  converged={eq.converged}  "
              f"F_r {fom.f_r:.4f} (d {d_fr:+.4f})  cyclen {fom.cyclen:.3f} "
              f"(d {d_cl:+.3f})  CBC {fom.cbc_max:.2f} (d {d_cb:+.2f})  "
              f"match_ok={match_ok}")
        ok = True
    except Exception as exc:                                    # noqa: BLE001
        # A failed chain is reported, never fatal: the run continues so one bad
        # record cannot cost the whole sweep.
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"    [FAILED] {result['error']}")
    finally:
        result["wall_seconds"] = round(time.monotonic() - t0, 1)
        # Purge on SUCCESS only.  On failure the work dir is the ONLY evidence
        # (MasterRunError even names it as retained) -- bootstrap.py learned this
        # the hard way when an unconditional rmtree destroyed cy02-hang forensics.
        if ok and not args.keep_work:
            shutil.rmtree(work_root, ignore_errors=True)
    return result


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate full cy1->equilibrium MASTER chains for stored patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--record-id", action="append", default=None,
                    help="store record_id or unique prefix (repeatable)")
    ap.add_argument("--pair", default=None, help="e.g. T6_T4")
    ap.add_argument("--feed", type=int, default=None)
    ap.add_argument("--library-id", default=None)
    ap.add_argument("--top-n", type=int, default=1,
                    help="with --pair: lowest-F_r valid converged rows")
    ap.add_argument("--package", default=DEFAULT_PACKAGE)
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--store", default="data/store/records.parquet")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--cy1-cap-efpd", default="auto",
                    help="'auto' = the cell's historical bootstrap cap, "
                         "'none' = natural-EOC cy1, or an EFPD value")
    ap.add_argument("--max-cycles", type=int, default=16)
    ap.add_argument("--consecutive", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--enable-pin-burnup", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--keep-rst", action="store_true",
                    help="also copy the MAS_RST.* binaries (~6.5 MB/cycle)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the run work dirs even on success")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from lpopt.design.bootstrap import library_aliases

    pkg = Path(args.package)
    pkg = pkg if pkg.is_absolute() else (BASE / pkg)
    pkg = pkg.resolve()
    store = Path(args.store)
    store = store if store.is_absolute() else (BASE / store)
    out_root, run_dir = Path(args.out_root), Path(args.run_dir)

    if args.cy1_cap_efpd == "auto":
        cap_override = "auto"
    elif args.cy1_cap_efpd.lower() == "none":
        cap_override = None
    else:
        cap_override = float(args.cy1_cap_efpd)

    if not args.dry_run and not Path(args.exe).is_file():
        raise SystemExit(f"MASTER executable not found: {args.exe} "
                         f"(the LOCAL box path is {DEFAULT_EXE})")
    aliases = library_aliases(pkg)
    rows = select_records(store, args)

    out_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"package : {pkg}  ({len(aliases)} fuel aliases)")
    print(f"exe     : {args.exe}")
    print(f"store   : {store}  (read-only)")
    print(f"out     : {out_root}")
    print(f"records : {len(rows)}")

    manifest = out_root / "regen_manifest.csv"
    new = not manifest.exists()
    results = []
    t0 = time.monotonic()
    for row in rows:
        res = regen_one(row, pkg=pkg, exe=args.exe, out_root=out_root,
                        run_dir=run_dir, aliases=aliases,
                        cap_override=cap_override, args=args)
        results.append(res)
        if args.dry_run:
            continue
        with manifest.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow(res)

    done = [r for r in results if not r["error"]]
    matched = [r for r in done if r["match_ok"]]
    print(f"\n{'='*78}\n{len(done)}/{len(results)} chains regenerated, "
          f"{len(matched)} match_ok, total wall {time.monotonic()-t0:.0f}s")
    for r in results:
        print(f"  {r['record_id'][:12]}  {r['pair']}/f{r['feed']}  "
              f"cycles={r['n_cycles']}  match_ok={r['match_ok']}  "
              f"dF_r={r['d_f_r']} dcyclen={r['d_cyclen']} dCBC={r['d_cbc_max']}  "
              f"{r['wall_seconds']}s  {r['error']}")
    if not args.dry_run:
        print(f"manifest -> {manifest}")
    return 0 if done or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
