"""Batch generalization of ``fr_arms.py`` -- the fuel lever over a POPULATION of
loading patterns instead of one.

``fr_arms.py`` answered "what does swapping the fresh fuel do to ONE core?".  It
found that the flatter lattice (E3_E4, FF 1.101/1.139) lowered F_r 1.5207 ->
1.5137 and map_cov 0.22208 -> 0.19668 on the flattest feasible core, while
RAISING node_peak 1.2085 -> 1.2604 -- because that pattern was optimized for the
ORIGINAL fuel.  A single point cannot separate "the fuel is better" from "this
pattern happens to like the fuel".

This script takes the top-K stored patterns of a SOURCE cell (pair + feed),
substitutes ONLY the two fresh batch identities for a TARGET pair, re-runs the
equilibrium chain for each, and reports the PAIRED distribution of
(target - source) for every figure of merit.  Everything else -- the 69 shuffle
cards, the feed, the symmetry class, the package, the equilibrium protocol -- is
byte-identical between each source record and its target twin, so the paired
difference is attributable to the assembly pin-power form function alone.

Usage (MASTER is expensive -- the issuer schedules these, not the agent)::

    python fr_transfer.py --target-pair E3_E4 --dry-run
    python fr_transfer.py --target-pair E3_E4 --select mixed --k 24 \
        --package ../3_GA_Surrogate/FEASIBLE_PACKAGE \
        --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe \
        --run-dir runs/fr_transfer_E3E4 --workers 16

To split one sweep across two boxes, give the second box the same --select and
--k N --offset N of the first: ``--k 100`` on box A and ``--k 65 --offset 100``
on box B cover disjoint halves of the same ranking (see ``--offset``).

Results are appended to ``<run-dir>/fr_transfer_results.jsonl`` (one line per
source pattern) and the harvested BOC assembly-power planes to
``<run-dir>/map_<source_record_id[:12]>.npy``.  Nothing is written to ``data/``.
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

#: Program feasibility screen (the same four constraints the promotion gate and
#: the elite selector use).  A source pattern that already violates one of these
#: is not part of the operating envelope the fuel decision is about, so it never
#: enters the paired sample.
SCREEN = {
    "f_r_max": 1.55,
    "cbc_max_max": 1550.0,
    "f_q_max": 2.41,
    "cyclen_min": 620.0,
    "cyclen_max": 645.0,
}

#: Source columns carried into the output so the pairing survives the run.  These
#: ARE the comparison -- an output line without them is an unusable measurement.
SOURCE_METRICS = ("f_r", "node_peak", "map_cov", "cyclen", "cbc_max", "f_q")

#: ``FOM.as_dict()`` key -> source column, for the paired delta table.
FOM_TO_SOURCE = {
    "F_r": "f_r",
    "node_peak": "node_peak",
    "map_cov": "map_cov",
    "cyclen": "cyclen",
    "CBC_max": "cbc_max",
    "F_q": "f_q",
}

STORE = BASE / "data/store/records.parquet"

RESULTS_NAME = "fr_transfer_results.jsonl"


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _screen(df: pd.DataFrame, relax: bool = False) -> pd.DataFrame:
    """Rows inside the program feasibility envelope (NaN fails every test).

    ``relax=True`` drops the cyclen/CBC band and keeps only the peaking axes.
    Needed for CROSS-REACTIVITY transfers (2026-08-11, E1_E2/f117 -> T5_T6/f117):
    the source cell's cyclen is a property of the SOURCE fuel (E1_E2@f117 median
    612 EFPD, below the 620 floor), while the more reactive target fuel runs the
    same patterns ~+10-20 EFPD longer -- screening sources on the source-fuel
    band would empty a pool whose TARGET rows land in-window.  The target rows
    are re-measured and re-screened by whoever consumes them.
    """
    keep = (df["f_r"] <= SCREEN["f_r_max"] * (1.25 if relax else 1.0)) & (
        df["f_q"] <= SCREEN["f_q_max"] * (1.25 if relax else 1.0))
    if not relax:
        keep &= ((df["cbc_max"] <= SCREEN["cbc_max_max"])
                 & (df["cyclen"] >= SCREEN["cyclen_min"])
                 & (df["cyclen"] <= SCREEN["cyclen_max"]))
    return df[keep]


def eligible_sources(pair: str, feed: int, relax: bool = False) -> pd.DataFrame:
    """Converged, map-bearing, feasible, pattern-distinct rows of one cell.

    Two de-duplications, both required: ``record_id`` because the store can carry
    the same evaluation twice (merged multi-PC kits), and ``pattern`` because the
    SAME 69 cards evaluated under two library ids are two record_ids but ONE
    experiment here -- running both would double-count that pattern in the paired
    statistics and silently weight the mean.
    """
    if not STORE.is_file():
        raise SystemExit(f"store not found: {STORE}")
    df = pd.read_parquet(STORE)
    cell = df[(df["case_pair"] == pair) & (df["feed"] == int(feed))]
    ok = cell[(cell["converged"] == True) & cell["node_peak"].notna()]  # noqa: E712
    ok = _screen(ok, relax=relax)
    # Deterministic representative per pattern (record_id order), so a re-run
    # selects the same K rows and the resume check keeps working.
    ok = ok.sort_values("record_id")
    ok = ok.drop_duplicates(subset=["record_id"]).drop_duplicates(subset=["pattern"])
    return ok.reset_index(drop=True)


def select_sources(pool: pd.DataFrame, select: str, k: int,
                   offset: int = 0) -> list[dict]:
    """K source rows in evaluation order, each tagged with why it was picked.

    ``flat``  -- node_peak ascending (the flattest cores);
    ``minfr`` -- f_r ascending (the lowest-peaking cores);
    ``mixed`` -- half of each, de-duplicated, topped up from the flat order.

    ``mixed`` is the default because the two orders are NOT the same population:
    the store's flattest core (node_peak 1.2085) has F_r 1.5207 and its lowest-F_r
    core (F_r 1.4636) has node_peak 1.2620 -- the objectives trade off inside the
    LP space, so a fuel verdict drawn from either tail alone is a verdict about
    that tail.

    ``offset`` skips the first N ranked entries, defined as *the exact set a run
    with the same ``select`` and ``k = N`` would have taken*: that set is removed
    from the pool and the selection rule is then applied afresh to the remainder.
    The set definition (rather than a rank slice) is required because ``mixed``
    is NOT prefix-stable in k -- it takes ``(k+1)//2`` from the flat order first,
    so the first 50 of a K=100 mixed ranking and the first 50 of a K=165 mixed
    ranking agree, but positions 50..99 do not.  Slicing ranks would therefore
    re-run patterns the K=N wave already covered.  With the set semantics,
    ``--k N`` and ``--k M --offset N`` are disjoint by construction and together
    reproduce what a single ``--k N+M`` run would have covered (as a set).

    That guarantee is for a TWO-way split, which is what it is for.  ``flat`` and
    ``minfr`` are plain prefixes and so chain to any number of parts, but a
    three-way ``mixed`` chain (0 / N1 / N1+N2) can repeat a pattern, because the
    excluded set at N1+N2 is not the union of the first two waves -- same
    non-prefix-stability.  Split ``mixed`` in two, or verify the ids first.
    """
    if offset < 0:
        raise ValueError(f"--offset must be >= 0, got {offset}")
    if offset:
        taken = {e["record_id"] for e in select_sources(pool, select, offset)}
        pool = pool[~pool["record_id"].astype(str).isin(taken)].reset_index(drop=True)

    flat_order = pool.sort_values(["node_peak", "record_id"])
    minfr_order = pool.sort_values(["f_r", "record_id"])

    picked: list[dict] = []
    seen: set[str] = set()

    def take(frame: pd.DataFrame, tag: str, limit: int) -> None:
        for _, row in frame.iterrows():
            if len(picked) >= limit:
                return
            rid = str(row["record_id"])
            if rid in seen:
                continue
            seen.add(rid)
            picked.append({
                "record_id": rid,
                "pattern": str(row["pattern"]),
                "picked_by": tag,
                "source": {m: (None if pd.isna(row[m]) else float(row[m]))
                           for m in SOURCE_METRICS},
            })

    if select == "flat":
        take(flat_order, "flat", k)
    elif select == "minfr":
        take(minfr_order, "minfr", k)
    elif select == "mixed":
        n_flat = (k + 1) // 2
        take(flat_order, "flat", n_flat)
        take(minfr_order, "minfr", k)
        # Overlap between the two tails leaves the sample short -- top up from the
        # flat order rather than returning fewer than K.
        take(flat_order, "flat", k)
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"unknown --select {select!r}")

    # Ranks continue past the offset so the two halves of a split sweep can be
    # concatenated without colliding ranks in the results files.
    for rank, entry in enumerate(picked):
        entry["rank"] = offset + rank
    return picked


# --------------------------------------------------------------------------- #
# substitution
# --------------------------------------------------------------------------- #
def pair_mapping(source_pair: str, target_pair: str) -> dict[str, str]:
    """``{source token -> target token}`` in E1-role / E2-role order.

    Positional, NOT sorted: ``E1_E2 -> E3_E4`` must send E1 (the flat/high-k role)
    to E3 and E2 (the peaky/low-k role) to E4.  A silent role swap changes which
    lattice sits in the centre-and-interior slots and would make the paired
    difference meaningless.
    """
    src = source_pair.split("_")
    tgt = target_pair.split("_")
    if len(src) != 2 or len(tgt) != 2:
        raise SystemExit(f"pair labels must be '<A>_<B>': {source_pair!r} -> {target_pair!r}")
    return {src[0]: tgt[0], src[1]: tgt[1]}


def substitute(pattern, mapping: dict[str, str]):
    """Replace the fresh batch identities; every shuffle card is untouched."""
    items = []
    for it in pattern.items:
        if it.is_fresh and it.batch in mapping:
            it = dataclasses.replace(it, batch=mapping[it.batch])
        items.append(it)
    return type(pattern)(tuple(items))


def substitute_checked(pattern, mapping: dict[str, str], target_pair: str,
                       feed: int, record_id: str):
    """:func:`substitute` + the invariants that make the pair comparable.

    The feed and the per-batch multiset must survive the swap exactly; only the
    batch NAMES may change.  A mismatch means the mapping hit the wrong role (or
    a card that was not fresh), which invalidates the whole comparison -- so it is
    a hard failure, never a warning.
    """
    out = substitute(pattern, mapping)
    src_feed = pattern.batch_feed()
    tgt_feed = out.batch_feed()
    expected = {mapping.get(b, b): n for b, n in src_feed.items()}
    if out.feed != pattern.feed:
        raise SystemExit(
            f"{record_id[:12]}: substitution changed the weighted feed "
            f"{pattern.feed} -> {out.feed}")
    if tgt_feed != expected:
        raise SystemExit(
            f"{record_id[:12]}: substitution changed the batch multiset "
            f"{src_feed} -> {tgt_feed} (expected {expected}) -- role swap?")
    if sorted(tgt_feed.values()) != sorted(src_feed.values()):
        raise SystemExit(
            f"{record_id[:12]}: batch multiset SIZES changed {src_feed} -> {tgt_feed}")
    # Also the vendor's own contract, checked here rather than 40 MASTER-minutes in.
    out.validate_case(target_pair, int(feed))
    return out


def library_dims(package_root: Path) -> tuple[int, int]:
    xsl = (package_root / "lib" / "MAS_XSL").read_text(errors="replace")
    comp = sum(1 for ln in xsl.splitlines() if ln.startswith("COMP "))
    refl = sum(1 for ln in xsl.splitlines() if ln.startswith("REFL "))
    return (comp + 3, comp + refl)


# --------------------------------------------------------------------------- #
# results io
# --------------------------------------------------------------------------- #
def already_done(results_path: Path) -> set[str]:
    """``source_record_id`` of every line already in the results file.

    A K=24 wave is ~24 equilibrium chains; losing the finished ones to an
    interrupted box would cost more than the whole experiment.
    """
    done: set[str] = set()
    if not results_path.is_file():
        return done
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = rec.get("source_record_id")
        if rid:
            done.add(str(rid))
    return done


def load_results(results_path: Path) -> list[dict]:
    out: list[dict] = []
    if not results_path.is_file():
        return out
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def target_metrics(rec: dict) -> dict[str, float | None]:
    """The six target scalars of one result line, in SOURCE column names."""
    fom = rec.get("fom") or {}
    vals: dict[str, float | None] = {}
    for fom_key, col in FOM_TO_SOURCE.items():
        v = rec.get(fom_key) if fom_key in ("node_peak", "map_cov") else fom.get(fom_key)
        vals[col] = None if v is None else float(v)
    return vals


def summarize(results: list[dict]) -> None:
    """Paired (target - source) table over every line the results file holds."""
    usable = [r for r in results
              if r.get("status") == "converged" and r.get("fom")]
    print("\n" + "=" * 78)
    print(f"PAIRED SUMMARY   lines={len(results)}  converged={len(usable)}")
    print("=" * 78)
    if not usable:
        print("no converged pairs yet -- nothing to summarize")
        return

    print(f"{'metric':<12}{'n':>4}{'mean src':>12}{'mean tgt':>12}"
          f"{'mean d':>12}{'median d':>12}{'better':>9}")
    improved_fr = 0
    for col in SOURCE_METRICS:
        src, tgt = [], []
        for r in usable:
            s = (r.get("source") or {}).get(col)
            t = target_metrics(r).get(col)
            if s is None or t is None or not np.isfinite(s) or not np.isfinite(t):
                continue
            src.append(float(s))
            tgt.append(float(t))
        if not src:
            print(f"{col:<12}{0:>4}{'--':>12}{'--':>12}{'--':>12}{'--':>12}{'--':>9}")
            continue
        s_arr, t_arr = np.asarray(src), np.asarray(tgt)
        d = t_arr - s_arr
        # "better" = lower, for every column except cyclen (longer is better).
        better = int((d > 0).sum()) if col == "cyclen" else int((d < 0).sum())
        if col == "f_r":
            improved_fr = better
        print(f"{col:<12}{len(d):>4}{s_arr.mean():>12.4f}{t_arr.mean():>12.4f}"
              f"{d.mean():>+12.4f}{float(np.median(d)):>+12.4f}{better:>9d}")

    print(f"\nF_r improved (target < source): {improved_fr} / {len(usable)}")

    feasible = 0
    for r in usable:
        t = target_metrics(r)
        vals = [t.get(c) for c in ("f_r", "cbc_max", "f_q", "cyclen")]
        if any(v is None for v in vals):
            continue
        if (t["f_r"] <= SCREEN["f_r_max"] and t["cbc_max"] <= SCREEN["cbc_max_max"]
                and t["f_q"] <= SCREEN["f_q_max"]
                and SCREEN["cyclen_min"] <= t["cyclen"] <= SCREEN["cyclen_max"]):
            feasible += 1
    print(f"target rows passing the feasibility screen: {feasible} / {len(usable)}"
          f"   (f_r<={SCREEN['f_r_max']}, cbc<={SCREEN['cbc_max_max']:.0f}, "
          f"f_q<={SCREEN['f_q_max']}, {SCREEN['cyclen_min']:.0f}<=cyclen"
          f"<={SCREEN['cyclen_max']:.0f})")

    nonconv = [r for r in results if r.get("status") != "converged"]
    if nonconv:
        print(f"NOT converged: {len(nonconv)} -> "
              + ", ".join(f"{r.get('source_record_id','?')[:12]}:{r.get('status')}"
                          f"/{r.get('failure') or '-'}" for r in nonconv))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="re-evaluate the top-K stored patterns of a source cell under "
                    "a different fresh-fuel pair (paired fuel-lever measurement)")
    ap.add_argument("--source-pair", default="E1_E2")
    ap.add_argument("--source-feed", type=int, default=121)
    ap.add_argument("--target-pair", required=True,
                    help="fresh pair substituted in, e.g. E3_E4 (E1-role first)")
    ap.add_argument("--select", default="mixed", choices=("flat", "minfr", "mixed"),
                    help="flat = node_peak ascending, minfr = f_r ascending, "
                         "mixed = half of each de-duplicated (default)")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N ranked entries before the top-K cut. "
                         "N is interpreted as a SET: the patterns a run with the "
                         "same --select and --k N would have taken are removed "
                         "from the pool, then the same rule picks K from the rest. "
                         "So '--k N' and '--k M --offset N' are guaranteed "
                         "disjoint -- which a rank slice would NOT be under "
                         "--select mixed, whose ranking is not prefix-stable in k. "
                         "Use it to split one sweep across two boxes (a 3-way "
                         "mixed chain is not guaranteed disjoint -- see the "
                         "select_sources docstring). Both boxes must read the "
                         "SAME store: the split is over the eligible pool, so a "
                         "stale records.parquet silently changes which patterns "
                         "the offset skips.")
    ap.add_argument("--package", default="../3_GA_Surrogate/FEASIBLE_PACKAGE")
    ap.add_argument("--exe", default="D:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    ap.add_argument("--run-dir", default="runs/fr_transfer")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cycles", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--relax-screen", action="store_true",
                    help="cross-reactivity transfer mode: drop the source-fuel "
                         "cyclen/CBC band from source selection (peaking axes kept, "
                         "widened 1.25x); target rows are re-screened downstream")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the selection + target assets; run NO MASTER")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(BASE))
    from lpopt.data.schema import unpack_pattern
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.vendor.masterrl.domain import CaseKey

    pkg = (BASE / args.package).resolve()
    if not pkg.is_dir():
        raise SystemExit(f"package not found: {pkg}")
    dims = library_dims(pkg)
    mapping = pair_mapping(args.source_pair, args.target_pair)

    pool = eligible_sources(args.source_pair, args.source_feed,
                            relax=args.relax_screen)
    if pool.empty:
        raise SystemExit(f"no converged, map-bearing, feasible rows for "
                         f"{args.source_pair}/feed-{args.source_feed}")
    if args.offset < 0:
        raise SystemExit(f"--offset must be >= 0, got {args.offset}")
    if len(pool) - args.offset < args.k:
        print(f"WARNING: only {len(pool)} eligible patterns ({max(len(pool) - args.offset, 0)} "
              f"left after --offset {args.offset}), --k {args.k} requested")
    selected = select_sources(pool, args.select, args.k, args.offset)

    run_dir = (BASE / args.run_dir).resolve()
    results_path = run_dir / RESULTS_NAME
    done = already_done(results_path)

    key = CaseKey(args.target_pair, int(args.source_feed))
    resolver = CaseAssetResolver(pkg, library_dims=dims)
    assets = resolver.resolve(key)
    src_assets = resolver.resolve(CaseKey(args.source_pair, int(args.source_feed)))

    print(f"source cell      : {args.source_pair}/feed-{args.source_feed}  "
          f"eligible={len(pool)}  select={args.select}  offset={args.offset}  "
          f"K={len(selected)}")
    print(f"target pair      : {args.target_pair}   mapping="
          + ", ".join(f"{a}->{b}" for a, b in mapping.items()))
    print(f"package          : {pkg}  library_dims(nbatch,ncomp)={dims}")
    print(f"source assets    : {src_assets.case_key.label} "
          f"fallback_level={src_assets.fallback_level} "
          f"restart={src_assets.restart_provenance}")
    print(f"TARGET assets    : {key.label} fallback_level={assets.fallback_level} "
          f"restart={assets.restart_provenance}")
    print(f"                   deck={assets.template_deck_path}")
    print(f"                   notes={'; '.join(assets.notes) or '-'}")
    if assets.fallback_level != 0:
        print("!" * 78)
        print(f"!! WARNING: target pair {args.target_pair} resolves at "
              f"fallback_level={assets.fallback_level} ({assets.kind}).")
        if assets.fallback_level < 0:
            print("!! NOTHING resolved -- this pair has no restart and/or no template")
            print("!! deck in the package.  Every chain would die at staging; add the")
            print("!! assets or pick a pair the package actually carries.")
        else:
            print("!! The restart is NOT this pair's own.  A fallback restart carries a")
            print("!! different burnt-fuel history into every chain and is a CONFOUND for")
            print("!! the fuel measurement: any (target - source) delta then mixes the")
            print("!! lattice change with a restart change.  Do not report this run as a")
            print("!! clean fuel-lever measurement.")
        print("!" * 78)

    # -- build the wave ---------------------------------------------------- #
    entries = []
    skipped = 0
    print(f"\n{'#':>3} {'source_record':<14}{'picked':<7}{'f_r':>9}{'node_pk':>9}"
          f"{'map_cov':>9}{'cyclen':>10}{'CBC':>10}{'f_q':>8}  batches")
    for entry in selected:
        rid = entry["record_id"]
        pattern = unpack_pattern(entry["pattern"])
        target_pattern = substitute_checked(
            pattern, mapping, args.target_pair, args.source_feed, rid)
        s = entry["source"]
        flag = "  SKIP(done)" if rid in done else ""
        print(f"{entry['rank']:>3} {rid[:12]:<14}{entry['picked_by']:<7}"
              f"{s['f_r']:>9.4f}{s['node_peak']:>9.4f}{s['map_cov']:>9.5f}"
              f"{s['cyclen']:>10.3f}{s['cbc_max']:>10.2f}{s['f_q']:>8.4f}"
              f"  {target_pattern.batch_feed()}{flag}")
        if rid in done:
            skipped += 1
            continue
        entry["target_pattern"] = target_pattern
        entries.append(entry)

    if skipped:
        print(f"\nresume: {skipped} of {len(selected)} already in {results_path.name}")

    if args.dry_run:
        print(f"\nDRY RUN -- no MASTER launched.  {len(entries)} chain(s) would run "
              f"into {run_dir}")
        summarize(load_results(results_path))
        return

    if not entries:
        print("nothing left to run")
        summarize(load_results(results_path))
        return

    from lpopt.data.flatness import map_cov, node_peak, slot_values
    from lpopt.search.verify import WaveEntry, WaveVerifier

    verifier = WaveVerifier(
        run_dir=run_dir, package_root=pkg, executable=args.exe,
        workers=args.workers, timeout=args.timeout, max_cycles=args.max_cycles,
        consecutive=2, library_dims=dims, harvest_maps=True,
    )
    wave = [
        WaveEntry(e["target_pattern"], key, assets,
                  {"source_record_id": e["record_id"], "picked_by": e["picked_by"],
                   "rank": e["rank"], "source": e["source"]})
        for e in entries
    ]

    t0 = time.time()
    outcomes = verifier.evaluate_wave(wave)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as fh:
        for oc in outcomes:
            rid = str(oc.meta.get("source_record_id"))
            rec = {
                "source_record_id": rid,
                "source_pair": args.source_pair,
                "source_feed": int(args.source_feed),
                "select": args.select,
                "picked_by": oc.meta.get("picked_by"),
                "rank": oc.meta.get("rank"),
                "source": oc.meta.get("source"),
                "target_pair": args.target_pair,
                "feed": oc.case_key.feed,
                "mapping": mapping,
                "status": oc.status,
                "n_cycles": oc.n_cycles,
                "wall_s": oc.wall_s,
                "restart_provenance": oc.restart_provenance,
                "fallback_level": assets.fallback_level,
                "failure": oc.failure,
                "fom": oc.fom.as_dict() if oc.fom else None,
            }
            if oc.maps is not None:
                # The canonical flatness scalars, NOT a bare np.max: the harvested
                # quarter-core plane carries NaN in every off-slot cell, so a plain
                # max returns NaN (observed 2026-08-02, first tier-0 sweep lost
                # node_peak on all four chains).  lpopt.data.flatness is the single
                # definition shared by the harvest path, the A/B scorer and the
                # promotion gate, so these numbers are comparable to the store's
                # -- which is exactly what the paired delta needs.
                arr = np.asarray(oc.maps, dtype=float)
                sv = slot_values(arr)
                rec["node_peak"] = float(node_peak(sv)[0])
                rec["map_cov"] = float(map_cov(sv)[0])
                # Persist the plane itself -- the verifier purges the case outputs
                # after harvest, so an unsaved map is an unrepeatable measurement.
                np.save(results_path.parent / f"map_{rid[:12]}.npy", arr)
            else:
                rec["node_peak"] = None
                rec["map_cov"] = None
            # Paired deltas inline so a reader of one line never has to rejoin the
            # store to see what the fuel did to THIS pattern.
            src = rec["source"] or {}
            tgt = target_metrics(rec)
            rec["delta"] = {
                c: (None if src.get(c) is None or tgt.get(c) is None
                    else float(tgt[c]) - float(src[c]))
                for c in SOURCE_METRICS
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(json.dumps(rec))
    print(f"\ntotal wall {time.time() - t0:.1f}s -> {results_path}")
    summarize(load_results(results_path))


if __name__ == "__main__":
    main()
