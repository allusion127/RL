"""verify-5-of-20 (objection K2) -- THE MASTER RUNNER.

Pre-registration: ``data/reports/v520_preregistration_20260810.md``, written
2026-08-10 BEFORE the candidates were generated and BEFORE this file existed.
Every rule this script applies is fixed there; nothing here may invent one.

THE QUESTION (memo `kcurve_fusion_memo_20260809.md` section 4, objection K2)

    At the champion's EXISTING F_r ranking skill, does verifying the model's
    top FIVE candidates instead of its top ONE recover more realized F_r than
    any model upgrade on the table -- with zero model changes?

    Simulated at rho = 0.65: top-1-of-20 regret 0.0636 -> top-5 regret 0.0092.
    That -0.0544 is 7.7x the entire measured fuel lever (-0.0070) and needs no
    retraining, no new features and no migration.  It is a SIMULATION.  This
    script measures it.

DESIGN (pre-registration section 2)
    * 20 candidates, frozen with their PREDICTED F_r ranking in
      ``runs/v520/candidates.json`` BEFORE any MASTER call (gate V5 re-checks the
      file's sha256 here).
    * ALL 20 verified -- native restarts only, ``fallback_level 0``.
    * Three policies scored COUNTERFACTUALLY from those same 20 labels, so the
      baseline costs zero extra chains:
          P1     = realized F_r of predicted-rank 1
          P5     = min realized F_r over predicted-ranks 1..5
          ORACLE = min realized F_r over all 20

PRE-REGISTERED CRITERIA -- verbatim, and these numbers do not move
    SUPPORTS K2  :  P1 - P5 >= 0.030
    REFUTES  K2  :  P1 - P5 <  0.010     (then the model, not the verification
                                          policy, is the bottleneck)
    GRAY ZONE    :  otherwise -- report only

VALIDITY GATES (section 5.2) -- ALL must hold or the verdict is VOID
    V1 all 20 chains converged
    V2 all 20 at fallback_level 0 (native restart provenance on every row)
    V3 the 20 patterns pairwise distinct
    V4 none already has a converged row in the store
    V5 candidates.json sha256 matches what the verdict quotes

SECONDARY (section 6, cannot change the verdict)
    Spearman(predicted F_r, realized F_r) over the 20 -- the K2 simulation
    assumed 0.65 -- plus the realized spread, which BOUNDS P1 - P5: a spread
    below 0.030 makes SUPPORTS unreachable by construction and is reported
    FIRST for exactly that reason.

Usage -- MASTER is expensive; the ISSUER launches this, not the agent::

    python v520_run.py --dry-run                     # resolve + plan, NO MASTER
    python v520_run.py --package ../3_GA_Surrogate/FEASIBLE_PACKAGE \
        --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe --workers 12
    python v520_run.py --score-only                  # re-print the table

Resumable: candidates already present in ``runs/v520/v520_results.jsonl`` are
skipped, and results are flushed after every chunk of ``--workers`` chains, so an
interrupted box loses at most one chunk.

WRITES: ``runs/v520/v520_results.jsonl`` and ``runs/v520/map_rank<NN>_<rid12>.npy``.
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

CANDIDATES_NAME = "candidates.json"
RESULTS_NAME = "v520_results.jsonl"

#: Pre-registration section 5.1.  Written as constants so the verdict cannot be
#: computed against a number that was chosen after the labels arrived.
SUPPORT_THRESHOLD = 0.030
REFUTE_THRESHOLD = 0.010

#: Section 2.7 -- the policy the K2 objection is about.
POLICY_K = 5

#: Section 6.1 -- the rank correlation the K2 simulation assumed.
ASSUMED_RHO = 0.65

#: Section 6.3 -- the program feasibility screen (identical to ``fr_transfer.SCREEN``).
SCREEN = {
    "f_r_max": 1.55,
    "cbc_max_max": 1550.0,
    "f_q_max": 2.41,
    "cyclen_min": 620.0,
    "cyclen_max": 645.0,
}

#: Section 3.1 / gate V2 -- this cell's own native restart.
EXPECTED_RESTART = "native:MAS_RST.APRQ_11_0635.19"


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_candidates(path: Path) -> tuple[dict, str]:
    """The frozen ranking + its sha256 (gate V5)."""
    if not path.is_file():
        raise SystemExit(
            f"frozen candidates not found: {path}\n"
            "Run `python v520_gen.py` first (budget-0, no MASTER).")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    cands = payload.get("candidates") or []
    if not cands:
        raise SystemExit(f"{path} holds no candidates")
    ranks = [int(c["predicted_rank"]) for c in cands]
    if sorted(ranks) != list(range(1, len(cands) + 1)):
        raise SystemExit(
            f"{path}: predicted_rank is not 1..{len(cands)} -- the frozen ranking "
            "is corrupt and no policy can be scored from it")
    return payload, hashlib.sha256(raw).hexdigest()


def load_results(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def done_ids(results: list[dict]) -> set[str]:
    return {str(r["record_id"]) for r in results if r.get("record_id")}


def _sha256_file(path) -> str | None:
    """sha256 of one asset, or ``None`` if it is absent/unreadable."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def asset_fingerprint(assets) -> dict:
    """sha256 of the restart + template deck the chains will actually consume.

    Batch 1 ran on the local box and batch 2 runs on 198, off two separate copies
    of FEASIBLE_PACKAGE.  ``fallback_level 0`` and a matching provenance STRING
    prove the resolver picked the right *slot*; they do not prove the two boxes
    hold the same *bytes*.  Recording the hashes makes the cross-box comparison
    auditable after the fact instead of assumed -- and it costs one file read.
    """
    return {
        "restart_path": str(getattr(assets, "restart_path", None)),
        "restart_sha256": _sha256_file(getattr(assets, "restart_path", None)),
        "deck_path": str(getattr(assets, "template_deck_path", None)),
        "deck_sha256": _sha256_file(getattr(assets, "template_deck_path", None)),
        "fallback_level": int(assets.fallback_level),
        "restart_provenance": str(assets.restart_provenance),
    }


def library_dims(package_root: Path) -> tuple[int, int]:
    """``(nbatch, ncomp)`` from ``lib/MAS_XSL`` -- identical to ``fr_transfer.py``."""
    xsl = (package_root / "lib" / "MAS_XSL").read_text(errors="replace")
    comp = sum(1 for ln in xsl.splitlines() if ln.startswith("COMP "))
    refl = sum(1 for ln in xsl.splitlines() if ln.startswith("REFL "))
    return (comp + 3, comp + refl)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _avg_ranks(x: list[float]) -> np.ndarray:
    """Average ranks with tie handling (the definition Spearman needs)."""
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average tied groups
    i = 0
    srt = a[order]
    while i < len(a):
        j = i
        while j + 1 < len(a) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho = Pearson on average ranks (ties handled).

    Implemented locally rather than imported so this script has no dependency
    beyond numpy; it is the textbook definition, and it agrees with
    ``scipy.stats.spearmanr`` including under ties.
    """
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if den <= 0.0:
        return float("nan")
    return float((rx * ry).sum() / den)


# --------------------------------------------------------------------------- #
# scoring -- the pre-registered table
# --------------------------------------------------------------------------- #
def report(payload: dict, cand_sha: str, results: list[dict],
           store_novelty: dict | None) -> None:
    cands = payload["candidates"]
    n_total = len(cands)
    by_rid = {str(r["record_id"]): r for r in results}

    rows = []
    for c in cands:
        r = by_rid.get(str(c["record_id"]))
        rows.append({
            "rank": int(c["predicted_rank"]),
            "record_id": str(c["record_id"]),
            "origin": c.get("origin"),
            "pred_f_r": float(c["pred"]["f_r"]),
            "result": r,
        })
    rows.sort(key=lambda d: d["rank"])

    print("\n" + "=" * 96)
    print("verify-5-of-20 (objection K2) -- PRE-REGISTERED SCORING")
    print("pre-registration: data/reports/v520_preregistration_20260810.md")
    print("=" * 96)
    print(f"candidates.json sha256 : {cand_sha}          <- gate V5: quote this in the verdict")
    print(f"store snapshot sha256  : {(payload.get('store') or {}).get('sha256')}")
    print(f"cell                   : {payload['case']['library_id']} | "
          f"{payload['case']['pair']} | feed {payload['case']['feed']}")
    print(f"model                  : {payload.get('model_dir')}  "
          f"(cond {payload.get('cond_schema')})  -- UNCHANGED, this arm ships no model")

    # -- per-candidate table ------------------------------------------------ #
    print("\n" + "-" * 96)
    print(f"{'rk':>3} {'record_id':<14}{'origin':<10}{'pred F_r':>10}{'REAL F_r':>10}"
          f"{'node_pk':>9}{'map_cov':>9}{'cyclen':>10}{'CBC':>9}{'F_q':>8}"
          f"{'ncyc':>5}  status")
    print("-" * 96)
    for d in rows:
        r = d["result"]
        if r is None:
            print(f"{d['rank']:>3} {d['record_id'][:12]:<14}{str(d['origin']):<10}"
                  f"{d['pred_f_r']:>10.4f}{'--':>10}{'--':>9}{'--':>9}{'--':>10}"
                  f"{'--':>9}{'--':>8}{'--':>5}  PENDING")
            continue
        fom = r.get("fom") or {}

        def _g(k, width=10, prec=4):
            v = fom.get(k)
            return (f"{'--':>{width}}" if v is None
                    else f"{float(v):>{width}.{prec}f}")

        npk = r.get("node_peak")
        mcv = r.get("map_cov")
        print(f"{d['rank']:>3} {d['record_id'][:12]:<14}{str(d['origin']):<10}"
              f"{d['pred_f_r']:>10.4f}{_g('F_r')}"
              f"{('--' if npk is None else f'{npk:.4f}'):>9}"
              f"{('--' if mcv is None else f'{mcv:.5f}'):>9}"
              f"{_g('cyclen', 10, 3)}{_g('CBC_max', 9, 1)}{_g('F_q', 8, 4)}"
              f"{int(r.get('n_cycles') or 0):>5}  {r.get('status')}"
              + (f" / {r.get('failure')}" if r.get("failure") else ""))

    # -- validity gates ----------------------------------------------------- #
    have = [d for d in rows if d["result"] is not None]
    conv = [d for d in have if d["result"].get("status") == "converged"
            and (d["result"].get("fom") or {}).get("F_r") is not None]
    v1 = len(conv) == n_total
    fb = {int(d["result"].get("fallback_level", -1)) for d in have}
    prov = {str(d["result"].get("restart_provenance")) for d in have}
    v2 = bool(have) and fb == {0} and prov == {EXPECTED_RESTART}
    keys = [c.get("pattern_key") or hashlib.blake2b(
        c["pattern"].encode("utf-8"), digest_size=16).hexdigest() for c in cands]
    v3 = len(set(keys)) == n_total
    v4 = None if store_novelty is None else bool(store_novelty.get("all_novel"))

    print("\n" + "-" * 96)
    print("VALIDITY GATES (pre-registration section 5.2) -- ALL must hold or the verdict is VOID")
    print("-" * 96)
    print(f"  V1 all {n_total} chains converged            : "
          f"{'PASS' if v1 else 'FAIL'}  ({len(conv)}/{n_total} converged, "
          f"{len(have)}/{n_total} attempted)")
    print(f"  V2 all at fallback_level 0, native restart : "
          f"{'PASS' if v2 else ('FAIL' if have else 'n/a')}  "
          f"levels={sorted(fb) if have else '-'}  provenance={sorted(prov) if have else '-'}")
    print(f"  V3 20 patterns pairwise distinct           : "
          f"{'PASS' if v3 else 'FAIL'}  ({len(set(keys))} unique of {n_total})")
    if v4 is None:
        print("  V4 none already converged in the store     : "
              "not re-checked this invocation (--check-novelty to re-run it); "
              "it was enforced at generation")
    else:
        print(f"  V4 none already converged in the store     : "
              f"{'PASS' if v4 else 'FAIL'}  "
              f"(store {store_novelty.get('store_rows')} rows, "
              f"{store_novelty.get('hits_any')} pattern/record_id hits)")
    print("  V5 candidates.json sha256                  : printed above; the "
          "verdict must quote it")

    if not conv:
        print("\nno converged label yet -- nothing to score")
        return

    # -- the three policies ------------------------------------------------- #
    realized = {d["rank"]: float(d["result"]["fom"]["F_r"]) for d in conv}
    have_ranks = sorted(realized)
    spread = max(realized.values()) - min(realized.values())

    print("\n" + "-" * 96)
    print("SECONDARY FIRST -- the power disclosure (pre-registration section 6.2)")
    print("-" * 96)
    print(f"  realized F_r over the {len(conv)} converged: min {min(realized.values()):.4f}"
          f"  max {max(realized.values()):.4f}  SPREAD {spread:.4f}")
    print(f"  P1 - P5 <= P1 - ORACLE <= spread.  A spread below {SUPPORT_THRESHOLD:.3f} "
          f"makes SUPPORTS unreachable by construction")
    if spread < SUPPORT_THRESHOLD:
        print("  ** SPREAD IS BELOW THE SUPPORT BAR -- any REFUTES below carries far less")
        print("     information than it appears to: it says 'these 20 elites were alike',")
        print("     NOT 'verifying more does not pay'.  Report it that way.")

    pred = [d["pred_f_r"] for d in conv]
    real = [float(d["result"]["fom"]["F_r"]) for d in conv]
    rho = spearman(pred, real)
    print(f"  Spearman(predicted F_r, realized F_r) over {len(conv)} = {rho:+.4f}"
          f"   (K2 simulation assumed {ASSUMED_RHO:+.2f})")
    if not math.isnan(rho):
        if rho < ASSUMED_RHO - 0.10:
            print("  -> ranking skill BELOW the simulation's assumption: a SUPPORTS is "
                  "understated, a REFUTES needs no rescue")
        elif rho > ASSUMED_RHO + 0.10:
            print("  -> ranking skill ABOVE the simulation's assumption: a REFUTES is "
                  "partly 'the model is already good at this', which is a DIFFERENT "
                  "sentence from 'the policy does not work'.  Thresholds do not move.")

    p1_rank = 1
    if p1_rank not in realized:
        print("\n!! predicted-rank 1 has no converged label -- P1 is undefined and the")
        print("!! verdict is VOID until it is re-run from the UNCHANGED candidates.json")
    top5 = [realized[k] for k in have_ranks if k <= POLICY_K]
    p1 = realized.get(p1_rank)
    p5 = min(top5) if top5 else None
    oracle = min(realized.values())
    oracle_rank = min(have_ranks, key=lambda k: (realized[k], k))

    print("\n" + "=" * 96)
    print("THE PRE-REGISTERED POLICY TABLE (section 2.7) -- one label set, three policies")
    print("=" * 96)
    print(f"  P1     (verify predicted-rank 1 only)      F_r = "
          f"{'undefined' if p1 is None else f'{p1:.4f}'}")
    print(f"  P5     (verify predicted-ranks 1..{POLICY_K})       F_r = "
          f"{'undefined' if p5 is None else f'{p5:.4f}'}"
          f"   [{len(top5)} of {POLICY_K} converged]")
    print(f"  ORACLE (verify all {n_total})                   F_r = {oracle:.4f}"
          f"   [at predicted-rank {oracle_rank}]")
    if p1 is not None and p5 is not None:
        gain = p1 - p5
        print(f"\n  P1 - P5      = {gain:+.4f}      <== THE K2 QUANTITY")
        print(f"  P1 - ORACLE  = {p1 - oracle:+.4f}")
        print(f"  P5 - ORACLE  = {p5 - oracle:+.4f}")
        print(f"\n  reference   simulated policy gain at rho={ASSUMED_RHO}:  0.0544")
        print("  reference   measured fuel lever (8 arms)      : -0.0070")

        print("\n" + "-" * 96)
        if gain >= SUPPORT_THRESHOLD:
            verdict = "SUPPORTS K2"
            rule = f"P1 - P5 = {gain:.4f} >= {SUPPORT_THRESHOLD:.3f}"
            note = ("verify-5-per-batch is the cheap win; every model-upgrade proposal "
                    "must now beat a measured, free baseline on realized F_r per "
                    "MASTER call")
        elif gain < REFUTE_THRESHOLD:
            verdict = "REFUTES K2"
            rule = f"P1 - P5 = {gain:.4f} < {REFUTE_THRESHOLD:.3f}"
            note = ("the verification policy is NOT the cheap win -- the MODEL is the "
                    "bottleneck, and memo section 3's mechanism ordering stands")
        else:
            verdict = "GRAY ZONE"
            rule = (f"{REFUTE_THRESHOLD:.3f} <= P1 - P5 = {gain:.4f} "
                    f"< {SUPPORT_THRESHOLD:.3f}")
            note = "promotes nothing, refutes nothing -- report only"
        gates_ok = v1 and v2 and v3 and (v4 is not False)
        if not gates_ok:
            print(f"VERDICT: VOID  (a section 5.2 validity gate failed; the would-be "
                  f"reading was {verdict})")
            print("  the only admissible repair is re-running the FAILED chains from the")
            print("  UNCHANGED candidates.json -- never re-generating or substituting")
        else:
            print(f"VERDICT: {verdict}")
        print(f"  rule : {rule}")
        print(f"  means: {note}")
        print("-" * 96)

        # section 6.3 -- reported as a curve, only k=5 is registered
        print("\nsensitivity curve (section 6.3 -- REPORTED, NOT A MENU: only k=5 is "
              "registered and only k=5 may be cited)")
        line = []
        for k in (1, 2, 3, 4, 5, 10, n_total):
            vals = [realized[r] for r in have_ranks if r <= k]
            if vals:
                line.append(f"k={k}:{p1 - min(vals):+.4f}")
        print("  P1 - Pk   " + "   ".join(line))

    # -- section 6.3 extras -------------------------------------------------- #
    feas = 0
    for d in conv:
        f = d["result"]["fom"]
        try:
            if (f["F_r"] <= SCREEN["f_r_max"] and f["CBC_max"] <= SCREEN["cbc_max_max"]
                    and f["F_q"] <= SCREEN["f_q_max"]
                    and SCREEN["cyclen_min"] <= f["cyclen"] <= SCREEN["cyclen_max"]):
                feas += 1
        except (KeyError, TypeError):
            pass
    print(f"\npassing the program feasibility screen: {feas} / {len(conv)}"
          f"   (F_r<={SCREEN['f_r_max']}, CBC<={SCREEN['cbc_max_max']:.0f}, "
          f"F_q<={SCREEN['f_q_max']}, {SCREEN['cyclen_min']:.0f}<=cyclen"
          f"<={SCREEN['cyclen_max']:.0f})")
    below = [r for r in realized.values() if r < 1.4636]
    print(f"below the cell incumbent F_r 1.4636: {len(below)} / {len(conv)}"
          + (f"   (best {min(below):.4f})" if below else "")
          + "   [reportable event, NOT a criterion]")


def check_novelty(payload: dict) -> dict:
    """Re-run gate V4 against the store on disk (read-only)."""
    import pandas as pd

    store_path = BASE / "data/store/records.parquet"
    df = pd.read_parquet(store_path, columns=["record_id", "pattern", "converged"])
    conv = df["converged"] == True  # noqa: E712
    pat_all = set(df["pattern"].astype(str))
    rid_all = set(df["record_id"].astype(str))
    pat_cv = set(df.loc[conv, "pattern"].astype(str))
    rid_cv = set(df.loc[conv, "record_id"].astype(str))
    hits_any = hits_conv = 0
    for c in payload["candidates"]:
        if c["pattern"] in pat_all or str(c["record_id"]) in rid_all:
            hits_any += 1
        if c["pattern"] in pat_cv or str(c["record_id"]) in rid_cv:
            hits_conv += 1
    return {"store_rows": int(len(df)), "hits_any": hits_any,
            "hits_converged": hits_conv, "all_novel": hits_any == 0}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="verify all 20 frozen candidates and score P1 / P5 / ORACLE "
                    "against the pre-registered rule")
    ap.add_argument("--run-dir", default="runs/v520")
    ap.add_argument("--package", default="../3_GA_Surrogate/FEASIBLE_PACKAGE")
    ap.add_argument("--exe", default="D:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-cycles", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--chunk", type=int, default=None,
                    help="chains per flush (default = --workers).  Results are "
                         "appended after every chunk, so an interrupted box loses "
                         "at most one chunk of work.")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve assets + print the plan; run NO MASTER")
    ap.add_argument("--score-only", action="store_true",
                    help="re-print the pre-registered table from existing results")
    ap.add_argument("--check-novelty", action="store_true",
                    help="re-run gate V4 against data/store (read-only)")
    args = ap.parse_args()

    sys.path.insert(0, str(BASE))

    run_dir = (BASE / args.run_dir).resolve()
    cand_path = run_dir / CANDIDATES_NAME
    results_path = run_dir / RESULTS_NAME
    payload, cand_sha = load_candidates(cand_path)
    cands = payload["candidates"]

    results = load_results(results_path)
    done = done_ids(results)

    novelty = None
    if args.check_novelty or args.score_only or args.dry_run:
        try:
            novelty = check_novelty(payload)
        except Exception as exc:  # noqa: BLE001 -- a gate readout must never kill a run
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

    print("=" * 96)
    print("verify-5-of-20 (objection K2) -- MASTER RUNNER")
    print("pre-registration: data/reports/v520_preregistration_20260810.md")
    print("=" * 96)
    print(f"candidates       : {cand_path}   n={len(cands)}")
    print(f"sha256 (gate V5) : {cand_sha}")
    print(f"cell             : {payload['case']['library_id']} | {pair} | feed {feed}")
    print(f"package          : {pkg}   library_dims(nbatch,ncomp)={dims}")
    print(f"assets           : {key.label}  fallback_level={assets.fallback_level} "
          f"kind={assets.kind}")
    print(f"                   restart={assets.restart_provenance}")
    print(f"                   deck={assets.template_deck_path}")
    fp = asset_fingerprint(assets)
    print(f"asset fingerprint: restart sha256 {fp['restart_sha256']}")
    print(f"                   deck    sha256 {fp['deck_sha256']}")
    print("                   (batch 1 ran on a different box off a different copy "
          "of FEASIBLE_PACKAGE;")
    print("                    these two hashes are what make the cross-box "
          "comparison auditable)")
    if assets.fallback_level != 0 or assets.restart_provenance != EXPECTED_RESTART:
        print("!" * 96)
        print("!! GATE V2 CANNOT PASS: the resolver did not land on this cell's own")
        print(f"!! native restart ({EXPECTED_RESTART}).  A fallback restart carries a")
        print("!! different burnt-fuel history into every chain, which makes the 20")
        print("!! labels non-comparable to each other AND to the store.  Fix the")
        print("!! package before spending 20 chains.")
        print("!" * 96)
        if not args.dry_run:
            return 2
    if novelty is not None and not novelty["all_novel"]:
        print(f"!! GATE V4 FAILS: {novelty['hits_any']} of {len(cands)} candidates are "
              f"already in the store -- re-generate before spending MASTER")
        if not args.dry_run:
            return 2

    # -- build the wave ----------------------------------------------------- #
    entries = []
    print(f"\n{'rk':>3} {'record_id':<14}{'origin':<10}{'pred F_r':>10}"
          f"{'pred cyc':>10}{'p_feas':>8}  batches")
    for c in cands:
        rid = str(c["record_id"])
        pattern = unpack_pattern(c["pattern"])
        pattern.validate_case(pair, feed)
        flag = "  SKIP(done)" if rid in done else ""
        print(f"{int(c['predicted_rank']):>3} {rid[:12]:<14}{str(c['origin']):<10}"
              f"{float(c['pred']['f_r']):>10.4f}{float(c['pred']['cyclen']):>10.2f}"
              f"{float(c['p_feas']):>8.3f}  {pattern.batch_feed()}{flag}")
        if rid in done:
            continue
        entries.append((c, pattern))

    if done:
        print(f"\nresume: {len(done)} of {len(cands)} already in {results_path.name}")

    if args.dry_run:
        # Import the LIVE-path modules here too.  A dry run that skips them would
        # pass on a box whose kit lacks `lpopt.data.flatness` or a compatible
        # `WaveVerifier` signature, and the failure would surface 20 chains later
        # -- which is exactly what a dry run exists to prevent.  This is a
        # cross-box readiness check, not decoration.
        try:
            from lpopt.data.flatness import map_cov, node_peak, slot_values  # noqa: F401
            from lpopt.search.verify import WaveEntry, WaveVerifier  # noqa: F401
            probe = WaveVerifier(
                run_dir=run_dir / "_dryrun_probe", package_root=pkg,
                executable=args.exe, workers=args.workers, timeout=args.timeout,
                max_cycles=args.max_cycles, consecutive=2, library_dims=dims,
                harvest_maps=True,
            )
            print(f"\nimport probe     : OK -- lpopt.data.flatness + "
                  f"WaveVerifier construct cleanly on this box "
                  f"(n_workers={probe.n_workers})")
            exe_ok = Path(str(args.exe)).is_file()
            print(f"MASTER exe       : {args.exe}  present={exe_ok}"
                  + ("" if exe_ok else "   <-- NOT FOUND: the live run would die at staging"))
        except Exception as exc:  # noqa: BLE001
            print(f"\nimport probe     : FAILED -- {type(exc).__name__}: {exc}")
            print("this box cannot run the live path; fix before launching")
            return 3
        print(f"\nDRY RUN -- no MASTER launched.  {len(entries)} chain(s) would run "
              f"into {run_dir}")
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
        consecutive=2, library_dims=dims, harvest_maps=True,
    )

    chunk = int(args.chunk or args.workers)
    chunk = max(1, chunk)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for start in range(0, len(entries), chunk):
        block = entries[start:start + chunk]
        wave = [
            WaveEntry(pat, key, assets,
                      {"record_id": str(c["record_id"]),
                       "predicted_rank": int(c["predicted_rank"]),
                       "pred_f_r": float(c["pred"]["f_r"]),
                       "origin": c.get("origin")})
            for c, pat in block
        ]
        print(f"\n>> wave {start // chunk + 1}: {len(wave)} chain(s), "
              f"{args.workers} workers")
        outcomes = verifier.evaluate_wave(wave)
        with results_path.open("a", encoding="utf-8") as fh:
            for oc in outcomes:
                rid = str(oc.meta.get("record_id"))
                rank = int(oc.meta.get("predicted_rank"))
                rec = {
                    "record_id": rid,
                    "predicted_rank": rank,
                    "predicted_f_r": oc.meta.get("pred_f_r"),
                    "origin": oc.meta.get("origin"),
                    "pair": pair,
                    "feed": int(oc.case_key.feed),
                    "library_id": payload["case"]["library_id"],
                    "status": oc.status,
                    "n_cycles": oc.n_cycles,
                    "wall_s": oc.wall_s,
                    "restart_provenance": oc.restart_provenance,
                    "fallback_level": assets.fallback_level,
                    "converged_at_cap": bool(oc.converged_at_cap),
                    "tolerance_margin": oc.tolerance_margin,
                    "failure": oc.failure,
                    "fom": oc.fom.as_dict() if oc.fom else None,
                    "candidates_sha256": cand_sha,
                    "assets": fp,
                }
                if oc.maps is not None:
                    # The canonical flatness scalars, NOT a bare np.max: the
                    # harvested quarter-core plane carries NaN in every off-slot
                    # cell, so a plain max returns NaN (observed 2026-08-02 --
                    # it silently lost node_peak on all four chains of the first
                    # tier-0 sweep).  ``lpopt.data.flatness`` is the single
                    # definition shared by the harvest path, the A/B scorer and
                    # the promotion gate, so these numbers are comparable to the
                    # store's -- which is what the policy comparison needs.
                    arr = np.asarray(oc.maps, dtype=float)
                    sv = slot_values(arr)
                    rec["node_peak"] = float(node_peak(sv)[0])
                    rec["map_cov"] = float(map_cov(sv)[0])
                    # Persist the plane: the verifier purges the case outputs
                    # after harvest, so an unsaved map is an unrepeatable
                    # measurement.
                    np.save(run_dir / f"map_rank{rank:02d}_{rid[:12]}.npy", arr)
                else:
                    rec["node_peak"] = None
                    rec["map_cov"] = None
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                fr = (rec["fom"] or {}).get("F_r")
                print(f"   rank {rank:>2} {rid[:12]}  {rec['status']:<13}"
                      f"pred {rec['predicted_f_r']:.4f} -> "
                      f"REAL {'--' if fr is None else f'{fr:.4f}'}"
                      f"  ({rec['wall_s']:.0f}s)")

    print(f"\ntotal wall {time.time() - t0:.1f}s -> {results_path}")
    try:
        novelty = check_novelty(payload)
    except Exception:  # noqa: BLE001
        novelty = None
    report(payload, cand_sha, load_results(results_path), novelty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
