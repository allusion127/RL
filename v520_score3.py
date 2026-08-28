"""verify-5-of-20 -- THE THREE-BATCH SCORER.  Reads labels; runs NO MASTER.

Implements, and only implements, the rule registered in
``data/reports/v520_addendum_b3_20260810.md`` section 5.1, with the mandatory
co-report of section 6.  It exists so the three-batch verdict is computed by the
registered arithmetic rather than by hand, where drift and rounding creep in.

    SUPPORTS K2   iff  mean(P1-P5) >= 0.030  AND  >= 2 of 3 batches P1-P5 >= 0.010
    REFUTES  K2   iff  mean(P1-P5) <  0.010  AND  >= 2 of 3 batches P1-P5 <  0.010
    UNRESOLVED at n=3   otherwise

The concordance guard is the point: P1-P5 is non-negative and right-skewed, so a
three-sample mean is dominated by its largest draw, and the observed 11x
batch-to-batch ratio (0.0071 vs 0.0784) is not hypothetical.  Without the guard a
single batch would decide the programme.

MANDATORY CO-REPORT (addendum section 6).  The SUPPORTS bar 0.030 sits BELOW the
measured random-ranking null (b1 0.0404 / b2 0.0242, mean 0.0323): a ZERO-SKILL
ranker would "SUPPORT K2" under the rule.  So every run prints, per batch, the
exact random-ranking null

    E_null[P1 - P5] = mean(F_r) - E[min of 5 drawn from the same 20]

computed by enumeration over order statistics from that batch's OWN 20 labels
(no extra MASTER, no simulation), and the three-batch mean excess over it.  If
that mean excess is <= 0, the registered verdict text may NOT claim the
verification policy beats model upgrades.

Usage::

    python v520_score3.py                     # all three batches
    python v520_score3.py --batch runs/v520 --batch runs/v520_b2

Reports median-of-three as a REGISTERED ROBUSTNESS READOUT that is explicitly
NOT a criterion (addendum 5.1) -- named so it can never be introduced later as
though it had been.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

BASE = Path(__file__).resolve().parent

#: addendum section 5.1 -- identical to the per-batch thresholds, NOT re-tuned.
SUPPORT_THRESHOLD = 0.030
REFUTE_THRESHOLD = 0.010
POLICY_K = 5

DEFAULT_BATCHES = ("runs/v520", "runs/v520_b2", "runs/v520_b3")
BATCH_LABELS = {"v520": "batch 1", "v520_b2": "batch 2", "v520_b3": "batch 3"}


def expected_min_of_k(values: list[float], k: int) -> float:
    """``E[min of k drawn without replacement]`` -- exact, by order statistics.

    ``P(value at ascending index i is the minimum of a k-subset)`` is
    ``C(n-i-1, k-1) / C(n, k)``: choose the other k-1 members from the strictly
    larger elements.  Exact enumeration, not a resampling estimate, so the null
    carries no Monte-Carlo error of its own.
    """
    s = sorted(values)
    n = len(s)
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    tot = comb(n, k)
    out = 0.0
    for i, v in enumerate(s):
        rest = n - i - 1
        if rest >= k - 1:
            out += v * comb(rest, k - 1) / tot
    return out


def score_batch(run_dir: Path) -> dict | None:
    """One batch's registered readouts, or ``None`` if it has no complete labels."""
    import v520_run as R

    cand_path = run_dir / "candidates.json"
    res_path = run_dir / "v520_results.jsonl"
    if not cand_path.is_file() or not res_path.is_file():
        return None
    payload, cand_sha = R.load_candidates(cand_path)
    results = R.load_results(res_path)
    by_rid = {str(r["record_id"]): r for r in results}

    ranked: list[tuple[int, float, float]] = []   # (rank, predicted, realized)
    n_total = len(payload["candidates"])
    for c in payload["candidates"]:
        r = by_rid.get(str(c["record_id"]))
        if r is None or r.get("status") != "converged":
            continue
        fom = r.get("fom") or {}
        if fom.get("F_r") is None:
            continue
        ranked.append((int(c["predicted_rank"]), float(c["pred"]["f_r"]),
                       float(fom["F_r"])))
    if not ranked:
        return None
    ranked.sort()
    realized = {rk: real for rk, _p, real in ranked}
    fr = [real for _rk, _p, real in ranked]

    p1 = realized.get(1)
    top5 = [realized[k] for k in sorted(realized) if k <= POLICY_K]
    p5 = min(top5) if top5 else None
    oracle = min(fr)
    oracle_rank = min(realized, key=lambda k: (realized[k], k))
    null = (sum(fr) / len(fr)) - expected_min_of_k(fr, POLICY_K)

    # every gate the per-batch rule requires, re-checked here
    have = [by_rid.get(str(c["record_id"])) for c in payload["candidates"]]
    have = [h for h in have if h is not None]
    v1 = len(ranked) == n_total
    v2 = bool(have) and {int(h.get("fallback_level", -1)) for h in have} == {0} \
        and {str(h.get("restart_provenance")) for h in have} == {R.EXPECTED_RESTART}
    keys = {c["pattern"] for c in payload["candidates"]}
    v3 = len(keys) == n_total

    return {
        "run_dir": run_dir.name,
        "label": BATCH_LABELS.get(run_dir.name, run_dir.name),
        "seed": (payload.get("counts") or {}).get("by_seed"),
        "candidates_sha256": cand_sha,
        "store_sha256": (payload.get("store") or {}).get("sha256"),
        "n": n_total, "n_converged": len(ranked),
        "P1": p1, "P5": p5, "ORACLE": oracle, "oracle_rank": oracle_rank,
        "gain": (None if (p1 is None or p5 is None) else p1 - p5),
        "spread": max(fr) - min(fr),
        "spearman": R.spearman([p for _rk, p, _r in ranked],
                               [r for _rk, _p, r in ranked]),
        "null": null,
        "under_150": sum(1 for x in fr if x < 1.50),
        "best": min(fr),
        "gates": {"V1": v1, "V2": v2, "V3": v3},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="three-batch verify-5-of-20 verdict, per the registered rule")
    ap.add_argument("--batch", action="append", default=[],
                    help="run dir (repeatable); default = the three registered ones")
    args = ap.parse_args()
    sys.path.insert(0, str(BASE))

    dirs = [(BASE / b) for b in (args.batch or DEFAULT_BATCHES)]
    batches = []
    for d in dirs:
        s = score_batch(d)
        if s is None:
            print(f"[pending] {d.name}: no complete labels yet")
        else:
            batches.append(s)

    print("=" * 100)
    print("verify-5-of-20 (objection K2) -- THREE-BATCH SCORING")
    print("rule: data/reports/v520_addendum_b3_20260810.md section 5.1")
    print("=" * 100)
    print(f"{'batch':<9}{'seed':>7}{'n':>5}{'P1':>9}{'P5':>9}{'ORACLE':>9}{'orank':>6}"
          f"{'P1-P5':>9}{'spread':>9}{'rho':>8}{'null':>9}{'excess':>9}  gates")
    for b in batches:
        seed = ",".join((b["seed"] or {}).keys()) if b["seed"] else "?"
        g = b["gain"]
        print(f"{b['label']:<9}{seed:>7}{b['n_converged']:>5}"
              f"{b['P1']:>9.4f}{b['P5']:>9.4f}{b['ORACLE']:>9.4f}{b['oracle_rank']:>6}"
              f"{g:>+9.4f}{b['spread']:>9.4f}{b['spearman']:>+8.3f}"
              f"{b['null']:>9.4f}{g - b['null']:>+9.4f}  "
              + ("all PASS" if all(b["gates"].values())
                 else "FAIL:" + ",".join(k for k, v in b["gates"].items() if not v)))

    if len(batches) < 3:
        print(f"\n{len(batches)}/3 batches scored -- the three-batch verdict needs all "
              "three.  No partial verdict is issued (addendum section 5).")
        return 0

    gains = [b["gain"] for b in batches]
    nulls = [b["null"] for b in batches]
    mean = sum(gains) / len(gains)
    med = sorted(gains)[len(gains) // 2]
    n_ge = sum(1 for g in gains if g >= REFUTE_THRESHOLD)
    n_lt = len(gains) - n_ge
    excess = [g - n for g, n in zip(gains, nulls)]
    mean_excess = sum(excess) / len(excess)
    n_pos_excess = sum(1 for e in excess if e > 0)

    print("\n" + "-" * 100)
    print("COMBINED (addendum section 5.1) -- the mean is the registered estimator")
    print("-" * 100)
    print(f"  per-batch P1-P5 : " + "   ".join(f"{g:+.4f}" for g in gains))
    print(f"  MEAN            : {mean:+.4f}     <== the registered combined statistic")
    print(f"  median          : {med:+.4f}     [REGISTERED ROBUSTNESS READOUT, NOT A "
          f"CRITERION]")
    print(f"  concordance     : {n_ge} of 3 batches >= {REFUTE_THRESHOLD:.3f}, "
          f"{n_lt} of 3 below")

    gates_ok = all(all(b["gates"].values()) for b in batches)
    if mean >= SUPPORT_THRESHOLD and n_ge >= 2:
        verdict, rule = "SUPPORTS K2", (
            f"mean {mean:.4f} >= {SUPPORT_THRESHOLD:.3f} AND {n_ge}/3 batches "
            f">= {REFUTE_THRESHOLD:.3f}")
    elif mean < REFUTE_THRESHOLD and n_lt >= 2:
        verdict, rule = "REFUTES K2", (
            f"mean {mean:.4f} < {REFUTE_THRESHOLD:.3f} AND {n_lt}/3 batches "
            f"< {REFUTE_THRESHOLD:.3f}")
    else:
        verdict, rule = "UNRESOLVED at n=3", (
            f"mean {mean:.4f} and the {n_ge}/{n_lt} split do not concur")

    print("\n" + "=" * 100)
    print(f"VERDICT: {verdict}" + ("" if gates_ok else "   -- VOID (a validity gate failed)"))
    print(f"  rule : {rule}")
    print("=" * 100)

    # -- the mandatory co-report (addendum section 6) ----------------------- #
    print("\nMANDATORY CO-REPORT -- the random-ranking null (addendum section 6)")
    print("-" * 100)
    print(f"  per-batch null  : " + "   ".join(f"{n:.4f}" for n in nulls)
          + f"   (mean {sum(nulls) / len(nulls):.4f})")
    print(f"  excess over null: " + "   ".join(f"{e:+.4f}" for e in excess))
    print(f"  MEAN EXCESS     : {mean_excess:+.4f}   "
          f"({n_pos_excess}/3 batches positive)")
    print(f"  the SUPPORTS bar is {SUPPORT_THRESHOLD:.3f}; the mean null is "
          f"{sum(nulls) / len(nulls):.4f}")
    if sum(nulls) / len(nulls) > SUPPORT_THRESHOLD:
        print("  ** THE BAR SITS BELOW THE NULL: a ZERO-SKILL ranker would clear it. **")
    if verdict.startswith("SUPPORTS"):
        if mean_excess <= 0:
            print("\n  REGISTERED VERDICT TEXT (section 6.2, binding -- mean excess <= 0):")
            print("    'top-5 verification recovers F_r because the champion cannot rank")
            print("     within an elite pool -- which is an argument for repairing the")
            print("     ranker, not for preferring the policy to it.'")
            print("    The claim that the verification policy beats model upgrades MAY NOT")
            print("    be made on this evidence.")
        elif n_pos_excess >= 2:
            print("\n  section 6.2: mean excess > 0 and the sign is consistent in "
                  f"{n_pos_excess}/3 batches --")
            print("    the policy gain EXCEEDS what a broken ranker explains, and K2's")
            print("    cost argument stands on its own terms.")
        else:
            print("\n  section 6.2: mean excess > 0 but the sign is NOT consistent "
                  f"({n_pos_excess}/3 positive) --")
            print("    report the excess with its scatter; the stronger reading is not "
                  "available.")
    print("\nproduction readout (verdict-inert): "
          + "   ".join(f"{b['label']} {b['under_150']}/{b['n_converged']} under 1.50, "
                       f"best {b['best']:.4f}" for b in batches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
