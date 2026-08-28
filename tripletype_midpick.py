"""Pick the MID fresh type for the first 3-type campaign — P5 vs T1 — by serving
predicted pin burnup under the v7 champion.

Registered procedure (`data/reports/tripletype_f125_prereg_20260817.md` §3):

* take the DONOR pair's converged f125 store rows (the same rows
  ``[search] elite_seed_cases`` will hand the campaign as elite parents);
* rebuild each as a genome and apply ``graded_morph`` under each candidate
  alphabet — the campaign's own cold-start operator, same fraction default,
  both arms driven by the SAME rng seed sequence so the two arms differ only in
  which type the morph writes;
* serve the v7 ensemble and read surrogate column 6 (``max_pin_burnup``) and
  column 0 (``F_r``);
* the arm whose predicted pin distribution sits further under the 78 gate wins,
  because pin is this cell's thin axis (2-type f125 winners: 76.96-77.96).

Nothing here writes to the store or the deck; it prints a table and a verdict.

    python tripletype_midpick.py --model data/models/s1h
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent

HOT = "P6253Z1G06N24"       # S3, e 5.7861, n_gd 24
COLD = "P6253Z2G10N24"      # S5, e 5.6023, n_gd 24
MIDS = {"P5": "P6253Z2G08N16",   # e 5.6685, n_gd 16 — uniform grading
        "T1": "P6253Z2G10N20"}   # e 5.6386, n_gd 20 — n_gd preserving
PAIR = f"{HOT}_{COLD}"
FEED = 125
PIN_GATE = 78.0
N_SEEDS_PER_PARENT = 4       # morph draws per parent (inner/outer slice is random)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/s1h")
    ap.add_argument("--store", default="data/store")
    ap.add_argument("--seed", type=int, default=5695)
    ap.add_argument("--out", default="data/reports/tripletype_midpick_20260817.json")
    args = ap.parse_args()

    from lpopt.data.schema import unpack_pattern
    from lpopt.model.model_api import PosValCnnBackend
    from lpopt.search.construct import CaseKey
    from lpopt.search.genome import GeneralOrbitGenome, graded_morph

    df = pd.read_parquet(Path(args.store) / "records.parquet")
    sub = df[(df["case_pair"].astype(str) == PAIR) & (df["feed"] == FEED)
             & (df["converged"] == True)]  # noqa: E712
    sub = sub.sort_values("f_r")
    print(f"donor parents: {len(sub)} converged f125 rows, "
          f"F_r {sub['f_r'].min():.4f} .. {sub['f_r'].max():.4f}")

    model = PosValCnnBackend.from_dir(
        Path(args.model), store_dir=args.store, library_id="paramA", device="cpu")
    print(f"model {args.model}: cond_schema={model.cond_schema} "
          f"targets={model.target_names}")

    parents = []
    for _, row in sub.iterrows():
        try:
            pat = unpack_pattern(str(row["pattern"]))
            parents.append((str(row["record_id"]), float(row["f_r"]), pat))
        except (ValueError, KeyError):
            continue

    results: dict[str, dict] = {}
    for alias, mid in MIDS.items():
        batches = tuple(sorted((HOT, mid, COLD)))
        rng = random.Random(args.seed)          # SAME seed for both arms
        pats, tagged = [], []
        for rid, fr, pat in parents:
            # feed 125 => N=31 > 30, so the parent is a single-cycle-discharge
            # core; the campaign's own CaseContext.allow_discharge says the same.
            genome = GeneralOrbitGenome.from_pattern(
                pat, allow_single_cycle_discharge=True)
            for _ in range(N_SEEDS_PER_PARENT):
                child = graded_morph(genome, rng, batches)
                counts = child.batch_counts
                if mid not in counts:
                    continue
                pats.append(child.to_pattern())
                tagged.append((rid, fr, dict(counts)))
        if not pats:
            print(f"{alias}: graded_morph produced no 3-type child — SKIPPED")
            continue
        case = CaseKey(f"{HOT}_{mid}_{COLD}", FEED)
        pred = model.predict(pats, case)
        pin = np.asarray(pred.mean)[:, 6]
        fr = np.asarray(pred.mean)[:, 0]
        cbc = np.asarray(pred.mean)[:, 1]
        ok = np.isfinite(pin)
        results[alias] = {
            "mid_type": mid, "case": case.pair, "n_seeds": int(len(pats)),
            "pin_n_finite": int(ok.sum()),
            "pin_min": float(np.nanmin(pin)), "pin_p50": float(np.nanmedian(pin)),
            "pin_max": float(np.nanmax(pin)),
            "pin_frac_under_gate": float(np.mean(pin[ok] <= PIN_GATE)) if ok.any() else float("nan"),
            "f_r_min": float(np.nanmin(fr)), "f_r_p50": float(np.nanmedian(fr)),
            "cbc_p50": float(np.nanmedian(cbc)),
            "mid_frac_p50": float(np.median([c.get(mid, 0) / max(sum(c.values()), 1)
                                             for _, _, c in tagged])),
        }
        r = results[alias]
        print(f"\n{alias} ({mid})  n={r['n_seeds']}")
        print(f"  pred pin BU   min {r['pin_min']:.3f}  p50 {r['pin_p50']:.3f}  "
              f"max {r['pin_max']:.3f}   under {PIN_GATE}: "
              f"{100*r['pin_frac_under_gate']:.1f}%")
        print(f"  pred F_r      min {r['f_r_min']:.4f}  p50 {r['f_r_p50']:.4f}")
        print(f"  pred CBC p50  {r['cbc_p50']:.2f}      mid feed frac p50 "
              f"{r['mid_frac_p50']:.3f}")

    if len(results) == 2:
        a, b = "P5", "T1"
        # DECISION RULE, fixed here in code before the numbers were seen: the mid
        # that protects the thin axis wins on p50 predicted pin; F_r breaks a tie
        # only when the pin p50 gap is below the pin head's own in-cell MAE (1.84).
        dpin = results[a]["pin_p50"] - results[b]["pin_p50"]
        if abs(dpin) >= 1.84:
            winner = a if dpin < 0 else b
            why = f"pin p50 gap {dpin:+.3f} exceeds the pin head's in-cell MAE 1.84"
        else:
            winner = a if results[a]["f_r_p50"] < results[b]["f_r_p50"] else b
            why = (f"pin p50 gap {dpin:+.3f} is INSIDE the pin head's in-cell MAE "
                   f"1.84 — not resolvable on pin; decided on predicted F_r p50")
        results["verdict"] = {"winner": winner, "mid_type": MIDS[winner],
                              "case": f"{HOT}_{MIDS[winner]}_{COLD}", "why": why,
                              "pin_p50_delta_P5_minus_T1": dpin}
        print(f"\nVERDICT: mid = {winner} ({MIDS[winner]})\n  {why}")

    out = REPO / args.out
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
