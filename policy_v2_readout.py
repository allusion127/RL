"""POST-HOC slices of the v2 gate fold.

Everything here is declared post-hoc in ``data/reports/policy_v2_results_20260817.md``
and gates nothing.  The pre-registered numbers are produced by
``python -m lpopt.policy.train_v2 --tables data/models/policy_v2/metrics.json``
and are not recomputed here.

Two readouts the gate cannot give on its own:

``strata``  the blind-vs-measured table by (move_class, fresh_radial_dir) — the
            table that diagnosed v1's failure (ablation §3).  It asks whether v2
            still inverts the ``batch_flip`` axis.
``slices``  the gate fold split into its INTERVENTIONAL half (verified single
            moves off elite parents, balanced by construction) and its CAMPAIGN
            half (``lpopt_genome`` optimiser edges).  The two are different
            questions and pooling them hides which one v2 answers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))


def _load(args):
    from lpopt.policy.v2 import build_splits_v2, load_universe_v2

    steps = load_universe_v2(args.steps)
    fold = build_splits_v2(steps, seed=args.base_seed)
    gate = steps[fold == "gate_cur"].reset_index(drop=True)
    z = np.load(args.probs)
    probs = z["gate_cur"].mean(axis=0)              # 5-seed ensemble
    if len(probs) != len(gate):
        raise SystemExit(f"probs {len(probs)} != gate rows {len(gate)} - "
                         f"the split or the corpus moved since the run")
    v1 = pd.read_csv(args.v1_baseline)
    gate["p_v2_fr"] = probs[:, 0]
    gate["p_v2_flat"] = probs[:, 1]
    for h in ("fr", "flat"):
        gate[f"p_v1_{h}"] = gate["child_record_id"].map(
            v1.set_index("child_record_id")[f"p_improve_{h}"]).to_numpy(float)
    return steps, fold, gate


def cmd_strata(args) -> int:
    _, _, gate = _load(args)
    g = gate[gate["improved_fr"].notna()].copy()
    t = g.groupby(["move_class", "fresh_radial_dir"]).agg(
        n=("child_record_id", "size"),
        v2_pred=("p_v2_fr", "mean"),
        v1_pred=("p_v1_fr", "mean"),
        measured=("improved_fr", "mean"),
        mean_d_f_r=("d_f_r", "mean"),
        best_d_f_r=("d_f_r", "min"),
    )
    print("## v2 vs v1 predicted, against measured, by stratum (gate fold, head `fr`)")
    print("NOTE v2's column is a normalized clipped EXPECTED IMPROVEMENT in [0,1],")
    print("     v1's is a PROBABILITY of improvement - compare RANKS, not levels.")
    print()
    print(t.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    for name in ("v2", "v1"):
        r = t[f"{name}_pred"].corr(t["measured"], method="spearman")
        print(f"  {name}: Spearman(stratum mean score, stratum improving rate) "
              f"= {r:+.3f} over {len(t)} strata")
    return 0


def cmd_slices(args) -> int:
    from lpopt.policy.train_v2 import (
        BASELINES, baseline_scores_v2, regret_at_k,
    )
    from lpopt.policy.train import auc, parent_blocked_auc

    steps, fold, gate = _load(args)
    train = steps[fold == "train"]
    print("## gate fold, sliced (POST-HOC - gates nothing)")
    print()
    for label, sel in (("interventional", gate["interventional"].to_numpy(bool)),
                       ("campaign", ~gate["interventional"].to_numpy(bool)),
                       ("ALL", np.ones(len(gate), bool))):
        blk = gate[sel]
        for head in ("fr", "flat"):
            col = "improved_fr" if head == "fr" else "improved_flat"
            sub = blk[blk[col].notna()]
            if len(sub) < 30:
                continue
            y = sub[col].astype(bool).to_numpy().astype(float)
            sc = {"policy": sub[f"p_v2_{head}"].to_numpy(float),
                  **baseline_scores_v2(sub, train, head,
                                       pd.DataFrame({
                                           "child_record_id": blk["child_record_id"],
                                           f"p_improve_{head}": blk[f"p_v1_{head}"]}))}
            parents = sub["parent_record_id"].to_numpy()
            print(f"### {label} / {head}  n={len(sub)} base={y.mean():.3f} "
                  f"parents={sub['parent_record_id'].nunique()}")
            for k in ("policy", *BASELINES):
                s = np.nan_to_num(sc[k], nan=-1e9)
                pb, npair = parent_blocked_auc(s, y, parents)
                print(f"  {k:<12} AUC {auc(s, y):.3f}   parent-blocked "
                      f"{pb:.3f} ({npair} pairs)")
            # regret@8 on this slice
            okr = blk["both_converged"].fillna(False).to_numpy(bool)
            gain = -blk[("d_f_r" if head == "fr" else "d_node_peak")].to_numpy(float)
            okr &= np.isfinite(gain)
            if okr.sum():
                line = []
                for k in ("policy", *BASELINES):
                    src = {"policy": blk[f"p_v2_{head}"].to_numpy(float),
                           **baseline_scores_v2(blk, train, head,
                                                pd.DataFrame({
                                                    "child_record_id": blk["child_record_id"],
                                                    f"p_improve_{head}": blk[f"p_v1_{head}"]}))}
                    s = np.nan_to_num(src[k], nan=-1e9)
                    a, _, keys = regret_at_k(s[okr], gain[okr],
                                             blk["parent_record_id"].to_numpy()[okr])
                    line.append(f"{k} {np.nanmean(a):.5f}" if len(a) else f"{k} n/a")
                print(f"  regret@8 ({len(keys)} parents): " + "  ".join(line))
            print()
    return 0


def cmd_tau(args) -> int:
    """The ``policy_prior_temperature`` v2 needs to tilt the pool as v1 did.

    ``construct._policy_pick`` samples candidate ``i`` with weight
    ``exp(score_i / tau)``.  v1's registered ``tau = 0.25`` was chosen against a
    PROBABILITY spread; v2's output is a normalized clipped expected improvement
    on a much tighter scale, so the same tau is a much weaker tilt.  This solves

        exp((p90 - p10) / tau_v2) = exp((p90_v1 - p10_v1) / 0.25)

    on the gate fold's own score distributions, i.e. "give v2 the same
    sampling-odds ratio between a good and a poor candidate that v1 had".
    """
    _, _, gate = _load(args)
    print("## policy_prior_temperature for v2 (gate-fold score spreads)")
    print()
    for head in ("fr", "flat"):
        v2 = gate[f"p_v2_{head}"].to_numpy(float)
        v1 = gate[f"p_v1_{head}"].to_numpy(float)
        s2 = float(np.percentile(v2, 90) - np.percentile(v2, 10))
        s1 = float(np.percentile(v1, 90) - np.percentile(v1, 10))
        tau = args.tau_v1 * s2 / s1 if s1 > 0 else float("nan")
        print(f"  {head:5s} v1 p90-p10 = {s1:.4f} at tau {args.tau_v1:.2f}"
              f"   ->  v2 p90-p10 = {s2:.4f}  requires tau = {tau:.4f}"
              f"   (odds ratio {np.exp(s1 / args.tau_v1):.1f}x)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default="data/policy/steps.parquet")
    ap.add_argument("--probs", default="data/models/policy_v2/probs.npz")
    ap.add_argument("--v1-baseline", default="data/design/policy_v2_v1_baseline.csv")
    ap.add_argument("--base-seed", type=int, default=20260817)
    ap.add_argument("--tau-v1", type=float, default=0.25)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("strata").set_defaults(func=cmd_strata)
    sub.add_parser("slices").set_defaults(func=cmd_slices)
    sub.add_parser("tau").set_defaults(func=cmd_tau)
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
