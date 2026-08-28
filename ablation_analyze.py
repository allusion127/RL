"""Registered analyses for the 1-move ablation wave.

Pre-registration: ``data/reports/ablation_wave_prereg_20260815.md`` §6.

SEPARATE FROM ``ablation_wave.py`` ON PURPOSE.  That file's sha256 is pinned in
the pre-registration §8 and is the artefact that was shipped to box 199; editing
it after launch would void the registration.  Everything post-hoc lives here.

Subcommands
-----------
``corpus``   append the wave's lineage edges to ``data/policy/steps.parquet``
             with ``lineage_source='ablation_paramA'``.  The rows are produced by
             ``mine_policy_corpus.build_steps`` ITSELF (not re-derived here), so
             all 77 columns are schema-identical to the mined corpus.
``analyze``  §6a leakage arbitration + §6c policy-v1 prospective test -> the
             results report tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

CAMPAIGN = "ablation_1move_T6T4"
LINEAGE_SOURCE = "ablation_paramA"
CELL = "T6_T4/f121/paramA"
SEED = 20260815

#: Overridable so the whole pipeline can be exercised against a SCRATCH copy of
#: the store + corpus before it is ever pointed at the canonical ones.
STORE = BASE / "data/store/records.parquet"
STEPS = BASE / "data/policy/steps.parquet"


def _set_paths(args) -> None:
    global STORE, STEPS
    if getattr(args, "store", None):
        STORE = Path(args.store)
    if getattr(args, "steps", None):
        STEPS = Path(args.steps)

#: The two reactivity-conserving instruments, in registered priority order.
INSTRUMENTS = ("fresh_relocate", "batch_swap")
#: Reported but NOT admissible for the leakage verdict (prereg §3b).
NOT_AN_INSTRUMENT = "batch_flip"


# --------------------------------------------------------------------------- #
# corpus append
# --------------------------------------------------------------------------- #
def build_wave_steps(campaign: str | None = None,
                     lineage: str | None = None) -> pd.DataFrame:
    """Mine ONLY one campaign's edges, with mine_policy_corpus's own builder.

    ``lineage`` overrides ``lineage_source``.  Passing ``lineage=None`` keeps
    whatever ``build_steps`` inferred (``lpopt_genome`` for store-native edges),
    which is the correct tag for an ordinary campaign back-fill.
    """
    import mine_policy_corpus as M

    campaign = campaign or CAMPAIGN
    store = pd.read_parquet(STORE)
    wave = store[store["campaign"] == campaign]
    if wave.empty:
        raise SystemExit(f"no {campaign} rows in the store - run merge-store first")
    need = set(wave["record_id"]) | set(wave["parent_record_id"].dropna())
    subset = store[store["record_id"].isin(need)].copy()

    enrichment = M.load_enrichment(BASE / "data/store/fuel_types.parquet")
    bands = M.cyclen_bands(BASE)
    steps = M.build_steps(subset, bands, enrichment, sa=None)
    steps = steps[steps["campaign"] == campaign].copy()
    # The ONLY column this module sets: the provenance tag the brief registered.
    # ``build_steps`` labels store-native edges ``lpopt_genome`` because it reads
    # the store's own parent_record_id; these edges are that, but they are also
    # the interventional set and must be selectable as such.
    if lineage is not None:
        steps["lineage_source"] = lineage
    return steps


def cmd_corpus(args) -> int:
    new = build_wave_steps(args.campaign,
                           None if args.lineage == "native" else args.lineage)
    print(f"[corpus] mined {len(new)} wave edges")
    print(f"[corpus] single_move: {int(new['single_move'].sum())}/{len(new)}")
    print(new.groupby(["move_class", "fresh_radial_dir"]).size().to_string())

    existing = pd.read_parquet(STEPS)
    print(f"[corpus] existing steps.parquet: {len(existing)} rows, "
          f"{len(existing.columns)} cols")
    missing = set(existing.columns) - set(new.columns)
    extra = set(new.columns) - set(existing.columns)
    if missing or extra:
        raise SystemExit(f"schema drift - missing {sorted(missing)} "
                         f"extra {sorted(extra)}")
    new = new[existing.columns]
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["parent_record_id", "child_record_id"], keep="last")
    print(f"[corpus] combined {len(combined)} rows "
          f"(+{len(combined) - len(existing)})")
    if args.dry_run:
        print("[corpus] DRY RUN - steps.parquet not written")
        return 0
    backup = STEPS.with_suffix(f".parquet.bak_pre_{args.campaign}")
    if not backup.exists():
        backup.write_bytes(STEPS.read_bytes())
        print(f"[corpus] backup -> {backup.name}")
    combined.to_parquet(STEPS, index=False)
    print(f"[corpus] wrote {STEPS}")
    return 0


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _improving(frame: pd.DataFrame) -> dict[str, float]:
    """Improving fractions, defined exactly as mine_policy_corpus.build_steps."""
    out: dict[str, float] = {}
    for label, fom, lower_better in (
            ("F_r v", "f_r", True), ("flat v", "node_peak", True),
            ("CBC v", "cbc_max", True), ("cyclen ^", "cyclen", False)):
        c, p = frame[f"child_{fom}"], frame[f"parent_{fom}"]
        ok = frame["both_converged"].fillna(False).astype(bool) & c.notna() & p.notna()
        better = (c < p) if lower_better else (c > p)
        n = int(ok.sum())
        out[f"n({label})"] = n
        out[label] = float(better[ok].mean()) if n else np.nan
        out[f"mean d_{fom}"] = float((c - p)[ok].mean()) if n else np.nan
    return out


def _paired_by_parent(frame: pd.DataFrame, cls: str, col: str):
    """Per-parent (mean outward - mean inward) of ``col`` for move class ``cls``."""
    sub = frame[(frame["move_class"] == cls)
                & frame["both_converged"].fillna(False).astype(bool)
                & frame[col].notna()]
    rows = []
    for pid, blk in sub.groupby("parent_record_id"):
        o = blk[blk["fresh_radial_dir"] == "outward"][col]
        i = blk[blk["fresh_radial_dir"] == "inward"][col]
        if len(o) and len(i):
            rows.append({"parent_record_id": pid, "n_out": len(o), "n_in": len(i),
                         "mean_out": o.mean(), "mean_in": i.mean(),
                         "diff": o.mean() - i.mean()})
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, reps: int = 20000, seed: int = SEED):
    """Percentile CI of the mean, resampling the PARENTS (the analysis unit)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _sign_test(diffs: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact binomial sign test (zeros dropped)."""
    from math import comb

    d = np.asarray(diffs, float)
    d = d[np.isfinite(d) & (d != 0)]
    n, pos = len(d), int((d > 0).sum())
    if n == 0:
        return 0, 0, np.nan
    tail = sum(comb(n, k) for k in range(0, min(pos, n - pos) + 1)) / 2 ** n
    return pos, n, float(min(1.0, 2 * tail))


def _fe_slope(frame: pd.DataFrame, cls: str, ycol: str,
              xcol: str = "d_fresh_enr_r_center"):
    """Parent-fixed-effects slope of ``ycol`` on ``xcol`` (within-parent demeaned).

    Parent difficulty is differenced out, so the slope is the dose-response of
    the move itself rather than a between-parent comparison.
    """
    sub = frame[(frame["move_class"] == cls)
                & frame["both_converged"].fillna(False).astype(bool)
                & frame[ycol].notna() & frame[xcol].notna()].copy()
    if len(sub) < 4:
        return np.nan, (np.nan, np.nan), 0
    sub["_y"] = sub[ycol] - sub.groupby("parent_record_id")[ycol].transform("mean")
    sub["_x"] = sub[xcol] - sub.groupby("parent_record_id")[xcol].transform("mean")
    if sub["_x"].std() == 0:
        return np.nan, (np.nan, np.nan), len(sub)
    slope = float(np.polyfit(sub["_x"], sub["_y"], 1)[0])
    # Cluster bootstrap over parents.
    rng = np.random.default_rng(SEED)
    pids = sub["parent_record_id"].unique()
    draws = []
    for _ in range(2000):
        pick = rng.choice(pids, size=len(pids), replace=True)
        blk = pd.concat([sub[sub["parent_record_id"] == p] for p in pick])
        if blk["_x"].std() > 0:
            draws.append(np.polyfit(blk["_x"], blk["_y"], 1)[0])
    ci = ((float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
          if len(draws) > 100 else (np.nan, np.nan))
    return slope, ci, len(sub)


# --------------------------------------------------------------------------- #
# policy-v1 prospective test
# --------------------------------------------------------------------------- #
def _auc(score: np.ndarray, truth: np.ndarray) -> float:
    """ROC AUC with mean ranks for ties."""
    score, truth = np.asarray(score, float), np.asarray(truth, bool)
    if truth.all() or not truth.any():
        return np.nan
    r = pd.Series(score).rank(method="average").to_numpy()
    n1, n0 = int(truth.sum()), int((~truth).sum())
    return float((r[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _parent_blocked_auc(frame: pd.DataFrame, score: str, truth: str) -> tuple[float, int]:
    """Fraction of same-parent (improving, non-improving) pairs ordered right."""
    wins = pairs = 0.0
    for _, blk in frame.groupby("parent_record_id"):
        pos = blk[blk[truth]][score].to_numpy()
        neg = blk[~blk[truth]][score].to_numpy()
        if not len(pos) or not len(neg):
            continue
        d = pos[:, None] - neg[None, :]
        wins += float((d > 0).sum() + 0.5 * (d == 0).sum())
        pairs += d.size
    return (wins / pairs if pairs else np.nan), int(pairs)


def _precision_at_k(frame: pd.DataFrame, score: str, truth: str,
                    k: int = 32, reps: int = 2000, seed: int = SEED) -> float:
    """Top-k precision over the whole labelled set, random tiebreak, averaged."""
    rng = np.random.default_rng(seed)
    s = frame[score].to_numpy(float)
    y = frame[truth].to_numpy(bool)
    if len(s) < k:
        k = len(s)
    vals = []
    for _ in range(reps):
        order = np.lexsort((rng.random(len(s)), -s))
        vals.append(y[order[:k]].mean())
    return float(np.mean(vals))


def cmd_analyze(args) -> int:
    import mine_policy_corpus as M

    steps = build_wave_steps(args.campaign, args.lineage)
    conv = steps["both_converged"].fillna(False).astype(bool)
    print(f"[analyze] {len(steps)} wave edges, {int(conv.sum())} both-converged")

    out: list[str] = []

    def add(s: str = "") -> None:
        out.append(s)
        print(s)

    # ---- §6a per-stratum table ------------------------------------------- #
    add("## Per-stratum outcomes")
    add()
    rows = []
    for (cls, direction), blk in steps.groupby(["move_class", "fresh_radial_dir"]):
        rows.append({"move_class": cls, "fresh_radial_dir": direction,
                     "n": len(blk), "n_conv": int(blk["both_converged"]
                                                  .fillna(False).sum()),
                     **_improving(blk)})
    strata = pd.DataFrame(rows).sort_values(["move_class", "fresh_radial_dir"])
    add(strata.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    add()

    # ---- §6a registered paired test --------------------------------------- #
    add("## REGISTERED PRIMARY TEST - outward vs inward at fixed class, "
        "paired within parent")
    add()
    verdicts = {}
    for cls in (*INSTRUMENTS, NOT_AN_INSTRUMENT):
        add(f"### {cls}" + ("" if cls in INSTRUMENTS
                            else "  (NOT admissible for leakage - prereg §3b)"))
        for col, name in (("d_cyclen", "cyclen"), ("d_f_r", "F_r"),
                          ("d_node_peak", "node_peak"), ("d_cbc_max", "CBC")):
            pt = _paired_by_parent(steps, cls, col)
            if pt.empty:
                add(f"  {name:<10} no paired parents")
                continue
            diffs = pt["diff"].to_numpy()
            lo, hi = _bootstrap_ci(diffs)
            pos, n, p = _sign_test(diffs)
            excl = "" if (np.isnan(lo) or lo <= 0 <= hi) else "  **CI excludes 0**"
            add(f"  {name:<10} mean(out-in) {diffs.mean():+.4f}  "
                f"95% CI [{lo:+.4f}, {hi:+.4f}]  sign {pos}/{n} p={p:.3f}"
                f"{excl}")
            if col == "d_cyclen":
                verdicts[cls] = {"mean": float(diffs.mean()), "lo": lo, "hi": hi,
                                 "pos": pos, "n": n, "p": p}
        slope, ci, n = _fe_slope(steps, cls, "d_cyclen")
        add(f"  dose-response d_cyclen ~ d_fresh_enr_r_center (parent FE): "
            f"slope {slope:+.3f} EFPD per unit  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            f"  n={n}")
        add()

    # reference arm
    ref = steps[steps["move_class"] == "rewire_swap"]
    r = _improving(ref)
    add(f"### rewire_swap x neutral (reference arm, no radial direction)")
    add(f"  n={len(ref)}  mean d_cyclen {r['mean d_cyclen']:+.4f}  "
        f"cyclen^ {r['cyclen ^']:.3f}  F_r v {r['F_r v']:.3f}  "
        f"flat v {r['flat v']:.3f}")
    add()

    # ---- §6c prospective policy test -------------------------------------- #
    add("## REGISTERED PROSPECTIVE TEST - policy v1 on the era it failed")
    add()
    pred = pd.read_csv(args.pred)
    m = steps.merge(pred, left_on="child_record_id", right_on="record_id",
                    how="inner", suffixes=("", "_pred"))
    m = m[m["both_converged"].fillna(False).astype(bool)].copy()
    add(f"  labelled + scored children: {len(m)}")

    # class_freq baseline fitted on the PRE-EXISTING corpus only.
    prior = pd.read_parquet(STEPS)
    prior = prior[(prior["lineage_source"] != LINEAGE_SOURCE)
                  & (~prior["cross_cell"].fillna(False).astype(bool))]
    for head, truth_col, fom in (("fr", "improved_fr", "f_r"),
                                 ("flat", "improved_flat", "node_peak")):
        col = f"p_improve_{'fr' if head == 'fr' else 'flat'}"
        sub = m[m[truth_col].notna()].copy()
        sub["_y"] = sub[truth_col].astype(bool)
        rate = (prior[prior[truth_col].notna()]
                .groupby("move_class")[truth_col].mean())
        sub["_class_freq"] = sub["move_class"].map(rate).fillna(rate.mean())
        rng = np.random.default_rng(SEED)
        sub["_random"] = rng.random(len(sub))
        sub["_periph"] = sub["d_fresh_share_periph"]
        add(f"### head `{head}`  (n={len(sub)}, base rate "
            f"{sub['_y'].mean():.3f})")
        for label, scorer in (("policy_v1", col), ("random", "_random"),
                              ("class_freq", "_class_freq"),
                              ("periph", "_periph")):
            auc = _auc(sub[scorer], sub["_y"])
            pb, npairs = _parent_blocked_auc(sub, scorer, "_y")
            p32 = _precision_at_k(sub, scorer, "_y")
            add(f"  {label:<12} AUC {auc:.3f}   parent-blocked {pb:.3f} "
                f"({npairs} pairs)   p@32 {p32:.3f}")
        add()

    # ---- predicted vs measured, by stratum -------------------------------- #
    add("## Blind prediction vs measured, by stratum")
    add()
    t = (m.assign(imp_fr=m["improved_fr"].astype("float"),
                  imp_flat=m["improved_flat"].astype("float"))
           .groupby(["move_class", "fresh_radial_dir"])
           .agg(n=("child_record_id", "size"),
                pred_fr=("p_improve_fr", "mean"),
                meas_fr=("imp_fr", "mean"),
                pred_flat=("p_improve_flat", "mean"),
                meas_flat=("imp_flat", "mean")))
    add(t.to_string(float_format=lambda v: f"{v:.3f}"))
    add()

    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n[analyze] tables -> {args.out}")
    json.dump(verdicts, open(args.out + ".verdicts.json", "w"), indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpus")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--campaign", default=CAMPAIGN)
    p.add_argument("--lineage", default=LINEAGE_SOURCE,
                   help="lineage_source tag, or 'native' to keep build_steps' own")
    p.add_argument("--store"); p.add_argument("--steps")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("analyze")
    p.add_argument("--pred", default="data/design/ablation_wave_policy_v1_pred.csv")
    p.add_argument("--out", default="data/reports/ablation_wave_tables.txt")
    p.add_argument("--campaign", default=CAMPAIGN)
    p.add_argument("--lineage", default=LINEAGE_SOURCE)
    p.add_argument("--store"); p.add_argument("--steps")
    p.set_defaults(func=cmd_analyze)

    args = ap.parse_args(argv)
    _set_paths(args)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
