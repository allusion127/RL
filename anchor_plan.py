"""Phase C, FIRST CUT — rank the mesh-v3 cells by the PREREG-1 §3 score.

    score = 0.4*GAP + 0.4*DIS + 0.2*REACH

SUPERSEDED FOR EXECUTION.  This script is kept because its output is the audit
trail for a decision that was later overturned: it ranks cells by how thin their
support is and how far the surrogate strays from the LRM, and it does that
INSIDE the v3 feed lattice, because PREREG-1 took the directive's premise (that
F_r is the binding constraint at high enrichment) at face value.

``cbc_wall.py`` then showed that premise is wrong — the whole lattice is closed
by soluble boron, not by F_r — so the anchor set was re-registered against the
boron wall in ``PREREG2_anchor_redirect_20260817.md`` and the deck that is
actually meant to run is ``anchors_meshv3_198.inp`` at the repo root.

**This script therefore no longer writes decks.**  It emits only
``anchor_plan.csv`` so the superseded ranking stays auditable; leaving a deck
generator here that emits the pre-redirect plan would be a loaded gun.

    python anchor_plan.py            # writes anchor_plan.csv, nothing else

NOTHING here contacts the fleet or runs MASTER.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
OUT = BASE / "data" / "reports" / "mesh_v3_20260817"

# ---- PREREG §3a budget (hard caps) ---------------------------------------- #
BUDGET_TOTAL = 250
BUDGET_PROBE = 160
BUDGET_CAMPAIGN = 80
PROBE_MIN, PROBE_MAX = 8, 16
CAMPAIGN_N, CAMPAIGN_MAX_CELLS = 40, 2
# ---- PREREG §3b score ------------------------------------------------------ #
W_GAP, W_DIS, W_REACH = 0.4, 0.4, 0.2
MAX_PER_E_LEVEL = 2
HIGH_E_MIN_SHARE = 0.60
HIGH_E = 5.8
REACH_W = {"실현 확인": 1.0, "미발견": 0.8, "격자-우회(fallback)": 0.6,
           "실현 불가(격자)": 0.0}


def gap_score(n: float) -> float:
    n = 0 if not np.isfinite(n) else int(n)
    return 1.0 if n == 0 else 0.6 if n < 25 else 0.3 if n < 100 else 0.0


def truth_floor(store: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Per cell, the LOWEST F_r any real MASTER core has ever reached on the
    anchor pair at that feed, and the cycle length it came with.

    REPORTED, NOT SCORED — the PREREG froze the score before these numbers
    existed and adding a term now would be fitting the rule to the answer.  It
    is here because it is the honest denominator for every tier claim: the
    surrogate's ``min_pred_f_r`` is a prediction, this is physics.

    Read it as an UPPER BOUND on the achievable floor, not the floor.  None of
    these rows came from a campaign that was minimising F_r at this cell — the
    one cell that did get such a campaign (T6_T4/f121) fell from its incidental
    minimum to 1.4749.  The gap between the two is exactly what Phase C buys."""

    g = store.groupby(["library_id", "case_pair", "feed"])
    out = []
    for r in d.itertuples():
        k = (r.anchor_library_id, r.anchor_pair, int(r.feed))
        if k not in g.groups:
            out.append((np.nan, np.nan, 0))
            continue
        q = store.loc[g.groups[k]]
        i = q.f_r.idxmin()
        out.append((float(q.f_r.min()), float(q.cyclen[i]), int(len(q))))
    return pd.DataFrame(out, columns=["fr_true_min", "cyclen_at_fr_min",
                                      "n_true"], index=d.index)


def build(mesh: pd.DataFrame, lrm: pd.DataFrame, store: pd.DataFrame,
          log=print) -> pd.DataFrame:
    d = lrm.merge(mesh[["cell", "max_pred_cyclen_any", "min_pred_f_r",
                        "n_feasible_tier1", "n_feasible_tier2",
                        "n_feasible_tier3", "tier_reached", "cyclen_tier3"]],
                  on="cell", how="left")
    d["surrogate_ceil"] = d.max_pred_cyclen_any
    d["d_surr_minus_lrm"] = d.surrogate_ceil - d.lrm_ceil_efpd
    d = pd.concat([d, truth_floor(store, d)], axis=1)
    d["fr_pred_minus_true"] = d.min_pred_f_r - d.fr_true_min

    # DIS — standardised WITHIN a feed column, because the two models' offset
    # drifts along the feed axis (the LRM's alpha is extrapolated at 125/129)
    # and an un-normalised difference would just rank the extrapolated columns.
    z = d.groupby("feed").d_surr_minus_lrm.transform(
        lambda s: (s - s.mean()).abs() / (s.std(ddof=0) or 1.0))
    d["DIS"] = z.clip(0, 1).fillna(0.0)
    d["GAP"] = [gap_score(v) for v in d.anchor_n_store_pair_feed]
    d["REACH"] = d.anchor_realizability.map(REACH_W).fillna(0.0)
    d["score"] = W_GAP * d.GAP + W_DIS * d.DIS + W_REACH * d.REACH
    d.loc[d.anchor_realizability == "실현 불가(격자)", "score"] = np.nan
    return d


def allocate(d: pd.DataFrame, log=print) -> pd.DataFrame:
    """Greedy take by score under the two PREREG §3b placement constraints."""

    d = d.copy()
    d["n_chains"] = 0
    d["role"] = ""
    pool = d[d.score.notna()].sort_values("score", ascending=False)

    # 1. the mini opening campaigns: the highest-scoring HIGH-e cells that are
    #    actually reachable.  These are the only cells that get enough calls to
    #    open a cell rather than merely measure it.
    camp = pool[(pool.e_target >= HIGH_E)
                & (pool.REACH > 0)].head(CAMPAIGN_MAX_CELLS)
    per_level: dict[float, int] = {}
    spent = 0
    for c in camp.itertuples():
        d.loc[d.cell == c.cell, ["n_chains", "role"]] = [CAMPAIGN_N, "campaign"]
        per_level[c.e_target] = per_level.get(c.e_target, 0) + 1
        spent += CAMPAIGN_N
    log(f"mini campaigns: {list(camp.cell)}  ({spent} chains)")

    # 2. probe chains, score-ordered, honouring both constraints
    probe_spent = 0
    hi_spent = spent          # campaigns are high-e by construction
    for c in pool.itertuples():
        if d.loc[d.cell == c.cell, "n_chains"].iloc[0] > 0:
            continue
        if probe_spent >= BUDGET_PROBE or spent >= BUDGET_TOTAL:
            break
        if per_level.get(c.e_target, 0) >= MAX_PER_E_LEVEL:
            continue
        n = int(round(PROBE_MIN + (PROBE_MAX - PROBE_MIN) * min(c.score, 1.0)))
        n = min(n, BUDGET_PROBE - probe_spent, BUDGET_TOTAL - spent)
        if n < PROBE_MIN:
            break
        d.loc[d.cell == c.cell, ["n_chains", "role"]] = [n, "probe"]
        per_level[c.e_target] = per_level.get(c.e_target, 0) + 1
        probe_spent += n
        spent += n
        if c.e_target >= HIGH_E:
            hi_spent += n
    log(f"probe chains: {probe_spent} over "
        f"{int((d.role == 'probe').sum())} cells")
    log(f"TOTAL {spent} / {BUDGET_TOTAL} chains; high-e (e>={HIGH_E}) share "
        f"{hi_spent/max(spent,1):.0%} (floor {HIGH_E_MIN_SHARE:.0%}) -> "
        f"{'OK' if hi_spent/max(spent,1) >= HIGH_E_MIN_SHARE else 'VIOLATION'}")
    return d


def main() -> int:
    logf = OUT / "anchor_plan.log"
    logf.write_text("", encoding="utf-8")

    def log(msg="") -> None:
        print(msg, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(str(msg) + "\n")

    lrm = pd.read_csv(OUT / "lrm_backbone.csv")
    mesh = pd.read_csv(OUT / "mesh_nodes.csv")
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    store = store[(store.valid == True) & (store.converged == True)]   # noqa: E712
    d = build(mesh, lrm, store, log)
    d = allocate(d, log)
    cols = ["cell", "e_target", "feed", "anchor_library_id", "anchor_pair",
            "anchor_e_core", "anchor_realizability", "anchor_n_store_pair_feed",
            "lrm_ceil_efpd", "surrogate_ceil", "d_surr_minus_lrm",
            "min_pred_f_r", "fr_true_min", "n_true", "fr_pred_minus_true",
            "cyclen_at_fr_min", "tier_reached", "n_feasible_tier1",
            "n_feasible_tier2", "n_feasible_tier3",
            "GAP", "DIS", "REACH", "score", "role", "n_chains"]
    d[cols].round(6).sort_values("score", ascending=False).to_csv(
        OUT / "anchor_plan.csv", index=False, encoding="utf-8")
    log(f"\nwrote {OUT/'anchor_plan.csv'}")
    log("\n--- the anchor set ---")
    log(d[d.n_chains > 0][
        ["cell", "role", "n_chains", "anchor_pair", "anchor_realizability",
         "anchor_n_store_pair_feed", "GAP", "DIS", "REACH", "score",
         "lrm_ceil_efpd", "surrogate_ceil", "min_pred_f_r", "fr_true_min"]]
        .sort_values("score", ascending=False).round(3).to_string(index=False))
    log("\n--- MASTER 실측 F_r 바닥 (보고용, 점수에 미반영) ---")
    t = d[d.n_true > 0]
    for lo, hi, nm in ((0, 5.55, "저농축 e<5.6"), (5.55, 9, "고농축 e>=5.6")):
        q = t[(t.e_target > lo) & (t.e_target < hi)]
        if not len(q):
            continue
        log(f"  {nm}: {len(q)}셀, 실측 최소 F_r {q.fr_true_min.min():.3f}–"
            f"{q.fr_true_min.max():.3f}; "
            + ", ".join(f"Tier-{i+1}(<={c}) {int((q.fr_true_min<=c).sum())}셀"
                        for i, c in enumerate((1.55, 1.65, 1.80))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
