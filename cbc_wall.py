"""Which constraint actually closes the high-enrichment region — and what opens it.

The task directive assumed the blocker at e >= 5.6 is the power-peaking factor
F_r, and asked for F_r to be relaxed there.  This script tests that assumption
against the labels that already exist (11 244 converged MASTER cores at
e_core >= 5.6) and finds it is wrong: the binding constraint is the soluble
boron concentration ``CBC_max <= 1600 ppm``.  Relaxing F_r alone opens ONE core
out of 11 244; relaxing CBC alone opens none; only a joint relaxation moves.

It then measures the two levers that DO move boron — feed and the number of
Gd-bearing pins — and lists the pairs that should be run because of it.

    python cbc_wall.py

Reads data/store only; writes CSVs under data/reports/mesh_v3_20260817/.
Runs no MASTER and touches no remote box.
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
OUT = BASE / "data" / "reports" / "mesh_v3_20260817"

HI_E = 5.6                      # "high enrichment" — above the ga80 ceiling
FR_LIMIT, FQ_LIMIT, CBC_LIMIT, AO_LIMIT = 1.55, 2.41, 1600.0, 0.30
MESH_FEEDS = (109, 113, 117, 121, 125, 129)
MAX_PAIR_SPREAD = 0.25
MIN_NGD = 22                    # the CBC-suppressing regime
_PARAMA_CANON = r"^P\d{4}Z\d"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg="") -> None:
        print(msg, flush=True)
        log_lines.append(str(msg))

    from lpopt.data.fuel_types import FuelLibrary

    fuel = FuelLibrary.from_parquet(BASE / "data/store/fuel_types.parquet")
    f = pd.read_parquet(BASE / "data/store/fuel_types.parquet")
    s = pd.read_parquet(BASE / "data/store/records.parquet")
    s = s[(s.valid == True) & (s.converged == True)]           # noqa: E712
    hi = s[s.e_core >= HI_E]
    log(f"high-enrichment labels: {len(hi)} converged cores at e_core >= {HI_E}")

    # ---------------------------------------------------- 1. which wall? --- #
    base = (hi.f_q <= FQ_LIMIT) & (hi.ao_abs.abs() <= AO_LIMIT)
    scen = [("current (F_r<=1.55, CBC<=1600)", 1.55, 1600.0),
            ("F_r ONLY -> 1.80", 1.80, 1600.0),
            ("F_r ONLY -> 2.20", 2.20, 1600.0),
            ("CBC ONLY -> 2000", 1.55, 2000.0),
            ("CBC ONLY -> 2600", 1.55, 2600.0),
            ("BOTH (1.80, 2000)", 1.80, 2000.0),
            ("BOTH (1.80, 2600)", 1.80, 2600.0)]
    rows = [dict(scenario=n, fr_cap=a, cbc_cap=b,
                 n_open=int((base & (hi.f_r <= a) & (hi.cbc_max <= b)).sum()))
            for n, a, b in scen]
    S = pd.DataFrame(rows)
    S["n_total"] = len(hi)
    S.to_csv(OUT / "cbc_wall_sensitivity.csv", index=False, encoding="utf-8")
    log("\n--- one-constraint-at-a-time sensitivity (the directive's premise) ---")
    log(S.to_string(index=False))
    log("\n=> relaxing F_r ALONE, as the directive proposed, opens "
        f"{int(S[S.scenario == 'F_r ONLY -> 1.80'].n_open.iloc[0])} of {len(hi)} cores. "
        "The wall is CBC.")

    v3 = hi[hi.feed.isin(MESH_FEEDS)]
    log(f"\nwithin the v3 feed lattice {MESH_FEEDS}: {len(v3)} cores, "
        f"min CBC_max = {v3.cbc_max.min():.2f} ppm "
        f"(limit {CBC_LIMIT:.0f}) -> "
        f"{int((v3.cbc_max <= CBC_LIMIT).sum())} cores under the limit")

    # ----------------------------------------------- 2. the boron levers --- #
    fp = f[f.library_id == "paramA"].set_index("type_id")

    def ngd(pair: str) -> float:
        a, b = pair.split("_", 1)
        try:
            return float((fp.n_gd[a] + fp.n_gd[b]) / 2)
        except Exception:                                       # noqa: BLE001
            return np.nan

    hp = hi[hi.library_id == "paramA"]
    g = hp.groupby(["case_pair", "feed"]).agg(
        n=("cbc_max", "size"), cbc_min=("cbc_max", "min"),
        e_core=("e_core", "mean"), fr_min=("f_r", "min")).reset_index()
    g["n_gd"] = [ngd(p) for p in g.case_pair]
    g["gd_wt"] = [float((fp.gd_wt[p.split("_", 1)[0]]
                         + fp.gd_wt[p.split("_", 1)[1]]) / 2) for p in g.case_pair]
    g = g[g.n_gd.notna()]
    g.round(4).to_csv(OUT / "cbc_by_pair_feed.csv", index=False, encoding="utf-8")

    X = np.c_[np.ones(len(g)), g.n_gd, g.e_core, g.feed]
    coef, *_ = np.linalg.lstsq(X, g.cbc_min.to_numpy(), rcond=None)
    resid = g.cbc_min.to_numpy() - X @ coef
    r2 = 1 - resid.var() / g.cbc_min.var()
    log("\n--- boron regression on the pair x feed minima ---")
    log(f"  CBC_min ~ {coef[0]:+.0f} {coef[1]:+.1f}*n_gd {coef[2]:+.0f}*e_core "
        f"{coef[3]:+.1f}*feed")
    log(f"  n = {len(g)} cells, R^2 = {r2:.3f}, rms residual = "
        f"{np.sqrt((resid ** 2).mean()):.0f} ppm")
    log(f"  correlations: n_gd {g.n_gd.corr(g.cbc_min):+.3f} | "
        f"gd_wt {g.gd_wt.corr(g.cbc_min):+.3f} (Gd CONCENTRATION is irrelevant) | "
        f"e_core {g.e_core.corr(g.cbc_min):+.3f}")

    # ------------------------------------- 3. the pairs worth running ------ #
    gp = f[(f.library_id == "paramA") & (~f.feature_poor.astype(bool))
           & (f.type_id.str.match(_PARAMA_CANON))]
    enr = dict(zip(gp.type_id, gp.u_avg_enrichment))
    run = set(s[s.library_id == "paramA"].case_pair.unique())
    cand = []
    for a, b in itertools.combinations(sorted(gp.type_id), 2):
        if abs(enr[a] - enr[b]) > MAX_PAIR_SPREAD:
            continue
        n = float((fp.n_gd[a] + fp.n_gd[b]) / 2)
        if n < MIN_NGD:
            continue
        try:
            ec = float(fuel.pair_e_core(a, b, 0.5, "paramA"))
        except Exception:                                       # noqa: BLE001
            continue
        if not np.isfinite(ec) or ec < HI_E:
            continue
        row = dict(pair=f"{a}_{b}", e_core=ec, n_gd=n,
                   gd_wt=float((fp.gd_wt[a] + fp.gd_wt[b]) / 2),
                   spread=abs(enr[a] - enr[b]), already_run=f"{a}_{b}" in run)
        for fd in (101, 105, 109, 117, 125):
            row[f"cbc_pred_f{fd}"] = float(np.array([1, n, ec, fd]) @ coef)
        cand.append(row)
    C = pd.DataFrame(cand).sort_values(["n_gd", "e_core"], ascending=[False, True])
    C.round(4).to_csv(OUT / "high_gd_candidates.csv", index=False, encoding="utf-8")
    log(f"\n--- paramA pairs with e_core >= {HI_E} and n_gd >= {MIN_NGD} ---")
    log(f"{len(C)} pairs exist, {int((~C.already_run).sum())} NEVER RUN "
        f"({int((C.n_gd >= 24).sum())} reach n_gd = 24, of which "
        f"{int(((C.n_gd >= 24) & (~C.already_run)).sum())} are unrun)")
    log(C[["pair", "e_core", "n_gd", "gd_wt", "spread", "already_run",
           "cbc_pred_f101", "cbc_pred_f109"]].round(3).to_string(index=False))

    best = C[(~C.already_run) & (C.cbc_pred_f109 <= CBC_LIMIT) & (C.e_core > 6.0)]
    if len(best):
        log("\n=> unrun pairs above e 6.0 predicted to pass CBC INSIDE the v3 "
            "feed lattice (f109):")
        log(best[["pair", "e_core", "n_gd", "cbc_pred_f109"]].round(3)
            .to_string(index=False))
        log("   residual rms is 101 ppm, so these are MARGINAL predictions — "
            "which is exactly what a small MASTER probe is for.")

    (OUT / "cbc_wall.log").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nwrote {OUT/'cbc_wall_sensitivity.csv'}, {OUT/'cbc_by_pair_feed.csv'}, "
        f"{OUT/'high_gd_candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
