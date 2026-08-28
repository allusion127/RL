"""Deliverable 3: composition-level frontier table + unexplored-cell ranking."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL")
SCR = Path(__file__).parent
sys.path.insert(0, str(SCR))
from dbx_parse import parse_all  # noqa: E402

N_CORE, P_TH_MW, M_HM_TU = 241, 3983.0, 102.031
SPR = P_TH_MW / M_HM_TU / 1000.0
FR_LIMIT = 1.55

pd.set_option("display.width", 340)
pd.set_option("display.max_columns", 90)
pd.set_option("display.max_rows", 250)

df = pd.read_excel(ROOT / "data/reports/scoping_mesh_20260815/feasible_database.xlsx",
                   sheet_name="cores")
recs, types = parse_all()
df["split"] = df.n_type1 / (df.n_type1 + df.n_type2)
df["B_c"] = df.EFPD * SPR
df["B_d"] = df.B_c * N_CORE / df.feed
df["bu_k1_mix"] = [df.split.iloc[i] * types[r.type1]["bu_k1"]
                   + (1 - df.split.iloc[i]) * types[r.type2]["bu_k1"]
                   for i, r in enumerate(df.itertuples())]

EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.90]
LBL = ["50-55", "55-60", "60-65", "65-70", "70-75", "75-90"]
df["split_bucket"] = pd.cut(df.split, EDGES, labels=LBL, right=False, include_lowest=True)

# ----------------------------------------------------------- store coverage -
store = pd.read_parquet(ROOT / "data/store/records.parquet")
st = store[store.library_id == "ga80"].copy()


def norm_pair(p):
    if not isinstance(p, str) or "_" not in p:
        return None
    a, b = p.split("_", 1)
    return "_".join(sorted([a, b]))


st["pkey"] = st.case_pair.map(norm_pair)
df["pkey"] = df.pair.map(norm_pair)
stv = st[st.valid.fillna(False)]
cov = (st.groupby(["pkey", "feed"]).size().rename("n_store_rows").to_frame()
       .join(stv.groupby(["pkey", "feed"]).size().rename("n_store_valid"), how="outer")
       .join(stv.groupby(["pkey", "feed"]).f_r.min().rename("store_f_r_min"), how="outer")
       .join(stv.groupby(["pkey", "feed"]).cyclen.max().rename("store_cyclen_max"), how="outer")
       .reset_index())
cov_pair = (st.groupby("pkey").size().rename("n_store_rows_pair_all_feeds").reset_index())

# ------------------------------------------------------------ frontier rows -
rows = []
for (pair, feed, bucket), g in df.groupby(["pair", "feed", "split_bucket"], observed=True):
    if len(g) == 0:
        continue
    b = g.loc[g.F_r.idxmin()]
    long_ = g.loc[g.EFPD.idxmax()]
    gp = g[g.pinmax_rodavg_GWd.notna()]          # pin-burnup known (equilibrium rows)
    gpf = gp if len(gp) else g.iloc[0:0]
    rows.append(dict(
        pair=pair, type1=g.type1.iloc[0], type2=g.type2.iloc[0],
        enrichment_segment=g.enrichment_segment.iloc[0], feed=int(feed),
        split_bucket=str(bucket), split_lo=EDGES[LBL.index(str(bucket))],
        split_mean=g.split.mean(), split_min=g.split.min(), split_max=g.split.max(),
        n_cores=len(g), n_feasible=int((g.F_r <= FR_LIMIT).sum()),
        n_distinct_splits=g.split.round(5).nunique(),
        realized_enrichment_mean=g.realized_enrichment.mean(),
        bu_k1_mix_mean=g.bu_k1_mix.mean(),
        # --- the frontier point: minimum F_r core -----------------------
        F_r_min=b.F_r, best_cid=b.cid, best_split=b.split,
        best_n_type1=int(b.n_type1), best_n_type2=int(b.n_type2),
        best_EFPD=b.EFPD, best_B_c=b.B_c, best_B_d=b.B_d,
        best_discharge_GWd=b.core_mean_discharge_GWd,
        best_CBC_max=b.CBC_max, best_cbc_margin=b.cbc_margin,
        best_F_q=b.F_q, best_AO_min=b.AO_min, best_AO_max=b.AO_max,
        best_pinmax_rodavg_GWd=b.pinmax_rodavg_GWd,
        best_pinmax_node_GWd=b.pinmax_node_GWd,
        best_pinmax_known=bool(pd.notna(b.pinmax_rodavg_GWd)),
        best_rodavg_ge68=bool(b.rodavg_ge68), best_rodavg_ge75=bool(b.rodavg_ge75),
        best_node_ge68=bool(b.node_ge68), best_node_ge75=bool(b.node_ge75),
        best_F_r_campaign=b.F_r_campaign, best_CBC_max_campaign=b.CBC_max_campaign,
        best_metrics_source=b.metrics_source, best_eq_ok=bool(b.eq_ok),
        best_ncyc=int(b.ncyc), best_feasible_at_metrics=bool(b.feasible_at_metrics),
        # --- cell aggregates --------------------------------------------
        F_r_p10=float(np.percentile(g.F_r, 10)), F_r_median=float(g.F_r.median()),
        EFPD_mean=g.EFPD.mean(), EFPD_max=g.EFPD.max(),
        B_d_mean=g.B_d.mean(),
        CBC_max_max=g.CBC_max.max(), cbc_margin_min=g.cbc_margin.min(),
        n_pinmax_known=len(gp),
        pinmax_rodavg_min=g.pinmax_rodavg_GWd.min(),
        pinmax_rodavg_mean=g.pinmax_rodavg_GWd.mean(),
        pinmax_node_min=g.pinmax_node_GWd.min(),
        pinmax_node_mean=g.pinmax_node_GWd.mean(),
        # fractions computed ONLY over rows whose pin burnup is known
        frac_rodavg_ge62=gpf.rodavg_ge62.mean() if len(gpf) else np.nan,
        frac_rodavg_ge68=gpf.rodavg_ge68.mean() if len(gpf) else np.nan,
        frac_rodavg_ge75=gpf.rodavg_ge75.mean() if len(gpf) else np.nan,
        frac_node_ge62=gpf.node_ge62.mean() if len(gpf) else np.nan,
        frac_node_ge68=gpf.node_ge68.mean() if len(gpf) else np.nan,
        frac_node_ge75=gpf.node_ge75.mean() if len(gpf) else np.nan,
        n_campaign_verified=int((g.metrics_source == "campaign").sum()),
        F_r_campaign_min=g.F_r_campaign.min(),
        # --- longest-cycle core in the cell (2nd frontier corner) --------
        EFPD_max_F_r=long_.F_r, EFPD_max_B_d=long_.B_d, EFPD_max_split=long_.split,
        pkey=norm_pair(pair),
    ))
FR = pd.DataFrame(rows)
FR = FR.merge(cov, on=["pkey", "feed"], how="left").merge(cov_pair, on="pkey", how="left")
for c in ("n_store_rows", "n_store_valid", "n_store_rows_pair_all_feeds"):
    FR[c] = FR[c].fillna(0).astype(int)
FR["store_gap_f_r"] = FR.store_f_r_min - FR.F_r_min      # >0 : DB flatter than we ever got
FR["unexplored"] = FR.n_store_valid < 25
FR["novel_type"] = FR.pair.isin(["E5_E6", "J7_J8"])

FR = FR.sort_values(["enrichment_segment", "feed", "pair", "split_lo"])
dest = ROOT / "data/reports/dbx_frontier_table.csv"
FR.round(6).to_csv(dest, index=False, encoding="utf-8")
print(f"wrote {dest}  ({len(FR)} rows x {FR.shape[1]} cols)")
print("cells:", FR.groupby("split_bucket").size().to_dict())
print("pairs:", FR.pair.nunique(), " (pair,feed) combos:", FR.groupby(['pair','feed']).ngroups)

# --------------------------------------------------------------- ranking ----
# Transparent rule (documented in the md note):
#   filter  n_cores >= 5  AND  F_r_min <= 1.535  AND  n_store_valid <= 25
#   sort    F_r_min ascending, then best_EFPD descending
c = FR[(FR.F_r_min <= 1.535) & (FR.n_cores >= 5) & (FR.n_store_valid <= 25)].copy()
c = c.sort_values(["F_r_min", "best_EFPD"], ascending=[True, False])
print(f"\n{len(c)} cells pass the screen; top 14:")
print(c.head(14)[["pair", "feed", "split_bucket", "n_cores", "F_r_min", "best_EFPD",
                  "best_B_d", "best_CBC_max", "best_cbc_margin",
                  "best_pinmax_rodavg_GWd", "best_pinmax_node_GWd", "best_pinmax_known",
                  "frac_node_ge75", "n_pinmax_known", "n_store_valid", "store_f_r_min",
                  "novel_type"]].round(3).to_string(index=False))
c.round(6).to_csv(SCR / "ranking.csv", index=False)

print("\nstore coverage of the 20 DB pairs (valid ga80 records, de-duplicated):")
pf = FR.drop_duplicates(["pair", "feed"])
pc = (FR.groupby("pair").agg(db_cores=("n_cores", "sum"), db_F_r_min=("F_r_min", "min"),
                             db_feeds=("feed", "nunique"))
      .join(pf.groupby("pair").n_store_valid.sum().rename("store_valid"))
      .join(pf.groupby("pair").n_store_rows_pair_all_feeds.first().rename("store_all_feeds"))
      .sort_values("store_valid"))
print(pc.to_string())
print("\npin-burnup coverage: %d of %d frontier cells have >=1 core with known pin burnup"
      % ((FR.n_pinmax_known > 0).sum(), len(FR)))
print("campaign-only cells (no pin burnup at all): %d" % (FR.n_pinmax_known == 0).sum())
FR.to_pickle(SCR / "frontier.pkl")
