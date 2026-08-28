"""Deliverable 1: calibrated-LRM backbone fit + surrogate-ceiling comparison."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(r"c:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL")
SCR = Path(__file__).parent
sys.path.insert(0, str(SCR))
from dbx_parse import parse_all  # noqa: E402

N_CORE = 241
P_TH_MW = 3983.0
M_HM_TU = 102.031
SPR = P_TH_MW / M_HM_TU / 1000.0

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 90)
pd.set_option("display.max_rows", 200)

XL = ROOT / "data/reports/scoping_mesh_20260815/feasible_database.xlsx"
df = pd.read_excel(XL, sheet_name="cores")
mesh = pd.read_csv(ROOT / "data/reports/scoping_mesh_20260815/mesh_nodes.csv")

recs, types = parse_all()
BU = np.asarray(recs["E1_E2"]["bu"], float)
KINF = {t: np.asarray(recs[d["pair"]]["kinf"][t], float) for t, d in types.items()}

df["split"] = df.n_type1 / (df.n_type1 + df.n_type2)
w1 = df.split.to_numpy()
df["bu_k1_mix"] = [w1[i] * types[r.type1]["bu_k1"] + (1 - w1[i]) * types[r.type2]["bu_k1"]
                   for i, r in enumerate(df.itertuples())]
df["B_c"] = df.EFPD * SPR
df["n_batch"] = N_CORE / df.feed
df["B_d_massbal"] = df.B_c * N_CORE / df.feed
df["massbal_dev_pct"] = 100.0 * (df.B_d_massbal / df.core_mean_discharge_GWd - 1.0)

e = df.realized_enrichment.to_numpy()
y = df.EFPD.to_numpy()
F = df.feed.to_numpy(float)
bk = df.bu_k1_mix.to_numpy()
feeds = sorted(set(F))
FD = np.array([[1.0 * (x == q) for q in feeds] for x in F])

out = {}


def stat(name, pred):
    r = pred - y
    return dict(model=name, rms=float(np.sqrt((r ** 2).mean())), bias=float(r.mean()),
                p95=float(np.percentile(abs(r), 95)), max=float(abs(r).max()))


# ---------------------------------------------------------------- model A --
gA = 2.0 / ((N_CORE / F + 1.0) * SPR)
XA = np.c_[gA, gA * e]
cA = np.linalg.lstsq(XA, y, rcond=None)[0]
predA = XA @ cA

# ---------------------------------------------------------------- model B --
def fB(p):
    return (FD @ p[:6]) * (p[6] + p[7] * e) - y


rB = least_squares(fB, [1.0] * 6 + [-14.0, 122.0], xtol=1e-14, ftol=1e-14)
predB = rB.fun + y
alphaB = dict(zip([int(f) for f in feeds], rB.x[:6]))
aB, bB = rB.x[6], rB.x[7]

# ---------------------------------------------------------------- model C --
def fC(p):
    return (FD @ p[:6]) * (p[6] + p[7] * bk) - y


rC = least_squares(fC, [1.0] * 6 + [0.0, 15.0], xtol=1e-14, ftol=1e-14)
predC = rC.fun + y
alphaC = dict(zip([int(f) for f in feeds], rC.x[:6]))
aC, bC = rC.x[6], rC.x[7]

# ---------------------------------------------------------------- model D --
def fD(p):
    return (FD @ p[:6]) * (p[6] + p[7] * e + p[8] * bk) - y


rD = least_squares(fD, [1.0] * 6 + [-14.0, 122.0, 0.0], xtol=1e-14, ftol=1e-14)
predD = rD.fun + y

# ------------------------------------------------------------ floors -------
cellmean = df.groupby(["enrichment_segment", "feed"]).EFPD.transform("mean").to_numpy()
cellpair = df.groupby(["enrichment_segment", "feed", "pair"]).EFPD.transform("mean").to_numpy()

models = [("A_analytic_LRM", predA, 2), ("B_cal_enrichment", predB, 7),
          ("C_cal_buk1", predC, 7), ("D_cal_enr_plus_buk1", predD, 8),
          ("FLOOR_cell_mean", cellmean, 36), ("FLOOR_cell_pair_mean", cellpair, None)]
fitq = []
for nm, pr, k in models:
    s = stat(nm, pr)
    s["n_param"] = k
    fitq.append(s)
FITQ = pd.DataFrame(fitq)
print(FITQ.round(3).to_string(index=False))
print(f"\nA: B1 = {cA[0]:+.4f} {cA[1]:+.4f}*e   (n = 241/feed)")
print(f"B: EFPD = alpha_F * ({aB:+.4f} {bB:+.4f}*e),  alpha = "
      f"{ {k: round(v, 6) for k, v in alphaB.items()} }")
print(f"C: EFPD = alpha_F * ({aC:+.4f} {bC:+.4f}*bu_k1_mix), alpha = "
      f"{ {k: round(v, 6) for k, v in alphaC.items()} }")
print(f"D: coef {np.round(rD.x, 5)}")

# alpha vs analytic 2/(n+1)
nb = N_CORE / np.array(feeds)
alpha_lrm = 2.0 / ((nb + 1.0) * SPR)
ref = feeds.index(109)
tab_alpha = pd.DataFrame({
    "feed": [int(f) for f in feeds],
    "n_batch": nb,
    "alpha_B": rB.x[:6],
    "alpha_B_norm109": rB.x[:6] / rB.x[:6][ref],
    "lrm_norm109": alpha_lrm / alpha_lrm[ref],
})
tab_alpha["lrm_minus_obs_pct"] = 100 * (tab_alpha.lrm_norm109 / tab_alpha.alpha_B_norm109 - 1)
print()
print(tab_alpha.round(5).to_string(index=False))

# ------------------------------------------------------- per-cell table ----
df["rA"], df["rB"], df["rC"], df["rD"] = predA - y, predB - y, predC - y, predD - y
df["pA"], df["pB"], df["pC"] = predA, predB, predC


def rms(s):
    return float(np.sqrt((np.asarray(s, float) ** 2).mean()))


cells = df.groupby(["enrichment_segment", "feed"]).agg(
    n_cores=("EFPD", "size"),
    n_pairs=("pair", "nunique"),
    n_feasible_db=("feasible_at_metrics", "sum"),
    e_real_mean=("realized_enrichment", "mean"),
    e_real_sd=("realized_enrichment", "std"),
    bu_k1_mix_mean=("bu_k1_mix", "mean"),
    EFPD_db_mean=("EFPD", "mean"), EFPD_db_sd=("EFPD", "std"),
    EFPD_db_min=("EFPD", "min"), EFPD_db_max=("EFPD", "max"),
    B_c_db_mean=("B_c", "mean"),
    B_d_massbal_mean=("B_d_massbal", "mean"),
    B_d_reported_mean=("core_mean_discharge_GWd", "mean"),
    massbal_dev_pct_mean=("massbal_dev_pct", "mean"),
    massbal_dev_pct_sd=("massbal_dev_pct", "std"),
    F_r_db_min=("F_r", "min"),
    CBC_max_db_max=("CBC_max", "max"),
    lrmA_efpd=("pA", "mean"), lrmA_bias=("rA", "mean"), lrmA_rms=("rA", rms),
    lrmB_efpd=("pB", "mean"), lrmB_bias=("rB", "mean"), lrmB_rms=("rB", rms),
    lrmC_efpd=("pC", "mean"), lrmC_bias=("rC", "mean"), lrmC_rms=("rC", rms),
).reset_index()

# ---- ceiling backbone: same functional form fit to the per-cell MAX EFPD ---
idxmax = df.groupby(["enrichment_segment", "feed"]).EFPD.idxmax()
top = df.loc[idxmax, ["enrichment_segment", "feed", "realized_enrichment", "EFPD", "pair"]]
Ft = top.feed.to_numpy(float)
FDt = np.array([[1.0 * (x == q) for q in feeds] for x in Ft])
et, yt = top.realized_enrichment.to_numpy(), top.EFPD.to_numpy()


def fCeil(p):
    return (FDt @ p[:6]) * (p[6] + p[7] * et) - yt


rCeil = least_squares(fCeil, [1.0] * 6 + [-14.0, 122.0], xtol=1e-14, ftol=1e-14)
alphaCeil = dict(zip([int(f) for f in feeds], rCeil.x[:6]))
aCe, bCe = rCeil.x[6], rCeil.x[7]
print(f"\nCEILING backbone (fit to 36 per-cell max EFPD): rms={rms(rCeil.fun):.2f} EFPD, "
      f"a={aCe:.4f} b={bCe:.4f} alpha={ {k: round(v,6) for k,v in alphaCeil.items()} }")


def ceil_pred(e_, f_):
    return np.array([alphaCeil[int(ff)] * (aCe + bCe * ee) for ee, ff in zip(np.atleast_1d(e_), np.atleast_1d(f_))])


def mean_pred(e_, f_):
    return np.array([alphaB[int(ff)] * (aB + bB * ee) for ee, ff in zip(np.atleast_1d(e_), np.atleast_1d(f_))])


# ------------------------------------------------------------ union grid ---
mesh = mesh.rename(columns={"e_target": "enrichment_segment"})
mk = mesh[["cell", "enrichment_segment", "feed", "per_fa_tU", "M_HM_tU", "ceil_cyclen",
           "ceil_B_cycle", "ceil_B_d", "min_pred_f_r", "n_feasible", "in_distribution",
           "n_store_pair_feed", "library_id"]].copy()

U = pd.merge(cells, mk, on=["enrichment_segment", "feed"], how="outer")
U["cell"] = U.apply(lambda r: r["cell"] if isinstance(r["cell"], str)
                    else f"e{r.enrichment_segment:.1f}_f{int(r.feed)}", axis=1)
U["zone"] = np.where(U.n_cores.notna() & U.ceil_cyclen.notna(), "db_and_mesh",
                     np.where(U.n_cores.notna(), "db_only", "mesh_only"))
U["n_batch"] = N_CORE / U.feed
U["extrapolated"] = U.enrichment_segment > 5.5

# backbone evaluated on the *segment* enrichment for mesh-only cells (no DB e).
seg_to_e = cells.set_index(["enrichment_segment", "feed"]).e_real_mean
U["e_for_backbone"] = [
    seg_to_e.get((r.enrichment_segment, r.feed), np.nan) for r in U.itertuples()]
# mesh-only cells: use the segment label + the mean DB offset (realized - segment)
off = float((cells.e_real_mean - cells.enrichment_segment).mean())
U["e_for_backbone"] = U.e_for_backbone.fillna(U.enrichment_segment + off)
U["backbone_mean_efpd"] = mean_pred(U.e_for_backbone, U.feed)
U["backbone_ceil_efpd"] = ceil_pred(U.e_for_backbone, U.feed)
U["backbone_mean_B_d"] = U.backbone_mean_efpd * SPR * N_CORE / U.feed
U["backbone_ceil_B_d"] = U.backbone_ceil_efpd * SPR * N_CORE / U.feed

U["d_surrogate_minus_lrmceil_efpd"] = U.ceil_cyclen - U.backbone_ceil_efpd
U["d_surrogate_minus_lrmmean_efpd"] = U.ceil_cyclen - U.backbone_mean_efpd
U["d_surrogate_minus_dbmax_efpd"] = U.ceil_cyclen - U.EFPD_db_max
U["d_surrogate_minus_lrmceil_B_d"] = U.ceil_B_d - U.backbone_ceil_B_d
U["d_surrogate_minus_lrmceil_pct"] = 100 * U.d_surrogate_minus_lrmceil_efpd / U.backbone_ceil_efpd

for c in ("lrmA_a", "lrmA_b"):
    pass
U["lrmA_a"], U["lrmA_b"] = cA[0], cA[1]
U["lrmB_a"], U["lrmB_b"] = aB, bB
U["lrmB_alpha"] = U.feed.map(alphaB)
U["lrmCeil_a"], U["lrmCeil_b"] = aCe, bCe
U["lrmCeil_alpha"] = U.feed.map(alphaCeil)
U["spr_gwd_per_efpd"] = SPR

cols = ["cell", "zone", "enrichment_segment", "feed", "n_batch", "extrapolated",
        "n_cores", "n_pairs", "n_feasible_db", "e_real_mean", "e_real_sd",
        "bu_k1_mix_mean", "EFPD_db_mean", "EFPD_db_sd", "EFPD_db_min", "EFPD_db_max",
        "B_c_db_mean", "B_d_massbal_mean", "B_d_reported_mean",
        "massbal_dev_pct_mean", "massbal_dev_pct_sd", "F_r_db_min", "CBC_max_db_max",
        "lrmA_efpd", "lrmA_bias", "lrmA_rms",
        "lrmB_efpd", "lrmB_bias", "lrmB_rms",
        "lrmC_efpd", "lrmC_bias", "lrmC_rms",
        "e_for_backbone", "backbone_mean_efpd", "backbone_ceil_efpd",
        "backbone_mean_B_d", "backbone_ceil_B_d",
        "ceil_cyclen", "ceil_B_cycle", "ceil_B_d", "min_pred_f_r", "n_feasible",
        "in_distribution", "n_store_pair_feed", "library_id", "per_fa_tU",
        "d_surrogate_minus_lrmceil_efpd", "d_surrogate_minus_lrmceil_pct",
        "d_surrogate_minus_lrmmean_efpd", "d_surrogate_minus_dbmax_efpd",
        "d_surrogate_minus_lrmceil_B_d",
        "lrmA_a", "lrmA_b", "lrmB_a", "lrmB_b", "lrmB_alpha",
        "lrmCeil_a", "lrmCeil_b", "lrmCeil_alpha", "spr_gwd_per_efpd"]
U = U[cols].sort_values(["enrichment_segment", "feed"]).round(6)
dest = ROOT / "data/reports/dbx_lrm_fit.csv"
U.to_csv(dest, index=False, encoding="utf-8")
print(f"\nwrote {dest}  ({len(U)} rows)")

print()
print(U[U.zone == "db_and_mesh"][
    ["cell", "EFPD_db_max", "backbone_ceil_efpd", "ceil_cyclen",
     "d_surrogate_minus_lrmceil_efpd", "d_surrogate_minus_dbmax_efpd",
     "ceil_B_d", "backbone_ceil_B_d", "min_pred_f_r", "F_r_db_min"]].round(2).to_string(index=False))
print()
print("mesh_only cells (LRM extrapolated):")
print(U[U.zone == "mesh_only"][
    ["cell", "backbone_ceil_efpd", "ceil_cyclen", "d_surrogate_minus_lrmceil_efpd",
     "d_surrogate_minus_lrmceil_pct", "ceil_B_d", "backbone_ceil_B_d"]].round(2).to_string(index=False))
print()
print("db_only cells (feed 101 - outside the model mesh):")
print(U[U.zone == "db_only"][
    ["cell", "EFPD_db_mean", "EFPD_db_max", "backbone_ceil_efpd", "B_d_massbal_mean",
     "F_r_db_min"]].round(2).to_string(index=False))

# massbal summary
print("\nmass balance B_d = B_c*241/feed  vs core_mean_discharge_GWd")
for nm, g in [("all", df), ("equilibrium", df[df.metrics_source == "equilibrium_ncyc12"]),
              ("campaign", df[df.metrics_source == "campaign"])]:
    d = g.massbal_dev_pct
    print(f"  {nm:12s} n={len(g):5d} mean={d.mean():+.3f}%  sd={d.std():.3f}%  "
          f"median={d.median():+.3f}%  MAD={(d - d.median()).abs().median():.3f}%  "
          f"min={d.min():+.3f}%  max={d.max():+.3f}%  |dev|>1%: {(d.abs() > 1).sum()}")
bad = df[df.massbal_dev_pct.abs() > 1]
print("  outliers:", bad[["cid", "pair", "feed", "EFPD", "core_mean_discharge_GWd",
                          "massbal_dev_pct", "ncyc", "metrics_source"]].round(3).to_string(index=False))

json.dump({"fit_quality": fitq,
           "A": {"a": cA[0], "b": cA[1]},
           "B": {"a": aB, "b": bB, "alpha": alphaB},
           "C": {"a": aC, "b": bC, "alpha": alphaC},
           "D": {"coef": rD.x.tolist()},
           "ceiling": {"a": aCe, "b": bCe, "alpha": alphaCeil, "rms": rms(rCeil.fun)},
           "alpha_table": tab_alpha.to_dict("records"),
           "spr": SPR},
          open(SCR / "lrm_coeffs.json", "w"), indent=1, default=float)
df.to_pickle(SCR / "cores_aug.pkl")
