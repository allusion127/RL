from __future__ import annotations
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

d = np.load(sys.argv[1], allow_pickle=True)
fold = d["fold"].astype(str); tr = fold == "train"; va = fold == "val"
y = d["f_xy"]; fr_cal = d["fr_cal"]; fr_raw = d["fr_raw"]; emb = d["emb"]
cell = d["cell"].astype(str)
big = sorted({c for c in cell[va] if (cell[va] == c).sum() >= 20})

def report(name, pred):
    e = pred[va] - y[va]
    rb = float(np.mean([spearmanr(y[(cell == c) & va], pred[(cell == c) & va]).statistic
                        for c in big]))
    print(f"{name:56s} MAE {np.mean(np.abs(e)):.4f}  bias {np.mean(e):+.4f}  rho {rb:.4f}"
          f"   {'WIN' if (np.mean(np.abs(e))<0.0767 and rb>0.7263) else ''}")

Z = (emb - emb[tr].mean(0)) / (emb[tr].std(0) + 1e-8)
A = np.logspace(-2, 5, 40)

def ridge_resid(base):
    m = RidgeCV(alphas=A).fit(Z[tr], (y - base)[tr])
    return base + m.predict(Z)

# (d) EXACTLY as-built composition (measured-fitted prior on the RAW predicted
#     f_r row) + a trained linear residual on the frozen trunk embedding.
base_d = 1.2161 * fr_raw - 0.2488
report("(d) as-built prior(RAW pred f_r) + ridge residual", ridge_resid(base_d))
# (e) prior refit on the model's own PREDICTED raw f_r + ridge residual
a2, b2 = np.polyfit(fr_raw[tr], y[tr], 1)
report(f"(e) prior refit on PRED-RAW ({a2:.4f}/{b2:+.4f}) + ridge resid",
       ridge_resid(a2 * fr_raw + b2))
# (f) prior on CALIBRATED pred f_r (refit) + ridge residual
a3, b3 = np.polyfit(fr_cal[tr], y[tr], 1)
report(f"(f) prior refit on PRED-CAL ({a3:.4f}/{b3:+.4f}) + ridge resid",
       ridge_resid(a3 * fr_cal + b3))
# (g) direct linear probe (no composition at all)
report("(g) direct ridge on emb (== linear f_xy row, no prior)",
       ridge_resid(np.zeros_like(y)))
# sanity: are (d)/(e)/(g) the same function? report max abs difference on val
pd_, pe_, pg_ = ridge_resid(base_d), ridge_resid(a2*fr_raw+b2), ridge_resid(np.zeros_like(y))
print(f"\nmax|d-g| val {np.max(np.abs(pd_[va]-pg_[va])):.4f}   "
      f"max|e-g| val {np.max(np.abs(pe_[va]-pg_[va])):.4f}")
# how much does the residual reduce z-space sd? (sigma sanity)
res_g = y - pg_
print(f"residual sd (val) after direct fit: {np.std(res_g[va], ddof=1):.4f}; "
      f"label sd {np.std(y[va], ddof=1):.4f}")
