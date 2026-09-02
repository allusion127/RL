from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

d = np.load(sys.argv[1], allow_pickle=True)
fold = d["fold"].astype(str); tr = fold == "train"; va = fold == "val"
y = d["f_xy"]; fr_cal = d["fr_cal"]; fr_raw = d["fr_raw"]; fr_meas = d["f_r_meas"]
emb = d["emb"]; cell = d["cell"].astype(str)
print(f"train {tr.sum()} val {va.sum()} emb {emb.shape}")

big = sorted({c for c in cell[va] if (cell[va] == c).sum() >= 20})
print(f"G3 cells {len(big)} rows {sum((cell[va]==c).sum() for c in big)}")

def rho_bar(pred):
    rs = []
    for c in big:
        m = (cell == c) & va
        rs.append(spearmanr(y[m], pred[m]).statistic)
    return float(np.mean(rs)), rs

def report(name, pred):
    e = pred[va] - y[va]
    rb, _ = rho_bar(pred)
    print(f"{name:52s} MAE {np.mean(np.abs(e)):.4f}  bias {np.mean(e):+.4f}  "
          f"sd {np.std(e, ddof=1):.4f}  rho_bar {rb:.4f}")
    return dict(name=name, mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                sd=float(np.std(e, ddof=1)), rho=rb)

R = []
# ---- bars ---------------------------------------------------------------
R.append(report("BAR serving proxy 1.2176*fr_cal-0.2519", 1.2176*fr_cal - 0.2519))
R.append(report("(ideal) prior on MEASURED f_r (1.2161/-0.2488)", 1.2161*fr_meas - 0.2488))
R.append(report("as-built: prior on RAW pred f_r", 1.2161*fr_raw - 0.2488))
# ---- (a) affine refit on PREDICTED calibrated f_r ------------------------
a, b = np.polyfit(fr_cal[tr], y[tr], 1)
print(f"\n(a) refit on predicted-calibrated f_r: a={a:.4f} b={b:+.4f}")
pa = a*fr_cal + b
R.append(report("(a) affine refit on PREDICTED-CAL f_r", pa))
a2, b2 = np.polyfit(fr_raw[tr], y[tr], 1)
print(f"(a') refit on predicted-RAW f_r: a={a2:.4f} b={b2:+.4f}")
R.append(report("(a') affine refit on PREDICTED-RAW f_r", a2*fr_raw + b2))

# ---- feature blocks -----------------------------------------------------
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

mu_e, sd_e = emb[tr].mean(0), emb[tr].std(0) + 1e-8
Z = (emb - mu_e) / sd_e
X_emb = Z
X_embfr = np.column_stack([Z, fr_cal])
ALPHAS = np.logspace(-2, 5, 40)

def ridge(X, target, base):
    m = RidgeCV(alphas=ALPHAS).fit(X[tr], target[tr])
    return base + m.predict(X), m.alpha_

def gbm(X, target, base, seed=0):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      max_depth=None, l2_regularization=1.0,
                                      random_state=seed).fit(X[tr], target[tr])
    return base + m.predict(X)

resid = y - pa
print()
p, al = ridge(X_emb, resid, pa);   R.append(report(f"(b) (a)+ridge resid on emb (alpha={al:.3g})", p))
p, al = ridge(X_embfr, resid, pa); R.append(report(f"(b+) (a)+ridge resid on emb+fr (alpha={al:.3g})", p))
R.append(report("(b-gbm) (a)+GBM resid on emb", gbm(X_emb, resid, pa)))
R.append(report("(b+gbm) (a)+GBM resid on emb+fr", gbm(X_embfr, resid, pa)))

zero = np.zeros_like(y)
p, al = ridge(X_emb, y, zero);   R.append(report(f"(c) ridge direct on emb (alpha={al:.3g})", p))
p, al = ridge(X_embfr, y, zero); R.append(report(f"(c+) ridge direct on emb+fr (alpha={al:.3g})", p))
R.append(report("(c-gbm) GBM direct on emb", gbm(X_emb, y, zero)))
R.append(report("(c+gbm) GBM direct on emb+fr", gbm(X_embfr, y, zero)))

print("\n--- vs bar MAE 0.0767 / rho 0.7263 ---")
for r in R:
    print(f"{r['name']:52s} {'WIN ' if (r['mae']<0.0767 and r['rho']>0.7263) else '    '}"
          f"dMAE {r['mae']-0.0767:+.4f} drho {r['rho']-0.7263:+.4f}")
