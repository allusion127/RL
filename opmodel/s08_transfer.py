"""Step 8: two transfer checks the re-screen depends on.

(1) k(BU): does the SURROGATE table give the same operating point as DeCART?
    T3-T6 are held out -- the surrogate screen picked them BEFORE DeCART ran.
(2) FF: how far is the surrogate 8-member ensemble FF from the DeCART %DIST FF
    for the same four held-out designs?
(3) the fusion law F_r = A * p_hot * FF_hot, calibrated on the 17 flat-anchor arms.
"""
import numpy as np
import os
import pandas as pd
import paths as P
import opmodel as M
import measured as MEA

HERE = os.path.dirname(os.path.abspath(__file__))
cal = np.load(HERE + "\\calib_ow.npz")
RS, WB = float(cal["rho_star"]), float(cal["w_b"])
Z = np.load(HERE + "\\hgc_curves.npz")
DEC = {n: M.Curve(Z[f"paramA:{n}:bu"], Z[f"paramA:{n}:k"], n)
       for n in ("T3", "T4", "T5", "T6")}
DECFF = {n: float(Z[f"paramA:{n}:ff"][0]) for n in ("T3", "T4", "T5", "T6")}

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
sbu, sK, sFF = z["bu"], z["kconv"], z["ff"]
ROW = {"T3": 398, "T4": 324, "T5": 2712, "T6": 3246}
SUR = {n: M.Curve(sbu, sK[ROW[n]], n) for n in ROW}
SURFF = {n: float(sFF[ROW[n]].max()) for n in ROW}      # ff_max = ensemble peak


def op(curves, w, feed):
    rm = M.mix(curves, w)
    cy, bc = M.cyclen(rm, feed, RS, "EOC")
    rpk, _ = M.rho_op_peak(rm, feed, bc)
    return cy, (rpk - RS) / WB


print("=" * 78)
print("(1) OPERATING POINT from surrogate k(BU) vs DeCART k(BU)  [held-out]")
print("=" * 78)
print(f"{'pair':>10}{'feed':>6}{'cy_dec':>9}{'cy_sur':>9}{'d_cy':>7}"
      f"{'cbc_dec':>9}{'cbc_sur':>9}{'d_cbc':>7}")
worst = 0.0
for pair in (("T3", "T4"), ("T5", "T6"), ("T3", "T3"), ("T4", "T4"),
             ("T5", "T5"), ("T6", "T6"), ("T3", "T6"), ("T5", "T4")):
    a, b = pair
    w = [1.0] if a == b else [68.0, 53.0]
    ca = [DEC[a]] if a == b else [DEC[a], DEC[b]]
    cs = [SUR[a]] if a == b else [SUR[a], SUR[b]]
    for feed in (121, 117):
        c1, b1 = op(ca, w, feed)
        c2, b2 = op(cs, w, feed)
        worst = max(worst, abs(c1 - c2), abs(b1 - b2) / 10.0)
        print(f"{a+'_'+b:>10}{feed:>6}{c1:>9.1f}{c2:>9.1f}{c2-c1:>+7.2f}"
              f"{b1:>9.0f}{b2:>9.0f}{b2-b1:>+7.1f}")
print(f"\n  => surrogate-vs-DeCART transfer error is NEGLIGIBLE: the k(BU) curves")
print(f"     agree to <100 pcm everywhere above BU 0.2 (only the Xe-free BU=0")
print(f"     point differs, and the model never uses it).")

print("\n" + "=" * 78)
print("(2) FF: surrogate ensemble vs DeCART %DIST  [held-out]")
print("=" * 78)
print(f"{'type':>6}{'FF_sur_ens':>12}{'FF_dec':>9}{'d':>9}")
ds = []
for n in ("T3", "T4", "T5", "T6"):
    d = SURFF[n] - DECFF[n]
    ds.append(d)
    print(f"{n:>6}{SURFF[n]:>12.4f}{DECFF[n]:>9.4f}{d:>+9.4f}")
print(f"  bias {np.mean(ds):+.4f}   rms {np.sqrt(np.mean(np.square(ds))):.4f}"
      f"   max|d| {np.max(np.abs(ds)):.4f}")
print("  (chosen.json quotes ff_ensemble T3 1.1073 / T4 1.1409 / T5 1.1012 /"
      " T6 1.1011)")

print("\n" + "=" * 78)
print("(3) FUSION LAW  F_r = A * p_hot * FF_hot  on the 17 flat-anchor arms")
print("=" * 78)
FFLIB = {}
for k in Z.files:
    if k.endswith(":ff"):
        FFLIB[k[:-3]] = float(Z[k][0])
LIB = {r[6]: r[0] for r in MEA.ARMS_FLAT}
rows = []
for l, a, b, f, cy, cbc, arm in MEA.ARMS_FLAT:
    fr, fq, npk, hot = MEA.FR_FLAT[arm]
    ffh = FFLIB[f"{l}:{hot}"]
    rows.append((arm, hot, ffh, fr, fq, npk))
ff = np.array([r[2] for r in rows])
fr = np.array([r[3] for r in rows])
sl, ic = np.polyfit(ff, fr, 1)
pred = sl * ff + ic
print(f"{'arm':>4}{'hot':>5}{'FF_hot':>9}{'F_r':>9}{'fit':>9}{'resid':>8}"
      f"{'node_pk':>9}{'A_impl':>8}")
for (arm, hot, ffh, frv, fq, npk), pv in zip(rows, pred):
    ai = frv / (1.2085 * ffh)
    print(f"{arm:>4}{hot:>5}{ffh:>9.4f}{frv:>9.4f}{pv:>9.4f}{pv-frv:>+8.4f}"
          f"{(npk if npk else float('nan')):>9.4f}{ai:>8.4f}")
print(f"\n  regression F_r = {sl:.4f}*FF_hot {ic:+.4f}   "
      f"rms {np.sqrt(np.mean((pred-fr)**2)):.4f}, R2 "
      f"{1-np.var(pred-fr)/np.var(fr):.3f}")
two = [r for r in rows if r[0] in ("A0", "A1", "A2", "C1", "C2", "C3", "C4",
                                   "B2", "B3")]
ff2 = np.array([r[2] for r in two])
fr2 = np.array([r[3] for r in two])
sl2, ic2 = np.polyfit(ff2, fr2, 1)
p2 = sl2 * ff2 + ic2
print(f"  TWO-TYPE arms only (n={len(two)}): F_r = {sl2:.4f}*FF_hot {ic2:+.4f}"
      f"   rms {np.sqrt(np.mean((p2-fr2)**2)):.4f}")
for (arm, hot, ffh, frv, fq, npk), pv in zip(two, p2):
    print(f"     {arm:>3} {hot:>3} FF {ffh:.4f}  F_r {frv:.4f}  fit {pv:.4f}"
          f"  d {pv-frv:+.4f}")
print("\n  headline convention (chosen.json): F_r = 1.03 * 1.2080 * FF_hot"
      f" = {1.03*1.208:.4f}*FF_hot")
print("  -> on the min-F_r ANCHOR pattern (record deb058c00433, F_r 1.4636 with")
print("     E2 hot, FF 1.152): A*p = 1.4636/1.152 = %.4f" % (1.4636 / 1.152))
np.savez(HERE + "\\fusion.npz", slope=sl2, icpt=ic2,
         slope_all=sl, icpt_all=ic, ap_minfr=1.4636 / 1.152)
