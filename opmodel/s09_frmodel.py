"""Step 9: the fusion law fails at FIXED pattern -- why, and what survives.

F_r = A * p_hot * FF_hot only holds if p_hot (the BOC power of the hottest fresh
slot) is unchanged.  Swapping in a MORE REACTIVE fresh batch on a pattern tuned
for the old fuel raises p_hot, and the measured F_r goes the wrong way.
Test: does a purely LATTICE-side statistic predict the measured node_peak?
"""
import numpy as np
import os
import opmodel as M
import measured as MEA

HERE = os.path.dirname(os.path.abspath(__file__))
cal = np.load(HERE + "\\calib_ow.npz")
RS, WB = float(cal["rho_star"]), float(cal["w_b"])
Z = np.load(HERE + "\\hgc_curves.npz")
CUR = {k[:-3]: M.Curve(Z[k[:-3] + ":bu"], Z[k[:-3] + ":k"])
       for k in Z.files if k.endswith(":bu")}
FFB = {k[:-3]: float(Z[k][0]) for k in Z.files if k.endswith(":ff")}

rows = []
for l, a, b, f, cy, cbc, arm in MEA.ARMS_FLAT:
    fr, fq, npk, hot = MEA.FR_FLAT[arm]
    if npk is None:
        continue
    cs = [CUR[f"{l}:{a}"]] if a == b else [CUR[f"{l}:{a}"], CUR[f"{l}:{b}"]]
    w = [1.0] if a == b else [68.0, 53.0]
    rm = M.mix(cs, w)
    bc = cy * M.RATE                       # use the MEASURED cycle burnup
    r_core = M.rho_op(rm, f, bc, 0.0)      # BOC core-average rho
    r_hot = CUR[f"{l}:{hot}"](0.5)         # the hot type's post-Xe fresh rho
    r_fresh = float(rm(0.5))
    rows.append(dict(arm=arm, hot=hot, ff=FFB[f"{l}:{hot}"], fr=fr, fq=fq,
                     npk=npk, dr_hot=r_hot - r_core, dr_fresh=r_fresh - r_core,
                     solo=(a == b), cy=cy, cbc=cbc))

print("=" * 78)
print("A. measured node_peak vs the fresh-to-core reactivity spread")
print("=" * 78)
print(f"{'arm':>4}{'hot':>5}{'solo':>6}{'FF':>8}{'d_rho_hot':>11}"
      f"{'d_rho_fresh':>13}{'node_pk':>9}{'F_r':>8}{'A=Fr/(np*FF)':>14}")
for r in rows:
    print(f"{r['arm']:>4}{r['hot']:>5}{str(r['solo']):>6}{r['ff']:>8.4f}"
          f"{r['dr_hot']:>11.5f}{r['dr_fresh']:>13.5f}{r['npk']:>9.4f}"
          f"{r['fr']:>8.4f}{r['fr']/(r['npk']*r['ff']):>14.4f}")

x = np.array([r["dr_fresh"] for r in rows])
y = np.array([r["npk"] for r in rows])
c = np.polyfit(x, y, 1)
p = np.polyval(c, x)
print(f"\n  node_peak = {c[0]:.3f}*d_rho_fresh {c[1]:+.4f}   "
      f"rms {np.sqrt(np.mean((p-y)**2)):.4f}   R2 "
      f"{1-np.var(p-y)/np.var(y):.3f}   (n={len(x)})")
A = np.array([r["fr"] / (r["npk"] * r["ff"]) for r in rows])
print(f"  A = F_r/(node_peak*FF_hot): mean {A.mean():.4f}  sd {A.std():.4f}"
      f"  min {A.min():.4f}  max {A.max():.4f}")

print("\n" + "=" * 78)
print("B. the CHAIN prediction  F_r = A * node_peak(d_rho) * FF_hot")
print("=" * 78)
Am = A.mean()
print(f"{'arm':>4}{'F_r meas':>10}{'F_r pred':>10}{'resid':>8}")
pr = Am * p * np.array([r["ff"] for r in rows])
fr = np.array([r["fr"] for r in rows])
for r, v in zip(rows, pr):
    print(f"{r['arm']:>4}{r['fr']:>10.4f}{v:>10.4f}{v-r['fr']:>+8.4f}")
print(f"  rms {np.sqrt(np.mean((pr-fr)**2)):.4f}   "
      f"vs the FIXED-p law rms "
      f"{np.sqrt(np.mean((1.3200*np.array([r['ff'] for r in rows])-fr)**2)):.4f}")

print("\n" + "=" * 78)
print("C. what the FIXED-pattern fusion law predicted vs what was measured")
print("=" * 78)
print("   flat anchor A*p = F_r(A0)/FF(E2) = 1.5207/1.1520 = 1.3200")
print(f"{'arm':>4}{'hot':>5}{'FF':>8}{'law':>9}{'measured':>10}{'error':>9}")
for r in rows:
    law = 1.3200 * r["ff"]
    print(f"{r['arm']:>4}{r['hot']:>5}{r['ff']:>8.4f}{law:>9.4f}"
          f"{r['fr']:>10.4f}{law-r['fr']:>+9.4f}")
np.savez(HERE + "\\frmodel.npz", np_slope=c[0], np_icpt=c[1], A=Am)
