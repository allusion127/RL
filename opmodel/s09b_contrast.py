"""Step 9b: the real driver of node_peak on a FIXED pattern is the reactivity
CONTRAST between the two fresh roles -- the zoning the pattern was built around.
T5/T6 differ by 0.05% U-235: the pattern degenerates into a single-type load and
node_peak jumps 1.21 -> 1.39, which is exactly why the flattest lattice measured
the WORST F_r."""
import numpy as np
import os
import opmodel as M
import measured as MEA

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(HERE + "\\hgc_curves.npz")
CUR = {k[:-3]: M.Curve(Z[k[:-3] + ":bu"], Z[k[:-3] + ":k"])
       for k in Z.files if k.endswith(":bu")}
FFB = {k[:-3]: float(Z[k][0]) for k in Z.files if k.endswith(":ff")}

rows = []
for l, a, b, f, cy, cbc, arm in MEA.ARMS_FLAT:
    fr, fq, npk, hot = MEA.FR_FLAT[arm]
    if npk is None:
        continue
    ca, cb = CUR[f"{l}:{a}"], CUR[f"{l}:{b}"]
    cs = [ca] if a == b else [ca, cb]
    w = [1.0] if a == b else [68.0, 53.0]
    rm = M.mix(cs, w)
    bc = cy * M.RATE
    dfresh = float(rm(0.5)) - M.rho_op(rm, f, bc, 0.0)
    # role contrast: post-Xe reactivity difference between the two fresh roles,
    # averaged over the first third of a cycle (the pattern's zoning window)
    ts = np.linspace(0.5, bc / 3.0, 20)
    contrast = float(np.mean(ca(ts) - cb(ts)))
    rows.append((arm, hot, FFB[f"{l}:{hot}"], fr, npk, contrast, dfresh))

print(f"{'arm':>4}{'hot':>5}{'FF':>8}{'contrast':>10}{'d_fresh':>10}"
      f"{'node_pk':>9}{'F_r':>8}")
for r in rows:
    print(f"{r[0]:>4}{r[1]:>5}{r[2]:>8.4f}{r[5]:>10.5f}{r[6]:>10.5f}"
          f"{r[4]:>9.4f}{r[3]:>8.4f}")

X = np.array([[1.0, r[5], r[6]] for r in rows])
y = np.array([r[4] for r in rows])
co, *_ = np.linalg.lstsq(X, y, rcond=None)
p = X @ co
print(f"\nnode_peak = {co[0]:.4f} {co[1]:+.4f}*contrast {co[2]:+.4f}*d_fresh")
print(f"  rms {np.sqrt(np.mean((p-y)**2)):.4f}  R2 {1-np.var(p-y)/np.var(y):.3f}")
for r, v in zip(rows, p):
    print(f"   {r[0]:>3} npk {r[4]:.4f} pred {v:.4f} d {v-r[4]:+.4f}")

ff = np.array([r[2] for r in rows])
fr = np.array([r[3] for r in rows])
A = float(np.sum(fr * p * ff) / np.sum((p * ff) ** 2))
frp = A * p * ff
print(f"\nF_r = A*node_peak_pred*FF_hot, A = {A:.4f}")
print(f"  rms {np.sqrt(np.mean((frp-fr)**2)):.4f}  max|d| "
      f"{np.abs(frp-fr).max():.4f}")
for r, v in zip(rows, frp):
    print(f"   {r[0]:>3} F_r {r[3]:.4f} pred {v:.4f} d {v-r[3]:+.4f}")
np.savez(HERE + "\\frmodel2.npz", c=co, A=A)
