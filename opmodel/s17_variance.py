"""Step 17: replace the empirical hump patch with a PHYSICAL term.

Unweighted core-average rho is the wrong criticality statistic: assemblies with
higher k carry more power, so the flux-weighted average exceeds the arithmetic
one.  Linearising the power sharing, p_j = 1 + a*(rho_j - rho_bar), gives

    rho_eff = rho_bar + a * Var_w(rho)          (exactly, to first order)

so the EOC criticality condition becomes  rho_bar + a*Var(rho) = rho*.
Fuel with a big Gd hump has a big EOC spread, which is why the unweighted model
ran short on exactly that fuel.  ONE extra global parameter, fitted with rho*.
"""
import numpy as np
import os
import opmodel as M
import measured as MEA
from s06_validate import CUR, weights   # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
OW = [r for r in MEA.ALL if r[3] in (117, 121)]


def pops(l, a, b, feed):
    """(weights, curves) of the whole equilibrium assembly population."""
    if a == b:
        fr = [(1.0, CUR[f"{l}:{a}"])]
    else:
        fr = [(68.0 / 121.0, CUR[f"{l}:{a}"]), (53.0 / 121.0, CUR[f"{l}:{b}"])]
    out = []
    for n, k in M.batch_weights(feed):
        for f, cv in fr:
            out.append((n * f, k, cv))
    return out


def rho_eff(pp, bc, a, t=0.0):
    w = np.array([p[0] for p in pp])
    r = np.array([p[2](t + p[1] * bc) for p in pp])
    w = w / w.sum()
    m = float(w @ r)
    v = float(w @ (r - m) ** 2)
    return m + a * v, m, v


def solve_bc(pp, a, rs, lo=8.0, hi=45.0):
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        # EOC: every batch has advanced one more cycle
        w = np.array([p[0] for p in pp])
        r = np.array([p[2]((p[1] + 1) * mid) for p in pp])
        w = w / w.sum()
        mn = float(w @ r)
        v = float(w @ (r - mn) ** 2)
        lo, hi = (mid, hi) if mn + a * v > rs else (lo, mid)
    return 0.5 * (lo + hi)


PP = [pops(l, x, y, f) for l, x, y, f, *_ in OW]
CYM = np.array([r[4] for r in OW])
CBM = np.array([r[5] for r in OW])


def score(a, rs):
    p = np.array([solve_bc(pp, a, rs) / M.RATE for pp in PP])
    return p


print("grid search over (a, rho*)")
best = None
for a in np.linspace(0.0, 60.0, 61):
    for rs in np.linspace(0.010, 0.030, 81):
        p = score(a, rs)
        s = float(np.mean(((p - CYM) / CYM) ** 2))
        if best is None or s < best[0]:
            best = (s, a, rs, p)
s, A, RS, P = best
print(f"  best: a = {A:.2f}   rho* = {RS:.5f}   cyclen rms = {100*np.sqrt(s):.3f}%"
      f"  ({np.sqrt(np.mean((P-CYM)**2)):.2f} EFPD)")
print(f"  bias {(P-CYM).mean():+.2f} EFPD  max|d| {np.abs(P-CYM).max():.1f} EFPD")
print("  (unweighted model was rms 5.89 EFPD / 0.91%; hump-patched 4.26 / 0.66%)")

# is the hump correlation gone?
TH = np.linspace(0.5, 12.0, 24)
hump = []
for (l, x, y, f, *_), pp in zip(OW, PP):
    cs = [CUR[f"{l}:{x}"]] if x == y else [CUR[f"{l}:{x}"], CUR[f"{l}:{y}"]]
    rm = M.mix(cs, weights(x, y, f))
    hump.append(float(np.max(rm(TH)) - rm(0.5)))
hump = np.array(hump)
rel = (P - CYM) / CYM
print(f"  corr(residual, hump) = {np.corrcoef(hump, rel)[0,1]:+.3f} "
      f"(was -0.687 for the unweighted model)")

# CBC on the same footing
rpk = []
for pp, cy in zip(PP, P):
    bc = cy * M.RATE
    feedbc = [rho_eff(pp, bc, A, t)[0] for t in np.linspace(0, bc, 21)]
    rpk.append(max(feedbc))
rpk = np.array(rpk)
WB = float(np.sum((rpk - RS) * CBM) / np.sum(CBM ** 2))
pc = (rpk - RS) / WB
lo = CBM <= 1900
print(f"\nCBC: w_B = {1e5*WB:.4f} pcm/ppm -> bias {(pc-CBM).mean():+.0f}, rms "
      f"{np.sqrt(np.mean((pc-CBM)**2)):.0f}, max|d| {np.abs(pc-CBM).max():.0f} ppm")
print(f"     in the gate region CBC<=1900 (n={lo.sum()}): bias "
      f"{(pc-CBM)[lo].mean():+.0f}, rms {np.sqrt(np.mean((pc-CBM)[lo]**2)):.0f},"
      f" max|d| {np.abs(pc-CBM)[lo].max():.0f} ppm")

print("\n=== lat1600-class points ===")
for (l, x, y, f, cy, cb, src), p, v in zip(OW, P, pc):
    if x in ("T3", "T4", "T5", "T6"):
        print(f"  {x}_{y}@f{f} ({src:>6}): cyclen {cy:7.1f} -> {p:7.1f} "
              f"({p-cy:+6.1f}, {100*(p-cy)/cy:+.2f}%);  CBC {cb:6.0f} -> "
              f"{v:6.0f} ({v-cb:+.0f})")

print("\n=== held-out: fit with ALL T3-T6 points removed ===")
mask = [x not in ("T3", "T4", "T5", "T6") for l, x, y, *_ in OW]
mask = np.array(mask)
best2 = None
for a in np.linspace(0.0, 60.0, 61):
    for rs in np.linspace(0.010, 0.030, 81):
        p = score(a, rs)
        s2 = float(np.mean(((p[mask] - CYM[mask]) / CYM[mask]) ** 2))
        if best2 is None or s2 < best2[0]:
            best2 = (s2, a, rs, p)
s2, A2, RS2, P2 = best2
print(f"  fit on n={mask.sum()}: a = {A2:.2f}, rho* = {RS2:.5f}, "
      f"rms {100*np.sqrt(s2):.3f}%")
for (l, x, y, f, cy, cb, src), p in zip(OW, P2):
    if x in ("T3", "T4", "T5", "T6"):
        print(f"    {x}_{y}@f{f} ({src:>6}): meas {cy:7.1f} pred {p:7.1f} "
              f"({100*(p-cy)/cy:+.2f}%)")

np.savez(HERE + "\\calib_var.npz", a=A, rho_star=RS, w_b=WB)
print(f"\nsaved calib_var.npz  (a={A}, rho*={RS}, w_B={1e5*WB:.4f} pcm/ppm)")
