"""Step 7: residual structure -- is rho* feed-dependent?  How much of the CBC
error is inherited from the cyclen error?  What is the right operating-window
calibration for a screen that only ever asks about f117 / f121?"""
import numpy as np
import os
import opmodel as M
import measured as MEA
from s06_validate import CUR, predict, fit_rho_star, weights  # noqa: F401

ALL = MEA.ALL

print("=" * 78)
print("A. rho* fitted SEPARATELY per feed (EOC variant)")
print("=" * 78)
print(f"{'feed':>5} {'n':>4} {'rho*':>9} {'rms%':>7}   {'n_res':>7}")
for f in sorted({r[3] for r in ALL}):
    rows = [r for r in ALL if r[3] == f]
    rs, rms = fit_rho_star(rows, "EOC")
    print(f"{f:>5} {len(rows):>4} {rs:>9.5f} {100*rms:>6.2f}% {241/f:>8.3f}")

print("\n--> a rising rho* with feed = leakage/margin grows as the core gets")
print("    more fresh fuel (steeper radial gradient). Fit rho*(n_res) linearly:")
xs, ys, ws = [], [], []
for f in sorted({r[3] for r in ALL}):
    rows = [r for r in ALL if r[3] == f]
    rs, _ = fit_rho_star(rows, "EOC")
    xs.append(241.0 / f)
    ys.append(rs)
    ws.append(len(rows))
c = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
print(f"    rho*(n) = {c[0]:+.6f}*n {c[1]:+.6f}   "
      f"(n=1.99 -> {np.polyval(c,1.9917):.5f}, n=2.06 -> "
      f"{np.polyval(c,2.0598):.5f}, n=2.39 -> {np.polyval(c,2.3861):.5f})")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("B. OPERATING-WINDOW calibration (feeds 117 and 121 only)")
print("=" * 78)
OW = [r for r in ALL if r[3] in (117, 121)]
RS, rms = fit_rho_star(OW, "EOC")
print(f"  n={len(OW)}  rho* = {RS:.5f}  cyclen rms = {100*rms:.2f}%")
res = []
for l, a, b, f, cy, cbc, src in OW:
    cyp, bc, rpk, tpk = predict(l, a, b, f, RS)
    res.append((cyp - cy, 100 * (cyp - cy) / cy, rpk, cbc, l, a, b, f, cy, src))
r = np.array([[x[0], x[1]] for x in res])
print(f"  cyclen: bias {r[:,0].mean():+.2f} EFPD, rms {np.sqrt((r[:,0]**2).mean()):.2f}"
      f" EFPD, max|d| {np.abs(r[:,0]).max():.1f} EFPD")

rpk = np.array([x[2] for x in res])
cbm = np.array([x[3] for x in res])
WB = float(np.sum((rpk - RS) * cbm) / np.sum(cbm ** 2))
pc = (rpk - RS) / WB
print(f"  CBC   : w_B = {1e5*WB:.4f} pcm/ppm  ->  bias {(pc-cbm).mean():+.0f},"
      f" rms {np.sqrt(((pc-cbm)**2).mean()):.0f}, max|d| {np.abs(pc-cbm).max():.0f} ppm")

# ---- how much CBC error is inherited from the cyclen error? ----
print("\n" + "=" * 78)
print("C. CBC with the MEASURED cycle length substituted for the predicted one")
print("=" * 78)
rpk2 = []
for l, a, b, f, cy, cbc, src in OW:
    cs = [CUR[f"{l}:{a}"]] if a == b else [CUR[f"{l}:{a}"], CUR[f"{l}:{b}"]]
    rm = M.mix(cs, weights(a, b, f))
    bc = cy * M.RATE                      # MEASURED cycle burnup
    rpk2.append(M.rho_op_peak(rm, f, bc)[0])
rpk2 = np.array(rpk2)
WB2 = float(np.sum((rpk2 - RS) * cbm) / np.sum(cbm ** 2))
pc2 = (rpk2 - RS) / WB2
print(f"  w_B = {1e5*WB2:.4f} pcm/ppm  ->  bias {(pc2-cbm).mean():+.0f},"
      f" rms {np.sqrt(((pc2-cbm)**2).mean()):.0f}, max|d| {np.abs(pc2-cbm).max():.0f} ppm")
print("  => the CBC model's OWN accuracy once cyclen is known; the difference vs")
print("     B is the cyclen error propagating into CBC.")
print(f"  dCBC/dcyclen sensitivity: ", end="")
d = []
for i, (l, a, b, f, cy, cbc, src) in enumerate(OW):
    cs = [CUR[f"{l}:{a}"]] if a == b else [CUR[f"{l}:{a}"], CUR[f"{l}:{b}"]]
    rm = M.mix(cs, weights(a, b, f))
    p1 = M.rho_op_peak(rm, f, (cy - 5) * M.RATE)[0]
    p2 = M.rho_op_peak(rm, f, (cy + 5) * M.RATE)[0]
    d.append((p2 - p1) / WB2 / 10.0)
print(f"{np.mean(d):+.1f} ppm per EFPD (mean over the window)")

# ---- worst offenders ----
print("\n" + "=" * 78)
print("D. WORST 12 POINTS in the operating window")
print("=" * 78)
order = np.argsort(-np.abs(pc - cbm))
print(f"{'case':>14}{'feed':>5}{'src':>8}{'cy_meas':>9}{'cy_err':>8}"
      f"{'cbc_meas':>9}{'cbc_err':>9}")
for i in order[:12]:
    l, a, b, f, cy, cbc, src = OW[i]
    print(f"{(a+'_'+b):>14}{f:>5}{src:>8}{cy:>9.1f}{res[i][0]:>+8.1f}"
          f"{cbc:>9.0f}{pc[i]-cbc:>+9.0f}")

np.savez(os.path.dirname(os.path.abspath(__file__)) + "\\calib_ow.npz",
         rho_star=RS, w_b=WB, rho_star_slope=c[0], rho_star_icpt=c[1])
print(f"\nADOPTED FOR THE SCREEN: rho* = {RS:.5f}, w_B = {1e5*WB:.4f} pcm/ppm")
print("saved calib_ow.npz")
