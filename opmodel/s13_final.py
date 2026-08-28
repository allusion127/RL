"""Step 13: FINAL calibration.

The linear-reactivity equilibrium assumes rho(BU) is straight over the residence.
It is not: a Gd burn-out HUMP in the first ~12 MWd/kgHM makes the batch-average
rho at EOC higher than the straight-line value, so the model runs SHORT on
hump-heavy fuel.  The lat1600 class (5.0-5.3% U, 16-24 low-wt Gd pins) is the
hump-heaviest fuel measured, and is exactly the class the 5,874-design table
lives in -- so this correction is not cosmetic, it is the difference between
screening at 620-645 and screening at 607-632.

  hump   = max_{0.5<=t<=12} rho_mix(t) - rho_mix(0.5)
  cyclen = cyclen_raw * (1 + a + b*hump)
"""
import numpy as np
import os
import opmodel as M
import measured as MEA
from s06_validate import CUR, predict, weights   # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RS = float(np.load(HERE + "\\calib_ow.npz")["rho_star"])
OW = [r for r in MEA.ALL if r[3] in (117, 121)]


def hump_of(rm):
    ts = np.linspace(0.5, 12.0, 24)
    return float(np.max(rm(ts)) - rm(0.5))


base = []
for l, a, b, f, cy, cbc, src in OW:
    cs = [CUR[f"{l}:{a}"]] if a == b else [CUR[f"{l}:{a}"], CUR[f"{l}:{b}"]]
    rm = M.mix(cs, weights(a, b, f))
    cyp, bc = M.cyclen(rm, f, RS, "EOC")
    base.append(dict(l=l, a=a, b=b, f=f, cy=cy, cbc=cbc, src=src,
                     cyp=cyp, rm=rm, hump=hump_of(rm)))

h = np.array([r["hump"] for r in base])
rel = np.array([(r["cy"] - r["cyp"]) / r["cyp"] for r in base])
A, B = np.polyfit(h, rel, 1)[::-1] * 0 + np.polyfit(h, rel, 1)[::-1]
corr = A + B * h
print(f"hump range {h.min():.4f} .. {h.max():.4f}")
print(f"correction  cyclen = cyclen_raw * (1 {A:+.5f} {B:+.5f}*hump)")
cyp2 = np.array([r["cyp"] for r in base]) * (1 + corr)
cym = np.array([r["cy"] for r in base])
raw = np.array([r["cyp"] for r in base])
print(f"  raw:       bias {(raw-cym).mean():+.2f} EFPD  rms "
      f"{np.sqrt(np.mean((raw-cym)**2)):.2f} EFPD  "
      f"({100*np.sqrt(np.mean(((raw-cym)/cym)**2)):.2f}%)  "
      f"max|d| {np.abs(raw-cym).max():.1f}")
print(f"  corrected: bias {(cyp2-cym).mean():+.2f} EFPD  rms "
      f"{np.sqrt(np.mean((cyp2-cym)**2)):.2f} EFPD  "
      f"({100*np.sqrt(np.mean(((cyp2-cym)/cym)**2)):.2f}%)  "
      f"max|d| {np.abs(cyp2-cym).max():.1f}")

# leave-one-CLASS-out: fit on ga80 only, test on the lat1600 four
gm = np.array([r["a"] not in ("T3", "T4", "T5", "T6") for r in base])
A2, B2 = np.polyfit(h[gm], rel[gm], 1)[::-1]
t = ~gm
p = raw[t] * (1 + A2 + B2 * h[t])
print(f"\n  HELD-OUT test (fit excludes all T3-T6 points, n={gm.sum()}):")
print(f"    correction 1 {A2:+.5f} {B2:+.5f}*hump")
for r, v in zip([b for b, k in zip(base, t) if k], p):
    print(f"      {r['a']}_{r['b']}@f{r['f']:<3} ({r['src']:>6}): meas "
          f"{r['cy']:7.1f}  raw {r['cyp']:7.1f} ({100*(r['cyp']-r['cy'])/r['cy']:+.2f}%)"
          f"  corrected {v:7.1f} ({100*(v-r['cy'])/r['cy']:+.2f}%)")

# ---- re-fit w_B with the corrected cycle burnup ----------------------------
rpk = []
for r, c in zip(base, corr):
    bc = r["cyp"] * (1 + c) * M.RATE
    rpk.append(M.rho_op_peak(r["rm"], r["f"], bc)[0])
rpk = np.array(rpk)
cbm = np.array([r["cbc"] for r in base])
WB = float(np.sum((rpk - RS) * cbm) / np.sum(cbm ** 2))
pc = (rpk - RS) / WB
print(f"\nCBC re-fit on the corrected cycle burnups: w_B = {1e5*WB:.4f} pcm/ppm")
print(f"  bias {(pc-cbm).mean():+.0f}  rms {np.sqrt(np.mean((pc-cbm)**2)):.0f}"
      f"  max|d| {np.abs(pc-cbm).max():.0f} ppm  (n={len(cbm)})")
lo = cbm <= 1900
print(f"  restricted to the CBC<=1900 region the gate lives in (n={lo.sum()}): "
      f"bias {(pc-cbm)[lo].mean():+.0f}  rms "
      f"{np.sqrt(np.mean((pc-cbm)[lo]**2)):.0f}  max|d| "
      f"{np.abs(pc-cbm)[lo].max():.0f} ppm")
print("\n  lat1600-class points after correction:")
for r, v, w in zip(base, pc, cyp2):
    if r["a"] in ("T3", "T4", "T5", "T6"):
        print(f"    {r['a']}_{r['b']}@f{r['f']} ({r['src']:>6}): cyclen "
              f"{r['cy']:.1f} -> {w:.1f} ({w-r['cy']:+.1f});  CBC "
              f"{r['cbc']:.0f} -> {v:.0f} ({v-r['cbc']:+.0f})")

np.savez(HERE + "\\calib_final.npz", rho_star=RS, w_b=WB, hump_a=A, hump_b=B)
print(f"\nFINAL: rho*={RS:.5f}  w_B={1e5*WB:.4f} pcm/ppm  "
      f"hump corr (1 {A:+.5f} {B:+.5f}*hump)")
print("saved calib_final.npz")
