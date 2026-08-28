"""Step 18: the exact numbers that go into OPSCREEN.md."""
import numpy as np
import os
import pandas as pd
import paths as P
import opmodel as M
import measured as MEA
from s06_validate import CUR, weights   # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(HERE + "\\calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
HA, HB = float(c["hump_a"]), float(c["hump_b"])
TH = np.linspace(0.5, 12.0, 24)
TC = np.linspace(0.5, 8.0, 12)


def full(l, a, b, feed):
    cs = [CUR[f"{l}:{a}"]] if a == b else [CUR[f"{l}:{a}"], CUR[f"{l}:{b}"]]
    rm = M.mix(cs, weights(a, b, feed))
    raw, _ = M.cyclen(rm, feed, RS, "EOC")
    hump = float(np.max(rm(TH)) - rm(0.5))
    cy = raw * (1 + HA + HB * hump)
    bc = cy * M.RATE
    rpk, _ = M.rho_op_peak(rm, feed, bc)
    con = float(np.mean(CUR[f"{l}:{a}"](TC) - CUR[f"{l}:{b}"](TC)))
    return dict(raw=raw, cy=cy, hump=hump, cbc=(rpk - RS) / WB, con=con)


KEY = [("paramA", "T5", "T6", 121, "fr_arms B3, 1 pattern"),
       ("paramA", "T5", "T6", 121, "elite-32 mean"),
       ("paramA", "T5", "T6", 117, "elite-32 mean"),
       ("paramA", "T3", "T4", 121, "fr_arms B2, 1 pattern"),
       ("paramA", "Q1", "Q2", 121, "fr_arms B0"),
       ("paramA", "Q7", "Q8", 121, "fr_arms B1"),
       ("ga80", "E1", "E2", 121, "fr_arms A0"),
       ("ga80", "E1", "E2", 117, "store median n=501"),
       ("ga80", "E3", "E4", 121, "elite-24 mean"),
       ("ga80", "J5", "J6", 121, "elite-32 mean"),
       ("ga80", "K3", "K4", 121, "elite-32 mean"),
       ("ga80", "E3", "E3", 121, "fr_arms C5 single-type")]
LOOK = {}
for l, a, b, f, cy, cb, src in MEA.ALL:
    LOOK.setdefault((l, a, b, f), []).append((cy, cb, src))

print("=" * 104)
print("VALIDATION TABLE (final calibration: rho*=%.5f, w_B=%.4f pcm/ppm, "
      "hump corr 1%+.5f%+.5f*hump)" % (RS, 1e5 * WB, HA, HB))
print("=" * 104)
print(f"{'case':>12}{'feed':>5} {'source':>24} {'cy_meas':>8}{'cy_pred':>8}"
      f"{'d':>7}{'d%':>7} {'cbc_meas':>9}{'cbc_pred':>9}{'d':>6} {'hump':>7}")
done = set()
for l, a, b, f, src in KEY:
    p = full(l, a, b, f)
    for cy, cb, s in LOOK.get((l, a, b, f), []):
        k = (l, a, b, f, s)
        if k in done:
            continue
        done.add(k)
        print(f"{a+'_'+b:>12}{f:>5} {s:>24} {cy:>8.1f}{p['cy']:>8.1f}"
              f"{p['cy']-cy:>+7.1f}{100*(p['cy']-cy)/cy:>+7.2f}"
              f" {cb:>9.0f}{p['cbc']:>9.0f}{p['cbc']-cb:>+6.0f} {p['hump']:>7.4f}")

print("\n=== whole-set residuals, operating window (feeds 117 + 121) ===")
res = []
for l, a, b, f, cy, cb, src in MEA.ALL:
    if f not in (117, 121):
        continue
    p = full(l, a, b, f)
    res.append((p["cy"] - cy, 100 * (p["cy"] - cy) / cy, p["cbc"] - cb, cb))
r = np.array(res)
print(f"  n={len(r)}  cyclen bias {r[:,0].mean():+.2f} EFPD, rms "
      f"{np.sqrt((r[:,0]**2).mean()):.2f} EFPD ({np.sqrt((r[:,1]**2).mean()):.2f}%),"
      f" max|d| {np.abs(r[:,0]).max():.1f} EFPD")
print(f"          CBC   bias {r[:,2].mean():+.0f} ppm, rms "
      f"{np.sqrt((r[:,2]**2).mean()):.0f} ppm, max|d| {np.abs(r[:,2]).max():.0f} ppm")
lo = r[:, 3] <= 1900
print(f"          CBC in the gate region (<=1900 ppm, n={lo.sum()}): bias "
      f"{r[lo,2].mean():+.0f}, rms {np.sqrt((r[lo,2]**2).mean()):.0f}, max|d| "
      f"{np.abs(r[lo,2]).max():.0f} ppm")
print(f"  90% of the CBC residuals lie within +-"
      f"{np.percentile(np.abs(r[lo,2]),90):.0f} ppm")

print("\n" + "=" * 104)
print("ZERO-COST RECOMMENDATION: the realized pairings that clear all three gates")
print("=" * 104)
df = pd.read_csv(P.LAT + "\\screen1600.csv")
IDX = {"T3": 398, "T4": 324, "T5": 2712, "T6": 3246}
FF = {n: float(df.ff_max.values[IDX[n]]) for n in IDX}
for a, b, f in (("T6", "T4", 121), ("T5", "T4", 121), ("T6", "T4", 117),
                ("T5", "T4", 117), ("T3", "T4", 121), ("T5", "T6", 121)):
    p = full("paramA", a, b, f)
    print(f"  {a}(68)_{b}(53hot)@f{f}: cyclen {p['raw']:.0f}..{p['cy']:.0f} EFPD"
          f"  CBC {p['cbc']:.0f} ppm  FF_hot {FF[b]:.4f}  contrast "
          f"{p['con']:+.4f}  F_r_floor {1.03*1.2085*FF[b]:.3f}")
