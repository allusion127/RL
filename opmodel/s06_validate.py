"""Step 6: calibrate (rho*, w_B) and report residuals on every measured point."""
import numpy as np
import os
import paths as P
import opmodel as M
import measured as MEA

Z = np.load(os.path.dirname(os.path.abspath(__file__)) + "\\hgc_curves.npz")
CUR = {}
for k in Z.files:
    if k.endswith(":bu"):
        base = k[:-3]
        CUR[base] = M.Curve(Z[base + ":bu"], Z[base + ":k"], base)
FF = {k[:-3]: float(Z[k][0]) for k in Z.files if k.endswith(":ff")}

# fresh-role split: 68/53 of the feed at the reference f121 patterns, scaled to
# other feeds.  A==B -> single population.
def weights(a, b, feed):
    if a == b:
        return [1.0]
    return [68.0, 53.0]


def predict(lib, a, b, feed, rho_star, variant="EOC"):
    ca, cb = CUR[f"{lib}:{a}"], CUR[f"{lib}:{b}"]
    cs = [ca] if a == b else [ca, cb]
    rm = M.mix(cs, weights(a, b, feed))
    cy, bc = M.cyclen(rm, feed, rho_star, variant)
    rpk, tpk = M.rho_op_peak(rm, feed, bc)
    return cy, bc, rpk, tpk


def fit_rho_star(rows, variant="EOC", lo=0.005, hi=0.035):
    """least squares in RELATIVE cyclen error."""
    best = None
    for rs in np.linspace(lo, hi, 601):
        e = [(predict(l, a, b, f, rs, variant)[0] - cy) / cy
             for l, a, b, f, cy, _, _ in rows]
        s = float(np.mean(np.square(e)))
        if best is None or s < best[0]:
            best = (s, rs)
    return best[1], np.sqrt(best[0])


SETS = {"ARMS_FLAT (1 pattern, f121)": MEA.ARMS_FLAT,
        "ARMS_MINFR (1 pattern)": MEA.ARMS_MINFR,
        "SCREEN11 (1 pattern)": MEA.SCREEN11,
        "ENSEMBLE (24-32 pattern means)": MEA.ENSEMBLE,
        "STORE (medians, feeds 101-141)": MEA.STORE}

print("=" * 78)
print("A. rho* CALIBRATION  (fit to cyclen only)")
print("=" * 78)
for variant in ("EOC", "B1"):
    for name, rows in list(SETS.items()) + [("ALL", MEA.ALL)]:
        rs, rms = fit_rho_star(rows, variant)
        print(f"  {variant:>3}  {name:<32} n={len(rows):>2}  "
              f"rho*={rs:.5f}  rms_rel={100*rms:.2f}%")
    print()

RS_FIT, _ = fit_rho_star(MEA.ALL, "EOC")
print(f"--> adopted rho* = {RS_FIT:.5f}  (EOC variant, fit on all "
      f"{len(MEA.ALL)} points)")
print("    (the bootstrap-diagnosis value was 0.0168, fit on 2 cy1 EOCs)")

# ---------------------------------------------------------------- CBC fit
print("\n" + "=" * 78)
print("B. CBC CALIBRATION   CBC = (rho_op_peak - rho*) / w_B")
print("=" * 78)
rpk_all, cbc_all = [], []
for l, a, b, f, cy, cbc, tag in MEA.ALL:
    _, _, rpk, _ = predict(l, a, b, f, RS_FIT)
    rpk_all.append(rpk)
    cbc_all.append(cbc)
rpk_all = np.array(rpk_all)
cbc_all = np.array(cbc_all)

wb_pt = (rpk_all - RS_FIT) / cbc_all
print(f"  implied w_B per point (pcm/ppm): mean {1e5*wb_pt.mean():.3f}  "
      f"sd {1e5*wb_pt.std():.3f}  min {1e5*wb_pt.min():.3f}  "
      f"max {1e5*wb_pt.max():.3f}")
WB = float(np.sum((rpk_all - RS_FIT) * cbc_all) / np.sum(cbc_all ** 2))
pred1 = (rpk_all - RS_FIT) / WB
print(f"  1-parameter fit (physics form, intercept forced):  "
      f"w_B = {1e5*WB:.4f} pcm/ppm")
print(f"     residual: rms {np.sqrt(np.mean((pred1-cbc_all)**2)):.1f} ppm, "
      f"max |d| {np.abs(pred1-cbc_all).max():.1f} ppm, "
      f"bias {(pred1-cbc_all).mean():+.1f} ppm")
A2, B2c = np.polyfit(rpk_all, cbc_all, 1)
pred2 = A2 * rpk_all + B2c
print(f"  2-parameter fit  CBC = {A2:.1f}*rho_peak {B2c:+.1f}")
print(f"     residual: rms {np.sqrt(np.mean((pred2-cbc_all)**2)):.1f} ppm, "
      f"max |d| {np.abs(pred2-cbc_all).max():.1f} ppm")
print(f"     implied w_B {1e5/A2:.4f} pcm/ppm, implied rho* {-B2c/A2:.5f}")
print(f"  (the OLD screen contour was CBC = 26176*rbar_peak + 133 on a 3-BATCH "
      f"surrogate convention)")

# ---------------------------------------------------------------- residuals
print("\n" + "=" * 78)
print("C. PER-POINT RESIDUALS  (rho*=%.5f, w_B=%.4f pcm/ppm)"
      % (RS_FIT, 1e5 * WB))
print("=" * 78)
print(f"{'set':>10} {'case':>16} {'feed':>5} | {'cy_meas':>8} {'cy_pred':>8}"
      f" {'d':>7} {'d%':>6} | {'cbc_meas':>8} {'cbc_pred':>8} {'d':>7}")
agg = {}
for name, rows in SETS.items():
    tag = name.split()[0]
    for l, a, b, f, cy, cbc, src in rows:
        cyp, bc, rpk, tpk = predict(l, a, b, f, RS_FIT)
        cbp = (rpk - RS_FIT) / WB
        agg.setdefault(tag, []).append((cyp - cy, 100 * (cyp - cy) / cy,
                                        cbp - cbc))
        nm = f"{a}_{b}" if a != b else f"{a}x{f}"
        print(f"{tag:>10} {nm:>16} {f:>5} | {cy:>8.1f} {cyp:>8.1f}"
              f" {cyp-cy:>+7.1f} {100*(cyp-cy)/cy:>+6.2f} |"
              f" {cbc:>8.0f} {cbp:>8.0f} {cbp-cbc:>+7.0f}")

print("\n" + "=" * 78)
print("D. RESIDUAL SUMMARY BY SET")
print("=" * 78)
print(f"{'set':>10} {'n':>4} {'cy bias':>9} {'cy rms':>8} {'cy rms%':>8}"
      f" {'cbc bias':>9} {'cbc rms':>9} {'cbc max|d|':>11}")
for tag, v in agg.items():
    a = np.array(v)
    print(f"{tag:>10} {len(a):>4} {a[:,0].mean():>+9.1f} "
          f"{np.sqrt((a[:,0]**2).mean()):>8.1f} "
          f"{np.sqrt((a[:,1]**2).mean()):>7.2f}% "
          f"{a[:,2].mean():>+9.0f} {np.sqrt((a[:,2]**2).mean()):>9.0f} "
          f"{np.abs(a[:,2]).max():>11.0f}")
a = np.array([x for v in agg.values() for x in v])
print(f"{'ALL':>10} {len(a):>4} {a[:,0].mean():>+9.1f} "
      f"{np.sqrt((a[:,0]**2).mean()):>8.1f} "
      f"{np.sqrt((a[:,1]**2).mean()):>7.2f}% "
      f"{a[:,2].mean():>+9.0f} {np.sqrt((a[:,2]**2).mean()):>9.0f} "
      f"{np.abs(a[:,2]).max():>11.0f}")

np.savez(os.path.dirname(os.path.abspath(__file__)) + "\\calib.npz",
         rho_star=RS_FIT, w_b=WB, cbc_a=A2, cbc_b=B2c)
print("\nsaved calib.npz")
