"""Step 12: is the cyclen residual predictable from a LATTICE statistic?

The two lat1600-class measurements (T3_T4, T5_T6) both sit on the negative side
(-0.9%, -1.7%).  If that is a class bias the screen must correct for it, because
every design in the 5,874-row table is the same class.
"""
import numpy as np
import os
import opmodel as M
import measured as MEA
from s06_validate import CUR, predict, weights   # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
cal = np.load(HERE + "\\calib_ow.npz")
RS, WB = float(cal["rho_star"]), float(cal["w_b"])
Z = np.load(HERE + "\\hgc_curves.npz")
FFB = {k[:-3]: float(Z[k][0]) for k in Z.files if k.endswith(":ff")}

OW = [r for r in MEA.ALL if r[3] in (117, 121)]
rows = []
for l, a, b, f, cy, cbc, src in OW:
    cyp, bc, rpk, tpk = predict(l, a, b, f, RS)
    ca, cb = CUR[f"{l}:{a}"], CUR[f"{l}:{b}"]
    cs = [ca] if a == b else [ca, cb]
    w = [1.0] if a == b else [68.0, 53.0]
    rm = M.mix(cs, w)
    # lattice statistics
    ts = np.linspace(0.5, 12.0, 20)
    hump = float(np.max(rm(ts)) - rm(0.5))       # Gd burn-out reactivity hump
    dfr = float(rm(0.5)) - M.rho_op(rm, f, bc, 0.0)
    slope = float((rm(bc + 2) - rm(bc - 2)) / 4.0)
    rows.append(dict(l=l, a=a, b=b, f=f, cy=cy, cbc=cbc, src=src,
                     e=100 * (cyp - cy) / cy, ecbc=(rpk - RS) / WB - cbc,
                     hump=hump, dfr=dfr, slope=slope, rpk=rpk))

e = np.array([r["e"] for r in rows])
print("=== cyclen residual (%) vs candidate lattice drivers ===")
for key in ("hump", "dfr", "rpk", "slope", "cbc"):
    x = np.array([r[key] for r in rows])
    c = np.corrcoef(x, e)[0, 1]
    print(f"  corr(resid%, {key:>6}) = {c:+.3f}")

print("\n=== residual by library / class ===")
for lab, sel in (("ga80 (all)", lambda r: r["l"] == "ga80"),
                 ("paramA old (P/Q/S/T0-T2)",
                  lambda r: r["l"] == "paramA" and r["a"][:1] in "PQS"),
                 ("lat1600 T3-T6",
                  lambda r: r["a"] in ("T3", "T4", "T5", "T6"))):
    s = [r for r in rows if sel(r)]
    if not s:
        continue
    v = np.array([r["e"] for r in s])
    print(f"  {lab:<26} n={len(s):>2}  mean {v.mean():+.2f}%  "
          f"sd {v.std():.2f}%  [{v.min():+.2f}, {v.max():+.2f}]")
    for r in s:
        if r["a"] in ("T3", "T4", "T5", "T6"):
            print(f"      {r['a']}_{r['b']}@f{r['f']} ({r['src']}): "
                  f"meas {r['cy']:.1f}  resid {r['e']:+.2f}%  "
                  f"cbc resid {r['ecbc']:+.0f}")

print("\n=== the ONLY in-class measurements ===")
inc = [r for r in rows if r["a"] in ("T3", "T4", "T5", "T6")]
v = np.array([r["e"] for r in inc])
print(f"  n={len(v)}  mean {v.mean():+.3f}%  sd {v.std(ddof=1):.3f}%")
print(f"  -> screen bias correction: multiply predicted cyclen by "
      f"{1-v.mean()/100:.4f}  (i.e. {-v.mean():+.2f}%)")
vc = np.array([r["ecbc"] for r in inc])
print(f"  CBC residual on the same points: mean {vc.mean():+.0f} ppm, "
      f"sd {vc.std(ddof=1):.0f} ppm  -> the model runs CONSERVATIVE (high) "
      f"on this class")
print("\n  NOTE n=4 and the four are only two distinct fuel pairs; treat the")
print("  correction as a centring, not a precision instrument.")
