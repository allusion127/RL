"""Step 21: the final top-5 candidate operating points, fully specified."""
import numpy as np
import os
import pandas as pd
import paths as P
import opmodel as M
from s06_validate import CUR as HCUR, weights   # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(HERE + "\\calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
HA, HB = float(c["hump_a"]), float(c["hump_b"])
fm = np.load(HERE + "\\frmodel2.npz")
NPC, AFUS = fm["c"], float(fm["A"])

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
bu0, K = z["bu"], z["kconv"]
FFM = df.ff_max.values
TH = np.linspace(0.5, 12.0, 24)
TC = np.linspace(0.5, 8.0, 12)


def row_of(uh, gw, ng, gp, pat, zon):
    m = (np.isclose(df.u_high, uh) & np.isclose(df.gd_wt, gw)
         & (df.n_gd == ng) & (df.gd_positions == gp)
         & (df.pattern == pat) & (df.zoning == zon))
    k = np.flatnonzero(m.values)
    assert len(k) == 1, (uh, gw, ng, gp, len(k))
    return int(k[0])


def evalpair(ca, cb, ffb, feed):
    cs = [ca] if cb is None else [ca, cb]
    w = [1.0] if cb is None else [68.0, 53.0]
    rm = M.mix(cs, w)
    hump = float(np.max(rm(TH)) - rm(0.5))
    raw, _ = M.cyclen(rm, feed, RS, "EOC")
    cy = raw * (1 + HA + HB * hump)
    bc = cy * M.RATE
    rpk, _ = M.rho_op_peak(rm, feed, bc)
    con = float(np.mean(ca(TC) - (cb or ca)(TC)))
    npk = NPC[0] + NPC[1] * con + NPC[2] * (
        float(rm(0.5)) - M.rho_op(rm, feed, bc, 0.0))
    return dict(raw=raw, cy=cy, hump=hump, cbc=(rpk - RS) / WB, con=con,
                npk=npk, ff=ffb, flr=1.03 * 1.2085 * ffb,
                fr_fix=AFUS * npk * ffb, dis=bc * (241.0 / feed))


CAND = [
    ("R1  T1(68) / T4(53)   @f117  [0 new]", "paramA", "T1", "T4", 117),
    ("R2  T6(68) / T4(53)   @f121  [0 new]", "paramA", "T6", "T4", 121),
    ("R3  P9(68) / T4(53)   @f117  [0 new]", "paramA", "P9", "T4", 117),
    ("R4  T5(68) / T4(53)   @f121  [0 new]", "paramA", "T5", "T4", 121),
    ("R5  P0(68) / T4(53)   @f117  [0 new]", "paramA", "P0", "T4", 117),
    ("R6  T5(68) / T6(53)   @f121  [0 new, TODAY'S CELL]", "paramA", "T5", "T6", 121),
    ("R7  T3(68) / T4(53)   @f121  [0 new]", "paramA", "T3", "T4", 121),
]
print("=" * 104)
print("REALIZED (zero DeCART, zero library rebuild -- one MASTER bootstrap each)")
print("=" * 104)
print(f"{'candidate':>44}{'raw':>7}{'cy':>7}{'CBC':>6}{'FFhot':>8}{'contr':>9}"
      f"{'hump':>7}{'npk':>6}{'Fr_flr':>8}{'disch':>7}")
Z2 = np.load(HERE + "\\hgc_curves.npz")
FFH = {k[:-3]: float(Z2[k][0]) for k in Z2.files if k.endswith(":ff")}
for lab, lib, a, b, f in CAND:
    r = evalpair(HCUR[f"{lib}:{a}"], HCUR[f"{lib}:{b}"], FFH[f"{lib}:{b}"], f)
    print(f"{lab:>44}{r['raw']:>7.1f}{r['cy']:>7.1f}{r['cbc']:>6.0f}"
          f"{r['ff']:>8.4f}{r['con']:>+9.4f}{r['hump']:>7.4f}{r['npk']:>6.3f}"
          f"{r['flr']:>8.3f}{r['dis']:>7.1f}")

NEW = [
    ("N1 @f121", (5.50, 8, 20, "2:2;4:1;6:3", "PB", "z1"),
     (5.00, 10, 20, "1:1;4:1;6:4", "PB", "z1"), 121),
    ("N2 @f121", (5.30, 6, 20, "2:0;2:2;5:1", "PB", "z1"),
     (5.05, 8, 20, "1:1;4:1;6:4", "PB", "z1"), 121),
    ("N3 @f121", (5.50, 6, 20, "2:2;5:1;6:3", "PB", "z1"),
     (5.05, 10, 20, "1:1;4:1;6:4", "PB", "z1"), 121),
    ("N4 @f117", (5.50, 6, 20, "1:1;3:1;6:4", "PB", "z1"),
     (5.15, 10, 20, "1:1;4:1;6:4", "PB", "z1"), 117),
    ("N5 @f117", (5.50, 6, 20, "2:0;4:1;5:5", "PB", "z1"),
     (5.15, 8, 20, "1:1;4:1;6:4", "PB", "z1"), 117),
]
print("\n" + "=" * 104)
print("NEW LATTICES (needs 1 DeCART wave + library rebuild + full re-bootstrap)")
print("=" * 104)
print(f"{'cand':>9}{'raw':>7}{'cy':>7}{'CBC':>6}{'FFhot':>8}{'FFcold':>8}"
      f"{'contr':>9}{'hump':>7}{'npk':>6}{'Fr_flr':>8}{'dFF':>8}{'dFr':>7}")
for lab, A, B, f in NEW:
    ia, ib = row_of(*A), row_of(*B)
    r = evalpair(M.Curve(bu0, K[ia]), M.Curve(bu0, K[ib]), FFM[ib], f)
    print(f"{lab:>9}{r['raw']:>7.1f}{r['cy']:>7.1f}{r['cbc']:>6.0f}"
          f"{r['ff']:>8.4f}{FFM[ia]:>8.4f}{r['con']:>+9.4f}{r['hump']:>7.4f}"
          f"{r['npk']:>6.3f}{r['flr']:>8.3f}"
          f"{r['ff']-1.1409:>+8.4f}{1.03*1.2085*(r['ff']-1.1409):>+7.3f}")
    print(f"          68-role: u_high {A[0]:.2f} / u_low "
          f"{df.u_low.values[ia]:.4f}  gd_wt {A[1]}  n_gd {A[2]}  "
          f"{A[3]}  {A[4]}/{A[5]}   (FF {FFM[ia]:.4f})")
    print(f"          53-role: u_high {B[0]:.2f} / u_low "
          f"{df.u_low.values[ib]:.4f}  gd_wt {B[1]}  n_gd {B[2]}  "
          f"{B[3]}  {B[4]}/{B[5]}   (FF {FFM[ib]:.4f})  <-- hot")

print("\n(dFF / dFr are versus the best REALIZED hot type, T4 FF_ens 1.1409;")
print(" Fr_flr = 1.03 * 1.2085 * FF_hot, the chosen.json headline convention)")
