"""Step 11: every ALREADY-REALIZED paramA pairing (37 types, 1369 ordered pairs)
at the real operating point.  Zero DeCART cost -- these are already in MAS_XSL.
"""
import numpy as np
import os
import itertools
import opmodel as M
import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))
cal = np.load(HERE + "\\calib_ow.npz")
RS, WB = float(cal["rho_star"]), float(cal["w_b"])
Z = np.load(HERE + "\\hgc_curves.npz")
TYPES = sorted(k[7:-3] for k in Z.files
               if k.startswith("paramA:") and k.endswith(":bu"))
CUR = {n: M.Curve(Z[f"paramA:{n}:bu"], Z[f"paramA:{n}:k"], n) for n in TYPES}
FF = {n: float(Z[f"paramA:{n}:ff"][0]) for n in TYPES}
des = P.load_designs()

TS = np.linspace(0.5, 8.0, 12)
RM = {n: float(np.mean(CUR[n](TS))) for n in TYPES}

CY_LO, CY_HI = 620.0, 645.0
FA = 68.0 / 121.0


def op(a, b, feed):
    cs = [CUR[a]] if a == b else [CUR[a], CUR[b]]
    w = [1.0] if a == b else [68.0, 53.0]
    rm = M.mix(cs, w)
    cy, bc = M.cyclen(rm, feed, RS, "EOC")
    rpk, _ = M.rho_op_peak(rm, feed, bc)
    return cy, (rpk - RS) / WB


rows = []
for a, b in itertools.product(TYPES, repeat=2):
    for feed in (121, 117):
        cy, cbc = op(a, b, feed)
        rows.append((a, b, feed, cy, cbc, FF[b], FF[a], RM[a] - RM[b]))

print("=" * 96)
print("A. the four lat1600 types in EVERY pairing (16 ordered pairs x 2 feeds)")
print("=" * 96)
print(f"{'E1(68)':>7}{'E2hot(53)':>10}{'feed':>6}{'cyclen':>9}{'CBC':>8}"
      f"{'FF_hot':>8}{'contrast':>10}  verdict")
T = ("T3", "T4", "T5", "T6")
for a, b in itertools.product(T, repeat=2):
    for feed in (121, 117):
        cy, cbc = op(a, b, feed)
        v = []
        v.append("CY-OK" if CY_LO <= cy <= CY_HI else
                 ("CY-LOW" if cy < CY_LO else "CY-HIGH"))
        v.append("CBC-OK" if cbc <= 1500 else
                 ("CBC-MARG" if cbc <= 1600 else "CBC-FAIL"))
        v.append("CON-OK" if RM[a] - RM[b] >= 0.026 else "CON-LOW")
        print(f"{a:>7}{b:>10}{feed:>6}{cy:>9.1f}{cbc:>8.0f}{FF[b]:>8.4f}"
              f"{RM[a]-RM[b]:>+10.5f}  {' '.join(v)}")

print("\n" + "=" * 96)
print("B. ALL 37 realized paramA types: pairs inside the 620-645 EFPD window")
print("=" * 96)
inw = [r for r in rows if CY_LO <= r[3] <= CY_HI]
print(f"  {len(inw)} of {len(rows)} ordered (pair, feed) combinations in window")
for cap in (1600, 1550, 1500):
    s = [r for r in inw if r[4] <= cap]
    print(f"  CBC<={cap}: {len(s):>4}   min FF_hot "
          f"{min((r[5] for r in s), default=float('nan')):.4f}")
    s2 = [r for r in s if r[7] >= 0.026]
    print(f"       + contrast>=0.026: {len(s2):>4}   min FF_hot "
          f"{min((r[5] for r in s2), default=float('nan')):.4f}")

print("\n  --- all in-window realized pairs with CBC <= 1600, by FF_hot ---")
s = sorted((r for r in inw if r[4] <= 1600), key=lambda r: r[5])
print(f"{'E1(68)':>7}{'E2hot(53)':>10}{'feed':>6}{'cyclen':>9}{'CBC':>8}"
      f"{'FF_hot':>8}{'FF_cold':>9}{'contrast':>10}")
for r in s[:40]:
    print(f"{r[0]:>7}{r[1]:>10}{r[2]:>6}{r[3]:>9.1f}{r[4]:>8.0f}{r[5]:>8.4f}"
          f"{r[6]:>9.4f}{r[7]:>+10.5f}")
if not s:
    print("   (none)")

print("\n  --- best realized pairs by FF_hot with CBC <= 1500 ---")
s = sorted((r for r in inw if r[4] <= 1500), key=lambda r: r[5])
for r in s[:25]:
    print(f"{r[0]:>7}{r[1]:>10}{r[2]:>6}{r[3]:>9.1f}{r[4]:>8.0f}{r[5]:>8.4f}"
          f"{r[6]:>9.4f}{r[7]:>+10.5f}")
if not s:
    print("   (none)")

print("\n" + "=" * 96)
print("C. per-type solo operating points (all 37), sorted by f121 cyclen")
print("=" * 96)
solo = []
for n in TYPES:
    c1, b1 = op(n, n, 121)
    c2, b2 = op(n, n, 117)
    d = des[n]
    solo.append((n, d["e1"], d["gd_wt"], d["n_gd"], FF[n], c1, b1, c2, b2))
print(f"{'type':>5}{'e1':>6}{'gdwt':>5}{'ngd':>4}{'FF':>8}"
      f"{'cy121':>8}{'cbc121':>8}{'cy117':>8}{'cbc117':>8}")
for t in sorted(solo, key=lambda x: x[5]):
    print(f"{t[0]:>5}{t[1]:>6.2f}{t[2]:>5.0f}{t[3]:>4}{t[4]:>8.4f}"
          f"{t[5]:>8.1f}{t[6]:>8.0f}{t[7]:>8.1f}{t[8]:>8.0f}")
