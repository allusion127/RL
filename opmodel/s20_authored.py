"""Step 20: the ZERO-TEMPLATE-AUTHORING subspace.

decks/MANIFEST.md: three dec_FA decks are already hand-authored and verified --
  (gd_wt 10, n_gd 16, `1:1;5:2;5:5`)  <- Y1/T3   base 5.8_5.1/FA/IGD_16/10_16_z1
  (gd_wt  6, n_gd 16, `1:1;5:2;5:5`)  <- Y3/T5, Y4/T6   base .../6_16_z1
  (gd_wt  8, n_gd 24, `1:1;4:1;5:5;6:3`) <- Y2/T4  base 260624/FA/IGD_24/8_24_z1
The chain rewrites e1/e2 at realization time (lattice.py edit_dec_text), so ANY
u_high on the 0.05 grid can be realized off these three decks with no pin-map
work at all -- only a DeCART wave + the library rebuild.
"""
import numpy as np
import os
import pandas as pd
import paths as P
import opmodel as M

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(HERE + "\\calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
HA, HB = float(c["hump_a"]), float(c["hump_b"])
fm = np.load(HERE + "\\frmodel2.npz")
NPC, AFUS = fm["c"], float(fm["A"])
HCAL = 0.0148

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
bu0, K = z["bu"], z["kconv"]

DECKS = [(10.0, 16, "1:1;5:2;5:5", "D16g10 (Y1/T3 deck)"),
         (6.0, 16, "1:1;5:2;5:5", "D16g6  (Y3/T5,Y4/T6 deck)"),
         (8.0, 24, "1:1;4:1;5:5;6:3", "D24g8  (Y2/T4 deck)")]
sel = []
for gw, ng, gp, lab in DECKS:
    m = (np.isclose(df.gd_wt, gw) & (df.n_gd == ng)
         & (df.gd_positions == gp) & (df.pattern == "PB")
         & (df.zoning == "z1"))
    for k in np.flatnonzero(m.values):
        sel.append((k, lab, float(df.u_high.values[k])))
sel.sort(key=lambda t: (t[1], t[2]))
print(f"{len(sel)} realizable-with-zero-template-work designs")

REAL = {398: "T3", 324: "T4", 2712: "T5", 3246: "T6"}
CUR = {k: M.Curve(bu0, K[k]) for k, _, _ in sel}
FFM = df.ff_max.values
TH = np.linspace(0.5, 12.0, 24)
TC = np.linspace(0.5, 8.0, 12)
print(f"{'deck':>26}{'u_high':>8}{'FF':>8}{'alias':>7}")
for k, lab, uh in sel:
    print(f"{lab:>26}{uh:>8.2f}{FFM[k]:>8.4f}{REAL.get(k,''):>7}")

rows = []
for ki, li, ui in sel:
    for kj, lj, uj in sel:
        cs = [CUR[ki]] if ki == kj else [CUR[ki], CUR[kj]]
        w = [1.0] if ki == kj else [68.0, 53.0]
        rm = M.mix(cs, w)
        hump = float(np.max(rm(TH)) - rm(0.5))
        con = float(np.mean(CUR[ki](TC) - CUR[kj](TC)))
        for feed in (121, 117):
            raw, _ = M.cyclen(rm, feed, RS, "EOC")
            cy = raw * (1 + HA + HB * hump)
            bc = cy * M.RATE
            rpk, _ = M.rho_op_peak(rm, feed, bc)
            cbc = (rpk - RS) / WB
            npk = NPC[0] + NPC[1] * con + NPC[2] * (
                float(rm(0.5)) - M.rho_op(rm, feed, bc, 0.0))
            nnew = int(ki not in REAL) + int(kj not in REAL)
            rows.append(dict(i=ki, j=kj, li=li, lj=lj, ui=ui, uj=uj, feed=feed,
                             raw=raw, cy=cy, cbc=cbc, ff=FFM[kj], con=con,
                             hump=hump, npk=npk, nnew=nnew,
                             flr=1.03 * 1.2085 * FFM[kj]))

for gate, cmin in ((1500.0, 0.026), (1600.0, 0.026)):
    s = [r for r in rows if 620 <= r["cy"] <= 645 and r["cbc"] <= gate
         and r["con"] >= cmin]
    s.sort(key=lambda r: (r["ff"], r["nnew"]))
    print(f"\n=== authored-deck pairs: cyclen 620-645, CBC<={gate:.0f}, "
          f"contrast>={cmin} -> {len(s)} ===")
    print(f"{'68 deck':>26}{'u':>6} {'53hot deck':>26}{'u':>6}{'feed':>5}"
          f"{'raw':>7}{'cy':>7}{'CBC':>6}{'FFhot':>8}{'con':>8}{'hump':>7}"
          f"{'npk':>6}{'Fr_flr':>7}{'new':>4}{'X':>2}")
    for r in s[:24]:
        print(f"{r['li']:>26}{r['ui']:>6.2f} {r['lj']:>26}{r['uj']:>6.2f}"
              f"{r['feed']:>5}{r['raw']:>7.1f}{r['cy']:>7.1f}{r['cbc']:>6.0f}"
              f"{r['ff']:>8.4f}{r['con']:>+8.4f}{r['hump']:>7.4f}{r['npk']:>6.3f}"
              f"{r['flr']:>7.3f}{r['nnew']:>4}"
              f"{'*' if r['hump'] > HCAL else ' ':>2}")

print("\n=== SAFEST authored-deck pairs (raw cyclen ALREADY in window, so the")
print("    hump correction is not load-bearing), CBC<=1500, contrast>=0.026 ===")
s = [r for r in rows if 620 <= r["raw"] <= 645 and 620 <= r["cy"] <= 645
     and r["cbc"] <= 1500 and r["con"] >= 0.026]
s.sort(key=lambda r: r["ff"])
print(f"{'68 deck':>26}{'u':>6} {'53hot deck':>26}{'u':>6}{'feed':>5}"
      f"{'raw':>7}{'cy':>7}{'CBC':>6}{'FFhot':>8}{'con':>8}{'new':>4}")
for r in s[:20]:
    print(f"{r['li']:>26}{r['ui']:>6.2f} {r['lj']:>26}{r['uj']:>6.2f}"
          f"{r['feed']:>5}{r['raw']:>7.1f}{r['cy']:>7.1f}{r['cbc']:>6.0f}"
          f"{r['ff']:>8.4f}{r['con']:>+8.4f}{r['nnew']:>4}")
if not s:
    print("   (none)")
