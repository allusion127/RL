"""Step 19: ALL 37 realized paramA types, all 1369 ordered pairs, both feeds,
with the FINAL (hump-corrected) model.  This is the zero-DeCART-cost answer."""
import numpy as np
import os
import itertools
import opmodel as M
import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(HERE + "\\calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
HA, HB = float(c["hump_a"]), float(c["hump_b"])
fm = np.load(HERE + "\\frmodel2.npz")
NPC, AFUS = fm["c"], float(fm["A"])
HCAL = 0.0148

Z = np.load(HERE + "\\hgc_curves.npz")
TYPES = sorted(k[7:-3] for k in Z.files
               if k.startswith("paramA:") and k.endswith(":bu"))
CUR = {n: M.Curve(Z[f"paramA:{n}:bu"], Z[f"paramA:{n}:k"], n) for n in TYPES}
FF = {n: float(Z[f"paramA:{n}:ff"][0]) for n in TYPES}
TH = np.linspace(0.5, 12.0, 24)
TC = np.linspace(0.5, 8.0, 12)
CON_MIN, GATE = 0.026, 1500.0

rows = []
for a, b in itertools.product(TYPES, repeat=2):
    cs = [CUR[a]] if a == b else [CUR[a], CUR[b]]
    w = [1.0] if a == b else [68.0, 53.0]
    rm = M.mix(cs, w)
    hump = float(np.max(rm(TH)) - rm(0.5))
    con = float(np.mean(CUR[a](TC) - CUR[b](TC)))
    for feed in (121, 117):
        raw, _ = M.cyclen(rm, feed, RS, "EOC")
        cy = raw * (1 + HA + HB * hump)
        bc = cy * M.RATE
        rpk, _ = M.rho_op_peak(rm, feed, bc)
        cbc = (rpk - RS) / WB
        rboc = M.rho_op(rm, feed, bc, 0.0)
        npk = NPC[0] + NPC[1] * con + NPC[2] * (float(rm(0.5)) - rboc)
        rows.append(dict(a=a, b=b, feed=feed, raw=raw, cy=cy, cbc=cbc,
                         ff=FF[b], con=con, hump=hump, npk=npk,
                         flr=1.03 * 1.2085 * FF[b]))

inw = [r for r in rows if 620 <= r["cy"] <= 645]
print(f"{len(inw)} of {len(rows)} (ordered pair, feed) combos in the "
      f"620-645 EFPD window")
for cap in (1600, 1500):
    s = [r for r in inw if r["cbc"] <= cap]
    s2 = [r for r in s if r["con"] >= CON_MIN]
    s3 = [r for r in s2 if r["hump"] <= HCAL]
    print(f"  CBC<={cap}: {len(s):>3}  |  +contrast>={CON_MIN}: {len(s2):>3} "
          f"(min FF_hot {min((r['ff'] for r in s2), default=np.nan):.4f})"
          f"  |  +no hump extrapolation: {len(s3):>3} "
          f"(min FF_hot {min((r['ff'] for r in s3), default=np.nan):.4f})")

print("\n=== every realized (pair, feed) passing cyclen + CBC<=1500 + "
      "contrast>=0.026, by FF_hot ===")
s = sorted((r for r in inw if r["cbc"] <= GATE and r["con"] >= CON_MIN),
           key=lambda r: r["ff"])
print(f"{'68':>4}{'53hot':>7}{'feed':>6}{'cy_raw':>8}{'cy':>7}{'CBC':>6}"
      f"{'FFhot':>8}{'contr':>9}{'hump':>8}{'npk':>6}{'Fr_flr':>8}{'X':>2}")
for r in s:
    print(f"{r['a']:>4}{r['b']:>7}{r['feed']:>6}{r['raw']:>8.1f}{r['cy']:>7.1f}"
          f"{r['cbc']:>6.0f}{r['ff']:>8.4f}{r['con']:>+9.4f}{r['hump']:>8.4f}"
          f"{r['npk']:>6.3f}{r['flr']:>8.3f}"
          f"{'*' if r['hump'] > HCAL else ' ':>2}")

print("\n=== same but CBC<=1600 (the raw program gate, no pattern headroom) ===")
s = sorted((r for r in inw if r["cbc"] <= 1600 and r["con"] >= CON_MIN),
           key=lambda r: r["ff"])
for r in s[:20]:
    print(f"{r['a']:>4}{r['b']:>7}{r['feed']:>6}{r['raw']:>8.1f}{r['cy']:>7.1f}"
          f"{r['cbc']:>6.0f}{r['ff']:>8.4f}{r['con']:>+9.4f}{r['hump']:>8.4f}"
          f"{r['npk']:>6.3f}{r['flr']:>8.3f}"
          f"{'*' if r['hump'] > HCAL else ' ':>2}")

print("\n=== the SAFE subset: no hump extrapolation, CBC<=1500, contrast>=0.026")
s = sorted((r for r in inw if r["cbc"] <= GATE and r["con"] >= CON_MIN
            and r["hump"] <= HCAL), key=lambda r: r["ff"])
for r in s:
    print(f"{r['a']:>4}{r['b']:>7}{r['feed']:>6}{r['raw']:>8.1f}{r['cy']:>7.1f}"
          f"{r['cbc']:>6.0f}{r['ff']:>8.4f}{r['con']:>+9.4f}{r['hump']:>8.4f}"
          f"{r['npk']:>6.3f}{r['flr']:>8.3f}")
if not s:
    print("   (none -- every in-window realized pair with usable contrast "
          "relies on a hump-corrected cycle length)")
