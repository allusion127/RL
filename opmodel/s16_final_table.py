"""Step 16: final candidate table, with the hump-correction EXTRAPOLATION flagged.

The hump correction was fitted over mixture humps 0.0000-0.0148 (the range the 55
measured operating points span).  Any candidate whose mixture hump exceeds that
is extrapolating; for those the honest cycle length is the interval
[raw, corrected], not the corrected number.
"""
import numpy as np
import os
import itertools
import pandas as pd
import paths as P
import opmodel as M

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(HERE + "\\calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
HA, HB = float(c["hump_a"]), float(c["hump_b"])
fm = np.load(HERE + "\\frmodel2.npz")
NPC, AFUS = fm["c"], float(fm["A"])
HUMP_CAL_MAX = 0.0148

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
bu0, K = z["bu"], z["kconv"]
keep = bu0 >= 0.2
BU = bu0[keep]
R = (1.0 - 1.0 / K)[:, keep]
FA = 68.0 / 121.0
REAL = {398: "T3", 324: "T4", 2712: "T5", 3246: "T6"}
CBC_GATE, CON_MIN = 1500.0, 0.026


def spec(k):
    r = df.iloc[k]
    t = REAL.get(k, "")
    return (f"u{r.u_high:.2f}/{r.u_low:.4f} gd{r.gd_wt:.0f}x{r.n_gd:.0f} "
            f"{r.gd_positions} {r.pattern}/{r.zoning}" + (f" [{t}]" if t else ""))


def enrich(d):
    i, j = d["i"], d["j"]
    d["cy_raw"] = d["cy"] / (1.0 + HA + HB * d["hump"])
    rf = FA * R[i, 0] + (1 - FA) * R[j, 0]
    rboc = RS + d["cbc"] * WB
    d["node_peak"] = NPC[0] + NPC[1] * d["contrast"] + NPC[2] * (rf - rboc)
    d["fr_fixed"] = AFUS * d["node_peak"] * d["ff_hot"]
    d["fr_floor"] = 1.03 * 1.2085 * d["ff_hot"]
    d["n_new"] = ((~np.isin(i, list(REAL))).astype(int)
                  + (~np.isin(j, list(REAL))).astype(int))
    d["extrap"] = d["hump"] > HUMP_CAL_MAX
    return d


HDR = (f"{'cyc':>6}{'(raw)':>7}{'CBC':>6}{'FFhot':>7}{'contr':>8}{'hump':>7}"
       f"{'npk':>6}{'Fr_fix':>7}{'Fr_flr':>7}{'new':>4}{'X':>2}")


def line(d, t):
    return (f"{d['cy'][t]:>6.1f}{d['cy_raw'][t]:>7.1f}{d['cbc'][t]:>6.0f}"
            f"{d['ff_hot'][t]:>7.4f}{d['contrast'][t]:>+8.4f}{d['hump'][t]:>7.4f}"
            f"{d['node_peak'][t]:>6.3f}{d['fr_fixed'][t]:>7.3f}"
            f"{d['fr_floor'][t]:>7.3f}{d['n_new'][t]:>4}"
            f"{'*' if d['extrap'][t] else ' ':>2}")


def show(d, mask, n, key, title):
    idx = np.flatnonzero(mask)
    print(f"\n{title}")
    if not len(idx):
        print("   (none)")
        return
    o = idx[np.argsort(d[key][idx])]
    seen, pick = set(), []
    for t in o:
        if d["j"][t] in seen:
            continue
        seen.add(d["j"][t])
        pick.append(t)
        if len(pick) >= n:
            break
    print(" # " + HDR)
    for q, t in enumerate(pick, 1):
        print(f"{q:>2} " + line(d, t))
        print(f"     68: {spec(d['i'][t])}")
        print(f"     53: {spec(d['j'][t])}")


for feed in (121, 117):
    d = enrich(dict(np.load(HERE + f"\\screen_final_{feed}.npz")))
    print("=" * 100)
    print(f"FEED {feed}  ({len(d['i']):,} pairs in window)   "
          f"X = hump>{HUMP_CAL_MAX} => the correction extrapolates, "
          f"true cyclen in [raw, cyc]")
    print("=" * 100)
    g = d["cbc"] <= CBC_GATE
    cc = d["contrast"] >= CON_MIN
    safe = ~d["extrap"]
    show(d, g & cc, 5, "ff_hot",
         "[A] CBC<=1500, contrast>=0.026, min FF_hot -- 2 new lattices allowed")
    show(d, g & cc & safe, 5, "ff_hot",
         "[B] same, and NO hump extrapolation (correction inside its "
         "calibration range)")
    show(d, g & cc & (d["n_new"] <= 1), 4, "ff_hot",
         "[C] at most ONE new lattice")
    show(d, g & (d["n_new"] == 0), 4, "ff_hot",
         "[D] ZERO new lattices (T3-T6 only; contrast NOT constrained)")

print("\n" + "=" * 100)
print("E. ALL 16 ordered T3-T6 pairings, hump-corrected, both feeds")
print("=" * 100)
IDX = {"T3": 398, "T4": 324, "T5": 2712, "T6": 3246}
CUR = {n: M.Curve(bu0, K[IDX[n]], n) for n in IDX}
TH = np.linspace(0.5, 12.0, 24)
TC = np.linspace(0.5, 8.0, 12)
print(f"{'68':>4}{'53hot':>7}{'feed':>6}{'cy_raw':>8}{'cy_cor':>8}{'CBC':>7}"
      f"{'FFhot':>8}{'contr':>9}{'hump':>8}{'npk':>6}{'Fr_flr':>8}  verdict")
for a, b in itertools.product(("T3", "T4", "T5", "T6"), repeat=2):
    cs = [CUR[a]] if a == b else [CUR[a], CUR[b]]
    w = [1.0] if a == b else [68.0, 53.0]
    rm = M.mix(cs, w)
    cyr, bc = M.cyclen(rm, 121, RS, "EOC")
    hump = float(np.max(rm(TH)) - rm(0.5))
    for feed in (121, 117):
        cyr, _ = M.cyclen(rm, feed, RS, "EOC")
        cyc = cyr * (1 + HA + HB * hump)
        bcc = cyc * M.RATE
        rb = M.rho_op(rm, feed, bcc, 0.0)
        cbc = (rb - RS) / WB
        con = float(np.mean(CUR[a](TC) - CUR[b](TC)))
        ff = float(df.ff_max.values[IDX[b]])
        npk = NPC[0] + NPC[1] * con + NPC[2] * (float(rm(0.5)) - rb)
        v = ("CY-OK" if 620 <= cyc <= 645 else
             ("CY-LOW" if cyc < 620 else "CY-HIGH"))
        v += " CBC-OK" if cbc <= 1500 else (
            " CBC-MARG" if cbc <= 1600 else " CBC-FAIL")
        v += " CON-OK" if con >= CON_MIN else " CON-LOW"
        v += "  *extrap" if hump > HUMP_CAL_MAX else ""
        print(f"{a:>4}{b:>7}{feed:>6}{cyr:>8.1f}{cyc:>8.1f}{cbc:>7.0f}"
              f"{ff:>8.4f}{con:>+9.4f}{hump:>8.4f}{npk:>6.3f}"
              f"{1.03*1.2085*ff:>8.3f}  {v}")
