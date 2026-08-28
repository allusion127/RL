"""Step 15: rank the surviving operating points and name the winners."""
import numpy as np
import os
import pandas as pd
import paths as P
import opmodel as M

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(HERE + "\\calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
fm = np.load(HERE + "\\frmodel2.npz")
NPC, AFUS = fm["c"], float(fm["A"])       # node_peak = c0 + c1*contrast + c2*dfresh

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
bu0, K = z["bu"], z["kconv"]
keep = bu0 >= 0.2
BU = bu0[keep]
R = (1.0 - 1.0 / K)[:, keep]
FA = 68.0 / 121.0
REAL = {398: "T3", 324: "T4", 2712: "T5", 3246: "T6"}

CBC_GATE = 1500.0        # 1600 program gate minus 100 ppm pattern/model headroom
CON_MIN = 0.026          # measured node_peak stays <=1.33 above this


def enrich(d):
    i, j = d["i"], d["j"]
    rf = FA * R[i, 0] + (1 - FA) * R[j, 0]            # post-Xe fresh mix rho
    rboc = RS + d["cbc"] * WB
    dfr = rf - rboc
    npk = NPC[0] + NPC[1] * d["contrast"] + NPC[2] * dfr
    d["dfresh"] = dfr
    d["node_peak"] = npk
    d["fr_fixed"] = AFUS * npk * d["ff_hot"]
    d["fr_floor"] = 1.03 * 1.2085 * d["ff_hot"]
    d["n_new"] = (~np.isin(i, list(REAL))).astype(int) + \
                 (~np.isin(j, list(REAL))).astype(int)
    return d


def spec(k):
    r = df.iloc[k]
    tag = REAL.get(k, "")
    return (f"{r.u_high:.2f}/{r.u_low:.4f} g{r.gd_wt:.0f}n{r.n_gd:.0f} "
            f"{r.gd_positions:<15} {r.pattern}/{r.zoning}"
            + (f"  [={tag}]" if tag else ""))


def show(d, mask, n, key, title, rev=False):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        print(f"\n{title}\n   (none)")
        return []
    o = idx[np.argsort(-d[key][idx] if rev else d[key][idx])]
    # de-duplicate on the hot type so the table is not 20 rows of one design
    seen, pick = set(), []
    for t in o:
        if d["j"][t] in seen:
            continue
        seen.add(d["j"][t])
        pick.append(t)
        if len(pick) >= n:
            break
    print(f"\n{title}")
    print(f"{'#':>2} {'cyclen':>7}{'CBC':>6}{'FFhot':>7}{'contr':>8}{'npk':>6}"
          f"{'Fr_fix':>7}{'Fr_flr':>7}{'new':>4}  E1-role(68) -> E2-role(53 hot)")
    for q, t in enumerate(pick, 1):
        print(f"{q:>2} {d['cy'][t]:>7.1f}{d['cbc'][t]:>6.0f}{d['ff_hot'][t]:>7.4f}"
              f"{d['contrast'][t]:>+8.4f}{d['node_peak'][t]:>6.3f}"
              f"{d['fr_fixed'][t]:>7.3f}{d['fr_floor'][t]:>7.3f}"
              f"{d['n_new'][t]:>4}")
        print(f"      A: {spec(d['i'][t])}")
        print(f"      B: {spec(d['j'][t])}")
    return pick


for feed in (121, 117):
    d = dict(np.load(HERE + f"\\screen_final_{feed}.npz"))
    d = enrich(d)
    print("=" * 92)
    print(f"FEED {feed}   ({len(d['i']):,} pairs in the 620-645 EFPD window)")
    print("=" * 92)

    base = d["cbc"] <= CBC_GATE
    show(d, base, 5, "ff_hot",
         f"[P1] CBC<={CBC_GATE:.0f}, minimise FF_hot  (NO contrast constraint "
         f"-- the T5_T6 failure mode)")
    show(d, base & (d["contrast"] >= CON_MIN), 5, "ff_hot",
         f"[P2] CBC<={CBC_GATE:.0f} AND contrast>={CON_MIN}, minimise FF_hot"
         f"   <-- the defensible screen")
    show(d, base & (d["contrast"] >= CON_MIN), 5, "fr_fixed",
         f"[P3] CBC<={CBC_GATE:.0f} AND contrast>={CON_MIN}, minimise the"
         f" PREDICTED fixed-pattern F_r")
    m = base & (d["contrast"] >= CON_MIN) & (d["n_new"] <= 1)
    show(d, m, 5, "ff_hot",
         "[P4] same, but at most ONE new lattice (the other is T3/T4/T5/T6)")
    m0 = base & (d["n_new"] == 0)
    show(d, m0, 5, "ff_hot", "[P5] ZERO new lattices (T3-T6 pairings only)")

    print("\n  --- Pareto front FF_hot vs contrast (CBC<=%.0f) ---" % CBC_GATE)
    sub = np.flatnonzero(base)
    o = sub[np.argsort(d["ff_hot"][sub])]
    best_con, front = -9.9, []
    for t in o:
        if d["contrast"][t] > best_con:
            best_con = d["contrast"][t]
            front.append(t)
    print(f"{'FF_hot':>8}{'contrast':>10}{'cyclen':>8}{'CBC':>6}{'npk':>6}"
          f"{'Fr_fix':>7}{'new':>4}")
    for t in front[::max(1, len(front) // 14)]:
        print(f"{d['ff_hot'][t]:>8.4f}{d['contrast'][t]:>+10.4f}"
              f"{d['cy'][t]:>8.1f}{d['cbc'][t]:>6.0f}{d['node_peak'][t]:>6.3f}"
              f"{d['fr_fixed'][t]:>7.3f}{d['n_new'][t]:>4}")
    print()
