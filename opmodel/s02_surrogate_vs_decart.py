"""Step 2: how far is the lat1600 SURROGATE k(BU) from the DeCART ground truth?

T3-T6 are a genuine held-out test: the surrogate screen picked them, then DeCART
computed them.  Whatever bias shows up here is the bias the re-screen must carry
when it scores the other 5,870 designs (which have surrogate k only).
"""
import numpy as np
import pandas as pd
import paths as P

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
sbu, sK = z["bu"], z["kconv"]
dec = P.load_decart(["T3", "T4", "T5", "T6"])
ROW = {"T3": 398, "T4": 324, "T5": 2712, "T6": 3246}

dbu = dec["T3"][0]
print("DeCART bu grid (n=%d):" % len(dbu), np.round(dbu, 2))
print("surrogate bu grid (n=%d):" % len(sbu), np.round(sbu, 2))

print("\n=== k_inf(BU): surrogate vs DeCART ===")
print(f"{'BU':>7}" + "".join(f"{n:>26}" for n in ("T3", "T4", "T5", "T6")))
print(f"{'':>7}" + "".join(f"{'sur':>9}{'dec':>9}{'d_pcm':>8}"
                           for _ in range(4)))
grid = [0, 0.5, 1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60]
tab = {}
for b in grid:
    line = f"{b:7.1f}"
    for n in ("T3", "T4", "T5", "T6"):
        s = float(np.interp(b, sbu, sK[ROW[n]]))
        d = float(np.interp(b, dec[n][0], dec[n][1]))
        rs, rd = 1 - 1 / s, 1 - 1 / d
        tab.setdefault(n, []).append((b, s, d, (rs - rd) * 1e5))
        line += f"{s:9.4f}{d:9.4f}{(rs-rd)*1e5:+8.0f}"
    print(line)

print("\n=== rho bias (surrogate - DeCART), pcm, by burnup band ===")
for n in ("T3", "T4", "T5", "T6"):
    a = np.array(tab[n])
    lo = a[(a[:, 0] >= 0) & (a[:, 0] <= 10), 3]
    mid = a[(a[:, 0] > 10) & (a[:, 0] <= 30), 3]
    hi = a[a[:, 0] > 30, 3]
    print(f"  {n}: BU0-10 {lo.mean():+7.0f}  BU10-30 {mid.mean():+7.0f}  "
          f"BU30+ {hi.mean():+7.0f}  |  all {a[:,3].mean():+7.0f} pcm")

# the quantity the screen actually used
print("\n=== screen-convention statistics: surrogate vs DeCART ===")
BC = 24.7327


def stat(bu, k):
    rho = 1 - 1 / k
    at = lambda x: np.interp(x, bu, rho)
    rbar_eoc = (at(BC) + at(2 * BC) + at(3 * BC)) / 3.0
    rp = max((at(t) + at(t + BC) + at(t + 2 * BC)) / 3.0
             for t in np.linspace(0, BC, 41))
    return rbar_eoc, rp, 26176.0 * rp + 133.0


print(f"{'':>4}{'rbar_eoc_sur':>14}{'rbar_eoc_dec':>14}{'d':>9}"
      f"{'rpk_sur':>10}{'rpk_dec':>10}{'cbc_sur':>9}{'cbc_dec':>9}")
for n in ("T3", "T4", "T5", "T6"):
    a = stat(sbu, sK[ROW[n]])
    b = stat(dec[n][0], dec[n][1])
    print(f"{n:>4}{a[0]:>14.5f}{b[0]:>14.5f}{a[0]-b[0]:>+9.5f}"
          f"{a[1]:>10.5f}{b[1]:>10.5f}{a[2]:>9.0f}{b[2]:>9.0f}")

# --- the single number the cyclen model consumes: all-fresh critical burnup ---
print("\n=== what the bias costs in B1 (all-fresh critical burnup) ===")
W = P.W_CY1
for RS in (0.0168,):
    for pair in (("T3", "T4"), ("T5", "T6")):
        for lab, getter in (("surrogate", lambda n: (sbu, sK[ROW[n]])),
                            ("DeCART", lambda n: dec[n])):
            bu_a, k_a = getter(pair[0])
            bu_b, k_b = getter(pair[1])
            ra = 1 - 1 / k_a
            rb = 1 - 1 / k_b
            lo, hi = 1.0, 70.0
            for _ in range(90):
                m = 0.5 * (lo + hi)
                v = (W[0] * np.interp(m, bu_a, ra)
                     + W[1] * np.interp(m, bu_b, rb)) / 241.0
                lo, hi = (m, hi) if v > RS else (lo, m)
            print(f"  {pair[0]}_{pair[1]:>3} rho*={RS} {lab:>9}: "
                  f"B1={lo:6.2f} MWd/kgHM = {lo/P.RATE:6.1f} EFPD")
print("\nobserved cy1 (MASTER bootstrap): T3_T4 894.09 EFPD (33.98), "
      "T5_T6 981.02 EFPD (37.28)")
