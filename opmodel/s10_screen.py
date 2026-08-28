"""Step 10: re-screen the FULL 5,874-design surrogate table against the REAL
operating point (equilibrium at feed 121 and 117), with the calibrated model.

Everything is vectorised over the design table:
  * rho matrix R (5874 x 61), the BU=0 Xe-free point dropped
  * solo equilibrium Bc for every design, by vectorised bisection
  * pair Bc estimated by the slope-weighted mean of the two solo Bc's, then
    EXACTLY re-solved for every pair that survives the estimate's band
"""
import numpy as np
import os
import time
import pandas as pd
import paths as P
import opmodel as M

HERE = os.path.dirname(os.path.abspath(__file__))
cal = np.load(HERE + "\\calib_ow.npz")
RS, WB = float(cal["rho_star"]), float(cal["w_b"])
print(f"calibration: rho* = {RS:.5f}, w_B = {1e5*WB:.4f} pcm/ppm")

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
bu0, K = z["bu"], z["kconv"]
keep = bu0 >= 0.2
BU = bu0[keep]
R = (1.0 - 1.0 / K)[:, keep]
FFMAX = df.ff_max.values
N = len(df)
print(f"{N} designs, rho grid {BU.min()}..{BU.max()} ({len(BU)} pts)")

WA, WB_ROLE = 68.0, 53.0                 # E1-role / E2-role fresh slots at f121
FA = WA / (WA + WB_ROLE)


def rho_at(x):
    """R interpolated at a PER-ROW burnup x (vector length N) -> (N,)"""
    x = np.clip(x, BU[0], BU[-1])
    j = np.clip(np.searchsorted(BU, x), 1, len(BU) - 1)
    x0, x1 = BU[j - 1], BU[j]
    w = (x - x0) / (x1 - x0)
    i = np.arange(len(x))
    return R[i, j - 1] * (1 - w) + R[i, j] * w


def rho_at_rows(rows, x):
    x = np.clip(x, BU[0], BU[-1])
    j = np.clip(np.searchsorted(BU, x), 1, len(BU) - 1)
    x0, x1 = BU[j - 1], BU[j]
    w = (x - x0) / (x1 - x0)
    return R[rows, j - 1] * (1 - w) + R[rows, j] * w


def g_solo(bc, feed):
    """EOC core-average rho of an all-one-type equilibrium core, per design."""
    out = np.zeros_like(bc)
    for n, k in M.batch_weights(feed):
        out += n * rho_at((k + 1) * bc)
    return out / M.NSLOT


def h_solo(bc, feed, t=0.0):
    out = np.zeros_like(bc)
    for n, k in M.batch_weights(feed):
        out += n * rho_at(t + k * bc)
    return out / M.NSLOT


def solo_bc(feed, lo=6.0, hi=45.0):
    a = np.full(N, lo)
    b = np.full(N, hi)
    for _ in range(60):
        m = 0.5 * (a + b)
        v = g_solo(m, feed) > RS       # still supercritical -> go deeper
        a = np.where(v, m, a)
        b = np.where(v, b, m)
    return 0.5 * (a + b)


t0 = time.time()
BC = {f: solo_bc(f) for f in (121, 117)}
SL = {}
for f in (121, 117):
    d = 0.25
    SL[f] = (g_solo(BC[f] + d, f) - g_solo(BC[f] - d, f)) / (2 * d)  # dg/dBc < 0
print(f"solo equilibria in {time.time()-t0:.1f}s")
for f in (121, 117):
    cy = BC[f] / M.RATE
    print(f"  f{f}: solo cyclen {cy.min():.0f}..{cy.max():.0f} EFPD "
          f"(median {np.median(cy):.0f}); slope {SL[f].mean():.5f}/MWd")

# self-check of the vectorised path against the scalar opmodel (s08 reference)
ROW = {"T3": 398, "T4": 324, "T5": 2712, "T6": 3246}
REF = {("T3", 121): 590.8, ("T4", 121): 583.2, ("T5", 121): 628.1,
       ("T6", 121): 634.6, ("T5", 117): 608.8, ("T6", 117): 615.1}
print("  self-check solo cyclen (vectorised vs scalar opmodel/s08):")
for (n, f), ref in REF.items():
    got = BC[f][ROW[n]] / M.RATE
    flag = "OK" if abs(got - ref) < 0.15 else "**MISMATCH**"
    print(f"    {n}@f{f}: {got:8.2f} vs {ref:8.2f}  {flag}")

# --- contrast: mean rho difference over the first third of a cycle ----------
TSAMP = np.linspace(0.5, 8.0, 12)
RSAMP = np.empty((N, len(TSAMP)))
for q, t in enumerate(TSAMP):
    RSAMP[:, q] = np.interp(t, BU, np.zeros(len(BU)))  # placeholder
for q, t in enumerate(TSAMP):
    j = max(1, int(np.searchsorted(BU, t)))
    w = (t - BU[j - 1]) / (BU[j] - BU[j - 1])
    RSAMP[:, q] = R[:, j - 1] * (1 - w) + R[:, j] * w
RMEAN = RSAMP.mean(axis=1)               # low-BU reactivity level of each design

CY_LO, CY_HI = 620.0, 645.0
BCLO, BCHI = CY_LO * M.RATE, CY_HI * M.RATE
print(f"\ntarget cycle burnup window: {BCLO:.3f} .. {BCHI:.3f} MWd/kgHM")

RESULTS = []
for feed in (121, 117):
    bc, sl = BC[feed], SL[feed]
    # slope-weighted estimate of the pair root (exact if g were linear in Bc)
    wi = FA * sl
    wj = (1 - FA) * sl
    t0 = time.time()
    cand_i, cand_j = [], []
    CH = 256
    for s in range(0, N, CH):
        e = min(s + CH, N)
        num = wi[s:e, None] * bc[s:e, None] + wj[None, :] * bc[None, :]
        den = wi[s:e, None] + wj[None, :]
        est = num / den
        m = (est > BCLO - 0.45) & (est < BCHI + 0.45)
        ii, jj = np.nonzero(m)
        cand_i.append(ii + s)
        cand_j.append(jj)
    ci = np.concatenate(cand_i)
    cj = np.concatenate(cand_j)
    print(f"\nf{feed}: {len(ci):,} ordered pairs pass the ESTIMATE band "
          f"({time.time()-t0:.1f}s)")

    # ---- exact root for the survivors, vectorised bisection -----------------
    t0 = time.time()

    def gpair(x):
        out = np.zeros(len(ci))
        for n, k in M.batch_weights(feed):
            out += n * (FA * rho_at_rows(ci, (k + 1) * x)
                        + (1 - FA) * rho_at_rows(cj, (k + 1) * x))
        return out / M.NSLOT

    a = np.full(len(ci), 8.0)
    b = np.full(len(ci), 40.0)
    for _ in range(50):
        m = 0.5 * (a + b)
        v = gpair(m) > RS
        a = np.where(v, m, a)
        b = np.where(v, b, m)
    bcp = 0.5 * (a + b)
    ok = (bcp >= BCLO) & (bcp <= BCHI)
    ci, cj, bcp = ci[ok], cj[ok], bcp[ok]
    print(f"   exact solve -> {len(ci):,} pairs in the {CY_LO:.0f}-{CY_HI:.0f} "
          f"EFPD window ({time.time()-t0:.1f}s)")

    # ---- CBC at BOC (rho_op peak sits at t=0 for every 2-batch core here) ---
    rb = np.zeros(len(ci))
    for n, k in M.batch_weights(feed):
        rb += n * (FA * rho_at_rows(ci, k * bcp)
                   + (1 - FA) * rho_at_rows(cj, k * bcp))
    rb /= M.NSLOT
    cbc = (rb - RS) / WB
    contrast = RMEAN[ci] - RMEAN[cj]
    RESULTS.append(dict(feed=feed, i=ci, j=cj, bc=bcp, cbc=cbc,
                        cy=bcp / M.RATE, contrast=contrast,
                        ff_hot=FFMAX[cj], ff_cold=FFMAX[ci]))

np.savez_compressed(HERE + "\\screen_pairs.npz",
                    **{f"{r['feed']}_{k}": v for r in RESULTS
                       for k, v in r.items() if k != "feed"})
print("\nsaved screen_pairs.npz")
for r in RESULTS:
    f = r["feed"]
    print(f"\n=== f{f}: {len(r['i']):,} in-window ordered pairs ===")
    for cap in (1600, 1550, 1500, 1450, 1400):
        m = r["cbc"] <= cap
        print(f"   CBC<={cap}: {m.sum():>10,} pairs   min FF_hot "
              f"{r['ff_hot'][m].min() if m.any() else float('nan'):.4f}")
    for cap in (1500,):
        for con in (0.0, 0.020, 0.026, 0.035, 0.044):
            m = (r["cbc"] <= cap) & (r["contrast"] >= con)
            print(f"   CBC<={cap} & contrast>={con:.3f}: {m.sum():>9,} pairs  "
                  f"min FF_hot "
                  f"{r['ff_hot'][m].min() if m.any() else float('nan'):.4f}")
