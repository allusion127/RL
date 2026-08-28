"""Step 14: the corrected operating-point screen over all 5,874 designs.

For every ORDERED pair (i = E1 role / 68 fresh slots, j = E2 'hot' role / 53
slots) and each feed in {121, 117}:
    cyclen  = equilibrium EOC solve  x  hump correction
    CBC     = (rho_core_BOC - rho*) / w_B
    contrast= mean rho difference over BU 0.5-8 (the zoning the LP relies on)
    FF_hot  = the 53-slot type's ensemble form factor
Survivors are written to screen_final_<feed>.npz.
"""
import numpy as np
import os
import time
import pandas as pd
import paths as P
import opmodel as M

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(HERE + "\\calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
HA, HB = float(c["hump_a"]), float(c["hump_b"])
print(f"rho*={RS:.5f}  w_B={1e5*WB:.4f} pcm/ppm  hump corr "
      f"(1 {HA:+.5f} {HB:+.5f}*hump)")

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
bu0, K = z["bu"], z["kconv"]
keep = bu0 >= 0.2
BU = bu0[keep]
R = np.ascontiguousarray((1.0 - 1.0 / K)[:, keep])
FFMAX = df.ff_max.values
N = len(R)
FA = 68.0 / 121.0

CY_LO, CY_HI = 620.0, 645.0
TH = np.linspace(0.5, 12.0, 24)                # hump window
TC = np.linspace(0.5, 8.0, 12)                 # contrast window


def _w(t):
    j = max(1, int(np.searchsorted(BU, t)))
    return j, (t - BU[j - 1]) / (BU[j] - BU[j - 1])


HS = np.empty((N, len(TH)))
for q, t in enumerate(TH):
    j, w = _w(t)
    HS[:, q] = R[:, j - 1] * (1 - w) + R[:, j] * w
CS = np.empty((N, len(TC)))
for q, t in enumerate(TC):
    j, w = _w(t)
    CS[:, q] = R[:, j - 1] * (1 - w) + R[:, j] * w
RMEAN = CS.mean(axis=1)


def rho_rows(rows, x):
    x = np.clip(x, BU[0], BU[-1])
    j = np.clip(np.searchsorted(BU, x), 1, len(BU) - 1)
    w = (x - BU[j - 1]) / (BU[j] - BU[j - 1])
    return R[rows, j - 1] * (1 - w) + R[rows, j] * w


def solo_bc(feed):
    a, b = np.full(N, 6.0), np.full(N, 45.0)
    idx = np.arange(N)
    for _ in range(60):
        m = 0.5 * (a + b)
        g = np.zeros(N)
        for n, k in M.batch_weights(feed):
            g += n * rho_rows(idx, (k + 1) * m)
        v = g / M.NSLOT > RS
        a = np.where(v, m, a)
        b = np.where(v, b, m)
    return 0.5 * (a + b)


HUMP_SOLO = HS.max(axis=1) - HS[:, 0]
CMAX = 1.0 + HA + HB * HUMP_SOLO.max()
CMIN = 1.0 + HA + HB * 0.0
RAW_LO = CY_LO / CMAX * M.RATE - 0.5
RAW_HI = CY_HI / CMIN * M.RATE + 0.5
print(f"raw Bc prefilter window {RAW_LO:.3f} .. {RAW_HI:.3f} MWd/kgHM")

OUT = {}
for feed in (121, 117):
    t0 = time.time()
    bc = solo_bc(feed)
    d = 0.25
    idx = np.arange(N)
    g1 = np.zeros(N)
    g2 = np.zeros(N)
    for n, k in M.batch_weights(feed):
        g1 += n * rho_rows(idx, (k + 1) * (bc + d))
        g2 += n * rho_rows(idx, (k + 1) * (bc - d))
    sl = (g1 - g2) / M.NSLOT / (2 * d)
    wi, wj = FA * sl, (1 - FA) * sl

    ci, cj = [], []
    CH = 128
    for s in range(0, N, CH):
        e = min(s + CH, N)
        est = ((wi[s:e, None] * bc[s:e, None] + wj[None, :] * bc[None, :])
               / (wi[s:e, None] + wj[None, :]))
        m = (est > RAW_LO) & (est < RAW_HI)
        ii, jj = np.nonzero(m)
        ci.append((ii + s).astype(np.int32))
        cj.append(jj.astype(np.int32))
    ci = np.concatenate(ci)
    cj = np.concatenate(cj)
    print(f"\nf{feed}: {len(ci):,} candidate ordered pairs "
          f"({time.time()-t0:.1f}s)")

    keep_i, keep_j, keep_bc, keep_cy, keep_cbc, keep_hp = [], [], [], [], [], []
    BW = M.batch_weights(feed)
    t0 = time.time()
    STEP = 3_000_000
    for s in range(0, len(ci), STEP):
        e = min(s + STEP, len(ci))
        I, J = ci[s:e], cj[s:e]
        a = np.full(len(I), 14.0)
        b = np.full(len(I), 34.0)
        for _ in range(28):
            m = 0.5 * (a + b)
            g = np.zeros(len(I))
            for n, k in BW:
                g += n * (FA * rho_rows(I, (k + 1) * m)
                          + (1 - FA) * rho_rows(J, (k + 1) * m))
            v = g / M.NSLOT > RS
            a = np.where(v, m, a)
            b = np.where(v, b, m)
        bcp = 0.5 * (a + b)
        mixH = FA * HS[I] + (1 - FA) * HS[J]
        hp = mixH.max(axis=1) - mixH[:, 0]
        cy = bcp / M.RATE * (1.0 + HA + HB * hp)
        ok = (cy >= CY_LO) & (cy <= CY_HI)
        if not ok.any():
            continue
        I, J, cy, hp = I[ok], J[ok], cy[ok], hp[ok]
        bcc = cy * M.RATE
        rb = np.zeros(len(I))
        for n, k in BW:
            rb += n * (FA * rho_rows(I, k * bcc) + (1 - FA) * rho_rows(J, k * bcc))
        rb /= M.NSLOT
        keep_i.append(I)
        keep_j.append(J)
        keep_bc.append(bcc)
        keep_cy.append(cy)
        keep_cbc.append((rb - RS) / WB)
        keep_hp.append(hp)
    I = np.concatenate(keep_i)
    J = np.concatenate(keep_j)
    cy = np.concatenate(keep_cy)
    cbc = np.concatenate(keep_cbc)
    hp = np.concatenate(keep_hp)
    con = RMEAN[I] - RMEAN[J]
    print(f"   -> {len(I):,} pairs with corrected cyclen in "
          f"[{CY_LO:.0f},{CY_HI:.0f}] ({time.time()-t0:.1f}s)")
    OUT[feed] = dict(i=I, j=J, cy=cy, cbc=cbc, hump=hp, contrast=con,
                     ff_hot=FFMAX[J], ff_cold=FFMAX[I])
    np.savez_compressed(HERE + f"\\screen_final_{feed}.npz", **OUT[feed])
    for cap in (1600, 1550, 1500, 1450):
        m = cbc <= cap
        print(f"   CBC<={cap}: {m.sum():>9,}  min FF_hot "
              f"{FFMAX[J][m].min() if m.any() else float('nan'):.4f}")
    for con_min in (0.000, 0.020, 0.026, 0.035, 0.044):
        m = (cbc <= 1500) & (con >= con_min)
        print(f"   CBC<=1500 & contrast>={con_min:.3f}: {m.sum():>8,}  "
              f"min FF_hot "
              f"{FFMAX[J][m].min() if m.any() else float('nan'):.4f}")
print("\nsaved screen_final_121.npz / screen_final_117.npz")
