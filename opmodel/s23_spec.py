"""Step 23: full spec sheet for the recommended new pair (Z1/Z2) and its
alternates, at both feeds, plus the sensitivity of the answer to rho* and w_B."""
import numpy as np
import pandas as pd
import paths as P
import opmodel as M

c = np.load("calib_final.npz")
RS, WB = float(c["rho_star"]), float(c["w_b"])
HA, HB = float(c["hump_a"]), float(c["hump_b"])
df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
bu0, K = z["bu"], z["kconv"]
TH = np.linspace(0.5, 12.0, 24)
TC = np.linspace(0.5, 8.0, 12)


def row(uh, gw, ng, gp):
    m = (np.isclose(df.u_high, uh) & np.isclose(df.gd_wt, gw)
         & (df.n_gd == ng) & (df.gd_positions == gp)
         & (df.pattern == "PB") & (df.zoning == "z1"))
    k = np.flatnonzero(m.values)
    assert len(k) == 1
    return int(k[0])


def ev(ia, ib, feed, rs=RS, wb=WB):
    rm = M.mix([M.Curve(bu0, K[ia]), M.Curve(bu0, K[ib])], [68.0, 53.0])
    hump = float(np.max(rm(TH)) - rm(0.5))
    raw, _ = M.cyclen(rm, feed, rs, "EOC")
    cy = raw * (1 + HA + HB * hump)
    bc = cy * M.RATE
    rpk, _ = M.rho_op_peak(rm, feed, bc)
    con = float(np.mean(M.Curve(bu0, K[ia])(TC) - M.Curve(bu0, K[ib])(TC)))
    return raw, cy, (rpk - rs) / wb, con, hump


PAIRS = {
    "Z1/Z2  (u5.50 gd8x20) / (u5.00 gd10x20)":
        ((5.50, 8, 20, "1:1;4:1;6:4"), (5.00, 10, 20, "1:1;4:1;6:4")),
    "Z1'/Z2 (u5.45 gd8x20) / (u5.00 gd10x20)":
        ((5.45, 8, 20, "1:1;4:1;6:4"), (5.00, 10, 20, "1:1;4:1;6:4")),
    "Z1/Z2' (u5.50 gd8x20) / (u5.05 gd10x20)":
        ((5.50, 8, 20, "1:1;4:1;6:4"), (5.05, 10, 20, "1:1;4:1;6:4")),
    "Z3/Z2''(u5.50 gd6x20) / (u5.35 gd10x20)":
        ((5.50, 6, 20, "1:1;4:1;6:4"), (5.35, 10, 20, "1:1;4:1;6:4")),
}
print(f"{'pair':>42}{'feed':>5}{'raw':>7}{'cyc':>7}{'CBC':>6}{'FFhot':>8}"
      f"{'FFcold':>8}{'contr':>9}{'hump':>7}{'Fr_flr':>8}")
for lab, (A, B) in PAIRS.items():
    ia, ib = row(*A), row(*B)
    for feed in (121, 117):
        raw, cy, cbc, con, hump = ev(ia, ib, feed)
        print(f"{lab:>42}{feed:>5}{raw:>7.1f}{cy:>7.1f}{cbc:>6.0f}"
              f"{df.ff_max.values[ib]:>8.4f}{df.ff_max.values[ia]:>8.4f}"
              f"{con:>+9.4f}{hump:>7.4f}"
              f"{1.03*1.2085*df.ff_max.values[ib]:>8.3f}")
    print(f"   68-role u_low {df.u_low.values[ia]:.4f} | "
          f"53-role u_low {df.u_low.values[ib]:.4f} | "
          f"rbar_eoc(screenK) {df.rbar_eoc.values[ia]:.5f}/"
          f"{df.rbar_eoc.values[ib]:.5f} | old cbc_pred "
          f"{df.cbc_pred.values[ia]:.0f}/{df.cbc_pred.values[ib]:.0f}")

print("\n=== sensitivity of Z1/Z2 @f121 to the calibration constants ===")
ia, ib = row(5.50, 8, 20, "1:1;4:1;6:4"), row(5.00, 10, 20, "1:1;4:1;6:4")
print(f"{'rho*':>9}{'w_B':>9}{'cyclen':>9}{'CBC':>7}")
for drs in (-0.0015, 0.0, +0.0015):
    for dwb in (-0.15e-5, 0.0, +0.15e-5):
        raw, cy, cbc, *_ = ev(ia, ib, 121, RS + drs, WB + dwb)
        print(f"{RS+drs:>9.5f}{1e5*(WB+dwb):>9.3f}{cy:>9.1f}{cbc:>7.0f}")
print("\n(+-0.0015 in rho* ~ the spread between the per-feed fits; "
      "+-0.15 pcm/ppm ~ the per-point scatter in w_B)")
