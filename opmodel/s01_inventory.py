"""Step 1: inventory the realized lattice types and check which of them the
lat1600 surrogate screen also covers (needed to calibrate surrogate -> DeCART)."""
import numpy as np
import pandas as pd
import paths as P

des = P.load_designs()
dec = P.load_decart()
print("designs.json aliases: %d ; hgc FA_*.out: %d" % (len(des), len(dec)))

df = pd.read_csv(P.LAT + "\\screen1600.csv")
z = np.load(P.LAT + "\\screen1600_raw.npz")
sbu, sK, sFF = z["bu"], z["kconv"], z["ff"]
print("screen table: %d designs; surrogate bu grid n=%d %.2f..%.2f"
      % (len(df), len(sbu), sbu.min(), sbu.max()))
print("screen u_high grid:", sorted(df.u_high.unique()))

rows = []
for a, r in sorted(des.items()):
    uh, gw, ng = r["e1"], r["gd_wt"], r["n_gd"]
    zv = r["zoning_variant"]
    gp = r.get("gd_positions")
    hit = ""
    if gp is not None:
        m = (np.isclose(df.u_high, uh) & np.isclose(df.gd_wt, gw)
             & (df.n_gd == ng) & (df.gd_positions == gp) & (df.zoning == zv))
        idx = np.flatnonzero(m.values)
        hit = str(idx[0]) if len(idx) == 1 else f"n={len(idx)}"
    k0 = dec[a][1][0] if a in dec else float("nan")
    rows.append((a, r["type_id"], uh, r["e2"], zv, gw, ng, gp or "-", k0, hit))

print(f"\n{'al':>3} {'type_id':>14} {'e1':>5} {'e2':>7} {'z':>3} {'gw':>4} "
      f"{'ngd':>4} {'gd_positions':>22} {'k_dec0':>8} {'screen_row':>10}")
for t in rows:
    print(f"{t[0]:>3} {t[1]:>14} {t[2]:>5} {t[3]:>7} {t[4]:>3} {t[5]:>4.0f} "
          f"{t[6]:>4} {t[7]:>22} {t[8]:>8.4f} {t[9]:>10}")

# any old type that happens to sit on the screen's u_high grid?
print("\nold types with e1 in the screen grid (5.00-5.50):")
for a, r in sorted(des.items()):
    if 4.99 <= r["e1"] <= 5.51:
        print("  ", a, r["type_id"], r.get("gd_positions", "NO gd_positions"))
