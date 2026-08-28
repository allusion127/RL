"""Step 4: first pass of the operating-point model against the points I already
have in hand, with BOTH cyclen variants, on DeCART curves."""
import numpy as np
import paths as P
import opmodel as M

dec = P.load_decart(["T3", "T4", "T5", "T6", "Q1", "Q2", "Q7", "Q8"])
C = {n: M.Curve(*dec[n], name=n) for n in dec}

PAIRS = {"T3_T4": ("T3", "T4"), "T5_T6": ("T5", "T6"),
         "Q1_Q2": ("Q1", "Q2"), "Q7_Q8": ("Q7", "Q8")}

print("=== cy1 all-fresh critical burnup B1 (census 129/112, rho*=0.0168) ===")
for p, (a, b) in PAIRS.items():
    rm = M.mix([C[a], C[b]], M.W_CY1)
    b1 = M.b1_allfresh(rm)
    print(f"  {p}: B1 = {b1:6.2f} MWd/kgHM = {b1/M.RATE:7.1f} EFPD")
print("  observed cy1 bootstrap: T3_T4 894.09 EFPD, T5_T6 981.02 EFPD")

print("\n=== weighting sensitivity: cy1 census (129/112) vs f121 roles (68/53) ===")
for p, (a, b) in PAIRS.items():
    r1 = M.b1_allfresh(M.mix([C[a], C[b]], (129, 112)))
    r2 = M.b1_allfresh(M.mix([C[a], C[b]], (68, 53)))
    print(f"  {p}: {r1/M.RATE:7.1f} vs {r2/M.RATE:7.1f} EFPD "
          f"(d={abs(r1-r2)/M.RATE:.1f})")

print("\n=== equilibrium cyclen, both variants ===")
print(f"{'case':>14}{'n_res':>7}{'Bc_B1':>8}{'cy_B1':>8}{'Bc_EOC':>8}{'cy_EOC':>8}"
      f"{'rho_boc':>9}{'rho_pk':>9}{'t_pk':>7}")
CASES = [("T5_T6", 121), ("T5_T6", 117), ("T5_T6", 101), ("T5_T6", 81),
         ("T3_T4", 121), ("T3_T4", 117),
         ("Q1_Q2", 121), ("Q7_Q8", 121)]
res = {}
for p, f in CASES:
    a, b = PAIRS[p]
    rm = M.mix([C[a], C[b]], M.W_CY1)
    cyA, bcA = M.cyclen(rm, f, variant="B1")
    cyB, bcB = M.cyclen(rm, f, variant="EOC")
    rpk, tpk = M.rho_op_peak(rm, f, bcB)
    rboc = M.rho_op(rm, f, bcB, 0.0)
    res[(p, f)] = dict(cyA=cyA, cyB=cyB, bcB=bcB, rpk=rpk, tpk=tpk, rboc=rboc)
    print(f"{p+'@f'+str(f):>14}{M.residence(f):>7.3f}{bcA:>8.2f}{cyA:>8.1f}"
          f"{bcB:>8.2f}{cyB:>8.1f}{rboc:>9.5f}{rpk:>9.5f}{tpk:>7.2f}")

print("\n=== known measurements in hand (from the task brief) ===")
MEAS = {("T5_T6", 121): (643.565, 1725.08),
        ("T5_T6", 117): (620.7, 1701.0),
        ("T3_T4", 121): (578.3, 1250.0)}
print(f"{'case':>14}{'meas_cy':>9}{'B1 err%':>9}{'EOC err%':>10}"
      f"{'meas_cbc':>10}{'w_B implied':>13}")
for k, (mc, mb) in MEAS.items():
    r = res[k]
    wb = (r["rpk"] - M.RHO_STAR) / mb
    print(f"{k[0]+'@f'+str(k[1]):>14}{mc:>9.1f}{100*(r['cyA']-mc)/mc:>+9.2f}"
          f"{100*(r['cyB']-mc)/mc:>+10.2f}{mb:>10.0f}{wb*1e5:>13.3f}")
print("  (w_B in pcm/ppm; a consistent value across cases == the CBC model works)")
