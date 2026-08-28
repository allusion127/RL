# OPSCREEN — re-screening the lat1600 design space against the OPERATING POINT

**Date** 2026-08-11 · **Scope** CPU only, read-only on `data/` and `0_APR1400/`, no MASTER, no
DeCART. Everything written under `scratchpad\lat1600_v2\`.
**Input design table** `scratchpad\lat1600\screen1600.csv` (5,874 designs) +
`screen1600_raw.npz` (surrogate `kconv`, `ff` on a 62-point BU grid).
**Ground truth** `5_RL\data\design\package\hgc\FA_*.{out,HGC}` (37 paramA lattices) and
`3_GA_Surrogate\FEASIBLE_PACKAGE\hgc\FA_*.HGC` (36 ga80 lattices), plus 79 MASTER-measured
equilibrium operating points harvested from `runs\fr_arms_*`, `runs\fr_transfer_*`,
`runs\fr_screen11\*` and `data\store\records.parquet`.

---

## 0. Verdict in six lines

1. A corrected operating-point model now predicts equilibrium **cyclen to 4.3 EFPD rms (0.66 %)**
   and **CBC to 37 ppm rms** over 55 measured points at feeds 117/121. It reproduces every
   deployment surprise the old screen missed.
2. The old screen was wrong in three independent ways at once (§4); the 1725 ppm at T5_T6@f121
   was **not** a modelling accident, it was the arithmetic the screen never did.
3. **Re-screened, T5_T6 has no operating point.** At f121 it is in the cycle window but
   **125–137 ppm over the 1600 gate as measured** (1725 fixed-pattern / 1737 elite-32 mean;
   model 1757); at f117 it is 102 ppm over (1702 measured, model 1706). No pairing or feed of
   T5+T6 clears both gates.
4. **The measured fuel lever on F_r does not exist the way the plan assumed.** On a fixed
   pattern, the flattest fuel measured the *worst* F_r (T5_T6 F_r 1.5795 vs E1_E2 1.5207) because
   T5 and T6 differ by under 200 pcm — 0.05 % U-235, same Gd — so the pattern's E1/E2 zoning
   collapses into a single-type load. Reactivity **contrast** between the two fresh roles, not
   FF, is what holds `node_peak` down (§6).
5. **Best already-realized answer: `T6(68) / T4(53) @ f121`** (or `T1(68)/T4(53) @ f117`) —
   zero DeCART, zero library rebuild, one MASTER bootstrap. But its FF_hot (1.1430) is **no
   better than ga80's E4 (1.1390) which is already in production**, so the T3–T6 wave bought
   nothing at the operating point.
6. **A new lattice does buy ≥ 0.005 FF — 3.6× over.** The open 20-pin layout `1:1;4:1;6:4` reaches
   FF_hot **1.1208** with adequate contrast and CBC 1397: ΔFF −0.018 vs the incumbent E4,
   ΔF_r −0.022 by the fusion law. Whether that is worth one DeCART wave + a library rebuild that
   re-stales all nine restarts (§9) is a program call, and it should be made **after** the free
   T6_T4 measurement, not before.

---

## 1. The model

Three ingredients, all from lattice `k(BU)` alone. Scripts: `opmodel.py`, `s13_final.py`.

### 1.1 Equilibrium cycle length

Discrete equilibrium, solved on the true (curved) `rho(BU)`, not on a linearisation:

```
batches at EOC:  241 = k*feed + remainder,  batch j has burnup (j+1)*Bc
criticality   :  sum_j w_j * rho((j+1)*Bc) / 241  ==  rho*
cyclen        :  Bc / RATE  *  (1 - 0.00548 + 1.35866*hump)
```

with `RATE = 3983/104.8/1000 = 0.0380058 MWd/kgHM per EFPD`,
`rho* = 0.02000`, and

```
hump = max_{0.5 <= t <= 12} rho_mix(t) - rho_mix(0.5)
```

* feed 121 → batches (121 @ Bc, 120 @ 2Bc); feed 117 → (117, 117, 7); feed 101 → (101, 101, 39).
  The 1+4N grid is handled exactly; no fractional-batch fudge.
* The fresh mixture is weighted **68 : 53** (the E1-role / E2-role fresh-slot split at f121,
  `designs.json` `lat1600_role`). Using the cy1 census 129:112 instead moves B1 by < 0.5 EFPD
  (`s04_firstcheck.py`), so the screen is insensitive to it.
* **The Xe-free BU = 0 lattice point is dropped everywhere.** MASTER runs equilibrium Xe;
  BU ≥ 0.2 is the comparable state. This is also the *only* place the lat1600 surrogate disagrees
  with DeCART by more than ~100 pcm (§3).

**Why the hump term.** The pure linear-reactivity form assumes every batch sits at exactly
`j*Bc` at EOC. Gd-bearing fresh assemblies are power-*suppressed* at BOC — that is the point of
the Gd — so they burn slower than average early in life and reach EOC below `Bc`, retaining
reactivity. A back-of-envelope check: 121/241 of the core running at p ≈ 0.85 for the first
third of a cycle is a burnup deficit ≈ 1.2 MWd/kgHM on half the core; at the measured
−670 pcm/(MWd/kgHM) EOC slope that is +400 pcm core-average ≈ +0.6 MWd/kgHM of extra cycle ≈
**+16 EFPD** — the same order and sign as the fitted coefficient produces. The correction cut
the operating-window rms from 5.89 to 4.26 EFPD and **improved all four held-out lat1600
points** when fitted with T3–T6 excluded (`s13_final.py`):

| held-out point | measured | raw | corrected |
|---|---:|---:|---:|
| T3_T4@f121 (B2) | 592.5 | 587.1 (−0.91 %) | 591.3 (**−0.20 %**) |
| T5_T6@f121 (B3) | 643.6 | 630.9 (−1.97 %) | 636.6 (**−1.08 %**) |
| T5_T6@f121 (elite-32) | 645.8 | 630.9 (−2.31 %) | 636.6 (**−1.43 %**) |
| T5_T6@f117 (elite-32) | 630.2 | 611.5 (−2.97 %) | 617.1 (**−2.09 %**) |

A physically cleaner alternative — flux weighting, `rho_eff = rho_bar + a*Var(rho)` — was tried
and **rejected**: the fit drives `a → 0` and the hump correlation survives untouched
(`s17_variance.py`). The empirical patch is the best available; its calibration range is
`hump ∈ [0, 0.0148]` and every candidate table below flags extrapolation beyond it.

### 1.2 Critical boron

Boron is the reactivity the core must hold **down**, so it is a difference, not a contour:

```
CBC(t)  = ( rho_core_op(t) - rho* ) / w_B ,     w_B = 5.3476e-5  (5.348 pcm/ppm)
CBC_max = max over t in [0, Bc]
```

`rho_core_op(t)` is the equilibrium core-average k-inf reactivity at time `t` into the cycle
(fresh batch at `t`, once-burned at `t+Bc`, …). For every 2-batch core in this study the maximum
falls at `t = 0`, so `CBC_max` is a BOC quantity; the `max` machinery is kept for generality.

`w_B` is a *single global constant* fitted to 55 measurements — no per-design freedom. 5.35
pcm/ppm is a physically sensible differential boron worth for an APR1400 at HFP with
1100–2200 ppm in the water.

### 1.3 Core F_r

```
F_r_floor = 1.03 * 1.2085 * FF_hot          (chosen.json headline convention: A=1.03, p_hot=1.2080)
```

kept **only** because the program's plan tables are written in it. §6 shows what it is worth.

---

## 2. Validation table — predicted vs measured

Final calibration: `rho* = 0.02000`, `w_B = 5.3476 pcm/ppm`, hump correction
`(1 − 0.00548 + 1.35866·hump)`. Script `s18_report.py`.

| case | feed | source | cyclen meas | pred | Δ | Δ% | CBC meas | pred | Δ | hump |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **T5_T6** | 121 | fr_arms B3 (1 pattern) | 643.6 | 640.1 | −3.4 | −0.53 % | 1725 | 1757 | +32 | 0.0148 |
| **T5_T6** | 121 | elite-32 mean | 645.8 | 640.1 | −5.7 | −0.88 % | 1737 | 1757 | +20 | 0.0148 |
| **T5_T6** | 117 | elite-32 mean | 630.2 | 620.5 | −9.7 | −1.54 % | 1702 | 1706 | +4 | 0.0148 |
| **T3_T4** | 121 | fr_arms B2 (1 pattern) | 592.5 | 594.0 | +1.6 | +0.27 % | 1187 | 1255 | +68 | 0.0128 |
| **Q1_Q2** | 121 | fr_arms B0 | 779.5 | 775.2 | −4.2 | −0.54 % | 2244 | 2156 | −88 | 0.0000 |
| **Q7_Q8** | 121 | fr_arms B1 | 792.9 | 796.0 | +3.2 | +0.40 % | 2137 | 1985 | −152 | 0.0000 |
| **E1_E2** | 121 | fr_arms A0 | 632.3 | 633.8 | +1.5 | +0.24 % | 1330 | 1361 | +30 | 0.0101 |
| E1_E2 | 121 | fr_arms A0 (minfr anchor) | 633.3 | 633.8 | +0.5 | +0.08 % | 1327 | 1361 | +34 | 0.0101 |
| E1_E2 | 121 | store median, n=1097 | 633.1 | 633.8 | +0.7 | +0.11 % | 1330 | 1361 | +31 | 0.0101 |
| **E1_E2** | 117 | store median, n=501 | 612.2 | 613.3 | +1.1 | +0.18 % | 1280 | 1319 | +39 | 0.0101 |
| E3_E4 | 121 | elite-24 mean | 633.3 | 630.0 | −3.3 | −0.52 % | 1373 | 1439 | +66 | 0.0053 |
| J5_J6 | 121 | elite-32 mean | 634.8 | 634.8 | −0.0 | −0.01 % | 1298 | 1317 | +18 | 0.0020 |
| K3_K4 | 121 | elite-32 mean | 642.5 | 644.2 | +1.7 | +0.27 % | 1318 | 1328 | +10 | 0.0032 |
| E3 ×121 (single type) | 121 | fr_arms C5 | 639.8 | 628.2 | −11.6 | −1.81 % | 1670 | 1733 | +63 | 0.0000 |

**Whole-set residuals, operating window (feeds 117 + 121, n = 55):**

| | bias | rms | max abs |
|---|---:|---:|---:|
| cyclen | **+0.05 EFPD** | **4.26 EFPD (0.66 %)** | 12.3 EFPD |
| CBC (all) | +3 ppm | 43 ppm | 152 ppm |
| CBC in the gate region (≤ 1900 ppm, n = 53) | **+7 ppm** | **37 ppm** | **86 ppm** |

90 % of the in-gate CBC residuals fall within **±67 ppm**. The two 100+ ppm misses are Q1_Q2 and
Q7_Q8 at 2137–2244 ppm — 6.6 %-enriched cores far outside anything the screen will propose.

**Answer to "can the CBC model do better than ±100 ppm?" — yes, but only just.** Quote it as
**±70 ppm (90 %)** in the 1100–1800 ppm band, and never tighter. That is why every gate below is
set at 1500 ppm, not 1600: 100 ppm for elite-pattern spread (measured: T5_T6@f121 elite-32 span
1715–1770, i.e. ±28 around the mean; T5_T6@f117 span 1647–1747, ±50) plus the model's own ±70.

Full 79-point table including feeds 101/125/141 is in `s06_validate.py` output; the model
degrades to 1.15 % rms there, systematically short at f101 (−1.5 to −3.2 %), which is why the
screen's calibration is restricted to the operating window.

---

## 3. Does the SURROGATE table transfer? (yes, for k; yes, for FF)

T3–T6 are a genuine held-out test — the surrogate screen picked them, then DeCART computed them.

* **k(BU)**: surrogate and DeCART agree to **< 100 pcm at every burnup ≥ 0.2 MWd/kgHM**. Only the
  Xe-free BU = 0 point differs (−2200 pcm), and the model never uses it. Consequence
  (`s08_transfer.py`): the full operating point computed from the surrogate curves vs the DeCART
  curves differs by **≤ 0.5 EFPD and ≤ 8 ppm** on all 16 T-pair/feed combinations. The re-screen
  can use the surrogate table for all 5,874 designs with no meaningful transfer penalty.
* **FF**: surrogate 8-member ensemble vs DeCART `%DIST` max, held out —
  T3 1.1073/1.1090, T4 1.1409/1.1430, T5 1.1012/1.1020, T6 1.1011/1.1020.
  Bias **−0.0014**, rms 0.0015, max 0.0021. Below the 0.005 decision threshold, so FF differences
  of 0.02 in the tables below are real.

---

## 4. Why the old screen missed — three independent errors

| # | old convention | what it should have been | cost |
|---|---|---|---|
| 1 | Screen K required `rbar_eoc ≥ −0.076292` at a **3-batch** `Bc = 24.7327` fixed a priori | 2-batch equilibrium (241 = 121 + 120), `Bc` **solved**, not assumed | T3_T4 landed 50 EFPD low; T5_T6 landed at the top of the window by luck |
| 2 | `CBC = 26176·rbar_peak + 133`, a contour regressed on ga80@f121 in the same 3-batch surrogate space | `CBC = (rho_core_BOC − rho*)/w_B` — a *difference* from criticality, with the real equilibrium batch structure | predicted 1501/1561 for T5/T6; the pair measured **1725** |
| 3 | statistics evaluated per **single lattice**, then eyeballed for the pair | the operating point is a property of the **pair × feed**, and CBC is dominated by the once-burned batch, not the fresh one | T3+T4 contour average 762 ppm vs 1187 measured |

The 3-batch convention and the boron contour were separately ~20 % wrong and in the same
direction, so the errors compounded rather than cancelled.

---

## 5. Re-screen: what survives at the operating point

Full pairwise sweep, both feeds, over all 5,874 designs in both role orders
(`s10_screen.py`, `s14_screen_final.py`; ~34 M ordered pairs prefiltered, ~11 M exactly solved
at f121 and ~2.6 M at f117). Results cached in `screen_final_121.npz`, `screen_final_117.npz`.

Gates: **cyclen ∈ [620, 645] EFPD** and **CBC ≤ 1500 ppm** (the 1600 program gate less 100 ppm of
pattern + model headroom).

| feed | in cycle window | + CBC ≤ 1500 | + contrast ≥ 0.026 | min FF_hot |
|---|---:|---:|---:|---:|
| 121 | 11,385,979 | 4,283,811 | 900,963 | 1.0983 → **1.1190** with contrast |
| 117 | 2,605,455 | 1,275,876 | 232,501 | 1.0984 → **1.1190** with contrast |

Without the contrast constraint the screen happily returns FF_hot 1.0983 — and every one of
those pairs is a T5_T6 clone (contrast −0.06, predicted `node_peak` 1.65). That constraint is the
whole lesson of the deployment; §6 is its evidence.

---

## 6. The F_r finding — the fusion law does not survive contact with the pattern

`fr_arms` gives 17 measurements on **one byte-identical loading pattern** with only the fresh
fuel swapped. Testing `F_r = A·p_hot·FF_hot` with the flat-anchor calibration
`A·p = F_r(A0)/FF(E2) = 1.5207/1.1520 = 1.3200` (`s09_frmodel.py`):

| arm | hot type | FF_hot | law | **measured** | error | measured node_peak |
|---|---|---:|---:|---:|---:|---:|
| A0 | E2 | 1.1520 | 1.5206 | 1.5207 | −0.000 | 1.2085 |
| C1 | H4 | 1.1710 | 1.5457 | 1.5258 | +0.020 | 1.2736 |
| C4 | J2 | 1.1460 | 1.5127 | 1.5297 | −0.017 | 1.3269 |
| **B2 (T3_T4)** | T4 | 1.1430 | 1.5088 | **1.5329** | −0.024 | 1.2145 |
| **B3 (T5_T6)** | T6 | 1.1020 | 1.4546 | **1.5795** | **−0.125** | **1.3906** |
| C5 (E3 ×121) | E3 | 1.1010 | 1.4533 | 1.5590 | −0.106 | 1.3875 |
| C6 (A2 ×121) | A2 | 1.1780 | 1.5550 | 1.8175 | −0.263 | 1.5514 |
| D1 (H2 ×121) | H2 | 1.1430 | 1.5088 | 1.6872 | −0.178 | 1.4613 |

The law holds to ±0.02 on the eight **two-type** arms and fails by 0.1–0.26 on every arm where the
two fresh roles are the same or nearly the same lattice. The discriminator is the role
**contrast** — the mean rho difference between the 68-slot and 53-slot types over BU 0.5–8:

| contrast | arms | measured node_peak | measured F_r |
|---|---|---:|---:|
| ≥ 0.043 | A0, A1, A2, B2 | 1.209 – 1.260 | 1.514 – 1.548 |
| 0.026 – 0.028 | C1, C2, C3, C4 | 1.274 – 1.327 | 1.526 – 1.536 |
| ≈ 0 | **B3**, C5, C6, D1–D4 | **1.387 – 1.551** | **1.559 – 1.818** |

`node_peak = 1.4210 − 4.1725·contrast − 3.4862·d_fresh` fits the 15 arms with rms 0.036,
R² 0.866 (`s09b_contrast.py`), and `A = F_r/(node_peak·FF_hot) = 1.035 ± 0.031` — *that* is where
the "A = 1.03" in `chosen.json` comes from. The plan's error was holding `p_hot = 1.2080` fixed
while changing the fuel; it is not fixed, it is a function of the fuel pair.

**This is not one unlucky pattern.** The 32-pattern elite transfer into T5_T6@f121 produced
F_r 1.5257 – 1.8100 (min 1.5257) and T5_T6@f117 1.6188 – 1.9990, while the incumbent E1_E2 cell
holds the record 1.4636. Thirty-two independent elite patterns could not recover the loss.

**Consequence for the screen**: `contrast ≥ 0.026` is a hard gate, and `F_r_floor` is reported as
what the LP could reach *if* it restores node_peak to the best value ever measured — not as a
prediction of what the current patterns will give.

---

## 7. Ranked candidates

`X` = the hump correction is extrapolating (mixture hump > 0.0148); for those rows the honest
cycle length is the **interval [raw, cyc]**, not the corrected value.
`Fr_flr = 1.03 × 1.2085 × FF_hot`. FF for realized types is the DeCART `%DIST` value; for new
designs it is the surrogate ensemble (which runs 0.0014 low, §3).

### 7a. Already realized — zero DeCART, zero library rebuild (`s19_realized_final.py`, `s21_top5.py`)

Every realized pairing of all 37 paramA types was screened (2,738 ordered pair × feed
combinations). Those clearing cyclen + CBC ≤ 1500 + contrast ≥ 0.026:

| # | 68-role | 53-role (hot) | feed | raw | cyc | CBC | FF_hot | contrast | hump | Fr_flr | X |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **R1** | **T1** (6.20 %, gd10×20) | **T4** | 117 | 626.3 | 634.0 | **1283** | 1.1430 | +0.0513 | 0.0131 | 1.423 | |
| **R2** | **T6** (5.30 %, gd6×16) | **T4** | 121 | 611.5 | 629.8 | **1383** | 1.1430 | +0.0747 | 0.0261 | 1.423 | X |
| R3 | P9 (5.80 %, gd10×20) | T4 | 117 | 614.9 | 623.5 | 1239 | 1.1430 | +0.0449 | 0.0144 | 1.423 | |
| R4 | T5 (5.25 %, gd6×16) | T4 | 121 | 608.0 | 626.7 | 1369 | 1.1430 | +0.0732 | 0.0267 | 1.423 | X |
| R5 | P0 (5.80 %, gd8×16) | T4 | 117 | 624.6 | 629.8 | 1467 | 1.1430 | +0.0812 | 0.0102 | 1.423 | |
| — | T5 | T6 *(today's cell)* | 121 | 630.9 | **640.1** | **1757** | 1.1011 | −0.0016 | 0.0148 | 1.372 | **CBC + contrast FAIL** |
| — | T3 | T4 | 121 | 587.1 | 594.0 | 1255 | 1.1430 | +0.0428 | 0.0128 | 1.423 | **cyclen FAIL (−26)** |

R1 and R5 need **no** correction at all — their *raw* prediction is already in window.
R2/R4 depend on a hump correction extrapolated 1.8× beyond its calibration range; their honest
range is 612–630 and 608–627 EFPD.

Every one of these has **T4 in the hot role**, FF 1.1430. That is the ceiling of what the
realized paramA set can do once contrast is enforced.

**One near-miss worth naming.** `S9(68, 5.80 %, gd6×20) / T3(53, 5.00 %, gd10×16) @ f117`:
raw 626.6 / corrected 629.9 EFPD, contrast +0.0270, hump 0.0079 (inside calibration), and
**FF_hot 1.1090** — a −0.030 gain over E4, better than anything a new lattice offers. It fails on
boron: **CBC 1599 ppm**, i.e. exactly on the 1600 gate with zero headroom for the model's ±70 or
the elite-pattern ±50. It is the only zero-cost route to FF_hot below 1.12, and it would become
viable if the gate ever moved to ~1750. Do not run it at 1600.

### 7b. The uncomfortable comparison — ga80, already in production

Screening the 36 ga80 lattices the same way (`s23`-adjacent query, 188 passing combinations):

| 68-role | 53-role (hot) | feed | raw | cyc | CBC | FF_hot | contrast |
|---|---|---:|---:|---:|---:|---:|---:|
| **A8** | **E4** | 121 | 630.5 | 635.1 | 1375 | **1.1390** | +0.0508 |
| G3 | E4 | 117 | 643.6 | 641.5 | 1358 | 1.1390 | +0.0476 |
| E3 | E4 | 121 | 628.9 | 630.0 | 1439 | 1.1390 | +0.0603 |
| E1 | E2 | 121 | 628.6 | 633.8 | 1361 | 1.1520 | +0.0490 |

`E3_E4@f121` is *measured* at 633.3 EFPD / 1373 ppm (elite-24 mean, 183 store rows) — already
in window, already under gate, already bootstrapped.

**ga80's E4 (FF 1.1390) is flatter than T4 (1.1430)**, sits in a fully bootstrapped library with
71,617 store rows, and `E3_E4@f121` is already a measured, in-window, under-gate cell.
**At the operating point, the T3–T6 wave bought nothing.** That is the honest scorecard of the
2026-08-11 realization: T5/T6 are flat but cannot hold their boron; T3/T4 hold boron but are no
flatter than fuel we already had.

### 7c. New lattices (`s14`/`s15`/`s21`)

Top of the contrast-constrained, CBC ≤ 1500, non-extrapolating front. `dFF`/`dFr` are versus the
incumbent E4 (1.1390).

| # | feed | 68-role | 53-role (hot) | raw | cyc | CBC | FF_hot | FF_cold | contrast | hump | Fr_flr | dFF | dFr |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Z** | 121 | u5.50 gd8×20 `1:1;4:1;6:4` | u5.00 gd10×20 `1:1;4:1;6:4` | 622.5 | **624.8** | **1397** | **1.1208** | 1.1202 | +0.0300 | 0.0067 | **1.395** | −0.018 | −0.022 |
| Z′ | 121 | u5.45 gd8×20 `1:1;4:1;6:4` | u5.00 gd10×20 `1:1;4:1;6:4` | 619.0 | 621.7 | 1382 | 1.1208 | 1.1201 | +0.0282 | 0.0072 | 1.395 | −0.018 | −0.022 |
| Z″ | 121 | u5.50 gd8×20 `1:1;4:1;6:4` | u5.05 gd10×20 `1:1;4:1;6:4` | 625.5 | 627.4 | 1409 | 1.1210 | 1.1202 | +0.0280 | 0.0063 | 1.395 | −0.018 | −0.022 |
| W | 117 | u5.50 gd6×20 `1:1;4:1;6:4` | u5.35 gd10×20 `1:1;4:1;6:4` | 623.5 | 632.5 | 1462 | 1.1217 | 1.1180 | +0.0304 | 0.0147 | 1.396 | −0.017 | −0.021 |
| N2 | 121 | u5.30 gd6×20 `2:0;2:2;5:1` | u5.05 gd8×20 `1:1;4:1;6:4` | 614.2 | 622.2 | 1500 | 1.1191 | 1.2106 | +0.0306 | 0.0135 | 1.393 | −0.020 | −0.025 |

`Z` is the recommendation if a new wave is authorised: **both roles are flat** (1.1202 / 1.1208,
so the answer does not depend on which role actually holds the hot slot), the raw cyclen is
already in window (the correction is not load-bearing), the hump is well inside calibration, and
CBC 1397 leaves **203 ppm** under the 1600 gate. N2 buys another 0.0017 of FF_hot but pays for it
with a 1.2106 cold role and a CBC sitting exactly on the screening gate — reject.

**Sensitivity of `Z`@f121 to the calibration** (`s23_spec.py`), over `rho* ± 0.0015` (the spread
between the per-feed fits) and `w_B ± 0.15 pcm/ppm` (the per-point scatter):

| | cyclen | CBC |
|---|---|---|
| range | **620.6 – 628.9 EFPD** | **1339 – 1458 ppm** |

Both stay inside the window and the gate across the whole uncertainty box. This is the only
candidate in the table with that property.

---

## 8. Does a new lattice buy ≥ 0.005 FF?

| option | best FF_hot with contrast ≥ 0.026 | ΔFF vs E4 (1.1390) | verdict |
|---|---:|---:|---|
| realized paramA (37 types) | 1.1430 (T4) | **+0.004** (worse) | no |
| realized ga80 (36 types) | 1.1390 (E4) | 0 (the incumbent) | — |
| new lattice, **frozen** template Gd layouts only | 1.1657 | +0.027 (much worse) | no |
| new lattice, **already-authored** lat1600 pin maps (Y1/Y2/Y3 decks at a new enrichment) | 1.1394 | +0.000 | **no** |
| new lattice, **open 20-pin** layout `1:1;4:1;6:4` | **1.1208** | **−0.018** | **yes, 3.6× the threshold** |

(The 1.1208 is the surrogate ensemble, which measured 0.0014 low against DeCART on T3–T6; the
DeCART-equivalent is ~1.1222 and ΔFF ~ −0.017. Still 3.4× the threshold.)

The open 20-pin layout is the entire gain. Reusing an authored deck at a different enrichment
(the cheapest conceivable new lattice — the chain rewrites e1/e2 at realization time, no pin-map
work) buys **0.0015**, three times *below* the threshold: not worth a DeCART second.

`ΔF_r` by the fusion law is **−0.022**. For scale, the *pattern* lever on the same fuel is
1.5207 → 1.4636 = −0.057, and the contrast mistake in T5_T6 cost +0.21. The fuel lever is real
but it is the smallest of the three.

---

## 9. Cost of the new wave, priced honestly

Driver `5_RL\realize_lat1600.py --designs Z1,Z2 --snapshot-ok`.

1. **Snapshot first, non-negotiable.** `MAS_XSL`/`MAS_HFF` keep exactly one `.bak`
   (`lpopt/design/library.py:95-101`). The current `.bak` is today's lat1600 rebuild; a second
   rebuild rotates it and **destroys the only rollback to the 33-type world**. Re-snapshot
   `lib/`, `designs.json`, `registry.json`, `fuel_types_paramA.parquet` under a new suffix before
   anything else.
2. **DeCART wave** — 2 lattices, ~13 min wall (the 4-way Y1–Y4 wave was ~13 min);
   TotalBatcher ~1 min.
3. **Union rebuild**, ncomp 42 → 44. `build_master_library` refuses partials, so all 39 HGCs are
   re-read.
4. **Every packaged restart goes stale again.** Today's set is `P0_P1, Q1_Q2, Q7_Q8, T3_T4,
   T5_T6, T5_T6_f117` in `bases\`, plus the `T5_T6_f101` / `T5_T6_f81` core decks —
   **8 existing + 1 new pair = 9 bootstraps**. Budget from observed history, not from the
   RUNBOOK's optimistic 4 min: the first T3_T4 attempt ran **8,744 s and failed**
   (`5_RL\t3t4_rerun.log`, `MasterRunError status 4294967295`) before the cy1-capped retry
   succeeded. Plan **2–5 h** with a retry tail.
5. **Elite pool for the new cell**: the store has zero rows for `Z1_Z2`; `elite_frac 0.65`
   starves on an empty cell (the J5_J6 precedent). `fr_transfer.py --k 32` ≈ 1–2 h, then a merge
   kit with `LIBRARY_ID = "paramA"`.
6. **Production fuel table**: `data\store\fuel_types.parquet` must be re-ingested, 37 → 39 paramA
   rows, or the model layer cannot see Z1/Z2.
7. Then the min_fr campaign itself.

**Total: roughly one working day of box 104**, and it invalidates the four restarts created
today. Results already measured on those restarts (fr_arms B2/B3, the 32+32 transfers) stay
valid as *results*; the restarts do not stay valid as *inputs*.

Pin maps are **not** hand work — `realize_lat1600.py` authors the open layout itself with hard
guards (guide tubes frozen at 1/8 `(0,0),(3,3),(4,3),(4,4)`, zoning cells untouched, exact Gd
census). Base decks for the 20-pin family exist for every `gd_wt`:

| design | base template | frozen layout | pin moves needed |
|---|---|---|---|
| Z1 (68-role) u5.50/4.6750, gd_wt 8, n_gd 20 | `0_APR1400\5.8_5.1\FA\IGD_20\8_20_z1\dec_FA_B03.inp` | `2:2;5:2;6:4` | `2:2 → 1:1`, `5:2 → 4:1` (`6:4` stays) |
| Z2 (53-role, hot) u5.00/4.2500, gd_wt 10, n_gd 20 | `0_APR1400\5.8_5.1\FA\IGD_20\10_20_z1\dec_FA_B05.inp` | `2:2;5:2;6:4` | same two moves |

Both use the **same** target layout `1:1;4:1;6:4`, so it is one map edit applied to two base decks
— exactly the Y3/Y4 pattern. Use each design's own `{gd_wt}_{n_gd}_z1` directory: the UO2G carrier
density varies with `gd_wt` (6 → 10.01, 8 → 9.95, 10 → 9.88 g/cc) and `edit_dec_text` never
touches it (`decks\MANIFEST.md`, "Template-structure surprises" §1). Zoning in both base decks was
verified to be exactly the PB set (`s22_template20.py`).

---

## 10. Recommendation

**Do this first — it costs one MASTER bootstrap and nothing goes stale:**

```
python -m lpopt design bootstrap --input design_lat1600_104.inp --pair T6_T4 --feed 121
python -m lpopt design bootstrap --input design_lat1600_104.inp --pair T1_T4 --feed 117
```

`T6_T4@f121` (R2) is predicted at 612–630 EFPD / 1383 ppm / contrast +0.0747 — the highest
contrast of any realized pair, higher even than the reference E1_E2 (+0.049). `T1_T4@f117` (R1)
is predicted at 626–634 EFPD / 1283 ppm with **no reliance on the hump correction**. Between
them they (a) hand the program a campaign cell today and (b) test the model's single largest
extrapolation — the hump term at hump ≈ 0.026 — at zero DeCART cost. If R2 lands in window, its
`Fr_flr` is 1.423.

Caveat on R1/R3/R5: T1, P9 and P0 are 5.8–6.2 % enriched. That is inside what this package has
already run (Q1_Q2 and Q7_Q8 are 6.6 %) but outside the 5.00–5.50 band the lat1600 screen was
deliberately scoped to. Whether the campaign cell may use them is a program decision, not a
physics one. R2/R4 (T5/T6 + T4) stay entirely inside 5.00–5.30 %.

**Then, and only then, decide on the wave.** The case for it is `Z`: ΔFF_hot −0.018, ΔF_r −0.022,
CBC 1397 with 203 ppm of gate margin, robust across the whole calibration uncertainty box. The
case against it is that −0.022 in F_r is a third of what re-optimizing the pattern already
delivers on existing fuel, and the wave re-stales nine restarts for a day.

**Do not** re-run T5_T6 at any feed hoping the boron comes down. Its equilibrium CBC is 1706 ppm
even at f117 and it has no role contrast; both failures are structural, not tuning.

---

## 11. What would falsify this

* **The hump correction.** Fitted on humps ≤ 0.0148, mechanism understood but coefficient
  empirical. R2/R4 (hump 0.026) are the test. If `T6_T4@f121` measures near 612 rather than 630,
  the correction is too strong and every `X`-flagged row in §7 moves out of window — `Z` and R1
  do not, which is why they are the recommendation.
* **The contrast gate at 0.026.** Drawn from 15 fixed-pattern arms and corroborated by 64 elite
  transfers. It is a statement about *these* loading patterns. A pattern family optimized from
  scratch for a zero-contrast fuel set might not need it — but nothing in the store demonstrates
  that, and 32 elite patterns failed to.
* **`w_B` constancy.** One global 5.348 pcm/ppm across 1114–2244 ppm and two libraries. The
  Q-pair misses (−88, −152 ppm) hint at mild worth degradation above ~2000 ppm. Irrelevant in the
  1300–1500 ppm band the recommendations live in; would matter if the gate were ever relaxed.
* **`F_r_floor`.** It is a *floor*, not a forecast: it assumes the LP restores `node_peak` to
  1.2085. No lat1600 measurement has yet done so. Read it as an ordering key.

---

## Files

| file | what |
|---|---|
| `paths.py`, `opmodel.py` | shared paths; the equilibrium + CBC model |
| `measured.py` | all 79 MASTER-measured operating points, with sources |
| `s01_inventory.py` | package designs ↔ screen table cross-reference |
| `s02_surrogate_vs_decart.py` | surrogate vs DeCART k(BU), held out on T3–T6 |
| `s05_hgc.py` → `hgc_curves.npz` | k(BU) + FF(BU) for all 73 lattices, both packages |
| `s06_validate.py`, `s07_refine.py` | rho* / w_B calibration, residual structure |
| `s08_transfer.py` | surrogate→DeCART operating-point transfer; fusion-law test |
| `s09_frmodel.py`, `s09b_contrast.py` | why the fusion law fails; the contrast model |
| `s12_bias.py`, `s13_final.py` → `calib_final.npz` | the hump correction and its held-out test |
| `s17_variance.py` | flux-weighting alternative, rejected |
| `s10`/`s14_screen_final.py` → `screen_final_{121,117}.npz` | the full pairwise sweep (cache pruned to the CBC ≤ 1600 subset: 5,656,382 pairs at f121 and 1,566,160 at f117; re-run `s14` for the untrimmed set) |
| `s15_rank.py`, `s16_final_table.py`, `s21_top5.py` | ranked candidate tables |
| `s11`/`s19_realized_final.py` | all 1,369 realized paramA pairings |
| `s20_authored.py` | the zero-template-authoring subspace |
| `s22_template20.py` | frozen Gd layouts in the IGD_16/20/24 base decks |
| `s23_spec.py` | Z1/Z2 spec sheet + calibration sensitivity |
