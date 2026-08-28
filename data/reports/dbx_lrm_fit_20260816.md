# Calibrated-LRM backbone from the 6,113-core feasible database

**Date** 2026-08-16 · **Source (read-only)** `data/reports/scoping_mesh_20260815/feasible_database.xlsx`
(sheet `cores`, 6,113 rows × 39 cols; 20 `P_<pair>` lattice sheets) ·
**Compared against** `data/reports/scoping_mesh_20260815/mesh_nodes.csv` (55 surrogate cells)
**Machine-readable output** `data/reports/dbx_lrm_fit.csv` (61 rows × 60 cols)

Nothing in `data/store/` or the workbook was modified. This report is evidence for
`data/reports/reloadmap_methodology_20260816.md`, which is written by another agent and is
not touched here.

---

## 0. Headline

1. **The mass-balance identity holds to −0.519 % ± 0.098 %** on 6,099 of 6,113 cores —
   the previously quoted −0.52 % ± 0.10 % is confirmed exactly. 14 rows (0.23 %) carry a
   corrupt `core_mean_discharge_GWd` and are the entire source of the fat tail.
2. **The textbook LRM batch law is wrong by ±1.8 % on the feed axis.** With
   `n = 241/feed` the two-parameter LRM leaves a 4.06 EFPD rms and a *structured*
   ±8.9 EFPD per-cell bias that no re-scaling of the effective batch number can remove.
   Six free per-feed calibration factors reduce rms to 2.36 EFPD — at the database's own
   within-cell scatter floor.
3. **Featurize with the k-inf curve, not the enrichment label.** Swapping the core-average
   enrichment for the composition-weighted `bu_k1` (burnup at k∞ = 1, taken straight from
   the `P_*` sheets) cuts rms a further 9 % to **2.16 EFPD** with the *same* parameter
   count, and the optimum critical k∞ lands on 1.000 — the LRM reactivity descriptor is
   literally `bu_k1`.
4. **The surrogate mesh's "gate-free ceiling" is not a ceiling.** `ceil_cyclen` sits
   *below* the database's own realised maximum EFPD in **24 of the 30 shared cells**
   (mean −8.2, worst −22.5 EFPD at `e5.3_f109`), while in the extrapolation band it runs
   *above* the LRM backbone by up to **+36.6 EFPD (+4.8 %) at `e6.0_f121`**. The surrogate
   under-calls where we have data and over-calls where we do not.

---

## 1. Conventions

| quantity | value | provenance |
|---|---|---|
| core assemblies `N` | 241 | `n_type1 + n_type2 == feed` in every row, and `M_HM_tU / per_fa_tU = 241.0` in `mesh_nodes.csv` |
| `per_fa_tU` (ga80) | 0.42337 | `mesh_nodes.csv` |
| `M_HM` | 102.031 tU | " |
| thermal power | 3983.0 MW | back-solved from `ceil_B_cycle / ceil_cyclen × M_HM` |
| specific rate `spr` | 0.03903716 GWd/tU per EFPD | `P / (M_HM · 1000)` |
| cycle burnup | `B_c = EFPD · spr` | |
| discharge burnup | `B_d = B_c · 241 / feed` | closed mass balance |
| batch number | `n = 241 / feed` (2.386 … 1.992) | |

`assembly_composition` describes the **feed batch only** — `n_type1 + n_type2` equals
`feed` in all 6,113 rows, never 241. Every core in the database is a two-type feed.

## 2. Mass balance `B_d = B_c · 241 / feed`

Deviation `100 · (B_d_massbal / core_mean_discharge_GWd − 1)`:

| subset | n | mean | sd | median | MAD | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| all rows | 6113 | −0.494 % | 0.662 % | −0.524 % | 0.048 % | −0.883 % | +27.63 % |
| `metrics_source = equilibrium_ncyc12` | 5127 | −0.494 % | 0.720 % | −0.525 % | 0.040 % | −0.883 % | +27.63 % |
| `metrics_source = campaign` | 986 | −0.499 % | 0.153 % | −0.505 % | 0.121 % | −0.839 % | −0.053 % |
| **all, excluding \|dev\| > 1 %** | **6099** | **−0.519 %** | **0.098 %** | — | — | −0.883 % | +0.44 % |

**The 14 outliers are a data defect, not a physics one.** All are `equilibrium_ncyc12`
rows whose `core_mean_discharge_GWd` is 42–50 GWd/tU where the mass balance demands
52–58 — i.e. the reported discharge burnup, not the cycle length, is wrong:

```
cid           pair   feed  EFPD    reported B_d   dev%    ncyc
e549a4545bfe  H3_H4  117   675.7   42.57         +27.63    8
fb79709e8396  K1_K2  101   581.5   43.33         +25.01    8
7712e1e0adfa  L1_L2  117   649.2   42.44         +23.00    8
42edbc15000f  J1_J2  109   608.9   46.63         +12.71   10
3d4a1c6110bc  E1_E2  113   612.6   46.10         +10.64    8
… 9 more between +2.2 % and +8.5 %
```

The 0.52 % offset itself is systematic and one-signed: the mass balance slightly
*over*-states discharge burnup because a fraction of the feed sits an extra cycle at the
`241 mod feed` remainder positions. Treat 0.52 % as a fixed correction, not noise.

## 3. Backbone model family and fit quality

All models predict cycle length; residual = prediction − database EFPD, over all 6,113 cores.

| model | form | k | rms (EFPD) | bias | p95 \|r\| | max \|r\| |
|---|---|---:|---:|---:|---:|---:|
| **A** analytic LRM | `EFPD = 2·B₁(e)/((n+1)·spr)`, `n = 241/feed`, `B₁ = a + b·e` | 2 | **4.061** | +0.02 | 7.70 | 12.31 |
| **B** feed-calibrated | `EFPD = α_F · (a + b·e)` | 7 | **2.364** | 0.00 | 4.68 | 12.50 |
| **C** feed-calibrated, k-curve | `EFPD = α_F · (a + b·bu_k1_mix)` | 7 | **2.159** | 0.00 | 4.27 | 12.29 |
| **D** both descriptors | `EFPD = α_F · (a + b·e + c·bu_k1_mix)` | 8 | **1.966** | 0.00 | 3.95 | 11.97 |
| *floor* — 36 cell means | saturated on the (segment, feed) grid | 36 | 2.627 | 0 | 4.84 | 18.15 |
| *floor* — cell × pair means | saturated on (segment, feed, pair) | 89+ | 1.678 | 0 | 3.34 | 9.26 |

Two things to read off this table.

*Models B–D beat the 36-cell-mean "floor".* That is not overfitting: `realized_enrichment`
and `bu_k1_mix` vary **within** a cell with the composition split, and a smooth 7-parameter
physics form exploits that where a per-cell constant cannot.

*Model D is within 17 % of the cell×pair floor* — i.e. almost everything left is
loading-pattern freedom, which no `(e, feed)` backbone can or should predict.

Fitted coefficients (also carried as columns in `dbx_lrm_fit.csv` so the CSV is
self-contained):

```
A:  B₁[GWd/tU] = −0.8580 + 7.5608·e            n = 241/feed
B:  EFPD = α_F · (−14.8543 + 126.1321·e)
    α = {101: 0.903213, 105: 0.931299, 109: 0.967220,
         113: 0.987078, 117: 1.003856, 121: 1.019689}
C:  EFPD = α_F · (20.7118 + 7.0967·bu_k1_mix)
    α = {101: 1.868727, 105: 1.926326, 109: 2.001895,
         113: 2.043172, 117: 2.077713, 121: 2.110541}
ceiling backbone (fit to the 36 per-cell MAX EFPD, rms 1.27 EFPD):
    EFPD_ceil = α_F · (3.3693 + 122.2393·e)
    α = {101: 0.912776, 105: 0.941476, 109: 0.973457,
         113: 0.994604, 117: 1.014169, 121: 1.027840}
```

## 4. Why the analytic LRM fails on the feed axis

The LRM predicts `α_F ∝ 2/((n+1)·spr)`. Normalising both to feed 109:

| feed | n = 241/feed | α observed | α analytic LRM | LRM − observed |
|---:|---:|---:|---:|---:|
| 101 | 2.3861 | 0.93382 | 0.94828 | **+1.55 %** |
| 105 | 2.2952 | 0.96286 | 0.97444 | +1.20 % |
| 109 | 2.2110 | 1.00000 | 1.00000 | 0 |
| 113 | 2.1327 | 1.02053 | 1.02498 | +0.44 % |
| 117 | 2.0598 | 1.03788 | 1.04941 | +1.11 % |
| 121 | 1.9917 | 1.05425 | 1.07329 | **+1.81 %** |

The analytic law is too *wide*: it stretches cycle length across the batch axis by
1–2 % more than the data does. Two candidate repairs were tested and both fail:

* **Free effective core size** `n = N_eff/feed` (a 1-parameter compression): a grid scan
  over `N_eff ∈ [120, 420]` bottoms out at `N_eff = 250.5` with rms 4.048 — a 0.3 %
  improvement on model A. The deficiency is in the *shape* of the `1/(n+1)` law, not its
  scale.
* **Polynomial in n** `B₁ = a + b·e + c·n + d·n²` (4 params): rms 2.806. Better, but still
  1.3× model B at fewer effective degrees of freedom on the feed axis.

**The f109 ridge.** Feed 109 sits ~1.3 % (≈ 8 EFPD) *above* any smooth curve drawn
through the other five feeds, in every one of the 20 pairs except `E1_E2`. Model A's
per-feed bias, which is the cleanest way to see it:

```
feed   101     105     109     113     117     121
bias  +2.94   +0.93   −6.46   −3.85   +0.42   +4.99   EFPD
rms    3.85    2.79    6.84    4.21    2.65    5.44
```

This is a genuine feature of the database (present at pair level, all segments, both
metrics sources), not a fit artefact. It is *not* explained by batch remainder
(`241 mod 109 = 23`, which is between the 105 and 113 remainders), by `ncyc`, or by the
equilibrium/campaign mix. **Any backbone used for outer-loop screening must carry a free
per-feed factor rather than the analytic `2/(n+1)`.** That is the single most transferable
result in this report.

A k∞-curve-based equilibrium solve was also tried — batch-average
`Σ wⱼ k_mix(j·B_c) = k_crit` with the real 62-point `P_*` curves, solved per composition.
With a single constant `k_crit` it lands at **8.04 EFPD rms** (`k_crit = 1.0134`); it only
reaches 3.30 once `k_crit` is allowed a quadratic dependence on `n`, which is just the
same per-feed calibration wearing a physics costume. The equal-power batch-averaging
assumption, not the reactivity data, is what breaks.

## 5. `bu_k1` beats enrichment as the fuel descriptor

Model C replaces `realized_enrichment` with `bu_k1_mix`, the feed-count-weighted burnup at
which the pair's k∞ crosses 1, read off the `P_*` depletion tables. Same parameter count,
9 % lower rms. Two corroborations:

* Fitting `EFPD = α_F · (a + b·B₁(k_crit))` where `B₁(k_crit)` is the burnup at which the
  mixed curve crosses a *free* `k_crit` recovers **`k_crit = 1.000`** and rms 2.158 —
  indistinguishable from model C. The LRM's single-batch burnup capability *is* `bu_k1`.
* The two descriptors are not redundant (model D, using both, gains another 9 %):
  enrichment carries the fissile inventory, `bu_k1` carries the Gd design and zoning that
  the enrichment label discards.

This is the direct argument for deliverable 2: the store's `ga80` rows carry
`u_avg_enrichment` = the *family anchor* (5.0, 5.1, …) and NaN for every zoning/Gd column,
so today's featurizer sees only the weakest of the two descriptors, and cannot see the four
`P_*` types it has no row for at all.

## 6. Residual systematics

**By segment** (model C): bias between −0.69 and +0.48 EFPD, rms 1.59 (e5.4) to 3.28
(e5.0). The 5.0 segment is the worst — it is also the segment with the widest realized
enrichment spread (sd 0.013) and the only one containing the deckless `E5_E6` pair.

**By feed** (model C): bias ≤ 0.03 EFPD everywhere by construction; rms 1.56 (f113) to
2.44 (f105).

**By pair** (model C, the systematics the backbone cannot see):

| pair | n | bias EFPD | rms |
|---|---:|---:|---:|
| `E5_E6` | 56 | **+5.61** | 6.71 |
| `J1_J2` | 452 | +2.36 | 2.91 |
| `J5_J6` | 122 | +1.65 | 1.94 |
| … 15 pairs within ±1.5 … | | | |
| `E1_E2` | 509 | −1.37 | 2.81 |
| `J7_J8` | 416 | −2.02 | 2.86 |
| `J3_J4` | 157 | −2.16 | 2.65 |

`E5_E6` is by far the worst-fit pair: the backbone over-predicts its cycle length by
5.6 EFPD. Its `bu_k1_mix` is only 0.4 GWd/tU below `E1_E2`'s, but its realised cycles run
~11 EFPD shorter at matched feed. **The two pairs the store cannot featurize at all
(`E5_E6`, `J7_J8`) are also the two the backbone predicts worst** — they carry information
no current feature captures, which is an argument for both the yaml merge and a campaign
there.

Worst per-cell rms (model C): `e5.0_f109` 4.93, `e5.0_f105` 4.80, `e5.1_f117` 3.62.
Median per-cell rms 1.71; model A's median is 4.30 with a worst-cell 8.91.

## 7. Surrogate ceiling vs LRM-from-real-data

`mesh_nodes.csv` `ceil_cyclen` is the surrogate's gate-free maximum predicted cycle length
per cell (`= max_pred_cyclen_any`, no safety-factor filter). The like-for-like LRM object
is the **ceiling backbone** of §3, fitted to the 36 per-cell *maximum* observed EFPD
(rms 1.27 EFPD — the DB's per-cell ceiling is remarkably smooth).

### 7.1 Inside the database footprint (30 shared cells)

`ceil_cyclen < EFPD_db_max` in **24 of 30** cells — the surrogate "ceiling" is beaten by
cores that actually exist. Mean gap −8.2 EFPD, worst −22.5.

Largest disagreements (negative = surrogate under-calls):

| cell | DB max EFPD | LRM ceiling | `ceil_cyclen` | Δ vs LRM | Δ vs DB max | Δ B_d |
|---|---:|---:|---:|---:|---:|---:|
| `e5.3_f109` | 637.1 | 635.8 | 614.6 | **−21.20** | −22.51 | −1.83 |
| `e5.3_f105` | 613.4 | 613.6 | 594.5 | **−19.14** | −18.94 | −1.71 |
| `e5.5_f109` | 662.1 | 659.9 | 643.6 | −16.31 | −18.52 | −1.41 |
| `e5.2_f109` | 625.1 | 625.2 | 609.4 | −15.75 | −15.68 | −1.36 |
| `e5.3_f113` | 650.5 | 649.5 | 634.0 | −15.51 | −16.51 | −1.29 |
| `e5.4_f105` | 627.6 | 627.3 | 611.8 | −15.49 | −15.76 | −1.39 |
| `e5.4_f109` | 649.9 | 648.5 | 633.9 | −14.57 | −16.02 | −1.26 |
| `e5.3_f117` | 661.3 | 662.2 | 649.3 | −12.93 | −12.01 | −1.04 |
| `e5.1_f109` | 612.2 | 612.4 | 599.7 | −12.74 | −12.54 | −1.10 |
| `e5.5_f105` | 639.6 | 637.9 | 626.1 | −11.81 | −13.51 | −1.06 |

The pattern is legible: the surrogate's deficit **grows with enrichment and shrinks with
feed**. It is worst along the e5.3 row (−12.9 to −21.2 at f105–f117) and collapses at the
`f121` column (e5.3_f121 only −2.0), where it flips positive at low enrichment
(`e5.0_f121` +5.4, `e5.5_f121` +6.7, `e5.1_f121` +3.7). `f121` is precisely the column the
mesh flags `in_distribution = True`; every other column is extrapolation for the surrogate
and it extrapolates *downward*.

The `f109` column carries the four largest deficits. That is the same f109 ridge from §4 —
the surrogate has not learned it either, so the two independent objects agree that f109 is
where the model's picture of the feed axis is weakest.

### 7.2 Outside the database footprint (25 mesh-only cells, e5.6–6.0)

Here the LRM must extrapolate 0.1–0.5 w/o beyond its calibration range, **and** the mesh
switches library (`ga80` → `paramA` at e ≥ 5.6, with `per_fa_tU` moving 0.42216–0.42408).
Both numbers are soft. The sign of the disagreement nevertheless reverses:

| cell | LRM ceiling (extrap.) | `ceil_cyclen` | Δ | Δ % |
|---|---:|---:|---:|---:|
| `e6.0_f121` | 759.7 | 796.4 | **+36.62** | +4.82 % |
| `e6.0_f117` | 749.6 | 779.3 | +29.66 | +3.96 % |
| `e6.0_f113` | 735.2 | 759.9 | +24.70 | +3.36 % |
| `e6.0_f105` | 695.9 | 716.4 | +20.51 | +2.95 % |
| `e6.0_f109` | 719.5 | 738.0 | +18.42 | +2.56 % |
| e5.6–5.9 (20 cells) | — | — | −13.9 … +10.6 | −2.1 % … +1.4 % |

**The whole e6.0 row is a discontinuity in the mesh itself.** The e5.9 → e6.0 step is
+38 to +43 EFPD for a 0.1 w/o increment, while e5.8 → e5.9 is +4.7 to +8.1 and the LRM
slope says ≈ +11.5. The e6.0 row is ~3.5× the LRM expectation and ~5× the previous mesh
step. Whatever produced it (library change, extrapolation blow-up, or a genuine paramA
design difference), **e6.0 should not be read off the map without a MASTER anchor.**

### 7.3 Feed 101 — six DB cells the mesh does not have

The mesh grid is feeds 105–121; the database also covers 101. The LRM ceiling backbone
extrapolates onto them cleanly (it was fitted with f101 included, so this is interpolation
in `e` only):

| cell | DB mean | DB max | LRM ceiling | B_d | best DB F_r |
|---|---:|---:|---:|---:|---:|
| `e5.0_f101` | 562.8 | 565.7 | 565.4 | 52.4 | 1.53 |
| `e5.1_f101` | 568.2 | 573.2 | 573.1 | 52.9 | 1.50 |
| `e5.2_f101` | 581.9 | 586.4 | 587.2 | 54.2 | 1.51 |
| `e5.3_f101` | 591.3 | 595.4 | 596.2 | 55.1 | 1.51 |
| `e5.4_f101` | 605.6 | 610.7 | 608.4 | 56.4 | 1.50 |
| `e5.5_f101` | 615.6 | 620.6 | 617.8 | 57.3 | 1.51 |

Adding feed 101 to the model mesh costs nothing — the backbone already predicts it to
better than 3 EFPD — and it is where the database's flattest cores live (F_r down to
1.496, see `dbx_frontier_table.csv`). The trade is discharge burnup: f101 pushes B_d to
52–57 GWd/tU and every f101 core in the database exceeds 75 GWd/tU node pin burnup.

## 8. What this feeds into the hybrid prescription

1. **Use a per-feed-calibrated LRM as the outer-loop backbone, never the analytic
   `2/(n+1)`.** Six numbers bought a 1.7× rms reduction and removed an ±8.9 EFPD
   structured bias. The coefficients are in `dbx_lrm_fit.csv`
   (`lrmB_a`, `lrmB_b`, `lrmB_alpha`, `lrmCeil_*`).
2. **Feed the backbone `bu_k1`, not the enrichment label** — same cost, better fit, and it
   generalises to fuel types the enrichment anchor collapses together.
3. **Do not treat `ceil_cyclen` as an upper bound inside e5.1–5.5.** It is beaten by real
   cores in 24/30 cells. Re-anchor the mesh to the LRM ceiling backbone there, and reserve
   the surrogate for the F_r / F_q / CBC axes it was actually trained on.
4. **Quarantine the e6.0 row** until a MASTER point exists; the mesh's own e5.9 → e6.0 step
   is internally inconsistent by a factor of ~4.
5. **Pin the 0.52 % mass-balance offset as a constant**, and drop the 14 corrupt
   `core_mean_discharge_GWd` rows from any fit that uses the reported discharge burnup.

## 9. `dbx_lrm_fit.csv` schema

61 rows = union of the 36 database cells and the 55 mesh cells (30 shared). Column
`zone ∈ {db_and_mesh (30), db_only (6, feed 101), mesh_only (25, e ≥ 5.6)}`;
`extrapolated = True` marks the 25 cells where the LRM is extrapolating in enrichment.

| group | columns |
|---|---|
| identity | `cell zone enrichment_segment feed n_batch extrapolated` |
| database | `n_cores n_pairs n_feasible_db e_real_mean e_real_sd bu_k1_mix_mean EFPD_db_{mean,sd,min,max} B_c_db_mean B_d_massbal_mean B_d_reported_mean massbal_dev_pct_{mean,sd} F_r_db_min CBC_max_db_max` |
| fit quality | `lrm{A,B,C}_efpd lrm{A,B,C}_bias lrm{A,B,C}_rms` (per-cell, model-A/B/C) |
| backbone | `e_for_backbone backbone_{mean,ceil}_efpd backbone_{mean,ceil}_B_d` |
| surrogate | `ceil_cyclen ceil_B_cycle ceil_B_d min_pred_f_r n_feasible in_distribution n_store_pair_feed library_id per_fa_tU` |
| disagreement | `d_surrogate_minus_lrmceil_efpd d_surrogate_minus_lrmceil_pct d_surrogate_minus_lrmmean_efpd d_surrogate_minus_dbmax_efpd d_surrogate_minus_lrmceil_B_d` |
| coefficients | `lrmA_a lrmA_b lrmB_a lrmB_b lrmB_alpha lrmCeil_a lrmCeil_b lrmCeil_alpha spr_gwd_per_efpd` (constant per row, so the file is self-contained) |

## 10. Caveats

* The database is a **gate-filtered sample** (`F_r ≤ 1.55` etc.), so its per-cell mean is
  the mean of *acceptable* cores, not of all cores. The mean backbone (model B) inherits
  that selection; the ceiling backbone does not (a maximum is a maximum). Use the ceiling
  backbone for any comparison against `ceil_cyclen`.
* Cells are unevenly populated (27 to 482 cores). Fits are per-core, so heavy cells
  dominate; the per-cell tables in the CSV let you re-weight.
* `e5.5_f109` rests on 27 cores and `e5.0_f101` on 56 — the two thinnest cells.
* `EFPD` for the 986 `metrics_source = campaign` rows is a campaign cycle length, not the
  ncyc-12 equilibrium value; they are pooled here because their mass-balance deviation is
  statistically identical (−0.499 % vs −0.494 %). They carry no pin-burnup columns.
* The e ≥ 5.6 mesh cells use library `paramA`, not `ga80`. The LRM extrapolation there is
  a ga80 curve evaluated off its own library; read the sign of the disagreement, not its
  magnitude.
