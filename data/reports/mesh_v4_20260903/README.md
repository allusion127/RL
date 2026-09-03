# mesh v4 — 25-cell scoping mesh under champion s1j (2026-09-03)

Model-only. **No MASTER was executed and nothing under `data/store` was written.**
All compute ran on **HOST_198** (`USER@HOST_198`, DESKTOP_HOST,
i9-13900 24C/32T, CPU only — `torch.cuda.is_available() == False`), in
`C:\Users\USER\lpopt_work\kit_frontier`. The local PC ran no computation.

## 1. What this is, and what changed from v3

| | mesh_v3_20260817 | **mesh_v4_20260903** |
|---|---|---|
| champion | s1g (cond v6b, 8 targets) | **s1j** (cond v8, 9 targets, f_xy head) |
| grid | 16 e-levels × 6 feeds = 96 (90 computed) | **5 × 5 = 25** |
| store | 74,477 rows | **76,793 rows / 67,880 converged** |
| lpopt | pre-`serve_provenance` | post-fix (2026-08-29) + s1j refit per-cell calibrations |
| readouts | node (max cyclen among F_r-feasible) | node **+ λ-optimum + F_xy + pin grade** |
| overlays | DB xlsx q95, LRM (v3 grid) | **measured store overlay + LRM frozen *and* refit** |

### 1a. THE GRID WAS UNDEFINED — what was used, and why

No 25-cell grid existed anywhere in the repo (grepped every `data/reports/*.md`
and every mesh script; the only grids that exist are v2's 55 and v3's 96/90).
The documented default adopted here is the **5×5 production grid of
`data/reports/feedgrid_pathfinder_20260815.md` §8** — the only 5×5 feed ×
enrichment lattice the programme has ever written down, and the one with real
converged MASTER support in every cell:

* `E_TARGETS = 5.0, 5.2, 5.4, 5.5, 5.9`  (pathfinder pairs E1_E2 5.00, K1_K2
  5.20, N1_N2 5.40, G3_G4 5.50, and the ~5.9 paramA pair)
* `FEEDS = 109, 113, 117, 125, 129`

Both are plain comma-separated CLI arguments (`scoping_mesh.py:497-501`), so **no
source change was needed**. Two consequences worth stating up front:

1. **Every one of the 25 cells is also a mesh_v3 cell**, so a v3→v4 delta exists
   for all 25 — that is why this grid was preferred over
   `dbx_lrm_fit.csv`'s 25 `zone=='mesh_only'` cells (which sit at f105, a feed
   v3 never computed).
2. The grid contains **f125 and f129, where the LRM α is extrapolated**, and it
   omits f121 — so no cell carries `in_distribution = True`. The α
   extrapolation is therefore load-bearing here, which is why both the frozen
   and a store-refitted LRM are reported (§5).

### 1b. The pair picks reproduce v3 exactly

`scoping_mesh.select_pairs` is called with its OWN full 16-target grid (the v4
subset is applied after the `is_primary` collapse, exactly as
`scoping_mesh.main --e-targets` does), so the pick rule and its tie-breaks are
byte-identical to v3's. Verified: **all 16 pairs and all 16 `e_core` values are
identical to `mesh_v3_20260817/pair_selection.csv`**. Only the `sup_pair` row
counts moved with the store (E1_E2 2744→3263, N1_N2 1273→1491,
P6253Z1G06N24_P6253Z1G08N20 708→748) — no tie flipped.

## 2. Recipe and seeds — identical to v3, by construction not by copy

`scratch_mesh_v4/mesh_v4_run.py` does **not re-implement** the search. It calls
`scoping_mesh.run_cell` unmodified and *records* what that call produced, by
wrapping `lpopt.search.construct.build_pool` and `model.predict` with pass-through
recorders. Every recorded pool is asserted against `run_cell`'s own
`n_candidates` / `n_in_band` / `n_feasible` / `node_record_id`, so a recorder
desync is a hard failure, not a silent divergence.

Therefore, unchanged from v3 and re-stated from `scoping_mesh.py` itself:

* RNG seed per cell `random.Random(SEED + feed*13 + round(e_core*100))`, `SEED = 20260815`
* two waves, `POOL_W0 = 800` then `POOL_W1 = 400` seeded with the top-24 feasible
  and the top-24 in-band near-misses → 1,200 candidates/cell
* elites `N_ELITE_BAND = 32` (half flattest band rows nearest in feed, half
  longest feasible) + `N_ELITE_DONOR = 12` globally flattest store-feasible donors,
  batch-remapped by `fr_transfer.pair_mapping/substitute`
* `E_CORE_TOL = 0.02`, `MAX_PAIR_SPREAD = 0.25`
* Tier-1 gate `F_r ≤ 1.55, F_q ≤ 2.41, CBC_max ≤ 1600, |AO| ≤ 0.30`; tiers
  `tier2 (1.65, 1800)`, `tier3 (1.80, 2200)`
* quantile heads off (`model.quantile_targets = ()`) — mean and epistemic σ are
  byte-identical with them on

**Reproducibility caveat (stated, not hidden):** the recipe and the seeds are
identical, but two inputs are NOT the same object v3 saw — the champion (s1g→s1j)
and the store (74,477→76,793 rows, plus the 2026-08-29 `e_core` backfill). The
store enters through `build_elites`' band, the donor set and the `n_store_*`
columns, so v4 is a *re-run of the same recipe on new inputs*, not a bit-for-bit
replay of v3. The pair picks (§1b) are the part that is verifiably unchanged.

## 3. The three added readouts

### 3a. λ-optimum on the (cyclen, F_r) frontier

`readout_axis.F_R_AXIS` — `objective min_fr_max_cycle → axis F_r (limit 1.55,
λ = 1000 EFPD/unit; from default)` — scored as `cyclen − 1000·F_r` over **the same
predicted-feasible set the node is drawn from**. The λ-optimum is always a point
of the Pareto front (a linear scalarisation optimum is non-dominated), so it is
reported both as its own `lam_*` columns and flagged on `mesh_v4_pareto.csv`
(`is_lam_opt`). `lam_is_node` records whether the registered λ reading and the
max-cyclen node are the same core — the check the programme requires precisely
because the F_r-only headline has been overturned by it before.

### 3b. F_xy — head mean, G4 σ barred

s1j is the first champion with a **direct f_xy head** (`target_idx 8`, mode
`direct`). It is read through the canonical accessor
`lpopt.search.acquisition.predict_fxy`, which returns the head's own MEAN with
`source = "head"` and then applies the **G4 sigma bar**: `ensemble.json`'s
`fxy_head.serve_sigma = "barred"` (G4 FAIL, 68 % coverage 0.831 > 0.80) is live on
the shipped checkpoint (`fxy_sigma_barred = True`, verified in the run log of
every shard), so the σ served is **not** the head's — it is the interim proxy
width `sqrt((1.2176·σ_Fr)² + (3.0·0.0476)²)`. Every `*_f_xy_sigma` column in the
outputs is that proxy width. Conformal intervals deliberately do not cover f_xy.

**The frontier and the gate remain F_r.** `scoping_mesh.py` still refuses
`--objective min_fxy` (prerequisite 2 — the F_r-named `mesh_nodes.csv` schema
consumed verbatim by `mesh_vs_db.py`, `scoping_mesh_fig.py` and `autoeng.py` — is
still open), and that refusal was respected: F_xy here is an **additive advisory
column**, never the ranking axis. `n_feasible_fxygate` (F_xy ≤ 1.65 substituted
for F_r ≤ 1.55, the other three Tier-1 constraints unchanged) is published as a
diagnostic of what an F_xy-gated map *would* look like — it is not a claim that
F_xy was optimised.

The F_xy pass is batched at 400 patterns. Verified against an un-batched run of
the same cell: identical to 8 decimal places on every column except timings
(`node_pred_f_xy` 1.616881772 vs 1.616881780 — float reduction order), and
byte-identical on every F_r-side value.

### 3c. Pin-BU grade

Surrogate column 6 (`max_pin_burnup`) at the node and at the λ-optimum, graded on
the style spec 20260817 §5 scale — `green < 62`, `yellow 62–68`, `orange 68–75`,
`red ≥ 75`, `미산출` when there is no prediction. **This is a grade scale, not the
delivery gate**: the pin limit is 80 GWd/tU on the pin axial peak
(`pinbu_definition_20260820.md` §3) and pin BU is advisory on the design map — it
never closes a cell. No campaign-measured `PIN_OVERRIDE` values were injected;
every grade here is the model's own prediction, which is the honest reading for a
model-only mesh.

## 4. Measured overlay (`db_*` columns, `mesh_v4_population.csv`)

Source is the **shipped store**, not the 2026-08-15 `feasible_database.xlsx` — the
xlsx carries no loading pattern and its `cid` resolves to nothing in `data/store`
(0/6113), so the model cannot be run on those cores.

* population = `valid == True & converged == True`
* **joint-clean** = Tier-1 (`F_r ≤ 1.55, F_q ≤ 2.41, CBC ≤ 1600, |AO| ≤ 0.30`)
  **AND `F_xy ≤ 1.65` where labelled.** Rows with no measured F_xy are kept and
  counted (`db_n_fxy_labelled`), never silently dropped — `readout_axis`'s rule
  that an unlabelled row is excluded from an F_xy frontier but must be reported,
  not deleted.
* per cell, scope level 1 = `(library_id, case_pair, feed)`; level 2 fallback =
  `(feed, |e_core − pair e_core| ≤ 0.05)`, recorded in `db_scope`.
* `db_best_cyclen` = max cyclen among joint-clean rows; `db_best_f_r` = min F_r
  among them; `db_lam_*` = the same λ reading as §3a applied to measured rows.

`mesh_v4_population.csv` is the whole-population scatter (every converged & valid
store row, with `joint_clean` flagged) — the background cloud the style spec
draws at alpha 0.10.

## 5. LRM — frozen constants AND a refit on the current store

Two objects, both reported, never mixed:

* **FROZEN**: `data/reports/dbx_lrm_fit.csv`, the DB-calibrated backbone
  `EFPD = α_F·(a + b·e)` fitted to the 6,113-core feasible database.
  mean `a = −14.854288, b = +126.132069`; ceiling `a = +3.369285, b = +122.239281`.
  α is fitted for feeds 101–121 and **extrapolated** to 125/129 by
  `lrm_mesh.fit_alpha` under PREREG §1b (form chosen by smallest max deviation
  from the store's own within-pair q95 feed ratio, *not* by LOO). The
  adjudication was re-run on the CURRENT store and both the chosen form and the
  delta from the frozen 2026-08-17 α values are printed in `post_v4.log`.
* **REFIT on the current store**: the same functional form fitted to
  `q95(cyclen)` (ceiling) and `mean(cyclen)` (mean) per `(library_id, case_pair,
  feed)` group with `n ≥ 25`, gauge `α_117 = 1`. This one **fits f125/f129
  rather than extrapolating to them**, which is its main value on this grid.

The LRM is evaluated at the pair's **realised `e_core`**, not the e-target label.
Cells at `e > 5.5` or `feed > 121` carry `e_extrapolated` / `feed_extrapolated`.

**Accuracy caveat, quoted from the programme's own methodology memo:** the LRM
skeleton's cyclen residual is ~4.26 EFPD rms (0.66 %) and does NOT meet the
programme acceptance bar (≤ 1 EFPD). `reloadmap_methodology_20260816.md` §P1 states
the map's job is 설계공간의 형상 제시 — showing the shape of the design space — not
optimum prediction. The LRM columns here are a backbone, not a competing predictor.

## 6. Results

Full tables: `summary_v4.txt` (sections A–G) and `post_v4.log`. Headline below.

### 6a. Per-cell table

`cyclen`/`F_r`/`F_xy`/pin are the **node** (max predicted cyclen among the
predicted-Tier-1-feasible pool); `db_*` are measured, joint-clean store rows;
`lrm_ceil` is the frozen ceiling backbone, `lrm_refit` the store-refit ceiling.
`—` = no feasible core / no joint-clean measured row.

| cell | feed | e_core | pair | cyclen_pred | f_r_pred | f_xy_pred | pin_grade | db_best_cyclen | db_best_f_r | lrm_cyclen | lrm_refit_cyclen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e5.0_f109 | 109 | 5.0000 | E1_E2 | 595.20 | 1.5494 | 1.6520 | red | 594.67 | 1.4787 | 598.25 | 581.12 |
| e5.0_f113 | 113 | 5.0000 | E1_E2 | 607.24 | 1.5436 | 1.6471 | red | — | — | 611.25 | 595.27 |
| e5.0_f117 | 117 | 5.0000 | E1_E2 | 628.50 | 1.5283 | 1.6169 | red | 625.87 | 1.5396 | 623.27 | 615.07 |
| e5.0_f125 | 125 | 5.0000 | E1_E2 | 647.51 | 1.5452 | 1.6353 | orange | 645.60 | 1.5085 | 648.60 | 640.98 |
| e5.0_f129 | 129 | 5.0000 | E1_E2 | 654.43 | 1.5442 | 1.6472 | orange | — | — | 661.12 | 647.48 |
| e5.2_f109 | 109 | 5.2000 | K1_K2 | — | — | — | 미산출 | — | — | 622.05 | 606.29 |
| e5.2_f113 | 113 | 5.2000 | K1_K2 | 619.25 | 1.5351 | 1.6343 | red | — | — | 635.57 | 621.06 |
| e5.2_f117 | 117 | 5.2000 | K1_K2 | 646.54 | 1.5482 | 1.6507 | orange | 645.34 | 1.5264 | 648.07 | 641.71 |
| e5.2_f125 | 125 | 5.2000 | K1_K2 | 667.46 | 1.5500 | 1.6404 | orange | 668.41 | 1.5194 | 674.41 | 668.74 |
| e5.2_f129 | 129 | 5.2000 | K1_K2 | 672.67 | 1.5490 | 1.6319 | orange | — | — | 687.42 | 675.53 |
| e5.4_f109 | 109 | 5.4000 | N1_N2 | — | — | — | 미산출 | — | — | 645.85 | 631.46 |
| e5.4_f113 | 113 | 5.4000 | N1_N2 | 647.96 | 1.5338 | 1.6041 | red | 646.39 | 1.4653 | 659.88 | 646.84 |
| e5.4_f117 | 117 | 5.4000 | N1_N2 | 657.09 | 1.5491 | 1.6327 | red | — | — | 672.86 | 668.35 |
| e5.4_f125 | 125 | 5.4000 | N1_N2 | 689.33 | 1.5466 | 1.6299 | orange | 690.29 | 1.5176 | 700.21 | 696.50 |
| e5.4_f129 | 129 | 5.4000 | N1_N2 | 693.38 | 1.5427 | 1.6269 | orange | — | — | 713.72 | 703.57 |
| e5.5_f109 | 109 | 5.5000 | G3_G4 | — | — | — | 미산출 | — | — | 657.75 | 644.05 |
| e5.5_f113 | 113 | 5.5000 | G3_G4 | 656.94 | 1.5452 | 1.6233 | red | — | — | 672.04 | 659.73 |
| e5.5_f117 | 117 | 5.5000 | G3_G4 | 668.19 | 1.5480 | 1.6320 | red | — | — | 685.26 | 681.67 |
| e5.5_f125 | 125 | 5.5000 | G3_G4 | 702.05 | 1.5487 | 1.6385 | orange | 700.31 | 1.5323 | 713.11 | 710.38 |
| e5.5_f129 | 129 | 5.5000 | G3_G4 | — | — | — | 미산출 | — | — | 726.87 | 717.59 |
| e5.9_f109 | 109 | 5.9087 | P6253Z1G08N20_P6257Z1G10N12 | — | — | — | 미산출 | — | — | 706.38 | 695.48 |
| e5.9_f113 | 113 | 5.9087 | ″ | — | — | — | 미산출 | — | — | 721.73 | 712.42 |
| e5.9_f117 | 117 | 5.9087 | ″ | — | — | — | 미산출 | — | — | 735.93 | 736.11 |
| e5.9_f125 | 125 | 5.9087 | ″ | — | — | — | 미산출 | — | — | 765.84 | 767.12 |
| e5.9_f129 | 129 | 5.9087 | ″ | — | — | — | 미산출 | — | — | 780.61 | 774.90 |

### 6b. Feasibility map — the headline change

| | v3 (s1g) | **v4 (s1j)** |
|---|---|---|
| Tier-1 open | 7 / 25 | **16 / 25** |
| Tier-2 only | 12 | 4 |
| Tier-3 only | 2 | 0 |
| closed at every tier | 4 | 5 |
| binding on closed cells | — | `cbc_max` 4, `f_r` 1 |

**9 cells opened at Tier-1 that v3 could not open; 0 cells closed that v3 had
open.** The whole e5.5 row except f109/f129, the whole f129 column at e≤5.4, and
e5.0_f109 / e5.2_f113 / e5.4_f125 are new. `min_pred_f_r` moved by
**mean −0.0126, median −0.0121** over the 25 cells (min −0.1412 at e5.5_f109, max
+0.1032 at e5.9_f117), while `max_pred_cyclen_any` barely moved
(**mean +0.45 EFPD**, median +0.77, range −6.50…+5.50). **The map opened on the
F_r axis, not the cycle-length axis** — s1j is flatter-predicting at fixed cycle
length, not longer-predicting.

On the 7 cells feasible under both champions, the node cycle length moved
**+1.82 EFPD mean** (−1.13…+4.24).

The e5.9 row is the exception and moved the other way: `min_pred_f_r` rose
+0.04…+0.10 and all five cells stay closed, four of them bound by `cbc_max`.
v3's one tier-3 cell there (e5.9_f117) is now `none`. **The CBC wall at e≈5.9 is
confirmed, not relieved, by the newer champion.**

### 6c. λ-optimum ≠ node in 14 of the 16 feasible cells

`cyclen − 1000·F_r` picks a *different* core from the max-cyclen node in 14 of
the 16 feasible cells; the two exceptions are e5.5_f113 and e5.5_f117, whose
Pareto fronts have a single point (`n_pareto = 1`). The λ reading trades **0.20–8.29 EFPD** of cycle length for **0.0006–0.0566**
of F_r; the largest trade is e5.0_f125 (−8.29 EFPD for −0.0566 F_r). This is
exactly the disagreement the registered rule exists to surface: **a headline read
off `pred_cyclen`/`pred_f_r` alone is not the λ answer**, and both are published.

### 6d. F_xy — every node is at or above the 1.65 limit's shoulder

`fxy_source = "head"` and `fxy_sigma_barred = True` on all 25 cells. Node F_xy
spans **1.6041–1.6520**; TWO of the 16 feasible nodes are *above* the 1.65
limit -- e5.0_f109 (1.6520) and e5.2_f117 (1.6507), both `node_f_xy_ok = False`.
(The other `node_f_xy_ok = False` entries are the 9 cells with no feasible core
at all.) The proxy σ
(the barred-sigma substitute) is **0.147–0.182**, i.e. the F_xy margin at every
node is a small fraction of its own uncertainty.

Substituting the F_xy gate for the F_r gate (`n_feasible_fxygate`) does **not**
close the map — it generally opens it wider (e5.4_f113: 138 F_r-feasible vs 287
F_xy-gate-feasible; e5.2_f125: 87 vs 103) and it re-opens three cells the F_r
gate closes (e5.2_f109 2, e5.4_f109 3, e5.5_f129 1). Under the **joint** F_r AND
F_xy gate the counts fall modestly (e.g. e5.0_f125 90 → 66). Read as a
diagnostic only: the frontier here is F_r (§3b).

### 6e. Pin-BU grade

At the node: **8 red (≥75), 8 orange (68–75), 9 미산출** (no feasible core). No
green or yellow anywhere — consistent with v3's 19/90 coverage being all orange
or red. The red cells are the low-feed end of e5.0–5.5 (f109–f117); orange
appears at f125/f129 where the discharge burnup per batch is lower. Advisory
only — the delivery gate is 80 GWd/tU on the pin axial peak.

### 6f. Measured overlay

20 of 25 cells resolve at level 1 `(library, pair, feed)`; the five e5.9 cells
fall back to `(feed, |Δe_core| ≤ 0.05)`. Store support per cell runs 9–501 rows.
**Joint-clean rows exist in only 8 of 25 cells** — the store contains 12–501 rows
in most cells but almost none that satisfy Tier-1 *and* F_xy ≤ 1.65 simultaneously
(`db_n_clean` = 0 in 17 cells). Where both exist, the model tracks the measured
best closely: `pred_cyclen − db_best_cyclen` spans **−0.96 … +2.64 EFPD** across
the 8 cells (mean ≈ +0.9), and the node's F_r sits **−0.011 … +0.071** from the
measured minimum F_r.

F_xy is labelled on **7,744 / 67,880 (11.4 %)** converged rows, so `db_n_fxy_labelled`
is reported per cell and unlabelled rows are counted, not deleted.

### 6g. LRM

The α-extrapolation adjudication re-run on the current store **reproduces the
2026-08-17 frozen values to machine precision** (`log_feed` chosen again; Δα ≤
4.4e-16 at f125 and f129) — the store's measured feed ratio has not moved enough
to shift the extrapolation. LOO would still have chosen `quad_feed`; the
registered rule overrules it, as before.

The **refit on the current store** (206 `(library, pair, feed)` groups with n ≥ 25,
e_core 4.7593–6.3420, all eight feeds 101–129) gives
`EFPD = α_F·(−50.928577 + 133.200084·e)` with gauge `α_117 = 1`, rms 7.452 EFPD.
Its α shape is a **materially better match to the store's own measured feed
ratio** than the DB-fitted one:

| feed | store measured q95 ratio (ref 117) | frozen α_ceil / α_ceil(117) | refit α_ceil |
|---|---|---|---|
| 109 | 0.9440 ± 0.0050 (n=19) | 0.9599 | **0.9448** |
| 113 | 0.9721 ± 0.0042 (n=9) | 0.9807 | **0.9678** |
| 121 | 1.0259 ± 0.0039 (n=18) | 1.0135 | **1.0274** |
| 125 | 1.0424 ± 0.0048 (n=19) | 1.0406 | **1.0421** |
| 129 | 1.0597 ± 0.0058 (n=5) | 1.0607 | **1.0527** |

Consequence on the map: the refit runs **8–17 EFPD BELOW** the frozen ceiling on
the low-feed / low-e corner (worst −17.13 at e5.0_f109) and converges to within
±3 EFPD at the high-e / high-feed corner, even crossing above it at e5.9_f117
(+0.18) and e5.9_f125 (+1.28). Both columns are published; neither is promoted.

Surrogate vs frozen LRM ceiling (`max_pred_cyclen_any − lrm_ceil_efpd`): the
surrogate sits **above** the LRM only in the e5.0 row (+3.0…+5.2 at f113/f117/f125)
and at e5.9_f125 (+3.2); everywhere else it is **below**, by up to −14.9 EFPD
(e5.5_f109). The v3-era finding that the surrogate over-runs the LRM ceiling at
the top of the grid is **not** reproduced by s1j on this grid.

### 6h. Runtime

5 shards × 6 torch threads, 5-way concurrent, on 24 physical cores. Per cell:
`run_cell` 281–311 s (median 302), the added F_xy pass 256–308 s (median 299),
total wall 557–620 s (median 601). Sweep wall clock **06:52 → 07:40:26 (~48 min)**;
all five shards exit 0. Post-processing 1 s. Peak python working set 12.7 GB
across 10 processes with 39.4 GB free.

## 7. Files

| file | what |
|---|---|
| `mesh_v4_cells.csv` | the 25 cells — every `scoping_mesh` node column, plus the λ / F_xy / pin-grade readouts, the measured overlay (`db_*`), the LRM columns and the v3 deltas (`v3_*`, `d_*`) |
| `mesh_v4_pareto.csv` | every point of every cell's (cyclen, F_r) Pareto front, with `rep` (min_f_r / knee / max_cyclen), `pred_f_xy`, `pin_grade`, `lam_score`, `is_lam_opt` |
| `mesh_v4_population.csv` | whole-population scatter: every valid & converged store row with `joint_clean` flagged |
| `lrm_v4_cells.csv` | LRM per cell — frozen mean/ceiling, store-refit mean/ceiling, α and its source, extrapolation flags, `B_cycle` / `B_d` |
| `provenance.json` | model + store + script SHA-256, store rows, seeds, pool sizes, gate limits, LRM constants (frozen and refit), the exact commands, runtimes |
| `pair_selection.csv` | the 16-target pair table produced on 198 under the shipped store |
| `mesh_v4_cells_[A-E].csv`, `mesh_v4_pareto_[A-E].csv` | the five raw shards before merge |
| `run_v4_[A-E].log`, `sh[A-E].out/.err` | per-shard logs |
| `post_v4.log` | merge + overlay + LRM + the per-cell headline table |
| `commands.json` | the ordered command list actually executed |
| `pairs_only.txt`, `smoke.txt`, `smoke2.txt` | the pair-table run and the two one-cell smoke tests (un-batched vs batched F_xy) |

## 8. Reproducing this on 198

```powershell
# 5 shards x 6 torch threads (30 of 32 logical)
& venv\Scripts\python.exe scratch_mesh_v4\mesh_v4_run.py `
    --out-dir data\reports\mesh_v4_20260903 --model s1j --tag _A `
    --threads 6 --e-targets 5.0 --feeds 109,113,117,125,129
# ... _B 5.2, _C 5.4, _D 5.5, _E 5.9 ...
& venv\Scripts\python.exe scratch_mesh_v4\mesh_v4_post.py `
    --out-dir data\reports\mesh_v4_20260903 --model s1j
```

`$env:PYTHONIOENCODING='utf-8'` is **required** — the kit's console is cp949 and
`scoping_mesh`'s own log lines contain em-dashes, which is how the first
`--pairs-only` attempt died (`UnicodeEncodeError: 'cp949' codec ...`).

## 9. What this run does NOT do

* **No figure was rendered.** `mesh_style2_fig.py` / `mesh_v3_fig.py` hard-code the
  v3 out-dir, the model name, the store row counts in their subtitles, a
  `PIN_OVERRIDE` table keyed to v3 cells, and a `GAP_HISTORY_REGISTERED`
  assertion on the v2 baseline. Adapting them is a judgement call on the
  override set and on whether a v4 scoreboard should keep asserting the v2-era
  baseline — decisions that belong to a pre-registration, not to a mesh
  regeneration. The CSVs contain everything the figures need.
* **No `--objective min_fxy` sweep.** The producer's refusal stands (§3b).
* **No MASTER, no anchors, no store writes.**
