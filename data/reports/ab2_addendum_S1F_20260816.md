# Round 10 — ADDENDUM: arm S1f (post-close-out refresh: feed support + interventional labels)

**Written 2026-08-16, BEFORE `data/splits/S1f.json` was persisted and before the
arm was launched.** Same discipline as `ab2_addendum_S1E_20260815.md` (round 9);
instrument, split rule, preservation checks and reporting rules inherited
unchanged. Only what is new is stated here.

**Data-growth retrain, not a mechanism arm.** Instrument = **gate-promote**
against the standing champion **`s1e`**.

---

## 1. Why this arm exists — and what it is NOT

**The T6_T4 min_fr loop is CLOSED.** The round-9 §5.4 close-out condition fired
exactly as registered: r8's in-band gain was **0.0048 < 0.005**. That clause was
written before r7/r8 ran, and it is being honoured rather than renegotiated.
**This arm does not continue that loop** — the r7/r8 rows in the increment are its
last output, not the start of a fourth refresh cycle.

What makes this round worth running is that the increment is **not more of the
same cell**. It is the "wider production push" the close-out clause named as the
legitimate alternative:

| source | rows | what it is |
|---|---:|---|
| `feedgrid_t1_*` (K1_K2_f109, E1_E2_f113, N1_N2_f113) | **450** | **genuinely new feed support** — feeds 109/113 in cells the store has never covered |
| `feedgrid_pf_*` (9 campaigns) | 85 | P0-pathfinder probes, feeds 113/117/129 |
| `ablation_1move_T6T4` · `batchswap_enum_T6T4` · `batchswap_enum_625_T6T4` | **583** | **interventional** single-move / batch-swap labels |
| `fpcamp_minfr_T6T4_r7` · `_r8` | 200 | the closed loop's final two rounds |
| **total** | **1,318** | |

Feeds in the increment: 109 (162) · 113 (302) · 117 (8) · 121 (783) · 129 (63).
Libraries: paramA 819 · ga80 499. Converged 1,143.

---

## 2. Frozen inputs

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `e86e934a1c031aeb1f38b9ac2ee44952420fbb4a13458554f46b6b41ba810ffa` | 21,942,421 |
| `data/store/maps.npz` | `bf66debc7f845d136f52b45f8efb5307a23f8b97702d232fee8fd71fe24b95fc` | 203,550,086 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| parent `data/splits/S1e.json` | `7437b1dbf58422ddcc5494406a7725db5a7f37882d791d3e4ea2b61856b67389` | 5,847,146 |

Rows **73,903**. `fuel_types.parquet` unchanged for the fourth round running.
**Control:** `data/models/s1e`, frozen on disk. 238 holds the round-9 store, so
this round pushes `data=True` and re-verifies every hash on the box.

### 2.1 Recipe is frozen by the scaling result — cited, not assumed

`data/reports/scaling_results_20260815.md` closed the capacity/ensemble question:

* **Capacity gate FAILED** (≥ +0.02 on ≥ 2 targets, CI excluding 0). Going bigger
  buys nothing; going *smaller* loses significantly on three of four targets
  (cyclen −0.0443, F_r −0.0186, map_cov −0.0173, every interval excluding zero).
  **10.4M sits at the knee.**
* **Ensemble 5 → 10 gate FAILED** on both legs. **Five members already saturate
  the epistemic term.**

So this arm runs the champion recipe **verbatim**: `width 224 / n-blocks 8 /
head-hidden 384`, **5 members (not 10)**, v6b, seeds 20260716–20260720. No
architecture or ensemble knob is touched, and none may be introduced mid-round.

### 2.2 Noise floors that bound how any delta may be reported

Registered now so no post-hoc delta can be over-read:

| quantity | floor | source |
|---|---|---|
| ensemble-identity noise, within-cell ρ **cyclen** | **~0.009** | E5′−E5, §3.3 |
| ensemble-identity noise, within-cell ρ **map_cov** | **~0.007** | E5′−E5, §3.3 |
| **reporting rule** | differences below **~0.01 on cyclen / map_cov are NOT attributable to the treatment** | §3.3 verbatim |
| single-seed spread, training-time within-case ρ F_r | sd 0.0126 → a single-seed difference carries **sd ≈ 0.018** | §3.3 |
| paired-bootstrap `mde80` on that surface | **≈ 0.009–0.015** | §3.2 |

**Binding consequence for this round:** any reported within-cell ρ movement below
~0.01 on cyclen/map_cov is reported as *within ensemble-identity noise*, never as
an effect of the data refresh.

---

## 3. The split — S1f

`build_split_S1b.py --parent S1e --name S1f --holdout-new-campaigns`. S1e
assignments verbatim; stable-hash 80/20 on the increment only. All 16 increment
campaigns are new-only, so the round-8 cell-key fix is again inert (no
pre-existing campaign received rows) but remains the correct path.

| | |
|---|---:|
| train / val | **61,898 / 12,005** (+1,088 / +230) |
| increment | 1,318 rows → 230 val, 1,088 train |
| cells created | 16 · cells grown: 0 |

| # | check | result |
|---|---|:--:|
| **a** | every S1e **val** id stays val | ✅ |
| **b** | every S1e **train** id stays train | ✅ |
| **c** | only the 1,318 new ids get fresh assignments | ✅ |
| **d** | every S1e cell present, ids retained, order intact | ✅ |
| **d′** | **`ab2_frozen_val_by_cell` forwarded verbatim — 36 cells / 3,207 rows** | ✅ |
| — | no train/val overlap · every store row in exactly one fold | ✅ |

### 3.1 The interventional waves have a NON-INDEPENDENT holdout — measured, registered

The 80/20 rule assumes rows within a cell are exchangeable. **The three
interventional waves violate that by construction**, and the measurement is:

| campaign | rows | **distinct parent patterns** | val rows |
|---|---:|---:|---:|
| `ablation_1move_T6T4` | 150 | **10** | 29 |
| `batchswap_enum_T6T4` | 213 | **6** | 43 |
| `batchswap_enum_625_T6T4` | 220 | **5** | 44 |

Each wave is a dense neighbourhood around a handful of base patterns, so a
"held-out" row is **one move away from train rows sharing its own parent**. Its
val performance measures interpolation inside a tiny neighbourhood, not
generalization.

**Registered consequences, fixed before the gate runs:**

1. **These 116 val rows (0.97% of the 12,005) are EXCLUDED from every reported
   secondary readout.** §5.2's T6_T4 figures are computed on the campaign cells
   only.
2. **They are flagged for any future paired-bootstrap scoring** on
   `curriculum_val_by_cell`. A round that scores those cells without excluding
   them will report an optimistic number.
3. **They do NOT put the primary instrument at risk.** gate-promote is a
   *no-regression* gate evaluated **per cell**: near-duplicate cells make the
   candidate look better in those cells, which cannot mask a regression in a
   different cell, and the gate's PASS condition does not reward improvement.
   So a false PASS cannot be manufactured this way.
4. **They stay in TRAIN, deliberately.** Local sensitivity labels are exactly
   what a surrogate should learn from; the defect is in using them to *measure*,
   not in learning from them. Routing them wholly to train was considered and
   rejected as a fourth rule variant for no gain, given (3).

---

## 4. The arm

Single arm, v6b recipe, **identical to `s1e`'s except `--split S1f`**; seeds
20260716–20260720, `--parallel-members 5`, **238 GPU 1**
(`CUDA_VISIBLE_DEVICES=1`), occupancy checked first even though both GPUs are
free. Power prior refits on the new train rows and may move off `s1e`'s
(M² 150 / extrap 2.0) — expected, registered, not a second change.

---

## 5. Decision rule — fixed before the gate runs

### 5.1 Primary

**`lpopt gate-promote`**, prev = `data/models/s1e`, new = the pulled candidate,
both scored on **S1f**. Its verdict decides. **Promotion is authorized on PASS**
this round (the coordinator's instruction), so the run is `--check-only` first
for the record and then without it, or once without — either way the gate JSON is
written to `data/reports/gate_s1f.json` before any promotion is acted on.

Generalization is judged on the **frozen 36-cell surface** (3,207 rows, carried
verbatim per §3 d′), which contains no T-cell, feedgrid or interventional rows.

### 5.2 Registered SECONDARY readouts — decide nothing

**(a) T6_T4 within-cell** — the continuing series, campaign cells only
(interventional waves excluded per §3.1):

| | S1c | S1d | S1e | **S1f** |
|---|---:|---:|---:|---:|
| val rows | 50 | 105 | 144 | **183** |
| cells ≥ 8 rows | 1 | 3 | 5 | **7** |
| true F_r sd | 0.0475 | 0.0928 | 0.0941 | **0.1014** |

Seven cells clears `MIN_CELLS_BCA = 6` for the first time, so a **BCa** interval
is available rather than the percentile fallback rounds 8–9 had to use. Still a
**fit check** (the candidate trains on the rest of those cells); in-train and
held-out figures reported separately.

**(b) Feedgrid cells — NEW, and the expected-weak prior stated honestly:**

| cell | val rows | true F_r min / median / max | sd |
|---|---:|---|---:|
| `feedgrid_t1_K1_K2_f109` | 22 | 1.7797 / 2.1256 / 2.8359 | 0.2709 |
| `feedgrid_t1_E1_E2_f113` | 20 | 1.8267 / 2.2228 / 2.7138 | 0.2549 |
| `feedgrid_t1_N1_N2_f113` | 19 | 1.8209 / 2.1162 / 3.1861 | 0.4095 |

61 val rows, 3 cells — exactly `MIN_CELLS = 3`, **below `MIN_CELLS_BCA = 6`, so
BCa degrades to percentile**; reported with that degradation named.

**Expected weak, and why — registered in advance:**

* These are the **first** labels the model has ever had in these cells; ~390 of
  the 450 are in the candidate's training set and **zero** in the champion's.
  A large apparent gain is therefore the *expected* outcome and is close to
  tautological — it is a fit check with an unusually strong prior, not evidence
  of transfer.
* Their F_r spans **1.78–3.19**, entirely **outside the decision band
  (F_r < 1.55)** the optimizer works in. Ranking skill on unoptimized production
  grids at F_r ≈ 2.1 says little about ranking skill among elites, which is the
  axis KILLER 2 put at ρ ≈ 0.
* n = 19–22 per cell with sd 0.25–0.41 is a wide, noisy band.

**Registered reading:** a gain here establishes that the model *absorbed* the new
feed labels. It does **not** establish that the search gains eyes at those feeds
— that claim requires a campaign in one of those cells, exactly as the T-cell
unblinding claim required r3/r4 rather than the S1c readout.

### 5.3 Falsification — registered before the gate runs

> If gate-promote FAILS, S1f is rejected and `s1e` stands. The registered reading
> is **not** "the new feed support does not help" — it is that this 1,318-row
> increment did not clear the no-regression bar. Because the increment is
> dominated by two structurally unusual sources (583 interventional rows around
> ≤10 parents, and 450 rows at F_r ≈ 2.1 far outside the decision band), the
> registered next step on a FAIL is to **re-run the refresh with the
> interventional waves held out of training** — separating "new feed support
> hurt" from "dense single-move neighbourhoods distorted the fit" — rather than
> to abandon feed-grid production.

### 5.4 What this round cannot conclude

No same-recipe-same-data control (the control is the standing champion, trained
on a smaller store). A PASS promotes a model; it does **not** establish that data
growth is the lever. Inherited verbatim from round 5 §5.4. Additionally, per
§2.2, **no reported delta below ~0.01 on cyclen/map_cov may be attributed to this
refresh at all.**

---

## 6. Pull, scoring and promotion

```
python -m lpopt.remote --input lpopt.inp pull --ts s1f        # -> data/models/s1f/

# record the gate first
python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1e --new data/models/s1f \
    --out data/reports/gate_s1f.json --check-only

# promotion is AUTHORIZED on PASS (§5.1): re-run without --check-only
python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1e --new data/models/s1f \
    --out data/reports/gate_s1f.json

ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/s1f  s1f_cand  S1f && \
    ./venv/bin/python eval_accuracy_split.py runs/s1e  s1f_champ S1f'
cd 5_RL/runs/s1f && scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_s1f_{cand,champ}.csv .
```

Both models served on **S1f**, identical rows and order; neither trained on the
230 new val rows. Optional frozen-surface cross-check:

```
cd 5_RL && AB2_BU_DIR=runs/s1f AB2_SPLIT=S1f \
  AB2_CONTROL_CSV=rows_s1f_champ.csv AB2_TREATED_CSV=rows_s1f_cand.csv \
  AB2_BU_OUT=data/reports/ab2_verdict_S1F_20260816.json python ab2_bu_verdict.py
```

Confirm `checks.surface_key` reads `ab2_frozen_val_by_cell` (3,207 rows). When
computing §5.2(a), **exclude** `ablation_1move_T6T4`, `batchswap_enum_T6T4` and
`batchswap_enum_625_T6T4` per §3.1.
