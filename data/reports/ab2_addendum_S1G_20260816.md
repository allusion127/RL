# Round 11 — ADDENDUM: arm S1g (f113 frontier labels)

**Written 2026-08-16, BEFORE `data/splits/S1g.json` was persisted and before the
arm was launched.** Same discipline as `ab2_addendum_S1F_20260816.md` (round 10);
instrument, split rule, preservation checks, recipe freeze and noise-floor
reporting rules inherited unchanged. Only what is new is stated here.

**Data-growth retrain.** Instrument = **gate-promote** against champion **`s1f`**.

---

## 1. Why this arm exists

`fpcamp_minfr_N1N2_f113` completed on 199: **100 rows, 98 converged, 40 below
F_r 1.55, floor 1.7243 → 1.4961 @ 641.6 EFPD** in a cell that previously had
**zero feasible cores**.

This is the smallest increment of any round and plausibly the highest-value one,
because of *what* it is rather than how much: these are **optimized frontier
labels at feed 113**, not production-grid samples. Round 10's mesh finding was
that `feedgrid` produce labels dragged unlabeled low-feed neighbours **away** from
DB truth; optimized low-feed labels are the stated antidote. §5.2(c) is the
registered test of exactly that.

Increment: 100 rows, `case_pair` N1_N2, feed 113, **library ga80**, converged 98,
F_r min 1.4961 / median 1.5665. The N1_N2 f113 cell now holds 287 rows.

---

## 2. Frozen inputs

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `dffd4d995fe7c7a4490c33da33f63c9d7b1c0b8e7397cc82e1eb04fd81edc621` | 21,979,093 |
| `data/store/maps.npz` | `a88cb4a57c074abdb6db7e677c4ae8aac0adb3ec0dda6b4b26556914fa9a476a` | 204,674,660 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| parent `data/splits/S1f.json` | `3b37fe563a19de1dd4587e4ba34c58275b97503c7e4acb4d47140d564f39b972` | 6,049,147 |

Rows **74,003**. `fuel_types.parquet` unchanged for the fifth round running.
Pre-merge backups exist as `*.bak_pre_N1N2f113_20260816`.
**Control:** `data/models/s1f`, frozen on disk (`DONE`, rc 0). 238 holds the
round-10 store, so this round pushes `data=True` and re-verifies hashes on the box.

**Recipe frozen by the scaling verdict** (`scaling_results_20260815.md`):
`width 224 / n-blocks 8 / head-hidden 384`, **5 members not 10**, v6b verbatim.
**Noise floors bind the reporting**, inherited from round 10 §2.2: within-cell ρ
movement below **~0.01 on cyclen / map_cov is NOT attributable** to the refresh
(ensemble-identity noise ~0.009 / ~0.007); single-seed spread carries
sd ≈ 0.018 on ρ F_r; `mde80` ≈ 0.009–0.015.

---

## 3. The split — S1g

`build_split_S1b.py --parent S1f --name S1g --holdout-new-campaigns`. S1f
assignments verbatim; stable-hash 80/20 on the increment only. Single new-only
campaign, so it receives a holdout under the round-8 cell-key logic.

| | |
|---|---:|
| train / val | **61,978 / 12,025** (+80 / +20) |
| cells created | 1 (`fpcamp_minfr_N1N2_f113`) · cells grown: 0 |

| # | check | result |
|---|---|:--:|
| **a** | every S1f **val** id stays val | ✅ |
| **b** | every S1f **train** id stays train | ✅ |
| **c** | only the 100 new ids get fresh assignments | ✅ |
| **d** | every S1f cell present, ids retained, order intact | ✅ |
| **d′** | **`ab2_frozen_val_by_cell` forwarded verbatim — 36 cells / 3,207 rows** | ✅ |
| — | no train/val overlap · every store row in exactly one fold | ✅ |

The round-10 §3.1 exclusion still stands: the three interventional waves
(`ablation_1move_T6T4`, `batchswap_enum_T6T4`, `batchswap_enum_625_T6T4`) have a
non-independent holdout and remain **excluded from every reported readout**.

---

## 4. The arm

v6b recipe, **identical to `s1f`'s except `--split S1g`**; seeds
20260716–20260720, `--parallel-members 5`, **238 GPU 1**
(`CUDA_VISIBLE_DEVICES=1`), occupancy checked first.

---

## 5. Decision rule — fixed before the gate runs

### 5.1 Primary

**`lpopt gate-promote`**, prev = `data/models/s1f`, new = the pulled candidate,
both on **S1g**. Promotion **authorized on PASS**; the gate JSON is written to
`data/reports/gate_s1g.json` before any promotion is acted on. Generalization is
judged on the **frozen 36-cell surface** (3,207 rows), which contains none of
these rows.

### 5.2 Registered SECONDARY readouts — decide nothing

**(a) The `N1_N2_f113` cell itself — fit check, near-tautological.**

| | |
|---|---|
| val rows | **20** (80 in train) |
| true F_r | min 1.4961 · median 1.5759 · max 1.9115 · sd 0.1003 |

The champion has **zero** of these rows; the candidate trains on 80 of the 100.
A large gain here is the *expected* result and is close to tautological — it
confirms the labels were absorbed, nothing more. One cell is **below
`ab_paired.MIN_CELLS = 3`**, so no paired cell bootstrap exists: this is a
descriptive 20-row Spearman/MAE with no interval, reported as such.

**(b) The standing T6_T4 series** (campaign cells only, waves excluded):
**183 val rows, 7 cells ≥ 8** — clears `MIN_CELLS_BCA = 6`, so a BCa interval is
available. F_r min 1.4749, sd 0.1014. Unchanged from round 10 by construction
(no T6_T4 rows in this increment), so it functions here as a **stability check**:
a *move* on this axis would be evidence of collateral drift from the f113 labels,
not of progress.

**(c) THE INTERESTING ONE — does f113 frontier data reduce low-feed pessimism?**

Reuses the existing `mesh_vs_db.py` (no new script), which already scores a named
ensemble's frontier bias against MASTER truth:

```
python mesh_vs_db.py --model s1g
```

Reported on the **f105 and f109** columns: `gap_total` (= `mesh_min_pred_f_r` −
`db_min_f_r`; positive ⇒ the model is *pessimistic* about the floor) and
`gap_pool`, s1g-vs-s1f. **The s1f baseline, pinned now so it cannot be
retro-fitted:**

| feed | segments | `gap_total` range | mean |
|---|---|---|---:|
| **105** | 5.0–5.5 | 0.140 … 0.251 | **0.207** |
| **109** | 5.0–5.5 | 0.066 … 0.160 | **0.129** |

**Registered expectation and its honest limit:** the new labels are at **f113**,
while this readout is at **f105/f109**. So this is a **transfer** test, not a
coverage test — it asks whether optimized frontier labels at one low feed pull the
model's floor estimate at *neighbouring* low feeds back toward truth. That is
precisely the round-10 mesh mechanism run in reverse, and it is the one readout
this round that could be informative rather than tautological. It is also the
weakest-powered: 12 cells, no interval, and a single retrain.

> **Direction is pre-committed: `gap_total` at f105/f109 should DECREASE.** An
> increase, or movement confined to f113 with f105/f109 flat, is registered in
> advance as *no transfer* — the labels helped where they landed and nowhere else.

**Operational note, registered because it is destructive:** `mesh_vs_db.py`
writes to **unsuffixed** `model_bias.csv` / `cell_verdicts.csv`. The current files
are the **s1f** baseline. Copy them to `*_s1f.csv` **before** running with
`--model s1g`, following the existing `*_s1e.csv` convention, or the baseline is
overwritten and the comparison is lost.

### 5.3 Falsification — registered before the gate runs

> If gate-promote FAILS, S1g is rejected and `s1f` stands. The registered reading
> is **not** "frontier labels do not help" — it is that this 100-row increment did
> not clear the no-regression bar. Because the increment is a *single cell at a
> single feed*, the registered next step on a FAIL is to check whether the failure
> is localized (a regression in the f113 neighbourhood) or global, and only then
> to decide between more low-feed frontier production and stopping. A FAIL here
> would also make §5.2(c) moot — an unpromotable model's bias shift is not
> actionable.

Additionally: if the gate PASSES but §5.2(c) shows **no** reduction in
`gap_total` at f105/f109, the registered conclusion is that **low-feed pessimism
is not fixable by neighbouring-feed frontier labels**, and the next intervention
must place optimized labels *in* the pessimistic cells rather than beside them.

### 5.4 What this round cannot conclude

No same-recipe-same-data control. A PASS promotes a model; it does **not**
establish that data growth is the lever. Inherited from round 5 §5.4. And per
§2, no delta below ~0.01 on cyclen/map_cov may be attributed to this refresh.

---

## 6. Pull, scoring and promotion

```
python -m lpopt.remote --input lpopt.inp pull --ts s1g        # -> data/models/s1g/

python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1f --new data/models/s1g \
    --out data/reports/gate_s1g.json --check-only
# promotion AUTHORIZED on PASS (§5.1): re-run without --check-only

ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/s1g  s1g_cand  S1g && \
    ./venv/bin/python eval_accuracy_split.py runs/s1f  s1g_champ S1g'
cd 5_RL/runs/s1g && scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_s1g_{cand,champ}.csv .

# (c) low-feed pessimism — PRESERVE THE BASELINE FIRST
cd 5_RL/data/reports/scoping_mesh_20260815
cp model_bias.csv model_bias_s1f.csv && cp cell_verdicts.csv cell_verdicts_s1f.csv
cd 5_RL && python mesh_vs_db.py --model s1g
```

Optional frozen-surface cross-check:

```
cd 5_RL && AB2_BU_DIR=runs/s1g AB2_SPLIT=S1g \
  AB2_CONTROL_CSV=rows_s1g_champ.csv AB2_TREATED_CSV=rows_s1g_cand.csv \
  AB2_BU_OUT=data/reports/ab2_verdict_S1G_20260816.json python ab2_bu_verdict.py
```

Confirm `checks.surface_key` reads `ab2_frozen_val_by_cell` (3,207 rows), and
exclude the three interventional waves from §5.2(b) per round 10 §3.1.
