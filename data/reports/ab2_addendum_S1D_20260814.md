# Round 8 — ADDENDUM: arm S1d (second T-cell refresh)

**Written 2026-08-14, BEFORE `data/splits/S1d.json` was persisted and before the
arm was launched.** Same discipline as `ab2_addendum_S1C_20260812.md` (round 7),
whose instrument, split rule, preservation checks and reporting rules are
inherited unchanged. Only what is new is stated here.

**Data-growth retrain, not a mechanism arm.** Instrument = **gate-promote**
against the standing champion **`s1c`**.

---

## 1. Why this arm exists — the loop is working, and decaying

The S1c retrain unblinded the T6_T4 search. Measured across `min_fr` rounds:

| round | model in the loop | best in band | feasible |
|---|---|---:|---:|
| r1 | `split_S1b` (T-blind) | 1.5549 | — |
| r2 | `split_S1b` (T-blind) | 1.5440 | 1 in 200 calls |
| **r3** | **`s1c`** | **1.5018** | 14 |
| **r4** | **`s1c`** | **1.4866** | 57 (43 in-band) |

Per-round gain is decaying: **−0.042 → −0.015**. That decay is the reason to
cycle the retrain now rather than run r5 on the same model — the model has
extracted what it can from the r1/r2 labels and the newest labels are the ones it
has not seen.

**This arm does not test the unblinding hypothesis** — round 7 already ran that
intervention and the campaign evidence above is its operational read-out. This
arm asks only whether the *next* refresh is admissible.

---

## 2. Frozen inputs

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `38cefb55c26adb60eea2d648a75021838807c6bd08021327d19e6120ac7428f6` | 21,523,121 |
| `data/store/maps.npz` | `c01c28684cff5e99fff623ba5e69d00ed0427c30fe7be0860ab6a4cb43c06814` | 194,520,131 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| parent `data/splits/S1c.json` | `2d5b44e493210f3e4ef1c4da77de8fec7838c32d5211b287217f43c9766b4b09` | 5,859,098 |

Rows **72,385** · increment vs S1c = **300**, *all* `case_pair = T6_T4`:
`fpcamp_minfr_T6T4_r4` 100 · `_r3` 100 · `fpcamp_minfr_T6T4` 92 · `r3_repro` 8.
Converged 276. The **T6_T4 cell now holds 670 rows**, min F_r **1.4866**.

`fuel_types.parquet` is **unchanged** from round 7 (same sha), so the round-7 §2.2
additive-table analysis carries over untouched.

**Control:** `data/models/s1c`, frozen on disk — a fresh v6b model on the parent
split. Same reuse justification as rounds 6–7; no fresh control is trained.

**238 holds the round-7 store**, not this one (`815656f3…` = 72,085). So this
round **must push data** — unlike round 6, where the frozen snapshot was already
on the box and `data=False` protected it. Hashes are re-verified on 238 after
transfer.

---

## 3. The split — S1d

Built with the same parameterized builder:
`build_split_S1b.py --parent S1c --name S1d --holdout-new-campaigns`.
S1c assignments verbatim; stable-hash 80/20 on the increment only.

### 3.1 A latent bug in the builder, found and fixed before S1d was cut

The builder read its "known cell" set from `groups['cells']`. That key is **never
updated** when a round promotes a new campaign to a cell, so it still lists the
original **36** while S1c actually scores **48** cells. Consequence, had it been
left alone:

> `fpcamp_minfr_T6T4` — a campaign that S1c promoted to a cell *and gave a
> 44-row val holdout* — would have had all **92** of its new rows sent to train,
> because it is absent from the stale `groups['cells']`. A campaign would get the
> 80/20 holdout in the round it first appears and then silently lose it forever
> after.

Fixed by reading the authoritative key: a "known cell" is a campaign that already
has a val holdout in the parent, i.e. a key of `curriculum_val_by_cell`
(∪ `groups['cells']` for safety). Effect on S1d: `fpcamp_minfr_T6T4`'s 92 rows
now split **17 val / 75 train** instead of 0 / 92.

**`groups['cells']` is deliberately left stale.** It is also what the *trainer*
reads for the curriculum sampling-weight cap (`train.py` ~1733), so rewriting it
would change training behaviour, not just fold assignment — a second change, and
inadmissible in a data-growth arm. Marked in-source with the ceiling and the
upgrade path (a dedicated arm that A/Bs the weight cap).

### 3.2 Result and the preservation checks

| | |
|---|---:|
| train / val | **60,649 / 11,736** (+245 / +55) |
| increment | 300 rows → 55 val, 245 train |
| cells created | 2 (`fpcamp_minfr_T6T4_r3`, `_r4`) |
| cells grown | 1 (`fpcamp_minfr_T6T4`, +17) |

| # | check | result |
|---|---|:--:|
| **a** | every S1c **val** id stays val | ✅ |
| **b** | every S1c **train** id stays train | ✅ |
| **c** | only the 300 new ids get fresh assignments | ✅ |
| **d** | every S1c cell present, ids retained, order intact | ✅ |
| **d′** | **`ab2_frozen_val_by_cell` forwarded verbatim — 36 cells / 3,207 rows** | ✅ |
| — | no train/val overlap · every store row in exactly one fold | ✅ |

`r3_repro` (8 rows, **0 converged**) contributes 0 val by the rule's
`n_conv >= 2` guard and lands wholly in train — correct, and noted so its absence
from the cell list is not read as a bug.

### 3.3 T6_T4 coverage

| | S1c | **S1d** |
|---|---:|---:|
| T6_T4 rows in **train** | 320 | **565** |
| T6_T4-family rows in **val** | 50 | **105** |

---

## 4. The arm

Single arm, v6b recipe, **identical to `s1c`'s except `--split S1d`**:

```
--ensemble 5 --split S1d --cond-schema v6b --width 224 --n-blocks 8
--head-hidden 384 --epochs 150 --num-workers 8 --device auto
--parallel-members 5 --base-seed 20260716
--map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3
--map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads
--quantile-weight 0.2 --promote-max-asm-bu
--distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4
--distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

Seeds 20260716–20260720. **238 GPU 1** (`CUDA_VISIBLE_DEVICES=1`), occupancy
checked first. The power prior refits on the new train rows and may move off
`s1c`'s — expected, registered, not a second change.

---

## 5. Decision rule

### 5.1 Primary

**`lpopt gate-promote --check-only`**, prev = `data/models/s1c`, new = the pulled
candidate, both scored on **S1d**. Its verdict decides. Generalization is judged
on the **frozen 36-cell surface** (`ab2_frozen_val_by_cell`, 3,207 rows, carried
verbatim per §3.2 d′), which contains **no T-cell rows**.

### 5.2 Registered SECONDARY readout — decides nothing

Served **F_r MAE** and **within-cell Spearman on the T6_T4 rows**,
champion-vs-candidate on identical rows — the same quantity round 7 reported
(**+0.077 → +0.513**), so the two rounds are directly comparable.

**It remains a FIT check, not a generalization check.** 565 T6_T4 rows are in the
candidate's training set and 0 in the champion's-era training beyond what `s1c`
already had, so the candidate is expected to look better on in-train rows almost
by construction. The in-train and held-out figures are reported **separately**.

Held-out power, disclosed in advance — materially better than round 7:

| | round 7 (S1c) | **round 8 (S1d)** |
|---|---:|---:|
| T6_T4-family val rows | 50 | **105** |
| cells clearing `MIN_CELL_ROWS = 8` | 1 | **3** (61 · 19 · 19) |
| true F_r spread (sd) | 0.0475 | **0.0928** |
| true F_r range | 1.554–1.771 | **1.4936–2.0678** |

Three cells is exactly `ab_paired.MIN_CELLS = 3`, the floor — so a paired cell
bootstrap becomes *technically* available but at minimum clusters, where BCa
degrades to percentile and the interval is wide. It is reported with that
degradation named, alongside the per-cell values, and it may not be dressed up as
a well-powered interval. `frtransfer_T6T4_f121` (6 rows) stays below the floor and
is descriptive only.

### 5.3 Falsification

> If gate-promote FAILS, S1d is rejected and `s1c` stands. The registered reading
> is **not** "T-cell labels stopped helping" — it is that this 300-row increment
> did not clear the no-regression bar. Given the decay already measured
> (−0.042 → −0.015), the registered next move is **not** a third identical
> refresh: it is either a wider T-cell production push or an acceptance that the
> cell is mined out, decided on the r5 campaign result rather than on another
> re-split.

### 5.4 What this round cannot conclude

No same-recipe-same-data control (the control is the standing champion, trained
on a smaller store). A PASS promotes a model; it does **not** establish that data
growth is the lever. Inherited verbatim from round 5 §5.4.

---

## 6. Pull, scoring and promotion

```
python -m lpopt.remote --input lpopt.inp pull --ts s1d        # -> data/models/s1d/

python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1c --new data/models/s1d \
    --out data/reports/gate_s1d.json --check-only

ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/s1d  s1d_cand  S1d && \
    ./venv/bin/python eval_accuracy_split.py runs/s1c  s1d_champ S1d'
cd 5_RL/runs/s1d && scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_s1d_{cand,champ}.csv .
```

Both models served on **S1d**, identical rows and order; neither trained on the
55 new val rows. Optional frozen-surface cross-check:

```
cd 5_RL && AB2_BU_DIR=runs/s1d AB2_SPLIT=S1d \
  AB2_CONTROL_CSV=rows_s1d_champ.csv AB2_TREATED_CSV=rows_s1d_cand.csv \
  AB2_BU_OUT=data/reports/ab2_verdict_S1D_20260814.json python ab2_bu_verdict.py
```

Confirm `checks.surface_key` reads `ab2_frozen_val_by_cell` (3,207 rows).
