# Round 9 — ADDENDUM: arm S1e (third T-cell refresh)

**Written 2026-08-15, BEFORE `data/splits/S1e.json` was persisted and before the
arm was launched.** Same discipline as `ab2_addendum_S1D_20260814.md` (round 8),
whose instrument, split rule, preservation checks and reporting rules are
inherited unchanged. Only what is new is stated here.

**Data-growth retrain, not a mechanism arm.** Instrument = **gate-promote**
against the standing champion **`s1d`**.

---

## 1. Why this arm exists — and the failure mode that interrupted the series

| round | model in the loop | best in band | feasible | note |
|---|---|---:|---:|---|
| r1 | `split_S1b` (T-blind) | 1.5549 | — | |
| r2 | `split_S1b` (T-blind) | 1.5440 | 1 / 200 calls | |
| r3 | `s1c` | 1.5018 | 14 | |
| r4 | `s1c` | 1.4866 | 57 (43 in-band) | |
| **r5** | `s1c` | **1.4972** | — | **regressed — cyclen-slide failure mode; fixed by `minfr_lambda` 200 → 400** |
| **r6** | **`s1d`** + λ 400 | **1.4797** | **67, ALL in-band**, 42 under 1.50 | |

Gap to the ga80 record **1.4636: 0.016**.

**r5 is a confounded round and is not evidence about `s1c`.** It carries a deck
failure (cyclen slide) *and* its fix, so the clean model-to-model comparison is
**r4 (`s1c`) → r6 (`s1d`) = 1.4866 → 1.4797, −0.007**. That is the per-round gain
this arm is cycling against, and it is decaying: **−0.042 → −0.015 → −0.007**.

The λ fix moved the search into the band — 67 of 67 feasible in-band at r6 versus
43 of 57 at r4 — so the *search* is no longer the binding constraint. What is left
to test is whether the model still has anything to gain from the newest labels.

---

## 2. Frozen inputs

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `f854e19646088c976360a6601bcf5da3e1fd3fc58f2470817a2a79678311557f` | 21,580,111 |
| `data/store/maps.npz` | `70fa87cbe569774631ec71658eb941ce9cce86a58538d94d41a871777224de0e` | 196,675,587 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| parent `data/splits/S1d.json` | `c2d077d1ba9eff5b44afeaa497862ac35e74160cc2c5601f1c77e87d0f513e7d` | 5,837,094 |

Rows **72,585** · increment vs S1d = **200**, all `case_pair = T6_T4`:
`fpcamp_minfr_T6T4_r5` 100 · `_r6` 100. Converged **195**. The **T6_T4 cell now
holds 870 rows**, min F_r **1.4797**.

`fuel_types.parquet` unchanged for the third round running (same sha), so the
round-7 §2.2 additive-table analysis still carries.

**Control:** `data/models/s1d`, frozen on disk (`DONE`, rc 0). Same reuse
justification as rounds 6–8; no fresh control is trained.

238 holds the round-8 store (`38cefb55…`), so this round **pushes data**
(`data=True`) and re-verifies every hash on the box after transfer.

---

## 3. The split — S1e

`build_split_S1b.py --parent S1d --name S1e --holdout-new-campaigns`. S1d
assignments verbatim; stable-hash 80/20 on the increment only.

Both increment campaigns (`_r5`, `_r6`) are **new-only**, so the round-8
cell-key fix — reading known cells from `curriculum_val_by_cell` rather than the
stale `groups['cells']` — is **inert this round**: no pre-existing campaign
received new rows. It is still the correct code path and is exercised by the
`--self-check`; recorded so its inertness here is not mistaken for absence.

### 3.1 Result and the preservation checks

| | |
|---|---:|
| train / val | **60,810 / 11,775** (+161 / +39) |
| increment | 200 rows → 39 val, 161 train |
| cells created | 2 (`_r5`, `_r6`) · cells grown: 0 |

| # | check | result |
|---|---|:--:|
| **a** | every S1d **val** id stays val | ✅ |
| **b** | every S1d **train** id stays train | ✅ |
| **c** | only the 200 new ids get fresh assignments | ✅ |
| **d** | every S1d cell present, ids retained, order intact | ✅ |
| **d′** | **`ab2_frozen_val_by_cell` forwarded verbatim — 36 cells / 3,207 rows** | ✅ |
| — | no train/val overlap · every store row in exactly one fold | ✅ |

### 3.2 T6_T4 coverage

| | S1c | S1d | **S1e** |
|---|---:|---:|---:|
| T6_T4 rows in **train** | 320 | 565 | **726** |
| T6_T4-family rows in **val** | 50 | 105 | **144** |

---

## 4. The arm

Single arm, v6b recipe, **identical to `s1d`'s except `--split S1e`**; seeds
20260716–20260720, `--parallel-members 5`, **238 GPU 1**
(`CUDA_VISIBLE_DEVICES=1`), occupancy checked first. The power prior refits on
the new train rows and may move off `s1d`'s (M² 110 / extrap 1.0) — expected,
registered, not a second change.

---

## 5. Decision rule

### 5.1 Primary

**`lpopt gate-promote --check-only`**, prev = `data/models/s1d`, new = the pulled
candidate, both scored on **S1e**. Its verdict decides. Generalization is judged
on the **frozen 36-cell surface** (3,207 rows, carried verbatim per §3.1 d′),
which contains no T-cell rows.

### 5.2 Registered SECONDARY readout — decides nothing

Served **F_r MAE** and **within-cell Spearman on the T6_T4 rows**,
champion-vs-candidate on identical rows — the same quantity rounds 7–8 reported,
so the series stays comparable.

**Still a FIT check, not a generalization check**: 726 T6_T4 rows are in the
candidate's training set. In-train and held-out figures are reported separately.

| | S1c | S1d | **S1e** |
|---|---:|---:|---:|
| T6_T4 val rows | 50 | 105 | **144** |
| cells ≥ `MIN_CELL_ROWS` (8) | 1 | 3 | **5** |
| true F_r sd | 0.0475 | 0.0928 | **0.0941** |
| true F_r min | 1.554 | 1.4936 | **1.4797** |

Five cells clears `ab_paired.MIN_CELLS = 3`, so a paired cell bootstrap is
available — but it is **below `MIN_CELLS_BCA = 6`, so BCa degrades to percentile**
and the interval is announced as such in `notes`. Reported with that degradation
named; not dressed up as a BCa interval. `frtransfer_T6T4_f121` (6 rows) remains
descriptive only.

### 5.3 Falsification

> If gate-promote FAILS, S1e is rejected and `s1d` stands. The registered reading
> is **not** "T-cell labels stopped helping" — it is that this 200-row increment
> did not clear the no-regression bar.

### 5.4 Registered CLOSE-OUT condition — fixed now, before r7/r8 run

Carried forward from the round-8 falsification clause and made proactive at the
coordinator's instruction, because the honest close-out matters as much as the
record:

> **If S1e's gate PASSES but the subsequent campaign round's in-band gain is
> `< 0.005`, the registered conclusion is that the T6_T4 cell is approaching its
> floor and the refresh loop is closed out.** The next move is then **not** a
> fourth identical refresh — it is either a deliberately wider T-cell production
> push (new pairs / new feed points, not more of the same cell) or an explicit
> declaration that the cell is mined out, whichever the evidence supports.

This is registered *before* the result so that a small gain cannot later be
presented as continued progress. The decay series to date —
**−0.042 → −0.015 → −0.007** — is already within a factor of ~1.4 of that
threshold, so the condition is live, not hypothetical.

Two guards on how it is read, also fixed now:

* **A confounded round does not count.** r5 carried a deck failure and its fix;
  any round whose deck changed materially is excluded from the gain series, as r5
  is here. The comparison must be model-to-model at a fixed deck.
* **The gap to the ga80 record (0.016) is context, not a target.** Closing it is
  not a condition of this arm, and failing to close it is not a failure of it.

### 5.5 What this round cannot conclude

No same-recipe-same-data control (the control is the standing champion, trained
on a smaller store). A PASS promotes a model; it does **not** establish that data
growth is the lever. Inherited verbatim from round 5 §5.4.

---

## 6. Pull, scoring and promotion

```
python -m lpopt.remote --input lpopt.inp pull --ts s1e        # -> data/models/s1e/

python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1d --new data/models/s1e \
    --out data/reports/gate_s1e.json --check-only

ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/s1e  s1e_cand  S1e && \
    ./venv/bin/python eval_accuracy_split.py runs/s1d  s1e_champ S1e'
cd 5_RL/runs/s1e && scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_s1e_{cand,champ}.csv .
```

Both models served on **S1e**, identical rows and order; neither trained on the
39 new val rows. Optional frozen-surface cross-check:

```
cd 5_RL && AB2_BU_DIR=runs/s1e AB2_SPLIT=S1e \
  AB2_CONTROL_CSV=rows_s1e_champ.csv AB2_TREATED_CSV=rows_s1e_cand.csv \
  AB2_BU_OUT=data/reports/ab2_verdict_S1E_20260815.json python ab2_bu_verdict.py
```

Confirm `checks.surface_key` reads `ab2_frozen_val_by_cell` (3,207 rows).

## §7 SERVED secondary readout (2026-08-15, post-promotion)

Both models served on S1e val (11,712 rows), identical rows/order; T6_T4 val = 144
rows (held out from BOTH models' training). Registered quantity, decides nothing:

| | s1d (prev champion) | s1e (new champion) |
|---|---:|---:|
| T6_T4 F_r MAE | **0.0301** | 0.0346 |
| T6_T4 pooled within-cell rho | **0.834** | 0.788 |
| per-campaign rho range | +0.20..+0.84 (r5 weak) | +0.51..+0.77 (uniform) |

Reading: s1e passed the PRIMARY gate (frozen 36-cell surface, worst drop 0.047 vs
eps 0.1388) but is NOT better than s1d on the active cell's held-out rows — the
+161-row increment did not raise in-cell ranking. Consistent with the decay series
(-0.042 -> -0.015 -> -0.007) approaching the cell floor. Interpretive consequence
for r8: if the in-band gain vs 1.4797 is < 0.005, the registered close-out fires
with BOTH legs supported (search plateau at r7 AND flat in-cell accuracy here).

## §8 r8 verdict and REGISTERED CLOSE-OUT (2026-08-15)

r8 (s1e, lambda 400, seed 1111): 63 feasible, 59 in-band. **In-band best F_r
1.4749 @ 625.46 EFPD — new cell record** (prior 1.4797, r6). CBC 1355.8, F_q
1.8233, all constraints clear. Gap to ga80 record 1.4636: 0.0113.

**Close-out clause FIRES**: gain 1.4797 - 1.4749 = 0.0048 < 0.0050 registered
threshold. Per pre-registration (S1E §5.3 note + orchestrator registration),
the T6_T4/f121 single-cell chase is declared to be at its floor. Supporting
legs: (1) r7 (second s1d round) regressed in-band; (2) served secondary (§7)
showed s1e did not raise in-cell accuracy over s1d. The 0.0002 margin is
acknowledged: the rule exists precisely so a marginal gain cannot be argued
into "one more round." Trajectory (in-band): r3 1.5018 -> r4 1.4866 -> r6
1.4797 -> r8 1.4749; each round = 100 MASTER calls.

Next registered moves: (a) 1-move ablation wave on this cell (fixes policy-net
era gap + settles leakage arbitration -- consumes the freed 199); (b) wide
production (feed-grid tranches, running on 198) instead of same-cell rounds.
