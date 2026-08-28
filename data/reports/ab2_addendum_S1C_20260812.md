# Round 7 — ADDENDUM: arm S1c (T-cell data refresh)

**Written 2026-08-12, BEFORE `data/splits/S1c.json` was persisted and before the
arm was launched.** Addendum to `ab2_preregistration_20260730.md`; mirrors
`ab2_addendum_SPLIT_20260810.md` (round 5), whose split-rebuild rule, four
preservation checks and gate-promote instrument are inherited unchanged. Only
what is stated here is new.

**This is a data-growth retrain, not a mechanism arm.** The decision instrument
is **gate-promote** against the standing champion `split_S1b`, not a paired cell
bootstrap.

---

## 1. Why this arm exists

Since the S1b freeze (71,155 rows) the store grew to **72,085**. The +930
increment contains the **first-ever T-cell coverage**: 468 rows on case_pairs
`T6_T4` (370), `T5_T6` (65), `T1_T4` (32), `T3_T4` (1), library `paramA`.

**The champion has never seen a single T-cell row.** Two full `min_fr` campaigns
on `T6_T4` therefore ran with the surrogate effectively blind: r1 best 1.5549,
r2 first-feasible 1.5440, a slow grind.

The operational hypothesis, stated as such: **giving the champion the T-cell
labels unblinds the search there.** The precedent is E1_E2 — `split_S1b` moved
that cell's proposal floor 1.520 → 1.4889. This arm does not *prove* that
mechanism; it reproduces the intervention in a new cell and lets gate-promote
decide whether the resulting model is admissible.

---

## 2. Frozen inputs

### 2.1 Store snapshot (72,085 rows)

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `815656f373885c5983e935d416a17a96294c0807421aa1bd597b8c7b8ff387c3` | 21,379,692 |
| `data/store/maps.npz` | `cc806d51ad3e03b30e8a2f966597c615689754af23e3ccf58902bd84bb431768` | 191,373,852 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| parent `data/splits/S1b.json` | `47c19989a3ca9046f0186c38ca1470805582791323727877ce156211858f1c5b` | 5,812,568 |

Rows 72,085 · increment vs S1b = **930**.

### 2.2 The fuel table changed — and the change is provably additive

`fuel_types.parquet` went **153 → 194** rows (2026-08-11: T-alias rows, so the
featurizer resolves `T0…T6` tokens). Verified by keyed diff on `(library_id,
type_id)` before this document was written:

| check | result |
|---|---|
| common keys | **153** |
| keys removed | **0** |
| keys added | **41**, all `paramA`: alias rows `T0…T6` (7), `P0…P9`, `Q0…Q9`, `S0…S9` (30) + 4 full design ids |
| columns compared per key | 53 (array-valued columns compared element-wise) |
| **pre-existing rows with any changed value** | **NONE — every one is byte-stable** |

**Registered consequence, and it bounds the asymmetry the promotion design
carries.** The champion `split_S1b` was trained *before* this table existed but
will be *served* through it. Because the change is a strict superset with **no
mutated values**, the champion's encoding is unchanged for every
`(library, type)` it could already resolve. The only difference is that a slot
loading a `T*` token now resolves to a real `FuelVec` instead of falling to the
absent-value path (`vec is None` → `origin_feature_poor = 1`, every `origin_*`
channel at the 0 sentinel).

That is still an asymmetry — the champion never *trained* on T-token features it
now *sees* — and it is inherent to promotion-by-gate rather than something this
arm introduces. It is registered, not fixed. It cuts **against** the candidate if
anything: the champion is handed better inputs than it trained on, so the
comparison is not tilted toward the arm.

**Both models are scored through the same 194-row table**, which is shipped to
238 with the store.

---

## 3. The split — S1c

### 3.1 Built by the S1b method, extended by one clause

Built with the **same script**, parameterized rather than forked
(`build_split_S1b.py --parent S1b --name S1c --holdout-new-campaigns`):

* **S1b's assignments verbatim** — no pre-existing id is ever reconsidered, so
  growth-invariance holds by construction, not by argument.
* **Stable-hash 80/20 on the increment only**, `k = round(0.20 · n_new_converged)`
  over `_hash01(record_id)`, ties by `record_id` — byte-for-byte the upstream
  `make_curriculum_split` ordering.

**The one new clause, and why it was necessary.** Measured first: **0 of the 930
increment rows fall in a known curriculum cell** — every one is legacy-pool. Under
the S1b rule verbatim, *every T-cell row would land in train and none in val*,
making the §5.3 secondary readout **structurally unmeasurable** (the same trap
round 5's `f_r ≤ 1.55` slice fell into). So the 80/20 holdout is extended to
campaigns that exist **only** in the increment.

This does **not** re-engage the hazard the S1b rule was protecting against. That
hazard is redrawing the *legacy group carving* (an RNG shuffle over ancestry
groups, `splits.py` ~405). A brand-new campaign has no existing assignment to
disturb, so holding 20% of it out moves nothing. Pre-existing legacy campaigns
still route wholly to train.

### 3.2 Result and the preservation checks

| | |
|---|---:|
| train / val | **60,404 / 11,681** (+770 / +160) |
| increment | 930 rows → 160 val, 770 train |
| new cells created | 12 (all increment-only campaigns) |
| pre-existing cells that grew | **0** |

| # | check | result |
|---|---|:--:|
| **a** | every S1b **val** id stays val | ✅ |
| **b** | every S1b **train** id stays train | ✅ |
| **c** | only the 930 new ids get fresh assignments | ✅ |
| **d** | every S1b cell still present, all ids retained, original order intact | ✅ |
| **d′** | **`ab2_frozen_val_by_cell` forwarded verbatim — 36 cells / 3,207 rows** | ✅ |
| — | no train/val overlap · every store row in exactly one fold | ✅ |

**d′ is load-bearing and was a live bug.** The builder previously rebuilt the AB2
surface from the parent's *curriculum* map; on S1b that map has already grown to
3,327, so S1c would have silently redefined the historical surface. It now
forwards the parent's `ab2_frozen_val_by_cell` when present. The 3,207-row
surface that rounds 1–3, BU and ADF are judged on is unchanged.

### 3.3 T-cell rows, where they went

| campaign | rows | converged | → val | → train |
|---|---:|---:|---:|---:|
| `fpcamp_minfr_T6T4` | 338 | 220 | **44** | 294 |
| `frtransfer_T6T4_f121` | 32 | 32 | 6 | 26 |
| `frtransfer_T5T6_f117` | 32 | 32 | 6 | 26 |
| `frtransfer_T5T6_f121` | 32 | 32 | 6 | 26 |
| `frtransfer_T1T4_f117` | 32 | 32 | 6 | 26 |
| **T-cell total** (by `case_pair`) | **468** | — | **69** | **399** |

Verified against the persisted `S1c.json` by resolving `case_pair ~ ^T\d+_T\d+$`
against the store: **train 399** (`T6_T4` 320 · `T5_T6` 52 · `T1_T4` 26 ·
`T3_T4` 1) · **val 69** · total 468.

**399 T-cell rows enter training — from zero.**

---

## 4. The arm

Single arm, v6b recipe, **identical to `split_S1b`'s except `--split S1c`**:

```
--ensemble 5 --split S1c --cond-schema v6b --width 224 --n-blocks 8
--head-hidden 384 --epochs 150 --num-workers 8 --device auto
--parallel-members 5 --base-seed 20260716
--map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3
--map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads
--quantile-weight 0.2 --promote-max-asm-bu
--distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4
--distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

Seeds 20260716–20260720. **238 GPU 1** (`CUDA_VISIBLE_DEVICES=1`), occupancy
checked first.

**Registered note:** the power prior refits on the new train rows and may move
off `split_S1b`'s (M² 150, extrap 2.0). Unlike the ADF arm — where an unchanged
prior was a *prediction*, because ADF channels do not enter `kinf_quarter` — here
the train set itself changed, so a shift is expected and is not a second change.

---

## 5. The decision rule — fixed before any result exists

### 5.1 Primary instrument

**`lpopt gate-promote --check-only`**, prev = `data/models/split_S1b`, new = the
pulled candidate. Its verdict decides. **PASS ⇒ promotion-eligible; FAIL ⇒ the
refresh is not promoted and `split_S1b` stands.** No §5.2 number may override it.

Generalization is judged where it always is: the **frozen 36-cell surface**
(`ab2_frozen_val_by_cell`, 3,207 rows, forwarded verbatim per §3.2 d′), which
contains **no T-cell rows** and is untouched by this increment.

### 5.2 Registered SECONDARY readout — decides nothing

Served **F_r MAE** and **within-cell Spearman on the `T6_T4` rows**, champion vs
candidate, on the same rows — the cell the next campaign will actually run.

**It is a FIT check, not a generalization check, and must be reported as one.**
294 of the 338 `fpcamp_minfr_T6T4` rows are in the *candidate's training set*.
The candidate is therefore expected to look better on them almost by
construction, and a gain there is evidence that training absorbed the labels —
**not** that the model will rank unseen T-cell candidates better.

The honest slice is the 44 held-out `fpcamp_minfr_T6T4` val rows, and its power is
disclosed now:

| | |
|---|---|
| held-out rows (`fpcamp_minfr_T6T4`) | **44** |
| their true `F_r` | min 1.5540 · median 1.6148 · max 1.7711 · **sd 0.0475** |
| cells clearing `MIN_CELL_ROWS = 8` | **1** (`frtransfer_T6T4_f121`'s 6 do not) |

One cell is below `ab_paired.MIN_CELLS = 3`, so **no paired cell bootstrap is
possible**; this is reported as a single-cell Spearman + MAE over 44 rows with no
interval over cells, exactly as round 3 §5.6 reported its 36-row observation.
Both the in-train (fit) and held-out (44-row) figures are reported side by side so
the difference between them is visible rather than blended.

### 5.3 Falsification

> If gate-promote FAILS, the refresh is rejected and `split_S1b` stands. The
> registered reading is **not** "T-cell labels do not help" — it is that this
> 930-row increment did not clear the no-regression bar. The next move would be to
> grow the T-cells deliberately rather than re-run this arm on a slightly larger
> store; a second undirected growth round scored the same way carries no more
> information than the first (R3 §5.8, inherited).

If gate-promote PASSES, what is established is that the candidate is admissible —
**not** that the search is unblinded. The unblinding claim is operational and is
settled by running the next `T6_T4` campaign against the promoted model and
comparing its proposal floor to r1's 1.5549 / r2's 1.5440, the way the E1_E2
precedent was read.

### 5.4 What this round cannot conclude (continued below)

No same-recipe-same-data control exists (the control is the standing champion,
trained on a smaller store). A PASS therefore promotes a model; it does **not**
establish that data growth is the lever. Inherited verbatim from round 5 §5.4.

---

## 6. Launch record — and its provenance, stated plainly

**I did not launch this arm.** When I reached the launch step the run was already
in flight on 238, started **2026-08-12 01:43:57**, ~2 minutes ahead of me. Rather
than start a second run — which would have contended for the same GPU and wasted
hours — I verified the running job against every pin in this document and adopted
it. What was verified, at 01:46:

| item | running job | this addendum | match |
|---|---|---|:--:|
| `records.parquet` | `815656f3…8ff387c3` | §2.1 | ✅ |
| `maps.npz` | `cc806d51…bb431768` | §2.1 | ✅ |
| `fuel_types.parquet` | `fc73ad29…81f78137` (194 rows) | §2.2 | ✅ |
| **`S1c.json`** | **`2d5b44e4…766b4b09`** | **byte-identical to the split I built locally** | ✅ |
| S1c contents | train/val 60,404 / 11,681 · `derived_from S1b` · `ab2_frozen_n_rows` 3,207 | §3.2 | ✅ |
| recipe | `--ensemble 5 --split S1c --cond-schema v6b --width 224 --n-blocks 8 --head-hidden 384 --epochs 150 --parallel-members 5 --base-seed 20260716` + the full v6b flag set | §4 | ✅ |
| GPU | `CUDA_VISIBLE_DEVICES=1` | §4 | ✅ |
| run / session | `runs/s1c` / `lpopt_s1c` | — | — |

**The split hash matching to the byte is not a coincidence and is worth
recording as evidence:** the builder is deterministic in `(parent, store,
stable-hash rule)`, so an independent build of S1c from S1b on this store
reproduces the same file exactly. That is the growth-invariance property of §3.1
demonstrated end-to-end rather than merely asserted.

Source on 238 is the round-6 push (`v6b` 58 / `v6c` 62 channels, `v6c[:58] ==
v6b`), so a `--cond-schema v6b` run is byte-identical to what `split_S1b`
trained under — the ADF block is append-only and inert here.

Power prior on the running job: **M² 150, extrap 2.0, n_fit 3,996, ρ 0.6946**
(champion: 3,998 / 0.6888). The shift is expected and registered in §4 — the
train set itself changed.

**Consequence for pushing:** nothing needed pushing. The store, fuel table and
split on 238 already match the pins byte-for-byte, so no `lpopt.remote push` was
issued and the frozen inputs were never at risk of being overwritten mid-run.

---

## 7. Pull, scoring and promotion

**Stage 1 — pull** (the issuer; no local `data/models/` write until then):

```
python -m lpopt.remote --input lpopt.inp pull --ts s1c      # -> data/models/s1c/
```

**Stage 2 — the PRIMARY instrument, gate-promote (§5.1).** `--check-only` is
mandatory for a look: without it a PASS promotes immediately.

```
python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/split_S1b \
    --new  data/models/s1c \
    --out  data/reports/gate_s1c.json \
    --check-only
```

**Stage 3 — the SECONDARY readout (§5.2), decides nothing.** Serve BOTH models on
**S1c** so the rows and their order are identical, then compare F_r MAE and
within-cell Spearman on the `T6_T4` rows, reporting the in-train and the 44
held-out figures separately:

```
ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/s1c        s1c_cand  S1c && \
    ./venv/bin/python eval_accuracy_split.py runs/split_S1b  s1c_champ S1c'

cd 5_RL/runs/s1c
scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_s1c_cand.csv .
scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_s1c_champ.csv .
```

`eval_accuracy_split.py` is already on 238 from round 5 and takes the split name
as its third argument, so it needs no change.

**Optional cross-check on the frozen 36-cell surface** (generalization, §5.1) —
the same apparatus the BU/ADF gates use, which now selects
`ab2_frozen_val_by_cell` automatically:

```
cd 5_RL && AB2_BU_DIR=runs/s1c AB2_SPLIT=S1c \
  AB2_CONTROL_CSV=rows_s1c_champ.csv AB2_TREATED_CSV=rows_s1c_cand.csv \
  AB2_BU_OUT=data/reports/ab2_verdict_S1C_20260812.json python ab2_bu_verdict.py
```

Confirm `checks.surface_key` reads `ab2_frozen_val_by_cell` (3,207 rows) before
trusting it.
