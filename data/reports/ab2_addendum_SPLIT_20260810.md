# Round 5 — ADDENDUM: arm SPLIT (split rebuild + data-growth retrain)

**Written 2026-08-10, BEFORE `data/splits/S1b.json` was persisted and before the
arm was launched.** Addendum to `data/reports/ab2_preregistration_20260730.md`
and to `data/reports/ab2_addendum_BU_20260810.md` (whose v6b recipe this arm
reuses unchanged). Only what is stated here is new. Anything added after the
result exists is post-hoc and must be labelled.

**This is NOT an A/B mechanism arm.** It is a data-growth retrain, so the
decision instrument is the standing **gate-promote** (no-regression + legacy
tail) against the incumbent champion — not a paired cell bootstrap on a frozen
surface. §5 says exactly what that means and what it cannot conclude.

**Order of work, auditable:** this document was written and saved before
`build_split_S1b.py --write` was run and before any training was launched. The
only facts known at the time of writing are the structural ones in §2–§3 — file
hashes, row counts, split arithmetic, the four split verifications, and the
F_r-coverage census — none of which involve a prediction from the new arm.

---

## 1. Why this arm exists

I flagged this myself when freezing the BU arm: *"the 1,242 newer rows enter
neither arm … should a later round want those rows in training, it needs a
rebuilt split — which would be a second change and a different arm."* This is
that arm.

`S1.json` enumerates `train_ids` / `val_ids` explicitly. Every row merged after
it was built is therefore in **neither** fold — invisible to training and
invisible to evaluation. That set has grown to **1,413 rows**, and it contains
almost the entire low-F_r population the programme now cares most about.

### 1.1 The blindness, measured

Converged rows only, `f_r` from the store:

| population | n | `f_r < 1.50` | `f_r < 1.55` | min `f_r` |
|---|---:|---:|---:|---:|
| **S1 train — everything the champion has ever seen** | 50,445 | **5** | 782 | 1.4636 |
| **the increment — seen by nothing** | 1,249 | **68** | 438 | 1.4666 |

**The champion was trained on five sub-1.50 cores.** The increment holds 68, and
every one of them routes to train under §3. That takes the sub-1.50 training
population from **5 → 73 (14.6×)** and the sub-1.55 population from 782 → 1,220
(+56%).

This is the direct answer to KILLER 2 / the v520 finding that elite-pool ranking
Spearman sits at ≈ 0: a model cannot rank a region it has five examples of. It
is also the fix R3's own falsification clause demanded — R3 §5.8 registered that
the next round must *"hold out a slice of the enriched cells so densification can
be measured where it happened"*, and §3.2 below does precisely that.

### 1.2 The itemisation does not reconcile — recorded, not hidden

The coordinator itemises the growth as "~350 verified rows … plus DECK1's
`fadj_*` 727 rows". The store shows **1,413** rows outside S1:

| pool | rows |
|---|---:|
| known curriculum cells (`5-5.25_f117` 250, `5-5.25_f125` 243, `5.5-5.75_f125` 234) | **727** |
| legacy pool (frtransfer_* 189, fpcamp{2,3,4}_199 300, v520_minfr_b{1,2,3} 60, debug_panel_* , frscreen_flat13, …) | **686** |
| **total** | **1,413** |

727 matches the `fadj_*` figure exactly. The remaining **686** exceeds the
"~350" by **336**. The likely benign explanation is that the itemisation counts
only rows merged since the BU freeze (1,413 − 1,242 = 171) plus some campaigns
twice, but **this is not verified** and is recorded as an open discrepancy, in
the same spirit as R3 §3.3. It does not affect the arm — the arm is whatever the
split assigns — but any downstream summary should quote **1,413**, not ~350.

---

## 2. Frozen inputs

### 2.1 Store snapshot

| item | value |
|---|---|
| `data/store/records.parquet` sha256 | `00a6ecb0f986209e514d386067e075db606efa90aba210ce43f6eae533d75027` |
| bytes | 21,059,894 |
| rows | **71,155** (converged 63,032) |
| `data/store/maps.npz` sha256 | `903ee1dada769becea3633ef97b6f0c84d8bc8703ea81b25989fa6ad306666b3` |
| bytes | 184,331,591 |
| `data/store/fuel_types.parquet` sha256 | `4ee9b16e4f595525c15168ea477fd92b6f39bb110147bf1b12c336f49c5c8ecd` |
| bytes | 61,296 |

### 2.2 Parent split

| item | value |
|---|---|
| `data/splits/S1.json` sha256 | `6ab35f25027c8355fe4f07bda2a200b1c5baf40d445257e255ed89dead650640` |
| train / val | 58,341 / 11,401 |

### 2.3 Incumbent champion — the control, frozen

`data/models/20260810_bu_T`, verified from `member_20260716/meta.json`:
`cond_schema = v6b`, `in_channels = 58`, `n_globals = 13`,
`map_prior_channel = 50`, `split = S1`, members `20260716 … 20260720`,
`power_prior = {m2_cm2: 150.0, extrap: 1.0, n_fit: 3998, within_cell_rho: 0.6923}`.

**No fresh control is trained this round.** gate-promote is a two-model
comparison against the standing champion, and that champion is frozen on disk,
so a contemporaneous A0 buys nothing the gate uses. This is a deliberate
departure from the AB2 rounds and is registered as such: it means this round
**cannot** separate "the new data helped" from "a second training run of the same
recipe drifts", because there is no same-recipe-same-data replicate. §5.4 states
the consequence.

---

## 3. The split rebuild — procedure registered BEFORE it was run

Built by `5_RL/build_split_S1b.py` (in the repo, auditable, dry-run by default)
into `data/splits/S1b.json`. **S1.json is not modified.**

### 3.1 Why not simply re-run `make_curriculum_split`

Two independent hazards, both visible in `splits.py`:

1. **The legacy pool's val carving is an RNG shuffle over ancestry groups**
   (`rng = random.Random(seed); rng.shuffle(candidates)`, ~line 405). The
   candidate list is derived from the store, so new campaigns change the group
   inventory, which changes what that shuffle selects. A regeneration can evict
   a whole evaluation band into val or pull one out — the 2026-07-18 lesson.
2. **The curriculum holdout is stable-hash but NOT size-invariant.** Per cell it
   takes the `val_count = round(0.20 · n_conv)` smallest-hash converged ids. A
   row's hash is invariant, but `val_count` grows with the cell, so a
   regeneration pulls *more* ids from the same sorted pool. Existing val ids are
   *likely* preserved (they hold the smallest hashes) — but nothing guarantees
   it, and "likely" is not a property an evaluation surface may rest on.

So the rebuild **copies S1's assignments verbatim and assigns only the
increment.** Growth-invariance then holds by construction, not by argument: no
pre-existing id is ever reconsidered.

### 3.2 The assignment rule

* **New row in a known curriculum cell** → the cell-wise **80/20 stable-hash
  holdout applied to the increment**: `k = round(0.20 · n_new_converged)`, take
  the smallest `_hash01(record_id)` ids (ties by record_id — byte-for-byte the
  upstream ordering), clamped to leave ≥ 1 new converged row in train.
  Non-converged new rows → train, as upstream.
* **New row in the legacy pool** → **train**. Growing the legacy val fold means
  redrawing the group carving, which is hazard 1. This is also R3's precedent:
  every new row went to train and val did not grow by a single row.

### 3.3 Result, and the four verifications (all run before `--write`)

| | |
|---|---:|
| new train / val | **59,634 / 11,521** |
| delta | **+1,293 / +120** |
| curriculum increment | 727 rows → 120 val, 607 train |
| legacy increment | 686 rows → 686 train, 0 val |

| # | check | result |
|---|---|:--:|
| **a** | every pre-existing S1 **val** id stays val | ✅ |
| **b** | every pre-existing S1 **train** id stays train | ✅ |
| **c** | **only** the 1,413 new ids receive fresh assignments | ✅ |
| **d** | the frozen 36-cell val rows survive **verbatim** — all 36 cells present, all 3,207 original ids retained, original order intact | ✅ |
| — | no train/val overlap | ✅ |
| — | every store row is in exactly one fold (0 quarantined) | ✅ |

All six were re-verified **independently after the file was written and reloaded
from disk**, not only inside the builder:

| item | value |
|---|---|
| `data/splits/S1b.json` sha256 | `47c19989a3ca9046f0186c38ca1470805582791323727877ce156211858f1c5b` |
| bytes | 5,812,568 |
| train / val | 59,634 / 11,521 |
| resolved against the store | train 59,634 (converged 51,574) · val 11,521 (converged 11,458) |
| `groups['cells']` / `curriculum_cell_cap` | 36 / 16.0 — copied unchanged |
| `curriculum_val_by_cell` total | 3,327 (= 3,207 frozen + 120 new) |
| `ab2_frozen_val_by_cell` | 36 cells / **3,207** rows, byte-identical to S1 |
| **train rows with `f_r < 1.50`** | **73** (champion's S1 train: 5) |

`tests/test_curriculum_split.py` (37 tests) green against the change.

**Three of the 36 frozen cells GAIN val rows** — `5-5.25_f117`,
`5-5.25_f125`, `5.5-5.75_f125`, +40 each. That is deliberate and it is the point
of §3.2: it is the R3-mandated held-out slice in the cells that were densified.
The historical 3,207-row AB2 surface is **preserved verbatim** in the manifest
under `groups['ab2_frozen_val_by_cell']` (+ `ab2_frozen_n_rows = 3207`), so
rounds 1–3 and BU stay exactly comparable even though `curriculum_val_by_cell`
has grown. **Any future AB2-style paired bootstrap must restrict to that key,
not to the grown `curriculum_val_by_cell`** — registered here so a later reader
cannot silently score on a moved surface and compare to MDE₈₀ = 0.00409.

### 3.4 Fairness of the grown val cells

Neither model trains on the 120 new val rows: the champion never saw any of the
1,413, and the candidate trains only on the other 607 + 686. So the grown cells
are a genuinely held-out comparison for both. What it does **not** control for is
that the candidate has seen 80% of that *distribution* and the champion 0% —
which is exactly the effect under test, not a confound, but it means a gain on
those three cells is evidence about **densification**, never about the recipe.

### 3.5 What the increment cannot fix

**Not one of the 68 sub-1.50 cores is in val** (verified: 0 of 68). They are all
legacy-pool rows and all route to train. So this round adds the low-F_r region to
*training* and adds **zero** low-F_r *evaluation*. The §5.3 secondary readout is
therefore measured on the pre-existing `f_r ≤ 1.55` val rows, which remain the
same rows the champion was measured on — a fair before/after, but a thin one.
Stated now so a null on it is read correctly.

---

## 4. The arm

Single arm. v6b recipe **exactly** as `bu_T`, one change: `--split S1b`.

```
python -m lpopt.model.train \
  --ensemble 5 --split S1b --cond-schema v6b --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --parallel-members 5 --base-seed 20260716 \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

Identical to the BU launch string except `--split S1b` (was `S1`). Seeds
20260716–20260720. GPU 0 only, after an idle check.

**Registered consequence of the recipe being fixed:** the power prior's
`(M², extrap)` is refit per run on the new train rows, so it may land on
different constants than the champion's `(150, 1.0)`. That is the design (fit and
serve must agree, `ab2_addendum_BU_20260810.md` §7.1) and is **not** a second
change; it is recorded here so it cannot be produced later as one.

---

## 5. The decision rule — fixed before any result exists

### 5.1 Instrument

**`lpopt gate-promote --check-only`**, previous = `data/models/20260810_bu_T`,
new = the pulled candidate. That is the repo's standing promotion authority
(no-regression across the curriculum `val_by_cell` cells + the legacy tail gate);
this addendum does not invent a threshold, it names the instrument and fixes
what is reported alongside it.

Both models are scored by the gate on the **same** split — `S1b` — so the
comparison is on identical rows in identical order.

### 5.2 Primary

The gate's own verdict. **PASS ⇒ promotion-eligible; FAIL ⇒ the rebuild is not
promoted and `20260810_bu_T` stands.** No axis of §5.3 may override it, and a
gain there may not be substituted for a gate failure.

### 5.3 Registered SECONDARY readout — reported either way

#### 5.3.1 The `f_r ≤ 1.55` slice is EMPTY. Measured before any result exists.

The readout this arm was asked for — within-cell served `f_r` Spearman on the
`f_r ≤ 1.55` slice — **cannot be computed on this surface at all**:

| split | val rows (converged) | `f_r ≤ 1.55` | `f_r < 1.50` | min val `f_r` |
|---|---:|---:|---:|---:|
| S1 | 11,338 | **0** | **0** | 1.5902 |
| **S1b** | 11,458 | **0** | **0** | 1.5821 |

Only **19** val rows sit below 1.60, and 17 of those are in a stale-cache
campaign. This is structural, not a consequence of the rebuild: **S1 had zero
too**, so the champion was never measurable on that slice either.

**Registered consequence:** the `f_r ≤ 1.55` readout is **VOID** and will be
reported as `n = 0`, never as a null. It follows that the memo's
champion-family **−0.018 [−0.19, +0.14]** and the v520 pools
**+0.31 / −0.31 / −0.14** were measured on *other* surfaces (fold C / the C2
slice / 20-core elite pools), so **no number this arm produces may be compared to
them.** Recording this now is the same discipline R3 §5.6 applied when it
disclosed, in advance, that its secondary observation was below `MIN_CELLS` and
could carry no interval evidence.

Absolute bands do not rescue it, because `f_r` *level* is almost entirely a
cell property (feed × enrichment), so an absolute cut selects whole cells rather
than the elite within them — `f_r ≤ 1.65` yields 474 rows but only **1** cell
with ≥ 8 rows; `f_r ≤ 1.75` yields 2,170 rows and only **4**.

#### 5.3.2 The substitute, fixed now — the WITHIN-CELL elite band

**Within-cell served `f_r` Spearman on the bottom quintile of true `f_r` inside
each cell**, champion vs candidate, on the same val rows, cells with ≥ 8 rows in
the band, aggregated by median over cells.

| property | value |
|---|---:|
| rows in band (S1b val) | **2,299** |
| cells scored (≥ 8 rows) | **41** of 42 |
| per-cell n | median 18, min 1, max 864 |
| band `f_r` range | 1.5821 … 2.2936 |

Why this and not another cut: it is what the optimizer actually does — rank
candidates *inside* one design cell and take the flattest few — and it is the
axis KILLER 2 and the v520 series put at ρ ≈ 0. It is well-defined, it has usable
n, and it is computed identically for both models on identical rows.

**It is a NEW quantity with no historical baseline.** Its champion value is
established by this run; it may not be compared to −0.018 or to the v520 pools.

#### 5.3.3 Expectation, stated honestly and in advance

With 68 new sub-1.50 cores against 50,445 existing training rows — **0.13% of the
corpus** — and **zero** new sub-1.50 *evaluation* rows (§3.5), and with the band
itself bottoming out at 1.58 rather than in the sub-1.50 region the new data
covers, **this may well not move, and it is not well positioned to detect the
effect even if the effect is real.** A null here falsifies nothing. A gain would
be the first positive signal on the ranking axis and would still need its own
round, on a surface that actually contains sub-1.50 rows, to confirm.

Reported alongside, deciding nothing: `f_r` MAE in the band, per-cell ρ and n,
and the (void, `n = 0`) `f_r ≤ 1.55` figures for the record.

### 5.4 What this round cannot conclude

Registered before the result, because it is the weakest link:

> There is **no same-recipe-same-data control**. The comparison is
> candidate(new data, new split) vs champion(old data, old split), so a gate PASS
> confounds "the new rows helped" with "an independent training run of the same
> recipe landed better". The AB2 rounds spent an arm on exactly that control for
> exactly this reason. **A PASS therefore promotes a model; it does not establish
> that data growth is a lever.** Claiming the latter needs a fresh
> same-data control, which this round deliberately does not buy.

### 5.5 Falsification

> If the gate FAILS, the rebuild is rejected and `20260810_bu_T` stands. The
> registered reading is **not** "more data does not help" — it is that this
> particular 1,413-row increment, dominated by 727 in-cell `fadj_*` rows and
> carrying only 68 sub-1.50 cores, did not clear the no-regression bar. The next
> move would then be to grow the low-F_r region deliberately (production, not a
> re-split) rather than to re-run this arm on a slightly larger store — a second
> undirected growth round scored the same way would carry no more information
> than the first (R3 §5.8, inherited verbatim).

---

## 6. Launch record — 2026-08-10, GPU 0

| | |
|---|---|
| remote run id | `runs/split_S1b` |
| tmux session | `lpopt_split_S1b` |
| GPU | 0 (idle before launch: 237 MiB, 0%, no compute apps, no tmux) |
| launched | 06:42:46 |
| power prior (refit on S1b train) | M² 150, extrap 2.0, n_fit 3,998, ρ 0.6888 |

Champion for reference: M² 150, extrap **1.0**, ρ 0.6923 — the constants moved,
exactly as §4 registered they would. Not a second change.

**Snapshot verified ON THE REMOTE after push** (`sha256sum` in `~/lpopt_ws`),
all five matching the §2 / §3.3 pins:

| file | remote sha256 |
|---|---|
| `data/store/records.parquet` | `00a6ecb0…33d75027` |
| `data/store/maps.npz` | `903ee1da…306666b3` |
| `data/store/fuel_types.parquet` | `4ee9b16e…9c5c8ecd` |
| `data/splits/S1.json` | `6ab35f25…ead650640` |
| **`data/splits/S1b.json`** | **`47c19989…858f1c5b`** |

`S1b` was additionally re-verified *inside the server venv* after transfer:
train/val 59,634/11,521, old-val ⊆ new-val, old-train ⊆ new-train,
`ab2_frozen_val_by_cell` identical to S1's `curriculum_val_by_cell`, v6b = 58
channels.

---

## 7. Scoring, pull and promotion — ready before the arm finishes

**Stage 1 — pull the candidate** (the issuer does this; no local write happens
until then):

```
python -m lpopt.remote --input lpopt.inp pull --ts split_S1b
# -> data/models/split_S1b/
```

**Stage 2 — the PRIMARY instrument, gate-promote (§5.1/§5.2).** `--check-only`
is mandatory for a look: without it a PASS promotes immediately.

```
python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/20260810_bu_T \
    --new  data/models/split_S1b \
    --out  data/reports/gate_split_S1b.json \
    --check-only
```

**Stage 3 — the SECONDARY readout (§5.3), reported either way.** Both models are
served on **S1b** so the rows and their order are identical:

```
ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/split_S1b split_cand  S1b && \
    ./venv/bin/python eval_accuracy_split.py runs/bu_T      split_champ S1b'

cd 5_RL/runs/split_S1b
scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_split_cand.csv .
scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_split_champ.csv .

python 5_RL/split_secondary_readout.py
# -> data/reports/split_secondary_readout_20260810.json
```

`eval_accuracy_split.py` is a split-aware copy of the box's `eval_accuracy.py`
(which hardcodes `S1.json`); it was shipped to `~/lpopt_ws/` before launch.
Serving the SPLIT arm with the stock script would silently score it on the wrong
val fold. `runs/bu_T` on the box is the same ensemble as
`data/models/20260810_bu_T` locally.

Both scorers were smoke-tested end-to-end on synthetic arms before the training
landed, so neither can fail for a mechanical reason on the night the run finishes.

---

## 8. Cost

Single arm, 5 members, `--parallel-members 5`, GPU 0. ~2.5–3 h wall (the BU arm's
two concurrent arms took ~25 min of CPU featurisation + GPU training; a single
arm has the GPU to itself). Serve cost unchanged — same architecture, same 58
channels, same 5 members.
