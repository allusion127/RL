# Policy net v2 — pre-registration

Written 2026-08-17, **before any v2 weight was trained**. This document is not
edited after training; every deviation is reported in
`data/reports/policy_v2_results_20260817.md`.

| artefact | fingerprint |
|---|---|
| corpus `data/policy/steps.parquet` | SHA-256 `fe53ac811131157f…` (28,084 rows × 80 cols) |
| blind v1 baseline `data/design/policy_v2_v1_baseline.csv` | SHA-256 `5ea778e908eecf6e…` (2,938 eval rows, 5 members) |
| code | `lpopt/policy/v2.py`, `lpopt/policy/train_v2.py`, launcher `train_policy_v2.py` |
| unchanged | `lpopt/policy/{data,net,train,scorer}.py` — v1 is a BASELINE this round and its feature layout is frozen |

---

## 0. Why there is a v2 at all

v1 passed its gate on the legacy SA corpus and then **failed prospectively at the
live operating point**: on 146 balanced, interventional, single-move ga80/paramA
children it scored parent-blocked AUC **0.492** on `fr` — chance — and p@32
0.094 against random's 0.156 (`ablation_wave_results_20260815.md` §3). Its single
largest commitment, a 0.64 probability gap favouring outward `batch_flip`, was
maximally wrong: 0 of those 10 improved while the inward `batch_flip` moves it
ranked lowest improved 3 of 10.

The post-mortem (§9) prescribed three fixes. All three are implemented here and
each is stated below with the measurement that motivates it. **This round is not
an attempt to make v1 better; it is a test of whether those three specific
corrections buy a move ranker at the era where v1 has none.**

---

## 1. Corpus, and what changed in it

### 1a. Delta mined this round

| source | rows | note |
|---|---:|---|
| `fpcamp_minfr_N1N2_f113` | **+21** | mined with `ablation_analyze.py corpus --campaign fpcamp_minfr_N1N2_f113 --lineage lpopt_genome`, i.e. `mine_policy_corpus.build_steps` itself |
| `fpcamp_minfr_E1E2_f109` | **0** | see §1c |

`steps.parquet`: 28,063 → **28,084**. Backup
`steps.parquet.bak_pre_fpcamp_minfr_N1N2_f113`. `data/store` was not written.

**The f113 yield is 21 edges from 100 MASTER calls, not ~100, and the reason is
structural rather than a harvest gap.** A `build_pool` child's parent is very
often a pool seed that was never itself evaluated — 96 of the 100 f113 records
carry a `parent_record_id` but only 21 of those parents exist in the store, so
only 21 rows are lineage edges at all. Of the 21, 14 are `feed_change_multi`
cross-cell edges (the parents come from `alsearch_N1_N2_f121/f125`, i.e. a
different feed) and are outside the policy's universe; **7 are usable same-cell
moves.** An OPEN campaign is a poor corpus source per MASTER call, and that is
worth registering before the fact rather than discovering it in the results.

An audit of the whole store confirms the corpus was otherwise already complete:
of 6,318 current-era lineage edges reachable from `data/store/records.parquet`,
6,297 were already mined and the f113 21 were the entire remainder.

### 1b. Schema change — `fresh_enr_mass` (post-mortem prescription 2)

`mine_policy_corpus.PHYSICS` gains `fresh_enr_mass`, the multiplicity-weighted
**total** fresh enrichment, so `build_steps` now emits
`parent_/child_/d_fresh_enr_mass` like every other physics descriptor. The
definition is `ablation_wave._fresh_enr_mass` verbatim (`Σ mult · enrichment`
over fresh slots); `ablation_wave.py` itself is **not** edited — its SHA-256 is
pinned in the ablation pre-registration.

The 28,063 pre-existing rows are backfilled by `policy_v2_corpus.py backfill`
from the `parent_pattern` / `child_pattern` strings the corpus already stores,
using `ring_profile` itself. Backup `steps.parquet.bak_pre_fresh_enr_mass`.
Measured on the same-cell corpus, which is the check that it is the right
coordinate:

| move class | n | max &#124;d_fresh_enr_mass&#124; |
|---|---:|---:|
| `rewire_swap` | 10,566 | 0.000000 |
| `batch_swap` | 477 | 1.1e-13 |
| `fresh_relocate` | 6,095 | 4.44 (56 rows nonzero — relocations between units of different multiplicity) |
| `batch_flip` | 1,458 | **1.2017** (1,450 rows nonzero) |

That is exactly the separation v1 could not make. `batch_flip` has a radial dose
of ~1e-5 and is the one operator that changes how much fresh reactivity is in the
core; the ablation wave measured `corr(d_cyclen, d_fresh_enr_mass) = 1.000` on it
against `corr(d_cyclen, d_fresh_enr_r_center) = 0.343`. Without this column the
network must express "outward" and "more reactivity" in the same coordinate, and
v1's failure is what that looks like off-distribution.

The appender was re-validated after the schema change: re-mining
`fpcamp_minfr_T6T4` reproduces its 41 canonical rows with **every one of the 80
columns identical** (`policy_v2_corpus.py verify`).

### 1c. The f109 cutoff, declared

`fpcamp_minfr_E1E2_f109` is **running on 199** and is left strictly alone. Its
read-only status probe at 2026-08-17 (`status_fpcamp_E1E2_f109_199.ps1`) reports
`wave_index 2`, `budget_spent 16` of 100, best-so-far F_r 1.5547 (infeasible),
and **zero rows merged into the local store**. It therefore contributes **no
rows** to this round, and the "fully-held-out cell/era test" the brief hoped for
does **not** exist here. That readout is deferred; it is not silently replaced by
something weaker.

---

## 2. What is being learned — the target redesign (prescription 3)

Two heads, as in v1, but the label is no longer binary:

```
y_head = min( max(0, -Δ_head), c_head ) / c_head          ∈ [0, 1]
```

with `Δ_fr = d_f_r`, `Δ_flat = d_node_peak`, and

| head | clip `c` | rule |
|---|---:|---|
| `fr` | **0.030** | the smallest 5e-3-rounded value that leaves every improvement in the current-era interventional set unsaturated (largest measured `-d_f_r` there: 0.0265) |
| `flat` | **0.035** | same rule (largest measured `-d_node_peak`: 0.0341) |

This is the **normalized clipped expected improvement**. The choice is registered
as ONE choice, from the distribution analysis below; there is no sweep over `c`,
over the loss, or over the alternative target.

### 2a. Why not the improving fraction

`improved_fr` is magnitude-blind. It scores a +1e-4 nudge off an elite and a
−2.18 rescue off a broken board identically. The ablation wave §2d measured what
that costs at the live era: inward `fresh_relocate` improves **14.3%** of the
time and outward **3.6%**, so the fraction ranks inward first — yet inward's mean
`d_f_r` is **+0.539** against outward's **+0.100**, because inward is a lottery
(catastrophic to +1.86) and outward is reliably mild (bounded at +0.21). Anything
trained on the fraction inherits that preference. v1 was.

### 2b. Why not the "reliable improver" band either

That was the registered alternative and **the data rejects it at both settings.**
On the current-era interventional rows (577 verified single moves off elite
parents), every one of the 42 improvements already has `|d_f_r| ≤ 0.0265`:

| band | improvements | "reliable" | identical to `improved_fr`? |
|---|---:|---:|---|
| 0.03 | 42 | 42 | **yes** |
| 0.05 | 42 | 42 | **yes** |
| 0.02 | 42 | 39 | no |
| 0.01 | 42 | 30 | no |

So a band at or above 0.03 is a **relabelling that changes nothing on the rows
the gate is decided on**, and tightening it to 0.01 changes something in the
wrong direction: it converts 12 of 42 genuine improvements into negatives,
including the *largest* ones — precisely the moves the deployment metric is
asking the policy to find. A band is still a binary, and a binary cannot say
"this move improves by 0.026 and that one by 0.001".

### 2c. Why the clip is load-bearing

Pooled over the whole corpus the improvement magnitude spans four orders — p50
0.019, p90 0.32, p99 1.63, max 2.18 — and essentially all of that spread is
**parent difficulty**, not move quality. An unclipped `E[max(0, −Δ)]` regression
would put nearly all of its gradient on a few hundred rescue-a-broken-board rows
and would learn the same board-difficulty signal the pooled deployment metric is
already confounded by (v1 prereg §4b, and the very failure mode §10 of that
document named). Clipped at the elite band, the target keeps **full magnitude
resolution exactly where the gate and the consumer live** — 0% of the
interventional improvements saturate — and flattens the difficulty tail to a
constant (24% of legacy improvements and 41% of pooled current-era improvements
saturate).

### 2d. Why expected improvement specifically

The consumer is `autoeng`'s PROBE stage: **8 MASTER calls, keep the best board**
(`autoeng.Target.probe_budget = 8`). Maximising the best of *k* draws is what
`E[max(0, −Δ)]` is the matched acquisition for; the improving fraction is the
probability-of-improvement acquisition, which is the magnitude-blind one. The new
deployment metric (§4d, `regret@8`) is the empirical form of the same quantity,
so target and metric are one object measured twice — deliberately, because the
v1 round's lesson is that a metric which does not look like the consumer will not
predict the consumer.

**Loss:** Huber (`smooth_l1`, β = 0.2 on the normalized scale) on the **sigmoid**
of the logit, masked per head, era-weighted (§3b). Sigmoid rather than a linear
head because the target is bounded in [0,1] by construction — and because it
keeps the serving contract byte-identical: `lpopt/policy/scorer.py` already
applies `sigmoid` and returns `[n, 2]` in [0,1], so v2 drops into the existing
`policy_prior` path without a line of change there.

---

## 3. Era handling

### 3a. Era as an input

`era_current` (1 if `library_id ∈ {ga80, paramA}`, else 0) joins the conditioning
vector. This is **not** the provenance leak v1 excluded: the era is a property of
the *cell*, named by the deck before any move is proposed, whereas
`lineage_source` describes how a row was produced and remains excluded. The
distinction is that a move v2 invents has an era and has no lineage source.

### 3b. Reweighting, with no knob

Training spans both eras. Per-row loss weight is `1.0` on legacy and
`w = n_train_legacy / n_train_current` (capped at 20) on the current era, so the
statement being registered is simply **"the two eras count equally"** and there
is no weight to tune. Realized: `n_legacy = 16,579`, `n_current = 815`, ratio
20.3 → the cap binds at **20.0**, giving the current era 49.6% of the loss mass
rather than exactly 50%. Declared in advance so it is not read later as a choice.

---

## 4. Splits and metrics

### 4a. The gate fold — current era only

Lineage connected components are computed **within the current era**; components
are ranked by labelled-row count (descending, ties by component key) and assigned
**alternately** — rank 0 → gate, rank 1 → train, rank 2 → gate, … No RNG and no
label enters this: component size is known before any outcome is read.

Alternation rather than a random draw because the scarce quantity is lumpy. Only
17 current-era components carry ≥ 12 labelled candidates, and `regret@8` needs a
parent with enough candidates for a top-8 to be a real selection; a random 50/50
over 544 components would routinely pile the high-fan-out mass on one side.

`val` is 10% of the remaining components, drawn independently inside each era,
and is used for early stopping and nothing else. **No legacy row is ever in the
gate.** Realized:

| fold | rows | current-era | interventional | cells | parents | base `fr` | mean `y_fr` |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17,394 | 815 | 296 | 78 | 3,078 | 0.347 | 0.163 |
| val | 1,932 | 90 | 0 | 40 | 379 | 0.346 | 0.162 |
| **gate_cur** | **1,006** | **1,006** | **281** | 46 | 369 | **0.143** | 0.074 |

`gate_cur` composition: `lpopt_genome` 725 / `batchswap_enum_625` 178 /
`ablation_paramA` 58 / `batchswap_enum` 45, and 4 of the 7 usable f113 rows.
15 of its parents carry ≥ 8 labelled candidates and **9 carry ≥ 10**, which is
the `regret@8` sample.

### 4b. Ranking — AUC and parent-blocked AUC, on `improved_*`

**The evaluation label is v1's binary label, not v2's target.** The training
target changed; the yardstick deliberately did not, so the comparison against v1
is on v1's own terms and cannot be won by redefining success. Parent-blocked AUC
is computed per parent (over same-parent improving/non-improving pairs, ties
half) and bootstrapped by **resampling parents**, which is the analysis unit the
gate fold was split on.

### 4c. Deployment metric, v1's shape — precision@32 of 256

Unchanged from v1 §4b, so the two rounds' numbers are comparable: 2,000
replicates, all scorers see identical draws and an identical per-replicate random
tiebreak, paired bootstrap on every difference.

### 4d. NEW — regret@8, shaped to the consumer

For every gate parent with **≥ 10** labelled candidates (a top-8 out of ≥10 is a
real selection; at exactly 8 every scorer scores 0):

```
gain      = −Δ                      (positive = improvement)
regret_p  = max(gain over all candidates) − max(gain over the 8 the scorer picks)
```

averaged over 256 random tiebreak draws, then over parents, with a paired
parent-bootstrap CI. Reported both absolutely (in F_r units) and **normalized**
by each parent's own `max(gain) − min(gain)` spread, so a mean over parents of
different difficulty means something. Note that a parent whose candidates are all
worsening still contributes: picking the least-bad of a bad neighbourhood is a
real ranking skill and is what a probe on a hard cell actually needs.

**n = 9 parents.** That is small, it is the honest maximum this corpus supports,
and it is why regret@8 gates the *recommendation* and not the model (§7).

### 4e. Reported, not gated

Calibration (Brier/ECE against `improved_*`), RMSE against the v2 target, the
`val` fold, per-seed spread, and the `flat` head throughout.

---

## 5. Baselines — the policy must beat all FOUR

| id | score | why |
|---|---|---|
| `random` | uniform noise | the floor |
| `class_freq` | train improving-rate of the row's `move_class` | the lookup table that **beat v1 on every metric** at this era (ablation §3) — the real incumbent |
| `periph` | `d_fresh_share_periph` | the engineer's radial rule of thumb |
| **`policy_v1`** | the shipped 5-member v1 CNN ensemble's `P(improve)` | **the bar this round exists to clear** |

`random` / `class_freq` / `periph` are fitted on `train` only. `policy_v1` is not
fitted at all: its probabilities are produced by
`python -m lpopt.policy.train_v2 --emit-v1-baseline …`, which loads
`data/models/policy_v1/cnn_seed*` and scores the eval folds through **v1's own**
`scalar_features` and delta channels, with a hard assertion that v1's scalar
layout has not drifted. The CSV is written and hashed **before any v2 weight
exists**, exactly as the ablation wave hashed its blind predictions.

---

## 6. Architecture, and the transfer-init decision

**One arm, `cnn`** — v1's PosValNet-trunk geometry unchanged (width 112, 6
residual blocks, FiLM after blocks 1/3/5, masked mean+max pool, 2 logits).
`cnn`-vs-`mlp` was v1's registered comparison and it was decisive (0.790 vs 0.643
test AUC); re-running it would spend compute on a settled question.

Inputs differ from v1 in exactly three scalars — `d_fresh_enr_mass`,
`parent_fresh_enr_mass`, `era_current` — taking the conditioning vector from 49
to 52 dims. `scalar_features_v2` **calls** `scalar_features` rather than copying
it, so v1's 36 columns are bit-identical inside v2's 39 and the two models differ
by the additions and by the target, and by nothing else.

The reactivity covariate enters as a **conditioning scalar, not a board channel**,
because `fresh_enr_mass` is a board global and the globals are what FiLM
modulates the trunk with. The per-slot fresh-enrichment field is already in the
delta block of the board tensor; what was missing was its *total*, which is a
scalar.

### 6a. s1g transfer-init is DECLINED — and why

The brief allows it. It is declined, and the reason is the target, not habit.

v1 declined an s1e warm start because s1e was trained with direct supervision on
the `f_r` and `node_peak` of the very store records that are this corpus's
parents and children (v1 prereg §6a, reason 2). **That objection is strictly
stronger for v2, not weaker.** v1's label was a comparison (`child < parent`);
v2's label is `-d_f_r` itself, clipped and scaled — i.e. a monotone function of
the exact quantity s1g was trained to regress on these exact boards. Every
`gate_cur` parent and child is in s1g's training store. An s1g-initialised policy
would carry indirect access to the held-out target and every number in §7 would
be unreadable.

The clean version remains the one v1 named: train a width-112 surrogate on a
split that *excludes* the policy's gate components, then run the comparison. That
is a separate round and is not this one.

---

## 7. GATE, and the separate recommendation bar

Evaluated on **`gate_cur`**, head **`fr`**, on the 5-seed probability-mean
ensemble.

**GATE — PASS requires BOTH:**

1. **parent-blocked AUC ≥ 0.65**, and its 95% parent-bootstrap CI lower bound
   **> 0.50**. (0.65 is the bar the ablation pre-registration set for v1 on this
   era. It is not lowered.)
2. **precision@32 greater than all four baselines** — `random`, `class_freq`,
   `periph` and `policy_v1` — with the **paired** bootstrap 95% CI on each
   difference excluding 0.

**FAIL otherwise. No partial credit, no post-hoc metric substitution.** A single
clause passing is reported as a single clause passing.

**RECOMMENDATION BAR — separate, and it governs deployment, not the model.**
Wiring v2 into `autoeng`'s PROBE stage additionally requires **regret@8 strictly
lower than `random` and than `policy_v1`**, each with a paired parent-bootstrap
95% CI on the difference excluding 0. Kept separate because n = 9 parents: a
metric that thin should not be able to fail a model that has demonstrably learned
a ranker, nor to license a deployment on its own.

The `flat` head is **reported in full and not gated.** Its current-era labels are
sparser (742 gate rows vs 1,006) and the candidate generator the mission is
aimed at is the F_r one; gating a head this round on a thinner sample would be
buying a coin flip.

---

## 8. Leakage discipline

`lpopt/policy/data.py::FORBIDDEN_COLUMNS` is unchanged and still enforced —
`scalar_features` raises if any outcome or provenance column reaches the feature
frame, and `scalar_features_v2` calls it. The three additions are checked against
that list explicitly:

* `d_fresh_enr_mass` / `parent_fresh_enr_mass` — functions of the parent pattern
  and the candidate pattern under the cell's fuel table. Known before evaluation,
  same standing as the `d_*` ring descriptors v1 already used.
* `era_current` — a function of `library_id`, a cell attribute (§3a).

**Parent FOMs stay excluded**, and this is now a stronger decision than it was
for v1: §2c shows the target's whole failure mode is parent difficulty leaking
into a move score, so handing the model the parent's own `f_r` would re-open by
the front door the confound the clip closes.

---

## 9. Falsification reading — what each outcome means

Spelled out now so it cannot be rationalised later.

* **Parent-blocked AUC ≈ 0.5 again.** The three prescriptions do not buy a
  current-era move ranker, and the deficit is DATA, not framing. The corpus
  audit in §1a says the store is exhausted, so the answer would be another
  interventional wave at a *different* cell/feed — not a fourth feature idea.
  **Recommendation would be: no wiring, and a second ablation wave.**
* **Parent-blocked AUC clears 0.65 but p@32 does not beat `class_freq`.** The
  learned signal is move-class and nothing more. Ship the lookup table as the
  generator prior and close the policy line. This is a live possibility:
  `class_freq` reached 0.658 parent-blocked at this era while v1 reached 0.492.
* **Both gate clauses pass but regret@8 does not beat `policy_v1`.** v2 ranks
  *whether* a move improves better than v1 does, but not *which* improvement is
  biggest — i.e. the target redesign did not transfer to the consumer's
  objective. Report as a gate pass with the deployment recommendation withheld,
  and the next step is more candidates per parent (the enumeration wave), not
  more model.
* **Gate passes and regret@8 clears.** Wire behind `policy_prior` with the v1
  safety rails intact (softmax, random floor, within-parent only) and run the
  paired A/B on elites-per-MASTER-call. A report is not an A/B.
* **v2 beats v1 on the gate but both are below `class_freq`.** The honest reading
  is that the era is carried by move class alone and the board tensor is not
  earning its place *here*, whatever it did on 260624. Say so plainly.

---

## 10. Training protocol

Fixed in advance; no sweep.

| knob | value |
|---|---|
| loss | Huber (`smooth_l1`, β = 0.2) on `sigmoid(logit)`, masked per head, era-weighted |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, grad-clip 5.0 |
| schedule | cosine to 120 epochs |
| batch | 256 |
| early stop | **era-weighted val LOSS**, patience 15 |
| augmentation | diagonal-mirror transpose, p = 0.5, train only (v1's validated index-swap recipe) |
| seeds | 5 (20260817 … 20260821) |
| headline | the 5-seed mean-probability ensemble; per-seed spread reported |
| no TTA | deterministic single-orientation evaluation |

Early stopping changed from v1's val AUC because v2's objective is a regression
and the current-era slice of `val` is 90 rows; a val AUC would be both a
different objective from the one being minimised and a very noisy one. The
held-out value of the actual objective is the like-for-like, low-variance choice.

---

## 11. Compute

238 (`USER@HOST_238:8022`), venv `~/lpopt_ws/venv`, pinned to **GPU 1** via
`lpopt_gpu1.inp`. Occupancy checked before launch: GPU 0 at 95% / 18,933 MiB
(another tenant), **GPU 1 at 0% / 1 MiB**, no tmux sessions. GPU 0 is left alone.
Markers follow the established convention (`heartbeat` every 15 s, `rc`,
`DONE`/`FAILED`) so `lpopt remote status/pull` work unchanged.

**Untouched, per the round's constraints:** 199 (the running `f109` campaign),
198 (the meshv3 anchors), and 181.
