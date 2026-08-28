# Capacity & ensemble scaling — pre-registration

**Written 2026-08-15, BEFORE any weight was trained.** Every number in §2 was
read off artifacts that already existed on disk; every number in §5–§7 is a rule,
not a result. Results go to `data/reports/scaling_results_20260815.md`. **This
document is not edited after the first run launches**; deviations are reported
there.

**Question being answered** (user, verbatim): *"파라미터 수나 앙상블 학습 수를
늘리면 더 나은 결과가 나오지 않을까"* — does scaling model capacity and/or
ensemble size improve (a) prediction accuracy and (b) search-relevant quality?

**No promotion in this round.** Even if the largest arm wins every registered
metric, it does not become the champion here. Promotion goes through the normal
`lpopt gate-promote` flow in a later, separately-registered round. This is stated
first so that a win cannot later be converted into a promotion after the fact.

---

## 1. Correction to the commissioning premise — read this before the design

The task that commissioned this experiment described the champion as
**"~1.5M params/member, conv stem 64 + 6 residual blocks"**. That is the
**library default** (`PosValNetConfig(width=112, n_blocks=6, head_hidden=256)`,
whose docstring quotes the 1.0M–2.5M acceptance band and whose `build_member`
asserts it). **It is not what the champion is.**

Read from `data/models/s1e/member_20260716/meta.json` and `s1e/run.sh`:

| | commissioning premise | **measured champion `s1e`** |
|---|---|---|
| width / blocks / head | 64 (or 112) / 6 / 256 | **224 / 8 / 384** |
| params per member | ~1.5M | **10,363,715** |
| cond schema | — | `v6b` |
| members | 5 | 5 (seeds 20260716–20260720) |
| split | S1e | S1e — train **60,810** / val **11,775** |
| best epoch (seed …716) | — | 68 of 150 (early stop, patience 15) |

The champion is **6.9× larger** than the premise assumed. It is the descendant of
the round-2 A/B arm **A6** (10,343,547 params), which was *itself* a
"structure + intermediate capacity" arm. **The capacity increase the question
proposes has already been taken once, and is already in the champion.**

Two consequences, both binding on the design:

1. The requested ladder (1.5M / 6M / 24M) is not anchored on anything real. It is
   re-anchored on the measured champion in §4, keeping the requested *span* and
   the requested *number of points*, and the deviation is documented there.
2. The commissioning brief's stated prior — *"evidence so far suggests the
   binding constraint is label density/noise, not capacity"* — is **contradicted
   by this repo's own measurements** on the noise half. §2 records what is
   actually measured, because a pre-registration that carries a false prior
   cannot state an honest falsification condition.

---

## 2. Prior art — capacity is not an untested axis. It has been tested three times.

### 2.1 What has already been run

| # | test | control → treated | params | measured effect |
|---|---|---|---|---|
| 1 | "A안 2× 용량 시험" (`dataset_adversarial_20260721.md`) | width 112 → 160 | ~2.2M → ~4.3M (2×) | **+0.011** within-cell rank |
| 2 | v5 `w160` arm (`v5_arm_dirs.json`, `data/models/20260721_105824`) | width 112 → 160 | 2× | folded into the same null |
| 3 | **A5 "순수 용량 확대"** (`hires_model_ab_design_20260725.md` §4.1, `hires_ab_verdict.json`) | `--width 256 --n-blocks 10 --head-hidden 512` vs B1 | **3,159,291 → 13,143,835 (4.2×)** | **primary gain +0.141** |

A5 was pre-registered as a **null-hypothesis arm** ("+0.02 이하"). Its null was
**rejected** — `hires_ab_verdict.json.capacity_clause.null_rejected == true`.
**Pure capacity does something.** But the same round measured, on the same rows:

| arm | params | primary gain | gain per 1M params |
|---|---:|---:|---:|
| **A1** (multiscale decoder, *structure*) | 4,312,251 | **+0.705** | +0.61 |
| **A5** (pure capacity) | 13,143,835 | +0.141 | **+0.014** |

**44× worse per parameter.** And A5 regressed two things capacity was supposed to
fix: its Nyquist power ratio was **0.441 vs the control's 0.451** (pure width made
high-wavenumber attenuation *worse*), and its honest-gate drop **0.294** was the
**worst of all eight arms**.

### 2.2 What the noise/density work actually concluded

`precision_ceiling_roadmap_20260725.md` is the report the commissioning brief's
prior points at. It concludes the **opposite** of that prior:

* **Label noise is not the ceiling.** Measured ρ upper bound is **≥ 0.997 on
  every target** (direct estimator: equilibrium convergence residual,
  `delta_efpd` RMS 0.104 EFPD on 38,851 rows). Lever ① "tighten MASTER tolerance"
  is **rejected** on measurement.
* **The gap is model error**, needing a **3.7×–12.4×** error reduction to reach
  ρ = 0.99.
* **Data density alone cannot close it either** — the learning-curve exponent is
  only **−0.21 to −0.38**, implying 55k–920k rows *per cell* for ρ = 0.99 against
  a current median of **66**. Density is scored a lever for 0.90–0.95, not 0.99.
* That report's own lever table ranks **"순수 용량 확대" last but one**, with an
  expected **+0.00–0.02**, and flags "세 번째 무효 확정 가능성".

So: neither "it's the labels" nor "it's the data" is the repo's measured position.
The measured position is **"it is the model, and the part of the model that has
paid is structure, not width."** This experiment tests whether that holds *above*
the champion's current 10.4M, which no prior run has probed.

### 2.3 Registered prior (stated before results, so it can be wrong on the record)

> Scaling from 10.4M to ~25M and ~38M under the champion recipe will produce
> **< +0.02 iso-ρ on every target**, and will **not** improve σ-calibration.
> The 0.46× point (4.8M) will lose **< 0.02** relative to the control — i.e. the
> champion is at or past the knee, and the curve is flat in both directions.

This is the hypothesis this round tries to **break**, not confirm. §7 fixes what
counts as breaking it.

---

## 3. Frozen inputs

Identical to the `s1e` arm (`ab2_addendum_S1E_20260815.md` §2); re-verified on 238
on 2026-08-15 before writing this document.

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `f854e19646088c976360a6601bcf5da3e1fd3fc58f2470817a2a79678311557f` | 21,580,111 |
| `data/store/maps.npz` | `70fa87cbe569774631ec71658eb941ce9cce86a58538d94d41a871777224de0e` | 196,675,587 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| `data/splits/S1e.json` | (5,847,146 B, unchanged since 2026-08-15 00:23) | 5,847,146 |
| distill cache `data/models/_v5_distill_soft.npz` | (3,259,641 B, 2026-07-27) | 3,259,641 |

**Read-only.** No store regeneration, no split rebuild, no MASTER call, no touch
of 199. `data/store` is not written by any step of this round.

### 3.1 Evaluation surface

`S1e` val fold, converged rows only — the same filter
`eval_accuracy_split.py` applies.

| | |
|---|---:|
| val rows (converged) | **11,712** |
| cells | **58** |
| cells with ≥ 8 rows | **50** |
| cells with ≥ 64 rows (the precision@32 surface) | **26** |
| rows with a finite map label (`node_peak`/`map_cov`) | **2,371** (20.2%) |
| map cells with ≥ 8 rows | **48** |
| `T6_T4` val rows | **144** in 6 cells (5 at ≥ 8 rows) |

**Registered caveat on pooling.** Two cells
(`0_Case:sa_2b_cache.stale-fb857c7a` 4,319 rows and
`…-2c04d78c` 3,342 rows) hold **65% of the val fold**. Any pooled statistic is
therefore a statistic about those two cells. **Every headline number in this
round is a MEDIAN ACROSS CELLS, never a pooled mean**, and the per-cell vector is
published. This is fixed now so it cannot be chosen after seeing results.

---

## 4. Axis A — capacity

### 4.1 The scaling rule (one rule, fixed in advance)

Compound scaling anchored on the champion, parameterised by a single width
multiplier **r**:

```
width       = round_to_32( 224 · r )          # 32 keeps GroupNorm(groups=8) legal
n_blocks    = round( 8 · sqrt(r) )            # depth grows as the sqrt of width
head_hidden = round_to_64( 384 · r )
```

Depth is scaled as √r rather than ∝ r deliberately: parameters go as
`n_blocks · width²`, so ∝r depth would make the top point ~5× larger than
intended for the same width, and the round's budget is fixed. This is *a* choice,
not *the* choice — it is registered so it cannot be swapped for a friendlier one
after the fact.

`r ∈ {1/√2, 1, √2, √3}`. Measured parameter counts (built and counted on 238
with the exact champion flag set, `in_channels=58`, `n_globals=13`,
`n_targets=8`, multiscale decoder + prior residual + quantile heads):

| arm | r | width | blocks | head | **params** | ×champion |
|---|---:|---:|---:|---:|---:|---:|
| **C0** | 0.707 | 160 | 7 | 256 | **4,788,419** | 0.46× |
| **C1 — CONTROL** | 1.000 | 224 | 8 | 384 | **10,363,715** | 1.00× |
| **C2** | 1.414 | 320 | 10 | 512 | **24,889,315** | 2.40× |
| **C3** | 1.732 | 384 | 11 | 640 | **38,493,347** | 3.71× |

Total span **8.0×**; **3.7× above the champion**, which is the region no prior run
has entered.

**Deviation from the commissioned ladder, and why.** The brief asked for
1.5M / 6M / 24M. Anchored on the *real* champion (§1) those numbers would place
two of three points *below* the incumbent and re-run an experiment already done
twice (§2.1). The delivered ladder keeps the requested point count (+1) and
comparable span, and spends the extra budget above the champion where the
information is. C2 = 24.9M is within 4% of the requested 24M point.

### 4.2 What is held fixed

Every arm runs the `s1e` command **verbatim** except `--width / --n-blocks /
--head-hidden` and `--out-dir`:

```
python -m lpopt.model.train --ensemble 1 --parallel-members 1 \
  --split S1e --cond-schema v6b --epochs 150 --num-workers 8 --device auto \
  --base-seed 20260716 \
  --width <W> --n-blocks <B> --head-hidden <H> \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior \
  --quantile-heads --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
  --out-dir runs/scale_C<k>
```

Same split, same featurization (`v6b`), same loss set, same augmentation, same
150-epoch schedule with patience 15, same seed **20260716**, **one seed each**.

**The control C1 is a fresh run, not the stored `s1e` member.** `s1e` trained its
five members jointly (`--parallel-members 5`); a 1-member run is a different
protocol. C1 re-runs seed 20260716 under the **sweep's** protocol
(`--ensemble 1 --parallel-members 1`) so the capacity comparison is
**1-seed vs 1-seed under one protocol**. The stored 5-member `s1e` ensemble is
reported alongside as context and **is never the capacity baseline** — comparing a
1-seed arm to a 5-member ensemble would confound Axis A with Axis B.

### 4.3 Learning rate — registered, with the confound named

**LR is NOT changed across the ladder.** All arms inherit the champion schedule:
base `lr=3.0e-4`, `lr_final=3.0e-5`, `lr_scaling=True` → effective `1.2e-3` at
`effective_batch=1024`, `warmup_epochs_base=20` → 80 effective.

Rationale: (i) it is what "same schedule" means, (ii) it is what the prior A5 arm
did, so this round stays comparable to it, (iii) the repo has no µP
implementation and inventing a width-LR rule here would add an untested factor to
a single-factor experiment.

**This creates a real confound and it is named in advance**: a wider net under an
unchanged LR can underperform for *optimization* reasons that look identical to
*capacity saturation*. The disambiguator is registered in §7.3 and is a **train-side**
quantity, so it cannot be reverse-engineered from the val result.

---

## 5. Axis B — ensemble size

Train **5 more members at the champion capacity**, seeds
**20260721, 20260722, 20260723, 20260724, 20260725**, using the `s1e` command
verbatim (`--ensemble 5 --parallel-members 5`) with only `--base-seed 20260721`
and `--out-dir runs/scale_E5b` changed. Compare:

* **E5** — the existing `s1e` members (seeds …716–…720),
* **E5′** — the new members (seeds …721–…725), a *replication* check,
* **E10** — all ten.

### 5.1 How the three ensembles are composed — no calibration confound

Ensembles are composed from **raw member forward passes**, not from the two
runs' fitted calibration artifacts. `s1e` and `scale_E5b` each fit their own
`calibration.json` / `cell_calibration.json` on their own 5 members; using them
would mean E5 and E10 differ by *both* member count *and* a refit post-hoc affine
map, and the round could not attribute the difference.

Procedure (single forward pass, then arithmetic):

1. `PosValCnnBackend._raw_forward` on each run → `mu_z[M,N,T]`, `log_sigma[M,N,T]`.
2. De-standardise **per run with that run's own** `tmean`/`tstd`
   (`members_raw = mu_z · tstd + tmean`) — so a constants mismatch cannot corrupt
   the concatenation.
3. Concatenate along the member axis to get the 10-member raw stack.
4. For any subset S: `mean = members_raw[S].mean(0)`,
   `epistemic = members_raw[S].std(0)`,
   `sigma_total = sqrt( mean_m(alea_m²) + epistemic² )` where
   `alea = exp(log_sigma) · tstd`.

Step 4 is `model_api.PosValCnnBackend._predict_raw_triplet` reproduced exactly
(that method's `total_t = sqrt((alea_raw**2).mean(axis=0) + epistemic_t**2)`), so
the composed σ is the σ the serving path forms.

**Registered pre-check (must pass or the axis is reported as void):** the two
runs' `target_zscore` `tmean`/`tstd` and `target_names` agree to within 1e-9 and
in the same order. If they do not, step 2 already protects the means, and the
fact is reported rather than papered over.

**Calibrated σ is reported as context for E5 only** (the champion as actually
served). It is not used for any E5-vs-E10 comparison.

---

## 6. Registered metrics — decided now, computed later

All on the S1e val fold, converged rows, identical rows and order for every arm.
Targets: **F_r, node_peak, map_cov, cyclen**. Map targets are scored on the 2,371
rows that carry a map label.

### 6.1 (a) Accuracy

**M1 — iso-group (within-cell) Spearman.** Group = `lpopt.model.folds.cell_key`.
Cells with **< 8 rows are dropped** (`MIN_CELL_ROWS = 8`, inherited from
`split_secondary_readout.py`). Spearman is computed per cell; the headline is the
**median across cells**; mean and the full per-cell vector are published.

**M2 — P@10%.** Per cell, let `k = max(4, round(0.10·n_cell))`. Take the model's
best `k` rows by predicted value (lowest for F_r / node_peak / map_cov, highest
for cyclen) and the truly best `k` by label. `P@10% = |intersection| / k`.
Headline = **median across cells** (same ≥8-row filter). Ties in the prediction
are broken by `record_id` ascending — fixed now so it is not a free parameter.

**M3 — `T6_T4` in-cell ρ, reported separately.** 144 val rows, 5 cells at ≥8.
Within-cell Spearman on `f_r`, median across those cells, plus F_r MAE. This is
the quantity rounds 7–9 reported (`s1d` 0.834 → `s1e` 0.788), carried so the
series stays comparable. **It is a small-n number and decides nothing here.**

### 6.2 (b) Search-relevant

**M4 — σ calibration.** For each row, `z = (mean − y) / sigma_total`. For nominal
two-sided Gaussian levels `α ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99}`,
empirical coverage is `mean( |z| ≤ Φ⁻¹((1+α)/2) )`. The registered scalar is

```
ECE_σ  =  mean over α of | empirical(α) − α |          (lower is better)
```

Reported per target for **F_r and cyclen**, with the full coverage curve. Also
reported: the sign of the miscalibration (over- vs under-confident), because a
"better ECE" reached by inflating σ is not the same win as one reached by
sharpening the mean.

**M5 — precision@32 on the acquisition's own key.**

*Terminology correction, registered:* the brief says "top-F_r-**LCB**". The
campaign minimises F_r and therefore consumes F_r at its **UCB** — the
conservative side (`acquisition.py`: `fr_ucb = mean[:,0] + kappa·std[:,0]`, and
`MinFrSpec.risk_z = 0.25`). This round uses the real key:

```
key = mu_F_r + 0.25 · sigma_F_r          # lower is better
```

Surface: the **26 cells with ≥ 64 converged val rows** (a top-32 needs a pool at
least twice that). Per cell, select the 32 lowest-key rows; `precision@32` is
`|selected ∩ (32 lowest true F_r)| / 32`. Headline = **mean and median across the
26 cells**.

**M6 — selection regret** (reported with M5, because precision alone can hide the
operationally relevant fact): per cell,
`regret = min(true F_r among the 32 selected) − min(true F_r in the cell)`, in F_r
units. Headline = median across the 26 cells. This is what the campaign actually
loses by trusting the model.

### 6.3 Uncertainty on the metric differences

Each headline difference is accompanied by a **paired cell bootstrap** — the
repo's own `lpopt.model.ab_paired.paired_cell_bootstrap`, at its standing
defaults `reps=2000, seed=0, alpha=0.05, method="bca", aggregate="median"`, on
the per-cell dicts produced by `lpopt.model.flat_metrics.cell_rho` and friends.
Defaults rather than fresh choices, so the intervals are comparable to every
prior A/B round.

BCa is legitimately available here: `ab_paired.MIN_CELLS_BCA = 6`, and both
surfaces clear it (50 cells for M1/M2, 26 for M5/M6). **If any surface ends up
under 6 cells after the ≥8-row filter, that metric's interval degrades to
percentile and is announced as such** in the results table — the disclosure
convention of `ab2_addendum_S1E_20260815.md` §5.2.

**1 seed per capacity arm means seed noise is NOT separable from the effect** —
see §8.

---

## 7. Decision rules and falsification — fixed before the first epoch

### 7.1 Axis A gate ("scaling helps")

> **PASS** if a capacity arm above the control shows **≥ +0.02 median iso-ρ over
> C1 on ≥ 2 of the 4 targets**, with the paired-cell-bootstrap interval on each
> such gain excluding 0.

### 7.2 Axis B gate ("more members help")

> **PASS** if E10 beats E5 on **both**: (i) a **lower ECE_σ** on F_r *and* cyclen,
> and (ii) a **higher mean precision@32**, with the paired interval on the
> precision@32 difference excluding 0.

E5′ vs E5 is the **noise yardstick**: two disjoint 5-member ensembles of the same
recipe should be equivalent. Whatever E5′−E5 measures is the floor below which an
E10−E5 difference means nothing, and it is reported next to it.

### 7.3 The registered disambiguator for a flat Axis A

A null on Axis A has two possible causes and they must not be conflated:

| observation | reading |
|---|---|
| C2/C3 reach a **lower final train loss** than C1 but no val gain | **genuine capacity saturation.** More parameters fit the training set better and buy nothing that generalises. The registered conclusion in §7.4 fires. |
| C2/C3 reach a **higher (worse) train loss** than C1 | **optimization failure, not capacity.** The unchanged LR (§4.3) did not suit the width. The round is reported as **INCONCLUSIVE on Axis A above the champion**, and the honest next step is a µP-style LR rule — *not* a claim that capacity is exhausted. |

Final train loss and `best_epoch` are recorded from each run's `train.log` /
`meta.json` for exactly this purpose.

### 7.4 Falsification readings, both directions

> **If Axis A is FLAT** (registered prior in §2.3 holds, and §7.3 attributes it to
> saturation): capacity above 10.4M is **not** the binding constraint. Combined
> with §2.1 (structure returned 44× more per parameter) and §2.2 (labels are not
> the ceiling, density has a −0.2…−0.4 exponent), the surviving levers are
> **structure** and **label information density per MASTER call** — i.e. the
> `precision_ceiling_roadmap_20260725.md` §4 harvest (EDIT 6 axial + all 30 EDIT 5
> burnup steps + map regime coverage). The result **strengthens the
> label-density/-resolution program** and closes pure width as a lever.
>
> **If Axis A PASSES**: the champion is under-parameterised and §2.3's prior is
> wrong on the record. The follow-up is a properly registered capacity arm at
> ≥ 3 seeds through `gate-promote` — **not** a promotion from this round.
>
> **If Axis B is FLAT**: the 5-member ensemble already saturates the epistemic
> term; σ is dominated by the aleatoric head and the acquisition's κσ shift will
> not sharpen by adding members. Ensemble width is closed as a lever and the
> honest σ improvement path is calibration/conformal, not more members.
>
> **If Axis B PASSES**: 10 members is a cheap, structure-free win on the
> acquisition's own key, and the cost (2× inference) is the only thing to weigh.

### 7.5 What a win does NOT authorise

No promotion, no champion change, no campaign re-pointing, no edit to
`lpopt/model/net.py` defaults. Deliverable is evidence.

---

## 8. What this round cannot conclude

1. **One seed per capacity arm.** Between-seed spread is not measured on Axis A.
   From the `s1e` log, five seeds of the *same* architecture ended at `spF_r`
   0.886–0.905 — a **~0.019 spread**, which is the same order as the §7.1
   threshold. **Any single-target gain of 0.02–0.03 on Axis A must be read as
   "within seed noise"** and is registered as such now, before it can be
   presented as a win. The gate deliberately requires **≥ 2 targets** for this
   reason, and it is still weak evidence.
2. **The ladder is width-primary.** A flat result falsifies *this* compound
   scaling rule, not every conceivable use of more parameters (deeper-but-narrow,
   mixture-of-experts, higher map-head resolution are untested).
3. **`node_peak` / `map_cov` ride on 20% of the val fold**, 97.9% of the map
   corpus is dataset A (`precision_ceiling_roadmap_20260725.md` §3.5). A map-target
   result is a statement about that regime, not about the core.
4. **No new labels.** The store is frozen (§3), so nothing here can speak to the
   density lever except by elimination.
5. **`T6_T4` (M3) is n=144 across 5 cells.** Descriptive only.

---

## 9. Execution

**GPU occupancy checked before writing this section.** `runs/policy_v1` is live on
238 and pinned to **`CUDA_VISIBLE_DEVICES=1`** (its `run.sh`, heartbeat fresh at
2026-08-15 15:41 KST, in its feature-cache build phase). **This round runs
entirely on `CUDA_VISIBLE_DEVICES=0`** and does not touch GPU 1, `runs/policy_v1`,
199, or any local MASTER regeneration.

Sequential, one run at a time, tmux + 15 s heartbeat + `rc` + `DONE`/`FAILED`
markers — the `s1e`/`policy_v1` launcher pattern verbatim.

| order | run | out-dir | params | projected wall |
|---:|---|---|---:|---:|
| 1 | C1 control | `runs/scale_C1` | 10.36M | ~50 min |
| 2 | C0 | `runs/scale_C0` | 4.79M | ~40 min |
| 3 | C2 | `runs/scale_C2` | 24.89M | ~1.5 h |
| 4 | C3 | `runs/scale_C3` | 38.49M | ~2.2 h |
| 5 | E5′ (5 members) | `runs/scale_E5b` | 5 × 10.36M | ~3.1 h |

Projection basis (`s1e/train.log`): featurization **1,112.8 s** (paid per run —
there is no on-disk encode cache) and a 5-member joint chunk in **9,980.9 s** at
10.36M. **~8.3 GPU-hours total, 5 training runs.** Budget cap: **if the total
exceeds 14 GPU-hours, C3 is dropped** and the round reports a 3-point ladder —
registered now so the cut is not a judgement call made while looking at results.

Scoring uses a scratch serving script (`eval_accuracy_sigma.py`, on 238 alongside
the existing `eval_accuracy_split.py`) that emits per-row `mean`, `epistemic_std`
and `sigma_total` in addition to the existing 7-column layout. **No file in
`lpopt/` is modified in this round.**

Checkpoints and per-run artifacts are pulled to
`data/models/scaling_20260815/`.

---

## 10. Reporting

`data/reports/scaling_results_20260815.md`: one table carrying **every** metric in
§6 for **every** arm against C1/E5, the §7 verdict per axis, the §7.3 train-loss
disambiguation, the §6.3 intervals, and an honest reading against §2.3's
registered prior. Metrics registered in §6 are reported **whether or not they
favour scaling**, and the §2.3 prior is quoted verbatim next to its outcome.
