# A/B round 4 — ADDENDUM: arm BU (burnup placement)

**Written 2026-08-10, BEFORE either arm has been trained and before
`paired_cell_bootstrap` has been called on this arm even once.** Addendum to
`data/reports/ab2_preregistration_20260730.md`, inheriting its estimator, cell
unit, frozen surface and non-regression rails unchanged, and following the
discipline of `data/reports/ab2_addendum_R3_20260801.md`. Only what is stated
here is new. Anything added after the intervals exist is post-hoc and must be
labelled.

**Order of work, auditable:** this document was written and saved before a line
of the arm's featurization was edited and before either arm was launched. The
only facts known at the time of writing are the structural ones in §3 — file
hashes, row counts, split composition, the library/feed composition of the
frozen surface, and the champion's own `meta.json` — none of which involve a
prediction from either arm. The verdict lands in
`data/reports/ab2_verdict_BU_20260810.md`.

**Provenance of the mechanism:** `data/reports/kcurve_fusion_memo_20260809.md`
§3(0) (the measurement), §4 **S1** (the risk it answers) and §5 **STEP 0b** (the
pre-registered arm). This addendum is STEP 0b, executed.

---

## 1. Why this arm exists

The AB2 programme has now falsified three of the four mechanisms
pre-registration §1 named:

| mechanism (§1) | status |
|---|---|
| loss / label engineering | **falsified** — round 1 (A1, A2, A3), `ab2_verdict_20260731.md` |
| ensemble / variance reduction | **falsified** — E10, `ab2_verdict_E10_20260731.md` |
| more labels in the deficient cells | **REJECT on the transfer reading** — R3, `ab2_verdict_R3_20260801.md` |
| verify-many-then-select | untested (and is not a model change) |

Arm **BU** is not a fifth member of that list. It is a different claim: that the
model's *inputs* have been systematically wrong, in a way nobody measured until
2026-08-09, and that the error is larger than anything the three falsified
mechanisms were reaching for.

`NOMINAL_CYCLE_BURNUP_MWD_KG = 22.0` (`lpopt/model/featurize.py:46`) is the
a-priori per-cycle burnup every burned slot's physics is evaluated at. Measured
implied constants over 12,174 trajectory cores (memo §3(0)):

| library | f101 | f121 | f141 |
|---|---:|---:|---:|
| ga80 | 24.94 | **28.69** | 31.14 |
| paramA | 30.47 | 35.33 | **36.77** |

The constant is wrong in every one of the six regimes, it is wrong
*systematically* in `(library_id, feed)` — and both of those are already model
inputs, so the error is a pure encoding defect, not missing information.

What it costs, as measured (memo §3(0)):

| quantity | value |
|---|---:|
| burned-slot placement error, mean \|ΔBU\| | **12.18 GWd/tU** (core sd 9.27, worst 21.7) |
| core-mean \|Δρ\| equivalent | **2801 pcm** (3930 pcm on the `power_prior` mid-cycle convention) |
| for comparison: 4-scalar k-curve compression error | 1402 pcm |
| ratio | **burnup placement is 5.1× the curve-precision term** |
| same-instrument S/N: \|Δnode_peak\| from the burn-state error | 0.1435 |
| same-instrument S/N: \|Δnode_peak\| from a random fuel swap | 0.1545 (ratio 0.93; ga80 1.04) |
| cores where the encoding error exceeds a whole fuel-swap effect | **45.8%** |

And it is fixable **a priori**. On an equilibrium core the previous-cycle slot is
a pure function of the pattern, so the shuffle source's own leakage-safe
quantities predict the true slot BOC burnup at out-of-fold \|err\| **2.011
GWd/tU, R² 0.952** (103,249 slots / 3,000 cores) — a 6× reduction, 1207 pcm. The
leakage-clean variant (every `cyclen`-derived column removed) gives 2.045 /
0.9505. Current-slot features alone give 7.787: **the physics lives in the
SOURCE slot**, which is why no previous feature round found it.

Registered as the reason this arm precedes the curve-precision half of v7:
*evaluating a k-inf curve 12 GWd/tU away from where the assembly actually sits
and then making the curve more precise produces a more precise wrong answer.*

---

## 2. The two arms

Both arms are trained **from scratch**, on the **same store snapshot**, with the
**same split file**, at the **same hyperparameters and the same five seeds**.

| arm | definition |
|---|---|
| **A0-BU** (CONTROL) | the champion recipe, `cond_schema = "v6"`, featurization **unchanged**. |
| **T-BU** (TREATED) | the champion recipe, `cond_schema = "v6b"` — regime `(library_id, feed)` nominal-burnup table **plus** the source-chain burnup channels. Nothing else. |

The control is **not** R3's A0, not round 1's A0 and not the champion. Round 3
§1.2 recorded that the round-1-era store and split are gone; this round refuses
to lean on any arm trained against a different snapshot. A0-BU is trained
contemporaneously, against the §3 hashes, for exactly that reason.

### 2.1 What `v6b` changes, exhaustively

Registered before the code was written, so the diff can be audited against it:

1. **Regime nominal-burnup table.** A hard-coded `(library_id, feed) → MWd/kgHM`
   lookup carrying the six measured constants above. It is **constants in
   source**, computed from no label at read time, so it is a-priori by
   construction. Fallback chain, in order: exact `(library, feed)` hit → linear
   interpolation in `feed` within that library → that library's mean (feeds
   outside the measured hull only) → the legacy `22.0` (library not in the
   table). It feeds (a) the `nominal_burnup` cell channel and (b) the burnup at
   which `power_prior.kinf_quarter` evaluates each slot's reference k-inf curve
   — which is what builds channels 48/50/51 (`origin_kinf_contrast`,
   `prior_power`, `prior_power_contrast`), i.e. the plane the map head's
   residual skip reads.
2. **Source-chain channels**, appended after the existing 52 (append-only,
   indices 0–51 and the 13 globals byte-identical to v6) — **six** of them, at
   indices 52–57, giving `in_channels = 58`. For each slot: the a-priori prior
   power of its shuffle source `direct[s]` (`src1_prior_power`) and of its
   second-order source `direct[direct[s]]` (`src2_prior_power`), the
   second-order source's radius (`src2_radius`; the *first* source's radius is
   already channel 16, `chain_source_radius`), the power-weighted
   chain-integrated nominal burnup (`chain_bu_integral`), and two presence gates
   (`src1_present`, `src2_present`).
3. **Nothing else.** No global is added. No head, loss, decoder, acquisition,
   calibration or serve-path change. Every training flag is A0's.

### 2.2 What `v6b` deliberately does NOT change — registered as descope

* **`physics_prior.py` (the scalar `cyclen` prior) keeps `22.0`.** It is a
  *fitted-residual* path whose `(alpha, beta)` refit absorbs a constant shift,
  and touching it would (i) make this two changes, and (ii) put the arm's own
  `cyclen` harm bound at risk through a second, unrelated mechanism. The memo's
  2801 pcm is measured on the `power_prior` mid-cycle convention, which is the
  path this arm does change.
* **`fuel_types.parquet`, `records.parquet`, `maps.npz`, `data/splits/` are not
  touched.** No migration, no backfill, no re-harvest.
* **The v7 k-curve / ADF / fusion work (memo §3(1)–(6)) is not in this arm.**
  STEP 0 already falsified P1 as a sufficient statistic and the memo's stopping
  condition forbids spending GPU on curve precision alone. This arm is burnup
  placement only, exactly as STEP 0b registers it.
* **`g_sym_class` serve/train inversion (memo §3(6)) is not fixed here.** It is a
  real defect of unmeasured size; bundling it would destroy the single-change
  property.

### 2.3 The one-change property, and the honest caveat about it

`v6b` bundles the regime table and the source-chain channels into ONE arm. They
are **not** separable by this experiment. That is deliberate and is exactly how
STEP 0b was pre-registered ("레짐별 nominal 상수 + 소스체인 연소도 채널만"), and
it is physically coherent — the chain-integrated burnup channel is *defined*
using the regime constant, so there is no version of "source chain only" that
does not also carry a per-cycle burnup scale. But it means a positive result
attributes to "burnup placement", not to either half.

**Registered consequence:** a PASS may not be reported as "the regime table
works" or "the source-chain channels work". It may only be reported as "correct
a-priori burnup placement is a lever". Decomposition needs its own round.

---

## 3. Frozen inputs — verified BEFORE the rule was fixed

### 3.1 Store snapshot (the canonical store; both arms, byte-identical)

| item | value |
|---|---|
| `data/store/records.parquet` sha256 | `4039bc96cffb52fb3f8c371aa94127fb17e70765775b452c6b29ae6068eeed3f` |
| bytes | 21,003,452 |
| rows | **70,984** (converged 62,864) |
| `data/store/maps.npz` sha256 | `51caf2dae528284209d39f701a020e5b4fe1a624eeafa66fbe1502f2b99af72b` |
| bytes | 183,175,993 |
| `data/store/fuel_types.parquet` sha256 | `4ee9b16e4f595525c15168ea477fd92b6f39bb110147bf1b12c336f49c5c8ecd` |
| bytes | 61,296 |

**Snapshot note (2026-08-10, before either arm was pushed).** This document was
first written against the 70,795-row snapshot
(`86cd1895…19f5262`). The coordinator then landed 189 MASTER-verified rows
(campaigns `frtransfer_E3E4` / `frtransfer_J1J2`, ga80 `E1_E2`-derived
fixed-pattern fuel-transfer cores, 38 of them at `F_r <= 1.50`). Because
**nothing had been pushed and neither arm had started**, the pin was moved
forward to the 70,984-row snapshot above and *both* arms are pushed from it. The
hashes above are re-verified **on the remote after transfer** (§7.1), so the pin
records what the GPU box actually holds, not what this box held at write time.
No arm ever sees a different snapshot from the other — that is the invariant, and
§3.3 shows why this particular growth cannot reach either arm regardless.

### 3.2 Split — the SAME file for both arms

| item | value |
|---|---|
| file | `data/splits/S1.json` |
| sha256 | `6ab35f25027c8355fe4f07bda2a200b1c5baf40d445257e255ed89dead650640` |
| bytes | 5,345,078 |
| kind / seed / name | `curriculum_group` / 0 / `S1` |
| train / val ids | **58,341 / 11,401** (overlap **0**) |
| frozen holdout (`groups.curriculum_val_by_cell`) | **36** cells, **3,207** rows |
| `protect_feed` | 121 |
| held-out groups | 8 |

This is the **same `S1.json` R3 was judged on** (identical sha256, §3.4 of the R3
addendum records `6ab35f25…ead650640`). The frozen 36-cell surface is therefore
the same surface rounds 1, E10 and R3 were scored on.

### 3.3 The store has grown since the split — and it cannot reach either arm

| quantity | value |
|---|---:|
| store rows now | 70,984 |
| rows named by `S1.json` (`train_ids ∪ val_ids`) | 69,742 |
| **rows in the store but in NEITHER partition** | **1,242** |

Because `S1.json` enumerates ids explicitly, both arms train on exactly the
58,341 `train_ids` and validate on exactly the 11,401 `val_ids`, regardless of
the store having grown. **The 1,242 newer rows — including the 189 that landed
mid-write (§3.1) — enter neither arm.** This is registered now, before the
result, for three reasons: it makes the training corpus identical to R3's; it
means this round is NOT confounded with a store change (unlike R3, whose
confound §2 had to disclose); and it means an `arm − control` difference here is
the featurization change and nothing else.

**Consequence for the snapshot question:** the 70,795 → 70,984 re-pin is
*inert for training*. It is recorded for provenance (both arms are pushed from
one snapshot and the remote hash proves it), not because it changes what either
arm learns. Should a later round want those 189 low-`F_r` rows in training, it
needs a rebuilt split — which would be a second change and a different arm.

### 3.4 The decision surface — and why the mechanism can act on it directly

The single most important structural fact of this round, verified before the
rule was fixed:

| property | value |
|---|---|
| frozen rows | 3,207 / 3,207 present in the store |
| libraries | **paramA 1,721 · ga80 1,486 — and nothing else** |
| dataset | 100% **P** |
| feeds | 101 (516) · 109 (687) · 117 (489) · 125 (619) · 133 (510) · 141 (386) |

**Both libraries the regime table measures are the only two libraries on the
scoring surface.** Feeds 101 and 141 are measured points of the table; 109, 117,
125 and 133 are *interpolated* between the measured 101/121/141 anchors. `f121`
never appears (`protect_feed = 121` holds every f121 row in train).

This is the exact defect R3 could not escape, inverted. R3's addendum §2.1 had
to register that "not one enriched row is in the scoring surface"; here the
mechanism acts on **every scored row**, in both libraries, with two of the six
feeds at measured anchors. **A null on this surface is therefore informative
about the mechanism, not merely about transfer** — and §5.6 fixes that reading
in advance.

### 3.5 Where the regime table is inert — disclosed before the result

| library | store rows | regime table entry | constant used by T-BU |
|---|---:|---|---|
| ga80 | 18,155 | measured (3 feeds) | 24.94 … 31.14, interpolated |
| paramA | 13,975 | measured (3 feeds) | 30.47 … 36.77, interpolated |
| 260624 | 29,976 | **none** | **22.0 (unchanged)** |
| 5.8_5.1 | 8,244 | **none** | **22.0 (unchanged)** |
| legacy_a | 634 | **none** | **22.0 (unchanged)** |

**On 38,854 of 70,984 store rows (54.7%) the regime half of T-BU is a no-op**;
only the source-chain channels differ there. That is a faithful implementation of
"fall back to 22.0 for an unknown library" rather than an extrapolation nobody
measured — inventing a constant for 260624 from a ga80/paramA fit would be
exactly the kind of unregistered inference this programme exists to refuse. The
consequence is registered: T-BU's training signal is *diluted* relative to what
the memo's 12,174-core measurement implies, because 54.9% of the corpus carries
the old encoding. The **decision surface**, however, is 100% covered (§3.4).

### 3.6 Champion recipe — verified against `20260729_054749/member_20260716/meta.json`

Read from disk, not assumed:

| item | value |
|---|---|
| `cond_schema` | `v6` |
| `net_config` | `width 224`, `n_blocks 8`, `head_hidden 384`, `in_channels 52`, `n_globals 13`, `map_head_mode "multiscale"`, `map_prior_channel 50`, `film_every 2`, `groups 8`, `n_map_channels 4`, `n_targets 8`, `n_quantiles 3`, `n_quantile_targets 2` |
| member dirs / seeds | `member_20260716 … member_20260720` → **20260716, 20260717, 20260718, 20260719, 20260720** |
| loss / recipe flags | `epochs 150`, `patience 15`, `lr 3e-4 → 3e-5`, `warmup 20`, `map_prior_residual True`, `map_spectral_weight 0.3`, `map_peak_weight 2.0`, `map_lambda 0.3`, `cyclen_physics_prior True`, `cyclen_rank_weight 0.1`, `quantile_heads True`, `quantile_weight 0.2`, `promote_max_asm_bu True`, `distill_targets data/models/_v5_distill_soft.npz`, `distill_weight 0.4`, `distill_min_match_frac 0.5`, `f_r_rank_weight 0.1`, `map_fr_consistency_weight 0.0`, `map_peak_topk_weight` absent (0.0), `traj_weight` absent (0.0), `cbc_provenance_offset` absent (false), `augment True`, `censor_dataset_a_pin_labels` default (true), `torch_compile False` |

**Both arms are pinned to exactly this**, differing only in `--cond-schema`
(`v6` vs `v6b`). The launch commands are transcribed verbatim in §7.

### 3.7 Power of the surface — the number the gate is built on

From round 1's own bootstrap on this same 36-cell surface
(`ab2_verdict_20260731.md` §2, arm A3 vs A0 on `T_cell_mae_node_peak`):

| item | value |
|---|---:|
| A0 level, `T_cell_mae_node_peak` | 0.10957 |
| paired bootstrap SE | 0.00146 |
| **MDE₈₀ = 2.80 · SE** | **0.00409** |

That is where the memo's 0.00409 comes from, and it is the bar §5.3 fixes.

### 3.8 Encoding quality — measured after implementation, before either arm ran

**Added 2026-08-10 after the featurization was written and before either arm was
launched. It measures the INPUT ENCODING, not an arm's predictions, and it does
not touch the §5 rule — which was fixed from round 1's bootstrap and is
unchanged.** It is recorded because the memo's central claim (12.18 GWd/tU of
placement error) had never been reproduced in this repo, and shipping 5 GPU-hours
on an unreproduced number would have been negligent.

Each encoding is compared against the **true slot BOC burnup** — `maps.npz`
`<key>__traj` plane 1 at step 0, the inherited burnup an assembly carries into
the cycle — over **13,619 burned slots / 400 trajectory cores** sampled at
`random_state=0`, with the champion's fitted `(M² = 150, extrap = 2.0)`:

| encoding | mean \|err\| | median \|err\| |
|---|---:|---:|
| `(age−1) · 22.0` — **what v6 uses today** | **12.075** | 11.844 |
| `(age−1) · B_regime` — T-BU's `nominal_burnup` | **7.874** | 7.018 |
| `B_regime · Σₖ P(srcₖ)` — T-BU's `chain_bu_integral` | 8.188 | **5.309** |

| by library | n | today | regime | chain |
|---|---:|---:|---:|---:|
| ga80 | 6,981 | 9.769 | 7.475 | **6.163** |
| paramA | 6,638 | 14.500 | **8.294** | 10.317 |

Three things this settles in advance:

1. **The memo's measurement reproduces.** 12.075 GWd/tU here vs 12.18 in memo
   §3(0), on an independent sample. The defect is real and is the size claimed.
2. **The regime constant alone removes 35% of the mean error**, in both
   libraries, with no tail risk. That half of the arm does not depend on the
   prior at all.
3. **The power weighting halves the median (11.84 → 5.31) but carries a heavier
   tail**, and is worse than regime-only in the mean for paramA. The cause is
   known and disclosed: the leading-order diffusion prior is fitted for *rank*
   (within-cell ρ 0.754) and its amplitude is inflated — at the fitted M² the
   slot power spans 0.13–3.14 where a real assembly spans ~0.4–1.4.
   **Registered design response:** BOTH estimators are in the channel inventory
   (`nominal_burnup` = flat regime, `chain_bu_integral` = power-weighted) so the
   network blends them rather than being forced onto either. Damping the power
   weight would fix the tail but would introduce a free parameter with no
   pre-registered value, which this arm may not carry.

**Registered limit on how this may be read:** it is a statement about the
*encoding*, and it is measured using labels. It is **not** evidence that the arm
will move `node_peak`, and it may not be cited toward the §5.3 gate. A better
input is a necessary, not a sufficient, condition — §5.6's falsification stands
exactly as written, and this table makes a null *more* informative, not less,
because it removes "the encoding did not actually change" as an explanation.

---

## 4. Unverified premises

Stated so a favourable result carries its assumptions visibly:

| premise | status |
|---|---|
| `parallel_members = 5` is a bit-identity contract, not a recipe change | **assumed** (inherited unverified from R3 §4) |
| the six measured regime constants (memo §3(0)) are correctly transcribed | **assumed on the memo**; they are hard-coded constants and are re-readable from the source |
| the memo's out-of-fold 2.011 GWd/tU / R² 0.952 was measured leakage-clean | **assumed on the memo** (the leakage-clean variant 2.045 / 0.9505 is quoted there) |
| both arms fit all six per-cell calibrations in one pass (pre-registration §7) | **verifiable after the fact** from each arm's `train.log`; checked before scoring |
| the remote venv accepts `cond_schema = "v6b"` | **verified at launch**, not assumed — a rejection surfaces as an immediate `FAILED` marker |

A null result does not depend on any of these. A favourable result does, and §5.6
says so.

---

## 5. The decision rule — fixed before any result exists

### 5.1 Surface, estimator, control — inherited unchanged

Frozen 36-cell holdout, 3,207 rows; **unit = the cell**, never the row; control =
**A0-BU**, never the champion and never a previous round's A0;
`lpopt/model/flat_ab.py` `paired_metric` → `lpopt/model/ab_paired.py`
`paired_cell_bootstrap` with `method="bca"`, `aggregate="median"`, `reps=2000`,
`alpha=0.05`, `seed=0`, sign-flipped so `theta > 0` means "T-BU beats A0-BU";
`flat_metrics.cell_mae` at `min_rows = MIN_CELL_ROWS`; `flat_ab._restrict` to the
frozen cell list. T-BU is reindexed onto A0-BU's `record_id` order before the
arena is built — in this apparatus the row alignment **is** the pairing
(`flat_ab.FlatArena.__post_init__`), and a verdict computed on mis-ordered rows
is silently meaningless.

Degradation rules inherited verbatim: `n_cells < 6` → BCa degrades to percentile,
announced in `notes`; `n_cells < 3` → `method="insufficient"`, which fails every
gain **and** every harm test; a point-mass resample → `method="degenerate"`,
likewise carrying **no** evidence. 36 cells clears both floors.

### 5.2 Target axis — one, named in advance

| axis | metric key | units |
|---|---|---|
| `node_peak` | **`T_cell_mae_node_peak`** ≡ `M7_cell_mae_node_peak` | — |

This arm has **one** primary axis, not three. The mechanism is a *spatial* burn-
state error whose measured propagation is through the diffusion prior into the
node power distribution (memo §3(0): \|Δnode_peak\| 0.1970 → 0.1400 as the burn
state is corrected). `cyclen`, `cbc_max` and `map_cov` are **harm rails only**
(§5.4) — they are not permitted to promote this arm, and a gain on them may not
be substituted for the gate. Registering a single primary axis is also what makes
the §5.3 bar honest: there is no multiplicity to argue about, unlike R3's
one-of-three.

### 5.3 Condition 1 — the gate, transcribed from the memo

> **`T_cell_mae_node_peak` improvement ≥ MDE₈₀ = 0.00409, with the BCa CI
> excluding 0.**

Both clauses, jointly:

1. `point ≥ 0.00409` (the improvement is at least the minimum this surface can
   detect at 80% power), **and**
2. `ci_lo > 0` (`PairedDiff.establishes_gain`).

A point estimate is never enough, and an interval that excludes zero on an effect
below its own MDE₈₀ is reported as **underpowered**, never as a pass. Requiring
the point estimate to clear MDE₈₀ *as well as* the interval to exclude zero is
stricter than every previous AB2 round, and it is deliberate: 0.00409 is 3.7% of
the 0.10957 level, the memo predicts a much larger move, and an arm that cannot
clear its own detection floor has not demonstrated the mechanism.

### 5.4 Condition 2 — harm bounds, transcribed from the memo

> harm bound: cyclen 셀중앙 MAE 악화 ≤ 0.10 EFPD, cbc_max ≤ 1.0 ppm,
> map_cov ≤ 0.002

Evaluated as `harm_upper = −ci_lo < ε` (`PairedDiff.bounds_harm`), on the same
paired bootstrap, same surface, same settings:

| axis | metric key | ε |
|---|---|---:|
| `cyclen` | `T_cell_mae_cyclen` (cell-median MAE) | **0.10 EFPD** |
| `cbc_max` | `T_cell_mae_cbc_max` | **1.0 ppm** |
| `map_cov` | `M7_cell_mae_map_cov` | **0.002** |

**Plus** the inherited `flat_ab.HARM_MARGINS` rails, unchanged and enforced:
`M2_flat_tercile_rho_{node_peak,map_cov}` 0.01; `M3_norm_p_at_8_{node_peak,map_cov}`
0.01; `M5_cell_rho_{f_q,cyclen}` 0.02; `M7_cell_mae_{node_peak,map_cov}` 0.005;
`M7_abs_bias_node_peak` 0.005. `M5_cell_rho_f_r` is scored and reported but
**excluded from the verdict** (F_r deferral, `flat_ab.FR_HARM_METRIC`, user
decision 2026-07-26) — unchanged for this round.

A straddle is **not** equivalence and routes to HOLD, never to PASS. Any of the
five variance axes (`node_peak`, `map_cov`, `f_q`, `cbc_max`, `cyclen`)
**established worse** (`ci_hi < 0`) is a REJECT on its own, inherited from R3
§5.3 clause 2 at full strength.

### 5.5 Verdicts

* **PASS** — §5.3 and §5.4 both hold. Earns a gate attempt against the champion
  (pre-registration §5.7); does not promote by itself.
* **HOLD** — §5.3 unmet with nothing established worse and no harm bound
  exceeded.
* **REJECT** — a harm bound is exceeded, **or** any of the five variance axes is
  established worse.
* **ESCALATE** — the target axis returns `insufficient` or `degenerate`.

### 5.6 Falsification / stop rule — transcribed from the memo, verbatim

> **정지조건**: 게이트 미달이면 §3(0)은 기각이고 §3(1)의 곡선 정밀도만 남는다 —
> 그러나 그 경우 KILLER 1이 그대로 서 있으므로 v7 전체를 재검토한다.

In English, and binding:

> **If the gate fails, memo §3(0) — "fix the burnup placement error first" — is
> REJECTED, and only the curve-precision half of v7 remains. And because
> KILLER 1 still stands (no assembly descriptor moves within-fuel pattern
> discrimination by more than 0.004 of Spearman ρ), the whole of v7 is then
> re-examined rather than continued.**

Two clarifications, registered now so neither can be softened afterwards:

* This is a **strong** falsification, unlike R3's. §3.4 verified that the
  mechanism acts on every scored row of the surface in both of its libraries, so
  a null here cannot be excused as a surface/mechanism mismatch the way R3's
  could. The only reading a null leaves open is the §3.5 dilution (54.9% of the
  training corpus carries the old encoding), and that is a statement about
  *training signal strength*, not about the surface.
* A PASS is attributed to "burnup placement", not to either half of the change
  (§2.3), and not to any claim about k-curves, ADF or fusion.

**Explicitly forbidden by this addendum**, as bigger-versions-of-the-same-knob or
as already-falsified: re-running this arm with more source-chain channels after a
null; adding the k-curve / ADF channels to it mid-round; `--ensemble 20`; a
larger `--traj-weight`; a larger `--map-peak-topk-weight`; and substituting a
gain on a harm rail for the §5.3 gate.

### 5.7 Power, reproduction, reporting

MDE₈₀ per axis from the same bootstrap (`PairedDiff.mde`, `MDE_POWER = 0.80`); an
effect below its own MDE₈₀ is reported **underpowered**, never null
(pre-registration §5.8). **Any established gain, any established-worse axis and
any harm violation is re-run at bootstrap seeds 0–4**, and the verdict states
whether it reproduces at every seed — gains included, because a single-axis gate
makes one fragile interval decisive. All intervals, notes, levels, methods and
MDEs go to `data/reports/ab2_verdict_BU_20260810.json`.

### 5.8 Registered comparability checks — run BEFORE the intervals are read

Transcribed from R3 §3.1/§3.2, which is how this apparatus is meant to be
audited:

1. `record_id` **set** identical across the two arms; count of arm-only ids on
   each side reported.
2. `record_id` **order**: if it differs, T-BU is reindexed onto A0-BU's order
   *before* the arena is built, and the reindex is stated in the verdict.
3. duplicate `record_id` count = 0 on both sides.
4. `true_*` columns bit-identical across arms (they come from the same store).
5. frozen slice: 36 cells, identical names, identical per-cell row counts, 3,207
   rows, all served.
6. both arms' `train.log` report **all six** per-cell calibrations fitted with no
   `PER-CELL CALIBRATION(S) MISSING` banner.
7. A0-BU's `meta.json` has `cond_schema == "v6"`, `in_channels == 52`; T-BU's has
   `cond_schema == "v6b"`, `in_channels == 58`, and **identical** `width`,
   `n_blocks`, `head_hidden`, `map_head_mode`, `map_prior_channel == 50`, seeds
   and every loss flag.
   *(Amended 2026-08-10 at implementation time: the draft of this line said 59.
   The source-chain block landed at **six** channels, not seven — the
   chain-vs-flat burnup RESIDUAL was dropped once it was seen to be exactly
   `chain_bu_integral − nominal_burnup`, i.e. a perfectly collinear input the
   network already has both terms of. A structural count, not a rule; §5.3–§5.6
   are untouched.)*
8. `target_zscore` identical across arms (the change is an input change, not a
   normalisation change).

A failure of 1–4 voids the pairing and routes to ESCALATE, not to a verdict.

---

## 6. Cost

| | A0-BU | T-BU |
|---|---|---|
| members / forward passes per candidate | 5 | 5 (**unchanged**) |
| input channels | 52 | 59 (+13.5% on the stem conv only) |
| serve-path code change | — | none beyond the encoder |
| campaign inference wave | ~20 min | ~20 min (**unchanged**) |
| training | ~2.2–2.6 h | ~2.3–2.7 h |

Encoder cost rises slightly (the source-chain gathers are free; the extra work is
one additional `power_maps_from_kinf` evaluation path already computed for
channel 50). The featurisation stage is CPU-bound and single-threaded and takes
~19 min before GPU utilisation appears; that is normal and is not a hang.

**Serve cost is essentially zero-delta**, which — exactly as R3 §5.3 argued — is
what would make a modest gain worth taking. It does **not** buy this arm a weaker
bar: §5.3 is stricter than R3's, because this arm has one axis and a surface the
mechanism can actually reach.

---

## 7. Exact launch commands (both arms, one flag apart)

Shipped through `python -m lpopt.remote train -- …` against the §3 snapshot on
`USER@HOST_238:8022`, GPU 0, `--parallel-members 5`.

**A0-BU — control**

```
python -m lpopt.model.train \
  --ensemble 5 --split S1 --cond-schema v6 --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --parallel-members 5 --base-seed 20260716 \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

**T-BU — treated** (A0-BU with `--cond-schema v6b`, and nothing else)

```
python -m lpopt.model.train \
  --ensemble 5 --split S1 --cond-schema v6b --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --parallel-members 5 --base-seed 20260716 \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

`--base-seed 20260716` with `--ensemble 5` reproduces the champion's five seeds
20260716…20260720. The two command strings differ in exactly one token.

### 7.1 Launch record — 2026-08-10, GPU 0

| | A0-BU (control) | T-BU (treated) |
|---|---|---|
| remote run id | `runs/bu_A0` | `runs/bu_T` |
| tmux session | `lpopt_bu_A0` | `lpopt_bu_T` |
| GPU | 0 | 0 |
| launched | 00:37:03 | 00:37:15 |

**Snapshot verified ON THE REMOTE after push** (`sha256sum` in `~/lpopt_ws`),
so the pin records what the GPU box holds, not what the workstation held:

| file | remote sha256 | matches §3 pin |
|---|---|:--:|
| `data/store/records.parquet` | `4039bc96…68eeed3f` | ✅ |
| `data/store/maps.npz` | `51caf2da…b99af72b` | ✅ |
| `data/store/fuel_types.parquet` | `4ee9b16e…9c5c8ecd` | ✅ |
| `data/splits/S1.json` | `6ab35f25…ead650640` | ✅ |

**Both arms were pushed once, from one snapshot, and read the same
`~/lpopt_ws/data`.** There is no path by which they can see different data. They
were also launched from **one source tarball** — `--no-install` on both, pushed
once before either — so the round-2 §3.2 "identical source for every launch"
check holds by construction.

**One local-vs-remote source divergence, disclosed.** After the push, one
defensive line was added locally to `regime_cycle_burnup`: the non-finite guard
was widened from `isnan` to `not isfinite` so an infinite `feed` could not raise
`OverflowError` from inside the encoder. **It is unreachable and cannot change a
single encoded bit**: `RecordInputs.coerce` builds `feed` with `int(...)`, so it
is always a finite Python int before the function is ever called, and every feed
in the store is one of 97…141. The remote therefore carries the pre-guard
source, which is behaviourally identical on every row either arm sees, and both
arms carry the *same* pre-guard source. Restarting two arms to ship an inert
guard was judged the wrong trade; it ships with the next push.

Remote schema verification, run in the server venv before launch:
`CHANNELS_BY_SCHEMA` carries `v6b`; `len(v6) == 52`, `len(v6b) == 58`;
`v6b[:52] == v6` (append-only); `regime_cycle_burnup('ga80', 121) == 28.69` and
`regime_cycle_burnup('260624', 121) == 22.0` (fallback);
`schema_uses_regime_burnup` True for v6b / False for v6.

**Confirmed in flight (both arms training, GPU 0):**

| | A0-BU | T-BU |
|---|---|---|
| featurised | train 58,341 / val 11,401 | train 58,341 / val 11,401 |
| power prior (fit on train rows only) | M² 110, extrap 2.0, n_fit 3,998, ρ 0.7545 | M² 150, extrap 1.0, n_fit 3,998, ρ 0.6923 |
| schedule | `effective_batch=1024 lr=1.20e-03 warmup_epochs=80 device_resident=True parallel_members=5 torch_compile=False` | **identical** |
| seeds | 20260716, 20260717, 20260718, 20260719, 20260720 | **identical** |
| VRAM | 30,898 MiB | 31,986 MiB |

Combined **63,136 of 97,887 MiB (64%)**, 34 GB headroom, GPU at 100% — the
concurrency decision below is validated in flight, not just estimated.

**The two power priors differ, and that is the design working.** `(M², extrap)`
is refit per run on train rows only, and T-BU fits it on the **regime-corrected**
k-inf field (`kinf_quarter_batch(..., regime_burnup=True)`), so it lands on
different constants than the control. Had the fit not been threaded, T-BU would
have served channel 50 on a field it never fit — the exact mismatch
`ab_score.py`'s docstring warns about. **Registered observation, not a result:**
T-BU's prior has a *lower* within-cell CoV rank correlation (0.6923 vs 0.7545).
That is a property of the leading-order prior under the corrected burn state, on
an objective (`map_cov` rank) that is not this arm's target axis; the network
learns the residual against it. It is recorded now, before any prediction, so it
cannot be produced later as a post-hoc explanation of either outcome.

**Why both arms run concurrently on GPU 0** (rather than sequentially): 5
parallel members measure ~36 GB, so two arms are ~72 GB of the card's 97.9 GB —
26 GB of headroom. The decisive reason is the ~19-minute CPU featurisation
stage, which is single-threaded and uses no GPU at all: run sequentially those
two stages serialise, run together they overlap. Concurrency also removes a
class of confound outright — both arms see one machine state, one source tree,
one store, with nothing able to change between them. If VRAM approaches the card
limit the treated arm is killed and re-run sequentially; the arms are
independent, so that costs wall time and nothing else.

---

## 9. Scoring — the command that executes §5, ready before the arms finish

Same two-stage shape rounds 1–3 used: the GPU box serves, this box does the
statistics. `eval_accuracy.py` (already on the box at `~/lpopt_ws/`) is the
producer; it keeps `record_id ∈ val_ids ∧ converged`, which is the 11,338-of-
11,401 served set every previous round was judged on.

**Stage 1 — serve both arms (GPU box):**

```
ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy.py runs/bu_A0 bu_A0 && \
    ./venv/bin/python eval_accuracy.py runs/bu_T  bu_T'
```

**Stage 2 — pull the served rows into `5_RL/runs/ab2_bu/`:**

```
cd 5_RL/runs/ab2_bu
scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_bu_A0.csv .
scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_bu_T.csv  .
```

**Stage 3 — the verdict (runs from any cwd):**

```
python 5_RL/ab2_bu_verdict.py
```

The scorer lives at **`5_RL/ab2_bu_verdict.py`**, in the repo, not in a session
scratchpad. Rounds 1–3 kept theirs in a scratchpad with the repo path baked in;
all three were wiped in a cleanup and had to be recovered from chat transcripts
(`ab2_verdict.py`, `ab2_e10.py`, `ab2_r3.py`). This one resolves the repo from
its own location and survives.

It re-verifies the §3 hashes, runs the §5.8 comparability checks, reindexes T-BU
onto A0-BU's `record_id` order, restricts to the 36 frozen cells read from
`S1.json → groups.curriculum_val_by_cell`, and applies §5.3/§5.4/§5.5 through
`flat_ab.paired_metric` → `ab_paired.paired_cell_bootstrap` (bca / median /
2000 / 0.05 / seed 0) with seeds 0–4 reproduction on every decisive interval.
Output: `data/reports/ab2_verdict_BU_20260810.json`.

**The scorer was smoke-tested end-to-end on synthetic arms before the real ones
finished** — 36 cells / 3,207 rows, `method="bca"`, both gate clauses, every
harm rail, verdict rendered — so scoring cannot fail for a mechanical reason on
the night the training lands.

---

## 8. Test inventory required before launch

Registered as a precondition, not as a report:

| check | requirement |
|---|---|
| `tests/test_leakage.py` | **all green**, including a v6b-parameterised byte-identity check: a labelled row and the same row with every label/metric/map column dropped must encode bit-for-bit identically under `cond_schema="v6b"`. |
| v6 byte-identity regression | a v6 encoder's `cells` and `globals` must be **bit-identical** before and after this change, on real store rows. If v6 moved, every paired difference measured against A0-BU is measuring the refactor (round 2 §9 made this same argument for the `_map_quarter` code motion). |
| append-only | `CHANNELS_BY_SCHEMA["v6b"][:52] == CHANNELS_BY_SCHEMA["v6"]`, and the v6b global list is identical to v6's. |
| `tests/test_featurize.py`, `tests/test_hires_bundle.py`, `tests/test_config.py` | green. |
| regime table | exact-hit, in-hull interpolation, out-of-hull library mean, and unknown-library `22.0` fallback each covered. |
