# Honest-gate no-regression: noise vs. real-regression analysis

**Candidate:** `data/models/20260718_154600` (cond_schema **v4**, trained on the new
`make_curriculum_split`, cells pinned 80/20 to train).
**Champion (previous):** `data/models/20260718_075428` (cond_schema **v3**).
**Gate result under review:** `new_cell` mean Sp 0.7498 **PASS**; `no_regression`
**FAIL**, worst_drop 0.1468, ε = 0.05.

The gate (`curriculum._gate_no_regression`) scores **both** models LIVE on the
**same** per-cell curriculum holdout (30 converged rows/cell), computes
`drop = old_spearman - new_spearman` per (cell, target), and fails if
`max(drop) > ε`. The 6 checks span 3 previously-done cells × {cyclen, f_r}.

**Provenance check (honest out-of-sample confirmed).** The 90 scored `record_id`s
in `data/curriculum/cells/5-5.25_f117/gate.json` were re-scored here and match the
live-rebuilt `make_curriculum_split` `curriculum_val_by_cell` exactly (30/30 per
cell). Because the per-cell 80/20 stable-hash holdout is growth-invariant, these
rows were held out of **both** the champion's and the candidate's training — the
comparison is out-of-sample vs out-of-sample (no phantom in-sample inflation).
Re-scoring reproduced every gate number to ±0.001 (see `old_sp`/`new_sp` below vs
gate.json), so the pipeline is faithful.

Method: both backends loaded via `PosValCnnBackend.from_dir` (cpu); Spearman via
`scipy.stats.spearmanr` (same as the gate). Bootstrap B = 20,000 paired resamples
of the 30 rows (row-index shared across old/new/truth). Jackknife = exact
leave-one-out and leave-two-out over the 30 rows. Per-member = candidate's 5
members scored individually from the raw ensemble forward.

---

## Task 1 — Paired bootstrap + jackknife (per check)

`drop = Sp_champion − Sp_candidate`. `P(>ε)` and `P(>0)` are over the 20k paired
resamples. LOO range = min/max of the 30 leave-one-out drops. loo2 = smallest drop
achievable by removing the single most-favorable **pair** of rows.

| cell | target | n | Sp champ | Sp cand | obs drop | boot mean | 95% CI (drop) | P(drop>0.05) | P(drop>0) | LOO drop range | loo2 best | 1-row fix? | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.25-5.5_f109 | **cyclen** | 30 | 0.985 | 0.838 | **0.147** | 0.157 | **[0.047, 0.314]** | **0.971** | **0.999** | [0.108, 0.163] | 0.088 | none (0/30) | **FAIL** |
| 5.25-5.5_f109 | f_r | 30 | 0.971 | 0.944 | 0.026 | 0.032 | [0.000, 0.077] | 0.154 | 0.975 | [0.020, 0.031] | 0.017 | all (30/30) | pass |
| 5.25-5.5_f117 | cyclen | 30 | 0.962 | 0.932 | 0.030 | 0.034 | [−0.046, 0.112] | 0.316 | 0.838 | [0.021, 0.043] | 0.010 | all | pass |
| 5.25-5.5_f117 | f_r | 30 | 0.923 | 0.900 | 0.023 | 0.024 | [−0.061, 0.126] | 0.239 | 0.708 | [0.005, 0.040] | −0.008 | all | pass |
| 5.25-5.5_f125 | cyclen | 30 | 0.894 | 0.862 | 0.032 | 0.037 | [−0.064, 0.150] | 0.367 | 0.787 | [0.018, 0.056] | 0.006 | 28/30 | pass |
| 5.25-5.5_f125 | **f_r** | 30 | 0.939 | 0.811 | **0.128** | 0.131 | **[0.022, 0.299]** | **0.896** | **0.994** | [0.089, 0.145] | 0.071 | none (0/30) | **FAIL** |

**Reading it.**
- **4 passing checks** are noise-consistent: their 95% CIs include (or nearly
  touch) 0, `P(drop>ε)` ≤ 0.37, and `P(drop>0)` ranges 0.71–0.98. These are small,
  possibly-zero regressions well inside n=30 sampling scatter.
- **2 failing checks are NOT sampling noise around zero.** For **f109 cyclen** and
  **f125 f_r**, `P(drop>0) = 0.999 / 0.994` (the candidate is essentially certainly
  worse on these rankings), and the bootstrap 95% CI **excludes 0**. Jackknife is
  decisive: **0 of 30** single-row removals bring either check to ≤ ε, and even the
  most-favorable **two-row** removal leaves drop = 0.088 / 0.071 (both still > ε).
  So the failures are not driven by 1–2 influential holdout rows.
- The one genuine caveat is the **threshold**, not the sign: both failing CIs have
  lower bounds *below* ε (0.047 and 0.022), and `P(drop>ε)` is 0.97 / 0.90 — high
  but not 1. The regression is real (>0); whether it exceeds the *specific* value
  0.05 carries residual n=30 uncertainty, especially for f125 f_r.

---

## Task 2 — Per-member breakdown of the two failing checks

Candidate (154600) has 5 members (seeds 20260716–20260720). Each member's Spearman
on the exact holdout, and its drop vs the champion ensemble:

**5.25-5.5_f109 · cyclen** (champion Sp = 0.985, candidate ensemble = 0.838)

| member seed | Sp | drop vs champ |
|---|---|---|
| 20260716 | 0.804 | 0.181 |
| 20260717 | 0.793 | 0.192 |
| 20260718 | 0.751 | 0.234 |
| **20260719** | **0.845** | **0.141** |
| 20260720 | 0.756 | 0.229 |

**5.25-5.5_f125 · f_r** (champion Sp = 0.939, candidate ensemble = 0.811)

| member seed | Sp | drop vs champ |
|---|---|---|
| 20260716 | 0.773 | 0.167 |
| 20260717 | 0.772 | 0.167 |
| 20260718 | 0.823 | 0.117 |
| **20260719** | **0.826** | **0.114** |
| 20260720 | 0.734 | 0.206 |

**Reading it.** The regression is **uniform across all five seeds**, not a 1–2 bad
members dragging the ensemble mean. The *best* single member (seed 20260719 in both
cases) still regresses by **0.141** (f109 cyclen) and **0.114** (f125 f_r) — both
far above ε. Member selection **cannot rescue** these checks: there is no subset of
members whose mean would reach the champion's ranking on these two cells. The
ensemble mean (0.838 / 0.811) is already near the top of its member spread, so the
deficit is a property of the trained candidate population, not of ensemble
averaging.

---

## Task 4a — Noise-aware ε calibration (what ε means at n=30)

Null model: two truly-equivalent models produce `drop` centered at 0. Its sampling
SD at n=30 is estimated by recentering the paired bootstrap. Under H0 the old/new
predictions are highly correlated, so the null SD is the **compressed** value the
near-equivalent (passing) checks exhibit, not the inflated SD of a genuinely
regressed check.

- **Null SD per check** σ₀ ≈ **0.042** (median of the 4 passing checks' recentered
  bootstrap SD; per-check SDs range 0.020–0.072, the larger ones being the
  genuinely-regressed checks whose spread is inflated by the real difference).
- **Within-cell correlation** of the (cyclen, f_r) drops ≈ **0.01** → the 6 checks
  behave as ~6 near-independent tests; the 3 cells use disjoint rows.

**False-reject rate of the current gate (ε = 0.05), under an equivalent candidate:**

| statistic | value |
|---|---|
| per-check P(drop > 0.05 \| H0) | **11.8 %** |
| **family-wise P(max of 6 > 0.05 \| H0)** | **≈ 53 %** |

So as configured, the honest gate would reject a *truly-equivalent* candidate more
than half the time on noise alone.

**ε for a controlled false-reject rate at n=30:**

| target | per-check ε | family-wise ε (max of 6) |
|---|---|---|
| 5 % false-reject | 0.069 | **0.101** |
| 10 % false-reject | 0.055 | 0.089 |

(Family-wise from a homogeneous-null max-of-6 simulation; matches the analytic
`σ₀·Φ⁻¹(0.95^{1/6}) = 0.101`.)

**Crucial:** even at the 5 % family-wise-calibrated ε ≈ **0.101**, the two observed
failing drops (**0.147** and **0.128**) still exceed it. Widening ε to a
statistically defensible, multiple-comparison-correct value does **not** turn these
two failures into passes. (The current ε = 0.05, however, is genuinely too tight —
it should be raised to ~0.10 regardless of this candidate's fate.)

---

## Task 3 — v3 control on the fixed split (attribution)

A v3-schema control (`data/models/20260718_162053`) was trained to mirror
`curriculum._retrain_remote_full` **exactly**: `remote.push` (source current) then
remote `train` with `--ensemble 5 --parallel-members 5 --split S1 --device auto
--num-workers 8 --cond-schema v3`. `data/splits/S1.json` was verified `kind =
curriculum_group` (the current curriculum split) **before** training — no rebuild
needed. All three models were then scored on the identical persisted holdout ids.

`drop = Sp_champion − Sp_model`. `*` = drop > ε (0.05). `Δ(v3−v4)` = v3fix Sp −
v4fix Sp (positive → v3 ranks better than v4 on that check).

| cell | target | champ 075428 (v3, **old** data) | v4-fixed 154600 (v4, new split) | v3-fixed 162053 (v3, new split) | Δ(v3−v4) |
|---|---|---|---|---|---|
| 5.25-5.5_f109 | **cyclen** | 0.985 | 0.838  (drop **+0.147***) | 0.836  (drop **+0.149***) | **−0.003** |
| 5.25-5.5_f109 | f_r | 0.971 | 0.944  (drop +0.026) | 0.888  (drop **+0.082***) | −0.056 |
| 5.25-5.5_f117 | cyclen | 0.962 | 0.932  (drop +0.030) | 0.971  (drop −0.009) | +0.039 |
| 5.25-5.5_f117 | f_r | 0.923 | 0.900  (drop +0.023) | 0.921  (drop +0.003) | +0.020 |
| 5.25-5.5_f125 | cyclen | 0.894 | 0.862  (drop +0.032) | 0.778  (drop **+0.116***) | −0.083 |
| 5.25-5.5_f125 | **f_r** | 0.939 | 0.811  (drop **+0.128***) | 0.833  (drop **+0.106***) | +0.022 |
| **worst drop / gate** | | — | **0.147 → FAIL** (2/6 checks) | **0.149 → FAIL** (4/6 checks) | |

**The v3 control also fails the gate — worse than v4.** v3-fixed fails **4 of 6**
checks (worst 0.149) vs v4-fixed's **2 of 6** (worst 0.147). On the worst check
(f109 cyclen) v3-fixed and v4-fixed are **identical** (0.836 vs 0.838; Δ = −0.003)
— that regression is **100 % split/data-inherent, 0 % schema**. On the checks where
the schemas differ, **v4 wins the two large ones** (f109 f_r +0.056, f125 cyclen
+0.083), so the v4 cond_schema is a **net improvement** over v3 on this split, not a
liability.

**Mechanism — negative transfer from the new cell's data (not in-sample inflation).**
The champion (`075428`, 07:54) trained on 34,347 rows **before** the current cell
`5-5.25_f117` was produced; both retrains trained on 34,954 rows **including** that
cell's 300 new converged rows (a new, *lower* e_core band 5.0–5.25). A regression-
error probe rules out champion memorization — on f109 cyclen the champion has both
higher Spearman **and** lower normalized MAE (0.59 vs 0.80/0.88), i.e. a genuinely
better fit, and its val_pred sample shares 0 ids with the holdout. The signature is
**cell-localized**: both retrains keep f117 intact (v3fix even beats the champion on
f117 cyclen, 0.971 vs 0.962) yet both degrade f109-cyclen and f125-f_r. Adding the
5.0–5.25 band data interfered with the adjacent 5.25–5.5 band's ranking — classic
curriculum negative transfer — and the champion escapes it only because it predates
that data. The honest gate is therefore firing **correctly** on a real effect.

---

## Verdict and recommendation

### 1. Are the two failing drops real or n=30 noise? — **REAL.**
Not sampling noise around zero: `P(drop>0) = 0.999` (f109 cyclen) and `0.994`
(f125 f_r); both bootstrap 95 % CIs exclude 0; jackknife is robust (0/30 single-row
and best two-row removals fix either); and the deficit is **uniform across all 5
seeds** (best member still −0.141 / −0.114). They also exceed the noise-aware
**family-wise ε ≈ 0.10**, so they are not an artifact of the gate's tight threshold.
(The *magnitude vs 0.05* still carries some n=30 uncertainty — CI lower bounds
0.047/0.022 — but the regression itself is certain.)

### 2. v4-specific or split-policy inherent? — **SPLIT/DATA-INHERENT.**
The v3 control on the identical fixed split **fails the gate harder** (4/6 checks,
worst 0.149) than v4 (2/6, worst 0.147). On the worst check the two schemas are
identical (0.836 vs 0.838). Root cause: adding cell `5-5.25_f117`'s 300 new rows
caused **negative transfer** onto f109-cyclen and f125-f_r; the champion avoids it
only by predating that data. **v4 is not the culprit — it is a net improvement over
v3 on this split.** Member selection is likewise irrelevant (uniform across seeds).

### 3. Which intervention does the evidence support?
- **NOT** a schema change / v4 rollback → v4 beats v3 on the fixed split.
- **NOT** member/seed selection → regression is uniform across all 5 members.
- **NOT** "promote via a noise-aware ε" → the drops (0.147, 0.128) exceed even the
  family-wise-calibrated ε ≈ 0.10.
- **YES — curriculum-level negative-transfer mitigation** for the `5-5.25_f117`
  step: up-weight / replay the regressing previous-cell rows (the per-member
  `cell_weight` cap is currently 8.0 — raise it for f109-cyclen & f125-f_r, or add
  rehearsal of those cells), then re-retrain and re-gate. Optionally audit the new
  cell's 300 labels for a regime/normalization shift that drives the interference.

### 4. Recommended gate decision.
**Under the current honest-gate contract (reject if any drop > ε): REJECT
`20260718_154600` — the regression is real, > ε, and not noise-excusable.** But the
reject should be read as a **curriculum-methodology signal, not a candidate defect**:
no retrain that includes the new cell can pass this gate against a champion that
predates it (the v3 control proves it, worse). So the actionable path is **re-retrain
`5-5.25_f117` with the transfer mitigation above**, not to keep looping the current
recipe.

If, instead, the curriculum owner judges that bounded, documented regressions are an
acceptable price for cross-band coverage (the champion is **blind** to the 5.0–5.25
band, which the whole curriculum exists to learn, and a strict ε=0.05 no-regression
rule is structurally incompatible with curriculum learning), then the promote choice
is unambiguous: **promote `20260718_154600` (v4), never the v3 control** — 154600 is
the best available model (passes new_cell 0.7498, dominates v3 on the fixed split),
and its two carried regressions (0.147, 0.128; absolute Sp still 0.84/0.81) are the
minimum achievable by any same-split retrain. This is a **policy** decision, not one
that noise statistics can make.

### 5. Gate hygiene (independent of this candidate).
The current **ε = 0.05 is statistically indefensible** for a max-of-6 statistic at
n=30: under a truly-equivalent candidate it false-rejects **11.8 % per check** and
**≈ 53 % family-wise**. For a 5 % family-wise false-reject the threshold should be
**ε ≈ 0.10** (per-check ≈ 0.07); for 10 % family-wise, ε ≈ 0.09. Recommended fixes,
in priority order: (a) raise ε to ~0.10; (b) enlarge the per-cell holdout (n=30 →
≥60–100) to shrink σ₀ ≈ 0.042 and tighten the gate honestly; (c) re-baseline the
no-regression champion to a **same-data-regime** model (e.g. compare against the v3/
v4 fixed retrains, not a champion trained on an older store) so the gate measures
candidate-vs-candidate transfer rather than candidate-vs-stale-reference. Note:
**even at ε = 0.10, `154600` still fails f109 cyclen (0.147)** — ε recalibration
alone does not promote it; only the policy decision or the transfer fix does.

---

### Appendix — provenance & reproducibility
- Champion `20260718_075428` (v3, `--split S1`, train 34347/val 6195, pre-`5-5.25_f117`).
- v4-fixed candidate `20260718_154600` (v4, curriculum split, train 34954/val 6008).
- v3-fixed control `20260718_162053` (v3, **same** curriculum split S1, this analysis).
- Holdout: 90 ids in `data/curriculum/cells/5-5.25_f117/gate.json`
  `no_regression.scored_record_ids`; verified == live `make_curriculum_split`
  `curriculum_val_by_cell` (30/30 per cell); held out of all trainings by
  stable-hash growth-invariance.
- Spearman via `scipy.stats.spearmanr`; gate numbers reproduced to ±0.001. Bootstrap
  B=20k (drop CIs) / 40k–60k (ε calibration) paired row resamples; jackknife exact
  LOO and leave-2-out; per-member from the raw ensemble forward (member-mean == the
  `.predict` ensemble to 0.0). No source or state file was modified; the v3 control
  is a new artifact under `data/models/`.
