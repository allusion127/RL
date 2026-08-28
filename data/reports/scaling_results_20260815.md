# Capacity & ensemble scaling — results

**Pre-registration**: `data/reports/scaling_prereg_20260815.md`, written and locked
**before any weight was trained**. Every metric below was fixed in §6 of that
document, and is reported here **whether or not it favours scaling**.

**Question** (user): *"파라미터 수나 앙상블 학습 수를 늘리면 더 나은 결과가
나오지 않을까"* — does scaling capacity and/or ensemble size improve
(a) prediction accuracy and (b) search-relevant quality?

**Answer in one line**: **No. Neither axis clears its pre-registered gate, and
the experiment had the statistical power to have seen it if it were there.**
Capacity above the champion is flat (best gain +0.009 against a +0.02 bar, with
`mde80` ≈ 0.009–0.015); ten members buy nothing that two independent five-member
ensembles do not already differ by. The one clearly *positive* finding is
negative in direction: **shrinking the champion hurts significantly**, so 10.4M
is at the knee, not past it.

**No promotion.** Per prereg §7.5 this round promotes nothing and changes no
default. `s1e` remains champion.

---

## 1. What ran

Five training runs, all `rc=0`, sequential on **238 GPU 0** (`policy_v1` held GPU
1 and finished on its own; it was never touched). Store hashes verified identical
to prereg §3 before launch.

| arm | width/blocks/head | params | ×champ | best ep | stop | wall |
|---|---|---:|---:|---:|---:|---:|
| **C0** | 160 / 7 / 256 | 4,788,419 | 0.46 | 25 | 40 | 403 s |
| **C1 — CONTROL** | 224 / 8 / 384 | 10,363,715 | 1.00 | 68 | 83 | 1,296 s |
| **C2** | 320 / 10 / 512 | 24,889,315 | 2.40 | 76 | 91 | 2,811 s |
| **C3** | 384 / 11 / 640 | 38,493,347 | 3.71 | 79 | 94 | 3,703 s |
| **E5′** | 224 / 8 / 384 ×5 | 5 × 10,363,715 | — | — | — | 9,512 s |

Surface: S1e val fold, **11,712 converged rows / 58 cells** (50 at ≥8 rows;
2,371 rows carry a map label; 26 cells at ≥64 rows for precision@32). All arms
scored on identical rows in identical order through the **serve path**.

Deviation from prereg §9: **none**. All five runs completed; the 14-GPU-hour
budget cap was not reached (~10.3 h total incl. dumps), so C3 was not dropped.

---

## 2. Axis A — capacity: **GATE FAILED (flat above, harmful below)**

### 2.1 M1 — iso-group (within-cell) Spearman, median across cells

| arm | ×champ | F_r (50c) | node_peak (48c) | map_cov (48c) | cyclen (50c) |
|---|---:|---:|---:|---:|---:|
| C0 | 0.46 | 0.9001 | 0.9060 | 0.9195 | 0.8311 |
| **C1 control** | 1.00 | 0.9175 | 0.9201 | 0.9490 | 0.8939 |
| C2 | 2.40 | **0.9285** | 0.9219 | **0.9520** | **0.8986** |
| C3 | 3.71 | 0.9219 | **0.9305** | 0.9435 | 0.8842 |

### 2.2 The registered test — paired cell BCa bootstrap vs C1

`ab_paired.paired_cell_bootstrap`, repo defaults (`reps=2000, seed=0, α=0.05,
method=bca, aggregate=median`). Point = **median of per-cell differences**.

| comparison | F_r | node_peak | map_cov | cyclen |
|---|---|---|---|---|
| **C2−C1** | +0.0005 [−0.0067, +0.0072] | **+0.0091 [0.0000, +0.0196]** | +0.0011 [−0.0029, +0.0101] | +0.0047 [−0.0047, +0.0134] |
| **C3−C1** | −0.0034 [−0.0101, +0.0064] | +0.0082 [−0.0049, +0.0175] | −0.0018 [−0.0085, +0.0030] | −0.0037 [−0.0128, +0.0026] |
| **C0−C1** | **−0.0186 [−0.0254, −0.0076]** | **−0.0099 [−0.0201, −0.0026]** | **−0.0173 [−0.0301, −0.0094]** | **−0.0443 [−0.0617, −0.0270]** |

> **Gate §7.1** (≥ +0.02 on ≥2 targets, CI excluding 0): **FAILED by both C2 and
> C3.** The largest gain anywhere above the control is **+0.0091** — less than
> half the bar — and it is the only one whose interval excludes zero.

### 2.3 This is a real null, not an underpowered one

The bootstrap reports `mde80`, the effect detectable at 80% power:

| comparison | F_r | node_peak | map_cov | cyclen |
|---|---:|---:|---:|---:|
| C2−C1 `mde80` | **0.0089** | **0.0120** | **0.0093** | **0.0149** |
| C3−C1 `mde80` | 0.0131 | 0.0134 | 0.0094 | 0.0092 |

**All eight are below the +0.02 gate.** Had a +0.02 capacity effect existed on
any of these targets, this design would have detected it. The null is
informative, not merely inconclusive — which is the reading prereg §7.4 fixed in
advance.

### 2.4 §7.3 disambiguation — saturation vs optimization failure

The pre-registered discriminator, read off `meta.json` `history.train_loss`:

| arm | min train loss | vs C1 | registered reading |
|---|---:|---|---|
| C0 | +0.1205 | much worse | underfits — consistent with its val loss |
| **C1** | **−0.1462** | — | control |
| **C2** | **−0.1505** | **better** | **genuine CAPACITY SATURATION** — fits train better, gains ≤0.009 on val |
| **C3** | **−0.1452** | **worse** | **OPTIMIZATION FAILURE** — inconclusive about capacity |

This distinction is load-bearing and was registered before the numbers existed:

* **C2 is the decisive arm.** At 2.40× it optimized *better* than the control
  (lower train loss) and still produced no val-side gain that clears the bar,
  with adequate power. That is capacity saturation in its clean form.
* **C3 is not evidence that capacity is exhausted.** At 3.71× under the unchanged
  LR (prereg §4.3) it failed to even match C1's train loss, so its flat/negative
  val result is attributed to optimization, and §7.3 requires it be reported as
  **INCONCLUSIVE on capacity above 2.4×**, not as a capacity ceiling.

### 2.5 The unregistered-but-reported finding: the champion is AT the knee

Prereg §2.3 predicted C0 would lose **< 0.02**. It lost **more than that on
three of four targets** (cyclen −0.0443, F_r −0.0186, map_cov −0.0173), every
interval excluding zero.

> **That half of the registered prior is REFUTED on the record.** The curve is
> *not* flat in both directions. 10.4M sits at the knee: you cannot buy accuracy
> by going bigger, and you would pay real accuracy to go smaller. The champion's
> width was, in hindsight, well chosen.

### 2.6 Search-relevant: the largest arm is actively harmful

| arm | M5 precision@32 mean | median | vs C1 (BCa) |
|---|---:|---:|---|
| C0 | 0.6911 | 0.7500 | −0.0312 [−0.0312, 0.0000] |
| **C1** | **0.7188** | 0.8125 | — |
| C2 | 0.7019 | 0.7813 | 0.0000 [−0.0312, +0.0156] |
| **C3** | **0.5673** | 0.6094 | **−0.1250 [−0.1719, −0.0938]** |

C3 degrades the acquisition's own selection metric by **−0.125, significantly**.
This reproduces the prior A5 pattern (`hires_model_ab_design_20260725.md` §14.3:
worst honest-gate drop 0.294, Nyquist ratio *below* control) — **pure width
scaling has now damaged search-relevant quality in two independent rounds.**

---

## 3. Axis B — ensemble 5 → 10: **GATE FAILED**

### 3.1 The numbers

| arm | members | ρ F_r | ρ node_peak | ρ map_cov | ρ cyclen | ECE_σ F_r | ECE_σ cyclen | prec@32 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E5** (s1e) | 5 | 0.9396 | 0.9360 | 0.9559 | 0.9281 | 0.1544 | **0.0884** | 0.7404 |
| **E5′** (new) | 5 | 0.9398 | 0.9262 | 0.9522 | 0.9106 | 0.1742 | 0.2123 | 0.7476 |
| **E10** | 10 | 0.9440 | 0.9330 | 0.9563 | 0.9243 | **0.1228** | 0.2017 | 0.7476 |

> **Gate §7.2** (lower ECE_σ on **both** F_r *and* cyclen, **and** higher mean
> precision@32 with CI excluding 0): **FAILED on both legs.**
> ECE_σ F_r improves (0.1544 → 0.1228) but **ECE_σ cyclen worsens**
> (0.0884 → 0.2017); precision@32 is **0.0000 [0.0000, +0.0156]** — straddles null.

### 3.2 The E5′ yardstick is what makes this decisive

E5′ is the same recipe as E5, differing only in seeds. Any E10−E5 difference
must beat this floor to mean anything:

| metric | **E10 − E5** | **E5′ − E5 (pure noise)** |
|---|---|---|
| ρ F_r | +0.0019 [+0.0004, +0.0037] | −0.0012 [−0.0033, +0.0011] |
| ρ cyclen | −0.0013 [−0.0039, −0.0000] | **−0.0089 [−0.0132, −0.0070]** |
| ρ map_cov | −0.0015 [−0.0038, 0.0000] | **−0.0068 [−0.0094, −0.0023]** |
| ρ node_peak | −0.0014 [−0.0048, +0.0023] | −0.0058 [−0.0118, +0.0005] |
| ECE_σ cyclen | +0.113 | **+0.124** |
| prec@32 mean | +0.0072 | +0.0072 |

**Every E10−E5 difference is smaller than the corresponding E5′−E5 difference.**
Two nominally equivalent five-member ensembles differ *significantly* on cyclen
(−0.0089) and map_cov (−0.0068) — larger than anything doubling the ensemble
produced. E10's statistically significant +0.0019 on F_r is real but
**operationally negligible**, and its ECE_σ cyclen "degradation" is smaller than
the E5↔E5′ disagreement, so it cannot be attributed to member count either.

**Verdict: 5 members already saturate the epistemic term. Do not go to 10.**

### 3.3 Methodological byproduct worth keeping

E5′−E5 is, incidentally, the first direct measurement of **ensemble-identity
noise** on this surface: **~0.009 on cyclen and ~0.007 on map_cov, significant.**
Future A/B rounds that compare two 5-member ensembles should treat within-cell ρ
differences below **~0.01 on cyclen/map_cov** as *not attributable to the
treatment*. Several past rounds have reported deltas in that range.

Related, from the training logs: five seeds of the identical architecture spread
**0.8677–0.8993** in training-time within-case ρ F_r (range 0.0316, sd 0.0126).
A single-seed difference therefore carries sd ≈ 0.0126·√2 ≈ **0.018** — which is
why prereg §8.1 pre-committed to reading any single-target 0.02–0.03 capacity
gain as within seed noise, and why §7.1 required two targets.

---

## 4. Registered metrics that did not discriminate — reported anyway

**M6 selection regret = 0.0000 for every single arm**, CI [0, 0], flagged
`degenerate` by the bootstrap. The top-32 by acquisition key always contains the
cell's true minimum F_r, in all seven arms. The metric is **saturated on this
surface and cannot fail** — it should not be reused as a discriminator without
raising K or restricting to harder cells. Reported because §6.2 registered it.

**M3 `T6_T4` in-cell ρ** (144 rows, 5 cells — registered as deciding nothing):
C0 0.373 · C1 0.447 · C2 0.477 · C3 0.564 · E5 0.649 · E5′ 0.567 · E10 0.563.
The spread between two equivalent ensembles (E5 0.649 vs E5′ 0.567) is larger
than most arm-to-arm differences, confirming it is too noisy to carry weight.

**M2 P@10%** moved by exactly 0.0000 in almost every comparison — the within-cell
top-decile sets are largely identical across arms; it added no information beyond
M1.

**σ-calibration caveat (flagged before results, per the interim note):** the
capacity arms are **1 seed**, so they have **no epistemic term** by construction
and their σ is aleatoric-only. Their M4 numbers are comparable *to each other*
but **not** to the 5-/10-member ensembles. The instability is severe and worth
recording: raw mean σ_cyclen is **10.6 EFPD (C2), 23.3 (C1), 98.8 (C3)** — a
9× swing across single seeds, versus a stable 29–37 for the ensembles.
**Single-model raw σ on this surface is not trustworthy; ensembling is what makes
σ usable at all** — but 5 members already achieve that.

---

## 5. Verdict against the registered prior

Prereg §2.3, quoted verbatim:

> *"Scaling from 10.4M to ~25M and ~38M under the champion recipe will produce
> < +0.02 iso-ρ on every target, and will not improve σ-calibration. The 0.46×
> point (4.8M) will lose < 0.02 relative to the control — i.e. the champion is at
> or past the knee, and the curve is flat in both directions."*

| clause | outcome |
|---|---|
| < +0.02 on every target at 25M/38M | **CONFIRMED** (max +0.0091) |
| no σ-calibration improvement from capacity | **CONFIRMED** (ECE erratic, C3 worst at 0.4328) |
| C0 loses < 0.02 | **REFUTED** — lost 0.0186–0.0443 on 3 of 4 targets |
| "flat in both directions" | **REFUTED** — flat above, steep below |

The prior was right about the upside and **wrong about the downside**, and the
corrected picture is more useful than the prior would have been: the champion is
sitting *on* the knee.

### Falsification reading (prereg §7.4)

Axis A is flat above the champion, and §7.3 attributes C2's flatness to genuine
saturation. The registered consequence therefore fires:

> Capacity above 10.4M is **not** the binding constraint. Combined with the prior
> art — structure returned **44× more per parameter** than width
> (`hires_model_ab_design_20260725.md` §14.3), labels are **not** the ceiling
> (ρ_max ≥ 0.997, `precision_ceiling_roadmap_20260725.md` §1), and density has a
> **−0.2…−0.4** learning-curve exponent (§3.3) — the surviving levers are
> **structure** and **label information density per MASTER call**. This result
> **strengthens the label-density/-resolution program**: the EDIT 6 axial parser,
> the full 30-step EDIT 5 harvest, and map regime coverage (§4.5 of that report,
> "추가 계산비 0, 저장 0.35 GiB").

**Pure width scaling is now falsified for the fourth time on this project** (2×
twice, 4.2× as A5 with +0.141 but 1/44 the per-parameter return of structure, and
now 2.4×/3.7× above the champion with ≤ +0.009).

---

## 6. Recommendation

1. **Do not scale capacity.** 10.4M is the knee. Do not go bigger (no gain, and
   38.5M significantly *harms* precision@32 by −0.125); do not go smaller (4.8M
   loses up to −0.044 iso-ρ).
2. **Do not go to 10 members.** 5 members already saturate the epistemic term;
   doubling inference cost buys less than the noise between two equivalent
   5-member ensembles. If σ quality is the goal, the lever is
   **calibration/conformal**, not more members.
3. **Spend the freed budget on label information density** — the EDIT 6 axial +
   30-step EDIT 5 harvest + map regime coverage. It is the only remaining
   order-of-magnitude lever identified anywhere in this repo's measurements, and
   every day it stays off, that day's production is permanently lower-resolution.
4. **One honest loose end.** C3's optimization failure means *capacity above
   2.4× has not been cleanly tested*. If anyone wants to close it, it needs a
   width-aware LR rule (µP-style), not another fixed-LR run. **Priority: low** —
   C2 was cleanly optimized, adequately powered, and flat.
5. **Adopt the noise floors** measured here in future A/B readings: ~0.01 (cyclen
   / map_cov) between 5-member ensembles, and ~0.018 for single-seed differences.

---

## 7. Artifacts

Pulled to `data/models/scaling_20260815/` (529 MB): `scale_C0/ scale_C1/ scale_C2/
scale_C3/ scale_E5b/` (checkpoints + `meta.json` + `train.log`),
`scaling_metrics.json` / `.csv`, `scale_master.log`, `dump_all.log`,
`raw_dumps/scaling_raw_*.npz` (per-member val-fold prediction stacks for all
seven arms), and the scripts (`dump_scaling.py`, `score_scaling.py`,
`scale_run.sh`, `dump_all.sh`).

**No file in `lpopt/` was modified in this round.** The store was read-only
throughout; 199 and local MASTER regeneration were untouched.

### Implementation notes (decisions made after the prereg, disclosed)

* **Scoring runs through the serve path** (`PosValCnnBackend`), not
  `train.predict_dataset`. Required: the **cyclen physics-prior add-back lives
  only in `_ensemble_raw`**, so scoring the raw head would have made every cyclen
  number silently wrong.
* **σ metrics use the uncalibrated raw composition** per prereg §5.1, for all
  arms including Axis A (the prereg fixed this only for Axis B). Justification:
  per-cell calibration is documented rank-preserving (`a > 0`), so **M1/M2/M5 are
  invariant to it**; only M4 is affected, and using raw σ keeps every arm on one
  footing. Serve-path calibrated values are in the npz dumps (`srv_*_cal`) for
  anyone who wants the other view.
* E10 was composed by concatenating de-standardised per-member stacks (each run
  de-standardised with **its own** `tmean`/`tstd`), exactly as prereg §5.1
  specified; `target_names` agreement was asserted at compose time.
