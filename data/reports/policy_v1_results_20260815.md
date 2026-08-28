# Policy net v1 — results

Pre-registration: `data/reports/policy_v1_prereg_20260815.md` (written before any
weight was trained; **not edited since** — every deviation is reported below).
Corpus: `data/policy/steps.parquet`, SHA-256 `6e3dc91f84471305…`.
Run: 238 GPU 1, `~/lpopt_ws/runs/policy_v1`, `rc=0`, `DONE`, 994 s wall for the
whole thing (10 members + the full bootstrap evaluation).
Checkpoints pulled to `data/models/policy_v1/`.
Every number below is rendered from `data/models/policy_v1/metrics.json` by
`python -m lpopt.policy.train --tables`, not transcribed.

---

## 1. RESULT — gate PASS for the CNN arm, on both heads

| | `cnn` | `mlp` |
|---|---|---|
| F_r head | **PASS** | FAIL |
| flatness head | **PASS** | FAIL |
| params | 1,690,114 | 144,898 |

On `test` (in-distribution, lineage-component-grouped, 1,615 / 1,387 labelled
rows), the 5-seed CNN ensemble beat **all three** pre-registered baselines on the
deployment metric with paired 95% CIs excluding zero, and cleared the AUC floor
with room to spare:

| head | AUC (95% CI) | precision@32 | random | class_freq | periph |
|---|---|---|---|---|---|
| `fr` | **0.790** [0.768, 0.811] | **0.842** | 0.364 (+0.479 [+0.281, +0.656]) | 0.477 (+0.365 [+0.188, +0.562]) | 0.441 (+0.401 [+0.219, +0.594]) |
| `flat` | **0.773** [0.748, 0.797] | **0.807** | 0.397 (+0.410 [+0.219, +0.594]) | 0.444 (+0.363 [+0.188, +0.531]) | 0.494 (+0.313 [+0.125, +0.500]) |

The advisory (non-gated) `cell_class_freq` lookup table reaches 0.499 / 0.477 —
also beaten by a wide margin.

**And it is ranking moves, not parents.** The confound the prereg flagged in
section 4b is controlled by the parent-blocked AUC, which differences out parent
difficulty entirely: **0.771 (`fr`, 1,739 pairs)** and **0.764 (`flat`, 1,551
pairs)** on `test`, against 0.482 / 0.491 for random and 0.530 / 0.555 for the
`periph` heuristic. Given a parent board and a set of candidate moves from it,
the policy orders the improving ones first roughly three times out of four.

Per-seed spread is tight — `fr` 0.769–0.778, `flat` 0.752–0.761 — so the result
is not a lucky seed; the 5-seed ensemble buys about +0.015 AUC over the mean
member.

## 2. Held-out transfer

| fold | what it tests | `fr` AUC | `flat` AUC | `fr` p@32 | beats all 3? |
|---|---|---|---|---|---|
| `test` | in-distribution | 0.790 | 0.773 | 0.842 | **yes / yes** |
| `heldout_cell` (`B1_C6/f121/260624`) | unseen cell, familiar library | **0.822** | **0.811** | **0.888** | **yes / yes** |
| `heldout_lib` (`5.8_5.1`, 5 cells) | unseen library, same era | 0.696 | 0.662 | 0.554 | no / no |
| `heldout_era` (ga80+paramA, 54 cells) | **the live operating point** | 0.650 | 0.682 | 0.328 | no / no |

**Cell transfer is the strong result.** On a whole cell the model never saw,
performance does not merely hold — it is the best fold in the study (p@32 0.888
on `fr`, parent-blocked AUC 0.826). Cells of a familiar library are, to this
model, interchangeable in exactly the way corpus section 9a said they would not
be for a *state* model. That is a real and useful finding: the move-scoring
problem generalises across cells far better than the board-value problem does.

**Era transfer degrades, exactly as the prereg predicted.** On ga80/paramA — the
feeds 101-141 near F_r=1.55 that the live program actually runs — AUC falls to
0.650/0.682 and the deployment metric no longer clears the baselines. The signal
is real (CIs [0.611, 0.688] and [0.632, 0.732] both exclude 0.5) and the
parent-blocked AUC for `fr` is a healthy 0.752, but this is **not** a passing
readout and the policy must not be wired into the running campaign on the
strength of the `test` column alone. This is the partial-failure mode the prereg
named as most likely, and it is what happened.

## 3. The pre-registered comparison: `cnn` vs `mlp` — the board tensor earns its place

The scalar-only control reaches 0.643 / 0.610 test AUC and beats **none** of the
three baselines on precision@32 at 95%. Its parent-blocked AUC is 0.585 / 0.595.

The gap is not marginal: 0.790 vs 0.643 AUC, 0.842 vs 0.579 precision@32, 0.771
vs 0.585 parent-blocked AUC. Twelve times the parameters buys roughly the whole
result. **Ship the CNN.** The 36 tabulated move descriptors — move class, edit
counts, ring deltas, radial direction — carry perhaps a third of the available
signal; the rest lives in the spatial detail of *which* slots moved and what
was in them, which only the board tensor sees.

## 4. Post-hoc control the prereg did NOT contain — the free physics baseline

Declared as post-hoc and therefore **not part of the gate**. It was run because
one channel of the input is `prior_power`, an analytic diffusion solve of the
board's power map that costs nothing at proposal time, and the honest question
is whether the network is doing anything the free physics does not already do.
Score = −(child prior power peak − parent prior power peak).

| fold | head | prior AUC | policy AUC | prior pb-AUC | policy pb-AUC | prior p@32 | policy p@32 |
|---|---|---|---|---|---|---|---|
| test | fr | 0.571 | **0.790** | 0.560 | **0.771** | 0.477 | **0.842** |
| test | flat | 0.564 | **0.773** | 0.558 | **0.764** | 0.481 | **0.807** |
| heldout_cell | fr | 0.620 | **0.822** | 0.617 | **0.826** | 0.610 | **0.888** |
| heldout_cell | flat | 0.608 | **0.811** | 0.602 | **0.831** | 0.586 | **0.866** |
| heldout_era | fr | **0.746** | 0.650 | 0.535 | **0.752** | **0.411** | 0.328 |
| heldout_era | flat | **0.755** | 0.682 | **0.647** | 0.570 | **0.265** | 0.253 |

In distribution and on the unseen cell the network beats the free physics
decisively — it is not a diffusion-solve wrapper. **On the unseen era the
relationship inverts on AUC and precision@32**: the analytic prior ranks
ga80/paramA moves better than the trained policy does (0.746 vs 0.650 on `fr`).

But the two are winning at *different things*, and the decomposition is the most
useful thing this study produced. On `heldout_era`/`fr` the prior's global AUC is
0.746 while its parent-blocked AUC is 0.535 — near chance. It is a good **board**
ranker and a poor **move** ranker: it identifies which boards are globally peaked,
which is most of the pooled-AUC signal on a fold whose base rate is 0.188. The
policy is the reverse (0.650 global, 0.752 parent-blocked). They are
complementary, and section 7 uses both.

## 5. Calibration — reported, not gated

`test` ECE 0.043 (`fr`) / 0.049 (`flat`), Brier 0.179 / 0.187. The reliability
curve is monotone and mildly **under**-confident across the whole range (bin
[0.2,0.3): predicted 0.245, observed 0.354; bin [0.9,1.0): predicted 0.935,
observed 1.000). Usable as a prior weight after a one-parameter temperature fit;
no recalibration was performed here.

On `heldout_era` calibration degrades sharply (ECE 0.111 `fr`, 0.200 `flat`) —
the model is confidently wrong about absolute probabilities off-distribution even
where its *ranking* still carries signal. Any deployment must use it as a ranker,
never as a probability, until an era-matched calibration set exists.

## 6. Deviations from the pre-registration

Four, all declared in the prereg itself except the last:

1. **Deployment metric batches are drawn from the fold, not from one parent.**
   The corpus cannot supply 256 candidates per parent (max 42; one parent has
   ≥32). Declared in prereg §4b; the parent-blocked AUC is the control and it
   passes.
2. **Realized split fractions undershoot nominal.** `test` landed at 8.2% rather
   than 10% because the greedy pack only takes whole lineage components. `test`
   also carries a higher base rate (0.389) than `train` (0.349). Both were
   declared in prereg §3.
3. **The s1e transfer-init comparison was dropped**, replaced by `cnn` vs `mlp`.
   Reasons in prereg §6a: no matched geometry (s1e is 10.4M params, 6× budget)
   and — decisively — s1e was trained with direct supervision on the `f_r` and
   `node_peak` of the very store records that are this corpus's parents and
   children, so a warm start would have contaminated every held-out number.
4. **Post-hoc addition:** the physics-prior baseline in section 4. It is
   reported outside the gate and did not change any pass/fail decision.

## 7. Integration sketch — how `construct.build_pool` would consume this

*Sketch only; no integration code was written this round.*

`lpopt/search/construct.py::build_pool` has exactly one surrogate hook today:
`_score_completions` (L368-390) calls `model.predict(patterns, …)` on **complete**
patterns and keeps the beam-best by `mean[:,0] + mean[:,2]` (F_r + CBC proxy).
The policy is a different kind of object — it scores an *edit*, not a board — so
it plugs in one stage earlier, at the two `mutate(...)` calls.

**Where.** Stage 1, elite mutation (L257): today the loop draws one child per
parent per round and admits the first novel one. Replace "first novel" with
"best of N": for each parent, generate `N ≈ 16` mutations, featurize the
(parent, child) pairs, score them with the policy, and admit the top `m`. That is
a drop-in over an existing loop — `_admit`'s dedup and cap are untouched — and it
is exactly the operation the parent-blocked AUC measures, which is why that
number, not the pooled one, is the readout that licenses this.

**How.** A `MoveScorer` with one method, mirroring `_score_completions`'s
tolerance for failure (L386 swallows every surrogate exception because "the
surrogate must never abort construction" — the prior must inherit that):

```
score(parent_pattern, [child_patterns], ctx) -> np.ndarray[n, 2]   # P(fr), P(flat)
```

Cost: one `FeatureEncoder.encode_slot_matrix` per distinct board (~19 ms) plus a
batched forward. The parent is encoded once per expansion and reused, so a
16-candidate expansion is ~17 encodes + one 16-row forward — comparable to the
existing beam scoring and far below a MASTER call.

**Combination, not replacement.** Section 4 shows the policy and the free
analytic prior win at different things off-distribution. The prior belongs where
it already is (ranking complete boards in `_score_completions`); the policy
belongs at the mutation step (ranking edits from a fixed parent). Wiring the
policy as a *replacement* for the board scorer would discard the half of the
signal that survives era transfer best.

**Safety rails the era result forces.**
- Sample from a **softmax over policy scores with temperature**, not `argmax` —
  a hard argmax on a scorer with 0.650 era AUC collapses diversity, and
  `build_pool`'s stage-3 diversity quota exists precisely to stop that.
- Keep a **floor of unscored random mutations** (stage 3/4 already provide it)
  so the pool cannot become policy-degenerate.
- Score only **within a fixed parent**, never across parents — the pooled metric
  is confounded and the parent-blocked metric is what passed.
- **Gate on an A/B**, not on this report: paired waves with the prior on and off,
  measuring elites-found-per-MASTER-call.

## 8. Verdict and next step

**Gate: PASS on `test` for both heads, CNN arm.** The corpus supports a learned
move-proposal policy; the flattening signal corpus section 4 measured is
learnable and substantially exceeds the engineer's radial rule of thumb
(precision@32 0.842 vs 0.441). The board tensor is load-bearing; the scalar
control fails the gate outright.

**Recommendation: A/B campaign wiring, but scoped to the SA-era libraries — and
one more data step before the live cells.**

The honest reading of section 2 is that this model is validated for 260624-family
cells (`heldout_cell` p@32 0.888) and *not* validated for the ga80/paramA cells
the live program runs. Two options, in order of cost:

1. **Cheap and immediately useful:** wire the prior into `build_pool` behind a
   flag and run the paired A/B on a 260624 cell, where the model is validated.
   That measures the thing that actually matters — elites per MASTER call — on
   ground where the readout is trustworthy, and it exercises the integration
   before the live cells depend on it.
2. **What the era gap actually needs:** the 1-move ablation wave corpus section
   10 already prescribes, run on a **ga80 or paramA** cell rather than the main
   260624 cell. A few thousand MASTER calls of balanced, single-move,
   direction-stratified data at the live operating point would fix the era gap
   at its source and would simultaneously settle the leakage-arbitration
   question that kept the third head out of scope. Retraining on it is then a
   few minutes.

More data, in other words — but *targeted* data, at the operating point where
the model is weak, not more of the corpus it is already saturated on. Doing (1)
and (2) in parallel costs nothing extra: the A/B needs no new MASTER budget
beyond its own waves, and the ablation wave is queued regardless.

## 9. Files

| path | what |
|---|---|
| `lpopt/policy/data.py` | universe, component-grouped splits, leakage guard, feature cache, dataset |
| `lpopt/policy/net.py` | `PolicyNet` — CNN arm (imports `FiLM`/`ResidualBlock` from `lpopt/model/net.py`) and MLP control |
| `lpopt/policy/train.py` | training loop, metrics, baselines, bootstrap, `--tables` renderer |
| `train_policy_v1.py` | ship corpus + launch on 238 + poll + pull (reuses `lpopt.remote`) |
| `tests/test_policy_v1.py` | 9 guards: leakage, split determinism, no component straddling train/test, transpose invariance, metric correctness |
| `data/models/policy_v1/` | 10 checkpoints (`{cnn,mlp}_seed2026081{5..9}/model.pt` + `meta.json`), `metrics.json`, `probs_*.npz`, `train.log` |
| `data/reports/policy_v1_prereg_20260815.md` | the pre-registration |

The corpus parquets and `data/store` were not modified.
