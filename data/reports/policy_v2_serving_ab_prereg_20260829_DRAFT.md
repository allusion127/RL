# Policy serving A/B/C — pre-registration (DRAFT)

**Status: DRAFT, not registered.** Nothing here is executable until (a) the F_xy
head ships and a wave's `selection.json` records `fxy_source = "head"`, and
(b) the user approves the MASTER budget in §8. Registration = this file renamed
without `_DRAFT` and its `corpus/model` fingerprints in §9 filled in.

Answers review `RL_core_loading_engineer_AI_review_2026-08-29.md` §6.2 (P0-01)
items 3–5 and §6.12 (P1-05). The serving wiring those items require is
implemented (Phase A-1, 2026-08-29); this document is the experiment that wiring
exists to run.

---

## 0. The question

Does the learned move policy, used at the elite-mutation step of `build_pool`,
make the campaign find better cores — not better AUC?

The repository's name and its stated goal claim a program that has *learned core
loading rules*. Today the search's performance comes from the surrogate
acquisition, the policy defaults to `off`, and no campaign has ever been decided
by it. That claim is therefore **unmeasured, in either direction**. This
experiment measures it, on the operating point the program actually runs, with
the metric the program is actually judged on.

It is a prospective test, and it is designed to be losable: §7 states in advance
what result retires the policy arm.

---

## 1. Arms

Three arms, one cell, one objective, identical everything else.

| Arm | `[acquisition] policy_prior` | Elite draw | Role |
|---|---|---|---|
| **A (control)** | `off` | unscored random mutation | the campaign as it ships today |
| **B (v1)** | `v1` | v1 ensemble ranks, softmax-samples | the registered baseline, incl. its own failure mode |
| **C (v2)** | `v2` | v2 ensemble ranks, softmax-samples | the treatment |

v1 is in the design **because it is expected to lose**. It failed prospectively
on this era at parent-blocked AUC 0.492 (chance) while passing on the legacy era
(`policy_v1_results_20260815.md`), and v2's whole construction is the correction
of that failure. An A/B/C in which B ≈ A and C > A is a far stronger statement
than C > A alone: it shows the campaign-level readout can *tell the two policies
apart*, which is exactly what an AUC number cannot demonstrate.

Every other knob is carried verbatim from the parent deck. In particular:

* `policy_prior_random_floor = 0.20` — unchanged in B and C. The floor is a
  registered safety rail, not a tuning parameter; removing it would change the
  arm from "policy tilts the draw" to "policy owns the draw", which is a
  different experiment.
* `policy_prior_candidates = 16` — unchanged.
* **`policy_prior_temperature` is NOT shared between B and C.** B uses v1's
  0.25; C uses `policy_prior_temperature_v2 = 0.08`. This is not a free
  parameter: v1's output is a probability with gate-fold p90−p10 spread 0.573,
  v2's is a normalized clipped expected improvement with spread 0.189, and 0.08
  is the value that reproduces v1's ~10× sampling-odds ratio on v2's scale
  (`policy_v2_results_20260817.md` §8 item 3, `policy_v2_readout.py tau`). Left
  at 0.25 the v2 softmax is nearly uniform and arm C would be a second copy of
  arm A — a treatment arm that is its own control. Both values are fixed HERE
  and are not to be tuned after seeing an outcome.
* `policy_prior_strict = true` in **all three arms**, including `off`. Under
  `off` it is inert; in B and C it converts an unloadable checkpoint into a
  campaign-start hard error. An arm that silently degraded to random mutation
  would enter the readout as a policy arm and destroy the comparison — the
  fail-open confusion P1-05 names. §6 restates the check that closes this.

## 1a. Stage 0 — the free precursor (`shadow_v2`)

Before spending a single MASTER call on arm C, run the *next scheduled campaign
of any kind* with `policy_prior = "shadow_v2"`. In that mode the pool is built
**exactly as `off` builds it** — same rng sequence, same candidates, same order,
asserted by `tests/test_construct.py::test_shadow_mode_builds_the_off_pool_and_
only_records` — and v2 merely scores every elite child into
`selection.json → policy_shadow_scores`.

That costs nothing and answers two questions the A/B/C cannot ask cheaply:

1. **Does v2 rank the boards the wave actually verified?** Join
   `policy_shadow_scores[record_id]` against `results.json` outcomes. If v2's
   score has no association with the realised Δobjective on ~100 verified
   children, arm C has no mechanism and the A/B/C should be deferred, not run.
2. **What is the realised score spread on live parents?** τ = 0.08 is derived
   from the *gate fold's* spread. If live parents produce a materially different
   spread, τ is re-derived from the shadow data BEFORE registration — that is a
   legitimate pre-registration edit; re-deriving it after seeing arm C's
   outcome is not.

Stage 0 has no stopping rule and no claim attached. It is instrumentation.

---

## 2. Cell, objective and budget parity

* **Cell** `T6_T4 / feed 121`, library `paramA` — the current live operating
  point and the cell with the deepest elite pool, so the elite-mutation arm the
  policy modifies is actually load-bearing. (Parent deck:
  `fpcamp_minfxy_T6T4_f121_199.inp`.)
* **Objective** `min_fxy`: minimize **F_xy** (MASTER `FXYP`), hard gate
  `f_xy_limit = 1.65`, with `f_r_limit = 1.55`, `cbc_limit = 1600.0` and
  `minfxy_pin_bu_limit = 78.0` retained as constraints. `harvest_maps = true`
  (load-bearing — F_xy exists only in the final cycle's `MAS_OUT`).
* **Budget parity** — identical `budget`, `wave_size`, `pool_size` and wave
  count in every arm. The comparison is *fixed-call*: what did each arm buy with
  the same number of MASTER evaluations.
* **Same parents** — every arm starts from the same store elites and the same
  `random_seed`, so wave 0's parent set is identical by construction. Arms
  diverge only where the policy acts: the elite child chosen from a parent's
  proposed batch.

### Gating precondition: the F_xy head

The current champion has **no `predict_fxy` head**; acquisition falls back to the
interim proxy `F_xy ≈ 1.2176·F_r − 0.2519` with an inflated sigma, and every
selection is tagged `fxy_source = "proxy"`. Running the A/B/C on the proxy would
measure *how well each policy serves an F_r surrogate wearing an F_xy label*,
and the honest phrasing of any result would have to be "F_r-surrogate search
with F_xy measured after the fact" — not an F_xy policy comparison.

**Registered precondition: every arm's every wave must record
`fxy_source = "head"`. A run with any `"proxy"` wave is void and is not
reported as a result.**

---

## 3. Metrics

Primary and secondary are fixed here. AUC is **not** a metric of this
experiment — it is the offline number §6.2 explicitly rejects as the decision
basis.

| # | Metric | Definition | Direction |
|---|---|---|---|
| P1 | **best verified objective** | min MASTER-measured `F_xy` over all deliverable cores in the run, at a fixed call count (evaluated at 32, 64 and the full budget) | lower better |
| P2 | **first feasible call** | index of the first MASTER call returning a core passing every gated axis (F_xy ≤ 1.65, F_r ≤ 1.55, CBC ≤ 1600, pin BU ≤ 78 measured) | lower better |
| S1 | **feasible yield** | feasible cores / MASTER calls spent | higher better |
| S2 | **regret@k** | `min(F_xy over the k boards the arm chose) − min(F_xy over the k best boards in that wave's pool)`, k = wave_size = 8, averaged over waves. Requires the pool's realised labels, so it is computed post hoc over verified boards only and reported with its n. | lower better |
| S3 | **Pareto hypervolume** | 2-D HV of the verified (F_xy, cyclen) front against a reference point fixed BEFORE the run: (F_xy 1.65, cyclen 600 EFPD) | higher better |
| S4 | **violation rate** | fraction of MASTER-verified cores breaching ANY gated axis | lower better |
| D1 | *diagnostic* | pool diversity: mean pairwise Hamming among elite children, per wave | report only |
| D2 | *diagnostic* | non-convergence / NaN rate per arm | report only |

D1 exists because the credible way for a policy arm to lose is by collapsing the
neighbourhood the diversity quota protects, and a bare P1 loss would not
distinguish that from "the policy ranks badly".

**Pre-registered baseline for P1** (measured, this exact cell): the population
minimum is **F_xy 1.5491** (F_r 1.5018, cyclen 621.3). The F_r record core
(F_r 1.4797 @ 623.6 EFPD) measures F_xy 1.5829 and is *not* the F_xy optimum.
Any arm not beating 1.5491 is a null result for that arm and is published as one.

---

## 4. Replication and analysis

* **n = 3 independent seeds per arm** (`[flow] random_seed` ∈ {1111, 2222,
  3333}), the same three in every arm. One campaign is one draw; a single-seed
  A/B on this program has been wrong before, and the measured noise floor is not
  negligible.
* Analysis is **paired by seed**: the unit is (seed, arm), and the contrast is
  C−A and B−A within seed. Report the per-seed table in full — three paired
  differences is a small n and the raw numbers must be visible, not just a
  summary statistic.
* Uncertainty: BCa bootstrap over waves for P1/S1/S3 (2000 resamples, the
  `_boot_ci` used elsewhere in this program), exact per-seed values for P2.
* **No optional stopping.** The budget in §8 is spent in full in every arm
  before any comparison is computed. Reading a partial result and stopping is
  the failure this line exists to prevent.

---

## 5. Data recorded (already implemented)

Per wave, `runs/<run>/waves/wave_NN/selection.json` carries:

* `policy_mode` — the deck value verbatim (`off` / `v1` / `v2` / `shadow_v2`)
* `policy_version` — what actually loaded (`""` when nothing did)
* `policy_fallback` — `true` iff a policy was requested and could not be loaded
* `policy_shadow_scores` — `record_id -> [fr, flat]`, shadow mode only
* `fxy_source` — `head` / `proxy` (pre-existing)

plus the per-slot selection record (`record_id`, `parent_record_id`, `origin`,
`p_feas`, `acq`, `exploit`, `margin`, `pred_mean`) and, in `results.json`, the
MASTER outcome for each. Review §6.2 item 3 ("record the candidates all three
arms selected on the same parent pool") is satisfied by
`(parent_record_id, record_id)` per arm plus the shared seed.

---

## 6. Validity checks — run BEFORE any metric is computed

An arm failing any of these is void, not adjusted.

1. `policy_fallback == false` in **every** wave of **every** arm.
2. `policy_version == "v2"` in every wave of C, `"v1"` in every wave of B,
   `""` in every wave of A.
3. `fxy_source == "head"` in every wave of every arm (§2).
4. Wave-0 parent sets identical across arms (compare the `parent_record_id`
   multiset).
5. The v2 checkpoint that served C reports the schema stamp
   `policy_schema = policy_move_v2`, `era_libraries = [ga80, paramA]`, and its
   `corpus_sha256` matches §9. `MoveScorerV2` refuses to load anything else, so
   this is a paperwork check rather than a defence, but it is recorded.
6. Arm A's pool is byte-identical to a `policy_prior` absent config for the same
   seed (`tests/test_policy_prior.py::test_flag_off_pool_is_identical_to_a_
   config_without_the_knob` guards the code path; the run records the seed).

---

## 7. Decision rule — stated before the run

Let ΔP1 = (arm's best verified F_xy) − (control's), paired within seed.

* **ADOPT v2 as the default elite prior** iff C beats A on **P1 in all three
  seeds** *and* does not lose on P2 (first-feasible call no worse than A's
  median + 8 calls) *and* S4 does not increase.
* **ADOPT for proposal only (shadow stays default)** if C beats A on P1 in 2 of
  3 seeds, or wins P1 while losing P2 — the policy finds better cores but costs
  calls to do it. Re-run with a larger budget before defaulting it on.
* **RETIRE the serving arm** if C does not beat A on P1 in at least 2 of 3
  seeds. The wiring stays (it is the instrument), the deck default stays `off`,
  and the finding is published as a null. **A null here is a real result**: it
  says the campaign's performance comes from the surrogate acquisition, which is
  a defensible statement the program is currently unable to make either way.
* **Diagnostic override**: if C wins P1 while D1 (diversity) collapses by more
  than half against A, the win is attributed to greediness, not to learned
  rules, and is re-run at a higher τ before any adoption.
* B's result does not gate the decision; it calibrates it. B ≈ A is the expected
  and reassuring outcome. **B > A would be a surprise that invalidates the v1
  post-mortem's reading and must be investigated before C is adopted** — two
  policies with opposite offline verdicts both winning at campaign level would
  mean the campaign-level metric is measuring the mechanism, not the policy.

---

## 8. Cost — the one open decision

Per arm per seed: `budget = 64` MASTER calls (8 waves × wave_size 8) — a
reduction from the parent deck's 100, chosen so the design fits one fleet
allocation. Total **3 arms × 3 seeds × 64 = 576 MASTER calls**.

That is a large number and it is the reason this file is a draft. Alternatives,
in the order they should be considered if 576 is refused:

1. **Stage 0 only** (`shadow_v2`) — 0 extra calls, rides on scheduled campaigns.
   Buys the ranking association and the τ check, buys no causal claim. This is
   the recommended first step regardless.
2. **Drop arm B** — 2 arms × 3 seeds × 64 = 384 calls. Loses the calibration
   argument of §1; C > A then rests on the offline post-mortem being right about
   v1.
3. **2 seeds** — 3 × 2 × 64 = 384 calls. Weaker: n = 2 cannot support "all three
   seeds" in §7, and the decision rule would have to be rewritten before
   registration, not after.

Sequencing: Stage 0 rides on the F_xy campaigns already scheduled; the A/B/C is
scheduled only after the F_xy head lands and passes its own gate.

---

## 9. Fingerprints — to be filled at registration

| Item | Value |
|---|---|
| v2 checkpoint dir | `data/models/policy_v2` (Run B, protocol revB, 5 × `cnn_seed*`) |
| v2 `policy_schema` | `policy_move_v2` |
| v2 `corpus_sha256` | `fe53ac811131157f99ee8be9b078b5072e6895959756a1c8cbe3d956f8ff0f88` |
| v1 checkpoint dir | `data/models/policy_v1` (CNN arm only) |
| surrogate champion | *(pending — the first checkpoint with a `predict_fxy` head)* |
| store snapshot sha | *(pending)* |
| deck files | *(pending — one per arm, cloned from `fpcamp_minfxy_T6T4_f121_199.inp`)* |

## 10. Deck diff, per arm

Cloned from `fpcamp_minfxy_T6T4_f121_199.inp`; `[acquisition]` gains, and
nothing else changes:

```toml
# arm A (control)
policy_prior = "off"
policy_prior_strict = true

# arm B (v1)
policy_prior = "v1"
policy_prior_model_dir = "data/models/policy_v1"
policy_prior_temperature = 0.25
policy_prior_strict = true

# arm C (v2)
policy_prior = "v2"
policy_prior_model_dir_v2 = "data/models/policy_v2"
policy_prior_temperature_v2 = 0.08
policy_prior_strict = true
```

`policy_prior_random_floor` (0.20), `policy_prior_candidates` (16) and
`policy_prior_threads` (4) are left at their defaults in all arms.

---

## 11. Addendum (2026-08-29): a serve-side provenance defect, found and fixed *before* the A/B

Found while porting the surrogate's `featurize.serve_provenance` fix into the
policy path. **The A/B in sections 1-10 must not be launched on the pre-fix
serving code**: arms B and C would have proposed paramA moves through an
inverted conditioning global.

### What was wrong

`lpopt/policy/scorer.py::MoveScorer._board` derived `(dataset, sym_class)` from
`featurize.library_provenance`, the historical-extractor map. It predates the
store's `dataset="P"` rows and answers `paramA -> ("A", "rot61")`. But
`mine_policy_corpus` writes each step row's **real store `dataset`**, and the
corpus census is

| library | dataset | rows |
|---|---|---|
| 260624 | A | 20,825 |
| 5.8_5.1 | A | 941 |
| ga80 | B | 65 |
| ga80 | P | 3,865 |
| paramA | P | 2,401 |

so all 2,401 paramA rows trained at `g_dataset_flag = 1.0` while every paramA
proposal was **served at 0.0** — 1 of the 13 conditioning globals inverted, on
one of the two live libraries.

ga80 was unaffected: `"B"` and `"P"` are the same encoder input
(`g_dataset_flag = 0.0 if dataset == "A" else 1.0`), so the old map happened to
be right there.

### Train or serve? — *serve only, for `dataset`*

The corpus rows carry the store's real `dataset`, so **the corpus is correct on
that half and no re-mine is needed for it**. `sym_class` is the opposite case:
`build_pattern_cache` did *not* read it from the row, it called
`library_provenance(lib)[1]`, so every ga80 board in the cache was built at
`g_sym_class = 0.0` (`"free69"`) even though 3,865 of the 3,930 ga80 corpus rows
say `"rot61"` in the store. Train and serve were **consistently** wrong there —
the shipped `data/models/policy_v2` checkpoint learned against 0.0 — so serving
must keep feeding it 0.0. That half is a *corpus* defect, not a serving one.

### The fix

One function, `lpopt/policy/data.py::corpus_provenance(library_id)`, is now the
single definition, called by `build_pattern_cache` (train) and `_board` (serve)
so the two cannot drift: `dataset` from `featurize.serve_provenance` (the store
truth), `sym_class` from `library_provenance` (what the corpus actually used).
`build_pattern_cache` additionally raises if any library's rows disagree with the
derived `dataset` on A/not-A, so a future mixed library fails loudly rather than
training on a featurization the serve path cannot reconstruct. The shipped
`policy_v2` checkpoint stays valid and needs no retrain.

### Measured effect

Serving re-scored against `data/models/policy_v2/probs.npz` on `gate_cur` rows
of the checkpoints' own corpus snapshot (`corpus_sha256`-selected
`steps.parquet.bak_pre_fpcamp_minfr_triple_f125`), max |ΔP(improve)|:

| library | before | after |
|---|---|---|
| ga80 (25 rows) | 5.4e-05 | 5.4e-05 |
| paramA (25 rows) | **0.087** | 2.2e-04 |

The residual ~2e-4 is the pre-existing float16 pattern-cache rounding (the cache
stores slots at float16; `build_pattern_cache` reduces the full-precision slots
into `g_fresh_mean_*` while `_board` reduces the rounded ones), not provenance.

### Why the existing parity test did not catch it

`test_v2_serving_reproduces_the_training_run_scores` samples `gate_cur` rows
0-3, which are **all ga80** — the library the old map got right. It passed
throughout. `gate_cur` is 538 ga80-P + 65 ga80-B + 403 paramA rows; the paramA
rows were off by up to 0.087, seventeen times the test's 5e-3 tolerance.

Three gates added to `tests/test_policy_prior.py` (§6b), all failing on the
pre-fix code and passing now:

* `test_corpus_provenance_reproduces_every_corpus_rows_dataset`
* `test_policy_serve_row_featurization_parity` — the policy analogue of
  `tests/test_model_api.py::test_serve_row_featurization_parity`; asserts slots
  and every non-`g_fresh_mean_*` global byte-equal between `_board` and
  `build_pattern_cache`, over both live libraries.
* `test_v2_serving_reproduces_the_training_probabilities_on_both_libraries` —
  the score-level gate, 8 `gate_cur` rows of *each* library.

### Carried to v3 (blocking)

`data/policy/steps.parquet` must be **re-mined with true provenance** —
`sym_class` read from the store row the way `dataset` already is — and the
pattern cache rebuilt, before a v3 policy is trained. At that point
`corpus_provenance` collapses into `featurize.serve_provenance` and can be
deleted. Until then, changing it invalidates `data/models/policy_v2`.

### Effect on this pre-registration

Sections 1-10 are unchanged and stand. The only amendment: **arms B and C run
the fixed serving code**; a paramA-carrying wave launched on the pre-fix code is
not a valid instance of arm B/C and must be discarded. Neither the shipped v1
nor the shipped v2 checkpoint is retrained.
