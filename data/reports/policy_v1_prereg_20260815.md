# Policy net v1 — pre-registration

Written 2026-08-15, **before any weight was trained**. Corpus:
`data/policy/steps.parquet` (SHA-256 stamped into every checkpoint's `meta.json`
and into `metrics.json`), built by `mine_policy_corpus.py` and characterised in
`data/reports/policy_corpus_20260815.md`. The corpus is read-only for this round.

Code: `lpopt/policy/{data,net,train}.py`. Launcher: `train_policy_v1.py`.
Results go to `data/reports/policy_v1_results_20260815.md` — this document is
not edited after training, and every deviation is reported there.

---

## 1. What is being learned

A **move scorer**, not a surrogate. Given

* the **parent** loading pattern,
* a **candidate move** (the edit that produces the child pattern),
* the **cell** (case_pair, feed, library),

emit two probabilities:

| head | label column | question |
|---|---|---|
| `fr` | `improved_fr` | does the child's F_r beat the parent's? |
| `flat` | `improved_flat` | does the child's node_peak beat the parent's? |

**The leakage-arbitration head (cyclen / CBC) is out of scope.** Corpus section
4g shows the two lineage eras disagree on the *sign* of the cycle-length
response to outward fresh loading (`lpopt_genome` outward `d_cyclen` −2.21,
`sa_mocha` outward `d_cyclen` +0.14), and the observational corpus cannot
de-confound move class from radial direction. The queued 1-move ablation wave
settles it; until then a leakage head would be fitting an artifact. This is a
scope decision, not an oversight, and v1 must not be read as evidence about
leakage in either direction.

## 2. Universe

`cross_cell == False` **and** at least one head labelled.

* cross-cell edges (feed morphs, pair transfers, MOCHA fuel-family repaints) are
  not actions in a fixed move space, so they are not moves the policy can propose;
* a row with neither label carries no supervision.

**n = 19,726** rows. `move_class == 'sa_unknown'` (923 rows) is **kept**: the
class is an input feature, not a target, and dropping it would silently delete
the MOCHA compound primitives the live search actually plays.

## 3. Splits — pre-registered, deterministic in `--base-seed 20260815`

### 3a. Grouping: lineage connected components, not rows

A random row split puts a board's siblings on both sides of the wall; the parent
tensor is then memorisable and the held-out number is inflated. The split unit
is therefore the **connected component of the (parent_record_id,
child_record_id) lineage graph within a cell**. No board reachable from a
training board can appear in test.

Cost: components are lumpy (corpus section 7 — `B1_C2/f121/260624` is 47% one
component), so a greedy pack that only takes whole components under target
undershoots the nominal 10%. The realized fractions below are the pre-registered
ones; they are what the gate is evaluated on.

### 3b. Three held-out families, removed before any random draw

| fold | rule | n | cells | base `fr` | base `flat` |
|---|---|---|---|---|---|
| `heldout_era` | `library_id ∈ {ga80, paramA}` | 1,305 (fr) / 870 (flat) | 54 | 0.188 | 0.128 |
| `heldout_cell` | `cell == B1_C6/f121/260624` | 785 | 1 | 0.439 | 0.438 |
| `heldout_lib` | `library_id == 5.8_5.1` | 772 | 5 | 0.295 | 0.288 |

* **`heldout_era` is the decision-relevant readout.** These 1,399 rows are
  exactly the `lpopt_genome` era — the live operating point (ga80/paramA, feeds
  101-141, near F_r = 1.55). Corpus section 10 point 2 warns that a random
  split would not test the extrapolation that deployment requires. Holding the
  whole era out costs 7% of the training rows and is the only honest test of
  "will this help the campaign that is actually running". Its base rates are
  low (0.19 / 0.13) so AUC is the reliable statistic there; precision@32 is
  reported but will be noisy.
* **`heldout_cell` is the brief's mid-size unseen cell** — 785 steps, near-balanced
  labels, node_peak present on every row, same library as the training mass.
  It is the only fold large enough to run the deployment metric *inside a single
  cell*, which is what deployment actually looks like.
* **`heldout_lib` is a whole unseen library** inside the SA era.

### 3c. Random split of the remainder

Within each remaining cell, whole components are drawn for `test` (nominal 10%),
then for `val` (nominal 10% of what is left). **`val` is used for early stopping
and nothing else.** It never enters the gate.

| fold | n | cells | n(fr) | base fr | n(flat) | base flat |
|---|---|---|---|---|---|---|
| train | 13,845 | 26 | 13,845 | 0.349 | 11,752 | 0.375 |
| val | 1,404 | 11 | 1,404 | 0.372 | 1,205 | 0.374 |
| test | 1,615 | 13 | 1,615 | 0.389 | 1,387 | 0.394 |
| heldout_cell | 785 | 1 | 785 | 0.439 | 785 | 0.438 |
| heldout_lib | 772 | 5 | 772 | 0.295 | 772 | 0.288 |
| heldout_era | 1,305 | 54 | 1,305 | 0.188 | 870 | 0.128 |

**Noted in advance:** `test` carries a higher base rate (0.389) than `train`
(0.349). That is the component split, not a bug — the whole components that fit
under 10% skew toward the smaller, easier cells. AUC is base-rate invariant;
precision@32 is compared against baselines *on the same rows*, so the comparison
stays fair, but the absolute precision@32 numbers are not comparable across folds.

## 4. Metrics

### 4a. Ranking — AUC per head

ROC AUC with mean ranks for ties, 95% CI from 2,000 row-resample bootstraps.

### 4b. DEPLOYMENT metric — precision@32 out of a 256-candidate batch

Draw 256 labelled rows without replacement from the evaluation fold, rank by
score, take the top 32, report the fraction that actually improved. 2,000
replicates; **all scorers see the identical draws and an identical per-replicate
random tiebreak**, so every comparison is paired and the frequency baselines
(which have only a handful of distinct values) are broken uniformly at random
rather than by array order.

**Deviation from the brief, declared here.** The brief asked for 256 candidates
*per parent*. The corpus cannot supply that: the busiest parent has 42 children,
exactly one parent has ≥ 32, sixteen have ≥ 16, and the median is 5. A batch is
therefore drawn from the evaluation fold rather than from one parent. This is
the honest analogue of `construct.build_pool`, which scores a pool assembled
from many parents at once — but it does leave a confound: a scorer could win by
recognising *easy parents* rather than *good moves*. Section 4c is the control
for exactly that, and both are reported side by side.

### 4c. Parent-blocked AUC — the strict generator-prior question

Over all (improving, non-improving) pairs **sharing a parent**: what fraction
does the policy order correctly? Ties count half. This differences out parent
difficulty entirely and needs no minimum-children threshold, so it uses every
parent with mixed labels. It is the number that says whether the policy can pick
the good move *from this board*.

### 4d. Calibration

Brier score and ECE over 10 equal-width bins, plus the reliability table.
**Reported, not gated** — v1 is a ranker; calibration matters for the later
generator-prior temperature, not for this gate.

## 5. Baselines — the policy must beat all three

Each is fitted on `train` only, exactly like the policy.

| id | score | why |
|---|---|---|
| `random` | uniform noise | the floor; its precision@32 is the fold's base rate |
| `class_freq` | train improving-rate of the row's `move_class` | corpus table 4a as a scorer — "just always play `rewire_swap`" |
| `periph` | `d_fresh_share_periph` | the engineer's rule of thumb as a continuous ranker — "push fresh outward" (corpus 4b: ρ = −0.131 vs `d_f_r`) |

**Advisory, reported but NOT gated:** `cell_class_freq`, the train improving-rate
of (cell, move_class) backing off to the class rate. It is a materially stronger
baseline than the three the brief named, and folding it into the gate after the
fact would be moving the goalposts. If the policy loses to it, that is stated
plainly in the results as a finding about how much of the signal is a lookup table.

## 6. Architecture and the ONE pre-registered comparison

Both arms share the input, the loss, the split, the schedule and the seeds.

**Arm `cnn` (1,690,114 params).** The PosValNet trunk with the heads swapped —
`FiLM` and `ResidualBlock` are imported from `lpopt/model/net.py`, not
re-implemented. Conv stem → 6 residual blocks (width 112) → FiLM(cond) after
blocks 1, 3, 5 → masked mean+max pool over the fuel cells → MLP → 2 logits.

Input tensor, 101 channels on the 19×19 mirror grid:

| block | n | content |
|---|---|---|
| parent | 58 | the full `cond_v6b` featurizer channels of the PARENT board |
| delta | 42 | `child − parent` on every channel that ever differs (selected on TRAIN only; the other 16 are pure geometry and are identically zero) |
| mask | 1 | slots whose encoding changed |

Conditioning vector, 49 dims: the 13 `cond_v6b` board globals of the parent +
36 move/context scalars (move-class one-hot ×8, `n_unit_edits`,
`n_slots_changed`, `single_move`, `swap_span`/`swap_radius` + presence flags,
the 7 `d_*` ring deltas, the 7 `parent_*` ring descriptors, `fresh_radial_dir`
and `burnt_periph_dir` one-hots, centred feed).

**Arm `mlp` (144,898 params).** The same conditioning vector, no board tensor,
3×256 MLP. This is the control: the convolutional apparatus must earn its place
against a scalar model costing a tenth of the parameters. **If the two tie, the
recommendation will be to ship the MLP.**

### 6a. The s1e transfer-init comparison is DROPPED — and why

The brief asked for a scratch-vs-s1e-encoder comparison. It is dropped for two
independent reasons, both fatal:

1. **No matched geometry exists.** The s1e champion trunk is width 224 /
   8 blocks / 10,363,715 parameters — 6× the budget the brief sets for this task
   and far too large for 13,845 labelled rows. Transferring it would confound
   "did the init help" with "did 6× capacity help", and a matched-geometry
   transfer would require first training a width-112 surrogate, which is a
   separate round.
2. **It would contaminate the held-out reading.** s1e was trained with direct
   supervision on `f_r` and `node_peak` of the store records that *are* this
   corpus's parents and children. An s1e-initialised or s1e-featurized policy
   would carry indirect access to held-out outcome labels, and every number in
   section 4 would be unreadable.

Reason 2 is the decisive one and it applies to any warm start from any surrogate
trained on the same store. The clean version for a later round: train a
width-112 surrogate on a split that *excludes* the policy's test and held-out
cells, then run the comparison. `cnn` vs `mlp` is the pre-registered comparison
in its place.

## 7. Training protocol

Fixed in advance; no sweep.

| knob | value |
|---|---|
| loss | BCE-with-logits, **masked** where a head's label is absent |
| head weighting | equal (base rates 0.35/0.38 are near-balanced; no `pos_weight`) |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, grad-clip 5.0 |
| schedule | cosine to 120 epochs |
| batch | 256 |
| early stop | mean val AUC over the two heads, patience 15 |
| augmentation | diagonal-mirror transpose, p = 0.5, **train only** |
| seeds | 5 per arm (20260815 … 20260819) |
| headline model | the 5-seed probability-mean ensemble; per-seed AUC spread also reported |
| no TTA | evaluation is deterministic, single orientation |

**Transpose augmentation** uses the on-the-fly recipe validated by
`mine_policy_corpus.py::verify_transpose` (0 violations in 1,500): both patterns
map to their mirror, every other column copies verbatim. It is implemented as an
index swap into a pre-built pattern cache, so no encoder call happens at train
time and nothing is materialised on disk. The claim that the 13 board globals
are transpose-invariant was **checked numerically, not assumed** (max abs
difference 3.6e-7 over the cache, i.e. float32 rounding).

## 8. Leakage discipline

The model may see anything computable from the parent board, the candidate edit
and the cell. `lpopt/policy/data.py::FORBIDDEN_COLUMNS` names every column that
would break that and `scalar_features` raises if one reaches the feature frame.

Excluded outcomes: all `child_*`, all `d_*` FOMs, all `improved_*`, `feasible_*`,
`*_converged`, `in_cyclen_band_child`, `map_cov`.

Two further exclusions are decisions, not leakage, and are declared so they
cannot be quietly reversed:

* **Provenance** — `lineage_source`, `campaign`, `generator`, `sa_accepted`,
  `source_move`, `single_move_evidence`. A move the policy invents has no
  provenance, so a model leaning on it would not transfer. `lineage_source` is
  additionally collinear with `heldout_era`, which would make that readout
  meaningless.
* **Parent FOMs** — `parent_f_r`, `parent_node_peak`, etc. These *are*
  legitimately available at proposal time, but they let a scorer win on "this
  parent is bad, anything helps" rather than on move quality — the exact
  confound section 4b already carries. A v2 may add the *surrogate-predicted*
  parent FOM, which is also available for an unevaluated parent.

The `d_*` ring descriptors ARE included. They are functions of the child
*pattern*, which the move determines, and are known before any evaluation — the
same information the `periph` baseline uses. That makes the baseline a strict
subset of the policy's input, which is the intended, fair form of that test.

## 9. GATE

Evaluated on **`test`** (in-distribution, component-grouped), on the 5-seed
ensemble, for **arm `cnn`** and separately for arm `mlp`.

**PASS requires, for BOTH heads:**

1. AUC 95% bootstrap CI **lower bound > 0.50**, and
2. mean precision@32 **greater than** `random`, `class_freq` and `periph`, with
   the **paired** bootstrap 95% CI on the difference **excluding 0** for all three.

**FAIL otherwise.** No partial credit, no post-hoc metric substitution. A single
head passing is reported as a single head passing.

Secondary readouts — reported in full, **not** part of the gate: the three
held-out families, the parent-blocked AUC, calibration, the advisory
`cell_class_freq` baseline, and the `cnn`-vs-`mlp` comparison.

## 10. Falsification reading — what a FAIL means

Spelled out now so it cannot be rationalised later.

* **Both heads fail the AUC floor.** There is no learnable move-quality signal
  at this corpus size and representation. The corpus is observational and its
  move distribution is unbalanced (corpus 4g); the answer is the 1-move ablation
  wave, not a bigger network. **Recommendation would be: more data, no A/B wiring.**
* **AUC clears but precision@32 does not beat `periph`.** The net has learned
  the radial rule of thumb and nothing beyond it. Ship the one-line heuristic as
  the generator prior and close the policy line until the ablation wave lands.
  This is a real possible outcome — corpus 4b measures ρ = −0.131 for
  `d_fresh_share_periph` vs `d_f_r`, which is not nothing.
* **Precision@32 beats all three but parent-blocked AUC ≈ 0.5.** The policy is
  ranking *parents*, not *moves*. It would still help `build_pool` prioritise
  which parents to expand, but it is not a move proposer and must not be wired
  in as one.
* **`test` passes but `heldout_era` AUC ≈ 0.5.** The policy works on the 260624
  SA corpus and does not transfer to the live ga80/paramA operating point. It
  must not be wired into the running campaign. **This is the most likely
  partial-failure mode** and the reason the era holdout exists.
* **`mlp` matches `cnn`.** The board tensor is not carrying information the
  scalars lack. Ship the MLP; the CNN is 12× the parameters for nothing.

## 11. Compute

238 (`USER@HOST_238:8022`), venv `~/lpopt_ws/venv` (python 3.11.13, torch
2.11.0+cu128). Occupancy checked before launch: both RTX PRO 6000 at 0%
utilisation, 239 MiB / 1 MiB of 97,887 MiB used, no tmux sessions, s1e `DONE`.
Pinned to GPU 1 via `lpopt_gpu1.inp`, leaving GPU 0 (the default deck's target)
free. 199 and the four local MASTER regen processes are untouched.

The run follows the established marker convention (`heartbeat` every 15 s, `rc`,
`DONE`/`FAILED`) so `lpopt remote status --ts policy_v1` and
`lpopt remote pull --ts policy_v1` work unchanged; only the launch script's
module name differs, because `lpopt/remote.py`'s run template hardcodes
`-m lpopt.model.train`.

10 runs (2 arms × 5 seeds) at ~1.7M params over 13,845 rows: minutes each.
