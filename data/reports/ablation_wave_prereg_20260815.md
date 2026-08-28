# 1-move ablation wave — PRE-REGISTRATION

**Written 2026-08-15, BEFORE any MASTER chain was launched on box 199.** The
plan manifest, the parent set, the stratified sample and the blind policy-v1
predictions all exist on disk and are hashed in §8 at the time of writing. No
label from this wave exists yet.

Registered by: `data/reports/ab2_addendum_S1E_20260815.md` §8 ("Next registered
moves: (a) 1-move ablation wave on this cell"), `policy_corpus_20260815.md` §10
("Run a dedicated 1-move ablation wave... Stratify that wave by radial
direction"), and `policy_v1_results_20260815.md` §8 option 2.

---

## 1. The two questions, and why observational data cannot answer either

### 1a. Leakage arbitration

`policy_corpus_20260815.md` §4b reports that moving fresh fuel outward does
**not** cost cycle length (`d_fresh_share_periph` vs `d_cyclen` = **+0.093**),
retracting an earlier opposite finding. §4g shows the two eras disagree on the
sign outright:

| lineage_source | outward `d_cyclen` | inward `d_cyclen` |
|---|---:|---:|
| `lpopt_genome` | **−2.2099** | −1.0869 |
| `sa_mocha` | **+0.1362** | −0.2455 |

The corpus itself states why this is unresolvable: *"neither era sampled
direction at fixed move class."* The confound has a hard structural cause, which
this wave makes explicit (§3a): **`rewire_swap` cannot have a radial direction at
all.** It leaves the fresh set and the batch labels untouched, so
`d_fresh_enr_r_center` is *identically* zero. `rewire_swap` is 10,536 of 19,820
same-cell corpus moves — so the corpus's "neutral" row **is** its rewire row, and
"direction" and "class" are not merely correlated in the corpus, they are partly
the same variable. Only an intervention that fixes the class and varies the
direction can separate them.

### 1b. Era gap

`policy_v1` passes its gate on `test` but fails all three baselines on
`heldout_era` (ga80 + paramA, the live operating point): AUC 0.650/0.682, p@32
0.328. 93% of its corpus is SA-era 260624. This wave produces current-era
(paramA) single-move labels at the live cell, which is what a v1.1 fine-tune
needs. **No training happens in this task.**

---

## 2. Cell, parents, budget

**Cell** `T6_T4/f121/paramA` — native-band restarts, 1,070 store rows, 918
converged, 343 feasible, in-band record **F_r 1.4749** (r8, campaign
`fpcamp_minfr_T6T4_r8`). The cell was declared at its floor by the S1E §8
close-out; this wave is not a further chase of that floor, it is an
instrumentation run on the boards the chase produced.

**Parents — 10, selected deterministically (no RNG).** Feasible + converged rows
of the cell (`mine_policy_corpus.feasibility`: F_r ≤ 1.55, F_q ≤ 2.41, CBC ≤ 1600,
|AO| ≤ 0.30, converged), taken in an interleaved priority order — top by F_r, top
by `node_peak`, mid-band (F_r 40–60th percentile) — with a **minimum pairwise
69-slot Hamming distance of 12** enforced greedily.

| record_id | family | campaign | F_r | node_peak | cyclen | CBC | fresh_enr_r_center |
|---|---|---|---:|---:|---:|---:|---:|
| `188c9a338d9f` | top_fr | r8 | **1.4749** | 1.2957 | 625.459 | 1355.76 | 6.3179 |
| `9b9fabe89cc1` | top_fr | r6 | 1.4797 | 1.3034 | 623.592 | 1338.61 | 6.3179 |
| `061b17a666cf` | top_fr | r5 | 1.4812 | 1.3033 | 618.626 | 1294.79 | 6.3604 |
| `a1c6984e161e` | top_fr | r8 | 1.4815 | 1.3100 | 623.893 | 1337.53 | 6.3179 |
| `1165441c31ea` | top_flat | r4 | 1.4877 | **1.2757** | 618.815 | 1308.12 | 6.2589 |
| `4a96217c80ee` | top_flat | r6 | 1.4950 | 1.2836 | 623.233 | 1334.60 | 6.3179 |
| `0b0facaeafcd` | top_flat | r7 | 1.5110 | 1.2793 | 623.026 | 1331.54 | 6.3179 |
| `ef22f67686de` | mid_band | r7 | 1.5019 | 1.3178 | 623.323 | 1335.66 | 6.3179 |
| `5ee81be08aad` | mid_band | r7 | 1.5019 | 1.3083 | 623.015 | 1331.78 | 6.3179 |
| `c533a3db5113` | mid_band | r8 | 1.5021 | 1.3286 | 622.881 | 1332.30 | 6.3179 |

Achieved pairwise Hamming: **min 13, median 25, max 39.**

**Budget.** 150 paid MASTER equilibrium chains; registered hard cap **160**
including reruns. The runner refuses to launch a plan whose paid count exceeds
the cap.

---

## 3. The stratification — and the four cells that are structurally empty

### 3a. Reachability is a physical fact, not a sampling choice

For each parent, `ablation_wave.py` enumerates the **complete** verified
single-move neighbourhood — every `rewire_swap`, every `fresh_relocate`, every
`batch_flip`, every `batch_swap` that `lpopt/search/genome.py`'s own operators
can produce, each passed through `validate()` and then **re-classified by
`mine_policy_corpus.classify_move`**, which is the authority: a candidate whose
net diff does not read back as the intended class, or which exceeds that class's
`SINGLE_MOVE_MAX_EDITS` bound, is dropped rather than relabelled. Every one of
the 15,901 enumerated children passed both checks.

The neighbourhood census over the 10 parents (n available, summed):

| move_class | inward | neutral | outward |
|---|---:|---:|---:|
| `rewire_swap` | **0** | 4,350 | **0** |
| `fresh_relocate` | 5,631 | 100 | 3,269 |
| `batch_swap` | 1,991 | 27 | 223 |
| `batch_flip` | 256 | **0** | 54 |

The brief anticipated "~15 strata". **Only 8 of the 12 (class × direction) cells
are non-empty, and only 7 are usable**, for reasons that are structural, not
sampling artefacts:

* `rewire_swap` × {outward, inward} — **identically empty.** A source exchange
  between two burned units touches neither the fresh set nor the batch labels, so
  the enrichment-weighted fresh radial centre is *exactly* unchanged. This is not
  a rare event; it is a theorem about the operator.
* `batch_flip` × neutral — empty. Repainting one fresh unit (or the centre)
  always perturbs the weighted centroid.
* `fresh_relocate` × neutral (100) and `batch_swap` × neutral (27) — measure-zero
  radius coincidences. **Excluded by design**: they are not "no radial move",
  they are "two units of coincidentally equal radius", which is a different
  physical statement and would pollute the neutral reference.

This is registered now so that the reduced stratum count reads as a discovered
constraint, not as a budget cut.

### 3b. The three instruments are not interchangeable — dose and reactivity

Two covariates separate them, both computed per candidate before launch:

* **dose** = `|d_fresh_enr_r_center|`, the magnitude of the radial intervention;
* **`d_fresh_enr_mass`** = change in multiplicity-weighted *total* fresh
  enrichment. `fresh_enr_r_center` is a *normalized* first moment and is blind to
  the total, so this is what says whether a move redistributed reactivity or
  changed how much there is.

Measured over the drawn sample:

| move_class | dose (median) | dose (max) | max abs `d_fresh_enr_mass` | role |
|---|---:|---:|---:|---|
| `fresh_relocate` | 0.052–0.076 | 0.260 | **0.000000** | **PRIMARY** leakage instrument: high dose, reactivity-conserving |
| `batch_swap` | 0.002–0.007 | 0.015 | **0.000000** | **SECONDARY**: mid dose, reactivity-conserving |
| `batch_flip` | 0.00001 | 0.00013 | **1.2017** | **NOT a direction instrument** — reactivity control |
| `rewire_swap` | 0.000000 | 0.000000 | 0.000000 | pure burnt-inventory reference (the corpus's "neutral") |

`batch_flip`'s radial dose is ~5,000× weaker than `fresh_relocate`'s while it is
the *only* class that changes the total fresh reactivity. Its outward/inward
label is therefore a near-degenerate radial contrast riding on a large reactivity
change. **Registered now: `batch_flip` is read as the reactivity-change control,
and its direction contrast is NOT admissible evidence about leakage.** Reading it
as a leakage result after the fact would be the error this paragraph exists to
prevent.

### 3c. Allocation

Per parent, 15 children; × 10 parents = **150**:

| stratum | per parent | total |
|---|---:|---:|
| `rewire_swap` × neutral | 3 | 30 |
| `fresh_relocate` × outward | 3 | 30 |
| `fresh_relocate` × inward | 3 | 30 |
| `batch_swap` × outward | 2 | 20 |
| `batch_swap` × inward | 2 | 20 |
| `batch_flip` × outward | 1 | 10 |
| `batch_flip` × inward | 1 | 10 |

Zero shortfalls: every stratum had more candidates available than requested.

**Within-stratum draw is rank-deterministic, not random.** Candidates are sorted
by dose (by `swap_span` for the zero-dose `rewire_swap` stratum) and `k` evenly
spaced ranks are taken. Consequences, both intended: each stratum spans its own
full dose range, so the direction contrast is simultaneously a **dose-response**
readout; and outward and inward samples of the same class occupy the **same
quantile positions**, which is what makes them dose-matched within parent.

---

## 4. Dedup — the free/paid split is 0/150, and that is a result

Every enumerated child's `record_id` is minted with the store's own
`compute_record_id(canonical, "paramA", "T6_T4", "ga80_produce")` and checked
against the canonical store. **0 of 15,901 enumerated children are already
labelled.** Only 8 store rows anywhere carry one of the 10 parents as
`parent_record_id`, and **none of them is a verified single move** (4
`fresh_relocate`, 2 `multi`, 2 `rewire_multi`, all above their single-move edit
bound).

This is not a defect of the dedup; it is a measurement of the corpus. The
campaigns mutate at `n_moves_early = 2` / `n_moves_late = 5`, and the one
1-move generator (`[search.local_search] n_moves = 1`) proposes on the surrogate
— corpus §6 records 1,990 such parents whose edges were never MASTER-evaluated.
**The store contains essentially no verified single-move edges from its own
elites.** Registered consequence: the free/paid split is expected to be 0/150 and
the wave buys genuinely new information rather than re-deriving it.

---

## 5. What runs

150 full MASTER equilibrium chains, one per child, on box 199
(`USER@HOST_199`, kit `C:/Users/USER/lpopt_work/kit_frontier`, 24 logical
cores, confirmed idle: `master=0`, `lpopt_py=0`). Harness:
`lpopt.search.verify.WaveVerifier` — the same verifier the campaigns and
`fr_transfer.py` use — with `harvest_maps=True`, `max_cycles=16`,
`consecutive=2`, `workers=16`, `host_reserve=1`, waves of 8.

Assets resolve natively (`T6_T4/f121` on the paramA package,
`data/design/package`); the runner **refuses to launch at any
`fallback_level != 0`** without an explicit override, because a fallback restart
carries a foreign burnt-fuel history into every chain and would confound every
parent→child delta. Restart and template-deck sha256 are printed into the log
header.

Outcomes become full store rows via `outcome_to_record` (`campaign =
ablation_1move_T6T4`, `generator = ablation_1move`, `stratum =
<move_class>|<direction>`, `parent_record_id` = the parent elite), written to a
**scoped** merge kit (~150 rows) rather than shipping the box's whole store back.
So the labels are also training data, and the lineage edges are first-class
store lineage that `mine_policy_corpus.lineage_edges` reads natively.

---

## 6. Registered analyses

### 6a. PRIMARY — interventional leakage arbitration

For each stratum, the improving fraction for F_r ↓, `node_peak` ↓, `cbc_max` ↓
and `cyclen` ↑, plus mean `d_*`, computed exactly as
`mine_policy_corpus._improving_table` / `_multi_objective_table` do.

**The registered test** is the outward-vs-inward contrast **at fixed move class,
paired within parent**, on the two reactivity-conserving instruments:

1. `fresh_relocate` (n = 30 vs 30) — primary, high dose;
2. `batch_swap` (n = 20 vs 20) — secondary, mid dose.

Statistic: within-parent paired difference in mean `d_cyclen` (outward − inward),
with a sign test across the 10 parents and a bootstrap CI over parents. The
dose-response check is the slope of `d_cyclen` on `d_fresh_enr_r_center` pooled
within class.

**Registered readings, fixed now:**

* **outward `d_cyclen` < inward, both instruments, CI excluding 0** → the leakage
  half of the rule of thumb is **confirmed at this era**; the corpus §4b
  retraction stands as a description of the observational mix, and the
  `lpopt_genome` sign in §4g is the causal one.
* **outward `d_cyclen` > inward, both instruments, CI excluding 0** → the
  `sa_mocha` sign is the causal one; outward loading is *free* at this operating
  point and the flattening head can be pushed outward without a cycle bill.
* **CI includes 0 on both** → the effect is smaller than 150 chains can resolve.
  This is a real answer and is reported as one: it bounds the effect rather than
  finding it, and the bound is the deliverable.
* **The two instruments disagree in sign** → dose-dependence or an instrument
  artefact; reported as unresolved, and the `fresh_relocate` (high-dose) reading
  is *not* promoted over the `batch_swap` one on the strength of its n alone.

`rewire_swap` × neutral is the reference arm: it fixes the "what does a move of
this cell do to cyclen when it touches no fresh fuel at all" baseline that both
direction arms are read against. `batch_flip` is reported but, per §3b, is
**not** admissible for the leakage verdict.

### 6b. SECONDARY — era-gap corpus rows

The new edges are appended to `data/policy/steps.parquet` with
`lineage_source = 'ablation_paramA'`, all 77 columns filled by
`mine_policy_corpus.build_steps` itself (never re-derived) so the rows are
schema-identical to the mined corpus. Registered as a deliverable, not a
hypothesis. No training in this task.

### 6c. PROSPECTIVE — policy v1 on the era it failed

All 150 candidate moves were scored by the `data/models/policy_v1` 5-member CNN
ensemble **before any label existed** (`data/design/ablation_wave_policy_v1_pred.csv`,
sha256 in §8). Metrics, on the era fold where v1 failed:

* **AUC** per head (`fr`, `flat`) over all labelled children;
* **parent-blocked AUC** — the validated readout (`policy_v1_results` §1); scores
  from different parents are not comparable, so the within-parent rank
  (`rank_fr_in_parent`) is carried in the prediction file;
* **precision@32** out of the 150, against the three pre-registered baselines
  (`random`, `class_freq`, `periph`) from `policy_v1_prereg` §5.

**The blind predictions, registered now:**

| move_class | direction | mean predicted P(improve F_r) | mean P(improve flat) |
|---|---|---:|---:|
| `batch_flip` | outward | **0.814** | **0.852** |
| `batch_flip` | inward | **0.173** | **0.163** |
| `batch_swap` | inward | 0.575 | 0.601 |
| `batch_swap` | outward | 0.353 | 0.394 |
| `rewire_swap` | neutral | 0.576 | 0.598 |
| `fresh_relocate` | outward | 0.468 | 0.554 |
| `fresh_relocate` | inward | 0.390 | 0.459 |

Overall spread: `p_improve_fr` 0.001–0.917 (mean 0.476), `p_improve_flat`
0.001–0.953 (mean 0.523).

**Falsification reading — fixed before the truth lands:**

* The model's single largest commitment is on `batch_flip`: a **0.64** gap
  between outward and inward. Per §3b that is the class whose radial dose is
  ~1e-5 and whose *reactivity* change is up to 1.20. **If that gap materialises,
  the model is reading reactivity, not radius**, and its apparent "radial rule"
  is a proxy — a finding that would change how the v1.1 features are framed. If
  the gap does not materialise, this is the model's largest error and it is
  concentrated where its input signal is weakest.
* On `fresh_relocate` — the *high-dose*, genuinely radial class — the model
  predicts only a **0.078** outward advantage, where the observational corpus
  (§4d, this pair's own cell family) reports 0.328 vs 0.144. **If the measured
  outward advantage on `fresh_relocate` is large, v1 under-uses the one axis the
  corpus says is real.**
* On `batch_swap` the model predicts **inward > outward** (0.575 vs 0.353),
  inverting the corpus rule of thumb. Either the model has found a real
  dose-dependent inversion, or it is wrong here; the wave decides.
* **Overall gate for the prospective test:** v1 is judged to have *transferred*
  to this era iff parent-blocked AUC ≥ 0.65 on `fr` **and** p@32 beats all three
  baselines. Anything less is recorded as a confirmed era failure — which is the
  expected outcome given `heldout_era` p@32 = 0.328, and is precisely the
  motivation for the v1.1 fine-tune this wave feeds.

This is a genuinely prospective test: the predictions are hashed and on disk
before a single chain runs.

---

## 7. What this wave cannot conclude

* **One cell, one pair, one feed, one library.** Every result is about
  `T6_T4/f121/paramA` near F_r ≈ 1.48–1.50. It does not license a claim about
  ga80, about other feeds, or about boards far from this operating point.
* **Ten parents, all near the cell floor.** The parents are elites; the
  neighbourhood of an elite is not the neighbourhood of an average board, and an
  improving fraction measured here will be *lower* than one measured from a
  mediocre parent for purely regression-to-the-mean reasons. The direction
  contrast is paired within parent precisely so that this bias cancels; the
  absolute improving fractions are **not** comparable to corpus §4a.
* **`batch_flip` cannot speak to leakage** (§3b), and `rewire_swap` cannot speak
  to direction at all (§3a).
* **No causal claim about the corpus eras.** This measures the causal sign *at
  this operating point*. It cannot retro-fit a cause onto the `sa_mocha` era's
  260624 boards.
* **The cyclen band is not used as a gate.** The deck on 199
  (`fpcamp_minfr_T6T4_199.inp`) still carries the placeholder
  `cycle_target_efpd = 633.0 ± 5.0`, which no r8 row satisfies (r8's best sits at
  625.46), so the S1E §8 "in-band" language is not reconcilable with the deck as
  shipped. Registered response: **this wave reads `cyclen` as a continuous
  parent-relative delta and uses no band at all.** The band discrepancy is
  flagged for the campaign owners and gates nothing here.

---

## 8. Frozen inputs (hashed at writing, before launch)

| artefact | sha256 | bytes |
|---|---|---:|
| `ablation_wave.py` | `89d05e83d96724a209454398f6ce4330ec7d7c3cb98944c9537b78c1d26530d1` | 38,510 |
| `data/design/ablation_wave_20260815.json` (plan) | `0693def74a2ec7897ad764721edce37850607e42dfb5b51728c7d6270433b6c4` | 360,041 |
| `data/design/ablation_wave_policy_v1_pred.csv` (blind predictions) | `03ff46637f83a1475ce5db540e6681b68125fa6462c96052407768cc78487c93` | 32,277 |

**Amendment 2, POST-hoc, disclosed (2026-08-16).** `ablation_wave.py` was
modified **after all three waves had run**, to fix `_done()`: it treated any
`record_id` present in the results jsonl as settled, including harness failures,
so a resume after a crash would silently skip them. The 625-branch wave hit
`[Errno 28] No space left on device` and exposed this. `_done()` now keys off the
project's own taxonomy (`verify.PHYSICS_KILL_FAILURES`): converged,
non-converged and physics kills are settled; staging / disk / exit-status
failures are not. New hash **`1b94c7128f41685b6b3852527ad8ff6625414f781009ad1ed9f17cae5f9280c1`**;
test `tests/test_ablation_resume.py` (10 cases, incl. replay of the real
incident file: 220 rows → 202 settled, 18 to re-run).

**No completed wave's results depend on this change** — all three ran on the
pre-fix code (`89d05e83…`), and the 625 resume was driven by a hand-filtered
jsonl, verified afterwards to match the fixed `_done()` exactly. The change is
recorded here rather than silently re-hashed because §8 is the section that
pins the artefact.

**Amendment 1, pre-launch, disclosed.** `ablation_wave.py` was first hashed at
`4a12966b…` / 38,516 B. A remote `--dry-run` against the real assets then died on
`UnicodeEncodeError: 'cp949'` — three em-dashes in *print* strings, fatal under
the box's default console codepage. They were replaced with ASCII hyphens
(−6 bytes) and the file re-hashed to `89d05e83…`. **No logic changed**: the diff
is three characters in output text. It is recorded here rather than silently
re-hashed because the whole point of §8 is that the artefacts are fixed. The plan
manifest and the blind-prediction file were **not** touched, and their hashes are
the originals. The launcher's plan-sha256 gate still pins `0693def7…`.

Read-only inputs: `data/store/records.parquet`, `data/store/fuel_types.parquet`,
`data/policy/elites.parquet`, `data/models/policy_v1/`. **The canonical store is
not written until the final `merge-store`.** Boxes 181, 198 and 238 are untouched.

Seed `20260815`. The sampler is rank-deterministic, so the seed breaks ties only;
re-running `plan` on an unchanged store reproduces the manifest byte-for-byte.

---

## 9. Deliverables

`data/reports/ablation_wave_results_20260815.md` (registered analyses §6),
`data/policy/steps.parquet` (+150 rows, `lineage_source='ablation_paramA'`),
`data/store/records.parquet` (+150 rows via `merge-store`),
`runs/ablation_1move_T6T4/` (jsonl, maps, scoped kit).
