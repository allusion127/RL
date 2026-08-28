# batch_swap deep-sample wave — PRE-REGISTRATION (addendum)

**Written 2026-08-15, BEFORE launch.** Addendum to
`data/reports/ablation_wave_prereg_20260815.md`, whose discipline, cell,
descriptor definitions (`mine_policy_corpus`), harness (`WaveVerifier`), kit path
and merge path are inherited unchanged. Only what is new is stated here.

---

## 1. Hypothesis

`batch_swap` is a **systematically under-played move class**: 4 of 19,820
same-cell corpus moves. The campaigns reach it only through
`_one_move`'s `batch_prob = 0.15`, halved again between flip and swap, and a swap
additionally requires two fresh units carrying *different* batch labels. The
1-move ablation wave gave it 40 chains — its first systematic exposure — and it
returned the cell record **F_r 1.4685** and the cell's #2 board **1.4740**, both
off the same parent, from only 4 samples of that parent's 224-board neighbourhood.

**Registered hypothesis:** a deeper `batch_swap` draw pushes the frontier below
**1.4685**, and possibly below the program-wide ga80 incumbent **1.4636**
(current gap **0.0049**).

**This does NOT reopen the S1E §8 campaign close-out.** That clause concerned
*campaign rounds* — 100 MASTER calls of guided search — and its supporting
evidence (search plateau at r7, flat in-cell served accuracy at §7) stands
untouched. As the ablation results framed it, the close-out was a statement about
**an instrument**, not about the cell. This wave is a different instrument
operating on a class that instrument almost never selects. A record found here is
not evidence that the guided-search loop should be restarted, and will not be
reported as such.

---

## 2. A correction to my own sizing, and what the cap actually buys

The ablation results §5 proposed "a `batch_swap` enumeration over the existing
elite set, which is ~200 fresh chains at most". **That estimate was wrong by
roughly 20×, and this wave is sized against the corrected number.**

A feed-121 board carries 30 fresh units split 16/14 or 15/15 between the pair's
two batches, so its `batch_swap` neighbourhood is 16 × 14 = **224 boards — per
parent**. Measured across 20 candidate elites: **every one has exactly 224**
(two have 225 at a 15/15 split), with 0–4 already labelled.

Therefore, at the registered cap of **220 chains**:

* "Exhaustive enumeration over the elite set" (20 parents ≈ **4,450** new boards)
  is out of reach by a factor of ~20.
* The cap buys **one** parent exhaustively, or a deep sample across several.

**Registered choice: a deep sample across six parents, not one exhaustive
parent.** Reasons, fixed now: the brief directs that the record core, its parent,
and the top-F_r/top-flat elites all be included; a record is a best-of-N draw and
spreading N across six basins hedges the risk that one basin is exhausted; and
the per-parent readout (§4b) needs more than one parent. The cost is that **this
wave is a deep SAMPLE, not an enumeration** — the word "enumeration" from my own
§5 is retired here rather than quietly redefined.

---

## 3. Allocation — 220 chains, exactly at cap

| parent | F_r | node_peak | nbhd | free | take | coverage | why |
|---|---:|---:|---:|---:|---:|---:|---|
| `d84668059508` | **1.4685** | 1.2926 | 224 | 3 | **70** | 31% | the record core — frontier expansion |
| `1165441c31ea` | 1.4877 | **1.2757** | 224 | 4 | **70** | 31% | produced the record from 4 samples; an ablation-wave parent, so §4b can audit it |
| `a4291805f655` | 1.4740 | 1.3071 | 224 | 1 | 25 | 11% | cell #2, the record's sibling |
| `abd38bc5b212` | 1.4747 | 1.3000 | 224 | 0 | 25 | 11% | cell #3, a different basin |
| `188c9a338d9f` | 1.4749 | 1.2957 | 224 | 4 | 20 | 9% | r8 top-F_r elite; also an ablation parent |
| `c6edd01be332` | 1.5117 | **1.2742** | 224 | 0 | 10 | 4% | best node_peak in the cell |

**220 paid, 0 free** (the 12 already-labelled neighbours are excluded from the
draw and reused as free labels in the analysis).

**Within-parent draw: proportional, NOT direction-stratified.** Even ranks over
`dose` across the parent's whole neighbourhood — an unbiased subsample. This is a
deliberate contrast with the ablation wave's balanced design and is what makes
§4b possible. Resulting mix: **182 inward / 30 outward / 8 neutral**, tracking the
true neighbourhood (~89% / ~9% / ~2%).

---

## 4. Registered readouts

### 4a. PRIMARY — best feasible F_r found

Reported against three fixed marks: the parent it came from, the current cell
record **1.4685**, and the ga80 incumbent **1.4636**. Feasibility is
`mine_policy_corpus.feasibility` (F_r ≤ 1.55, F_q ≤ 2.41, CBC ≤ 1600, |AO| ≤ 0.30,
converged) — a record that fails any axis is reported as infeasible and does not
count. `cyclen` is reported for every candidate record; **no cyclen band gates
anything** (the deck's `633.0 ± 5.0` remains unreconciled — ablation prereg §7).

**Registered readings:**
* **new feasible best < 1.4636** → the under-played-class hypothesis is confirmed
  in its strong form: one neglected operator closed a gap that four campaign
  rounds could not.
* **new feasible best in [1.4636, 1.4685)** → confirmed in its weak form: the
  class is productive, the incumbent stands.
* **no improvement on 1.4685** → the record was a lucky draw from a class that is
  *not* systematically rich; the ablation wave's `batch_swap` result was tail luck
  and should be reported as such. This is a real possible outcome and is named
  now so it cannot later be reframed.

### 4b. SECONDARY — per-parent improvement distribution vs the stratified estimate

For the two parents present in both waves, the ablation wave's **direction-balanced
4-sample** estimate of the improving fraction is compared against this wave's
**proportional** deep sample:

| parent | ablation 4-sample (balanced) | this wave |
|---|---:|---|
| `1165441c31ea` | 2/4 = **0.500** | 70-sample, proportional |
| `188c9a338d9f` | 0/4 = **0.000** | 20-sample, proportional |

**Registered prediction, fixed now.** The ablation design sampled 50% outward
where the true neighbourhood is ~9% outward, and measured `batch_swap` improving
fractions of 0.400 inward / 0.250 outward. A balanced design therefore
*under*-estimates the true per-parent improving rate. Reweighting to the true mix
predicts **0.389**, against the naive balanced pooled value of **0.325**.

**This wave should measure a pooled `batch_swap` improving fraction near 0.389,
not 0.325.** If it does, the ablation wave's balanced-design numbers must be
reweighted before being read as neighbourhood rates — a correction that applies to
every stratified table in the ablation results §4, and one worth knowing before
the v1.1 fine-tune consumes them. If instead the measured rate lands near 0.325,
the direction mix does not drive the rate and the balanced tables can be read
directly.

Also reported: the full per-parent distribution of `d_f_r` (min / p25 / median /
mean / max / frac<0), since the ablation wave established that mean and improving
fraction disagree in direction off an elite (results §2d).

### 4c. Corpus append

`lineage_source='batchswap_enum'`, via `ablation_analyze.py corpus` — the same
appender, which calls `mine_policy_corpus.build_steps` itself. Validated there by
reproducing campaign `fpcamp_minfr_T6T4`'s 41 canonical rows exactly.

**Also folded in:** the 22 `fpcamp_minfr_T6T4_r8` edges flagged as missing in
ablation results §6, appended with their correct native tag
`lineage_source='lpopt_genome'` (they are ordinary campaign edges, not
interventional). Free — no MASTER.

**Not done this wave:** blind policy-v1 scoring. v1 already failed its
prospective gate at chance level on this era (ablation results §3); a second
prospective test on a single move class adds nothing and is out of scope.

---

## 5. What this wave cannot conclude

Everything in ablation prereg §7 still binds — one cell, one pair, one feed, one
library, elite parents. Additionally:

* **One move class.** Nothing here speaks to `rewire_swap`, `fresh_relocate` or
  `batch_flip`, and a `batch_swap` record does not make it the best class in
  general — only that it was under-sampled relative to its yield.
* **Six parents at 4–31% coverage.** A null result bounds the class's yield at
  this depth; it does not prove the neighbourhoods are barren, because 69–96% of
  each remains unseen.
* **Best-of-N is a biased estimator of the frontier.** Reporting the minimum of
  220 draws will overstate what a *typical* batch_swap achieves; §4b's
  distributional readout, not the record, is the honest summary of the class.
* **No causal claim about why the campaigns miss it.** The 4/19,820 figure is an
  observation about the operator mix, not a measured counterfactual.

---

## 6. Frozen inputs (hashed before launch)

| artefact | sha256 | bytes |
|---|---|---:|
| `batchswap_wave.py` | `56c11457380720630fab842b24d61251d9adeafd741393d6488492c16208b90f` | 9,111 |
| `data/design/batchswap_wave_20260815.json` | `bb579a6afbc3b22c681a7c659929fd92a9d8c58b4ce4d274525737128feaebb9` | 484,576 |

Inherited unchanged and NOT re-hashed: `ablation_wave.py` (`89d05e83…`), which
supplies the enumerator, annotator, runner and kit builder. `batchswap_wave.py`
rebinds only its two provenance constants; the file is not edited.

Store state at writing: `records.parquet` 73,020 rows (`f8a64dc8…`),
`steps.parquet` 27,608 rows (`aa87e80f…`). Canonical store is read-only until the
final `merge-store`. Boxes 181, 198, 238 untouched. Seed `20260815`; the draw is
rank-deterministic, so the seed breaks ties only.
