# 1-move ablation wave — RESULTS

Pre-registration: `data/reports/ablation_wave_prereg_20260815.md` (written and
hashed **before** launch). Every analysis below is one the pre-registration named;
nothing here is a post-hoc test presented as a planned one. Two deviations and
one unregistered robustness check are declared in §7.

---

## 1. RESULT

**150 chains run on box 199, 146 converged (97.3%), 0 harness errors.** The four
non-convergences are all `non_finite_flux` — honest physics kills about the
pattern, not harness faults. Total wall **6,081 s (1.69 h)**, median 294 s and 11
equilibrium cycles per chain, at `workers=16` on 24 logical cores.

**Free / paid split: 0 free, 150 paid** — exactly as pre-registered (§4).

| question | verdict |
|---|---|
| **Leakage arbitration** | **SETTLED. Outward fresh loading DOES cost cycle length at this era.** Both reactivity-conserving instruments agree, both CIs exclude zero, and their dose-response slopes agree across a 14× dose range. The corpus §4b retraction is itself refuted — its `+0.093` has the **wrong sign**, not merely an attenuated one. |
| **Policy v1 on this era** | **FAILS the registered gate, decisively.** Parent-blocked AUC 0.492 (`fr`) — chance. p@32 0.094 vs random 0.156. A move-class lookup table beats it on every metric. Its single largest blind commitment was maximally wrong. |
| **Corpus** | +150 rows, `lineage_source='ablation_paramA'`, all 150 verified single moves. |
| **Bonus** | **New cell record F_r 1.4685** (prior 1.4749), fully feasible — found by a move class the search had played 4 times in 19,820. |

---

## 2. PRIMARY — the leakage arbitration

### 2a. The registered paired test

Within-parent (outward − inward) at fixed move class, 10 parents, bootstrap CI
over parents, exact sign test. `d_cyclen` in EFPD; negative = outward runs shorter.

| instrument | mean(out−in) | 95% CI | sign | p | dose-response slope (parent FE) | 95% CI |
|---|---:|---|---:|---:|---:|---|
| **`fresh_relocate`** (primary, high dose, reactivity-conserving) | **−1.7415** | [−2.1336, −1.3342] | 0/10 | **0.002** | **−8.37 EFPD/unit** | [−10.58, −5.88] |
| **`batch_swap`** (secondary, mid dose, reactivity-conserving) | **−0.1128** | [−0.1995, −0.0262] | 1/10 | **0.021** | **−9.41 EFPD/unit** | [−15.53, −4.46] |
| `batch_flip` *(inadmissible — §2c)* | +3.1827 | [+1.6005, +3.9941] | 9/10 | 0.021 | +13,682 | [−13,779, +147,932] |

**Both admissible instruments give the same sign, both CIs exclude zero, and the
registered reading "outward `d_cyclen` < inward on both instruments" fires.**

The strongest evidence is not the two point estimates but their **slopes**:
−8.37 and −9.41 EFPD per unit `d_fresh_enr_r_center`, with overlapping CIs, from
two structurally different operators whose dose ranges differ by ~14× (median
|dose| 0.052–0.076 vs 0.002–0.007). A shared confound would have to scale with
dose across both operators to produce that. The mean effects differ by ~15×
precisely *because* the doses do — which is what a real dose-response looks like.

In physical units, on `fresh_relocate`: **outward median −1.25 EFPD, inward
median +0.28 EFPD — a 1.53 EFPD spread from one move.** Against the radially
neutral reference arm (`rewire_swap`, mean `d_cyclen` +0.42 EFPD), outward is the
side that loses cycle length.

### 2b. The corpus's own statistic, recomputed interventionally

`policy_corpus_20260815.md` §4b reports `d_fresh_share_periph` vs `d_cyclen` =
**+0.093** over 19,726 observational moves and concludes *"on this corpus, loading
fresh outward does NOT cost cycle length"*, retracting an earlier opposite
finding. The same correlation on this balanced, interventional, reactivity-conserving
sample:

| statistic | corpus (observational) | this wave (interventional, conserving) |
|---|---:|---:|
| `d_fresh_share_periph` vs `d_cyclen` | **+0.093** | **−0.635** |
| `d_fresh_enr_r_center` vs `d_cyclen` | — | **−0.786** |
| `d_fresh_share_periph` vs `d_f_r` | −0.131 | **−0.545** |

**The retraction is retracted.** The observational estimate is not a weak version
of the truth, it is the wrong sign. And the flattening half — which the corpus
believed and trained on — is real but was attenuated ~4× by the same confound.
Both halves of the engineer's rule of thumb are confirmed, and the corpus was
under-reading both.

### 2c. Why `batch_flip` was excluded — the prediction that paid off

The pre-registration (§3b) fixed, before any data, that `batch_flip` is *not* a
leakage instrument: its radial dose is ~1e-5 while it is the only class that
changes total fresh reactivity (|Δ| up to 1.20). Measured:

* **`corr(d_cyclen, d_fresh_enr_mass) = 1.000`** (n=20) — its cycle-length
  response is *entirely* the reactivity change;
* `corr(d_cyclen, d_fresh_enr_r_center) = 0.343` — radius explains almost none of it;
* `fresh_relocate` and `batch_swap` have `max|d_fresh_enr_mass| = 0.000000` — exactly conserving.

And `batch_flip` carries the **opposite** cyclen sign (+3.18). Pooling all classes
— which is exactly what the observational corpus does — mixes this reactivity
effect into the radial estimate and drags it positive. **This is the mechanism of
the §4g era disagreement, identified.** Had the exclusion not been registered in
advance, the honest-looking move after the fact would have been to pool all three
classes and report a muddle.

### 2d. Flattening, and a metric caveat that matters

Outward also improves the flattening objectives on both location statistics —
`fresh_relocate` mean(out−in) for F_r **−0.404** [−0.532, −0.236] and for
`node_peak` **−0.357** [−0.459, −0.222], both CIs excluding zero.

**But the improving *fraction* points the other way, and that is a real finding
about the metric, not about the physics.** Off an elite parent, improvement is a
tail event, and the two directions have very different tails:

| `fresh_relocate` `d_f_r` | min | p25 | median | mean | max | frac < 0 |
|---|---:|---:|---:|---:|---:|---:|
| inward | −0.0198 | +0.0169 | +0.1464 | +0.5388 | **+1.8586** | **0.143** |
| outward | −0.0023 | +0.0447 | +0.1074 | +0.1002 | +0.2083 | 0.036 |

Outward is **reliably mild** (bounded at +0.21); inward is **heavy-tailed**
(catastrophic to +1.86, mean dragged 5× above its median) but its longer left
tail crosses zero slightly more often. So: outward is the better *expected* move
and the safer move; inward is the higher-variance lottery. The improving-fraction
metric — the one the corpus tabulates and the one `policy_v1` is trained on —
prefers the lottery. Anything trained on improving-fraction alone inherits that
preference, which is worth knowing before the v1.1 fine-tune.

---

## 3. PROSPECTIVE — policy v1 on the era it failed

146 labelled children, scored blind before launch
(`ablation_wave_policy_v1_pred.csv`, sha256 `03ff4663…`, hashed in the
pre-registration).

| head `fr` (base rate 0.205) | AUC | parent-blocked AUC | p@32 |
|---|---:|---:|---:|
| **policy_v1** | 0.512 | **0.492** | **0.094** |
| random | 0.455 | 0.450 | 0.156 |
| class_freq | **0.648** | **0.658** | **0.327** |
| periph | 0.539 | 0.548 | 0.187 |

| head `flat` (base rate 0.151) | AUC | parent-blocked AUC | p@32 |
|---|---:|---:|---:|
| **policy_v1** | 0.529 | 0.551 | **0.000** |
| random | 0.407 | 0.383 | 0.062 |
| class_freq | **0.611** | **0.632** | **0.214** |
| periph | 0.536 | 0.559 | 0.139 |

**Registered gate: parent-blocked AUC ≥ 0.65 on `fr` AND p@32 beating all three
baselines. FAILS both, on both heads.** Parent-blocked AUC 0.492 on `fr` is
indistinguishable from chance — given a board and 15 candidate moves, v1 cannot
order them. It is beaten by `random` on the `fr` deployment metric, and on `flat`
**zero of its top 32 improved**. The `class_freq` lookup table beats it
everywhere, which says the little signal there is at this era is carried by move
class alone.

This is worse than `heldout_era` (AUC 0.650/0.682, p@32 0.328) predicted, and the
reason is visible in the blind-vs-measured table:

| move_class | direction | predicted P(improve F_r) | measured |
|---|---|---:|---:|
| `batch_flip` | **outward** | **0.814** | **0.000** |
| `batch_flip` | inward | 0.173 | 0.300 |
| `batch_swap` | inward | 0.575 | 0.400 |
| `batch_swap` | outward | 0.353 | 0.250 |
| `fresh_relocate` | inward | 0.416 | 0.143 |
| `fresh_relocate` | outward | 0.459 | 0.036 |
| `rewire_swap` | neutral | 0.576 | 0.300 |

**The registered falsification reading fires on its first clause.** v1's single
largest commitment — a 0.64 gap favouring outward `batch_flip` — is not merely
wrong, it is maximally wrong: it ranked those moves highest and **0 of 10
improved**, while the inward `batch_flip` moves it ranked lowest improved 3 of 10.
The pre-registration named this in advance as the diagnostic: `batch_flip` is the
class with negligible radial dose and large reactivity change. **v1 is reading
the batch labels / reactivity channel and reading it backwards at this era.** Its
"radial rule" is a proxy that inverts off-distribution.

The second clause also fires: on `fresh_relocate`, the genuinely radial
high-dose class, v1 predicted a near-null direction effect (0.459 vs 0.416) where
the interventional truth is a large, reliable one. **It under-uses the one axis
that is real and over-uses the one that is an artefact.**

**Consequence: `policy_v1` must not be wired into any ga80/paramA campaign.** The
`heldout_era` fold was not pessimistic enough — a prospective test on balanced
data found it at chance.

---

## 4. Per-stratum outcomes

`v` = fraction improving (lower is better); `^` = fraction with longer cycle.

| move_class | direction | n | conv | F_r v | mean d_f_r | flat v | mean d_node_peak | CBC v | mean d_cbc_max | cyclen ^ | mean d_cyclen |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch_flip | inward | 10 | 10 | 0.300 | +0.0183 | 0.200 | +0.0297 | 0.900 | −23.10 | 0.100 | −1.654 |
| batch_flip | outward | 10 | 10 | 0.000 | +0.1046 | 0.000 | +0.1037 | 0.100 | +27.56 | 0.900 | +1.529 |
| batch_swap | inward | 20 | 20 | 0.400 | +0.1903 | 0.100 | +0.1783 | 0.050 | +12.61 | 0.900 | +0.104 |
| batch_swap | outward | 20 | 20 | 0.250 | +0.0192 | 0.150 | +0.0280 | 0.900 | −8.48 | 0.450 | −0.009 |
| fresh_relocate | inward | 30 | 28 | 0.143 | +0.5389 | 0.071 | +0.4285 | 0.321 | +16.83 | 0.821 | +0.824 |
| fresh_relocate | outward | 30 | 28 | 0.036 | +0.1002 | 0.036 | +0.0448 | 0.393 | +3.39 | 0.250 | −0.975 |
| rewire_swap | neutral | 30 | 30 | 0.300 | +0.0784 | 0.400 | +0.0750 | 0.200 | +4.92 | 0.700 | +0.419 |

Absolute improving fractions are **not** comparable to corpus §4a — every parent
here is an elite (prereg §7). The within-parent contrasts are the readout.

Note the CBC column: outward lowers boron on both conserving instruments
(mean(out−in) −12.5 and −21.1 ppm, both CIs excluding zero) *and* shortens the
cycle. Less reactivity held in-core, shorter cycle, lower boron — physically
coherent, and consistent with leakage being the mechanism.

---

## 5. Bonus — a new cell record from a class the search never played

| | |
|---|---|
| record | `d84668059508…` **F_r 1.4685** (prior cell record 1.4749) |
| feasible | F_q 1.8328, CBC 1303.16, \|AO\| 0.0232, cyclen 618.49 — clears all four axes |
| move | `batch_swap`, units 48↔51 — a **2-slot label exchange**, `n_slots_changed = 2` |
| parent | `1165441c31ea…` (F_r 1.4877) — the **node_peak** elite, *not* the F_r leader |

One move off a non-leading elite beat the cell record by 0.0064, and beat its own
parent by 0.0192. The cell now holds 1,220 rows, 425 feasible, record 1.4685.

**This does not reopen the S1E §8 close-out, and should not be read as doing so.**
That clause was about *campaign rounds* — 100 MASTER calls of guided search — and
its evidence (search plateau, flat in-cell accuracy) stands. What this shows is
narrower and more useful: **the close-out was a statement about an instrument,
not about the cell.** A different instrument, on the same cell, at the same
budget, found a better board.

The likely reason is visible in the corpus: `batch_swap` appears **4 times in
19,820 same-cell moves**. The campaigns essentially never play it
(`batch_prob = 0.15`, split between flip and swap, and a swap needs two fresh
units of different batches). An exhaustive single-move sweep plays it 20 times per
wave. The cheap follow-up is not another campaign round — it is a `batch_swap`
enumeration over the existing elite set, which is ~200 fresh chains at most and
needs no new machinery.

---

## 6. Corpus rows added

**Method: a compatible appender, `ablation_analyze.py corpus`** — *not* an edit to
`mine_policy_corpus.py`. It calls `mine_policy_corpus.build_steps` **itself** on
the store subset holding the wave's children and their parents, so all 77 columns
are produced by the corpus's own code path. The only column this module sets is
`lineage_source`. Validation: the same subset-mining procedure was run against
campaign `fpcamp_minfr_T6T4` and reproduced its 41 canonical `steps.parquet` rows
**exactly**, every column identical.

`ablation_wave.py` was deliberately left untouched after launch — its sha256 is
pinned in the pre-registration and it is the artefact that ran on 199.

| | before | after |
|---|---:|---:|
| `data/policy/steps.parquet` | 27,458 | **27,608** (+150) |
| `data/store/records.parquet` | 72,870 | **73,020** (+150) |
| `data/store/maps.npz` | 72,081 | **72,227** (+146) |

All 150 new steps carry `lineage_source='ablation_paramA'`, `single_move=True`
(150/150), `cell='T6_T4/f121/paramA'`, and a `parent_record_id` that is a
first-class store lineage edge. Backup: `steps.parquet.bak_pre_ablation20260815`.

**Known gap, not fixed here:** `steps.parquet` still does not contain campaign
`fpcamp_minfr_T6T4_r8` (22 edges) — the corpus was mined before r8 was harvested.
Out of scope for this wave; flagged for the next corpus rebuild.

---

## 7. Deviations and limits

**Registered deviations (2), declared:**

1. **7 usable strata, not ~15.** Four of the 12 (class × direction) cells are
   structurally empty and one is excluded by design. Registered in prereg §3a
   *before* launch, with the reason: `rewire_swap` cannot have a radial direction,
   which is a theorem about the operator, not a sampling shortfall.
2. **p@32 is drawn from the whole 146-row labelled set**, not a 256-row batch —
   the wave is 150 rows. Registered in prereg §6c. All scorers see identical
   draws and identical random tiebreaks, so the comparison stays paired.

**Unregistered robustness check (added after seeing the data, declared as such):**
the paired test was re-run on within-parent **medians** instead of means, because
the `fresh_relocate` `d_f_r` distribution turned out heavy-tailed (§2d). The
cyclen verdict is unchanged — `fresh_relocate` −1.62 [−2.08, −1.13] sign 0/10
p=0.002; `batch_swap` −0.113 [−0.200, −0.026] sign 1/10 p=0.021. It changes no
conclusion and is reported so the mean-based headline is not load-bearing alone.

**Limits, carried from prereg §7 and still binding:** one cell, one pair, one
feed, one library, ten elite parents near F_r ≈ 1.48–1.50. The sign of the
leakage effect is established *at this operating point*; it does not retro-fit a
cause onto the `sa_mocha` era's 260624 boards, and it does not license a claim
about ga80 or other feeds. `batch_flip` says nothing about leakage; `rewire_swap`
says nothing about direction. No cyclen band was used as a gate — and the deck
band discrepancy noted in prereg §7 (`cycle_target_efpd = 633.0 ± 5.0`, which no
r8 row satisfies) remains unreconciled and is flagged for the campaign owners.

---

## 8. Artefacts

| path | what |
|---|---|
| `data/reports/ablation_wave_prereg_20260815.md` | pre-registration (pre-launch, hashed) |
| `data/reports/ablation_wave_results_20260815.md` | this report |
| `data/reports/ablation_wave_tables.txt` | raw analysis output |
| `ablation_wave.py` | plan/score/run/kit — sha256 `89d05e83…`, ran on 199 |
| `ablation_analyze.py` | corpus appender + registered analyses |
| `data/design/ablation_wave_20260815.json` | plan manifest, sha256 `0693def7…` |
| `data/design/ablation_wave_policy_v1_pred.csv` | blind predictions, sha256 `03ff4663…` |
| `runs/ablation_1move_T6T4/ablation_results.jsonl` | 150 raw outcomes, sha256 `61d172b3…` |
| `runs/ablation_1move_T6T4/kitdata/` | merged kit (150 rows, 146 maps) |
| `runs/ablation_1move_T6T4/ablation_1move_out.log` | run log with asset fingerprints |

Post-merge: `data/store/records.parquet` sha256 `f8a64dc8…`,
`data/policy/steps.parquet` sha256 `aa87e80f…`.

Boxes 181, 198 and 238 were untouched. Box 199 is now free; its kit still holds
`runs/ablation_1move_T6T4/` and `ablation_1move_rc.txt` (rc 0).

---

## 9. What this changes

1. **Train the leakage-arbitration head.** Corpus §10 deferred it pending this
   wave. The sign is now established interventionally at the live operating point,
   with a dose-response, on 96 reactivity-conserving single moves.
2. **Do not deploy `policy_v1` on ga80/paramA.** Chance-level parent-blocked AUC
   on a prospective, balanced test.
3. **Fix the feature framing before v1.1, not just the data.** The failure is
   structured, not noisy: v1 leans on a reactivity/label proxy and inverts it
   off-distribution. `d_fresh_enr_mass` — the covariate that made this diagnosable
   — is *not* in the corpus schema and should be added; it separates "moved
   reactivity" from "changed how much there is", which is exactly the distinction
   v1 fails to make.
4. **Reconsider the improving-fraction training target** (§2d): it prefers the
   high-variance move over the better-expected one.
5. **Enumerate `batch_swap` over the elite set** (§5) — 4 examples in 19,820, and
   it produced the cell record on its first systematic exposure.
