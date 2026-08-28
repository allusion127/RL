# First 3-fresh-type (graded) campaign — `S3 _ <mid> _ S5` / feed 125 — PRE-REGISTRATION

**Written 2026-08-17, BEFORE the deck was hashed and before anything was launched
on 199.** Design note: `data/reports/tripletype_design_20260817.md` (BUILD+TEST,
no campaign). Model: the v7 champion from
`data/reports/ab2_addendum_S1H_20260817.md`. The 2-type predecessor this campaign
must beat: `data/reports/fpcamp_HGD569_f125_results_20260817.md`.

Everything below was fixed before any 3-type number existed. Where a value could
only be computed after the v7 model finished training, the **rule** is frozen
here and in code (§3), and only the resulting number is stamped in.

---

## 1. Why this campaign, and why here

Radial **grading** — stepping fresh reactivity in more than two levels — is the
textbook, non-exotic way to flatten radial power, i.e. to attack F_r directly.
Every campaign this programme has run used **two** fresh types. This is the first
that uses three.

The cell is chosen because it is the programme's best core and its failure mode is
narrow. At `P6253Z1G06N24_P6253Z2G10N24` / feed 125 the 2-type campaign closed
with a core that is **clean on four of five gates**:

| axis | best joint-clean 2-type core | gate | |
|---|---:|---:|---|
| **F_r** | **1.6357** | 1.55 | ❌ the sole open axis |
| CBC_max | 1565.46 | 1600 | ✅ (34.54 ppm margin) |
| F_q | 2.0346 | 2.41 | ✅ |
| \|AO\| | 0.0266 | 0.30 | ✅ |
| predicted pin BU | **76.955** | 78 | ✅ (1.045 margin) |

`record_id 8b9acbcd…`, found call 11/60. Predicted pin BU across the whole top-F_r
group sits **76.96–77.96** — inside the gate, but by **0.04–1.04 GWd/tU**. So the
programme is one axis away from its first deliverable-grade high-enrichment core,
and the axis is exactly the one grading is for.

**The pin margin is what makes the mid-type choice a real decision, not a
preference** (§3): a mid type with fewer Gd rods regrades the middle ring hotter,
and there is ~1 GWd/tU of headroom to spend.

---

## 2. The cell, the alphabet and the case id

| role | type_id | alias | e [w/o] | n_gd | Gd₂O₃ wt% |
|---|---|---|---:|---:|---:|
| hot | `P6253Z1G06N24` | S3 | 5.7861 | 24 | 6 |
| **mid** | **P5 or T1 — decided in §3** | | | | |
| cold | `P6253Z2G10N24` | S5 | 5.6023 | 24 | 10 |

| mid candidate | type_id | alias | e | n_gd | equal-thirds e_core | grading steps (hot→mid→cold) |
|---|---|---|---:|---:|---:|---|
| uniform-grading | `P6253Z2G08N16` | **P5** | 5.6685 | **16** | 5.685780 | 0.118 / 0.066 |
| n_gd-preserving | `P6253Z2G10N20` | **T1** | 5.6386 | **20** | 5.675821 | 0.148 / 0.036 |

2-type reference: `pair_e_core` 50/50 = **5.694438**.

Resolver probed directly against the real paramA package **before** this document
was finished (not assumed):

```
CaseAssetResolver(data/design/package, ..., library_id='paramA',
                  registry_aliases=data/design/package/registry.json)
  P5  triple -> _pair_e_core 5.685780   alias_case_key S3_P5_S5 @125
  T1  triple -> _pair_e_core 5.675821   alias_case_key S3_T1_S5 @125
  2-type pair -> 5.694438               alias_case_key S3_S5    @125
```

Both triples score on the **level-3 `pair_ecore`** rung (finite e_core ⇒ no
neutral fallback), and every member aliases. `|Δe|` to the pair is 0.009 (P5) /
0.019 (T1), so the same `bases/P0_P1/MAS_RST.APRQ_11_0705.02` restart the 2-type
f109/f125 campaigns used is the expected pick — the launch script gates on its
presence.

**Feed 125 fixed** (n_fresh 31 = 1+4N, single-cycle-discharge core), unchanged
from the 2-type predecessor, so the comparison changes exactly one thing: the
size of the fresh alphabet.

---

## 3. The mid-type decision — rule frozen before the model existed

`tripletype_midpick.py`, sha256
**`090e4721498b3f76894f7f4a3b7b3002609a4c4cbb56426b0dae18cd91449416`**, written
and hashed **while the v7 ensemble was still training** — i.e. before any 3-type
prediction could be read. It:

1. takes the donor pair's **converged f125 store rows** (the same rows the
   campaign gets as elite parents, §5);
2. rebuilds each as a genome and applies `graded_morph` under each candidate
   alphabet, **same rng seed sequence in both arms**, so the arms differ only in
   which type the morph writes;
3. serves the v7 ensemble and reads surrogate column 6 (`max_pin_burnup`) and
   column 0 (`F_r`).

**Decision rule, in code before the numbers:**

> the mid with the lower **p50 predicted pin BU** wins, provided the gap is at
> least the pin head's own in-cell MAE (**1.84**); inside that band the pin axis
> cannot resolve the two and the tie breaks on **p50 predicted F_r**.

The MAE gate is the honest part: a 0.3 GWd/tU difference between two arms of a
head whose in-cell MAE is 1.84 is not a measurement, and pretending otherwise
would dress a coin flip as physics.

Dry run of the morph plumbing on the real donors (before the model existed):
**292/292 seeds reached the third type for BOTH candidates**, feed exactly
preserved (31 → 31), e.g. `{S5:15, S3:16} → {S5:15, S3:10, mid:6}`. So neither
arm is advantaged by morph feasibility.

**VERDICT — stamped from `data/reports/tripletype_midpick_20260817.json`.**
292 seeds per arm, both from the 73 converged f125 donors, served under `s1h`:

| | pred pin BU min / p50 / max | under 78 | pred F_r min / p50 | pred CBC p50 |
|---|---|---:|---|---:|
| **P5** (n_gd 16) | 74.748 / **75.950** / 87.624 | 74.3 % | 1.6668 / 1.7594 | 1666.88 |
| **T1** (n_gd 20) | 74.915 / **75.691** / 86.529 | 76.7 % | 1.6560 / **1.7126** | **1620.83** |

> **Winner: T1 (`P6253Z2G10N20`), the n_gd-preserving mid.** Pin p50 gap is
> **+0.258 in P5's disfavour — INSIDE the pin head's in-cell MAE (1.84)**, so the
> pin clause did *not* fire: the axis cannot resolve the two arms and the
> pre-registered tie-break on predicted F_r p50 decided it (T1 leads by 0.047).
> **Stated plainly: T1 won on F_r, not on pin — pin merely failed to object.**
> Two unregistered signals point the same way (T1 has the lower predicted CBC,
> which matters at this feed per R3, and the larger share under the pin gate);
> they are reported for completeness and carried no decision weight.

**Case id: `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24`** (alias `S3_T1_S5`).

Worth recording because it is the first quantitative hint on the programme's
thin axis: **both** arms predict pin BU *below* the 2-type MEASURED band
(76.96–77.96). That is the grading/residence argument showing up in the
surrogate. It is a prediction from a model that has never seen a 3-type input
(§7), so R-PIN is where it gets checked against MASTER rather than believed.

---

## 4. The marks — pinned before launch

| mark | requirement |
|---|---|
| **PRIMARY / full-feasible** | a MASTER-verified core with **F_r ≤ 1.55 AND CBC ≤ 1600 AND F_q ≤ 2.41 AND \|AO\| ≤ 0.30**, with predicted pin ≤ 78. This would be the **programme's first deliverable-grade high-enrichment core** — it does not exist anywhere in the project (0/57 at 2-type f125, 0/40 at f109, 0/178 anchors). |
| **STRETCH — the number to beat** | **F_r < 1.6357** CBC-clean: beat the 2-type joint frontier at this exact cell/feed. This is the only honest "grading helps" bar, because it holds cell, feed, e_core band, model family and budget fixed and changes only the alphabet. |
| SECONDARY | F_r < 1.6088 (the 2-type raw, CBC-violating best) — a weaker claim, reported labelled, never as the headline. |
| PIN | predicted pin BU of the best-F_r cores vs the 2-type band **76.96–77.96**. Grading that buys F_r by hollowing the mid ring would show up here as pin **rising**; that trade must be reported even when F_r improves. |
| **NULL** | F_r does not fall below 1.6357 in 60 calls. Registered reading: **"a third fresh type does not buy F_r at this cell"** — reported plainly, with no rescue narrative. Given §7's model caveat, NULL is a genuinely likely outcome and is a publishable result about grading at this cell, not a failed run. |

**Registered asymmetry:** the 2-type mark 1.6357 was produced by a 60-call
campaign whose elite pool was *all-infeasible backfill* (its own prereg §6). This
campaign's pool is seeded with that campaign's **own converged output** (§5). So a
small F_r improvement is **not** attributable to grading alone — better seeding is
confounded with the third type. Only a *large* move is informative, and §8 says
what would separate the two.

---

## 5. Elite seeding — the cold start, and the code change it required

The design note's cold-start path (§3.3) is: morph the optimized 2-type elites
into 3-type seeds rather than re-randomizing the board. `mutate(..., batches=<3
types>)` draws `graded_morph` automatically, so the operator is reachable **once
the pool has 2-type parents**.

**It did not have them.** `CampaignDriver._store_elites` selects parents by
`case_pair == <the campaign's case>`, and a graded case id
`A_B_C` matches **zero** store rows. Left alone, the first 3-type campaign would
have started from an **empty elite pool** — random and heuristic boards only — and
would then have NULLed for a plumbing reason wearing the costume of a physics
result.

**Change made, disclosed here rather than buried:** a new default-off deck knob
`[search] elite_seed_cases`, a list of DONOR case ids whose converged store rows
also seed the elite-mutation parent set.

| property | how it is held |
|---|---|
| existing decks unaffected | default is empty; the parent set, rng draw order and pool are byte-identical when unset (test 1) |
| donors are **parents only** | routed through a new `_elite_seed_rows`, deliberately NOT through `_case_store_rows`, so the wave fine-tune **holdout is untouched** (test 3) |
| donors are never in-cell truth | they keep their own `case_pair`; no store row is rewritten, relabelled or copied |
| non-converged donors excluded; self-naming is a no-op | test 4 |
| the cold start actually works | `build_pool` over the donors produces elite children carrying the third type, feed preserved (test 5) |

### 5.1 The graded-budget guard — found by measurement, not by reading the code

With elite seeding working, a dry run of the real deck exposed a second, larger
problem. **5 of the 8 wave-0 picks were 2-type boards.** At live pool settings
(2,500 candidates, real donors, real deck) the pool was **59.5 % 2-type at
wave 0** (48.3 % at wave 4), and the acquisition ranked those *first*.

The mechanism is legitimate and therefore easy to miss: a 3-type alphabet admits
boards that feed only two of its members, and ordinary mutation of a 2-type
donor produces exactly those. `graded_morph` fires only on a fraction of elite
draws; the local-search (exploit) arm never introduces a type at all, so it walks
graded boards straight back down to two.

**Why that is fatal to this campaign specifically, not merely untidy:** a 2-type
board under this case id is **the same physical core** as a 2-type board under
the pair's case id — identical `%LPD_SHF`, identical MASTER answer. The store
already holds **57 MASTER-verified 2-type cores at this exact cell and feed**.
So those calls would spend a 60-call budget re-measuring a population we already
own, selected by a model that has never seen a 3-type input (§7). The campaign
would then have reported "grading did not beat 1.6357" while having spent most
of its budget not testing grading.

**Second knob, same discipline as the first:** `[search]
require_all_fresh_types` (default **off**), which sets
`CaseContext.require_all_batches` and is enforced in the two places candidates
are born — `build_pool._admit` and `acquisition.local_search`'s neighbour loop.

| property | how it is held |
|---|---|
| default off | existing decks byte-identical (test 6) |
| no-op for a 2-type case | a pair board always feeds both members; pool record_ids identical with the flag on and off, so even the rng draw order is unperturbed (test 8) |
| graded pools admit only full-alphabet boards | test 7, which also asserts feed is preserved — the guard **filters**, it never rewrites a board |
| the knob actually reaches the context | deck → `SearchConfig` → `CaseContext` asserted end-to-end (test 9) |

**R-GRADE survives, and that was a requirement, not a bonus.** The guard forbids
*deleting* the third type, not *disliking* it: the search can still drive the mid
down to a single assembly. The re-run dry run shows it doing precisely that —
**8/8 candidates 3-type**, with the exploit slots already pushing the mid to
**4 of 31** assemblies while the diversity slots sit at 16–36. So "does the
optimizer want the third type?" is now read off a *fraction*, which is a real
measurement, instead of off a *deletion*, which was an artefact.

`tests/test_elite_seed_cases.py` — **9 tests** (5 seeding + 4 guard), all
passing. Adjacent suites re-run green as one sweep: `test_triple_type`,
`test_construct`, `test_config`, `test_campaign_stub`, `test_acquisition`,
`test_frontier_search`, `test_lean_store_elites`, `test_elite_objective`,
`test_fuelcost_search`, `test_flatness_campaign` — **310 passed**.

**Donor: `P6253Z1G06N24_P6253Z2G10N24` (both feeds).** What that hands the
campaign, from the canonical store:

| feed | rows | converged | F_r min / p50 | joint-clean (sans F_r) | best joint-clean F_r |
|---|---:|---:|---|---:|---:|
| 125 | 84 | 73 | 1.6088 / 1.6726 | 15 | **1.6357** |
| 109 | 83 | 56 | 1.6743 / 1.8046 | 38 | 1.6743 |

`elite_top_k = 32`, ranked best-objective-first, so the parent set is the
**best ~32 F_r boards this pair has ever produced** — a materially stronger start
than either 2-type campaign had. f109 parents are feed-morphed to N=31 by the
existing `_morph_feed` path; that is standard behaviour, not new.

**This also fixes the 2-type campaigns' disclosed weakness** (their store was
never refreshed with their own results). Registered as a *confound*, not a win:
see §4's asymmetry note.

---

## 6. Deck — every knob, and why

Identical to `fpcamp_minfr_HGD569_f125_199.inp` except the lines named here.
Deck sha256 is pinned in the launch script's gate.

| knob | value | change from the 2-type deck |
|---|---|---|
| `[case] pair` | the winning triple (§3) | **the whole point** |
| `[case] feed` | 125 | unchanged |
| `[model] model_dir` | `data/models/s1h` | v7 champion (s1g cannot encode 18 globals) |
| `[model] cond_schema` | `v7` | v6b → v7 |
| `[search] elite_seed_cases` | `["P6253Z1G06N24_P6253Z2G10N24"]` | **new** (§5) |
| `[search] require_all_fresh_types` | `true` | **new** (§5.1) |
| `[search] near_miss_f_r` | **1.70** (was 1.75) | the tight n_moves=1 arm now arms between the 2-type joint frontier (1.6357) and the 2-type in-cell median (1.6726); the floor it was set against has moved |
| `[acquisition] objective` | `min_fr_max_cycle` | unchanged |
| `minfr_lambda` | **1000.0** | unchanged — F_r strictly dominates; 0 would delete the F_r term and silently invert the run into cycle maximisation |
| `f_r_limit` / `cbc_limit` / `f_q_limit` / `ao_abs_limit` | 1.55 / 1600 / 2.41 / 0.30 | unchanged |
| `minfr_pin_bu_limit` | **78.0** | unchanged (LEU+ 80 − 2.0 model margin) |
| `budget` / `n_waves` / `wave_size` | 60 / 7 / 8 | unchanged |
| cycle target/tolerance | 723.1 ± 40 EFPD, **REPORT-ONLY** | **NO cycle band gates anything** |
| `[verify] package_root` | `data/design/package` | paramA routing, unchanged |
| `[produce] template_fallbacks` | `[]` | must stay empty on paramA |

---

## 7. Model caveat — the strongest one this programme has had to write

From `ab2_addendum_S1H_20260817.md` §4.2, measured not argued: **every row in the
74,477-row training store is 2-type.** On that corpus the five composition globals
v7 added are degenerate —

* `g_type_frac_1` ≡ `g_split_frac` (max abs diff 0.0),
* `g_type_frac_3` ≡ **0** and `g_n_fresh_types` ≡ **2/3**, both constant,
* `g_e_type_std` ≡ `sqrt(w(1−w))·g_e_split` (max abs diff 6.0e−08).

So the v7 model has **never seen a 3-type input**, and two of its composition
channels have **never varied**. This campaign's every candidate drives
`g_type_frac_3 > 0` and `g_n_fresh_types = 1.0` — values outside the training
range on channels whose learned weights were fitted against constants.

**Registered consequences:**

* wave-0 rankings are **out of distribution**, more so than the 2-type f125 run
  (whose caveat was only "never seen this cell");
* the wave fine-tune gate has **no in-cell holdout** for this case (zero store
  rows), so it will read `explore`/NaN and cannot veto — the same blindness the
  2-type deck disclosed, here total. Registered in advance so a blind gate is not
  later reported as a passed gate;
* **the load-bearing early check is the design note's checklist item 6**: do the
  `graded_morph` seeds' MASTER-verified F_r beat their own 2-type parents in
  wave 0–1? That is a comparison against measured truth and does not depend on
  the model being right about anything.

---

## 8. Registered readouts — all reported whatever they say

* **R1** best F_r, joint-clean and raw, vs **1.6357** and **1.6088**.
* **R2** PRIMARY: any core passing all four gates (+ pin ≤ 78). Count, not anecdote.
* **R3** CBC under F_r pressure. The 2-type run found CBC materially binding at
  this feed (16/57 pass, median 1614.63, r(F_r,CBC) = −0.45 excl. control).
  Grading changes the radial reactivity distribution and therefore the critical
  boron; report the full (F_r, CBC) Pareto front and flag any F_r-feasible core
  that crosses 1600.
* **R4** F_q — non-binding at 2-type (52/57 pass); report whether that survives.
* **R-PIN** predicted pin BU of the best-F_r cores vs the 2-type band 76.96–77.96
  and the 78 gate. **Reported even when — especially when — F_r improves.**
* **R-GRADE** *the readout that is new to this campaign*: the **feed composition
  of the winning cores** — how many assemblies of each of the three types, and
  where radially. Registered questions, both answerable from `state.json` +
  `waves/`: (a) what **mid-type fraction** does the search converge to — does it
  hold a substantial middle ring, or drive it to the 1-assembly floor the guard
  leaves it? (b) does the radial ordering come out **monotone** (hot inner →
  cold outer or the reverse)? (a) is the cleanest falsification available:
  **a search that spends its budget driving the mid fraction to the floor has
  answered the question in the negative**, whatever F_r does. Baseline to
  compare against, from the pre-launch dry run: the exploit slots opened at
  **4/31**, the diversity slots at 16–36/31.
* **R-SEED** the confound control demanded by §4: F_r of the **wave-0 morphed
  seeds** against **their own 2-type parents** (`parent_record_id` is recorded in
  each `waves/wave_NN/selection.json`, so this is a paired comparison against
  MASTER truth, not against the model). This is the readout that separates
  "grading helped" from "better seeding helped", and after §5.1 it is the ONLY
  one that can: the guard means this campaign no longer produces 2-type cores of
  its own to compare with. The 2-type control is the **existing 57-row
  population** at the same cell, feed and budget — a cleaner control than an
  in-campaign mixture chosen by an out-of-distribution model would have been.

---

## 8.1 Launch stamp — what was actually armed

Everything above was written before launch. This section records the arming.

| item | value |
|---|---|
| deck | `fpcamp_minfr_TRIPLE_f125_199.inp` sha256 `B042D49AFC274EA2DA630627D49DD0518205E71EA9E44BC41F90BF39D7E342F1` |
| store on 199 | sha256 `B5B29460A0C1C1E6AD3E3B2B9F410ED021E306099A91ECEEC8B05775B739CAE7` (= canonical, 74,477 rows) |
| model | `data/models/s1h`, `cond_schema=v7`, 5 members, `DONE` |
| gate | `data/reports/gate_s1h.json` — **PASS**, no-regression `worst_drop 0.0214` vs `eps 0.1388`, `blind_targets []`, legacy tail PASS |
| run dir | `runs/fpcamp_minfr_triple_f125` (fresh) |
| launched | 2026-08-17 on 199, `Win32_Process Create ReturnValue=0`, all four launch gates passed (deck sha, store sha, `cond_schema=v7`, level-3 restart present) |

**On-box preflight before arming** (read-only, started no MASTER): deck loads;
donors resolve to **129 converged rows** (73 @ f125, 56 @ f109, best F_r 1.6088);
resolver returns `e_core 5.675821` and alias `CaseKey('S3_T1_S5', 125)` — level-3
scorable, no neutral fallback; model loads at `cond_schema v7` with 5 members;
context reports `batches=(S3,T1,S5) n_fresh=31 discharge=True require_all=True`.

---

## 9. Fleet, provenance, and what is NOT touched

* **199 only**, launched idle (busy gate refuses rather than stacks; deck-hash
  gate refuses on any edit). **198 / 181 / 238 untouched.**
* 199's `data/store/records.parquet` is refreshed to the canonical
  74,477-row copy (sha256 pinned in the launch script) **because the donor rows
  are the point of §5** — a deliberate reversal of the 2-type run's
  "don't reship" rule, recorded here rather than done quietly. Pre-ship backup
  kept on the box.
* Fresh run dir; ship-don't-remote-edit (every file scp'd whole).
* **Concurrent-edit hazard, disclosed:** another agent extended the repo's
  fresh-type cap from 3 to 5 (`MAX_FRESH_TYPES`) **while this arm was training**.
  The v7 *encoding* is pinned at width 3 and is unaffected — verified, not
  assumed, by featurizing the same 64 store rows under the training snapshot on
  238 and under the current local tree and comparing hashes:
  cells `ffaacccf94bb2e10bd8c19bac96ca47f`, globals
  `15b10bf6ccc37681fbac1fadee54bc15`, **identical on both**, shape (64,62,19,19)
  / (64,18). The source shipped to 199 is re-verified against that fingerprint
  immediately before shipping.
* Known cosmetic defects carried over from the 2-type runs: the
  `[optimize][DEPRECATED]` banner for `min_fr_max_cycle` is expected, and
  `lpopt report`'s best-patterns table ranks by cycle distance and will not show
  the F_r winner — read `state.json → best`.
