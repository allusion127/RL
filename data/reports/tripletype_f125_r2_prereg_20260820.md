# 3-fresh-type (graded) campaign, ROUND 2 — `S3_T1_S5` / feed 125 — PRE-REGISTRATION

**Written 2026-08-20, BEFORE the deck was hashed and before anything was
launched on 199.** Predecessor: `data/reports/tripletype_f125_prereg_20260817.md`
(round 1) / `data/reports/tripletype_f125_results_20260817.md` (round-1
results, CLOSED). Pin definition (official margin, user-confirmed 2026-08-20):
`data/reports/pinbu_definition_20260820.md`.

---

## 1. Why round 2, and why nothing else changes

Round 1 closed with the programme's best core anywhere: joint-clean
**F_r 1.5993, CBC 1597.33, F_q 1.9968, |AO| 0.0261**, measured pin BU
**74.38 PASS** (5/5 measured, `pinbu_definition_20260820.md` §4). It clears
four of five gates and misses F_r by **+0.0493**.

Round 1's own §9 registered the reason to come back: **the frontier had NOT
stalled at the 60-call budget** — the cumulative joint-clean frontier moved
`1.6579 → 1.6486 → 1.6105 → 1.6036 → 1.5993` at calls 1 → 3 → 9 → 19 →
**57/60**. The best core landed on the second-to-last call. That is the f109
pattern (budget-limited, not search-limited), not the 2-type f125 pattern
(which stalled at call 17). Round 1's own text: *"Budget 60 is undersized for
this cell in its 3-type form; an extension is a new decision with a new deck,
not a silent top-up."* This deck is that decision.

**Everything about the cell, feed, alphabet, objective and gate set is
unchanged.** The only things that differ from round 1 are: a fresh seed, the
champion (s1i v8 in place of s1h v7 — the first checkpoint that has seen any
3-type row), and an elite pool that now also contains round 1's own 49
converged triples as direct parents (in addition to the 2-type pair donor
round 1 already had). Full diff in the deck header
(`fpcamp_minfr_TRIPLE_f125_r2_199.inp`).

---

## 2. The marks — pinned before launch

| mark | requirement |
|---|---|
| **PRIMARY / full-feasible (TARGET)** | a MASTER-verified core with **F_r ≤ 1.55 AND CBC ≤ 1600 AND F_q ≤ 2.41 AND \|AO\| ≤ 0.30**, with predicted pin ≤ 78. This would be the **programme's first deliverable-grade high-enrichment core** — it does not exist anywhere in the project. |
| **STRETCH** | F_r < 1.5993 CBC-clean — beat round 1's own joint frontier at this exact cell/feed. |
| SECONDARY | F_r < 1.5956 (round-1 raw, CBC-violating best) — weaker, reported labelled, never as headline. |
| PIN | measured pin BU of the top new cores vs round-1's measured band (74.16–75.58, winner 74.38) and the 80 GWd/tU licensing limit (pin axial peak, per `pinbu_definition_20260820.md`). |
| **NULL — the honest one this round is watching for** | the 3-type F_r–CBC frontier closes **above 1.55** — i.e. CBC becomes the binding wall before F_r can reach the licensing bar, so no amount of further F_r-only search at this cell/case/objective can produce a full-feasible core. If this happens, the registered next step is **not** a third round at this cell but a hand-off to the fuel-design/blanket axis (a different lever than radial grading of the same three-type alphabet). |

**Registered asymmetry, honestly smaller than round 1's.** Round 1's
confound (grading vs seeding) was measured and roughly split by
`hgd569_f125_seedctl_results_20260817.md` (−0.0179 grading / −0.0185
seeding, of the −0.0364 total). This round changes seeding again (round 1's
own 49 triples added as direct parents) **and** the champion (s1h → s1i).
Any F_r improvement here cannot be cleanly attributed between "more budget at
the same search", "a better-seeded elite pool" and "a champion that has seen
this cell before" — this round does not attempt to separate them; it exists
to answer the practical question (does the frontier keep moving toward 1.55,
or does CBC wall it first), not the mechanistic one round 1 and its SEEDCTL
control already answered.

---

## 3. Registered watch — CBC, named before launch

Round 1's joint-clean winner sat **2.67 ppm** under the CBC gate (1597.33 vs
1600.0) — the closest any joint-clean f125 core in the programme has come to
that wall. Round 1's own R3 found CBC materially *relieved* versus 2-type
(59.2% pass vs 28.1%) with a flipped, positive `r(F_r, CBC)` — meaning **at
3-type, pushing F_r further down tends to push CBC up, not down**. That is
the opposite of the 2-type trade-off and it is exactly the mechanism that
would wall F_r before 1.55.

**This round records, explicitly, whether CBC becomes the wall:**

* report the full (F_r, CBC) Pareto front among round-2's converged cores;
* report whether any core reaches F_r ≤ 1.55 at all, and if so, whether it is
  CBC-clean;
* report whether the CBC-clean frontier (the honest reportable number) stalls
  **above** 1.55 even while the raw (CBC-violating) frontier continues past
  it — that specific pattern is the fingerprint of "CBC is the true wall,
  not F_r search headroom", and if it appears, this document's NULL is what
  fires: **"the 3-type F_r-CBC frontier closes above 1.55 at this cell"**,
  handed to the fuel-design/blanket axis as the next lever, not to a round 3
  budget top-up.

---

## 4. Mid-type fraction — registered continuation of R-GRADE

Round 1's winner carried **20 mid-type assemblies of 125 fresh** (16%), a
"transition ring", not the equal-thirds design-note composition (~42 each)
and not the 4-assembly floor the guard leaves reachable. This round reports
the same trajectory: the mid-type fraction of (a) all converged cores, (b)
the joint-clean subset, (c) the new winner, compared against round 1's
distribution (4:11, 8:8, 12:10, 16:17, 20:1, 24:2). A material shift away
from ~16–20% under a stronger elite pool and a cell-aware champion would be
worth registering even though it decides nothing on its own.

---

## 5. Deck — every knob that changed, and why

Full detail in the deck header. Summary:

| knob | round 1 | round 2 | why |
|---|---|---|---|
| `[flow] random_seed` | 5695 | **5697** | fresh, unused anywhere in the repo |
| `[model] model_dir` | `data/models/s1h` | **`data/models/s1i`** | v8, the first champion trained on any 3-type row (39 of round 1's 49 converged triples; `ab2_addendum_S1I_20260817.md` §4) |
| `[model] cond_schema` | `v7` | **`v8`** | 20 globals; the two new fields (`type_frac_4/5`) are still identically zero for a 3-type case and contribute no learned signal (`ab2_addendum_S1I` §3.2) — the live change is the re-encoding of the fields v7 already had, plus 60 extra training rows |
| `[search] elite_seed_cases` | `["P6253Z1G06N24_P6253Z2G10N24"]` | **`["P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24", "P6253Z1G06N24_P6253Z2G10N24"]`** | round 1's own 49 converged triples now seed directly (same alphabet, no morph); the pair donor is unchanged and still carries both 2-type f125 campaigns (`hgd569_f125` + `hgd569_f125_seedctl`, 121 converged f125 rows) |
| `[search] require_all_fresh_types` | `true` | **`true`** (unchanged, confirmed on) | the graded-budget guard; every call must buy a genuinely 3-type core |
| run dir / tag | `runs/fpcamp_minfr_triple_f125` | **`runs/fpcamp_minfr_triple_f125_r2`** | fresh run dir |
| `minfr_lambda` | 1000.0 | **1000.0 (kept)** | F_r strictly dominates |
| cycle band | report-only, gates nothing | **report-only, gates nothing (kept)** | unchanged |
| 5 gates (`f_r_limit`/`cbc_limit`/`f_q_limit`/`ao_abs_limit`/`minfr_pin_bu_limit`) | 1.55 / 1600 / 2.41 / 0.30 / 78 | **unchanged** | `minfr_pin_bu_limit=78` is the official margin per `pinbu_definition_20260820.md` (licensing limit 80 on the measured pin axial peak, minus the 2.0 GWd/tU model margin) |
| budget / n_waves / wave_size | 60 / 7 / 8 | **unchanged** | same house budget as round 1 |

Deck sha256: `B5683D4FB2F32E9E218DF0D6551928766C2FFF26B34EC78B1EC7E6A893FE0116`
(`fpcamp_minfr_TRIPLE_f125_r2_199.inp`).

---

## 6. Elite seeding — verified against the store before launch

| donor case | rows | converged | F_r min / median |
|---|---:|---:|---|
| `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24` (round-1 triple, **own case**) | 60 | 49 | 1.5956 / 1.6287 |
| `P6253Z1G06N24_P6253Z2G10N24` (2-type pair, both f125 campaigns) | 227 (144 @ f125) | 177 (121 @ f125) | 1.6036 / 1.6623 (f125 subset) |

Both donor cases resolved and counted directly against the local canonical
store (`data/store/records.parquet`, 74,657 rows) before the deck was
finalised — not assumed. `elite_top_k = 32` ranks the parent set
best-objective-first, so round 2 starts from **round 1's own best ~32
F_r boards at this exact cell**, a materially stronger cold start than round
1 itself had (round 1's pool had never seen this case before; round 2's has
seen 49 converged examples of it).

---

## 7. Model caveat — narrower than round 1's, not gone

`data/reports/ab2_addendum_S1I_20260817.md` §1: s1i is a **non-regression**
arm (gate `data/reports/gate_s1i.json`, `pass: true`), not an accuracy bid —
no gain on the 2-type corpus was ever the goal, and none is claimed. §3.2
there: `g_type_frac_4` / `g_type_frac_5` are identically zero on every
training row (no 4/5-type cores exist anywhere), so they are **inert** —
any observed difference from s1h must come from the re-encoding of the
fields v7 already had, or from the 39 (of 49) round-1 triples s1i's train
fold saw, not from the two new fields having learned anything.

**Registered reading:** treat s1i here as "the champion that has glimpsed
this exact cell 39 times", not "the champion that has mastered it". The wave
fine-tune gate now has a real (if thin) in-cell holdout — round 1's own
converged rows — unlike round 1's fully blind `explore`/NaN state, so it can
score skill this round, even if the signal is weak at n=49.

---

## 8. Registered readouts

* **R1** best F_r, joint-clean and raw, vs round 1's 1.5993 / 1.5956 and the
  1.55 licensing bar.
* **R2** PRIMARY: any core passing all four gates (+ predicted pin ≤ 78).
  Count, not anecdote.
* **R3 / CBC-WALL** — §3 above: the full (F_r, CBC) Pareto front; whether
  CBC becomes binding before F_r reaches 1.55.
* **R-PIN** predicted AND (top-5 new cores) **measured** pin BU vs round 1's
  measured band (74.16–75.58) and the 80 GWd/tU limit — the definitive
  observable per `pinbu_definition_20260820.md`.
* **R-GRADE** — §4 above: mid-type fraction trajectory, compared to round
  1's distribution.
* **R-BUDGET** — does the frontier keep moving late in the budget (as round
  1's did, call 57/60) or stall early (as the SEEDCTL control's did, call
  3/60)? A stronger elite pool could plausibly cause an early stall (the
  best answer is already close to the elite mean); report which pattern
  round 2 shows, as a read-out about how much further budget would still buy.

---

## 9. Fleet, provenance, launch stamp

* **199 only**, launched idle (busy gate refuses rather than stacks;
  deck-hash gate refuses on any edit). **198 / 181 / 238 untouched.**
* 199's `data/store/records.parquet` is refreshed to the local canonical copy
  (sha256 `FBDBAFBADB11BDF37EB0FF7F776A5D037BE040F514AA1BF663E515B699D614C6`,
  74,657 rows) before launch — the round-1 donor rows are the entire point of
  `elite_seed_cases`.
* Fresh run dir (`runs/fpcamp_minfr_triple_f125_r2`); ship-don't-remote-edit.
* Launch/status scripts: `launch_fpcamp_TRIPLE_f125_r2_199.ps1`,
  `status_fpcamp_TRIPLE_f125_r2_199.ps1`,
  `run_fpcamp_minfr_TRIPLE_f125_r2_199.bat`.
* Known cosmetic defects carried over unchanged from round 1: the
  `[optimize][DEPRECATED]` banner for `min_fr_max_cycle` is expected;
  `lpopt report`'s best-patterns table ranks by cycle distance and will not
  show the F_r winner — read `state.json → best`; `state.json → best_overall`
  surfaces the raw (possibly CBC-violating) core, not the joint-clean one —
  apply the CBC/F_q/AO filter before quoting a winner.

**Launch stamp (filled at launch time, not before):**

| item | value |
|---|---|
| deck sha256 (local, pre-ship) | `B5683D4FB2F32E9E218DF0D6551928766C2FFF26B34EC78B1EC7E6A893FE0116` |
| store sha256 (local, pre-ship) | `FBDBAFBADB11BDF37EB0FF7F776A5D037BE040F514AA1BF663E515B699D614C6` |
| model | `data/models/s1i`, `cond_schema=v8`, gate `data/reports/gate_s1i.json` PASS |
| run dir | `runs/fpcamp_minfr_triple_f125_r2` (fresh) |
| launched | — filled after `Invoke-CimMethod` returns |
