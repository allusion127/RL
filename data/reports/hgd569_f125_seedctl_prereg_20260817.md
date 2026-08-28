# R-SEED control — 2-type, donor-enriched elites — PRE-REGISTRATION

**Written 2026-08-17, BEFORE launch on 199.** Demanded verbatim by
`data/reports/tripletype_f125_results_20260817.md` §8 (R-SEED): "re-run the
*2-type* deck at this cell with `elite_seed_cases` pointing at the same donor
population. Same seeding, one type fewer. That single run separates the two,
and it is cheap — the deck already exists." This document, the deck
(`fpcamp_minfr_HGD569_f125_SEEDCTL_199.inp`), and the launch/status scripts are
that run.

---

## 1. The confound being closed

The 3-type campaign's headline — joint-clean F_r **1.5993** vs the 2-type joint
frontier **1.6357** (Δ **−0.0364**) — changed two things at once relative to
the original 2-type run (`fpcamp_minfr_HGD569_f125_199.inp`):

| axis | original 2-type | 3-type |
|---|---|---|
| fresh-type alphabet | 2 | 3 |
| elite parent pool | 32-row all-infeasible backfill | 32 real optimized boards, ranked best-objective-first from a 129-row converged donor pool (73 @ f125, 56 @ f109) via `[search] elite_seed_cases` |
| model | s1g (v6b) | s1h (v7) |

This run changes **only** the elite pool and the model (both forced together —
s1h is the only checkpoint able to score the pool correctly with the current
codebase, and using a stale s1g here would itself be a second, unregistered
confound). The alphabet stays at 2. Everything else — cell, feed, objective,
gates, budget, box, `near_miss_f_r`, cycle report-only band — is byte-identical
to the original 2-type deck.

## 2. How parity is actually achieved (verified, not assumed)

The 3-type deck's `elite_seed_cases = ["P6253Z1G06N24_P6253Z2G10N24"]` named
the 2-type pair as an external donor because the 3-type case's own
`case_pair` matches zero store rows. **This run's own `case_pair` IS that
donor.** Reading `lpopt/search/campaign.py` (`_elite_seed_rows`,
lines 1018-1019) confirms a donor entry equal to `self.ctx.pair` is filtered
out before the store is even queried — so setting the knob here would be a
proven no-op. It is omitted.

The mechanism that reproduces the 3-type run's advantage is
`_case_store_rows` (own-case, **no feed filter**, `campaign.py:992-999`)
against a store that has, since the original 2-type run, been enriched by
that very run's own +57 converged output. Verified directly on box 199 before
writing this document (read-only, no MASTER started):

| check | value |
|---|---|
| store on 199, sha256 | `4BE89C61856CCD2B41DDDC9B546B1217AC95BDBF019225CDAFFFFE73C5DA8ADD` |
| store on 199, total rows | 74,537 |
| pair rows / converged | 167 / **129** (73 @ f125, 56 @ f109) |
| local canonical store, total rows | 74,537 (sha256 `B55E108D7D793B190878CFFCBD770ABDCA24E93746FF0A405AEEEE8BDF5F57BC` — different byte layout, identical row content per the counts above) |

129 converged, split 73/56 across feeds — **identical to what
`elite_seed_cases` fed the 3-type run** (prereg §5 there: "f125 84 rows/73
converged... f109 83 rows/56 converged"). **No store reship was performed**:
box 199's store already reached this state through the 3-type run's own
harvest and was left in place; the launch script pins this exact SHA256 as a
gate so any drift refuses rather than silently seeding from a different
population.

`elite_top_k = 32` (unchanged) then ranks this 129-row pool best-objective-
first — the same selection rule, same donor rows, same feasibility-first
ordering (`_store_elites`) as the 3-type run used.

## 3. The marks

| mark | value |
|---|---:|
| original 2-type joint-clean (the run this control most resembles) | **1.6357** |
| 3-type joint-clean (the run whose delta is under test) | **1.5993** |
| implied full delta | −0.0364 |

**Registered question:** how much of that −0.0364 is attributable to the
elite-seeding/model change alone, with the alphabet held at 2?

**Verdict rule, fixed before launch:**

| control joint-clean F_r | reading |
|---|---|
| **≤ 1.61** | seeding (+ model) explains most of the −0.0364; the 3-type alphabet's own contribution is small |
| **≥ 1.625** | seeding is not the main driver; the 3-type effect is real, at **≥ ~0.02** (1.6357 − 1.625) |
| strictly between 1.61 and 1.625 | genuinely mixed — both effects contribute at comparable scale; report the exact split, no forced call |

Both readings are reported whatever they say, per house discipline (no
rescue narrative on either side).

## 4. What this run does NOT resolve

* It does not distinguish "better model (s1h vs s1g)" from "better elite
  seeding" — both changed together, deliberately, because the current
  codebase requires it (s1g cannot score a candidate pool as well-calibrated
  on this pair's own recent output as s1h is, and re-training a v6b-schema
  s1g clone would be a manufactured, unrepresentative comparison). If the
  control lands near 1.5993, the honest reading is "seeding-and-model, not
  specifically the third type" — not "seeding alone."
* It does not re-test whether the 3-type run's own within-campaign
  correlations (r(mid, F_r) = −0.42, r(mid, CBC) = +0.64, tripletype results
  §4) are real — those are unconfounded by construction (computed across
  49 same-seeding cores) and stand regardless of this run's outcome.
* Budget 60 carries the same risk both predecessors flagged: neither 2-type
  nor 3-type frontier had clearly stalled by call 60 in every prior run at
  this cell (3-type: best found call 57/60). A NULL or PARTIAL reading here
  is not proof of a stalled frontier, only of what 60 calls found.

## 5. Launch stamp (filled at launch)

| item | value |
|---|---|
| deck | `fpcamp_minfr_HGD569_f125_SEEDCTL_199.inp`, sha256 `8E5D3A75F4200797F4C6A59D4022A8F665C980921A54F1C44D8A8C8AAC24B341` |
| model | `data/models/s1h` (v7), gate `data/reports/gate_s1h.json` PASS |
| store on 199 (pinned, not reshipped) | sha256 `4BE89C61856CCD2B41DDDC9B546B1217AC95BDBF019225CDAFFFFE73C5DA8ADD`, 74,537 rows, pair converged 129 |
| run dir | `runs/fpcamp_minfr_hgd569_f125_seedctl` (fresh) |
| random_seed | 5696 (fresh — grep of every `.inp` in the repo confirms unused) |
| budget | 60 (7 waves × 8 + 4 reserve) |
| box | 199 only; 198 / 181 / 238 untouched |
