# R-SEED control — 2-type, donor-enriched elites — RESULTS

**Run 2026-08-17→18 on box 199.** Deck `fpcamp_minfr_HGD569_f125_SEEDCTL_199.inp`
(sha256 `8E5D3A75F4200797F4C6A59D4022A8F665C980921A54F1C44D8A8C8AAC24B341`), run
dir `runs/fpcamp_minfr_hgd569_f125_seedctl`, champion `data/models/s1h` (v7),
seed 5696, **rc = 0**, 8 waves, **60/60 calls**. Pre-registration:
`data/reports/hgd569_f125_seedctl_prereg_20260817.md`, written before launch.
This is the run demanded by `tripletype_f125_results_20260817.md` §8 (R-SEED):
re-run the 2-type deck with the same donor-enriched elite pool the 3-type run
got, alphabet held at 2, to split the −0.0364 headline into a
seeding-attributable part and a grading-attributable part.

---

## 1. HEADLINE: the confound splits almost exactly in half

| mark | value |
|---|---:|
| original 2-type joint-clean (all-infeasible-backfill seeding, s1g) | **1.6357** |
| **this control (donor-enriched seeding, s1h, still 2-type)** | **1.6172** |
| 3-type joint-clean (donor-enriched seeding, s1h, 3-type) | **1.5993** |

**Verdict, per the rule pinned in the prereg §3:** control joint-clean 1.6172
falls in the **mixed band** (strictly between 1.61 and 1.625) — neither
"seeding explains most" nor "grading is real at ≥0.02" fires cleanly. The
actual split:

| contributor | delta | share of the total −0.0364 |
|---|---:|---:|
| elite-seeding + model change (this run, alphabet held at 2) | **−0.0185** | **50.8 %** |
| the third fresh type itself (3-type minus this control) | **−0.0179** | **49.2 %** |

**Both effects are real and land within 3 % of an even split.** The 3-type
campaign's −0.0364 headline was genuinely confounded, exactly as its own
prereg warned — and now that the confound is measured rather than merely
flagged, grading survives with roughly half the credit, not all of it.

---

## 2. The numbers behind the headline

**Raw best:** F_r **1.6036**, CBC 1614.27 (CBC-violating, not reported as
headline), F_q 2.0204, |AO| 0.0245, cyclen 731.5 EFPD. Found call **36/60**,
wave 4. `state.json → best_overall` surfaces this same row — the familiar
defect (best_overall is the raw, not the CBC-clean, pick) carried over
unchanged from every prior run.

**Joint-clean winner (CBC≤1600 ∧ F_q≤2.41 ∧ |AO|≤0.30):** F_r **1.6172**, CBC
1564.36, F_q 2.009, |AO| 0.0269, cyclen 730.497 EFPD. Found call **3/60**,
**wave 0** — the very first wave, drawn straight from the donor-enriched
elite pool. No call in waves 1–7 beat it on the joint-clean axis; the
frontier on this readout stalled immediately, unlike the 3-type run's (best
at call 57/60) or the original 2-type run's own joint-clean winner (call
11/60).

| readout | control (n=48 converged) |
|---|---:|
| gate pass: F_r≤1.55 | 0 (0 %) |
| gate pass: CBC≤1600 | 10 (20.8 %) |
| gate pass: F_q≤2.41 | 45 (93.75 %) |
| gate pass: \|AO\|≤0.30 | 48 (100 %) |
| joint (CBC ∧ F_q ∧ AO) | **10 / 48 (20.8 %)** |
| all 4 (incl. F_r) | 0 |
| CBC min / p50 / max | 1561.27 / 1615.19 / 1716.42 |
| F_r min / p50 / max | 1.6036 / 1.6436 / 4.6942 (one outlier row) |
| Pearson r(F_r, CBC) | **+0.598** |

---

## 3. Comparison table — all three runs at this cell/feed, same objective/gates/budget

| | original 2-type | 3-type | **this control** |
|---|---:|---:|---:|
| alphabet | 2 | 3 | 2 |
| elite pool | 32-row all-infeasible backfill | 32 real boards, donor-enriched | 32 real boards, donor-enriched (same donor) |
| model | s1g (v6b) | s1h (v7) | s1h (v7) |
| joint-clean F_r | 1.6357 | 1.5993 | **1.6172** |
| raw F_r | 1.6088 | 1.5956 | 1.6036 |
| CBC pass rate (n) | 28.1 % (57) | 59.2 % (49) | **20.8 % (48)** |
| CBC median | 1614.63 | 1596.06 | **1615.19** |
| r(F_r, CBC) | −0.45 | +0.19 | **+0.60** |
| convergence | **95.0 % (57/60)** | 81.7 % (49/60) | **80.0 % (48/60)** |
| non_finite_flux errors | (few) | 11 | **12** |
| joint-clean found at call | 11/60 | 57/60 | **3/60** |

Two things this comparison makes visible that neither predecessor's own
report could see in isolation:

**(a) The CBC relief the 3-type run reported (§3/§R3 there) is NOT a seeding
artifact.** This control has the *same* donor-enriched elite pool the 3-type
run had, stays 2-type, and its CBC pass rate (20.8 %) and median (1615.19)
sit essentially where the *original* 2-type run's did (28.1 %, 1614.63) — not
where the 3-type run's did (59.2 %, 1596.06). If the CBC relief were a
seeding effect, it should show up here too. It does not. **This is
independent, unconfounded support for the 3-type campaign's R3 claim that
grading — not better seeding — relieves boron.**

**(b) The reported sign-flip of r(F_r, CBC) is NOT 3-type-specific, and the
3-type report's framing of it needs a correction.** The 3-type run measured
r = +0.19 and read the flip (from the original's −0.45) as evidence the third
type "genuinely" changes the F_r/CBC trade-off. This control — still 2-type —
measures **r = +0.60, a stronger positive flip than the 3-type run's own**.
The flip tracks the seeding/model change, not the alphabet. **This does not
undermine the 3-type report's §3 mid-fraction correlations**
(r(mid, F_r) = −0.42, r(mid, CBC) = +0.64 — those are computed *within* the
3-type run's own 49 same-seeding cores and do not depend on this control at
all), but the *aggregate* "sign flipped, therefore grading changed the
trade-off" argument built on top of the r(F_r,CBC) comparison is weaker than
reported: seeding alone reproduces (and exceeds) the flip.

**(c) Convergence dropped even without a third type.** 80.0 % here is *worse*
than the 3-type run's own 81.7 %, and both are well below the original's
95.0 %. The 3-type report flagged its own convergence drop as "unexplained."
This control shows the drop is not 3-type-specific either — it tracks the
seeding/model change. The proximate mechanism is visible in the label
provenance: 39/60 slots in this run were `exploit/local` (a tightly-packed
hill-climb around donor-enriched elites, same dynamic the 3-type run's design
note flagged at 41/49), which plausibly pushes more candidates toward
marginal, harder-to-converge board regions. Not proven here, but the
convergence drop can no longer be filed under "third fresh type" — it
predates the third type.

---

## 4. What this run does and does not settle

* **Settles:** the 3-type campaign's −0.0364 headline is genuinely
  confounded, and the confound is now measured, not just flagged. Grading
  keeps roughly half the credit (−0.0179 of −0.0364).
* **Settles, unregistered bonus:** the CBC relief (§3 of this doc) is a real
  grading effect, independently supported.
* **Complicates, does not settle:** the F_r/CBC correlation-sign argument
  the 3-type report used as *secondary* support for its R3 mechanism claim.
  The primary support (within-run mid-fraction correlations) is untouched.
* **Does not settle:** model (s1g vs s1h) vs seeding-pool content as two
  separate causes within this control's own −0.0185. Both changed together
  here, deliberately (prereg §4) — s1g cannot be run against this store
  representatively (it predates the schema this pool's rows were scored
  under in the 3-type comparison), so isolating them needs a fourth,
  not-yet-run deck (s1h + original backfill pool, or s1g + donor pool) if
  that split is ever wanted.
* **Does not settle:** deliverability — no pin-BU was measured or predicted
  here (`minfr_pin_bu_limit` screens the search but `max_pin_burnup` is null
  on every row, same as every `optimize`-path campaign to date).

---

## 5. Honest notes

1. `e_split` is NaN on all 60 rows — the same non-issue every `optimize`-path
   campaign shows (only the `produce` path fills it); not a defect here
   specifically.
2. The wave online-update gate read `objective+`/`objective-` throughout
   (waves 0–4 improving-skill, 5–7 not) rather than the 3-type run's fully
   blind `explore`/NaN — this case has a large in-cell holdout
   (`_case_store_rows` returns 129+ rows, growing as this run's own converged
   output lands), unlike the 3-type case's zero. Not a registered readout,
   noted for completeness.
3. `no_improve` reached 6 by wave 6 — the search had genuinely stalled after
   the wave-0 elite pool handed it the eventual joint-clean winner; the
   remaining ~50 calls found nothing better on that axis. Budget 60 was not
   undersized for *this* run, unlike both predecessors' own "frontier still
   descending at call 60" caveats.

---

## 6. Provenance

| item | value |
|---|---|
| deck | `fpcamp_minfr_HGD569_f125_SEEDCTL_199.inp` sha256 `8E5D3A75F4200797F4C6A59D4022A8F665C980921A54F1C44D8A8C8AAC24B341` |
| model | `data/models/s1h` (v7, cond_schema v7) |
| case | `P6253Z1G06N24_P6253Z2G10N24`, feed 125, paramA |
| random_seed | 5696 |
| store on 199 before run | sha256 `4BE89C61856CCD2B41DDDC9B546B1217AC95BDBF019225CDAFFFFE73C5DA8ADD`, 74,537 rows, pair converged 129 (73 f125 + 56 f109) — **not reshipped**, already carried the required donor rows |
| store on 199 after run | sha256 `6815035438B4CD4AE2C21087159A4D98B673427DB40FEE8AF3A43CA18780D240`, 74,597 rows |
| local canonical before / after | 74,537 → **74,597** (+60) |
| store backups | `data/store/records.parquet.bak_pre_SEEDCTLf125_20260817`, `data/store/maps.npz.bak_pre_SEEDCTLf125_20260817` |
| merge | `lpopt merge-store` — 60 new / 0 upgraded / 74,537 duplicate, 144 maps merged, +0 ledger (dry-run inspected first, then real) |
| run dir (199) | `C:\Users\USER\lpopt_work\kit_frontier\runs\fpcamp_minfr_hgd569_f125_seedctl` |
| harvested | scratchpad `harvest_seedctl_f125/{run/labels.jsonl, run/state.json, run/waves/, run/report.md, data/store/}` + `fpcamp_minfr_hgd569_f125_seedctl_out.log` |
| launch/status scripts | `launch_fpcamp_HGD569_f125_SEEDCTL_199.ps1`, `status_fpcamp_HGD569_f125_SEEDCTL_199.ps1`, `run_fpcamp_minfr_HGD569_f125_SEEDCTL_199.bat` |
| fleet | 199 only; 198 / 181 / 238 untouched |
