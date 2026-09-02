# Pre-registration — F_xy-era coverage-fill produce round r1 (box HOST_199)

**Date** 2026-08-29 · **Deck** `produce_fxyera_r1_199.inp` · **Run dir** `runs/produce_fxyera_r1`
**Mode** `produce` (DoE, objective-neutral) · **Library** paramA (single-resolver rule)
**Store at write time** `data/store/records.parquet`, 74,657 rows / 66,078 converged
**Status** WRITTEN BEFORE THE DECK WAS HASHED. Hashes pinned in §9. Nothing launched.

---

## 0. One-paragraph statement

The optimisation target changed from `F_r` to `F_xy` on 2026-08-29. `F_xy` is MASTER's
`FXYP` ("MAXIMUM PIN PLANAR POWER"), hard limit 1.65. `FXYP` is printed **only in
`MAS_OUT`** — `MAS_SUM`'s `SUMMARY EDIT 3` carries `AO / FQN / FRN / FQP / FRP` and has no
`FXYP` column, and `MAS_SUM` is the only file the canonical parser reads
(`lpopt/vendor/masterrl/master.py:406-418`). Consequently **the entire 74,657-row store has
zero `f_xy` labels**, and for almost all of it the `MAS_OUT` that could supply them was
deleted at chain end. This round is therefore two things at once: (a) the next instalment of
the standing 2026-07-25 coverage-fill directive, executed through the objective-neutral
`produce` path, and (b) **the first `f_xy` label batch the programme will ever hold** — every
converged chain's final equilibrium `MAS_OUT` is retained on disk so the F_xy parser now being
designed (`data/reports/fxy_switch_design_20260829.md`) can retro-label it without spending a
single extra MASTER cycle.

---

## 1. Why this round, and why `produce` rather than a campaign

The 2026-07-25 user directive is standing and explicit: *"노심 계산 결과에 무관하게 다양한
데이터셋 생성이 학습에 유리 → 스케쥴링해 빈 영역을 채워라 … 목적함수 탐색이 아니라 커버리지
기반 공백 충전."* The instrument for that is `lpopt produce`, not `optimize`; `flat_power` /
`min_fr` decks are objective-biased by construction and are explicitly excluded.

There is a standing counter-argument on file and it is addressed, not ignored.
`scoping_mesh_20260815/README.md` §9.5 concludes *"다음 생산은 같은 셀에 더 붓기보다 저장전량
저-F_r 영역(1.50–1.55)을 직접 겨냥해야 한다"*, and `reloadmap_methodology_20260816.md` §4 (P2)
holds that direct MASTER spend should be per-cell **min-F_r optimisation**, not production
sampling. Both were written in the F_r era, when the labels a produce round buys (cyclen,
F_r, CBC, maps) were already dense. In the F_xy era that premise no longer holds: the
quantity we are now optimising has **no labels at all**, at any F_r, in any cell. A broad
retained-`MAS_OUT` round is the cheapest possible way to obtain the first population of them,
and it costs nothing that a min-F_r campaign would otherwise buy — frontier-pushing stays with
the `optimize` campaigns (TRIPLE r2 is live). The §9.5 instruction is nevertheless honoured
inside this round: **arm B (26.3% of the budget) is sited directly on the F_r 1.50–1.60 band**
(see §3), which §2 shows *is* the F_xy = 1.65 wall.

---

## 2. The measurement that sets the round's design: where is F_xy = 1.65?

Nothing in the store answers "what is F_xy?", so it was measured directly from retained
MASTER work dirs still on the coordinator disk — 541 pairs of (`MAS_OUT`, `MAS_SUM`) from
`runs/produce_run_20260725_135935` (399, the 2026-07-25 paramA fill round, final cycles,
`keep_success` forced by `harvest_maps`) and `runs/fpcamp_minfr_T6T4*` (142, the T6_T4/f121
min_fr campaign). For each dir: `F_xy` = max over burnup steps of the `MAS_OUT` line
`MAXIMUM PIN PLANAR POWER (FXYP)=`; `F_r` = `max(FRP)` over `MAS_SUM` EDIT 3, which is
*exactly* what the store's `f_r` column is (`master.py:411`).

| `F_r` band | n | `F_xy` min | `F_xy` median | `F_xy`/`F_r` min | median | p90 | n with `F_xy` ≤ 1.65 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.50 – 1.55 | 3 | 1.6522 | 1.6524 | 1.0711 | 1.0717 | 1.0840 | 0 |
| 1.55 – 1.60 | 33 | 1.6502 | 1.6820 | 1.0376 | 1.0642 | 1.0807 | 0 |
| 1.60 – 1.65 | 37 | 1.6734 | 1.7070 | 1.0406 | 1.0522 | 1.0726 | 0 |
| 1.65 – 1.75 | 65 | 1.7242 | 1.8061 | 1.0259 | 1.0661 | 1.0787 | 0 |
| 1.75 – 2.00 | 101 | 1.8531 | 2.0841 | 1.0254 | 1.0833 | 1.1137 | 0 |
| 2.00 – 2.50 | 237 | 2.1107 | 2.3806 | 1.0548 | 1.0985 | 1.1381 | 0 |
| ≥ 2.50 | 65 | 3.1839 | 4.0668 | 1.2522 | 1.3754 | 1.4874 | 0 |

**Three findings, all load-bearing for this deck.**

1. **`F_xy` ≤ 1.65 is approximately the iso-constraint `F_r` ≤ 1.55.** At the median ratio
   1.065 the wall maps to `F_r` = 1.549; the ratio spread (1.038–1.081 in the relevant bands)
   puts the equivalent `F_r` in roughly [1.53, 1.59]. The band to over-sample is therefore
   **`F_r` ∈ [1.50, 1.60]** — which is precisely the region §9.5 of the mesh readout asked for
   on independent grounds. The new objective and the old data-gap recommendation agree.
2. **Not one of the 541 measured cores clears 1.65** (minimum `F_xy` = 1.6502, on a core whose
   `F_r` is 1.5553). The new hard limit sits *at* the current frontier, not comfortably inside
   it. Any plan that assumes the F_xy-feasible set is already populated is wrong.
3. **The ratio is not a constant.** It drifts from ~1.04 near the wall to ~1.38 at `F_r` ≥ 2.5,
   so `F_xy` cannot be predicted from the stored `f_r` by a scalar and there is no shortcut
   around producing real labels. (This is an input to the parser design, not a claim about it.)

*Caveats, stated:* the 541 dirs are what happened to survive on the coordinator, not a random
sample of the corpus; the `fpcamp_*` dirs include non-final cycles; only 3 dirs fall in the
1.50–1.55 band. The finding that matters (the wall is near `F_r` ≈ 1.55) is supported by 73
dirs across the 1.50–1.65 range and is robust to the sampling caveat. It is registered here as
a **prediction to be tested by this round's own labels** (§7).

---

## 3. The coverage gaps this round fills

### 3.1 Per-cell counts computed for this pre-registration

From `data/store/records.parquet` (`converged == True`), grouped on
`(library_id, case_pair, feed)`. Store totals: 74,657 rows, 66,078 converged; feed histogram
`97:23 · 101:2946 · 105:285 · 109:4156 · 113:822 · 117:3619 · 121:44292 · 125:4270 · 129:274 ·
133:2813 · 137:239 · 141:2339`.

**Gap 1 — the F_xy boundary band is a feed-121, low-enrichment artefact.** Rows with
`f_r ∈ [1.40, 1.62]` (the F_xy ≈ [1.45, 1.73] neighbourhood of the wall): **4,353 of 66,078
converged rows (6.6%)**, of which **83.2% are feed 121** and **97.1% sit at `e_core` ≤ 5.5**.
Their distribution:

| library | pair | feed | band rows | `f_r` floor | e_core |
|---|---|---:|---:|---:|---:|
| paramA | `T6_T4` | 121 | 896 | 1.4605 | 4.893 |
| ga80 | `E1_E2` | 121 | 590 | 1.4636 | 5.000 |
| 260624 | `C1_C4` | 121 | 410 | 1.5656 | 5.400 |
| 5.8_5.1 | `A01_A02` | 121 | 400 | 1.6036 | 5.480 |
| ga80 | `K1_K2` | 121 | 238 | 1.4902 | 5.200 |
| ga80 | `N1_N2` | 113 | 122 | 1.4653 | 5.400 |
| ga80 | `E1_E2` | 109 | 80 | 1.4787 | 5.000 |
| paramA | `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24` | 125 | 57 | 1.5864 | 5.676 |
| paramA | `T5_T6` | 121 | 22 | 1.5257 | 5.016 |
| paramA | `P6253Z1G06N24_P6253Z2G10N24` | 125 | 10 | 1.6036 | 5.694 |
| paramA | `T3_T4` | 121 | 1 | 1.5329 | 4.760 |

At `e_core ≥ 5.65` — the LEU+ delivery band, i.e. the whole point of the project — **the entire
corpus holds 67 band rows, all at feed 125, none below `F_r` 1.586**. In the F_xy era that is
the single most consequential hole in the store.

**Gap 2 — every high-band paramA cell is starved at the corpus-native feed.** feed 121 is
44,292 of 66,078 converged rows overall, yet across the ten paramA `P*` pairs its per-cell
counts are 22–73 (`P6253Z1G06N24_P6253Z1G08N20` f117 = 22, `P6253Z1G08N20_P6257Z1G06N24`
f121 = 42, `P6661Z1G08N16_P6661Z1G08N20` f121 = 51, `P6656Z1G10N16_P6656Z2G08N12` f121 = 73,
`P6257Z1G06N24_P6257Z1G10N12` f117 = 26). This is the residue of the 2026-07-25 gap map
(`fill_199.inp` header, "f121 is entirely ABSENT from every high band"): the column was opened
but never filled.

**Gap 3 — the paramA feed lattice is step-8, so f113/f129 are structurally empty.** paramA
carries `{101, 109, 117, 125, 133, 141} + 121` only; f113 and f129 are absent for every pair
except `P6257Z1G06N24_P6257Z1G10N12`, which the 2026-08-15 pathfinder opened with 12 rows
each. **18 cells (9 pairs × 2 feeds) remain unopened**, and
`feedgrid_pathfinder_20260815.md` §10.3 states they open with no bootstrap via the same
level-2 `pair_feed` seed proven at e5.94.

**Gap 4 — the record cell has never been run at any other feed.** `T6_T4` holds 1,495
converged rows, all at feed 121, and owns the programme's `F_r` record (1.4605). Whether its
F_xy advantage is a property of the fuel pair or of the feed is unmeasured.

**Gap 5 — pin burnup is unmeasured exactly where the frontier is.** `max_pin_burnup` is
non-null on 0 of `T6_T4`'s 1,495 rows, 5 of the 3-type f125 cell's 107, 5 of hgd569 f125's 121,
0 of hgd569 f109's 56. Pin BU is the binding delivery constraint (limit 80 GWd/tU on the pin
axial peak, `pinbu_definition_20260820.md` §1-§2) and it is advisory in the mesh but a **real
gate at delivery** (§3 of that document). See §6 for how this round addresses it.

### 3.2 What the deck does about each gap

| Arm | Share | Gap addressed | Strata |
|---|---:|---|---|
| A space-filling | 26.3% | Gap 2 | 5 thinnest paramA `(pair, feed)` cells, wide `split_w1`, `random`/`heuristic` |
| B boundary | 26.3% | Gap 1 | the 5 paramA cells whose converged population already lives in the `F_r` 1.50–1.60 band |
| C exploit | 21.1% | Gap 1 + Gap 5 | `elite_perturb` in the 4 current paramA frontier cells |
| D rule-seed | 15.8% | rule applicability | G4 `rule_biased` (`rm1i`) in 3 cells, incl. the first graded-core test of R-03 |
| E adversarial/OOD | 10.5% | Gaps 3, 4 | `T6_T4` @ f109 (feed transfer on the record cell); `P6656Z1G10N16_P6656Z2G08N12` @ f129 (missing column) |

Deliberately **not** in r1: the ga80 half of the map (`N1_N2`/f113, `E1_E2`/f109). `ProduceDriver`
builds exactly one `CaseAssetResolver` per run (`produce.py:_run_library_id` /
`_default_resolver`); a mixed-library deck routes every paramA stratum through
`FEASIBLE_PACKAGE` and kills it — *"the single-resolver rule is not negotiable"*
(`feedgrid_pathfinder_20260815.md` §9). The ga80 half is round **r2**, a sibling deck to be
shipped after r1 drains. paramA was chosen for r1 because it owns four of the five named
frontier cells and the whole `e_core ≥ 5.65` delivery band.

---

## 4. Sampling ratios (external review §7.6)

The review prescribes 25% space-filling · 25% boundary/uncertainty · 20% policy/surrogate
exploit · 15% rule/expert seed · 10% adversarial-OOD · 5% replicate QC.

Mapping onto what the `produce` path can actually express:

| Review class | Realisation here | Design share | Review |
|---|---|---:|---:|
| space-filling | `random` 0.6 / `heuristic` 0.4, `split_w1 = [0.3, 0.7]`, thinnest cells | 26.3% | 25% |
| boundary / uncertainty | **cell siting**, not generator: the cells whose population is already at the F_xy wall | 26.3% | 25% |
| policy / surrogate exploit | `elite_perturb` 0.7 (`elite_objective = "flat_feasible"`) — the only exploit the modelless produce path has | 21.1% | 20% |
| rule / expert seed | G4 `rule_biased` 0.7 with `rule_bias_metric = "rm1i"` (engineering rule R-03) | 15.8% | 15% |
| adversarial / OOD | two distinct out-of-envelope axes, each with an explicit `ood_reason` | 10.5% | 10% |
| replicate QC | **not expressible in `produce`** — see §6 | 0% (→ phase 2) | 5% |

**Ratio robustness.** All 19 strata carry `n_target = 40` and `priority = 120`, so the driver's
round-robin-within-priority fill (`produce.py` module docstring, "fill waves of `workers`
entries from the highest-priority unmet stratum (round-robin within a priority)") rotates
across strata one **wave** at a time. Verified on the dry run: 38 waves, 19 strata, each filled
40/40. If the box is reclaimed early the realised mix stays close to the design mix, because
the rotation is uniform. This is why the arms are expressed as *stratum counts* (5/5/4/3/2) and
not as unequal `n_target`s.

**Why boundary is achieved by siting and not by a generator.** There is no `F_r`- or
`F_xy`-targeted `elite_objective` in the code; the menu is `cyclen` / `flat` / `flat_feasible`
(`produce.py:ELITE_OBJECTIVES`). `flat` orders by `node_peak` ascending, and `node_peak` was
measured **anti**-correlated with `F_r` inside the feasible region (Pearson −0.750,
2026-08-09) — a flat-first parent rule would pull *away* from the F_xy wall. The honest
alternative is to let the cell do the work: `T6_T4`/f121 has 896 of 1,495 converged rows in the
band, `T5_T6`/f121 has 22 of 33, so an unbiased `random`/`heuristic` draw in those cells lands
on the boundary about 60–67% of the time. Adding an F_r-targeted elite objective would be a
code change and is out of scope for this round; it is filed as a follow-up in §10.

---

## 5. Budget — derivation from measured throughput

**Cadence.** Taken from `data/produce/ledger.jsonl`, restricted to the nine stratum names of
`fill_199.inp` and to the window in which only HOST_199 was serving them (2026-07-26 07:38 →
22:17; the 16-wide waves in the ledger before that window are box 198 running `fill_198b.inp`,
which shares those stratum names). In that window: **34 waves of exactly 24 chains**, median
inter-wave interval **24.32 min** (mean 25.92, p10 19.15, p90 31.39), span 14.26 h.

- per-wave rate: 24 / 24.32 min = **59.2 chains/h**
- realised sustained rate over the 14.26 h window: **55.6 chains/h**
- conservative (p90 wave): 24 / 31.39 min = 45.9 chains/h

That measurement is from a **paramA produce run with `harvest_maps = true` and
`[produce] max_cycles = 14`** — the same library, the same harvest setting and the same cycle
cap as this deck — so the paramA cost premium (×1.35 over ga80,
`feedgrid_pathfinder_20260815.md` §6) is already inside the number and is not applied twice.

**Yield.** Converged labels per chain, measured by joining ledger `record_id`s to the store:
the 2026-08-15 feedgrid production run is the clean case (all 450 chains matched a store row)
and gives **305/450 = 0.678**. The `fill_*` strata give 930/2,148 = 0.433 as a hard lower bound
(953 of those chains have no store row, most likely the 198b half never merged) and
930/1,195 = 0.778 as the upper. The pathfinder's own guidance is "plan N/0.75 ≈ 1.33 N".
**0.68 is used.**

**Sizing.**

| quantity | value |
|---|---:|
| target converged labels (19 strata × 40) | 760 |
| chains at 0.68 converged/chain | ~1,118 |
| wall time at 55.6 chains/h | **20.1 h** |
| wall time at the conservative 45.9 chains/h | 24.4 h |
| phase-2 pin-BU + replicate-QC wave (§6), 40 chains at 12 workers | ~1.6 h |
| **total** | **~21.7 h** (20.1–24 h band) |
| hard fence: `--max-chains` in the launcher `.bat` | **1250** |
| retained-`MAS_OUT` disk, at the measured 9.5 MB/chain (5.6 GB / 592 dirs, `runs/produce_run_20260725_135935`) | **~11 GB** |
| launcher free-disk refusal threshold | 30 GB |

---

## 6. Pin burnup and replicate QC — why they are phase 2, not strata

Both were requested for this round. Neither is expressible inside `lpopt produce` without a
code change, which this round is forbidden to make. The reasons are specific:

**Pin burnup.** `max_pin_burnup` is written only when the equilibrium runner is constructed
with `enable_pin_burnup=True`, which turns on the `%EDT_OPT ipin` PPI edit
(`vendor/masterrl/equilibrium.py:461-466` → `burnup.enable_ppi_output`). It is **not a deck
knob**. `WaveVerifier._default_factory` — the factory `produce` and `optimize` both use —
hard-codes `enable_pin_burnup=False` (`lpopt/search/verify.py:851`). The flag is reachable
from exactly two places: `[design].enable_pin_burnup` (bootstrap only,
`design/pathfinder.py:222`) and `curriculum.make_pin_burnup_verifier` (`curriculum.py:1017`).

**Replicate QC.** `produce` dedups on `record_id = sha256(canonical_pattern | library_id |
case_pair | deck_knobs)`, replayed from the ledger *and* the store at start
(`produce.py:_reconstruct`). Re-drawing an existing record's pattern is therefore counted as a
`dup` and skipped — a produce stratum **cannot** re-run a stored pattern by construction.

**Phase 2 does both in one pass, with code that already exists.** `pinbu_wave.py` is the
fixed-pattern replay harness: it takes named store records, replays each one's own pattern
through the same asset resolution, but with `make_pin_burnup_verifier`, and it checks
determinism by requiring the replay to reproduce the stored `f_r` / `cyclen` / `cbc_max` to
`TOL_F_R` / `TOL_CYCLEN` / `TOL_CBC` (a chain that does not is reported
`determinism_ok=False` and its pin value is not merged). The `keep_success = true` variant
deck `pinbu_wave_keep_199.inp` already exists on the kit and retains the final cycle dir, so a
single phase-2 wave yields **all three** products at once:

1. measured `max_pin_burnup` on this round's own converged frontier cores (Gap 5);
2. a MASTER **determinism** measurement — which, because the retained dir now includes
   `MAS_OUT`, is directly a determinism measurement **for `F_xy`** (replay `FXYP` vs the r1
   `FXYP` for the same pattern), i.e. the review's 5% replicate-QC arm;
3. retained `MAS_OUT` for the replayed cores.

**Phase-2 plan (pre-registered, to be executed only after r1 drains):** 40 chains — 20 drawn
from r1's own converged rows nearest the F_xy wall, plus 20 **exact re-runs of r1 records**
(the replicate set). `python pinbu_wave.py plan` is read-only and runs on the coordinator; the
run uses `pinbu_wave_keep_199.inp` at 12 workers; `patch --dry-run` before any store write.
This is registered as part of r1's budget (§5) and its own launcher trio is **not** part of
this deliverable.

---

## 7. Success criteria and falsifiable predictions

**Primary (must all hold, else the round is a failure and is reported as one):**

- **P1 — retention.** ≥ 95% of converged chains leave a readable `MAS_OUT` under
  `runs/produce_fxyera_r1/produce_run_*/master_work/worker_*/*/`, and every retained `MAS_OUT`
  contains at least one `MAXIMUM PIN PLANAR POWER (FXYP)=` line. This is the round's actual
  deliverable; a scalar-label-only outcome is a failed round.
- **P2 — yield.** ≥ 600 converged labels (79% of the 760 target) within 24 h.
- **P3 — no cell dies.** No stratum reports `STALLED = True`, and no stratum's harness-error
  rate exceeds 15%.
- **P4 — routing.** Zero rows with `restart_provenance` starting `unresolved`. (Pre-verified:
  `python val_assets.py produce_fxyera_r1_199.inp` → `RESULT resolved=19 failed=0`, §9.)

**Secondary (measured and reported; failure does not invalidate the round):**

- **S1 — the wall location.** Prediction from §2: among this round's converged cores, the
  `F_xy` / `F_r` ratio in the `F_r ∈ [1.50, 1.65]` region has median in **[1.04, 1.08]**, so
  `F_xy = 1.65` corresponds to `F_r` in **[1.53, 1.59]**. If the measured median falls outside
  that interval the §2 sample was unrepresentative and every boundary-siting decision in §3.2
  must be recomputed before r2.
- **S2 — first F_xy-feasible core.** Prediction: **r1 produces at least one converged core with
  `F_xy` ≤ 1.65**, and it comes from arm B or C in `T6_T4`/f121. (No core in the §2 sample of
  541 cleared it; the minimum was 1.6502.) If r1 produces none, the F_xy-feasible set is empty
  under current fuel and the programme's next move is a fuel-design question, not a loading
  question — that is a decision-grade negative result.
- **S3 — the `flat_feasible` exploit rule.** Prediction: in `T6_T4`/f121, arm C's `F_xy`
  distribution is **not shifted above** arm B's (same cell, unbiased). If it is shifted above,
  the `flat_feasible` parent rule is anti-correlated with `F_xy` (consistent with the −0.750
  `node_peak`/`F_r` result) and must be replaced by an F_xy-targeted `elite_objective` before
  r2 — a code change, filed in §10.
- **S4 — R-03 on graded cores.** `rm_fresh_face_adjacency` has never been exercised on a
  3-fresh-type core. Reported: arm D's `fxy_rule_S3T1S5_f125` converged fraction and `F_xy`
  distribution vs its own 0.3 `random` control share, and vs arm B's 3-type stratum.
- **S5 — feed transfer on the record cell.** `T6_T4` @ f109 vs f121 at matched `split_w1`:
  Δ`F_xy`, Δ`cyclen`, Δ`cbc_max`. First measurement of whether the record cell's advantage is a
  pair property or a feed property.
- **S6 — the `promote()` fix in the field.** Arm E's f129 stratum is the field test. Reported:
  `non_finite_flux` rate vs the 20/89 = 22% measured in the 2026-08-15 f129 column, and whether
  the log shows `[produce] promoted <cell> -> ...` (level-1 cache populated).

**Analysis screens** applied when reading the labels (not applied during production — this is
an objective-neutral round): CBC ≤ 1600 ppm, `F_q` ≤ 2.41, |AO| ≤ 0.30, and `F_xy` ≤ 1.65 once
the parser lands. Pin burnup is **advisory** in any map readout and a **gate** only in a
delivery verdict (`pinbu_definition_20260820.md` §3).

---

## 8. Analysis plan

1. **Merge.** `python -m lpopt merge-store` from the returned kit `data/` folder (staging shape
   `<dir>/store/records.parquet` + `<dir>/produce/ledger.jsonl`). Back up
   `data/store/records.parquet` and `maps.npz` first, to
   `*.bak_pre_fxyera_r1_20260829`, per house practice.
2. **F_xy retro-labelling — the headline product.** Run the F_xy parser
   (`data/reports/fxy_switch_design_20260829.md`, in design) over the retained
   `master_work/**/MAS_OUT` set, key by the chain's `record_id`, and write `f_xy` as a new
   store column. **This is the first `f_xy` label batch in the programme.** Report: n labelled,
   the `F_xy`/`F_r` regression by band (S1), the count clearing 1.65 (S2), and the per-cell
   `F_xy` floor table alongside the existing `F_r` floor table.
   *Retro-labelling is not retroactive beyond this round:* a chain produced without retention
   loses its `F_xy` permanently, which is the whole reason `[master] keep_success = true` and
   `[verify] harvest_maps = true` are both pinned in the deck.
3. **Coverage delta.** Re-run the §3.1 tables and report the change in band rows per cell, in
   the `e_core ≥ 5.65` band count (currently 67), and in the paramA f113/f129 cell count
   (currently 18 unopened → 17 if arm E succeeds).
4. **Arm comparison.** Per arm and per cell: converged fraction, `F_xy` distribution, `F_r`
   distribution, CBC distribution, `node_peak` / `map_cov`. S3 and S4 are read here.
5. **Transfer readout.** S5, and the `ood_reason` tagging for arm E rows.
6. **Phase 2.** `pinbu_wave.py plan` → `run` (deck `pinbu_wave_keep_199.inp`) → `patch --dry-run`
   → `patch`. Report the determinism table including the new `F_xy` column (§6.2) and the
   delivery verdicts against 80 GWd/tU.
7. **Retrain hook.** Labels enter the next incremental split via `build_split_S1b.py`'s
   parameterised form (parent assignment preserved, increment-only stable-hash assignment).
   No retrain is pre-registered here.

---

## 9. Frozen artefacts and hashes

Written **after** this document, in this order: deck → `lpopt check` → `val_assets.py` →
`produce --dry-run` → hash → launcher trio.

| artefact | sha256 |
|---|---|
| `produce_fxyera_r1_199.inp` | `6B72DDA2DFDA3124327CEB3B17D07EBB5B58281426BD7C6F15B5ABB7C57CD380` |
| `data/store/records.parquet` (the store the kit must carry) | `D2196B5EC0F53D59432DA071DC063CA35FB54BA832BA2A0B0356A5D9535F4B0F` |
| `data/store/fuel_types.parquet` | `FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137` |

**Local validation performed (no MASTER, nothing launched):**

- `python -m lpopt check --input produce_fxyera_r1_199.inp` → `11 PASS, 11 FAIL, 1 SKIP`.
  The 11 FAILs are **byte-identical to those of `fill_199.inp`, `newfeed_198.inp` and the live
  `fpcamp_minfr_TRIPLE_f125_r2_199.inp`** (all three give the same `11 PASS, 11 FAIL, 1 SKIP`).
  Every FAIL is `verify.template` on a `cores/*/bootstrap/MAS_INP_cy02.inp` — a **cycle-1
  bootstrap deck** being length-checked as if it were a reload template (`%LPD_B&C has 14/40
  rows (< 80)`). It is a pre-existing property of `check` against the paramA package layout,
  not a defect of this deck, and it is not on the produce path (which uses the resolver, see
  next line).
- `python val_assets.py produce_fxyera_r1_199.inp` → **`RESULT resolved=19 failed=0`**; every
  stratum resolves restart **Y** and template **Y** through the same single resolver `produce`
  will build. Levels: 5 native (`T6_T4`, `T5_T6`, `T1_T4`, `T3_T4`), 1 level-2 `pair_feed`
  (`fxy_ood_T6T4_f109` → the pair's own f121 restart, exactly as designed), 13 level-3
  `pair_ecore` off `bases/{P0_P1, Q1_Q2, Q7_Q8}`.
- `python -m lpopt produce --input produce_fxyera_r1_199.inp --dry-run --max-chains 900` →
  `760 chains (760 converged / 0 nonconverged / 0 errors), 0 dedup skips`, **38 waves, all 19
  strata 40/40**. `rule_biased` drew without starving. The `elite_perturb → random` fallback
  count (103) is a known dry-run artefact: `--dry-run` uses a run-scoped empty store
  (`produce.py:370`), so no elite parents exist. The **real** pools were measured directly
  against the canonical store under this deck's `[acquisition]` gates (CBC ≤ 1600 / `F_q` ≤ 2.41
  / |AO| ≤ 0.30): `T6_T4` **1,453**, 3-type f125 **55**, hgd569 f125 **25** (63 pair-wide),
  hgd569 f109 **38** (63 pair-wide). No exploit stratum starves. The dry-run output directory
  was deleted afterwards so `runs/produce_fxyera_r1` is created fresh by the launcher.
- `pytest tests/test_config.py tests/test_produce_ledger.py tests/test_paramA_produce_kit.py -q`
  → see §11.

---

## 10. Registered risks, and follow-ups this round may not perform

| # | risk / gap | disposition |
|---|---|---|
| R1 | No `F_r`/`F_xy`-targeted `elite_objective` exists; arm C uses `flat_feasible`, whose ordering key is anti-correlated with `F_r` (−0.750). | Measured as S3. If S3 fails, add an `f_xy` elite objective (code change) before r2. |
| R2 | f129 ran 20 `non_finite_flux` of 89 chains (22%) in 2026-08-15 because its only seed was an f125 restart four feed-steps away. | Arm E is 1 of 19 strata (5.3% of budget). `resolver.promote()` is now wired (`produce.py:1320-1334`); S6 is its field test. |
| R3 | Retained `MAS_OUT` costs ~11 GB; HOST_199 had ~53 GB free at the 2026-08-16 archive sweep and has run several campaigns since. | Launcher refuses below **30 GB** free and prints the figure; `status_*` reports free space every probe. |
| R4 | The §2 F_xy measurement rests on 541 opportunistically retained dirs, only 3 of them in `F_r` 1.50–1.55. | Registered as prediction S1, to be tested by this round's own labels. |
| R5 | ga80 frontier cells (`N1_N2`/f113, `E1_E2`/f109) are not in r1. | Single-resolver rule. Filed as round r2 (sibling deck, same structure, `library_id = "ga80"`, `[verify] package_root = "FEASIBLE_PACKAGE"`). |
| R6 | `produce` cannot express replicate QC (dedup by `record_id`) or pin BU (`verify.py:851`). | Phase 2 via the existing `pinbu_wave.py` + `pinbu_wave_keep_199.inp`. §6. |
| R7 | The kit's store may predate 2026-08-29; a stale store starves the elite pools and mis-dedups. | Launcher **store sha256 gate** (§9 table). It refuses rather than running on the wrong store. |
| R8 | `MAX_FRESH_TYPES` is documented as 3 in `tripletype_design_20260817.md` but the code says **5** (`lpopt/search/genome.py:88`). | Code is authoritative. The deck's only graded case is 3-type, well inside either value. |

---

## 11. Operator instructions

Nothing in this round has been launched. The coordinator must, in order:

**Step 1 — ship (coordinator → HOST_199 kit `C:\Users\USER\lpopt_work\kit_frontier`).** The kit
already carries the code (2026-08-20 build), champion `data/models/s1i`, and
`design/package` (1379 MB); only these five files are new or must be refreshed:

```
produce_fxyera_r1_199.inp
launch_produce_fxyera_r1_199.ps1
run_produce_fxyera_r1_199.bat
status_produce_fxyera_r1_199.ps1
data/store/records.parquet          (+ data/store/fuel_types.parquet if stale)
```

Transfer with binary mode and **do not edit any of them on the remote box** — PowerShell's
UTF-8 writes a BOM and a BOM corrupts the TOML deck (2026-08-12 lesson). The launcher's
sha256 gates exist to catch exactly that.

**Step 2 — confirm the box is genuinely idle.** Remote `tasklist` shows a blank owner column
over ssh, so ownership must be confirmed interactively by the user before any launch
(non-intrusion rule, 2026-07-25). The launcher additionally refuses on its own if any
`master4.0m4_r1` process or any `lpopt|ablation|batchswap|mesh` python process is alive.

**Step 3 — launch (the ONLY command that starts MASTER):**

```
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_produce_fxyera_r1_199.ps1"
```

**Step 4 — monitor (read-only, starts nothing):**

```
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_produce_fxyera_r1_199.ps1"
```

`HOST_199` is the box's identifier in this document; the scripts themselves carry the literal
host string used by the existing `*_199` launcher family.

---

## 12. Test evidence

`pytest tests/test_config.py tests/test_produce_ledger.py tests/test_paramA_produce_kit.py -q`
— result recorded in §12.1 below at pre-registration time.

### 12.1

```
$ python -m pytest tests/test_config.py tests/test_produce_ledger.py tests/test_paramA_produce_kit.py -q
........................................................                 [100%]
56 passed in 9.28s
```

Per file: `tests/test_config.py` **24 passed** · `tests/test_produce_ledger.py` **25 passed** ·
`tests/test_paramA_produce_kit.py` **7 passed**. **56 passed, 0 failed, 0 skipped.** No code
under `lpopt/` was modified by this round; the suite is the unchanged 2026-08-20 build.
