# Measured pin-burnup wave — PRE-REGISTRATION (2026-08-20)

Registered **before** any MASTER call. Machine-readable twin with every pinned
number: `data/reports/pinbu_wave_prereg_20260820.json`
(`schema: pinbu_wave_prereg_v1`, 44 targets, pinned against
`data/store/records.parquet` @ 22,196,891 bytes and champion `data/models/s1i`).

---

## 1. The gap this wave closes

`max_pin_burnup` is written only when the equilibrium runner is constructed with
`enable_pin_burnup=True`, which turns on the `%EDT_OPT ipin=1` MAS_PPI edit
(`vendor/masterrl/equilibrium.py:461-466` → `burnup.enable_ppi_output`). That
flag is reachable from exactly two places:

| path | where |
|---|---|
| `[design].enable_pin_burnup` | `design/pathfinder.py:222` — bootstrap only |
| `curriculum.make_pin_burnup_verifier` | `curriculum.py:1017`, sets it at `:1097` |

`lpopt optimize` uses neither: `WaveVerifier._default_factory` hard-codes
`enable_pin_burnup=False` (`search/verify.py:851`). **Consequence, verified in
the store: all 2,270 `fpcamp_*` rows have `max_pin_burnup` null, and feed 113
has zero measured pin burnup anywhere in the 74,597-row store.**

Two live decisions rest on a predicted number from a head that
`data/reports/pinbu_forensics.md` shows has ~0 within-cell held-out rank skill
(§A1.2, held-out Spearman −0.06…+0.64, prediction spread collapses to σ 2.2
against an actual 5.1):

* **(a) DELIVERY.** The four opened-cell winners are predicted-only against the
  LEU+ 80 GWd/tU peak-pin limit. Neither ga80 deck even carried a pin gate
  (`minfr_pin_bu_limit` first appears in the 2026-08-17 hgd569 deck), so their
  80-limit status is a retrospective prediction.
* **(b) LOW-FEED MAP VERDICTS.** `mesh_multitype_20260818/README.md` §5.1
  reports 7 low-feed cells with no joint-clean candidate, attributes it to the
  pin axis (our in-band minimum predicted pin 76.3–83.7 against a 78 gate, while
  CBC floors sit at 1110–1339 against a 1600 gate), and compares against DB
  reference cores at 70.5–74.3 actual — "our pin prediction is up to ~9 GWd/tU
  pessimistic in the low-feed band". §5.1 freezes the low-feed verdicts until
  the head is recalibrated.

## 2. The measurement path (built for this wave)

`pinbu_wave.py` — a fixed-pattern re-evaluation harness. It replays a stored
record's **own pattern** through the **same asset resolution its campaign used**,
but builds the verifier with `curriculum.make_pin_burnup_verifier` instead of the
default. `fr_arms.py` is the fixed-pattern precedent (store pattern → `WaveEntry`
→ `WaveVerifier.evaluate_wave`); `make_pin_burnup_verifier` is the pin-burnup
precedent. Deck: `pinbu_wave_199.inp` (one deck, both libraries — the resolver is
built per `library_id` and the verifier derives `package_root` + `%GEN_DIM` dims
from it, `curriculum.py:1060-1065`).

**One chain per core.** MASTER is deterministic on this fleet (the `fr_arms` A0
control reproduced F_r to 0.0000) and `%EDT_OPT ipin` is an EDIT flag — output
only. Replication buys nothing; the determinism check below buys everything.

## 3. Registered controls (these are the falsifiers)

| control | rule | if it fires |
|---|---|---|
| **determinism** | every chain must reproduce its stored labels: \|Δf_r\| ≤ 0.002, \|Δcyclen\| ≤ 0.5 EFPD, \|Δcbc_max\| ≤ 2.0 ppm | `determinism_ok=False`; the pin value is **refused at merge** — it measured a different evaluation than the store row records |
| **restart provenance** | the re-run must resolve the same restart the campaign recorded (`pair_feed:…0677.23` / `…0615.88` for ga80, `pair_ecore:…0705.02` for paramA) | refused at merge |
| **record identity** | `compute_record_id(pattern.canonical(), library_id, pair, PRODUCE_DECK_KNOBS)` must reproduce the planned id, checked **before** the chain runs | core skipped, zero spend |
| **free skill readout** | f_q / ao_abs / max_assembly_burnup are also re-measured; they were never at issue, so any drift is a harness fault, not a result |

## 4. Target set — 44 cores, predictions pinned

### 4a. Delivery (20 cores): top-5 joint-clean per campaign

Joint-clean = `cbc_max ≤ 1600 ∧ f_q ≤ 2.41 ∧ |ao| ≤ 0.30` (all four decks), plus
`f_r ≤ 1.55` where the campaign reached it. The two paramA cells never got below
1.55 (best 1.6088 / 1.5956 raw), so F_r is their ranking objective, exactly as
their result reports rank them.

| group | winner | stored F_r | **s1i predicted pin** (top-5 range) | predicted verdict vs 80 |
|---|---|---|---|---|
| `N1N2_f113` (ga80, N1_N2/113) | `c9bc21e9513b` | 1.4961 | **85.86 – 87.20** | 5/5 **FAIL** |
| `E1E2_f109` (ga80, E1_E2/109) | `3153f3b39eff` | 1.4787 | **82.36 – 83.77** | 5/5 **FAIL** |
| `HGD569_f125_2type` (paramA) | `8b9acbcda6c7` | 1.6357 | **75.06 – 75.99** | 5/5 PASS |
| `HGD569_f125_3type` (paramA) | `852d233d5421` | 1.5993 | **74.90 – 75.13** | 5/5 PASS |

### 4b. Calibration (24 cores): the low-feed pin-pessimism cells

The 7 cells §5.1 flagged, plus E1_E2/f109 (the e5.0 delivery cell, the only large
low-predicted-pin population in the band). Each cell is cut into equal-width
predicted-pin bins over its own pinned window and the **lowest-F_r** core in each
bin is drawn (declared selection: binning fixes the span of the regressor, so the
within-bin choice is spent on the region the verdicts live in — see §7).

| cell | n | predicted-pin span drawn | stored F_r span | measured pin already in store |
|---|---:|---|---|---|
| `N1_N2` / f113 | 7 | 79.44 – 93.19 | 1.51 – 1.90 | **0** |
| `L1_L2` / f113 | 3 | 81.44 – 91.06 | 1.75 – 1.91 | **0** |
| `G3_G4` / f113 | 3 | 81.67 – 92.63 | 1.65 – 1.75 | **0** |
| `E1_E2` / f109 | 5 | 77.31 – 86.36 | 1.51 – 1.90 | 33 (all F_r ≥ 1.80) |
| `K1_K2` / f109 | 3 | 82.96 – 88.17 | 1.73 – 1.78 | **0** |
| `L1_L2` / f109 | 1 | 79.55 | 2.07 | 34 (all F_r ≥ 1.73) |
| `N1_N2` / f109 | 1 | 85.30 | 1.82 | 40 (all F_r ≥ 1.70) |
| `G3_G4` / f109 | 1 | 85.96 | 1.83 | 50 (all F_r ≥ 1.83) |

Budget asymmetry is deliberate: **13 of 24 slots go to f113**, where the store
holds no measured pin at all, so every chain there is a first. The three f109
cells that already hold 34–50 labels get one core each, placed at the low
predicted-pin end their existing labels never reach.

## 5. Registered prior (this is what makes the readout falsifiable)

The store **already** holds 157 measured pin labels at four of these cells, from
the curriculum-cell produce campaigns. Scored with s1i (read-only, before this
wave):

| cell | n measured | measured pin range | **bias (pred − meas)** | MAE |
|---|---:|---|---:|---:|
| `L1_L2`/f109 | 34 | 75.0 – 96.5 | **−1.22** | 1.63 |
| `N1_N2`/f109 | 40 | 81.6 – 97.3 | **−0.62** | 1.33 |
| `G3_G4`/f109 | 50 | 81.3 – 102.5 | **−0.91** | 1.32 |
| `E1_E2`/f109 | 33 | 76.1 – 93.6 | **−1.29** | 1.59 |

Every one of those rows is at F_r ≥ 1.70 — the unoptimized region, inside the
training support. **On that support the head is not 9 GWd/tU pessimistic; it is
~1 GWd/tU optimistic.** So §5.1's "~9 GWd pessimistic" cannot yet be read as a
head bias: it is a comparison of *our* cores against *different* DB cores
(e.g. our `N1_N2` against DB `N3_N4`). Two hypotheses, separated in advance:

* **H1 — head bias.** The head over-predicts pin on optimized low-feed cores.
  Predicts wave bias ≈ **+9** (pred − meas) on the calibration set.
* **H2 — pool deficit.** The head is roughly unbiased and our candidate pool is
  genuinely worse on pin than the DB's reference cores. Predicts wave bias ≈
  **−1 ± 2** (the prior above, carried out of support).

## 6. Decision rules — fixed now, before the data

1. **Delivery verdict, per core:** measured `max_pin_burnup` ≤ 80.0 → PASS,
   > 80.0 → FAIL. Reported per group as *k/5 PASS* plus the winner's own verdict.
   A core whose chain fails any §3 control has **no verdict** — never a
   substituted prediction.
2. **Pin-head accuracy:** bias = mean(pred − meas), MAE = mean\|pred − meas\|,
   reported **per feed** (109 / 113 / 125) and pooled, with n.
3. **H1 vs H2:** decided on the calibration set's per-feed bias with a
   bootstrap 95% CI. CI excluding +9 and containing 0 ± 2 → **H2** (the low-feed
   map closure is a search/design result, not a model defect, and §5.1's
   pessimism claim must be withdrawn as a cross-core artefact). CI containing +9
   → **H1** (recalibrate before any low-feed verdict). CI spanning both → the
   wave is underpowered; report so and claim neither.
4. **Recalibration recommendation:** if \|pooled bias\| > **2.0 GWd/tU** — the
   model margin the decks already spend (`minfr_pin_bu_limit` 78 = 80 − 2) — then
   recommend fitting `lpopt/model/pinbu_physics.py::fit_pinbu_physics` for the
   next champion (**note: no champion s1c…s1i ships a `pinbu_physics.json`; the
   served pin column is the raw head today**). If \|bias\| ≤ 2.0, recommend
   leaving the serve path alone and record the measured MAE as the margin's
   empirical basis.
5. **No verdict is upgraded by this wave on any axis other than pin burnup.**
   F_r/CBC/cyclen re-measurements are determinism controls, not new results.

## 7. Declared limitations

* The within-bin draw selects on **F_r**, deliberately (§4b). It is legitimate
  because the regression is measured-pin **on predicted-pin** and the bins, not
  the F_r ranking, set that regressor's span — but it means the calibration
  estimates the residual *in the operating region*, and does not claim to be the
  cell-average residual.
* Five of the eight calibration cells have no store core below predicted pin ≈
  79, so the 74–79 sub-band is covered only by `E1_E2`/f109 and the two paramA
  delivery groups. A calibration curve below 77 is not claimed.
* `mesh_multitype` §5.1 assumes DB `pinmax_rodavg_GWd` and our `max_pin_burnup`
  share a definition. That assumption is **not** tested by this wave — this wave
  compares our prediction against our own MAS_PPI measurement, on the same core.
* Recomputing §5.1's DB range gives 66.91–74.27, not the quoted 70.5–74.3 (the
  66.91 outlier is `e5.4_f113`). Flagged; not load-bearing here.

## 8. Spend

44 chains, box **199** (idle at registration: `master=0 python=0`; 24 logical
cores, 57.8 GB free RAM, 44.3 GB free disk). 12 pinned workers → 4 chunks. The
comparable campaigns ran ~20 min/chain at 8 workers *while also finetuning a
model*, so ≈1.5 h expected. Boxes 198 / 181 / 238 untouched. No run dir other
than `runs/pinbu_wave` is written; the canonical store is read-only until the
separate `pinbu_wave.py patch` step, which backs up first and patches
`max_pin_burnup` / `max_assembly_burnup` **in place by record_id**
(`lpopt/tools/backfill_flatness.py` precedent) — never `write_records`, which
would replace the whole row, and never `lpopt merge-store`, which classifies an
equal-rank duplicate as `duplicate` and writes nothing (`multi_pc.py:1374`).
