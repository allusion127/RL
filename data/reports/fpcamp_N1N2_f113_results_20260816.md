# Low-feed opening campaign — N1_N2 / feed 113 / ga80 — RESULTS

**Run 2026-08-16 on box 199.** Deck `fpcamp_minfr_N1N2_f113_199.inp`
(sha256 `d93ecff3343959762b6ef80dd067ea98f14cd3901d903e2792e32d0e8782734d`),
run dir `runs/fpcamp_minfr_N1N2_f113`, champion `data/models/s1f`, seed 1201,
`rc=0`. Pre-registration is the deck header itself, written and hashed before
launch; nothing below revises a mark that was set after seeing a result.

---

## 1. RESULT

| | |
|---|---|
| **Best feasible F_r** | **1.4961** @ cyclen **641.635 EFPD** |
| record_id | `c9bc21e9513b86235d1c8a1f81da926ac71f840247ebaa5f87188b336f8c5eca` |
| CBC_max / F_q / \|AO\| | 1415.54 ppm / 1.8152 / 0.0452 |
| node_peak / map_cov | 1.2739 / 0.2476 |
| n_cycles / restart | 11 / `pair_feed:MAS_RST.APRQ_11_0677.23` |
| fresh composition | **N1 ×77, N2 ×36** (113 fresh, split 0.681) |
| **Feasible cores found** | **41 / 100** MASTER calls |
| Converged | **98 / 100 (98.0%)** — 2 `non_finite_flux` |
| Pareto front size | **6** (4 feasible, 2 infeasible) |
| Waves | 13 (12 + 4-call reserve), budget 100/100 |

### Against every registered mark

| mark | value | campaign best | verdict |
|---|---|---|---|
| **PRIMARY** — any feasible core at this cell | none existed | **41 found** | ✅ **MET** |
| F_r ≤ 1.55 licensing gate | 1.55 | 1.4961 | ✅ **−0.0539** |
| **SECONDARY** — DB truth, best core in this cell | 1.5374 | 1.4961 | ✅ **−0.0413** |
| DB feasible band, this cell | 1.5374 – 1.5499 | 1.4961 | ✅ below the whole band |
| DB best N1_N2 core at **any** feed | 1.5024 (f105) | 1.4961 | ✅ **−0.0063** |
| Cell produce floor | 1.7243 | 1.4961 | ✅ **−0.2282** |
| Scoping-mesh bias-corrected floor | 1.5960 | 1.4961 | ✅ −0.0999 |
| N1_N2 store record, any feed | 1.4932 (f121, cy 673.3) | 1.4961 | +0.0029 (not beaten) |
| ga80 program incumbent | 1.4636 (E1_E2/f121) | 1.4961 | +0.0325 (not beaten) |

Both the PRIMARY and SECONDARY criteria in the pre-registration are met. The
NULL and PARTIAL readings do not apply.

### One comparison worth stating separately

The **global minimum F_r over the entire 6113-core verified database**
(`eq_ok ∧ feasible_at_metrics`, all pairs, all feeds) is **1.4958**, at
N3_N4/feed-101, EFPD 603.7. This campaign's core is **1.4961 — 0.0003 above
that global minimum, at 641.6 EFPD, i.e. +37.9 EFPD of cycle.** Stated
carefully: this is one core against a database built on different cells, and
0.0003 is far inside anyone's convergence tolerance, so the honest claim is
*parity with the best core in the database at a substantially longer cycle*,
not a record.

---

## 2. What this settles

The scoping mesh (`data/reports/scoping_mesh_20260815/`, `cell_verdicts.csv`
row 29) classified this exact cell **pool-starved** and predicted:

| mesh prediction | measured |
|---|---|
| `mesh_min_pred_f_r` 1.6226 | **1.4961** |
| `mesh_n_feasible` **0** | **41** |
| `corrected_floor` 1.5960 | **1.4961** |
| `gap_data` +0.1869, "unmoved by two champion swaps" | **closed and inverted** |

The mesh's own diagnosis named this campaign's mechanism as the culprit —
*"모자란 것은 셔플 카드(장전 배치) 자체이고, 그 이유는 엘리트 부모가 feed 121
패턴을 feed-morph 해서 만들어졌기 때문이다"* (what is missing is the loading
arrangement itself, because the elite parents were made by feed-morphing
feed-121 patterns). That conclusion was drawn from a **model-only** mesh: 1200
predictions, zero MASTER calls, no wave-to-wave learning.

**The mechanism was not the problem; the absence of in-cell labels was.** The
same cross-feed transfer, given 100 MASTER labels and a per-wave finetune,
produced 41 feasible cores. The registered fallback reading — "the cell needs
its own retrain loop (r3/r4 pattern)" — is **not** required.

Two structural facts explain why the transfer was cheaper than it looked:

1. **The morph is small.** In the octant genome, feed 121 → 113 is
   `n_fresh 30 → 28` — a **2-unit edit**, not "remove 8 assemblies". f125 → 113
   is 3 units. All 32 elite parents morphed cleanly (pre-flighted before launch).
2. **The answer was not the DB's answer.** The winner sits at split **0.681**
   (N1 ×77 / N2 ×36); the DB's best core in this cell sits at **0.540**
   (N1 ×61 / N2 ×52). The campaign did not re-find the database's core — it
   found a different, better one. This is consistent with the readout's
   observation that DB optima spread over splits 0.50–0.76.

---

## 3. (cyclen, F_r) Pareto front

Non-dominated on F_r ↓ / cyclen ↑ over the 98 converged rows. All FOMs were
recorded, so this front is available for a multi-objective re-read if the
definition of the optimum changes.

| F_r | cyclen | CBC_max | F_q | \|AO\| | node_peak | wave | feasible |
|---|---|---|---|---|---|---|---|
| **1.4961** | 641.635 | 1415.5 | 1.8152 | 0.0452 | 1.2739 | 10 | ✅ |
| 1.5079 | 641.799 | 1424.1 | 1.8617 | 0.0429 | 1.3145 | 11 | ✅ |
| 1.5118 | 642.212 | 1423.1 | 1.8441 | 0.0430 | 1.2926 | 8 | ✅ |
| 1.5273 | **642.316** | 1425.5 | 1.8590 | 0.0430 | 1.2992 | 9 | ✅ |
| 1.5836 | 649.330 | 1367.4 | 1.9462 | 0.0497 | 1.3766 | 0 | ✗ |
| 1.5855 | **651.103** | 1368.0 | 1.9581 | 0.0504 | 1.3894 | 0 | ✗ |

The front is **only 0.68 EFPD wide across its entire feasible span**
(641.635 → 642.316) while F_r moves 0.0312. Inside the feasible region there is
effectively **no cycle-length price for peaking** at this cell — the two axes
are not in tension, which is the same structure the E1_E2 analysis found. The
two long-cycle points (649–651 EFPD) are wave-0 boards and are infeasible on
F_r: measured against the cycle-max feasible point (1.5273 @ 642.316) they buy
**+8.79 EFPD for +0.0582 F_r**; against the winner, +9.47 EFPD for +0.0894.

**Feasible set (n=41):** F_r min 1.4961 / p25 1.5148 / p50 1.5273 / max 1.5500;
cyclen 632.8 – 642.3; CBC ≤ 1425.5; F_q ≤ 1.9370; |AO| ≤ 0.0452. Every gate
except F_r cleared with wide margin — **F_r was the only binding constraint**,
which is exactly what the objective assumed.

---

## 4. Wave-by-wave trajectory

| wave | n | F_r min | F_r med | cum F_r min | cyclen min–max | n feasible | τ |
|---|---|---|---|---|---|---|---|
| 0 | 8 | 1.5836 | 1.6094 | 1.5836 | 630.9 – 651.1 | 0 | 0.30 |
| 1 | 8 | 1.5529 | 1.6175 | 1.5529 | 633.2 – 648.2 | 0 | 0.30 |
| 2 | 8 | 1.5450 | 1.6052 | **1.5450** | 631.3 – 645.4 | **1** | 0.30 |
| 3 | 8 | 1.5148 | 1.5656 | **1.5148** | 634.0 – 645.0 | 2 | 0.30 |
| 4 | 7 | 1.5324 | 1.5798 | 1.5148 | 627.5 – 640.1 | 3 | 0.33 |
| 5 | 8 | 1.5061 | 1.5430 | **1.5061** | 635.3 – 645.9 | 5 | 0.31 |
| 6 | 8 | 1.5289 | 1.5465 | 1.5061 | 638.8 – 645.2 | 4 | 0.40 |
| 7 | 8 | 1.5194 | 1.5680 | 1.5061 | 636.4 – 641.7 | 2 | 0.40 |
| 8 | 8 | 1.5006 | 1.5390 | **1.5006** | 636.1 – 645.2 | 5 | 0.40 |
| 9 | 7 | 1.5186 | 1.5398 | 1.5006 | 639.6 – 647.5 | 5 | 0.50 |
| 10 | 8 | **1.4961** | 1.5508 | **1.4961** | 637.1 – 642.3 | 4 | 0.46 |
| 11 | 8 | 1.5013 | 1.5143 | 1.4961 | 638.6 – 643.5 | 6 | 0.48 |
| 12* | 4 | 1.5146 | 1.5208 | 1.4961 | 638.4 – 640.4 | 4 | 0.55 |

Reading it:

* **Monotone, never stalled.** Five improvements (waves 0→2→3→5→8→10), the last
  at wave 10 of 12 — the budget was **not** oversized; the frontier was still
  moving when it ran out.
* **First feasible core at call 23** (wave 2), from a cell with zero.
* **Median F_r fell from 1.6094 to 1.5208** — the whole distribution moved, not
  just the tail. That is the FRONTIER-label yield purpose (b) asked for.
* **The finetune gate accepted on objective skill in all 13 waves**
  (`gate=objective+`), so every wave's labels were absorbed.
* **92 of 98 converged rows** landed below the cell's previous produce floor of
  1.7243; **28 landed below the DB truth 1.5374.**
* Origin of the 41 feasible cores: **37 `local`, 2 `elite`, 2 `guided`.** The
  cross-feed elites seeded the basin; the local-search refinement arm did the
  work inside it. Neither alone would have been sufficient.

---

## 5. Corpus / store deltas

| | before | after |
|---|---|---|
| canonical store rows | 73,903 | **74,003** (+100) |
| maps.npz keys | — | **+294** |
| N1_N2/f113 rows | 187 | **287** |
| N1_N2/f113 converged | 128 | **226** |
| **N1_N2/f113 feasible** | **0** | **41** |
| N1_N2/f113 F_r floor | 1.7243 | **1.4961** |

Merge: `lpopt merge-store` — **100 new / 0 upgraded / 0 conflicts**, ledger +0
lines (expected: `lpopt optimize` does not write the produce ledger). Dry-run
was run and inspected first. Canonical store was read-only until this merge;
backups `records.parquet.bak_pre_N1N2f113_20260816` and
`maps.npz.bak_pre_N1N2f113_20260816` were taken beforehand. Kit-side backup:
`records.parquet.bak_pre_N1N2f113_20260816` on 199. Local store was verified a
strict superset of the kit's (0 kit-only record_ids) before the kit was
refreshed pre-launch. **198 / 181 / 238 untouched.**

**For the next champion cycle:** s1f's registered frontier bias at this cell was
**+0.0266** (pessimistic — it over-predicts F_r on the frontier), measured on
produce rows with an F_r floor of 1.72. The 41 feasible cores and 28 sub-1.5374
rows are the first optimised labels this cell has ever had, and are the direct
correction material for that bias. Re-fitting has not been done here.

---

## 6. Deck decisions, audited against the outcome

### λ_Fr = 1000, not 0.0 — the brief's instruction had to be inverted

The brief specified `minfr_lambda = 0.0` for "pure F_r minimisation", with a
check for λ=0 degeneracy. **It is degenerate, in the opposite direction.**
`acquisition.py:474` is `scalar = cyclen_lcb − lam_fr · fr_ucb`; at λ=0 the F_r
term vanishes and the acquisition becomes **pure cycle-length maximisation**,
with F_r surviving only as a gate. No division, no NaN — a silent inversion.
It would also have put the two halves of the campaign in contradiction, since
best-tracking is hard-coded lexicographic (`−F_r·1e6 + cyclen`, campaign.py:1457,
and 1e6 is not a knob). λ=1000 is the code's own documented "F_r strictly
dominates" default and makes acquisition agree with best-tracking.

**Did it matter? Yes — early, and only early.** Scalars on the real verified rows:

| board | F_r | cyclen | λ=200 | λ=400 | λ=1000 |
|---|---|---|---|---|---|
| wave-0 longest (infeasible) | 1.5855 | 651.10 | **334.00** | **16.90** | −934.40 |
| wave-0 long-cycle (infeasible) | 1.5836 | 649.33 | 332.61 | 15.89 | −934.27 |
| wave-2 first feasible | 1.5450 | 632.79 | 323.79 | 14.79 | **−912.21** |

At λ=200/400 the **infeasible** wave-0 long-cycle boards outrank the first
feasible core; at λ=1000 they rank last. That is the steering that mattered
while the search was descending from 1.58 through 1.55.

**Correction to an interim claim made during the run:** I earlier suggested
λ=400 would have rejected the winning board. That is wrong. Once the winner
exists it strictly dominates most of the field (low F_r *and* high cyclen), and
**all three λ values rank it first**. λ shaped the early descent, not the final
ranking. The precise claim is the table above, and it is a demonstration of the
objective's preference ordering over verified rows — not a counterfactual
simulation of what the search would have sampled.

### No cycle band — as directed, and it cost nothing

`cycle_target_efpd` / `cycle_tolerance_efpd` gated nothing, as pre-registered
and re-verified on the kit (`feasibility_limits_for` returned
`cyclen_lo=None, cyclen_hi=None`). Their only effect was the report readout:
`best.distance = 18.065 = |641.635 − 659.7|`. The user directive cost the
campaign nothing — the winner runs at 641.6 EFPD, comfortably above the 625
programme reference, and the feasible front spans just 0.68 EFPD.

### Asset resolution — level 2, exactly as predicted

**All 98 converged rows** carry `pair_feed:MAS_RST.APRQ_11_0677.23`, the f121
base restart, resolved through the level-2 pair_feed ladder because
`CaseKey.folder` for feed ≠ 121 is `N1_N2_f113` and no such base exists.
Predicted in the deck header before launch; 100% match. Convergence was **98%**
versus this cell's 68.4% history, so the level-2 restart is not merely adequate
here — the morphed-elite region is markedly better behaved than the produce
sampling that established the old floor.

---

## 7. Honest notes and defects found

1. **The auto-generated `report.md` does not show the winner.** `lpopt report`
   ranks "Best verified loading patterns" by **cycle-distance to
   `cycle_target_efpd`**, regardless of objective, so its rank-1 row is
   F_r 1.527 (cyclen 642.3) and the actual F_r winner (1.4961) is absent from
   the table. The authoritative best is `state.json → best`, which agrees with
   the CLI RESULT line. This is a report-layer defect for every F_r-objective
   run, not a problem with this campaign. Do not read that table as the result.

2. **The `[constraints]` MTC block was inert.** The log states: *"SDM/MTC gate:
   no delivery ranking to verify (objective='min_fr_max_cycle'; the gate targets
   the flat_power delivery candidates, decision D9) — gate not run"*.
   `post_verify_calls = 0`. I carried the block from both parent decks, which
   also carry it on this objective; it does nothing here. It should be dropped
   from the next min_fr deck, and the parent decks' own copies are equally
   inert. No budget was spent on it.

3. **`promote()` never ran**, as pre-registered — it is called only from
   `ProduceDriver`, never from `lpopt optimize`. Every call resolved at level 2.
   A promoted f113 restart exists in the *local* store's history
   (`promoted:MAS_RST.APRQ_20_0633.23`, from the feedgrid campaign) but is not on
   the 199 kit; shipping it would be the cheap way to start a follow-up at level 1.

4. **The `_done()` fix the brief asked me to verify is not in lpopt at all** —
   `def _done` exists only in `ablation_wave.py`. `lpopt optimize` resumes from
   its own `state.json`/`labels.jsonl`. Nothing to verify; nothing was at risk.

5. **The result is one campaign at one cell with one seed.** The winner is a
   single MASTER-verified core; `n_cycles = 11` with no cap hit and the
   convergence tolerance discipline unchanged (`max_cycles 16, consecutive 2`).
   It has **not** been re-verified by an independent fixed-pattern control run,
   and the pre-delivery MTC/SDM gate has **not** been run on it. Both are
   prerequisites before this core goes anywhere near a delivery claim.

6. **Not beaten:** the N1_N2 store record at any feed (1.4932, f121) and the
   ga80 programme incumbent (1.4636, E1_E2/f121). This campaign opened a cell;
   it did not set a programme record.

---

## 7a. ADDENDUM 2026-08-17 — pin burnup was not treated, and should have been

Added after the fact, in response to the autoeng builder's precheck reporting a
**node pin-burnup cliff in parts of the f109 column**. The same question applies
here and this report did not ask it. Recording the correction rather than
quietly editing the headline.

**What was missing.** Sections 1-6 rank this campaign's cores on F_r, cyclen,
CBC, F_q and |AO| — the five axes `min_fr_max_cycle` gates — and say nothing
about pin burnup. `min_fr_max_cycle` does **not** gate pin burnup
(`feasibility_limits_for` returns `max_pin_burnup: None` for this objective), so
nothing in the campaign screened it.

**What is now known about this cell** (`data/reports/dbx_frontier_table.csv`,
N1_N2/f113, 45 DB cores, `n_pinmax_known = 45`):

| quantity | value |
|---|---|
| best DB core `pinmax_node_GWd` | **73.30** — **passes** the 75 limit |
| best DB core `pinmax_rodavg_GWd` | 66.91 — passes |
| all 45: `pinmax_node` min / mean | 73.20 / 75.28 |
| `frac_node_ge75` | **0.378** (17 of 45) |
| `frac_rodavg_ge75` | 0.089 (4 of 45) |

This is **the mildest of the three cells examined** — N1_N2/f109 is
`frac_node_ge75 = 1.000` (a dead end) and E1_E2/f109 is 0.886. Here the DB's own
best-F_r core clears the node limit.

**What cannot be checked, and why.** All **98 / 98** converged campaign rows
carry `max_pin_burnup = null`. Pin burnup is harvested only when the verifier
sets `enable_pin_burnup=True`, which `curriculum.py:29` states the default
verifier does not; the flag is reachable only as `[design].enable_pin_burnup` and
there is **no deck knob for `lpopt optimize`**. The store's `max_pin_burnup` for
this pair is not a usable substitute (different scale, and measured on
high-F_r produce cores, not on the campaign's frontier).

**RESOLVED 2026-08-17 — and the answer is bad.** The f109 campaign supplied a
validated method, and it was applied to this campaign's 41 feasible cores.
Predicted `max_pin_burnup` (s1g, `target_names` index 6):

| | value |
|---|---|
| 41 feasible cores: min / p50 / max | **86.14 / 86.71 / 88.26** |
| ≥ 75 GWd/tU | **41 / 41 (100%)** |
| ≥ 80 GWd/tU (lpopt `*_pin_bu_limit`) | **41 / 41 (100%)** |
| **the winner at F_r 1.4961** | **86.75** |
| validation (N1_N2 pair-wide, n=171 measured rows) | bias **+0.17**, MAE **2.16** |

> **정의 각주 (2026-08-20 사용자 확정)**: 핀연소도 한계치 **80 GWd/tU** 는
> **핀 axial peak** — 즉 우리가 측정/예측해 온 `max_pin_burnup`(3-D 핀 노드
> 첨두) — 에 건다. 봉평균은 보조 관측량이며 판정축이 아니다. 아래 판정은
> 공식 관측량 그대로이므로 **유효하다**. `data/reports/pinbu_definition_20260820.md` · `pinbu_audit_20260820.md` §8.
>
> **실측 확증 (2026-08-20)**: 이 41기 중 F_r 상위 5기를 MASTER 로 재측정한 결과
> **85.26–86.78**(승자 **86.19**) — **5/5 FAIL**. 예측(86.14–88.26)이 실측과
> 0.4 GWd/tU 안에서 맞았다. `pinbu_wave_results_20260820.md` §2 (`N1N2_f113`).

**Every core this campaign found breaches the 80 GWd/tU limit by ~6-8 GWd/tU,
about 3 MAE clear of it.** The headline verdict for this campaign is therefore
the same as f109's: **opened for F_r, but pin-limited for delivery.** The F_r
result in §1 stands as an F_r result and nothing more.

**The DB cell character was misleading, and that is the transferable lesson.**
This cell looked *benign* on the DB node metric (37.8% exceedance, DB best
passing at 73.30) — milder than E1_E2/f109's 88.6% — yet its campaign cores are
the **more** pin-loaded of the two (86.71 vs 82.57 p50). The DB's cores and ours
are different populations: the search drives toward low F_r, which concentrates
burnup, and nothing in `min_fr_max_cycle` pushes back. **DB pin statistics are
context, not a proxy for our own cores.**

Prerequisites before any delivery claim, now two: the un-run MTC/SDM gate (§7.5)
and a **pin-BU-enabled MASTER re-verification** of the top cores (curriculum
WaveVerifier or design/bootstrap path — both set `enable_pin_burnup`; the
campaign path cannot). Full analysis and the program-level remediation list:
`data/reports/fpcamp_E1E2_f109_results_20260817.md` §7.

Full treatment and the pre-registered verdict rule for the live f109 campaign:
`data/reports/fpcamp_E1E2_f109_pinburnup_prereg_20260817.md`.

---

## 8. Paths

| artefact | path |
|---|---|
| deck (pre-registration header) | `fpcamp_minfr_N1N2_f113_199.inp` |
| launcher / arm script / status probe | `run_fpcamp_minfr_N1N2_f113_199.bat`, `launch_fpcamp_N1N2_f113_199.ps1`, `status_fpcamp_N1N2_f113_199.ps1` |
| run dir (on 199) | `C:\Users\USER\lpopt_work\kit_frontier\runs\fpcamp_minfr_N1N2_f113` (10.3 GB, 1340 files) |
| campaign log (on 199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_N1N2_f113_out.log` |
| harvested labels / state / report | scratchpad `harvest/run/{labels.jsonl,state.json,report.md,status.json}` |
| canonical store (merged) | `data/store/records.parquet` (74,003 rows) |
| store backups | `data/store/records.parquet.bak_pre_N1N2f113_20260816`, `data/store/maps.npz.bak_pre_N1N2f113_20260816` |
| DB truth source | `data/reports/scoping_mesh_20260815/feasible_database.xlsx` (sheet `cores`) |
| mesh prediction source | `data/reports/scoping_mesh_20260815/cell_verdicts.csv` row 29 |
