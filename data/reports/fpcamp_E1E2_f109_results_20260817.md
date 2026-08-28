# Low-feed opening campaign r2 — E1_E2 / feed 109 / ga80 — RESULTS

**Run 2026-08-17 on box 199.** Deck `fpcamp_minfr_E1E2_f109_199.inp`
(sha256 `1a967b37a93b57f9dc93e7b2a8f2f23422ed740c9299e16d0196d026392ad247`),
run dir `runs/fpcamp_minfr_E1E2_f109`, champion `data/models/s1g`, seed 1109,
`rc=0`. Pre-registration is the deck header, hashed before launch; pin-burnup
marks and the verdict rule were pre-registered mid-run at 8/100 calls with zero
feasible cores, in `fpcamp_E1E2_f109_pinburnup_prereg_20260817.md`.

---

## 1. HEADLINE VERDICT: opened for F_r, but PIN-LIMITED FOR DELIVERY

The pre-registered verdict rule fires. **All 52 feasible cores this campaign
found are predicted to exceed the LEU+ 80 GWd/tU pin-burnup limit**, by a margin
far larger than the model's error at this cell. This campaign did **not** produce
a delivery-ready core, and the F_r result below must not be read as if it had.

| | predicted `max_pin_burnup` |
|---|---|
| 52 feasible cores: min / p50 / max | **81.31 / 82.57 / 83.77** |
| ≥ 75 GWd/tU | **52 / 52 (100%)** |
| ≥ 80 GWd/tU (lpopt's own `*_pin_bu_limit`) | **52 / 52 (100%)** |
| the F_r winner | **83.16** |

> **정의 각주 (2026-08-20 사용자 확정)**: 핀연소도 한계치 **80 GWd/tU** 는
> **핀 axial peak** — 즉 우리가 측정/예측해 온 `max_pin_burnup`(3-D 핀 노드
> 첨두) — 에 건다. 봉평균은 보조 관측량이며 판정축이 아니다. 아래 판정은
> 공식 관측량 그대로이므로 **유효하다**. `data/reports/pinbu_definition_20260820.md` · `pinbu_audit_20260820.md` §8.
>
> **그리고 이 예측은 실측으로 확증되었다** (2026-08-20, 44-체인 웨이브):
> 이 캠페인의 F_r 승자 5기를 MASTER 로 재측정한 결과 **81.76–82.38**
> (승자 **82.11**) — **5/5 FAIL**. 예측 오차 bias +0.59 / MAE 0.59.
> `pinbu_wave_results_20260820.md` §2 (`E1E2_f109`).

**The prediction is trustworthy here, and that was checked before it was used.**
s1g's pin-BU head was validated against the **33 rows at this exact cell that
carry measured `max_pin_burnup`**: bias **−1.39** (it *under*-predicts), **MAE
1.84**, sd 1.88. Correcting for that bias moves the set *up*, to ~82.7 p50. The
lowest single predicted value in the whole feasible set (81.31) clears the 80
limit by more than 0.7 MAE even before the bias correction. There is no reading
of this that puts these cores under the limit.

**A second, independent line of evidence agrees.** The DB's own cores in this
cell are node-pin-pressured: `frac_node_ge75 = 0.886` (39 of 44), and the DB's
best-F_r core sits at 78.43 GWd/tU node. Two different metrics, two different
core populations, same conclusion.

**What this is not.** It is not evidence the campaign made burnup worse. The
store's measured `max_pin_burnup` for this cell has p50 **83.33** across its 33
labelled rows — our cores at 82.57 sit *slightly below* the cell's normal
operating level. The pin-burnup problem is a property of the **operating point**,
not of this search.

**The program-level gap this exposes.** `min_fr_max_cycle` does not gate pin
burnup (`feasibility_limits_for` returns `max_pin_burnup: None`), and the
`optimize` verify path does not even *harvest* it (0 of 100 rows carry it; the
flag is `[design].enable_pin_burnup` and is unreachable from an optimize deck).
So a campaign on this objective can run to completion, report 52 "feasible"
cores, and never learn that none of them is deliverable. **Both opening campaigns
have now done exactly that.** See §7.

---

## 2. RESULT (F_r axis)

Reported second, per the verdict rule.

| | |
|---|---|
| **Best feasible F_r** | **1.4787** @ cyclen **593.148 EFPD** |
| record_id | `3153f3b39effbf6660082ec4bf8abe0e393bcd2e361743b6809d27839a21ab61` |
| CBC_max / F_q / \|AO\| | 1201.56 ppm / 1.8313 / 0.0451 |
| node_peak / map_cov | 1.2675 / 0.2794 |
| n_cycles / restart | 10 / `pair_feed:MAS_RST.APRQ_11_0615.88` |
| found at | wave 3 |
| **Feasible cores** | **52 / 100** |
| Converged | **100 / 100 (100.0%)** |
| Pareto front size | **6** (4 feasible, 2 infeasible) |

### Against the registered marks

| mark | value | delta | |
|---|---|---|---|
| **PRIMARY** — any feasible core at this cell | none existed | **52 found** | ✅ |
| 1.55 licensing gate | 1.55 | **−0.0713** | ✅ |
| **SECONDARY** — DB truth, this cell | 1.5366 | **−0.0579** | ✅ |
| **STRETCH** — ga80 programme incumbent | 1.4636 | **+0.0151** | ✗ not met |
| cell produce floor | 1.7692 | **−0.2905** | |
| s1g predicted floor | 1.5654 | −0.0867 | |
| s1g bias-corrected floor | 1.5458 | −0.0671 | |
| f113 campaign's final best | 1.4961 | **−0.0174** | |
| global DB min, all 6113 cores | 1.4958 | **−0.0171** | |

The stretch criterion was registered as "not expected" and was not met; it came
within 0.0151. Note the campaign passed f113's *final* result (1.4961, reached at
call 88) by call **32**.

---

## 3. R2 — CALLS-TO-FIRST-FEASIBLE: **17 vs 23**. Transfer accelerated the opening.

The headline registered question. Measured from `labels.jsonl` ordering, both
campaigns scored identically:

| | f113 (s1f) | **f109 (s1g)** |
|---|---|---|
| **calls to first feasible** | **23** | **17** (−26%) |
| wave of first feasible | 2 | 2 |
| feasible by end of wave 2 | 1 | **3** |
| DB truth beaten at call | ~32 | **24** |
| total feasible / 100 | 41 | **52** |
| converged | 98 | **100** |

**Registered direction: `< 23` → transfer accelerated the opening. It did.**

The result is *stronger* than the raw comparison, because both pre-registered
confounds ran **against** this cell: it had the larger data gap (+0.2326 vs
+0.1869) and the worse historical convergence (60.6% vs 68.4%). The harder cell
opened faster anyway.

**And the acceleration is s1g's prior, not in-campaign learning.** The finetune
gate **rejected every one of the 13 waves** (`gate=objective-`), against f113
where it accepted all 13. The model absorbed the f113 labels during *training*;
the per-wave updates added nothing measurable at this cell. That is the cleanest
available evidence that the transfer is real and that it lives in the champion,
consistent with §10.2 of the mesh readout attributing 86% of the f109 column
gain to the model rather than the pool.

---

## 4. R3 — (cyclen, F_r) Pareto front

| F_r | cyclen | CBC_max | F_q | \|AO\| | node_peak | wave | feasible |
|---|---|---|---|---|---|---|---|
| **1.4787** | 593.148 | 1201.6 | 1.8313 | 0.0451 | 1.2675 | 3 | ✅ |
| 1.4962 | 594.068 | 1208.4 | 1.8407 | 0.0451 | 1.2916 | 11 | ✅ |
| 1.5042 | 594.091 | 1209.8 | 1.8534 | 0.0447 | 1.2987 | 10 | ✅ |
| 1.5158 | **594.672** | 1212.8 | 1.8832 | 0.0449 | 1.2949 | 12 | ✅ |
| 1.5782 | 597.141 | 1252.3 | 1.9657 | 0.0400 | 1.3998 | 1 | ✗ |
| 1.5790 | **598.373** | 1274.8 | 1.9762 | 0.0385 | 1.3904 | 0 | ✗ |

The feasible span is **1.52 EFPD wide** (593.148 → 594.672) while F_r moves
0.0371 — the same structure f113 showed (0.68 EFPD / 0.0312). Inside the feasible
region there is effectively no cycle-length price for peaking. The two long-cycle
points are wave-0/1 boards, infeasible on F_r.

**Feasible set (n=52):** F_r 1.4787 – 1.5500 (p50 ~1.5158); cyclen 592.8 – 594.7;
CBC ≤ 1306.9; F_q ≤ 1.9370; |AO| ≤ 0.0459. Every gate except F_r cleared with
very wide margin — **CBC in particular is ~300 ppm under its limit**, so F_r was
again the only binding constraint.

---

## 5. R4 — feasible yield curve and wave trajectory

| wave | n | F_r min | F_r med | cum F_r min | cyclen min–max | **n feasible** | gate | τ |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 1.5790 | 1.5887 | 1.5790 | 568.9 – 598.4 | 0 | − | 0.30 |
| 1 | 8 | 1.5547 | 1.5790 | 1.5547 | 581.0 – 597.1 | 0 | − | 0.30 |
| 2 | 8 | 1.5176 | 1.5764 | **1.5176** | 581.9 – 594.1 | **3** | − | 0.30 |
| 3 | 8 | **1.4787** | 1.5206 | **1.4787** | 584.2 – 593.8 | 5 | − | 0.30 |
| 4 | 8 | 1.5136 | 1.5330 | 1.4787 | 584.6 – 594.1 | 5 | − | 0.30 |
| 5 | 8 | 1.5149 | 1.5309 | 1.4787 | 573.9 – 593.7 | 5 | − | 0.30 |
| 6 | 8 | 1.5151 | 1.5316 | 1.4787 | 572.9 – 593.6 | 5 | − | 0.30 |
| 7 | 8 | 1.5140 | 1.5356 | 1.4787 | 585.6 – 594.1 | 5 | − | 0.30 |
| 8 | 8 | 1.5158 | 1.5232 | 1.4787 | 582.3 – 595.3 | 5 | − | 0.31 |
| 9 | 8 | 1.5113 | 1.5230 | 1.4787 | 574.7 – 593.5 | 5 | − | 0.32 |
| 10 | 8 | 1.5042 | 1.5215 | 1.4787 | 574.6 – 594.1 | 5 | − | 0.36 |
| 11 | 8 | 1.4962 | 1.5164 | 1.4787 | 582.0 – 594.1 | 5 | − | 0.35 |
| 12* | 4 | 1.5137 | 1.5160 | 1.4787 | 592.8 – 594.7 | 4 | − | 0.41 |

* **Yield is flat and high at 5/wave** from wave 3 onward — 52 total against
  f113's 41. The cell is dense with feasible cores once found.
* **The frontier converged early.** All descent happened in waves 0-3; nine
  subsequent waves produced 44 more feasible cores but no new best. That is the
  opposite of f113, whose last improvement came at wave 10 of 12. **Here the
  budget WAS oversized for the F_r axis** — roughly 60 calls would have found the
  same winner. Worth knowing for the next cell.
* **100/100 converged**, against a 60.6% historical rate for this cell. As at
  f113, the morphed-elite region is far better behaved than produce sampling.
* **All 100 rows** carry `pair_feed:MAS_RST.APRQ_11_0615.88` — the level-2
  resolution predicted in the deck header, matched 100/100 (as at f113, 98/98).

---

## 6. Store deltas

| | before | after |
|---|---|---|
| canonical store rows | 74,003 | **74,103** (+100) |
| maps.npz keys | — | **+300** |
| E1_E2/f109 rows | 292 | **392** |
| E1_E2/f109 converged | 177 | **277** |
| **E1_E2/f109 feasible** | **0** | **52** |
| E1_E2/f109 F_r floor | 1.7692 | **1.4787** |

Merge: **100 new / 0 upgraded / 0 conflicts**, ledger +0 (expected). Dry-run
inspected first. Backups `records.parquet.bak_pre_E1E2f109_20260817` and
`maps.npz.bak_pre_E1E2f109_20260817`. Canonical store read-only until the merge.
**198 / 181 / 238 untouched.**

---

## 7. The pin-burnup gap is now a two-campaign, program-level finding

Applying the same validated method to the **closed f113 campaign**:

| | f113 (N1_N2) | f109 (E1_E2) |
|---|---|---|
| feasible cores | 41 | 52 |
| predicted pin BU min / p50 / max | **86.14 / 86.71 / 88.26** | 81.31 / 82.57 / 83.77 |
| ≥ 80 GWd/tU | **41 / 41 (100%)** | **52 / 52 (100%)** |
| winner's predicted pin BU | **86.75** | 83.16 |
| validation bias / MAE | +0.17 / 2.16 (pair-wide, n=171) | −1.39 / 1.84 (in-cell, n=33) |
| DB `frac_node_ge75` for the cell | 0.378 | 0.886 |

**Both campaigns' cores breach the limit, and f113's are worse.** Its winner at
F_r 1.4961 carries a predicted pin BU of **86.75** — 6.75 over the 80 limit and
roughly 3 MAE clear of it.

> **실측 확증 (2026-08-20)**: 두 캠페인의 승자를 모두 MASTER 로 재측정했다 —
> f113(N1_N2) 승자 **86.19**, f109(E1_E2) 승자 **82.11**. **둘 다 80 초과, 10/10
> FAIL**. 이 절의 프로그램 수준 판정은 예측이 아니라 **실측으로 확정**되었다.
> 관측량·한도 정의는 `data/reports/pinbu_definition_20260820.md`.

**The DB cell character did NOT predict this.** f113's cell looked benign on the
DB node metric (37.8% exceedance, DB best passing at 73.30) and its campaign
cores are nonetheless the *more* pin-loaded of the two. The reason is that the
DB's cores and our cores are different populations: our search drives toward low
F_r, which concentrates power and burnup, and nothing in the objective pushes
back. **DB pin statistics are context, not a proxy for our own cores.** That is a
lesson for future cell selection, and it is why the prereg's instinct to demand a
direct measurement was right.

### What must change

1. **Gate pin burnup on this objective.** `min_fr_max_cycle` has a
   `max_pin_burnup: None` entry in `feasibility_limits_for` while
   `min_fuel_cost` and `fr_boundary` both gate it at 80. Adding the same gate is
   a small, well-precedented change and would have prevented both campaigns from
   reporting undeliverable cores as "feasible".
2. **Harvest pin burnup in the optimize path.** `enable_pin_burnup` is
   `[design]`-scoped only; the campaign verifier never sets it, so 0 of 200 rows
   across both campaigns carry a measured value. Everything above rests on a
   validated *prediction* because a measurement was not available.
3. **Re-verify with MASTER before any delivery claim.** A pin-BU-enabled
   re-verification of the top ~5 cores from each campaign (curriculum
   WaveVerifier or design/bootstrap path, both of which set the flag) is a small
   bounded job and is the only thing that settles this definitively.

---

## 8. Honest notes

1. **This campaign's F_r result is real and its delivery status is not.** 52
   MASTER-verified constraint-feasible cores exist at a cell that had none; the
   best is 1.4787. None is deliverable on present evidence.
2. **Pin burnup is predicted, not measured** — stated wherever it is used. The
   in-cell validation (MAE 1.84, n=33) is strong but it is still a model.
3. **No independent re-verification.** The winner has not been re-run as a
   fixed-pattern control; the MTC/SDM gate did not run (correctly — it is inert
   on this objective, which is why the `[constraints]` block was dropped from
   this deck after f113 measured it doing nothing).
4. **Budget was oversized on the F_r axis** (§5) — the frontier stopped moving at
   wave 3. The extra calls bought 44 more feasible cores, which is real label
   value for the model, but no better core.
5. **`lpopt report`'s best-patterns table again does not show the winner** (it
   ranks by cycle distance). Read `state.json → best`. Known defect, unchanged.
6. **Not beaten:** the ga80 programme incumbent (1.4636). Reached +0.0151.
7. **The cell choice was mine and it was contested.** N1_N2/f109 was the intuitive
   pick (f113's cores as 1-unit-morph donors); E1_E2 was chosen because it was the
   only `model-biased` cell in the column, with a corrected floor already inside
   the gate and a 6.5× deeper elite pool. The outcome is consistent with that
   reasoning but does not prove N1_N2 would have failed — that comparison was not
   run.

---

## 9. Paths

| artefact | path |
|---|---|
| deck (pre-registration) | `fpcamp_minfr_E1E2_f109_199.inp` |
| pin-burnup pre-registration | `data/reports/fpcamp_E1E2_f109_pinburnup_prereg_20260817.md` |
| launcher / arm / status | `run_fpcamp_minfr_E1E2_f109_199.bat`, `launch_fpcamp_E1E2_f109_199.ps1`, `status_fpcamp_E1E2_f109_199.ps1` |
| run dir (199) | `C:\Users\USER\lpopt_work\kit_frontier\runs\fpcamp_minfr_E1E2_f109` |
| campaign log (199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_E1E2_f109_out.log` |
| canonical store | `data/store/records.parquet` (74,103 rows) |
| store backups | `data/store/records.parquet.bak_pre_E1E2f109_20260817`, `maps.npz.bak_pre_E1E2f109_20260817` |
| predecessor campaign | `data/reports/fpcamp_N1N2_f113_results_20260816.md` |
| DB / mesh sources | `data/reports/dbx_frontier_table.csv`, `data/reports/scoping_mesh_20260815/{mesh_nodes,cell_verdicts}.csv` |
