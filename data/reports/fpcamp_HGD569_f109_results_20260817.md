# High-Gd boron-opened cell, F_r assault — `P6253Z1G06N24_P6253Z2G10N24` / feed 109 — RESULTS

**Run 2026-08-17 on box 199.** Deck `fpcamp_minfr_HGD569_f109_199.inp`
(sha256 `244384F8DF986BA048FACB5A15911E88B28F1BF89FA8250D36B081A334564509`),
run dir `runs/fpcamp_minfr_hgd569_f109`, champion `data/models/s1g`, seed 5694,
`rc=0`. Pre-registration: `data/reports/fpcamp_HGD569_f109_prereg_20260817.md`
(written before launch, deck-hash gated). Scored against the readouts
registered there (§3): R1, R2 (PRIMARY), R3, R4, R5.

---

## 1. HEADLINE VERDICT: STRETCH met — the largest F_r move measured at any cell, gate not reached

The pre-registered SECONDARY and STRETCH criteria both fire; PRIMARY does not.

| mark (prereg §4) | requirement | result | |
|---|---|---|---|
| PRIMARY | any MASTER-verified core passing all 4 gates | 0 / 40 converged | not met |
| **SECONDARY** | F_r < 2.0481 with CBC ≤ 1600 | **1.6743**, CBC 1540.99 | ✅ met |
| **STRETCH** | F_r ≤ 1.8375 (pair's f125 best, harder feed) | **1.6743** (−0.1632) | ✅ met |
| PARTIAL | F_r < 2.0481 but not < ~1.84, or CBC crosses 1600 | superseded by STRETCH | n/a |
| NULL | floor doesn't move meaningfully below 2.0481 | floor moved **−0.3738** | **does not apply** |

The pre-registration (§2b-ii) explicitly did *not* expect the gate to be reached
and pre-computed the scale: the precedent (T6_T4/f121) moved its own floor by
−0.32 and that size of move, applied here, "lands at ~1.73, not inside 1.55."
**This campaign moved −0.3738 — larger than the precedent's own move — and
landed at 1.6743, better than the pre-registered no-surprise expectation.**
That closes **75.1%** of the 0.4981 in-cell distance to the 1.55 licensing gate.
The gate itself was not reached (margin +0.1243), exactly as pre-registered as
the likely outcome.

---

## 2. R1 — best F_r vs both pinned marks

| | |
|---|---|
| **Best feasible-search F_r** | **1.6743** @ cyclen **675.392 EFPD** |
| record_id | `614c83b9a31ec9b7e273c3ddd586f1815bc6c653157dc83e90721fb85183f255` |
| CBC_max / F_q / \|AO\| | 1540.99 ppm / 2.0982 / 0.0268 |
| n_cycles / restart | 10 / `pair_ecore:MAS_RST.APRQ_11_0705.02` (level 3, as predicted) |
| found at | call 57 / 60, wave 7 (the reserve wave) |
| Converged | **40 / 60 (66.7%)** — errors 20/60, all `non_finite_flux` |
| Calls per usable label | **1.500** — inside the pre-registered 1.4–1.7 band (§7) |

### Against the registered marks

| mark | value | delta | |
|---|---|---|---|
| **in-cell f109 floor (like-for-like)** | 2.0481 | **−0.3738** | ✅ new floor |
| 1.55 licensing gate | 1.55 | +0.1243 | not reached |
| **f125 STRETCH mark (cross-feed, labelled separately)** | 1.8375 | **−0.1632** | ✅ beaten |
| precedent's own move, applied at this cell's scale | ~1.73 | −0.0557 | beaten |

The 1.8375 mark is the pair's F_r at **feed 125**, a different (harder) feed —
reported here only as the pre-registered cross-feed comparison, not as
like-for-like. The like-for-like mark is 2.0481.

---

## 3. R2 — PRIMARY: 4-constraint joint pass count

**0 / 40 converged (0 / 60 total).** No core passed
F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30 simultaneously — F_r is the
sole blocker:

| gate | pass rate (n=40 converged) | binding? |
|---|---|---|
| F_r ≤ 1.55 | **0 / 40 (0%)** | **yes — the only one** |
| CBC ≤ 1600 | 40 / 40 (100%) | no |
| F_q ≤ 2.41 | 38 / 40 (95%) | no |
| \|AO\| ≤ 0.30 | 40 / 40 (100%) | no |

Adding the new predicted-pin-BU gate (≤78, §5) changes nothing here — it
cannot subtract from an already-empty feasible set. **5-constraint joint pass
count: 0 / 40**, same as the 4-constraint count.

---

## 4. R3 — the F_q axis: F_r-closed or F_q-closed?

**F_r-closed, not F_q-closed.** F_q moved further and faster than F_r did:

| quantity | in-cell anchor floor | campaign (n=40 converged) | |
|---|---|---|---|
| **F_q** min / p50 / max | 2.6120 / 2.9465 / 4.3621 | **2.082** / 2.1968 / 3.5481 | −0.53 at the floor |
| F_q ≤ 2.41 pass rate | **0 / 16 (0%)** | **38 / 40 (95%)** | |
| F_r min | 2.0481 | 1.6743 | −0.3738 |
| F_r ≤ 1.55 pass rate | 0 / 16 (0%) | 0 / 40 (0%) | unchanged |

F_q dropped out as a binding constraint the moment the search began minimizing
F_r — the two move together at this cell (driving down the radial peak also
flattens the assembly-level peak), so the campaign never had to trade F_q for
F_r. **F_r remains the sole open axis at f109**; F_q is functionally resolved.

---

## 5. R4 — (F_r, CBC) Pareto front

CBC has enormous slack throughout — never within 50 ppm of binding:

| F_r | CBC_max | F_q | \|AO\| | cyclen | wave | call | pred. pin BU |
|---|---|---|---|---|---|---|---|
| **1.6743** | 1540.99 | 2.0982 | 0.0268 | 675.392 | 7 | 57 | 81.13 |
| 1.6867 | 1508.78 | 2.0820 | 0.0326 | 672.215 | 5 | 45 | 83.69 |
| 1.6902 | 1505.22 | 2.1029 | 0.0309 | 673.295 | 6 | 53 | 83.22 |
| 1.7198 | 1482.35 | 2.1468 | 0.0299 | 666.203 | 1 | 13 | 87.27 |
| 1.7257 | 1469.85 | 2.1556 | 0.0337 | 662.839 | 0 | 1 | 87.55 |
| 1.7499 | 1439.64 | 2.1248 | 0.0356 | 661.676 | 1 | 10 | 86.63 |
| 1.7643 | 1431.00 | 2.1370 | 0.0374 | 659.234 | 0 | 2 | 86.69 |
| 1.8832 | 1420.37 | 2.3683 | 0.0370 | 642.965 | 1 | 16 | 87.16 |

**CBC axis**: campaign min/p50/max = **1420.37 / 1507.32 / 1550.09** ppm, 40/40
(100%) under the 1600 gate. The campaign's own floor sits **+15.21 ppm above**
the in-cell anchor floor (1405.16) — an F_r-driven search doesn't probe CBC
directly and never needed to; CBC is the least-constraining of the four gates
at this cell, consistent with why it was chosen (prereg §1: widest boron
opening in the high-enrichment set). The 195 ppm anchor margin is confirmed to
persist at the F_r optimum, not just at the sample (registered concern, §2d).

---

## 6. R5 — predicted pin-burnup gate: shaping, not steering

**Verdict: F_r floor, not "pin gate took over."** Both registered diagnostics
point the same way.

| | |
|---|---|
| predicted `max_pin_burnup`, 40 converged: min / p50 / max | **81.09 / 83.96 / 93.32** |
| ≤ 78 (this campaign's gate) | **0 / 40 (0%)** |
| ≤ 80 (lpopt's licensing-scale default) | **0 / 40 (0%)** |
| winner's predicted pin BU | **81.13** — closest any core came to the gate |
| Pearson r(F_r, predicted pin BU), 40 converged rows | **+0.637** |

> **정의 각주 (2026-08-20 사용자 확정)**: 핀연소도 한계치 **80 GWd/tU** 는
> **핀 axial peak** — 즉 우리가 측정/예측해 온 `max_pin_burnup`(3-D 핀 노드
> 첨두) — 에 건다. 봉평균은 보조 관측량이며 판정축이 아니다. 아래 판정은
> 공식 관측량 그대로이므로 **유효하다**. `data/reports/pinbu_definition_20260820.md` · `pinbu_audit_20260820.md` §8.

**Diagnostic 1 — does pin fall alongside F_r, or does F_r stall while pin sits
near 90?** It falls alongside: predicted-pin p50 fell wave-over-wave from
**88.15 (wave 0) → 81.54 (wave 7)** in step with F_r's cumulative minimum
falling **1.7257 → 1.6743**. Per the pre-registered rule this is the "gate is
shaping" signature, not "gate is steering."

**Diagnostic 2 — which term dominates the penalty, computed per-candidate
(not at the fixed dry-run point)?** Using the registered formula
`term_fr=((F_r−1.55)/0.01)²`, `term_pin=((pin−78)/1.0)²`: at the winner,
`term_fr=154.5` vs `term_pin=9.8` — **94.0% of the penalty is F_r.** Every
point on the Pareto front (§5) is 85–95% F_r-dominant. The pre-registered
dry-run crossover (F_r≈1.67, computed at a fixed pin=89.8) does not describe
this campaign's actual operating region: real candidates carried lower
predicted pin (81–93, not a flat 89.8), which pulls each candidate's own
crossover point down with it — the winner's own crossover (pin=81.13) is
F_r≈1.5813, well below its achieved 1.6743. The search stayed inside the
F_r-dominant regime throughout; it was never captured by the pin term.

**What this does not mean.** None of the 40 converged rows would pass a pin
gate either way (0/40 ≤ 78, 0/40 ≤ 80) — so even the F_r-optimal core here
remains undeliverable on the predicted pin-burnup metric, exactly as both
prior min_fr campaigns (E1_E2/f109, N1_N2/f113) found. Closing F_r further
would not, by itself, produce a deliverable core; pin burnup is still
unresolved program-wide (§10, this and the two E1E2/N1N2 reports).

**Wave trajectory** (frontier moved in waves 0, 1, 5, and 7 — the *last*
wave — and was flat in 2–4 and 6):

| wave | n conv | F_r min | F_r med | cum min | cyclen range | pred. pin p50 |
|---|---|---|---|---|---|---|
| 0 | 6 | 1.7257 | 1.8169 | 1.7257 | 639.3–665.2 | 88.15 |
| 1 | 6 | 1.7198 | 1.7462 | 1.7198 | 643.0–667.6 | 87.22 |
| 2 | 5 | 1.7693 | 1.7967 | 1.7198 | 663.9–670.9 | 85.89 |
| 3 | 4 | 1.7838 | 1.8228 | 1.7198 | 667.7–672.8 | 83.09 |
| 4 | 5 | 1.7567 | 1.7923 | 1.7198 | 665.2–672.5 | 83.50 |
| 5 | 6 | 1.6867 | 1.7187 | **1.6867** | 648.7–675.0 | 83.80 |
| 6 | 4 | 1.6902 | 1.7102 | 1.6867 | 670.9–676.0 | 83.41 |
| 7* | 4 | **1.6743** | 1.7008 | **1.6743** | 673.9–676.5 | 81.54 |

**Registered finding (prereg §9): "if the frontier is still moving at wave 7
that is itself a registered finding — this cell behaves unlike both
precedents."** It did move at wave 7 — the campaign's best core was found on
the very last call of the budget (call 57/60). Unlike E1_E2/f109 (frontier
converged by wave 3, budget oversized) this cell's frontier was still
descending when the budget ran out. **60 calls was not oversized here; if
anything the campaign ended while still improving.** The finetune gate
rejected all 8 waves (`gate=objective-`, tau flat at 0.30 the entire run,
never adapted) — all improvement came from the base s1g champion plus normal
BO refinement, not from in-campaign model updates.

---

## 7. Store deltas

| | before | after |
|---|---|---|
| canonical store rows | 74,357 | **74,417** (+60) |
| maps.npz keys | — | **+120** |
| HGD569 pair @ f109 rows | 23 | **83** (+60) |
| HGD569 pair @ f109 converged | 16 | **56** (+40) |
| **HGD569 pair @ f109 F_r floor** | **2.0481** | **1.6743** |

Merge: `lpopt merge-store` — **60 new / 0 upgraded / 74,357 duplicate**, ledger
+0 (expected: `lpopt optimize` does not write the produce ledger). Dry-run was
run and inspected first; 0 flagged conflicts against the recognized curriculum
set. Backups taken before merge: `data/store/records.parquet.bak_pre_HGD569f109_20260817`,
`data/store/maps.npz.bak_pre_HGD569f109_20260817`. **198 / 181 / 238
untouched.**

---

## 8. Honest notes

1. **STRETCH met, PRIMARY not.** 1.6743 is a real, MASTER-verified F_r floor
   for this cell at f109 — the best F_r ever measured here — but it is not a
   4-constraint-feasible, deliverable core. It is 0.1243 over the 1.55 gate.
2. **Pin burnup is predicted, not measured**, exactly as §5 of the
   pre-registration flagged: `enable_pin_burnup` is unreachable from an
   `optimize` deck, so 0/60 rows here carry a measured value. The R5 analysis
   above rests on the acquisition function's own `pred_mean` column 6,
   recovered from `waves/wave_NN/selection.json` and joined to the verified
   labels by `record_id` — every one of the 40 converged rows matched (40/40).
3. **The NULL reading does not apply and is not being retrofitted as a
   partial success.** SECONDARY and STRETCH were both registered in advance
   and both fired on the actual data; this is not a borderline call.
4. **Convergence rate (66.7%) is lower than either closed min_fr precedent**
   (E1_E2 100%, N1_N2 not directly comparable) — consistent with the
   pre-registered expectation of a never-run pair reopening at level-3
   resolution with no cache (§7), and the observed 1.500 calls/label sits
   inside the pre-registered 1.4–1.7 band almost exactly.
5. **No independent re-verification.** The winner has not been re-run with
   `enable_pin_burnup` set, so its *measured* pin burnup is unknown. That
   remains the only thing that would settle deliverability, as in E1_E2/N1N2.
6. **`lpopt report`'s best-patterns table does not show this winner** (ranks
   by cycle distance, known defect, unchanged). Read `state.json → best_overall`.
7. **This report answers R1, R2, R3, R4, R5 as registered**; none required a
   mid-run amendment (contrast E1_E2, whose pin-burnup marks were
   pre-registered mid-run because the deck predated that gate — this deck
   shipped with the gate already in place).

---

## 9. Paths

| artefact | path |
|---|---|
| deck (pre-registration, hashed) | `fpcamp_minfr_HGD569_f109_199.inp` |
| pre-registration | `data/reports/fpcamp_HGD569_f109_prereg_20260817.md` |
| launcher / bat / status | `launch_fpcamp_HGD569_f109_199.ps1`, `run_fpcamp_minfr_HGD569_f109_199.bat`, `status_fpcamp_HGD569_f109_199.ps1` |
| run dir (199) | `C:\Users\USER\lpopt_work\kit_frontier\runs\fpcamp_minfr_hgd569_f109` |
| campaign log (199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_hgd569_f109_out.log` |
| harvested labels / state / wave selections | scratchpad `harvest_hgd569/{labels.jsonl,state.json,campaign_out.log,wave0N_selection.json}` |
| canonical store (merged) | `data/store/records.parquet` (74,417 rows) |
| store backups | `data/store/records.parquet.bak_pre_HGD569f109_20260817`, `data/store/maps.npz.bak_pre_HGD569f109_20260817` |
| anchor / mesh evidence | `data/reports/mesh_v3_20260817/README.md` §5e |
| pin-gate precedent | `data/reports/fpcamp_E1E2_f109_results_20260817.md` §7 |
| NULL-reading destination (not invoked here) | `data/reports/tripletype_design_20260817.md` |
