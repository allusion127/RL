# High-Gd boron-opened cell, F_r assault, PIVOT TO FEED 125 — `P6253Z1G06N24_P6253Z2G10N24` / feed 125 — RESULTS

**Run 2026-08-17 on box 199.** Deck `fpcamp_minfr_HGD569_f125_199.inp`, run
dir `runs/fpcamp_minfr_hgd569_f125`, champion `data/models/s1g`, seed 5695,
`rc=0`. Pre-registration: `data/reports/fpcamp_HGD569_f125_prereg_20260817.md`
(written before launch, deck-hash gated, drafted by a prior agent). Scored
against the readouts registered there (§3): **R1, R2 (PRIMARY), R3, R4,
R-PIN** — all five answered below, none required a mid-run amendment.

---

## 1. HEADLINE VERDICT: STRETCH met cleanly — beats f109's own state of the art, gate still not reached, and R-PIN fires

| mark (prereg §4) | requirement | result | |
|---|---|---|---|
| PRIMARY | any MASTER-verified core passing all 4 gates | 0 / 57 converged | not met |
| **SECONDARY** | F_r < 1.8375 (beat f125's own anchor) with CBC ≤ 1600 | **1.6357**, CBC 1565.46 | ✅ met |
| **STRETCH** | F_r ≤ 1.6743 (match/beat f109's dedicated-campaign winner) | **1.6357** (−0.0386), CBC-clean | ✅ met |
| REACH | F_r ≤ 1.55 (the licensing gate) | 1.6357, +0.0857 over | not reached |
| PARTIAL | F_r improves but not past ~1.6743, or CBC crosses 1600 while F_r falls | superseded by STRETCH; CBC pressure documented separately (§4/R3) | n/a |
| NULL | floor doesn't move meaningfully below 1.8375 | floor moved **−0.2018** (to 1.6357, CBC-clean) / **−0.2287** (to 1.6088, raw) | **does not apply** |

The CBC-clean joint frontier (F_r ∧ CBC ∧ F_q ∧ AO all passing except F_r
itself) sits at **1.6357** — beating f109's own dedicated-campaign winner
(1.6743) by **−0.0386**, on a *weaker* elite seed (prereg §6: no warm-start
from f109's improved rows) and a *thinner* CBC margin (prereg §1). A raw
(CBC-violating) core at **1.6088** exists but is not a fair "beats the state
of the art" claim since f109's 1.6743 was itself CBC-clean; the honest,
apples-to-apples comparison is the 1.6357 core. **R-PIN — the headline
readout for this deck — also fires**: predicted pin burnup at every one of
the top F_r cores (76.96–77.96) sits below both f109's winner (81.13) and the
78 deliverability gate, confirming the residence-time argument in the
pre-registration's §1 (see §6 below).

---

## 2. R1 — best F_r vs both pinned marks

| | joint-clean (all gates but F_r) | raw best-overall |
|---|---|---|
| **F_r** | **1.6357** | 1.6088 |
| record_id | `8b9acbcda6c73a37d3c3b2842daf1ea350f966f0d34bcdd9cbfc6b301950e3dc` | `e871b935696de61c84b58abd61c175a9c0938e9c6ac3012a99023127e0fc7f13` |
| CBC_max | 1565.46 (≤1600 ✅) | 1614.50 (**+14.50 over 1600** ❌) |
| F_q / \|AO\| | 2.0346 / 0.0266 | 2.0208 / 0.0247 |
| cyclen | 730.850 | 731.033 |
| n_cycles / restart | 11 / `pair_ecore:MAS_RST.APRQ_11_0705.02` (level 3, as predicted) | 11 / same |
| found at | call 11 / 60, wave 1 | call 17 / 60, wave 2 |
| predicted pin BU (s1g, R-PIN) | **76.955** | 77.235 |

`state.json → best_overall` tracks the raw core (1.6088) because the
in-campaign objective ranks on F_r alone, not CBC — same defect noted for
`lpopt report`'s best-patterns table in the f109 report (§8.6 there); the
honest state-of-the-art for this cell/feed is the CBC-clean 1.6357 core.

### Against the registered marks

| mark | value | delta (joint-clean 1.6357) | delta (raw 1.6088) | |
|---|---|---|---|---|
| **in-cell f125 anchor floor (like-for-like)** | 1.8375 | **−0.2018** | −0.2287 | ✅ new floor either way |
| **f109 dedicated-campaign winner (cross-feed SOTA)** | 1.6743 | **−0.0386** | −0.0655 (CBC-infeasible) | ✅ beaten cleanly |
| 1.55 licensing gate | 1.55 | +0.0857 | +0.0588 | not reached |
| precedent-based optimistic projection (prereg §2c) | 1.4637 | +0.1720 | +0.1451 | not reached — the optimistic case did not materialize |

Convergence: **57 / 60 (95.0%)** — 3 errors, all `non_finite_flux` (calls 6,
7, 29), a materially higher rate than f109's own campaign (66.7%) despite the
same never-run-pair / level-3-resolution / no-cache conditions.

---

## 3. R2 — PRIMARY: 4-constraint joint pass count

**0 / 57 converged.** No core passed
F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30 simultaneously — F_r is
again the sole blocker, but CBC is now genuinely close behind it (contrast
f109, §4):

| gate | pass rate (n=57 converged) | binding? |
|---|---|---|
| F_r ≤ 1.55 | **0 / 57 (0%)** | **yes — the only one that fully blocks** |
| CBC ≤ 1600 | **16 / 57 (28.1%)** | **yes — thin, and materially binding** |
| F_q ≤ 2.41 | 52 / 57 (91.2%) | mildly |
| \|AO\| ≤ 0.30 | 57 / 57 (100%) | no |

**Joint pass (CBC ∧ F_q ∧ AO, sans F_r): 13 / 57 (22.8%)** — the population
that determines whether the campaign found a "clean except F_r" core. It did:
the R1 winner (1.6357) is a member of this 13, and stands 4 of 5 gates clear
(the fifth, predicted pin BU, also clears — §6) with F_r as the sole open
axis. That is a new milestone for the programme: the closest any
MASTER-verified core has come to 5-gate feasibility (F_r ≤ 1.55 excluded).
Adding the predicted-pin gate (≤78) changes nothing to the 4-constraint
count (it cannot subtract from an empty F_r-feasible set): **5-constraint
joint pass count: 0 / 57**, same as the 4-constraint count.

---

## 4. R3 — CBC under F_r pressure: the registered risk materialized

The pre-registration flagged this as "high stakes" (thin 38.69 ppm anchor
margin vs f109's 194.84 ppm) and it was right to. Unlike f109 (CBC never
within 50 ppm of binding, 40/40 pass), **CBC became a real, binding
constraint at f125**:

| | f109 campaign (n=40) | f125 campaign (n=57) |
|---|---|---|
| CBC min / p50 / max | 1420.37 / 1507.32 / 1550.09 | **1532.23 / 1614.63 / 1668.89** |
| CBC ≤ 1600 pass rate | 40 / 40 (100%) | **16 / 57 (28.1%)** |
| Pearson r(F_r, CBC) | not binding, no trade | **−0.20 all rows / −0.45 excl. control** |

The **median** converged core at f125 sits *above* the CBC gate — a genuine
reversal from f109. The negative correlation confirms a real trade-off:
pushing F_r down tends to push CBC up at this feed, unlike the F_r/F_q pair
(§5, which move together). **No F_r-driven candidate crossed 1600 at the
frontier itself** — the R1 winner's CBC (1565.46) still clears with 34.54 ppm
to spare — but that margin is thinner than even the anchor's own margin
(38.69 ppm), meaning the search did not find slack to spend on CBC as it
minimized F_r; it operated right at the edge of it. The (F_r, CBC) Pareto
front, joint-clean cores only, sorted by F_r:

| F_r | CBC_max | F_q | \|AO\| | cyclen | wave | call | pred. pin BU |
|---|---|---|---|---|---|---|---|
| **1.6357** | 1565.46 | 2.0346 | 0.0266 | 730.850 | 1 | 11 | 76.96 |
| 1.6606 | 1563.48 | 2.0480 | 0.0290 | 729.287 | 1 | 13 | 77.14 |
| 1.6625 | 1596.41 | 2.0935 | 0.0267 | 730.621 | 5 | 46 | 77.96 |
| 1.6644 | 1532.45 | 2.0428 | 0.0294 | 726.379 | 0 | 1 | 76.97 |
| 1.6733 | 1564.77 | 2.0957 | 0.0267 | 730.580 | 0 | 3 | 77.35 |

(Full 13-row joint-pass set behind these top 5; CBC ranges 1532–1600 ppm
across all of them, i.e. every joint-clean core sits in the *upper* half of
the gate's own range — consistent with the trade-off above.)

---

## 5. R4 — the F_q axis: expected non-binding, holds under search pressure

The prereg expected F_q to be non-binding (in-cell anchor floor already
passing, 2/16). It holds:

| | f109 campaign | f125 campaign |
|---|---|---|
| F_q min / p50 / max | 2.082 / 2.1968 / 3.5481 | **2.0173 / 2.0733 / 2.8683** |
| F_q ≤ 2.41 pass rate | 38 / 40 (95%) | **52 / 57 (91.2%)** |
| Pearson r(F_r, F_q) | +0.637 (report-level, not per-row) | **+0.978** |

F_r and F_q move together even more tightly at f125 than at f109 (r=0.978 vs
the reported +0.637) — driving F_r down keeps dragging F_q down with it, so
F_q never became a second binding axis. AO also moves with F_r here (r=0.659,
new — not reported for f109) but was never close to its own gate (max 0.049
vs limit 0.30).

---

## 6. R-PIN — the headline readout: residence-time argument confirmed in direction, not in full magnitude

**Verdict: predicted pin BU falls meaningfully below both f109's winner and
the 78 gate, at every one of the top F_r cores served.** Served via the s1g
ensemble, same path as `mesh_vs_db.py --model s1g`
(`PosValCnnBackend.predict(...).mean[:, 6]`, surrogate column 6 =
`max_pin_burnup`, exact-match per `lpopt/model/model_api.py`), for the top 5
joint-clean cores plus the raw best-overall core (6 unique cores; f109's
winner re-served through the identical code path as a same-run consistency
check — recovered **81.131**, matching the previously reported 81.13 to
0.001, confirming apples-to-apples comparability):

| tag | record_id (short) | F_r | CBC | predicted pin BU |
|---|---|---|---|---|
| joint best | `8b9acbcd…` | 1.6357 | 1565.46 | **76.955** |
| joint #2 | `0884517b…` | 1.6606 | 1563.48 | 77.138 |
| joint #3 | `d9cf666d…` | 1.6625 | 1596.41 | 77.963 |
| joint #4 | `9d28d004…` | 1.6644 | 1532.45 | 76.968 |
| joint #5 | `3b9d9cc2…` | 1.6733 | 1564.77 | 77.355 |
| raw best-overall | `e871b935…` | 1.6088 | 1614.50 (CBC-fail) | 77.235 |
| **served range** | | | | **76.955 – 77.963** |
| f109 winner (re-served, same path) | `614c83b9…` | 1.6743 | 1540.99 | **81.131** |

> **정의 각주 (2026-08-20 사용자 확정)**: 핀연소도 한계치 **80 GWd/tU** 는
> **핀 axial peak** — 즉 우리가 측정/예측해 온 `max_pin_burnup`(3-D 핀 노드
> 첨두) — 에 건다. 봉평균은 보조 관측량이며 판정축이 아니다. 아래 판정은
> 공식 관측량 그대로이므로 **유효하다**. `data/reports/pinbu_definition_20260820.md` · `pinbu_audit_20260820.md` §8.
>
> **실측 확증 (2026-08-20)**: 이 5기를 MASTER 로 재측정한 결과 **75.47–76.49**
> (joint best 실측 **75.47**) — **5/5 PASS**, 80 한도 대비 여유 3.5–4.5 GWd/tU.
> R-PIN 의 방향·수준 모두 확증. `pinbu_wave_results_20260820.md` §2
> (`HGD569_f125_2type`).

**Against both pinned questions in the prereg:**

1. **Does predicted pin fall below f109's winner (81.13)?** Yes, at every
   served core — by **−3.17 to −4.18**, median **−3.94**.
2. **Does it fall to ≤ 78 (this campaign's own gate)?** Yes, at **all 6**
   served cores, with margin **0.04 – 1.05** GWd/tU to spare. The joint-best
   core (F_r 1.6357) clears the pin gate with **1.045** to spare — it is 4 of
   5 gates clean (CBC, F_q, AO, predicted-pin), F_r alone still open. Had F_r
   also cleared 1.55 this would be the first 5-constraint-passing core in the
   programme; it does not, so that milestone remains open, but this is the
   closest any core has come to it.

**Magnitude check against the prereg's own naive projection.** §1 scaled
f109's winner (81.13) by the residence ratio 241/125 ÷ 241/109 = 0.872,
projecting **~70.7 GWd/tU**. The actual served value at the equivalent core
(joint best, 76.96) is **higher** than that naive projection by **+6.3** —
the drop is real and gate-clearing, but roughly **40% smaller** than a pure
residence-scaling argument implied. The prereg itself flagged why (§1): peak
pin burnup depends on local power peaking, which the search is
simultaneously reshaping, not only on core-average residence — that caveat
is confirmed by the data. **Net reading: the residence-time argument is
validated in direction and in the practical outcome that matters (clears the
78 gate at the F_r frontier), not in the naive linear magnitude.**

*Caveat, unchanged from f109*: this is the surrogate's **prediction**, not a
measured pin burnup — `enable_pin_burnup` is unreachable from an `optimize`
deck, so 0/57 rows carry a measured value. `state.json`'s own `best_overall`
entry also carries `max_pin_burnup: null`, confirming no in-campaign
verification occurred. No independent re-verification run has been made on
any of the 6 served cores.

---

## 7. Wave trajectory and call-order dynamics — the frontier stalled early, unlike f109

Unlike f109 (best core found on the very last call, 57/60, budget arguably
undersized), **f125's frontier converged fast and then went flat**:

| wave | n conv | n joint | F_r min | F_r med | cum min (raw) | cum min (joint) | cyclen range |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 4 | 1.6644 | 1.7046 | 1.6644 | 1.6644 | 709.1–733.7 |
| 1 | 8 | 6 | **1.6357** | 1.6746 | 1.6357 | **1.6357** | 725.8–731.0 |
| 2 | 8 | 2 | **1.6088** | 1.6829 | **1.6088** | 1.6357 | 725.3–731.2 |
| 3 | 7 | 0 | 1.6394 | 1.6570 | 1.6088 | 1.6357 | 729.6–733.3 |
| 4 | 8 | 0 | 1.6419 | 1.6637 | 1.6088 | 1.6357 | 730.5–734.0 |
| 5 | 8 | 1 | 1.6307 | 1.6632 | 1.6088 | 1.6357 | 727.7–732.7 |
| 6 | 8 | 0 | 1.6443 | 1.6583 | 1.6088 | 1.6357 | 730.3–733.7 |
| 7* | 4 | 0 | 1.6327 | 1.6591 | 1.6088 | 1.6357 | 731.5–736.6 |

**Both marks — first sub-1.8375 (anchor floor) and first sub-1.6743 (f109
SOTA) — were beaten on call 1, wave 0** (F_r = 1.6644 already clears both).
The raw cumulative minimum stopped moving after **call 17 (wave 2)**; the
joint-clean cumulative minimum stopped moving even earlier, after **call 11
(wave 1)** — **43 to 49 further calls out of 60 produced zero improvement**
on either frontier. This is the opposite pattern from f109 (still descending
at wave 7) and closer to the E1_E2/f109 precedent (frontier stalled early,
budget oversized).

One nuance: the finetune gate did **not** reject all 8 waves here (contrast
f109, which rejected all 8): `gate=objective-` for waves 0–5, then
**`gate=objective+` for waves 6 and 7** — two in-campaign model updates did
land. They did not translate into further F_r improvement (§ table above),
consistent with `state.json`'s `no_improve: 8` (the run's own feasible-core
improvement counter — `best` stayed `null` throughout, since 0/57 ever passed
all 4 gates). `post_verify_violators: []` confirms no post-hoc constraint
violations were caught either.

---

## 8. Comparison table — f109 vs f125, same pair/cell

| axis | f109 (closed, this pair/cell) | f125 (this campaign) | |
|---|---|---|---|
| joint-clean F_r frontier | 1.6743 | **1.6357** | −0.0386 |
| raw F_r frontier (may violate CBC) | 1.6743 (same, was CBC-clean) | 1.6088 | −0.0655 |
| cyclen at frontier | 675.392 | 730.850 | +55.458 |
| CBC at frontier | 1540.99 (59.01 ppm margin) | 1565.46 (34.54 ppm margin) | thinner margin |
| CBC campaign pass rate | 40/40 (100%) | 16/57 (28.1%) | CBC now genuinely binding |
| F_q at frontier | 2.0982 | 2.0346 | improved |
| F_q campaign pass rate | 38/40 (95%) | 52/57 (91.2%) | roughly similar |
| predicted pin BU at frontier | 81.13 | **76.96** | **−4.17, clears 78 gate** |
| convergence rate | 40/60 (66.7%) | 57/60 (95.0%) | notably higher |
| frontier still moving at wave 7? | **yes** (call 57/60) | **no** (stalled by call 17/60) | opposite dynamics |
| finetune gate | rejected all 8 waves | rejected 6, accepted 2 (waves 6–7) | partial acceptance, no payoff |
| 4-gate joint pass (n) | 0/40 | 0/57 | unchanged |
| 5-gate joint pass incl. pin | 0/40 | 0/57 | unchanged |
| joint-clean-except-F_r pass (4 of 5 gates) | not reported at f109 | **13/57 (22.8%)** | new best-ever proximity to full feasibility |

---

## 9. Store deltas

| | before | after |
|---|---|---|
| canonical store rows | 74,417 | **74,477** (+60) |
| HGD569 pair @ f125 rows | 24 | **84** (+60) |
| HGD569 pair @ f125 converged | 16 | **73** (+57) |
| **HGD569 pair @ f125 F_r floor** | 1.8375 | **1.6088** (raw) / **1.6357** (CBC-clean) |
| store backups | `data/store/records.parquet.bak_pre_HGD569f125_20260817`, `data/store/maps.npz.bak_pre_HGD569f125_20260817` |

---

## 10. Honest notes

1. **STRETCH met cleanly, PRIMARY not.** 1.6357 is a real, MASTER-verified,
   4-of-5-gate-clean F_r floor for this pair at f125 — better than f109's own
   dedicated-campaign winner on a weaker elite seed and a thinner CBC margin.
   It remains 0.0857 over the 1.55 gate. The lower raw value (1.6088) exists
   but is CBC-infeasible and should not be quoted without that caveat.
2. **CBC is no longer free at this cell.** The pre-registered risk (§1, R3)
   materialized: CBC pass rate collapsed from 100% (f109) to 28.1% (f125),
   and the median converged core sits above the 1600 gate. This is the first
   min_fr campaign in the programme where CBC showed a real, measurable
   trade-off against F_r (r=−0.45 excl. control) rather than acting as slack.
3. **Pin burnup is predicted, not measured**, same limitation as f109. The
   R-PIN analysis rests on the s1g ensemble's `pred_mean` column 6, served
   directly by re-running `PosValCnnBackend.predict()` on the 6 selected
   record patterns (not recovered from wave-selection JSON this time, since
   the top cores were identified after harvest) — the f109 winner was
   re-served through the identical path as a same-run sanity check and
   recovered 81.131 against the previously reported 81.13.
4. **The residence-time argument is validated in direction and in the gate
   outcome, not in the naive linear magnitude** (§6): actual pin drop is
   about 40% smaller than the prereg's own back-of-envelope projection
   (76.96 served vs ~70.7 projected), attributable to the reshaping of local
   power peaking that the prereg itself flagged as a caveat.
5. **The frontier stalled early — the opposite of f109.** Both cumulative
   minima (raw and joint-clean) stopped moving by call 17/60; the last
   ~45 calls produced no further F_r improvement despite two in-campaign
   finetune acceptances (waves 6–7). This is a genuine, registered-adjacent
   finding: 60 calls appears **oversized** for this cell/feed, unlike f109's
   own campaign, which was still improving when the budget ran out.
6. **Convergence rate (95.0%) is notably higher than f109's own campaign
   (66.7%)** despite identical never-run-pair / level-3-resolution / no-cache
   starting conditions — not explained by this analysis; flagged for anyone
   revisiting asset-resolution behaviour at this cell.
7. **No independent re-verification.** None of the 6 served cores has been
   re-run with `enable_pin_burnup` set; their *measured* pin burnup is
   unknown. That remains the only thing that would fully settle
   deliverability for the joint-best core.
8. **`lpopt report`'s / `state.json`'s own best-tracking surfaces the
   CBC-infeasible raw core (1.6088), not the CBC-clean joint core (1.6357)**
   — the in-campaign objective optimizes F_r alone. Anyone reading
   `state.json → best_overall` directly should apply the same CBC filter used
   here before quoting a "winner."

---

## 11. Next-step options

The frontier stalled by call 17/60 (§7) and R-PIN's headline question (does
predicted pin drop below 81.13/78?) is now answered — both push against
spending a further 60-call extension on *this exact cell/feed* under the
same s1g champion and elite seed:

* **60-call extension (same deck/cell/feed):** low expected value per §7 —
  no F_r improvement occurred in the last ~45 calls of the budget already
  spent, and the two late finetune acceptances (waves 6–7) did not move the
  frontier either. An extension would mainly re-sample the same plateau
  unless the elite pool or champion changes.
* **3-type track (a genuinely new lever):** more promising given F_r is
  confirmed as the sole blocker at both feeds for this pair/cell (2/2
  campaigns), and CBC has now shown it can bind too (§4) — a pattern change
  (not just a feed change) is the more likely lever to move F_r past 1.55 at
  this cell, per the same logic the prereg used to justify trying f125 after
  f109 (§1: "one lever changes... only one of which the f109 run could
  answer" — that lever is now spent).
* **Stop and consolidate:** both min_fr campaigns at this pair (f109, f125)
  are closed with STRETCH-level results and 0/97 total feasible cores; a
  reasonable checkpoint before committing further 60-call budgets is to
  fold R-PIN's result (pin burnup is not the bottleneck at this pair — F_r
  and, now, CBC are) into the programme-level picture across all closed
  min_fr campaigns before choosing the next cell/pair/lever.

This report does not choose between the three; it lays out the evidence
(§7's stall, §4's new CBC risk, §6's resolved pin question) that should drive
whichever is chosen next.

---

## 12. Paths

| artefact | path |
|---|---|
| deck (pre-registration, hashed) | `fpcamp_minfr_HGD569_f125_199.inp` |
| pre-registration | `data/reports/fpcamp_HGD569_f125_prereg_20260817.md` |
| launcher / bat / status | `launch_fpcamp_HGD569_f125_199.ps1`, `run_fpcamp_minfr_HGD569_f125_199.bat`, `status_fpcamp_HGD569_f125_199.ps1` |
| run dir (199) | `C:\Users\USER\lpopt_work\kit_frontier\runs\fpcamp_minfr_hgd569_f125` |
| campaign log (199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_hgd569_f125_out.log` |
| harvested labels / state / log | scratchpad `harvest_hgd569_f125/run/{labels.jsonl,state.json,fpcamp_minfr_hgd569_f125_out.log,report.md}` |
| pin-BU serve script + output | scratchpad `serve_f125_pin.py`, `f125_pin_serve.csv` |
| labels copy used for this analysis | scratchpad `f125_labels.jsonl` |
| canonical store (merged) | `data/store/records.parquet` (74,477 rows) |
| store backups | `data/store/records.parquet.bak_pre_HGD569f125_20260817`, `data/store/maps.npz.bak_pre_HGD569f125_20260817` |
| f109 sibling (closed) | `data/reports/fpcamp_HGD569_f109_prereg_20260817.md`, `data/reports/fpcamp_HGD569_f109_results_20260817.md` |
| pin-serve precedent | `mesh_vs_db.py` (`--model s1g` path, surrogate column 6) |
| pin-gate precedent (earlier campaigns) | `data/reports/fpcamp_E1E2_f109_results_20260817.md` §7 |
