# First 3-fresh-type (graded) campaign — `S3_T1_S5` / feed 125 — RESULTS

**Run 2026-08-17 on box 199.** Deck `fpcamp_minfr_TRIPLE_f125_199.inp`
(sha256 `B042D49A…42F1`), run dir `runs/fpcamp_minfr_triple_f125`, champion
`data/models/s1h` (v7), seed 5695, **rc = 0**, 8 waves, **60/60 calls**.
Pre-registration: `data/reports/tripletype_f125_prereg_20260817.md`, written
before the deck was hashed and before anything was launched. Every readout
registered there (R1, R2, R3, R4, R-PIN, R-GRADE, R-SEED) is answered below —
including the one that could not be answered.

---

## 1. HEADLINE: STRETCH met — grading beat the 2-type frontier, and the pin axis moved a long way

| mark (prereg §4) | requirement | result | |
|---|---|---|---|
| **PRIMARY** | any core passing F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30 | **0 / 49** | not met |
| **STRETCH** | **F_r < 1.6357** CBC-clean (beat the 2-type joint frontier) | **1.5993**, CBC 1597.33 | ✅ **met, −0.0364** |
| SECONDARY | F_r < 1.6088 (2-type raw best) | 1.5956 raw / 1.5993 clean | ✅ met on both readings |
| NULL | floor does not fall below 1.6357 | floor fell −0.0364 (clean) / −0.0132 (raw-vs-raw) | **does not apply** |

**The joint-clean core: F_r 1.5993, CBC 1597.33, F_q 1.9968, |AO| 0.0261,
cyclen 730.5 EFPD, predicted pin BU 75.53.** Found on **call 57 of 60**, wave 7
(reserve). Feed composition **hot / mid / cold = 57 / 20 / 48** assemblies.

It clears **four of five gates with margin** and misses only F_r, by **+0.0493**.
The 2-type predecessor at this identical cell, feed, budget, objective and gate
set missed by +0.0857. **Grading closed 43 % of the remaining distance to the
licensing limit.**

The raw best (1.5956) violates CBC by 3.24 ppm and is reported only as the raw
figure; the honest state of the art for this cell is the **CBC-clean 1.5993**.

---

## 2. R1 — against both pinned marks

| | 3-type joint-clean | 3-type raw | 2-type joint-clean | 2-type raw |
|---|---:|---:|---:|---:|
| **F_r** | **1.5993** | 1.5956 | 1.6357 | 1.6088 |
| CBC_max | **1597.33** ✅ | 1603.24 ❌ | 1565.46 ✅ | 1614.50 ❌ |
| F_q / \|AO\| | 1.9968 / 0.0261 | 2.0133 / 0.0264 | 2.0346 / 0.0266 | 2.0208 / 0.0247 |
| cyclen | 730.5 | 730.3 | 730.85 | 731.03 |
| predicted pin BU | **75.526** | 75.532 | 76.955 | 77.235 |
| found at | call 57/60 | call 58/60 | call 11/60 | call 17/60 |

| mark | delta (joint-clean 1.5993) |
|---|---|
| 2-type joint frontier **1.6357** | **−0.0364** ✅ |
| 2-type raw best 1.6088 | −0.0095 ✅ |
| 2-type f125 anchor floor 1.8375 | −0.2382 |
| f109 dedicated-campaign winner 1.6743 | −0.0750 |
| **1.55 licensing gate** | **+0.0493** — not reached |

Convergence **49 / 60 (81.7 %)**, 11 `non_finite_flux` errors — materially worse
than the 2-type sibling's 95.0 % at the same cell. Flagged, not explained.
Restart resolution was **100 % level-3 `pair_ecore:MAS_RST.APRQ_11_0705.02`** on
all 49, exactly as the prereg predicted; no neutral fallback occurred.

---

## 3. R2 — PRIMARY: gate pass counts

**0 / 49 passed all four.** F_r remains the sole blocker, but the constraint
landscape changed markedly versus 2-type:

| gate | 3-type (n=49) | 2-type (n=57) | |
|---|---:|---:|---|
| F_r ≤ 1.55 | **0 (0 %)** | 0 (0 %) | still the only full blocker |
| CBC ≤ 1600 | **29 (59.2 %)** | 16 (28.1 %) | **materially relieved** |
| F_q ≤ 2.41 | **49 (100 %)** | 52 (91.2 %) | relieved |
| \|AO\| ≤ 0.30 | 49 (100 %) | 57 (100 %) | slack |
| **joint (CBC ∧ F_q ∧ AO)** | **29 / 49 (59.2 %)** | 13 / 57 (22.8 %) | **2.6× more clean cores** |

Adding the predicted-pin gate (≤ 78) subtracts nothing: **8/8** of the top-F_r
cores pass it (§6). So the 5-constraint count is **0/49**, same as the
4-constraint count — F_r alone.

---

## 4. R3 — CBC under F_r pressure: the 2-type trade-off did not reproduce

| | 2-type f125 (n=57) | 3-type f125 (n=49) |
|---|---|---|
| CBC min / p50 / max | 1532.23 / **1614.63** / 1668.89 | 1571.34 / **1596.06** / 1668.09 |
| pass ≤ 1600 | 28.1 % | **59.2 %** |
| Pearson r(F_r, CBC) | **−0.45** (excl. control) | **+0.19** |

The 2-type run's median core sat *above* the boron gate; the 3-type run's sits
*below* it. And the sign of the F_r/CBC relationship **flipped** — at 2-type,
pushing F_r down pushed CBC up; at 3-type that antagonism is gone. This is the
largest single practical gain of the round and it was not a registered
expectation: R3 was written anticipating the 2-type risk would repeat.

**Mechanism, measured:** the mid-type fraction is what moves both axes, in
opposite directions —

| correlation over the 49 converged cores | |
|---|---:|
| r(mid-type assemblies, **F_r**) | **−0.42** — more mid ⇒ lower F_r |
| r(mid-type assemblies, **CBC**) | **+0.64** — more mid ⇒ higher CBC |

So the third type is a genuine control knob with a genuine price, and the
optimum is interior. That is the single most useful physical result of this
campaign.

## 5. R4 — F_q

Non-binding and **improved**: 100 % pass (49/49) versus 91.2 % at 2-type; the
winner's F_q is 1.9968 against the 2-type winner's 2.0346.

---

## 6. R-PIN — the deliverability axis moved ~1.5 GWd/tU

Predicted pin burnup (`s1h` surrogate column 6) of the top-8 F_r cores:

| | 3-type | 2-type reference |
|---|---|---|
| range | **75.227 – 76.040** | 76.96 – 77.96 |
| **joint-clean winner** | **75.526** | 76.955 |
| under the 78 gate | **8 / 8** | (all, but by ≤ 1.04) |
| margin to 78 | **≈ 2.5** | ≈ 1.0 |

> **정의 각주 (2026-08-20 사용자 확정)**: 핀연소도 한계치 **80 GWd/tU** 는
> **핀 axial peak** — 즉 우리가 측정/예측해 온 `max_pin_burnup`(3-D 핀 노드
> 첨두) — 에 건다. 봉평균은 보조 관측량이며 판정축이 아니다. 아래 판정은
> 공식 관측량 그대로이므로 **유효하다**. `data/reports/pinbu_definition_20260820.md` · `pinbu_audit_20260820.md` §8.
>
> **실측 확증 (2026-08-20)**: 3종 상위 5기를 MASTER 로 재측정한 결과
> **74.16–75.58**(승자 실측 **74.38**) — **5/5 PASS**. 2종 실측(75.47–76.49)
> 대비 계단화 이득도 실측으로 재현되었다(−1.4 GWd/tU 수준).
> `pinbu_wave_results_20260820.md` §2 (`HGD569_f125_3type`).

**R-PIN fires.** The margin on the programme's thinnest axis roughly doubled
while F_r simultaneously improved — the two did not trade against each other.
The mid-pick's prediction (§3 of the prereg: both arms predicted *below* the
2-type measured band) is borne out by the campaign's own served values.

**Limitation, unchanged and load-bearing:** this is **predicted**, not measured.
No core here was re-run with `enable_pin_burnup`. Measured pin burnup for the
winner remains unknown, and that is the one thing that would settle
deliverability.

---

## 7. R-GRADE — the optimizer KEPT the third type, and used it as a transition ring

The pre-registered falsification was: does the search drive the mid fraction to
the floor the guard leaves it (4 assemblies)? **It did not.**

| | mid-type assemblies (of 125 fresh) |
|---|---|
| all 49 converged | min 4 · **p50 12** · max 24 |
| 29 joint-clean | min 4 · **p50 8** · max 20 |
| **joint-clean winner** | **20** (16 % of the feed) |
| distribution | 4:11, 8:8, 12:10, 16:17, 20:1, 24:2 |

The winning composition is **57 / 20 / 48** — a *thin-to-moderate* middle band,
not the equal thirds the design note proposed (which would be ~42 each) and not
the floor. The mass of the population sits at 12–16.

**Registered reading:** the third type earns its place, but as a **transition
ring**, not as a third of the core. The design note's e-matched
(0.380/0.333/0.286) and equal-thirds compositions are both far from where the
search actually went, and future graded decks should not assume them.

---

## 8. R-SEED — REGISTERED AND NOT ANSWERABLE. The §4 confound stands.

R-SEED was the control that was supposed to separate "grading helped" from
"better seeding helped". **It failed on power, and the failure is structural, not
bad luck.**

Only **2** converged cores in the entire run have a `parent_record_id` pointing
at a 2-type store row — both in wave 0, and they disagree (−0.0143 and +0.0582).
n = 2 is not a measurement. The cause: **41 of the 49 converged cores came from
the `local` hill-climb**, whose parent is an *in-campaign* candidate, so the
parent chain leaves the store after wave 0 and the paired comparison has almost
no support.

**Consequence, stated plainly.** This campaign's elite pool was seeded with the
2-type campaign's own converged output (129 donor rows, best 1.6088), while the
2-type campaign it is being compared against ran from an all-infeasible
backfill. **The −0.0364 improvement is therefore confounded: grading and better
seeding both changed, and this run cannot apportion them.** The prereg said so in
advance (§4, "Registered asymmetry") and nothing here removes it.

What *is* unconfounded, and worth more than the headline number:

* **§4's mechanism correlations** (r(mid, F_r) = −0.42, r(mid, CBC) = +0.64) are
  *within-campaign*, computed across 49 cores that all had the same seeding.
  Seeding cannot produce a within-run dose-response relationship between the
  mid-type fraction and two independent physics axes.
* **§7's composition result** — the optimizer, free to sit at the 4-assembly
  floor, chose 20 for its best core.

Those two say grading does real work. The headline F_r delta, on its own, does
not.

**The clean experiment, named now rather than after the fact:** re-run the
*2-type* deck at this cell with `elite_seed_cases` pointing at the same donor
population. Same seeding, one type fewer. That single run separates the two, and
it is cheap — the deck already exists.

---

## 9. The wave gate never fired, exactly as registered

**0 finetune accepts, 0 rejects.** The case has zero store rows, so
`_holdout_rows` was empty and the online gate could neither veto nor halt — read
as `explore`/`objective±` throughout. Registered in the prereg §7 as blindness,
not later reinterpreted as a pass. All improvement came from the base v7
champion plus BO refinement, not from in-campaign updates.

**The frontier had NOT stalled at the budget.** The cumulative joint-clean
frontier moved at calls 1 → 3 → 9 → 19 → **57**:

`1.6579 → 1.6486 → 1.6105 → 1.6036 → 1.5993`

The best core arrived on the **second-to-last call**. This is the f109 pattern,
not the 2-type f125 pattern (which stalled at call 17). **Budget 60 is
undersized for this cell in its 3-type form**; an extension is a new decision
with a new deck, not a silent top-up.

---

## 10. Honest notes

1. **STRETCH met, PRIMARY not.** 1.5993 is a real, MASTER-verified, 4-of-5-gate
   core, 0.0493 short of the licensing limit. It is the best core the programme
   has produced at this cell by any measure.
2. **The improvement is confounded with seeding** (§8). The strongest
   unconfounded evidence for grading is the within-run dose-response, not the
   frontier delta.
3. **Pin burnup is predicted, not measured** (§6). Unchanged limitation.
4. **Convergence fell to 81.7 %** from the 2-type sibling's 95.0 % under
   identical resolution conditions. 11 `non_finite_flux` failures. Unexplained;
   flagged for anyone touching graded-core deck generation.
5. **The model was blind here in a strong sense.** `s1h` had never seen a 3-type
   input and two of its five composition channels were constant in training
   (prereg §7). It nonetheless steered a 60-call budget to a −0.0364 improvement.
   Read that as "the v6c/v7 backbone generalized", not as "the composition
   channels worked" — nothing here can attribute it to them.
6. **`e_split` is NaN on all 60 rows.** Design-note checklist item 7 asked
   whether a triple's stored `e_split` fills with hot−cold. It does not — but
   this is **not a triple-specific defect**: every `optimize`-path row in the
   store has NaN `e_split` (the 2-type f125 and N1_N2 f113 campaigns included);
   only the `produce` path writes the column. The featurizer computes
   `g_e_split` from the pattern at serve time, so nothing downstream is
   affected. Checklist item 7 is answered "not applicable to this path".
7. **`state.json → best_overall` again surfaces the CBC-violating raw core**
   (1.5956), not the CBC-clean 1.5993. Same defect as both 2-type runs; apply
   the CBC filter before quoting a winner.
8. **Two search knobs were added for this campaign** (`elite_seed_cases`,
   `require_all_fresh_types`), both default-off and byte-identical for existing
   decks, both covered by `tests/test_elite_seed_cases.py` (9 tests). The second
   one was load-bearing: 41 of 49 converged cores came from the `local`
   hill-climb, which without the guard walks a graded board straight back to two
   types — as it did in the pre-launch dry run (5 of 8 wave-0 picks).

---

## 11. Provenance

| item | value |
|---|---|
| deck | `fpcamp_minfr_TRIPLE_f125_199.inp` sha256 `B042D49AFC274EA2DA630627D49DD0518205E71EA9E44BC41F90BF39D7E342F1` |
| model | `data/models/s1h` (v7, 18 globals, 5 members) — promoted this same day, `data/reports/gate_s1h.json` PASS |
| case | `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24` (alias `S3_T1_S5`), feed 125, paramA |
| store before / after | 74,477 → **74,537** (+60); triple cell 0 → 60 rows / 49 converged |
| store backups | `data/store/records.parquet.bak_pre_TRIPLEf125_20260817`, `data/store/maps.npz.bak_pre_TRIPLEf125_20260817` |
| merge | `lpopt merge-store` — 60 new / 0 upgraded / 74,477 duplicate, 147 maps merged, +0 ledger |
| harvested run | scratchpad `harvest_triple_f125/runs/fpcamp_minfr_triple_f125/{labels.jsonl,state.json,waves/,report.md}` + `fpcamp_minfr_triple_f125_out.log` |
| fleet | 199 only; 198 / 181 / 238 untouched |
