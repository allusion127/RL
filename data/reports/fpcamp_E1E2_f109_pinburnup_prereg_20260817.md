# Pin-burnup marks — PRE-REGISTRATION ADDENDUM to the E1_E2/f109 campaign

**Written 2026-08-17, mid-run, BEFORE any result exists.** At the moment of
writing the campaign stood at **8 / 100 MASTER calls** (wave 0 complete, wave 1
in flight), best F_r 1.5790, and **zero feasible cores found**. Nothing below was
chosen after seeing an outcome.

Trigger: the autoeng builder's precheck reports a **node pin-burnup cliff in
parts of the f109 column** — `min_fr_max_cycle` does not gate pin burnup, and the
frontier note under-listed this. This addendum pins the marks for the running
cell, states what the harvest can and cannot measure, and fixes the verdict rule
in advance. It changes no execution: the deck is hashed and running unmodified.

---

## 1. The cliff is real, and the metric reading is confirmed

Reproduced from `data/reports/dbx_frontier_table.csv` (per-core DB pin columns).
The builder's example checks out exactly: **N1_N2/f109 has 41 DB cores (21 + 20
across two split buckets) and `frac_node_ge75 = 1.00` in both** — every one
exceeds the 75 GWd/tU node limit. That cell is a licensing dead end for delivery
regardless of its F_r.

## 2. MARKS FOR THIS CAMPAIGN'S CELL — E1_E2 / feed 109 (PINNED)

> **E8 소급 기재 (2026-08-20)**: 이 사전등록이 §3 에서 "세 스케일(우리
> `max_pin_burnup` · DB `pinmax_rodavg` · DB `pinmax_node`)은 서로 비교불가"라고
> 등록한 것은 **옳았고**, 이제 그 대응이 확정되었다: 우리 `max_pin_burnup` ≡ DB
> **`pinmax_node_GWd` 와 동일 계층**(핀 노드 첨두), DB `pinmax_rodavg_GWd` 는 우리
> 보조 관측량 `max_rod_avg_burnup` 쪽이다. **수치 수준은 여전히 같지 않다**
> (공유 셀 +9~20%, 공통 코어 0건). 공식 한도 **80 GWd/tU 는 노드 축**에 건다
> (`data/reports/pinbu_definition_20260820.md`). 아래 §2 의 DB 수치는 **DB 자신의 축**으로 읽을 것.

44 DB cores, 43 feasible, one split bucket (55-60), `n_pinmax_known = 44`:

| quantity | value |
|---|---|
| **best core** `pinmax_rodavg_GWd` | **71.68** |
| **best core** `pinmax_node_GWd` | **78.43** |
| best core `rodavg_ge75` | **False** (passes) |
| best core `node_ge75` | **True** (**exceeds**) |
| all 44: `pinmax_rodavg` min / mean | 67.97 / 70.57 |
| all 44: `pinmax_node` min / mean | 73.36 / 76.40 |
| **`frac_rodavg_ge75`** | **0.000** — none exceed on rod-average |
| `frac_node_ge68` | 1.000 |
| **`frac_node_ge75`** | **0.886** — 39 of 44 exceed on node |

**Registered reading of this cell, before results:** E1_E2/f109 is **not** a dead
end like N1_N2/f109 (0.886 vs 1.000), but it is **substantially pin-pressured**.
The rod-average axis is clean (0% ≥ 75, and all 44 sit far under the LEU+ 80
figure lpopt uses for `*_pin_bu_limit`). The **node** axis is where the pressure
is: the DB's own best-F_r core here is at **78.43**, over the 75 limit, and only
**5 of 44** DB cores (11.4%) come in under it, with the floor at 73.36.

So the pre-registered expectation is explicit: **if this campaign opens the cell
on F_r, its cores are more likely than not to sit above the 75 GWd/tU node limit,
because 88.6% of the DB's cores in this exact cell do.**

### Context cells

| cell | `frac_node_ge75` | best core node | reading |
|---|---|---|---|
| N1_N2 / f109 | **1.000** (41/41) | 79.97 / 82.27 | dead end |
| **E1_E2 / f109 (this run)** | **0.886** (39/44) | **78.43** | pin-pressured |
| N1_N2 / f113 (closed campaign) | 0.378 (17/45) | **73.30** (passes) | comparatively benign |

## 3. WHAT THE HARVEST CAN AND CANNOT MEASURE — registered now, not discovered later

**The campaign will produce NO measured pin burnup.** This is not a prediction;
it is verified from the closed f113 run and the code:

* All **98 / 98** converged f113 campaign rows have `max_pin_burnup = null`.
  `max_assembly_burnup` is present on 98/98.
* Only **1,763 of 14,490** converged ga80 store rows (12.2%) carry
  `max_pin_burnup` at all; at E1_E2/f109 it is **33 of 292**.
* Cause: pin burnup is harvested only when the verifier's evaluator sets
  `enable_pin_burnup=True`. `curriculum.py:29` states plainly that *"the default
  produce verifier does not"*. The flag is reachable only as
  `[design].enable_pin_burnup` (config.py:721, consumed by the design /
  bootstrap / pathfinder paths) — **there is no deck knob that turns it on for
  `lpopt optimize`.** So the brief's instruction (2), read literally ("store rows
  carry the column where available"), cannot be honoured for this campaign's own
  cores: the column will be empty for all of them.

**The store's `max_pin_burnup` is NOT a drop-in substitute for either DB column.**
At E1_E2/f109 the 33 rows that do carry it read **76.12 / 83.33 / 93.64**
(min/p50/max) against DB `pinmax_rodavg` 67.97-71.68 and `pinmax_node`
73.36-78.43. It is on a different — higher — scale than both. Two confounds, not
one: the metric definition may differ, **and** those 33 rows are produce-sampling
cores at F_r ≥ 1.7692, a different core population from the DB's feasible
low-F_r set. Substituting one for the other would be a false comparison, so this
report will not make it.

### Therefore, what §"pin burnup" of the results report WILL contain

1. **Model-predicted `max_pin_burnup`** for the campaign's feasible cores. s1g
   does predict it — `target_names` index **6** of
   `['f_r','f_q','cbc_max','cyclen','ao_abs','discharge_burnup','max_pin_burnup','max_assembly_burnup']`.
   Reported as a **prediction, labelled as such**, with its calibration status
   stated. It is evidence, not a measurement, and will not be called one.
2. **Measured `max_assembly_burnup`** (expected ~100% coverage), reported as the
   related-but-distinct quantity it is — assembly-average, not pin, and not
   comparable to the 75/80 limits.
3. **The DB-side verdict for the cell** (section 2 above), which is the strongest
   real evidence available about the pin-burnup character of this cell.
4. **An explicit statement of what would settle it**: a pin-BU-enabled
   re-verification of the campaign's top feasible cores (the curriculum-style
   WaveVerifier path, or the design/bootstrap path, both of which set
   `enable_pin_burnup=True`). That is a small, bounded MASTER job on ~5 cores and
   is the recommended follow-up — not something this campaign can retrofit.

## 4. VERDICT RULE — fixed in advance

Applied to whatever the harvest shows, in this order:

* **If the campaign finds no feasible core** — pin burnup is moot; report the DB
  cliff as context for any future attempt at this cell.
* **If feasible cores are found and the evidence (DB cell character + predicted
  pin BU) indicates they systematically exceed the 75 GWd/tU node limit** — the
  headline verdict is **"opened for F_r but pin-limited for delivery"**. The F_r
  number is reported *after* that sentence, not before it, and the campaign is
  **not** described as having produced a deliverable core.
* **If the evidence is genuinely mixed or the prediction is too weak to
  adjudicate** — say exactly that, recommend the pin-BU re-verification of §3.4,
  and do **not** resolve the ambiguity in the campaign's favour.

## 5. A correction owed to the closed f113 report

`data/reports/fpcamp_N1N2_f113_results_20260816.md` headlines **F_r 1.4961**
without any pin-burnup treatment, and its winner's `max_pin_burnup` is `null`
for the reason given in §3. By the standard the builder is applying here, that
report is incomplete. N1_N2/f113 is the mildest of the three cells above
(`frac_node_ge75` 0.378, and the DB's best core there passes at 73.30), so the
correction is unlikely to overturn its result — but "unlikely to overturn" is not
"checked". A pin-burnup section is being added to that report stating the DB cell
character, the null-column fact, and the same unresolved-until-re-verified
status. Flagged here so the omission is on the record from the moment it was
noticed, not from whenever it is fixed.
