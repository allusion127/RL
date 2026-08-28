# PRE-REGISTRATION — high-Gd boron-opened cell, F_r assault

**`P6253Z1G06N24_P6253Z2G10N24` / feed 109 / paramA (n_gd 24, e_core 5.694) — box 199**

Written **2026-08-17, before launch.** Nothing below was chosen after seeing a
result. The deck (`fpcamp_minfr_HGD569_f109_199.inp`) is hashed and the launcher
refuses to start if the hash does not match.

| | |
|---|---|
| objective | `min_fr_max_cycle`, λ_Fr = 1000 |
| gates | F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30 ∧ **predicted pin BU ≤ 78** |
| cyclen | **no band** — recorded, subordinate tie-break only |
| budget | **60** (7 waves × 8 + 4 reserve) |
| model | s1g (8th champion), `library_id = paramA` |
| seed | 5694 (fresh) |
| run dir | `runs/fpcamp_minfr_hgd569_f109` on 199 |

---

## 1. Why this cell

The mesh-v3 anchors (2026-08-17, box 198, 254 chains / 178 converged rows,
`data/reports/mesh_v3_20260817/README.md`) broke the soluble-boron wall at high
enrichment — and broke it **widest here, not at the flagship**:

| pair | e_core | feed | CBC floor | cores ≤ 1600 |
|---|---|---|---|---|
| **`P6253Z1G06N24_P6253Z2G10N24`** (this) | 5.694 | **109** | **1405.2** | **16 / 16** |
| 〃 | 5.694 | 125 | 1561.3 | 4 / 16 |
| `P6253Z1G06N24_P6253Z2G06N24` | 5.690 | 109 | 1555.2 | 1 / 10 |
| `P6656Z1G06N24_P6661Z2G10N24` (flagship) | 6.139 | 109 | 1584.0 | 1 / 24 |

Every one of the sixteen sampler cores at this cell clears the 1600 ppm limit,
with **195 ppm of margin at the floor**. The flagship cleared it by 16 ppm on one
core in twenty-four. Boron has stopped being the binding constraint here; F_r has
become it. README §5e-4 names this campaign as the next step in as many words.

## 2. THE MARKS — pinned

### 2a. In-cell truth (this pair @ feed 109)

Store campaign tag `mv3_e569_f109` holds 23 rows at this cell, of which **16
converged and carry a FOM** — the other 7 are cy1-bootstrap chains that produced
none. Across both feeds the pair holds 47 rows / **32 converged**, which is
exactly the 32 elite slots (§6). Verified on box 199 after the store ship:

| quantity | min | p50 | max | vs gate |
|---|---|---|---|---|
| **CBC_max** | **1405.16** | 1497.88 | 1599.63 | **16 / 16 pass** 1600 |
| **F_r** | **2.0481** | 2.2639 | 3.0851 | 0 / 16 pass 1.55 |
| **F_q** | **2.6120** | 2.9465 | 4.3621 | **0 / 16 pass** 2.41 |
| \|AO\| | 0.0196 | 0.0285 | 0.0977 | 16 / 16 pass 0.30 |
| cyclen | 634.39 | 651.27 | 665.95 | (ungated) |
| e_core | 5.6649 | 5.6919 | 5.7256 | resolver e_core 5.6944 |
| max_assembly_burnup | 72.999 | 82.104 | 91.616 | (reported) |
| max_pin_burnup | — | — | — | **not measured on a single row** |

`n_pass_cbc = 16`, `n_pass_both = 0` (anchor readout).

### 2b. TWO CORRECTIONS TO THE BRIEF — on the record before launch

**(i) 1.8375 is not this cell's number.** The brief that ordered this campaign
pinned "CBC floor 1405.2 ppm / F_r floor 1.8375" as one cell's pair of marks.
They are two different cells:

* **1405.2 ppm is correct** — this pair at feed 109.
* **1.8375 is the SAME PAIR at feed 125**, not 109. It is the F_r of the best of
  the 22 CBC-passing anchor cores across the whole anchor set (F_r 1.8375,
  F_q 2.3074, CBC 1566.7, \|AO\| pass, 716.2 EFPD, e_core 5.675) — README §5e-4's
  "first high-enrichment core to pass three of the four constraints".
* **This cell's own f109 F_r floor is 2.0481**, +0.21 worse.

Both are pinned, each against its own cell. Readout R1 scores the campaign
against **2.0481** (like-for-like) and reports 1.8375 separately and explicitly
labelled cross-feed. Scoring an f109 result against an f125 mark would be a false
comparison and is barred in advance.

**(ii) The distance is larger than the precedent's own move.** 2.0481 → 1.55 is
**−0.498** in-cell. The precedent the README cites is T6_T4/f121: produce
byproduct floor 1.79 → dedicated min_fr campaign record 1.4749, **−0.32**. A
−0.32-sized move from this cell's f109 floor lands at ~1.73, not inside 1.55.
**This campaign is therefore NOT expected to reach the gate on the precedent's
own scale.** It is expected to measure how much of the 0.498 closes. Registering
that now removes the option of calling a 1.75 result a disappointment or a 1.65
result a triumph after the fact.

### 2c. The third constraint nobody has been watching — F_q

In-cell F_q floor is **2.6120** and **every one of the 16 anchor rows fails the
2.41 gate.** At f125 the same pair reaches 2.2810 and passes. So at this feed the
cell may be **F_q-closed, not F_r-closed**. This is registered as a co-equal
question (R3), not a footnote, precisely because the campaign objective does not
minimise F_q — it only gates it. A run that drives F_r to 1.6 while F_q sits at
2.7 has not opened the cell, and the report must say so in those words.

### 2d. Boron-regression caveat (README §5e-1)

Across the 11 anchor cells the LRM residual was **positive in 10 of 11 with a
+98 ppm bias** — extrapolating to a never-run pair, the regression **under-predicts
boron by ~100 ppm**, and its "pass" predictions must be read with +100 ppm added.
At this cell the residual was +56 (predicted 1349, measured 1405.2).

**Nothing in this campaign rests on that regression.** 1405.2 ppm is a MASTER
measurement of 16 cores, and the 1600 gate is applied to measured CBC on every
row. The caveat is pinned because any claim about how much boron margin survives
**at the F_r optimum** — a point nobody has measured — is an extrapolation. R4
exists to measure it instead of assuming it.

## 3. Registered readouts

| | |
|---|---|
| **R1** | Best F_r vs the in-cell f109 floor **2.0481**; separately and labelled, vs the pair's f125 core **1.8375**. |
| **R2** | **PRIMARY** — did any MASTER-verified core pass all four gates (F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30)? The anchors produced **zero** such core anywhere (JOINT Tier-1 = 0 / 178). |
| **R3** | The F_q axis (§2c). Is the cell F_r-closed or F_q-closed at f109? |
| **R4** | CBC under F_r pressure — the (F_r, CBC) Pareto front, not just the F_r minimum. The 195 ppm margin exists at the *sample*, not at the optimum. |
| **R5** | Predicted pin burnup of the best cores, and whether it falls with F_r or sits flat (§5, the crossover diagnostic). |

## 4. Success / NULL — fixed in advance

* **PRIMARY** — a MASTER-verified core passing all four gates at this cell.
* **SECONDARY** — F_r < 2.0481 (beat the cell's own floor) with CBC still ≤ 1600.
  Any such core is the first F_r improvement ever measured here.
* **STRETCH** — F_r ≤ 1.8375 at f109, matching the pair's f125 best at the harder
  feed.
* **PARTIAL** — F_r falls below 2.0481 but not below ~1.84, or falls while CBC
  crosses 1600. Publishable: it prices the F_r/CBC trade at the first cell where
  the trade is visible.
* **NULL** — the floor does not move meaningfully below 2.0481 in 60 calls.
  **Registered reading: "2-type lattice F_r floor"** — a two-fresh-type loading
  pattern at n_gd 24 / e 5.69 cannot flatten the radial peak far enough, and the
  lever is the **lattice**, not the LP. That reading **hands the problem to the
  3-type track** and is a result, not a failure. It is why the budget is 60.

## 5. NEW: the predicted pin-burnup gate — first campaign to carry it

`min_fr_max_cycle` gated nothing on pin burnup until today, while `min_fuel_cost`
/ `fr_boundary` / `flat_power` all gated it at 80 GWd/tU. Both closed min_fr
campaigns therefore reported cores breaching the LEU+ 80 limit as feasible — 100%
of f113's 41 and 100% of f109's 52, on a validated prediction
(`fpcamp_E1E2_f109_results_20260817.md` §7). The gate now lives in
`MinFrSpec.pin_bu_limit` (acquisition) and `feasibility_limits_for` (campaign),
with deck knob `minfr_pin_bu_limit`.

**78.0, not 80.0.** The 2.0 is model margin: the s1g pin head validated in-cell at
E1_E2/f109 with MAE 1.84 and bias **−1.39** (it *under*-predicts), so a core
predicted at 78.0 is expected at ~79.4 — still inside the licensing 80.

**Three things it is and is not:**

1. It screens the **prediction** (surrogate column 6, `max_pin_burnup`), not a
   measurement. `enable_pin_burnup` is `[design]`-scoped and `lpopt optimize`
   cannot set it, so this run will again harvest **zero measured pin BU**. A
   pin-BU-enabled re-verification of the top cores remains the only thing that
   settles it.
2. It does **not** change the MEASURED feasibility set. `is_feasible` passes a
   missing pin BU by design and every row here will have one missing. So R2
   ("first 4-constraint core") means exactly what it has always meant.
3. **It does not stay second-order, and the crossover is computed here rather
   than discovered later.** The penalty is a sum of squared normalized excesses;
   F_r's width is 0.01 and pin BU's is 1.0 GWd/tU, so the two terms are equal at

   > **F_r_crossover = 1.55 + 0.01 · (pin_pred − 78)**

   A local dry run of this exact deck (2026-08-17, StubEvaluator, s1g live)
   predicted pin BU **89.8** on a wave-0 exploit candidate (against
   max_assembly_burnup 76.7). At pin 89.8 the crossover is **F_r ≈ 1.67**:

   * above F_r 1.67 — F_r dominates (≈87% of the penalty at the cell's 2.05
     floor); the search minimises F_r;
   * below F_r 1.67 — **the pin term dominates** and the search is steered by
     burnup rather than by F_r.

   This is **intended behaviour, not a defect**: a core at F_r 1.55 with a
   predicted pin BU near 90 is undeliverable, and chasing it is the failure this
   gate exists to stop. But it must be read into the result. **If the campaign
   stalls in the 1.65–1.75 band, the report must distinguish "F_r floor" from
   "pin gate took over."** The diagnostic (R5): if predicted pin BU falls
   alongside F_r, the gate is shaping; if F_r stalls while pin sits near 90, the
   gate is steering. This cell's measured `max_assembly_burnup` p50 is 82.1 over
   16 cores, so **the pin gate is expected to bind on essentially every
   candidate.** Raising the knob to 80.0 moves the crossover only to F_r ≈ 1.65 —
   it is not an escape and is not being used as one.

## 6. Elite seeding — what actually resolves (checked, not assumed)

`_case_store_rows` filters the store by `case_pair` **only**. Therefore:

* **In pool** — this pair's own 32 FOM-carrying rows: 16 at f109 (n_fresh 27,
  native) and 16 at f125 (n_fresh 31, feed-morphed down 4 units). With
  `elite_top_k = 32` that fills every elite slot.
* **Not in pool** — the sibling high-Gd pairs (`..._P6253Z2G06N24`,
  `..._P6257Z1G06N24`, `P6656*`). "Cross-pair morphs" were in the brief; they do
  **not** resolve through the elite path, exactly as the E1_E2 deck found for the
  f113 donor cores. Recorded rather than papered over.
* **Every elite is infeasible.** `n_pass_both = 0` for this pair at both feeds
  (F_q 2.612 floor at f109; F_r 1.8375 floor at f125). `_store_elites` is
  feasibility-first and then backfills with the best infeasible converged rows —
  here it is *all* backfill. This is the M5-pilot hazard documented at
  `campaign.py:985-1000`, accepted knowingly because the alternative (an empty
  elite pool) is strictly worse. Wave-0 skill against these 32 parents is the
  honest baseline.

## 7. Asset resolution — level 3 (`pair_ecore`), proven for this exact cell

Verified locally against the same package 199 carries:

```
restart     data/design/package/bases/P0_P1/MAS_RST.APRQ_11_0705.02
provenance  pair_ecore:MAS_RST.APRQ_11_0705.02
template    data/design/synth_decks/P6253Z1G06N24_P6253Z2G10N24/MAS_INP_cy12.inp
e_core      5.6944      library_dims (40, 42)
```

All 16 f109 anchor rows carry exactly that provenance string. Spot-check wave 0
of `labels.jsonl`; every row must read it. The f125 rows resolved one rung better
(`pair_feed:...0659.89`) only because produce had promoted a level-1 cache entry
— `lpopt optimize` never calls `promote()`, so **this run resolves at level 3
again and that is expected, not a regression.**

**The P0_P1 seed hazard did not fire.** `anchors_meshv3_198.inp` warned that
`MAS_RST.APRQ_11_0705.02` predates the 33→37 type rebuild. README §5f records the
outcome: the P0_P1-seeded strata ran clean, no `NOT DEFINED IN LPD_B&C`, no cy1
bootstrap needed, 16/16 rows converged here.

**Expected early-wave non-convergence.** README §5f-3: the first stratum on a
never-run pair burned 16 non-finite chains for 24 rows (1.67 chains/row); the
next, after level-1 promotion, burned 6. This run reopens at level 3 with no
cache, so budget ~1.4–1.7 MASTER calls per usable label early. **60 calls ≈ 40–50
usable labels, not 60.** Pre-registered so a thin wave 0 is not read as a broken
launch.

## 8. Model caveat — s1g has never seen this cell

s1g's per-cell calibration is fitted on ga80 (`calibration_library_id: ga80`).
For this paramA cell at feed 109 / ebin 5.7 there is no per-cell entry and the
prediction runs uncalibrated. This pair's 47 rows entered the store on
2026-08-17, **after** s1g was trained, so wave-0 predictions are genuinely
out-of-distribution and the online wave gate (`gate_skill_*`) is doing real work
from wave 1. Judge wave-0 skill against the 32 elite rows.

## 9. Budget 60, not 100 — with the evidence

Both min_fr precedents stalled long before 100:

* N1_N2/f113 — 41 feasible / 100 calls, frontier stopped moving by wave ~3–5.
* E1_E2/f109 — 52 feasible / 100 calls; its results report §8.4 says outright
  that "budget was oversized on the F_r axis — the frontier stopped moving at
  wave 3. The extra calls bought 44 more feasible cores … but no better core."

60 = 7 × 8 + 4 reserve. **If the frontier is still moving at wave 7 that is
itself a registered finding** (this cell behaves unlike both precedents) and an
extension is a new decision with a new deck, not a silent top-up.

## 10. Fleet / hygiene

199 only; verified idle at arming (master 0, python 0, 54.7 GB free). **198 / 181
/ 238 untouched.** Canonical store read-only until the final merge. Fresh run dir.
Every file scp'd in whole (ship-don't-remote-edit). The
`[optimize][DEPRECATED]` banner for `min_fr_max_cycle` is expected
(`campaign.py:72`) and is not a failed launch. `lpopt report`'s "best verified
loading patterns" table ranks by cycle distance regardless of objective and will
not show the F_r winner — read `state.json → best` (known defect).

## 11. Paths

| artefact | path |
|---|---|
| deck | `fpcamp_minfr_HGD569_f109_199.inp` |
| launcher / bat / status | `launch_fpcamp_HGD569_f109_199.ps1`, `run_fpcamp_minfr_HGD569_f109_199.bat`, `status_fpcamp_HGD569_f109_199.ps1` |
| run dir (199) | `C:\Users\USER\lpopt_work\kit_frontier\runs\fpcamp_minfr_hgd569_f109` |
| campaign log (199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_hgd569_f109_out.log` |
| return code (199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_hgd569_f109_rc.txt` |
| anchor evidence | `data/reports/mesh_v3_20260817/README.md` §5e, `.../anchors_measured.csv` |
| pin-gate precedent | `data/reports/fpcamp_E1E2_f109_results_20260817.md` §7 |
