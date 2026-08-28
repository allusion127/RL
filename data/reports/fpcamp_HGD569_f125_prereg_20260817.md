# PRE-REGISTRATION — high-Gd boron-opened cell, F_r assault, PIVOT TO FEED 125

**`P6253Z1G06N24_P6253Z2G10N24` / feed 125 / paramA (n_gd 24, e_core 5.694) — box 199**

Written **2026-08-17, before launch**, immediately after the f109 sibling
campaign closed (STRETCH met, F_r 2.0481 → **1.6743**;
`data/reports/fpcamp_HGD569_f109_results_20260817.md`). Nothing below was
chosen after seeing an f125 result. The deck
(`fpcamp_minfr_HGD569_f125_199.inp`) is hashed and the launcher refuses to
start if the hash does not match.

| | |
|---|---|
| objective | `min_fr_max_cycle`, λ_Fr = 1000 — unchanged from f109 |
| gates | F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30 ∧ predicted pin BU ≤ 78 — unchanged |
| cyclen | **no band** — recorded, subordinate tie-break only |
| budget | **60** (7 waves × 8 + 4 reserve) — unchanged |
| model | s1g (8th champion), `library_id = paramA` — unchanged |
| seed | **5695** (fresh; f109 used 5694) |
| run dir | `runs/fpcamp_minfr_hgd569_f125` on 199 |
| the ONE substantive change from the f109 deck | `feed = 109 → 125` (`n_fresh` 27 → 31) |

---

## 1. Why f125, given f109 already ran

Same pair, same cell (n_gd 24, e_core 5.694). One lever changes — feed
109 → 125 — and two things move with it, only one of which the f109 run could
answer:

1. **F_r starting point.** The f125 anchor sample's own F_r floor is 1.8375,
   already lower than f109's anchor floor (2.0481) was. But f109's *dedicated
   campaign* then found **1.6743** — lower than f125's raw anchor floor. So
   the honest bar for f125 is not "beat 1.8375" (the easy, anchor-relative
   read); it is "beat 1.6743", the freshest MASTER-verified state of the art
   for this pair, set on the sibling feed five hours earlier the same day.
2. **Pin burnup — the real reason to spend a second 60-call budget.**
   `estimate_discharge_burnup` (`lpopt/design/bootstrap.py:99-112`) scales
   residence as `241/feed`: 109 → 2.211 cycles, 125 → 1.928 cycles, a **12.8%
   cut**. f109's winner carried predicted pin BU **81.13** — the closest
   either min_fr precedent (E1_E2/f109, N1_N2/f113) has come to the 78 gate.
   Scaling that same core's burnup by the residence ratio (0.872) lands at
   **~70.7 GWd/tU** — comfortably under 78. This is **not** a validated pin-BU
   prediction: peak pin burnup depends on local power peaking, which the
   search is simultaneously reshaping, not only on core-average residence.
   It is the physical rationale for spending this campaign, and R-PIN below
   exists to measure it rather than assume it.

**CBC is not free here the way it was at f109.** The f125 anchor CBC floor is
**1561.31 ppm — only 38.69 ppm under the 1600 gate** (f109's margin was 194.84
ppm). Minimising F_r under a much thinner boron cushion is a genuine,
registered risk this deck carries that f109 did not.

## 2. The marks — pinned

### 2a. In-cell truth, feed 125 (16 converged rows)

Queried directly from `data/store/records.parquet`
(`case_pair == "P6253Z1G06N24_P6253Z2G10N24" & feed == 125`), cross-checked
against `data/reports/mesh_v3_20260817/anchors_measured.csv` (`fr_min` /
`cbc_min` agree to the reported precision):

| quantity | min | p50 | max | vs gate |
|---|---|---|---|---|
| **CBC_max** | **1561.31** | 1670.85 | 1728.04 | **4 / 16 pass** 1600 |
| **F_r** | **1.8375** | 2.2252 | 3.1030 | 0 / 16 pass 1.55 |
| **F_q** | **2.2810** | 2.9918 | 4.0658 | **2 / 16 pass** 2.41 |
| \|AO\| | 0.0207 | 0.0285 | 0.0630 | 16 / 16 pass 0.30 |
| cyclen | 711.927 | 723.082 | 737.465 | (ungated) |
| e_core | 5.662805 | 5.701052 | 5.727485 | anchor readout e_core 5.6985 |
| max_assembly_burnup | 66.482 | 72.033 | 74.414 | (reported) |
| max_pin_burnup | — | — | — | **not measured on a single row** (same limitation as f109) |

`n_pass_cbc = 4`, `n_pass_both = 0` (`joint_tier = "none"`, same as f109's
own anchor set).

### 2b. Cross-feed marks (this pair, feed 109) — labelled, never conflated

| mark | value |
|---|---|
| f109 anchor floor (16 rows) | F_r 2.0481 |
| **f109 dedicated-campaign winner** | **F_r 1.6743**, CBC 1540.99, F_q 2.0982, predicted pin BU **81.13** |

### 2c. The distance to close — three framings, all registered

* **Anchor-relative (the easy bar):** 1.8375 → 1.55 is **−0.2875** in-cell.
* **SOTA-relative (the honest bar):** 1.8375 → 1.6743 is **−0.1632** just to
  *match* what f109 already achieved on the sibling feed; beating it needs
  more.
* **Precedent-based expectation:** f109's own campaign moved its floor by
  **−0.3738** (2.0481 → 1.6743) in 60 calls, on this exact pair, hours ago —
  the freshest and most directly relevant precedent available, more relevant
  than T6_T4/f121's −0.32 (which f109 itself was scored against). Applying
  −0.3738 to f125's own anchor floor (1.8375) lands at **1.4637 — inside the
  1.55 gate.** This is registered as the optimistic case, not the expected
  one: f125's search dynamics differ from f109's on two axes that could cut
  either way — a thinner CBC margin (§1) and an elite pool that is *not*
  warm-started from f109's own improved rows (§6).

## 3. Registered readouts

| | |
|---|---|
| **R1** | Best F_r vs the in-cell f125 anchor floor **1.8375** (like-for-like) and, separately and labelled cross-feed, vs f109's dedicated-campaign winner **1.6743** (the true state of the art for this pair, not just its anchor). |
| **R2** | **PRIMARY** — did any MASTER-verified core pass all four gates (F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30)? Unmet anywhere in the project to date (0/178 anchors, 0/40 converged f109 campaign rows). |
| **R3** | CBC under F_r pressure — **high stakes here**. Margin at the anchor floor is only 38.69 ppm (vs f109's 194.84). Report the full (F_r, CBC) Pareto front and flag explicitly if any F_r-driven candidate crosses 1600. |
| **R4** | The F_q axis. In-cell floor already passes the gate (2.2810 ≤ 2.41, 2/16 anchors pass) — unlike f109 (0/16). Expect non-binding; report whether that holds under search pressure. |
| **R-PIN** | **The headline readout for this deck.** Predicted pin BU (surrogate `pred_mean` column 6, joined via `waves/wave_NN/selection.json` by `record_id`, same method as f109 §6) of the best F_r cores. Does it fall *below* f109's winner (81.13)? Does it fall to **≤ 78** (this campaign's own gate) — the actual deliverable-grade bar? A core clearing F_r ≤ 1.55 **and** predicted pin ≤ 78 simultaneously would be the **first 5-constraint-passing core the whole high-enrichment programme has produced.** Even short of that: if predicted pin BU drops meaningfully below 81.13 at f125 without F_r reaching 1.55, that alone validates (or refutes) the residence-time argument in §1, independent of whether the licensing gate is reached. |

## 4. Success / NULL — fixed in advance

* **PRIMARY** — a MASTER-verified core passing all four gates at f125. No such
  core exists anywhere in the project to date.
* **SECONDARY** — F_r < 1.8375 (beat this cell's *own* f125 anchor floor)
  with CBC still ≤ 1600. First F_r improvement ever measured at f125.
* **STRETCH** — F_r ≤ 1.6743 — match or beat f109's dedicated-campaign
  winner, the true cross-feed state of the art, not just its anchor.
* **REACH** — F_r ≤ 1.55 (the licensing gate itself). Registered as
  plausible via the precedent-based projection (§2c, 1.4637), not assumed.
* **PARTIAL** — F_r falls below 1.8375 but not below ~1.6743, **or** CBC
  crosses 1600 while F_r falls (the thin-margin risk, R3). Either is
  publishable: it prices the F_r/CBC trade at a cell where that trade now has
  real stakes for the first time.
* **NULL** — the floor does not move meaningfully below 1.8375 in 60 calls.
  **Registered reading:** feed alone (without a lattice change) does not move
  this pair's F_r floor past what the f109 campaign already found — i.e.
  ~1.6743 is a *pair* property, not a feed-109-specific one — and the
  residence-time argument for pin BU stands or falls independently of
  whether F_r itself moves further.

## 5. The predicted pin-burnup gate — unchanged from f109

`minfr_pin_bu_limit = 78.0`, not 80.0 (2.0 model margin; s1g pin head in-cell
MAE 1.84, bias −1.39, under-predicting). Screens the **prediction** only
(surrogate target index 6); does not change the measured feasibility set
(`is_feasible` passes a missing pin BU by design — every row here will have
one missing, same as f109). Crossover formula unchanged:

> F_r_crossover = 1.55 + 0.01 · (pin_pred − 78)

f109 found the search stayed F_r-dominant throughout (85–95% of the penalty
at every Pareto point; winner's own crossover pin=81.13 → F_r≈1.58, well
below its achieved 1.6743). **If the residence-time argument in §1 is right,
f125's predicted pin values should sit lower than f109's at the same F_r**,
which — via the same crossover formula — pushes the crossover point down too
(a lower `pin_pred` makes `term_pin` smaller at any given F_r, so F_r stays
dominant even more comfortably than at f109). R-PIN is where this is checked
against data instead of assumed.

## 6. Elite seeding — weaker than it could be, disclosed rather than fixed

`_case_store_rows` filters by `case_pair` **only**, so f109's 56 newly
converged rows (including the 1.6743 winner) *would* enrich this campaign's
elite pool if the store on 199 carried them. **It does not.** Verified before
writing this deck: 199's `data/store/records.parquet` is 22,131,144 bytes —
the same pre-f109-merge, 74,357-row copy shipped for the f109 launch, not the
local 74,417-row post-merge canonical. Per the task's own scoping ("edits
ONLY: feed 109→125…"), the store is **not** being reshipped for this run:
elite seeding here uses exactly the same 32-row, all-infeasible backfill pool
f109 used (16 f109-native + 16 f125 feed-morphed rows, `elite_top_k = 32`,
`n_pass_both = 0` at both feeds). This is a genuine, disclosed limitation —
this campaign does **not** get to warm-start from its own sibling's improved
result. If f125 still beats 1.8375 by a wide margin under this weaker seed,
that is a stronger result than f109's own on that axis (f109's elites were
also all infeasible, so the comparison is fair on that count at least).

## 7. Asset resolution — level 3 (`pair_ecore`), confirmed for f125 by direct probe

Checked directly on 199 before writing this deck (not assumed):

```
Test-Path bases\P0_P1\MAS_RST.APRQ_11_0705.02   -> True
Get-ChildItem package\cores        (this pair)  -> no entry
Get-ChildItem package\synth_decks  (this pair)  -> no entry
```

This run resolves at level 3 off `bases/P0_P1`, exactly as f109 did
(`pair_ecore:MAS_RST.APRQ_11_0705.02`) — the f125 *anchor* rows resolved one
rung better only because `produce` (not `optimize`) had promoted a cache
entry in a different package copy at anchor time; `lpopt optimize` never
calls `promote()`, so level 3 is expected here too, not a regression. Budget
~1.4–1.7 MASTER calls per usable label early, per f109's own observed rate
(1.500, same package/pair state).

## 8. Model caveat — s1g has never seen this cell

s1g's per-cell calibration is fitted on ga80; this paramA cell has no
per-cell entry at either feed. Wave-0 predictions are out-of-distribution;
the online wave gate does real work from wave 1 — same caveat as f109, whose
finetune gate rejected all 8 waves (improvement came from the base champion
plus BO refinement, not in-campaign updates). Watch whether f125 repeats
that pattern.

## 9. Budget 60 — same as f109, with one new caveat

f109's own frontier was **still descending at wave 7** (best core found on
call 57/60, the last reserve call) — unlike both prior min_fr precedents
(N1_N2/f113, E1_E2/f109), whose frontiers had stalled by wave 3–5. That means
the "60 is enough" assumption this deck inherits from f109's own
pre-registration is **weaker here than it was for f109 itself**: if f125 also
fails to stall by wave 7, that is now the *second* cell to do so, and an
extension is — as before — a new decision with a new deck, not a silent
top-up. 60 = 7 × 8 + 4 reserve, unchanged.

## 10. Fleet / hygiene

199 only; verified idle at arming (master 0, python 0, 54.1 GB free). 198 /
181 / 238 untouched. Canonical store read-only, and — per §6 — deliberately
*not* refreshed for this run. Fresh run dir. Every file scp'd in whole
(ship-don't-remote-edit). The `[optimize][DEPRECATED]` banner for
`min_fr_max_cycle` is expected (`campaign.py:72`). `lpopt report`'s
best-patterns table does not show the F_r winner (ranks by cycle distance,
known defect, unchanged) — read `state.json → best`.

## 11. Paths

| artefact | path |
|---|---|
| deck | `fpcamp_minfr_HGD569_f125_199.inp` |
| launcher / bat / status | `launch_fpcamp_HGD569_f125_199.ps1`, `run_fpcamp_minfr_HGD569_f125_199.bat`, `status_fpcamp_HGD569_f125_199.ps1` |
| run dir (199) | `C:\Users\USER\lpopt_work\kit_frontier\runs\fpcamp_minfr_hgd569_f125` |
| campaign log (199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_hgd569_f125_out.log` |
| return code (199) | `C:\Users\USER\lpopt_work\kit_frontier\fpcamp_minfr_hgd569_f125_rc.txt` |
| f109 sibling (closed) | `data/reports/fpcamp_HGD569_f109_prereg_20260817.md`, `data/reports/fpcamp_HGD569_f109_results_20260817.md` |
| anchor evidence | `data/reports/mesh_v3_20260817/README.md` §5e, `.../anchors_measured.csv` |
| pin-gate precedent | `data/reports/fpcamp_E1E2_f109_results_20260817.md` §7 |
