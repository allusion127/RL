# v4 A/B + validate_gate re-run for cell 4 (`5-5.25_f117`)

Date: 2026-07-18. Evidence-first evaluation of whether the cond_v4 model
`data/models/20260718_114204` (43 cell channels, 13 globals, 7 targets,
calibration.json fitted) fixes the enrichment-band interference that made cell 4
fail its gate under the v3-retrain `data/models/20260718_100439`.

Models under test (all loaded via `PosValCnnBackend.from_dir`, store `data/store`,
`ga80`, cpu):
- **v3 champion** `20260718_075428` — cond_schema **v3**, **26** cell channels, 10 globals. (= cell 4 `champion_before`; the anchor-band champion.)
- **v3 retrain (rejected)** `20260718_100439` — cond_schema v3, 26ch. (Failed cell-4 gate earlier.)
- **v4** `20260718_114204` — cond_schema **v4**, **43** cell channels, 13 globals.

Mirroring: all gate numbers were produced by calling the real
`CurriculumDriver._gate_newcell` / `_gate_no_regression` (from `lpopt/curriculum.py`)
on the live `data/curriculum/state.json`. **Faithfulness check:** re-running those
same methods with the rejected `100439` reproduced the recorded state.json values
to full precision (new_cell mean_spearman `0.44922939531462863`; no_regression
worst_drop `0.6467771900973378`; cyclen 0.976088→0.329311). So the v4 numbers
below are the exact numbers the driver would compute.

---

## VERDICT

**v4 does NOT fix the band interference to a gate-passing level. Prediction gates
do NOT both pass → mini campaign NOT run → NO champion swap → state.json UNCHANGED.**

- `new_cell` (v4): **PASS** (mean within-case Spearman 0.501 ≥ threshold 0.0 — but the threshold is trivially 0.0, so this is not evidence of quality).
- `no_regression` (v4 vs v3 champion 075428, ε=0.05): **FAIL** — worst_drop **0.382**, and **all 6 checks exceed ε**.
- Blind A/B: v4 is **worse than or ~equal to the v3 champion on 3 of 4 cells**, including the new cell 4. v4 only marginally wins cell 3.

Nuance (the one positive): v4 **substantially repaired the cyclen collapse**
(the most dramatic v3-retrain failure): cyclen no-regression drops fell from
0.647/0.579/0.561 (v3-retrain) to 0.068/0.161/0.115 (v4). But (a) those cyclen
drops still exceed ε=0.05, and (b) the **f_r regression is a separate, pre-existing
band-interference effect that v4 does NOT fix** (drops ~0.18–0.38, essentially
identical to the v3-retrain). So v4 is a *partial* fix on cyclen only, insufficient
to pass, and buys no net ranking skill over the v3 champion.

---

## 1. Blind A/B on cells 1–4 (v3 champion 075428 vs v4 114204)

Identical stored blind-probe patterns (unpacked from each cell's
`blind_probe.json` `candidates`), scored fresh with each model; per-target MAE and
within-case rank Spearman vs the stored live-MASTER actuals (mirrors
`_transfer_stats` / `_gate_newcell`). `discharge_burnup` has **no live actual**
(the WaveVerifier FOM does not expose it — probe-predicted only), so it is
un-scorable here (n=0) and omitted from the table, exactly as the driver does.

Spearman (higher better) — **bold** = better model:

| cell | target | v3 Sp | v4 Sp |
|---|---|---|---|
| 5.25-5.5_f117 (c1, n=11) | f_r | **0.409** | -0.255 |
| | f_q | **0.500** | -0.109 |
| | cbc_max | 0.745 | **0.791** |
| | cyclen | 0.709 | **0.864** |
| | ao_abs | 0.918 | **0.964** |
| | max_pin_burnup | **0.427** | 0.400 |
| 5.25-5.5_f109 (c2, n=11) | f_r | **0.791** | 0.464 |
| | f_q | **0.900** | 0.564 |
| | cbc_max | **0.909** | 0.891 |
| | cyclen | **0.855** | 0.800 |
| | ao_abs | 0.691 | **0.982** |
| | max_pin_burnup | -0.209 | **0.191** |
| 5.25-5.5_f125 (c3, n=11) | f_r | 0.527 | **0.545** |
| | f_q | 0.573 | **0.782** |
| | cbc_max | **0.873** | 0.800 |
| | cyclen | **0.873** | 0.845 |
| | ao_abs | 0.927 | **0.945** |
| | max_pin_burnup | 0.727 | **0.736** |
| 5-5.25_f117 (c4/new, n=13) | f_r | **0.841** | 0.264 |
| | f_q | **0.813** | 0.242 |
| | cbc_max | 0.220 | **0.845** |
| | cyclen | **0.912** | 0.731 |
| | ao_abs | 0.802 | **0.923** |
| | max_pin_burnup | **0.330** | 0.000 |

MAE (lower better):

| cell | target | v3 MAE | v4 MAE |
|---|---|---|---|
| c1 | f_r | 0.300 | **0.225** |
| | f_q | 0.416 | **0.322** |
| | cbc_max | **32.32** | 52.44 |
| | cyclen | **8.83** | 15.01 |
| | ao_abs | **0.0110** | 0.0127 |
| | max_pin_burnup | **5.85** | 16.52 |
| c2 | f_r | 0.403 | **0.356** |
| | f_q | 0.645 | **0.581** |
| | cbc_max | **30.69** | 137.59 |
| | cyclen | **7.38** | 53.39 |
| | ao_abs | **0.0079** | 0.0088 |
| | max_pin_burnup | **2.86** | 17.60 |
| c3 | f_r | 0.351 | **0.222** |
| | f_q | 0.465 | **0.280** |
| | cbc_max | **24.30** | 26.67 |
| | cyclen | **5.66** | 9.38 |
| | ao_abs | 0.0070 | **0.0046** |
| | max_pin_burnup | **1.70** | 5.39 |
| c4 | f_r | 0.499 | **0.281** |
| | f_q | 0.627 | **0.406** |
| | cbc_max | 80.63 | **37.22** |
| | cyclen | **5.88** | 14.70 |
| | ao_abs | **0.0081** | 0.0086 |
| | max_pin_burnup | **4.13** | 11.62 |

Per-cell mean within-case Spearman over the 6 FOM targets:

| cell | v3 | v4 | winner |
|---|---|---|---|
| 5.25-5.5_f117 (c1) | **0.618** | 0.442 | v3 |
| 5.25-5.5_f109 (c2) | **0.656** | 0.648 | v3 (≈tie) |
| 5.25-5.5_f125 (c3) | 0.750 | **0.776** | v4 |
| 5-5.25_f117 (c4/new) | **0.653** | 0.501 | v3 |

Reading: v4 tends to **lower F_r/F_q ranking skill** on the anchor band and blow up
**cbc_max / cyclen / max_pin_burnup absolute error** (c2 cyclen MAE 7.4→53.4;
c2 cbc_max 30.7→137.6; max_pin_burnup MAE roughly triples/quadruples everywhere).
It gains on ao_abs and on cbc_max **ranking** for c4. Because v4 was trained on the
**same rows** (no new data), the wider 43-channel feature space appears to be
underfit / noise-adding on the anchor band rather than helpful. The v4 lattice
k-inf / branch-coefficient / XS-ADF-FF features did not deliver a net ranking gain.

---

## 2. Gate math under v4 (mirrors `_phase_validate_gate`)

### (a) new_cell — cell 4 `5-5.25_f117`, model v4 (`_gate_newcell`)
Blind-probe chains reused as holdout (13 converged), no extra MASTER calls.

| target | n | MAE | Spearman |
|---|---|---|---|
| f_r | 13 | 0.2806 | 0.2637 |
| cbc_max | 13 | 37.225 | 0.8446 |
| f_q | 13 | 0.4056 | 0.2418 |
| cyclen | 13 | 14.697 | 0.7308 |
| ao_abs | 13 | 0.00856 | 0.9231 |
| max_pin_burnup | 13 | 11.619 | 0.0000 |

mean_spearman = **0.50065**, threshold `gate_new_cell_min_spearman` = **0.0** →
**PASS** (but the bar is 0.0; note the max_pin_burnup Spearman is 0.0 and f_r/f_q
are ~0.24 — poor). For reference the rejected v3-retrain 100439 scored mean 0.449;
so on the new cell itself v4 (0.501) is a hair better, but the v3 *champion*
(075428) blind A/B on cell 4 was 0.653 — better than both retrains.

### (b) no_regression — cells 1–3 stored holdout replay (`_gate_no_regression`)
old = v3 champion `20260718_075428`; new = v4 `20260718_114204`; ε = **0.05**;
holdout = each done cell's stored P/converged rows, `head(200)` (=150 each), within-case
Spearman on cyclen & f_r. **3-way** vs the rejected v3-retrain 100439 (recorded):

| cell | target | old (075428) | v3-retrain 100439 (new_sp / drop) | **v4 114204 (new_sp / drop)** | v4 pass? |
|---|---|---|---|---|---|
| 5.25-5.5_f117 | cyclen | 0.9761 | 0.3293 / 0.6468 | **0.9080 / 0.0681** | FAIL |
| 5.25-5.5_f117 | f_r | 0.9387 | 0.7010 / 0.2377 | **0.7102 / 0.2285** | FAIL |
| 5.25-5.5_f109 | cyclen | 0.9739 | 0.3951 / 0.5788 | **0.8129 / 0.1609** | FAIL |
| 5.25-5.5_f109 | f_r | 0.9539 | 0.5614 / 0.3926 | **0.5718 / 0.3821** | FAIL |
| 5.25-5.5_f125 | cyclen | 0.9475 | 0.3861 / 0.5614 | **0.8326 / 0.1149** | FAIL |
| 5.25-5.5_f125 | f_r | 0.9725 | 0.7728 / 0.1997 | **0.7887 / 0.1839** | FAIL |

**no_regression: FAIL** — worst_drop **0.3821** (f_r, cell 2) ≫ ε 0.05; every one of
the 6 checks exceeds ε.

Interpretation of the 3-way:
- **cyclen** — v4 is a large improvement over the v3-retrain (drops 0.068/0.161/0.115
  vs 0.647/0.579/0.561). This is the specific "enrichment-band interference" the
  hypothesis targeted, and v4's per-assembly lattice features **did materially
  dampen it**. But the drops still exceed 0.05, so it does not clear the gate.
- **f_r** — essentially unchanged between v3-retrain and v4 (drops ~0.18–0.38 in
  both). This regression is band-interference that v4 does **not** fix; it is the
  binding failure (worst_drop comes from f_r).

### (c) transfer_curve — not modified (no gate pass; no state write performed).

---

## 3. Mini user_criteria campaign (budget 16)

**NOT RUN.** Per the task and the driver's ordering, the live-MASTER mini campaign
runs only after the prediction gates pass. `no_regression` FAILED, so no MASTER
compute was spent. (For context: for cell 4 this campaign is also structurally
near-impossible to pass — `gate_cyclen_target`=613.84 = median of the converged
probe cyclen ⇒ `blind_best_distance`=0.0 ⇒ "progress" requires a verified chain
within 1e-9 of target; and `feasible` requires f_r ≤ `gate_min_f_r`=1.55, whereas
every A2_A8/E1_E2/E3_E4/J1_J2 chain in this band yields f_r ≈ 1.8–3.6. All of
cells 1–3 also had `feasible_or_progress=false` yet passed their gates because the
driver's real gate is `new_cell ∧ no_regression`; the mini campaign is advisory.)

---

## 4. State changes

**NONE.** `data/curriculum/state.json` was not modified. `champion_model_dir`
remains `data/models/20260718_100439` (unchanged); cell `5-5.25_f117` stays at
phase `validate_gate` (unchanged). No `gate_v4.json` written to the cell dir. The
only file written is this report and its raw sidecar
`data/reports/_v4_ab_gate_raw.json`. All model loads/predicts were read-only.

---

## 5. Bottom line on the hypothesis

> "v4's per-assembly lattice-result features fix the enrichment-band interference."

**Rejected as a gate-passing fix; partially true mechanistically.** v4 clearly
attacked the right failure — it cut the cyclen no-regression collapse by ~4–8× —
but (1) even the repaired cyclen drops still exceed ε=0.05, (2) it leaves the f_r
band-interference regression untouched (worst_drop 0.382), and (3) on a clean blind
A/B over identical patterns v4 is not better than the incumbent v3 champion 075428
(loses c1, c2, c4; wins only c3) and inflates cbc_max/cyclen/max_pin_burnup MAE on
the anchor band. Because v4 was trained on the **same store rows** as the v3 models,
the wider feature schema alone — without new cell-4-adjacent data — does not buy
enough. Recommendation: keep 075428 as the operative champion; do not promote v4;
the path forward is more data in/near the 5–5.25 band (and/or fixing the f_r
band-transfer regression), not the feature expansion by itself.
