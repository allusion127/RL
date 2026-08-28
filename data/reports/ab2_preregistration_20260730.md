# A/B round 2 — VARIANCE arms: pre-registration

**Written 2026-07-30, BEFORE any arm has been trained.** Every threshold, metric
and decision below is fixed as of this document. Anything added after the arms
return is a post-hoc analysis and must be labelled as one.

---

## 1. Why this round exists

The flat_power campaign (verified-optimal candidates) leaves these gaps against
the acceptance bar:

| axis | measured | bar | gap |
|---|---:|---:|---|
| CBC \|err\| (median) | 15.7 ppm | ≤ 1 ppm | **16x** |
| node_peak | 0.021 | ~1e-3 | **20x** |
| F_q | 0.055 | — | open |
| map_cov | 0.006 | — | open |
| cyclen | 0.74 EFPD | — | **passes** |

Systematic bias on all of these is already removed: six per-cell affine
calibrations (`cyclen`, `f_r`, `cbc_max`, `f_q`, `ao_abs`, `flatness`
{node_peak, map_cov}) are fitted into every model dir. What is left is
**VARIANCE**, which a calibration cannot touch — only the model can. Round 2
therefore tests model changes, one per arm, as a pre-registered A/B.

**Falsification, registered now:** if no arm improves its target axis with a CI
excluding 0, the conclusion is that *loss/label engineering is not the lever for
this residual* and the next round must move to a different mechanism (more
labels in the deficient cells, verify-many-then-select, or an ensemble/variance
reduction change) — **not** to a bigger version of the same knob.

---

## 2. Frozen inputs

### 2.1 Store snapshot

| item | value |
|---|---|
| `data/store/records.parquet` sha256 | `f01526ccbe2e8e5d527c744553da6c83e6262fff8084e7327a4a09aa188b6434` |
| `records.parquet` bytes | 20,013,908 |
| rows | **67,979** (converged 60,487) |
| `data/store/maps.npz` sha256 | `20c3e2a5198646b32ac3ef1e392ab22e91e4ac5c3583800d4f676c92683a5330` |
| `maps.npz` bytes | 157,072,899 |
| maps.npz keys | 60,721 = 39,549 legacy 4-plane + 10,586 `__traj` + 10,586 `__axial` |
| rows with a map | 39,571 |
| rows with a `__traj` | **10,586** (100% joinable to `records`, 100% converged) |
| cbc-labelled converged rows (`cbc_kind == "max"`) | 49,228 |

CBC label-convention provenance over those 49,228 rows
(`lpopt.model.dataset_torch.cbc_provenance_codes`):

| group | rows | share |
|---|---:|---:|
| `master_native` (Dataset P; **the serve convention**) | 21,062 | 42.8% |
| `mocha_native` (Dataset A) | 27,592 | **56.0%** |
| `ga_native` (Dataset B) | 574 | 1.2% |

### 2.2 Split — the SAME file for all four arms

| item | value |
|---|---|
| file | `data/splits/S1.json` |
| sha256 | `02d095d88bc0d851de16a36c11dae050e36445b7098b32e7bb360851d2898b99` |
| bytes | 5,216,379 |
| kind / seed | `curriculum_group` / 0 |
| train / val | **56,578 / 11,401** (id overlap **0**) |
| holdout cells (`groups.curriculum_val_by_cell`) | **36** cells, 3,207 rows; min 60 / median 90 rows per cell |
| curriculum cells / cap | 36 / 16.0 |
| held-out groups | 8 of 44 |
| `__traj` rows in train / val | 8,888 / 1,698 |
| cbc provenance in train | master 25,096 · mocha 30,908 · ga 574 |
| cbc provenance in val | master 3,455 · mocha 7,946 · **ga 0** |

**Registered consequence:** the holdout carries **no `ga_native` rows**, so arm
A2's `ga_native` offset is trainable but not measurable on the holdout. A2's
verdict rests on the `mocha_native` offset only; the `ga_native` value is
reported as provenance, never as evidence.

### 2.3 Champion (the incumbent this round must eventually beat)

`data/models/20260729_054749` — 5 members, seeds **20260716..20260720**,
`cond_schema=v6`, `width=224`, `n_blocks=8`, `head_hidden=384`, 10,351,619
trainable params/member. All four arms reuse **exactly these seeds**.

### 2.4 Trajectory-label coverage — disclosed limitation

All 10,586 `__traj` records are **Dataset P**. They span `e_core` 5.00–6.36 and
all eleven feed values (101…141), i.e. every one of the 141 Dataset-P
`(feed, e_core-bin)` map cells is covered. But the 12 **Dataset A** map cells —
`feed=121`, `e_core` 5.35–5.60 — have **zero** trajectory coverage, and those
cells hold 27,816 of the 39,568 mapped converged rows (**70.3%** of the mapped
corpus).

So A1's supervision is structurally confined to Dataset P. This is a large
improvement on the 894-record state of `flatness_first_program_20260725.md` §3
(which had zero coverage in the mega-cells *and* only DoE `fill_*` provenance),
but it is not uniform coverage, and the decision rule below does not pretend it
is: A1's target-axis gain is additionally reported **restricted to P cells**, and
a gain that appears only in the pooled statistic while being absent in P cells is
recorded as unexplained rather than as a win.

---

## 3. The four arms

Each arm = the champion recipe + **exactly one** change. A0 is not optional: it
is the control the protocol requires (`lpopt/model/flat_ab.py`
`ControlMissingError`), because `arm − champion` confounds the model change with
the store change (67,979 rows now vs the champion's snapshot), and the store is
the more probable source.

| arm | change | target axis |
|---|---|---|
| **A0** | none — champion recipe re-trained on this store | (control) |
| **A1** | `--traj-weight 0.3` | cyclen + cbc_max MAE |
| **A2** | `--cbc-provenance-offset` | cbc_max MAE |
| **A3** | `--map-peak-topk-weight 2.0` | node_peak + f_q MAE |

### 3.1 Exact train commands

All four are `lpopt.remote` invocations; `--no-install` because the source is
pushed once before A0 and is byte-identical for all four arms.

**A0 — control**

```
python -m lpopt.remote --input lpopt.inp --no-install train -- \
  --ensemble 5 --split S1 --cond-schema v6 --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

**A1 — trajectory supervision** (A0 + one flag)

```
python -m lpopt.remote --input lpopt.inp --no-install train -- \
  --ensemble 5 --split S1 --cond-schema v6 --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
  --traj-weight 0.3
```

**A2 — provenance-conditioned CBC** (A0 + one flag)

```
python -m lpopt.remote --input lpopt.inp --no-install train -- \
  --ensemble 5 --split S1 --cond-schema v6 --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
  --cbc-provenance-offset
```

**A3 — map-loss peak focus** (A0 + one flag)

```
python -m lpopt.remote --input lpopt.inp --no-install train -- \
  --ensemble 5 --split S1 --cond-schema v6 --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
  --map-peak-topk-weight 2.0
```

`--map-peak-topk` defaults to 5 (of 69 slots) and is left at its default in A3;
`--traj-anchors` defaults to `0,0.25,0.5,0.75,1` and is left at its default in A1.

### 3.2 Pre-launch verification (run before shipping)

1. `git diff` (or the pushed tarball) is identical for all four launches.
2. A0's `member_*/meta.json` has `train_config.traj_weight == 0.0`,
   `cbc_provenance_offset == false`, `map_peak_topk_weight == 0.0`, and
   `net_config.n_traj_anchors == n_traj_planes == n_cbc_provenance_groups == 0`;
   `model.pt` contains no key matching `traj` or `cbc`.
3. Each arm's `train.log` reports **all six** per-cell calibrations fitted (see
   §7 — this was previously not the case) and contains no
   `PER-CELL CALIBRATION(S) MISSING` banner.
4. Each arm's `meta.json` `target_zscore` is identical to A0's (the arms change
   the loss, not the normalisation).

---

## 4. What each arm changes (files : lines)

### A1 — trajectory supervision

| file : lines | what |
|---|---|
| `lpopt/data/traj.py` (**new**, 256 lines) | the `__traj` label contract: `load_traj` (117), `slot_mean_burnup` (140), `cycle_burnup_fraction` (152), `anchor_indices` (172), `anchor_planes` (186), `stack_anchor_traj` (217); `DEFAULT_ANCHORS = (0.0, 0.25, 0.5, 0.75, 1.0)` (85), `MAX_ANCHOR_FRAC_ERROR = 0.10` (96) |
| `lpopt/model/dataset_torch.py` : 129–130, 143–144, 228–255, 291–301 | `include_traj` / `traj_anchors`, `LPDataset._traj`, three additive item keys |
| `lpopt/model/net.py` : 66, 193–208, 353–371, 387–410, 411–438, 440–441, 486–489 | `TRAJ_MAP_CHANNELS = (1,2,3)`, config `n_traj_anchors` / `n_traj_planes`, `traj_film`, the `_map_quarter` extraction, `_traj_quarter`, the optional third `forward` argument, the `out["traj"]` emission |
| `lpopt/model/train.py` : 51–53, 246–267, 444–460, 781–795, 915–923, 962–971, 1487–1491, 1508–1517, 1553–1557, 1783–1797, 1824–1825, 1870–1872, 2044–2056, 2272–2273 / 2281–2282 / 2417–2418 / 2426–2427; CLI 2791–2799, 2871–2877 | `TrainConfig.traj_weight` / `traj_anchors`, `traj_loss`, `_Norm.z_traj`, precompute + batch-gather plumbing, the loss branch, head sizing from the ATTACHED labels, `meta["traj_head"]`, `--traj-weight` / `--traj-anchors` (+ `_parse_traj_anchors` at 578) |

**Mechanism.** The `__traj` label is the full EDIT5 burnup ladder,
`float16[n_steps, 3, 9, 9]`, planes `(power, burnup, kinf)`. Its **endpoints are
bit-identical to the legacy 4-plane stack** — verified on the store, 0/150
mismatches:

```
maps[0] boc_power  == traj[ 0, 0]        maps[2] eoc_burnup == traj[-1, 1]
maps[1] eoc_power  == traj[-1, 0]        maps[3] eoc_kinf   == traj[-1, 2]
```

That identity is what makes this a *minimal* use: the arm re-runs the **existing**
map decoder on a trunk feature FiLM-modulated by `[globals, cycle-burnup
fraction]` and reads channels `(1,2,3)`, so at fraction 1 the readout's labels ARE
the EOC map labels, the map head's own z-score constants are the right ones, and
the only new parameters are one FiLM (**+104,160** of 10.35M, +1.0%). No new
decoder, no new normalisation, no new serve path.

The burnup coordinate is `(b̄_t − b̄_0)/(b̄_T − b̄_0)` on the 69-slot mean of the
cumulative-burnup plane: 0 at BOC, 1 at EOC, deck-family independent, and needing
no EFPD (the purge deletes it — the constraint that forced
`lpopt/data/axial.py`'s two-anchor design). Measured step spacing in fraction:
median 0.043, p95 0.053, max 0.057 — comfortably inside the 0.10 anchor
tolerance, so every anchor is honestly supported on every sampled record (0/150
partially masked). The model is conditioned on the **achieved** fraction, never
the requested one.

**Why this targets cyclen/CBC.** `cyclen` is the endpoint of the boron let-down
trajectory. A model supervised only at BOC and EOC sees the two ends of that
curve and must infer the path; the residual variance lives in the path.

### A2 — provenance-conditioned CBC

| file : lines | what |
|---|---|
| `lpopt/model/dataset_torch.py` : 456–465, 467–493, 495–500 | `CBC_PROVENANCE_GROUPS = ("master_native", "mocha_native", "ga_native")` (index 0 = reference), `cbc_provenance_labels`, `cbc_provenance_codes` |
| `lpopt/model/net.py` : 209–223, 372–378 | config `n_cbc_provenance_groups`, the `n−1`-wide `cbc_provenance_offset` parameter, **never read by `forward`** |
| `lpopt/model/train.py` : 74 (`_CBC_IDX`), 268–280, 972–975 (`cbc_prov` tensor), 1491, 1524–1544 (the loss-only shift), 1799–1808, 1826, 1873–1875, 2057–2071 (`meta["cbc_provenance_offset"]`); CLI 2800–2805, 2878 | `TrainConfig.cbc_provenance_offset`, the loss-only shift in Z units, the meta stamp, `--cbc-provenance-offset` |

**Mechanism.** The model's (MASTER-native) `cbc_max` prediction is mapped into
the row's own label convention before the residual is taken, inside the `cbc`
column of the regression loss and nowhere else — the rank hinges, the
distillation pull, the quantile pinball and every map term still see the
unshifted `mu`. Two design decisions are load-bearing:

* **The reference group has no parameter at all.** The tensor is `n−1` wide and
  index 0 is a `cat`-ed structural zero. Serving cannot drift off-convention,
  because there is nothing to drift.
* **The offset is learned in Z units, not ppm.** Adam's per-step displacement is
  bounded by ~`lr`; at the scaled `lr` of 1.2e-3 and ~7,350 steps a ppm-scale
  parameter could move at most ~9 ppm over the entire run, so a ppm
  parameterisation would have made this arm a silent no-op. The gap is 100–410
  ppm = 0.26–1.06 z at `tstd(cbc) = 386.32` ppm. `meta.json` records both units.

**Serve contract.** `forward` never reads the parameter (asserted by test:
setting the offset to 37.0 leaves `mu` bit-identical). Nothing in `model_api`,
the calibrations, or the gate changes.

### A3 — map-loss peak focus

| file : lines | what |
|---|---|
| `lpopt/model/train.py` : 111–128 (config), 386–411 (`top_k_slot_weight`), 414–442 (`map_loss`), 1545–1549 (the call); CLI 2806–2813, 2879–2882 | `map_peak_topk` (5), `map_peak_topk_weight` (0.0), `--map-peak-topk-weight`, `--map-peak-topk` |

**Mechanism.** `map_peak_weight = 2.0` (already in the champion) is a
*continuous* re-weighting by `relu(map_z)`: every above-average node gets extra
weight in proportion to its heat, so most of the extra gradient lands on the
broad warm region — the ~30 above-average nodes of a plane collectively outweigh
its single hottest node by an order of magnitude. `map_peak_topk_weight` is the
*rank-based* complement: a further `1 + w` on the K hottest slots **of each
plane, selected by the LABEL**. `F_q` and `node_peak` are order statistics of
exactly those nodes.

Selecting on the label (not the prediction) is deliberate: peak *location* is the
noise-free part of a map label (22/22 exact reproduction under the diagonal
transpose check, `transpose_noise_measured_20260725.md`), while its *value*
carries the float16 harvest quantum. A label-selected set does not move with the
model, which is what keeps this a supervision knob rather than a self-confirming
one. Masked and NaN slots can never be selected, and K is clamped to the number
of valid slots. The two knobs multiply. **+0 parameters.**

---

## 5. Decision rule — fixed before any result exists

### 5.1 Surface

* **Unit of comparison:** the **cell**, not the row. Rows inside one design cell
  share a design, a library and a generator, so a row bootstrap understates the
  interval.
* **Rows:** the S1 holdout (§2.2), restricted to the frozen cell list — 36 cells.
  Every arm is scored on the **same rows in the same order**; that alignment is
  the pairing (`flat_ab.FlatArena.__post_init__` enforces it).
* **Estimator:** `lpopt/model/ab_paired.py` `paired_cell_bootstrap` —
  `method="bca"`, `aggregate="median"`, `reps=2000`, `alpha=0.05`, `seed=0`,
  sign-flipped so `theta > 0` always means "arm beats A0".
  * `n_cells < 6` → BCa degrades to percentile, announced in `notes`.
  * `n_cells < 3` → `method="insufficient"`; no interval, fails every gain AND
    every harm test.
  * a point-mass resample → `method="degenerate"`; likewise carries **no**
    evidence. 36 cells is comfortably above both floors.
* **Control:** A0. Never the champion. A comparison against the champion
  confounds the model change with the store change.

### 5.2 Condition 1 — the arm must ESTABLISH its target gain

For each arm, every axis in `flat_metrics.AB2_ARM_TARGET_AXES` must have
`ci_lo > 0` (`PairedDiff.establishes_gain`). A point estimate is never enough.

| arm | axes that must ALL improve |
|---|---|
| A1 | `T_cell_mae_cyclen`, `T_cell_mae_cbc_max` |
| A2 | `T_cell_mae_cbc_max` |
| A3 | `T_cell_mae_node_peak`, `T_cell_mae_f_q` |

These are `flat_metrics.cell_mae` (per-cell mean absolute error, `min_rows`
default) with `higher_is_better=False`, registered in
`flat_metrics.AB2_TARGET_METRICS`. They are deliberately a **separate** tuple
from `PRE_REGISTERED_METRICS`: that registry is the flatness program's frozen
scoring surface and every arm ever judged on it was judged on exactly those 18
keys. Appending to it would retroactively change that surface.

`T_abs_bias_cbc_max` is reported alongside A2 and **decides nothing** — the six
per-cell calibrations already remove bias, so a bias move is a diagnostic, not
the effect under test.

### 5.3 Condition 2 — the arm must not REGRESS any enforced gate axis

Every metric in `flat_ab.HARM_MARGINS` must satisfy
`PairedDiff.bounds_harm(margin)`, i.e. `−ci_lo < margin`, at its frozen epsilon:

| axis | epsilon |
|---|---:|
| `M2_flat_tercile_rho_{node_peak,map_cov}` | 0.01 |
| `M3_norm_p_at_8_{node_peak,map_cov}` | 0.01 |
| `M5_cell_rho_{f_q,cyclen}` | 0.02 |
| `M7_cell_mae_{node_peak,map_cov}` | 0.005 |
| `M7_abs_bias_node_peak` | 0.005 |

`M5_cell_rho_f_r` is scored and its harm bound reported, but is **excluded from
the verdict** while the F_r deferral holds (`flat_ab.FR_HARM_METRIC`, user
decision 2026-07-26) — unchanged for this round.

An interval that straddles the null (`PairedDiff.straddles_null`) answers
neither question and is **not** evidence of equivalence. Such an arm is HOLD.

### 5.4 Condition 3 — A1 only: the coverage check

A1's target-axis gain is recomputed restricted to the Dataset-P holdout cells
(§2.4). If the pooled gain establishes but the P-restricted gain does not, A1 is
**ESCALATE**, not PROMOTE: the trajectory labels exist only in P cells, so a gain
that is absent there cannot be attributed to the mechanism under test.

### 5.5 Verdicts

* **PROMOTE-candidate** — conditions 1 + 2 (+ 3 for A1) all hold.
* **HOLD** — condition 1's interval straddles the null and condition 2 holds.
* **REJECT** — condition 2 fails (an enforced harm bound is exceeded).
* **ESCALATE** — condition 3 fails for A1, or a target axis returns
  `insufficient` / `degenerate`.

### 5.6 Stacking

Two PROMOTE-candidates may be **stacked** into one combined arm only if they are
orthogonal on this evidence: their target-axis sets are disjoint **and** neither
shows a harm point estimate worse than its epsilon on the other's target axis.
A2 and A3 are disjoint by construction (cbc vs node_peak/f_q); A1 overlaps A2 on
`T_cell_mae_cbc_max`, so **A1 and A2 may not be stacked on round-2 evidence** —
a combined A1+A2 arm would need its own A/B. A stacked arm is a NEW arm and gets
a new run; it is never promoted on the sum of two separate results.

### 5.7 Then, and only then: gate-promote vs the champion

The round-2 winner (or the stacked winner) faces the existing honest gate against
`20260729_054749` via `lpopt/model/ab_decide.py` — `PRIMARY_IMPROVEMENT = 0.15`
on Δ75/SD, `MAX_REGRESSION = 0.05`, `SECONDARY_RHO_DROP = −0.02`, and
`promotion_allowed`. Round 2 decides which model *earns a gate attempt*; it does
not promote anything by itself.

### 5.8 Pre-disclosed minimum detectable effect

At 36 clusters, `alpha = 0.05`, `MDE_POWER = 0.80` (`ab_paired.MDE_POWER`), the
detectable paired median difference is `≈ 2.80 · se`, where `se` is the bootstrap
standard error reported per metric. `se` is data-dependent and therefore
**reported, not predicted**, here; the commitment is that the MDE is computed and
recorded from the same bootstrap that produces the interval, before any arm is
declared a winner, and that an arm whose observed effect is below its own MDE is
reported as underpowered rather than as null.

---

## 6. Expected GPU cost

Champion baseline on the same hardware (from its `train.log`): featurisation
985 s + five members 1865/968/1720/554/1240 s ≈ **2.1 h** wall for the whole
5-member run (early stopping at epochs 71–100, patience 15).

This store is 5.7% larger in train rows (53,528 → 56,578), so A0 ≈ **2.2–2.6 h**.

| arm | extra per-step cost | extra memory | estimate |
|---|---|---|---|
| A0 | — | — | 2.2–2.6 h |
| A1 | **+34%** measured (5 anchors × the shared decoder, on the ~16% of rows carrying a trajectory; rows without one are skipped inside `forward` — at 100% coverage the same code is +135%, so the row-skip is worth ~4x) plus ~1–2 min of label reads at featurisation | +262 MiB train / +53 MiB val of device-resident planes (`max_resident_gib = 40`) | 2.9–3.5 h |
| A2 | one gather + one add per step, < 1% | +0.5 MiB (`int64` codes) | 2.2–2.6 h |
| A3 | one 81-element `topk` per (row, channel) per step, ~1–2% | 0 | 2.2–2.6 h |

**Total ≈ 9.5–11.3 GPU-hours** for the four arms. Sequential on one GPU that is
~10–11 h wall; two arms per GPU on two GPUs is ~5–6 h.

Parameter counts: A0/A3 10,351,619 per member; A1 10,455,779 (+104,160, one
FiLM); A2 10,351,621 (+2 scalars).

---

## 7. Also fixed this round (not an arm)

`fit_cell_calibrations` silently dropped calibration artifacts when `model_dir`
arrived as a `str`: `lpopt/model/cell_calibrate.py` `_fit_cell_affine_target`
joined `model_dir / out_name` at the very END of a multi-minute serve-path fit,
so a `str` caller (`curriculum._fit_cell_calibrations` passes
`_retrain_local_full`'s return value, which is a `str`) did the whole fit and then
died on `TypeError: unsupported operand type(s) for /: 'str' and 'str'` — inside a
per-target `except Exception` that printed one `WARNING:` line and moved on.

The champion's own artifact timestamps show the footprint: the training run at
07:50 wrote `cell_calibration.json` + `f_r_calibration.json`; `cbc` landed in a
second manual pass at 17:17; `f_q`, `ao_abs` and `flatness` needed a **third**
pass at 18:23. The champion's `train.log` reports only two calibrations fitted.

Fixes:

* **Root:** `model_dir = Path(model_dir)` at the top of
  `cell_calibrate._fit_cell_affine_target` (`cell_calibrate.py:625`) and of
  `cell_calibrate.fit_flatness_calibration` (`cell_calibrate.py:915`).
* **Loudness:** `train._report_calibration_failure` (`train.py:2518`) prints the
  exception **type**
  and a full traceback **to stdout** (so it lands in `train.log` next to its
  banner), and `fit_cell_calibrations` ends with a
  `PER-CELL CALIBRATION(S) MISSING FROM <dir>` banner naming every missing
  artifact. Failures are still swallowed by default — a missing sidecar must
  never lose a multi-hour run — but are now recorded in the returned dict under
  `"failed"`, and `strict=True` re-raises for callers that would rather fail than
  ship a half-calibrated model dir (`train.py:2518–2540` the reporter,
  `2578–2581` the `Path` coercion + the `"failed"` key, `2615–2617` / `2628–2629`
  / `2649–2650` the call sites, `2651–2657` the summary banner).
* **Round-2 target axes:** `flat_metrics.AB2_TARGET_METRICS` /
  `AB2_ARM_TARGET_AXES` (`flat_metrics.py:427–450`) — additive; the frozen
  18-key `PRE_REGISTERED_METRICS` surface is untouched.

Consequence for this round: all four arms emit all six calibrations in one pass,
so the arms are calibration-comparable **by construction** rather than by
someone remembering to re-run three scripts.

---

## 8. Deliberately descoped

1. **Trajectory labels are not transposed under augmentation.** The diagonal
   transpose relabels `cells` only; `maps` (and now `traj`) reuse the base-row
   label. That is the champion's pre-existing convention. It is questionable for
   a radial map, but changing it would be a SECOND change and would break
   one-change-per-arm. Registered as a known inherited limitation, not fixed
   here.
2. **No trajectory OBJECTIVE.** A1 is supervision only. Whether the search should
   optimise a cycle-worst-case flatness instead of BOC flatness is the separate
   question `flatness_first_program_20260725.md` §3 registers its own
   falsification protocol for; nothing here pre-empts it.
3. **No serve-time trajectory prediction.** `out["traj"]` appears only when a
   caller passes `traj_frac`, which the serving path never does. Exposing a
   predicted burnup trajectory to the acquisition function is a separate design.
4. **`ga_native` offset is unmeasurable this round** (§2.2) — reported, not
   evidence. Fixing it needs Dataset B rows in the holdout, i.e. a split change,
   which would break "same split file for all arms".
5. **No trajectory coverage in Dataset A.** Producing `__traj` for the `feed=121`
   / `e_core` 5.35–5.60 mega-cells is a production task, not a model change.
   Condition 3 (§5.4) is how this round refuses to over-claim without it.
6. **`PRE_REGISTERED_METRICS` untouched.** Round-2 axes live in
   `AB2_TARGET_METRICS`; the flatness program's frozen 18-key surface is
   unchanged.
7. **No `--traj-anchors` / `--map-peak-topk` sweep.** Both are left at their
   defaults. A knob sweep inside an arm would make the arm a family and destroy
   the single-change property.
8. **`torch.compile` stays off.** A1's row-skip uses `nonzero`, which is a
   data-dependent shape and would trigger recompiles. The champion recipe has
   `torch_compile=False`, so nothing is lost.
9. **No deck / curriculum threading.** The three knobs are CLI-only, reachable
   through `lpopt.remote train -- …` — which is exactly how the arms launch.
   `curriculum._v5_train_config` / `_v5_train_flags` already do not thread
   `map_peak_weight`, `f_r_rank_weight` or the `distill_*` family either, so the
   champion recipe is not reproducible through the curriculum path today. Closing
   that gap is a separate, pre-existing task and is deliberately not bundled into
   a variance A/B.

---

## 9. Test inventory

| suite | tests | covers |
|---|---:|---|
| `tests/test_ab2_variance_arms.py` (**new**) | 60 | **the `_map_quarter` code-motion proof** — `forward` is compared BIT-FOR-BIT against the verbatim pre-extraction body at four shapes including the champion's (52ch / 224w / 8 blocks / multiscale + physics-prior skip + quantile head), because if the control arm's arithmetic moved then every paired difference measured against it is measuring the refactor; the traj label contract on real labels (endpoint identity, burnup clock, slot masking, malformed-label rejection, tie resolution, anchor tolerance); dataset + precompute + batch-gather plumbing on a synthetic store with a KNOWN step structure; the traj head (one-FiLM cost, both-dimensions gate, identity-FiLM ⇒ map-head equality, row skipping, gradient reach); flag-off byte-identity for all three arms (state_dict digest, forward-output keys, optimizer step with labels present); A2 provenance grouping / reference pinning / serve-invariance / z-units / gradient; A3 top-K selection, mask & NaN exclusion, off-identity, multiplication with `map_peak_weight`; end-to-end train → meta stamp → serve; **and the one-change-per-arm proof — the real `main` parser is driven on the four §3.1 commands and each arm's `TrainConfig` is diffed field-by-field against A0's** |
| `tests/test_auto_calibration.py` | 17 (+5 new) | the `str`/`Path` root fix, the loud-skip contract (exception type, traceback, missing-artifact banner), `strict=True`, and the `str` vs `Path` equivalence |
| touched-file regression | 371 across `test_model_net`, `test_net_shape_flags`, `test_axial_head`, `test_quantile_heads`, `test_dataset_torch`, `test_cyclen_rank_loss`, `test_train_parallel`, `test_auto_calibration`, `test_freeze_finetune`, `test_cell_calibrate`, `test_map_calibration`, `test_store`, `test_flatness`, `test_ab2_variance_arms` | all green |
| `test_flat_ab` / `test_ab_harness` / `test_rule_metrics` | 188 | the frozen scoring surface is unchanged by the additive `AB2_TARGET_METRICS` |
| `test_hires_bundle` / `test_v5_training_integration` | in the 283-test targeted run | the `map_loss` signature extension is backward-compatible for existing positional callers |
| **whole suite** | **1684 passed, 1 skipped, 1 failed** (24 m 22 s) | the one failure is `test_remote_infer.py::test_remote_gpu_matches_local_cpu_determinism`: remote-GPU vs local-CPU numerics on champion `20260721_061913` overshot its 1e-3 tolerance by 20% (1.197e-03) on the **sigmoid of the conv logit**. Pre-existing / environmental, not caused by this work: the conv path (`head_trunk` → `conv_head`) is untouched, and the map path is *proved* bit-identical by the code-motion test above. Re-running it would require touching the remote host, which is out of scope for this task. |
