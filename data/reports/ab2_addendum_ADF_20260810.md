# Round 6 — ADDENDUM: arm ADF (assembly discontinuity factors)

**Written 2026-08-10, BEFORE the v6c featurizer was implemented and before the
arm was launched.** Addendum to `data/reports/ab2_preregistration_20260730.md`,
inheriting its estimator, cell unit and non-regression rails unchanged, and
following the discipline of `ab2_addendum_BU_20260810.md`. Only what is stated
here is new. Anything added after the intervals exist is post-hoc and must be
labelled.

**Provenance of the mechanism:** `kcurve_fusion_memo_20260809.md` §3(1) channel
55, §2(5) (the leave-pair-out regression: ADF alone captures 74% of the k-curve
R² gain and *fixes* two regressions the k-curve created), and §7 — the STEP 0
H-family MASTER null test, which falsified P1 and named ADF as the mechanism:
treated ‖Δnode_peak‖ 0.0072 against a negative control of 0.0005, a **14×**
effect ratio, with ΔF_r 0.0351 = 5× the entire measured fuel lever.

---

## 1. A finding that reframes this arm, established before it was designed

**Three of the channels this arm was specified to add are already in v6b.**
Verified against `CHANNELS_BY_SCHEMA["v6b"]`:

| channel | status in the CONTROL (v6b / `split_S1b`) |
|---|---|
| `origin_adf_corner_g2` | **already present**, index 38 (since v4) |
| `origin_cr1_worth` | **already present**, index 34 (since v4) |
| `origin_ff_pin_max` | **already present**, index 30 (since v4) |
| `origin_adf_face_g2` | **absent — new** |
| `origin_adf_face_g1` | **absent — new** |
| `origin_adf_corner_g1` | **absent — new** |

Two consequences, both registered before any result:

1. **The arm is narrower than "add ADF".** It tests whether the **face** ADFs and
   the **group-1 corner** ADF add discrimination *beyond* the corner-g2 /
   control-rod-worth / pin-form-factor block v6b already carries. Naming it
   "the ADF arm" without this qualification would overstate it.
2. **The control is NOT structurally blind to H2/H4**, contrary to the framing
   this arm was commissioned under. Read off `fuel_types.parquet`:

| quantity | H2 | H4 | Δ | in the control? |
|---|---:|---:|---:|:--:|
| `adf_face_g2` | 1.08941 | 1.12447 | **+0.03506** | ❌ new |
| `adf_corner_g1` | 0.98649 | 1.00922 | +0.02273 | ❌ new |
| `adf_face_g1` | 1.01086 | 1.02025 | +0.00939 | ❌ new |
| `adf_corner_g2` | 1.27872 | 1.30325 | +0.02453 | ✅ **already** |
| `ff_pin_max` | 1.14300 | 1.17100 | +0.02800 | ✅ **already** |
| `cr1_worth` | 12087.5 | 11455.9 | **−631.6 pcm** | ✅ **already** |
| `u_avg_enrichment` | 5.50 | 5.50 | 0.00000 | — |

The control already sees a 631 pcm control-rod-worth difference and a 0.028
form-factor difference between H2 and H4. **Whatever the control predicts for
D1−D2, it is not predicting it blind**, and §5.4 rewrites the secondary readout
accordingly.

The negative control stays clean: H1 vs H3 are matched to ≤ 0.004 on **every**
one of the six, new and old alike (largest: `adf_corner_g2` −0.00395,
`cr1_worth` +24.9 pcm). So the D3/D4 arm remains a valid null for both models.

---

## 2. The arm

### 2.1 TREATED — `cond_schema = "v6c"`, append-only after v6b's 58

Four channels, indices 58–61, giving `in_channels = 62`. Indices 0–57 and all 13
globals are byte-identical to v6b, so a v6b/v6 checkpoint keeps loading.

| idx | channel | value | ref / scale |
|---:|---|---|---|
| 58 | `origin_adf_face_g1` | group-1 face ADF of the slot's fresh origin | (x − 1.001) / 0.014 |
| 59 | `origin_adf_face_g2` | group-2 face ADF | (x − 1.068) / 0.061 |
| 60 | `origin_adf_corner_g1` | group-1 corner ADF | (x − 0.968) / 0.029 |
| 61 | `origin_adf_present` | {0,1} presence gate for the block | sentinel |

Constants follow the repo's `_V4_SCALES` convention (**ref = population median,
scale ≈ half-range**) measured on the 117 harvested types, so each channel is
O(1) over the actual population:

| column | n | min | median | max | half-range |
|---|---:|---:|---:|---:|---:|
| `adf_face_g1` | 117 | 0.99260 | 1.00105 | 1.02025 | 0.01383 |
| `adf_face_g2` | 117 | 1.02338 | 1.06844 | 1.14588 | 0.06125 |
| `adf_corner_g1` | 117 | 0.95145 | 0.96792 | 1.00960 | 0.02907 |

**`adf_corner_g1` and `adf_face_g1` are included** (the coordinator left this to
my judgement). Cost is two extra stem-conv input planes out of 62 — negligible —
and both genuinely discriminate the H-family treated pair (+0.0227 and +0.0094)
while staying matched on the negative control. Excluding them would have thrown
away signal for no saving.

Each is traced to the slot's fresh chain origin through the **same
`_trace_chain`** the v6b channels use, read defensively via `getattr(vec, …, None)`,
NaN/absent → 0.0 with `origin_adf_present` distinguishing a genuine centered zero
from an unharvested one — the `origin_kinf_present` / `origin_lattice_present`
precedent, unchanged.

**One presence gate suffices**: all six ADF/CR/FF columns have *identical*
coverage (117 of 153 types), so they are present or absent together.

### 2.2 Coverage — and why the type-level gap does not bite

| library | types | with ADF |
|---|---:|---:|
| 260624 | 12 | 12 |
| 5.8_5.1 | 24 | 24 |
| CPHA | 12 | 12 |
| paramA | 33 | 33 |
| **ga80** | 70 | **36** |
| legacy_a | 2 | **0** |
| **total** | **153** | **117** |

The ga80 36/70 gap looks alarming and is not, because what matters is
resolvability at the **slot** level after the origin trace, not the type table:

| surface | slots sampled | ADF resolved |
|---|---:|---:|
| **frozen 36-cell decision surface** | 20,700 | **20,700 (100.0%)** |
| whole store | 27,600 | 27,393 (99.2%) |

**Every slot of every row on the scoring surface resolves.** The unharvested ga80
types are ones the store's patterns barely load. Registered now so a null cannot
later be explained away as "the channels were mostly zero" — they are not.

### 2.3 What this arm does NOT change

No head, loss, decoder, acquisition, calibration or serve-path change. No
k-curve precision work (memo §3(1) first half), no fusion (§3(3)–(5)), no
`g_sym_class` fix (§3(6)). Every training flag is `split_S1b`'s.

---

## 3. Frozen inputs

### 3.1 Store snapshot — the SPLIT-arm snapshot, re-verified, sourced from 238

This arm trains on **exactly** the snapshot `split_S1b` was trained on. The
canonical local store has since grown to **71,317 rows**; those **162
post-freeze rows are registered as excluded** and the store is sourced from
238's frozen copy, not from the local working tree.

| item | sha256 | bytes |
|---|---|---:|
| `records.parquet` | `00a6ecb0f986209e514d386067e075db606efa90aba210ce43f6eae533d75027` | 21,059,894 |
| `maps.npz` | `903ee1dada769becea3633ef97b6f0c84d8bc8703ea81b25989fa6ad306666b3` | 184,331,591 |
| `fuel_types.parquet` | `4ee9b16e4f595525c15168ea477fd92b6f39bb110147bf1b12c336f49c5c8ecd` | 61,296 |
| `S1b.json` | `47c19989a3ca9046f0186c38ca1470805582791323727877ce156211858f1c5b` | 5,812,568 |

All four verified byte-identical on 238 **and** re-verified after staging
locally. Rows 71,155; `S1b` train/val 59,634 / 11,521.

### 3.2 The control — reused, not retrained

`data/models/split_S1b` (promoted champion, `cond_schema = v6b`, 58 channels,
seeds 20260716–20260720, power prior M² 150 / extrap 2.0 / ρ 0.6888).

**Control-reuse justification, registered:** unlike the BU round — where the
incumbent predated the store snapshot and a fresh A0 was therefore mandatory —
`split_S1b` is a **fresh v6b model trained on exactly this snapshot and exactly
this split**, with the same recipe and the same five seeds. It already *is* the
same-data same-recipe control, so training a second one would buy nothing the
gate uses.

### 3.3 The confound that reuse does NOT cover — hardware. THREE venue states.

This round's weakest link, and it did **not** exist in the BU round (where both
arms ran concurrently on one card). The venue moved twice after this addendum was
first written; **all three states are recorded, with what each got wrong.**

| # | when | venue | outcome |
|---|---|---|---|
| **(a)** | 2026-08-10, first draft | registered as *238 Server Edition 600 W control* vs *181 Max-Q 300 W treated* | **the premise was WRONG — see below** |
| **(b)** | 2026-08-10 14:30:55 | **181 `ctrp-csh2`, Max-Q 300 W, Windows**, fresh `venv_gpu` (torch 2.11.0+cu128, sm_120) | **launched, then KILLED at ~101 min by user order** — the box overheated (88 °C at 99% util, 300 W cap). Run dir and launcher deleted; `venv_gpu` retained. No usable output. |
| **(c)** | 2026-08-10 16:15:12 | **238, GPU 1, Server Edition 600 W, Linux** — `CUDA_VISIBLE_DEVICES=1`, tmux `lpopt_adf_v6c` | **final venue** |

**Correction to (a), on the coordinator's information and consistent with the
run scripts on disk.** State (a) asserted the control `split_S1b` was trained on
a 600 W Server Edition. **That is wrong.** `runs/split_S1b/run.sh` and
`runs/bu_T/run.sh` both pin `CUDA_VISIBLE_DEVICES=0`, and the card in 238's slot
0 *at that time* was the **Max-Q**, which has since been physically replaced by a
Server Edition. So **the control was trained on a 300 W Max-Q, in 238, on
Linux** — the same host and stack the treated arm now uses, but a different
physical card at a different power limit.

**Why (c) is the most comparable venue available.** Against the control, venue
(c) holds constant: host, OS, kernel, driver, CUDA/torch build, filesystem,
the frozen store and split, the recipe and the five seeds. It differs in exactly
one axis — **the physical card and its power limit (Max-Q 300 W → Server Edition
600 W)**. Venue (b) would additionally have crossed OS (Linux → Windows), a
separately-built venv and a different CPU, so **(c) strictly dominates (b) on
comparability**, and the heat kill removed the choice anyway. The residual
confound is therefore **card + power only**, not card + OS + stack.

It still cannot be driven to zero: GPU training is not bit-deterministic, so
`arm − control` folds in run-to-run *and* card-induced variation on top of the
channel effect. Retraining a v6b control on the 600 W card would close it, and
this round is scoped not to do that.

> **Registered consequence (reworded 2026-08-10, superseding the original
> rule).** The original rule said a marginal result should trigger "a
> same-hardware v6b control", which now means *rerun on the Max-Q* — moot, since
> that card is out of 238 and 181 is off-limits for heat. **Replacement rule: a
> gate outcome landing within roughly one MDE₈₀ of the bar, in either direction,
> is reported as `UNRESOLVED-HARDWARE` — not as PASS, HOLD or REJECT.** Only an
> effect that clears the bar by more than its own MDE₈₀, or fails it by more
> than that, may be called a verdict on this evidence.

### 3.4 Training venue — changed, and part of the record

**Superseded twice; the final venue is (c) of §3.3.** The table below is the
FINAL state. The 181 attempt (b) is retained in §3.3 only as a record.

| | control `split_S1b` | **this round (final)** |
|---|---|---|
| box | 238 (Linux) | **238 (Linux)** — same host |
| GPU index | 0 | **1** (`CUDA_VISIBLE_DEVICES=1`) |
| card | RTX PRO 6000 **Max-Q, 300 W** (since replaced) | RTX PRO 6000 **Server Edition, 600 W** |
| VRAM | 97,887 MiB | 97,887 MiB |
| stack | server venv, torch 2.11.0+cu128 | **identical server venv** |
| launch | tmux + DONE/FAILED markers | **tmux `lpopt_adf_v6c`, same mechanism** |
| store / split | frozen snapshot in `~/lpopt_ws` | **the same files, hashes re-verified, never re-pushed** |

The push that carried v6c used **`data=False`**, so `~/lpopt_ws/data` was not
rewritten: all four frozen hashes were re-verified byte-identical *after* the
source push (§3.1). Only the `lpopt` source tree changed on the box.

Idle verified before launch: GPU 0 at 682 MiB / 0% util, no compute jobs, no
other user's SCALE/MASTER processes (only an ASUS updater and tailscale).
181's MASTER production role has ended per the 2026-08-10 reassignment.
**Expect the OLD wall-clock (~2.5–3.5 h)**: the Max-Q is 300 W-limited, so this
is the slower of the two cards, not the upgrade.

**Environment bootstrap (new, and part of the record).** 181's existing kit venv
carries **torch 2.11.0+cpu** — unusable for GPU training. Rather than replace
torch inside a working production venv, a **fresh** venv was built at
`C:\Users\USER\lpopt_work\venv_gpu` from the box's Python 3.11.9, leaving
`kit_frontier\venv` untouched. Verified on the box before launch:

| check | result |
|---|---|
| torch | **2.11.0+cu128**, `cuda_avail True` |
| device / capability | RTX PRO 6000 Blackwell Max-Q, **sm_120** |
| 4096² CUDA matmul | finite, `MATMUL_OK` |
| VRAM free | **93.5 / 95.6 GiB** |
| registered schemas | v2…v6b **+ v6c** |
| channel counts | v6b 58 → v6c 62, `v6c[:58] == v6b` |
| `S1b` train/val | 59,634 / 11,521 |
| store rows | 71,155 |
| v6c encode | `(62,19,19)`, all finite |
| **v6b prefix under v6c** | **bit-identical**, globals bit-identical |

Data on 181 re-hashed after transfer, all four matching §3.1:
`records.parquet 00a6ecb0…`, `maps.npz 903ee1da…`,
`fuel_types.parquet 4ee9b16e…`, `S1b.json 47c19989…`.

Launch mechanism: `run_adf_v6c.bat` (chcp 65001 + `PYTHONUTF8` + log +
`rc`/`DONE`/`FAILED` markers) started detached via
`Invoke-CimMethod Win32_Process Create`. `schtasks` is documented to silently
no-op on this box and is not used.

---

## 4. Exact launch command

Identical to `split_S1b`'s except `--cond-schema v6c` (was `v6b`).

```
python -m lpopt.model.train \
  --ensemble 5 --split S1b --cond-schema v6c --width 224 --n-blocks 8 \
  --head-hidden 384 --epochs 150 --num-workers 8 --device auto \
  --parallel-members 5 --base-seed 20260716 \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

**Registered prediction, checkable from the log:** the power prior must refit to
**exactly M² = 150, extrap = 2.0** — `kinf_quarter` does not read ADF, and the
fit is deterministic (`random_state=0`). A different value would mean something
other than the ADF channels moved, and is grounds to stop and investigate rather
than to score.

---

## 5. The decision rule — fixed before any result exists

### 5.1 Surface, estimator, control — inherited from the BU round

Frozen **36-cell** holdout, **3,207** rows, read from
`S1b.json → groups['ab2_frozen_val_by_cell']` (the verbatim historical surface
preserved by the split rebuild — **not** the grown `curriculum_val_by_cell`, per
`ab2_addendum_SPLIT_20260810.md` §3.3). Unit = the cell. Control = `split_S1b`.
`flat_ab.paired_metric` → `ab_paired.paired_cell_bootstrap`, `method="bca"`,
`aggregate="median"`, `reps=2000`, `alpha=0.05`, `seed=0`, sign-flipped so
`theta > 0` means "treated beats control". Treated reindexed onto the control's
`record_id` order before the arena is built. Degradation rules
(`insufficient` / `degenerate` / percentile fallback) inherited verbatim.

### 5.2 Target axis

`T_cell_mae_node_peak` ≡ `M7_cell_mae_node_peak`. One primary axis, as in BU.
`cyclen`, `cbc_max`, `map_cov` are harm rails only and may not promote the arm.

### 5.3 Condition 1 — the gate

**Both clauses jointly:**

1. `point ≥ max(MDE₈₀_measured, 0.00409)` — the measured MDE₈₀ from this arm's
   own bootstrap, floored at the round-1-derived 0.00409 bar so the bar can only
   ever get *stricter*, never looser, than the one BU cleared; **and**
2. `ci_lo > 0`.

### 5.4 Condition 2 — harm bounds

Inherited verbatim from BU: `T_cell_mae_cyclen` ≤ 0.10 EFPD, `T_cell_mae_cbc_max`
≤ 1.0 ppm, `M7_cell_mae_map_cov` ≤ 0.002, **plus** the full `flat_ab.HARM_MARGINS`
rails; where a metric appears in both, the **stricter** epsilon binds.
`M5_cell_rho_f_r` scored, reported, excluded from the verdict (F_r deferral).
Any of the five variance axes established worse (`ci_hi < 0`) is a REJECT.

### 5.5 Verdicts

**PASS** / **HOLD** / **REJECT** / **ESCALATE** exactly as BU §5.5. Seeds 0–4
reproduction on every decisive interval (§5.7 of BU, inherited).

### 5.6 Registered SECONDARY readout — the H-family, with the premise corrected

Serve **both** models on the four all-fresh H-family patterns (`fr_arms.py`
D1–D4, fixed pattern `b0ff11ef16de`, 121 fresh slots loaded with a single type)
and report each model's predicted Δnode_peak against MASTER:

| contrast | MASTER measured | threshold |
|---|---:|---:|
| **D1 − D2** (H2 vs H4, treated) | **+0.0072** | 0.005 |
| **D3 − D4** (H1 vs H3, negative control) | **+0.0005** | 0.005 |

**The premise is corrected per §1.** The original framing — "the treated model
should discriminate H2/H4 where the control structurally cannot" — is **false**:
the control already carries `cr1_worth` (Δ −631 pcm), `ff_pin_max` (Δ +0.028) and
`adf_corner_g2` (Δ +0.0245) for exactly this pair. So:

* **This is not a can-vs-cannot test.** It is a *does-the-face-ADF-help* test.
  Both models can in principle separate H2 from H4.
* **Registered readings, fixed now.** Treated closer to +0.0072 than control ⇒
  the face ADFs add discrimination. Both far off ⇒ neither encoding reaches this
  physics, and the D1/D2 result stands as an open challenge to the whole
  descriptor programme. Control already accurate ⇒ **the corner-g2/CR/FF block
  was sufficient and this arm was unnecessary**, which is a legitimate and
  publishable outcome.
* Either model firing on the negative control (|Δ| > 0.005 for D3−D4) marks the
  readout **VOID**, exactly as the MASTER protocol does.

**It decides nothing** — §5.3 is the instrument. **Power disclosure:** n = 4
patterns, no interval, and these are 121-fresh-slot cores far outside the
equilibrium-pattern distribution both models were trained on. It is a
directional sanity check, not evidence.

**Label hygiene, verified before registering:** the D1–D4 MASTER results live in
`5_RL/runs/fr_arms_d/fr_arms_results.jsonl` — a `runs/` artifact, **not** in
`data/store/records.parquet` — so no training row of either model contains them.
Re-verified mechanically by the scoring script: all four constructed D-arm
patterns are **absent from the store** (checked, 4/4 absent).

#### 5.6.1 The CONTROL's baseline, measured 2026-08-10 before the treated model existed

The control needs no training, so its half of this readout was computed
immediately — **before the treated arm finished, and recorded here rather than
after**:

| arm | MASTER | control (v6b, 58 ch) | abs err |
|---|---:|---:|---:|
| D1 (H2) | 1.4613 | 1.4077 | −0.0536 |
| D2 (H4) | 1.4541 | 1.4008 | −0.0533 |
| D3 (H1) | 1.3916 | 1.4260 | +0.0344 |
| D4 (H3) | 1.3911 | 1.4266 | +0.0355 |

| contrast | MASTER | control | |
|---|---:|---:|---|
| **D1 − D2** (treated) | **+0.0072** | **+0.0069** | error **0.0003** |
| **D3 − D4** (neg. control) | +0.0005 | −0.0006 | inside 0.005, **does not fire** |

**Two things follow, both registered now.**

1. **The third registered reading of §5.6 has already landed: the control is
   already accurate on this contrast.** It reproduces the STEP 0 treated
   difference to 0.0003 — **24× smaller than the effect** and 1/17 of the
   0.005 threshold — while correctly staying silent on the negative control. On
   this test the v4 corner-g2 + CR-worth + form-factor block is **sufficient**,
   and the face ADFs have essentially no headroom left to demonstrate. Whatever
   the treated arm returns here, it cannot show much, and **a "treated also gets
   it right" result is not evidence for the arm.**
2. **But the control is right for a narrow reason.** Its absolute levels are
   wrong by −0.053 / +0.035, and it gets the **across-pair ordering backwards**:
   MASTER puts the D1/D2 cores ~0.07 *above* D3/D4, the control puts them ~0.02
   *below*. It reproduces the *within-pair differential* while misplacing the
   pairs entirely. So §5.6 tests a genuinely narrow skill, and its near-perfect
   control result must not be read as "the control understands the H family".

**Consequence for the round, registered:** §5.6 is now near-vacuous as a
discriminator and the arm rests on the §5.3 gate essentially alone. This is
disclosed before the gate is computed so it cannot be presented afterwards as
either a success or an excuse.

### 5.7 Falsification

> If the gate fails, the claim "the face/corner-g1 ADFs are a lever for node
> peaking, on top of the corner-g2 + CR-worth + form-factor block v6b already
> carries" is **rejected**. Given that memo §7 already falsified P1 (k-curve +
> enrichment insufficiency) by MASTER, a failure here would leave the
> assembly-descriptor route with **no** demonstrated lever on node_peak and would
> put KILLER 1 — no descriptor moves within-fuel pattern discrimination by more
> than 0.004 of ρ — in possession of the field. The registered next step in that
> case is **not** another descriptor arm; it is the verify-many-then-select
> mechanism (pre-registration §1's last untested successor, and K2's 7.7×
> cheaper intervention).

Explicitly forbidden by this addendum: adding the k-curve precision channels to
this arm mid-round; re-running it with more descriptor channels after a null;
`--ensemble 20`; substituting the §5.6 H-family readout for the §5.3 gate.

---

## 6. Pull, scoring and promotion — ready before the arm finishes

Back on 238, so the standard `lpopt.remote` contract applies throughout; the
181 SFTP route is obsolete and is not used.

**Stage 1 — pull the candidate** (the issuer does this; no local `data/models/`
write happens until then):

```
python -m lpopt.remote --input lpopt.inp pull --ts adf_v6c
# -> data/models/adf_v6c/
```

**Stage 2 — the PRIMARY gate (§5.3).** Same instrument and surface as the BU
round. The control's served rows already exist locally at
`5_RL/runs/split_S1b/rows_split_champ.csv` (served on S1b val), so only the
treated arm needs serving. `eval_accuracy_split.py` is already on 238 from the
SPLIT round:

```
ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/adf_v6c adf_cand S1b'

cd 5_RL/runs/adf_v6c
scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_adf_cand.csv .
cp ../split_S1b/rows_split_champ.csv .
```

Then the paired bootstrap on the **frozen 36-cell** surface, control =
`rows_split_champ.csv`:

```
cd 5_RL
AB2_BU_DIR=runs/adf_v6c AB2_SPLIT=S1b \
AB2_CONTROL_CSV=rows_split_champ.csv AB2_TREATED_CSV=rows_adf_cand.csv \
AB2_BU_OUT=data/reports/ab2_verdict_ADF_20260810.json \
python ab2_bu_verdict.py
```

`ab2_bu_verdict.py` now **prefers `ab2_frozen_val_by_cell` whenever the split
carries it**, so it restricts to the verbatim **3,207**-row historical surface
rather than S1b's grown 3,327 — the two differ by the 120 rows the split rebuild
added to three cells, and MDE₈₀ = 0.00409 is only comparable on the former. The
selected key is stamped on the artifact as `checks.surface_key`; **confirm it
reads `ab2_frozen_val_by_cell` before trusting the verdict.**

**Stage 3 — the SECONDARY H-family readout (§5.6), reported either way:**

```
python 5_RL/adf_hfamily_readout.py --treated data/models/adf_v6c
# -> data/reports/adf_hfamily_readout_20260810.json
```

Already run for the control alone (§5.6.1); it re-runs both models once the
treated dir exists, re-checks label hygiene, and refuses if any D-arm pattern
turns out to be a store row.

---

## 7. Cost

Single arm, 5 members, `--parallel-members 5`, **238 GPU 1 (Server Edition,
600 W)**. Expect ~2–2.5 h — faster than the 300 W-capped estimate the 181 plan
carried, and the card is not shared. Serve cost: +4 input planes on the stem conv
only, 5 members unchanged, inference wave unchanged.

**Wasted cost, recorded:** the killed 181 attempt burned ~101 min of wall and
produced nothing usable; its run dir and launcher were deleted so no partial or
stale `FAILED` state can reach later scoring. `venv_gpu` on 181 was retained.
