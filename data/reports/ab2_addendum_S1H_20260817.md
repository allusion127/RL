# Round 12 — ADDENDUM: arm S1h (`cond_schema = v7`, the 3-fresh-type scorer)

**Written 2026-08-17, BEFORE `data/splits/S1h.json` was consumed by any trainer,
before the push to 238 and before the arm was launched.** Same discipline as
`ab2_addendum_S1G_20260816.md` (round 11); instrument, split rule, preservation
checks, recipe freeze and noise-floor reporting rules inherited unchanged. Only
what is new is stated here.

**This is NOT a clean data-growth retrain.** It is an *enabling* retrain: the arm
exists because the 3-type campaign cannot be run at all without a v7 checkpoint.
Instrument = **gate-promote** against champion **`s1g`**. Read §4.1 before
quoting any number from it — three things move at once this round and the
addendum says so up front rather than in a footnote.

---

## 1. Why this arm exists

`data/reports/tripletype_design_20260817.md` extended the searcher to 3 fresh
types (`MAX_FRESH_TYPES = 3`, `graded_morph`, `case_batches`, triple asset
resolution) and added `cond_v7` = v6c + 5 appended composition globals
(13 → 18). Its §3.2 states the blocker plainly:

> 현 챔피언(s1g)은 **v6b/v6c 계열**이고 글로벌 13개다. `cond_v7` 은 18개 →
> **checkpoint 호환 불가**, `model_api` 가 글로벌 차원 불일치로 정직하게 거부한다.

So this round is not optional and its value is not measured by the gate delta.
Without a promoted v7 ensemble there is **no model that can score a triple**, and
the first 3-type campaign is unrunnable. The gate's job here is narrower than
usual: confirm the v7 candidate is **not worse** than `s1g` on the frozen
surface, so that switching schema does not silently cost the programme the
8th champion's accuracy.

The increment is real but incidental: 474 rows that post-date S1g, from three
optimized frontier campaigns and eleven mesh-v3 anchor strata (§3).

---

## 2. Frozen inputs

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `b5b29460a0c1c1e6ad3e3b2b9f410ed021e306099a91eceec8b05775b739cae7` | 22,147,028 |
| `data/store/maps.npz` | `8af4447eee46960b682171be55b5cbfda14dcbb9d73a2efb11d518433adeecfe` | 209,199,023 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| parent `data/splits/S1g.json` | `591a9d6bf8e65ee57b329120fa8e6c7fe065295f01f7d6722631da4991d86b58` | 5,964,459 |
| built `data/splits/S1h.json` | `5dfb6b05f05d698a82881d26db7d2deb0fd9639d6789e64cd24e9030ee31947a` | — |

Rows **74,477** (65,922 converged). `fuel_types.parquet` unchanged for the sixth
round running. Pre-merge backups exist as
`*.bak_pre_E1E2f109_20260817`, `*.bak_pre_meshv3anchor_20260817`,
`*.bak_pre_HGD569f109_20260817`, `*.bak_pre_HGD569f125_20260817`.
**Control:** `data/models/s1g`, frozen on disk (`DONE`, rc 0). 238 holds the
**round-10** store (74,003 rows, `records.parquet` 21,979,093 B, mtime Aug 16
18:50), so this round pushes `data=True` and re-verifies sizes on the box.

**Recipe frozen by the scaling verdict** (`scaling_results_20260815.md`):
`width 224 / n-blocks 8 / head-hidden 384`, **5 members not 10**, v6b flag set
verbatim — **except `--cond-schema v6b → v7`, which is the point of the arm.**

**Noise floors bind the reporting**, inherited unchanged from round 10 §2.2:
within-cell ρ movement below **~0.01 on cyclen / map_cov is NOT attributable**
to this refresh (ensemble-identity noise ~0.009 / ~0.007); single-seed spread
carries sd ≈ 0.018 on ρ F_r; `mde80` ≈ 0.009–0.015.

---

## 3. The split — S1h

`build_split_S1b.py --parent S1g --name S1h --holdout-new-campaigns`. S1g
assignments verbatim; stable-hash 80/20 on the increment only. `--self-check`
run first and **PASS** (flag OFF ⇒ val growth 0; flag ON ⇒ val growth 73 of 474
new-only rows; a/b/c growth-invariance held both ways).

| | |
|---|---:|
| train / val | **62,379 / 12,098** (+401 / +73) |
| cells created | **14** · cells grown: **0** |
| rows landing in a pre-existing cell | 0 (every new row is a new-only campaign) |

Increment composition — **all 474 rows are new-only campaigns**, so every one of
them receives the 80/20 holdout:

| campaign | new | conv | → val / train |
|---|---:|---:|---|
| `fpcamp_minfr_E1E2_f109` | 100 | 100 | 20 / 80 |
| `fpcamp_minfr_hgd569_f109` | 60 | 40 | 8 / 52 |
| `fpcamp_minfr_hgd569_f125` | 60 | 57 | 11 / 49 |
| `mv3_flag_e614_f109` | 40 | 24 | 5 / 35 |
| `mv3_e569_f125` | 24 | 16 | 3 / 21 |
| `mv3_ctl_ngd22_e616_f109` | 23 | 16 | 3 / 20 |
| `mv3_e569_f109` | 23 | 16 | 3 / 20 |
| `mv3_e583_f109` | 22 | 16 | 3 / 19 |
| `mv3_flag_e614_f117` | 22 | 16 | 3 / 19 |
| `mv3_ctl_ngd22_e616_f125` | 22 | 16 | 3 / 19 |
| `mv3_e583_f125` | 22 | 16 | 3 / 19 |
| `mv3_flag_e614_f125` | 21 | 16 | 3 / 18 |
| `mv3_e583_f117` | 19 | 16 | 3 / 16 |
| `mv3_e569b_f109` | 16 | 10 | 2 / 14 |

**Note, registered:** the coordinator's brief named the increment as
"hgd569 f109/f125 + the mv3_ anchor strata". The store also carries
`fpcamp_minfr_E1E2_f109` (100 rows), merged after S1g was frozen and therefore
part of this increment by the *only* rule the builder has — "not in the parent".
It is included; the brief's list was not exhaustive and no row was hand-picked.

| # | check | result |
|---|---|:--:|
| **a** | every S1g **val** id stays val | ✅ |
| **b** | every S1g **train** id stays train | ✅ |
| **c** | only the 474 new ids get fresh assignments | ✅ |
| **d** | every S1g cell present, ids retained, order intact | ✅ |
| **d′** | **`ab2_frozen_val_by_cell` forwarded verbatim — 36 cells / 3,207 rows** | ✅ |
| — | no train/val overlap · every store row in exactly one fold | ✅ |

The round-10 §3.1 exclusion still stands: the three interventional waves
(`ablation_1move_T6T4`, `batchswap_enum_T6T4`, `batchswap_enum_625_T6T4`) have a
non-independent holdout and remain **excluded from every reported readout**.

---

## 4. The arm

v6b recipe, identical to `s1g`'s **except `--split S1h` and
`--cond-schema v7`**; seeds 20260716–20260720, `--parallel-members 5`,
**238 GPU 1** (`CUDA_VISIBLE_DEVICES=1`), occupancy checked first (both GPUs
idle at arming: 237 MiB / 1 MiB used, 0 % util, no tmux server).

```
--ensemble 5 --split S1h --cond-schema v7 --width 224 --n-blocks 8 \
--head-hidden 384 --epochs 150 --num-workers 8 --device auto \
--parallel-members 5 --base-seed 20260716 --map-decoder multiscale \
--map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
--cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
--promote-max-asm-bu --distill-targets data/models/_v5_distill_soft.npz \
--distill-weight 0.4 --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

### 4.1 THE CONFOUND — three changes, one arm, stated before the gate runs

`s1g` is **v6b: 58 cell channels, 13 globals**. The candidate is **v7: 62 cell
channels, 18 globals** on a store 474 rows larger. Measured directly, not
assumed:

| | s1g (v6b) | S1h candidate (v7) | delta |
|---|---:|---:|---|
| cell channels | 58 | 62 | **+4** — `origin_adf_face_g1/_g2`, `origin_adf_corner_g1`, `origin_adf_present` (the v6c ADF block; `v6c[:58] == v6b`) |
| globals | 13 | 18 | **+5** — `g_type_frac_1..3`, `g_e_type_std`, `g_n_fresh_types` |
| train rows | 61,978 | 62,379 | **+401** |

So a gate delta this round is attributable to **schema OR data OR the ADF block
in combination**, and to none of them individually. That is inherent — v7 is
*defined* on v6c, so the ADF channels ride along whether or not they are wanted.
Worse, the ADF block has its own history: the **`ADF` arm of 2026-08-10**
(`ab2_addendum_ADF_20260810.md`, `ab2_verdict_ADF_20260810.json`) tested exactly
these 4 channels at S1b and **`condition1_gate.PASS = false`** — the block did
not clear its pre-registered effect bar and v6c was never promoted. This round
re-admits it as a *carrier*, not on evidence that it helps.

**Registered consequence:** no sentence in the results report may read "v7 helped"
or "the composition moments helped". The only claim this arm can support is
*"the v7 candidate does / does not regress against s1g on the frozen surface"*.

### 4.2 THE NEW GLOBALS ARE UNLEARNABLE FROM THIS CORPUS — measured, not argued

Every one of the 74,477 store rows is a **2-type** record. Encoding a 400-row
random sample at v7 and at v6c (`FeatureEncoder.encode_batch`, real store +
`fuel_types.parquet`) gives:

| check | result |
|---|---|
| v7 cells vs v6c cells | **bit-identical** |
| v7 globals `[:13]` vs v6c globals | **bit-identical** |
| `g_type_frac_1` vs existing `g_split_frac` | `max abs diff = 0.0` — **exactly redundant** |
| `g_type_frac_2` | `1 − g_type_frac_1` (affine; sole exception below) |
| `g_type_frac_3` | **constant 0.0**, 1 unique value in 400 |
| `g_n_fresh_types` | **constant 0.666667**, 1 unique value in 400 |
| `g_e_type_std` vs `sqrt(w(1−w))·g_e_split` | `max abs diff = 6.0e-08` — **deterministic function of two existing globals** |

That is the honest statement of what v7 buys: on the *current* corpus the five
new globals carry **zero** information not already present. They are not features
the model can learn from — they are **representational capacity** so that a
3-type input is encodable at all and is not silently aliased onto a 2-type
encoding. Their weights at the end of this training are fitted on a corpus where
three of them never vary; at campaign time the triple case will drive
`g_type_frac_3 > 0` and `g_n_fresh_types = 1.0` for the first time, i.e. **out of
distribution on channels the model has never seen move.**

**Registered in advance:** this is the strongest reason the first 3-type campaign
must be read as an *exploration*, not as a model-guided optimum, and the reason
the design note's checklist item 6 ("`graded_morph` seed 의 F_r 이 2종 부모 대비
개선되는지 첫 wave 에서 판정") is the load-bearing check rather than the
predicted ranking.

**Encoder quirk, disclosed:** the degenerate self-pair `A1_A1` (**3 rows** in the
whole store) makes `_composition_globals` read both members off the same
`batch_feed` key, so its fractions sum to 2 and `g_n_fresh_types` reads 2/3
rather than 1/3. 3 / 74,477 rows; not fixed inside this arm (a code change
during a frozen-recipe retrain is inadmissible), recorded so the next reader
does not rediscover it as a bug in the results.

---

## 5. Decision rule — fixed before the gate runs

### 5.1 Primary

**`lpopt gate-promote`**, prev = `data/models/s1g`, new = the pulled candidate,
both on **S1h**. Promotion **authorized on PASS** (coordinator instruction); the
gate JSON is written to `data/reports/gate_s1h.json` before any promotion is
acted on. Generalization is judged on the **frozen 36-cell surface**
(3,207 rows), which contains none of these 474 rows.

**Why the gate is fair across a schema change** — registered, because it is the
one thing a reader will doubt: `PosValCnnBackend.from_dir` rebuilds each model's
encoder from **its own checkpoint `meta.json`**, so the v6b champion is served at
v6b and the v7 candidate at v7, on the identical row set and order. And per §4.2
the v7 encoding of a 2-type row is the v6c encoding plus five deterministic
functions of channels already present: the candidate is handed **no label
information the champion lacks**. The comparison is therefore between two
scorers, not between two information sets.

Precedent for the bar: `gate_s1g.json` passed at `worst_drop 0.0456` against
`epsilon 0.1388`, `blind_targets []`, legacy tail PASS at `epsilon 2.0`.

### 5.2 Registered SECONDARY readouts — decide nothing

**(a) The standing `T6_T4` series** (campaign cells only, waves excluded per
§3.1): **183 val rows, 8 cells, 7 cells ≥ 8** — clears `MIN_CELLS_BCA = 6`, BCa
available. True F_r min 1.4749, sd 0.1014. **Unchanged from round 11 by
construction** (no T6_T4 rows in this increment), so it functions as a
**stability check**: movement here is evidence of collateral drift from the
schema change, not of progress. This is the single most informative secondary
readout this round precisely *because* its data did not change — it isolates the
schema swap on a surface the increment cannot touch.

**(b) The three new frontier campaign cells — fit checks, near-tautological.**
The champion has **zero** of these rows; the candidate trains on ~80 % of each.

| cell | val rows | true F_r min / median / max | sd |
|---|---:|---|---:|
| `fpcamp_minfr_E1E2_f109` | 20 | 1.5134 / 1.5552 / 2.1227 | 0.1662 |
| `fpcamp_minfr_hgd569_f125` | 11 | 1.6088 / 1.6599 / 1.8813 | 0.0732 |
| `fpcamp_minfr_hgd569_f109` | 8 | 1.6867 / 1.7807 / 1.8581 | 0.0555 |

Three cells is **exactly `ab_paired.MIN_CELLS = 3` and below
`MIN_CELLS_BCA = 6`**, so any interval degrades to percentile; n = 8–20 per cell.
A gain here is the *expected* result and confirms only that the labels were
absorbed. **These are the first optimized-frontier labels the paramA high-Gd cell
has ever had**, which is why they are reported at all.

**(c) The eleven `mv3_*` anchor strata — the OOD-breadth readout.**
34 val rows over 11 cells, true F_r min 1.8184 / median 2.3187 / max 3.6902,
sd 0.4164. Eleven cells clears `MIN_CELLS_BCA = 6` on cell count, but **2–5 rows
per cell** is thin and the F_r span sits **entirely outside the decision band**
(F_r < 1.55) the optimizer works in — the round-11 §5.2(b) caveat applies
verbatim. Reported as breadth coverage, never as search skill.

**(d) NOT RUN this round: `mesh_vs_db.py` low-feed pessimism.** Round 11 §5.2(c)
registered `gap_total` at f105/f109 as its interesting readout. It is **dropped
here deliberately**, and the reason is registered rather than discovered later:
the script scores a *named ensemble's* frontier bias against MASTER truth, and a
v6b-vs-v7 comparison on that axis would inherit the whole §4.1 confound with no
way to attribute it. Re-running it would produce a number nobody could read.
The **s1f/s1g baselines in `data/reports/scoping_mesh_20260815/` are therefore
left untouched** — no `model_bias.csv` overwrite occurs this round.

### 5.3 Falsification — registered before the gate runs

> If gate-promote **FAILS**, S1h is rejected and `s1g` stands as champion. The
> registered reading is **not** "v7 is worse" — with §4.1's three simultaneous
> changes, a FAIL cannot be attributed. It means the v7 candidate did not clear
> the no-regression bar, and the champion line does not move.

**And the consequence must be stated in advance, because it is unusual:** a FAIL
does **not** merely postpone a model upgrade, it **blocks the 3-type campaign**,
since no promoted checkpoint can score an 18-global input. The registered next
step on a FAIL is therefore a decision for the coordinator, not an automatic
one, between exactly three options, in this order of preference:

1. **Disentangle** — retrain at `--cond-schema v7` on the **parent split S1g**
   (data held fixed at the champion's own corpus). That isolates schema+ADF from
   data growth and is the only run that could say which caused the FAIL.
2. **Run the campaign under an unpromoted v7 model**, explicitly labelled as
   such, with the campaign report carrying the failed gate. Admissible only as a
   deliberate, recorded exception — a campaign steered by a model that failed
   its own no-regression gate is exploration, not a deliverable.
3. **Stop** and re-scope the 3-type work.

**No option is taken silently.** On FAIL this agent reports and does not promote.

### 5.4 What this round cannot conclude

No same-recipe-same-data control, and this round has *more* than the usual
number of moving parts (§4.1). A PASS promotes a model; it establishes neither
that data growth is the lever, nor that the composition moments are, nor that
the ADF block has been vindicated — §4.2 shows the composition moments could not
have been learned from this corpus at all. Inherited from round 5 §5.4. And per
§2, no delta below ~0.01 on cyclen/map_cov may be attributed to this refresh.

---

## 6. Pull, scoring and promotion

```
python -m lpopt.remote --input lpopt.inp pull --ts s1h        # -> data/models/s1h/

python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1g --new data/models/s1h \
    --out data/reports/gate_s1h.json --check-only
# promotion AUTHORIZED on PASS (§5.1): re-run without --check-only

ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
    ./venv/bin/python eval_accuracy_split.py runs/s1h  s1h_cand  S1h && \
    ./venv/bin/python eval_accuracy_split.py runs/s1g  s1h_champ S1h'
cd 5_RL/runs/s1h && scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_s1h_{cand,champ}.csv .
```

Optional frozen-surface cross-check:

```
cd 5_RL && AB2_BU_DIR=runs/s1h AB2_SPLIT=S1h \
  AB2_CONTROL_CSV=rows_s1h_champ.csv AB2_TREATED_CSV=rows_s1h_cand.csv \
  AB2_BU_OUT=data/reports/ab2_verdict_S1H_20260817.json python ab2_bu_verdict.py
```

Confirm `checks.surface_key` reads `ab2_frozen_val_by_cell` (3,207 rows), and
exclude the three interventional waves from §5.2(a) per round 10 §3.1.

---

## 7. What happens on PASS

`s1h` becomes the 9th champion and the **first checkpoint in the programme that
can score a 3-fresh-type core.** The first 3-type campaign
(`P6253Z1G06N24 _ <mid> _ P6253Z2G10N24` at feed 125) is then armed against it
under its own pre-registration, with the mid type chosen by serving predicted
pin BU for both candidates (`P6253Z2G08N16` / P5 and `P6253Z2G10N20` / T1) under
this model — a decision this addendum deliberately does **not** pre-empt, since
the numbers that decide it do not exist until `s1h` does.
