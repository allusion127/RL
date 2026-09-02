# `s1j` — 11th champion (promoted 2026-08-30)

| | |
|---|---|
| promoted | **2026-08-30** |
| former path | `data/models/20260829_194532` (pointer left at `data/models/20260829_194532.PROMOTED_TO_s1j.txt`) |
| predecessor | `data/models/s1i` (10th champion, promoted 2026-08-17) |
| arm | **arm 3 — `--fxy-direct`** (`--init-from data/models/s1i --freeze-trunk-cyclen`) |
| schema | `cond_schema = v8` (unchanged from s1i — decks' `cond_schema` key does NOT move) |
| what is new | a **direct `f_xy` head** (9th target, `fxy_head.mode = "direct"`, `target_idx = 8`, `select_weight = 0.5`), served OUTSIDE the frozen 7-column surrogate contract via `predict_fxy` |

Binding pre-registration: `data/reports/fxy_head_prereg_20260829.md` **Amendment C**.
Verdict document: `data/reports/fxy_head_results_arm3_20260829.md`.

---

## 1. Verdict (results report §1)

| gate | bar | measured | verdict |
|---|---|---:|---|
| **G1** legacy no-regression | `pass == true`, `blind_targets == []` | worst enforced drop **0.011364** vs ε **0.1388** (N=108) / **0.1422** (N=144); 36 cells / 144 checks; `blind_targets = []`, `unavailable = []` | **PASS** |
| **G2′** MAE(f_xy), n=793 | < **0.0767** | **0.066300** (bias −0.003137, resid sd 0.099290) | **PASS** |
| **G3′** within-cell ρ̄, 11 cells | > **0.7263** | **0.790392** (Δ +0.064092) | **PASS** |
| **G4** 68% coverage | ∈ [0.55, 0.80] | **0.831021** (σ̄ 0.119787) | **FAIL — over-wide** |

Disposition **C.4**: G1·G2′·G3′ PASS → promote to `s1j`. G4 FAIL does **not** change the
disposition but **bars the head σ from serving** (§4 / B.5 / C.4).

Companion readings (not verdicts, both refit on the FIXED serve path):
`PROXY on s1i` MAE **0.073173**, ρ̄ **0.715696** — arm 3 beats the fixed bars AND the
re-measured proxy, so C.3 #2's "split reading" clause never fired.

## 2. σ BAR (G4) — what is enforced, and where

**The f_xy head's σ is never served.** `min_fxy` ranks on `cyclen_LCB − λ·F_xy_UCB`, so a
σ that is 83.1% -wide (vs the 68.3% nominal) would silently widen every UCB. §10.2 of the
results report left the mechanism to the orchestrator and named the conservative default:
**keep the existing proxy σ convention (`resid_sd 0.0476`, `K 3.0`)**. That is what is
implemented, minimally:

* `ensemble.json` of this dir carries `fxy_head.serve_sigma = "barred"` (with `reason`,
  `bar` and `verdict` fields).  Member `meta.json` files were **NOT touched** — their
  sha256 are the graded provenance of the results report §12.
* `PosValCnnBackend.from_dir` reads that flag (also honoured if a member meta ever carries
  it) and exposes `backend.fxy_sigma_barred`.
* `lpopt.search.acquisition.predict_fxy` keeps the head's **mean** and `source = "head"`
  but replaces σ with the proxy's inflated σ (`fxy_proxy`, i.e.
  `sqrt((a·σ_Fr)² + (K·resid_sd)²)`, `a = 1.2176`, `K·resid_sd = 3.0 · 0.0476`).
* `lpopt.search.acquisition.fxy_conformal_upper` returns `None` while the bar is set, which
  is its documented "keep the proxy sigma" path.
* Unit tests: `tests/test_fxy_head.py::test_barred_head_sigma_serves_the_proxy_sigma`
  and siblings.

Lifting the bar requires a new pre-registration and a coverage re-measurement — not a
config edit.

## 3. Provenance

### 3.1 Contents (all moved by the rename, nothing re-copied)

`member_20260716 … member_20260720` (5 × `meta.json` + `model.pt` + `val_pred.npz`),
`DONE`, `rc = 0`, `heartbeat`, `train.log`, `run.sh`, `ensemble.json`,
`calibration.json`, `cyclen_physics_prior.json`, `power_prior.json`, the 6 per-cell
calibrations below and their 6 `.bak_pre_servefix_20260829` backups.

| file | sha256 |
|---|---|
| `member_20260716/meta.json` | `f0af69c0f54261dec61f253e3828fdc5df742f0915440678c885b96bb4112e7b` |
| `calibration.json` (σ isotonic + Platt; **not** refit — never touches the serve path) | `ce92ab906952b6bd389e0ae9ba3062d03592d6492ece2371cbe2be846fda50e4` |
| `ensemble.json` (**edited at promotion** — carries the σ bar of §2; was `861ab433a30f8e9bd81d26bb9b953509662af86f1485b78350339c3e931a39e1`) | `75cdc81874f8ad3b972c3d8a5ab210c7f973b1743299b2aaa4e533aa0f36c8ac` |

Member `meta.json` files were deliberately left untouched: `member_20260716/meta.json`'s
sha256 above is the graded provenance of the results report §12, and re-writing it to
carry the σ bar would have broken that chain. Editing `ensemble.json` instead has a
second, wanted effect — `ensemble.json` IS part of `remote.checkpoint_fingerprint`, so
the HOST_238 mirror will now actually re-push this dir instead of silently skipping it
(the per-cell calibrations alone would not have moved the fingerprint; servefix §6.1).

### 3.2 per-cell calibrations — REFIT on the FIXED serve path (2026-08-29/30)

Execution record: `data/reports/servefix_calibration_refit_20260829.md` (§3, §5).
Both this dir and `s1i` were refit, which is what dissolved the C.3 #3 asymmetry
(results report §9.1).

| file | sha256 |
|---|---|
| `cell_calibration.json` (cyclen — **copied from `s1i`** per the `--freeze-trunk-cyclen` parity rule) | `1f4b741583d3977ed92d7eb859ecdf7128f62982a85323de53df5f21b441e90c` |
| `f_r_calibration.json` | `d4e963304dc3dcf85d662c73a34d29229b3cf28bd660bbd6c545f67c68f2c9b7` |
| `cbc_calibration.json` | `9152620559d7aaa4274b0e30c24fb1330843fdf514b047f786a1cf40e081d20d` |
| `f_q_calibration.json` | `babd655d2163019cc0a60b6a1369f6905aa94629f3425f696c10f8667d85be50` |
| `ao_abs_calibration.json` | `b3991af39a3d7866a04193dd424e8531feb0ffb0db40cac90e6414e69ae18ce9` |
| `flatness_calibration.json` | `c91870dd4aa54ac05f0e9a2d8a347d32f7860380dd1f85ed2c82c18bcd68bed1` |

**R1 (servefix §6, binding): code and calibrations move as ONE set.** Shipping this dir
without the serve-path code — or the reverse — is forbidden.  What must travel with it,
as of 2026-08-30:

| file | sha256 | note |
|---|---|---|
| `lpopt/model/featurize.py` | `6977344dafbd770c9b1bc40e370db6c189320e301f8fa49570a25f927b575e36` | UNCHANGED (the FIXED serve path the gate was scored on) |
| `lpopt/model/model_api.py` | `5a713eaac47284a22493f93c7bdb08163d8ea3c444188b6b89f8c4b8849a39e6` | was `94229de9…` at scoring time; the ONLY delta is the σ-bar reader of §2 — featurization, calibration and every predict path are untouched, and `tests/test_model_api.py::test_serve_row_featurization_parity` still passes |
| `lpopt/search/acquisition.py` | `f1100d1fff7fb52f18e955c0e7d4a6d284ca0e4923f3d52514abd7b194a6b306` | σ-bar enforcement in `predict_fxy` / `fxy_conformal_upper` |
| `lpopt/search/campaign.py` | `a7d0caf4653df8f3d8a8e25ab32e16481fb3d3d6eb3d4fa025a1dfc15b97beec` | the `[F_xy SIGMA BARRED]` readout banner |

`remote.ensure_checkpoint` does **not** hash the per-cell calibrations, so a HOST_238
mirror will silently keep stale ones unless the remote `FINGERPRINT` is deleted — but
this promotion also edited `ensemble.json`, which IS in the fingerprint, so this
particular dir will re-push.

### 3.3 Gate / scoring artifacts

| item | sha256 |
|---|---|
| `data/reports/gate_fxy_arm3_20260829_checkonly.json` (G1) | `371f0e5a589023d5156da0954a54825375fb41e39f9651b0875f0b2e54be948f` |
| `data/reports/fxy_gate_eval_arm3_20260829.json` (G2′/G3′/G4) | `f70136d11f17097c281e86024acd31fe4622ff089178c0728659d8ff3f578a4e` |
| `data/reports/fxy_gate_eval_arm3_20260829.py` (scorer) | `c191141a2325e0f068cba3ea1dee119e49aa829ace594d318e064b740ddbbc12` |
| `data/store/records.parquet` at scoring time (75,793 rows) | `f38666e9f1508d35d33e0c22f583c5479c6f09cac748201b494b47c8cfeca6ea` |

## 4. What promotion did NOT do

* `fit_map_calibration` was **not** re-fit — `data/store/map_calibration.json` still carries
  `fit.model_id = "split_S1b"` / `2026-08-10T12:37` / `n_cells_fitted = 2`. The lineage
  skipped it at s1h and s1i too, so §10.1 #9 makes it a recorded staleness (4 generations),
  not an obligation.
* The A.5 / A.6 bars (0.0463 / 0.8944) were **not** met and are **not** retired — they were
  never the promotion condition (C.4). See results report §1.
* No claim that "direct removed the bias" or that arm 3 beats arm 2: the paired
  cell-clustered BCa CI [−0.004133, +0.015511] contains 0 — a **tie** (results §8, §13).
* GPU deviation stays an open record: this dir was trained with `CUDA_VISIBLE_DEVICES=1`
  (`run.sh`, deck `lpopt_gpu1.inp`) against `lpopt.inp`'s standing "GPU 0 only until
  re-authorised" instruction (results §10.4). It affects none of G1–G4.
