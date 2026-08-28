# Measured pin-burnup wave — RESULTS (2026-08-20)

Scored strictly against the pre-registered rules in `data/reports/pinbu_wave_prereg_20260820.md` §6. Machine-readable twin: `pinbu_wave_results_20260820.json`.

## 1. Run integrity

| chains | converged | determinism ok | provenance ok | pin present | **usable** |
|---:|---:|---:|---:|---:|---:|
| 44/44 | 44 | 44 | 41 | 44 | **41** |

Dropped (no verdict, no statistic — a failed control is a refusal, never a downgraded result):

* `83befc5b4212` calib_N1N2_f113 — provenance promoted:MAS_RST.APRQ_20_0633.23 -> pair_feed:MAS_RST.APRQ_11_0677.23
* `0c1792ca0ee6` calib_K1K2_f109 — provenance promoted:MAS_RST.APRQ_20_0589.25 -> pair_feed:MAS_RST.APRQ_11_0633.21
* `e89822d9d675` calib_K1K2_f109 — provenance promoted:MAS_RST.APRQ_20_0589.25 -> pair_feed:MAS_RST.APRQ_11_0633.21

Determinism control (measured − stored, converged chains):

| axis | n | max abs delta | tolerance |
|---|---:|---:|---:|
| f_r | 44 | 0 | 0.002 |
| cyclen | 44 | 0.008 | 0.5 |
| cbc_max | 44 | 0.01 | 2.0 |
| f_q | 44 | 0.0001 | — |
| ao_abs | 44 | 0 | — |
| max_assembly_burnup | 44 | 0.001 | — |

## 2. Delivery verdicts (LEU+ limit 80.0 GWd/tU)

> **정의 각주 (2026-08-20 사용자 확정)**: 핀연소도 한계치 **80 GWd/tU** 는
> **핀 axial peak** — 즉 우리가 측정/예측해 온 `max_pin_burnup`(3-D 핀 노드
> 첨두) — 에 건다. 봉평균은 보조 관측량이며 판정축이 아니다. 아래 판정은
> 공식 관측량 그대로이므로 **유효하다**. `data/reports/pinbu_definition_20260820.md` · `pinbu_audit_20260820.md` §8.

| group | predicted | **measured PASS/n** | winner measured pin | winner verdict | best measured pin | ≤78 |
|---|---|---:|---:|---|---:|---:|
| `N1N2_f113` | FAIL | **0/5** | 86.19 | **FAIL** | 85.26 | 0/5 |
| `E1E2_f109` | FAIL | **0/5** | 82.11 | **FAIL** | 81.76 | 0/5 |
| `HGD569_f125_2type` | PASS | **5/5** | 75.47 | **PASS** | 75.47 | 5/5 |
| `HGD569_f125_3type` | PASS | **5/5** | 74.38 | **PASS** | 74.16 | 5/5 |

### N1N2_f113

| rank | record | stored F_r | predicted pin | **measured pin** | pred − meas | verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `c9bc21e9513b` | 1.4961 | 86.10 | **86.189** | -0.09 | FAIL |
| 2 | `ae1e2004b018` | 1.5006 | 87.20 | **85.260** | +1.94 | FAIL |
| 3 | `e0aad53068db` | 1.5013 | 86.06 | **85.296** | +0.77 | FAIL |
| 4 | `01e0d680c407` | 1.5061 | 85.86 | **85.439** | +0.42 | FAIL |
| 5 | `6c5f917021d7` | 1.5079 | 85.90 | **86.778** | -0.87 | FAIL |

### E1E2_f109

| rank | record | stored F_r | predicted pin | **measured pin** | pred − meas | verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `3153f3b39eff` | 1.4787 | 83.77 | **82.113** | +1.66 | FAIL |
| 2 | `2357edd0e2a4` | 1.4962 | 82.36 | **81.996** | +0.37 | FAIL |
| 3 | `4dba870bae69` | 1.5042 | 82.51 | **81.757** | +0.75 | FAIL |
| 4 | `5370d57034ec` | 1.5113 | 82.50 | **82.373** | +0.13 | FAIL |
| 5 | `7887d05bc27d` | 1.5114 | 82.41 | **82.379** | +0.03 | FAIL |

### HGD569_f125_2type

| rank | record | stored F_r | predicted pin | **measured pin** | pred − meas | verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `8b9acbcda6c7` | 1.6357 | 75.09 | **75.471** | -0.38 | PASS |
| 2 | `0884517be247` | 1.6606 | 75.33 | **75.843** | -0.51 | PASS |
| 3 | `d9cf666de322` | 1.6625 | 75.99 | **76.494** | -0.50 | PASS |
| 4 | `9d28d004ced1` | 1.6644 | 75.06 | **75.980** | -0.92 | PASS |
| 5 | `3b9d9cc2e090` | 1.6733 | 75.59 | **75.722** | -0.13 | PASS |

### HGD569_f125_3type

| rank | record | stored F_r | predicted pin | **measured pin** | pred − meas | verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `852d233d5421` | 1.5993 | 74.98 | **74.378** | +0.61 | PASS |
| 2 | `ea5f105ef08a` | 1.6024 | 75.13 | **74.156** | +0.98 | PASS |
| 3 | `0ecd683ed940` | 1.6036 | 74.90 | **75.501** | -0.60 | PASS |
| 4 | `72895ec55e16` | 1.6096 | 74.99 | **75.480** | -0.49 | PASS |
| 5 | `66a9a1bcea74` | 1.6105 | 74.93 | **75.582** | -0.66 | PASS |

## 3. Pin-head accuracy (predicted − measured, GWd/tU)

| slice | n | bias | MAE | sd | 95% CI on bias |
|---|---:|---:|---:|---:|---|
| **POOLED** | 41 | -0.32 | 1.02 | 1.38 | [-0.74, +0.08] |
| feed 109 | 14 | +0.25 | 0.84 | 1.31 | [-0.43, +0.91] |
| feed 113 | 17 | -0.83 | 1.44 | 1.61 | [-1.59, -0.13] |
| feed 125 | 10 | -0.26 | 0.58 | 0.60 | [-0.58, +0.12] |
| delivery | 20 | +0.12 | 0.64 | 0.80 | [-0.21, +0.48] |
| calibration | 21 | -0.75 | 1.39 | 1.67 | [-1.45, -0.08] |
| `E1E2_f109` | 5 | +0.59 | 0.59 | 0.66 | [+0.14, +1.15] |
| `HGD569_f125_2type` | 5 | -0.49 | 0.49 | 0.29 | [-0.73, -0.28] |
| `HGD569_f125_3type` | 5 | -0.03 | 0.67 | 0.77 | [-0.60, +0.59] |
| `N1N2_f113` | 5 | +0.43 | 0.82 | 1.04 | [-0.36, +1.30] |
| `calib_E1E2_f109` | 5 | +0.15 | 0.51 | 0.69 | [-0.39, +0.72] |
| `calib_G3G4_f109` | 1 | -2.80 | 2.80 | 0.00 | [+nan, +nan] |
| `calib_G3G4_f113` | 3 | -1.60 | 1.60 | 0.70 | [-2.39, -1.03] |
| `calib_K1K2_f109` | 1 | -0.35 | 0.35 | 0.00 | [+nan, +nan] |
| `calib_L1L2_f109` | 1 | +3.09 | 3.09 | 0.00 | [+nan, +nan] |
| `calib_L1L2_f113` | 3 | -2.07 | 2.07 | 1.02 | [-3.14, -1.11] |
| `calib_N1N2_f109` | 1 | -0.09 | 0.09 | 0.00 | [+nan, +nan] |
| `calib_N1N2_f113` | 6 | -0.89 | 1.55 | 1.97 | [-2.40, +0.42] |

> **E5 정정 (2026-08-20, `pinbu_audit_20260820.md` §4.3–4.4/E5)**: 위 **feed 113 슬라이스
> (bias −0.83 / MAE 1.44, n=17)** 는 **일반화 추정치가 아니다** — 17기 중 **14기가
> S1i train fold 안**이고 17/17 이 학습코어로부터 Hamming 4–54(중앙 7) 안에 있어
> 사실상 in-sample 재현이다. 같은 셀에서 학습 fold 근접 코어가 **0개**인 basin 을
> 고르자 같은 헤드가 **−5.93**(MAE 5.93, n=5)로 무너졌다
> (`f113_pin_results_20260820.md` §10.2). 이 표의 슬라이스 통계를 납품 영역으로
> 외삽하지 말 것.

Registered prior — the labels the store already held at these cells (all F_r ≥ 1.70, i.e. inside the training support and outside the operating region), re-scored with the same champion:

| cell | n | bias | MAE | measured span | min F_r |
|---|---:|---:|---:|---|---:|
| E1_E2/f109 | 33 | -1.29 | 1.59 | 76.1–93.6 | 1.80 |
| G3_G4/f109 | 50 | -0.91 | 1.32 | 81.3–102.5 | 1.83 |
| L1_L2/f109 | 34 | -1.22 | 1.63 | 75.0–96.5 | 1.73 |
| N1_N2/f109 | 40 | -0.62 | 1.33 | 81.6–97.3 | 1.70 |
| **pooled prior** | 157 | **-0.98** | 1.45 | | |

## 4. Registered hypothesis test → **H2**

Calibration-set bias 95% CI: **[-1.45, -0.08]** against H1 = +9 (head bias) and H2 = -1 ± 2 (pool deficit).

the head is ~unbiased here; the low-feed map closure is a search/design result, and mesh_multitype 5.1's pessimism claim is a cross-core artefact

> **E5 단서 (2026-08-20)**: H2 채택 자체는 유지되나 **적용 범위가 좁다**. 이
> 검정의 교정집합은 대부분 학습 지지영역 안에 있었고(위 §3 각주), H1/H2 이분법은
> `B_asm`·`ratio` **2요소 분해**를 누락한다(감사 §4.2 — 붕괴한 쪽은 `B_asm`
> 다리 −4.31, `ratio` 다리 −1.93). `mesh_multitype` §5.1 의 pessimism 주장은
> **철회**되었으며, 그 재서술은 cross-core artefact 가 아니라 **관측량 대응 오류
> (DB rodavg ↔ 우리 node) + 저급전 설계결손**이다 — 같은 README §5.1 참조.

## 5. Calibration curve

`measured = 1.0320 × predicted -1.953` (n=21, r=0.933, residual sd 1.70), fitted over predicted 77.3–92.6 → measured 76.5–93.7. Outside that predicted span the curve is not claimed.

> **E5b (2026-08-20)**: 이 교정곡선의 21기 역시 대부분 학습 지지영역 안에서
> 얻어졌다. **납품 영역(F_r ≤ 1.55, feed 113)으로 외삽 금지** — 그 영역에서 같은
> 헤드의 실측 오차는 −5.93 이다(`pinbu_audit_20260820.md` §4.4).

## 6. Recalibration recommendation

Pooled bias **-0.32** GWd/tU against the 2.0 trigger → **NOT triggered**.

LEAVE THE SERVE PATH ALONE. The measured |bias| is inside the 2.0 GWd/tU margin the decks already spend (minfr_pin_bu_limit 78 = 80-2); record the measured MAE as that margin's empirical basis.

> **E4 정정 (2026-08-20, `pinbu_audit_20260820.md` §4.5/§8.4)**: 서브 경로 자체는 **무결**이
> 코드 수준에서 확인되었다(핀 열 전 항목 동일, 정적 s1i 로 3자리 재현). 그러나 이
> 권고의 근거인 pooled bias **−0.32 는 41기 중 32기가 S1i train fold 안**에서 나온
> 값이므로 **납품 영역의 일반화 근거로 쓸 수 없다**. `minfr_pin_bu_limit=78` 이
> 예측-A 대 공식 한도 A=80 의 2.0 마진이라는 점은 확정 정의 아래에서 옳지만
> (`data/reports/pinbu_definition_20260820.md` §5), 그 2.0 은 **OOD 과소예측(신 basin 최대 −6)** 을 흡수하지
> 못한다. 납품 후보는 `pinbu_wave.py` **실측 검증**을 선행조건으로 건다.


---

*Sections 7–9 are the operator record, appended after the automated readout
(`pinbu_analyze.py` regenerates §1–6 only).*

## 7. The three refused chains

All three reproduced their stored `f_r` / `cyclen` / `cbc_max` **exactly**
(`determinism_ok = True`) but resolved a different restart than the store row
records: their campaigns ran from a **promoted elite** restart that this kit's
`data/produce/promoted` does not carry, so the resolver fell back to the
package's `pair_feed` restart. The pre-registered rule (§3) refuses them, and
they are refused — their pin values are **not** in the store.

| record | cell | measured pin | note |
|---|---|---:|---|
| `83befc5b4212` | N1_N2 / f113 | 91.296 | `promoted:…_20_0633.23` → `pair_feed:…_11_0677.23` |
| `0c1792ca0ee6` | K1_K2 / f109 | 88.346 | `promoted:…_20_0589.25` → `pair_feed:…_11_0633.21` |
| `e89822d9d675` | K1_K2 / f109 | 90.257 | `promoted:…_20_0589.25` → `pair_feed:…_11_0633.21` |

Worth recording as a **side finding, not a result**: reaching the identical
equilibrium (to 0.000 on F_r) from a *different* seed restart is direct evidence
that the equilibrium is restart-independent in these cells. That would justify a
future amendment relaxing the provenance control — but an amendment needs its own
pre-registration, so these three stay refused here.

## 8. Store merge

`python pinbu_wave.py patch --results runs/pinbu_wave/pinbu_wave_results.jsonl --tag pinbu_20260820`

* backup `data/store/records.parquet.bak_pre_pinbu_20260820`
* **41** rows patched in place by `record_id`; 74,597 rows, row order preserved
* only `max_pin_burnup` changed (`max_assembly_burnup` was already stored and the
  re-measurement agreed to ≤0.001); **no other column touched** — verified by
  column-wise comparison against the backup
* `max_pin_burnup` non-null 40,767 → 40,808

These are the programme's **first measured pin-burnup labels on its own campaign
cores**, and the first at feed 113 anywhere in the store (0 → 17 rows).

Dataset-P fit pool for `pinbu_physics` / the pin head, after the merge:

| feed | 101 | 109 | **113** | 117 | 125 | 133 | 141 | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rows | 129 | 462 | **17** | 749 | 310 | 150 | 137 | 1,954 |

(1,794 ga80 + 160 paramA. Dataset A's 38,854 pin labels remain censored by
default — `dataset_torch.censor_dataset_a_pin_labels` — so this ~1.95k pool is
the head's entire teacher.)

## 9. Retrain handoff — s1j (RECOMMENDED, NOT LAUNCHED)

**Readiness: ready, but thin. This wave does not by itself justify a retrain.**

What has accrued since the `S1i` split froze (2026-08-17 22:28):

| increment | size | note |
|---|---:|---|
| new rows (`fpcamp_minfr_hgd569_f125_seedctl`) | 60 (48 converged) | the only new rows |
| new `max_pin_burnup` labels on existing rows | 41 | +2.1% of the pin teacher pool |
| — of those, at a feed with no prior coverage (113) | 17 | 0 → 17 |

The 60 new rows are ~0.08% of the 74,537-row split — far below anything that has
moved a champion before. The pin labels are the substantive part, and they are
targeted (a previously empty feed), but they are 2% of an already-small pool.

**Recommendation: do NOT retrain on this increment alone.** Fold it into the next
retrain that has an independent reason to run. Two things make that case
stronger, and neither needs a champion:

1. **`max_pin_burnup` is still gate-advisory** (`config.gate_advisory_targets`,
   from `pinbu_forensics.md` §A2) because its within-cell held-out rank skill was
   ~0. This wave says nothing about *rank* — it measured 41 cores at 41 different
   predicted values and found the *magnitude* calibrated (§3). Promoting the axis
   out of advisory still requires the within-cell rank evidence it has never had.
2. **`pinbu_physics` is unfitted on every champion s1c…s1i** — no
   `pinbu_physics.json` exists outside `data/models/20260719_084819`, so the
   served pin column is the raw head. §6 says leave it alone: fitting an affine
   correction to a measured bias of −0.32 GWd/tU would be fitting noise.

If a retrain is run for other reasons, the recipe is unchanged from s1i:

```bash
# 1. grow the split (never regenerate — see build_split_S1b.py header)
python build_split_S1b.py --parent S1i --name S1j \
    --holdout-new-campaigns --self-check --write

# 2. train on the GPU box (238, GPU 1), s1i's run.sh verbatim with --split S1j
CUDA_VISIBLE_DEVICES=1 python -m lpopt.model.train --ensemble 5 --split S1j \
  --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 --epochs 150 \
  --num-workers 8 --device auto --parallel-members 5 --base-seed 20260716 \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 \
  --map-peak-weight 2.0 --cyclen-physics-prior --quantile-heads \
  --quantile-weight 0.2 --promote-max-asm-bu \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 --out-dir runs/s1j
```

The store snapshot to ship to the GPU box is the post-merge
`data/store/records.parquet` (74,597 rows, `max_pin_burnup` non-null 40,808).
