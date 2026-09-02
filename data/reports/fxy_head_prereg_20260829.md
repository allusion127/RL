# 사전등록(pre-registration) — `f_xy` prior-residual head (F_xy 전환 P4) — **FINAL**

- 작성일 2026-08-29 · 상태 **확정(FINAL)**: 수치·임계값·판정규칙은 학습 실행 **전에** 고정되었으며
  실행 후 변경하지 않는다. `fxy_head_prereg_20260829_DRAFT.md`(라벨 1,361행 시점)를 **supersede** 한다.
- 근거 설계서 `data/reports/fxy_switch_design_20260829.md` §3 (구속력) · 선례
  `ab2_preregistration_20260730.md`, `ab2_addendum_S1I_20260817.md` (게이트·ε·노이즈 보고 규칙 승계)
- 대상 코드: `lpopt/model/{physics_prior.py,net.py,train.py,al_retrain.py}` · 챔피언 `data/models/s1i`

---

## 1. 가설

- **H1 (1차, veto).** head를 붙여도 legacy 축이 **회귀하지 않는다** (`gate-promote` PASS).
- **H2 (1차).** head의 f_xy holdout 절대오차가 **train fold에서 적합된 선형 prior의 holdout 잔차
  산포보다 작다** — prior가 설명하지 못하는 성분을 실제로 학습했다.
- **H3 (1차).** 라벨이 충분한 cell 안에서 head의 **cell 내 순위**가 선형 prior의 순위 이상이다.
- **H4 (건전성).** head의 σ가 공허하지도 과대하지도 않다 (경험적 커버리지 sanity).

H1은 구조적으로 상당 부분 보장되어 있으나(§6) **측정으로 확인**한다 — "구조상 안전하므로
측정하지 않았다"는 본 프로젝트에서 허용되지 않는 논증이다.

---

## 2. 데이터 — 현재 store에서 재계산(2026-08-29)

`data/store/records.parquet` = **74,717행**, `f_xy` 보유 **6,236행**
(converged **6,218**, 미수렴 18). 아래 모든 통계는 **converged 6,218행** 기준.

| 항목 | 값 |
|---|---|
| `f_xy ≤ 1.65` | **1,093** / 6,218 (17.6%) |
| `F_r ≤ F_xy ≤ F_q` 성립 | **6,218 / 6,218 (100%)** |
| `F_xy / F_r` | mean **1.0800**, sd **0.0297** |
| (case_pair, feed) cell 수 | **119** |
| S1i **TRAIN** fold 라벨 `N_train` | **5,295** (≥ `MIN_FXY_LABELS` = 200 통과) |
| S1i **VAL**(honest holdout) 라벨 `N_hold` | **760** |
| **어느 fold에도 없는** 라벨 | **163** (§2.1) |
| holdout 라벨 ≥ 1 인 cell | **49** |
| holdout 라벨 **≥ 20** 인 cell (게이트 대상) | **9** (합계 432행) |

fold 판정은 `train.py` 경로 그대로: `_load_split` → `SplitManifest.record_ids(fold)` →
`dataset_torch.LPDataset._resolve_ids` (train_ids / val_ids 원본 리스트, cap 없음).

### 2.1 등록해 두는 불일치 — store가 S1i보다 앞서 있다

`S1i.json.groups.records_sha256` = `b55e108d…` (74,537행) ≠ 현재 store `cf495c7d…` (74,717행).
차이 180행 중 **163행이 f_xy 라벨을 가지며 train/val 어느 쪽에도 속하지 않는다**
(`fpcamp_minfr_N1N2_f113_pin` 59, `fpcamp_minfr_triple_f125_r2` 57, `fpcamp_minfr_hgd569_f125_seedctl` 47).
`_resolve_ids`가 manifest의 id만 쓰므로 이 163행은 **학습에도 평가에도 들어가지 않는다** — 조용히
버려지는 것이 아니라 여기에 등록해 둔다. 재분할 없이 진행하며, 위 `N_train`/`N_hold`가 실제 값이다.

> **누수 규칙.** prior 계수 `(a,b)`는 `train.resolve_fxy_prior` → `physics_prior.fit_fxy_prior`가
> **train fold 행만으로** 적합한다. holdout 행은 prior 적합에도 z-score 상수에도 들어가지 않는다.

### 2.2 선형 prior `f_xy = a·f_r + b` — 재계산

| 적합 집합 | a | b | r | resid sd | n |
|---|---:|---:|---:|---:|---:|
| **corpus 전체**(converged 라벨) | **1.2176** | **−0.2519** | 0.9895 | **0.0476** | 6,218 |
| **S1i TRAIN fold** ← 학습이 실제로 쓸 값 | **1.2148** | **−0.2459** | 0.9892 | **0.0481** | 5,295 |
| (참고) P4 보고서, 1,361행 | 1.2220 | −0.2650 | — | 0.0404 | 1,361 |

**TRAIN 적합 prior를 holdout(760행)에 적용한 잔차:**
sd **0.0466**, bias −0.0026, MAE **0.0357**, max|resid| **0.2942**.

per-cell TRAIN 적합(train 라벨 ≥ 20 인 71 cell): 기울기 `a` 0.5357 / 중앙값 1.2631 / 1.5263,
잔차 sd 0.0077 / 중앙값 0.0457 / 0.0683 (n-가중평균 0.0417), r 최소 0.3695 →
**per-cell 산포는 global affine이 설명하지 못하는 성분이며, 그것이 residual head의 몫이다.**

---

## 3. arm (실행 커맨드 — 확정)

`python -m lpopt.model.al_retrain --champion data/models/s1i --add-fxy-head --dry-run`
이 인쇄한 3단계를 그대로 실행한다.

```
[1] python -c "from lpopt.model.al_retrain import refresh_distill_cache; \
      refresh_distill_cache('data/models/s1i', out_path='data/models/_v5_distill_soft.npz')"
[2] python -m lpopt.remote --input lpopt.inp push
[3] python -m lpopt.remote --input lpopt.inp train -- \
      --ensemble 5 --split S1i --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 \
      --epochs 150 --num-workers 8 --device auto \
      --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
      --cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
      --promote-max-asm-bu --promote-fxy \
      --init-from data/models/s1i --freeze-trunk-cyclen \
      --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
      --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

실행 호스트 **HOST_238** (`HOST_238:8022`, `USER`), **GPU 1**.

> **등록해 두는 편차(승인 필요).** `lpopt.inp [remote] gpu = 0` 이며 주석은 *"사용자 지시 2026-07-24:
> GPU 0 고정 (GPU 1 사용 금지, 재허가 전까지 auto 금지)"* 다. GPU 1 실행은 그 상시 지시와 충돌한다 —
> deck의 `gpu`를 1로 바꾸기 **전에** 명시적 재허가를 받는다. 재허가가 없으면 GPU 0으로 돌리고
> 그 사실을 결과에 적는다. (본 문서는 deck을 수정하지 않는다.)

| 이름 | 정의 |
|---|---|
| `HEAD` | 본 실행의 f_xy head (prior + residual), `search.acquisition.predict_fxy`로 서빙 |
| `PRIOR` | 같은 실행의 prior 단독 (`a·F_r + b`, residual ≡ 0) — H2의 baseline |
| `PROXY` | 현행 대리변수 `FXY_PROXY_SLOPE·F_r + FXY_PROXY_INTERCEPT` = `1.1221·F_r − 0.0831` |
| `INCUMBENT` | 챔피언 `data/models/s1i` — 무회귀 비교 기준 |

`node_peak`은 대리 후보에서 **제외**(설계서 §1.2/§3.6: 실측 상관 0.735–0.854).
**target 계약 주의.** s1i의 `target_names`는 **8열** (`f_r, f_q, cbc_max, cyclen, ao_abs,
discharge_burnup, max_pin_burnup, max_assembly_burnup`)이다. "7 legacy target"은 vendor의 7열
surrogate 계약을 가리키며 실제 무회귀 대상은 그 7열 + `max_assembly_burnup` = **8 legacy 행**,
`--promote-fxy`가 `f_xy`를 **9번째로 append** 한다.

---

## 4. 게이트 (실행 전 확정)

**G1은 veto다. G1 실패 시 G2/G3의 결과와 무관하게 reject.**

### G1 (1차, veto) — legacy 무회귀, honest holdout

> `lpopt gate-promote --prev data/models/s1i --new data/models/<new_ts> --out gate_fxy.json --check-only`
> 의 `pass == true` (`curriculum.gate_no_regression` + `gate_legacy_tail` 모두).

- holdout = `S1i.json.groups.curriculum_val_by_cell` (3,914 id) — 양 챔피언 모두 학습에서 제외된 행.
- 채점 family `NOREG_TARGETS` = `cyclen`, `f_r`, `node_peak`, `map_cov`. 기본 강제는
  `cyclen`/`node_peak`/`map_cov`이나 **본 실행은 `f_r`도 강제한다** (f_xy prior가 `f_r` 예측을
  직접 읽는다) → deck에 `[curriculum] gate_noreg_fr_guard_enabled = true`.
- ε는 family-wise adaptive: `eps_N = σ0·Φ⁻¹(0.95^(1/N))`, `σ0 = _NOREG_SIGMA0 = 0.042`, 설정값이 floor.
  s1c–s1i 계보의 실측 ε는 **0.1388114093847** (36 cell × 3 강제축 = **N=108**). `f_r` 강제로
  N = 36 × 4 = **144** → 본 실행의 ε는 **0.14216194539159127**. N이 달라지면 ε도 공식대로 달라진다.
- `blind_targets`가 비어 있지 않으면 PASS로 읽지 않는다 (측정되지 않은 guard = 판정 불가).

### G2 (1차) — f_xy 절대오차

> 라벨된 S1i **VAL** 760행에서 `MAE(f_xy) < 0.0466`
> (= §2.2의 **TRAIN 적합 prior의 holdout 잔차 sd**).

- 산출: `lpopt.model.evaluate`의 per-target 표(`mae`) 및 학습 로그 `train.fxy_metrics`의 `mae_f_xy`.
- **부수 보고(게이트 아님, 사후 승격 금지):** 같은 prior의 holdout **MAE는 0.0357**이다.
  `MAE_HEAD < 0.0357`이면 H2의 강한 형태가 성립하고, `0.0357 ≤ MAE_HEAD < 0.0466`이면
  "prior의 산포는 이겼으나 평균오차는 이기지 못했다"로 **그렇게** 기록한다.
- DRAFT가 후보로 남겨둔 0.029(설계서, HOST_199 2 cell n=192)는 **채택하지 않는다** — 현재 실측
  산포는 0.0466–0.0481이다. 기준은 **본 실행이 실제로 적합할 prior의 holdout 산포**로 고정한다.

### G3 (1차) — cell 내 순위

> holdout 라벨 **≥ 20**인 **9개 cell**에서, `within_case_spearman`(min_case = 20, cell별 ρ의
> **비가중 평균**) 기준 `ρ_HEAD ≥ ρ_PRIOR`.

- cell 정의는 **(case_pair, feed)**. 선형 prior는 `f_r`의 단조증가 함수이므로
  `ρ_PRIOR ≡ ρ(f_r, f_xy)` (측정 `f_r` 사용 = prior/PROXY가 서빙에서 결코 받지 못하는 이상적 입력).
  따라서 **이 바는 보수적으로 높다** — 그래도 낮추지 않는다. **바(실측, 고정):**

| cell | holdout n | ρ_PRIOR | prior resid sd | prior MAE |
|---|---:|---:|---:|---:|
| `E1_E2/f109` | 29 | 0.9783 | 0.0381 | 0.0365 |
| `E1_E2/f117` | 43 | 0.9799 | 0.0544 | 0.0464 |
| `E1_E2/f121` | 40 | 0.7659 | 0.0359 | 0.0292 |
| `E1_E2/f125` | 52 | 0.9857 | 0.0576 | 0.0437 |
| `E3_E4/f125` | 21 | 0.9610 | 0.1006 | 0.0738 |
| `G3_G4/f125` | 36 | 0.9516 | 0.0389 | 0.0317 |
| `J5_J6/f121` | 44 | 0.9663 | 0.0287 | 0.0266 |
| `N1_N2/f113` | 20 | 0.9018 | 0.0244 | 0.0203 |
| `T6_T4/f121` | 147 | 0.9083 | 0.0322 | 0.0263 |
| **비가중 평균 (판정 기준)** | 432 | **0.9332** | 0.0456(pooled) | 0.0338 |
| (참고) n-가중 평균 | | 0.9280 | | |

- **판정: `ρ̄_HEAD ≥ 0.9332`** (9 cell 비가중 평균), 그리고 per-cell 표를 함께 인쇄한다.
- **동률/근소차:** 차이가 `0 ± 0.05` 안이면 점 추정으로 판정하지 않는다.
  `lpopt.model.ab_paired`의 cell-clustered paired BCa CI를 산출하고, CI가 0을 포함하면
  **tie**로 기록한다 — 승리로 읽지 않는다 (`ab_decide` 교훈: 점 비교 승격 = 지표 옷을 입은 tiebreak).
- **주의(사전 등록):** `train.fxy_metrics`의 `within_cell_spearman_f_xy`는 `case_pair` **단독**
  그룹핑(feed 무시)이므로 참고로만 읽고 판정은 위 표로 한다 (case_pair 단독 holdout ρ_PRIOR:
  11 cell 평균 0.9686, 전역 ρ 0.9908).

### G4 (건전성) — σ 커버리지

> 라벨된 holdout 760행에서 head의 **68% 구간 경험적 커버리지가 [0.55, 0.80]** 안에 들 것.

- 산출: `|f_xy_true − μ| ≤ σ` 의 비율 (calibration 적용 후의 σ). 범위 밖이면 head는 승격하되
  **σ는 서빙에서 사용 금지**(μ만 사용). 0.80 초과 = 공허하게 넓음, 0.55 미만 = 위험하게 좁음.

### 판정 표 (실행 후 채울 것)

| 게이트 | 지표 | 기준 | 측정값 | 판정 |
|---|---|---|---|---|
| G1 | `gate-promote --check-only` | `pass == true`, ε = 0.14216194539159127 (N=144), `blind_targets == []` | TBD | TBD |
| G2 | MAE(f_xy), holdout n=760 | `< 0.0466` | TBD | TBD |
| G2b | vs PRIOR MAE (보고) | `< 0.0357` 이면 H2 강한 형태 | TBD | TBD |
| G3 | ρ̄ within-cell, 9 cell | `≥ 0.9332` (tie 시 paired BCa CI) | TBD | TBD |
| G4 | 68% 커버리지 | `∈ [0.55, 0.80]` | TBD | TBD |

### 처분 규칙 (사전 확정)

- **PASS (G1·G2·G3 모두)** → 새 체크포인트를 **`s1j`** 로 승격, deck `model_dir` 갱신.
- **FAIL-G1** → **reject.** 재시도는 새 사전등록을 요구한다.
- **FAIL-G2만** / **FAIL-G3만** (G1 PASS) → 챔피언 **s1i 유지**, head는 **`shadow`** 로만 배포
  (`fxy_source = "head"`로 기록하되 acquisition 판정에는 쓰지 않는다).
- **G4 실패** → 위 처분을 바꾸지 않되 σ 사용 금지를 명시.

---

## 5. 동결 산출물 (sha256)

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `cf495c7d82b16cbfe4216333ca4d266a324514c223bd7e0a2c38f799445326cc` | 22,315,679 |
| `data/splits/S1i.json` | `87aa6564cd1f9fd009113a4941de9313ec417b9f953b0215781b833a08b58b82` | 6,008,432 |
| `data/models/s1i/member_20260716/meta.json` | `32bfb282370b16f7827c75966645a2fd4796cb3800f495850a7201bfc4fb5ec5` | 43,197 |

> `s1i`에는 `member_0/`가 없다 — 멤버 디렉터리는 seed 이름(`member_20260716` … `member_20260720`)이고
> 위는 첫 멤버(seed 20260716)다.

---

## 6. 왜 G1이 통과할 것으로 기대하는가 (그리고 왜 그래도 측정하는가)

레시피는 freeze-finetune이다: `--init-from data/models/s1i --freeze-trunk-cyclen --promote-fxy`.
(1) trunk(`stem`/`blocks`/`films`/`head_trunk`/`conv_head`)는 `requires_grad=False`이고 optimizer에서
제외된다 — decoupled weight decay조차 건드리지 못한다. (2) `mu_head`/`log_sigma_head`의 **cyclen 행**은
backward hook으로 grad가 0이 된다. (3) f_xy 행이 읽는 `f_r` 예측은 `net.PosValNet._compose_fxy`에서
`detach()` 되어 있어 f_xy loss의 gradient는 **F_r head로 흐르지 않는다**. (4) 챔피언 가중치는
`train._graft_appended_target_rows`가 prefix로 이식하고 나머지 key는 `strict=True`로 적재된다(8열→9열).

따라서 legacy 축이 움직일 수 있는 경로는 `mu_head`/`log_sigma_head`의 **cyclen 아닌 legacy 행**
뿐이며, 이들도 f_xy loss로부터는 gradient를 받지 않는다. 그럼에도 이 실행은 distillation 항과
`--f-r-rank-weight 0.1`을 함께 켜므로 legacy 행은 여전히 학습된다 — **그러므로 G1은 형식이 아니라 실측이다.**

---

## 7. 2차 결정 — proxy vs head, 그리고 proxy 상수 갱신 권고

search 계층은 `has_fxy_head()`가 거짓이면 `acquisition.fxy_proxy`로 **fallback** 한다. 현행 상수
(`lpopt/search/acquisition.py`)는 192 core / 2 cell 시절 값이다: `FXY_PROXY_SLOPE = 1.1221`,
`FXY_PROXY_INTERCEPT = -0.0831`, `FXY_PROXY_RESID_SD = 0.0293`, `FXY_PROXY_SIGMA_K = 3.0`. 6,218행 재적합:

| 계수 | bias | resid sd | MAE | p95\|resid\| | max\|resid\| | @1.65 오분류 (feasible/infeasible) |
|---|---:|---:|---:|---:|---:|---|
| 현행 `1.1221 / −0.0831` | **+0.0103** | 0.0543 | 0.0385 | 0.1029 | 0.3743 | 109 / 150 |
| **corpus 재적합 `1.2176 / −0.2519`** | −0.0000 | **0.0476** | **0.0365** | **0.0803** | 0.3125 | 188 / 69 |
| P4 시점 `1.2220 / −0.2650` (n=1,361) | +0.0050 | 0.0476 | 0.0368 | 0.0829 | 0.3151 | 226 / 56 |

**권고 (search owner에게 후속 작업으로 이관 — 본 문서는 코드를 수정하지 않는다):**

1. `FXY_PROXY_SLOPE → 1.2176`, `FXY_PROXY_INTERCEPT → -0.2519` (6,218 core / 119 cell 재적합;
   편향 +0.0103 → 0.0000, p95 잔차 0.1029 → 0.0803).
2. `FXY_PROXY_RESID_SD → 0.0476`. 현행 0.0293은 **실측 산포의 62%에 불과**하며 `SIGMA_K = 3.0`
   인플레이션조차 그 과소평가 위에 얹혀 있다. `K`는 3.0 유지 (max 잔차 0.31 — 낙관 금지).
3. 위 3개는 **한 커밋으로**. 기울기만 올리고 sd를 두면 `@1.65` 오분류가
   false-infeasible(150) → false-feasible(188)로 **방향이 바뀐다** — 넓힌 σ가 상쇄해야 한다.
4. head 승격(PASS) 시에도 proxy는 fallback 경로로 남고, `shadow` 처분에서는 **유일한** 경로다.

부수 발견(기록만): `lpopt.inp [model] cond_schema = "v7"` 이지만 `data/models/s1i/*/meta.json`은
**v8**이고 deck 주석은 s1h를 서술한다 — deck이 stale. deck owner 후속 확인 사항.

---

## 8. 사전에 금지하는 주장 (설계서 §3.6 승계)

- head의 f_xy 예측으로 **미측정 core의 `F_xy ≤ 1.65` 적합성을 선언**하는 것 — head는 acquisition의
  신호이지 licensing 판정이 아니다. 인도 판정은 MASTER 측정값으로만 한다.
- **prior 단독**으로 feasibility를 말하는 것 (max 잔차 0.29 > 게이트 폭).
- `F_r ≤ 1.55`가 `F_xy ≤ 1.65`를 함의한다는 주장 / MOCHA의 "Fxy ≤ 1.55 표본 1개" 통계 인용.
- §2.1의 163행(어느 fold에도 없음)을 근거로 한 어떤 성능 주장도 금지.

## 9. 미도입을 사전에 기록해 두는 것 (P4 범위 밖)

- **f_xy conformal 구간**: `conformal.CONFORMAL_TARGETS`는 7열 index로 키를 잡으며 f_xy 열이 없다.
  1차에서는 구간을 **내지 않는다** — 공허한 구간보다 없는 편이 정직하다.
- **f_xy per-cell calibration**: holdout 라벨 ≥ 20 cell이 9개뿐 → P4에서 도입하지 않는다.
- **7열 surrogate 계약 확장**: f_xy는 `predict_fxy`로 계약 **바깥에서** 서빙된다.
- **S1j 재분할**: §2.1의 163행을 흡수하는 재분할은 본 실행 범위 밖(분할이 바뀌면 위 바가 전부 무효).

---

## 10. labels snapshot — cell별 (case_pair, feed) · converged 라벨만

| cell | train | holdout(VAL) | 미분류 | 합계 |
|---|---:|---:|---:|---:|
| `T6_T4/f121` | 591 | 147 | 0 | 738 |
| `E1_E2/f121` | 614 | 40 | 0 | 654 |
| `E1_E2/f125` | 220 | 52 | 0 | 272 |
| `E1_E2/f117` | 194 | 43 | 0 | 237 |
| `J5_J6/f121` | 178 | 44 | 0 | 222 |
| `G3_G4/f125` | 167 | 36 | 0 | 203 |
| `N1_N2/f113` | 93 | 20 | 59 | 172 |
| `E3_E4/f121` | 166 | 0 | 0 | 166 |
| `E1_E2/f109` | 127 | 29 | 0 | 156 |
| `N3_N4/f121` | 135 | 0 | 0 | 135 |
| `H1_H2/f121` | 135 | 0 | 0 | 135 |
| `N1_N2/f121` | 126 | 0 | 0 | 126 |
| `H3_H4/f121` | 125 | 0 | 0 | 125 |
| `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24/f125` | 39 | 10 | 57 | 106 |
| `P6253Z1G06N24_P6253Z2G10N24/f125` | 46 | 11 | 47 | 104 |
| `E3_E4/f125` | 63 | 21 | 0 | 84 |
| **나머지 103 cell** | 2,276 | 307 | 0 | 2,583 |
| **합계 (119 cell)** | **5,295** | **760** | **163** | **6,218** |

> DRAFT §2가 경고한 "단일 cell 집중"은 해소되었다: 최대 cell이 전체의 11.9%, holdout 라벨 ≥ 20
> cell이 9개이므로 §4의 G3는 **판정 가능(conclusive)** 하다.
---

# Amendment A — 2026-08-29 · 분할 `S1i` → **`S1j`** (실행 전 개정)

본 절은 §1–§10을 **재작성하지 않는다.** 아래 항목만 **덮어쓰며(binding)**, 언급되지 않은
모든 조항(H1–H4, G1의 형식, G4, 처분 규칙, §8 금지 주장)은 원문 그대로 유효하다.
개정 시점은 **학습 실행 전**이며, 사후 조정이 아니다.

## A.1 왜 개정하는가 — §2.1의 163행을 흡수한다

§2.1은 store(74,717행)가 `S1i`(74,537행)보다 앞서 있어 **f_xy 라벨 163행이 train/val 어느
fold에도 속하지 않는다**고 등록해 두었고, §9는 재분할을 "본 실행 범위 밖"으로 미뤄 두었다.
그 재분할을 **실행 전에** 수행했다. 따라서 §2.1의 caveat과 §8의 "163행 근거 주장 금지" 조항은
**해소(resolved)** 되었다 — 더 이상 버려지는 라벨이 없다. §9의 "S1j 재분할 미도입" 항목은
본 개정으로 **supersede** 된다.

## A.2 새 분할 `S1j`

```
python build_split_S1b.py --parent S1i --name S1j --holdout-new-campaigns --write
```

| 항목 | 값 |
|---|---|
| 파일 | `data/splits/S1j.json` |
| sha256 | `321950cb8fc965118569b5afa7943f2f2b09b39b231da228a177eac86e7ba3b1` |
| bytes | 6,033,916 |
| `groups.derived_from_split` | `S1i` (`87aa6564…`) |
| `groups.records_sha256` | `cf495c7d82b16cbfe4216333ca4d266a324514c223bd7e0a2c38f799445326cc` — **현재 store와 일치** |
| 총 행 | **74,717** = train **62,575** + val **12,142** (합이 store 전체와 정확히 같다) |
| f_xy 라벨(converged) | **6,218** = train **5,425** + val **793** + **어느 fold에도 없음 0** |
| holdout 라벨 ≥ 1 cell | 49 |
| holdout 라벨 ≥ 20 cell (G3 대상) | **11** (합계 486행) |
| `groups.curriculum_val_by_cell` | 3,948 id / 87 cell |

§5의 동결 산출물 표에서 `data/splits/S1i.json` 행은 위 `S1j` 행으로 대체된다.
`data/store/records.parquet`(`cf495c7d…`)와 `data/models/s1i/member_20260716/meta.json`
(`32bfb282…`)는 변경 없다.

## A.3 arm — 실행 커맨드 (§3 [3]의 `--split`만 변경)

```
[3] python -m lpopt.remote --input lpopt.inp train -- \
      --ensemble 5 --split S1j --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 \
      --epochs 150 --num-workers 8 --device auto \
      --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
      --cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
      --promote-max-asm-bu --promote-fxy \
      --init-from data/models/s1i --freeze-trunk-cyclen \
      --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
      --distill-min-match-frac 0.5 --f-r-rank-weight 0.1
```

`[1]`·`[2]`는 §3 그대로다. 실행 호스트/GPU 조건(§3의 "등록해 두는 편차")도 그대로 유효하다.

## A.4 선형 prior 재적합 — **S1j TRAIN fold**

| 적합 집합 | a | b | r | resid sd | n |
|---|---:|---:|---:|---:|---:|
| corpus 전체(converged 라벨) | 1.2176 | −0.2519 | 0.9895 | 0.0476 | 6,218 |
| **S1j TRAIN fold** ← 학습이 실제로 쓸 값 | **1.2161** | **−0.2488** | 0.9894 | **0.0478** | 5,425 |
| (참고) S1i TRAIN fold — §2.2 | 1.2148 | −0.2459 | 0.9892 | 0.0481 | 5,295 |

**TRAIN 적합 prior를 holdout(793행)에 적용한 잔차:**
sd **0.0463**, bias **−0.0026**, MAE **0.0355**, max|resid| **0.2939**.

누수 규칙(§2.2 인용문)은 변경 없다: prior 계수는 train fold 행만으로 적합된다.

## A.5 **G2 (구속력 갱신)** — f_xy 절대오차

> 라벨된 S1j **VAL** **793행**에서 `MAE(f_xy) < 0.0463`
> (= A.4의 TRAIN 적합 prior의 holdout 잔차 sd).

- 부수 보고(게이트 아님, 사후 승격 금지): 같은 prior의 holdout **MAE는 0.0355**.
  `MAE_HEAD < 0.0355`이면 H2의 강한 형태, `0.0355 ≤ MAE_HEAD < 0.0463`이면
  "prior의 산포는 이겼으나 평균오차는 이기지 못했다"로 기록한다.
- §4 G2의 나머지 문구(산출 경로, 0.029 후보 불채택)는 그대로다. 기준은 S1i의 0.0466/0.0357에서
  **S1j의 0.0463/0.0355로** 대체된다.

## A.6 **G3 (구속력 갱신)** — cell 내 순위

> holdout 라벨 **≥ 20**인 **11개 cell**(합계 486행)에서, cell 정의 **(case_pair, feed)**,
> cell별 ρ의 **비가중 평균** 기준 **`ρ̄_HEAD ≥ 0.8944`**.

`ρ_PRIOR ≡ ρ(f_r, f_xy)`(측정 `f_r` 사용)라는 §4 G3의 정의와 보수성 논거는 그대로다.
잔차 sd / MAE는 A.4의 TRAIN 적합 prior 기준.

| cell | holdout n | ρ_PRIOR | prior resid sd | prior MAE |
|---|---:|---:|---:|---:|
| `E1_E2/f109` | 29 | 0.9783 | 0.0389 | 0.0367 |
| `E1_E2/f117` | 43 | 0.9799 | 0.0549 | 0.0463 |
| `E1_E2/f121` | 40 | 0.7659 | 0.0364 | 0.0293 |
| `E1_E2/f125` | 52 | 0.9857 | 0.0582 | 0.0437 |
| `E3_E4/f125` | 21 | 0.9610 | 0.1030 | 0.0737 |
| `G3_G4/f125` | 36 | 0.9516 | 0.0395 | 0.0316 |
| `J5_J6/f121` | 44 | 0.9663 | 0.0290 | 0.0271 |
| `N1_N2/f113` | 32 | 0.8730 | 0.0313 | 0.0248 |
| `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24/f125` | 22 | 0.5738 | 0.0307 | 0.0331 |
| `P6253Z1G06N24_P6253Z2G10N24/f125` | 20 | 0.8943 | 0.0306 | 0.0347 |
| `T6_T4/f121` | 147 | 0.9083 | 0.0324 | 0.0263 |
| **비가중 평균 (판정 기준)** | 486 | **0.8944** | 0.0451(pooled) | 0.0339 |
| (참고) n-가중 평균 | | 0.9081 | | |

> **바가 0.9332 → 0.8944로 내려간 이유를 명시해 둔다 (완화가 아니다).** 163행이 fold에
> 흡수되면서 (a) triple-type cell 2개가 holdout ≥ 20 문턱을 새로 넘었고
> (`…N20_…N24/f125` ρ 0.5738, `…N24/f125` ρ 0.8943), (b) `N1_N2/f113`의 holdout이
> 20 → 32행으로 늘며 ρ가 0.9018 → 0.8730으로 재측정되었다. 즉 **prior가 원래 약했던 영역이
> 이제 채점에 들어온 것**이며, 바는 그 영역을 포함한 실측값으로 다시 고정되었다.
> S1i의 9 cell만 따로 떼어 채점하는 사후 선택은 **금지한다.**

- 동률/근소차 규칙(±0.05 안이면 `ab_paired`의 cell-clustered paired BCa CI, CI가 0을 포함하면
  **tie**)과 `train.fxy_metrics`의 `case_pair` 단독 그룹핑 주의는 §4 G3 그대로다.

## A.7 G1 — ε는 S1j holdout에서 재산출한다

G1의 형식·veto 성격·`blind_targets == []` 조건은 §4 그대로다. 다만 holdout이 바뀌었으므로:

- holdout = `S1j.json.groups.curriculum_val_by_cell` (**3,948 id / 87 cell**).
- ε는 §4의 **동일한 공식** `eps_N = σ0·Φ⁻¹(0.95^(1/N))`, `σ0 = 0.042`로 산출한다. §4에 적힌
  `0.14216194539159127`은 **S1i holdout의 N=144에서 나온 값**이므로 본 실행에는 쓰지 않는다.
  게이트 실행이 보고하는 실제 `N`과 그로부터 나온 ε를 **판정 표에 그대로 기록**한다.
  (공식·σ0·강제 축 집합 — `cyclen`/`node_peak`/`map_cov` + 본 실행이 추가로 강제하는 `f_r` —
  은 변경되지 않는다. 바뀐 것은 채점되는 cell 수뿐이다.)

## A.8 판정 표 (개정판, 실행 후 채울 것)

| 게이트 | 지표 | 기준 | 측정값 | 판정 |
|---|---|---|---|---|
| G1 | `gate-promote --check-only` | `pass == true`, ε = σ0·Φ⁻¹(0.95^(1/N)) (N·ε 기록), `blind_targets == []` | TBD | TBD |
| G2 | MAE(f_xy), holdout n=793 | `< 0.0463` | TBD | TBD |
| G2b | vs PRIOR MAE (보고) | `< 0.0355` 이면 H2 강한 형태 | TBD | TBD |
| G3 | ρ̄ within-cell, 11 cell | `≥ 0.8944` (tie 시 paired BCa CI) | TBD | TBD |
| G4 | 68% 커버리지 | `∈ [0.55, 0.80]` | TBD | TBD |

## A.9 §7 권고의 처리 (기록)

§7의 proxy 상수 권고 1–3은 **한 커밋으로** `lpopt/search/acquisition.py`에 반영되었다:
`FXY_PROXY_SLOPE = 1.2176`, `FXY_PROXY_INTERCEPT = -0.2519`, `FXY_PROXY_RESID_SD = 0.0476`,
`FXY_PROXY_SIGMA_K = 3.0` (유지). §7의 부수 발견(deck의 `cond_schema` stale)도 함께 처리하여
`lpopt.inp`/`lpopt_gpu1.inp`의 `[model] cond_schema`를 s1i meta와 같은 **v8**로 맞추었다.
이는 §4의 어떤 기준도 바꾸지 않는다 — PROXY는 §3의 비교 대상 이름일 뿐 게이트 축이 아니다.


---

# Amendment B — 2026-08-29 · **arm 2** (arm 1 FAIL 이후의 재시도, 실행 전 개정)

본 절은 §1–§10 과 Amendment A 를 **재작성하지 않는다.** 아래 항목만 **덮어쓰며(binding)**,
언급되지 않은 모든 조항(H1, G1의 형식·veto 성격, §8 금지 주장, 누수 규칙)은 그대로 유효하다.
개정 시점은 **arm 2 실행 전**이며, arm 1 의 채점 결과
(`data/reports/fxy_head_results_20260829.md`)는 이미 확정·기록되었다. **arm 1 의 바를
사후에 낮추는 것이 아니다** — arm 1 은 A.5/A.6 바로 채점되어 **FAIL** 로 종결되었고,
그 판정은 본 개정으로 바뀌지 않는다.

## B.1 왜 arm 2 인가 — arm 1 이 실패한 네 가지 원인 (측정)

| # | 원인 (측정 근거) | 결과 |
|---|---|---|
| 1 | best-epoch/early-stop 이 `composite` 로만 이루어지고 `fxy_metrics` 는 설계상 거기 들어가지 않는다 | best epoch 4·7·8·29·37 → **잔차가 학습될 시간이 없었다** (잔차 mean −0.0077 / sd 0.0259, 라벨 산포의 8%) |
| 2 | μ-only warmup 이 `warmup_epochs_effective = 80` 까지 간다 (`--epochs 150`, batch ratio 4×) — 그 동안 `log_sigma` 는 **gradient 를 전혀 받지 않는다** | 선택된 체크포인트(≤37)의 σ 는 **초기값** 0.562 = 라벨 sd 0.336 의 1.67배 → G4 커버리지 0.989 |
| 3 | `calibrate.fit_calibration` 이 모듈 상수 `TARGETS`(7열)만 순회했다 | `calibration.json` 에 `f_xy` 곡선이 **없다** → `predict_fxy` 의 "calibrated σ" 는 사실 raw σ. G4 는 **구조적으로 도달 불가**였다 |
| 4 | `net._compose_fxy` 는 raw `mu[f_r]` 행 위에서 합성하는데, `f_r_calibration.json` 은 그 **뒤에**(`predict` 안에서) 적용된다 | f_xy bias −0.081 = 1.2161 × s1i 의 **무보정** F_r bias(−0.0655). 잔차가 0인 동안 이것이 head 의 **바닥**이다 |

## B.2 오프라인 실험 — 어떤 설계가 얼마나 갈 수 있는가 (실행 전, CPU, S1j fold)

**방법.** 챔피언 `data/models/s1i` 를 CPU 로 로드해 S1j 의 **라벨된 converged 6,218행**
(train 5,425 / val 793) 전부에 대해 (i) 5멤버 앙상블 평균 `mu` 를 물리 단위로,
(ii) `f_r_calibration.json` 을 서빙과 **동일한 경로**(`_calib_cell_keys` → `apply_affine_calibration`,
bin 0.05, 102 cell + per-library fallback)로 적용한 **보정된** F_r 을,
(iii) 각 멤버의 `head_trunk` 출력(384차원)을 forward hook 으로 받아 멤버평균한
**동결 trunk 임베딩**을 추출했다. 모든 적합(ridge α는 `RidgeCV`, 임베딩 표준화 상수 포함)은
**train fold 5,425행에서만** 이루어졌고 val 793행은 어떤 적합에도 들어가지 않았다.

재현 (read-only, 코드·deck·챔피언 디렉터리를 건드리지 않는다):

```
python data/reports/fxy_arm2_offline_20260829_extract.py <out.npz>   # ~5분 (CPU)
python data/reports/fxy_arm2_offline_20260829_design.py  <out.npz>   # B.2 표
python data/reports/fxy_arm2_offline_20260829_design2.py <out.npz>   # #d/#e/#g 행
```

**왜 ridge 가 옳은 추정기인가.** arm 2 는 `--freeze-trunk-cyclen` 이므로
`stem`/`blocks`/`films`/`head_trunk`/`conv_head` 가 전부 동결되고, f_xy 를 내는 것은
그 384차원 임베딩 위의 **`mu_head` 한 줄(선형)** 이다. 즉 **ridge-on-embedding 이 arm 2 의
가설공간 그 자체**다. GBM 행은 비선형 여유(=trunk 를 푸는 미래 arm 의 상한)를 보여줄 뿐
**arm 2 에 대한 주장이 아니다**.

### B.2 표 — 같은 793행, 같은 11 cell(486행) 비가중 평균 ρ̄

| # | 추정기 | MAE | bias | ρ̄ | 바 통과? |
|---|---|---:|---:|---:|:--:|
| — | **BAR — 오늘 서빙 중인 proxy** `1.2176·F_r_pred_cal − 0.2519` | **0.0767** | +0.0053 | **0.7263** | — |
| — | (이상적 입력) prior on **측정** F_r — A.5/A.6 바의 출처 | 0.0355 | +0.0025 | 0.8944 | (도달 불가) |
| 0 | **arm 1 의 합성식**: 측정-적합 prior(1.2161/−0.2488)를 **raw 예측** F_r 위에 (잔차≡0) | 0.1056 | **−0.0772** | 0.7246 | ✗ |
| a | 예측 **보정** F_r 위 affine 재적합 (1.2038 / −0.2242) | 0.0760 | +0.0074 | 0.7263 | 경계 |
| a′ | 예측 **raw** F_r 위 affine 재적합 (1.3735 / −0.4397) | 0.0894 | +0.0135 | 0.7246 | ✗ |
| b | (a) + **ridge 잔차** on 동결 trunk 임베딩 | 0.0701 | +0.0058 | 0.7590 | ✅ |
| d | (0) as-built 합성 + ridge 잔차 | **0.0701** | +0.0042 | 0.7609 | ✅ |
| e | (a′) 예측-raw 재적합 prior + ridge 잔차 | 0.0702 | +0.0041 | **0.7626** | ✅ |
| g | **direct** — 합성 없이 ridge 직접 (선형 f_xy 행 그 자체) | 0.0708 | +0.0047 | 0.7664 | ✅ |
| — | (참고, 비선형) (a) + GBM 잔차 | 0.0662 | +0.0015 | 0.8392 | (arm 2 밖) |
| — | (참고, 비선형) GBM 직접 | 0.0676 | −0.0023 | 0.8444 | (arm 2 밖) |

> **행 0 에 대한 주석.** 이 행은 arm 1 의 **합성식**을 `s1i` 의 raw 예측 F_r 위에서 재현한
> 것이지 arm 1 체크포인트의 실측이 아니다. 후보 체크포인트의 실측 head 는
> 0.1051 / −0.0809 / 0.7422 (결과보고서 §3·§8)로, 두 숫자의 차이는 후보의 F_r 행이
> `s1i` 의 것과 미세하게 다르고 후보에는 거의 0인 잔차가 얹혀 있기 때문이다. 결론
> (편향이 F_r 합성에서 오고, 잔차가 사실상 없다)은 두 읽기에서 동일하다.

### B.3 이 표에서 읽히는 것 (설계 결정의 근거)

1. **ρ̄ 는 F_r 합성만으로는 절대 움직이지 않는다.** prior 는 F_r 의 단조증가 함수이므로
   cell 내 Spearman 은 어떤 (a, b) 를 써도 동일하다 — 보정 F_r 0.7263, raw F_r 0.7246.
   **바 ρ̄ > 0.7263 은 곧 "잔차가 순위 정보를 실제로 추가했는가"** 이며, 그것이 arm 2 가
   답해야 할 유일한 새 질문이다. (arm 1 의 ρ̄ 0.7422 는 거의 죽은 잔차가 낸 +0.016 이다.)
2. **잔차가 학습되기만 하면 합성 방식은 사실상 무관하다** (b/d/e/g 가 0.0701–0.0708,
   0.759–0.766 로 모두 같은 자리에 모인다). 동결 trunk 위에서 `mu[f_r]` 도 같은 임베딩의
   선형함수이므로 **"prior + 선형잔차" 와 "direct" 는 같은 가설공간**이다 — 합성은 용량이
   아니라 **초기화이자 열화 시의 바닥**이다.
3. **합성 방식이 결정하는 것은 그 "바닥"이다**: raw·측정적합 0.1056 → 예측-raw 재적합
   0.0894 → 예측-보정 0.0760. arm 1 은 잔차가 죽었을 때 **가장 나쁜 바닥**으로 떨어졌다.
4. **그럼에도 "보정 F_r 위 합성"을 채택하지 않는다.** 그것이 최선의 바닥(0.0760)이지만,
   `f_r_calibration.json` 은 **학습 시점에 존재하지 않으며**(학습 종료 후에 적합된다)
   서빙에서만 적용하면 train 과 serve 가 서로 다른 양을 읽게 된다 — 본 과제가 금지하는
   불일치다. 대신 **prior 를 모델 자신의 예측 F_r 위에서 적합**한다(#e): 합성이 읽는 바로
   그 행 위에서 적합되므로 **train 과 serve 가 정의상 동일**하고, bias −0.077 → +0.014 로
   구조적 편향이 사라지며, 바닥이 0.0894 로 올라간다. 최적점에서는 (b)·(e)·(g) 가 동률이므로
   이 선택은 **최적값을 희생하지 않는다**.
5. **A.5 의 바 0.0463 은 어떤 F_r-합성 설계로도 도달할 수 없다.** 그 바는 *측정* F_r 을
   입력으로 받는 prior 의 잔차 산포이고, 서빙 경로는 측정 F_r 을 **결코 받지 않는다**.
   오프라인 상한(선형 head)은 0.0701 이다. 이것은 A.5 를 완화하는 근거가 아니라,
   **A.5 가 채점하던 대상이 arm 2 의 판단 대상과 다르다**는 사실의 기록이다(B.5 참조).

## B.4 arm 2 의 정의 — 코드 변경 (실행 전 확정, 전부 커밋됨)

| # | 변경 | 파일 | 기본값(=회귀 없음) |
|---|---|---|---|
| 1 | `TrainConfig.fxy_prior_on_predicted` / `--fxy-prior-on-predicted` — prior 를 **모델 자신의 예측 F_r** 위에서 적합 (`refit_fxy_prior_on_predicted`, `--init-from` 필수, 라벨된 train 행 only) | `lpopt/model/train.py` | `False` |
| 2 | `TrainConfig.fxy_prior_residual` / `--fxy-direct` — 합성을 끄고 f_xy 행이 **절대값**을 예측 (`fxy_ref_idx = -1` → 네트워크는 합성-off 와 byte-identical) | `lpopt/model/train.py`, `net.py`(무변경) | `True` (=현행) |
| 3 | `TrainConfig.fxy_select_weight` / `--fxy-select-weight` — best-epoch/early-stop 점수 = `composite + w·fxy_select`, `fxy_select = ρ_cell(f_xy) − MAE(f_xy)/σ_z(f_xy)` (composite 와 같은 형태). legacy 축은 veto 를 유지한다 | `lpopt/model/train.py` | `0.0` (=legacy 선택과 **완전 동일**) |
| 4 | epoch 마다 `train.log` 에 `fxyMAE / fxyRho / n / sel` 기록; `promote_fxy` + `warm ≥ epochs/2` 이면 **σ 미학습 경고**를 인쇄 | `lpopt/model/train.py` | 로그만 |
| 5 | `fit_calibration` 이 **체크포인트의 `target_names`** 를 순회한다(+ `LPDataset` 을 같은 promotion 으로 구성). `apply_calibration` 은 이미 이름 기준이므로 `f_xy` 곡선이 제 열에 앉는다 | `lpopt/model/calibrate.py` | 7열 체크포인트는 byte-identical |
| 6 | meta `fxy_head` 에 `mode`(`prior_residual`/`direct`) · `prior_source`(`measured`/`predicted`) · `select_weight` 기록 | `lpopt/model/train.py` | 추가 키만 |

테스트: `tests/test_fxy_head.py` §10 (신규 12 케이스) — 예측-F_r 재적합이 F_r head 편향을
**흡수**하고 z-공간 왕복이 정확할 것, degenerate reference 는 `None`(측정 적합 유지),
라벨된 행만 적합, 모델의 train/eval 모드 불변, direct 모드가 합성-off 와 동일할 것,
`fxy_select` 의 정의, **실제 4-epoch 학습에서 best epoch 가 `select_score` 의 argmax 일 것**,
`fit_calibration` 이 `f_xy` 곡선을 쓸 것, 그리고 그 곡선이 `predict_fxy` 의 σ 에 **도달**할 것.

```
pytest tests/test_fxy_head.py tests/test_model_api.py tests/test_calibrate.py \
       tests/test_al_retrain.py tests/test_model_net.py -q      # 98 passed
```

## B.5 **바 (구속력, arm 2)** — 서빙 proxy 를 이겨야 한다

> arm 2 가 답하는 질문은 **"head 를 acquisition 에 넣을 것인가"** 이고, 그 대안은 이상적
> prior 가 아니라 **오늘 실제로 돌고 있는 proxy** 다. 따라서 arm 2 의 승격 조건은
> §8(결과보고서)이 확정한 서빙 proxy 실측치를 **둘 다** 이기는 것이다.

라벨된 S1j **VAL 793행**, cell 정의 (case_pair, feed), holdout 라벨 ≥ 20 인 **11 cell / 486행**
— A.6 과 **정확히 같은 집합**이다.

| 게이트 | 지표 | arm 2 기준 | 출처 |
|---|---|---|---|
| **G1** | `gate-promote --check-only` | `pass == true`, ε = σ0·Φ⁻¹(0.95^(1/N)) (N·ε 를 기록), `blind_targets == []` | **A.7 그대로, 무변경** |
| **G2′** | MAE(f_xy) | **< 0.0767** | 서빙 proxy on `s1i` (결과보고서 §8) |
| **G3′** | ρ̄ within-cell, 11 cell 비가중 평균 | **> 0.7263** | 〃 |
| **G4** | 68% 커버리지 | **∈ [0.55, 0.80]** | **§4 G4 그대로, 무변경** |

- **G2′ 와 G3′ 는 AND 다.** 하나만 넘으면 FAIL 이다 (proxy 대비 "더 낫다"고 말하려면
  레벨과 순위 둘 다여야 한다 — arm 1 은 순위만 +0.016 이면서 레벨은 37% 나빴다).
- **동률/근소차:** ρ̄ 차이가 `0 ± 0.05` 안이면 점 추정으로 판정하지 않는다. A.6 과 동일하게
  `lpopt.model.ab_paired` 의 cell-clustered paired BCa CI 를 산출하고 CI 가 0 을 포함하면
  **tie**(= FAIL, 승리로 읽지 않는다)로 기록한다. bias(+0.0053 대비)도 함께 인쇄한다.
- **A.5 / A.6 의 바(0.0463 / 0.8944)는 폐기되지 않는다 — 함께 보고한다.** 그 두 숫자는
  *측정* F_r 을 입력으로 받는 이상적 추정기의 자리이며, arm 2 가 그것을 넘지 못한다는
  사실은 §8 의 금지 조항(head 로 `F_xy ≤ 1.65` 적합성을 선언하지 말 것)을 계속 지탱한다.
  **두 읽기를 모두 판정표에 적는다.**
- **부수 보고(게이트 아님):** 오프라인 예측치는 MAE ≈ 0.070 / ρ̄ ≈ 0.76 (B.2 #e).
  실측이 이 자리에서 크게 벗어나면 그 자체가 조사 대상이다 — **사후에 바를 옮기지 않는다.**
- **G4 실패 시 처분:** §4 G4 그대로 — 처분을 바꾸지 않되 **σ 서빙 사용 금지**를 명시한다.

### 처분 규칙 (arm 2)

- **G1 PASS · G2′ PASS · G3′ PASS** → head 를 f_xy 서빙 경로로 승격하고 체크포인트를
  **`s1j`** 로 승격, deck `model_dir` 갱신. proxy 는 fallback 으로 남는다.
- **FAIL-G1** → **reject** (§4 그대로: 재시도는 새 사전등록을 요구한다).
- **G1 PASS · G2′ 또는 G3′ 중 하나라도 FAIL** → 챔피언 `s1i` 유지, **head 는 배포하지
  않는다**(shadow 포함). 근거: 결과보고서 §8 이 확정했듯 proxy 를 못 이기는 head 는
  이미 서빙 중인 추정기의 열화판이며 shadow 가 관측할 새 신호가 없다.
- **B.2 #g(direct) 로의 전환은 arm 2 의 결과로 자동 발동하지 않는다** — 별도 개정을 요구한다.

## B.6 arm 2 실행 커맨드 (확정, **본 문서는 실행하지 않는다**)

`[1]`·`[2]` 는 §3 그대로다.

```
[1] python -c "from lpopt.model.al_retrain import refresh_distill_cache; \
      refresh_distill_cache('data/models/s1i', out_path='data/models/_v5_distill_soft.npz')"

[2] python -m lpopt.remote --input lpopt_gpu1.inp push

[3] python -m lpopt.remote --input lpopt_gpu1.inp train -- \
      --ensemble 5 --split S1j --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 \
      --epochs 150 --num-workers 8 --device auto \
      --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
      --cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
      --promote-max-asm-bu --promote-fxy \
      --init-from data/models/s1i --freeze-trunk-cyclen \
      --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
      --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
      --fxy-prior-on-predicted --fxy-select-weight 0.5 --warmup-epochs 2
```

A.3 대비 **바뀐 것은 마지막 줄 세 개와 deck 이름뿐이다.**

| 플래그 | 값 | 왜 (B.1 의 몇 번을 고치는가) |
|---|---|---|
| `--fxy-prior-on-predicted` | (on) | #4 — 합성이 읽는 raw `mu[f_r]` 행 위에서 prior 를 적합해 F_r head 의 무보정 편향을 **흡수**한다. train/serve 동일. |
| `--fxy-select-weight` | `0.5` | #1 — best-epoch/early-stop 이 f_xy 를 **본다**. legacy 축의 veto 는 `composite` 항으로 유지된다. |
| `--warmup-epochs 2` | (기본 20 → 2) | #2 — batch ratio 4× 이므로 `warmup_epochs_effective = 8` (arm 1 은 20 → **80**). μ-only 구간이 patience(15)보다 짧으므로 **어떤 멤버도 NLL 을 한 번도 켜보지 않은 채 종료될 수 없고**, `log_sigma` 는 실제로 학습된다. `--init-from` 으로 legacy head 가 이미 수렴해 있어 긴 μ warmup 의 안정화 효과는 필요 없다. |
| (#3·#5·#6) | 코드 | 플래그 없이 항상 적용 — `f_xy` σ 곡선 적합, per-epoch 로그, meta 기록. |

> **등록해 두는 편차 (승인 필요, §3·§9.2 와 동일).** 위 커맨드는 `lpopt_gpu1.inp`
> (`[remote] gpu = 1`)를 쓴다. `lpopt.inp` 의 상시 지시는 *"사용자 지시 2026-07-24:
> GPU 0 고정, 재허가 전까지 auto 금지"* 다. **orchestrator 가 GPU 1 재허가를 확인해
> 기록한 뒤에만 [3] 을 실행한다.** 재허가가 없으면 `--input lpopt.inp` 로 GPU 0 에서
> 돌리고 그 사실을 결과에 적는다. (이 선택은 G1–G4 의 어떤 수치에도 영향을 주지 않는다.)

> **arm 1 이 남긴 운영 선행조건 (결과보고서 §10.5).** arm 1 의 산출물
> `data/models/20260829_135208` 은 per-cell 보정 5종이 적합되기 전에 pull 된 **미완료
> 스냅샷**이다. arm 2 는 `rc`/`DONE` 과 `f_r/cbc/f_q/ao_abs/flatness_calibration.json`
> **5종이 모두 존재하는 완료본**을 pull 한 뒤에 채점한다 — 특히 G1 은 서빙 경로를
> 채점하므로 보정이 빠진 스냅샷으로 읽으면 안 된다.

## B.7 채점 (arm 2, 실행 후 채울 것)

채점기는 arm 1 과 **같은 스크립트** `data/reports/fxy_gate_eval_20260829.py` 를 쓰되
`CAND` 만 arm 2 의 디렉터리로 바꾼다 (바 상수 `G2_BAR`/`G3_BAR` 는 A.5/A.6 읽기를 위해
그대로 두고, G2′/G3′ 는 같은 출력의 `errors.HEAD` / `G3.rho_head_mean` 을 위 표의
0.0767 / 0.7263 과 비교해 판정한다 — 스크립트는 read-only 이며 수정하지 않는다).

| 게이트 | 기준 | 측정값 | 판정 |
|---|---|---|---|
| G1 | `pass == true`, N·ε 기록, `blind_targets == []` | TBD | TBD |
| **G2′** | MAE < **0.0767** | TBD | TBD |
| **G3′** | ρ̄ > **0.7263** (tie 시 paired BCa CI, tie = FAIL) | TBD | TBD |
| G4 | 커버리지 ∈ [0.55, 0.80] | TBD | TBD |
| (참고) G2 | A.5 바 < 0.0463 | TBD | (승격 조건 아님) |
| (참고) G3 | A.6 바 ≥ 0.8944 | TBD | (승격 조건 아님) |
| (참고) bias | proxy 의 +0.0053 과 비교 | TBD | (판정 아님) |

---

# Amendment C — 2026-08-29 · **arm 3** (`--fxy-direct`, 합성 없음) · 실행 전 개정

본 절은 §1–§10 · Amendment A · Amendment B 를 **재작성하지 않는다.** 아래 항목만 **덮어쓰며(binding)**,
언급되지 않은 모든 조항(H1, G1 의 형식·veto 성격, §8 금지 주장, 누수 규칙, A.7)은 그대로 유효하다.
**arm 1 · arm 2 의 판정은 확정·불변이며 본 개정이 바꾸지 않는다** — arm 1 은 A.5/A.6 바로,
arm 2 는 B.5 바로 채점되어 각각 FAIL 로 종결되었고, 그 두 결과보고서
(`fxy_head_results_20260829.md`, `fxy_head_results_arm2_20260829.md`)는 그대로 둔다.

> **등록해 두는 시점.** 본 개정은 **arm 3 학습이 원격에서 진행 중일 때** 작성되었다. arm 3 은
> 오케스트레이터가 이미 투입한 `--fxy-direct` 실행이며, 그 실행이 쓰는 코드는 **개정 시점의
> push 본**이다. 본 문서는 **채점 규칙만** 고정하며 학습을 건드리지 않는다. 바·판정규칙은
> arm 3 의 어떤 산출도 관측하기 전에 확정되었고(측정값은 §C.6 이 전부 `TBD`), 사후에 이동하지 않는다.

## C.1 왜 arm 3 인가 — B.5 가 요구한 "별도 개정"

Amendment B.5 의 처분 규칙 마지막 줄은 이렇게 등록해 두었다:

> **B.2 #g(direct) 로의 전환은 arm 2 의 결과로 자동 발동하지 않는다 — 별도 개정을 요구한다.**

본 절이 그 개정이다. 근거는 arm 2 결과보고서가 **측정으로 특정한 단일 원인**이다(그 문서 §6.2):

- head 편향 −0.088948 의 **93%** 가 `net._compose_fxy` 가 읽는 **raw `mu[f_r]` 행**에서 왔다
  (평균 prior 기울기 ā = 1.15118 × raw F_r bias −0.071912 = −0.082783).
- `--fxy-prior-on-predicted`(B.4 #1)는 그 편향을 **흡수하지 못했다**: 적합은 `LPDataset` 텐서
  (= store 행의 자기 provenance)로, 서빙은 **다른 featurization** 으로 forward 했기 때문이다.
- **arm 3 은 그 채널 자체를 제거한다.** `--fxy-direct` 는 `fxy_ref_idx = -1` 로 합성을 끄므로
  f_xy 행은 `mu[f_r]` 을 **읽지 않는다**. "F_r 행에서 상속되는 편향"이 **구조적으로 부재**하며,
  arm 3 이 답하는 질문은 정확히 하나다: **합성을 뺀 동결-trunk 선형 head 가 서빙 proxy 의
  레벨(MAE)과 순위(ρ̄)를 동시에 이기는가.**
- 오프라인 상한(B.2 #g, direct): MAE 0.0708 / bias +0.0047 / ρ̄ 0.7664 — b/d/e 와 동률이며,
  B.3 #2 가 적었듯 "prior + 선형잔차"와 "direct"는 동결 trunk 위에서 **같은 가설공간**이다.
  **이 숫자는 게이트가 아니라 부수 보고**이고, 실측이 여기서 크게 벗어나면 그 자체가 조사 대상이다.

**arm 3 이 주장하지 않는 것.** "direct 로 바꾸면 편향이 사라진다"는 **가설이지 결과가 아니다.**
합성을 끄면 `mu[f_r]` 경유 편향은 사라지지만 f_xy 행이 자기 편향을 학습할 수 있다.
§8 의 금지 조항은 그대로 적용된다.

## C.2 arm 3 의 정의 — 코드 변경 없음

**arm 3 은 새 학습 코드를 요구하지 않는다.** B.4 표의 6개 변경은 이미 전부 커밋·검증되어 있고,
arm 3 은 그중 **이미 존재하는 플래그의 다른 조합**일 뿐이다.

| 플래그 | arm 2 | **arm 3** | 왜 |
|---|:--:|:--:|---|
| `--fxy-direct` (`cfg.fxy_prior_residual = False`) | off | **on** | B.1 #4 의 채널을 제거 — f_xy 행이 `mu[f_r]` 을 읽지 않는다 |
| `--fxy-prior-on-predicted` | on | **off (미전달)** | 합성이 없으므로 무의미하다. `train.py` 의 `refit_fxy = fxy_prior_on_predicted and fxy_idx >= 0 and fxy_ref_idx >= 0` 은 `fxy_ref_idx = -1` 에서 이미 `False` 이므로 전달 여부와 무관하게 **동작이 같다** — 커맨드에서 빼는 것은 의도의 기록이다 |
| `--fxy-select-weight 0.5` | on | **on (동일)** | B.1 #1 — best-epoch/early-stop 이 f_xy 를 본다 |
| `--warmup-epochs 2` | on | **on (동일)** | B.1 #2 — `warmup_epochs_effective = 8`, σ 가 실제로 학습된다 |
| 그 밖의 모든 인자 | — | **B.6 [3] 과 문자 단위로 동일** | 비교 가능성 |

meta 의 `fxy_head.mode` 는 `direct` 로 기록된다(B.4 #6 — `train.py`: `"prior_residual" if fxy_ref_idx >= 0 else "direct"`).
prior 는 **여전히 적합·기록되지만 아무것도 합성하지 않는다**(`--fxy-direct` 의 help 문구 그대로).

### 실행 커맨드 (확정, **본 문서는 실행하지 않는다**)

`[1]`·`[2]` 는 B.6 그대로다(`_v5_distill_soft.npz` 가 arm 2 이후 변경되지 않았다면 `[1]` 은 생략 가능하며,
생략했다면 그 사실을 결과에 적는다).

```
[3] python -m lpopt.remote --input lpopt_gpu1.inp train -- \
      --ensemble 5 --split S1j --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 \
      --epochs 150 --num-workers 8 --device auto \
      --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
      --cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
      --promote-max-asm-bu --promote-fxy \
      --init-from data/models/s1i --freeze-trunk-cyclen \
      --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
      --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
      --fxy-direct --fxy-select-weight 0.5 --warmup-epochs 2
```

B.6 대비 **바뀐 것은 마지막 줄뿐이다** (`--fxy-prior-on-predicted` → `--fxy-direct`).

> **GPU 1 편차 · 완료본 선행조건**: B.6 의 두 인용 블록을 **그대로 승계**한다.
> (i) `lpopt_gpu1.inp` 사용은 `lpopt.inp` 의 상시 지시(*"사용자 지시 2026-07-24: GPU 0 고정,
> 재허가 전까지 auto 금지"*)와 충돌하므로 orchestrator 가 재허가를 확인해 기록한 뒤에만 실행한다.
> (ii) 채점은 `rc`/`DONE` 과 per-cell 보정 5종(`f_r/cbc/f_q/ao_abs/flatness_calibration.json`)이
> **모두 존재하는 완료본**을 pull 한 뒤에 한다.

## C.3 채점 경로 — **어느 serve path 인가 (구속력)**

B.7 은 후보를 **서빙 경로**로 채점한다고 고정했다. 2026-08-29 의 train/serve 포렌식이 그 서빙
경로에서 **featurization 결함**을 찾아 고쳤으므로, "서빙 경로"가 무엇을 가리키는지 **여기서
명시적으로 고정한다.**

**arm 3 은 FIXED serve path 로 채점한다.**

| 항목 | 값 |
|---|---|
| 채점 코드 | `lpopt/model/model_api.py` sha256 **`94229de9e332c7faa66529f51b03d107f20098b1758f999b9c73ec8cfb21e6a2`** (110,976 B) |
| 〃 | `lpopt/model/featurize.py` sha256 **`6977344dafbd770c9b1bc40e370db6c189320e301f8fa49570a25f927b575e36`** (86,245 B) |
| 무엇이 달라졌나 | `PosValCnnBackend._record_inputs` 가 `featurize.library_provenance` 대신 **`featurize.serve_provenance`** 로 `(dataset, sym_class)` 를 유도한다 (import 1줄 + 호출 1줄 + docstring; `featurize.py` 는 `serve_provenance`/`SERVE_DATASET`/`SERVE_SYM_CLASS`/`_DATASET_A_LIBRARIES` **추가만**). `_record_inputs` 의 다른 어떤 필드도, `predict`/`predict_fxy`/보정 훅의 어떤 산술도 바뀌지 않았고 `library_provenance` 자체는 **byte-identical** 이다 |
| 왜 | `library_provenance` 는 캠페인 행(`dataset="P"`) 도입 **이전**의 추출기 지도다. ga80 → `("B","free69")`, paramA → `("A","rot61")` 로 답하지만 store 74,717행의 실제 census 는 ga80 **B 574 / P 18,973**, paramA **P 16,316** 이다. 즉 서빙은 **ga80 요청마다 `g_sym_class` 를, paramA 요청마다 `g_dataset_flag` 를 학습 때와 반대로** 넣고 있었다 (cond v8 20개 global 중 2개) |
| 크기 (같은 793행, 같은 arm-2 후보) | raw `mu[f_r]` bias: `predict` **−0.071912** → **+0.001965**(= `predict_rows_raw`, 학습 featurization). 분해: **ga80 577행 −0.0992 → +0.0044**, paramA 216행 +0.0011 → −0.0045. `g_sym_class` **하나만** store 값으로 되돌려도 −0.0719 → **+0.0035** — **편향의 사실상 전부가 ga80 의 `g_sym_class`** 이며, paramA·e_core·library 해석은 원인이 아니다 |
| 나머지 필드는 이미 정합 | 같은 793행에서 `pattern`·`feed`·`case_pair`·`library_id`·`e_core`·`e_split` 은 **0/793 불일치** (effective-library 가 두 라이브러리 모두 round-trip 하고, 2026-08-29 e_core backfill 이 store 컬럼을 서빙 recipe 에 정확히 맞췄다) |
| parity gate | `tests/test_model_api.py::test_serve_row_featurization_parity` — 실제 store 50행(ga80 25 + paramA 25, S1j val 우선)에서 `predict` 의 featurization 과 `predict_rows_raw` 가 **모든 cond global · 모든 cell 채널 · raw 앙상블 평균을 1e-6 이내**로 일치할 것. 수정 전 코드를 주입하면 이 테스트는 `['g_sym_class']` 로 **FAIL** 한다 |
| parity 실측 (실제 챔피언 `s1i`, 793행 전부) | FIXED path 의 서빙 raw F_r(`_ensemble_raw`) 과 `predict_rows_raw` 가 **max\|Δ\| = 0.00e+00 — bit-identical** (양쪽 bias +0.006665 / MAE 0.050393). ORIGINAL path 에서는 서빙 raw 가 **−0.065512** 로 `predict_rows_raw` 의 +0.006665 와 갈라져 있었다. **보정 후**로 읽으면 반대로 ORIGINAL +0.002552 / MAE 0.052678 → FIXED **+0.084694 / MAE 0.102616** 인데, 이는 `f_r_calibration.json` 이 ORIGINAL path 위에서 적합되었기 때문이다(아래 귀결 3) |

**등록해 두는 귀결 (사후에 바꾸지 않기 위해 미리 적는다).**

1. **arm 2 의 판정은 이 수정으로 바뀌지 않는다.** arm 2 는 **당시의 서빙 경로**로 채점되었고
   그 판정(G2′ FAIL → AND 규칙에 의해 미승격)은 확정이다. 수정된 경로에서의 재채점은
   `data/reports/fxy_gate_eval_arm2b_20260829.json` 에 **정보 제공용**으로 남기며
   **재심(re-adjudication)이 아니다.**

   > **실측 (arm 2b, 정보 제공용 · 판정 아님).** 하네스
   > `data/reports/fxy_gate_eval_arm2b_20260829.py` (sha `1879cbef5946d122969a2cf8ac4afe055bf5b75fdcd68b746ec87adf94b84cc8`,
   > 15,814 B) = arm-2 하네스에서 **`OUT` 과 헤더 주석만** 바꾼 사본. 같은 793행·같은 11 cell:
   >
   > | 항목 | ORIGINAL path (arm 2 확정 판정) | FIXED path (arm 2b, 참고) |
   > |---|---:|---:|
   > | **G2′** MAE(f_xy) < 0.0767 | 0.108560 **FAIL** | **0.066886 PASS** |
   > | **G3′** ρ̄ > 0.7263 | 0.768111 PASS | **0.785331 PASS** |
   > | **G4** 커버리지 ∈ [0.55, 0.80] | 0.634300 PASS | **0.827238 FAIL** |
   > | head bias | −0.088948 | **−0.000821** |
   > | head σ̄ | 0.130321 | 0.120539 |
   > | (동반 판독) `PROXY on s1i` MAE | 0.076721 | **0.132102** |
   > | (동반 판독) `PROXY on s1i` bias | +0.005270 | **+0.105287** |
   > | (동반 판독) `PROXY on s1i` ρ̄ | 0.726296 | 0.714088 |
   > | (고정 자) `s1i` **보정** F_r MAE | 0.052678 | **0.102616** |
   > | (고정 자) `s1i` cyclen MAE | 1.852122 | **2.400414** |
   > | (고정 자) `s1i` CBC_max MAE | 14.210126 | **16.607513** |
   >
   > **독립 검증 — arm-2 결과보고서 §6.3 의 괴리가 닫힌다.** 그 절은 학습 로그의 best-epoch
   > `fxyMAE` 와 서빙 단일-멤버 MAE 사이의 **+0.035 괴리**를 등록해 두고 "featurization 경로가
   > 다르기 때문"이라고 적었다. 같은 793행에서:
   >
   > | seed | 학습 로그 `fxyMAE` | 서빙 MAE (ORIGINAL) | 서빙 MAE (**FIXED**) |
   > |---|---:|---:|---:|
   > | 20260716 | 0.0762 | 0.1138 | **0.0763** |
   > | 20260717 | 0.0772 | 0.1229 | **0.0772** |
   > | 20260718 | 0.0756 | 0.1052 | **0.0756** |
   > | 20260719 | 0.0769 | 0.1450 | **0.0769** |
   > | 20260720 | 0.0752 | 0.1039 | **0.0752** |
   >
   > **소수 4자리까지 일치한다** (20260716 만 +0.0001). 학습 경로와 서빙 경로가 같은 수를 내는
   > 것이 train/serve identity 의 정의이며, 이것이 본 수정에 대한 가장 강한 독립 증거다.
   >
   > **읽는 법 (두 가지를 분리해야 한다).**
   > (i) **head 의 개선은 진짜다** — `_compose_fxy` 가 읽는 raw `mu[f_r]` 의 편향이
   > −0.0719 → +0.0020 으로 바뀌었고, 평균 prior 기울기 1.15118 을 곱한 +0.0851 이 그대로
   > head bias 를 −0.0889 → −0.0008 로 옮겼다(산술 일치). 이는 arm-2 결과보고서 §6.2 가
   > 특정한 원인이 **재학습 없이** 제거된 것이다.
   > (ii) **proxy·legacy 축의 악화는 가짜다** — `f_r/cbc/f_q/ao_abs/cell_calibration.json` 은
   > **ORIGINAL path 위에서 적합**되었으므로 FIXED path 에서는 이미 무편향인 raw 값에 +0.073 을
   > 더한다. 순위(ρ_global)는 거의 그대로거나 개선되었고(F_q·node_peak·map_cov 개선) MAE 만
   > 나빠진 것이 그 증거다 — **아핀 보정은 단조라 순위를 못 바꾸고 레벨만 바꾼다.**
   > 즉 이 표의 아래 다섯 줄은 "**보정 재적합이 필요하다**"는 뜻이지 모델이 나빠졌다는 뜻이 아니다.
   >
   > **가정적 판정 (기록만 한다, 발동하지 않는다).** FIXED path 로 채점했다면 B.5 의 AND 규칙에서
   > G2′∧G3′ 가 모두 PASS 이므로 **처분이 뒤집힌다**(승격). G4 는 FAIL 이지만 §4/B.5 의 G4 처분
   > 규칙은 "처분을 바꾸지 않되 σ 서빙 사용 금지"이므로 처분에 영향이 없다. **단 G1 은 FIXED path
   > 로 재실행되어야 하며 본 표에 없다** — G1 은 veto 이고 보정 아티팩트가 얹힌 서빙 열을 채점하므로
   > 위 (ii) 의 영향을 직접 받는다. **따라서 "arm 2 는 사실 통과였다"고 말할 수 없다.**
   > 본 문서는 이 사실만 기록하고 **어떤 승격·개명·deck 수정도 하지 않는다** — 결정은 orchestrator 다.
2. **바 0.0767 / 0.7263 은 ORIGINAL serve path 에서 측정된 값이다.** 본 개정은 지시대로
   **바를 이동하지 않는다.** 다만 FIXED path 에서는 **서빙 proxy 자신의 값도 움직이므로**,
   채점자는 §C.6 의 "동반 판독" 두 칸(같은 793행·같은 11 cell 에서 FIXED path 로 재측정한
   `PROXY on s1i` 의 MAE 와 ρ̄)을 **반드시 함께 인쇄한다**. 그 두 칸은 **판정에 쓰지 않는다** —
   판정은 고정 상수 0.0767 / 0.7263 으로만 한다. 두 판독이 엇갈리면(고정 바는 통과하지만
   재측정 proxy 에는 패배) 그 사실을 결과보고서에 **명시**하고 처분은 orchestrator 에게 넘긴다.
3. **per-cell 보정 5종은 서빙 경로로 적합된다** (`cell_calibrate._fit_cell_affine_target` 이
   `backend.predict(...)` 를 호출한다). 따라서 FIXED path 아래에서 **기존 챔피언의
   `cell_/f_r_/cbc_/f_q_/ao_abs_calibration.json` 은 더 이상 자기 적합점 위에 있지 않다.**
   arm 3 후보는 학습 종료 후 자기 보정을 (그 시점 코드로) 적합하지만, 비교 대상 `s1i` 는
   ORIGINAL path 에서 적합된 아티팩트를 그대로 쓴다. 채점자는 이 **비대칭을 결과보고서에 적고**,
   `s1i` 를 FIXED path 로 재보정하는 결정은 orchestrator 에게 넘긴다(본 문서는 결정하지 않는다).

   > **운영 경고 (등록).** 위 (ii) 때문에, **보정 5종을 FIXED path 로 재적합하지 않은 채
   > 진행 중인 캠페인에 본 수정을 배포하면 안 된다.** `acquisition` 은 `backend.predict` 의
   > **보정된** 열을 읽으므로, 재적합 전에는 F_r/cyclen/CBC 의 **레벨**이 실측대로 악화된다
   > (같은 793행에서 `s1i` 보정 F_r MAE 0.0527 → 0.1026, cyclen 1.85 → 2.40 EFPD).
   > 순위는 보존되므로 **순위만 쓰는 스크리닝**은 안전하지만, `F_r ≤ 1.55` 같은 **수준 임계**를
   > 쓰는 어떤 경로도 재적합 전에는 신뢰할 수 없다. 재적합은
   > `cell_calibrate.fit_cell_affine{,_fr,_cbc,_fq,_ao}` 를 FIXED path 로 다시 돌리는 것이며,
   > **본 문서의 범위 밖이고 `data/models/*` 를 쓰므로 여기서 수행하지 않았다.**
4. **원격 GPU 박스는 아직 수정 전 코드다.** `remote_infer.run_request` 는 원격에서
   `_record_inputs` 를 실행하므로, push 전까지 local(수정) ↔ remote(미수정) 는 **같은 입력에
   다른 featurization** 을 쓴다. 실측: `tests/test_remote_infer.py::test_remote_gpu_matches_local_cpu_determinism`
   의 `mu_z` 최대 차이가 기지의 **4.282e-03 → 1.209e+00** 으로 벌어진다(수정 전 코드를 주입하면
   4.282e-03 로 복귀). **arm 3 채점은 로컬 CPU 로 하며 원격 스크리닝을 켜지 않는다.**
5. **`lpopt/policy/scorer.py` 는 여전히 `library_provenance` 를 쓴다.** 같은 결함을 갖지만 본
   포렌식의 범위(모델 서빙 경로) 밖이므로 **고치지 않았고**, 여기에 미결로 등록한다.

**만약 orchestrator 가 수정을 되돌리기로 결정하면**, arm 3 은 ORIGINAL serve path
(= 위 두 파일에서 `serve_provenance` → `library_provenance` 로 되돌린 것)로 채점하고
**그 사실과 그 때의 sha256 을 결과보고서에 적는다.** 어느 쪽이든 **바는 이동하지 않는다.**

## C.4 바 (구속력, arm 3) — Amendment B 와 **동일**

라벨된 S1j **VAL 793행**, cell 정의 (case_pair, feed), holdout 라벨 ≥ 20 인 **11 cell / 486행**
— A.6 · B.5 와 **정확히 같은 집합**이다.

| 게이트 | 지표 | arm 3 기준 | 출처 |
|---|---|---|---|
| **G1** | `gate-promote --check-only` | `pass == true`, ε = σ0·Φ⁻¹(0.95^(1/N)) (N·ε 를 기록), `blind_targets == []` | **A.7 그대로, 무변경** |
| **G2′** | MAE(f_xy) | **< 0.0767** | **B.5 그대로, 무변경** |
| **G3′** | ρ̄ within-cell, 11 cell 비가중 평균 | **> 0.7263** | **B.5 그대로, 무변경** |
| **G4** | 68% 커버리지 | **∈ [0.55, 0.80]** | **§4 G4 그대로, 무변경** |

- **G2′ 와 G3′ 는 AND 다.** 하나만 넘으면 FAIL 이다.
- **동률/근소차:** ρ̄ 차이가 `0 ± 0.05` 안이면 점 추정으로 판정하지 않는다. B.5 와 동일하게
  `lpopt.model.ab_paired` 의 cell-clustered paired BCa CI 를 산출하고 CI 가 0 을 포함하면
  **tie**(= FAIL, 승리로 읽지 않는다)로 기록한다. bias(+0.0053 대비)도 함께 인쇄한다.
- **A.5 / A.6 의 바(0.0463 / 0.8944)는 폐기되지 않는다 — 함께 보고한다.**
- **G4 실패 시 처분:** §4 G4 그대로 — 처분을 바꾸지 않되 **σ 서빙 사용 금지**를 명시한다.
- **부수 보고(게이트 아님):** 오프라인 예측치 MAE ≈ 0.0708 / bias ≈ +0.0047 / ρ̄ ≈ 0.7664 (B.2 #g).
  실측이 이 자리에서 크게 벗어나면 그 자체가 조사 대상이다 — **사후에 바를 옮기지 않는다.**

### 처분 규칙 (arm 3) — **B.5 와 동일**

- **G1 PASS · G2′ PASS · G3′ PASS** → head 를 f_xy 서빙 경로로 승격하고 체크포인트를
  **`s1j`** 로 승격, deck `model_dir` 갱신. proxy 는 fallback 으로 남는다.
- **FAIL-G1** → **reject** (§4 그대로: 재시도는 새 사전등록을 요구한다).
- **G1 PASS · G2′ 또는 G3′ 중 하나라도 FAIL** → 챔피언 `s1i` 유지, **head 는 배포하지
  않는다**(shadow 포함). 근거는 B.5 그대로: proxy 를 못 이기는 head 는 이미 서빙 중인 추정기의
  열화판이며 shadow 가 관측할 새 신호가 없다.
- **arm 3 의 결과로 어떤 후속 arm 도 자동 발동하지 않는다** — arm 4 는 별도 개정을 요구한다.

## C.5 채점기

arm 2 와 **같은 하네스**를 쓰되 `CAND`/`OUT` 만 arm 3 의 디렉터리로 바꾼 사본
`data/reports/fxy_gate_eval_arm3_20260829.py` 를 만든다. 바 상수 `G2P_BAR`/`G3P_BAR`
(0.0767 / 0.7263)는 **손대지 않는다**. arm-1 하네스 `fxy_gate_eval_20260829.py`
(sha `b2bbc084ec12d631b9f251fa3d038da34b047d543b5c4fb89963f1bfb8dde0ca`)와 arm-2 하네스
`fxy_gate_eval_arm2_20260829.py` (sha `b638e0de71b81abd6c9ab0de0357fa65cf1eaed5e6e4e3af7fd650655a37ee94`)는
**변경 금지**다.

## C.6 채점 (arm 3) — **채워짐 2026-08-30**

기입 출처: `data/reports/fxy_head_results_arm3_20260829.md` §1
(결과 JSON `fxy_gate_eval_arm3_20260829.json` `f70136d1…`, G1 JSON
`gate_fxy_arm3_20260829_checkonly.json` `371f0e5a…`). 후보는
`data/models/20260829_194532` 였고, 본 판정에 따라
**`data/models/s1j` 로 승격**되었다 (2026-08-30,
`data/models/s1j/PROMOTION.md`). **바는 이동하지 않았다.**

| 게이트 | 기준 | 측정값 | 판정 |
|---|---|---|---|
| G1 | `pass == true`, N·ε 기록, `blind_targets == []` | `pass = true`; **N = 108, ε = 0.1388114093847** (실행값) / **N = 144, ε = 0.14216194539159127** (사전등록 재산출); worst enforced drop **0.011364**; `blind_targets = []`, `unavailable = []`; 36 cell / 144 checks | **PASS** |
| **G2′** | MAE < **0.0767** | **0.066300** (n = 793; 바의 0.865배) | **PASS** |
| **G3′** | ρ̄ > **0.7263** (tie 시 paired BCa CI, tie = FAIL) | **0.790392** (11 cell, Δ = +0.064092). 재측정 proxy 와의 차 0.074696 > 0.05 이므로 **tie 구간 밖** → 점추정 판정. 확인용 BCa: **+0.074696, CI [+0.021430, +0.264692]** (0 미포함) | **PASS** |
| G4 | 커버리지 ∈ [0.55, 0.80] | **0.831021** (σ̄ 0.119787) — **상단 초과(over-wide)** | **FAIL** |
| (참고) G2 | A.5 바 < 0.0463 | 0.066300 | FAIL (승격 조건 아님) |
| (참고) G3 | A.6 바 ≥ 0.8944 | 0.790392 (prior ρ̄ 0.894372) | FAIL (승격 조건 아님) |
| (참고) bias | proxy 의 +0.0053 과 비교 | **−0.003137** (resid sd 0.099290) | (판정 아님) |
| **동반 판독 (판정 아님)** | FIXED path 재측정 `PROXY on s1i` MAE | **0.073173** | — |
| **동반 판독 (판정 아님)** | FIXED path 재측정 `PROXY on s1i` ρ̄ (11 cell) | **0.715696** | — |
| (기록) | 채점에 쓴 `model_api.py` sha256 · FIXED / ORIGINAL | `94229de9e332c7faa66529f51b03d107f20098b1758f999b9c73ec8cfb21e6a2` — **FIXED** (`featurize.py` `6977344dafbd770c9b1bc40e370db6c189320e301f8fa49570a25f927b575e36`) | — |
| (기록) | meta `fxy_head.mode == "direct"` 인가 | **예 — 5개 멤버 전부 `"direct"`** (`target_idx = 8`, `select_weight = 0.5`, `n_labelled_train = 5425`) | — |
| (기록) | `[1]` distill 캐시 재생성 여부 · GPU 재허가 기록 | 캐시 **재생성 안 함** (`_v5_distill_soft.npz` mtime 2026-08-29 13:52:12, arm 2·arm 3 공용). **GPU 재허가는 미확인** — 채점자가 확인할 수 없는 orchestrator 기록 사항이며 2026-08-30 승격 시점에도 **미해결로 남았다** (`run.sh` = `CUDA_VISIBLE_DEVICES=1`, deck `lpopt_gpu1.inp` vs `lpopt.inp` 의 "GPU 0 고정" 상시 지시). G1~G4 어느 수치에도 영향 없음 | — |

> **처분 (C.4 적용, 실행됨 2026-08-30).** G1·G2′·G3′ PASS → **`s1j` 승격**.
> **G4 FAIL → head 의 σ 는 서빙되지 않는다** — 처분은 바뀌지 않는다(§4/B.5/C.4).
> 구현: `data/models/s1j/ensemble.json` 의 `fxy_head.serve_sigma = "barred"` 를
> `PosValCnnBackend` 가 읽고, `acquisition.predict_fxy` 가 head 의 **평균과
> `source = "head"` 는 유지하되 폭은 proxy σ 규약**(`resid_sd 0.0476` × `K 3.0`)으로
> 대체하며, `fxy_conformal_upper` 는 `None`(= proxy 스크린 유지)을 돌려준다.
> 회귀 시험: `tests/test_fxy_head.py` §12. 바의 해제는 config 편집이 아니라
> **새 사전등록과 새 커버리지 측정**을 요구한다.
