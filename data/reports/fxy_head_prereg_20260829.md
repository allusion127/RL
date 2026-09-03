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


---

# Amendment D — 2026-08-31 · **arm 4** (trunk fine-tune + ratio-anchored 합성) · 실행 전 개정

본 절은 §1–§10 · Amendment A · B · C 를 **재작성하지 않는다.** 아래 항목만 **덮어쓰며(binding)**,
언급되지 않은 모든 조항(H1, G1 의 형식·veto 성격, §8 금지 주장, 누수 규칙, A.7 의 ε 공식)은
그대로 유효하다. **arm 1 · arm 2 · arm 3 의 판정은 확정·불변이며 본 개정이 바꾸지 않는다.**
arm 3 은 C.4 바로 채점되어 PASS → `s1j` 로 승격되었고(2026-08-30), 그 판정은 그대로 둔다.
C.4 의 마지막 줄이 등록한 "arm 4 는 별도 개정을 요구한다"의 그 개정이 본 절이다.

> **등록해 두는 시점.** 본 개정은 **arm 4 학습을 투입하기 전에** 작성되었다. 아래 D.2 의 모든
> 수치는 **오프라인 실험**(챔피언 `s1j` 를 CPU 로 재적재해 얻은 예측·임베딩 위의 ridge/GBM)의
> 산출이며, arm 4 후보는 **아직 존재하지 않는다**(D.6.3 의 측정값은 전부 `TBD`). 바는 arm 4 의
> 어떤 산출도 관측하기 전에 확정되었고 사후에 이동하지 않는다.

---

## D.1 왜 arm 4 인가 — arm 3 승격 이후 측정된 결손

arm 3 은 **레벨(MAE)** 을 얻었으나 **캠페인 내부 순위**를 얻지 못했다. 세 개의 독립 관측:

| 출처 | 관측 |
|---|---|
| `minfxy_T6T4_f121_r1_results_20260830.md` §6.3 | 95 라벨행에서 head ρ_s **+0.392** < proxy **+0.439**; joint-clean 65행에서 head **+0.016**; `exploit` slot 62행에서 **−0.383**. regret@8 head 0.0432 vs proxy 0.0517 (sign test p ≈ 1.0, **유의하지 않음**) |
| `pinbu_wave_minfxy_r1_results_20260830.md` §5.1 / M8 | `F_r`-era 5 core 에서 model-free ratio(`1.0640 × F_r`, **측정** `F_r`) MAE **0.008** vs head 0.042–0.060 — **5–7배**. 20 라벨행 ρ(pred, meas) **−0.114** (p = 0.63). head bias 가 family 별로 부호를 뒤집는다(−0.0075 / +0.0420) |
| `intervention_wave_r1_results_20260830.md` §6.1–6.2 (H3·H4) | `F_xy:F_r` 전달계수가 move family 로 갈린다(농축-반경 1.23–1.42 vs Gd/격자 0.55–0.73). `d_fresh_enr_r_center ≡ 0` 인 개입이 `F_xy` 를 **0.07** 움직인다(E1E2_f109 +0.0712 p=1.9e-6, E1E2_f121 +0.0689 p=0.0026) |

**arm 4 가 답하는 질문(단 하나):** cell 안에서, **서빙 가능한 입력만으로** ratio proxy 와
arm-3 head 를 순위에서 이기는 구성은 무엇인가.

---

## D.2 오프라인 실험 (실행 전, CPU, read-only)

**하네스** `data/reports/fxy_arm4_offline_20260831.py`
(sha256 `884298bd48abb472e7681b2eb562782deedc4e14cc08e374f826101113d43457`, 29,640 B),
결과 JSON `data/reports/fxy_arm4_offline_20260831.json`
(sha256 `9b126649a4988d58fccaf7d806a2778e186f2802fd990fe9651341e328b7f928`).

```
python data/reports/fxy_arm4_offline_20260831.py extract <cache.npz>   # ~6분 (CPU)
python data/reports/fxy_arm4_offline_20260831.py design  <cache.npz> \
       --json data/reports/fxy_arm4_offline_20260831.json
```

**방법.** 챔피언 `data/models/s1j` 를 CPU 로 적재해 store 의 라벨된 converged `f_xy` **7,645행**
전부에 대해 (i) 5멤버 앙상블 평균 `mu` 를 물리 단위로(= `predict_fxy` 가 내는 바로 그 열),
(ii) 서빙과 동일 경로의 **보정된** `F_r`, (iii) 각 멤버 `head_trunk` 출력(384차원)의 멤버평균,
(iv) cond-v8 **20 globals**, (v) 후보 (c)/(c′) 기술자를 추출했다. 모든 적합은 **S1j train fold
5,429행에서만** 이루어졌다. 코드·deck·체크포인트·store 를 **수정하지 않았다**.

**하네스 검증.** 같은 하네스가 `s1j` head 에 대해 산출한 값은 **VAL MAE 0.0663 / ρ̄ 0.7907**,
arm-3 채점기(`fxy_gate_eval_arm3_20260829.py`)의 **0.066300 / 0.790392** 와 소수 4자리에서
일치한다. 슬라이스가 793 → 794행으로 1행 늘었음에도 그렇다(D.5) — 본 오프라인 표는 arm-3
판정과 **같은 자를 쓴다**.

### D.2.1 평가 슬라이스

| 이름 | 정의 | n |
|---|---|---:|
| **VAL** | S1j val fold ∩ 라벨 ∩ converged | **794** |
| **VAL11** | 그중 holdout 라벨 ≥ 20 인 **11 cell** (A.6·B.5·C.4 와 동일 집합) | 487 |
| **R1** | campaign `fpcamp_minfxy_t6t4_f121_r1` — **전량 fold 밖**(train 0 / val 0) | **95** |
| **R1jc** | R1 ∩ joint-clean (minfxy r1 prereg 정의: `valid & cbc≤1600 & f_q≤2.41 & abs(ao)≤0.30 & f_r≤1.55`) | **65** |
| **PINBU** | `pinbu_wave_minfxy_r1_manifest.json` 의 25 record_id (r1 top-20 + `F_r`-era 5) | **25** |

R1 이 어느 fold 에도 없다는 것은 하네스가 **assert 로 강제**한다 — 순위 판독이 학습 누수가
아님이 구조적으로 보장된다.

### D.2.2 기준선과 후보 (같은 794 / 95 / 65 / 25행)

ρ̄ 는 VAL11 의 cell별 Spearman 비가중 평균, ρ 는 해당 슬라이스의 단일 Spearman.
`emb` = 동결 trunk 임베딩 384차원, `g` = cond-v8 20 globals, `MOM` = (c′) 반경 모멘트 블록,
`GD` = (c) Gd/격자 블록. `RESID` = ratio prior 에 대한 잔차(후보 b), `LOGR` = log-ratio 표적(후보 d).

| # | 추정기 | VAL MAE | VAL ρ̄ | R1 ρ | R1jc ρ | PIN ρ |
|---|---|---:|---:|---:|---:|---:|
| — | **서빙 PROXY** `1.2176·F_r_pred_cal − 0.2519` | 0.0721 | 0.7156 | +0.4664 | +0.1170 | −0.4217 |
| — | **RATIO on 예측 F_r** (`r = 1.0819`, train 적합) | 0.0737 | 0.7156 | +0.4664 | +0.1170 | −0.4217 |
| — | (이상, **서빙 불가**) `1.0588 ×` **측정** `F_r` — pinbu 5배 승리의 그 추정기 | 0.0528 | 0.8945 | **+0.8984** | **+0.7201** | +0.2632 |
| — | (이상, 서빙 불가) affine prior on **측정** `F_r` | 0.0355 | 0.8945 | +0.8984 | +0.7201 | +0.2632 |
| **0** | **HEAD `s1j` (arm 3, direct) — 무회귀 기준** | **0.066256** | **0.790696** | **+0.488352** | **+0.204812** | **−0.487110** |
| a1 | (ctl) ridge ABS [emb+g] — arm 3 의 가설공간 재적합 | 0.0722 | 0.7908 | +0.6184 | +0.3585 | −0.5194 |
| b1 | **(b)** ridge RESID(ratio prior) [emb+g] | 0.0718 | 0.7817 | +0.6330 | +0.3891 | −0.3694 |
| b2 | **(b-aff/RAW)** ridge resid on affine(예측 **raw** `F_r`) = **arm 2 의 합성** | 0.0719 | 0.7862 | **+0.6367** | **+0.4175** | −0.4552 |
| d1 | **(d)** ridge LOGR [emb+g] | 0.0717 | 0.7882 | +0.6219 | +0.3998 | −0.4117 |
| a2 | (ctl) **GBM** ABS [emb+g] — 용량만 | 0.0692 | 0.8312 | +0.5977 | +0.3473 | +0.1266 |
| b3 | **(b) GBM** RESID(ratio) [emb] | **0.0659** | 0.8398 | +0.5293 | +0.2813 | +0.2443 |
| b4 | **(b-aff/RAW) GBM** resid on affine(예측 raw `F_r`) — **arm 4 의 오프라인 대응물** | 0.0666 | **0.8370** | +0.5283 | +0.2586 | **+0.2497** |
| d2 | **(d) GBM** LOGR [emb] | 0.0661 | **0.8483** | +0.4974 | +0.2188 | +0.1993 |
| d3 | **(d) GBM** LOGR [emb+g+MOM] | 0.0662 | 0.8479 | +0.5672 | +0.2881 | +0.1381 |
| c1 | **(c)** ridge/GBM + **Gd/격자 블록** | — | **D.3 참조 — 이득 없음** | | | |
| — | (등록된 함정) LOGR 을 **측정** `F_r` 로 적합하고 **예측** `F_r` 로 서빙 | 0.0703 | 0.7855 | +0.5799 | +0.4120 | −0.2963 |

**cell-clustered paired BCa (VAL11, 11 cell, 2,000 reps, seed 0) 및 R1 행-부트스트랩
(2,000 reps, seed 0), 대조 = HEAD `s1j`:**

| 후보 | ΔVAL ρ̄ | 95% CI | ΔR1 ρ | 95% CI |
|---|---:|---|---:|---|
| a1 ridge ABS [emb+g] | +0.0042 | [−0.0171, +0.0316] | **+0.1300** | **[+0.0104, +0.2546]** |
| d1 ridge LOGR [emb+g] | +0.0015 | [−0.0184, +0.0320] | **+0.1335** | **[+0.0377, +0.2365]** |
| a2 GBM ABS [emb+g] | +0.0406 | [−0.0239, +0.0677] | +0.1094 | [−0.0043, +0.2293] |
| b3 GBM RESID [emb+g] | +0.0312 | [−0.0075, +0.0818] | +0.0111 | [−0.0748, +0.0922] |
| **b4 (b-aff/RAW) GBM** | **+0.0451** | [−0.0080, +0.0657] | +0.0400 | [−0.0278, +0.1107] |
| d2 GBM LOGR [emb] | **+0.0538** | **[+0.0014, +0.0839]** | +0.0091 | [−0.0626, +0.0805] |
| d3 GBM LOGR [emb+g+MOM] | +0.0478 | [−0.0003, +0.0680] | **+0.0788** | **[+0.0174, +0.1538]** |

> **등록해 두는 정직한 요약: 오프라인 증거는 두 바를 CI 로 동시에 넘는 구성을 찾지 못했다.**
> VAL 을 CI 로 이기는 것은 d2 하나, R1 을 CI 로 이기는 것은 a1·d1·d3 이다. 나머지는 전부 tie 다.
> arm 4 는 **그 사실을 알고** 투입되며, D.6.2 의 forecast 는 그대로 기록된다.

### D.2.3 표에서 읽히는 것 — **순위의 병목은 `F_xy` 사상이 아니라 `F_r` 예측이다**

`ratio × F_r` 은 그 `F_r` 입력의 **단조 변환**이므로 cell 내 Spearman 은 그 `F_r` 열의
Spearman **그 자체**다(B.3 #1 의 재확인). 따라서 pinbu 가 측정한 "ratio 가 head 를 5배 이겼다"는
**측정 `F_r` 에 대한 진술**이지 서빙 가능한 추정기에 대한 진술이 아니다. 분해:

| 슬라이스 | n | ρ(**측정** `F_r`, `f_xy`) | ρ(**예측** `F_r`, `f_xy`) | ρ(**예측** `F_r`, **측정** `F_r`) |
|---|---:|---:|---:|---:|
| VAL | 794 | +0.9902 | +0.9622 | +0.9728 |
| **R1** | 95 | **+0.8984** | +0.4664 | **+0.5096** |
| **R1jc** | 65 | **+0.7201** | +0.1170 | **+0.1735** |
| PINBU | 25 | +0.2632 | −0.4217 | **−0.3993** |
| VAL11 (cell별 평균) | 487 | 0.8945 | — | **0.7894** |

**세 줄로:**
1. 서빙 ratio proxy 의 R1 순위는 **+0.466 로 arm-3 head(+0.488)보다 낮다.** "ratio 를 순위에서
   이겨라"는 과제는 **서빙 경로에서는 이미 충족되어 있으며**, 이길 대상이 아니다.
2. **elite basin 안에서 `F_r` 예측 자체의 cell 내 순위 충실도가 무너진다** — R1jc 에서
   ρ(예측 `F_r`, 측정 `F_r`) = **+0.1735**, PINBU 에서 **−0.3993**. `F_xy` 를 `F_r` 로부터
   합성하든 직접 예측하든, 그 위에 얹힌 어떤 추정기도 이 천장 위로 크게 오를 수 없다.
3. 따라서 arm 4 의 레버는 **표적 파라미터화((b)/(d))도, 새 기술자((c))도 아니라 표현 용량**이다 —
   동결 trunk 위 GBM 이 VAL ρ̄ 를 0.79 → 0.83~0.85 로 올린다. B.2 가 이미 등록한 대로
   **GBM 행은 "trunk 를 푸는 미래 arm 의 상한"** 이며, 그 미래 arm 이 arm 4 다.

---

## D.3 후보 (c) — **데이터가 없다.** 채택하지 않는다

`intervention_wave_r1_results_20260830.md` §13.2-2 가 지목한 축은
`fuel_types.parquet` 의 `n_gd` / `gd_wt` / `gd_u_enr` / `zone_pin_count` 의 반경 가중 모멘트다.
하네스가 그 블록을 실제로 만들어 넣고 측정한 결과:

| 사실 | 근거 (JSON 의 `descriptor_stats` / `ablation`) |
|---|---|
| `zone_pin_count` 는 **ga80·paramA 양쪽에 존재하지 않는다** | 194 type 중 `260624`/`5.8_5.1`/`CPHA` 만 보유. VAL·R1 전 행에서 mean 0, sd **0.00000** |
| `gd_wt` / `gd_u_enr` 는 **ga80 에 0% 존재** | 라이브러리별 non-null 비율 ga80 **0.000**, paramA 1.000. VAL 794행의 slot 커버리지 평균 **0.2733** |
| `n_gd` 는 **type 표에서** ga80 의 51.4% 만 보유 — 단 **실제 쓰인 행에서는 커버리지 1.000** | harvested descriptor 전체(ADF·kconv 포함)가 같은 51.4% 게이트를 공유하지만, 라벨된 core 가 실제로 참조하는 type 들은 전부 채워져 있다. 즉 `n_gd` 는 **있고, 그런데도 도움이 되지 않는다**(아래 두 줄) |
| R1 cell 안에서 Gd 재고는 **사실상 상수** | `n_gd_mean` R1 sd **0.247** (mean 20.00), `n_gd_r_center` R1 sd **0.0041** |
| ablation: 이득 없음 | ΔVAL ρ̄ ridge −0.0011 / −0.0105, GBM +0.0024 / +0.0044; **ΔR1 ρ −0.0115 / −0.1347 / −0.0176 / −0.0312** (네 조합 전부 악화) |

> **판정: (c) 는 schema 문제 이전에 DATA 문제다.** H4 가 성립한 3개 축퇴 cell 은 **전부 ga80**
> 이고, 그 라이브러리에는 H4 를 설명할 Gd 기술자 자체가 harvest 되어 있지 않다. cond v9 로
> 채널을 늘려도 넣을 값이 없다. **arm 4 는 (c) 를 채택하지 않으며, 스키마를 건드리지 않는다.**
> 대신 D.7-①(fuel table backfill)을 **별도 구현 과제**로 등록한다.

**(c′) — 이미 존재하는 per-slot 기술자의 반경 모멘트** (`u_avg_enrichment`,
`reactivity_swing_pcm`, `depletion_slope_pcm_per_gwd`, `kinf_eol50`, `adf_face_g2`,
`ff_pin_max`, `bu_peak_gwd` 의 centroid / fresh-centroid / spread; 새 fuel 컬럼 불필요)는
**동결 trunk 위 선형 probe 에만 이득을 준다**(ridge ΔVAL ρ̄ **+0.0151 / +0.0158**), GBM 에서는
**+0.0071 / −0.0006** 으로 사라진다. 즉 (c′) 의 이득은 **동결 trunk 의 용량 한계가 만든 증상**이며
trunk 를 풀면 네트워크가 스스로 계산할 수 있는 양이다. **(c′) 도 스키마 변경으로 채택하지 않고,
(a) 에 흡수된 것으로 기록한다.**

---

## D.4 **arm 4 의 정의 (구속력)**

> **arm 4 = (a) trunk fine-tune + (b) ratio-anchored 합성.**
> `s1j` 에서 출발해 **trunk 를 낮은 LR 로 푼다**(cyclen 행은 계속 gradient 0). `f_xy` 행은
> `--fxy-direct` 대신 **`--fxy-prior-on-predicted`** 로 돌아가, 모델 자신의 raw `mu[f_r]` 위에
> 적합된 affine prior 에 대한 **잔차**를 회귀한다.

| 요소 | arm 3 | **arm 4** | 근거 |
|---|:--:|:--:|---|
| `--init-from` | `s1i` | **`s1j`** | 승격된 챔피언에서 출발 (무회귀 기준도 `s1j`) |
| trunk | `--freeze-trunk-cyclen` (동결) | **fine-tune**, `lr × 0.05`, cyclen 행 hook 유지 | D.2.3 #3 — 용량이 유일하게 확인된 순위 레버 |
| `f_xy` 합성 | `--fxy-direct` | **`--fxy-prior-on-predicted`** | b4 (MAE 0.0666) vs a2 (0.0692): **ratio anchor 가 레벨을 되찾는다.** 순위는 tie |
| `--fxy-select-weight` | 0.5 | **0.5 (동일)** | B.1 #1 |
| `--warmup-epochs` | 2 | **2 (동일)** | B.1 #2 |
| distill / rank | 0.4 / 0.1 | **0.4 / 0.1 (동일)** | trunk 가 풀리므로 이제 **G1 의 실질적 방어선** |
| cond schema | v8 | **v8 (변경 없음)** | D.3 — (c)·(c′) 미채택 |

**LR 배수 0.05 는 등록된 선택이며 사후에 조정하지 않는다.** trunk 가 head LR 의 5% 를 받는다
= 재학습이 아니라 fine-tune 이라는 뜻이고, 이 값에 대한 오프라인 증거는 **없다**(GBM 대리물은
LR 을 모른다). 배수를 바꾸는 재시도는 **새 사전등록을 요구한다**(D.6.4).

### D.4.1 채택하지 않은 후보와 그 이유 (사전 기록)

- **(d) log-ratio 표적**: 같은 용량에서 (b) 보다 근소 우위(GBM: ρ̄ 0.8483 vs 0.8398 @emb,
  0.8408 vs 0.8303 @emb+g)이나 **차이는 tie 밴드(±0.05) 안**이고, 구현은 `net._compose_fxy` 의
  **물리공간 곱셈 합성 + 표적 변환 + 보정 경로**를 요구한다(D.7-②). **+0.008 을 위해 다섯 파일을
  바꾸지 않는다.** arm 4 가 G5 를 놓치면 **arm 5** 로 승격 후보이며, 그때는 새 개정을 요구한다.
- **(c) / (c′)**: D.3.
- **재분할(S1k 분할)**: 하지 **않는다.** 현재 store 에는 어느 fold 에도 없는 라벨 **1,422행**이 있고
  그중 95행이 R1 이다. 재분할하면 (i) A.6/B.5/C.4/D.6 의 바가 전부 무효가 되고 (ii) **R1 이
  holdout 이 아니게 되어 G5 자체가 성립하지 않는다.** 1,422행을 학습에 넣는 arm 은 **별도
  사전등록**을 요구한다.

### D.4.2 실행 커맨드 (확정, **본 문서는 실행하지 않는다**)

```
[1] python -c "from lpopt.model.al_retrain import refresh_distill_cache; \
      refresh_distill_cache('data/models/s1j', out_path='data/models/_v5_distill_soft.npz')"

[2] python -m lpopt.remote --input lpopt.inp push

[3] python -m lpopt.remote --input lpopt.inp train -- \
      --ensemble 5 --split S1j --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 \
      --epochs 150 --num-workers 8 --device auto \
      --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
      --cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
      --promote-max-asm-bu --promote-fxy \
      --init-from data/models/s1j --trunk-finetune-lr-mult 0.05 \
      --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
      --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
      --fxy-prior-on-predicted --fxy-select-weight 0.5 --warmup-epochs 2
```

C.2 의 [3] 대비 바뀐 것은 **네 곳뿐**이다: `--init-from s1i` → **`s1j`**,
`--freeze-trunk-cyclen` → **`--trunk-finetune-lr-mult 0.05`**(신규 플래그, D.7-③),
`--fxy-direct` → **`--fxy-prior-on-predicted`**, 그리고 deck 이 **`lpopt.inp`(GPU 0)** 이다.

> **`[1]` 은 생략 불가다.** `data/models/_v5_distill_soft.npz` 는 **`s1i` 로 만든 캐시**이며
> (mtime 2026-08-29 13:52:12, arm 2·arm 3 공용) arm 4 의 teacher 는 **`s1j`** 여야 한다.
> trunk 가 풀리는 arm 에서 distill 은 G1 의 주 방어선이므로, 낡은 teacher 로 돌리면
> **G1 을 옛 챔피언 쪽으로 끌어당긴다.** 재생성하지 않았다면 그 사실을 결과에 적고 판정을 보류한다.

> **GPU.** 본 arm 은 **`lpopt.inp`(`[remote] gpu = 0`)** 를 쓴다 — §3 의 상시 지시
> (*"사용자 지시 2026-07-24: GPU 0 고정, 재허가 전까지 auto 금지"*)와 충돌하지 않는다.
> arm 2·arm 3 이 `lpopt_gpu1.inp` 로 돌면서 남긴 **미해결 편차**(arm 3 결과보고서 §10.4)를
> arm 4 는 **반복하지 않는다.** GPU 1 로 돌리려면 orchestrator 가 재허가를 **먼저 기록**해야 한다.

> **선행조건 (C.2 승계).** 채점은 `rc`/`DONE` 과 per-cell 보정 6종이 **모두 존재하는 완료본**을
> pull 한 뒤에 한다. 채점 경로는 C.3 의 **FIXED serve path** 고정(= 현행 코드), 후보와 `s1j`
> **양쪽 다** 그 경로에서 적합된 보정을 지닌 채 채점한다(arm 3 §9.1 의 대칭 유지).

---

## D.5 데이터 — store 가 다시 앞서 있다 (등록)

| 항목 | C.6 채점 시점 | **본 개정 시점 (2026-08-31)** |
|---|---|---|
| `data/store/records.parquet` sha256 | `f38666e9…` (75,793행) | **`747d37ae2a50cc25742f98cc4770694354032ec6e060eeb34d54b75eb830cb50`** (**76,693행**, 22,782,850 B) |
| converged `f_xy` 라벨 | 6,218 (S1j 구축 시) | **7,645** (미수렴 포함 7,667) |
| S1j fold 귀속 | train 5,425 / val 793 / none 0 | **train 5,429 / val 794 / none 1,422** |
| `f_xy` 라벨 ≥ 100 인 cell | — | **15** (≥ 200: 10; 전체 127 cell) |
| VAL11 (holdout ≥ 20) | 11 cell / 486행 | **11 cell / 487행 — cell 집합 불변** |
| `data/splits/S1j.json` sha256 | `321950cb…` | **`321950cb8fc965118569b5afa7943f2f2b09b39b231da228a177eac86e7ba3b1` — 불변** |

- **cell 집합은 바뀌지 않았고**, 늘어난 1행(`T6_T4/f121` 147 → 148)은 판정을 움직이지 않는다:
  같은 하네스가 `s1j` 에 대해 794행에서 **0.0663 / 0.7907**, arm-3 채점기는 793행에서
  **0.066300 / 0.790392** 를 냈다. **바를 793행으로 되돌리지 않는다** — arm 4 는 **794행**으로
  채점하고, 대조군 `s1j` 도 **같은 794행**으로 다시 잰다(짝지어진 비교).
- **어느 fold 에도 없는 1,422행**은 §2.1 이 등록한 것과 같은 종류의 불일치이며, D.4.1 의
  결정에 따라 **학습에 들어가지 않는다** — 그중 R1 95행과 PINBU 25행은 **G5 / 동반 판독의
  채점 대상**이다(학습에 없으므로 누수가 아니다).

---

## D.6 **바 (구속력, arm 4)** — arm 3 을 레벨과 순위 **양쪽에서** 이겨야 한다

라벨된 S1j **VAL 794행**, cell 정의 (case_pair, feed), holdout 라벨 ≥ 20 인 **11 cell / 487행**.
대조군은 **`s1j`(= arm 3)** 이며, **같은 행·같은 코드·같은 보정 상태**에서 다시 측정한다.

| 게이트 | 지표 | arm 4 기준 | 출처 |
|---|---|---|---|
| **G1** (veto) | `gate-promote --check-only` | `pass == true`, ε = σ0·Φ⁻¹(0.95^(1/N)) (N·ε 기록), `blind_targets == []`, `unavailable == []` | **A.7 그대로, 무변경** |
| **G2‴** | MAE(`f_xy`), VAL n=794 | **< 0.066256** (arm 3, 794행 실측) | 본 개정 D.2.2 행 0 |
| **G3‴** | ρ̄ within-cell, 11 cell 비가중 평균 | **> 0.790696** (arm 3, 794행 실측) | 〃 |
| **G5** (신설) | **캠페인 내부 순위** — R1 95행의 Spearman ρ(pred, meas `F_xy`) | **> 0.488352** (arm 3) **이면서** paired 행-부트스트랩(2,000 reps, seed 0) CI 가 **0 을 배제** | 〃 |
| **G4** | 68% 커버리지, VAL n=794 | **∈ [0.55, 0.80]** | §4 G4 그대로 |

- **G2‴ · G3‴ · G5 · G4 는 AND 다.** 하나라도 놓치면 FAIL 이다(단 G4 에는 D.6.4 의 명시적
  예외가 하나 있으며, 그것도 여기 사전에 적혀 있다). G1 은 veto 다.
- **G1 이 이번에는 형식이 아니다.** §6 은 arm 1–3 에서 G1 이 통과할 것으로 기대한 근거를
  "trunk 가 `requires_grad=False` 이므로 legacy 가 움직일 경로가 거의 없다"로 적었다.
  **arm 4 는 그 논거를 스스로 제거한다.** 남는 방어선은 (i) cyclen 행 gradient-0 hook,
  (ii) distillation(0.4, teacher = `s1j`), (iii) `--f-r-rank-weight 0.1`, (iv) 낮은 trunk LR
  뿐이다. **G1 은 arm 4 의 1차 가설이며 실측으로만 답한다.**
- **G4 를 arm 4 에서 승격 조건으로 올리는 이유 (등록).** arm 3 은 G4 FAIL(0.831, over-wide)
  이었고 처분 규칙에 따라 **σ 서빙이 금지된 채** 승격되었다. 그 금지는 `min_fxy` 의 `F_xy_UCB`
  폭을 proxy 규약으로 대체하게 만드는 **실질적 제약**이며(arm 3 §5), 영구화되어서는 안 된다.
  arm 3 §7.3 은 초과분이 **앙상블 epistemic 항의 과대 합산**에서 온다고 측정했고(단일 멤버
  0.46–0.68 → 앙상블 0.83), trunk 를 푸는 arm 4 는 멤버 다양성을 바꾸므로 **이 축은 다시
  측정되어야 한다.**
- **동률/근소차 (G3‴).** Δρ̄ 가 `0 ± 0.05` 안이면 점 추정으로 판정하지 않는다. B.5/C.4 와
  동일하게 `lpopt.model.ab_paired.paired_cell_bootstrap` 의 cell-clustered paired BCa
  (11 cell, 2,000 reps, seed 0)를 산출하고 CI 가 0 을 포함하면 **tie = FAIL** 로 기록한다.
- **G5 의 CI 는 조건이지 tie-breaker 가 아니다.** R1 은 단일 cell 이므로 cell-clustering 이
  불가능하다 — 95행의 **행 부트스트랩**(2,000 reps, seed 0, 같은 행 위의 paired Δρ)을 쓴다.
  점 추정만으로는 G5 를 통과할 수 없다.
- **G5 의 바 0.488352 는 `s1j` 를 R1 95행에서 재측정한 값이다.** minfxy r1 결과보고서 §6.3 이
  기록한 **+0.392** 는 **wave 별 서빙 checkpoint 13종**으로 잰 값이므로 서로 다른 양이다
  (그 문서의 판정은 그대로 유효하다). arm 4 도 **단일 checkpoint 로** 같은 95행에서 재는 이상,
  대조군은 **0.488352** 여야 한다. 두 수를 결과보고서에 **함께** 적는다.
- **A.5 / A.6 의 바(0.0463 / 0.8944)와 B.5/C.4 의 바(0.0767 / 0.7263)는 폐기되지 않는다 —
  함께 보고한다.** 전자는 *측정* `F_r` 을 받는 이상적 추정기의 자리이고(D.2.2 3–4행),
  후자는 arm 2·3 이 채점된 자리다. **세 읽기를 모두 판정표에 적는다.**

### D.6.1 동반 판독 (판정 아님, 반드시 인쇄)

| 항목 | arm 3 (`s1j`) 실측 | arm 4 |
|---|---:|---|
| R1jc(65행) ρ | **+0.2048** | TBD |
| PINBU(25행) ρ | **−0.4871** | TBD |
| PINBU(25행) MAE | 0.0274 | TBD |
| R1 MAE | 0.0562 | TBD |
| 서빙 PROXY 재측정 (같은 794행) MAE / ρ̄ | 0.0721 / 0.7156 | TBD |
| ρ(예측 `F_r`, 측정 `F_r`) — R1 / R1jc / PINBU | **+0.5096 / +0.1735 / −0.3993** | TBD |
| `--fxy-prior-on-predicted` 의 **prior 표류** | (arm 3 은 direct → 해당 없음) | meta 의 `(a,b)` vs 학습 종료 후 재적합 `(a,b)` |

> 마지막 줄의 근거: `train.refit_fxy_prior_on_predicted` 는 `--init-from` 직후 **한 번만**
> 적합된다. trunk 가 풀리면 `mu[f_r]` 이 학습 중 이동하므로 그 prior 는 **낡을 수 있다**.
> 잔차가 표류를 흡수하는 것이 설계 의도이나 **가정이지 결과가 아니다** — 채점자가 최종
> 체크포인트로 같은 train 행에서 재적합해 비교하고 기록한다(read-only). 두 `(a,b)` 가 크게
> 갈리면 그 자체가 arm 5 의 근거다.

### D.6.2 오프라인 forecast (게이트 아님, **사후에 바를 옮기지 않는다**)

arm 4 의 가장 가까운 오프라인 대응물은 **b4 = (b-aff/RAW) GBM** 이다:
**VAL MAE 0.0666 / ρ̄ 0.8370 / R1 ρ +0.5283 / R1jc +0.2586 / PIN +0.2497.**

| 게이트 | 바 | b4 의 자리 | forecast |
|---|---:|---:|---|
| G2‴ | < 0.066256 | 0.066621 | **아슬하게 미달 (+0.00037)** |
| G3‴ | > 0.790696 | 0.8370 | 점추정 통과 (Δ +0.045), 단 BCa CI 가 0 을 포함 → **tie 위험** |
| G5 | > 0.488352 & CI ∌ 0 | +0.5283 (Δ +0.040, CI [−0.028, +0.111]) | **점추정 통과, CI 미달** |
| G4 | ∈ [0.55, 0.80] | (오프라인 probe 는 σ 를 내지 않는다) | 예측 불가 |

**즉 오프라인 증거는 arm 4 를 "통과 유력"이라고 말하지 않는다.** 그럼에도 투입하는 근거는
두 가지이며, 둘 다 사전에 적어 둔다:
1. **오프라인 probe 는 단일 회귀기이고 arm 4 는 5멤버 앙상블이다.** arm 3 §7.3 이 같은 축에서
   측정한 앙상블 효과는 **MAE −0.009 / ρ̄ +0.03** 이다. 그 폭이면 b4 의 0.0666 은 G2‴ 안으로
   들어오고 G5 의 CI 도 0 에서 떨어질 수 있다 — **가능성이지 예측이 아니다.**
2. **GBM 은 trunk 를 푸는 arm 의 상한이 아니라 대리물이다**(B.2 의 등록된 읽기). 실제
   fine-tune 은 임베딩 자체를 바꾸므로 b4 보다 나을 수도, 나쁠 수도 있다.

**실측이 b4 의 자리에서 크게 벗어나면 그 자체가 조사 대상이다.**

### D.6.3 판정 표 (실행 후 채울 것)

| 게이트 | 기준 | 측정값 | 판정 |
|---|---|---|---|
| G1 (veto) | `pass == true`, N·ε 기록, `blind_targets == []` | TBD | TBD |
| **G2‴** | MAE < **0.066256** (n=794) | TBD | TBD |
| **G3‴** | ρ̄ > **0.790696** (11 cell, tie 시 BCa → tie = FAIL) | TBD | TBD |
| **G5** | R1 ρ > **0.488352** **&** 행-부트스트랩 CI ∌ 0 (n=95) | TBD | TBD |
| **G4** | 커버리지 ∈ [0.55, 0.80] | TBD | TBD |
| (참고) | B.5/C.4 바 0.0767 / 0.7263 | TBD | (승격 조건 아님) |
| (참고) | A.5/A.6 바 0.0463 / 0.8944 | TBD | (승격 조건 아님, 도달 불가) |
| (기록) | `[1]` distill 캐시를 **`s1j`** 로 재생성했는가 | TBD | — |
| (기록) | deck / GPU (`lpopt.inp`, gpu 0) | TBD | — |
| (기록) | meta `fxy_head.mode` == `prior_residual`, `prior_source` == `predicted`, `trunk_finetune_lr_mult` == 0.05 | TBD | — |
| (기록) | 채점 `model_api.py` / `featurize.py` sha256 | TBD | — |

### D.6.4 처분 규칙 (사전 확정)

- **G1 PASS · G2‴ · G3‴ · G5 · G4 모두 PASS** → **`s1k`** 로 승격, deck `model_dir` 갱신,
  **head σ 서빙 금지를 해제**(G4 를 실제로 통과했으므로). 새 체크포인트의 `ensemble.json` 에는
  `fxy_head.serve_sigma = "barred"` 를 **쓰지 않는다**.
- **FAIL-G1** → **reject.** trunk 를 푼 arm 이 legacy 를 회귀시켰다는 것이고, 재시도는 새
  사전등록(다른 LR 배수 등)을 요구한다. **본 개정의 0.05 를 사후에 낮추어 재실행하지 않는다.**
- **G1 PASS · G2‴/G3‴/G5 중 하나라도 FAIL** → 챔피언 **`s1j` 유지**, arm 4 는 배포하지
  않는다(shadow 포함). 근거는 B.5 그대로: 현행 서빙 추정기를 못 이기는 head 는 그 열화판이다.
- **G4 만 FAIL (G1·G2‴·G3‴·G5 전부 PASS)** → **승격하되 σ 서빙 금지를 `s1j` 와 동일하게
  유지**한다. 이것이 AND 규칙의 **유일한 예외**이며 사전에 적어 둔다(사후 완화가 아니다).
- **arm 4 의 결과로 어떤 후속 arm 도 자동 발동하지 않는다** — arm 5((d) log-ratio) 는 별도
  개정을 요구한다.

---

## D.7 별도 구현 과제 (본 문서는 구현하지 않는다)

번호 ③ 은 **arm 4 실행의 선행조건**이고, ①·② 는 arm 4 와 독립이다.

### ③ `--trunk-finetune-lr-mult <m>` — arm 4 [3] 이 요구하는 유일한 신규 플래그

현재 `--freeze-trunk-cyclen` 은 **두 가지를 묶어서** 한다: (i) `_FREEZE_WHOLE_MODULES`
(`stem`/`blocks`/`films`/`head_trunk`/`conv_head`) 전체의 `requires_grad_(False)` + optimizer
제외, (ii) `mu_head`/`log_sigma_head`/`quantile_head` 의 **cyclen 행 gradient-0 hook**.
arm 4 는 (ii) 를 유지한 채 (i) 만 풀어야 한다. 현재 CLI 에는 **`--lr` 자체가 없고** deck 도
`lr` 을 싣지 않으므로(`TrainConfig.lr = 3.0e-4` 가 유일한 출처), 배수 플래그가 필요하다.

| # | 변경 | 파일 |
|---|---|---|
| 1 | `TrainConfig.trunk_finetune_lr_mult: float = 0.0` (0 = OFF, 기존과 byte-identical) | `lpopt/model/train.py` |
| 2 | `_apply_freeze_trunk_cyclen` 을 분기: `mult > 0` 이면 `requires_grad_(False)` 루프를 **건너뛰고** cyclen hook 만 건다 | 〃 |
| 3 | `_build_member_optim` 에 trunk 전용 param group (`lr = lr * mult`) 추가. 기존 `weight_decay=0` 그룹(행-마스크 head)은 그대로 | 〃 |
| 4 | CLI `--trunk-finetune-lr-mult`, `--init-from` 필수, `--freeze-trunk-cyclen` 과 **상호 배타**(둘 다 주면 `ap.error`) | 〃 |
| 5 | meta 에 `trunk_finetune_lr_mult` 기록; trainable `n_params` 가 실제로 늘어나는지 학습 로그에 인쇄 | 〃 |
| 6 | `_graft_appended_target_rows` 는 **손대지 않는다** — `s1j` 는 이미 9열이므로 graft 가 발생하지 않는다 | (확인만) |

**요구 시험** (`tests/test_fxy_head.py` 신규 §13):
(a) `mult = 0` 이 현행과 byte-identical, (b) `mult > 0` 에서 trunk param 이 `requires_grad=True`
이고 optimizer group 의 lr 이 정확히 `lr * mult`, (c) **cyclen 행이 여전히 학습되지 않을 것**
— 4-epoch 실학습 후 `mu_head.weight[_CYCLEN_IDX]` 와 `bias[_CYCLEN_IDX]` 가 champion 과
bit-identical, (d) `--freeze-trunk-cyclen` 과 동시 지정 시 `SystemExit`,
(e) `--trunk-finetune-lr-mult` 만 주고 `--init-from` 을 빼면 `SystemExit`.

### ① `fuel_types` Gd/격자 backfill — (c) 를 되살리려면 이것이 먼저다

`ga80` 70 type 중 `gd_wt`/`gd_u_enr` 는 **0건**, `zone_pin_count` 는 `ga80`·`paramA`
**144 type 전부 0건**. `n_gd` 는 ga80 에서 51.4%. H4 가 성립한 3 cell 이 전부 ga80 이므로
**backfill 없이는 cond v9 가 넣을 값을 갖지 못한다.** 과제는 스키마가 아니라 harvest 다:
`data/store/fuel_types.parquet`
(sha256 `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137`, 64,343 B) 의
세 컬럼을 ga80 설계 원본에서 채우는 것. **완료 후에야** (c) 의 오프라인 재측정(본 하네스를
그대로 재실행 — GD 블록·ablation·`descriptor_stats` 가 이미 들어 있다)이 의미를 갖고,
그때 이득이 확인되면 **cond v9 를 별도 사전등록**한다. 그 v9 는 새 체크포인트를 요구하며
`s1j`/`s1k` 와 **호환되지 않는다**(§9 의 "7열 계약" 주의와 같은 종류의 비용).

**참고로 등록해 두는 v9 의 형태 (구현하지 않는다).** 채널을 늘리는 최소 형태는
`_V5_DROPPED_CHANNELS` 에서 `origin_n_gd`/`origin_gd_wt` **2개를 되살리는 것**이 아니라 —
그것은 2026-07-20 의 "독봉 불가지론" 지시를 되돌리는 것이다 — **poison-agnostic 한 반경
모멘트 2개를 GLOBAL 로 추가**하는 것이다(예: 흡수체 적재 가중 반경 centroid 와 그 fresh 한정판).
어느 쪽이든 본 개정의 범위 밖이며, 위 backfill 없이는 두 형태 모두 ga80 에서 0 을 낸다.

### ② (d) log-ratio 파라미터화 — arm 5 후보

`net.PosValNet._compose_fxy` 는 **z 공간에서 덧셈**으로 합성한다
(`mu[f_xy] += a·mu[f_r].detach() + b`). log-ratio 는 **물리공간 곱셈**
(`f_xy = exp(residual) × F_r_pred`)이므로 (i) 네트가 `tmean`/`tstd` 를 알고 물리공간에서
합성하거나 (ii) 표적을 `log f_xy` 로 바꾸고 de-normalisation 을 그에 맞추어야 한다.
영향 파일: `net.py`(합성), `train.py`(표적 변환·prior 적합·meta), `dataset_torch.py`(표적 프레임),
`calibrate.py`(σ 곡선의 단위), `model_api._ensemble_raw`(역변환). **arm 4 가 G5 를 놓친
경우에만** 착수하며 새 사전등록을 요구한다. 오프라인 기대이득은 **ΔVAL ρ̄ +0.008**(tie 밴드 안).

---

## D.8 §8 이 금지한 주장 — 본 개정에서의 준수

- **"ratio proxy 를 순위에서 이기는 것이 arm 4 의 목표"라고 적지 않는다.** D.2.3 #1 이
  측정으로 보였듯 **서빙 ratio proxy 의 R1 순위(+0.4664)는 이미 arm-3 head(+0.4884)보다 낮다** (정확값 0.466428 vs 0.488352).
  pinbu 의 "5배" 는 **측정 `F_r`** 에 대한 진술이며 서빙 경로의 경쟁자가 아니다.
- **"trunk 를 풀면 순위가 오른다"고 주장하지 않는다.** GBM 행은 대리물이고, D.6.2 는 forecast 가
  두 게이트에서 아슬하다는 것을 **사전에** 적었다.
- **head 의 `f_xy` 로 미측정 core 의 `F_xy ≤ 1.65` 적합성을 선언하지 않는다** (§8 그대로).
- **(c) 가 "쓸모없다"고 주장하지 않는다** — 측정된 것은 *현재 fuel table 로는 만들 수 없다*
  는 것이며, D.7-① 이 그 조건을 적는다.
- **§2.1 / D.5 의 fold 밖 라벨을 근거로 한 학습 성능 주장은 계속 금지된다.** R1·PINBU 는
  **학습에 들어가지 않으므로** 순위 판독의 대상일 뿐, "모델이 이 행들을 학습했다"는 주장의
  근거가 아니다.
- **바를 사후에 옮기지 않는다.** G2‴/G3‴/G5 의 세 상수(0.066256 / 0.790696 / 0.488352)는 arm 4 의
  어떤 산출도 관측하기 전에 `s1j` 를 같은 슬라이스에서 재측정해 고정했다.

**D.6.3 판정 스탬프(2026-08-31):** arm 4 (`20260902_122746`) — G1 PASS(worst drop 0.017241, ε 0.1388/0.1422) · G2‴ **FAIL** 0.067404 (바 0.066256) · G3‴ PASS 0.827749 (CI [+0.0047,+0.1125]) · G5 **FAIL** ρ 0.535126 (점 통과, paired CI [−0.053,+0.138] ∋ 0) · G4 **FAIL** 0.872796 → D.6.4: s1j 유지, arm 4 미배포. 결과: `fxy_head_results_arm4_20260831.md`.

---

# Amendment E — 2026-09-03 · **arm 5** (`f_xy` 합성행 위의 within-cell **pairwise** 랭킹 항 + 랭킹 조항 G6 · parity veto G7) · 실행 전 개정

본 절은 §1–§10 · Amendment A · B · C · D 를 **재작성하지 않는다.** 아래 항목만 **덮어쓰며(binding)**,
언급되지 않은 모든 조항(H1, G1 의 형식·veto 성격, A.7 의 ε 공식, §8 금지 주장, 누수 규칙,
C.3 의 FIXED serve path 고정)은 그대로 유효하다. **arm 1 · 2 · 3 · 4 의 판정은 확정·불변이며 본
개정이 바꾸지 않는다.** arm 4 는 D.6 바로 채점되어 G2‴/G5/G4 FAIL → `s1j` 유지·미배포로 종결되었고
(D.6.3 판정 스탬프, 2026-08-31), 그 판정은 그대로 둔다.

본 개정이 답해야 하는 상위 지시는 D.6.4 의 "arm 5 는 별도 개정을 요구한다"가 아니라
**`minfxy_E1E2_f121_r2_results_20260831.md` §11.5 의 판정**이다:

> **"head 승격 기준 G1–G4 에 within-cell ranking 조항을 추가하기 전까지 어떤 새 head 도 wave 를
> 랭크하지 않는다."** (r2 RANK 게이트 3/3 FAIL: R-a +0.2526 < +0.30 · R-b −0.1157 < 0 ·
> R-c 승 1/13; 같은 76행에서 proxy 는 **+0.3060** 으로 그 선을 넘었다. head 는 **level estimator 로
> 강등**되었고 r3 는 `s1i`/proxy 로 랭크한다.)

따라서 본 개정의 **1차 산출물은 체크포인트가 아니라 조항 G6** 이며, arm 5 는 그 조항이 실제로
무엇을 가려내는지 시험하는 첫 후보다.

> **등록해 두는 시점.** 본 개정은 **arm 5 학습을 투입하기 전에** 작성되었다. 아래 E.1–E.2 의 모든
> 수치는 **오프라인 실험**(챔피언 `s1j` 와 arm-4 체크포인트 `runs/20260902_122746` 의 예측·임베딩
> 위의 probe/ridge, HOST_238 GPU 1, read-only)의 산출이며, **arm 5 후보는 아직 존재하지 않는다**
> (E.5.4 판정표는 전부 `TBD`). 바는 arm 5 의 어떤 산출도 관측하기 전에 확정되었고 사후에
> 이동하지 않는다. **arm 4 의 실측값은 전부 "설계 시점에 알고 있던 값"으로 명시한다** — 그것으로
> 바를 정한 곳(E.5 G2⁵ 의 비열등 마진)은 그 사실과 반론을 함께 적었다(E.10-7).

---

## E.1 왜 arm 5 인가 — 강등의 원인은 feature 가 아니라 **목적함수**다 (측정)

세 판단 중 하나를 고르는 문제였다: (i) 랭킹 축이 feature space 에 없다(r2 §11.3 의 추측),
(ii) 표적 파라미터화가 틀렸다(D.7-② log-ratio), (iii) 표적은 있는데 **레벨 목적함수가 그것을
버린다**. 아래 세 실험이 (iii) 을 지목하고 (i) 을 **반증**한다.

### E.1.1 진단 — basin 안에서 무엇이 무너지는가

`data/reports/_wf_scratch/wcr2b_power.log` (슬라이스 분산) · `wcr3_ceiling.log` (학습 가능 천장):

| 슬라이스 | n | 측정 `f_xy` sd | head 레벨오차 \|err\| | ρ(**측정** `F_r`, `f_xy`) | ρ(**예측** `F_r`) | ρ(head) |
|---|---:|---:|---:|---:|---:|---:|
| R1 all (`T6_T4`/f121) | 95 | 0.0932 | 0.0491 | +0.8984 | +0.4971 | +0.5347 |
| R1jc elite | 65 | 0.0299 | 0.0327 | +0.7201 | +0.2028 | +0.2632 |
| R2 all (`E1_E2`/f121) | 99 | 0.1612 | 0.0262 | +0.7129 | +0.6963 | +0.6608 |
| **R-a joint-clean** | 76 | 0.0253 | 0.0140 | +0.3786 | +0.3650 | +0.2858 |
| **R-b exploit** | 64 | **0.0114** | **0.0117** | **+0.1738** | +0.0644 | −0.0681 |

**두 줄로.** (1) `exploit` slot 에서 추정기 자신의 레벨오차(0.0117)가 **정렬해야 할 분산 전체
(0.0114)와 같은 크기**다. 레벨 목적함수는 이것을 고칠 유인이 없다 — cell 평균이 맞으면 cell 내
분산을 줄이는 쪽이 MAE 최적이다. (2) **측정** `F_r` 조차 R-b 를 +0.1738 로밖에 정렬하지 못한다.
즉 `F_r` 에 anchor 된 어떤 랭커도 exploit slot 에서 +0.17 근방이 천장이며, `f_xy` 잔차가 cell
안에서 anchor 를 **떠날 수 있어야** 그 위로 간다. 이것이 pairwise 항이 보상하고 레벨 항이
보상하지 않는 바로 그 자유도다.

### E.1.2 반증 시험 — **순서는 trunk 임베딩 안에 있다** (r2 §11.3 의 추측을 뒤집는다)

basin 행 위에서 **leave-one-WAVE-out** ridge (자기 자신을 예측하지 않는다), arm-4 `head_trunk`
384차원 임베딩 vs cond-v8 20 globals (`wcr3_ceiling.log`):

| 슬라이스 | n | LOWO ρ (emb384) | ρ (globals 20) | emb+globals | 서빙 head | proxy |
|---|---:|---:|---:|---:|---:|---:|
| **R-a joint-clean** | 76 | **+0.4688** | −0.1899 | +0.4717 | +0.2858 | +0.3650 |
| **R-b exploit** | 64 | **+0.2567** | −0.4918 | +0.2485 | −0.0681 | +0.0644 |
| R2 all | 99 | +0.7092 | −0.2046 | +0.7079 | +0.6608 | +0.6963 |
| R1jc (5-fold CV) | 65 | +0.5631 | — | +0.5636 | +0.2632 | +0.2028 |

**축은 임베딩에 있고, 20 globals 에는 없다.** r2 §11.3 의 "랭커가 갈라야 할 축이 feature space 에
없다"는 추측은 이 표로 **반증**된다. 버리는 것은 목적함수다.

### E.1.3 forecast device — 동결 임베딩 probe (게이트 아님, D.6.2 의 전례를 승계)

S1j **train fold 에서만** 적합한 선형 probe, 3 seed, GPU 1 (`wcr1b_probe.log` / `.csv`).
P0 = 레벨 전용(대조), HL30 = 제안 구성(w=3.0, gate cell, min_gap 0.005, margin 0.1 z,
`min(f_xy) ≤ 1.60` 쌍 ×3):

| 추정기 | VAL MAE | VAL11 ρ̄ | ρ̄ elite q.50 | R1 | R1jc | R2 | **R-a** | **R-b** | R-c(regret) | PIN | span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **P0 레벨 전용** | 0.07355 | 0.7763 | 0.4829 | +0.6159 | +0.3260 | +0.5930 | **+0.1405** | **−0.3010** | 0.0076 | −0.3607 | 0.457 |
| H03 hinge w=0.3 | 0.07319 | 0.7812 | 0.5067 | +0.6289 | +0.3591 | +0.6044 | +0.1643 | −0.2593 | 0.0062 | −0.3421 | 0.377 |
| H10 hinge w=1.0 | 0.07295 | 0.7814 | **0.5256** | +0.6306 | +0.3582 | +0.6310 | +0.2226 | −0.1662 | 0.0050 | −0.3348 | 0.426 |
| H30 hinge w=3.0 | 0.07301 | 0.7818 | 0.5033 | +0.6344 | +0.3623 | +0.6748 | +0.3164 | −0.0178 | 0.0082 | −0.3198 | 0.554 |
| **HL30 w=3.0 low1.60×3** | 0.07382 | 0.7754 | 0.5090 | +0.6244 | +0.3600 | **+0.7494** | **+0.4826** | **+0.2515** | 0.0084 | −0.3333 | 0.586 |
| HE30 elite-pair ×3 | 0.07355 | 0.7776 | 0.5154 | +0.6251 | +0.3460 | +0.7069 | +0.3869 | +0.0966 | 0.0085 | −0.3343 | 0.609 |
| HEO30 elite-pairs only | 0.07594 | 0.7692 | 0.5170 | +0.6048 | +0.2761 | +0.7358 | +0.4551 | +0.2188 | 0.0109 | −0.4434 | 0.662 |
| CEN30 centred w=3.0 | 0.07543 | 0.7851 | 0.4961 | +0.6272 | +0.3513 | +0.5982 | +0.1523 | −0.2829 | 0.0068 | −0.3715 | 0.345 |
| **LN30 listnet w=3.0** | 0.07335 | 0.7758 | 0.4856 | +0.6138 | +0.3267 | +0.5918 | **+0.1370** | **−0.3058** | 0.0076 | −0.3893 | 0.435 |
| — HEAD arm4 (캐시) | 0.06741 | 0.8278 | 0.4659 | +0.5347 | +0.2632 | +0.6608 | +0.2858 | −0.0681 | 0.0138 | +0.1648 | 0.320 |
| — proxy arm4 (캐시) | 0.15065 | 0.7692 | 0.3719 | +0.4971 | +0.2028 | +0.6963 | +0.3650 | +0.0644 | 0.0115 | +0.1889 | 0.365 |

**표에서 읽히는 것 — 그리고 표에 있는 불리한 줄까지 함께 등록한다.**

1. **P0 가 서빙 head 의 실패 양식을 재현한다** (R-a +0.1405 / R-b −0.3010 ↔ 서빙 head +0.2858 /
   −0.0681). 같은 임베딩·같은 device 에서 hinge 만 더하면 R-a +0.4826 / R-b +0.2515 로 움직인다.
   **원인은 표현이 아니라 목적함수**라는 주장의 실험적 근거는 이것이다.
2. **용량 반응이 단조**다 — R-a: 0.1405 → 0.1643 → 0.2226 → 0.3164 (w=0 → 0.3 → 1 → 3, boundary
   가중 없이). 운 좋은 draw 가 아니다.
3. **HL30 은 E.1.2 가 잰 학습 가능 천장에 붙는다** (R-a +0.4826 vs LOWO +0.4688; R-b +0.2515 vs
   +0.2567). 이 축에서 더 짜낼 것은 별로 없다.
4. **레벨 비용은 +0.00027, ρ̄ 비용은 −0.0009** (P0 → HL30).
5. **listwise 는 무효 — 등록된 음성 결과.** LN10/LN30 (R-a +0.1392/+0.1370, R-b −0.3027/−0.3058)
   과 cell-centred CEN10/CEN30 (+0.1402/+0.1523) 은 P0 와 구별되지 않는다. **"listwise/pairwise"
   각도는 pairwise 로 확정된다.**
6. **불리한 줄 ①: R-c(regret, 작을수록 좋음)는 hinge 로 오히려 나빠진다** — P0 0.0076 → HL30
   0.0084. 랭킹 ρ 의 이득이 wave 단위 regret 으로 **번역되지 않는다.** R-c 가 report-only 인 두
   번째 이유이며(첫째는 검정력 0.17), G6c 를 두는 이유다.
7. **불리한 줄 ②: PINBU 에서 probe 전 구성이 −0.32 ~ −0.44** 인데 서빙 head 는 +0.1648 이다.
   probe 는 그 슬라이스에서 신뢰할 수 없다. PINBU 는 게이트에서 제외한다(E.4.1 에 두 번째,
   더 결정적인 이유가 있다).
8. **불리한 줄 ③: HL30 은 VAL elite band(ρ̄ q.50)의 argmax 가 아니다** — H10 이 0.5256 으로 위다.
   이는 G6a 가 이 구성을 고른 기준이 **아니라는** 증거이며, G6a 를 config-blind 조항으로 쓰는
   근거다(E.5.2).
9. **seed sd 정정.** "모든 랭킹 열에서 sd ≤ 0.004" 는 **사실이 아니다** — HL30 의 R-b sd 0.0071,
   HEO30 의 ρ̄ elite sd 0.0104, LN10/LN30 의 PIN sd 0.0241. R-a 열에서만 sd ≤ 0.0038 이다.

### E.1.4 감독 공급 — 손실이 실제로 볼 쌍이 있는가 (측정)

`wcr5_cellcensus.log` (S1j train fold, 라벨된 `f_xy` 5,429행) · `arm5_pairsupply.json`:

| 항목 | gate cell `(case_pair, feed)` | legacy cell `(feed, e_core-bin, dataset)` |
|---|---:|---:|
| cell 수 / ≥2행 cell | 119 / 108 | 78 / 68 |
| 참여 행 | 5,418 (train 의 **99.8%**) | 5,419 (99.8%) |
| 정렬 가능 쌍 @ `min_gap` 0.005 | **526,578** | 707,693 |
| 그중 elite 쌍 (양쪽이 cell 하위 절반) | **124,084** | 167,572 |
| batch 당 같은-cell 라벨쌍 (측정 평균 / p05 / zero-frac) | **195.3 / 118.8 / 0.000** | 257.5 / 180.8 / 0.000 |
| 최대 cell | `E1_E2/f121` **614**, `T6_T4/f121` **595** | 804 |

**cell 정의는 재사용이 아니라 실제 코드 변경이다.** legacy `cyclen_cell` 은 43/78 cell 에서
gate cell 을 **2개 이상 섞고**(최대 5), gate cell 28/119 이 legacy cell 여러 개로 **쪼개진다**
(최대 6). 게이트가 재는 분할과 손실이 최적화하는 분할이 다르면 안 된다 → 신규 `fxy_cell`
(E.8-②). **쌍 공급이 0 인 batch 는 측정상 존재하지 않는다**(zero-frac 0.000).

---

## E.2 준우승안에서 이식한 것 (명시적 크레딧, 그리고 이식하지 **않은** 것)

### E.2.1 **serve-path parity 결손 → 신설 veto G7** (arm5-power 발견, arm5-logratio 가 독립 재현)

두 개의 독립 하네스가 같은 것을 보았다: **캠페인이 기록한 예측이 같은 체크포인트의 재채점과
일치하지 않는다.** `wf5_arm5_eval_evidence_20260903.txt` (r2 100 선택행 중 store 매칭 99행):

| 항목 | 값 |
|---|---|
| `selection.json` vs 재채점 `f_xy` | mean \|Δ\| **0.01383**, max **0.06831**, ρ **0.7277** |
| 같은 행의 `F_r` | mean \|Δ\| **0.01472** — **같은 크기** ⇒ 결손은 head 가 아니라 **상류(패턴)** 에 있다 |
| wave 0 | mean **+0.00000**, sd **0.00001**, 8/8 — **정확** |
| wave 1–12 | 평균 −0.0200 … +0.0233 로 흩어진다 |
| σ | 균일 +0.12205 (acquisition 의 `inflated` 인자 — 결손 아님) |
| slot 별 ρ (served → 재채점 s1j) | exploit −0.1157 → −0.0064 · explore +0.3826 → +0.4130 · control +0.9182 → +0.8909 |

가장 검약한 설명은 **wave 0 이후 생성된 후보에서 store 에 기록된 패턴이 채점된 패턴과
byte-identical 하지 않다**는 것이다. 진단되기 전까지 **캠페인 기록 예측은 게이트 입력으로
허용되지 않는다** — 그런데 r2 의 RANK 게이트(R-a/R-b/R-c)가 바로 그 경로에서 계산되었다.

> **그러나 이것으로 r2 판정을 다시 열지 않는다 (구속력).** 재채점하면 R-a 의 head_s1j 는
> +0.2526 → **+0.3260** 으로 등록선 +0.30 을 넘지만, **같은 76행에서 proxy 도 +0.3060 →
> +0.4049 로 함께 오른다.** head−proxy 의 부호와 순서는 **바뀌지 않는다**(−0.053 → −0.079).
> r2 의 NULL 과 RANK FAIL 은 그대로 유효하다. 재채점 수치를 "head 가 사실은 선을 넘었다"로 읽는
> 것은 E.9 가 금지하는 주장이다.

**G7 의 성격 (등록).** parity 결손은 (i) **G6 채점을 무효화하지 않는다** — G6 는 `selection.json`
이 아니라 FIXED serve path 재채점으로만 계산되기 때문이다. (ii) 그러나 **배포(shadow 포함)를
막는다** — shadow ranker 는 서빙 경로가 기록한 열을 남기는 장치이고, 그 경로가 재현되지 않으면
그 열은 해석 불가다. 따라서 **G7 은 배포 veto 이지 학습/채점 veto 가 아니다.**

### E.2.2 **campaign 층화 + row pooling 금지** (arm5-logratio, 세 안의 측정이 일치)

elite 행을 cell 을 가로질러 **pool 하면 안 된다**. `wcr2b_power.log`:
`POOLED elite (R1jc+R-a+PINBU, n=141)` 에서 head−proxy 는 **Δ +0.4697 [+0.3192, +0.6231]** 이지만
같은 값이 R-a 에서 −0.0793, R1jc 에서 +0.0604 다. **cell 간 레벨 차가 cell 내 랭킹 능력으로
위장한 것**이다. G6 의 모든 통계는 **슬라이스별 within-cell ρ 를 계산한 뒤 평균**하며, 행을
pool 하지 않는다.

### E.2.3 **R-c 는 report-only, 검정력을 함께 인쇄** (세 안 합의)

`wcr2b_power.log`: 13 wave 부호검정은 참 승률 0.6 에서 검정력 **0.17**, 0.7 에서 0.42.
26 wave 에서도 0.23 / 0.63 이다. **13 wave 로는 어떤 결론도 나오지 않는다.** R-c 를 구속력 있는
AND 게이트로 두는 것은 통과할 수 없는 게이트를 등록하는 것이다.

### E.2.4 **per-cell affine `f_xy` 보정 — 측정으로 기각** (arm5-power 의 음성 결과, 채택)

`wf5_arm5_eval_evidence_20260903.txt`:

| 검증 | 결과 |
|---|---|
| 랭킹 불변성 | R2jc head_arm4 **+0.285781** ↔ headcal **+0.285781**, R1jc **+0.263161** ↔ **+0.263161** (양의 affine 은 단조) |
| 레벨 | VAL794 0.06741 → **0.06838**, R1 0.04912 → **0.05403** (악화) |
| 근거 | r2 §6.2 — cell 별 (a,b) 는 라운드 간에도 전이되지 않는다 (`E1_E2/f121` joint-clean 기울기 1.4781 → 0.9052) |

> **arm 5 는 per-cell `f_xy` 보정을 도입하지 않는다.** 랭킹을 못 고치고 레벨을 악화시킨다.

### E.2.5 **fold C(어느 split 에도 없는 1,521행) 와 σ cross-fit — 제한적으로 채택**

`lpopt/model/folds.py` 는 이 집합을 이미 **fold C `new_unseen`** 으로 정의하고
`UNCONTAMINATED_FOLD = "C"` 로 표시하며, 동시에 다음을 **의무화**한다:

> "A large share of fold C is model-PROPOSED (`alsearch*`) … Metrics conditioned on a model's own
> proposals are not the same estimand as metrics on independent production, so `proposal_mask`
> splits them and callers report both." (`folds.py` docstring / `proposal_mask`, 측정 근거
> "peak within-cell ρ 0.646 vs 0.802")

이 때문에 **fold C 를 레벨 게이트 슬라이스로 쓰지 않는다.** 그 행들은 이전 서빙 모델이 자기
예측으로 골라 계산시킨 후보이고, 선택자(s1j) 가 optimizer's curse 를 먹는다 — 실제로 fold C 에서
s1j bias −0.03261 vs arm4 −0.00254 이며, 이 비대칭이 그대로 "arm4 가 레벨에서 이겼다"로 읽힌다.
**G2 는 VAL794 에 남는다**(슬라이스 쇼핑 금지). fold C 는 (i) **동반 판독**으로, (ii) **config-blind
랭킹 슬라이스의 공급원**으로만 쓰며, 언제나 `proposal_mask` 분할을 함께 인쇄한다.

**σ.** arm5-power 의 cross-fit 스칼라(k 0.663 → VAL 0.733)는 제출된 산출물에 **없다**; 제출물에
있는 것은 TRAIN 적합 k 가 커버리지를 **바닥 아래로** 떨구는 표다 (arm4 k 0.4095 → VAL 0.5164,
per-cell k → 0.4975; s1j k 0.5265 → 0.5630). 따라서 **G4 는 D 그대로 두고**(예상 FAIL,
`serve_sigma = "barred"` 유지), σ 보정은 **별도 구현 과제 E.8-⑩** 로 등록하되 **cross-fit 표를
먼저 공표할 것**을 조건으로 단다.

### E.2.6 **(d) log-ratio 의 처분 — D.7-② 는 arm 5 가 아니다** (기각 사유를 등록)

D.7-② 는 "arm 4 가 G5 를 놓치면 arm 5 후보"라고 적었다. 측정 결과 그 후보는 arm 5 가 되지
않는다:

- 제안된 이득의 **전량이 anchor 를 측정 `F_r` 로 바꾸는 데서 나온다**(파라미터화 단독은 손해:
  own-anchor 에서 logratio +0.4161 < affine +0.4724). 그런데 그 증거는 **arm-4 의 동결 trunk**
  위에서만 성립하고(`s1j` 동결 trunk 에서는 λ=1 이 proxy 를 소수 4자리까지 재현), arm 5 는
  그 trunk 를 갖지 않는다.
- 제안 자신의 산출물(`arm5_logratio_a5rank_arm4.json`)에 **절대표적 GBM 대조군**(`CTL_ABS_GBM_emb`)
  이 들어 있고 R-a +0.5084 를 낸다 — 주장된 +0.198 이득의 대부분이 "arm-4 임베딩을 GBM 으로 읽으면
  나온다"는 뜻이다. 이 대조군은 제안 본문 표에서 빠져 있었다.
- 대리물(GBM)이 arm-4 head 의 R-a 를 +0.4724 로 재현하는데 실측은 +0.2858 이다 — **대리물 오차
  0.187 이 주장 효과 0.090 의 2배**다.

> **처분:** D.7-② 는 **열린 채로 유지**하되 arm 5 가 아니다. 뒤에 log-ratio arm 을 제안하려면
> (i) `CTL_ABS` 대조군을 등록 arm 으로 포함하고, (ii) 튜닝에 쓰지 않은 슬라이스에서 채점하며,
> (iii) 동결 trunk 가 아니라 **fine-tune 된 trunk 를 갖는 설계**(예: arm-4 trunk 동결 후 head 만
> 재학습)로 증거를 다시 만들어야 한다.

---

## E.3 **arm 5 의 정의 (구속력)**

> **arm 5 = arm 4 (D.4) 그대로 + `f_xy` 합성행 위의 within-cell pairwise margin-rank 항.**
> `s1j` 에서 출발하는 trunk fine-tune(lr×0.05), `--fxy-prior-on-predicted` 합성, distill 0.4,
> `--f-r-rank-weight 0.1`, cond v8, split S1j — **전부 arm 4 verbatim**. 바뀌는 것은 새 손실 항의
> 7개 플래그뿐이다.

| 요소 | arm 4 | **arm 5** | 근거 |
|---|:--:|:--:|---|
| `f_xy` 손실 | 레벨(Huber) + select | **레벨 + pairwise hinge (w=3.0)** | E.1.3 — P0 가 실패 양식을 재현하고 hinge 가 고친다 |
| hinge 부착 지점 | — | **합성행** `out["mu"][:, fxy_idx]` | `net.py:553` 이 `mu` 를 `_compose_fxy` 로 내보내고, `net.py:518` 이 `mu[f_r]` 를 `detach()` 한다 → **F_r head 로 gradient 가 구조적으로 못 간다** |
| cell 정의 | (해당 없음) | **`(case_pair, feed)` = gate cell** | E.1.4 — legacy cell 은 43/78 에서 gate cell 을 섞는다 |
| `min_gap` | — | **0.005** | E.7-② — MASTER `FXYP` 반복잡음 **0.000000** 실측(6회 재실행), 여유 3자릿수 |
| boundary 가중 | — | `min(f_xy) ≤ **1.60**` 쌍 ×**3.0** | 두 캠페인 basin 이 1.53–1.58. **magic number 임을 등록**하고 대체안을 E.3.1 에 사전 지정 |
| trunk / init / distill / rank / schema | — | **arm 4 와 동일** | 쌍대 비교를 위해 다른 축을 건드리지 않는다 |

**w = 3.0 · low_thresh = 1.60 · low_weight = 3.0 · min_gap = 0.005 · margin = 0.1 z · cell = gate
는 실행 전에 동결되며 사후에 재조정하지 않는다** (D.6.4 의 규칙 승계). 값을 바꾸는 재시도는
**새 사전등록**을 요구한다.

### E.3.1 사전 지정된 대체안 (사후 교체 금지)

`1.60` 은 오늘의 basin 에 묶인 상수다. **E.7 step 0 이 대상 셀의 basin 이 1.53–1.58 을 벗어났음을
보이는 경우에만**, 그리고 **`[3]` 실행 전에 기록하는 경우에만**, elite-quantile 형태로 대체할 수
있다: `HE30`(cell 하위 절반 쌍 ×3 — R-a +0.3869 / R-b +0.0966, VAL MAE 0.07355) 또는
`HEO30`(elite 쌍만 — R-a +0.4551 / R-b +0.2188, VAL MAE 0.07594). **결과를 본 뒤의 교체는 금지**다.

### E.3.2 실행 커맨드 (확정, **본 문서는 실행하지 않는다**)

```
[0] # 선행조건 — E.7. 수행하고 스탬프하지 않으면 [3] 을 실행하지 않는다.
    sha256sum lpopt/features/featurize.py lpopt/model/model_api.py lpopt/model/net.py lpopt/model/train.py

[1] python -c "from lpopt.model.al_retrain import refresh_distill_cache; \
      refresh_distill_cache('data/models/s1j', out_path='data/models/_v5_distill_soft.npz')"
    sha256sum data/models/_v5_distill_soft.npz    # teacher 는 s1j 여야 한다 (arm 4 정오 참조)

[2] python -m lpopt.remote --input <등록된 deck> push          # 캐시도 함께 ship

[3] python -m lpopt.remote --input <등록된 deck> train -- \
      --ensemble 5 --split S1j --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 \
      --epochs 150 --num-workers 8 --device auto \
      --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
      --cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
      --promote-max-asm-bu --promote-fxy \
      --init-from data/models/s1j --trunk-finetune-lr-mult 0.05 \
      --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
      --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
      --fxy-prior-on-predicted --fxy-select-weight 0.5 --warmup-epochs 2 \
      --fxy-rank-weight 3.0 --fxy-rank-cell gate --fxy-rank-margin-z 0.1 \
      --fxy-rank-min-gap 0.005 --fxy-rank-low-thresh 1.60 --fxy-rank-low-weight 3.0 \
      --fxy-select-band 0.50

[4] pull; rc=0 / DONE / 5 멤버 / per-cell 보정 6종을 확인한 완료본만 채점한다 (C.2 승계)
[5] python -m lpopt.model.gate_promote --check-only ...            # G1
[6] python data/reports/fxy_gate_eval_arm5_YYYYMMDD.py             # G2⁵/G3‴/G5/G6/G4 + G7
```

**D.4.2 대비 바뀐 것은 마지막 두 줄의 7개 플래그뿐이다.** `--ensemble` 이하 모든 knob 이 arm 4
verbatim 이므로 비교는 **쌍대(paired)** 다. 기존 CLI 로 그대로 쓰는 플래그:
`--ensemble --split --cond-schema --width --n-blocks --head-hidden --epochs --num-workers --device
--map-decoder --map-prior-residual --map-spectral-weight --map-peak-weight --cyclen-physics-prior
--quantile-heads --quantile-weight --promote-max-asm-bu --promote-fxy --init-from
--trunk-finetune-lr-mult --distill-targets --distill-weight --distill-min-match-frac
--f-r-rank-weight --fxy-prior-on-predicted --fxy-select-weight --warmup-epochs`.
**신규 7개는 전부 미구현이며 E.8 의 구현 과제다** (`--fxy-rank-weight` / `--fxy-rank-cell` /
`--fxy-rank-margin-z` / `--fxy-rank-min-gap` / `--fxy-rank-low-thresh` / `--fxy-rank-low-weight` /
`--fxy-select-band`). deck 은 E.7-④ 에서 **실행 전에** 등록한다.

---

## E.4 평가 슬라이스와 검정력 (r2 판정이 요구한 산술)

`wcr2b_power.log` (paired 행 부트스트랩 4,000 reps, seed 0) · `wcr4_clause.log`
(cell-clustered 2,000 reps, seed 0). Δ = arm-4 head − arm-4 자기 proxy, **설계 시점 실측**:

| 통계 | 재표집 단위 | n | Δ | SE | 95% CI | **minDet@80%** | 게이트? |
|---|---|---:|---:|---:|---|---:|---|
| **G6a VAL elite band q=0.50** | **11 cell** | 246 | **+0.0940** | 0.0416 | **[+0.0218, +0.1830]** | **0.116** | **구속** |
| G6a band q=0.75 (동반) | 11 cell | 365 | +0.1165 | 0.0457 | [+0.0373, +0.2113] | 0.128 | 보고 |
| VAL11 전체 q=1.00 (동반) | 11 cell | 487 | +0.0586 | 0.0221 | [+0.0183, +0.1062] | 0.062 | 보고 |
| **G6b config-blind OOF elite (3 slice)** | **slice** | ≈165 | **E.7-⑥ 에서 산출** | 〃 | 〃 | 〃 | **구속(veto)** |
| G5 R1 (arm 4 형식) | 행 | 95 | (vs s1j) | — | [−0.0529, +0.1384] | ≈0.13 | **구속 (D 승계)** |
| R1 (head−proxy, 동반) | 행 | 95 | +0.0376 | 0.0152 | [+0.0108, +0.0694] | 0.043 | 보고 |
| R1jc elite (동반) | 행 | 65 | +0.0604 | 0.0308 | [+0.0037, +0.1239] | 0.086 | 보고 |
| **R-a joint-clean** | 행 | 76 | −0.0793 | 0.0844 | [−0.2543, +0.0821] | **0.236** | **보고 전용 (튜닝 슬라이스)** |
| **R-b exploit** | 행 | 64 | −0.1325 | 0.1351 | [−0.4075, +0.1267] | **0.379** | 보고 전용 |
| **PINBU** | 행 | 20 | −0.0241 | 0.0936 | [−0.2296, +0.1635] | **0.262** | 보고 전용 (E.4.1) |
| R-c 부호검정 | 13 wave | 104 | — | — | — | 검정력 **0.17** @ p=0.6 | 보고 전용 |
| (금지) POOLED elite | 행 | 141 | +0.4697 | 0.0780 | [+0.3192, +0.6231] | — | **사용 금지** (E.2.2) |

**r2 의 RANK 게이트는 스스로 아무것도 결정할 수 없었다** — R-a 에서 Δρ = 0.10 을 보려면 **425행**
(같은 셀에서 100-call 라운드 4회 더), R-b 에서는 **918행**이 필요하다. 본 개정은 그 사실을
결과를 본 뒤가 아니라 **바를 정하기 전에** 적는다.

### E.4.1 **선택 오염의 처리 (본 개정이 승자안을 고치는 지점)**

승자안은 R-a 에서 13개 구성을 채점해 HL30 을 고르고, 그 다음 **R-a 를 포함한 평균**을
"discriminating veto" 로 등록했다. 이는 조항을 그것을 만든 튜닝으로 통과시키는 구조다.
본 개정은 다음을 **구속력 있게** 정한다:

1. **R-a 는 튜닝 슬라이스다 → 보고 전용, 게이트 아님.** `low_thresh = 1.60` 도 그 셀의 basin 에서
   나왔다.
2. **PINBU 는 독립 슬라이스가 아니다.** `wcr2b_power.log`: manifest 25 id 중 캐시 매칭 25,
   그중 **fold r1 20 / train 4 / val 1**, 캠페인 내역
   `fpcamp_minfxy_t6t4_f121_r1: 20`. 즉 PINBU 는 R1jc 와 **같은 셀·같은 캠페인의 부분집합**이며
   슬라이스 평균에 넣으면 `T6_T4/f121` 을 이중 계산한다. **제외 사유는 "probe 성적이 나빠서"가
   아니라 중복이다.**
3. **veto 는 arm-5 의 어떤 하이퍼파라미터도 고르지 않은 곳에서 채점한다.** 오늘 MASTER 비용 0
   으로 쓸 수 있는 그런 슬라이스는 두 종류다:
   - **VAL elite band (G6a).** HL30 은 이 축의 argmax 가 아니다(E.1.3-8) → 선택 기준이 아니었다.
   - **fold C 의 joint-clean cell 중 두 캠페인 셀이 아닌 것 (G6b).** probe 는 이 셀들을 **한 번도
     채점하지 않았다**(wcr1b 의 슬라이스는 VAL/R1/R2/PIN 뿐). `wf5_arm5_eval_evidence_20260903.txt`
     의 joint-clean cell 인구:

     | cell | joint-clean 행 (OOF+VAL) | G6b 슬라이스? |
     |---|---:|---|
     | `E1_E2/f121` | 197 | ✗ 튜닝 셀 (R-a) |
     | `T6_T4/f121` | 156 | ✗ 관측된 셀 (R1jc/PINBU) |
     | **`N1_N2/f113`** | **95** | ✓ |
     | **`E1_E2/f109`** | **49** | ✓ |
     | **`J5_J6/f121`** | **21** | ✓ |
     | `K3_K4/f121` | 6 | ✗ (n < 20) |

     설계 시점 참고치(5 cell 집계, 튜닝 셀 포함): arm4 ρ̄ **+0.2234** vs 자기 proxy **+0.1416**,
     Δ **+0.0818 [+0.0088, +0.1725]**, 4/5 cell 승, se 0.0420 (min30 4 cell: Δ +0.1128
     [+0.0430, +0.1853], 4/4). **3-cell config-blind 부분집합의 per-cell 값은 아직 산출되지
     않았으며 E.7-⑥ 이 그것을 만들고 바를 스탬프한다.**

---

## E.5 **바 (구속력, arm 5)** — Amendment D 의 바를 **전부 승계**하고 세 조항을 **추가**한다

라벨된 S1j **VAL 794행**, cell 정의 `(case_pair, feed)`, holdout 라벨 ≥ 20 인 **11 cell / 487행**.
대조군은 **`s1j`** 이며 **같은 행·같은 코드·같은 보정 상태**에서 다시 측정한다(C.3 FIXED serve
path 고정, 후보와 `s1j` 양쪽 다 그 경로의 보정을 지닌 채).

### E.5.1 승계 (D.6 verbatim, 완화 없음)

| 게이트 | 지표 | 바 | 상태 |
|---|---|---|---|
| **G1** (veto) | `gate-promote --check-only` | `pass == true`, ε = σ0·Φ⁻¹(0.95^(1/N)) (N·ε 기록), `blind_targets == []`, `unavailable == []` | **A.7 그대로** |
| **G2‴** | MAE(`f_xy`), VAL n=794 | **< 0.066256** | **D.6 그대로** |
| **G3‴** | ρ̄ within-cell, 11 cell | **> 0.790696**, 동률밴드 ±0.05 안이면 cell-clustered paired BCa(11 cell, 2,000 reps, seed 0), CI 가 0 을 포함하면 **tie = FAIL** | **D.6 그대로** |
| **G5** | R1 95행 ρ(pred, meas) | **> 0.488352** **이면서** paired 행 부트스트랩(2,000 reps, seed 0) CI 가 0 을 배제 | **D.6 그대로** |
| **G4** | 68% 커버리지, VAL n=794 | **∈ [0.55, 0.80]** | **§4 그대로** |

> **왜 G2‴/G5 를 완화하지 않는가.** 승자안은 G2 를 비열등 밴드(0.068912)로 넓히고 G5 를
> "방향 보고"로 강등할 것을 제안했다. 본 개정은 **승격 바를 완화하지 않는다** — D 의 바가 곧
> 승격 바이고, arm 5 의 승격 바는 D 의 **진부분집합이 아니라 상위집합**이어야 한다. 대신 승자안이
> 실제로 요구한 것(레벨을 조금 잃고 랭킹을 얻는 후보를 **버리지 않고 기록**하는 것)은 **승격이
> 아닌 처분** 으로 따로 정의한다(E.6-B). 그 처분은 서빙을 바꾸지 않으므로 D 의 바를 건드리지 않는다.

### E.5.2 신설 **G6 — within-cell ranking 조항** (r2 §11.5 가 요구한 조항)

> **G6a (VAL 측, 검정력 있음, MASTER 비용 0).** VAL11 의 각 cell 에서 측정 `f_xy` 가 그 cell 의
> **중앙값 이하**인 행(band q = 0.50, **11 cell / 246행**)에 대한 within-cell Spearman 의 **비가중
> 평균** ρ̄₀.₅₀. **PASS 는 둘 다 필요**하다: (i) 후보 ρ̄₀.₅₀ **> 0.4334**(= 같은 행에서 재측정한
> `s1j`), (ii) 후보의 **자기 서빙 proxy** 대비 Δ > 0 이고 **cell-clustered paired 부트스트랩
> (11 cell, 2,000 reps, seed 0) CI 가 0 을 배제**. 등록 minDet@80% = **0.116**. q = 0.75(365행)
> 는 동반 인쇄하되 바가 아니다.
>
> **G6b (fold C 측, config-blind, THE VETO).** E.4.1-3 의 **3개 셀** — `N1_N2/f113`(jc 95),
> `E1_E2/f109`(jc 49), `J5_J6/f121`(jc 21) — 에서 각각 **자기 셀 안에서** within-cell Spearman 을
> 계산한 뒤 **슬라이스 평균**한다(행 pool 금지). **PASS 는 둘 다 필요**하다: (i) 후보의 슬라이스
> 평균이 **같은 체크포인트의 서빙 proxy** 를 넘고, (ii) **슬라이스 수준 paired 부트스트랩
> CI 가 0 을 배제**. 슬라이스별 최소 n = 20 미만이면 그 슬라이스는 **unscored 이지 pass 가
> 아니다**. `folds.proposal_mask` 로 proposal/non-proposal 을 **나누어 둘 다 인쇄**하며, 한쪽이
> 20행 미만이면 그 분할은 caveat 로만 적는다. **바(= `s1j` 의 같은 행 값)와 등록 minDet 은
> E.7-⑥ 이 `[3]` 실행 전에 산출해 스탬프한다.**
>
> **G6c (기전 점검, report-only).** 13개 r2 exploit wave 에서 (예측 span)/(측정 span) 의 중앙값이
> arm-4 의 **0.320** 에서 proxy 의 **0.365** 쪽으로 올라야 한다(probe forecast 0.586). 서빙 경로에서
> 잰 같은 양(head 0.122 / proxy 0.223, 원격 `wcr0_slices.py` 산출 — **로컬 미회수, 채점기가 재산출**)
> 을 함께 인쇄한다. G6a/G6b 를 통과하면서 이 값이 0.12–0.32 에 머무는 후보는 **운으로 통과한
> 것이며 조사 대상**이다.
>
> **결합 규칙.** 캠페인/셀은 **따로 채점한 뒤 평균**한다. 행 pooling 은 금지된다(E.2.2).
> R-a·R-b·R-c·PINBU 는 **각자의 minDet 을 옆에 인쇄한 채 보고만** 한다 — 어떤 후속 라운드도
> 그 슬라이스의 작은 음수를 판정으로 읽어서는 안 된다.

### E.5.3 신설 **G7 — serve-path parity** (배포 veto)

> 후보가 랭크하게 될(또는 shadow 기록을 남기게 될) 캠페인에서, `selection.json` 의 기록 예측과
> **같은 체크포인트로 store 패턴을 재채점한 값**이 **mean \|Δ\| ≤ 0.005 이고 ρ ≥ 0.99** 여야 한다.
> 현재 상태: r2 / `s1j` 에서 **mean \|Δ\| 0.01383, ρ 0.7277 → FAIL**. G7 은 **배포(shadow 포함)의
> veto** 이며 학습·채점의 veto 가 아니다(E.2.1).

### E.5.4 판정 표 (실행 후 채울 것)

| 게이트 | 기준 | 측정값 | 판정 |
|---|---|---|---|
| G1 (veto) | `pass == true`, N·ε 기록, `blind_targets == []` | TBD | TBD |
| **G2‴** | MAE < **0.066256** (n=794) | TBD | TBD |
| **G3‴** | ρ̄ > **0.790696** (11 cell, tie = FAIL) | TBD | TBD |
| **G5** | R1 ρ > **0.488352** & 행 부트스트랩 CI ∌ 0 (n=95) | TBD | TBD |
| **G4** | 커버리지 ∈ [0.55, 0.80] | TBD | TBD |
| **G6a** (신설) | ρ̄₀.₅₀ > **0.4334** & 자기 proxy 대비 cell-clustered CI ∌ 0 (11 cell/246행) | TBD | TBD |
| **G6b** (신설, veto) | 3 config-blind slice 평균 > 자기 proxy & slice CI ∌ 0 (바는 E.7-⑥ 스탬프) | TBD | TBD |
| **G6c** (신설, 보고) | span 비 중앙값 (offline 0.320 → ? / served 0.122 → ?) | TBD | — |
| **G7** (신설, 배포 veto) | mean \|Δ\| ≤ 0.005 & ρ ≥ 0.99 | TBD | TBD |
| (동반) | G2⁵ 비열등 판독 MAE < **0.068912** (= 0.066256 + LOO spread 0.002656) | TBD | (승격 조건 아님, E.6-B) |
| (동반) | 서빙 proxy 재측정 MAE / ρ̄ (같은 794행) — **정의를 E.7-⑤ 에서 고정** | TBD | — |
| (동반) | 운용점 MAE — r2 99행 / elite 행, 대조 = **arm-4 head 0.02625**(proxy 아님) | TBD | — |
| (동반) | R-a / R-b / R-c / PINBU + 각 minDet (0.236 / 0.379 / power 0.17 / 0.262) | TBD | — |
| (동반) | fold C 1,521행 레벨·랭킹 (proposal / non-proposal 분할 인쇄) | TBD | — |
| (동반) | R1jc(65) 4번째 슬라이스를 넣은 G6b 평균 | TBD | (구속 아님) |
| (기록) | `[1]` distill 캐시 teacher = `s1j` sha | TBD | — |
| (기록) | featurize/model_api/net/train sha256 | TBD | — |
| (기록) | deck / GPU | TBD | — |
| (기록) | meta `fxy_head.rank = {weight, margin, min_gap, low_thresh, low_weight, cell, mean_pairs_per_step}` | TBD | — |
| (기록) | epoch 당 유효 쌍 수 / 기여 cell 수 (0 이면 손실이 꺼진 것) | TBD | — |

---

## E.6 처분 규칙 (사전 확정)

- **A · 승격 (`s1k`).** `G1 ∧ G2‴ ∧ G3‴ ∧ G5 ∧ G4 ∧ G6a ∧ G6b ∧ G7` **전부 PASS** →
  `s1k` 로 승격, deck `model_dir` 갱신. 단 **랭커 승격은 아니다**: r2 판정에 따라 r3/r4 의
  acquisition 은 등록된 proxy/`s1i` 로 계속 랭크하고, 후보의 `fxy_mean` 은 **shadow 로 기록**된다.
  **랭커로의 승격은 G6b 가 이전에 보지 않은 두 번째 cell-round 에서 다시 통과할 때만** 가능하다
  (4번째 슬라이스가 붙으면 조항의 minDet 이 0.122 → 0.106 으로 내려간다, `wcr4_clause.log`).
- **A′ · G4 만 FAIL** (나머지 전부 PASS) → **승격하되 `fxy_head.serve_sigma = "barred"` 를 유지**한다.
  D.6.4 의 **유일한 예외**를 그대로 승계한다.
- **B · shadow record (승격 아님, 배포 아님).** `G1 ∧ G3‴ ∧ G6a ∧ G6b ∧ G7` PASS 이고
  레벨이 **비열등 밴드 안**(MAE < 0.068912) 이지만 G2‴ 또는 G5 가 FAIL → **챔피언은 `s1j` 로
  유지되고 deck 은 바뀌지 않으며 acquisition 은 그대로다.** 후보는 서빙되지 않고, wave 마다
  평가된 8개 후보에 대해 `fxy_mean` 열만 기록된다(E.8-⑧). 이것은 **서빙 동작을 바꾸지 않으므로
  D 의 승격 바를 완화하지 않는다.** 목적은 단 하나 — **104행의 전향적(prospective) 랭킹 시험을
  MASTER 비용 0 으로 얻는 것**이다. 이 상태에서 어떤 성능 주장도 하지 않는다.
- **C · G6a 또는 G6b FAIL** → 챔피언 `s1j` 유지, **shadow 조차 하지 않는다.** head 는 r2 판정대로
  **level estimator 로 남는다.**
- **D · G7 FAIL** → 다른 게이트 결과와 무관하게 **A·B 어느 배포도 하지 않는다.** 채점 결과는
  기록하고, parity 진단(E.7-①)을 마친 뒤 **재채점 없이** 같은 체크포인트로 배포 여부를 다시 묻는다.
- **E · G1 FAIL** → **reject.** trunk 를 푼 arm 이 legacy 를 회귀시켰다는 뜻이고, 재시도는 새
  사전등록을 요구한다. **본 개정의 w=3.0 을 사후에 낮추어 재실행하지 않는다.**
- **arm 5 의 결과로 어떤 후속 arm 도 자동 발동하지 않는다.**

---

## E.7 선행조건 (step 0 — 전부 `[3]` 이전, 스탬프 필수)

① **G7 parity 진단** (배포 경로의 blocking). wave 별로 `selection.json` 후보와 같은
`record_id` 의 store `pattern` 을 대조하고 양쪽을 재채점한다. wave 0 이 1e-5 로 일치하고
wave 1–12 가 흩어지며 `F_r` 이 같은 크기로 흩어진다는 사실은 **선택과 실행 사이의 패턴
정규화/합법화 단계**를 지목한다. 도구는 E.8-⑨.

② **`FXYP` 반복잡음 — 이미 측정되었다. 선행조건은 이것으로 해소된다(등록).**
`pinbu_wave_minfxy_r2_results_20260903.md` §0: **M2b** 기록 core `6c2243ff` 6회 정확 재실행에서
`F_xy` spread **0.000000**(`F_xya`·pin·cyclen 도 0.000000), **M2a** between-core \|ΔF_xy\|
**0.000000** (25/25 core, 30/30 행), **M6** 6개 scalar 축 bit-exact 30/30.
→ **`min_gap = 0.005` 는 측정 잡음(인쇄 정밀도에서 0)보다 3자릿수 위이고, "잡음이 0.005 를
넘으면 R-b 의 천장이 무너진다"는 승자안의 위험 분기는 발동하지 않는다.** 이 항목은 실행
선행조건이 아니라 **해소된 것으로 기록**한다.

③ **distill 캐시 teacher = `s1j`** 재생성 + sha 스탬프. arm 4 는 사후 정오로만 확인되었다
(HOST_238 캐시 sha `9d4b489d…`, 4,409,412 B, `.bak_s1i_teacher_20260831` 보존,
`distill_refresh_s1j.log` "66136 soft-target rows … 178 teacher cells"). trunk 가 풀리는 arm 에서
distill 은 **G1 의 주 방어선**이므로 낡은 teacher 는 G1 을 옛 챔피언 쪽으로 끌어당긴다.

④ **deck / GPU 를 실행 전에 등록**한다 (`lpopt.inp` = gpu 0 vs `lpopt_gpu1.inp`).
arm 2·3·4 가 남긴 미해결 편차를 네 번째로 반복하지 않는다.

⑤ **serving proxy 의 정의를 채점기에서 고정**한다. 현재 설계 문서들에 **서로 다른 세 개의
"proxy" 레벨 값**이 있다 — `s1j` 0.072083 (D.2.2 / arm4 §9), arm4 0.070643 (arm 4 결과보고서 §1
동반 판독), 그리고 캐시 규약 0.15065 (`wcr1b_probe.log`). G2 의 동반 판독과 G6a/G6b 의 비교
대상이 이것이므로, **어느 구성인지(어느 `F_r` 열·어느 (a,b)·어느 보정)** 를 실행 전에
한 문장으로 고정하고 sha 와 함께 스탬프한다.

⑥ **G6b 의 바를 스탬프한다.** `s1j` 와 arm 4 를 3개 config-blind 셀(`N1_N2/f113`,
`E1_E2/f109`, `J5_J6/f121`)의 joint-clean 행에서 채점해 per-cell ρ, 슬라이스 평균,
슬라이스 수준 부트스트랩 SE 와 minDet, `proposal_mask` 분할을 산출한다. **동시에 HL30 probe 를
구성을 동결한 채 같은 셀에서 채점해 forecast 를 남긴다.** 이 forecast 가 나쁘게 나와도
**하이퍼파라미터를 재조정하는 것은 금지**되며, 허용되는 것은 **arm 을 취소하는 것뿐**이다.
슬라이스 수준 SE 가 minDet > 0.15 를 함의하면, 조항을 **넓히거나**(fold C 의 non-joint-clean
elite band 추가) **veto 를 전향적 슬라이스로 미루는** 결정을 **`[3]` 이전에** 기록한다.

⑦ **source sha 스탬프.** `featurize.py` / `model_api.py` / `net.py` / `train.py` 의 sha256 을
`train.log` 머리와 `meta.json` 에 인쇄한다 (arm 4 는 원격 소스를 검증하지 못했다).

---

## E.8 별도 구현 과제 (본 문서는 구현하지 않는다)

①–⑦ 은 `[3]` 의 **선행조건**(신규 플래그), ⑧–⑩ 은 분리 가능하다.
**모든 기본값은 오늘의 동작과 동일하며 `weight = 0` 은 byte-identical 이다.**

| # | 변경 | 파일 | 근거(로컬 확인) |
|---|---|---|---|
| ① | `f_xy_rank_loss(...)` — `f_r_rank_loss` (`train.py:703–745`) 의 구조적 복사본. **합성행** `out["mu"][:, fxy_idx]` 에 걸고 `batch["target_mask"][:, fxy_idx]` 로 마스크한다. cyclen 과 달리 prior 를 되더할 필요가 **없다**(prior 가 이미 `_compose_fxy` 안에 있다) | `lpopt/model/train.py` | `net.py:553` `"mu": self._compose_fxy(self.mu_head(feat))`; `net.py:518` `mu[:, fxy_ref_idx].detach()` |
| ② | `fxy_cell_codes(df)` — `(case_pair, feed)` 의 dense factorization, 미해소 행은 `-1`. `cyclen_cell_codes` (`dataset_torch.py:458–491`) 의 형제 | `lpopt/model/dataset_torch.py` | E.1.4 — legacy cell 은 43/78 에서 gate cell 을 섞는다 |
| ③ | `tensors["fxy_cell"]` 을 `train.py:1169–1170` 옆에 추가하고, `train.py:1836` 의 optional-key 목록에 `"fxy_cell"` 을 넣는다 | `train.py` | 현행 rank 손실 두 개는 `"cyclen_cell" in batch` 로 게이트된다(`train.py:1912, 1927`) |
| ④ | `TrainConfig`: `fxy_rank_weight = 0.0`, `fxy_rank_margin_z = 0.1`, `fxy_rank_min_gap = 0.005`, `fxy_rank_low_thresh = 1.60`, `fxy_rank_low_weight = 3.0`, `fxy_rank_cell = "gate"` | 〃 | `f_r_rank_weight` (`train.py:141`) 와 같은 형태 |
| ⑤ | CLI `--fxy-rank-weight / --fxy-rank-margin-z / --fxy-rank-min-gap / --fxy-rank-low-thresh / --fxy-rank-low-weight / --fxy-rank-cell {gate,legacy}`; `--fxy-rank-weight > 0` 인데 `--promote-fxy` 가 없으면 `ap.error` | 〃 | — |
| ⑥ | `--fxy-select-band 0.50` — best-epoch 선택이 G6a 가 재는 축을 보게 한다. 현재 `fxy_metrics` (`train.py:1388–1441`) 는 `within_case_spearman` 을 **`case_pair` 단독** 으로 부르는데 모든 게이트는 `(case_pair, feed)` 를 쓴다 — **등록된 불일치**이며 band 옵션은 gate cell 을 써야 한다 | 〃 | `train.py:1409` 의 docstring |
| ⑦ | 로그: epoch 당 **평균 유효 쌍 수**와 기여 cell 수; meta `fxy_head.rank = {...}` | 〃 | 손실이 조용히 0 이 되는 실패를 잡는 유일한 계기 |
| ⑧ | *(r3 측, 분리 가능)* `_write_wave_artifacts` 에 **shadow ranker 열** — 서빙 랭커를 바꾸지 않고 비서빙 head 의 `fxy_mean` 을 전향적으로 기록 | `lpopt/search/campaign.py` | E.6-B. **G7 PASS 전에는 쓰지 않는다** |
| ⑨ | *(분리 가능)* `lpopt campaign rescore --run <dir> --model <dir>` — wave 별 \|Δ\|·ρ 를 run dir 에 쓰고 캠페인 런처의 사후 게이트로 건다 | 〃 | **G7 의 계측기**(E.7-①) |
| ⑩ | *(분리 가능, arm 아님)* `f_xy` σ 의 **out-of-fold cross-fit 스칼라 보정**을 `calibration.json` 에 기록. 오늘 `predict_fxy` 는 σ 열을 **raw 로 통과**시킨다. **cross-fit 표를 먼저 공표할 것** — 제출된 산출물의 TRAIN 적합 k 는 커버리지를 바닥 아래로 떨군다(arm4 k 0.4095 → VAL 0.5164) | `lpopt/model/calibrate.py`, `model_api.py` | G4 는 두 번(0.831 → 0.873) 과대 방향으로 실패했다 |

**요구 시험** (`tests/test_fxy_head.py` 신규 §14, D.7-③ 의 형식을 따른다):
(a) `fxy_rank_weight = 0` 이 현행과 **byte-identical**;
(b) 쌍은 `f_xy` 라벨이 있는 행끼리, **같은 `(case_pair, feed)` 코드 안에서만** 형성된다;
(c) `min_gap` 이 규정대로 거르고, 유효 쌍이 없는 batch 는 **정확히 0** 을 기여한다;
(d) `fxy_rank_weight = 3.0` 으로 **4 epoch 실학습 후** `mu_head.weight[_FR_IDX]` / `bias[_FR_IDX]`
와 cyclen 행이 `--init-from` 체크포인트와 **bit-identical** (detach 와 hook 이 둘 다 성립);
(e) `--fxy-rank-weight` 만 주고 `--promote-fxy` 를 빼면 `SystemExit`;
(f) `--fxy-rank-cell legacy` 가 `cyclen_cell` 그룹핑을 **정확히** 재현;
(g) `fxy_cell` 코드가 채점기가 쓰는 gate cell key 와 fixture 에서 **일치**;
(h) ⑨ 의 parity 하네스가 r2 **wave 0** 을 1e-5 로 재현(현재 참값이 알려진 유일한 wave).

---

## E.9 사전에 금지하는 주장 (§8 · D.8 승계, 본 개정에서의 준수)

- **head 의 `f_xy` 로 미측정 core 의 `F_xy ≤ 1.65` 적합성을 선언하지 않는다** (§8 그대로).
- **"arm 5 가 최적 core 를 만든다"고 적지 않는다.** 이 arm 은 **정렬**에 관한 것이고, r2 phase-2 는
  이 셀의 delivery-grade 최선이 여전히 `F_r` 시대 core `a785eded` (1.5295) 임을 확정했다.
- **재채점(parity) 수치로 r2 판정을 다시 열지 않는다.** 재채점하면 head 도 proxy 도 함께 오르고
  (+0.2526→+0.3260 vs +0.3060→+0.4049) **head−proxy 의 순서는 바뀌지 않는다**(E.2.1).
- **R-a / R-b / R-c / PINBU 결과를 판정으로 읽지 않는다.** 각 minDet(0.236 / 0.379 / 검정력 0.17 /
  0.262)을 옆에 인쇄하는 것이 그 금지의 집행 방식이다.
- **elite 행을 cell 을 가로질러 pool 한 통계를 인용하지 않는다** (Δ +0.4697 은 cell 간 레벨 아티팩트).
- **probe 를 상한으로 부르지 않는다** (D.6.2 caveat 승계). probe 는 레벨에서 서빙 head 보다
  **어디서나 0.006 나쁘고**(0.0734 vs 0.0674) PINBU 에서 **부호가 반대**다.
- **"trunk 를 풀면/hinge 를 걸면 순위가 오른다"를 실측 전에 주장하지 않는다.** E.1.3 은 forecast 이고
  E.10 은 그 forecast 가 어디서 위태로운지를 미리 적는다.
- **arm 5 가 σ 를 고친다고 주장하지 않는다.** G4 는 다시 FAIL 할 것으로 예상하며
  `serve_sigma = "barred"` 는 유지된다.
- **§2.1 / D.5 의 fold 밖 라벨을 근거로 한 학습 성능 주장은 계속 금지된다.** fold C·R1·R2·PINBU 는
  학습에 들어가지 않으므로 순위 판독의 대상일 뿐이다.
- **바를 사후에 옮기지 않는다.** E.5.1 의 다섯 상수와 E.5.2 의 0.4334 는 arm 5 의 어떤 산출도
  관측하기 전에 고정되었고, G6b 의 바는 E.7-⑥ 이 **`[3]` 이전에** 스탬프한다.

---

## E.10 등록해 두는 반론과 약점 (전부 실행 전에 적는다)

1. **선택 다중성.** HL30 은 R-a 에서 13개 구성(+ 별도 스크립트 4개) 중 최우수로 뽑혔다. 완화책은
   (i) 게이트를 튜닝 슬라이스에서 **떼어냈다**(E.4.1), (ii) 용량 반응이 단조다, (iii) 구성은
   실행 전에 동결된다. 그럼에도 **R-a 의 +0.4826 은 선택된 최대값이며 unbiased 추정치가 아니다.**
2. **이득이 튜닝 셀에 몰려 있다.** probe 에서 P0 → HL30 은 R-a 를 **+0.342** 움직이지만 R1jc 는
   **+0.034**(0.3260 → 0.3600, sd 0.0077/0.0041) 밖에 움직이지 않는다. `low_thresh 1.60` 자체가
   그 셀의 basin(1.53–1.58)에서 나왔다. **G6b 가 config-blind 셀에서 채점되는 이유**이자,
   G6b 가 실패할 가장 그럴듯한 시나리오다.
3. **basin 정렬축은 셀 간 전이가 약하거나 음수다.** `wcr3_ceiling.log`: R1jc → R-a 는 CV 가 고른
   λ=1e1 에서 **ρ −0.1203**(1e2 에서 +0.0738, 1e3 에서야 +0.4752), R-a → R1jc 는 1e1 에서
   **+0.0051**. 이는 "두 번째 미관측 cell-round 에서 다시 통과할 때만 랭커로 승격"이라는
   E.6-A 의 조건을 정당화하는 동시에, 그 조건이 **쉽게 충족되지 않을 것**임을 예고한다.
4. **basin 임베딩이 축퇴되어 있다.** `wcr3_ceiling.log`: R-b 는 eff.rank **4.8**, PC1 이 분산의
   59.8% (R-a 7.2 / 44.8%, VAL11 11.9 / 29.7%). exploit slot 에는 정렬에 쓸 임베딩 spread 자체가
   적다 — 천장 +0.2567 의 물리적 이유다.
5. **G6b 는 슬라이스가 3개다.** 등록 SE 는 슬라이스를 독립 행-부트스트랩으로 다루므로 **슬라이스
   간 분산 성분을 추정하지 못하고 낙관적**이다. 그래서 바를 "고정 ρ 수준"이 아니라 **"자기 proxy 를
   CI 로 이길 것"** 으로 둔다 — 그 형태는 낙관적 SE 에 덜 민감하다.
6. **서빙/오프라인 추정기 괴리(E.2.1)가 r2 를 떨어뜨린 0.047 보다 크다** (R-a head 0.073, proxy
   0.099). G6 의 구성타당도는 G7 에 의존하며, 그래서 **G7 이 배포를 막는다.**
7. **레벨 비열등 마진 0.002656 의 출처와 그에 대한 반론.** 이 값은 arm 4 §7.3 의 5개 LOO 앙상블
   MAE 폭(0.066993 – 0.069649)이다. **같은 표를 arm-4 결과보고서는 "어떤 LOO 조합도 G2‴ 를
   통과하지 못한다"는 반대 논지로 썼다.** 본 개정은 그래서 이 마진을 **승격 바로 쓰지 않고**
   E.6-B 의 비배포 처분에만 쓴다. 그럼에도 이것이 arm 5 가 D 대비 **유일하게 느슨해지는 지점**이며,
   심사자가 거부할 권리가 있는 지점이다.
8. **랭킹 ρ 의 이득이 regret 으로 번역되지 않는다** (probe R-c: P0 0.0076 → HL30 0.0084, 나빠짐).
   G6c 를 두고 R-c 를 보고 전용으로 두는 이유다. arm 5 가 G6a/G6b 를 통과하고도 wave 단위 선택을
   개선하지 못할 수 있다 — 그 가능성을 **사전에** 적는다.
9. **G1 은 형식이 아니다.** trunk 가 풀린 채 **새로운 gradient 경로**(f_xy 랭킹 항)가 추가된다.
   방어선은 `_compose_fxy` 의 `detach`, cyclen 행 gradient-0 hook, distill 0.4,
   `--f-r-rank-weight 0.1`, trunk lr ×0.05 뿐이다. arm 4 는 이 구성에서 worst enforced drop
   0.017241 (ε 의 12%) 로 안전했지만, arm 5 의 항은 **새 경로**다.
10. **probe 는 동결 임베딩 위의 선형 head 다.** 실제 fine-tune 은 임베딩을 움직인다 — 더 나을 수도,
    나쁠 수도 있다(D.6.2 의 등록된 caveat, arm 4 에서 실제로 나쁜 쪽으로 어긋났다).

---

## E.11 무엇이 arm 5 를 **학습 전에** 반증하는가 (둘 다 싸다)

- **E.7-⑥ 의 config-blind forecast 가 3개 셀에서 proxy 를 넘지 못하면**, arm 5 의 전제("cell 안의
  순서는 임베딩에 있고 목적함수가 버린다")는 두 캠페인 셀 밖에서 성립하지 않는 것이다. 그 경우
  **하이퍼파라미터를 다시 고르지 말고 arm 을 취소**한다.
- **`wcr3_ceiling.py` 를 새 셀에서 재실행해 LOWO ρ ≈ 0 이 나오면** 같은 결론이다.
- (해소됨) `FXYP` 반복잡음 분기는 E.7-② 의 실측 **0.000000** 으로 발동하지 않는다.

**E.5.4 판정 스탬프(TBD):** arm 5 — G1 TBD · G2‴ TBD · G3‴ TBD · G5 TBD · G4 TBD · G6a TBD ·
G6b TBD · G7 TBD → E.6 처분 TBD. 결과 문서: `fxy_head_results_arm5_YYYYMMDD.md` (미작성).

---

# Amendment E — **E.7 step-0 스탬프** · 2026-09-03 (append-only)

본 절은 **기록**이다. E.1–E.11 의 어떤 바도 바꾸지 않으며, `[3]` 실행 **이전에** 산출된
수치만 담는다. 모든 계산은 HOST_238 GPU 1 / read-only, 로컬 계산 없음.
산출물: `~/lpopt_ws/scratch/wcrE0_step0.py` → `wcrE0_step0.log` / `wcrE0_g6b.json`,
`~/lpopt_ws/scratch/wcrE0_forecast.py` → `wcrE0_forecast.log`.

## E.7-⑦ · source sha256 (238 `~/lpopt_ws/src`, 로컬과 **일치 확인**)

| 파일 | sha256 |
|---|---|
| `lpopt/model/featurize.py` | `5e720a5e50455f071fa5dd365e441b0d9255900be868266a722c9176b3a00baa` |
| `lpopt/model/model_api.py` | `e657184d8f181deaf73edc33c5c5df8448d39fbb56913c60ab560fb74ad36c8c` |
| `lpopt/model/net.py` | `ab1330bbc02c33dab7e8f13339e5f0328c612a605cfed9570b180a2871f4f69f` |
| `lpopt/model/train.py` | `bed17a9d7d74b597982ea47dd42891d6b7144830a0a1050b11f2ddb92a6b6afa` |
| `lpopt/model/al_retrain.py` | `1ca927614f0db49d7d02ad4a2514d64fbf54d02dd1d964032ee82744216b5c85` |
| `lpopt/model/dataset_torch.py` | `1f24bd99e86d6670c9450f7abb0889b37fc41494dc368adbd8f4b2a9c38ca7fa` |

> **등록된 결손.** `train.py` 는 이 sha 들을 `train.log` 머리 / `meta.json` 에 **인쇄하지 않는다**
> (E.7-⑦ 이 요구한 코드 변경은 E.8 목록에 없고, 본 실행에서 코드를 수정하지 않았다). 대신 위 표를
> 이 스탬프와 run dir 의 `SOURCE_SHA256` 파일에 남긴다. `meta.json` 은 `vendor_manifest_sha256`
> 만 담는다. **다음 arm 이전에 해소할 것.**

## E.7-③ · distill 캐시 teacher = `s1j` (재생성 불필요, 검증만)

`data/models/_v5_distill_soft.npz` sha256 =
`9d4b489d6cfff3470dfce8777ff1d926da91d1e85429441f0d9cf7ee4255ba4d`
(4,409,412 B, 2026-09-02 12:14) — **등록값 `9d4b489d…` 와 일치**. `distill_refresh_s1j.log`
"66136 soft-target rows … 178 teacher cells". `s1i` teacher 백업
(`.bak_s1i_teacher_20260831`, 4,404,948 B) 그대로 보존. **arm 5 는 arm 4 와 동일한 teacher 를
쓰며 재생성하지 않는다** (E.3 의 쌍대 비교 요건).

## E.7-④ · deck / GPU 등록

**deck = `lpopt_gpu1.inp`** (로컬 sha256
`d8ee3274085cce47f43c1655ba2c43de15a59fc3d891b18d58686ec91445d078`,
`[remote] gpu = 1`, `model_dir = data/models/s1j`). **GPU 1 고정**, `CUDA_VISIBLE_DEVICES=1`.
arm 4(`runs/20260902_122746/run.sh`) 와 **같은 deck·같은 GPU** 이므로 쌍대 비교가 성립한다.
(`lpopt.inp` 의 "GPU 0 고정" 지시에 대한 미해소 편차는 s1j·arm 4 와 동일하게 계속 기록으로 남는다.)

## E.7-① · **serving-vs-offline 괴리 진단** (G7 계측)

r2 `selection.json` 100행 중 오프라인 캐시 매칭 **99행**. 대조군은 같은 체크포인트(`s1j`)로
store 패턴을 재채점한 `scratch/a5/cache_s1j.npz`
(sha256 `5dfb5f58417eef67f826d3e5a1381dfb58ef666088610b784e886b6c18b9c644`).

| 항목 | 값 |
|---|---|
| `f_xy` 서빙기록 vs 재채점 | mean \|Δ\| **0.01383** · max **0.06831** · ρ **0.7277** |
| `F_r`(cal) 같은 행 | mean \|Δ\| **0.01472** · max 0.05960 · ρ 0.7018 |
| `F_r`(raw) 같은 행 | mean \|Δ\| **0.01535** · ρ 0.7018 |
| wave 0 (n=8) | d_fxy **+0.00000** (sd 0.00001, max 0.00001), d_F_r **+0.00000** — **정확** |
| wave 1–12 | 평균 −0.02002 … +0.02330 로 흩어짐 (wave 3 에서 max \|Δ\| 0.06831) |
| `fxy_source` census | **`head` 100/100** — 구성(head vs prior)의 차이가 **아니다** |
| `origin` census | local 63 · elite 32 · guided 4 · heuristic 1 |
| `fxy_sigma` | 서로 다른 값 다수 (0.1471 … 0.1572) — E.2.1 의 "균일 +0.12205" 는 **다른 축척**이며 본 스탬프가 정정한다 |

**진단 (E.7-① 이 요구한 특정).**

1. **head 의 합성/정규화 차이가 아니다.** 선택된 100행 전부 `fxy_source = head` 이고,
   서빙 `f_xy` ≈ 1.182428·(서빙 `F_r`) − 0.185231 (resid sd 0.014798, R² 0.98727),
   오프라인 `f_xy` ≈ 1.195229·(오프라인 `F_r_cal`) − 0.206827 (resid sd 0.008496, R² 0.99579).
   **두 경로의 prior 기울기·절편이 같은 계열**이고 잔차 구조만 다르다. 서빙 경로가 raw `F_r` 을
   썼다는 가설도 기각된다 — 서빙 `F_r` 은 `fr_raw`(mean \|Δ\| 0.01535)보다 `fr_cal`
   (0.01472)에 더 가깝다.
2. **결손은 상류(입력 패턴/피처)에 있다.** wave 별 d_fxy 와 d_F_r 의 상관은 wave 2·3·6·7·8·9·11
   에서 **+0.95 이상**, 전체 **+0.8284**, 회귀 `d_fxy ≈ 0.8030·d_F_r + 0.0019`.
   즉 `f_xy` 의 이동은 **`F_r` 의 이동으로 거의 전부 설명된다**. `f_xy` head 행만 어긋났다면
   기울기는 0 이어야 하고 `F_r` 은 흔들리지 않아야 한다.
3. **wave 0 만 정확하다.** wave 0 후보는 seed corpus 에서 직접 오고, wave ≥ 1 후보는 campaign 이
   생성·합법화(legalisation)한 패턴이다. 따라서 가장 검약한 설명은 E.2.1 이 적은 그대로 —
   **선택 시점에 채점된 패턴과 store 에 기록된 패턴이 wave ≥ 1 에서 byte-identical 하지 않다.**
   본 스탬프는 그 설명을 좁혔다: 합성식도, 보정도, σ 축척도, `F_r` 열 선택도 아니다. **패턴이다.**
4. **따라서 G7 은 FAIL 상태 그대로**(mean \|Δ\| 0.01383 > 0.005, ρ 0.7277 < 0.99)이며,
   E.6-D 에 따라 arm 5 의 **배포(shadow 포함)를 막는다.** 채점은 막지 않는다 — G6 는 FIXED
   serve path 재채점으로만 계산된다(E.2.1).

**R-a 76행 재확인 (E.9 준수):** 서빙 head **+0.2526** → 재채점 **+0.3260**, 같은 행 proxy 재채점
**+0.4049**. slot 별 서빙→재채점: control +0.9182 → +0.8909 · explore +0.3826 → +0.4130 ·
exploit −0.1157 → −0.0064. **head−proxy 의 부호와 순서는 바뀌지 않는다. r2 판정을 다시 열지 않는다.**

## E.7-⑤ · 서빙 proxy 정의 **고정**

세 개의 서로 다른 "proxy" 값은 **같은 열의 서로 다른 사용법**임이 확인되었다 (VAL794):

| 정의 | s1j | arm4 |
|---|---:|---:|
| `fr_cal` 을 **그대로** `f_xy` 로 읽음 (캐시 규약) | 0.153096 | 0.150653 |
| `a·fr_cal + b`, (a,b) 를 S1j TRAIN 에서 재적합 | a 1.201066 / b −0.217798 → 0.071295 | a 1.209309 / b −0.235092 → 0.070225 |
| 체크포인트가 **출하한** prior (a,b) | **0.072083** | **0.070643** |
| (대조) head | 0.066257 | 0.067406 |

> **고정 (구속력).**
> - **RANK proxy** := 후보 체크포인트 **자신의** 보정된 예측 `F_r` (`fr_cal`). 양의 affine 은
>   단조이므로 어떤 (a,b) 를 쓰든 within-cell 순위가 같다 → G6a/G6b 는 `fr_cal` 을 직접 쓴다.
> - **LEVEL proxy** := `a·fr_cal + b`, (a,b) 는 **체크포인트가 출하한** `f_xy` prior
>   (s1j 0.072083 / arm4 0.070643).
> - 캐시 규약 0.15065 는 **proxy 가 아니라 미보정 `F_r` 열**이며 레벨 판독에 인용하지 않는다.
> - 채점기 `fxy_gate_eval_arm5_*.py` 는 이 두 정의만 인쇄한다.

## E.7-⑥ · **G6b 바 스탬프** (config-blind 3 셀, joint-clean OOF+VAL)

joint-clean = `cbc_max ≤ 1600 ∧ f_q ≤ 2.41 ∧ |ao_abs| ≤ 0.30 ∧ f_r(측정) ≤ 1.55`
(r2 보고서 · `wcr0_slices.py` 와 동일). 행 pooling 없음 — 셀별 ρ 를 낸 뒤 평균.

| cell | n | val / oof | proposal | 캠페인 | **`s1j`** | proxy_s1j | arm4 | proxy_arm4 |
|---|---:|---|---:|---|---:|---:|---:|---:|
| `N1_N2/f113` | 95 | 13 / 82 | **0/95** | intervention 82 · fpcamp_minfr 6 · _pin 7 | **+0.1541** | −0.0035 | +0.1410 | +0.0085 |
| `E1_E2/f109` | 49 | 10 / 39 | **0/49** | intervention 39 · fpcamp_minfr 10 | **−0.2478** | −0.3657 | −0.0206 | −0.2532 |
| `J5_J6/f121` | 21 | 21 / 0 | **0/21** | fpcamp5 12 · fpcamp6 3 · frtransfer 6 | **+0.3678** | +0.3242 | +0.1917 | +0.2339 |
| **슬라이스 평균** | 165 | | | | **+0.0914** | **−0.0150** | **+0.1040** | **−0.0036** |

**`proposal_mask` 분할:** 세 셀 모두 **proposal 0행** (`alsearch*` 캠페인 없음) → 분할이 존재하지
않는다. `folds.py` 의 의무 인쇄는 "non-proposal 165 / proposal 0" 으로 충족되며, fold C 의
optimizer's-curse 비대칭(s1j bias −0.03261)은 **이 세 슬라이스에 적용되지 않는다.**
셀별 최소 n = 21 ≥ 20 → **세 슬라이스 모두 scored**.

**슬라이스 수준 paired 부트스트랩 (3 슬라이스, 2,000 reps, seed 0):**

| 대조 | Δ | 95% CI | SE | **minDet@80%** | 승 | CI ∌ 0 |
|---|---:|---|---:|---:|---:|---|
| **`s1j` − proxy_s1j** | **+0.1064** | [−0.0013, +0.1880] | 0.0473 | **0.132** | 3/3 | ✗ |
| arm4 − proxy_arm4 | +0.1076 | [−0.0426, +0.2501] | 0.0762 | 0.213 | 2/3 | ✗ |
| arm4 − `s1j` | +0.0127 | [−0.1760, +0.2342] | 0.1065 | 0.298 | 1/3 | ✗ |

> **G6b 바 (구속력, 확정).** 후보의 슬라이스 평균이 **같은 체크포인트의 `fr_cal` proxy** 를 넘고,
> 위와 동일한 3-슬라이스 paired 부트스트랩(2,000 reps, seed 0)의 CI 가 0 을 배제할 것.
> 참고 수준: `s1j` 슬라이스 평균 **+0.0914** (자기 proxy −0.0150 대비 Δ +0.1064),
> arm4 **+0.1040** (자기 proxy −0.0036 대비 Δ +0.1076).
> **등록 minDet@80% = 0.132** (바를 정의하는 `s1j` − proxy_s1j 쌍에서 산출).
> 두 참조 후보 모두 CI 가 0 을 배제하지 못한다 — **이 조항은 arm 4 를 통과시키지 않는다.**
>
> **E.7-⑥ 의 "minDet > 0.15" 분기에 대한 사전 결정 (`[3]` 이전 기록, 구속력).**
> 바를 정의하는 쌍의 minDet 은 0.132 로 문턱 아래지만 arm4 인스턴스는 0.213 으로 위에 있다 —
> 추정치 자체가 불안정하다. 그럼에도 **조항을 넓히지 않고(fold C non-jc elite band 추가 없음),
> veto 를 전향적 슬라이스로 미루지도 않는다.** 이유 둘: (i) 바의 형태가 "고정 ρ 수준"이 아니라
> **"자기 proxy 를 CI 로 이길 것"** 이므로 낙관적 SE 에 덜 민감하다(E.10-5 가 이미 등록한 논거),
> (ii) 아래 config-blind forecast 의 Δ(+0.2431)가 **비관적 minDet 0.213 마저 상회**한다.
> 이 결정은 어떤 arm-5 산출도 관측하기 전에 내려졌고 사후에 뒤집지 않는다.

**동반 (구속 아님) 4번째 슬라이스 R1jc (n=65):** `s1j` +0.2048 / proxy_s1j +0.1176 ·
arm4 +0.2632 / proxy_arm4 +0.2028.

## E.7-⑥ (후반) · **config-blind forecast** — 동결 HL30 probe, 같은 3 셀 (E.11 반증 시험)

`scratch/wcrE0_forecast.py` (arm-4 동결 임베딩 + `fr_cal`, S1j **train fold 에서만** 적합,
3 seed, 1,200 step, GPU 1). 구성은 E.3 이 동결한 **HL30 그대로** (w = 3.0 · gate cell ·
margin 0.1 z · min_gap 0.005 · low ≤ 1.60 ×3.0). 세 셀은 probe 가 **한 번도 채점하지 않은**
슬라이스다(wcr1b 의 슬라이스는 VAL/R1/R2/PIN 뿐).

| 추정기 | `N1_N2/f113` | `E1_E2/f109` | `J5_J6/f121` | **슬라이스 평균** |
|---|---:|---:|---:|---:|
| P0 레벨 전용 (대조) | +0.0318 | +0.0636 | +0.2731 | **+0.1228** |
| (seed sd) | 0.0056 | 0.0040 | 0.0075 | 0.0037 |
| **HL30 (arm 5 구성)** | **+0.2290** | **+0.2308** | **+0.2588** | **+0.2395** |
| (seed sd) | 0.0065 | 0.0037 | 0.0063 | 0.0054 |
| — arm4 head (캐시) | +0.1410 | −0.0206 | +0.1917 | +0.1040 |
| — arm4 proxy `fr_cal` | +0.0085 | −0.2532 | +0.2339 | −0.0036 |
| — `s1j` head (캐시) | +0.1541 | −0.2478 | +0.3678 | +0.0914 |

> **E.11 반증 시험 → 발동하지 않는다 (arm 5 진행).** HL30 의 슬라이스 평균 **+0.2395** 는 자기
> proxy(−0.0036)를 **Δ +0.2431** 로 넘고, **3/3 셀**에서 proxy 를 이기며, P0 대비 **+0.1167** 이다.
> E.10-2 가 "가장 그럴듯한 실패 시나리오"로 등록한 것 — 이득이 `low_thresh 1.60` 이 나온 튜닝 셀
> basin 에만 몰려 있다 — 은 **이 세 셀에서 재현되지 않는다**: P0 → HL30 이동이
> `N1_N2/f113` +0.1972, `E1_E2/f109` +0.1672 로 R-a 의 +0.342 와 같은 자릿수다.
> `J5_J6/f121` 은 −0.0143 으로 유일하게 움직이지 않는다(그 셀은 P0 가 이미 +0.2731).
>
> **그럼에도 이것은 forecast 이지 상한이 아니다** (D.6.2 caveat 승계, E.9). probe 는 동결 임베딩
> 위의 선형 head 이고 실제 fine-tune 은 임베딩을 움직인다 — arm 4 에서 실제로 나쁜 쪽으로
> 어긋났다. **이 표로 arm 5 의 통과를 예고하지 않는다.**

## E.7-② · `FXYP` 반복잡음 — 해소됨 (재확인 없음)

`pinbu_wave_minfxy_r2_results_20260903.md` §0 의 실측 0.000000 을 그대로 승계한다.
`min_gap = 0.005` 는 변경하지 않는다. E.3.1 의 대체안 발동 조건(대상 셀 basin 이 1.53–1.58 밖)도
검토했으나 **발동하지 않는다** — 위 forecast 가 세 config-blind 셀에서 `low ≤ 1.60` 구성 그대로
proxy 를 이기므로 `1.60` 을 대체할 근거가 없다. **HE30/HEO30 로 교체하지 않는다.**

## 선행조건 종결

| # | 항목 | 상태 |
|---|---|---|
| ① | G7 parity 진단 | **완료** — 결손은 상류 패턴, G7 FAIL 유지(배포 veto) |
| ② | `FXYP` 잡음 | **해소** (승계) |
| ③ | distill teacher = `s1j` | **완료** — sha `9d4b489d…` 일치 |
| ④ | deck / GPU 등록 | **완료** — `lpopt_gpu1.inp`, GPU 1 |
| ⑤ | proxy 정의 고정 | **완료** — RANK = `fr_cal`, LEVEL = 출하 prior |
| ⑥ | G6b 바 + config-blind forecast | **완료** — 바 스탬프, 등록 minDet 0.132, forecast PASS |
| ⑦ | source sha 스탬프 | **부분** — 표는 남겼으나 `train.log`/`meta.json` 자동 인쇄는 **미구현(등록된 결손)** |

→ **`[3]` 실행 허용.**

## `[3]` 실행 기록 — arm 5 학습 투입 (2026-09-03 13:21:34 KST)

| 항목 | 값 |
|---|---|
| run dir | `~/lpopt_ws/runs/20260903_132134` (238) |
| 로그 | `runs/20260903_132134/train.log` · 래퍼 `runs/arm5_20260903_132134.log` |
| 런처 | `scratch/arm5_launch.sh` (arm 4 `runs/20260902_122746/run.sh` 의 구조 복제 + `SOURCE_SHA256` 스탬프) |
| PID | 3572047 (`lpopt.model.train`), 래퍼 3572040 |
| GPU | **GPU 1** (`CUDA_VISIBLE_DEVICES=1`), deck `lpopt_gpu1.inp` |
| distill teacher | `data/models/_v5_distill_soft.npz` sha `9d4b489d…` (`s1j`, 미변경) |
| init / split | `--init-from data/models/s1j` · `--split S1j` |
| source sha | run dir 의 `SOURCE_SHA256` (위 E.7-⑦ 표와 동일) |
| 예상 소요 | **약 2.5 h** — arm 4 실측 12:27:48 → 14:59:51 (2 h 32 m; featurize 2,102 s + 5 멤버 + per-cell 보정 6종) |

**커맨드 (D.4.2 대비 마지막 7개 플래그만 추가, 그 외 arm 4 verbatim):**

```
CUDA_VISIBLE_DEVICES=1 ~/lpopt_ws/venv/bin/python -m lpopt.model.train \
  --ensemble 5 --split S1j --cond-schema v8 --width 224 --n-blocks 8 --head-hidden 384 \
  --epochs 150 --num-workers 8 --device auto \
  --map-decoder multiscale --map-prior-residual --map-spectral-weight 0.3 --map-peak-weight 2.0 \
  --cyclen-physics-prior --quantile-heads --quantile-weight 0.2 \
  --promote-max-asm-bu --promote-fxy \
  --init-from data/models/s1j --trunk-finetune-lr-mult 0.05 \
  --distill-targets data/models/_v5_distill_soft.npz --distill-weight 0.4 \
  --distill-min-match-frac 0.5 --f-r-rank-weight 0.1 \
  --fxy-prior-on-predicted --fxy-select-weight 0.5 --warmup-epochs 2 \
  --fxy-rank-weight 3.0 --fxy-rank-cell gate --fxy-rank-margin-z 0.1 \
  --fxy-rank-min-gap 0.005 --fxy-rank-low-thresh 1.60 --fxy-rank-low-weight 3.0 \
  --fxy-select-band 0.50 \
  --out-dir runs/20260903_132134
```

`lpopt.model.al_retrain --arm5 --champion data/models/s1j` 가 합성하는 `[3]` 과 **토큰 단위로
동일**하다(원격 래퍼 `lpopt.remote --input <deck> train --` 대신 238 에서 직접 기동한 것만 다르며,
238 `src` 의 6개 모듈 sha 가 로컬과 일치함을 사전 확인했다).

**기동 확인 (train.log 머리, 2026-09-03 14:28 KST).**

```
=== featurized train=62575 val=12142 in 3897.2s ===          # arm 4 와 동일한 행 수
=== distillation: soft targets on 54057/62575 train rows (weight 0.4) ===   # arm 4 와 동일
=== f_xy head: prior f_xy = 1.2161*f_r -0.2486 on 5429 labelled train rows (r=0.9894, resid sd=0.0478)
=== f_xy rank hinge: w=3.0 cell=gate margin=0.1z min_gap=0.005 low<=1.6 x3.0 ===
=== f_xy selection band: q=0.5 on GATE cells (case_pair, feed) ===
=== trunk fine-tune: lr x 0.05 (trunk group lr=6e-05, head lr=0.0012); cyclen rows still gradient-masked ===
  [seed 20260716] epoch 0 ... fxyMAE=0.1048 fxyRho=0.843 fxyRhoBand=0.489/11c rkPairs=188 rkCells=28.3 sel=0.8889
  [seed 20260716] epoch 1 ... fxyMAE=0.0895 fxyRho=0.840 fxyRhoBand=0.462/11c rkPairs=197 rkCells=28.6 sel=0.8940
```

- **손실 항이 살아 있다 (E.8-⑦ 의 실패 계기 통과).** batch 당 유효 쌍 `rkPairs` = 188 / 197,
  기여 cell `rkCells` ≈ 28 — E.1.4 가 gate cell 에서 예측한 **측정 평균 195.3 / p05 118.8** 과
  일치한다. `rkPairs = 0` 인 batch 는 나타나지 않았다.
- **`--fxy-select-band 0.50` 이 gate cell 위에서 동작한다** (`fxyRhoBand=…/11c`), E.8-⑥ 이
  등록한 `case_pair` 단독 호출 불일치가 이 실행에서는 발생하지 않는다.
- **등록해 두는 미세 편차 두 가지 (사후 해석 금지, 기록만).**
  (i) 라벨된 `f_xy` train 행이 arm 4 의 **5,425** 에서 **5,429** 로 4행 늘었다 (store 가
  2026-09-02 이후 4행 증가). featurize 행 수(62,575 / 12,142)와 distill 매칭(54,057)은 동일하다.
  (ii) power prior 의 `within_cell_rho` 가 0.7000 → **0.6962** 로 움직였다. 쌍대성은 실질적으로
  유지되지만 **byte-level 쌍대는 아니다** — G6 판정에서 arm 4 와의 차이를 0.001 오더로 해석하지 않는다.
- **소요 시간 정정.** featurize 가 arm 4 의 2,102.8 s 에서 **3,897.2 s** 로 늘었다 (GPU 1 을
  다른 프로세스와 공유). 총 소요는 arm 4 의 2 h 32 m 이 아니라 **약 3.5–4 h** 로 예상한다.

---

# Amendment E — **E.7-(a) 근본원인·수정** · 2026-09-03 (append-only)

E.7-① 은 r2 `selection.json` 의 서빙 예측과 store 패턴의 s1j 재채점 사이의 괴리
(mean |Δf_xy| 0.01383, ρ 0.7277, wave 0 만 정확)를 **"채점된 패턴 ≠ 검증/기록된 패턴"** 으로
좁혔다. 본 절은 그 진단을 **반증**하고 실제 원인을 특정한다.

## 1. 재현 — 패턴은 byte-identical 하다 (가설 기각)

`runs/fpcamp_minfxy_e1e2_f121_r2/` 의 선택 100행 전부에 대해:

| 검사 | 결과 |
|---|---|
| `labels.jsonl` 패턴 == store `records.parquet` 패턴 | **100/100 일치** |
| 기록된 패턴 → `compute_record_id(pattern.canonical(), 'ga80', 'E1_E2', PRODUCE_DECK_KNOBS)` | **100/100 이 서빙 `record_id` 를 재생성** (wave 0–12, wave 당 8/8) |

`record_id` 는 패턴의 순수 함수(`Pattern.canonical()` 은 위치 기반이며 대칭 접기가 없다)이므로,
서빙 시각에 채점된 패턴과 store 에 기록된 패턴은 **wave ≥ 1 에서도 byte-identical** 이다.
합법화/정규화/repair/packing 어느 것도 사이에 끼어들지 않았다.

## 2. 실제 원인 — **캠페인이 wave 마다 서빙 체크포인트를 바꾼다**

`campaign.py::_run_wave` 의 step 7 (online update) 은 `WaveUpdater.update` 로 challenger 를
fine-tune 하고, 게이트가 수용하면 `self.champion_ckpt = str(self._save_champion())`
(**`lpopt/search/campaign.py:2666`**) 로 **서빙 가중치를 교체**한다.
r2 의 `logs/events.jsonl` 은 **wave 0–12 전부 `gate_accepted: true`** 이고
`models/champion_wave_00 … 12` 13개가 실제로 존재한다. 즉

- **wave 0** 은 기동 체크포인트(= `s1j`)가 서빙 → s1j 재채점이 **정확**,
- **wave k ≥ 1** 은 `champion_wave_{k−1}` 이 서빙 → s1j 재채점은 **다른 가중치와 비교**한 것.

**양성 대조 (238, CPU).** r2 wave 1 의 8행을 store 패턴 그대로, wave 1 을 실제로 서빙한
`champion_wave_00` 으로 재채점:

| 재채점 체크포인트 | mean \|Δf_xy\| | max | mean \|ΔF_r\| | max |
|---|---:|---:|---:|---:|
| `s1j` (E.7-① 이 쓴 것) | 0.009948 | 0.027842 | 0.029814 | 0.043650 |
| **`champion_wave_00` (실제 서빙)** | **0.000000** | **0.000001** | 0.002396 | 0.002437 |

`f_xy` 는 **정확히 재현**된다. 따라서 E.7-① 의 |Δ| 는 **서빙 경로의 결손이 아니라
오프라인 재채점의 사양 오류**다 — 캠페인의 "그 모델"은 하나의 체크포인트가 아닌데
단일 체크포인트로 13 wave 를 재채점했다.

> `F_r`(pred_mean[0]) 에는 올바른 체크포인트에서도 **거의 상수인 +0.0024** 잔차가 남는다
> (mean 0.002396 / max 0.002437). G7 이 채점하는 `f_xy` 는 0.000000 이므로 본 개정의 판정에는
> 들어가지 않으나, **미해결 저차 항목으로 등록**한다 (사후 해석 금지).

**따라서 진짜 결손은 계측의 결손이다**: `selection.json` 이 (i) 채점된 보드의 digest 도,
(ii) 그 wave 를 서빙한 체크포인트도 적지 않았기 때문에, 두 가설("다른 패턴" vs "다른 가중치")을
아티팩트만으로 구분할 수 없었고 잘못된 쪽으로 좁혀졌다.

## 3. 수정 (근본 지점, 최소 diff)

| 파일 | 변경 |
|---|---|
| `lpopt/search/verify.py` | `SCORED_DIGEST_KEY` · `ScoredPatternMismatch` · `assert_scored_pattern_parity()` 신설 |
| 〃 `WaveVerifier._eval_entry` | 덱 staging 직전, **try 바깥에서** — **검증되는 보드**가 채점된 보드인지 확인 (MASTER 호출 전에 실패하며, `error` 라벨로 흡수되지 않는다) |
| 〃 `outcome_to_record` | store 행 작성 직전 — **기록되는 보드**가 채점된 보드인지 확인 |
| `lpopt/search/campaign.py` | `WaveEntry.meta` 3개 생성지점 전부에 `cand.pattern.digest` 를 stamp |
| 〃 `_run_wave` 선두 | `self._serving_ckpt = str(self.champion_ckpt)` — step 7 의 champion swap **이전에** 서빙 체크포인트를 스냅샷 |
| 〃 `_write_wave_artifacts` | `selection.json` 에 행별 **`pattern_digest`**, wave별 **`served_checkpoint`** 기록 |

digest 가 stamp 되지 않은 항목(테스트 더블·구 resume 항목)에서는 무효(no-op)다 — 출처의 부재를
불일치의 증거로 쓰지 않는다. `_save_champion`/보정 코드는 건드리지 않았다.

**시험 (238, CPU).** `tests/test_campaign_stub.py` **34 passed**,
`tests/test_selection_pattern_parity.py` **4 passed**.
`tests/test_selection_replay.py` 는 **1 failed** 이나 **본 수정과 무관한 기존 실패**다 —
수정 전 소스를 복원한 격리 트리에서 재실행해 `assert max(ex_fr) < min(xp_fr)`
(**1.6093995144002475 < 1.6073469533854863**) 가 **자릿수까지 동일하게** 실패함을 확인했다.
이 단언은 `build_pool`/`score_pool` 의 수치 순서에 대한 것이고, 본 수정의 코드는 그 경로에서
한 줄도 실행되지 않는다 (원인은 모듈 주석이 등록한 champion/store 드리프트로 보이며,
본 개정의 범위가 아니다).

`tests/test_selection_pattern_parity.py` 신설 — (1) StubEvaluator 2-wave 캠페인에서
선택된 모든 후보에 대해 `selection.json.pattern_digest == labels.jsonl 패턴의 digest` 이고 그
패턴이 서빙 `record_id` 를 재생성함을 wave 별로 단언, `served_checkpoint` 존재 단언;
(2) 합법화 단계(parity guard) 단위시험 3종 — 무변경 통과 / 궤도-합법 1-swap 거부 / stamp 부재 시 무효.

## 4. 과거 캠페인 재채점 — **가능하다, 단 wave별 체크포인트로**

패턴 parity 가 증명되었으므로 **store 패턴에서 오프라인 재채점이 가능하다.** 단 재채점은
**해당 wave 를 서빙한 체크포인트**로 해야 하며, 단일 체크포인트 재채점은 반복하지 않는다.

| 캠페인 | 보존된 champion | 재채점 가능한 wave |
|---|---|---|
| **r2** `fpcamp_minfxy_e1e2_f121_r2` | `champion_wave_00…12` (13/13) | **wave 0–12 전부** (wave 0 은 `s1j`) |
| **r1** `fpcamp_minfxy_t6t4_f121_r1` | `champion_wave_{04,05,08,09,12}` (5/13) | wave 0(`s1j`) · 5 · 6 · 9 · 10 — **5/13 만** |

## 5. G7 에 대한 함의

- E.7-① 의 **G7 FAIL (mean |Δ| 0.01383 > 0.005, ρ 0.7277 < 0.99) 은 잘못 지정된 대조군에서
  나온 값**이다. 올바른 대조군(서빙 체크포인트)에서 wave 1 은 |Δf_xy| = 0.000000 이다.
- 따라서 **E.6-D 의 배포 차단 근거를 이 수치로는 더 이상 유지할 수 없다.** 다만 본 절은
  **판정을 바꾸지 않는다** — G7 은 r2 13 wave 전체를 각 wave 의 서빙 체크포인트로 재측정한 뒤에
  다시 스탬프한다 (E.7-⑨ 의 `campaign rescore` 가 그 계측기이며, `served_checkpoint` 필드가
  이제 그것을 well-posed 하게 만든다).
- 재측정 전까지 G7 은 **미측정(unstamped)** 이며, "PASS" 로 읽어서는 안 된다.
- 본 수정은 **앞으로의** 캠페인에서 패턴 괴리를 wave 발생 시점에 실패시키므로, 같은 종류의
  오진이 반복되지 않는다.

**운영 기록.** 작업 중 238 `~/lpopt_ws/src/lpopt/search/campaign.py` 가 14:41:57 KST 에 다른
트랙에 의해 덮어써져 본 수정이 일시 소실되었다(diff 상 외부 변경은 없었다). 재전송 후
`campaign.py` / `verify.py` / 신규 테스트 3개 파일의 sha256 **로컬 == 238 일치를 재확인**했다.


---

# Amendment E — **E.7-(a)-2 G7 재측정** · 2026-09-03 (append-only)

E.7-(a) §5 가 요구한 대로 r2 (`fpcamp_minfxy_e1e2_f121_r2`) 의 **13 wave 전부**를
**각 wave 를 실제로 서빙한 체크포인트**로 재채점하고 G7 을 다시 스탬프한다.
E.7-① 의 단일 체크포인트(s1j) 재채점은 폐기한다.

**계측기.** `data/reports/g7_rescore_r2_20260903.py` (독립 스크립트, lpopt 코드 무수정),
238 CPU 실행. 패턴 출처는 **store `records.parquet`** (`selection.json` 의 `record_id` 키).
서빙 경로는 캠페인과 동일: `PosValCnnBackend.from_dir(..., device="cpu")` → `.predict()`
→ `acquisition.predict_fxy()`; 비교 대상은 `selection.json` 의 `fxy_mean` 및 `pred_mean[0]`.
체크포인트 배정: wave 0 → `data/models/s1j`(기동 챔피언), wave k≥1 → `champion_wave_{k−1}`.
`fxy_source` 는 기록·재채점 **13/13 모두 `head`**.

## 1. wave 별 결과 (n = 100 행)

| wave | 서빙 체크포인트 | n | mean \|Δf_xy\| | max \|Δf_xy\| | ρ | mean \|ΔF_r\| | F_r cell affine shift |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `s1j` | 8 | 0.0000003 | 0.0000005 | 1.0000000 | 0.000027 | -0.002390 |
| 1 | `champion_wave_00` | 8 | 0.0000003 | 0.0000005 | 1.0000000 | 0.002396 | +0.000000 |
| 2 | `champion_wave_01` | 8 | 0.0000002 | 0.0000005 | 1.0000000 | 0.002406 | +0.000000 |
| 3 | `champion_wave_02` | 8 | 0.0000003 | 0.0000004 | 1.0000000 | 0.002383 | +0.000000 |
| 4 | `champion_wave_03` | 8 | 0.0000003 | 0.0000006 | 1.0000000 | 0.002383 | +0.000000 |
| 5 | `champion_wave_04` | 8 | 0.0000003 | 0.0000005 | 1.0000000 | 0.002398 | +0.000000 |
| 6 | `champion_wave_05` | 8 | 0.0000003 | 0.0000005 | 1.0000000 | 0.002385 | +0.000000 |
| 7 | `champion_wave_06` | 8 | 0.0000002 | 0.0000004 | 1.0000000 | 0.002400 | +0.000000 |
| 8 | `champion_wave_07` | 8 | 0.0000002 | 0.0000004 | 1.0000000 | 0.002398 | +0.000000 |
| 9 | `champion_wave_08` | 8 | 0.0000003 | 0.0000005 | 1.0000000 | 0.002401 | +0.000000 |
| 10 | `champion_wave_09` | 8 | 0.0000003 | 0.0000005 | 1.0000000 | 0.002397 | +0.000000 |
| 11 | `champion_wave_10` | 8 | 0.0000002 | 0.0000005 | 1.0000000 | 0.002407 | +0.000000 |
| 12 | `champion_wave_11` | 4 | 0.0000002 | 0.0000003 | 1.0000000 | 0.002386 | +0.000000 |

## 2. 통합 판정 — **G7 PASS**

| 통계 (13 wave 통합, n = 100) | 값 | 등록된 바 | 판정 |
|---|---:|---:|:--:|
| mean \|Δf_xy\| | **2.614e-07** | ≤ 0.005 | **PASS** |
| ρ (서빙 vs 재채점 f_xy) | **1.0000000000** | ≥ 0.99 | **PASS** |
| max \|Δf_xy\| | 6.327e-07 | — | — |

**G7 = PASS.** f_xy 는 13 wave 전부에서 **기계 정밀도 수준으로 재현**된다
(max \|Δ\| = 6.33e-07, 등록 바보다 4 자릿수 여유). E.7-① 의 FAIL
(mean \|Δ\| 0.01383, ρ 0.7277) 은 E.7-(a) 가 특정한 대로 **대조군 오지정의 산물**이었고,
올바른 wave 별 체크포인트에서는 잔존하지 않는다.

## 3. `F_r` 의 +0.0024 잔차 — **상수이며, 출처는 보정(cell affine)이다**

| | mean signed ΔF_r | sd | mean \|ΔF_r\| |
|---|---:|---:|---:|
| 보정 적용 (기본 서빙 경로) | +0.002204 | 0.000650 | 0.002206 |
| 보정 해제 (`apply_fr_calibration=False`) | +0.002395 | — | 0.002395 |

**상수인가 — 그렇다.** wave 1–12 의 wave 내 sd 는 0.000016–0.000036 으로
`selection.json` 의 `pred_mean` **4자리 반올림 바닥(mean \|Δ\| ≈ 0.000025)** 과 같은 크기다.
즉 반올림을 걷어내면 잔차는 **행·wave 를 가로질러 상수 +0.00239** 다.
(통합 sd 0.00065 는 wave 0 의 0 잔차가 섞인 데서 온 것이지 산포가 아니다.)

**출처 — 보정이다, raw 가 아니다.** 결정적 증거:

- **wave 0** (`s1j`): 보정 적용 시 ΔF_r = **+0.000001** (sd 0.000030 = 반올림 바닥) — **정확 재현**.
  같은 행을 보정 해제로 재채점하면 **+0.002391** 로 벌어진다. `s1j` 의 per-cell F_r affine
  shift 는 이 100행에서 **−0.002390**.
- **wave 1–12** (`champion_wave_*`): affine shift가 **정확히 0.000000** — 즉 보정이
  **적용되지 않았다**. 보정 적용·해제 재채점이 **동일**하고, 둘 다 서빙값보다 **+0.0024 높다**.
- 파일 목록이 이유를 말한다. `s1j/` 는 `f_r_calibration.json` · `cell_calibration.json` ·
  `cbc_calibration.json` … 를 갖고 있으나 `champion_wave_00 … 11/` 은
  `backend.json` · `calibration.json` · `ensemble.json` · `feature_ood.json` **네 개뿐**이다
  (per-cell 보정 아티팩트 **0개**).

따라서 잔차의 부호와 크기는 **잃어버린 per-cell F_r affine 보정과 정확히 일치**한다
(+0.00239 ≈ −(−0.002390)). 서빙 시각의 in-memory 백엔드는 `s1j` 에서 로드한 보정을
계속 들고 있었고, `_save_champion` 이 쓴 챔피언 디렉터리는 그 아티팩트를 **동반 저장하지
않는다**. 그래서 `from_dir` 오프라인 재채점만 보정이 빠진다 — 이는
`lpopt/model/model_api.py:154-158` 의 기존 주석("campaign served an `s1j` descendant with the
per-cell LEVEL calibration GONE")이 등록해 둔 현상과 같은 것이다.

> **범위.** 이는 **오프라인 재현성**의 결손이지 서빙 예측의 결손이 아니다 (서빙은 보정을 썼다).
> G7 이 채점하는 `f_xy` 는 head 출력이라 이 경로를 타지 않으며 영향이 0 이다.
> 사후 해석 금지 원칙에 따라 **미해결 저차 항목으로 유지**하되, 원인은 이제 특정되었다:
> **`_save_champion` 이 per-cell 보정 세트를 챔피언 디렉터리에 복사하지 않는다.**

## 4. E.6-D (배포 차단) 에 대한 함의

- E.6-D 의 배포 차단은 **G7 FAIL 을 근거로 삼고 있었고, 그 근거는 이로써 소멸한다.**
  G7 은 이제 **측정되었으며 PASS** 다 (미측정 상태 해소).
- 따라서 **G7 은 더 이상 배포를 막지 않는다.** 다만 본 절은 **G7 하나만** 재스탬프한다 —
  E.6-D 의 다른 구속(G6b veto 등)은 건드리지 않으며, 배포 여부는 그 게이트들이 각자 판정한다.
- **재채점 없는 재판정 금지 조항(E.6-D)** 은 충족되었다: 본 재측정은 배포 판정 이전에,
  등록된 바(0.005 / 0.99)를 **변경 없이** 적용해 수행되었다.
- 남는 조건: 위 §3 의 F_r 보정 누락은 **`F_r` 을 쓰는 오프라인 재현·감사 경로**에 한해
  유효하며, 챔피언 저장이 보정 세트를 동반하도록 고치기 전까지 과거 챔피언의 `F_r`
  오프라인 재채점값은 **+0.0024 만큼 비보정**임을 명시해야 한다.

**아티팩트.** `data/reports/g7_rescore_r2_20260903.json` (wave 별 + 통합 통계),
`data/reports/g7_rescore_r2_20260903.py` (계측기).
238: `~/lpopt_ws/scratch/r2_ckpts/{champion_wave_00..11, sel/}`, `~/lpopt_ws/data/models/s1j`.
