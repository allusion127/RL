# 능동 프론티어 루프 (Active Frontier Loop) — 원칙의 정식화 · 구현 현황 · 갭 사양

**작성** 2026-09-03 · **성격** 설계 사양 + 갭 분석. **코드 변경 0 · MASTER 0 · DeCART 0 · 로컬 연산 0.**
**선행** `phaseA2_ood_conformal_20260829.md` · `sample_efficiency_kpi_20260903.md` ·
`assembly_sigma_chain_retrodiction_20260903.md` · `assembly_on_demand_tasks_20260903.md` ·
`feedgrid_pathfinder_20260815.md` · `policy_v31_prereg_20260831_DRAFT.md`

---

## 0. 원칙 (오너 진술, 2026-09-03, 축어)

> "기존 AI 및 surrogate 모델들의 최대 단점은 외삽 성능이 크게 떨어진다는 거야. 하지만 우리
> 프론티어 AI 모델은 내삽 범위를 벗어날 때 새로운 실제 계산을 통해 노물리 특성들을 학습
> 데이터로 해석하고 이를 최적화에 사용하는 능동적인 모델인거지."

이 문장을 이 저장소의 언어로 옮기면 **세 개의 함수와 그 사이의 배선**이다.

| 절 | 함수 | 지금 있는 것 | 지금 없는 것 |
|---|---|---|---|
| "내삽 범위를 벗어날 때" | **detect** — OOD·conformal·trust region·mesh tier·σ 사다리 | **거의 전부 구현됨** (§1) | 셀 단위 *정량* coverage 지표, geometry 채널 (퇴화) |
| "새로운 실제 계산을 통해" | **escalate** — campaign / produce / wave / bootstrap / DeCART | **전부 구현됨, 전부 수동 기동** (§2) | 판정 → 라벨 계획으로의 **자동 변환** (§4a) |
| "학습 데이터로 해석하고 최적화에 사용" | **assimilate** — merge → split → retrain → gate → promote | **전부 구현됨, 반자동** (§3) | 자동 트리거 · VOI 스케줄러 (§4b·§4c) |

**핵심 KPI는 정확도가 아니라 표본효율이다.** 등록된 실측:
SA 13런 **36,106 MASTER 콜 / 2,357 wall-hour** 로 F_r 1.55 미달(최저 1.5656);
lpopt 는 8셀 중 7셀에서 셀 기록 +0.005 를 **27–126 콜**(중앙값 57)에 재현
(`sample_efficiency_kpi_20260903.md` §1-1, §1-4). 목적함수는 오너가 정한 MASTER 실측량
(`F_r` / `F_xy` 등)이고, 장기 목표 `F_r 1.35` 는 LP 축이 아니라 **집합체 설계축**으로만
도달 가능하다 (LP 축 바닥은 T6_T4 1.4605, 900콜로도 미개선 — 동 §3).

> **본 문서가 반대하는 단 하나의 동작:** OOD 판정을 *거절*로만 소비하는 것.
> `reject` 는 안전장치이지 학습이 아니다. 능동 모델은 같은 판정을 **라벨 주문서**로 읽는다.

---

## 1. 외삽 감지 — 지금 구현되어 있는 신호들

각 신호가 **무엇을 말하는가**와 **발화 시 지금 무엇을 하는가**를 분리해 적는다.

### 1.1 OOD guard (feature envelope) — `[acquisition] ood_policy`

* **신호**: 후보 패턴이 챔피언 학습 개체군의 **연료타입 특징 포락**을 벗어나는가.
  `PosValCnnBackend.feature_ood_types` → `ood_guard.feature_ood_vecs` 를 **그대로** 호출
  (`lpopt/search/acquisition.py:2486-2521`). 앙상블 σ 로는 답할 수 없는 질문이다 —
  멤버 전원이 같은 manifold 를 공유하므로 off-manifold 에서 *일치하면서 함께 틀린다*
  (동 파일 `:2410-2424` 의 설계 주석).
* **3분류, `unknown` 은 안전값이 아니다**: `absent`(백엔드에 probe 없음, 아무것도 flag 안 함) /
  `available`+정상 / `available`+예외 → **fail-closed 로 flag** 하고 `ood_guard_errors` 계수
  (`:2513-2519`).
* **발화 시 동작** (`apply_safety_shield`, `:2626-2745`):

| `ood_policy` | 동작 | 코드 |
|---|---|---|
| `"warn"` **(기본값)** | flag 를 계산해 pool·dossier 에 **붙이기만** 한다. 점수 불변, 제거 없음 | `:2462`, `:2682-2686` |
| `"escalate"` | flagged 후보의 `exploit`·`rank` 를 `-inf` → exploit 슬롯 탈락 + `_run_wave` elite carry-over(`isfinite(exploit)`) 탈락. **`raw_epi`·`p_feas` 는 불변이므로 explore/control 슬롯에는 남는다** | `:2688-2695` |
| `"reject"` | 풀에서 제거 | `:2696-2700` |

* **현행 덱은 전부 기본값이다.** `SafetyShield.active` 가 `False` 이면 `_run_wave` 는 shield 를
  **호출조차 하지 않는다** (`campaign.py:2361`, `acquisition.py:2472-2474`).
* **치명적 전제**: 현재 학습 개체군은 **geometry 채널이 상수**라 포락이 퇴화 `[0,0]` 이고,
  기존 ga80/paramA 셀에서는 **flag 가 0** 이다 (`phaseA2_ood_conformal_20260829.md` §5.5;
  `assembly_on_demand_tasks_20260903.md` 작업 #7 이 같은 이유로 geometry 채널을 제외).
  → **OOD guard 는 신규 연료형이 들어오는 순간에만 신호가 된다.** 슬라이스 Z 가 그 순간이다.

### 1.2 Conformal chance constraint — `conformal_gate` / `conformal_alpha`

* **신호**: 모델의 **분포무관 유한표본 상한** `U_c(x)` 가 인허가 한계 `L_c` 를 넘는가.
  `predict_interval` → `model/conformal.py` 를 그대로 소비 (`acquisition.py:2523-2545`).
* **게이트 축은 교집합 4개**: `CONFORMAL_GATE_AXES = (("f_r",0),("cbc_max",1),("f_q",2),("ao_abs",4))`
  (`:2437-2447`). `max_pin_burnup` 은 이미 78.0 vs 인허가 80.0 의 model-margin haircut 으로
  screen 되므로 **이중계상 방지로 제외**; `cyclen` 은 양측 band 라 단측 상한 게이트 대상 아님.
* **발화 시 동작**: 유한 상한이 limit 을 넘으면 **제거**, 축별 계수 `conformal_rejected_by_axis`
  (`:2705-2741`). **유한 상한이 없는 축은 제거 사유가 아니다** — 기존 `μ+κσ` screen 이 그대로
  서고 그 축은 `ScoredPool.conformal_unfit` 에 기록된다 (`:2270-2295`, `:2731-2737`).
  즉 fail-open 이되 **침묵하지 않는다**.
* **F_xy 는 계약 밖**: head **와** `conformal.json` 의 `f_xy` 항목이 **둘 다** 있을 때만 bound 를
  만든다 (`fxy_conformal_upper`, `:2546-2594`). 없으면 `conformal_fxy: "proxy"` 로 남기고
  **proxy σ screen** 을 유지 — 예측 F_r 의 affine 사상
  `1.2176·F̂_r − 0.2519`, 잔차 sd `0.0476`, 폭 계수 `k=3.0` (`:645-656`, `:673-675`).
  현재 `CONFORMAL_TARGETS` 에 `f_xy` 가 **없으므로** F_xy 축은 어떤 arm 에서도 proxy 다.
* **인도 반영**: `delivery.json` 의 `ranked`/`excluded` **전 항목**과 `best`/`best_overall` 이
  `ood_flag`(True/False/**None**=미평가) 와 `conformal_unfit_axes` 를 싣는다. **`is_deliverable`
  술어는 바뀌지 않았다** (`phaseA2…` §1.6, §4-3). `conformal.json` 없는 챔피언은 gated 축
  전체가 `conformal_unfit` 으로 나열된다 — 결함이 아니라 "여기는 교정되지 않았다"는 정직한 보고.

### 1.3 Trust region τ — 지지격자 게이트 (`lpopt/search/acquisition.py:2127-2196`)

* **신호**: 후보의 `(feed, e_core-bin)` 이 스토어가 라벨로 지지하는 bin 인가.
  `in_region` / `is_frontier`(1 feed-step 또는 1 e-band 밖) / `sigma_scale`
  (frontier 면 `frontier_sigma_inflation` 배로 **σ 를 부풀림** — 즉 감지가 *점수*로 들어가는
  유일한 경로) / `observe` 가 `promote_after` 라벨에서 bin 을 지지집합으로 승격.
* **발화 시 동작**: σ 팽창 + `region` 플래그. **제거하지 않는다.** 캠페인 셀 자신은 항상 in-region.

### 1.4 Mesh Tier 게이트 · `in_distribution` (`scoping_mesh.py`)

* `TIERS = (("tier1",1.55,1600.0),("tier2",1.65,1800.0),("tier3",1.80,2200.0))` (`:87-89`).
  Tier-1 이 프로젝트 제약집합이고 **Tier-3 은 관측 전용 · 인허가 주장 없음** (`:85-86`).
  셀마다 `tier_reached` 와 **어느 제약이 묶는지**를 기록 (`:432`, `:644-646`).
* `in_distribution = bool(feed == 121)` (`:613`, `:634`) — **이것이 현재의 유일한 셀 단위
  내삽/외삽 라벨이고, 문자 그대로 "feed 가 121인가"다.** feed 축 실제 분포는 이미 훨씬 넓다
  (97~141, 121 은 61%; `feedgrid_pathfinder_20260815.md` §0) → **이 플래그는 과보수적이며
  §4d 의 coverage 지표로 대체되어야 한다.**
* `mesh_vs_db.py` 가 DB 진실값 대비 `model_bias` / `cell_verdicts.csv` 를 만든다 —
  셀별 **편향 부호와 크기**가 여기 있고 autoeng 의 사전 마크(`mesh_min_pred_f_r`,
  `corrected_floor`)로 소비된다 (`autoeng.py:386-568`).

### 1.5 per-cell 교정 가용성 (`lpopt/model/cell_calibrate.py`)

* 6개 아티팩트 (`cell_calibration.json`(cyclen) / `f_r` / `cbc` / `f_q` / `ao_abs` /
  `flatness`) 가 **셀별 affine 보정**을 담는다. 라벨이 적은 셀은 `_crossfit_choice` 가
  intercept-only 로 떨어지고, 더 적으면 pooled fallback 으로 떨어진다 →
  **"이 셀에는 교정이 없다"가 곧 외삽 신호다.**
* 2026-08-29 serve-path featurization 수정이 **`backend.predict` 경유로 적합된 6종 전부를
  무효화**했고 재적합으로 복구했다 (`servefix_calibration_refit_20260829.md` §1-2).
  `calibration.json`(σ isotonic + Platt)은 학습 featurization 을 타므로 **무영향**.
  → **교정 아티팩트의 `split`/`source` 필드는 감지 신호로 읽을 수 있으나 지금 아무도 안 읽는다.**

### 1.6 σ_chain — 모델-프리 사슬의 짝지은 차분 오차

* **측정 완료** (`assembly_sigma_chain_retrodiction_20260903.md`):
  `σ_chain,paired` = **0.071–0.077 (F_r)** / **0.077–0.083 (F_xy)**, LOO 재적합 기준.
  절대 MAE 0.054 → **짝지으면 0.077 로 오히려 커진다** (지배항 `node_peak` 잔차가
  후보마다 contrast 가 달라 공통모드가 아니다, 동 §4).
* **사다리 판정 = 세 번째 칸(> 0.020) = 크기 판정 불가.** 코드에 박혀 있다:
  `need.py:80-118 sigma_ladder_verdict` → `triggers_valid False`,
  `need.py:119-135 launch_allowed` 가 **F1 에서 발사 거부**.
* **살아남은 용도 2개**: 하드 게이트 `contrast ≥ 0.026` (B3 반례가 그 유효성의 증거,
  `opscreen_chain.b3_counterexample`) 와 FLOOR 라벨 `Fr_flr = 1.03·1.2085·FF_hot`.
  **금지되는 용도 1개**: `ΔF̂_xy` 의 *크기*로 후보를 자동 발사하는 것.
* 헤드도 못 이긴다: s1j 단일점 Δ +0.0333 / +0.0439 이지만 셀내 순위는 **RANK 3/3 FAIL**,
  pinbu r1 셀내 ρ **−0.11** (동 §5). → **§4b 의 VOI 는 예측 크기가 아니라 커버리지·정보량
  위에 세워야 한다.**

---

## 2. 실제 계산으로의 에스컬레이션 — 지금 구현된 도구들

전부 존재하고 전부 **사람이 기동한다.** 각 항목: 무엇을 라벨하는가 / MASTER·DeCART 비용 /
라벨이 스토어에 들어가는 경로.

| 도구 | 라벨하는 것 | 비용 (실측) | 스토어 진입 |
|---|---|---|---|
| **campaign wave** (`lpopt optimize`) | 목적함수 축 위의 후보 코어 (converged FOM 7축 + 맵) | 라운드당 **100 콜** = 12 wave × 8 + reserve 4 (`fpcamp_minfxy_E1E2_f121_r2_199.inp:278-286`) | `runs/<tag>/labels.jsonl` → `lpopt merge-store --from` |
| ↳ 슬롯 구성 | exploit 5 / explore 2 / control 1 (동 `:281-284`); τ 는 feasible 라벨 등장 후 분위수 스케줄 (`acquisition.py:2917-2927`), 게이트는 exploit `p_feas ≥ τ`, explore `≥ τ/2` (`:3005-3015`) | — | — |
| **produce / anchor deck** (`lpopt produce`) | **목적함수 중립 DoE 라벨** — 커버리지 채움 | `anchors_meshv3_198.inp`: 198 박스에 **214 chain** (184 격자 + 30 f101 교정, 상한 250); `produce_fxyera_r1` 는 F_xy 최초 라벨 배치 (arm B 26.3% 를 F_r 1.50–1.60 대역에 배치) | 동일. **produce 만이 `CaseAssetResolver.promote()` 를 호출**해 level-1 restart 캐시를 채운다 (`produce.py:1255` → `:1295-1329`; campaign 은 못 한다. `assets.py:867` 의 `promote()` 는 그 밖에 호출자가 없다 — `feedgrid_pathfinder_20260815.md` §0-3) |
| **intervention wave** | 짝지은 개입 효과 (parent-blocked) — 정책 코퍼스의 **게이트 parent 를 만드는 유일한 원천** | 셀당 **160 콜** (`sample_efficiency_kpi…` §1-3) | `intervention_wave_r1_rows.csv` + 스토어 병합 |
| **ablation / batchswap 열거 wave** | 1-move / 1-batch-swap **전수 열거** | ablation 150 콜 · batchswap 220 콜 | 동일 |
| **pinbu wave** | `max_pin_burnup` (verify 가 `enable_pin_burnup=False` 하드코딩이라 캠페인 행은 전부 null) + 재현성 반복 | r2 manifest **30 target** | `pinbu_wave.py run` → 병합 |
| **bootstrap chain** (`lpopt/design/bootstrap.py:135 make_band_restart`) | **신규 (pair, feed) 셀의 band seed** — cy1 fresh-core → EquilibriumRunner 반복 → 최종 `MAS_RST` | 셀당 cy1 + reload 사이클 (`bootstrap_max_cycles` 16, T6_T4 선례 11 사이클); 슬라이스 Z 는 **9회** (기존 8 pair 재부트스트랩 + 신규 Z1_Z2 1) | `bases/<folder>/` 에 restart 저장 → 이후 모든 produce/verify 가 재사용 |
| **신규 집합체 축** (슬라이스 Z) | 새 연료 설계 그 자체 | **DeCART2D 2 케이스(181)** → HGC 게이트(238) → 라이브러리 재빌드(199) → **부트스트랩 9** → **캠페인 2 arm × 100 = 200 MASTER** + MTC ~10 → 총 **~210 MASTER + 부트스트랩 9**, 약 2–2.5 근무일 (`assembly_slice_Z_prereg_20260903_DRAFT.md` §9); 구현 임계경로 **≈ 9.5 일** (`assembly_on_demand_tasks_20260903.md` 부록 2) | HGC → `MAS_XSL`/`MAS_HFF` 재빌드 → `designs.json` → 부트스트랩 → 캠페인 라벨 |

**오케스트레이터.** `autoeng.py` 가 셀 하나를 **precheck → prereg(덱 sha256 고정) → probe 8콜 →
open 100콜 → harvest+merge → retrain+gate → mesh 갱신 → report** 로 끝까지 몬다
(`autoeng.py:1158-1330`). 예산: `probe_budget 8` / `open_budget 100` / `master_budget_total 500`
(`autoeng.toml:28-33`). 셀 순서는 전이 거리로 정렬 (`order_targets`, `:1373-1433`),
"개방된 셀"은 **로그가 아니라 스토어**에서 읽는다 (`opened_cells_from_store`, `:1338-1372`,
`OPENED_MIN_FEASIBLE = 5`).

---

## 3. 새 라벨의 학습 반영 — 현행 케이던스

```
merge-store ──► build_split_S1b(--holdout-new-campaigns) ──► al_retrain(--arm4/--arm5)
     │                                                              │
     │                                                       distill 캐시 갱신
     ▼                                                              ▼
 cell_calibrate 재적합                              gate-promote (--check-only → 승격)
     │                                                              │
 mine_policy_corpus --v31 ──► G1/G2/G3/RANK 게이트 ──────────────────┘
```

| 단계 | 진입점 | 지금 자동인가 |
|---|---|---|
| 병합 | `lpopt merge-store --from <cell>/data` (dry-run 먼저) | **autoeng 안에서는 자동** (`autoeng.py:1252-1258`), 그 밖은 수동 |
| split 증분 | `build_split_S1b.py --parent <p> --name <s> --holdout-new-campaigns [--write]` — **기본이 dry-run**, `MUST_HOLD` 4종(옛 val 유지 / 옛 train 유지 / …) 전부 통과해야 기록 (`:273-279`), `_self_check` 가 새-only 홀드아웃 절을 검증 (`:70-125`) | 반자동 |
| 재학습 | `al_retrain.plan_al_retrain(..., arm4=, arm5=)` — arm4 = D.4.2 fine-tune(`--init-from` + trunk LR mult + `fxy_prior_on_predicted`), arm5 = **arm4 축어 + 6개 rank knob + `--fxy-select-band 0.50`** (`al_retrain.py:341-406`, 상수 `:123-135`). 표준 앙상블 레시피는 `TRAIN_RECIPE` 로 동결, **라운드당 델타는 `--split`/`--ts` 뿐** (`autoeng.py:97-110`) | 수동 (autoeng 는 명령만 조립) |
| distill 캐시 | `al_retrain.refresh_distill_cache` → `data/models/_v5_distill_soft.npz`; **arm4 는 s1j 캐시여야 하고 stale s1i 캐시는 오염** (`:308-340`, `:377-380`) | 수동 |
| 교정 재적합 | `train.fit_cell_calibrations` 가 학습 후 호출; 서빙 경로가 바뀌면 **전량 무효** → 수동 재적합 (`servefix_calibration_refit_20260829.md` §2) | 수동 |
| 정책 코퍼스 | `mine_policy_corpus.py --v31` (`:2219 cmd_v31`) → `data/policy/steps_v31.parquet`; **기본이 DRY RUN**, `--apply` 필요 (`:2327`) | 수동 |
| 게이트 | `lpopt gate-promote --check-only` → `gate_no_regression` + `gate_legacy_tail` (`cli.py:1658-1743`). PASS 일 때만 `state.json` + `lpopt.inp` **원자적 승격** | autoeng 안에서 자동, **FAIL 시 `retrain_promote_fail` 휴먼 게이트** |
| 정책 게이트 | v3.1 절 1(parent-blocked AUC) / 절 2A·2B(NDCG@4-of-8) / 절 3(**within-cell** M4) / 절 4(서빙 스케일 p90−p10 ≥ 0.15) + r2 의 **RANK gate** 규율 (`policy_v31_prereg_20260831_DRAFT.md` §6a). 처분은 `shadow_v3.1` 계측까지 | 수동 |
| 지도 갱신 | `scoping_mesh.py --model <arm>` (**MASTER 0콜**) → `mesh_vs_db.py --model <arm>` → `cell_verdicts.csv` | autoeng 안에서 자동, **승격 실패 시 skip** |

**측정된 사실 두 개가 이 절의 설계를 구속한다.**
(1) **탐색 캠페인은 정책 게이트 parent 를 하나도 만들지 않는다** — 실측 100 콜당 0개
(`policy_v31_prereg…` §1d, `_wf_scratch/wf_campaign_yield.csv`). parent 를 사는 것은 **개입 wave** 다.
(2) v3.1 라운드에서 **MASTER 콜을 하나도 쓰지 않고 게이트 pool 을 38/23 → 79/41 로 키운 조치는
cross-fit(W0)** 이었다 (동 §6-W0). → **"라벨을 더 사기 전에 통계 재사용부터"가 등록된 우선순위다.**

---

## 4. 갭 분석 — 루프를 **자율적·능동적**으로 만들기 위해 없는 것

### 4a. 판정 → **라벨 계획**으로의 변환기 (없음)

지금 OOD/conformal/trust-region 판정의 종착지는 셋뿐이다: 경고 · 강등 · 제거.
**"그러니 여기를 실제로 계산하라"는 출력이 없다.** 필요한 것은 wave report 와 mesh verdict 를
읽어 **등록 가능한 라벨 주문서**를 쓰는 얇은 계층이다.

* **입력**: `waves/wave_NN/selection.json` 의 `shield` 블록(정책·guard 상태·축별 제거 계수·
  `n_candidates → n_remaining`) + `delivery.json` 의 `conformal_unfit_axes` +
  `cell_verdicts.csv` + 셀별 스토어 행수.
* **출력** (신규 `data/autoeng/<run>/label_plan.json`): `{어느 계산(campaign/produce/
  intervention/pinbu/bootstrap/DeCART), 몇 콜, 어느 셀·어느 축, 왜(발화한 신호와 그 수치),
  예상 정보이득}`.
* **배선 지점은 이미 있다**: `autoeng.toml:57` 의
  `pause_for_approval = ["new_assembly","retrain_promote_fail","budget_exceeded"]` 에
  **`ood_escalation` 을 네 번째 게이트로 추가**하고, `AutoEngineer._pause`(`autoeng.py:1463-1467`)
  가 계획 JSON 을 첨부한 채 멈추게 한다. 승인 후 `--resume` 이 그대로 이어받는다.
* **fleet 규칙은 손대지 않는다**: `_fleet_guard`/`guard_argv`(`autoeng.py:274-303`)가 모든 argv 를
  `forbidden = ["HOST_181","HOST_198","40.181","40.198"]` (`autoeng.toml:66`) 로
  선별한다. **집합체 축은 181 DeCART 를 요구하므로 정책 예외를 명시적으로 기록하기 전에는
  계획서만 쓰고 실행하지 않는다** (`assembly_on_demand_tasks_20260903.md` 부록 1, 작업 #10b).
* **신규 집합체는 이미 HALT 한다** — `plan_cell` 이 `new_assembly` 타깃에 precheck 한 스텝만
  두고 `gate="new_assembly"` 로 멈춘다 (`autoeng.py:1174-1180`). **그 HALT 를 "거절"이 아니라
  "주문서 제출"로 바꾸는 것이 이 항목의 전부다.**

### 4b. 정보가치(VOI) 기준 — MASTER 콜당 기대 프론티어 이득 (없음)

세 행동 중 하나를 고르는 **비교 가능한 스칼라**가 없다:
(i) 현재 셀 더 파기 · (ii) 커버리지 확장(새 feed/e_core bin) · (iii) 새 집합체 열기.

* **지금 없다는 근거는 코드다**: lpopt 의 모든 목적함수는 **케이스를 고정한 채 패턴을 랭킹**한다 —
  `score_min_fxy`(`acquisition.py:841`), `score_min_fr_max_cycle`(`:489`),
  `MinFuelCostSpec`(`:1054`), `score_flat_power`(`:1799`). **`FuelDesign` 축 위의 목적함수 항은
  0개**이고 "설계 1달러당 기대개선"은 정의된 적이 없다 (`assembly_on_demand_tasks…` §8 서문).
* **제안 형태** (등록 대상):
  `VOI(a) = E[Δbest_objective | a] / calls(a)`, 세 행동 각각에 대해
  - (i) `Δ/100calls` 의 최근 2라운드 추정 (`sample_efficiency_kpi…` §3 이 이미 산출).
    **< 0.002 이면 "바닥"으로 선언하고 예산을 이전** — 이것이 이미 등록된 규칙(K4)이다.
  - (ii) 지지 bin 승격 확률 × 그 bin 의 DB 프론티어 여유 (`mesh_pareto.csv` + `TrustRegion.
    promote_after`).
  - (iii) **크기 예측을 쓰지 않는다.** σ_chain = 0.077 이 `ΔF̂_xy` 크기 기반 자동 발사를 금지했다
    (§1.6). 대신 **부호/구조 게이트**(`contrast ≥ 0.026`)와 **FLOOR 라벨**만으로 후보를 정렬하고,
    선택 자체는 물리실험으로 판정한다.
* **선행 의존**: 작업 #4(screener) → #7(need signal) → #8(가상 로스터) → #9(발사 규칙).
  현재 `need.py` 는 **골격만** 있고 `need_signal` 은 `None` 을 돌려준다 (`need.py:136`).

### 4c. 자동 재학습·승격 트리거 (없음) — 단, **정직한 게이트 규율은 유지**

* **트리거 제안**: 마지막 승격 이후 (a) 신규 converged 라벨 ≥ N_new **또는** (b) 새로 개방된
  셀 ≥ 1 **또는** (c) `ood_flagged > 0` 인 wave 발생 → `build_split_S1b --holdout-new-campaigns`
  → `al_retrain` → `gate-promote --check-only`.
* **규율(변경 금지)**: ① 게이트는 `--check-only` 가 먼저이고 PASS 에서만 승격
  (`autoeng.py:1268-1290`). ② FAIL 은 **휴먼 게이트**로 남긴다 (`retrain_promote_fail`).
  ③ split 은 `MUST_HOLD` 전부 통과 전 기록 금지. ④ **게이트 폴드를 보고 임계를 정하지 않는다**
  (`policy_v31_prereg…` §2·§4). ⑤ 승격 시 **per-cell 교정 6종의 유효성 검사**를 트리거에
  포함한다 — 서빙 featurization 이 바뀌면 전량 무효가 된 전례가 있다 (§1.5).
* **MASTER 를 쓰기 전에 통계부터**: v3.1 의 W0(cross-fit)가 **0콜로 게이트 pool 을 2배** 만든
  선례가 등록되어 있다. 자동 트리거는 이 순서를 강제해야 한다.

### 4d. KPI — 정의 · 식 · 데이터 출처

기존 K1–K5 (`sample_efficiency_kpi_20260903.md` §5)를 **능동 루프용 4개로 확장**한다.
전부 **MASTER 검증 콜만** 계수하고 surrogate 평가는 별항 (`screen_ratio ≈ 312 : 1`, 동 §2).

| KPI | 식 | 데이터 출처 | 현재값 |
|---|---|---|---|
| **A1 calls-to-frontier@ε** (= K1) | 사전 동결한 셀 기록 `R_cell` 에 대해 `min{ n : best_n ≤ R_cell + ε }`, ε = 0.005 (F_r). 미달은 `>N` + 최종 gap 병기. **사후 계산 무효** | `runs/*/labels.jsonl` 누적 궤적 + `data/store/records.parquet` | 중앙값 **57**, 최대 126 (7/8 셀) |
| **A2 OOD-셀 라벨당 오차감소** | `ΔMAE_c / n_c`. `ΔMAE_c` = 라벨 투입 전후 셀 `c` 의 홀드아웃 target MAE 차, `n_c` = 그 셀에 새로 들어간 converged 라벨 수. **cross-fit 폴드에서만 계산** | `kpi_probe_out.json` 형식의 셀별 `{target}_ALL/_BAND` MAE (현재 E1_E2/f121 `f_r_ALL` 0.0405 / `_BAND` 0.0209 등) + probe readout (`autoeng.py:1548-1556`) | **미측정** (전후 쌍이 기록되지 않음) |
| **A3 커버리지 확장률** | `(|S_after| − |S_before|) / calls`, `S` = `TrustRegion` 지지 bin 집합 (`promote_after` 회 이상 라벨된 `(feed, e-bin)`). 부수 지표로 `Δ(#cells with in_distribution=True)` | `TrustRegion.from_store` 재계산 (모델 불필요) + `scoping_mesh.py` 의 `n_store_pair_feed` 열 | **미측정.** `in_distribution` 이 `feed==121` 상수라 현재 형태로는 무의미 (`scoping_mesh.py:613`) |
| **A4 외삽영역 프론티어 갱신 수** | `#{ 라벨 : ood_flag=True 또는 in_region=False 인 후보에서 나왔고, 그 셀의 best 를 갱신 }` / 총 콜 | `waves/*/selection.json` 의 후보별 `ood_flag` + `region` + `labels.jsonl` | **구조적으로 0** — 현행 개체군에서 `ood_flag` 가 발화하지 않는다 (§1.1). **슬라이스 Z 가 이 KPI 의 첫 측정 기회다** |

**병기 필수** (K4): `prior_rows`(캠페인 시작 시 셀 행수 — 시딩 비용의 대리), `first_feasible_call`,
`n_feasible/N`, `Δ/100calls`, wall-hours. **lpopt 캠페인은 콜별 wall 을 기록하지 않는다**
(`labels.jsonl`/`status.json` 모두 결측; 동 §6-3) → **P0 의 첫 작업.**

### 4e. 가드레일 (변경 금지, 자동화가 침범해서는 안 되는 것)

1. **사전등록 우선.** 덱 헤더가 곧 prereg 이고 sha256 이 발사 게이트다 (`autoeng.py:1508-1523`).
   자동 계획도 **MASTER 콜 이전에** prereg 를 쓰고 해시를 고정해야 한다.
2. **로컬 PC 연산 금지.** 편집·Read/Grep·전송·`Get-FileHash` 만. `D:\DeCART_MASTER\BIN` 에
   실행파일이 실재하지만 쓰지 않는다 (`assembly_on_demand_tasks…` 부록 1).
3. **호스트 배치 동결.** 199 = 캠페인·produce·라이브러리 재빌드·MTC / 238 = 학습(GPU1,
   `lpopt_gpu1.inp`)·게이트·테스트 / 181 = DeCART **전용, forbidden 예외 기록 전 금지** /
   198 = production only. `_fleet_guard` 가 argv 단위로 강제.
4. **인허가 게이트는 자동화 대상이 아니다.** SDM/MTC 는 별도 CLI 단계
   (`lpopt sdm-mtc --top-k 5`, `[constraints] post_verify_top_k = 5`, arm당 ~5 콜);
   `select_topk_feasible` 은 **`|cyclen − 625|` 근접순이지 F_xy 순이 아니므로** PRIMARY 후보가
   top-k 안에 들었는지 확인이 필요하다 (`assembly_on_demand_tasks…` 작업 #21, #21b).
   pin burnup 은 model-margin haircut(78.0 vs 80.0)으로만 screen 하고 **conformal 이중계상 금지**.
5. **fail-closed 를 fail-open 으로 바꾸지 않는다.** guard 예외 = flag, `absent` = `None`(미평가)
   이지 `False` 아님. 자동화가 이 3분류를 2분류로 접으면 안 된다.

---

## 5. 단계 계획

### P0 — 지금 있는 런에 KPI 를 계측한다 (MASTER 0콜, 238 전용)

| # | 작업 | 파일 | 추정 |
|---|---|---|---|
| P0-1 | 캠페인 콜별 **wall-clock 기록** 추가 (`labels.jsonl` 에 `eval_wall_s`) | `lpopt/search/campaign.py` 라벨 기록부 | 0.3 일 |
| P0-2 | **A1/A3 계측기** — `runs/*/labels.jsonl` 누적 궤적에서 `calls-to-record@ε`·`AUF@100/300`·지지 bin 증분을 산출하는 읽기 전용 스크립트 | 신규 `kpi_frontier.py` (238 `scratch/kpi/` 의 `kpi_analyze.py` 계보 재사용) | 0.5 일 |
| P0-3 | **A2 전후 쌍 기록** — 병합 직전/직후 셀별 홀드아웃 MAE 를 `kpi_probe_out.json` 형식으로 스냅샷 | `autoeng.py::_do_probe_readout` 확장 + merge 전후 훅 | 0.5 일 |
| P0-4 | `in_distribution` 을 **지지-bin 기반**으로 교체 (현행 `feed==121` 상수) | `scoping_mesh.py:613,634` | 0.3 일 |
| P0-5 | shield 계수·`conformal_unfit_axes` 를 셀 보고서에 노출 (지금은 JSON 안에만) | `autoeng.py::_do_cell_report` | 0.2 일 |

**소계 ≈ 1.8 일.** 산출: 등록 가능한 KPI 기준선 + A4 를 측정할 준비.

### P1 — autoeng 의 에스컬레이션 정책 (승인 일시정지 포함)

| # | 작업 | 파일 | 추정 |
|---|---|---|---|
| P1-1 | `label_plan.json` 스키마 + 생성기 (§4a 입출력) | 신규 `lpopt/search/escalation.py` | 1.0 일 |
| P1-2 | `pause_for_approval` 에 **`ood_escalation`** 추가 + `_pause` 가 계획서를 첨부 | `autoeng.toml:57`, `autoeng.py:1463-1479`, `AUTOENG_CONFIG_KEYS`(`:217`) | 0.3 일 |
| P1-3 | 계획 → 덱 생성 재사용 (`build_deck`/`render_prereg` 를 produce·intervention 모드로 일반화) | `autoeng.py:800-1010` | 0.8 일 |
| P1-4 | 181 예외 정책 기록 문서 + `_fleet_guard` 우회가 **불가능함**을 테스트로 고정 | `tests/test_autoeng_fleet.py` | 0.3 일 |
| P1-5 | **첫 실화(live fire)는 슬라이스 Z** — `ood_policy = "escalate"` 를 켠 arm 을 대조와 짝지어 등록 (`phaseA2…` §5.1 의 A0/A1/A2/A3/A4 설계 그대로) | 신규 prereg | — |

**소계 ≈ 2.4 일** (+ 슬라이스 Z 실행 예산 ~210 MASTER).
**전제**: `phaseA2…` §5.5 — OOD 가 **실제로 발화하는 셀**이 없으면 arm 이 곧 control 이다.
그 셀은 신규 연료형이 들어오는 시점에만 생기므로 **P1-5 는 슬라이스 Z 와 같은 시점에 설계한다.**

### P2 — VOI 스케줄러

| # | 작업 | 파일 | 추정 |
|---|---|---|---|
| P2-1 | `Δ/100calls` 바닥 판정(< 0.002)을 **코드에 박고** 예산 이전을 제안하게 한다 | `escalation.py` + `order_targets`(`autoeng.py:1373-1433`) | 0.5 일 |
| P2-2 | 커버리지 항 — 지지 bin 승격 확률 × DB 프론티어 여유 | `mesh_pareto.csv` 소비, `escalation.py` | 0.7 일 |
| P2-3 | 세 행동의 `VOI = E[Δbest]/calls` 비교표를 계획서에 싣기 (**결정은 여전히 사람**) | `label_plan.json` | 0.5 일 |
| P2-4 | 재학습 자동 트리거 (§4c) — 조건 충족 시 `--check-only` 까지 자동, 승격은 게이트 | `autoeng.py` 스텝 추가 | 0.5 일 |

**소계 ≈ 2.2 일.** **제외**: 크기 예측 기반 자동 발사 — σ_chain 판정으로 **금지**(§1.6).

### P3 — 집합체 축을 루프에 넣는다

`assembly_on_demand_tasks_20260903.md` 의 임계경로 **≈ 9.5 일** (#1 lattice 저작 → #2 프리플라이트 →
#16 compliance → #4a/#4 screener → #5 사슬 → #10/#10b 181 러너 → #11 HGC 게이트 →
#12/#13/#13b/#13c 패키지 재생성 → #15 라이브러리 게이트 → #21/#21b MTC) + 실행 **2–2.5 근무일**.
그 위에 본 문서가 더하는 것은 **두 가지뿐**이다:

* **P3-1** — HGC/라이브러리 게이트 결과(`hgc_gates.py:97-100` 의 G-H4 회귀바:
  `|Δk| ≤ 100 pcm`, `|ΔFF| ≤ 0.0021`)를 §4a 의 계획서 스키마에 **판정 근거로 편입**.
* **P3-2** — 라운드 2 의 발사 규칙(#7/#8/#9, 2.4일)은 **σ 를 0.020 아래로 내리는 새 측정이
  선행되어야** 열린다 (`need.py:80-118` 이 이미 거부한다). 그 전까지 집합체 축의 escalation 은
  **계획서 + 물리실험**이지 자동 트리거가 아니다.

---

## 6. 이 문서가 주장하지 **않는** 것

1. OOD 게이트를 켜는 것이 더 나은 결과를 낸다는 증거는 **아직 없다** (`phaseA2…` §5).
2. 사슬이 헤드를 이긴다는 주장은 **데이터로 지지되지 않는다** (`assembly_sigma…` §5).
3. SA 대비 표본효율 비교는 **셀·목적함수·초기조건이 전부 다르다.** 성립하는 것은 *수렴 속도의
   형태*이고, 동일 셀 SA 대조군과 cold-start lpopt 대조군이 **아직 없다**
   (`sample_efficiency_kpi…` §4, §6-1·2).
4. `F_r 1.35` 는 LP 축의 목표가 아니다. LP 축 바닥은 1.4605 이고 900콜로 열리지 않았다 —
   남은 0.0144 조차 **연산자(move set) 문제**로 재분류되었다 (동 §3).

---

## 7. P0 착수 스탬프 (append-only, 2026-09-03)

**본 절은 추가 전용이다.** §0–§6 은 수정하지 않았다. 아래는 §5 의 P0 (계측, MASTER 0콜,
238 전용)로 실제 착지한 것과 그 위치다. 전부 읽기 전용 계측이며 기존 경로의 기본 동작은
바이트 동일하다 — 새 아티팩트는 *추가*되고, 기존 컬럼·필드는 하나도 바뀌지 않았다.

### P0-1 — 콜별 wall-clock + MASTER/surrogate 계수 (§4d "P0 의 첫 작업")

| 항목 | 위치 |
|---|---|
| `_wall_s(outcome)` — `WaveOutcome.wall_s`(`lpopt/search/verify.py:1129,1144,1153`, 이미 존재)를 라벨용 스칼라로. **미측정은 `0.0` 이 아니라 `None`** (§4e 규칙 5) | `lpopt/search/campaign.py:269` |
| surrogate 평가 카운터 초기화 (guided / user_criteria) | `campaign.py:1089`, `campaign.py:3688` |
| 웨이브당 surrogate 평가 누적 (local search 이후 풀 크기) | `campaign.py:2498`, `campaign.py:4479` |
| `labels.jsonl` 행에 `wall_s` · `cumulative_master_calls` · `cumulative_surrogate_evals` (guided) | `campaign.py:2628-2630` |
| 동 3필드 (user_criteria lean/active) | `campaign.py:3915-3917` |
| `events.jsonl` wave 행에 동 3필드 (`wall_s` = 그 웨이브 콜별 wall 의 **합** = 비용이지 경과시간 아님) | `campaign.py:3166-3168`, `_log_event` 시그니처 `campaign.py:3155` |

`verify.py` 는 **변경하지 않았다** — 콜별 wall 은 `_eval_entry` 가 이미 측정해
`WaveOutcome.wall_s` 로 반환하고 있었고, 결측은 기록부(캠페인) 쪽이었다. 근본원인 최소 diff.

### P0-2 — KPI 계측기 (A1 / K2 / K3 / K4)

신규 `lpopt/tools/kpi_calls_to_frontier.py` — `sample_efficiency_kpi_20260903.md` §5 의
K1–K4 정의를 그대로 구현. MASTER 0콜, 체크포인트 로드 0, 런 디렉터리 밖 기록 0.

| KPI | 함수 | 비고 |
|---|---|---|
| A1 `calls-to-frontier@ε` (=K1) | `compute_kpi` `:426` / `trajectory` `:251` | `R_cell` 은 **발사 시점 스토어에서 동결**(`freeze_baseline` `:154` → `kpi_baseline.json`). 동결본이 없으면 계산은 하되 `frozen=false` · **`valid=false`** 로 낙인 (§5 K1 "사후 계산 무효") |
| K2 `calls-to-incumbent` | `compute_kpi` `:426` | 미달 시 `verdict = "no new information"` |
| K3 `AUF@N` | `auf` `:307` | 런이 N 에 못 미치면 **보고하지 않는다** (셀 간 비교는 동일 N 에서만 성립) |
| K4 병기 | `compute_kpi` `:426` | `screen_ratio`, `prior_rows`, `first_feasible_call`, `n_feasible/N`, `Δ/100calls`(`delta_per_100calls` `:330`) + `< 0.002` **floor 판정**, wall-hours |
| A4 | `a4_ood_frontier_updates` `:370` | `selection.json` 의 후보별 `ood_flag` × 기록 갱신. `ood_flag=None` 은 **`unknown`** 이지 clean 아님 (§4e 규칙 5) |

산출: `<run_dir>/kpi.json` (`write_kpi` `:541`).
CLI: `lpopt kpi --run <dir> [--store] [--metric] [--epsilon] [--feasible-only] [--post] [--freeze]`
(`lpopt/cli.py:680` `cmd_kpi`, 파서 `lpopt/cli.py:1985`).

### P0-3 — A2 전후 스냅샷

* **발사 시**: `CampaignDriver._kpi_launch_snapshot` (`campaign.py:1945`, 호출
  `campaign.py:2236` — `_write_status("running")` 직후)가 `kpi_baseline.json` +
  `ood_snapshot_pre.json` 을 쓴다. pre = 그 셀의 **기존 라벨 행**에 대한 발사 챔피언의
  target별 MAE(`kpi_probe_out.json` 의 `<target>_ALL` 레이아웃) + A3 coverage 플래그.
  최대 256행으로 상한, 전부 try/except — 계측이 캠페인 예산을 먹을 수 없다.
  **덮어쓰지 않는다**: `--resume` 이 기록을 재동결하면 A1 이 무효가 되기 때문.
* **수확 시**: `_kpi_harvest` (`campaign.py:1931`, 호출 `campaign.py:3493` —
  status.json/report.md 보다 **앞**)가 `ood_snapshot_post.json` + `kpi.json` 을 쓴다.
  post 는 `waves/*/selection.json` 의 `pred_mean` 과 `labels.jsonl` 의 검증값을 맞대어
  계산하므로 **모델도 GPU 도 필요 없다**. pre 행(학습에 들어간 행)에 대해 post 행은
  발사 챔피언 기준 **진짜 홀드아웃**이고, 그래서 `ΔMAE` 와 `ΔMAE / n_new`
  (`error_reduction_per_label`)가 비로소 측정 가능해진다.
* CLI 재계산: `lpopt kpi --run <dir> --post` (`snapshot_post` `:671`).

### P0-4 — `in_distribution` 을 지지-bin 기반으로 (플래그, 기본 OFF)

신규 `lpopt/search/coverage.py` — bin 산술은 `acquisition._e_bin` 을 **재사용**해
trust region 과 커버리지 정의가 갈라질 수 없게 했다.

* `support_bins` `:62` — `S` = `promote_after`(기본 16) 이상 라벨된 `(feed, e-bin)`.
* `in_distribution` `:76` — e_core 미상은 feed 만으로 admit (`TrustRegion.in_region` 과 동일 관용).
* `scoping_mesh.py`: `--coverage-in-distribution` (+ `--coverage-e-core-band`,
  `--coverage-promote-after`) `:507-518`, 헬퍼 `_in_dist` `:627`, 적용 `:649`(pareto front)
  · `:671`(node). **플래그 없으면 `bool(feed == 121)` 그대로** — `mesh_vs_db.py` /
  `autoeng.py` / `scoping_mesh_fig.py` 의 `mesh_nodes.csv` 소비는 바이트 동일.

### P0-5 — 셀 보고서가 A1–A4 를 출력

* `report.md`: `lpopt/report/report.py:389` `_kpi_section` (삽입 `:715`, Figures 절 직전).
  `kpi.json` 이 **있을 때만** 절이 나타난다 — 미계측 런의 보고서는 P0 이전과 동일.
* `status.json`: `campaign._kpi_status_extra` (`campaign.py:288`) 를 splat 으로 병합
  (`campaign.py:2028` guided, `campaign.py:4838` user_criteria). `kpi.json` 이 없으면
  `{}` 이므로 **진행 중 status.json 은 변경 없음**.
* 렌더러: `kpi_markdown` (`kpi_calls_to_frontier.py:786`), 요약 블록
  `kpi_status_block` (`:843`). 사후 A1 은 표에 **"post-hoc, INVALID as A1"** 로 찍힌다.

### 테스트 (238, `../venv/bin/python -m pytest`)

| 파일 | 개수 | 덮는 것 |
|---|---|---|
| `tests/test_kpi_frontier.py` | 28 | 합성 런디렉터리 + 합성 스토어로 A1/K2/K3/K4/A4, 사후 A1 무효 낙인, A2 pre/post 와 `ΔMAE/label`, 지지-bin coverage, mesh 플래그 기본 OFF, report/status 노출, trajectory 불변식 |
| `tests/test_campaign_kpi_accounting.py` | 8 | StubEvaluator 로 16콜 캠페인 실주행 — 라벨 3필드, 콜 인덱스 단조, surrogate ≫ MASTER, `wall_s` 실측, events 행 일치·합 일치, 발사 동결 + pre, 수확 kpi.json + post, status/report 의 A1–A4 |

### P0 에서 **하지 않은 것** (P1 이후)

* §5 P0-5 의 나머지 절반 — shield 계수 · `conformal_unfit_axes` 를 **autoeng 셀 보고서**
  (`autoeng.py::_do_cell_report`)에 노출하는 것은 착수하지 않았다. 본 P0 는 캠페인
  `report.md`/`status.json` 까지이고, autoeng 쪽 보고서는 `label_plan.json` 스키마(P1-1)와
  같은 파일을 건드리므로 P1 과 함께 한 번에 하는 편이 diff 가 작다.
* A2 의 **cross-fit 폴드** 판(§4d 원문 "cross-fit 폴드에서만 계산"). 지금 구현된 pre/post 는
  *발사 챔피언 기준 홀드아웃*이며, 재학습 후 폴드 MAE 차는 P2-4(자동 재학습 트리거)에서
  `al_retrain` 산출물과 짝지어야 정직하게 나온다.
* A4 는 여전히 **구조적으로 0** 일 수 있다 — 현행 개체군에서 `ood_flag` 가 발화하지 않기
  때문(§1.1). 계측기는 준비되었고, 첫 실측 기회는 §5 P1-5 의 슬라이스 Z 다.

### 앵커 주의

위 `file:line` 은 **2026-09-03 P0 착지 시점** 기준이다. `campaign.py` 는 본 작업과
동시에 다른 작업(E.7-(a) `served_checkpoint` / `SCORED_DIGEST_KEY`)이 편집 중이었고,
그 변경분은 위 줄번호에 이미 반영되어 있다. 이후 편집으로 다시 밀릴 수 있으므로
검색 기준은 줄번호가 아니라 심볼명(`_wall_s`, `_kpi_status_extra`,
`_kpi_launch_snapshot`, `_kpi_harvest`, `surrogate_evals`,
`cumulative_master_calls`)이다.
