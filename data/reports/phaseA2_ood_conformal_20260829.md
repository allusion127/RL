# Phase A-2 — OOD·conformal을 report-only에서 탐색·인도 동작으로 연결

- 작성일: 2026-08-29
- 대상: `lpopt` (APR1400 equilibrium-cycle loading-pattern optimizer)
- 근거: 외부 검토서 `RL_core_loading_engineer_AI_review_2026-08-29.md` §6.5 (P0-04), §8.5 (uncertainty & safety shield), §9 Phase A 작업 6
- 성격: 구현 보고서. **기본 동작(default)은 변하지 않는다.**

---

## 0. 한 문단 요약

검토서 §6.5의 지적은 "OOD guard는 warning surface이고 conformal interval은 report-only이므로,
둘 다 계산되고 출력된 뒤 **아무 것도 바꾸지 않는다**"였다. 이번 변경은 그 두 신호를
(1) acquisition의 **랭킹/풀**과 (2) 인도(delivery) dossier의 **필드**로 연결한다.
`ood_guard.py` / `conformal.py` 자체는 **한 줄도 바꾸지 않았다** — 두 모듈은 그대로 소비된다.
새 의존성은 없다. 기본값(`ood_policy = "warn"`, `conformal_gate = false`)에서
shield는 아예 실행되지 않으므로 wave 구성은 이전과 byte-identical이다.

---

## 1. 무엇이 바뀌었나

### 1.1 `[acquisition]` 새 키 3개 — `lpopt/config.py`

| 키 | 기본값 | 의미 |
|---|---|---|
| `ood_policy` | `"warn"` | `"warn"` = 현재 동작(경고만). `"escalate"` = OOD 후보의 exploit 점수를 `-inf`로 강등(explore/control tier로만 남김). `"reject"` = 풀에서 제거. |
| `conformal_gate` | `false` | `true`면 gated licensing 축에 split-conformal **상한**을 hard chance constraint `U_c(x) ≤ L_c`로 적용. |
| `conformal_alpha` | `0.10` | interval을 읽을 miscoverage 수준. `conformal.DEFAULT_ALPHAS`(0.10 / 0.32) 중 하나여야 한다. |

- `ood_policy` 검증: `_validate_ood_policy` (`_VALID_OOD_POLICIES = {"warn","escalate","reject"}`).
- `conformal_alpha` 검증: `_validate_conformal_alpha` → `_valid_conformal_alphas()`가
  `lpopt.model.conformal.DEFAULT_ALPHAS`를 **직접 import** 한다. 하드코딩된 사본을 두지 않았으므로
  fitter가 레벨을 늘리면 deck 검증도 자동으로 따라온다 (`tests/test_safety_shield.py`가 두 집합의 일치를 assert).
- fit되지 않은 레벨(예: 0.05)을 요구하면 `conformal_quantile`이 `+inf`(공허한 구간)를 돌려주므로,
  조용히 통과시키는 대신 **deck 로드 시점에 거절**한다.

### 1.2 Safety shield — `lpopt/search/acquisition.py`

새 공개 표면 (기존 함수는 어느 것도 시그니처가 바뀌지 않았다):

- `SafetyShield` (frozen dataclass) — `ood_policy` / `conformal_gate` / `conformal_alpha`.
  `SafetyShield.from_config(cfg.acquisition)`, `.active`(기본값에서 `False`).
- `ood_flags(model, patterns)` → `(flags, guard_state, n_errors)`.
  `PosValCnnBackend.feature_ood_types`(→ `ood_guard.feature_ood_vecs`)를 **그대로** 호출한다.
- `conformal_upper(model, ctx, patterns, alpha=…)` → `[N,7]` 상한 또는 `None`.
  `PosValCnnBackend.predict_interval`(→ `conformal.interval_arrays`)을 그대로 호출한다.
- `fxy_conformal_upper(...)` — F_xy 전용 경로 (§1.4).
- `conformal_gate_axes(limits)` — 해당 모드가 **실제로 gate하는** 축만 반환.
- `apply_safety_shield(model, ctx, scored, shield, limits)` → `(ScoredPool, report)`.
- `CONFORMAL_GATE_AXES`, `OOD_POLICIES`, `OOD_POLICY_{WARN,ESCALATE,REJECT}`.

`ScoredPool`에 두 필드 추가(기본값 있음, 직접 생성하는 기존 호출자 무영향):
`ood_flag: np.ndarray`(bool), `conformal_unfit: tuple[tuple[str, ...], ...]`.

**적용 순서는 의도적이다**: OOD 정책이 먼저, conformal gate가 나중.
manifold를 벗어난 후보의 conformal interval은 **그 manifold 위에서** 교정된 것이므로
그 후보에 대한 bound가 아니다. 따라서 OOD로 걸린 후보를 conformal이 "통과"시키는 순서는 성립할 수 없다.

#### OOD 정책이 exploit 점수에 하는 일

`escalate`는 flagged 후보의 `exploit`과 `rank`를 `-inf`로 만든다. 이것이 두 곳에 동시에 걸린다:

1. `compose_wave`의 exploit 슬롯 정렬 키가 `rank`이므로 exploit tier에서 밀려난다.
2. `CampaignDriver._run_wave`의 elite carry-over가 `isfinite(exploit)`로 거르므로
   다음 wave의 elite seed로도 올라가지 못한다.

`p_feas` / `raw_epi`는 **건드리지 않는다**. explore 슬롯은 `raw_epi`로만 정렬하므로
OOD 후보는 "탐험 대상"으로는 남는다 — 검토서의 규칙은 "off-manifold에서 surrogate 단독 exploit 금지"이지
"보지도 말라"가 아니다 (§6.5 개선 2).

#### 상태 3분류 (검토서 §8.5: `unknown`은 안전한 값이 아니다)

| guard 상태 | 의미 | 동작 |
|---|---|---|
| `absent` | 백엔드에 `feature_ood_types`가 없다 | 아무 것도 flag하지 않고, report에 `absent`로 남긴다. dossier의 `ood_flag`는 `None`(= 미평가)이지 `False`가 아니다. |
| `available` + 정상 | guard가 판정했다 | bool |
| `available` + 예외 | guard가 실행되지 못했다 | **fail-closed**: 해당 후보를 flagged로 처리하고 `ood_guard_errors`에 계수 |

### 1.3 Conformal chance constraint

`CONFORMAL_GATE_AXES = (("f_r",0), ("cbc_max",1), ("f_q",2), ("ao_abs",4))` —
`feasibility_limits_for`에 hard limit이 있고 `conformal.CONFORMAL_TARGETS`에 fit 항목이 있는 **교집합**이다.

- `max_pin_burnup`은 conformal target이 있지만 **일부러 제외**했다. 탐색은 이 축을
  model-margin haircut(78.0 vs licensing 80.0)으로 이미 screen하므로 conformal half-width를
  얹으면 같은 margin을 이중 계상한다.
- `cyclen`은 양쪽 edge를 갖는 band이고 이 gate는 단측 상한이므로 제외.
- 해당 모드가 ungate하는 축(`flat_power`/`fr_boundary`의 `f_r` = `None`)은 **gate하지 않는다**.
  없는 제약을 conformal이 새로 만들어내면 안 된다.

**제거 규칙**: gated 축 중 하나라도 유한한 conformal 상한이 limit을 넘으면 풀에서 제거하고
축별로 계수한다(`conformal_rejected_by_axis`).
유한한 상한이 **없는** 축은 제거 사유가 되지 않는다 — 기존 `μ + κ·σ` screen이 그대로 서 있고,
그 축은 해당 후보의 `conformal_unfit`에 기록된다. 즉 gate는 screen을 **대체**하되,
대체할 수 없는 곳에서는 조용히 사라지지 않고 "여기는 교정되지 않았다"고 말한다.

### 1.4 F_xy — head + fit이 둘 다 있을 때만

F_xy는 7-컬럼 surrogate 계약 **밖**에서 서빙되므로 `predict_interval`에 아예 나오지 않는다.
`fxy_conformal_upper`는 다음 두 조건이 **모두** 참일 때만 bound를 만든다:

1. `has_fxy_head(model, ctx)` — 실제 checkpoint에 f_xy head가 있는가.
   (`hasattr`이 아니라 빈 리스트로 **호출**해서 판정한다. 실제 백엔드는 메서드를 항상 정의하고
   head가 없으면 `None`을 돌려준다.)
2. 챔피언의 `conformal.json`에 `per_target["f_xy"]` 항목이 있는가.

둘 중 하나라도 없으면 `None`을 돌려주고 **해당 모드의 proxy σ screen을 그대로 둔다**
(`FXY_PROXY_*`, `μ + κ·σ`). 이때 축은 `conformal_unfit`에 넣지 **않는다** — screen이 없는 것이 아니라
proxy screen인 것이고, 그 사실은 wave report의 `conformal_fxy: "proxy"`로 남는다.
interim proxy는 예측 F_r의 affine 함수이고 그 분산은 fit residual이지 교정된 model error가 아니므로,
head용으로 fit된 conformal 분위수가 proxy를 인증해 주지 않는다.

### 1.5 캠페인 배선 — `lpopt/search/campaign.py`

- `CampaignDriver.safety_shield = acq.SafetyShield.from_config(self.acq)`.
- `_run_wave` 3b 단계: local search **후**, elite carry-over와 `compose_wave` **전**에
  `apply_safety_shield`를 적용한다(`shield.active`일 때만). 이 위치여야 escalate가 두 소비자에
  동시에 걸리고, reject가 두 곳 모두에서 사라진다.
- `WaveReport`에 `ood_flagged` / `ood_escalated` / `ood_rejected` / `conformal_rejected` 추가
  (모두 기본 0 → 구 `state.json`도 그대로 로드된다).
- `waves/wave_NN/selection.json`:
  - 후보별 `ood_flag` (모든 모드에서 기록; guard가 없으면 `None`).
  - shield가 실제로 돈 wave에만 `shield` 블록(정책, guard 상태, 축별 제거 계수, `n_candidates → n_remaining`).
- `_selected_safety(patterns)` — **정책과 무관하게** 선택된 후보에 대해 guard와 interval을 평가한다.
  탐색이 무엇을 하든 dossier는 상태를 말해야 하기 때문이다. wave당 최대 `wave_size`개이므로 비용은 무시할 수 있다.

### 1.6 인도 dossier — boolean은 그대로

`is_deliverable` / `unknown_axes`의 **술어는 바뀌지 않았다.**
추가된 것은 두 개의 보고 필드뿐이다:

- `ood_flag`: `True` / `False` / `None`(미평가 — resume으로 이어받은 행)
- `conformal_unfit_axes`: conformal 교정이 덮지 못하는 gated 축 목록 (`None` = 미평가)

들어가는 곳:

- `CampaignDriver._best_dict(...)` → `best` / `best_overall` (모든 모드)
- `_maybe_update_best`의 target_cycle 계열 인라인 dict (동일 두 필드)
- `_write_delivery()` → `delivery.json`의 `ranked` / `excluded` **모든 항목**

`conformal.json`이 없는 챔피언에서는 `conformal_unfit_axes`가 gated 축 전체를 나열한다.
이것은 결함이 아니라 정직한 보고다 — 검토서 §8.5의 "calibration cell 미지원"이 바로 이 상태이고,
`unknown`은 안전한 값이 아니라 추가 계산이 필요한 상태다.

### 1.7 부수 변경 1건 — `lpopt/model/model_api.py`

`PosValCnnBackend.conformal_cell_keys(patterns, case, cell)` 공개 접근자 추가.
7-컬럼 계약 밖의 target(f_xy)에 대해 fit과 **동일한** `(feed, e_core-bin)` 키가 필요한데,
private `_conformal_cell_keys`를 외부에서 찌르지 않기 위한 얇은 래퍼다. 기존 동작 변화 없음.

---

## 2. 기본값은 바뀌지 않았다 (무엇을 검증했나)

- `AcquisitionConfig()` → `ood_policy == "warn"`, `conformal_gate is False`, `conformal_alpha == 0.10`.
- `SafetyShield.from_config(AcquisitionConfig()).active is False` → `_run_wave`가 shield를 **호출조차 하지 않는다**.
- 기본 deck의 `selection.json`은 후보별 `ood_flag` 한 필드만 늘어나고 `shield` 블록은 없다.
- `WaveReport`의 네 계수는 모두 0.
- 기존 스위트(`test_acquisition` / `test_flatness_campaign` / `test_campaign_stub` / `test_delivery`
  / `test_config` / `test_ood_guard` / `test_conformal` / `test_elite_objective`) 무변경 통과.

---

## 3. 어떻게 켜나

```toml
[acquisition]
# 1) OOD 후보를 exploit tier에서 강등 (풀에는 남김)
ood_policy = "escalate"

# 2) 또는 완전 배제 (production 모드 권고 — 검토서 §6.5 개선 2)
# ood_policy = "reject"

# 3) conformal 상한을 hard chance constraint로
conformal_gate = true
conformal_alpha = 0.10          # 90% interval; 0.32 = 68%
```

전제 조건:

- `ood_policy != "warn"`은 백엔드가 `feature_ood_types`를 노출해야 실효가 있다
  (`PosValCnnBackend`는 노출한다. 노출하지 않는 백엔드는 wave report에 `ood_guard: "absent"`로 남는다).
- `conformal_gate = true`는 챔피언 디렉터리에 `conformal.json`이 있어야 실효가 있다
  (`python -m lpopt.model.conformal` 경로의 `fit_conformal`).
  없으면 아무 후보도 제거되지 않고 모든 gated 축이 `conformal_unfit`에 기록된다 — **fail-open이지만 침묵하지 않는다**.

읽는 곳:

- 진행 중: `[optimize][shield]` 로그 라인 (제거/강등이 있을 때만)
- wave별: `waves/wave_NN/selection.json`의 `shield` 블록
- 캠페인 요약: `state.json` / `CampaignResult.wave_reports`의 네 계수
- 인도: `delivery.json`의 항목별 `ood_flag` / `conformal_unfit_axes`, `best`의 같은 두 필드

---

## 4. 아직 하지 않은 것 (명시적 범위 밖)

1. **σ를 바꾸지 않는다.** conformal은 축의 *screen*을 대체하는 것이고,
   exploit 스칼라 내부의 `μ + κ·σ` UCB 항 자체는 그대로다. 즉 gate는 순위를 재정의하지 않고
   feasible 집합을 좁힌다. objective 스칼라까지 conformal화하는 것은 별도 사전등록이 필요하다.
2. **cell fallback risk penalty 없음** (검토서 §6.5 개선 4). `interval_arrays`가 돌려주는
   `from_cell` 플래그는 아직 소비하지 않는다 — 현재는 "유한한 bound가 있는가/없는가"의 2분류만 쓴다.
   global fallback을 per-cell fit과 구분해 penalty를 주는 것은 다음 단계다.
3. **`is_deliverable`의 boolean은 그대로다.** 과제 지시가 명시적으로 술어 변경을 금지했고,
   그 편이 옳다: OOD flag를 인도 술어에 접어 넣으면 "왜 떨어졌는가"가 측정 결측과 구별되지 않는다.
4. **pin burnup 축은 conformal gate에 넣지 않았다** (§1.3의 이중 계상 사유).

---

## 5. 전향적(prospective) 검증이 측정해야 할 것

이 변경은 **행동을 바꾸는 스위치**를 추가했을 뿐, 그 스위치를 켜는 것이 더 나은
결과를 낸다는 증거는 아직 없다. 검토서 §6.5의 주장(OOD에서 서로게이트 단독 exploit 금지,
conformal 상한을 hard constraint로)이 이 코드베이스에서 참인지는 다음으로만 확인된다.

### 5.1 Arm 설계 (paired, 같은 seed / 같은 셀 / 같은 budget)

| arm | `ood_policy` | `conformal_gate` |
|---|---|---|
| A0 (control) | `warn` | `false` |
| A1 | `escalate` | `false` |
| A2 | `reject` | `false` |
| A3 | `warn` | `true` |
| A4 | `escalate` | `true` |

동일 `[case]`·동일 `budget`·동일 rng seed. arm당 최소 2개 셀(하나는 T6_T4/f121 계열 champion 셀,
하나는 OOD가 실제로 발화하는 셀 — 신규 연료형 또는 geometry variant가 들어간 셀).

### 5.2 1차 지표 (primary)

1. **예측-측정 불일치의 상한 위반율**: MASTER 측정치가 `μ + κ·σ`를 벗어난 wave 후보 비율,
   그리고 conformal 상한을 벗어난 비율. A3/A4에서 후자가 명목 `1 - α`(0.90) 이상으로
   실제 커버되는지 — 이것이 conformal gate의 존재 이유다.
2. **false-feasible 유입률**: acquisition이 feasible로 판정했으나 측정에서 hard limit을
   위반한 core의 wave당 개수. A2/A4가 A0보다 낮아야 gate가 값을 한다.

### 5.3 2차 지표 (secondary)

3. **예산 효율**: feasible label 1건당 MASTER 호출 수. reject는 풀을 좁히므로
   여기서 **악화될 수 있다** — 그 악화폭이 1차 지표 개선의 대가로 받아들일 만한지가 실제 판단이다.
4. **최적점 손실**: 같은 budget에서 도달한 최적 objective(모드별 F_xy / F_r / node_peak)의 차이.
   gate가 진짜 해를 잘라내면 여기서 드러난다.
5. **풀 붕괴 빈도**: `n_remaining == 0`인 wave 수. 0이 아니면 gate가 너무 공격적이거나
   OOD envelope가 현재 개체군에 대해 과도하게 좁다는 뜻이다.
6. **강등 vs 배제의 차이**: A1과 A2의 diversity(선택된 pattern들의 pairwise Hamming 분포).
   escalate가 explore 슬롯을 통해 diversity를 보존한다는 설계 주장의 직접 검증.

### 5.4 사전 정의된 판정

- conformal 실측 커버리지가 명목 미만이면(§5.2-1) **gate를 켜지 않는다**. 상한이 상한이 아니기 때문이다.
  이 경우 필요한 것은 gate가 아니라 `fit_conformal`의 재적합/재검증이다.
- OOD arm이 1차 지표를 개선하지 못하면서 §5.3-3만 악화시키면, 기본값은 `warn`으로 남기고
  production 모드(신규 연료형 / geometry variant 도입 시)에서만 `reject`를 쓴다.
- 어느 arm도 §5.2를 개선하지 못하면, 문제는 정책이 아니라 **envelope의 margin**(`DEFAULT_MARGIN = 0.5`)
  또는 conformal cell 해상도(`DEFAULT_BIN_WIDTH = 0.25`, `DEFAULT_MIN_CELL = 20`)이며,
  다음 실험은 그 두 상수를 대상으로 해야 한다.

### 5.5 이 실험이 성립하려면 먼저 필요한 것

- `f_xy` conformal fit (§1.4의 조건 2). 현재 `CONFORMAL_TARGETS`에 `f_xy`가 없으므로
  F_xy 축은 어떤 arm에서도 proxy screen으로만 돈다. F_xy가 목적함수인 모드에서
  conformal gate의 값을 측정하려면 이것이 선행되어야 한다.
- OOD가 **실제로 발화하는** 셀. 현재 학습 개체군은 geometry 채널에서 상수이므로,
  기존 ga80/paramA 셀만으로 A1/A2를 돌리면 flag가 0이고 arm이 곧 control이 된다.
  즉 이 실험은 신규 연료형·geometry variant 도입과 **같은 시점에** 설계되어야 한다.

---

## 6. 변경 파일

| 파일 | 변경 |
|---|---|
| `lpopt/config.py` | `[acquisition]` 3키 + validator 2개 (`_validate_ood_policy`, `_validate_conformal_alpha`, `_valid_conformal_alphas`) |
| `lpopt/search/acquisition.py` | `SafetyShield`, `ood_flags`, `conformal_upper`, `fxy_conformal_upper`, `conformal_gate_axes`, `_subset_pool`, `apply_safety_shield`; `ScoredPool`에 `ood_flag` / `conformal_unfit` |
| `lpopt/search/campaign.py` | shield 배선(3b 단계), `WaveReport` 4필드, `selection.json`의 `ood_flag` + `shield`, `_selected_safety` / `_row_safety_fields`, `_best_dict` / `_write_delivery` dossier 필드 |
| `lpopt/model/model_api.py` | `conformal_cell_keys` 공개 접근자 |
| `tests/test_safety_shield.py` | 신규 (29건) |

`lpopt/model/ood_guard.py`, `lpopt/model/conformal.py` — **무변경**.
