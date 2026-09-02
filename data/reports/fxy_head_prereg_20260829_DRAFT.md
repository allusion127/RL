> **SUPERSEDED (2026-08-29).** 본 문서는 `data/reports/fxy_head_prereg_20260829.md` (FINAL, 라벨 6,236행 기준)로 대체되었다 — 아래의 N·임계값(0.029 포함)은 폐기되었으며 인용하지 말 것.

# 사전등록(pre-registration) — `f_xy` prior-residual head (F_xy 전환 P4) — **DRAFT**

- 작성일: 2026-08-29
- 상태: **DRAFT.** 라벨 backfill이 진행 중이며(로컬 ~840 + HOST_199/2_LP 추가 예정),
  **학습 실행 전에 §2의 N과 §4의 임계값을 확정하고 DRAFT 꼬리표를 제거해야 한다.**
  본 문서를 확정하기 전에 학습을 돌리면 그 결과는 사전등록된 것이 아니다.
- 근거 설계서: `data/reports/fxy_switch_design_20260829.md` §3.4 (구속력)
- 대상 코드: `lpopt/model/**` (P4에서 구현 완료), 챔피언 `data/models/s1i`

---

## 1. 무엇을 주장하려는가 (가설)

**H1 (1차).** `f_xy` prior-residual head는 라벨된 holdout에서 현행 대리변수보다
**cell 내 순위를 더 잘 맞춘다.**

**H2 (2차).** head의 절대 오차가 prior 단독 오차보다 작다 —
즉 head가 `F_xy = a·F_r + b`가 설명하지 못하는 성분을 실제로 학습했다.

**H3 (무해성, veto).** head를 붙이는 것이 기존 7개 target의 성능을 떨어뜨리지 않는다.

H3은 구조적으로 상당 부분 보장되어 있다(§5) — 그럼에도 **측정으로 확인**한다.
"구조상 안전하므로 측정하지 않았다"는 본 프로젝트에서 허용되지 않는 논증이다.

---

## 2. 데이터와 분할 (실행 전 확정할 것)

| 항목 | 값 | 상태 |
|---|---|---|
| store 전체 라벨 행 수 | **1,361** (2026-08-29 스냅샷, 전량 converged) | 진행 중 (증가 예정) |
| train fold 라벨 행 수 `N_train` | **TBD** (≥ `MIN_FXY_LABELS` = 200, 아니면 학습이 `ValueError`로 거부) | 분할 후 확정 |
| holdout 라벨 행 수 `N_hold` | **TBD** | 분할 후 확정 |
| 라벨된 cell 수 | **TBD** (현재 알려진 소스는 7 cell; 최대 물량은 `T6_T4/f121`에 690행 집중) | 분할 후 확정 |
| split | 챔피언과 동일 `S1i` | 확정 |
| holdout 정의 | `lpopt gate-promote`가 쓰는 **curriculum-val per-cell holdout** (`val_by_cell`), 양쪽 챔피언 모두 학습에서 제외된 행 | 확정 |

> **누수 규칙.** prior 계수 `(a, b)`는 `train.resolve_fxy_prior`가 **train fold 행만으로**
> 적합한다(`physics_prior.fit_fxy_prior`, cyclen/power prior와 동일 계약).
> holdout 행은 prior 적합에도, z-score 상수에도 들어가지 않는다.
>
> **주의(확정 필요).** 라벨의 대다수가 단일 cell(`T6_T4/f121`)에 몰려 있으면
> "cell 내 ρ"의 표본이 사실상 1 cell이 된다. §4의 게이트를 판정하기 전에
> **cell별 라벨 분포표를 먼저 출력**하고, 라벨된 cell이 2개 미만이면
> 1차 게이트는 **판정 불가(inconclusive)** 로 기록한다 — 통과로 읽지 않는다.

---

## 3. 비교 대상 (arm)

| 이름 | 정의 |
|---|---|
| `HEAD` | 본 실행이 산출하는 f_xy head (prior + residual), `predict_fxy`로 서빙 |
| `PRIOR` | 같은 실행의 prior 단독 (`a·F_r_pred + b`, residual ≡ 0). head의 출발점이자 H2의 baseline |
| `PROXY` | 현행 대리변수. **`F_r` 예측**을 F_xy 순위의 대리로 쓴 것 (설계서 §3.6의 "할 수 있는 것") |
| `INCUMBENT` | 챔피언 `data/models/s1i` — 7개 legacy target의 무회귀 비교 기준 |

`node_peak`은 대리 후보에서 **제외**한다. 실측 상관 0.735~0.854, residual sd
0.057~0.064로 게이트 폭과 같은 오더이며(설계서 §1.2/§3.6), 이를 대리로 쓴 주장은
설계서가 명시적으로 금지한다.

---

## 4. 게이트 (실행 전 확정 — 통과/실패 규칙)

세 게이트를 **모두** 통과해야 head를 서빙 경로에 승격한다.

### G1 (1차) — cell 내 순위

> 라벨된 holdout에서 `HEAD`의 **within-cell Spearman ρ(f_xy)** 가 `PROXY`의 ρ 이상.

- 산출: `lpopt.model.evaluate`의 per-target 표 (`within_case_spearman`,
  `within_case_spearman_sd`, `n_cases`) — f_xy는 checkpoint의 `target_names`를 따라
  자동으로 표에 들어온다.
- 판정: `ρ_HEAD - ρ_PROXY >= 0`.
- **동률/근소차 처리:** 점 추정만으로 판정하지 않는다. 차이가 `0 ± 0.05` 안이면
  `lpopt.model.ab_paired`의 **cell-clustered paired BCa CI**를 산출하고, CI가 0을
  포함하면 **동률(tie)** 로 기록한다 — 승리로 읽지 않는다.
  (근거: `ab_decide` 문서화된 교훈 — 점 비교만으로 한 승격은 "지표 옷을 입은 tiebreak"였다.)

### G2 (2차) — 절대 오차

> 라벨된 holdout에서 `HEAD`의 **MAE(f_xy) < 0.029** (= 실측 prior residual sd, pooled n=192).

- 산출: 같은 per-target 표의 `mae`, 그리고 학습 중 로그의 `mae_f_xy`
  (`train.fxy_metrics`, 라벨된 부분집합만 채점).
- 0.029는 **prior 단독의 성능**이므로, 이 게이트는 곧 "head가 prior보다 낫다"(H2)와 같다.
  `MAE_HEAD >= MAE_PRIOR`이면 head는 아무것도 배우지 못한 것이며 승격하지 않는다.
- **경고(사전 기록).** 0.029는 **HOST_199 2개 cell(n=192)** 에서 나온 수치다.
  `regen T6_T4`(n=690)에서는 0.0320이었다. holdout이 T6_T4 위주면 0.029는
  가혹한 기준이 된다 — 그 경우에도 **기준을 사후에 완화하지 않는다.**
  대신 두 수치를 모두 보고하고 판정은 0.029로 한다.
- **실측(2026-08-29, backfill 진행 중 스냅샷).** store 74,717행 중 `f_xy` 보유
  **1,361행**(전량 converged). 전체에 대한 prior 적합은
  `f_xy = 1.2220·f_r − 0.2650`, r = 0.9914, **residual sd = 0.0404**.
  즉 corpus 전체 기준의 prior 성능은 0.029가 아니라 **0.040**이다.
  → **실행 전 결정 사항(TBD):** G2의 임계값을 (i) 설계서의 0.029로 둘지,
  (ii) 실제 train fold에서 적합된 prior의 residual sd로 둘지 확정한다.
  어느 쪽이든 **학습 전에** 확정하고, 학습 후에 바꾸지 않는다.
  (train fold 한정 값은 학습 로그의 `=== f_xy head: prior ... resid sd=... ===`
  줄과 checkpoint meta의 `fxy_head.prior.resid_sd`에 기록된다.)

### G3 (veto) — legacy 7 target 무회귀

> `lpopt gate-promote` (= `curriculum.gate_no_regression` + `gate_legacy_tail`)가
> `INCUMBENT` 대비 **PASS**.

- 채점 family `NOREG_TARGETS` = `cyclen`, `f_r`, `node_peak`, `map_cov`.
  강제 family는 기본 `NOREG_ENFORCED_DEFAULT` = `cyclen`, `node_peak`, `map_cov`
  (`f_r`은 기본 report-only). **본 실행에서는 `f_r`도 강제한다** —
  f_xy prior가 `f_r` 예측을 직접 읽으므로 그 축의 침묵은 여기서만큼은 허용되지 않는다.
  → deck에 `[curriculum] gate_noreg_fr_guard_enabled = true`.
- `gate_legacy_tail`(Dataset-A 고-cyclen tail의 cyclen MAE)도 기본대로 함께 판정한다.
- 판정: `gate.json`의 `pass == true`.

### 판정 표 (실행 후 채울 것)

| 게이트 | 지표 | 기준 | 측정값 | 판정 |
|---|---|---|---|---|
| G1 | ρ_within-cell(f_xy): HEAD vs PROXY | `>= 0` (동률 시 paired CI) | TBD | TBD |
| G2 | MAE(f_xy) | `< 0.029` | TBD | TBD |
| G2b | MAE(HEAD) vs MAE(PRIOR) | HEAD < PRIOR | TBD | TBD |
| G3 | `gate-promote` | PASS (f_r 강제 포함) | TBD | TBD |

---

## 5. 왜 G3가 통과할 것으로 기대하는가 (그리고 왜 그래도 측정하는가)

레시피는 freeze-finetune이다: `--init-from data/models/s1i --freeze-trunk-cyclen --promote-fxy`.

1. trunk(`stem`/`blocks`/`films`/`head_trunk`/`conv_head`)는 `requires_grad=False`이고
   optimizer에서 제외된다 — decoupled weight decay조차 건드리지 못한다.
2. `mu_head`/`log_sigma_head`의 **cyclen 행**은 backward hook으로 grad가 0이 된다.
3. f_xy 행이 읽는 `f_r` 예측은 `detach()` 되어 있으므로(`net.PosValNet._compose_fxy`),
   f_xy loss의 gradient는 **F_r head로 흐르지 않는다**.
4. 챔피언 가중치는 `train._graft_appended_target_rows`가 prefix로 이식하고,
   나머지 key는 여전히 `strict=True`로 적재된다.

따라서 legacy 축이 움직일 수 있는 경로는 `mu_head`/`log_sigma_head`의
**cyclen 아닌 legacy 행들** 뿐이며, 이들도 f_xy loss로부터는 gradient를 받지 않는다.
그럼에도 이 실행은 distillation 항과 `--f-r-rank-weight`를 함께 켜므로
legacy 행은 여전히 학습된다 — **그러므로 G3는 형식이 아니라 실측이다.**

---

## 6. 사전에 금지하는 주장 (설계서 §3.6 승계)

- head의 f_xy 예측을 근거로 **미측정 core의 F_xy ≤ 1.65 적합성을 선언**하는 것.
  head는 acquisition의 신호이지 licensing 판정이 아니다. 인도(delivery) 판정은
  MASTER 측정값으로만 한다.
- **prior 단독**으로 feasibility를 말하는 것. prior의 최대 잔차는 0.08~0.10이고
  게이트 폭보다 크다(설계서 §3.4.4 경고).
- `F_r ≤ 1.55`가 `F_xy ≤ 1.65`를 함의한다는 주장 (실측 반례: E1_E2 18/52, regen T6_T4 41/213).
- MOCHA의 "Fxy ≤ 1.55 표본 1개" 통계를 lpopt 문맥에 인용하는 것 (한계값이 다름).

## 7. 미도입을 사전에 기록해 두는 것 (P4 범위 밖)

- **f_xy conformal 구간**: `conformal.CONFORMAL_TARGETS`는 7열 surrogate index로 키를 잡고
  f_xy에는 해당 열이 없으며, cell당 라벨 수가 `DEFAULT_MIN_CELL`(20)에 한참 못 미친다.
  1차에서는 f_xy 구간을 **내지 않는다** — 공허한 구간보다 없는 편이 정직하다(설계서 §3.4.3).
- **f_xy per-cell calibration**: 같은 이유(라벨 희소)로 P4에서 도입하지 않는다.
  라벨이 cell당 20행을 넘기면 `cell_calibrate`의 intercept-only 경로로 별도 결정.
- **7열 surrogate 확장**: vendor `TARGET_NAMES` 확장은 별도 결정 사항이며 본 실행 범위 밖.
  f_xy는 `predict_fxy`로 7열 계약 **바깥에서** 서빙된다.

---

## 8. 실행 커맨드 (확정)

`§2`의 N이 확정되고 본 문서에서 DRAFT가 제거된 뒤에만 실행한다.

```
python -m lpopt.model.al_retrain --champion data/models/s1i --add-fxy-head --dry-run
```
가 인쇄하는 3단계(teacher refresh → push → remote train)를 그대로 실행한다. 자세한 명령줄은
P4 구현 보고서에 기록되어 있다.

게이트:
```
lpopt gate-promote --prev data/models/s1i --new data/models/<new_ts> --out gate_fxy.json --check-only
```
