# RESULTS — `f_xy` prior-residual head **arm 2** · 후보 `data/models/20260829_163820`

채점 기준: `data/reports/fxy_head_prereg_20260829.md` **Amendment B**(구속력) — G1(A.7 그대로) ·
**G2′ MAE < 0.0767** · **G3′ ρ̄ > 0.7263** · G4 커버리지 ∈ [0.55, 0.80], **G2′∧G3′는 AND**.
A.5/A.6의 이상적-입력 바(0.0463 / 0.8944)도 B.5의 지시대로 **함께 보고**한다.
사전등록의 바·공식·판정규칙은 **하나도 수정하지 않았다.** 본 문서는 채점만 하며
**승격·개명·deck 수정을 수행하지 않는다** — 처분은 §10, 실행은 orchestrator.

arm 1 결과(`data/reports/fxy_head_results_20260829.md`)는 확정·불변이며 본 문서가 바꾸지 않는다.

산출 아티팩트

| item | path |
|---|---|
| G1 게이트 JSON | `data/reports/gate_fxy_arm2_20260829_checkonly.json` |
| G2′/G3′/G4 수치 JSON | `data/reports/fxy_gate_eval_arm2_20260829.json` |
| 채점 스크립트 (arm-1 하네스 사본, read-only) | `data/reports/fxy_gate_eval_arm2_20260829.py` |
| 보조 probe — e_core cell key 양방향 / F_r 편향 | `data/reports/fxy_gate_eval_arm2_ecore_probe_20260829.json` |
| 보조 probe — 멤버별 prior/잔차 분해 (VAL) | `data/reports/fxy_gate_eval_arm2_resid_probe_20260829.json` |
| 보조 probe — 같은 분해 (TRAIN fold) | `data/reports/fxy_gate_eval_arm2_trainfold_probe_20260829.json` |

> **arm-1 하네스는 손대지 않았다.** `data/reports/fxy_gate_eval_20260829.py`
> sha256 `b2bbc084ec12d631b9f251fa3d038da34b047d543b5c4fb89963f1bfb8dde0ca` (11,214 B) —
> 채점 전후 동일. arm 2 사본은 `CAND`/`OUT`과 **G2′/G3′·서빙 proxy 비교 블록**만 추가했다.

---

## 1. 판정표 (B.7을 채운 것)

| 게이트 | 지표 | arm 2 기준 | 측정값 | 판정 |
|---|---|---|---|---|
| **G1** | `gate-promote --check-only` | `pass == true`, ε = σ0·Φ⁻¹(0.95^(1/N)), `blind_targets == []` | `pass = true` · **N = 108, ε = 0.1388114093847**(실행값) / **N = 144, ε = 0.14216194539159127**(사전등록 f_r 강제 재산출) · worst enforced drop **0.012032** · `blind_targets = []` · `unavailable = []` · legacy tail PASS | **PASS** (두 읽기 모두) |
| **G2′** | MAE(f_xy), holdout n=793 | **< 0.0767** | **0.108560** | **FAIL** (바의 **1.415배**) |
| **G3′** | ρ̄ within-cell, 11 cell | **> 0.7263** (tie = FAIL) | **0.768111** · Δ vs PROXY_s1i **+0.041815** · paired BCa CI **[+0.020405, +0.067615]** (0 미포함 → **tie 아님**) | **PASS** |
| **G4** | 68% 커버리지 | ∈ [0.55, 0.80] | **0.634300** (σ̄ 0.1303, median 0.1233) | **PASS** |
| (참고) G2 | A.5 바 < 0.0463 | — | 0.108560 | FAIL (승격 조건 아님) |
| (참고) G2b | A.5 강한 형태 < 0.0355 | — | 0.108560 | 불성립 |
| (참고) G3 | A.6 바 ≥ 0.8944 | — | 0.768111 (Δ vs PRIOR **−0.126262**) | FAIL (승격 조건 아님) |
| (참고) bias | proxy의 **+0.005270** 과 비교 | — | **−0.088948** (부호 반전, 절대값 **16.9배**) | 판정 아님 |

**AND 규칙(B.5)에 따라 G2′ FAIL 하나로 승격 조건은 성립하지 않는다.**

**가설 판정.** H1(veto) **지지** — legacy 무회귀 확인(§2).
H2/H3의 arm-2 형태: **순위(ρ)는 이겼고 레벨(MAE)은 졌다** — arm 1과 정확히 같은 실패 패턴이며,
B.5가 "둘 다여야 한다"고 미리 못박은 이유가 그대로 재현되었다.
H4 **지지** — arm 2에서 처음으로 σ가 학습되었고 커버리지가 창 안에 들어왔다(§5).

**arm 1 대비 (같은 793행, 같은 11 cell)**

| 축 | arm 1 | arm 2 | 방향 |
|---|---:|---:|---|
| MAE(f_xy) | 0.10506 | **0.10856** | 악화 −0.0035 |
| bias | −0.08089 | **−0.08895** | 악화 |
| ρ̄ (11 cell) | 0.7422 | **0.76811** | 개선 **+0.0259** |
| σ̄ | 0.562 | **0.1303** | 개선 (라벨 sd 0.3364 대비 0.39배) |
| 68% 커버리지 | 0.9887 | **0.6343** | **FAIL → PASS** |

즉 **B.1의 #1·#2·#3(선택기준·warmup·σ 곡선)은 고쳐졌고, #4(합성이 읽는 F_r 행)는 고쳐지지 않았다.**
근거는 §6.

---

## 2. G1 — legacy 무회귀 (veto 축), PASS

실행 커맨드 (arm 1과 **동일 형식·동일 deck**):

```
python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1i --new data/models/20260829_163820 \
    --out data/reports/gate_fxy_arm2_20260829_checkonly.json --check-only
```

콘솔 (원문):

```
[no-regression] REPORT-ONLY axes (scored, NOT enforced): f_r — their drop cannot block
promotion (worst 0.0051 vs eps 0.1388). ...
RESULT: gate PASS -> NOT promoted (--check-only); re-run without --check-only to promote
```

| 항목 | 값 |
|---|---|
| `pass` | **true** (`no_regression` true · `legacy_tail` true) |
| 채점 cell / 행 | **36 cell** · **3,327행** (cyclen), node_peak·map_cov 각 36 cell / 1,765행 |
| `blind_targets` / `unavailable` | **[]** / **[]** (사전등록 조건 충족) |
| worst enforced drop | **0.012032** (`node_peak`, `5-5.25_f101`, 0.9576 → 0.9455, n=33) |
| legacy tail | PASS, ε 2.0, **worst_mae_increase 0.0** (세 밴드 모두 개선 −0.0355 / −0.0050 / −0.0186) |

### 2.1 ε — 두 읽기 (A.7이 요구한 실측 N·ε)

deck 수정 금지 원칙상 게이트는 계보와 같은 `lpopt.inp`(f_r report-only)로 실행했고,
"f_r 강제" 읽기는 **같은 실행의 `checks` 배열에서 동일 공식으로 재산출**했다(재채점 없음).

| 읽기 | 강제 축 | N | ε = 0.042·Φ⁻¹(0.95^(1/N)) | worst enforced drop | 판정 |
|---|---|---:|---|---:|---|
| 실행값 (f_r report-only) | cyclen, node_peak, map_cov | **108** | **0.1388114093847** | 0.012032 | PASS |
| **사전등록 (f_r 강제)** | + f_r | **144** | **0.14216194539159127** | 0.012032 | **PASS** |

worst drop이 어느 ε의 **8.7%** 에 불과해 두 읽기가 같은 판정을 낸다.
(arm 1의 worst drop 0.024138 보다도 작다 — 후보는 arm 1보다 legacy 축을 **덜** 흔들었다.)

### 2.2 축별 drop (36 cell, 144 checks)

| 축 | 강제 | n | worst drop | median drop | worst cell |
|---|:--:|---:|---:|---:|---|
| `cyclen` | ✅ | 36 | **+0.00045** | +0.00000 | `6-6.25_f101` (0.8254 → 0.8249, n=93) |
| `node_peak` | ✅ | 36 | **+0.01203** | −0.00414 | `5-5.25_f101` (0.9576 → 0.9455, n=33) |
| `map_cov` | ✅ | 36 | **+0.00488** | −0.00319 | `5.25-5.5_f133` (0.9591 → 0.9542, n=27) |
| `f_r` | (재산출 시 ✅) | 36 | **+0.00511** | −0.00058 | `5.75-6_f117` (0.9256 → 0.9205, n=60) |

`cyclen`의 median drop이 정확히 0인 것은 `--freeze-trunk-cyclen`의 구조적 귀결이다(arm 1과 동일).

### 2.3 holdout 정합성 (arm 1과 동일한 미세 불일치, 재확인)

게이트는 `S1j.json`을 읽지 않고 store에서 curriculum split manifest를 재계산한다. 36 cell
3,327행 중 **3행**(0.09%)이 `S1j.json.groups.curriculum_val_by_cell`(3,948 id / 87 cell)과
다르며, 그 3행은 S1j **train** fold에 속한다(`5-5.25_f117` 1행, `5-5.25_f125` 2행).
S1i train에도 속하므로 **두 챔피언 모두에게 in-sample** — 비교 대칭성은 깨지지 않으나
"3,327행 전부가 honest holdout"이라고는 쓸 수 없어 여기 등록한다. **arm 1과 동일한 3행이다.**

---

## 3. G2′ — f_xy 절대오차, **FAIL**

라벨된 S1j **VAL 793행** (converged; A.2의 793과 정확히 일치, §11에서 재확인).

| 추정기 | MAE | bias | resid sd | p95 abs | max abs |
|---|---:|---:|---:|---:|---:|
| **HEAD** (`predict_fxy`, 후보) | **0.108560** | **−0.088948** | 0.123330 | 0.3341 | 0.7808 |
| **PROXY on `s1i`** — **G2′의 바** (서빙 경로) | **0.076721** | +0.005270 | 0.109090 | 0.2202 | 0.5928 |
| PROXY on 후보 (후보 자신의 보정된 F_r) | 0.075330 | +0.003659 | 0.107475 | 0.2280 | 0.5764 |
| (이상적 입력) PRIOR on **측정** F_r (A.5 바의 출처) | 0.035533 | +0.002584 | 0.046321 | 0.0790 | 0.2939 |
| (이상적 입력) PROXY on **측정** F_r | 0.035529 | +0.002164 | 0.046298 | 0.0791 | 0.2935 |

- **G2′ 바 `< 0.0767` 대비 1.415배 → FAIL.** A.5 바 `< 0.0463` 대비로는 2.34배 → 참고 FAIL.
- head 오차는 **여전히 편향이 주도**한다: |bias| 0.0889 이 MAE 0.1086의 **82%**
  (arm 1은 77%). 잔차 sd는 head 0.1233 vs proxy 0.1091 로 **산포도 proxy보다 넓다**.
- **후보 체크포인트가 낼 수 있는 최선의 f_xy 추정은 그 자신의 proxy(0.0753)이지 head가 아니다.**
- B.5의 부수 보고 대조: 오프라인 예측(#e) MAE ≈ 0.0702. 실측 0.1086 은 그보다 **55% 나쁘고**,
  #a′ 의 *바닥*(잔차≡0) 0.0894 보다도 **나쁘다**. 이 이탈은 B.5가 "조사 대상"이라고 미리
  등록한 사건이며, 원인은 §6에서 측정으로 특정한다. **바는 사후에 움직이지 않았다.**

---

## 4. G3′ — cell 내 순위, **PASS**

cell 정의 **(case_pair, feed)**, holdout 라벨 ≥ 20 인 **11 cell / 486행** — A.6과 **정확히 같은 집합**.

| cell | n | ρ_HEAD | ρ_PROXY(s1i, 서빙) | ρ_PRIOR (측정 F_r) | MAE_HEAD | MAE_PRIOR |
|---|---:|---:|---:|---:|---:|---:|
| `E1_E2/f109` | 29 | 0.8926 | 0.8828 | 0.9783 | 0.1397 | 0.0367 |
| `E1_E2/f117` | 43 | 0.8138 | 0.8141 | 0.9799 | 0.1904 | 0.0463 |
| `E1_E2/f121` | 40 | **0.8120** | 0.7876 | 0.7659 | 0.0811 | 0.0293 |
| `E1_E2/f125` | 52 | 0.9064 | 0.9135 | 0.9857 | 0.1573 | 0.0437 |
| `E3_E4/f125` | 21 | 0.8000 | 0.7571 | 0.9610 | 0.2264 | 0.0737 |
| `G3_G4/f125` | 36 | 0.8644 | 0.8301 | 0.9516 | 0.1368 | 0.0316 |
| `J5_J6/f121` | 44 | 0.9074 | 0.8956 | 0.9663 | 0.0592 | 0.0271 |
| `N1_N2/f113` | 32 | 0.8084 | 0.7092 | 0.8730 | 0.0611 | 0.0248 |
| `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24/f125` | 22 | 0.4331 | 0.3235 | 0.5738 | 0.0341 | 0.0331 |
| `P6253Z1G06N24_P6253Z2G10N24/f125` | 20 | 0.4180 | 0.3098 | 0.8943 | 0.0450 | 0.0347 |
| `T6_T4/f121` | 147 | 0.7931 | 0.7659 | 0.9083 | 0.0392 | 0.0263 |
| **비가중 평균 (판정 기준)** | 486 | **0.76811** | **0.72630** | **0.89437** | 0.1086 | 0.0355 |
| (참고) n-가중 평균 | | 0.79976 | | | | |

- **점 추정 0.76811 > 바 0.7263 → 통과.** Δ = **+0.041815** 는 `0 ± 0.05` 밴드 **안**이므로
  B.5의 동률 조항이 **발동**했다: `ab_paired.paired_cell_bootstrap`
  (cell-clustered paired BCa, 11 cell, reps 2000, seed 0, aggregate mean)
  → point **+0.041815**, **CI [+0.020405, +0.067615]**, se 0.012155, `straddles_null = false`,
  `establishes_gain = true`, mde80 0.03405. **CI가 0을 포함하지 않으므로 tie가 아니다 → G3′ PASS.**
- 11 cell 중 **10 cell에서 head가 서빙 proxy보다 높다** (`E1_E2/f117` 만 −0.0003).
  B.3 #1이 예고한 대로 이 이득은 **오직 잔차에서만** 나올 수 있다 —
  prior는 F_r의 단조함수라 어떤 (a,b)로도 ρ를 못 움직인다.
- **A.6의 이상적 바 0.8944 대비로는 Δ −0.126262 → 참고 FAIL** (밴드 밖이라 A.6의 CI 조항은
  발동하지 않는다). 측정 F_r을 받는 추정기 자리는 여전히 도달 불가이며,
  §12의 금지 조항은 그대로 유지된다.

---

## 5. G4 — σ 커버리지, **PASS** (arm 2에서 처음)

| 항목 | arm 1 | **arm 2** |
|---|---:|---:|
| 68% 경험적 커버리지 (`abs(f_xy − μ) ≤ σ`) | 0.9887 | **0.63430** |
| σ̄ / median / min / max | 0.562 / — / — / — | **0.13032** / 0.12328 / 0.01746 / 0.34493 |
| 라벨 sd (793행) | 0.3364 | 0.3364 |

- 창 `[0.55, 0.80]` 안 → **PASS**. σ 서빙 사용 금지 조항은 발동하지 않는다.
- 두 코드 변경이 실제로 도달했음을 이 숫자가 증언한다: `warmup_epochs_effective = 8`
  (arm 1은 80)로 `log_sigma`가 학습되었고, `fit_calibration`이 `target_names`를 순회하여
  `calibration.json`의 `targets`가 **9열**(`…, "f_xy"`)이며 `isotonic`에 **`f_xy` 곡선이 존재**한다
  (s1i의 `targets`는 7열, f_xy 없음). §11.3 참조.
- **주의(등록):** G4 PASS는 σ만의 건전성이다. μ의 편향(−0.0889)은 σ 창 안에 들어와 있어도
  사라지지 않으며, G2′ 판정을 완화하지 않는다.

---

## 6. 왜 G2′가 실패했는가 — 측정으로 특정한 단일 원인

### 6.1 잔차는 이번엔 살아 있다 (B.1 #1·#2 해소)

멤버별 (VAL 793행, `predict_fxy` 경로 분해: `composed = a·mu[f_r]_raw + b + residual`):

| seed | best ep / early stop | prior (a, b) | 잔차 mean | 잔차 sd | prior-only bias | composed bias |
|---|---|---|---:|---:|---:|---:|
| 20260716 | 32 / 47 | 1.18579, −0.19545 | +0.01044 | 0.03137 | −0.08401 | −0.07357 |
| 20260717 | 43 / 58 | 1.19950, −0.20583 | −0.01735 | 0.02838 | −0.08509 | −0.10245 |
| 20260718 | **75 / 90** | 1.10815, −0.06975 | −0.00594 | 0.03730 | −0.05864 | −0.06458 |
| 20260719 | 30 / 45 | 1.20566, −0.23450 | −0.00983 | 0.02843 | −0.11046 | −0.12029 |
| 20260720 | 54 / 69 | 1.05678, +0.04611 | −0.00071 | 0.03455 | −0.08315 | −0.08385 |

- 잔차 sd 0.028–0.037 = 라벨 sd 0.3364의 **8–11%** (arm 1: 0.0259 = 8%). 절대 크기는
  arm 1과 비슷하나 **순위 정보를 실제로 담고 있다**: ρ̄ 0.7422 → 0.7681 (§4).
- 잔차 mean은 −0.017 ~ +0.010 로 **편향의 원인이 아니다**.
- **early stop이 f_xy를 본다**: 모든 멤버의 best epoch가 `select_score = composite + 0.5·fxy_select`의
  argmax이고(멤버 meta `best_metrics.select_score`), best epoch가 30–75로 arm 1의 4–37보다 훨씬 뒤다.
- **등록해 두는 정정:** 오케스트레이터가 전달한 early-stop `47/58/45/45/69` 중 3번째 값은 오기다.
  `train.log` 실측은 **47 / 58 / 90 / 45 / 69** (best epoch 32 / 43 / 75 / 30 / 54).

### 6.2 편향의 전부는 **합성이 읽는 F_r 행**에서 온다 (B.1 #4 미해소)

`net._compose_fxy`는 네트워크 안에서 **raw `mu[f_r]`** 위에 합성한다. 같은 793행에서:

| 경로 | raw F_r bias | raw F_r MAE |
|---|---:|---:|
| `predict`/`predict_fxy` 경로 (CaseKey에서 provenance 재구성) | **−0.071912** | 0.085479 |
| `predict_rows_raw` 경로 (store 행의 **자기 `library_id`** 로 featurize) | **+0.001965** | 0.049139 |
| 후보의 **보정된** F_r (`f_r_calibration.json` 적용 후, 서빙 열) | +0.001228 | 0.051610 |
| `s1i` raw / 보정 | −0.065512 / +0.002552 | — |

- 793행 중 **216행이 `library_id = paramA`**(577행 ga80). `predict`는 CaseKey에 provenance가
  없어 `_effective_library`로 재구성하는데, 그 경로의 raw F_r은 **−0.0719** 만큼 낮다.
  같은 행을 store의 자기 provenance로 featurize하면 raw bias는 **+0.0020** 로 사라진다.
- **결과 산술이 정확히 맞는다:** 평균 prior 기울기 ā = 1.15118 →
  `ā × (−0.071912) = −0.082783`, 여기에 잔차 mean(−0.0049)을 더하면 −0.0877,
  실측 head bias **−0.088948**. 즉 **편향의 93%가 이 한 항이다.**
- **그래서 "예측 F_r 위에서 prior를 적합"(B.4 #1)이 편향을 흡수하지 못했다.**
  `refit_fxy_prior_on_predicted`는 `LPDataset` 텐서(=store 행의 자기 provenance)로 forward하여
  `np.polyfit`으로 적합하므로 **적합 시점 잔차 평균은 정의상 0**이다. 그런데 서빙의
  `_compose_fxy`는 **다른 featurization의 raw `mu[f_r]`** 를 읽는다.
  TRAIN fold 5,425행에서 최종 체크포인트를 서빙 경로로 재현하면 prior-only bias가
  **−0.0749 ~ −0.1279**(앙상블 composed bias −0.1007)로 VAL과 같은 크기다 —
  **fold 문제가 아니라 경로 문제**임을 확인한다.
- **B.3 #4가 "train과 serve가 정의상 동일"하다고 적은 전제가 성립하지 않았다.**
  오프라인 B.2 #a′/#e 는 **서빙 백엔드 하나의 경로**에서 적합·평가했기에 편향이 흡수되었고
  (bias +0.0135 / +0.0041), 실제 학습은 그 경로가 아닌 곳에서 적합했다.
  이것이 오프라인 0.0702 → 실측 0.1086 격차의 원인이다.

### 6.3 학습 로그의 `fxyMAE`와 서빙 MAE의 괴리 (등록)

| 읽기 | 값 |
|---|---:|
| 학습 로그 best-epoch `fxyMAE` (멤버 5개, 같은 793행) | 0.0762 / 0.0772 / 0.0756 / 0.0769 / 0.0752 |
| 서빙 `predict_fxy` 단일 멤버 MAE | 0.1138 / 0.1229 / 0.1052 / 0.1450 / 0.1039 |
| 서빙 앙상블 MAE | **0.1086** |

- 같은 행·같은 지표인데 **약 +0.035** 차이가 난다. §6.2가 그 전부를 설명한다
  (featurization 경로가 다르고, `fxyMAE`는 학습 경로에서 계산된다).
- **판정은 서빙 값으로 한다.** B.5가 답하는 질문이 "head를 acquisition에 넣을 것인가"이고,
  acquisition은 `acquisition.predict_fxy` → `PosValCnnBackend.predict_fxy` 를 통과하기 때문이다.
  학습 로그의 0.076은 바 0.0767 과 사실상 동률이지만, **그 숫자는 서빙에서 관측되지 않는다.**
- 이 괴리는 **후보의 결함이 아니라 서빙 경로 전체의 성질**이다(§6.2 표에서 `s1i`도 raw −0.0655).
  legacy 축은 `f_r_calibration.json` 등 per-cell 보정이 이 오프셋을 흡수하므로 드러나지 않고,
  **보정을 받지 않는 f_xy 열만 그대로 노출된다.**

---

## 7. 멤버 건전성

### 7.1 leave-one-out / 단일 멤버 (같은 793행, 같은 11 cell)

| 구성 | MAE | bias | ρ̄ | 커버리지 |
|---|---:|---:|---:|---:|
| **전체 5멤버 (판정값)** | **0.10856** | −0.08895 | **0.76811** | 0.6343 |
| drop 20260716 | 0.10961 | −0.09279 | 0.75674 | 0.6343 |
| drop 20260717 | 0.10663 | −0.08557 | 0.77004 | 0.6393 |
| drop 20260718 | 0.11266 | −0.09504 | 0.75879 | 0.5902 |
| drop 20260719 | 0.10258 | −0.08111 | 0.76282 | 0.6482 |
| drop 20260720 | 0.11464 | −0.09022 | 0.76795 | 0.5712 |
| 단일 20260716 | 0.11383 | −0.07357 | 0.75004 | 0.2434 |
| 단일 20260717 | 0.12291 | −0.10245 | 0.72135 | 0.3291 |
| 단일 20260718 | 0.10521 | −0.06458 | 0.73937 | 0.5044 |
| 단일 20260719 | 0.14501 | −0.12029 | 0.73349 | 0.2346 |
| 단일 20260720 | 0.10390 | −0.08385 | 0.71577 | 0.5183 |

- **어떤 멤버를 빼도 G2′는 통과하지 못한다** (최선 0.10258 vs 바 0.0767).
  **어떤 멤버를 빼도 G3′는 통과한다** (최악 0.75674 vs 바 0.7263). **판정은 멤버 선택에 불변이다.**
- 단일 멤버의 커버리지가 0.23–0.52 로 낮고 앙상블이 0.634 인 것은 정상이다 —
  `calibrated_t`가 aleatoric + **epistemic**(멤버 분산)을 합치기 때문이며, LOO에서도 0.57–0.65로 유지된다.
- 멤버 `20260720`은 arm 1과 마찬가지로 composite가 낮다(0.7348 vs 0.79–0.81) — **s1i에서 상속된
  성질**이며 이번 학습이 만든 것이 아니다. 그 멤버를 빼도 MAE는 오히려 나빠진다(0.1146).

### 7.2 학습 완결성 — 후보는 **완료본**이다 (arm 1의 §10.5 선행조건 충족)

| 증거 | 값 |
|---|---|
| `rc` / `DONE` | **`0`** / **존재** |
| per-cell 보정 5종 | `f_r`(102 cell) · `cbc_max`(102) · `f_q`(102) · `ao_abs`(102) · `flatness`(node_peak 83 / map_cov 83) — **train.log에 5줄 모두 존재** |
| cyclen per-cell / physics prior | 챔피언 `s1i`에서 **COPIED** (동결 head, 재적합 안 함) |
| 마지막 로그 줄 | `trained 5 member(s) into runs/20260829_163820` |
| 실패 배너 | 없음 |
| 아티팩트 대조 | 후보 JSON 10종 = `s1i` JSON 10종 (**완전 일치**, §11.3) |
| `calibration.json.targets` | **9열** — `[f_r, f_q, cbc_max, cyclen, ao_abs, discharge_burnup, max_pin_burnup, max_assembly_burnup, f_xy]`, `isotonic`에 `f_xy` 곡선 존재 (**arm 1·s1i에는 없음**) |

**arm 1과 달리 "보정 결여 때문에 MAE 비교가 불공정하다"는 단서가 필요 없다** — §9의 고정 자는 그대로 읽힌다.

---

## 8. proxy vs head — 서빙 경로 정면 비교 (B.5의 판단 대상)

`acquisition.fxy_proxy`는 `prediction.mean[:,0]` = **보정된** F_r을 읽는다. 따라서 정직한 비교는 아래다(같은 793행).

| 경로 | MAE | bias | ρ̄ (11 cell) | σ̄ | 68% 커버리지 |
|---|---:|---:|---:|---:|---:|
| **PROXY on `s1i`** — 오늘 실제로 작동 중인 fallback (**바**) | **0.07672** | **+0.00527** | 0.72630 | 0.1982 | 0.9369 |
| PROXY on 후보 (후보 자신의 보정된 F_r) | **0.07533** | +0.00366 | 0.72677 | 0.2041 | 0.9521 |
| **HEAD** (후보 `predict_fxy`) | 0.10856 | −0.08895 | **0.76811** | 0.1303 | 0.6343 |
| (이상적 상한) PRIOR on **측정** F_r | 0.03553 | +0.00258 | 0.89437 | — | — |

**읽기.**

1. head는 서빙 proxy 대비 **레벨이 41.5% 나쁘고**, 편향은 +0.005 → −0.089 로 **부호가 뒤집히며
   17배 커진다**. 반대로 **순위는 유의하게 낫다**(+0.0418, CI가 0을 배제).
   B.5가 미리 규정한 대로 **레벨과 순위 둘 다여야 하며, 하나만으로는 승격하지 않는다.**
2. arm 1과 달리 이번 head는 **proxy의 재현이 아니다** — 잔차가 실제 순위 정보를 더했다(§4·§6.1).
   그러나 그 정보를 **−0.089의 계통 편향에 실어서** 내보낸다. `F_xy ≤ 1.65` 같은 **수준 임계**를
   쓰는 acquisition에서 그 크기의 낙관 편향은 순위 이득으로 상쇄되지 않는다
   (라벨의 `f_xy ≤ 1.65` 비율이 17.6%뿐이고, 편향은 벽 쪽으로 향한다).
3. **후보를 승격하더라도 f_xy는 head가 아니라 후보의 proxy를 쓰는 편이 낫다**(0.0753 < 0.1086).
   이는 §10의 처분과 독립적인 사실로 기록해 둔다.
4. §6.2가 특정한 원인은 **재학습 없이 고칠 수 있는 종류**다(합성이 읽는 F_r 행을 서빙 경로와
   일치시키거나, f_xy 열에 자체 per-cell 보정을 두거나). 다만 그 선택은 **새 개정을 요구**하며
   본 문서는 결정하지 않는다.

---

## 9. 고정 자(尺) — legacy 축, 후보 vs `s1i` (같은 793행)

> arm 1과 달리 **후보와 `s1i`가 같은 보정 아티팩트 10종을 모두 갖추고 있으므로 MAE 비교도 공정하다.**

| 축 | n | ρ_global s1i → 후보 | ρ_cellmean s1i → 후보 | MAE s1i → 후보 |
|---|---:|---|---|---|
| `cyclen` | 793 | 0.995378 → 0.995380 | 0.82309 → 0.82334 | 1.85212 → **1.85153** |
| `F_r` | 793 | 0.96742 → 0.96831 | 0.79088 → 0.79230 | 0.052678 → **0.051610** |
| `F_q` | 793 | 0.96458 → 0.96556 | 0.79660 → 0.80326 | 0.079257 → **0.078294** |
| `CBC_max` | 793 | 0.98496 → 0.98550 | 0.80290 → 0.81579 | 14.2101 → **13.5595** |
| `node_peak` | 790 | 0.96361 → 0.96770 | — | 0.048772 → **0.046614** |
| `map_cov` | 790 | 0.98460 → 0.98637 | — | 0.011565 → **0.010709** |

**여섯 축 전부 ρ·MAE 모두 개선 또는 불변이다.** arm 1의 ⚠ 표시(보정 결여로 인한 MAE 악화)는
완료본에서 **전부 사라졌다**. `cyclen`은 동결 head의 귀결로 사실상 불변.

---

## 10. 처분 (Amendment B.5 처분 규칙의 적용 — 실행은 orchestrator)

B.5: **G1·G2′·G3′ 모두 PASS → `s1j` 승격** / **FAIL-G1 → reject** /
**G1 PASS · G2′ 또는 G3′ 중 하나라도 FAIL → 챔피언 `s1i` 유지, head는 배포하지 않는다(shadow 포함).**

측정 결과는 **G1 PASS · G2′ FAIL · G3′ PASS · G4 PASS** 이다.

1. **승격 없음.** `s1j` 승격 **금지**. 챔피언은 **`data/models/s1i`로 유지**되며,
   `lpopt.inp`/`lpopt_gpu1.inp`의 `model_dir`은 손대지 않는다.
   `fpcamp_minfxy_T6T4_f121_r1_199.inp`는 `data/models/s1i` 그대로 launch한다
   (minfxy prereg §5.3: "승격되어 있지 않으면 s1i 그대로 launch, head를 기다려 라운드를 미루지 않는다").
2. **shadow 배포도 하지 않는다.** B.5가 이 경우를 명시적으로 규정했다("shadow 포함").
   arm 1과 달리 head가 새 신호(순위 +0.042)를 갖고 있는 것은 사실이나, **처분 규칙은 결과에
   따라 해석하지 않는다.** 그 신호는 §6.2의 원인 규명과 **새 사전등록**을 통해 회수한다.
3. **`f_xy` 서빙은 proxy 단일 경로로 유지.** A.9의 상수(1.2176 / −0.2519 / resid_sd 0.0476 / K 3.0)
   가 계속 유효하며, §8 표가 그 선택을 지지한다.
4. **G1 PASS이므로 "reject"(새 사전등록 없이는 재시도 금지) 조항은 발동하지 않는다.**
   실패한 것은 head의 레벨이지 legacy 무회귀가 아니다.
5. **B.2 #g(direct) 전환은 자동 발동하지 않는다**(B.5 명시). §6.2의 원인은 direct 전환과 무관하다 —
   direct도 같은 featurization 경로를 통과한다.
6. **다음 arm의 최소 조건 (설계 제안, 본 문서는 결정하지 않는다):** §6.2가 특정한
   `predict` 경로 raw `mu[f_r]` 오프셋(−0.0719)을 **학습·서빙 어느 한 쪽으로 통일**하는 것.
   구체적 후보는 (a) prior 재적합을 학습 featurization이 아닌 **서빙 featurization**에서 수행,
   (b) `_compose_fxy`가 보정된 F_r을 읽도록 변경, (c) f_xy 열 전용 per-cell 보정 도입.
   **어느 것도 재학습을 반드시 요구하지는 않는다** — (c)는 현재 체크포인트 위에서 가능하다.
   어느 쪽이든 **새 사전등록이 필요**하다.

### 10.1 PASS였다면 orchestrator가 해야 했을 편집 (참고 — **본 판정에서는 발동하지 않는다**)

아래는 요청에 따라 열거만 한다. **G2′ FAIL이므로 하나도 실행하지 않는다.**

| # | 편집 | 상세 |
|---|---|---|
| 1 | 디렉터리 개명 | `data/models/20260829_163820` → `data/models/s1j` |
| 2 | `lpopt.inp` | `[model] model_dir = "data/models/s1j"` (41행) + 주석을 s1j 계보로 갱신. `cond_schema`는 후보 meta가 **v8**이므로 **변경 없음** |
| 3 | `lpopt_gpu1.inp` | 같은 위치(40행) `model_dir` 및 주석 갱신 |
| 4 | 캠페인 deck | `fpcamp_minfxy_T6T4_f121_r1_199.inp` 156행 `model_dir` **한 줄만** → `data/models/s1j`. 다른 knob 금지(deck 자체 "LAUNCH-TIME MODEL SUBSTITUTION" 절, minfxy prereg §5.3) |
| 5 | deck 재해시 | 현재 `4AF8B0218666EC83E0E9357C6FB268F179301EE30BEC9CD1CAA89C3144C2FAC5` (16,957 B) → 새 값 산출 |
| 6 | launcher | `launch_fpcamp_minfxy_T6T4_f121_r1_199.ps1` 38행 `$modelName = 's1j'`, 63행 `$want = <새 deck sha>` |
| 7 | prereg stamp | `data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md` **§9.1** 표의 "launch-time 치환 시 (s1j)" 두 TBD 행에 새 deck sha256 과 `data/models/s1j/member_20260716/meta.json` sha256 을 stamp. (deck 주석이 말하는 "pre-registration §9"는 **head prereg의 §9가 아니라 minfxy prereg §9.1**이다 — head prereg §9는 "미도입 항목" 절이며 stamp 대상이 아니다) |
| 8 | head prereg | `data/reports/fxy_head_prereg_20260829.md` **B.7** 판정표에 본 문서 §1의 측정값 기입 |
| 9 | `fit_map_calibration` 재적합 | **계보는 승격 시 이것을 하지 않았다.** `data/store/map_calibration.json` 의 `fit.model_id = "split_S1b"`, `fitted_at = 2026-08-10T12:37`, `n_cells_fitted = 2` — s1h·s1i 승격을 **모두 건너뛰었다**. `ab2_addendum_S1I_20260817.md` §6의 승격 체크리스트도 `pull` + `gate-promote` 두 줄뿐이다. 따라서 s1j 승격 시에도 **계보 준수만으로는 재적합 의무가 없다.** 다만 이 아티팩트가 3세대 낡았다는 사실은 별도 항목으로 남긴다 |

> **선행 차단 사항 (승격 여부와 무관, 지금 유효).** launcher가 gate하는 store sha256
> `0334E2D2E303CD8E82373861603F82A57054FC2FC6139F84C360822580644D9D` 는 **e_core backfill 이전**
> 값이다. 현재 store는 `4CFF270B1020C87A2EC41BE3FE9595C481970197D01FC5AB58A174B194225057` 이므로
> **오늘 launcher를 돌리면 `MINFXY1 REFUSED: store sha256 mismatch` 로 거부된다.**
> minfxy prereg §9.1의 store 행과 launcher `$wantStore` 를 함께 갱신하지 않으면 캠페인은 출발하지 못한다.
> (본 문서는 어떤 파일도 수정하지 않았다.)

---

## 11. 출처와 해시 (provenance)

### 11.1 store parquet — 학습 시작 후 두 차례 수정되었다 (라벨·피처는 불변)

| 스냅샷 | sha256 | bytes | 비고 |
|---|---|---:|---|
| 사전등록 §5 동결값 = **trainer가 본 것** | `cf495c7d82b16cbfe4216333ca4d266a324514c223bd7e0a2c38f799445326cc` | 22,315,679 | `…bak_pre_nullphantom_20260829` |
| arm 1 채점 시점 | `0334e2d2e303cd8e82373861603f82a57054fc2fc6139f84c360822580644d9d` | 22,142,229 | `…bak_pre_ecore_backfill_20260829` (= minfxy launcher가 gate하는 값) |
| **현재 store (본 채점이 사용)** | `4cff270b1020c87a2ec41be3fe9595c481970197d01fc5ab58a174b194225057` | 22,144,665 | 2026-08-29 15:59, e_core backfill 후 |

**전 컬럼 대조 (74,717행 × 41열, record_id 집합·컬럼 집합 동일):**

```
현재 vs trainer가 본 것 : DIFF COLS = [('parent_record_id', 2528), ('e_core', 12607), ('e_split', 5093)]
현재 vs arm-1 채점본     : DIFF COLS = [('e_core', 12607), ('e_split', 5093)]
```

`e_core` 12,607행 중 **실질 변경은 979행**(|Δ| > `ECORE_BACKFILL_TOL` = 0.005, 최대 |Δ| 0.0684)과
**null 채움 274행**이며, 나머지 11,354행은 `backfill_e_core`가 resolvable 행 전체를 재기록하며
남긴 **부동소수 재직렬화**다(중앙값 |Δ| 8.9e-16; |Δ| > 1e-9 인 행은 1,342 = 979 + 363의 tol 이하 미세drift).
`e_split`은 **5,093행 전부 null → 값**이고 기존 값이 바뀐 행은 **0**이다.
오케스트레이터가 전달한 979 / 274 와 정확히 일치한다.

**라벨·피처 컬럼은 전부 bit-identical** — `f_xy`·`f_r`·`converged`·`case_pair`·`feed`·`pattern`
어느 것도 움직이지 않았다(793행 슬라이스에서 개별 확인). 따라서 현재 store로 채점하는 것이 정당하다.

### 11.2 **e_core 정정이 cell key를 바꾸었는가 — 양방향 측정**

793행 VAL 슬라이스에서:

| cell 정의 | 사용처 | 변경된 행 |
|---|---|---:|
| **(case_pair, feed)** | **G3/G3′의 채점 cell** | **0 / 793** |
| (feed, e_core-bin @0.05), **store 컬럼**에서 산출 | 서빙 어디에도 쓰이지 않음 | **51 / 793** |
| (feed, e_core-bin @0.05), **서빙 recipe**(`backend.cyclen_e_core(pattern)`) | per-cell 보정 lookup | **0 / 793** (정의상 pattern의 함수) |

- **G2′/G3′/G4 는 e_core 정정에 완전히 불변이다.** 채점 cell이 (case_pair, feed) 이고
  그 키를 만드는 세 컬럼이 bit-identical이기 때문이다. 이 축에서는 "양방향" 재계산이
  **수치적으로 동일한 두 계산**이 된다.
- **서빙 경로도 불변이다.** `model_api._calib_cell_keys` 는 store 컬럼이 아니라
  `cyclen_e_core(pattern)` 을 쓴다. 실측: 서빙 recipe 키와 **backfill 후** store 컬럼 키는
  **793/793 완전 일치**, **backfill 전** 컬럼 키와는 **51행 불일치**
  (주요 이동: `feed=121 ebin 4.90→4.85` 23행, `feed=109 5.30→5.25` 16행, `feed=125 5.70→5.65` 7행).
  → **backfill은 store 컬럼을 서빙 recipe에 맞춘 것이며, 서빙이 읽는 값을 바꾸지 않았다.**
- 그럼에도 요청대로 **(feed, e_core-bin) cell 위의 보조 ρ̄를 양방향으로** 산출한다
  (게이트가 아니며 판정에 쓰지 않는다):

| e_core 출처 | ≥20행 cell | 행 | ρ̄_HEAD | ρ̄_PROXY(s1i) | Δ |
|---|---:|---:|---:|---:|---:|
| **backfill 후** (= 서빙 recipe와 동일) | 16 | 649 | **0.82102** | 0.80253 | +0.01849 |
| backfill 전 | 15 | 650 | 0.82637 | 0.80896 | +0.01741 |

두 읽기 모두 **부호가 같고 크기도 같다** — cell 정의를 (feed, e_core-bin)으로 바꾸어도
"head가 proxy보다 순위가 낫다"는 결론은 유지되며, **어떤 판정도 바뀌지 않는다.**

### 11.3 후보 체크포인트

| 항목 | 값 |
|---|---|
| dir | `data/models/20260829_163820` |
| `rc` / `DONE` / `heartbeat` | `0` / 존재 / `1787996514` = 2026-08-29 18:41:54 (학습 종료 시각) |
| `ensemble.json` (sha `861ab433…`) | 5 멤버, `split: S1j`, `parallel_members: 1` |
| `target_names` | legacy 8 + **`f_xy` (idx 8)** |
| `calibration.json` (sha `28d05ba7…`, 39,598 B) | `targets` **9열, `f_xy` 포함**; `isotonic.f_xy` 곡선 존재; `n_val_used = 12,079` — **s1i는 7열, f_xy 없음** |
| `member_20260716/meta.json` (sha `abca088f…`, 37,305 B) | `cond_schema v8`, `vendor_manifest_sha256 0c4c69c2…` (**s1i와 동일**), `n_params` 2,269,412 / total 10,381,509 |
| `train_config` | `warmup_epochs 2`, `epochs 150`, `fxy_prior_on_predicted true`, `fxy_prior_residual true`, `fxy_select_weight 0.5`, `init_from data/models/s1i`, `freeze_trunk_cyclen true`, 224/8/384 |
| `schedule` | `effective_batch 1024`, `batch_scale 4.0`, **`warmup_epochs_effective 8`** (arm 1은 80) |
| `fxy_head` (meta, 멤버별) | `mode prior_residual`, `prior_source predicted`, `select_weight 0.5`, `n_labelled_train 5425`, `serve_affecting true`, prior `split "train:predicted_f_r"` |
| 멤버별 prior (a, b, pearson, resid_sd) | 1.18579/−0.19545/0.9842/0.05811 · 1.19950/−0.20583/0.9725/0.07653 · 1.10815/−0.06975/0.9560/0.09643 · 1.20566/−0.23450/0.9853/0.05606 · 1.05678/+0.04611/0.9388/0.11322 |
| best epoch / early stop | 32/47 · 43/58 · **75/90** · 30/45 · 54/69 |
| 완결성 | **완료본** (§7.2) |
| 실행 | `run.sh`: `CUDA_VISIBLE_DEVICES=1`, deck `lpopt_gpu1.inp`, 커맨드 문자열은 **B.6과 일치** |

> 오케스트레이터가 전달한 prior `1.1858·F_r_pred − 0.1954` 는 **멤버 `20260716`의 값**이다.
> **prior는 멤버마다 따로 적합된다** (a 1.0568–1.2057, resid_sd 0.056–0.113) — 앙상블 단일 상수가 아니다.

### 11.4 그 밖의 동결 산출물

| item | sha256 | bytes |
|---|---|---:|
| `data/splits/S1j.json` | `321950cb8fc965118569b5afa7943f2f2b09b39b231da228a177eac86e7ba3b1` | 6,033,916 |
| `lpopt.inp` (채점 전후 동일) | `bbee1bf3cc8252ba7208cd8104fc21f7f4244eb6bec4a5ae2f6d101d5ef3245e` | 13,260 |
| `lpopt_gpu1.inp` (동일) | `11653fd13d78b8ff06e312e74e08779d9823974974209b6729c6eb64a62b6d4a` | 9,987 |
| `fpcamp_minfxy_T6T4_f121_r1_199.inp` (동일) | `4af8b0218666ec83e0e9357c6fb268f179301ee30bec9cd1caa89c3144c2fac5` | 16,957 |
| `data/curriculum/state.json` (동일) | `479b4b0e1eaf691e5d744bea6871a44cde6322ba8cda3a1e11b59f0ec1a9fccf` | 2,781,344 |
| `data/reports/fxy_gate_eval_20260829.py` (arm-1 하네스, **미변경**) | `b2bbc084ec12d631b9f251fa3d038da34b047d543b5c4fb89963f1bfb8dde0ca` | 11,214 |

S1j 재계산 검증(현재 store): 총 74,717행 / 라벨(converged) 6,218 / train 5,425 · val 793 · 미분류 0,
G3 대상 **11 cell / 486행**, ρ̄_PRIOR **0.89437** — **A.2/A.6과 소수 4자리까지 일치**.
**사전등록의 바는 실행 후 어떤 재계산으로도 이동하지 않았다.**

### 11.5 등록해 두는 실행 편차 — GPU 1

`run.sh`는 `CUDA_VISIBLE_DEVICES=1`, deck은 `lpopt_gpu1.inp` (`[remote] gpu = 1`)다.
B.6은 이것이 상시 지시(`lpopt.inp`: *"사용자 지시 2026-07-24: GPU 0 고정, 재허가 전까지 auto 금지"*)와
충돌하며 **orchestrator가 재허가를 확인해 기록한 뒤에만 실행**하라고 등록해 두었다.
재허가가 있었는지는 본 채점자가 확인할 수 없다 — **orchestrator가 기록할 사항**이다.
(G1–G4의 어떤 수치에도 영향을 주지 않는다.)

### 11.6 재현

```
python -m lpopt gate-promote --input lpopt.inp --prev data/models/s1i \
    --new data/models/20260829_163820 \
    --out data/reports/gate_fxy_arm2_20260829_checkonly.json --check-only
python data/reports/fxy_gate_eval_arm2_20260829.py   # -> fxy_gate_eval_arm2_20260829.json
```

본 채점은 **코드를 수정하지 않았고**, deck·`state.json`·챔피언 디렉터리·후보 디렉터리·
arm-1 하네스를 **건드리지 않았다**. 새로 만든 파일은 위 두 JSON, 보조 probe JSON 3종,
채점 스크립트 `data/reports/fxy_gate_eval_arm2_20260829.py`, 그리고 본 문서뿐이다.

---

## 12. 사전등록 §8이 금지한 주장 — 본 문서에서의 준수

- head의 f_xy 예측으로 미측정 core의 `F_xy ≤ 1.65` 적합성을 **선언하지 않았다.**
  A.6의 이상적 바(0.8944)를 넘지 못했고 편향이 −0.089 이므로 그럴 자격이 없다.
- **prior 단독으로 feasibility를 말하지 않았다.** §3·§8의 PRIOR 열은 **바의 출처**로서만 쓰였다.
- `F_r ≤ 1.55 ⇒ F_xy ≤ 1.65`를 주장하지 않았고, MOCHA 통계를 인용하지 않았다.
- G3′ PASS를 "head가 proxy보다 낫다"로 **일반화하지 않았다** — 측정된 것은 **11 cell 안의 순위**이며,
  같은 head가 **레벨에서는 proxy보다 41.5% 나쁘다**(§8).
