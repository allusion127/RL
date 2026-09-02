# RESULTS — `f_xy` head **arm 3** (`--fxy-direct`) · 후보 `data/models/20260829_194532`

- 사전등록: `data/reports/fxy_head_prereg_20260829.md` **Amendment C** (구속력).
- 채점 경로: **FIXED serve path** (C.3) — `model_api.py` sha `94229de9…`,
  `featurize.py` sha `6977344d…`. 두 값 모두 C.3 표와 **일치 확인**.
- 채점 시점의 보정: **양쪽(후보·`s1i`) 모두 FIXED path 에서 재적합**된 per-cell 보정 6종.
  실행 기록은 `data/reports/servefix_calibration_refit_20260829.md`.
  이로써 C.3 #3 이 등록해 둔 **비대칭이 해소되었다**(아래 §9.1).
- 채점기: `data/reports/fxy_gate_eval_arm3_20260829.py`
  (sha `c191141a2325e0f068cba3ea1dee119e49aa829ace594d318e064b740ddbbc12`, 16,199 B)
  = arm-2c 하네스에서 **`CAND` 와 `OUT` 두 줄만** 바꾼 사본. 바 상수 `G2P_BAR = 0.0767` /
  `G3P_BAR = 0.7263` **무변경**. arm-1·arm-2 하네스는 **건드리지 않았다**.
- **본 문서는 어떤 승격·개명·deck 수정도 하지 않았다.** 처분 실행은 orchestrator 다.

---

## 1. 판정표 (C.6 을 채운 것)

| 게이트 | 기준 | 측정값 | 판정 |
|---|---|---|---|
| **G1** | `gate-promote --check-only` `pass == true`, `blind_targets == []` | `pass = true` · **N = 108, ε = 0.1388114093847**(실행값) / **N = 144, ε = 0.14216194539159127**(사전등록 f_r 강제 재산출) · worst enforced drop **0.011364** · `blind_targets = []` · `unavailable = []` · 36 cell / 144 checks | **PASS** (두 읽기 모두) |
| **G2′** | MAE(f_xy) **< 0.0767**, holdout n=793 | **0.066300** | **PASS** (바의 0.865배) |
| **G3′** | ρ̄ within-cell **> 0.7263**, 11 cell 비가중 평균 | **0.790392** (Δ = +0.064092) | **PASS** |
| **G4** | 68% 커버리지 **∈ [0.55, 0.80]** | **0.831021** (σ̄ 0.119787) | **FAIL** (상단 초과) |
| (참고) G2 | A.5 바 < 0.0463 | 0.066300 | FAIL — 승격 조건 아님 |
| (참고) G3 | A.6 바 ≥ 0.8944 | 0.790392 (prior ρ̄ 0.894372) | FAIL — 승격 조건 아님 |
| (참고) bias | proxy 의 +0.0053 과 비교 | **−0.003137** (resid sd 0.099290) | 판정 아님 |
| **동반 판독 (판정 아님)** | FIXED path·재적합 `PROXY on s1i` MAE | **0.073173** | — |
| **동반 판독 (판정 아님)** | FIXED path·재적합 `PROXY on s1i` ρ̄ (11 cell) | **0.715696** | — |
| (기록) | 채점에 쓴 `model_api.py` sha256 | `94229de9…` — **FIXED** | — |
| (기록) | meta `fxy_head.mode == "direct"` 인가 | **예 — 5개 멤버 전부 `"direct"`** | — |
| (기록) | `[1]` distill 캐시 재생성 여부 | **재생성 안 함** — `_v5_distill_soft.npz` mtime 2026-08-29 13:52:12, arm 2(16:38)·arm 3(19:45) 두 실행이 **같은 캐시**를 썼다 | — |
| (기록) | GPU 재허가 기록 | `run.sh` 가 `CUDA_VISIBLE_DEVICES=1`, deck `lpopt_gpu1.inp`. 재허가 유무는 **채점자가 확인할 수 없다 — orchestrator 기록 사항** (§10.4) | — |

### 처분 (C.4 처분 규칙의 적용)

C.4: **G1 PASS · G2′ PASS · G3′ PASS → `s1j` 승격.** G4 실패 시 처분은 §4/B.5 그대로
"**처분을 바꾸지 않되 σ 서빙 사용 금지**".

측정은 **G1 PASS · G2′ PASS · G3′ PASS · G4 FAIL** 이다.

> ## **PASS → `s1j` 로의 승격을 권고한다.** 단 **head 의 σ 를 서빙에 쓰지 않는다** (G4).

두 판독(고정 바 vs 재측정 proxy)이 **엇갈리지 않는다** — C.3 #2 가 요구한 대조에서
후보는 고정 바(0.0767 / 0.7263)와 재측정 proxy(0.073173 / 0.715696)를 **둘 다** 이긴다.
따라서 C.3 #2 의 "엇갈리면 orchestrator 에게 넘긴다" 조항은 **발동하지 않는다**.

---

## 2. G1 — legacy 무회귀 (veto 축), PASS

`data/reports/gate_fxy_arm3_20260829_checkonly.json`
(sha `371f0e5a589023d5156da0954a54825375fb41e39f9651b0875f0b2e54be948f`).

| 항목 | 값 |
|---|---|
| `pass` | **true** |
| 강제(enforced) 축 | `cyclen`, `node_peak`, `map_cov` |
| report-only 축 | `f_r` (worst drop 0.0045, ε 대비 무해) |
| cell 수 | 36 |
| checks | 144 (= 36 × 4 축) |
| worst drop (enforced) | **0.011364** |
| worst drop (any axis) | 0.011364 |
| `blind_targets` | **[]** |
| `unavailable` | **[]** |

### 2.1 ε — 두 읽기 (A.7 이 요구한 실측 N·ε)

`eps_N = σ0·Φ⁻¹(0.95^(1/N))`, `σ0 = 0.042`:

| 읽기 | N | ε | worst drop | 판정 |
|---|---:|---:|---:|---|
| 실행값 (강제 3축 × 36 cell) | **108** | **0.1388114093847** | 0.011364 | **PASS** (여유 12.2배) |
| 사전등록 재산출 (`f_r` 까지 강제 가정) | **144** | **0.14216194539159127** | 0.011364 | **PASS** |

두 ε 모두 공식으로 재계산해 자릿수까지 일치함을 확인했다.

### 2.2 arm 2 (같은 조건, 재적합 후) 와의 대조

| | arm 2 (`…163820`) | **arm 3 (`…194532`)** |
|---|---:|---:|
| `pass` | true | **true** |
| worst enforced drop | 0.011029 | **0.011364** |
| `f_r` report-only worst | 0.0038 | 0.0045 |
| `blind_targets` | [] | **[]** |

두 후보 모두 veto 를 통과하며 차이는 무의미하다 (ε 의 8% 수준).

---

## 3. G2′ — f_xy 절대오차, **PASS**

라벨된 S1j VAL **793행**, `predict_fxy` 의 `source = "head"`.

| 추정기 | MAE | bias | resid sd | RMSE | p95 abs |
|---|---:|---:|---:|---:|---:|
| **HEAD (arm 3, direct)** | **0.066300** | **−0.003137** | 0.099290 | 0.099277 | 0.2206 |
| HEAD (arm 2, prior-residual, 재적합 후) | 0.066886 | −0.000821 | 0.099900 | 0.099840 | 0.2192 |
| PROXY on `s1i` (서빙, 재적합 후) | 0.073173 | +0.001872 | — | — | — |
| PROXY on 후보 자신의 served F_r | 0.072149 | — | — | — | — |
| PRIOR (측정 F_r 위의 `1.2161·f_r − 0.2488`) | — | — | — | — | — |

**바 0.0767 대비 0.066300 — PASS.** arm 2 가 ORIGINAL path 에서 기록한 0.108560(FAIL)과
비교하면, 그 실패의 원인이던 `mu[f_r]` 경유 편향이 arm 3 에서는 **구조적으로 부재**하다
(C.1: `fxy_ref_idx = -1` 이므로 f_xy 행이 `mu[f_r]` 을 읽지 않는다).

**부수 보고 (게이트 아님) 대비.** C.4 가 등록한 오프라인 상한은 MAE ≈ 0.0708 /
bias ≈ +0.0047 / ρ̄ ≈ 0.7664 였다. 실측 **0.0663 / −0.0031 / 0.7904** 는 그 자리에서
크게 벗어나지 않았다(오히려 세 축 모두 소폭 우수). **조사 대상 아님.**

---

## 4. G3′ — cell 내 순위, **PASS**

cell = (case_pair, feed), holdout 라벨 ≥ 20 인 **11 cell / 486행** — A.6·B.5·C.4 와 동일 집합.

| | ρ̄ (11 cell 비가중 평균) |
|---|---:|
| **HEAD (arm 3)** | **0.790392** |
| HEAD (arm 2, 재적합 후) | 0.785331 |
| **바 (B.5/C.4 고정)** | **0.7263** |
| PROXY on `s1i` (재측정, 동반 판독) | 0.715696 |
| PRIOR (측정 F_r) | 0.894372 |

Δ(HEAD − 바) = **+0.064092**. |HEAD − 재측정 proxy| = 0.074696 > 0.05 이므로 C.4 의
tie 규칙(±0.05 이내에서만 BCa 판정)은 **적용 구간 밖**이며 점 추정으로 판정한다 →
`tie = False`, **PASS**.

그럼에도 확인을 위해 cell-clustered paired BCa 를 산출했다 (`ab_paired.paired_cell_bootstrap`,
2,000 reps, seed 0, 11 cell, dropped 0):

| 대조 | point | 95% CI | 결론 |
|---|---:|---|---|
| **HEAD(arm 3) − PROXY on `s1i`(재적합)** | **+0.074696** | **[+0.021430, +0.264692]** | CI 가 0 을 포함하지 않음 → **gain 확립** |
| HEAD(arm 3) − HEAD(arm 2) | +0.005061 | [−0.004133, +0.015511] | 0 을 포함 → **tie** (§8) |

### 4.1 cell 별 ρ (11 cell)

| cell | n | ρ HEAD | ρ PRIOR | ρ PROXY | MAE HEAD |
|---|---:|---:|---:|---:|---:|
| `E1_E2/f109` | 29 | 0.9044 | 0.9783 | 0.9783 | 0.0635 |
| `E1_E2/f117` | 43 | 0.8324 | 0.9799 | 0.9799 | 0.0942 |
| `E1_E2/f121` | 40 | **0.8075** | 0.7659 | 0.7659 | 0.0461 |
| `E1_E2/f125` | 52 | 0.9242 | 0.9857 | 0.9857 | 0.0823 |
| `E3_E4/f125` | 21 | 0.8104 | 0.9610 | 0.9610 | 0.1301 |
| `G3_G4/f125` | 36 | 0.8690 | 0.9516 | 0.9516 | 0.0686 |
| `J5_J6/f121` | 44 | 0.9076 | 0.9663 | 0.9663 | 0.0242 |
| `N1_N2/f113` | 32 | 0.7990 | 0.8730 | 0.8730 | 0.0322 |
| `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24/f125` | 22 | 0.3055 | 0.5738 | 0.5738 | 0.0337 |
| `P6253Z1G06N24_P6253Z2G10N24/f125` | 20 | 0.7263 | 0.8943 | 0.8943 | 0.0408 |
| `T6_T4/f121` | 147 | 0.8080 | 0.9083 | 0.9083 | 0.0367 |

가장 약한 cell 은 3-fresh-type `P6253…N20…/f125` (n=22, ρ 0.3055) 로, PRIOR 도 0.5738 로
낮은 구간이다 — head 특유의 실패가 아니라 그 cell 자체가 어렵다.
**캠페인 표적 cell `T6_T4/f121`(n=147)** 에서 head 는 ρ 0.8080 · MAE 0.0367 이다.

---

## 5. G4 — σ 커버리지, **FAIL**

| 항목 | 값 |
|---|---:|
| 68% 커버리지 | **0.831021** |
| 허용 구간 | [0.55, 0.80] |
| σ̄ | 0.119787 |
| σ median | (JSON `G4.sigma_median`) |

**과대(over-wide) 방향의 실패**다 — σ 가 실제 오차보다 넓어 83.1% 가 ±1σ 안에 들어온다
(정규 기준 68.3%). arm 2 도 재적합 후 0.827238 로 같은 방향으로 실패했다
(ORIGINAL path 에서는 0.634300 으로 PASS 였다).

**처분 영향 없음, 단 σ 서빙 금지.** §4/B.5/C.4 의 G4 조항은 "처분을 바꾸지 않되 σ 서빙
사용 금지"다. 따라서 승격 권고는 유지되며, **`predict_fxy` 의 σ 를 acquisition 의 UCB/LCB
폭이나 어떤 신뢰구간에도 쓰지 않는다.** `min_fxy` 목적함수가 `F_xy_UCB` 를 쓰므로
이는 실질적 제약이다 — 승격 시 orchestrator 가 **head σ 대신 기존 proxy σ 규약
(resid_sd 0.0476 · K 3.0)을 유지할지 별도로 결정**해야 한다.

---

## 6. 동반 판독 — FIXED path·재적합에서 서빙 proxy 자신의 값 (C.3 #2 이행)

C.3 #2 는 **판정에 쓰지 않되 반드시 함께 인쇄**하라고 등록했다.

| | ORIGINAL path (바가 측정된 곳) | **FIXED path + 재적합 (지금)** |
|---|---:|---:|
| `PROXY on s1i` MAE | 0.076721 (= 바 0.0767) | **0.073173** |
| `PROXY on s1i` ρ̄ | 0.726296 (= 바 0.7263) | **0.715696** |
| `PROXY on s1i` bias | +0.005270 | +0.001872 |
| `PROXY on s1i` 68% 커버리지 | — | 0.943253 |

**바는 이동하지 않았다.** 판정은 고정 상수 0.0767 / 0.7263 으로만 했다.
재측정 proxy 는 MAE 가 개선(0.0767 → 0.0732)되고 ρ̄ 는 소폭 하락(0.7263 → 0.7157)했는데,
**arm 3 은 두 기준 모두에서 이긴다** — 즉 "고정 바는 통과하지만 재측정 proxy 에는 패배"
라는 엇갈림은 **발생하지 않았다**.

---

## 7. 멤버 건전성

### 7.1 학습 완결성 — 후보는 완료본이다 (C.2 선행조건 충족)

`rc = 0`, `DONE` 존재, `member_20260716…20260720` 5개, per-cell 보정 5종 + flatness 가
pull 시점에 **모두 존재**. `--fxy-direct` 가 실제로 걸렸음을 학습 로그가 확인한다:

```
=== f_xy head: DIRECT mode — no prior composition; the row regresses ABSOLUTE f_xy.
    (Reported baseline prior f_xy = 1.2161*f_r -0.2488 on 5425 labelled train rows, resid sd=0.0478) ===
```

5개 멤버 meta 전부 `fxy_head.mode = "direct"`, `target_idx = 8`, `select_weight = 0.5`,
`n_labelled_train = 5425`, prior `a = 1.216147350612086` / `b = −0.24878433827113983`
(= 채점기의 `PRIOR_A`/`PRIOR_B` 와 **정확히 일치**).

### 7.2 per-member early stop (train.log)

| seed | early stop @ | best epoch | best-epoch `fxyMAE` | `fxyRho` | `sel` |
|---|---:|---:|---:|---:|---:|
| 20260716 | 47 | 32 | 0.0755 | 0.828 | 1.1111 |
| 20260717 | 45 | 30 | 0.0755 | 0.815 | 1.0863 |
| 20260718 | 90 | 75 | 0.0749 | 0.817 | 1.0971 |
| 20260719 | 35 | 20 | 0.0749 | 0.800 | 1.0905 |
| 20260720 | 55 | 40 | 0.0757 | 0.810 | 1.0248 |

다섯 멤버가 서로 다른 epoch 에서 멈췄고(20~75) `fxyMAE` 는 0.0749~0.0757 로 촘촘하다 —
**퇴화한 멤버 없음.** arm 2 대비(0.0752~0.0772) 분산이 더 작다.

### 7.3 leave-one-out / 단일 멤버 (같은 793행, 같은 11 cell)

| 구성 | MAE | ρ̄ | bias | 커버리지 |
|---|---:|---:|---:|---:|
| **전체 5-멤버** | **0.066300** | **0.790392** | −0.003137 | 0.8310 |
| drop 20260716 | 0.067236 | 0.780963 | −0.003638 | 0.8197 |
| drop 20260717 | 0.066110 | 0.791572 | −0.003013 | 0.8285 |
| drop 20260718 | 0.066443 | 0.788583 | −0.001932 | 0.8222 |
| drop 20260719 | 0.068061 | 0.784085 | −0.005016 | 0.8172 |
| drop 20260720 | 0.066270 | 0.775165 | −0.002086 | 0.8159 |
| 단일 20260716 | 0.075543 | 0.762104 | −0.001132 | 0.4565 |
| 단일 20260717 | 0.075494 | 0.751900 | −0.003634 | 0.6280 |
| 단일 20260718 | 0.074901 | 0.736972 | −0.007955 | 0.6406 |
| 단일 20260719 | 0.074885 | 0.731907 | +0.004379 | 0.4641 |
| 단일 20260720 | 0.075724 | 0.729407 | −0.007343 | 0.6847 |

- **어떤 LOO 조합도 G2′·G3′ 를 통과한다** (MAE 0.0661~0.0681 < 0.0767, ρ̄ 0.775~0.792 > 0.7263).
  판정이 한 멤버에 걸려 있지 않다.
- **단일 멤버조차 다섯 개 모두 두 바를 통과한다** (MAE 0.0749~0.0757, ρ̄ 0.729~0.762).
  앙상블 효과는 MAE −0.009 / ρ̄ +0.03 이며 판정의 근거가 아니다.
- 단일 멤버 커버리지가 0.46~0.68 인데 앙상블에서 0.83 으로 오른다 — G4 초과가
  **앙상블 epistemic 항의 과대 합산**에서 온다는 신호다(σ 서빙 금지의 근거를 보강한다).
- 학습 로그 best-epoch `fxyMAE`(0.0749~0.0757)와 서빙 단일-멤버 MAE(0.0749~0.0757)가
  **소수 4자리까지 일치**한다 — arm-2 결과보고서 §6.3 이 등록해 둔 train/serve 괴리가
  arm 3 에서도 닫혀 있음을 재확인한다.

---

## 8. arm 3 vs arm 2 정면 비교 (같은 793행·같은 11 cell, 둘 다 FIXED path + 재적합)

| | arm 2 (`--fxy-prior-on-predicted`) | **arm 3 (`--fxy-direct`)** |
|---|---:|---:|
| MAE | 0.066886 | **0.066300** |
| bias | **−0.000821** | −0.003137 |
| RMSE | 0.099840 | **0.099277** |
| ρ̄ (11 cell) | 0.785331 | **0.790392** |
| σ̄ | 0.120539 | 0.119787 |
| 68% 커버리지 | 0.827238 | 0.831021 |
| G2′ | PASS | **PASS** |
| G3′ | PASS | **PASS** |
| G4 | FAIL | **FAIL** |

paired BCa (cell-clustered, arm = arm 3, control = arm 2, 11 cell):
**point +0.005061, CI [−0.004133, +0.015511] → CI 가 0 을 포함 → tie.**

> **읽는 법.** arm 3 이 MAE·RMSE·ρ̄ 에서 근소하게 앞서지만 **통계적으로 구분되지 않는다.**
> B.3 #2 가 예고한 대로 "prior + 선형잔차"와 "direct"는 동결 trunk 위에서 **같은 가설공간**
> 이며, 이 실측이 그 예고를 확인한다. **arm 3 의 가치는 arm 2 를 이긴 것이 아니라,
> 합성 채널을 제거해도 성능이 유지된다는 것**(= 구조적으로 더 단순한 쪽이 동등)이다.
> arm 2 의 ORIGINAL-path FAIL 은 featurization 결함 탓이었고 arm 3 은 그 채널 자체가 없다.

---

## 9. 고정 자(尺) — legacy 축, 후보 vs `s1i` (같은 793행, **양쪽 재적합**)

| 축 | n | MAE `s1i` | MAE arm 3 | ρ_global `s1i` | ρ_global arm 3 | ρ_cellmean `s1i` | ρ_cellmean arm 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cyclen | 793 | 1.621244 | **1.619684** | 0.9962 | 0.9962 | 0.8760 | 0.8760 |
| F_r | 793 | 0.049914 | **0.049132** | 0.9725 | **0.9728** | 0.7868 | **0.7892** |
| F_q | 793 | 0.075211 | **0.074112** | 0.9703 | **0.9706** | 0.7994 | **0.8019** |
| CBC_max | 793 | 12.479430 | **11.994568** | 0.9852 | 0.9852 | 0.8622 | **0.8762** |
| node_peak | 790 | 0.046648 | **0.044112** | 0.9714 | **0.9740** | — | — |
| map_cov | 790 | 0.010791 | **0.010186** | 0.9872 | **0.9880** | — | — |

**여섯 축 전부에서 후보가 챔피언과 같거나 낫다.** 회귀 축 없음.
cyclen 이 사실상 동일한 것은 `--freeze-trunk-cyclen` 의 정의대로다(head 가 byte-identical,
보정도 챔피언 것의 복사본).

### 9.1 C.3 #3 이 등록한 비대칭 — **해소되었다**

C.3 #3 은 "후보는 자기 보정을 적합하지만 `s1i` 는 ORIGINAL path 보정을 쓴다"는 비대칭을
결과보고서에 적으라고 등록했다. **본 채점에서는 그 비대칭이 존재하지 않는다** —
`s1i`·arm 2·arm 3 세 dir 모두 FIXED path 에서 재적합된 보정을 지닌 채 채점되었다
(`servefix_calibration_refit_20260829.md` §3·§5). 재적합 전 상태로 채점했다면 §9 의
`s1i` 열은 F_r 0.102616 / cyclen 2.400414 / CBC 16.607514 였을 것이며, 그 비교는
모델이 아니라 보정의 낡음을 재는 것이 되었을 것이다.

---

## 10. 처분 (C.4 처분 규칙의 적용 — 실행은 orchestrator)

측정 결과는 **G1 PASS · G2′ PASS · G3′ PASS · G4 FAIL** 이다.

1. **`s1j` 승격을 권고한다.** C.4: "G1 PASS · G2′ PASS · G3′ PASS → head 를 f_xy 서빙
   경로로 승격하고 체크포인트를 `s1j` 로 승격, deck `model_dir` 갱신. proxy 는 fallback
   으로 남는다." **본 문서는 그 편집을 실행하지 않았다** — 목록은 §10.1.
2. **G4 FAIL → head σ 서빙 금지.** 처분은 바뀌지 않으나 `predict_fxy` 의 σ 를
   acquisition 폭·신뢰구간에 쓰지 않는다. `min_fxy` 가 `F_xy_UCB` 를 쓰므로,
   승격 시 **UCB 폭을 어디서 얻을지**를 orchestrator 가 명시적으로 정해야 한다
   (기존 proxy σ 규약 resid_sd 0.0476 · K 3.0 유지가 보수적 기본값이다).
3. **arm 3 의 결과로 어떤 후속 arm 도 자동 발동하지 않는다** (C.4). arm 4 는 별도 개정을 요구한다.
4. **GPU 1 편차는 미해결 기록 사항이다.** `run.sh` 가 `CUDA_VISIBLE_DEVICES=1`,
   deck 이 `lpopt_gpu1.inp` 인데 이는 `lpopt.inp` 의 상시 지시(*"사용자 지시 2026-07-24:
   GPU 0 고정, 재허가 전까지 auto 금지"*)와 충돌한다. B.6/C.2 는 orchestrator 가
   재허가를 확인·기록한 뒤에만 실행하라고 등록했다. **채점자는 재허가 유무를 확인할 수
   없다.** G1~G4 어떤 수치에도 영향이 없다.
5. **arm 1·arm 2 의 판정은 불변이다.** 두 결과보고서는 그대로 둔다.
   `fxy_gate_eval_arm2c_20260829.json` 은 **arm-2c 판독(정보 제공용)**이며 재심이 아니다(§11).

### 10.1 승격 시 orchestrator 가 해야 할 **아홉 가지 편집** (열거만 — 본 문서는 실행하지 않았다)

| # | 편집 | 상세 |
|---|---|---|
| 1 | 디렉터리 개명 | `data/models/20260829_194532` → `data/models/s1j` |
| 2 | `lpopt.inp` | 41행 `model_dir = "data/models/s1j"` + 주석을 s1j 계보로 갱신. `cond_schema` 는 후보 meta 가 **v8** 이므로 **변경 없음** |
| 3 | `lpopt_gpu1.inp` | 40행 `model_dir` 및 주석 갱신 (같은 챔피언을 가리켜야 한다) |
| 4 | 캠페인 deck | `fpcamp_minfxy_T6T4_f121_r1_199.inp` **156행 `model_dir` 한 줄만** → `data/models/s1j`. 다른 knob 금지 (deck 자체 "LAUNCH-TIME MODEL SUBSTITUTION" 절, minfxy prereg §5.3) |
| 5 | deck 재해시 | 현재 `4AF8B0218666EC83E0E9357C6FB268F179301EE30BEC9CD1CAA89C3144C2FAC5` (16,957 B) → 치환 후 새 sha256 산출 |
| 6 | launcher | `launch_fpcamp_minfxy_T6T4_f121_r1_199.ps1` **38행 `$modelName = 's1j'`**, **63행 `$want = <새 deck sha>`** |
| 7 | prereg stamp | `data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md` **§9.1** 의 "launch-time 치환 시 (s1j)" 두 TBD 행에 새 deck sha256 과 `data/models/s1j/member_20260716/meta.json` sha256 (**현재 `f0af69c0f54261dec61f253e3828fdc5df742f0915440678c885b96bb4112e7b`**) 을 stamp |
| 8 | head prereg | `data/reports/fxy_head_prereg_20260829.md` **C.6** 판정표에 본 문서 §1 의 측정값 기입 |
| 9 | `fit_map_calibration` 재적합 | **계보는 승격 시 이것을 하지 않았다** — `data/store/map_calibration.json` 의 `fit.model_id = "split_S1b"`, `fitted_at = 2026-08-10T12:37`, `n_cells_fitted = 2` 로 s1h·s1i 승격을 모두 건너뛰었다. 따라서 s1j 승격 시에도 **계보 준수만으로는 의무가 아니다.** 이 아티팩트가 4세대 낡았다는 사실만 별도 항목으로 남긴다 |

> **승격과 무관하게 지금 유효한 선행 차단 두 가지.**
> (i) **launcher store gate.** `$wantStore = '0334E2D2E303CD8E82373861603F82A57054FC2FC6139F84C360822580644D9D'`
> 는 e_core backfill 이전 값이다. 현재 store 는
> `F38666E9F1508D35D33E0C22F583C5479C6F09CAC748201B494B47C8CFECA6EA` (75,793행) 이므로
> 지금 launcher 를 돌리면 `MINFXY1 REFUSED: store sha256 mismatch` 로 거부된다.
> minfxy prereg §9.1 store 행과 `$wantStore` 를 함께 갱신해야 한다.
> (ii) **보정·코드 동반 배포.** 승격된 dir 을 HOST_238/HOST_199 로 보낼 때
> `servefix_calibration_refit_20260829.md` §6 의 규칙(특히 `ensure_checkpoint` 지문이
> per-cell 보정을 해싱하지 않아 **push 를 건너뛴다**는 점)을 반드시 적용한다.

---

## 11. arm-2c 판독 (정보 제공용 — **재심 아님**)

C.3 #1 은 수정된 경로에서의 arm-2 재채점을 **정보 제공용**으로만 남기라고 등록했다.
`fxy_gate_eval_arm2c_20260829.json` (sha `395005f00119ead9599115dca8c1c019bf3fb529fa842c721670afaddc0093d4`)
는 그 arm-2b 판독을 **양쪽 보정 재적합 후**로 갱신한 것이다.

| 항목 | arm 2 확정 판정 (ORIGINAL, 재적합 전) | arm 2b (FIXED, 재적합 전) | **arm 2c (FIXED, 재적합 후)** |
|---|---:|---:|---:|
| **G1** | PASS | PASS | **PASS** (worst 0.011029, ε 0.13881/0.14216) |
| **G2′** MAE < 0.0767 | 0.108560 **FAIL** | 0.066886 PASS | **0.066886 PASS** |
| **G3′** ρ̄ > 0.7263 | 0.768111 PASS | 0.785331 PASS | **0.785331 PASS** |
| **G4** ∈ [0.55, 0.80] | 0.634300 PASS | 0.827238 FAIL | **0.827238 FAIL** |
| head bias | −0.088948 | −0.000821 | −0.000821 |
| (동반) PROXY on `s1i` MAE | 0.076721 | 0.132102 | **0.073173** |
| (동반) PROXY on `s1i` ρ̄ | 0.726296 | 0.714088 | **0.715696** |
| (고정 자) `s1i` 보정 F_r MAE | 0.052678 | 0.102616 | **0.049914** |
| (고정 자) `s1i` cyclen MAE | 1.852122 | 2.400414 | **1.621244** |
| (고정 자) `s1i` CBC_max MAE | 14.210126 | 16.607513 | **12.479430** |

> **읽는 법.** head 관련 수치(G2′/G3′/G4/bias)는 arm 2b 와 **완전히 같다** — 재적합은
> per-cell 보정을 바꿀 뿐 `predict_fxy` 의 head 출력을 바꾸지 않기 때문이다. 움직인 것은
> **동반 판독과 고정 자 다섯 줄**이며, arm 2b 에서 "가짜 악화"로 등록해 둔 그 다섯 줄이
> 재적합으로 **ORIGINAL 값보다도 좋은 자리로 돌아왔다.** 이것이 arm-2b 결과보고서의
> 진단("아핀 보정은 단조라 순위를 못 바꾸고 레벨만 바꾼다 — 재적합이 필요하다")에 대한
> 실측 확인이다.
>
> **arm 2 의 확정 판정(G2′ FAIL → 미승격)은 그대로다.** 본 표는 재심이 아니며
> arm 2 후보를 승격 대상으로 되살리지 않는다. arm 3 이 별도 개정(Amendment C)에 따라
> 채점된 유일한 구속력 있는 후보다.

---

## 12. 출처와 해시 (provenance)

| 항목 | 값 |
|---|---|
| store `data/store/records.parquet` | `f38666e9f1508d35d33e0c22f583c5479c6f09cac748201b494b47c8cfeca6ea` · 75,793행 |
| 채점 슬라이스 | S1j VAL · converged · `f_xy` notna → **793행** (재적합 중 store 변동에도 불변, `servefix…md` §2.1) |
| G3′ cell 집합 | 11 cell / 486행 (≥ 20 라벨) |
| `lpopt/model/model_api.py` | `94229de9e332c7faa66529f51b03d107f20098b1758f999b9c73ec8cfb21e6a2` (110,976 B) — FIXED |
| `lpopt/model/featurize.py` | `6977344dafbd770c9b1bc40e370db6c189320e301f8fa49570a25f927b575e36` (86,245 B) — FIXED |
| parity gate | `tests/test_model_api.py::test_serve_row_featurization_parity` → **1 passed** |
| 후보 `member_20260716/meta.json` | `f0af69c0f54261dec61f253e3828fdc5df742f0915440678c885b96bb4112e7b` |
| 채점기 arm 3 | `data/reports/fxy_gate_eval_arm3_20260829.py` `c191141a2325e0f068cba3ea1dee119e49aa829ace594d318e064b740ddbbc12` (16,199 B) |
| 채점기 arm 2c | `data/reports/fxy_gate_eval_arm2c_20260829.py` `9eea5a43f35488c08ef469677f398fe04caa8bb7e920b5c55c67fb8fb8689670` (16,043 B) |
| 결과 JSON arm 3 | `data/reports/fxy_gate_eval_arm3_20260829.json` `f70136d11f17097c281e86024acd31fe4622ff089178c0728659d8ff3f578a4e` |
| 결과 JSON arm 2c | `data/reports/fxy_gate_eval_arm2c_20260829.json` `395005f00119ead9599115dca8c1c019bf3fb529fa842c721670afaddc0093d4` |
| G1 JSON arm 3 | `data/reports/gate_fxy_arm3_20260829_checkonly.json` `371f0e5a589023d5156da0954a54825375fb41e39f9651b0875f0b2e54be948f` |
| G1 JSON arm 2c | `data/reports/gate_fxy_arm2c_20260829_checkonly.json` `ab3132ce996ae17b3e3f95c26ae96eb7f3737e7f833d7189737408346aa500af` |
| 후보 per-cell 보정 6종 sha256 | `servefix_calibration_refit_20260829.md` §5 |

### 12.1 재현

```
python -m lpopt gate-promote --input lpopt.inp --prev data/models/s1i \
    --new data/models/20260829_194532 \
    --out data/reports/gate_fxy_arm3_20260829_checkonly.json --check-only
python data/reports/fxy_gate_eval_arm3_20260829.py    # -> fxy_gate_eval_arm3_20260829.json
```

본 채점은 **코드를 수정하지 않았고**, deck·`state.json`·챔피언 디렉터리 구조·후보
디렉터리 구조·arm-1/arm-2 하네스를 **건드리지 않았다.** 새로 만들거나 갱신한 파일은
세 model dir 의 per-cell 보정(+`.bak_pre_servefix_20260829` 백업), 위 네 개 JSON,
채점 스크립트 `fxy_gate_eval_arm3_20260829.py`, 그리고 본 문서와
`servefix_calibration_refit_20260829.md` 뿐이다.

---

## 13. 사전등록 §8 이 금지한 주장 — 본 문서에서의 준수

- **"direct 가 편향을 없앴다"고 주장하지 않는다.** C.1 이 등록한 대로 그것은 가설이었다.
  실측은 head bias **−0.003137** 로 작지만 0 이 아니며, arm 2(재적합 후, −0.000821)보다
  **오히려 크다.** 합성 채널 제거가 편향을 0 으로 만든 것이 아니라, 두 구성 모두
  FIXED path 위에서는 편향이 무시할 수준이라는 것이 실측이다.
- **"arm 3 이 arm 2 보다 낫다"고 주장하지 않는다.** paired CI 가 0 을 포함한다(§8) — **tie** 다.
- **"head 가 proxy 를 대체해야 한다"를 G4 근거 없이 주장하지 않는다.** σ 는 서빙 금지다(§5).
- **A.5/A.6 바(0.0463 / 0.8944)를 통과했다고 주장하지 않는다** — 둘 다 FAIL 이며
  §1 에 그대로 인쇄했다. 이 두 바는 폐기되지 않았고 승격 조건도 아니다(C.4).
- **바를 사후에 옮기지 않았다.** 0.0767 / 0.7263 은 ORIGINAL path 에서 측정된 상수 그대로
  쓰였고, 재측정 proxy(0.0732 / 0.7157)는 동반 판독으로만 인쇄했다.
