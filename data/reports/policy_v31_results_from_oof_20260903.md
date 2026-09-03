# policy v3.1 게이트 결과 — 저장된 OOF 로짓에서 재계산

> **예비 — 181 복제 런으로 확정 예정.**
> 재학습 없음. 238의 `runs/policy_v31/` 크로스핏 런이 5 블록 × 3 시드(stage 1+2,
> λ∈{0, 0.3, 1.0}) 학습을 끝내고 `logits_oof.npz`를 쓴 뒤 `gate_report_v31`
> 안에서(pandas-3 boolean `fillna`, `metrics_v31.py:718`) 죽었다. 그 결함은
> 수정되었고(local↔238 sha256 `98ca662f…` 동일), 게이트는 가중치를 하나도 읽지
> 않으므로 저장된 로짓만으로 **정확히** 재계산된다.

## 0. 재계산이 정확한 이유 · 입력

`train_v3.main()`의 크로스핏 분기는

```
report = m31.gate_report_v31(steps, splits, xf["logits_fxy"], v2=v2,
                             gate_auc=GATE_AUC, gate_auc_ci_lo=GATE_AUC_CI_LO,
                             transfer_auc=TRANSFER_AUC, seed=args.base_seed)
```

만 호출하고, `xf["logits_fxy"]`는 `train_crossfit_v31`이 반환 직전에
`np.savez_compressed(out_dir/"logits_oof.npz", fxy=oof, fold=fold_col)`로 쓴
**바로 그 배열**이다. 따라서 파일을 그 자리에 넣는 것은 근사가 아니라 동일 입력이다.
`--report-only`/resume 경로는 train_v3.py에 없어 독립 스크립트
`~/lpopt_ws/scratch/v31_report_from_oof.py`로 같은 호출을 재현했다.

| 입력 | sha256 | 등록값 대조 |
|---|---|---|
| `data/policy/steps_v31.parquet` | `ed74a6b4…f589ad` | `STEPS_V31_SHA256` 일치 |
| `data/policy/v31_split/splits_v31.csv` | `a62c9937…c951e34` | `SPLITS_V31_SHA256` 일치 |
| `data/design/policy_v3_v2_baseline.csv` | `0cf2416d…a902660` | 블라인드 v2 CSV |
| `runs/policy_v31/logits_oof.npz` | 21,215행 중 2,663 유한(pool 2,503 + f109 160), 나머지 NaN = train/val (판정 대상 아님) | 저장된 `fold` 열이 split 파일 `fold`와 완전 일치 |

`base_seed = 20260903`(멤버 시드 20260903–05).
출력: `runs/policy_v31/metrics_v31_from_oof.json`.

## 1. λ 아암 — 게이트는 한 아암에만 존재한다

§2c/§9d는 라운드 전체에 λ **하나**를 `val` 평균 Spearman만으로 고른다. 따라서
`logits_oof.npz`에는 **선택된 λ의 OOF 로짓만** 스티치되어 있고, 나머지 두 아암의
stage-2 가중치는 저장되지 않았다. **λ별 게이트 표는 복구 불가이며 애초에 등록되지도
않았다.** 남는 것은 선택 통계뿐이다:

| λ | val 평균 ρ (5블록×3시드) | 선택 |
|---:|---:|---|
| 0.0 | +0.4082 | |
| **0.3** | **+0.4167** | **선택** (동률 시 작은 λ) |
| 1.0 | +0.4167 | |

(stage 1 평균 +0.4070.) λ=0.3 과 1.0 이 4자리에서 동률이라 `select_stage2_lambda`의
"동률은 작은 λ" 규칙이 실제로 발동했다. **λ=0 은 선택되지 않았으므로 §6d의
"λ=0 선택 + 게이트 PASS → listwise 라인 종결" 행은 발동하지 않는다.**

## 2. §6a 절별 판정 (λ=0.3, pool 판정행 39 live parent)

| 절 | 등록 바 | 관측 | 판정 |
|---|---|---|---|
| **1** parent-blocked AUC | ≥ 0.65 이고 95% CI 하한 > 0.50 | **0.7215** [0.6510, 0.7794], mixed parent 85 | **PASS** |
| **2A** NDCG@4-of-8 vs 적합 베이스라인 4 | 넷 모두 우월 + 각 paired CI 가 0 배제 | random +0.2242 [+0.0635, +0.3841] n80 34.2 · periph +0.2417 [+0.1124, +0.3670] n80 18.0 · gd_rule +0.2450 [+0.1051, +0.3777] n80 20.9 · **class_freq +0.0811 [−0.0496, +0.2126] n80 166.8** | **FAIL** (class_freq CI 0 미배제) |
| **2B** policy_v2 비열등 δ=0.05 단측95% | 하한 > −0.05 | Δ **+0.1257**, sd 0.3516, 하한 **+0.0331**, n=39 live(정보 34), **실현 검정력 0.93** | **PASS** |
| **2C** policy_v2 우월 | 게이트 아님(보고만) | 절 1 per-scorer: policy 0.7215 vs policy_v2 0.5822; NDCG paired Δ +0.1093 [+0.0330, +0.1805] | 보고 |
| **3** within-cell | 자격 셀(live≥10 ∧ rankable)마다 class_freq·gd_rule 우월, policy_v2 대비 δ 이상 패배 없음 | `E1_E2/f121` 10 live, rankable(class_freq p=0.0035) → **FAIL** (class_freq Δ −0.0398 [−0.2442, +0.1704]) · `N1_N2/f113` 11 live, rankable(gd_rule p=0.0245) → **FAIL** (gd_rule Δ −0.0252, class_freq CI 0 미배제) · 나머지 9 셀 live<10 → 판정 불가 | **FAIL** |
| **4** 서빙 스케일 | Platt 서빙 p90−p10 ≥ 0.15 | **0.1065** (raw 0.0773, logit 5.045; Platt a=0.3155 b=−1.0519, calib n=2,503 base 0.0887) | **FAIL** |
| **5** fr/flat 무회귀 | 구조적 (`assert_stage2_init_is_stage1`) | 폴드 통계 아님 — 체크포인트 수준에서 학습 중 단언됨 | 구조적 PASS |
| 셀 편중 | 한 셀 ≤ 40% | 최대 `N1_N2/f113` 12/39 = **28.2%** | 무효화 없음 |

**GATE = FAIL** (2A ∧ 3 ∧ 4 미충족).

보고 전용: **ECE 0.1004 · Brier 0.1567** (§5c대로 게이트 아님; 등록 사유대로
0.05를 이미 넘는다). 측정 MDE(pool 39 parent): 0.1244(80%검정력) / 0.0980(screen z=1.96),
관측 NDCG 0.6131, 순열 p=0.0005.

## 3. §6b 전이 바 — `E1_E2/f109/ga80` (160행 · 16 live)

| 항목 | 바 | 관측 | 판정 |
|---|---|---|---|
| parent-blocked AUC | ≥ 0.60 | **0.8939** | PASS |
| NDCG@4 vs random | CI 0 배제 | +0.5342 [+0.3199, +0.7354] | PASS |
| vs periph | CI 0 배제 | +0.5916 [+0.5019, +0.6751] | PASS |
| vs class_freq | CI 0 배제 | +0.3481 [+0.2660, +0.4287] | PASS |
| **vs gd_rule** | CI 0 배제 | **+0.0925 [−0.0197, +0.2173]** | **FAIL** |
| vs policy_v2 | 보고만 | Δ +0.1178, 하한 −0.0323, 검정력 0.58 | 보고 |

**전이 = FAIL** — pb-AUC는 크게 통과하나 gd_rule 대비 CI가 0을 배제하지 못한다.
policy NDCG 0.8710 vs gd_rule 0.7785. 셀 MDE 0.2029(80%) / 0.1599(screen) —
사전등록 표의 0.160은 screen 관례이며 그 값과 일치한다(수렴 확인).

## 4. §6d 처분

* 절 2B는 **검정력 0.93에서 PASS** 다. 사전등록이 가장 걱정한 "70% 검정력 시험"은
  실제로는 39 live · Δ +0.126 · sd 0.352 로 훨씬 유리하게 실현되었다. 즉
  **v3.1이 policy_v2보다 나쁘다는 증거는 없다**(우월은 §6c대로 주장하지 않는다).
* 그러나 **절 3이 두 자격 셀 모두에서 FAIL** 이므로 r2 판정 조항이 발동한다 —
  **어떤 새 랭커도 wave 를 랭크하지 않는다.** `gd_rule` / `burnt_absmov_r` 은
  생성기 prior 후보로만 남는다.
* **절 4 FAIL**(폭 0.107 < 0.15)이므로 τ_v3.1 을 유도할 수 없고 A/B/C 는 차단이다.
  이는 랭킹 문제가 아니라 서빙 스케일 문제이며, 다음 라운드는 목표 재정의
  (parent 내 z-정규화 / 순수 listwise)로 간다.
* 절 2A FAIL 은 "회귀 방지 절"의 실패다. v3 실측(live 23)에서 class_freq Δ
  +0.2846 [+0.1523, +0.4101] 이던 것이 크로스핏 pool(live 39)에서 +0.0811
  [−0.0496, +0.2126] 로 내려앉았다. **평가 폴드가 바뀐 것과 모델이 바뀐 것을
  이 런만으로는 분리할 수 없다** — 181 복제 런과 §9d arm ii 대조가 필요하다.

## 5. 유보 사항

1. 이 수치는 **λ=0.3 아암 단일**이다. λ=0 / 1.0 의 게이트는 재학습 없이는 없다.
2. 절 5는 여기서 계산되지 않는다(체크포인트 단언). 크래시가 게이트 리포트
   안에서 났으므로 stage-1 비트동일성 단언 자체는 통과한 뒤였다.
3. 순열 통계(rankability·MDE)는 `PERM_SEED`/`seed=20260903`, 2,000 reps 고정이라
   재실행 시 재현되지만, 부트스트랩 CI는 4,000 draw의 표본오차를 갖는다.
4. 베이스라인 `policy_v2` 는 §3b의 **선언된 편차**(크로스핏 pool 위에서 재발행된
   블라인드 CSV)에 의존한다. 그 CSV는 v3.1 가중치가 존재하기 전에 쓰였다.
5. **181 복제 런의 수치와 대조하기 전까지 확정으로 인용하지 않는다.**
