# Policy net v3.1 — pre-registration (DRAFT)

작성 2026-09-03. **DRAFT 이며 아직 등록본이 아니다.** 등록본이 되는 조건은 셋이다:
(i) 아래 fingerprint 표의 빈칸이 STEP 0-a 의 명령으로 채워지고, (ii) STEP 0-b 의
사전-런 census 6종이 계산되어 부록으로 해시되고, (iii) **v3.1 가중치가 하나도
존재하지 않는 시점**에 그 상태로 동결되는 것. 그 이후 이 문서는 편집하지 않으며
모든 이탈은 `data/reports/policy_v31_results_<date>.md` 에 적는다.

| artefact | fingerprint |
|---|---|
| corpus `data/policy/steps_v31.parquet` | STEP 0-a 에서 기록 (예상 **28,989행 × 111열**; v3 의 28,889×107 + r2 캠페인 100행 + 신규 4열, §4d) |
| 직전 corpus `data/policy/steps_v3.parquet` | SHA-256 `100ee50ed5c75725…`, 10,486,078 B, 28,889행 × 107열 — **바이트 보존, 재-mine 하지 않는다** |
| store 스냅샷 | `~/lpopt_ws/scratch/records_r2_76793.parquet`, 76,793행, `22854B72…` (2026-09-02 20:33 로컬 store) |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91d…`, 64,343 B, 194행 × 55열 |
| 기존 코드 (v3.1 이전 상태) | `lpopt/policy/v3.py` · `train_v3.py` · `scorer.py` · `lpopt/search/construct.py` · `mine_policy_corpus.py` — STEP 0-a 에서 해시 |
| 선행 문서 | `policy_v3_prereg_20260831.md` · `policy_v3_results_20260831.md` · `minfxy_E1E2_f121_r2_results_20260831.md`(§6.4 RANK gate · §8 NULL 귀속 · §11.2) · `intervention_wave_r1_results_20260830.md` |
| 병행 문서 | f_xy **head**(s1i/s1j/arm4 서로게이트)의 G1–G4 within-cell ranking 조항은 별도 문서에서 등록된다. **본 문서는 그 파일에 의존하지 않는다** — 여기서 등록하는 것은 policy net(move ranker)의 within-cell 조항이며 두 조항은 대상도 폴드도 다르다(§6a-절3) |

**이 문서의 출처.** 세 개의 v3.1 제안이 심사되었고 점수는 listwise 6.5 ·
descriptors 6.5 · transfer 6.25 였다. 본 문서는 **listwise 제안을 골격으로** 하고
(target/loss · cross-fit · 스케일 회복 · 등록 gain 위의 게이트), descriptors
제안의 **측정된 부정 결과**(비보존 Gd 계열 기각, burnt 부격자 결손, regret 포화)와
transfer 제안의 **전이 셀 재선정·라벨 계획 산술**을 흡수한다. 세 제안이 심사에서
지적받은 결함 — 비열등 검정력 산술 오류, 파일럿 없는 기전, 결과를 보고 고른 두 열,
n=75 에서 불가능한 TOST, 게이트 폴드를 보고 정한 임계 — 은 §2·§4·§6 에서
**명시적으로 정정**했고 정정 사실 자체를 등록한다.

---

## 0. 왜 v3.1 인가 — 세 개의 측정이 v3 판정을 판정 불가로 만들었다

1. **v3 는 진 것이 아니라 갈리지 못했다.** GATE 절 2(`regret@4-of-8`)에서
   `class_freq`·`policy_v2` 상대 점추정은 v3 가 4.2배·2.7배 낮았고 CI 만 0 을
   배제하지 못했다(results §0). 필요 parent 는 raw gain 에서 51·77(재계산 52·78),
   보유 38.
2. **등록 gain 으로 다시 재면 그 지표는 등록된 k 에서 아예 축퇴한다.** 구현 편차
   §1.6 은 지표 gain 을 `y_fxy` 가 아니라 raw `−d_f_xy` 로 계산했다. 등록 gain
   으로 재계산하면 `regret@4-of-8` 의 v3-vs-`policy_v2` **정보 parent 는 38 중
   2 개**이고 점추정 부호가 **음수(−0.0074)** 로 뒤집힌다
   (`_wf_scratch/wf_power_gate_registered.json` key `"4"`). 38 parent 중 34 개가
   양쪽 다 0 인 동점이므로(results §3.1) **지표 자체가 해상도를 잃었다.**
3. **라벨 계획의 전제가 틀렸다.** results §5-R1 은 "r2 wave 1개면 `class_freq` 절이
   결판난다"고 적었다. 측정하면 **탐색 캠페인은 게이트 parent 를 하나도 만들지
   못한다**: `fpcamp_minfxy_e1e2_f121_r2` 100행 = 45 parent · 최대 fan-out 6 ·
   **≥8후보 parent 0개**. 코퍼스의 28개 캠페인 중 ≥8후보 parent 를 만든 탐색
   캠페인은 `v520_minfr_b1` 하나(19-child parent 1개)뿐이다
   (`_wf_scratch/wf_campaign_yield.csv`). 게이트 parent 를 만드는 것은 **개입
   wave 뿐**이다(160행 = 20 parent × 8후보).

여기에 r2 판정이 하나를 더 얹었다. `minfxy_E1E2_f121_r2_results` §6.4 의 RANK gate
3/3 FAIL 과 §11.2 의 기계적 귀결: **"head 승격 기준에 within-cell ranking 조항을
추가하기 전까지 어떤 새 랭커도 wave 를 랭크하지 않는다."** policy net 은 그 판정의
대상 head 는 아니지만, 같은 규율을 policy net 게이트에도 건다(§6a-절3).

**시험하는 것은 하나다: 등록 gain 위에서 판정 가능한 게이트를 만들고, 그 위에서
v3.1 이 (a) 적합 베이스라인 넷을 이기고, (b) `policy_v2` 에 비열등하며, (c) 그
우열이 셀 안에서도 성립하고, (d) 사전 등록된 난이도를 만족하는 셀로 전이되는가.
"`policy_v2` 를 이긴다"는 이 라운드에서 게이트하지 않는다 — 그 이유는 §6a-절2C 에
검정력 산술과 함께 등록한다.**

---

## 1. 물려받는 측정 — 전부 본 문서 작성 시점에 계산되어 있다

### 1a. 등록 gain vs raw gain — 같은 앙상블, 같은 폴드, 다른 결론

`gate_cur`(1,137행 · 540 F_xy 라벨 · 38 개의 ≥8후보 parent), `runs/policy_v3/probs.npz`
5-seed 확률평균, 4,000 draw paired parent-bootstrap. Δ = baseline − v3(양수 = v3 우위).

| gain | k | vs `class_freq` | 정보 parent | vs `policy_v2` | 정보 parent | n80(v2) |
|---|---:|---|---:|---|---:|---:|
| raw `−d_f_xy` | 4 | +0.00361 | 10/38 | +0.00183 | 9/38 | 159 |
| **등록 `y_fxy`** | **4** | +0.02682 | 8/38 | **−0.00737** | **2/38** | — (부호 음) |
| 등록 `y_fxy` | 3 | +0.04945 | 10/38 | −0.00789 | 2/38 | — (부호 음) |
| 등록 `y_fxy` | 2 | +0.05549 | 6/38 | +0.00776 | 9/38 | 4,534 |
| 등록 `y_fxy` | 1 | +0.13628 | 18/38 | +0.02842 | 8/38 | 349 |

출처 `_wf_scratch/wf_power_gate_{registered,raw}.json` · `p31g_eval.csv`.
**부호가 k 에 의존한다**(k=1 +0.0284, k=2 +0.0078, k=3 −0.0079, k=4 −0.0074).
따라서 **k 를 결과를 보고 고르는 것은 이 라운드에서 금지된다**(§5b).

**경고 — 다음 라운드가 다시 밟지 않도록 등록한다.**
`_wf_scratch/v31nc_metric_power.csv` 의 행 라벨 `"regret@4-of-8 (registered)"` 는
**오라벨이며 실제 내용은 raw gain** 이다(그 파일의 v3 값 0.001108 = raw 값; 등록
gain 값은 0.009298). 그 파일에서 인용된 `need_n` 53/78 은 raw gain 수치다. v3.1 의
어떤 표도 이 파일을 등록 gain 의 근거로 인용해서는 안 된다.

### 1b. `regret@4-of-8` 은 포화되었다 — 소비자 지표를 게이트로 쓸 수 없다

등록 gain 에서 38 parent 중 v3 의 regret > 0 인 parent 는 2개(둘 다 HGD569/f125),
`policy_v2` 상대 차이가 0 이 아닌 parent 도 2개다. 그 위에서 paired CI 를 여는 것은
사실상 n=2 의 검정이다. **결정: `regret@4-of-8`(등록 gain)은 보고하되 게이트하지
않는다.** 대신 within-parent 순위 정보를 전부 쓰는 **NDCG@4-of-8(등록 gain)** 과
**gain-가중 within-parent 일치도(wPBC)** 를 게이트한다.

이 교체는 "v3 가 이기는 지표를 골랐다"가 아니다 — 교체 사유는 **정보 parent
수(2/38)** 라는 구조 사실이고, 교체 후에도 **k=4 는 그대로 유지된다**(§5b). 그리고
교체된 지표에서 v3 가 `policy_v2` 를 이기지 못한다는 사실도 같이 등록되어
있다(§6a-절2B, Δ +0.0423, CI 가 0 을 포함).

### 1c. cell · parent census (현행 era, ≥8후보 parent / 그중 **live**)

live = 그 parent 의 후보 중 하나 이상이 `y_fxy > 0` 인 parent (= 등록 gain 에서
순위지을 것이 실제로 있는 parent). dead parent 는 모든 채점자에게 항등적으로 0 을
주므로 paired 검정에 기여하지 않는다.

| cell | F_xy 행 | ≥8후보 | **live** | live율 | \|d_f_xy\| p50 | parent-max gain 중앙(clip 단위) |
|---|---:|---:|---:|---:|---:|---:|
| `E1_E2/f109/ga80` | 160 | 20 | **16** | **0.80** | 0.0784 | **0.239** |
| `E1_E2/f121/ga80` | 339 | 22 | 10 | 0.45 | 0.0455 | 0.000 |
| `N1_N2/f113/ga80` | 166 | 19 | 12 | 0.63 | **0.0213** | 0.078 |
| `T6_T4/f121/paramA` | 260 | 17 | 8 | 0.47 | 0.0818 | 0.000 |
| HGD569 `f125/paramA` | 152 | 15 | 8 | 0.53 | 0.0357 | 0.005 |
| `G3_G4/f125/ga80` | 98 | 3 | 1 | — | 0.1924 | 0.000 |
| `E1_E2/f125/ga80` | 67 | 3 | 2 | — | 0.1598 | 0.658 |
| **합** | 1,242 | **99** | **57** | 0.58 | | |

출처 `_wf_scratch/wf_cell_difficulty.csv` · `wf_resolving_by_cell.csv` ·
`nc2_E9_parents.csv`. 폴드별(현 f113 홀드아웃): `gate_cur` 38/23 · `train` 36/18 ·
`val` 6/4 · `prospective_cell` 19/12.

**live 는 k 에 의존하지 않는다** — `res_y_k1 = res_y_k2 = res_y_k3 = res_y_k4` 가
모든 셀에서 성립한다(`wf_resolving_by_cell.csv`). 즉 §1b 의 포화는 k 선택 문제가
아니라 **등록 gain 이 만드는 양성 희소성**이다(게이트 폴드 540행 중 `y_fxy>0` 65행).

### 1d. 캠페인 수율 — 어떤 호출이 게이트 parent 를 사는가

| 종류 | 대표 | 행 | parent | ≥8후보 | **live** | 중앙 fan-out |
|---|---|---:|---:|---:|---:|---:|
| 개입 wave | `intervention_E1E2_f109` | 160 | 20 | 20 | 16 | 8.0 |
| 개입 wave | `intervention_E1E2_f121` | 160 | 20 | 20 | 8 | 8.0 |
| 개입 wave | `intervention_N1N2_f113` | 159 | 20 | 19 | 12 | 8.0 |
| 개입 wave | `intervention_HGD569_f125_v2` | 152 | 20 | 15 | 8 | 8.0 |
| 개입 wave | `intervention_T6T4_f121` | 143 | 18 | 17 | 8 | 8.0 |
| **개입 5 wave 합** | | **774** | **98** | **91** | **52** | |
| 탐색 캠페인 22종 | `fpcamp_*` · `v520_*` | 419 | 269 | **1** | 1 | 1.0–2.5 |
| **r2 캠페인** | `fpcamp_minfxy_e1e2_f121_r2` | **100** | **45** | **0** | **0** | 1.0 (최대 6) |

개입 wave 1개 = **160 MASTER 호출 → live parent 평균 10.4개**(52/5).
탐색 캠페인 100 호출 → live parent **0개**. 이 표가 §7 라벨 계획의 전부다.

---

## 2. 학습 대상 — target / loss

### 2a. 등록 상수는 v3 를 글자 그대로 승계한다 (변경 금지)

```
gain_fxy = max(0, -d_f_xy)
feasible = both_converged
           AND d_cyclen  >= -CYCLEN_TOL                  (CYCLEN_TOL = 5.0 EFPD)
           AND child_f_r <= max(parent_f_r, F_R_LIMIT)   (F_R_LIMIT = 1.55)
y_fxy    = min(gain_fxy * feasible, c_fxy) / c_fxy   in [0,1],  c_fxy = 0.060
```

`fr`/`flat` head 의 정의·clip(0.030/0.035)도 v3 = v2 승계. **이 세 상수 중 하나라도
바뀌면 그것은 v3.1 이 아니라 새 라운드다** — 게이트 통계는 어떤 모델링 변경보다
이 상수에 민감하다(게이트 폴드 전체 양성 65행).

### 2b. 2단 학습 — stage 1 은 v3 그대로, stage 2 만 새 것

**stage 1** = `train_v3` revB **바이트 동일 재현**(3 head, masked soft-target BCE,
`w_era × w_parent`). 산출 체크포인트가 v3.1 의 trunk 이자 `fr`/`flat` 출력이다.

**stage 2** = trunk 동결 + `fr`/`flat` 출력 동결 + **새 `fxy` 분기**
(자체 `2W + n_cond → 256 → 256 → 1` MLP, 동결 head 의 `fxy` 행으로 최종층 초기화),
lr 1e-4, 손실

```
L2 = masked_bce_soft(fxy)
     + λ · (1/|G|) Σ_{p∈G} Σ_{i∈p} −q_i · log softmax_p(z_fxy)_i
```

교사 `q` 는 **raw gain** 위에서 만든다:
`q_i ∝ exp(u_i / T) · feasible_i`, `u = −d_f_xy`, `T = 0.060`(등록 clip 상수 재사용,
**새 자유상수 없음**), infeasible 후보는 `q=0` 이되 softmax 분모에는 남기고, ε=0.10
균등 평활. 그룹 `G` = 학습 폴드에서 F_xy 라벨 후보가 2개 이상인 parent.

**왜 교사는 raw gain 이고 게이트는 등록 gain 인가 — 측정 근거.** 등록 gain 교사는
`Σ y_fxy = 0` 인 parent 를 전부 버리므로 학습 그룹이 57 parent/368행 →
**35 parent/232행** 으로 줄고, 2-seed 파일럿에서 `fxy` pb-AUC 가 0.8371(raw 교사) →
**0.7029**(등록 교사, full-net) / **0.6869**(등록 교사, head-only) 로 무너진다.
**목표는 순서만 유도하면 되고, 지표가 clip·실현가능성 게이트를 정의한다** — 둘을
교사에서 일치시키는 것이 파괴적이라는 것이 이 라운드의 설계 발견이다.

**왜 2단인가 — 측정 근거.** listwise 항을 full-net 에 걸면 `fr` head 가 무너진다:
2-seed 파일럿에서 λ=0 대조 `fr` 0.7808, λ=0.3 **0.6664**, λ=1 0.6900. 0.6664 는
v3 가 등록한 `fr` 회귀 바닥(0.728 − 0.05 = **0.678**) 아래다. 동결-분기 구조에서는
`fr`/`flat` 이 **구조적으로** stage-1 과 비트 동일하므로 회귀 절이 운이 아니라
설계로 통과한다.

### 2c. λ 선택 규칙 — `val` 단독, 사전 등록, 그리고 사전 기대

λ ∈ {0, 0.3, 1.0}, 선택 통계는 **`val` 3-head 평균 Spearman 단독**. `gate_cur`·
cross-fit 폴드·`prospective_cell` 의 어떤 수치도 λ 선택에 쓰지 않는다.

**사전 기대를 결과 전에 적는다: λ=0 이 선택될 가능성이 높다.** 2-seed 파일럿의
`val` Spearman 은 λ=0 **0.4557** > λ=0.3 0.4465 > λ=1 0.4445 였다(파일럿 로그
`~/lpopt_ws/scratch/p31_lw*`; **3-seed 본 학습에서 재산출해 결과 문서 §1 에
기록한다** — 위 값은 2-seed 파일럿이므로 판정 근거가 아니라 기대치다).
**λ=0 이 선택되면 v3.1 은 listwise 항 없이 §3(cross-fit) + §4(feature) +
§5c(스케일) + §6(게이트)만으로 출하되며, 그것은 정당한 등록 결과다.** listwise
라인은 그 경우 측정을 붙여 종결한다.

### 2d. stage 2 이전에 반드시 돌리는 파일럿 하나 (등록)

head-only gradient 가 실패한다는 증거는 **등록 gain 교사와 교락되어 있다**
(e1 = 등록교사 × head-guard `fxy` 0.6869 / `fr` 0.6494 vs e2 = 등록교사 × full-net
0.7029 / 0.6819). **raw 교사 × head-guard 칸은 한 번도 돌지 않았다.** 이 칸을
2-seed 로 먼저 돌리고(예상 ~300 s, GPU 1) 결과를 결과 문서 §1 에 적는다. 목적은
"동결 분기가 필요했는가"를 사후에 합리화하지 않는 것이며, **파일럿 결과가 무엇이든
stage-2 구조는 바꾸지 않는다** — 구조는 절 5(`fr` 회귀)를 구조적으로 만족시키기
위한 것이지 파일럿에서 고른 것이 아니다.

---

## 3. 학습 데이터 · fold · 가중

### 3a. cross-fit — 이 라운드에서 가장 싼 검정력, MASTER 호출 0

v3 의 단일 교대 gate/train 분할을 **K=5 계보-성분(component) blocked cross-fit**
으로 바꾼다. parent 가 아니라 성분 단위로 접는 것은 v3 의 leakage 규율을 그대로
유지하기 위해서다. 현행 era 전체를 out-of-fold 로 채점하고, `prospective_cell` 은
**K개 폴드 어디에도 들어가지 않는다.**

| 구성 | 게이트 pool ≥8후보 / live |
|---|---:|
| v3 실현 (`N1_N2/f113` 홀드아웃, 단일 분할) | **38 / 23** |
| cross-fit, 홀드아웃 `N1_N2/f113` | 80 / 45 |
| **cross-fit, 홀드아웃 `E1_E2/f109` (등록본)** | **79 / 41** |
| 단일 분할, 홀드아웃 `E1_E2/f109` (기각) | 37 / **16** |

cross-fit 없이 `E1_E2/f109` 를 홀드아웃으로 빼면 게이트 live 가 **16개**로 떨어져
어떤 절도 판정할 수 없다. **cross-fit 이 전이 셀 교체를 가능하게 만드는 전제이며
둘은 한 묶음으로 등록된다.**

### 3b. 블라인드 v2 베이스라인 재발행 (등록된 이탈)

현행 `policy_v3_v2_baseline.csv` 는 3,286 eval 행만 덮는다. cross-fit 은 현행 era
전체를 채점하므로 재발행이 필요하다. **v2 가중치는 v3 이전에 동결되었고 채점은
결정적이므로** 재발행은 블라인드를 약화시키지 않는다. 규율: 재발행 CSV 는
**v3.1 가중치가 하나도 존재하기 전에** 만들어 sha256 을 결과 문서 §1 에 기록하고,
이 문단 자체가 "이탈이며 그 사유는 이것"이라는 사전 등록이다. 심사자가 여전히
"약화된 블라인드"라고 부를 수 있다는 점도 함께 적는다.

### 3c. `prospective_cell` = `E1_E2/f109/ga80` — 사전 등록된 난이도 조건으로 고른다

**등록 조건(라벨만 쓰고 어떤 채점자도 쓰지 않는다). 넷 모두 만족해야 자격:**

1. ≥8 F_xy 후보 parent **≥ 20개**
2. 그중 live 비율 **≥ 0.70**
3. parent-max gain 중앙값 **≥ 0.10 clip 단위**
4. 순열-귀무 MDE ≈ `0.55/√n_live` **≤ 0.20**

| cell | (1) | (2) | (3) | (4) | 자격 |
|---|---:|---:|---:|---:|---|
| **`E1_E2/f109/ga80`** | 20 | **0.80** | **0.239** | **0.160** | ✅ **유일 자격** |
| `N1_N2/f113/ga80` (v3 홀드아웃) | 19 | 0.63 | 0.078 | 0.142 | ❌ (1)(2)(3) |
| `E1_E2/f121/ga80` | 22 | 0.45 | 0.000 | 0.174 | ❌ (2)(3) |
| `T6_T4/f121/paramA` | 17 | 0.47 | 0.000 | 0.194 | ❌ (1)(2)(3) |
| HGD569 `f125/paramA` | 15 | 0.53 | 0.005 | 0.227 | ❌ (1)(2)(3)(4) |
| `G3_G4/f125` · `E1_E2/f125` | 3 · 3 | — | — | 0.55 · 0.39 | ❌ (1)(4) |

(4) 열 중 **0.160 · 0.142 · 0.227 은 순열 귀무로 실측한 값**
(`policy_v31_transfer_redesign_20260903.csv` `mde_screen`)이고, 나머지는 같은 파일이
등록한 어림식 `0.55/√n_live` 로 유도한 값이다. STEP 0-b 가 전 셀을 실측으로
채운다.

**네 조건 모두 채점자를 쓰지 않는다.** f109 의 v3 in-sample 수치(pb-AUC 0.931,
NDCG 0.892, 순열 z 5.54)는 **f109 가 v3 의 학습·게이트 폴드 안에 있을 때** 잰
값이므로 **자격 근거로도 헤드룸 근거로도 인용하지 않는다** — 인용하면 v3 가 f113
에서 저지른 오류(전이 셀을 가설로 고르기)를 반대 방향으로 반복하는 것이다.

**교체 사유는 results §5-R2 가 등록한 것 그대로다.** f113 은 여섯 채점자와 세 head
전부가 우연 수준이었고(§3.4), 그 셀의 FAIL 이 지지하는 명제는 "v3 는 전이에
실패한다"가 아니라 **"이 셀은 전이 바로 쓸 수 없다"** 였다. R2 는 "전이 바는 셀의
난이도를 사전에 등록해야 한다"고 요구했고 위 넷이 그 등록이다.

**측정된 비용(등록).** 게이트가 f109 의 live 16 을 잃고 f113 의 live 12 를 얻는다.
단일 분할이면 23 → 16, cross-fit 이면 45 → **41**. 이 비용은 §7 Wave 1 이 갚는다.

### 3d. 실현 폴드 (STEP 0-b 에서 재계산해 부록에 고정한다)

| fold | 정의 | 예상 |
|---|---|---|
| `prospective_cell` | `E1_E2/f109/ga80` 전량 | 160행 · 160 F_xy · 20 ≥8후보 · 16 live |
| cross-fit pool | 나머지 현행 era, 성분 blocked K=5 | ~1,281행 · 542 F_xy · **79 ≥8후보 · 41 live** · 48 cell |
| `val` | pool 잔여 성분의 10%, era 별 독립 추출, early stop·λ 선택 전용 | ~71 F_xy 행 |
| `calib` | cross-fit out-of-fold 예측 전체(~1,000행) | Platt 적합 전용(§5c) |
| `train` (폴드별) | 나머지 전부(legacy 포함) | K = 5 개 |

`w_era`·`w_parent` 는 v3 §3c 그대로이며 손잡이가 아니다. **신규 규율: `val` 성분은
cross-fit 채점에서 제외한다** — v3.1 은 λ 를 `val` 로 고르므로 `val` parent 가
게이트 pool 안에 있으면 선택과 판정이 같은 행을 쓰게 된다(tests §9a-H(e)).

---

## 4. Feature — 하나는 기각, 하나는 추가 (구현 과제)

### 4a. R4(비보존 Gd/격자 계열)는 **측정으로 기각한다** — 65열을 넣지 않는다

results §5-R4 는 "`Σ mult·X` 는 `batch_swap` 아래 보존되므로 비보존 기술자가
필요하다"고 권고했다. 그 계열을 실제로 지어(반경 모멘트 35 + 무부호 슬롯 교란 16 +
연료형별 중심 변위·링 플럭스 14 = **65열**) 현행 era 1,307행 · 97 mixed parent ·
9 cell 에서 **leave-one-CELL-out ridge probe**(within-parent 평균제거 목표)로
측정한 결과:

| 계열 | Δ pb-AUC [parent-boot CI] | Δ regret@4 [CI] | 판정 |
|---|---|---|---|
| baseline 51 스칼라 | 0.6060 (기준) | 0.01790 (기준) | — |
| +RAD (35) | +0.0040 [−0.0177, +0.0257] | −0.00697 [−0.0289, +0.0080] | **null** |
| +ABS (16) | +0.0349 [−0.0005, +0.0763] · cell-clustered [−0.0215, +0.0551] | +0.00197 [−0.0078, +0.0131] | **null** |
| +DSP (14) | **−0.0809 [−0.1382, −0.0259]** | **−0.04675 [−0.0785, −0.0211]** | **유해** |
| +NC 전체 (65) | −0.0414 [−0.0954, +0.0142] | **−0.02535 [−0.0449, −0.0071]** | **유해** |

출처 `_wf_scratch/nc2_E3_loco.csv` · `nc2_E6_bycell.csv`.

**그리고 R4 의 전제가 겨눈 곳이 틀렸다.** `batch_swap` 형제쌍 96개는 **51 스칼라
전체에서 동점이 0개**다(`nc2_E2_ties.csv`) — 질량 보존 맹점은 v3 의 12 신규
스칼라에는 실재하되, v2 가 물려준 39열(`swap_span`·`swap_radius`·`fresh_r_center`·
농축 모멘트)이 이미 그 자리를 덮는다. 반경 가중 Gd 모멘트는 같은 구성을 Gd 에
적용한 것이라 그 39열과 거의 공선이고, ridge 가 그것을 확인한다.

**등록 처분: 비보존 Gd 계열은 넣지 않는다. 이 라운드는 R4 라인을 닫는다.**
(단 §4a 의 probe 는 ridge 이지 신경망이 아니다 — 이것은 **측정된 null 이지 불가능
증명이 아니며**, DSP 의 유해는 CI 가 0 을 배제하는 강한 결과, ABS 의 null 은 이
n 에서 미결정이라고 적는다.)

### 4b. 실제로 비어 있는 것은 **burnt(shuffled) 부격자**다

보드 토큰은 fresh 가 `F:<batch>:<rot>`, shuffled 가 `S:<restart>:<row>:<col>:<rot>`
다. v3 의 51 스칼라는 조성 정보를 **F 토큰에서만** 읽고, S 토큰은
`once/twice_burnt_periph_share`(연소-등급 share, 동급 교환에 보존)로만 들어온다.
따라서 **같은 등급의 burnt 두 장을 재배치하는 rewire 는 보드와 F_xy 를 바꾸면서
격자 계열의 move-level 채널을 전부 0 으로 둔다.**

| 사실 | 값 | 출처 |
|---|---|---|
| rewire 블록의 격자 move-level 채널 all-zero 비율 | **7개 (cell, move_class) 블록에서 1.00** (E1_E2/f109 rewire_swap 20 · E1_E2/f121 rewire_multi 50 · rewire_swap 36 · T6_T4/f121 rewire_multi 31 · rewire_swap 19 · N1_N2/f113 rewire_swap 21 · HGD569 rewire_swap 20 = **197행**) | `nc2_E1b_rawzero.csv` |
| 현행 era rewire 행 전체 | **213 / 1,307 (16.3%)**, 11개 블록 | `nc2_E1_conservation.csv` |
| 51 스칼라에서 아직 동점인 형제쌍 | **24쌍, 전부 `rewire_multi`** — 그중 등록 gain 이 다른 쌍 **12** | `nc2_E2_ties.csv` |

**두 표의 외견상 모순을 여기서 확정한다.** `nc2_E1_conservation.csv` 는
`allzero_v3_12 = 0.0`, `nc2_E1b_rawzero.csv` 는 `raw12_allzero = 1.0` 이다. 집계
대상이 다르다 — v3 의 12 스칼라 중 **move-level 9채널**(`d_fresh_gd_mass` ·
`d_fresh_gd_r_center` · `d_fresh_gd_share_periph` · `d_fresh_gdwt_mass` ·
`d_fresh_kinf0_mass` · `d_fresh_kinf0_r_center` · `n_fresh_type_changed` ·
`fresh_type_multiset_changed` · `fresh_gd_contrast`)은 rewire 에서 항등적으로 0 이고,
나머지 3열(`parent_fresh_gd_mass` · `parent_fresh_kinf0_mass` · `gdwt_present`)은
0 이 아니지만 **parent 안에서 상수**라 within-parent 랭킹 정보가 0 이다. 결과 문서는
이 구분을 **열 목록으로** 다시 확정해야 한다(등록 항목).

**따라서 이 결손의 크기는 정확히 이만큼이다: 24 형제쌍(그중 12쌍이 gain 상이) +
213행의 within-parent 해상도.** 그 이상으로 팔지 않는다.

### 4c. 추가하는 것은 정확히 2열, 그리고 **선택 규칙**을 등록한다

burnt 계열 22열을 지어 보았고 그중 둘만 넣는다. **선택은 p-값이 아니라 구조 규칙으로
한다** — 심사에서 지적된 "coverage 로 포장한 outcome-selection"을 막는 유일한
방법이다.

> **등록 선택 규칙.** burnt 계열 중 (i) 동급 burnt 두 장의 **순수 치환 아래 비보존**
> 이고(= `Σ mult·g(r)·X` 형태로 표현 불가, 절대값 항을 포함), (ii) 현행 era rewire
> 행의 100% 에서 살아 있는 열을 **전부** 넣는다. 이 규칙을 만족하는 열은 정확히
> `burnt_absmov`, `burnt_absmov_r` 둘이다. `rew_cor_*` · `rew_flux_*` 는 (i) 을
> 만족하지 않는 **보존형 모멘트**이므로 **p-값과 무관하게 제외**된다
> (원 제안이 넣으려던 `rew_cor_r2` 는 p 0.0103 으로 자기 Bonferroni 선 0.0023 도
> 넘지 못했고, 여기서는 구조 규칙으로 먼저 탈락한다).

```
r_prev(slot) = |(row,col) from S token − core centre| / max     # 직전 주기 위치
r_now(slot)  = SLOT_RADIUS(slot) / max
burnt_absmov   = Σ_{burnt slots} mult · |r_now − r_prev|
burnt_absmov_r = Σ_{burnt slots} mult · r_now · |r_now − r_prev|
```

둘 다 **burnt 가 하나도 움직이지 않으면 항등적으로 0** 이므로 별도 liveness 플래그가
필요 없다. 물리적 해석: "이번 주기의 연소 재고가 자기가 탄 반경 대비 얼마나 멀리
재배치되었는가" — 주변부 F_xy 를 만드는 burnt 출력 불일치의 직접 손잡이다.

**등록 근거는 커버리지다**(§4b 표: 213/1,307 · 7블록 · 24쌍/12쌍). **비등록 판독으로만
보고**: `burnt_absmov_r` 의 within-parent 쌍 일치도 68쌍 0.279 (p 3.6e-4,
Bonferroni 0.05/22 통과), 방향은 "재배치가 클수록 F_xy 가 나쁘다", 이동 크기
대리변수가 아니다(같은 쌍에서 `n_slots_changed` 0.587 p 0.30, `swap_span`/
`swap_radius` 0.412 p 0.18; `n_slots_changed` 가 동일한 22쌍 부분집합에서도 0.273).
**그러나 68쌍 중 61쌍이 `E1_E2/f121` 한 셀이므로 이 수치로 게이트하지 않는다** —
그렇게 하면 v3 가 H3/Gd 로 저지른 오류를 그대로 반복한다.

**반증 판독(등록).** r3 의 rewire-rich wave 가 **다른 셀**에서 `burnt_absmov_r`
일치도 < 0.50 을 내면 두 열을 철회하고 F_xy burnt-격자 라인을 닫는다.

**22열 전체를 넣지 않는 이유(측정).** rewire 행 LOCO 에서 51+22 는 within-parent
일치도를 0.5441 → **0.3971** 로 떨어뜨린다. 학습행 530 · 양성 79 에서 22열은 살 수
없다.

### 4d. 코퍼스 열 — `mine_policy_corpus` 구현 과제

`steps_v31.parquet` = `steps_v3.parquet`(107열) **+ 4열**:

| 열 | 종류 | 정의 |
|---|---|---|
| `burnt_absmov` | move-level, **feature** | §4c |
| `burnt_absmov_r` | move-level, **feature** | §4c |
| `burnt_slots_moved` | move-level, 진단 | `r_now ≠ r_prev` 인 burnt 슬롯 수 |
| `burnt_token_complete` | bool, 진단 | 두 pattern 의 모든 S 토큰이 `<restart>:<row>:<col>` 로 파싱되는가 |

→ **85(v2) → 107(v3) → 111(v3.1)열**. 스칼라 조건 벡터 **51 → 53**.
`scalar_features_v31` 은 `scalar_features_v3` 를 **호출**하고 v3 의 51열은 v3.1 안에서
**비트 동일**해야 한다(tests §9a-H(a)). `burnt_token_complete` 는 feature 가 아니라
진단이며 학습 벡터에 들어가는 것은 2열뿐이다. 스케일은 현행 era 동일-셀 행의
p05/p95 를 반올림해 **상수로 등록**하고 적합하지 않는다(전 폴드 공통이므로
정규화 leakage 를 만들지 않는다).

**동시에 수행하는 코퍼스 갱신, 사전 등록된 기대치와 함께.** r2 캠페인
`fpcamp_minfxy_e1e2_f121_r2` 100행(99 수렴)을 코퍼스에 넣는다. 기대:
parent 45 · parent-해결 가능 쌍 81 · feasible 64 · `y_fxy>0` 33 ·
**신규 ≥8후보 parent 0개** · 신규 ≥4후보 parent 4개(정보 parent 약 2). 즉
**r2 행은 학습행과 level 지도를 늘리고 게이트 절은 하나도 사지 않는다.** phase-2
`pinbu_wave_minfxy_r2` 25행은 parent/child move 가 아니므로 step 을 만들지 않는다
(STEP 0-b 에서 확인하고, 만들면 그 사실을 이탈로 기록한다).

---

## 5. 평가

### 5a. 지표 요약

| # | 지표 | gain | 폴드 | 역할 |
|---|---|---|---|---|
| M1 | parent-blocked AUC (`improved_fxy`) | — | cross-fit pool | **게이트 절 1** |
| M2 | **NDCG@4-of-8** | 등록 `y_fxy` | cross-fit pool | **게이트 절 2A / 2B** |
| M3 | gain-가중 within-parent 일치도 (wPBC) | 등록 `y_fxy` | cross-fit pool | 절 2 보조(같은 방향 요구) |
| M4 | 셀별 M2 | 등록 `y_fxy` | cross-fit pool | **게이트 절 3 (within-cell)** |
| M5 | `regret@{1,2,3,4}-of-8` | 등록 + raw 병기 | cross-fit pool | **보고 전용**(§1b) |
| M6 | M1 + M2 | 등록 `y_fxy` | `prospective_cell` | **전이 바** |
| M7 | LOCO(leave-one-cell-out) ridge probe pb-AUC | 등록 `y_fxy` | 현행 era 9 cell | 보고(probe 이지 게이트 아님) |
| M8 | 캘리브레이션(ECE/Brier) · 점수 폭 · seed 산포 · `fr`/`flat` | — | 전부 | 보고 + 절 4 · 절 5 |
| M9 | 전향적 A/B/C | — | `min_fxy` 캠페인 | **이 라운드에서 등록하지 않는다**(§6c) |

**평가 라벨은 `improved_fxy`(M1)와 등록 gain `y_fxy`(M2–M6)이며 학습 목표가 아니다.**
구현은 **모든 랭킹 통계에 `y_fxy` 를 먹여야 하고**, raw `−d_f_xy` 를 먹이면 실패하는
단위시험을 둔다(tests §9a-H(c)). 이것이 v3 편차 §1.6 을 양쪽에서 닫는다.

### 5b. k 는 4 로 고정한다 — 사후 선택 금지

§1a 가 보인 대로 v3-vs-v2 부호는 k 에 의존한다. **v3.1 은 v3 가 등록한 k=4 를 그대로
쓴다**(사유는 v3 §5b 의 데이터 구조 그대로: 개입 parent 는 후보가 정확히 8개이고
≥10후보 parent 는 코퍼스에 6개뿐이다). 지표는 `regret` → `NDCG`+`wPBC` 로 바뀌지만
**k 는 바뀌지 않으며** k ∈ {1,2,3,4} 전 sweep 을 보고한다. **어떤 k 에서든 베이스라인이
v3.1 을 CI 배제로 이기면 그 사실을 헤드라인에 적는다**(반-k-쇼핑 조항).

### 5c. 서빙 스케일과 τ — 편차 §1.7 을 닫는다

측정(게이트 폴드 `fxy` 행): v3 의 **확률** p90−p10 = 0.0324 이지만 **로짓**
p90−p10 = **9.66**, parent 내 로짓 폭 중앙 **8.64**. `policy_v2` 는 확률 0.160 /
로짓 4.92 / parent 내 3.75. **즉 "붕괴한 스케일"은 모델 결함이 아니라 15% 기저율
위에서 sigmoid 가 압축한 서빙 아티팩트다** — v3 는 v2 보다 parent 안에서 더 넓게
벌린다.

* v3.1 은 `z_fxy` 에 **단조 Platt 사상**(`a·z + b`)을 걸어 서빙한다. 순위 보존이므로
  어떤 랭킹 통계도 미화할 수 없다.
* Platt 은 **`calib` 폴드(cross-fit out-of-fold 예측 ~1,000행)에서만** 적합한다.
  `val` 71행에서 적합하면 기저율을 넘겨(평균 0.160 vs 실측 0.071) binned ECE 가
  0.0521 → 0.0858 로 악화된다 — 측정된 사실이며 그래서 `calib` 를 쓴다.
* **τ_v3.1 은 게이트를 열기 전에** v1 규칙(p10–p90 폭에서 ~10배 sampling-odds)으로
  유도해 결과 문서 §1 에 적는다. 참고: val-Platt 로는 폭 0.0324 → 0.2488,
  τ 0.0142 → 0.1086.
* **ECE ≤ 0.05 는 게이트하지 않는다.** 측정된 두 구성(원본 0.0521, val-Platt 0.0858)
  모두 그 선을 넘지 못하므로, 등록하면 **이미 아는 FAIL 을 등록하는 것**이 된다.
  ECE·Brier 는 보고하고 게이트되는 것은 절 4 의 **폭 조건**뿐이다.
* **미해결 불일치(등록).** results §3.3 은 같은 폴드의 폭을 **0.0449**, τ_v3 를
  **0.0196** 으로 기록했고 이번 재측정은 0.0324 / 0.0142 다. **등록값 0.0449 를
  권위값으로 두고, 결과 문서는 두 수치를 행·열 정의 수준에서 화해시켜야 한다**
  (등록 항목; 어느 쪽이든 로짓 폭 9.66 에는 영향이 없다).
* `construct.py` 의 τ=0.0 가드는 **로짓 스케일로 재표현**되어야 한다. 재표현하지
  않으면 가드가 다른 의미의 온도를 조용히 통과시킨다(§9a-F).

### 5d. 베이스라인 — v3 의 다섯을 그대로 승계

`random` · `class_freq` · `periph` · `gd_rule` · `policy_v2`. 앞 넷은 각 cross-fit
폴드의 `train` 에서만 적합한다. `policy_v2` 는 적합하지 않고 §3b 의 재발행 CSV 로
채점한다. **`class_freq` 의 NA 폴백 수정(v3 편차 §1.5)은 유지한다** — 베이스라인을
강하게 만드는 방향의 수정이다.

---

## 6. GATE · 전이 바 · 처분 — 검정력 산술과 함께

평가 대상: cross-fit out-of-fold 앙상블(K=5 × 3 seed), head `fxy`,
gain = **등록 `y_fxy`**, paired parent-bootstrap 4,000 draw **+ cell-clustered**
bootstrap 병기. **모든 CI 옆에 정보 parent 수와 n80 을 인쇄한다.**

### 6a. 절과 그 검정력

**절 1 — parent-blocked AUC ≥ 0.65 이고 95% CI 하한 > 0.50.**
v3 실측 0.840 [0.769, 0.910](mixed parent 37). 위험 낮음. 회귀 방지 절이다.

**절 2A — NDCG@4-of-8(등록)이 적합 베이스라인 넷 전부보다 높고 각 paired CI 가 0 을
배제한다.** 오늘 이미 통과하는 절이며 그 사실을 먼저 적는다:

| vs | Δ (v3, live parent 23) | CI | n80 |
|---|---:|---|---:|
| `random` | +0.5319 | [+0.3764, +0.6793] | **3.9** |
| `periph` | +0.5242 | [+0.3999, +0.6396] | **2.5** |
| `class_freq` | +0.2846 | [+0.1523, +0.4101] | **9.9** |
| `gd_rule` | +0.2097 | [+0.1184, +0.3047] | **9.8** |

(출처 `_wf_scratch/v31_power_table.csv` · `p31g_eval.csv`.) **따라서 절 2A 는 새
정보를 사지 않는다 — 회귀 방지 절로 등록한다.** 이 절만 통과한 PASS 는 "v3.1 이
v3 를 망가뜨리지 않았다"는 뜻이며 그 이상으로 읽지 않는다.

**절 2B — `policy_v2` 에 대한 비열등, 마진 δ = 0.05, 단측 95%.**
판정: NDCG@4(등록)의 paired 차이 **하한 > −0.05**.

관측: Δ = **+0.0423**, CI [−0.0632, +0.1525], **sd = 0.2724**, live parent 23,
정보 parent 15.

| 필요 live parent | 기준 |
|---:|---|
| **24** | 관측 Δ 가 참일 때 단측 CI 가 −0.05 를 넘는 최소 n (`policy_v31_transfer_redesign_20260903.csv` `ni_sizing`) |
| **54** | 같은 조건에서 **검정력 80%**: `n = ((1.645+0.842)·sd/(Δ+δ))²` = (2.4866·0.2724/0.0923)² |
| **184** | **참값이 정확히 0** 일 때의 80% 검정력 |
| 39 / 52 / 74 | δ 를 0.03 / 0.02 / 0.01 로 조일 때의 CI 기준 n |

보유: cross-fit + f109 홀드아웃 = **41 live**. 그 지점의 검정력은 **Δ = Δ_obs 이면
70%, 참값이 0 이면 32%** 다. §7 Wave 1 이 +11~14 를 더해 **52–55 → 검정력 79–81%.**

> **심사에서 지적된 오류를 여기서 정정한다.** 원 제안은 "cross-fit 이면 하한
> −0.037 → PASS 예상"이라고 적었다. 그것은 **점추정을 상수로 두고 반폭만 잰
> 계산**이며 검정력이 아니다. 위 표가 등록되는 산술이고, **41 live 에서 절 2B 를
> 여는 것은 70% 검정력 시험**이라는 사실을 결과 문서 헤드라인에 적는다.

> **동등성(TOST)을 등록하지 않는 이유도 산술로 적는다.** 심사 대상 제안 하나는
> "75 informative parent 에서 ±0.05 동등성이 판정 가능"이라 적었으나, sd = 0.2724
> 에서 n=75 의 단측 반폭은 1.666·0.2724/√75 = **0.0524 > 0.05** 이므로 관측값이
> 정확히 0 이어도 TOST 는 통과하지 못한다. ±0.05 동등성에는 **n ≳ 192** 가 필요하다.
> 따라서 v3.1 은 **동등성 대신 비열등만** 등록하고, "동등"이라는 단어를 결과에
> 쓰지 않는다.

**절 2C — `policy_v2` 우월은 게이트하지 않는다.** 사유를 산술과 함께 등록한다:
NDCG@4(등록)에서 CI 배제에 **113 live parent**, 80% 검정력에 **326**;
`regret@1-of-8`(등록)에서는 **349**. 개입 wave 당 live parent 10.4개이므로 CI
기준으로도 **약 11 wave(~1,760 MASTER 호출)**, 검정력 기준으로는 **약 31 wave
(~5,000 호출)** 다. 더 깊은 사유는 모델링으로 고칠 수 없다 — 셀별
ρ(`d_f_xy`, `d_f_r`) 가 **0.878–0.967**(results §3.2)이므로 F_r 랭커 대비 얻을 수
있는 효과 자체가 작다. **이것은 게이트 절이 아니라 예산 질문이며 그렇게 분리해
올린다.**

**절 3 — within-cell ranking 조항 (r2 판정이 요구한 것).**
절 2A 의 `class_freq` · `gd_rule` 우월이 **자격 셀마다 개별적으로** 성립하고,
`policy_v2` 에 대해 **어떤 자격 셀에서도 δ 이상 지지 않아야** 한다.

* **자격 규칙(사전 등록).** (i) 셀의 live parent **≥ 10** — 임의값이 아니라
  `class_freq` 상대 셀내 n80 = **9.9**(Δ 0.2846, sd 0.3193)에서 온 수다.
  (ii) 셀이 **순위 가능**해야 한다 — 다섯 베이스라인 중 **최소 하나**가 그 셀의
  within-parent 순열 귀무를 p ≤ 0.05 로 벗어난다. **v3.1 자신의 점수는 이 판정에
  쓰지 않으며**, 이 검정은 STEP 0-b 에서 사전 계산·해시한다.
* **오늘의 자격 셀(f109 홀드아웃 + cross-fit 기준):** `N1_N2/f113` 12 live ·
  `E1_E2/f121` 10 live. HGD569 8 · `T6_T4` 8 · `E1_E2/f125` 2 · `G3_G4` 1 은 (i)
  미달. **(ii) 는 f113 에서 통과하지 못할 가능성이 높다**(results §3.4: 여섯
  채점자 전부 우연 수준) — 그 경우 f113 은 **판정 불가**로 기록되고 FAIL 로 세지
  않는다. **이 처리를 결과 전에 등록하는 것이 이 절의 핵심이다.**
* **셀 편중 조항.** 절 2B 판정에 쓰인 live parent 중 **한 셀이 40% 를 넘으면 판정을
  무효**로 한다. 오늘 41 live 의 최대 셀 지분은 f113 12/41 = **29%** 로 통과한다.
* **다중성.** 자격 셀 × 베이스라인의 가족은 **연언(conjunction) 요구**로 등록하므로
  보정하지 않는다(보수적 방향).

**절 4 — 서빙 스케일.** `calib` 폴드에서 적합한 Platt 점수의 게이트 폴드
p90−p10 **≥ 0.15**(`policy_v2` 의 0.160 과 같은 자릿수), 그리고 **τ_v3.1 이 게이트를
열기 전에 유도되어 문서화**될 것. ECE·Brier 는 보고한다(§5c).

**절 5 — `fr`/`flat` 회귀 없음.** `fr` parent-blocked AUC **≥ 0.678**(= v2 등록
0.728 − 0.05). §2b 의 동결 구조로 **구조적으로** 만족되어야 하며, 만족되지 않으면
그것은 구현 결함이다(tests §9a-H(b) 가 stage-1 과의 비트 동일성을 검사한다).

**GATE PASS = 절 1 ∧ 2A ∧ 2B ∧ 3 ∧ 4 ∧ 5.** 부분점수 없음, 사후 지표 대체 없음.

### 6b. 전이 바 (게이트 산출 후 한 번 연다)

`prospective_cell` = `E1_E2/f109/ga80`(160행 · 20 ≥8후보 · 16 live · MDE 0.160):

* parent-blocked AUC **≥ 0.60**,
* NDCG@4(등록)가 `random` · `class_freq` · `periph` · `gd_rule` 넷보다 높고 각
  paired CI 가 0 을 배제(셀내 n80 ≤ 10 이므로 16 live 로 판정 가능),
* **`policy_v2` 는 보고만 하고 게이트하지 않는다** — 한 셀 16 live 에서 필요치
  113(CI) / 326(80%) 은 불가능하다. **이 비대칭을 사전에 적는다.**

f109 는 §3c 의 네 조건을 만족하므로 **PASS 든 FAIL 이든 해석 가능하다** — 그것이
f113 대비 이 라운드가 사는 유일한 것이다.

### 6c. A/B/C (M9) — 등록하지 않는다

v3 결과 §4-2 의 선행조건 둘(모든 wave 가 `fxy_source = "head"`, §8-F·G serving 델타)
이 여전히 미충족이고, r2 판정 §11.2 는 **r3 의 랭커를 `s1i` 로 되돌렸다**. 따라서
v3.1 이 게이트를 통과해도 이 라운드의 처분은 `shadow_v3.1` 계측까지다.

### 6d. 처분표 — 결과 조합이 무엇을 허가하는가

| 결과 | 처분 |
|---|---|
| 절 1·2A·2B·3·4·5 전부 PASS + 전이 PASS | `shadow_v3.1` 계측 실행, τ 확정. **serving 교체는 하지 않는다**(§6c). v3.1 을 다음 라운드의 랭커 후보로 승격 |
| **절 2B FAIL 이고 실현 검정력 < 80%** | **판정 불가(UNDECIDED)** 로 기록한다. FAIL 로 적지 않는다 — 이 규칙을 결과 전에 등록하는 것이 §1b 의 교훈이다. serving 은 v2 유지 |
| 절 2B FAIL 이고 검정력 ≥ 80% | v3.1 은 `policy_v2` 보다 나쁘다. **F_xy 신경망 랭커 라인을 닫고** `policy_v2` 를 유지한다 |
| λ=0 이 `val` 에서 선택 + 게이트 PASS | cross-fit · 스케일 · burnt 2열만 출하하고 **listwise 라인을 측정과 함께 종결** |
| 절 3 이 자격 셀에서 FAIL | r2 판정의 조항이 발동한다 — **어떤 새 랭커도 wave 를 랭크하지 않는다.** `gd_rule` / `burnt_absmov_r` 은 생성기 prior 후보로만 남는다 |
| 절 4 FAIL (폭 < 0.15) | 서빙 스케일 문제이지 랭킹 문제가 아니다. τ 를 유도하지 못하므로 A/B/C 는 영구 차단이고, 다음 라운드는 목표 재정의(parent 내 z-정규화 / 순수 listwise)로 간다 |
| 절 5 FAIL | 동결 분기가 설계대로 동작하지 않은 것이다. **중단하고 재설계한다. 바를 낮추지 않는다** |
| 전이 FAIL (f109) | f113 과 달리 **진짜 전이 실패**로 기록된다(셀이 자격 조건을 만족하므로). 다음 지렛대는 **셀 수**이지 feature 가 아니다 |

---

## 7. 라벨 계획 — 어떤 호출이 어떤 절을 사는가

wave = `intervention_wave.py`, 20 parent × 8 designed move = **160 MASTER 호출**,
실측 수율 **≥8후보 parent 15–20 / live 8–16(평균 10.4)**.

| # | 대상 | 호출 | 사는 것 | 기대 live |
|---|---|---:|---|---:|
| **W0** | **없음 — cross-fit** | **0** | 게이트 pool 38/23 → **79/41**. 절 2B 를 70% 검정력으로 열 수 있게 하고 절 3 의 자격 셀을 2개로 만든다. **MASTER 호출 하나 쓰기 전에 이것부터 한다** | +18 |
| — | 코퍼스 r2 병합 | 0 | 학습행 +100, level 지도 +33. **게이트 절은 0**(§4d) | **0** |
| **W1** | `E1_E2/f121/ga80` | 160 | 절 2B 를 **79–81% 검정력**으로(41 → 52–55). 절 3 의 f121 을 10 → ~21 live 로. live율 0.45 인 셀이므로 설계의 **비관 케이스 교정**도 겸한다 | +11~14 |
| **W1 정지검사** | — | — | 실현 live 수율 < 0.5 이면 **절 2B 라인을 폐기**하고 라운드를 §6d 의 판정 불가 행으로 닫는다. W2 를 사지 않는다 | — |
| **W2** | `T6_T4/f121/paramA` | 160 | 절 3 의 세 번째 자격 셀(8 → ~19 live), 셀 편중 조항의 여유 | +8~14 |
| — | **하지 않는다** | — | 탐색 캠페인(`fpcamp_*`)을 절 2 의 대책으로 쓰는 것. 실측 **100 호출당 게이트 parent 0개** | 0 |
| — | **별도 예산 안건** | ~1,760(CI) / ~5,000(80%) | 절 2C(`policy_v2` 우월). 게이트하지 않으므로 이 라운드 예산이 아니다 | +113 / +326 |

**octet 설계 — 절 3 이 요구하는 within-cell 규율.** 각 wave 의 parent 당 8후보는
(1) incumbent(`policy_v2`) 최고점 후보, (2) challenger 최고점 후보(= (1) 과 다를 것),
(3) level 추정자가 feasible-and-improving 으로 예측한 후보 ≥3개, (4) 방향 대비 후보
≥2개 로 구성한다. **선택은 MASTER 호출 이전에 점수만으로 이뤄지고 (1)(2) 는 두
채점자에 대해 대칭이다.** 단 **(3) 은 대칭이 아니다** — level 추정자의 순서가 어느 한
채점자와 더 닮았을 수 있다. 그래서 **wave 의 25%(20 중 5 parent)는 (1)(2) 제약 없이
무작위로 뽑아 marginal 층으로 남긴다.** 두 층을 모두 보고하고 게이트는 조건부 층에서
연다. (3) 의 비대칭 크기는 W1 자료로 **측정해 결과 문서에 적는다**(level 추정자 순위
vs v3.1 / vs `policy_v2` 일치도).

---

## 8. 반증 판독 — 미리 적어 나중에 합리화할 수 없게

* **절 2B 가 하한 −0.05 를 아슬하게 넘는다.** 41–55 live 에서 이것은 70–81% 검정력의
  단일 시험이고 **비열등은 동등이 아니다.** 결과 문서는 "v3.1 은 `policy_v2` 와
  순위에서 구별되지 않으며, 이 라운드가 산 것은 판정 가능한 게이트와 유도 가능한
  τ 다"라고 적어야 하며 **"v2 를 이겼다"고 적어서는 안 된다.**
* **λ=0 이 선택된다.** listwise 항이 등록 gain 위에서 무익하다는 뜻이며 그것은
  실패가 아니라 **측정**이다(§2c). 2-seed 파일럿의 등록 gain 판독이 이미 그
  방향이었다: λ=1 은 NDCG@4 를 0.8308 → 0.8419 로 올리는 대신 `regret@1-of-8` vs v2
  를 +0.0284 → +0.0019 로, wPBC 를 0.8876 → 0.8792 로 낮췄다. **raw gain 소비자
  지표에서만 깨끗이 개선된다**(regret@4 0.00111 → 0.00057) — 그것은 이 라운드가
  게이트하지 않는 gain 이다.
* **burnt 2열이 아무 것도 바꾸지 않는다.** rewire 행은 게이트 폴드의 16% 이고 그중
  ≥8후보 parent 에 드는 것은 일부다. **기대 효과는 pb-AUC·NDCG 에서 noise 안이며,
  등록 근거는 커버리지이지 효과가 아니다**(§4c). 효과가 없다고 두 열이 틀린 것은
  아니며, 틀렸다는 증거는 §4c 의 반증 판독뿐이다.
* **cross-fit 이 폴드당 학습행을 줄여 절 1 이 내려간다.** 그러면 원인은 cross-fit
  이고 대조는 **같은 코퍼스·같은 cross-fit 위의 v3 재적합**이다(§9d 의 2-arm 규율).
  대조 없이 cross-fit 을 탓하지 않는다.
* **`E1_E2/f109` 전이가 FAIL 한다.** f113 과 달리 셀이 자격 조건을 만족하므로 이번엔
  **진짜 전이 실패**다. 등록된 1순위 후보 사유는 `gd_wt` 결측이 **아니다** —
  results §3.4 가 그것을 이미 기각했다(`gdwt_present=0` 쪽 pb-AUC 0.874 > 1 쪽
  0.768). 새 1순위 후보는 **셀 특이적 gain 스케일**(f109 의 parent-max gain 중앙
  0.239 는 게이트 셀들의 5–50배)이며, 검사는 게이트를 **f109 의 gain 분포로 재가중해
  재채점**하는 것이다.
* **절 3 이 f113 에서 "판정 불가"로 빠진다.** 예상된 결과다(§6a). FAIL 로 적지 않는
  대신 **f113 이 두 라운드 연속 아무 것도 판정하지 못했다**는 사실을 적고, 그 셀을
  게이트 통계에서 영구 제외할지 다음 라운드에 묻는다.
* **HGD569 `batch_swap` 미스가 그대로 남는다.** v3.1 은 fresh 격자 계열을 바꾸지
  않으므로(§4a 가 그 라인을 닫았다) 이 미스는 **알면서 이월**한다. 게이트 폴드의
  유일한 실질 regret 미스가 그것이라는 사실도 함께 이월한다.
* **`policy_v2` 재발행 CSV 가 논쟁이 된다.** §3b 의 논거(가중치 동결·결정적 채점·
  v3.1 이전 해시)가 받아들여지지 않으면, 절 2B 는 **원 3,286행 CSV 가 덮는 부분집합**
  에서 다시 계산해 함께 보고한다. 그 부분집합의 live parent 수를 결과 문서에 적는다.

---

## 9. 구현 과제 · 학습 프로토콜 · 실행 명령

### 9a. 코드 델타 (전부 신규 파일; v3 경로는 실행 가능한 상태로 보존한다)

| # | 파일 | 델타 |
|---|---|---|
| A | `mine_policy_corpus.py` | S 토큰 파서 + §4d 의 4열, `backfill-v31` 모드 → `data/policy/steps_v31.parquet`. **`steps_v3.parquet` 은 바이트 보존.** r2 캠페인 100행 병합 |
| B | `lpopt/policy/v31.py` (신규) | `POLICY_SCHEMA_V31 = "policy_move_v31"`, `NEW_SCALARS_V31 = ("burnt_absmov","burnt_absmov_r")`, `scalar_features_v31`(= `scalar_features_v3` **호출** + 2열), `build_splits_v31`(K=5 성분 blocked cross-fit + `holdout_cell` + `calib` 폴드 + `val` 성분 제외) |
| C | `lpopt/policy/train_v31.py` (신규) | stage 1 = v3 revB 그대로 호출, stage 2 = 동결 trunk + 신규 `fxy` 분기 + BCE + λ·listwise(교사 §2b), parent 단위 그룹 배치, `--lam-grid`, `--xfit-k`, `--emit-v2-baseline` |
| D | 지표 모듈 | `regret`/`NDCG`/`wPBC` 전부 `y_fxy` 를 먹는다. 등록·raw 두 gain 병기, 셀별 표, **모든 CI 옆에 정보 parent 수와 n80 인쇄** |
| E | `lpopt/policy/scorer.py` | `MoveScorerV31`(스키마 스탬프 강제, Platt 사상을 적용한 로짓 서빙). **v2·v3 경로 무편집** |
| F | `lpopt/search/construct.py` | `POLICY_MODES` 에 `"v31"`, `"shadow_v31"`. `policy_prior_model_dir_v31`, `policy_prior_temperature_v31`. **τ 가드를 로짓 스케일로 재표현**(§5c) |
| G | `scratch/v31_precensus.py` (신규) | STEP 0-b 의 census 6종: 폴드 실현치 · 셀 live · **셀 순위가능성 순열검정(절 3-(ii))** · r2 병합 효과 · burnt 커버리지 · v2 베이스라인 해시 |
| H | `tests/test_policy_v31.py` | (a) v3 의 51열이 v3.1 안에서 비트 동일 · (b) `fr`/`flat` 출력이 stage-1 체크포인트와 비트 동일 · (c) 게이트 통계가 raw gain 을 받으면 raise · (d) `holdout_cell` 이 K개 폴드 어디에도 없음 · (e) `val` 성분이 게이트 pool 에 없음 · (f) burnt 2열이 rewire 행에서 비영이고 fresh-only move 에서 0 · (g) serving 재현(로짓+Platt) |

### 9b. 학습 프로토콜 — 사전 고정, sweep 없음

| knob | 값 |
|---|---|
| stage 1 | v3 revB 전부 승계: AdamW, lr 1e-3, wd 1e-4, grad-clip 5.0, cosine 120 epoch, batch 256, patience 15, width 112, residual block 6, 대각 미러 p=0.5(train 전용) |
| stage 2 | 동결 trunk · 동결 `fr`/`flat`; 신규 `fxy` 분기 lr **1e-4**, 40 epoch, patience 10 |
| 교사 | raw gain, `T = 0.060`, ε = 0.10, feasibility mask |
| λ | {0, 0.3, 1.0} — **`val` 평균 Spearman 단독 선택**(§2c) |
| cross-fit | K = 5, 성분 blocked, seed 3개(20260903–20260905) → 멤버 15 |
| 헤드라인 | 폴드별 OOF 점수의 seed 평균; seed 산포 보고 |
| TTA / 전이 초기화 | 없음 / 거부(v3 §8a 사유 승계) |
| 예상 소요 | v3 의 1,833 s × 약 3 = **약 90분**, GPU 1 단독 |

### 9c. 정확한 실행 명령 — HOST_238 GPU 1

전제: `lpopt_gpu1.inp`(238 = `USER@HOST_238:8022`, venv `~/lpopt_ws/venv`),
발사 전 `nvidia-smi` 로 **GPU 1** 점유 확인. **GPU 0 은 다른 사용자 것이므로 절대
건드리지 않는다.** 199 / 198 / 181 무접촉.

```bash
# STEP 0-a) 지문 — 이 문서의 빈칸을 채운다 (v3.1 가중치 이전)
ssh -p 8022 USER@HOST_238 'cd ~/lpopt_ws && \
  sha256sum data/policy/steps_v31.parquet data/store/fuel_types.parquet \
            lpopt/policy/v31.py lpopt/policy/train_v31.py lpopt/policy/scorer.py'

# STEP 0-b) 사전-런 census 6종 (라벨만 사용, 채점자 미사용)
ssh -p 8022 USER@HOST_238 'cd ~/lpopt_ws && \
  venv/bin/python scratch/v31_precensus.py \
      --steps data/policy/steps_v31.parquet \
      --holdout-cell "E1_E2/f109/ga80" --xfit-k 5 --base-seed 20260903 \
      --out data/reports/_wf_scratch/v31_precensus_20260903.json'

# 0) 블라인드 v2 베이스라인 재발행 — v3.1 가중치가 존재하기 전에
ssh -p 8022 USER@HOST_238 'cd ~/lpopt_ws && \
  venv/bin/python -m lpopt.policy.train_v31 \
    --steps data/policy/steps_v31.parquet \
    --fuel-types data/store/fuel_types.parquet \
    --v2-model-dir data/models/policy_v2 \
    --holdout-cell "E1_E2/f109/ga80" --xfit-k 5 --base-seed 20260903 \
    --emit-v2-baseline data/design/policy_v31_v2_baseline.csv && \
  sha256sum data/design/policy_v31_v2_baseline.csv'

# 1) 등록된 필수 파일럿 — raw 교사 x head-only gradient (2 seed, 약 300 s)
ssh -p 8022 USER@HOST_238 'cd ~/lpopt_ws && \
  CUDA_VISIBLE_DEVICES=1 venv/bin/python scratch/p31j_listwise_guard.py \
      --teacher raw --guard head --seeds 2 \
      --out data/reports/_wf_scratch/p31j_guard_20260903.json'

# 2) 본 학습 — 로컬 런처가 ship -> launch -> poll -> pull 을 수행
python train_policy_v31.py --ts policy_v31 --seeds 3 --xfit-k 5 --epochs 120 \
    --extra "--holdout-cell E1_E2/f109/ga80 --base-seed 20260903 --protocol revB \
             --lam-grid 0,0.3,1.0 --gain registered --calibrate platt-oof"
```

런처가 238 에서 실제로 실행하는 것(`runs/policy_v31/run.sh`):

```bash
CUDA_VISIBLE_DEVICES=1 ~/lpopt_ws/venv/bin/python -m lpopt.policy.train_v31 \
    --steps data/policy/steps_v31.parquet \
    --fuel-types data/store/fuel_types.parquet \
    --cache data/policy/_feature_cache_v31.npz \
    --v2-baseline data/design/policy_v31_v2_baseline.csv \
    --holdout-cell "E1_E2/f109/ga80" \
    --xfit-k 5 --xfit-block component --calib-fold oof \
    --seeds 3 --base-seed 20260903 --epochs 120 --patience 15 \
    --stage2-lr 1e-4 --stage2-epochs 40 --lam-grid 0,0.3,1.0 \
    --teacher raw --teacher-temp 0.060 --teacher-eps 0.10 \
    --batch-size 256 --lr 1e-3 --weight-decay 1e-4 \
    --width 112 --n-blocks 6 --protocol revB \
    --gain registered --calibrate platt-oof \
    --device auto --num-workers 8 \
    --out-dir runs/policy_v31 > runs/policy_v31/train.log 2>&1
```

heartbeat 15초 · `rc` · `DONE`/`FAILED` 마커는 기존 규약 그대로이므로
`lpopt remote status/pull` 이 변경 없이 동작한다. 표는 전사하지 않고
`python -m lpopt.policy.train_v31 --tables runs/policy_v31/metrics.json` 로 렌더한다.

### 9d. arm 은 둘뿐이다 — sweep 아님

(i) **v3.1**(위 명령), (ii) **대조: 같은 코퍼스·같은 cross-fit 위에서 v3 재적합**
(`--no-burnt --lam 0 --stage2 off`). 대조가 있어야 "cross-fit 때문인가 / feature
때문인가 / listwise 때문인가"가 사후 추측이 아니라 분해가 된다. λ 격자는 arm 이
아니라 §2c 의 `val` 선택이며 게이트에는 **선택된 λ 하나만** 올라간다.

---

## 10. Leakage 규율

`FORBIDDEN_COLUMNS` 는 이미 `child_f_xy` / `d_f_xy` / `improved_fxy` / `parent_f_xy`
를 포함하고 `scalar_features_v31` 이 `scalar_features_v3` 를 호출하므로 그 가드를
승계한다. §4c 의 두 신규 열은 **(parent pattern, child pattern)** 만의 함수이며 제안
시점에 알려져 있다 — v1 의 `d_*` ring 기술자와 같은 지위다. 추가 규율 넷:

1. **Platt 사상은 `calib`(OOF)에서만 적합**한다. `val` 도 게이트 폴드도 쓰지 않는다.
2. **λ 는 `val` 단독**으로 고르고, `val` 성분은 게이트 pool 에서 제외한다(§3d).
3. **홀드아웃 셀은 K개 폴드 어디에도 들어가지 않는다**(tests §9a-H(d)).
4. burnt 2열의 스케일은 **상수로 등록**하고 폴드 안에서 적합하지 않는다(§4d).

`lineage_source` 제외 유지, parent FOM 제외 유지, `era_current` 는 셀 속성이므로
유지 — 전부 v3 §9 승계.

---

## 11. 산출물 · 무접촉 · 답하지 못하는 것

**산출:** 본 문서(동결본) · `policy_v31_results_<date>.md` ·
`lpopt/policy/{v31,train_v31}.py` · `train_policy_v31.py` ·
`data/policy/steps_v31.parquet` · `data/design/policy_v31_v2_baseline.csv` ·
`data/models/policy_v31/`(15 체크포인트 · `metrics.json` · `probs_oof.npz` ·
`calib.json` · `train.log`) · `tests/test_policy_v31.py` ·
`data/reports/_wf_scratch/v31_precensus_20260903.json` ·
`data/reports/_wf_scratch/p31j_guard_20260903.json`.

**수정하지 않는 것:** `data/policy/steps_v3.parquet`(바이트) ·
`lpopt/policy/{v3,train_v3,v2,train_v2,data,net}.py` 의 기존 동작 ·
`data/models/{policy_v2,policy_v3,s1i,s1j}` · `runs/policy_v3` ·
`intervention_wave.py` · `ablation_wave.py` · store · deck · 원격 199 / 198 / 181.

**답하지 못하는 것 (미리 적는다):**

* **`policy_v2` 우월** — 113(CI) / 326(80%) live parent 가 필요하고 이 라운드는
  41–55 를 갖는다. §6a-절2C 에 산술과 함께 등록했다.
* **`policy_v2` 와의 동등성** — sd 0.2724 에서 ±0.05 TOST 는 n ≳ 192 를 요구한다.
  이 라운드는 **비열등만** 답한다(§6a).
* **HGD569 `batch_swap`** — fresh 격자 계열을 닫았으므로 이월된다(§8).
* **burnt 2열의 참 효과** — 68쌍 중 61쌍이 한 셀이므로 이 라운드가 답하지 않는다.
  답은 r3 의 rewire-rich wave 가 **다른 셀**에서 준다(§4c 반증 판독).
* **`f113` 을 게이트에서 영구 제외할 것인가** — 두 라운드 연속 판정 불가였다는
  사실만 적고 결정은 다음 라운드로 넘긴다.
* **비보존 Gd 계열의 신경망 상 가능성** — §4a 는 ridge probe 의 null 이며 신경망
  불가능 증명이 아니다. 다만 이 n 에서 재시도할 근거는 없다고 적는다.
* **multi-step** — v3 §11 그대로. v3.1 도 1-step pairwise/listwise ranker + planner
  prior 이며 D3 연속 episode 는 여전히 없다.
* **head(s1i/s1j/arm4)의 G1–G4 within-cell 조항** — 별도 문서. 본 문서의 절 3 은
  policy net 에만 적용되며, 두 조항을 서로의 근거로 인용하지 않는다.

---

## STEP 0 freeze stamp — 2026-09-03

STEP 0-a 를 HOST_238(`USER@HOST_238:8022`, `~/lpopt_ws`, `venv`)에서 실행해
§0 fingerprint 표의 빈칸을 채운다. **이 절은 추가(append)만 하며 위 본문은 편집하지
않았다.** 실행 시점에 v3.1 가중치는 하나도 존재하지 않는다
(`data/models/policy_v31/` 없음, `runs/policy_v31*` 없음).

### S0.1 입력 지문

| artefact | SHA-256 | bytes | shape |
|---|---|---:|---|
| `data/store/records.parquet` (store 스냅샷) | `16e311af4465e735b38daf7abf999268fac27946c1c5cc279114607d9ee917ba` | 22,810,322 | 76,793 × 41 |
| `data/policy/steps_v3.parquet` (**바이트 보존 확인**) | `100ee50ed5c757257d98cb425914dc63728a16ca23e1938a732b22a06628ffcd` | 10,486,078 | 28,889 × 107 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 | 194행 |

`steps_v3.parquet` 은 빌드 전후로 sha 동일 — §11 "수정하지 않는 것" 준수.
store 스냅샷은 §0 표가 참조하던 `scratch/records_r2_76793.parquet`(`22854B72…`) 가
아니라 **현행 `data/store/records.parquet`** 이며, 행수(76,793)는 같다. 사용한 파일을
그대로 등록한다.

### S0.2 산출 코퍼스 `data/policy/steps_v31.parquet`

명령: `cd ~/lpopt_ws/src && ../venv/bin/python mine_policy_corpus.py --v31 --apply`

| 항목 | 값 |
|---|---|
| SHA-256 | `ed74a6b4cc68683075c4bf9304fd6ad32c11de49eb3d6e41be7a6730ebf589ad` |
| bytes | 10,886,412 |
| shape | **28,970행 × 111열** |
| 열이름 순서 sha-256(앞 16) | `81b43bd21494fd4b` |
| 신규 4열 | `burnt_absmov` · `burnt_absmov_r` · `burnt_slots_moved` · `burnt_token_complete` |

**등록된 기대치와의 이탈 1건.** §0 표·§4d 는 28,989행(= 28,889 + r2 100행)을
예상했다. 실현치는 **28,970행(+81)** 이다 — r2 캠페인 100 store 행이 lineage
edge 로는 **81개**만 만든다(45 parent). 나머지 19행은 parent/child edge 를 만들지
않는다. 열수 111 은 예상대로다.

### S0.3 신규 열 커버리지 (`--v31` 이 인쇄한 값 그대로)

| 판독 | 값 |
|---|---|
| `burnt_absmov` live, 라벨된 현행-era 동일-셀 rewire 행 | **74.9%** (200/267) |
| `burnt_absmov_r` live, 같은 행집합 | **97.4%** (260/267) |
| **PAIR** live, 같은 행집합 | **97.4%** (260/267) |
| `burnt_absmov` live, 전체 현행-era 동일-셀 rewire 행 | 73.7% (320/434) |
| `burnt_absmov_r` live, 같은 행집합 | 95.6% (415/434) |
| **PAIR** live, 같은 행집합 | 95.6% (415/434) |
| `burnt_slots_moved == 0` 행에서 두 열의 max\|·\| | **0.000e+00** (5,211행) — 항등 0 요구 충족 |
| `burnt_token_complete` | **100.0%** 코퍼스 |
| 코퍼스 전체 비영 행수 | `burnt_absmov` 18,888 · `burnt_absmov_r` 23,327 · `burnt_slots_moved` 23,759 |

**등록된 기대치와의 이탈 2건.** §4c-(ii) 는 "현행 era rewire 행의 **100%** 에서
살아 있는 열을 전부 넣는다"고 워딩했으나 **어느 열도 100% 에 도달하지 않는다**
(97.4% / 74.9%). 미달분은 **반경과 다중도가 같은 슬롯 사이의 burnt 교환**이며,
순수 반경 모멘트는 대칭으로 상쇄되므로 어떤 `g(r)` 로도 살릴 수 없다.
구현은 "PAIR 판독"(둘 중 하나라도 live) 으로 완화했고 그 완화가 여기서 등록된다.
`burnt_slots_moved` 는 그 미달 행의 100% 에서 live 다.
또한 rewire 가 아닌 move class 12,864/18,075 행도 burnt 를 실제로 재배치하므로
(`fresh_relocate` 6,511 · `feed_change_multi` 3,380 · `sa_unknown` 1,694 ·
`multi` 744 · `remove_fresh_unit` 355 · `add_fresh_unit` 180) 불변식의 행집합은
`~rewire` 가 아니다 — 이 사실도 등록한다.

**§4d 스케일 상수 (현행-era 동일-셀 p05/p95, 상수로 등록하고 폴드 안에서 적합하지
않는다):** `burnt_absmov` **−2.2245 / +3.2300**, `burnt_absmov_r` **−1.5662 / +2.5419**.

### S0.4 r2 병합이 게이트에 산 것 — **0** (§4d 기대 그대로)

| 항목 | 값 |
|---|---|
| 신규 edge | 81행, 전부 `fpcamp_minfxy_e1e2_f121_r2` |
| 신규 parent | 45 |
| ≥8후보 parent (코퍼스 전체) | v3 **101** → v3.1 **101** |
| live parent (코퍼스 전체) | v3 **60** → v3.1 **60** |
| **r2 가 만든 신규 ≥8후보 parent** | **0** |
| **r2 가 만든 신규 live parent** | **0** |

45 r2 parent 중 4개가 ≥8후보 집합에 들어 있으나 **넷 다 v3 코퍼스에 이미 있던
parent** 다. §4d 의 "게이트 절은 하나도 사지 않는다"가 실측으로 확인되었다.
phase-2 `pinbu_wave_minfxy_r2` 25행은 step 을 만들지 않았다(신규 edge 의 campaign
분포가 r2 단일값).

### S0.5 K=5 성분-blocked cross-fit 분할 (§3a / §3d)

명령:
```bash
cd ~/lpopt_ws/src && CUDA_VISIBLE_DEVICES=1 ../venv/bin/python -m lpopt.policy.train_v3 \
  --steps data/policy/steps_v31.parquet --holdout-cell "E1_E2/f109/ga80" \
  --xfit-k 5 --base-seed 20260903 --device cpu --out-dir data/policy/v31_split
```
(`--xfit-k` 는 EMISSION 모드이며 가중치를 하나도 학습하지 않는다.)

| artefact | SHA-256 | bytes |
|---|---|---:|
| `data/policy/v31_split/splits_v31.csv` | `a62c9937e55ad47cf42b9a09c8d5a220efa8cdccacf03409017d52fd9c951e34` | 1,564,591 |
| `data/policy/v31_split/xfit_census.json` | `c0197d318ccbcf1ee63f1e64e17395dec37c0f9cb8dbe0c9989936fff42439e6` | 2,514 |

파라미터: `k = 5` · `seed = 20260903` · `holdout_cell = "E1_E2/f109/ga80"` ·
`val_frac = 0.05`(v3.py 가 등록한 **선언된 이탈**; §3d 의 0.10 아님).

**실현 폴드 (§3d 표의 빈칸을 채운다):**

| fold | 행 | F_xy 행 | ≥8후보 | **live** |
|---|---:|---:|---:|---:|
| `prospective_cell` (`E1_E2/f109/ga80`) | 160 | 160 | 20 | **16** |
| `pool` (cross-fit, = `calib`) | 2,503 | 1,155 | **72** | **39** |
| `val` | 1,052 | 73 | 7 | 2 |
| `train` (legacy 포함) | 17,500 | 0 | 0 | 0 |

`calib_rows = 2503` (= pool 전량, §5c).

**블록별:**

| block | train 행 / F_xy / ≥8 / live | eval 행 / F_xy / ≥8 / live |
|---|---|---|
| 0 | 19,312 / 885 / 54 / 26 | 691 / 270 / 18 / 13 |
| 1 | 19,586 / 892 / 57 / 32 | 417 / 263 / 15 / 7 |
| 2 | 19,441 / 942 / 58 / 33 | 562 / 213 / 14 / 6 |
| 3 | 19,549 / 949 / 59 / 32 | 454 / 206 / 13 / 7 |
| 4 | 19,624 / 952 / 60 / 33 | 379 / 203 / 12 / 6 |

`val` 은 전 블록에서 동일(1,052행 / 73 F_xy / 7 ≥8 / 2 live)이며 어떤 블록의
`eval` 에도 들어가지 않는다. 홀드아웃 셀은 K개 블록 어디에도 없다.

**게이트 pool live = 39 이며 등록 floor `MIN_POOL_LIVE_V31 = 39` 를 정확히 만족한다
(여유 0).** §3a 헤드라인 "79/41" 은 `val` 을 깎기 **전**의 수이고, §6a-절 2B 의
검정력은 **41 이 아니라 39** 위에서 다시 진술되어야 한다 — 이 정정은 코드가 등록한
것이며 여기서 재확인한다.

### S0.6 코드 지문 (v3.1 가중치 이전 상태)

| 파일 | SHA-256 |
|---|---|
| `mine_policy_corpus.py` | `e95d376eac54d60a67b08f043448256e598713553500ed05e38fedc9c34a462d` |
| `lpopt/policy/v3.py` | `8bf60e74ff70f37dd3fc08258fd5da7a93a90cd9a2e50511a97f8dc1052e1f1a` |
| `lpopt/policy/train_v3.py` | `4effc901dca4bbc6f398f56622a6e4f8714f4cf2534ccfb750a415239ceb74e9` |
| `lpopt/policy/scorer.py` | `07517906e081b3ef2e76aa8e8c68f6319129c11070d6d04a5c7ba7a669fb90ef` |
| `lpopt/search/construct.py` | `2a44e3b44492d917de7b96c874f312672e308900161ecef2cd05c38a798bf6ab` |
| `train_policy_v3.py` | `f839b3b86a8f8c8bc730d4910f84378d9c915b2843e192b766973ba868e91545` |
| `tests/test_policy_v31.py` | `e10c52e8c3077cda252c5d1371488b96edced559b38848974bc0b86456c69863` |

로컬 `SRC` 와 238 의 위 파일들은 **sha 동일**(동기 확인 완료).
§9a 가 등록한 신규 파일명(`lpopt/policy/v31.py`, `train_v31.py`,
`train_policy_v31.py`, `scratch/v31_precensus.py`)은 **존재하지 않는다** — v3.1 델타는
`v3.py` / `train_v3.py` / `mine_policy_corpus.py` **안에 additive 플래그로**
구현되었다(`--v31`, `--stage2`, `--xfit-k`, `--lam-grid`, `*_V31` 심볼). 파일 배치의
이 이탈을 등록한다.

### S0.7 STEP 0-b 및 본 학습은 **아직 실행 불가** (등록된 차단)

* `scratch/v31_precensus.py`(§9a-G, census 6종)와
  `scratch/p31j_listwise_guard.py`(§2d 필수 파일럿)는 **238 에 존재하지 않는다.**
  따라서 STEP 0-b 는 이 스탬프에서 수행되지 않았고, §3c-(4) MDE 실측·절 3-(ii)
  셀 순위가능성 순열검정은 **미계산 상태**다.
* §9c 의 본 학습(`--stage2 on`)은 **구현이 스스로 거부한다**
  (`assert_v3_path_untouched`): `--stage2 on` 단독은 v3 의 단일 분할(37/16, §3a 가
  REJECTED 로 등록한 arm) 위에서 학습하게 되므로 거부되고, `--stage2 on --xfit-k 5`
  는 emission 분기가 먼저 반환하므로 거부된다. 해제 조건은 코드가 명시한다:
  **prereg 델타 D — §5d 의 네 베이스라인을 각 블록의 train 폴드에서 재적합해
  out-of-fold 점수열을 K개 적합으로 잇는 지표 모듈** 이 착지해야 한다.
* 따라서 **본 스탬프 시점에 v3.1 가중치는 존재하지 않으며, 존재할 수 없다.**
  §3b 의 블라인드 v2 베이스라인 재발행(`--emit-v2-baseline`)은 지금 실행 가능하고
  실행 순서상 여기가 옳은 자리다.

---

## STEP 0-b stamp — 2026-09-03

STEP 0-b(§9a-G 의 census 6종), §2d 의 필수 파일럿, §3c-(4) MDE 실측, §3b 블라인드
v2 베이스라인 재발행, 그리고 **prereg 델타 D**(§5d/§6 지표 모듈)를 HOST_238 에서
착지·실행했다. **이 절도 추가(append)만 하며 위 본문과 STEP 0 freeze stamp 는
편집하지 않았다.** 실행 시점에 v3.1 가중치는 여전히 하나도 존재하지 않는다
(`data/models/policy_v31/` 없음, `runs/policy_v31*` 없음).

### S0b.1 델타 D 착지 — `lpopt/policy/metrics_v31.py`

§S0.7 이 등록한 차단 해제 조건이 충족되었다. 모듈은 §5d 의 네 베이스라인
(`random`/`class_freq`/`periph`/`gd_rule`)을 **각 블록의 train 폴드에서 재적합**하고
out-of-fold 점수열을 K개 적합으로 잇는다. 적합 코드는 재구현이 아니라 v3 결과가
쓴 그 코드(`train_v3.baseline_scores_v3` · `gd_rule_sign`)를 **호출**한다.
`policy_v2` 는 적합하지 않고 §3b 의 블라인드 CSV 로 채점된다(§5d 그대로).

**재적합이 실제로 블록마다 다르다는 실측 증거** — 다섯 블록의 `gd_rule` 부호:

| block | fit 행 / 라벨 | fit base rate | `gd_rule` 부호 | eval 행 |
|---:|---|---:|---:|---:|
| 0 | 19,312 / 885 | 0.1729 | **−1** | 691 |
| 1 | 19,586 / 892 | 0.1771 | **+1** | 417 |
| 2 | 19,441 / 942 | 0.2038 | **+1** | 562 |
| 3 | 19,549 / 949 | 0.2002 | **+1** | 454 |
| 4 | 19,624 / 952 | 0.2048 | **−1** | 379 |

부호가 블록에 따라 뒤집힌다는 것은 "K개 적합"이 이름이 아니라 사실이라는 뜻이며,
단일 적합으로 채점했다면 블록 0·4 의 eval 행 1,070개가 **자기 라벨이 정한 부호로**
채점되었을 것이다.

**차단 해제는 두 조건의 연언이고 코드가 둘 다 검사한다**
(`train_v3.delta_d_status`): (i) 모듈이 import 되고, (ii) `--splits` 가 §S0.5 의
동결 배정 sha256 `a62c9937…` 을 가리킬 것. 실측:

```
--stage2 on --xfit-k 5 --splits data/policy/v31_split/splits_v31.csv
  -> {'enabled': True, 'version': 'v31', 'crossfit': True}
--stage2 on --xfit-k 5                       (--splits 없음)  -> REFUSED
--stage2 on --xfit-k 5 --splits <다른 바이트>                  -> REFUSED
--stage2 on            (--xfit-k 없음)                        -> REFUSED (영구)
```

`--xfit-k` 는 `--splits` 가 있을 때만 **소비**이고, 없으면 여전히 EMISSION 이다.
배정 재산출 검증: 동일 명령으로 `scratch/v31/emit/splits_v31.csv` 를 다시 만들면
sha256 이 `a62c9937…` 로 **바이트 동일** — §S0.5 는 재현 가능하다.

**ECE ≤ 0.05 는 게이트하지 않는다** — §5c 가 등록한 대로. 모듈은 ECE·Brier 를
계산해 `calibration.gated = false` 와 그 사유 문자열을 함께 싣는다. 게이트되는
서빙 조항은 절 4 의 **폭**뿐이다.

### S0b.2 census 1 — 실현 폴드 (§3d, 동결 배정 위에서 재계산)

| fold | 행 | F_xy 행 | ≥8후보 | live | cell |
|---|---:|---:|---:|---:|---:|
| `prospective_cell` | 160 | 160 | 20 | **16** | 1 |
| `pool` (= `calib`) | 2,503 | 1,155 | **72** | **39** | 56 |
| `val` | 1,052 | 73 | 7 | 2 | 35 |
| `train` (legacy) | 17,500 | 0 | 0 | 0 | 32 |

**§S0.5 와 완전히 일치한다.** 게이트 pool live = 39 = 등록 floor
`MIN_POOL_LIVE_V31`, 여유 0.

### S0b.3 census 2 — 셀 live

| cell | fold | 행 | F_xy | ≥8후보 | **live** |
|---|---|---:|---:|---:|---:|
| `E1_E2/f109/ga80` | holdout | 160 | 160 | 20 | **16** |
| `E1_E2/f121/ga80` | pool | 422 | 410 | 21 | **10** |
| `N1_N2/f113/ga80` | pool | 151 | 150 | 17 | **11** |
| `T6_T4/f121/paramA` | pool | 873 | 244 | 16 | 8 |
| HGD569 `f125/paramA` | pool | 136 | 136 | 13 | 7 |
| `E1_E2/f125/ga80` | pool | 163 | 67 | 3 | 2 |
| `G3_G4/f125/ga80` | pool | 136 | 87 | 2 | 1 |
| `E1_E2/f117` · `J5_J6/f121` · `G3_G4/f121` · `N3_N4/f113` · HGD 3-type | pool | — | 2–26 | 0 | 0 |

§6a 절 3 의 오늘 예상(f113 12 live · f121 10 live)에 대해 실현치는 **11 / 10**.

### S0b.4 census 3 — 셀 순위가능성(절 3-ii)과 §3c-(4) 실측 MDE

다섯 베이스라인의 **parent 내 순열 귀무**(2,000 draw, v3.1 점수 미사용):

| cell | live | min p (5 베이스라인) | 순위가능 | **MDE 실측(screen, 1.96·sd)** | MDE(80% 검정력, 2.4866·sd) | 등록치 |
|---|---:|---:|---|---:|---:|---:|
| `E1_E2/f109/ga80` (holdout) | 16 | **0.0005** | ✅ | **0.1584** | 0.2010 | 0.160 |
| `E1_E2/f121/ga80` | 10 | **0.0035** | ✅ | 0.1917 | 0.2432 | 0.174(어림) |
| `N1_N2/f113/ga80` | 11 | **0.0245** | ✅ | 0.1788 | 0.2269 | 0.142 |
| `T6_T4/f121/paramA` | 8 | 0.0055 | ✅ | 0.2278 | 0.2891 | 0.194(어림) |
| HGD569 `f125/paramA` | 7 | 0.0080 | ✅ | 0.2414 | 0.3063 | 0.227 |
| `E1_E2/f125/ga80` | 2 | 0.0185 | ✅ | 0.4890 | 0.6204 | 0.39(어림) |
| `G3_G4/f125/ga80` | 1 | 0.2284 | ❌ | 0.6829 | 0.8664 | 0.55(어림) |
| `E1_E2/f117` · `J5_J6/f121` | 0 | — | — | — | — | — |

**절 3 자격 셀(live ≥ 10 그리고 순위가능) = `E1_E2/f121`(10) · `N1_N2/f113`(11), 둘.**

> **등록된 기대와의 이탈 1건 — 결과 전에 적는다.** §6a 절 3 은 "(ii) 는 f113 에서
> 통과하지 못할 가능성이 높다"고 적었다(results §3.4: 여섯 채점자 전부 우연 수준).
> **실측은 그 반대다: f113 은 p = 0.0245 로 순위가능하며 자격 셀이다.** 따라서 §6a
> 가 준비해 둔 "f113 은 판정 불가로 기록하고 FAIL 로 세지 않는다"는 조항은
> **발동하지 않는다** — f113 은 판정된다. 이 사실이 게이트 숫자가 존재하기 전에
> 기록되었다는 것이 요점이다. (results §3.4 는 **v3 의 단일 분할 게이트 폴드**에서
> 잰 것이고 여기는 cross-fit pool 의 out-of-fold 열이므로 행집합이 다르다.)
>
> **MDE 규약(등록).** §3c-(4) 의 `mde_screen` 은 순열 귀무 평균의 **1.96·sd**
> (양측 95% 검출선)이며, 그 규약으로 f109 는 **0.1584**(등록 0.160, 오차 1%)로
> 재현된다 — 조건 (4) `≤ 0.20` 은 계속 만족된다. 80% 검정력 규약(2.4866·sd)에서는
> 0.2010 이며 **두 수는 같은 측정의 두 규약**이다. 결과 문서는 등록 표와 비교할 때
> `mde_screen` 열을 쓴다. f113 의 0.1788 vs 등록 0.142 차이는 live 12 → 11 의
> 행집합 차이다.

### S0b.5 census 4 — r2 병합이 게이트에 산 것: **0** (§S0.4 재확인)

| 항목 | v3 코퍼스 | v3.1 코퍼스 |
|---|---:|---:|
| ≥8후보 parent (전체 parquet) | **101** | **101** |
| live parent (전체 parquet) | **60** | **60** |
| ≥8후보 / live (라벨 필터 universe) | 99 / 57 | 99 / 57 |

신규 edge 81행 · 신규 parent 45 · campaign 전부 `fpcamp_minfxy_e1e2_f121_r2`.
**산 게이트 parent = ge8 +0, live +0.** §4d·§S0.4 그대로다.

### S0b.6 census 5 — burnt 열 커버리지 (§S0.3 재판독)

전체 parquet 28,970행 기준(라벨 필터 universe 는 21,215행이며 두 행집합을 섞지
않는다). "rewire" 는 클래스 **가족**(`rewire_swap` + `rewire_multi`)이다.

| 판독 | 실측 | §S0.3 등록 |
|---|---:|---:|
| 현행-era rewire 행 | **434** | 434 ✅ |
| 그중 라벨된 행 | **267** | 267 ✅ |
| `burnt_absmov_r` live, 라벨된 267행 | **97.38%** | 97.4% ✅ |
| **PAIR** live, 라벨된 267행 | **97.38%** | 97.4% ✅ |
| `burnt_absmov` live, 라벨된 267행 | **80.15%** (214/267) | **74.9%** (200/267) ⚠️ |
| `burnt_slots_moved == 0` 행에서 max·절대값 | **0.000e+00** (5,211행) | 0.000e+00 (5,211행) ✅ |
| 코퍼스 비영 행수 (`absmov`/`absmov_r`/`slots_moved`) | **18,888 / 23,327 / 23,759** | 18,888 / 23,327 / 23,759 ✅ |
| `burnt_token_complete` | **100.0%** | 100.0% ✅ |

**이탈 1건(등록).** 분모(267)와 다른 모든 판독이 바이트 수준으로 일치하는데
`burnt_absmov` 의 분자만 200 → **214** 다. 코퍼스가 아니라 "live" 판정의 정의
차이이며(여기서는 `nan→0` 후 `≠ 0`), **어느 쪽도 §4c-(ii) 의 100% 워딩에 도달하지
않는다**는 §S0.3 의 결론은 바뀌지 않는다. 결과 문서가 두 판정을 화해시킨다.

### S0b.7 census 6 · §3b 블라인드 v2 베이스라인 재발행

**v3.1 가중치가 하나도 존재하지 않는 시점에** 재발행했다(§3b 의 순서 규율).

| 항목 | 값 |
|---|---|
| 경로 | `data/design/policy_v31_v2_baseline.csv` |
| **SHA-256** | `942a45bfc56e2f3829776802abd618e5ecfce4b6a739b6d88e7c3f4ece8fa858` |
| bytes / 행 | 706,262 / **3,715** (+헤더) |
| 앙상블 | `data/models/policy_v2` 5 멤버, 스탬프 `policy_move_v2` / v2 / (ga80, paramA) |
| `pool` 커버리지 | **1.000** |
| `prospective_cell` 커버리지 | **1.000** |

현행 `policy_v3_v2_baseline.csv`(3,286행)는 **바이트 무접촉**이다.

> **실행상의 이탈 1건(등록).** 238 의 `runs/policy_v2` 체크포인트는 서빙 스탬프가
> 없어(`policy_schema = None`) `emit_v2_baseline` 이 등록대로 **거부**했다. 스탬프된
> 승격본은 로컬 `SRC/data/models/policy_v2` 에만 있었으므로 그 5개 체크포인트를
> 238 로 전송하고(모델 바이트 md5 동일 확인) 거기서 채점했다. 계산은 전부 238 에서
> 일어났고, 스탬프 강제는 우회하지 않고 **충족**시켰다.

### S0b.8 §2d 필수 파일럿 — raw 교사 × head-guard

`scratch_v31/p31j_listwise_guard.py`, GPU 1, λ=1.0, 2 seed(20260831–2), 120 epoch /
patience 15, 843 s. listwise 그룹 **57 parent / 368행** — §2b 가 등록한 수 그대로.

| 칸 | 교사 | gradient | `fxy` pb-AUC | `fr` pb-AUC |
|---|---|---|---:|---:|
| pilot A | raw | full net | 0.8371 | — |
| e1 | 등록 | head guard | 0.6869 | 0.6494 |
| e2 | 등록 | full net | 0.7029 | 0.6819 |
| **p31j (본 파일럿)** | **raw** | **head guard** | **0.8115** | **0.7897** |

(gate_cur 폴드, 2-seed 앙상블, 라벨 540 / 1,137행, mixed pair 313 / 1,355.
`flat` 0.7187. `prospective_cell`(f113) 은 fr 0.4467 / flat 0.5052 / fxy 0.5461 로
세 head 모두 우연 수준 — results §3.4 의 f113 판독과 같은 방향이다.)

**판독.** 비어 있던 칸이 채워졌고, 교락이 풀린다: **`fxy` 붕괴를 일으킨 것은
head-guard 가 아니라 등록-gain 교사다**(raw×head 0.8115 vs 등록×head 0.6869,
Δ = +0.125; raw×full 0.8371 과는 −0.026). 그리고 raw 교사에서는 **`fr` 이 0.7897
로 등록 회귀 바닥 0.678 위**에 있다 — 등록 교사의 0.6494 는 그 아래였다.

**§2d 가 등록한 대로 이 결과는 stage-2 구조를 바꾸지 않는다.** 동결 분기는 절 5 를
**구조적으로** 만족시키기 위한 것이고(§2b), 파일럿에서 고른 것이 아니다. 이 칸이
말하는 것은 "head-guard 가 필요했는가"가 아니라 **"head-guard 는 무해하다"** 이며,
그 구별을 사후가 아니라 여기서 적는다.

### S0b.9 코드 지문 (여전히 v3.1 가중치 이전)

로컬 `SRC` 와 238 `~/lpopt_ws/src` 는 아래 전부 **sha 동일**(전송 후 확인).

| 파일 | SHA-256 |
|---|---|
| `lpopt/policy/metrics_v31.py` (신규, **델타 D**) | `e4fb88adad5f0ef3b88896bda7a8b6bde5f6b16916f1e7766e1732d939afb2fd` |
| `lpopt/policy/train_v3.py` (배선·차단 해제·cross-fit 루프) | `c064421d2eaeb642864cb2b5ffb0dcdf0f49777f3b24109a7f784ceafdb65082` |
| `train_policy_v3.py` (`--v31` 명령 렌더) | `ff2cd80a09abbd459626549c116153b5f39686f8cd79540dcd15a0bae1f9ace5` |
| `tests/test_metrics_v31.py` (신규) | `64f2cbd568084f00a8759a0517f22db14f834b54c7b9f3243bbe216207b830a9` |
| `tests/test_policy_v31.py` | `17192141a41e6c215fb25f7a03403449a33497f24eaea39ecec7de6c9f3e984f` |
| `scratch_v31/v31_precensus.py` (§9a-G) | `04c7e310d26196470bc49ed28569006010946ded98a690e392479ce7230d4062` |
| `scratch_v31/p31j_listwise_guard.py` (§2d) | `07ad66b533b7ff6e6179131b8ae922309d42437b571db39b8740c56ca149382e` |
| `data/design/policy_v31_v2_baseline.csv` (§3b) | `942a45bfc56e2f3829776802abd618e5ecfce4b6a739b6d88e7c3f4ece8fa858` |

`lpopt/policy/v3.py` · `lpopt/policy/scorer.py` · `lpopt/search/construct.py` ·
`mine_policy_corpus.py` · `data/policy/steps_v3.parquet` ·
`data/policy/steps_v31.parquet` · `data/policy/v31_split/splits_v31.csv` ·
`data/design/policy_v3_v2_baseline.csv` 는 §S0.1/§S0.2/§S0.5/§S0.6 의 sha 그대로
**무접촉**이다.

시험(238, `../venv/bin/python -m pytest -q`): `test_metrics_v31.py` **26 passed** ·
`test_policy_v31.py` **43 passed** · `test_policy_v3.py` **40 passed** — v3 경로
회귀 없음.

### S0b.10 이제 실행 가능한 명령 (등록본; 본 스탬프 시점에 **실행하지 않았다**)

```bash
# 본 학습 — arm i (v3.1).  λ 격자는 arm 이 아니라 val 선택이다(§2c/§9d).
CUDA_VISIBLE_DEVICES=1 ~/lpopt_ws/venv/bin/python -m lpopt.policy.train_v3 \
    --steps data/policy/steps_v31.parquet \
    --fuel-types data/store/fuel_types.parquet \
    --cache data/policy/_feature_cache_v31.npz \
    --v2-baseline data/design/policy_v31_v2_baseline.csv \
    --holdout-cell "E1_E2/f109/ga80" \
    --xfit-k 5 --splits data/policy/v31_split/splits_v31.csv \
    --stage2 on --lam-grid 0,0.3,1.0 \
    --teacher raw --teacher-temp 0.060 --teacher-eps 0.10 \
    --stage2-lr 1e-4 --stage2-epochs 40 \
    --seeds 3 --base-seed 20260903 --epochs 120 --patience 15 \
    --batch-size 256 --lr 1e-3 --weight-decay 1e-4 \
    --width 112 --n-blocks 6 --protocol revB \
    --device auto --num-workers 8 \
    --out-dir runs/policy_v31 > runs/policy_v31/train.log 2>&1

# 로컬 런처(ship -> launch -> poll -> pull)로도 같은 문자열이 렌더된다
python train_policy_v3.py --v31 --ts policy_v31 --seeds 3 --base-seed 20260903
python train_policy_v3.py --v31 --print-command        # 문자열만 인쇄
```

**등록된 이탈 2건.**

1. §9c 는 `python -m lpopt.policy.train_v31` 을 등록했으나 §S0.6 이 이미 적었듯
   v3.1 델타는 `train_v3.py` 안에 additive 플래그로 구현되어 있다. 위 명령이
   실제로 실행 가능한 문자열이다.
2. §9d 의 **arm ii(대조: `--no-burnt --lam 0 --stage2 off`)는 렌더하지 않는다** —
   `--no-burnt` 는 존재하지 않고, `featurize_round` 가 스탬프에서 feature 계약을
   강제하므로 53-scalar 코퍼스를 51-scalar 경로로 흘릴 수 있는 CLI 가 없다.
   런처는 이 사실을 인쇄하며, 조용히 플래그를 빠뜨린 다른 비교를 대신 실행하지
   않는다. §9d 의 3-way 분해는 그 플래그가 착지한 뒤에야 가능하다.

### S0b.11 산출물

`data/reports/v31_step0b/` — `v31_precensus_20260903.json`(census 6종 전부) ·
`p31j_guard_20260903.json`(§2d) · `precensus.log` · `p31j.log` · `emit_v2.log`.
238 원본은 `~/lpopt_ws/scratch/v31/`.

**본 스탬프 시점에 v3.1 가중치는 여전히 존재하지 않는다.** STEP 0-b 와 §3b 는
끝났고, §9c 본 학습은 이제 차단되지 않지만 **실행되지 않았다.**
