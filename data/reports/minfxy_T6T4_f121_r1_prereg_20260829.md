# `min_fxy` 캠페인 ROUND 1 — `T6_T4` / feed 121 — PRE-REGISTRATION

- 작성일: **2026-08-29**
- 대상: `lpopt` (APR1400 equilibrium-cycle loading-pattern optimizer), BOX 199
- 성격: **사전등록(pre-registration).** 본 문서 작성 과정에서 코드 수정 **없음**, 원격 접속 **없음**,
  MASTER 호출 **없음**. 산출물은 문서 + deck + launcher trio 뿐이다.
- 근거 결정: 사용자 결정 2026-08-29(구속력 있음) — **최적화 대상 `F_r` → `F_xy` (MASTER `FXYP`)
  전환, hard limit `F_xy ≤ 1.65`. `F_r ≤ 1.55`는 constraint로 유지.**
- 설계서: `data/reports/fxy_switch_design_20260829.md` (§3 search, §4 실행순서 **P5**)
- 이 문서가 사전등록하는 실행: `python -m lpopt optimize --input fpcamp_minfxy_T6T4_f121_r1_199.inp`

> **이 캠페인은 프로그램 최초의 `min_fxy` 라운드다.** 목적함수 전환의 baseline이며,
> `policy_prior = "off"` — 학습 policy A/B(§10)는 **이 라운드에 포함하지 않는다.**

---

## 0. 근거 규칙

- 코드에 대한 모든 주장은 해당 파일을 직접 읽어 확인했다. 파일/함수/라인을 본문에 명시한다.
- 수치는 **실측(measured)** 과 **추정** 을 구분한다. 표기가 없으면 실측이다.
- 실측의 출처는 2026-08-29 시점의 `data/store/records.parquet` (**74,717행**,
  sha256 `0334E2D2…` — §9)이다. 재현 가능한 필터 조건을 본문에 함께 적는다.
- 본 문서에서 **joint-clean** 은 `converged=True & valid=True` 이면서
  `cbc_max ≤ 1600 & f_q ≤ 2.41 & |ao_abs| ≤ 0.30 & f_r ≤ 1.55` 를 만족하는 행을 뜻한다.
  `f_xy` 라벨이 있는 행만 센다.
- 원격 호스트는 `HOST_199`(캠페인 실행 박스)로만 부른다.

---

## 1. 왜 지금, 왜 이 셀인가

### 1.1 목적함수 전환은 실제로 답을 바꾼다 — 이 셀에서 측정된다

`T6_T4/f121`은 `min_fr` 캠페인을 여덟 라운드(r1~r8) 돌린 셀이고, 그 라운드들이 남긴
`f_xy` retro-label이 프로그램에서 가장 많다. 그 라벨을 라운드별로 읽으면 **F_r 추격이
F_xy를 단조적으로 악화시켰다**는 것이 바로 보인다 (실측, 셀 내 `f_xy`-라벨 행 기준):

| 라운드 | `f_xy` 라벨 | 라운드 내 **min F_xy** | 라운드 내 **min F_r** |
|---|---:|---:|---:|
| `fpcamp_minfr_T6T4_r3` | 96 | **1.5491** | 1.4984 |
| `fpcamp_minfr_T6T4_r5` | 97 | 1.5549 | 1.4812 |
| `fpcamp_minfr_T6T4_r4` | 94 | 1.5688 | 1.4866 |
| `fpcamp_minfr_T6T4_r6` | 97 | 1.5829 | **1.4797** |

F_r은 `1.4984 → 1.4797` (**−0.0187**) 개선되는 동안 F_xy는 `1.5491 → 1.5829`
(**+0.0338**) 나빠졌다. 즉 **F_r을 목적함수로 삼은 탐색은 F_xy를 적극적으로 밀어냈다.**
같은 사실의 단면: 이 셀의 joint-clean **F_r 최소 core**
(`9b9fabe8…`, F_r 1.4797, CBC 1338.61, cyclen 623.59)의 측정 `F_xy = 1.5829`로,
새 축에서는 **12위권**이다. 목적함수 전환은 순위를 실제로 재배열한다.

**등록된 미지수 하나.** 이 셀의 *진짜* F_r 기록(1.4605, `batchswap_enum_T6T4`)과
r7·r8·`ablation_1move_T6T4`의 행들은 **`f_xy` 라벨이 0건**이다(MAS_OUT 미보존). 따라서
"F_r 기록 core의 F_xy"라는 진술은 **라벨이 존재하는 1.4797 core에 한정**되며, 1.4605
core의 F_xy는 **측정된 바 없다**. 이 라운드는 그 공백을 메우지 않는다.

### 1.2 `min_fr` 예산과 충돌하지 않는다

이 셀의 `min_fr` 단일셀 추격은 **이미 종결**되었다: `ab2_addendum_S1E_20260815.md` §8 —
r8 in-band 이득 `1.4797 − 1.4749 = 0.0048 < 0.0050` 로 사전등록된 close-out 조항이 발동했고,
"셀이 바닥에 근접"으로 선언되었다. 따라서 이 셀에 100 call을 쓰는 것은 진행 중인 F_r
프로그램의 예산을 빼앗지 않는다.

### 1.3 셀 결정 — `T6_T4/f121` (r1), `E1_E2/f121` (r2)

설계서 §4/P3의 권고를 따르되, **반대 증거를 먼저 적는다.**

| 항목 (실측, 2026-08-29 store) | **`T6_T4/f121`** | `E1_E2/f121` |
|---|---:|---:|
| `f_xy` 라벨 (전체 / converged·valid) | **746 / 738** | 654 / 654 |
| joint-clean 라벨 행 | 215 | **413** |
| joint-clean 중 `F_xy ≤ 1.65` | 174 | **348** |
| **셀 incumbent min F_xy** | 1.5491 | **1.5295** |
| incumbent record_id | `46e687ed…` | `a785eded…` |
| joint-clean cyclen 범위 (EFPD) | 614.66 – 627.34 | 629.91 – 639.80 |
| joint-clean CBC 최대 (ppm) | 1383.21 | 1369.38 |
| corr(F_xy, F_r) — joint-clean | **0.282** | 0.737 |
| 셀 내 `min_fr` 캠페인 상태 | **종결(S1E §8)** | 진행 이력 있음 |

**반대 증거 (기록해 둔다).** joint-clean 물량과 incumbent 품질은 모두 `E1_E2/f121`이 낫다.
프로그램 전체의 joint-clean **F_xy 최소값 1.5295는 이 셀이 아니라 `E1_E2/f121`에 있다.**

**그럼에도 r1을 `T6_T4/f121`로 정한 근거 (셋 다 이 라운드에서만 성립):**

1. **라벨 밀도.** 746건은 프로그램 최다이며, `min_fxy`의 elite pool은 **측정된 F_xy로
   정렬**된다(§7). elite pool의 품질이 곧 탐색의 출발점이므로, 이 셀은 라벨 기반 elite가
   가장 두껍다. 셀별 prior `(a, b)`를 **적합하고 검증할 수 있는 유일한 셀**이기도 하다.
2. **예산 비충돌.** §1.2. `E1_E2/f121`은 F_r 프로그램의 활성 셀이었다.
3. **가장 날카로운 사전등록 질문.** §1.1의 라운드별 역행이 이 셀에서만 측정되어 있다.
   "F_r 추격이 밀어낸 0.0338을 `min_fxy`가 되찾는가"는 이 셀에서만 falsifiable하다.
   그리고 **corr(F_xy, F_r) = 0.282** 라는 낮은 상관은, proxy arm(§5)이 실제로 얼마나
   눈먼지를 가장 가혹하게 시험한다 — `E1_E2/f121`의 0.737에서는 proxy가 우연히 잘 맞아
   전환의 가치가 과대평가될 위험이 있다.

**`E1_E2/f121`은 r2로 지명한다.** r1 종료 후 별도 deck·별도 사전등록으로 진행하며,
r1의 결과를 top-up 하는 방식은 금지한다(TRIPLE r1 §9의 선례와 동일 규칙).

**교차 인용 규칙 (등록).** r1의 결과는 **셀 내 비교로만** headline 이 된다. 1.5295 이하를
내지 못하면 "프로그램 기록"이라고 부를 수 없다. §2의 STRETCH가 정확히 이 조건이다.

---

## 2. 마크 — launch 전에 못 박는다

### 2.1 왜 "`F_xy ≤ 1.65`"가 PRIMARY가 될 수 없는가

과제 지시는 "`F_xy ≤ 1.65` 최초 deliverable-grade core"를 PRIMARY로 제안했다.
**실측이 그것을 기각한다:**

- store 전체에서 `f_xy ≤ 1.65` 인 행은 **1,095건**이다.
- joint-clean(converged·valid + 4축) `f_xy` 라벨 행 **1,089건 중 948건(87.1%)** 이
  이미 `F_xy ≤ 1.65` 다.
- `T6_T4/f121` 만 보면 joint-clean 215건 중 **174건(80.9%)** 이 이미 통과한다.
- 그 215건은 **전부** 아래 §6의 report-only cyclen band(611.3–631.3 EFPD) 안에 있다.

즉 `1.65` gate는 이 셀에서 **slack** 이며, 그것만으로는 아무것도 구별하지 못한다.
따라서 PRIMARY는 **셀 incumbent를 마진 이상으로 이기는 것**으로 정의한다.

### 2.2 마크 표

| mark | 요구조건 |
|---|---|
| **PRIMARY (search)** | MASTER-verified core로 **측정 `F_xy ≤ 1.5441`** (= 셀 incumbent 1.5491 − 등록 마진 0.005) **이고** 나머지 gate가 전부 측정치로 clean: `F_r ≤ 1.55`, `CBC ≤ 1600`, `F_q ≤ 2.41`, `\|AO\| ≤ 0.30`, **예측** pin BU ≤ 78. 0.005 마진은 이 셀의 close-out 조항이 쓴 임계값과 동일한 수치이며(§1.2), 같은 셀에서 "미미한 이득을 진보로 포장"하는 것을 막기 위해 재사용한다. |
| **PRIMARY (delivery)** | 위 core가 **phase-2 `pinbu_wave`** 에서 **측정** pin axial peak ≤ 80 GWd/tU 를 받는 것. **이 라운드 단독으로는 도달 불가**하며(§5.3), 그것은 결함이 아니라 사전등록된 구조다. |
| **STRETCH** | 측정 `F_xy ≤ 1.5295` — 프로그램 전체 joint-clean 최소값(`E1_E2/f121`, `a785eded…`)을 **셀을 옮기지 않고** 넘어서는 것. 이것에 도달해야만 "프로그램 기록"이라고 부를 수 있다. |
| SECONDARY | 측정 `F_xy < 1.5491` (마진 없이 incumbent를 이김). 라벨을 붙여 보고하되 **headline 금지** — 0.005 미만 이득은 §1.2 close-out 논리상 진보로 셈하지 않는다. |
| **NULL — 이 라운드가 실제로 예상하는 결과** | 100 call 안에 `F_xy ≤ 1.5441` core가 나오지 않음. **그대로 출판 가능한 결과다.** 벽의 귀속은 §4에서 미리 고정한다 — 사후에 고를 수 없다. |
| PIN | phase-2에서 회수한 측정 pin BU를, r1의 F_xy 상위 core에 대해 licensing 80 및 기존 측정대역과 대조. |
| **PARITY (최적점)** | §2.3. |

### 2.3 최적점 parity 마크 (예측 vs MASTER)

r1 종료 시 `state.json → best_overall` core 1건에 대해, wave 선택 시점의 예측과 MASTER
측정을 대조한다. 등록된 수용 기준(프로그램 수용선):

| 축 | 기준 | 적용 arm |
|---|---|---|
| `cyclen` | \|Δ\| ≤ **1 EFPD** | 두 arm 모두 **gated** |
| `cbc_max` | \|Δ\| ≤ **1 ppm** | 두 arm 모두 **gated** |
| `f_xy` | \|Δ\| ≤ **0.01** | **head arm(s1j)만 gated.** proxy arm(s1i)은 **보고만 하고 gate하지 않는다** — proxy는 예측이 아니라 F_r 회귀이며, 그 σ는 `K=3.0`으로 부풀려져 있다(§5.1). proxy arm의 \|Δ\| 를 parity 실패로 읽는 것은 범주 오류다. |
| `f_r` | 보고만 (constraint 축, 목적 축 아님) | 두 arm |

parity 실패는 캠페인 실패가 아니라 **모델 캘리브레이션 관측치**로 기록한다. 단
head arm에서 `f_xy` parity가 깨지면 **§8-C 표가 r2의 gating 항목**이 된다.

---

## 3. 목적함수와 gate — 코드에서 확인한 그대로

`objective = "min_fxy"` (`campaign.py:624-666`, `acquisition.py:730-1000`).

- **acquisition exploit scalar**: `cyclen_LCB − minfxy_lambda · F_xy_UCB`
  (`score_min_fxy`), `risk_z = 0.25` 로 cyclen은 LCB, F_xy는 UCB — 양방향 보수적.
- **verified best-tracking**: `_campaign_objective` 의 `−f_xy·1e6 + cyclen`
  (`campaign.py:1751-1760`) — 엄격한 사전식 (F_xy asc, cyclen desc).
  **측정 `f_xy`가 없는 수렴 행은 "더 나쁜" 것이 아니라 `−inf`(UNSCORABLE)** 이다.
- **hard tier**: `F_r / F_q / CBC / |AO|` 는 UCB 초과분, 예측 pin BU는
  `max(0, pin_UCB − 78)`, F_xy는 `max(0, F_xy_UCB − 1.65)/0.01`. 제곱합에
  `_MAXCYCLE_CONSTRAINT_TIER` 를 곱해 감산하므로, 예측 위반은 모든 feasible 후보 아래로
  가라앉되 "덜 위반한 순"으로 정렬된다.
- **`fxy_std` 가 non-finite면** penalty는 0이지만 `constraint_ok = False` — 측정되지 않은
  licensing 축을 조용히 feasible로 부르지 않는다.

### 3.1 `harvest_maps` 는 knob 이 아니라 전제다

F_xy(=`FXYP`)는 **`MAS_SUM`에 없다.** 유일한 출처는 마지막 equilibrium cycle의
`MAS_OUT`이고, 그 파일은 `harvest_maps` 가 verifier의 `keep_success` 를 켤 때에만 살아남는다
(`verify.WaveVerifier`). **별도의 `keep_success` deck key는 존재하지 않는다** — 과제 지시의
"keep_success on"은 `harvest_maps = true` 로 **함의**되는 것이지 따로 켜는 값이 아니다.
`harvest_maps = false` 면 `CampaignDriver` 가 **생성 자체를 거부**한다
(`campaign.py:646-654`). launcher는 deck 해시 gate 위에 `harvest_maps = true` 문자열 gate를
하나 더 얹는다(중복이지만, 이 축은 실패하면 라운드 전체가 무의미해진다).

---

## 4. 등록된 감시 — 어느 벽인지 사후에 고를 수 없다

NULL(§2.2)이 나왔을 때 **허용되는 귀속은 아래 셋뿐이며, 판정 기준을 지금 고정한다.**

### 4.1 CBC 벽 — 이 셀에서는 **예상 벽이 아니다** (실측)

`min_fr` TRIPLE 라운드의 NULL 가설은 "F_r이 1.55에 닿기 전에 CBC가 1600 벽에 먼저 부딪힌다"
였고, 그 셀에서는 frontier가 1597.33(여유 2.67 ppm)이라 실제로 near-binding 이었다.
**이 셀은 다르다:**

- joint-clean CBC 최대 = **1383.21 ppm** → gate까지 **216.79 ppm** 여유
- 1600에서 20 ppm 이내 행 = **0건**
- corr(F_xy, CBC) = **−0.175** (joint-clean, n=215) — 약하고, 부호도 "F_xy를 낮추면 CBC가
  오른다"는 압력을 시사하지 않는다.

> **등록 판정.** NULL을 CBC 벽으로 설명하려면, r1의 F_xy 상위 10 core의 CBC가
> **1550 ppm 이상**이어야 한다. 그렇지 않으면 CBC 귀속은 **금지**한다.

### 4.2 F_r 벽 — 실측상 현재 binding 아님

이 셀에서 `f_xy` 라벨이 있는 전체 738행 기준, **F_xy 하위 30 core 중 `F_r > 1.55` 는 0건,
`CBC > 1600` 0건, `F_q > 2.41` 0건, `|AO| > 0.30` 0건**이다. 즉 오늘 알려진 F_xy frontier는
전부 joint-clean 내부에 있고, **제약이 F_xy를 막고 있다는 증거는 없다.**

> **등록 판정.** NULL을 F_r 벽으로 설명하려면, r1에서 **F_xy가 더 낮지만 `F_r > 1.55`라서
> 탈락한 core가 3건 이상 실측**되어야 한다. 없으면 F_r 귀속은 **금지**한다.

### 4.3 Proxy 실명(blindness) — 이 라운드의 **주 가설**

joint-clean slice에서 corr(F_xy, F_r) = **0.282**, 그 slice의 셀별 회귀는
`F_xy = 0.4811·F_r + 0.8904` (resid sd 0.0300, n=215) — **기울기가 0.48**이다.
전역 proxy가 쓰는 기울기는 **1.2176**이다. 즉 `F_r ≤ 1.55` 안쪽 basin에서는
**F_r이 F_xy를 거의 설명하지 못하고, proxy는 두 배 이상 과민한 기울기로 순위를 매긴다.**

> **등록 판정 (proxy arm에만 적용).** NULL이 나오고 §4.1·§4.2가 모두 기각되면,
> 등록된 결론은 **"F_r-surrogate 탐색은 F_xy frontier에 도달하지 못한다"** 이며 이는
> **f_xy head(`s1j`)를 지지하는 증거**로 기록된다 — 설계서 §3.6이 미리 적어 둔 바로 그
> 결과다. 이 경우 r2의 전제조건은 셀 교체가 아니라 **head 승격**이다.
>
> **head arm(`s1j`)에서 같은 NULL이 나오면** 위 결론은 성립하지 않으며, 등록된 다음 수는
> "이 셀의 F_xy가 1.5491 근방에서 실제로 바닥"이라는 가설과, `min_fxy` 탐색 자체의 예산
> 부족을 분리하는 r2(같은 셀, 같은 objective, fresh seed)다.

### 4.4 λ slide 감시

`minfxy_lambda = 1000.0` 은 F_xy가 cyclen을 압도하도록 만든다(0.01 F_xy = 10 EFPD).
이 셀의 `min_fr` 라운드들은 λ=400을 썼는데, 그것은 **r4/r5가 0.01 F_r을 사기 위해
cyclen 617–619로 미끄러진 실측 실패**에 맞춰 자른 anti-slide haircut이었다. F_xy에 대해서는
그런 측정이 아직 없으므로 λ는 code default 1000에서 시작한다(§6).

> **등록 판정.** r1의 F_xy 상위 10 core의 median cyclen 이 incumbent 621.28 EFPD 대비
> **5 EFPD 이상 낮으면**, "λ=1000이 slide를 샀다"로 기록하고 **r2의 λ를 이 라운드의 자체
> 측정으로 재적합**한다. F_r 라운드의 400을 그대로 물려받지 않는다.

---

## 5. 챔피언 — 두 arm, launch 시점에 결정

### 5.1 arm A — `s1i` (proxy). **오늘 launch 하면 이 arm이다.**

`data/models/s1i` 는 10번째 챔피언(`cond_schema = v8`, gate `data/reports/gate_s1i.json`
PASS)이고, **`predict_fxy` head가 없다** — 모델 디렉터리에 f_xy calibration이 없고
`acq.has_fxy_head` 가 `False` 를 돌려준다(빈 pattern 리스트로 실제 호출해 확인하는 probe).

- acquisition은 **INTERIM PROXY** `F_xy ≈ 1.2176·F_r − 0.2519` 로 순위를 매긴다
  (`acquisition.FXY_PROXY_SLOPE / _INTERCEPT`, 2026-08-29 refit: n=6,218 core / 119 cell,
  r 0.9895, residual sd 0.0476). σ 는 `sqrt((1.2176·σ_Fr)² + (3.0·0.0476)²)` — 즉
  **최소 0.1428** 로, 의도적으로 비관적이다.
- 모든 wave의 `selection.json` 에 `fxy_source = "proxy"` 가 기록된다(dry-run으로 확인).
- launch 배너 `[optimize][F_xy PROXY] …` 는 **정상 동작**이다.
- **경고 (문서 결함 정정).** scratch example deck
  `fpcamp_minfxy_T6T4_f121_199.inp` 헤더가 인용한 `1.1221 / −0.0831` 은 **폐기된 값**이다
  (192 core / 2 cell 적합, bias +0.0103, 분산 약 38% 과소). 새 deck 헤더는 refit 값으로
  정정했다.

**이 셀에서 proxy가 얼마나 맞는가 (실측):**

| 대상 행 | n | bias (측정−proxy) | sd | max\|e\| |
|---|---:|---:|---:|---:|
| `T6_T4/f121` 라벨 전체 | 738 | **+0.0027** | 0.0350 | — |
| `T6_T4/f121` **joint-clean** | 215 | **+0.0275** | 0.0329 | 0.1145 |

joint-clean slice에서 proxy는 **F_xy를 0.0275 과대예측**한다(보수적). gate 1.65는 slack
이므로 그 편향이 후보를 잘라내지는 않지만, **순위**는 F_r 순위로 붕괴한다(§4.3).

**arm A의 등록된 기대치.** 이 arm은 사실상 `min_fr` 탐색을 다시 도는 것이다. §1.1이
보여주듯 그 탐색은 이 셀에서 **F_xy를 악화시켰다.** 따라서 arm A의 **사전등록된 예측은
NULL** 이며, PRIMARY 달성 시에는 그 원인을 acquisition이 아니라 **elite pool(§7, 측정 F_xy로
정렬)** 과 wave fine-tune에 귀속해야 한다.

**arm A가 주장할 수 있는 것 / 없는 것** (설계서 §3.6):
- 가능: "N개 수렴 core, 그중 M개가 **측정** `F_xy ≤ X`" / 셀별 `(a,b)` 재적합 /
  frontier-by-call.
- **불가**: "F_xy를 최적화했다". 정직한 표현은 **"F_r-surrogate 탐색 + 사후 F_xy 측정"**.

### 5.2 arm B — `s1j` (head). launch 시점에 승격되어 있으면 이 arm이다.

`data/models/s1j` 는 **현재 존재하지 않는다**(디렉터리 없음). f_xy head의 승격 조건은
`data/reports/fxy_head_prereg_20260829.md` 에 이미 사전등록되어 있고, **G1·G2·G3 전부
PASS** 여야 `s1j` 로 승격된다.

- `predict_fxy` head가 실제 값을 돌려주면 `fxy_source = "head"`, PROXY 배너는 나오지 않는다.
- **§2.3의 `f_xy` parity 마크(\|Δ\| ≤ 0.01)가 LIVE 가 된다.**
- **arm B의 등록된 기대치.** §4.3의 판정이 뒤집힌다: head arm에서 PRIMARY가 나오면
  "F_xy를 실제로 최적화했다"고 말할 수 있고, NULL이 나오면 그것은 proxy 실명이 아니라
  **셀 바닥 또는 예산 부족**의 증거다.
- **caveat (기록).** head는 프로그램 전체 6,218 라벨 중 이 셀이 **11.9%(738건)** 를
  차지하는, 심하게 편중된 코퍼스로 학습된다. 이 셀에서의 성능은 **fit 검사에 가깝고
  generalization 검사가 아니다.** cell-holdout 결과는 head prereg 쪽에서 읽는다.

### 5.3 launch-time substitution rule (구속력 있음)

> deck은 `[model] model_dir = "data/models/s1i"` 로 **ship** 한다.
> **launch 명령을 내리기 전에** f_xy head가 `data/models/s1j` 로 승격되어 있으면
> (head prereg의 G1·G2·G3 모두 PASS):
> 1. deck에서 **`model_dir` 한 줄만** `data/models/s1j` 로 바꾼다. **다른 knob은 금지.**
>    (`cond_schema` 는 s1j의 `meta.json` 이 v8이 아닐 경우에만, 그 사실과 함께 바꾼다.)
> 2. deck을 **재해시**한다.
> 3. `launch_fpcamp_minfxy_T6T4_f121_r1_199.ps1` 의 `$want` 와 `$modelName` 을 갱신한다.
> 4. **본 문서 §9의 표에 새 deck sha256과 s1j meta sha256을 stamp** 한다.
>
> 승격되어 있지 않으면 **s1i 그대로 launch** 한다. head를 기다리느라 라운드를 미루지 않는다 —
> arm A의 NULL 자체가 §4.3에서 head를 지지하는 등록된 증거이기 때문이다.
> 어느 arm이 돌았는지는 `selection.json → fxy_source` 가 **기계적으로** 증언하며,
> 결과 문서는 그 필드를 인용해야 한다. 사람의 기억을 인용해서는 안 된다.

### 5.4 pin PPI harvest — deck knob이 아니다 (등록된 구조적 한계)

`max_pin_burnup` 은 equilibrium runner가 `enable_pin_burnup=True` 로 생성될 때에만 쓰이며,
`optimize`/`produce` 가 함께 쓰는 factory `WaveVerifier._default_factory` 는 이를
**`False` 로 하드코딩**한다(`lpopt/search/verify.py:851`).
`produce_fxyera_r1_prereg_20260829.md` §6이 같은 사실을 이미 등록했다.

**따라서 이 라운드가 만드는 어떤 행도 `deliverable = True` 가 될 수 없다.**
`is_deliverable` 은 gate된 모든 축이 **측정**되어 있을 것을 요구하고
(`campaign.py:459-475`), pin BU는 전부 UNKNOWN이 되므로 `unknown_axes` 가 매 행마다
`("max_pin_burnup",)` 를 반환한다. **실측 확인**: `fpcamp*` 계열 캠페인의 store 행 중 pin BU를
가진 것은 phase-2에서 역merge된 소수(5~11건/캠페인)뿐이고, `T6_T4/f121` joint-clean 라벨
215행의 pin BU 보유 수는 **0** 이다.

> **phase-2 (사전등록, r1 배수 후에만 실행).** `pinbu_wave.py` + `pinbu_wave_keep_199.inp`
> 로 40 chain — r1의 수렴행 중 F_xy 벽에 가장 가까운 20건 + **r1 record의 정확한 재실행
> 20건**(replicate set). 이 한 번의 wave가 세 가지를 동시에 준다:
> (1) r1 frontier core의 **측정** pin BU(=PRIMARY(delivery) 판정),
> (2) `keep_success` 로 `MAS_OUT` 이 남으므로 **F_xy 자체의 determinism 측정**
>     (같은 pattern 재실행의 `FXYP` 대 r1의 `FXYP`),
> (3) 재실행 core의 `MAS_OUT` 보존.
> phase-2의 launcher trio는 **본 산출물에 포함되지 않는다.**

---

## 6. Deck — 바뀐 knob 전부, 그리고 그 이유

deck: `fpcamp_minfxy_T6T4_f121_r1_199.inp` (scratch example `fpcamp_minfxy_T6T4_f121_199.inp`
에서 파생). 아래 표에 없는 것은 **전부 example deck과 동일**하다.

| knob | example deck | **이 deck** | 이유 |
|---|---|---|---|
| `[flow] random_seed` | 1111 | **1650** | 1111은 **이 셀에서** `fpcamp_minfr_T6T4_r8` 이 이미 소진했다. 1650은 repo 어디에서도 쓰이지 않았고, 게이트 1.65를 이름으로 가진다 |
| `[flow] title` | s1e 문구 | 갱신 | 챔피언·마크를 반영 |
| `[model] model_dir` | `data/models/s1e` | **`data/models/s1i`** | s1e는 6번째 챔피언이며 v8 이전이다. s1i가 현 챔피언(§5.1). **launch-time에 s1j로 치환 가능(§5.3)** |
| `[model] cond_schema` | `v6b` | **`v8`** | `s1i/member_20260716/meta.json` 이 v8이다. v6b 선언은 조용한 serving mismatch |
| `[master] use_all_cores` / `host_reserve` | `false` / 1 | **`true` / 0** | 199는 전부 P-core (TRIPLE r2 deck과 동일 처리) |
| `[design]` 블록 | 없음 | **추가** | paramA routing을 명시(`anchors_meshv3_198.inp` idiom, TRIPLE r2와 동일) |
| `[acquisition] budget` | 100 | **100 (12×8+4, 명시)** | 이 셀의 `min_fr` 라운드(r3~r8)가 전부 쓴 예산과 **동일** → call 대 call 비교 가능 |
| `[acquisition] policy_prior` | (미지정, 기본 off) | **`"off"` 명시** | §10 |
| `cycle_target_efpd` | 633.0 | **621.3** | 633은 `E1_E2/f121` 의 operating point이지 이 셀의 것이 아니다. 621.3은 **incumbent core 자신의 cyclen** 이므로 `distance` 가 "이겨야 할 core로부터 몇 EFPD" 로 읽힌다. **report-only** |
| `cycle_tolerance_efpd` | 5.0 | **10.0** | band 611.3–631.3 은 이 셀 joint-clean cyclen 전 범위(614.66–627.34)를 덮는다 → **아무것도 배제하지 않고 판독 라벨로만 기능** |
| `[search] near_miss_top_k` | 8 | 8 (주석: flat_power 전용 INERT) | 오해 방지 주석만 추가 |
| run dir / tag | — | **`runs/fpcamp_minfxy_t6t4_f121_r1`** | fresh |

**변경하지 않은, 그러나 사전등록해야 하는 값들:**

| knob | 값 | 등록 사유 |
|---|---|---|
| `objective` | `min_fxy` | 이 문서의 전부 |
| `minfxy_lambda` | **1000.0** | code default이자 TRIPLE f125 라운드들이 쓴 값. `λ=0`이면 F_xy 항이 **삭제**되어 (`scalar = cyclen_lcb − λ·fxy_ucb`) 조용히 cycle 최대화로 뒤집힌다. 이 셀 joint-clean cyclen 스프레드는 12.7 EFPD(sd 2.46)뿐이라 F_xy가 엄격히 지배하고 cyclen은 near-tie만 정렬한다. **`min_fr` 라운드의 400을 물려받지 않는 이유와 slide 감시는 §4.4** |
| `f_xy_limit` | **1.65** | 2026-08-29 사용자 결정. objective의 hard gate이자 delivery gate |
| `f_r_limit` | 1.55 | licensing (`delivery.LICENSING_FR_LIMIT`). **이제는 순수 constraint** — 행을 정렬하지 않는다 |
| `cbc_limit` / `f_q_limit` / `ao_abs_limit` | 1600 / 2.41 / 0.30 | 2026-08-11 프로그램 결정 및 licensing |
| `minfxy_pin_bu_limit` | 78.0 | LEU+ 80 − 2.0 model margin(`pinbu_definition_20260820.md`). **예측치만** 심사(§5.4) |
| `minfxy_cyclen_lo` / `_hi` | **unset** | 둘 다 설정하면 cyclen이 hard two-edge constraint로 승격된다. `min_fr_max_cycle` 때와 동일하게 **secondary tie-break로 유지** |
| `[verify] harvest_maps` | **true** | §3.1. LOAD-BEARING |
| `[produce] template_fallbacks` | `[]` | paramA에서 비워 두어야 한다(ga80 reload deck이 resolution을 이기고 `%GEN_DIM` sanity gate에서 죽는다) |
| `[constraints] mtc/sdm` | true / false, `post_verify_top_k = 5` | licensing chain 속성, report-only. 캐리 |

**주의 (F_q와의 정합, 설계서 부록 2).** 실측 `F_q/F_xy = 1.1586 ± 0.0218` 이므로
`F_xy = 1.65` 는 `F_q ≈ 1.91` 을 함의한다 — `F_q ≤ 2.41` 대비 훨씬 빡빡하다. 이 셀
joint-clean의 실제 `F_q` 최대도 2.41에 한참 못 미친다. **즉 `F_q` gate는 이 라운드에서
non-binding 이며, 그것이 binding으로 바뀌면 이례 사건으로 보고한다.**

---

## 7. Elite pool — 코드로 검증했다 (요구된 확인 항목)

과제 지시는 "elite pool이 `min_fxy`에서 **측정 F_xy** 로 정렬되는지 확인하고, `F_r` 로
정렬한다면 launch 전 필수 수정으로 표시하라"고 했다. **확인 결과: 수정 불필요.**

- `CampaignDriver._store_elites` (`campaign.py:1261-1298`) 는 후보 행을
  `_campaign_objective` 로 점수화한다.
- `_campaign_objective` 의 `min_fxy` 분기(`campaign.py:1751-1760`) 는
  **`−f_xy·1e6 + cyclen`** — 즉 **측정 `f_xy` 오름차순, cyclen 내림차순**. `f_r` 은 등장하지
  않는다. **F_r로 정렬하지 않는다.**
- 정렬은 **feasible-first 2-tier**: `is_feasible_search` 를 통과한 행이 먼저, 그다음 나머지.
  각 tier 내부는 위 objective 순.

**등록해야 할 미묘한 점 2가지 (결함 아님, 관측 대상):**

1. `is_feasible_search` 는 **결측 `f_xy` 를 통과시킨다**(`campaign.py:311-390`, 의도된 설계 —
   그렇지 않으면 최초의 `min_fxy` 캠페인이 아예 시작할 수 없다). 따라서 라벨 없는 행도
   feasible tier에 들어올 수 있다. 그러나 그 행들의 objective는 `−inf` 이므로 **tier 바닥에
   가라앉는다.**
2. 이 셀에는 joint-clean **라벨 행이 215개** 있고 `elite_top_k = 32` 이므로,
   **32개 elite 슬롯은 전부 F_xy-라벨 행으로 채워진다** (실측 기반 확인).

**`elite_seed_cases` 는 설정하지 않는다.** 이 셀 자신의 store 행이 프로그램에서 가장 두꺼운
F_xy 라벨 개체군이며, 다른 operating point의 행을 donor로 들여오면 그 행들의 F_xy 순위가
이 지점에서 비교 가능하지 않다.

---

## 8. 등록된 분석 — r1 결과 문서가 반드시 담아야 하는 것

이 목록은 **결과를 보기 전에** 고정된다. 사후에 항목을 추가/삭제할 수 없다.

**A. frontier-by-call.** call 1..100에 대한 running min 측정 `F_xy` 곡선. 함께 표시:
incumbent 선 1.5491, PRIMARY 선 1.5441, STRETCH 선 1.5295. **최고 기록이 몇 번째 call에서
났는지**를 명시한다 — 그것이 "예산이 binding이었는가"에 답한다(TRIPLE r1 §9의 선례).

**B. λ-check on F_xy.** 실현된 `(F_xy, cyclen)` 산점도 위에 exploit scalar
`cyclen − 1000·F_xy` 의 등고선을 얹는다. 보고 항목:
(i) 상위 10 core의 median cyclen 대 incumbent 621.28 — §4.4의 slide 판정,
(ii) **cyclen이 실제로 순위를 바꾼 후보 쌍의 수** (F_xy 동률 근방에서만 일어나야 한다),
(iii) 만약 λ가 slide를 샀다면 r2용 재적합 λ 제안값.

**C. proxy-vs-measured F_xy calibration table.** r1 자체 라벨로 아래 표를 채운다.
사전 등록된 비교 기준(이 셀, 현 store)을 함께 싣는다:

| 대상 | n | slope a | intercept b | resid sd | proxy bias | proxy sd | max\|e\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전역 proxy (현행 상수) | 6,218 | 1.2176 | −0.2519 | 0.0476 | — | — | 0.31 |
| `T6_T4/f121` 라벨 전체 (사전) | 738 | 1.1103 | −0.0768 | 0.0332 | +0.0027 | 0.0350 | — |
| `T6_T4/f121` joint-clean (사전) | 215 | 0.4811 | +0.8904 | 0.0300 | +0.0275 | 0.0329 | 0.1145 |
| **r1 자체 라벨** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **r1 + 사전 라벨 결합** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

추가 필수 항목: `F_xy ≤ 1.65` 근방 slice의 **오분류 표**(proxy가 feasible이라 했는데 측정이
위반, 그리고 그 반대). head arm이면 head 예측으로 같은 표를 만든다.

**D. deliverable vs search-feasible 카운트.** 세 수를 **분리해서** 보고한다:
`n_converged` / `n_feasible`(=`is_feasible_search`) / `n_deliverable`(=`is_deliverable`).
**`n_deliverable = 0` 이 예상값이며 그것은 실패가 아니다**(§5.4). 두 predicate가 갈라지는
이유를 한 문장으로 재진술해야 한다.

**E. `unknown_axes` 히스토그램.** 수렴 행별 `unknown_axes` 의 분포.
**등록된 기대: 모든 행이 정확히 `["max_pin_burnup"]`.** 여기에 `"f_xy"` 가 하나라도
나타나면 **harvest 결함**이며(§3.1) 그 행 수를 반드시 보고한다 —
`_campaign_objective` 상 그 행들은 UNSCORABLE(`−inf`)이므로 "나쁜 결과"가 아니라
**측정되지 않은 결과**다.

**F. arm 선언.** 결과 문서 첫 표에 `selection.json → fxy_source` 의 wave별 값을 싣고,
`"proxy"` 이면 §5.1의 "주장 가능/불가" 문구를 그대로 인용한다.

---

## 9. Frozen artefacts / fleet / provenance

### 9.1 sha256 (launch 전, 로컬 기준)

| item | sha256 | bytes |
|---|---|---:|
| ~~**deck** `fpcamp_minfxy_T6T4_f121_r1_199.inp` (s1i 판, **2026-08-30 치환으로 폐기**)~~ | ~~`4AF8B0218666EC83E0E9357C6FB268F179301EE30BEC9CD1CAA89C3144C2FAC5`~~ | 16,957 |
| scratch example `fpcamp_minfxy_T6T4_f121_199.inp` (파생 원본, 실행 안 함) | `27234CC338A5B655D503B477100ACD875D08E9992861D6EA945BAB321436CCF3` | 8,963 |
| ~~**store** `data/store/records.parquet` (74,717행)~~ | ~~`0334E2D2E303CD8E82373861603F82A57054FC2FC6139F84C360822580644D9D`~~ | 22,142,229 |
| **store (STAMPED 2026-08-30)** `data/store/records.parquet` (**75,793행**) | `F38666E9F1508D35D33E0C22F583C5479C6F09CAC748201B494B47C8CFECA6EA` | 22,538,411 |
| `data/store/fuel_types.parquet` | `FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137` | 64,343 |
| champion `data/models/s1i/member_20260716/meta.json` | `32BFB282370B16F7827C75966645A2FD4796CB3800F495850A7201BFC4FB5EC5` | 43,197 |
| `launch_fpcamp_minfxy_T6T4_f121_r1_199.ps1` (**STAMPED 2026-08-30** — `$modelName`/`$want`/`$wantStore` 갱신; 이전 `1D9D0D38…`, 8,598 B) | `36ACB60D601864BD1649A0E840C33E5A8E1B79803D051C7572B5196F73B6B665` | 9,537 |
| `run_fpcamp_minfxy_T6T4_f121_r1_199.bat` | `F3002666EECC7F5A2BD23071298ED7308D2EDDF629B0507358DECFFEED44CE49` | 2,750 |
| `status_fpcamp_minfxy_T6T4_f121_r1_199.ps1` | `78D527E85910CFFB533ACBA2F7A1358E42FCE6E759ACF96759F4EB4D68D0AC3C` | 4,700 |
| **launch-time 치환 (s1j) — 실행됨 2026-08-30** deck sha256 | `BEF3519E720FE1F94FE1448EF3046FCE1EB15BD94DBA5DD1F4E9B2F3976C95C9` | 16,957 |
| **launch-time 치환 (s1j) — 실행됨 2026-08-30** `data/models/s1j/member_20260716/meta.json` sha256 | `F0AF69C0F54261DEC61F253E3828FDC5DF742F0915440678C885B96BB4112E7B` | 37,223 |

> **STAMP — 2026-08-30, §5.3 launch-time substitution EXECUTED.**
> 사유: `data/models/20260829_194532` (arm 3, `--fxy-direct`) 가
> `data/reports/fxy_head_results_arm3_20260829.md` 의 판정 **G1 PASS · G2′ PASS · G3′ PASS ·
> G4 FAIL** 로 처분 C.4 에 따라 **`data/models/s1j` (11번째 챔피언) 으로 승격**되었다.
> §5.3 절차대로 **deck 의 `model_dir` 한 줄만** `data/models/s1j` 로 바꾸고(다른 knob 무변경,
> `cond_schema` 는 s1j meta 가 v8 이라 그대로), 재해시하여 launcher `$want` 와 `$modelName`
> 을 갱신했다. deck 의 나머지 주석 줄(“s1i = 10th champion … NO predict_fxy head”,
> `cond_schema` 옆 “matches s1i/…”)은 **한 줄 규칙 때문에 의도적으로 건드리지 않았다** —
> 실제 서빙 대상은 위 표의 s1j 해시다.
> **G4 FAIL 의 귀결: head 의 σ 는 서빙되지 않는다.** `data/models/s1j/ensemble.json` 이
> `fxy_head.serve_sigma = "barred"` 를 stamp 하고 `acquisition.predict_fxy` 가 head 의
> **평균만** 쓰며 폭은 기존 proxy σ 규약(`resid_sd 0.0476` × `K 3.0`)으로 대체한다
> (`data/models/s1j/PROMOTION.md` §2). 따라서 §2.3 parity 표의 `f_xy` 행은 **head arm 으로서
> gated** 이지만, 그 UCB 폭은 proxy 규약임을 함께 읽어야 한다.
> **store 행도 같은 날 stamp 했다** — e_core backfill 로 74,717 → **75,793행**
> (`F38666E9…`). 이 갱신 없이는 launcher 가 `MINFXY1 REFUSED: store sha256 mismatch` 로
> 거부한다. **phase-2 `pinbu_wave` 병합(§5.4)이 들어오면 store 해시가 다시 바뀌므로
> 이 행과 launcher `$wantStore` 를 함께 재stamp 해야 한다.**

launcher가 gate하는 값: deck sha = 위 **치환(s1j) deck 행**, store sha = 위 **store (STAMPED 2026-08-30) 행**. 취소선 두 행은 치환 전 값이며 더 이상 gate 하지 않는다.
deck을 한 바이트라도 고치면 launcher는 **거부**한다(치환 절차 §5.3을 밟지 않는 한).

### 9.2 검증 (launch 전에 로컬에서 수행함, MASTER 미호출)

| 점검 | 명령 | 결과 |
|---|---|---|
| deck 파싱·자산 해석 | `python -m lpopt check --input fpcamp_minfxy_T6T4_f121_r1_199.inp` | `11 PASS / 11 FAIL / 1 SKIP` — **참조 deck과 완전히 동일한 카운트** (`fpcamp_minfr_TRIPLE_f125_r2_199.inp`, `fpcamp_minfxy_T6T4_f121_199.inp` 모두 동일). FAIL은 조정 박스의 로컬 bootstrap 템플릿(`%LPD_B&C` 40행) 과 MASTER 실행파일 부재 — **deck 결함이 아니다.** `case: mode=fixed pair=T6_T4 feed=121` 해석 및 `bases\T6_T4\MAS_RST.APRQ_10_0615.11` PASS |
| stub 목적함수 배선 | `python -m lpopt optimize --input … --dry-run --budget 8` (StubEvaluator, MASTER 미호출, run dir은 scratch) | 정상 완료. 배너 실측: `[optimize] min_fxy objective = cyclen_LCB - 1000 * F_xy_UCB \| HARD F_xy <= 1.650 \| F_r stays a CONSTRAINT at 1.550 \| pin BU <= 78.0` 및 `[optimize][F_xy PROXY] the served model exposes NO 'predict_fxy' head: candidates are ranked on the INTERIM proxy F_xy ~ 1.2176*F_r -0.2519 with an inflated sigma …` — **refit 상수가 실제로 배선되어 있음을 확인.** `selection.json → fxy_source = "proxy"` 확인. RESULT 줄이 F_xy 축으로 읽힘 |
| config 회귀 | `python -m pytest tests/test_config.py -q` | **45 passed** |

### 9.3 fleet 규칙

- **199 전용.** busy gate는 stack 하지 않고 **거부**한다. 특히
  **`produce_fxyera_r1` 이 도는 동안 거부**한다 — 이름 기반 검사 + rc 파일 부재 검사 두 겹.
  198 / 181 / 238 은 건드리지 않는다.
- fresh run dir `runs/fpcamp_minfxy_t6t4_f121_r1`. launcher가 선삭제한다.
- ship-don't-remote-edit: 모든 파일을 통째로 scp 한다.
- store는 이 실행을 위해 **canonical 판을 그대로 사용**한다(§9.1 sha) — 746건의 f_xy
  라벨이 곧 elite pool이기 때문이다(§7).
- 디스크 gate 20 GB: `harvest_maps` 가 수렴 chain마다 마지막 cycle dir을 보존한다.

### 9.4 알려진 판독 함정 (선례 그대로 캐리)

- **NOTE (R)** — `lpopt report` 의 "Best verified loading patterns" 표는 objective와
  무관하게 **cycle distance 로 정렬**하며 F_xy 승자를 보여주지 **않는다.**
  `state.json → best_overall` 을 읽어야 한다. 기존 결함, 변동 없음.
- **NOTE (P)** — `deliverable` 은 전 행 `false` 가 정상이다(§5.4).
- **NOTE (D)** — `min_fxy` 는 `min_fr_max_cycle` 과 달리 DEPRECATED 배너를 내지 않는다.
  대신 **PROXY 배너**가 arm A의 정상 출력이다.

---

## 10. 이 라운드가 하지 않는 것

1. **policy A/B를 접붙이지 않는다.** `data/reports/policy_v2_serving_ab_prereg_20260829_DRAFT.md`
   는 **DRAFT이며 등록되지 않았다** — 그 문서 자신이 실행 조건으로 "(a) f_xy head가 ship
   되고 wave의 `selection.json` 이 `fxy_source = "head"` 를 기록할 것, (b) 사용자가 MASTER
   예산을 승인할 것"을 걸어 두었고, 두 조건 모두 오늘 충족되지 않았다.
   **본 deck은 `policy_prior = "off"` 를 명시**한다. 목적함수와 pool 구성 arm을 한 번에
   바꾸면 결과를 둘 중 어느 쪽에도 귀속할 수 없다. **이 라운드가 곧 그 A/B의 control arm(A)
   이 될 후보다** — 다만 A/B는 자기 자신의 사전등록·자기 자신의 deck으로 돈다.
2. **r2를 미리 승인하지 않는다.** `E1_E2/f121` r2는 §1.3에서 지명만 하며, 실행은 r1 판독
   후 **새 deck + 새 사전등록**으로 결정한다. r1 run dir에 예산을 top-up 하는 것은 금지한다.
3. **phase-2 `pinbu_wave` 를 이 산출물에 포함하지 않는다.** §5.4에 계획만 등록한다.
4. **코드를 고치지 않는다.** §7의 elite-pool 확인 결과 수정 필요 항목은 **없었다.**
   만약 launch 전에 `_campaign_objective` 의 `min_fxy` 분기가 바뀐다면, 본 사전등록은
   무효이며 다시 써야 한다.
5. **원격에 아무것도 올리거나 실행하지 않았다.** launcher trio는 작성만 되었고,
   `Invoke-CimMethod` 는 호출되지 않았다.

---

## 11. 요약 — 한 문단

`T6_T4/f121` 에서 `min_fxy` 를 100 call 돌린다. 이 셀은 F_xy 라벨이 프로그램 최다(746)이고,
`min_fr` 추격이 종결되었으며, 그 추격이 **F_xy를 1.5491 → 1.5829 로 악화시킨 것이 측정되어
있다.** 이겨야 할 수는 **1.5491**, PRIMARY 선은 **1.5441**, 프로그램 기록선은
**1.5295**(다른 셀)다. `F_xy ≤ 1.65` 는 이 셀에서 이미 joint-clean 행의 80.9%가 통과하므로
마크로 쓰지 않는다. 오늘 launch 하면 챔피언은 `s1i` 이고 acquisition은 **F_r proxy로
순위를 매긴다** — joint-clean basin에서 corr(F_xy, F_r)=0.282 이므로 **사전등록된 예측은
NULL** 이며, 그 NULL은 f_xy head를 지지하는 증거로 기록된다. `s1j` 가 launch 전에 승격되면
`model_dir` 한 줄만 바꾸고 재해시한다. pin BU는 이 라운드에서 측정되지 않으므로
`deliverable` 은 전 행 false 이고, 납품 등급은 phase-2 `pinbu_wave` 가 준다.

**STAMP 2026-08-30 04:5x (pinbu phase-2 이후 재스탬프):** store `data/store/records.parquet` sha256 = `72516916F5D59A738BA95CE2A7D56F0F2E9F514E61DD654BE4BE6127D175CE5D` (75,793행, `max_pin_burnup` 32셀 patch — `pinbu_wave_fxyera_r1_results_20260830.md`); launcher `$wantStore` 동일 값으로 갱신. deck sha 불변(BEF3519E…).
