# 06 — 학습된 장전 규칙 (Learned Loading Rules)

> **이 문서의 주제** `lpopt`이 APR1400 평형주기 장전모형(LP)을 최적화하면서, **집합체 연소 특성**과 **주요 결과 인자**(cyclen, CBC, F_r, F_q, node_peak/map_cov, 핀·집합체 연소도, AO, SDM/MTC)를 분석하여 **"최적 평형주기 장전모형을 찾는 규칙"을 무엇을 어떻게 학습·발견했는가**를 한 곳에 정리한다.
> 세 층으로 구성된다.
> 1. **결과 인자 사전**(§1) · **집합체 표현**(§2) — 규칙이 말을 거는 대상의 정의
> 2. **공개 엔지니어링 규칙 지식베이스**(§3) — 문헌이 이미 아는 것, 그리고 본 프로젝트가 채택/미채택한 부분
> 3. **AI가 실측으로 확정·발견한 규칙**(§4) — 71,517~73,903행 MASTER 검증 노심과 사전등록 통제실험에서 **채굴(mined)** 된 규칙. 이 문서의 핵심.
>
> 그리고 **규칙의 코드 구현**(§5)과 **다주기/향후 확장**(§6).

**출처 규약.** 본 문서의 모든 수치는 저장소 내 리포트(`data/reports/*.md`)와 코드에 근거한다. 각 규칙에 근거 리포트·수치·날짜·신뢰도를 붙였다. 근거 없이 저자가 덧붙인 해석은 **(추정)** 으로 표시한다.

**주요 근거 문서**

| 파일 | 역할 |
|---|---|
| `data/reports/loading_rules_v1_20260811.md` | 규칙집 v1 — 채굴→적대적 재검증→acid test 전 과정 |
| `docs/reference/PWR_commercial_core_loading_pattern_engineering_rules_KO.md` | 공개자료 기반 PWR 장전 규칙 종합 보고서(§3의 원본) |
| `data/reports/kcurve_fusion_memo_20260809.md` | F_r 융합법칙, BOC node_peak↔F_r 역상관 |
| `data/reports/ab2_addendum_ADF_20260810.md` / `_BU_20260810.md` | 집합체 표현 판정(H2/H4), 연소도 배치 |
| `data/reports/ablation_wave_results_20260815.md` 외 | 연산자·피처 기여 어블레이션 |
| `data/reports/pinbu_definition_20260820.md` / `pinbu_audit_20260820.md` | 핀 연소도 정의·감사 |
| `data/reports/sdm_mtc_limits_20260726.md` | SDM/MTC 한계 |

---

## 1. 결과 인자 사전 — 정의·단위·제약값·LP 민감도

`lpopt`이 한 장전모형(LP)에 대해 MASTER 평형주기 계산을 돌려 얻는 라벨과, 그 라벨에 걸리는 제약값이다. **제약값은 "일반적 PWR 값"이 아니라 본 프로젝트가 실제로 덱과 코드에 넣어 쓴 값**이다.

### 1.1 제약 요약표

| 인자 | 정의 | 단위 | 프로젝트 사용값 | 코드 위치 |
|---|---|---|---|---|
| `cyclen` | 평형주기 길이 | EFPD | 타깃 **625**(±2 기본, `min_fuel_cost` 밴드 **615–635** = 625±10) · 규칙집 타당범위 **620–645** · 캠페인별 report-only 밴드(예: 633±5) | `lpopt/config.py:460-461`, `:314` (`fuelcost_cyclen_lo/hi` 615/635) |
| `cbc_max` | 주기 중 최대 임계붕소농도 | ppm | **1550 → 1600**(2026-08-11 프로그램 결정) | `lpopt/config.py:464` (기본 1550), 신규 덱 `cbc_limit = 1600.0` |
| `f_r` | 반경방향 첨두 (F_ΔH 계열) | – | 인허가 한계 **1.55** (`LICENSING_FR_LIMIT`); `flat_power`에서는 안전 게이트 **1.70** | `lpopt/search/delivery.py:43`, `config.py` `flatpower_fr_limit=1.7` |
| `f_q` | 3-D 출력 첨두 | – | **2.41** | `lpopt/config.py:465` |
| `node_peak` | 집합체(노드) 출력 최댓값 = **F_xy** | – | 목적함수(1차항), 하드 한계 없음 | `lpopt/data/flatness.py` |
| `map_cov` | 노드 출력의 다중도 가중 변동계수 | – | 목적함수(2차항, `w_cov` 기본 0.5) | `lpopt/data/flatness.py` |
| `max_assembly_burnup` | 집합체 평균 연소도 최댓값 | GWd/tU | 예측 타깃(8번째 타깃, v5에서 승격), 게이트는 핀 축이 대행 | `promote_max_asm_bu` |
| `max_pin_burnup` | **핀 axial(노드) 첨두** 연소도 | GWd/tU | 한계 **80** · 예측 게이트 **78**(=80−2.0 모델 마진, `min_fr_max_cycle`) | `lpopt/config.py:301` (`minfr_pin_bu_limit=78.0`), 타 목적함수는 80.0 |
| `AO` | 축방향 출력 옵셋 | – | \|AO\| ≤ **0.30** | `lpopt/config.py:466` (`ao_abs_limit`) |
| MTC | 감속재온도계수 | pcm/°C | **−54.0 ≤ MTC ≤ +9.0** | `lpopt/search/sdm_mtc.py`, 덱 `mtc_max_pcm_per_c=9.0` / `mtc_min_pcm_per_c=-54.0` |
| SDM | 정지여유도 | pcm | 요구 **10,870** ( = 10,180 + 690 ) | 덱 `sdm_required_pcm=10870.0` |

### 1.2 각 인자의 정의와 실측으로 알려진 LP 민감도

#### cyclen — 주기길이 [EFPD]

- **정의**: MASTER 평형주기 수렴 후의 주기 길이. 평형은 노심을 반복 재장전하여 고정점에 수렴시킨 상태이므로, cyclen은 "이 패턴을 무한히 반복했을 때의 주기".
- **제약 사용법**: 목적함수에 따라 위상이 다르다.
  - `target_cycle` — cyclen을 `cycle_target_efpd ± cycle_tolerance_efpd`에 고정하고 그 안에서 CBC/F_r 최소화.
  - `min_fr_max_cycle` — cyclen은 **게이트가 아니라** 스칼라화 항 (`cyclen_LCB − λ_Fr · F_r_UCB`)의 부호로만 들어온다. λ_Fr가 "F_r 1단위를 몇 EFPD에 살 것인가"의 가격표(기본 1000, 실전 200/400 사용).
  - `min_fuel_cost` — cyclen 밴드 **615–635**의 양 끝이 모두 하드 게이트.
- **LP 민감도(실측)**:
  - **feed(신연료 장전 수)가 오르면 cyclen이 오른다.** 전 pair에서 단조. E1_E2: f109 550–595 → f129 620–657 EFPD; G3_G4: 611–648 → 687–724 EFPD. (`feedgrid_pathfinder_20260815.md` §2). 브리핑 단계의 직관("저feed=장주기")과 **반대 방향**이었다.
  - `kbud`(패턴 반응도 예산) +0.001당 cyclen **+0.72 EFPD** — CBC(+13.8 ppm)에 비해 20배 불리한 교환비. → **cyclen은 슬랙 변수**(규칙 R3, §4).
  - **cyclen 이탈 > ~5 EFPD는 map_cov를 파괴**한다(반응도 정합 게이트, `loading_rules_v1` 확립사실 4).
  - **cyclen 밴드가 저밴드 F_r을 묶고 있었다** — cyclen 무게이트 `fr_boundary` 생산 캠페인에서 전 6밴드 서브-1.55 F_r 실측 달성(최저 E1_E2_f125 1.5085).
  - **가능영역 내부에서는 cyclen-F_r 트레이드오프가 사실상 없다**: feasible 폭이 N1N2/f113 0.68 EFPD(F_r 0.0312 이동), E1E2/f109 1.52 EFPD(F_r 0.0371 이동) (`fpcamp_N1N2_f113_results_20260816.md`, `fpcamp_E1E2_f109_results_20260817.md`).

#### CBC — 최대 임계붕소농도 [ppm]

- **정의**: 주기 중 최대 임계붕소농도(`cbc_max`).
- **제약 사용법**: 하드 게이트. **1550 → 1600 완화(2026-08-11 사용자 결정)**. 근거는 "E1_E2/f121 패턴 축이 소진되었고(끝점 추정 1.4624 vs 기록 1.4636), **F_r 1.45는 붕소 문제**"라는 판정. 1550~1600 회랑에 새 격자 설계(저 FF)가 들어갈 자리가 생겼다. 2026-08-11 이전 결과와 비교할 때는 재스크린이 필요하다.
- **LP 민감도(실측)**: CBC는 **반응도-외곽 축(§4 F1)의 공동 예산**이다.
  - `kbud` +0.001 → CBC **+13.8 ppm** (붕소 ~10 pcm/ppm과 수치 정합).
  - periphery-k(R2) rho **+0.691**, hot-fresh-outboard(R11) **+0.606**, k-gradient(R1) **+0.577**, ring3 fresh(R10) **+0.410** — 전부 CBC "지출".
  - 반대로 twice-burned를 hot ring에(R12) −0.291, inward migration(G4) −0.125, 짧은 travel(R7 회피) — CBC "수입".
  - `travel`(셔플 반경 이동거리) +1.0 pitch → CBC **+84 ppm**, cyclen +4.6 EFPD.
  - **CBC는 cyclen으로 되살 수 없다**(교환비 20:1 불리).

#### F_r — 반경방향 첨두

- **정의**: 반경방향 출력 첨두(F_ΔH 계열). 라이선싱 한계 1.55.
- **융합법칙(실측)**: `F_r ≈ A · max_{BOC,EOC steps, slots}( p_nom · FF_origin(BU_nom) )`, **A = 1.047**(in-band 중앙값, IQR [1.022, 1.084], rms 0.0496, n=679; 전구간 법칙은 A=1.0255/rms 0.035). 즉 **F_r은 (노드 출력) × (그 노드에 놓인 집합체의 핀 form-function)의 최댓값**이다 — 노심 배치와 집합체 격자설계가 곱으로 만난다.
- **LP 민감도**: §4의 F_r 문법 전체(R6/R8/R9/R10/R11/R15)가 여기에 붙는다. 핵심은 **FF 반경 기울기(R8)** — 저FF 연료를 출력 크레스트 외곽에, 고FF를 안쪽에.
- **F_r ↔ F_q**: 같은 방향으로 움직인다(r = +0.637 ~ +0.978). 실전에서 F_q는 사실상 F_r에 종속이고, **F_r만이 결박축**인 캠페인이 반복 관측되었다.

#### node_peak / map_cov — 평탄도 (`lpopt/data/flatness.py`)

한 레코드의 맵은 69-slot **1/4 노심**이고, 각 슬롯은 `multiplicity` w_i개의 물리 집합체를 대표한다(52슬롯×4, 16슬롯×2, 중앙×1, Σ = **241** = APR1400 전노심).

```
w_i       = SLOTS[i].multiplicity
p_bar     = Σ(w_i p_i) / Σ(w_i)                # 실측 1.0000 (0.9999–1.0001)
node_peak = max_i p_i                          # == F_xy
map_cov   = sqrt( Σ(w_i (p_i - p_bar)^2) / Σ(w_i) ) / p_bar
```

- **가중이 본질이다.** 수확된 맵은 이미 노심 평균으로 정규화되어 있으므로 `node_peak = nanmax`가 **정확히 반경 집합체 첨두계수 F_xy**다. 비가중 69-slot 평균은 중앙값 1.0233(0.983–1.088)이라 레코드마다 다른 분모를 쓰게 되어 물리량으로 보고할 수 없다. 이 모듈은 두 스칼라의 **단일 정의처**다(과거 지역 복사본이 서로 다른 두 숫자를 만들어 보고서를 재조정해야 했던 사건이 계기).
- **라벨 잡음 실측**(`flatness_noise_measured_20260726.md`, 전치쌍 22쌍, 추가 MASTER 콜 0): σ(node_peak) = **0.00556**, σ(map_cov) = **0.000507**. 전역 상한 ρ는 두 축 모두 **≥0.9997** → **잡음은 병목이 아니다**. 단 최심층 평탄대(node_peak ≤ 1.394)에서만 상한 0.947(스토어)/0.973(fold C)로 0.95 기준선에 붙는다.
- **LP 민감도**: node_peak의 1차 변수는 **반경 방향 반응도 기울기 `g_k_radgrad`**로, 단독으로 node_peak 오차의 **41%** 를 설명한다. 그리고 **가능영역 내부에서 BOC node_peak와 F_r은 역상관**(13-pt dose Pearson **−0.750**) — 평탄한 노심과 저 F_r 노심은 같은 노심이 아니다(§4.6).
- **프런티어 궤적**(E1_E2/f121): node_peak 1.284 → 1.2285 → 1.2085 → 1.1932 → **1.1899**(fpcamp4, 2026-08-10). v3/v4 연속 ~1.19로 포화 조짐.

#### max_assembly_burnup / max_pin_burnup — 연소도

- **`max_assembly_burnup`**: 집합체 평균 연소도의 최댓값. v5에서 예측 타깃으로 승격.
- **`max_pin_burnup`**: **핀 axial(노드) 첨두** — 평형 최종주기 EOC에서 `MAS_PPI`의 `BPIN(:,:,:)` (nzc×npin×npin) **3-D raw 최댓값**. 축방향 평균화 없음.
  - **한계 80 GWd/tU**(사용자 확정 2026-08-20, `pinbu_definition_20260820.md`).
  - **예측 게이트 78** = 80 − 2.0 GWd/tU 모델 마진(`minfr_pin_bu_limit`).
  - 보조 관측량 **rod-average**(`max_rod_avg_burnup`, `HZ(:)` 가중 축평균 후 (i,j) 최대)는 **기록 전용**. 실측 비 A/B = **1.0878 ± 0.0030**(5기).
  - 파서: `lpopt/data/pinppi.py`(봉평균 병산), 노드 첨두는 벤더 burnup 파서.
- **물리 분해(실측)**: `pin_burnup = ratio × B_asm`, ratio ≈ 1.13–1.23. corr(B_asm, 실측 pin) = **+0.922**. 다만 **B_asm만 낮추는 전략은 실패한다** — B_asm을 76.0→68.8(−7.2)로 낮추면 ratio가 1.134→1.223(+7.9%)로 반비례 상승해 순개선은 86.19→84.20의 2.0 GWd에 그쳤다(`f113_pin_results_20260820.md`).
- **핀 축의 이중 지위**(사용자 정책 2026-08-20):
  - **설계지도(그물망)** — **advisory**. 셀을 닫지 않고 표기만(노드 테두리 색). 판정은 F_r·CBC·F_q·\|AO\| 4축.
  - **캠페인 납품 판정** — **정식 게이트**. 실측 `max_pin_burnup` ≤ 80, 초과 FAIL.

#### AO — 축방향 출력 옵셋

- **제약**: \|AO\| ≤ 0.30.
- **실측 상태**: 전 코퍼스 \|AO\|max **0.278 < 0.30**, std ~0.014, 밴드 무관. 즉 **LP 셔플만으로는 AO 분산을 만들 수 없다**(축방향 자유도가 필요).
- **프로젝트 판정(사용자 2026-07-25)**: AO는 "측정 불가 결함"이 아니라 **"무해한 여유 상태"** 로 기록한다. APR1400급 대형 노심 + 붕산수 반응도 제어에서는 AO가 기준치 한참 아래이므로 예측 필요성이 사실상 없다. 축방향 자유도는 **확장하지 않는다**(소형 노심 + 제어봉 제어, 예: i-SMR로 갈 때 탑재).

#### SDM / MTC (`lpopt/search/sdm_mtc.py`, `sdm_mtc_limits_20260726.md`)

- **MTC**: 판정 창 **−54.0 ≤ MTC ≤ +9.0 pcm/°C** (APR1400 DCD Tier 2 Table 4.3-3 포락값을 `Δρ/°C ×10⁻⁴` → pcm/°C 로 ×10 환산). 반응도는 `rho_pcm=(k−1)/k×1e5`, 2점 폴백은 `(rho[T+Δ]−rho[T−Δ])/(2Δ)`, Δ=5.0 °C. **실측 사례 `boc_hzp` +1.21 pcm/°C → PASS**, 창 폭 63 pcm 대비 여유가 크다. 즉 현재 운전점에서 **MTC는 slack**이다.
- **SDM**: 요구 **10,870 pcm** = 10,180(CEA 총 반응도 여유) + 690(정미 제어봉가 불확실도). 계산: `W_ARI = rho_ARO − rho_ARI`, `available = W_ARI − max_i(worth_i)`, `margin = available − 10,870`. **현재 구성은 구조적으로 FAIL**이다: 설정의 `scram_banks`/`stuck_candidate_banks`가 `[R1..R5]`만 담고 **정지뱅크 A/B가 빠져 있다**. A+B가 EOC worth의 74%(12.32/16.70 %Δρ)를 차지하므로 A/B 없이는 물리적으로 통과 불가. 실측 `boc_hzp`: W_ARI 9,684.8 / worst_stuck 1,606.9 / available 8,078.0 → **margin −2,792 pcm**. → **이는 물리 제약이 아니라 설정 결함으로 판단**되어, 캠페인 덱은 `sdm_enable=false` (`mtc_enable=true`)로 운영하며 SDM은 INCONCLUSIVE로만 보고한다. MASTER `%ROD_CFG` 그룹 수 한계 92 — A+B+R1~R5 = 81은 OK, P(12) 추가 시 93으로 크래시.

### 1.3 수용 기준 (최적점에서의 예측–MASTER 일치)

프로그램의 합격선은 전역 MAE가 아니라 **캠페인이 실제로 제안하는 최적 후보에서의 예측–실측 일치**다(사용자 확정 2026-07-30).

| 축 | 허용치 |
|---|---|
| cyclen | ≤ 1 EFPD |
| CBC | ≤ 1 ppm |
| F_q · map_cov · node_peak | ~0.001 오더 (수십 pcm 수준) |
| AO | 현 단계에서 중요하지 않음 |

전이쌍 잡음 하한(cyclen 0.08 EFPD, CBC 0.31 ppm) 위이므로 물리적으로 도달 가능하다.

---

## 2. 집합체 연소 특성 인자 — "집합체를 무엇으로 표현하는가"

### 2.1 원칙 — 집합체는 격자 핀맵이 아니라 **물리 출력 벡터**

**피처 우선순위 결정(사용자 확정 2026-07-18)**: 집합체 표현의 **주 피처 = 결과값** (격자계산 산출물)이다.

- `FA_*.out` / `.sum`의 **k-inf(BU) 곡선 특성점**
- HGC의 **2군 단면적**, **pin form function 최대치(`ff_pin_max`)**, **ADF**

이유는 **임의의 가연성독봉·설계제원에 자동 일반화되는 "행동 서명"** 이기 때문이다. `inp` 설계축(zoning 농축도, Gd 제원)은 보조 피처·폴백으로만 쓰고, **identity 임베딩은 금지**(감사로 부재 확인).

이 원칙은 2026-08-17 사용자 지시("노심-크기 불변 엔지니어")에서 다시 확인되었다 — 17×17에 국한되지 않고 **집합체의 지오메트리보다 output(k-curve, F_r, F_q, MTC, FTC, 핵종 inventory 등)** 을 받아 다양한 크기의 노심(APR1400/OPR1000/i-SMR)에 장전한다.

### 2.2 k-inf(BU) 곡선 특성점 9채널 (`lpopt/data/fuel_types.py` `KCONV_SHAPE_COLUMNS`)

기준 k-inf(BU) 소진곡선을 **반응도 공간** `rho = (k−1)/k × 1e5` [pcm]에서 읽어 **가연성흡수체의 holddown → burnout release 서명**을 포착한다. **어떤 흡수체가 만들었는지와 무관한 형상 서명**이라는 것이 요점이다(오늘 Gd, 내일 IFBA/Er/Dy).

| # | 채널 | 의미 |
|---|---|---|
| 1 | `kinf_dip` | 억제 트로프의 k-inf — 첫 burnout hump 상승부 직전의 국소 최소 |
| 2 | `bu_dip_gwd` | 그 트로프의 연소도 [GWd/tU] (**트로프 시점**) |
| 3 | `kinf_peak` | 트로프 이후 hump 최댓값 (흡수체 완전 소진 시점의 k-inf) |
| 4 | `bu_peak_gwd` | 그 hump의 연소도 (**burnout 타이밍**) |
| 5 | `reactivity_swing_pcm` | `rho_peak − rho_dip` — **holddown release 크기** |
| 6 | `rho_boc_minus_peak_pcm` | `rho(0) − rho_peak` — BOC 반응도(무제논 fresh boost)의 hump 대비 **억제 깊이** |
| 7 | `depletion_slope_pcm_per_gwd` | hump 이후 준선형 소진구간(`bu_peak..min(60, last)`)의 `d(rho)/dBU` — **주기길이를 좌우하는 소진 감쇠율** (BU=0 무제논점은 항상 제외) |
| 8 | `kinf_eol50` | 50 GWd/tU에서 보간한 k-inf — **방출 k** |
| 9 | `kconv_is_monotone` | 두드러진 hump가 없으면 1.0 — **흡수체 강도 게이트**. 단조 곡선에서는 1·2·5가 NaN, 3·4는 BU=0 값으로 축퇴 |

모델 입력으로는 8채널로 정규화되어 들어간다(`lpopt/model/featurize.py` `_V5_SHAPE_EXTRA`, 정규화 상수는 실측 모집단 중앙값/robust half-span, n=92–117): `origin_reactivity_swing`(/2500), `origin_depletion_slope`((x+600)/130), `origin_bu_peak`((x−19)/12), `origin_bu_dip`((x−7)/7), `origin_rho_boc_minus_peak`(/2800), `origin_kinf_eol50`((x−0.957)/0.05), `origin_kconv_monotone`, `origin_kconv_present`(블록 존재 게이트).

**독봉 불가지론(poison-agnostic) 확정(사용자 지시 2026-07-20)**: 학습 채널에서 독봉 특정 축(`n_gd`, `gd_wt`, `gd_u_enr` 및 관련 전역 통계)을 **제거**하고, 위 곡선 형상 채널이 보편 독봉 서명을 담당한다. v5 스키마에서 채널 교체 + 제거 어블레이션 A/B로 **형상 채널의 실재를 입증**했다(v5_noshape arm = Gd만 제거한 대조군). `fuel_types`의 Gd 컬럼은 부기용으로만 유지한다.

### 2.3 HGC 유래 채널 (`COND_V4_COLUMNS`)

| 그룹 | 컬럼 | 의미 |
|---|---|---|
| 핀 form function | `ff_pin_max` | 집합체 내 핀 출력 form function 최댓값 — **F_r 융합법칙의 FF 인자** |
| 2군 단면적 | `xs_d1`,`xs_d2`,`xs_a1`,`xs_a2`,`xs_nf1`,`xs_nf2`,`xs_s12` | 확산계수·흡수·ν생성·산란(1→2) |
| ADF | `adf_face_g1`,`adf_face_g2`,`adf_corner_g1`,`adf_corner_g2` | 면/코너 assembly discontinuity factor, 2군 |
| 반응도 계수 | `boron_worth`,`doppler_coef`,`mtc_dmod`,`cr1_worth` | 붕소가·도플러·감속재밀도·제어봉가 |
| 부기 | `zone_pin_count` | zoning 핀 수 |

추가로 **핀 연소도 첨두비 곡선 요약**(`PIN_BU_COLUMNS`)이 있다: `ratio(BU) = BRP/BU`(첨두핀/집합체평균 연소도 비)를 `r_inf + paramA/BU`로 적합하고 (`pin_bu_r_inf`, `pin_bu_paramA`, `pin_bu_ratio_asym`, `pin_bu_bu_max`, `pin_bu_n_pts`), BU→0 극한은 `ff_pin_max`가 고정한다. 그리고 **핀셀 기하**(`GEOM_COLUMNS`: `pin_pitch`,`asm_pitch`,`r_pellet`,`r_clad_in`,`r_clad_out`,`p_over_d`, `v_mod_over_v_fuel`)는 dec inp의 `GEOM` 블록에서 수확한다(ga80/legacy는 NaN).

### 2.4 충분통계 판정 — (k곡선 + 농축도)는 **충분하지 않다**

**H2/H4 널테스트**(`kcurve_fusion_memo_20260809.md`, `ab2_addendum_ADF_20260810.md`): k곡선이 57 pcm 이내로 일치하고 농축도가 동일하며 **ADF만 3.2% 다른** 두 집합체가 **ΔF_r 0.0351의 노심 차이**를 만들었다(음성 대조 0.0005). 즉:

- **(k곡선 + 농축도) = 충분통계 아님** — 확정.
- **두 번째 특성 = ADF** — 회귀 설명력 +0.0305 R² (농축도는 +0.0058).

다만 **후속 판정이 중요하다**: face-ADF를 새 채널로 추가한 v6c arm(58→62ch)은 동결면에서 node_peak MAE **−0.0144 악화**(CI 전체 음수, harm 4건, 5축 established-worse) 로 **REJECT**되었다. 이유는 **ADF 가설의 유효 성분(`adf_corner_g2`, `cr1_worth`, `ff_pin_max`)이 이미 v4부터 입력에 있었고**, 대조군이 H2/H4 판별을 이미 통과했기 때문이다(예측 +0.0069 vs 실측 +0.0072). → **"ADF가 두 번째 특성"이라는 결론은 기존 채널로 이미 충족된 상태**였다. 교훈: *널테스트가 지목한 물리량이 이미 다른 이름으로 입력에 있는지 먼저 확인하라.*

### 2.5 연소도 배치 — 곡선 정밀도보다 **5.1배** 큰 오차원

`ab2_addendum_BU_20260810.md` / 챔피언 `data/models/20260810_bu_T` (cond_schema v6b).

- 5개 arm 연속 기각 후 **첫 사전등록 통과 메커니즘**이 "**연소도 배치 수정**"이었다.
- 무엇을 고쳤나: `NOMINAL_CYCLE_BURNUP = 22.0` 고정값을 **(library, feed)별 실측 테이블**로 교체(ga80 f121 = 28.69 등 6값) + 소스체인 채널 6개 추가(52→58ch, append-only).
- 결과: 게이트 node_peak 셀중앙 MAE **+0.01734**(기준 0.00409의 **4.2배**), cyclen·CBC·map_cov 동반 개선, `gate-promote` 양 게이트 PASS.
- **교훈(핵심)**: **곡선을 정밀화하는 것보다, 그 곡선을 "어느 연소도에서 읽는가"가 5.1배 큰 오차원이었다**(2801 vs 1402 pcm). 집합체 표현에서 "적재 연소도 위치"는 곡선 자체와 동급 이상의 1급 인자다.

이 사실은 규칙 측에서도 재확인된다 — **R18(chain-burnup-validity)**: 체인 유도 슬롯 연소도 `BU_nom = B_regime · Σ p_nom[direct 체인]`이 맵 수확 BOC 연소도를 **순위 수준** 으로 재현한다(679행 median within-row Spearman 0.618 → 재검증 0.694, MAE 8.04–8.21 GWd/tU). 즉 **규칙 적용에 연소도 라벨이 불필요**하다. 단 FF(BU) 구분 (fresh 0 / once ~25–30 / twice ~50–60)에는 충분하나, **<5 GWd/tU 절대 구분에는 사용 금지**.

---

## 3. 공개 엔지니어링 규칙 지식베이스 — 무엇을 채택했고 무엇을 안 했나

원본: `docs/reference/PWR_commercial_core_loading_pattern_engineering_rules_KO.md` (2026-07-29, 공개 규제문서·국제기구 보고서·연료공급사 기술자료·국립연구소 보고서· 학술논문 종합). 아래는 **본 프로젝트와의 접점**만 발췌·판정한 요약이다.

### 3.1 규칙의 5분류 (H/P/N/O/Q)

| 분류 | 의미 | 최적화 코드 구현 권장 |
|---|---|---|
| **H** 경성 제약 | 위반 시 안전해석·기술사양·연료설계 기준 불만족 | 즉시 탈락 / 보수적 screening |
| **P** 플랜트·공급사 규칙 | 승인된 설계절차·경험·기계적 호환성 | 명시 금지 또는 repair |
| **N** 물리 휴리스틱 | 좋은 후보를 빨리 만드는 경험칙 | seed 생성, mutation bias, **soft penalty** |
| **O** 목적함수/선호 | 경제성·여유·운전유연성 개선 지표 | 다목적 Pareto |
| **Q** 검증/QA | 계산·연료이력·현장 loading 오류 방지 절차 | 독립 확인, 추적성, 전노심 재계산 |

> **본 프로젝트의 채택**: 이 분류를 그대로 따른다. §4의 규칙들은 **G군 = H 상당 (하드 게이트)**, **F1–F6군 = N 상당(소프트 페널티)** 로 소비된다(§5.4). KNFC McFLOP이 Ring-of-Fire를 "완전 금지 → 가중 penalty"로 바꿔 국소최적 문제를 완화한 사례를 규칙 소비의 표준으로 삼는다.

### 3.2 OUT–IN vs IN–OUT vs 저누설

문헌의 판정은 **"둘 다 유효하나 목적함수가 다른 조건부 전략"** 이다.

| 전략 | 개념 | 장점 | 약점 |
|---|---|---|---|
| OUT–IN | 신연료를 바깥에, 이후 안쪽으로 | 신연료 출력을 누설로 억제, 단순 평탄화 | 중성자 경제성 저하, 용기 fluence 증가 |
| IN–OUT | 신연료를 안쪽에, 조사연료를 바깥으로 | 중성자 경제성, 잔류반응도 활용 | 내부 fresh/burned interface peaking, CBC·MTC·SDM 관리 |
| IN–OUT–OUT / 저누설 | 고연소도 연료를 외곽 buffer 층에 | 용기 조사량 저감 | 외곽 stranded reactivity, 내부 peaking |
| Scatter / checkerboard | 고반응도 연료를 분산 | radial/pin peaking 억제 | 반응도 불연속 면의 pin peak |
| Ring-of-Fire | 고반응도 연료가 특정 반경대에 띠 형성 | 특정 목적에 유리할 수 있음 | 연속 띠의 peaking, 플랜트별 금지 |
| Hybrid | 내부 scatter + 외곽 buffer | 실제 상용노심에 가장 가까움 | 규칙 복잡, 3-D 검증 필요 |

문헌은 특히 **"BOC k∞가 높은 집합체는 외곽에 둔다"는 보편 법칙이 아니다** 라고 명시한다 — OUT–IN이나 특정 평탄화에서는 성립하나, 저누설 IN–OUT에서는 대체로 반대.

> **본 프로젝트의 실측 판정(§4)**: `lpopt`이 채굴한 **R1(radial-k-gradient)** 은 "BOC k-inf가 반경과 함께 **상승**하도록"(저k 안쪽, 고k 바깥쪽)이다 — 형태상 OUT–IN 계열이다. 그러나 이는 **평형주기 + 고정 feed 스펙 + node_peak/F_r 목적** 이라는 본 문제의 조건 아래 within-cell 상관으로 측정된 것이고, **CBC를 지불한다** (rho +0.577). 문헌이 경고한 대로 **보편 법칙으로 승격시키지 않았고**, 실제로 **K-case_pair에서 부호 반전**(R2/R11)이 관측되어 도메인을 제한했다. 문헌의 "보편 법칙 아님"이 본 데이터에서 그대로 재현된 셈이다.

### 3.3 대칭 규칙 (문헌 S-01 ~ S-08)

- 1/4 거울대칭 seed는 강력한 **P/N** 실무 규칙이지 물리 법칙이 아니다.
- **거울대칭과 90° 회전대칭을 구분**해야 한다(quarter 반사경계는 mirror를 요구).
- 대칭 위치의 재고 multiplicity(일반 4 / 축 2 / 중앙 1)를 지켜야 한다.
- 1/8 대칭은 대각선 거울대칭까지 성립할 때만.
- 최종 후보는 전노심 3-D unique-history로 재계산.

> **채택**: 노심 **1/4 대칭**과 multiplicity(4/2/1, Σ=241)는 게놈 구조에 **하드 인코딩**되어 있다(§5.1). 집합체 **1/8 대칭**은 설계규칙 R2로 채택. **미채택**: 전노심 unique-history / serial-ID 추적, 집합체 orientation(회전) 자유도. 본 프로젝트는 **type-map 수준의 평형주기 문제**를 푼다.

### 3.4 인접 금지 · checkerboard · Ring-of-Fire (문헌 R-03~R-06, A-03/A-04)

문헌이 인용하는 실제 사례:
- Yamamoto(상용노심 강의): inboard에서 **무BA 신연료의 side-by-side 제한**, 경우에 따라 **대각 인접도 제한**.
- INL demonstration: **Gd 연료의 수평·수직 인접 금지**.
- KNFC APR1400(McFLOP): inner core의 **인접 fresh와 Ring-of-Fire 억제** → checkerboard에 가까운 패턴.

> **채택/실측 대조(§4)**:
> - **fresh face 인접 최소화(RM1i)** → **살아 있다**. `R5`로 CONFIRMED (within-cell +0.227 node_peak / +0.295 map_cov, 엘리트 캠페인에서 오히려 증폭).
> - **대각 인접 회피(RM2i)의 보편 적용** → **DEAD**. 최신 캠페인에서 부호 반전 (fpcamp4-6 −0.665/−0.597/−0.425, live minFr +0.538). 스토어 양성은 RM1i 축 편승이었다.
> - **checkerboard** → **엘리트 레짐 한정 정밀화 규칙**(R19). 스토어 전역에서는 null(−0.007), 엘리트 캠페인에서만 −0.65~−0.79. in-band에서 RM1i와 −0.96 공선.
> - **Ring-of-Fire류 "fresh는 무조건 주변부로"** → **DEAD**. 셀 간 부호 반전 (node_peak −0.770 E1_E2 vs +0.219 E3_E4). 이식 가능한 형태는 넓은 외곽대 r>6.5(R10)뿐. 즉 **문헌 휴리스틱 4개 중 1개만 그대로 살아남고, 1개는 레짐 조건부, 2개는 데이터가 죽였다.**

### 3.5 다주기 관점 (문헌 B-07, B-13, §3.5)

- 현 주기 EOC k 최대화는 고반응도 연료를 중요도 높은 위치에 과집중시켜 **다음 주기 재고를 악화**시킬 수 있다. 연속 2주기 최적화 연구가 이를 강조한다.

> **본 프로젝트의 위치**: `lpopt`은 **평형주기(equilibrium)** 문제를 푼다 — 즉 "이 패턴을 무한 반복했을 때의 고정점"이므로 다주기 재고 문제가 **구조적으로 소거**되어 있다. 전이주기(cy1→평형)는 §6의 최종 로드맵으로 등록되어 있으나 **착수 금지** 상태다.

### 3.6 핀 / 3-D / 열수력 연계 (문헌 M-01~M-14, T-01~T-10, B-02)

- 최종 배치는 assembly-average 출력만으로 승인할 수 없다 — 핀 출력복원, 3-D 축형상, 열수력, 제어봉 사고, 연료성능, CIPS/CRUD, 용기 조사량, 불확실도까지 연결.
- **B-02**: 집합체 평균과 **최대 pin burnup을 모두 제한**한다.

> **채택**: **핀 연소도(핀 axial peak, 80 GWd/tU)** 를 정식 게이트로 채택(§1.2). `F_q`(3-D 첨두) 2.41, MTC 창, SDM 요구치도 채택. **미채택(현 단계)**: DNBR/subchannel, CIPS/CRUD, 용기 fast fluence, 연료성능 (PCI/corrosion/bow), 제어봉 사고해석(rod ejection/dropped rod), startup physics test 연계. SDM은 채택했으나 뱅크 구성 결함으로 현재 INCONCLUSIVE 운영(§1.2).

### 3.7 채택/미채택 요약표

| 문헌 규칙군 | 본 프로젝트 | 비고 |
|---|---|---|
| H/P/N/O/Q 분류 | **채택** | G군=하드, F군=소프트 페널티(§5.4) |
| 1/4 노심 대칭 + multiplicity | **채택(하드)** | 게놈 구조 |
| 1/8 집합체 대칭 | **채택(설계규칙 R2)** | 2026-07-23 신설 |
| 패턴 내 전 feed 동일 농축도 스펙 | **채택(설계규칙 R1)** | zoning = 0.85×main |
| fresh face 인접 제한 | **채택(소프트)** | 실측 CONFIRMED |
| fresh 대각 인접 제한 | **미채택** | 실측 DEAD(부호 반전) |
| checkerboard | **조건부 채택** | 엘리트 레짐 정밀화만 |
| Ring-of-Fire / "fresh는 주변부" | **미채택** | 실측 DEAD |
| 저누설 IN-OUT vs OUT-IN | **데이터가 결정** | R1은 형태상 OUT-IN, 단 CBC 지불 |
| 집합체 orientation(회전) 자유도 | **미채택** | type-map 문제, pin/quadrant 이력 없음 |
| 축방향 자유도 / AO 축 | **미채택(사용자 확정)** | 대형노심+붕산제어에서 AO는 무해한 여유 |
| 제어봉 사고해석·rod worth 최적화 | **미채택** | SDM만 부분 채택(현재 INCONCLUSIVE) |
| 다주기 재고 최적화 | **구조적으로 소거** | 평형주기 문제 |
| serial-ID / 전노심 unique history | **미채택** | type-map 수준 |
| 핀 연소도 한계 | **채택(게이트 80, 예측 78)** | 관측량 = 핀 axial peak |
| 용기 fluence / CIPS·CRUD / DNBR | **미채택** | 향후 확장 후보 |

---

## 4. AI가 실측으로 확정·발견한 규칙 (핵심)

### 4.0 방법론 — "학습된 규칙"이 되기 위한 세 관문

`data/reports/loading_rules_v1_20260811.md` (2026-08-11). 대상은 ga80 라이브러리 평형노심 1/4 대칭 LP(69-slot quarter, 241 집합체), 주 검증 셀 `E1_E2|f121`. 데이터 기반은 `data/store/records.parquet` **71,517행**(+), `maps.npz` EDIT5 nodal planes, `fuel_types.parquet`; 심판 모델은 챔피언 `data/models/split_S1b`(v6b).

1. **채굴(mining)** — 순수 `(pattern, fuel_table) → 스칼라` 함수로 정의된 메트릭을 **셀 내부(within-cell) 상관으로만** 평가. 셀 = `campaign | case_pair | feed`. **셀 간 풀링은 feed/e_core와 교락되므로 금지.** 셀 단위 split-half holdout.
2. **적대적 재검증** — 채굴자와 **독립된** 파서·체인 추적기·맵 로더·통계 코드로 전 수치를 재계산. 결과: **재현 실패 0건**, 단 헤드라인 부속 주장 5건은 강등·기각. 최종 verdict: **18 CONFIRMED / 6 DOMAIN-LIMITED / 0 DEAD-on-reproduction**.
3. **구성 가능성(acid test)** — 규칙만으로(탐색 없이, 모델 없이, 스토어 엘리트 복사 없이) 노심을 지어서 100-call 캠페인 기록에 접근하는지 MASTER로 확인(§5.4).

**확립된 기전 사실 6개**(모든 규칙이 이와 정합해야 함):

- `F_r = A · max(p·FF(BU))`, A ≈ 1.03–1.047 — 그리고 in-band에서 BOC node_peak는 F_r과 **반상관**(r = −0.750, 13-pt dose).
- **역할 분리가 평탄화 기전 그 자체**: E1-role 68슬롯 mean p ≈ 0.805 / E2-role 53슬롯 mean p ≈ 1.131. C5/C6 동질장전 파괴실험에서 node_peak 1.22 → 1.55.
- `g_k_radgrad`(반경 방향 반응도 기울기)가 node_peak 오차의 **41%** 를 단독 설명.
- cyclen 이탈 > ~5 EFPD는 map_cov를 파괴(반응도 정합 게이트).
- 슬롯 BOC 연소도는 소스 체인을 통한 패턴의 **순수 함수**(depth ≤ 2).
- **타당범위(feasible band)**: `f_r ≤ 1.55`, `cbc_max ≤ 1600`, `f_q ≤ 2.41`, `620 ≤ cyclen ≤ 645`.

**독립성 구조 — 다이얼은 20개가 아니라 6개다.** 중복도 분석(`E1_E2|f121` feasible n=381, 19개 변형 메트릭) 결과 규칙 공간의 실질 자유도는 **~6개 독립 인자 + 2개 포화 게이트**다. 특히 in-band에서 `kbud ≈ hot_share ≈ periphery-k`(상관 0.94–0.99)이므로 **붕소 예산과 평탄화 다이얼은 한 레버**다.

각 규칙의 서술 형식: **서술 / 적용법 / 근거 / 유효범위 / 위반 시 대가**.

---

### 4.1 G군 — 게이트 규칙 (탐색 전 강제; in-band에서 분산 0이므로 상관이 아니라 게이트)

#### G1. [center-cold-fresh] 노심 중앙 슬롯에는 두 fresh 배치 중 저-kinf0(고 Gd) 배치를 둔다
신뢰도: **CONFIRMED** (A8 + D3)
- **적용법**: `a_center_is_cold(pattern, fuel_table)` — slot 0 점유 배치의 kinf0가 fresh 쌍 중 최소이면 1.0. E1_E2 테이블에서는 홀수 `n_hot` ⇔ centre = E1.
- **근거**: `P(feasible | cold center) = 0.646` vs `P(feasible | hot center) = 0.007` (`E1_E2|f121`, n=585/414, Fisher **p = 8.3e-117**); `J5_J6|f121` 0.605 vs 0.024 (p=1.2e-13); 재검증에서 신규 게이트 셀 3개 추가(`J1_J2|f121` p=2e-15, `J1_J2|f117` p=2.3e-44, `K1_K2|f117` p=1.4e-7). 쌍별 within-cell 118셀: centre=high-k 시 Δmedian f_r = **+0.044**, 62.7% 양성, p=0.0073; kinf0 갭이 큰 ga80 E/H 쌍은 **+0.311**.
- **유효범위**: 전 셀 일반화 확인. 갭이 작은 라이브러리에서는 효과 축소(+0.036).
- **위반 시 대가**: feasibility 승산 **~1/250**(64.6% → 0.7%); F_r 중앙값 +0.5 오더 악화 (2.063 vs 1.536). **Construct에서 다른 어떤 결정보다 먼저 고정할 것.**

#### G2. [no-center-cross-fresh] 1번 축 궤도 유닛에 fresh를 두지 않는다
신뢰도: **CONFIRMED** (B2) — 중앙은 항상 fresh이므로 위반 시 5집합체 fresh 십자가 형성.
- **적용법**: `b_c1_fresh` = `ROW_OFFSETS[0]+1` 슬롯 점유가 fresh면 1.0 (축-쌍둥이 동일, 스토어 불일치율 0.0000).
- **근거**: within-cell rho **+0.262** node_peak (188셀, n=12,367, 음성 10.1%), **+0.284** f_r (257셀, n=22,719), +0.281 f_q; AUC 0.653/0.665; 최신 캠페인 전부 양성 (fpcamp4 +0.23 … live minFr f_r **+0.51**, p=2.7e-7); holdout 셀에서 더 강함(+0.385).
- **유효범위**: 스토어 전역. **feasible E1_E2에서는 분산 0(전원 준수) → in-band에서는 상관이 아니라 게이트.**
- **위반 시 대가**: within-cell 평균 이동 node_peak +0.29, f_r +0.36; cyclen 무영향(−0.006).

#### G3. [구조 불변량] 쌍둥이 대칭·비순환 depth≤2 체인·242−2F 연령 센서스는 탐색하지 말고 강제한다
신뢰도: **CONFIRMED** (D7)
- **적용법**: `GeneralOrbitGenome.validate()` — twinbreak=0, selffeed=0, age3 초과분 = `max(0, 242 − 2·feed)` 고정.
- **근거**: 71,517행 중 twinbreak>0 = **0행**, selffeed>0 = **0행**; age3_excess의 within-cell 분산 보유 셀은 607셀 중 **1셀**. `tolerance_margin`은 어떤 family-D 메트릭에도 반응하지 않음(|wrho| ≤ 0.045).
- **위반 시 대가**: 검증된 규칙이 하나도 적용되지 않는 미지 공간으로 이탈. **연령 혼합비·tolerance_margin에 규칙 용량을 쓰지 말 것**(비-레버).

#### G4. [inward-migration-converges] 연소 연료의 순 이동은 안쪽으로
신뢰도: **CONFIRMED** (D6) — 순-외향 셔플은 지배적 수렴 실패 모드다.
- **적용법**: `inward` = `Σ_burned w_s·(radius(direct_source(s)) − radius(s)) / (241−feed) ≥ 0` (constructor는 **≥ 0.75** 사용).
- **근거**: converged 판별 weighted AUC **0.681**, IQR [0.635, 0.721], 202셀, n=27,839, p=6.4e-57. **동일 메트릭이 최강 평탄화 축**: map_cov wrho **−0.502**(99% 음성), f_r **−0.422**(92% 음성). 재검증 정밀화: cbc 중앙값 −0.125(69% 음성) — 안쪽 이동은 붕소도 **약하게 돕는다**.
- **위반 시 대가**: 수렴 승산 약 절반; 평탄도 대폭 악화. cyclen은 +1.8 EFPD/pitch만 유의.

---

### 4.2 F1 — 반응도-외곽 축 (master axis: 저k 안쪽 / 고k 바깥쪽)

> in-band에서 `k_radgrad`, `k_outer_mean`, `kbud`, `hot_share`, `inward`, `rm1i`(−), `rm2i`, `checker`, `hot_burned_nom`(−)이 |rho| 0.74–0.99로 **한 다발**이다. **하나의 레버를 여러 이름으로 세지 말 것.**

#### R1. [radial-k-gradient] BOC 집합체 k-inf가 반경과 함께 상승하도록 배열하라
신뢰도: **CONFIRMED** (A1) — 저k 안쪽, 고k 바깥쪽. 기울기가 가파를수록 node_peak 1차 감소, in-band F_r도 감소.
- **적용법**: `a_k_radgrad` = 다중도 가중 `Pearson corr(k_slot, radius)` over 69 slots.
- **근거**: node_peak rho **−0.707** (`E1_E2|f121` n=346, p=1.1e-53, top-decile AUC 0.90), −0.494 (`J5_J6`), −0.555 (`J1_J2|f121`); Stouffer p=7.9e-47. f_r −0.376 / −0.531 / −0.374 (E3_E4 / J5_J6 / J1_J2|f121). fresh-periphery 축을 partial한 후에도 −0.20..−0.46이 전 5셀에서 유지; partial|cyclen −0.469. **확립 사실 3(g_k_radgrad 41%)의 a-priori 확인 — harvest 맵 불필요.**
- **유효범위**: E/J 셀 확인. **`E3_E4`는 node_peak에 대해 null**(그 셀의 in-band node_peak 산포는 비-반경 기전 — Construct 주의). K셀 미확인.
- **위반 시 대가**: `dnode_peak/dmetric` = −0.52..−0.71 (metric sd 0.023–0.046 → 기울기 +1sd = node_peak −0.012..−0.032); f_r 기울기 −0.055..−0.132/unit. **CBC 비용 동반: within-cell cbc rho +0.577.**

#### R2. [periphery-k-mean] 최외곽 링 13슬롯(48집합체)을 밴드가 허용하는 최고 BOC k-inf로 채운다
신뢰도: **DOMAIN-LIMITED — E/J 셀 한정** (A2)
- **적용법**: `a_k_outer_mean` = `PERIPHERY_MASK` 13슬롯의 다중도 가중 평균 `k_boc`.
- **근거**: R1과 동일 축(within-cell rho +0.80..+0.91). node_peak **−0.758** (E1_E2, p=8e-66); f_r −0.326..−0.539 (4/5셀), top-decile AUC 0.61–0.78.
- **유효범위**: **`K3_K4`에서 node_peak +0.388 (p=0.028) 유의 부호반전, `K1_K2` f_r +0.319 — K-case_pair에는 적용 금지.**
- **위반 시 대가(유효 도메인 내)**: `df_r/dk_outer` = −0.63..−1.90/unit; `dnode_peak` −6.7/unit (E1_E2). **CBC 비용 최대(+0.691).**

#### R3. [cbc-follows-k-budget] 반응도 예산은 붕소를 먼저, 주기길이를 나중에 정한다
신뢰도: **CONFIRMED** (D1) — family-D 중 **최강 이식성**.
- **적용법**: `kbud` = `(1/241)·Σ w_s·k_b(s)((age_s−1)·22)`.
- **근거**: `cbc_max` wrho +0.330(중앙값 +0.536, IQR [+0.314, +0.696], 286셀, n=48,242, 음성 6%, p=1.7e-59, holdout +0.325/+0.335; 재검증 +0.311 / 13.3 ppm); cyclen wrho +0.120. **kbud +0.001당 CBC +13.8 ppm, cyclen은 +0.72 EFPD뿐** — 13.8 ppm/0.001은 붕소 ~10 pcm/ppm과 수치 정합. 소형 셀 holdout에서 오히려 강화(+0.525).
- **위반 시 대가**: E1↔E2 전노심 1집합체 스왑(Δkbud ~2.8e-4) = CBC **~+3.9 ppm**. 예산 초과분은 cyclen으로 회수 불가(**교환비 20:1 불리**).
- **함의**: **예산은 CBC 게이트 바로 아래에 놓고 cyclen을 슬랙 변수로 취급하라.**

#### R4. [feed-split-serves-the-gates] 평탄도는 fresh 분할에 무감하므로 붕소 게이트가 허용하는 최대 고-kinf0 share를 선택하라
신뢰도: **CONFIRMED** (D2)
- **적용법**: `hot_share` = `Σ(w_s over 고kinf0 fresh 슬롯) / feed`. minfr 프로파일 **64/121**(16유닛), flat 프로파일 **68/121**(17유닛) — 두 셀 기록이 정확히 그 지점(F_r 기록 1.4636 @64, flat 기록 1.1899 / map_cov 0.2147 @68–72).
- **근거**: → `cbc_max` wrho +0.325 (284셀, n=48,180); → cyclen +0.148; → f_r / f_q / node_peak / map_cov **전부 null**(+0.026..+0.035). kbud와 한 축.
- **위반 시 대가**: +1% high-k share당 CBC **+3.13 ppm**(재검증 +3.08)·cyclen +0.198 EFPD; **평탄도 비용 0** — 분할을 평탄도에 쓰는 것은 낭비, 게이트에만 써라.

#### R5. [rm1i-alive] 내부(비주변) fresh-fresh 면접촉을 최소화하라
신뢰도: **CONFIRMED** (B3) — 명시적 사망판정 요청에 대한 답: **NOT DEAD**.
- **적용법**: `rule_metrics.rm_fresh_face_adjacency(pattern, inboard=True)` (다중도 가중, mirror-expanded 17×17).
- **근거**: 스토어 within-cell **+0.227** node_peak / **+0.295** map_cov (286셀, n≈39.6k); fpcamp4 **+0.735**(p=1e-17), fpcamp5/6 +0.573/+0.588, live minFr f_r +0.450(p=1e-5); fpcamp_minfr만 +0.192(p=0.055)로 약화·비반전. AUC 0.662/0.641; cyclen/cbc 경유 아님.
- **유효범위**: 전 스토어 + 전 엘리트 캠페인. in-band E1_E2에서는 F1 축과 공선 (checker와 −0.96).
- **위반 시 대가**: 가중 면접촉 쌍당 node_peak **+0.0026**(스토어) / **+0.0068**(엘리트 셀); 전형 셀 within-cell 범위 64쌍 → 레버 폭 ~0.17 node_peak.

---

### 4.3 F2 — 크레스트 점유·셔플 이동거리 축

#### R6. [fresh-off-mid-crest] 저 F_r을 원하면 fresh를 중반경 출력 크레스트(4.5 < r ≤ 6.5, 68집합체)에서 치워라
신뢰도: **CONFIRMED** (A6) — **단, 최평탄 노심은 정반대로 한다**(§4.11 상충).
- **적용법**: `a_fresh_ring2_share` = 4.5 < radius ≤ 6.5 슬롯의 다중도 가중 fresh 비율.
- **근거**: f_r **+0.508** (E1_E2, p=2.4e-26), +0.599 (J5_J6), +0.280 (J1_J2|f121); Stouffer p=2.6e-32; holdout 신규 `K1_K2` +0.605 (p=1.5e-4). **목적 상충 직접 측정**: 동일 메트릭 vs node_peak (E1_E2) **−0.680** (p=2.7e-48).
- **유효범위**: E3_E4는 동부호 null(+0.108). **주의(재검증): node_peak 상충은 E1_E2/J5_J6 특이 — `J1_J2`에서는 크레스트 fresh가 둘 다 해친다(+0.406).**
- **위반 시 대가**: 크레스트에 전노심 1집합체 추가할 때마다 F_r **+0.003..+0.004**; E1_E2에서는 같은 이동이 node_peak −0.019.

#### R7. [short-shuffle-travel] 연소 집합체는 전주기 연소 반경 근처에 재장전하라
신뢰도: **CONFIRMED** (D4) — 반경 이동거리는 kbud·inward와 **독립적으로** CBC와 cyclen을 올린다.
- **적용법**: `travel` = `Σ_burned w_s·|radius(s) − radius(direct_source(s))| / (241−feed)`; 보조 `stay` = |Δr| ≤ 1.0 pitch 가중 비율.
- **근거**: → `cbc_max` wrho **+0.377** (288셀, n=48,873, p=4.8e-60, holdout +0.392/+0.364); partial|kbud +0.416, partial|inward +0.397, partial|cyclen +0.336 — **독립 축**. → cyclen +0.249; map_cov +0.302, f_r +0.234. `stay` 미러(cbc −0.228, 91% 음성).
- **위반 시 대가**: 가중 평균 이동 +1.0 pitch당 CBC **+84 ppm**(재검증 +82.9)· cyclen +4.6 EFPD; 동일 링 재장전 10% 증가 ≈ CBC −15 ppm. 긴 out-to-in / in-to-out 도약은 다른 곳에서 값을 치러야만 정당화된다.

---

### 4.4 F3 — FF 반경 기울기 (독립 인자; 가장 이식성 높은 F_r 규칙)

#### R8. [ff-gradient-down] 기원 배치의 `ff_pin_max`가 반경과 함께 하강하도록 배열하라
신뢰도: **CONFIRMED** (A4) — 저FF 연료를 출력 크레스트 외곽에, 고FF 연료를 안쪽에.
- **적용법**: `a_ff_radgrad` = 다중도 가중 `corr(FF_origin, radius)`; **양수(FF 외향 상승)가 나쁨**.
- **근거**: f_r **양성 9/9셀** (holdout `K3_K4` +0.449, p=0.0099 포함): +0.390 (E1_E2), +0.430 (E3_E4), +0.467 (J5_J6), +0.479 (J1_J2|f121); Stouffer p=9.0e-33. hot-fresh-outboard와 축 공유(−0.46..−0.80)이나 partial 신호 유지 (E1_E2 +0.310, E3_E4 +0.210, J1_J2|f121 +0.287). 기전상 `F_r = A·max(p·FF)`의 직접 표현.
- **유효범위**: **전 셀 — K셀 포함 부호 일관, family-A 중 유일하게 무제한 이식.**
- **위반 시 대가**: 단위 기울기당 F_r **+0.067**(중앙값 OLS; +0.016..+0.102); within-cell 1sd(0.09–0.16) = F_r +0.007..+0.010.

---

### 4.5 F4 — 내륜/외곽대 점유 (독립 점유 다이얼)

#### R9. [fresh-out-of-inner-ring] fresh를 내륜 r < 2.5(중앙 포함 21집합체)에 넣지 마라
신뢰도: **CONFIRMED** (A5)
- **적용법**: `a_fresh_ring0_share` = radius ≤ 2.5 슬롯(quarter 8슬롯)의 가중 fresh 비율.
- **근거**: f_r +0.322 (E1_E2, p=1.2e-10), +0.310 (E3_E4), **+0.605** (J5_J6, p=7.9e-14), +0.233 (J1_J2|f121); Stouffer p=1.2e-24; holdout 유의 반전 없음(K1_K2 −0.06 ns).
- **유효범위**: 분산 있는 전 셀. node_peak에는 E1_E2 약양성뿐 — **F_r 규칙**.
- **위반 시 대가**: 내륜 전노심 1집합체당 F_r +0.003..+0.005.

#### R10. [fresh-into-outer-band] fresh를 외곽대 r > 6.5(104집합체)로 밀어라
신뢰도: **CONFIRMED** (A7) — **이것이 이식 가능한 형태**다. 원시 주변부-한정 fresh 수 (RM4 축)는 부호 불일치로 **사용 금지**.
- **적용법**: `a_fresh_ring3_share` = radius > 6.5 슬롯의 가중 fresh 비율.
- **근거**: f_r 음성 5/5셀: −0.315 (E1_E2), −0.448 (E3_E4), **−0.623** (J5_J6, p=8.5e-15); Stouffer p=1.9e-29; K1_K2 +0.098 ns(반전 아님). **대조**: `a_fresh_periph_share`(최외곽 13슬롯만)는 셀 간 부호 반전 (node_peak −0.770 E1_E2 vs +0.219 E3_E4) — RM4 반증 재현.
- **위반 시 대가**: 외곽대 전노심 1집합체당 F_r −0.002..−0.006. CBC 비용 동반(+0.410).

---

### 4.6 F5 — hot/cold fresh 상대배치 및 노령연료 배치

#### R11. [hot-fresh-outboard] 두 fresh 타입 중 더 뜨거운 쪽(고kinf0·저Gd·저FF)을 더 큰 반경에
신뢰도: **DOMAIN-LIMITED — E/J 셀 한정** (A3)
- **적용법**: `a_hot_minus_cold_rmean` = (hot fresh 가중 평균반경) − (cold fresh 가중 평균반경) [pitch].
- **근거**: f_r 음성 5/5 채굴 셀: −0.274 (E1_E2), −0.439 (E3_E4), −0.510 (J5_J6), −0.401 (J1_J2|f121); Stouffer p=3.2e-25; ff-gradient 조건부 생존(E3_E4 −0.230, J5_J6 −0.262).
- **유효범위**: **`K3_K4`에서 node_peak +0.422 (p=0.016) 유의 반전, f_r +0.274 ns — K-case_pair 적용 금지**(R2와 함께 family A의 경계).
- **위반 시 대가(도메인 내)**: hot 타입 외향 이동 1 pitch당 F_r **−0.016**(중앙값). node_peak 규칙은 아님(2/5셀). CBC 비용 +0.606.

#### R12. [hot-ring-twice-burned] 셀 명목 출력맵의 hot ring에 twice-burned를 주차하라
신뢰도: **CONFIRMED** (B1); feed < 121에만 twice-burned 존재.
- **적용법**: hot ring H = 누적 가중 다중도 ≥ 48/241까지의 최고출력 슬롯 집합. metric = `Σ_{i∈H} w_i·[age_i==3] / Σ_{i∈H} w_i` (H는 셀 명목 BOC 맵에서 도출; constructor는 fold-공유 상수 사용).
- **근거**: within-cell rho **−0.388** node_peak (94셀, 95.7% 음성, n=5,274), −0.360 f_r, −0.464 map_cov; AUC 0.755; partial|b_hot_fresh −0.332/−0.298, partial|cyclen −0.394 — **독립**. 재검증 보너스: **cbc도 낮춘다(−0.291, 98% 음성)** — A군 반응도-외곽 비용을 **되사는** 규칙.
- **위반 시 대가**: hot ring 가중 share −0.1당 node_peak +0.22 / F_r +0.24 손실.

#### R13. [park-twice-burned-inboard] 3-batch 노심(feed < 121)에서 twice-burned 무게중심을 안쪽에 유지하라
신뢰도: **CONFIRMED** (D5)
- **적용법**: `age3_r` = age ≥ 3 슬롯의 가중 평균반경 / 최대반경 (0..1).
- **근거**: → `cbc_max` wrho **+0.412, 음성 셀 0%** (86셀, n=9,758, p=2.6e-26; partial|kbud +0.497); → map_cov **+0.480**(0% 음성); → f_r +0.316.
- **위반 시 대가**: 무게중심 외향 0.1(노심반경비)당 CBC **+32 ppm**, map_cov +0.044; 안쪽 유지 비용은 cyclen ~−0.8 EFPD뿐.

#### R14. [hot-ring-fresh-two-regime] hot ring의 fresh share는 2-레짐 레버다
신뢰도: **CONFIRMED** (B4) — 거친 탐색에서는 fresh를 hot ring **밖으로**, in-band 엘리트 레짐에서는 챔피언들이 의도적으로 fresh 앵커를 hot ring에 **얹는다**.
- **적용법**: R12와 동일 H; metric = `Σ_{i∈H} w_i·fresh_i / Σ w_i`. **밴드에서 멀면 스토어 부호, in-band 구성 시 반전 부호 적용.**
- **근거**: 스토어 within-cell +0.230 node_peak / +0.258 f_r (285–329셀, n≈40–50k; partial|RM1i +0.211/+0.238 — RM1i와 독립 축); 최신 캠페인 전부 반전: fpcamp4 **−0.721/−0.668**, fpcamp5 −0.33, fpcamp6 −0.34, fpcamp_minfr −0.467/−0.654 (전부 p ≤ 1e-3). 재검증 FLAG: 가중 스토어 rho는 +0.121/+0.173로 집계 민감 (중앙값 +0.34/+0.37, 부호 일치).
- **유효범위**: **2-레짐 구조 자체가 유효범위 선언 — 레짐 오판이 최대 위험.**
- **위반 시 대가**: 스토어 레짐에서 share +0.1당 node_peak +0.057 / f_r +0.083; 엘리트 레짐에서 share +0.1당 f_r −0.096을 놓침.

---

### 4.7 F6/C군 — 피크 캐리어 법칙과 서로게이트 (심층 탐색 셀·스타일 선택 전용)

#### R15. [burned-peak-carrier] feasible 노심에서 BOC nodal peak를 BURNED 집합체(BOC 연소도 ~8–15 GWd/tU)에 넘겨라
신뢰도: **DOMAIN-LIMITED — 심층 탐색 셀(E1_E2/E3_E4) 한정** (C1)
- **적용법**: `carrier_fresh` = `is_fresh(occupant(argmax_s p_boc[s]))`.
- **근거**: E1_E2 AUC 0.653 (n=346), 가중 0.582. 승자 대비: 저F_r 데실 캐리어 **89% burned @ 14.3 GWd/tU** vs 평탄 데실 **74% fresh @ 4.9**; lowFr−flat node_peak +0.0597 CI [+0.011, +0.089]. **재검증이 "5/5셀" 주장 기각: J1_J2 / J5_J6 / K3_K4 AUC 0.413 / 0.484 / 0.324.** 깊이 구배: `rho(node_peak, f_r)` = **−0.289 (E1_E2) → +0.931 (K3_K4)** — 캐리어 스위치는 평탄 앵커 포화 후 마지막 ~0.02–0.04의 F_r을 여는 열쇠이며, **탐색이 얕은 셀에서는 보이지 않는다.**
- **유효범위**: `within-cell rho(node_peak, f_r) < 0`인 심층 탐색 셀. 92.5%의 feasible 행이 이미 명목 최고온 슬롯에 fresh를 두므로(포화) 이 규칙은 **실현된(realized)** 캐리어에 관한 것이다. 명목 평균 맵은 fresh 캐리어를 99.6% 예측하나 실현은 52.4% — ~0.01 맵 해상도의 경주라 **집행에는 맵 헤드가 필요**.
- **위반 시 대가**: 캐리어 fresh→burned 전환 = 평균 F_r ~−0.005(심층 셀 승자 −0.01..−0.04), 평탄도 비용 F_xy +0.03..+0.06.

#### R16. [fr-fusion-surrogate-outer-loop] 융합 서로게이트는 크기(magnitude) 거부권으로만 써라
신뢰도: **DOMAIN-LIMITED — magnitude veto only** (C3)
- **적용법**: `fr_hat = 1.047 · max_{BOC,EOC steps, slots}( p_nom · FF_origin(BU_nom) )`. constructor는 `fr_hat ≤ 1.65` 게이트로 사용.
- **근거**: 크기 법칙 byte-exact 재현: `A_med = 1.0472`, IQR [1.022, 1.084], rms 0.0496–0.0499, n=679 — 전구간 법칙 A=1.0255 / rms 0.035의 in-band 연장. **순위 주장 강등(재검증)**: elite-half +0.26–0.272는 md5 fold 해시에서만 재현, 메트릭이 그룹당 2–3레벨 준이산, `J5_J6` 부호반전(−0.38..−0.68); within-campaign +0.025 null.
- **오용 시 대가**: 스타일 내 순위 기대이득 0, 홀드아웃 반전 위험.
- **보강(`kcurve_fusion_memo_20260809.md`)**: 융합의 **형태(시점)** 가 결정적이다. **BOC 스냅샷 형태**는 결정대역(F_r ≤ 1.55)에서 ρ = **−0.113**(엘리트 역랭킹), **궤적형(주기 최대)** 형태는 rms 0.0350, ρ = **+0.862**(결정대역 rms 0.0081 / MAE 0.0058). 단 셀 내부 판별자로는 못 쓰고(+2.7%), **셀 간 보정에서만 가치** (142셀 평균 R² 0.9752→0.9949, rms −55%). 그리고 **직접 F_r 헤드가 융합보다 27% 우수** (fold C MAE 0.0699 vs 0.0890) — 맵헤드 오차가 A·FF(~1.18배)로 증폭되기 때문.

#### R17. [burned-hot-product-style] 후보 스타일 중에서는 burned-측 hot product가 높은 쪽을 선호하라
신뢰도: **DOMAIN-LIMITED — 약한 스타일 타이브레이커** (C4)
- **적용법**: `hot_burned_nom` = `max_{burned s} max( p_nomB[s]·FF(BU_nom[s]), p_nomE[s]·FF(BU_nom[s] + B_regime·p_nomB[s]) )`.
- **근거**: cell×fold rho vs f_r = **−0.209** CI [−0.357, −0.088], 8/10 그룹 음성 (재검증 −0.204 md5 / −0.190 sha1); within-style +0.036 null.
- **위반 시 대가**: 스타일 간 IQR당 F_r +0.0062 포기 — 작다. **동률 판정용.**

#### R18. [chain-burnup-validity] (지지 규칙) 체인 유도 슬롯 연소도가 맵 수확 BOC 연소도를 순위 수준으로 재현한다
신뢰도: **CONFIRMED** (C5) — §2.5 참조. `BU_nom = B_regime · Σ p_nom[direct 체인]`; 679행 median within-row Spearman 0.618 (재검증 **0.694**), MAE 8.04–8.21 GWd/tU. **규칙 적용에 연소도 라벨 불필요.** **<5 GWd/tU 절대 구분에는 사용 금지.**

---

### 4.8 엘리트 정밀화 규칙

#### R19. [elite-diagonal-checkerboard] in-band 엘리트에서 fresh 접촉은 면접촉이 아니라 대각접촉으로
신뢰도: **DOMAIN-LIMITED — 엘리트 fpcamp 레짐** (B5) — **구성 단계 정밀화 규칙**이며 거친 탐색 규칙이 아니다.
- **적용법**: `b_checker` = `rm_fresh_diag_adjacency(inboard=True) − rm_fresh_face_adjacency(inboard=True)`.
- **근거**: 스토어 null(−0.007, 46.9% 음성); fpcamp4 **−0.785/−0.765** (p≈1e-21, n=97), fpcamp5 −0.665/−0.720, fpcamp6 −0.648/−0.756, fpcamp_minfr −0.247/−0.207; 일반 셀 feasible 부분집합 −0.156.
- **유효범위**: live minFr −0.092 ns; feasible E1_E2 내부에서 rm1i 축과 **−0.96 공선(붕괴)** — 독립 레버가 아니라 **R5의 정밀화 표현**으로 취급.
- **위반 시 대가(엘리트 셀 내)**: 대비 쌍당 node_peak −0.0057 / f_r −0.0059 포기; 무작위 레짐 모집단에서는 기대 이동 0.

---

### 4.9 DEAD 목록 — 통설이 제안할 법하지만 데이터가 죽인 규칙

**채택 금지. 각각 사망 방식이 교훈이다.**

1. **"fresh는 무조건 주변부로"** (RM4, `a_fresh_periph_share`, mean fresh radius) — 셀 간 부호 반전(node_peak −0.770 E1_E2 vs +0.219 E3_E4). 이식 가능한 형태는 **R10만**.
2. **"fresh를 뜨거운 자리에서 항상 치워라"**(naive 첨두-회피 보편형) — 2-레짐(R14)에서 엘리트 측이 정확히 반대(fpcamp4 −0.721). **최평탄 노심은 fresh 고FF 연료에 피크를 얹는다.** 보편 규칙으로는 사망, **레짐 조건부로만 생존**.
3. **"FF > y인 fresh를 명목 p > z 슬롯에 금지"**(집행형 임계 규칙) — C2에서 명시 기각: in-band 92.5% 포화(역할 분리는 feasibility 전제조건이지 판별자가 아님), 임계값 홀드아웃 전이 실패(train +0.428/+0.25 → holdout −0.161/−0.37/−0.42). **포화된 조건을 규칙으로 격상하지 말 것.**
4. **RM2i 최소화(내부 fresh 대각 인접 회피)의 보편 적용** — 최신 캠페인에서 부호 반전 (fpcamp4-6 −0.665/−0.597/−0.425; live minFr +0.538). 스토어 양성은 RM1i 축 편승 (공선 +0.385; partial diag|face +0.080).
5. **fresh-burned 혼합쌍 극대화** — 스토어 null(−0.006/+0.024, 부호검정 ns); 엘리트 −0.60..−0.78은 RM1i의 거울(공선 −0.511). 독립 레버 아님.
6. **내부 체커보드 차수**(RM6 변형 `b_chk_inb`) — zero-order −0.105가 partial|RM1i +0.020으로 소멸.
7. **주변부/교차 fresh쌍 "보호"**(`b_ff_face_pp`/`b_ff_face_x`) — partial|RM1i −0.043/−0.034로 붕괴; AUC 0.364는 독립 기량이 아님.
8. **전축 twin fresh 수**(`b_twin_fresh`) — +0.045, 26% 음성, AUC 0.545. 신호는 내축(k=1–4)에만 있음(G2 관련).
9. **연령 혼합비 튜닝 / `tolerance_margin` 최적화 / twin 깨기 / self-feed** — 구조적 **비-레버**(G3). 242−2F 항등식이 twice-burned 수를 고정; twinbreak/selffeed는 71,517행 중 0회; `tolerance_margin`은 전 메트릭에 |wrho| ≤ 0.045.
10. **fresh ring1(2.5–4.5) vs node_peak** — 수치는 통과(+0.63/+0.24/+0.21)하나 E1_E2 지배 + E3_E4 데실 AUC 모순으로 채굴 단계에서 이미 배제(정직성 기록).
11. **once vs twice-burned 배치 구분 규칙 (in-band)** — 검증 불가: feasible 891행 전부 depth 1(twice-burned 0개). **부재를 규칙 부재로 오독하지 말 것** — feed < 121 스토어에서는 R12/R13이 유효.

**규칙집 v1 이후 추가로 죽은 가설**

12. **(k-inf 곡선 + 농축도) = 집합체 충분통계** — STEP0 MASTER null test로 **최종 반증** (처리군 ΔF_r 0.0351 / Δnode_peak 0.0072 vs 음성대조 0.0016 / 0.0005, **효과비 14배**).
13. **고정 패턴에서 연료 스왑의 방향·크기 예측 가능** — Δnode_peak 부호적중 **0/8**, Spearman −0.214, 진폭 2.6–4.2배 과소.
14. **LP 탐색과 집합체 설계의 분리 가능** — MASTER 실측 source→target node_peak 상관 **−0.036**(모델 주장 +0.803). **"평탄한 LP를 먼저 찾고 나중에 연료를 갈아끼운다"는 워크플로는 반증됨.**
15. **F_r = A · p_boc(상수) · FF (순진 분해식)** — C6 널테스트로 반증: 같은 `FF_hot`(1.178)인 두 노심이 F_r 1.8175 vs 1.5483(**Δ0.269**). **p_boc는 상수가 아니다** — 저출력 68칸 연료 교체만으로 node_peak 1.2217→1.5514. 고정 p_boc 가정 R² **0.142** vs 실측 node_peak 사용 R² **0.901**.
16. **반응도 정합(cyclen 정합)이 map_cov/node_peak를 결정한다** — |Δcyclen|이 설명하는 Δmap_cov R² 0.093 / Δnode_peak R² 0.020뿐. **기각**(초기 결론의 자체 정정).
17. **농축도가 두 번째 유효 특징** — ADF에 압도적 열세(ΔR² +0.0058 vs +0.0305). 기각.
18. **face-ADF 채널 추가**(v6c, 58→62ch) — 게이트 point **−0.01444** (CI [−0.01935, −0.01112], 전 구간 음수), harm rail `T_cell_mae_cyclen` −0.300까지 위반 → **REJECT**(§2.4).
19. **PCA 형상점수 3개**(v7 채널 후보) — 순열중요도 ≤ 0.0019, 라벨이 보상 안 함. 기각.
20. **policy_v1의 캠페인 배선** — parent-blocked AUC **0.492**(우연 수준). 기각(§4.10-L).
21. **`batch_swap`이 모든 분지에서 유효한 연산자** — 625 분지에서 10배 붕괴(p=0.0028). 반증.
22. **템플릿 레이아웃 그대로의 신규 DeCART 격자가 F_r을 개선** — 984개 합법 설계 전량이 대체 대상 ga80 타입보다 더 뾰족함(1.1658 vs E1 1.146). **레이아웃을 개방해야만 역전.**
23. **MASTER tolerance 강화로 정확도 획득** — 실측 기각(라벨 상한이 이미 ρ ≥ 0.9997, 비용 +10~20%, ρ 기여 0.000).

---

### 4.10 규칙집 v1 이후 확정·발견된 규칙 (2026-08-11 ~ 08-20)

#### (A) 누설-중재(leakage arbitration) — SETTLED
`ablation_wave_results_20260815.md` (2026-08-15, 150체인, 146수렴 97.3%). 신뢰도 **높음(개입 실험)**.

**"fresh를 반경 바깥으로 놓으면 주기길이가 짧아진다"** — 두 개의 **반응도 보존 계기**로 확정.

| 계기 | mean(out−in) Δcyclen | 95% CI | 부호검정 | p | dose-response 기울기 |
|---|---:|---|---|---|---|
| `fresh_relocate` (고선량) | **−1.7415** | [−2.13, −1.33] | 0/10 | 0.002 | **−8.37** EFPD/unit [−10.58, −5.88] |
| `batch_swap` (중선량) | **−0.1128** | [−0.20, −0.03] | 1/10 | 0.021 | **−9.41** EFPD/unit [−15.53, −4.46] |
| `batch_flip` (비적격 계기) | +3.1827 | — | 9/10 | 0.021 | 잡음(배제) |

- `batch_flip` 배제 근거: `corr(Δcyclen, Δfresh_enr_mass) = 1.000` (n=20) — 변화가 전부 반응도이고 방사방향 도즈는 거의 무관(corr with r_center = 0.343).
- **관측적 코퍼스의 반대 부호 결론을 재반박**: `policy_corpus`의 `d_fresh_share_periph` vs `d_cyclen` = **+0.093**(부호 오류) → 개입 표본에서는 **−0.635**. 평탄화 쪽도 −0.131 → **−0.545**로 코퍼스가 ~4배 과소읽음.
- **평탄화도 outward가 우위**: F_r mean(out−in) **−0.404** [−0.532, −0.236], node_peak **−0.357** [−0.459, −0.222].
- **함정 경고**: 그런데 **"개선 비율(improving fraction)"은 반대 방향**을 가리킨다 — inward 개선율 0.143(꼬리 위험 큼, max +1.86) vs outward 0.036(안전하나 상한 +0.21). **개선-비율 지표로 정책을 학습시키면 복권형(위험한) 쪽을 선호하게 된다.**

> **규칙(확정)**: **fresh 외곽 배치 = 평탄화 이득 + cyclen 지불.** 부호는 확정, 기울기는 부모집합 의존적이므로 **부호와 자릿수로만 인용**한다 (batchswap 웨이브 n=213에서 기울기 −9.41 → **−21.19**, CI 비중첩).

#### (B) cyclen 밴드가 저밴드 F_r을 묶는다 — 확정 (2026-07-23~24)
`fr_boundary` 생산 캠페인(cyclen 무게이트, e5.0–5.5 × f113–125)에서 **전 6밴드 서브-1.55 F_r 실측**(최저 `E1_E2_f125` **1.5085**, `J1_J2_f121` 1.5095; 스토어 51,338행, 서브-1.55 라벨 491개). → **cyclen 밴드 제약이 저밴드 F_r을 묶고 있었음이 실측 확정.** 신뢰도 **높음**. 동시에 in-band 안에서는 cyclen–F_r 트레이드가 거의 없다(§1.2) — 즉 밴드는 **경계에서만** 구속한다.

#### (C) `batch_swap` 연산자 — 618 분지 특이성
`batchswap_wave_results_20260815.md`(220체인, 100% 수렴) / `batchswap625_wave_results_20260815.md`(220체인). 신뢰도 **높음(사전등록 + Fisher 검정)**.

- **618 분지**: 신규 셀 기록 **F_r 1.4605**(feasible, 4축 통과) — ga80 기존치 1.4636보다 −0.0031 낮음. 개선 비율 **0.052**(11/213).
- **625 분지**: 개선 비율 **0.005**(1/218) — **10배 붕괴**. 최고 ΔF_r −0.0009. **Fisher 정확검정 p = 0.0028** — 잡음이 아님.
- **결론**: `batch_swap`은 19,820 동일-셀 이동 중 **4회**만 재생된 저재생 클래스였고 618 분지에서는 강력했으나 **일반화되지 않는다**. **연산자 일반화 권고는 철회**되고 "**branch-dependent, 사전예측 불가**"로 약화되었다.
- **부수 규칙(운영)**: **λ-목적 검사는 모든 프런티어 판독에 의무**다. F_r 단독 기록 1.4605는 λ=400 목적함수로는 r8 기록(1.4749 @ 625.46)에 **net −1.68 EFPD-eq**로 진다(ga80 기존치 대비 −14.07). **F_r 단독 헤드라인이 2회 연속 뒤집혔다.**
- **셀 내부 두 분지**: ~618 EFPD 분지(F_r 프런티어가 사는 곳)와 ~625 EFPD 분지 (캠페인 최적점이 사는 곳)가 물리적으로 구분된다.

#### (D) 어블레이션 웨이브에서 각 연산자/피처의 기여
- **연산자 클래스별 개선율**(`policy_corpus_20260815.md`; same-cell MOVE 19,820개, F_r 라벨 19,726개 중 개선 6,801 = **34.3%**):

  | 무브 클래스 | n | F_r 개선율 | 평탄도 개선율 | CBC 개선율 | cyclen 연장률 |
  |---|---:|---:|---:|---:|---:|
  | `rewire_swap` | 10,536 | 37.5% | 38.0% | 44.0% | 52.8% |
  | `fresh_relocate` | 6,021 | 30.0% | 35.5% | — | — |
  | `batch_flip` | 1,438 | **44.4%** | — | 52.2% | — |
  | `batch_swap` | 4 | (저재생) | — | — | — |

- **방사방향 규칙(관측)**: fresh를 outward로 밀면 F_r 개선 **41.2%** vs inward 23.1%; node_peak 44.8% vs 26.2%. 연속변수 상관 `d_fresh_share_periph` vs `d_f_r` = −0.131, vs `d_node_peak` = −0.110.
- **엘리트 방사분석**: F_r-elite는 all-comers 대비 주변부 fresh 비율 평균 **+0.015** (13셀 중 10셀 양수), node_peak-elite는 **+0.042**(11셀 중 10셀 양수) — **좋은 노심은 fresh를 더 바깥쪽에 놓는다.** all-comers조차 이미 peripheral fresh 0.57 vs inner 0.42로 in-out 성향.
- **피처(모델) 어블레이션 — 구조 > 용량**: 고분해능 A/B(2026-07-25, 7-arm 사전등록)에서 **구조 개선**(멀티스케일 맵 디코더 + 확산 출력프라이어 잔차 + 스펙트럼 손실 + 국소대비) = Δ75/SD 1.41 → 0.70(**+0.705**) vs **순수 용량 4.2배**(13.14M) = **+0.141** → **구조가 용량의 5배, 파라미터 효율은 44배**. 폭만 키우면 나이퀴스트 파워비가 오히려 악화.
- **스케일링 무릎점**(2026-08-15): 용량 무릎점 = 현 구조 10.4M×5(224/8/384). 확대·축소·앙상블10 전부 기각. **실측 노이즈 바닥**: 앙상블쌍 ~0.01(cyclen/map_cov), 단일시드 ρF_r sd ~0.018 — **이하의 델타는 개선 주장 불가**.
- **메커니즘 장부**: 손실/라벨공학 **DEAD**, 앙상블 **DEAD**, 결핍셀 라벨증산 **ALIVE** (스플릿 arm PASS), verify-many **SUPPORTED**, 연소도 배치 **PASS**, face-ADF **DEAD**.

#### (E) 셀 우세 — `J5_J6` > `K3_K4`
동일 32패턴 정면비교에서 **`J5_J6`가 `K3_K4`를 27/32 우세**(F_r · 평탄도 · 주기여유 3축). → `J5_J6`가 차기 셀로 선정(fpcamp5). 신뢰도 **중간**(단일 비교 실험, n=32).

#### (F) feed 자유도가 필수임의 실증
스팟체크에서 `(5.5 wt%, f109, 625 EFPD)` best |Δ| = **1.35 EFPD**로 양호했으나, **f117은 물리한계(자연 cyclen 666 EFPD)로 목표 밖**이었다. 즉 목표 주기길이를 고정하면 **feed를 자유변수로 두지 않고는 도달 불가능한 셀이 존재**한다 → `user_criteria` 모드의 feed 자유도가 필수. 신뢰도 **높음(실측)**. 보강: **feed ↑ → cyclen ↑ 단조**(§1.2)이므로 feed는 cyclen을 옮기는 1차 레버다.

#### (G) F_r 1.45는 붕소 문제 — CBC 완화의 근거
2026-08-11 사용자 결정. 근거: `E1_E2/f121` 패턴 축 소진(끝점 추정 **1.4624** vs 기록 **1.4636**) + **F_r 1.45는 붕소 문제**라는 판정 → `cbc_limit` **1550 → 1600**. 그 회랑 덕분에 신규 저-FF 격자쌍이 존재할 수 있게 되었다(Y3 FF 1.1012 / CBC_pred 1501, Y4 FF 1.1011 / CBC_pred **1562** — **1550에서는 Y4 절반이 구성상 infeasible**). 신뢰도 **높음(설계 결정 + 스크린 수치)**.

#### (H) 저 FF 격자 설계 축 — 실측 판정
`flat_assembly_fr_plan_20260802.md` / `flat_assembly_fr_verdict_20260809.md`.

- **레이아웃 > 농축도**: 동일 노브(u_high 5.00, gd_wt 6, n_gd 20)에서 템플릿 레이아웃 `2:2;5:2;6:4` FF = **1.1659** vs ga80식 `1:1;4:1;6:4` FF = **1.1166** → **ΔFF 0.049 ⇒ ΔF_r 0.065**. 농축도 튜닝 가치 ~0.002 — **레이아웃이 농축도의 25배 가치**. 신뢰도 **높음**.
- **BLOCKER(중요 negative)**: 984개 합법 설계(4개 동결 템플릿 레이아웃) 전수 열거 결과 **매칭 반응도에서 전부가 대체하려는 ga80 타입보다 더 뾰족함** (1.1658 vs E1 1.146; 1.1753 vs E2 1.152). **레이아웃을 개방해야만**(89개 합법 배치, 5,874 설계) 역전(1.1151 vs 1.146 = −0.031).
- **사전등록 반증조건 F1 발화**: 3점(A0/A1/A2)으로는 기울기 0.912 / R² 0.980으로 확증처럼 보였으나 **7점으로 늘리자 기울기 0.508** → **"고정 p_boc" 순진 분해식 반증**.
- **연료 레버 vs 장전모형 레버**: 장전모형 탐색 F_r 이득 **−0.057** vs 연료 레버 중앙값 **−0.0069** — **8배**. 그리고 383개 실측 f121 노심에서 pair별 중앙 F_r을 `FF_hot`에 회귀하면 기울기 0.185, r = 0.37 — **자유 장전모형에서는 LP가 FF 차이를 흡수한다.**
- **앵커 의존성**: minfr 앵커(F_r 최적 패턴)에서는 방향이 **뒤집힌다** — A0 1.4636 → E3_E4(평탄 연료) **1.4877(+0.0241 악화)**. → **"평탄 집합체를 넣으면 F_r이 낮아진다"는 보편 규칙이 아니다.**

> **규칙(확정)**: 집합체 격자 설계(FF)는 실재하는 레버이나 **2차항**이다. 지배항은 연료 반응도가 유발하는 **노드 출력 재분포**다. 단일 패턴 치환은 도박이고, **모집단 수준·평탄 앵커에서만** 이득이 나온다.

#### (I) 핀 연소도 ↔ 농축도 ↔ feed — "F_r은 열렸으나 핀으로 막힌다"
신뢰도 **높음(MASTER 25기 실측 판정: 10 PASS / 15 FAIL)**.

| 캠페인 | feed | F_r 승자 | 예측 핀 | 실측 핀 | 판정 |
|---|---|---:|---:|---:|---|
| `N1_N2` / f113 | 113 | **1.4961** | 86.75 | **86.19** | **0/5 FAIL** |
| `E1_E2` / f109 | 109 | **1.4787** | 83.16 | **82.11** | **0/5 FAIL** |
| f113 핀게이트(P5) | 113 | 1.5074 | 77.09 | **84.20** | **0/5 FAIL** (−7.11 오차) |
| `HGD569` / f125 2종 | 125 | 1.6357 | 76.96 | **75.47** | **5/5 PASS** |
| `HGD569` / f125 3종 | 125 | 1.5993 | ~75.2 | **74.38** | **5/5 PASS** |

- **저 feed(= 높은 노심평균 농축도) 쪽에서 핀 조건이 깨진다.** `E1_E2/f109`는 52개 feasible 코어 **전부(100%)** 예측핀 ≥ 80(81.31–83.77); `N1_N2/f113`는 41개 전부 86.14–88.26.
- **`min_fr_max_cycle`은 핀을 전혀 게이트하지 않았다**(`feasibility_limits_for`가 `max_pin_burnup: None` 반환) → **"F_r로는 열렸으나 납품에서는 핀으로 막힘"** 이 3개 캠페인에서 반복된 **프로그램 수준 발견**. 이후 `minfr_pin_bu_limit = 78.0`이 5번째 게이트 축으로 추가되었다(2026-08-17).
- **DB 셀 특성은 자사 코어의 대리지표가 아니다**: `N1_N2/f113`은 DB에서 온건해 보였으나 (`frac_node_ge75 = 0.378`, DB best 73.30 PASS) 우리 탐색이 저-F_r을 향하면서 **연소도를 집중**시켜 전량 위반.
- **예측 신뢰도의 진짜 판별자는 학습 지지영역까지의 거리**: `n_within30`(Hamming 30 이내 학습 코어 수)과 오차의 Spearman **+0.553**(p=0.00018). 학습 코어가 0인 분지에서 bias **−5.93**, ≥28이면 +0.35. optimizer's curse는 **아니다**(몬테카를로 20만회 상한 −2.12~−2.44로 관측의 ≤40%).
- **78 마진의 한계**: 2.0 GWd/tU 마진을 산정한 근거(feed-113 bias −0.83, n=17)가 **14/17이 학습 fold 내부**였다 — 일반화 검정이 아니었다. **납품 목표 캠페인은 승자를 반드시 실측하라**(예측만으로 출하 금지).
- **핀은 B_asm만으로 낮출 수 없다**: `pin = ratio × B_asm`에서 B_asm 76.0→68.8(−7.2)로 낮추면 ratio가 1.134→1.223(+7.9%)로 반비례 상승 → 순개선 86.19→84.20의 **2.0 GWd**뿐.

#### (J) 실측 DB 대조 — 모델은 사실상 무편향, "미발견"은 커버리지 공백
`data/reports/scoping_mesh_20260815/comparison_readout.md` / `reloadmap_methodology_20260816.md`. 신뢰도 **높음**.

- 프런티어에서 모델의 F_r 편향은 평균 **|0.028|**(챔피언 s1e 기준 **|0.024|**) — **모델은 사실상 무편향**.
- pool-starved 20셀 분해: **총격차 +0.132 = 모델편향 |0.028| + 자료격차 +0.162 + 탐색격차 −0.044.** → **문제는 학습자료가 그 영역을 담지 않은 것.**
- "열렸다" 예측 **8/8 정확**, "닫혔다" 예측 **22/22 오답** — **정확한 모델의 정확한 침묵이 "닫힘"으로 잘못 렌더링**되고 있었다.
- **비최적 라벨의 편향 전파**: feedgrid 371행 라벨은 F_r ≤ 1.55가 0개(비최적 생산샘플). 직접 라벨을 받은 3셀은 −0.0366(개선)인데 **라벨 못 받은 이웃 15셀은 +0.0075 (비관 방향으로 오염)**, 라벨 전혀 없는 f105 열까지 +0.0144 밀림. → **앵커링은 반드시 min-F_r 최적화로 하고, 생산샘플링으로 하지 말 것.**
- **DB 대표점 정의가 결론을 뒤집는다**: 대표점을 "최장주기 노드"에서 "최저 F_r Pareto 대표"로 바꾸자 평균 격차 **+0.0259 → −0.0090**(8셀 중 5셀에서 우리 모델이 DB보다 평탄).
- **재장전 지도 방법론 처방**: 지도의 **뼈대(주기길이·방출연소도 축)는 서로게이트가 아니라 실계산 보정 LRM**이 그려야 하고, 서로게이트는 **안전인자 오버레이**여야 한다. 근거: `B_d = B_c × 241/feed` 항등식이 **−0.519% ± 0.098%** 로 성립(y축은 이미 LRM 항등식, MASTER 38,220행 평균 −0.002%), 보정 LRM의 cyclen rms **4.26 EFPD (0.66%)**.
- **교과서 LRM의 `2/(n+1)` 형태가 feed축에서 ±1.8% 틀린다**: `f109 ridge` — f109가 나머지 5개 feed를 지나는 매끄러운 곡선보다 ~1.3%(≈8 EFPD) 위(20 pair 중 19 pair). → **아웃루프 스크리닝 백본은 해석적 형태 대신 반드시 feed별 자유계수를 가져야 한다.**
- **`bu_k1`이 농축도보다 나은 서술자**: enrichment → `bu_k1_mix` 치환만으로 동일 파라미터 수에서 cyclen rms **9% 개선**(2.364 → 2.159 EFPD). 두 서술자는 상보적 (농축도 = 핵분열 재고량, `bu_k1` = Gd 설계/조닝 정보).
- **F_r 최저치의 feed 의존은 U자형**(DB 6,113 코어 기준): f101 1.4958 / f105 1.5024 / f109 **1.5290(최악)** / f113 1.5226 / f117 1.5129 / f121 1.4971. 그런데 **f101–f105는 그 평탄도를 핀연소도로 지불한다**(해당 코어 전량이 node 핀 75 초과) → **LEU+ 인허가용으로는 f117–f121 계열이 방어 가능**하며 이 영역이 스토어가 가장 덜 다룬 곳이다.

#### (K) 다종(multi-type) 배치 규칙 — R1 경계에서 부호가 갈린다
`data/reports/mesh_multitype_20260818/README.md`(90셀 스윕, MASTER 미사용) + `tripletype_f125_results_20260817.md` + `hgd569_f125_seedctl_results_20260817.md`.

**K1 — 계단화 이득은 설계규칙 R1 경계에서 정확히 갈린다.** 신뢰도 **높음(부호 만장일치)**

| 부류 | n셀 | 평균 ΔF_r(3종−2종) | 이득/손해 | 핀 Δ |
|---|---:|---:|---|---:|
| **R1 mono-spec 3종**(한 농축 사양 안에서 Gd 계단화) | 10 | **−0.0378** (최대 −0.1377) | **10 이득 / 0 손해** | **−0.70** GWd/tU |
| R1 cross-spec 3종(서로 다른 농축 사양 혼합) | 19 | **+0.0749** | 0 이득 / **19 손해** | +0.43 |

- **읽는 법**: 한 농축 사양 안에서 Gd를 계단화하면(ga80 E3/E1/E2 = n_gd 16/20/24) F_r과 핀연소도가 **함께 내려간다**. 반면 노심평균 농축도를 맞춘 채 **서로 다른 농축 사양을 섞는 것**(K1_J1_E1 = 5.2/5.1/5.0)은 계단화가 아니라 그냥 다른 연료 세트이고 일관되게 **나빠진다**.
- **R1(패턴 내 전 feed 동일 농축도 스펙)은 인허가 제약으로 쓰인 규칙인데, 계단화가 도움이 되는 경계와 정확히 일치했다** — **규칙을 물리로 확인해 준 결과**이지 규칙을 완화할 근거가 아니다.
- **4종은 0/42셀**에서 joint-clean 노심을 못 냈으나 이는 학습 저장소에 4종이 0행이라 **모델이 4종을 모른다는 사실의 재확인**이지 물리 판정이 아니다(전 행 `EXTRAPOLATION` 표기).

**K2 — mid-type은 "소량 전이환(transition ring)"으로.** 신뢰도 **중간(단일 셀 within-run 상관)** `tripletype_f125` within-run: `r(mid-type 개수, F_r)` = **−0.42**, `r(mid-type, CBC)` = **+0.64** → **내부 최적점이 존재하는 진짜 제어 노브**. 최적 조성 hot/mid/cold = **57 / 20 / 48**(mid는 feed의 **16%**) — 설계노트가 제안한 등분(1/3씩 ≈ 42)도, 최저치(4)도 아니다. 29개 clean 코어의 mid 중앙값 8.

**K3 — 계단화는 CBC를 완화한다.** 신뢰도 **중간~높음(대조군으로 독립 확증)** CBC ≤ 1600 통과율 2종 28.1% → 3종 **59.2%**(clean 코어 13/57 → 29/49, **2.6배**). 2종 대조군(동일 시딩, alphabet 2 고정)은 20.8%로 원본 2종과 유사 → **CBC 완화는 시딩 아티팩트가 아니라 계단화 고유 효과**. (단 `r(F_r, CBC)` 부호 반전은 alphabet이 아니라 **시딩**을 따라간 것으로 정정됨.)

**K4 — 헤드라인 이득의 절반은 시딩이다.** 신뢰도 **높음(통제 실험)** 2종(구식 시딩) 1.6357 → 2종(donor-enriched 시딩) **1.6172** → 3종(동일 시딩) **1.5993**. 분해: **시딩+모델 −0.0185(50.8%) / 3번째 연료종 자체 −0.0179(49.2%)** — **거의 정확히 반반**. → 다종 이득을 인용할 때는 **−0.018 수준**으로 말해야 한다.

**K5 — 스윕은 캠페인 도달거리의 하한.** 유일한 실측 대조점(`HGD569/f125`)에서 예측 Δ −0.0044 vs 실측 Δ **−0.0364**: **부호 재현, 크기는 8배 과소**(비 0.12; 핀은 0.37). 2종 바닥의 절대 수준은 잘 맞음(예측 1.6331 vs 실측 1.6357, 0.0026 차). → **방향은 인용 가능, 크기의 신뢰구간은 주장하지 않는다.**

#### (L) 정책 코퍼스가 드러낸 "개선 무브"의 통계
`policy_corpus_20260815.md` / `policy_v1_results_20260815.md` / `policy_v2_results_20260817.md`.

- **코퍼스 규모**: store 72,685행(64,405 수렴), lineage edge 27,458개. **같은 셀 MOVE 19,820개**가 정책 액션공간의 본체이고, cross-cell TRANSFER 7,638개는 별도.
- **코퍼스의 형태**: 최장 lineage 79 edge인데 **최장 F_r-개선 체인은 7** — 코퍼스는 "좋은 보드 주변 **1-step 이웃집합**"이지 장기 개선 궤적이 아니다. → **정책은 "다음 한 수"를 배우지, 계획을 배우지 않는다.**
- **v1 판정**: in-distribution에서는 강함(F_r AUC 0.790, parent-blocked **0.771**, p@32 0.842; 미지 **셀** 전이도 최고 AUC 0.822) — **그러나 시대(era) 전이 실패** (실제 운영점 ga80+paramA에서 AUC 0.650, p@32 0.328로 3개 베이스라인에 못 미침). **전향 개입 실험에서는 parent-blocked AUC 0.492(우연 수준)**; 특히 v1은 outward `batch_flip`을 P=0.814로 최상위 랭크했으나 **실측 0/10 개선** — **v1은 반응도/배치 라벨 프록시를 학습했고 분포 밖에서 부호가 뒤집힌다.** → **배선 금지.**
- **v2 판정**: 현재 era 가중 + `d_fresh_enr_mass` 공변량 + 손실/조기종료 재설계 (Huber-on-sigmoid → BCE-with-logits + rank 기반 조기종료; 원 프로토콜은 5시드 전부 epoch 0 조기종료로 **학습되지 않은 객체**를 산출했다). 결과: 현재 era parent-blocked AUC **0.728**(CI 하한이 0.50을 0.001 차로 통과), regret@8 **0.00366**(4개 베이스라인 최선), **precision@32는 class_freq를 못 이겨 게이트 절2 FAIL** → **부분 통과**. v1의 대표 오류를 v2가 교정: 문제의 층을 하위에서 두 번째로 랭크, Spearman(층평균점수, 층개선율) **v2 +0.605 vs v1 +0.344**, 개입 half에서 parent-blocked AUC **0.851 vs 0.614**, regret@8 **10.6배 낮음**.
- **채굴 고갈**: 현재 era lineage 6,318개 중 **6,297개가 이미 채굴 완료** — 다음 증분은 반드시 **새로 측정(개입 웨이브)** 해야 한다.
- **CNN vs MLP**: 보드텐서(CNN 1.69M) AUC 0.790 vs 스칼라 전용(MLP 145K) 0.643 → 36개 무브 서술자(무브 클래스·edit 수·ring 델타·방사방향)는 신호의 약 1/3만 담당한다. **배치의 공간 구조가 나머지를 담는다.**
- **물리 프라이어와 상보적**: 확산 파워맵 프라이어는 좋은 **보드** 랭커 (parent-blocked 0.535, 거의 우연)이고 정책은 좋은 **무브** 랭커(0.752). era 밖에서는 프라이어가 정책을 이긴다(0.746 vs 0.650).

---

### 4.11 규칙 간 상충 — 측정된 트레이드 구조

#### (1) 평탄도(node_peak) vs F_r — in-band 반상관 기전

가능영역 안에서 두 목적은 같은 방향이 **아니다**(13-pt dose r = −0.750). 같은 레버가 부호를 바꾼다.

| 레버 | node_peak | f_r | 셀 |
|---|---|---|---|
| fresh-on-crest (R6) | **−0.680** | **+0.508** | E1_E2 |
| 피크 캐리어 fresh (R15) | −0.0288 (AUC 0.336) | AUC 0.582 (fresh→고 F_r) | 심층 셀 |
| hot-ring fresh share (R14) | 엘리트 −0.72 | 엘리트 −0.67 | fpcamp4 (동방향) |

- **flat 프로파일**: 피크를 fresh 고FF 앵커에 **얹는다**(R14 엘리트 부호, R6 역방향 허용, R1/R2 최대). 대가: F_r +0.03..+0.06.
- **minfr 프로파일**: 평탄 앵커(역할 분리)가 포화된 뒤 피크를 burned(~8–15 GWd/tU)에 **넘긴다**(R15) + R6/R8/R9/R10 F_r 문법 최대. 대가: node_peak +0.0597 [+0.011, +0.089].
- **이 상충은 탐색 깊이 의존적**: `rho(node_peak, f_r)` = −0.289(심층 E1_E2) vs **+0.931**(얕은 K3_K4). **얕은 셀에서는 두 목적이 아직 동행**하므로 상충 관리가 불필요하고, 기록 경신 국면에서만 프로파일 분기가 의미를 갖는다.
- **실측 분기점**: `E1_E2/f121` F_r 기록 1.4636(n_hot 64) vs flat 기록 1.1899(n_hot 68) — **feed split부터 갈라진다**(R4).

> **주의(정정)**: `flatness_first_program_20260725.md`는 셀내 `rho(node_peak, F_r) = 0.983`(map_cov 통제 후 편상관 0.901)을 근거로 node_peak을 주항으로 승격했다. 이 **양의 상관은 넓은 모집단(밴드 밖 포함)** 의 성질이고, 위의 **−0.750은 가능영역 내부(13-pt dose)** 의 성질이다. **두 수치는 모순이 아니라 서로 다른 모집단을 잰 것**이며, 규칙 적용 시 자기 위치(밴드 내/외)를 알아야 한다는 R14/R19의 레짐 원칙과 같은 이야기다.

#### (2) CBC 공동예산 — family-A 레버는 붕소로 지불한다

| 지출 (cbc rho) | 수입 (cbc rho) |
|---|---|
| periphery-k (R2) **+0.691** | hot-ring-twice-burned (R12) **−0.291** (98% 음성) |
| hot-fresh-outboard (R11) +0.606 | inward migration (G4) 중앙값 −0.125 (69% 음성) |
| k-gradient (R1) +0.577 | 짧은 travel (R7 회피) — travel이 +0.377이므로 회피가 곧 수입 |
| ring3 fresh (R10) +0.410 | twice-burned 안쪽 (R13) — `age3_r` +0.412 회피 |

환율은 **R3**이 정한다: **+0.001 kbud = +13.8 ppm CBC = +0.72 EFPD** — **cyclen으로는 CBC를 되살 수 없다.** Construct는 A-family 레버를 쓸 때마다 CBC를 공동예산에서 차감해야 하며(constructor의 p98 envelope cap이 이 규율), cyclen은 620–645 밴드 중심에 두는 **슬랙 변수**로만 취급한다.

#### (3) 공선성 지도 — 다이얼은 6개뿐

`E1_E2|f121` feasible(n=381) 인자 구조:

| 인자 | 구성 규칙 |
|---|---|
| **F1** 반응도-외곽 | R1 · R2 · R4 · R5(−) · R19 · kbud · inward (in-band에서 kbud ≈ hot_share ≈ periphery-k, 0.94–0.99) |
| **F2** 크레스트 / travel | R6 · R7 |
| **F3** ff_radgrad | R8 (독립) |
| **F4** ring0 vs ring3 | R9 · R10 |
| **F5** hot_cold_dr + hot-ring fresh | R11 · R14 |
| **F6** hot_nom_traj | R16 |
| + 포화 게이트 2 | G1 · G2 |

**점수 함수를 짤 때 같은 인자 안의 규칙에 가중을 중복 배분하면 그 축만 과대집행된다.**

---

## 5. 규칙이 코드에 구현된 방식

### 5.1 하드 제약 — `lpopt/search/genome.py` · `construct.py`

구조 검증은 전부 **`GeneralOrbitGenome.validate()`**(`lpopt/search/genome.py:249-328`) 한 곳에 집중되고, 위반은 `GenomeError`로 raise된다. `construct.py`는 이 예외를 잡아 후보를 그냥 버린다(soft-reject; 예: `lpopt/search/construct.py:274-276`, `:396-397`, `:419-420`).

| 규칙 | 구현 | 위치 |
|---|---|---|
| **feed 그리드 1+4N** | `feed_from_fresh_units` / `fresh_units_from_feed`가 항등식 강제; grid 밖 N은 `GenomeError` | `genome.py:120-134`, `:900-907` |
| **노심 1/4 대칭 + twin/orbit** | 지오메트리가 `ORBIT_UNITS`(4중/2중/1중 궤도단위)로 표현. `from_pattern`이 각 unit의 두 arm의 fresh/burned·batch·rotation 일치를 검사 | `genome.py:379-421` (특히 `:382-386`) |
| **설계규칙 R1 — 패턴 내 전 feed 동일 농축도 스펙** | `Pattern.validate_case`가 패턴의 fresh batch 집합이 case pair(`"E1_E2"` 등) 밖으로 못 나가게, feed가 매니페스트와 일치하게 검사 | `lpopt/vendor/masterrl/domain.py:213-221`; 호출: `rule_construct.py:562`, `rule_acid_run.py:352` |
| **설계규칙 R2 — 집합체 1/8 대칭 · rotation 규약** | `Pattern.validate_quarter_conventions` — 중심 슬롯 fresh/rotation=0, shuffle 카드 rotation이 궤도군에 따라 1/2로 고정, 중복 shuffle 카드 금지 | `domain.py:223-242` |
| **연령 센서스 242−2F** | `twice_burned_count(feed) = 242 − 2·feed`; depth-2 edge 수 = `60 − 2N` | `genome.py:149-152`, `:137-140`, `validate():295-317` |
| **depth ≤ 2 체인 · 비순환** | `_depths()`가 순환·미해결 소스를 탐지; `max_depth > max_shuffle_depth`(기본 2)면 거부 | `genome.py:216-245`, `:287-293` |
| **self-feed 금지 · 소스 유일성** | 소스 중복 소비 검사 + `fresh_set ∩ burned_set` 상호배타 | `genome.py:280-284`, `:269-272` |
| **strict consumption** | N ≤ 30이면 미소비 fresh 0; N > 30(할인방전)이면 `2N−60`개만 허용 | `genome.py:300-305`, `:312-327` |

> **중요 — 인접 금지는 하드 제약이 아니다.** RM1/RM2는 명시적으로 **soft-penalty 후보로만** 채택되어 있고(`rule_metrics.py:153-157` `VALIDATED_PENALTY_METRICS`), 위반해도 `GenomeError`를 던지지 않는다. 이는 §3.1의 "휴리스틱을 H로 승격하지 말라"와 §4.9의 DEAD 목록(대각 인접 보편 적용 사망)에 따른 **의도적 설계**다. 하드로 구현된 유일한 인접류 규칙은 `rule_construct.py`의 G2뿐이다.

### 5.2 규칙 지표 — `lpopt/search/rule_metrics.py`

모든 함수는 순수 `pattern → float`(RM5만 `map → float`). **부호 규약: 양수 = 더 나쁨**(node_peak 기준).

| 함수 | 계산 | 상태 |
|---|---|---|
| `rm_fresh_face_adjacency` (RM1, `:265-289`) | 전노심 17×17 face-인접 fresh-fresh 쌍의 다중도 가중 카운트; `inboard=True`면 양쪽 다 non-periphery | **채택(soft)** — RM1i rho +0.235, RM1 +0.085 |
| `rm_fresh_diag_adjacency` (RM2, `:292-308`) | 대각 4방향 버전 | **채택** — RM2i +0.172, RM2 +0.076 (단 §4.9-4: 보편 적용은 DEAD) |
| `rm_reactivity_mismatch` (RM3, `:314-340`) | 인접 슬롯 반응도지수 RI 절대차의 가중합; RI = 원산지 enrichment × `BURN_FACTOR`(연령별 감쇠) | **report-only**(null, RM1과 −0.885 공선) |
| `rm_fresh_periphery` (RM4, `:346-367`) | 외곽 링(`PERIPHERY_MASK`)의 fresh 가중 카운트 | **report-only** — 채택 시 node_peak 악화 실측 |
| `rm_peripheral_power_share` (RM5, `:373-414`) | BOC 맵의 외곽/전체 출력비 | **report-only, 패턴 함수 아님**(같은 맵에서 유도 → 순환논리) |
| `rm_checkerboard_degree` (RM6, `:420-442`) | fresh face-이웃이 하나도 없는 fresh 비율 | **refuted**(잡음 수준) |

공용 헬퍼: `fresh_mask`(`:172-174`), `reactivity_index`(`:209-259`, **원산지까지 shuffle-chain 추적**), `enrichment_by_batch`(`:190-206`). `rule_penalty(patterns, weights)`(`:457-500`)는 **`VALIDATED_PENALTY_METRICS`(rm1/rm1i/rm2/rm2i) 만 허용**하고 그 외 키는 `ValueError`를 던진다(`:480-488`) — **RM3~6의 오용을 코드가 막는다.**

### 5.3 규칙 기반 구성기 — `rule_construct.py`

**모델 없음, 스토어 엘리트 복사 없음, 루프 내 MASTER 없음.**

**게이트 G1–G8** (`rule_construct.py:15-40` 독스트링):

| 게이트 | 내용 | 위치 |
|---|---|---|
| G1 | 중심 슬롯 = E2 (저-kinf0 `COLD_BATCH`) | `:111`, `:247` |
| G2 | 첫 축 orbit unit(`C1_UNIT = 52`)에 fresh 금지 | `:119`, `:424-427` |
| G3 | 구조 불변량은 `GeneralOrbitGenome.validate()`에 위임 | §5.1 |
| G4 | 순-내향 연소연료 이동 `inward ≥ 0.75` (`GATE_INWARD_MIN`) | `:158`, `:569` |
| G5 | `fr_hat = 1.047·max(p_nom·FF(BU)) ≤ 1.65` 크기 거부권 (`GATE_FR_HAT_MAX`) | `:159`, `:311`, `:569` |
| G6 | (flat 프로파일만) 명목 최고출력 슬롯(`HOTTEST_SLOT = 46`)에 fresh 배치 | `:314`, `:370-372` |
| G7 | 링별 fresh unit 고정 census `(1, 4, 7, 18)` (`RING_CENSUS`) | `:116`, `:384-392` |
| G8 | 고-k(E1) unit 수: minfr **16**, flat **17** (`N_HOT_UNITS`) | `:114`, `:395-399` |

**소프트 점수(LOWER = better)** — `score_minfr`(`:334-350`) / `score_flat`(`:353-372`):

| 항 (대응 규칙) | minfr 계수 | flat 계수 |
|---|---|---|
| `rm1i` (R5) | +0.0068 | +0.0028 |
| `ff_radgrad` (R8) | +0.067 | +0.03 |
| `hot_cold_dr` (R11) | −0.016 | −0.005 |
| `hot_burned_nom` (R17) | −0.24 (clip 1.20–1.36) | — |
| `k_radgrad` (R1) | 목표 0.63 quadratic pull ×0.03 | −0.60·min(cap) |
| `k_outer` (R2) | — | −3.0·(min(val, 1.1542) − 1.10) |
| `hb_fresh` (R14) | 목표 0.65 quadratic ×0.05 | −0.35·min(val, 0.8333) |
| `travel` (R7) | 목표 1.80 quadratic ×0.02 | 목표 1.85 quadratic ×0.02 |
| `inward` (G4) | 목표 1.35 quadratic ×0.02 | 목표 1.40 quadratic ×0.02 |
| G6 위반 | — | +1.0 |

공통 `_env_pen`(`:322-331`)은 채굴 데이터의 **p98 범위**를 벗어나지 않게 하는 envelope 페널티다 (`k_radgrad > 0.72`, `travel > 2.30`, `inward < 0.75`) — **§4.11(2) CBC 공동예산 규율의 코드 표현**이다.

**이동 연산자 3종**(`_moves`, `:418-465`; 전부 **폐쇄 이동** = census 보존): ① 40% 링 내부 fresh/burned 역할 교환, ② 25% E1/E2 batch 교환, ③ 35% wiring source 교환. `construct_one`(`:468-485`)은 first-improvement greedy descent(최대 10 pass × 320 attempt).

**두 프로파일의 차이가 곧 §4.11(1) 상충 구조의 코드 표현이다** — minfr는 F_r 문법(R6/R8/R9/R10/R17)을, flat은 F1 축 최대화(R1/R2)와 피크-온-fresh 허용(R14 엘리트 부호, G6)을 집행한다.

### 5.4 규칙 산성 테스트 — `rule_acid_run.py`와 그 결과

**설계**: 프로파일당 200개 생성(시드 minfr 1000–1199 / flat 5000–5199), 400개 전부 상이, 스토어 71,517행 대비 신규성 확인, 게이트 탈락 0. 후보 8기 스테이징 (프로파일별 예측-스크린 통과 상위 2 + **최대-Hamming 중위 정직성 프로브 2**, sha256 인용).

**사전등록 성공 기준**(`rule_acid_run.py:16-30`, 체인 실행 **전** 확정):

- `SUCCESS(minfr)` = 4개 minfr 후보의 실현 F_r 최소 ≤ **1.479** (기록 1.4636 + 0.015), 동시에 `cbc_max ≤ 1600` · `f_q ≤ 2.41` · `620 ≤ cyclen ≤ 645`.
- `SUCCESS(flat)` = 4개 flat 후보의 실현 node_peak 최소 ≤ **1.205** (기록 1.1899 + 0.015),
  + `f_r ≤ 1.55` 및 동일 게이트.
- **유효성 게이트 V1–V5**: 8기 전수 수렴 / fallback 0 + 지정 native restart / 상호 상이 / 스토어 신규 / sha256 인용. **게이트 실패 시 검증 전체 VOID**(`:252-255`).
- 보조(보고만): 8기에 대한 Spearman(pred, real).

**심판 예측(사전등록 시점)**: minfr 최선 pred F_r 1.5262(기록 대비 +0.0626), flat 최선 pred node_peak 1.2914(기록 대비 +0.1015). **판독 앵커**: 같은 심판이 스토어의 기록 자체를 **+0.043 / +0.036 과대**예측한다(엘리트 꼬리 압축; in-band 벌크는 bias +0.010 / MAE 0.019 / Spearman +0.80). 심판-일관 좌표로는 최선 구성이 기록의 자기예측 대비 F_r +0.020 / node_peak +0.066 — **시험은 사전에 결정되어 있지 않았다.**

**결과 (2026-08-11 실행, MASTER 8체인, 전원 수렴, 전원 native restart): FAILURE**

| 프로파일 | 규칙제 최고(실측) | 사전등록 기준 | 탐색 기록 |
|---|---|---|---|
| minfr | F_r **1.5771** | ≤ 1.479 | 1.4636 |
| flat | node_peak **1.3245** | ≤ 1.205 | 1.1899 |

**정직한 해석** — 사전등록 FAILURE 갈래 그대로:

1. **규칙이 잡은 것 — 가능영역의 문법.** 8기 전원 수렴, cyclen 633.5~637.4(전원 창 내), CBC 1354~1415(전원 1600 이하), F_q 전원 2.41 이하. D-계열(반응도 예산)과 게이트 규칙은 **손으로 지어도 작동한다**. **무작위 합법 패턴의 F_r이 2.5~4.6**임을 감안하면 규칙만으로 **1.52~1.65**에 도달한 것은 **공간의 대부분을 건넌 것**이다.
2. **규칙이 못 잡은 것 — 마지막 ~0.1.** 탐색 최적점(1.4636 / 1.1899)과의 잔차는 규칙의 선형 결합이 아니라 **미세 배열 상호작용**에 산다. 챔피언 심판도 규칙제 노심의 F_r을 **+0.065 과소예측**했다(off-manifold 신호) — **규칙제 노심은 학습 분포 밖**이다.
3. **따라서 규칙의 정당한 소비처**는 생성기 프라이어 · 소프트 페널티 · 실현가능성 게이트이지 **단독 최적화기가 아니다**. "규칙 그 자체의 학습"은 **필요조건의 명시화**까지 도달했고, 충분조건의 나머지 절반은 **암묵적 모델 + 탐색**에 남아 있다 — 이것이 이 실측의 결론이다.

### 5.5 캠페인의 규칙 소비법 — 명시적 절반과 암묵적 절반

- **분업**: 규칙집은 학습된 지식의 **명시적 절반**(사람이 읽고, 감사하고, 인허가 문서에 인용 가능). 챔피언 넷은 **암묵적 절반**(맵 미세구조, 0.01-수준 캐리어 경주, E3_E4류 비-반경 기전). 규칙 6인자가 설명하는 축 위에서 심판의 in-band Spearman은 +0.80이고, **규칙-구성 노심의 예측 잔차가 바로 암묵적 절반의 크기**다. **어느 쪽도 다른 쪽을 대체하지 않는다.**
- **검증된 소비 형태는 소프트 페널티다**: 하드 컷이 아니라 **~0.02 오더의 소프트 페널티**를 생성기 점수에 더한다(RM1i의 쌍당 효과 +0.0026..+0.0068과 정합하는 스케일). **하드 게이트로 승격하는 것은 G군(G1–G4)처럼 위반 시 feasibility 자체가 붕괴하는 규칙뿐.**
  1. **생성기 사전분포**: G1–G4 + ring census + `n_hot`(R4)은 **표본공간 정의**로 — 위반 후보를 아예 만들지 않는다.
  2. **소프트 점수**: F1–F5 레버는 검증된 효과크기를 가중으로, 단 §4.11(3) 공선성 지도에 따라 **인자당 1회만** 배분하고 §4.11(2) CBC 공동예산으로 cap.
  3. **외부 루프 veto**: R16(`fr_hat` 크기), R17(스타일 동률)은 **절대 내부 순위기로 격상하지 않는다** — 홀드아웃 반전이 그 대가임이 두 번 측정되었다.
  4. **심판은 심판으로**: 챔피언 모델은 후보 스코어링(57 s / 400, CPU)과 스크린에 쓰되 꼬리 압축(+0.043 / +0.036)을 알고 읽는다. **규칙이 제안하고, 모델이 심사하고, MASTER가 판결한다.**
- **레짐 스위치의 소유권**: R14/R19처럼 레짐 의존 규칙의 부호 선택은 규칙집이 아니라 **캠페인 상태(밴드 내/외)** 가 결정한다 — **생성기는 자기 위치를 알아야 한다.**
- **정책망의 소비**(`lpopt/policy/scorer.py`): `MoveScorer.score(parent, children, ctx)`가 5-멤버 CNN 앙상블로 2개 헤드 확률(P(F_r 개선), P(평탄도 개선))을 반환하되 **랭커이지 확률이 아니다**(ECE 0.111–0.200, parent-blocked AUC 0.771) — **임계값 판정 금지, 순위만 사용**. `construct.py`의 elite-mutation 단계에서 argmax가 아닌 **softmax 샘플링**(temperature = `policy_prior_temperature`)으로 편집 후보를 뽑는다. 특징벡터는 코퍼스 채굴 함수를 **그대로 재사용**해 학습–서빙 drift를 원천 차단한다. 기본값은 off이며 A/B로만 배선한다.
- **FF 레버의 측정 도구**: `fr_arms.py`는 **고정 패턴**에 fresh batch identity만 바꿔가며 F_r 변화를 순수 FF 기여로 귀속시키는 **측정 실험**(스코어링이 아님). `fr_transfer.py`는 이를 집단으로 확장해 feasibility 스크린 (`f_r ≤ 1.55`, `cbc_max ≤ 1550`, `f_q ≤ 2.41`, `620 ≤ cyclen ≤ 645`) 후 `flat`(node_peak 오름차순) / `minfr`(f_r 오름차순) / `mixed`(기본) 랭킹으로 K개를 골라 타깃 pair로 **연료만 치환**한 paired delta를 잰다 — **신규 셀의 elite-pool 부트스트랩 표준 절차**다.

### 5.6 물리 프라이어와 핀 물리 모델

- **`lpopt/model/physics_prior.py`** — kinf 기반 cyclen 프라이어 + **잔차 학습**. 69슬롯을 fresh 원산지까지 체인 추적하여 원산지 타입의 k-inf(BU) 곡선으로 2구간 선형 반응도곡선을 만든다:

  ```
  rho_j(b) = rho_j* + s_j·(b − bu_j*)         (b ≥ bu_j*, burnout decay)
  rho_j(b) = rho_j* − S_j·(1 − b/bu_j*)       (b <  bu_j*, holddown release)
  rho_bar  = Σ w_j·rho_j(b_j) / Σ w_j ,  b_j = bu_j0 + NOMINAL_CYCLE_BURNUP/2
  B_cycle  = (rho_bar − RHO_LEAK_PCM) / D ,   RHO_LEAK_PCM = 3500.0
  cyclen_prior = alpha·B_cycle + beta         (train fold 최소자승 1회 적합)
  ```

  여기서 `rho_j* = (kinf_peak−1)/kinf_peak × 1e5`, `s_j = depletion_slope_pcm_per_gwd`, `S_j = reactivity_swing_pcm` — **§2.2의 곡선 형상 채널이 그대로 물리 프라이어의 입력**이다. 적합 실패 시 `alpha = 0` 폴백(train 평균 상수)이라 **잔차 왕복이 항상 정확**하다. 서빙은 `predict() = prior + residual`(라벨 접근 없이 계산 가능). **실측 분업**: 물리 프라이어 단독의 셀내 ρ = **0.064** [−0.005, 0.172](통계적으로 0) — 즉 **셀 내부 변별력은 100% 신경망 잔차 헤드**가 만들고, 프라이어는 **셀 간 스케일**을 담당한다(MAE 21.75 → 5.06, −76.7%).
- **`lpopt/model/pinbu_physics.py`** — 핀 연소도 물리 재구성: `pin_bu = a·[ ratio_type(B_asm) · B_asm ] + b`, `ratio(BU) = clip(r_inf + paramA/BU, ratio_asym, ff_pin_max)`. `B_asm`은 cyclen 예측에서 에너지 밸런스로 역산하고(`k_peak(feed)` = feed별 중앙값), affine `(a, b)`는 train fold의 Dataset-P 행에서만 Huber 적합 (2-D 격자 ↔ 3-D `MAS_PPI` 정의 간극 보정). **현 상태**: 챔피언 어디에도 `pinbu_physics.json`이 없어 **raw 헤드가 그대로 서빙 중**이다 — 처방은 등록되어 있으나 미적용(§4.10-I의 −5.93 과소예측 배경).
- **`lpopt/search/sdm_mtc.py`** — post-verification 단계(탐색 루프 **밖**). 후보 자신의 최종 사이클 deck + 수렴 restart에 대해 MASTER branch deck을 실행한다. MTC는 `%EXE_RHO` 브랜치로 ±ΔT_mod(기본 5.0 °C) 섭동 후 keff 2점 중심차분이 1차 추정이고 텍스트 파싱은 폴백/교차확인이다(크기 기반 휴리스틱 재스케일은 감사 후 제거, 단위 변환만). SDM은 `%EXE_ROD`로 ARO → ARI(scram bank 삽입) → 봉별 stuck 케이스를 실행하고 `available = W_ARI − worst_stuck`, `margin = available − required`. `flat_power` 목적함수가 F_r을 목적에서 뺀 결과 제어봉가/누설이 감시되지 않으므로 **납품 후보에는 이 단계를 강제**한다. `mtc_gated`/`sdm_gated`는 사용자가 `[constraints]`에서 명시적으로 값을 세팅하지 않으면 **report-only**이며, **"측정 부재는 결코 pass가 아니다"** 가 코드에 명문화되어 있다.

---

## 6. 다주기·전이주기 관점과 향후 규칙 확장

### 6.1 실제 다주기 셔플 방법론 — KNGR cy1–cy8

`data/reference/kngr_18m_cy01_08/methodology_notes.md` (출처: KNGR 18개월 다주기 해석 계산서, 3983 MWth). **이것은 실제 PSAR 계산서의 방법론**이며, `lpopt`의 평형주기 문제 설정을 실무에 정박시킨다.

- **코드 체계**: `ROCS`(3-D 노달 코어 시뮬레이터) + `cord`(격자/단면적 프로세서)
  + 후처리(`rocsedit` / `nconex` / `centaur`).
- **탐색 절차**: ① 상대출력밀도 ~1.30 미만 패턴 탐색 → ② 연소계산 → ③ 가능하면 **Fxy < 1.55** 탐색.
- **설계기준**: 목표 **Fxy 1.55**, 주기길이 **468 EFPD** ( = 365 × 1.5 × 0.9(가동률) × 0.95(부하율) ), **최대 양(most positive) MTC(HFP) < 0**.
- **배치 구조**: **cy1 = 241체(초기노심 전량)** → **cy2 feed 80체** → **cy3–cy8 feed 각 92체**. cy2가 과도기이고 cy3부터 근평형이 안정된다.
- **달성 EFPD**: cy1 454 / cy2 366 / cy3 463 / cy4 473 / cy5 458 / cy6 463 / cy7 466 / cy8 464 — **근평형이나 목표 468 EFPD에 소폭 미달**. 최종설계 단계의 **feed 농축도 선정으로 보정**하기로 하고 **그대로 접수**되었다.
- **최대 Fxy(raw, 보정 전)**: cy1 1.4950 / **cy2 1.5697(목표 1.55 초과)** / cy3 1.5569 / cy4 1.5398 / cy5 1.5516 / cy6 1.5488 / cy7 1.5422 / cy8 1.5372. **cy2 초과분은 "집합체 회전으로 최종설계 단계에서 우회 가능"** 하다고 판단하여 수정 없이 접수.
- **MTC**: BOC −1.16 ~ −0.23 ×10⁻⁴/°F, EOC −3.25 ~ −2.13 ×10⁻⁴/°F — 목표 충족.
- **모델링 관행**: cy1 모델을 후속 사이클에 **재사용**하고 배치·조성 지정만 갱신. 알베도 경계조건은 KNGR 고유 반복계산 없이 **선행 노형의 방사 알베도 세트를 그대로 재사용** (노심 크기·기하 동일이 근거). 집합체 가중계수·형상 소둔함수, 축방향 경계조건도 동일하게 차용.

**본 프로젝트가 여기서 읽어야 할 것**

1. **평형은 cy3부터다.** 실제 다주기 계산에서도 cy2가 과도기이고 cy3–cy8이 근평형이다 — `lpopt`의 평형주기 고정점 문제 설정은 이 구간을 직접 겨냥한다.
2. **Fxy 1.55는 실무 설계기준으로 실재**한다. 본 프로젝트의 `f_r_limit = 1.55`와 동일 계층이다. 그리고 실제 설계도 **cy2에서 이를 초과했고, 집합체 회전이라는 우리가 채택하지 않은 자유도로 우회**했다(§3.3 미채택 항목의 실무적 의미).
3. **주기길이 미달을 feed 농축도로 보정**한다는 실무 관행은, 본 프로젝트의 **"feed 자유도가 필수"**(§4.10-F)와 정확히 같은 이야기다.
4. **알베도 세트의 노형 간 재사용**은 §6.4의 "노심-크기 불변 인코더 + 반사체/알베도 노드" 방향과 같은 구조적 가정을 공유한다. (추정)
5. **정직한 고지**: 이 방법론 노트는 **OUT-IN vs 저누설 전략의 명시적 선언, 노심 대칭 규약, 집합체 회전 세부규칙을 직접 서술하지 않는다** — cy2 피킹을 "적절한 집합체 회전으로 우회 가능"하다는 언급만 있다. **따라서 이 문서로부터 전략 계열을 추론해서는 안 된다.**

### 6.2 전이주기 엔지니어 — 최종 로드맵 (염두만, 착수 금지)

사용자 로드맵(2026-08-17): **1주기부터 시작하여 전이(transient) 주기 계산까지 최적화하는 엔지니어 AI로 발전** — 초기노심(cy1) → 전이주기들 → 평형 도달 전 구간의 다주기 최적화가 최종 형태다. 참조 자산은 위 KNGR 계산서(cy1~cy8 셔플 맵 + 방법론)와 `regen_chain`(cy1 → 평형 결정론 재계산 체계; 수집 14,180 케이스 + **재생성 전체체인 716개** 보유). **착수·수정 금지 — 현행 우선순위 유지.**

문헌 관점(§3.5)에서 이 확장은 **B-07 / B-13**(다음 2–3주기 재고 관점, 단일주기 최적화가 다주기에서 불리할 수 있음)을 다시 문제로 불러들인다. 현재 평형주기 설정은 그 문제를 **구조적으로 소거**하고 있으므로, 전이주기로 확장하는 순간 **§4의 규칙 대부분이 "평형 조건 하에서"라는 유효범위 단서를 달게 된다.** (추정)

### 6.3 축방향 확장 — 블랭킷/컷백 (계획만, 착수 금지)

- **범위 한정**: 집합체 **축방향 설계**(axial blanket, cutback)까지만이며, **LP의 축방향 자유도는 2026-07-25 금지 결정을 유지**한다(§1.2 AO 참조).
- **설계 제약(사용자 2026-08-17)**: 블랭킷 길이 **15 cm 고정**(APR1400 한정), **블랭킷 농축도는 자유 변수**. 축 메시 노드 15.24 cm(6")와의 0.24 cm 차이는 착수 시 **1노드 근사** 권고.
- 계획서: `data/reports/axial_upgrade_plan_20260817.md`. **착수는 승인 대기.**
- **필요 라벨은 이미 파싱되어 버려지고 있다**: `parse_mas_sum`이 EDIT5의 **30 연소단계 × 69집합체 × 4량**(batch/power/burnup/kinf)을 이미 전량 메모리에 파싱하지만 저장 단계에서 2단계·4평면(276 float)만 남긴다. EDIT5 전 30단계 수확은 **추가 파싱비 0, +12.1 KiB/레코드**(28k 기준 0.325 GiB); EDIT6(축방향 30단계 × 25평면)은 **파서 ~10줄, 파싱 0.06 ms/레코드, +1.5 KiB/레코드**(0.039 GiB). 반면 핀 3-D(`MAS_PPI`)는 레코드당 0.84 MiB(28k = **23 GiB**), 파싱 0.2 s/레코드이고 **집합체 내부 대칭이 실측으로 성립하지 않아**(좌우/상하/전치 전부 False, tol 1e-4) 옥탄트 접기가 불가능하다 → 표적 서브셋만. → **"오늘 안 켜면 오늘 생산분은 영구 손실"**(소급 불가).
- 이 라벨들은 **정밀도 로드맵의 2순위 레버**이기도 하다. 라벨 잡음 상한은 이미 전 타깃 ρ ≥ 0.997이고(따라서 MASTER tolerance 강화는 기각), 셀당 밀도만으로 ρ 0.99에 도달하려면 셀당 5.5만~92만 표본(함대 ~16년)이 필요하다 — **데이터는 0.90~0.95의 레버이지 0.99의 레버가 아니다.**

### 6.4 노심-크기 불변 확장 (v3 방향, 설계서만)

사용자 지시(2026-08-17): **17×17에 국한되지 않고 집합체의 지오메트리보다 output(k-curve, F_r, F_q, MTC, FTC, 핵종 inventory 등)을 받아서 다양한 크기의 노심 (APR1400, OPR1000, i-SMR 등)에 장전해보고 중성자 누설 및 다양한 결과 인자들을 해석하여 빠르게 최적 노심을 도출하는 모델.**

함의 세 가지:
1. **집합체 표현 = 격자 핀맵이 아닌 물리 출력 벡터** — §2.1/§2.4에서 이미 실증된 원칙이며, 이 확장이 그 원칙을 **요구사항으로 승격**시킨다.
2. **노심 인코더를 크기-불변으로** 전환(마스크드 FCN 또는 GNN + **반사체/알베도 노드로 누설을 학습**).
3. 두 번째 노형(OPR1000이 최저비용 후보)·i-SMR 라벨 생산 경로가 필요하다. 그리고 **i-SMR로 가면 AO 축이 되살아난다**(소형 노심 + 제어봉 제어) — §1.2의 "AO 미채택" 결정은 APR1400 한정 결정이다.

**현재 상태**: 설계서(`coreagnostic_v3_design_20260817.md`)만 보유하고 **구현은 착수 직후 중지**되었다. 최우선은 그물망(설계지도) 완성이다.

### 6.5 향후 규칙 확장 후보 (미착수 목록)

| 후보 | 근거 | 상태 |
|---|---|---|
| SDM 뱅크 구성 수정(정지뱅크 A/B 추가) | A+B가 EOC worth의 74%(12.32/16.70 %Δρ) — 현재 구조적 FAIL(§1.2) | **결함으로 등록, 미수정** |
| `min_fr_max_cycle`에 핀 축 상시 게이트 | 3개 캠페인에서 "F_r로 열리고 핀으로 막힘" 반복(§4.10-I) | 2026-08-17 `minfr_pin_bu_limit=78.0`으로 부분 반영 |
| 핀 헤드를 저feed × e5.2–5.5 구간에서 재교정 | 그 구간 학습행 0, bias −5.93까지 관측 | 실측 라벨 22개 확보(N1_N2/f113) |
| `pinbu_physics` 분리 예측(B_asm × ratio) 배선 | raw 헤드가 그대로 서빙 중 | 처방 등록, 미적용 |
| LRM 백본을 설계지도 x축으로 승격(하이브리드) | 평활·단조 구조 보장, 챔피언 버전 불변, 추가계산 0 | 처방 등록, 미착수 |
| feed별 자유계수 LRM 재적합 | 해석적 `2/(n+1)`가 feed축에서 ±1.8% 오차(f109 ridge) | 미착수 |
| `E5_E6` / `J7_J8` pair 편입 | 백본이 가장 못 맞추는 두 pair이자 `fuel_types`에 타입 부재 | yaml 병합이 전제조건 |
| 다종 4종 라벨 생산 | 4종 학습행 0 → 전 예측이 외삽 | 미착수 |
| 축방향 블랭킷/컷백 | §6.3 | 승인 대기 |
| 전이주기(cy1→평형) | §6.2 | 착수 금지 |
| 용기 fluence · CIPS/CRUD · DNBR · 제어봉 사고해석 | 문헌 L/M/C 계열(§3.6) | 미채택 |

---

## 부록 A. 규칙 일람 (한 장 요약)

| ID | 이름 | 인자 | 판정 | 핵심 수치 |
|---|---|---|---|---|
| G1 | center-cold-fresh | 게이트 | CONFIRMED | P(feasible) 0.646 vs 0.007, p=8.3e-117 |
| G2 | no-center-cross-fresh | 게이트 | CONFIRMED | rho +0.262 node_peak / +0.284 f_r |
| G3 | 구조 불변량 | 게이트 | CONFIRMED | twinbreak/selffeed 0 / 71,517행 |
| G4 | inward-migration | 게이트 | CONFIRMED | 수렴 AUC 0.681; map_cov −0.502 |
| R1 | radial-k-gradient | F1 | CONFIRMED | node_peak −0.707 (E3_E4 null) |
| R2 | periphery-k-mean | F1 | DOMAIN-LIMITED (E/J) | node_peak −0.758; **K셀 반전** |
| R3 | cbc-follows-k-budget | F1 | CONFIRMED | +0.001 kbud = +13.8 ppm / +0.72 EFPD |
| R4 | feed-split-serves-gates | F1 | CONFIRMED | +1% share = CBC +3.13 ppm, 평탄도 0 |
| R5 | rm1i-alive | F1 | CONFIRMED | node_peak +0.227 / 엘리트 +0.735 |
| R6 | fresh-off-mid-crest | F2 | CONFIRMED | f_r +0.508 / node_peak −0.680 (상충) |
| R7 | short-shuffle-travel | F2 | CONFIRMED | +1 pitch = CBC +84 ppm |
| R8 | ff-gradient-down | F3 | CONFIRMED (전 셀) | f_r 양성 9/9셀, +0.067/unit |
| R9 | fresh-out-of-inner-ring | F4 | CONFIRMED | f_r +0.322..+0.605 |
| R10 | fresh-into-outer-band | F4 | CONFIRMED | f_r −0.315..−0.623 |
| R11 | hot-fresh-outboard | F5 | DOMAIN-LIMITED (E/J) | f_r −0.016/pitch; **K셀 반전** |
| R12 | hot-ring-twice-burned | F5 | CONFIRMED (feed<121) | node_peak −0.388, cbc −0.291 |
| R13 | park-twice-burned-inboard | F5 | CONFIRMED (feed<121) | cbc +0.412 (음성 셀 0%) |
| R14 | hot-ring-fresh-two-regime | F5 | CONFIRMED (2-레짐) | 스토어 +0.230 / 엘리트 −0.721 |
| R15 | burned-peak-carrier | F6 | DOMAIN-LIMITED (심층 셀) | AUC 0.653; 89% burned @14.3 |
| R16 | fr-fusion-surrogate | F6 | DOMAIN-LIMITED (크기 veto) | A=1.0472, rms 0.0496 |
| R17 | burned-hot-product-style | F6 | DOMAIN-LIMITED (타이브레이커) | rho −0.209 |
| R18 | chain-burnup-validity | 지지 | CONFIRMED | Spearman 0.618–0.694 |
| R19 | elite-diagonal-checkerboard | F1 정밀화 | DOMAIN-LIMITED (엘리트) | 스토어 null / 엘리트 −0.785 |

**v1 이후 추가 확정 규칙**

| ID | 내용 | 판정 | 핵심 수치 |
|---|---|---|---|
| A | 누설-중재: fresh 외곽 = 평탄화 이득 + cyclen 지불 | SETTLED(개입) | −8.37 / −9.41 EFPD/unit |
| B | cyclen 밴드가 저밴드 F_r을 묶는다 | 확정 | 무게이트 시 전 6밴드 서브-1.55 |
| C | `batch_swap`은 618 분지 특이 | 반증(일반화) | 0.052 vs 0.005, p=0.0028 |
| E | `J5_J6` > `K3_K4` | 중간 | 27/32 우세 |
| F | feed 자유도 필수 | 확정 | f117 물리한계 666 EFPD |
| G | F_r 1.45는 붕소 문제 | 설계 결정 | cbc_limit 1550→1600 |
| H | 격자 FF는 2차항, 레이아웃 > 농축도 25배 | 확정 | ΔFF 0.049 ⇒ ΔF_r 0.065; LP 레버가 8배 |
| I | 저feed·고농축에서 핀이 결박축 | 확정 | 25기 중 10 PASS / 15 FAIL |
| J | 모델 무편향, 격차는 커버리지 | 확정 | 프런티어 편향 \|0.028\|(s1e \|0.024\|) |
| K | 다종: R1 mono-spec만 이득 | 확정(부호 만장일치) | −0.0378 (10/10) vs +0.0749 (0/19) |
| L | 정책은 "다음 한 수"만 배운다 | 확정 | 최장 개선 체인 7 (lineage 79) |

## 부록 B. 용어

| 용어 | 의미 |
|---|---|
| LP | Loading Pattern, 장전모형 |
| cell(셀) | `campaign \| case_pair \| feed` — 규칙 채굴의 층화 단위 |
| within-cell | 셀 내부 상관. 셀 간 풀링은 feed / e_core와 교락되므로 금지 |
| in-band / 가능영역 | `f_r ≤ 1.55`, `cbc ≤ 1600`, `f_q ≤ 2.41`, `620 ≤ cyclen ≤ 645` |
| feed | 주기당 신연료 장전 집합체 수(1+4N 격자) |
| `e_core` | 노심 평균 농축도 [wt%] |
| `kbud` | 패턴 반응도 예산 = 다중도 가중 슬롯 k 평균 |
| `hot_share` / `n_hot` | fresh 중 고-kinf0 타입의 비율 / 유닛 수 |
| `travel` | 연소 집합체의 가중 평균 반경 이동거리 [pitch] |
| `inward` | 연소 집합체의 순 내향 이동량 |
| `ff_pin_max` (FF) | 집합체 핀 form function 최댓값 |
| ADF | Assembly Discontinuity Factor |
| LRM | Linear Reactivity Model |
| acid test | 규칙만으로 노심을 지어 MASTER로 판정하는 사전등록 실증 |
| verdict 등급 | CONFIRMED / DOMAIN-LIMITED / DEAD |
| 신뢰도 | 높음 = 사전등록 개입 실험 또는 MASTER 실측 / 중간 = 단일 실험·관측 상관 / (추정) = 저자 추론 |

---

*본 문서의 수치는 `data/store/records.parquet` 스냅샷(규칙집 v1 시점 71,517행; 캐노니컬 73,903행)의 함수다. 규칙의 수치는 스냅샷과 함께 갱신하되, **사전등록된 성공 기준(§5.4)은 소급 수정하지 않는다.***
