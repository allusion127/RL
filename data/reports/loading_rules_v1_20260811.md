# 장전 규칙집 v1 — 학습된 LP 최적화 규칙 (loading_rules_v1_20260811)

- 작성일: 2026-08-11
- 대상: ga80 라이브러리 평형노심 4분기 대칭 LP (69-slot quarter, 241 집합체), 주 검증 셀 E1_E2|f121
- 상태: 채굴 → 적대적 재검증 완료(18 CONFIRMED / 6 DOMAIN-LIMITED / 0 DEAD-on-reproduction), 실증(acid test) 8기 스테이징 완료·미실행
- 데이터 기반: `5_RL/data/store/records.parquet` 71,517행(+), `maps.npz` EDIT5 nodal planes, `fuel_types.parquet`; 챔피언 모델 `data/models/split_S1b` (v6b)

---

## 1. 서문 — "학습된 규칙"의 의미

이 문서의 규칙은 사람이 고안한 휴리스틱이 아니라, **71,517개 MASTER 검증 노심**과 금주의 통제 실험(13-pt dose, C5/C6 동질장전 파괴실험, 연료 스왑 계열)에서 **채굴(mined)** 된 통계적 법칙이다. 각 규칙은 다음 세 관문을 통과했다.

1. **채굴**: 순수 (pattern, fuel_table) → 스칼라 함수로 정의된 메트릭을 셀 내부(within-cell) 상관으로만 평가. 셀 = campaign|case_pair|feed — 셀 간 풀링은 feed/e_core와 교락되므로 금지. 셀 단위 split-half holdout 적용.
2. **적대적 재검증**: 채굴자와 독립된 파서·체인 추적기·맵 로더·통계 코드로 전 수치를 재계산. 재현 실패 규칙 0건, 단 헤드라인 부속 주장 5건은 강등·기각(해당 규칙의 유효범위에 반영).
3. **구성 가능성(acid test)**: 규칙만으로 — 탐색 없이, 모델 없이, 스토어 엘리트 복사 없이 — 노심을 지어서(`rule_construct.py`), 100-call 캠페인이 찾은 기록에 접근하는지 MASTER로 확인한다(§4). **규칙이 진짜라면 규칙만으로 지은 노심이 좋아야 한다** — 이것이 본 프로그램의 북극성이다.

모든 규칙은 금주 확립된 기전 사실(established mechanism facts) 6개와 정합적이어야 하며, 실제로 규칙군 전체가 그 사실들을 패턴 측 통계량으로 재서술한다. 특히:
- F_r = A·max(p·FF(BU)), A~1.03 — 그리고 in-band에서 BOC node_peak는 F_r과 **반상관**(r=-0.750, 13-pt dose).
- 고반응도 연료를 저출력 슬롯에 두는 역할 분리(E1-role 68슬롯 mean p~0.805 / E2-role 53슬롯 mean p~1.131)는 평탄화 기전 그 자체다(C5/C6 동질장전: node_peak 1.22→1.55).
- g_k_radgrad(반경 방향 반응도 기울기)는 node_peak 오차의 41%를 단독 설명하는 1차 변수다.
- cyclen 이탈 >~5 EFPD는 map_cov를 파괴한다(반응도 정합 게이트).
- 슬롯 BOC 연소도는 소스 체인을 통한 패턴의 순수 함수다(depth≤2).

**타당범위(feasible band)**: f_r≤1.55, cbc_max≤1600, f_q≤2.41, 620≤cyclen≤645. (주의: cbc 한도는 신규 1600이며, 스토어 내 비교 행들은 1550 하에서 생산되었다.)

**독립성 구조**: 중복도 분석(E1_E2|f121 feasible n=381, 19개 변형 메트릭) 결과 규칙 공간의 실질 자유도는 **~6개 독립 인자 + 2개 포화 게이트**다. 아래 §2는 이 인자 구조대로 묶는다. 20개 규칙을 20개 다이얼로 오해하면 안 된다 — 특히 in-band에서 kbud ≈ hot_share ≈ periphery-k (상관 0.94–0.99)로 **붕소 예산과 평탄화 다이얼이 한 레버**다.

---

## 2. THE RULES

각 규칙: **서술 / 적용법 / 근거 / 유효범위 / 위반 시 대가**. 판정은 재검증 verdict를 따른다.

### 2.0 공통 정의

- 슬롯 기하: `lpopt/vendor/masterrl/domain.py` SLOTS (69 quarter slots, multiplicity w_s ∈ {1,2,4}, Σ=241). radius = hypot(row,col) [pitch].
- 체인 추적: 각 shuffle card를 동일 패턴 내 fresh 기원까지 추적(평형 체인, depth≤2). age = 1(fresh)/2(once)/3(twice). BU_nom[s] = B_regime·Σ_k p_nom[direct_k(s)].
- k_slot: 기원 배치의 kinf 곡선(fuel_types kinf0/10/20/30 + kinf_eol50, knots (bu_k1,1.0))을 (age−1)·22 GWd/tU에서 보간.
- FF: 기원 배치 ff_pin_max(연소 시 clip(r_inf+paramA/BU, [ratio_asym, ff_pin_max])).
- 명목 맵 p_nom: 셀 평균 BOC/EOC EDIT5 평면 — 단, **fold-공유 프로토콜 필수**(자기 맵 누출 금지; leave-one-out 셀평균은 −own_map/(n−1) 인공 음성분을 주입해 기각됨).

---

### 2.1 G군 — 게이트 규칙 (탐색 전 강제; in-band에서 분산 0이므로 상관이 아니라 게이트로 존재)

**G1. [center-cold-fresh / low-k-in-center] 노심 중앙 슬롯에는 반드시 두 fresh 배치 중 저-kinf0(고 Gd) 배치를 둔다.** (A8 CONFIRMED + D3 CONFIRMED)
- 적용법: `a_center_is_cold(pattern, fuel_table)` = slot 0 점유 배치의 kinf0가 fresh 쌍 중 최소이면 1.0. E1_E2 테이블에서는 홀수 n_hot ⇔ centre=E1.
- 근거: P(feasible|cold center)=0.646 vs P(feasible|hot center)=0.007 (E1_E2|f121, n=585/414, Fisher p=8.3e-117); J5_J6|f121 0.605 vs 0.024 (p=1.2e-13); 재검증에서 신규 게이트 셀 3개 추가(J1_J2|f121 p=2e-15, J1_J2|f117 p=2.3e-44, K1_K2|f117 p=1.4e-7). 쌍별 within-cell 118셀: centre=high-k 시 Δmedian f_r = +0.044, 62.7% 양성, p=0.0073; kinf0 갭 큰 ga80 E/H 쌍은 +0.311.
- 유효범위: 전 셀 일반화 확인. 갭이 작은 라이브러리에서는 효과 축소(+0.036).
- 위반 시 대가: feasibility 승산 ~1/250 (64.6%→0.7%); F_r 중앙값 +0.5 오더 악화(2.063 vs 1.536). **Construct에서 다른 어떤 결정보다 먼저 고정할 것.**

**G2. [no-center-cross-fresh] 1번 축 궤도 유닛(quarter (0,1)/(1,0))에 fresh를 두지 않는다** — 중앙은 항상 fresh이므로 위반 시 5집합체 fresh 십자가 형성. (B2 CONFIRMED)
- 적용법: `b_c1_fresh` = ROW_OFFSETS[0]+1 슬롯 점유가 fresh면 1.0 (축-쌍둥이 동일, 스토어 불일치율 0.0000).
- 근거: within-cell rho +0.262 node_peak (188셀, n=12,367, 음성 10.1%), +0.284 f_r (257셀, n=22,719), +0.281 f_q; AUC 0.653/0.665; 최신 캠페인 전부 양성(fpcamp4 +0.23 … live minFr f_r +0.51, p=2.7e-7); holdout 셀에서 더 강함(+0.385).
- 유효범위: 스토어 전역. **feasible E1_E2에서는 분산 0(전원 준수) → in-band에서는 상관이 아니라 게이트.**
- 위반 시 대가: within-cell 평균 이동 node_peak +0.29, f_r +0.36 (binary 0→1); cyclen 무영향(−0.006).

**G3. [구조 불변량] 쌍둥이 대칭·비순환 depth≤2 체인·242−2F 연령 센서스는 탐색하지 말고 강제한다.** (D7 CONFIRMED)
- 적용법: `GeneralOrbitGenome.validate()` — twinbreak=0, selffeed=0, age3 초과분 = max(0, 242−2·feed) 고정.
- 근거: 71,517행 중 twinbreak>0 = 0행, selffeed>0 = 0행; age3_excess within-cell 분산 보유 셀 1/607. tolerance_margin은 어떤 family-D 메트릭에도 반응 없음(|wrho|≤0.045).
- 유효범위: 전 스토어 구조적.
- 위반 시 대가: 스토어 외 미지 영역 — 검증된 규칙이 하나도 적용되지 않는 공간으로 이탈. 연령 혼합비·tolerance_margin에 규칙 용량을 쓰지 말 것(비-레버).

**G4. [inward-migration-converges] 연소 연료의 순 이동은 안쪽으로(소스 반경 > 목적지 반경): 순-외향 셔플은 지배적 수렴 실패 모드다.** (D6 CONFIRMED)
- 적용법: `inward` = Σ_burned w_s·(radius(direct_source(s)) − radius(s)) / (241−feed) ≥ 0 (constructor는 ≥0.75 사용).
- 근거: converged 판별 weighted AUC 0.681, IQR [0.635,0.721], 202셀, n=27,839, p=6.4e-57. 동일 메트릭이 최강 평탄화 축: map_cov wrho −0.502(99% 음성), f_r −0.422(92% 음성). 재검증 정밀화: cbc 중앙값 −0.125(69% 음성) — 안쪽 이동이 붕소도 **약하게 돕는다**("비용 0"은 보수적 서술이었음).
- 유효범위: 전 스토어.
- 위반 시 대가: 수렴 승산 약 절반; 평탄도 대폭 악화. cyclen +1.8 EFPD/pitch만 유의.

---

### 2.2 F1 — 반응도-외곽 축 (master axis: 저k 안쪽 / 고k 바깥쪽)

in-band에서 k_radgrad, k_outer_mean, kbud, hot_share, inward, rm1i(−), rm2i, checker, hot_burned_nom(−)이 |rho| 0.74–0.99로 한 다발. **하나의 레버를 여러 이름으로 세지 말 것.**

**R1. [radial-k-gradient] BOC 집합체 k-inf가 반경과 함께 상승하도록 배열하라(저k 안쪽, 고k 바깥쪽); 기울기가 가파를수록 node_peak 1차 감소, in-band F_r도 감소.** (A1 CONFIRMED)
- 적용법: `a_k_radgrad` = 다중도 가중 Pearson corr(k_slot, radius) over 69 slots.
- 근거: node_peak rho −0.707 (E1_E2|f121 n=346, p=1.1e-53, top-decile AUC 0.90), −0.494 (J5_J6 n=103), −0.555 (J1_J2|f121 n=29); Stouffer p=7.9e-47. f_r −0.376/−0.531/−0.374 (E3_E4/J5_J6/J1_J2|f121). fresh-periphery(RM4/RM1i) 축 partial 후에도 −0.20..−0.46 전 5셀 유지; partial|cyclen −0.469. 확립 사실 3(g_k_radgrad 41%)의 a-priori 확인 — harvest 맵 불필요.
- 유효범위: E/J 셀 확인; **E3_E4는 node_peak에 대해 null**(그 셀의 in-band node_peak 산포는 비-반경 기전 — Construct 주의). K셀 미확인.
- 위반 시 대가: dnode_peak/dmetric = −0.52..−0.71 (metric sd 0.023–0.046 → 기울기 +1sd = node_peak −0.012..−0.032); f_r 기울기 −0.055..−0.132/unit. **CBC 비용 동반: within-cell cbc rho +0.577(§3.2).**

**R2. [periphery-k-mean] 최외곽 링 13슬롯(48집합체)을 밴드가 허용하는 최고 BOC k-inf로 채운다.** (A2 **DOMAIN-LIMITED — E/J 셀 한정**)
- 적용법: `a_k_outer_mean` = PERIPHERY_MASK 13슬롯의 다중도 가중 평균 k_boc.
- 근거: R1과 동일 축(within-cell rho +0.80..+0.91). node_peak −0.758 (E1_E2, p=8e-66); f_r −0.326..−0.539 4/5셀, top-decile AUC 0.61–0.78.
- 유효범위: **K3_K4에서 node_peak +0.388 (p=0.028) 유의 부호반전, K1_K2 f_r +0.319 — 재검증에서 발견된 유일한 유의 반전 축의 하나.** K-case_pair에는 적용 금지.
- 위반 시 대가(유효 도메인 내): df_r/dk_outer = −0.63..−1.90/unit (sd 0.004–0.011 → 1sd = F_r −0.003..−0.015); dnode_peak −6.7/unit (E1_E2). CBC 비용 최대(+0.691, §3.2).

**R3. [cbc-follows-k-budget] 패턴의 반응도 예산은 붕소를 먼저, 주기길이를 나중에 정한다: kbud +0.001당 CBC +13.8 ppm, cyclen은 +0.72 EFPD뿐 — 예산은 CBC 게이트 바로 아래에 놓고 cyclen을 슬랙 변수로 취급하라.** (D1 CONFIRMED)
- 적용법: `kbud` = (1/241)·Σ w_s·k_b(s)((age_s−1)·22).
- 근거: cbc_max wrho +0.330(중앙값 +0.536, IQR [+0.314,+0.696], 286셀, n=48,242, 음성 6%, p=1.7e-59, holdout +0.325/+0.335; 재검증 +0.311/13.3 ppm); cyclen wrho +0.120. 13.8 ppm/0.001은 붕소가 ~10 pcm/ppm과 수치 정합 — 확립 사실 4의 정량화. 소형 셀 holdout에서 오히려 강화(+0.525).
- 유효범위: 전 스토어, 전 셀 일반화(family-D 중 최강 이식성).
- 위반 시 대가: E1↔E2 전노심 1집합체 스왑(Δkbud~2.8e-4) = CBC ~+3.9 ppm. 예산 초과분은 cyclen으로 회수 불가(교환비 20:1 불리).

**R4. [feed-split-serves-the-gates] 평탄도는 fresh 배치 분할에 무감하므로, 붕소 게이트가 허용하는 최대 고-kinf0 share를 선택하라; E1_E2/f121의 feasible 최적은 121 중 64–68 high-k.** (D2 CONFIRMED)
- 적용법: `hot_share` = Σ(w_s over 고kinf0 fresh 슬롯)/feed. minfr 프로파일 64/121(16유닛), flat 프로파일 68/121(17유닛) — 두 셀 기록이 정확히 그 지점(F_r 기록 1.4636 @64, flat 기록 1.1899/map_cov 0.2147 @68–72).
- 근거: →cbc_max wrho +0.325(284셀, n=48,180); →cyclen +0.148; →f_r/f_q/node_peak/map_cov 전부 null(+0.026..+0.035). kbud와 한 축(상호 partial 소멸).
- 유효범위: 전 스토어(0.14–0.87 share 범위 관측).
- 위반 시 대가: +1% high-k share당 CBC +3.13 ppm(재검증 +3.08)·cyclen +0.198 EFPD; 평탄도 비용 0 — **분할을 평탄도에 쓰는 것은 낭비, 게이트에만 써라.**

**R5. [rm1i-alive] 내부(비주변) fresh-fresh 면접촉을 최소화하라 — RM1i는 죽지 않았고, 최신 엘리트 캠페인에서 오히려 증폭되었다.** (B3 CONFIRMED — 명시적 사망판정 요청에 대한 답: NOT DEAD)
- 적용법: `rule_metrics.rm_fresh_face_adjacency(pattern, inboard=True)` (다중도 가중, mirror-expanded 17x17).
- 근거: 스토어 within-cell +0.227 node_peak / +0.295 map_cov (286셀, n=39.6k — 문서화된 +0.235/+0.295 재현); fpcamp4_199 +0.735(p=1e-17), fpcamp5/6 +0.573/+0.588, live minFr f_r +0.450(p=1e-5); fpcamp_minfr만 +0.192(p=0.055)로 약화·비반전. AUC 0.662/0.641; cyclen/cbc 경유 아님(재검증).
- 유효범위: 전 스토어 + 전 엘리트 캠페인. in-band E1_E2에서는 F1 축과 공선(checker와 −0.96).
- 위반 시 대가: 가중 면접촉 쌍당 node_peak +0.0026(스토어) / +0.0068(엘리트 셀); 전형 셀 within-cell 범위 64쌍 → 레버 폭 ~0.17 node_peak.

---

### 2.3 F2 — 크레스트 점유·셔플 이동거리 축

**R6. [fresh-off-mid-crest] 저F_r을 원하면 fresh를 중반경 출력 크레스트(4.5<r≤6.5, 68집합체)에서 치워라 — 단, 최평탄 노심은 정반대로 한다(§3.1 상충 참조).** (A6 CONFIRMED)
- 적용법: `a_fresh_ring2_share` = 4.5<radius≤6.5 슬롯의 다중도 가중 fresh 비율.
- 근거: f_r +0.508 (E1_E2, p=2.4e-26), +0.599 (J5_J6), +0.280 (J1_J2|f121); Stouffer p=2.6e-32; holdout 신규 K1_K2 +0.605 (p=1.5e-4). 목적 상충 직접 측정: 동일 메트릭 vs node_peak (E1_E2) −0.680 (p=2.7e-48).
- 유효범위: E3_E4는 동부호 null(+0.108). **주의(재검증): node_peak 상충은 E1_E2/J5_J6 특이 — J1_J2에서는 크레스트 fresh가 둘 다 해친다(+0.406).**
- 위반 시 대가: 전노심 1집합체를 크레스트에 추가할 때마다 F_r +0.003..+0.004 (share unit당 +0.22..+0.30); E1_E2에서는 같은 이동이 node_peak −0.019.

**R7. [short-shuffle-travel] 연소 집합체는 전주기 연소 반경 근처에 재장전하라: 반경 이동거리는 예산(kbud)·inward와 독립적으로 CBC와 cyclen을 끌어올린다.** (D4 CONFIRMED)
- 적용법: `travel` = Σ_burned w_s·|radius(s) − radius(direct_source(s))| / (241−feed); 보조 `stay` = |Δr|≤1.0 pitch 가중 비율.
- 근거: →cbc_max wrho +0.377(288셀, n=48,873, p=4.8e-60, holdout +0.392/+0.364); partial|kbud +0.416, partial|inward +0.397, partial|cyclen +0.336 — 독립 축. →cyclen +0.249; map_cov +0.302, f_r +0.234. stay 미러(cbc −0.228, 91% 음성).
- 유효범위: 전 스토어.
- 위반 시 대가: 가중 평균 이동 +1.0 pitch당 CBC +84 ppm(재검증 +82.9)·cyclen +4.6 EFPD; 동일 링 재장전 10% 증가 ~ CBC −15 ppm. 긴 out-to-in/in-to-out 도약은 다른 곳에서 값을 치러야만 정당화된다.

---

### 2.4 F3 — FF 반경 기울기 (독립 인자; 가장 이식성 높은 F_r 규칙)

**R8. [ff-gradient-down] 기원 배치의 ff_pin_max가 반경과 함께 하강하도록 배열하라 — 저FF 연료를 출력 크레스트 외곽에, 고FF 연료를 안쪽에.** (A4 CONFIRMED)
- 적용법: `a_ff_radgrad` = 다중도 가중 corr(FF_origin, radius); **양수(FF 외향 상승)가 나쁨.**
- 근거: f_r 양성 9/9셀(holdout K3_K4 +0.449, p=0.0099 포함): +0.390 (E1_E2), +0.430 (E3_E4), +0.467 (J5_J6), +0.479 (J1_J2|f121); Stouffer p=9.0e-33. hot-fresh-outboard와 축 공유(−0.46..−0.80)이나 partial 신호 유지: E1_E2 +0.310, E3_E4 +0.210, J1_J2|f121 +0.287. 기전상 F_r = A·max(p·FF)의 직접 표현(사실 1).
- 유효범위: **전 셀 — K셀 포함 부호 일관, family-A 중 유일하게 무제한 이식.** 자체 독립 인자(F3).
- 위반 시 대가: 단위 기울기당 F_r +0.067(중앙값 OLS; +0.016..+0.102); within-cell 1sd(0.09–0.16) = F_r +0.007..+0.010.

---

### 2.5 F4 — 내륜/외곽대 점유 (독립 점유 다이얼)

**R9. [fresh-out-of-inner-ring] fresh를 내륜 r<2.5(중앙 포함 21집합체)에 넣지 마라; 하나 넣을 때마다 F_r 상승.** (A5 CONFIRMED)
- 적용법: `a_fresh_ring0_share` = radius≤2.5 슬롯(quarter 8슬롯)의 가중 fresh 비율.
- 근거: f_r +0.322 (E1_E2, p=1.2e-10), +0.310 (E3_E4), +0.605 (J5_J6, p=7.9e-14), +0.233 (J1_J2|f121); Stouffer p=1.2e-24; holdout 유의 반전 없음(K1_K2 −0.06 ns).
- 유효범위: 분산 있는 전 셀. node_peak에는 E1_E2 약양성뿐 — F_r 규칙.
- 위반 시 대가: 내륜 전노심 1집합체당 F_r +0.003..+0.005 (share unit당 +0.07..+0.10).

**R10. [fresh-into-outer-band] fresh를 외곽대 r>6.5(104집합체 — 주변부보다 넓은 밴드)로 밀어라; 이것이 이식 가능한 형태다. 원시 주변부-한정 fresh 수(RM4 축)는 부호 불일치로 사용 금지.** (A7 CONFIRMED)
- 적용법: `a_fresh_ring3_share` = radius>6.5 슬롯의 가중 fresh 비율.
- 근거: f_r 음성 5/5셀: −0.315 (E1_E2), −0.448 (E3_E4), −0.623 (J5_J6, p=8.5e-15); Stouffer p=1.9e-29; K1_K2 +0.098 ns(반전 아님). 대조: `a_fresh_periph_share`(최외곽 13슬롯만)는 셀 간 부호 반전(node_peak −0.770 E1_E2 vs +0.219 E3_E4) — RM4 반증 재현.
- 유효범위: 전 셀(K 약화). CBC 비용 동반(+0.410, §3.2).
- 위반 시 대가: 외곽대 전노심 1집합체당 F_r −0.002..−0.006 (share unit당 −0.20..−0.64).

---

### 2.6 F5 — hot/cold fresh 상대배치 및 노령연료 배치

**R11. [hot-fresh-outboard] 두 fresh 타입 중 더 뜨거운 쪽(고kinf0, 저Gd, 저FF)을 더 큰 반경에 — 냉 fresh 안쪽, 열 fresh 바깥쪽.** (A3 **DOMAIN-LIMITED — E/J 셀 한정**)
- 적용법: `a_hot_minus_cold_rmean` = (hot fresh 가중 평균반경) − (cold fresh 가중 평균반경) [pitch].
- 근거: f_r 음성 5/5 채굴 셀: −0.274 (E1_E2), −0.439 (E3_E4), −0.510 (J5_J6), −0.401 (J1_J2|f121); Stouffer p=3.2e-25; ff-gradient 조건부 생존(E3_E4 −0.230, J5_J6 −0.262).
- 유효범위: **K3_K4에서 node_peak +0.422 (p=0.016) 유의 반전, f_r +0.274 ns — K-case_pair 적용 금지(R2와 함께 family A의 경계).**
- 위반 시 대가(도메인 내): hot 타입 외향 이동 1 pitch당 F_r −0.016(중앙값; −0.007..−0.037). node_peak 규칙 아님(2/5셀). CBC 비용 +0.606(§3.2).

**R12. [hot-ring-twice-burned] 셀 명목 출력맵의 hot ring(누적 가중 다중도 ≥48/241까지의 최고출력 슬롯 집합 H)에 twice-burned를 주차하라.** (B1 CONFIRMED; feed<121에만 twice-burned 존재)
- 적용법: metric = Σ_{i∈H} w_i·[age_i==3] / Σ_{i∈H} w_i (H는 셀 명목 BOC 맵에서 도출; constructor는 fold-공유 상수 사용).
- 근거: within-cell rho −0.388 node_peak (94셀 95.7% 음성, n=5,274), −0.360 f_r, −0.464 map_cov; AUC 0.755(보호); partial|b_hot_fresh −0.332/−0.298, partial|cyclen −0.394 — 독립. 재검증 보너스: **cbc도 낮춘다(−0.291, 98% 음성)** — A군 반응도-외곽 비용을 되사는 규칙.
- 유효범위: feed<121 셀 전체(94–107셀 방향 만장일치; 탐색적 광폭 셀 포함).
- 위반 시 대가: hot ring 가중 share −0.1당 node_peak +0.22 / F_r +0.24 손실 (기울기 −2.21/−2.35 per unit).

**R13. [park-twice-burned-inboard] 3-batch 노심(feed<121)에서 twice-burned 무게중심을 안쪽에 유지하라; 외향 이동은 측정된 모든 셀에서 CBC와 평탄도를 동시에 해쳤다.** (D5 CONFIRMED)
- 적용법: `age3_r` = age≥3 슬롯의 가중 평균반경 / 최대반경 (0..1).
- 근거: →cbc_max wrho +0.412, **음성 셀 0%** (86셀, n=9,758, p=2.6e-26; partial|kbud +0.497, 재검증 +0.498); →map_cov +0.480(0% 음성); →f_r +0.316.
- 유효범위: feed<121 전 셀.
- 위반 시 대가: 무게중심 외향 0.1(노심반경비)당 CBC +32 ppm, map_cov +0.044; 안쪽 유지 비용은 cyclen ~−0.8 EFPD뿐.

**R14. [hot-ring-fresh-two-regime] hot ring의 fresh share는 2-레짐 레버다: 거친 탐색에서는 fresh를 hot ring 밖으로(share 높을수록 나쁨), in-band 엘리트 레짐에서는 챔피언들이 의도적으로 fresh 앵커를 hot ring에 얹는다(share 높을수록 좋음).** (B4 CONFIRMED — 확립 사실 1–2의 배열 통계량 버전)
- 적용법: R12와 동일 H; metric = Σ_{i∈H} w_i·fresh_i / Σ w_i. **밴드에서 멀면 스토어 부호, in-band 구성 시 반전 부호 적용.**
- 근거: 스토어 within-cell +0.230 node_peak / +0.258 f_r(285–329셀, n~40–50k; partial|RM1i +0.211/+0.238 — RM1i와 독립 축); 최신 캠페인 전부 반전: fpcamp4 −0.721/−0.668, fpcamp5 −0.33, fpcamp6 −0.34, fpcamp_minfr −0.467/−0.654 (전부 p≤1e-3). 재검증 FLAG: 가중 스토어 rho는 +0.121/+0.173로 집계 민감(중앙값 +0.34/+0.37, 부호 일치).
- 유효범위: 2-레짐 구조 자체가 유효범위 선언이다 — 레짐 오판이 최대 위험.
- 위반 시 대가: 스토어 레짐에서 share +0.1당 node_peak +0.057 / f_r +0.083; 엘리트 레짐에서 share +0.1당 f_r −0.096을 놓침.

---

### 2.7 F6/C군 — 피크 캐리어 법칙과 서로게이트 (심층 탐색 셀·스타일 선택 전용)

**R15. [burned-peak-carrier] feasible 노심에서 BOC nodal peak를 BURNED 집합체(BOC 연소도 ~8–15 GWd/tU)에 넘겨라: fresh 캐리어는 더 평탄하지만 F_r이 높고, 셀 최저-F_r 노심들은 연소 연료가 약간 더 높은 피크를 지게 한다.** (C1 **DOMAIN-LIMITED — 심층 탐색 셀(E1_E2/E3_E4) 한정**)
- 적용법: `carrier_fresh` = is_fresh(occupant(argmax_s p_boc[s])); 점유 연소도는 체인으로(R18).
- 근거: E1_E2 AUC 0.653 (n=346), 가중 0.582; 승자 대비: 저F_r 데실 캐리어 89% burned @14.3 GWd/tU vs 평탄 데실 74% fresh @4.9 (재검증 88%@14.2 / 74%@5.1, 프리미엄 +0.098); lowFr−flat node_peak +0.0597 CI [+0.011,+0.089]. **재검증이 "5/5셀" 주장 기각: J1_J2/J5_J6/K3_K4 AUC 0.413/0.484/0.324.** 깊이 구배 정량화: rho(node_peak,f_r) = −0.289 (E1_E2) → +0.931 (K3_K4) — 캐리어 스위치는 평탄 앵커 포화 후 마지막 ~0.02–0.04의 F_r을 여는 열쇠이며, 탐색이 얕은 셀에서는 보이지 않는다.
- 유효범위: within-cell rho(node_peak,f_r)<0인 심층 탐색 셀. 92.5%의 feasible 행이 이미 명목 최고온 슬롯에 fresh를 두므로(포화) 이 규칙은 **실현된** 캐리어에 관한 것이다; 명목 평균 맵은 fresh 캐리어를 99.6% 예측하나 실현은 52.4% — ~0.01 맵 해상도의 경주라서 **집행에는 맵 헤드 필요**.
- 위반 시 대가: 캐리어 fresh→burned 전환 = 평균 F_r ~−0.005 (심층 셀 승자 수준 −0.01..−0.04), 평탄도 비용 F_xy +0.03..+0.06.

**R16. [fr-fusion-surrogate-outer-loop] fr_hat = 1.047·max_{BOC,EOC steps, slots}(p_nom·FF_origin(BU_nom))를 크기(magnitude) 거부권으로만 써라 — 스타일/셀 선택의 외부 루프 veto이지, 한 생성기 스타일 내부의 순위기가 아니다.** (C3 **DOMAIN-LIMITED — magnitude veto only**)
- 적용법: fold-공유 명목맵 + 체인 BU + fuel_types FF; A = median(f_r/hot) = 1.0472.
- 근거: 크기 법칙 byte-exact 재현: A_med 1.0472, IQR [1.022,1.084], rms 0.0496–0.0499, n=679 — 전구간 법칙 A=1.0255/rms 0.035의 in-band 연장. **순위 주장 강등(재검증): elite-half +0.26–0.272는 md5 fold 해시에서만 재현, 메트릭이 그룹당 2–3레벨 준이산, J5_J6 부호반전(−0.38..−0.68 양 fold 방식); within-campaign +0.025 null.**
- 유효범위: 크기 veto(constructor는 fr_hat≤1.65 게이트로 사용)와 셀 선택만. 순위기로 쓰지 말 것.
- 위반 시 대가(오용 시): 스타일 내 순위 기대이득 0, 홀드아웃 반전 위험.

**R17. [burned-hot-product-style] 후보 스타일들 중에서는 burned-측 hot product가 높은 쪽을 선호하라 — 명목 고온 위치가 피크 부담 능력이 큰 연소 연료로 덮인 스타일.** (C4 **DOMAIN-LIMITED — 약한 스타일 타이브레이커**)
- 적용법: `hot_burned_nom` = max_{burned s} max(p_nomB[s]·FF(BU_nom[s]), p_nomE[s]·FF(BU_nom[s]+B_regime·p_nomB[s])).
- 근거: cell×fold rho vs f_r = −0.209 CI [−0.357,−0.088], 8/10그룹 음성(재검증 −0.204 md5 / −0.190 sha1, 62–75% 음성); within-style +0.036 null. R15와 같은 기전의 패턴 측 읽기.
- 유효범위: 앵커/스타일 선택만; 순열 순위 금지.
- 위반 시 대가: 스타일 간 IQR당 F_r +0.0062 (CI [−0.0109,−0.0013] 방향) 포기 — 작다. 동률 판정용.

**R18. [chain-burnup-validity] (지지 규칙) 체인 유도 슬롯 연소도 BU_nom = B_regime·Σ p_nom[direct 체인]은 맵 수확 BOC 연소도를 순위 수준으로 재현한다 — 규칙 적용에 연소도 라벨이 불필요하다.** (C5 CONFIRMED)
- 근거: 679행 median within-row Spearman 0.618(재검증 0.694 — 더 좋음), MAE 8.04–8.21 GWd/tU. 확립 사실 5의 feasible-band 확인.
- 유효범위: FF(BU) 구분(fresh 0 vs once ~25–30 vs twice ~50–60)에 충분한 ~0.6 순위 충실도. **<5 GWd/tU 절대 구분에는 사용 금지.**

---

### 2.8 엘리트 정밀화 규칙

**R19. [elite-diagonal-checkerboard] in-band 엘리트에서, 고정된 내부 fresh-쌍 예산 하에 fresh 접촉은 면접촉이 아니라 대각접촉으로(체커보드) — 구성 단계 정밀화 규칙이며 거친 탐색 규칙이 아니다.** (B5 **DOMAIN-LIMITED — 엘리트 fpcamp 레짐**)
- 적용법: `b_checker` = rm_fresh_diag_adjacency(inboard=True) − rm_fresh_face_adjacency(inboard=True).
- 근거: 스토어 null(−0.007, 46.9% 음성); fpcamp4 −0.785/−0.765 (p~1e-21, n=97), fpcamp5 −0.665/−0.720, fpcamp6 −0.648/−0.756, fpcamp_minfr −0.247/−0.207; 일반 셀 feasible 부분집합 −0.156(얇지만 동부호).
- 유효범위: **live minFr −0.092 ns; feasible E1_E2 내부에서 rm1i 축과 −0.96 공선(붕괴) — 독립 레버가 아니라 R5의 정밀화 표현으로 취급.**
- 위반 시 대가(엘리트 셀 내): 대비 쌍당 node_peak −0.0057 / f_r −0.0059 포기(관측 64쌍 범위 ~0.36 f_r); 무작위 레짐 모집단에서는 기대 이동 0.

---

### 2.9 DEAD 목록 — 통설이 제안할 법하지만 데이터가 죽인 규칙

채택 금지. 각각 사망 방식이 교훈이다.

1. **"fresh는 무조건 주변부로" (RM4, `a_fresh_periph_share`, mean fresh radius)** — 셀 간 부호 반전(node_peak −0.770 E1_E2 vs +0.219 E3_E4). 스토어 전역 반증이 in-band에서도 재현. 이식 가능한 형태는 R10(넓은 외곽대 r>6.5)뿐.
2. **"fresh를 뜨거운 자리에서 항상 치워라" (naive 첨두-회피, 보편형)** — 2-레짐(R14)에서 엘리트 측이 정확히 반대(fpcamp4 −0.721). 최평탄 노심은 fresh 고FF 연료에 피크를 **얹는다**(확립 사실 1–2). 보편 규칙으로는 사망, 레짐 조건부로만 생존.
3. **"FF>y인 fresh를 명목 p>z 슬롯에 금지" (집행형 임계 규칙)** — C2에서 명시적으로 기각: in-band 92.5% 포화(E1/E2 역할 분리는 feasibility 전제조건이지 판별자가 아님), 임계값 홀드아웃 전이 실패(train +0.428/+0.25 → holdout −0.161/−0.37/−0.42). RM4와 같은 교훈: 포화된 조건을 규칙으로 격상하지 말 것.
4. **RM2i 최소화(내부 fresh 대각 인접 회피)의 보편 적용** — 최신 캠페인에서 부호 반전(fpcamp4-6 −0.665/−0.597/−0.425; live minFr +0.538 — 부호 혼돈 확인). 스토어 양성은 RM1i 축 편승(공선 +0.385; partial diag|face +0.080). 보편 규칙으로 사망.
5. **fresh-burned 혼합쌍 극대화** — 스토어 null(−0.006/+0.024, 부호검정 ns); 엘리트 −0.60..−0.78은 RM1i의 거울(공선 −0.511). 독립 레버 아님.
6. **내부 체커보드 차수(RM6 변형, b_chk_inb)** — zero-order −0.105가 partial|RM1i +0.020으로 소멸. RM6과 같은 사망.
7. **주변부/교차 fresh쌍 "보호" (b_ff_face_pp/b_ff_face_x)** — partial|RM1i −0.043/−0.034로 붕괴; AUC 0.364는 독립 기량이 아님. RM4 역독해 실패 모드.
8. **전축 twin fresh 수(b_twin_fresh)** — +0.045, 26% 음성, AUC 0.545. 신호는 내축(k=1–4)에만 있음(G2 관련).
9. **연령 혼합비 튜닝 / tolerance_margin 최적화 / twin 깨기 / self-feed** — 구조적 비-레버(G3). 242−2F 항등식이 twice-burned 수를 고정(within-cell 분산 0); twinbreak/selffeed는 71,517행 중 0회; tolerance_margin은 전 메트릭에 |wrho|≤0.045.
10. **fresh ring1(2.5–4.5) vs node_peak** — 수치 통과(+0.63/+0.24/+0.21)이나 E1_E2 지배 + E3_E4 데실 AUC 모순으로 채굴 단계에서 이미 배제(정직성 기록).
11. **once vs twice-burned 배치 구분 규칙 (in-band)** — 검증 불가: feasible 891행 전부 depth 1 (twice-burned 0개). 부재를 규칙 부재로 오독하지 말 것 — feed<121 스토어에서는 R12/R13이 유효.

---

## 3. 규칙 간 상충 — 측정된 트레이드 구조

### 3.1 평탄도(node_peak) vs F_r: in-band 반상관 기전

feasible band 안에서 두 목적은 같은 방향이 아니다(확립 사실 1: 13-pt dose r=−0.750). 같은 레버가 부호를 바꾼다:

| 레버 | node_peak | f_r | 셀 |
|---|---|---|---|
| fresh-on-crest (R6 메트릭) | −0.680 | +0.508 | E1_E2 |
| 피크 캐리어 fresh (R15) | −0.0288 (AUC 0.336) | AUC 0.582 (fresh→고F_r) | 심층 셀 |
| hot-ring fresh share (R14) | 엘리트 −0.72 | 엘리트 −0.67 | fpcamp4 (동방향, 단 스토어 레짐에선 둘 다 +) |

- **flat 프로파일**: 피크를 fresh 고FF 앵커에 얹는다(R14 엘리트 부호, R6 역방향 허용, R1/R2 최대). 대가: F_r +0.03..+0.06.
- **minfr 프로파일**: 평탄 앵커(역할 분리)가 포화된 뒤 피크를 burned(~8–15 GWd/tU)에 넘긴다(R15) + R6/R8/R9/R10 F_r 문법 최대. 대가: node_peak +0.0597 (CI [+0.011,+0.089]).
- 이 상충은 **탐색 깊이 의존**: rho(node_peak,f_r) = −0.289 (심층 E1_E2) vs +0.931 (얕은 K3_K4). 얕은 셀에서는 두 목적이 아직 동행하므로 상충 관리가 불필요하고, 기록 경신 국면에서만 프로파일 분기가 의미를 갖는다.
- 실측 분기점: E1_E2/f121 F_r 기록 1.4636 (n_hot 64) vs flat 기록 1.1899 (n_hot 68) — feed split부터 갈라진다(R4).

### 3.2 CBC 공동예산 — family-A 레버는 붕소로 지불한다

반응도-외곽 축(F1)은 within-cell로 CBC와 강한 양의 상관:

| 지출 (cbc rho) | 수입 (cbc rho) |
|---|---|
| periphery-k (R2) +0.691 | hot-ring-twice-burned (R12) −0.291 (98% 음성) |
| hot-fresh-outboard (R11) +0.606 | inward migration (G4) 중앙값 −0.125 (69% 음성) |
| k-gradient (R1) +0.577 | 짧은 travel (R7) — travel이 +0.377이므로 회피가 곧 수입 |
| ring3 fresh (R10) +0.410 | twice-burned 안쪽(R13) — age3_r +0.412 회피 |

환율은 R3이 정한다: **+0.001 kbud = +13.8 ppm CBC = +0.72 EFPD** — cyclen으로는 CBC를 되살 수 없다. Construct는 A-family 레버를 쓸 때마다 CBC를 공동예산으로 차감해야 하며(constructor의 p98 envelope cap이 이 규율), cyclen은 620–645 밴드 중심에 두는 슬랙 변수로만 취급한다(확립 사실 4: cyclen 이탈 >~5 EFPD는 map_cov 파괴).

### 3.3 공선성 지도 — 다이얼은 6개뿐

E1_E2|f121 feasible(n=381) 인자 구조: **F1** 반응도-외곽(R1·R2·R4·R5(−)·R19·kbud·inward — in-band에서 kbud≈hot_share≈periphery-k 0.94–0.99: 붕소 예산과 평탄화 다이얼이 한 레버), **F2** 크레스트/travel(R6·R7), **F3** ff_radgrad(R8, 독립), **F4** ring0 vs ring3(R9·R10), **F5** hot_cold_dr + hot-ring fresh(R11·R14), **F6** hot_nom_traj(R16), + 포화 게이트 2(G1·G2). 점수 함수를 짤 때 같은 인자 안의 규칙에 가중을 중복 배분하면 그 축만 과대집행된다.

### 3.4 프로파일별 가중 (constructor 구현 그대로)

- 공통 게이트 G1–G8: center=E2, 축유닛 52 fresh 금지, twin/depth-1/census 불변량, inward≥0.75, fr_hat≤1.65, 명목 최고온 슬롯 46 fresh(flat), ring census (1,4,7,18) fresh 유닛.
- **minfr**: n_hot 16유닛(64/121); soft = RM1i +0.0068/pair, ff_radgrad +0.067, hot_cold_dr −0.016/pitch, hot_burned_nom(R17) 선호; R6 F_r 방향.
- **flat**: n_hot 17유닛(68/121); soft = k-gradient/periphery-k 최대(R1·R2), 피크-온-fresh 허용(R14 엘리트 부호).

---

## 4. 실증 계획 — staged acid test (사전등록 완료, 미실행)

**설계**: `rule_construct.py` — 규칙만의 탐욕 구성기(시드 first-improvement descent, 3개 폐쇄 이동: ring-보존 역할 스왑 / E1-E2 배치 스왑 / 배선 소스 스왑). 모델 없음, 스토어 엘리트 없음, 루프 내 MASTER 없음. 프로파일당 200개 생성(시드 minfr 1000–1199 / flat 5000–5199), 400개 전부 상이, 스토어 71,517행 + 미병합 `runs/v520/candidates.json` 대비 신규성 확인, 게이트 탈락 0.

**심판 예측** (챔피언 `data/models/split_S1b` v6b, referee 역할만): minfr 최선 pred F_r 1.5262 (기록 1.4636 대비 +0.0626), flat 최선 pred node_peak 1.2914 (기록 1.1899 대비 +0.1015). **판독 앵커**: 같은 심판이 스토어의 기록 자체를 +0.043/+0.036 과대예측한다(엘리트 꼬리 압축; in-band 벌크는 bias +0.010 / MAE 0.019 / Spearman +0.80). 심판-일관 좌표로는 최선 구성이 기록의 자기예측 대비 F_r +0.020 / node_peak +0.066 — **시험은 살아있고 사전에 결정되지 않았다. 예측 공간만으로는 판정 불가, MASTER 배치가 결정한다.**

**후보 8기** (`scratchpad\rules\acid_batch\candidates.json`, sha256 `2cfbc11e58026c79694e17d05f343ad5947e21e9e88f06d56bb9a91c80db52d1`): 프로파일별 예측-스크린 통과 상위 2 + 최대-Hamming 중위 정직성 프로브 2. 러너 `rule_acid_run.py` + `run_rule_acid.bat` (workers 12→박스에서 8로 클램프, 무해; dry-run PASS — 자산은 셀 고유 native restart `native:MAS_RST.APRQ_11_0635.19`, fallback 0).

**사전등록 성공 기준** (양쪽 스크립트 헤더에 고정, 체인 실행 전 확정):
- SUCCESS(minfr) = 4개 minfr 후보의 실현 F_r 최소 ≤ **1.479** (기록+0.015), cbc_max≤1600 · f_q≤2.41 · 620≤cyclen≤645 동시 충족.
- SUCCESS(flat) = 4개 flat 후보의 실현 node_peak 최소 ≤ **1.205**, f_r≤1.55 + 동일 게이트.
- 유효성 V1–V5: 8기 전수 수렴; fallback 0 + 상기 native restart; 상호 상이; 스토어 신규; sha256 인용. 각주: cbc 게이트는 신규 1600 한도(스토어 비교행은 1550 생산분).
- 보조(보고만): 8기에 대한 Spearman(pred, real).

**결과의 의미**:
- **SUCCESS**: 규칙이 탐색의 성과를 압축했다는 직접 증명 — "여러 시도를 하면서 최적 패턴을 만드는 규칙 그 자체의 학습"이라는 북극성의 실증. 규칙집은 캠페인 생성기의 1급 사전분포로 승격되고, 다음 단계는 규칙-구성 → 소규모 국소탐색 하이브리드의 호출 수 절감 정량화.
- **부분 성공(minfr만)**: 예측 공간이 시사하는 대로 flat 문법(F1 축 + 피크-온-fresh)이 F_r 문법보다 불완전하다는 판정 — E3_E4 비-반경 기전(R1 유효범위의 구멍)이 1순위 후속 채굴 대상.
- **FAILURE(양쪽)**: 규칙은 within-cell 상관으로는 진실이나 **구성적 충분조건이 아니다** — 엘리트를 만드는 정보가 ~6인자 밖(맵 수준 미세구조, R15의 0.01-해상도 경주 등)에 있다는 뜻. 이 경우 규칙집은 탐색 공간 축소기(게이트+사전분포)로 역할을 낮추고, 명시적 절반의 확장은 맵 헤드 기반 규칙(캐리어 집행 등)으로 향한다. 실패해도 사전등록 덕에 결과는 정보다 — 기준 사후 조정 금지.

---

## 5. 모델과의 관계 — 명시적 절반과 암묵적 절반

- **분업**: 이 규칙집은 학습된 지식의 **명시적 절반** — 사람이 읽고, 감사하고, 인허가 문서에 인용할 수 있는 형태다. 챔피언 넷(split_S1b v6b)은 **암묵적 절반** — 규칙이 포착 못 하는 잔차(맵 미세구조, 0.01-수준 캐리어 경주, E3_E4류 비-반경 기전)를 담는다. 실측: 규칙 6인자가 설명하는 축 위에서 심판의 in-band Spearman은 +0.80이고, 규칙-구성 노심의 예측 잔차가 바로 암묵적 절반의 크기다. 어느 쪽도 다른 쪽을 대체하지 않는다.
- **캠페인의 규칙 소비법 — 검증된 패턴은 소프트 페널티다**: RM1i A/B 교훈을 표준으로 삼는다 — 하드 컷이 아니라 **~0.02 오더의 소프트 페널티**를 생성기 점수에 더하는 방식이 캠페인에서 실증된 소비 형태다(RM1i의 쌍당 효과 +0.0026..+0.0068과 정합하는 스케일). 하드 게이트로 승격하는 것은 G군(G1–G4)처럼 위반 시 feasibility 자체가 붕괴하는 규칙뿐이다. 구체적으로:
  1. **생성기 사전분포**: G1–G4 + ring census + n_hot(R4)은 표본공간 정의로 — 위반 후보를 아예 만들지 않는다.
  2. **소프트 점수**: F1–F5 레버는 검증된 효과크기를 가중으로(constructor의 soft score 구성 그대로), 단 §3.3 공선성 지도에 따라 인자당 1회만 배분하고 §3.2 CBC 공동예산으로 cap.
  3. **외부 루프 veto**: R16(fr_hat 크기), R17(스타일 동률), C2/RM4류는 절대 내부 순위기로 격상하지 않는다 — 홀드아웃 반전이 그 대가임이 두 번 측정되었다.
  4. **심판은 심판으로**: 챔피언 모델은 후보 스코어링(57 s/400, CPU)과 스크린에 쓰되, 꼬리 압축(+0.043/+0.036)을 알고 읽는다. 규칙이 제안하고, 모델이 심사하고, MASTER가 판결한다.
- **레짐 스위치의 소유권**: R14/R19처럼 레짐 의존 규칙의 부호 선택은 규칙집이 아니라 캠페인 상태(밴드 내/외)가 결정한다 — 생성기는 자기 위치를 알아야 한다.
- **갱신 규약**: 규칙의 수치는 `records.parquet` 스냅샷의 함수다. acid test 라벨과 runs/v520 병합 후 v2에서 재검증 수치를 갱신하되, 사전등록된 성공 기준(§4)은 소급 수정하지 않는다.

---

*근거 산출물: 검증 verdict `scratchpad\rules\validated.json`, 재검증 코드 `val_metrics.py`/`val_stats.py`/`val_maps.py`/`val_c_md5.py`, per-cell 테이블 `val_percell.csv`, 구성 세트 `constructions_minfr.json`/`constructions_flat.json`, 스코어 테이블 `scored_constructions.parquet`, 스테이징 배치 `acid_batch\candidates.json`. 메트릭 구현: `family_a.py`, `lpopt/search/rule_metrics.py`(RM1i 등), 구성기 `5_RL/rule_construct.py`, 러너 `5_RL/rule_acid_run.py`.*

---

## 7. 실증(acid test) 결과 — 2026-08-11 실행, 사전등록 기준 적용

MASTER 8체인, 전원 수렴, 전원 native restart. 판정: **FAILURE** (양 프로파일 모두 기준 미달).

| 프로파일 | 규칙제 최고 (실측) | 사전등록 기준 | 탐색 기록 |
|---|---|---|---|
| minfr | F_r **1.5771** | ≤ 1.479 | 1.4636 |
| flat | node_peak **1.3245** | ≤ 1.205 | 1.1899 |

### 정직한 해석 (사전등록 §4의 두 갈래 중 FAILURE 갈래)

1. **규칙이 잡은 것 — 가능영역의 문법.** 8기 전원이 수렴했고, cyclen 633.5~637.4(전원 창 내),
   CBC 1354~1415(전원 1600 이하), F_q 전원 2.41 이하. D-계열(반응도 예산)과 게이트 규칙은
   손으로 지어도 작동한다. 무작위 합법 패턴의 F_r이 2.5~4.6임을 감안하면 규칙만으로 1.52~1.65에
   도달한 것은 공간의 대부분을 건넌 것이다.
2. **규칙이 못 잡은 것 — 마지막 ~0.1.** 탐색이 찾는 최적점(1.4636/1.1899)과의 잔차는 규칙
   26개의 선형 결합이 아니라 미세 배열 상호작용에 산다. 챔피언 심판도 규칙제 노심의 F_r을
   +0.065 과소예측했다(오프-매니폴드 신호) — 규칙제 코어는 학습 분포 밖이다.
3. **따라서 규칙의 정당한 소비처는 §5에 등록된 그대로다**: 생성기 프라이어·소프트 페널티·
   실현가능성 게이트(캠페인 후보풀의 문법 검사)이지, 단독 최적화기가 아니다. "규칙 그 자체의
   학습"은 필요조건의 명시화까지 도달했고, 충분조건의 나머지 절반은 암묵적 모델+탐색에 남아
   있다 — 이것이 이 실측의 결론이다.
