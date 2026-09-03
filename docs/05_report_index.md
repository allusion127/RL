# 05 — 리포트 색인

`lpopt` 프로그램이 2026-07-16 → 2026-08-31에 생산한 **모든 markdown 리포트**의 색인이다.
연대기는 [04_timeline_and_results.md](04_timeline_and_results.md)를 보라.

**표기 규약**

- 링크는 이 문서(`docs/`)에서 저장소 루트로 올라가는 상대경로(`../data/reports/…` = 루트 기준 `data/reports/…`).
- **유형** — `사전등록`(라벨 존재 전에 규칙을 동결한 문서) · `결과`(사전등록에 대해 채점한 문서) ·
  `판정`(verdict / 채택·기각 결론) · `설계`(계획·설계서, 실행 전) · `메모`(분석 노트·참고) · `감사`(적대적 검토·포렌식).
- 계산기 박스는 공개 저장소 규칙상 **HOST_238**(학습 GPU) · **HOST_199 / HOST_198 / HOST_181 / HOST_104**(MASTER 생산 PC).
- 총 **138건** — `data/reports/` 123건(하위 디렉터리 9건 포함; 최상위만 세면 114건) +
  저장소 내 기타 5건 + 저장소 미포함 캠페인 리포트 10건.
- **§10**이 2026-08-29 목적축 전환(`F_r` → `F_xy`) 이후의 20건을 따로 모아 시간순으로 싣는다.

---

## 1. 모델 · 정확도 (19건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [model_report.md](../data/reports/model_report.md) | 07-17 | 결과 | PosValNet 5멤버(1.59M/멤버) 기준선 S0/S1/S2/S4 수용 판정표 | S2/S4 cyclen·CBC R² PASS, S0/S1 FAILED; 셀내 Spearman은 트리 대비 전부 PASS |
| [phaseD_S1_report.md](../data/reports/phaseD_S1_report.md) | 07-17 | 결과 | S1 단독 재평가 — 전역 R²와 셀내 ρ의 괴리 노출 | f_r R² 0.862인데 셀내 ρ 0.898 — **전역 지표는 셀 규모차로 왜곡된다** |
| [fuel_types_v4_harvest.md](../data/reports/fuel_types_v4_harvest.md) | 07-18 | 메모 | cond_v4 물리 피처 17열 수확 커버리지·범위(120행×34열, filled 84) | glued-negative 파서 수정이 CR1 17블록·ga80 `bu_k1` 36/36 NaN 복구 |
| [pinbu_forensics.md](../data/reports/pinbu_forensics.md) | 07-18 | 감사 | `max_pin_burnup` 랭킹 실패 원인 추적 | "−0.818"은 n=11 아티팩트지만 **셀내 OOS 일반화 실패는 실재**; per-type BRP 처방은 무효 증명 |
| [kinf_shape_features.md](../data/reports/kinf_shape_features.md) | 07-20 | 설계 | k-conv 곡선형상 9채널 정의·커버리지 + v5 채널 계획 | 형상 인자가 **poison-agnostic 보편 독봉 서명** — Gd 설계축 대체 가능 |
| [v5_experiment.md](../data/reports/v5_experiment.md) | 07-21 | 판정 | v5 통합 A/B 5-arm 결정표(36셀 셀내 ρ·MAE·P@8 + legacy tail) | **채택 `v5_distill_w160`** (cyclen ρ 0.7666 / f_r ρ 0.9016 / f_r MAE 0.1402) |
| [dataset_adversarial_20260721.md](../data/reports/dataset_adversarial_20260721.md) | 07-21 | 감사 | 36셀 셀내 Spearman 전표 + 데이터셋 적대적 검토 7소견 | 관측 0.49–0.94는 **잡음 한계가 아니라 모델/피처 한계**; F_r<1.55 라벨 579건이 전부 f121 |
| [parity_round1c_20260722.md](../data/reports/parity_round1c_20260722.md) | 07-22 | 결과 | `fuelcost_round1c` 8셀 292후보 예측-실측 패리티 전수 | **F_r 경계 순위스킬 결여가 최대 약점**(전체 ρ 0.43, 경계 0.13); cyclen ρ 0.923 |
| [parity_round1c_inline.md](../data/reports/parity_round1c_inline.md) | 07-22 | 결과 | 같은 패리티의 셀별 축약판 + 해석 | F_r 비보수 편향 −0.084(78.3%) → **feasible 선언은 반드시 실측으로** |
| [nodal_power_parity_20260723.md](../data/reports/nodal_power_parity_20260723.md) | 07-23 | 결과 | map head 노드 출력분포 공간·순위 정확도 정밀 분석 | BOC 분포 ρ 0.948; **최경계 F_r<1.65에서 맵피크가 스칼라를 역전**(0.41 vs 0.04) → 하이브리드 근거 |
| [model_accuracy_20260725.md](../data/reports/model_accuracy_20260725.md) | 07-25 | 결과 | 챔피언 #7 전 타깃 정확도 종합(무오염 fold C 1,753행) | 최약축 = F_r 1.55 근방 ρ −0.018 + **불확실성 전면 붕괴**(cyclen σ 866 EFPD) |
| [cyclen_nodepeak_resolution_20260725.md](../data/reports/cyclen_nodepeak_resolution_20260725.md) | 07-25 | 결과 | cyclen 3단 분해 + 맵헤드 69슬롯 전체맵 재생성 + 2-D FFT | **#7이 실제 학습한 파라미터는 6,298개(0.2%)** — 공간 표현은 동결 트렁크 위 644-파라미터 판독기 |
| [precision_ceiling_roadmap_20260725.md](../data/reports/precision_ceiling_roadmap_20260725.md) | 07-25 | 판정 | 셀내 ρ 0.99 달성 가능성 — 라벨 잡음 상한 3추정자 + 학습곡선 | **라벨은 병목이 아니다**(상한 ρ ≥ 0.997). 0.99엔 셀당 5.5만–92만 표본; **tol 강화 기각** |
| [transpose_noise_measured_20260725.md](../data/reports/transpose_noise_measured_20260725.md) | 07-25 | 결과 | 전치쌍 24쌍/22쌍 수렴 실측으로 잡음 상한을 상계→직접측정 승격 | **★정정: F_r<1.55 경계대 상한은 0.84** — 그 구간 ρ 0.9는 물리적으로 불가능 |
| [hires_model_ab_design_20260725.md](../data/reports/hires_model_ab_design_20260725.md) | 07-25 | 사전등록 | 고해상도·고분해능 7-arm A/B 설계 + 확산 출력맵 프라이어 실증 | 맵 라벨 해상도 상한 = 현재값(69슬롯×4채널); 분해능 확대의 진짜 경로는 **프로듀서 측 harvest** |
| [hires_ab_results.md](../data/reports/hires_ab_results.md) | 07-25 | 결과 | 8-arm fold C 채점표(Δ₇₅/SD, 셀내 ρ, 게이트 프록시) | **채택 A6** — 구조 개선이 순수 용량의 5배, 파라미터 효율 44배 |
| [model_accuracy_20260729.md](../data/reports/model_accuracy_20260729.md) | 07-29 | 결과 | 승격 직후 두 챔피언 iso ρ 쌍대 비교(36셀 동일 홀드아웃) | **node_peak iso ρ 0.801 → 0.907**(P@10% 0.52 → 0.74); 전역 r만 인용하면 과장 |
| [model_evolution_20260816/README.md](../data/reports/model_evolution_20260816/README.md) | 08-16 | 결과 | 챔피언 8세대를 같은 고정 평가면(36셀 3,207행)에 전부 재서빙 | **잡음 바닥 초과 개선은 1→2세대(v6→v6b) 단 한 번**; "평평한 것이 실패는 아니다" |
| [dual_trunk_cyclen_isolation_DRAFT.md](dual_trunk_cyclen_isolation_DRAFT.md) | 07-24 | 설계 | cyclen 동결 경로 + F_r 가변 경로의 듀얼 트렁크 초안 | **미구현, 승인 대기.** Option B(동결 stem 공유) 우선 권고, Option C 기각 |

## 2. A/B · 게이트 (25건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [gate_noise_analysis.md](../data/reports/gate_noise_analysis.md) | 07-18 | 감사 | 정직 게이트 no-regression의 잡음 대 실제회귀 분해(부트스트랩 B=20k) | 홀드아웃 정직성 확인(30/30 일치); **ε 0.05는 오기각률 53% → 0.10으로 재보정** |
| [v4_ab_gate.md](../data/reports/v4_ab_gate.md) | 07-18 | 판정 | cond_v4(43ch)가 셀 4 게이트를 고치는지 블라인드 A/B | **v4 승격 기각** — cyclen 붕괴는 4–8배 줄였으나 ε 초과, f_r 밴드간섭 0.382 잔존 |
| [ab2_preregistration_20260730.md](../data/reports/ab2_preregistration_20260730.md) | 07-30 | 사전등록 | A/B 라운드 2 VARIANCE arms 모(母) 사전등록(추정자·임계·harm 레일) | 수용기준 갭 CBC 16× · node_peak 20×; **남은 것은 분산이고 보정은 못 건드린다** |
| [ab2_verdict_20260731.md](../data/reports/ab2_verdict_20260731.md) | 07-31 | 판정 | 라운드 1 arm A1/A2/A3 채점 | **A1·A2 HOLD, A3 REJECT.** 5개 arm×축 중 0건 이득 → **손실/라벨 공학 반증** |
| [ab2_addendum_E10_20260731.md](../data/reports/ab2_addendum_E10_20260731.md) | 07-31 | 사전등록 | arm E10(`--ensemble 10`) 부속 사전등록 + 비용 선언 | 서브 비용 영구 2배가 2축 기준선의 근거 |
| [ab2_verdict_E10_20260731.md](../data/reports/ab2_verdict_E10_20260731.md) | 07-31 | 판정 | E10 채점 | **REJECT** — 5개 분산축 중 0건 개선. **`--ensemble 20`은 명시적으로 금지** |
| [ab2_addendum_R3_20260801.md](../data/reports/ab2_addendum_R3_20260801.md) | 08-01 | 사전등록 | arm R3(결핍셀 라벨 증산) 부속 사전등록 | 서브 비용 zero-delta가 1축 기준선의 유일한 정당화 |
| [ab2_verdict_R3_20260801.md](../data/reports/ab2_verdict_R3_20260801.md) | 08-01 | 판정 | R3 채점 | **REJECT** — `map_cov` 이득 확립(첫 gain)했으나 cyclen established worse. 프로그램 최고 난도 판정 |
| [v520_preregistration_20260810.md](../data/reports/v520_preregistration_20260810.md) | 08-10 | 사전등록 | verify-5-of-20(objection K2) 설계·결정규칙·유효성 게이트 | "≳0.03 F_r을 못 이기면 내 전제가 틀렸고 모델이 병목"이라는 자기 반증조건 |
| [v520_addendum_b2_20260810.md](../data/reports/v520_addendum_b2_20260810.md) | 08-10 | 사전등록 | 배치 2 복제(HOST_198) 부속 — 자산 해시로 교차박스 교란 폐쇄 | 배치 1 P1−P5 +0.0071은 단일 배치로 결론 불가 |
| [v520_addendum_b3_20260810.md](../data/reports/v520_addendum_b3_20260810.md) | 08-10 | 사전등록 | 배치 3(결정 배치) 부속 + 3배치 채점 도구 등록 | **SUPPORTS** — P1−P5 평균 +0.0410, 랜덤널 초과 +0.0111 → top-5 검증 정책 채택 |
| [ab2_addendum_BU_20260810.md](../data/reports/ab2_addendum_BU_20260810.md) | 08-10 | 사전등록 | arm BU(연소도 배치) — `NOMINAL_CYCLE_BURNUP` 고정을 실측 테이블로 | **PASS·승격** — node_peak 셀중앙 MAE +0.01734(기준의 4.2배). 5 arm 연속 기각 후 첫 통과 |
| [ab2_addendum_ADF_20260810.md](../data/reports/ab2_addendum_ADF_20260810.md) | 08-10 | 사전등록 | arm ADF(face ADF 3채널, v6c) — 착수 전에 결론을 예고한 사전등록 | 추가하려던 채널 3개가 **v4부터 이미 대조군에 있었다** → 이후 **REJECT**(MAE −0.0144 악화) |
| [ab2_addendum_SPLIT_20260810.md](../data/reports/ab2_addendum_SPLIT_20260810.md) | 08-10 | 사전등록 | arm SPLIT(스플릿 재구축 + 데이터 성장) | **PASS·승격 `split_S1b`** — 저-F_r 학습코어 5 → 73, 엘리트 F_r MAE 0.856 → 0.594 |
| [ab2_addendum_S1C_20260812.md](../data/reports/ab2_addendum_S1C_20260812.md) | 08-12 | 사전등록 | 라운드 7 — 첫 T-셀 라벨 468행 편입 재학습 | **PASS·승격 `s1c`.** 챔피언은 T-셀 행을 한 줄도 본 적이 없었다 |
| [ab2_addendum_S1D_20260814.md](../data/reports/ab2_addendum_S1D_20260814.md) | 08-14 | 사전등록 | 라운드 8 — 두 번째 T-셀 refresh | **PASS·승격 `s1d`.** 라운드당 이득 감쇠 −0.042 → −0.015가 재학습 시점의 근거 |
| [ab2_addendum_S1E_20260815.md](../data/reports/ab2_addendum_S1E_20260815.md) | 08-15 | 사전등록 | 라운드 9 — 세 번째 refresh + **캠페인 종결 조항** | **PASS·승격 `s1e`**, 그리고 **T6_T4 종결 발화**(r8 이득 0.0048 < 0.0050) |
| [ab2_addendum_S1F_20260816.md](../data/reports/ab2_addendum_S1F_20260816.md) | 08-16 | 사전등록 | 라운드 10 — 종결 후 feed 지지 + 개입 라벨 refresh | **PASS·승격 `s1f`(7대).** 개입 583행은 학습 포함·측정 제외 규칙 등록 |
| [ab2_addendum_S1G_20260816.md](../data/reports/ab2_addendum_S1G_20260816.md) | 08-16 | 사전등록 | 라운드 11 — f113 최적화 프런티어 라벨 100행 | **PASS·승격 `s1g`(8대).** 최소 증분이자 최고가치 — produce 라벨의 해독제 |
| [ab2_addendum_S1H_20260817.md](../data/reports/ab2_addendum_S1H_20260817.md) | 08-17 | 사전등록 | 라운드 12 — `cond_v7`(글로벌 18) 3종 채점기 enabling 재학습 | 세 가지가 동시에 움직이는 라운드임을 각주가 아니라 서두에 명시 |
| [ab2_results_S1H_20260817.md](../data/reports/ab2_results_S1H_20260817.md) | 08-17 | 결과 | S1h 채점 + 프로버넌스 전표 | **PASS·승격 `s1h`(9대).** worst_drop 0.0214/ε 0.1388. **`f_r`은 REPORT-ONLY 축**임을 자진 고지 |
| [ab2_addendum_S1I_20260817.md](../data/reports/ab2_addendum_S1I_20260817.md) | 08-17 | 사전등록 | 라운드 13 — `cond_v8`(조성 폭 3→5) 4·5종 채점기 | **등록된 기대 = 이득 없음, 기준은 비회귀뿐.** 이후 PASS → `s1i`(10대) |
| [scaling_prereg_20260815.md](../data/reports/scaling_prereg_20260815.md) | 08-15 | 사전등록 | 용량·앙상블 스케일링 A/B 설계(승격 없음 선언 포함) | 커미셔닝 전제(챔피언 ~1.5M) 정정: 실제 10.4M |
| [scaling_results_20260815.md](../data/reports/scaling_results_20260815.md) | 08-15 | 판정 | 스케일링 두 축 채점 | **전부 기각.** 축소만 유의하게 악화 → **10.4M이 무릎점**. **★노이즈 바닥 ~0.01 / ~0.018 실측** |
| [adversarial_review_20260719.md](../data/reports/adversarial_review_20260719.md) | 07-19 | 감사 | 5렌즈 적대적 공격 → 독립 재현 → 트리아지 | **GREEN(조건부), 확정 major 0건.** 확정 3건은 전부 dormant 소비처. 누락 5건은 날조하지 않음 |

## 3. 캠페인 · 프런티어 (16건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [extract_report.md](../data/reports/extract_report.md) | 07-16 | 결과 | Dataset A 추출 감사 전표(파일별 행수·라이브러리·지지 히스토그램) | 38,854 고유 레코드, 감사 ground truth와 정확히 일치. **feed 121 편중 최초 관측** |
| [flatness_first_program_20260725.md](../data/reports/flatness_first_program_20260725.md) | 07-25 | 설계 | F_r 은퇴 후 평탄도·노드첨두 목적함수 프로그램(개정판, D11) | 가중치 역전(node_peak 1.0/map_cov 0.5) · **최평탄점 미납품 규칙** · 판정장치 전면 교체 |
| [flat_assembly_fr_verdict_20260809.md](../data/reports/flat_assembly_fr_verdict_20260809.md) | 08-09 | 판정 | 평탄 집합체 → 노심 F_r TIER 0 실측 판정 | **모집단 수준에서만 이득**(단일 패턴은 9종 중 1건). **연료는 2차 레버**(탐색이 8배) |
| [session_report_20260810.md](../data/reports/session_report_20260810.md) | 08-10 | 결과 | 08-10 하루의 판정 3건·챔피언 2교체·프런티어 종합 | node_peak **1.1899**; F_r<1.50 인구 10 → 96행; J5_J6가 K3_K4를 27/32 우세 |
| [feedgrid_pathfinder_20260815.md](../data/reports/feedgrid_pathfinder_20260815.md) | 08-15 | 결과 | feed 축 25셀 생산 그리드 pathfinder | **전제 반증** — 스토어 feed 축은 이미 넓다(f121 61%, 99.7% 아님); 22/25 셀이 기존 라벨 보유 |
| [ablation_wave_prereg_20260815.md](../data/reports/ablation_wave_prereg_20260815.md) | 08-15 | 사전등록 | 1-move 어블레이션 웨이브(누설 중재 + policy v1 전향 검정) | 관측 데이터로 답할 수 없는 두 질문을 **개입 실험**으로 정의 |
| [ablation_wave_results_20260815.md](../data/reports/ablation_wave_results_20260815.md) | 08-15 | 결과 | 150체인/146 수렴 개입 실험 채점 | **누설 SETTLED**(외곽 fresh는 cyclen을 지불) · **policy v1 낙제**(AUC 0.492) · 셀 기록 1.4685 |
| [batchswap_wave_prereg_20260815.md](../data/reports/batchswap_wave_prereg_20260815.md) | 08-15 | 사전등록 | `batch_swap` 심층 표본 웨이브(4/19,820 저노출 클래스) | 종결 조항을 재개하지 않음을 명시(캠페인 라운드가 아니라 열거) |
| [batchswap_wave_results_20260815.md](../data/reports/batchswap_wave_results_20260815.md) | 08-15 | 결과 | 220체인/100% 수렴 채점 | **신기록 F_r 1.4605**(ga80 1.4636 하회 3건). **단 λ-목적에서는 r8 보드가 승리** |
| [batchswap625_wave_prereg_20260815.md](../data/reports/batchswap625_wave_prereg_20260815.md) | 08-15 | 사전등록 | 같은 클래스의 625 EFPD 분지 전이 가설 | in-band 1.4636 돌파를 스트레치 목표로 등록 |
| [batchswap625_wave_results_20260815.md](../data/reports/batchswap625_wave_results_20260815.md) | 08-15 | 결과 | 220체인 채점 — **음성 결과로 보고** | **전이 실패**(개선률 0.005 vs 0.052, p=0.0028). **λ-검사 의무화**, 연산자 일반화 철회 |
| [viz_20260815/README.md](../data/reports/viz_20260815/README.md) | 08-15 | 메모 | T6_T4 최적화 **과정 자체**의 시각화(930행/790 수렴, 13프레임) | out-in 반경 서명이나 **상위 노심의 차이는 링 선택이 아니라 링 안의 배열** |
| [fpcamp_N1N2_f113_results_20260816.md](../data/reports/fpcamp_N1N2_f113_results_20260816.md) | 08-16 | 결과 | 저-feed 개방 캠페인 `N1_N2`/f113(챔피언 s1f, 100콜) | **F_r 1.4961 @ 641.6 EFPD, feasible 41/100**, 바닥 1.7243 → 1.4961. (핀은 08-20 FAIL) |
| [fpcamp_E1E2_f109_pinburnup_prereg_20260817.md](../data/reports/fpcamp_E1E2_f109_pinburnup_prereg_20260817.md) | 08-17 | 사전등록 | 캠페인 8/100 콜·feasible 0 시점에 등록한 핀 연소도 마크 | 자기 프로그램의 이전 보고서를 고발 — "뒤집힐 가능성이 낮다 ≠ 확인했다" |
| [fpcamp_E1E2_f109_results_20260817.md](../data/reports/fpcamp_E1E2_f109_results_20260817.md) | 08-17 | 결과 | 저-feed 개방 캠페인 `E1_E2`/f109 채점 | **F_r로는 열렸으나(1.4787) 납품엔 핀-제한** — 52/52 전부 예측 핀 > 80 |
| [fpcamp_HGD569_f109_prereg_20260817.md](../data/reports/fpcamp_HGD569_f109_prereg_20260817.md) | 08-17 | 사전등록 | 붕소-개방 고-Gd 셀 F_r 강습(feed 109, 예산 60, 핀 게이트 78) | 게이트 도달을 기대하지 않는다고 미리 명시 |

## 4. 정책 (7건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [loading_rules_v1_20260811.md](../data/reports/loading_rules_v1_20260811.md) | 08-11 | 판정 | 71,517 MASTER 노심에서 채굴한 **장전 규칙집 v1** 26규칙 + acid test | **18 CONFIRMED / 6 DOMAIN-LIMITED / 0 DEAD.** 규칙만으로 F_r 1.52–1.65, 잔차 ~0.1 |
| [policy_corpus_20260815.md](../data/reports/policy_corpus_20260815.md) | 08-15 | 결과 | 개선-스텝 코퍼스 채굴(27,458 스텝, same-cell MOVES 19,820) | SA 로그 21,766 복구가 cross-cell 비율을 75% → 28%로 낮췄다; 보드 디코드 실패 0건 |
| [policy_v1_prereg_20260815.md](../data/reports/policy_v1_prereg_20260815.md) | 08-15 | 사전등록 | 무브 스코어러 v1 설계(CNN vs MLP, 베이스라인 3종, 누출 가드 9건) | 서로게이트가 아니라 **무브 스코어러**임을 명시 |
| [policy_v1_results_20260815.md](../data/reports/policy_v1_results_20260815.md) | 08-15 | 결과 | v1 채점 | **CNN arm 양 헤드 PASS**(1.69M), MLP FAIL — 단 in-distribution 한정 |
| [policy_v2_prereg_20260817.md](../data/reports/policy_v2_prereg_20260817.md) | 08-17 | 사전등록 | v2 설계(시대 데이터·`d_fresh_enr_mass`·개선률 지표 교체) | v1이 라이브 운전점에서 실패한 세 원인에 하나씩 대응 |
| [policy_v2_results_20260817.md](../data/reports/policy_v2_results_20260817.md) | 08-17 | 결과 | v2 채점(Run A 등록 프로토콜 + Run B 선언된 이탈) | **GATE FAIL**(2조항 중 1, 1개 baseline) / **DEPLOYMENT BAR PASS**(4개 전부, 양 헤드) |
| [autoeng_design_20260817.md](../data/reports/autoeng_design_20260817.md) | 08-17 | 설계 | 자동 엔지니어 `autoeng.py` 7단계 파이프라인 설계서 | **새 알고리즘 0** — 증명된 조각의 조합. 신규 기여는 사전등록 자동화·상태 로그·인간 게이트 |

## 5. 그물망 · 다종 (11건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [scoping_mesh_20260815/README.md](../data/reports/scoping_mesh_20260815/README.md) | 08-15 | 결과 | `(e_core × feed)` 55셀 스코핑 재장전 설계지도(MASTER 0, 추론만) | 주기길이 목표 없이 **안전인자 만족 하 최대화**, 셀마다 Pareto 대표점 3종 |
| [scoping_mesh_20260815/comparison_readout.md](../data/reports/scoping_mesh_20260815/comparison_readout.md) | 08-15 | 판정 | 그물망 vs MASTER 가용노심 DB 6,113기 대조(s1e/s1f/s1g 3판) | **열렸다고 한 곳 8/8 맞고 닫혔다고 한 곳 22/22 틀렸다** — 원인은 **학습자료 사각지대** |
| [dbx_lrm_fit_20260816.md](../data/reports/dbx_lrm_fit_20260816.md) | 08-16 | 결과 | DB 6,113기로 보정 LRM 백본 적합·검정 | 질량수지 −0.519%±0.098%; **교과서 LRM은 feed축에서 ±1.8% 틀리다**; `bu_k1` featurize가 −9% |
| [dbx_frontier_note_20260816.md](../data/reports/dbx_frontier_note_20260816.md) | 08-16 | 메모 | 조성수준 프런티어 표 286셀 × 74열(스크리닝 프라이어) | `N3_N4@f101`·`N1_N2@f105`는 **코어 100%가 75 GWd/tU 초과 = 인허가 막다른 길** |
| [reloadmap_methodology_20260816.md](../data/reports/reloadmap_methodology_20260816.md) | 08-16 | 메모 | 재장전 설계 그물망을 실계산으로 그리는 법(문헌 23편) | **뼈대는 보정 LRM이 그리고 서로게이트는 안전인자 오버레이** |
| [mesh_style_spec_20260817.md](../data/reports/mesh_style_spec_20260817.md) | 08-17 | 설계 | 그물망(메쉬) 그림 **표준 양식** 사양 — 축·등-feed선·노드 채움/테두리·범례 | 사용자 지정 양식으로 통일. **핀 등급이 없는 격자점은 "미산출"로 남긴다(완전성보다 정직성)** |
| [mesh_v3_20260817/PREREG_mesh_v3_20260817.md](../data/reports/mesh_v3_20260817/PREREG_mesh_v3_20260817.md) | 08-17 | 사전등록 | 고농축 확장 메쉬 v3(e 5.0–6.5 × f109–129) 배분·예산 동결 | 착수 전에 **지시서 전제 2건을 반증**하고 바로잡음 |
| [mesh_v3_20260817/PREREG2_anchor_redirect_20260817.md](../data/reports/mesh_v3_20260817/PREREG2_anchor_redirect_20260817.md) | 08-17 | 사전등록 | 앵커 재조준(F_r 조준 → 붕소 조준) 새 사전등록 | PREREG-1 §5 무효화 조건에 따른 정식 이탈. 3개 예측을 지출 전에 기록 |
| [mesh_v3_20260817/README.md](../data/reports/mesh_v3_20260817/README.md) | 08-17 | 결과 | v3 90셀 스윕 + MASTER 앵커 254체인/178 수렴행 | **⚠ 병목은 F_r이 아니라 CBC** · **LRM 확증**(ρ 0.975) · **붕소 벽 돌파, 병목은 F_r로 이동**(Tier-1 0개) |
| [mesh_multitype_20260818/PREREG_multitype_mesh_20260818.md](../data/reports/mesh_multitype_20260818/PREREG_multitype_mesh_20260818.md) | 08-18 | 사전등록 | 다종(2/3/4종) 90셀 스윕 + 앵커 캠페인 설계 | **지지도 등급을 전 행에 인쇄**; 4종은 어떤 결론에도 단독 사용 금지 |
| [mesh_multitype_20260818/README.md](../data/reports/mesh_multitype_20260818/README.md) | 08-18 | 결과 | 다종 스윕 채점 + P3b 앵커 캠페인 중단 기록 | **계단화 부호가 R1 경계에서 정확히 갈린다**(mono 10/10 이득, cross 19/19 손해); 3종은 paramA 전용 |

## 6. 연료설계 · 축방향 · v3 (11건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [sdm_mtc_limits_20260726.md](../data/reports/sdm_mtc_limits_20260726.md) | 07-26 | 메모 | APR1400 SDM/MTC 인허가 제한치 확정(DCD Tier 2 Ch.4 Rev.3) | MTC −54.0 ~ +9.0 pcm/°C(HZP) · **SDM required 10,870 pcm = 10,180 + 690** |
| [flat_assembly_fr_plan_20260802.md](../data/reports/flat_assembly_fr_plan_20260802.md) | 08-02 | 설계 | 평탄 집합체 → 노심 F_r 실행 실험 계획(TIER 0/1) | **TIER 0은 무비용으로 오늘 발사 가능**; 템플릿 Gd 레이아웃 결함 ΔFF 0.049 ⇒ ΔF_r 0.065 |
| [kcurve_fusion_memo_20260809.md](../data/reports/kcurve_fusion_memo_20260809.md) | 08-09 | 판정 | (k곡선+농축도) 표현 → 평탄도 → F_r 융합 제안 의사결정 메모 | **CONDITIONAL GO.** 충분통계 **반증**(효과비 14배), **두 번째 특성 = ADF**, 곡선 단독 업그레이드 금지 |
| [opmodel/OPSCREEN.md](../opmodel/OPSCREEN.md) | 08-11 | 판정 | lat1600 설계공간을 **운전점**(보정 LRM)에 재스크리닝 | cyclen 4.3 EFPD rms / CBC 37 ppm rms; **T5_T6는 운전점 없음**; **FF가 아니라 반응도 contrast** |
| [pitch_radius_readiness.md](../data/reports/pitch_radius_readiness.md) | 07-19 (추정) | 설계 | 핀 피치·반경 최적화의 타당성·준비도 검토(실행 스펙 포함) | **가능하되 하드 스코핑** — 집합체 피치 토큰 20.7772 동결이 조건. 피치 상한 +1.06% |
| [tripletype_design_20260817.md](../data/reports/tripletype_design_20260817.md) | 08-17 | 설계 | 3-신연료종(graded) 확장 — 9개 파일 최소 변경 + 테스트 | **2종 경로 바이트 동일** 유지(sha256 대조). `graded_morph()` 신규 연산자 |
| [tripletype_f125_prereg_20260817.md](../data/reports/tripletype_f125_prereg_20260817.md) | 08-17 | 사전등록 | 최초 3종 캠페인 `S3_<mid>_S5`/f125 사전등록(중간종 선택 규칙 포함) | 계단화는 F_r을 직접 치는 교과서 정공법 — 이 셀은 실패모드가 좁아서 선정 |
| [tripletype_f125_results_20260817.md](../data/reports/tripletype_f125_results_20260817.md) | 08-17 | 결과 | 첫 3종 캠페인 채점(60/60 콜, 8웨이브) | **STRETCH 달성 −0.0364: joint-clean F_r 1.5993**(57번째 콜). PRIMARY 0/49 |
| [tripletype_f125_r2_prereg_20260820.md](../data/reports/tripletype_f125_r2_prereg_20260820.md) | 08-20 | 사전등록 | 3종 라운드 2 — 예산만 늘린 재실행 | r1은 **예산 한계이지 탐색 한계가 아님**(프런티어가 57/60 콜에서도 이동). **결과 리포트 미존재** |
| [coreagnostic_v3_design_20260817.md](../data/reports/coreagnostic_v3_design_20260817.md) | 08-17 | 설계 | 노심 불가지(core-agnostic) v3 설계·타당성·갭 인벤토리(1,200행) | **비전의 절반은 이미 구현**, 나머지는 상수 튜플 파라미터화. **핵심 작업 = 누설의 노심크기 함수화** |
| [axial_upgrade_plan_20260817.md](../data/reports/axial_upgrade_plan_20260817.md) | 08-17 | 설계 | 축방향 블랭킷·Gd 컷백 업그레이드 계획(PLANNING ONLY) | **착수 금지, 승인 대기.** 블랭킷 15 cm 고정·농축도 자유(사용자 확정) |

## 7. 핀 연소도 (14건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [fpcamp_HGD569_f109_results_20260817.md](../data/reports/fpcamp_HGD569_f109_results_20260817.md) | 08-17 | 결과 | 고-Gd 셀 f109 F_r 강습 채점(핀 게이트 78 적용) | **STRETCH 달성** — 바닥 2.0481 → **1.6743**, 어느 셀에서 측정된 것보다 큰 F_r 이동 |
| [fpcamp_HGD569_f125_prereg_20260817.md](../data/reports/fpcamp_HGD569_f125_prereg_20260817.md) | 08-17 | 사전등록 | 같은 셀 feed 125 피벗(유일한 실질 변경 = feed) | f109 종료 직후·f125 결과 존재 전에 등록 |
| [fpcamp_HGD569_f125_results_20260817.md](../data/reports/fpcamp_HGD569_f125_results_20260817.md) | 08-17 | 결과 | f125 채점(R1–R4 + R-PIN 5 판독 전부 응답) | **STRETCH 달성 1.6357 CBC-clean**, 예측 핀 76.96 통과 → 5게이트 최근접(당시) |
| [hgd569_f125_seedctl_prereg_20260817.md](../data/reports/hgd569_f125_seedctl_prereg_20260817.md) | 08-17 | 사전등록 | R-SEED 대조군 — 알파벳 2 고정 + 동일 도너 시딩 | 3종 헤드라인이 두 가지를 동시에 바꿨다는 교란을 정면으로 닫는 설계 |
| [hgd569_f125_seedctl_results_20260817.md](../data/reports/hgd569_f125_seedctl_results_20260817.md) | 08-17 | 결과 | 대조군 채점(60/60 콜) | **교란이 거의 반으로 갈린다** — 2종+도너 1.6172(혼합대), 2종 1.6357, 3종 1.5993 |
| [pinbu_wave_prereg_20260820.md](../data/reports/pinbu_wave_prereg_20260820.md) | 08-20 | 사전등록 | 실측 핀 연소도 웨이브 44체인(기계판독 쌍둥이 JSON 동반) | `lpopt optimize`가 핀을 아예 측정하지 않는 배선 갭을 문서화 |
| [pinbu_wave_results_20260820.md](../data/reports/pinbu_wave_results_20260820.md) | 08-20 | 결과 | 44체인 채점 + 납품 판정표 | **f113 0/5 · f109 0/5 FAIL, hgd569 2종·3종 각 5/5 PASS.** 프로버넌스 불일치 3건은 드롭 |
| [pinbu_audit_20260820.md](../data/reports/pinbu_audit_20260820.md) | 08-20 | 감사 | 핀 판정 사슬 적대적 감사(관측량 정의·DB 대응·바이어스 귀인) | 관측량 불일치 확정 · DB 대응은 `pinmax_node` 계층(+9~20%) · **바이어스는 커스가 아니라 OOD 외삽** |
| [pinbu_rodavg_20260820.md](../data/reports/pinbu_rodavg_20260820.md) | 08-20 | 감사 | 봉평균 재파싱으로 M2를 닫으려는 시도 | **VOID** — `keep_success=false`라 `MAS_PPI` 49기 전원 0/49 생존. MASTER 재실행 필요 |
| [pinbu_rodavg_true_20260820.md](../data/reports/pinbu_rodavg_true_20260820.md) | 08-20 | 결과 | 5기 재실행 실측 봉평균(HZ 가중) — M2 종결 | 봉평균 76.2–77.4 vs 노드 83.0–84.2, 비 **1.0878±0.0030**(DB 대리계수와 일치). 판정 역전 구조 노출 |
| [pinbu_definition_20260820.md](../data/reports/pinbu_definition_20260820.md) | 08-20 | 판정 | **핀 연소도 관측량·한계치 확정 정의(사용자 판정)** | **한계 80 GWd/tU, 관측량 = 핀 axial peak.** 지도에서 advisory, 납품에서 real gate |
| [f113_pin_prereg_20260820.md](../data/reports/f113_pin_prereg_20260820.md) | 08-20 | 사전등록 | 핀 게이트 활성 상태의 `N1_N2`/f113 재캠페인 | "F_r 기록은 유효하나 납품 가능한 적이 없었다"를 전제로 등록 |
| [f113_pin_results_20260820.md](../data/reports/f113_pin_results_20260820.md) | 08-20 | 결과 | 재캠페인 채점 + 보고층 갭 폭로 | **캠페인 `best_overall`은 신뢰 불가**(예측 핀 86.05, 실패 분지의 Hamming 15 이웃). 유일 후보 1.5074 |
| [pinbu_wave_f113pin5_prereg_20260820.md](../data/reports/pinbu_wave_f113pin5_prereg_20260820.md) | 08-20 | 사전등록 | 위 결과가 등록만 하고 실행하지 않은 5기 실측 점검 | 예측값은 **재계산하지 않고 인용**; 실행 결과 **0 PASS** |

## 8. 인프라 · 잡음 (2건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [inference_backend.md](../data/reports/inference_backend.md) | 07-21 | 메모 | 캠페인 스크리닝 추론 백엔드를 덱에서 선택(local_cpu / remote_gpu) | 원격 3.5×@20k, 결과 일치 \|Δf_r\| ≤ 1.4e-3; **서버 불가 시 로컬 폴백**(부분 점수 금지) |
| [flatness_noise_measured_20260726.md](../data/reports/flatness_noise_measured_20260726.md) | 07-26 | 결과 | 평탄도 두 축 라벨 잡음 상한 실측(MASTER 추가 콜 0) | σ 0.00556 / 0.000507, 상한 ρ ≥ 0.9997. **★게이트 슬라이스에 두 축 라벨이 0개** |

## 9. 참고 자산 · 저장소 안내 (3건)

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [data/README.md](../data/README.md) | 07-16 | 메모 | `data/` 디렉터리 레이아웃 안내(스토어·ledger·splits·reports) | **`data/`는 버전관리 대상이 아니다** — 이 README가 `.gitkeep` 역할 |
| [data/reference/kngr_18m_cy01_08/README.md](../data/reference/kngr_18m_cy01_08/README.md) | 08-17 (추정) | 메모 | KNGR 18개월 다주기 계산서(cy1–cy8) PDF 추출 산출물 안내 | cy1–cy8 셔플 맵 CSV + 사이클 요약 — **전이주기 엔지니어(장기 목표)의 참조 자산** |
| [data/reference/kngr_18m_cy01_08/methodology_notes.md](../data/reference/kngr_18m_cy01_08/methodology_notes.md) | 08-17 (추정) | 메모 | 같은 계산서의 방법론 발췌(코드·설계요건·결과표 인덱스) | 엔진은 ROCS 3-D 노달; 설계요건 Fxy 1.55 / 468 EFPD / MTC(HFP) < 0 |

## 10. `F_xy` 목적축 전환 (20건, 2026-08-29 ~ 08-31)

2026-08-29 사용자 확정으로 목적함수가 `F_r` → **`F_xy`**(한계 1.65)로 바뀐 뒤 생산된 문서다.
설계 → 데이터층 → 헤드 arm 3회 → 캠페인 4건 → 결함 포렌식 1건의 순서로 읽으면 시간순이 된다.

**설계 · 데이터층 · 안전**

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [fxy_switch_design_20260829.md](../data/reports/fxy_switch_design_20260829.md) | 08-29 | 설계 | 목적축 전환 설계서 — `F_xy` 정의·출처(`MAS_OUT` FXYP)·`F_r` 관계 실측·소급라벨 물량·P1~P5 실행단계 | `F_xy/F_r = 1.0694 ± 0.0181`이나 **게이트는 대체 불가**(F_r 통과 52 중 18이 F_xy 탈락). **`node_peak ≠ F_xy` 코드 오기 정정**. 조인 키는 `Pattern.digest16`(`record_id[:16]` 일치 0건) |
| [phaseA2_ood_conformal_20260829.md](../data/reports/phaseA2_ood_conformal_20260829.md) | 08-29 | 결과 | 검토 P0-04 응답 — OOD 정책 3종 + conformal 경성 게이트를 랭킹·풀·`delivery.json`에 연결 | `ood_guard.py`/`conformal.py` **무편집**. **기본값에서 실드가 실행조차 되지 않아 웨이브 바이트 동일**(테스트 29건이 단언). `f_xy`는 conformal 밖 |
| [servefix_calibration_refit_20260829.md](../data/reports/servefix_calibration_refit_20260829.md) | 08-29 | 감사 | ★서빙 경로 피처화 결함(`library_provenance` 유물)과 per-cell 보정 **18/18 재적합** | `g_sym_class` 하나가 raw F_r bias −0.0719의 **사실상 전부**. 재적합 후 `s1i` F_r MAE 0.1026 → **0.0499**, cyclen 2.40 → **1.62**, CBC 16.61 → **12.48**. **등록 규칙 R1: 코드와 보정은 한 세트로 이동** |

**`f_xy` 헤드 — 사전등록과 3 arm**

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [fxy_head_prereg_20260829_DRAFT.md](../data/reports/fxy_head_prereg_20260829_DRAFT.md) | 08-29 | 사전등록 | 헤드 사전등록 초안 | 초안 — 아무것도 결정하지 않는다 |
| [fxy_head_prereg_20260829.md](../data/reports/fxy_head_prereg_20260829.md) | 08-29 | 사전등록 | G1~G4 + Amendment A(스플릿 S1j·바 재유도) / B(서빙경로 야드스틱) / C(arm 3 채점 경로) | **G2 MAE < 0.0463 · G3 ρ̄ ≥ 0.8944**(측정 F_r 위) → arm 2부터 **G2′ < 0.0767 AND G3′ > 0.7263**(서빙 프록시) 구속. A.5/A.6은 **폐기 아님, 병기** |
| [fxy_head_results_20260829.md](../data/reports/fxy_head_results_20260829.md) | 08-29 | 결과 | arm 1 — 합성 prior 잔차 헤드 | **G2/G3/G4 전부 FAIL**(0.105 / 0.742 / 0.99). 원인 3건 특정. MAE 분해로 **"측정 F_r → 예측 F_r 대체"가 오차의 전부**임을 확립 |
| [fxy_head_results_arm2_20260829.md](../data/reports/fxy_head_results_arm2_20260829.md) | 08-29 | 결과 | arm 2 — 원인 4건 수정(select-weight·warmup·보정·prior-on-predicted) | **G2′ FAIL 0.1086** / G3′ PASS 0.768 / **G4 PASS 0.634**(σ 보정 최초 성공). 잔여 편향의 **93%**가 서빙 경로 결함으로 설명됨 → §8 발견의 출발점 |
| [fxy_head_results_arm3_20260829.md](../data/reports/fxy_head_results_arm3_20260829.md) | 08-30 | 결과 | arm 3 — `--fxy-direct`, **`s1j` 승격 판정** | **G1·G2′(0.0663)·G3′(0.790) PASS · G4 FAIL(0.831)** → **승격 + `f_xy` σ 서빙 금지**. arm 2c와 **통계적 동률**(BCa CI가 0 포함) — "direct가 낫다"를 주장하지 않는다 |

**캠페인 — 사전등록과 결과**

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [produce_fxyera_r1_prereg_20260829.md](../data/reports/produce_fxyera_r1_prereg_20260829.md) | 08-29 | 사전등록 | `F_xy` 시대 첫 생산 라운드 — 19 strata × 40, `MAS_OUT` 보존 = 라벨 그 자체 | 설계 단계 실측이 이미 벽을 예고: 541 보존 디렉터리에서 `F_xy ≤ 1.65` **0기**, 최소 1.6502 → 벽은 대략 `F_r ≈ 1.549` |
| [pinbu_wave_fxyera_r1_prereg_20260830.md](../data/reports/pinbu_wave_fxyera_r1_prereg_20260830.md) | 08-30 | 사전등록 | phase-2 40체인 — 핀BU 실측 + `F_xy` 결정성 QC | M1 핀 ≥80% · M2 \|ΔF_xy\| ≤ 0.002 · M3 provenance |
| [pinbu_wave_fxyera_r1_results_20260830.md](../data/reports/pinbu_wave_fxyera_r1_results_20260830.md) | 08-30 | 결과 | 같은 웨이브의 판정 | **M2 `ΔF_xy = 0.000000` 40/40 → `F_xy` 완전 결정론적**. **M1 기각 14/20**(feed 109 **0/5** vs feed 121 5/5 — 핀 헤드 bias가 feed로 갈린다). 처분: **바인딩 제약은 핀이 아니라 `F_xy`** |
| [minfxy_T6T4_f121_r1_prereg_20260829.md](../data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md) | 08-29 | 사전등록 | 첫 `min_fxy` 캠페인 — PRIMARY ≤1.5441 / STRETCH ≤1.5295 / NULL 귀속 사전 고정 | `F_r` 추격이 `F_xy`를 악화시킨 이력(r3 1.5491 → r6 1.5829), 프록시는 셀내 기울기 0.48로 무력 → **헤드 필요** |
| [minfxy_T6T4_f121_r1_results_20260830.md](../data/reports/minfxy_T6T4_f121_r1_results_20260830.md) | 08-30 | 결과 | 100/100 콜, 13웨이브 | **PRIMARY 달성 F_xy 1.5322 @ 콜 57**, STRETCH 미달(+0.0027). **CBC 패리티 FAIL 3.572 · `f_xy` 헤드 패리티 FAIL 0.0543**. 프런티어가 움직인 콜 **4회**, 이후 43% 무개선. **헤드의 랭킹 우위 미확인**. D1~D6 편차 기록(**D3 = 재개 시 σ-bar 소실**) |
| [pinbu_wave_minfxy_r1_prereg_20260830.md](../data/reports/pinbu_wave_minfxy_r1_prereg_20260830.md) | 08-30 | 사전등록 | 25체인 — r1 top-20 핀 실측 + `F_r` 기록군 5기의 `F_xy` 첫 측정 | 전 타깃 native restart(promoted 드리프트 구조적 부재), M1~M8 |
| [pinbu_wave_minfxy_r1_results_20260830.md](../data/reports/pinbu_wave_minfxy_r1_results_20260830.md) | 08-30 | 결과 | ★**프로그램 최초의 `is_deliverable = True` 노심** | `bf3a70b2`: F_xy 1.5322 · **핀 63.760 실측** · 전 축 한계 내. 스토어 deliverable **0 → 25행**. **M4가 헤드라인 정정** — `F_r` 기록 코어의 `F_xy` = **1.5402** → r1 실질 이득 **0.008**. **모델-프리 비율(MAE 0.008)이 헤드(0.042)를 5배** 이김 |
| [intervention_wave_r1_prereg_20260829.md](../data/reports/intervention_wave_r1_prereg_20260829.md) | 08-29 | 사전등록 | Campaign A — Causal Move Atlas. 5셀 × 20부모 × 8무브 = 800 유료 체인 | 외부 검토 §7.2 D1이 요구한 **의사결정 데이터**의 첫 층. H1~H4 + P1~P6 |
| [intervention_wave_r1_results_20260830.md](../data/reports/intervention_wave_r1_results_20260830.md) | 08-30 (+Amendment 1·2, 08-31) | 결과 | 판정 + HGD569 오염 정정 + v2 재실행 재평가 | **H1 cell-conditional 강등**(부호 5/5, 유의 3/5) · **H1b 08-15 크기 3.37배 과대추정 철회** · **H3 성립 — 전달계수 1.23–1.42(농축·반경) vs 0.55–0.73(Gd·격자)** · **H4 Gd/격자 기술자 결손 확립**. Amendment 2에서 공통모드 **+35.05 → +0.08 EFPD**, 축퇴 **20/20 → 0/19** |

**결함 포렌식**

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [hgd569_degeneracy_memo_20260830.md](../data/reports/hgd569_degeneracy_memo_20260830.md) | 08-30 | 감사 | ★`WaveVerifier` resolver 미배선 → alias no-op → MASTER가 batch id를 앞 2자로 흡수 | 폭발반경은 **`intervention_HGD569_f125` 160행뿐**. `T6_T4`가 무사했던 것은 **버그의 반증이 아니다**. R1~R6 권고 — **`%LPD_SHF` 로스터 검사 한 줄이 160체인을 첫 체인에서 멈췄을 것** |

**정책 · 종결**

| 파일 | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| [policy_v2_serving_ab_prereg_20260829_DRAFT.md](../data/reports/policy_v2_serving_ab_prereg_20260829_DRAFT.md) | 08-29 | 사전등록 | 검토 P0-01 응답 — v2 서빙 A/B(arm A `off` / B `v1` / C `v2`) | **`_DRAFT` = 미등록.** τ를 arm별로 다르게 준다(0.25 vs **0.08** — spread 0.573 vs 0.189). Stage 0 `shadow_v2`는 **0콜**. 비용 576콜이 유일한 미결. Addendum: 발사 전 scorer provenance 결함 수정(paramA max \|ΔP\| 0.087 → 2.2e-04) |
| [tripletype_f125_r2_results_20260829.md](../data/reports/tripletype_f125_r2_results_20260829.md) | 08-29 | 결과 | 3종 계단화 라운드 2 — 08-20 사전등록의 결과 문서 | **NULL 발화**: 클린 프런티어 1.5999에서 정지, raw는 1.5864까지 가지만 그 사이 8계단 전부 CBC 위반 → **3종 `F_r`-only 라인 폐쇄**, 라운드 3 등록상 금지. 웨이브 미세조정 게이트 **8/8 거부** |

## 11. 캠페인 산출 리포트 (저장소 미포함, 10건)

초기 `data/campaigns/` 캠페인이 자동 생성한 리포트다.
**캠페인 아티팩트는 용량 때문에 이 공개 저장소에 포함되지 않는다** — 경로만 기록한다.

| 파일 (원본 경로) | 날짜 | 유형 | 한 줄 요약 | 핵심 판정 |
|---|---|---|---|---|
| `data/campaigns/fuelcost_round1c/alsearch_E1_E2_f117_minFE/report.md` | 07-22 | 결과 | `E1_E2`/f117, 예산 40/40, 목표 625 EFPD ±2에서 Max CBC 최소화 | **검증 feasible LP 0/40** |
| `data/campaigns/fuelcost_round1c/alsearch_E1_E2_f121_minFE/report.md` | 07-22 | 결과 | `E1_E2`/f121, 예산 40/40 | **0/40** |
| `data/campaigns/fuelcost_round1c/alsearch_E1_E2_f125_minFE/report.md` | 07-22 | 결과 | `E1_E2`/f125, 예산 32/32 | **0/32** |
| `data/campaigns/fuelcost_round1c/alsearch_J1_J2_f117_minFE/report.md` | 07-22 | 결과 | `J1_J2`/f117, 예산 40/40 | **0/40** |
| `data/campaigns/fuelcost_round1c/alsearch_J1_J2_f121_minFE/report.md` | 07-22 | 결과 | `J1_J2`/f121, 예산 36/36 | **0/36** |
| `data/campaigns/fuelcost_round1c/alsearch_K1_K2_f117_minFE/report.md` | 07-22 | 결과 | `K1_K2`/f117, 예산 40/40 | **0/40** |
| `data/campaigns/fuelcost_round1c/alsearch_K1_K2_f121_minFE/report.md` | 07-22 | 결과 | `K1_K2`/f121, 예산 32/32 | **0/32** |
| `data/campaigns/fuelcost_round1c/alsearch_L1_L2_f117_minFE/report.md` | 07-22 | 결과 | `L1_L2`/f117, 예산 32/32 | **0/32** |
| `data/campaigns/live_5.5-5.75_f141_maxEFPD/report.md` | (07월) | 결과 | `H3_H4`/f141 고농축·고feed 셀, 예산 56/100 | **0/56** — 이 밴드는 목표 창 밖 |
| `data/campaigns/live_5.5-5.75_f141_minFr/report.md` | (07월) | 결과 | `H3_H4`/f141, 예산 100/100 | **0/100** |

> 이 8+2 캠페인의 **feasible 0건**이 [parity_round1c_20260722.md](../data/reports/parity_round1c_20260722.md)의
> 분석 대상이며, "F_r 경계 순위스킬 결여"라는 프로그램 초기의 핵심 진단을 낳았다.

---

## 2026-09-03 업데이트

이 창에 추가된 최상위 리포트: `pinbu_wave_minfxy_r2_results_20260903.md`,
`fxy_era_adversarial_verification_20260831.md`(+ Amendment E in `fxy_head_prereg_20260829.md`),
`policy_v31_prereg_20260831_DRAFT.md`, `policy_v31_results_from_oof_20260903.md`,
`fr135_feasibility_scoping_20260903.md`, `sample_efficiency_kpi_20260903.md`(+png),
`sample_efficiency_control_prereg_20260903_DRAFT.md`, `active_frontier_loop_spec_20260903.md`,
`assembly_on_demand_design_v2_20260903.md`, `assembly_on_demand_tasks_20260903.md`,
`assembly_slice_Z_prereg_20260903_DRAFT.md`, `slice_Z_pipeline_runbook_20260903.md`,
`assembly_on_demand_r1_implementation_20260903.md`, `assembly_sigma_chain_retrodiction_20260903.md`,
`sliceZ_s6_diagnosis_20260903.md`, `bd_backfill_plan_20260903.md`, 그리고
`mesh_v4_20260903/`(README·README_figures·대표 figure 4장·`fr_vs_db/` 비교 세트).
