# 적대적 검토 리포트 (Adversarial Review) — 5_RL LP 최적화 학습 파이프라인

- 작성 시각: 2026-07-19
- 대상: `5_RL` (LP 최적화 학습·탐색 스택 — MASTER 프로덕션 2기 가동 중: 로컬 `5-5.25_f125`, 원격 PC2)
- 검토 방식: 5개 렌즈(integrity / physics / leakage / model / search) 적대적 공격 → 독립 재현 검증(verify) → 트리아지
- 산출 경로: `data/reports/adversarial_review_20260719.md`
- 실행 무결성: `data/` 하위(records.parquet, state.json, 원장, 모델, parquet)와 실행 중 프로세스는 **읽기 전용**. 허용된 단일 파일 쓰기만 수행.

> **입력 전달 무결성 경고 (F-0의 잔존).** 작성 단계가 수신한 페이로드는 `JSON.stringify(confirmed).slice(0,20000)`로 잘려, **CONFIRMED/PARTIAL 8건 중 3건만 전문 전달**되었습니다(각 항목이 verify 블록 포함 ~6.5 KB이라 3건에서 20 KB 소진). 또한 MINOR 목록도 `slice(0,3000)`으로 **24건 중 약 17건 제목만** 도착했습니다. 본 리포트는 **전달된 3건의 확정(부분확정) 발견을 전문 근거 그대로** 기록하며, **전달되지 않은 5건은 날조하지 않습니다**(§6 참조). 이는 이전 실행에서 완전 차단(0건 수신)으로 기록된 **F-0 오케스트레이터 보간 결함의 부분 잔존**이며, 트리아지를 파일로 영속화 후 작성기가 파일을 읽는 구조(§5 장기 권고)로 전환하면 근절됩니다.

---

## 1. 종합 판정

**전체 신뢰도 등급: `양호 / GREEN (조건부·provisional)` — 확정된 major 결함 0건. 인허가 프로덕션 지속 안전.**

전달된 증거 범위에서 **원 제기(attack)가 major로 올린 3건은 독립 재현 검증에서 기술적 핵심(kernel)은 참으로 확인되었으나 인과·심각도 프레이밍이 반박되어 전부 minor로 하향**되었고, **완전 반박(REFUTED) 버킷은 0건**, minor/info 24건에는 무결성 핵심 점검(record_id·upsert·split·maps.npz)이 모두 CLEAN으로 포함됩니다. 확정된 3건은 하나의 일관된 물리적 주제 — **"Dataset-A의 대리(surrogate) burnup 라벨(≈1.08×조립체 burnup, 실측 MAS_PPI 아님)이 학습 타깃·탐색 랭킹·시그마 보정에 스며들었다"** — 로 수렴하지만, 셋 다 **인허가 수용(acceptance) 경로 바깥의 자문/스크리닝/보정 단계에 국한**됩니다. 결정적으로 (a) 커리큘럼 신규셀 게이트는 `mean_spearman`에서 `max_pin_burnup`을 자문(advisory)으로 제외하고 `discharge_burnup`은 `PROBE_TARGETS`에 없어 셀을 오탈락시키지 않으며(`curriculum.py:68-69`), (b) 최종 수용 게이트 `p_feasible`는 F_r/CBC/F_q/|ASI| 4축(surrogate col 0,1,2,4)만 읽고 pin(col 6)을 절대 소비하지 않으며(`acquisition.py:80-85`), (c) `discharge` 밴드를 켜는 `gate_discharge_target`은 기본 `None`이고 코드베이스 어디에서도 설정되지 않아 MASTER/커리큘럼 생산에서 **완전 비활성**입니다(`config.py:543`). 따라서 세 결함은 모두 **잠재적 발등찍기(latent footgun)이며 현행 생산 무결성에는 영향 없음**입니다. 다만 등급은 **조건부**입니다 — CONFIRMED/PARTIAL 8건 중 5건(모델/탐색 렌즈 추정)이 작성 페이로드에서 잘려 전달되지 않았으므로, 최종 등급 확정은 그 5건이 전달된 3건과 동일한 major→minor 하향 패턴을 따른다는 재확인을 전제로 합니다(§6).

---

## 2. 확정 발견 (심각도순)

세 건 모두 **원 제기 major → 검증 후 minor**. 공통 주제: Dataset-A surrogate burnup 라벨의 침투. 아래는 전달된 3건 전문.

| ID | 심각도(원→확정) | 렌즈 | 위치 | 요약 | 생산 영향 |
|----|----------------|------|------|------|-----------|
| L3-C-CALIB-PINSIGMA | major → **minor** | leakage | `calibrate.py:141`; `model_api.py`; `acquisition.py`(col 6); champion `calibration.json` | pin σ 등온보정이 학습에서 마스킹된 Dataset-A 대리 잔차로 96% 적합 → P 대비 ~1.75× 과대·평탄한 σ | 없음(소비처 dormant: `pin_bu_limit` 기본 None) |
| L2-01 | major → **minor** | physics | `dataset_torch.py:52`; `curriculum.py:69`; `acquisition.py:867,891-903` | `discharge_burnup`이 A-only(feed-121) 학습 타깃인데 게이트 검증 없이 탐색의 TIER-2 지배 랭킹항으로 배선 | 없음(`gate_discharge_target` 항상 미설정, 커리큘럼 OFF) |
| L2-02 | major → **minor** | physics | `dataset_torch.py:52`; `curriculum.py:68`; `records.parquet` | 학습된 두 burnup축은 각각 단일 데이터셋 지지, 유일한 A+B+P 전지지축 `max_assembly_burnup`은 미학습 | 없음(하드 인허가 게이트 의존 없음) — 역량 공백 |

### L3-C-CALIB-PINSIGMA — pin-burnup 보정 σ가 학습하지 않는 Dataset-A 대리 잔차로 96% 적합됨

- **증거(독립 재현 확인):**
  - **학습/보정 마스킹 불일치.** 학습은 A의 pin 라벨을 마스킹(`LPDataset` 기본 `censor_dataset_a_pin_labels=True`, `dataset_torch.py:84,123`; CLI `--censor-a-pin-labels` 기본 True; 챔피언 체인 200838/221932/002850 전부 마스킹 학습). 반면 보정은 **마스킹하지 않음** — `fit_calibration`이 val 데이터셋을 `censor_dataset_a_pin_labels=False`로 하드코딩(`calibrate.py:141`, 주석 "censoring is a training-loss concern only").
  - **구성.** `max_pin_burnup` 등온적합에 들어가는 val 6,071행 = A 5,831행(**96.0%**) + P 240행. 마스킹을 적용했다면 P 240행만 생존 → 적합의 96%가 헤드가 결코 학습하지 않는 행.
  - **스케일 불일치(val fold).** A: pin 평균 70.87 / std 1.92, pin/assembly 비 **1.0799±0.014**. P: pin 평균 86.86 / std 6.91, 비 **1.180±0.031**. 서로 다른 물리량(대리 격자 peaking factor vs 실측 MAS_PPI 3-D 첨두).
  - **골드 재적합(실제 5-멤버 챔피언 002850 로드 → S1 val 예측 → 프로덕션 `_fit_isotonic` 재적합).** ALL(A+P) 재적합 = σ_pred[3.48,20.87]→σ_cal[7.53,8.58]로 저장된 `calibration.json`[7.53,8.58]을 소수 2자리까지 재현. 분해: A-only[7.53,9.16], **P-only(헤드의 실제 물리량)[3.87,5.14]**. A-지배 프로덕션 곡선은 P행의 불확실성을 평균 σ_cal=8.43으로 매핑하나 P-적정값은 4.83 → **1.75× 과대**. 체인 재현: 200838[1.21,10.77], 221932[6.99,9.69].
  - **중요 단서.** Pearson corr(σ_pred, |err|)≈0 (A·P·ALL 모두) → 모델의 pin σ는 애초에 무정보(uninformative). 누수는 σ를 *평탄하게* 만든 원인이 아니라 **잘못된 상수 레벨(P 기준 ~1.75× 과대)로 고정**했을 뿐.
  - **소비처 추적.** `p_feasible`는 col 0,1,2,4만 읽고 pin col 6 미소비(`acquisition.py:80-85`). 벤더 `target_cycle` 보상은 pin을 게이트할 수 있으나(`reward.py:291`) `make_constraints`가 `max_pin_burnup_limit`을 설정하지 않아 1e9 > DISABLED 1e8 = 비활성. **유일 소비처는 `score_user_criteria`** — `_CRITERIA_GATED_AXES=(6,"pin_bu_limit",1.0)`, UCB shift `μ+0.25·σ`(`acquisition.py:909-921`), **`pin_bu_limit`이 설정될 때만 활성**. 현재 기본 None/report-only(`user_criteria_ref.inp`에 `# pin_bu_limit = 62.0` 주석) → **dormant, 인허가 pin 한계 설정 즉시 무장**.
- **왜 minor인가(반박 요소):** "leakage"는 부정확한 명명 — val fold는 학습에서 held-out이라 어떤 **보고 지표도 오염·팽창시키지 않음**(train/test contamination 아님). 실제 결함은 **학습/보정 마스킹 불일치로 인한 분포 불일치(A-대리-스케일) pin 보정**. 주석 "censoring is a training-loss concern only"는 A·P가 동일 물리량인 6개 타깃에 대해서는 옳고, A가 P와 **다른 물리량**인 `max_pin_burnup`에 대해서만 틀림.
- **영향:** 기술적 결함은 확정. 실사용 영향은 **없음(dormant)** — 유일 소비처가 기본 비활성이고, 그마저 pin σ가 무정보라 UCB shift는 잘못된 상수(P 기준 ~1.75× 과대)만 주입. 인허가 pin 한계를 설정하는 순간(pin 헤드가 만들어진 바로 그 용도)에 처음으로 실제 영향 발생.
- **최소 수정안(전달된 fix_proposal은 truncate됨 — 증거에 근거해 도출):** 보정의 pin 등온적합을 학습과 일치시킬 것 — `fit_calibration`에서 `max_pin_burnup` 축에 한해 Dataset-A pin 라벨을 마스킹(또는 fidelity-consistent P 행만으로 pin 등온 적합)하여, σ를 헤드가 실제 예측하는 물리량 스케일로 보정. pin σ가 무정보이고 소비처가 dormant이므로 이는 **긴급 수정이 아니라 발등찍기 방지**. MASTER 가동 중 data/model/process 무변경.

### L2-01 — `discharge_burnup`: 게이트 검증 없는 A-only 학습 타깃이 탐색의 지배 랭킹항으로 배선

- **증거(독립 재현 확인):** `discharge_burnup ∈ TARGETS = True`(`dataset_torch.py:52`). 수렴행 non-null **A=38,851 / B=0 / P=0** → "학습 가능" discharge 질량의 94.8%가 A, B/P는 정확히 0. A 지지: feed **121 only**, 라이브러리 {260624:29,976; 5.8_5.1:8,244; legacy_a:631}, e_core 5.35–5.64. 모든 커리큘럼 셀은 feed {109,117,125}·라이브러리 ga80·e_core 5.0–5.5 → **feed·라이브러리 모두 disjoint**. `max_pin_burnup`과 달리 discharge는 **비검열(uncensored)**. `PROBE_TARGETS`에 없고(`curriculum.py:69`, 주석 "not on the FOM, probe-predicted only") `NOREG_TARGETS`(cyclen,f_r만)에도 없어 **어떤 커리큘럼 게이트도 이를 MASTER 실측으로 채점하지 않음**. 그러나 `acquisition.py`는 이 예측을 **TIER-2 지배 랭킹항**으로 배선(docstring `:867` "Hierarchy (dominant first): (1) cyclen band, (2) discharge_burnup band"; `:891-903`): `spec.target_discharge_burnup` 설정 시 밴드 위반이 `_CRITERIA_BAND_TIER=1e8 ≫ CONSTRAINT 1e4 ≫ F_r ~1`로 within-band 항을 전부 지배. A에서 corr(discharge,cyclen)=**0.947**, 비=0.078(CV 0.67%) → 헤드는 사실상 cyclen을 재유도.
- **왜 minor인가(반박 요소, 3중):**
  1. **pin-BU 쌍둥이 아님.** pin-BU는 라벨 *날조*(A의 1.08×조립체 peaking 대리) + *교차 데이터셋 정의 불일치*(A 1.080 vs 실측 P 1.181) 때문에 검열됨. discharge는 **둘 다 없음** — A값은 진짜 감손 출력(`extract_a.py:350`, cyclen 구동)이고, 벤더 FOM(`domain.py:268`)에 discharge 축이 없어 B/P=0은 **구조적**(상충 라벨 아님). 공격 제목 자체가 "no cross-dataset labels"를 인정 — 그것이 pin-BU를 버그로, 이를 비버그로 만든 요인.
  2. **레짐 주장 역전.** `gate_discharge_target` 기본 None(`config.py:543`)이며 **어디에서도 미설정**(전 run grep clean) → 커리큘럼/MASTER(feed 109/117/125)에서 밴드 **OFF**. ON은 user_criteria 탐색뿐이고, 최근 3개 라이브 캠페인(`20260718_025717/033909/053537`)은 **feed 121 = A 자신의 feed**·ga80·e_core 5.2에서 구동 — 공격이 주장한 feed-109/117/125가 아님. feed는 A와 일치(in-distribution), ga80 라이브러리만 OOS.
  3. **두 완화 장치 누락.** (a) 최종 탐색 타당성은 **검증된 cyclen에서 유도한 독립 에너지밸런스 discharge 추정**(`campaign.py:1450-1462`, `estimate_discharge_burnup=P·cyclen/HM·(241/feed)`)을 사용하며 annotation-only('NOT a hard gate'); A-헤드는 스크리닝/랭킹에만 영향, 수용에는 절대 미관여. (b) A의 discharge/cyclen 비 0.078·CV 0.67% → discharge는 cyclen의 준-결정론적 선형함수라 헤드는 검증된 축(cyclen, within-case spearman 0.65–0.95)을 재유도, 0.947 상관은 무해한 물리.
- **영향:** (i) 모델 — discharge에 held-out 지표 없음(`model_report` split_metrics는 ao_abs·cbc_max·cyclen·f_q·f_r만 검증), 단 discharge=const·cyclen이고 cyclen은 검증되어 유계. (ii) 탐색 — 밴드는 스크리닝에만·opt-in(기본 off)·feed-121(in-feed)에서만 활성, 수용은 독립 에너지밸런스+검증 cyclen 사용. (iii) 커리큘럼/MASTER — **영향 0**(`gate_discharge_target` 미설정, PROBE/NOREG에 부재).
- **최소 수정안:** (1) `model_report` split_metrics에 `discharge_burnup`(+`max_pin_burnup`) 추가해 held-out-A within_case_spearman/R2 추적(현재 무언 생략). (2) `score_user_criteria`에서 discharge 밴드를 A-only 헤드 대신 **수용 단계가 이미 신뢰하는 동일 에너지밸런스 추정**(`estimate_discharge_burnup(pred_cyclen,...)`)으로 구동 → OOS ga80 A-헤드를 1e8 지배 tier에서 제거, 검증된 cyclen축과 일관화. (3) 가드: `gate_discharge_target`이 feed≠121 또는 라이브러리∉{260624,5.8_5.1,legacy_a}에서 설정되면 경고/단언(현재 미설정이므로 latent-footgun 가드).

### L2-02 — burnup 타깃 선택이 교차 데이터셋 커버리지와 역전: 학습된 두 축은 단일 데이터셋 지지, 전지지축은 미학습

- **증거(독립 재현 확인, records.parquet 41,680행 수렴):** `max_assembly_burnup` — **in_TARGETS=False**, A=38,851/B=72/P=1,623, feed 그리드 P=[97,105,109,113,117,121,125] **7개** = 모든 burnup축 중 최광 커버리지·유일한 A+B+P 전지지. `discharge_burnup` — in_TARGETS=True, A=38,851/**B=0/P=0**, A feed 121 only. `max_pin_burnup` — in_TARGETS=True, A=38,851(검열)/B=0/P=1,260, 검열 후 **P만**(feed 109/117/125). 즉 검열 후 pin은 P 단독, discharge는 A 단독 학습 → **disjoint 데이터셋이라 두 burnup 타깃은 결코 교차 검증 불가**. 반면 `max_assembly_burnup`은 A·P 실질 지지에도 미학습(`curriculum.py:68` "max_assembly_burnup column 5 is always NaN in this model" 축자 정확); `pinbu_forensics.md`는 이 축이 "pin rank의 93%를 구동, 훨씬 잘 일반화"한다고 명시. 검열은 프로덕션에서 활성(`config.py:273` 기본 True, local/remote retrain 양쪽 스레딩).
- **왜 minor인가(반박 요소, 4중):**
  1. "poorly-supported"가 discharge를 오표기 — 38,851행은 **최대 지지**. 실제 한계는 단일-*데이터셋*/단일-feed(121)로 다른(그리고 타당한) 우려. 공격이 커버리지 폭과 표본 지지를 혼동.
  2. "INVERTED selection"은 오류를 함의하나 pin의 P-only 지지는 **의도적·문서화된 라벨 충실도 결정** — A pin은 MOCHA-cache 대리(~1.08×조립체, CV 1.2%)로 P의 실측 MAS_PPI 3-D pin(~1.18×, CV 2.7%)과 **다른 물리량**. A 검열이 **옳음**(A는 실제 평가되는 P 레짐의 나쁜 교사). pin A↔P 교차검증은 물리적으로 무의미.
  3. "cross-dataset validation"은 시스템의 **비목표** — split은 per-cell **within-cell** holdout(`curriculum.py:1257`; 포렌식은 n=30 held-out으로 검증). 파이프라인이 쓰지 않는 검증 모드를 공격.
  4. `max_assembly_burnup` 승격은 **이미 문서화된 단계적 권고**(`pinbu_forensics.md` rec #2, "first-class TARGET로 추가, pin은 residual로 매핑", 'GPU 검증 필요'로 명시 유예, "MASTER 가동 중 blind 미구현"). 미발견 결함이 아니라 알려진 백로그.
- **실제 커널(유효):** `max_assembly_burnup`은 최광 커버(3 데이터셋·7 feed)이자 물리적 우월(BOC/셔플 결정, pin-rank 93% 구동, 우수한 일반화) burnup 물리량인데 **미학습** → 모델은 잘 일반화하는 burnup 역량을 결여하고 약한 자문 pin축·단일-feed discharge축에 의존. 진짜 역량 공백/기회비용.
- **영향(유계 → minor):** (i) 모델 — 완만. pin은 known-weak(held-out within-cell Spearman ~0)이나 이미 자문 강등; discharge는 단일-feed로 user-criteria 탐색 시 OOD 리스크; max_assembly 역량은 단순 부재(기회비용). (ii) 탐색 — 낮음. `pin_bu_limit` 기본 None=report-only(`acquisition.py:747`), discharge는 소프트 tolerance-band(부재 시 중립). 누락 축에 의존하는 하드 인허가 게이트 없음. (iii) 커리큘럼 — 절연. pin은 신규셀 mean_spearman 게이트에서 제외(advisory), discharge는 PROBE_TARGETS에 부재 → 약한/누락 burnup축이 셀을 오탈락시키지 않음.
- **최소 수정안:** 알려진·의도적 유예 개선 기회이지 활성 버그 아님 — MASTER 2기 가동 중 blind 변경 금지. `pinbu_forensics.md` rec #2를 **GPU 검증 실험으로 스케줄** — `TARGETS`에 `max_assembly_burnup`을 schema-additive first-class 헤드로 추가(3 데이터셋·7 feed 저장, 우수 일반화), `max_pin_burnup`은 그 위 residual로 모델링. 채택은 측정된 held-out within-cell Spearman(전후)으로 게이트. A-pin 검열 유지(옳음). GPU 실험 검증 전 라이브 TARGETS/model/config 무편집.

---

## 3. 반박된 주장 (왜 문제가 아닌가)

완전 반박(REFUTED) 버킷은 **0건**입니다. 그러나 위 3건은 verify에서 **기술 커널은 확정, 인과·심각도 프레이밍은 반박**된 PARTIAL이므로, 반박된 하위 주장을 아래에 정리합니다(이것이 "왜 major가 아닌가"의 실체).

| 원 제기 하위 주장 | 판정 | 반박 근거(요지) |
|-------------------|------|-----------------|
| discharge_burnup은 잡힌 pin-BU 불일치의 "쌍둥이" | **반박** | A값은 진짜 감손 출력, 벤더 FOM에 discharge축 없어 B/P=0은 구조적. 라벨 날조·교차 불일치 둘 다 부재(`extract_a.py:350`, `domain.py:268`) |
| A-only 헤드가 ga80 feed-109/117/125 커리큘럼 게이트를 지배 | **반박** | `gate_discharge_target` 항상 미설정 → 커리큘럼 OFF. 밴드 ON인 라이브 캠페인은 feed-121(=A 자신, in-distribution) |
| A-only discharge 예측이 최종 수용을 오도 | **반박** | 수용/타당성은 검증된 cyclen 기반 독립 에너지밸런스 추정 사용, A-헤드는 스크리닝/랭킹 전용(`campaign.py:1450-1462`) |
| discharge_burnup은 "poorly-supported" | **반박** | 38,851행 = 전 타깃 중 최대 지지. 실제 이슈는 단일-데이터셋/단일-feed(다른 우려) |
| burnup 타깃 선택이 "오류로 역전(inverted)" | **반박** | pin P-only는 의도적 라벨-충실도 결정(A 대리 1.08× vs 실측 1.18×), A 검열은 옳음 |
| 두 burnup축이 "교차검증 불가"라 결함 | **반박** | 교차-데이터셋 검증은 시스템 비목표(per-cell within-cell holdout 사용) |
| pin σ 보정이 "leakage(누수)" | **반박** | val fold는 held-out이라 어떤 보고 지표도 팽창 안 됨. 실제는 train/calibrate 마스킹 불일치(오염 아님) |
| pin σ 누수가 σ를 평탄하게 만든 원인 | **반박** | corr(σ_pred,|err|)≈0 → pin σ는 애초 무정보. 누수는 상수 레벨만 ~1.75× 오설정 |

---

## 4. 미검증 minor / info 목록 (전달분 17/24)

verify로 심각도 하향 확정된 것이 아니라 attack 단계의 저심각·정보성 항목(재검증 미수행). **info=결함 없음 확인, minor=경미 개선점.**

**무결성(L1)**
- `L1-01` (info) record_id 무결성 CLEAN — 전체 store 재계산 0 불일치, 교차 데이터셋 오염 0.
- `L1-02` (info) Upsert 시맨틱 CLEAN — converged 강등 없음, ledger done/error가 store P와 완전 조정.
- `L1-03` (minor) PC2(pandas 3.0.3) kit parquet이 본 머신(2.3.0)에서 in-memory dtype drift 유발 — 허용되나 미강제 불변식 의존.
- `L1-04` (minor) Feature 구멍 — 유효 P0_pathfinder 행이 e_core/e_split 조건화 피처 결측.
- `L1-05` (minor) Ledger가 완전한 감사 추적 아님 — store P 226행이 현행 ledger.jsonl에 항목 없음.
- `L1-06` (info) Split fold 무결성 CLEAN but 스냅샷이 라이브 store 대비 stale, quarantine은 비포함(empty quarantined_by_cell)으로.
- `L1-07` (info) maps.npz 무결성 CLEAN — 모든 키가 유효 record_id, orphan/dangling 없음, 균일 (4,9,9) float16.

**물리(L2)**
- `L2-03` (minor) Dataset-A의 cbc_max 감독이 enrichment-truncated — 고 e_core `5.8_5.1` 라이브러리(8,244행) 전체가 boc_only·마스킹 → CBC 헤드가 LEU+ 고농축 edge에 정확히 구멍.
- `L2-04` (minor) Dataset-B의 f_r/f_q가 feasibility-selected 준-축퇴 부분표본 — F_r 제약 edge에 pin(std 0.008), P의 ga80 f_r은 1.53–4.77 span.
- `L2-05` (info) PASS한 clean 점검 — boron-letdown 불변식, duplicate-conflict, cyclen 단조성, fuel_types v4 물리 부호, surrogate 열-순서 매핑.

**누수(L3)**
- `L3-A-BLINDPROBE` (info) 7개 done 셀 blind probe 진짜 blind — probe 챔피언 학습 후 probed-셀 행 생산, store에 probe 패턴 부재.
- `L3-B-GATEHOLDOUT` (info) 게이트 no-regression holdout 행이 챔피언 체인(200838→221932→002850) 각 후보 train fold에 부재.
- `L3-D-SAFEINPUTS-V4` (info) SAFE_INPUT_FIELDS가 라벨과 disjoint 유지 — v4 피처 추가는 per-type a-priori 격자 물리, 라벨 아님.
- `L3-E-TRANSPOSE` (info) Transpose 증강이 교차-fold 누수 없음 — val 행의 대각-미러 쌍둥이가 train fold에 부재.

**모델(L4)**
- `L4-03` (minor) 앙상블 epistemic 신호가 cyclen·cbc에서 near-collapse(멤버 0.99+ 상관) → OOD 셀에서 total σ 확대 불가.
- `L4-04` (minor) v4 격자-채널 활용은 실재하나 집중 — 17개 추가 채널 중 ~6개 사실상 inert, f_r 랭킹이 harvested 블록 거의 미사용.
- `L4-05` (minor, 제목 truncate) Headline 지표는 bit-exact 재현, 그러나 published per-member best_metrics [이하 잘림].

> 나머지 약 7건(24 − 17)의 minor 제목은 `slice(0,3000)`로 미전달(§6). 렌즈 분포상 추가 L4(모델)·L5(search) 항목으로 추정.

---

## 5. 권고 조치 (우선순위)

### 즉시 (프로덕션 무변경 — 문서/추적)
1. **[프로세스] 트리아지 파일 영속화 + 작성기 파일 읽기 전환.** 인-프롬프트 `slice(0,20000)` 보간 대신 confirmed/refuted/minor 전체를 `data/reports/`에 JSON으로 먼저 기록. F-0 완전 근절(§6). 재발 시 본 리포트처럼 5/8만 전달되는 손실 방지.
2. **[추적] `model_report` split_metrics per_target에 `discharge_burnup`·`max_pin_burnup` 추가**(L2-01 fix #1). held-out 성능을 무언 생략에서 명시 추적으로 — 코드 소량, 모델/데이터 무변경.
3. **[검증] 전달 누락 5건 재확인.** CONFIRMED/PARTIAL 5건이 전달 3건과 동일한 major→minor 패턴인지 재-run으로 확인해 §1 등급을 조건부에서 확정으로 승격.

### 다음 재학습(GPU) 시 — blind 변경 금지, 검증 후 채택
4. **[L2-02] `max_assembly_burnup`을 first-class 헤드로 승격 실험**(`pinbu_forensics.md` rec #2). schema-additive, `max_pin_burnup`을 residual로. 채택은 held-out within-cell Spearman(전후)으로 게이트. A-pin 검열 유지.
5. **[L3-C] pin σ 보정을 학습과 일치**(`fit_calibration`에서 `max_pin_burnup` 축만 A 검열, 또는 P-only 적합). σ를 헤드 실제 물리량 스케일로. dormant 소비처라 발등찍기 방지 목적.
6. **[L2-03] cbc_max의 고농축 edge 구멍** — `5.8_5.1` 라이브러리 boc_only 마스킹으로 LEU+ 고 e_core에서 CBC 감독 부재. 재학습 시 해당 edge 데이터 확보/보완 검토.

### 장기 (인허가 품질 추적성)
7. **[L2-01] discharge 밴드를 A-only 헤드 → 에너지밸런스 추정으로 재배선**(수용 단계와 일관화) + `gate_discharge_target`이 feed≠121/OOS 라이브러리에서 설정 시 단언 가드.
8. **[프로비넌스] 각 적대적 리포트에 (a) 입력 트리아지 JSON 경로+해시, (b) 검증 팬아웃 run ID, (c) 재현 커맨드 헤더 부착.** 인허가 등급 QA 추적을 기계 검증 가능하게.
9. **[회귀] 오케스트레이터→작성기 계약 스모크 테스트** — 더미 발견 3~4건으로 리포트 정확 반영·비-truncation 확인(보간/절단 회귀 방지).

---

## 6. 부록 — 입력 전달 무결성 상세 (F-0 잔존)

- **전달 사실:** 작성 페이로드는 `CONFIRMED/PARTIAL findings (8): <JSON.stringify(confirmed).slice(0,20000)>`. 각 확정 항목이 verify 블록 포함 ~6.5 KB이라 **3건에서 20 KB 소진 → L2-01, L2-02, L3-C-CALIB-PINSIGMA 3건만 전문 도착, 나머지 5건 미도착**. MINOR는 `slice(0,3000)`로 **24건 중 ~17건 제목만 도착**. REFUTED=0(전달됨).
- **복구 시도(읽기 전용):** 저장소·스크래치패드·`tasks/*.output` 전수 검색. `tasks/wimr9q931.output`은 **이전(F-0 완전차단) 실행의 워크플로 출력**으로, 렌즈별 attack 프리뷰의 *첫 발견만*(L1-01, L2-01, L3-C, L4-01) 보존 — 누락 5건의 verify 전문 미포함. 별도 트리아지 JSON 파일 부재 확인. **누락 5건은 복구 불가이며 날조하지 않음.**
- **참고(미검증·발견 아님):** 이전 run 프리뷰에서 모델 렌즈 `L4-01` "Calibrated sigma is systematically over-confident on the OOD curriculum cells (worst: cyclen), because isotoni…" 제목 흔적 확인. 이는 L4-03(앙상블 σ near-collapse)과 σ-보정 주제가 겹치므로, **누락된 CONFIRMED/PARTIAL 5건에 모델(L4)·탐색(L5) 렌즈 항목이 포함될 개연성**을 가리킴. 재-run 시 이 축(OOD σ 과신) 우선 확인 권고.
- **코드 근거 재확인(본 작성 중 읽기):** `dataset_torch.py:52`(TARGETS에 두 burnup축), `:84/:123`(A pin만 검열), `config.py:273`(검열 기본 True)·`:539`(gate_advisory=[max_pin_burnup])·`:543`(gate_discharge_target=None), `curriculum.py:68-69`(max_assembly 상시 NaN·discharge probe-only)·`:190-195`(NOREG=cyclen,f_r), `acquisition.py:740`(BAND_TIER=1e8)·`:748-753`(pin col 6 gated)·`:867/:891-903`(discharge 지배 계층)·`:80-85`(p_feasible col 0,1,2,4), `calibrate.py:141`(보정 censor=False). 전달 3건의 인용은 전부 라이브 코드와 일치.

### 실행 무결성
- `data/` 하위(records.parquet·state.json·원장·모델·parquet)·실행 프로세스 **읽기 전용, 무변경.** 허용된 단일 파일 쓰기(`adversarial_review_20260719.md`)만 수행. 임시 파일은 스크래치패드 한정.
