# 집합체 온디맨드 설계 (assembly-on-demand) — 아키텍처 · 단계 계획 **v2**

**지시일** 2026-09-02 · **작성** 2026-09-03 · **상태** 설계안 v2 (구현 0 · DeCART 0 · MASTER 0)
**선행** `assembly_on_demand_design_20260902.md` (v1) · **성격** v1의 **정정 후계본**
**수신** 프로그램 오너 · 비평 2건(physics-and-licensing-realism / engineering-feasibility-and-data-flow) 반영

**사용자 지시 (축어, 2026-09-02)**

> "DeCART2D surrogate와 연계하여 AI 모델 자체적으로 필요한 집합체가 있으면 후보 추출 후
> DeCART2D 실계산으로 HGC 생성해서 사용하도록 해"

메모리 노드: `lpopt-assembly-on-demand`. 관련 계보: `lpopt-round10-state`, `lpopt-north-star`,
`lpopt-acceptance-bar`.

**v2가 v1과 다른 점 한 줄.** v1의 §3 트리거는 **모델-프리 사슬의 오차를 13배 과소평가**했고
(0.008은 *측정* `F_r` 입력에서만 성립), 인허가 게이트 중 **SDM/MTC를 통째로 빠뜨렸으며**,
빌드 불가능한 `n_gd`를 열었고, 패키지 재생성 단계를 누락했다. v2는 이 네 가지를 포함해
비평이 세운 사실오류를 **인용된 파일:라인에서 하나씩 재확인한 뒤** 고쳤고,
**비평이 틀린 4건은 근거와 함께 반려**했다(부록 D). 전문 변경 로그는 부록 D.

**이 문서의 모든 "있다/없다"는 소스를 읽고 확인했다.** 로컬 연산 금지 규칙에 따라 파케이·모델을
열지 않았으므로, 행 수·커버리지 수치는 코드 상수 또는 기존 리포트 인용이며 출처를 매번 밝혔다.
**v2에서 새로 바이트 단위로 확인한 것**은 §0.6 하단과 부록 A에 ★로 표시했다.

---

## 0. 요약 (한 페이지)

### 0.1 판정 네 줄

1. **폐루프의 6단계 중 5단계는 이미 코드로 존재한다.** 설계 스펙(`design/spec.py`) → DeCART 덱
   생성(`design/lattice.py`) → 별칭·매니페스트(`DesignRegistry`) → TotalBatcher 라이브러리
   빌드(`design/library.py`) → fuel_types 수확(`data/fuel_types.py`) → MASTER 부트스트랩
   (`design/bootstrap.py`) 은 `curriculum._generate_band_designs`(curriculum.py:1504)가 이미
   자동으로 돌린다. 없는 것은 **첫 단계, 즉 "AI가 필요한 집합체를 판단하는" 트리거뿐이다.**
   lpopt 어디에도 `FuelDesign` 축 위의 목적함수 항은 없다 (§3.1).

2. **그러나 5단계가 그대로 있어도 이 축은 값이 없다.** 프로덕션 체인
   (`lattice.edit_dec_text`, lattice.py:114)은 **숫자 3개(UO2 92235, UO2_2 92235, UO2G 6408)와
   CASEID만** 고치고 **핀맵·밀도줄을 절대 건드리지 못한다**(소스 재확인, lattice.py:114-147).
   OPSCREEN §8은 **동결 레이아웃으로 만들 수 있는 최선(FF 1.1657)이 incumbent ga80 E4(1.1390)보다
   나쁘다**고 판정했다. 문턱을 넘는 것은 **개방 20핀 레이아웃 `1:1;4:1;6:4`** 하나다.
   → **핀맵 저작 없는 온디맨드 설계는 사전에 실패가 보장되어 있다.**
   저작 코드는 `realize_lat1600.author_template`(realize_lat1600.py:180)에 **일회성 스크립트로**
   존재하며, 산출물은 `0_APR1400`이 아니라 **별도 트리 `5_RL/templates_lat1600/`** 로 간다
   (★ realize_lat1600.log:2-30 로 확인). **작업 #1은 이것을 `design/lattice.py`의 1급 가드
   함수로 승격하는 것이다.**

3. **★ 트리거는 "예측"이 아니라 "사전등록된 소급검증 바(bar)"여야 한다 — v1의 가장 큰 오류.**
   v1은 모델-프리 사슬의 오차를 **0.008**로 적고 `ΔF_xy ≥ 0.020` 을 발사 조건으로 삼았다.
   재확인 결과 **0.008은 *측정* `F_r`를 입력했을 때의 단위환산 오차**이고
   (`pinbu_wave_minfxy_r1_results_20260830.md:145-152`, 표 헤더가 `ratio (1.0640 × F_r)`),
   **예측 `F̂_r`를 넣는 — 이 설계가 필요로 하는 — 모드의 유일한 실측은 +0.0614**
   (`minfxy_E1E2_f121_r2_results_20260831.md:131` 부근, s1j 헤드 자신의 +0.0333보다 나쁨)다.
   해석적 전파도 같은 답을 준다 — `A = 1.035 ± 0.031`, `node_peak` rms 0.036 (n=15,
   OPSCREEN.md:246-250) → **σ(F_xy) ≈ 0.065**. 즉 **v1의 발사 바는 자기 잡음의 1/3**이었다.
   게다가 OPSCREEN.md:253-255 는 이 사슬을 **"floor 추정기이지 현행 패턴의 예측이 아니다"** 라고
   명시적으로 금지하고, 반례(B3 = T5_T6: 표에서 **가장 평탄한 격자** FF_hot 1.1020인데 **최악의
   측정 `F_r` 1.5795**)를 같은 절에 싣고 있다.
   → **v2의 트리거는 §3.4의 사전등록 소급검증(task #0)이 `σ_chain`을 측정한 뒤에만 정의된다.**
   신경망 헤드는 **그림자 신호(shadow)** 로만 기록하고 어떤 게이트도 잡지 않는다.

4. **★ 인허가 게이트가 하나 빠져 있었다 — MTC/SDM.** `lpopt/search/sdm_mtc.py`는 이미 이식
   완료된 **납품 전 인허가 하네스**(Decision D9, 2026-07-25)이고 APR1400 DCD Table 4.3 한계를
   들고 있다 — MTC 창 **[−54, +9] pcm/°C**, SDM `required = 10180 + 690 = 10870 pcm`,
   `post_verify_delivery`(sdm_mtc.py:1492). 모듈 독스트링이 **이 축이 만드는 위험을 그대로**
   적고 있다: *"평탄화는 제어봉 가치를 단조 저하시키고 누설을 키우는데, 어떤 탐색축도 그것을
   측정하지 않는다."* v1의 PRIMARY는 `F_xy/F_r/CBC/F_q/|AO|/pin BU`에서 멈췄다.
   → **v2는 MTC를 PRIMARY의 경성 게이트로 올린다.** 단 **SDM은 오늘 게이트가 될 수 없다** —
   `campaign._rod_model()`(campaign.py:3012-3023)이 **설계상 `None`을 반환**하며
   ("lpopt has no full-core asset package … the gate then reports SDM as INCONCLUSIVE rather than
   inventing a rod map"), 따라서 SDM은 **보고 전용(INCONCLUSIVE)** 으로 등록하고 전제조건을
   별도 작업으로 뗀다 (§7.4, 작업 #20). 이 부분은 physics 비평의 요구를 **절반만** 수용한 것이다
   (부록 D-R1).

### 0.2 아키텍처 한 줄 (v1 유지)

> **집합체 = 설계축(actuator)이 아니라 관측 기술자(descriptor)다.**
> 탐색·학습·목적함수는 전부 기술자 공간에서 돌고, 핀맵/농축도/Gd는 그 기술자를 실현하는
> **역사상(inverse map)** 일 뿐이다. v5 결정(2026-07-20, poison-agnostic 곡선형상 8채널)과
> `coreagnostic_v3_design_20260817.md`의 판정을 그대로 연장한 것이다.

### 0.3 폐루프 (6 단계, ⓪ 추가)

```
 ⓪ 소급검증        ① 트리거          ② 후보 추출          ③ 실계산        ④ 변환·등록      ⑤ 캠페인    ⑥ 판정
 사슬의 σ를        σ가 허용하면      기술자 목표 →        DeCART2D        HGC→TotalBatcher store 행   사전등록 게이트
 기존 측정행으로   k·σ 바를 정의     설계공간 역사상 →    (16 branch,     → MAS_XSL/HFF    → 재학습    PRIMARY/SECONDARY
 측정 (DeCART 0,   (그 전에는       surrogate(A) 스크리닝 334 state)      → 패키지 재생성  (cond v9)   /NULL  + MTC 게이트
 MASTER 0)         발사 없음)       → 다양성/중복제거      (§5)           → fuel_types     (§7)        (§7.4)
   (§3.4)            (§3.5)            (§4)                              (§6)
   ▲                                                                                                     │
   └───────────── 실패 시 능동학습 포인트 / σ 과대 시 트리거 폐기 → 짝지은 물리실험으로 전환 ──────────────┘
```

### 0.4 비용 — v1에서 **재유도**함 (라벨 오류 2건, 누락 3건 수정)

| 항목 | 시간 | 근거 · v1 대비 변경 |
|---|---|---|
| **⓪ 소급검증 (task #0)** | **~0.5 일, DeCART 0 / MASTER 0** | **신규.** 238에서 기존 측정행만 사용 (§3.4) |
| surrogate(A) 스크리닝 (라운드 1 격자 ≈ 3.7×10³ 설계) | **~7 s** (GPU 541 case/s) / **~7 min** (CPU 8.6 case/s) | SURROGATE_USAGE.md §2·§6. **v1의 10⁵ 설계·3분은 격자 산수 오류** (§4.3) |
| DeCART2D 2 케이스 | **직렬 3,084 s/case** (199 실측) 2-wide ≈ **1.7 h** / **omp 4-way 735–750 s/case** (box 104 실측) ≈ **0.25 h** | manifest_199.json / ★ realize_lat1600.log:31-34. **v1은 box 104 체제를 몰랐다** |
| TotalBatcher 라이브러리 재빌드 (39 타입) | **≤ 1 min (N=16에서만 관측)** | 181 16-FA 빌드 타임스탬프. **N=37/39에서는 미측정**으로 표기 |
| **패키지 재생성** (cores 10개 템플릿 + synth_decks purge) | **~수 분** | **신규.** §6.3 (v1 누락) |
| MASTER 부트스트랩 **9회** (기존 base 8 + 신규 pair 1) | **2–5 h + 재시도 꼬리** | OPSCREEN.md:388-393 **"8 existing + 1 new pair = 9 bootstraps"**. **v1은 "밴드 1개"로 오기** |
| 단일 실패 꼬리 (관측) | **8,744 s = 2.43 h** | `t3t4_rerun.log`, `MasterRunError 4294967295` |
| 전이 스윕 (`fr_transfer --k 32`) | **1–2 h** | T6T4 덱 헤더 |
| **캠페인 2 arm × 100 콜** | **200 MASTER 콜** | **신규 계상** (v1 누락). 짝지은 대조군은 선택이 아니라 필수 |
| **MTC 사후검증** (`post_verify_delivery`, top_k 5) | **≈ 10 MASTER 콜 × 20–60 s** | **신규.** sdm_mtc.py 독스트링 + config `[sdm_mtc] top_k=5`, `branch_timeout_s=300` |
| **phase-2 pin 웨이브** (측정 pin BU) | **별도 웨이브** (선례 r2 phase-2 = 30 체인) | **신규 계상.** r2 본 웨이브는 `측정 pin 0/99` |
| **수직 슬라이스 1회 총계** | **약 1.5–2 근무일** | v1의 "1 근무일"은 부트스트랩 라벨 오류 + 캠페인/pin/MTC 누락의 합 |
| 디스크 | ~36 MB/케이스 + MAS_XSL +385,849 B/FA + MAS_HFF +404,857 B/FA | 실측 (§6.4) |

### 0.5 사전등록 게이트 (요약, 전문 §7.4)

| 마크 | 기준 |
|---|---|
| **PRIMARY** | 온디맨드 타입 ≥1개를 fresh로 쓰는 MASTER 검증 노심이 **측정** `F_xy ≤ 1.5245` **이고** `F_r ≤ 1.55` ∧ `CBC ≤ 1600` ∧ `F_q ≤ 2.41` ∧ `\|AO\| ≤ 0.30` ∧ **측정** pin BU `≤ 80` ∧ **★ MTC ∈ [−54, +9] pcm/°C (`post_verify_delivery` PASS)** |
| SECONDARY | 측정 `F_xy < 1.5322` (deliverable incumbent `bf3a70b2`를 마진 없이 이김). 라벨 표기, **headline 금지** |
| **NULL** | 예산 안에서 `F_xy < 1.5295` 인 클린 노심 0건 → 집합체 축도 LP 축과 같이 바닥 → 축 폐쇄 |
| **필수 대조군** | 동일 주·동일 챔피언·동일 예산으로 incumbent 셀(E1_E2/f121)에서 **짝지은 arm** 동시 실행 |
| **보고 전용** | SDM (rod model 부재로 **INCONCLUSIVE**, §7.4), 신경망 헤드 예측(그림자) |

### 0.6 존재하지 않는 것 (요약, 전체 목록 부록 B)

- `FuelDesign` 축 위의 acquisition 항 — **없다.**
- 프로덕션 체인의 **핀맵 저작** — **없다** (일회성 스크립트에만).
- **LEU+ 농축도 상한의 인허가 근거** — 워크스페이스 어디에도 없다 (§4.2).
- surrogate(A) → 2군 XS / ADF / 분지계수 / HGC — **없다. 원리적으로 없다**(noBranch).
- `enforce_new_type`(compliance.py:282) 의 **호출자** — 테스트 외 0개.
- **SDM용 full-core rod model** — **없다** (campaign.py:3012-3023이 명시).
- 패키지의 **`.sum` / `dec_FA_*.inp`** — **스테이징되지 않는다** (★ hgc/ 에 `.HGC` 37 + `.out` 37뿐).
- `FuelDesign`의 **레이아웃 필드** — `type_id`·`key`·`as_dict()` 전부 레이아웃 맹목 (★ spec.py:85-111).

**★ v2에서 새로 바이트 확인한 것:** MAS_XSL/HFF 크기 산술(6개 N에서 정확 일치, §6.4);
z1/z2 옥탄트 8행(§4.1); 패키지 `bases` 8 / `cores` 10 / `hgc` 37+37; `designs.json` 37 type_id;
`templates_lat1600` 트리; `realize_lat1600.log` DeCART wall; **(A) `du` 포락과 준법 비율의 충돌**(§4.2).

---

## 1. 왜 지금인가 — 이 지시는 프로그램 계보상 등록된 다음 레버다

(v1 §1 유지. 표의 근거는 재확인했고 수치 표기만 정정했다.)

| 날짜 | 판정 | 이 지시와의 관계 |
|---|---|---|
| 2026-08-30 | **intervention r1 H3 성립**: `F_xy:F_r` 전달계수가 농축·반경 무브 **1.23–1.42** vs Gd/격자 무브 **0.55–0.73** | Gd/격자 축이 **다른 물리**임이 측정됨 |
| 2026-08-30 | **H4 성립**: Gd/격자 기술자 결손 확립 (`d_fresh_enr_r_center ≡ 0` 인데 `F_xy` +0.07 이동) | 모델이 이 축을 **보지 못한다** |
| 2026-08-31 | **F_xy 헤드 arm 4 Amendment D**: 후보 (c) "Gd/격자 채널 추가"를 **데이터 부재로 기각** | 채널을 넣으려면 **연료를 먼저 만들어야** 한다 |
| 2026-09-02 | **min_fxy r2 판정 = NULL**: 100 제출 / **99 수렴** / 73 feasible / **0 deliverable**, best 1.5437(콜 12), incumbent 1.5295 대비 **+0.0142**, 이후 **88콜 무이득**. §5 귀속에서 CBC벽·F_r벽·F_q 배제 → **"셀 바닥"**. 등록된 다음 수: **"r3는 Gd/격자 descriptor 레버"** | **이 지시가 그 r3다** |
| **2026-09-03 02:07** | **r2 phase-2 pin 웨이브 완주 (30/30)** | incumbent `a785eded`의 **측정 pin BU 수확 진행 중 — 본 문서 작성 시점 미확정(pending)**. §7.4 PRIMARY의 pin 축 기준선이 여기서 확정된다 |

> **v1 §1의 표기 오류 정정:** v1은 "88콜 무이득"(§1)과 "100 콜 … 0/99"(§7.4)를 다르게 적었다.
> 정본은 **100 제출 / 99 수렴 / best는 콜 12 / 이후 88콜이 아무것도 못 움직임**이다.

**동시에, 정직하게 기록해야 할 반대 증거가 있다.**
`flat_assembly_fr_verdict_20260809.md`는 연료 레버가 **2차항**임을 측정했다 — 24개 elite 패턴 ×
E3_E4에서 `F_r` 중앙값 이득 **−0.0069**, 같은 기간 패턴 탐색 단독 이득 **−0.057** (8배).
383개 실측 feed-121 노심에서 pair 중앙값 `F_r` 을 `FF_hot`에 회귀하면 기울기 0.185 / r 0.37.
OPSCREEN §8도 같은 서열을 준다 — **패턴 레버 −0.057 > 대조 실수 비용 +0.21 > 연료 레버 −0.022.**
→ 이 축의 존재 이유는 "연료가 LP보다 세다"가 아니라 **"LP가 이미 바닥을 쳤을 때 바닥 자체를
옮기는 유일한 남은 레버"** 다. §7.4의 NULL 분기는 이 반대 증거를 그대로 반영한다.

---

## 2. 용어 정리 — 이 워크스페이스에는 "surrogate"가 네 개 있다

(v1 §2 유지 — 비평 양측 모두 이 절을 검증-정상으로 확인했다.)

| 기호 | 이름 | 스케일 | 입력 | 출력 | 이 설계에서의 역할 |
|---|---|---|---|---|---|
| **(A)** | `6_DeCART_Surrogate` (v2026-07-22) | **집합체 (2-D 격자)** | 설계변수 10 스칼라 + Gd 위치 | K-CONV(62), pinmap(62,8,8), FF, `peak_max`, `crossing_bu` | **후보 스크리닝 (§4.4)** |
| (B) | `2_LP` MOCHA 평형노심 surrogate | **평형 노심** | PREFIX MASTER cycle 2/3/4 물리량 56 feature | `max_fxyp` 외 6 | 무관 |
| (C) | `2_LP/MOCHA/surrogate_adapter.py` | 다리 (B→A) | ReloadRule의 fresh 타입 | `fr_asm_max`, `kconv_flatness` | **선례로만 참조** (§4.5). 기본 OFF |
| (D) | `terminal_surrogate.py` 등 | 노심 (advisory) | 규칙 해시 / 사이클 시퀀스 | `max_frp` / 다목적 | 무관 |

**(A)의 정체 (README.md:1-6):** "DeCART2D 2D 격자계산(noBranch)을 대체하는 서로게이트".
K-CONV는 k-eff도 K-CRIT도 아니다. **HGC를 만들지 않고, 2군 상수·ADF·분지계수를 만들지 않는다.**
→ **(A)는 랭킹 도구이지 라이브러리 생산 도구가 아니다.**
`5_RL` 코드 안에 `6_DeCART_Surrogate` 를 참조하는 **기능적** 지점은 0개다
(독스트링 언급 1건: realize_lat1600.py:172).

---

## 3. ① 트리거 — **전면 재작성**

### 3.1 지금 없는 것 (v1 유지)

- lpopt의 모든 목적함수는 **케이스를 고정한 채 패턴을 랭킹한다** —
  `score_min_fxy`(acquisition.py:841), `score_min_fr_max_cycle`(:489), `MinFuelCostSpec`(:1054),
  `score_flat_power`(:1799).
- **`FuelDesign` 축 위의 목적함수 항은 0개.** "설계 1달러당 기대개선"은 정의된 적이 없다.
- 다만 **가상 집합체를 꽂을 구멍은 두 개 뚫려 있다** — `construct.build_pair_universe(types=...)`
  (construct.py:770), `achievable_e_core_interval`(:748), `screen_e_core_band`(:847);
  그리고 `fuelcost_search.cell_fe_prior`(:97) 계열의 FE 가격 사슬 (전부 소스 확인).

### 3.2 ★ 모델-프리 사슬의 정직한 정체 — **floor 추정기**

v1의 §3.4는 이 사슬을 "물리적 심장"이라 부르고 정확도를 0.008로 적었다. 재확인 결과:

| 단계 | 식 | 적합 품질 | **정직한 해석** |
|---|---|---|---|
| 격자 → FF | (A) `peak_max` | 홀드아웃 abs err max 1.2e-3; T3–T6 실측 대조 **bias −0.0014 / rms 0.0015 / max 0.0021** | **신뢰 가능.** 유일하게 0.005 결정문턱 아래 |
| 쌍 → node_peak | `node_peak = 1.4210 − 4.1725·contrast − 3.4862·d_fresh` | rms **0.036**, R² 0.866, **n=15** | 지배적 오차원 |
| → F_r | `F_r = A · node_peak · FF_hot`, **A = 1.035 ± 0.031** | R² 0.901 (13 arms, 동일 패턴) | 두 번째 오차원 |
| → F_xy (**측정** F_r 입력) | `F_xy = r · F_r`, `r` = 1.0546–1.0634 (중앙 1.0640) | **MAE 0.008** | **단위환산.** 예측이 아님 |
| → F_xy (**예측** F̂_r 입력) | 동일 식 | **+0.0614 (n=1, r2 반사실)** | **이 설계가 쓰는 모드. 헤드(+0.0333)보다 나쁨** |
| 운전점 (cyclen/CBC) | opmodel 이산 평형해 + `w_B = 5.348 pcm/ppm` | cyclen rms **4.26 EFPD**, CBC **±70 ppm (90%)** | 게이트 설정에 사용 가능 |

**전파 산수 (재현 가능).**
σ(F_r)|node_peak = 1.035 × 1.12 × 0.036 = **0.042** ;
σ(F_r)|A = 0.031 × 1.30 × 1.12 = **0.045** ;
직교합 **0.061** ; × r(1.059) → **σ(F_xy) ≈ 0.065.**
→ **해석적 0.065와 실측 0.0614가 일치한다.** 두 독립 추정이 같은 답을 준다.

**OPSCREEN이 이 사용을 명시적으로 금지한다.** OPSCREEN.md:253-255 —
*"`F_r_floor` is reported as what the LP could reach **if** it restores `node_peak` to the best
value ever measured — **not as a prediction of what the current patterns will give.**"*
같은 절의 반례가 B3(T5_T6): **FF_hot 1.1020으로 표에서 가장 평탄**한데 **측정 `F_r` 1.5795로 최악**;
B2(T3_T4)는 FF_hot 1.1430인데 1.5329. 그리고 **32개 독립 elite 패턴이 T5_T6의 손실을 회복하지
못했다**(OPSCREEN.md:249-251). C6 널 테스트(121슬롯 전부 A2 → 동일 FF_hot 1.178에서
`F_r` 1.8175 vs A8_A2의 1.5483)는 같은 교훈의 극단이다.

> **등록:** 이 사슬은 **하한(floor) 추정기 + 순서키(ordering key)** 다.
> **크기 판정에 쓰려면 σ_chain을 먼저 측정해야 하고, 그 전에는 어떤 발사 규칙도 집행 불가다.**

### 3.3 ★ 발사 규칙 — v1의 `ΔF_xy ≥ 0.020` 폐기, 사전등록 k·σ 바로 대체

**v1 규칙 1 (폐기).** "예측 `ΔF_xy ≥ 0.020`; 근거: r1→r2 실질 이득 0.008". 두 군데가 틀렸다 —
(a) **0.008은 r1 자신의 정정 이득**(1.5402 → 1.5322, pinbu r1 §5.2)이고 r1→r2 이득은 **음수**(+0.0142
악화)다; (b) 0.020은 σ_chain(≈0.065)의 **1/3**이라 집행 불가능하다.

**v2 규칙 (사전등록).**

```
발사 조건 — 아래 다섯을 모두 만족할 때만 DeCART 웨이브를 쏜다.

F1  task #0 (§3.4) 이 완료되어 σ_chain,paired 가 측정되어 있다.               ← 없으면 발사 금지
F2  예측 ΔF̂_xy(후보 − incumbent, 짝지은 차분) ≤ −k·σ_chain,paired , k = 2.
    ★ σ_chain,paired 는 절대오차가 아니라 **차분 오차**다. 계통성분이 상쇄되므로
      절대 σ(0.065)보다 작을 수 있다 — 얼마나 작은지가 task #0의 핵심 산출물이다.
F3  후보가 (A) 학습 **경계(bounds)** 안: u_high∈[5.00,7.00], du∈[0.40,0.80],
    gd_u∈[3.00,4.30], gd_wt∈{6..10}, n_gd∈{0,4,8,12,16,20,24}, 레이아웃 Chebyshev ≥ 2.
    ★ **격자(step) 위반은 허용, 경계 위반은 금지.** 근거: T3/T4 는 du=0.75 로 0.1 격자를
      벗어난 채 스크린되었고 DeCART 대조에서 <100 pcm 로 통과했다 (§4.2). 반면 경계 밖
      외삽은 n_gd=28 에서 +1,750 pcm 로 무너진다. v1의 "`validate_design` errors == []"
      조건은 T3/T4 를 소급 기각하므로 **틀렸다.**
F4  역할 대조 `contrast ≥ 0.026` (쌍 단위). 집합체는 홀로가 아니라 쌍으로만 값이 있다.
F5  운전점: opmodel 로 cyclen ∈ [620,645] EFPD ∧ CBC ≤ 1500 ppm (1600 − 100 마진).

σ_chain,paired 가 0.02를 넘으면 → **트리거는 성립하지 않는다.** 그 경우 등록된 귀결은
"예측으로 발사하지 않고, §8.2 슬라이스를 **짝지은 물리실험**으로만 수행한다" 이다 (§3.5).
```

**신경망 헤드의 지위 — 그림자(shadow)로 강등.** s1j의 셀내 `F_xy` 순위 충실도는 **측정된 실패**다
(min_fxy r2 RANK 게이트 3/3 FAIL; pinbu r1 헤드 셀내 ρ −0.11; arm 4 G5 R1 ρ 0.535, n=95 CI ∋ 0).
그러나 **모델-프리 사슬이 헤드를 이긴다는 v1의 주장도 근거가 없다** — 유일한 동일 모드 비교에서
사슬 +0.0614 vs 헤드 +0.0333으로 **헤드가 나았다**. 따라서 v2는 둘 다 게이트에서 뺀다:

> **헤드 예측·사슬 예측 모두 `wave_prereg.json` 에 기록하고, task #0의 소급검증에서 같은 척도로
> 점수 매기며, 어느 쪽도 라운드 1의 발사·랭킹·판정을 잡지 않는다.**
> (r2 판정의 등록 조항 유지: within-cell ranking 조항이 G1–G4에 들어가기 전 어떤 새 헤드도
> wave 랭킹 금지. 라운드 1의 `model_dir = s1i`.)

**T-A/T-B/T-C 계층 (v1 §3.2-3.3 유지, 지위만 강등).** `FuelVec` 섭동 기반 필요신호 `N`(T-A),
가상 로스터(T-B), 셀 FE 가격(T-C)은 **후보 *열거*의 힌트**로만 남긴다 — geometry 채널은 OOD
포락이 퇴화 [0,0]이라 제외. **크기 판정은 §3.4가 σ를 준 뒤에만.**

### 3.4 ★ task #0 — 소급검증 사전등록 (DeCART 0, MASTER 0, 로컬 0)

**목적.** 사슬의 **짝지은 차분 오차** `σ_chain,paired` 를 측정한다. 이것이 F2의 바를 정한다.
**집행 위치.** HOST_238 (`ssh -p 8022 USER@HOST_238`), `~/lpopt_ws`, venv,
스토어 사본 `scratch/records_r2_76793.parquet`. **읽기 전용, 산출은 리포트 1개.**

**입력 (전부 이미 디스크에 있다)**

| 소스 | 무엇 | 위치 |
|---|---|---|
| `fr_arms` 15–17 arm | **바이트 동일한 단일 장전패턴**에서 fresh 연료만 교체 — 측정 `FF_hot`·측정 `node_peak`·측정 `F_r` | OPSCREEN.md:220-235 (`measured.py`) |
| **T3–T6** 4개 저작 lat1600 타입 | 설계 튜플 + 레이아웃 + (A) 예측 FF + DeCART 실측 FF + HGC k곡선 | `designs.json`(4행), `s02_surrogate_vs_decart.py`, `hgc_curves.npz` |
| OPSCREEN **B-arm** (B2 T3_T4 / B3 T5_T6) | 저작 격자가 실제 노심으로 간 유일한 사례 2건 | OPSCREEN.md:222-224 |
| 측정 `F_xy` 라벨 | `a785eded` 1.5295/`F_r` 1.4694 (E1_E2/f121, ga80) · `bf3a70b2` 1.5322/1.4857 (T6_T4/f121) · `4d70ab6f` 1.5402/1.4605 · `0a83b654` 1.5471 · `eaf914e5` 1.5480 · `456199b3` 1.5498 · `188c9a33` 1.5684 | pinbu r1 §5 표, r2 결과 |
| r2 수렴행 99개 | 셀내 분포 | 스토어 |

**절차 (사전등록)**

```
P1  node_peak 회귀를 **leave-one-arm-out** 으로 재적합한다 (n=15는 이 사슬의 적합 모집단이므로
    in-sample 잔차는 오차를 과소평가한다).  각 홀드아웃 arm에 대해 설계입력만으로
      F̂_xy = r · A · node_peak(contrast, d_fresh) · FF_hot
    를 계산한다.  A 도 홀드아웃에서 재추정한다.
P2  절대 성능 보고: bias, MAE, p95  (기대: MAE ≈ 0.04–0.07)
P3  ★ **짝지은 차분** 성능 보고 — 모든 (i,j) 쌍에 대해
      MAE[ (F̂_xy,i − F̂_xy,j) − (F_xy,i − F_xy,j) ]
    를 (a) **동일 패턴 내**(fr_arms) 와 (b) **셀 간**(측정 F_xy 라벨 7개) 로 **분리** 보고한다.
    이 두 숫자가 각각 σ_chain,paired(동일패턴) / σ_chain,paired(셀간) 이다.
P4  헤드 그림자 점수: 동일 행에 s1i·s1j 예측을 붙여 같은 척도로 P2/P3를 반복 (게이트 아님).
P5  T3–T6 4행으로 **밀도·Xe 규약 계통편차**를 재확인 (§4.1 R4/R5의 회귀 검사).
```

**수용 기준 (사전등록, 사후 조정 금지)**

| σ_chain,paired (셀 간) | 판정 | 귀결 |
|---|---|---|
| **< 0.005** | 사슬이 결정문턱을 해상한다 | F2의 `k=2` 바 그대로 발사 |
| **0.005 – 0.020** | 순서키로만 유효 | 바 = `2σ`. OPSCREEN §8 후보가 이 바를 넘는지 **재계산**해서 넘으면 발사 |
| **> 0.020** | 사슬은 크기를 못 잰다 | **트리거 폐기.** §8.2 슬라이스를 **짝지은 물리실험**으로 재정의하고 (§3.5) 트리거 설계는 v3로 이월 |

**예상 (등록).** §3.2의 두 추정(0.065 / 0.0614)은 **절대** 오차다. 짝지은 차분은 `A`의 계통성분이
상쇄되어 더 작을 수 있으나, 지배항인 `node_peak` rms 0.036은 **후보마다 contrast가 다르므로
상쇄되지 않는다** → **0.02 초과가 기저 시나리오**다. 즉 **세 번째 행이 유력하다.**
이것을 미리 적어 두는 이유는, 그 경우에도 슬라이스는 여전히 가치가 있고(§3.5) 다만
**"예측으로 정당화된 발사"라는 서사만 폐기**되기 때문이다.

### 3.5 ★ 트리거가 성립하지 않아도 슬라이스는 한다 — 단, 서사를 바꾼다

v1 §3.5는 "슬라이스에서는 트리거를 만들지 않는다"였다. v2는 한 걸음 더 간다.

> **σ_chain,paired > 0.02 이면, 슬라이스는 "예측된 이득을 실현하는 작업"이 아니라
> **"짝지은 물리실험"** 이다** — `fr_arms` 방식으로, **동일 패턴·동일 셀에서 fresh 쌍만 교체**하여
> `ΔF_xy` 를 **직접 측정**한다. 이 설계에서는 사슬이 필요 없다: 효과를 예측하지 않고 재기 때문이다.
> 대가는 "1 라운드로 축의 값을 판정한다"는 야심을 **"1 라운드로 축의 *부호*를 측정한다"** 로
> 낮추는 것이고, 이득은 판정이 **모델 오차에 의존하지 않게** 되는 것이다.
> 파이프라인(②–⑥)은 어느 서사에서도 동일하게 필요하다.

---

## 4. ② 후보 추출 — 설계공간, 스크리닝, 다양성, 라운드당 개수

### 4.1 두 개의 설계 포락이 서로 다르다 (v1 유지 + 3건 정정)

| 축 | (A) 학습 포락 (predict.py:109-121) | lpopt paramA 생산 포락 (spec.py:31-41) | **v2 판정** |
|---|---|---|---|
| `u_high` / `e1` | 5.00–7.00, 0.05 step | {5.0, 5.4, 5.8, 6.2, 6.6} | **§4.2의 결합 제약이 실효 상한을 정한다** |
| `du` / `e2` | **du 0.40–0.80, 0.1 step** | ratio {0.85, 0.92} | **★ 두 규약이 충돌한다 — §4.2** |
| `gd_u` | 3.00–4.30, 0.05 step | **4.0 고정** (`GD_CARRIER_ENR`) | **고정 유지.** 4.3→4.0 이득 +0.0011~0.0019 FF = 분해능 이하 → 널 축 |
| `gd_wt` | {6,7,8,9,10} | {6,8,10} | **{6,8,10}로 제한.** 7·9는 템플릿 부재 + UO2G 밀도 규약 부재 |
| `n_gd` | {0,4,8,12,16,20,24} | {12,16,20,24} | **★ v1 정정: {12,16,20,24}로 제한.** 아래 참조 |
| **Gd 위치** | 10 후보셀, 상호 Chebyshev ≥ 2, **총 89 레이아웃** (n12:18, n16:24, n20:28, n24:19) | **축 자체가 없음** | **★ 유일하게 값이 있는 축** |
| pattern | PA / PB | zoning_variant z1 / z2 | **★ `z1 ⇔ PB`, `z2 ⇔ PA` (반전). §4.1a** |

> **★ 정정 1 — `n_gd` 는 {12,16,20,24} 뿐이다 (v1은 "{0,4,8,12,16,20,24} 전체 개방 권고"였다).**
> 세 겹으로 막혀 있다: (a) `FuelDesign.__post_init__`(spec.py:64-67)이 `n_gd <= 0 or n_gd % 4`
> 를 raise → **`n_gd = 0` 은 생성 자체가 불가**; (b) `_TEMPLATE_ROOTS`(lattice.py:32-35)는
> `5.8_5.1/FA → (12,16,20)`, `260624/FA → (24,)` 만 알고, 광역 폴백 스캔도 디스크에 없는
> `IGD_4/IGD_8` 을 찾지 못한다 → `LatticeError`; (c) `fuel_types.py:1477` 이 **"디렉터리 이름이
> gd_wt/n_gd의 정본"** 이라고 못 박으므로, 저작 덱은 반드시 실재하는 `{gd}_{n}_z{z}` 디렉터리에
> 놓여야 한다. 저-Gd 확장은 **템플릿 패밀리(덱 + 디렉터리 + 밀도줄) 신규 저작**이 선행되는
> 별도 과제다 — `edit_dec_text` 도 `author_template` 도 그것을 하지 못한다.

> **★ 정정 2 — Gd 레이아웃 89종은 `n_gd`별 합계지 `n_gd`당 개수가 아니다.**
> `SURROGATE_USAGE.md:294` — "유효 배치 89종 (n12:18, n16:24, n20:28, n24:19)".
> v1 §4.3의 열거식 `... × {n_gd} × {89 레이아웃} × ...` 는 **약 4배 중복계산**이다.
> 올바른 형태는 `{(n_gd, layout) ∈ 89 pairs}` (§4.3).

**★ 정정 3 — UO2G 밀도 규약 불일치의 크기와 출처.**
(A)는 내부 공식으로 `6%→9.942 / 8%→9.850 / 10%→9.760 g/cc` (SURROGATE_USAGE.md:249-250,
`predict.py:1071` 의 `rho_train` 공식과 동일), 동결 템플릿 계열은 `6→10.01 / 8→9.95 / 10→9.88`
(realize_lat1600.py:232-238, 디스크에서 확인됨). 차이는 **0.68% @6 / 1.02% @8 / 1.23% @10** —
**셋 다 `predict.py:1070-1074` 자신의 0.5% 경고 문턱을 넘는다**, 그리고 **최대치인 gd_wt 10 이
슬라이스 Z2의 패밀리다.**
v1이 인용한 `SURROGATE_USAGE §7.7`의 **"+1.6%"** 는 **다른 패밀리**(GA 시대 덱
`D:\GA_Based_Screening+Optimize\FA\6_16_z1\dec_FA_X2.inp`, UO2G 밀도가 gd_wt 무관 10.01 고정)의
숫자다 → **인용 정정.** `edit_dec_text`는 밀도줄을 손대지 않고(lattice.py:114-147 재확인),
작업 #1의 `author_gd_layout` 도 핀 위치만 다루므로 **이 편차는 사슬 안에서 고칠 수 없다.**

**★ 정정 4 — 그런데 이 계통편차는 이미 측정되어 있고, 크지 않다 (R4/R5 강등).**
v1은 Xe 규약 불일치를 "새로 발견, 미등록"이라 적고 G-H4를 "밀도·Xe 편차를 처음 측정하는 곳"이라
했다. 틀렸다. **T3–T6이 바로 그 측정이다** — 동결 `0_APR1400` 템플릿(밀도 10.01/9.95/9.88,
`xenon TR`)에서 저작되어 DeCART로 계산되었고, (A) 대조에서
**BU ≥ 0.2 전 구간 k 오차 < 100 pcm** (Xe-free BU=0만 −2,200 pcm, 미사용),
**FF bias −0.0014 / rms 0.0015 / max 0.0021** (gd_wt 6·8·10 전부 포함)로 통과했다
(OPSCREEN.md:165-179). `s08_transfer.py`는 여기서 파생된 운전점 차이가 **≤ 0.5 EFPD, ≤ 8 ppm**
임까지 보였다.
→ **R4/R5는 "수백 pcm 스크린 편향"이 아니라 "≤100 pcm / ≤0.0021 FF 로 이미 유계"** 로 강등한다.
**G-H4는 첫 측정이 아니라 회귀검사**이며, 문턱(100 pcm / 0.0021)은 그대로 유지한다 —
그 값이 정확히 T3–T6의 실측 최대치이기 때문이다.

**Xe 규약 (결정 유지).** (A)는 평형 제논 가정(README §5, `predict.py` 고정조건 "boron 500ppm /
xenon EQ / PA zoning"), 동결 프로덕션 템플릿은 `xenon TR`, 2_LP의 2026-08-21 캠페인은 한 줄만
고쳐 `xenon EQ`로 갔다. → **프로덕션 HGC는 `xenon TR` 유지** (Q3). 한 MAS_XSL 안에 서로 다른 Xe
모드의 COMP를 섞는 것은 검증된 바 없고, 기존 37 paramA + 80 ga80 세트가 전부 TR이다.
(A)의 EQ 가정은 위 §정정 4가 준 유계 안의 **스크린 전용 계통오차**다.

#### 4.1a ★ R17 종결 — `z1 ⇔ PB`, `z2 ⇔ PA` (직관과 **반대**)

v1은 이것을 "대응 확인 필요"로 열어 두었다. 디스크에서 바이트로 닫힌다.

```
predict.py:67-71   FIXED = {(0,0):9, (3,3):6, (4,3):8, (4,4):9}
                   ZONING_COMMON = {(1,0),(3,2),(4,2),(5,3),(5,4)}
                   PA = ZONING_COMMON | {(7,c) for c in range(8)}      ← 행 7 전체
                   PB = ZONING_COMMON | {(7,6),(7,7)}                  ← 행 7의 끝 두 칸

0_APR1400/5.8_5.1/FA/IGD_16/6_16_z1/dec_FA_A01.inp  옥탄트 8행째: 1 1 1 1 1 1 2 2   → PB
0_APR1400/5.8_5.1/FA/IGD_16/6_16_z2/dec_FA_A02.inp  옥탄트 8행째: 2 2 2 2 2 2 2 2   → PA
(공통 zoning 셀 (1,0),(3,2),(4,2),(5,3),(5,4) 와 안내관 (0,0),(3,3),(4,3),(4,4) 는 양쪽 일치)
```

독립 확인: OPSCREEN.md:410 — "Zoning in both base decks was verified to be exactly the **PB** set
(`s22_template20.py`)" — 두 덱 모두 `*_20_z1` 이다.

> **왜 중요한가.** `2_LP/MOCHA/config.py:349-358` 의 기본값이 `surrogate_pattern = "PA"` 다.
> **z1 설계를 PA로 스크린하면 조용히 틀린 패밀리를 평가한다** — 오류도 경고도 없다.
> → **작업 #4의 `screen.py` 는 `Z_TO_PATTERN = {"z1": "PB", "z2": "PA"}` 를 모듈 상수로 두고
> 임포트 시 어서션한다.** R17 종결.

### 4.2 ★ 인허가·포락 제약 — 그리고 **새로 발견된 결합 상한**

**코드에 있다 (노심 수준, v1 유지):**

| 제약 | 값 | 위치 |
|---|---|---|
| `F_r` | ≤ 1.55 | `delivery.LICENSING_FR_LIMIT`(delivery.py:43) |
| `F_xy` | ≤ 1.65 | `delivery.LICENSING_FXY_LIMIT`(delivery.py:50) |
| `F_q` | ≤ 2.41 | `config.f_q_limit` |
| CBC | ≤ 1550 (프로그램 게이트 1600) | `config.cbc_limit` |
| \|AO\| | ≤ 0.30 | `config.ao_abs_limit` |
| 핀 연소도 | **80 GWd/tU** (핀 axial peak), 예측 게이트 78 | config.py:347,358,383 |
| **MTC** | **[−54, +9] pcm/°C** | `config.ConstraintsConfig.mtc_min/max_pcm_per_c`(config.py:1228-1232), `[sdm_mtc]`(config.py:1281-1282) |
| **SDM** | **≥ 10,870 pcm** (= 10,180 + 690) | `[sdm_mtc] sdm_required_pcm`(config.py:1283-1285) |
| 준법 R1/R2/R3 | 패턴당 농축 스펙 1개 / 옥탄트 대칭 `OCTANT_TOL 1e-3` / 1/4 회전대칭 | `data/compliance.py` |

**코드에 없다 (사용자 확인 필요):**

> **농축도 상한의 인허가 근거가 이 워크스페이스 어디에도 없다.** 사실상의 경계 셋은 전부
> 물리/데이터/프로그램 경계다 — (A) 학습 포락 ≤ 7.00 w/o; 실현된 paramA가 6.6 w/o까지 감
> (단 CBC 2137–2244 ppm으로 운전 불가); lat1600 스크린이 5.00–5.50으로 스코핑
> (OPSCREEN.md:430: **"프로그램 결정이지 물리 결정이 아니다"**).
> → **가정 A1 (§10).** 5.50 w/o.

#### ★ 새 발견 — 준법 비율(0.85)과 (A) `du` 포락(≤0.80)이 **`u_high`를 5.333 w/o로 묶는다**

`du ≡ e1 − e2` 이고 준법 R1이 `e2 = 0.85·e1` 을 요구하므로 **`du = 0.15·e1`**. (A)의 경계는
`du ∈ [0.40, 0.80]` (predict.py:109-112, `_snap_to_grid`가 `lo−1e-9 ≤ v ≤ hi+1e-9` 를 강제).

```
du ≤ 0.80  ∧  du = 0.15·e1     ⟹   e1 ≤ 5.3333 w/o
du ≥ 0.40                       ⟹   e1 ≥ 2.667  (비구속)
u_high ≥ 5.00 (설계 관심 하한)  ⟹   ★ 실효 창: u_high ∈ [5.00, 5.333], e2 ∈ [4.25, 4.533]
```

> **결론 1.** **Q1의 5.50 w/o 상한은 구속하지 않는다.** 먼저 구속하는 것은 인허가가 아니라
> **서로게이트의 `du` 경계**다. 이 사실은 v1에도 두 비평에도 없다.
>
> **결론 2 — 슬라이스 후보 Z1이 포락 밖이다.** OPSCREEN.md:402의 **Z1 = u5.50 / e2 4.6750**
> 은 `du = 0.825 > 0.80` → `validate_design` 이 `"du=0.825 outside [0.4, 0.8]"` 를 반환한다.
> **v1 §8.2는 이 후보를 그대로 슬라이스에 넣었다.** §8.2에서 재지정한다.
>
> **결론 3 — 준법 허용오차를 쓰면 5.50을 지킬 수 있다.** `zone_ratio_flag` 는 `|e2/e1 − 0.85| ≤ 0.03`
> 를 본다 (compliance.py:69-71). `e2 = 4.70` 이면 비율 0.8545 (|Δ| = 0.0045 ≤ 0.03 **통과**)이고
> `du = 0.80` 으로 **(A) 격자 위에 정확히 앉는다.** → **Z1′ = u5.50 / e2 4.70** 이 권고안이다.
> 단 이는 "ratio를 0.85로 고정"하는 가정 A2의 완화를 요구한다 → **§10 A2에 오버라이드 항목으로 명시.**
>
> **결론 4 — 격자(step)는 경계가 아니다.** `_snap_to_grid` 는 0.1 격자를 벗어나면 오류를 내지만,
> **T3/T4(`du = 0.75`)는 그 격자 밖에서 스크린되어 DeCART 대조 <100 pcm 로 통과했다.**
> 따라서 F3(§3.3)은 **경계만** 강제하고 격자는 강제하지 않는다.

### 4.3 역사상 (기술자 → 설계) 과 준법 게이트

T-B가 내놓는 목표 기술자 `d*` 와 DeCART가 받는 설계 튜플 사이는 **(A)의 대량 정방향 평가 + 최근접
탐색**으로 잇는다 — 역모델을 학습할 필요가 없다.

```
1. 설계 격자 열거  : {u_high ∈ [5.00, 5.333] 0.05 step = 7점}          ← §4.2 결합 상한
                     × {e2 = 0.85·u_high  (또는 A2 완화 시 ratio 창)}   ← du 는 자유축이 아님
                     × {gd_wt ∈ 6,8,10 = 3}
                     × {(n_gd, layout) ∈ 89 쌍}                        ← ★ n_gd별 합계
                     × {pattern ∈ PB(z1), PA(z2) = 2}
                     ≈ 7 × 3 × 89 × 2 = **3,738 설계**   (gd_u 4.0 고정)
2. (A) 배치 예측   : GPU 541 case/s → **~7 s** ; CPU 8.6 case/s → **~7 min**
3. 기술자 투영     : kconv → (kinf0/10/30, bu_k1, 곡선형상 8) ; pinmap → FF, ff 분포
4. 최근접          : z-정규화 기술자 공간(= ood_guard 스케일)에서 ‖d(design) − d*‖ 최소
5. 준법 게이트     : compliance.enforce_new_type(spec)   ← ★ §4.3a 주의
6. 역할 대조 게이트: contrast ≥ 0.026 (쌍 단위)
7. 운전점 게이트   : opmodel 로 cyclen ∈ [620,645] ∧ CBC ≤ 1500 ppm
```

> **★ v1 §4.3의 `O(10^5) 설계 ≈ 3분` 은 두 겹으로 틀렸다:** 89 레이아웃 중복계산(4배)과
> `du`를 자유축으로 센 것(5배). 실제는 **3.7×10³ 설계**이고, 그래서 **스크리닝 호스트 선택이
> 라운드 1의 병목이 아니다** — CPU로도 7분이다. (비평 양측이 제기한 "3.2 h" 시나리오는
> 10⁵ 설계 전제에서만 성립한다. 부록 D-R3.)

#### 4.3a ★ `enforce_new_type` 은 0.92를 **기각하지 않는다 — 덮어쓴다**

v1 §4.3과 §10-Q2와 작업 #16은 모두 "게이트를 배선하면 0.92 후보가 전부 기각된다"고 적었다.
소스가 다르게 말한다 (compliance.py:308-317):

```python
target_zone = ZONE_RATIO_TARGET * enr_main          # 0.85 * e1
if out.get("enr_zone") is None:
    out["enr_zone"] = target_zone                   # ← 조용히 채워 넣는다
else:
    if zone_ratio_flag(enr_main, enr_zone) != "pass":
        raise ComplianceError(...)                  # ← 명시적으로 준 값만 검사
```

`FuelDesign`에서 파생한 spec이 `enr_main`만 넘기면 **`e2`가 `0.85·e1`로 조용히 재작성**되고,
**0.92에서 돌린 서로게이트 스크린이 무효가 된 채로 통과한다.**
→ **작업 #16은 `enr_main`과 `enr_zone`을 *둘 다* 넘겨야 하고, 테스트는 채워진 값이 아니라
`ComplianceError` 를 어서트해야 한다.** (슬라이스 Z1/Z2는 둘 다 0.85 근방이라 영향 없음.)

**두 번째 배선 함정 — 옥탄트 맵 형식.** `enforce_new_type` 은 `pin_map` 을
`is_octant_symmetric(pin_map, n=16)` 로 검사한다 → **평탄화된 16×16 전맵**을 원한다.
그런데 저작 산출물은 **8행 하삼각(옥탄트)** (`realize_lat1600._triangle`)이고 **확장기가 없다.**
→ **작업 #1에 `octant_to_full(rows) -> 16×16` 을 함께 넣는다.**

**남은 모순은 실재한다 (v1 유지).** `spec.DESIGN_GRID["ratio"] = {0.85, 0.92}` vs
`ZONE_RATIO_TARGET 0.85 / TOL 0.03` → **0.92는 0.85±0.03 밖**이고, 라이브 registry에 0.92 타입이
실재한다 (`P6257Z2G08N16` → 6.2/5.7 = 0.919). 지금까지 터지지 않은 이유는 **`enforce_new_type`
의 호출자가 테스트 외 0개**이기 때문이다. 단 **`DESIGN_GRID`는 LHS 표본 격자이지 검증기가 아니다**
— `FuelDesign.__post_init__`(spec.py:56-70)은 `0 < e2 ≤ e1` 만 본다. 따라서 "spec.py bounds"를
비율 제약으로 읽으면 0.85 고정이 되고, 그것이 §4.2 결론 1의 5.333 상한을 낳는다 (§10 A2).

### 4.4 (A) 서로게이트의 입출력과 갭 (v1 유지)

**입력** (predict.py:109-121): `u_high`, `du`(→`u_low = u_high − du`), `gd_u`, `gd_wt`, `n_gd`,
`gd_positions` (1/8 맵 10 후보셀, 상호 Chebyshev ≥ 2; 대각 = 4핀, 비대각 = 8핀, 합 = `n_gd`),
`pattern ∈ {PA, PB}`. `n_gd = 0` 이면 `gd_u/gd_wt/gd_positions` 는 학습 플레이스홀더로 강제
(단 §4.1 정정 1에 따라 **lpopt 쪽에서 n_gd=0은 생성 불가**).

**출력** (predict.py:941-960): `bu_grid`(62), `kconv`(62), `pinmap`(62,8,8), `peak_pin_power`(62),
스칼라 `peak_max`, `peak_max_bu`, `k_bu0`, `crossing_bu`.

**갭 — lpopt가 필요한데 (A)가 주지 않는 것.**

| lpopt 채널군 | (A) 제공? | 실계산 필요 이유 |
|---|---|---|
| `kinf0/10/20/30`, `bu_k1`, k-conv 형상 8채널 | ✅ | — |
| `ff_pin_max` | ✅ (`peak_max`) | — |
| `xs_d1/d2/a1/a2/nf1/nf2/s12` | ❌ | HGC `%MACX` |
| `adf_face_g1/g2`, `adf_corner_g1/g2` | ❌ | HGC `%ADFT` / MAS_HFF |
| `boron_worth`, `doppler_coef`, `mtc_dmod`, `cr1_worth` | ❌ | **BOC 분지 상태에서만** — noBranch 원리적 불가 |
| 핀 연소도 첨두 곡선 | ❌ | `.sum` EDIT 3 BRP 열 (**★ 그런데 `.sum`은 패키지에 없다 — §6.3**) |
| **HGC 자체** | ❌ | **noBranch → MASTER 라이브러리 불가** |
| **불확실도** | ❌ | `uncertainty: {available: false, reason: predictor_has_no_uq}` |

→ **(A)는 랭킹만 한다. 실계산은 선택이 아니라 필수다.**
**(A)의 알려진 한계:** `n_gd` 격자 **밖 외삽 금지** (n_gd=28에서 k BOC 오차 **+1,750 pcm**);
최대 오차는 BU=0 부근과 설계공간 모서리(`gd_wt 10 × n_gd 24`); FF는 미세하게 과대예측
(보수적 방향); **불확실도 출력 없음.**

### 4.5 (C) 어댑터 — 선례이자 반면교사 (v1 유지)

`2_LP/MOCHA/surrogate_adapter.py`는 (B)의 optimizer가 (A)를 부르는 완성된 다리이고 독스트링이
*"advisory only … an unavailable/invalid prediction must not be used as a hard reject"* 라고
못 박는다. **그러나 실전에서 돌지 않는다** — `surrogate_advisory_enabled = False` 기본값이고
`config_apr1400.yaml` 에 **`surrogate_fuel_catalog` 항목이 없다** (연료타입 → 격자설계 사상이
비어 다리가 물리적으로 건널 수 없다).
→ **교훈: lpopt 쪽 다리는 카탈로그를 코드가 아니라 `designs.json` 에서 읽는다.**
paramA는 이미 `{type_id, e1, e2, zoning_variant, gd_wt, n_gd, alias, gd_u_enr}` 를 기록하므로
카탈로그가 이미 존재한다. **단 `gd_positions`는 37행 중 4행(T3–T6)에만 있다 (★ 확인)** →
온디맨드 타입은 이를 **필수**로 승격한다 (§6.5).

### 4.6 다양성 / 중복제거 / 라운드당 개수 (v1 유지 + 상한 재유도)

**중복제거:** 설계 튜플 완전일치 + **기술자 공간 근접 중복**(z-정규화 거리 < 0.25 접기).
**다양성:** 설계 공간이 아니라 **기술자 공간에서 greedy max-min** (척도 = ood_guard z-스케일).
**역할 쌍 제약:** 후보는 낱개가 아니라 **(68-슬롯 역할, 53-슬롯 역할) 쌍**으로 뽑고
`contrast ≥ 0.026` 미달 쌍은 기각. 근거: contrast ≈ 0 인 arm들이 node_peak 1.387–1.551 /
`F_r` 1.559–1.818 로 흩어졌다 (OPSCREEN §6 표).

**라운드당 개수 = 4 (역할 쌍 2개), 상한 6.** 제약은 DeCART가 아니라 **하류 비용**이고,
v1보다 **더 비싸다**:

- 라이브러리 재빌드는 **부분 재빌드를 거부**한다 (library.py:104-108 stale 가드) → 전체 HGC 재판독.
- `MAS_XSL` 재빌드는 **모든 `MAS_RST.*` 를 무효화**한다 → **base 8개 + 신규 pair = 9회 부트스트랩.**
- **★ `cores/` 10개 템플릿과 `synth_decks/` 가 전부 stale이 된다** (§6.3) — v1 누락.
- `%GEN_DIM` nbatch/ncomp가 타입당 +1 (paramA 37 → `(40,42)`; +2 → `(42,44)`).
- **★ 같은 `(gd_wt, n_gd, z)` 에 두 레이아웃을 동시에 넣을 수 없다** (§6.2) → 라운드 4개는
  서로 다른 `(gd,n,z)` 를 쓰거나 작업 #1의 파일명 유일화가 선행되어야 한다.

---

## 5. ③ DeCART2D 실계산

### 5.1 스펙 → 입력 생성 (두 경로)

**경로 1 — MATERIAL 전용 (기존, 동결 레이아웃).**
`resolve_template`(lattice.py:49-71)이 `(gd_wt, n_gd, z)` 로 `IGD_<n>/<gd>_<n>_z<z>/dec_FA_*.inp`
를 고르고(`sorted(glob(...))[0]`), `edit_dec_text`(lattice.py:114)가 **숫자 3개 + CASEID만** 바꾼다.
지오메트리·핀맵·밀도·BRANCH·DEPL 블록은 **바이트 동일**.

**경로 2 — 핀맵 저작 (신규, 개방 레이아웃) ← 작업 #1.**
`realize_lat1600.author_template`(realize_lat1600.py:180-222)이 옥탄트 삼각형(8행) 안에서 셀 id
`3`(UO2G)을 옮긴다. 가드는 이미 있다 — `_triangle`(:137), `_census`(:165), `_n_gd_of`(:170),
안내관/zoning 셀 불가침, 인구조사 == `n_gd`.

**★ 산출물이 어디로 가는지 (v1 누락, 확인함).** `build_template_tree`(realize_lat1600.py:225-258)는
저작 덱을 `0_APR1400` 이 **아니라** `5_RL/templates_lat1600/<subtree>/IGD_<n>/<gd>_<n>_z<z>/`
**에 쓰고 파일명은 항상 `dec_FA_lat1600.inp`** 다 (realize_lat1600.log:2-30 로 확인:
`10_16_z1`, `8_24_z1`, `6_16_z1` 세 디렉터리). 그리고 **베이스는 설계별 자기 `gd_wt` 로 해석**한다
— 밀도가 `gd_wt`에 묶여 있고 `edit_dec_text`가 밀도를 안 고치기 때문 (독스트링에 그대로 적혀 있다).
또 **`n_gd ∈ (16, 24)` 가 아니면 `SystemExit`** 한다 → **n_gd 20인 슬라이스는 이 함수의 확장이 선행.**

**승격 시 추가해야 할 가드 (v1 + 3건):**

- `compliance.enforce_new_type` 호출 — **`enr_main`+`enr_zone` 동시 전달** + **`octant_to_full` 확장** (§4.3a)
- Chebyshev ≥ 2 (`MIN_CHEB`), 대각 4핀/비대각 8핀 인구조사 합 == `n_gd`
- BOM 없는 ASCII 출력
- `nxfile` 줄이 **실행 호스트에 실재하는 경로**를 가리키는지 (§5.4)
- **★ 파일명 유일화** `dec_FA_<type_id>.inp` + `resolve_template` 에 설계별 덱 경로 오버라이드
  (같은 `(gd,n,z)` 에 두 레이아웃이 공존할 수 있어야 라운드 4개가 성립한다)
- **★ `n_gd ∈ {12,16,20,24}` 전부에 대한 서브트리 규칙** (현행 함수는 16/24만 안다)

`edit_dec_geom_text`(lattice.py:184)의 지오메트리 축은 **이번 범위 밖**이다 (집합체 피치 20.7772
하드 거부, 대칭 ±3% 격자 불가; 핀 피치 상한 ≈ +1.06%).

### 5.2 MASTER 라이브러리 계약이 요구하는 분지/연소 세트 (절대 생략 불가, v1 유지)

동결 골든 덱 `0_APR1400/260624/FA/IGD_20/6_20_z1/dec_FA_B01.inp` 기준.

```
DEPL   burnup 0.0 0.2 0.5 1 -45/1.0 -80/2.5              → 62 점 (0 → 80 GWd/tU)
BRANCH ×16, 공통 17-점 격자 "0 0.2 0.5 1 3 5 7 10 15 20 25 30 40 50 60 70 80"
   1 BORON / 2 TFUEL / 3-8 DMOD1..6 / 9 CR1 REFERENCE / 10 CR1 BOR / 11-16 CR1 DMOD1..6
EDIT   grp 1 26        ← HGC 파일 생성 활성화 (2군, 1.8554 eV)
EDIT   isotope <45 ids> ← %MICX 소수군 미시 XS
EDIT   INVENTORY 1 1 1 -45  ← .out 핀별 MASS(g) → u_avg_enrichment / u_mass_g
```

→ **HGC = 62 + 16×17 = 334 `%TITL` 상태 블록**, 태그 9종 각 334회 + 말미 `%FINE` 1회.
`MAS_XSL` COMP 헤더의 `BURN VAR DMOD ADF DUM = 62 17 6 0 0` 로 그대로 전파된다.
**바이트 크기가 곧 게이트다** — 모든 APR1400 FA HGC는 정확히 **7,395,955 B**; `n_gd = 0` 인
V01/V02만 6,867,567 B.
**분지를 빼면?** `CRD1*` 는 CR1 분지에서만 나오고 MASTER 덱은 `%JOB_MDL irod=2` 로 돈다.
그때 MASTER가 죽는지 조용히 틀리는지는 **미검증**(R11). → **정책: 어떤 분지도 생략하지 않는다.**

### 5.3 ★ 호스트 — box 104를 표에 추가 (v1 누락)

| | HOST_199 | HOST_181 | **box 104** |
|---|---|---|---|
| CPU | i9-13900, 24C/24T, 2.0 GHz | Ryzen 9 9950X, 16C/32T, 4.3 GHz | (미기록) |
| DeCART | `C:\DecartMaster`(완전, `libiomp5md.dll` 포함) + `C:\DECART_MASTER`(dll 없음) | `D:\DeCART_MASTER` + `_ex` (**dll 없음**) | **`D:\DeCART_MASTER\BIN` — omp exe 실사용 이력** |
| 자산 | lpopt 생산 키트 | kit_frontier (2026-07-29 stale) | **`0_APR1400/` + `templates_lat1600/` + paramA 패키지 + MASTER** |
| 디스크 | C: 42.4 GB / D: 6,216.8 GB | C: 659 GB / D: 7.1 GB | (미기록) |
| 부하 | **lpopt 생산 호스트** (r2 웨이브) | 타 사용자(kgt) ICSBEP 6 레인 | (미기록) |
| **실측 DeCART** | **3,084 s/case** (51.4 min, 직렬 2-wide, V09–V14) | **미측정** | **★ 735–750 s/case, omp 4-way** (realize_lat1600.log:31-34; `design_lat1600_104.inp:48` 이 omp exe 지정) |

> **★ v1의 "23-스레드 707 s 는 재현 미확인" 은 워크스페이스 자신의 영수증에 의해 반박된다.**
> `5_RL/realize_lat1600.log:31-34` 가 `T3 wall=750s / T4 750s / T5 735s / T6 735s` 를 기록하며,
> 이는 **T3–T6 저작 개방 레이아웃 웨이브**(즉 이 설계가 하려는 바로 그 작업)를 4-way로 돌린 것이다.
> **omp 경로는 box 104에서 실증되었다.**

**exe/dll 함정 (유지).** `lattice.DEFAULT_DECART_EXE = D:\DeCART_MASTER\BIN\decart2d1.1m5omp.exe`
인데 `libiomp5md.dll` 이 181의 `D:\DeCART_MASTER\BIN` 에도, 199의 `C:\DECART_MASTER\bin` 에도 없다.
→ **작업 #2: exe/dll 프리플라이트 + 직렬 폴백.** **호스트는 dll 유무로 고른다.**

**오케스트레이터 결정 (Q4, §10 A4):** **HOST_181, 실행 루트 `C:\`, 직렬 exe, 큐 상한 4 (32스레드).**
계획 수치는 **199의 3,084 s/case** 를 쓴다 (181 미측정; 9950X이므로 더 빠를 가능성이 크다 —
"보수적이지만 귀속이 틀린" 숫자임을 명시한다). **199는 MASTER 전용으로 남긴다.**
**box 104는 더 빠르고 자산이 모두 있으나 §10에서 사용자 오버라이드 항목으로 올린다** —
omp 4-way면 2 격자가 **~13 분**이다.
라이브러리 빌드는 HGC가 떨어진 곳에서 하고 패키지를 캠페인 호스트로 배송한다 (§6.6).

### 5.4 ★ 러너 일원화 — 큐 ps1 vs `run_batch` (v1은 둘을 섞었다)

v1은 §5.4에서 `run_decart_eq_xesm_queue_*.ps1` 을 채택하면서 §8.1과 등록 경로에서는
`lattice.run_batch → DecartRun → DesignSource → stage_hgc` 를 가정했다. **둘은 호환되지 않는다** —
큐는 `runs/<type>/FA_<T>_0101.HGC` + `manifest.json` 을 내고 `FuelDesign`·별칭·`registry.json` 을
모르며, **직렬 exe의 SHA-256을 강제**하고 `decart2d1.1m5.exe` 프로세스 수를 센다 (omp면 throw).

> **결정: `lattice.run_batch` 를 정본으로 한다** (멱등성 `_hgc_looks_valid`, 올바른 리네이밍,
> `DesignSource` 직결). 큐에서 **좋은 성질 두 개만 프리플라이트 헬퍼로 이식**한다 —
> ① exe / XS 라이브러리 **SHA-256 대조**, ② 덱의 `nxfile` 줄을 스테이징된 로컬 `DML.BIN` 으로
> **정규식 재작성**. 케이스 목록은 `design_wave.json` 에서 읽고 `manifest.json` 에
> `design`(설계 튜플 + `gd_positions` + 예측 FF/k)을 함께 기록한다.

### 5.5 예산

| 항목 | 값 |
|---|---|
| 케이스당 wall | **직렬 3,084 s** (199 실측) / **omp 4-way 735–750 s** (104 실측) / 181 미측정 |
| 2 케이스 | 직렬 2-wide ≈ **1.7 h** / omp 4-way ≈ **0.25 h** |
| 케이스당 산출 | HGC 7,395,955 B + `.out` ~27.3 MB + `.sum` 767,500 B + stdout ~88 KB ≈ **35.6 MB** |
| 타임아웃 | `[design] decart_timeout = 5400 s` (config.py:906) — 직렬 실측의 1.75배, **얇다** → **7200 s 권고** |
| 동시성 | `[design] max_parallel = 4` (config.py:905). 큐를 폐기했으므로 **이 값이 정본** |
| 멱등성 | `run_batch`(lattice.py:409)는 `FA_<alias>.HGC` + `.out` 이 있으면 재실행하지 않는다 |

---

## 6. ④ HGC → MASTER 라이브러리 변환과 lpopt 등록

### 6.1 변환 (TotalBatcher / PROLOG 4.1) — v1 유지

```
스테이징 1 디렉터리 : MAS_REF(2,008 B) + 모든 FA_<alias>.HGC + prolog41m4.exe + TotalBatcher4.exe
PATH 앞에 스테이징 디렉터리 (TotalBatcher 가 prolog 를 맨이름으로 shell out)
TotalBatcher4.exe  (인자 없음, 옵션 파일 없음)
→ MAS_XSL = MAS_REF 축어(REFL 5블록) ++ COMP FA_<alias> 블록 × N
→ MAS_HFF = FA 당 폼펑션 세트
```

**가드 (구현됨, library.py:76-150 재확인):** 기존 `MAS_XSL/MAS_HFF` 는 **단일 `.bak` 세대**로 회전
(`bak.unlink()` 후 `rename`); 요청에 없는 `*.HGC` 가 스테이징에 있으면 `LibraryBuildError`
(**부분 재빌드 거부**); 빌드 후 `COMP <name>` 존재·HFF 이름 존재·COMP 수 == 요청 수 검증;
`LibraryBuild.set_names` 로 **COMP 순서를 반환**한다 (→ G-H3c에 쓴다).

> **★ 롤백 위험 (유지).** `.bak` 은 **한 세대뿐**이다. **두 번째 재빌드가 유일한 롤백을 파괴한다.**
> → **규칙: 모든 재빌드 전에 `lib/` + `bases/` + `cores/` + `registry.json` + `designs.json` 을
> `E:\lpopt_archive\` 로 스냅샷.** 선례: `data/design/package/lib.snap_20260811`.

**튜닝 노브는 없다.** TotalBatcher는 인자·옵션 파일을 받지 않고, 연소격자(62)·분지격자(17)·DMOD
수(6)는 전부 DeCART 덱에서 상속된다. PROLOG `INP` 는 디스크에 쓰이지 않아 실제 `%OPTN`/`%DH2O` 는
**알 수 없다** — 온디맨드 타입은 **동일 경로**로 만들어지므로 이 미지수는 상수로 상쇄된다.
**PROLOG 옵션을 바꾸려는 시도는 절대 하지 않는다.**

### 6.2 ★ 타입 정체성이 레이아웃을 담지 않는다 — 이 축의 구조적 결함

`FuelDesign.type_id`(spec.py:85-91)와 `.key`(spec.py:94-99)는 `(e1x10, e2x10, z, gd_wt, n_gd)`
만의 함수이고 `as_dict()`(spec.py:103-111)도 레이아웃을 담지 않는다.
`write_designs_manifest`(package.py:40-56)는 여기에 `alias` 와 `gd_u_enr = 4.0` 을 덧붙일 뿐이다.
`DesignRegistry.alias`(spec.py:188-200)는 **`type_id` 로 키잉**되어 기존 id면 기존 별칭을 반환한다.

> **위험 (실재).** 저작한 개방 레이아웃 설계가 기존 튜플과 같은 `type_id` 로 떨어지면
> **기존 타입의 별칭을 돌려받고, `stage_hgc` 가 그 `FA_<alias>.HGC` 를 물리적으로 다른 격자로
> 덮어쓴다.** 그 타입을 참조한 모든 스토어 행·restart·`case_pair` 가 조용히 틀어진다 —
> HGD569급 치환이되 **한 겹 더 깊어**(덱이 아니라 라이브러리) `validate_reload_deck` 의 로스터
> 검사가 볼 수 없다.

**★ 다만 비평이 놓친 완화 요인 두 가지 (확인함).**

1. **저작 덱은 별도 트리로 간다.** `build_template_tree` 는 `5_RL/templates_lat1600/` 에 쓰고
   `resolve_template` 에 그 루트를 넘긴다 → `0_APR1400` 의 동결 템플릿과 섞이지 않는다.
   T3(`P5042Z1G10N16`, 개방 레이아웃)과 `P6656Z1G10N16`(동결, 같은 `(10,16,z1)`)이 오늘 공존하는
   이유가 이것이다.
2. **슬라이스 후보는 충돌하지 않는다.** `designs.json` 의 37 type_id를 전수 확인한 결과
   `P5547Z1G08N20`(Z1′) 도 `P5042Z1G10N20`(Z2) 도 **없다**. → **슬라이스는 이 결함에 걸리지 않는다.**

**그러나 라운드 2 이후에는 걸린다**, 그리고 **오늘도 한 겹 걸린다**:
`resolve_template` 은 `sorted(glob("dec_FA_*.inp"))[0]` 를 취하고 `author_template` 은 항상
`dec_FA_lat1600.inp` 를 쓴다 → **같은 `(gd_wt, n_gd, z)` 에 두 레이아웃이 한 트리에 공존 불가**
= §4.6의 "라운드당 4개"가 막힌다.

**★ 처방 (작업 #1에 포함, 착수 전 확정):**

1. `FuelDesign` 에 `gd_positions: tuple[tuple[int,int], ...] | None = None` 추가(spec.py:47),
   `key` 와 `as_dict()` 에 포함.
2. `type_id` 는 **레이아웃이 동결 템플릿과 다를 때만** 태그를 붙인다 —
   `f"{base}L{sha1(layout_str)[:3]}"`. **기존 37개 id는 바이트 불변**이어야 하고
   (T3/T5/T6는 이미 개방 레이아웃인데 id는 무태그다 → **소급 재명명 금지**),
   `designs.json` 의 37 id가 라운드트립으로 그대로 나오는 단위 테스트를 건다.
3. T3–T6의 `gd_positions` 를 `FuelDesign` 으로 **백필**해 미래의 동결-쌍둥이와 `key` 가 충돌하지
   않게 한다.
4. `DesignRegistry.alias` 는 `type_id` 가 같은데 기록된 `gd_positions` 가 다르면 **raise** (재사용 금지).
5. 저작 덱 파일명을 `dec_FA_<type_id>.inp` 로 유일화하고 `resolve_template` 에 설계별 경로
   오버라이드를 추가.
6. `gd_positions` 를 `designs.json` **필수 필드**로 승격(§6.5)하고 저작 덱 자체를 스테이징(§6.3).

### 6.3 ★ 패키지 재생성 — v1이 통째로 빠뜨린 단계

**v1 §6.2의 "`resolver.paramA_library_dims` 가 덱과 라이브러리를 자동 정합시킨다"는 거짓이다.**
`paramA_library_dims`(resolver.py:70-88)는 `lib/MAS_XSL` COMP 로스터에서 **기대 `(nbatch, ncomp)`
를 계산할 뿐**이고, `validate_reload_deck`(assets.py:296-330)이 그와 다른 `%GEN_DIM` 을 가진 덱을
**거부**한다. 그리고 `_resolve_template`(assets.py:718-760)은 디스크의
`cores/<pair>/<id>/MAS_INP_cy*.inp` 를 **차원 검사 없이 우선 사용**한다.

**실측 확인:** `data/design/package/cores/T6_T4/bootstrap/MAS_INP_cy02.inp:27` 이
`10 10 27 40 42` (= 37 타입) 다. 타입을 2개 더하면 `(42,44)` 가 되어 **`cores/` 10개 폴더가 전부
stale**, 기존 paramA 쌍이 **Popen 이전 게이트에서 하드 실패**한다. 코드가 이 실패모드를
`assets.py:820-833` 에 그대로 문서화하고 있다("a package can carry a STALE bootstrap deck …
wrong `%GEN_DIM`").

**★ 필수 단계 (신규, S5b):**

```
1. 재빌드 후 cores/<pair>/<seed>/MAS_INP_cy*.inp 를 package.write_core_template()(package.py:91)
   으로 **전부 재생성** (새 alias 로스터로).  현재 대상 10개:
   P0_P1, Q1_Q2, Q7_Q8, T1_T4_f117, T3_T4, T5_T6, T5_T6_f101, T5_T6_f117, T5_T6_f81, T6_T4
2. data/design/synth_decks/ 를 **purge** — paramA type_id 공간 쌍 18개가 여기 캐시되어 있다
   (P5849Z1G08N16_P6257Z2G08N16, P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24 등 실측).
3. bases/ 8개 pair 재부트스트랩 (§0.4의 9회 중 8회).
```

**★ `.sum` 과 `dec_FA_*.inp` 는 패키지에 들어가지 않는다.**
`lattice.harvest`(lattice.py:317-347)는 `<caseid>_0101.HGC → FA_<alias>.HGC` 와 `<caseid>.out →
FA_<alias>.out` 만 만들고, `package.stage_hgc`(package.py:60-75)도 `.HGC`/`.out` 만 복사한다.
디스크 확인: `data/design/package/hgc/` = **`.HGC` 37 + `.out` 37, 그 외 0개.**
`fuel_types.py:1678` 이 이 사실을 알고 있다 — *"No .sum in a package → parse_hgc_full covers the
whole curve + coeffs."* 그리고 `ingest_fuel_types` 독스트링은
*"`zone_pin_count` stays NaN unless a `dec_FA_*.inp` is staged"* 라고 적는다.

→ **귀결 두 개:**
(a) **v1 §7.3의 v9 채널 4개는 `.sum` 원천이 없다** → **HGC에서 재유도**해야 한다 (§7.3 재작성);
(b) 작업으로 **`.sum` 과 저작 덱을 스테이징**한다 (`DesignSource.sum_path` 추가, `harvest` 가
`FA_<alias>.sum` 을 남기고 `stage_hgc` 가 `.sum` 과 `hgc/dec_FA_<alias>.inp` 를 복사).
후자는 **핀맵을 산문(`gd_positions` 문자열)이 아니라 바이트로 감사 가능**하게 만든다.

### 6.4 ★ 검증 게이트 (크기 산술 정정 + 신규 3건)

| 게이트 | 검사 | 근거 |
|---|---|---|
| **G-H1 구조** | `%TITL` 334개; `CASE ::` census = REFERENCE 62 + 분지 17×16; `%DIST`/`%MACX`/`%MICX`/`%ADFT` 각 334; 말미 `%FINE` 1 | §5.2 |
| **G-H1b 크기** | `n_gd ∈ {12,16,20,24}` → **정확히 7,395,955 B**. 그 외 값은 **FAIL이 아니라 ABSTAIN + 수동 검토** (`n_gd=0` 의 6,867,567 B 는 기전 미상, `n_gd ∈ {4,8}` 은 생산된 적 없음) | 전 라이브러리 실측 |
| **G-H1c 유효성** | `_hgc_looks_valid`(lattice.py:374) | 기존 |
| **G-H2 Gd 인구조사** | `count_gd_pins_from_hgc`(**`lpopt/data/fuel_types.py:554`**, 호출부 realize_lat1600.py:336) == 요청 `n_gd` | ★ v1의 인용 위치 오류 정정 |
| **G-H3 크기 산술 ★정정** | `MAS_HFF == 404,857 · N` (**등식**) ∧ `MAS_XSL == 2,010 + 385,849·N_(n_gd>0) + 377,461·N_(n_gd=0)` (**등식**) | **아래 6개 N에서 정확 일치 확인** |
| **G-H3b 핵종 로스터 ★신규** | 새 `COMP FA_<alias>` 블록의 핵종 로스터가 기존 블록과 **정확히 일치**(`BP01*`/`SB10*`/`MACX*`/`CRD1*` 포함) ∧ 헤더가 `62 17 6 0 0` | 개수만 세면 `irod=2` 가 먹는 필드를 못 본다 |
| **G-H3c COMP 순서 ★신규** | `LibraryBuild.set_names` 의 **기존 prefix 순서가 불변**이고 신규 별칭이 뒤에 붙는다 | 별칭 순서 위험(§9 R21) |
| **G-H4 스크린 대조 (회귀검사)** | BU ≥ 0.2 전 구간 \|k_(A) − k_DeCART\| ≤ **100 pcm**; \|FF_(A) − FF_DeCART\| ≤ **0.0021** | T3–T6 실측 최대치 |
| **G-H5a cy1 덱 ★분리** | `%GEN_DIM` 차원 ∧ `%LPD_BCH` 로스터 ∧ `%LPD_C&X`/`%LPD_HFF` 이름이 `MAS_XSL`/`MAS_HFF` 에 존재 | **`validate_reload_deck` 을 쓰면 안 된다** — `%LPD_BCH` 를 담은 덱을 구조적으로 거부한다(assets.py:325-327) |
| **G-H5b reload 덱** | cy≥2 덱에 `validate_reload_deck` | 원래 용도 |
| **G-H5c 수렴 ★정정** | "평형 ≥2 사이클"이 아니라 **`bootstrap_max_cycles = 16` + `cy1_cap_efpd` 로 수렴**할 때까지 (`make_band_restart` 가 5-FOM 비교를 2회 안정할 때까지 구동; T6_T4는 11 사이클) | config.py:907,915 |
| **G-H6 덱 에코** | MASTER 출력 에코가 새 별칭을 실제 이름으로 부른다 | HGD569 진단 선례 |

**G-H3 산술 검증 (v2에서 직접 확인):**

| N | 구성 | MAS_XSL 실측 | `2,010 + 385,849·N` | MAS_HFF 실측 | `404,857·N` |
|---:|---|---:|---:|---:|---:|
| 11 | lib/MAS_XSL.bak 세대 | 4,246,349 | 4,246,349 ✓ | 4,453,427 | 4,453,427 ✓ |
| 12 | `0_APR1400/*/hgc` | 4,632,198 | 4,632,198 ✓ | — | — |
| **16** | `2_LP/artifacts/seq_canary` (**V01/V02 가 n_gd=0**) | 6,158,818 | `2,010 + 385,849·14 + 377,461·2` = 6,158,818 ✓ | 6,477,712 | 6,477,712 ✓ |
| 33 | `lib.snap_20260811` | 12,735,027 | 12,735,027 ✓ | 13,360,281 | 13,360,281 ✓ |
| 37 | 현행 `lib/` | 14,278,423 | 14,278,423 ✓ | 14,979,709 | 14,979,709 ✓ |
| 80 | `FEASIBLE_PACKAGE` | 30,869,930 | 30,869,930 ✓ | — | — |

→ **절편은 2,008이 아니라 2,010이다** (2,008 B `MAS_REF` + CRLF). v1의 `2,008 + 385,849·N` 은
**모든 N에서 −2 B 어긋난다.** `n_gd=0` COMP 블록은 **377,461 B** (Δ −8,388 B = 빠진 `BP01*` 표).
**허용오차가 아니라 등식으로 게이트한다.**

**G-H4 실패 시 처분 (유지):** 노심에 쓰지 않고 **능동학습 포인트로 기록**. 단 **(A) 재학습은 이
체크아웃에서 불가능**하다 (학습 매니페스트·헬퍼 스크립트 부재, 부록 B).

### 6.5 등록 (별칭 · 덱 카드 · library_id · 스키마)

**별칭.** `_ALIAS_LETTERS = "PQSTUVWXYZABCDEFGHIJKLMNO"`(**R 제외**, 반사체 R1/R2/R3 충돌 회피)
× 숫자 10 = 250 별칭; `<pkg>/registry.json` 에 영속화. 현행 paramA 37개 (`P0`…`T6`).

**두 개의 토큰 공간 (HGD569 사고의 근원, 유지).** 패턴·`case_pair`·스토어 행·피처는 **type_id 공간**,
MASTER 덱은 **alias 공간**. 번역은 `prepare_cycle1_deck`(assets.py:953), `alias_pattern`(:984),
`alias_case_key`(:1007) 세 곳에서만. 현재 방어: `CaseAssetResolver.__init__` 이 `package_root` 에서
브리지를 **자동 유도**(assets.py:616-619), `validate_reload_deck` 이 Popen **전에** 2자 초과 id와
`%LPD_B&C` 로스터 밖 id를 거부.

> **★ v1 작업 #14의 심각도 하향.** v1은 "`WaveVerifier` 를 `resolver=` 없이 만드는 호출자를
> 하나라도 남기면 사고가 재발한다"고 적었다. **소스는 이미 방어한다** — `verify.py:852-863` 이
> `resolver is None` 이면 `CaseAssetResolver(self.package_root, …)` 를 만들고, 그 주석이
> HGD569를 명시적으로 인용하며 자동 유도 이유를 설명한다.
> → **작업 #14는 사고 방지가 아니라 위생(hygiene)** 이다. **진짜 잔여 위험은 좁다** —
> `registry.json` 이 없는 `package_root`, 또는 명시적 `registry_aliases={}`.
> → **가드를 테스트로 옮긴다: `is_paramA_library(library_id)` 이면 `resolver.type_to_alias != {}`.**

**덱 카드 (타입 1개 추가 = 4곳):**

```
%GEN_DIM  nx ny nz nbatch ncomp     :  nbatch += 1 , ncomp += 1   (37 → "10 10 27 40 42")
%LPD_B&C  <alias>  -1  25*<idx>  -5 :  1 줄 추가
%LPD_C&X  <idx> FA_<alias> 0        :  1 줄 추가
%LPD_HFF  <idx> FA_<alias>          :  1 줄 추가
```

**library_id — `paramA` 확장 (v1 결정 유지).** 새 id(`odA`)를 만들면
`_REGIME_CYCLE_BURNUP_MWD_KG`(featurize.py:63-66)에 항목이 없어 **library-mean → 22.0 폴백**으로
조용히 피처가 오염된다. 재부트스트랩 비용(9회)이 그 오염보다 훨씬 싸고 눈에 보인다.

**스토어 스키마 — 변경 0줄.** `record_id = sha256(canonical_pattern | library_id | case_pair |
deck_knobs)`; `case_pair` 는 2–5 멤버; `e_core`/`e_split` 은 `case_e_core`(fuel_types.py:1945)가
U-질량 가중으로 파생.

**`designs.json` 스키마 확장 (유일한 스키마 변경).**
`gd_positions` 를 **선택 → 필수**로 승격 (현재 37행 중 4행에만 존재). 추가 필드:
`provenance`("on_demand_r<N>"), `screen_ff`, `screen_k0`, `screen_crossing_bu`, `screen_model_sha`,
`screen_pattern`(PA/PB — §4.1a), `decart_wall_s`, `hgc_sha256`, `deck_sha256`.
선례: `realize_lat1600_result.json`, `validation_manifest.json`.

### 6.6 ★ 호스트 간 패키지 동기화 (v1은 `lib/` 만 보냈다)

`paramA_rows`(fuel_types.py:1625-1690)는 `paramA_root` 아래의 **`designs.json` 과 `hgc/FA_*.out`
둘 다**를 필요로 하고, `ingest_fuel_types` 를 캠페인 호스트에서 **다시 돌려야**
`data/store/fuel_types.parquet` 가 갱신된다.

> **동기화 세트 (SHA-256 매니페스트 동반):** `lib/`, `hgc/`(`.HGC` + `.out` + 신규 `.sum` +
> `dec_FA_*.inp`), `registry.json`, `designs.json`, `cores/`(재생성본).
> **재-ingest 를 슬라이스의 명시 단계(S6b)로 둔다.**

---

## 7. ⑤ 캠페인에서의 사용

### 7.1 새 타입 행이 스토어에 들어가는 경로 (신규 코드 0, 유지)

`[case] pair = P<new>_<partner>`, `[model] library_id = "paramA"`,
`[verify] package_root = "data/design/package"`, `[produce] template_fallbacks = []` 로 캠페인을
돌리면 표준 경로가 그대로 행을 쓴다.

**콜드스타트.** 브랜드 뉴 pair는 스토어 행 0개로 시작하므로 `elite_frac 0.65` 가 굶는다
(J5_J6 선례). 등록된 해법은 **32-체인 고정-LP 전이 스윕**(`fr_transfer.py --target-pair … --k 32`)
을 먼저 병합하는 것 → **온디맨드 라운드의 필수 선행 단계 (S8).**

**자산 사다리.** `CaseAssetResolver.resolve`(assets.py:912-951): 0 native → 1 promoted →
2 same-pair nearest feed → 3 nearest-`e_core` pair → 4 configured neutral. `strict_restart=True`
는 레벨 ≥3 거부. **새 pair는 레벨 0 native 를 부트스트랩으로 만들어 주는 것이 원칙**
(레벨 3 폴백은 intervention r1 ga80 사고의 원인이었다).

**★ 캠페인 설정 추가 (MTC 게이트 전제).** `post_verify_delivery` 는 **후보 자신의 수렴 restart** 를
요구하고, 그것이 없으면 후보를 SKIP한다("no retained converged work dir (verify.keep_success /
harvest_maps off?) — candidate is not licence-verifiable", campaign.py:3005-3007).
→ **슬라이스 캠페인은 `[verify] keep_success = true`, `harvest_maps = true` 로 돌린다.**

### 7.2 목적함수 — 변경 없음

`score_min_fxy`(acquisition.py:841) 그대로. 목적은 **min(max F_xy), F_r 은 제약** (사용자 확인
2026-08-30). r2 판정에 따라 **r3 `model_dir = s1i`** (s1j f_xy 헤드는 RANK 게이트 3/3 실패로
레벨 추정기 강등; within-cell ranking 조항이 G1–G4에 들어가기 전 새 헤드의 wave 랭킹 금지).

### 7.3 ★ 재학습 — cond_schema v9 (원천을 `.sum` → HGC 로 교체)

**설계 원칙 (v5 교리 연장):** **핀맵에서 유도되는 채널은 만들지 않는다.** 모델은 원인(Gd 위치)이
아니라 **결과(곡선 형상·FF 곡선·대조)** 를 본다.
**이유는 순전히 데이터다.** ga80은 `gd_wt` 0%·`zone_pin_count` 전무이고(arm 4 Amendment D가 후보
(c)를 이 이유로 기각), ga80 HGC는 80개 중 36개만 디스크에 있다. 핀맵 유래 채널을 넣으면 ga80
스토어 행이 전부 부재-센티넬로 떨어져 **v4 꼬리 붕괴(cyclen 40–50 EFPD 과소예측)를 재현**한다.

**★ v1 대비 원천 변경.** v1은 네 채널을 `.sum` EDIT 2/3에서 뽑겠다고 했으나 **`.sum` 은 어느
패키지에도 없다** (§6.3). → **HGC에서 유도한다:**

| 신규 슬롯 채널 | 정의 | **v2 원천** | ga80 가용 |
|---|---|---|---|
| `origin_ff_hot` | FF 를 BOC 스칼라가 아니라 **고온 BU 창(10–25 GWd/tU) 평균**으로 | **HGC `%DIST` map-1 의 BU별 핀출력 → FF(BU)** | 36/70 |
| `origin_ff_slope` | FF(BU) 의 BU 0→20 기울기 | 동상 | 36/70 |
| `origin_ff_bu_peak` | FF 최대가 나타나는 BU | 동상 | 36/70 |
| `origin_rho_mid` | BU 0.5–8 구간 평균 rho | **HGC `%TITL` 2행 k-inf → rho(BU)** | 36/70 |
| **신규 전역** | | | |
| `g_fresh_contrast` | 68-슬롯 역할과 53-슬롯 역할의 평균 rho 차 (BU 0.5–8) | 위에서 파생 | 파생 |

`g_fresh_contrast` 가 **가장 값이 크다** — node_peak을 R² 0.866으로 설명하는 유일한 측정된
집합체 변수이고 (OPSCREEN §6), 지금 인코더의 전역 13/18/20 어디에도 역할 대조 항이 **없다**.

**★ "손해가 0" 표현 정정.** v1은 "k-곡선/FF-곡선 유래 채널의 ga80 커버리지는 36/70으로 오늘과
동일 — 손해가 0"이라 적었다. 정확히는 **네 신규 채널이 ga80 70개 중 34개에서 결손**이고 그 슬롯은
신규 채널에서 0.0 센티넬을 받는다 — v4 꼬리붕괴와 **같은 기전, 34/70 규모**(70/70이 아님).
→ 표현을 **"기존 곡선 채널과 동일한 34/70 결손 (증가분 0)"** 로 바꾸고, **v9 prereg의 G1
무회귀 게이트를 ga80 부분집합에서 별도로 평가**한다.

**v9 규칙 (기존 관행):** `CHANNELS_BY_SCHEMA`(featurize.py:623)에 `"v9"` 추가, v6c/v7/v8 항목
불변(체크포인트 보호); `_COMPOSITION_WIDTH_BY_FLAG` 는 v8의 5-wide 승계; 부재 시 `_norm_opt` 0.0
센티넬 + `origin_kconv_present` 게이트; 정규화 상수는 수확 모집단의 중앙값/robust half-span;
재학습은 **사전등록 후**, 게이트는 G1 무회귀 + 신규 축 개선.

**등록된 전제조건:** v9는 온디맨드 타입 행이 충분히 쌓인 뒤에만 의미가 있다. 라운드 1
(2–4 타입 × 100 콜)로는 부족하다 → **v9는 라운드 2 이후 과제.** 라운드 1은 **cond v8 + s1i** 로
그대로 돈다 (새 타입은 코드 변경 없이 채점된다).

### 7.4 ★ 성공의 측정 — 사전등록 게이트 (전문, MTC 추가)

**Incumbent 두 개를 구분한다.**

| | 값 | 정체 | 상태 |
|---|---|---|---|
| 프로그램 joint-clean 최소 `F_xy` | **1.5295** | `a785eded…`, `fpcamp_minfr_199`, E1_E2/f121, ga80. `F_r` 1.4694 · CBC 1330.81 · `F_q` 1.8422 · \|AO\| 0.0404 · cyclen 638.639 · assy BU 56.572 | **핀 측정 진행 — r2 phase-2 웨이브 2026-09-03 02:07 완주(30/30), 수확 pending** |
| **deliverable** 최소 `F_xy` | **1.5322** | `bf3a70b2…`, T6_T4/f121, paramA. `F_r` 1.4857 · CBC 1337 · `F_q` 1.853 · \|AO\| 0.024 · **핀 63.760 측정** · cyclen 622.1 | **전 축 측정·한계 내** |

**마크:**

| 마크 | 요건 |
|---|---|
| **PRIMARY** | 온디맨드 타입 ≥1개를 fresh로 쓰는 MASTER 검증 노심이 **측정** `F_xy ≤ 1.5245` (= 1.5295 − 0.005) **이고** `F_r ≤ 1.55` ∧ `CBC ≤ 1600` ∧ `F_q ≤ 2.41` ∧ \|AO\| ≤ 0.30 ∧ **측정** pin BU ≤ 80 ∧ **★ `post_verify_delivery` 가 MTC ∈ [−54, +9] pcm/°C 로 PASS** |
| STRETCH | 측정 `F_xy ≤ 1.5195` |
| SECONDARY | 측정 `F_xy < 1.5322`. **라벨 표기, headline 금지** — 0.005 미만은 close-out 논리상 진보로 세지 않는다 |
| **NULL (등록)** | 예산 안에 `F_xy < 1.5295` 인 클린 행 **0건** → 연료 설계축도 LP 축과 같이 이 프로그램의 `F_xy` 바닥(≈1.53)을 옮기지 못한다 = **바닥은 플랜트 수준 성질**. 귀결: **집합체 축 폐쇄** (승계 레버는 슬라이스 결과를 보고 결정, §10 A7) |
| **보고 전용** | **SDM** — 아래 참조 · 신경망 헤드 예측(그림자) · 사슬 예측 |

> **★ SDM 은 오늘 게이트가 될 수 없다 (physics 비평의 절반 반려).**
> `campaign._rod_model()`(campaign.py:3012-3023)이 **의도적으로 `None`** 을 반환한다 —
> *"The campaign deck is QUARTER-core and carries no `%ROD_CFG`/`%ROD_MAP` … lpopt has no
> full-core asset package, so this returns `None` until one exists — and the gate then reports
> SDM as INCONCLUSIVE rather than inventing a rod map."*
> `[constraints]` 도 `sdm_enable = False` 가 기본이고 `sdm_gated` 는 한계값이 있어야 참이 된다.
> → **SDM 은 라운드 1에서 INCONCLUSIVE 로 기록하고, "MOCHA 의 `build_apr1400_rod_model` +
> full-core 평형 체인을 lpopt 자산으로 이식"을 별도 작업 #20 으로 뗀다.**
> **MTC 는 오늘 바로 게이트가 된다** — 전용 자산이 필요 없고 `%EXE_RHO` 두 점 차분만 쓴다.
> 설정: `[constraints] mtc_enable = true`, `mtc_min_pcm_per_c = -54`, `mtc_max_pcm_per_c = 9`;
> `[sdm_mtc] top_k = 5`, `mtc_delta_c = 5.0`, `branch_timeout_s = 300`.
> 예산은 `status.json` 의 `post_verify_master_calls` 로 **탐색 예산과 분리해서** 보고한다
> (모듈이 "추정이 아니라 계수"라고 명시).

**필수 대조군 (등록).** 새 타입은 연료 **와 셀**(e_core, pair, restart, elite pool)을 **동시에**
바꾼다. 짝지은 대조 없이는 귀속이 불가능하다 (r1의 grading-vs-seeding 교란 선례).
→ **같은 주, 같은 챔피언(s1i), 같은 예산(100콜), 같은 목적으로 incumbent 셀(E1_E2/f121)에서
arm 을 하나 더 돌리고 difference-in-differences 로 보고한다.** (총 200 MASTER 콜, §0.4에 계상.)

**★ 검정력 / 잡음 — v1의 "재시작 잡음 0" 표현 정정.**
`ΔF_xy = 0.0000` (r1 phase-2 40/40, pinbu r1 M2 20/20, 합 60/60)은 **replay 재현성**이다 —
`pinbu_wave_minfxy_r1_results_20260830.md:101` 이 *"every `F_xy` number … is replay-exact"* 라고
쓰며 **동일 덱·동일 restart** 임을 명시한다. 그런데 **이 설계의 중심 행위(라이브러리 재빌드)가
모든 restart를 새로 만든다.**
→ **restart 변경 민감도는 미측정이다.** 그것을 재는 것이 **부정 통제**의 역할이다:

> **부정 통제 (등록).** 새 타입을 **쓰지 않는** 패턴도 새 라이브러리 위에서 돌게 된다.
> 그런 행들이 예전보다 좋아지면 그것은 연료 효과가 아니라 **재부트스트랩/재시작 효과**다
> → **그 크기를 먼저 빼고 이득을 계산한다.** 이 값이 곧 restart-변경 잡음의 측정치다.

**기저율.** r2가 정해 준다 — **100 콜 LP 탐색이 `< 1.5295` 를 0/99 로 못 찾았다.** 온디맨드
타입이 같은 예산에서 그것을 찾으면 그 자체로 강한 증거다.

---

## 8. ⑥ 구현 태스크 · 최소 수직 슬라이스

담당은 에이전트 트랙으로 표기한다 (코딩은 Opus 5 medium-effort 에이전트 위임, 메인 세션은
오케스트레이션 — `coding-agent-preference` 메모리).

### 8.1 태스크 목록 (v1 19개 → v2 21개; 우선순위 재배열)

| # | 트랙 | 작업 | 파일 범위 | 테스트 | 선행 |
|---|---|---|---|---|---|
| **0** | **T-SCR** | **★ 소급검증 — `σ_chain,paired` 측정** (§3.4). 238, 읽기 전용, DeCART/MASTER 0 | 신규 분석 스크립트 + 리포트 1 | leave-one-arm-out 재현성; 헤드 그림자 점수 동시 산출 | — |
| 1 | **T-LAT** | **핀맵 저작 승격** → `lattice.author_gd_layout(...)` + `octant_to_full` + `FuelDesign.gd_positions` + 레이아웃 태그 `type_id` + 파일명 유일화 + `n_gd ∈ {12,16,20,24}` 서브트리 규칙 | `design/lattice.py`, `design/spec.py:47,94,103` | ① 인구조사/안내관/Chebyshev≥2/옥탄트대칭/BOM-free 각 1 ② **기존 37 type_id 라운드트립 불변** ③ 같은 (gd,n,z) 두 레이아웃 공존 | — |
| 2 | T-LAT | `nxfile` 호스트 재작성 + exe/dll 프리플라이트(직렬 폴백) + exe/XS SHA-256 대조 — **큐 ps1에서 이 두 성질만 이식**(§5.4) | `lattice.launch_decart` | 없는 nxfile 로 fail-fast; dll 부재 시 직렬 폴백 | — |
| 3 | T-LAT | `xenon TR` 상수화 | `lattice.py` | 덱 diff | A3 |
| 4 | **T-SCR** | 신규 `lpopt/design/screen.py` — (A) 경로 임포트, `Engines`+`predict_cases`, **경계(bounds)만 강제**(§3.3 F3), `Z_TO_PATTERN={"z1":"PB","z2":"PA"}` 어서션 | 신규 ~300줄. (C) `surrogate_adapter.py:188-215` 지연 임포트 패턴 재사용 | (A) 부재 시 graceful degrade; `--self-test` 허용오차 통과; **z1→PB 어서션** | — |
| 5 | T-SCR | OPSCREEN 사슬 이식 (contrast/node_peak/F_r/ratio/opmodel) — **floor 라벨을 코드에 새긴다** | `screen.py` 순수 함수 | T3–T6 값 재현 (FF 1.1073/1.1409/1.1012/1.1011) | 4 |
| 6 | T-SCR | 다양성/중복제거 + 역할쌍 contrast ≥ 0.026 | `screen.py` | 합성 로스터 결정성 | 5 |
| 7 | T-TRG | 필요신호 `N`(T-A) — `FuelVec` 섭동 유한차분, geometry 채널 제외 | 신규 `design/need.py` | 채널 순열 불변; OOD 채널 제외 | 4 |
| 8 | T-TRG | 가상 로스터(T-B) — `build_pair_universe(types=...)` 주입 | `construct.py` 읽기전용 + `need.py` | 가상 타입의 e_core 밴드 | 7 |
| 9 | T-TRG | **발사 규칙 F1–F5** + 판정 JSON. **F1은 task #0 산출물을 읽는다** | `need.py` | 다섯 조건 각각의 거부 케이스 | 0, 8 |
| 10 | **T-RUN** | `run_batch` 파라미터화 (`design_wave.json`, manifest에 `design` 필드), `decart_timeout 7200` | `lattice.py`, config | dry-run | 1,2 |
| 11 | T-RUN | 게이트 G-H1/G-H1b(ABSTAIN 규칙)/G-H1c/G-H2/G-H4 | 신규 `design/hgc_gates.py` | 정상 HGC PASS, 절단 HGC FAIL, n_gd=0 ABSTAIN | 10 |
| 12 | T-LIB | 재빌드 전 `lib/`+`bases/`+`cores/`+`registry.json`+`designs.json` E: 스냅샷 자동화 | `design/library.py` 래퍼 | 스냅샷 존재 확인 후에만 빌드 | — |
| 13 | T-LIB | `designs.json` 스키마 확장 (`gd_positions` 필수 + provenance/screen/sha 필드) | `design/package.py:40` | 구 매니페스트 하위호환 | 1 |
| **13b** | **T-LIB** | **★ `.sum` + 저작 덱 스테이징** — `DesignSource.sum_path`, `harvest` 가 `FA_<alias>.sum` 보존, `stage_hgc` 가 `.sum`/`dec_FA_<alias>.inp` 복사 | `lattice.py:317`, `package.py:31,60` | 패키지에 `.sum` 37+N, `dec_FA_*` N | 1 |
| **13c** | **T-LIB** | **★ 패키지 재생성** — `cores/` 10개 `write_core_template()` 재생성 + `synth_decks/` purge (§6.3) | 신규 `design/package.py` 헬퍼 | 재생성 후 `validate_reload_deck` 이 새 dims로 통과 | 12 |
| 14 | T-LIB | `WaveVerifier(resolver=None)` — **위생 정리 + 테스트 가드**(§6.5) | 6 호출자 + 신규 테스트 | `is_paramA_library` 이면 `type_to_alias != {}` | — |
| 15 | T-LIB | G-H3 / G-H3b / G-H3c / G-H5a / G-H5b / G-H5c | `hgc_gates.py` + 부트스트랩 래퍼 | 6개 N의 크기 등식; 로스터 일치; COMP prefix 불변 | 11,12 |
| 16 | T-MOD | `enforce_new_type` 배선 — **`enr_main`+`enr_zone` 동시 전달** + `octant_to_full` | `screen.py`/`spec.py` | **0.92 가 `ComplianceError` 를 낸다** (채워진 값 아님) | 1, A2 |
| 17 | T-MOD | ga80 곡선 커버리지 감사 (36/70 재확인) — v9 전제 | 읽기 전용(238) | — | — |
| 18 | T-MOD | cond v9 사양서 + 사전등록 (**HGC 유도**, §7.3) | `featurize.py` + prereg md | 채널 목록 동결 | 17, 라운드1 종료 |
| 19 | T-DOC | 라운드 1 사전등록 (§7.4 → 정식 prereg) | `data/reports/assembly_on_demand_r1_prereg_*.md` | — | §10 확정 |
| **20** | **T-LIC** | **★ SDM 전제조건** — MOCHA `build_apr1400_rod_model` + full-core 평형 체인을 lpopt 자산으로 이식 → `campaign._rod_model()` 이 실제 모델을 반환 | `search/sdm_mtc.py` 소비자 + 신규 자산 | SDM 이 INCONCLUSIVE 를 벗어난다 | — (라운드 1 비차단) |
| **21** | **T-DOC** | **★ MTC 게이트 배선** — `[constraints] mtc_enable/min/max`, `[verify] keep_success/harvest_maps` | 슬라이스 캠페인 덱 | `post_verify_master_calls` 가 status.json에 계수됨 | 19 |

### 8.2 ★ 최소 수직 슬라이스 — 후보를 **완전 지정**한다

**v1의 결함 두 개.** (a) 후보 "Z"를 레이아웃만으로 적어 S1/S2가 실행 불가였다;
(b) ΔFF −0.018을 **ga80 E4(1.1390)** 대비로 인용했는데 §7.4의 PRIMARY incumbent 는
**E1_E2/f121의 `a785eded`** 다 (E1 FF 1.146 / E2 1.152). 둘 다 고친다.

#### 8.2a 후보 쌍 — 완전 지정 (OPSCREEN.md:319, 402-410 + §4.2 포락 정정)

| | **Z1 (68-슬롯 역할)** | **Z2 (53-슬롯 역할, hot)** |
|---|---|---|
| `u_high` (e1) | **5.50 w/o** | **5.00 w/o** |
| `e2` | **4.70 w/o** ★ (OPSCREEN 원안 4.6750에서 +0.025) | **4.25 w/o** |
| `du` = e1 − e2 | **0.80** — (A) 경계 상한 **정확히 위**, 0.1 격자 위 | **0.75** — 경계 안, 격자 밖 (T3/T4 선례) |
| ratio e2/e1 | **0.8545** (준법 0.85 ± 0.03 **통과**, \|Δ\| = 0.0045) | **0.8500** (정확) |
| `gd_u` | 4.0 (고정) | 4.0 (고정) |
| `gd_wt` | **8** | **10** |
| `n_gd` | **20** | **20** |
| `zoning_variant` | **z1** → **(A) pattern = PB** ★ | **z1** → **PB** ★ |
| Gd 레이아웃 (옥탄트) | **`1:1;4:1;6:4`** | **`1:1;4:1;6:4`** (동일) |
| 베이스 덱 | `0_APR1400\5.8_5.1\FA\IGD_20\8_20_z1\dec_FA_B03.inp` (동결 `2:2;5:2;6:4`) | `0_APR1400\5.8_5.1\FA\IGD_20\10_20_z1\dec_FA_B05.inp` (동결 `2:2;5:2;6:4`) |
| 핀 이동 | `2:2 → 1:1`, `5:2 → 4:1` (`6:4` 유지) — **2 이동** | 동일 2 이동 |
| UO2G 캐리어 밀도 | 9.95 g/cc (gd_wt 8 패밀리) | 9.88 g/cc (gd_wt 10 패밀리) |
| 예상 `type_id` | `P5547Z1G08N20` (+ 레이아웃 태그) | `P5042Z1G10N20` (+ 레이아웃 태그) |
| 기존 37 id와 충돌 | **없음 (확인함)** | **없음 (확인함)** |
| 저작 출력 트리 | `5_RL/templates_lat1600/5.8_5.1/FA/IGD_20/8_20_z1/` (**신규 디렉터리**) | `.../10_20_z1/` (**신규 디렉터리**) |

**★ 왜 e2 를 4.6750 → 4.70 으로 올렸는가.** OPSCREEN 원안 `u5.50/4.6750` 은 `du = 0.825` 로
**(A)의 `du` 경계 [0.40, 0.80] 밖**이다 (§4.2). `validate_design` 이 거부하고, 경계 밖 외삽은
`n_gd = 28` 선례(+1,750 pcm)가 금지한다. **`e2 = 4.70` 이면 `du = 0.80` 으로 경계·격자 위에
정확히 앉고 준법 허용오차 안에 있다.** 대안 두 가지는 §10 A2에 오버라이드로 올린다 —
(i) ratio 를 0.85로 엄격 고정 → `u_high ≤ 5.333` → **Z1* = u5.30 / e2 4.505**;
(ii) OPSCREEN 원안 유지 → **(A) 스크린 불가**(soft `--inp` 모드의 경고 경로만 남음).

**★ 스크린 값은 재계산 대상이다.** OPSCREEN.md:319의 Z 행
(cyclen raw 622.5 / corr 624.8, CBC 1397, FF_hot 1.1208, 상대역 FF 1.1202, **contrast +0.0300**,
`d_fresh` 0.0067, `Fr_flr` 1.395, ΔFF −0.018, ΔF_r −0.022)은 `e2 = 4.6750` 에서의 값이다.
+0.025 w/o 변경은 스크린 분해능 이하로 보이지만 **S1에서 정확한 튜플로 재스크린**한다.
DeCART 등가 FF는 **1.1222 / ΔFF −0.017** 이다 (서로게이트가 0.0014 낮게 나온다, OPSCREEN.md:352-353).

#### 8.2b ★ 사슬로 본 슬라이스의 기대값 — **부호가 정해지지 않는다**

incumbent `a785eded` 는 E1_E2/f121, 측정 `F_r` 1.4694 / `F_xy` 1.5295. E1_E2 의 역할 대조는
**+0.049** (OPSCREEN.md:427), hot 역할 E2 의 `FF_hot` 은 **1.1520**, 그 셀의 fr_arms 기준점
A0 의 **측정** `node_peak` 은 **1.2085**.

**(a) 사슬을 양쪽에 일관되게 적용하면 (회귀 `node_peak`):**

```
node_peak(Z)      = 1.4210 − 4.1725(0.0300) − 3.4862(0.0067) = 1.2725
node_peak(E1_E2)  = 1.4210 − 4.1725(0.049)  − 3.4862(0.0023) = 1.2085   (= 측정치와 일치, 자기정합)
product(Z)        = 1.2725 × 1.1208 = 1.4262
product(E1_E2)    = 1.2085 × 1.1520 = 1.3922
ΔF_r  = 1.035 × (1.4262 − 1.3922) = **+0.035  (Z 가 나쁨)**
ΔF_xy ≈ 1.059 × 0.035            = **+0.037  (Z 가 나쁨)**
```

**(b) floor 단위로 보면 (양쪽 `node_peak` 을 1.2085로 동결 = OPSCREEN 의 `Fr_flr`):**

```
ΔF_r  = 1.03 × 1.2085 × (1.1208 − 1.1520) = **−0.039  (Z 가 좋음)**
ΔF_xy ≈ **−0.041**
(참고: OPSCREEN §8 의 −0.022 는 같은 식을 ga80 E4 1.1390 에 대해 쓴 값이다.)
```

> **★ 이것이 이 문서의 핵심 숫자다.** 같은 사슬이 같은 후보에 대해
> **[−0.041, +0.037]** 의 구간을 준다 — **0을 넉넉히 포함한다.**
> 차이는 오직 **`node_peak` 을 LP가 회복하느냐**이고, OPSCREEN.md:253-255 는 그 회복이
> **가정이지 예측이 아니라고** 명시한다. Z의 contrast 0.0300 은 게이트(0.026)는 넘지만
> incumbent(0.049)보다 **낮고**, contrast 0.026–0.028 밴드의 arm 들은 `node_peak` 1.274–1.327 로
> 측정되었다(≥0.043 밴드는 1.209–1.260). **Z는 그 낮은 밴드에 앉는다.**
> → **슬라이스는 "예측된 −0.018을 실현하는 작업"이 아니라 "부호가 미정인 효과를 측정하는 실험"이다.**
> §3.4 task #0가 σ를 재는 이유, §3.5가 서사를 물리실험으로 바꾸는 이유가 여기 있다.

**대조군 arm 은 이 구간 때문에 선택이 아니라 필수다.** 짝지은 차분만이 위 두 해석 사이를 가른다.

#### 8.2c 슬라이스 단계 (v1 S0–S9 → v2 S0–S9b)

```
S0   스냅샷        lib/ + bases/ + cores/ + registry.json + designs.json → E:\lpopt_archive\     (#12)
S0b  ★ task #0    σ_chain,paired 측정 (238, 읽기전용).  DeCART 0 / MASTER 0.                      (#0)
                  ★ 이 결과가 S9 의 판정 문장을 정한다 (예측 실현 vs 물리 측정)
S1   스크린 1회    (A) 로 Z1/Z2 정확 튜플의 kconv/FF/contrast 예측, opmodel 운전점               (#4,#5)
                  게이트: cyclen ∈ [620,645] ∧ CBC ≤ 1500 ∧ contrast ≥ 0.026 ∧ **bounds 통과**
                  ★ pattern = PB (z1) 를 어서션.  du = 0.80 / 0.75 를 로그에 남긴다
S2   덱 저작       author_gd_layout 로 dec_FA_<type_id>.inp 2개, 신규 디렉터리 8_20_z1 / 10_20_z1  (#1)
                  가드 전부 통과, octant_to_full → enforce_new_type(enr_main, enr_zone)           (#16)
S3   DeCART       181 / C: 루트 / 직렬 / 상한 4 / timeout 7200 s → FA_<alias>.HGC/.out/.sum       (#10)
                  예상 wall 2 케이스 ≈ 1.7 h (직렬) — box 104 omp 면 ≈ 0.25 h
S4   HGC 게이트    G-H1 / G-H1b / G-H1c / G-H2 → G-H4 (**회귀검사**, 100 pcm / 0.0021)            (#11)
S5   라이브러리    스냅샷 확인 → TotalBatcher 재빌드 (37 + 2 = 39), G-H3/G-H3b/G-H3c              (#15)
                  기대: MAS_XSL = 2,010 + 385,849×39 = 15,050,121 B ; MAS_HFF = 404,857×39 = 15,789,423 B
S5b  ★ 패키지 재생성  cores/ 10개 재생성 (dims 42,44) + synth_decks purge                          (#13c)
S6   등록          alias 배정 → registry.json / designs.json(gd_positions 필수) → stage(.sum, 덱)  (#13,#13b)
S6b  ★ 동기화·재ingest  패키지 세트 배송 + ingest_fuel_types 재실행 (39 paramA 행)                 (§6.6)
S7   MASTER smoke  G-H5a(cy1) → 부트스트랩 **수렴까지**(max 16 사이클) → G-H5b/G-H5c/G-H6         (#15)
                  ★ 9회: bases 8 재부트스트랩 + Z1_Z2 신규 1
S8   전이 스윕     fr_transfer --target-pair Z1_Z2 --k 32 → 스토어 병합 (콜드스타트 해소)          (기존)
S9   캠페인 2 arm  min_fxy, 각 100콜, s1i, [verify] keep_success/harvest_maps = true               (기존)
                  arm A = Z1_Z2/f121 (신규) · arm B = E1_E2/f121 (짝지은 대조)  = 200 MASTER 콜
S9b  ★ MTC 게이트  post_verify_delivery (top_k 5) → MTC PASS/FAIL, SDM INCONCLUSIVE 기록          (#21)
                  + phase-2 pin 웨이브 (top-k 측정 pin BU) → §7.4 판정
```

**슬라이스가 일부러 하지 않는 것:** 트리거 구현(T-TRG 전부, #7–#9), 다후보 스크린(§4.6),
cond v9(§7.3), SDM 자산 이식(#20).

**중단점 (등록):**
- **S0b 에서 `σ_chain,paired > 0.020`** → 판정 문장을 §3.5의 물리실험 서사로 **교체**하고 계속한다
  (중단하지 않는다 — 파이프라인 가치는 서사와 무관하다).
- **S4 의 G-H4 실패** → **거기서 멈춘다.** 스크린이 실계산을 예측하지 못하면 §4 전체의 전제가
  무너지므로 라이브러리를 건드릴 이유가 없다.
- **S5b 이후 기존 pair 하나가 `validate_reload_deck` 을 통과하지 못하면** → 재생성이 불완전한
  것이므로 캠페인 전에 멈춘다.

---

## 9. ⑦ 리스크 · 미지수 · 해소 방법

| # | 리스크 / 미지수 | 증거 | 영향 | 해소 | 트랙 |
|---|---|---|---|---|---|
| **R1** | **연료 레버가 2차항** — 축 전체가 NULL 일 수 있다 | 24 elite 패턴 `F_r` 중앙값 이득 −0.0069 vs 패턴탐색 −0.057 (8배); OPSCREEN 서열 −0.057 / +0.21 / −0.022 | 프로그램 1.5–2일 × N | §7.4 사전등록 + 짝지은 대조군. **1 라운드로 결정** | T-DOC |
| **R2** | **핀맵 저작 없으면 사전 실패 확정** | OPSCREEN §8: 동결 최선 FF 1.1657 > incumbent 1.1390 | 축 무가치 | 작업 #1 선행 필수 | T-LAT |
| **R3 ★** | **사슬이 후보의 부호를 못 정한다** — [−0.041, +0.037] | §8.2b (OPSCREEN 자체 수치로 유도) | 판정 서사 붕괴 | **task #0** 로 σ 측정 → §3.5 물리실험 전환 | T-SCR |
| **R4** | UO2G 밀도 규약 불일치 0.68/1.02/**1.23**% (셋 다 0.5% 경고 초과) | realize_lat1600.py:232-238 vs predict.py:1071 | 스크린 편향 | **이미 유계**: T3–T6 홀드아웃 ≤100 pcm / ≤0.0021 FF. G-H4는 회귀검사 | T-RUN |
| **R5** | Xe 규약 불일치 (A) EQ vs 템플릿 TR | README §5 vs 덱 OPTION | 스크린 계통오차 | **R4와 동일 유계에 포함.** 라이브러리 내 혼합 금지 | T-LAT |
| **R6** | `DEFAULT_DECART_EXE` 가 omp 인데 dll 이 181/199에 없다 | 디스크 확인 | 실행 실패 | 프리플라이트 + 직렬 폴백(#2). **box 104에는 있다** | T-LAT |
| **R7** | LEU+ 농축 상한의 인허가 근거 부재 | 워크스페이스 0건 | 상한 미정 | §10 A1. **단 실제 구속은 (A) `du` 경계(5.333)** | — |
| **R8** | `enforce_new_type` 이 0.92를 **덮어쓴다** (기각 아님) + 옥탄트 확장기 부재 | compliance.py:308-317, `is_octant_symmetric(n=16)` | 준법 게이트 무력 + 스크린 무효화 | 작업 #16 (두 값 전달 + 확장기) | T-MOD |
| **R9** | ga80 흡수체 기술자 부재 | arm 4 Amendment D | v9 제약 | v9는 곡선 유래만, **HGC 원천**(§7.3) | T-MOD |
| **R10** | 모델의 셀내 `F_xy` 순위 실패 | r2 RANK 3/3 FAIL 등 | 기울기 트리거 불가 | 헤드를 **그림자**로 강등(§3.3) | T-TRG |
| **R11** | 분지 누락 시 MASTER 거동 미검증 | 진술 없음 | 조용한 오계산 | **정책: 생략 금지** + G-H1/G-H3b | T-RUN |
| **R12** | 호스트 경합 | ssh 실측 | 무흔적 소멸 재발(2026-08-30 = 디스크 풀) | 181 + C: 루트 (§10 A4); 199 사용 시 D: 루트 + `workers ≤ 21` + C: ≥ 30 GB | T-RUN |
| **R13** | lat1600 설계표(`screen1600.csv` 5,874)가 이 머신에 없다 | 스크래치패드 세션 삭제 | 후보 풀 재생성 필요 | **저비용** — 라운드 1 격자는 3.7×10³ 설계 = CPU 7분(§4.3). `E:\lpopt_archive\opmodel_20260829\` 에 재스크린 캐시 | T-SCR |
| **R14** | (A) 재학습 불가 (매니페스트·헬퍼 부재) | metadata.json 이 발행 PC 경로 | 능동학습 루프 미폐쇄 | 예측만 사용. 실패 설계는 기록만 | — |
| **R15** | MASTER `ncomp` 상한 미상 | 최대 관측 80 COMP / ncomp 85 | 누적 시 잠재 | 37→39 는 여유 큼. 누적 60 초과 시 프로브 | T-LIB |
| **R16** | 오케스트레이션 주체 | 2026-08-30 병행 세션 사건 | 발사 충돌 | §10 A5 | — |
| ~~R17~~ | ~~PA/PB ↔ z1/z2 미확인~~ | — | — | **★ 종결: `z1 ⇔ PB`, `z2 ⇔ PA` (§4.1a)** | — |
| **R18** | MAS_HFF 61 레코드 vs COMP 헤더 `BURN 62` | 실측 | 기전 미상, 실무 영향 없음 | 기록만 | — |
| **R19** | `decart_timeout 5400 s` 가 직렬 실측 3,084 s 의 1.75배 | config.py:906 | 오탐 타임아웃 | **7200 s** (#10) | T-RUN |
| **R20** | `n_gd = 0` HGC 크기가 다르다 (6,867,567 B), `{4,8}` 은 생산 이력 0 | 실측 | 크기 게이트 오탐 | G-H1b 를 **ABSTAIN** 규칙으로 (§6.4) | T-RUN |
| **R21 ★** | **별칭 순서 위험 (미검증)** | `_ALIAS_LETTERS = "PQSTUVWXYZABCDEFGHIJKLMNO"` 는 `P…Z` 다음 `A…O`; `MAS_XSL` COMP 순서는 TotalBatcher 내부 스캔 순서라 **관측되지 않았다** | 첫 `A*` 별칭(≈101번째 타입)이 앞으로 정렬되면 모든 composition index 재번호 | **G-H3c** 로 `set_names` prefix 불변 어서션 (기전을 단정하지 않고 검사한다) | T-LIB |
| **R22 ★** | **SDM 이 게이트가 아니다** | campaign.py:3012-3023 `_rod_model → None` | 인허가 축 하나가 미검증 상태로 납품 | **INCONCLUSIVE 로 명시 기록** + 작업 #20 | T-LIC |
| **R23 ★** | **같은 `(gd,n,z)` 에 두 레이아웃 공존 불가** | `resolve_template` `sorted(glob)[0]` + `author_template` 고정 파일명 | §4.6 "라운드당 4개" 차단 | 작업 #1 파일명 유일화 + 경로 오버라이드 | T-LAT |
| **R24 ★** | **`type_id` 가 레이아웃 맹목** | spec.py:85-99 | 미래 라운드에서 기존 타입 HGC 덮어쓰기 | 작업 #1 (2)–(4). **슬라이스는 충돌 없음(확인)** | T-LAT |

---

## 10. ★ 오케스트레이터 가정 (v1 §10 "결정 요청" 을 대체)

v1은 7개 결정을 사용자에게 열어 두었다. **v2는 오케스트레이터가 값을 정하고 문서에 기록**하며,
사용자가 뒤집을 수 있는 지점을 명시한다. **어떤 발사도 A5 확정 전에는 없다.**

| # | 가정 (v2가 채택한 값) | 근거 | **사용자가 뒤집으면 무엇이 바뀌나** |
|---|---|---|---|
| **A1** | **LEU+ 농축 상한 = 5.50 w/o** | lat1600 선례; OPSCREEN.md:430 이 "프로그램 결정"이라 명시. **인허가 근거는 워크스페이스에 없다 — 이 값은 가정이다** | 상한을 낮추면(예 5.00) Z1이 탈락하고 후보 풀이 절반 이하로 준다. **높여도 이득 없다** — (A) `du` 경계가 먼저 구속한다(A2 참조) |
| **A2** | **비율은 준법 창 0.85 ± 0.03 안에서 자유**; 슬라이스 Z1은 **0.8545 (e2 4.70)** 로 잡는다 | `zone_ratio_flag` 가 허용오차를 본다(compliance.py:69-71,184); `DESIGN_GRID` 는 **검증기가 아니라 LHS 표본 격자**(spec.py:31-37, `__post_init__` 는 `0<e2≤e1` 만 검사) | **"정확히 0.85"로 조이면** → `du = 0.15·e1 ≤ 0.80` → **`u_high ≤ 5.333`** → Z1을 **u5.30 / e2 4.505** 로 재지정해야 하고 스크린을 다시 돌려야 한다. **0.92를 허용하면** 준법 R1 위반이므로 **인허가 판단이 필요하다** |
| **A3** | **`xenon TR`** (온디맨드 HGC) | 한 MAS_XSL 안 Xe 혼합 미검증; 기존 37 paramA + 80 ga80 전부 TR | EQ로 가면 라이브러리 전체를 EQ로 재생산해야 한다 (117 격자 재계산) |
| **A4** | **DeCART 웨이브 = HOST_181**, 실행 루트 `C:\`, **직렬 exe**(omp dll 부재), 큐 상한 4 (32스레드), timeout 7200 s. **199는 MASTER 전용** | dll 유무 실측; 199는 생산 호스트 | **box 104로 옮기면** omp 4-way가 **실증**되어 있어(735–750 s/case) 2 격자가 **~13분**이고, `0_APR1400` + `templates_lat1600` + paramA 패키지 + MASTER가 **모두 그 박스에 있다**. **오케스트레이터의 권고는 사실 104이며, A4는 보수적 기본값이다** |
| **A5** | **이 세션이 오케스트레이션**; 메모 v2 + 슬라이스 prereg 완료 전까지 **발사 없음** | 2026-08-30 병행 세션 사건 미해결 | 발사 권한을 다른 세션에 주면 **runs/orchestration 영수증 규약을 먼저 확정**해야 한다 |
| **A6** | **슬라이스 우선** — 손으로 고른 **1 쌍(2 격자)** 을 파이프라인 전체에 통과시킨 뒤 4-후보 라운드 | v1 §8.2 유지. **단 "1 격자" 옵션은 폐기** — `contrast` 는 쌍의 성질이고, 단일 신규 격자를 기존 타입과 짝지으면 그것이 T5_T6 실패 모드다 | 바로 4-후보로 가면 R23(레이아웃 파일명 충돌)과 9회 부트스트랩을 동시에 맞는다 |
| **A7** | **NULL 발화 시 승계 레버는 슬라이스 결과를 본 뒤 결정** (v1의 "축 폐쇄 후 대기"에서 완화) | 승계 후보(feed/주기길이 교환, 인허가 기저 변경)의 상대가치는 슬라이스가 주는 `ΔF_xy` 부호·크기에 달려 있다 | 지금 고정하면 측정 전에 레버를 버리게 된다 |
| **A8 ★** | **스크리닝 호스트 = HOST_238** — 단 (A)는 **`USER2` 계정** `/home/USER2/lattice_surrogate/kpin_pa` 에 있고 프로젝트 계정은 `USER@…:8022` 다 | SURROGATE_USAGE.md:153-159 | **접근 확인이 필요하다.** 단 라운드 1 격자는 3.7×10³ 설계라 **CPU 로도 7분**이므로 병목이 아니다 — `USER` 홈에 모델 사본을 두는 것으로도 충분하다 |
| **A9 ★** | **SDM 은 라운드 1에서 INCONCLUSIVE**, MTC 만 게이트 | campaign.py:3012-3023 | SDM 을 게이트로 요구하면 **작업 #20(full-core rod model 이식)이 슬라이스의 선행조건**이 되어 일정이 늘어난다 |

---

## 부록 A — 인용 (파일:라인). ★ = v2에서 새로/다시 확인

**설계 스펙 / 별칭**
- `5_RL/lpopt/design/spec.py:31-37` `DESIGN_GRID` ★ (검증기 아님, LHS 격자) / `:41` `GD_CARRIER_ENR = 4.0`
  / `:47` `FuelDesign` / `:56-70` `__post_init__` ★ (**`n_gd <= 0 or n_gd % 4` raise**)
  / `:85-91` `type_id` ★ / `:94-99` `key` ★ (**레이아웃 맹목**) / `:103-111` `as_dict` ★
  / `:146` `_ALIAS_LETTERS` (R 제외) / `:154` `DesignRegistry` / `:188-200` `alias` ★ (type_id 키잉)
- `5_RL/lpopt/design/package.py:31` `DesignSource` ★ (`hgc_path`/`out_path` 뿐)
  / `:40-56` `write_designs_manifest` ★ (`gd_u_enr = 4.0` 하드코딩, 레이아웃 없음)
  / `:60-75` `stage_hgc` ★ (**`.HGC` + `.out` 만**) / `:91` `write_core_template` / `:112` `ingest_fuel_types` ★

**격자 덱**
- `5_RL/lpopt/design/lattice.py:29` `DEFAULT_DECART_EXE`(omp) / `:32-36` `_TEMPLATE_ROOTS` ★ (**12/16/20 + 24**)
  / `:45` `_dir_name` / `:49-71` `resolve_template` ★ (`sorted(glob)[0]`)
  / `:114-147` `edit_dec_text` ★ (**숫자 3 + CASEID; 밀도 불변**) / `:184` `edit_dec_geom_text`
  / `:297-313` `launch_decart` / `:317-347` `harvest` ★ (**`.sum` 미보존**) / `:374` `_hgc_looks_valid` / `:409` `run_batch`
- `5_RL/realize_lat1600.py:137` `_triangle` / `:165` `_census` / `:170` `_n_gd_of` / `:180-222` `author_template` ★
  (**`dec_FA_lat1600.inp` 고정 파일명**) / `:225-258` `build_template_tree` ★ (**`templates_lat1600` 트리,
  `n_gd ∈ (16,24)` 만, gd_wt별 밀도 10.01/9.95/9.88**) / `:232-238` 밀도 규약 / `:336` `count_gd_pins_from_hgc` 호출부
- ★ `5_RL/realize_lat1600.log:2-30` 저작 트리 3개 / `:31-34` **DeCART wall 750/750/735/735 s**
  / `:35-38` Gd census / `:39-41` union rebuild 37 COMP → ncomp 42
- ★ `5_RL/design_lat1600_104.inp:1-20, 48` **box 104 전용 덱, omp exe 지정**
- `0_APR1400/260624/FA/IGD_20/6_20_z1/dec_FA_B01.inp` 골든 덱 (DEPL 62 + BRANCH 16 + EDIT, `xenon TR`)
- ★ `0_APR1400/5.8_5.1/FA/IGD_16/6_16_z1/dec_FA_A01.inp` 옥탄트 8행 `1 1 1 1 1 1 2 2` → **PB**
- ★ `0_APR1400/5.8_5.1/FA/IGD_16/6_16_z2/dec_FA_A02.inp` 옥탄트 8행 `2 2 2 2 2 2 2 2` → **PA**

**라이브러리 / 등록**
- `5_RL/lpopt/design/library.py:80-150` `build_master_library` ★ (`.bak` 1세대, stale 거부,
  `set_names` 반환) / `:110-115` 부분 재빌드 거부
- ★ 크기 실측: `lib/MAS_XSL` 14,278,423 · `MAS_HFF` 14,979,709 (N=37) ·
  `lib.snap_20260811` 12,735,027 / 13,360,281 (N=33) · `*.bak` 4,246,349 / 4,453,427 (N=11) ·
  `0_APR1400/*/hgc/MAS_XSL` 4,632,198 (N=12) · `2_LP/artifacts/seq_canary/MAS_XSL` 6,158,818 (N=16,
  V01/V02 가 n_gd=0) · `3_GA_Surrogate/FEASIBLE_PACKAGE/lib/MAS_XSL` 30,869,930 (N=80) · `MAS_REF` 2,008 B
- `5_RL/lpopt/design/coredeck.py:389-391` `library_dims` / `%LPD_B&C/C&X/HFF` / `CoreParams.wide 20.7772`
- `5_RL/lpopt/design/bootstrap.py` `make_band_restart` / `library_aliases`
- ★ 패키지 실측: `bases/` **8** (P0_P1, Q1_Q2, Q7_Q8, T1_T4_f117, T3_T4, T5_T6, T5_T6_f117, T6_T4) ·
  `cores/` **10** (+T5_T6_f101, T5_T6_f81) · `hgc/` **`.HGC` 37 + `.out` 37, 그 외 0** ·
  `cores/T6_T4/bootstrap/MAS_INP_cy02.inp:27` **`10 10 27 40 42`** ·
  `synth_decks/` **paramA type_id 쌍 18 + ga80 쌍 11** · `designs.json` **37 type_id, `gd_positions` 4행뿐**

**해석 / 가드**
- `5_RL/lpopt/search/assets.py:296-330` `validate_reload_deck` ★ (**`%LPD_BCH` 거부**, `%GEN_DIM` 대조)
  / `:718-760` `_resolve_template` ★ (**차원 미검사 우선순위**) / `:815-840` stale 덱 문서화 ★
  / `:616-619` 브리지 자동유도 / `:912-951` `resolve` 사다리 / `:953/:984/:1007` 번역 3곳
- `5_RL/lpopt/search/resolver.py:70-88` `paramA_library_dims` ★ (**기대치 계산만**) / `:89-101` `is_paramA_library`
- ★ `5_RL/lpopt/search/verify.py:852-863` `resolver is None` → `CaseAssetResolver(package_root, …)` **자동 유도**
- `5_RL/lpopt/vendor/masterrl/master.py:360-430` MAS_SUM 엄격 파싱

**인허가 (신규 절)**
- ★ `5_RL/lpopt/search/sdm_mtc.py:1-80` 독스트링 (D9, DCD Table 4.3, MTC `[-54,+9]`,
  SDM `10180 + 690 = 10870`, "≈2 extra MASTER calls per candidate, ~20-60 s",
  "flattening monotonically degrades control-rod worth") / `:290-303` `RodModel` / `:321-327` 기본 한계
  / `:1492-1514` `post_verify_delivery` / `:1587` `post_verify_topk`
- ★ `5_RL/lpopt/search/campaign.py:3005-3007` "not licence-verifiable" (keep_success/harvest_maps)
  / `:3012-3023` **`_rod_model() → None`** / `:3078` `rod_model=self._rod_model()`
- ★ `5_RL/lpopt/config.py:1117-1118` `mtc_limit/sdm_limit` / `:1228-1237` `mtc_enable/sdm_enable`
  / `:1243-1251` `mtc_gated/sdm_gated` / `:1272-1303` `[sdm_mtc]` (`top_k 5`, `branch_timeout_s 300`)

**목적함수 / 납품 / 준법**
- `acquisition.py:489/841/1054/1799` · `delivery.py:43/50/54-58/179`
- `fuelcost_search.py:97/166/195/269/294/318` ★ (시그니처 확인)
- `construct.py:748/770/847` ★ (시그니처 확인)
- `compliance.py:69-73` 상수 / `:184` `zone_ratio_flag` / **`:282-330` `enforce_new_type` ★
  (`enr_zone` 부재 시 채움, `pin_map` 은 16×16 요구; 호출자 테스트 외 0개)**
- `data/schema.py:44-58` `record_id` · `data/fuel_types.py:554` `count_gd_pins_from_hgc` ★
  / `:1477` "디렉터리 이름이 정본" ★ / `:1625-1690` `paramA_rows` / `:1678` "No .sum in a package" ★ / `:1945` `case_e_core`
- `model/featurize.py:63-66/349-411/454-480/623-637/644/651-654/661-672/709/725/790-825/838-908`
- `model/ood_guard.py` (geometry 채널 퇴화)

**DeCART 서로게이트 (A)**
- `6_DeCART_Surrogate/surrogate/predict.py:67-71` **`FIXED`/`ZONING_COMMON`/`ZONING_BY_PATTERN`** ★
  / `:107-121` `GRIDS`(**`du` 0.4–0.8**)/`GD_WT_VALUES`/`N_GD_VALUES`/`MIN_CHEB` ★
  / `:124-133` `_snap_to_grid` ★ (**경계 밖 = 오류**) / `:147-215` `validate_design` ★
  / `:497` `kcurve_case_features` / `:899` `Engines` / `:941/961` `predict_case(s)`
  / `:1058-1085` `inp_row_warnings` ★ (**soft 모드**) / `:1070-1074` **0.5% `gd_density` 경고** ★
- `SURROGATE_USAGE.md:45-46` CPU 8.6 case/s ★ / `:148-165` **서버 = HOST_238, user `USER2`** ★
  / `:196-225` 541 case/s, 로컬 3080 Ti 220 case/s ★ / `:249-250` **Gd 밀도 내부 규약 9.942/9.850/9.760** ★
  / `:288-300` **89 배치 = n12:18 + n16:24 + n20:28 + n24:19** ★ / `:325-342` §7.7 (**GA 시대 덱, 10.01 고정, +1.6%**) ★
- `README.md:1-6` (범위) / §5 (고정조건: boron 500 / **xenon EQ** / PA zoning) / §6 (정확도)
- `2_LP/MOCHA/surrogate_adapter.py` 독스트링 11-13 (advisory only) · `2_LP/MOCHA/config.py:349-358`
  (`surrogate_advisory_enabled=False`, **`surrogate_pattern = "PA"` 기본값** ★)

**운전점 / 물리 사슬 (OPSCREEN.md, 전부 ★ 재확인)**
- `:150-161` CBC ±70 ppm(90%) / `:165-179` **T3–T6 홀드아웃: k <100 pcm (BU≥0.2), FF bias −0.0014 /
  rms 0.0015 / max 0.0021; s08_transfer ≤0.5 EFPD, ≤8 ppm**
- `:200-212` 5,874 설계 재스크린, 게이트 cyclen [620,645] ∧ CBC ≤1500 / `:215-235` **fr_arms 17 arm 표
  (B3 T5_T6: FF_hot 1.1020, 측정 F_r 1.5795; B2 T3_T4: 1.1430 / 1.5329)**
- `:238-246` contrast 밴드표 (≥0.043 → node_peak 1.209–1.260; 0.026–0.028 → 1.274–1.327; ≈0 → 1.387–1.551)
- `:246-250` **`node_peak = 1.4210 − 4.1725·contrast − 3.4862·d_fresh`, rms 0.036, R² 0.866, n=15;
  `A = 1.035 ± 0.031`** / `:249-251` 32 elite 패턴 회복 실패 / **`:253-255` floor 면책조항**
- `:319` **Z 행** (cyc 622.5/624.8, CBC 1397, FF_hot 1.1208, 1.1202, contrast +0.0300, d_fresh 0.0067,
  Fr_flr 1.395, ΔFF −0.018, ΔF_r −0.022) / `:340-356` §8 레이아웃 판정표 + **DeCART 등가 1.1222 / −0.017**
- `:358-410` §9 웨이브 가격 (**"8 existing + 1 new pair = 9 bootstraps"**, 8,744 s 실패,
  `.bak` 1세대, 2 격자 ~13분, **Z1/Z2 튜플과 베이스 덱**, **"PB set 검증"**) / `:413-436` §10 (**5.00–5.50 = 프로그램 결정**)
- `s02_surrogate_vs_decart.py`, `s09b_contrast.py`, `s08_transfer.py`, `s10_screen.py`, `s14_screen_final.py`

**계보 / 판정**
- `minfxy_E1E2_f121_r2_results_20260831.md` (NULL; 100/99/73/0; best 1.5437 @콜 12; **:131 부근
  헤드 +0.0333 / 비율 반사실 +0.0614 / proxy +0.0503** ★)
- `pinbu_wave_minfxy_r1_results_20260830.md` (**:101 replay-exact 20/20** ★; **:133-141 5개 F_r-era
  측정 F_xy 표** ★; **:145-152 추정기 스코어카드 — `ratio (1.0640 × F_r)` MAE 0.008** ★;
  **:162-166 r1 정정 이득 0.0080** ★)
- `minfxy_E1E2_f121_r2_prereg_20260831.md` · `intervention_wave_r1_results_20260830.md`
  · `hgd569_degeneracy_memo_20260830.md` · `coreagnostic_v3_design_20260817.md`
  · `flat_assembly_fr_plan_20260802.md` §1.3 (E1 1.146 / E2 1.152) · `flat_assembly_fr_verdict_20260809.md`
  · `pitch_radius_readiness.md` · `kinf_shape_features.md`
- **r2 phase-2 pin 웨이브: 2026-09-03 02:07 완주 (30/30). `a785eded` 측정 pin 수확 pending**
  (오케스트레이터 통보, 본 문서 작성 시점 리포트 미발행)

---

## 부록 B — 존재하지 않는 것 / 확인하지 못한 것

**코드에 존재하지 않음 (확인함)**
1. `FuelDesign` 축 위의 acquisition 항 — 0개.
2. 프로덕션 체인의 핀맵 저작 — `lattice.py` 에 없음.
3. `compliance.enforce_new_type` 의 호출자 — 테스트 외 0개.
4. `5_RL` 안의 `6_DeCART_Surrogate` **기능적** 참조 — 0개 (독스트링 1건: realize_lat1600.py:172).
5. `2_LP/MOCHA/config_apr1400.yaml` 의 `surrogate_fuel_catalog` — 항목 자체가 없음.
6. LEU+ 농축 / Gd 장입의 **인허가** 상한 진술 — 워크스페이스 전체 0건.
7. IFBA / Er / Dy 격자 — 0건.
8. ★ **SDM 용 full-core rod model** — 없음 (campaign.py:3012-3023 이 명시).
9. ★ **패키지의 `.sum` / `dec_FA_*.inp`** — 스테이징 경로 자체가 없음.
10. ★ **`FuelDesign` 의 레이아웃 필드 / `octant_to_full` 확장기 / 저작 덱 파일명 유일화** — 전부 없음.
11. ★ **`IGD_0` / `IGD_4` / `IGD_8` 템플릿 디렉터리** — 어느 트리에도 없음.
12. ★ **`templates_lat1600` 의 `IGD_20`** — 없음 (슬라이스가 처음 만든다).

**파일이 이 머신에 없음**
13. `screen1600.csv` 외 lat1600 설계표 (스크래치패드 세션 삭제). 재스크린 캐시
    `screen_final_{117,121}.npz` 는 `E:\lpopt_archive\opmodel_20260829\`.
14. (A) 학습 매니페스트 `manifest_all_al_lowgd.csv`, `manifest_PA.csv`, `ff_catalog.npz`, 헬퍼 8종
    → **(A) 재학습·`--case-id` 모드 불가.**

**읽지 못했거나 확인하지 못함 (정직)**
15. **파케이를 열지 않았다** (로컬 연산 금지). 행수·커버리지는 코드 상수/리포트 인용이며 서로
    정합하지 않는다 (131×39 / 153×55 / xs_* 158·194) — **근사치로 취급.**
16. `search/campaign.py`(4,846줄) 전체, `model/train.py` 미독 — 케이스 알파벳 선택 경로는
    덱 주석과 리포트로만 안다 (단 `_rod_model`/`_maybe_post_verify` 는 ★ 직접 읽었다).
17. PROLOG4.1 / MASTER4.0 매뉴얼 PDF 는 HOST_199 에만 있고 읽기전용 ssh 로 읽을 수 없다 →
    PROLOG `%OPTN` 과 MASTER 라이브러리 검증 규칙은 **추론**이다.
18. HOST_181 의 DeCART 실측 wall time 없음. **box 104 의 750 s 는 실측이지만 box 104 의 CPU/RAM 은
    이 문서에서 확인하지 못했다** (호스트 테이블의 해당 칸은 비워 두었다).
19. `MAS_HFF` 61 vs `BURN 62` 불일치의 원인 (R18).
20. ★ **TotalBatcher 의 COMP 순서 결정 규칙** — 관측 불가 (R21). G-H3c 는 기전이 아니라 결과를 검사한다.
21. ★ **`n_gd = 0` HGC 가 6,867,567 B 인 이유** — 미확인 (G-H1b 를 ABSTAIN 으로 둔 이유).
22. `3_GA_Surrogate/README_RL.md` 등 OneDrive 탈수화 파일 — ga80 설계 출처 가능성 있으나 열 수 없다.

---

## 부록 C — 이 문서가 만들지 않은 것

본 문서는 **설계안 v2** 다. 코드 0줄, DeCART 0회, MASTER 0회, 파케이 읽기 0회, 발사 0회.
착수 시 **첫 산출물은 task #0(소급검증)** 이고, 그 다음이 §8.2 슬라이스의 **정식 사전등록 문서**다.
**§10 의 A1–A9 는 오케스트레이터 가정이며, 사용자가 뒤집을 지점을 각 행에 적어 두었다.**

---

## 부록 D — 변경 로그 (v1 → v2)

### D.1 수용한 정정 (근거를 파일:라인에서 재확인한 뒤 반영)

| # | v1 | v2 | 근거 |
|---|---|---|---|
| 1 | 사슬 오차 0.008, 발사 바 `ΔF_xy ≥ 0.020` | **σ(F_xy) ≈ 0.065** (해석) / **+0.0614** (실측). 바를 **사전등록 `k·σ_chain,paired`** 로 교체, task #0 신설 | pinbu r1:145-152 (`ratio (1.0640 × F_r)` = **측정** F_r), r2:131 부근 반사실, OPSCREEN:246-250 |
| 2 | "비율 추정기가 헤드보다 5–7배 정확" | **두 행으로 분리**: `r·F_r_meas` 0.008(단위환산) / `r·F̂_r` +0.0614(n=1). "5–7배" 주장 삭제 | 동상 |
| 3 | 사슬을 트리거로 승격 | **floor 추정기**로 명시 + **B3(T5_T6) 반례** 등록 + C6 널테스트 병기 | OPSCREEN:253-255, :222-224, :249-251 |
| 4 | 인허가 게이트에 SDM/MTC 없음 | **MTC 를 PRIMARY 경성 게이트로**, `post_verify_delivery` + `post_verify_master_calls` 예산 계상, S9b 신설 | sdm_mtc.py:1-80, :1492 |
| 5 | `n_gd` "{0,4,8,12,16,20,24} 전체 개방 권고" | **{12,16,20,24}로 제한** | spec.py:64-67, lattice.py:32-36, fuel_types.py:1477, 디스크 |
| 6 | R4/R5 를 "미측정 위험"으로, G-H4 를 "첫 측정" | **이미 유계 (≤100 pcm / ≤0.0021 FF)**, G-H4 는 **회귀검사** | OPSCREEN:165-179 |
| 7 | "MASTER 부트스트랩 (밴드 1개) 2–5 h" | **9회(기존 8 + 신규 1)**, 단일 실패 꼬리 2.43 h 명시 | OPSCREEN:388-393, `bases/` 8개 실측 |
| 8 | G-H5 = `validate_reload_deck` + "평형 ≥2 사이클" | **G-H5a/G-H5b/G-H5c 로 분할** (`%LPD_BCH` 거부 / reload 전용 / **수렴까지**) | assets.py:325-327, bootstrap.py, config.py:907,915 |
| 9 | 후보 "Z" 레이아웃만, "1–2 격자" | **Z1/Z2 완전 지정 (2 격자 고정)**, ΔFF −0.017(DeCART 등가) 병기 | OPSCREEN:402-410, :352-353 |
| 10 | 설계 열거 `O(10^5) ≈ 3분` | **3.7×10³ 설계** (89 레이아웃은 합계, `du` 는 자유축 아님) → GPU 7 s / CPU 7 min | SURROGATE_USAGE:294, predict.py:107-112 |
| 11 | R17 "PA/PB ↔ z1/z2 미확인" | **종결: `z1 ⇔ PB`, `z2 ⇔ PA` (반전)** + `screen.py` 어서션 | 두 덱 옥탄트 8행 + predict.py:67-71 + OPSCREEN:410 |
| 12 | "`ΔF_xy` 재시작 잡음 0 → 측정 잡음은 제약 아님" | **replay 재현성**으로 재명명; **restart-변경 민감도는 미측정**이고 부정 통제가 그것을 잰다 | pinbu r1:101 |
| 13 | 작업 #14 "하나라도 남기면 사고 재발" | **위생으로 강등** + 테스트 가드로 이동 | verify.py:852-863, assets.py:616-619 |
| 14 | "게이트를 배선하면 0.92 후보가 기각된다" | **덮어쓴다** — `enr_main`+`enr_zone` 동시 전달 필수, 테스트는 `ComplianceError` 어서트 | compliance.py:308-317 |
| 15 | 밀도 차 "1.0% / 0.7%", 출처 §7.7 | **0.68 / 1.02 / 1.23%**, 출처 `realize_lat1600.py:232-238`; §7.7 의 +1.6% 는 **다른 패밀리** | 계산 + predict.py:1070-1074 |
| 16 | `%MICX`/`CRD1*` 내용 미검사 | **G-H3b 신설** (핵종 로스터 일치 + 헤더 `62 17 6 0 0`) | §5.2 |
| 17 | TotalBatcher "~1 min (41 타입)" | **"N=16 에서만 관측, N=37/39 미측정"** | 타임스탬프의 한계 |
| 18 | G-H3 `2,008 + 385,849·N` | **`2,010 + 385,849·N_(n_gd>0) + 377,461·N_(n_gd=0)`, 등식** | **6개 N 에서 정확 일치 확인** |
| 19 | G-H1b 정확 바이트 FAIL | `{12,16,20,24}` 만 FAIL, 그 외 **ABSTAIN** | 기전 미상 |
| 20 | DeCART 51.4 min/case 를 181 계획에 사용 | **호스트별 귀속 명시** (199 직렬 3,084 s 실측 / 104 omp 735–750 s 실측 / 181 미측정) | manifest_199, realize_lat1600.log:31-34 |
| 21 | "r1→r2 실질 이득 0.008" | **r1 자신의 정정 이득**; r2 이득은 음수(+0.0142 악화) | pinbu r1:162-166 |
| 22 | 헤드 MAE 0.0663 과 "5–7배" 혼용 | 기준선 하나로 통일(0.042–0.060) 또는 삭제 | pinbu r1:145-152 |
| 23 | 별칭 순서 위험 미등록 | **R21 등록 + G-H3c** (단, 기전은 단정하지 않음) | spec.py:146, library.py |
| 24 | v9 "손해가 0" | **"기존 곡선 채널과 동일한 34/70 결손 (증가분 0)"** + ga80 부분집합 별도 G1 | §7.3 |
| 25 | `resolver.paramA_library_dims` 가 "자동 정합" | **거짓 — 삭제.** `cores/` 10개 재생성 + `synth_decks` purge 를 **S5b 로 신설** | resolver.py:70-88, assets.py:718-760, :820-833, `cy02.inp:27` |
| 26 | v9 채널을 `.sum` 에서 수확 | **`.sum` 은 패키지에 없다** → **HGC 유도**로 교체 + 작업 #13b(`.sum`/덱 스테이징) | lattice.py:317-347, package.py:60-75, fuel_types.py:1678, 디스크 |
| 27 | 호스트 표에 box 104 없음 | **추가.** omp 4-way 실증 (750 s/case), 자산 3종 모두 보유 | realize_lat1600.log, design_lat1600_104.inp |
| 28 | 캠페인 200콜 / pin 웨이브 / MTC 콜 미계상 | **§0.4 에 전부 계상**, 총계 1일 → **1.5–2일** | — |
| 29 | 큐 ps1 과 `run_batch` 혼용 | **`run_batch` 로 일원화**, 큐에서 SHA 대조 + `nxfile` 재작성만 이식 | §5.4 |
| 30 | `enforce_new_type` 에 옥탄트 맵 전달 가정 | **확장기 없음** → `octant_to_full` 을 작업 #1 에 포함 | compliance.py `is_octant_symmetric(n=16)` vs `_triangle` |
| 31 | 패키지 배송 = `lib/` 만 | **동기화 세트 정의 + 재-ingest (S6b)** | fuel_types.py:1625-1690 |
| 32 | `count_gd_pins_from_hgc` @ realize_lat1600.py:318-325 | **`lpopt/data/fuel_types.py:554`** (호출부가 realize_lat1600.py:336) | — |
| 33 | "88콜 무이득" vs "100콜 0/99" 혼재 | **100 제출 / 99 수렴 / best 콜 12 / 이후 88콜 무이득** 으로 통일 | r2 결과 |
| 34 | §10 결정 요청 7건 | **§10 오케스트레이터 가정 A1–A9** (뒤집을 지점 명시) | 지시 |

### D.2 ★ v2 가 새로 발견한 것 (두 비평 모두 놓친 것)

| # | 발견 | 근거 | 영향 |
|---|---|---|---|
| **N1** | **준법 비율 0.85 와 (A) `du ∈ [0.40,0.80]` 이 `u_high ≤ 5.333 w/o` 를 강제한다.** Q1 의 5.50 상한은 **구속하지 않는다** | `du = 0.15·e1`; predict.py:107-112, `_snap_to_grid` 경계검사 | 설계공간 창이 `[5.00, 5.333]` 로 축소 → 열거 격자가 3.7×10³ 으로 줄고, **A1(농축 상한)의 실무 의미가 사라진다** |
| **N2** | **슬라이스 후보 Z1(u5.50/4.6750)이 (A) 포락 밖** (`du = 0.825`) — v1 §3.3 조건 2 가 자기 후보를 기각한다 | 동상 | **Z1 을 e2 4.70 (du 0.80, ratio 0.8545) 으로 재지정** (§8.2a) |
| **N3** | **`validate_design` 의 0.1 격자는 물리 경계가 아니다** — T3/T4 는 `du = 0.75` 로 격자 밖에서 스크린되어 DeCART 대조 <100 pcm 통과 | designs.json + OPSCREEN:165-179 | 발사 조건 F3 을 **"경계만 강제, 격자는 불강제"** 로 정의 (v1 의 `errors == []` 는 T3/T4 를 소급 기각) |
| **N4** | **사슬을 양쪽에 일관 적용하면 슬라이스의 부호가 뒤집힌다**: floor 단위 −0.041 vs 회귀 단위 **+0.035** | OPSCREEN:319(Z), :427(E1_E2 contrast 0.049), :215-235(A0 node_peak 1.2085, FF 1.1520) | **슬라이스의 서사 전체**를 "예측 실현"에서 "부호 측정"으로 바꾼다 (§8.2b) |
| **N5** | **SDM 은 lpopt 에서 게이트가 될 수 없다** (`_rod_model → None`) | campaign.py:3012-3023 | physics 비평의 PRIMARY 요구를 **절반만** 수용; 작업 #20 신설 |
| **N6** | **저작 덱은 `templates_lat1600` 별도 트리로 간다**, 파일명은 항상 `dec_FA_lat1600.inp`, `build_template_tree` 는 **n_gd 16/24 만** | realize_lat1600.py:225-258, .log:2-30 | 레이아웃 충돌 위험의 **정확한 범위**를 정한다(R23/R24) — 슬라이스는 안전, 라운드 2는 아님. 그리고 **n_gd 20 확장이 슬라이스의 선행 작업**이다 |
| **N7** | **슬라이스 후보의 `type_id` 는 기존 37개와 충돌하지 않는다** (`P5547Z1G08N20`, `P5042Z1G10N20` 부재) | designs.json 전수 | eng 비평의 "슬라이스가 기존 타입 HGC 를 덮어쓴다" 시나리오를 **슬라이스에 한해 반려** |
| **N8** | **MTC 게이트는 `[verify] keep_success/harvest_maps` 를 요구한다** | campaign.py:3005-3007 | 슬라이스 캠페인 덱에 명시 (작업 #21) |

### D.3 ★ 반려하거나 조건부로만 수용한 비평 항목

| 비평 | 주장 | v2 판정 | 근거 |
|---|---|---|---|
| **physics #4** | "PRIMARY 에 `mtc ∈ [−54,+9]` **와** `sdm_margin ≥ 0` 을 PASS 조건으로 추가" | **절반 반려.** MTC 만 게이트. **SDM 은 오늘 구조적으로 INCONCLUSIVE** | `campaign._rod_model()` 이 의도적으로 `None` 을 반환하고 그 이유를 독스트링에 적는다 (campaign.py:3012-3023); `[constraints] sdm_enable` 기본 False |
| **physics #10 / eng #13** | "스크리닝이 CPU 에서 **3.2 h**(65배) — HIGH" | **심각도 하향(HIGH → LOW).** 산수는 맞지만 **10⁵ 설계 전제**에서만 성립. 실제 라운드 1 격자는 **3.7×10³** 이라 CPU 로도 **7분** | §4.3 (89 = n_gd별 합계 + `du` 는 ratio 고정으로 종속축). 접근 확인 요구 자체는 **수용**(A8) |
| **physics #11** | "올바른 개수는 `11×5×3×2×89 ≈ 29,370`" | **수용하되 갱신.** `du` 를 자유축으로 세면 여전히 5배 과대. ratio 고정 시 **7×3×89×2 ≈ 3,738** | 동상 |
| **physics #19** | "G-H3 는 `\|MAS_XSL − (2008 + 385849·N)\| ≤ 8` 의 **허용오차**로" | **반려 — 더 강한 형태를 채택.** 절편 **2,010** 이면 **6개 N 에서 오차 0** 이므로 허용오차가 필요 없다. `n_gd=0` 항까지 넣어 **등식**으로 게이트 | 실측 11/12/16/33/37/80 |
| **physics #24** | 별칭 순서 위험을 **확립된 사실**로 서술 | **조건부 수용.** `MAS_XSL` COMP 순서 결정 규칙은 **관측 불가**(TotalBatcher 내부)이므로 기전을 단정하지 않는다. **결과를 검사**하는 G-H3c 로 대체 | library.py 는 `set_names` 를 돌려줄 뿐 순서 규칙을 정하지 않는다 |
| **eng #17** | "슬라이스가 기존 타입의 HGC 를 덮어쓸 수 있다 — 착수 전 필수 수정" | **결함은 전면 수용, 슬라이스 차단성은 반려.** ① 저작 덱은 **별도 트리**로 가고 ② 슬라이스 두 후보의 `type_id` 는 **기존 37개 어디와도 충돌하지 않는다**. 단 **R23(같은 (gd,n,z) 파일명 충돌)은 슬라이스에서도 실재**하므로 작업 #1 은 여전히 선행 필수 | realize_lat1600.py:225-258 + .log; designs.json 전수 |
| **eng #7** | "ΔF_xy ≈ 0.019 < 발사 문턱 0.020 이므로 슬라이스가 자기 규칙을 못 넘는다" | **논점 소멸.** 0.020 바 자체가 폐기되었고(§3.3), 더 근본적으로 §8.2b 가 **부호조차 미정**임을 보인다 | §3.3, §8.2b |
| **eng #11** | "혹은 append 된 COMP 가 기존 index 를 안 건드린다는 약한 가설을 먼저 검증하라" | **부분 수용.** 그 검증을 **G-H3c** 로 넣되, `LIBRARY_BUILD.md §5` 의 "모든 `MAS_RST.*` 무효화" 진술이 있으므로 **9회 부트스트랩 예산은 그대로 계상**한다 (가설이 맞으면 절약, 틀리면 필수) | LIBRARY_BUILD.md §5 |
