# 집합체 온디맨드 — **파일 단위 구현 작업 목록** (의존 순서)

**작성** 2026-09-03 · **상태** 착수 대기 (코드 0줄)
**선행** `assembly_on_demand_design_v2_20260903.md` (설계안 v2) · `assembly_slice_Z_prereg_20260903_DRAFT.md` (슬라이스 사전등록)
**성격** v2 §8.1의 21개 태스크를 **의존 순서로 재배열**하고, 재비평 2건이 잡은 항목을
**작업 단위로 흡수**한 것. 각 작업은 **트랙 · 파일 · 테스트 · 추정 · 실행 호스트**를 갖는다.

**담당 규약.** 코딩은 Opus 5 medium-effort 에이전트에 위임, 메인 세션은 오케스트레이션
(`coding-agent-preference` 메모리). **로컬 PC(`mk-ctrp`, = 문서상의 "box 104")에서는
어떤 연산도 하지 않는다** — 편집·커밋은 로컬, **실행·테스트는 원격**.

---

## 0. 요청 주제 → 절 매핑 (의존 순서로 재배열했다)

| 요청 주제 | 이 문서의 절 | 왜 이 순서인가 |
|---|---|---|
| lattice.py 핀맵 저작 승격 | **§1** | **모든 것의 뿌리.** 저작 없이는 후보가 사전에 실패 확정 (OPSCREEN §8) |
| compliance 배선 | **§2** | 저작 산출물(옥탄트 맵)을 소비한다 → §1 직후 |
| screen.py 서로게이트 브리지 (`z1⇔PB` + 연료타입→설계 카탈로그) | **§3** | 후보를 고르는 단계. §1의 `gd_positions` 필드를 쓴다 |
| 181 DeCART 큐 러너 (2_LP 큐 스크립트 재사용) | **§4** | §1이 만든 덱을 §3이 승인한 뒤에야 실행 대상이 있다 |
| 패키지 재빌드 단계 | **§5** | HGC가 떨어진 뒤 |
| `.sum` 스테이징 + HGC 유도 채널 | **§6** | 재빌드와 같은 파일들을 만지므로 §5 직후. v9는 라운드 2 |
| sdm_mtc 납품 게이트 | **§7** | 캠페인이 돌아야 검증 대상이 있다 |
| **acquisition의 `FuelDesign` 축 트리거 항** | **§8** | **의존상 마지막이다** — §3의 스크리너·§1의 카탈로그·task #0의 σ가 전부 있어야 정의된다. 슬라이스는 이것을 **쓰지 않는다** (A6: 슬라이스 우선) |
| 문서 | **§9** | — |

**의존 그래프 (굵은 화살표 = 슬라이스 임계경로):**

```
   #0 σ 소급검증 ─────────────────────────────(병렬, 슬라이스 비차단)──────────→ #9
                                                                                  ↑
   S0c (A)접근 ──→ #4 screen.py ──→ #5 사슬 ──→ #6 다양성 ─────→ #7 need ──→ #8 로스터
        │              ↑                                                          
   #1 lattice 저작 ═══╪══→ #16 compliance ═══╗                                    
        ║             │                       ║                                    
        ╠══→ #2 프리플라이트 ══→ #10 러너 ════╬══→ #11 HGC게이트 ══→ #15 lib게이트    
        ║                                     ║          ║                        
        ╠══→ #13 designs 스키마               ║       #12 스냅샷                   
        ╚══→ #13b .sum/덱 스테이징 ═══════════╝          ║                        
                                                      #13c 패키지 재생성           
                                                            ║                      
                                                   #21 MTC 독립단계 ══→ #21b 확인   
                                                            ║                      
                                              #17 ga80 감사 ──→ #18 cond v9 (라운드2)
                                              #20 SDM rod model (비차단)
                                              #14 resolver 위생 (비차단)
                                              #19 prereg 문서 (완료: DRAFT)
```

---

## §1. `lattice.py` 핀맵 저작 승격 — `realize_lat1600.author_template` 의 1급화

> **왜 첫 번째인가.** 프로덕션 체인 `lattice.edit_dec_text`(lattice.py:114-147)는 **숫자 3개
> (UO2 92235 / UO2_2 92235 / UO2G 6408) + CASEID만** 고치고 **핀맵·밀도줄을 건드리지 못한다.**
> OPSCREEN §8: 동결 레이아웃으로 도달 가능한 최선 FF는 **1.1657** 로 incumbent ga80 E4(1.1390)보다
> 나쁘다. 문턱을 넘는 것은 **개방 20핀 레이아웃 `1:1;4:1;6:4`** 하나뿐(FF 1.1208, DeCART 등가 1.1222).
> → **저작 없는 온디맨드 설계는 사전에 실패가 보장되어 있다.**

### 작업 #1 — `lattice.author_gd_layout()` + `octant_to_full()` + `FuelDesign.gd_positions`

| | |
|---|---|
| **트랙** | T-LAT |
| **파일** | `5_RL/lpopt/design/lattice.py` (신규 함수 2개 + `resolve_template` 오버라이드 인자)<br>`5_RL/lpopt/design/spec.py:47` (`gd_positions` 필드), `:94-99` (`key`), `:103-111` (`as_dict`), `:85-91` (`type_id` 레이아웃 태그)<br>`5_RL/lpopt/design/spec.py:188-200` (`DesignRegistry.alias` 가드)<br>**참조 구현**: `5_RL/realize_lat1600.py:137,165,170,180-222,225-258` |
| **내용** | ① `author_gd_layout(base_deck_text, gd_positions, n_gd) -> str` — 옥탄트 삼각형(8행) 안에서 셀 id `3`(UO2G) 이동. 가드는 참조 구현에 이미 있다(`_triangle`, `_census`, `_n_gd_of`, 안내관/zoning 셀 불가침, 인구조사 == `n_gd`)<br>② `octant_to_full(rows) -> 16×16` — **확장기가 지금 없다**. `compliance.is_octant_symmetric(pin_map, n=16)` 이 평탄화된 전맵을 받으므로 필수<br>③ `FuelDesign.gd_positions: tuple[tuple[int,int], ...] | None = None` → `key`/`as_dict()` 에 포함<br>④ `type_id` 는 **레이아웃이 동결 템플릿과 다를 때만** `f"{base}L{sha1(layout)[:3]}"` 태그. **기존 37 id는 바이트 불변**(T3/T5/T6는 이미 개방 레이아웃인데 무태그 — 소급 재명명 금지)<br>⑤ `DesignRegistry.alias` : `type_id` 는 같은데 **기록된 설계 튜플/`gd_positions` 가 다르면 raise**<br>⑥ 저작 덱 파일명 **`dec_FA_<type_id>.inp`** 로 유일화 + `resolve_template` 에 설계별 경로 오버라이드 (현행은 `sorted(glob("dec_FA_*.inp"))[0]`, 참조 구현은 항상 `dec_FA_lat1600.inp` → **같은 `(gd_wt,n_gd,z)` 에 두 레이아웃 공존 불가** = R23)<br>⑦ `n_gd ∈ {12,16,20,24}` 전부에 대한 서브트리 규칙 (`build_template_tree` 는 지금 **16/24 만** 알고 그 외엔 `SystemExit` → **슬라이스의 n_gd 20이 여기 걸린다**)<br>⑧ T3–T6의 `gd_positions` 백필 |
| **테스트** | ① 인구조사(대각 4핀/비대각 8핀 합 == `n_gd`) / 안내관 `(0,0),(3,3),(4,3),(4,4)` 불가침 / zoning 공통셀 불가침 / Chebyshev ≥ 2 / 옥탄트 대칭 / **BOM 없는 ASCII** 각 1건<br>② **`designs.json` 의 기존 37 `type_id` 가 라운드트립으로 바이트 불변** (회귀)<br>③ 같은 `(8,20,z1)` 에 서로 다른 두 레이아웃이 **공존**하고 각각 자기 덱으로 해석된다<br>④ `octant_to_full` 산출이 `is_octant_symmetric(…, n=16, tol=1e-3)` 통과<br>⑤ `n_gd = 20` 에서 `SystemExit` 이 나지 않는다<br>⑥ 레지스트리 가드: 같은 `type_id` + 다른 `gd_positions` → raise |
| **추정** | **1.5 일** |
| **실행** | 편집 로컬 / **테스트는 238** (`~/lpopt_ws`, venv) |
| **차단** | #2, #10, #13, #13b, #16 전부 |

### 작업 #2 — exe/dll 프리플라이트 · `nxfile` 재작성 · SHA-256 대조

| | |
|---|---|
| **트랙** | T-LAT |
| **파일** | `5_RL/lpopt/design/lattice.py:29` (`DEFAULT_DECART_EXE`), `:297-313` (`launch_decart`) |
| **내용** | `DEFAULT_DECART_EXE` 는 **omp** exe 인데 `libiomp5md.dll` 이 181에도 199에도 없다(로컬에만 있다) → **직렬 폴백**. 2_LP 큐 스크립트에서 **좋은 성질 두 개만 이식**: ① exe / XS 라이브러리 **SHA-256 대조** (`run_decart_eq_xesm_queue_181.ps1:6-15`), ② 덱의 `nxfile` 줄을 스테이징된 로컬 XS 경로로 **정규식 재작성**. 큐 자체는 채택하지 않는다(§4) |
| **테스트** | 없는 `nxfile` 로 fail-fast; dll 부재 시 직렬 폴백; 해시 불일치 시 raise (임시 파일 픽스처) |
| **추정** | 0.5 일 |
| **실행** | 테스트 238 (해시는 픽스처) / 실사용 181 |
| **선행** | — (#1과 병렬 가능) |

### 작업 #3 — `xenon TR` 상수화

| | |
|---|---|
| **트랙** | T-LAT |
| **파일** | `5_RL/lpopt/design/lattice.py` |
| **내용** | 온디맨드 HGC는 **`xenon TR`** 유지 (가정 A3). 한 `MAS_XSL` 안에 서로 다른 Xe 모드의 COMP를 섞는 것은 검증된 바 없고 기존 37 paramA + 80 ga80이 전부 TR이다. (A)의 EQ 가정은 G-H4가 유계로 묶는 **스크린 전용 계통오차** |
| **테스트** | 저작 덱 diff 에 `xenon` 카드가 베이스와 동일 |
| **추정** | 0.1 일 |
| **실행** | 238 |
| **선행** | — |

---

## §2. compliance 배선 — `enforce_new_type` 의 유일한 프로덕션 호출자를 만든다

> **현황:** `compliance.enforce_new_type`(compliance.py:282-330)의 **호출자는 테스트 외 0개**다.

### 작업 #16 — `enforce_new_type` 배선 (`enr_main` **와** `enr_zone` 동시 전달 + 전맵 전달)

| | |
|---|---|
| **트랙** | T-MOD |
| **파일** | `5_RL/lpopt/design/screen.py` (신규, §3) 또는 `5_RL/lpopt/design/spec.py` 의 어댑터<br>`5_RL/lpopt/data/compliance.py:300-326` (읽기 전용 소비) |
| **내용** | ① **`enr_zone` 을 반드시 함께 넘긴다.** 소스가 `out.get("enr_zone") is None` 이면 `0.85·enr_main` 으로 **조용히 채워 넣는다**(compliance.py:308-311) — 기각이 아니라 **덮어쓰기**다. `enr_main` 만 넘기면 **0.92에서 돌린 서로게이트 스크린이 무효가 된 채로 통과**한다<br>② **`pin_map` 을 넘긴다.** `pin_map` 은 **선택 인자**여서(`if pin_map is not None and not is_octant_symmetric(...)`, compliance.py:321-322) 생략하면 R2 검사가 조용히 통과한다 → **게이트가 요구해서가 아니라 우리가 선택해서** `octant_to_full` 결과를 넘긴다<br>③ 슬라이스 Z1′/Z2 는 둘 다 0.85 근방이라 ①의 영향을 받지 않지만, **라운드 2의 0.92 후보가 여기 걸린다** |
| **테스트** | ① `enr_main=6.2, enr_zone=5.7` (ratio 0.919) → **`ComplianceError` 를 어서트**한다. **채워진 값을 어서트하면 안 된다**<br>② `enr_zone` 생략 시 반환값이 `0.85·enr_main` 임을 어서트해 "덮어쓰기" 동작을 회귀로 고정<br>③ `pin_map` 생략 시 R2가 통과함을 어서트(현행 동작 고정) + 전맵 전달 시 비대칭 맵이 raise |
| **추정** | 0.3 일 |
| **실행** | 238 |
| **선행** | #1 (`octant_to_full`), 가정 A2 |

> **남은 모순 (수정하지 않는다, 기록만).** `spec.DESIGN_GRID["ratio"] = {0.85, 0.92}`(spec.py:33)
> vs `ZONE_RATIO_TARGET 0.85 / TOL 0.03`(compliance.py:69-71) → **0.92는 창 밖**이고 라이브
> 레지스트리에 0.92 타입이 실재한다(`P6257Z2G08N16` → 6.2/5.7 = 0.919). 지금까지 터지지 않은
> 이유는 호출자가 0개였기 때문이다. `DESIGN_GRID` 는 **LHS 표본 격자이지 검증기가 아니다**
> (`__post_init__` 는 `0 < e2 ≤ e1` 만 본다) — 배선 후 0.92 타입을 **재생성**하려는 시도만
> 실패하고 기존 행은 영향받지 않는다.

---

## §3. `screen.py` — 서로게이트 브리지 (`z1⇔PB` 어서션 + 연료타입→설계 카탈로그)

### 작업 #4a (S0c) — (A) 체크포인트 접근 확보 **[경성 선행, 신규]**

| | |
|---|---|
| **트랙** | T-SCR |
| **파일** | 코드 없음. 산출 = 매니페스트 1개 |
| **내용** | (A)는 **`USER2` 계정** `/home/USER2/lattice_surrogate/kpin_pa` 에 있고(`SURROGATE_USAGE.md:148-165`) 프로젝트 계정은 `USER@HOST_238:8022` 다. `Engines.__init__`(predict.py:899)은 `dataset/bu_grid.npy` + `runs_root` 앙상블 체크포인트를 요구한다. **`screen1600.csv` 는 이 머신에 없다** → 예측은 **재사용이 아니라 재생성**이다 |
| **산출** | `USER` 홈에서 읽히는 (A) 트리 사본 + `predict.py --self-test` 통과 로그 + 체크포인트 SHA-256 매니페스트 |
| **테스트** | `--self-test` 허용오차 통과 |
| **추정** | 0.5 일 (접근 협의 포함) |
| **실행** | **238** |
| **차단** | #4, #5, #6, 슬라이스 S1 |

### 작업 #4 — `lpopt/design/screen.py` 신설 (브리지)

| | |
|---|---|
| **트랙** | T-SCR |
| **파일** | **신규** `5_RL/lpopt/design/screen.py` (~300줄)<br>참조 패턴: `2_LP/MOCHA/surrogate_adapter.py:188-215` 의 **지연 임포트** |
| **내용** | ① (A) 경로 임포트 + `Engines` + `predict_cases`. **(A) 부재 시 graceful degrade** (advisory only — 어댑터 독스트링 *"an unavailable/invalid prediction must not be used as a hard reject"*)<br>② **`Z_TO_PATTERN = {"z1": "PB", "z2": "PA"}` 를 모듈 상수로 두고 임포트 시 어서션.** R17 종결 근거: `predict.py:67-71` 의 `PA = ZONING_COMMON ∪ {(7,c)}` / `PB = ZONING_COMMON ∪ {(7,6),(7,7)}` 와 `0_APR1400/5.8_5.1/FA/IGD_16/6_16_z1/dec_FA_A01.inp` 옥탄트 8행 `1 1 1 1 1 1 2 2`(→PB) / `…/6_16_z2/dec_FA_A02.inp` `2 2 2 2 2 2 2 2`(→PA). **직관과 반대다.** `2_LP/MOCHA/config.py:349-358` 의 기본값이 `surrogate_pattern = "PA"` 이므로 **z1 설계를 PA로 스크린하면 조용히 틀린 패밀리를 평가한다 — 오류도 경고도 없다**<br>③ **연료타입 → 설계 카탈로그**: 코드가 아니라 **`data/design/package/designs.json` 에서 읽는다.** (C) 어댑터가 실전에서 죽어 있는 이유가 `config_apr1400.yaml` 에 `surrogate_fuel_catalog` **항목 자체가 없어서** 다리가 물리적으로 건널 수 없기 때문이다. paramA는 이미 `{type_id, e1, e2, zoning_variant, gd_wt, n_gd, alias, gd_u_enr}` 를 기록한다. **`gd_positions` 는 37행 중 4행(T3–T6)에만 있으므로 온디맨드 타입은 필수 필드로 승격**(#13)<br>④ **경계(bounds)만 강제, 격자(step)는 강제하지 않는다.** `u_high ∈ [5.00,7.00]`, `du ∈ [0.40,0.80]`, `gd_u ∈ [3.00,4.30]`, `gd_wt ∈ {6,8,10}`, `n_gd ∈ {12,16,20,24}`, Chebyshev ≥ 2. 근거: T3/T4는 `du = 0.75` 로 0.1 격자를 벗어난 채 스크린되어 DeCART 대조 <100 pcm 통과; 반면 **경계 밖 외삽은 n_gd = 28에서 +1,750 pcm 로 무너진다**. `validate_design` 의 `errors == []` 조건은 T3/T4를 소급 기각하므로 쓸 수 없다<br>⑤ `--enumerate` 서브커맨드 — 사전등록 §1.3의 창(`u_high ∈ [5.00,5.50]` × `ratio ∈ [0.82,0.88]` ∧ `du ∈ [0.40,0.80]` × `gd_wt` × `(n_gd,layout) ∈ 89 쌍` × `z`)에서 설계 수를 **산출**한다 (v2의 3,738은 `ratio = 0.85` 고정 전제의 수) |
| **테스트** | ① (A) 부재 시 graceful degrade (임포트 실패해도 모듈 로드)<br>② **`Z_TO_PATTERN` 어서션** — `z1 → "PB"`; `"PA"` 를 강제하면 raise<br>③ `--self-test` 허용오차 통과<br>④ 경계 밖(`du = 0.825`) 거부 / 격자 밖(`du = 0.75`) **통과**<br>⑤ 카탈로그가 `designs.json` 에서 읽히고 `gd_positions` 부재 행이 경고를 남긴다 |
| **추정** | **1.5 일** |
| **실행** | **238** |
| **선행** | #4a |

### 작업 #5 — OPSCREEN 사슬 이식 (순수 함수)

| | |
|---|---|
| **트랙** | T-SCR |
| **파일** | `5_RL/lpopt/design/screen.py` (순수 함수 절)<br>이식 원본: `5_RL/opmodel/s09b_contrast.py:20-49`, `s09_frmodel.py`, `s08_transfer.py`, `opmodel.py` |
| **내용** | `contrast` / `d_fresh` / `node_peak` / `F_r` / `ratio` / opmodel 운전점을 옮기되 **floor 라벨을 코드에 새긴다**: 함수 독스트링에 OPSCREEN.md:253-255 의 면책조항(*"reported as what the LP could reach **if** it restores node_peak … not as a prediction"*)과 반례 B3(T5_T6: 표에서 **가장 평탄한** FF_hot 1.1020인데 **최악의 측정 `F_r` 1.5795**)를 그대로 인용한다.<br>**`d_fresh` 는 `rm(0.5) − rho_op(rm, f, bc, 0.0)`(s09b:27)이다 — OPSCREEN 표의 `hump` 열이 아니다.** (v2 §8.2b가 이 두 열을 혼동했다.)<br>`Fr_flr ≡ 1.03 × 1.2085 × FF_hot`, `Fr_fix = A · npk · FF_hot` 를 **둘 다** 반환하고, 호출자가 **구간**을 받게 한다 |
| **테스트** | ① T3–T6 값 재현 (FF 1.1073 / 1.1409 / 1.1012 / 1.1011)<br>② `s16_out.txt` feed 121 **[B] #3** 재현: `npk 1.293`, `Fr_fix 1.495`, `Fr_flr 1.395`, `contrast +0.0308`, `hump 0.0056`<br>③ `d_fresh` 와 `hump` 가 **서로 다른 값**임을 어서트 (회귀 — v2 오독 방지) |
| **추정** | 1 일 |
| **실행** | 238 |
| **선행** | #4 |

### 작업 #6 — 다양성 / 중복제거 / 역할쌍 contrast 게이트

| | |
|---|---|
| **트랙** | T-SCR |
| **파일** | `5_RL/lpopt/design/screen.py` |
| **내용** | 설계 튜플 완전일치 + **기술자 공간 근접 중복**(z-정규화 거리 < 0.25) 접기; 다양성은 설계공간이 아니라 **기술자 공간 greedy max-min**(척도 = `ood_guard` z-스케일); 후보는 낱개가 아니라 **(68-슬롯, 53-슬롯) 쌍**으로 뽑고 **`contrast ≥ 0.026`** 미달 쌍 기각. 근거: contrast ≈ 0 인 arm 들이 `node_peak` 1.387–1.551 / `F_r` 1.559–1.818 로 흩어졌다 |
| **테스트** | 합성 로스터에서 선택 결정성(동일 입력 → 동일 출력); contrast 0.020 쌍 기각 |
| **추정** | 0.5 일 |
| **실행** | 238 |
| **선행** | #5 |

---

## §4. 181 DeCART 큐 러너 — 2_LP 큐 스크립트의 **성질만** 이식

> **결정 (v2 §5.4 유지).** 러너 정본은 **`lattice.run_batch`** 다 — 멱등성(`_hgc_looks_valid`),
> 올바른 리네이밍, `DesignSource` 직결을 갖는다. `2_LP/artifacts/run_decart_eq_xesm_queue_181.ps1`
> 은 `FuelDesign`·별칭·`registry.json` 을 모르고 `runs/<type>/FA_<T>_0101.HGC` + `manifest.json` 만
> 낸다 → **큐를 채택하지 않고, 검증된 성질을 가져온다.**

**이식할 성질 (원본 파일:라인):**

| 성질 | 원본 | 값 |
|---|---|---|
| exe SHA-256 선검사 | `run_decart_eq_xesm_queue_181.ps1:6,13-15` | `5F0F10F10BD4CC6546173C266DA3FDE72BDF1A09A191C59629FF7B4B0AF006CE` |
| XS 라이브러리 SHA-256 | 동상 `:5,7` | `AEF86EEBFB8B6398D0A45164C70E0FB04FCB5066546A12A3BBAB9106AF64E377` (`D:\DeCART_MASTER\LIB\DML-E71N047G018-PV01-cr08.BIN`) |
| 직렬 강제 | 동상 `:16` | `$env:OMP_NUM_THREADS = '1'` |
| 전역 프로세스 수 게이트 | 동상 `:32` | `decart2d1.1m5.exe` 프로세스 < 2 |
| 완료 판정 | 동상 `:57` | stdout 에 `JOB FINISHED` |
| 입력 해시 영수증 | 동상 `:63` | `input_sha256` |
| **`.sum` 보존** | 동상 `:55` | 큐는 `FA_<type>.sum` 을 남긴다 — **`lattice.harvest` 는 남기지 않는다** (#13b) |
| 배포·중지·감사 | `deploy_decart_eq_queue_181.ps1`, `stop_decart_eq_queue_181.ps1`, `audit_decart_eq_181.ps1`, `audit_decart_queue_task_181.ps1` | 스케줄드 태스크 등록·상태·정지 (재사용) |

### 작업 #10 — `run_batch` 파라미터화 + 프리플라이트 결합

| | |
|---|---|
| **트랙** | T-RUN |
| **파일** | `5_RL/lpopt/design/lattice.py:409` (`run_batch`), `:317-347` (`harvest`), `5_RL/lpopt/config.py:905-906` |
| **내용** | ① 케이스 목록을 **`design_wave.json`** 에서 읽는다; `manifest.json` 에 `design`(설계 튜플 + `gd_positions` + 예측 FF/k)을 함께 기록<br>② `[design] decart_timeout` **5400 → 7200 s** (199 직렬 실측 3,084 s 의 2.33배; 현행 1.75배는 얇다)<br>③ `[design] max_parallel` 은 **2** 로 (실증된 181 레시피). 큐를 폐기했으므로 이 값이 정본<br>④ #2의 프리플라이트를 호출 |
| **테스트** | dry-run 이 케이스 2개를 올바른 덱·별칭으로 계획; 타임아웃 값이 config에서 읽힌다; 멱등성(HGC 존재 시 재실행 안 함) |
| **추정** | 0.5 일 |
| **실행** | 테스트 238 / **실사용 181** (`C:\Users\USER\lpopt_work\assembly_slice_Z_20260903\`) |
| **선행** | #1, #2 |

### 작업 #10b — 181 스테이징 스크립트 (S2b) **[신규]**

| | |
|---|---|
| **트랙** | T-RUN |
| **파일** | **신규** `5_RL/stage_slice_Z_181.ps1` (2_LP `deploy_*` 패턴 재사용) |
| **내용** | 181은 DeCART exe + XS 는 갖고 있으나 **`0_APR1400` 도 `templates_lat1600` 도 paramA 패키지도 없다**(자산 행: "kit_frontier 2026-07-29 stale"). 저작된 덱 2개 + 실행 루트 생성 + `nxfile` 재작성 + **SHA-256 매니페스트 대조**를 한 스크립트로. **`autoeng.toml:59-66` 의 `forbidden = [… "HOST_181" …]` 정책 예외가 기록되기 전에는 실행 금지** 문구를 스크립트 헤더에 박는다 |
| **테스트** | 로컬 dry-run(파일 목록 + 해시 계산만) — **전송은 오너 승인 후** |
| **추정** | 0.3 일 |
| **실행** | 전송은 로컬 세션(파일 복사) / 검증은 **181** |
| **선행** | #1, #10 |

### 작업 #11 — HGC 게이트 G-H1 / G-H1b / G-H1c / G-H2 / G-H4

| | |
|---|---|
| **트랙** | T-RUN |
| **파일** | **신규** `5_RL/lpopt/design/hgc_gates.py`<br>소비: `5_RL/lpopt/data/fuel_types.py:554` (`count_gd_pins_from_hgc`), `5_RL/lpopt/design/lattice.py:374` (`_hgc_looks_valid`) |
| **내용** | `%TITL` **334** = DEPL 62 + BRANCH 16×17; `%DIST`/`%MACX`/`%MICX`/`%ADFT` 각 334; 말미 `%FINE` 1.<br>**G-H1b 크기**: `n_gd ∈ {12,16,20,24}` → **정확히 7,395,955 B**; 그 외 값은 **FAIL이 아니라 ABSTAIN + 수동 검토** (`n_gd=0` 의 6,867,567 B 는 기전 미상, `{4,8}` 은 생산 이력 0).<br>**G-H2**: `count_gd_pins_from_hgc` == 요청 `n_gd`.<br>**G-H4 (회귀검사)**: BU ≥ 0.2 전 구간 \|k_(A) − k_DeCART\| ≤ **100 pcm**; \|FF_(A) − FF_DeCART\| ≤ **0.0021** — 문턱은 T3–T6 홀드아웃의 **실측 최대치**(OPSCREEN.md:165-179)이므로 첫 측정이 아니라 회귀검사다 |
| **테스트** | 정상 HGC PASS / 절단 HGC FAIL / `n_gd=0` ABSTAIN / Gd 인구조사 불일치 FAIL |
| **추정** | 0.7 일 |
| **실행** | 238 (픽스처) / **181 또는 199** (실 HGC) |
| **선행** | #10 |

---

## §5. 패키지 재빌드 단계

### 작업 #12 — 재빌드 전 스냅샷 자동화

| | |
|---|---|
| **트랙** | T-LIB |
| **파일** | `5_RL/lpopt/design/library.py` 래퍼 |
| **내용** | `lib/` + `bases/` + `cores/` + `registry.json` + `designs.json` → `E:\lpopt_archive\<tag>\`. **`.bak` 은 한 세대뿐이고 두 번째 재빌드가 유일한 롤백을 파괴한다**(library.py:80-150: `bak.unlink()` 후 `rename`). 선례: `data/design/package/lib.snap_20260811` |
| **테스트** | 스냅샷이 존재하고 해시가 일치할 때만 빌드가 진행 |
| **추정** | 0.3 일 |
| **실행** | 199 |
| **선행** | — |

### 작업 #13 — `designs.json` 스키마 확장

| | |
|---|---|
| **트랙** | T-LIB |
| **파일** | `5_RL/lpopt/design/package.py:40-56` (`write_designs_manifest`) |
| **내용** | **`gd_positions` 를 선택 → 필수**로 승격(현재 37행 중 4행에만 존재). 신규: `provenance`, **`e2` 정확값**(§7 함정 — `type_id` 는 0.1 w/o 양자화라 4.6750과 4.70이 같은 `47` 로 접힌다), `screen_ff`, `screen_k0`, `screen_crossing_bu`, `screen_model_sha`, **`screen_pattern`(PA/PB)**, `decart_wall_s`, `hgc_sha256`, `deck_sha256`. 현행은 `gd_u_enr = 4.0` 하드코딩 + 레이아웃 없음 |
| **테스트** | 구 매니페스트 하위호환(누락 필드 → `None`); 신규 행에 `gd_positions` 없으면 raise |
| **추정** | 0.4 일 |
| **실행** | 238 |
| **선행** | #1 |

### 작업 #13c — **패키지 재생성** (v1/v2 초안이 통째로 빠뜨린 단계)

| | |
|---|---|
| **트랙** | T-LIB |
| **파일** | **신규 헬퍼** `5_RL/lpopt/design/package.py` (`regenerate_core_templates`)<br>소비: `package.py:91` (`write_core_template`), 검증 `5_RL/lpopt/search/assets.py:296-330` |
| **내용** | 타입 2개를 더하면 `%GEN_DIM` 이 `10 10 27 40 42` → **`10 10 27 42 44`** 가 되어 **`cores/` 10개 폴더가 전부 stale** 이 되고 기존 paramA 쌍이 **Popen 이전 게이트에서 하드 실패**한다. 코드가 이 실패모드를 `assets.py:815-840` 에 그대로 문서화한다(*"a package can carry a STALE bootstrap deck … wrong `%GEN_DIM`"*). `resolver.paramA_library_dims`(resolver.py:70-88)는 **기대치를 계산할 뿐 정합시키지 않는다** — v1의 "자동 정합" 서술은 거짓이다.<br>① `cores/` **10개** 재생성: `P0_P1, Q1_Q2, Q7_Q8, T1_T4_f117, T3_T4, T5_T6, T5_T6_f101, T5_T6_f117, T5_T6_f81, T6_T4`<br>② `data/design/synth_decks/` **purge** (paramA 쌍 18 + ga80 쌍 11 캐시)<br>③ `bases/` 8개 pair 재부트스트랩 트리거 (총 9회 중 8회) |
| **테스트** | 재생성 후 **기존 10개 pair 전부**가 `validate_reload_deck` 통과; purge 후 `synth_decks/` 비어 있음 |
| **추정** | 0.7 일 |
| **실행** | **199** |
| **선행** | #12 |

### 작업 #15 — 라이브러리 게이트 G-H3 / G-H3b / G-H3c / G-H5a / G-H5b / G-H5c

| | |
|---|---|
| **트랙** | T-LIB |
| **파일** | `5_RL/lpopt/design/hgc_gates.py` + 부트스트랩 래퍼 (`design/bootstrap.py` 소비) |
| **내용** | **G-H3 (등식, 허용오차 없음)**: `MAS_HFF == 404,857·N` ∧ `MAS_XSL == 2,010 + 385,849·N_(n_gd>0) + 377,461·N_(n_gd=0)`. **절편은 2,008이 아니라 2,010** (2,008 B `MAS_REF` + CRLF) — 6개 N(11/12/16/33/37/80)에서 오차 0. **N=39 기대값: `MAS_XSL = 15,050,121 B`, `MAS_HFF = 15,789,423 B`**<br>**G-H3b**: 새 `COMP FA_<alias>` 블록의 핵종 로스터가 기존 블록과 **정확히 일치**(`BP01*`/`SB10*`/`MACX*`/`CRD1*` 포함) ∧ 헤더 `62 17 6 0 0`. 개수만 세면 `irod=2` 가 먹는 필드를 못 본다<br>**G-H3c**: `LibraryBuild.set_names` 의 **기존 prefix 순서 불변** + 신규 별칭이 뒤에 append (R21 — `_ALIAS_LETTERS` 는 `P…Z` 다음 `A…O` 라 첫 `A*` 별칭이 앞으로 정렬될 위험이 있으나 TotalBatcher 내부 순서 규칙은 **관측 불가**이므로 기전을 단정하지 않고 **결과를 검사**한다)<br>**G-H5a (cy1)**: `%GEN_DIM` ∧ `%LPD_BCH` 로스터 ∧ `%LPD_C&X`/`%LPD_HFF` 이름 존재. **`validate_reload_deck` 을 쓰면 안 된다** — `%LPD_BCH` 를 담은 덱을 구조적으로 거부한다(assets.py:325-327)<br>**G-H5b**: cy ≥ 2 덱에만 `validate_reload_deck`<br>**G-H5c 수렴**: "평형 ≥2 사이클"이 아니라 `bootstrap_max_cycles = 16` + `cy1_cap_efpd` 안에서 `make_band_restart` 의 5-FOM 비교 **2회 안정** (T6_T4는 11 사이클) |
| **테스트** | 6개 N의 크기 등식 재현; 로스터 일치/불일치; COMP prefix 불변; `%LPD_BCH` 덱에 `validate_reload_deck` 을 쓰면 실패함을 회귀로 고정 |
| **추정** | 1 일 |
| **실행** | 199 |
| **선행** | #11, #12 |

### 작업 #14 — `WaveVerifier(resolver=None)` **위생 + 테스트 가드** (비차단)

| | |
|---|---|
| **트랙** | T-LIB |
| **파일** | 6개 호출자 + 신규 테스트 (`5_RL/lpopt/search/verify.py:852-863` 은 읽기 전용) |
| **내용** | v1은 "하나라도 남기면 HGD569가 재발한다"고 적었으나 **소스가 이미 방어한다** — `resolver is None` 이면 `CaseAssetResolver(self.package_root, …)` 를 만들고 주석이 HGD569를 명시적으로 인용한다. → **사고 방지가 아니라 위생.** 진짜 잔여 위험은 좁다: `registry.json` 없는 `package_root`, 또는 명시적 `registry_aliases={}` |
| **테스트** | `is_paramA_library(library_id)` 이면 `resolver.type_to_alias != {}` |
| **추정** | 0.2 일 |
| **실행** | 238 |
| **선행** | — |

---

## §6. `.sum` 스테이징 + HGC 유도 채널

> **사실 확인:** `data/design/package/hgc/` = **`.HGC` 37 + `.out` 37, 그 외 0개.**
> `lattice.harvest`(lattice.py:317-347)는 `.HGC`/`.out` 만 만들고, `package.stage_hgc`
> (package.py:60-75)도 그 둘만 복사한다. `fuel_types.py:1678` 이 이 사실을 알고 있다 —
> *"No .sum in a package → parse_hgc_full covers the whole curve + coeffs."*
> → **v1의 v9 채널 4개는 원천이 없었다.**

### 작업 #13b — `.sum` + 저작 덱 스테이징

| | |
|---|---|
| **트랙** | T-LIB |
| **파일** | `5_RL/lpopt/design/lattice.py:317-347` (`harvest`), `5_RL/lpopt/design/package.py:31` (`DesignSource`), `:60-75` (`stage_hgc`) |
| **내용** | ① `DesignSource.sum_path` 추가<br>② `harvest` 가 `FA_<alias>.sum` 을 **보존**한다 (181 큐는 이미 남긴다 — `run_decart_eq_xesm_queue_181.ps1:55`)<br>③ `stage_hgc` 가 `.sum` 과 `hgc/dec_FA_<alias>.inp` 를 복사한다<br>④ **★ 덱은 작업 디렉터리에서 복사하면 안 된다** — `harvest` 가 `(wd/"decart.inp").unlink(missing_ok=True)`(lattice.py:346)로 지운다. **`templates_lat1600/…/dec_FA_<type_id>.inp` 또는 `DecartRun.deck_path` 에서** 복사한다.<br>효과: 핀맵이 산문(`gd_positions` 문자열)이 아니라 **바이트로 감사 가능**해진다 (`ingest_fuel_types` 독스트링: *"`zone_pin_count` stays NaN unless a `dec_FA_*.inp` is staged"*) |
| **테스트** | 패키지에 `.sum` 이 `37 + N` 개, `dec_FA_*` 가 `N` 개; `harvest` 후에도 덱 원본이 살아 있다 (회귀) |
| **추정** | 0.4 일 |
| **실행** | 238 (픽스처) / 199 (실 패키지) |
| **선행** | #1 |

### 작업 #17 — ga80 곡선 커버리지 감사 (v9 전제)

| | |
|---|---|
| **트랙** | T-MOD |
| **파일** | 읽기 전용 분석 |
| **내용** | ga80 HGC가 80개 중 **36개만 디스크에 있다**는 사실을 재확인하고 슬롯 커버리지를 센다. v9의 신규 4채널은 **ga80 70개 중 34개에서 결손**이고 그 슬롯은 0.0 센티넬을 받는다 — v4 꼬리붕괴(cyclen 40–50 EFPD 과소예측)와 **같은 기전, 34/70 규모**(70/70이 아니다). v1의 "손해가 0" 표현은 **"기존 곡선 채널과 동일한 34/70 결손 (증가분 0)"** 으로 정정 |
| **테스트** | — (감사 산출물 1개) |
| **추정** | 0.3 일 |
| **실행** | **238** (읽기 전용) |
| **선행** | — |

### 작업 #18 — cond_schema **v9** 사양서 + 사전등록 (**라운드 2 과제**)

| | |
|---|---|
| **트랙** | T-MOD |
| **파일** | `5_RL/lpopt/model/featurize.py:623` (`CHANNELS_BY_SCHEMA` 에 `"v9"` 추가) + prereg md |
| **내용** | **설계 원칙: 핀맵에서 유도되는 채널은 만들지 않는다.** 모델은 원인(Gd 위치)이 아니라 **결과(곡선 형상·FF 곡선·대조)** 를 본다. 이유는 데이터다 — ga80은 `gd_wt` 0%·`zone_pin_count` 전무이고, 핀맵 유래 채널은 ga80 행을 전부 부재-센티넬로 떨어뜨린다.<br>**원천을 `.sum` → HGC 로 교체**: `origin_ff_hot`(HGC `%DIST` map-1 의 BU별 핀출력 → FF(BU) 의 10–25 GWd/tU 평균), `origin_ff_slope`, `origin_ff_bu_peak`, `origin_rho_mid`(HGC `%TITL` 2행 k-inf → rho(BU) 의 BU 0.5–8 평균), 전역 **`g_fresh_contrast`**(68-역할과 53-역할의 평균 rho 차).<br>**`g_fresh_contrast` 가 가장 값이 크다** — `node_peak` 을 R² 0.866으로 설명하는 유일한 측정된 집합체 변수인데 현 인코더의 전역 13/18/20 어디에도 역할 대조 항이 **없다**.<br>규칙: v6c/v7/v8 항목 불변(체크포인트 보호); `_COMPOSITION_WIDTH_BY_FLAG` 는 v8의 5-wide 승계; 부재 시 `_norm_opt` 0.0 센티넬 + `origin_kconv_present` 게이트; **G1 무회귀 게이트를 ga80 부분집합에서 별도 평가** |
| **테스트** | 채널 목록 동결 테스트; v8 체크포인트가 v9 추가 후에도 로드된다 |
| **추정** | 1 일 |
| **실행** | 238 |
| **선행** | #17, **라운드 1 종료** (2–4 타입 × 100 콜로는 부족하다) |

---

## §7. `sdm_mtc` 납품 게이트

> **핵심 사실 (재비평이 잡았고 재확인했다).** 캠페인 내부의 MTC 게이트는 **`min_fxy` 에서
> 구조적으로 도달 불가**다: `build_delivery_payload` 가 `objective != "flat_power"` 이면 `None`
> (campaign.py:533-534, *"only that mode defines a delivery ranking"*), `_maybe_post_verify` 가
> *"no delivery ranking to verify … gate not run"* 으로 조기반환(campaign.py:3053-3062).
> r2 덱이 이것을 **Defect D1**로 적어 두었고(`input_deck.inp:37-45`) r2 `status.json` 은
> `post_verify_master_calls: 0` 이다.

### 작업 #21 — MTC를 **독립 CLI 단계**로 배선

| | |
|---|---|
| **트랙** | T-LIC |
| **파일** | 슬라이스 캠페인 덱 2개 (arm A / arm B) + 실행 런북. 코드 변경 **0줄** (`5_RL/lpopt/cli.py:680` `cmd_sdm_mtc` 를 그대로 쓴다) |
| **내용** | ① 덱의 `[constraints]`: `mtc_enable = true`, `mtc_min_pcm_per_c = -54.0`, `mtc_max_pcm_per_c = 9.0`, **`post_verify_top_k = 5`** ← 라이브 게이트가 읽는 **유일한** top_k(sdm_mtc.py:1517; config.py:1240 기본 **3**). `[sdm_mtc] top_k` 는 `post_verify_topk` 훅 전용이고 그 훅 독스트링이 *"additive — NOT wired into the live loop"*<br>② `[verify] harvest_maps = true` — **`keep_success` 키는 존재하지 않는다.** r2 덱 자신: *"forces the verifier's keep_success … there is no separate keep_success key"*(`input_deck.inp:186-190`). **v2 작업 #21의 `keep_success` 항목은 삭제**<br>③ 실행: `lpopt sdm-mtc --run runs/<arm> --input <arm>.inp --top-k 5`. 이 경로는 `candidates_from_delivery(run_dir, None, top_k)` → 비면 **`select_topk_feasible` 폴백**(cli.py:741-746)이라 **`min_fxy` 에서 실제로 돈다**<br>④ 덱에 **죽은 노브를 남기지 않는다** — 캠페인 내부 게이트는 도달 불가임을 덱 주석에 적는다<br>⑤ 예산: MTC 축 1개 × 후보 5 = **arm당 ~5 MASTER 콜**(config.py:1227 *"~1 extra MASTER call each"*), 2 arm = **~10**. `branch_timeout_s = 300`, `mtc_delta_c = 5.0` |
| **테스트** | 덱 로드 시 `limits.mtc_gated == True`; `post_verify_master_calls`(또는 CLI의 `calls`)가 0이 아님을 실행 후 확인 |
| **추정** | 0.3 일 (문서·덱) |
| **실행** | **199** |
| **선행** | #19 (사전등록 승인) |

### 작업 #21b — `select_topk_feasible` 의 선택 순서 확인 **[신규]**

| | |
|---|---|
| **트랙** | T-LIC |
| **파일** | 테스트 신규 (`5_RL/lpopt/search/sdm_mtc.py:1235-1281` 은 읽기 전용) |
| **내용** | `select_topk_feasible` 은 `runs/<ts>/candidates/*/*/meta.json` 중 `feasible == True` 를 **`\|cyclen − 625\|` 근접순**으로 정렬한다 — **`F_xy` 순이 아니다.** → **PRIMARY 후보(최소 측정 `F_xy`)가 top-5 안에 들었는지 확인하고, 안 들었으면 들 때까지 `--top-k` 를 올려 재실행**한다(추가 1 MASTER 콜/후보). 또한 `_find_candidate_restart` 는 **후보 자신의 수렴 restart** 를 요구하고 교차 폴백을 금지한다 |
| **테스트** | 합성 `candidates/` 트리에서 정렬 키가 cyclen 근접임을 어서트; restart 없는 후보가 skip 된다 |
| **추정** | 0.2 일 |
| **실행** | 238 |
| **선행** | #21 |

### 작업 #20 — SDM 전제조건 (full-core rod model 이식) **[라운드 1 비차단]**

| | |
|---|---|
| **트랙** | T-LIC |
| **파일** | `5_RL/lpopt/search/sdm_mtc.py` 소비자 + 신규 자산; 이식 원본 = MOCHA `build_apr1400_rod_model` |
| **내용** | `campaign._rod_model()`(campaign.py:3012-3023)이 **설계상 `None`** 을 반환한다: *"The campaign deck is QUARTER-core and carries no `%ROD_CFG`/`%ROD_MAP` … lpopt has no full-core asset package, so this returns `None` until one exists — and the gate then reports SDM as INCONCLUSIVE rather than inventing a rod map."* `lpopt sdm-mtc` CLI 경로도 `sdm_params` 를 넘기지 않으므로 동일하다.<br>**★ 트랩 사전등록:** 프로그램적 훅 `post_verify_topk(rod_model=…)` 의 기본 `scram_banks = ("R1"…"R5")`(sdm_mtc.py:1620-1624)는 **config가 스스로 금지한 설정**이다 — `config.py:1295-1303` 이 *"an R-only default cannot reach the 10,870 pcm requirement for ANY pattern — every candidate would FAIL for a reason that is a config defect, not physics"* 라고 적고 `[sdm_mtc]` 기본값을 `["R1"…"R5","B","A"]` 로 둔다. → **`scram_banks`/`stuck_candidate_banks` 는 반드시 `[sdm_mtc]` 에서 가져온다** |
| **테스트** | SDM이 INCONCLUSIVE를 벗어난다; 뱅크 스코프가 `[sdm_mtc]` 에서 온다 |
| **추정** | 2–3 일 (자산 이식) |
| **실행** | 199 |
| **선행** | — (라운드 1 비차단) |

---

## §8. acquisition 의 `FuelDesign` 축 트리거 항 — **의존상 마지막**

> **지금 없는 것.** lpopt의 모든 목적함수는 **케이스를 고정한 채 패턴을 랭킹한다** —
> `score_min_fxy`(acquisition.py:841), `score_min_fr_max_cycle`(:489), `MinFuelCostSpec`(:1054),
> `score_flat_power`(:1799). **`FuelDesign` 축 위의 목적함수 항은 0개**이고 "설계 1달러당
> 기대개선"은 정의된 적이 없다. 다만 가상 집합체를 꽂을 구멍은 뚫려 있다 —
> `construct.build_pair_universe(types=...)`(construct.py:770), `achievable_e_core_interval`(:748),
> `screen_e_core_band`(:847), `fuelcost_search.cell_fe_prior`(:97) 계열.
>
> **슬라이스는 §8을 쓰지 않는다** (가정 A6: 슬라이스 우선). §8은 **라운드 2의 자동 발사**를 위한 것이다.

### 작업 #0 — σ_chain 소급검증 (**병렬 트랙, 슬라이스 비차단**)

| | |
|---|---|
| **트랙** | T-SCR |
| **파일** | 신규 분석 스크립트 1개 + 리포트 1개 (코드 프로덕션 변경 0) |
| **내용** | 사슬의 **짝지은 차분 오차** `σ_chain,paired` 를 측정한다. 입력은 전부 디스크에 있다 — `fr_arms` 15–17 arm(**바이트 동일한 단일 장전패턴**에서 fresh만 교체), T3–T6 4행, OPSCREEN B-arm 2건, 측정 `F_xy` 라벨 7개, r2 수렴행 99개.<br>P1 `node_peak` 회귀를 **leave-one-arm-out** 재적합(n=15는 적합 모집단이라 in-sample 잔차가 오차를 과소평가) — `A` 도 홀드아웃에서 `s09b_contrast.py:49` 와 **동일한 최소자승 스케일**로 재추정<br>P2 절대 성능(bias/MAE/p95) · P3 **짝지은 차분** 을 (a) 동일 패턴 내 (b) 셀 간 으로 **분리** 보고 · P4 헤드 그림자 점수 · P5 T3–T6 4행으로 밀도·Xe 계통편차 회귀검사<br>**★ 표기 정정:** `+0.0614` 는 **이 사슬의 측정이 아니다** — `minfxy_E1E2_f121_r2_results_20260831.md:133` 의 그 행은 `ratio 1.0588 · F̂_r` 이고 `F̂_r` 은 **s1j 헤드 예측**(자체 오차 +0.0439)이다. **"해석적 0.065와 실측 0.0614가 일치한다"는 문장을 삭제**하고 σ 주장을 해석적 전파 하나에 둔다 |
| **통과바** | `<0.005` → `k=2` 바 그대로 · `0.005–0.020` → 바 = `2σ`, 후보 재계산 · `>0.020` → **F1–F5 트리거 폐기**, 라운드 2도 물리실험. **기저 시나리오는 세 번째** (`node_peak` rms 0.036은 후보마다 contrast가 달라 차분에서 상쇄되지 않는다) |
| **테스트** | leave-one-arm-out 재현성(동일 시드 → 동일 수); 헤드 그림자 점수 동시 산출 |
| **추정** | 0.5 일 |
| **실행** | **238** (`scratch/records_r2_76793.parquet`, 읽기 전용) |
| **선행** | — |

### 작업 #7 — 필요신호 `N` (T-A)

| | |
|---|---|
| **트랙** | T-TRG |
| **파일** | **신규** `5_RL/lpopt/design/need.py` |
| **내용** | `FuelVec` 섭동 유한차분으로 "어떤 기술자 방향이 목적함수를 움직이는가"를 잰다. **geometry 채널은 제외** — `ood_guard` 포락이 퇴화 `[0,0]` |
| **테스트** | 채널 순열 불변; OOD 채널 제외 확인 |
| **추정** | 1 일 · **실행** 238 · **선행** #4 |

### 작업 #8 — 가상 로스터 (T-B)

| | |
|---|---|
| **트랙** | T-TRG |
| **파일** | `5_RL/lpopt/search/construct.py` (읽기 전용 소비) + `need.py` |
| **내용** | `build_pair_universe(types=...)` 에 가상 타입을 주입해 도달 가능한 `e_core` 밴드를 본다 |
| **테스트** | 가상 타입의 `e_core` 밴드가 `achievable_e_core_interval` 과 정합 |
| **추정** | 0.7 일 · **실행** 238 · **선행** #7 |

### 작업 #9 — 발사 규칙 F1–F5 + 판정 JSON

| | |
|---|---|
| **트랙** | T-TRG |
| **파일** | `5_RL/lpopt/design/need.py` |
| **내용** | **F1** task #0 완료 & σ 측정됨 (없으면 발사 금지) · **F2** 짝지은 예측 `ΔF̂_xy ≤ −k·σ_chain,paired`, `k = 2` · **F3** (A) **경계** 안 (격자 위반 허용, 경계 위반 금지) · **F4** 역할 대조 `contrast ≥ 0.026` (쌍 단위) · **F5** 운전점 `cyclen ∈ [620,645]` ∧ `CBC ≤ 1500`.<br>**신경망 헤드는 그림자로 강등** — s1j 셀내 `F_xy` 순위 충실도는 측정된 실패(r2 RANK 3/3 FAIL; pinbu r1 헤드 셀내 ρ −0.11; arm 4 G5 R1 ρ 0.535, n=95 CI ∋ 0)이고, **사슬이 헤드를 이긴다는 주장도 근거가 없다**(유일한 동일 모드 비교에서 헤드가 나았다). **둘 다 게이트에서 뺀다** |
| **테스트** | 다섯 조건 각각의 거부 케이스 1건씩; F1 미충족 시 발사 거부 |
| **추정** | 0.7 일 · **실행** 238 · **선행** #0, #8 |

---

## §9. 문서

### 작업 #19 — 사전등록 (완료, 승인 대기)

| | |
|---|---|
| **트랙** | T-DOC |
| **파일** | `5_RL/data/reports/assembly_slice_Z_prereg_20260903_DRAFT.md` (**작성 완료**) |
| **내용** | 후보 Z 완전 지정 · σ 실험 · DeCART 실행 계획 · 게이트 · 캠페인 · 마크 · 예산 · 처분 |
| **다음** | **오너 승인** → `_DRAFT` 접미사 제거하여 동결 |
| **추정** | — · **선행** §11 가정 A1–A10 확정 |

---

## 부록 1 — 무엇이 어디서 도는가

| 호스트 | 주소 | 이 계획에서의 역할 | 금지 |
|---|---|---|---|
| **로컬 PC** | `mk-ctrp` (= 문서상의 "box 104") | **편집·Read/Grep·파일 전송만** | **모든 연산 금지** (사용자 상시 규칙). `D:\DeCART_MASTER\BIN` 에 omp exe + `libiomp5md.dll` + prolog + MASTER가 **실재**하지만 쓰지 않는다 |
| **238** | `USER@HOST_238:8022`, `~/lpopt_ws` venv | task #0 · 스크리닝 · **덱 저작** · 모든 단위 테스트 | — |
| **181** | `USER@HOST_181` | **DeCART2D 웨이브** (직렬, 2-wide, `C:\Users\USER\lpopt_work\…`) | `autoeng.toml:59-66` 이 **forbidden** 으로 등재 → **예외 기록 전 실행 금지** |
| **199** | `USER@HOST_199`, `C:\Users\USER\lpopt_work\kit_frontier` | 라이브러리 재빌드 · 패키지 재생성 · MASTER 부트스트랩 9회 · 전이 스윕 · 캠페인 · MTC | lpopt **생산 호스트** — DeCART로 경합시키지 않는다 |

**패키지 이송 시 따라가는 것 (확인함):** `data/design/package/lib/` 안에
**`prolog41m4.exe`(1,451,520 B)** 와 **`TotalBatcher4.exe`(300,544 B)** 가 이미 스테이징되어 있다
→ 패키지를 199로 보내면 빌드 툴이 함께 간다.

## 부록 2 — 추정 합계

| 트랙 | 작업 | 합계 |
|---|---|---:|
| T-LAT | #1, #2, #3 | **2.1 일** |
| T-MOD | #16, #17, #18 | 1.6 일 (#18은 라운드 2) |
| T-SCR | #4a, #4, #5, #6, #0 | **4.0 일** |
| T-RUN | #10, #10b, #11 | **1.5 일** |
| T-LIB | #12, #13, #13b, #13c, #14, #15 | **3.0 일** |
| T-LIC | #21, #21b, #20 | 2.5–3.5 일 (#20은 비차단) |
| T-TRG | #7, #8, #9 | 2.4 일 (라운드 2) |
| **슬라이스 임계경로 소계** (#1,#2,#16,#4a,#4,#5,#10,#10b,#11,#12,#13,#13b,#13c,#15,#21,#21b) | | **≈ 9.5 일** |
| 슬라이스 **실행** (사전등록 §9) | | **≈ 2–2.5 근무일** |

## 부록 3 — 재비평이 이 목록을 바꾼 곳

| 재비평 | 반영된 작업 |
|---|---|
| MTC가 `min_fxy` 에서 안 돈다 (physics #1 / eng A1) | **#21 전면 재작성** (독립 CLI), **#21b 신설** |
| `top_k` 노브 (physics #2 / eng A2) | #21 — `[constraints] post_verify_top_k = 5`, 예산 arm당 5 |
| `keep_success` 는 키가 아니다 (eng A9) | #21 — 항목 **삭제** |
| `d_fresh` ≠ `hump` (physics #3/#4) | **#5 테스트 ③ 신설** (두 값이 다름을 어서트) |
| (A) 접근이 임계경로 (physics #11) | **#4a (S0c) 신설, 경성 선행** |
| 181에 자산 없음 (eng A5) | **#10b 신설** |
| box 104 = 로컬 PC (eng A4) | 부록 1 — **사용 금지 명시** |
| `pin_map` 은 선택 인자 (eng A10) | #16 — "게이트가 요구"가 아니라 "우리가 선택해 넘긴다" |
| `harvest` 가 덱을 지운다 (eng A11) | #13b ④ |
| `n_gd` 물리 근거 (eng A13) | #4 ④ — `SURROGATE_USAGE.md:143` |
| `scram_banks` R-only 트랩 (eng A14) | #20 — 트랩 사전등록 |
| task #0 결정 귀결 (physics #13) | #0 — **병렬·비차단**으로 명시, 바는 라운드 2 전용 |
| 열거 창 모순 (physics #12 / eng A6) | #4 ⑤ — `--enumerate` 가 238에서 산출, 3,738 인용 금지 |
| `type_id` 양자화 충돌 (physics #14) | #1 ⑤ + #13 (`e2` 정확값 기록) |
| G-H3 출처 라벨 (physics #6 / eng A12) | #15 — 사전등록 §4.3 표를 정본으로 |
