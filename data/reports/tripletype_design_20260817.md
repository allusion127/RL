# 3-신연료종 (graded) 장전 확장 — 설계 노트

**2026-08-17 · BUILD + TEST only.** 캠페인 미착수, MASTER 미실행, 재학습 미실시.

기존 탐색기는 신연료를 **2종(pair, `A_B`)** 으로만 다뤘다. 상용 노심은 3종 이상을
써서 반경방향 반응도를 더 잘게 계단화(grading)하고, 그것이 F_r을 낮추는 정공법이다.
이 변경은 **3종 캠페인이 가능해지는 최소 변경**만 넣었고, **2종 경로는 바이트 동일**을
유지한다(사후 sha256 대조로 확인).

---

## 1. 무엇이 바뀌었나

| 파일 | 변경 |
|---|---|
| `lpopt/search/genome.py` | `case_batches()` (2종/3종 케이스 문자열 파싱, 2종은 vendor `_pair_batches` 그대로 위임) · `MAX_FRESH_TYPES=3` · `GeneralOrbitGenome.batch_counts` · **`graded_morph()`** 신규 연산자 · `_one_move` 배치 분기 확장 · `random_genome`이 3종 알파벳 사용 |
| `lpopt/data/fuel_types.py` | `mix_e_core()` (n종 조성가중 e_core) · `case_e_core()` (2종은 기존 `pair_e_core` 호출 그대로 위임 → 바이트 동일 + duck-typed fake 호환) · `FuelLibrary.case_e_core()` |
| `lpopt/search/assets.py` | `_pair_e_core()` 가 2~3종 처리 → **level-3 `pair_ecore` 사다리가 triple 셀도 점수화** · `alias_case_key()` 가 멤버별 alias 매핑(3종 포함) |
| `lpopt/search/construct.py` | `CaseContext.batches -> tuple[str, ...]` (캠페인 경로가 3종을 그대로 탐색) |
| `lpopt/search/campaign.py` | `_case_e_core()` 조성가중화 (기존 `a, b = split("_")` 는 triple 에서 ValueError 후 조용히 store median 으로 새던 자리) |
| `lpopt/search/produce.py` | `_apply_split()` 3종 분기(첫 종 `w1`, 나머지 균등) · `_enrichment()` 조성 e_core + `max-min` e_split |
| `lpopt/search/boundary_probe.py` | `case_batches` 사용 (triple 하드크래시 제거) |
| `lpopt/design/coredeck.py` | `_lpd_bch_quarter()` 가 2종/3종 부트스트랩 맵 생성(3종은 `X1` 셀을 b/c 교대 배치) |
| `lpopt/design/bootstrap.py` | `make_band_restart()` 멤버 파싱을 `case_batches` 로 (`partition("_")` 는 triple 에서 `b="P5_S5"` 로 깨졌다) |
| `lpopt/model/featurize.py` | `e_split` 을 전 멤버 `max-min` 으로 일반화(2종에서 `|e_A−e_B|` 와 항등) · **`cond_v7` 신규 스키마** |
| `tests/test_triple_type.py` | 신규 41 케이스 |
| `tests/test_v5_schema.py`, `tests/test_leakage.py` | 스키마 tripwire / leakage 목록에 `v7` 추가 |

### 1.1 `graded_morph` — 2종 엘리트 → 3종 cold-start
최적화가 끝난 2종 노심은 **이미 반경방향으로 계단화되어 있다.** 그래서 3종 공간의
초기 seeding은 판을 다시 무작위화하는 게 아니라, *가장 많은* 배치에서 *반경 극단 쪽
연속 슬라이스*(기본 34%)를 *가장 적은* 종으로 재라벨하는 것이다. 라벨은 구조적으로
무해하므로 **feed · wiring · depth-2 수가 정확히 보존**된다(테스트로 고정).
2종 알파벳에서는 no-op이라 2종 move stream이 바이트 동일하게 남는다.

### 1.2 `cond_v7` — 조성 모멘트 (additive)
셀 채널은 **v6c와 완전 동일**(62ch). 슬롯 채널은 이미 자기 origin type을 들고 있어서
알파벳이 늘어도 새 채널이 필요 없다. 늘어난 건 **글로벌 벡터뿐(13 → 18)**, append-only:

| 모멘트 | 글로벌 |
|---|---|
| 평균 | `g_e_core` (기존) |
| 퍼짐 | `g_e_split` = 급전 종들의 `max−min` (기존, 2종에서 항등) |
| 종별 분율 | **`g_type_frac_1..3`** (사전순, 3까지 0-padding) |
| 2차 모멘트 | **`g_e_type_std`** (급전분율 가중 표준편차) |
| 종 수 | **`g_n_fresh_types`** (급전된 distinct 종 / 3) |

`g_e_type_std` 가 핵심이다: {5.60, 5.79} 50/50 과 {5.60, 5.67, 5.79} 은 평균도
`max−min` 도 같지만 std가 다르다 — 즉 **계단화 자체를 모델이 볼 수 있게 하는 유일한 채널.**
2종 레코드에서 `g_type_frac_1 == g_split_frac`, `g_type_frac_3 == 0` 으로 환원된다.

---

## 2. 2종 동일성 판정

변경 **직전** 코드에서 sha256 스냅샷을 떠 두고 변경 후 재대조 → **15/15 키 전부 바이트 동일.**

* 2종 genome 이동열 (3 pair × 4 feed × 61 패턴, 고정 seed) — `76a4149a…`
* v3 / v6b / v6c featurization (5개 library × 6행 = 30행의 cells·globals) — 각각 동일
* `pair_e_core` 값들, cycle-1 / reload 덱 텍스트 — 동일

세 해시는 `tests/test_triple_type.py` 에 상수로 고정되어 회귀 시 즉시 잡힌다.

---

## 3. 첫 3종 캠페인에 필요한 것

### 3.1 어떤 triple 을 고를 것인가 — e5.69 셀

현 챔피언 셀 `P6253Z1G06N24_P6253Z2G10N24` @ feed 109 (paramA, e_core 5.6944)은
붕소벽이 열린 셀이고(CBC floor 1405.2, 16/16 통과) **F_r 이 구속조건**이다
(in-cell floor 2.0481 @f109 / 1.8375 @f125, 게이트 1.55). 계단화가 바로 그 F_r 을
겨냥한다.

권고 triple (**모두 이미 paramA 패키지 roster 에 존재** — alias `S3` / `P5` / `S5`):

| 역할 | type_id | alias | e [w/o] | n_gd | Gd₂O₃ wt% |
|---|---|---|---|---|---|
| hot | `P6253Z1G06N24` | S3 | 5.7861 | 24 | 6 |
| **mid (신규)** | **`P6253Z2G08N16`** | **P5** | **5.6685** | 16 | 8 |
| cold | `P6253Z2G10N24` | S5 | 5.6023 | 24 | 10 |

* case id: `P6253Z1G06N24_P6253Z2G08N16_P6253Z2G10N24` (store/type_id 공간),
  runner 는 `alias_case_key` 로 `S3_P5_S5` 로 자동 변환됨.
* 균등 1/3 조성 e_core = **5.6858** (pair 5.6944 대비 −0.009).
  **pair 와 e_core 를 정확히 맞추려면** 급전분율 `(hot, mid, cold) = (0.380, 0.333, 0.286)`
  → e_core 5.6944. 이러면 *같은 e_core · 같은 feed · 계단만 더 촘촘한* 통제된 비교가 된다.
* 반응도 계단: 0.066 / 0.118 w/o — 대안 `P6253Z2G10N20`(T1, 5.6386, n_gd 20)은
  계단이 0.036 / 0.148 로 더 비대칭이지만 **n_gd 를 20으로 유지**한다.
  mid 에 P5(n_gd 16)를 쓰면 중간 1/3의 Gd 봉수가 24→16으로 줄어드는 점은
  pin-BU 게이트(≤78) 재확인 대상. **n_gd 보존이 우선이면 T1, 계단 균일성이 우선이면 P5.**

### 3.2 모델 — 재학습 필요 (필수)
* 현 챔피언(s1g)은 **v6b/v6c 계열**이고 글로벌 13개다. `cond_v7` 은 18개 → **checkpoint
  호환 불가**, `model_api` 가 글로벌 차원 불일치로 정직하게 거부한다. 3종 셀을 *예측으로*
  이끌려면 `cond_schema="v7"` 로 **전량 재학습**이 필요하다.
* 다만 v7 은 v6c 위의 append-only 이고 2종 레코드가 바이트 동일하게 featurize 되므로,
  **기존 ~39k 2종 레코드를 그대로 학습에 재사용**할 수 있다(`g_type_frac_3=0`,
  `g_e_type_std` 는 2종 값). 즉 3종 라벨은 처음부터 많이 필요하지 않고,
  2종 코퍼스가 v7 의 사전(prior) 역할을 한다.
* 재학습 전 단계로는 **모델 없이 sampler/DoE 로 3종 라벨을 먼저 쌓는 것**이 정석이다
  (produce 경로가 3종 split 을 그리게 이미 확장됨).

### 3.3 Cold start — graded morph 로 seeding
* 해당 pair 는 store 에 47행(수렴 32행, f109/f125)이 있다. 이들 엘리트를
  `from_pattern` → `graded_morph(..., batches=(hot, mid, cold))` 로 3종화하면
  **feed·wiring 보존 + 중간종만 반경 슬라이스로 삽입**된 seed 가 나온다.
  캠페인 pool 에서는 `mutate(..., batches=<3종>)` 이 이 연산자를 확률 0.25×batch_prob 로
  자동 인출한다.
* **restart**: `S3_P5_S5_f109` 전용 base restart 는 없다. resolver level-3
  (`pair_ecore`) 가 조성 e_core 5.686(또는 e-matched 5.694)로 점수화해서
  **|Δe| ≈ 0.01 인 기존 pair restart** 를 집어온다 — 이번 변경으로 triple 도 이
  사다리를 탄다(예전엔 `len(parts)!=2` 로 None 반환 후 neutral 로 추락했다).
* deck: reload 덱의 `%LPD_SHF` 는 위치별 종을 이미 인코딩하므로 **덱 변경 불필요**.
  cycle-1 부트스트랩이 필요할 때만 `build_cycle1_deck(aliases, (hot, mid, cold))`.

### 3.4 착수 전 체크리스트
1. [ ] mid 종 확정: **P5(계단 균일) vs T1(n_gd 보존)** — pin-BU ≤78 게이트 영향 판단
2. [ ] 급전분율 확정: 균등 1/3(e 5.686) vs e-matched (0.380/0.333/0.286 → 5.6944)
3. [ ] `[[produce.strata]]` 또는 `[case] pair` 에 3종 case id 기입, feed 109 고정
4. [ ] resolver 로그에서 level-3 히트 및 `|Δe|` 확인 (neutral 로 떨어지면 중단)
5. [ ] 3종 라벨 N개(≥60) 수집 후 `cond_schema="v7"` 재학습 → 앙상블 교체
6. [ ] `graded_morph` seed 의 F_r 이 2종 부모 대비 개선되는지 **첫 wave 에서 판정**
7. [ ] 3종 레코드가 store 에 들어간 뒤 `e_split`(=max−min) 값이 hot−cold 로 채워지는지 확인

---

## 4. 테스트

`tests/test_triple_type.py` 41 케이스 통과. 커버리지:
알파벳 파싱/거부 · **2종 이동열 sha256 고정** · graded_morph 2종 no-op ·
3종 invariant(1+4N, 60−2N, strict 소비) · **5,000-move 3종 퍼즈** ·
graded_morph 구조보존/단일donor/반경연속/배치 비움 방지 ·
**2종 엘리트 → `mutate(batches=3종)` 로 3rd type 도달(≥35/40 seed)** · **3종 %LPD_SHF 왕복
(canonical·digest·from_pattern 동일)** · `validate_case` · `CaseContext` ·
`mix_e_core ≡ pair_e_core` · triple resolver 점수화 · alias 매핑 ·
**v6c featurization sha256 고정 + v7 prefix 동일** · v7 3종 인코딩 ·
std 판별력(같은 spread, 다른 std) · cycle-1 덱 sha256 고정.

인접 스위트 재실행: `test_genome_general` / `test_leakage` / `test_cond_v6b` /
`test_cond_v6c` / `test_v5_schema` / `test_construct` / `test_assets` /
`test_design` / `test_produce_ledger` / `test_campaign_stub` /
`test_boundary_probe` / `test_paramA_produce_kit` / `test_vendor_closure` /
`test_fuel_types` / `test_geomcheck` → **전부 통과**.

**전체 스위트: 1913 passed / 6 failed / 2 skipped (25분).** 실패 6건은 모두 이번
변경과 무관하며, 다음 두 가지로 증명했다.

| 실패 | 원인 |
|---|---|
| `test_featurize::test_core_enrichment_split_reproduces_stored_e_core` | store 정합 — paramA 행의 저장된 `e_core` 가 갱신된 `fuel_types.parquet` 과 불일치 (4.8715 vs 4.8926) |
| `test_cell_calibrate::test_cyclen_e_core_keys_paramA_into_a_real_bin` | **동일 원인·동일 수치** (`core_enrichment_split` 경유, 미변경 함수) |
| `test_axial_head` 3건 | `maps.npz` 축방향 라벨 정합 (오늘 05:05 store 재작성) |
| `test_remote_infer::test_remote_gpu_matches_local_cpu_determinism` | GPU/CPU 수치 허용오차 (max\|local−remote\| ~4e-3) |

증명 (1) — **호출 추적**: 위 실패 테스트 전체를 실행하면서 이번에 추가한
`mix_e_core` · `case_e_core` · `case_batches` · `graded_morph` 를 래핑한 결과
**호출 0회**. 즉 신규 코드는 이 경로에 존재하지 않는다.
증명 (2) — **되돌림**: 신규 `fuel_types` 코드를 제거한 상태에서
`test_core_enrichment_split_reproduces_stored_e_core` 가 동일하게 실패함을 확인.

> (참고) 앞의 두 e_core 실패는 저장된 `e_core` 컬럼이 낡은 것이므로, 3종 캠페인
> 착수와 별개로 **paramA 행 e_core 재계산/백필**을 검토할 것.

---

## 부록 — 3종 → **5종** 확장 (2026-08-17, 같은 날)

운영자 지시 **"3~5종 그물망"**. 위 3종 확장 **직후, 그 위에** 얹었다.
BUILD + TEST only — 재학습·캠페인·MASTER 모두 미착수.

### A.1 무엇이 바뀌었나 — 상한값과 블록 폭뿐

| 파일 | 변경 |
|---|---|
| `search/genome.py` | `MAX_FRESH_TYPES` **3 → 5** · `case_batches` 오류문/도크 일반화 · `graded_morph`에 **선택 인자 `donor`/`target`** (알파벳의 **임의 쌍** 사이 변환) |
| `search/assets.py` | `_pair_e_core` · `alias_case_key` 의 하드코딩 `<= 3` → `MAX_FRESH_TYPES` (import 추가) |
| `design/coredeck.py` | `_lpd_bch_quarter` 의 `X1` 존을 `types[1:]` **라운드로빈**으로 (2종=항상 `b`, 3종=`b/c` 교대와 **수식적으로 동일** → 두 덱 모두 바이트 동일) |
| `design/bootstrap.py` | 오류문 `2..MAX_FRESH_TYPES` |
| `model/featurize.py` | **`cond_v8` 신규** · `MAX_FRESH_TYPES` 3 → 5 · `_COMPOSITION_WIDTH_BY_FLAG` (스키마별 블록 폭 고정) |
| `tools/probe_assets.py` | `pair.partition("_")` → `case_e_core` (3종 이상 덱에서 **NULL e_core 오경보**가 나던 자리; 3종 diff 가 놓친 site) |
| `tests/test_triple_type.py` | 4/5종 · v8 · **3종 sha 핀** 추가 (85 케이스) |
| `tests/test_v5_schema.py`, `tests/test_leakage.py` | tripwire / leakage 목록에 `v8` |

**나머지는 손대지 않았다.** 3종 diff 의 모든 site 를 다시 감사한 결과
`case_batches` · `GeneralOrbitGenome.batch_counts` · `_batch_flip` · `_batch_swap` ·
`_one_move` · `random_genome` · `mix_e_core` · `case_e_core` · `CaseContext.batches` ·
`campaign._case_e_core` · `produce._apply_split` / `_enrichment` · `boundary_probe` ·
`bootstrap.make_band_restart` · `featurize` 의 `e_split(max−min)` 는 **이미 `batches`
전체를 도는 알파벳 일반 코드**였다. 즉 4·5종 경로는 3종 경로에 알파벳만 길어진 것이다.
구조 불변식(1+4N 급전격자, 60−2N, strict 소비, depth 캡)은 **배치 라벨을 읽지 않으므로**
알파벳 길이와 무관하다.

### A.2 `cond_v8` — v7 의 조성 블록을 3폭 → 5폭

셀은 여전히 **v6c 62ch 그대로**, v6c 글로벌 13개 접두도 **인덱스 불변**.
바뀐 건 조성 블록뿐: 글로벌 **13 → 20**.

| 모멘트 | v7 | **v8** |
|---|---|---|
| 종별 분율 | `g_type_frac_1..3` | **`g_type_frac_1..5`** |
| 2차 모멘트 | `g_e_type_std` | 동일 |
| 종 수 | `g_n_fresh_types` = distinct/**3** | distinct/**5** |
| 글로벌 수 | 18 | **20** |

**v7 은 한 비트도 건드리지 않았다** — v7 재학습(238)이 진행 중이므로 폭을 스키마별로
`_COMPOSITION_WIDTH_BY_FLAG` 에 **못 박았다**(v7→3, v8→5). 기존 항목 수정은 곧
학습된 체크포인트의 입력을 조용히 옮기는 일이므로 금지다.

v8 은 **재인코딩이 아니라 넓히기**다: 2종·3종 레코드는 v8 에서 v7 과 **정확히 같은 정보**를
싣는다(셀 동일 · v6c 글로벌 동일 · `frac_1..3` 동일 · `g_e_type_std` 동일,
`frac_4/5 = 0`, `g_n_fresh_types` 만 분모 3→5 로 재정규화 — 값 자체는
`v8×5 == v7×3` 로 왕복 보존). 따라서 **기존 ~39k 2종 코퍼스 + 3종 라벨을 그대로
v8 학습에 재사용**할 수 있다. 이 재현은 테스트로 고정되어 있다.

### A.3 4/5종 캠페인에 **구조적으로 추가로 필요한 것 — 없다**

3종 착수 체크리스트(§3.4)가 그대로 5종에도 적용된다. 개별 확인:

* **덱** — `%LPD_SHF` 는 위치별 종을 인코딩하므로 5종도 **덱 변경 불필요**.
  cycle-1 부트스트랩만 `build_cycle1_deck(aliases, (t1..t5))` — 라운드로빈으로 5종이
  같은 legacy 존에 퍼진다. `%GEN_DIM` 은 이미 `len(aliases)` 로 스케일된다.
* **restart** — level-3 (`pair_ecore`) 사다리가 5종 셀의 조성 e_core 를 계산해
  `|Δe|` 로 기존 pair restart 를 집어온다(3종과 동일한 rung).
* **cold start** — `graded_morph` 기본 인출은 *가장 적은* 종을 겨냥하므로 2종 엘리트에서
  반복 인출하면 없는 종을 차례로 채운다(4종·5종 도달 테스트로 고정). 방향을 정해
  재계단화하려면 `donor=`/`target=` 으로 임의 쌍을 직접 지정한다.
* **produce** — `_apply_split(w1, 나머지 균등)` 과 `_enrichment` 는 이미 n종이다.
* **모델** — v7 과 똑같이 **전량 재학습 필요**(글로벌 18 vs 20 → `model_api` 가 정직하게 거부).
  v8 을 쓰려면 v8 로 재학습해야 하고, 그 전 단계는 sampler/DoE 로 라벨을 쌓는 것이다.

**단 하나의 비구조적 관문**(5종이 아니라 *3종 이후 전부*에 해당, 기존부터 그러함):
`data/compliance.py::is_cross_anchor` 는 멤버가 정확히 2개가 아니면 **cross-anchor 로 간주**해
`assert_mono_anchor` 가 하드 실패한다. 이건 R1 정책(급전 집합체 간 농축 사양 혼합 금지)이고,
`multi_pc` 로스터 export 와 `frontier_search._PAIR_ECORE` 에만 걸린다 — 캠페인/produce 경로는
지나지 않는다. **3종과 5종이 완전히 동일한 상태**이므로 이번에 손대지 않았다.
graded 케이스를 로스터에 올릴 때 R1 을 어떻게 읽을지는 **정책 판단**이므로 별건으로 결정할 것.

### A.4 동일성 판정

| 대상 | 판정 |
|---|---|
| **2종 이동열** sha256 | `76a4149a…` **불변** |
| **3종 이동열** sha256 (신규 핀 `a40d6886…`) | 3종 빌드의 `graded_morph` 를 **축자 재구성**해 대조 → **비트 동일** |
| v6c featurization cells/globals sha256 | 둘 다 **불변** |
| 2종 cycle-1 덱 sha256 `086434d4…` | **불변** |
| v7 글로벌 (18, 폭 3, 분모 3) | **불변** |
| leakage 바이트 동일성 | v3/v6/v6b/v6c/v7 + **v8** 전부 통과 |

### A.5 테스트

`tests/test_triple_type.py` **85 케이스 통과**(41 → 85). 추가분:
5종 알파벳 파싱 + 상한 정확도(6종 거부) · **3종 이동열 sha 핀** ·
4/5종 invariant + **각 5,000-move 퍼즈** · 4/5종 전 종 도달 ·
**5종 %LPD_SHF 왕복**(canonical·digest·from_pattern) · 5종 `validate_case` ·
4/5종 `CaseContext` · 5원 조성 `mix_e_core` · 4/5종 resolver 점수화 + 6종 거부 ·
5종 alias 매핑(초과 시 fail-safe) · `graded_morph` **임의 쌍 20조합 변환**(단일 donor ·
반경 연속 · 타 종 불변 · donor 비우기 방지) · 불가 쌍 거부 · 알파벳 순회 ·
**v8 append/폭/이름** · v7 불변 · **2종·3종의 v7↔v8 정보 동일성** · 4/5종 v8 인코딩 ·
5종 mesh vs 2종 hull 의 std 판별력 · 4/5종 cycle-1 덱.

인접 스위트 재실행: `test_genome_general` · `test_leakage` · `test_cond_v6b` ·
`test_cond_v6c` · `test_v5_schema` · `test_construct` · `test_assets` · `test_design` ·
`test_produce_ledger` · `test_campaign_stub` · `test_boundary_probe` ·
`test_paramA_produce_kit` · `test_vendor_closure` · `test_fuel_types` · `test_geomcheck`
→ **252 passed**.
`test_featurize` · `test_config` · `test_model_api` · `test_model_net` · `test_fuel_cond_v4` ·
`test_net_shape_flags` · `test_hires_bundle` · `test_compliance` · `test_v5_experiment` ·
`test_v5_training_integration` · `test_dataset_torch` · `test_splits` · `test_store`
→ **236 passed / 1 failed**. 그 1건은 §4 에 이미 기록된 **기존 실패**
(`test_core_enrichment_split_reproduces_stored_e_core`, 4.8715 vs 4.8926 — 저장된
`e_core` 컬럼 노후) 로 수치까지 동일하다.
