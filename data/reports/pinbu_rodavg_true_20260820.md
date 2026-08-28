# 핀 연소도 봉평균 — 실측 종결 (M2 close, 2026-08-20)

**성격**: `pinbu_rodavg_20260820.md`(M2 재파싱 종결 시도, VOID 판정)의 후속.
그 문서가 "재파싱만으로는 불가, MASTER 재실행 필요"로 결론낸 지점에서 시작해,
박스 199에서 5기 f113-PIN 코어를 `[master] keep_success = true`로 **재실행**하고,
살아남은 실제 `MAS_PPI` 바이트로 HZ 가중 봉평균을 **실측**했다.

---

## 0. RESULT

**5기 전원 TRUE 봉평균 첨두 76.2–77.4 GWd/tU, 노드 첨두(83.0–84.2) 대비
1.083–1.090배 낮다 — DB 대리계수 1.0886±0.0089 와 사실상 정확히 일치
(우리 비 평균 1.0878±0.0030, n=5).** 파서 정합성 게이트 (a) 5/5 통과 (실측
`max_pin_burnup`과 바이트 단위 완전 일치). 결정론 재확인: 5/5 FOM 완전
동일 (기존 purge-run과 소수점까지 일치).

**조건부 판정** (M1 미확정이므로 VOID 유지, 아래 §5 참조): 봉평균 62/68/75
사다리 — **5/5 전원 FAIL**(값이 76.2 이상이라 75도 넘는다). 봉평균 **80** —
**5/5 전원 PASS** (여유 2.6–3.8 GWd/tU). **단, 같은 5기의 노드 첨두는
82.983–84.242 로 LEU+ 80 GWd/tU 를 이미 5/5 초과 — 한도가 노드에 구속되면
이 5기는 이미 전원 FAIL 이다** (`pinbu_wave.py`의 `delivery_verdict`
필드가 기존에도 이렇게 산출해 왔음, §4). 이 노드-대-봉평균 판정 역전이
M1(한도가 어느 관측치에 구속되는지)이 열려 있는 한 실제 판정 불가능한 이유다.

---

## 1. 방법

### 1.1 재실행 (박스 199, keep_success=true)

`pinbu_wave_199.inp`를 복사해 `pinbu_wave_keep_199.inp`(§ 헤더에 변경사유
명시)를 만들고 `[master] keep_success = false` → `true` 한 줄만 바꿨다. 그 외
모든 덱 노브([produce], [verify], [design], [case])는 바이트 단위 동일 —
같은 평가를 재현하되 최종 수렴 사이클의 작업 디렉터리만 보존한다.

대상: `runs/pinbu_wave_f113pin5`와 동일한 5개 record_id (플랜:
`data/reports/pinbu_wave_f113pin5_prereg_20260820.json`, SHA256
`90E5B94F91185B6D5981B41285EBF1262177E747926EEE747F356511E40156C1`, 박스199
원본과 로컬 사본 해시 일치 확인 후 사용):

| record_id (12자) | e_core | f_r pattern rank |
|---|---:|---:|
| `2ad9de110b1d` | 5.4 | 1 |
| `6de15f03c5b6` | 5.4 | 2 |
| `5c077310d891` | 5.4 | 3 |
| `817f32c7de0c` | 5.4 | 4 |
| `e36f10d2b3ad` | 5.4 | 5 |

실행: `python pinbu_wave.py run --plan data/reports/pinbu_wave_f113pin5_prereg_20260820.json
--deck pinbu_wave_keep_199.inp --run-dir runs/pinbu_wave_keep_f113pin5`
(박스 199, house pattern — 기존 `launch_pinbu_wave_f113pin5_199.ps1`의 busy-gate
/ RAM·disk 체크를 그대로 따르되 5분 내 끝나는 짧은 웨이브라 전경(SSH 동기)
실행). 총 wall 282s, 5/5 converged.

### 1.2 결정론 재확인

| record_id | purge-run (기존) pin | keep-run (신규) pin | 일치 |
|---|---:|---:|---|
| `2ad9de110b1d` | 84.202 | 84.202 | 완전 일치 |
| `6de15f03c5b6` | 83.849 | 83.849 | 완전 일치 |
| `5c077310d891` | 84.172 | 84.172 | 완전 일치 |
| `817f32c7de0c` | 82.983 | 82.983 | 완전 일치 |
| `e36f10d2b3ad` | 84.242 | 84.242 | 완전 일치 |

`determinism_ok=True` 5/5 (f_r/cyclen/cbc_max 모두 허용오차 내). 결정론이
사전 검증대로 확인되어, keep-run의 MAS_PPI를 원래 저장된 코어의 실측치로
간주하는 전제가 성립한다.

### 1.3 산출물 회수

각 워커의 최종 수렴 사이클 디렉터리에서 `MAS_PPI.*`(11,057,820 bytes 5개
전부 동일 크기 — 같은 노심 지오메트리), `MAS_INP`, `MAS_SUM`을 로컬로
회수했다 (`MAS_HFF`/`MAS_XSL`/`MAS_RST`는 파싱에 불필요해 제외):

```
runs/pinbu_wave_keep_f113pin5/ga80/master_work/worker_00/{MAS_PPI.APRQ_22_0646.02, MAS_INP, MAS_SUM}
runs/pinbu_wave_keep_f113pin5/ga80/master_work/worker_01/{MAS_PPI.APRQ_23_0645.49, MAS_INP, MAS_SUM}
runs/pinbu_wave_keep_f113pin5/ga80/master_work/worker_02/{MAS_PPI.APRQ_23_0645.85, MAS_INP, MAS_SUM}
runs/pinbu_wave_keep_f113pin5/ga80/master_work/worker_03/{MAS_PPI.APRQ_22_0643.05, MAS_INP, MAS_SUM}
runs/pinbu_wave_keep_f113pin5/ga80/master_work/worker_04/{MAS_PPI.APRQ_22_0645.50, MAS_INP, MAS_SUM}
```

박스 199 원본은 그대로 남아 있다 (`C:\Users\USER\lpopt_work\kit_frontier\runs\pinbu_wave_keep_f113pin5`,
정리하지 않음 — 여유 디스크 42.7GB 중 55MB만 사용).

### 1.4 워커 → record_id 매핑

`MAS_SUM`의 EDIT5 최대조립체연소도로 매핑했다 (각 record의 저장된
`max_assembly_burnup`과 소수점까지 일치, 유일하게 식별 가능):

| worker | MAS_SUM `max_burnup_assembly` | `max_assembly_burnup` | → record_id |
|---|---|---:|---|
| `worker_00` | K10 | 68.824 | `2ad9de110b1d` |
| `worker_01` | L9  | 69.709 | `6de15f03c5b6` |
| `worker_02` | N13 | 69.516 | `5c077310d891` |
| `worker_03` | N13 | 69.379 | `817f32c7de0c` |
| `worker_04` | N13 | 69.595 | `e36f10d2b3ad` |

---

## 2. 파서 정합성 게이트 (a) — PASS 5/5

`lpopt/data/pinppi.py`(신규, §6)가 계산한 raw(비가중) 3-D 최댓값이
`lpopt/vendor/masterrl/burnup.py:parse_ppi_max_pin_burnup()`(기존, 이미
`max_pin_burnup`으로 store에 반영되어 있던 값)의 출력과 **완전 일치**해야
한다는 게이트다. 5기 실제 바이트로 검증:

| record_id | 기존 harvested `max_pin_burnup` | 신규 파서 raw_max (전 조립체 스캔) | 위치 | 일치 |
|---|---:|---:|---|---|
| `2ad9de110b1d` | 84.202 | 84.202 | K10/z8/i3/j3 | 예 |
| `6de15f03c5b6` | 83.849 | 83.849 | L9/z8/i14/j1 | 예 |
| `5c077310d891` | 84.172 | 84.172 | N13/z8/i3/j3 | 예 |
| `817f32c7de0c` | 82.983 | 82.983 | N13/z8/i3/j3 | 예 |
| `e36f10d2b3ad` | 84.242 | 84.242 | N13/z8/i3/j3 | 예 |

신규 파서는 기존 파서와 달리 **69개 조립체 블록 전체**를 스캔해서 이 최댓값을
찾는다(기존 파서는 SUMMARY EDIT5가 지목한 조립체 하나만 본다) — 그럼에도
정확히 같은 값·같은 위치로 수렴했다는 것은 (i) 조립체 블록 분리 정규식
재사용이 올바르고 (ii) NPIN/NZC 파싱이 올바르고 (iii) 노드 첨두가 실제로
EDIT5 최대조립체 안에 있다(다른 조립체가 이를 능가하지 않는다)는 것을 5기
모두에서 독립적으로 확인한 것이다. 게이트 통과 후에만 §3의 봉평균 값을
신뢰한다(작업지시의 명시적 조건).

---

## 3. TRUE 봉평균 (HZ 가중, 전 조립체 스캔) — 5기 결과

HZ(:) = Record 1의 NZC개 축방향 연료평면 두께(균일 15.24cm, NZC=25 —
`HEIGHT: 381.00` = 25×15.24 자기정합 확인). 각 조립체·각 핀(i,j)에 대해
`Σ(BPIN[layer,i,j]·HZ[layer]) / Σ(HZ)`를 계산해 조립체 내 최댓값을 구하고,
69개 조립체 전체에서 최댓값(=봉평균 첨두)을 취했다:

| record_id | 노드 첨두 (기존, GWd/tU) | 노드 위치 | **TRUE 봉평균 첨두** (GWd/tU) | 봉평균 위치 | 비(노드/봉평균) |
|---|---:|---|---:|---|---:|
| `2ad9de110b1d` | 84.202 | K10/z8/i3/j3 | **77.4345** | K10/i3/j3 | 1.0874 |
| `6de15f03c5b6` | 83.849 | L9/z8/i14/j1 | **77.4398** | L9/i3/j3 | 1.0828 |
| `5c077310d891` | 84.172 | N13/z8/i3/j3 | **77.2196** | N13/i3/j3 | 1.0900 |
| `817f32c7de0c` | 82.983 | N13/z8/i3/j3 | **76.2455** | N13/i3/j3 | 1.0884 |
| `e36f10d2b3ad` | 84.242 | N13/z8/i3/j3 | **77.2638** | N13/i3/j3 | 1.0903 |

관찰: 봉평균 첨두가 위치한 조립체는 5/5 노드 첨두 조립체와 **동일**(K10,
L9, N13×3) — 전 조립체 스캔이 굳이 다른 조립체를 찾아내지 않았다(물리적으로
타당). 다만 `6de15f03c5b6`은 같은 조립체 안에서도 **핀 위치가 다르다**
(노드 i14/j1 vs 봉평균 i3/j3) — 단순히 노드값을 축소한 게 아니라 축방향
형상이 다른 핀에서 진짜로 다른 첨두를 만든다는 증거(가중치 적용이 실제로
작동한다는 방증, `tests/test_pinppi.py::test_rod_average_uses_hz_weights_not_plain_mean`
와 같은 성질).

---

## 4. 비(ratio) 분포 vs DB 1.0886±0.0089 — 교차확인 (이번에는 실측)

`pinbu_rodavg_20260820.md`가 "미해결"로 남긴 교차확인을 **처음으로 실측
데이터로 수행**한다:

| 통계 | 5기 실측 node/rodavg 비 |
|---|---:|
| n | 5 |
| 평균 ± sd | **1.0878 ± 0.0030** |
| 중앙값 | 1.0884 |
| 범위 | 1.0828 – 1.0903 |

DB 인구 축 계수 1.0886±0.0089 와 비교: 평균 차이 0.0008(DB sd의 9%),
5기 전원이 DB±1sd 밴드 안에 있다. **DB 대리계수가 우리 노심에도 사실상
정확히 성립한다** — 이전까지는 검증되지 않은 외부 대리값이었지만, 이번
실측으로 (적어도 이 5기, N1N2/f113 계열에 대해) 신뢰할 수 있는 근사임이
확인됐다. 우리 sd(0.0030)가 DB sd(0.0089)보다 오히려 작은 것은 5기가 모두
같은 캠페인·같은 지오메트리·비슷한 e_core(5.4)에서 나온 근접 패턴들이라
분산이 자연히 작기 때문으로, DB(1,959행, 훨씬 넓은 설계공간)와 직접 비교시
과신하지 않아야 한다.

---

## 5. 판정표 — 조건부 (M1 미확정, VOID 유지)

### 5.1 봉평균(TRUE) 기준 사다리

| 한도 (GWd/tU, 봉평균 축) | PASS | FAIL | n |
|---:|---:|---:|---:|
| 62 | 0 | 5 | 5 |
| 68 | 0 | 5 | 5 |
| 75 | 0 | 5 | 5 |
| **80** | **5** | **0** | **5** |

75에서도 전원 FAIL인 이유: 5기 봉평균이 76.2–77.4로 75를 이미 넘는다(여유
없음, 최소 마진 0.25 GWd/tU). 80에서는 전원 PASS, 최소 마진 2.56 GWd/tU
(`817f32c7de0c`).

### 5.2 노드 기준 (대조, 기존 파이프라인 필드)

`pinbu_wave.py`의 기존 `delivery_verdict` 필드(`max_pin_burnup <= 80`)가
같은 5기에 대해 이미 산출해 놓은 값:

| record_id | 노드 첨두 | `delivery_verdict`(노드@80) |
|---|---:|---|
| `2ad9de110b1d` | 84.202 | **FAIL** |
| `6de15f03c5b6` | 83.849 | **FAIL** |
| `5c077310d891` | 84.172 | **FAIL** |
| `817f32c7de0c` | 82.983 | **FAIL** |
| `e36f10d2b3ad` | 84.242 | **FAIL** |

**5/5 노드 기준으로는 이미 FAIL, 5/5 봉평균(TRUE, 80 한도) 기준으로는
PASS.** 이 정반대 결과가 M1을 형식적 절차가 아니라 실질적 판정 분기점으로
만든다 — 어느 쪽이 맞는지에 따라 이 5기(납품 후보, `N1N2_f113_pin_reverify`
그룹)의 인허가 결론이 완전히 뒤바뀐다.

---

## 6. M1 — 여전히 사용자 차단 상태 (변경 없음)

LEU+ 80 GWd/tU 한도가 **노드**(단일 층 핀 첨두)를 구속하는지 **봉평균**(축
방향 스미어드 핀 첨두)을 구속하는지 명문화한 1차 문서가 여전히 저장소에
없다. 이번 조사는 그 물음 자체에 답하지 않았다 — 다만 §5.2 가 보여주듯,
이번에는 **답에 따라 이미 산출된 판정이 뒤집힌다**는 것을 실측으로
정량화했다는 점이 다르다. M1은 사용자/규제 확인 없이는 닫히지 않는다.

---

## 7. Store 패치

`data/store/records.parquet`에 신규 컬럼 `max_rod_avg_burnup`을 추가했다
(append-only tail, `lpopt/data/schema.py`의 `LATE_COLUMNS` 관례를 따름 —
`node_peak`, `map_cov` 다음에 3번째 tail 컬럼으로 추가, 기존 36개 고정
컬럼은 불변). 게이트 (a) 5/5 통과를 확인한 뒤에만 다음 5행에 값을 썼다:

* 백업: `data/store/records.parquet.bak_pre_rodavg_20260820` (패치 전 상태
  그대로, 최초 1회만 생성)
* 패치 전: 5행 모두 `max_rod_avg_burnup = NaN`, 다른 74,652행도 NaN
  (신규 컬럼이라 전량 null이 정상 상태)
* 패치 후: 지정된 5개 record_id만 §3의 값을 기록, 나머지 74,652행은
  그대로 NaN. 전체 행 수 74,657 불변(신규 행 없음, 기존 행 5개만 갱신)

| record_id | `max_pin_burnup`(기존) | `max_assembly_burnup`(기존) | `max_rod_avg_burnup`(신규) |
|---|---:|---:|---:|
| `2ad9de110b1d...` | 84.202 | 68.824 | 77.43448 |
| `6de15f03c5b6...` | 83.849 | 69.709 | 77.43984 |
| `5c077310d891...` | 84.172 | 69.516 | 77.21960 |
| `817f32c7de0c...` | 82.983 | 69.379 | 76.24552 |
| `e36f10d2b3ad...` | 84.242 | 69.595 | 77.26376 |

---

## 8. 파서 + 테스트 (repo 반영)

* **신규 모듈**: `lpopt/data/pinppi.py` — `parse_ppi_core_rod_average_peak()`
  (전 조립체 스캔, 코어 단위 TRUE 봉평균 첨두 + 대조용 raw 노드 최댓값을
  한 패스로 계산), `parse_ppi_assembly_rod_average()`(조립체 단위). 기존
  `lpopt/vendor/masterrl/burnup.py`의 `_PPI_BLOCK`/`_PPI_PIN_BURNUP` 정규식을
  재사용(import)해 블록 경계가 두 파서 사이에서 절대 어긋나지 않도록 했다 —
  vendor 파일 자체는 수정하지 않았다.
* **신규 테스트**: `tests/test_pinppi.py` (5 테스트, 전부 PASS) —
  실바이트 픽스처 `tests/data/mas_ppi_k10_fixture.txt`(51.9KB, 2026-08-20
  keep-run의 `worker_00/MAS_PPI.APRQ_22_0646.02`에서 K10 조립체 블록의
  FANAME 줄 + PIN 3-D BURNUP DISTRIBUTION 구간만 바이트 그대로 발췌 — 사이
  기록 2-8은 두 파서 모두 읽지 않아 생략, 발췌한 부분은 원문 그대로).
  * `test_gate_a_reproduces_node_peak` — 게이트(a) 자체를 단위테스트로 고정
    (84.202, K10/z8/i3/j3)
  * `test_rod_average_peak_on_real_bytes` — 봉평균 77.43448 GWd/tU 재현,
    node/rodavg 비가 DB 밴드 안에 있는지 회귀 체크
  * `test_rod_average_uses_hz_weights_not_plain_mean` — HZ를 실제로
    가중치로 쓰는지(단순평균으로 몰래 퇴화하지 않는지) 확인
  * `test_hz_line_rejects_wrong_count` / `test_fixture_is_one_real_assembly_block`
    — 입력 방어 및 픽스처 자체 검증
* **스키마**: `lpopt/data/schema.py` — `max_rod_avg_burnup` 필드를
  `CanonicalRecord`/`PARQUET_SCHEMA`/`LATE_COLUMNS`에 추가(append-only,
  기존 assert 가드 전부 통과 확인).
* **회귀 확인**: 스키마 tail을 문자 그대로 assert하던 기존 테스트 2건
  (`tests/test_backfill_flatness.py::test_columns_are_appended_after_the_frozen_prefix`,
  `::test_columns_round_trip_through_parquet`)을 3-컬럼 tail로 갱신, 21/21
  재통과 확인. (`tests/test_axial_head.py::test_stored_labels_regenerate_the_ao_abs_column`,
  `tests/test_featurize.py::test_core_enrichment_split_reproduces_stored_e_core`
  2건은 이번 변경과 무관한 기존 실패 — ao_abs/e_core 재계산 문제로 축방향
  맵·연료 라이브러리 데이터에 관한 것이며 `max_rod_avg_burnup`/스키마 tail과
  접점이 없다. 조사 범위 밖으로 남겨둔다.)

## 9. 다음 지렛대 (범위 밖, 기록만)

* `pinbu_wave.py`의 `cmd_run`/`cmd_patch`는 아직 `max_rod_avg_burnup`을
  자동으로 채우지 않는다 — 이번 조사는 5기에 대해 수동으로(§1, §7) 채웠다.
  다음 웨이브부터 두 관측치를 한 번에 얻으려면 `cmd_run`이
  `parse_ppi_core_rod_average_peak`도 호출하도록 배선해야 한다(파서 자체는
  이미 완성·검증됨, 배선만 남음).
  이번 지시 범위("측정 chore")는 5기 실측 + 파서/테스트 반영까지였다.
