# Slice Z S6 부트스트랩 실패 진단 — T3_T4 (2026-09-03)

**결론 먼저**: S5 라이브러리 재빌드(N 37→39)는 **무죄**. 패키지 롤백 **불필요**.
원인은 `[design].cy1_cap_efpd`를 UNSET으로 둔 것. cy1이 자연 EOC 894.09 EFPD까지
연소되어 cy02 재장전 노심이 발산 → MASTER가 NaN 루프에 빠져 3600 s 타임아웃.

---

## 1. 증거

### 1.1 잔존 작업 디렉터리
`.../bootstrap_work/T3_T4/master/bootstrap-7a75a69dbf-uggjt1qj/`

| 파일 | 크기 | mtime |
|---|---|---|
| MAS_INP | 8,736 B | 03:12:14 |
| MASTER.stdout | 317 B | 03:12:15 (이후 정지) |
| MASTER.stderr | 0 B | — |
| MAS_OUT | 5,666,095 B | **04:12:14** (= 시작 +60분, 계속 증가) |

MAS_XSL/MAS_HFF/MAS_RST 없음 → `PurgingEquilibriumRunner`가 실패 시 스테이징 입력만 정리. 정상.

**행(hang)이 아니라 발산 루프**다. stdout은 `INITIALIZE ...`에서 멈췄지만 MAS_OUT은
1시간 내내 커졌다. 프롬프트·PAUSE·파일 누락 대기 없음.

### 1.2 정지 지점 = cy02 BOC 붕산 탐색 4번째 외부반복
MAS_OUT 총 114,984행. 첫 NaN = 1546행.

```
   1   0.000 0.988489    10.56 7.90268E-02
   2   0.000 1.017280   665.86 1.22448E-02
   3   0.000 1.003516  1002.14 1.66974E-02     <- 오차가 오히려 증가
   4   0.000      NaN  1288.91         NaN     <- 발산
MGOUTER   11   20       NaN          NaN         NaN
```
이후 `MGOUTER 11 20 NaN`이 **약 57,000회** 무한 반복.
MASTER V4.00 MOD3에는 NaN 종료 가드가 없다(`err < epsflx` 비교가 NaN이면 항상 false).
수치 발산이 무한 루프로 바뀌고, 유일한 방어선이 `[master].timeout = 3600`이다.

### 1.3 cy1은 완벽히 정상
`bootstrap_work/T3_T4/cy1/`: 23.83 s, JOB FINISHED, 붕산 1819.20→10.56 ppm,
EOC **894.09 EFPD / 34.97 MWd/kgHM**, `MAS_RST.APRQ_01_0894.09` 생성.
→ 현재 39-type MAS_XSL/MAS_HFF는 **정상적으로 읽히고 정상 물리를 낸다**.

### 1.4 덱 비교 (스냅샷 vs 현행 vs 작업디렉터리)
스냅샷 `cores/T3_T4/bootstrap/MAS_INP_cy02.inp` ↔ 현행 동일 파일 차이는 **4건뿐**:

| 항목 | 스냅샷(pre-S5) | 현행(post-S5) |
|---|---|---|
| `%GEN_DIM` | `10 10 27 40 42` | `10 10 27 42 44` |
| 재시작 파일 | `MAS_RST.APRQ_01_`**`0597.70`** | `MAS_RST.APRQ_01_`**`0894.09`** |
| `%LPD_B&C` | — | `T7`, `T8` 배치 추가 |
| `%LPD_C&X` / `%LPD_HFF` | — | `38 FA_T7`, `39 FA_T8` 추가 |

dims·배치·COMP·HFF 확장은 S5 의도대로 **정합**(배치 42 = R1~R3+39종, COMP 44 = 반사체 5+39).
MAS_OUT의 COMP 에코도 1~39가 `FA_P0`…`FA_T8`로 정상 출력(G-H6 통과).
현행 템플릿 ↔ 작업디렉터리 차이는 `%LPD_SHF`뿐 — 템플릿의 `F P0 0` 자리표시자를
드라이버가 실제 셔플맵으로 채운 것. 정상.

**→ 유일한 실질적 차이는 재시작 파일의 연소 깊이: 597.70 vs 894.09 EFPD.**

---

## 2. 격리 시험 (scratch_s6diag, 199)

### (a) 잔존 덱 그대로 + 현행 라이브러리 → **재현**
`testA_asis/`: 900 s 초과, 첫 NaN이 MAS_OUT **1547행, 반복 4, 1288.91 ppm**으로
운영 실패와 **비트 단위 동일**. 결정론적 재현. (프로세스 정리 완료, 잔존 0)

### (b) cy1을 597.70 EFPD로 CAP + **현행(N=39) 라이브러리** → **정상 완료**
- `testB_cy1_a/`: 17.3 s 완료, 산출물이 정확히 `MAS_RST.APRQ_01_`**`0597.70`**
  (스냅샷 템플릿이 참조하던 바로 그 이름·크기 6,475,024 B).
- `testB_cy2_a/`: **같은 덱, 같은 셔플맵, 같은 현행 라이브러리**, 재시작 파일만 교체
  → **19.6 s 완료, NaN 0건**, `MAS_RST.APRQ_02_0639.71` 생성, cyclen 639.71 EFPD.

> 스냅샷 라이브러리(N=37) 시험은 **불필요해졌다**. 현행 라이브러리로 cy02가
> 완주했으므로 COMP 순서·HFF 레코드 수·ADF·반사체 결함 가설은 모두 배제된다.

### (c) 독립 교차 검증 — 재빌드 이전에도 같은 고장이 있었다
로컬 `data/design/package/bootstrap_work/T5_T6_f101/master/bootstrap-2f5102b10d-dqsi61cx`
(2026-08-11, **구 37-type 라이브러리** MAS_XSL 14,278,423 B):
MAS_OUT 5,853,254 B / mtime 정확히 시작 +60분, 꼬리가 `MGOUTER 11 20 NaN` 무한 반복.
→ NaN 무한 루프는 **S5 이전부터 존재하던 MASTER 고유 고장 모드**다.

---

## 3. 판정

**주원인 (P1)** — `produce_sliceZ_bootstrap_199.inp`가 `cy1_cap_efpd`를 의도적으로
UNSET으로 둔 것. 헤더의 D-4 논거("cap을 걸면 R_Fr이 프로토콜 변경을 흡수한다")는
**사실관계가 반대**다. 기존 `bases/` 재시작들은 이미 cap을 걸고 만들어졌다:

- 원칙식 `cap = 2*B1 / (241/feed + 1)`  (coredeck.py:330, config.py:920)
- T5_T6 f81 → 493.60 EFPD, f101 → 579.40 EFPD.
  cy1 노심은 feed와 무관한데 값이 다르다 → cap이 적용된 증거.
  두 값 모두 B1 = 981.0으로 식을 **소수점까지** 만족.
- T3_T4 feed 121: B1 = 894.09, 241/121+1 = 2.99174 → **597.70** =
  스냅샷 템플릿이 참조하던 값과 **정확히 일치**.

즉 **cap을 빼는 것이 곧 프로토콜 변경**이며, 동시에 이번 실패의 직접 원인이다.
uncapped cy1(894.09 EFPD, 34.97 MWd/kgHM)은 cy02에 평형 1주기 이월분보다
1.5배 깊은 이월 배치를 물려주어 BOC-2 CBC/노달 반복이 발산한다.

**부차 원인 (P2)** — MASTER에 NaN 가드가 없어 수치 발산이 무한 루프가 된다.
T5_T6_f101은 cap을 걸었는데도 cy3의 60 EFPD 지점에서 발산했다.
**cap은 필요조건이지 충분조건이 아니다.** 방어 장치가 반드시 있어야 한다.

---

## 4. 정확한 수정

1. **`produce_sliceZ_bootstrap_199.inp` `[design]`에 cap을 설정한다.**
   페어마다 B1이 다르므로 `[design].cy1_cap_efpd` 단일 스칼라로는 전 페어를 덮을 수 없다.
   페어별로 `--cy1-cap-efpd`를 주되, 값은 반드시 `2*B1/(241/feed+1)`로 계산한다.
   - T3_T4 / feed 121 → `--cy1-cap-efpd 597.70`  (**검증 완료: cy02 정상 완주**)
   - 나머지 페어는 uncapped cy1을 1회 돌려 B1을 얻은 뒤 식을 적용.
     (B1 측정용 cy1은 ~20 s이므로 비용 무시 가능.)
   - **전 페어에 일괄 적용한다. 섞지 않는다** — 이것이 기존 `bases/`와 같은 프로토콜이다.
   - 헤더의 D-4 블록은 근거가 뒤집혔으므로 함께 정정할 것.

2. **NaN 워치독을 추가한다 (P2).**
   `[master].timeout = 3600`이 유일한 방어선이라 발산 1건당 1시간을 태운다.
   `run_cycle1` / `MasterRunner`에 MAS_OUT 꼬리의 `NaN` 감시를 넣어
   즉시 `MasterRunError`로 중단시킬 것. 부수적으로 `timeout`도 600~900 s로 낮춘다
   (정상 1주기는 17~24 s).

3. **패키지 롤백: 불필요.** `lib/MAS_XSL`(15,050,121 B) / `MAS_HFF`(15,789,423 B) /
   `%GEN_DIM 10 10 27 42 44` / cores 재생성 결과 모두 정상 검증됨.
   `D:\lpopt_archive_199\pkg_snapshots\sliceZ_20260903T055605Z\`는 보관만 하고 복원하지 않는다.

4. **재실행 전 정리** — 실패한 `bases/T3_T4/`에는 아무것도 기록되지 않았다(n_cycles 0).
   `bootstrap_work/T3_T4/`와 `scratch_s6diag/`는 증거로 보존 중.
   현행 `cores/T3_T4/bootstrap/MAS_INP_cy02.inp`는 드라이버가 실행 중 제자리 수정하여
   `MAS_RST.APRQ_01_0894.09`를 참조하고 있다. 다음 실행 시 자동 갱신되나, 수동 점검 권장.

---

## 5. 게이트 판정

| 게이트 | 결과 |
|---|---|
| G-H5a (cy1 덱 dims/roster/이름) | **PASS** — cy1 23.83 s 정상 완주 |
| G-H5b (재장전 덱) | **PASS** — cap 적용 시 cy02 19.6 s 정상 완주 |
| G-H5c (수렴) | **미도달** — cy02에서 중단. cap 적용 후 재시도 필요 |
| G-H6 (T7/T8 에코) | **PASS** — MAS_OUT COMP 목록에 `38 FA_T7`, `39 FA_T8` 정상 출력 |

**슬라이스는 중단할 필요 없다.** 덱 한 줄(cap) 수정 후 S6 재개.
