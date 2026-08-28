# 핀 연소도 봉평균 재파싱 — M2 종결 시도 (2026-08-20)

**성격**: `pinbu_audit_20260820.md` §M2 후속 조치. 읽기 전용 조사 + 파생 산출물
생성. MASTER 는 돌리지 않았다. 저장소의 다른 어떤 파일도 수정하지 않았다.

---

## 0. RESULT

**M2 는 재파싱으로 종결되지 않는다. MAS_PPI 는 49기 전원에서 0/49 생존.**
`runs/pinbu_wave*` 두 웨이브 모두 `[master] keep_success = false` 로 실행되었고,
박스 199 를 직접 열어 확인한 결과 `MAS_PPI*` 파일이 단 하나도 없다
(`master_work\worker_NN` 전부 빈 디렉터리). §M2 의 "`keep_success=True` 로
보존되어 있으면 재파싱만으로 산출 가능" 이라는 조건부 전제가 **거짓으로
확인**되었다. HZ 가중 봉평균은 **MASTER 를 재실행하지 않는 한 만들 수 없다.**

---

## 1. 확인 방법

### 1.1 코드 추적

`pinbu_wave_199.inp:51` — `keep_success = false`. 이 한 덱이 44-웨이브
(`run-dir runs/pinbu_wave`) 와 5기 f113-PIN 웨이브(`run-dir
runs/pinbu_wave_f113pin5`, `launch_pinbu_wave_f113pin5_199.ps1:22-25` 에서
동일 덱 해시를 게이트) **양쪽 모두**에 쓰였다.

`lpopt/curriculum.py:1069-1075` (`make_pin_burnup_verifier`):
```python
harvest = bool(getattr(cfg.verify, "harvest_maps", False))   # 이 덱: False
keep_success = bool(cfg.master.keep_success) or harvest      # False or False = False
```

`lpopt/search/verify.py:366-390` (`PurgingEquilibriumRunner` 독스트링, 확정 근거):
> "the FINAL equilibrium cycle's dir ... is left untouched, so the vendor's own
> `keep_success` decides its fate: kept ... when `True`, **cleaned by the vendor
> at chain end when `False`**."

`lpopt/vendor/masterrl/equilibrium.py:467-473` 은 체인 도중 매 사이클을
`keep_success=True` 로 개별 호출해 유지하지만(연쇄를 위해 restart 가 필요하므로),
`equilibrium.py:518-521` 이 체인 **종료 시** `self.keep_success`(여기서는 `False`)
가 아니면 `successful_dirs` 전부를 `_clean()` 으로 rmtree 한다 — MAS_PPI 를 담은
**최종 수렴 사이클 디렉터리도 예외 없이 포함**된다.

즉 두 웨이브 모두 설계상 "성공(수렴)한 체인은 MAS_PPI 를 남기지 않는다."
`master.py:818-850` 이 MASTER 종료 직후 그 자리에서 `parse_ppi_max_pin_burnup()`
로 3-D 배열을 **즉시 스칼라(최댓값)만 추출**해 `metrics` 에 태우기 때문에,
로컬 jsonl 의 `measured.max_pin_burnup` 값 자체는 정당하다 — 다만 그 추출에 쓰인
원본 3-D `BPIN` 배열은 이 파이프라인 어디에도 영속화되지 않는다.

### 1.2 박스 199 직접 확인 (SSH, `USER@HOST_199`)

```
Get-ChildItem runs\pinbu_wave,runs\pinbu_wave_f113pin5 -Recurse -File -Filter MAS_PPI*
  → count = 0
```

`runs\pinbu_wave\ga80\master_work\worker_00..11` 및
`runs\pinbu_wave_f113pin5\ga80\master_work\worker_00..04` — **전부 빈 디렉터리**
(임시 작업폴더 잔존물 없음, `_clean()` 이 정상 완주했다는 뜻). 남은 것은
`produce_cases\<pair_feed>\*__MAS_RST.*` (체인 시작 restart 캐시)뿐이며 이는
PPI 와 무관하다. 실패(미수렴) 체인도 없다 — 두 jsonl 모두
`status == "converged"` 49/49.

### 1.3 결론

**49/49 이 순수 재파싱으로 복구 불가.** 부분 생존조차 없다 — "어떤 사슬이
purge 되었는지 목록"을 만들 필요가 없을 만큼 전량 purge 다. M2 를 실제로
닫으려면 최소 1기라도 `keep_success=true` (또는 `[verify] harvest_maps=true` 로
강제 유도)로 **다시 MASTER 를 돌려야** 하며, 이는 이번 작업 지시("MASTER
재실행 불필요/금지")의 범위를 벗어난다.

---

## 2. 산출물

`data/reports/pinbu_rodavg_20260820.csv` — 49기 전원 (44-웨이브 44 + f113-PIN
5). 컬럼:

| 컬럼 | 내용 |
|---|---|
| `record_id`, `group`, `role`, `case_pair`, `feed`, `campaign` | 기존 식별자 |
| `measured_node_peak_GWd` | **실측** (기존값, `pinbu_wave*_results.jsonl` 그대로) |
| `measured_assembly_avg_GWd` | 실측 `max_assembly_burnup` (참고) |
| `node_over_assembly_ratio` | 실측 대 실측의 참고비 (아래 §3) |
| `computed_rodavg_max_GWd` | **공란** — MAS_PPI 부재로 계산 불가 |
| `rodavg_source` | `MAS_PPI_PURGED_0_OF_49_SURVIVE` |
| `measured_node_over_rodavg_ratio` | **공란** — 봉평균이 측정된 적이 없어 비를 만들 분모가 없음 |
| `db_ratio_estimate_rodavg_GWd` | `node / 1.0886` — DB 인구축 계수를 그대로 적용한 **추정치** (감사 §2.2–2.3 과 동일 방법, 49기 전체로 확장 재계산, 새 측정 아님) |
| `verdict_80_rodavg_conditional` | 전원 `VOID (M1 undefined AND M2 unmeasurable)` |

---

## 3. 판정표 (조건부 — 새 측정 아님, 감사 §2 방법의 49기 전체 재계산)

DB 축방향계수 1.0886 을 그대로 적용한 **추정** 사다리 (한도 대비 est-PASS):

| 한도 (GWd/tU, 봉평균 축) | est-PASS | est-FAIL | n |
|---:|---:|---:|---:|
| 62 | 0 | 49 | 49 |
| 68 | 0 | 49 | 49 |
| 75 | 15 | 34 | 49 |
| **80** | **39** | **10** | **49** |

납품(delivery) 25기 (44-웨이브 20 + f113-PIN 5) 만 분리:

| 그룹 | n | est-PASS@80 |
|---|---:|---:|
| `N1N2_f113` (44-웨이브 납품) | 5 | 5/5 |
| `E1E2_f109` (44-웨이브 납품) | 5 | 5/5 |
| `HGD569_f125_2type` | 5 | 5/5 |
| `HGD569_f125_3type` | 5 | 5/5 |
| `N1N2_f113_pin_reverify` (핀게이트 5기) | 5 | 5/5 |

25/25 est-PASS — 감사 §2.4 의 "B-추정 기준 25 PASS" 와 정확히 일치 (교차검증
성공, 새 정보 아님).

**단, 이 표의 모든 셀은 판정이 아니라 추정이다.** 실제 판정(PASS/FAIL)은
다음 두 조건이 모두 확정되어야 한다:
1. **M1** — LEU+ 80 이 봉평균을 구속한다는 1차 문서 (여전히 미확보, §5).
2. **M2** — 우리 코어의 실측 봉평균 첨두 (본 조사로 **재파싱으로는 확정 불가로
   확정**; 재측정 필요).

두 조건 중 하나라도 미충족이면 판정은 **VOID** 다. 현재 둘 다 미충족이므로
**49기 전원, 62/68/75/80 사다리 전 칸이 VOID** 다. 위 표의 숫자는 "만약 두
조건이 감사의 가정대로 성립한다면"의 시나리오값일 뿐이다.

---

## 4. 비 분포 대 DB 1.0886 ± 0.0089 — 교차확인 결과

**직접 비교는 수행할 수 없다.** `measured_node_over_rodavg_ratio` 를 만들려면
같은 코어에서 실측 node 와 실측 rodavg 가 둘 다 있어야 하는데, 49기 전원
rodavg 실측치가 0개다 (§1.3). 따라서 "우리 실측 비 분포 vs DB 1.0886±0.0089"
교차확인은 **이번 조사로도 미해결로 남는다** — 감사 원문이 이미 명시한 한계
(`pinbu_audit_20260820.md` §1.5, §6 M2)와 동일 상태다.

대신 로컬에서 만들 수 있는 유일한 실측 비 — **node/assembly** (봉평균이 아닌
집합체평균 기준) — 를 49기 전체로 갱신했다:

| 통계 | 44-웨이브+f113핀게이트 49기, `node/assembly` |
|---|---:|
| n | 49 |
| 중앙값 | 1.1545 |
| 평균 ± sd | 1.1669 ± 0.0305 |

감사 §1.4 가 실측 1,959행(P-데이터셋)으로 낸 중앙값 1.1695 와 비슷한 대역이며
(이번 49기는 그 부분집합), **DB 의 node/rodavg 1.0886 과는 다른 축(분모가
rodavg 아닌 assembly)이라 직접 비교 대상이 아니다.** 감사 §1.4 의 분해
(1.1695/1.0886=1.074, 반경첨두로 물리적 타당) 논리는 그대로 유효하지만, 그
논리 자체가 "우리 값=node" 라는 **계층** 판정이지 rodavg 수치를 낸 것이 아니다.

**결론**: DB 비 1.0886±0.0089 는 여전히 **외부, 미검증 대리(proxy)** 로만
남는다. 우리 코어 자신의 축방향계수는 이번 조사 후에도 **한 번도 측정된 적
없다.**

---

## 5. M1 — 여전히 사용자 차단 상태

`pinbu_audit_20260820.md` §6 M1 은 변경되지 않았다: LEU+ 80 GWd/tU 가
봉평균(rod-average)을 구속하는지 노드(node)를 구속하는지 명문화한 1차
문서가 저장소·`참고자료`·회의자료 어디에도 없다. 이번 조사는 그 물음에
어떤 새로운 근거도 더하지 않았다 — **M1 은 사용자/규제 확인 없이는 닫히지
않는다.**

---

## 6. M2 를 실제로 닫으려면

1. `pinbu_wave_199.inp` (또는 그 파생 덱) 에서 `[master] keep_success = true`
   로 최소 1기(권장: 대표 코어 몇 기)를 **재측정**한다 — 이번 지시 범위 밖.
2. 생존한 `MAS_PPI.*` 를 이 문서 §1.1 이 정리한 헤더 스펙
   (`Section 2 Record 1`: `FANAME EXTX EXTY NZF HZ(:)`, `MASTER4.0_UM_rev01.txt`
   :12884-12902, 12884-12922) 대로 파싱해 `HZ(:)` 를 얻는다. **주의**: 이 헤더
   포맷은 이번 조사에서도 실제 MAS_PPI 원본으로 검증된 적이 없다(저장소
   전체에 원본 사본이 없음, 감사 §1.3 도 동일하게 명시) — 재측정 후 실제
   바이트를 보고 `_PPI_BLOCK` 정규식이 HZ 리스트를 올바르게 파싱하는지
   **먼저 확인**해야 한다. 그전에 축약 코드를 작성하는 것은 미검증 포맷에
   대한 추측이 되므로 이번 조사에서는 코드를 추가하지 않았다.
3. `BPIN(:,:,:)` 를 `HZ` 가중 평균해 `(i,j)` 최대를 취하면 봉평균 첨두가
   나온다 — `burnup.py:parse_ppi_max_pin_burnup()` 과 동일한 블록 추출 로직
   (`_PPI_BLOCK`, `_PPI_PIN_BURNUP`) 을 재사용하되 max 대신 가중평균 축약을
   추가하는 새 함수로 (기존 컬럼 불변, 감사 §7.3 권고 그대로).
