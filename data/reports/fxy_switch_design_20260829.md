# F_xy 전환 설계서 — 최적화 목적함수를 F_r(FRP)에서 F_xy(FXYP)로

- 작성일: 2026-08-29
- 대상: `lpopt` (APR1400 equilibrium-cycle loading-pattern optimizer)
- 근거: 사용자 결정 2026-08-29 (구속력 있음) — **최적화 대상을 F_r → F_xy 로 전환, hard limit `max F_xy ≤ 1.65`**
- 성격: 설계서(design document). 본 문서 작성 과정에서 코드 수정은 **없음**(read-only 조사 + 실측).

---

## 0. 이 문서의 근거 규칙

- 코드에 대한 모든 주장은 해당 파일을 직접 읽어 확인했다. 파일/함수/라인은 본문에 명시한다.
- 수치는 **실측(measured)** 과 **추정(추정)** 을 구분한다. 표기가 없으면 실측이다.
- 원격 호스트는 `HOST_199`(캠페인 실행 박스), `HOST_238`(GPU 학습 서버), `HOST_198`/`HOST_181`(보조 박스)로만 부른다.
- 본 문서의 실측은 2026-08-29 시점의 `data/store/records.parquet` (74,657행)와 `HOST_199`의 `kit_frontier/runs` 트리에서 직접 수집했다.

### 0.1 핵심 결론 요약

1. **F_xy는 MAS_SUM에 없다.** EDIT3 컬럼은 `FQN FRN FQP FRP` 뿐이며 FXYP/FXYA는 `MAS_OUT`의 P2D edit에만 나온다. 따라서 label 확보에는 **MAS_OUT 보존 경로**가 필수다 (실측 확인).
2. **F_r은 F_xy의 충분한 대리변수가 아니다.** 실측 192 core(2 cell)에서 `F_r ≤ F_xy ≤ F_q`가 항상 성립하고 `F_xy/F_r = 1.0694 ± 0.0181`이지만, E1_E2/f109에서는 `F_r ≤ 1.55`를 통과한 core의 **18/52가 `F_xy ≤ 1.65`에서 탈락**한다. 즉 목적함수 전환은 실제로 답을 바꾼다.
3. **MASTER 재계산 없이 즉시 라벨링 가능한 물량 1,343건(수렴 1,272건)** — `HOST_199` 578 + `2_LP/LOW_Fr_MASTER_result/regen` 698 + 로컬 curated 188(중복 제거). 7개 셀에 걸쳐 있으며 store의 1.8%다. 추가 배치(`LOW_Fr` 13,466 case deck 매핑)로 누적 4,000~5,000까지 확장 가능(추정).
4. work dir 이름은 과제 지시서가 기술한 `<record_id16>__MAS_RST...`가 **아니다.** 실제로는 `<Pattern.digest16>__<restart_tag>`이며(`verify.py:931`, `domain.py:186`), record_id(64-hex sha256)와는 다른 해시다. 매핑은 store의 `pattern` 컬럼에서 `Pattern.digest`를 재계산해 join해야 한다 (실측으로 578/586 유일 매칭 확인).
5. 코드 주석 결함 1건 발견: `lpopt/data/flatness.py`가 `node_peak == F_xy`라고 단언한다. `node_peak`은 **BOC assembly power map의 69-slot 최대값**(= assembly radial peaking, FRA 계열)이고 MASTER의 FXYP(pin planar)와 **다른 물리량**이다. 본 전환과 함께 반드시 정정해야 한다.
6. **챔피언이 바뀐다(실측).** `T6_T4/f121` 690 core에서 F_r 기록 core(`F_r 1.4797 @ 623.6 EFPD`)의 측정 `F_xy = 1.5829`인 반면, 개체군의 F_xy 최소는 **다른 core**(`F_xy 1.5491`, `F_r 1.5018`, 621.3 EFPD)다. 목적함수 전환은 순위를 실제로 재배열한다.

---

## 1. 정의 — F_xy가 무엇이고 무엇이 아닌가

### 1.1 MASTER가 출력하는 peaking factor의 계층

MAS_OUT의 `$P2D_n` 블록(depletion step 1개당 1블록)에 다음이 한 번씩 출력된다 (`HOST_199`의 실제 MAS_OUT에서 확인):

```
$P2D_1          0.000 DAY          0.000 EFPD
          MAXIMUM PIN     PLANAR POWER (FXYP)=      1.6333  AT (P ,9 , 6, 9,16)
          MAXIMUM ASSMBLY PLANAR POWER (FXYA)=      1.4364  AT (P ,10, 5)
          LAYER  AXIAL ELEVAT (CM)  LAYER-AVERAGED POWER  MAX. POWER  MAX. PIN POWER  LAYER FXY
          ... (layer별 표, 축방향 layer 수만큼)
```

- `FXYP` 위치 튜플 `(P, 9, 6, 9, 16)` = (assembly column letter, assembly row, **axial plane index**, pin i, pin j). plane index가 들어있다는 것이 곧 "평면(planar)별로 계산된 pin 출력의 최대"라는 증거다.
- `FXYA` 위치 튜플은 `(P, 10, 5)` = (col, row, plane) — pin 인덱스가 없다.

정의를 정리하면:

| 기호 | MASTER 출력 | 물리적 정의 | 위치 |
|---|---|---|---|
| `F_q` | `FQP` | 전 노심 pin 3-D 최대 출력 (nodal pin peak) | MAS_SUM EDIT3 + MAS_OUT |
| `F_xy` | `FXYP` | **각 axial plane에서 계산한 pin planar 출력의, 전 plane·전 step 최대** | **MAS_OUT 전용** |
| `F_r` | `FRP` | pin 2-D radial peaking (축방향 적분/평균한 pin 출력의 최대) | MAS_SUM EDIT3 + MAS_OUT |
| `FXYA` | `FXYA` | assembly planar 출력의 plane 최대 (pin 미해상) | MAS_OUT 전용 |
| `FRA` | `FRA` | assembly 2-D radial peaking | MAS_OUT 전용 |
| `F_z` | `FZ` | axial peaking | MAS_OUT 전용 |
| `node_peak` | (MASTER 아님) | lpopt이 EDIT5 BOC assembly power map 69 slot에서 취한 최대값 | store 컬럼 |

### 1.2 부등식과 실측된 계수

물리적으로 `F_r ≤ F_xy ≤ F_q`가 성립한다:
- `F_r`는 축방향으로 평균한 pin 출력의 최대이므로, 특정 plane에서의 pin 출력 최대(`F_xy`)를 넘을 수 없다.
- `F_q`는 `F_xy`가 정의된 pin·plane 조합을 포함한 3-D 최대이므로 `F_q ≥ F_xy`.
- 근사적으로 `F_q ≈ F_xy · F_z` 급의 관계이며, 정확히는 `F_xy`와 `F_z`의 최대 위치가 일치하지 않아 `F_q ≤ F_xy · F_z`이다.

**실측(2026-08-29, `HOST_199`의 MAS_OUT 192개 core, 2개 cell)**:

| 셀 | n(물리적) | F_r 중앙 | F_xy 중앙 | F_xy/F_r 평균±sd | F_xy−F_r 평균 | corr(F_xy,F_r) | corr(F_xy,F_q) | corr(F_xy,node_peak) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1_E2 / f109 | 94 | 1.5330 | 1.6520 | 1.0737 ± 0.0195 | 0.1157 | 0.9431 | 0.9468 | 0.7351 |
| N1_N2 / f113 | 98 | 1.5665 | 1.6500 | 1.0653 ± 0.0157 | 0.1040 | 0.9634 | 0.9504 | 0.7986 |
| **pooled** | **192** | — | — | **1.0694 ± 0.0181** (p05 1.0351 / p95 1.0934) | — | 0.9505 | — | — |

- `F_r ≤ F_xy` : 192/192 성립.
- `F_xy ≤ F_q` : 192/192 성립. `F_q/F_xy = 1.1586 ± 0.0218` (= 실효 `F_z` 상당).
- pooled 선형 적합: `F_xy = 1.1221·F_r − 0.0831`, r = 0.9505, residual sd = 0.0293.
- cell별 적합: E1_E2 `1.1264·F_r − 0.0826` (resid sd 0.0315), N1_N2 `1.1416·F_r − 0.1209` (resid sd 0.0253).

### 1.3 왜 1.65인가 — 그리고 왜 F_r 1.55로는 대체 불가한가

`F_xy ≤ 1.65`는 상용 PWR의 planar pin peaking에 대한 licensing-style 한계선의 자리에 놓인 값이다(사용자 결정). 본 설계는 그 값을 물리적으로 재유도하지 않고 **주어진 hard limit으로 취급**한다. 다만 다음 두 실측이 이 값의 운영상 의미를 규정한다:

1. **1.65는 현재 frontier 개체군의 중앙값 근방이다.** E1_E2/f109에서 F_xy 중앙 1.6520, N1_N2/f113에서 1.6500. 즉 지금까지 `min_fr` 목적으로 만든 core의 **약 40~50%만** 새 한계를 통과한다 (E1_E2 39/94, N1_N2 49/98). 새 목적함수는 즉시 유효한 압력을 만든다.
2. **F_r 게이트로 F_xy 게이트를 대신할 수 없다.**

| 셀 | F_r≤1.55 통과 | F_xy≤1.65 통과 | 둘 다 | **F_r 통과·F_xy 탈락** | F_r 탈락·F_xy 통과 |
|---|---:|---:|---:|---:|---:|
| E1_E2/f109 (n=94) | 52 | 39 | 34 | **18** | 5 |
| N1_N2/f113 (n=98) | 41 | 49 | 41 | **0** | 8 |

E1_E2에서 `F_r ≤ 1.55` 통과 core의 **34.6%(18/52)** 가 `F_xy > 1.65`이다. 두 축의 불일치는 cell 의존적이며(N1_N2에서는 0건), 이는 `F_xy`가 `F_r`에 없는 **축방향 출력 형상(plane별 분포)** 정보를 담고 있음을 뜻한다. 따라서 F_r 대리 최적화는 원리적으로 안전하지 않다.

### 1.4 "planar"가 배제하는 것

`F_xy`는 각 plane 내부의 pin 출력 분포만 본다. 따라서 다음은 `F_xy` 값에 직접 반영되지 않는다:
- **축방향 blanket / cutback 설계**: 본 과제에서 축방향 zoning은 fuel type 정의에 고정되어 있고 loading pattern 탐색 변수가 아니다. 따라서 `F_xy`에 대한 이들의 기여는 **전 후보에 대해 상수** 취급이 타당하다 (추정: 단, 3-type graded case에서 type별 축방향 zoning이 다르면 상수 가정이 깨진다 — §5.7).
- **축방향 출력 분포 자체(F_z, AO)**: `F_xy`는 plane 내부만 보므로 축방향 tilt는 `|AO|` 게이트가 계속 담당한다.
- **plane 간 상호작용**: `F_xy`는 max over plane이므로, "몇 개의 plane이 뜨거운가"는 표현하지 못한다. 이 정보가 필요하면 `LAYER FXY` 표를 별도로 수확해야 한다(§1.5).

### 1.5 label로 무엇을 쓸 것인가 — 결정

**결정: `f_xy` = 최종 equilibrium cycle의 MAS_OUT에 나타나는 모든 depletion step의 FXYP 중 유한 최대값.**

근거:
1. **equilibrium = 최종 cycle.** `lpopt`의 label 계약은 이미 전부 최종 cycle 기준이다 (`verify.py:_maps_from_equilibrium_result`가 `result.cycles[-1].work_dir / "MAS_SUM"`을 읽는다). `PurgingEquilibriumRunner`가 중간 cycle work dir을 즉시 삭제하므로 (`verify.py:_generated_restart`), 남아 있는 MAS_OUT은 **정의상 최종 cycle의 것**이다. 별도 cycle 선택 로직이 필요 없다.
2. **전 step 최대.** 설계 한계는 주기 전체에 적용된다. MOCHA의 `parse_mas_out_max_fxyp`도 같은 정의를 쓰며 docstring에 그 이유를 명시한다("the design metric is therefore the maximum of all finite values in the cycle"). `MasterSummary.metrics()`의 `max_FRP`/`max_FQP`도 같은 규약(전 step 최대)이므로 `f_xy`만 다른 규약을 쓰면 축 간 비교가 깨진다.
3. **step 수 실측**: E1_E2/f109 사례에서 한 cycle의 MAS_OUT에 P2D 블록 25개(그 중 2개가 0 EFPD — no-Xe / eq-Xe BOC), FXYP·FXYA·FQP 각 25개. 파일 크기 1,068,472 byte.

**보조 label — 수확 가치 판정:**

| 후보 | 판정 | 근거 |
|---|---|---|
| `f_xya` (max FXYA) | **수확 권고 (nullable, report-only)** | 같은 정규식 1회 스캔으로 공짜. `FXYP/FXYA` 비(pin-to-assembly amplification)는 "pattern이 만든 assembly 수준 불균일" 대 "격자 내부 pin peaking"을 분리한다. MOCHA의 `parse_mas_out_peaking_features`가 이미 `out_fxyp_fxya_ratio_*`를 feature로 쓴다. |
| `LAYER FXY` 표 | **1차 전환에서는 미수확** | step당 layer 수만큼(수십 행) × 25 step = 레코드당 수백~수천 값. 저장 비용이 `maps_hires`(레코드당 ~14 KiB)에 준하고, 1차 목표(스칼라 gate + head)에는 불필요. 축방향 형상 학습을 하려면 이미 `maps_hires["axial"]`(EDIT6, 25 node)이 있다. **P4 이후 재검토.** |
| `f_z` (max FZ), `f_ra` (max FRA) | **선택 수확** | 같은 스캔에서 공짜. `F_q ≈ F_xy·F_z` 진단과 physics prior 적합에 유용. 단 store 컬럼 추가는 최소화가 원칙이므로 **1차에는 넣지 않고**, retro-backfill 도구의 JSON 사이드카에만 기록 (추정: 필요해지면 LATE_COLUMNS에 추가). |

---

## 2. 라벨 소스 인벤토리 (실측)

`f_xy`는 MAS_SUM에 없으므로 **MAS_OUT이 남아 있는 곳만이 label 소스**다. 세 곳을 실제로 스캔했다.

### 2.0 현재 store 상태 (기준선)

`data/store/records.parquet` (2026-08-29 실측):

| 항목 | 값 |
|---|---:|
| 총 행 | 74,657 |
| `converged = True` | 66,078 |
| `maps_key` 보유 (EDIT5 map 수확) | 45,162 |
| `node_peak` 보유 | 43,670 |
| `max_pin_burnup` 보유 | 40,813 |
| **`f_xy` 컬럼** | **없음** (39개 컬럼 목록에 부재) |

가장 큰 단일 campaign은 `0_Case:sa_2b_cache` 계열 38,218행(Dataset A). 이들의 원본은 `2_LP/0_Case/sa_2b_cache.jsonl`이며, 그 `rec.metrics`는 `max_fqn / max_frn / max_fqp / max_frp` 등 14개 키뿐 — **`fxyp` 키는 63 MB 전체에서 0회** (실측 grep). 즉 Dataset A는 원리적으로 재파싱 없이는 f_xy를 얻을 수 없다.

### 2.1 (a) `SRC/runs/**` — 로컬 lpopt 런

`MAS_OUT` 총 **2,711개 / 1,527.5 MB** (평균 0.56 MB, p50 0.23, max 5.58). `runs/` 아래 2,642개(90개 campaign dir), 밖 69개(`data/campaigns/…` 63, `data/design/package/bootstrap_work/…` 6). 상위 campaign: `produce_run_20260725_135935` 592, `transpose_pairs` 122, `cur4cb4q` 120, `fpcamp_minfr_T6T4_r1` 103 …

**결정적 발견 — 로컬 잔존물의 대부분은 최종 cycle이 아니다.**

- `harvest_maps = true`로 실행된 campaign(로컬 4개: `fpcamp_minfr_T6T4`, `_r1`, `fpcamp_minfr_triple_f125_r2`, `flat_power_104`)만이 `keep_success`를 강제해 **의도적으로 최종 cycle을 보존**한다.
- 나머지는 `keep_success=False` 런이며, `EquilibriumRunner._clean` / `_safe_rmtree`가 `OSError`를 삼키므로 **Windows에서 rmtree가 실패한 dir만 우연히 살아남은 것**이다(MAS_XSL/MAS_HFF가 link count 818/819의 하드링크). 400개 표본의 chain 위치(`MAS_INP`의 `%JOB_IDE` cycle − dir 이름의 seed cycle) 분포는 **+1(첫 cycle) 218건(54%)**, +8~+13(말기) 110건(28%).
- 즉 "MAS_OUT이 있다"는 "그 후보의 어떤 cycle"이지 "수렴한 최종 equilibrium cycle"이 아니다. **파일별 cycle 판정이 필수**다.

실측 수율(직접 파싱 + join): `fpcamp_minfr_T6T4` 67→61 파싱→46 매칭, `_r1` 103→91→91, `flat_power_104` 40→40→40, `fpcamp_minfr_triple_f125_r2` 0(잔존물 없음).

→ 로컬 curated pool = **188 record_id**(중복 제거 후), 물리적 수렴 164행. `f_xy` 평균비 `F_xy/F_r = 1.0619 ± 0.0145`, `F_xy ≤ 1.65` 통과 7/164 (이 campaign들은 F_r 중앙 1.645의 고-peaking 개체군이다).

### 2.2 (b) `2_LP` 자산

#### `LOW_Fr_MASTER_result`

| 항목 | 값 |
|---|---:|
| `MAS_OUT` 총 개수 | **21,872** |
| `manifest.csv` 레코드 | 14,186 (`qualifies=True` 14,180) |
| chain_class | `final_cycle_only` 13,466 / `restart_chain` 718 / `full_chain_cy1` 2 |
| origin | srv181 6,336 / local_5RL 2,643 / srv199 1,872 / srv198 1,463 / local_3GA 1,169 / local_eqlp_ws 703 |
| `manifest.csv`의 `max_fxyp` 컬럼 | **없음** (컬럼: origin, case_name, case_path, chain_class, n_cycles, cyc_min, cyc_max, files, size_MB, kinds, pair, feed, lpd_shf, qualifies, notes) |
| `regen/regen_manifest.csv` | **724행**, 컬럼에 **`record_id`(64-hex) 포함**; `stored_f_r/regen_f_r/…`만 있고 fxyp 없음 |
| `regen/` 체인 | 716개 완전 체인 (`cy01`→`cyNN`, 각 cycle에 `MAS_INP/MAS_SUM/MAS_OUT` + `_meta_cycle.json`) |

**`regen/`이 최고 가치 자산이다.** 매니페스트가 lpopt store의 **full 64-hex `record_id`를 직접 들고 있고**, 각 체인의 최종 cycle MAS_OUT이 그대로 있다. 직접 파싱·조인 실측:

- 724행 중 **699개 체인에서 최종 cycle MAS_OUT 파싱 성공** (18개는 최종 cycle MAS_OUT 부재, 7개는 dir 부재)
- store와 **698행 join**, 그 중 **690행이 converged·물리적**
- 전량 `T6_T4 / feed 121` — 즉 **F_r 프런티어 셀 그 자체**
- 측정: `F_r` 1.4797~1.9605 (중앙 1.5865), `F_xy` 1.5491~2.1232 (중앙 1.6826), `F_xy/F_r = 1.0620 ± 0.0201`
- **`F_xy ≤ 1.65` 통과 230/690**, `F_r ≤ 1.55` 통과 213
- **F_r 통과·F_xy 탈락 41건**, F_r 탈락·F_xy 통과 58건
- 현재 F_r 기록 보유 core (`F_r = 1.4797 @ 623.6 EFPD`)의 측정 **`F_xy = 1.5829`** — 1.65를 통과한다
- 그러나 이 개체군의 **F_xy 최소는 다른 core** (`F_xy = 1.5491`, `F_r = 1.5018`, cyclen 621.3). **목적함수를 바꾸면 챔피언도 바뀐다.**

`final_cycle_only` 13,466건에 대한 300건 파일럿(`MAS_INP`의 `%LPD_SHF` → `Pattern` → `digest` → store join):

- FXYP 파싱 + 패턴 파싱 동시 성공 **208/300**
- store 매칭 **143/300** (47.7%)
- join 후 물리적 수렴 71행

→ 전량 처리 시 **약 6,000행 store 매칭, 약 3,000행 사용 가능 라벨** (추정; 파일럿 비율 단순 외삽, record_id 중복 미보정).

#### MOCHA 측 자산 (참조만 — 재사용은 코드 로직만)

- `scripts/build_fxyp_prefix_dataset.py`: `--output` 기본값 없음(required). `root.rglob("cy02")`로 케이스를 찾고 prefix cycle (2,3,4) → terminal `max(cycles)`. 스키마 `mocha-equilibrium-surrogate-dataset-v2-fxyp`, `provenance.fxyp_source = "MAS_OUT:MAXIMUM PIN PLANAR POWER (FXYP)"`.
- 산출물은 `2_LP/artifacts/`에 존재: `fxyp_prefix_dataset{,_rich}.jsonl` 각 **2,739행**, `…_rich_plus_exact48.jsonl` 2,862행, `fxyp_exact48_rich_converged.jsonl` 123행(audit: `case_dirs_seen 1718 / usable 913`). 그 밖에 `work/low_fxyp_inventory_20260824/low_fxyp_inventory.csv` **424행**(MAS_OUT 경로 → FXYP max/min 직접 매핑), `artifacts/e_master_181_final_fxyp_20260821.json` 286값, `fxyp_model_benchmark{,_v2}.json` 등.
- **`2_LP` 전체에 parquet 0개.** 이 자산들은 전부 JSONL/CSV/JSON이며 **lpopt `record_id` 체계가 아니다**(MOCHA 자체 `sample_id`/`duplicate_hash`/`case_dir`). 따라서 **직접 backfill 소스로 쓸 수 없고**, 원본 MAS_OUT 트리 재파싱이 안전하다.
- MOCHA의 어느 `METHODOLOGY.md`에도 Fxy 서술이 없다(3개 문서 전부 `fxy` 0회). Fxy 계약은 artifact 레벨 문서에만 있으므로 **lpopt 쪽 문서화는 처음부터 새로 써야 한다**(3.10).

> **정합성 경고 — MOCHA는 F_xy 한계를 1.55로 쓴다.** `2_LP/artifacts/fxyp_surrogate_contract.md`는 `max_fxyp`를 primary target으로 선언하면서 저-Fxy slice 기준을 **1.55**로 잡고("Fxy≤1.55 표본은 전체 913개 중 1개"), 그 1.55는 원래 F_r의 한계값(`PSEUDOCODE.md`의 `fr | max_frp | 1.55`)이 옮겨 붙은 것으로 보인다. **본 과제의 결정은 `F_xy ≤ 1.65`다.** 같은 이름의 축에 두 한계가 병존하므로, MOCHA의 "저-Fxy slice가 비어 있다"는 진술을 lpopt 문맥에 인용해서는 안 된다 — 1.65 기준으로는 실측 개체군의 **33~40%**가 이미 통과한다.

### 2.3 (c) `HOST_199` 원격 kit (read-only `dir`로 표본 확인)

`C:\Users\USER\lpopt_work\kit_frontier\runs\` 아래 15개 campaign dir.

| 항목 | 값 |
|---|---:|
| `MAS_OUT` | **597** |
| `MAS_SUM` (성공 최종 cycle 표지) | 538 |
| `NONFINITE_FLUX` (물리 kill 표지) | 51 |
| 고유 `Pattern.digest` | 586 |
| MAS_OUT 1개 크기(표본) | 1,068,472 byte |
| 총 용량 (추정) | 약 620 MB |

campaign별: `fpcamp_minfr_N1N2_f113` 100, `fpcamp_minfr_E1E2_f109` 100, `fpcamp_minfr_hgd569_f109` 63, `fpcamp_minfr_triple_f125` 61, `fpcamp_minfr_hgd569_f125` 61, `fpcamp_minfr_N1N2_f113_pin` 61, `fpcamp_minfr_triple_f125_r2` 60, `fpcamp_minfr_hgd569_f125_seedctl` 60, `batchswap_enum_625_T6T4` 18, `fpcamp_mt1_e50_f113` 8, `pinbu_wave_keep_f113pin5` 5.

**로컬과 달리 `HOST_199`의 잔존물은 신뢰할 수 있는 최종 cycle이다** (실측 근거 3가지):
1. 597 파일 / 586 고유 digest = 후보당 1.019개 dir. 중간 cycle이 살아남았다면 후보당 8~13개여야 한다.
2. 표본 dir 3개의 sibling restart가 각각 `APRQ_22 → APRQ_23`, `APRQ_20 → APRQ_21`, `APRQ_22 → APRQ_23`. seed는 `APRQ_11`이므로 chain의 10~12번째 = **최종 cycle**.
3. 이 campaign들은 전부 `[verify] harvest_maps = true` 데크(`fpcamp_minfr_*_199.inp`)로 실행되었고, 그것이 `keep_success`를 강제한다.

FXYP 표본 (E1_E2/f109, 한 core): 25개 step, 최대 **1.6333** `AT (P ,9 , 6, 9,16)`.

### 2.4 결론 — 0-MASTER-cost 로 retro-label 가능한 물량

세 소스를 record_id 기준으로 합집합한 실측 결과:

| 소스 | 매핑 방식 | store 매칭 record |
|---|---|---:|
| `HOST_199 kit_frontier` | `Pattern.digest` (dir 접두 16-hex) | **578** |
| `2_LP/LOW_Fr_MASTER_result/regen` | `record_id` (매니페스트 직접) | **698** |
| `SRC/runs` (harvest_maps campaign 4개) | `Pattern.digest` | **188** |
| **합집합 (중복 제거)** | | **1,343** |
| ㄴ `converged=True` | | **1,272** |
| ㄴ 물리적(F_r<2.0)·수렴 | | **1,247** |

중복은 `regen ∩ local` 121건뿐 (199는 다른 셀·다른 후보라 겹침 0). 셀 분포: `T6_T4/f121` 714, `N1_N2/f113` 156, triple `…N20…/f125` 105, `hgd569/f125` 101, `E1_E2/f109` 94, `hgd569/f109` 38, 나머지 소수.

**즉 store의 1.8%(1,343/74,657)를 MASTER 재계산 없이 즉시 라벨링할 수 있다.** 여기에 `LOW_Fr_MASTER_result`의 `final_cycle_only` 13,466건을 deck-파싱 경로로 처리하면 **누적 약 4,000~5,000행(5~7%)** 까지 갈 수 있다(추정).

I/O 비용: `HOST_199` 620 MB + `regen` 약 700 MB + 로컬 curated 약 120 MB ≈ **1.4 GB 순차 read**, 단일 스레드 기준 수 분. `LOW_Fr` 전량은 21,872 파일이며 용량은 훨씬 크므로 P2 후반 별도 배치로 돌린다.

---

## 3. 설계

### 3.1 Parser — 어디에 둘 것인가

**결정: 새 비-vendor 모듈 `lpopt/data/fxy.py`.** MOCHA `master_sum.py`의 `_FXYP_RE` / `parse_mas_out_max_fxyp` 로직을 포팅한다(로직 복사, MOCHA 원본은 수정하지 않음).

근거:

- `lpopt/vendor/masterrl/master.py`는 vendor 스냅샷이고 `lpopt/vendor/masterrl/VENDOR_MANIFEST.json`이 sha256으로 무결성을 고정한다. `lpopt vendor-check`(`cli.py:124-175`)가 on-disk 해시와 manifest 해시를 비교해 불일치면 `integrity_failed=1`로 종료한다. 현재 `master.py`의 manifest note는 `"patched: CREATE_NO_WINDOW … Only deliberate delta from the pinned snapshot"` — **의도적 delta는 정확히 1건뿐**이라는 것이 문서화된 계약이다. FXYP 파서를 여기 넣으면 그 계약이 깨지고 vendor 리베이스마다 재적용 부담이 생긴다.
- lpopt에는 이미 "MASTER 산출물 파서는 비-vendor 모듈"이라는 선례가 둘 있다: `lpopt/data/edit5.py`(MAS_SUM EDIT5/EDIT6), `lpopt/data/pinppi.py`(MAS_PPI). `fxy.py`는 그 세 번째다.

API 초안 (`lpopt/data/fxy.py`):

```python
FXYP_RE, FXYA_RE, P2D_STEP_RE           # 컴파일된 정규식 (MOCHA와 동일 패턴)
FXY_GARBAGE_CEILING: float = 4.0        # 발산 가드 (5.4)

@dataclass(frozen=True)
class FxyPeaks:
    f_xy: float; f_xya: float | None; n_steps: int; suspect: bool

def max_fxyp(path_or_text) -> float                       # 엄격: 값 없으면 ValueError
def parse_mas_out_peaks(path_or_text) -> FxyPeaks         # 엄격
def fxy_from_work_dir(work_dir: Path) -> FxyPeaks | None  # 관용: 절대 raise 안 함
```

- **엄격/관용 이중 계약.** `max_fxyp`은 값이 없으면 `ValueError` — MOCHA docstring이 명시한 대로 "FRP로 조용히 대체되는 것"을 막는다. `fxy_from_work_dir`은 절대 raise 하지 않고 `None`을 돌려준다 — `_maps_from_equilibrium_result`가 이미 쓰는 "label harvest 실패가 wave를 죽여서는 안 된다" 계약과 동일하다.
- 정규식은 실제 출력의 가변 공백(`MAXIMUM PIN     PLANAR POWER (FXYP)=`)을 `\s+`로 흡수해야 한다. MOCHA 패턴이 이미 그렇게 되어 있고 `HOST_199` 실물에서 매칭을 확인했다. (참고: 단일 공백 리터럴로 grep 하면 0건이 나온다 — 실제로 조사 중 한 번 발생한 함정이다.)

### 3.2 Harvest 지점 — 어디서 읽을 것인가

**(A) vendor `master.py:run()`, work dir 삭제 직전(`retain_success` 분기 앞).** 장점은 모든 MASTER 호출이 대상이 되고 `_store_cache(key, metrics)`에 함께 저장할 수 있다는 것. 단점이 결정적이다: **vendor 수정 필수**(3.1의 manifest 계약 파기), `MasterMetrics`에 필드를 더하면 캐시 직렬화와 `replace()` 호출부까지 파급, 그리고 **중간 cycle의 FXYP까지 무차별 수집**되어 1.5의 "최종 cycle" 정의를 harvest 지점에서 보장할 수 없다.

**(B) `lpopt/search/verify.py`의 `HarvestingEquilibriumEvaluator.evaluate()` — 권고.**

- `_maps_from_equilibrium_result` / `_hires_from_equilibrium_result` / `_eq_provenance`와 **완전히 같은 자리**에서 `result.cycles[-1].work_dir / "MAS_OUT"`을 읽는다. 이 시점의 work dir은 **정의상 최종 equilibrium cycle**이다(`PurgingEquilibriumRunner._generated_restart`가 직전 cycle dir을 즉시 rmtree; 2.3에서 실물 확인).
- `harvest_maps=true`가 이미 `keep_success`를 강제하므로(`verify.py:754`) **MAS_OUT은 이미 그 자리에 살아 있다.** 추가 MASTER 비용 0, 추가 I/O는 레코드당 1회 스캔(1.07 MB). vendor 무수정.
- `metadata["fxy"]` additive 키로 내려보낸다 → `WaveOutcome.fxy` → `CanonicalRecord.f_xy`. 세 지점 모두 `maps`/`maps_hires`/`eq_provenance`가 이미 지나간 경로다.

**결정: (B).** 보완 둘: (1) 발산 런은 `status="error"`/`failure="non_finite_flux"`로 끝나 수확 분기를 통과하지 않으므로 `f_xy`는 `None`으로 남는다(올바른 동작, 5.2). (2) map harvest와 동일하게 **수렴 런에서만** 기록한다 — 비수렴 cycle의 FXYP는 equilibrium 값이 아니다. `harvest_fxy`는 독립 플래그(기본 true)로 두되, `harvest_maps=false`면 keep_success가 강제되지 않아 MAS_OUT이 사라지므로 config 검증에서 그 조합에 경고를 낸다.

### 3.3 Schema / store

`lpopt/data/schema.py`의 `LATE_COLUMNS` 선례를 그대로 따른다 (현재 `("node_peak", "map_cov", "max_rod_avg_burnup")`).

```python
# CanonicalRecord 끝에 append (중간 삽입 금지 — FROZEN_COLUMNS 36열 prefix 불변)
f_xy: float | None = None     # MAS_OUT FXYP, 최종 equilibrium cycle 전 step 최대
f_xya: float | None = None    # MAS_OUT FXYA (보조, report-only)

LATE_COLUMNS = ("node_peak", "map_cov", "max_rod_avg_burnup", "f_xy", "f_xya")
# PARQUET_SCHEMA 끝에 ("f_xy", pa.float64()), ("f_xya", pa.float64())
```

- `schema.py` 하단 3개 assert(dataclass/pyarrow lockstep, append-only, `len(FROZEN_COLUMNS) == 36`)를 그대로 통과한다.
- `store.ensure_schema_columns`는 `LATE_COLUMNS` 중 없는 것만 NaN으로 채우므로 **기존 74,657행 parquet과 기존 multi-PC kit이 그대로 merge 된다.** 별도 마이그레이션 불필요.
- `store._quality_rank`의 tie-break 비트(`node_peak`/`map_cov` 존재 여부)에 `f_xy`를 **반드시 추가한다.** 그렇지 않으면 라벨을 가진 행이 라벨 없는 재기록에 조용히 null로 덮인다 — `node_peak`에서 실제로 발생했던 결함과 같은 종류이며, 그 결함 때문에 이 비트가 도입되었다.

**Backfill 도구: `lpopt/tools/backfill_fxy.py`** (`backfill_flatness.py`를 형판으로).

계약 4개를 그대로 승계한다: **idempotent**(같은 값이면 미기록, 변경 없으면 파일도 안 씀) / **atomic**(`store._atomic_write` + `frame_to_table`) / **order-preserving**(`record_id` 키로 fresh read에 적용, `append=True` 금지) / **never destructive**(매핑 실패는 세고 null 유지, 추정 금지).

flatness backfill과 다른 점: 소스가 store 내부(`maps.npz`)가 아니라 **외부 MAS_OUT 트리**다. 따라서 `--root <dir>`를 복수로 받고 `--mode {record_id,digest,deck}`를 지정하며, `(mode, root)`별 카운트를 리포트에 남긴다. 부동소수 tolerance는 float16 왕복이 없으므로 `flatness`의 `2**-10`이 아니라 `rtol=1e-9`로 충분하다(MAS_OUT 텍스트는 소수 4자리 고정).

**work dir → record 매핑 3-mode (모두 실측 검증)**

| mode | 소스 | 키 | 실측 수율 |
|---|---|---|---|
| `record_id` | `2_LP/LOW_Fr_MASTER_result/regen/regen_manifest.csv` | `record_id` 컬럼(64-hex, 직접) | 724행 중 **698 join / 690 물리적 수렴** |
| `digest` | `HOST_199 kit_frontier/runs/*/…/worker_NN/<16hex>__<restart_tag>/`, `SRC/runs/…` 동형 | dir 이름 앞 16자 = `Pattern.digest`; store `pattern` 컬럼에서 재계산해 join | 199: 586 unique 중 **578 유일 매칭**(모호 0), 로컬 curated: **188** |
| `deck` | 임의의 MASTER case dir | `MAS_INP`의 `%LPD_SHF` → `geometry.to_canonical_from_shf` → `Pattern.digest` → join | `LOW_Fr` final_cycle_only 300건 pilot: 파싱 208, **매칭 143** |

`deck` 모드의 로직은 이미 `lpopt/data/extract_a.py`의 stage (d)("for each `runs/*/cases/*` dir take the final `cyNN`, parse `MAS_INP %LPD_SHF` → canonical key → match")에 존재하므로 **재구현이 아니라 재사용**이다.

> **지시서 정정.** work-dir 이름은 `<record_id16>__MAS_RST...`가 아니다. `verify.py:931`이 case dir을 `f"{entry.pattern.digest}__{restart_tag}"`로 만들고, `master.py:745-747`이 그 이름을 `safe_name`(40자 절단)으로 받아 `tempfile.mkdtemp(prefix=f"{safe_name}-{key[:10]}-")`의 접두로 쓴다. 최종 형태는
> `<digest16>__<seed restart 파일명>-<contentkey10>-<mkdtemp8>`.
> `Pattern.digest = sha256(canonical)[:16]`(`domain.py:186-187`)이고 `record_id = sha256(canonical|library_id|case_pair|deck_knobs)`(`schema.py:compute_record_id`)이므로 **서로 다른 해시**다.
> **실측**: `HOST_199`의 586개 dir 접두사 중 store `record_id[:16]`과 일치하는 것 **0건**, `Pattern.digest`와 일치하는 것 **578건**.

`digest`는 pattern-only이므로 원리적으로 library/case가 다른 두 레코드에 충돌할 수 있다. 실측 586건은 전부 유일했지만, backfill 도구는 **충돌 시 기록하지 않고 `n_ambiguous`로 센다**(추정 금지). 캠페인 dir이 셀을 고정하므로 필요하면 `(campaign, digest)` 복합 키로 좁힌다.

**주의 (agent 실측):** campaign의 `labels.jsonl`은 64-hex `record_id`를 들고 있지만 16-hex digest 필드는 **없다**. `runs/fpcamp_minfr_T6T4_r1`의 work dir 접두 3개를 그 campaign의 `labels.jsonl`에서 찾으면 0건이다. 즉 **on-disk join 경로는 존재하지 않으며 반드시 `pattern` → digest 재계산이 필요하다.** (예외: `runs/transpose_pairs/outcomes.json`은 `orig_digest`/`tp_digest` 16-hex 필드를 가진다.)

### 3.4 Model — `f_xy` head

**핵심 제약: 라벨 희소성.** 전환 직후 store 74,657행 중 `f_xy` 보유 행은 **1,343행(1.8%)**, retro-backfill을 최대로 밀어도 **약 5,000행(7% 미만)**(추정). 반면 `f_r`은 66,078행이 가지고 있다.

#### 3.4.1 배치 — 8번째 dataset target으로

`promote_max_asm_bu` 선례를 그대로 쓴다 (`dataset_torch.py:76-95`):

```python
TARGETS = ("f_r","f_q","cbc_max","cyclen","ao_abs","discharge_burnup","max_pin_burnup")
TARGETS_WITH_ASM_BU = TARGETS + ("max_assembly_burnup",)
TARGETS_WITH_FXY    = TARGETS + ("f_xy",)            # 신규
def targets_for(promote_max_asm_bu=False, promote_fxy=False) -> tuple[str, ...]
```

- **APPEND만** 한다. `cyclen == index 3`은 rank loss와 cell calibration이 키로 쓰므로 절대 이동 불가(코드 주석이 명시).
- `LPDataset._targets`의 NaN 마스킹은 **이미 완전히 generic**하다(`valid = converged and not math.isnan(fv)`). f_xy 라벨이 없는 행은 mask 0 → `train.regression_loss`의 `_masked_mean`에서 자동 제외되고 다른 target으로 계속 학습된다. **특수 처리 코드 불필요.**
- `TrainConfig.promote_fxy` 플래그 + `train.py`의 meta 기록(`"promote_fxy": bool(cfg.promote_fxy)`) — `promote_max_asm_bu`가 하는 것과 1:1. `evaluate.py`가 target inventory를 하드코드 목록이 아니라 checkpoint에서 읽는 구조도 이미 갖춰져 있다.
- 두 플래그 조합(`promote_max_asm_bu` + `promote_fxy` = 9 target)도 `targets_for`가 순서를 결정하도록 하고, `model_api._to_surrogate`는 이름 기반 scatter이므로 자동으로 안전하다.

#### 3.4.2 서빙 — 7-column surrogate 계약을 건드리지 않는다

`surrogate.TARGET_NAMES`는 7열로 동결(vendor)이고 `_TARGET_TO_SURROGATE_COL`에 `f_xy` 슬롯이 없다.

- ❌ 기존 컬럼 재사용 — 불가. 남는 컬럼이 없고, 억지로 재사용하면 해당 축의 gate가 오염된다(`model_api` 주석이 `discharge_burnup`에 대해 정확히 같은 이유로 거부한다).
- ✅ **`predict_map_flatness` 선례를 따른다.** `PosValCnnBackend.predict_fxy(patterns, case, cell) -> (mean, std)`를 새로 만들고 acquisition에 **별도 인자로 전달**한다.

`score_flat_power(prediction, peak_mean, peak_std, spec, cov_mean, cov_std, patterns)`가 정확히 그 형태다 — `node_peak`/`map_cov`는 7열 밖에 있고 별도 배열로 들어와 UCB를 만든다. `f_xy`도 동일하게 `score_min_fxy(prediction, fxy_mean, fxy_std, spec)`.

부수 효과:

- `p_feasible`(컬럼 0/1/2/4 고정)은 **당분간 f_xy 축을 갖지 못한다.** `score_min_fxy` 안에서 f_xy 게이트를 `constraint_ok`/`penalty`로 처리하고(pin BU가 col 6에서 하는 것과 같은 방식), `p_feasible`는 F_r/CBC/F_q/|AO|만 계속 담당한다.
- head가 성숙해 7열 확장이 정당화되면 vendor `TARGET_NAMES` 확장이 **별도 결정**으로 올라간다(추정: 그 시점까지 불필요).

#### 3.4.3 Calibration / conformal

- **cell calibration**: `cell_calibrate.py`는 이미 target별 별도 아티팩트 파일 패턴을 쓴다(`FR_CALIB_NAME="f_r_calibration.json"`, `CBC_CALIB_NAME`, `FLATNESS_TARGETS`). `FXY_CALIB_NAME="fxy_calibration.json"`, `FXY_CALIB_SCHEMA="cell_f_xy_affine_v1"`을 같은 형태로 추가한다. 라벨이 희소하므로 초기에는 **intercept-only(bias) 보정**만 허용하고 slope는 잠근다 — `_crossfit_choice`가 이미 `affine_margin` 미달 시 intercept-only를 고르므로 `_AFFINE_MARGIN_FXY`를 크게(예 0.02) 잡으면 자연히 그렇게 동작한다.
- **conformal**: `CONFORMAL_TARGETS`는 `(이름, 7열 인덱스)` 튜플이라 f_xy를 넣을 자리가 없고, `DEFAULT_MIN_CELL = 20`도 라벨 수 대비 빡빡하다. **1차에서는 f_xy conformal 미도입.** 3.6의 "측정 기반 gate"가 그 자리를 대신한다.

#### 3.4.4 Physics prior (권고: 도입)

실측 회귀는 `f_xy`가 이미 알려진 축에서 대부분 설명됨을 보여준다:

| prior 후보 | 적합 | r | residual sd | max abs resid |
|---|---|---:|---:|---:|
| `f_xy ~ f_r` (HOST_199 2셀, n=192) | `1.1221·f_r − 0.0831` | 0.9505 | **0.0293** | 0.096 |
| `f_xy ~ f_r` (regen T6_T4, n=690) | `1.0453·f_r + 0.0266` | 0.9357 | **0.0320** | — |
| `f_xy ~ f_q` (E1_E2) | `0.8560·f_q + 0.0159` | 0.947 | 0.0305 | 0.082 |
| `f_xy ~ node_peak` (E1_E2 / N1_N2) | `0.99·node_peak + 0.35` | 0.735 / 0.799 | **0.057~0.064** | 0.25 |

**권고: `f_xy` head는 절대값이 아니라 잔차 `f_xy − prior(f_r_pred, f_q_pred)`를 회귀한다.** `power_prior`/`physics_prior`가 각각 map / cyclen에 대해 쓰는 것과 같은 구조이며, `map_prior_residual = true` 데크 스위치가 이미 존재한다. 효과:

- f_xy 라벨이 1,300행뿐이어도, prior가 **66,078행짜리 f_r head의 정확도를 그대로 상속**한다.
- head는 "F_r이 같은데 F_xy가 왜 다른가"(= plane별 pin 출력 형상)만 학습하면 된다. 이 잔차의 스케일은 0.03 오더로, 실측 노이즈 바닥(단일시드 ρF_r sd ~0.018)보다는 크되 게이트 폭보다는 작다.
- prior 계수는 **cell별로 fit**한다: 기울기가 E1_E2 1.1264 / N1_N2 1.1416 / T6_T4 1.0453로 셀 간 실제 차이가 있다. `flat_scale.json` / `map_calibration.json`과 같은 per-cell JSON 아티팩트로 저장한다.

> **경고.** prior의 잔차 최대는 0.08~0.10 수준이다. `1.65` 게이트에서 0.09는 그대로 오판이다. **prior는 head의 출발점이지 gate가 아니다.** prior 단독으로 feasibility를 선언하는 코드 경로는 만들지 않는다.

### 3.5 Search — 새 목적함수 `min_fxy`

#### 3.5.1 구조 (`min_fr_max_cycle`의 λ 구조를 미러링)

`config._VALID_OBJECTIVES`에 `"min_fxy"`를 추가한다(현재 6개: `target_cycle, max_cycle_min_fr, min_fr_max_cycle, min_fuel_cost, fr_boundary, flat_power`).

```python
@dataclass(frozen=True)
class MinFxySpec:                  # acquisition.py, MinFrSpec 미러
    lam_fxy: float = 1000.0        # F_xy가 cyclen을 엄격히 지배
    risk_z: float = 0.25
    f_xy_limit: float = 1.65       # PRIMARY 목적 + hard gate
    f_r_limit: float = 1.55        # 제약으로 유지 (3.5.2)
    cbc_limit: float = 1600.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30
    pin_bu_limit: float = 78.0
    cyclen_lo: float | None = None # 밴드(선택), min_fuel_cost 형태
    cyclen_hi: float | None = None
    fxy_bias: float | None = None  # per-cell de-bias (map_calibration 형태)
    fxy_sigma_extra: float | None = None

def score_min_fxy(prediction, fxy_mean, fxy_std, spec) -> MinFxyScore:
    #  scalar  = cyclen_LCB - lam_fxy * F_xy_UCB
    #  penalty = sum(gated axis excess^2)   <- F_r/F_q/CBC/|AO|(7열) + pin(col 6) + F_xy(외부 배열)
    #  total   = scalar - _MAXCYCLE_CONSTRAINT_TIER * penalty
```

- `F_xy_UCB = fxy_mean + risk_z * fxy_std` — `score_flat_power`가 `peak_ucb`를 만드는 방식과 동일. `_debias` / `_inflate` 훅도 같이 붙인다.
- **`fxy_std`가 NaN(head 없음)이면 `constraint_ok = False`.** pin-BU 게이트가 쓰는 바로 그 규칙(`score_min_fr_max_cycle`의 `pin_known` 처리)이다: 예측 불가한 축을 조용히 통과시키지 않는다.
- `score_pool_min_fxy`는 `score_pool_min_fr`을 미러링하되, `make_minfxy_constraints`는 `p_feasible`/`feasibility_margin`용으로 **F_r/CBC/F_q/|AO| 4축만** 넘긴다(3.4.2).
- λ 크기: `minfr_lambda = 1000.0`이 "0.01 F_r 감소 = 10 EFPD"를 뜻했다. F_xy의 셀 내 산포는 F_r과 같은 오더(sd 0.08~0.09)이므로 **`minfxy_lambda = 1000.0`을 그대로 시작값으로 쓴다.**

#### 3.5.2 F_r의 새 역할 — **제약으로 유지 권고**

| 선택지 | 판정 |
|---|---|
| F_r 완전 제거(`fr_boundary`/`flat_power`처럼 sentinel 1e12) | **비권고** |
| `F_r ≤ 1.55` hard constraint 유지 | **권고** |

1. F_r는 여전히 독립적인 licensing 축이고, 실측에 **F_r 탈락·F_xy 통과** core가 존재한다(E1_E2 5건, N1_N2 8건, regen 58건). F_r을 빼면 이들이 feasible로 승격된다 — 즉 F_r 제거는 실제로 답을 바꾸는 완화이지 무해한 정리가 아니다.
2. `f_r`는 66,078행짜리 성숙한 head를 가지며 `p_feasible`(col 0)·`feasibility_margin`·`cell_calibrate`(`f_r_calibration.json`)·`delivery.compliance_margin`이 전부 그 위에 서 있다. 유지 비용이 사실상 0이다.
3. F_xy head가 미성숙한 초기에는 **F_r 게이트가 부분적 안전망**으로 작동한다: 실측 `F_xy/F_r` p95 = 1.0934이므로 `F_r ≤ 1.509`면 `F_xy ≤ 1.65`가 p95 수준에서 함의된다.

단, `delivery`의 랭킹 키는 F_xy 기준으로 승격되어야 한다(3.5.5).

#### 3.5.3 `flat_power`의 F_xy 게이트

`flat_power`는 F_r을 objective에서 퇴역시키고 `flatpower_fr_limit = 1.70` 안전 게이트만 남긴 모드다. 여기에 대응 게이트를 추가한다:

```toml
flatpower_fxy_limit = 1.65     # FlatPowerSpec.fxy_gate; fr_gate와 동일한 binary veto
```

`score_flat_power`의 `fr_gate_violated`와 같은 방식(TIER 1회 차감, grading 아님). `FlatPowerSpec`에 `fxy_limit / fxy_bias / fxy_sigma`를 추가하고 `fxy_gate` property를 `fr_gate`와 동형으로 만든다. **이 게이트는 flat_power를 실질적으로 바꾼다**: `node_peak`과 `f_xy`의 상관은 0.735~0.854에 불과해, 평탄도 최적해가 F_xy에서 안전하다는 보장이 전혀 없다.

#### 3.5.4 feasibility — `is_feasible`에 f_xy 축 추가

`campaign.feasibility_limits_for`에 `"f_xy"` 키를 추가하고 `FEASIBILITY_LIMIT_KEYS`를 확장한다:

```python
FEASIBILITY_LIMIT_KEYS = ("f_r","cbc_max","f_q","ao_abs","max_pin_burnup",
                          "cyclen_lo","cyclen_hi","f_xy")
```

**결측 처리 — 이 설계에서 가장 민감한 결정이다.**

현재 `is_feasible`(`campaign.py:263-315`)은 두 그룹이다:

- `cbc_max / f_q / ao_abs / f_r / cyclen` — 결측이면 **REJECT**
- `max_pin_burnup` — 결측이면 **PASS** (MASTER가 판정하며, 엄격 reject는 elite pool을 고갈시킨다)

`f_xy`는 전환 직후 store의 98.2%에서 결측이다. 따라서:

| 판정 함수 | f_xy 결측 시 | 근거 |
|---|---|---|
| `is_feasible_search()` — 후보 랭킹 / elite / best-tracking / outer weights | **PASS** | pin-BU 선례. 엄격 reject는 첫 라벨이 도착하기 전까지 feasible 집합을 0으로 만들어 탐색 자체를 굶긴다. |
| `is_deliverable()` — 납품 / 최종 판정 / delivery.json | **REJECT** | 측정되지 않은 licensing 축은 만족한다고 부를 수 없다. |

> 이 분리는 새 발명이 아니라 **`2_계산/RL_core_loading_engineer_AI_review_2026-08-29.md` §6.4(P0-03)의 권고를 이행**하는 것이다. 그 문서는 3-상태 판정표(측정 PASS / 측정 FAIL / UNKNOWN)와 `is_feasible_search()` / `is_deliverable()` 분리를 §6.4·§11-2·Phase A-4에서 세 번 요구하고, Phase A 종료 기준으로 "UNKNOWN 인도 0건"을 명시한다.
>
> **F_xy 전환은 이 분리를 더 미룰 수 없게 만든다.** 현재의 단일 `is_feasible`을 그대로 쓰면 둘 중 하나를 골라야 하는데, PASS를 고르면 "F_xy로 최적화한다"면서 F_xy를 한 번도 검사하지 않은 core를 납품 가능이라 부르게 되고, REJECT를 고르면 첫 캠페인이 아예 시작되지 않는다. 따라서 **3.5.4의 분리를 P1의 필수 산출물로 격상한다.**

과도기 대안(분리 구현이 P1에 못 들어갈 경우): `is_feasible(row, limits, *, require_measured=("f_xy",))` 키워드로 호출부가 엄격도를 선택한다. 기본은 관용(탐색), `delivery.select_delivery`와 report의 "인도 후보" 표만 엄격 호출. 이는 임시방편이며 목표 상태는 두 함수 분리다.

#### 3.5.5 Delivery / report / figures

- `delivery.compliance_margin(f_r, limit=LICENSING_FR_LIMIT=1.55)` 옆에 `compliance_margin_fxy(f_xy, limit=1.65)`를 **추가**(기존 F_r 버전 유지). `select_delivery`의 정렬 키를 F_xy margin 우선, F_r margin 차순으로 바꾼다.
- `report.py`
  - `_LIMITS`에 `"f_xy": 1.65` 추가.
  - `_FR_UNGATED_OBJECTIVES` 옆에 `_FXY_GATED_OBJECTIVES = frozenset({"min_fxy", "flat_power"})` 신설.
  - deck 없이 `lpopt report`가 도는 경로용 `_DEFAULT_FXY_LIMIT = 1.65` 상수. (deck-less 경로에서 게이트를 빠뜨려 캠페인이 거부한 행을 feasible이라 부른 결함이 pin-BU와 cyclen band에서 각각 한 번씩 있었다 — 반복하지 않는다.)
- `campaign._best_dict`에 `"f_xy"`, `"f_xya"`, `"f_xy_margin_to_limit"`, `"compliance_margin_fxy"` 키 추가. `_flat_columns`와 같은 자리에 `_fxy_columns(row)`.
- `cli.py`의 RESULT 요약문에 `min_fxy` 분기 추가(`min_fr_max_cycle` 분기 형태 그대로: feasible best / best_overall + margin). `min_fr_max_cycle` 분기의 하드코드 문자열 `"(<= 1.55)"`는 이미 존재하는 결함(한계값이 데크에서 오지 않음)이므로 새 분기에서는 `spec.f_xy_limit`을 포맷한다.
- **λ-목적 검사 의무 이관.** 등록된 규칙 "λ-목적 검사는 모든 프런티어 판독에 의무(F_r 단독 헤드라인이 2회 연속 뒤집혔다)"는 그대로 유효하되 **판정 축이 F_xy로 바뀐다.** 프런티어 판독은 `min(F_xy)` 단독이 아니라 밴드 내 `cyclen_LCB − λ_fxy·F_xy`로 읽는다. `anchor_readout.py` / `autoeng.py` 등 판독 스크립트의 목적 계산을 F_xy로 교체한다.
- figures: 그물망(mesh) 표준 양식의 **노드 테두리 등급은 봉최대연소도로 고정**(사용자 2026-08-17 지시)이다. F_xy는 노드 채움/테두리를 건드리지 말고 **별도 패널**로 낸다.

### 3.6 Fallback — f_xy head가 없는 동안 무엇을 주장할 수 있는가

P1~P3 구간에는 f_xy head가 없다. 이때 캠페인은 다음과 같이만 운영·주장할 수 있다.

**할 수 있는 것**

- MASTER가 실제로 산출한 `f_xy`로 **사후 판정**: 캠페인이 만든 모든 수렴 core에 대해 `f_xy ≤ 1.65` PASS/FAIL을 측정값으로 확정한다(harvest가 P1에 들어가므로 무비용).
- `F_r`(또는 `node_peak`) 대리 최적화로 후보를 **생성**하고, F_xy는 **측정으로만 채점**한다. `F_xy ≈ 1.05~1.14·F_r`이므로 `F_r ≤ 1.55` 탐색은 F_xy 관점에서 무작위보다 훨씬 낫다.
- "이 캠페인이 만든 N개 core 중 M개가 측정 `F_xy ≤ 1.65`를 만족한다"는 **사실 주장**.
- cell별 prior 계수 `(a, b)`의 **적합과 검증**.

**할 수 없는 것 (명시적 주장 금지)**

- "F_xy를 최적화했다" — 목적함수가 F_xy를 보지 못했다. 정확한 표현은 "F_r 대리 탐색 + F_xy 측정 판정".
- 미측정 core의 F_xy 예측값 인용. head가 없으므로 예측 자체가 없다.
- `F_r ≤ 1.55`가 `F_xy ≤ 1.65`를 함의한다는 주장 — 실측 반례가 E1_E2에서 18/52, regen T6_T4에서 41/213 있다.
- `node_peak`을 F_xy 대리로 쓴 gate 주장 — 상관 0.735~0.854, residual sd 0.057~0.064로 게이트 폭과 같은 오더다.
- MOCHA의 "Fxy≤1.55 표본 1개" 통계를 lpopt 문맥에 인용 — 한계값이 다르다(2.2 경고).

### 3.7 추가할 테스트 (파일별)

| 파일 | 내용 |
|---|---|
| `tests/test_fxy_parse.py` (신규) | `parse_mas_out_peaks`: 가변 공백 매칭, 25-step 픽스처의 max, FXYA 동시 추출, 값 없음 → `ValueError`, NaN/`****` 혼입 시 유한값만 채택, `FXY_GARBAGE_CEILING` 초과 시 `suspect=True` |
| `tests/test_verify_stub.py` | `HarvestingEquilibriumEvaluator`가 `metadata["fxy"]`를 실어 보내는지, 비수렴 결과에서는 안 실리는지, MAS_OUT 부재 시 `None`으로 조용히 넘어가는지 |
| `tests/test_verify_purge.py` | 중간 cycle purge 후에도 최종 cycle MAS_OUT이 살아 있는지(= harvest 가능) |
| `tests/test_store.py` / `tests/test_v5_schema.py` | `LATE_COLUMNS` 확장 후 구 parquet round-trip, `ensure_schema_columns` NaN 채움, `_quality_rank`가 f_xy 보유 행을 우선하는지, schema assert 3종 |
| `tests/test_backfill_fxy.py` (신규) | idempotent(2회 실행 시 2번째는 미기록) / atomic / order-preserving / 매핑 3모드 / 모호 digest 미기록(`n_ambiguous`) |
| `tests/test_acquisition.py` | `score_min_fxy` 단조성, `fxy_std=NaN` → `constraint_ok=False`, λ 지배성, TIER penalty, `score_flat_power`의 `fxy_gate` veto |
| `tests/test_report_feasibility.py` | `feasibility_limits_for("min_fxy")`가 f_xy 키를 내는지, deck-less report가 1.65를 적용하는지, 캠페인과 report의 feasible 집합 동일성 |
| `tests/test_delivery.py` | `compliance_margin_fxy`, f_xy 결측 → `is_deliverable` REJECT / `is_feasible_search` PASS |
| `tests/test_config.py` | `objective="min_fxy"` 수용, `f_xy_limit`/`flatpower_fxy_limit` 파싱, `harvest_fxy` + `harvest_maps=false` 조합 경고 |
| `tests/test_dataset_torch.py` | `targets_for(promote_fxy=True)` 순서, f_xy 결측행 mask 0, 기존 인덱스 불변 |
| `tests/test_model_api.py` | `predict_fxy` 반환 형상, head 없는 체크포인트에서 NaN 반환 |
| `tests/test_vendor_closure.py` | vendor 무수정 — 새 코드가 vendor를 import 하되 수정하지 않음 |

### 3.8 Config 키 (전량 신규, 기본값 보수적)

```toml
[acquisition]
objective = "min_fxy"          # _VALID_OBJECTIVES 확장
f_xy_limit = 1.65              # PRIMARY 목적 + hard gate     (신규)
minfxy_lambda = 1000.0         # minfr_lambda와 동형          (신규)
minfxy_pin_bu_limit = 78.0     #                              (신규)
f_r_limit = 1.55               # 유지 — 이제 순수 제약
cbc_limit = 1600.0
f_q_limit = 2.41
ao_abs_limit = 0.30
flatpower_fxy_limit = 1.65     # flat_power 모드 안전 게이트   (신규)

[verify]
harvest_maps = true            # 기존. keep_success 강제 → MAS_OUT 보존 (전제조건)
harvest_fxy  = true            # 신규. 기본 true 권고 (비용 ~0)

[model]
promote_fxy = false            # 신규. P4까지 false
fxy_prior_residual = true      # 신규. head 도입 시 default
```

> `[constraints]` 블록은 **쓰지 않는다.** 그 섹션은 SDM/MTC 사후검증 전용이며(`config.py:1061` docstring: "this section is the user knob surface — whether an axis runs and what limit it is judged against", 대상은 MTC/SDM), `f_r_limit`/`cbc_limit`/`f_q_limit`이 실제로 사는 곳은 `[acquisition]`(`config.py:463-466`)이다. 지시서의 `[constraints] f_xy_limit = 1.65`는 이 이유로 **`[acquisition] f_xy_limit`으로 정정**한다.

### 3.9 `fpcamp_*` deck diff

`fpcamp_minfr_TRIPLE_f125_r2_199.inp`를 기준으로 한 최소 diff(그 외 byte-identical 유지):

```diff
 [verify]
 package_root = "data/design/package"
 harvest_maps = true
+harvest_fxy  = true            # 최종 cycle MAS_OUT에서 FXYP/FXYA 수확

 [acquisition]
-objective = "min_fr_max_cycle"
+objective = "min_fxy"
-minfr_lambda = 1000.0
+minfxy_lambda = 1000.0
+f_xy_limit = 1.65              # PRIMARY 목적 + hard gate
 f_r_limit = 1.55               # 이제 제약 전용
 cbc_limit = 1600.0
 f_q_limit = 2.41
 ao_abs_limit = 0.30
-minfr_pin_bu_limit = 78.0
+minfxy_pin_bu_limit = 78.0
```

데크 헤더의 `[optimize][DEPRECATED]` 경고는 `min_fr_max_cycle` 전용이다. `min_fxy`는 신규 production 모드이므로 `campaign.py:436-444`의 DEPRECATED 목록에 넣지 않는다.

### 3.10 문서 갱신

| 문서 | 갱신 내용 |
|---|---|
| `data/README.md` | store layout 설명에 `f_xy` / `f_xya` 컬럼 추가 |
| `2_계산/PWR_commercial_core_loading_pattern_engineering_rules_KO.md` (규칙 사전) | F_xy 항 신설 — 정의, `F_r ≤ F_xy ≤ F_q`, 1.65 한계, 그리고 이미 있는 규칙 **R-06**("고·저반응도 연료를 교차 배치해 출력분포를 평탄화한다 / 부작용: 반응도 mismatch가 지나치면 **interface pin peak 증가**")와의 연결 — 그 interface pin peak이 정확히 F_xy가 잡는 양이다 |
| `lpopt/data/flatness.py` | **결함 정정.** `node_peak == F_xy` 단언 2곳(모듈 docstring과 `node_peak = max_i p_i  # == F_xy`)을 `node_peak == BOC assembly radial peaking (FRA 계열)`로 수정하고, MASTER FXYP(pin planar)와 다른 물리량임을 명시 |
| `lpopt/search/acquisition.py` `FlatPowerSpec` docstring | `node_peak`을 "licensing-relevant signal"이라 부르는 문장에 F_xy와의 관계(상관 0.74~0.85, 대리 불가) 추가 |
| `data/reports/` | 본 문서 + P4 사전등록(pre-registration) 문서 |

---

## 4. 실행 순서와 예산

의존성: **P1 → P2 → P3 → P4 → P5**. P2는 P1의 parser에만 의존하므로 P1 완료 즉시 병렬 착수 가능.

### P1 — parser + schema + harvest (MASTER 호출 0)

| 산출물 | 파일 |
|---|---|
| FXYP/FXYA 파서 | `lpopt/data/fxy.py` (신규) |
| harvest | `lpopt/search/verify.py` — `HarvestingEquilibriumEvaluator.evaluate` / `WaveOutcome.fxy` / `_result_to_outcome` / `outcome_to_record` |
| schema | `lpopt/data/schema.py` — `f_xy`/`f_xya` LATE_COLUMNS 추가, `lpopt/data/store.py` `_quality_rank` 비트 |
| feasibility 분리 | `lpopt/search/campaign.py` — `is_feasible_search` / `is_deliverable` (3.5.4, **P1 필수**) |
| config | `[verify] harvest_fxy`, `[acquisition] f_xy_limit` 파싱 |
| 테스트 | 3.7 표의 신규 3개 + 기존 8개 갱신 |
| 문서 정정 | `flatness.py`의 `node_peak == F_xy` 오기 |

**예산: MASTER 호출 0.** 검증은 기존 픽스처 + `HOST_199`/`regen`에서 회수한 실제 MAS_OUT 표본으로 한다.
**종료 기준:** `pytest tests/` 전량 green, `lpopt vendor-check` integrity OK(vendor 무변경 확인), 기존 74,657행 parquet round-trip.

### P2 — retro-backfill (MASTER 호출 0)

| 단계 | 소스 | 매핑 | 기대 라벨 |
|---|---|---|---:|
| P2-a | `2_LP/LOW_Fr_MASTER_result/regen` (716 체인) | `record_id` 직접 | **698** (수렴 690) |
| P2-b | `HOST_199 kit_frontier/runs` (597 MAS_OUT, 약 620 MB 회수) | `Pattern.digest` | **578** (수렴 527) |
| P2-c | `SRC/runs`의 `harvest_maps=true` campaign 4개 | `Pattern.digest` | **188** (수렴 164) |
| **소계 (중복 제거)** | | | **1,343 (수렴 1,272)** |
| P2-d (선택, 후반) | `LOW_Fr_MASTER_result` `final_cycle_only` 13,466 | `deck` (`%LPD_SHF`) | **약 6,000 매칭 / 약 3,000 사용 가능** (추정) |
| P2-e (선택) | `SRC/runs`의 비-curated 2,400여 MAS_OUT | `deck` + cycle 판정 필수 | **약 700** (추정; 말기 cycle 비율 28% 실측 기준) |

- P2-b는 `HOST_199`에서 MAS_OUT을 **로컬로 회수하지 않고** 원격에서 파싱해 `{digest: f_xy}` JSON만 회수하는 편이 낫다(620 MB 전송 회피). 이는 이미 등록된 자원 배정 원칙("199에서 연산 후 결과 CSV만 회수, 렌더링은 로컬")과 일치한다.
- P2-d/P2-e는 **cycle 판정 필터**가 반드시 붙어야 한다(2.1). 필터 없이 넣으면 중간 cycle의 FXYP가 equilibrium 라벨로 오염된다.
- 도구: `python -m lpopt.tools.backfill_fxy --root <dir> --mode <record_id|digest|deck> [--dry-run]`.

**예산: MASTER 호출 0.** I/O 약 1.4 GB(P2-a~c) + 별도 배치(P2-d).
**종료 기준:** dry-run 리포트의 `n_populated / n_ambiguous / n_unreadable` 합이 스캔 파일 수와 일치, 2회 연속 실행 시 두 번째는 `wrote=False`(idempotent).

### P3 — 첫 f_xy 라벨 production wave (`HOST_199`)

**권고 셀: `T6_T4 / feed 121`.** 근거:

1. retro-label이 압도적으로 많다(**714/1,247, 57%**). cell별 prior `(a, b)`를 **지금 적합하고 검증할 수 있는 유일한 셀**이다 (regen 690 + 로컬 curated).
2. 이 셀의 `min_fr` 캠페인은 **이미 종결**되었다(r8 이득 0.0048 < 0.005). 따라서 진행 중인 F_r 프로그램과 예산이 충돌하지 않는다.
3. **사전등록 가능한 명확한 질문**이 이미 실측으로 서 있다: 이 셀의 기존 개체군에서 F_r 기록 core(`F_r 1.4797`)의 측정 `F_xy = 1.5829`인데, 개체군 내 F_xy 최소는 **다른** core(`F_xy 1.5491`, `F_r 1.5018`)다. **`min_fxy` 캠페인이 1.5491을 깰 수 있는가**가 그대로 pre-registered hypothesis가 된다. 이런 형태의 기준선을 가진 셀은 다른 곳에 없다.

**차순위: 현 프런티어 셀 `S3_T1_S5`(= `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24`) / f125.** retro-label 105건으로 prior 적합에는 얇지만, round-2가 방금 종료되어 데크·자산·restart가 모두 준비되어 있다. 단 이 셀은 CBC가 near-binding(1597.33 vs 1600)이므로, F_xy 압력이 CBC 벽에 먼저 부딪힐 위험이 있다 — 그 자체가 사전등록할 관측 사항이다.

**예산 (셀당 1라운드, `fpcamp` 표준 데크 기준)**

| 항목 | 값 |
|---|---:|
| `budget`(검증 후보 수) | 60 (7 wave x 8 + reserve 4) |
| 후보당 equilibrium chain 길이 | 중앙 11.0 cycle (평균 11.12, 최대 15) — 실측 |
| **MASTER process call** | **약 660** (60 x 11) |
| 신규 f_xy 라벨 | 최대 60 (수렴률 실측 82~100%이므로 **약 50~58**) |
| `harvest_fxy` 추가 비용 | 0 MASTER call, 후보당 MAS_OUT 1회 스캔(약 1 MB) |

두 셀 모두 돌리면 `budget` 120, **약 1,320 MASTER process call**.

**종료 기준:** 모든 수렴 core에 측정 `f_xy`가 붙어 store에 기록됨(harvest 결측률 0), 사전등록한 F_xy 기준선 대비 결과 판독(3.6의 "할 수 있는 것"만 주장).

### P4 — f_xy head 재학습 (`HOST_238` GPU1) — 사전등록 필수

- 입력: P2 + P3 이후의 f_xy 라벨 **약 1,400~1,900행**(P2-d까지 하면 4,000~5,000행, 추정).
- 학습: `promote_fxy = true`, `fxy_prior_residual = true`(3.4.4). 챔피언 계보 규칙(비회귀 게이트)을 그대로 적용.
- **사전등록 항목** (`data/reports/fxy_head_prereg_<date>.md`):
  1. 승격 기준 — 기존 7 target 전부에 대한 **비회귀**(현 챔피언 대비) + f_xy에 대한 정직 홀드아웃 MAE/RMSE/bias/Spearman + **`F_xy ≤ 1.65` 근방 slice의 별도 보고**.
  2. **노이즈 바닥 선언**: 등록된 실측 바닥은 앙상블쌍 약 0.01(cyclen/map_cov), 단일시드 ρF_r sd 약 0.018이다. f_xy prior의 residual sd가 0.029~0.032이므로, **head가 prior 대비 개선을 주장하려면 그 이상의 마진이 필요**하다. 이 수치를 사전에 못 박는다.
  3. 셀 편중 경고: 라벨의 57%가 `T6_T4/f121`이다. **cell-holdout 평가를 필수**로 하고, 미학습 셀에서의 성능을 별도 보고한다.
  4. 실패 시 행동: 승격 실패면 head 없이 P5를 돌리지 않는다(3.6의 fallback으로 되돌아간다).
- **예산: MASTER 호출 0**, GPU 학습 1회(챔피언 재학습 실적 기준).

### P5 — `min_fxy` 캠페인

- 데크: 3.9의 diff를 P3에서 쓴 데크에 적용.
- 셀: P4가 통과한 셀(최소 `T6_T4/f121`). 프런티어 셀은 head의 셀-외 성능이 확인된 뒤.
- **예산: 셀당 `budget` 60 → 약 660 MASTER process call.**
- 종료 기준: 사전등록한 λ-목적(`cyclen_LCB − λ_fxy·F_xy`) 기준 개선 + 측정 `F_xy ≤ 1.65` 만족 core 산출 + 5개 게이트(F_r/CBC/F_q/|AO|/pin) 동시 통과.

### 총계

| 단계 | MASTER process call | 신규 f_xy 라벨 | 누적 라벨 |
|---|---:|---:|---:|
| P1 | 0 | 0 | 0 |
| P2 (a~c) | 0 | 1,343 | 1,343 |
| P2-d (선택) | 0 | 약 3,000 (추정) | 약 4,300 |
| P3 (2셀) | 약 1,320 | 약 100~116 | 약 1,450 / 4,400 |
| P4 | 0 | 0 | — |
| P5 (1셀) | 약 660 | 약 50~58 | — |
| **합계** | **약 1,980** | | |

---

## 5. 리스크

### 5.1 MAS_OUT 크기와 I/O

실측 크기: `HOST_199` 1.07 MB/파일, 로컬 평균 0.56 MB(최대 5.58), `LOW_Fr` 트리 21,872 파일. harvest 시점 비용은 후보당 1회 순차 스캔이라 무시 가능(11-cycle chain의 MASTER 실행 시간 대비 <0.1%, 추정). **위험은 backfill 배치 쪽**으로, P2-d는 수 GB~수십 GB 순차 read다 — 완화: (1) 원격에서 파싱해 `{key: f_xy}` JSON만 회수, (2) 청크 스트리밍 스캔, (3) 캠페인 단위 재개 가능 배치(`--resume`). 디스크: `HOST_199` 여유 42 GB(실측)이므로 60-call 캠페인 1회는 안전하나, 누적되면 기존과 같은 아카이빙 이관이 필요하다.

### 5.2 NONFINITE / 발산 런

`HOST_199`에 `NONFINITE_FLUX` 표지 **51개**(MAS_OUT 597 중). `_FAILED_DIR_KEEP`가 실패 dir에 MAS_OUT을 남기므로 **backfill이 이들을 무차별로 읽으면 발산 중의 FXYP가 equilibrium 라벨로 기록된다.** 완화(전부 필수): dir에 `NONFINITE_FLUX`가 있으면 건너뛰고, `MAS_SUM`이 없으면 건너뛰며(실패 dir은 trim되어 MAS_SUM이 지워진다), store의 해당 행이 `converged=False`/`valid=False`면 기록하지 않는다.

### 5.3 다중 cycle chain — 어느 cycle의 MAS_OUT인가

**이것이 가장 큰 데이터 오염 위험이다.**

- `HOST_199`: 안전. 후보당 dir 1.019개, restart가 chain 말기(`APRQ_22→23`), 전 캠페인이 `harvest_maps=true`.
- `SRC/runs`: **위험.** 2,711개 중 curated는 4개 캠페인뿐, 나머지는 Windows rmtree 실패로 우연히 남은 것이다. 400건 표본에서 **54%가 chain의 첫 cycle**(+1), 말기(+8~+13)는 28%.
- `LOW_Fr_MASTER_result`: `regen/`(716 체인)은 cycle이 `cyNN` dir로 명시되어 안전. `final_cycle_only` 13,466건은 이름이 그렇게 주장할 뿐이므로 **표본 검증 후에만** 신뢰한다.
- 판정 수단(우선순위): (1) `cyNN` dir 이름, (2) `MAS_INP`의 `%JOB_IDE APRQ <n>` vs dir 이름의 seed cycle, (3) sibling `MAS_RST.APRQ_<n>_*` 중 큰 쪽, (4) 마지막 `%EDT_OPT`의 `ippi=1`(가장 확실하나 표본 250건 중 2건뿐 — 이 신호만으로 필터하면 거의 다 버린다).
- **설계 결정: backfill은 cycle 판정을 통과하지 못한 파일을 기록하지 않고 `n_cycle_unverified`로 센다**(추정 금지).

### 5.4 발산 런의 FXYP garbage

실측: E1_E2/f109의 100개 중 6개가 `F_r > 2.0`이며 최대 `F_xy = 5.1656 / F_r = 3.5066 / F_q = 7.2756`. 수렴 판정을 받았음에도 물리적으로 무의미하다. 완화: (1) `FXY_GARBAGE_CEILING = 4.0` 초과 시 `suspect=True`로 기록하지 않고 카운트, (2) **기록 시점에 `f_r <= f_xy <= f_q` 부등식 검사**(실측 882/882 성립; 위반은 파서 결함 또는 오염), (3) 학습 시 기존 `converged & valid`에 더해 두 검사를 통과한 행만 f_xy mask=1.

### 5.5 head의 라벨 희소성

1,343행(1.8%)은 f_r의 66,078행 대비 49배 적고 셀 편중도 심하다(`T6_T4/f121` 57%). 완화: prior residual 회귀(3.4.4)로 f_r head의 정확도를 상속, 나머지 7 target은 전량 라벨로 계속 학습(NaN mask), P4 사전등록에 cell-holdout 필수화, conformal은 셀당 라벨 20개를 넘길 때까지 보류. **잔여 위험(완화 불가)**: 미학습 셀에서의 F_xy 외삽 — pin-burnup head가 신규 분지에서 −5.93 GWd/tU 과소예측한 전례가 정확히 이 실패 모드다. 그래서 **납품 판정은 반드시 측정값으로**(3.5.4의 `is_deliverable`).

### 5.6 train/serve parity

(1) 학습은 8 target(`TARGETS_WITH_FXY`)인데 서빙은 7열 surrogate + 별도 `predict_fxy` 경로다 — 두 경로가 같은 head를 읽는지 테스트로 고정한다. (2) f_xy head 없는 구 체크포인트를 새 acquisition이 로드하면 `fxy_std=NaN` → `constraint_ok=False`로 **모든 후보가 infeasible**이 된다(의도된 안전 동작이지만 캠페인이 조용히 굶는다). 완화: `CampaignDriver.__init__`에서 `objective=="min_fxy"`인데 head가 없으면 **즉시 hard error** — 리뷰 §6.1의 "production 모드 fallback hard fail" 권고와 같다. (3) `cell_calibrate`의 f_xy 보정이 pooled fallback으로 떨어질 때의 무성 부정확 — 기존 `f_r`/`cbc`와 같은 위험이며 같은 방식(source 필드 로깅)으로 관측한다.

### 5.7 vendor manifest / 계약

설계상 vendor 파일은 **건드리지 않는다**(3.1, 3.2-B). `lpopt vendor-check` integrity는 불변이어야 하며 이를 사전 점검에 넣는다. 특히 `train.py:1206-1208`이 `VENDOR_MANIFEST.json`의 sha256을 체크포인트 meta에 기록하므로, vendor를 건드리면 **기존 모든 체크포인트의 vendor 서명이 달라져 비교가 깨진다** — (A)안을 기각한 실질적 이유이기도 하다.

### 5.8 그 밖

- **한계값 혼선**: 같은 조직 안에서 MOCHA는 F_xy 한계를 1.55로, lpopt는 1.65로 쓰게 된다(2.2). 두 프로젝트의 산출물을 교차 인용할 때 반드시 한계값을 명시한다.
- **`node_peak`의 이름 충돌**: 코드가 `node_peak == F_xy`라고 적어 두었기 때문에, 정정 전에는 리뷰어가 "F_xy는 이미 있다"고 오독할 위험이 크다. 3.10의 정정은 선택이 아니라 **P1 필수 항목**이다.
- **3-type graded case의 축방향 상수 가정**: 1.4에서 축방향 zoning을 후보 간 상수로 취급했으나, type별 zoning이 다른 graded triple(`S3_T1_S5`)에서는 성립하지 않을 수 있다(추정). 해당 셀의 F_xy 판독에는 이 caveat를 붙인다.
- **`F_xy ≤ 1.65`의 여유 축소**: 실측 개체군의 33~40%만 통과하므로, F_r 시절보다 feasible 집합이 좁아진다. `min_fuel_cost` / `flat_power` 등 다른 목적함수에도 이 게이트가 들어가면(3.5.3) 그 캠페인들의 feasible 수율이 함께 떨어진다 — 예산 계획에 반영해야 한다.

---

## 부록. 미해결 / 사용자 확인 필요

1. **`F_xy ≤ 1.65`의 출처 문서.** 본 설계는 사용자 결정으로 받았다. licensing 근거 문서(DCD/기술기준 조항)를 규칙 사전에 명시할 수 있으면 `f_r_limit 1.55` / `f_q_limit 2.41`과 같은 수준으로 근거를 붙일 수 있다.
2. **F_q 한계 2.41과의 정합.** 실측 `F_q/F_xy = 1.1586 ± 0.0218`이므로 `F_xy = 1.65`는 `F_q ≈ 1.91`을 함의한다 — 2.41 대비 여유가 크다. 즉 **F_xy 1.65가 F_q 2.41보다 훨씬 빡빡한 게이트**다. 이것이 의도된 것인지 확인이 필요하다.
3. **`min_fuel_cost` / `fr_boundary` / `target_cycle`에도 F_xy 게이트를 넣을 것인가.** 본 설계는 `min_fxy`(신규)와 `flat_power`(안전 게이트)에만 넣었다. 나머지 모드는 F_xy를 report-only로 둔다 — 확대 여부는 별도 결정.
4. **P2-d(13,466건 deck 매핑) 실행 승인.** 수 시간~수십 시간 I/O 배치이며, 셀 커버리지를 크게 넓히지만 cycle 판정 필터의 신뢰도에 의존한다.
