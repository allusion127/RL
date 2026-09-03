# 집합체 On-Demand 설계 R1 구현 보고 (2026-09-03)

거버닝 문서: `assembly_on_demand_tasks_20260903.md` (§0–§9), `assembly_slice_Z_prereg_20260903_DRAFT.md`, `assembly_on_demand_design_v2_20260903.md`.
4개 트랙(T-LAT / T-SCREEN / T-PKG / T-RUN) + 후속 fixup 라운드 통합 결과.

---

## 1. 무결성 검증 (local ↔ HOST_238 sha256)

변경된 전 파일 **21개 byte-identical 확인**. 불일치 0건.

| 파일 | sha256 (앞 16자) |
|---|---|
| lpopt/design/lattice.py | dba1b7aef6671a9b |
| lpopt/design/spec.py | 4f00d6079ad622a8 |
| lpopt/design/compliance.py | 6ffd976699136191 |
| lpopt/design/package.py | b6f47d91c2b93348 |
| lpopt/design/screen.py | 6365a8509e1dd07a |
| lpopt/design/opscreen_chain.py | af2003a697c22cb0 |
| lpopt/design/need.py | 83719dbd09facb5c |
| lpopt/design/library.py | 0d09fd8bac01d44f |
| lpopt/design/fuel_types.py | 70ca7b32bb7cd6de |
| lpopt/design/hgc_gates.py | 9fb4a8890815d1b1 |
| lpopt/search/verify.py | c26a4d8472ebc7e9 |
| lpopt/search/sdm_mtc.py | 1b9fea4689c838ea |
| lpopt/cli.py | 0b53ffcdf8f74173 |
| lpopt/config.py | 985b41943286aa91 |
| tests/test_design_lattice_author.py | db42c333b35b9529 |
| tests/test_design_compliance.py | 4549e97a0917444b |
| tests/test_design_package_sum.py | 76eeee97402b2fb9 |
| tests/test_design_screen.py | a94478113b70a70c |
| tests/test_opscreen_chain.py | b7d4e32f98aac45b |
| tests/test_design_package_rebuild.py | 7383d8f3a05af716 |
| tests/test_design_library_gates.py | 33b4dd3d37c46a6e |
| tests/test_wave_verifier_resolver_guard.py | 198bb9d9fcd0ab3b |
| tests/test_decart_runner.py | 47e88bde93e31605 |
| tests/test_hgc_gates.py | 65c058ed24eab9d5 |
| tests/test_sdm_mtc_topk.py | d2e3376380c60804 |
| tests/fixtures/core_template_T5_T6_bootstrap_cy02.inp | 45f1eca75a848512 |

명명 정정 2건 (트랙 보고서와 실제 트리 불일치):
- T-RUN이 `lpopt/design/decart_queue.py` / `tests/test_decart_queue.py`로 보고한 것은 실제로는 **`tests/test_decart_runner.py`** (신규 모듈 없음, `lpopt/search/sdm_mtc.py`·`lpopt/cli.py`·`lpopt/config.py` 수정으로 구현됨).
- T-RUN이 `scripts/decart_stage_181.ps1`로 보고한 스크립트는 실제로는 리포 루트의 **`stage_slice_Z_181.ps1`** (로컬 전용, 238 미전송, 미실행 — 의도대로).

---

## 2. 통합 테스트 (HOST_238)

```
ssh -p 8022 USER@HOST_238 "cd ~/lpopt_ws/src && ../venv/bin/python -m pytest \
  tests/test_design_*.py tests/test_opscreen_chain.py tests/test_decart_runner.py \
  tests/test_hgc_gates.py tests/test_sdm_mtc_topk.py tests/test_wave_verifier_resolver_guard.py \
  tests/test_assets*.py tests/test_verify*.py tests/test_sdm_mtc*.py tests/test_fuel_types.py \
  tests/test_deck_alias_guard.py -q -p no:cacheprovider"
```

**결과: 394 passed, 7 failed, 11 skipped, 1 warning (4.94 s).**

실패 7건 — **전부 `tests/test_fuel_types.py`, 전부 베이스라인 환경 결손** (`/tmp/pytest_full2.log`의 17건 중 이 서브셋에 해당하는 몫):

| 테스트 | 원인 |
|---|---|
| test_mass_parser_b1_range | AssertionError — `0_APR1400/260624/FA` 부재 |
| test_mass_parser_feeds_table | KeyError `no fuel row for (library_id='260624', type_id='B1')` |
| test_library_type_collision | 동상 |
| test_5851_alias_rows | 동상 |
| test_pair_e_core_via_library | 동상 |
| test_build_persists_parquet_and_roundtrips | 동상 |
| test_kinf_columns_nan_when_absent_finite_when_harvested | 동상 |

근거: 238에 `~/lpopt_ws/0_APR1400`, `~/lpopt_ws/3_GA_Surrogate` 트리 자체가 없음(`No such file or directory`). 또한 `tests/test_fuel_types.py`는 4개 트랙이 건드린 모듈(`lpopt.design.*`, `lpopt.search.verify`, `sdm_mtc`, `hgc_gates`)을 **하나도 import 하지 않음**. 회귀 아님.

경고 1건: `test_design_package_rebuild.py` tarfile CVE-2007-4559 DeprecationWarning (Python 3.11 표준 동작, 무해).

---

## 3. 과제별 현황

| # | 과제 | 트랙 | 파일 | 테스트 | 상태 |
|---|---|---|---|---|---|
| 1 | 격자 저작 (`author_lattice`, 옥탄트→전체, gd 위치 정규화) | T-LAT | `design/lattice.py`, `design/spec.py` | test_design_lattice_author.py (34→) | 완료. `FuelDesign.gd_positions` 필드 승격은 **미완**(키 5-튜플 유지, 37종 shipped id 보존 목적) |
| 2 | 준수성 검사 | T-LAT | `design/compliance.py` | test_design_compliance.py (9) | 완료. DESIGN_GRID ratio {0.85, 0.92} vs ZONE_RATIO_TARGET 0.85±0.03 **모순은 docstring에 기록만 하고 미수정**(과제 지시대로) |
| 3 | 설계공간 열거 / 스크리닝 | T-SCREEN | `design/screen.py` | test_design_screen.py | 완료. `enumerate` → count 22962, n_u_high 11, enrichment_pairs 43, gd_wt 3, layout_pairs 89, zoning 2, layouts_by_n_gd {12:18,16:24,20:28,24:19} |
| 4 | OPSCREEN σ-chain 역예측 | T-SCREEN | `design/opscreen_chain.py`, 보고서 `assembly_sigma_chain_retrodiction_20260903.md` | test_opscreen_chain.py | 완료 (분석 스크립트는 238 `~/lpopt_ws/scratch/opscreen/sigma_chain.py`, 산출물 sha256 `5ea1bb46…dd0aa4ae`, 재실행 byte-identical) |
| 4a | 대리모델(surrogate) 트리 238 스테이징 | T-SCREEN | — | — | **미완 — 아래 §4 참조** |
| 5 | need 신호 골격 | T-SCREEN | `design/need.py` | (스켈레톤) | 부분. `need_signal()`은 NotImplementedError; F1 인터록·σ 사다리만 구현되고 측정 σ가 이를 거부 |
| 10 | run_batch 파라미터화 (template_root / preflight / 타임아웃 / 병렬도) | T-RUN | `search/sdm_mtc.py`, `cli.py`, `config.py` | test_decart_runner.py, test_sdm_mtc_topk.py (23) | 완료. **T-RUN이 open item으로 남긴 config.py:905-913 씨앗은 해소됨** — 238 현재값 `max_parallel: int = 2`, `decart_timeout: int = 7200` (등록 레시피와 일치). 슬라이스 Z 덱의 명시적 override 불필요 |
| 10b | HOST_181 스테이징 스크립트 | T-RUN | `stage_slice_Z_181.ps1` (로컬) | PowerShell AST parse-only PASS | 완료(작성·구문검사만). 미실행, 238 미전송, 181 미스테이징 |
| 11 | HGC 게이트 (G-H2~G-H5) | T-RUN | `design/hgc_gates.py` | test_hgc_gates.py (22) | 완료. G-H4 임계값은 T3–T6 홀드아웃 실측 최대치 기반 — 실제 HGC 산출물과의 첫 대면은 미실시 |
| 12 | 패키지 스냅샷 | T-PKG | `design/package.py` | test_design_package_rebuild.py (28) | 완료. 실제 `data/design/package` 대상 실행은 안 함(임시 패키지로만 검증) |
| 13 | designs manifest 스키마 + gd_positions 요구 | T-PKG | `design/package.py` | test_design_package_sum.py, rebuild | 완료. **무조건 필수화는 아님** — 저작 마커 필드 존재 시 또는 `require_gd_positions=True`일 때만 발동(pathfinder.py:198 / curriculum.py:1587 기존 경로 보존) |
| 13c | 코어 템플릿 재생성 + resolver 정합 | T-PKG | `design/package.py`, `search/verify.py` | test_wave_verifier_resolver_guard.py (10) | 완료. `regen --dry-run` → new_dims [5,7], n_stale 1, stale `['T5_T6']`; 비-dry 실행은 "OLD library 기반 1건" 사유로 **거부**(정상) |
| 15 | HGC 밴드/재시작 게이트 | T-PKG | `design/library.py` | test_design_library_gates.py (31) | 부분. 순수함수+테스트만; **`bootstrap.py`에 미배선**(bootstrap.py byte-identical, 프로덕션 경로 무변경) |
| 16 | 준수성 배선 (write_authored_deck) | T-LAT | `design/lattice.py` | test_design_lattice_author.py | 부분. `enforce_new_type`은 **on-demand 경로(`write_authored_deck`)에만** 적용. `write_dec_deck`/`run_batch`/`pathfinder.py:33`은 기존 동작 보존(하드룰 2) |
| 17 | 커브 커버리지 감사 | T-PKG | `design/fuel_types.py` | test_design_library_gates.py | 부분. **헬퍼만 구현, 감사 산출물 미생성** — `fuel_types.parquet` 빌드가 238에서 소스 트리 부재로 불가 |
| 21 | 슬라이스 Z 캠페인 덱 (arm A/B) + 런북 | T-RUN | — | 스텁 executor로 CLI 배관만 검증 | **미완 — 덱/런북 자체는 미작성**(199 실행 아티팩트, 코드 0줄 과제) |

**신규 모듈 미생성 결정**: `lpopt/design/registry.py` (별칭 레지스트리는 `spec.py`의 `DesignRegistry`가 이미 담당 — 데드 서피스 회피), `lpopt/search/assets.py` 헬퍼(#13c는 resolver의 `_pair_of_folder`/`_feed_of_folder`를 함수-로컬 import로 재사용해 단일 리더 유지).

---

## 4. 열린 항목 (결정·조치 필요)

### 4-1. #4a 대리모델 접근 — **미해결, 최우선 블로커**
- surrogate 트리를 238에 스테이징하지 못함. `tar | ssh` 및 `scp -r` 두 경로 모두 Claude Code 자동모드 권한 분류기가 거부.
- 결과: **`--self-test` 통과 로그 없음, 체크포인트 SHA-256 매니페스트 없음** (#4a의 잔여 산출물 2건).
- 다운스트림은 착지 즉시 동작하도록 작성됨: `SurrogateBridge`가 `~/lattice_surrogate/kpin_pa`를 먼저 탐색.
- `USER2` 경로는 모드 비트에서 막힘 — USER2/root 권한 필요, 본 작업 범위 밖.
- **필요 결정**: 운영자가 수동 스테이징할지, 권한 예외를 열지.

### 4-2. σ_chain 판정 (T-SCREEN 보고서 기준)
- 역예측 산출물은 재현 가능(sha256 고정)하나, **측정 σ가 need 사다리의 인터록을 통과하지 못함** → `need_signal()`이 의도적으로 NotImplementedError 유지.
- P4 head shadow score는 **재계산이 아니라 인용**(238 CPU 추론 금지 + 타 트랙 소유). 출처: `minfxy_E1E2_f121_r2_results_20260831.md`.
- **필요 결정**: σ 사다리 임계값을 재등록할지, 아니면 need 신호 없이 슬라이스 Z를 진행할지.

### 4-3. 아직 필요한 결정
1. **DESIGN_GRID ratio 모순** — {0.85, 0.92} vs ZONE_RATIO_TARGET 0.85±0.03. 현재 문서화만. 어느 쪽을 정본으로 할지 확정 필요.
2. **`FuelDesign.gd_positions` 승격 여부** — 승격하면 #13의 요구를 무조건화(한 줄)할 수 있으나 37종 shipped type_id 및 `tests/test_design.py:41` 계약에 영향.
3. **`screen.py` 카탈로그 소스** — 거버닝 문서(§3 #4 ③)대로 `designs.json`을 읽도록 구현. 과제 브리프가 제시한 `data/store/fuel_types.parquet` 대안과 불일치(그리고 `lpopt/design/fuel_types.py`가 아니라 `lpopt/data/fuel_types.py`가 실모듈). 정본 확정 필요.
4. **#15 게이트 배선** — `bootstrap.py`의 `make_band_restart` / rebuild 래퍼에 G-H5a/b/c, G-H3/H3b/H3c 연결(게이트당 1줄). #11 착지 후 일괄 처리 권장.
5. **#17 감사 산출물** — 238에 `0_APR1400`/`3_GA_Surrogate` 동기화 후, 또는 199에서 `audit_curve_coverage(build_fuel_table(...), "ga80")` 실행 필요 (ga80 36/70 결손 확정).
6. **#21 슬라이스 Z 덱 2종(arm A/B) + 런북** 작성 — 실행 트랙 담당.
7. **`gd_wt ∈ {7, 9}` 및 `n_gd ∈ {0,4,8}`** 는 base deck 부재/SURROGATE_USAGE.md:143 근거로 폐쇄. `SurrogateBounds`에 명시. 개방 여부 결정 필요.

---

## 5. 미실행 / 의도적 보류
- 로컬 PC 연산 0건(하드룰 1). 로컬에서 한 유일한 "실행"은 `stage_slice_Z_181.ps1`의 PowerShell AST 파스(스크립트 미실행).
- 181 스테이징 0건, 181/199 launch 0건.
- sha-pinned 하네스(`ablation_wave.py`, `batchswap_wave.py`, `pinbu_wave.py`, `intervention_wave.py`), 덱, store parquet, model dir, `data/design/package` 무변경(하드룰 3).
- E:\lpopt_archive 스냅샷 미채취, 실 패키지 재생성 미실행 — 둘 다 199 운영자 작업.
- 어떤 assertion도 약화하지 않음.
