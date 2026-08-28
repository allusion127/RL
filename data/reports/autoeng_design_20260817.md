# autoeng — 자동 엔지니어 설계서

**작성 2026-08-17** · 드라이버 `autoeng.py` · 설정 `autoeng.toml` · 테스트 `tests/test_autoeng.py`
· 이 문서는 **설계 문서**이고, 실행 결과가 아니다. 아직 아무것도 발사하지 않았다.

---

## 0. 무엇을 만들었는가

사용자 요구 (원문):

> "이젠 모델로 엔지니어링 규칙을 적용해서 새로운 집합체, 새로운 feed 수 조합에 대해서도
> 쉽게 최적해로 도달하지 않나? 하나의 자동 엔지니어가 있는 것과 같았으면 좋겠어"

`autoeng.py`는 `(pair, feed, library)` 목표 셀 목록을 받아, 셀마다 **사람이 2026-08-16에
N1_N2/f113에서 손으로 돌린 그 순서 그대로** 실행한다.

```
0 PRECHECK → 1 PROBE → 2 OPEN → 3 HARVEST+MERGE → 4 RETRAIN+GATE → 5 MAP UPDATE → 6 NEXT CELL
```

**새로운 알고리즘은 하나도 없다.** 모든 단계는 이미 이 저장소에서 증명된 조각의 *조합*이다.
드라이버가 새로 기여하는 것은 정확히 세 가지다: 사전등록 자동화, 추가 전용 상태 로그,
설정으로 켜고 끄는 인간 게이트.

---

## 1. 아키텍처 — 무엇을 어디서 가져왔는가

| 단계 | 조합한 코드 | 근거 (증명된 선례) |
|---|---|---|
| **0 PRECHECK** | `lpopt.search.resolver.build_case_resolver` + `CaseAssetResolver.resolve` (사다리 dry-run), `data/store/records.parquet`, `data/reports/dbx_frontier_table.csv`, `scoping_mesh_20260815/cell_verdicts.csv` | f113 덱 헤더의 "THE MARKS" 블록 — 사람이 손으로 하던 일 |
| **1 PROBE** | `CurriculumDriver._phase_blind_probe` (plan 12.3) | `data/curriculum/cells/*/blind_probe.json` (36셀 실적) |
| **2 OPEN** | `lpopt optimize`, 노브는 `fpcamp_minfr_N1N2_f113_199.inp`에서 **그대로** 승계 | `fpcamp_N1N2_f113_results_20260816.md` — 100콜에 41 feasible, 첫 feasible 23콜 |
| **3 HARVEST+MERGE** | `lpopt merge-store` + 병합 전 백업 규율 | 같은 보고서 §5 — "100 new / 0 upgraded / 0 conflicts" |
| **4 RETRAIN+GATE** | `build_split_S1b.py` → `lpopt.remote push/train/status/pull` → `lpopt gate-promote` | 라운드 1–11, `ab2_addendum_S1G_20260816.md` (8연속 승격) |
| **5 MAP UPDATE** | `scoping_mesh.py` → `mesh_vs_db.py` → `--figure-only` | `scoping_mesh_20260815/README.md` |
| **6 ORDER** | `dbx_frontier_note_20260816.md` §4 스크린 + `comparison_readout.md` §10.3 전이 측정 | 이웃 feed 이득의 82–86%가 모델 기여 |

### 왜 최상위 스크립트인가 (`lpopt autoeng`이 아니라)

`lpopt/config.py`의 덱 로더는 **미지의 키를 하드 에러**로 다룬다. `[autoeng]` 섹션을 넣으려면
`_SECTIONS`, 데이터클래스, 검증기를 모두 건드려야 하는데, autoeng은 캠페인 *모드*가 아니라
명령들을 지휘하는 *오케스트레이터*다. 그리고 이 저장소는 이미 최상위 오케스트레이션 스크립트
관례를 갖고 있다 — `scoping_mesh.py`, `mesh_vs_db.py`, `ablation_wave.py`, `build_split_S1b.py`,
`train_policy_v1.py`. `autoeng.py`는 그 관례를 따른다. **발자국이 가장 작은 선택.**

---

## 2. 셀당 30 스텝 (전량 구체적인 명령)

| # | 단계 | 위치 | 스텝 | MASTER |
|---|---|---|---|---|
| 1 | 0-precheck | inproc | `precheck` — 자산 사다리·스토어 지지량·DB/메시 사전분포·모델 예측 하한 | |
| 2 | 0-precheck | inproc | `prereg` — 사전등록 + 덱 생성 + sha256 고정 | |
| 3 | 0-precheck | inproc | `arm_scripts` — probe.py / run_*.bat / launch_*.ps1 | |
| 4 | 0-precheck | local | `ship_kit` — kit로 scp | |
| 5 | 1-probe | 199 | `probe_launch` | **8** |
| 6 | 1-probe | 199 | `probe_wait` — rc 폴링 | |
| 7 | 1-probe | local | `probe_pull` | |
| 8 | 1-probe | inproc | `probe_readout` — target별 mae/bias/spearman/cov | |
| 9 | 2-open | 199 | `open_launch` | **100** |
| 10 | 2-open | 199 | `open_wait` | |
| 11 | 3-harvest | local | `harvest_run` — labels/state/status/report **만** (런디렉터리는 f113에서 10.3 GB였다) | |
| 12 | 3-harvest | local | `harvest_data` — kit의 `data/` (merge-store가 먹는 것) | |
| 13 | 3-harvest | inproc | `store_backup` — `.bak_pre_<cell>_<date>` | |
| 14 | 3-harvest | local | `merge_dryrun` | |
| 15 | 3-harvest | local | `merge` | |
| 16 | 4-retrain | inproc | `retrain_prereg` — 입력 sha256 고정 | |
| 17 | 4-retrain | local | `split_dryrun` | |
| 18 | 4-retrain | local | `split_write` | |
| 19 | 4-retrain | 238 | `train_push` | |
| 20 | 4-retrain | 238 | `train_launch` — GPU1 | |
| 21 | 4-retrain | 238 | `train_wait` | |
| 22 | 4-retrain | local | `train_pull` | |
| 23 | 4-retrain | local | `gate_check` — **게이트** `retrain_promote_fail` | |
| 24 | 4-retrain | local | `gate_promote` — PASS일 때만 | |
| 25–29 | 5-map | inproc/199/local | `mesh_baseline` → `mesh_recompute` → `mesh_pull` → `mesh_verdict` → `mesh_figure` | |
| 30 | 6-report | inproc | `cell_report` + `AUTOENG_LOG.md` | |

셀당 **108 MASTER 콜** (프로브 8 + 개방 100).

### 5단계가 조건부인 이유 (`skip_if = gate_failed`)

챔피언이 바뀌지 않았으면 메시를 다시 계산할 이유가 없다. `comparison_readout.md` §10.2가
그 근거다 — 스토어만 100행 늘었을 때 `min_pred_f_r`의 평균 이동은 **−0.0036**이고 55셀 중
52셀이 비트 동일했다. 풀 효과는 노이즈 안이다. 이건 노브가 아니라 **측정된 의존관계**다.

그리고 `mesh_recompute`가 `mesh_verdict`보다 **먼저** 온다. `comparison_readout.md` §10.3의
기록된 함정이다: `mesh_vs_db.py`의 `gap_total`은 디스크의 `mesh_nodes.csv`에서만 나오므로
메시를 다시 만들지 않고 `--model s1g`만 바꾸면 **"전이 없음"이라는 정반대 판독**이 나온다.
autoeng은 그 순서를 코드에 박아두었다.

---

## 3. 사전등록 자동화 — 자동화가 가장 먼저 잃어버릴 것

f113의 사전등록은 **덱 헤더 자체**였다. autoeng은 그 관례를 기계화한다.

`prereg` 스텝은 **MASTER 콜이 하나라도 나가기 전에** 다음을 측정해 덱 헤더에 못 박는다:

1. **우리 스토어의 이 셀 바닥** — 행수/수렴수/제약만족수, F_r 하한과 p10, cyclen 범위,
   과거 수렴률(→ 100콜에서 기대되는 유효 라벨 수), 신뢰영역 지지량.
2. **DB 진실값** — 코어 수, `F_r_min`, EFPD, CBC, F_q, 최적 조성 split, 노드 핀연소도.
3. **모델 자신의 예측 하한** — `mesh_min_pred_f_r`, `f_r_bias_tail`, `corrected_floor`,
   `gap_total/data/pool/search` 분해, 판정(`pool-starved` / `model-biased` / …).
4. **엘리트 부모 집합** — 이 pair에서 모든 feed에 걸쳐 존재하는 제약만족 행 수와 feed별 분포.
   f113을 끌고 간 교차-feed 전이 메커니즘이 이 셀에서 얼마나 두꺼운지가 여기서 드러난다.
5. **성공 기준** PRIMARY / SECONDARY / PARTIAL / **NULL** — NULL 분기까지 미리 쓴다.
6. **등록된 유의사항(REGISTERED CAVEATS)** — 사후에 처음 등장하면 안 되는 것들.
7. **자산 해석 예측** — 레벨과 provenance를 미리 적고, 모든 수렴 행이 그 값을 갖는지 사후 확인.

그리고 덱의 sha256이 **발사 스크립트의 게이트 값**이 된다. 헤더를 손대면 해시가 바뀌고
런처가 발사를 거부한다. 사전등록이 문서가 아니라 **강제 장치**가 되는 지점이다.

### 실제로 검증됨

N1_N2/f109에 대해 생성된 헤더가 아래를 자동으로 적어냈다 (전부 측정값):

```
1. 우리 스토어      166행 / 122수렴 / 0 제약만족,  F_r 하한 1.6940,  cyclen 596.9–632.3
                   과거 수렴률 73.5% → 100콜에서 ~73 유효 라벨,  신뢰영역 지지량 166
2. DB 진실값        41 코어,  F_r min 1.5420 @ 648.0 EFPD,  split 0.514 (56×53)
                   → 데이터 갭 +0.1520
3. 모델 예측 하한   1.5780 (feasible 0),  bias_tail +0.0168,  보정 하한 1.5612
                   gap_total +0.0400 / data +0.1560 / pool +0.0232 / search −0.1328
                   판정: pool-starved
4. 엘리트 부모      67행 (f113 41, f121 14, f125 12),  최상 부모 F_r 1.4932
등록 유의사항       DB 핀연소도 절벽 — 이 셀 DB 코어 100%가 노드 75 GWd/tU 초과 (프론티어 코어 79.97)
자산                level 2  pair_feed:MAS_RST.APRQ_11_0677.23   (f113과 동일한 사다리)
```

`retrain_prereg` 스텝이 고정한 입력 해시는 `ab2_addendum_S1G_20260816.md` §2가 손으로 적은
값과 **바이트 단위로 일치**했다 (`records.parquet` `dffd4d99…`, `maps.npz` `a88cb4a5…`).
사람의 규율을 기계가 그대로 재현한다는 증거다.

---

## 4. 인간 게이트 철학

v1 기본값: `pause_for_approval = ["new_assembly", "retrain_promote_fail", "budget_exceeded"]`.
**그 외 전부 자율.**

원칙은 하나다 — **되돌릴 수 없거나, 비싸거나, 사람이 기준을 바꾸고 싶어질 지점에서만 멈춘다.**

| 게이트 | 왜 사람인가 |
|---|---|
| `new_assembly` | DeCART2D → 라이브러리 → 부트스트랩 체인은 비싸고, paramA 선례상 사람의 판단이 여러 번 필요하다. v1은 **사전점검까지만 하고 멈춘다.** |
| `retrain_promote_fail` | 게이트 FAIL은 *결과*지 오류가 아니다. 등록된 폴백은 "현 챔피언 유지 + 플래그"이며, **자동화가 기준을 완화하는 길은 없다.** 사람이 그 폴백을 받아들일지만 정한다. |
| `budget_exceeded` | MASTER 콜은 실제 시간과 전기다. 다음 셀의 예산을 조용히 당겨쓰는 대신 멈춘다. |

멈추지 **않는** 것들도 의도적이다. 자산 해석 실패, 병합 충돌, 스텝 rc≠0은 그냥 **실패로
기록하고 셀을 중단**한다 — 승인 요청이 아니라 버그 신호이기 때문이다.

게이트가 걸리면 상태 로그에 `gate_pause` 이벤트가 남고 종료 코드 10을 반환한다.
재개는 `python autoeng.py --config autoeng.toml --resume`.

---

## 5. 함대 배분과 가드

| 박스 | 용도 |
|---|---|
| **199** | 캠페인 · 프로브 · 모델 전용 메시 스윕 |
| **238** | 학습만 (`lpopt_gpu1.inp`로 GPU 1 고정) |
| **181** | **절대 사용 금지** |
| **198** | 생산 전용 — autoeng이 건드리지 않는다 |

가드는 두 겹이다.
1. `load_autoeng_config`가 설정 파일이 금지 박스를 가리키면 **로드 자체를 거부**한다.
2. `guard_argv`가 **실행 직전 모든 argv를 검사**한다. 계획 단계에서 새 스텝이 잘못된 호스트를
   달고 들어와도 실행되지 않는다. 테스트가 전 30 스텝의 `argv`와 `poll_argv`를 통과시킨다.

발사 스크립트(`launch_*.ps1`)는 f113 런처가 증명한 세 게이트를 그대로 갖는다:
**busy 게이트** (쌓지 않고 거부), **덱 해시 게이트**, **사전조건 게이트** (챔피언·스토어 존재).
발사는 `Invoke-CimMethod Win32_Process Create` — `schtasks`는 이 함대에서 조용히 no-op 한다
(`ab2_addendum_ADF_20260810.md` §3).

---

## 6. 상태 / 재개 / MASTER 콜 원장

`data/autoeng/<run_id>/state.jsonl` — **한 줄 하나의 JSON, 추가 전용, 절대 재작성 없음.**

```json
{"seq":12,"ts":…,"iso":"2026-08-17T00:53:28","kind":"step_done","cell":"N1_N2_f109",
 "step":"open_launch","master_calls":100,"result":{…}}
```

* `kill -9` 비용은 **최대 한 스텝**이다. 로드할 때 마지막 줄이 깨져 있으면 마지막 정상 줄까지
  **파일을 잘라낸다**. 이 복구가 없으면 다음 추가 쓰기가 깨진 줄에 이어붙어 살아남은 기록까지
  망가진다 (테스트 `test_state_log_is_append_only_and_survives_a_torn_tail`).
* `master_calls`는 `step_done` 페이로드에 실려 있으므로 **원장이 크래시를 견딘다.**
* 재개는 로그 재생: 이미 `done`/`skipped`인 스텝은 건너뛴다. 실패한 스텝은 다시 시도한다.
* 실패한 스텝은 **0콜로 기록**된다 — 쓰지 않은 예산을 청구하지 않는다.

`python autoeng.py --config autoeng.toml --status`가 원장을 사람이 읽는 형태로 출력한다.

---

## 7. 셀 순서 — 전이 인식

```
정렬 키 = (신규집합체 여부, 이미 열린 셀까지의 거리, 핀연소도 절벽, DB F_r_min, pair, feed)
```

* **거리**: 같은 pair면 `|Δfeed|/4`, 다른 pair면 `10 + |Δfeed|/4`.
* **"이미 열린 셀"은 autoeng 자신의 로그가 아니라 정본 스토어에서 읽는다.** 제약만족 행이
  5행 이상인 (pair, feed)가 열린 셀이다. N1_N2/f113은 사람이 열었고, 전이 논거는 *누가
  만들었든 존재하는 라벨*에 대한 것이므로 반드시 포함되어야 한다.
* **왜 인접성이 지렛대인가**: `comparison_readout.md` §10.3이 측정했다 — f113 라벨이 흡수된
  뒤 이웃 열의 비관 편향이 f105 −0.0407, f109 −0.0516 움직였고 그중 **82% / 86%가 모델
  기여**(나머지는 풀 효과)였다. f109의 하락폭은 라벨을 넣은 f113 열 자신(−0.0460)보다 컸다.
  인접 셀은 실제로 더 싸다.
* **절벽 강등**: 모든 split 버킷의 `frac_node_ge75 == 1.0`이면 뒤로 민다
  (`dbx_frontier_note_20260816.md` §5 — 핀연소도 천장이 완화되지 않는 한 인허가 막다른 길).
* **신규 집합체 목표는 항상 마지막.** 멈추는 목표가 값싼 셀을 막으면 안 된다.

---

## 8. 파생 규칙 — f113 대비 검증

셀마다 바뀌는 노브는 **10개뿐**이고 (`CELL_OVERRIDE_KEYS`), 나머지는 부모 덱에서 그대로
승계된다. 테스트가 부모 덱과 생성 덱을 TOML 수준에서 전수 비교한다.

| 노브 | 규칙 | f113에서 사람이 고른 값 | 규칙이 주는 값 |
|---|---|---|---|
| `random_seed` | `2000 + hash(cell) % 800`, 이미 쓴 시드는 회피 | 1201 | 2459 (f109) — 충돌 없음 |
| `cycle_target_efpd` | DB 진실값 코어의 EFPD (**기록 전용, 아무것도 게이트하지 않음**) | 659.7 | 동일 규칙 |
| `cycle_tolerance_efpd` | 셀의 관측 cyclen 범위를 **실제로 덮도록** 넓힘, 5 단위 올림 | 30.0 | **45.0** — 사람의 30은 617.5를 덮지 못했다 |
| `near_miss_f_r` | 스토어 하한과 편향보정 하한의 중점, 양쪽에서 0.02 여유 | 1.65 | **1.66** — 같은 논거, 0.01 차이 |
| 나머지 6개 | pair/feed/model_dir/library_id/budget/title | | 기계적 |

두 곳에서 사람과 다르다. 둘 다 **아무것도 게이트하지 않는 노브**이고, 규칙 쪽이 더 정직하다
(`cycle_tolerance`는 사람이 적은 "이 셀의 전 범위를 덮는다"는 주장을 실제로 만족시킨다).
차이는 생성 덱 헤더에 명시된다.

### 기록된 결함 수정을 자동으로 승계

`[constraints]` 블록은 `min_fr_max_cycle`에서 **완전히 불활성**이고, f113 결과 보고서 결함 #2가
"다음 min_fr 덱에서는 빼라"고 적었다. autoeng은 그 블록을 **자동으로 뺀다**. 자동화가 사람의
실수를 복제하는 대신 사람의 교정을 승계하는 예다.

---

## 9. v1이 할 수 있는 것 / 할 수 없는 것 (정직하게)

### 할 수 있다 (테스트로 증명됨, 32/32 통과)

* 목표 셀에 대해 **자산 사다리를 실제로 dry-run** 하고 레벨/provenance를 예측한다.
  (f109 → level 2 `pair_feed:MAS_RST.APRQ_11_0677.23` — f113과 같은 경로)
* 스토어/DB/메시에서 **모든 마크를 측정**하고 사전등록 헤더에 못 박는다.
* f113 레시피를 **한 글자도 다시 타이핑하지 않고** 셀별 덱을 생성하고, 그 덱이 실제 엄격
  로더를 통과함을 확인한다.
* busy/해시/사전조건 게이트를 가진 발사 스크립트와 프로브 스크립트를 생성한다.
* 30 스텝의 완전한 실행 계획을 구체적 명령줄까지 산출한다 (`--dry-run`).
* `kill -9`를 견디는 원장을 쓰고, 재개 시 완료 스텝을 건너뛴다.
* 세 인간 게이트를 발동한다.
* 금지 박스를 향한 명령을 실행 전에 거부한다.
* 한국어 셀 보고서와 `AUTOENG_LOG.md` 실험노트를 쓴다.

### 할 수 없다 / 아직 검증되지 않았다

1. **원격 구간(199/238 ssh·scp·폴링)은 배선되었지만 이 파일이 한 번도 실행해 본 적이 없다.**
   `RecordingRunner`로만 검증했다. **첫 실전 실행은 반드시 사람이 지켜봐야 한다.**
   특히 `harvest_run`/`harvest_data`의 `scp` 인자 형태, kit 쪽 경로 구분자, 폴링 문자열
   (`"0"`, `"DONE"`)은 실물에서 한 번 확인해야 한다.
2. **신규 집합체는 사전점검 후 정지**한다. DeCART2D 레그는 v1 범위 밖이다.
3. **OPEN 레시피는 한 셀의 증거다. 법칙이 아니다.** f113은 41/100을 냈지만, 그건 교차-feed
   엘리트 부모가 26개나 있던 셀이었다. 부모가 얇은 셀에서 정체하면 그건 버그가 아니라 결과이며,
   생성되는 모든 사전등록이 그 NULL 분기를 미리 적는다.
4. **핀연소도는 게이트되지 않는다.** `min_fr_max_cycle`에는 핀연소도 축이 없다. N1_N2/f109처럼
   DB 코어 100%가 노드 75 GWd/tU를 넘는 셀에서는 **낮은 F_r 코어를 찾아도 인도 불가일 수
   있다.** 사전점검이 이걸 경고로 올리지만 **막지는 않는다** — 프론티어 라벨 자체는 여전히
   모델에 유용하기 때문이다. 이 판단은 사람이 목표 목록을 만들 때 내려야 한다.
5. **메시는 셀마다 다시 돈다.** 챔피언이 바뀔 때만이지만, 55셀 스윕은 199에서 30–90분이다.
   여러 셀을 연달아 열 때 이게 가장 큰 비-MASTER 비용이다.
6. **`build_split_S1b.py`의 S1x 이름은 한 글자 순열로 추정한다** (`s1g → s1h`). 챔피언 디렉터리가
   타임스탬프 이름이면 `next_arm`이 예외를 던지고, 사람이 arm 이름을 정해야 한다.
7. **동시 실행 보호가 없다.** 같은 `run_id`로 두 개를 띄우면 원장이 뒤섞인다. 199의 busy
   게이트가 캠페인 중복은 막지만, 드라이버 수준의 잠금은 없다.
8. **"최적해"의 정의는 여전히 F_r 최소화다.** 사용자가 정의를 바꾸면 (`user_criteria`) v1은
   따라가지 못한다 — v2 항목이다.

---

## 10. v2 항목

| 항목 | 무엇을 바꾸나 | 언제 |
|---|---|---|
| **정책망 생성기** | 지금 OPEN의 후보는 엘리트 모프 + 로컬 서치다. `lpopt/policy/`의 정책망이 제안자가 되면 "장전 규칙을 학습한 모델이 직접 배치를 짓는" 북극성에 한 걸음 가까워진다. | era-gap이 닫힌 뒤 (`policy_v1_results_20260815.md` 기준) |
| **`user_criteria` 목적함수** | `[case] mode = "user_criteria"`와 `UserCriteriaDriver`가 이미 있다. 목표 정의가 F_r에서 바뀔 때 autoeng의 OPEN 스텝이 그 모드를 쓰도록 하면 된다 (덱 승계 구조라 노브 교체만으로 가능). | 사용자가 최적의 정의를 바꿀 때 |
| **신규 집합체 완전 자동화** | DeCART2D → `design build-lib` → `design bootstrap` → 라이브러리 등록까지 자동. paramA 선례가 경로를 이미 증명했다. | 설계 체인이 두 번 이상 재현되고 나서 |
| **메시 증분화** | 챔피언이 바뀐 셀 근방만 다시 계산. §10.2가 이미 "52/55 셀이 비트 동일"임을 측정했으므로 근거는 있다. | 셀 개방이 5회를 넘어 메시 비용이 지배적이 될 때 |
| **다목적 판독** | 모든 FOM이 기록되므로 `(cyclen, F_r, 핀연소도)` 파레토를 셀 보고서에 자동 삽입 가능. | 인허가 쪽에서 핀연소도 천장이 확정될 때 |
| **드라이버 잠금 + 알림** | `run_id` 잠금 파일과 게이트 발동 시 알림. | 무인 운전을 실제로 시작할 때 |

---

## 11. 파일과 첫 실행 절차

| 파일 | 역할 |
|---|---|
| `autoeng.py` | 드라이버 (계획 + 실행 + 상태 + 게이트 + 보고) |
| `autoeng.toml` | 목표 목록 · 함대 · 예산 · 게이트 설정 (미지 키는 하드 에러) |
| `tests/test_autoeng.py` | 32개 단위 테스트 (`RecordingRunner` 주입 — MASTER·ssh 없음) |
| `data/autoeng/<run_id>/state.jsonl` | 추가 전용 원장 |
| `data/autoeng/<run_id>/cells/<cell>/` | `precheck.json` · `prereg.md` · 덱 · 스크립트 · `report.md` |
| `data/reports/AUTOENG_LOG.md` | 사람이 읽는 실험노트 (한국어) |

```bash
# 1. 계획만 본다 (아무것도 실행하지 않는다)
python autoeng.py --config autoeng.toml --dry-run

# 2. 첫 셀만, 사람이 지켜보며
python autoeng.py --config autoeng.toml --max-cells 1

# 3. 게이트에 걸리거나 kill 된 뒤
python autoeng.py --config autoeng.toml --status
python autoeng.py --config autoeng.toml --resume
```

**첫 실행 권고**: `--max-cells 1`로 시작하고, 4번 스텝(`ship_kit`)과 5번 스텝(`probe_launch`)
사이에서 한 번 멈춰 kit에 올라간 파일과 발사 스크립트를 눈으로 확인할 것. 원격 구간은
설계상 옳지만 아직 실물에서 돌아본 적이 없는 유일한 부분이다.
