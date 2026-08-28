# 축방향 업그레이드 계획 — 축방향 블랭킷 · Gd 컷백 (APR1400 연료설계 부분집합)

**2026-08-17 · PLANNING ONLY.** 코드 변경 0, MASTER 실행 0, DeCART 실행 0, 재학습 0.
**사용자 승인 전 어떤 Stage 도 착수하지 않는다** (§6).

이 문서는 `coreagnostic_v3_design_20260817.md` §3.8 / §7 이 **i-SMR 전용 별도 프로그램**으로
분리해 둔 "축방향 4중 확장" 중, **APR1400 연료설계에 해당하는 부분집합만** 계획한다.

---

## 1. 목표와 경계

### 1.1 무엇을 하려는가

엔지니어 AI가 **집합체 축방향 설계**를 설계변수로 다루게 한다. 구체적으로 두 가지다.

**(A) 축방향 블랭킷 (axial blanket)** — 연료봉 상·하단에 저농축(또는 천연) UO2 구간을 둔다.
축방향 중성자 누설을 줄여 **핵연료 이용률(주기길이 · 방출연소도)** 을 올리고, 단부 출력을
낮춰 **F_q 여유**를 만든다.

**(B) Gd 컷백 (cutback)** — 단부 구간에서 가연성 흡수체(Gd2O3) 농도/개수를 줄이거나 없앤다.
단부는 누설로 이미 출력이 낮으므로 그곳의 Gd는 **반응도를 낭비**한다. 컷백은 그 낭비를
회수하고, 동시에 EOC 축방향 형상(saddle)을 조정한다.

### 1.2 설계변수 정의 (제안)

현 `FuelDesign` (`lpopt/design/spec.py:47`) 은 5축 — `e1, e2, zoning_variant, gd_wt, n_gd` —
이며 **축방향으로 균질**하다. 제안하는 확장은 이 5축을 **축방향 존(zone)별로** 갖는 것이다.

```
AxialFuelDesign = 존 분할 + 존별 (e1, e2, zoning_variant, gd_wt, n_gd)
```

| 변수 | 기호 | 후보 값 | 물리 단위 근거 |
|---|---|---|---|
| 블랭킷 길이 (편단) | `L_b` | 0 / 1 / 2 노드 | **1 노드 = 15.24 cm = 정확히 6 인치.** `coredeck.CoreParams.zmesh = ("30.0","25*15.24","30.0")` (`coredeck.py:83`). 상용 블랭킷 6"/12" 가 **메시 변경 없이** 정수 노드로 떨어진다 |
| 블랭킷 농축도 | `e_b` | 0.71 ~ 3.2 w/o | 존 격자의 `UO2` 92235 (`lattice.edit_dec_text`가 이미 편집하는 토큰) |
| 블랭킷 Gd | — | **없음 (고정)** | 블랭킷 정의상 무-Gd. → **템플릿 갭**, §2.4 |
| 컷백 길이 (편단) | `L_c` | 0 / 1 / 2 / 3 노드 | 블랭킷 안쪽에 인접. `L_b + L_c ≤ 6` (편단 25% 이내) |
| 컷백 Gd 프로파일 | `f_gd` | `gd_wt` 배율 {0, 0.5, 0.75, 1.0} 또는 `n_gd` 감축 {24→20→16→12} | `gd_wt` 는 자유 수치 편집 가능 / `n_gd` 는 **템플릿이 있는 값만** (IGD_12/16/20/24, `lattice.py:32`) |
| 상하 대칭 | — | **대칭 고정 (기본)** | 비대칭은 AO 를 직접 흔들어 §1.3 경계에 접근한다. 기본 대칭, 비대칭은 별도 승인 사항 |

대칭 고정 + 위 후보 값이면 **존은 최대 3종** (블랭킷 / 컷백 / 중앙), 즉 **연료종 1개당
격자 계산 3건**이다 (§3 Stage-B 비용의 기반).

### 1.3 경계 — 범위 밖인 것 (명시)

> **2026-07-25 사용자 지시는 그대로 유효하다:**
> *"축방향 자유도 확장 안 함 — 어떤 덱·계획에도 넣지 말 것. APR1400급 대형 노심 +
> 붕산수 반응도 제어에서는 AO가 기준치 한참 아래라 예측 필요성이 사실상 없음."*
> (`coreagnostic_v3_design_20260817.md:456-459`)

**이 계획은 그 지시를 위반하지 않는다.** 이유는 범주가 다르기 때문이다:

| 구분 | 무엇 | 이 계획 |
|---|---|---|
| **장전 축방향 자유도** | `Pattern` = 69장 `%LPD_SHF` 카드. 축 카드를 추가해 "같은 집합체를 축방향으로 다르게 배치" | ❌ **범위 밖. 손대지 않는다** |
| **제어봉 위치 결정변수화** | 붕소 → 제어봉 제어 패러다임 전환 | ❌ **범위 밖** |
| **집합체 축방향 설계** | 블랭킷/컷백은 **집합체 제작 속성**. `e1/e2/gd_wt/n_gd` 와 **똑같은 경로**(DESIGN_GRID → DeCART 템플릿 → alias → 라이브러리)로 들어간다 | ✅ **이 계획** |
| **축방향 라벨 축** | EDIT6 수확 · F_z 타깃 승격 | ✅ **이 계획** (coreagnostic §3.8 의 (c)(d) 에 해당) |

`Pattern` 은 69장 `%LPD_SHF` 그대로다. 축방향은 **집합체 정체성 안에 접혀 들어가고**,
탐색기는 여전히 순수 2-D 셔플을 한다. coreagnostic §3.8 의 4중 갭 중 **(a) 결정변수 확장 ·
(b) 제어 패러다임은 건드리지 않고, (c) 라벨 축 · (d) F_z/AO 타깃 승격만** 가져온다.

### 1.4 정직한 동기 — AO 준수가 이유가 아니다

**중요.** 전 코퍼스 `|AO|max = 0.278 < 제약 0.30`, std ~0.014 이다
(`coreagnostic_v3_design_20260817.md:459`). 즉 **APR1400 에서 AO 는 구속 제약이 아니다.**
"AO 를 맞추기 위해 블랭킷이 필요하다"는 서사는 **측정과 어긋난다.**

블랭킷/컷백의 정직한 payoff 는 셋뿐이고, Stage-D 의 반증은 정확히 이것을 겨눈다:

1. **중성자 경제** — 축 누설 감소 → 주기길이 / 방출연소도. (진짜 payoff. 크기는 미측정)
2. **F_q 여유** — 3D F_q 는 현재 타깃이자 게이트값(전원 2.41 이하,
   `flat_assembly_fr_verdict_20260809.md:83`). 단부 출력 저하는 F_q 에 직접 온다
3. **LEU+ 인허가 맥락** — 집합체 **평균** 농축도를 낮추면서 중앙 농축도는 유지 →
   운송/저장 임계안전 및 연소도 크레딧 논의에 재료가 된다 (본 계획은 이 항목을
   **주장하지 않는다**. 요구조건은 인허가 측에서 내려와야 한다)

**만약 (1)(2) 어느 것도 유의하게 개선되지 않으면 이 프로그램은 null 이고, 중단한다.**

---

## 2. 현 자산 진단 (읽고 인용)

### 2.1 EDIT6 파서 — **있다. 그리고 이미 켜져 있다** (통념 정정)

| 자산 | 위치 | 상태 |
|---|---|---|
| EDIT6 파싱 | `lpopt/data/edit5.py:153` `_parse_edit6` | ✅ 동작. **플레인 수를 데이터에서 읽는다** (하드코딩 아님) |
| 스택 생성 | `lpopt/data/edit5.py:337` `stack_axial` | ✅ `(n_steps, n_planes) float32`, BOTTOM→TOP |
| 수확 | `lpopt/search/verify.py:495` `_hires_from_equilibrium_result` | ✅ **무조건 실행** (`HarvestingEquilibriumEvaluator.evaluate`, 수렴 시) |
| 저장 | `search/produce.py:1192-1196`, `search/campaign.py:1746-1749` | ✅ `<record_id>__axial` 키로 `maps.npz` 기록 |
| 라벨 의미 계약 | `lpopt/data/axial.py` (388줄) | ✅ 25 연료노드, 노심평균 1 정규화(편차 1.95e-4), `axial_offset()` 이 MASTER EDIT3 AO 를 **6e-5** 로 재현 |

**실측 (2026-08-17, `data/store/maps.npz`)**: 전체 키 74,992 = 베이스 맵 44,992 +
`__axial` **15,000** + `__traj` 15,000. 즉 **축방향 라벨 커버리지 33.3%**.
수확은 2026-07-25 이후 산출된 레코드에만 붙어 있고, **소급 불가**다
(`produce.py:1191` "NOT retroactive — a record produced without these loses the
resolution permanently").

> **정정**: "EDIT6 파서가 꺼져 있다"는 것은 부정확하다. **수확은 켜져 있고 라벨이 15,000행
> 쌓여 있다.** 꺼져 있는 것은 **모델 헤드**와 **타깃 승격**이다. 이것이 Stage-A 를 싸게 만든다.

**꺼져 있는 것 / 버려지는 것:**

* **EFPD 축이 버려진다.** `_parse_edit6` 는 `efpd` 를 파싱하지만(`edit5.py:145`),
  `stack_axial` 은 정렬에만 쓰고 **값 배열에 넣지 않는다**(`edit5.py:347-357`).
  그래서 `axial.py` 는 `ANCHORS = ("boc","eoc")` 두 개로 제한된다 (`axial.py:66`,
  모듈 docstring "The step axis has NO stored EFPD"). → **Stage-A 의 가장 싼 항목.**
* **`N_PLANES = 25` 는 `load_axial` 에서 하드 폭 검사**다 (`axial.py:62, 90`).
  파서는 유연하지만 로더는 아니다. → **축 메시를 바꾸면 라벨 계약이 깨진다** (§5).

### 2.2 모델 축 헤드 — **구현되어 있고, 챔피언에서 꺼져 있다**

| 자산 | 위치 | 상태 |
|---|---|---|
| 헤드 | `lpopt/model/net.py:191-192, 339-347, 478-484` | `n_axial_anchors × n_axial_modes` Linear. **`cfg` 기본값 0/0 → 미등록** (파라미터/출력키가 pre-axial 과 완전 동일) |
| 손실 | `lpopt/model/train.py:537` `axial_loss` | 표준화 계수 공간 masked Huber. `--axial-head / --axial-rank 6 / --axial-weight 0.2` |
| 데이터셋 | `lpopt/model/dataset_torch.py:127` `include_axial` | **기본 False** |
| 기저 | `lpopt/data/axial.py:189` `AxialBasis` | 앵커별 평균 + 영합 직교 성분. **rank 6 이면 leave-one-CAMPAIGN-out F_z 재구성 오차 9.3e-3** (within-cell F_z 스프레드 0.019 의 절반) — 기저는 정확도 병목이 아니다 |
| 서빙 | `lpopt/model/model_api.py:1262-1298` `has_axial` / `predict_axial` | ✅ 준비됨. 챔피언 meta 에 `axial_head` 없으면 False |
| 재학습 배선 | `lpopt/model/al_retrain.py:102-162` | ✅ 레시피 → CLI 플래그까지 이미 연결 |
| 테스트 | `tests/test_axial_head.py` (836줄) | ✅ 존재 |

**즉 Stage-C 는 "구현"이 아니라 "켜기 + 게이트 통과"다.**

### 2.3 현 타깃 — F_q · AO 는 이미 스칼라 타깃, **F_z 만 없다**

`net.py:14-15`, `N_TARGETS = 7`:
`[f_r, f_q, cbc_max, cyclen, ao_abs, discharge_burnup, max_pin_burnup]`

* **3D F_q**: 이미 타깃 + 셀 보정 대상(`train.py:2607`) + 게이트값 2.41
* **AO**: `ao_abs` 이미 타깃 + 보정 대상(`train.py:2608`)
* **F_z**: **없다.** 그러나 `axial.derived_metrics` 가 프로파일의 **해석적 함수**로
  `f_z / ao / asi / saddle_depth` 를 전부 준다 (`axial.py:164`)
  → **F_z 전용 스칼라 헤드를 새로 만들면 안 된다** (§5). 프로파일 헤드 하나에서 유도한다.

`saddle_depth` 는 EOC 새들 깊이와 `F_z` 의 상관 **+0.985**, `|AO|` 와는 **-0.58 (부호가 틀림)**
(`axial.py:42-46`). **블랭킷/컷백이 만드는 것이 정확히 이 새들 구조**이므로,
`|AO|` 만 보는 감시로는 이 프로그램의 효과를 볼 수 없다. 프로파일이 원시량이다.

### 2.4 MASTER 축방향 능력 — **덱 포맷이 이미 지원한다. 이것이 핵심 발견이다**

`lpopt/design/coredeck.py:161-178` `_lpd_static`:

```python
    for idx, alias in enumerate(aliases, start=1):
        lines.append(f"        {alias:<7} -1              25*{idx:<12} -5")   # %LPD_B&C
```

이 `%LPD_B&C` 한 줄은 **배치(batch) → 축방향 노드별 조성(composition) 리스트**다:
`-1`(하부축반사체 `REF_AXIAL_B`) + **`25*idx`(연료 25노드 전부 같은 조성)** + `-5`(상부축반사체).

> **`25*idx` 는 MASTER 의 한계가 아니라 우리의 선택이다.**
> 블랭킷/컷백 덱은 같은 카드에서 반복자만 바꾸면 된다:
> `-1  2*idx_blk  21*idx_mid  2*idx_blk  -5` (편단 2노드 블랭킷)
> `-1  1*idx_blk  2*idx_cut  19*idx_mid  2*idx_cut  1*idx_blk  -5` (블랭킷+컷백)
>
> **축 메시 변경 불필요, MASTER 입력 문법 확장 불필요, 반사체 처리 불변.**

따라야 하는 부수 변경 (전부 같은 파일, 좁다):

| 항목 | 현재 | 필요 |
|---|---|---|
| `_dims(n_types)` (`coredeck.py:105`) | `(3+n, 5+n)` — **batch 수와 comp 수를 같은 n 으로 묶어놨다** | **분리**: `nbatch = 3 + n_batches`, `ncomp = 5 + Σ(존 수)`. 배치 1개가 조성 여러 개를 참조하는 것이 축 zoning 의 정의 |
| `%LPD_C&X` (`coredeck.py:169-173`) | 조성 idx → `FA_<alias>` | 존마다 별도 `FA_<alias>` 엔트리 |
| `%LPD_HFF` (`coredeck.py:175-177`) | 조성 idx → `FA_<alias>` 핀 형상함수 | 존마다 별도 (블랭킷 존은 핀 파워 분포가 다르다) |
| `library_dims()` 게이트 (`coredeck.py:385`) | WaveVerifier 가 `(3+n, 5+n)` 을 검사 | 새 계산식으로 갱신 |
| alias 예산 (`spec.py:_alias_pool`) | 25 letters × 10 digits = **250**, 현재 사용 **37** (`data/design/package/registry.json`) | 존 3배 → 111. **여유 있음.** 240 전그리드 사전생성 시에만 구속(720 > 250) |

### 2.5 격자 체인 — **블랭킷 템플릿이 없다. 이것이 유일한 실질 갭이다**

`lpopt/design/lattice.py:32` `_TEMPLATE_ROOTS`:
```
("5.8_5.1/FA", (12, 16, 20)),   ("260624/FA", (24,))
```
템플릿은 `IGD_{12,16,20,24}/{gd}_{n}_z{1,2}/dec_FA_*.inp` 로만 존재하고,
`resolve_template` 은 `(gd_wt, n_gd, z)` 로 디렉터리를 찾는다. `FuelDesign.__post_init__`
는 **`n_gd` 가 4의 양의 배수**여야 통과시킨다 (`spec.py:74`).

* **컷백 존 = 실현 가능.** `gd_wt` 는 자유 수치 편집(`edit_dec_text`, `lattice.py:114`)이고,
  `n_gd` 감축도 12/16/20/24 안에서는 기존 템플릿이 있다. **신규 템플릿 불필요.**
* **블랭킷 존 (무-Gd) = 템플릿 없음.** `n_gd = 0` 은 스펙에서도 거부되고 템플릿도 없다.

  두 경로:
  * **(i) 정공법 — 무-Gd 핀맵 템플릿 신규 작성.** DeCART 덱 1종을 새로 만든다.
    **불변조건 강제**: 조립체 피치 20.7772 · 가이드튜브 `cellgeo 3-6` 은
    기존 템플릿과 바이트 동일이어야 한다 (`edit_dec_geom_text` 의 하드가드가 이미
    그것을 지키는 로직을 갖고 있다 — `lattice.py:228-237`). 사람 검토 필요.
  * **(ii) 부트스트랩 근사 — 기존 IGD 템플릿에 `gd_wt → 0`.**
    이때 UO2G 위치는 **Gd 없는 4.0 w/o 캐리어 핀**으로 남는다 (`GD_CARRIER_ENR = 4.0`,
    `spec.py:39`). 이것은 **균질 블랭킷이 아니다** — 물리적으로 유효한 격자이긴 하나
    설계 의도와 다르다. **E2E 배관 검증용으로만 쓰고, 물리 결론에는 쓰지 않는다.**

**격자 비용 실측 기준**: **DeCART 4-병렬 1 웨이브 = 4 격자 ≈ 13 분 wall**
(`flat_assembly_fr_plan_20260802.md:24, 397`; `run_batch(max_parallel=4)`, `lattice.py:409`).
결과물 ~35 MB/HGC + ~7.4 MB/라이브러리 블록.

**라이브러리 조성표**: `design/library.py` `build_master_library` 는 HGC 목록을
TotalBatcher4 에 넣어 `MAS_XSL`/`MAS_HFF` 를 만들고 COMP 수를 센다. 존이 늘면
**HGC 파일 수와 COMP 수가 그대로 늘 뿐** — 빌드 로직 변경 불필요, 다만
`LibraryBuild.ncomp` 가 `_dims` 게이트와 맞아야 한다 (§2.4).

### 2.6 특징(featurizer) — 축 설계 채널이 없다

`model/featurize.py:340-343` 의 `origin_*` 블록은 `origin_enrichment / origin_n_gd /
origin_gd_wt / origin_axial_z2` 를 갖는다. **`origin_axial_z2` 는 축방향이 아니라
반경 edge-zoning(z1/z2) 변형**이다 (이름이 오해를 부른다).
블랭킷/컷백 채널은 없다 → Stage-C 에서 **append-only** 로 추가 (v8 스키마).

### 2.7 라벨 밀도 프로그램과의 시너지 — **한 번의 수확 업그레이드가 둘을 다 먹인다**

`scaling_results_20260815.md` §5-6 의 등록된 결론:

> 용량은 **무릎점(10.4M)** 이고 더 키워도 안 는다. 남은 지렛대는 **구조**와
> **MASTER 콜당 라벨 정보밀도**뿐. …"**EDIT 6 축방향 파서**, 전 30스텝 EDIT 5 수확,
> 맵 레짐 커버리지 (추가 계산비 0, 저장 0.35 GiB)" … "**이것이 이 저장소의 측정에서
> 확인된 유일하게 남은 자릿수급 지렛대이고, 꺼져 있는 하루하루가 영구히 저해상도다.**"

즉 **Stage-A 는 이 계획이 없어도 해야 하는 일**이다. 이 계획은 그 위에 올라탄다:
Stage-A 의 비용은 축방향 프로그램의 **한계비용이 아니라 공유비용**이다.

---

## 3. 단계 계획

각 Stage: **산출물 · 비용 · 게이트 · 반증**. 게이트를 통과 못 하면 다음 Stage 로 가지 않는다.

### Stage-A — 라벨 인프라 (MASTER 0, DeCART 0, 기존 2D 러닝과 무간섭)

**산출물**

| # | 항목 | 내용 |
|---|---|---|
| A1 | EFPD 축 보존 | `stack_axial` 이 버리는 `efpd` 를 `<record_id>__axial_efpd` **신규 키**로 병기 저장. 기존 `__axial` 배열은 **바이트 동일 유지** |
| A2 | 앵커 확장 여지 | `ANCHORS` 를 `("boc","eoc")` 에서 EFPD-주소지정 가능 앵커(예: `mid`, `bu@50%`)로 확장 가능하게 (기본값은 **바꾸지 않는다**) |
| A3 | 파생 라벨 승격 | `f_z`, `saddle_depth` 를 `records.parquet` 파생 컬럼으로 기록 (`derived_metrics` 그대로 사용) |
| A4 | 커버리지 리포트 | 15,000/44,992 (33.3%) 의 밴드·캠페인·셀별 분포. 어느 셀이 축 라벨 결손인지 |
| A5 | AO 교차검증 | **전 코퍼스**에서 `axial_offset(EDIT6 profile)` vs 저장된 `ao_abs`(EDIT3 유래) 대조 |

**비용**: 코드 + 1회 재스캔. **MASTER 콜 0, DeCART 0, 재학습 0.** 저장 증가는
`__axial_efpd` 가 레코드당 ~27 float32 ≈ 108 B → 15,000행 ≈ **1.6 MB** (무시 가능).
`__axial` 자체는 이미 저장되어 있다.

**게이트**
* **G-A1**: A5 의 AO 항등식이 **전 코퍼스에서** `|Δ| ≤ 1e-3` (894행 표본에서는 6e-5,
  `axial.py:33-38`). EDIT3 와 EDIT6 는 **독립 출처**이므로 이것이 파서 계약의 진짜 검증이다
* **G-A2**: 기존 7 타깃 파이프라인 산출물 **바이트 동일** (append-only 증명, tripletype 의
  sha256 스냅샷 선례 — `tripletype_design_20260817.md` §2)
* **G-A3**: `__axial` 배열 자체가 변경되지 않았음 (해시 대조)

**반증 / 중단조건**
* AO 항등식이 전 코퍼스에서 깨지면(예: 특정 덱 패밀리에서 플레인 수가 25 가 아니거나
  정규화가 다름) → **파서 계약이 틀린 것.** 여기서 멈추고 고친다. Stage-B/C/D 전부 보류
* 커버리지가 특정 밴드에 극단 편중(예: 고농축 밴드에 축 라벨 0)이면 → Stage-C 재학습에서
  그 밴드는 축 supervision 이 없다. 그 사실을 먼저 기록하고 Stage-C 게이트를 조정

**무간섭 보장**: 전부 append-only 키 + 신규 파생 컬럼. `TARGETS` 불변, `include_axial`
기본 False 불변, 챔피언 불변. **진행 중인 2D 캠페인/러닝을 멈출 필요가 없다.**

---

### Stage-B — 축방향 격자 체인 (존별 HGC + 라이브러리 확장 + 부트스트랩 1건 E2E)

**산출물**

| # | 항목 | 내용 |
|---|---|---|
| B1 | 스펙 확장 | `AxialFuelDesign`(존 분할 + 존별 5축). **`n_zones == 1` 이면 기존 `FuelDesign` 과 정확히 동치** |
| B2 | 덱 생성 | `_lpd_static` 의 `25*idx` → 존 반복자 리스트. `_dims` 를 `(nbatch, ncomp)` 로 분리. `%LPD_C&X`/`%LPD_HFF` 존별 엔트리. `library_dims` 게이트 갱신 |
| B3 | 블랭킷 템플릿 | 무-Gd 핀맵 DeCART 템플릿 1종 (§2.5 경로 (i)). **경로 (ii)(gd_wt→0)는 배관 검증에만** |
| B4 | 격자 웨이브 | 1 연료종 × 3 존(블랭킷/컷백/중앙) → 2종 노심이면 **6 격자** |
| B5 | 라이브러리 | TotalBatcher 재빌드, COMP 수 검증 |
| B6 | **부트스트랩 1건 E2E** | 1개 (pair, feed) 셀에 대해 `make_band_restart` — cy1 + 평형체인 → 수렴 restart |

**비용**
* 격자: **6 격자 = 2 웨이브 ≈ 26 분 wall** + ~210 MB (35 MB × 6)
* 라이브러리 빌드: 분 단위 (TotalBatcher, `timeout_s=3600` 여유값)
* 부트스트랩 1건: **기존 밴드 부트스트랩 1건과 동일 비용** (cy1 1회 + 평형 체인 n 사이클).
  MASTER 콜 수는 `BootstrapResult.cycles_needed` 로 실측 기록
* 신규 코드: `design/` 4개 파일의 좁은 수정 + 테스트

**게이트**
* **G-B1 (동일성)**: `n_zones=1` 로 생성한 cy1/reload 덱이 **오늘의 덱과 바이트 동일**
  (sha256). tripletype 이 "2종 경로 바이트 동일 15/15" 로 확립한 그 방식 그대로
* **G-B2 (문법)**: `validate_reload_deck` + `library_dims` 통과. **MASTER 가 다중-COMP
  `%LPD_B&C` 를 받아들이고 `ncomp` 상한에 걸리지 않는다**
* **G-B3 (수렴)**: B6 부트스트랩이 수렴 (`converged=True`, cap 아님)
* **G-B4 (물리 방향성)**: 블랭킷 노심의 EDIT6 BOC 프로파일이 동일 (pair, feed) 무-블랭킷
  기준 대비 **단부 노드 출력이 낮고 F_z 가 낮다**. 방향이 반대면 덱이 틀린 것

**반증 / 중단조건**
* G-B1 실패 → 리팩터가 샜다. 되돌린다
* G-B2 에서 `ncomp` 상한 초과 → **밴드별 라이브러리 분할** (plan 12.1 이 이미 예견한 대응,
  `library.py:12` "split by enrichment band if exceeded"). 그래도 안 되면 존 수를 2로 축소
* G-B4 실패(단부 출력이 안 내려감) → `%LPD_B&C` 조성 순서/방향(BOTTOM→TOP) 오해.
  EDIT6 방향은 `axial.py:11-14` 에 확정되어 있으니 대조로 즉시 판정 가능
* 블랭킷 템플릿(B3) 검토가 막히면 → **경로 (ii)로 배관만 검증하고 Stage-C 는 보류.**
  물리 결론을 근사 격자로 내지 않는다

---

### Stage-C — 모델 (축 헤드 온 + 타깃 승격 + 재학습 게이트)

**산출물**

| # | 항목 | 내용 |
|---|---|---|
| C1 | 헤드 온 | `--axial-head --axial-rank 6 --axial-weight 0.2` (기본값 이미 존재, `train.py:245-247`). `include_axial=True` |
| C2 | F_z 승격 | **별도 스칼라 헤드 없이** `predict_axial` → `derived_metrics` 로 F_z/AO/ASI/saddle 제공. `model_api.has_axial()` 이 True 인 챔피언 |
| C3 | 설계 채널 (v8, append-only) | `origin_blanket_nodes` (L_b/2), `origin_blanket_enr` (기존 enr 스케일), `origin_cutback_nodes` (L_c/3), `origin_cutback_gd_frac`, `origin_axial_zoned` (존재 게이트). **비-zoning 레코드에서 전부 0 → v7 재현** |
| C4 | 셀 보정 | `f_z` 를 `train.py:2607` 의 셀-아핀 보정 루프에 편입 (`f_q`/`ao_abs` 와 동일 취급) |
| C5 | 재학습 A/B | 기존 재학습 게이트 머신러리 (`al_retrain.py`, `gate_retrain*.json`) |

**비용**: 재학습 1회 (5멤버 × 10.4M — 챔피언과 동일 구성). 축 헤드 추가 파라미터는
`head_hidden × (A×K) = head_hidden × 12` 로 **무시 가능**. DeCART 0, MASTER 0.

**게이트**
* **G-C1 (무해)**: 축 헤드 ON 이 기존 7 타깃을 **노이즈 바닥 이상 악화시키지 않는다**.
  바닥값은 등록되어 있다 — 5멤버 앙상블 간 **~0.01** (cyclen/map_cov), 단일 시드 간
  **~0.018** (`scaling_results_20260815.md` §6.5). 판정: `Δρ ≥ -0.01`
* **G-C2 (유용)**: held-out F_z 가 **"코퍼스 평균 프로파일" 베이스라인을 이긴다.**
  기저 자체의 재구성 한계는 F_z 오차 **9.3e-3** 로 알려져 있으므로(`axial.py:180-183`),
  그보다 나쁘면 헤드가 아무것도 배우지 않은 것
* **G-C3 (v7 환원)**: v8 채널이 전부 0 인 레거시 레코드에서 v7 featurization 과 **바이트 동일**
* **G-C4 (수용기준 정합)**: 최적점 예측-MASTER 일치 기준(cyclen ≤1 EFPD, CBC ≤1 ppm)이
  축 헤드 ON 에서도 유지

**반증 / 중단조건**
* G-C2 실패 → **헤드를 끈 채로 둔다.** 계획을 "라벨만" 으로 축소하고 Stage-D 는 폐기.
  이것은 실패가 아니라 **싸게 얻은 음성 결과**다 (재학습 1회 비용)
* G-C1 실패 → `axial_weight` 를 낮춰 1회 재시도. 그래도 악화면 헤드 OFF
* 축 라벨 33% 커버리지가 문제라면(G-A4 리포트) → masked loss 가 이미 결손을 다루므로
  크래시는 없으나, **F_z 예측이 라벨 있는 캠페인 쪽으로 편향**될 수 있다. leave-one-CAMPAIGN-out
  으로 확인 (기저 검증이 이미 쓰는 프로토콜)

---

### Stage-D — 설계 루프 통합 (해석적 요구조건 우선 원칙 유지)

**원칙 (변경 없음)**: **해석적 요구조건을 먼저 쓴다.** MASTER/DeCART 콜은 그 요구조건을
해석적으로 통과한 후보에만 쓴다. 블랭킷/컷백은 "돌려보고 좋으면 채택"하는 축이 아니다.

**산출물**

| # | 항목 | 내용 |
|---|---|---|
| D1 | 해석적 요구조건서 | 착수 **전에** 기록: (a) 목표 Δ(주기길이) 또는 Δ(방출연소도), (b) 그것을 사려면 필요한 축 누설 감소량 — 1-D 2군 누설 추정 `B_z² = (π/(H+2δ))²` 기반, (c) 그로부터 나오는 `L_b`·`e_b` 후보 범위, (d) F_q 여유 목표. **예측 없이 결정 가능한 부분은 전부 여기서 결정** |
| D2 | 스크리닝 | 모델 예측(F_z/F_q/cyclen) + 해석적 스크린 → 살아남은 후보만 격자 |
| D3 | 축 변수를 설계 루프 축으로 | `AxialFuelDesign` 을 기존 `e1/e2/gd` 축과 동렬로. **`Pattern` 은 불변** |
| D4 | 귀속 실험 (필수) | 챔피언 후보에 대해 **동일 `e_core` 에서 zoning 만 제거한 대조군** |

**비용**: 후보당 격자 3건(=1 웨이브 미만) + 평형 체인 1건. D1 은 **계산 0**.

**게이트**
* **G-D1**: 산출된 설계가 목적함수에서 챔피언을 이기고, **동시에** F_q ≤ 2.41,
  CBC ≤ 1600, |AO| ≤ 0.30 을 만족 (현행 게이트값 그대로)
* **G-D2 (귀속)**: D4 대조군 대비 이득이 **노이즈 바닥(~0.01)을 넘어** 남는다
* **G-D3 (해석 일치)**: 실측 Δ(주기길이)가 D1 의 해석적 예측과 **부호가 같고 자릿수가 맞는다**

**반증 / 중단조건 (프로그램 전체의 최종 반증)**
* **G-D2 실패 — 이것이 가장 가능성 높은 null 이다.** 동일 `e_core` 에서 zoning 을 제거해도
  이득이 그대로면, 블랭킷은 **농축도를 다시 라벨링한 것**일 뿐이다 → **축을 폐기한다**
* G-D3 실패 → 해석 모델이 틀렸거나 격자가 틀렸다. 둘 다 확인 전까지 확장 금지
* AO 가 0.30 에 접근하기 시작하면(비대칭 존을 허용한 경우) → **즉시 대칭 고정으로 되돌린다**

---

## 4. 비용 총괄

| Stage | DeCART | MASTER | 재학습 | wall (추정) | 저장 | 되돌림 가능? |
|---|---|---|---|---|---|---|
| **A** 라벨 인프라 | 0 | **0** | 0 | 코드 + 1회 재스캔 | +~2 MB | 예 (append-only) |
| **B** 격자 체인 | **6 격자 = 2 웨이브** | 부트스트랩 1건 (cy1+체인) | 0 | **≈26 분** + 부트스트랩 | ~210 MB HGC + 라이브러리 | 예 (`n_zones=1` 바이트 동일 게이트) |
| **C** 모델 | 0 | 0 | **1회** | 재학습 1회 | 체크포인트 | 예 (헤드 미등록 = pre-axial 과 동일 state_dict) |
| **D** 설계 루프 | 후보당 ≈1 웨이브 | 후보당 평형체인 1건 | 0 | 후보 수에 비례 | — | 예 |

**요약**: **Stage-A 는 사실상 공짜이고, 이 계획이 없어도 해야 한다**(§2.7).
**Stage-B 가 최초의 실질 지출이며 ≈26 분 DeCART + 부트스트랩 1건**으로, 저장소 기준
"작은 실험" 규모다(비교: `flat_assembly_fr_plan` 이 4 격자 13 분을 "EXPENSIVE" 로 표기).
**Stage-C 는 재학습 1회.** 즉 **Stage-D 이전까지 총 지출은 DeCART 26 분 + MASTER 부트스트랩
1건 + 재학습 1회**이고, 그 시점에 프로그램의 성패를 판정할 수 있다.

**가장 비싼 것은 계산이 아니라 B3(블랭킷 무-Gd 템플릿) 의 사람 검토**다. 기하 불변조건
(조립체 피치 · 가이드튜브)을 지키는 신규 DeCART 핀맵이므로 자동화 대상이 아니다.

---

## 5. 하지 말 것

1. **축방향 장전 자유도.** `%LPD_SHF` 에 축 카드를 넣지 않는다. `Pattern` 은 69장 그대로.
   **2026-07-25 사용자 지시** (`coreagnostic_v3_design_20260817.md:456`, `:1086`
   "축방향 자유도를 APR1400/OPR1000 v3에 넣기 — 이미 금지")
2. **제어봉 위치 결정변수화.** 붕소 제어 패러다임 유지. i-SMR 프로그램 소관
3. **축 메시 변경.** `nz=27`, `25*15.24 cm` 고정. `axial.load_axial` 이 폭 25 를 하드 검사
   (`axial.py:90`) — 메시를 바꾸면 15,000행 라벨 계약이 즉시 깨진다.
   블랭킷 길이는 **노드 정수배로만** 표현한다 (6"/12" 가 정확히 맞는다)
4. **F_z 전용 스칼라 헤드 신설.** `derived_metrics` 가 프로파일의 해석적 함수다
   (`axial.py:164`). 두 개의 진실원천을 만들지 않는다
5. **기존 라벨 소급 재계산 시도.** 불가능하다 (`produce.py:1191` "NOT retroactive").
   결손 33%→100% 를 위해 MASTER 를 다시 돌리지 않는다
6. **조립체 피치 / 가이드튜브 기하 변경.** `edit_dec_geom_text` 하드가드
   (`lattice.py:228-237`) 를 우회하지 않는다
7. **`gd_wt→0` 근사 격자로 물리 결론 내기.** 배관 검증 전용 (§2.5 (ii))
8. **AO 준수를 동기로 서술하기.** `|AO|max = 0.278 < 0.30`. 측정과 어긋난다 (§1.4)
9. **비대칭 상하 블랭킷을 기본값으로.** AO 를 직접 흔든다. 별도 승인 사항
10. **Stage 를 건너뛰기.** 특히 **A 의 AO 교차검증 없이 C 로 가지 않는다** —
    라벨 계약이 틀렸는데 헤드를 켜면 틀린 것을 학습한다
11. **캠페인 착수 · 신규 리포트 남발.** 이 문서가 계획이다

---

## 6. 착수 트리거 제안

> **사용자 승인 전 어떤 Stage 도 착수하지 않는다.** 이 문서는 계획이며, 코드/덱/실행
> 어느 것도 만들지 않았다.

| Stage | 착수 조건 (제안) |
|---|---|
| **A** | **사용자 승인만으로 충분.** MASTER/DeCART 0, 되돌림 가능, append-only,
진행 중 러닝과 무간섭. 라벨 밀도 프로그램(`scaling_results` §6.3)이 이미 독립적으로 요구 |
| **B** | (i) A 게이트 전부 통과, **그리고** (ii) 사용자가 **블랭킷 스펙을 지정** — 길이(노드 수)와
농축도 후보. 인허가/설계 측 요구조건이 있으면 그것이 우선. 스펙 없이 격자를 돌리지 않는다 |
| **C** | B 게이트 전부 통과, 그리고 **축 라벨 커버리지가 재학습을 지탱하는지** A4 리포트로 확인 |
| **D** | C 게이트 전부 통과, 그리고 **D1 해석적 요구조건서가 먼저 작성·승인**됨.
D1 없이 D2 를 시작하지 않는다 (해석적 요구조건 우선 원칙) |

**권고 순서**: A 를 먼저 승인해도 잃을 것이 없다 (계산비 0, 되돌림 가능, 독립적 가치 있음).
B 는 **블랭킷 스펙이 정해진 뒤**에 판단하는 것이 맞다.

---

## 부록 — 인용 위치 색인

| 사실 | 파일:행 |
|---|---|
| `%LPD_B&C` 가 축 노드별 조성을 이미 받는다 (`25*idx`) | `lpopt/design/coredeck.py:167` |
| 축 메시 = 30.0 + 25×15.24 + 30.0 (1 노드 = 6") | `lpopt/design/coredeck.py:83` |
| `(nbatch, ncomp)` 가 하나의 `n_types` 로 묶여 있음 | `lpopt/design/coredeck.py:105-107` |
| 격자 템플릿 = IGD_{12,16,20,24} 뿐 (무-Gd 없음) | `lpopt/design/lattice.py:32`, `spec.py:74` |
| `gd_wt` 는 자유 수치 편집 가능 | `lpopt/design/lattice.py:114-140` |
| 기하 불변 하드가드 (피치·가이드튜브) | `lpopt/design/lattice.py:228-237` |
| DeCART 4-병렬 웨이브 ≈ 13 분 | `data/reports/flat_assembly_fr_plan_20260802.md:24, 397` |
| alias 풀 250, 현재 37 사용 | `lpopt/design/spec.py:_alias_pool`, `data/design/package/registry.json` |
| EDIT6 파서 (플레인 수를 데이터에서 읽음) | `lpopt/data/edit5.py:153-175` |
| `stack_axial` 이 EFPD 를 버림 | `lpopt/data/edit5.py:337-357` |
| EDIT6 수확이 켜져 있음 (무조건) | `lpopt/search/verify.py:495`, `:594-596` |
| 저장 (소급 불가) | `lpopt/search/produce.py:1189-1196` |
| 축 라벨 15,000 / 베이스 44,992 (33.3%) | `data/store/maps.npz` (2026-08-17 실측) |
| 라벨 계약 · AO 항등식 6e-5 · `N_PLANES=25` 하드검사 | `lpopt/data/axial.py:9-47, 62, 90` |
| 기저 rank 6, F_z 재구성 9.3e-3 | `lpopt/data/axial.py:177-184` |
| saddle_depth ↔ F_z +0.985, ↔ \|AO\| −0.58 | `lpopt/data/axial.py:40-46` |
| 축 헤드 구현 · 기본 0/0 (꺼짐) | `lpopt/model/net.py:191-192, 339-347` |
| `axial_loss` · CLI 플래그 | `lpopt/model/train.py:245-247, 537` |
| `include_axial` 기본 False | `lpopt/model/dataset_torch.py:127` |
| `has_axial` / `predict_axial` 서빙 | `lpopt/model/model_api.py:1262-1298` |
| 재학습 레시피 배선 | `lpopt/model/al_retrain.py:102-162` |
| 현 7 타깃 (f_q·ao_abs 포함, f_z 없음) | `lpopt/model/net.py:14-15, 48` |
| 셀-아핀 보정 루프 | `lpopt/model/train.py:2601-2608` |
| 2026-07-25 축방향 자유도 금지 지시 | `data/reports/coreagnostic_v3_design_20260817.md:446-465, 1086` |
| \|AO\|max 0.278 < 0.30 | `data/reports/coreagnostic_v3_design_20260817.md:459` |
| 라벨 밀도가 남은 유일한 자릿수급 지렛대 | `data/reports/scaling_results_20260815.md:244-270` |
| 노이즈 바닥 ~0.01 / ~0.018 | `data/reports/scaling_results_20260815.md` §6.5 |
| 바이트 동일 sha256 스냅샷 선례 | `data/reports/tripletype_design_20260817.md` §2 |
| F_q 게이트 2.41 / CBC 1600 | `data/reports/flat_assembly_fr_verdict_20260809.md:83` |

## §7 사용자 확정 설계 제약 (2026-08-17, 착수 전 등록)

- **블랭킷 길이 = 15 cm 고정 (APR1400 한정)** — 길이는 설계변수에서 제외.
- **블랭킷 농축도 = 자유 설계변수** (무-Gd 전제는 유지, 농축도 수준은 최적화 대상).
- 모델링 주기: 현 축 메시 노드는 15.24 cm(6") — 15 cm 지정과 0.24 cm 차이.
  Stage-B 착수 시 결정 필요: (a) 1노드 근사(15.24 cm)로 표현하고 차이를 문서화,
  (b) 상하단 노드만 15 cm로 비균등 메시 조정. 권고는 (a) — MASTER 노달 정확도
  대비 0.24 cm는 소음 수준이며 메시 변경 리스크가 더 큼. 착수 시 재확인.
- cutback: 기존 계획대로 (길이·Gd 프로파일 설계변수 유지).
