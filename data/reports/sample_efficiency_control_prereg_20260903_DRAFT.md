# 표본효율 동일-셀 대조실험 — E1_E2 / f121 / ga80 — 사전등록 (DRAFT)

**작성 2026-09-03. 아무것도 실행되지 않았다** (로컬 연산 0 / DeCART 0 / MASTER 0 / 199·181·238 쓰기 0).
본 문서는 **초안(DRAFT)** 이다. §8 의 이식 태스크가 끝나고 §9 의 게이트가 전부 PASS 하기 전에는
**발사 승인 문서가 아니다**. 발사 시점에 §10 스탬프를 채우고 파일명에서 `_DRAFT` 를 떼는 것이
정본화 절차다.

- 선행: `data/reports/sample_efficiency_kpi_20260903.md` (이하 **KPI 문서**). 그 §6.1 이 등록한
  공백을 닫는 것이 본 실험의 유일한 존재 이유다:
  > *"동일 셀 SA 대조군이 없다. T6_T4/f121 또는 E1_E2/f121 에서 MOCHA SA 를 랜덤 시드로 1,000콜
  > 돌리는 것이 유일한 apples-to-apples 실험이다. 이것 없이는 K1 의 비교항이 없다."*
- 관련: `produce_fxyera_r1_prereg_20260829.md` §3.1 (셀 인구),
  `transpose_noise_measured_20260725.md` (라벨 잡음),
  `assembly_on_demand_design_20260902.md` §7.4 (incumbent 정의),
  `slice_Z_pipeline_runbook_20260903.md` (199 선행 점유).

---

## 0. 한 문단 진술 · 그리고 KPI 문서 정정 2건

주장은 **"SA 가 10,000–30,000 회의 실제 노심계산으로 하는 일을, 학습된 프론티어 AI 는 수십~수백
회로 한다"** 이다. KPI 문서는 이것이 현재 데이터로 **증명 불가**임을 확인했다 — SA 자산과 lpopt
캠페인은 셀·목적함수·초기조건이 하나도 겹치지 않는다. 본 실험은 **셀 하나**
(`ga80 / E1_E2 / feed 121 / e_core 5.00`) 위에서 세 팔을 돌려 그 비교항을 만든다:
**(A) MOCHA SA 랜덤 콜드스타트 1,000콜 × 시드 2**, **(B) lpopt 콜드스타트 300콜 × 시드 2**,
**(C) lpopt 웜스타트(기존 자산, 참조)**. 주 지표는 KPI §5 의 **K1 / K2 / K3** 이며 카운터는
**MASTER 평형체인 1회 = 1콜** 뿐이다.

**본 문서를 쓰면서 발견한 KPI 문서의 오류 2건 — 결과 발표 전에 정정해야 한다.**

| # | KPI 문서의 서술 | 실측 | 근거 |
|---|---|---|---|
| **E-1** | §4-2: *"SA 는 MOCHA 제약 어닐링 스칼라 J(F_r ≤ 1.55 는 제약, 목적이 아님) … 'F_r 을 최소화하려 애쓴 SA' 는 데이터에 없다."* | **틀렸다.** 최소 2개 런의 `run_meta.json` 이 `constraints[0] = {name: fr, parameter: max_frp, limit: 1.55, kind: max, weight: 1.0, is_objective: true, ratchet: true}`, `weights = {"fr": 1.0}`, `aggregation = tchebycheff`, `zstar_fixed = {"fr": 2.2501}` 를 기록한다. **SA 는 F_r 을 단독 목적으로 돌았다.** | `2_LP/0_Case/runs/20260705-190653_random_s116968488/run_meta.json`, `…/20260702-204511_random_s1991870508/run_meta.json` |
| **E-2** | §1-4 표의 `wall(h)` 열 합계 **2,357** 을 "wall-hour" 로 인용 | 그 열은 **Σ`eval_wall_s`(레인-시간)** 이지 벽시계가 아니다. 7,285콜 런의 실제 경과는 **71.06 h**(2026-07-05 19:08:30 → 07-08 18:12:08, `run_meta.timing`), Σ`eval_wall_s` 는 502.08 레인-시간. 8-워커 병렬이므로 **벽시계/콜 ≈ 34.7 s**, 레인/콜 ≈ 245 s | 동상 `run_meta.timing` (`n_eval 7360`, `n_master_calls 70506`, `eval_per_hour 103.6`) |

**E-1 은 본 실험에 유리한 정정이다** — MOCHA 는 이미 "F_r 최소화 SA" 로 검증된 설정을 갖고 있어
A 팔의 목적함수 전환이 *신규 튜닝이 아니라 기보유 설정의 재사용*이다. **E-2 는 불리한 정정이다** —
"2,357 wall-hour" 는 대외 인용에서 즉시 내려야 한다(실제 벽시계는 그 1/8 수준).

**이 문서가 하지 않는 것.** "lpopt 가 SA 보다 좋은 노심을 찾는다"를 증명하려는 것이 아니다.
셀 기록 1.4636 은 lpopt 계보가 만든 값이고 A 팔은 그것을 모른 채 출발한다 — 비대칭은 남는다(§4.4).
증명 대상은 오직 **"같은 셀·같은 게이트·같은 카운터에서 기록 근방에 도달하는 데 드는 MASTER 콜
수의 차이"** 하나다.

---

## 1. 셀 — 동결

| 항목 | 값 | 출처 |
|---|---|---|
| library | `ga80` | `fpcamp_minfr_199.inp` `[model] library_id`; MASTER 자산은 199 `kit_frontier\FEASIBLE_PACKAGE\{hgc,lib}` (36 × `FA_*.HGC`, `MAS_XSL` 30,869,930 B, `MAS_HFF` 32,388,560 B) |
| pair | `E1_E2` | 동상 `[case] pair`. `FA_E1.HGC` / `FA_E2.HGC` 각 7,395,955 B 실재 확인 |
| feed | **121** (241 FA 중 fresh 121) | 동상 `[case] feed`; MOCHA `core.feed_fa_count: 121` — **이미 동일** |
| e_core | **5.000** (241 FA U-질량 가중 초기농축도 평균) | `produce_fxyera_r1_prereg_20260829.md` §3.1 |
| 스토어 인구 (2026-08-29) | **590 행** | 동상. C 팔의 `prior_rows` |
| **셀 기록 `R_cell` (F_r)** | **1.4636** @ cyclen 633.329 EFPD | `fpcamp_199` 98번째 콜; KPI §1-1; `batchswap625_analyze.py:39` |
| **incumbent `F_xy`** | **1.5295** (`a785eded…`, `fpcamp_minfr_199`; F_r 1.4694 · CBC 1330.81 · F_q 1.8422 · \|AO\| 0.0404 · cyclen 638.639) | `assembly_on_demand_design_20260902.md` §7.4 |
| **F_r 라벨 잡음 σ** | **0.00595** (전치 재시작, n=22) | `transpose_noise_measured_20260725.md` |

`R_cell = 1.4636` 과 `F_xy incumbent = 1.5295` 는 **본 문서 시점에 동결**한다. 실험 중 다른 캠페인이
이 셀에서 더 낮은 값을 만들어도 **마크는 갱신하지 않는다**(사후 이동 금지).

**셀 선택 사유.** (a) E1_E2/f121 은 KPI §3 에서 **완전 포화**(5라운드 연속 Δ/100콜 = 0.0000)로
판정된 유일한 셀 — lpopt 쪽 바닥이 확립돼 있어 A 팔의 도달/미도달 판정이 깨끗하다.
(b) T6_T4 의 기록은 guided 캠페인이 아니라 `batchswap_enum` 열거가 잡았으므로(KPI §3) 비교 대상이
오염된다. (c) ga80 의 MASTER 자산이 199 에 **이미 MOCHA 가 요구하는 형식 그대로** 있다(§8).

---

## 2. 가설 — 결과 전에 문장으로 고정

- **H1 (주가설, 자릿수).** `K1@0.010(A) / K1@0.010(B) ≥ 3`.
- **H2 (강한 형태).** A 팔은 1,000콜 안에 `R_cell + 0.010` 에 **도달하지 못한다**.
- **H3 (시딩 귀속).** `K1@0.010(B) ≤ 300` — 스토어 elite 없이도 프론티어에 닿는다.
  **H3 가 깨지면** 본 프로그램의 "직관"은 사전 데이터의 재인용이며, 이것이 가장 중요한
  등록된 반증 경로다.
- **H4 (귀무).** 세 팔의 K1 이 서로 2배 이내 → 셀이 쉬웠다. 등록된 다음 수는 **셀 교체**
  (N1_N2/f113 — KPI §3 에서 유일하게 아직 이득 구간)이지 예산 증액이 아니다.

**등록된 사전 예측 (정직한 불확실성 포함).**

| 양 | 예측 | 근거 | 예측이 틀릴 수 있는 이유 |
|---|---|---|---|
| `K1@0.005(B)` | **80–300콜** (점추정 150) | C 팔 계보는 41콜이었으나 590행 위에서다. 콜드스타트는 elite 부모가 없어 초기 wave 가 random/heuristic 으로 채워진다(`construct.py:651-666`) | 챔피언 s1j 가 이 셀을 학습했다(N-7) → 훨씬 빠를 수 있다 |
| `K1@0.010(A)` | **>1000 (미도달) 0.55 / 400–1000콜 0.45** | MOCHA 13런의 자기-best+0.005 도달 중앙값 1,248콜; F_r 1.55 미도달 | **E-1 정정 반영**: 그 13런은 이미 F_r 단독 목적이었다 → SA 가 이 목적에 무능한 것이 아니라 *셀/라이브러리가 달랐을* 뿐일 수 있다. 예측을 미도달 0.6→0.55 로 낮춘 이유가 이것이다 |
| `K3 AUF@300` | `AUF(B) < AUF(A)` 를 2/2 시드쌍에서 | 궤적 전체 지표라 운 의존이 낮다 | A 가 초기 대량 표본으로 평균을 빨리 내리면 역전 가능 |

---

## 3. 팔 정의

### 3.1 A — MOCHA SA 콜드스타트

| 항목 | 값 |
|---|---|
| 엔진 | `MOCHA.py run --start random` (랜덤 초기 LP; `screening.random_rule()` `MOCHA/screening.py:104-136`) |
| 예산 | **1,000 평형체인** (`--max-evals 1000`), **하드 펜스** |
| 시드 | **2** (`--sa-seed` / `--rule-seed` 쌍) |
| GA 사전탐색 | **끔** — `sa.global_exploration.enabled: false`. 켜두면 evaluated-GA 가 48×15 ≈ 720 평가를 먹어 "SA 콜드스타트"가 아니게 된다 |
| 목적 | `objective.parameters.fr` 단독 ratchet 목적 (§4.2) |
| 게이트 | Tier-1 4종 + pin BU 78.0 + cyclen ≥ 625 |
| 워커 | `sa.parallel_workers: 12`, `sa.pcore_cpus: 0..23` (199 는 전원 P-core) |
| 스테이지 | `stage_max_samples 32` / `stage_max_accept 12` / `max_stages 300` / `cooling_alpha 0.97` / `t0: null`(Park χ₀=0.99 자동보정) — **전부 현행 정본 유지** |

### 3.2 B — lpopt 콜드스타트

| 항목 | 값 |
|---|---|
| 챔피언 | **s1j** (`data/models/s1j`, `cond_schema = v8`) — 현행 챔피언 (11번째, 2026-08-30 승격) |
| 예산 | **300콜** (`[acquisition] budget = 300`, `wave_size = 8`, `n_waves = 37`, `reserve = 4`) |
| 시드 | **2** (`[flow] random_seed`) |
| 목적 | `objective = "min_fr_max_cycle"`, **`minfr_lambda = 1000.0`** (기본값) |
| 게이트 | `f_r_limit 1.55` / **`cbc_limit 1600.0`** / `f_q_limit 2.41` / `ao_abs_limit 0.30` / `minfr_pin_bu_limit 78.0` |
| 평형 | `[master] max_cycles = 16`, `consecutive = 2`, workers 8 |

### 3.3 콜드스타트 스위치 — 정확한 덱 키

덱은 **TOML** 이고 섹션은 `[flow] [case] [master] [verify] [model] [search]
[search.trust_region] [search.local_search] [acquisition] [produce] [constraints]` 이다.
**`[optimize]` 섹션은 존재하지 않는다** (`optimize` 는 CLI 서브커맨드, `cli.py:1915`).
`config.py:1342-1360` 화이트리스트 밖 키는 `load_config` 가 **하드 에러**로 거부한다.

```toml
[search]
elite_top_k      = 0     # 스토어 elite 부모 0개  (campaign.py:1516 `ranked[:0]`)
elite_seed_cases = []    # 타 케이스 도너 없음    (기본값, campaign.py:1459-1485)
near_miss_f_r    = 0.0   # near-miss 부모 arm 끔  (campaign.py:1544 의 문서화된 off 스위치)
elite_frac       = 0.0   # elite 변이 arm 지분 0
guided_frac      = 0.30
diversity_frac   = 0.70  # heuristic/random 구성으로 강제

[model]
store_dir = "data/store_coldE1E2"   # ★ 홀드아웃 스토어 (아래)
```

부모 리스트가 비면 elite arm 은 통째로 스킵되고(`construct.py:482-576` 의 `while parents and …`
가드), `_guided_prefix`(`construct.py:671-690`)는 `_heuristic_genome`(`construct.py:163`) /
`genome.random_genome` 로 폴백하며, 부족분은 `construct.py:651-666` 의 **순수 random 채움**
(주석: *"if elites/guided starved (wave 0, empty store), fill random"*)이 흡수한다.
**이것이 코드가 이미 갖고 있는 정직한 콜드스타트 경로다.**

**★ `elite_top_k = 0` 만으로는 콜드스타트가 완성되지 않는다.** 스토어는 아래 4곳에서 **여전히
읽힌다** — 그래서 홀드아웃 스토어가 필수다:

| 잔존 스토어 읽기 | 위치 | 영향 |
|---|---|---|
| `TrustRegion.from_store` | `campaign.py:1082` → `acquisition.py:2192` | `in_region` 마스크 = **유일한 하드 필터**(`acquisition.py:602-603`). 셀 인구가 선택 가능 영역을 정한다 |
| `_replay_rows` | `campaign.py:1605` | wave 파인튜닝 리플레이 |
| `_holdout_rows` | `campaign.py:1617` → `_case_store_rows` | **게이트 홀드아웃이 대상 셀 자신의 스토어 행** |
| map 캘리브레이션 / fuel types | `campaign.py:906/937`, `campaign.py:1322` | 보조 |

**홀드아웃 스토어 정의.** `data/store_coldE1E2` = 정본 `records.parquet` 에서
**`case_pair == "E1_E2"` 인 행 전부 제거**한 사본. `_case_store_rows`(`campaign.py:1445-1449`)는
**`case_pair` 로만 필터하고 feed 는 보지 않으므로** 셀 정의를 pair 수준으로 잡는다 —
그렇지 않으면 f109·f113 의 E1_E2 행이 `_morph_feed`(`construct.py:141`)로 f121 에 들어온다.

**추가 등록 주의 2건.**
1. `elite_top_k` 는 **과부하 키**다 — 같은 값이 캠페인 **내부**의 `prev_top` 이월도 막는다
   (`campaign.py:2381`). `elite_top_k=0` 은 정당한 캠페인 내 활용까지 끈다. 본 실험은 코드 변경
   없이 **이 결합을 받아들이고 기록한다**. B 팔에 **불리한** 방향이므로 H1/H3 통과 시 보수적 결론이다.
2. **`explore` 전용 wave 0 키도, 모델을 끄는 키도 없다.** wave 구성은 정수 슬롯
   `exploit/explore/control` 뿐이고(`acquisition.py:2930/2952`), 마지막 reserve wave 는
   `campaign.py:2390-2391` 에서 **강제 all-exploit** 이 된다. `--dry-run` 은 `StubEvaluator` 라
   MASTER 를 안 돌려 대조군으로 무용하다.

### 3.4 C — lpopt 웜스타트 (참조, 기존 데이터, 새 계산 없음)

`fpcamp_199`(200콜, 혼합 목적, 기록 1.4636 을 98콜에) + `fpcamp_minfr_199`(100콜, `min_fr_max_cycle`).

**★ C 는 A·B 와 knob-동일하지 않다.**

| knob | C (`fpcamp_minfr_199.inp`) | A·B 등록값 | 동일? |
|---|---|---|---|
| `minfr_lambda` | **200.0** | 1000.0 | ✗ |
| `cbc_limit` | **1550.0** | 1600.0 | ✗ |
| `model_dir` | **`split_S1b`** (`cond_schema=v6b`) | `s1j` (`v8`) | ✗ |
| `near_miss_f_r` | 1.52 | 0.0 (B) | ✗ |
| `[constraints]` | `mtc_enable=true, post_verify_top_k=5` | B 는 미설정 | ✗ |
| `f_r / f_q / ao` 한계 | 1.55 / 2.41 / 0.30 | 동일 | ✓ |
| 셀·feed·library | E1_E2 / 121 / ga80 | 동일 | ✓ |

따라서 **C 는 K1 표에 참조행으로만 싣고 H1/H3 검정에는 넣지 않는다.** C′(λ=1000·CBC1600·s1j 웜스타트
300콜×2)를 새로 돌리면 검정에 넣을 수 있으나 **본 라운드 범위 밖**(추가 ~29 h)이며, H3 는
**B vs A** 로 충분히 검정되므로 불필요하다고 등록한다.

---

## 4. 목적함수·게이트 동일성 감사

### 4.1 두 스칼라의 실제 형태

**lpopt `min_fr_max_cycle`** (`lpopt/search/acquisition.py:509-548`; spec 생성 `campaign.py:795-804`;
`MinFrSpec` `acquisition.py:429-454`):

```
shift      = risk_z · σ_calibrated                          (acquisition.py:513)
cyclen_LCB = μ_cyclen − shift_cyclen                        (:515)
F_r_UCB    = μ_F_r    + shift_F_r                           (:516)
scalar     = cyclen_LCB − minfr_lambda · F_r_UCB            (:517)
penalty    = Σ_axes [max(0, UCB − limit)/width]²  +  [max(0, pinUCB − 78)]²   (:521-546)
total      = scalar − 1.0e4 · penalty                       (:548)
```
게이트 축과 **폭** `_MINFR_GATED_AXES`(`acquisition.py:414-419`):
`f_r` 0.01 · `cbc` 25.0 · `f_q` 0.05 · `ao_abs` 0.02. 계층 상수 `_MAXCYCLE_CONSTRAINT_TIER = 1.0e4`(`:216`).

**MOCHA (production `sa.method: mocha`)** — `MOCHA/mocha_annealing.py`, `MOCHABoltzmannAnnealer.evaluate()`
`:478-602`:

```
E      = e_obj + e_con                                             (:592)
e_obj  = objective_weight · [ max_i(λ_i·g_i) + ρ·Σ_i(λ_i·g_i) ]    (:550, ρ = tchebycheff_rho = 0.05)
g_i    = max(0, (raw_i − z*_i)/scale_i)                            (_g(), :374-379)
scale_i= |limit| (>1e-12) else 1.0                                 (_scale_for(), :183-198)
         ※ objective.normalization_default 가 있으면 scale_i = sqrt(그 값)   (:184-191)
e_con  = Σ_i β_i · cap·(1 − exp(−v_i²/cap)),  v_i = 위반/scale_i    (_constraint_cost(), :349-358)
         cap = objective.term_cap = 5.0 ; β_i = spec.weight or mocha.constraint_weight_default
수용   α = min(1, exp(−ΔE / max(T, temperature_floor)) · q_ratio)   (:604-644, q_ratio = 1)
냉각   T ← max(T·cooling_alpha, floor)  단일 출처                    (finish_stage(), :696-702)
T0     = −mean(Δ⁺)/ln(χ₀), χ₀ = sa.initial_acceptance = 0.99       (:678-695)
```
실패 클래스는 E 를 **대체**한다(가산 아님): invalid 100.0, 비수렴 50.0, 비유한 metric → invalid.

### 4.2 동일하게 만들 수 있는 knob — **전부 YAML/덱, 코드 변경 0**

| 항목 | lpopt | MOCHA (`config_apr1400.yaml`) | 현재값 → 등록값 |
|---|---|---|---|
| F_r 최소화 | `objective = "min_fr_max_cycle"` | `objective.parameters.fr.is_objective` / `.ratchet` | `false/false` → **`true/true`** (L201). **★ 이 설정은 신규가 아니다 — 20260705·20260702 런이 이미 이 설정이었다**(§0 E-1) |
| F_xy 목적 제거 | (해당 없음) | `objective.parameters.fxy.enable` | `true` → **`false`** (L200) |
| F_r 게이트 1.55 | `f_r_limit` | `fr.limit` | **이미 1.55** ✓ (목적항이면서 `spec.limit≠0` 이면 하드캡 항이 별도 가산됨 — `mocha_annealing.py:519-523`) |
| CBC ≤ 1600 | `cbc_limit` (기본 **1550**) | `boc.limit` | MOCHA **이미 1600** ✓ / **lpopt 덱에 `cbc_limit = 1600.0` 명시 필요** |
| F_q ≤ 2.41 | `f_q_limit` | `fq_pin.limit` | **이미 2.41** ✓ |
| \|AO\| ≤ 0.30 | `ao_abs_limit` | `axial_offset.limit` | **이미 0.30** ✓ |
| **게이트 폭** | 0.01 / 25.0 / 0.05 / 0.02 | `objective.normalization_default` (현재 **부재**, 로더는 지원, `mocha_annealing.py:184-191`) | **추가**: `{fr: 1.0e-4, boc: 625.0, fq_pin: 2.5e-3, axial_offset: 4.0e-4}` → `scale = sqrt(·) = 0.01 / 25.0 / 0.05 / 0.02` — **lpopt 폭과 정확히 일치** |
| soft cap 제거 | (없음) | `objective.term_cap` | `5.0` → **`0`** (lpopt 에 대응물 없음) |
| 게이트 가중 | 전부 1 (Σexcess²) | 항별 `weight` | `axial_offset` 만 0.1 → **1.0** 으로 통일 |
| 제약 어닐링 | (없음) | `sa.schedule_enabled` | **이미 `false`** ✓ → `active_limit == spec.limit`(`_active_limit()` `:292-304`). 켜면 fr 이 1.80 에서 시작해 동일성이 깨진다 |
| pin BU | `minfr_pin_bu_limit = 78.0` | `pin_burnup.limit` | `75.0` → **`78.0`** (`pinbu_definition_20260820.md` 의 공식 마진) |
| feed | `[case] feed = 121` | `core.feed_fa_count` | **이미 121** ✓ |
| e_core | 5.000 | `core.feed_avg_enrichment` / `feed_enrichment_tol` | `5.40/0.05` → **`5.00/0.10`** (정의 동일: 241 FA U-질량 가중 초기농축도 평균) |
| pair 고정 | `[case] mode="fixed", pair="E1_E2"` | `core.fuel_types` | `[]`(자동 12종) → **`['E1','E2']`** — 로스터 자체가 pair 가 되어 `C(2,2)=1`, family swap 이 항등변환 (§4.3 N-6) |
| 콜 카운터 | `budget_spent += 1` / 후보 (`campaign.py:2463`) | `--max-evals` | 동일 정의 ✓ |

### 4.3 동일하게 만들 수 **없는** knob — 전부 명시

| # | 항목 | 왜 못 맞추는가 | 등록된 처분 |
|---|---|---|---|
| **N-1** | **평가 오라클** | MOCHA 는 후보마다 **실제 MASTER 3-D 평형체인**(평균 9.58 MASTER 사이클/평가)을 돈다. lpopt 는 **대리모형 μ·σ** 로 순위를 매기고 UCB/LCB(`risk_z=0.25`)를 쓴다. MOCHA 에는 σ 가 없어 `cyclen_LCB − λ·F_r_UCB` 의 대응물이 **원리적으로 없다** | 무해화 불가. **이것이 측정 대상 자체다.** KPI §2 의 `screen_ratio`(≈312:1)를 B 팔에 병기 |
| **N-2** | **스칼라화 형태** | MOCHA = **이동하는 기준점 z\* 를 갖는 증강 가중 Tchebycheff**(`reference_mode: best_so_far`, `utopia_offset 0.05`, `ideal_robust p05/window 5`, `_commit_reference_snapshot()` `:315-341`) → **J 가 epoch 의존적**이다(같은 metric 이 stage 마다 다른 J). lpopt = **고정 선형** 스칼라. `aggregation: linear` 로 바꾸면 `e_obj = objective_weight·Σ(value/scale)`(`:585-590`)이 되지만 그것은 기준점 기계를 통째로 잃는 **다른 코드 경로**다 | **같게 만들려는 시도를 포기한다.** 대신 양쪽을 *"F_r 을 단독 최소화, 나머지는 Tier-1 게이트"* 라는 **의도 수준**에서 정렬한다. K1/K2/K3 는 **검증된 F_r 값 위에서만** 계산되므로 스칼라 형태에 의존하지 않는다 — 이것이 본 설계가 N-1·N-2 를 견디는 이유다 |
| **N-3** | **λ 의 절대 스케일** | λ=1000 은 "EFPD per unit F_r" 이다. MOCHA 는 단일 목적일 때 `_normalized_weights()`(`:211-223`)가 λ_fr=1.0 으로 강제하고 `e_obj = 3.0·1.05·g_fr` → **실효 F_r 압력 ≈ 2.03 / 단위 F_r**(정규화 에너지 단위). 두 수는 **단위가 달라 비교 불가** | λ 를 문자 그대로 맞추려 하지 않는다. **주기길이의 지위로 맞춘다**(N-4) |
| **N-4** | **주기길이의 지위** | lpopt: cyclen 이 **목적에 가산**. MOCHA: `cyc` 가 **625 EFPD 하한 제약** | MOCHA `cyc.limit = 625.0` **유지**, lpopt λ=1000 **유지**. 대신 **공통 후처리 필터로 `cyclen ≥ 625` 를 양쪽에 동일 적용**하여 K1 계산에서 같은 조건을 만든다. 이것이 N-4 를 무해화하는 등록된 규칙이다 |
| **N-5** | **게이트가 하드가 아니다(양쪽 다)** | lpopt: 4 게이트는 **연질 페널티**(`total = scalar − 1e4·penalty`, `:548`), 진짜 하드 필터는 **trust region** 뿐(`:602-603`). MOCHA: 위반은 e_con 가산일 뿐 기각이 아니다. 게다가 MOCHA 의 F_r 게이트는 **사후**(MASTER 전에는 F_r 을 모른다)라 평가를 절약하지도 못한다 | **양쪽 다 "연질"임을 명기**하고 마크는 **사후 하드 필터**로 판정한다(§5) |
| **N-6** | **pair 고정 knob 부재** | MOCHA 에 `pair`/`case`/`cell` 키가 없다. `sa.type_move_family_floor: 0.15` 는 **0 이 될 수 없는 하한**이라 family swap 이 항상 살아있다 | `core.fuel_types: ['E1','E2']` 로 **로스터를 pair 로 축소** → 구성적으로 고정. knob·코드 변경 불필요 |
| **N-7** | **평형 수렴 판정 기준** | lpopt: **5-FOM** 연속창(`cyclen 0.10 EFPD, CBC 1.0 ppm, F_q/F_r/AO 각 1e-3`; `lpopt/vendor/masterrl/equilibrium.py:41-57`), `consecutive=2`, `max_cycles=16`. MOCHA: **EFPD 1축만**(`equilibrium.eps_efpd: 1.0`), `eps_consecutive: 5`, `min_cycles: 3`, `max_cycles: 15` → **같은 LP 도 다른 평형상태를 낸다** | **lpopt 의 5-FOM 을 공통 정의로 삼는다**(셀 기록 1.4636 이 그 기준으로 측정됐다). MOCHA 를 바꾸는 것은 코드 변경(§8 P-3, **하지 않는다**). 대안: MOCHA 는 자기 기준으로 탐색하고, **A 팔의 running-best 패턴만 lpopt 검증기로 재평가**하여 마크를 계산한다 — **K1/K2/K3 는 전부 이 재평가 값 위에서** 산출된다(재평가 ~40콜, 예산 외 별도 계상) |
| **N-8** | **탐색 알고리즘·이동 연산자** | MOCHA = 동기 병렬 Metropolis SA(300 stage × 32 sample), 이동은 `swap_burned_sources .38 / swap_fresh_burned .26 / change_fresh_type .28 / compound_shuffle .08`(`optimizer.py:870-895`). lpopt = 대리모형 획득함수 순위 + `n_moves_early 2 / n_moves_late 5` 변이·빔롤아웃 | 무해화 불가. **비교의 대상 자체**로 인정하고 기록만 한다 |
| **N-9** | **잔여 오염 — 챔피언 s1j 는 이 셀을 학습했다** | 어떤 덱 키로도 모델을 셀-무지하게 만들 수 없다 | **정직하게 등록하고 남긴다.** B 는 "스토어를 안 보는 콜드스타트"이지 "이 셀을 처음 보는 콜드스타트"가 아니다. 완전 무지 팔은 **E1_E2 홀드아웃 재학습 체크포인트**가 필요하며 본 라운드 밖이다(§12) |
| **N-10** | **코어 모델 덱** | 둘 다 `master4.0m4_r1.exe` 를 쓰고 둘 다 quarter-core(`plant_id APRQ`, `nz 27`, `power_mw 3983`)지만, MOCHA 는 `master_io.py:362-453` 의 파이썬 f-string 으로 `MAS_INP` 를 생성하고 lpopt 는 kit 의 자체 생성기를 쓴다. **덱이 같다는 증거가 없다** | **G-0 교차검증(§9)이 본 실험의 전제조건이다.** FAIL 이면 발사하지 않는다 |
| **N-11** | **실패 의미론** | MOCHA 는 E 를 50(비수렴)/100(invalid)로 **대체**한다. lpopt 대리모형에는 비수렴 클래스가 없다 | 무해화 불가. 오류 콜은 예산에서 차감하되 K1 분모에서 제외하고 명기 |

**한 줄 요약.** *동일하게 만들 수 있는 것 = 셀(pair·feed·e_core·library) · 게이트 한계와 폭 ·
목적의 방향 · 콜 카운터 — **전부 설정 파일에서**. 동일하게 만들 수 없는 것 = 평가 오라클(N-1) ·
스칼라화 형태(N-2) · 탐색 알고리즘(N-8) · 평형 판정(N-7, 재평가로 우회) · 코어 덱(N-10, G-0 로 검증).*

### 4.4 남는 비대칭 — 반드시 병기

1. **A 는 `R_cell` 을 모른다.** 1.4636 은 lpopt 계보가 만든 값이다. A 가 못 넘어도
   "SA 가 못 한다"가 아니라 "SA 가 1,000콜로는 못 한다"만 말한다.
2. **B 의 챔피언은 오염돼 있다** (N-9).
3. **A 의 SA 튜닝은 이 목적에 최적화된 것이 아니다.** 현행 냉각/스테이지 설정을 그대로 쓴다.
   본 실험은 "튜닝된 SA"가 아니라 **"현행 MOCHA"** 를 잰다.
4. **B 의 `elite_top_k=0` 은 캠페인 내 이월도 끈다** (§3.3 주의 1) — B 에 불리한 방향.

---

## 5. 등록 마크 (K1 / K2 / K3)

카운터는 **MASTER 평형체인 1회 = 1콜**. 대리모형 평가·스크리닝은 계수하지 않고 K4 로 보고한다.
모든 마크는 **N-7 의 공통 재평가 값** 위에서, **공통 사후 하드 필터**
`F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ |AO| ≤ 0.30 ∧ cyclen ≥ 625` 를 통과한 행에 대해서만 계산한다.

**K1 — calls-to-record@ε (주 마크)**
> 누적 MASTER 콜 n 시점의 필터 통과 best F_r 이 `R_cell + ε = 1.4636 + ε` 이하가 되는 최소 n.
> 미달 시 `>N` 으로 보고하고 최종 gap 병기.

| ε | 값 | 지위 |
|---|---|---|
| 0.005 | ≤ 1.4686 | KPI 문서 연속성용. **★ 라벨 잡음 σ=0.00595 보다 작다** — 단일 교차는 잡음일 수 있다 |
| **0.010** | ≤ 1.4736 | **잡음-강건 주 마크**(≈1.7σ). **H1/H2/H3 의 검정은 이 값으로 한다** |
| 0.020 | ≤ 1.4836 | 보조(초기 궤적 형태) |
| 0.000 | ≤ 1.4636 | 기록 도달. 보고만 |

**도달 인정 규칙(단발 튐 배제).** K1 이 성립한 콜 이후 **연속 3콜 이상** best 가 문턱 아래를
유지해야 도달로 인정한다.

**K2 — calls-to-incumbent**
> A·B 는 콜드스타트이므로 incumbent = **자기 wave 0 / stage 0 best**
> (= "언제부터 새 정보를 만들기 시작하는가"). C 는 정의상 스토어 incumbent 1.4636.
> 보조로 `F_xy` incumbent **1.5295** 대비 K2 를 **부수 관찰**로만 보고한다(headline 금지 —
> 두 팔 모두 F_xy 는 목적이 아니다).

**K3 — AUF@100 / AUF@300**
> `AUF@N = (1/N) Σ_{n=1..N} (best_n − R_cell)`. 낮을수록 좋다. **팔 간 비교는 반드시 같은 N**
> → 주 비교는 **AUF@300**. A 는 추가로 `AUF@1000` 을 기록한다.

**K4 — 필수 병기** (KPI §5 그대로)
`screen_ratio`(대리평가/MASTER콜), `prior_rows`, `first_feasible_call`, `n_feasible/N`,
`Δ/100calls`(마지막 두 구간), 그리고 objective/게이트/library 셀 지문.
**wall 은 K1–K3 어디에도 들어가지 않는다**(§7 계측 공백).

---

## 6. 검정력과 판정 규칙 — 결과 전에 고정

**설계.** 시드를 팔 사이에 **짝짓는다**: A-s1 ↔ B-s1, A-s2 ↔ B-s2. n=2 쌍은 비모수 검정에
턱없이 부족하다 — **본 실험은 p-value 를 보고하지 않는다.** 사전 등록된 **효과크기 문턱**으로만
판정한다.

**H1 판정:**
> `K1@0.010(A)/K1@0.010(B) ≥ 3` 이 **2/2 시드쌍**에서 성립 → H1 **지지**.
> 1/2 → **미결**, 등록된 다음 수는 시드 2개 추가(각 팔 +1,000/+300콜).
> 0/2 → H1 **기각**.
> A 미도달(`>1000`) + B 도달이면 비율 `>3.3` 으로 자동 성립.

**H3 판정:**
> `K1@0.010(B) ≤ 300` 이 **2/2 시드**에서 성립 → H3 지지.
> 0/2 → H3 기각: "직관"은 사전 데이터에 의존한다. **발생 시 KPI 문서의 헤드라인 문구를
> 그 자리에서 수정한다**(§11 D4).

**잡음 기저와 최소 검출 효과.** F_r 라벨 잡음 σ=0.00595. ε=0.010 문턱의 단일 교차 오경보는
정규 근사로 ≈ P(Z>1.68) ≈ 0.047(한쪽)이며 위의 연속-3콜 규칙이 이를 더 낮춘다.
**두 팔의 K1 이 2배 이내로 다르면 n=2 로는 구분 불가하다. 그래서 문턱을 3배로 잡았다 —
그보다 작은 차이는 본 실험이 말할 수 없고, 말하지 않는다.**

---

## 7. 예산 · 호스트 계획 · 런타임

**호스트: 199 단독.** 181 / 198 / 238 **미사용**.
**슬라이스 Z 파이프라인(S4→S9c, `slice_Z_pipeline_runbook_20260903.md`) 이 전부 종결된 뒤 순차
실행.** idle 게이트로 발사하며 바쁘면 쌓지 않고 거부한다.
**★ 슬라이스 Z 는 199 의 `data/design/package` 라이브러리를 재빌드한다(런북 §2).** 본 실험이 쓰는
것은 그 패키지가 아니라 **`FEASIBLE_PACKAGE`(ga80)** 이므로 원칙적으로 영향이 없으나,
`FEASIBLE_PACKAGE\lib\{MAS_XSL,MAS_HFF}` 의 해시를 **슬라이스 Z 전후 모두** §10 에 기록하고
바뀌었으면 **G-0 를 재실행**한다.

**측정된 처리량 — 추정이 아니라 실측:**

| 원천 | 값 |
|---|---|
| MOCHA 콜당 레인시간, 2_LP 호스트, 7,360 평가 | 평균 `eval_wall_s` **248.11 s**, 중앙 247.52, 최대 407.66 (`sa_log.csv`) |
| 같은 런 벽시계 | 71.06 h / 7,360 평가, 8 워커 → **34.7 s/콜** (`run_meta.timing`, `eval_per_hour 103.6`) |
| MOCHA 콜당 레인시간, **199**, 260624 케이스 | **502–508 s** (`D:\eqsur_data_199_20260813\runs\runs\20260813-081755_ga-basin_b00-r00_s538697511\sa_log.csv`) |
| MOCHA 199 전체 런, `--workers 12 --max-evals 600` | 05:25:23 → 15:04:32 KST = **9.65 h** (`status.json`) |
| lpopt `fpcamp_minfxy_e1e2_f121_r2`, 100콜, workers 8, **199** | run dir 생성 13:28:08 → `report.md` 18:15:08 = **4.78 h → 172 s/콜** |

199 의 MOCHA 레인시간(505 s)이 2_LP 호스트(248 s)의 2배인 것은 **미해명**이며(코어 고정·경쟁
프로세스·케이스 차이 후보), 보수적으로 **199 값을 쓴다**.

| 팔 | 콜 | 시드 | 총 콜 | 워커 | 추정 wall | 밴드 |
|---|---:|---:|---:|---:|---:|---|
| A (MOCHA SA) | 1,000 | 2 | 2,000 | 12 | **~28 h** | 22–40 h |
| B (lpopt cold) | 300 | 2 | 600 | 8 | **~29 h** | 26–34 h |
| A 마크 재평가 (N-7) | — | — | ~40 | 8 | ~2 h | 1–3 h |
| G-0 교차검증 (§9) | — | — | 20 × 2 하네스 | 8/12 | ~3 h | 2–5 h |
| G-1 MOCHA 커미셔닝 | 40 | 1 | 40 | 12 | ~0.6 h | — |
| **합계** | | | **~2,700 콜** | | **~63 h** | 52–83 h |

**≈ 2.6 일 199 전용.** A 근거: `1000 × 505 s ÷ 12 워커 × 1.3(스테이지 충전 손실) = 15.2 h/시드`.
B 근거: `300 × 172 s = 14.3 h/시드`.

**하드 펜스(런처에 박는다):** A `--max-evals 1000`(시드당), B `[acquisition] budget = 300`.
**중간 증액 금지** — 예산 초과는 새 결정이고 새 문서다.

**★ 등록된 계측 비대칭.** MOCHA 는 `sa_log.csv` 에 **콜마다 `eval_wall_s`** 를 남긴다
(31개 열: `stage, eval, tag, move, status, J, J_cur, J_con, J_obj, accepted, T, eval_wall_s,
MOCHA_T, efpd, cyc_bu, boc_ppm, eoc_ppm, max_fqp, max_frp, max_frn, max_abs_ao, discharge_bu,
max_pin_bu, max_asm_bu, n_cycles, aggregation, g_fr, zstar_fr, MOCHA_delta_E, MOCHA_alpha,
infra_error, detail`; `optimizer.py:3546-3552`). lpopt 의 `optimize` 경로는 **콜별 wall 을
저장하지 않는다** — `WaveOutcome.wall_s` 는 `verify.py:1129/1144` 에서 계산되지만
`labels.jsonl`(`campaign.py:2478-2487`)에도 `status.json`(`campaign.py:1887-1917`)에도 없다
(`produce` 경로만 `produce.py:1238` 에 저장). **처분: B 의 wall 은 wave 단위
(`status.json.updated` 델타, 8콜 입도)로만 보고하고, wall 은 어떤 마크에도 쓰지 않는다.**

---

## 8. 이식 태스크 — 199 kit 위에서 MOCHA 를 ga80 로 돌리려면

**결론: 호환된다. 새 물리 자산(DeCART 재계산)은 필요 없다.** 근거는 두 하네스가 **같은 실행파일과
같은 라이브러리 포맷**을 쓴다는 199 실측이다.

| 확인 항목 | MOCHA 케이스 (`D:\LEUP\1_Calculation\0_APR1400\260624`) | lpopt ga80 (`kit_frontier\FEASIBLE_PACKAGE`) | 판정 |
|---|---|---|---|
| MASTER 실행파일 | `master4.0m4_r1.exe` | 동일 (`C:\DeCART_MASTER\BIN\master4.0m4_r1.exe`, 덱 `[master] executable`) | **동일** |
| 라이브러리 구성 | `hgc\{FA_B1…C6.HGC(12), MAS_XSL 4,632,198, MAS_HFF 4,858,284, MAS_REF 2,008, prolog41m4.exe, TotalBatcher4.exe}` | `hgc\{FA_A2…N6.HGC(36)}` + `lib\{MAS_XSL 30,869,930, MAS_HFF 32,388,560}` | **동일 3종 구성**, 크기만 타입 수 차이 |
| HGC 파일 크기 | 7,395,955 B | 7,395,955 B (`FA_E1.HGC`, `FA_E2.HGC` 실재 확인) | **동일 포맷** |
| MASTER 가 소비하는 자산 | `MAS_XSL`/`MAS_HFF` 를 후보 workdir 로 **복사**(`master_run.py:228-232`), 덱은 `MAS_INP` 로 생성(`:259`), `%JOB_TYP` 은 **파일명만** 참조(`master_io.py:367-368`) | lpopt 도 동일하게 `MAS_XSL` 30,869,930 B 를 `master\master_work\worker_NN\…` 로 복사 (199 실측) | **동일 규약** |
| 코어 사양 | `feed_fa_count 121`, 241 FA, quarter 61 셀, 2-type | `[case] feed 121`, pair 2-type | **동일** |
| MOCHA 소재 (199) | `C:\Users\USER\lpopt_work\eqsur_data_199_20260813\repo\{MOCHA.py, MOCHA\, bin\master4.0m4_r1.exe}` — **2026-08-13 완주 이력**(exit 0) | — | **이미 있다** |
| Python | 199 `3.11.9`; 기존 런은 `C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe` | — | 재사용 |

**포팅 작업 (P-1 … P-5):**

- **P-1 (필수, 저비용).** ga80 MOCHA 케이스 디렉터리 조립:
  `D:\LEUP\1_Calculation\0_APR1400\ga80\hgc\` ←
  `FEASIBLE_PACKAGE\lib\{MAS_XSL, MAS_HFF}` + `FEASIBLE_PACKAGE\hgc\{FA_E1.HGC, FA_E2.HGC}`
  + `260624\hgc\{MAS_REF, prolog41m4.exe, TotalBatcher4.exe}`.
  **36-타입 통합 `MAS_XSL` 을 그대로 쓴다 — 재배치(TotalBatcher) 불필요.** MASTER 는 덱의
  `%JOB_TYP` 이 지목한 `FA_<type>` 만 꺼내 쓰고, pair 고정은 `core.fuel_types: ['E1','E2']` 로
  한다(N-6). **첫 게이트: `FA_E1`/`FA_E2` 가 이 `MAS_XSL` 안에 배치돼 있는지 확인**(없으면 P-1b).
- **P-1b (조건부).** 위 확인이 실패하면 `library.build_library()`(`MOCHA/library.py:68+`)로
  `FA_E1.HGC`+`FA_E2.HGC`+`MAS_REF` 만으로 2-타입 `MAS_XSL`/`MAS_HFF` 를 **재배치**한다.
  **DeCART 재계산은 여전히 불필요**(HGC 가 이미 있다). 199 필수(Windows 전용 exe), ~수 분.
- **P-2 (필수).** `config_apr1400.yaml` 패치 — 199 에 이미 있는 `patch_data_199_config.py` 와
  동일한 방식:
  - `paths.master_exe`, `paths.case_dir → …\0_APR1400\ga80`, `paths.run_dir → D:\se_control_20260903\runs`
    (**★ 정본 YAML 의 `case_dir`/`run_dir` 은 사라진 `2026\1_Calculation\…` 트리를 가리키는 死경로다.
    199 패치에서 반드시 덮어쓴다**)
  - `core.feed_avg_enrichment: 5.00`, `core.feed_enrichment_tol: 0.10`, `core.fuel_types: ['E1','E2']`
  - `core.fuel_data: {E1: {...}, E2: {...}}` — `fa_dir/FA_*.out` 가 없으므로 **수동 fallback**
    (`config_apr1400.yaml:104` 가 지원). `u_mass_g`·`enrichment` 는 lpopt 의
    `config/fuel_types_dbx_extracted.yaml` / `data/store/fuel_types.parquet` 에서 가져온다
  - `objective.parameters` §4.2 표대로 (fr 목적화, fxy off, pin_burnup 78, 폭 `normalization_default`,
    `term_cap: 0`, `axial_offset.weight: 1.0`)
  - `sa.global_exploration.enabled: false`, `sa.parallel_workers: 12`, `sa.pcore_cpus: 0..23`
- **P-3 (하지 않는 것을 등록).** MOCHA 평형 판정을 lpopt 5-FOM 으로 바꾸는 코드 변경.
  **본 라운드는 하지 않는다.** N-7 의 재평가 우회를 쓴다.
- **P-4 (필수, 저비용).** B 팔 홀드아웃 스토어 `data/store_coldE1E2` 생성 —
  정본 `records.parquet` 에서 `case_pair == "E1_E2"` 행 제거. **238 에서 수행**하고 199 로 전송.
- **P-5 (필수).** 런처·상태 스크립트 3종
  (`launch_se_control_A_199.ps1`, `launch_se_control_B_199.ps1`, `status_se_control_199.ps1`) —
  하우스 관행대로 idle 게이트·덱해시 게이트·`--max-evals` 하드펜스 포함.

**현재까지 발견된 비호환은 없다.** 미확인 잔여 리스크는 두 가지뿐이며 둘 다 P-1 의 첫 게이트에서
드러난다: (i) 통합 `MAS_XSL` 의 `FA_E1`/`FA_E2` 포함 여부, (ii) `core.fuel_data` 수동 입력이
`screening.feed_spec_screen()`(`MOCHA/screening.py:379-428`)의 U-질량 가중 평균 계산에 충분한지.

---

## 9. 실행 전 게이트 — 전부 PASS 해야 발사한다

| 게이트 | 내용 | 기준 | FAIL 시 |
|---|---|---|---|
| **G-0 (교차검증, 전제조건)** | 이 셀의 기존 검증 코어 **10개**(F_r 1.46–1.55)를 A·B 두 하네스로 각각 재평가 | 모든 코어에서 **\|ΔF_r\| ≤ 0.00595**(라벨 잡음 1σ), \|ΔCBC\| ≤ 5 ppm, \|Δcyclen\| ≤ 1 EFPD | **발사 중단.** 두 하네스의 코어 덱이 다르다는 뜻이며 비교가 성립하지 않는다. 그리고 그 사실 자체를 KPI 문서 §4 "성립하지 않음" 목록에 **6번 항목으로 추가**한다 |
| **G-1 (MOCHA 커미셔닝)** | A 설정으로 40콜 단축 런 1회 | 40콜 안에 `F_r ≤ 1.55` feasible ≥ 1, 수용률 0.2–0.8, 비수렴 < 10%, `screened` 비율 < 20% | 냉각/스테이지 재튜닝 후 재시도. **재튜닝 비용은 본 예산 밖** |
| **G-2 (B 콜드스타트 실증)** | B 덱 `--dry-run` 후 wave 0 후보 출처 감사 | wave 0 후보 8개 중 스토어 유래 부모 **0개**; `store_coldE1E2` 의 `case_pair=="E1_E2"` 행 **0행** | 덱 수정 |
| **G-3 (199 유휴)** | master / python / decart 프로세스 0, 슬라이스 Z S9c 종결 | 확인 (2026-09-03 현재 0/0/0) | 대기 |
| **G-4 (해시)** | §10 스탬프 기입, 로컬=199 바이트 일치 | 확인 | 재전송 |

---

## 10. 스탬프 — 발사 시점에 기입 (지금 채우지 않는다)

| 항목 | 값 |
|---|---|
| A `config_apr1400.yaml`(패치 후) sha256 | `— 발사 시 기입` |
| A 패치 스크립트 `patch_se_control_199.py` sha256 | `—` |
| A 런처 `launch_se_control_A_199.ps1` sha256 | `—` |
| A 시드 (`--sa-seed` / `--rule-seed`) s1 / s2 | `—` / `—` |
| B 덱 `se_control_B_E1E2_f121_199.inp` sha256 | `—` |
| B `[flow] random_seed` s1 / s2 | `—` / `—` |
| 정본 스토어 `records.parquet` sha256 / bytes | `16E311AF4465E735B38DAF7ABF999268FAC27946C1C5CC279114607D9EE917BA` / `22,810,322` (2026-09-03 로컬=199 일치, **슬라이스 Z 이전 값 — 발사 시 재측정**) |
| B 홀드아웃 `store_coldE1E2` sha256 / 제거 행수 | `—` |
| ga80 케이스 `…\0_APR1400\ga80\hgc\*` 파일별 sha256 | `—` |
| `FEASIBLE_PACKAGE\lib\{MAS_XSL,MAS_HFF}` sha256 (슬라이스 Z **전/후** 명시) | `—` (크기 30,869,930 / 32,388,560) |
| `FA_E1.HGC` / `FA_E2.HGC` sha256 | `—` (각 7,395,955 B) |
| 모델 `data/models/s1j` (cond_schema v8) + 승격 게이트 산출물 | `—` |
| `R_cell` 동결값 | **1.4636** (이동 금지) |
| G-0 결과 (10패턴 max\|ΔF_r\| / max\|ΔCBC\| / max\|Δcyclen\|) | `—` |
| G-1 결과 (feasible 수 / 수용률 / 비수렴률 / screened 율) | `—` |
| A 발사 / 종료 KST (시드별) | `—` |
| B 발사 / 종료 KST (시드별) | `—` |

---

## 11. 처분표 — 각 결과가 "적은 실제 계산" 주장에 대해 무엇을 뜻하는가

| # | 관측 | 해석 | 등록된 처분 (주장 문구) |
|---|---|---|---|
| **D1** | `K1(A) = >1000` 이고 `K1(B) ≤ 300`, 2/2 | **주장 성립.** 동일 셀·동일 게이트에서 SA 는 1,000콜로 못 간 곳에 프론티어 AI 는 ≤300콜로 간다 | 대외 문구를 **실측으로 교체**: *"동일 셀·동일 게이트·동일 MASTER 카운터에서 SA 콜드스타트는 1,000콜 안에 기록+0.010 에 도달하지 못했고, 학습 기반 탐색은 콜드스타트로 N콜에 도달했다."* **"10,000–30,000 iteration" 표현 폐기**(실측이 아니다). **"2,357 wall-hour" 도 폐기**(§0 E-2) |
| **D2** | `K1(A)/K1(B) ≥ 3` 이나 A 도 예산 내 도달, 2/2 | **주장 약화 성립.** 자릿수가 아니라 배수다 | 문구를 **배수로** 낮춘다. *"SA 는 못 한다"* 는 삭제 |
| **D3** | `1 < K1(A)/K1(B) < 3` | **미결.** 본 실험이 구분할 수 없는 구간(§6) | 시드 2개 추가가 등록된 다음 수. **그 전에는 어떤 배수도 인용하지 않는다** |
| **D4** | `K1(B) > 300` (B 미도달) 이나 `K1(C) = 41` | **H3 기각.** 표본효율의 원천은 학습된 정책이 아니라 **셀당 590행의 사전 데이터**다 | KPI §5 K5 문구를 **수정**: *"…이는 셀당 수백 행의 사전 데이터 위에서만 성립하며, 그 데이터 없이는 성립하지 않는다."* 등록된 후속: **사전 데이터 축적 비용을 KPI 에 계상하는 회계 개정** |
| **D5** | `K1(A) ≤ K1(B)` | **주장 기각.** | 헤드라인 즉시 철회. 등록된 후속: MOCHA 를 프론티어 파이프라인의 **연산자로 편입** 검토(KPI §6.4 의 move-set 문제와 같은 계열) |
| **D6** | 세 팔 모두 ≤100콜 (H4) | 셀이 쉬웠다 | 본 셀 대조군 **폐기**, N1_N2/f113 으로 이전. "이 셀에서는 구분 불가"로만 보고 |
| **D7** | **G-0 FAIL** | 두 하네스의 코어 덱이 다르다 | **실험 전체 중단.** KPI §4 "성립하지 않음"에 **6번 항목(코어 모델 불일치)** 추가 — 그것만으로도 문서 가치가 있다 |
| **D8** | A 팔 비수렴/인프라 오류 > 10% | MOCHA 가 ga80 위에서 불안정 | 원인 규명 후 재발사. 오류 콜은 예산에서 차감하되 **K1 분모에서 제외**하고 명기 |
| **D9** | A 가 `F_r < 1.4636` (기록 경신) | SA 가 lpopt 계보의 바닥을 뚫었다 | **KPI §3 의 "E1_E2/f121 완전 포화" 판정이 틀렸다는 뜻.** 셀 기록을 갱신하고, 포화 판정 방법론(Δ/100콜 < 0.002)을 재검토 대상으로 등록 |

---

## 12. 이 라운드가 결론지을 수 없는 것

1. **"프론티어 AI 가 최적해를 찾는다"** — 본 실험은 **기록 근방 도달 속도**만 잰다.
   1.4636 이 이 셀의 전역최적이라는 증거는 어디에도 없다(D9 참조).
2. **다른 셀로의 일반화** — 셀 1개, 시드 2개. KPI 문서의 8셀 통계를 대체하지 않는다.
3. **완전 무지 콜드스타트** — 챔피언 s1j 는 E1_E2 를 학습했다(N-9). 진짜 답은
   **E1_E2 홀드아웃 재학습 체크포인트**가 필요하며 이것이 등록된 후속 1순위다.
4. **wall-clock 비용 비교** — B 팔의 콜별 wall 이 기록되지 않는다(§7).
   비용 비교를 하려면 `campaign.py:2478` 에 `wall_s`/`n_cycles`/`core_class` 를 추가하는 패치가
   먼저다. 본 라운드는 wall 을 마크로 쓰지 않는다.
5. **SA 튜닝의 최적성** — 현행 MOCHA 설정을 그대로 쓴다. "튜닝된 SA"가 아니라
   **"현행 MOCHA"** 를 잰다.
6. **surrogate 선별의 물리적 정당성** — B 의 MASTER 콜당 312회 대리 심사가 *옳은* 심사인지는
   본 실험이 묻지 않는다. `screen_ratio` 를 병기할 뿐이다.

---

### 산출 예정물

- 결과 문서: `data/reports/sample_efficiency_control_results_2026MMDD.md`
- A 궤적: `D:\se_control_20260903\runs\runs\*\{sa_log.csv, run_meta.json, best.json}`
  (콜별 `eval_wall_s`·`n_cycles` 포함)
- B 궤적: `runs/se_control_B_s{1,2}/{labels.jsonl, status.json, state.json}`
- A 마크 재평가 (N-7): `runs/se_control_A_reverify/labels.jsonl`
- G-0 교차검증표: `data/reports/se_control_G0_crosscheck_2026MMDD.csv`

---

## 부록 A. 준비 스탬프 — 2026-09-03 (append-only, 발사 아님)

**본 부록은 §10 스탬프가 아니다.** §10 은 발사 시점에 채운다. 여기 적는 것은 **준비물의 지문**이다.
이 시점까지 **실행된 노심계산은 0** 이다 — MASTER 0 / DeCART 0 / MOCHA 0, 199 에 대한 쓰기 0,
238 은 콜드 스토어 생성(pandas/numpy)과 **StubEvaluator 드라이런**만.

### A-1. P-4 콜드 스토어 (238 에서 생성, 로컬로 전송 완료)

| 항목 | 값 |
|---|---|
| 원본 `~/lpopt_ws/data/store/records.parquet` | **76,793 행** / 22,810,322 B / sha256 `16E311AF4465E735B38DAF7ABF999268FAC27946C1C5CC279114607D9EE917BA` (§10 의 정본 값과 일치) |
| 제거된 `case_pair == "E1_E2"` 행 | **4,016 행** — **전부 `library_id = ga80`**. feed 별: 121:1,486 · 117:636 · 125:592 · 109:552 · 101:232 · 113:180 · 133:156 · 141:110 · 129:28 · 105:24 · 137:20 |
| 콜드 스토어 행수 | **72,777 행** / 21,546,371 B |
| 콜드 `records.parquet` sha256 | **`4EC1181D99B11E2DF3E73C9FBE7493E3E585401022EC82DA3EE5CDA844E90745`** |
| `maps.npz` | 키가 `record_id` 다(`store.py:709` `write_maps`, `:779` `maps()`) → 제거 가능. **75,634 → 73,383 키** (2,251 제거), 209,900,557 B, sha256 `388527D64F38F91F810DABCB0FAEAB11CFE126B584F9D09F2945F4C622B76FE2`. 200 MB 라 로컬로는 전송하지 않았다 — **199 로는 238 에서 직접 보낸다** |
| 부속 파일 | `fuel_types.parquet`(64,343 B) · `flat_scale.json`(8,726 B) · `map_calibration.json`(4,294 B) 를 그대로 복사. 전부 store-local 이고 pair 별 행이 없다 |
| 238 위치 | `~/lpopt_ws/scratch/store_coldE1E2/` (+ `_build_manifest.json`, 생성 스크립트 `~/lpopt_ws/scratch/build_cold.py`) |
| 로컬 위치 | `5_RL/data/store_coldE1E2/records.parquet` (전송 후 sha 재확인 일치) |

> **행수 주의.** §1 은 이 셀의 스토어 인구를 **590 행**(2026-08-29, E1_E2/f121)으로 적는다.
> 제거 대상은 그것이 아니라 **모든 feed 의 E1_E2 행 4,016** 이다 — §3.3 이 등록한 대로
> `_case_store_rows` 가 pair 로만 필터하기 때문이다. 두 수는 모순이 아니라 서로 다른 집합이다.

### A-2. 준비된 파일과 지문 (전부 미실행)

| 파일 | bytes | sha256 |
|---|---:|---|
| `data/design/sa_control/config_sa_control_E1E2_f121.yaml` (P-2) | 14,606 | `A4C56F33010369B9C66FB76B03A3FEDD6532056ADA32ED6567F7FD7C26E75075` |
| `stage_sa_control_199.ps1` (P-1) | 11,326 | `AD4EB4F74A72EB69D89C654F3631D9415D0E15E371C4451A6A64A00A397CF68C` |
| `fpcamp_minfr_E1E2_f121_cold_s1_199.inp` (B, seed 90301) | 10,959 | `C29C28B732F538B01B095AD137C29EC38596D747F9CE59766C8C7018F217A260` |
| `fpcamp_minfr_E1E2_f121_cold_s2_199.inp` (B, seed 90302) | 10,959 | `B25898C32EEDE82131DD66925E7FFDC40E6716D8D723E29ABF7ADF25BC26A845` |
| `launch_fpcamp_minfr_E1E2_f121_cold_199.ps1` | 14,334 | `5A8858463F72319D3D899004AE01B4C0CA8E90398BFE1FEED9A3D87799C74291` |
| `run_fpcamp_minfr_E1E2_f121_cold_199.bat` | 1,980 | `1ABFEC983E3EDEBD148D0049F84EB36E4D7A5BED9FBBE37107E977C6BB4E8514` |
| `status_fpcamp_minfr_E1E2_f121_cold_199.ps1` | 5,510 | `DB2DA07DB7DBF9E29C829386C2A86BB3E32AD7B9D9536DCE9A4BA29CA8369933` |
| `run_sa_control_A_199.ps1` (P-5 골격) | 10,729 | `F5DF708A04AE829AE11041398CA67274261F0BD17AC300B9978B8D4667E4FE3F` |
| 파생 원본 `2_LP/MOCHA/config_apr1400.yaml` | 39,158 | `F043DE6578E8F900C9F2F94FDFD74D38312AD1A6F0BDCE40158962DBB8606050` |

**명명 주의.** §8 P-5 는 `launch_se_control_{A,B}_199.ps1` / `se_control_B_*.inp` 를 적었다.
실제 산출물은 하우스 관행(`fpcamp_*` / `run_*` / `status_*`)에 맞춰
`fpcamp_minfr_E1E2_f121_cold_s{1,2}_199.inp` · `launch|run|status_fpcamp_minfr_E1E2_f121_cold_199.*` ·
`run_sa_control_A_199.ps1` 로 지었다. **§8 의 이름을 이 이름으로 읽는다.**

### A-3. P-1 스테이징 목록 (199 실측 크기, 2026-09-03 읽기전용 조회)

행선지는 **새 디렉터리** `C:\Users\USER\lpopt_work\sa_control\` 하나뿐이다. kit 의
`data / design / package / lib / bases / cores / store` 는 **읽기만** 한다.

| 원본 (199) | → sa_control 내 위치 | bytes |
|---|---|---:|
| `kit_frontier\FEASIBLE_PACKAGE\hgc\FA_E1.HGC` | `case\ga80\hgc\` | 7,395,955 |
| `kit_frontier\FEASIBLE_PACKAGE\hgc\FA_E2.HGC` | `case\ga80\hgc\` | 7,395,955 |
| `kit_frontier\FEASIBLE_PACKAGE\lib\MAS_XSL` (36 세트 통합) | `case\ga80\hgc\` | 30,869,930 |
| `kit_frontier\FEASIBLE_PACKAGE\lib\MAS_HFF` | `case\ga80\hgc\` | 32,388,560 |
| `D:\LEUP\1_Calculation\0_APR1400\260624\hgc\MAS_REF` | `case\ga80\hgc\` | 2,008 |
| 〃 `prolog41m4.exe` | `case\ga80\hgc\` | 1,451,520 |
| 〃 `TotalBatcher4.exe` | `case\ga80\hgc\` | 300,544 |
| `C:\DeCART_MASTER\BIN\master4.0m4_r1.exe` | `bin\` | 3,832,320 |
| `eqsur_data_199_20260813\repo\` 전체 (158 파일) | `repo\` | 21,644,532 |
| ├ `MOCHA.py` | | 772,998 |
| ├ `MOCHA\` (27 파일) | | 786,303 |
| ├ `bin\master4.0m4_r1.exe` | | 3,832,320 |
| └ `terminal_advisory_seed43.json` (GA advisory; GA off 이므로 무용, 이력용) | | 6,187,945 |
| **기본 합계** | | **105,281,324** (~100.4 MiB) |
| *(선택)* `FEASIBLE_PACKAGE\bases\E1_E2\MAS_RST.APRQ_11_0635.19` | `case\ga80\bases\E1_E2\` | 6,612,464 |

C: 여유 41.6 GiB, D: 여유 6,213 GiB (실측). **선택 항목은 기본으로 복사하지 않는다** —
A 팔은 `--start random` 이고 MOCHA 는 `paths.seed_restart: ''` 로 자체 시드 restart 를 만든다.
lpopt 의 restart 를 넣으면 두 팔이 시작 노심을 공유하게 되는데 §3.1 은 그것을 등록하지 않았다.

**이 라운드에서 새로 드러난 사실 3건 (전부 P-1/P-2 를 바꾼다):**

1. **`MOCHA.py run` 에는 `--config` 가 없다.** `main_run()` 은 `ROOT/MOCHA/config_apr1400.yaml`
   **한 경로만** 연다(`MOCHA.py:15016`, `ROOT` = `MOCHA.py` 의 디렉터리). 따라서 A 팔의 YAML 은
   *전달*되는 것이 아니라 스테이징된 repo 사본의 `repo\MOCHA\config_apr1400.yaml` 로
   **설치**되어야 한다. `stage_sa_control_199.ps1` 이 이 설치를 수행하고 원본을 `.orig` 로 남긴다.
2. **`case\ga80\hgc\` 에는 `FA_E1.HGC`, `FA_E2.HGC` 두 개만 넣어야 한다.**
   `config.sync_fuel_types_from_hgc()` → `discover_fuel_types_from_hgc()`
   (`MOCHA/config.py:1291-1313, 1422-1428`)가 그 디렉터리의 `FA_*.HGC` **파일명에서**
   `core.fuel_types` 를 **재생성해 YAML 값을 덮어쓴다**. 36개를 다 넣으면 로스터가 36종으로
   조용히 복구되어 §4.3 N-6 의 pair 고정이 무효화된다. 스테이징·런처 양쪽에 게이트를 넣었다.
3. **`ga80` 의 U-질량 데이터가 어디에도 없다.** 238 `data/store/fuel_types.parquet` 는 ga80 전
   타입의 `u_mass_g` 가 NaN 이고, `config/fuel_types_dbx_extracted.yaml` 은 농축도만 있다.
   따라서 `core.fuel_data` 는 **E1/E2 에 같은 `u_mass_g`(138.85, paramA 74종 평균)를 주고
   농축도만 실측값(5.0424 / 5.0466)** 으로 채웠다 — 질량이 같으면 U-질량 가중 평균이
   개수 가중 평균으로 축약되어 절대 스케일이 상쇄된다. **§8 잔여 리스크 (ii) 는 아직 열려 있다**:
   `screening.feed_spec_screen()` 가 이 입력으로 e_core ≈ 5.045 를 내는지 G-1 에서 확인한다
   (5.00 ± 0.10 안).

### A-4. P-2 YAML diff 요약 (정본 `2_LP/MOCHA/config_apr1400.yaml` 대비 14건 + 부수 3건)

| # | 키 | 정본 | 등록값 | 근거 |
|---|---|---|---|---|
| 1 | `paths.master_exe` | `D:\DeCART_MASTER\...` (199 에 없음) | `sa_control\bin\master4.0m4_r1.exe` | 199 실측: exe 는 `C:\DeCART_MASTER\BIN\` |
| 2 | `paths.case_dir` | 사라진 `2026\1_Calculation\...` 死경로 | `sa_control\case\ga80` | §8 P-2 |
| 3 | `paths.run_dir` | 사라진 `2026\2_LP\0_Case` 死경로 | `D:\sa_control_20260903\runs` | C: 여유 41.6 GiB 뿐 |
| 4 | `core.feed_avg_enrichment` | 5.40 | **5.00** | §4.2 |
| 5 | `core.feed_enrichment_tol` | 0.05 | **0.10** | §4.2 |
| 6 | `core.fuel_types` | 키 없음(dataclass 기본 `["A0","A1"]`) | **`['E1','E2']`** | N-6 |
| 7 | `core.fuel_data` | `{}` | E1/E2 수동 fallback | A-3 항목 3 |
| 8 | `objective.parameters.fxy` | `enable: true` (F_xy 목적) | **블록 삭제** | §4.2 |
| 9 | `objective.parameters.fr` | `is_objective/ratchet = false/false` | **`true/true`** | §4.2 · §0 E-1 |
| 10 | `...pin_burnup.limit` | 75.0 | **78.0** | §4.2 |
| 11 | `...axial_offset.weight` | 0.1 | **1.0** | §4.2 (lpopt 는 4축 동일가중) |
| 12 | `objective.normalization_default` | 부재 | **`{fr: 1.0e-4, boc: 625.0, fq_pin: 2.5e-3, axial_offset: 4.0e-4}`** → `_scale_for` 가 sqrt → 0.01/25.0/0.05/0.02 | §4.2. `cyc`·`pin_burnup` 은 **의도적으로 미지정**(lpopt 에도 대응 폭이 없다) |
| 13 | `objective.term_cap` | 5.0 | **0.0** | §4.2. `cap <= 0` 이면 exp 포화 분기를 통째로 건너뛴다(`mocha_annealing.py:353-355`) |
| 14 | `sa.global_exploration.enabled` | true | **false** | §3.1 (GA 는 ~720 평가를 먹는다) |
| 부수 | `sa.parallel_workers` / `sa.pcore_cpus` | 8 / `[]` | **12 / 0..23** | §3.1 (199 전원 P-core) |
| 부수 | `notifications.slack.enabled` | true | false | 무인 실행 |
| 부수 | `sdm_mtc.enabled` | true | **false** | SDM/MTC 분기는 최적화 콜이 아니어서 §5 카운터를 오염시킨다 |

**의도적으로 그대로 둔 것**: `fr.limit 1.55` · `boc.limit 1600.0` · `fq_pin.limit 2.41` ·
`axial_offset.limit 0.30` · `cyc.limit 625.0` · `feed_fa_count 121` · `schedule_enabled false` ·
`cooling_alpha 0.97` · `t0 null` · `stage_max_samples/accept 32/12` · `max_stages 300` ·
`equilibrium.*`(N-7: 코드 변경 P-3 은 하지 않는다) · `move_probabilities` · `mocha.*`.
예산은 YAML 이 아니라 런처의 `--max-evals 1000` 이다(`sa.max_evals: null` 유지).

### A-5. B 팔 덱 드라이런 (238, StubEvaluator, MASTER 0)

`~/lpopt_ws` 에서 `python -m lpopt optimize --input <deck> --dry-run` 으로 두 덱을 실행.
`data/store_coldE1E2` 는 A-1 의 콜드 스토어를 가리키는 심볼릭 링크.

- **덱 로드 통과.** `config.py:1342-1360` 화이트리스트 거부 없음 — 즉 §3.3 의 콜드스타트 키
  (`elite_top_k=0`, `elite_seed_cases=[]`, `near_miss_f_r=0.0`, `elite_frac=0.0`,
  `diversity_frac=0.70`, `[model] store_dir="data/store_coldE1E2"`)가 **전부 유효 키**다.
- **wave 배너 (seed 1):**

  ```
  {"type":"wave","wave":0,"budget_spent":8,"cumulative_master_calls":8,
   "cumulative_surrogate_evals":432,"converged":8,"feasible":0,"gate_mode":"explore","gate_accepted":true}
  {"type":"wave","wave":1,"budget_spent":16,"cumulative_master_calls":16,
   "cumulative_surrogate_evals":864,"converged":8,"feasible":1,"gate_mode":"explore","gate_accepted":true}
  {"type":"wave","wave":2,"budget_spent":24,"cumulative_master_calls":24,
   "cumulative_surrogate_evals":1294,"converged":8,"feasible":1,"gate_mode":"explore","gate_accepted":true}
  ```

- **G-2 의 절반이 여기서 이미 PASS 한다.** wave 0 의 8개 후보 `origin` 은
  `local x5 / heuristic x1 / random x2` — **elite 유래 0개**, `parent_record_id` 전부 null.
  §3.3 이 예측한 폴백 경로(`construct.py:651-666` 의 random 채움)가 실제로 그 경로다.
- **콜당 대리평가 ~= 54** (432/8). 이는 스텁 경로의 수치이지 §4.3 N-1 의 `screen_ratio ~= 312:1`
  이 아니다 — 실런에서 다시 측정한다.
- 드라이런은 **MASTER 를 돌리지 않으므로** §3.3 주의 2 대로 **대조군으로는 무용**하다.
  여기서 얻는 것은 "덱이 로드되고 콜드 경로로 구성된다"뿐이다.

### A-6. 발사 전 남은 일 (이 부록 시점 기준)

1. **199 는 아직 바쁘다** — 조회 시점에 `master4.0m4_r1 x1`, `python x4`(kit 부트스트랩).
   **G-3 미충족.** 슬라이스 Z S9c 종결 확인이 먼저다.
2. **스테이징 미실행** — `stage_sa_control_199.ps1` 은 작성만 됐다. 실행 순서는
   (a) 스테이징 → (b) YAML 을 199 로 전송하고 `-Force` 재실행(설치 + 해시 기록) →
   (c) 콜드 스토어(records + maps)를 238 → 199 `kit_frontier\data\store_coldE1E2\` 로 전송 →
   (d) 덱·런처 4종을 kit 루트로 전송.
3. **미확인 게이트 (i)**: 통합 `MAS_XSL` 안에 `FA_E1`/`FA_E2` 세트가 실제로 배치돼 있는가.
   스테이징 스크립트가 확인하고, 실패하면 P-1b(2-타입 재배치, DeCART 불필요)로 분기한다.
4. **미확인 게이트 (ii)**: `core.fuel_data` 수동 입력으로 `feed_spec_screen()` 의 e_core 가
   5.045 로 나오는가 (A-3 항목 3).
5. **G-0 미실행** — 10개 검증코어 교차검증. `run_sa_control_A_199.ps1` 은 `G0_CROSSCHECK_PASS.json`
   영수증이 없으면 거부한다. **G-0 FAIL 이면 §11 D7 대로 실험 전체 중단이다.**
6. **G-1 미실행** — `run_sa_control_A_199.ps1 -Commission` (40콜).
7. **RESTAMP 미완** — `run_sa_control_A_199.ps1` 의 `$wantCfg` 는 아직 자리표시자다
   (199 에 설치된 YAML 의 해시가 나와야 채운다). B 런처의 콜드 스토어 핀은 A-1 값으로
   스탬프돼 있으나, **스토어를 재생성하면 다시 스탬프해야 한다.**
8. **§10 스탬프 미기입** — 발사 시점 항목이다. 본 부록은 그 자리를 대신하지 않는다.

**이 부록을 쓴 시점까지 실행된 노심계산: 0.**

### A-5 보충 — seed 2 배너와 등록해야 할 경고 1건

seed 2 덱도 동일하게 로드된다. 실제 배너 두 줄:

```
[optimize][DEPRECATED] objective='min_fr_max_cycle' is a RETIRED production mode
 (flatness-first program S10 STOP): it steers the search by F_r.
 Kept runnable for reproduction / A-B baselines only — use objective='flat_power' for production.
[optimize] campaign dryrun_cold_s2 case=E1_E2/feed-121 budget=300 spent=0 dry_run=True
```

**등록.** `min_fr_max_cycle` 은 현재 lpopt 에서 **은퇴한 프로덕션 모드**이고, 엔진 스스로
"reproduction / A-B baselines 용으로만 실행 가능"이라고 인쇄한다. 본 실험은 정확히 그 용도이므로
사용은 정당하다. 다만 **결과 문서는 이 문장을 그대로 인용해야 한다** — B 팔은
"현행 프로덕션 lpopt"가 아니라 **"F_r 목적으로 돌린 lpopt"** 이며, 이는 A 팔을
"튜닝된 SA 가 아니라 현행 MOCHA"로 한정한 §4.4-3 과 대칭인 한정이다.
플랫니스 우선(`flat_power`) 프로그램과의 관계는 본 라운드의 범위 밖이다.

두 드라이런은 wave 3 이후 중단했다(스텁 300콜은 ~2.5 h 이고 더 얻을 정보가 없다).
238 에는 산출물만 남았다: `~/lpopt_ws/scratch/dryrun_cold_s{1,2}/`,
`~/lpopt_ws/data/store_coldE1E2` (→ `scratch/store_coldE1E2` 심링크). 워크스페이스 루트에
복사했던 덱 2개는 삭제했다. **238 의 다른 작업(진행 중이던 s1j 학습 런)은 건드리지 않았다.**
