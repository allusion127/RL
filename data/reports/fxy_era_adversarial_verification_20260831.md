# ADVERSARIAL VERIFICATION — `F_xy` 시대 5개 주장 · 3 렌즈 반증 감사

- **대상 주장:** C1(bf3a70b2 deliverable) · C2(`s1j` 승격) · C3(serve provenance 수정) · C4(HGD569 범위 160행) · C5(격리행 누출 불가)
- **설계:** 주장 5 × 렌즈 3 = **15 판정 예정**, **14 회수**(C2 의 "숫자 재계산" 렌즈 1건이 agent 실패로 유실 — §2.2)
- **실행일:** 2026-09-02 (워크플로 `wf_ac3197c3-9b0`) · **작성일:** 2026-09-03
- **원 저널:** `C:/Users/MK/.claude/projects/c--Users-MK-Desktop-CT-RPL-2-Project-KNF-LEU------------------2026-2---/e627baec-f015-4caf-86fc-4e4c11eaa355/subagents/workflows/wf_ac3197c3-9b0/journal.jsonl` (60행: `started` 15 + `result` 14 + `failed` 10 + 재개 `started` 21)
- **계산 위치:** 전량 HOST_238 (`ssh -p 8022 USER@HOST_238`, `~/lpopt_ws`, `venv/bin/python`). 로컬은 `grep`/`sed` 읽기만.
- **부산물:** `data/reports/_wf_scratch/c3_adjudication_20260902.csv`, `_wf_scratch/c4_scan.py`, `_wf_scratch/c4_degen.py`, `_wf_scratch/c4_r1_degen.py`
- **파일명 날짜 주의:** 파일명은 `F_xy` 시대 문서군 명명(`*_20260831`)을 따랐으나 **실제 검증 실행일은 2026-09-02** 이다. 같은 종류의 라벨-실행일 괴리를 C4 렌즈가 별도로 지적했다(§6.4(b)).

> **총평. 5개 중 2개가 그대로 섰고(C1·C4, 0/3 반증), 2개가 3/3 전부 반증되었으며(C3·C5), 1개(C2)는 게이트 절반만 살아남았다.**
> 반증된 것들의 공통 성질이 중요하다 — **틀린 숫자는 거의 없었다. 틀린 것은 "범위"와 "메커니즘"이다.** C3 는 서로게이트 서빙 경로만 고쳤는데 "모든 서빙 경로"라고 썼고, C5 는 `converged` 만 거르는 코드를 두고 "격리행은 누출될 수 없다"고 썼으며, C2 는 코드가 12시간 뒤에야 참이 된 문장을 승격 문서에 절대명제로 박아 두었다. **주장문을 코드가 실제로 보장하는 범위로 좁혀 다시 쓰는 것이 이 감사의 제1 산출물이다.**

---

## 1. 판정 요약

| 주장 | 렌즈 반증 | 최종 판정 | 핵심 사유 |
|---|---|---|---|
| **C1** — `bf3a70b2…` 는 deliverable 급 노심 (F_xy 1.5322 / F_r 1.4857 / CBC 1337 / F_q 1.853 / \|AO\| 0.0244 / pin 63.760, ΔF_xy=0, `is_deliverable`=True) | **0 / 3** | **UPHELD** (경고 포함) | 6개 축·한계·결정성·모집단이 전부 독립 재현. 다만 `is_deliverable(row, limits)` 는 **2인자**이고 SDM/MTC 는 미완이라 "deliverable-grade ≠ 인허가 통과" |
| **C2** — `s1j` 승격은 유효 (G1/G2′/G3′ 통과 + 서빙이 σ bar 를 지킴) | **1 / 2** (렌즈 1건 유실) | **UPHELD-WITH-CAVEAT** | 게이트 절반은 해시까지 완전 재현. **σ bar 절반은 생산에서 거짓이었고(D3), 인용된 r2 로그는 존재하지 않으며, `PROMOTION.md` §3.2 R1 해시 3/4 가 D3 수정 후 stale** |
| **C3** — serve provenance 결함은 완전 수정, 다른 서빙 경로 없음, train/serve parity 성립 | **3 / 3** (major·major·minor) | **REFUTED** | 수정은 **서로게이트 경로 한정**. policy scorer v1/v2 는 설계상 `library_provenance` 의 `free69` 를 계속 사용. 게다가 **`g_e_core` train/serve 불일치가 살아 있고 게이트가 `head(25)` 라 이를 볼 수 없다** |
| **C4** — HGD569 alias-no-op 은 정확히 160행에만, 152행 in-place 갱신, 8행 격리 | **0 / 3** | **UPHELD** (경고 포함) | deck 레벨 전수 census 로 160 파일 정확 일치, 다른 트리 전부 0. 다만 **격리가 `converged=False` 라는 출처 불명 상태에 의존**하고 re-kit 이 이를 되돌릴 수 있음 |
| **C5** — 격리/무효행은 학습·엘리트·정책 코퍼스에 누출될 수 없다 | **3 / 3** (전부 major) | **REFUTED** | "누출 불가"라는 불변식은 **코드에 존재하지 않는다**. 필터는 전부 `converged` 만 보고 `valid` 를 보지 않으며, `build_steps` 에는 필터가 아예 없고(무효 자식 edge **775건**), `dataset_torch._maps` 는 미수렴 행의 맵을 그대로 학습 타깃으로 넣는다 |

### 1.1 렌즈별 원시 판정 (저널 `result` 14건)

| 주장 | 렌즈 | agentId | `refuted` | `confidence` | `severity` |
|---|---|---|---|---|---|
| C1 | L1 숫자 재계산 | `ac9cf274b9b42f8e4` | false | 0.93 | none |
| C1 | L2 코드 경로 | `ae742fcbd9b34f719` | false | 0.93 | minor |
| C1 | L3 출처·프로세스 | `a6d85aa82ed76d37a` | false | 0.90 | minor |
| C2 | L1 숫자 재계산 | `abb816eec3dfef829` | — | — | **FAILED (판정 없음)** |
| C2 | L2 코드 경로 | `a0799edeaa86823f0` | false | 0.82 | minor |
| C2 | L3 출처·프로세스 | `a152f80664240e247` | **true** | 0.90 | **major** |
| C3 | L1 숫자 재계산 | `ad2eb5d3c42de47b7` | **true** | 0.90 | **major** |
| C3 | L2 코드 경로 | `a094544d5db4bcb37` | **true** | 0.88 | **major** |
| C3 | L3 출처·프로세스 | `a2b2959eb6efc710b` | **true** | 0.78 | minor |
| C4 | L1 숫자 재계산 | `a544d7d3b63f225df` | false | 0.92 | none |
| C4 | L2 코드 경로 | `a2e717e444724aa51` | false | 0.90 | minor |
| C4 | L3 출처·프로세스 | `ab7e14477fd1633d9` | false | 0.90 | minor |
| C5 | L1 숫자 재계산 | `a6de236910a755853` | **true** | 0.93 | **major** |
| C5 | L2 코드 경로 | `a8383661416d1a6d9` | **true** | 0.90 | **major** |
| C5 | L3 출처·프로세스 | `a94a49ed5005c0825` | **true** | 0.90 | **major** |

---

## 2. 렌즈 정의와 감사 설계

### 2.1 세 렌즈 (모든 주장에 동일하게 적용)

- **L1 — 숫자 재계산.** *"recompute every number in the claim from the data files (on HOST_238 for parquet/npz work; local Grep for text archives). Refute if any number, count, or flag does not reproduce."*
- **L2 — 코드 경로 추적.** *"trace the code path that produces the claimed behaviour (read the functions, follow imports, check defaults and every caller). Refute if a caller, default, or branch violates the claim."*
- **L3 — 출처·프로세스.** *"check provenance and process — hashes, timestamps, which artefact/version was actually used, pre-registration compliance, whether the claim rests on a stale copy. Refute if the evidence chain has a gap."*

세 렌즈 모두 **`refuted=true` 를 기본값**으로 지시받았다(*"Default to refuted=true if you cannot reproduce a load-bearing number"*). 즉 C1·C4 의 0/3 은 **적대적 기본값을 뚫고 나온 유지 판정**이며, 그만큼 무겁게 읽어야 한다.

### 2.2 회수하지 못한 판정 1건 (C2 · L1)

저널 30행 `{"type":"failed","key":"v2:93c42e77…","agentId":"abb816eec3dfef829"}`. 이 에이전트의 지시문(`agent-abb816eec3dfef829.jsonl` 1행)은 **CLAIM C2 + L1(숫자 재계산)** 이었다. 재개 시도(48–60행의 `started` 블록, `agentId` `ad28b64639fc927c6`)도 `result` 를 남기지 않았다.

**따라서 C2 의 최종 판정은 3렌즈가 아니라 2렌즈에 근거한다.** 하필 유실된 것이 "게이트 숫자를 처음부터 다시 계산하는" 렌즈인데, 남은 두 렌즈가 각각 독립적으로 G1/G2′/G3′/G4 값을 원본 JSON 에서 재확인했으므로 실질 공백은 크지 않다. 다만 **C2 를 "3렌즈 통과"로 인용해서는 안 된다.**

> 참고: 저널 31–47행에는 별개 키 9건이 `started`→`failed` 로 연달아 남아 있다(`f8f5a200…`, `06bcefe8…`, `af13c545…`, `dc923ca9…`, `0105fb97…`, `788ee2e6…`, `de10f0bd…`, `6d74f33a…`, `ffb822f8…`). 해당 agent 로그는 전부 ~52 KB 의 조기 종료본으로, **이 감사의 5개 주장 판정에는 기여하지 않았다.**

---

## 3. C1 — `bf3a70b2…` 는 deliverable 급 노심 → **UPHELD**

### 3.1 주장 원문

> **CLAIM C1:** Record `bf3a70b2…` (campaign `fpcamp_minfxy_t6t4_f121_r1`) is a deliverable-grade core: F_xy 1.5322, F_r 1.4857, CBC 1337, F_q 1.853, |AO| 0.0244, `max_pin_burnup` 63.760 all MEASURED by MASTER and within limits (1.65/1.55/1600/2.41/0.30/80), determinism ΔF_xy=0 on replays, and `is_deliverable(row)` returns True.

### 3.2 렌즈별 판정

| 렌즈 | 판정 | 한 줄 |
|---|---|---|
| L1 숫자 | **재현 (severity none)** | 6개 축·한계·`is_deliverable`·ΔF_xy·모집단 25행 전부 일치 |
| L2 코드 | **재현 (minor)** | `deliverable_limits`→`unknown_axes`→`is_deliverable` 실행 결과 True. 다만 `cyclen` 은 게이트되지 않고, pin 정의는 core-wide 스캔이 아님 |
| L3 출처 | **재현 (minor)** | 사전등록 시각 선행 확인. 다만 **캠페인 자체 산출물이 아직 `deliverable=false` 라고 말하고 있음** |

### 3.3 재현된 근거 (인용)

**① 저장소 행.** 238 에 로컬 store 를 올려 확인(`scp -P 8022 data/store/records.parquet USER@…:~/lpopt_ws/scratch/records_local_20260902.parquet`, sha 도착 검증):

```
record_id bf3a70b20e508c7c01d15fd62bccc653376f65072a7507a9b3fdc755898ed982 → nmatch 1 (idx 75849)
f_xy 1.5322 · f_xya 1.3803 · f_r 1.4857 · f_q 1.853 · cbc_max 1337.38 · ao_abs 0.0244
cyclen 622.101 · max_pin_burnup 63.76 · max_assembly_burnup 55.782 · converged True · valid True
campaign fpcamp_minfxy_t6t4_f121_r1 · case_pair T6_T4 · feed 121 · library paramA
restart native:MAS_RST.APRQ_10_0615.11
```

같은 `pattern` 을 갖는 행은 **1건**(중복·그림자 레코드 없음). "CBC 1337" 은 1337.38 의 반올림 — 무해.

**② 한계값은 deck 자체의 것.** `fpcamp_minfxy_T6T4_f121_r1_199.inp` `[acquisition]`: `f_xy_limit=1.65`, `f_r_limit=1.55`, `cbc_limit=1600.0`, `f_q_limit=2.41`, `ao_abs_limit=0.30`, `minfxy_pin_bu_limit=78.0`, `minfxy_cyclen_lo/hi` **의도적 미설정**. `feasibility_limits_for(cfg.acquisition,"min_fxy")` → `{'cbc_max':1600.0,'f_q':2.41,'ao_abs':0.3,'f_r':1.55,'max_pin_burnup':78.0,'cyclen_lo':None,'cyclen_hi':None,'f_xy':1.65}`, `deliverable_limits(...)` 가 pin 만 **80.0**(`DELIVERABLE_PIN_BU_LIMIT`, `campaign.py:254-258`)으로 재해석. 사후 맞춤이 아니다.

**③ 술어 실행.** `unknown_axes(row, lim)` → `()`; `is_feasible_search` → True; **`is_deliverable(row, lim)` → True** (`lpopt/search/campaign.py:567`). 여유: f_xy 0.1178 · f_r 0.0643 · cbc 262.62 · f_q 0.557 · |AO| 0.2756 · pin 16.24 — 결과 보고서 표와 동일.

**④ ΔF_xy = 0 을 독립 재구성.** replay JSONL 의 `deltas_measured_minus_stored` 에는 f_xy 가 없어(6축만 0.0) 두 렌즈가 각각 digest 조인을 새로 만들었다: `digest_of_packed(row["pattern"]) == "58d2748831b85e3d"`, 이 digest 가 `fxy_backfill_199_pinbu_wave_minfxy_r1_20260830.csv` 2행(f_xy 1.5322, f_xya 1.3803, `cycle_evidence=final`, `sane=1`, `efpd_max=622.101`). 결정적인 대목은 **작업 디렉터리가 다르다**는 것 —
원 캠페인 `…/worker_00/58d2748831b85e3d__MAS_RST.APRQ_10_0615.1-**6988c38383-gxbprs7t**` vs replay `…/pinbu_wave_minfxy_r1/…/58d2748831b85e3d__…-**ff0e289cea-tirsbnvx**` — 즉 **서로 독립한 두 번의 MASTER 실행이 같은 F_xy** 를 냈다. 25 체인 전수: 25/25 조인, **max |ΔF_xy| = 0.0**, 25/25 `determinism_ok`·`provenance_ok`, CSV 25행 전부 `final`/`sane=1`.

**⑤ 모집단.** 같은 한계로 store 76,693행 전수 적용 → **deliverable 25행**(`fpcamp_minfxy_t6t4_f121_r1` 20 + `batchswap_enum_T6T4` 4 + `fpcamp_minfr_T6T4_r8` 1), F_xy 1.5322–1.5890 · F_r 1.4605–1.5238 · pin 62.451–63.878, **`bf3a70b2` 가 F_xy 최소**. 보고서 §7 과 자릿수까지 일치.

**⑥ 사전등록 순서.** `pinbu_wave_minfxy_r1_prereg_20260830.md` mtime Aug 30 21:39 → wave wall 1262 s(workers=12), `results.jsonl` mtime 22:01, rc=0. 준수.

### 3.4 반증에는 이르지 못한 경고 (전부 기록)

1. **`is_deliverable(row)` 는 존재하지 않는다.** 실제 시그니처는 `is_deliverable(row, limits)`. True 판정은 **이 deck 의 한계에 조건부**다. `deliverable_limits` 는 인허가 1.55 를 일반적으로 강제하지 않는다(docstring 이 flat_power 의 1.70 을 유지한다고 명시). 이 행은 1.4857 이라 둘 다 통과하므로 무해.
2. **`cyclen` 은 게이트되지 않았다.** deck 이 `minfxy_cyclen_lo/hi` 를 비워 두어 `deliverable_limits` 가 `None` 을 전파하고 `_DELIVERABLE_AXIS_COLUMNS` 가 해당 열을 건너뛴다. 결과 보고서의 ``| `cyclen` | (band) | 622.101 | — | check |`` 행은 **장식**이다 — 계산된 판정이 없다.
3. **`max_pin_burnup` 은 core-wide 스캔이 아니다.** `master.py:818-849` → `burnup.parse_ppi_max_pin_burnup(text, X, Y)`, docstring "the maximum 3-D pin burnup in **ONE** PPI assembly block"(`burnup.py:139-144`), `lpopt/data/pinppi.py:3-10` 도 동일 명시. 정의 문서(`pinbu_definition_20260820.md` §2)가 이를 **선언으로 해소**했으므로 코드는 비준된 정의와 일치한다. core-wide 동치는 n=5 f113 노심에서만 경험적으로 확인. 여유 16.24 GWd/tU 를 고려하면 실무상 무해.
4. **"deliverable-grade" ≠ 인허가 통과.** `build_delivery_payload` 는 `objective != "flat_power"` 면 `None` 을 반환하므로 이 min_fxy 캠페인은 **`delivery.json` 을 쓰지 않았다.** `sdm_mtc_targets.jsonl` 에 95개 타깃(이 레코드는 55행)만 있고 `sdm_mtc.json` 은 없다 — **D9 SDM/MTC 사전-인도 게이트는 실행된 적이 없다.**
5. **캠페인 자체 산출물이 아직 반대로 말한다.** `runs/fpcamp_minfxy_t6t4_f121_r1/status.json` best 블록은 `"deliverable": false`, `"unknown_axes": ["max_pin_burnup"]`; `report.md` 는 `DELIVERABLE … : 0` 과 `_No DELIVERABLE verified LP this campaign_`. Aug 30 pin 백필 이후 **재렌더되지 않았다.** 캠페인 산출물만 읽는 감사자는 정반대 결론에 도달한다.
6. pin 예측 오차 **+2.8293**(예측 66.589 vs 측정 63.760), `max_rod_avg_burnup` 은 NaN(비게이트·미측정).
7. 인접 사실(반증 아님): 같은 보고서 §5.2 가 r1 의 F_xy 이득을 **0.0169 → 0.0080** 으로 자체 수정했다(당시 미라벨 상태였던 `4d70ab6f` 가 1.5402 를 갖고 있었고 F_r/F_q/CBC 에서 `bf3a70b2` 를 지배). "어느 노심을 인도하는가"에 영향을 주지만 C1 의 6축 판정에는 무관.

### 3.5 최종 판정 — **UPHELD**

숫자·한계·술어·결정성·모집단이 모두 독립 재현되었다. 단, **인용할 때의 정확한 문구**는 다음이어야 한다:

> `is_deliverable(row, limits) = True` — **r1 deck 자체의 게이트**(F_xy 1.65 / F_r 1.55 / CBC 1600 / F_q 2.41 / |AO| 0.30 / pin 80) 하에서, `unknown_axes = ()`. **SDM/MTC/축방향 확인은 미완**이며 `delivery.json` 은 존재하지 않는다.

### 3.6 조치 (전부 **미착수**)

| # | 조치 | 위치 | 상태 |
|---|---|---|---|
| C1-1 | `status.json`·`report.md` 재렌더 또는 `SUPERSEDED-BY` 스탬프 (현재 `deliverable=false` 로 C1 과 정면 충돌) | `runs/fpcamp_minfxy_t6t4_f121_r1/` | 미착수 |
| C1-2 | 결과 보고서 §6 의 `cyclen` 행을 `UNGATED (band unset in deck)` 로 교체 | `pinbu_wave_minfxy_r1_results_20260830.md` | 미착수 |
| C1-3 | `fxy_backfill_*.csv` 에 `record_id`(또는 digest 매핑) 열 추가 — 현재 조인이 `sha256(pattern)[:16]` 재유도라는 미문서화 홉에 의존 | `data/reports/` | 미착수 |
| C1-4 | min_fxy 라운드에 인도 dossier 가 필요하면 `build_delivery_payload` 의 `if objective != "flat_power": return None` 조기 반환을 **의도적 결정으로서** 재검토 | `campaign.py` | 미착수 (설계 판단 필요) |

---

## 4. C2 — `s1j` 승격의 유효성 → **UPHELD-WITH-CAVEAT**

### 4.1 주장 원문

> **CLAIM C2:** The `s1j` promotion is valid: arm 3 (`data/models/s1j`, `--fxy-direct`) passed G1 no-regression vs `s1i` and G2′ MAE<0.0767 / G3′ ρ̄>0.7263 on the FIXED serve path with refit calibrations, **and serving honours the σ bar** (`predict_fxy` returns head mean + proxy σ; wave checkpoints carry `serve_sigma=barred`). Sources: … **the running r2 log lines** `"[optimize][F_xy SIGMA] wave N checkpoint … carries serve_sigma=barred"`.

### 4.2 렌즈별 판정

| 렌즈 | 판정 | 한 줄 |
|---|---|---|
| L1 숫자 | **유실** | agent `abb816eec3dfef829` 실패 (§2.2) |
| L2 코드 | 유지 (conf 0.82, minor) | σ bar **읽기** 경로는 실재·현행. 그러나 "생산에서 참"은 2026-08-30 12:39 이후에만. **별건 결함 1건 발견** |
| L3 출처 | **반증 (major)** | r2 로그 부재 · `PROMOTION.md` R1 해시 3/4 stale · D3 를 실증 |

### 4.3 살아남은 절반 — 게이트 (해시까지 일치)

- **G1**: `gate_fxy_arm3_20260829_checkonly.json` → `"pass": true`, `"epsilon": 0.1388114093847`, `"worst_drop": 0.011363636363636354`, `blind_targets: []`, `unavailable: []`, `prev = data/models/s1i`, `new = data/models/20260829_194532`(→ `s1j` 로 개명). N=144 읽기 `ε = 0.14216194539159127` 과 내적 정합: 0.1422/0.1388 = 1.0245 = Φ⁻¹(0.95^(1/144))/Φ⁻¹(0.95^(1/108)) = 3.386/3.305.
- **G2′**: bar 0.0767 · 측정 **0.06630023009541522** · PASS. **G3′**: bar 0.7263 · 측정 **0.7903921327831772** (Δ +0.06409213278317727) · PASS. **G4**: coverage **0.8310214375788146** > 0.80 → **FAIL**(σ̄ 0.11978711326476908).
- **동반 판독**(같은 JSON): `PROXY_S1I_SERVE.mae` **0.0731733146183275**, `serve_path.PROXY_S1I.rho_mean` **0.7156957178250644**. 후보가 고정 바와 재측정 proxy 를 **둘 다** 이겼으므로 사전등록 C.3 #2 의 "split reading" 조항은 실제로 발동하지 않았다.
- **바를 사후에 옮기지 않았다:** `fxy_gate_eval_arm3_20260829.py:65-66` 에 `G2P_BAR=0.0767` / `G3P_BAR=0.7263` 하드코딩, `fxy_head_prereg_20260829.md:656`("Amendment C … 실행 전 개정"), `:852-853`("B.5 그대로, 무변경"), `:741` 이 채점 빌드(`model_api.py 94229de9…`, `featurize.py 6977344d…`)를 **실행 전에** 고정.
- **산출물 해시 전수 일치:** `sha256sum data/models/s1j/*` → `ensemble.json 75cdc818…`, `calibration ce92ab90…`, `member_20260716/meta f0af69c0…`, `cell 1f4b7415…`, `f_r d4e96330…`, `cbc 91526205…`, `f_q babd655d…`, `ao_abs b3991af3…`, `flatness c91870dd…` — `PROMOTION.md` §3.1/§3.2 와 동일. 게이트 산출물(`371f0e5a…`, `f70136d1…`, `c191141a…`)도 일치. **238 미러도 동기**: `ssh … 'sha256sum data/models/s1j/{ensemble,f_r_calibration,cell_calibration}.json'` → `75cdc818… / d4e96330… / 1f4b7415…`.
- **σ bar 읽기 경로는 실재:** `acquisition.py:717-727`(barred 면 `fxy_proxy` sd 로 치환, 상수 `FXY_PROXY_SLOPE/SIGMA_K/RESID_SD` at `:654-656`), `model_api.py:616-638`(ensemble.json → member meta fallback → `fxy_sigma_barred`), `acquisition.py:2570`(barred 동안 conformal upper = None), 호출자 전수 `acquisition.py:988/:2045`, `intervention_wave.py:986`; 직접 backend 호출은 `tripletype_midpick.py:133` 하나뿐이고 mean 만 사용. 단위시험 `tests/test_fxy_head.py:1008 test_barred_head_sigma_serves_the_proxy_sigma`, `:1048` 존재.

### 4.4 부러진 절반 — σ bar 와 증거 사슬

**A) 생산에서 거짓이었다 (defect D3).** `minfxy_T6T4_f121_r1_results_20260830.md:481` 이 이미 등록: *"head sigma bar 가 resume 에서 유실됐다. wave 11-12(call 89-100)가 head 자체 sigma(0.054-0.196)로 서빙됐다 … `_save_champion` 이 쓰는 run-dir checkpoint 에 `ensemble.json` 이 없다."* 실증:

```
find runs/fpcamp_minfxy_t6t4_f121_r1/models -maxdepth 2 -name ensemble.json | wc -l   → 0
grep -c serve_sigma runs/.../champion_wave_12/member_20260716/meta.json runs/.../champion_wave_12/backend.json → 0, 0
```
(champion_wave_{04,05,08,09,12} 전부 `backend.json`/`calibration.json`/`feature_ood.json`/`member_*` 만 보유.)
독립 재구성도 같은 결론: 로그된 `exploit` 을 두 σ 가설로 역산하면 wave 11-12 에서 **head-σ 가설 median |residual| 3.0e-05 vs proxy 27.2**(같은 보고서 `:254-264`). **100 acquisition call 중 마지막 12건이 bar 가 막으려던 바로 그 σ 로 랭크되었다.**

**B) 인용된 로그가 존재하지 않는다.**
```
grep -rl "carries serve_sigma" --include="*.log" . | wc -l   → 0
```
문자열은 `lpopt/search/campaign.py:2765`, 그 `.pyc`, `run_fpcamp_minfxy_E1E2_f121_r2_199.bat` 에만 있다. "running r2" 도 없다 — `fpcamp_minfxy_e1e2_f121_r2_out.log` 부재(런처 `launch_fpcamp_minfxy_E1E2_f121_r2_199.ps1:174` 가 정확히 그 이름을 쓴다), `runs/` 에 r2 디렉터리 없음. 출처는 `minfxy_E1E2_f121_r2_prereg_20260831.md:603` — **"그 줄이 나와야 한다. 그 줄의 부재가 결함이다(D3)"** 라는 *사전등록된 기대치*다. **C2 는 기대치를 관측인 것처럼 인용했다.** (r2 는 box 199 에서 돌고 로그는 `C:\Users\USER\lpopt_work\kit_frontier\…` 에 남으므로, 확인하려면 199 에서 끌어와야 한다.)

**C) `PROMOTION.md` §3.2 R1 매니페스트가 stale.** R1 은 *"code and calibrations move as ONE set"* 이라며 `model_api.py 5a713eaa…`, `acquisition.py f1100d1f…`, `campaign.py a7d0caf4…` 를 고정한다. 현재 측정:

| 파일 | `PROMOTION.md` 선언 | 실제 (mtime) |
|---|---|---|
| `model_api.py` | `5a713eaa…` | **`139c82fc12c48ff0d212de3708a575344ebfa0606f22453a39a5267987334b04`** (Aug 30 12:26) |
| `acquisition.py` | `f1100d1f…` | **`84ff2f11eea751389e8894dbeda45d512b668314cf3343ab3af7270120603732`** (Aug 30 12:39) |
| `campaign.py` | `a7d0caf4…` | **`6d6425bba9f1b0605a31f0cf26f21b0da08faee8f7c602c9f39feb8a9ac7fcaa`** (Aug 30 12:39) |
| `featurize.py` | `6977344d…` | 일치 |

`PROMOTION.md` 는 04:07 작성 — 즉 **D3 수정(12:26/12:39)이 8시간 뒤**다. 차이는 보호적이지만 **R1 이 재스탬프되지 않아 "one set" 규칙이 더 이상 출하 코드를 지목하지 못한다.** 선언된 `5a713eaa…` 는 `data/reports/` 와 deck 전체 grep 에서 0 히트 — **어떤 산출물로도 뒷받침되지 않는 링크**. 드리프트 사실 자체는 다른 문서(`fxy_head_results_arm4_20260831.md:349`)에만 기록되어 있다.

**D) L2 가 발견한 별건 결함 (D3 와 같은 계열, 어떤 가드도 없음).** `PosValCnnBackend.save()`(`model_api.py:2060-2092`)는 members·`ensemble.json`·`calibration.json`·`feature_ood.json`·`backend.json` 을 쓰지만 **per-cell 보정 6종**(`cell_/f_r_/cbc_/f_q_/ao_abs_/flatness_calibration.json`)은 쓰지 않는다. `from_dir`(`:736-760`)는 넘겨받은 디렉터리에서만 그것들을 읽고, `campaign.py:1660-1662` 의 resume 은 `backend_factory(self.champion_ckpt)`(= `<run_dir>/models/champion_wave_NN`)를 쓴다. **따라서 `--resume` 은 per-cell 수준 보정이 전부 사라진 `s1j` 후손을 서빙한다.** σ 는 무관하지만 **F_r ≤ 1.55 / F_q / CBC / |AO| 수준 게이트와 cyclen LCB 가 영향을 받는다** — 사전등록 C.3 #3 이 등록한 "수준 임계값을 믿을 수 없다" 위험이고, R1 의 "one set" 규칙 위반이다. D3 가드는 `serve_sigma` 만 본다.

**E) 다음 승격을 위한 경고.** arm 4 후보 `data/models/20260902_122746/ensemble.json` 에는 `fxy_head` 블록 자체가 없다 → **개명 방식으로 승격하면 bar 가 없는 체크포인트가 출하된다.**

### 4.5 최종 판정 — **UPHELD-WITH-CAVEAT**

- **유지:** *"arm 3 이 Amendment C 하에서 FIXED serve path + refit 보정으로 G1/G2′/G3′ 를 통과했다"* — 해시까지 완전 재현.
- **반증:** *"serving honours the σ bar (wave checkpoints carry `serve_sigma=barred`)"* — **2026-08-30 12:39 이전에는 거짓**이었고(r1 call 89-100), 그 이후로도 **실제 실행에서 검증된 적이 없다.** 인용 근거로 제시된 r2 로그는 존재하지 않는다.
- **권장 재서술:**
  > arm 3 은 FIXED serve path 에서 G1/G2′/G3′ 를 통과했고(재현 가능, 해시 무결), `predict_fxy` 는 barred 체크포인트에 대해 proxy σ 를 대입한다(코드 + 단위시험). **wave 체크포인트의 bar 지속성은 승격 시점에 깨져 있었고(D3: r1 call 89-100 이 head 자체 σ 로 서빙), `_save_champion` 가드가 2026-08-30 12:39 에 이를 고쳤으나 실 실행에서는 아직 미검증이다.**

### 4.6 조치

| # | 조치 | 상태 |
|---|---|---|
| C2-1 | `PROMOTION.md` §3.2 R1 재스탬프 (`model_api.py 139c82fc…`, `acquisition.py 84ff2f11…`, `campaign.py 6d6425bb…`), 구 행은 날짜와 함께 보존, `5a713eaa…` 가 어떤 산출물로도 확인되지 않음을 명기 | **진행 중** (별도 수정 워크플로) |
| C2-2 | `PROMOTION.md` §2 의 절대명제 *"the head's sigma is never served"* → *"2026-08-30 12:39 이후 코드에서만; r1 call 89-100 은 서빙했다(D3)"* 로 수정하고 D3 를 §2 에 등록 | **진행 중** (C2-1 과 동일 재스탬프 작업) |
| C2-3 | r2 로그를 box 199 에서 실제로 끌어와 `[optimize][F_xy SIGMA] wave NN checkpoint … carries serve_sigma='barred'` 존재를 확인한 뒤에만 인용 (부재 시 199 kit 이 수정 이전판이라는 뜻) | 미착수 |
| C2-4 | **D3 의 보정 판본**: `PosValCnnBackend.save()` 가 per-cell 보정 6종을 파생 체크포인트로 라운드트립하도록 확장(`_save_ensemble_meta` 와 동일 방식, 복사만·합성 금지) + `_save_champion` 에 보정 집합 동일성 단언 추가 | 미착수 |
| C2-5 | 승격 절차에 `fxy_head.serve_sigma` 스탬프(또는 명시적 G4 통과 기록) 요구를 넣어 개명 승격으로 bar 없는 체크포인트가 나가지 못하게 할 것 | 미착수 |

---

## 5. C3 — serve provenance 결함의 "완전" 수정 → **REFUTED (3/3)**

### 5.1 주장 원문

> **CLAIM C3:** The serve-path featurization defect (`library_provenance` inverting `g_sym_class` for ga80 / `g_dataset_flag` for paramA) is **fully fixed** by `featurize.serve_provenance()` and **no other serving path** (policy scorer, remote screening `lpopt/remote.py`, `autoeng.py`, `scoping_mesh.py`, tools) still uses the inverted provenance; **train/serve parity holds**.

### 5.2 렌즈별 판정 — **세 렌즈 전부 반증**

| 렌즈 | 판정 | 반증 지점 |
|---|---|---|
| L1 숫자 | **REFUTED (major)** | ① policy scorer v1/v2 ② S1j-val 793행 중 **188행**이 `g_e_core` 1e-6 위반 ③ "2,401 paramA corpus rows" 는 현재 2,388 ④ 인용 출처 오기 |
| L2 코드 | **REFUTED (major)** | ① scorer 기본값 v1 ② `tests/test_featurize.py` **현재 red** ③ 게이트가 `head(25)` 라 구조적으로 못 잡음 |
| L3 출처 | **REFUTED (minor)** | ① scorer v1/v2 실행 확인 ② 574 legacy 행 비-parity ③ remote 스크리닝에 소스 게이트 없음 |

### 5.3 재현된 절반 — 서로게이트 경로는 실제로 고쳐졌다

- **census 완전 일치**(`featurize.py:816-822`, box store 74,717행 기준): `260624/A 29,976 · 5.8_5.1/A 8,244 · ga80/B/free69 574 · ga80/P/rot61 18,973 · legacy_a/A 634 · paramA/P/rot61 16,316`. `_DATASET_A_LIBRARIES` 를 store 에서 재유도 → `{'260624','5.8_5.1','legacy_a'}` 정확.
- **행 단위 대조:** `serve_provenance` vs store → `g_dataset_flag` 불일치 **0**, `sym_class` 불일치 **574**(문서화된 extract_b 수확분만). `library_provenance` vs store → `g_dataset_flag` **16,316**(전부 paramA), `sym_class` **18,973**(전부 campaign ga80). 주장이 말한 두 개의 뒤집힘 정확히 일치.
- **강제 역전 대조군:** `model_api.serve_provenance = library_provenance` 로 바꾸면 50/50 행이 정확히 `['g_dataset_flag','g_sym_class']` 에서 깨진다.
- **효과 크기 독립 재현:** S1j val 라벨 슬라이스 793행(ga80 577 + paramA 216, `fxy_head_prereg_20260829.md:749` 와 일치)에서 실제 `s1i` 앙상블로 raw F_r bias — **구 `library_provenance` −0.065512**(보고서 §6.2 와 소수 6자리까지 동일) → **`serve_provenance` +0.006665**(`predict_rows_raw` +0.006668). ga80 한정 −0.093938 → +0.010619, "store sym_class 만" +0.010566 — **ga80 `g_sym_class` 뒤집힘이 오프셋의 거의 전부**. (`s1j`: −0.069233 → +0.003756.)
- **서빙 진입점은 하나:** `model_api.py:72` import, `:992` `_record_inputs` 사용, 그것이 `_encode_batch`(`:1414`) → `_raw_forward_local` → `_ensemble_raw` → `predict`/`predict_extra`/`predict_convergence`/`predict_fxy`/`position_values` 의 유일 featurizer. remote 는 두 번째 featurizer 가 아니다 — `remote.py:485` 가 (canonical pattern, pair, feed, library_id) 만 보내고 `remote_infer.run_request` 가 같은 `_raw_forward_local` 을 호출하며 `featurize.py` sha 가 양쪽 `6977344dafbd770c9b1bc40e370db6c189320e301f8fa49570a25f927b575e36` 로 동일. `autoeng.py`/`scoping_mesh.py`/`lpopt/tools/*` 는 `library_provenance`·`RecordInputs`·`encoder.encode` 참조 **0**. `RecordInputs(` 생성은 트리 전체에 **3곳**(`model_api.py:996`, `policy/data.py:437`, `policy/scorer.py:274`).

### 5.4 반증 ① — 다른 서빙 경로가 **있다** (policy scorer v1/v2)

```python
lpopt/policy/data.py:356-359
    def corpus_provenance(lib):
        return serve_provenance(lib)[0], library_provenance(lib)[1]   # ← sym_class 절반은 여전히 구 맵

lpopt/policy/scorer.py:129
    class MoveScorer:  _provenance = staticmethod(corpus_provenance)   # MoveScorerV2 가 상속
```
실행 확인:
```
SCORERS → v1 {'ga80': ('P','free69'), 'paramA': ('P','rot61'), '260624': ('A','rot61')}
          v2 {'ga80': ('P','free69'), …}
          v3 {'ga80': ('P','rot61'), …}
corpus_provenance('ga80') == ('P','free69')   vs   serve_provenance('ga80') == ('P','rot61')
```
`get_scorer(...)` 의 시그니처 기본값이 `version="v1"`(`scorer.py:557`)이고 `search/construct.py:262` 가 v3 이 아닌 모든 `policy_prior` deck 모드에서 v1/v2 를 고른다. **오직 `MoveScorerV3`(`scorer.py:482-483`, `_provenance = provenance_v3 = serve_provenance`)만 깨끗한데, `data/models/policy_v3` 는 존재하지 않는다**(`runs/policy_v3/{metrics.json,probs.npz}` 뿐). 즉 **배포 가능한 정책 scorer 는 v1/v2 이고, 뒤집힌 맵이 오늘도 서빙 경로에 살아 있다.**

이는 **의도된 설계**이며 코퍼스와 자기정합적이다 — `data/policy/_feature_cache_v2.npz` 의 ga80 보드 1,391 중 **1,387 개가 `g_sym_class` 0.0**(예외 4건은 라이브러리 교차 패턴 충돌), `_feature_cache_v3.npz` 는 45,016 전부 1.0; `data.py:337-350` 도 *"a corpus defect, not a serving one"* 이라고 명시. **문제는 코드가 아니라 주장문이다 — C3 는 코드가 하는 일의 정반대를 단언했다.**

### 5.5 반증 ② — train/serve parity 가 **깨져 있고, 게이트가 이를 볼 수 없다** (실질 결함)

**저장소 시험이 지금 red 다:**
```
ssh … 'venv/bin/python -m pytest tests/test_featurize.py -q'  → 1 failed, 23 passed
tests/test_featurize.py:189  test_core_enrichment_split_reproduces_stored_e_core
    assert 4.871464678734045 == 4.892576507278004 ± 1e-9
```
원인은 `core_enrichment_split`(`lpopt/data/fuel_types.py:2000`) — `_record_inputs`(`model_api.py:993`)가 서빙 시각에 `e_core` 를 재구성하는 그 레시피 — 가 store 열을 재현하지 못한다는 것.

| 측정 | L1(`ad2eb5d3`) | L2(`a094544d`) |
|---|---|---|
| store-wide paramA 불일치 | **9.8 %** (max \|Δ\| **0.040936**) | **1,338 / 16,316** (그중 converged 1,138), max \|Δe_core\| **0.0684** |
| S1j-val 슬라이스 위반 | **188 / 216 paramA = 87 %** (793행 슬라이스 기준), worst \|Δ\| 0.02733 | paramA S1j-val converged 전수 **228 / 2,280**, 히스토그램 `{'g_e_core': 228}` |
| `e_split` | 0 불일치 | 0 불일치 |
| 타 라이브러리 | 260624 / 5.8_5.1 / ga80 / legacy_a 전부 정확 0 | 동일 |
| 예측 영향 | max\|serve − predict_rows_raw\| on F_r = **3.68e-4** (793행) vs 표본 50행에서는 0.0 | 228행에서 cyclen mean −0.0069 / max **0.137 EFPD**, cbc_max mean −0.0118 / max **0.557 ppm**, f_r max 0.0006 |

불일치는 feed 121/125/109 와 `fpcamp_minfr_T6T4*` / `fpcamp_minfr_hgd569_*` 캠페인에 집중. **이것은 `model_api.py:1000-1006` 의 docstring("`e_core` / `e_split` differ on **0 rows** — the 2026-08-29 e_core backfill aligned the store column with this recipe exactly")과 arm-2 보고서 §11 을 정면으로 반박한다.**

**게이트가 못 잡는 이유는 구조적이다:** `tests/test_model_api.py:463-464` 의 `test_serve_row_featurization_parity` 가 라이브러리당 `k.head(25)` 를 쓴다 — 무작위 표본도, 최악 사례도 아니다. 위반 228행 중 **head(25) 안에 든 것은 0건**. 즉 **228건(또는 188건)의 위반을 통과시키면서 계속 green 이었다.**

영향 크기 자체는 프로젝트 수용기준(cyclen ≤ 1 EFPD, CBC ≤ 1 ppm) **아래**다. 그러나 주장이 단언한 값은 0 이었고, 실제는 0 이 아니다.

### 5.6 반증 ③ — 나머지 (minor)

- **574 legacy ga80 행은 parity 밖.** store 는 `('B','free69')` 인데 serve 는 `('P','rot61')` 을 돌려준다. 라이브러리당 4,000행 표본 감사: `cfg=ga80` → `dataset_mismatch 117`, `sym_mismatch 117`(전부 store dataset='B'), 117/4000 ≒ 574/20,027. docstring 은 0.77 % 를 인정하지만 **parity 시험은 dataset='P' 행만 표본**하므로 게이트가 이를 볼 수 없다. `cfg=paramA` 는 0 불일치, `_effective_library != library_id` 는 0행.
- **remote 스크리닝에 소스 게이트가 없다.** `ensure_checkpoint`/`checkpoint_fingerprint`(`remote.py:420-445`)는 member meta + backend/calibration/ensemble json 만 해시하고 **소스는 명시적으로 제외**한다. `remote_infer.SCHEMA_VERSION` 은 여전히 **1** — 자체 주석이 *"so a stale remote install (older lpopt src) fails loudly"* 라고 말하는데도. 오늘은 동기 상태(`diff -q` 6개 파일 SAME, deck 31개 전부 `inference="local_cpu"`)라 **잠복 위험**이지, 활성 결함은 아니다.
- **stale 숫자:** `policy/data.py:337` 과 `scorer.py:249` 가 "2,401 paramA corpus rows" 를 단언하지만 현재 `steps.parquet` 은 **2,388**(총 28,084 · box 기준). ga80 3,930 / dataset=P 3,865 는 정확히 재현.
- **인용 오기:** 라이브러리별 분해(−0.0992/+0.0011/+0.0044/−0.0045/+0.0035)는 `fxy_head_results_arm2_20260829.md` 가 아니라 **`fxy_head_prereg_20260829.md:749`** 에 있다. arm-2 결과 문서에는 §6.2 의 −0.071912/+0.001965 쌍만 있다.

### 5.7 최종 판정 — **REFUTED**

핵심 메커니즘(서로게이트 서빙 경로의 provenance 수정)은 견고하게 재현되었지만, **주장의 범위 문장 두 개가 모두 거짓**이다. 권장 재서술:

> `featurize.serve_provenance` 는 **서로게이트 서빙 경로(`model_api._record_inputs`)와 policy v3 에 한해** provenance 결함을 닫는다. **policy scorer v1/v2 는 출하 체크포인트와의 train/serve 정합을 위해 `library_provenance` 유래 `sym_class`(ga80 → `free69`)를 의도적으로 유지**하며, 그 절반은 policy v3 이 승격될 때 비로소 닫힌다. **train/serve parity 는 `g_e_core` 에서 아직 깨져 있다**(paramA 약 9.8 %, S1j-val 슬라이스 188/216).

### 5.8 조치

| # | 조치 | 상태 |
|---|---|---|
| C3-1 | 주장문 범위 정정(위 재서술) — 보고서·메모 전반 | 미착수 |
| C3-2 | **`g_e_core` 파열 봉합**: paramA 신규분(feed 109/121/125, `fpcamp_minfr_T6T4*`/`hgd569`)에 e_core 백필 재실행, 또는 `_record_inputs` 가 store 행의 `e_core` 를 우선하고 미관측 패턴에만 레시피로 폴백. `model_api.py:1000-1006` docstring 의 "0 rows" 도 정정 | 미착수 |
| C3-3 | **게이트 대표성 확보**: `tests/test_model_api.py:463-464` 의 `k.head(25)` → 시드 무작위 표본 또는 S1j-val 전수. 추가로 store 전수 `core_enrichment_split(...) == e_core` 단언(1e-6) | 미착수 |
| C3-4 | 574 legacy ga80(dataset='B') 슬라이스를 parity 시험에 명시적 xfail/known-exception 으로 등록 | 미착수 |
| C3-5 | policy v3 승격 또는 명시적 유예: `runs/policy_v3/cnn_seed2026083*` → `data/models/policy_v3` 복사 후 deck/`construct._policy_pick` 기본값 전환, 아니면 잔존 `g_sym_class` 역전을 알리는 기동 경고 추가 | 미착수 (승격은 별도 판단 필요) |
| C3-6 | remote 소스 게이트: featurization 의미 변경 시 `remote_infer.SCHEMA_VERSION` 증가, 또는 `checkpoint_fingerprint` 형제로 `featurize.py`+`model_api.py` 해시 추가 | 미착수 |
| C3-7 | `policy/data.py:337`·`scorer.py:249` 의 2,401 → 2,388 정정, 분해 인용을 `fxy_head_prereg_20260829.md:749` 로 재지정 | 미착수 |

---

## 6. C4 — HGD569 결함의 범위는 정확히 160행 → **UPHELD**

### 6.1 주장 원문

> **CLAIM C4:** The HGD569 alias-no-op defect (`WaveVerifier` without resolver → raw type ids in `%LPD_SHF` → MASTER ran P6) affected **exactly the 160** `intervention_HGD569_f125` rows and nothing else … The 152 corrected rows are upgraded in place and the 8 stale rows are excluded from elites/training.

### 6.2 렌즈별 판정

| 렌즈 | 판정 | 한 줄 |
|---|---|---|
| L1 숫자 | **재현 (none, conf 0.92)** | store·deck·archive·physics 4중 확인, 전부 160 |
| L2 코드 | **재현 (minor)** | 14개 `WaveVerifier(` 호출부 전수 감사. resolver 없는 6곳 중 store 를 쓰는 것은 `ablation_wave` 뿐 |
| L3 출처 | **재현 (minor)** | 5,000+ deck 전수 스캔에서 다른 트리 0. 날짜 라벨·`--unconverge` 기록 불일치 발견 |

### 6.3 재현된 근거

**① 라벨 기반 범위.** paramA long-pair 행 **15,354**(메모 §5.1 과 정확 일치). generator 분해: `random 7,063 · heuristic 6,163 · elite_perturb 1,588 · local 197 · intervention_1move 160 · rule_biased 76 · g3_elite_boundary 48 · elite 44 · guided 15` → 15,354 − 160 = **15,194**("정상"). `ablation_1move`/`batchswap_enum`/`batchswap_enum_625`/`transfer` 의 long-pair 행 **0**. 라이브러리 전체 long-type 행 27,917 = `{paramA 15,354, 5.8_5.1 8,244, 260624 4,319}` 이며 legacy 12,563행은 generator 가 비어 있는 pre-lpopt 임포트분. 패턴 공간 교차검증(`pattern` 열 `[A-Z]\d{4}Z\d`) → 15,354행, **short `case_pair` + long pattern 인 행 0** → case_pair 필터에서 숨은 행 없음.

**② Deck 레벨 전수 census (라벨 독립 — 가장 강한 증거).** `%LPD_SHF` 의 `F <batch>` 카드에 2자 초과 토큰이 있는지로 스캔:

| 트리 | deck 수 | 히트 |
|---|---|---|
| `E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/intervention_wave_r1` | 1,647 | **160 파일**(`intervention_HGD569_f125/produce_cases/P6253Z1G06N24_P6253Z2G10N24_f125/` 아래에만) |
| `intervention_wave_r1_hgd569_fix` | 160 | 0 |
| `fpcamp_*` / `pinbu_*` (kit_frontier_runs) | 61×5 + … | 0 |
| `E:/lpopt_archive/199_runs` | 1,869 | 0 |
| `campaigns_20260829` | 458 | 0 |
| `runs_aug_20260829` | 1,853 | 0 |
| `3GA_runs_20260829` | 232 | 0 |
| `runs_july_20260829` | 20 | 0 |
| `D_lpopt_archive_199_runs` | 263 | 0 |
| `kit_pc2_v4_runs` | 23 | 0 |
| 라이브 `E:/lpopt_data/5_RL/runs` | 969 | **동일한 160 파일** |

> 두 렌즈가 서로 다른 정규식을 써서 **파일 수는 160 으로 동일**하나 occurrence 수는 다르게 셌다(L1 `1,440` = 파일당 9, L3 `633`, 패턴 `^\s*F [A-Z][A-Z0-9]{2,}`). 판정에 쓰이는 것은 **파일 수 160** 이며 여기서 두 렌즈가 일치한다.

**단일 변수 증명** — 같은 패턴 해시 `00822755d6ef9de5` 의 두 실행:
```
r1  line 33: F P6253Z1G06N24  0, 1 L 14 2, F P6253Z1G06N24  0, 1 R 12 2, …
fix line 33: F S3             0, 1 L 14 2, F S3             0, 1 R 12 2, …      (shuffle 카드 byte-identical)
```
**MASTER 자신의 에코(양성 대조):** r1 `MAS_OUT` → `"P6 P6 P6 P6 P6 P6 P6 P6 P6 R1"`(전량 P6 노심), fix → `"S5 S3 S3 S3 S3 S3 S3 S5 S5 R1"`(설계 노심 복원), `fpcamp_minfr_triple_f125` → S3/S5/T1 3종 혼합. **null-glob 아티팩트가 아님이 증명된다.**

**③ 코드 경로 감사(메모 표를 확장).** `WaveVerifier(` 호출 14곳 —
`resolver=` **전달**: `campaign.py:1331`, `campaign.py:3305`, `produce.py:472/497`(다른 렌즈 표기 480/509), `curriculum.py:1045`, `curriculum.py:1112`, `transpose_pairs.py:193`, `intervention_wave.py:1161`.
`resolver=` **부재**: `ablation_wave.py:707`, `fr_arms.py:188`, `fr_transfer.py:522`, `rule_acid_run.py:362/386`, `v520_run.py:593/622`, `design/pathfinder.py:268`.
이 6곳 중 **store 행을 쓰는 것은 `ablation_wave` 뿐**(`:809` `StoreWriter`, `:864-865` `write_records`). `fr_arms`/`fr_transfer`/`rule_acid_run`/`v520_run` 은 `records.parquet` 을 **읽기만** 하고, `pathfinder` 는 구조적으로 면역(`pathfinder.py:216` 이 `pair` 를 alias 로 조립). **메모 표에서 빠져 있던 `curriculum.py:1045/1112`·`transpose_pairs.py:193` 을 두 렌즈가 각각 읽어 확인했고, 셋 다 resolver 를 넘긴다** — 누락이 미등록 결함 경로를 만들지는 않았다.
store 교차검증(generator 별 case_pair 토큰 최대 길이): `ablation_1move 2/2`(150행) · `batchswap_enum 2/2`(213) · `batchswap_enum_625 2/2`(220) · `transfer 2/2`(392) — **long pair 를 가진 generator 는 `intervention_1move` 뿐이며 800행 중 160행**.
가장 날카로운 공격선도 무력화됐다: `rule_biased` 의 paramA long-pair 76행은 `fxy_rule_S3T1S5_f125`(42) + `fxy_rule_e585_f125`(34) 캠페인 = `produce.py` 산출물이지 `rule_acid_run.py`(아무것도 안 씀) 산출물이 아니다.

**④ 물리 대역 시험(결과 측 확인).** `case_pair P6253Z1G06N24_P6253Z2G10N24` 의 store 행 전체(393~480행, 9개 캠페인) 중 **`cyclen > 750 EFPD` 인 행은 정확히 1건** — 잔존 stale 행 `4cbfb4f6…`, 763.934. 대역:

| 캠페인 | cyclen 대역 (EFPD) |
|---|---|
| `fpcamp_minfr_hgd569_f109` | 639.3 – 676.5 |
| `fpcamp_minfr_hgd569_f125` | **709.051 – 736.612** |
| `_seedctl` | 712.775 – 734.932 |
| `fxy_exp_HGD569_f109` | 631.2 – 676.0 |
| `fxy_exp_HGD569_f125` | **707.972 – 741.499** |
| `mv3_e569_f109` / `f125` | 634.4 – 665.9 / 711.927 – 737.465 |
| tripletype 5개 캠페인 | 697.612 – 736.919 |
| **`intervention_HGD569_f125_v2`** | **727.477 – 738.084** (mean 731.238, conv 152/152) |
| `intervention_HGD569_f125` (stale) | **763.934** (conv 0/8) |

결함 대역(762–771)은 격리행 밖 어디에도 없다.

**⑤ In-place 갱신과 격리.** 사전 백업 대비: record_id 집합 **동일**(160, 대칭차 0), store 행 수 76,693 불변 → **추가 append 가 아니라 제자리 갱신**. 152행 cyclen 761.639–770.597 → 727.477–738.084 (max \|Δ\| **38.440**), 사후 `converged=True valid=True failure=""` 152/152, restart `pair_ecore:MAS_RST.APRQ_11_0705.02`. 8행은 `converged=False valid=False failure="alias_noop_P6_20260830"`(store 전체에서 이 라벨 8행뿐), 그중 7행은 FOM 전부 NaN, `4cbfb4f682229f97…` 만 `cyclen 763.934 / f_r 1.6983 / f_xy 1.7719 / node_peak 1.3818` 보유.
`steps.parquet`(28,889×85, sha `2a34f7b38d83390f…`, 사전 백업 `5fa791b8b3197179…`): `lineage_source intervention_HGD569_f125` = 152, 8 stale id 는 parent·child 어디에도 **0회**. `steps_v3.parquet`(28,889×107)도 동일. `S1i.json`/`S1j.json`(Aug 29, wave 이전)에 160 id **0건**, "HGD569"/"intervention" 문자열도 0.
코드 측 수정 확인: `verify.py:861` 이 fallback 을 `package_root`+`library_dims` 에서 유도(주석에 *"memo 20260830 §3 — 160 chains"* 명기), `assets.py:296/397-433` `validate_reload_deck` 의 `allowed_batch_ids` 로스터 게이트(실패 문구 `:441`), `tests/test_deck_alias_guard.py` 존재.

### 6.4 반증에 이르지 못한 경고

- **(a) 격리가 "출처 불명 상태"에 의존한다 (severity 를 minor 로 만든 이유).** `E:/lpopt_archive/199_runs_20260830/…/intervention_wave_r1/intervention_HGD569_f125/ablation_results.jsonl` 은 지금도 `"record_id": "4cbfb4f682229f97…", "cyclen": 763.934, "converged": true` 를 담고 있다. `ablation_wave.py kit` 은 이 jsonl 에서 store 행을 재구성하고, merge-store 는 `_quality_rank = converged*8 + valid*4 + flat*2 + fxy` 로 순위를 매긴다. **r1 디렉터리를 re-kit 하면 rank 12 의 converged 행이 rank 0 의 격리행을 밀어내고 763.934 를 조용히 복원한다.**
- **(b) 날짜 라벨과 산출물 mtime 불일치.** "AMENDMENT 2026-08-31"/"AMENDMENT 2 (HGD569 v2) 2026-08-31" 이 표제인 작업의 산출물은 전부 **9월 2일**이다: fix `ablation_results.jsonl` 09:40 · `fxy_backfill_199_intervention_hgd569_fix_20260831.csv` 09:39 · `intervention_wave_r1_rows_v2.csv` 09:56 · `records.parquet` 09:45 · `steps.parquet` 09:56 · 보고서 `.md` 10:08 · `steps_v3` 11:56.
- **(c) 문서화된 프로세스가 디스크와 모순.** AMENDMENT 1 §C 는 *"본 실행에서는 지시대로 `--unconverge` 를 쓰지 않았다 — 153행은 여전히 `converged=True` 다"* 라고 쓰고, AMENDMENT 2 §I 는 어느 단계가 이 열을 내렸는지 규명하지 못했다고 적었다. **백업이 이를 판정한다:** `records.parquet.bak_pre_hgd569_unconverge_20260830`(Aug 30 23:23)은 153 `converged=True/valid=False` + 7 F/F, `..._unconverge2_20260830`(23:00)도 존재, `bak_pre_hgd569_fix_20260831`(Sep 2 09:42)은 **160행 전부 `converged=False`**. → **`--unconverge` 패스가 8월 30일에 실제로 실행됐다.** 방향은 안전(격리 강화)이라 C4 의 마지막 절은 오히려 **과소 주장**이다.
- **(d) 위험 창.** Aug 30 21:13(r1 merge) → 약 23:00(unconverge) 사이 **153개 오염행이 `converged=True` 상태로** 트리의 converged-only 엘리트/학습 필터에 노출돼 있었다. 다만 그 창에 걸치는 모델 산출물은 없다(`s1j` Aug 30 03:45, `policy_v3` Sep 2 13:08).
- **(e) deck 증거가 없는 하위 주장 1건.** `produce_fxyera_r1` 의 deck 은 삭제됐다(`runs/produce_fxyera_r1` 과 아카이브 양쪽에서 `*.inp` 0건). "fxy_* HGD569 stratum 은 깨끗하다"는 절은 **코드 경로(`produce.py` 가 resolver 를 넘김) + FOM 대역**의 간접 증거에만 의존한다.
- **(f) 메모의 R2 권고는 미이행.** `ablation_wave.py:707` 은 여전히 `resolver=` 를 넘기지 않는다 — 수정은 fallback 을 강화하는 쪽으로 이뤄졌다. 기능적으로는 fix 실행 deck 으로 검증됐지만, 권고문 자체는 미완.
- **(g) 표현 정밀화.** 현재 store 에서 `campaign=='intervention_HGD569_f125'` 는 stale 8행뿐이고 152행은 `..._f125_v2` 다. **"160행"은 `record_id` 로만 잘 정의된다**(불변임을 확인).

### 6.5 최종 판정 — **UPHELD**

deck 레벨 전수 census(다른 트리 5,000+ deck 에서 0 히트), MASTER 에코 양성 대조, generator census, 물리 대역 시험, record_id 집합 불변 — **네 겹의 독립 증거가 모두 160 을 가리킨다.**

### 6.6 조치

| # | 조치 | 상태 |
|---|---|---|
| C4-1 | **merge 경로에 격리 denylist**: 기존 행이 `failure` 라벨 `alias_noop_*` 를 달고 있으면 StoreWriter/merge-store 가 갱신을 거부(더 높은 접미사 `_v2`/`_v3` 캠페인 태그만 예외). 없으면 re-kit 이 763.934 를 복원 | 미착수 |
| C4-2 | `campaign.py:1373`·`:1401` 엘리트/시드 필터에 `& (valid == True)` 추가(`:3607`·`:1526` 은 이미 그러함) → `valid=False` 만으로 배제가 성립하게 | **진행 중** (C5 의 `store.trustworthy()` 정규화 작업에 포함) |
| C4-3 | AMENDMENT 1 §C·AMENDMENT 2 §I 정정: `--unconverge` 는 2026-08-30 23:00–23:23 에 실행되었다(백업이 증거) | 미착수 |
| C4-4 | AMENDMENT 날짜 라벨 2026-08-31 → 실제 실행일 2026-09-02 로 정정(`fxy_backfill_..._20260831.csv` 포함) | 미착수 |
| C4-5 | 메모 §3 호출부 표에 `curriculum.py:1045/1112`·`transpose_pairs.py:193`·`rule_acid_run.py:362`·`v520_run.py:593` 추가, `pathfinder.py:268` 이 구조적 면역인 이유(`:216`) 명기 | 미착수 |
| C4-6 | R2 완결: `ablation_wave.py:707` 이 `cmd_run` 이 이미 만든 resolver 를 `WaveVerifier(resolver=…)` 로 넘기게 (강화된 fallback 은 다중 방어로 남김) | 미착수 |
| C4-7 | `produce_fxyera_r1` 증거 공백 해소: stratum 당 deck 1개 보존 또는 방출된 `%LPD_SHF` fresh 토큰을 run jsonl 에 기록 | 미착수 |
| C4-8 | `4cbfb4f682229f97…` 의 FOM 열(`cyclen/f_r/f_xy/node_peak/maps_key`) null 처리 검토 | 미착수 (C5-2 와 중복 대응 가능) |

---

## 7. C5 — 격리/무효행은 누출될 수 없다 → **REFUTED (3/3, 전부 major)**

### 7.1 주장 원문

> **CLAIM C5:** Quarantined/invalid rows **cannot leak** into surrogate training, elite pools, or the policy corpus: training (`train.py`, `dataset_torch.py`, `curriculum.py`), elite selection (`campaign.py _store_elites/_elite_seed_rows`, `produce.py`), pinbu/intervention planning, and **`mine_policy_corpus.build_steps` all exclude `converged=False` rows** … `steps.parquet` (28,889) and `steps_v3.parquet` contain **no edges from quarantined children**.

### 7.2 재현된 절반 (이것뿐이다)

- store 76,693행 교차표: `converged×valid` → **T/T 67,781 · F/T 152 · F/F 8,760 · T/F 0**. 즉 `valid=False` 는 오늘 `converged=False` 의 진부분집합이다.
- `campaign=="intervention_HGD569_f125"` = **정확히 8행**, 전부 `valid=False, converged=False`, `failure="alias_noop_P6_20260830"`. `steps.parquet`/`steps_v3.parquet` 에서 이 **8개 id 는 parent·child 모두 0회**.
- 엘리트 선택과 pinbu/intervention 계획은 실제로 거른다: `campaign.py:1367-1373`·`:1402`·`:1419`, `produce.py:646/679`, `mine_policy_corpus.build_elites`(`:889-897`→`feasibility()`, `:617`); `pinbu_wave.py:175`(`valid and converged`)·`:298`; `intervention_wave.py:321`.

### 7.3 반증 ① — `build_steps` 에는 필터가 **없다**, 그리고 steps 파일에 무효 자식 edge 가 **775건** 있다

`mine_policy_corpus.py:672-703` `lineage_edges` 는 `parent_record_id.notna()` + store 멤버십 + self-loop 배제만 필터한다. `build_steps`(`:706-885`, `:721-722`)는 그것을 그대로 받아 **`both_converged` 열을 덧붙일 뿐**(`:793-798`, `:856`, `:872`). `main()`(`:1954`)은 `pd.read_parquet(args.store)` 로 store 전체를 넘긴다. 모듈 자신의 리포트 문구가 이를 시인한다(`:1562` *"the remainder has a non-converged endpoint …"*).

측정(로컬 현행 파일 기준):

| 파일 | 행수 | `child_converged=False` | `parent_converged=False` | **`valid=False` 자식** | `valid=False` 부모 |
|---|---|---|---|---|---|
| `steps.parquet` (sha `2A34F7B38D83390F…`) | 28,889 × 85 | 786 | 4 | **775** | 3–4 |
| `steps_v3.parquet` (sha `100EE50ED5C75725…`) | 28,889 × 107 | 786 | 4 | **775** | 3–4 |

> 부모 측 수치는 렌즈 간 3 과 4 로 갈렸다(L1·L3 은 4, L2 는 3). 미조정 상태로 기록한다 — 자식 측 **775** 는 세 렌즈 모두 동일하다.

775건의 lineage 출처: `{lpopt_genome 767, ablation_paramA 4, batchswap_enum_625 2, intervention_T6T4_f121 1, intervention_N1N2_f113 1}` — **그중 2건은 정책 v3 게이트가 읽는 바로 그 개입(interventional) 층이다.**

**하류 완화는 존재하나 주장이 말한 것과 다르다:** `lpopt/policy/v3.py:141,169` 와 `train_v3.py:385` 는 라벨을 `both_converged` 로 마스킹한다. 그 게이트를 통과한 누출 실측: `load_universe_v3("data/policy/steps_v3.parquet")` → 21,134행, 그중 **2행이 `both_converged=False`** 이고 **둘 다 격리 자식 + `d_f_xy` 보유** — `d_f_xy.notna()` 갈래(`v3.py:216-219`)로 들어온다(`d_f_xy` 는 수렴 게이트 없이 계산되는 자식−부모 차이). loss mask 는 0 이라 gradient 영향은 없지만 **텐서 안에 있고 피처 정규화 모집단에 포함된다.**

**재구축 위험이 더 크다:** 격리 8행은 전부 store 안에 있는 `parent_record_id` 를 갖고 self-loop 도 아니다. 현행 76,693행 store 로 `lineage_edges` 를 전면 재구축하면 **`lpopt_genome` edge 7,402건이 나오고 그중 8건이 격리 자식**(그리고 809 converged=False / 797 valid=False 자식). **현재의 부재는 가드가 아니라 append 아티팩트다** — `quarantine_campaign` 이 캠페인 태그로 r1 edge 를 지웠고 v2 재-append 가 `intervention_HGD569_f125_v2` 로만 범위 지정됐기 때문. 게다가 `drop_steps` 는 `--steps` 로 넘긴 **단일 파일만** 건드린다(`steps_v3.parquet` 이 깨끗한 것은 그 뒤에 재구축됐기 때문 — mtime Sep 2 11:56).

보고서 `line 791` 이 "+152 not +160" 의 이유로 적은 *"격리된 8행이 store 에서 `converged=False` 이므로 `build_steps` 가 edge 를 만들지 않는다"* 는 **틀렸다.**

### 7.4 반증 ② — 서로게이트 학습은 `converged=False` 행을 **배제하지 않고 학습한다**

`S1j.json` 을 현행 store 에 대해 재계산:

```
train_ids n = 62,575  →  converged=False 8,518 (13.6%),  valid=False 8,377 (13.4%)
val_ids   n = 12,142  →  converged=False    63,          valid=False    56
```

이는 **의도된 설계**이지만 주장과 정반대다:
- `model/splits.py:321-324` — *"Everything else in the cell (… plus **ALL NON-CONVERGED ROWS**) goes to TRAIN"*, `:365` 는 `converged` 만 본다.
- `dataset_torch.py:174-179` — `self.df` 는 `_resolve_ids(split_manifest, …)` 로만 만들어지며 수렴·유효 필터가 없다. `:186-212` 는 회귀 `target_mask` 만 0 으로 만든다. `:293-294` — `conv_label = 1.0 if row["converged"] else 0.0`, `conv_mask = 0.0` 은 `converged_at_cap` 일 때만.
- `train.py:1892-1894` — `convergence_loss(out["conv_logit"], batch["conv_label"], batch["conv_mask"])`, `cfg.conv_weight` 기본 1.0(`train.py:105`). **격리 8행은 전부 `converged_at_cap=False` → 마스크되지 않은 음성 샘플**이며, 그 cells/globals 가 공유 stem/blocks/films/head_trunk 를 통과한다.
- 두 번째 학습 경로도 동일: `model_sklearn.py:54-56` docstring(*"Non-converged rows with finite last-iterate values are kept…"*), `:83` `converged=bool(_g("converged"))`, `valid` 검사 없음.
- `train.py` 안의 유일한 `converged` 필터는 `:1050-1051`(power-prior 적합 헬퍼)로 데이터셋 경로가 아니다.

**게다가 이것은 잘못 라벨된 음성이다.** `records.parquet.bak_pre_hgd569_unconverge_20260830` 을 읽으면 `4cbfb4f682229f97…` 는 flip 이전에 **`converged=True`, `cyclen=763.934`** 였다. → **MASTER 가 실제로 수렴시킨 노심을 "수렴하지 않는다"고 모델에 가르치게 된다.**

`splits.py:441-448` 의 "QUARANTINE" 은 **미도달 커리큘럼 셀**을 뜻하는 완전히 다른 개념이다. 주장은 두 개념을 혼동했다.

### 7.5 반증 ③ — 맵 헤드가 열려 있다 (**차단(blocking) 급**)

```python
lpopt/model/dataset_torch.py:215-230   _maps → mask = np.isfinite(maps)     # converged/valid 검사 없음
lpopt/model/dataset_torch.py:245       _axial → bool(row["converged"]) 로 게이트   # 형제는 게이트한다
lpopt/model/dataset_torch.py:271       traj  → bool(row["converged"]) 로 게이트
```
`PosValDataset.df` 는 manifest id 로만 만들어지고(`:178`), `map_loss` 는 `train.py:1888` 에서 `cfg.map_lambda`(`:1894`)로 가중되며 `map_fr_consistency_loss` 가 `:1938` 에 있다.

**현행 store 에서** 격리행 `4cbfb4f682229f97…` 는 `maps_key = 4cbfb4f68…`, `node_peak 1.3818`, `f_q 2.1388`, `cbc_max 1795.84` 를 갖는다 — **alias-no-op P6 단일조성 노심에서 수확된 맵**이다. `maps_key` 를 가진 미수렴 행은 현행 store 에 **4건**, `S1j` 가 만들어진 Aug-29 store 에는 **0건**. 즉 **다음 split/재학습이 그 결함 맵을 맵 헤드에 넣는다.** 보고서 §I.2 의 *"이 트리의 표준 필터(`converged == True`)에 걸러진다 … 구멍은 이 행에 대해서는 닫혀 있다"* 는 **맵 헤드에 대해 거짓**이다.

### 7.6 반증 ④ — 불변식이 아니라 운영자 습관이다

모든 필터가 **`converged == True` 하나만** 본다: `campaign.py:1373`(`_case_store_rows`)·`:1401`(`_elite_seed_rows`)·`:3607`(`_verified_idx`); `produce.py:538/646/679`; `intervention_wave.py:321`(`joint_clean`); `splits.py:365`; `physics_prior.py:345/373/458`; `folds.py:198-199`; `curriculum.py:652/797/1863`; `conformal.py:387`; `c2_slice.py:379-380`; `pinbu_physics.py:457-458`. **`valid` 를 함께 보는 곳은 `campaign.py:1526`(`_replay_rows`) 단 하나**이고, 제대로 하는 호출자는 `pinbu_wave.py:175/298` 뿐이다.

격리 도구 자신의 docstring 이 최고의 반증 근거다 — `lpopt/tools/quarantine_campaign.py:40-49`:
> *"`valid=False` alone does NOT remove a row from the search: … the elite pools, replay/holdout draws and **every model training filter key on `converged == True` and do not look at `valid` at all** … A quarantined-but-converged row therefore still seeds elites and still trains surrogates."*

그리고 이 도구는 **`--unconverge` 없이** 실행됐다(보고서 line 526). 실제로 위반이 일어났다: 백업 `bak_pre_hgd569_unconverge_20260830` 시점에 캠페인 160행은 **153 `converged=True & valid=False` + 7 F/F** 였다 — 즉 격리 시각부터 unconverge 시각까지 **153개 알려진 불량행이 모든 엘리트 풀과 모든 학습 필터에 완전히 살아 있었다.**

오늘 `converged=True & valid=False` 가 0행인 것은 **MASTER 미수렴이라는 우연**이지 설계가 아니다.

### 7.7 최종 판정 — **REFUTED**

주장이 지목한 네 경로 중 두 곳(`dataset_torch`/`train.py`, `mine_policy_corpus.build_steps`)이 주장의 **정반대**로 동작하고, steps 파일 하위 주장은 일반 무효행에 대해 거짓(775건)이며, 엘리트 경로는 `valid` 를 아예 보지 않는다. 재현된 것은 8행 카운트와 "오늘 `invalid∧converged == 0`" 이라는 우연뿐이다.

권장 재서술(검증 가능한 형태):
> **미수렴 행은 train fold 에 보존되어(`S1j` 62,575 중 8,518) 수렴 헤드의 지도 신호로 쓰인다. `dataset_torch._targets`/`_maps` 에서 회귀·맵 타깃이 0 가중으로 마스킹되어야 하며, 그때에만 비물리 값이 회귀 손실에 도달하지 않는다** — 그런데 `_maps` 는 현재 마스킹하지 않는다(§7.5). 정책 코퍼스는 무효 끝점을 **보존**하고 `both_converged` 로 마스킹할 뿐이다.

### 7.8 조치

| # | 조치 | 상태 |
|---|---|---|
| C5-1 | **정규 술어 도입**: `lpopt/data/store.py` 에 `trustworthy(df) = converged.fillna(False) & valid.fillna(True)` 를 만들고 12곳 이상의 open-coded `df["converged"] == True` 를 전부 경유시킴(`campaign.py:1373/1401/3607`, `produce.py:538/646/679`, `intervention_wave.py:321`, `splits.py:365` …). `valid=False` 행이 엘리트 풀·train fold 에 도달하지 못함을 단언하는 시험 추가 | **진행 중** |
| C5-2 | **`_maps` 게이팅 (차단 급)**: `dataset_torch.py:_maps`(~219) 를 `_axial:245` 와 동일하게 — `if not (bool(row["converged"]) and bool(row.get("valid", True))): return NaN, zeros`. 없으면 다음 재학습에서 `4cbfb4f6…` 의 P6 맵이 `map_loss`(`train.py:1888`)로 들어간다 | **진행 중** |
| C5-3 | **코퍼스 원천 차단**: `mine_policy_corpus.lineage_edges`(`:681-703`)에서 부모/자식이 `valid=False` 인 edge 제거, `both_converged` 옆에 `both_valid` 열 방출(`build_steps:856,872`), `policy/v3.py:141,169`·`v2.py:126` 에서 마스킹. **주의:** `intervention_wave.cmd_corpus` 의 schema-drift 가드 때문에 열 추가는 80→85 열 때와 같은 마이그레이션 경로가 필요 — `steps.parquet` 과 `steps_v3.parquet` 을 한 번에 백업·갱신 | **진행 중** |
| C5-4 | 불변식을 **검사 가능**하게: `set(steps.parent ∪ steps.child) ∩ set(records[~valid].record_id) == ∅` 를 라이브 산출물에 대해 단언하는 시험. **현재 775 / 3–4 로 실패한다** | 미착수 |
| C5-5 | `quarantine_campaign.drop_steps` 가 모든 steps 산출물(`steps.parquet` **및** `steps_v3.parquet`)을 쓸도록 확장 | 미착수 |
| C5-6 | 1–5 가 들어가기 전까지 `quarantine_campaign.py` 의 `--unconverge` 를 **기본값**으로 하거나 `--no-unconverge` 없이는 `--apply` 를 거부하도록 (도구 docstring 이 이미 `valid=False` 단독은 무력하다고 시인) | 미착수 |
| C5-7 | 기록 정정: 보고서 line 791 의 사유("`build_steps` 가 edge 를 만들지 않는다")는 거짓 — 실제 사유는 캠페인 태그 드롭 + `_v2` 범위 재-append. §I.2 의 "닫혀 있다" 는 스칼라 타깃으로 한정해야 하며 맵 헤드는 열려 있다 | 미착수 |
| C5-8 | 학습 조항 재서술(§7.7) — 현행 문구는 검토자에게 거짓 안심을 준다 | 미착수 |

---

## 8. HOST_238 stale-store 위생 발견 (감사 부산물이자 이번 감사의 최대 프로세스 교훈)

### 8.1 사실

**14개 렌즈 중 최소 8개가 독립적으로 같은 것을 보고했다:** 이 감사에 넘겨진 전제 *"the box holds … `data/store/records.parquet` (76,693 rows, same as local)"* 는 **거짓**이었다.

| 산출물 | HOST_238 (`~/lpopt_ws/`) | 로컬 (권위본, 감사 시점) |
|---|---|---|
| `data/store/records.parquet` | **74,717행** · 22,315,679 B · **Aug 29 13:20** · sha256 `cf495c7d82b16cbfe4216333ca4d266a324514c223bd7e0a2c38f799445326cc` | **76,693행** · 22,782,850 B · Sep 2 09:45 · sha256 `747d37ae2a50cc25742f98cc4770694354032ec6e060eeb34d54b75eb830cb50` |
| `data/policy/steps.parquet` | **28,084행** · 9,257,478 B · **Aug 17** | **28,889행** · 9,556,122 B · Sep 2 09:56 · sha `2a34f7b38d83390f…` |
| `data/policy/steps_v3.parquet` | 28,889행 · Sep 2 — **동기** | 28,889행 |
| `data/models/20260902_122746` (arm 4) | **부재** (렌즈 1건 보고) | 존재 |

box store 에는 `campaign` 이 HGD569 인 행 **0건**, `failure=="alias_noop_P6_20260830"` **0건**, `record_id bf3a70b2…` **nmatch 0** — **2026-08-30 격리 이전 스냅샷**이다.

### 8.2 왜 위험한가 — **공허한 확인(vacuous confirmation)**

C4·C5 를 box 의 자체 store 로 검증했다면 결과는 이렇다: *격리행이 존재하지 않으므로 "격리행은 누출되지 않았다"가 자명하게 참.* **참인 주장을 조용히 반증하거나(C1: `nmatch 0` → "레코드가 없다"), 거짓인 주장을 조용히 확인하는(C5) 두 방향의 오류가 모두 가능하다.** 실제로 C1 렌즈 하나는 이 함정을 명시적으로 기록했다: *"future verification runs must use `~/lpopt_ws/scratch/c4/records.parquet` … or they will silently 'refute' true claims."*

**두 번째 차수의 오염:** 누출 건수 자체가 어느 `steps.parquet` 을 쓰느냐에 달라진다 — 로컬 28,889행 파일에서 **775**, box 의 28,084행 파일에서 **772**. 감사 숫자가 파일 버전의 함수인 상태였다.

### 8.3 렌즈들이 쓴 우회

대부분의 렌즈가 동일한 패턴을 택했다:
```
scp -P 8022 data/store/records.parquet USER@HOST_238:~/lpopt_ws/scratch/records_local_20260902.parquet
ssh -p 8022 USER@HOST_238 'cd ~/lpopt_ws && sha256sum scratch/records_local_20260902.parquet'
   → 747d37ae2a50cc25742f98cc4770694354032ec6e060eeb34d54b75eb830cb50   (도착 후 sha 재검증)
```
`steps.parquet`, `steps_v3.parquet`, 그리고 E: 드라이브 백업(`records.parquet.bak_pre_hgd569_unconverge_20260830`, `..._fix_20260831`)도 같은 식으로 올려 대조했다. **감사가 성립한 것은 렌즈들이 전제를 의심했기 때문이지 인프라가 옳았기 때문이 아니다.**

### 8.4 부수 발견

- **`S1j.json` 은 격리 이전 store 에 묶여 있다.** `groups.records_sha256 = cf495c7d82b16cbf…`(= box 의 74,717행 스냅샷). **좋은 소식:** `s1i`/`s1j` 어느 것도 HGD569 오염분으로 학습되지 않았다. **나쁜 소식:** 따라서 **`S1j` 는 격리 가드가 작동한다는 증거가 될 수 없다** — 단지 그 행들이 존재하기 전에 만들어졌을 뿐이다.
- **box 환경이 감사 중 변형됐다.** 렌즈 하나가 `venv/bin/pip install pytest`(pytest 9.1.1)를 실행하고 `~/lpopt_ws/tests` 에 시험 트리를 내려놓았다(`REPO_ROOT` 해석용). 본인이 명시 고지했으나, **238 의 venv 는 더 이상 원상태가 아니다.**
- **오늘(2026-09-03) 기준 추가 노후화.** 이 감사가 "현행"으로 삼은 76,693행 스냅샷조차 이미 갱신됐다 — `~/lpopt_ws/scratch/records_r2_76793.parquet`(**76,793행**, 2026-09-02 20:33)이 `fpcamp_minfxy_e1e2_f121_r2` 의 신규 100행을 포함한다. **§7.3 의 775 같은 숫자를 재현하려면 어느 스냅샷인지 반드시 명시해야 한다.**

### 8.5 조치

| # | 조치 | 상태 |
|---|---|---|
| H-1 | 238 의 `data/store/records.parquet`·`data/policy/steps.parquet` 재동기(또는 "stale mirror" 로 명시 표시) | **진행 중** (별도 수정 워크플로) |
| H-2 | `data/splits/S1j.json` 이 `cf495c7d…`(74,717행, 격리 이전)에 고정되어 있으며 **HGD569 봉쇄에 대해 아무 말도 할 수 없음**을 메모리 다이제스트에 기록 | 미착수 |
| H-3 | **규칙화**: 앞으로 모든 store 기반 주장은 `records.parquet` 의 sha256 과 행수를 함께 적는다. 이번 라운드에서 렌즈들이 이미 사실상 이 규칙을 강제했다(§8.3) | 미착수 (규칙 채택 필요) |
| H-4 | 감사·실험 실행 전 box 산출물 신선도 확인을 표준 프리앰블에 넣기(`ls -l` + `sha256sum` 3파일) | 미착수 |
| H-5 | 238 venv 에 설치된 `pytest` 와 `~/lpopt_ws/tests` 트리를 정리하거나, 정식 감사 환경으로 승격해 기록 | 미착수 |

---

## 9. 조치 통합 — 수정 워크플로 진행 상태

**별도 수정 워크플로가 이미 구현 중인 항목(진행 중, 재기획 금지):**

1. **`store.trustworthy()` 정규 술어** 도입 및 전 필터 경유 — C5-1, C4-2 를 흡수
2. **`dataset_torch._maps` 게이팅** — C5-2 (차단 급)
3. **정책 코퍼스 drop** (`lineage_edges`/`build_steps` 에서 무효 끝점 제거) — C5-3
4. **`PROMOTION.md` 재스탬프** (§3.2 R1 해시 + §2 D3 등록) — C2-1, C2-2
5. **238 store 갱신** — H-1

**미착수 항목 중 우선순위 상위 5개 (이 감사가 새로 드러낸 것들):**

| 순위 | 항목 | 근거 | 성격 |
|---|---|---|---|
| 1 | **C3-2 `g_e_core` train/serve 파열** — paramA 약 9.8 %, S1j-val 188/216, 저장소 시험이 **지금 red** | §5.5 | 실질 결함 (수용기준 이하지만 0 이 아님) |
| 2 | **C3-3 게이트 대표성** — `k.head(25)` 가 228건 위반을 통과시킴 | §5.5 | 검출 능력 부재 (1번을 영구히 은폐) |
| 3 | **C2-4 per-cell 보정 라운드트립 누락** — `--resume` 이 수준 보정 없는 모델을 서빙 | §4.4 D | D3 계열, 가드 전무 |
| 4 | **C4-1 merge denylist** — r1 re-kit 이 763.934 를 rank 12 로 복원 가능 | §6.4(a) | 격리 취소 위험 |
| 5 | **C5-4 불변식 시험** — 현재 775/3–4 로 실패 | §7.8 | 회귀 방지 |

---

## 10. 이 검증 자체의 한계 (정직하게)

1. **C2 는 2렌즈 판정이다** (§2.2). "3렌즈 통과"로 인용 금지.
2. **렌즈 간 미조정 수치 2건**: steps 파일의 `valid=False` **부모** edge 수(3 vs 4), C4 deck 스캔의 occurrence 수(1,440 vs 633 — 정규식 차이, **파일 수 160 은 일치**). 판정을 바꾸지 않으므로 조정하지 않고 기록만 한다.
3. **스냅샷 고정 부재**: 감사는 2026-09-02 09:45 의 76,693행 store 를 "현행"으로 썼다. 오늘 기준으로 이미 76,793행 스냅샷이 있다(§8.4). 숫자 재현 시 스냅샷을 반드시 명시할 것.
4. **`produce_fxyera_r1` deck 증거 공백**(§6.4(e))과 **E: 드라이브 구형 트리 전수 미검**(frontier/fill/mv3/debug_panel — 광역 `rg` 가 20 s 에서 타임아웃)은 C4 에서 간접 증거로만 덮였다.
5. **렌즈는 전부 "반증하라"는 적대적 지시를 받았다.** 따라서 C1·C4 의 유지 판정은 강하게 읽어도 되지만, C3·C5 의 반증은 *"주장문이 코드보다 넓다"* 는 형태의 반증이 대부분이며 **코드가 위험하게 잘못됐다는 주장은 §7.5(맵 헤드) 하나뿐**이다. 그 하나가 유일하게 "차단 급"으로 표시된 이유다.

---

## 부록 — 2026-09-03 정합 갱신

**성격.** 본문 §1–§10 은 **2026-09-02 실행 · 2026-09-03 작성** 시점의 기록이며 **한 글자도 수정하지 않는다.** 이 부록은 그 이후 별도 수정 워크플로(2026-09-03)가 실제로 닫은 항목을 **항목별로 대조**하고, 본문이 "미착수"·"진행 중"·"지금 red" 로 적어 둔 문장 중 **오늘 기준 거짓이 된 것**을 지목한다. §1 표의 판정(UPHELD/REFUTED)은 감사 시점의 기록으로 그대로 유효하다 — 바뀐 것은 **판정이 아니라 코드와 데이터**다.

**검증 방법.** 아래 숫자는 전부 본 부록 작성자가 직접 재측정했다. 계산은 전량 HOST_238(`ssh -p 8022 USER@HOST_238`, `~/lpopt_ws/src`, `../venv/bin/python`), 로컬은 `grep`/`sed`/`sha256sum` 읽기만. §8.5 H-3 이 요구한 대로 **스냅샷을 먼저 고정한다:**

| 항목 | 값 |
|---|---|
| 검증 대상 store (238) | `~/lpopt_ws/data/store/records.parquet` · **76,793행 × 41열** · 22,814,014 B · Sep 2 20:35 · sha256 `22854b72a4966935550fd322da29fcab58fbfa19fdbb84124df444791b9c329d` |
| 로컬 권위본 | `data/store/records.parquet` · **76,793행 × 41열** · 22,810,322 B · Sep 3 02:16 · sha256 `16e311af4465e735b38daf7abf999268fac27946c1c5cc279114607d9ee917ba` |
| 소스 정합 | 인용 파일 11종(`store.py`·`dataset_torch.py`·`featurize.py`·`model_api.py`·`fuel_types.py`·`mine_policy_corpus.py` + 시험 5종)이 로컬 ↔ 238 `~/lpopt_ws/src` **sha256 11/11 일치** |

마지막 행이 중요하다: **238 에서 돌린 시험 결과는 로컬 트리의 결과와 동일**하다. §8.2 가 경고한 "공허한 확인" 함정을 이 부록은 소스 해시 대조로 먼저 막고 시작한다.

---

### 부록 A — C5 (§7): `trustworthy` 정규화 **완료**, `_maps` 누출 **차단 완료**

§7.8 이 **진행 중**으로 남긴 C5-1·C5-2·C5-3(및 이를 흡수한 §6.6 C4-2)은 **구현되어 병합되었다.**

**A.1 정규 술어가 실재한다** — `lpopt/data/store.py`:

| 심볼 | 위치 | 의미 |
|---|---|---|
| `valid_flags(df)` | `lpopt/data/store.py:201` | 열 부재/NULL → **True**(무증거는 격리 아님), 명시적 `valid=False` 만 격리 |
| `trustworthy(df)` | `lpopt/data/store.py:220` | `converged & valid_flags` — `converged` NULL 은 False, `valid` NULL 은 True |
| `row_valid(row)` | `lpopt/data/store.py:241` | `valid_flags` 의 행 단위 쌍둥이 |
| `row_trustworthy(row)` | `lpopt/data/store.py:247` | `trustworthy` 의 행 단위 쌍둥이 |

`:195-200` 주석이 §7.6 의 반증을 그대로 인용해 설계 근거로 박아 두었다 — *"Every training / elite / replay / holdout / corpus filter in the tree used to key on `converged` ALONE, which makes a quarantine UNENFORCEABLE."* 그리고 `:227` 은 **역사적 store 에서는 이 술어가 정확히 `converged == True` 와 같다**는 것을 시험으로 못 박았다(`test_store.py::test_trustworthy_matches_converged_when_no_row_is_quarantined`) — 즉 정규화가 기존 숫자를 흔들지 않음이 보장된다.

**A.2 경유 지점 — 7개 모듈 22곳** (실행 호출부만; 주석·docstring 언급 제외):

| 모듈 | 호출 위치 |
|---|---|
| `lpopt/curriculum.py` | `:656`, `:803`, `:1873`, `:1945`, `:2063` |
| `lpopt/model/dataset_torch.py` | `:193`, `:236`, `:261`, `:287`, `:320` |
| `lpopt/model/train.py` | `:356`, `:1056` |
| `lpopt/search/campaign.py` | `:1378`, `:1406`, `:1531`, `:3618` |
| `lpopt/search/produce.py` | `:542`, `:652`, `:685` |
| `lpopt/search/boundary_probe.py` | `:199` |
| `mine_policy_corpus.py` | `:733`, `:919` |

§7.8 C5-1 이 지목한 `campaign.py:1373/1401/3607`·`produce.py:538/646/679` 는 전부 갱신되었고(행 번호는 주석 추가로 소폭 이동), §6.6 C4-2 가 요구한 엘리트/시드 필터의 `& (valid == True)` 도 이 경유로 성립한다.

**A.3 `_maps` 게이팅(차단 급, C5-2) — 닫혔다.** `lpopt/model/dataset_torch.py:236`:

```python
if self.include_maps and row_trustworthy(row) and maps_key is not None and not (...):
```

`:226-231` docstring 이 결함을 명시적으로 시인한다 — *"this method used NOT to be [gated] … Without the gate the map head kept training on the rows the scalar heads had already masked out — the leak this predicate closes."* §7.5 가 지목한 `4cbfb4f6…` 의 P6 단일조성 맵은 이제 `map_loss`(`train.py:1888`)에 도달할 수 없다. 형제 게이트도 동일 술어로 통일되었다(`:261` `_axial`, `:287` `_traj`, `:193` 회귀 `target_mask`, `:320` `conv_mask`).

**A.4 정책 코퍼스 원천 차단(C5-3) — 닫혔다.** `mine_policy_corpus.py`:

- `build_steps`(`:707`) 첫 실행 줄이 `store = store[valid_flags(store)]`(`:733`) — **`lineage_edges` 호출 이전**이다. `:725-731` docstring 이 그 이유를 못 박는다: 뒤에서 거르면 `parent_record_id.isin(known)` 검사에 늦고, `both_converged` 로 하류 마스킹해도 소용없다(*"the column is about CONVERGENCE, and a `converged=True, valid=False` row passes it"*). 즉 §7.3 의 **무효 자식 edge 775건**은 이제 구조적으로 생성되지 않는다.
- `build_elites`(`:902`) 도 `pool = store[valid_flags(store)].copy()`(`:919`). `:902-908` docstring 이 §7.6 의 최악 시나리오를 그대로 적어 두었다 — 순위가 **F_r 오름차순**이라 격리행이 "누출"에 그치지 않고 **셀의 rank 1 에 앉는다**는 것.
- 비수렴 행은 **의도적으로 보존**된다(실패한 자식은 move-proposal 신호의 음성 절반) — §7.7 의 권장 재서술과 정확히 같은 형태다.

**A.5 시험 (238 실행, 2026-09-03)** — `cd ~/lpopt_ws/src && ../venv/bin/python -m pytest <file> -q`:

| 파일 | 결과 |
|---|---|
| `tests/test_lean_store_elites.py` | **22 passed** (8.45 s) |
| `tests/test_store.py` | **31 passed** (0.83 s) |
| `tests/test_dataset_torch.py` | **22 passed** (10.25 s) |
| `tests/test_policy_v3.py` | **39 passed, 1 skipped** (4.09 s) |

**A.6 남은 잔여(정직하게).** 정규화는 "전 필터"가 아니라 **엘리트·학습·코퍼스 경로**를 덮었다. 아직 맨 `converged` 를 쓰는 곳:

- `lpopt/model/splits.py:365` — **설계상 정상**이다. §7.4 가 확인했듯 splits 는 비수렴 행을 일부러 TRAIN 으로 보내고, 격리 차단은 하류 `dataset_torch` 의 마스크가 담당한다(A.3). 다만 §7.8 C5-1 의 열거에 이 파일이 들어 있었으므로 **경유하지 않았다는 사실 자체는 기록해 둔다.**
- `intervention_wave.py:321` — 여전히 `frame["converged"].fillna(False)` 단독.
- `pinbu_wave.py:175/298` — 동작은 이미 옳으나(`valid and converged` 인라인) 정규 술어를 경유하지는 않는다. §7.6 이 "제대로 하는 유일한 호출자"로 꼽았던 그곳이다.
- **C5-4(불변식 시험)는 여전히 미착수.** A.4 는 *앞으로* 생성될 코퍼스를 막지만, **라이브 `steps.parquet`/`steps_v3.parquet` 에 대해 `set(parent ∪ child) ∩ set(records[~valid]) == ∅` 를 단언하는 시험은 아직 없다.** §7.3 이 지적한 "현재의 부재는 가드가 아니라 append 아티팩트" 라는 성질은 **재구축을 해야 비로소** 해소된다.

---

### 부록 B — C3 (§5.5): `g_e_core` train/serve 파열은 **닫혔다**. 게이트 대표성(head(25))은 **여전히 열려 있다**

§5.5 는 두 가지를 주장했다: (i) `tests/test_featurize.py` 가 **지금 red** 이고 `g_e_core` 파열이 살아 있다(paramA 약 9.8 %, S1j-val **188/216**), (ii) 게이트가 `k.head(25)` 라 구조적으로 못 잡는다. **(i)은 오늘 거짓이고, (ii)는 오늘도 참이다.**

**B.1 시험은 green 이다.** 238 실행: `tests/test_featurize.py` → **24 passed** (4.81 s), 0 failed. §5.5 가 인용한 `tests/test_featurize.py:189 test_core_enrichment_split_reproduces_stored_e_core` 의 `assert 4.871464678734045 == 4.892576507278004 ± 1e-9` 는 재현되지 않는다. **§5.5 의 "저장소 시험이 지금 red 다" 와 §9 우선순위 1번의 "저장소 시험이 지금 red" 는 이 시점부터 무효다.**

**B.2 무엇이 고쳤는가 — 시험이 아니라 데이터.** 이 구별이 중요하다(시험을 느슨하게 고쳐 green 을 산 것이 아님을 보이려면):

- `tests/test_featurize.py` 는 **바이트 무변경** — mtime Aug 29 20:02, sha256 `8f795a4ff1af73fadc4c812e4f19c918b04a37909962cccd3f8c7266ee3f78f6`. 즉 **같은 단언이 이제 통과**한다.
- `lpopt/data/fuel_types.py` 도 **무변경**(Aug 17 15:26) — 레시피 `core_enrichment_split` 자체는 손대지 않았다.
- 새로 생긴 것은 `lpopt/data/store.py:561` **`backfill_e_core(store_dir, *, dry_run=True, tol=ECORE_BACKFILL_TOL, backup_suffix=None)`**(`store.py` mtime Sep 2 20:50). docstring `:568-578` 이 원인을 정확히 짚는다 — 파열의 정체는 레시피 결함이 아니라 **produce/campaign 행이 패턴이 함의하는 값 대신 캠페인 전체에 상수인 *nominal* 농축도(계획된 50/50 또는 1/N split, 언제나 `e_split` null 동반)를 실었던 것**이다. 추출기(Dataset A/B) 행은 원래부터 float 정밀도로 일치했고 byte-identical 로 남는다.
- 즉 §5.8 **C3-2 의 두 선택지 중 A안(e_core 백필 재실행)** 이 채택되었다. B안(`_record_inputs` 가 store 의 `e_core` 를 우선)은 **채택되지 않았다** — `lpopt/model/model_api.py:993` 의 `_record_inputs` 는 지금도 `core_enrichment_split(self.fuel, lib, pattern.batch_feed())` 로 재구성하고, `lpopt/model/featurize.py:1659-1671` `_estimate_e_core` 도 같은 함수에 위임한다. (`featurize.py:1542-1546` 은 `inp.e_core` 를 우선하고 null 일 때만 레시피로 폴백하는데, `_record_inputs` 가 항상 값을 채워 주므로 실서빙에서는 레시피 값이 쓰인다.) **따라서 파열을 닫은 것은 store 열을 레시피에 맞춘 것이지, 서빙이 store 를 읽게 만든 것이 아니다.**

**B.3 전수 프로브 — 0 / 76,793.** 시험이 표본이므로(B.4) 필자가 직접 **전수** 재측정했다. 238 에서 `~/lpopt_ws/scratch/ecore_probe_20260903.py`, 위 표의 76,793행 store 와 `data/store/fuel_types.parquet` 사용:

```
parquet metadata rows = 76793  cols = 41  row_groups = 1
rows with e_core notna = 76793
checked            = 76793
e_core mismatches  = 0      (helper returned None: 0)
e_split mismatches = 0
max |d e_core|     = 1.7763568394002505e-15
max |d e_split|    = 0.0
per-library (rows, bad):
  260624 (29,976, 0) · 5.8_5.1 (8,244, 0) · ga80 (20,127, 0)
  legacy_a (634, 0)  · paramA (17,812, 0)
```

최대 편차 **1.78e-15** 는 배정밀도 반올림이며 시험 허용오차 `1e-9` 보다 6자리 아래다. **§5.5 표의 "store-wide paramA 불일치 9.8 % (max |Δ| 0.040936)" · "1,338 / 16,316" · "S1j-val 188 / 216 = 87 %" · "paramA S1j-val converged 전수 228 / 2,280" 은 전부 이 스냅샷에서 0 이다.** paramA 행 수가 16,316 → **17,812** 로 늘어난 뒤에도 0 이다(r2 신규분 포함). 즉 후속 fixup 이 보고한 *"exhaustive e_core probe, 0/76,793 mismatches"* 가 **옳고**, 본문 §5.5 와 §9 우선순위 1번이 **틀렸다** — 감사 시점에는 옳았으나 지금은 아니다.

곁가지 정정: §5.8 C3-2 는 `model_api.py:1000-1006` docstring 의 "differ on **0 rows**" 도 고치라고 했는데, **이제 그 docstring 이 참이 되었으므로 정정 대상이 아니다.**

**B.4 그러나 게이트 대표성(C3-3)은 여전히 열려 있다 — 그리고 본문이 짚은 것보다 넓다.**

§5.5 는 `tests/test_model_api.py:463-464` 의 `k.head(25)` 를 지목했다. 오늘 확인:

```
tests/test_model_api.py:464     frames.append(k.head(25))        # test_serve_row_featurization_parity
```

**여전히 `head(25)` 다.** 게다가 본문이 놓친 것이 하나 더 있다 — **B.1 에서 green 을 낸 그 시험 자신도 `head(25)` 표본이다:**

```
tests/test_featurize.py:185-186   for lib, grp in df[df["e_core"].notna()].groupby("library_id"):
                                      for _, row in grp.head(25).iterrows():
```

**따라서 "24 passed" 는 그 자체로 파열이 닫혔다는 증거가 되지 못한다** — 라이브러리당 25행, 총 125행만 본다. §5.5 가 "228건 위반 중 head(25) 안에 든 것은 0건" 이라고 지적한 바로 그 맹점이 이 시험에도 그대로 있다. **파열이 닫혔다는 근거는 B.3 의 전수 프로브(76,793행)이며, 그 프로브는 아직 CI/시험에 편입되어 있지 않다.** 즉 지금 상태는 *"결함은 없앴으나 결함이 없음을 지키는 게이트는 없다"* 이다.

§5.8 **C3-3 은 미착수 → 진행 중**으로 갱신한다: 실행 중인 구현 워크플로가 `head(25)` 를 시드 무작위 표본 또는 S1j-val 전수로 교체하고 store 전수 `core_enrichment_split(...) == e_core` 단언(1e-6)을 추가하는 작업으로 다루고 있다. **완료 판정은 `tests/test_model_api.py:464` 와 `tests/test_featurize.py:186` 두 곳 모두에서 `head(25)` 가 사라진 것을 확인한 뒤에 내려야 한다.**

§5.8 의 나머지(C3-1 범위 정정, C3-4 574 legacy xfail, C3-5 policy v3 승격, C3-6 remote 소스 게이트, C3-7 2,401→2,388)는 본 부록 시점에 **변동 없음**. §5.4 의 policy scorer v1/v2 반증도 그대로 유효하다 — 이 부록은 `g_e_core` 축만 갱신한다.

---

### 부록 C — C2 (§4.4 C·D): §5.1 해시 재스탬프 **완료**, per-cell 보정 라운드트립 **진행 중**

**C.1 재스탬프는 실재하고, 오늘 정확하다.** `data/models/s1j/PROMOTION.md`:

- `:140` **`## 5. SERVE-PATH HASH RE-STAMP — 2026-09-03`** (부제: *rewritten; supersedes the 2026-09-02 §5 and its 01:29 addendum*)
- `:156` **`### 5.1 Current stamp (fenced)`** — 현행 4행 표(`:158-163`)
- `:222` `### 5.3 Superseded stamps` — 구 값 전량 보존(§4.6 C2-1 이 요구한 "구 행은 날짜와 함께 보존")
- `:262-288` `### 5.4 Stamping procedure` — **sha fence**: 쓰기 전 `sha256sum` → 로컬 push → 라이브 미러 해시 → 쓰기 → **즉시 `sha256sum -c` 재검증, 4× OK 아니면 폐기 후 재시도**. `:286` 이 이 절 자신이 그 fence 하에서 쓰였고 사후 검증이 4× OK 였음을 기록한다.

필자의 독립 재측정 — **로컬 트리와 238 `~/lpopt_ws/src` 양쪽, 4/4 파일 전부 해시·바이트수 일치:**

| 파일 | §5.1 선언 sha256 | 선언 bytes | 로컬 | 238 |
|---|---|---:|---|---|
| `lpopt/model/model_api.py` | `139c82fc12c48ff0…334b04` | 117,059 | 일치 | 일치 |
| `lpopt/search/acquisition.py` | `84ff2f11eea75138…603732` | 167,193 | 일치 | 일치 |
| `lpopt/search/campaign.py` | `525ed360df3c2f29…62fc20` | 249,676 | 일치 | 일치 |
| `lpopt/model/featurize.py` | `5e720a5e50455f07…3a00baa` | 89,119 | 일치 | 일치 |

**§4.4(C) 의 "R1 매니페스트가 stale" 은 해소되었다.** 근거 없던 링크 `5a713eaa…` 는 §5.3 에 이력으로만 남았고, `:256` 이 *"R1 still binds — shipping this dir means shipping the four files stamped in §5.1"* 로 규칙과 표를 다시 연결한다. §4.6 **C2-1·C2-2 → 완료.**

한 가지 표시해 둘 것: §4.6 C2-1 이 재스탬프 값으로 적었던 `campaign.py 6d6425bb…` 는 **이미 다시 움직였다**(현행 `525ed360…`). §5.1 이 스스로 경고하는 드리프트(*"These hashes WILL drift again … and that is not a defect"*)의 첫 실례다. **본문 §4.6 C2-1 의 `6d6425bb…` 를 현행 값으로 인용하지 말 것.**

**C.2 per-cell 보정 라운드트립(C2-4·§4.4 D) — 진행 중.** `PosValCnnBackend.save()`(`lpopt/model/model_api.py:2060`)가 현재 기록하는 것:

```
member_<seed>/ (save_member)  ·  _save_ensemble_meta(out, written)  ·  CALIB_NAME
_FEATURE_OOD_NAME             ·  _BACKEND_MANIFEST
```

**σ 절반은 닫혔다** — `_save_ensemble_meta` 가 파생 체크포인트에 `ensemble.json` 을 라운드트립하며, 그 docstring 이 D3 를 그대로 서술한다(*"per-wave `models/champion_wave_NN` carried members + `backend.json` but no `ensemble.json`, so `--resume` reloaded a `s1j` descendant with…"*). **수준(level) 절반은 아직이다** — `from_dir` 이 `:741`(`cell_cpath = d / CELL_CALIB_NAME`) 이하에서 읽는 **per-cell 보정 6종**(`cell_/f_r_/cbc_/f_q_/ao_abs_/flatness_calibration.json`)을 `save()` 는 여전히 쓰지 않는다. 따라서 §4.4(D) 가 등록한 결함 — **`--resume` 이 수준 보정 없는 `s1j` 후손을 서빙하고 F_r ≤ 1.55 / F_q / CBC / |AO| 게이트와 cyclen LCB 가 영향을 받는다** — 은 **오늘도 유효**하다.

§4.6 **C2-4 는 미착수 → 진행 중**(현재 구현 중). 완료 판정 기준은 §4.6 이 적은 그대로다: `save()` 가 6종을 복사로 라운드트립(합성 금지) + `_save_champion` 에 보정 집합 동일성 단언.

§4.6 C2-3(r2 로그를 box 199 에서 확인)·C2-5(승격 절차에 `serve_sigma` 스탬프 요구)는 **변동 없음(미착수)**. §4.5 의 재서술 권고도 그대로 유효하다.

---

### 부록 D — H-1 (§8): 238 store 미러는 **76,793행으로 갱신되었다** (단, 한 걸음 뒤)

§8.1 이 기록한 stale 미러(74,717행 · Aug 29 13:20 · sha `cf495c7d…`)는 **해소되었다.**

```
$ ssh -p 8022 USER@HOST_238 "ls -l ~/lpopt_ws/data/store/records.parquet"
-rw-r--r--. 1 USER ctrp 22814014 Sep  2 20:35 /home/USER/lpopt_ws/data/store/records.parquet

pyarrow.parquet.ParquetFile(...).metadata
  num_rows = 76793   num_columns = 41   num_row_groups = 1
  sha256   = 22854b72a4966935550fd322da29fcab58fbfa19fdbb84124df444791b9c329d
```

`~/lpopt_ws/src/data` 는 `~/lpopt_ws/data` 로의 **심볼릭 링크**이므로(`readlink -f` 로 확인), `REPO_ROOT/data/store/records.parquet` 를 읽는 시험군(`tests/test_featurize.py`, `tests/test_store.py` 등)은 **바로 이 갱신본을 읽는다.** 부록 A.5·B.1 의 시험 결과와 B.3 의 전수 프로브가 같은 파일에 대한 것임이 이로써 보장된다. §8.1 이 기록한 부재 항목들도 해소되었다 — 이 스냅샷에는 HGD569 캠페인 행과 `bf3a70b2…` 가 모두 들어 있다(§8.2 의 "공허한 확인" 위험 소멸).

**다만 미러는 이미 한 걸음 뒤다 — 이것이 H-1 을 "완료"가 아니라 "행수 정합"으로만 적는 이유다.** 238 미러의 sha `22854b72…` 는 로컬의 **백업본** `data/store/records.parquet.bak_pre_pinbu_minfxy_r2_20260903`(22,814,014 B)와 **바이트 동일**하다. 로컬 권위본은 그 뒤 Sep 3 02:16 에 `pinbu_minfxy_r2` 백필로 **제자리 갱신**되어 sha `16e311af…`(22,810,322 B)가 되었다. 두 파일 모두 **76,793행 × 41열** — 행 추가가 아니라 열 값 갱신이므로 **행수 기반 주장은 양쪽에서 동일**하고, 부록 B.3 의 프로브 결과는 `e_core`/`e_split` 축에 관한 한 그대로 유효하다. 그러나 **FOM 값을 인용하는 재현은 어느 sha 인지 반드시 밝혀야 한다.**

§8.5 **H-1 → 실질 완료(행수·격리행 정합), 단 sha 는 백필 1회분 뒤짐.** §8.5 H-3(스냅샷 sha·행수 동시 표기 규칙)은 본 부록이 서두 표에서 **실제로 준수**했다. H-2·H-4·H-5 는 변동 없음 — 특히 §8.4 의 **`S1j.json` 이 `cf495c7d…`(74,717행, 격리 이전)에 고정되어 있어 HGD569 봉쇄에 대해 아무 말도 할 수 없다**는 사실은 오늘도 그대로다.

---

### 부록 E — 상태 델타 요약

| 항목 | 본문 상태 (2026-09-03 작성) | 오늘 상태 | 증거 |
|---|---|---|---|
| C5-1 `trustworthy` 정규화 (= C4-2) | 진행 중 | **완료** | `store.py:201/220/241/247`, 7모듈 22 호출부, `test_store.py` 31 passed |
| C5-2 `_maps` 게이팅 (차단 급) | 진행 중 | **완료** | `dataset_torch.py:236`, `test_dataset_torch.py` 22 passed |
| C5-3 코퍼스 원천 차단 | 진행 중 | **완료** | `mine_policy_corpus.py:733`(build_steps)·`:919`(build_elites), `test_policy_v3.py` 39 passed·1 skipped |
| C5-4 불변식 시험 | 미착수 | **미착수** | 라이브 steps 산출물 재구축 필요 |
| C3-2 `g_e_core` 파열 | 미착수 · "시험 지금 red" | **완료 (본문 문장 무효)** | `test_featurize.py` **24 passed**; 전수 프로브 **0 / 76,793**, max\|Δ\| 1.78e-15 |
| C3-3 게이트 대표성 `head(25)` | 미착수 | **진행 중 — 여전히 열림** | `test_model_api.py:464` **및** `test_featurize.py:186` 둘 다 `head(25)` |
| C2-1 / C2-2 `PROMOTION.md` 재스탬프 | 진행 중 | **완료** | `PROMOTION.md:140/156/222/262`, 4/4 해시·바이트 로컬·238 일치 |
| C2-4 per-cell 보정 라운드트립 | 미착수 | **진행 중** | `model_api.py:2060` `save()` 가 6종 미기록; `from_dir:741` 은 읽음 |
| H-1 238 store 미러 | 진행 중 | **실질 완료** (76,793행 × 41열), sha 는 백필 1회분 뒤짐 | `ls -l` + pyarrow metadata + sha 대조 |

**한 줄 총평.** §1 표의 다섯 판정 중 **REFUTED 두 건(C3·C5)의 실질 결함 부분이 닫혔다** — C5 는 정규 술어·맵 헤드 게이트·코퍼스 차단 세 갈래가 모두 병합되었고, C3 의 `g_e_core` 파열은 store 백필로 전수 0 이 되었다. **그러나 감사가 지목한 더 깊은 병 — "게이트가 결함을 볼 수 없다"(C3-3 `head(25)`)와 "불변식이 검사되지 않는다"(C5-4) — 는 아직 열려 있다.** 부록 B.4 는 이를 한 단계 악화된 형태로 재확인했다: **파열이 닫혔음을 보증한 그 시험조차 라이브러리당 25행 표본**이며, 전수 증거는 이 부록의 일회성 프로브뿐 CI 에 없다. §10 의 한계 목록에 다음 한 줄을 더할 것을 권한다 — **"시험이 green 이라는 사실은, 그 시험의 표본 범위를 함께 적기 전에는 인용하지 않는다."**
