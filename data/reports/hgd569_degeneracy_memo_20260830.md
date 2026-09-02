# HGD569_f125 축퇴 + 공통모드 오프셋 — 원인 확정 메모

**작성 2026-08-30. read-only 조사(코드·데이터 무편집).**
대상: `intervention_wave_r1` / cell `HGD569_f125` / case_pair `P6253Z1G06N24_P6253Z2G10N24` / feed 125 / kit paramA.
선행: `data/reports/intervention_wave_r1_results_20260830.md` §5.1(축퇴), §7(공통모드), §11-1, §14-5, §15.

---

## 0. 결론 한 줄

두 이상현상은 **같은 하나의 원인**이다: 이 wave 의 deck 생성 경로에서 **`type_id → 2-char alias` 번역이 조용히 무효(no-op)** 가 되어 `%LPD_SHF` 에 `F P6253Z1G06N24  0` / `F P6253Z2G10N24  0` 라는 **`%LPD_B&C` 에 존재하지 않는 batch id** 가 기록되었고, MASTER 는 이를 **둘 다 batch `P6`** 로 해석했다. 그 결과 (a) 두 fresh type 을 맞바꾸는 개입은 MASTER 에게 완전한 no-op 이 되고, (b) 노심 전체가 설계와 다른 단일 조성(`FA_P6`)으로 계산되어 +35 EFPD 오프셋이 생겼다.

**§7 이 유력 후보로 지목한 "synth deck" 은 무죄다.** deck 파일 자체는 정상이며, §15 가 유력하다고 적은 "슬롯 기하 축퇴(octant 대칭 사상)" 가설은 **기각**된다.

---

## 1. 결정적 증거 — MASTER 자신의 `== CORE LOADING PATTERN` echo

두 MAS_OUT 은 **같은 restart**(`MAS_RST.APRQ_11_0705.02`, `pair_ecore`, L3)에서 출발한다.

**개입 wave (본 wave, 자식 chain)**
`E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/intervention_wave_r1/intervention_HGD569_f125/master_work/worker_00/0ea999534cc895d6__.../MAS_OUT`

```
 == CORE LOADING PATTERN
            P6  P6  P6  P6  P6  P6  P6  P6  P6  R1
            P6  P6  P6  P6  P6  P6  P6  P6  P6  R1
            ... (모든 연료 위치 P6) ...
```
→ EOC restart `MAS_RST.APRQ_20_0768.83` (cyclen 768.83 EFPD)

**parent 를 라벨링한 원 campaign (`fpcamp_minfr_hgd569_f125`)**
`E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/fpcamp_minfr_hgd569_f125/master/master_work/worker_00/0210553df47169ac__.../MAS_OUT`

```
 == CORE LOADING PATTERN
            S3  S3  S3  S5  S3  S5  S5  S3  S5  R1
            S3  S3  S5  S5  S5  S5  S5  S3  S3  R1
            ... (S3/S5 혼합) ...
```
→ EOC restart `MAS_RST.APRQ_20_0732.83` (cyclen 732.83 EFPD)

대조군: 같은 wave 의 `T6T4_f121` 은 `T4/T6` 혼합으로 정상 echo 된다(=paramA 이지만 case_pair 가 이미 alias 공간이라 번역이 필요 없다).

**즉 개입 wave 의 HGD569 노심은 fresh 뿐 아니라 carryover 위치까지 전부 batch `P6` 로 해석되었다.** 설계된 노심이 아니다.

---

## 2. deck 수준 증거

### 2.1 synth deck 원본은 정상

`data/design/synth_decks/P6253Z1G06N24_P6253Z2G10N24/MAS_INP_cy12.inp`
sha256[:16] = **`cddba86904810b7c`** (report 등록값과 일치), 192행.

`%LPD_B&C` = **reflector 3행(R1/R2/R3) + fuel 37행**, batch id 는 전부 **2-char paramA alias** `P0 P1 … P9 Q0 … Q9 S0 … S9 T0 … T6`, comp 1–37, `%LPD_C&X`/`%LPD_HFF` 는 `FA_<alias>`. `%GEN_DIM` nbatch=40, ncomp=42.

- **fresh type 항목이 한 행으로 합쳐진 곳은 없다.** 37행 전부 distinct batch·distinct comp 다.
- 이 표는 pair 별이 아니라 **library 전체 roster** 다(`lpopt/design/coredeck.py:_lpd_static`, `resolver.paramA_library_dims` → `bootstrap.library_aliases`). `T6_T4` 가 쓴 packaged deck 의 `%LPD_B&C` 와 **동일한 표**다.
- 이 cell 의 두 fresh type 은 이 표 안에 **정상적으로 존재한다**: `P6253Z1G06N24 → S3`(comp 24), `P6253Z2G10N24 → S5`(comp 26). (`data/store/fuel_types.parquet` `source_flags` 의 `alias:S3` / `alias:S5`, 그리고 `data/design/package/registry.json`.)

→ **"두 fresh type 이 한 `%LPD_B&C` 행으로 해석된다"는 가설은 deck 파일 차원에서는 거짓이다.** 무너진 것은 deck 이 아니라 **pattern → deck token 사상**이다.

### 2.2 실제로 emit 된 deck

개입 wave: `runs/intervention_wave_r1/intervention_HGD569_f125/produce_cases/P6253Z1G06N24_P6253Z2G10N24_f125/<sha16(pattern)>__MAS_RST.APRQ_11_0705.02/MAS_INP_cy12.inp`

```
%LPD_SHF
        F P6253Z1G06N24  0, 1 L 14 2, F P6253Z1G06N24  0, ...
```

원 campaign(archive, 같은 경로 구조):

```
%LPD_SHF
        F S3  0, 1 L 14 2, F S3  0, F S5  0, ...
```

**두 deck 을 `%LPD_SHF` 본문만 제외하고 diff 하면 차이 0줄** (183/183행 일치): `%JOB_TYP` restart 참조, `%GEN_DIM`, `%LPD_B&C`, `%LPD_C&X`, `%LPD_HFF`, 소진 chain 전부 byte-identical. **단일 변수 실험이 성립한다 — 차이는 SHF fresh batch token 뿐이다.**

`F <batch> <rot>` 카드는 `lpopt/vendor/masterrl/domain.py:104` 의 `f"F {self.batch:<2}  {self.rotation}"` 로 찍힌다. 폭 2는 **최소폭**이므로 13자 이름이 그대로 나간다. MASTER 는 이 이름을 `%LPD_B&C` 에서 찾지 못하고 **앞 2자 `P6`** 로 귀결시킨다(에러도, 경고도 없다 — `MASTER.stderr` 비어 있음).

### 2.3 형제 deck 은 byte-identical 이 아니다

과제 (2)의 질문에 대한 답: **아니다.** 20쌍 전부에서 `MAS_INP_cy12.inp` 는 서로 다르다(0/20 identical). 차이는 5개 슬롯의 `F P6253Z1G06N24` ↔ `F P6253Z2G10N24` 뿐이고 `%LPD_SHF` 의 shuffle 행(`1 L 14 2` 등)은 완전히 동일하다. **deck 은 달랐고, MASTER 가 그 차이를 보지 못했다.** → `%LPD_SHF` 조사 대상은 shuffle 행이 아니라 fresh 행이다(위 §1·§2.2).

---

## 3. 근본 원인 — `WaveVerifier` 의 내부 fallback resolver

경로를 끝까지 따라가면 다음 한 줄이다.

```
lpopt/search/verify.py:849
    self.resolver = resolver or CaseAssetResolver(self.package_root or ".")
```

- `alias_pattern()` (`lpopt/search/assets.py:834`)은 `self.type_to_alias` 가 비면 **조용히 pattern 을 그대로 반환**한다. `alias_case_key()` 도 같다.
- 이 두 함수를 실제로 호출하는 것은 **verifier 의 `self.resolver`** 다: `verify.py:987`(`alias_case_key`), `verify.py:995`(`prepare_cycle1_deck` → 내부 `alias_pattern`, `assets.py:829`), `verify.py:1069`(`_eval_entry` 의 `alias_pattern`).
- `ablation_wave.py:707` 의 `WaveVerifier(...)` 호출에는 **`resolver=` 인자가 없다.** 따라서 위 fallback 이 발동하여 `registry_aliases` 가 **빈 dict** 인 resolver 가 만들어진다.
- `intervention_wave.py:1105` 는 `extra["registry_aliases"] = paramA_registry_aliases(package)` 를 준비하고 `A.CaseAssetResolver = resolver_shim` (`intervention_wave.py:1149`)으로 monkeypatch 하지만, **`lpopt/search/verify.py:65` 가 모듈 로드 시점에 `from .assets import CaseAssetResolver` 로 원본 클래스를 이미 자기 네임스페이스에 바인딩**했다. `_cell_binding` 은 patch 직전에 `import lpopt.search.verify as V` 를 실행하므로 verify 의 바인딩은 **항상 원본**이다. → **shim 이 deck 을 실제로 쓰는 지점을 빗나간다.**
- `ablation_wave.cmd_run` 이 resolve 에 쓰는 resolver(함수 내부 import → shim 적용됨)는 **restart·template 선택에만** 관여한다. 그래서 `restart_provenance` 는 parent 와 정확히 일치했고(P5 0/800), deck sanity gate(`library_dims`는 별도 인자로 전달됨)도 통과했다. **오염은 오직 SHF batch id 에만 나타난다.**

### 왜 다른 cell 은 멀쩡했나

| 경로 | `resolver=` 전달 | HGD569 류(long type_id) 영향 |
|---|---|---|
| `lpopt/search/campaign.py:1331`, `:3305` (`_resolver()` → `build_case_resolver`) | ✅ | 없음 — fpcamp / fxy_exp / mv3 / tripletype 전부 여기 |
| `lpopt/search/produce.py:472,497` | ✅ | 없음 — produce/kit 경로 |
| `ablation_wave.py:707` | ❌ | **발생** |
| `fr_arms.py:188`, `fr_transfer.py:522`, `rule_acid_run.py:386`, `v520_run.py:622`, `lpopt/design/pathfinder.py:268` | ❌ | 잠재 위험(아래 §5) |

`T6T4_f121` 도 같은 결함 경로를 탔지만 **case_pair 가 이미 alias 공간(`T6_T4`)** 이라 번역이 필요 없어 무해했다. 즉 **round 1 에서 T6T4 가 정상이었던 것은 이 버그의 반증이 아니다.**

---

## 4. 두 이상현상이 어떻게 유도되는가

### (a) `batch_swap` 20/20 축퇴 — 정보량 0

두 fresh type 이 모두 `P6` 로 귀결되므로 **fresh type label 만 바꾸는 개입은 MASTER 입력으로서 항등**이다. 측정으로 확인:

- pattern 을 fresh label 만 blind 처리(`F:*:0 → F:X:0`)하면 수렴 153 chain 이 **113개 class** 로 뭉치고, **20개 class 가 정확히 3-member** 다.
- 그 20개 class 안에서 `(cyclen, F_r, F_q, node_peak)` 불일치 **0/20**.
- 각 class 구성: `batch_swap|outward` + `batch_swap|inward` + `batch_flip|(in|out)` → **`batch_swap` 40 chain + `batch_flip` 20 chain = 60 chain 이 실은 20개 노심**이다. **중복 40 chain (cell 예산의 25%, wave 800 의 5.0%)**.
- 반면 `fresh_relocate` 는 fresh/burnt 슬롯의 **점유 자체**를 바꾸므로 붕괴하지 않는다(축퇴 class 0). 그래서 §3.1 의 HGD569 `fresh_relocate` 통계는 살아 있다.

report §11-1 의 "슬롯 기하 축퇴", §15 의 "octant 대칭 사상" 은 **불필요한 가설이었다.** 원인은 기하가 아니라 **token 사상 붕괴**다.

### (b) 공통모드 +35 EFPD

노심 전체(fresh + carryover)가 `FA_P6` 로 계산되었다. `P6` = `P6257Z1G06N24`.

| alias | design | u_avg (w/o) | gd_wt | kinf0 | kinf20 |
|---|---|---:|---:|---:|---:|
| **P6 (실제 사용됨)** | P6257Z1G06N24 | **5.8812** | 6.0 | 1.16241 | **1.17221** |
| S3 (의도) | P6253Z1G06N24 | 5.7861 | 6.0 | 1.15638 | 1.16819 |
| S5 (의도) | P6253Z2G10N24 | 5.6023 | 10.0 | 1.12485 | 1.11256 |

설계 e_core 5.6944 (50/50) → 실현 5.8812 (단일). Gd 무게도 6/10 혼합 → 6.0 단일. **더 반응도가 높은 노심이므로 cycle 이 길어지는 방향이 물리적으로 정합**하고, 크기도 자릿수가 맞는다.

측정된 vs-parent 공통모드(재계산, store 기준):

| move_class·dir | n | d_cyclen | d_F_r | d_F_xy | d_node_peak |
|---|---:|---:|---:|---:|---:|
| `rewire_swap`·neutral | 20 | **+35.053** | +0.098 | +0.106 | +0.049 |
| `batch_swap`·inward | 20 | +34.942 | +0.087 | +0.101 | +0.039 |
| `batch_swap`·outward | 20 | **+34.942** (inward 와 비트 동일) | +0.087 | +0.101 | +0.039 |
| `batch_flip`·inward / outward | 10 / 10 | +35.253 / +34.630 | +0.097 / +0.076 | +0.110 / +0.092 | +0.050 / +0.027 |
| `fresh_relocate`·inward / outward | 36 / 37 | +34.998 / +33.872 | +0.140 / +0.132 | +0.179 / +0.177 | +0.068 / +0.060 |

parent 평균 cyclen 731.514 / F_r 1.6390 / F_xy 1.7014 → child 평균 766.179 / 1.7501 / 1.8379. store 의 campaign 별 f125 밴드도 같은 이야기를 한다: `fpcamp_minfr_hgd569_f125` 709–737, `_seedctl` 713–735, `fxy_exp_HGD569_f125` 708–741, **`intervention_HGD569_f125` 762–771**.

**restart 는 범인이 아니다** — parent 20개와 child 160개 모두 `pair_ecore:MAS_RST.APRQ_11_0705.02`, `library_id=paramA` 로 동일하다. **deck 도 파일로서는 범인이 아니다** — §2.2 의 non-SHF diff 0줄. 범인은 SHF token 이다.

---

## 5. 영향 범위 — store / corpus / 다른 cell

### 5.1 오염된 행: **정확히 `intervention_HGD569_f125` 160행**

`data/store/records.parquet` 에서 paramA·long-type_id case_pair 행 15,354개 중 결함 경로(`ablation_wave`/`intervention_wave` 계열)에서 나온 것은 **`generator == 'intervention_1move'` 이면서 long pair 인 160행뿐**이다.

| generator | long-pair 행 | 경로 | 상태 |
|---|---:|---|---|
| `intervention_1move` | **160** | ablation_wave `cmd_run` | **🔴 오염** |
| `heuristic` / `random` / `elite_perturb` / `local` / `elite` / `guided` / `rule_biased` / `g3_elite_boundary` | 15,194 | campaign / produce (`resolver=` 전달) | ✅ 정상 |
| `ablation_1move`, `batchswap_enum`, `batchswap_enum_625`, `transfer` | 0 (전부 short pair) | ablation_wave 등 | ✅ 무해(alias 공간) |

명시 확인:

- **`produce_fxyera_r1` 의 `fxy_*_HGD569_f125` / `fxy_*_HGD569_f109` strata — 오염 없음.** `fxy_exp_HGD569_f125` 49행은 generator `elite_perturb`/`heuristic`/`random`, cyclen 707.972–741.499 로 정상 밴드다.
- **tripletype campaign (`fpcamp_minfr_triple_f125`, `_r2`, `fxy_bnd|exp|rule_S3T1S5_f125`) — 오염 없음.** archive MAS_OUT 의 `CORE LOADING PATTERN` 이 **`S3/S5/T1` 3종 혼합**으로 정상 echo 되고, cyclen 698–737 로 정상 밴드다.
- `mv3_e569_f125`, `fpcamp_minfr_hgd569_f109/_f125/_seedctl` — 정상(§1 의 S3/S5 echo).

### 5.2 파생 자산

- `data/policy/steps.parquet` — `campaign == intervention_HGD569_f125` **160행**. `d_cyclen`/`d_f_r`/`d_f_xy`/`d_node_peak`/`d_cbc_max` 절대값 전원 무효.
- `runs/intervention_wave_r1/intervention_HGD569_f125/` — `ablation_results.jsonl` 160행, `fxy_sidecar.jsonl`, `map_*.npy` 153개, `produce_cases/**` 160 deck.
- `data/store/maps.npz` 의 해당 `maps_key`, `fxy_backfill_199_intervention_wave_r1_20260830.csv` 의 HGD569 행.
- `data/reports/intervention_wave_r1_results_20260830.md` §3.4 raw 열, `effects_by_cell` 의 HGD569 행, s1i 블라인드 채점의 HGD569 성분(+6.98 EFPD 편향의 전량).

### 5.3 **무효화되지 않는 것** (report §7 의 판단은 유지된다)

- **쌍 내부 (out − in) 대조 전부.** 오프셋은 parent 공통모드이므로 정확히 상쇄된다. H1(`fresh_relocate` 5/5 부호, pooled p=8.1e-8), H3, H4 는 안전하다.
- 단, **HGD569 의 `fresh_relocate` out−in 은 "설계 노심"이 아니라 "P6 단일 노심"에서 측정된 것**이다. 부호·순위는 그 노심에 대해 유효하지만, **HGD569_f125 라는 cell 의 물리를 대표하지 않는다.** §3.1 의 HGD569 행(−1.1930 EFPD, p=0.065 등)에는 이 caveat 를 달아야 한다 — report §7 은 여기까지는 적지 않았다.
- **H2 의 판정은 바뀐다.** §5.1 이 "prereg census 가 놓친 두 번째 종류의 방향 축퇴(슬롯 기하)" 로 적은 것은 물리 현상이 아니라 **harness 결함**이다. "판정 불가"라는 결론 자체는 유지되지만 사유가 다르다.

---

## 6. 권고 — 우선순위 순

### R1 (필수, 코드) — `WaveVerifier` 의 조용한 fallback resolver 제거

`lpopt/search/verify.py:849` 의 `resolver or CaseAssetResolver(self.package_root or ".")` 는 **paramA 에서 조용히 틀린 deck 을 만든다.** 최소 수정 두 가지 중 하나:

- **(권장) fail-fast**: `library_dims != LIBRARY_DIMS`(= paramA) 인데 `resolver=None` 이면 `ValueError`. paramA 는 반드시 alias bridge 를 가진 resolver 를 요구한다.
- 또는 `alias_pattern`/`alias_case_key` 가 **fresh batch id 가 `%LPD_B&C` roster 에 없으면 raise**. 이쪽이 더 근본적이다 — 지금은 "roster 에 없는 batch id" 가 MASTER 까지 무검증으로 흘러간다.

추가로 **`prepare_cycle1_deck` 직후의 `validate_reload_deck` 에 "SHF 의 모든 fresh batch id ∈ `%LPD_B&C` batch id 집합" 검사를 넣는다.** 이 한 줄이 있었으면 800 chain 중 160 이 첫 chain 에서 멈췄다. `validate_reload_deck` 은 이미 dims·restart 참조·fresh-core map 을 보는 자리이므로 자연스러운 위치다.

### R2 (필수, 코드) — monkeypatch 지점 교정

`intervention_wave._cell_binding` 은 `lpopt.search.assets.CaseAssetResolver` 만 patch 한다. deck 을 실제로 쓰는 것은 `verify` 모듈의 바인딩이다. 올바른 해법은 patch 가 아니라 **`ablation_wave.cmd_run` 이 이미 만든 resolver 를 `WaveVerifier(resolver=...)` 로 넘기는 것**이다(`campaign.py`/`produce.py` 가 하는 그대로). 이러면 R1 의 fail-fast 와 함께 shim 자체가 불필요해진다.

동일 결함이 있는 나머지 호출부도 같이 본다: `fr_arms.py:188`, `fr_transfer.py:522`, `rule_acid_run.py:386`, `v520_run.py:622`, `lpopt/design/pathfinder.py:268`. (지금까지 이들이 만든 store 행은 전부 short-pair 라 무해했지만, long-pair cell 을 태우는 순간 같은 사고가 난다.)

### R3 (권장, deck/token 정책) — token 공간을 하나로 고정

근본적으로는 **"pattern 은 type_id 공간, deck 은 alias 공간"** 이라는 이중 공간이 사고의 원천이다. 두 선택지:

- **(a) alias 공간으로 통일** — store `pattern`/`case_pair` 를 paramA 에서도 alias 로 적는다. 마이그레이션 비용이 크고 과거 라벨과의 조인이 깨진다. **비권장.**
- **(b) 번역을 유일 지점으로 좁히고 검증한다** — `prepare_cycle1_deck` 안에서만 번역하고(현재 구조 유지), 번역 후 roster 검증(R1)을 필수화한다. **권장.** 코드 변경이 작고 과거 자산과 호환된다.

`coredeck.py` 의 docstring("Fuel *type names* here are the 2-char MASTER aliases (already MASTER-safe)")은 **입력 계약**을 서술하고 있는데 그 계약을 강제하는 코드가 없다. R1 이 그 강제다.

### R4 (필수, 실험) — §14-5 재평가의 **범위를 넓힌다**

report §14-5 는 "parent 20개를 이번 wave 와 같은 자산 경로로 재평가해서 오프셋이 사라지는지" 를 결정적 시험으로 제안했다. 이 메모의 진단이 맞다면 **그 실험은 오프셋을 재현할 뿐 아무것도 고치지 못한다**(같은 결함 경로 = 같은 P6 노심 = parent 도 761–770 으로 나옴 → "오프셋 소멸"로 오독될 위험). 대신:

**R4-a (진단, 2 chain, ~0.03 h — 먼저)**: 임의의 HGD569 자식 pattern 하나를 골라 `%LPD_SHF` 만 손으로 `S3`/`S5` 로 치환한 deck 과 원래 deck 을 MASTER 로 각각 돌린다. 예측: 전자는 ~731 EFPD 대, 후자는 ~766 EFPD 대. 이것이 원인 확정의 **최소 비용 실험**이며, MAS_OUT 의 `CORE LOADING PATTERN` 이 S3/S5 로 돌아오는 것만 봐도 (MASTER 를 끝까지 돌리지 않고) 확인된다.

**R4-b (복구, 160 chain, ~1.8 h — R1/R2 수정 후)**: `intervention_HGD569_f125` 160 chain 을 수정된 경로로 **전량 재평가**한다. parent 재평가가 아니라 **자식 재평가**다. parent 20행은 원래 정상 경로에서 나왔으므로 손댈 필요가 없다.

### R5 (즉시, 무비용) — 데이터 격리

- `data/store/records.parquet` 의 `intervention_HGD569_f125` 160행을 **`valid=False` 로 내리거나** 최소한 report/policy 파이프라인의 명시 제외 목록에 넣는다(현재 report §7·§13.2-5 의 "영구 prospective holdout" 처방은 결과적으로 옳지만 사유가 틀려 있다 — 사유를 "cell 의 OOD 성격"이 아니라 "**deck emission 결함**"으로 갱신).
- `data/policy/steps.parquet` 의 같은 160행도 동일.
- report §5.1·§7·§11-1·§15 에 본 메모를 참조하는 정오표를 단다: **축퇴는 슬롯 기하가 아니라 token 사상 붕괴이며, 두 이상현상은 하나의 원인이다.**

### R6 (참고) — 이미 들어간 guard 의 한계

report AMENDMENT(2026-08-30 23:xx)의 `intervention_wave.py` **core-degeneracy guard** 는 plan 시점에 형제의 core digest 를 비교해 축퇴 쌍을 걸러낸다. 이는 **증상(예산 낭비)** 을 막을 뿐이며, `+35 EFPD` 오프셋(=노심 자체가 틀림)은 **전혀 잡지 못한다.** guard 를 R1 의 roster 검증으로 대체하거나 병행해야 한다.

---

## 7. 재현 절차 (read-only)

```bash
# (1) deck token 확인 — 개입 wave 와 원 campaign
sed -n '/%LPD_SHF/,/%LPD_B&C/p' \
  runs/intervention_wave_r1/intervention_HGD569_f125/produce_cases/\
P6253Z1G06N24_P6253Z2G10N24_f125/*/MAS_INP_cy12.inp | head
sed -n '/%LPD_SHF/,/%LPD_B&C/p' \
  "E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/fpcamp_minfr_hgd569_f125/\
master/produce_cases/P6253Z1G06N24_P6253Z2G10N24_f125/*/MAS_INP_cy12.inp" | head

# (2) MASTER 의 해석 확인
grep -A 12 "CORE LOADING PATTERN" <각 run 의 master_work/worker_00/*/MAS_OUT>

# (3) 축퇴 확인 — fresh label 을 blind 처리하면 20개 3-member class, FOM 불일치 0
#     (본 메모 §4(a) 의 스크립트)
```

## 8. 남는 미확정

- MASTER 가 `P6253Z1G06N24` 를 `P6` 로 귀결시키는 **정확한 파싱 규칙**(고정폭 A2 읽기인지 prefix 매칭인지)은 소스가 없어 확정하지 못했다. 관측(모든 위치 → `P6`, stderr 무음)과 정합하는 가장 단순한 설명이 "앞 2자"이며, **어느 규칙이든 결론은 같다**(roster 에 없는 batch id 는 무검증으로 흡수된다) — 그래서 R1 의 사전 검증이 필요하다.
- carryover(shuffle) 위치까지 `P6` 로 echo 된 기전 — fresh batch 조회 실패가 batch 배열 초기화 전체를 오염시키는 것으로 보이나, 벤더 소스 없이 단정하지 않는다. 영향의 크기(+35 EFPD)는 이미 측정되어 있다.
