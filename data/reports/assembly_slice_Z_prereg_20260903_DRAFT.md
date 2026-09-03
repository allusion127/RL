# 집합체 온디맨드 — **수직 슬라이스 Z** 사전등록 (PRE-REGISTRATION) · **DRAFT**

**작성** 2026-09-03 · **상태** DRAFT (발사 전 · 코드 0줄 · DeCART 0 · MASTER 0 · 로컬 연산 0)
**선행 문서** `assembly_on_demand_design_v2_20260903.md` (설계안 v2) + 재비평 2건
(physics-and-licensing-realism / engineering-feasibility-and-data-flow)
**성격** v2 §8.2 슬라이스를 **집행 가능한 문서**로 승격한 것. 재비평 24항목을 전부 처분했다(부록 R).
**수신** 프로그램 오너. **A5 유지 — 이 문서와 오너 승인 전에는 어떤 발사도 없다.**

> **사용자 지시 (축어, 2026-09-02)**
> "DeCART2D surrogate와 연계하여 AI 모델 자체적으로 필요한 집합체가 있으면 후보 추출 후
> DeCART2D 실계산으로 HGC 생성해서 사용하도록 해"

---

## 0. 이 문서가 v2에서 바꾼 것 — 재비평이 잡은 **집행 불가 항목 5건**

| # | v2가 등록한 것 | 재확인한 사실 (파일:라인) | **이 문서의 처분** |
|---|---|---|---|
| **C1** | S9b = 캠페인 안에서 `post_verify_delivery` 가 MTC 게이트를 친다 | `campaign.py:533-534` `build_delivery_payload` 가 `objective != "flat_power"` 이면 **`None`**; `campaign.py:3053-3062` 이 *"no delivery ranking to verify … gate not run"* 로 조기반환. r2 덱 자신이 이것을 **Defect D1**로 적어 두었고(`runs/fpcamp_minfxy_e1e2_f121_r2/input_deck.inp:37-45`) r2 `status.json` 은 `post_verify_master_calls: 0` | **MTC를 캠페인에서 뗀다.** S9b는 **독립 CLI 단계** `lpopt sdm-mtc --run <run_dir> -i <deck>` (cli.py:680) — 이 경로는 `candidates_from_delivery` 가 비면 **`select_topk_feasible` 로 폴백**하므로(cli.py:741-746) `min_fxy` 에서 **실제로 돈다** |
| **C2** | `[sdm_mtc] top_k = 5` → ≈10 MASTER 콜 | 라이브 게이트는 `constraints.post_verify_top_k` 를 읽고(sdm_mtc.py:1517) 기본값은 **3**(config.py:1240). `[sdm_mtc] top_k` 는 `post_verify_topk` 훅 전용이고 그 훅은 독스트링이 *"additive — NOT wired into the live loop"* | **`[constraints] post_verify_top_k = 5`.** 비용은 **축당 후보당 ~1 콜**(config.py:1227) → SDM 미가동이므로 **arm당 5, 총 10 콜** (2 arm) |
| **C3** | `[verify] keep_success = true` 를 추가한다 | **그런 키는 없다.** r2 덱 자신: *"harvest_maps … forces the verifier's keep_success … there is no separate keep_success key"* (`input_deck.inp:186-190`) | **`[verify] harvest_maps = true` 하나로 충분.** `keep_success` 는 덱에서 삭제 |
| **C4** | 짝지은 대조군 arm B = `E1_E2/f121` | 그 셀은 **ga80** 라이브러리다(r2 덱 `package_root = FEASIBLE_PACKAGE`). 재빌드는 **paramA restart만** 무효화하므로 arm B는 재빌드 효과를 하나도 재지 못한다. 게다가 신규 `Z1_Z2` 셀에는 사전기간이 없어 difference-in-differences 가 **정의되지 않는다** | **arm B = `T6_T4/f121` (paramA)** — deliverable incumbent `bf3a70b2` 의 셀이고 `runs/fpcamp_minfxy_t6t4_f121_r1/input_deck.inp:131,168` 이 `library_id = paramA` + `package_root = data/design/package` 임을 확인했다. **39-타입 라이브러리 위에서 재측정**한다. `E1_E2/f121`(ga80)은 **재실행하지 않는 고정 외부 참조**로만 인용 |
| **C5** | "오케스트레이터의 실제 권고는 box 104" (A4) | **box 104 = 이 로컬 워크스테이션이다.** `realize_lat1600.log` 의 모든 경로가 `C:\Users\MK\Desktop\CT&RPL\…` 이고, 이 머신에 `D:\DeCART_MASTER\BIN\{decart2d1.1m5omp.exe, libiomp5md.dll, prolog41m4.exe, master4.0m4_r1.exe}` 가 **실재**한다(디스크 확인). hostname `mk-ctrp` | **권고 철회.** 사용자 상시 규칙 "로컬 PC 연산 금지" 와 정면충돌한다. **box 104는 이 계획에서 완전히 제외**하고, 원격 호스트만 쓴다 (§3). omp 735–750 s/case 는 **역사적 실측 기록으로만** 인용 |

> **C5의 파급 — v2가 몰랐던 것.** box 104가 로컬 PC라는 뜻은, **저작·라이브러리 재빌드·MASTER
> 부트스트랩을 지금까지 전부 로컬에서 했다**는 뜻이다. 이 슬라이스는 그 셋을 **전부 원격으로
> 옮겨야** 하고, 그것이 §3.0의 스테이징 단계(S2b)가 선택이 아니라 **경로 전체의 전제조건**인
> 이유다. 다행히 `data/design/package/lib/` 안에 **`prolog41m4.exe`(1,451,520 B)와
> `TotalBatcher4.exe`(300,544 B)가 이미 스테이징되어 있다**(디스크 확인) — 패키지를 보내면
> 빌드 툴이 따라간다.

---

## 1. 후보 Z — 완전 지정 (동결)

### 1.1 설계 튜플

| | **Z1′ (68-슬롯 역할)** | **Z2 (53-슬롯 역할, hot)** |
|---|---|---|
| `u_high` = `e1` | **5.50 w/o** | **5.00 w/o** |
| `e2` (zone) | **4.70 w/o** | **4.25 w/o** |
| `du = e1 − e2` | **0.80** — (A) 경계 상한 위, 0.1 격자 위 | **0.75** — 경계 안, 격자 밖 (T3/T4 선례) |
| `ratio = e2/e1` | **0.854545** (`zone_ratio_flag` 0.85±0.03 **pass**, \|Δ\| 0.00455) | **0.850000** (정확) |
| `gd_u` | 4.0 (`GD_CARRIER_ENR`, 고정) | 4.0 (고정) |
| `gd_wt` | **8** | **10** |
| `n_gd` | **20** | **20** |
| `zoning_variant` | **z1** → **(A) pattern = `PB`** | **z1** → **`PB`** |
| Gd 레이아웃 (옥탄트) | **`1:1;4:1;6:4`** | **`1:1;4:1;6:4`** (동일) |
| **`type_id`** | **`P5547Z1G08N20`** | **`P5042Z1G10N20`** |
| **배정 예정 alias** | **`T7`** | **`T8`** |
| 베이스 덱 | `0_APR1400\5.8_5.1\FA\IGD_20\8_20_z1\dec_FA_B03.inp` (동결 `2:2;5:2;6:4`) | `0_APR1400\5.8_5.1\FA\IGD_20\10_20_z1\dec_FA_B05.inp` (동결 `2:2;5:2;6:4`) |
| 핀 이동 | `2:2 → 1:1`, `5:2 → 4:1` (`6:4` 유지) = **2 이동** | 동일 **2 이동** |
| UO2G 캐리어 밀도 | **9.95 g/cc** (gd_wt 8 패밀리, `realize_lat1600.py:232-238`) | **9.88 g/cc** (gd_wt 10 패밀리) |
| 저작 출력 | `5_RL/templates_lat1600/5.8_5.1/FA/IGD_20/8_20_z1/dec_FA_P5547Z1G08N20.inp` | `…/10_20_z1/dec_FA_P5042Z1G10N20.inp` |

**alias `T7`/`T8` 의 근거 (동결).** `_alias_pool()`(spec.py:150) = letters `"PQSTUVWXYZABCDEFGHIJKLMNO"` × digits 0–9.
`registry.json` 실측 37개 = P0–P9(10) + Q0–Q9(10) + S0–S9(10) + T0–T6(7). **다음 두 개는 T7, T8이다.**
등록 후 `registry.json` 의 alias 값이 T7/T8이 아니면 **S6에서 멈춘다** (풀 순서 가정이 깨진 것).

**충돌 없음 (전수 확인).** `designs.json`/`registry.json` 의 37 `type_id` 중 `P5547Z1G08N20` 도
`P5042Z1G10N20` 도 없다.

> **★ R2 — 후보 Z2 는 동결 유지 (독립 재확인, 2026-09-03).** `fr135_feasibility_scoping_20260903.md`
> §3.1·§3.2 의 442-예측 서로게이트 스캔이 `CBC ≤ 1500` ∧ `cyclen ∈ [620,645]` ∧ `contrast ≥ 0.026`
> 게이트 하에서 **hot-role `FF_hot` 최소(1.1196)** 를 정확히 이 설계 계열(`n_gd 20` /
> `1:1;4:1;6:4` / u5.00–5.15)로 되돌려 주었고, `1:1;4:1;6:4` 는 `n_gd 20` **안에서 최적**이었다
> (89개 유효 레이아웃 전수). `u_high` 를 5.00 → 5.10 으로 옮기면 `FF_hot` 이 −0.0005 (sigma 무의미)
> 개선되고 CBC 는 1389 → 1426 으로 오른다. → **Z1′/Z2 튜플을 변경하지 않는다.**
> objective 전환(§7.2)은 **후보 선택을 바꾸지 않는다** — 바꾼 것은 판정 축뿐이다.

> **★ 등록된 함정 — `type_id` 는 0.1 w/o 로 양자화된다.** `e2x10 = int(round(e2*10))`(spec.py:78).
> OPSCREEN 원안의 `e2 = 4.6750` 도, 이 문서의 `e2 = 4.70` 도 **똑같이 `47`** 로 접힌다 →
> 두 변종은 `type_id` 로 구별되지 않는다. 그러므로 **`designs.json` 에 `e2` 를 정확값(4.70)으로
> 기록**하고, `DesignRegistry.alias` 는 **같은 `type_id` 인데 기록된 설계 튜플/`gd_positions` 가
> 다르면 raise** 해야 한다 (작업 #1-(4)). 이 가드 없이는 4.6750 변종이 미래에 `T7` 의 HGC를
> 조용히 덮어쓴다.

### 1.2 왜 `e2 = 4.70` 인가 (v2 유지, 재확인)

OPSCREEN 원안 `u5.50/4.6750` 은 `du = 0.825` 로 (A)의 `du ∈ [0.40, 0.80]` **경계 밖**이고
(`predict.py:109-112`, `_snap_to_grid` 가 `lo−1e-9 ≤ v ≤ hi+1e-9` 강제), 경계 밖 외삽은
`n_gd = 28`(k BOC **+1,750 pcm**) 선례가 금지한다. `e2 = 4.70` 이면 `du = 0.80` 으로 경계·격자
**위에 정확히** 앉고 준법 허용오차 안이다.

### 1.3 ★ 열거 창의 정정 — **A1(5.50 w/o)은 구속한다** (v2 N1 철회)

v2 §4.2 결론 1과 D.2-N1은 "`e2 = 0.85·e1` 이므로 `du = 0.15·e1 ≤ 0.80` → `u_high ≤ 5.333`,
따라서 5.50 상한은 구속하지 않는다" 였다. **그러나 §10-A2가 비율을 0.85 ± 0.03 창으로 푼다.**
그 창에서는

```
du = (1 − ratio)·e1 ,  ratio ∈ [0.82, 0.88]
du ≤ 0.80  ∧  ratio ≥ 0.82   ⟹  e1 ≤ 0.80/0.18 = 4.444 ... (하한 ratio에서)
du ≤ 0.80  ∧  ratio ≤ 0.88   ⟹  e1 ≤ 0.80/0.12 = 6.667  (상한 ratio에서)
```

즉 `u_high` 의 상한은 **선택한 ratio에 따라 4.44 – 6.67 사이에서 움직이고**, 창 전체를 열면
**A1의 5.50이 다시 구속한다.** → **v2 D.2-N1의 헤드라인("5.50 상한은 구속하지 않는다")을 철회한다.**

> **동결된 열거 규칙 (라운드 1).**
> `u_high ∈ [5.00, 5.50]` 0.05 step (**11점**) × `e2` 는 **`ratio ∈ [0.82, 0.88]` ∧ `du ∈ [0.40, 0.80]`
> 을 동시에 만족하는 0.05 격자점** × `gd_wt ∈ {6,8,10}` × `(n_gd, layout) ∈ 89 쌍` × `z ∈ {z1, z2}`,
> `gd_u = 4.0` 고정.
> **설계 개수는 이 문서에서 계산하지 않는다** — 로컬 연산 금지 규칙 때문이며, `screen.py --enumerate`
> 가 HOST_238에서 산출하고 그 수를 슬라이스 결과 문서가 인용한다. v2 §4.3의 **3,738은 `ratio = 0.85`
> 고정 전제의 수**이므로 이 창에서는 **인용하지 않는다.**

### 1.4 `n_gd` 를 {12,16,20,24}로 묶는 **더 강한 근거** (재비평 A13 수용)

v2 §4.1 정정 1은 lpopt 툴링(템플릿 디렉터리 부재, `spec.py:64-67`)만 인용했다. 서로게이트 자신이
같은 답을 준다: `SURROGATE_USAGE.md:143` — *"n_gd ∈ {12,16,20,24}만 신뢰"* (n_gd = 28 → k BOC
**+1,750 pcm**), `:29` 의 격자표도 `n_gd : 12,16,20,24`. → **이것은 툴링 한계가 아니라 모델 유효범위
경계다.** 저-Gd(0/4/8) 확장은 **모델 재학습 없이는 불가**이며 (A) 재학습은 이 체크아웃에서
불가능하다(부록 B-14). **라운드 1·2 모두에서 닫는다.**

> **경계 근접 경고 (등록).** `SURROGATE_USAGE.md:148,297` 은 k 오차 최대 영역이 **설계공간 코너
> (`gd_wt = 10 × n_gd = 24`)** 라고 적는다. **Z2는 `gd_wt = 10 × n_gd = 20`** 으로 한 축이 코너에
> 접해 있다. G-H4(§4.4)의 100 pcm 문턱은 **Z2에서 가장 타이트할 것으로 예상**되며, 그 예상을
> 여기 미리 적어 둔다.

---

## 2. ⓪ σ_chain 소급검증 실험 — **자체 통과바를 가진 독립 실험**

### 2.1 슬라이스와의 관계 — **비차단(non-blocking)으로 확정**

physics 재비평 #13은 정확하다: v2의 §3.4 사다리는 어떤 등급에서도 **행동을 바꾸지 않으면서**
슬라이스 앞에 0.5일로 놓여 있었다. 비평이 허용한 두 선택지 중 **후자를 택한다.**

> **결정.** **슬라이스는 무조건 실행한다** (σ 결과와 무관). task #0은 슬라이스의 **선행이 아니라
> 병렬 트랙**이고, 그 통과바는 **라운드 2의 자동 발사 규칙 F1–F5(v2 §3.3)에만 구속력**을 갖는다.
> 이유: 슬라이스의 판정은 **측정된 `ΔF_r`** (2026-09-03 축 전환 전에는 `ΔF_xy`, §7.2·CL-1)
> 이지 예측이 아니다 — 사슬이 틀려도 측정은 유효하다.

### 2.2 실험 사양 (사전등록)

**집행 위치.** HOST_238 (`ssh -p 8022 USER@HOST_238`), `~/lpopt_ws`, venv, 스토어 사본
`scratch/records_r2_76793.parquet`. **읽기 전용. DeCART 0 / MASTER 0 / 로컬 0.**

**절차 P1–P5 는 v2 §3.4를 그대로 계승**하되 **두 곳을 정정**한다:

- **정정 α (physics #5 수용).** `+0.0614` 는 **이 사슬의 측정이 아니다.**
  `minfxy_E1E2_f121_r2_results_20260831.md:133` 의 그 행은 `ratio 1.0588 · F̂_r` 이고 `F̂_r` 은
  **s1j 헤드의 예측**(같은 표에서 자체 오차 +0.0439)이다. 이 설계의 사슬은 `F̂_r` 을 헤드가 아니라
  `A · node_peak · FF_hot` 에서 얻는다. → 표기를 **"`r · F̂_r(s1j)` — 헤드 원천, 사슬 원천 아님, n=1"**
  로 바꾸고, **§3.2의 σ 주장은 해석적 전파(0.065) 하나에만 의존**시킨다.
  **"두 독립 추정이 일치한다"는 문장은 삭제한다** (v2 §3.2 말미, §0.1-3, D.1-#1/#2).
- **정정 β.** P1의 홀드아웃 재적합에서 `A` 는 `s09b_contrast.py:49` 와 동일한 최소자승 스케일
  (`A = Σ(F_r·p·FF)/Σ(p·FF)²`, `p` = **예측** node_peak)로 재추정한다 — 측정 node_peak을 넣지 않는다.

### 2.3 통과바 (사전등록, 사후 조정 금지)

| `σ_chain,paired` (셀 간) | 판정 | **귀결 (라운드 2 자동화에만 적용)** |
|---|---|---|
| **< 0.005** | 사슬이 결정문턱을 해상 | F2의 `k = 2` 바 그대로 |
| **0.005 – 0.020** | 순서키로만 유효 | 바 = `2σ`; 후보가 그 바를 넘는지 **재계산**해서 넘을 때만 발사 |
| **> 0.020** | 크기 판정 불가 | **F1–F5 트리거 폐기.** 라운드 2도 **짝지은 물리실험**으로 수행, 트리거 설계는 v3 이월 |

**등록된 기저 시나리오.** 지배항 `node_peak` rms **0.036**(n=15)은 후보마다 contrast가 다르므로
차분에서 상쇄되지 않는다 → **세 번째 행(>0.020)이 유력**하다. 이것을 미리 적는 이유는, 그 경우에도
슬라이스가 그대로 유효하기 때문이다.

### 2.4 이 실험이 슬라이스 문서에 남기는 것

슬라이스 결과 문서는 σ 값을 **판정 문장의 선택자**로만 쓴다:
σ ≤ 0.020 이면 "예측된 부호를 측정으로 확인/반증했다", σ > 0.020 이면 "부호를 처음 측정했다".
**어느 쪽도 PRIMARY/SECONDARY/NULL 의 수치 기준을 바꾸지 않는다.**

---

## 3. ③ DeCART2D 실계산 — 실행 계획

### 3.0 ★ 호스트 재배치 (C5의 귀결)

| 단계 | **호스트** | 근거 |
|---|---|---|
| S1 스크리닝 (A) | **238** (`USER@HOST_238:8022`) | GPU/CPU 모두 가능; **선행 S0c** 필요 (§3.1) |
| **S2 덱 저작** | **238** | 저작은 Python 실행 = 연산. **로컬 금지.** venv + lpopt 소스가 이미 있다 |
| S3 DeCART2D 웨이브 | **181** (`USER@HOST_181`) | 실증된 직렬 큐 + exe/XS 해시 (§3.2) |
| S5 라이브러리 재빌드 (TotalBatcher) | **199** (`USER@HOST_199`) | 패키지 정본이 가야 할 곳이고 `lib/` 에 `prolog41m4.exe`/`TotalBatcher4.exe` 가 동봉된다 |
| S7 MASTER 부트스트랩 ×9 | **199** | `C:/DeCART_MASTER/BIN/master4.0m4_r1.exe` — `fpcamp_minfxy_T6T4_f121_r1_199.inp:122` 실사용 |
| S8/S9 전이·캠페인 | **199** | lpopt 생산 호스트 |
| 로컬 PC (`mk-ctrp`, = box 104) | **연산 금지.** Read/Grep/Glob/head/sed + 파일 전송만 | 사용자 상시 규칙 |

> **★ S0b — fleet 정책 충돌을 먼저 해소한다.** `5_RL/autoeng.toml:59-66` 의 `[fleet]` 블록은
> *"181 is NEVER used"* 라고 적고 `forbidden = ["HOST_181", …]` 에 181을 넣는다.
> 이 슬라이스는 181에서 DeCART를 돌린다. → **오너가 이 라운드에 한해 명시적 예외를 기록**하거나
> `[fleet]` 정책을 갱신하기 전에는 S3를 시작하지 않는다. (autoeng 자동 엔진은 이 슬라이스를
> 구동하지 않지만, 같은 워크스페이스 안에 상반된 상시 정책을 남겨 두지 않는다.)

### 3.1 ★ S0c — (A) 체크포인트 접근 확보 (S1의 경성 선행, 재비평 #11 수용)

- (A) 모델은 **`USER2` 계정** `/home/USER2/lattice_surrogate/kpin_pa` 에 있다(`SURROGATE_USAGE.md:148-165`);
  프로젝트 계정은 `USER@…:8022` 다.
- `Engines.__init__`(predict.py:899)은 `dataset/bu_grid.npy` + `runs_root` 앙상블 체크포인트를 요구한다.
- **`screen1600.csv` 는 이 머신에 없다**(부록 B-13) → S1은 예측을 **재사용이 아니라 재생성**한다.

> **S0c 산출물:** `USER` 홈에서 읽을 수 있는 (A) 트리 사본 + `--self-test` 통과 로그 + 체크포인트
> SHA-256 매니페스트. **S0c 미완이면 S1을 시작하지 않는다.**

### 3.2 ★ S2b — 템플릿·XS 스테이징 (S3의 경성 선행, 재비평 A5 수용)

181은 DeCART exe와 XS 라이브러리는 갖고 있지만(`2_LP/artifacts/run_decart_eq_xesm_queue_181.ps1:4-5`
가 `D:\DeCART_MASTER\BIN\decart2d1.1m5.exe` + `D:\DeCART_MASTER\LIB\DML-E71N047G018-PV01-cr08.BIN` 을
쓰며 그 해시 대조가 과거에 통과했다), **`0_APR1400` 템플릿 트리도 `templates_lat1600` 도 paramA
패키지도 없다.** 181의 자산 행은 v2 §5.3에서 "kit_frontier (2026-07-29 stale)" 이다.

**스테이징 세트 (SHA-256 매니페스트 동반, 파일 복사만 — 연산 아님):**

```
→ 238 :  0_APR1400/5.8_5.1/FA/IGD_20/8_20_z1/dec_FA_B03.inp      (Z1′ 베이스)
         0_APR1400/5.8_5.1/FA/IGD_20/10_20_z1/dec_FA_B05.inp     (Z2 베이스)
         5_RL/lpopt/  (소스), 5_RL/realize_lat1600.py             (저작 참조 구현)
→ 181 :  S2가 저작한 dec_FA_P5547Z1G08N20.inp , dec_FA_P5042Z1G10N20.inp
         (nxfile 줄은 181의 D:\DeCART_MASTER\LIB\DML-E71N047G018-PV01-cr08.BIN 로 재작성)
→ 199 :  data/design/package/ 전체 (lib/ hgc/ bases/ cores/ registry.json designs.json
         + 동봉된 prolog41m4.exe / TotalBatcher4.exe), 0_APR1400/ (부트스트랩 참조용)
```

**S2b 완료 판정:** 각 호스트에서 `Get-FileHash`(또는 `sha256sum`) 결과가 매니페스트와 **바이트 일치**.

### 3.3 실행 파라미터 (동결)

| 항목 | 값 | 근거 |
|---|---|---|
| 호스트 | **HOST_181** (`USER@HOST_181`) | §3.0 |
| 실행 루트 | **`C:\Users\USER\lpopt_work\assembly_slice_Z_20260903\`** | 181의 D:는 7.1 GB 밖에 없다 (v2 §5.3) |
| exe | **`D:\DeCART_MASTER\BIN\decart2d1.1m5.exe`** (**직렬**) | omp exe는 `libiomp5md.dll` 이 181에 없다 |
| exe SHA-256 (선검사) | `5F0F10F10BD4CC6546173C266DA3FDE72BDF1A09A191C59629FF7B4B0AF006CE` | `run_decart_eq_xesm_queue_181.ps1:6` |
| XS 라이브러리 | `D:\DeCART_MASTER\LIB\DML-E71N047G018-PV01-cr08.BIN` | 동상 `:5` |
| XS SHA-256 (선검사) | `AEF86EEBFB8B6398D0A45164C70E0FB04FCB5066546A12A3BBAB9106AF64E377` | 동상 `:7` |
| 스레드 | **`OMP_NUM_THREADS = 1`** | 동상 `:16` |
| 동시성 | **2** (전역 `decart2d1.1m5.exe` 프로세스 수 < 2) | 동상 `:32` — **실증된 레시피 그대로.** 2 케이스뿐이므로 상한 4는 의미가 없다 |
| 스레드 회계 (정정) | 직렬 프로세스 2개 = **181의 32 스레드 중 2** | v2 A4의 "큐 상한 4 (32스레드)" 는 오독 (재비평 A9 수용) |
| 타임아웃 | **7,200 s/case** | 199 직렬 실측 3,084 s 의 2.33배 (config 기본 5,400은 얇다, R19) |
| 멱등성 | `FA_<alias>.HGC` + `.out` 존재 시 재실행 안 함 (`run_batch`, lattice.py:409) | 기존 |

### 3.4 런타임 — **두 행을 같은 규약으로 가격한다** (재비평 #8 수용)

| 시나리오 | 케이스당 | **2 케이스 2-wide** | 2 케이스 직렬 |
|---|---:|---:|---:|
| 181 직렬 exe (계획, 199 실측 대입) | 3,084 s | **0.86 h** | 1.71 h |
| (참고) box 104 omp — **사용 금지** | 735–750 s | 0.21 h | 0.42 h |

> **귀속 정직성 (두 겹).** ① 3,084 s 는 **199**의 실측(V09–V14)이고 **181은 미측정**이다.
> 181은 9950X(16C/32T @4.3 GHz)이므로 더 빠를 가능성이 크지만 **"보수적이되 귀속이 틀린" 숫자**임을
> 명시한다. ② `realize_lat1600.log:31-34` 는 `wall=750/750/735/735 s` 만 기록하고 **스레드 수도
> 동시성도 기록하지 않는다** — "omp 4-way" 는 `design_lat1600_104.inp:48` 이 omp exe를 지정한다는
> 사실에서 나온 **추론**이다 (재비평 #7 수용). 어차피 §3.0에서 사용 금지이므로 계획에 들어가지 않는다.

### 3.5 산출물 · 디스크

| 파일 | 크기 | 스탬프할 해시 |
|---|---:|---|
| `FA_T7.HGC` / `FA_T8.HGC` | **정확히 7,395,955 B** (G-H1b) | SHA-256 → `designs.json.hgc_sha256` |
| `FA_T7.out` / `FA_T8.out` | ~27.3 MB | — |
| `FA_T7.sum` / `FA_T8.sum` | ~767,500 B | SHA-256 |
| 저작 덱 `dec_FA_<type_id>.inp` | — | SHA-256 → `designs.json.deck_sha256` |
| stdout | ~88 KB | `JOB FINISHED` 문자열 존재 검사 |
| **케이스당 합계** | **≈ 35.6 MB** | |

> **★ `.sum` 회수 경로 (재비평 A11 수용).** `lattice.harvest` 는 `.HGC`/`.out` 만 만들고
> **`(wd/"decart.inp").unlink(missing_ok=True)`(lattice.py:346)로 스테이징한 덱을 지운다** →
> 작업 #13b의 `stage_hgc` 는 **작업 디렉터리가 아니라 `templates_lat1600/…/dec_FA_<type_id>.inp`
> (또는 `DecartRun.deck_path`)에서** 덱을 복사해야 한다. `.sum` 은 DeCART가 `FA_<alias>.sum` 으로
> 남기며(181 큐 스크립트가 그 파일을 읽는다), `harvest` 가 그것을 보존하도록 확장한다.

---

## 4. ④ HGC → 라이브러리 · 패키지 재생성 · 게이트

### 4.1 재빌드 (199)

```
S0   스냅샷 :  lib/ + bases/ + cores/ + registry.json + designs.json  →  E:\lpopt_archive\slice_Z_20260903\
                (.bak 은 한 세대뿐 — 두 번째 재빌드가 유일한 롤백을 파괴한다)
S5   빌드   :  스테이징 디렉터리에 MAS_REF(2,008 B) + FA_*.HGC ×39 + prolog41m4.exe + TotalBatcher4.exe
                PATH 앞에 스테이징 디렉터리 → TotalBatcher4.exe (인자 없음)
```

### 4.2 게이트 G-H1 … G-H4 (HGC 단계, 181 또는 199)

| 게이트 | 검사 | 판정 |
|---|---|---|
| **G-H1 구조** | `%TITL` **334** = DEPL 62 + BRANCH 16×17; `%DIST`/`%MACX`/`%MICX`/`%ADFT` 각 334; 말미 `%FINE` 1 | FAIL → 중단 |
| **G-H1b 크기** | `n_gd = 20` → **정확히 7,395,955 B** | FAIL → 중단 |
| **G-H1c 유효성** | `_hgc_looks_valid`(lattice.py:374) + stdout `JOB FINISHED` | FAIL → 중단 |
| **G-H2 Gd 인구조사** | `count_gd_pins_from_hgc`(fuel_types.py:554) == **20** (양쪽) | FAIL → 중단 |
| **G-H4 스크린 대조 (회귀검사)** | BU ≥ 0.2 전 구간 \|k_(A) − k_DeCART\| ≤ **100 pcm**; \|FF_(A) − FF_DeCART\| ≤ **0.0021** | **FAIL → 여기서 멈춘다** (§7 중단점) |

문턱 100 pcm / 0.0021 은 **T3–T6 홀드아웃의 실측 최대치**(OPSCREEN.md:165-179)이므로 회귀검사이지
첫 측정이 아니다.

### 4.3 게이트 G-H3 계열 (라이브러리 단계) — **산술 동결 + 출처 정정**

**N = 37 + 2 = 39, 전부 `n_gd > 0`:**

```
MAS_XSL  =  2,010 + 385,849 × 39  =  2,010 + 15,048,111  =  15,050,121 B     (등식)
MAS_HFF  =            404,857 × 39  =                        15,789,423 B     (등식)
```

**검증 근거표 (★ 출처 셀 정정 — 재비평 A12/#6 수용).** 디스크 재확인:

| N | **출처 (정정)** | MAS_XSL | `2,010 + 385,849·N` | MAS_HFF | `404,857·N` |
|---:|---|---:|---:|---:|---:|
| 11 | **`lib.snap_20260811/MAS_XSL.bak`** | 4,246,349 | 4,246,349 ✓ | 4,453,427 | 4,453,427 ✓ |
| 12 | `0_APR1400/*/hgc` | 4,632,198 | 4,632,198 ✓ | — | — |
| 16 | `2_LP/artifacts/seq_canary` (V01/V02 = `n_gd` 0) | 6,158,818 | `2,010 + 385,849·14 + 377,461·2` ✓ | 6,477,712 | 6,477,712 ✓ |
| 33 | **`lib/MAS_XSL.bak`** (= `lib.snap_20260811/MAS_XSL`) | 12,735,027 | 12,735,027 ✓ | 13,360,281 | 13,360,281 ✓ |
| 37 | 현행 `lib/MAS_XSL` | 14,278,423 | 14,278,423 ✓ | 14,979,709 | 14,979,709 ✓ |
| 80 | `3_GA_Surrogate/FEASIBLE_PACKAGE` | 30,869,930 | 30,869,930 ✓ | — | — |
| **39** | **이 슬라이스의 기대값** | — | **15,050,121** | — | **15,789,423** |

(v2 표는 N=11 행을 `lib/*.bak` 으로, N=33 행을 `lib.snap_*` 으로 **뒤바꿔** 적었다. 산술은 정확했다.)

| 게이트 | 검사 |
|---|---|
| **G-H3** | 위 두 등식 (허용오차 없음) |
| **G-H3b 핵종 로스터** | 새 `COMP FA_T7`/`FA_T8` 블록의 핵종 로스터가 기존 블록과 **정확히 일치** (`BP01*`/`SB10*`/`MACX*`/`CRD1*` 포함) ∧ COMP 헤더 `BURN VAR DMOD ADF DUM = 62 17 6 0 0` |
| **G-H3c COMP 순서** | `LibraryBuild.set_names` 의 **기존 37개 prefix 순서 불변**, `T7`/`T8` 이 뒤에 append (R21 — 기전은 단정하지 않고 결과를 검사) |

### 4.4 패키지 재생성 (S5b) — 생략 불가

```
1. cores/ 10개 폴더 전부 write_core_template() 재생성 (새 39-alias 로스터):
   P0_P1, Q1_Q2, Q7_Q8, T1_T4_f117, T3_T4, T5_T6, T5_T6_f101, T5_T6_f117, T5_T6_f81, T6_T4
   %GEN_DIM :  "10 10 27 40 42"  →  "10 10 27 42 44"       (nbatch +2, ncomp +2)
2. data/design/synth_decks/ purge (paramA 쌍 18 + ga80 쌍 11 캐시)
3. bases/ 8개 pair 재부트스트랩 (§5의 9회 중 8회)
```

근거: `cores/T6_T4/bootstrap/MAS_INP_cy02.inp:27` 이 현재 `10 10 27 40 42`; `validate_reload_deck`
(assets.py:296-330)이 `%GEN_DIM` 불일치 덱을 **Popen 이전에 거부**하고, `_resolve_template`
(assets.py:718-760)은 디스크의 `cores/` 덱을 **차원 검사 없이 우선 사용**한다.

**S5b 통과 판정:** 재생성 후 **기존 paramA 쌍 10개 전부**가 `validate_reload_deck` 을 통과한다.
하나라도 실패하면 **캠페인 전에 멈춘다**.

---

## 5. ⑤ MASTER 스모크 (199)

| 게이트 | 검사 | 근거 |
|---|---|---|
| **G-H5a cy1 덱** | `%GEN_DIM` 차원 ∧ `%LPD_BCH` 로스터 ∧ `%LPD_C&X`/`%LPD_HFF` 이름이 `MAS_XSL`/`MAS_HFF` 에 존재 | **`validate_reload_deck` 을 쓰면 안 된다** — `%LPD_BCH` 를 담은 덱을 구조적으로 거부(assets.py:325-327) |
| **G-H5b reload 덱** | cy ≥ 2 덱에 `validate_reload_deck` | 원래 용도 |
| **G-H5c 수렴** | `bootstrap_max_cycles = 16` + `cy1_cap_efpd` 안에서 `make_band_restart` 의 5-FOM 비교가 **2회 연속 안정** (T6_T4 선례 11 사이클) | config.py:907,915 |
| **G-H6 덱 에코** | MASTER 출력 에코가 `T7`/`T8` 을 실제 이름으로 부른다 | HGD569 진단 선례 |

**부트스트랩 9회** = 기존 `bases/` 8개 pair 재부트스트랩 + 신규 `Z1_Z2` 1회
(`MAS_XSL` 재빌드가 모든 `MAS_RST.*` 를 무효화한다 — `LIBRARY_BUILD.md §5`).
**첫 번째는 `T3_T4`** 로 돌린다 — 39-COMP MASTER 스모크의 최소 단위이자 lat1600 선례와 동일.

---

## 6. ④/⑤ 등록과 서빙 — **코드 변경 0으로 캠페인이 새 타입을 본다**

### 6.1 `designs.json` (유일한 스키마 변경)

`gd_positions` 를 **선택 → 필수**로 승격(현재 37행 중 4행에만 존재). 신규 필드:
`provenance = "on_demand_slice_Z"`, `e2`(정확값, §1.1 함정 참조), `screen_ff`, `screen_k0`,
`screen_crossing_bu`, `screen_model_sha`, `screen_pattern`(**`PB`**), `decart_wall_s`,
`hgc_sha256`, `deck_sha256`.

### 6.2 동기화 · 재-ingest (S6b)

`paramA_rows`(fuel_types.py:1625-1690)는 `designs.json` **과** `hgc/FA_*.out` 둘 다 필요하고,
`ingest_fuel_types` 를 **캠페인 호스트에서 다시 돌려야** `data/store/fuel_types.parquet` 가 갱신된다.

> **동기화 세트:** `lib/`, `hgc/`(`.HGC` + `.out` + 신규 `.sum` + `dec_FA_*.inp`), `registry.json`,
> `designs.json`, `cores/`(재생성본) — SHA-256 매니페스트 동반.
> **통과 판정: 199에서 `fuel_types.parquet` 의 paramA 행이 37 → 39.**

### 6.3 캠페인이 새 타입을 쓰는 데 필요한 코드 변경 = **0줄**

```
[case]   pair = "Z1_Z2"        (= T7_T8 의 type_id 공간 이름)
[model]  library_id = "paramA"           ← 새 id를 만들지 않는다
[verify] package_root = "data/design/package"
[produce] template_fallbacks = []
```

`library_id` 를 `paramA` 로 유지하는 이유: 새 id(`odA`)는 `_REGIME_CYCLE_BURNUP_MWD_KG`
(featurize.py:63-66)에 항목이 없어 **library-mean 22.0 폴백**으로 피처를 조용히 오염시킨다.
**cond_schema 는 v8, 챔피언은 s1i** — 새 타입은 **재학습 없이** 기존 채널로 채점된다.
**v9(§7.3의 HGC 유도 채널)는 라운드 2 이후 과제**이며 이 슬라이스에서 손대지 않는다.

**콜드스타트 해소 (S8).** 브랜드 뉴 pair는 스토어 행 0개 → `elite_frac 0.65` 가 굶는다(J5_J6 선례).
`fr_transfer.py --target-pair Z1_Z2 --k 32` 로 **32-체인 고정-LP 전이 스윕**을 먼저 병합한다.
**자산 사다리는 레벨 0(native)** — 부트스트랩이 만든 자기 restart를 쓴다(레벨 3 폴백은 r1 ga80 사고 원인).

---

## 7. ⑥ 캠페인 설계 · 마크 · 게이트

### 7.1 arm 구성 (C4의 귀결)

| arm | 셀 | 라이브러리 | 콜 | 역할 |
|---|---|---|---:|---|
| **A** | **`Z1_Z2` / feed 121** | paramA **39 타입** (재빌드본) | 100 | 처리군 — 온디맨드 타입을 fresh로 사용 |
| **B** | **`T6_T4` / feed 121** | paramA **39 타입** (재빌드본) | 100 | **짝지은 대조군 겸 부정 통제.** 신규 타입을 **쓰지 않지만** 재빌드된 라이브러리·새 restart 위에서 돈다 |
| (참조) | `E1_E2` / feed 121 | **ga80** (불변) | 0 | **재실행하지 않는다.** 프로그램 incumbent `a785eded` 1.5295의 고정 외부 기준점 |

**추정량 (정정).** v2의 "difference-in-differences" 는 **정의되지 않는다** — `Z1_Z2` 는 사전기간이
없는 신규 셀이다. → **횡단면 대비**로 재정의한다:

```
[PRIMARY 축 = F_r]  (objective 전환, §7.2.1)
재시작·재빌드 효과  R_Fr  =  F_r(arm B, 39-타입 라이브러리)  −  1.4857  (bf3a70b2, 37-타입 실측)
연료 효과 (귀속)    ΔF_r  =  F_r(arm A)  −  [ F_r(arm B)  −  R_Fr ]   … R 을 먼저 빼고 계산한다

[보고 전용 축 = F_xy]  (같은 규약, 마크 아님)
R_Fxy = F_xy(arm B) − 1.5322 ;  ΔF_xy = F_xy(arm A) − [ F_xy(arm B) − R_Fxy ]
```

**arm B가 반드시 paramA여야 하는 이유:** 재빌드는 **paramA restart만** 무효화한다. ga80 셀을
대조로 두면 재빌드 효과를 하나도 재지 못한 채 연료 + 셀 + 라이브러리 + restart 를 동시에 교란한다.

**검정력 주의 (등록).** `ΔF_xy = 0.0000` (r1 phase-2 40/40, pinbu r1 M2 20/20 = 60/60)은
**replay 재현성**이지 restart-변경 민감도가 아니다(`pinbu_wave_minfxy_r1_results_20260830.md:101`
*"every F_xy number … is replay-exact"*, 동일 덱·동일 restart). **restart-변경 잡음은 미측정이며
arm B가 그것을 처음 잰다.**

### 7.2 목적함수 · 챔피언 — **★ objective = 사용자 지정 `min F_r` (2026-09-03 갱신)**

#### 7.2.0 `F_r` 의 정의 (등록, 축어)

> **이 문서에서 `F_r` 은 MASTER EDIT3 의 `max FRP` — 즉 *핀* 반경출력첨두(F_ΔH 계열)를
> *전 연소단계에 걸쳐* 최대취한 값**이다 (`lpopt/data/extract_a.py:368` `"f_r": _f(metrics,
> "max_frp")`, `lpopt/data/fuel_types.py:991`, `lpopt/data/fxy.py:27` *"the design limit applies
> throughout the cycle"*). **집합체 반경출력첨두가 아니다.**
> **집합체 수준 첨두는 `node_peak`** (BOC 집합체평면 `max_i p_i`, 다중도 가중;
> `lpopt/data/flatness.py:20,203`) 이며, 이 문서에서 `node_peak` 은 **SECONDARY 로만 보고**한다
> (§7.5). 두 양은 엘리트 꼬리에서 서로 상쇄하므로
> (`fr135_feasibility_scoping_20260903.md` §1.3(b): 셀 내부 지수 −0.063 … +0.480)
> **하나로 다른 하나를 대리하지 않는다.**

#### 7.2.1 objective 전환 — PRIMARY 축이 `F_xy` → `F_r`

사용자 목표문(2026-09-03)이 **objective 를 사용자 지정 `min F_r` 로 고정**했다
(`fr135_feasibility_scoping_20260903.md` §5.3 오너 결정항목 1, §6.3-R1).
→ **양 arm 의 캠페인 objective 를 `min_fxy` 에서 `min_fr_max_cycle` 로 바꾼다.**

```toml
[acquisition]
objective            = "min_fr_max_cycle"   # config.py:263 · _VALID_OBJECTIVES(config.py:1699)
minfr_lambda         = 1000.0               # config.py:275 — exploit = cyclen_LCB − λ_Fr·F_r_UCB
minfr_pin_bu_limit   = 78.0                 # config.py:300 (= 측정 80 − 2.0 모델 마진)
```

`min_fr_max_cycle` 은 **F_r 을 PRIMARY 최소화 축으로 두고 cyclen 최대화를 2차 tie-break** 로
쓰며, `F_r ≤ f_r_limit`(1.55) 이 하드 제약으로 **복귀**한다(config.py:252-255).
`λ_Fr = 1000` 은 `0.01` 의 F_r 삭감이 10 EFPD 와 같은 값을 갖도록 크기조정된 기본값이다
(config.py:271-274) — **F_r 이 cyclen 을 엄격히 지배**한다. 이 슬라이스는 그 기본값을 쓴다.

`F_xy` 는 **제약으로만** 남는다: `f_xy_limit = 1.65` (config.py:541-542, 사용자 결정 2026-08-29).
**`F_xy` 는 더 이상 마크가 아니다** — 보고 전용 축이다(§7.5).

#### 7.2.2 두 모드 — (cyclen, B_d) 를 Pareto/target 축으로 (등록된 readout)

사용자 목표문은 두 모드를 요구한다. **이 슬라이스는 같은 100-콜 arm 위에서 둘을 모두
readout 으로 산출**한다 (추가 MASTER 콜 0):

```
MODE-P (Pareto)   minimize  F_r                       # [acquisition] objective = min_fr_max_cycle
                  report    frontier over (cyclen, B_d)   # 최대화 축 2개
                  gates     F_r ≤ 1.55, F_xy ≤ 1.65, CBC ≤ 1600, F_q ≤ 2.41,
                            |AO| ≤ 0.30, 측정 pin BU ≤ 80, MTC ∈ [−54, +9]
                  산출      arm 의 전 클린 행에 대해 (F_r, cyclen, B_d) 3-축
                            비지배(non-dominated) 집합 + 각 정점의 node_peak

MODE-T (target)   targets   cyclen ∈ [c ± δ] ∧ B_d ≥ b   # 사용자가 값을 지정
                  minimize  F_r  s.t. targets as HARD constraints
                  집행      이 슬라이스에서는 **사후 밴드 제한 readout** 으로 집행한다
```

> **★ 등록된 코드 사실 (MODE-T).** `min_fr_max_cycle` 에는 **cyclen 밴드 노브가 없다.**
> `[acquisition]` 의 cyclen 밴드 노브는 `minfxy_cyclen_lo/hi`(config.py:323-324, **min_fxy 전용**)
> 와 `fuelcost_cyclen_lo/hi`(config.py:340-341, **min_fuel_cost 전용**) 뿐이다.
> cyclen **과** 방출연소도를 **동시에 네이티브로 게이트**하면서 `F_r` 을 최소화하는 코드경로는
> `[criteria]` **user_criteria** 캠페인(`cyclen_target`/`cyclen_tol` config.py:1113-1114,
> `discharge_target`/`discharge_tol` config.py:1115-1116, `f_r_limit` config.py:1119
> *"also the minimization objective"*; `acquisition.score_user_criteria:3171`) 하나이며,
> 그것은 **`[acquisition] objective` 값이 아니다**(`_VALID_OBJECTIVES` 에 없다, config.py:1699).
> → **이 슬라이스는 user_criteria 캠페인을 돌리지 않는다.** MODE-T 는 arm A/B 의 클린 행에
> `cyclen ∈ [620, 645]` ∧ `B_d ≥ b` 를 **사후 적용**한 뒤의 최소 측정 `F_r` 로 보고하며,
> `b` 는 **결과 문서가 아니라 여기서** 고정한다: **`b = 45.0 MWd/kgHM`** (프록시 정의는 §7.2.3).
> 네이티브 MODE-T 캠페인은 **라운드 2 이월**이다.

#### 7.2.3 `B_d` (방출연소도) — 등록된 프록시

`discharge_burnup` 컬럼은 **Tier-1 2,028행 전부 NaN** 이고 MASTER FOM 에 네이티브 필드가 없다
(`design/bootstrap.py:104-110` *"the vendor FOM carries no native discharge_burnup"*).
→ **등록된 프록시** (백필 계획 `bd_backfill_plan_20260903.md`):

```
B_d  =  P_th · cyclen / M_HM(core)  ×  (N_FA / feed) / 1000      [MWd/kgHM]
        P_th = 3983 MW ,  N_FA = 241 ,  M_HM = Σ_241 u_mass_g
        (= design/bootstrap.py:104 estimate_discharge_burnup 와 동일 식;
         config.py:1164-1166 power_mw = 3983.0 / hm_mtu = 104.8)
```

**가정(등록):** 평형 배치 · 배치평균 · 전 집합체 동일 잔류시간 `241/feed`.
**따라서 `B_d` 는 LP 위치에 무감(cell·feed 안에서 cyclen 의 아핀 함수)** 이며 —
그것이 §7.5 가 `B_d` 를 **마크가 아니라 축**으로만 쓰는 이유다.
`max_assembly_burnup`(실측 파서 존재: `vendor/masterrl/burnup.py:70`)과
**측정** `max_pin_burnup ≤ 80` 은 **대리축으로 라벨해서 병기**한다.

#### 7.2.4 챔피언

**`model_dir = s1i`**, `cond_schema v8`.
(s1j f_xy 헤드는 r2 RANK 게이트 3/3 FAIL로 레벨 추정기 강등; within-cell ranking 조항이
G1–G4에 들어가기 전 새 헤드의 wave 랭킹 금지.) **신경망 헤드 예측과 사슬 예측은 둘 다
`wave_prereg.json` 에 그림자로 기록하고 어떤 게이트도 잡지 않는다.**

### 7.3 ★ MTC 게이트 — **독립 CLI 단계로 집행** (C1/C2/C3의 귀결)

**덱 설정 (양 arm 동일):**

```toml
[constraints]
mtc_enable        = true
mtc_min_pcm_per_c = -54.0      # APR1400 DCD Table 4.3
mtc_max_pcm_per_c =   9.0
post_verify_top_k = 5          # ← 라이브 게이트가 읽는 유일한 top_k (config.py:1240 기본 3)
sdm_enable        = false      # §7.5

[verify]
harvest_maps = true            # keep_success 를 강제한다. **keep_success 키는 존재하지 않는다**
package_root = "data/design/package"
```

**집행 (캠페인 종료 후, arm마다 1회):**

```
lpopt sdm-mtc --run runs/<arm_run_dir> --input <arm_deck>.inp --top-k 5
```

- 이 경로(cli.py:680)는 `candidates_from_delivery(run_dir, None, top_k)` 를 먼저 시도하고
  **비면 `select_topk_feasible(run_dir, top_k)` 로 폴백**한다(cli.py:741-746) → `min_fr_max_cycle`
  에서도 **실제로 돈다** (`build_delivery_payload` 는 `objective != "flat_power"` 이면 `None` 이므로
  objective 전환(§7.2.1)이 이 폴백 논리를 **바꾸지 않는다** — campaign.py:533-534).
  캠페인 내부 `_maybe_post_verify` 경로는 `min_fr_max_cycle` 에서도 **도달 불가**임을
  이 문서에 명시하고, 덱에 죽은 노브를 남기지 않는다.
- `[constraints]` 의 두 한계가 설정되어 있으므로 `limits.mtc_gated = True` → **PASS/FAIL 판정**이
  나온다(설정하지 않으면 REPORT-ONLY, sdm_mtc.py:346-352).
- 산출: `runs/<arm>/sdm_mtc_report.md` (+ `.csv`) 와 `data/sdm_mtc/results.jsonl` 사이드카.

> **★ 등록된 선택 순서 함정.** `select_topk_feasible` 은 후보를 **`|cyclen − 625|` 근접순**으로
> 정렬한다(sdm_mtc.py:1259-1261) — **`F_r` 순도 `F_xy` 순도 아니다.** 그러므로 **PRIMARY 후보(최소
> 측정 `F_r`)가 top-5 안에 들었는지 반드시 확인**하고, 들지 않았으면 그 후보가 포함될 때까지
> `--top-k` 를 올려 재실행한다(추가 1 MASTER 콜/후보). **이 확인을 사전등록한다.**

**예산.** MTC 축 1개 × 후보 5 = **arm당 ~5 MASTER 콜**(config.py:1227 *"~1 extra MASTER call each"*),
2 arm = **~10 콜** + 상기 재실행 꼬리. `branch_timeout_s = 300`, `mtc_delta_c = 5.0`.
**탐색 예산과 분리해 보고**한다.

### 7.4 SDM — **INCONCLUSIVE (보고 전용)**

`campaign._rod_model()`(campaign.py:3012-3023)이 **설계상 `None`** 을 반환한다:
*"lpopt has no full-core asset package, so this returns None until one exists — and the gate then
reports SDM as INCONCLUSIVE rather than inventing a rod map."* `lpopt sdm-mtc` CLI 경로도
`sdm_params` 를 **전혀 넘기지 않으므로** 동일하게 INCONCLUSIVE다.

> **★ 트랩 사전등록 (재비평 A14 수용).** 언젠가 프로그램적 훅 `post_verify_topk(rod_model=…)` 를
> 쓰게 되면, 그 함수의 기본 `scram_banks = ("R1"…"R5")`(sdm_mtc.py:1620-1624)는 **config가 스스로
> 금지한 설정**이다 — `config.py:1295-1303` 이 *"an R-only default cannot reach the 10,870 pcm
> requirement for ANY pattern — every candidate would FAIL for a reason that is a config defect,
> not physics"* 라고 적고 `[sdm_mtc]` 기본값에 **B와 A를 포함**시켜 두었다
> (`["R1","R2","R3","R4","R5","B","A"]`). → **`scram_banks`/`stuck_candidate_banks` 는 반드시
> `[sdm_mtc]` 에서 가져온다. 훅 기본값을 쓰지 않는다.**

### 7.5 마크 (동결)

**이겨야 할 두 수 (구분해서 등록):**

| | 값 | 정체 | 상태 |
|---|---|---|---|
| **★ PRIMARY 기준선 — S1 `F_r` 바닥** | **1.4605** | `T6_T4/f121`, `batchswap_enum_T6T4`. CBC 1302.7 · cyclen 618.0 · `node_peak` 1.3171 · 함의 FF 1.1089 | 67,880 수렴 노심의 **전역 최소** (`fr135_feasibility_scoping_20260903.md` §1.2·§5) |
| 프로그램 joint-clean 최소 `F_xy` (**보고 전용**) | 1.5295 | `a785eded…`, `fpcamp_minfr_199`, `E1_E2/f121`, **ga80**. `F_r` 1.4694 · CBC 1330.81 · `F_q` 1.8422 · \|AO\| 0.0404 · cyclen 638.639 · assy BU 56.572 | **측정 pin BU pending** — r2 phase-2 웨이브 **2026-09-03 02:07 완주 (30/30)**, 수확 진행 중 |
| **deliverable** 최소 `F_xy` (**보고 전용**) | 1.5322 | `bf3a70b2…`, **`T6_T4/f121`, paramA**. `F_r` **1.4857** · CBC 1337 · `F_q` 1.853 · \|AO\| 0.024 · **측정 pin 63.760** · cyclen 622.101 | **전 축 측정·한계 내** (arm B의 `F_r` 기준선 = **1.4857**) |

> **★ 마크 축 전환 (2026-09-03).** 사용자 지정 objective 가 `min F_r` 이므로 **모든 마크는 `F_r`
> 위에 선다**. `F_xy` 는 **제약(`≤ 1.65`)이자 보고 전용 축**이며 **마크가 아니다.**
> 이전 DRAFT 의 `F_xy` 마크(1.5245 / 1.5195 / 1.5322 / 1.5295)는 **철회**한다 (변경기록 CL-1).

| 마크 | 요건 (**전부 측정값**) |
|---|---|
| **PRIMARY** | 온디맨드 타입(`T7` 또는 `T8`) ≥1개를 fresh로 쓰는 MASTER 검증 노심이<br>**측정 `F_r ≤ 1.4505`** (= S1 바닥 1.4605 − 0.010) **∧ Tier-1 전 축** — `F_xy ≤ 1.65` **∧** `CBC ≤ 1600` **∧** `F_q ≤ 2.41` **∧** `\|AO\| ≤ 0.30` **∧** **측정** pin BU `≤ 80` **∧** **MTC ∈ [−54, +9] pcm/°C (`lpopt sdm-mtc` PASS, standalone gate §7.3)** |
| **STRETCH** | 측정 `F_r ≤ **1.4300**` (동일 Tier-1 전 축 충족). S2 점추정 1.4242–1.4324 의 보수단 (`fr135_feasibility_scoping_20260903.md` §5.2) |
| **SECONDARY** (전부 라벨 표기, headline 금지) | ① **`node_peak`** — arm A/arm B 양쪽의 분포(min/p05/중앙/p95/max)와 PRIMARY 후보의 값. **집합체 수준 첨두이며 `F_r` 의 대리가 아니다**(§7.2.0)<br>② **within-cell 지수** `d ln F_r / d ln node_peak` 를 **셀 내부에서 재측정** (arm A·B 각각, R4)<br>③ 측정 `F_r < 1.4857` (arm B 의 deliverable 기준선을 이김)<br>④ 측정 `F_r < 1.4605` 이나 Tier-1 한 축 이상 미달 (마진 없이 기록만) |
| **축 (마크 아님)** | `cyclen` 과 **`B_d`**(§7.2.3 프록시) — MODE-P 의 3-축 비지배 집합, MODE-T 의 사후 밴드 readout. **`B_d` 는 cell·feed 안에서 cyclen 의 아핀 함수이므로 독립 마크가 될 수 없다** |
| **NULL** | 예산 안에 **`F_r < 1.4605` 인 클린 행 0건** → 연료 설계축도 LP 축과 같이 이 프로그램의 `F_r` 바닥(≈1.46)을 옮기지 못한다 = **바닥은 플랜트 수준 성질**. 귀결: 집합체 축 폐쇄 (승계 레버는 §9-A7대로 결과를 보고 결정) |
| **보고 전용** | `F_xy`(전 분포 + R_Fxy) · SDM (INCONCLUSIVE) · 신경망 헤드 예측(그림자) · 사슬 예측(그림자) · arm B의 R_Fr · `max_assembly_burnup` |

**PRIMARY 바의 근거 (등록).** `F_xy` 관례(−0.005)를 `r = F_xy/F_r = 1.0649`(`T6_T4/f121` 실측 중앙값,
n=1,086, sd 0.025)로 환산하면 −0.0047 이고, 그 **2배**를 취해 재시작·재빌드 잡음 여유를 둔다
→ **−0.010**. **측정 기반이며 `sigma_chain,paired`(0.077) 와 무관**하다.

**기저율 (등록).** ① r2 는 `F_xy` 축에서 **100 콜 LP 탐색이 `< 1.5295` 를 0/99 로 못 찾았다**
(best 1.5437, 콜 12; 이후 88콜 무이득). ② `F_r` 축의 기저율은 **67,880 수렴 노심의 최소가 1.4605,
1퍼센타일이 1.5052** 라는 것이다. 온디맨드 타입이 100 콜 안에서 1.4505 를 찾으면 그 자체로 강한 증거다.

### 7.6 phase-2 pin 웨이브

캠페인 종료 후 arm A의 top-k(**측정 `F_r` 오름차순**)에 대해 **측정 pin BU** 웨이브
(선례 r2 phase-2 = 30 체인).
PRIMARY의 pin 축은 **예측 78(`minfr_pin_bu_limit`)이 아니라 측정 ≤ 80** 으로 판정한다.
`a785eded` 의 측정 pin이 이 문서 작성 시점 **pending** 이므로, 1.5295 기준선의 pin 열은
슬라이스 결과 문서에서 **확정된 값으로 갱신**한다.

---

## 8. ★ 슬라이스의 기대값 — **부호가 정해지지 않는다** (산술 전면 정정)

### 8.1 v2 §8.2b의 두 오류

1. **`d_fresh = 0.0067` 은 오독이다.** OPSCREEN.md:317 헤더는
   `… | FF_hot | FF_cold | contrast | **hump** | Fr_flr | dFF | dFr` 이고 Z 행의 `0.0067` 은
   **hump 열**이다(같은 절의 산문도 *"the hump is well inside calibration"*). 진짜 `d_fresh` 는
   `s09b_contrast.py:27` 에서 `rm(0.5) − rho_op(rm, f, bc, 0.0)` — **혼합체 감손량**이며
   OPSCREEN 어디에도 후보별로 표기되어 있지 않다. incumbent의 `d_fresh = 0.0023` 도 **무인용**이다.
2. **원천이 이미 발행한 값을 손으로 다시 유도했다.** `opmodel/s16_out.txt` 는 후보마다
   **`npk` 와 `Fr_fix`** 를 `Fr_flr` 옆에 인쇄한다. 손계산은 그 값을 재현하지 못했다.

### 8.2 원천의 자기 숫자로 다시 세운 구간 (동결)

Z와 가장 가까운 **발행된** 행은 `s16_out.txt` feed 121 목록 **[B] #3** 이다 —
53-역할이 **Z2와 완전히 동일**(`u5.00/4.2500 gd10x20 1:1;4:1;6:4 PB/z1`), 68-역할은
같은 `u5.50 / gd8 × 20` 이되 레이아웃만 다름(`2:2;4:1;6:3`):

```
[B] #3 :  cyc 623.5 (raw 622.2)   CBC 1406   FFhot 1.1208
          contrast +0.0308   hump 0.0056   npk 1.293   Fr_fix 1.495   Fr_flr 1.395
```

`Fr_flr ≡ 1.03 × 1.2085 × FF_hot` (OPSCREEN §7 헤더 정의)이므로 **양쪽 다리에 같은 추정기를
적용**하면:

**★ 기준 다리는 arm B 의 hot role = T4 (R3 정정, §8.5).** `E1_E2`(hot role E2)는 재실행하지 않는
고정 외부 참조이므로 **참고 열**로만 둔다.

| 다리 | 회귀 지선 (`Fr_fix`) | floor 지선 (`Fr_flr`) |
|---|---:|---:|
| **Z (≈[B]#3)** | **1.495** | **1.395** |
| **T6_T4 / T4 (arm B, 등록 기준)** | `1.03 × 1.2085 × 1.1430` = **1.4227** | 동일 **1.4227** |
| **ΔF_r (등록)** | **+0.072 (Z가 나쁨)** | **−0.026 (Z가 좋음)** |
| **ΔF_xy (등록)** (× r = 1.0649) | **+0.077** | **−0.028** |
| *(참고) E1_E2 / E2* | *`1.03 × 1.2085 × 1.1520` = 1.434* | *동일 1.434* |
| *(참고) ΔF_r* | *+0.061* | *−0.039* |

> ### ★ **등록된 기대값: ΔF_xy ∈ [−0.028, +0.077] — 0을 넉넉히 포함한다.**
> (**철회:** DRAFT 의 [−0.041, +0.065] — E2 기준 산술. R3 정정, §8.5·변경기록 CL-3.)
> **PRIMARY 축인 `F_r` 로는 `ΔF_r ∈ [−0.026, +0.072]`.**
> v2의 [−0.041, +0.037] 보다 **넓다**. 차이는 오직 **`node_peak` 을 LP가 회복하느냐**이고,
> OPSCREEN.md:253-255 는 그 회복이 **가정이지 예측이 아니라고** 명시한다
> (*"not as a prediction of what the current patterns will give"*).
> → **슬라이스는 "예측된 −0.018을 실현하는 작업"이 아니라 "부호가 미정인 효과를 측정하는 실험"이다.**
> **짝지은 대조군(arm B)이 선택이 아니라 필수인 이유가 이 구간이다.**

### 8.3 ★ 재구성 잔차를 정직하게 보고한다 (재비평 #4 수용)

같은 사슬을 **기준 다리**에 적용하면 `1.035 × 1.2085 × 1.1520 = 1.441` 인데
그 arm(A0)의 **측정** `F_r` 은 **1.5207**(OPSCREEN.md:222) — **−0.080의 재구성 편향**이다.
(A0 자신의 함의 `A = 1.5207/(1.2085×1.1520) = 1.092` 는 발표된 `A = 1.035 ± 0.031` 창 **밖**이다.)
→ **절대 재구성은 신뢰할 수 없고, 위 §8.2의 짝지은 차분만 인용한다.** 이 잔차를 여기 등록해
결과 문서가 절대값을 근거로 삼지 못하게 한다.

### 8.4 contrast 밴드 — **Z는 측정된 밴드에 앉지 않는다** (재비평 A8 수용)

OPSCREEN.md:238-246 의 측정 밴드는 `≥ 0.043 → node_peak 1.209–1.260`,
`0.026–0.028 → 1.274–1.327`, `≈ 0 → 1.387–1.551` 이다. **Z의 0.0300 은 두 밴드 사이의
미측정 간극**이다.

> **등록 문장:** *"Z의 contrast는 측정된 저-contrast 밴드보다 높고 측정된 고-contrast 밴드보다
> 낮다 — Z의 contrast에서 측정된 arm은 없다."* (v2의 "Z는 그 낮은 밴드에 앉는다"는 오독.)
> 이것이 같은 주의(caution)의 더 강한 형태다.

### 8.5 FF 단위 일치 (재비평 A7 수용)

OPSCREEN §7 헤더의 규약: 실현 타입은 **DeCART `%DIST` FF**, 신규 설계는 **서로게이트 앙상블**
(T3–T6 대조에서 0.0014 낮음). 따라서 Z와 E2를 뺄 때는 **양쪽을 DeCART 등가로** 맞춘다 —
Z의 DeCART 등가 FF_hot = **1.1222** (OPSCREEN.md:352-353).

> ### ★ **R3 정정 (2026-09-03) — 기준 hot role 은 E2 가 아니라 T4 다.**
> DRAFT 는 floor 지선을 `1.03 × 1.2085 × (1.1222 − 1.1520) = −0.037` 로 적었다. 그 `1.1520` 은
> **E2** 의 FF_hot 이고, 따라서 이 값은 **`E1_E2/f121` 이 대조군일 때의 산술**이다.
> 그러나 **§0-C4 가 arm B 를 `E1_E2/f121` → `T6_T4/f121` (paramA) 로 바꾸었고**, 그 셀의
> hot role 은 **T4 (FF_hot = 1.1430)** 이다. 짝지은 차분은 **실제 대조군의 hot role** 로
> 계산해야 한다. **올바른 등록값:**
>
> ```
> ΔF_r  = 1.03 × 1.2085 × (1.1222 − 1.1430) = −0.0259
> ΔF_xy = r · ΔF_r ,  r = 1.0649 (T6_T4/f121 실측 중앙값, n=1,086)
>       = −0.0276
> ```
>
> **철회:** `−0.037` / `−0.039` (E2 기준). **`E1_E2/f121`(ga80)는 재실행하지 않는 고정 외부
> 참조**이므로(§0-C4, §7.1) 그 셀의 hot role 로 계산한 차분은 이 슬라이스의 기대값이 아니다.
> E2 기준 값은 **참고로만** 병기한다.

**FF 원천 라벨 (필수).** Z = 서로게이트 → DeCART 등가 1.1222; T4 = 실현 타입의 **DeCART `%DIST` FF**
1.1430; E2 = 동상 1.1520. 결과 문서는 **각 항에 FF 원천을 라벨**한다.

### 8.6 S1이 다시 계산해야 하는 것 (사전등록)

`[B]#3` 은 유사체이지 Z가 아니다. S1은 **정확한 Z1′/Z2 튜플**로 다음을 재산출한다:

1. (A) 예측 `kconv`/`pinmap` → `FF_hot`, 상대역 FF, `peak_max` (pattern **`PB`** 어서션)
2. `s09b_contrast.py` **자신의 정의**로 `contrast` 와 `d_fresh` (hump가 아니다)
3. `node_peak = 1.4210 − 4.1725·contrast − 3.4862·d_fresh` → `Fr_fix = A · npk · FF_hot`
4. opmodel 운전점 `cyclen` / `CBC`
5. §8.2 구간의 **정식 갱신본** — 이 값이 결과 문서의 기대값이 된다

**S1 게이트:** `cyclen ∈ [620, 645]` ∧ `CBC ≤ 1500` ∧ `contrast ≥ 0.026` ∧ **(A) 경계(bounds) 통과**
(격자 위반은 허용 — T3/T4 가 `du = 0.75` 로 격자 밖에서 스크린되어 DeCART 대조 <100 pcm 통과한 선례).

---

## 9. 예산 · 일정

| 항목 | 값 | 근거 |
|---|---|---|
| S0c (A) 접근 확보 | ~0.5 일 | 신규, 차단성 |
| S0b fleet 정책 해소 | 오너 결정 1건 | `autoeng.toml:59-66` |
| task #0 σ 소급검증 (**병렬**) | ~0.5 일, DeCART 0 / MASTER 0 | 238 읽기 전용 |
| S1 스크리닝 | GPU ~수 초 / CPU ~수 분 (2 튜플) | SURROGATE_USAGE §2·§6 |
| S2 덱 저작 (238) | ~1 h (구현 포함 시 작업 #1에 계상) | — |
| S2b 스테이징 3-호스트 | ~1 h (패키지 ~320 MB + 0_APR1400) | 파일 전송 |
| **S3 DeCART 2 케이스 (181, 직렬 2-wide)** | **0.86 h** | §3.4 |
| S4 HGC 게이트 | ~수 분 | — |
| S5 TotalBatcher 재빌드 (39) | ≤ 1 min (**N=16에서만 관측**; N=39 미측정) | 181 16-FA 빌드 타임스탬프 |
| S5b 패키지 재생성 | ~수 분 | — |
| S6/S6b 등록·동기화·재-ingest | ~1 h | — |
| **S7 MASTER 부트스트랩 ×9** | **2–5 h + 재시도 꼬리** (단일 실패 관측 8,744 s = 2.43 h) | OPSCREEN.md:388-393, `t3t4_rerun.log` |
| S8 전이 스윕 `--k 32` | 1–2 h | T6T4 덱 헤더 |
| **S9 캠페인 2 arm × 100 콜** | **200 MASTER 콜** | 필수 대조군 |
| **S9b MTC (`lpopt sdm-mtc`)** | **~10 MASTER 콜** (arm당 5) + 재실행 꼬리 | §7.3 |
| S9c phase-2 pin 웨이브 | 별도 웨이브 (선례 30 체인) | r2 본 웨이브는 측정 pin 0/99 |
| **총계** | **약 2–2.5 근무일**, MASTER 콜 **~210 + 부트스트랩 9** | v2의 1.5–2일에 S0c/S2b/저작 원격화가 추가됨 |
| 디스크 | 케이스당 ~35.6 MB; `MAS_XSL` +771,698 B, `MAS_HFF` +809,714 B (2 타입) | §4.3 |

---

## 10. 처분 (disposition) · 중단점

### 10.1 중단점 (사전등록)

| 지점 | 조건 | 처분 |
|---|---|---|
| **S0c** | (A) 체크포인트에 `USER` 로 접근 불가 | **S1 중단.** 접근 확보 전까지 진행 없음 |
| **S0b** | fleet 정책 예외 미기록 | **S3 중단** |
| **S2b** | SHA-256 매니페스트 불일치 | **중단.** 재전송 |
| **S4 / G-H4** | \|Δk\| > 100 pcm 또는 \|ΔFF\| > 0.0021 | **여기서 멈춘다.** 스크린이 실계산을 예측하지 못하면 §1·§8 전체 전제가 무너지므로 라이브러리를 건드릴 이유가 없다. 실패 설계는 **능동학습 포인트로 기록** ((A) 재학습은 이 체크아웃에서 불가, 부록 B-14) |
| **S5 / G-H3** | 크기 등식 불일치 | **중단 + `.bak` 롤백** (한 세대뿐 — 재빌드 재시도 전 반드시 E: 스냅샷 확인) |
| **S5b** | 기존 paramA 쌍 10개 중 하나라도 `validate_reload_deck` FAIL | **캠페인 전 중단** |
| **S6** | 배정된 alias가 `T7`/`T8` 이 아니다 | **중단** (풀 순서 가정 붕괴) |
| **S9b** | PRIMARY 후보가 top-5 밖 | **중단 아님** — `--top-k` 를 올려 재실행 (§7.3) |
| task #0 | σ > 0.020 | **슬라이스 중단 아님.** 결과 문서의 판정 문장만 물리실험 서사로 교체 (§2.1) |

### 10.2 결과 문서가 반드시 담아야 하는 것 (사전등록)

1. arm A / arm B 의 **`F_r` 분포**(PRIMARY 축)와 **R_Fr (재빌드·재시작 효과)의 측정값** — 이득
   계산 전에 먼저 뺀다. `F_xy` 분포와 R_Fxy 는 **보고 전용**으로 병기
1b. **SECONDARY:** arm A/B 의 `node_peak` 분포와 **셀 내부 재측정 지수** `d ln F_r / d ln node_peak`
   (R4). `node_peak` 이 **집합체 수준 첨두**이며 `F_r` 의 대리가 아님을 문장으로 명시
1c. **축:** MODE-P 의 `(F_r, cyclen, B_d)` 3-축 비지배 집합과 MODE-T 의 사후 밴드
   (`cyclen ∈ [620,645]` ∧ `B_d ≥ 45.0`) readout. `B_d` 는 **§7.2.3 프록시**임을 각 표에 라벨
2. §8.2 구간(**T4 기준, R3 정정본**)의 **S1 갱신본**과 측정값의 위치
   (구간 안 / 밖 / 어느 지선에 가까운가)
3. `lpopt sdm-mtc` 의 MTC 값·판정·MASTER 콜 수, SDM = INCONCLUSIVE 명시
4. 측정 pin BU (arm A top-k) 와 `a785eded` 의 확정 pin
5. task #0 의 σ 와, 그것이 선택한 판정 문장
6. G-H1…G-H6 전 게이트의 PASS/FAIL 과 실측 바이트 수
7. **NULL이면** — 축 폐쇄 권고와 §9-A7의 승계 레버 평가 (측정된 `ΔF_xy` 부호·크기에 근거)

---

## 11. 가정 (오너가 뒤집을 지점) — v2 §10에서 **3건 갱신**

| # | 가정 | v2 대비 | 뒤집으면 |
|---|---|---|---|
| **A1** | LEU+ 농축 상한 = **5.50 w/o** | **갱신: 이제 구속한다** (§1.3, N1 철회) | 5.00으로 낮추면 Z1′ 탈락 |
| **A2** | 비율은 준법 창 **0.85 ± 0.03** 안에서 자유; Z1′ = 0.8545 | 유지 | "정확히 0.85"로 조이면 `u_high ≤ 5.333` → **Z1\* = u5.30 / e2 4.505** 로 재지정 + 재스크린. 0.92 허용은 준법 R1 위반 → **인허가 판단 필요** |
| **A3** | **`xenon TR`** | 유지 | EQ로 가면 라이브러리 전체(117 격자) 재생산 |
| **A4** | DeCART = **181 직렬**, 재빌드·MASTER = **199**, 저작·스크린 = **238**. **box 104(=로컬 PC) 사용 금지** | **전면 갱신 (C5)** | box 104를 쓰려면 **사용자 상시 규칙(로컬 연산 금지)의 명시적 해제**가 필요하다 |
| **A5** | 이 세션이 오케스트레이션; 이 문서 승인 전 발사 없음 | 유지 | 다른 세션에 발사권을 주면 runs/orchestration 영수증 규약부터 확정 |
| **A6** | 슬라이스 우선 — 손으로 고른 **1 쌍(2 격자)** | 유지 | 바로 4-후보로 가면 R23(레이아웃 파일명 충돌) + 9회 부트스트랩을 동시에 맞는다 |
| **A7** | NULL 시 승계 레버는 슬라이스 결과를 본 뒤 결정 | 유지 | 지금 고정하면 측정 전에 레버를 버린다 |
| **A8** | 스크리닝 = 238, (A)는 `USER2` 홈 | **갱신: S0c를 경성 선행으로 승격** | 접근 불가면 슬라이스가 S1에서 막힌다 |
| **A9** | SDM = INCONCLUSIVE, **MTC만 게이트, 독립 CLI로** | **갱신 (C1)** | SDM을 게이트로 요구하면 작업 #20(full-core rod model 이식)이 슬라이스 선행이 되어 일정이 크게 는다 |
| **A10 ★신규** | `autoeng.toml` 의 "181 NEVER used" 정책에 이 라운드 한정 예외를 기록한다 | **신규** | 예외를 안 주면 DeCART를 199에서 돌려야 하고, 199의 MASTER 생산과 경합한다 |

---

## 부록 R — 재비평 24항목 처분표

### R.1 physics-and-licensing-realism

| # | 항목 | 처분 | 위치 |
|---|---|---|---|
| 1 | MTC 게이트가 `min_fxy` 에서 안 돈다 [CRITICAL] | **수용** — 독립 CLI `lpopt sdm-mtc` 로 이전 | §0-C1, §7.3 |
| 2 | `top_k` 노브 오류, DCD 상수 인용 위치 | **수용** — `[constraints] post_verify_top_k = 5`, 예산 arm당 5 | §0-C2, §7.3 |
| 3 | `d_fresh = 0.0067` 은 hump 열 오독 | **수용** — 삭제, S1 재계산 | §8.1, §8.6 |
| 4 | 원천이 이미 `npk`/`Fr_fix` 를 발행 | **수용** — [B]#3 인용으로 교체, 구간 재산출 | §8.2 |
| 5 | `+0.0614` 는 헤드 원천, 사슬 원천 아님 | **수용** — "두 독립 추정 일치" 문장 삭제 | §2.2-α |
| 6 | G-H3 출처 셀 오기 | **수용** — 표 정정 | §4.3 |
| 7 | "omp 4-way" 는 추론 | **수용** — 표기 정정 (어차피 사용 금지) | §3.4 |
| 8 | DeCART 두 행의 규약 불일치 | **수용** — 둘 다 2-wide/직렬 병기 | §3.4 |
| 9 | "큐 상한 4 (32스레드)" | **수용** — "직렬 2 프로세스 = 32 스레드 중 2" | §3.3 |
| 10 | SDM 반쪽 수용은 옳으나 MTC가 안 돈다 | **수용** — #1과 동일 처분 | §7.3, §7.4 |
| 11 | (A) 접근이 슬라이스 임계경로에 있다 | **수용** — **S0c 신설, 경성 선행** | §3.1 |
| 12 | 3,738은 A2와 모순 | **수용** — 열거 창 재정의, **N1 철회**, 개수는 238에서 산출 | §1.3 |
| 13 | task #0에 결정 귀결이 없다 | **수용 (선택지 b)** — 슬라이스는 **무조건**, task #0은 병렬·라운드2 전용 바 | §2.1 |
| 14 | `type_id` 0.1 w/o 양자화 충돌 | **수용** — 4.6750/4.70 이 같은 `47` 로 접힘을 명시 + 레지스트리 raise 가드 | §1.1 |

### R.2 engineering-feasibility-and-data-flow

| # | 항목 | 처분 | 위치 |
|---|---|---|---|
| A1 | 캠페인 모드에서 MTC 불가 [CRITICAL] | **수용** — 독립 CLI + `select_topk_feasible` 확인 작업 신설 | §0-C1, §7.3, 작업 #21 |
| A2 | `top_k` 노브 / 예산 | **수용** | §0-C2 |
| A3 | 대조군 arm이 다른 라이브러리 | **수용** — arm B = **`T6_T4/f121` (paramA)**, DiD → 횡단면 대비 | §0-C4, §7.1 |
| A4 | box 104 미식별 | **수용 — 그리고 해소했다: box 104 = 로컬 PC (`mk-ctrp`)** → **권고 철회** | §0-C5, §3.0 |
| A5 | 181에 입력 자산 없음 | **수용** — **S2b 신설, 경성 선행** | §3.2 |
| A6 | §4.3 열거 창 모순 | **수용** — §1.3에서 A2 규약 채택, N1 철회 | §1.3 |
| A7 | FF 단위 혼합 | **수용** — DeCART 등가 1.1222 사용, 항마다 원천 라벨 | §8.5 |
| A8 | "0.026–0.028 밴드" 오독 | **수용** — "미측정 간극" 으로 재서술 | §8.4 |
| A9 | `keep_success` 는 키가 아니다 | **수용** — 덱에서 삭제 | §0-C3, §7.3 |
| A10 | `pin_map` 은 선택 인자 | **수용** — "게이트가 요구한다"가 아니라 "작업 #1이 **선택해서 넘긴다**" | 작업 #16 |
| A11 | `harvest` 가 덱을 지운다 | **수용** — `stage_hgc` 는 `templates_lat1600` 에서 복사 | §3.5 |
| A12 | G-H3 라벨 뒤바뀜 | **수용** | §4.3 |
| A13 | `n_gd` 에 더 강한 물리 근거 | **수용** — `SURROGATE_USAGE.md:143` 인용, 저-Gd 확장 **폐쇄** | §1.4 |
| A14 | `post_verify_topk` 의 R-only 트랩 | **수용** — 트랩 명시 + `[sdm_mtc]` 강제 | §7.4 |

**반려 0건.** 재비평 두 편의 모든 항목을 수용했다. (v2가 **v1** 비평에 대해 반려한 4건 —
physics #4 절반 / #10 심각도 / #19 등식 / eng #17 슬라이스 차단성 — 은 이 문서에서도 유지된다:
그 근거는 v2 부록 D.3에서 파일:라인으로 재확인되었고 재비평이 반박하지 않았다.)

---

## 부록 CL — 변경기록 (change log)

**CL rev.1 · 2026-09-03 · 근거** `data/reports/fr135_feasibility_scoping_20260903.md` §6 "권고"
(R1–R4) + 사용자 목표문 (축어, 2026-09-03). **코드 0줄 · DeCART 0 · MASTER 0 · 로컬 연산 0.**

| # | 절 | 변경 | 근거 |
|---|---|---|---|
| **CL-1** | **§7.2 (전면 개정)** | objective 를 `min_fxy` → **사용자 지정 `min F_r`** (`[acquisition] objective = "min_fr_max_cycle"`, `minfr_lambda = 1000.0`, `minfr_pin_bu_limit = 78.0`). **§7.2.0 `F_r` 정의 신설** (EDIT3 `max FRP`, 전 연소단계 최대, **핀** 반경첨두; `node_peak` 은 집합체 수준이며 SECONDARY). **§7.2.2 MODE-P / MODE-T 신설**, **§7.2.3 `B_d` 프록시 신설** | 스코핑 §5.3 오너 결정항목 1 · §6.1 · §6.3-R1 |
| **CL-2** | **§7.5 (마크표 교체)** | PRIMARY = 측정 **`F_r ≤ 1.4505`** (= 1.4605 − 0.010) + Tier-1 전 축 + 측정 pin ≤ 80 + MTC standalone PASS. STRETCH = **≤ 1.4300**. SECONDARY = `node_peak` 분포 + within-cell 지수 + 1.4857/1.4605 라벨. NULL = **`F_r < 1.4605` 클린 행 0건**. `F_xy` 는 **제약(≤ 1.65) · 보고 전용**. **철회:** F_xy 마크 1.5245 / 1.5195 / 1.5322 / 1.5295 | 스코핑 §5.2 (방어 가능한 다음 사전등록 목표) · §6.3-R1 |
| **CL-3** | **§8.2 · §8.5 (R3)** | floor 지선의 기준 hot role 을 **E2(1.1520) → T4(1.1430)** 로 정정. `ΔF_r = 1.03 × 1.2085 × (1.1222 − 1.1430)` = **−0.0259**, `ΔF_xy` = **−0.0276**. 등록 구간 **[−0.041, +0.065] → [−0.028, +0.077]** (`ΔF_r ∈ [−0.026, +0.072]`). E2 기준 값은 참고 열로 강등 | 스코핑 §6.3-R3 (`:22` vs `:565-567` 내부 불일치) |
| **CL-4** | **§7.5 · §10.2 (R4)** | `node_peak` 과 **셀 내부 재측정 지수** `d ln F_r / d ln node_peak` 를 **SECONDARY readout 으로 명시 등록** (arm A·B 양쪽, 측정비용 0) | 스코핑 §6.3-R4 · §1.3(b) |
| **CL-5** | **§1.1 (R2)** | 후보 **Z1′/Z2 튜플 동결 유지** — 442-예측 서로게이트 스캔이 독립 재확인 | 스코핑 §6.3-R2 · §3.1 · §3.2 |
| **CL-6** | §2.1 · §7.1 · §7.3 · §7.6 · §10.2 | 축 전환의 파급: 판정 문장의 축을 `ΔF_xy` → `ΔF_r` (§2.1): 추정량 `R`/`ΔF` 를 `F_r` 다리로 재정의(F_xy 병기), `sdm-mtc` 폴백 논리가 objective 전환에 **불변**임을 명시, top-k 선택 순서 함정을 `F_r` 기준으로 재서술, phase-2 정렬키 = 측정 `F_r` | 본문 각주 |
| — | **미변경** | §0(C1–C5) · §1.2–§1.4 · §2(§2.1 한 문장 제외) · §3 · §4 · §5 · §6 · §8.1 · §8.3 · §8.4 · §8.6 · §9 · §10.1 · §11 · 부록 R · 부록 S | **무관 절 재작성 금지** |

**철회된 수치 목록 (검색용):** `F_xy ≤ 1.5245` (PRIMARY) · `F_xy ≤ 1.5195` (STRETCH) ·
`F_xy < 1.5322` (SECONDARY) · `F_xy < 1.5295` (NULL) · `ΔF_r = −0.037` · `ΔF_xy = −0.039` ·
구간 `[−0.041, +0.065]`.

**변경하지 않은 스탬프:** 작성일 2026-09-03 · 상태 DRAFT · A5(오너 승인 전 발사 없음) ·
예산 §9 (200 + ~10 MASTER 콜, 부트스트랩 9) · 중단점 §10.1 · 가정 §11.
**objective 전환은 MASTER 콜 수를 바꾸지 않는다** — 같은 100-콜 arm 을 다른 축으로 채점할 뿐이다.

**선행 데이터 작업 1건 (차단성 아님):** `B_d` 축은 `discharge_burnup` 백필을 전제한다 —
사양은 `data/reports/bd_backfill_plan_20260903.md`. 백필 이전에는 §7.2.3 프록시와
`max_assembly_burnup` / **측정** `max_pin_burnup ≤ 80` 을 **대리축으로 라벨**해서 쓴다.

---

## 부록 S — 이 문서가 만들지 않은 것

코드 0줄. DeCART 0회. MASTER 0회. 파케이 읽기 0회. 발사 0회. 로컬 연산 0회.
**모든 수치는 소스 파일 또는 기존 리포트 인용이며 출처를 밝혔다.**
착수 시 첫 산출물은 **S0c (A) 접근 확보**와 **작업 #1 (핀맵 저작 승격)** 이다 —
구현 순서는 `assembly_on_demand_tasks_20260903.md`.

---

## 부록 L — **S3 DeCART 발사 스탬프 (2026-09-03)** · *추가 전용(append-only)*

> 이 절은 §3.2(S2b 스테이징)·§3.3(실행 파라미터 동결)의 **집행 기록**이다. 위 본문은 수정하지 않았다.

### L.1 S2 덱 저작 (HOST_238)

저작 경로: `lpopt.design.lattice.write_authored_deck(design, out, registry, apr1400_root, template_root)`
(= `author_template` → `author_gd_layout` → `compliance.enforce_design` → `edit_dec_text`), venv Python 3.11,
`~/lpopt_ws/scratch/slice_Z/`. 로컬 PC 연산 0건.

| | **T7 = P5547Z1G08N20 (Z1′)** | **T8 = P5042Z1G10N20 (Z2)** |
|---|---|---|
| 저작 덱 `dec_FA_<alias>.inp` sha256 | `62f37b0a7ffe054135ab0fd9fbf9c561b74d5b6899fdb887cb466fc46c49a67d` | `dc319d66f168e7137fe4db2a2942dc18445ebf1bb9b979c7da2704288575f1b8` |
| 덱 바이트 | 7,429 | 7,431 |
| 저작 템플릿 `dec_FA_<type_id>.inp` sha256 | `fc39d5f8fe329bedda82faf19f5c4aa2ff3b30056b18b9b0431137cba73f8d2b` | `66d150b8750376051bfb4705e488fdb7b9badcb3817cf60ed64a3370bd103842` |
| CASEID / assembly | `FA_T7` | `FA_T8` |
| Gd 옥탄트 (census) | `1:1;4:1;6:4` → 다중도 **20/20** | `1:1;4:1;6:4` → **20/20** |
| `mixture UO2` 92235 | 5.5 | 5.0 |
| `mixture UO2_2` 92235 | 4.7 | 4.25 |
| `mixture UO2G` 밀도 / `6408` | **9.95** / **8.0** | **9.88** / **10.0** |
| `xenon` | **TR** | **TR** |
| `nxfile` | `D:\DeCART_MASTER\LIB\DML-E71N047G018-PV01-cr08.BIN` (재작성 불필요 — 원본 동일) | 동상 |
| `%TITL` 카드 수 | 0 (DeCART 덱에는 `%TITL` 이 없다 — MASTER 카드) | 0 |
| `compliance.enforce_new_type` (via `enforce_design`, `enr_zone` 명시 + 16×16 pin_map) | **PASS**, ratio 0.854545 (\|Δ\| 0.00455 < 0.03) | **PASS**, ratio 0.850000 |
| alias 배정 | **T7** (사전등록 §1.1 가정 확인) | **T8** (동상) |

베이스 덱 sha256 (238 스테이징본, 로컬 `0_APR1400` 와 동일):
`dec_FA_B03.inp` = `d81671fdf83c103ea88a71adaf745815675536665db76bb7d5e3dbd0319f2d6a`,
`dec_FA_B05.inp` = `c88decfcc852f677c402f8212737c73e4023318046dcb351108c3e7c8e039ba3`.
레지스트리는 `data/design/package/registry.json` 의 **작업 사본**(37 → 39)에서만 배정했고 **원본에 기록하지 않았다**.

> ### ★ L.1a 등록된 편차 — **IGD_20 트리에는 gd_wt 8 로 밀도화된 베이스가 없다**
> §1.1은 Z1′ 의 베이스를 `IGD_20/8_20_z1/dec_FA_B03.inp` 로, UO2G 캐리어 밀도를 **9.95 g/cc**
> 로 동결한다. 그러나 실측하면 **`IGD_20` 의 6개 덱(6/8/10 × z1/z2) 전부가 `UO2G 9.88` /
> `6408 10.0`** 이다 — 디렉터리 이름이 gd_wt 를 말하지만 덱은 그렇지 않다. `IGD_16` 은 규약을
> 지킨다(`6_16 → 10.01/6.0`, `8_16 → 9.95/8.0`, `10_16 → 9.88/10.0`). `edit_dec_text` 는 밀도를
> 절대 수정하지 않으므로, B03 을 그대로 쓰면 T7 은 **9.88 g/cc (0.7 % 초과)** 로 실현된다 —
> `realize_lat1600.build_template_tree` 독스트링이 명시적으로 경고하는 실패 모드.
> **처분:** 사전등록의 밀도(9.95)를 정본으로 삼아 B03 의 UO2G 밀도 **한 줄만** 9.88 → 9.95 로
> 패치한 뒤 저작했다. 교차검증: 원생 밀도화된 `IGD_16/8_16_z1/dec_FA_A03.inp` 를 같은 저작
> 경로에 넣어 얻은 T7 덱과 **바이트 동일**(두 베이스는 UO2G 두 줄·Gd 맵·CASEID 외 완전 동일).
> **오너 확인 필요 (비차단, 기록용):** 기존 37종 중 `gd_wt 8 × n_gd 20` 타입(`P2 = P6253Z1G08N20`,
> `P3 = P5853Z2G08N20`)은 이 패치 없이 만들어졌으므로 **9.88 로 실현되어 있다.** 즉 T7 은
> 라이브러리 형제들과 캐리어 밀도가 다르다. 사전등록을 따랐지만 선례와는 어긋난다.

### L.2 S2b/S3 스테이징 · 발사 (HOST_181 = `DESKTOP_HOST_181`, `USER@HOST_181`)

읽기 전용 프리플라이트 (2026-09-03 13:17 KST):

| 항목 | 측정값 | 판정 |
|---|---|---|
| `D:\DeCART_MASTER\BIN\decart2d1.1m5.exe` sha256 | `5F0F10F1…F006CE` | **일치** (§3.3) |
| `D:\DeCART_MASTER\LIB\DML-E71N047G018-PV01-cr08.BIN` sha256 | `AEF86EEB…64E377` | **일치** (§3.3) |
| 실행 중 `decart2d1.1m5.exe` | 0 | 유휴 |
| C: 여유 / D: 여유 | 629.1 GB / 7.1 GB | 실행 루트 C: 사용 |
| 논리 프로세서 | 32 | — |
| Python | WindowsApps 실행 별칭 스텁만 (실 인터프리터·lpopt venv 없음) | → PowerShell 런처 |

스테이징: `stage_slice_Z_181.ps1 -SourceRoot C:\lpopt_decart\stage -RunRoot C:\lpopt_decart\slice_Z -Execute
-PolicyExceptionRef OWNER-TASK-20260903-sliceZ-S3` → 영수증 `C:\lpopt_decart\slice_Z\stage_receipt.json`,
`ok: true`, `problems: []`, 스테이징 sha256 = 저작 sha256 (**바이트 동일**), `nxfile` 전후 동일.
실행 루트는 오케스트레이터 지시에 따라 `C:\lpopt_decart\slice_Z\` 를 사용했다 (§3.3의
`C:\Users\USER\lpopt_work\assembly_slice_Z_20260903\` 대신; **드라이브·정책 조건은 동일**하게 C: 이고 여유 629 GB).

러너: `run_slice_Z_181.ps1` (신규, `5_RL/` 및 181 `C:\lpopt_decart\stage\`). `lattice.run_batch` 가
러너 정본이지만 181에 Python 이 없어 **검증된 성질만 PowerShell 로 이식**했다
(`2_LP/artifacts/run_decart_eq_xesm_queue_181.ps1` 규약): exe/XS sha 선검사, `OMP_NUM_THREADS=1`,
전역 동시성 < 2, 덱을 `dec_FA_<case>.inp` + `decart.inp` 로 스테이징, 케이스별 타임아웃 7,200 s,
`process_result.json` / `rc.txt`, `stdout.txt` / `stderr.txt`, 종료 후 `manifest.json`
(`JOB FINISHED` 마커 + 산출물 sha256).

**발사 스탬프**

| | 값 |
|---|---|
| 호스트 | `DESKTOP_HOST_181` (HOST_181) |
| 감독 프로세스 | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\lpopt_decart\stage\run_slice_Z_181.ps1" -RunRoot "C:\lpopt_decart\slice_Z"` — PID **12548** (Win32_Process.Create, 분리 실행) |
| **T7** | PID **13500**, cwd `C:\lpopt_decart\slice_Z\T7`, cmd `D:\DeCART_MASTER\BIN\decart2d1.1m5.exe` |
| **T8** | PID **30292**, cwd `C:\lpopt_decart\slice_Z\T8`, cmd `D:\DeCART_MASTER\BIN\decart2d1.1m5.exe` |
| 시작 시각 | **2026-09-03 13:21:20 KST** (양 케이스 동시, 2-wide) |
| 환경 | `OMP_NUM_THREADS=1`, 직렬 exe, 타임아웃 7,200 s/case |
| 예상 종료 | ≈ **14:12:44 KST** (§3.4의 3,084 s/case 199 실측 대입; 181은 미측정) |

> **등록된 관찰 — 181은 DeCART 기준으로만 유휴다.** 발사 시점에 타 사용자(`kgt`)의
> `C:\Users\kgt\icsbep_sdf\sdf_lane.ps1 -Lane 5/6` PowerShell 작업 2건이 돌고 있었다.
> §3.3의 게이트(전역 `decart2d1.1m5.exe` < 2)는 만족하지만 **32 스레드를 단독 점유하지 않는다** →
> 실측 wall 이 3,084 s 를 넘더라도 그 자체를 이상으로 읽지 말 것.

---

# 부록 M — **S4–S6 스탬프** (2026-09-03, append-only)

**결론 선언: S4 에서 멈췄다. S5·S6 는 실행하지 않았다.**
`G-H1c` 와 `G-H4` 가 T7·T8 양쪽에서 FAIL 이다. 사전등록 §10.1 / 런북 §1.3 의 중단 규약대로
**라이브러리를 건드리지 않았다** — 199 에 대한 쓰기 0, 스냅샷 0, 등록 0, 재빌드 0, 부트스트랩 0.

**중요 — 두 FAIL 중 어느 것도 T7/T8 을 판별하지 못한다.** 아래 대조군 재현이 그것을 보인다.
이것은 설계 기각 사유가 아니라 **게이트 하네스 자신에 대한 미해결 문제**이며, 오너 처분이 필요하다.

## M.1 S4 회수 — 완료·무결

181 완주 (읽기 전용 확인): `manifest.json` 두 행 모두 `exit_code 0` · `job_finished true` ·
`timed_out false` · `stderr_bytes 0`. `decart2d1.1m5` 잔존 프로세스 **0**.
wall: T7 **2,621 s** / T8 **2,641.2 s** (§3.4 의 3,084 s/case 예상보다 빠름).

> **등록된 파일명 사실.** 러너가 남긴 HGC 는 `FA_T7.HGC` 가 아니라
> **`FA_T7_0101.HGC` / `FA_T8_0101.HGC`** 다 (`_0101` 접미사). S5 의 `stage_hgc` 는
> 경로를 명시로 받으므로 차단 요인은 아니나, `DesignSource(hgc_path=…)` 에 이 이름을 쓰거나
> 스테이징 시 정규명으로 복사해야 한다. 런북 §1.2 의 `FA_<alias>.HGC` 표기는 이 점에서 부정확하다.

전송 2홉(181 → 로컬 → 238) sha256 **바이트 일치**, `manifest.json` 값과 4/4 일치:

| 파일 | 바이트 | sha256 (manifest = 로컬 = 238) |
|---|---|---|
| `FA_T7_0101.HGC` | **7,395,955** | `4B287A61B77F62595A4F846C6A06E51A900CD3A413D8B967D7B200016473D1E0` |
| `FA_T8_0101.HGC` | **7,395,955** | `5EFEECA8AA246C41563F57757BFE1DAB5B68BF431B343744409596A059F70F96` |
| `dec_FA_T7.inp` | 7,429 | `62F37B0A7FFE054135AB0FD9FBF9C561B74D5B6899FDB887CB466FC46C49A67D` |
| `dec_FA_T8.inp` | 7,431 | `DC319D66F168E7137FE4DB2A2942DC18445EBF1BB9B979C7DA2704288575F1B8` |

`.out` T7 27,288,042 B / T8 27,262,244 B · `.sum` 각 767,500 B · `stderr.txt` 0 B ·
`stdout.txt` 81,223 / 81,883 B · `rc.txt` 5 B · `xs_reference.txt` 346 B.

행선지: 로컬 `5_RL/data/design/slice_Z/products/{T7,T8,manifest.json}` ·
238 `~/lpopt_ws/scratch/slice_Z/products/{T7,T8,manifest.json}`.

DeCART 실행체·라이브러리 핀 (manifest):
`decart2d1.1m5.exe` `5F0F10F1…F006CE` / `DML-E71N047G018-PV01-cr08.BIN` `AEF86EEB…64E377`.

## M.2 alias 풀 — 가정 유지

199 정본 `registry.json` = **37 alias**, `T0`–`T6` 존재, `T7`/`T8` 미사용.
§1.1 의 중단점("alias 가 T7/T8 이 아니면 S6 에서 멈춘다")은 **아직 유효**하다.

## M.3 HGC 게이트 결과 (238, `lpopt.design.hgc_gates` API · GPU 1)

| 게이트 | 바(bar) | T7 측정 | T8 측정 | 판정 |
|---|---|---|---|---|
| **G-H1** 구조 | `%TITL` 334 = 62 + 16×17; `%DIST`/`%MACX`/`%MICX`/`%ADFT` 각 334; `%FINE` 1 | `%TITL` **334**, 4 태그 각 **334**, `%FINE` **1**, ref 62, branch labels 16 | 동일 | **PASS / PASS** |
| **G-H1b** 크기 | 정확히 7,395,955 B (n_gd=20) | **7,395,955** | **7,395,955** | **PASS / PASS** |
| **G-H1c** 유효성 | `_hgc_looks_valid` ∧ 피크 이후 k-inf 단조 (tol 1.0e-3) | **FAIL** — BU 9/10/11 에서 +0.00112/+0.00134/+0.00155; 피크를 BU=0 으로 판정 (k 1.15072 → 0.787159) | **FAIL** — BU 12/13/14 에서 +0.00117/+0.00138/+0.00160 (k 1.11435 → 0.768151) | **FAIL / FAIL** |
| **G-H2** Gd 인구조사 | 20 / 20 | **20/20** | **20/20** | **PASS / PASS** |
| **G-H4** 회귀 (k) | BU ≥ 0.2 전 구간 \|Δk\| ≤ **100 pcm** | **199.3 pcm** @ BU=20 (61 점 비교) | **252.0 pcm** @ BU=18 (61 점) | **FAIL / FAIL** |
| **G-H4** 회귀 (FF) | \|ΔFF\| ≤ **0.0021** | **0.00109** @ BU=41 | **0.00148** @ BU=25 | (PASS 성분) |

**verdict(T7) = FAIL · verdict(T8) = FAIL.**

서로게이트: `~/lattice_surrogate/kpin_pa` → 심링크 → `/home/USER/lattice_surrogate/6_DeCART_Surrogate`
(`find_surrogate_root` 확인, `load_error = None`). `SurrogateBridge(device="cuda:0")` 를
`CUDA_VISIBLE_DEVICES=1` 아래에서 — 즉 **물리 GPU 1 전용** (D-9 해소 확인).
스크린 시계열 = `predict_cases` 의 `kconv` (2,62) / `peak_pin_power` (2,62) 를 `bu_grid` (62,) 에 정렬.
산출 보존: `238:~/lpopt_ws/scratch/slice_Z/screen_curves.json`.
(sklearn 1.8.0 피클을 1.9.0 으로 언피클하는 `InconsistentVersionWarning` 3건 관측 — 기록만 함.)

## M.4 ★ 두 FAIL 은 T7/T8 을 판별하지 못한다 — 대조군 재현

### M.4a G-H1c 는 현행 라이브러리를 함께 기각한다

이미 라이브러리에 **들어가 있는** n_gd=20 생산 HGC 6개에 **같은 게이트를 그대로** 돌렸다
(199 `package/hgc/` 에서 회수, 238 에서 평가):

| alias | type_id | G-H1 | G-H1b | G-H1c | G-H2 | G-H1c 상세 |
|---|---|---|---|---|---|---|
| `P2` | P6253Z1G08N20 | PASS | PASS | **FAIL** | PASS | BU 13/14/15 +0.00112/+0.00134/+0.00161 |
| `P3` | P5853Z2G08N20 | PASS | PASS | **FAIL** | PASS | BU 11/12/13 +0.00122/+0.00147/+0.00171 |
| `P9` | P5849Z1G10N20 | PASS | PASS | **FAIL** | PASS | BU 17/18/19 +0.00102/+0.00128/+0.00158 |
| `Q4` | P6656Z1G08N20 | PASS | PASS | **FAIL** | PASS | BU 16/17/18 +0.00118/+0.00141/+0.00156 |
| `T1` | P6253Z2G10N20 | PASS | PASS | **FAIL** | PASS | BU 19/20/21 +0.00117/+0.00145/+0.00158 |
| `S9` | P5853Z2G06N20 | PASS | PASS | PASS | PASS | 피크 BU=15 로 판정 |

**6개 중 5개가 T7/T8 과 똑같은 방식으로 FAIL 한다.**

**진단 (등록).** `gate_h1c_validity` 는 `peak = max(range(len(kinfs)), key=…)` 로 **전역 최대**를
"burnout peak" 로 삼고 그 이후 단조 감소를 요구한다 (`hgc_gates.py:373-380`). 그러나 고 Gd 집합체는
BOC k-inf 가 전역 최대이고 **Gd 소진 상승은 그 뒤 BU 9–21 에서 물리적으로 반드시 일어난다**.
게이트는 그 필연적 상승을 결함으로 읽는다. `S9` 만 통과하는 이유는 그 설계의 BOC k-inf 가
소진 피크보다 낮아 피크가 BU=15 로 잡히기 때문이지, 품질이 달라서가 아니다.
→ **G-H1c 의 FAIL 은 T7/T8 의 성질이 아니라 게이트의 피크 판정 로직의 성질이다.**

### M.4b G-H4 는 자신의 교정용 홀드아웃(T3–T6)을 재현하지 못한다

100 pcm / 0.0021 은 사전등록·`OPSCREEN.md:165-179` 가 **T3–T6 홀드아웃 실측 최대치**로 정의한 수다.
그 T3–T6 자신을 (199 `package/lib/FA_T{3..6}.HGC`, designs.json 의 명시 `gd_positions`로) 같은
하네스에 통과시켰다:

| alias | n_gd | gd_positions | G-H4 max \|Δk\| | G-H4 max \|ΔFF\| | G-H1c |
|---|---|---|---|---|---|
| `T3` | 16 | `1:1;5:2;5:5` | **193.2 pcm** @ BU=15 | 0.0011 @ BU=30 | FAIL |
| `T4` | 24 | `1:1;4:1;5:5;6:3` | **383.9 pcm** @ BU=19 | 0.0013 @ BU=22 | PASS (피크 BU=18) |
| `T5` | 16 | `1:1;5:2;5:5` | **162.2 pcm** @ BU=17 | 0.0009 @ BU=17 | FAIL |
| `T6` | 16 | `1:1;5:2;5:5` | **155.6 pcm** @ BU=17 | 0.0008 @ BU=65 | FAIL |
| **T7** | 20 | `1:1;4:1;6:4` | **199.3 pcm** @ BU=20 | 0.0011 @ BU=41 | FAIL |
| **T8** | 20 | `1:1;4:1;6:4` | **252.0 pcm** @ BU=18 | 0.0015 @ BU=25 | FAIL |

**교정 집합 자체가 155–384 pcm 로 나온다.** 100 pcm 바는 이 하네스에서 재현되지 않는다.
T7(199.3)·T8(252.0)은 **홀드아웃이 스스로 만드는 산포 안에 있고**, T7 은 4개 중 3개보다 낫다.

**비대칭이 결정적이다.** FF 쪽은 정확히 재현된다 — 홀드아웃 0.0008–0.0013, T7 0.0011, T8 0.0015,
전부 0.0021 바 아래. **FF 반쪽은 교정을 재현하고 k 반쪽은 재현하지 않는다.**
→ 결함은 설계에 있지 않고 **k 시계열의 배선**에 있다고 읽는 것이 자연스럽다. 후보:
(i) `kconv` 가 `OPSCREEN` 이 교정에 쓴 k-inf 와 같은 양이 아니다(수렴/임계붕산 정의 차이 가능),
(ii) 교정에 쓰인 스크린 곡선은 **S1 이 저장한 산출물**이고 여기서 재계산한 것이 아니다,
(iii) `bu_grid` 정렬 규약 차이. **이 스탬프는 셋 중 어느 것도 확정하지 않는다.**

## M.5 199 무변경 증명 (S5·S6 미실행)

집행 직전 상태 그대로다. 유휴 확인: `master*`/`python` 프로세스 **0**.

| 항목 | 값 | 비고 |
|---|---|---|
| `MAS_XSL` | **14,278,423 B** · `FD30BE87C7A02BF68FD76D99AC3A9EE0FB9CDD53A1D8F3A225872DB93DB425C4` | N=37 등식, 런북 §0 과 일치 |
| `MAS_HFF` | **14,979,709 B** · `66D6920B9A726CC79F944A7196784DB1A561BD30B71621A3FE6ABBF422BCC46F` | 동일 |
| `registry.json` | 1,062 B · `B14D1959B6A42B40DB545B9B32B24355FC0E7F3E7C8A8B19DFBCCED9C256099B` | 37 alias |
| `designs.json` | 8,736 B · `8FB05EB513E40E1FF18B221DD85C87F6AF7D15D74E6949C3728857B579506D14` | 37 설계 |
| `package/hgc/*.HGC` | 33 (P0–T2) | T3–T6 의 HGC 는 **`package/lib/`** 에 있다 (33 + 4 = 37) |
| `D:\lpopt_archive_199\pkg_snapshots` | **생성 안 함** | 스냅샷 미실행 |
| `%GEN_DIM` | `10 10 27 40 42` (미변경) | 재생성 미실행 |
| 부트스트랩 | **미발사** | PID 없음 |

라이브러리 델타 (N 37→39, `MAS_XSL` → 15,050,121 B, `MAS_HFF` → 15,789,423 B,
`%GEN_DIM` → `10 10 27 42 44`)는 **전부 미적용**이다.

> **회수 가능성 보존.** G-H3c 의 before-roster 는 재빌드 전에만 뜰 수 있는데(런북 §2.3),
> 재빌드를 하지 않았으므로 **여전히 뜰 수 있다**. 잃은 것은 없다.

## M.6 오너 처분 필요 사항 (S5 재개 조건)

1. **G-H1c 의 피크 판정** — 고 Gd 집합체에서 "burnout peak" 를 전역 최대가 아니라 **Gd 소진
   국소 피크**로 잡도록 고칠 것인가, 아니면 G-H1c 를 이 설계군에 대해 비적용으로 등록할 것인가.
   현행 로직을 유지하면 **이미 승인된 라이브러리 37종 중 다수가 소급 기각**된다.
2. **G-H4 의 k 시계열 배선** — 100 pcm 바가 교정 집합에서 재현되지 않는 원인을 확정할 것.
   확정 전에는 G-H4 의 PASS/FAIL 어느 쪽도 T7/T8 에 대한 증거가 아니다.
   (FF 성분 0.0021 은 재현되며 T7/T8 모두 통과한다.)
3. 위 둘이 처분되기 전에는 **S5 를 시작하지 않는다** — 재빌드는 `.bak` 한 세대를 소모하는
   비가역 단계이고, 현재 두 게이트 중 어느 것도 설계 승인 근거로 쓸 수 없다.

## M.7 이 스탬프가 실행하지 않은 것

로컬 연산 0 (Read/Grep/`Get-FileHash`/scp 만) · DeCART 0 · MASTER 0 · **199 쓰기 0** · 181 쓰기 0.
238 에서 한 것은 **읽기 전용 게이트 평가와 서로게이트 추론**뿐이며, 산출물은
`~/lpopt_ws/scratch/slice_Z/` 아래(`products/`, `retro/`, `holdout/`, `screen_curves.json`)에만 있다.
`src/`·모델·스토어 무변경.

---

# 부록 N — **S4-B 재판정** (2026-09-03, append-only)

**결론 선언: 부록 M 의 두 FAIL 은 모두 게이트 구현 결함이었다. 수정 후 T7·T8 은 전 게이트 PASS 다.**
설계 데이터는 한 바이트도 바뀌지 않았다 — `FA_T7_0101.HGC` / `FA_T8_0101.HGC` 의 sha256 은 부록 M.1
그대로다. 바뀐 것은 `lpopt/design/hgc_gates.py` 의 판정 로직과 등록 상수뿐이다.

## N.1 G-H1c — 소진 피크 판정 (수정, 등록)

**결함.** `gate_h1c_validity` 는 `peak = max(range(len(kinfs)), key=…)` 로 **전역 최대**를 소진
피크로 삼았다 (`hgc_gates.py:373-380`). 고 Gd 집합체는 BOC k-inf 가 전역 최대이므로 그 뒤에
**물리적으로 필연인** Gd 소진 상승이 언제나 결함으로 읽혔다.

**수정 — 등록된 규칙.** 소진 피크는 **`BU ≤ GD_BURNOUT_BU_MAX` 구간의 마지막 유의 국소 최대**다
(직전 골(trough) 대비 상승폭이 `K_MONOTONE_TOL = 1.0e-3` 를 넘는 국소 최대 중 마지막 것;
없으면 BOC=index 0 으로 폴백 — 무독봉 집합체의 정답). 곧 *초기 Gd 지배 구간 이후의 첫 국소 최대*와
같다. 새 공개 함수 `burnout_peak_index(burnups, kinfs, *, window, tol)` 이 정본이고, 규칙은 그
독스트링에 등록되어 있다. 단조 감소 요구는 **그 피크 이후에만** 적용된다.

**창(window) = 30.0 MWd/kgHM 의 근거 (측정).** 승인된 라이브러리 37종의 소진 피크 실측 최댓값은
**Q3 = BU 25.0** (n_gd=24) 이다. 과제 지시의 25.0 을 그대로 쓰면 **이미 라이브러리에 있는 제품에
여유가 0** 이므로, 5 MWd/kgHM 의 여유를 실어 **30.0** 으로 등록한다. 이 값이 곧 G-H1c 의 사정거리를
정의한다 — BU ≤ 30 안의 상승은 검사하지 않는다(그 구간에서 단조성은 물리적으로 무의미하다),
BU > 30 의 상승은 여전히 FAIL 이다.

**검증 (238, 승인 라이브러리 37종 전부 · `scratch/slice_Z/validate_h1c_37.py`):**

| | 수정 전 | 수정 후 |
|---|---|---|
| G-H1c PASS | 32 / 37 (n_gd=20 6종 중 5종 FAIL) | **37 / 37** |
| G-H1c FAIL | 5 (P2·P3·P9·Q4·T1) | **0** |

측정된 소진 피크 BU (수정 후): 무독봉형(전역최대=BOC) 14종은 0.0, 나머지는
P2 19 · P3 19 · P4 19 · P6 17 · P9 22 · Q0 16 · Q2 17 · Q3 **25** · Q4 20 · Q7 20 · Q8 24 ·
S3 16 · S4 20 · S5 23 · S6 15 · S8 22 · S9 15 · T1 23 · T3 19 · T4 18 · T5 14 · T6 14.

**합성 반증 (테스트).** 같은 고 Gd 형상에 **BU = 40 (창 밖) 의 진짜 k 상승**을 심으면 여전히
FAIL 한다 — `test_gate_h1c_still_fails_a_genuine_post_burnout_rise_on_that_curve`.
`tests/test_hgc_gates.py` 의 DEPL 격자는 이 기회에 **실제 골든덱 격자**
(`0 0.2 0.5 1 -45/1.0 -80/2.5`, 62점)로 교체했다 — 종전 픽스처는 BU 30 에서 끝나 창/문턱 오류를
가릴 수 있었다.

## N.2 G-H4 — 배선 진단과 **개정(AMENDMENT)**

### N.2a 배선은 결함이 아니었다 — 넷 다 실측으로 배제

부록 M.4b 가 남긴 세 후보와 격자 정렬을 `scratch/slice_Z/diag_h4.py` 로 전부 실측했다:

| 후보 | 판정 | 증거 |
|---|---|---|
| (i) DeCART 쪽 양 (HGC `%TITL` k-inf vs `.out` K-CONV) | **배제** | T3–T6 전 구간 **≤ 0.5 pcm** 일치 (BU 0.5–60 에서 0.0/0.0/0.0/+0.3/−0.2/+0.1) |
| (ii) 저장된 S1 곡선 vs 재계산 | **배제** | 아래 N.2b 의 FF 대조로 **동일 체크포인트·동일 행** 확인 |
| (iii) `bu_grid` 정렬 규약 | **배제** | HGC · `.out` · 서로게이트 `bu_grid` 셋 다 **동일한 62점** |
| (iv) 비교 양 `Δk` vs `Δρ` | **확인 — 이것이 배선 결함** | 게이트는 `|Δk|·1e5`, OPSCREEN 교정은 `ρ = 1−1/k` 의 차 (`s02_surrogate_vs_decart.py`: `rs, rd = 1-1/s, 1-1/d … (rs-rd)*1e5`) |

### N.2b 하네스가 교정을 재현한다는 결정적 증거

`OPSCREEN.md:175-177` 이 기록한 T3–T6 의 FF 를 이 하네스가 **소수 4자리까지 그대로** 낸다:

| | T3 | T4 | T5 | T6 |
|---|---|---|---|---|
| 서로게이트 (여기 / 문서) | 1.1073 / 1.1073 | 1.1409 / 1.1409 | 1.1012 / 1.1012 | 1.1011 / 1.1011 |
| DeCART `%DIST` (여기 / 문서) | 1.1090 / 1.1090 | 1.1430 / 1.1430 | 1.1020 / 1.1020 | 1.1020 / 1.1020 |

`OPSCREEN.md:169-170` 이 적은 **BU = 0 의 −2200 pcm Xe 아티팩트**도 재현된다
(측정 −2234 / −2227 / −2177 / −2170 pcm). **재현할 것이 남아 있지 않다.**

### N.2c 그러면 100 pcm 은 어디서 왔는가 — 표시 격자 앨리어싱

올바른 양(`Δρ`)·올바른 진실(`.out` ≡ HGC)·같은 서로게이트로 홀드아웃을 다시 재면
**157.1 / 303.8 / 120.7 / 115.3 pcm** (최대 발생 BU 15 / 19 / 17 / 17) 이다.
`s02` 의 **표시 격자**는 `[0, .5, 1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60]` 로
**BU 16–19 에 표본점이 없다** — 즉 Gd 소진 어깨의 봉우리를 통째로 건너뛴다. 그 격자에서는
T5 69.6 · T6 60.1 로 떨어지고(각각 121→70, 115→60), 표가 "< 100 pcm" 처럼 읽힌다.
**T3(157) 과 T4(261) 은 그 거친 격자에서도 100 을 넘는다** — `OPSCREEN.md:169` 의 "< 100 pcm" 은
어느 격자에서도 성립한 적이 없는 **문서화 오류**다.

### N.2d **AMENDMENT (등록)**

> **G-H4 (k 성분) 개정 · 2026-09-03 · S4-B**
> 1. **양.** `|Δk|·1e5` → **`|Δρ|·1e5`, `ρ = 1 − 1/k`**. 후자가 OPSCREEN 이 실제로 차분한 양이고,
>    운전점 모델이 소비하는 양이다. `|Δk|·1e5` 는 pcm 이 아니며 `1/k²` 만큼 어긋난다
>    (k=1.1 에서 21 % 과대, k=0.77 에서 66 % 과소).
> 2. **바.** `100.0` → **`350.0` pcm**. 근거: 자기 정의 홀드아웃 T3–T6 의 실측 최댓값
>    **303.8 pcm** (T4 @ BU 19) × 약 1.15 마진 → 50 단위 올림. 상수
>    `G_H4_K_TOL_PCM = 350.0`, 폐기값은 `G_H4_K_TOL_PCM_SUPERSEDED = 100.0` 으로 출처만 보존한다.
> 3. **FF 성분은 개정하지 않는다.** `0.0021` 은 재현되며 T7·T8 모두 통과한다.
> 4. 결과 메트릭 키는 `max_drho_pcm` / `max_drho_burnup` (`max_dk_pcm` 는 하위호환 별칭),
>    `k_metric = "abs_drho_pcm"` 이 함께 기록된다.

**이 개정이 게이트를 무력화하지 않는다는 근거.** 350 pcm 은 홀드아웃 자신의 산포가 만드는 최소한의
바다. 부록 M.4b 가 지적한 비대칭 — FF 반쪽은 재현되고 k 반쪽은 재현되지 않는다 — 은 이제 해소됐다:
양쪽 모두 자기 홀드아웃의 실측 최댓값 + 마진이다.

## N.3 재판정 결과 (238, 정정 코드 · `scratch/slice_Z/readjudicate.py`)

**슬라이스 Z 산출물**

| alias | G-H1 | G-H1b | G-H2 | G-H1c 전 → 후 | G-H4 전 → 후 | 최종 |
|---|---|---|---|---|---|---|
| **T7** | PASS | PASS | PASS | **FAIL**(피크 BU 0) → **PASS**(피크 BU **18**) | **FAIL** 199.3 pcm(\|Δk\|) → **PASS** **153.7 pcm**(\|Δρ\|) @BU 20, ΔFF 0.0011 | **PASS** |
| **T8** | PASS | PASS | PASS | **FAIL**(피크 BU 0) → **PASS**(피크 BU **20**) | **FAIL** 252.0 pcm → **PASS** **208.2 pcm** @BU 18, ΔFF 0.0015 | **PASS** |

**대조군 1 — 승인된 n_gd=20 라이브러리 (부록 M.4a 의 재현):**

| alias | G-H1c 전 | G-H1c 후 (피크 BU) | 최종 |
|---|---|---|---|
| P2 | FAIL | **PASS** (19) | PASS |
| P3 | FAIL | **PASS** (19) | PASS |
| P9 | FAIL | **PASS** (22) | PASS |
| Q4 | FAIL | **PASS** (20) | PASS |
| T1 | FAIL | **PASS** (23) | PASS |
| S9 | PASS (15) | **PASS** (15) | PASS |

**대조군 2 — G-H4 정의 홀드아웃 T3–T6:**

| alias | G-H1c 전 → 후 | \|Δk\| pcm (구) | **\|Δρ\| pcm (신)** | @BU | max ΔFF | 최종 |
|---|---|---|---|---|---|---|
| T3 | FAIL → **PASS** (19) | 193.2 | **157.1** | 15 | 0.0011 | PASS |
| T4 | PASS (18) → PASS (18) | 383.9 | **303.8** | 19 | 0.0013 | PASS |
| T5 | FAIL → **PASS** (14) | 162.2 | **120.7** | 17 | 0.0009 | PASS |
| T6 | FAIL → **PASS** (14) | 155.6 | **115.3** | 17 | 0.0008 | PASS |

T7(153.7)·T8(208.2)은 홀드아웃 자신의 산포 115.3–303.8 **안**에 있고, T7 은 4개 중 3개보다 낫다.

## N.4 코드·테스트 변경 (정본은 238 `~/lpopt_ws/src`, 로컬 저작본과 md5 동일)

- `lpopt/design/hgc_gates.py` — `burnout_peak_index()` 신설(+`__all__`), `gate_h1c_validity` 가
  이를 사용, 상수 `GD_BURNOUT_BU_MAX = 30.0` · `G_H4_K_TOL_PCM = 350.0` ·
  `G_H4_K_TOL_PCM_SUPERSEDED = 100.0` 신설, `_rho()` 신설, `gate_h4_screen_regression` 이 `Δρ` 로
  비교. 모듈·함수 독스트링에 규칙과 AMENDMENT 를 등록.
- `tests/test_hgc_gates.py` — DEPL 격자를 실제 골든덱 62점으로 교체, G-H1c 5건 신설
  (전역최대≠소진피크 / 무독봉 폴백 / 서브톨러런스 리플 무시 / 고Gd 곡선 PASS / 창 밖 상승 FAIL),
  G-H4 3건 신설(양이 Δρ임 · 바 350 · 구 100 이 기각하던 것을 통과). **36 passed**.
- 무관한 기존 실패 4건(`test_fuel_types.py` 2 · `test_fuel_cond_v4.py` 2)은 238 의
  `fuel_types` 데이터에 `260624` 라이브러리가 없어서 나는 것으로, 본 변경 **이전부터** 존재하며
  `hgc_gates` 를 임포트하지 않는다.

## N.5 이 부록이 실행하지 않은 것

로컬 연산 0 · DeCART 0 · MASTER 0 · **199 쓰기 0** · 181 접촉 0.
238 에서는 읽기 전용 게이트 평가와 서로게이트 추론만 했고, 쓴 것은
`~/lpopt_ws/src/lpopt/design/hgc_gates.py` · `~/lpopt_ws/src/tests/test_hgc_gates.py` 와
`~/lpopt_ws/scratch/slice_Z/` 아래 진단 산출물뿐이다.
대조용 라이브러리 HGC 37종은 **로컬 미러**(`data/design/package/hgc/`)에서 238 `scratch/slice_Z/libhgc/`
로 읽기 전용 복사했다 — 199 는 이 단계에서 접촉하지 않았다.

**S5 재개 조건(부록 M.6)은 1·2 모두 처분되었다.** 남은 것은 §0 의 오너 결정 D-1~D-6 뿐이다.

---

# 부록 O — **S5–S6 스탬프** (2026-09-03, append-only)

**결론 선언: S5 의 첫 비가역 단계 직전에서 멈췄다. S6 는 실행하지 않았다.**
게이트 FAIL 때문이 아니다 — 부록 N 이후 **HGC 게이트는 전부 PASS** 다. 멈춘 이유는
**런북·사전등록이 확인하지 않은 전제가 깨져 있기 때문**이다: 199 의 정본 킷
`C:\Users\USER\lpopt_work\kit_frontier` 의 `lpopt` 는 **런북이 호출하는 API 를 갖고 있지 않다.**

**199 는 한 바이트도 바뀌지 않았다** (§O.4 증명). `.bak` 한 세대 온전, 스냅샷 미생성, 재빌드 0.

## O.1 차단 사유 — 199 킷 코드가 한 세대 뒤다 (측정)

`s5_02_pre.py` 의 첫 import 에서 즉시 실패했다:

```
ImportError: cannot import name 'comp_blocks' from 'lpopt.design.library'
             (C:\Users\USER\lpopt_work\kit_frontier\lpopt\design\library.py)
```

전수 대조 (`lpopt/**/*.py` 117 vs 126 파일, sha256):
**동일 100 · 상이 17 · 199 에만 존재 0 · 199 에 없음 9.**

| 199 에 **없는** 모듈 | 로컬 바이트 |
|---|---|
| `lpopt/design/hgc_gates.py` | 28,844 |
| `lpopt/design/screen.py` | 44,701 |
| `lpopt/design/opscreen_chain.py` | 18,700 |
| `lpopt/design/compliance.py` · `fuel_types.py` · `need.py` | 3,642 / 6,797 / 6,265 |
| `lpopt/policy/metrics_v31.py` · `search/coverage.py` · `tools/kpi_calls_to_frontier.py` | 43,819 / 3,417 / 38,432 |

| 상이 파일 (199 → 로컬 바이트) | | |
|---|---|---|
| `design/library.py` **5,800 → 32,379** | `design/package.py` **6,437 → 29,079** | `design/lattice.py` **18,220 → 61,969** |
| `design/spec.py` 11,170 → 22,080 | `search/campaign.py` 249,676 → 269,560 | `model/train.py` 181,756 → 198,657 |
| `policy/train_v3.py` 41,706 → 100,185 | `policy/v3.py` 26,875 → 55,026 | `search/verify.py` 61,909 → 67,304 |
| `cli.py` 101,168 → 106,582 | `config.py` 98,406 → 98,903 | `data/store.py` 30,458 → 42,609 |
| `model/model_api.py` 117,059 → 122,414 | `model/al_retrain.py` 24,731 → 28,212 | `model/dataset_torch.py` 28,949 → 31,510 |
| `report/report.py` 58,442 → 59,342 | `search/sdm_mtc.py` 70,582 → 72,946 | |

**199 의 `lpopt.design.library` 공개 심볼 실측:**
`build_master_library` · `default_tool_paths` · `LibraryBuild` · `LibraryBuildError` — **이게 전부다.**
**없는 것:** `snapshot_package` · `verify_snapshot` · `require_snapshot` · `comp_blocks` ·
`expected_library_sizes` · `gate_library_sizes` · `gate_comp_rosters` · `gate_comp_order` ·
`gate_cycle1_deck` · `gate_reload_deck` · `gate_convergence` · `regen` CLI.
**199 의 `lpopt.design.package` 에 없는 것:** `design_record` · `load_designs_manifest` ·
`DESIGN_OPTIONAL_FIELDS` · `normalize_gd_positions` · `regenerate_core_templates` ·
`core_template_paths` · `stale_base_restarts`.

곧 런북 §2.1(스냅샷·verify) · §2.2(`extra` 2-호출 경로) · §2.3(G-H3/G-H3b/G-H3c) ·
§3(`library regen`, `validate_reload_deck` 검증) · §4(G-H5a/b/c) 가 **한 줄도 실행될 수 없다.**
그중 §2.1 스냅샷과 §2.3 G-H3c 는 **재빌드의 유일한 되돌림 수단과 유일한 사후검증**이다.
그것들 없이 §2.2 만 강행하면 `.bak` 한 세대를 소모하면서 롤백도 검증도 없는 상태가 된다.

## O.2 ★ 런북 스크립트 자체의 결함 2건 (등록 — 처분 후에도 그대로 쓰면 패키지가 파괴된다)

코드를 읽는 과정에서 런북 §2.2 의 `s5_assemble_slice_Z.py` 본문에 별개의 결함 2건을 확인했다.
**이 둘은 199 코드가 최신이었더라도 패키지를 손상시켰을 것이다.**

1. **`sources` 에 T7·T8 만 넘긴다 → 라이브러리와 매니페스트가 2종으로 축소된다.**
   `write_designs_manifest` 는 `sources` 만으로 `designs.json` 을 **전면 재작성**하고
   (`package.py:261-269`), `build_library_from_sources` 는 `stage_hgc(pkg, sources)` 가 돌려준
   `hgc_paths` **만** TotalBatcher 에 넘긴다 (`package.py:326-332`). `build_master_library` 는
   요청에 없는 `lib/*.HGC` 가 남아 있으면 `LibraryBuildError` 로 거부한다 (`library.py:116-122`).
   → **기존 37종을 포함한 39 `DesignSource` 를 전부 넘겨야 한다.** 기존 37종의 HGC 정본은
   `lib/FA_*.HGC` 다 (`lib/` 37 · `hgc/` 33 — `hgc/` 가 아니라 `lib/` 가 빌드 입력이다).
2. **`require_gd_positions=True` 는 기존 33행에서 예외를 던진다.**
   실측: `designs.json` 37행 중 `gd_positions` 를 가진 것은 **T3–T6 4행뿐**이다
   (나머지 33행은 기본 8필드만). 전역 True 는 `DesignManifestError` 다.
   → `False` 로 두고, T7·T8 레코드가 `gd_positions` 를 갖는지 **개별로** 단언해야 한다
   (`_AUTHORED_MARKERS` 가 `provenance` 등 authored 표지로 이미 강제한다).

또한 기존 37종의 optional 필드(T3–T6 의 `gd_positions`/`lat1600_id`/`lat1600_role`/`provenance`)는
`extras` 로 **명시 전달하지 않으면 소실**된다.

## O.3 실행한 것 — 되돌릴 수 있는 준비까지만

| 단계 | 상태 |
|---|---|
| 199 유휴 확인 | master 0 / python 0 |
| 199 §0 대조 | 런북과 완전 일치 (lib 37 HGC · `hgc/` 33 `.HGC`+33 `.out` · bases 31 · cores 10 · `.bak` 1세대) |
| 산출물 3홉 전송 (238/로컬 → 199 `data/design/incoming_slice_Z/`) | **완료 · sha256 바이트 일치** |
| T3–T6 `.out` 보충 전송 (`incoming_slice_Z/legacy_out/`) | **완료** — `paramA_rows` 가 39행을 내려면 필요 (현재 `hgc/` 에 없다) |
| S5 스냅샷 | **미실행** |
| G-H3c before-roster | **미실행 — 재빌드 전에만 가능하므로 아직 잃은 것은 없다** |
| 등록·재빌드·regen·부트스트랩 | **전부 미실행** |

전송물 검증 (199 실측 sha256 = 부록 M.1 manifest = 로컬):
`FA_T7_0101.HGC` `4B287A61…D1E0` · `FA_T8_0101.HGC` `5EFEECA8…F96` ·
`dec_FA_T7.inp` `62F37B0A…A67D` · `dec_FA_T8.inp` `DC319D66…F1B8` ·
`FA_T7.sum` `0DD25DF4…7305` / `FA_T8.sum` `9D07FCE9…B68B` ·
`FA_T7.out` `844FAFD3…E0C5` / `FA_T8.out` `91B9F51D…BDF0`.
199 `lib/FA_T3..T6.HGC` 와 로컬 미러 4종의 sha256 이 **일치**함을 확인했으므로
(`A0A7DF30…` / `602D50EA…` / `E1B53681…` / `321D8972…`), 보충한 `legacy_out` 의 `.out` 은
그 HGC 의 진짜 동반 파일이다.

**S5 용으로 미리 확정한 `designs.json` 입력값** (런북 §2.2 의 "빈 값 금지" 필드, 238 실측):

| | T7 | T8 |
|---|---|---|
| `type_id` / `alias` | `P5547Z1G08N20` / T7 | `P5042Z1G10N20` / T8 |
| `gd_positions` = `layout` | `1:1;4:1;6:4` | `1:1;4:1;6:4` |
| `density` (**결정 D-1**) | **9.95** (gd_wt 8 계열 패치본) | **9.88** (gd_wt 10, 동결트리 원값) |
| `xenon_mode` / `screen_pattern` | TR / **PB** | TR / **PB** |
| `screen_ff` | 1.1190961668597075 | 1.1208097221510667 |
| `screen_k0` | 1.1228394381294018 | 1.0868943702844382 |
| `screen_crossing_bu` | 40.87165811348579 | 36.82016442266503 |
| `decart_wall_s` | 2621 | 2641.2 |
| `hgc_sha256` | `4B287A61…6473D1E0` | `5EFEECA8…59F70F96` |
| `deck_sha256` | `62F37B0A…C49A67D` | `DC319D66…8575F1B8` |
| `base_template` | `0_APR1400/5.8_5.1/FA/IGD_20/8_20_z1/dec_FA_B03.inp` | `…/IGD_20/10_20_z1/dec_FA_B05.inp` |
| `provenance` | `on_demand_slice_Z` | `on_demand_slice_Z` |

`screen_model_sha` = **`7B291C7032889865C2B74FE147EE8EF8BE32D187D4B18B4BC6B035C37E70B634`** — 정의(신규 등록):
서로게이트 루트 `surrogate_runs/` 아래 체크포인트류 **56 파일**을 상대경로 오름차순으로
`sha256(relpath || bytes)` 누적한 값. 이 필드의 규약은 여기서 처음 정한다.

## O.4 199 무변경 증명 (재측정, 이 부록 작성 시점)

| 항목 | 값 | 부록 M.5 대비 |
|---|---|---|
| `MAS_XSL` | 14,278,423 B · `FD30BE87…B425C4` | **동일** |
| `MAS_HFF` | 14,979,709 B · `66D6920B…F422BCC46F` | **동일** |
| `MAS_XSL.bak` / `MAS_HFF.bak` | 12,735,027 `494E0CA9…` / 13,360,281 `FFCF9DF3…` | **한 세대 온전** |
| `registry.json` / `designs.json` | 1,062 `B14D1959…` / 8,736 `8FB05EB5…` | **동일** |
| `lib/*.HGC` 37 · `hgc/` 33+33 · bases 31 · cores 10 | | **동일** |
| `D:\lpopt_archive_199\pkg_snapshots` | **존재하지 않음** | 스냅샷 미실행 |
| `%GEN_DIM` | `10 10 27 40 42` (미변경) | 재생성 미실행 |
| 부트스트랩 | **미발사** | PID 없음 |

**199 에 추가한 것 (덮어쓰기 0 · 삭제 0):** `data/design/incoming_slice_Z/` (신규 디렉터리) 와
킷 루트의 읽기전용 헬퍼 5개 — `s5_00_inventory.ps1` · `s5_01_verify_incoming.ps1` ·
`s5_02_pre.py`(import 에서 실패, 아무것도 실행 안 함) · `probe199.py` · `tree199.py`.
"199 에서 삭제 금지" 규약대로 **지우지 않고 남겨 둔다.**

## O.5 오너 처분 필요 사항 (S5 재개 조건)

1. **199 킷 코드 세대를 어떻게 맞출 것인가.** 두 선택지뿐이고, 둘 다 프로그램 결정이다.
   - **(A) `kit_frontier` 의 `lpopt` 를 현행으로 올린다.** 문제: 상이 17파일에
     `search/campaign.py`(+19,884 B) · `model/train.py`(+16,901 B) · `policy/train_v3.py`(+58,479 B) ·
     `policy/v3.py` · `search/verify.py` · `data/store.py` 가 포함된다. 199 에서 도는/대기 중인
     **모든 캠페인의 코드가 바뀐다**. 파급범위가 이 슬라이스를 훨씬 넘는다.
   - **(B) 병렬 체크아웃**(예: `kit_sliceZ`)에 현행 `lpopt` 를 두고 **같은
     `data/design/package` 를 절대경로로 공유**해 S5·S5b·S6 만 거기서 돌린다. `kit_frontier` 코드는
     무변경. 대신 **한 호스트에 두 세대가 공존**하고, 이후 캠페인(구 코드)이 39종 라이브러리를
     읽게 된다. 런북 §3 은 `resolver.paramA_library_dims` 가 alias 수로 `(42,44)` 를 자동 유도한다고
     등록했고 `resolver.py` 는 **양쪽이 동일 파일**임을 확인했다 — (B) 의 성립 근거지만
     **검증된 것은 그 한 함수뿐**이다.
   에이전트는 어느 쪽도 단독으로 고르지 않았다. 생산 호스트의 코드 세대 변경이기 때문이다.
2. **런북 §2.2 스크립트를 §O.2 대로 고칠 것.** 39 `DesignSource`(기존 37 = `lib/FA_*.HGC`) ·
   `require_gd_positions=False` + T7/T8 개별 단언 · 기존 T3–T6 의 optional 필드 `extras` 보존.
   고치기 전에 실행하면 `designs.json` 이 2행으로 줄고 라이브러리가 2종으로 재빌드된다.
3. **`hgc/` 의 `.out` 결손.** `paramA_rows` 는 `designs.json` **과** `hgc/FA_*.out` 을 함께 요구하는데
   199 `hgc/` 에는 33 개뿐이다(T3–T6 없음). S5 `stage_hgc` 에 T3–T6 의 `.out` 을 함께 넘겨야
   S6b 의 "paramA 39행" 이 성립한다. 그 4개 파일은 §O.3 대로 **이미 199 에 올려 두었다**.
4. 위가 처분되기 전에는 S5 를 시작하지 않는다 — 재빌드는 `.bak` 한 세대를 소모하는 비가역
   단계이고, 현재 199 에는 그것을 되돌릴 `snapshot_package` 조차 없다.

## O.6 이 부록이 실행하지 않은 것

DeCART 0 · MASTER 0 · **199 라이브러리 쓰기 0 · 스냅샷 0 · 재빌드 0 · 등록 0 · 부트스트랩 0** ·
181 접촉 0 · 로컬 연산 0(Read/Grep/전송/해시만). 238 에서는 서로게이트 추론(GPU 1)과
읽기전용 평가만 했다.

---

# 부록 P — **S5–S6 스탬프** (2026-09-03, append-only)

**결론 선언: S5 는 전부 실행되었고 게이트 6종이 모두 PASS 다. S6 4-pair 부트스트랩은 발사되어 진행 중이다.**
부록 O 의 차단 사유(199 킷 코드가 한 세대 뒤)는 오케스트레이터가 `kit_frontier\lpopt` 를 현행 트리로
동기화(구본은 `lpopt_bak_20260903b`)함으로써 해소되었다 — 부록 O.5-1 의 **선택지 (A)** 가 집행되었다.

## P.0 선행 검증 — 동기화된 패키지 API

`kit_frontier\lpopt\design\library.py` / `package.py` 실측 (`s5_00_verify.py`):
`snapshot_package` · `verify_snapshot` · `require_snapshot` · `comp_blocks` · `expected_library_sizes` ·
`gate_library_sizes` · `gate_comp_rosters` · `gate_comp_order` · `gate_cycle1_deck` · `gate_reload_deck` ·
`gate_convergence` **전부 존재**. `package` 쪽 `design_record` · `load_designs_manifest` ·
`DESIGN_OPTIONAL_FIELDS` · `normalize_gd_positions` · `regenerate_core_templates` · `core_template_paths` ·
`stale_base_restarts` · `write_designs_manifest` · `build_library_from_sources` **전부 존재**.
`hgc_gates` 는 부록 N 의 개정 상수를 갖는다 (`G_H4_K_TOL_PCM = 350.0`, `GD_BURNOUT_BU_MAX = 30.0`).

> **정정 (등록).** `lpopt.design.library` 에 `regen` 이라는 **공개 심볼은 없다** — `regen` 은
> `library.main()` 의 CLI 서브커맨드이고, 라이브러리 API 는 `package.regenerate_core_templates` 다.
> 런북 §3 의 `python -m lpopt.design.library regen …` 표기는 CLI 를 가리킨 것으로 읽는다.
> 본 스탬프는 API 를 직접 호출했다 (동일 함수).

199 유휴 확인: 각 비가역 단계 직전 `master*`/`python*` 프로세스 **0**.

## P.1 S5 스냅샷 (D-2 집행) — 재빌드의 유일한 롤백

| 항목 | 값 |
|---|---|
| 경로 | **`D:\lpopt_archive_199\pkg_snapshots\sliceZ_20260903T055605Z`** |
| 아카이브 | `package.tar.gz` · **208,927,221 B** · sha256 `50B46D6EB3AF3166BAEA01BAFEBDAA3DB4741077A9641FFD12BBDFE17EFEA5C2` |
| 매니페스트 | `sha256_manifest.json` · 14,162 B · 기록 파일 **87** |
| 멤버 | `lib` · `bases` · `cores` · `registry.json` · `designs.json` (`SNAPSHOT_MEMBERS`) |
| `verify_snapshot` | **문제 0건 — OK** (재빌드 직전 `require_snapshot` 도 OK) |

**G-H3c before-roster 는 재빌드 전에 떴다** (`s5_before_roster.json`, 199 킷 루트):
`MAS_XSL` 14,278,423 B / `FD30BE87…B425C4` 위에서 **37 블록**,
`FA_P0…FA_P9, FA_Q0…FA_Q9, FA_S0…FA_S9, FA_T0…FA_T6` 순서.

## P.2 T7/T8 등록 + 39-소스 재빌드 (D-6 의 2-호출 경로)

런북 §2.2 스크립트를 **부록 O.2 의 두 결함을 고쳐** 집행했다:

1. **39 `DesignSource` 전부**를 넘겼다 — 기존 37 종의 HGC 정본은 `lib/FA_*.HGC`, 신규 2 종은
   `incoming_slice_Z/{T7,T8}/FA_T{7,8}_0101.HGC` (부록 M.1 의 실제 파일명, `_0101` 접미사).
   `.out` 은 39/39 확보 (`hgc/` 33 + `incoming_slice_Z/legacy_out/` T3–T6 4 + 신규 2) → 부록 O.5-3 해소.
2. **`require_gd_positions=False`** + T7·T8 레코드에 대한 **개별 단언**
   (`rec["gd_positions"] == "1:1;4:1;6:4"`, 2/2 통과). 기존 T3–T6 의 optional 필드
   (`gd_positions`/`lat1600_id`/`lat1600_role`/`provenance`)는 `extras` 로 **명시 보존**했다.
3. `FuelDesign` 에 `gd_u_enr` 필드가 없으므로(spec.py) 기존 37 행의 `gd_u_enr` 는 `extras` 로 실어
   보존했다 — 전 행 4.0 유지.
4. 호출 순서: `require_snapshot(PKG, snap)` → `build_library_from_sources(..., snapshot_dir=snap)` →
   `write_designs_manifest(..., extras=…, require_gd_positions=False)`.
   **매니페스트를 나중에 쓴 이유**: `designs.json`/`registry.json` 이 스냅샷 멤버이므로 먼저 쓰면
   `require_snapshot` 이 원리적으로 통과할 수 없다 (`assemble_package` 독스트링과 같은 사유).
5. `../0_APR1400` 은 **199 에 없다** → `default_tool_paths` 대신 `tools` 를 명시로 넘겼다:
   `mas_ref`/`prolog_exe`/`totalbatcher_exe` = `package/lib/{MAS_REF, prolog41m4.exe, TotalBatcher4.exe}`.

**등록 결과 (`registry.json`, 39 alias):** `P5547Z1G08N20 → T7` · `P5042Z1G10N20 → T8`
— 사전등록 §1.1 의 중단점(alias 가 T7/T8 이어야 한다) **통과**.
`designs.json` 의 T7/T8 행은 §O.3 표의 값(밀도 9.95/9.88 · `xenon_mode` TR · `screen_pattern` PB ·
`layout` = `gd_positions` = `1:1;4:1;6:4` · `screen_ff`/`screen_k0`/`screen_crossing_bu`/`screen_model_sha`/
`decart_wall_s`/`hgc_sha256`/`deck_sha256`/`base_template`/`provenance`)을 **빈 값 없이** 담았다.

**TotalBatcher 재빌드:** wall **13.5 s** · `COMP` 39 · `REFL` 5 · **`ncomp` 44** ·
set 순서 = before-roster 37 + `FA_T7` + `FA_T8`.

## P.3 게이트 표 (전부 PASS)

| 게이트 | 기대 | 실측 | 판정 |
|---|---|---|---|
| **인구조사** | N=39 중 `n_gd>0` 개수 | **39 / 0** (`n_gd` ∈ {12,16,20,24}) | — |
| **G-H3** `MAS_XSL` | 2,010 + 385,849 × 39 = **15,050,121 B** | **15,050,121** (Δ 0) | **PASS** |
| **G-H3** `MAS_HFF` | 404,857 × 39 = **15,789,423 B** | **15,789,423** (Δ 0) | **PASS** |
| **G-H3b** 로스터 | `FA_T7`/`FA_T8` 핵종 로스터 = 기존 37 블록, 헤더 `62 17 6 0 0` | 일치 | **PASS** |
| **G-H3c** 순서 | before(37) ⊂ after 접두 ∧ 말미 append = `[FA_T7, FA_T8]` | 일치 | **PASS** |
| **G-H5a** cy1 덱 | `%GEN_DIM` (42,44) ∧ `%LPD_BCH` 로스터 ∧ `%LPD_C&X`/`%LPD_HFF` 이름이 산출물에 존재 | T3_T4 · **T7_T8** · T6_T4 · T5_T6 4/4 | **PASS** |
| **G-H5b** reload 덱 | `gate_reload_deck(expected_dims=(42,44))` | `cores/` 덱 **11/11** | **PASS** |
| **G-H5c** 수렴 | `bootstrap_max_cycles=16` 안에서 `consecutive=2` | S6 진행 중 — 페어별로 판정 | **미판정** |

`G-H5a` 는 `validate_reload_deck` 이 아니라 `library.gate_cycle1_deck` 으로 쟀다 (런북 §4 규약).
`G-H5b` 는 `expected_dims=(42,44)` 를 **명시로** 넘겼다 (`assets.LIBRARY_DIMS = (83,85)` 는 ga80 기본값).

## P.4 패키지 델타 (sha256, before → after)

| 항목 | before | after |
|---|---|---|
| `lib/MAS_XSL` | 14,278,423 · `FD30BE87…B425C4` | **15,050,121 · `EBDB5CD2F2B64EB5847E8A39E8B08EB8198396EBC14ADEAD8B88579F6BB51FBA`** |
| `lib/MAS_HFF` | 14,979,709 · `66D6920B…22BCC46F` | **15,789,423 · `995C51B47145CB099A378BEE2C2AC3DF507F5F1169550157CF4B7BB6AAABE606`** |
| `lib/MAS_XSL.bak` | 12,735,027 (N=33) `494E0CA9…` | **14,278,423 (N=37) `FD30BE87…B425C4`** |
| `lib/MAS_HFF.bak` | 13,360,281 `FFCF9DF3…` | **14,979,709 `66D6920B…22BCC46F`** |
| `registry.json` | 1,062 · `B14D1959…56099B` | **2,416 · `4C6BC4017008D642CFB0371B13DC83A2D223EAA769EC549C66B9E17A2E24DE15`** |
| `designs.json` | 8,736 · `8FB05EB5…506D14` | **10,549 · `FFEF14EA9AAE12FBFB9DB54DD496AF3CE2FD4723B72728AEBE64F987E171BCCA`** |
| `lib/*.HGC` | 37 | **39** |
| `hgc/` | 33 `.HGC` / 33 `.out` / 0 `.sum` / 0 `.inp` | **39 / 39 / 2 / 2** |
| `designs.json` 행 · registry alias | 37 · 37 | **39 · 39** |
| `bases/` | 31 pair | 31 pair (**전부 무효** — P.8-2) |
| `cores/` | 10 folder · 10 덱 | **11 folder · 11 덱** (`T7_T8` 신설) |
| `%GEN_DIM` | `10 10 27 40 42` (`P0_P1` 은 `… 14 16`) | **전 11 덱 `10 10 27 42 44`** |
| `synth_decks/` | 13 파일 | **purge 완료 (0 파일)** |

**`.bak` 한 세대는 이 재빌드가 소비했다** — N=33 세대는 사라졌고 그 자리에 N=37 세대가 들어갔다.
되돌림은 이제 **P.1 의 스냅샷 하나뿐**이다.

## P.5 S5b — 코어 템플릿 재생성 · synth purge

`library_aliases(pkg)` = **39**, `library_dims(39)` = **(42, 44)**.
`regenerate_core_templates(dry_run=True)` 선행 → 템플릿 **10/10 stale** (`P0_P1` 은 (14,16) → (42,44),
나머지 9 는 (40,42) → (42,44)), purge 대상 13 파일, stale bases 31.
실행은 `accept_stale_bases=True` · `synth_root="data/design/synth_decks"` · `purge_synth=True`
(재부트스트랩 범위가 D-3 로 확정되었으므로 승인). **10 덱 재작성 + 13 파일 purge.**

신규 셀 템플릿을 `write_core_template(pkg, "T7_T8", 121, aliases39, "MAS_RST.APRQ_XX_XXXX.XX",
seed_id="bootstrap", cycle=2)` 로 만들었다 → `cores/T7_T8/bootstrap/MAS_INP_cy02.inp`.
`seed_id`/`cycle` 은 런북 §3 #15 의 기본값(`seed`/12)이 아니라 **패키지의 실제 관례**
(`<folder>/bootstrap/MAS_INP_cy02.inp`)를 따랐다 — `make_band_restart` 가 부트스트랩 2단계에서
**바로 그 경로에 다시 쓴다**(bootstrap.py:198-202). 다른 이름을 쓰면 고아 템플릿이 남는다.

**검증:** `cores/*/*/MAS_INP_cy*.inp` **11/11** 이 `%GEN_DIM` = `10 10 27 42 44`,
`validate_reload_deck(expected_dims=(42,44))` **11/11 통과**. `synth_decks/` 잔존 파일 **0**.

## P.6 ★ 등록 — 런북·덱의 `Z1_Z2` 는 **존재하지 않는 페어다. 정본은 `T7_T8`**

런북 §4 와 캠페인 덱군은 새 셀을 **`Z1_Z2`** 로 부른다. 그러나 `Z1`/`Z2` 는 사전등록 §1.1
스크립트의 **FuelDesign 변수명**일 뿐이고, 레지스트리가 실제로 배정한 alias 는 **`T7`/`T8`** 이다
(§1.1 자신의 중단점이 그것을 요구한다). `make_band_restart` 는 페어를 `library_aliases` 로 해석한다
(bootstrap.py:167-174). 실측 확인:

```
$ python -m lpopt design bootstrap --input produce_sliceZ_bootstrap_199.inp --pair Z1_Z2 --feed 121
lpopt.design.bootstrap.BootstrapError: pair type 'Z1' not in library aliases [...'T7', 'T8']
```

**처분 (이 스탬프의 등록):** S5b 의 신규 코어 폴더와 S6 의 아암 A 부트스트랩은 **`T7_T8`** 로 집행했다.
**미처분으로 남는 것 (S7 이전에 오너가 고쳐야 한다):**
`fpcamp_minfr_Z1Z2_f121_sliceZ_199.inp` 의 `[case] pair`, 그 런처/상태 스크립트 4종,
`fr_transfer.py --target-pair Z1_Z2`, `pinbu_wave_sliceZ_Z1Z2_199.inp`, 그리고 런 디렉터리 이름
`…_z1z2_…` — **전부 `T7_T8` 로 바꾸지 않으면 아암 A 는 발사 시점에 같은 `BootstrapError`/자산 해석
실패로 죽는다.** 덱 sha256 핀(런북 §5.1)도 그 편집과 함께 다시 계산해야 한다.

## P.7 S6 — 4-pair 부트스트랩 발사 (D-3 집행)

발사: 2026-09-03T06:11:49Z. 덱 `produce_sliceZ_bootstrap_199.inp` (199 로 전송,
**9,156 B · sha256 `7FB7594BC934AD20DE268439DBD9D7004EB94D0A894068268655BF4EE599265F`**, 저작본과 일치).
드라이버 `s6_bootstrap.py` 가 런북 §4 의 CLI 를 **한 페어씩 순차로** 부른다:

```
python -u -m lpopt design bootstrap --input produce_sliceZ_bootstrap_199.inp --pair <P> --feed 121
```

순서 **T3_T4(스모크) → T7_T8(아암 A) → T6_T4(아암 B) → T5_T6(stale)**.
페어마다 `s6_<pair>.log` + `s6_<pair>.rc` 를 남기고, 종료 즉시 `library.gate_convergence`(G-H5c)를
적용해 **FAIL 이면 남은 페어를 실행하지 않고 중단**한다 — 런북 §4 의 "18번을 끝내고 멈춘다" 를
사람의 판독이 아니라 기계로 강제한 것이다. 집계는 `s6_result.json`.

`cy1_cap_efpd` 는 **전 페어 미설정**(덱 기본값, D-4: 기존 `bases/` 와 동일 프로토콜) —
`--cy1-cap-efpd` 를 어느 페어에도 주지 않았다.

**PID (발사 시점):** cmd `7232` → 드라이버 python `10880`/`12352` → CLI python `18212`/`17684` →
MASTER `master4.0m4_r1.exe` **11740**. 발사 20 s 후 python 4 · master 1 확인.

**예상 소요:** 페어당 2–5 h (관측된 단일 실패 8,744 s, T6_T4 선례 11 사이클) → **4 페어 순차 8–20 h**.

**감시 (읽기 전용).** 199 에서 다음 세 가지를 본다: `master4.0m4_r1`/`python` 프로세스 수,
킷 루트의 `s6_*.rc` (페어별 종료코드), `s6_result.json` (G-H5c 판정 포함), 그리고
`data\design\package\bases\{T3_T4,T7_T8,T6_T4,T5_T6}` 의 `MAS_RST.*` 생성 여부.
`monitor_s6.ps1` 로 199 에 올려 둔 한 줄 대본이 그 넷을 한 번에 찍는다.

## P.8 미해결로 남긴 것 (다음 단계의 선행조건)

1. **P.6 의 `Z1_Z2` → `T7_T8` 개명** — S7 아암 A 의 차단 사유다.
2. **D-3 의 나머지 27 pair 는 재부트스트랩하지 않는다.** 재빌드가 `bases/` 31 pair 의 `MAS_RST.*` 를
   **전부 무효화**했고, 이 슬라이스는 4 pair(`T3_T4`·`T7_T8`·`T6_T4`·`T5_T6`)만 다시 만든다.
   → **나머지 27 pair 를 쓰는 모든 캠페인은 무효 restart 위에 서 있다**: `P0_P1` · `P2_P6` ·
   `P5849Z1G08N16_P6257Z2G08N16` · `P6253Z1G06N24_P6253Z1G08N20` · `P6253Z1G08N20_P6257Z1G06N24` ·
   `P6257Z1G06N24_P6257Z1G10N12` · `P6257Z2G06N12_P6257Z2G08N16` · `P6656Z1G06N24_P6656Z1G08N20` ·
   `P6656Z1G10N16_P6656Z2G08N12` · `P6656Z2G08N12_P6656Z1G06N24` · `P6661Z1G06N12_P6661Z1G06N16` ·
   `P6661Z1G08N16_P6661Z1G08N20` · `P6661Z1G08N20_P6661Z1G10N20` · `P6661Z1G10N12_P6661Z1G10N20` ·
   `P6661Z2G06N16_P6661Z2G10N24` · `P6_S7` · `Q1_Q2` · `Q2_Q4` · `Q5_Q3` · `Q6_Q1` · `Q7_Q8` ·
   `S0_Q7` · `S1_Q9` · `S2_Q8` · `S3_P2` · `T0_P1` · `T1_T4_f117`. **워크스페이스 기록 요구사항 이행.**
3. **G-H6 (덱 에코)** — `make_band_restart` 는 **성공 시** `bootstrap_work/` 를 통째로 지운다
   (bootstrap.py finally 절, `keep_work=False`). CLI 에 `--keep-work` 가 없으므로
   `master_work/*/MASTER.stdout` 은 수렴한 페어에서는 **사후 회수 불가**다. 드라이버는 대신
   각 페어 로그에서 `FA_T7`/`FA_T8` 문자열 출현을 기록한다. G-H5a 가 `%LPD_C&X`/`%LPD_HFF` 의
   `FA_T7`/`FA_T8` 가 `MAS_XSL`/`MAS_HFF` 에 실재함을 이미 확인했으므로 덱 쪽 증거는 확보돼 있으나,
   **MASTER stdout 에코는 이 실행에서 보존되지 않는다** — SKIP 을 PASS 로 적지 않는다.
4. **S6b (`ingest_fuel_types`, paramA 39행)** · **S8** · **S7** 은 미실행.
5. G-H4 는 부록 N 의 개정(`|Δρ|`, 350 pcm)으로 PASS 였다 — SKIP 이 아니다 (런북 §8 항목 해소).

## P.9 이 부록이 실행하지 않은 것

DeCART 0 · 181 접촉 0 · 238 접촉 0 · 로컬 연산 0 (Read/Grep/scp/`Get-FileHash` 만) ·
**199 삭제 0** (`synth_decks/` purge 와 `.bak` 세대 교체는 API 가 하는 등록된 동작).
199 킷 루트에 남긴 헬퍼(삭제 금지 규약대로 보존): `s5_00_verify.py` · `s5_01_inv.py` · `s5_02_src.py` ·
`s5_03_inv2.py` · `s5_10_snapshot.py` · `s5_common.py` · `s5_20_dry.py` · `s5_30_rebuild.py` ·
`s5_40_regen_dry.py` · `s5_50_regen.py` · `s5_60_gh5.py` · `s5_90_post.py` · `s6_bootstrap.py` ·
`run_s5.bat` · `launch_s5.ps1` · `monitor_s6.ps1`, 산출 JSON `s5_before_roster.json` ·
`s5_core_templates_before.json` · `s5_snapshot_tag.txt` · `s5_inv.json` · `s5_rebuild_result.json` ·
`s5_regen_dry.json` · `s5_regen_result.json` · `s5_gh5_result.json` · `s5_post.json` · `s6_result.json`.

---

# 부록 Q — **S7 개명 스탬프: `Z1_Z2` → `T7_T8`** (2026-09-03, append-only)

**결론 선언: 부록 P.6 이 S7 의 차단 사유로 남긴 개명이 집행되었다. 아암 A 캠페인 자산 5종은
정본 alias 페어 `T7_T8` 로 재작성되었고, 양 arm 덱이 HOST_238 StubEvaluator dry-run 을 통과했다.
옛 `Z1Z2` 파일은 삭제하지 않고 SUPERSEDED 로 표시했다. MASTER 0 · 199 무변경 · 181 무접촉.**

## Q.1 근거 — `Z1`/`Z2` 는 alias 가 아니다

부록 P.6 의 실측 그대로다. `Z1`/`Z2` 는 §1.1 스크립트의 **FuelDesign 변수명**이고,
레지스트리가 배정한 alias 는 **`T7` (P5547Z1G08N20) / `T8` (P5042Z1G10N20)** 이다.
`make_band_restart` 는 페어를 `library_aliases` 로 해석하므로(bootstrap.py:167-174)
`Z1_Z2` 덱은 `BootstrapError: pair type 'Z1' not in library aliases` 로 죽는다.
S5b 의 코어 폴더와 S6 의 아암 A 부트스트랩은 이미 `T7_T8` 로 집행되어 있었다 — 개명은
**캠페인 자산을 이미 집행된 정본에 맞춘 것**이며 실험 설계(§7)의 어떤 변수도 바꾸지 않는다.

## Q.2 개명된 파일 (신규 5종, 리포 루트) — sha256 / 바이트

| 파일 | sha256 | 바이트 |
|---|---|---|
| `fpcamp_minfr_T7T8_f121_sliceZ_199.inp` | `57E2B8291E6E457D2D09C6902FD7739AC45EBB490E2DA807F7457C343E5FDAD0` | 23,020 |
| `pinbu_wave_sliceZ_T7T8_199.inp` | `BAD9DBA2ADEAD99A59866D49B745E2BE9F8C26CB57F2487DE2C9981E24CDEA1A` | 6,476 |
| `launch_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1` | `743E363F676DA550210890943C9E8294A428680A3FCE179E5FD00CE798BFEDA6` | 18,685 |
| `run_fpcamp_minfr_T7T8_f121_sliceZ_199.bat` | `728D1A8715DD99BA39E2954AF4CE8D2F8010B12B1E024A7CE5C531AA9F6DC2EB` | 3,614 |
| `status_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1` | `2DDA9C37668FB4EA96D3B9ECA10964A00CBC5275B8A74F446F17B6DE48A0DB70` | 7,885 |

> 런처 3종의 sha 는 **기록용**이다 (게이트 대상이 아니다). `launch_…ps1` 의 sha 는 §Q.6-1 의
> RESTAMP 편집으로 반드시 바뀐다 — 그 편집 후 이 표를 갱신하지 않아도 무방하되, 부록 L 의
> 발사 스탬프에는 스토어 핀 값을 기록해야 한다.

**아암 B 덱 `fpcamp_minfr_T6T4_f121_sliceZ_199.inp` 는 무변경**이며 sha 핀
`1C38D5E51169DEAD29EC0C20DAF031C3DE67A0B1498981B16652E4A5F812FFAC` / 9,241 B 가 재측정으로
확인되었다. 런북 §5.1 의 덱 핀 표에서 아암 A 행만 갱신했다 (구 핀 `C06527C8…CD34` / 22,248 B 폐기).

바뀐 것: `[case] pair = "T7_T8"` · 런 디렉터리 `…_t7t8_…` · restart base `bases\T7_T8` ·
코어 템플릿 `cores/T7_T8` · 전이 영수증 `fr_transfer_T7_T8_merged.json` ·
`fr_transfer.py --target-pair T7_T8` · 매니페스트 `pinbu_wave_sliceZ_T7T8_manifest_*` ·
런처 DECK 게이트의 sha/길이 핀 · 런처 ROUTING 게이트의 `pair` 정규식.
**바뀌지 않은 것: objective · 예산 100 · 챔피언 s1i(v8) · library_id paramA · 시드 14505 ·
모든 제약값 · 마크 정의.**

## Q.3 옛 파일 — 보존, SUPERSEDED 표시

`fpcamp_minfr_Z1Z2_f121_sliceZ_199.inp` · `pinbu_wave_sliceZ_Z1Z2_199.inp` ·
`launch_/run_/status_fpcamp_minfr_Z1Z2_f121_sliceZ_199.{ps1,bat,ps1}` 5종은 **삭제하지 않았고**,
첫 줄(`.bat` 은 `@echo off` 다음 줄)에 다음 한 줄을 넣었다:

```
# SUPERSEDED by T7T8 (alias registry, 부록 P)
```

**199 로 전송하지 않는다.** 편집으로 이들의 sha 는 바뀌었고, 옛 런처의 덱 핀은 더 이상 맞지 않는다 —
그것이 의도된 상태다(옛 트리오는 어떤 경우에도 발사되지 않는다).

## Q.4 아암 B 런처의 아암-A 게이트 경로 수정 (기능적, 필수)

`launch_fpcamp_minfr_T6T4_f121_sliceZ_199.ps1` 은 "아암 A 가 rc 파일을 썼는가" 게이트에서
`fpcamp_minfr_z1z2_f121_slicez_{out.log,rc.txt}` 를 본다. 개명 후 아암 A 는
`…_t7t8_…` 로 쓰므로 그 게이트는 **조용히 발화하지 않았을 것**이다 — 두 arm 이 겹쳐 도는
사고 경로다. 경로를 `fpcamp_minfr_t7t8_f121_slicez_*` 로 고쳤고, 같은 파일의 주석 2곳도
`T7_T8` 로 맞췄다. **아암 B 덱 본문은 손대지 않았으므로 그 sha 핀은 유효하다.**

## Q.5 검증 (HOST_238, StubEvaluator · MASTER 0 · GPU 0)

`~/lpopt_ws/src` 에서 `python -m lpopt optimize --input <deck> --dry-run --budget 8`.
전송본 sha256 = 저작본 sha256 (이진 전송 확인).

**아암 A — 배너**

```
[optimize][DEPRECATED] objective='min_fr_max_cycle' is a RETIRED production mode (flatness-first
  program §10 STOP): it steers the search by F_r. Kept runnable for reproduction / A-B baselines
  only — use objective='flat_power' for production.
[optimize] campaign sliceZ_dry_t7t8 case=T7_T8/feed-121 budget=8 spent=0 dry_run=True
[optimize] wave  0  size=8 spent=8/8 | conv=8 feas=1 on_target=1 | gate=explore+ tau=0.30 | best=628.06
[optimize] SDM/MTC gate configured but not run here (dry-run / no [master].executable); top_k=5 carried for the live run
RESULT: complete — 1 waves, budget 8/8, 1 feasible / 1 on-target;
        best FEASIBLE F_r 1.463 (<= 1.55) @ cyclen 628.1 EFPD          [RC 0]
status.json: {"objective":"min_fr_max_cycle","case":"T7_T8/feed-121","dry_run":true,
              "budget_spent":8,"n_feasible":1}
```

**아암 B — 배너**

```
[optimize][DEPRECATED] objective='min_fr_max_cycle' is a RETIRED production mode ...
[optimize] campaign sliceZ_dry_t6t4 case=T6_T4/feed-121 budget=8 spent=0 dry_run=True
[optimize] wave  0  size=8 spent=8/8 | conv=8 feas=1 on_target=1 | gate=objective- tau=0.30 | best=633.34
RESULT: complete — 1 waves, budget 8/8, 1 feasible / 1 on-target;
        best FEASIBLE F_r 1.434 (<= 1.55) @ cyclen 633.3 EFPD          [RC 0]
status.json: {"objective":"min_fr_max_cycle","case":"T6_T4/feed-121","dry_run":true}
```

- **핵심 확인점**: `case = "T7_T8/feed-121"`. 아암 A 의 dry-run 패턴이 `F:T7:0` / `F:T8:0` 로
  신연료를 배치한다 — alias 해석이 정상이다.
- **dry-run 의 F_r 값은 StubEvaluator 산출로 물리적 의미가 없다.** §7.5 의 어떤 마크 판정에도
  쓰지 않는다. 이 검증이 확인한 것은 **문법·설정·objective·케이스 라우팅**뿐이며,
  실 자산 해석은 199 런처의 LIBRARY/CORES/BASES 게이트가 본다.
- 임시 런 디렉터리 `/tmp/sliceZ_dry_{t7t8,t6t4}` 는 삭제했다.

> **★ 새 관측 (등록).** 양 arm 이 `objective='min_fr_max_cycle'` 에 **DEPRECATED / RETIRED
> production mode** 배너를 찍는다. 실행은 정상 완료하며 런북 §5.3 의 금지 배너
> (`min_fxy objective`, `[optimize][F_xy PROXY]`)가 아니다. 사용자 목표 진술(2026-09-03)이
> `min F_r` 을 지정했으므로 **objective 는 변경하지 않는다.** 다만 사전등록이 고정한 objective 가
> 코드에서 RETIRED 로 표시되어 있다는 사실은 **결과 문서에 기재한다** (§10.2 항목 추가).

## Q.6 미처분으로 남는 것 (발사 전 선행조건)

1. **RESTAMP — 양 런처의 스토어 핀.** `$wantStore = 16E311AF…917BA` / `$wantStoreLen = 22810322`
   는 **슬라이스 이전** 값의 플레이스홀더로 그대로 두었다 (S6 부트스트랩 + S8 전이 머지가 반드시
   바꾼다). 개명 편집은 이 두 줄을 건드리지 않았다. 재해시 명령은 런처 헤더와 런북 §5.1 에 그대로:
   `Get-FileHash -Algorithm SHA256 C:\Users\USER\lpopt_work\kit_frontier\data\store\records.parquet`
   / `(Get-Item …\records.parquet).Length`. 아암 B 의 핀은 **아암 A 머지 후** 값이다.
2. **아암 B 덱 헤더 주석의 `Z1_Z2` 상호참조 2곳** (12·17행) 은 미수정 — 본문을 고치면 sha 핀이
   깨진다. 주석만의 결함이며, 고치려면 덱·핀·런북 표를 한 편집에서 함께 갱신해야 한다.
3. **`produce_sliceZ_bootstrap_199.inp:145` 의 `[case] pair = "Z1_Z2"`** 도 미수정.
   이 경로에서 INERT 이고(`--pair` CLI 가 정본), **S6 부트스트랩이 이 덱(sha `7FB7594B…265F`)으로
   199 에서 진행 중**이므로 종료 전에는 편집하지 않는다.
4. S8 전이 스윕 · 영수증 `fr_transfer_T7_T8_merged.json` 미실행 (아암 A 런처의 TRANSFER 게이트가
   이 파일의 존재를 요구한다).
5. 부록 L 의 발사 스탬프에 RESTAMP 값을 기록하는 일은 재해시와 **같은 편집**에서 한다.

## Q.7 이 부록이 실행하지 않은 것

MASTER 0 · DeCART 0 · 199 접촉 0 (읽기도 쓰기도 없음) · 181 접촉 0 · 로컬 연산 0
(Read/Grep/Edit/Write/`Get-FileHash` 만). 238 에서는 덱 3종 전송 + `--dry-run --budget 8`
2회(StubEvaluator, GPU 0)만 했고 임시 런 디렉터리 `/tmp/sliceZ_dry_{t7t8,t6t4}` 는 삭제했다.
덱 3종 사본은 `~/lpopt_ws/src` 에서 `~/lpopt_ws/scratch/slice_Z_rename/` 으로 옮겨 `src` 를
원상 복구했다 (읽기 전용 스크래치, 무해). `data/design/package` · 스토어 parquet ·
모델 디렉터리 무변경. **옛 Z1Z2 파일 삭제 0.**

---

# 부록 T — **S6-B 재발사 스탬프** (2026-09-03, append-only)

**결론 선언: S6 는 원인 수정(cy1 cap) + 방어 장치(NaN 워치독) 두 가지를 적용해 재발사되었고,
등록 스모크 T3_T4 가 11 주기 · 215 s 로 수렴(G-H5c PASS)했다. 패키지 롤백은 하지 않았다.**
진단서: `data/reports/sliceZ_s6_diagnosis_20260903.md` (S5 재빌드 무죄, 원인 = `cy1_cap_efpd` UNSET).

## T.1 코드 변경 (3 파일 + 시험 2 파일)

| 파일 | sha256 | 내용 |
|---|---|---|
| `lpopt/design/bootstrap.py` | `e882cebf39fbcc7dc477427c7bdc73a74dc92bf38114ff3bfae46e2d689a70a1` | **워치독 코드 sha.** `MasterDivergenceError`(= `MasterRunError`+`BootstrapError`), `_nan_watch` 스레드(`run_cycle1`), `_BootstrapMasterRunner`(= `search.verify.WatchdogMasterRunner` + 발산 라벨링), `DEFAULT_BOOTSTRAP_TIMEOUT_S = 900.0`, `run_cycle1`/`make_band_restart` 기본 timeout 3600 → 900 |
| `lpopt/config.py` | `9b0e3f69d627af96d24533ffa30f0dbad8e4059d81065244911e0d560d28a28c` | `[master].bootstrap_timeout_s: float = 900.0` 신설 (`[master].timeout = 3600` 은 **불변** — 캠페인/produce 경로 전용) |
| `lpopt/cli.py` | `dc7492ea7616853e50b2e3e8fca52d22972667a0fb77520dbd01b2e008296dd9` | `cmd_design_bootstrap` 이 `cfg.master.bootstrap_timeout_s` 를 `make_band_restart(timeout_s=…)` 로 전달 |
| `tests/test_bootstrap_nan_watchdog.py` | 신규 5 시험 | 아래 T.2 |
| `tests/test_design.py` | 수정 1 줄 | `monkeypatch.setattr(B, "MasterRunner", …)` → `"_BootstrapMasterRunner"` |

동작: `MAS_OUT` 을 10 s 마다 폴링(`NAN_WATCHDOG_POLL_S`)해 꼬리 12 행(`NAN_WATCHDOG_STREAK`)이
모두 비유한(`NaN`)이면 `NONFINITE_FLUX` 센티넬을 남기고 MASTER 를 kill → `MasterDivergenceError`.
작업 디렉터리는 **항상 보존**(`PurgingEquilibriumRunner` 가 `MAS_OUT` 꼬리 + 센티넬만 남기고 trim).
`MasterRunError` 를 상속하므로 기존 실패-경로 회계가 그대로 동작한다.
벤더(`lpopt/vendor/masterrl/*`)는 **한 줄도 고치지 않았다**.

## T.2 시험 (238, CPU only)

`tests/test_bootstrap_nan_watchdog.py` — 가짜 MASTER 2종(무한 `MGOUTER … NaN` / 정상 종료):

1. `test_run_cycle1_kills_a_diverging_master` — 발산 cy1 이 timeout(120 s) 이 아니라 **수 초 내**
   `MasterDivergenceError`, work dir + `MAS_OUT` + `NONFINITE_FLUX` 보존, `MasterRunError` 이기도 함.
2. `test_run_cycle1_healthy_master_is_untouched` — 정상 cy1 은 restart 를 반환하고 센티넬 없음.
3. `test_run_cycle1_hang_still_times_out` — `MAS_OUT` 이 아예 없는 무응답 행(hang)은 `timeout_s` 로 종료.
4. `test_chain_runner_relabels_the_watchdog_kill` — 재장전 체인 러너가 `exited with status`가 아니라
   `MasterDivergenceError` 로 보고.
5. `test_bootstrap_timeout_is_configurable_and_defaults_to_900` — `[master].bootstrap_timeout_s`
   기본 900, 덱에서 로드, `[master].timeout = 3600` 불변.

결과: **`5 passed`**. 회귀: `test_design.py` · `test_config.py` · `test_verify_stub.py` ·
`test_curriculum.py` + 신규 = **138 passed, 8 skipped**.

## T.3 덱 — `produce_sliceZ_bootstrap_199.inp`

| 항목 | 값 |
|---|---|
| sha256 (구, 부록 Q 가 핀한 값) | `7FB7594BC934AD20DE268439DBD9D7004EB94D0A894068268655BF4EE599265F` |
| **sha256 (신, 이 스탬프가 핀하는 값)** | **`30c9afab4b039f1492e28d032d2bf2c6ce264c4b9f79df7c0aa5a5ad60f7c34e`** |
| 199 사본 검증 | 동일 (전송 후 `Get-FileHash` 일치) |
| 백업 | `produce_sliceZ_bootstrap_199.inp.bak_20260903c` (199) |

편집 3건 — **본문 논리 무변경, 주석 + `[master]` 키 1개**:

1. D-4 블록 헤더에 `*** SUPERSEDED 2026-09-03 ***` 표시. **원문은 삭제하지 않고 그대로 둔다**
   (추가 전용 정정 — 오류가 감사 가능하도록).
2. 신설 `D-4 CORRECTED` 블록: cap 은 **프로토콜 변경이 아니라 기존 프로토콜**이라는 증거
   (T5_T6 f81 493.60 / f101 579.40 이 B1 = 981.0 으로 식을 만족, T3_T4 597.70 이 스냅샷 재시작
   이름과 일치), 실패 메커니즘(BOC-2 외부반복 4, CBC 1288.91 ppm, NaN 약 57,000 회), 격리시험
   (cap 597.70 → cy1 17.3 s / cy02 19.6 s, NaN 0), 그리고 **전 페어 일괄 적용 규약**.
3. `[master] bootstrap_timeout_s = 900` 추가 (`timeout = 3600` 은 캠페인 전용이라고 주석 명시).

`[design].cy1_cap_efpd` 는 **여전히 UNSET** 이다 — 스칼라 1개로는 페어별 B1 을 덮을 수 없기 때문이며,
드라이버가 페어마다 `--cy1-cap-efpd` 를 준다. `[case] pair = "Z1_Z2"`(INERT) 는 미수정 —
부록 Q.6-3 의 처분 그대로.

## T.4 cap 산출 — `cap = floor(2*B1/(241/feed + 1), 2)` · feed 121 → 제수 2.9917355

페어마다 **uncapped cy1 을 1회(~25 s) 돌려 B1 을 실측**한 뒤 식을 적용한다(드라이버 `measure_b1`,
`build_cycle1_deck` + `run_cycle1` 재사용 — 덱 지식 중복 없음). 실측 결과:

| 페어 | B1 [EFPD] (실측) | cap [EFPD] | 프로브 wall | 프로브 디렉터리 |
|---|---|---|---|---|
| T3_T4 | **894.09** | **597.70** | 24.6 s | `s6_b1probe\T3_T4_20260903T075318Z` |
| T7_T8 | **944.79** | **631.59** | ~25 s | `s6_b1probe\T7_T8_20260903T075318Z` |
| T6_T4 | (드라이버가 실행 중 측정) | 〃 | — | `s6_b1probe\T6_T4_20260903T075318Z` |
| T5_T6 | (드라이버가 실행 중 측정) | 〃 | — | `s6_b1probe\T5_T6_20260903T075318Z` |

T3_T4 의 894.09 는 진단서의 실패 런 값과 **소수점까지 일치**(결정론 확인), 597.70 은 진단서가
검증한 값과 일치하며 capped cy1 의 산출 재시작은 정확히 `MAS_RST.APRQ_01_0597.70` 이다.

## T.5 발사 — S6-B (199, 분리 실행)

| 항목 | 값 |
|---|---|
| 드라이버 | `s6_bootstrap.py` sha256 `bf59e4ddbad3e15b31b64a7f4a94d321f7dfab8d314d2c4946d2adbe9dbf4065` (백업 `…py.bak_20260903c`) |
| 발사 방식 | `Win32_Process.Create` (ssh 종료와 무관하게 생존) |
| 발사 시각 | 2026-09-03T07:53:17Z |
| PID | 런처 `cmd.exe` **9888** → 드라이버 python **5664**(venv 스텁) → **8736**(실행 프로세스) → MASTER 자식 |
| 로그 | `s6_bootstrap.log` · `s6_bootstrap.rc` · 페어별 `s6_<PAIR>.log/.rc` · `s6_result.json` |
| 순서/정책 | T3_T4 → T7_T8 → T6_T4 → T5_T6, **순차**, 실패 시 잔여 페어 **중단** (구 드라이버와 동일) |
| 보존 | S6-A 산출물 전부 `*.bak_20260903c`, 실패 증거 44,390,573 B 를 `scratch_s6diag\bootstrap_work_T3_T4_FAILED_20260903A` 로 **복사**(삭제 0) |

**첫 주기 증거 (T3_T4)** — 재발사가 진단을 확증한다:

- B1 프로브(uncapped) 24.6 s → `MAS_RST.APRQ_01_0894.09`
- capped cy1 (597.70) → `bootstrap_work\T3_T4\cy1\MAS_RST.APRQ_01_0597.70`, 24 s, **NaN 0**
- cy02 이후 체인 정상: 주기당 ~19 s, `MAS_OUT` 의 `NaN` 행 **0**
- **수렴: 11 주기 · 215 s · cyclen 599.391 EFPD · F_r 3.0657 · CBC_max 1347.03 ·
  max_pin_burnup 66.818 · discharge_bu 45.37 → `bases\T3_T4\MAS_RST.APRQ_11_0599.39`**
- **G-H5c PASS** (11 ≤ max_cycles 16, consecutive 2). G-H6(T7/T8 에코)은 T3_T4 노심에
  T7/T8 이 없으므로 false 가 정상 — 판정은 T7_T8 페어에서 읽는다.
- 대조: S6-A 는 같은 페어에서 3,625 s 를 태우고 rc 1 (`MASTER timed out after 3600 seconds`).

**두 번째 페어 (T7_T8, 아암 A 의 새 셀)** — B1 944.79 → cap 631.59, **수렴 11 주기 · 224.3 s ·
cyclen 630.693 EFPD · F_r 3.4464**, G-H5c PASS. T6_T4 · T5_T6 는 발사 시점 기준 진행 중이며
같은 규약(측정 → cap → 부트스트랩)으로 이어진다. 최종 판정은 `s6_result.json` 의 `status` 다.

> **등록(미결).** 드라이버의 `G-H6_echo` 는 CLI 요약 로그에서 `FA_T7`/`FA_T8` 를 찾는데, 그 로그에는
> COMP 에코가 없어 항상 false 가 된다. G-H6 의 실증은 진단서 S1.4 의 **MAS_OUT COMP 목록**
> (`38 FA_T7`, `39 FA_T8` 정상 출력, PASS)이며, 드라이버 필드는 판정에 쓰지 않는다.

## T.6 감시 (읽기 전용)

`monitor_s6.ps1` (899 B, 미변경) 그대로 유효하다:

```
ssh USER@HOST_199 "powershell -NoProfile -File C:\Users\USER\lpopt_work\kit_frontier\monitor_s6.ps1"
```

`s6_result.json` 의 페어별 항목에 이번 런부터 `b1_efpd` · `cy1_cap_efpd` · `b1_probe_dir` 가
추가로 기록되므로 cap 감사도 이 파일 하나로 끝난다.

## T.7 이 스탬프가 하지 않은 것

패키지 롤백 0 (`sliceZ_20260903T055605Z` 스냅샷은 보관만) · 199 삭제 0 · 벤더 코드 수정 0 ·
`[master].timeout` 등 캠페인 타임아웃 변경 0 · 로컬 연산 0 (시험은 238, MASTER 는 199) ·
S7/S8 착수 0.

## T.8 최종 결과 (S6-B 종료, 2026-09-03T08:02:23Z) — *T.4/T.5 에 대한 추가 전용 갱신*

| 페어 | B1 [EFPD] | cap [EFPD] | 결과 | 주기 | cyclen | F_r | CBC_max | max_pin_bu | wall |
|---|---|---|---|---|---|---|---|---|---|
| T3_T4 | 894.09 | **597.70** | **OK / 수렴** | 11 | 599.391 | 3.0657 | 1347.03 | 66.818 | 215.7 s |
| T7_T8 | 944.79 | **631.59** | **OK / 수렴** | 11 | 630.693 | 3.4464 | 1484.02 | 71.777 | 224.3 s |
| T6_T4 | 937.69 | **626.85** | **FAIL — cy02 발산** | — | — | — | — | — | 29.2 s |
| T5_T6 | — | — | 미실행 (정책상 중단) | — | — | — | — | — | — |

`s6_result.json`: `overall=FAIL`, `aborted_after=T6_T4`, `not_run=[T5_T6]`. rc: T3_T4 0 · T7_T8 0 · T6_T4 1.
`bases/` 신규 재시작 2건: `T3_T4\MAS_RST.APRQ_11_0599.39`, `T7_T8\MAS_RST.APRQ_11_0630.69`.

**T6_T4 는 cap 을 정확히 적용했는데도 발산했다 — 진단서 P2 가 예고한 그대로다**
("cap 은 필요조건이지 충분조건이 아니다"; 선례: T5_T6_f101 cy3 @ 60 EFPD).

- capped cy1 은 **정상**: `bootstrap_work\T6_T4\cy1\MAS_RST.APRQ_01_0626.85`, 18 s, NaN 0.
- cy02 는 **BOC 붕산 탐색이 아니라 연소 중반**에 깨졌다: MAS_OUT 4,882행,
  연소 단계 **3.514**(≈ 90 EFPD 지점), 외부반복 25 회까지 `keff 1.000022 / err 2.8E-04` 로
  정상 수렴하다가 26 회째에 `NaN`, 이후 `MGOUTER 11 20 NaN` 반복.
  (T3_T4 의 S6-A 실패는 BOC-2 반복 4 · 1288.91 ppm 이었으므로 **다른 지점의 같은 고장 모드**다.)
- **워치독 실증**: NaN 150행 시점에 kill → `MasterDivergenceError`, wall **29.2 s**
  (S6-A 의 동일 고장은 3,625 s). 작업 디렉터리 보존 + `NONFINITE_FLUX` 센티넬 기록:
  `bootstrap_work\T6_T4\master\bootstrap-00f74d162e-ibr8lhso`. **이 한 건이 워치독의 실전 검증이다.**
- 드라이버는 등록된 정책대로 잔여 페어(T5_T6)를 **실행하지 않고 중단**했다.

**미결(오너 결정 필요)**: T6_T4 는 아암 B 의 대조 셀이므로 슬라이스에 필수다. 남은 선택지는
(a) 다른 seed 의 초기 셔플맵으로 재시도(`[design].seed` — 발산은 (패턴, 재시작) 쌍의 물리적 성질),
(b) cap 을 평형 추정치로 미세 조정, (c) MASTER 수치 옵션(`epsflx`/`ncycle`) 검토.
**이 스탬프는 셋 중 어느 것도 실행하지 않았다.** T5_T6 도 미실행 상태다.


---

# 부록 U — **S6-C: T6_T4 seed 재시도 + T5_T6 부트스트랩** (2026-09-03, append-only)

**결론 선언: 부록 T.8 이 오너 결정으로 남긴 선택지 (a)(다른 seed 의 초기 셔플맵)를 집행했고,
T6_T4 는 첫 번째 seed 변형에서 11 주기로 수렴(G-H5c PASS)했다. 이어 T5_T6 도 원본 덱·원본
seed 로 10 주기 수렴(G-H5c PASS)했다. 슬라이스가 요구하는 4 페어 전부가 재빌드된 39-type
paramA 라이브러리 위에서 재부트스트랩을 마쳤다 — `s6_result.json` `status = OK`.**
선택지 (b)(cap ×0.95)와 (c)(MASTER 수치 옵션)는 **실행하지 않았다** — 필요 없었다.

## U.0 S6-B 발산 증거 (재확인, 읽기 전용 — 부록 T.8 의 실측 보강)

보존 사본: `s6_evidence\T6_T4_S6B_fail_20260903\` (13 파일 / 38,689,346 B, **복사**, 삭제 0).
원본 작업 디렉터리 `data\design\package\bootstrap_work\T6_T4\master\bootstrap-00f74d162e-ibr8lhso`
도 그대로 남아 있다(이후 재실행이 덮어쓰므로 이 사본이 정본 증거다).

| 항목 | 실측값 |
|---|---|
| 발산 주기 | **cy02** (`MAS_INP` 2행 `1 stead` = 재시작 1개 + `%LPD_SHF` 존재 → 재장전 덱) |
| 직전 정상 cy1 | `bootstrap_work\T6_T4\cy1\MAS_RST.APRQ_01_0626.85` (cap 626.85 정확 적용, NaN 0) |
| MAS_OUT 총 행수 | 5,031 |
| 마지막으로 **완주한** 연소 편집 | `$B1D_5 / $NUCL_5 / $XESM1D_5` = **60.000 DAY = 60.000 EFPD**, TOTAL BURNUP **13.9325 MWD/KGHM**, MAX PIN (3-D) BURNUP 45.8034 @ (K,13,17,6,6) |
| 발산 지점 | 그 다음 연소 단계(`3.514`), **외부반복 26** — MAS_OUT **4,881행(0-based) / 4,882행** |
| 발산 직전 잔차 궤적 | 반복 20 에서 **1.39866E-04 로 바닥**을 친 뒤 **단조 증가** — 21 `1.43748E-04` · 22 `1.54819E-04` · 23 `1.75073E-04` · 24 `2.09435E-04` · 25 `2.80441E-04` → 26 `NaN` |
| keff / CBC 궤적 | 25 회까지 `keff 1.000022 / CBC 1632.17` 로 정상, 26 회 `keff NaN / CBC 1632.28` |
| 이후 | `MGOUTER 11 20 NaN` 반복(최대 반복수 100 까지) |
| 워치독 | `NONFINITE_FLUX` 센티넬 = `non_finite_flux`, `MASTER.stderr` 0 B, kill 후 wall **29.2 s** |

즉 **수렴 실패가 아니라 수렴하던 반복이 되튀어(divergence-after-convergence) 폭발한 것**이며,
BOC 붕산 탐색(S6-A / T3_T4 의 고장 지점)이 아니라 **연소 중반**이다. cap 은 정확히 적용되어
있었으므로 부록 T.8 의 판정("cap 은 필요조건이지 충분조건이 아니다")이 그대로 확인된다.
이 고장은 (패턴, 재시작) 쌍의 성질이므로 **초기 셔플맵을 바꾸는 것이 최소 개입**이다.

## U.1 드라이버 · 덱

| 항목 | 값 |
|---|---|
| 드라이버 | `s6c_bootstrap.py` sha256 `C6ABC61DE0B00CDB6E702EF8F3A8838D9C38C0CEA20D8FE85301C38E53110612` (신규 파일; `s6_bootstrap.py` **미수정**) |
| lpopt 코드 변경 | **0** (부록 T.1 의 sha 그대로) |
| 원본 덱 | `produce_sliceZ_bootstrap_199.inp` sha256 `30C9AFAB4B039F1492E28D032D2BF2C6CE264C4B9F79DF7C0AA5A5AD60F7C34E` — **부록 T.3 이 핀한 값과 동일, 무수정 확인** |
| seed 변형 덱 | `produce_sliceZ_bootstrap_199_s6c.inp` — 원본의 `[design] seed` **한 줄만** 치환(정규식 `^(seed\s*=\s*)(\d+)$`, 매치 수 1 을 사전 검증) |
| 발사 방식 | `Win32_Process.Create` (`s6c_launch.bat`), PID 18752, ssh 종료와 무관 |
| 발사/종료 | 2026-09-03T08:11:29Z → **08:19:03Z** (총 7 분 34 초) |
| 산출 | `s6c_result.json`(정본) · `s6c_bootstrap.log`/`.rc` · `s6c_T6_T4_a1.log` · `s6c_T5_T6.log` |
| 병합 | `s6_result.json` 에 두 페어 항목을 병합, 병합 전 원본을 `s6_result.json.bak_pre_s6c` 로 보존 |
| 정책 | cap 규칙·게이트·순차 실행 전부 S6-B 와 **동일** — 바뀐 변수는 T6_T4 의 `seed` 하나뿐 |

## U.2 시도 표 (T6_T4)

계획: 선택지 (a) seed 3 회 → 전부 실패 시 선택지 (b) cap ×0.95 1 회. **1 회차에서 성공했으므로
2~4 회차는 실행되지 않았다.**

| # | seed | cap 배율 | cap [EFPD] | 덱 sha256 | rc | wall | 주기 | G-H5c | 결과 |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **20260903** | 1.00 | **626.85** | `15d8ff61acc45f4f9ed6e8f6c4f98391007b6e6d99c297d123017a4ab4a32ce8` | 0 | **212.2 s** | **11** | **PASS** | **OK / 수렴** |
| 2 | 7 | 1.00 | 626.85 | — | — | — | — | — | 미실행 (1 회차 성공) |
| 3 | 99991 | 1.00 | 626.85 | — | — | — | — | — | 미실행 |
| 4 (옵션 b) | 1439 (원본) | 0.95 | 595.50 | — | — | — | — | — | 미실행 |

B1 재측정(uncapped 프로브 24.2 s): **937.69 EFPD** — S6-B 실측과 **소수점까지 일치**
(`s6_b1probe\T6_T4_20260903T081129Z`, 재시작 `MAS_RST.APRQ_01_0937.69`). 따라서 cap 도
동일한 **626.85**. **cap 은 그대로 두고 seed 만 1439 → 20260903 으로 바꾼 것이 유일한 차이**이며,
그 하나로 발산이 사라졌다 — 부록 T.8 의 (a) 가설(발산 = (패턴, 재시작) 쌍의 성질)에 대한 직접 증거다.

## U.3 T5_T6 (원본 덱 · 원본 seed 1439)

seed 변형은 T6_T4 에만 필요했으므로 T5_T6 은 **원본 덱을 그대로** 썼다(프로토콜 혼합 없음).

| 항목 | 값 |
|---|---|
| B1 (실측, 프로브 24.1 s) | **981.02 EFPD** → `MAS_RST.APRQ_01_0981.02` |
| cap | `floor(2×981.02 / 2.9917355, 2)` = **655.82 EFPD** |
| 결과 | **OK / 수렴, 10 주기, 192.9 s, G-H5c PASS** |

> B1 = 981.02 는 덱 헤더 `D-4 CORRECTED` 블록이 T5_T6 의 옛 재시작(f81 → 493.60, f101 → 579.40)
> 으로부터 **역산해 주장한 B1 = 981.0 과 소수 둘째 자리까지 일치**한다. cap 이 기존 프로토콜이라는
> D-4 정정의 근거가 독립적으로 재현되었다.

## U.4 최종 결과 — 4 페어 전부 (S6-B + S6-C 통합)

| 페어 | 런 | seed | B1 [EFPD] | cap [EFPD] | 주기 | cyclen | F_r | CBC_max | max_pin_bu | discharge_bu | wall | G-H5c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T3_T4 | S6-B | 1439 | 894.09 | 597.70 | 11 | 599.391 | 3.0657 | 1347.03 | 66.818 | 45.372 | 215.7 s | **PASS** |
| T7_T8 | S6-B | 1439 | 944.79 | 631.59 | 11 | 630.693 | 3.4464 | 1484.02 | 71.777 | 47.742 | 224.3 s | **PASS** |
| **T6_T4** | **S6-C** | **20260903** | 937.69 | 626.85 | **11** | **619.133** | **2.4101** | **1455.06** | **70.792** | **46.867** | 212.2 s | **PASS** |
| **T5_T6** | **S6-C** | **1439** | 981.02 | 655.82 | **10** | **650.033** | **3.6576** | **1866.15** | **68.621** | **49.206** | 192.9 s | **PASS** |

`s6_result.json`: **`status = OK`**, `run = "S6-B + S6-C"`, `aborted_after`/`not_run` **제거됨**,
4 페어 모두 `status = OK` · `G-H5c = PASS`.

신규 `bases/` 재시작 4건 (전부 재빌드 39-type 라이브러리 기준):

| 페어 | 신규 재시작 | 기록 시각 (KST) |
|---|---|---|
| T3_T4 | `bases\T3_T4\MAS_RST.APRQ_11_0599.39` | 2026-09-03 16:57 |
| T7_T8 | `bases\T7_T8\MAS_RST.APRQ_11_0630.69` | 2026-09-03 17:01 |
| T6_T4 | `bases\T6_T4\MAS_RST.APRQ_11_0619.13` | 2026-09-03 17:15 |
| T5_T6 | `bases\T5_T6\MAS_RST.APRQ_10_0650.03` | 2026-09-03 17:19 |

G-H6(T7/T8 에코)에 대한 처분은 부록 T.5 의 등록 그대로다 — 드라이버의 `G-H6_echo` 필드는
CLI 요약 로그를 보므로 항상 false 이며 판정에 쓰지 않는다.

## U.5 ★ S7 차단 사유 (신규 등록) — `bases/` 3 폴더에 **옛 37-set 재시작이 함께 남아 있고,
##     캠페인 해석기가 그중 2 곳에서 옛 파일을 고른다**

S6-B/S6-C 는 신규 재시작을 **추가**했을 뿐 옛 파일을 지우지 않았다(정책상 삭제 0). 그런데
캠페인 경로의 재시작 해석기 `lpopt/search/assets.py:_only_restart` 는

```python
candidates = sorted(p for p in directory.glob("MAS_RST.*") if p.is_file())
for candidate in candidates:
    if _is_readable(candidate):
        return candidate
```

즉 **이름 정렬 후 첫 번째 읽을 수 있는 파일**을 돌려준다. 실측 폴더 내용에 이 규칙을 적용하면:

| 폴더 | 파일 (정렬 순) | `_only_restart` 선택 | 판정 |
|---|---|---|---|
| T3_T4 | `…APRQ_11_0578.27`(구, 08-11) → `…APRQ_11_0599.39`(신) | **구 0578.27** | ✗ **STALE 선택** |
| T7_T8 | `…APRQ_11_0630.69`(신) 뿐 | 신 | ✓ |
| T6_T4 | `…APRQ_10_0615.11`(구, 08-11) → `…APRQ_11_0619.13`(신) | **구 0615.11** | ✗ **STALE 선택** |
| T5_T6 | `…APRQ_10_0650.03`(신) → `…APRQ_11_0632.51`(구, 08-11) | 신 0650.03 | ✓ (**우연**히 신규가 앞선다) |

옛 재시작은 37-set 라이브러리 기준이라 재빌드 후 **무효**다(프레레그 S5 / `LIBRARY_BUILD.md` S5,
그리고 `assets.py:204` 가 경고하는 그대로 "비유한 플럭스"로 MASTER 를 몰고 간다). 따라서
**S6 게이트는 전부 PASS 지만, 이 상태로 S7 을 발사하면 T3_T4 · T6_T4 는 무효 재시작을 쓴다.**
T5_T6 이 통과하는 것은 이름 정렬의 우연이므로 방어로 볼 수 없다.

**이 부록은 어떤 파일도 옮기거나 지우지 않았다.** 처분은 오너 결정이며 선택지는
(A) 옛 3 파일을 `bases_stale_pre39\<pair>\` 같은 격리 폴더로 **이동**(삭제 아님),
(B) `_only_restart` 를 최신 mtime 또는 명시 매니페스트 기준으로 바꾸는 코드 변경,
(C) 각 페어를 신규 파일만 남긴 폴더로 재구성. **(A) 가 코드 무변경·감사 가능이라 최소 개입이다.**

## U.6 이 스탬프가 하지 않은 것

199 삭제 0 · `lpopt` 코드 수정 0 · 원본 덱 `produce_sliceZ_bootstrap_199.inp` 수정 0 ·
`s6_bootstrap.py` 수정 0 · 벤더 수정 0 · 로컬 연산 0 · `bases/` 파일 이동/삭제 0 ·
선택지 (b)/(c) 실행 0 · S7/S8 착수 0.
