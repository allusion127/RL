# `min_fxy` 캠페인 ROUND 1 — `T6_T4` / feed 121 — RESULTS

**사전등록** `data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md` (구속력 있음, §9.1 STAMP 2건 포함)
**Run dir** `runs/fpcamp_minfxy_t6t4_f121_r1` · **Deck** `fpcamp_minfxy_T6T4_f121_r1_199.inp` (`BEF3519E…`)
**Champion served** `data/models/s1j` (11번째, `fxy_head.serve_sigma = "barred"`) · **objective** `min_fxy` · `policy_prior = "off"`
**Box** HOST_199 (call 1–88) → **HOST_181** (call 89–100, canonical 완료 `rc=0` 09:15) · **Log** `fpcamp_minfxy_t6t4_f121_r1_out.log` (199 leg)
**사전등록된 대로 채점했다.** 이 판독을 위해 deck·harness·분석 코드를 고치지 않았다. 원격 접속 없음, MASTER 추가 호출 없음.

> **프로그램 최초의 `min_fxy` 라운드다.** 아래 모든 수치는 실측이며, 출처는
> 2026-08-30 시점의 `data/store/records.parquet` (**75,893행**, sha256 `255F0E41…`)와
> run dir의 `labels.jsonl` / `waves/*/selection.json` / `state.json` 이다.
> **joint-clean** 은 사전등록 §0의 정의를 그대로 쓴다 — `converged & valid` 이면서
> `f_r ≤ 1.55 & cbc_max ≤ 1600 & f_q ≤ 2.41 & |ao_abs| ≤ 0.30`, `f_xy` 라벨 보유.

---

## 0. 마크 — 사전등록 §2.2 / §2.3 대조

| mark | 등록된 요구조건 | 측정 | 판정 |
|---|---|---|---|
| **PRIMARY (search)** | 측정 `F_xy ≤ 1.5441` + joint-clean 전 축 + 예측 pin BU ≤ 78 | **`F_xy = 1.5322`** / `F_r` 1.4857 · CBC 1337.38 · `F_q` 1.8530 · \|AO\| 0.0244 · 예측 pin 64.28 | **✅ 달성** (선 대비 −0.0119, incumbent 1.5491 대비 **−0.0169**) |
| **STRETCH** | 측정 `F_xy ≤ 1.5295` (프로그램 joint-clean 최소, `E1_E2/f121`) | 1.5322 (**+0.0027**) | ❌ 미달 |
| SECONDARY | 측정 `F_xy < 1.5491` (마진 없이 incumbent 이김) | 1.5322 / 1.5337 / 1.5490 — **3건** | (달성, headline 금지 조항은 PRIMARY 달성으로 무효) |
| **NULL** | 100 call 안에 `F_xy ≤ 1.5441` 없음 | call **57**에서 달성 | **해당 없음** — §4.1·§4.2·§4.3 귀속 판정은 전부 moot |
| **PARITY — cyclen** | \|Δ\| ≤ 1 EFPD, gated | \|Δ\| = **0.349** | ✅ PASS |
| **PARITY — cbc_max** | \|Δ\| ≤ 1 ppm, gated | \|Δ\| = **3.572** | ❌ **FAIL** |
| **PARITY — f_xy (head arm)** | \|Δ\| ≤ 0.01, **head arm gated** | \|Δ\| = **0.0543** | ❌ **FAIL** → §8-C 표가 r2의 gating 항목이 된다 (§2.3 후단 발동) |
| PARITY — f_r | 보고만 | \|Δ\| = 0.0034 | 보고 |
| **PIN / PRIMARY (delivery)** | phase-2 측정 pin ≤ 80 | 측정 pin BU **0/95**, `deliverable` **0/95**, `unknown_axes = ("max_pin_burnup",)` 95/95 | **미판정 — 사전등록된 구조**(§5.4). phase-2 `pinbu_wave` 필요 |
| **arm 선언** | `selection.json → fxy_source` 인용 | **13/13 wave 전부 `"head"`**, `policy_mode = "off"` | arm B (§5.2). §5.1의 "주장 불가" 문구는 적용되지 않는다 |

**한 줄 요약.** `T6_T4/f121`에서 100 call로 **PRIMARY를 달성했고 STRETCH는 0.0027 차로 놓쳤다.**
셀 incumbent는 `1.5491 → 1.5322` 로 내려갔고, 이 core는 프로그램 joint-clean `F_xy` 순위 **3위**다.
그러나 **head arm의 `f_xy` parity가 깨졌고(0.0543)**, 아래 §6이 보이듯 **head의 within-campaign
랭킹 능력은 proxy보다 낫지 않다** — 이득의 출처는 acquisition의 F_xy 순위가 아니라 elite pool과
local search 쪽이다. 그리고 §9에 등록되지 않은 편차 6건이 있으며, 그중 **D3(σ bar 유실)** 과
**D6(host 이관)** 은 r2 이전에 처리되어야 한다.

---

## 1. 실행 요약

| 항목 | 값 |
|---|---|
| budget | 100 / 100 (12 wave × 8 + reserve wave 4) |
| wave 수 | 13 (`wave_00` … `wave_12`) |
| converged / error | **95 / 5** (error 전부 `non_finite_flux`, wave 2·6·7·10·11 각 1건) |
| `f_xy` 라벨 | **95 / 95** 수렴행 (harvest 결함 0건 — §5) |
| search-feasible (`is_feasible_search`) | **61** (`status.json → n_feasible`) |
| joint-clean (licensing 4축) | **65** |
| **deliverable** (`is_deliverable`) | **0** — 사전등록된 기대값 |
| slot 배분 | exploit 64 / explore 24 / control 12 |
| origin | local 64 / elite 30 / guided 4 / random 1 / heuristic 1 |
| gate 수락 | wave 4·5·8·9·12 (`objective+`), 나머지 8개 wave `objective−` |
| `ood_flag` | 100건 전부 `false` |
| best_overall | `bf3a70b2…`, `F_xy` 1.5322, cyclen 622.101, wave 7 / call 57 |

`converged` 95행의 실측 위반 분해 (사후 라벨 기준):

| 위반 조합 | 행수 |
|---|---:|
| clean (`F_xy ≤ 1.65` 포함 전 축 통과) | **61** |
| `F_xy` 만 위반 | 4 |
| `F_xy` + `F_r` | 27 |
| `F_r` 만 위반 | 2 |
| `F_xy` + `F_r` + `F_q` | 1 |

`CBC > 1600` **0건**, `|AO| > 0.30` **0건**. `61 = search-feasible` 와 정확히 일치한다 —
`is_feasible_search` 가 licensing 4축 + `F_xy ≤ 1.65` 를 보는 반면 joint-clean 정의는
`F_xy` gate를 포함하지 않으므로 **65 − 61 = 4** 가 정확히 "`F_xy`만 위반" 행이다.

---

## 2. PRIMARY core — 최적점 상세

| | 값 |
|---|---|
| `record_id` | `bf3a70b20e508c7c01d15fd62bccc653376f65072a7507a9b3fdc755898ed982` |
| call / wave / slot / origin | **57** / 7 / `exploit` / `local` |
| `F_xy` (MASTER `FXYP`) | **1.5322** (limit 1.65, margin 0.1178) |
| `F_xya` | 1.3803 |
| `F_r` | 1.4857 (limit 1.55, margin 0.0643) |
| `CBC_max` / `F_q` / \|AO\| | 1337.38 ppm / 1.8530 / 0.0244 |
| `cyclen` | 622.101 EFPD (band 611.3–631.3 내부, incumbent 621.28 대비 **+0.82**) |
| `node_peak` / `map_cov` | 1.3376 / 0.2547 |
| `max_assembly_burnup` (측정) | 55.782 |
| `max_pin_burnup` | **미측정** → `deliverable = false`, `unknown_axes = ("max_pin_burnup",)` |
| `conformal_unfit_axes` | `f_r, cbc_max, f_q, ao_abs` |

### 2.1 PARITY — 최적점 예측 vs MASTER (사전등록 §2.3)

surrogate 5축은 **`waves/wave_07/selection.json → pred_mean` 을 그대로** 인용한다(선택 시점의
로그값). `f_xy` 와 `node_peak` 은 selection.json에 기록되지 않으므로(→ §9 D-LOG),
**그 wave가 실제로 서빙한 checkpoint** 를 재적재해 재산출했다. wave별 checkpoint 귀속은
§6.0에서 로그된 `pred_mean` 과 대조해 기계적으로 확인했다.

| 축 | 예측 | 측정 | Δ (pred − meas) | 기준 | 판정 |
|---|---:|---:|---:|---|---|
| `cyclen` | 621.7525 | 622.101 | **−0.3485** | \|Δ\| ≤ 1 EFPD | ✅ **PASS** |
| `cbc_max` | 1333.808 | 1337.380 | **−3.572** | \|Δ\| ≤ 1 ppm | ❌ **FAIL** |
| **`f_xy` (head mean)** | **1.5865** | **1.5322** | **+0.0543** | \|Δ\| ≤ 0.01, head arm gated | ❌ **FAIL** |
| `f_xy` (proxy, 참고) | 1.5505 | 1.5322 | +0.0183 | 보고만 | 보고 |
| `f_r` | 1.4823 | 1.4857 | −0.0034 | 보고만 | 보고 |
| `f_q` | 1.8291 | 1.8530 | −0.0239 | — | 보고 |
| \|AO\| | 0.0232 | 0.0244 | −0.0012 | — | 보고 |
| `node_peak` | 1.3543 | 1.3376 | +0.0167 | — | 보고 |
| 예측 `max_pin_burnup` | 64.279 | (미측정) | — | ≤ 78 예측 심사 | ✅ 통과 |

**읽는 법.** parity 실패는 캠페인 실패가 아니라 model calibration 관측치다(§2.3). 다만
등록된 후속 조항이 **두 건** 발동한다:

1. **`f_xy` head parity FAIL** → 사전등록 §2.3 후단에 따라 **§8-C 표(§6)가 r2의 gating 항목**이 된다.
   주의: 이 core에서 head는 **보수적 방향으로 0.054 틀렸다**(실제가 예측보다 좋았다). 즉
   head가 이 core를 "덜 좋다"고 본 채로 뽑았다는 뜻이며, §6의 랭킹 결과와 정합한다.
2. **`cbc_max` parity FAIL (3.57 ppm)** — 두 arm 모두 gated인 프로그램 수용선이다. 이 셀의 CBC는
   1600 gate에서 262 ppm 떨어져 있어 **feasibility에는 영향이 없지만**, 1 ppm 수용선 자체는 깨졌다.
   전 95행 기준 `cbc_max` MAE는 7.65 ppm이므로 이것은 최적점만의 이상이 아니라 **축 전체의 수준**이다(§6.1).

---

## 3. A — frontier-by-call (사전등록 §8-A)

joint-clean 측정 `F_xy` 의 running minimum. 등록된 3개 기준선: incumbent **1.5491**,
PRIMARY **1.5441**, STRETCH **1.5295**.

| wave | calls | wave 내 joint-clean min | frontier (누적) | 기준선 대비 |
|---:|---|---:|---:|---|
| 0 | 1–8 | 1.5876 | 1.5876 | — |
| 1 | 9–16 | 1.5982 | 1.5876 | — |
| 2 | 17–24 | 1.6046 | 1.5876 | — |
| 3 | 25–32 | 1.5950 | 1.5876 | — |
| 4 | 33–40 | 1.5958 | 1.5876 | — |
| 5 | 41–48 | 1.5883 | 1.5876 | — |
| 6 | 49–56 | **1.5731** | **1.5731** | incumbent까지 +0.0240 |
| **7** | **57–64** | **1.5322** | **1.5322** | **incumbent −0.0169 / PRIMARY −0.0119 / STRETCH +0.0027** |
| 8 | 65–72 | 1.5490 | 1.5322 | — |
| 9 | 73–80 | 1.5337 | 1.5322 | — |
| 10 | 81–88 | 1.5777 | 1.5322 | — |
| 11 | 89–96 | 1.5783 | 1.5322 | — |
| 12 | 97–100 | 1.5819 | 1.5322 | — |

frontier가 실제로 움직인 call은 **4회뿐**이다:

| call | wave | slot / origin | `F_xy` | `F_r` | CBC | cyclen |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0 | exploit / local | 1.6276 | 1.5423 | 1319.80 | 621.095 |
| 2 | 0 | exploit / local | 1.5876 | 1.4927 | 1331.31 | 622.697 |
| 49 | 6 | exploit / local | 1.5731 | 1.5238 | 1324.25 | 621.361 |
| **57** | **7** | **exploit / local** | **1.5322** | 1.4857 | 1337.38 | 622.101 |

> **예산은 binding이 아니었다.** 기록은 **57번째 call**에 났고, 이후 **43 call(43%)이 frontier를
> 1 tick도 움직이지 못했다.** wave 8·9는 각각 1.5490 / 1.5337 로 근접했으나 넘지 못했고,
> wave 10–12는 1.577–1.582 대역으로 후퇴했다. 사전등록 §8-A가 요구한 답: **예산 부족이 아니라
> 탐색이 정체했다.** r2를 "같은 셀·더 큰 예산"으로 짜는 것은 이 곡선이 지지하지 않는다.

부수 관측: wave 10 이후의 후퇴 구간은 **HOST_181 이관(call 89~)** 및 **σ bar 유실(§9 D3)** 구간과
겹친다. 다만 wave 10(call 81–88)은 이관 이전이고 이미 후퇴가 시작되었으므로, 정체를 D3에
귀속할 근거는 없다 — **정체는 D3보다 먼저 시작되었다.**

---

## 4. B — `F_xy` 산점 구조와 λ-check (사전등록 §8-B)

### 4.1 `F_xy` – `F_r` / `F_xy` – CBC (converged 95행, `F_xy` 구간별)

| `F_xy` bin | n | mean `F_xy` | mean `F_r` | `F_r` min–max | mean CBC | CBC min–max | mean cyclen |
|---|---:|---:|---:|---|---:|---|---:|
| (1.50, 1.55] | 3 | 1.5383 | 1.4956 | 1.4857–1.5098 | 1339.13 | 1325.25–1354.77 | 622.23 |
| (1.55, 1.60] | 28 | 1.5862 | 1.4949 | 1.4769–1.5238 | 1336.87 | 1313.26–1348.26 | 622.93 |
| (1.60, 1.65] | 32 | 1.6227 | 1.5204 | 1.4932–1.5562 | 1327.95 | 1301.72–1362.09 | 621.80 |
| (1.65, 1.70] | 18 | 1.6744 | 1.5799 | 1.5402–1.6196 | 1322.79 | 1269.04–1407.91 | 620.97 |
| (1.70, 1.75] | 5 | 1.7191 | 1.6327 | 1.5783–1.6793 | 1334.22 | 1313.18–1349.57 | 621.23 |
| (1.75, 1.90] | 7 | 1.8340 | 1.6980 | 1.6620–1.7332 | 1324.79 | 1310.05–1339.98 | 620.59 |

상관 (joint-clean 65행): `corr(F_xy, F_r)` **+0.715** (Spearman +0.720),
`corr(F_xy, CBC)` **−0.435**, `corr(F_xy, cyclen)` **−0.433**, `corr(F_xy, node_peak)` **+0.075**.

> **사전등록 대비 가장 큰 놀라움.** 사전등록 §1.3은 이 셀의 joint-clean `corr(F_xy, F_r)` 을
> **0.282** 로, §4.3은 회귀 기울기를 **0.4811** 로 등록했다(과거 215행). **r1 자신의 joint-clean
> 65행에서는 0.715 / 기울기 1.0351 이다.** 즉 `min_fxy` 탐색이 실제로 방문한 basin은
> `min_fr` 라운드들이 남긴 basin보다 **두 축이 훨씬 강하게 묶여 있는 영역**이다.
> 이것은 §4.3의 "proxy 실명" 논거의 전제를 이 라운드 안에서는 약화시킨다 — 그리고 §6의
> "head가 proxy를 이기지 못했다"와 정확히 같은 방향의 증거다.
> `F_xy` 를 낮추면서 CBC가 오히려 **내려가는**(−0.435) 것도 확인됐다: CBC는 이 셀에서
> `F_xy` 추격의 대가가 아니다.

### 4.2 λ-objective check (`minfxy_lambda = 1000.0`)

- **band-internal λ-optimal 과 `F_xy`-only headline 이 동일한 core다.** cyclen band
  611.3–631.3 내부 joint-clean 65행(= 전체, 아래 참조) 중 `cyclen − 1000·F_xy` 최대는
  `bf3a70b2…` (λ-scalar −910.099), `F_xy` 최소도 같은 core. **λ가 headline을 바꾸지 않았다.**
- **cyclen이 순위를 뒤집은 후보 쌍: 2,080쌍 중 32쌍 (1.5%).** 모두 `|ΔF_xy| < 0.00787` 근방에서만
  발생했다 — band cyclen spread가 7.871 EFPD 이므로 λ=1000 하에서 cyclen이 이길 수 있는 최대
  `F_xy` 차이가 정확히 그 값이다. **설계 의도대로 near-tie만 정렬했다.**
- **slide 감시 (§4.4).** 상위 10 core의 median cyclen = **622.633 EFPD**, incumbent 621.28 대비
  **+1.353**. 등록 판정선은 "5 EFPD 이상 낮으면 slide" 였다. → **slide 없음. λ 재적합 불필요.**
  `min_fr` 라운드의 λ=400 을 물려받지 않은 결정은 사후적으로도 옳았다.
- 수렴 95행 전부가 band 611.3–631.3 내부다 (실측 cyclen 범위 **616.118–627.356**). 사전등록 §6의
  "band는 아무것도 배제하지 않고 판독 라벨로만 기능한다"는 예측대로 성립했다.

**r2용 λ 제안: 변경 없음(1000.0).** slide 증거가 없고, cyclen이 순위를 바꾼 비율이 1.5%에 불과하다.

---

## 5. D·E — deliverable vs search-feasible, `unknown_axes` (사전등록 §8-D/E)

| 카운트 | 값 | 근거 |
|---|---:|---|
| `n_converged` | **95** | `labels.jsonl`, `status = "converged"` |
| `n_feasible` (`is_feasible_search`) | **61** | `status.json → n_feasible`; labels의 `feasible` 합과 일치 |
| `n_deliverable` (`is_deliverable`) | **0** | 측정 `max_pin_burnup` 보유행 0 |

**두 predicate가 갈라지는 이유(한 문장, 재진술).** `is_feasible_search` 는 *측정되지 않은* 축을
통과시켜 최초의 `min_fxy` 캠페인이 굶지 않게 하는 반면, `is_deliverable` 은 gate된 **모든** 축이
**측정**되어 있을 것을 요구하므로(`campaign.py:459-475`), pin BU가 하나도 없는 이 라운드에서는
전자가 61, 후자가 0이 된다.

**`unknown_axes` 히스토그램.**

| `unknown_axes` | 행수 |
|---|---:|
| `["max_pin_burnup"]` | **95 / 95** |
| `"f_xy"` 포함 | **0** |

> **등록된 기대와 정확히 일치한다.** `f_xy` 가 하나도 나타나지 않았다는 것은
> `harvest_maps = true` 가 의도대로 `MAS_OUT` 을 보존해 **수렴한 모든 chain에서 `FXYP` 를
> 회수했다**는 뜻이다(사전등록 §3.1). UNSCORABLE(`−inf`) 행은 0건이다.
> 예측 pin BU는 95행 전부 **63.53–67.99** 로 78 gate에서 10 GWd/tU 이상 여유가 있었다 —
> 예측 심사는 어떤 후보도 잘라내지 않았다.

---

## 6. C — 예측 vs 측정, 그리고 "head가 proxy보다 잘 골랐는가" (사전등록 §8-C/F)

### 6.0 방법 — wave별 serving checkpoint 귀속

`selection.json` 은 **`f_xy` 예측을 기록하지 않는다**(→ §9 D-LOG). 따라서 각 wave가 실제로
서빙한 checkpoint를 재적재해 `predict_fxy` 를 다시 호출했다. 귀속은 추정이 아니라 **검증**했다:
각 wave의 로그된 `pred_mean` 을 후보 checkpoint의 `predict()` 출력과 대조해 유일하게 일치하는
것을 골랐다.

| wave | serving checkpoint | 로그 `pred_mean` 대조 max err |
|---|---|---:|
| 0–4 | `data/models/s1j` (deck ship판) | 9e-05 |
| 5 | `models/champion_wave_04` | 0.130 (아래 주) |
| 6–8 | `models/champion_wave_05` | 0.131 |
| 9 | `models/champion_wave_08` | 0.124 |
| 10–12 | `models/champion_wave_09` | 0.131 (wave 11·12는 **9e-05**) |

주: 0.13은 per-row 잡음이 아니라 **전 후보 공통의 상수 offset**(Δ`f_r` 0.002, Δ`cyclen` 0.122)이다.
디스크에서 다시 적재한 wave(0–4, 11–12)만 1e-4로 정확히 맞는데, 이는 in-memory champion과
저장본의 미세 차이(직렬화)로 설명된다. 귀속 자체는 모든 wave에서 유일하게 결정된다.

**부수적으로, 이 재구성이 §9 D3(σ bar 유실)을 기계적으로 증명했다.** 로그된 `exploit`
(= `cyclen_LCB − 1000·F_xy_UCB − 10⁴·penalty`)을 두 σ 가설로 재계산했을 때:

| wave | `exploit` 재현 median \|residual\| — **proxy σ** 가설 | — **head 자체 σ** 가설 |
|---|---:|---:|
| 0–4 | **4e-05 ~ 6e-05** | 33 ~ 35 |
| 6–10 | **0.122** | 14 ~ 37 |
| **11–12** | 27.2 | **3.0e-05** |

wave 11·12(= call 89–100)만 **head 자체 σ로 서빙되었다.** 사전등록 §9.1 STAMP가 못 박은
"head σ는 서빙되지 않는다"가 마지막 12 call에서 깨졌다. 원인과 처분은 §9 D3.

### 6.1 예측 vs 측정 — 95 라벨행 (s1j 계열, wave별 serving checkpoint)

| 축 | n | MAE | bias (pred − meas) | sd | max\|e\| | Spearman ρ | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`f_xy` (head mean)** | 95 | **0.0514** | **−0.0237** | 0.0703 | 0.2334 | **+0.392** | +0.708 |
| `f_xy` (proxy, 반사실) | 95 | 0.0646 | −0.0422 | 0.0756 | 0.2683 | **+0.439** | +0.704 |
| `f_r` | 95 | 0.0453 | −0.0243 | 0.0587 | 0.2478 | +0.502 | +0.754 |
| `cyclen` (EFPD) | 95 | **0.9051** | −0.3620 | 1.1114 | 3.4551 | +0.837 | +0.856 |
| `cbc_max` (ppm) | 95 | **7.6464** | −2.2854 | 9.8027 | 32.409 | +0.852 | +0.894 |
| `node_peak` | 95 | 0.0330 | −0.0021 | 0.0492 | 0.2204 | +0.580 | +0.742 |
| `f_q` | 95 | 0.0633 | −0.0289 | 0.0852 | 0.3428 | +0.540 | +0.715 |
| \|AO\| | 95 | 0.0010 | −0.0005 | 0.0014 | 0.0057 | +0.592 | +0.827 |

**사전등록된 serving yardstick 대조 (§8-C 필수 항목).**

| 기준 | 등록값 | r1 실측 | 판정 |
|---|---:|---:|---|
| `f_xy` **head** holdout MAE | 0.066 | **0.0514** | ✅ **더 좋다** (−0.0146) |
| `f_xy` **proxy** MAE | 0.073 | **0.0646** | ✅ 더 좋다 (−0.0084) |

> 즉 **level 정확도로는 head가 등록 기준을 넘겼고 proxy도 넘겼다.** 그러나 아래 §6.3이 보이듯
> **level 정확도와 랭킹 능력은 이 셀에서 분리된다.**

축별 주의: `cyclen` MAE 0.905 EFPD는 §2.3의 1 EFPD 수용선과 **거의 같다** — 최적점 parity의
cyclen PASS(0.349)는 여유가 아니라 운에 가깝다. `cbc_max` MAE 7.65 ppm은 1 ppm 수용선의
**7.6배**이며, §2.1의 CBC parity FAIL은 축 전체 수준의 반영이다.

### 6.2 `F_xy` 회귀·proxy 캘리브레이션 표 (사전등록 §8-C 표를 채운다)

| 대상 | n | slope a | intercept b | resid sd | proxy bias | proxy sd | max\|e\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전역 proxy (현행 상수) | 6,218 | 1.2176 | −0.2519 | 0.0476 | — | — | 0.31 |
| `T6_T4/f121` 라벨 전체 (사전) | 738 | 1.1103 | −0.0768 | 0.0332 | +0.0027 | 0.0350 | — |
| `T6_T4/f121` joint-clean (사전) | 215 | 0.4811 | +0.8904 | 0.0300 | +0.0275 | 0.0329 | 0.1145 |
| **r1 자체 라벨** | **95** | **1.0745** | **−0.0174** | **0.0256** | **−0.0422** | 0.0756 | 0.2683 |
| **r1 자체 라벨, joint-clean** | **65** | **1.0351** | **+0.0438** | **0.0212** | **−0.0329** | 0.0485 | 0.1340 |
| **r1 + 사전 라벨 결합 (셀 전체)** | **922** | **1.2095** | **−0.2344** | **0.0417** | −0.0040 | 0.0417 | 0.3206 |
| **결합, joint-clean** | **280** | **0.6607** | **+0.6162** | **0.0289** | −0.0259 | 0.0307 | 0.1145 |

부호 규약 주의: 사전등록 §5.1 표는 `측정 − proxy`, 위 표의 `proxy bias` 는 `proxy − 측정`이다
(§8-C 표 헤더와 동일). 부호를 뒤집으면 사전 joint-clean의 +0.0275 가 −0.0275 에 대응한다.

> **결합 적합이 사전 값을 크게 움직였다.** joint-clean 기울기가 **0.4811 → 0.6607** 로 올랐고,
> 라벨 전체 기울기는 1.1103 → 1.2095 로 전역 proxy(1.2176)에 오히려 접근했다. r1이 추가한
> 95행이 셀의 `F_xy`–`F_r` 관계 추정을 **결합 방향으로** 끌어당겼다. r2의 셀별 `(a,b)` 는
> 이 결합 값을 써야 하며, 사전등록 §1.3이 근거로 삼은 0.282/0.4811 은 더 이상 이 셀의
> 최신 추정이 아니다.

**`F_xy ≤ 1.65` 근방 오분류 표 (필수 항목).** 95 라벨행, gate 1.65.

| ranker | 판정 기준 | TP (pred ok / meas ok) | **FP (pred ok / meas 위반)** | FN (pred 위반 / meas ok) | TN |
|---|---|---:|---:|---:|---:|
| head | mean ≤ 1.65 | 56 | **10** | 7 | 22 |
| head | **서빙된 UCB** (mean + 0.25σ_proxy) ≤ 1.65 | 49 | **8** | 14 | 24 |
| proxy | mean ≤ 1.65 | 59 | **12** | 4 | 20 |
| proxy | 서빙된 UCB ≤ 1.65 | 49 | **9** | 14 | 23 |

> head가 proxy보다 FP를 2건 줄였다(10 vs 12; UCB 기준 8 vs 9). 이것이 head의 **유일하게 확인된
> 실질 우위**다. 서빙된 UCB는 두 ranker 모두 FP를 줄이는 대신 FN을 4→14 / 7→14 로 키웠다 —
> `K = 3.0` 의 비관성이 의도대로 작동했다는 뜻이며, 이 라운드에서 gate 1.65는 slack이므로
> 그 대가는 지불할 만했다.

### 6.3 head가 proxy보다 **잘 골랐는가** — regret@8

**정의(명시).** 각 wave의 8개 후보 pool 전체는 `selection.json` 에 보존되지 않으므로(→ §9 D-LOG),
비교는 **그 wave가 실제로 평가한 8개(reserve wave는 4개)** 위에서 한다.
`regret@8(ranker) = F_xy(ranker가 1위로 놓은 후보의 측정값) − (그 wave 8개의 측정 최소값)`.
0이면 그 ranker가 해당 wave의 최선을 1순위로 뽑았다는 뜻이다.

| wave | n | wave min | head top-1 | regret | proxy top-1 | regret | 실제 acq rank-1 | regret |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 1.5876 | 1.6276 | +0.0400 | 1.5876 | **0.0000** | 1.6276 | +0.0400 |
| 1 | 8 | 1.5982 | 1.5982 | **0.0000** | 1.6672 | +0.0690 | 1.5982 | 0.0000 |
| 2 | 7 | 1.6046 | 1.6223 | +0.0177 | 1.6223 | +0.0177 | 1.6307 | +0.0261 |
| 3 | 8 | 1.5950 | 1.5950 | **0.0000** | 1.6406 | +0.0456 | 1.6077 | +0.0127 |
| 4 | 8 | 1.5958 | 1.7048 | +0.1090 | 1.6300 | +0.0342 | 1.6300 | +0.0342 |
| 5 | 8 | 1.5883 | 1.5890 | +0.0007 | 1.5890 | +0.0007 | 1.5890 | +0.0007 |
| 6 | 7 | 1.5731 | 1.5731 | **0.0000** | 1.6116 | +0.0385 | 1.5731 | 0.0000 |
| **7** | 7 | **1.5322** | **1.5322** | **0.0000** | 1.6337 | +0.1015 | **1.5322** | 0.0000 |
| 8 | 8 | 1.5490 | 1.8068 | +0.2578 | 1.8068 | +0.2578 | 1.8068 | +0.2578 |
| 9 | 8 | 1.5337 | 1.5992 | +0.0655 | 1.5992 | +0.0655 | 1.5992 | +0.0655 |
| 10 | 7 | 1.5777 | 1.6074 | +0.0297 | 1.5777 | **0.0000** | 1.5871 | +0.0094 |
| 11 | 7 | 1.5783 | 1.6042 | +0.0259 | 1.6042 | +0.0259 | 1.5832 | +0.0049 |
| 12 | 4 | 1.5819 | 1.5973 | +0.0154 | 1.5973 | +0.0154 | 1.5973 | +0.0154 |
| **평균** | | | | **0.0432** | | **0.0517** | | **0.0359** |

**판정: head의 우위는 확인되지 않았다.**

- 평균 regret은 head **0.0432** vs proxy **0.0517** — head가 0.0085 낫다. 그러나 wave 단위 승패는
  **head 승 4 / 패 3 / 무 6**. sign test로 p ≈ 1.0, **유의하지 않다.**
- **실제 acquisition(`exploit` scalar) 의 regret 0.0359 가 head-mean 단독 순위(0.0432)보다 낮다.**
  즉 `cyclen_LCB` 항과 hard-tier penalty가 head의 순위 오류를 부분적으로 상쇄했다.
- 순위 상관(Spearman, 예측 vs 측정 `F_xy`)은 오히려 proxy가 낫다:

| slice | n | head ρ_s | proxy ρ_s |
|---|---:|---:|---:|
| 전체 라벨행 | 95 | +0.392 | **+0.439** |
| joint-clean | 65 | **+0.016** | +0.179 |
| `exploit` slot only | 62 | **−0.383** | −0.144 |

**feasible basin 안에서 head의 랭킹 능력은 0이고(ρ +0.016), 실제로 예산의 64%를 쓴 `exploit`
slot 위에서는 음의 상관(−0.383)이다.** head의 전체 ρ +0.392는 explore/control이 뽑아 온
나쁜 후보(`F_xy` 1.7~1.9)를 나쁘다고 맞힌 데서 나온 것이지, 좋은 후보들 사이를 가른 능력이 아니다.

> **사전등록 §5.2 / §4.3에 대한 판정.** head arm에서 PRIMARY가 나왔으므로 §4.3의 "proxy 실명"
> 결론은 발동하지 않고, 형식적으로는 **"`F_xy` 를 실제로 최적화했다"** 고 말할 수 있다.
> **그러나 그 문장은 이 라운드의 증거보다 강하다.** 정직한 진술은:
> **"`F_xy` head의 mean이 후보를 랭크했고(fxy_source='head', 13/13 wave), level 정확도는 등록
> 기준을 넘겼으나, feasible basin 내부의 랭킹 능력은 proxy 대비 개선되지 않았다. 1.5322 의
> 출처는 acquisition의 `F_xy` 순위가 아니라 측정 `F_xy` 로 정렬된 elite pool(§7)과 local search다."**
> 사전등록 §5.1이 arm A에 부과한 "불가" 문구는 arm B에는 적용되지 않지만, 위 진술은
> 그것과 실질적으로 같은 겸손을 요구한다.

---

## 7. `T6_T4/f121` `min_fr` 시대(r1–r8)와의 비교 — 무엇이 바뀌었나

### 7.1 스칼라

셀 내 `f_xy` 라벨 보유 캠페인의 joint-clean 최소값:

| 캠페인 | joint-clean n | **min `F_xy`** | min `F_r` | median `F_xy` |
|---|---:|---:|---:|---:|
| **`fpcamp_minfxy_t6t4_f121_r1`** | **65** | **1.5322** | 1.4769 | **1.6016** |
| `fpcamp_minfr_T6T4_r3` | 14 | 1.5491 | 1.4984 | 1.6068 |
| `fpcamp_minfr_T6T4_r5` | 43 | 1.5549 | 1.4812 | 1.6145 |
| `fpcamp_minfr_T6T4` (base) | 31 | 1.5648 | 1.5082 | 1.6123 |
| `fpcamp_minfr_T6T4_r4` | 57 | 1.5688 | 1.4866 | 1.6474 |
| `fpcamp_minfr_T6T4_r6` | 67 | 1.5829 | 1.4797 | 1.5969 |

`min_fr` r3→r6 구간에서 `F_xy` 가 1.5491 → 1.5829 로 **+0.0338** 악화했다는 사전등록 §1.1의
관측을 기준으로 하면, **r1은 그 손실을 전부 되돌리고 0.0169 를 더 벌었다** (1.5829 → 1.5322,
**−0.0507**). 사전등록이 던진 핵심 질문 — "F_r 추격이 밀어낸 0.0338을 `min_fxy` 가 되찾는가" —
의 답은 **되찾았고 초과했다** 이다.

**대칭 관측 하나.** `min_fxy` 라운드가 부수적으로 만들어 낸 최소 `F_r` 은 **1.4769**(call 84)로,
이 셀의 `min_fr` r5(1.4812)·r6(1.4797)보다 낮고 r8의 λ-opt(1.4749)·기록(1.4605)보다는 높다.
**목적함수를 바꿨는데 `F_r` 축에서 8라운드 중 5등을 100 call로 해냈다** — 두 축이 이 basin에서
같이 움직인다는 §4.1의 상관(+0.715)과 정합한다.

### 7.2 고른 core의 구조 — `rule_metrics.py` descriptor

`lpopt/search/rule_metrics.py` 의 pattern-only descriptor로 두 시대의 상위 core를 비교했다
(RM5 `rm_peripheral_power_share` 는 BOC power map을 요구하므로 제외).

| 집합 | n | mean `F_xy` | mean `F_r` | **`F_xy`/`F_r`** | cyclen | node_peak | RM1 | RM1i | RM2 | RM2i | RM4 | RM6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **r1 상위 10 (by `F_xy`)** | 10 | 1.5612 | 1.4977 | **1.0425** | 622.71 | 1.3307 | **84.0** | **44.0** | **100.4** | **68.4** | 32.0 | **0.107** |
| `min_fr` 시대 상위 10 (by `F_r`) | 10 | 1.5965 | 1.4854 | **1.0748** | 622.41 | 1.3053 | 90.8 | 49.6 | 94.4 | 61.6 | 32.4 | 0.061 |

개별 core:

| core | `F_xy` | `F_r` | ratio | cyclen | node_peak | RM1 | RM1i | RM2 | RM2i | RM4 | RM6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **r1 winner `bf3a70b2…`** | **1.5322** | 1.4857 | 1.0313 | 622.101 | 1.3376 | 80 | 40 | 104 | 72 | 32 | 0.1405 |
| r1 #2 `d06a56d6…` | 1.5337 | 1.5098 | 1.0158 | 622.974 | 1.3351 | 84 | 44 | 104 | 72 | 32 | 0.1074 |
| r1 min-`F_r` `70dbb5fe…` | 1.5777 | 1.4769 | 1.0683 | 624.022 | 1.2910 | 92 | 52 | 92 | 60 | 32 | 0.0413 |
| 셀 `F_xy` incumbent r3 `46e687ed…` | 1.5491 | 1.5018 | 1.0315 | 621.275 | 1.3333 | 80 | 40 | 104 | 72 | 32 | 0.1405 |
| **`F_r` 기록 `4d70ab6f…` (1.4605)** | (라벨 없음) | 1.4605 | — | 618.021 | 1.3171 | **80** | **40** | **104** | **72** | **32** | **0.1405** |
| `F_r` λ-opt r8 `188c9a33…` (1.4749) | (라벨 없음) | 1.4749 | — | 625.459 | 1.2957 | 92 | 52 | 92 | 60 | 32 | 0.0413 |
| `F_r` r6 `9b9fabe8…` | 1.5829 | 1.4797 | 1.0697 | 623.592 | 1.3034 | 92 | 52 | 92 | 60 | 32 | 0.0413 |

**세 가지가 읽힌다.**

1. **`F_xy`/`F_r` ratio가 내려갔다: 1.0748 → 1.0425.** `min_fxy` 가 고른 core는 같은 `F_r` 수준에서
   더 낮은 `F_xy` 를 낸다 — 목적함수 전환이 **같은 축을 다시 타는 것이 아니라 다른 trade-off
   지점을 골랐다**는 직접 증거다.
2. **leakage pattern은 바뀌지 않았다.** RM4(`rm_fresh_periphery`, 주변부 fresh 다중도 가중 개수)는
   32.0 vs 32.4 로 사실상 동일하다. **두 시대 모두 low-leakage 배치를 유지했고, 차이는 주변부가
   아니라 inboard에 있다.**
3. **fresh 배치가 face-clustering에서 diagonal-spreading으로 옮겨갔다.** `F_xy` 시대는
   RM1 84.0(vs 90.8) / RM1i 44.0(vs 49.6)로 **fresh-fresh face 인접이 적고**, RM2 100.4(vs 94.4) /
   RM2i 68.4(vs 61.6)로 **대각 인접이 많으며**, RM6(checkerboard degree) 0.107(vs 0.061)로
   **fresh가 더 자주 서로 face를 맞대지 않는다.** RM1i는 `rule_metrics` 문서가 node_peak/`f_q`/`f_r`의
   인과 운반자로 지목한 변수다(ρ +0.235/+0.282/+0.254). 즉 **`min_fxy` 는 RM1i를 낮추는 방향으로
   갔고 `min_fr` 은 그 반대로 갔다.**

> **가장 눈에 띄는 단일 사실.** r1 winner `bf3a70b2…` 의 descriptor 6개(80/40/104/72/32/0.1405)는
> 셀의 `F_xy` incumbent `46e687ed…`(r3) 및 **`F_r` 기록 core `4d70ab6f…`(1.4605, `batchswap_enum`)**
> 와 **완전히 동일하다.** 즉 세 core는 같은 fresh-placement family이고 burned shuffle만 다르다.
> `min_fr` r4–r8이 옮겨 간 family(RM1 92 / RM6 0.0413)는 `F_r` 을 계속 낮췄지만
> **`F_xy` 는 그 family에서 1.5698–1.5829 에 머물렀다.** `min_fxy` r1은 100 call을 써서
> **1.4605 기록 core가 속한 family로 되돌아왔고**, 그 family 안에서 `F_xy` 를 1.5322로 눌렀다.
>
> **경고 (등록해 둔다).** 그 1.4605 core에는 여전히 `f_xy` 라벨이 없다(MAS_OUT 미보존, 사전등록
> §1.1의 "등록된 미지수"). 위 정합은 **descriptor 수준의 동형성**이지 그 core의 `F_xy` 측정이
> 아니다. **r1은 그 공백을 메우지 않았다** — 사전등록이 예고한 그대로다. 이 family 동형성은
> phase-2에서 그 core를 replay할 강한 근거가 되지만, 그 자체가 근거는 아니다.
> node_peak은 `F_xy` 시대가 오히려 **높다**(1.3307 vs 1.3053) — `F_xy`(planar)와 node_peak(axial)은
> 이 셀에서 같은 것을 재지 않는다(joint-clean 상관 +0.075). `F_xy` 최적화가 축방향 평탄도를
> 개선한다는 주장은 **이 데이터가 지지하지 않는다.**

---

## 8. 등록된 감시(§4)의 사후 판정

NULL이 나오지 않았으므로 §4.1–§4.3의 귀속 판정은 **전부 moot** 다. 그럼에도 등록된 대로
숫자를 남긴다 — r2의 사전등록이 이 값을 다시 쓸 것이기 때문이다.

| 감시 | 등록된 판정선 | 실측 | 결과 |
|---|---|---|---|
| §4.1 CBC 벽 | NULL 시 상위 10 core CBC ≥ 1550 이어야 귀속 허용 | 상위 10 최대 **1354.77** (gate까지 245 ppm), `corr(F_xy,CBC)` **−0.435** | 귀속 **금지** (moot). CBC는 이 셀에서 벽이 아니다 — 사전등록의 예측대로 |
| §4.2 `F_r` 벽 | NULL 시 "`F_xy` 더 낮은데 `F_r`>1.55 로 탈락" 3건 이상 | **0건** (`F_xy < 1.5322` 인 행 자체가 0) | 귀속 **금지** (moot) |
| §4.3 proxy 실명 | arm A NULL 시 head 지지 증거 | arm **B**가 돌았고 PRIMARY 달성 | 발동 안 함. 대신 §6.3이 **head 랭킹 우위 미확인**을 기록 |
| §4.4 λ slide | 상위 10 median cyclen이 621.28 대비 −5 EFPD 이상 | **+1.353 EFPD** | **slide 없음.** λ=1000 유지 |
| §6 `F_q` non-binding | binding으로 바뀌면 이례 사건 | 95행 중 `F_q > 2.41` **1건**(`F_xy`·`F_r` 동시 위반행) | non-binding 유지. 이례 사건 아님 |

---

## 9. 편차 (deviations) — 사전등록에 없던 것

| # | 편차 | 영향 |
|---|---|---|
| **D1** | **post-verify (top-k MASTER 재검증 / SDM-MTC) 미실행.** `state.json → post_verify_done = false`, `post_verify_calls = 0`, `post_verify_violators = []`. | **원인은 crash가 아니라 구조다.** HOST_181 log: `SDM/MTC gate: no delivery ranking to verify (objective='min_fxy'; the gate targets the flat_power delivery candidates, decision D9) — gate not run`. deck의 `post_verify_top_k = 5` 는 `min_fxy` 에서 **inert** 하다. → §10 처분 |
| **D2** | 로컬 log 말미의 `UnicodeEncodeError: 'cp949' … '\u2014'` (`campaign.py:518` print). | **canonical run에는 영향 없음.** 이 traceback은 HOST_199에서 돌던 **폐기된 duplicate resume** 의 것이다. canonical run은 HOST_181에서 `rc=0` 로 완료하고 `report.md` 를 썼다(receipt `E188572A…`). 다만 **console 인코딩 하나로 프로세스가 죽는 취약점**은 남아 있다 |
| **D3** | **head σ bar가 resume에서 유실됐다.** wave 11–12(call 89–100)가 head 자체 σ(0.054–0.196)로 서빙됐다 — §6.0의 `exploit` 재구성으로 증명. | 원인: `_save_champion` 이 쓰는 run-dir checkpoint에 **`ensemble.json` 이 없다**(`champion_wave_*/` 확인). `fxy_sigma_barred` 는 그 파일의 `fxy_head.serve_sigma` 를 읽으므로(`model_api.py:610-626`), resume이 champion을 재적재하는 순간 **G4 bar가 조용히 풀린다.** 영향: 마지막 12 call의 acquisition은 사전등록 §9.1 STAMP가 못 박은 규약과 다른 UCB를 썼다. **frontier에는 영향 없음**(1.5322는 call 57, wave 11·12는 무개선). **코드 결함으로 등록하며 r2 이전 수정 필요** |
| **D4** | launcher hash 불일치. §9.1 STAMP는 `36ACB60D…`(9,537 B), 현재 `D8345C2B…`(9,614 B). | 치환 stamp 이후 launcher가 다시 편집됐다(주석 추가로 보임). deck·store gate 로직 자체는 §9.1 값을 그대로 들고 있다. **재실행 전 재stamp 필요** |
| **D5** | store가 launcher의 `$wantStore` 를 지나쳤다. `$wantStore = 72516916…`(75,793행), 현재 `255F0E41…`(**75,893행**). | r1의 100행이 merge된 정상 진행. **지금 launcher를 그대로 돌리면 `MINFXY1 REFUSED: store sha256 mismatch` 로 거부된다** — 의도된 동작이며, r2 deck은 자기 값으로 stamp해야 한다 |
| **D6** | **host 이관 199 → 181.** 사전등록 §9.3은 "**199 전용**"을 등록했다. | `orchestration/migration_stop_199_20260830T083229.json`: 08:32, wave 11, spent 88/100, 사유 "user-directed continuation on 181", `stopped: true`. HOST_181(DESKTOP_HOST_181)이 09:15 `rc=0` 로 완료. **call 1–88 = 199, call 89–100 = 181.** run dir의 `labels.jsonl`·`state.json` 은 181 receipt의 sha256과 **byte-identical** 이므로 canonical 판은 181이다. D3가 정확히 이 이관 지점에서 발생했다 |
| **D-LOG** | `selection.json` 이 `f_xy` 예측을 기록하지 않는다. `pred_mean` 은 7열 surrogate(`f_r, cbc_max, f_q, cyclen, ao_abs, max_assembly_burnup, max_pin_burnup`)뿐이고, `ScoredPool.fxy_ucb` / head mean / 후보 pool 전체는 버려진다(`campaign.py:2593-2632`). | 그래서 §6은 checkpoint 재적재로 복원해야 했고, §6.3의 regret은 **평가된 8개** 위에서만 정의할 수 있었다(pool 전체 랭킹 비교 불가). **`min_fxy` 라운드에서 목적축 예측을 남기지 않는 것은 판독 결함**이다. 별도 작업으로 수정 중 |

---

## 10. Provenance

### 10.1 sha256 — §9.1 대조

| item | 실측 sha256 | bytes | §9.1 대비 |
|---|---|---:|---|
| **deck** `fpcamp_minfxy_T6T4_f121_r1_199.inp` | `BEF3519E720FE1F94FE1448EF3046FCE1EB15BD94DBA5DD1F4E9B2F3976C95C9` | 16,957 | ✅ **일치** (s1j 치환판) |
| run dir `input_deck.inp` | `BEF3519E…` (동일) | 16,957 | ✅ 실행된 deck이 stamp된 deck임을 확증 |
| **store** `data/store/records.parquet` (**75,893행**) | `255F0E41707CB4EF64D843FD19DB81531C12AB3A969F6F8F06C87E0AF5561A51` | 22,570,584 | ⚠️ §9.1 STAMP `72516916…`(75,793행)에서 **+100행** = r1 harvest (D5) |
| `data/store/fuel_types.parquet` | `FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137` | 64,343 | ✅ 일치 |
| champion `data/models/s1j/member_20260716/meta.json` | `F0AF69C0F54261DEC61F253E3828FDC5DF742F0915440678C885B96BB4112E7B` | 37,223 | ✅ 일치 |
| `data/models/s1j/ensemble.json` (**serve_sigma = "barred"**) | `75CDC81874F8AD3B972C3D8A5AB210C7F973B1743299B2AAA4E533AA0F36C8AC` | 847 | §9.1 미등록 — D3의 근거 파일이므로 여기서 stamp |
| `launch_…_r1_199.ps1` | `D8345C2B4AE24642F4BDD696D3BCB4EA720FD69AABB6FA82A4DAE3580CC25C52` | 9,614 | ❌ §9.1 `36ACB60D…`(9,537 B)와 불일치 (D4) |
| `run_…_r1_199.bat` | `F3002666EECC7F5A2BD23071298ED7308D2EDDF629B0507358DECFFEED44CE49` | 2,750 | ✅ 일치 |
| `status_…_r1_199.ps1` | `78D527E85910CFFB533ACBA2F7A1358E42FCE6E759ACF96759F4EB4D68D0AC3C` | 4,700 | ✅ 일치 |
| scratch example `fpcamp_minfxy_T6T4_f121_199.inp` | `27234CC338A5B655D503B477100ACD875D08E9992861D6EA945BAB321436CCF3` | 8,963 | ✅ 일치 (실행 안 함) |

run-dir 산출물 (HOST_181 completion receipt `fpcamp_resume_181_complete_v1` 와 대조):

| 파일 | 로컬 sha256 | receipt | 판정 |
|---|---|---|---|
| `labels.jsonl` | `897029C29C8B170A4EB5F7D907BE1DF4B809F601546038BF3D6F5867FEED865E` | 동일 | ✅ byte-identical |
| `state.json` | `E188572A79B4E528E436359F692EB2E3A82BA285B39CBC6802B068202EE0518B` | 동일 | ✅ byte-identical |
| `status.json` | `47D82075A34A31C8A2D3FE7EAEDAA59CCE81EB5FA4E88A33BCB0C324F7588FA7` | `D96E1CAF…` | ⚠️ 다름 — `updated: 2026-08-30T11:17:38` 로 로컬 report 재생성 시 재기록됨. `best`/`n_feasible` 등 내용은 동일 |

### 10.2 resume / 이관 이력

| 시각 (KST) | 사건 | 근거 |
|---|---|---|
| — | HOST_199, wave 0–10 실행, spent 88/100 | `fpcamp_minfxy_t6t4_f121_r1_out.log` line 4–14 |
| wave 10 직후 | 프로세스 사망 → cmd 오류 1줄(`파일 이름, 디렉터리 이름 또는 볼륨 레이블 구문이…`) 후 재기동 | log line 15–16 |
| 08:32:24 | **migration stop.** wave 11, spent 88, best_fxy 1.5322 고정. `state`/`status`/`labels` sha 봉인 | `orchestration/migration_stop_199_20260830T083229.json` |
| ~08:3x–09:15 | **HOST_181 (DESKTOP_HOST_181) resume**, wave 11·12 실행, spent 100/100 | `resume_181.stdout.log`, receipt |
| **09:15:13** | **canonical 완료 `rc=0`**, `report.md` 작성, SDM/MTC gate "not run" | receipt `fpcamp_resume_181_complete_v1` |
| ~09:45, ~10:16 | HOST_199에서 obsolete duplicate resume 2회 (log의 banner 3회 반복 = spent=88 재기동) | log line 17–24 |
| 10:15:56 | **duplicate stop.** "199 duplicate resume is obsolete because canonical campaign completed 100/100 on 181; freeze for E migration", `labels: 88`, `master: 0` | `orchestration/duplicate_stop_for_e_migration_20260830T1016.json` |
| — | duplicate 프로세스가 wave 11·12를 재실행한 뒤 `UnicodeEncodeError` 로 사망 (log line 25–51) | log tail |

> **과제 브리핑 대비 정정 2건.** (a) 10:16의 종료는 "unexplained external kill"이 아니라
> **E 마이그레이션을 위한 의도된 duplicate stop** 으로 receipt에 기록되어 있다.
> (b) post-verify 미실행의 원인은 `UnicodeEncodeError` 가 **아니다** — canonical run(181)은
> 그 예외 없이 `rc=0` 로 끝났고, gate는 `min_fxy` 에서 구조적으로 inert 하다(D1).
> duplicate가 재실행한 wave 11·12는 결정론적으로 동일한 결과(conv=7 feas=5 / conv=4 feas=4)를
> 냈고, 그 행은 canonical run dir·store 어디에도 들어가지 않았다(store campaign 행 = 정확히 100).

### 10.3 E: 백업

| 경로 | 내용 |
|---|---|
| `E:\fpcamp_minfxy_migration_199_to_181_20260830\fpcamp_minfxy_migration_199_to_181_20260830.tar` | 08:36, 7,024,492,544 B — 이관 시점(spent 88) kit 스냅샷 |
| `E:\fpcamp_minfxy_migration_199_to_181_20260830\fpcamp_minfxy_finish_181_20260830_final.tar` | 09:20, 5,538,172,928 B — 181 완료본 |
| `E:\fpcamp_minfxy_migration_199_to_181_20260830\final_summary_181\` | `resume_181.stdout.log` / `.stderr.log` / `resume_181_complete_receipt.json` / `labels.jsonl` / `state.json` / `status.json` / `report.md` |
| `E:\lpopt_archive\199_runs_20260830\` | `TRANSFER_MANIFEST_20260830.md`, `kit_frontier_runs`, `kit_frontier_data_produce`, `kit_pc2_v4_runs`, `D_lpopt_archive_199_runs` |
| `E:\lpopt_archive\store_backups_20260829\` | store/fuel_types/maps 백업 세트 (r1 이전 판) |

---

## 11. 처분 (disposition) — 사전등록에 따른 다음 수

### 11.1 이 라운드가 확정한 것

1. **PRIMARY 달성.** 셀 incumbent `1.5491 → 1.5322`. `min_fr` 시대가 밀어낸 0.0338을 되찾고 초과했다.
2. **STRETCH 미달.** 1.5295(`E1_E2/f121`)를 넘지 못했으므로 사전등록 §1.3의 교차 인용 규칙에 따라
   **이 결과를 "프로그램 기록"이라고 부를 수 없다.** 프로그램 joint-clean `F_xy` 순위 **3위**다.
3. **예산은 binding이 아니었다.** 기록은 call 57, 이후 43 call 무개선.
4. **λ=1000은 옳았다.** slide 없음, band-internal λ-optimal = headline core.
5. **head는 level에서 등록 기준을 넘겼으나 랭킹에서 proxy를 이기지 못했다.**

### 11.2 post-verify 미실행으로 **막힌** 항목 (D1)

| 항목 | 상태 |
|---|---|
| top-5 MASTER 재verification (`post_verify_top_k = 5`) | **미실행** — `F_xy` 재현성·`FXYP` determinism이 이 라운드 안에서 확인되지 않았다 |
| SDM / MTC licensing chain | **미실행** — `_rod_model()` 이 `None` 을 돌려주는 구조적 한계와 D9 scoping이 겹쳐 `min_fxy` 에서 gate 자체가 inert |
| PRIMARY (delivery) 판정 | **불가** — pin BU 0/95 (사전등록된 구조, §5.4) |

### 11.3 권고 — phase-2 wave를 `pinbu_wave_fxyera_r1` 형식으로 즉시 편성

사전등록 §5.4가 등록한 phase-2를 **`pinbu_wave.py` + `pinbu_wave_keep_199.inp`** 로 실행하되,
`pinbu_wave_fxyera_r1_prereg/results_20260830` 의 형식을 그대로 따른다(target set + replicate set,
`keep_success` 로 `MAS_OUT` 보존). 이 한 번의 wave가 D1이 막은 세 가지를 동시에 푼다:
(1) frontier core의 **측정 pin BU** → PRIMARY(delivery) 판정,
(2) 같은 pattern 재실행의 `FXYP` 대조 → **`F_xy` determinism**(= post-verify 재verification의 대체),
(3) `MAS_OUT` 보존.

**target set — r1 joint-clean `F_xy` 상위 5 (phase-2 재실행 대상):**

| # | `record_id` | `F_xy` | call / wave | `F_r` | CBC | cyclen | 예측 pin |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | `bf3a70b20e508c7c01d15fd62bccc653376f65072a7507a9b3fdc755898ed982` | **1.5322** | 57 / 7 | 1.4857 | 1337.38 | 622.101 | 64.28 |
| 2 | `d06a56d6b92851b4a18d8ffbafcb27c93daeed8363fba0cd139750f390722f8e` | 1.5337 | 79 / 9 | 1.5098 | 1354.77 | 622.974 | — |
| 3 | `8b5221442ce89222e1f0c769547fb5d06135ac9a0968ee038aca55df555510d8` | 1.5490 | 67 / 8 | 1.4912 | 1325.25 | 621.619 | — |
| 4 | `e1495cbe169efbfb9abc8b2ecdb2f489da5b13cfded0de6bd70ef01b231754c7` | 1.5577 | 69 / 8 | 1.5076 | 1337.29 | 622.291 | — |
| 5 | `2613a46a257a46cf21c5287822864ac8096261499e46550425515af0a2365389` | 1.5631 | 58 / 7 | 1.4966 | 1341.54 | 623.086 | — |

**phase-2에 추가 편성을 권고하는 항목 (§7.2 발견에 근거, 별도 사전등록 필요):**
`F_r` 기록 core `4d70ab6f75d4a0be…`(1.4605, `batchswap_enum_T6T4`)와 λ-opt `188c9a338d9ffae7…`(1.4749)의
**`F_xy` 측정**. 두 core는 `f_xy` 라벨이 0건이고(사전등록 §1.1의 "등록된 미지수"), 전자는 r1 winner와
descriptor가 **완전히 동일한 family** 다. 이 두 값이 없으면 "`min_fxy` 가 `min_fr` 보다 나은 core를
고른다"는 진술은 라벨이 있는 부분집합에만 한정된다.
**주의:** phase-2 merge 후 store sha가 다시 바뀌므로 §9.1 행과 launcher `$wantStore` 를 재stamp 해야 한다(D5).

### 11.4 r2 조건 (사전등록 §10-2에 따라 **여기서 승인하지 않는다** — 새 deck·새 사전등록 필요)

r2 사전등록이 **반드시** 담아야 할 항목:

1. **§8-C 표가 gating 항목이 된다** — 사전등록 §2.3 후단이 head arm `f_xy` parity FAIL(0.0543)로 발동했다.
2. **셀별 `(a,b)` 를 결합 적합으로 갱신** — joint-clean 기울기 0.4811 → **0.6607**(n=280), 라벨 전체
   1.1103 → **1.2095**(n=922). §1.3이 인용한 `corr = 0.282` 도 갱신되어야 한다(r1 자체 0.715).
3. **D3 수정 확인 gate** — champion save가 `fxy_head.serve_sigma` 를 보존하는지, 또는 resume이
   deck의 `model_dir` 로부터 bar를 재확인하는지. 미수정 상태로는 "한 라운드 = 하나의 serving 규약"을
   보장할 수 없다.
4. **D-LOG 수정 확인** — `selection.json` 이 `f_xy` mean/σ/source와 (가능하면) pool 랭킹을 남기는지.
   남지 않으면 r2의 §8-C도 checkpoint 재적재로만 채워진다.
5. **셀 선택 재검토.** §3의 frontier가 call 57 이후 43 call 정체를 보였으므로, **같은 셀 재시도(r2 same-cell)
   보다 `E1_E2/f121`(사전등록 §1.3이 r2로 지명)이 지지된다** — 그 셀의 joint-clean 물량(413)·incumbent
   품질(1.5295)·`corr(F_xy,F_r)` 0.737 모두 이 라운드에서 관측된 basin 구조와 정합한다.
   단 §1.3의 근거였던 "이 셀 corr 0.282" 가 0.715로 갱신되었으므로, "proxy 실명을 가장 가혹하게
   시험한다"는 셀 선택 논거는 **r2에서 재작성되어야 한다.**
6. **head 승격 재심.** §6.3이 보인 랭킹 무능(joint-clean ρ +0.016, exploit slot ρ −0.383)은 G4 FAIL과
   별개의 문제다. r2에 head를 다시 서빙하려면 **within-cell ranking gate**(예: feasible slice Spearman)를
   승격 조건에 추가해야 한다 — 현행 G1–G4는 level만 본다.
7. `policy_prior` 는 계속 `"off"`. 이 라운드는 policy A/B의 control arm(A) 후보로 남는다(§10-1).

---

## 12. 정직한 주석

- **§6.3의 결론은 이 라운드 안에서 검정력이 낮다.** wave당 8개, 13 wave — head 승 4 / 패 3 / 무 6.
  "head가 proxy보다 낫지 않다"는 **"낫다는 증거가 없다"** 이지 "나쁘다"가 아니다. 다만
  joint-clean ρ +0.016 과 exploit-slot ρ −0.383 은 regret보다 표본이 크고(65, 62) 방향이 일관된다.
- **§6.0의 `exploit` 역산은 재구성이다.** 검증은 두 겹으로 했다: (a) wave별 checkpoint 귀속을
  로그된 `pred_mean` 과 대조해 유일하게 결정했고, (b) 역산된 `exploit` 이 proxy-σ 가설에서
  wave 0–10 전부 median \|residual\| ≤ 0.13(대부분 1e-4)로 재현됐다. 그럼에도 이것은
  **로그가 남겼어야 할 값의 복원**이며, D-LOG가 고쳐지면 r2에서는 직접 읽어야 한다.
- **1.5322의 공로를 acquisition에 돌리지 않는다.** 사전등록 §5.1이 arm A에 요구한 귀속 규칙
  (elite pool + wave fine-tune)이 arm B에서도 사실상 성립한다 — §6.3이 그 이유다.
  `elite_top_k = 32` 슬롯이 전부 측정 `F_xy` 로 정렬된 라벨 행으로 채워진다는 **사전등록 §7**의 검증이
  이 라운드의 실질적 엔진이었다.
- **`node_peak` 은 이 objective의 대리변수가 아니다.** joint-clean 상관 +0.075, 그리고 `F_xy` 시대
  상위 10의 node_peak이 `F_r` 시대보다 **높다**(1.3307 vs 1.3053). `F_xy` 개선을 축방향 평탄도
  개선으로 읽으면 안 된다.
- **`cbc_max` 예측 정확도(MAE 7.65 ppm)가 §2.3의 1 ppm 수용선과 8배 어긋난다.** 이 셀에서 CBC gate는
  slack이라 결과에 영향이 없었지만, CBC가 near-binding인 셀(예: TRIPLE f125, frontier 1597.33)에서
  같은 수용선을 적용하면 **parity는 구조적으로 실패한다.** 프로그램 수용선 자체의 재검토 대상으로 기록한다.

---

## 13. 경로

- 사전등록: `data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md`
- 설계서: `data/reports/fxy_switch_design_20260829.md` · head 승격: `data/reports/fxy_head_results_arm3_20260829.md`, `data/models/s1j/PROMOTION.md`
- phase-2 선례: `data/reports/pinbu_wave_fxyera_r1_prereg_20260830.md`, `…_results_20260830.md`
- run dir: `runs/fpcamp_minfxy_t6t4_f121_r1/` (`labels.jsonl`, `state.json`, `status.json`, `report.md`, `waves/wave_00…12/selection.json`, `orchestration/*.json`, `models/champion_wave_{04,05,08,09,12}`)
- log: `fpcamp_minfxy_t6t4_f121_r1_out.log` (199 leg) · `E:\fpcamp_minfxy_migration_199_to_181_20260830\final_summary_181\resume_181.stdout.log` (181 leg, canonical)
- 코드 근거: `lpopt/search/campaign.py` (`_campaign_objective` 1751-1760, `is_deliverable` 459-475, `_write_wave_artifacts` 2593-2632, `_maybe_post_verify` 2874), `lpopt/search/acquisition.py` (`score_min_fxy` 841-935, `predict_fxy` 680-726, `fxy_proxy` 662-677), `lpopt/model/model_api.py` (`fxy_sigma_barred` 610-626, `predict_fxy` 1808-1852), `lpopt/search/rule_metrics.py`, `lpopt/search/verify.py:851`
