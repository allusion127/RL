# 3-fresh-type (graded) campaign, ROUND 2 — `S3_T1_S5` / feed 125 — RESULTS

**Run 2026-08-20 on HOST_199, harvested·merged 2026-08-29.** Deck `fpcamp_minfr_TRIPLE_f125_r2_199.inp` (sha256 `B5683D4F…0116`), run dir `runs/fpcamp_minfr_triple_f125_r2`, champion `data/models/s1i` (cond_v8), seed 5697, **rc = 0**, 8 waves, **60/60 calls**. Pre-registration: `data/reports/tripletype_f125_r2_prereg_20260820.md` (deck hash 이전 작성). Round-1 결과: `data/reports/tripletype_f125_results_20260817.md`. Seed control: `data/reports/hgd569_f125_seedctl_results_20260817.md`. Pin 정의: `data/reports/pinbu_definition_20260820.md`.

Prereg §8 에 등록된 readout(R1 · R2 · R3/CBC-WALL · R-PIN · R-GRADE · R-BUDGET)을 **전부** 아래에서 답한다. 답할 수 없는 것은 답할 수 없다고 적는다.

---

## 0. HEADLINE — 등록된 NULL 이 발화했다

| 마크 (prereg §2) | 요구 | 측정 | 판정 |
|---|---|---|---|
| **PRIMARY / full-feasible** | F_r ≤ 1.55 ∧ CBC ≤ 1600 ∧ F_q ≤ 2.41 ∧ \|AO\| ≤ 0.30, 예측 pin ≤ 78 | **0 / 58** | **미달** |
| **STRETCH** | F_r < 1.5993 **CBC-clean** (round-1 joint frontier 돌파) | **1.5999** (CBC 1593.80) | **미달, +0.0006** |
| SECONDARY | F_r < 1.5956 (round-1 raw) | **1.5864** (CBC 1615.15 ❌) | 달성, **−0.0092** — 라벨된 raw 수치, headline 아님 |
| PIN | 상위 신규 core 실측 pin BU vs round-1 실측대(74.16–75.58) 및 80 GWd/tU | **실측 0 건** (`max_pin_burnup` 58/58 null) | **측정 안 됨** — 예측만 (§8) |
| **NULL** | 3-type F_r–CBC frontier 가 **1.55 위에서 닫힘** — CBC 가 F_r 보다 먼저 벽이 됨 | **CBC-clean frontier 는 1.5999 에서 정지, raw frontier 는 1.5864 까지 계속 진행** | **🔴 발화 (FIRES)** |

**한 문장으로**: round 2 는 60 콜을 더 써서 **raw F_r 을 −0.0092 내렸지만 그 이득 전부가 CBC 게이트 바깥에서 발생했다.** 보고 가능한(CBC-clean) frontier 는 round 1 의 1.5993 을 **넘지 못했고**(1.5999, +0.0006 나쁨), 두 라운드를 합친 이 cell 의 107 개 수렴 core 중 **F_r < 1.5993 인 17 기 가운데 CBC-clean 은 0 기**, 그 17 기의 최소 CBC 는 **1602.58 ppm** 이다. 이것이 prereg §3 이 사전에 "CBC 가 진짜 벽이라는 지문(fingerprint)"이라고 명시한 바로 그 패턴이다.

> **harvest 메모 정정**: 로그 headline 의 `best-overall F_r 1.586 @ cyclen 731.5 EFPD` 는 cyclen band **바깥이 아니다.** 덱의 band 는 `cycle_target_efpd = 730.5` ± `cycle_tolerance_efpd = 40.0` → **690.5–770.5 EFPD** (report-only, 아무것도 gate 하지 않음). 731.5 는 target 에서 **1.05 EFPD** 떨어진 band 정중앙이며, 수렴한 58 기 **전부**(714.876–736.919)가 band 안이다. §6 의 λ 검증도 이 사실 위에 선다.

---

## 1. Run integrity

| | |
|---|---|
| Budget | 60 / 60 (7 waves × 8 + 4-call reserve), `--no-early-stop` |
| 수렴 | **58 / 60 (96.7 %)** — round 1 의 81.7 % 대비 **크게 개선** |
| 실패 | 2 × `non_finite_flux` — call 32 (wave 3, `control`/guided), call 48 (wave 5, `control`/random) |
| Candidate origin (수렴 58) | local 39, elite 12, guided 4, heuristic 3 |
| Slot | exploit 39, explore 14, control 5 |
| Restart provenance | `pair_ecore:MAS_RST.APRQ_11_0705.02` — **58/58**, level-3, prereg 예측대로 |
| n_cycles | 11 (일부 10), `converged_at_cap` 0 건 |
| Wall clock | 2026-08-20 18:31:09 → 20:43:07 (약 2 h 12 m), status 갱신 20:46:21 |
| Store merge | 60 new / 0 upgraded — canonical **74,657 → 74,717** |

수렴률 81.7 % → 96.7 % 는 round 1 이 "flagged, not explained" 로 남긴 항목의 자연 후속이다. 이 라운드에서 실패한 2 건이 **모두 `control` 슬롯**(guided · random, 즉 elite 로부터 가장 먼 후보)이라는 점은, round 1 의 11 건이 graded 덱 생성 자체의 결함이라기보다 **elite pool 이 in-cell 예제를 갖지 못했을 때의 탐색 분산** 쪽이었음을 시사한다. 측정치로 기록하되 인과로 주장하지 않는다.

---

## 2. R1 — 두 마크에 대한 정면 대조

| | **round 2 clean** | round 2 raw | round 1 clean | round 1 raw | SEEDCTL (2-type) | 2-type clean |
|---|---:|---:|---:|---:|---:|---:|
| **F_r** | **1.5999** | 1.5864 | 1.5993 | 1.5956 | 1.6172 | 1.6357 |
| CBC_max | **1593.80** ✅ | 1615.15 ❌ | 1597.33 ✅ | 1603.24 ❌ | 1564.36 ✅ | 1565.46 ✅ |
| F_q | 2.0042 ✅ | 1.9896 ✅ | 1.9968 | 2.0133 | 2.009 | 2.0346 |
| \|AO\| | 0.0227 ✅ | 0.0224 ✅ | 0.0261 | 0.0264 | 0.0269 | 0.0266 |
| cyclen (EFPD) | 728.652 | 731.549 | 730.503 | 730.3 | 730.497 | 730.85 |
| node_peak | 1.3358 | **1.3263** | 1.3401 | — | — | — |
| 예측 pin BU | 74.508 | 74.796 | 74.98 (실측 74.378) | 75.53 | — | 76.955 |
| 발견 콜 | **52 / 60** | 45 / 60 | 57 / 60 | 58 / 60 | 3 / 60 | 11 / 60 |

| 기준선 | round-2 **clean 1.5999** 의 델타 |
|---|---|
| round-1 clean **1.5993** (STRETCH 바) | **+0.0006 — 미달** |
| round-1 raw 1.5956 | +0.0043 |
| SEEDCTL 2-type control 1.6172 | −0.0173 |
| 2-type joint frontier 1.6357 | −0.0358 |
| **1.55 licensing gate** | **+0.0449 — 도달 못 함** |

| 기준선 | round-2 **raw 1.5864** 의 델타 |
|---|---|
| round-1 raw 1.5956 (SECONDARY 바) | **−0.0092 — 달성** |
| 1.55 gate | +0.0364 (로그의 `margin −0.036`) |

**읽는 법**: raw 축은 움직였고 clean 축은 움직이지 않았다. 이 두 문장이 같은 캠페인에서 동시에 참이라는 것이 이 라운드의 전체 결과다.

`node_peak` 은 부수적으로 이 cell 의 프로그램 최저치를 갱신했다 (raw winner 1.3263; round-1 캠페인 최소 1.3359, 중앙 1.3573). 이 축은 이 덱에서 gate 하지 않으며 headline 도 아니다 — 기록만 한다.

---

## 3. R2 — PRIMARY: gate 통과 수

| gate | round 2 (n=58) | round 1 (n=49) | |
|---|---:|---:|---|
| F_r ≤ 1.55 | **0 (0 %)** | 0 (0 %) | 여전히 유일한 완전 blocker |
| CBC ≤ 1600 | **26 (44.8 %)** | 29 (59.2 %) | **악화** — §4 |
| F_q ≤ 2.41 | 55 (94.8 %) | 49 (100 %) | 탈락 3 건은 전부 `control`/heuristic outlier |
| \|AO\| ≤ 0.30 | 58 (100 %) | 49 (100 %) | 여유 (max 0.0393) |
| **joint (CBC ∧ F_q ∧ AO)** | **26 / 58 (44.8 %)** | 29 / 49 (59.2 %) | **clean 비율 −14.4 %p** |

예측 pin ≤ 78 을 더해도 상위 core 에서 빠지는 것은 없다(§8): 5-제약 카운트도 **0/58**, F_r 단독 blocker. round 1 과 같은 결론, 다른 이유 — round 1 은 "F_r 만 남았다"였고, round 2 는 "F_r 을 더 내리자 CBC 가 따라 올라왔다"이다.

F_q 를 깬 3 건은 call 16 (F_q 2.5346, F_r 1.9605), call 24 (3.6263, 2.9063), call 56 (2.5738, 2.0091) — 셋 다 `control` 슬롯의 heuristic 후보다. 통계에 남기되 아래 상관계수는 이들을 포함/제외한 두 값을 모두 적는다.

---

## 4. R3 / CBC-WALL — 등록된 감시가 그대로 관측되었다

### 4.1 누적 joint-clean frontier (콜 순서)

| 콜 | wave | slot / origin | **F_r** | CBC | F_q | \|AO\| | cyclen |
|---:|---:|---|---:|---:|---:|---:|---:|
| 6 | 0 | explore / elite | 1.6583 | 1574.39 | 2.0628 | 0.0239 | 721.83 |
| 7 | 0 | explore / elite | 1.6060 | 1592.59 | 2.0223 | 0.0236 | 723.17 |
| **52** | **6** | **exploit / local** | **1.5999** | **1593.80** | 2.0042 | 0.0227 | 728.65 |

`1.6583 → 1.6060 → 1.5999` — **단 3 계단**, 총 이동 −0.0584, 종점 1.5999. Round 1 은 같은 정의로 5 계단 `1.6579 → 1.6486 → 1.6105 → 1.6036 → 1.5993` (콜 1 → 3 → 9 → 19 → 57)이었다.

### 4.2 누적 raw frontier (CBC 무시)

| 콜 | wave | **F_r** | CBC | clean? |
|---:|---:|---:|---:|---|
| 1 | 0 | 1.6180 | 1602.19 | ❌ |
| 3 | 0 | 1.6096 | 1602.47 | ❌ |
| 4 | 0 | 1.6038 | 1603.96 | ❌ |
| 9 | 1 | 1.5986 | 1604.95 | ❌ |
| 17 | 2 | 1.5971 | 1604.63 | ❌ |
| 21 | 2 | 1.5968 | 1604.42 | ❌ |
| 25 | 3 | 1.5882 | 1616.94 | ❌ |
| **45** | **5** | **1.5864** | **1615.15** | ❌ |

**8 계단 전부가 CBC 를 깬다.** raw frontier 는 콜 45 까지 계속 내려갔고 clean frontier 는 콜 52 에서 1.5999 에 멈췄다. 이것이 prereg §3 의 문장 — *"the CBC-clean frontier stalls above 1.55 even while the raw frontier continues past it — that specific pattern is the fingerprint"* — 의 문자 그대로의 관측이다.

### 4.3 (F_r, CBC) Pareto front — plot data

수렴 58 기에 대한 (F_r ↓, CBC ↓) 비지배 집합. **1600 ppm 선이 F_r 1.5999 에서 프론트를 자른다.**

| # | 콜 | wave | **F_r** | **CBC_max** | F_q | \|AO\| | cyclen | CBC gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 45 | 5 | **1.5864** | 1615.15 | 1.9896 | 0.0224 | 731.55 | ❌ +15.15 |
| 2 | 44 | 5 | 1.5916 | 1607.33 | 1.9893 | 0.0242 | 730.69 | ❌ +7.33 |
| 3 | 27 | 3 | 1.5925 | 1604.77 | 1.9924 | 0.0239 | 730.46 | ❌ +4.77 |
| 4 | 21 | 2 | 1.5968 | 1604.42 | 1.9980 | 0.0237 | 730.36 | ❌ +4.42 |
| **5** | **52** | **6** | **1.5999** | **1593.80** | 2.0042 | 0.0227 | 728.65 | ✅ −6.20 |
| 6 | 7 | 0 | 1.6060 | 1592.59 | 2.0223 | 0.0236 | 723.17 | ✅ |
| 7 | 54 | 6 | 1.6067 | 1589.32 | 2.0081 | 0.0229 | 727.84 | ✅ |
| 8 | 31 | 3 | 1.6081 | 1588.50 | 2.0142 | 0.0245 | 724.06 | ✅ |
| 9 | 22 | 2 | 1.6329 | 1588.38 | 2.0632 | 0.0225 | 723.08 | ✅ |
| 10 | 46 | 5 | 1.6333 | 1588.30 | 2.0310 | 0.0261 | 724.20 | ✅ |
| 11 | 47 | 5 | 1.6415 | 1586.08 | 2.0450 | 0.0262 | 724.14 | ✅ |
| 12 | 38 | 4 | 1.6458 | 1580.35 | 2.0638 | 0.0243 | 722.36 | ✅ |
| 13 | 6 | 0 | 1.6583 | 1574.39 | 2.0628 | 0.0239 | 721.83 | ✅ |

프론트 위에서 F_r 을 1.5999 → 1.5864 로 −0.0135 내리는 대가는 CBC **+21.35 ppm** 이다. **1600 ppm 게이트는 그 구간의 첫 1/3 지점에서 이미 소진된다.**

### 4.4 F_r 구간별 CBC 분포 — 벽의 모양

| F_r bin | n | CBC min | CBC p50 | CBC max | clean |
|---|---:|---:|---:|---:|---:|
| (1.58, 1.60] | 16 | 1593.80 | 1607.87 | 1617.74 | **1** |
| (1.60, 1.62] | 18 | 1588.50 | 1596.81 | 1628.56 | 11 |
| (1.62, 1.64] | 15 | 1588.30 | 1595.23 | 1621.33 | 10 |
| (1.64, 1.66] | 4 | 1574.39 | 1583.22 | 1595.30 | 4 |
| (1.66, 1.70] | 1 | 1649.93 | — | — | 0 |
| (1.70, ∞) | 4 | 1600.17 | 1684.95 | 1696.02 | 0 |

가장 낮은 F_r 구간이 **가장 높은 CBC 중앙값**을 갖고, clean 은 16 기 중 1 기뿐이다.

### 4.5 두 라운드 pooled — 벽에 대한 가장 강한 진술

| | round 1 | round 2 | **pooled (case 전체)** |
|---|---:|---:|---:|
| 수렴 core | 49 | 58 | **107** |
| CBC-clean | 29 (59.2 %) | 26 (44.8 %) | 55 (51.4 %) |
| raw min F_r | 1.5956 | **1.5864** | **1.5864** |
| clean min F_r | 1.5993 | 1.5999 | **1.5993** |
| **F_r < 1.5993 인 core** | 4 | 13 | **17** |
| **그중 CBC-clean** | 0 | 0 | **0** |
| 그 17 기의 min CBC | — | — | **1602.58 ppm** |

120 콜, 107 수렴, F_r 1.5993 아래로 내려간 17 기 중 **CBC 를 지킨 core 는 하나도 없다.** 이것이 NULL 판정의 실증 근거다.

### 4.6 상관계수 — round 1 의 메커니즘은 재현되지 않았다

| | round 1 (n=49) | round 2 (n=58) | round 2 (n=55, heuristic outlier 3 제외) |
|---|---:|---:|---:|
| r(F_r, CBC) | **+0.189** | +0.683 | **−0.081** |
| r(F_r, CBC) — clean 부분집합 | — | — | **−0.521** (n=26) |
| r(F_r, CBC) — 탐색 구름 F_r < 1.63 | — | — | **−0.206** (n=43) |
| r(mid-type, F_r) | **−0.417** | +0.339 | **+0.282** |
| r(mid-type, CBC) | **+0.638** | −0.038 | **−0.581** |
| CBC p50 | 1596.06 | 1602.16 | 1600.99 |

**두 mid-type 도즈-반응 상관이 모두 부호를 뒤집었다.** round 1 의 "mid 를 늘리면 F_r ↓ / CBC ↑" 는 round 2 에서 "mid 를 늘리면 F_r ↑ / CBC ↓" 로 나타난다. round 2 의 population 이 훨씬 높은 mid 대역(§9)에 앉아 있으므로 **같은 곡선의 반대편 가지를 본 것**일 수 있지만, 이 캠페인만으로는 그렇게 단정할 수 없다. **측정치로 기록하고, 설명하지 않는다.** round 1 §4 가 "the single most useful physical result" 라고 부른 도즈-반응은 **이 라운드에서 재현되지 않았고, 따라서 그 지위는 격하된다** — 프로그램이 그 메커니즘 위에 새 결정을 세우려면 전용 실험이 필요하다.

r(F_r, CBC) 전체값 +0.683 은 heuristic outlier 3 건이 만드는 지렛대 효과다. 정직한 수치는 n=55 의 **−0.081**(무상관)이며, **프론티어에서의 교환은 상관계수가 아니라 §4.3 의 Pareto front 가 보여준다** — 모집단 전체가 무상관이어도 경계면은 명확히 기울어져 있다.

---

## 5. R-BUDGET — 예산은 답이 아니었다

| | round 1 | round 2 |
|---|---|---|
| clean frontier 계단 수 | 5 | **3** |
| 마지막 clean 개선 콜 | 57 / 60 (거의 끝) | 52 / 60 (거의 끝) |
| clean frontier 총 이동 | −0.0586 (1.6579 → 1.5993) | −0.0584 (1.6583 → 1.5999) |
| raw frontier 마지막 개선 | 58 / 60 | 45 / 60 |
| **clean 종점** | **1.5993** | **1.5999 (개선 없음)** |

Round 1 은 "frontier 가 예산에서 멈추지 않았다 → 예산이 구속 자원"이라고 읽었고, 이 라운드는 **그 읽기를 반증한다.** 두 배의 예산(120 콜), 더 강한 elite pool(round-1 자체의 49 수렴 triple 을 직접 부모로), in-cell 을 본 챔피언(s1i)을 모두 투입하고도 **보고 가능한 frontier 는 소수점 넷째 자리에서 움직이지 않았다**(+0.0006, 노이즈 이하).

wave 별 clean best: 1.6060 / 1.6207 / 1.6179 / 1.6081 / 1.6080 / 1.6333 / **1.5999** / 1.6020 — 단조 개선이 아니라 1.60 근방의 평탄한 배회다. **"예산 부족"이 아니라 "그 축의 끝"이다.**

---

## 6. λ-objective 검증 (프로그램 규칙)

덱: `minfr_lambda = 1000.0`, `objective = min_fr_max_cycle`.

* acquisition scalar (예측 위): `cyclen_LCB − λ·F_r_UCB` (`lpopt/search/acquisition.py::score_min_fr_max_cycle`).
* campaign best-tracking (측정 라벨 위): 엄격 lexicographic `−F_r·1e6 + cyclen` (`lpopt/search/campaign.py::_campaign_objective`).

**band-internal λ-optimal vs F_r-only headline:**

| 대상 집합 | n | λ-optimal (`cyclen − 1000·F_r`) | F_r-only lexicographic | 일치? |
|---|---:|---|---|---|
| 수렴 전체 | 58 | 콜 45 — F_r 1.5864, cyclen 731.549 | 콜 45 (동일) | ✅ |
| **band-internal** (690.5–770.5) | **58 (전부)** | 콜 45 (동일) | 콜 45 (동일) | ✅ |
| band ∧ joint-clean | 26 | **콜 52 — F_r 1.5999, cyclen 728.652** | 콜 52 (동일) | ✅ |

**결론: λ 는 이 라운드에서 아무것도 바꾸지 않는다.** 수렴 58 기의 cyclen 전폭은 714.876 – 736.919 = **22.0 EFPD** 이고, λ = 1000 은 22.0/1000 = **0.022 F_r** 이하로 붙은 후보들만 재정렬한다. 실제로 clean 집합의 λ 상위 6 기(1.5999 / 1.6020 / 1.6067 / 1.6080 / 1.6081 / 1.6104)는 F_r 순서와 완전히 동일하다. **headline 을 F_r-only 로 읽어도 λ-objective 로 읽어도 같은 core 가 나온다** — cycle band 가 report-only 이고 전 core 가 band 안에 있으므로, band 제약을 켜도 결과는 불변이다.

---

## 7. 예측-측정 parity — s1i 챔피언, 이 58 행 위에서

예측 출처는 `runs/fpcamp_minfr_triple_f125_r2/waves/wave_NN/selection.json` 의 `pred_mean` (60/60 후보 전부 보존, 수렴 58 전부 매칭). 7-target head 순서는 `(f_r, cbc_max, f_q, cyclen, ao_abs, discharge_burnup, max_pin_burnup)` (`lpopt/model/net.py`, `lpopt/search/acquisition.py::_MINFR_PINBU_COL = 6`).

### 7.1 population parity (predicted − measured)

| target | n=58 bias | n=58 MAE | n=55 bias | n=55 MAE | n=26 clean bias | n=26 clean MAE |
|---|---:|---:|---:|---:|---:|---:|
| **F_r** | −0.0307 | 0.0378 | **−0.0264** | 0.0277 | −0.0281 | 0.0296 |
| **CBC_max** (ppm) | −7.88 | 9.43 | **−5.98** | 7.61 | −2.05 | 3.92 |
| F_q | −0.0433 | 0.0535 | −0.0407 | 0.0418 | −0.0464 | 0.0472 |
| **cyclen** (EFPD) | +0.631 | 1.416 | **+0.920** | 1.238 | +1.435 | 1.435 |
| \|AO\| | +0.0032 | 0.0036 | +0.0035 | 0.0035 | +0.0038 | 0.0038 |

(n=55 는 heuristic control outlier 3 건 제외.) 상위 10 기(측정 F_r 기준): F_r bias **−0.0166** (MAE 0.0166), CBC **−8.92** (MAE 8.92), cyclen **+0.203** (MAE 0.293).

**s1i 는 F_r 을 체계적으로 낙관한다** — 전 구간에서 예측이 측정보다 0.017 – 0.031 낮다. 최소 예측 F_r 은 1.5706 이었고 **F_r ≤ 1.55 로 예측된 후보는 0/60**; 그럼에도 60 콜이 소진된 것은 acquisition 이 least-infeasible-first 로 순위를 매기기 때문이다(설계대로). `p_feas` 는 min 0.000 / p50 0.174 / max 0.229 로 **0.3 을 넘은 후보가 하나도 없었고**, 실현 feasible 도 0/58 — reliability 는 "항상 낮게 불렀고 항상 맞았다".

origin 별 F_r bias: local −0.0284 (n=39), elite −0.0179 (n=12), guided −0.0333 (n=4), heuristic −0.1083 (n=3, MAE 0.222).

### 7.2 수용기준 — 최적점에서

프로그램 수용 바: **cyclen ≤ 1 EFPD, CBC ≤ 1 ppm** (at the optimum).

| 최적점 | cyclen err | 판정 | CBC err | 판정 |
|---|---:|---|---:|---|
| **joint-clean optimum (콜 52, F_r 1.5999)** | **+0.882 EFPD** | ✅ PASS | **+1.000 ppm** | ✅ PASS (경계선) |
| raw / λ optimum (콜 45, F_r 1.5864) | −0.224 EFPD | ✅ PASS | **−13.864 ppm** | ❌ **FAIL** |

콜 52 전 축 (pred → meas): F_r 1.5745 → 1.5999 (−0.0254), CBC 1594.80 → 1593.80 (+1.000), F_q 1.9617 → 2.0042 (−0.0425), cyclen 729.534 → 728.652 (+0.882), |AO| 0.0277 → 0.0227 (+0.0050), 예측 pin 74.508, `p_feas` 0.2124.

**모집단 전체로는 바를 통과하지 못한다**: cyclen ≤ 1 EFPD **29/58**, CBC ≤ 1 ppm **6/58**, 둘 다 **3/58**. |F_r err| ≤ 0.01 은 **6/58**, ≤ 0.005 는 **0/58**. 즉 **"최적점 한 점에서는 바를 만족하지만 그것은 국소적 우연에 가깝다"** — 특히 CBC 는 1 ppm 바에서 통과율 10.3 % 다. 이 라운드의 parity 는 수용기준을 **최적점에서만** 만족한다고 읽어야 하며, 그 최적점이 CBC 벽 **안쪽**(clean)이라는 점은 우연이 아니라 CBC bias 가 clean 부분집합에서 가장 작기 때문(−2.05 ppm)이다.

### 7.3 계산 불가로 남는 두 축 — 정직하게

* **node_peak**: `pred_mean` 은 7-target global head 이며 `node_peak` / `map_cov` 는 여기에 **없다**(map/flat head 소관, `_FLAT_TARGETS`). `selection.json` 은 이 축의 예측을 **저장하지 않는다** → **parity 계산 불가.** 측정치만 존재: node_peak min 1.3263 / p50 1.3497 / max 2.4442, map_cov min 0.2714 / p50 0.2803 (58/58 라벨, `maps.npz` 에 3 배열/행).
* **col 5 (`discharge_burnup`)**: 예측은 있으나(63.7 – 64.1 대역) `optimize` 경로는 이 열을 라벨하지 않는다 — 58/58 null → **parity 불가.** `max_assembly_burnup`(63.768 – 70.610)과 수치가 가깝지만 **다른 물리량** (core AVERAGE vs assembly max)이므로 대리로 쓰지 않는다.

---

## 8. R-PIN — 이 라운드에는 실측이 없다

**`max_pin_burnup` 은 58/58 null. `max_rod_avg_burnup` 도 58/58 null.** 이 캠페인은 `enable_pin_burnup` 재실행(MAS_PPI 수확)을 하지 않았고, round-2 의 58 개 pattern 중 **pin 실측이 있는 store 행과 일치하는 것은 0 건** 이다. 따라서 prereg §8 의 R-PIN("상위 5 기 실측")은 **이 문서에서 답할 수 없다.** 예측만 보고한다.

| | round 2 (예측, s1i) | round 1 (실측 5 기) |
|---|---|---|
| 전체 예측 pin BU | min 74.434 · p50 74.635 · max 80.610 | — |
| 상위 8 (raw F_r) | 74.615 – 74.882 | — |
| 상위 8 (joint-clean) | **74.434 – 74.672** | — |
| joint-clean winner | **74.508** | 74.98 예측 → **74.378 실측** |
| ≤ 78 gate | **55 / 58** (탈락 3 건 = heuristic outlier) | 5 / 5 |
| **실측 band** | **없음** | **74.156 – 75.582, 5/5 PASS** |

**전이 추정(측정 아님)**: `data/reports/pinbu_wave_results_20260820.md` §3 은 **동일 cell · 동일 feed · 동일 챔피언 s1i** 에 대해 pin-head 정확도를 실측했다 — `HGD569_f125_3type` slice **n=5, bias −0.03, MAE 0.67, sd 0.77, 95 % CI [−0.60, +0.59]**. 이 bias 를 round-2 clean 상위 8 기의 예측 (74.43 – 74.67)에 적용하면 실측 기대값은 **74.4 – 74.7 GWd/tU**, 80 GWd/tU licensing limit 대비 **약 5.3 GWd/tU 여유**로 나온다. 이는 **전이 추정이며 이 라운드의 실측이 아니다.** round 1 이 남긴 한계 ("predicted, not measured")는 이 라운드에서도 **해소되지 않았다.**

정의는 `pinbu_definition_20260820.md` §2 를 그대로 따른다: 한계치 80 GWd/tU 는 **핀 axial peak**(`max_pin_burnup`, MAS_PPI `BPIN` 3-D 최댓값)에 걸며, 덱의 `minfr_pin_bu_limit = 78` 은 그 80 에서 2.0 GWd/tU model margin 을 뺀 **예측값 screening** 이다.

---

## 9. R-GRADE — mid-type 분율이 크게 이동했다

full-core 환산(rot61 cache key, 중심 +1 · 그 외 +4, 합 125 검증 완료). 이 계산은 round-1 의 공표 분포 `4:11, 8:8, 12:10, 16:17, 20:1, 24:2` 와 winner `57/20/48` 을 **정확히 재현**하여 방법을 검증한 뒤 적용했다.

| | round 1 (n=49) | round 2 (n=58) |
|---|---|---|
| 전체 min / p50 / max | 4 / **12** / 24 | 12 / **24** / 48 |
| clean min / p50 / max | 4 / **8** / 20 | 16 / **28** / 48 |
| 분포 | 4:11, 8:8, 12:10, 16:17, 20:1, 24:2 | 12:1, 16:5, 20:7, 24:18, 28:12, 32:1, 36:4, 40:4, 44:5, 48:1 |
| **clean winner** hot/mid/cold | **57 / 20 / 48** (mid 16.0 %) | **49 / 28 / 48** (mid **22.4 %**) |
| raw winner hot/mid/cold | 61 / 16 / 48 | 57 / 24 / 44 (mid 19.2 %) |

* **floor 로 붕괴하지 않았다** — round 1 과 같은 결론. `require_all_fresh_types` 가 남긴 4-assembly 하한 근처는 이 라운드에서 아예 방문되지 않았고(최소 12), 전체 중앙은 **2 배(12 → 24)** 로 올라갔다.
* clean winner 의 mid 분율 16.0 % → **22.4 %**. prereg §4 가 "worth registering even though it decides nothing on its own" 이라고 미리 등록한 **material shift 가 실제로 일어났다.**
* 다만 이 이동은 **elite pool 교체와 완전히 교락(confounded)** 되어 있다 — round-2 의 부모 32 기는 round-1 의 최상위 boards 다. "탐색이 mid 를 더 좋아하게 되었다"와 "더 좋은 부모가 mid 가 많았다"를 이 런은 분리하지 못한다.
* §4.6 의 부호 역전과 함께 읽으면: round 2 는 **round 1 과 다른 mid 대역에서 작동했고**, 그 대역에서 mid 는 F_r 을 **낮추지 않았다**(r = +0.28). 이는 mid 분율의 최적이 **round-1 대역(16 % 근방) 쪽**임을 시사하지만, clean frontier 가 두 대역에서 사실상 같은 곳(1.5993 / 1.5999)에 멈췄다는 사실이 그 시사를 실질적으로 무력화한다 — **mid 분율은 이 벽을 옮기지 못한다.**

---

## 10. Wave fine-tune gate — 이번엔 눈을 뜨고 있었고, 8/8 거부했다

Round 1 은 in-cell holdout 이 0 행이라 gate 가 **판정 불능**(`explore`/NaN) 이었다. Round 2 는 round-1 의 49 수렴 행이 holdout 에 들어와 **처음으로 점수를 매겼다** — 그리고 **8 waves 전부 거부**했다 (accepts 0, rejects 8).

| wave | 거부 사유 | champion → challenger cumulative skill |
|---:|---|---|
| 0 | `ao_abs` skill 퇴행 0.206 < 0.264 − ε | 0.4736 → 0.4547 |
| 1 | cumulative skill 악화 0.454 < 0.490 − ε | 0.4896 → 0.4538 |
| 2 | `cyclen` 퇴행 0.649 < 0.676 − ε | 0.5137 → 0.5446 |
| 3 | `cyclen` 퇴행 0.645 < 0.673 − ε | 0.4975 → 0.5517 |
| 4 | `cbc_max` 퇴행 0.679 < 0.718 − ε | 0.5312 → 0.5784 |
| 5 | `cbc_max` 0.677 < 0.725 − ε; `cyclen` | 0.5471 → 0.5820 |
| 6 | `cbc_max` 0.686 < 0.738 − ε; `cyclen` | 0.5483 → 0.5985 |
| 7 | `cbc_max` 0.678 < 0.737 − ε | 0.5307 → 0.5655 |

wave 0 챔피언 holdout skill: F_r 0.774, CBC 0.735, F_q 0.656, cyclen 0.694, \|AO\| 0.264. wave 7: F_r 0.722, CBC 0.737, F_q 0.587, cyclen 0.668, **\|AO\| −0.247**(붕괴). 주목할 점은 wave 2 – 6 에서 **cumulative skill 은 challenger 가 더 높았는데도** per-target 퇴행 규칙(`gate_epsilon = 0.02`)이 거부했다는 것 — gate 는 설계대로 보수적으로 작동했다. **개선은 전부 base s1i + BO refinement 에서 왔고, in-campaign 업데이트는 한 번도 채택되지 않았다.**

---

## 11. 등록된 처분 (REGISTERED DISPOSITION)

### 11.1 어떤 분기가 발화했는가

**prereg §2 의 NULL 이 발화했다.** 등록 문구 그대로:

> *"the 3-type F_r–CBC frontier closes **above 1.55** — i.e. CBC becomes the binding wall before F_r can reach the licensing bar, so no amount of further F_r-only search at this cell/case/objective can produce a full-feasible core."*

발화 근거(전부 사전 등록된 관측):

1. F_r ≤ 1.55 도달 core **0/58**, 두 라운드 pooled **0/107** (§3, §4.5).
2. CBC-clean frontier 가 **1.5999 에서 정지**했고 raw frontier 는 **1.5864 까지 계속 진행** — prereg §3 이 명명한 "fingerprint" (§4.1 – 4.2).
3. pooled 로 F_r < 1.5993 인 17 기 중 CBC-clean **0 기**, 최소 CBC **1602.58 ppm** (§4.5).
4. 2 배 예산 · 강화 elite · in-cell 챔피언으로도 clean frontier **불변** (+0.0006) (§5).

동시에 **STRETCH 는 미달, SECONDARY 는 달성**이다. SECONDARY 는 prereg 가 "weaker, reported labelled, never as headline" 로 못박았으므로 headline 이 되지 않는다.

### 11.2 prereg 가 지시하는 다음 단계

> *"If this happens, the registered next step is **not** a third round at this cell but a hand-off to the fuel-design/blanket axis (a different lever than radial grading of the same three-type alphabet)."*

**따라서 이 cell 에서의 round 3 은 등록상 금지된다.** 다음 지렛대는 **fuel-design / blanket 축** — 같은 3-type alphabet 의 radial grading 이 아니라 **fuel type 정의 자체**(축방향 zoning / blanket / Gd 로딩)를 바꾸는 축이다. 이 문서는 그 축의 설계를 하지 않는다; 등록된 hand-off 를 기록할 뿐이다.

### 11.3 2026-08-29 — 목적함수가 F_xy 로 전환되었다 (별건, 상위 결정)

같은 날, 프로그램의 최적화 대상이 **F_r (FRP) → F_xy (MASTER `FXYP`), hard limit `max F_xy ≤ 1.65`** 로 전환되었다 (사용자 결정 2026-08-29, 구속력 있음; 설계서 `data/reports/fxy_switch_design_20260829.md`).

이 결과보고에 대한 함의는 **딱 두 가지**다:

1. **본 3-type F_r-only 라인은 종결(closed)이다.** §11.2 의 hand-off 는 여전히 유효하지만, 이 cell 에서 **F_r 을 목적함수로 하는** 어떤 재시작도 더는 프로그램의 현행 목표가 아니다.
2. **이 cell 에서 재시작이 일어난다면 새 목적함수(F_xy ≤ 1.65) 아래에서 일어난다.** 그 경우 본 문서의 마크 · frontier · NULL 은 **F_r 축에 대한 종결 기록**으로 남고, 새 축의 판정은 새 pre-registration 이 정한다.

**본 문서는 F_xy 를 분석하지 않는다.** 근거: 이 캠페인의 58 행에 F_xy 라벨이 **없다** — canonical store 스키마(39 열)에 `f_xy` 열 자체가 존재하지 않으며, F_xy(FXYP)는 `MAS_SUM` 이 아니라 `MAS_OUT` 의 P2D edit 에만 나온다 (설계서 §0.1-1). 이 캠페인의 `MAS_OUT` 보존 여부는 로컬 아티팩트만으로 확인할 수 없으므로 **retro-label 가능 여부도 이 문서에서는 미확인으로 남긴다.** 라벨이 없는 축에 대해 수치를 만들지 않는다.

---

## 12. 정직한 메모

1. **STRETCH 를 0.0006 차이로 놓쳤다.** 이 숫자를 "거의 달성"으로 읽으면 안 된다 — 두 라운드 pooled 로 F_r 1.5993 아래에 CBC-clean 이 **한 기도 없다**는 §4.5 가 본질이고, 0.0006 은 그 벽 위에서의 노이즈다.
2. **raw 개선(−0.0092)은 전부 CBC 게이트 밖에서 발생했다.** raw frontier 8 계단 전부가 CBC 위반이다(§4.2). raw 수치를 인용할 때는 반드시 CBC 위반 사실을 함께 인용해야 한다.
3. **round 1 의 mid-type 도즈-반응이 재현되지 않았고, 부호가 뒤집혔다**(§4.6). round 1 이 "가장 유용한 물리 결과"라고 부른 항목의 지위는 격하된다. flagged, not explained.
4. **pin BU 는 여전히 예측이다**(§8). round 1 과 동일한 한계가 그대로 남아 있으며 이 라운드는 그것을 해소하지 않았다. §8 의 74.4 – 74.7 기대값은 **전이 추정**이지 실측이 아니다.
5. **수용기준은 최적점에서만 만족한다**(§7.2). 모집단 기준 CBC ≤ 1 ppm 은 6/58 (10.3 %) 이다. "s1i 가 이 cell 을 맞춘다"고 일반화하면 안 된다.
6. **`state.json → best_overall` 이 또 CBC 위반 core 를 표면화한다** (1.5864 / CBC 1615.15). round 1 · 2-type 전 캠페인과 동일한 알려진 결함. **winner 를 인용하기 전에 CBC/F_q/AO 필터를 반드시 적용할 것.** `lpopt report` 의 best-patterns 표 역시 cycle distance 로 정렬하므로 F_r winner 를 보여주지 않는다(이 런은 feasible 0 이라 표 자체가 비었다).
7. **`e_split` 는 58/58 NaN.** round 1 §10-6 과 동일 — `optimize` 경로의 전역 특성이지 triple 고유 결함이 아니다. featurizer 가 serve time 에 `g_e_split` 을 pattern 에서 계산하므로 downstream 영향 없음.
8. **교락은 이 라운드에서도 해소되지 않았다.** seed · champion · elite pool 이 동시에 바뀌었다(prereg §2 에 사전 등록). 다만 이 라운드의 **결론(NULL)은 교락에 둔감하다** — "더 좋은 seeding 이 clean frontier 를 못 옮겼다"는 진술은 seeding 이 좋아졌다는 사실을 오히려 강화 근거로 쓴다.
9. **`[optimize][DEPRECATED]` 배너는 정상**이다(`min_fr_max_cycle` 은 은퇴한 production mode, 재현 / A-B 용으로만 유지). §11.3 의 전환은 이 배너가 앞서 예고하던 방향과 같은 방향이다.

---

## 13. Provenance

| item | value |
|---|---|
| deck | `fpcamp_minfr_TRIPLE_f125_r2_199.inp`, 15,488 B, sha256 **`B5683D4FB2F32E9E218DF0D6551928766C2FFF26B34EC78B1EC7E6A893FE0116`** — prereg §5 등록값과 **일치** |
| model | `data/models/s1i` (cond_v8, 20 globals, 5 members), gate `data/reports/gate_s1i.json` PASS |
| case | `P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24` (alias `S3_T1_S5`), feed 125, paramA, e_core 5.675821 |
| run log | `fpcamp_minfr_TRIPLE_f125_r2_199_out.log`, 1,583 B, sha256 `11B193F4040460F3546CC4FC8E5BEDC43294C312F38150233498461186408145`; `_rc.txt` = **0** |
| **labels** | `runs/fpcamp_minfr_triple_f125_r2/labels.jsonl`, **60 행**, 139,313 B, sha256 **`7363DFAE5ED81AC055922A8083AD77E022BAE3A9EA4C505ED3DDFB14ED8678D1`** |
| state / status | `state.json` 13,833 B sha256 `955EDE04EB9639F1EEF0F9FE6E41ACFDAC453412FDD461932CEC716DCACA87D1` · `status.json` 2,062 B sha256 `62090174EEF0EBACC9F3191C319927C4DDDFC337DA800F9128456DCD37EE181B` |
| waves | `waves/wave_00..07/{selection.json, results.json}` — 예측 `pred_mean` 60/60 후보 보존 (§7 parity 의 출처) |
| sdm / mtc | `sdm_mtc_targets.jsonl`, **58 행**, 34,162 B, sha256 `4EF75CCD4AAB751F5CD16358B08A194F8DB98A48E2563B4048FEF115376F3E1B` (ts 18:31:09 → 20:43:07) |
| figures | `figures/{budget_curve, ga600_overlay, parity, p_feas_reliability}.png` |
| logs | `logs/events.jsonl` (8 wave 레코드) |
| **store (pre-launch)** | 74,657 행, sha256 **`FBDBAFBADB11BDF37EB0FF7F776A5D037BE040F514AA1BF663E515B699D614C6`** — prereg §9 등록값과 **일치**(백업 `records.parquet.bak_pre_TRIPLEf125r2_20260829` 로 재계산 확인) |
| **store (post-merge)** | `data/store/records.parquet` **74,717 행** (+60), 22,243,803 B, sha256 `D2196B5EC0F53D59432DA071DC063CA35FB54BA832BA2A0B0356A5D9535F4B0F` |
| campaign rows | `campaign == "fpcamp_minfr_triple_f125_r2"` → **60 행 / 58 수렴**; case pooled 120 행 / 107 수렴 |
| **maps** | `data/store/maps.npz` **75,634 엔트리** (211,835,960 B), sha256 `008A7D32143D3AD39674D1D355CB146667C360CDC92C96F5548BE1363548DDB2`; 신규 **+174 = 58 행 × 3** (`<key>`, `<key>__axial`, `<key>__traj`), `maps_key` 58/58 매칭 |
| store 백업 | `records.parquet.bak_pre_TRIPLEf125r2_20260829` (22,217,067 B, 74,657 행) · `maps.npz.bak_pre_TRIPLEf125r2_20260829` (211,103,842 B, 75,460 엔트리) |
| launch / status 스크립트 | `launch_fpcamp_TRIPLE_f125_r2_199.ps1` · `status_fpcamp_TRIPLE_f125_r2_199.ps1` · `run_fpcamp_minfr_TRIPLE_f125_r2_199.bat` |
| fleet | **HOST_199 only.** HOST_198 / HOST_181 / HOST_238 untouched. |

---

## 14. 문서 상태

* 이 문서는 `tripletype_f125_r2_prereg_20260820.md` 의 마크에 대해서만 판정한다. 사후에 새 마크를 만들지 않았고, 등록된 마크를 완화하지도 않았다.
* **본 3-type / feed 125 / F_r-only 라인은 이 문서로 CLOSED 된다** — 등록된 NULL 발화(§11.1), 등록된 hand-off(§11.2), 2026-08-29 목적함수 전환(§11.3)의 세 가지 독립적 이유 모두에 의해.
* 재현: `python -m lpopt optimize --input fpcamp_minfr_TRIPLE_f125_r2_199.inp
--run-dir runs/fpcamp_minfr_triple_f125_r2 --no-early-stop` (HOST_199, 위 deck / store sha256 하에서).
