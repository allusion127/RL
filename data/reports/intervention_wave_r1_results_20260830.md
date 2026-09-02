# Campaign A round 1 — Causal Move Atlas (`intervention_wave_r1`) — RESULTS

**Run 2026-08-30 on HOST_199, 수확·merge 2026-08-30.** 5 cells × 20 parents × 8 moves = **800 paid chain**, `rc = paramA=0 ga80_resume=0`, MASTER wall **8.94 h** (32,183 s, 예산 14.4 h). Run dir `runs/intervention_wave_r1/intervention_<cell>/`.
Pre-registration: `data/reports/intervention_wave_r1_prereg_20260829.md` (+ STAMP 2026-08-30 12:0x store re-stamp, DEVIATION 2026-08-30 17:0x ga80 `--allow-fallback` 누락 → E1E2_f109 거부 → `check_restart` 게이트 하 재개).
Plan manifest `data/design/intervention_wave_r1.json` (sha `F82CE029…FE20D`, 등록값 그대로).
선행: `ablation_wave_results_20260815.md`, `batchswap_wave_results_20260815.md`, `policy_v2_results_20260817.md`.

prereg §5 의 가설 H1–H4, §6 의 P1–P6, §8 의 G1–G5 를 **전부** 아래에서 답한다. 답할 수 없는 것은 답할 수 없다고 적는다.

---

## 0. HEADLINE

| 마크 | 등록된 요구 | 측정 | 판정 |
|---|---|---|---|
| **H1** leakage 부호 보편성 | 5/5 cell 에서 `fresh_relocate` 쌍 대조 `mean(out−in) d_cyclen < 0` **이고** 각 cell p<0.05 | 부호 **5/5 음수**, 유의 **3/5** (E1E2_f109 8.4e-6 · N1N2_f113 0.0095 · E1E2_f121 0.038; HGD569 0.065 · T6T4 0.636). pooled −1.2353, sign 60/195, **p=8.1e-8** | **부분 성립 — 등록된 바(bar) 미달**. 반증 조건(부호 뒤집힘 + p<0.05)은 **어느 cell 에서도 발생하지 않음** → 규칙은 **cell-conditional** 로 강등, 폐기 아님 |
| **H1b** T6T4 자체 재현 | ablation 의 −1.7415 [−2.1336, −1.3342], sign 0/10 을 parent 20·쌍 40 으로 재현 | **−0.5171 [−1.020, −0.037], sign 18/40, p=0.636.** 두 CI 는 **겹치지 않는다** | **🔴 재현 실패.** ablation 의 효과크기는 **3.37× 과대추정**으로 철회. dose slope 도 −8.37 → **−1.031 [−3.88, +1.68]** (같은 cell·같은 척도) |
| **H2** `batch_swap` 618 가지 밖 일반성 | signed 2 cell 모두에서 `mean(out−in) d_f_r < 0` | T6T4 −0.0179 (8/19, p=0.648, CI [−0.054,+0.017]). **HGD569 는 20/20 쌍이 out=in 으로 정확히 0** — 구조적 공집합 | **판정 불가 + 일반성 부정.** T6T4 를 618/625 가지로 쪼개면 부호가 **반대**(−0.0349 vs +0.0055), 둘 다 무의미. batchswap wave 의 −0.1442 [−0.1888,−0.1013] 는 **어디에서도 재현되지 않음** → **branch/cell 특이, policy feature 전역 사용 불가** |
| **H3** F_xy / F_r 반응 분기 | `fresh_relocate` 는 두 축 부호 일치 **&** 최소 1개 class 에서 불일치 | 부호 일치 4/5 cell(비-null cell 전부). 불일치: E1E2_f109 `batch_flip\|neutral` (F_xy −0.0027 vs F_r +0.0100). **전달계수 F_xy:F_r 이 move family 별로 갈림** — 농축-반경 계열 1.23–1.42, Gd/격자 계열(`batch_swap\|neutral`) **0.55–0.73** | **✅ 성립.** "F_xy 는 F_r 의 단조 변환" 은 round 1 범위에서 **기각**. `predict_fxy` 의 proxy 경로는 Gd/격자 move 에 대해 **정당화되지 않음** |
| **H4** 축퇴 cell 의 `batch_swap` 이 F_xy 를 움직이는가 | 3 축퇴 cell 각각에서 `batch_swap\|neutral` vs `rewire_swap\|neutral` 의 d_F_xy 분포가 다름 (parent 평균 부호검정 p<0.05) | E1E2_f109 **+0.0712, 20/20, p=1.9e-6** · E1E2_f121 **+0.0689, 17/20, p=0.0026** · N1N2_f113 +0.0071, 12/20, p=0.503 | **✅ 2/3 cell 성립.** `d_fresh_enr_r_center ≡ 0` 인데 F_xy 가 0.07 움직인다 → **현재 feature set 에 Gd/lattice 기술자가 빠져 있다는 것이 측정으로 확립** |
| **P1–P6** | §6 | 전부 충족 (§1) | **✅** |
| **G1–G5** | §8 | G1✅ G2✅ G3 1.000✅ G4✅ G5✅ | **✅ corpus 편입 조건 충족** |
| **🔴 신규 (등록 밖)** | — | **HGD569_f125 의 vs-parent delta 전체가 공통모드 +34.71 EFPD / +0.098 F_r / +0.106 F_xy 만큼 이동해 있다.** 중립 대조군 `rewire_swap` 이 그 오프셋을 그대로 보여준다 | **cell 격리 필요** (§7) |

**한 문장으로**: outward fresh loading 이 cycle length 를 깎는다는 것은 **5개 cell 전부에서 같은 부호로 재현되었고 pooled 로 p=8×10⁻⁸ 이지만**, ablation 이 그 규칙을 발견한 바로 그 cell(T6T4_f121)에서 효과크기는 **3.4배 작았고 유의하지 않았다**. 그리고 이 wave 는 등록하지 않았던 것을 하나 얻었다 — **농축 가중 반경 기술자가 전혀 보지 못하는 Gd/lattice 배치 레버가 F_xy 를 0.07 움직인다**(H4).

---

## 1. Run integrity — P1–P6

| cell | kit | chains | converged | error | F_xy parsed / conv | restart provenance (run == parent) | wall |
|---|---|---:|---:|---:|---:|---|---:|
| `T6T4_f121` | paramA | 160 | 159 (99.4%) | 1 | 159/159 = **100%** | `native:MAS_RST.APRQ_10_0615.11` (L0) | 6,205 s |
| `HGD569_f125` | paramA | 160 | 153 (95.6%) | 7 | 153/153 = **100%** | `pair_ecore:MAS_RST.APRQ_11_0705.02` (L3) | 6,093 s |
| `E1E2_f121` | ga80 | 160 | 160 (100%) | 0 | 160/160 = **100%** | `native:MAS_RST.APRQ_11_0635.19` (L0) | 6,973 s |
| `E1E2_f109` | ga80 | 160 | 160 (100%) | 0 | 160/160 = **100%** | `pair_feed:MAS_RST.APRQ_11_0615.88` (L2) | 6,471 s |
| `N1N2_f113` | ga80 | 160 | 159 (99.4%) | 1 | 159/159 = **100%** | `pair_feed:MAS_RST.APRQ_11_0677.23` (L2) | 6,441 s |
| **합** | | **800** | **791 (98.9%)** | **9** | **791/791 = 100%** | **불일치 0 / 800** | **32,183 s = 8.94 h** |

| # | 기준 | 임계 | 측정 | |
|---|---|---|---|---|
| P1 | 수렴 chain 중 F_xy 파싱 비율 | ≥95% | **100.0%** (791/791), cell 별로도 전부 100% | ✅ |
| P2 | 전체 수렴률 | ≥90% | **98.9%** | ✅ |
| P3 | harness error / cell | ≤5% | HGD569 **4.4%**(7/160), N1N2·T6T4 0.6%, E1E2 0.0% | ✅ (HGD569 는 여유 0.6%p) |
| P4 | 두 멤버 모두 수렴한 쌍 | ≥200/240 | **230/240** (`fresh_relocate` 191, `batch_swap` 39) | ✅ |
| P5 | restart 불일치 chain | 0 | **0/800** | ✅ |
| P6 | 벽시계 | ≤20 h | **8.94 h** (실측 cadence **89.5 chains/h**, 계획 가정 55.6 의 1.61배) | ✅ |

실패 9건은 **전부 `non_finite_flux`** 이며 harness/staging/디스크 실패는 0건이다. HGD569 에 7건이 몰린 것(4.4%)은 이 cell 이 프로그램 실현가능 행 0개인 OOD arm 이라는 사실과 정합적이지만, **인과로 주장하지 않고 측정치로만 기록한다.**

---

## 2. F_xy 3중 교차검증 — sidecar × store × 독립 MAS_OUT 재스캔

| 대조 | 매칭 | max \|Δ\| | 1e-4 초과 |
|---|---:|---:|---:|
| `fxy_sidecar.jsonl` (analyze 입력) ↔ `data/store/records.parquet` | 791 | **0** | **0** |
| sidecar ↔ `fxy_backfill_199_intervention_wave_r1_20260830.csv` (독립 재스캔) | 790 | **0** | **0** |
| `f_xya` 동일 대조 | 790 | **0** | **0** |
| 부수 확인: `child_f_r` / `child_cyclen` ↔ store | 791 | **0** | — |

**불일치 0건.** 요구된 1e-4 허용오차 안이 아니라 **비트 동일**이다.

조인 키 주의: backfill 의 `digest16` 은 `record_id[:16]` 이 **아니라** `sha256(pack_pattern)[:16]`(`lpopt/tools/backfill_fxy.digest_of_packed`, `vendor/masterrl/domain.py:187`)이다. `record_id` 로 조인하면 0건 매칭되므로 **pattern digest 로 조인해야 한다.**

**개수 차 2건의 내역** (재스캔 801행 = sane 790 + insane 11):

- 9건: `digest16` 자체가 없는(`nonfinite`, digest NaN) 행 — plan 의 **error 9 chain 과 정확히 일치**.
- 1건 `057b6269afe341dc` (`superseded`) — T6T4 의 error chain `5dfbcff3…`, sidecar 도 `no_result`. **양쪽 일치.**
- 1건 `682428e95e91f1d7` (`first_cycle`) — T6T4 `7fc1b15f…` (`fresh_relocate|outward`). **sidecar/store 는 f_xy=1.8151 을 들고 있고 재스캔은 보존된 MAS_OUT 을 최종 평형 사이클로 인정하지 않아 값을 내지 않았다.** 값의 불일치가 아니라 **재스캔이 더 보수적인 분류를 한 1건**이다. 이 행은 §3 이후의 F_xy 표에서 `T6T4 fresh_relocate` 의 n_F_xy 가 F_r 대비 작은 이유 중 하나다.

---

## 3. 조건부 효과 — per-cell (1차 통계량: 쌍 내부 out−in, 정확 이항 부호검정)

### 3.1 `fresh_relocate` (191쌍 / 등록 200쌍)

| cell | n쌍 | d_cyclen (EFPD) | d_F_r | d_F_xy | d_node_peak | leakage 매개 부호 |
|---|---:|---|---|---|---|---|
| `E1E2_f109` | 40 | **−3.1616** [−4.264, −2.063] · 6/40 · **p=8.4e-6** | **−0.1061** [−0.140,−0.074] · 10/40 · **0.0022** | **−0.1283** [−0.168,−0.086] · 11/40 · **0.0064** | **−0.0250** · 11/40 · **0.0064** | **정합** (4축 전부 음수) |
| `E1E2_f121` | 40 | **−0.2825** [−0.598, +0.014] · 13/40 · **0.038** | **−0.0674** [−0.092,−0.044] · 4/40 · **1.9e-7** | **−0.0677** [−0.103,−0.033] · 12/40 · **0.017** | **−0.0378** · 13/40 · **0.038** | **정합** |
| `N1N2_f113` | 39 | **−1.0122** [−1.692, −0.355] · 11/39 · **0.0095** | −0.0512 [−0.090,−0.011] · 17/39 · 0.522 | −0.0771 [−0.129,−0.021] · 14/39 · 0.108 | −0.0186 · 18/39 · 0.749 | **정합** (부호 4/4 음수, 평탄도는 무의미) |
| `HGD569_f125` | 36 | −1.1930 [−1.810, −0.617] · 12/36 · 0.065 | −0.0065 [−0.033,+0.019] · 17/36 · 0.868 | **+0.0003** [−0.035,+0.037] · 18/36 · 1.000 | −0.0082 · 13/36 · 0.132 | **탈동조** — cycle 손실은 있는데 평탄도 이득이 **0** |
| `T6T4_f121` | 40 | −0.5171 [−1.020, −0.037] · 18/40 · 0.636 | **+0.0133** [−0.028,+0.051] · 21/40 · 0.875 | **+0.0112** [−0.043,+0.064] · 20/36 · 0.618 | **+0.0155** · 23/40 · 0.430 | **역전** — outward 가 cycle 도 깎고 평탄도도 **나쁘게** 한다 |
| **POOLED** | **191/195** | **−1.2353** · 60/195 · **p=8.1e-8** | **−0.0443** · 69/195 · **5.4e-5** | **−0.0546** · 75/191 · **0.0037** | **−0.0149** · 78/195 · **0.0064** | — |

**leakage 매개 부호는 cell 마다 같지 않다**: ga80 E1E2 두 cell 과 N1N2 는 "outward → 누설↑ → cycle↓ ∧ peaking↓" 이라는 정합적 매개를 보이고, HGD569 는 **cycle 만 잃고 평탄도 이득이 없으며**, T6T4_f121 은 **평탄도까지 나빠진다**. 즉 ablation atlas 를 만든 그 cell 이 5개 중 매개 부호가 **역전된** 유일한 cell 이다.

### 3.2 parent-FE dose-response `d_cyclen ~ d_fresh_enr_r_center` (`fresh_relocate`)

| cell | slope (EFPD/unit) | 95% CI | n |
|---|---:|---|---:|
| `E1E2_f109` | **−30.285** | [−33.98, −26.80] | 80 |
| `HGD569_f125` | −9.967 | [−14.78, −5.47] | 73 |
| `N1N2_f113` | −9.518 | [−14.40, −4.08] | 79 |
| `E1E2_f121` | −2.542 | [−5.36, −0.05] | 80 |
| `T6T4_f121` | **−1.031** | [−3.88, **+1.68**] | 80 |
| **POOLED** | **−8.842** | [−11.45, −6.65] | 392 |

부호는 **5/5 음수**, cell 간 크기는 **29배** 벌어진다. prereg §3 대로 dose 척도는 cell 간 비교 불가이므로 **부호와 cell 내부 순위만** 읽는다. 단 `T6T4_f121` 은 ablation 과 **같은 cell·같은 척도**이므로 like-for-like 비교가 되며, ablation 의 **−8.37** 은 이번 CI [−3.88, +1.68] **밖**이다. batchswap wave 의 −21.19 도 마찬가지다. → **T6T4 의 dose 상수는 세 번 다 다르게 나왔다. 상수를 보고하지 않는다.**

### 3.3 `batch_swap` — signed cell 2개

| cell | n쌍 | d_F_r | d_F_xy | d_cyclen | d_node_peak |
|---|---:|---|---|---|---|
| `T6T4_f121` | 19 | −0.0179 [−0.054,+0.017] · 8/19 · 0.648 | −0.0519 [−0.096,−0.009] · 6/17 · 0.332 | −0.0757 · 9/19 · 1.000 | −0.00006 · 10/19 · 1.000 |
| `HGD569_f125` | 20 | **0.000000** · sign_n=**0** | **0.000000** · 0 | **0.000000** · 0 | **0.000000** · 0 |
| **branch split (T6T4)** ~618 | 11 | −0.0349 · 4/11 · 0.549 | −0.0780 · 3/11 · 0.227 | −0.2206 · 4/11 · 0.549 | — |
| **branch split (T6T4)** ~625 | 8 | **+0.0055** · 4/8 · 1.000 | −0.0039 · 3/6 · 1.000 | **+0.1235** · 5/8 · 0.727 | — |

### 3.4 vs-parent 한계효과 — 중립 대조군(`rewire_swap|neutral`) 보정 전/후, pooled

`rewire_swap` 은 fresh 집합도 batch label 도 건드리지 않아 `d_fresh_enr_r_center ≡ 0` 인 등록된 중립 대조군이다. parent 별 그 값을 빼면 cell 별 공통모드(§7)가 제거된다.

| move_class · dir | d_F_xy raw / **adj** | d_F_r raw / **adj** | d_node_peak raw / **adj** | d_cyclen raw / **adj** |
|---|---|---|---|---|
| `fresh_relocate` · inward | 0.1876 / **0.1575** | 0.1448 / **0.1168** | 0.0641 / **0.0451** | +6.267 / **−0.232** |
| `fresh_relocate` · outward | 0.1337 / **0.1031** | 0.1011 / **0.0728** | 0.0498 / **0.0305** | +5.143 / **−1.480** |
| `batch_swap` · inward | 0.1130 / **0.0427** | 0.0815 / **0.0234** | 0.0387 / **0.0084** | +17.97 / **−0.049** |
| `batch_swap` · outward | 0.0883 / **0.0200** | 0.0726 / **0.0157** | 0.0390 / **0.0094** | +17.48 / **−0.067** |
| `batch_swap` · neutral | 0.0563 / **0.0491** | 0.0771 / **0.0670** | 0.0767 / **0.0636** | −0.208 / **−0.284** |
| `batch_flip` · inward | 0.0658 / **−0.0045** | 0.0520 / **−0.0058** | 0.0321 / **−0.0015** | +15.72 / **−1.222** |
| `batch_flip` · outward | 0.0792 / **0.0129** | 0.0626 / **0.0064** | 0.0333 / **0.0080** | +18.95 / **+0.724** |
| `batch_flip` · neutral | 0.0086 / **0.0014** | 0.0125 / **0.0024** | 0.0060 / **−0.0071** | −0.009 / **−0.084** |
| `rewire_swap` · neutral | 0.0309 / 0 (기준) | 0.0289 / 0 | 0.0197 / 0 | +7.065 / 0 |

**raw 열은 그대로 읽으면 안 된다.** `batch_swap`/`batch_flip` 의 signed stratum 은 signed cell 2개(그중 HGD569)에서만 나오므로 raw 의 +17~19 EFPD 는 전부 HGD569 의 공통모드이고, 보정 후에는 **−0.05 ~ +0.72 EFPD** 로 사라진다. `analyze` 는 이 보정을 하지 않으므로(§11-2) **보고에는 adj 열을 쓴다.**

### 3.5 burn-state 층 (`effects_by_burn_state`)

| stratum | n | 비고 |
|---|---:|---|
| `once` | 413 (`fresh_relocate` 313 + `rewire_swap` 100) | 계획 420, 손실 7 |
| `fresh` | 299 (`batch_swap` 199 + `batch_flip` 100) | 계획 300 |
| `twice_plus` | 79 (`fresh_relocate` 전용) | 계획 80 — **f109/f113 에서만** 공급 |

`fresh_relocate` 의 방향 대조는 burn-state 에 따라 **한 자릿수 배로** 갈린다:

| burn_state | outward d_cyclen | inward d_cyclen | out−in |
|---|---:|---:|---:|
| `once` (n=313) | +7.512 | +7.825 | **−0.313** |
| `twice_plus` (n=79) | **−4.157** | +0.033 | **−4.190** |

`twice_plus` 에서 outward fresh loading 의 cycle 손실이 **13배** 크다. 다만 prereg §12-3 이 사전 공개한 대로 **`twice_plus` 는 f121 에서 구조적으로 0**(`depth2_edges = 0`)이고 79개 전부 f109/f113 에서 왔으므로, round 1 에서 **burn-state 효과와 feed 효과는 분리되지 않는다.** 이것은 관측이지 인과 분해가 아니다. → r2 설계 항목(§14).

---

## 4. H1 — 판정: 부분 성립, cell-conditional 로 강등

등록된 바는 "5/5 cell 에서 음수 **이고** 각각 p<0.05" 였다. 측정은 **부호 5/5, 유의 3/5** 다. 등록된 반증 조건(어느 cell 에서 부호가 뒤집히고 그 cell 이 p<0.05)은 **발생하지 않았다.** 따라서:

- "outward fresh loading 은 cycle length 를 깎는다" 는 **폐기되지 않는다.** pooled 191쌍에서 −1.2353 EFPD, sign 60/195, **p=8.1e-8** 이고 5개 cell 부호가 모두 같다.
- 그러나 **크기는 cell 마다 −0.28 에서 −3.16 EFPD 까지 11배** 벌어지고, 두 cell 에서는 cell 내부만으로 노이즈와 구분되지 않는다.
- prereg §8 G4 는 "최소 3개 cell 에서 같은 부호" 를 요구한다 → **d_cyclen 5/5, d_F_r 4/5, d_F_xy 3/5** 로 충족. 정책 v3 에 pooled feature 로 들어갈 수 있으나 **cell 이질성이 §3.1 에 보고된 상태로만** 들어간다(G5).

### 4.1 T6T4 자체 재현 실패 — 등록된 가능성이 발화했다

prereg §5-H1 은 "**자체 재현이 실패할 가능성을 명시적으로 등록한다**" 고 적었다. 발화했다.

| | ablation wave (2026-08-15) | intervention r1 (본 wave) |
|---|---|---|
| 설계 | parent 10 · 쌍 20 | parent 20 · **쌍 40** |
| `mean(out−in) d_cyclen` | **−1.7415** | **−0.5171** |
| 95% CI | [−2.1336, −1.3342] | [−1.020, **−0.037**] |
| sign test | **0/10, p=0.002** | **18/40, p=0.636** |
| dose slope | −8.37 EFPD/unit | **−1.031** [−3.88, +1.68] |

두 CI 는 겹치지 않는다(−1.3342 vs −1.020). **효과크기 3.37× 과대추정.** 이는 batchswap wave 가 ablation 의 parent 당 n=4 stratum 추정치를 5.8× 과대추정으로 판정한 것과 **동일한 실패 모드의 세 번째 관측**이다. 방향 부호는 살아남고 크기는 매번 죽는다.

> **읽는 법.** 이번 값이 참이고 ablation 이 거짓이라는 뜻이 아니라, **parent 10개 표본의 stratum 평균은 이 문제에서 신뢰구간이 광고된 것보다 훨씬 넓다**는 뜻이다. 등록된 결론: **cell 내부 stratum 평균은 parent ≥20 이전에는 크기를 보고하지 않는다.**

---

## 5. H2 — 판정: 판정 불가(HGD569 구조적 공집합) + 일반성 부정(branch split)

### 5.1 HGD569_f125 `batch_swap` 은 방향 정보를 0 비트 산다

20개 parent 전부에서 `batch_swap` 의 outward 자식과 inward 자식이 **동일한 수렴 노심**을 냈다:

- 4개 스칼라 반응(`f_r`, `cyclen`, `f_xy`, `node_peak`)이 **인쇄 정밀도 전 자리 동일** — 20/20 쌍.
- nodal power map(`map_*.npy`, shape (4,9,9))이 **유한 노드 전부에서 max\|Δ\| = 0** — 20/20 쌍. (대조군: 같은 wave 의 T6T4 `batch_swap` 은 2.84–4.61, HGD569 `fresh_relocate` 는 15.3–35.1)
- 그런데 두 자식의 `pattern` 은 **다르고**(69 슬롯 중 5장), `move_tag` 도 다르고(예: `bs:34<->59` vs `bs:5<->46`), `record_id` 도 다르다. dedup 은 통과했다.
- 두 fresh type 은 라이브러리상 **다른 물건**이다: `P6253Z1G06N24` (u_avg 5.7861, gd_wt 6.0, kinf0 1.15638) vs `P6253Z2G10N24` (5.6023, 10.0, 1.12485), alias `S3` vs `S5`.

즉 **prereg §2.5 의 축퇴 census(= fresh type 의 `u_avg_enrichment` 동일 여부)로는 잡히지 않는 두 번째 종류의 방향 축퇴**가 존재한다: 열거자가 고른 outward 스왑과 inward 스왑이 **동일 노심을 실현**하는 슬롯 기하 축퇴다. paid 예산 800 중 **40 chain (5.0%)** 이 여기에 쓰였다.

→ H2 의 두 signed cell 중 하나가 통째로 비었으므로, 등록된 반증 조건("두 cell 의 부호가 서로 반대")은 **평가 불가**다. 이것은 analyze 결함이 아니라 **plan 단계 census 의 범위 문제**이며 §11-1 에 보고한다(코드는 건드리지 않았다).

### 5.2 618 가지 밖 일반성 — 부정

`T6T4_f121` 의 parent 20개는 cyclen 616.16–625.21 EFPD 에 걸쳐 있다(~618 branch 11, ~625 branch 9). `batch_swap` 쌍 대조를 branch 로 쪼개면 부호가 **가지마다 반대**이고 둘 다 무의미하다(§3.3). 그리고 batchswap wave 의 213-chain 추정치 **−0.1442 [−0.1888, −0.1013]** 는 ~618 가지에서조차 재현되지 않는다(−0.0349, CI 는 그 구간을 포함하지 않는다).

**등록된 결론 그대로 적용**: `batch_swap` 의 방향 효과는 **branch/cell 특이적이며 policy feature 로 전역 사용 불가**다. 이 결론은 prereg §5-H2 의 반증 조항이 예고한 문구다.

---

## 6. H3 · H4 — F_xy 가 F_r 의 단조 변환이 아니라는 것을 두 방향에서

### 6.1 H3 — 성립

**첫째 절(`fresh_relocate` 두 축 부호 일치)**: 효과가 null 이 아닌 4개 cell 에서 전부 일치한다(E1E2_f109 −/−, E1E2_f121 −/−, N1N2_f113 −/−, T6T4 +/+). HGD569 만 F_xy +0.0003 / F_r −0.0065 로 갈리지만 **둘 다 0 근방이고 p=1.000 / 0.868** 이므로 부호를 읽을 값이 아니다.

**둘째 절(최소 1개 class 에서 불일치)**: 충족한다.

- 직접 부호 불일치: `E1E2_f109` `batch_flip|neutral` — d_F_xy **−0.0027** (parent 평균 부호검정 5/20, **p=0.041**) vs d_F_r **+0.0100** (7/20, p=0.263). 같은 개입이 F_xy 를 내리고 F_r 을 올린다.
- 그보다 강한 것은 **전달계수의 분기**다. 중립 보정 후 F_xy:F_r 비:

| move family | class | F_xy : F_r |
|---|---|---:|
| 농축-반경 | `fresh_relocate` inward / outward | **1.35 / 1.42** |
| 농축-반경 | `batch_swap` signed inward / outward | 1.83 / 1.28 |
| **Gd/격자 (축퇴)** | `batch_swap\|neutral` | **0.73** |
| **Gd/격자 (축퇴)** | `batch_flip\|neutral` | **0.55** |
| 중립 기준 | `rewire_swap\|neutral` | (0, 기준) |
| 쌍 대조 확인 | `fresh_relocate` pooled out−in | −0.0546 / −0.0443 = **1.23** |
| 쌍 대조 확인 | `batch_swap` T6T4 out−in | −0.0519 / −0.0179 = **2.90** |
| H4 대조 확인 | `bs−rw` E1E2_f109 / E1E2_f121 | **0.58** / 1.08 |

농축을 반경 방향으로 옮기는 move 는 **F_xy 를 F_r 보다 1.2–1.4배 더** 움직이고, 농축을 옮기지 않고 Gd 배치만 바꾸는 move 는 **F_r 을 F_xy 보다 더** 움직인다(비 0.55–0.73). 두 계열 사이에서 전달계수가 **약 1.9–2.6배 갈린다.**

> **정책 v3 결론**: "F_xy 는 F_r 의 단조 변환" 은 round 1 범위에서 **기각**된다. `predict_fxy` 의 proxy 경로(F_r 열의 아핀 변환)는 **농축-반경 move 에 대해서는 잠정적으로 쓸 수 있으나 Gd/격자 move 에 대해서는 부호와 크기를 모두 틀린다.** f_xy head 없는 체크포인트로 Gd move 를 랭킹하면 안 된다.

### 6.2 H4 — 2/3 cell 성립. 등농축 pair 에서 Gd 배치가 F_xy 를 움직인다

두 class 모두 `d_fresh_enr_r_center ≡ 0` 이다. `rewire_swap` 은 연소 이력만 재배선하고 `batch_swap` 은 Gd 배치를 바꾼다. parent 평균 차 `batch_swap|neutral − rewire_swap|neutral`:

| cell | n parents | d_F_xy | d_F_r | d_node_peak | d_cyclen |
|---|---:|---|---|---|---|
| `E1E2_f109` | 20 | **+0.0712** · 20/20 · **p=1.9e-6** | +0.1218 · 20/20 · 1.9e-6 | +0.1189 · 20/20 · 1.9e-6 | −0.7146 · 0/20 · 1.9e-6 |
| `E1E2_f121` | 20 | **+0.0689** · 17/20 · **p=0.0026** | +0.0637 · 18/20 · 4.0e-4 | +0.0502 · 19/20 · 4.0e-5 | −0.0629 · 11/20 · 0.824 |
| `N1N2_f113` | 20 | +0.0071 · 12/20 · 0.503 | +0.0156 · 16/20 · 0.012 | +0.0216 · 15/20 · 0.041 | −0.0738 · 11/20 · 0.824 |

**E1E2 두 cell 에서 결정적으로 성립**(f109 는 20/20 만장일치), `N1N2_f113` 에서는 F_xy 축이 null 이다(F_r/node_peak 축은 유의). 크기는 cell 의존이다.

> **이것이 round 1 의 가장 큰 신규 사실이다.** 농축 가중 반경 기술자가 **정확히 0** 만큼 변하는 개입이 F_xy 를 **0.07** 움직인다. F_xy 하드 한계가 1.65 이고 이 wave 의 parent F_xy 가 1.53–1.83 인 것을 감안하면 0.07 은 **정책적으로 큰 값**이다. prereg §5-H4 가 적은 대로: **관측 코퍼스로는 만들 수 없는 결론**이며, 현재 feature set 에 **Gd/lattice 기술자가 없다는 것을 측정으로 확립**한다.

---

## 7. 🔴 등록 밖 발견 — HGD569_f125 의 공통모드 baseline 이동

중립 대조군이 잡아낸 것이다. cell 별 `rewire_swap|neutral` 의 vs-parent delta(= 개입이 반경 농축 분포를 전혀 바꾸지 않았을 때의 delta):

| cell | n | d_cyclen | d_F_r | d_F_xy | d_node_peak |
|---|---:|---:|---:|---:|---:|
| `E1E2_f109` | 20 | +0.023 | +0.009 | +0.004 | +0.016 |
| `E1E2_f121` | 20 | +0.030 | +0.007 | +0.007 | +0.008 |
| `N1N2_f113` | 20 | +0.174 | +0.014 | +0.011 | +0.016 |
| `T6T4_f121` | 20 | +0.043 | +0.016 | +0.027 | +0.010 |
| **`HGD569_f125`** | 20 | **+35.053** | **+0.098** | **+0.106** | **+0.049** |

**중립 개입이 cycle length 를 35 EFPD 늘릴 수는 없다.** HGD569 의 4개 move class 전부가 같은 오프셋을 진다(`batch_flip` +34.6/+35.3, `batch_swap` +34.9, `fresh_relocate` +34.0/+35.1, `rewire_swap` +35.1). parent cyclen 729.6–736.6 vs child 761.6–770.6 이고, cell 전체 평균 **+34.71 ± 1.71 EFPD** 다. 즉 **이 wave 의 child 평가 경로가 store 의 HGD569 parent 라벨을 만든 경로와 다르다.**

독립 확증: 블라인드 s1i 예측의 `d_cyclen` 편향이 전체 **+6.98 EFPD** 인데 **HGD569 를 빼면 +0.095 EFPD (MAE 0.53)** 이다. 모델은 나머지 4개 cell 의 parent→child 를 정확히 맞혔고 HGD569 에서만 34 EFPD 를 놓쳤다 — 물리 효과가 아니라 **cell 국소 데이터/자산 문제**라는 뜻이다.

가장 유력한 후보는 이 cell 이 유일하게 쓰는 **synth deck** `data/design/synth_decks/P6253Z1G06N24_P6253Z2G10N24/MAS_INP_cy12.inp` (sha 앞 16 `cddba86904810b7c`, prereg §4 에서 이번 wave 를 위해 도입)이며, store 의 HGD569 라벨은 그 이전 자산으로 만들어졌다. **restart 는 범인이 아니다** — run 과 parent 모두 `pair_ecore:MAS_RST.APRQ_11_0705.02` 로 일치한다(P5 0건). **여기서는 후보로만 기록하고 인과로 주장하지 않는다.** 확정에는 parent 재평가가 필요하다(§14-5).

### 이 발견이 무엇을 무효화하고 무엇을 무효화하지 않는가

- **무효화하지 않음**: 모든 쌍 내부 (out − in) 대조. 오프셋은 parent 공통모드이므로 **정확히 상쇄된다.** H1·H2·H3 의 1차 통계량은 전부 안전하다.
- **무효화하지 않음**: H4(같은 parent 안의 두 class 차)도 상쇄되며, H4 는 HGD569 를 쓰지 않는다(축퇴 3 cell 만).
- **무효화함**: HGD569 의 **모든 vs-parent 한계효과**(§3.4 raw 열, `effects_by_cell` 의 HGD569 행), 그리고 **`steps.parquet` 에 들어간 HGD569 160행의 `d_cyclen`/`d_f_r`/`d_node_peak`/`d_cbc_max` 절대값**.
- **처방**: HGD569_f125 의 160 corpus 행은 **정책 v3 의 pooled 회귀 학습에서 제외**하고 진단 슬라이스로만 쓴다(G5 조항). prereg §7 이 이미 이 cell 을 **영구 prospective holdout 후보**로 표시해 둔 것이 그대로 발효된다.

---

## 8. 사전 등록된 블라인드 예측 채점 (`data/design/intervention_wave_r1_s1i_pred.csv`, sha `5135B59E…F3A0`)

s1i 에는 f_xy head 가 없고 F_xy 예측은 `pred_f_xy_source='proxy'` 다(prereg §5·§12-5).

| 절대 수준 (child, n=791) | bias | MAE | RMSE | Spearman |
|---|---:|---:|---:|---:|
| F_r | +0.0295 | 0.0521 | 0.0686 | **+0.808** |
| cyclen | +7.321 | 8.008 | 16.804 | **+0.989** |
| node_peak | −0.0084 | 0.0355 | 0.0481 | +0.730 |
| F_xy (proxy) | +0.0321 | 0.0641 | 0.0877 | +0.768 |

| delta (parent→child) | bias | MAE | Spearman | 부호 일치 |
|---|---:|---:|---:|---:|
| d_F_r | +0.0453 | 0.0539 | +0.676 | 0.848 |
| d_cyclen (전체) | +6.979 | 7.330 | +0.325 | 0.650 |
| **d_cyclen (HGD569 제외, n=638)** | **+0.095** | **0.530** | — | — |
| d_node_peak | +0.0147 | 0.0309 | +0.657 | 0.784 |
| d_F_xy (proxy) | +0.0547 | 0.0663 | +0.656 | 0.830 |

### 등록된 관전 포인트 — 예고대로 갈렸다

prereg §5 는 "s1i 는 `fresh_relocate` 의 방향 부호는 맞히고 `batch_swap` 의 방향 부호는 틀릴 것" 이라고 적었다.

| class | 반응 | 예측 out−in | 측정 out−in | |
|---|---|---:|---:|---|
| `fresh_relocate` | cyclen | **−1.175** | **−1.235** | ✅ 부호 일치, 크기도 5% 이내 |
| `fresh_relocate` | F_r | −0.0132 | −0.0443 | ✅ 부호 일치 |
| `fresh_relocate` | F_xy | −0.0161 | −0.0546 | ✅ 부호 일치 |
| `batch_swap` | **cyclen** | **+0.211** | **−0.037** | **❌ 부호 반대 — 예고 적중** |
| `batch_swap` | F_r | −0.0001 | −0.0087 | ✅ (사실상 0 예측) |

s1i 의 `fresh_relocate` 방향 예측은 **크기까지 맞다**(−1.175 vs −1.235). `batch_swap` 은 부호를 틀렸지만 측정 자체가 유의하지 않으므로(§5) **모델의 실패라기보다 그 stratum 이 아직 측정되지 않았다**는 쪽으로 읽는다.

---

## 9. E1E2_f109 fallback-restart 재개 caveat (DEVIATION 2026-08-30 17:0x)

`run_intervention_wave_r1_199.bat` 의 ga80 호출에 `--allow-fallback` 이 빠져 ablation runner 의 자체 가드가 `E1E2_f109` 를 거부하고 프로세스가 종료되었다(rc 미기록). paramA 2 cell + `E1E2_f121` = 480 chain 은 이미 완료된 상태였고, `resume_intervention_wave_r1_ga80_199.bat` 이 `E1E2_f109`·`N1N2_f113` 를 `--allow-fallback` + `intervention_wave.check_restart` 게이트 하에서 재개했다.

**중요 — delta 는 restart-consistent 하다.** `E1E2_f109` 의 parent 는 **전부** `pair_feed:MAS_RST.APRQ_11_0615.88` 에서 라벨되었고, 이 wave 의 child **160/160 도 동일한** `pair_feed:MAS_RST.APRQ_11_0615.88` 에서 돌았다(run 로그 `parents label(s) ['pair_feed:MAS_RST.APRQ_11_0615.88']`, restart sha 앞 16 `7e42617be33d5aca`). 즉 **parent 와 child 가 같은 연소 이력을 공유하므로 parent/child delta 는 restart 변화와 섞이지 않는다.** ablation 가드가 경고한 것은 `fallback_level != 0` 자체이고, 그것은 "이 cell 의 native restart 가 아니다" 는 뜻이지 "parent 와 child 가 다르다" 는 뜻이 아니다 — 후자를 검사하는 것이 `check_restart` 이고 그 결과는 **0/800 불일치**다(P5).

같은 구조가 `N1N2_f113`(`pair_feed`, level 2)과 `HGD569_f125`(`pair_ecore`, level 3)에도 적용된다. 재개는 `ablation_wave._done` 계약대로 cell 별 jsonl 에서 이루어졌고(`done 0` 에서 시작), 재개된 두 cell 은 각각 160 chain 을 완주했다(수렴 160/160, 159/160).

**남는 한계(수치가 아니라 해석의)**: 이 세 cell 의 **절대 수준**은 자기 자신의 native restart 가 아닌 연소 이력 위에 서 있다. 따라서 cell 간 절대 F_xy/cyclen 비교는 하지 않고 **cell 내부 대조만** 보고한다 — 이 문서의 모든 headline 수치가 그 규칙을 따른다.

---

## 10. Corpus append — `data/policy/steps.parquet`

```
python intervention_wave.py corpus --plan data/design/intervention_wave_r1.json \
    --store data/store/records.parquet --steps data/policy/steps.parquet
```

| | before | after |
|---|---:|---:|
| rows | **28,097** | **28,897** (+800) |
| cols | 80 | **80** (schema drift 0) |
| sha256 | `F6B877BB…1BCF8` (prereg §9 등록값 일치) | `8E91AAC5…1B32D` |
| bytes | 9,263,631 | 9,514,593 |

**백업 (append 전, 요청 경로)**: `E:/lpopt_data/5_RL/backups/steps.parquet.bak_pre_intervention_r1_20260830` — sha256 `F6B877BBD8C71705FC41A11BB36C764E1613E25790930F91E308BEB4FB71BCF8`, 9,263,631 bytes, **원본과 비트 동일**. (`corpus` 자신도 `data/policy/steps.parquet.bak_pre_intervention_wave_r1` 를 추가로 남겼다.)

### lineage_source 분포 (after)

| lineage_source | rows |
|---|---:|
| `sa_mocha` | 21,766 |
| `lpopt_genome` | 5,748 |
| `batchswap_enum_625` | 220 |
| `batchswap_enum` | 213 |
| **`intervention_T6T4_f121`** | **160** |
| **`intervention_HGD569_f125`** | **160** |
| **`intervention_E1E2_f121`** | **160** |
| **`intervention_E1E2_f109`** | **160** |
| **`intervention_N1N2_f113`** | **160** |
| `ablation_paramA` | 150 |

### 무결성

| 항목 | 값 |
|---|---|
| `(parent_record_id, child_record_id)` 중복 — 신규 내부 | **0** |
| 동일 — 파일 전체 | **0** |
| 신규 child 가 기존 corpus 에 이미 있음 | **0** |
| `single_move` | **800 / 800 = 1.000** (cell 별로도 160/160) |
| `cross_cell` | **False 800/800** — 모든 edge 가 **same-cell**, cross-cell edge **0** |
| `both_converged` | True 791 / False 9 |
| target null (`d_f_r`/`d_cyclen`/`d_cbc_max`/`d_node_peak`) | **9행** — harness error chain. 등록된 정책(검토서 §7.8-6, 실패 레코드 삭제 금지)대로 **남긴다.** 학습 시 `both_converged` 로 거른다 |
| 신규 parent 가 기존 corpus 의 child 이기도 함 | **56** → parent-blocked split 은 **lineage_source 를 가로질러** `parent_record_id` 로 블록해야 한다 |

### move_class × direction 균형 — 계획 vs 달성

| move_class | outward 계획/달성 | inward 계획/달성 | neutral 계획/달성 | 합 |
|---|---|---|---|---:|
| `fresh_relocate` | 200 / **200** | 200 / **200** | 0 / 0 | 400 |
| `batch_swap` | 40 / **40** | 40 / **40** | 120 / **120** | 200 |
| `rewire_swap` | 0 / 0 | 0 / 0 | 100 / **100** | 100 |
| `batch_flip` | 19 / **19** | 21 / **21** | 60 / **60** | 100 |
| **합** | **259 / 259** | **261 / 261** | **280 / 280** | **800** |

**계획 대비 100% 달성, shortfall 0.** cell 별 배분도 매니페스트와 일치한다(paramA cell: `batch_swap`·`batch_flip` signed; ga80 3 cell: `batch_swap`·`batch_flip` 전량 `neutral_degenerate`).

단, **§5.1 의 발견으로 "달성" 의 의미가 하나 줄어든다**: HGD569 의 `batch_swap` signed 40행은 라벨상 outward/inward 이지만 **정보량이 0** 이다(out 자식과 in 자식이 동일 노심). 실효 signed `batch_swap` 대조는 **T6T4 의 19쌍뿐**이다.

### G1–G5 판정

| gate | 조건 | 측정 | |
|---|---|---|---|
| G1 | P1–P5 충족 | §1 전부 충족 | ✅ |
| G2 | schema drift 0 ∧ `(parent, child)` 중복 0 | 80→80 cols, 중복 0 | ✅ |
| G3 | cell 별 `single_move` ≥ 0.98 | **1.000** (5/5 cell) | ✅ |
| G4 | 최소 3 cell 에서 `fresh_relocate` 쌍 대조 동부호 | d_cyclen **5/5**, d_F_r 4/5, d_F_xy 3/5 | ✅ |
| G5 | pooled 학습 전 cell 이질성 보고 | §3.1·§3.2·§7 | ✅ |

**결론: 800행 전부 정책 v3 코퍼스에 편입 가능.** 단 §7 에 따라 `intervention_HGD569_f125` 160행은 pooled 회귀 fold 에서 제외하고 진단 슬라이스로만 쓴다.

---

## 11. 도구 소견 (수정하지 않고 보고만 — `lpopt/` 및 wave 스크립트 무편집)

1. **`plan` 의 방향 축퇴 census 가 슬롯 기하 축퇴를 놓친다.** §2.5 census 는 fresh type 의 `u_avg_enrichment` 동일 여부만 본다. `HGD569_f125` 의 `batch_swap` 은 두 type 의 농축이 **다른데도**(5.786 vs 5.602) outward/inward 형제가 **동일 노심**을 실현했다(20/20, 스칼라 4축 및 power map 비트 동일). 제안: plan 시점에 두 형제의 **canonical core digest 를 비교**하고 일치하면 `pair_role='neutral_degenerate'` 로 재분류. 미적용 시 r2 에서도 signed 예산의 일부가 그대로 소실된다(round 1 실측 40 chain = 5.0%).
2. **`analyze` 에 cell baseline 진단이 없다.** `effects_by_cell`/`effects_pooled` 는 vs-parent delta 를 그대로 낸다. `HGD569_f125` 의 +34.71 EFPD 공통모드가 pooled `batch_swap`/`batch_flip` cyclen 을 **+17~19 EFPD 로 오염**시키고 있는데(중립 보정 후 −0.05~+0.72), 도구는 이를 표시하지 않는다. 제안: `neutral_control_offset` 표(cell × 반응, `rewire_swap|neutral` 평균)와 그것을 뺀 `effects_*_adj` 표를 추가. 등록된 중립 대조군이 이미 설계에 있으므로 계산은 공짜다.
3. **`corpus` 가 만드는 80열 스키마에 F_xy 가 없다.** `steps.parquet` 에는 `child_f_xy`/`d_f_xy` 열이 존재하지 않는다(`mine_policy_corpus.build_steps` 가 F_xy 이전 스키마). 이 wave 의 **1차 반응이 코퍼스에 도달하지 않는다.** 정책 v3 는 `child_record_id` → `data/store/records.parquet.f_xy` 조인 또는 `data/reports/intervention_wave_r1_rows.csv` 로 라벨을 복원해야 한다. 같은 이유로 `burn_state` stratum 도 코퍼스에 없다(`burnt_periph_dir` 등 다른 축만 있음) — 매니페스트에서 조인 필요.
4. **backfill 조인 키가 문서 밖에서는 함정이다.** `digest16` 은 `sha256(pack_pattern)[:16]` 이며 `record_id[:16]` 으로 조인하면 **0건 매칭**된다(§2). 재현하는 사람을 위해 여기에 적어 둔다.

(1)–(4) 중 **어느 것도 이번 결과를 무효화하지 않는다.** (1) 은 예산 손실, (2)(3)(4) 는 보고·후처리에서 본 문서가 이미 메웠다.

---

## 12. Provenance

| artefact | sha256 | bytes | 비고 |
|---|---|---:|---|
| `intervention_wave.py` | `4D545E814A050953703769A684019F855D9C1944F16EE5D782E5B4B0AA88FDC6` | 75,031 | **prereg §9 등록값 일치** |
| `data/design/intervention_wave_r1.json` | `F82CE02943893D5132FFEC9321ADFA1757C3CF6DC30624CF392E93C2D86FE20D` | 2,660,532 | **일치** |
| `data/design/intervention_wave_r1_s1i_pred.csv` | `5135B59E75F176C39F450FD05C8E489A1F5B100AA307A146687AA69FB4D8F3A0` | 417,800 | **일치** |
| `ablation_wave.py` | `1B94C7128F41685B6B3852527AD8FF6625414F781009AD1ED9F17CAE5F9280C1` | 40,237 | **일치, 무편집** |
| `ablation_analyze.py` | `58D6F779D2F35A4CE24FFE489B96CEC52E2A8939EEC3214CA24A3B67C285E95E` | 19,216 | **일치, 무편집** |
| `mine_policy_corpus.py` | `78B798FA3537744F5E0B51026DA6E4530A7B1F7C12A9BAB511B23DF549A7C9AB` | 91,468 | **일치, 무편집** |
| `data/store/fuel_types.parquet` | `FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137` | 64,343 | **일치** |
| `data/store/records.parquet` (merge 후) | `73701E33F07291E17609BA30D025E2A5B7A423FEB69F08D23DE4EC23EBE0C85F` | 22,780,281 | 76,693행, `intervention_*` 800행. prereg pin `4CFF270B…` 및 STAMP `255F0E41…` 이후 **다시 이동** — 오늘의 merge 반영 |
| `data/policy/steps.parquet` (append 전) | `F6B877BBD8C71705FC41A11BB36C764E1613E25790930F91E308BEB4FB71BCF8` | 9,263,631 | **prereg §9 등록값 일치** |
| `data/policy/steps.parquet` (append 후) | `8E91AAC520D9A84B63B39B27CC3B9B1519AE7E3A75EB2EB51267EB952A61B32D` | 9,514,593 | 28,897행 |
| `E:/lpopt_data/5_RL/backups/steps.parquet.bak_pre_intervention_r1_20260830` | `F6B877BB…1BCF8` | 9,263,631 | append 전 백업, 비트 동일 |
| `data/reports/fxy_backfill_199_intervention_wave_r1_20260830.csv` | `C71B005AD9ECE14B5D4669CF5B63653443548B29695FA44149D9C180B587B97D` | 176,855 | 801행 / sane 790 |

### 이 문서가 만든 분석 산출물 (`data/reports/`)

| file | sha256 (앞 16) | bytes |
|---|---|---:|
| `intervention_wave_r1_effects_by_cell.csv` | `C6E5860FF1938D50` | 6,880 |
| `intervention_wave_r1_effects_pooled.csv` | `295B86F2F3CA8483` | 2,519 |
| `intervention_wave_r1_effects_by_burn_state.csv` | `06786076E5937C64` | 3,076 |
| `intervention_wave_r1_paired_by_cell.csv` | `60135A0D136B3DFE` | 2,669 |
| `intervention_wave_r1_paired_pooled.csv` | `71A6D783FDB9407A` | 732 |
| `intervention_wave_r1_parent_blocked_signs.csv` | `74796C9DDE66E4AB` | 9,841 |
| `intervention_wave_r1_rows.csv` | `4A34610803079670` | 450,055 |

재현 명령:

```bash
python intervention_wave.py analyze --plan data/design/intervention_wave_r1.json \
    --run-dir runs/intervention_wave_r1 --out-dir data/reports
python intervention_wave.py corpus --plan data/design/intervention_wave_r1.json \
    --store data/store/records.parquet --steps data/policy/steps.parquet
```

---

## 13. 정책 v3 에 대한 의미

### 13.1 어떤 move class 가 이제 cell 별 균형 잡힌 interventional label 을 가지는가

| move_class | 방향 라벨 | 균형 잡힌 interventional label 을 가진 cell | 학습 사용 등급 |
|---|---|---|---|
| `fresh_relocate` | outward / inward, parent 내 dose-matched 쌍 | **5/5 cell** (T6T4 40쌍 · HGD569 36 · E1E2_f121 40 · E1E2_f109 40 · N1N2_f113 39 = **191쌍 382행**) | **pooled 가능** (G4 통과). cell 별 크기 이질성(§3.1)을 feature 로 노출하거나 cell-conditioned 로 학습. HGD569 는 delta 절대값 제외 |
| `batch_swap` (signed) | outward / inward | **실효 1 cell** — T6T4 19쌍. HGD569 20쌍은 정보량 0(§5.1) | **pooled 금지.** cell/branch-conditioned 진단용. 618/625 가지에서 부호가 반대(§5.2) |
| `batch_swap` (neutral, Gd/격자) | 방향 없음, `rewire_swap` 대조로 식별 | **3/3 축퇴 cell** (E1E2_f121·E1E2_f109·N1N2_f113, 각 40행 = **120행**) + 대조군 60행 | **신규 feature 축으로 사용 권장.** F_xy 에 대한 유의 효과 2/3 cell(§6.2) |
| `rewire_swap` | 구조적 중립 (`d_fresh_enr_r_center ≡ 0`) | **5/5 cell**, 각 20행 = **100행** | **중립 baseline 으로 사용.** 이 wave 가 HGD569 오프셋을 잡아낸 것이 그 가치의 증명 |
| `batch_flip` | parent 간 무작위화 (paramA signed, ga80 neutral) | signed 2 cell(T6T4 20 · HGD569 20), neutral 3 cell(60행) | **약함.** `d_fresh_enr_mass ≠ 0`(최대 0.735)이라 반응성 보존이 깨져 leakage 대조로 읽으면 안 됨(prereg §2.5) |

### 13.2 feature / 목적함수 측면의 처방

1. **F_xy head 를 실어야 한다.** 이 wave 는 F_xy 라벨 791개를 만들었지만 `steps.parquet` 이 F_xy 열을 갖지 않는다(§11-3). 먼저 조인으로 라벨을 붙이고, `predict_fxy` 의 proxy 경로는 **Gd/격자 move 에 대해 비활성화**한다 — 전달계수가 그 계열에서 0.55–0.73 으로 뒤집힌다(§6.1).
2. **Gd/lattice 기술자를 추가한다.** H4 가 측정으로 확립한 결손이다: `d_fresh_enr_r_center ≡ 0` 인 개입이 F_xy 를 0.07 움직인다. 후보 축은 `fuel_types.parquet` 의 `n_gd`/`gd_wt`/`gd_u_enr`/`zone_pin_count` 를 반경 가중한 모멘트다.
3. **`fresh_relocate` 방향 feature 는 cell-conditioned 로 들어간다.** 부호는 전역이지만 크기는 11–29배 이질적이다(§3.1·§3.2). 전역 상수 계수는 학습시키지 않는다.
4. **`batch_swap` 방향 feature 는 넣지 않는다.** 세 번의 측정(ablation −0.1711, batchswap −0.1442, 본 wave −0.0179)이 단조 감쇠했고, branch 로 쪼개면 부호가 갈린다.
5. **split 은 등록대로.** parent-blocked(`parent_record_id`, lineage_source 를 가로질러 — 신규 parent 56개가 기존 child 다) + cell holdout. **`HGD569_f125` = 영구 prospective holdout** 로 확정한다(실현가능 행 0인 유일 cell, 그리고 §7 의 baseline 문제로 pooled 회귀에서 어차피 빠진다).
6. **크기 보고 규칙.** parent < 20 인 stratum 평균의 효과크기는 보고하지 않는다(§4.1 의 3.37×, batchswap 의 5.8× — 같은 실패 모드 두 번).

---

## 14. round 2 cell 에 대한 함의

1. **signed `batch_swap` cell 을 새로 찾아야 한다.** round 1 은 signed cell 2개를 샀는데 실효는 1개였다(§5.1). r2 후보는 (a) 두 fresh type 의 `u_avg_enrichment` 가 다르고 (b) **plan 시점 core-digest 대조에서 outward/inward 형제가 다른 노심을 실현하는** cell 이어야 한다. (b) 는 현재 census 에 없는 검사다(§11-1).
2. **`depth2_edges > 0` 인 cell 을 더 넣어 burn-state × feed 교락을 깬다.** round 1 의 `twice_plus` 79행은 전부 f109/f113 에서 왔고 f121 은 구조적으로 0이다. `twice_plus` 의 out−in d_cyclen 이 `once` 의 **13배**(−4.19 vs −0.31)라는 신호가 있으므로(§3.5), 이것을 feed 효과와 분리하는 것이 r2 의 가장 값싼 큰 이득이다.
3. **독립 pair 수를 늘린다.** round 1 은 5 cell 이지만 `E1E2_f121`/`E1E2_f109` 가 같은 pair 이므로 **독립 pair 는 4개**다(prereg §12-7). pooled 부호검정에서 5를 독립으로 세지 않았고, r2 는 이 숫자를 늘리는 쪽으로 배분한다.
4. **Gd 적재가 다른 cell 을 의도적으로 배치한다.** H4 가 성립한 2 cell(E1E2)과 성립하지 않은 1 cell(N1N2)의 차이가 무엇인지가 round 1 의 가장 값진 미해결 질문이다. Gd 대비가 큰 pair 를 최소 3개 추가하면 그 자체로 §13.2-2 의 feature 설계 데이터가 된다.
5. **HGD569_f125 를 r2 에 다시 넣기 전에 parent 를 재평가한다.** 지금 상태로는 이 cell 의 vs-parent delta 를 쓸 수 없다(§7). 20개 parent 를 이번 wave 와 **같은 자산 경로**(synth deck `cddba86904810b7c` + `pair_ecore:MAS_RST.APRQ_11_0705.02`)로 재평가하면 +34.71 EFPD 오프셋이 사라지는지가 결정적 시험이다 — 사라지면 원인이 deck 임이 확정되고 160행이 전부 회복된다. 20 chain, 약 0.25 h.
6. **cell 수 확장은 코드 변경이 아니다.** prereg §12-1 대로 `CELLS_R1` 에 줄을 더하는 일이고, round 1 은 cadence **89.5 chains/h** 를 실측했다(계획 가정의 1.61배). 100-context 목표에서 cell 당 160 chain 이면 **cell 하나당 약 1.8 h** 다.

---

## 15. 답하지 못한 것

- **HGD569 +34.71 EFPD 오프셋의 원인.** synth deck 을 유력 후보로 지목했을 뿐 확정하지 못했다. §14-5 가 그 실험이다.
- **HGD569 `batch_swap` 형제가 동일 노심을 내는 기전.** 두 패턴이 다르고 두 fuel type 이 라이브러리상 다르다는 것까지 확인했고, power map 이 유한 노드 전부에서 동일하다는 사실까지 측정했다. 슬롯 기하 축퇴가 유력하지만 **octant 대칭 사상 여부를 직접 검증하지 않았다.**
- **`twice_plus` 효과가 burn-state 때문인지 feed 때문인지.** round 1 설계상 분리 불가(prereg §12-3 이 사전 공개).
- **`batch_swap` 의 참값.** 세 wave 가 −0.1711 → −0.1442 → −0.0179 로 단조 감쇠했다. 이것이 표본 크기의 함수인지 cell/branch 의 함수인지는 r2 이전에 답할 수 없다.

**AMENDMENT 2026-08-30 23:xx (post-run tooling):** 결과 §도구 소견 3건 구현 후 스크립트 sha 변경 — `intervention_wave.py` 393509456C715B81…, `mine_policy_corpus.py` 5377151AAEC14F2F…, `ablation_analyze.py` E5BC29F90EE8749F… (core-degeneracy guard, neutral-control offset/`effects_*_adj`, corpus F_xy 열). r1 실행·분석 artefact는 변경 전 버전으로 생성됨(published CSV 불변); steps.parquet 80→85열 마이그레이션(백업 `steps.parquet.bak_pre_fxy_cols_20260830`).

---

## AMENDMENT 2026-08-31 (결함 확정 · 격리 · 재평가 계획) — §5.1 · §7 · §11-1 · §14-5 · §15 정오

**본 절은 위 본문의 세 곳을 정정한다.** 근거는 `data/reports/hgd569_degeneracy_memo_20260830.md`(2026-08-30, read-only 조사).

### A. 결함 — 하나의 원인, 두 증상

§7 의 `HGD569_f125` 공통모드 +34.71 EFPD 와 §5.1 의 `batch_swap` 20/20 축퇴는 **같은 harness 결함**이다: 이 wave 의 deck emission 경로에서 `type_id → 2-char alias` 번역이 **조용히 무효(no-op)** 가 되어 `%LPD_SHF` 에 `F P6253Z1G06N24 0` 이 기록되었고, `%LPD_B&C` 에 없는 이 batch id 를 MASTER 가 **경고 없이 `P6` 로 흡수**했다. 노심 전체(fresh+carryover)가 `FA_P6` 단일 조성으로 계산되었다 — **설계된 노심이 아니다.**

정오 사항:

1. **§5.1 / §11-1 / §15 의 "슬롯 기하 축퇴 · octant 대칭 사상" 가설은 기각된다.** 축퇴는 기하가 아니라 token 사상 붕괴다. H2 의 "판정 불가" 결론 자체는 유지되나 사유가 다르다 — 물리 현상이 아니라 **harness 결함**이다.
2. **§7 이 지목한 "synth deck" 은 무죄다.** deck 파일(`cddba86904810b7c`)은 정상이며, 개입 wave deck 과 원 campaign deck 은 `%LPD_SHF` 본문을 제외하면 183/183행 일치한다.
3. **§14-5 의 처방(parent 20개 재평가)은 폐기한다.** 같은 결함 경로로 parent 를 재평가하면 오프셋이 **재현**될 뿐이며 "오프셋 소멸"로 오독될 위험이 있다. 올바른 실험은 **자식 160개의 재평가**다(아래 D).
4. **살아남는 것:** 쌍 내부 (out − in) 대조는 전부 유효하다(오프셋은 parent 공통모드로 상쇄). H1·H3·H4 는 안전하다. 단 §3.1 의 `HGD569` 행은 "설계 노심"이 아니라 "P6 단일 노심"에서 측정된 값이므로 이 cell 의 물리를 대표하지 않는다.

### B. 근본 원인 수정 (코드)

| 파일 | sha256[:16] | 수정 |
|---|---|---|
| `lpopt/search/assets.py` | `CD34EFA4134E5CF7` | `CaseAssetResolver` 가 `registry_aliases=None` 일 때 `<package_root>/registry.json` 에서 alias bridge 를 **스스로 적재**(`registry_aliases_from_package`). `validate_reload_deck` 에 **`%LPD_SHF` roster gate** 추가 — 모든 `F <id>` 는 2자 이하이고 deck 자신의 `%LPD_B&C` 에 존재해야 한다(`allowed_batch_ids=` 로 외부 지정 가능). |
| `lpopt/search/verify.py` | `D18250855D366046` | `WaveVerifier` 의 fallback resolver 가 `package_root or "."` 대신 **`package_root` 에서** 만들어진다(+`library_dims` 전달). 조용한 no-op 이 불가능해진다. |
| `lpopt/search/resolver.py` | `73585B42AB92FBB6` | `paramA_registry_aliases` 가 위 단일 구현을 재수출(bridge reader 1개). |
| `intervention_wave.py` | `F402EEFED4BE60FF` | 검증기 subclass 가 **`resolver=` 를 명시적으로 전달**(`campaign.py`/`produce.py` 와 동일 계약). paramA cell 인데 bridge 가 비면 `SystemExit`. `--campaign-suffix` 추가. |
| `ablation_wave.py` | `1B94C7128F41685B` | **무편집**(prereg §8 sha 고정 유지). |

기존 monkeypatch(`A.CaseAssetResolver`)는 남는다 — 그것은 `ablation_wave.cmd_run` 자신의 **자산 해결용** resolver 에 `synth_root`/`fuel_library` 를 주입하는 유일한 통로이기 때문이다. deck 을 실제로 쓰는 지점은 이제 patch 가 아니라 인자로 결선된다.

**회귀시험**: `tests/test_deck_alias_guard.py`(`D806FB599A9EFF8A`, 8 tests) — 수정 전 6 실패 / 수정 후 8 통과. 3-type 다중문자 type_id synth-deck 으로 (1) `WaveVerifier` emission 이 2자 alias 만 낸다, (2) raw type_id 를 담은 deck 은 guard 가 거부한다, (3) bridge 없는 구성은 **조용히 emit 하지 않고 첫 chain 에서 error** 가 된다를 고정. `pytest tests/test_verify_stub.py tests/test_intervention_wave.py tests/test_assets.py tests/test_campaign_stub.py tests/test_deck_alias_guard.py -q` → **127 passed, 4 skipped**.

실자산 확인(로컬, MASTER 미기동): 실제 plan 후보 1개를 실제 package/synth deck 으로 staging 하면 `%LPD_SHF` fresh id = `['S3','S5']`, raw type_id 0건; bridge 를 비운 재현 구성은 `DeckValidationError` 로 **deck 이 staging 되지도 않는다**.

### C. 데이터 격리 (실행 완료)

`python -m lpopt.tools.quarantine_campaign --campaign intervention_HGD569_f125 --failure alias_noop_P6_20260830 --backup-tag hgd569_quarantine --apply`
(도구 `lpopt/tools/quarantine_campaign.py` `5CDDF36B863532D7`, 시험 `tests/test_quarantine_campaign.py` `755C7161818FF8A7`, dry-run 기본)

| 파일 | before sha256[:16] | after sha256[:16] | 변경 |
|---|---|---|---|
| `data/store/records.parquet` | `F7430821400523D1` | `BAB841A75B347063` | 76,693행 **불변**; `intervention_HGD569_f125` 160행에 `valid=False`, `failure="alias_noop_P6_20260830"`. **삭제 없음** — record_id 를 점유한 채로 두어야 재평가가 같은 행을 upgrade 한다. |
| `data/policy/steps.parquet` | `828397E63E8E6541` | `5FA791B8B3197179` | 28,897 → **28,737**행(해당 160 edge drop). |

백업: `E:/lpopt_data/5_RL/backups/records.parquet.bak_pre_hgd569_quarantine_20260830`, `…/steps.parquet.bak_pre_hgd569_quarantine_20260830`.

`steps.parquet` 은 **플래그가 아니라 drop** 이다: `intervention_wave.cmd_corpus` 가 `set(existing.columns) - set(new.columns)` 로 schema drift 를 거부하므로 `quarantined` 열을 더하면 **정정 재실행 자체가 막힌다**. 행은 백업에서 복구 가능하고, 재실행이 `build_steps` 로 결정적으로 재생성한다.

> ⚠️ **`valid=False` 만으로는 elite pool / 학습에서 빠지지 않는다.** 이 트리의 elite·replay·holdout·모델 학습 필터는 거의 전부 `converged == True` 만 본다(`campaign.py:1373/1401/3607`, `produce.py:538/646/679`, `curriculum.py`, `model/*`, `boundary_probe.py:194`). `valid` 를 함께 보는 곳은 `CampaignDriver._replay_rows`(`campaign.py:1526`) **하나뿐**이다. 따라서 격리된 160행은 여전히 elite seed·학습 표본으로 들어갈 수 있다. 즉시 배제가 필요하면 `--unconverge`(`converged=False` 동시 설정, opt-in)를 쓰거나 필터를 `valid` 까지 보도록 고치는 것이 근본책이다. **본 실행에서는 지시대로 `--unconverge` 를 쓰지 않았다** — 153행은 여전히 `converged=True` 다.
>
> upsert 관점에서는 무해하다: `_quality_rank = converged*8 + valid*4 + flat*2 + fxy` 이므로 격리행은 ≤11, 재평가행(converged·valid·map·F_xy)은 15 → **재평가가 항상 이긴다**(`dedup_upsert`, 동점은 incumbent 우선).

### D. 재평가 (§14-5 대체, 미실행 — 원격 접근 없음)

**대상은 자식 160 chain 이다.** parent 20행은 정상 경로(`campaign.py`, `resolver=` 전달)에서 나왔으므로 손대지 않는다.

* run dir: `runs/intervention_wave_r1_hgd569_fix` (신규). round 1 dir 은 읽지도 쓰지도 않는다 — 같은 dir 로 돌리면 `_done()` 이 160개를 전부 "settled" 로 건너뛴다.
* campaign tag: `intervention_HGD569_f125_v2` (`--campaign-suffix _v2`). `record_id = sha256(pattern|library|pair|deck_knobs)` 는 campaign 을 해싱하지 않으므로 **record_id 는 동일**하고, C 의 rank 규칙에 따라 격리행을 **제자리에서 upgrade** 한다.
* restart: `pair_ecore:MAS_RST.APRQ_11_0705.02` — 등록값과 일치(로컬 dry-run 확인). **`--allow-fallback` 은 필수다**: 이 cell 은 native restart 가 없어 `fallback_level=3` 이고 `ablation_wave.cmd_run` 의 게이트는 `!= 0` 이다. 완화가 아니다 — parent 와 **같은** restart 이며, `intervention_wave.check_restart`(parent restart 일치 게이트)는 그대로 살아 있다.
* deck: synth `cddba86904810b7c` (변경 없음).

**STAGE 1 — 2-chain 진단** `diag_intervention_wave_r1_hgd569_fix_199.bat` (`749CBCEBC90EAC5F`, `--max-chains 2`, ~0.03 h). 세 가지가 모두 성립해야 STAGE 2 로 간다:

1. **emit 된 deck**(`produce_cases/**/MAS_INP_cy12.inp`)의 `%LPD_SHF` fresh id 가 `S3`/`S5` 뿐이고 `P6253Z*` 0건 — MASTER 없이 즉시 판정되는 결정적 검사(bat 의 CHECK 1, PowerShell 정규식);
2. **MASTER 자신의 echo**: `master_work/worker_00/*/MAS_OUT` 의 `== CORE LOADING PATTERN` 이 **S3/S5 혼합**(bat 의 CHECK 2, `Select-String -Context 0,12`). round 1 은 여기서 전 위치 `P6` 였다. `harvest_maps=True` → `keep_success` 이므로 최종 cycle work dir 이 남아 사후 확인이 가능하다;
3. `ablation_results.jsonl` 의 cyclen 이 이 cell 의 **실제 밴드 ~708–741 EFPD**(round 1 은 762–771).

**STAGE 2 — 160 chain 재평가 + kit** `resume_intervention_wave_r1_hgd569_fix_199.bat` (`634660D938DBBF77`). `PYTHONIOENCODING=utf-8`/`PYTHONUTF8`/`chcp 65001` 설정, 24 workers, `--allow-fallback`, 종료 후 `kit --campaign-suffix _v2` 로 `kitdata` 생성 → 조정자에서 `merge-store`. 예산 ~1.8 h. 두 bat 모두 **재실행이 곧 resume** 이며 round 1 산출물을 건드리지 않는다.

**선행 조건:** 위 B 표의 4개 파일을 199 박스로 복사해야 한다(sha 대조). `ablation_wave.py` 는 `1B94C7128F41685B` 로 **변경 없음**을 확인할 것.

### E. 남는 조치

* §5.2 의 잠재 위험 호출부 — `fr_arms.py:188`, `fr_transfer.py:522`, `rule_acid_run.py:386`, `v520_run.py:622`, `lpopt/design/pathfinder.py:268` 은 여전히 `resolver=` 없이 `WaveVerifier` 를 만든다. B 의 수정으로 **이제 안전하다**(package 에서 bridge 를 적재하고, 실패해도 deck gate 가 막는다). 그래도 명시 전달이 낫다.
* §13.2-5 의 "`HGD569_f125` 영구 prospective holdout" 처방은 **결과적으로 옳았으나 사유가 틀렸다** — 사유를 "cell 의 OOD 성격"이 아니라 "**deck emission 결함**"으로 갱신한다. 재평가가 끝나면 이 cell 은 정상 cell 이므로 holdout 사유를 다시 판단해야 한다.
* §11-1 의 core-degeneracy guard 는 **증상(예산 낭비)** 만 막는다. 노심 자체가 틀린 것은 B 의 roster gate 가 막는다. 병행 유지.

---

## AMENDMENT 2 (HGD569 v2) 2026-08-31 — 재평가 실행 결과 · §1 · §3 · §5 · §7 · §8 · §10 · §13 · §14-5 · §15 정오

**AMENDMENT 1(D) 의 STAGE 1·2 가 실행되었다.** `intervention_HGD569_f125` 의 자식 160 chain 을 수정된 자산 경로로 전량 재평가했고(campaign `intervention_HGD569_f125_v2`), 그 결과로 **HGD569 cell 을 v2 run 으로 갈아끼운 분석 view** 를 다시 돌렸다. r1 CSV 는 보존하고 v2 산출물은 `data/reports/intervention_wave_r1_*_v2.csv` 로 별도 발행한다.

분석 view 구성(원 run dir 무편집): `E:/lpopt_data/5_RL/harvest/intervention_wave_r1_v2/` 에 4개 원 cell dir + v2 cell dir 을 **디렉터리 junction** 으로 걸고 `--run-dir` 로 지정. 매니페스트의 `run_subdir` 계약(`intervention_<cell>`)을 그대로 만족하므로 `analyze` 는 plan 을 바꾸지 않고 v2 를 읽는다.

```
python intervention_wave.py analyze --plan data/design/intervention_wave_r1.json \
    --run-dir E:/lpopt_data/5_RL/harvest/intervention_wave_r1_v2 --out-dir data/reports
python intervention_wave.py corpus --plan data/design/intervention_wave_r1.json \
    --cells HGD569_f125 --campaign-suffix _v2 --wave hgd569_v2
```

**한 문장으로**: **+35 EFPD 공통모드는 사라졌고(중립 대조군 오프셋 35.053 → 0.083 EFPD), §5.1 의 `batch_swap` 20/20 축퇴는 결함의 산물이었음이 확정되었다(4축 비트 동일 20/20 → 0/19).** H2 는 "판정 불가" 에서 벗어나 **부호 기준 성립(2/2 음수)** 으로 바뀌고, H1·H3·H4 의 판정은 유지되며, HGD569 는 §3.1 의 "탈동조" cell 에서 **4축 정합 cell** 로 이동한다.

---

### A. 재평가 run 무결성 — HGD569_f125 v2

| 항목 | r1 (결함) | **v2 (수정)** | |
|---|---|---|---|
| chains | 160 | **160** — subplan 의 후보 record_id 집합·pattern 이 r1 과 **완전 동일** | 단일변수 실험 |
| converged | 153 (95.6%) | **152 (95.0%)** | P2 ✅ |
| error | 7 `non_finite_flux` | **8 `non_finite_flux`** (r1 7건을 **전부 포함** + 1건 추가) | P3 임계 5.0% **정확히 충족**(초과 아님) |
| F_xy parsed / conv | 153/153 | **152/152 = 100%** | P1 ✅ |
| restart | `pair_ecore:MAS_RST.APRQ_11_0705.02` | **동일**, sha16 `5b72b23035493ab5`, `fallback_level=3`, 로그 `parents label(s) ['pair_ecore:…0705.02']` | P5 ✅ (`check_restart` 통과) |
| deck | synth `cddba86904810b7c` | **동일** | 변경 0 |
| cyclen 밴드 | 762–771 | **727.477–738.084** (평균 731.238) | 원 campaign 밴드 709–741 안 |
| wall | 6,093 s | **6,489.5 s** (+ STAGE 1 진단 197.5 s = 합 1.86 h) | P6 ✅ |
| rc | — | `run=0 kit=0` | ✅ |

**wave 합계(v2 view)**: 800 chain, **790 수렴**(r1 791), error 10(T6T4 1 · N1N2 1 · HGD569 8). 두 멤버 모두 수렴한 쌍 **233/240**(r1 234).

**F_xy 3중 교차검증 — 다시 비트 동일.**

| 대조 | 매칭 | max \|Δ\| |
|---|---:|---:|
| `fxy_sidecar.jsonl` ↔ `records.parquet` (v2 152행) | 152 | **0** |
| sidecar ↔ `fxy_backfill_199_intervention_hgd569_fix_20260831.csv` (독립 재스캔) | 152 | **0** |
| store ↔ 재스캔 (`f_xy` 및 `f_xya`) | 152 | **0** |

재스캔 161행 = sane **152** + insane 9(= `nonfinite` 8 — error chain 8건과 정확히 일치 — 및 `superseded` 1건 `dfd5aaa3d27c2b0f`). §2 와 같은 구조이며 **불일치 0건**이다.

---

### B. §7 정오 — 공통모드는 사라졌다

`neutral_control_offset` (등록된 중립 대조군 `rewire_swap|neutral` 의 cell 별 vs-parent 평균, n=20 · parent 20):

| cell | d_cyclen r1 → **v2** | d_F_r r1 → **v2** | d_F_xy r1 → **v2** | d_node_peak r1 → **v2** |
|---|---|---|---|---|
| `E1E2_f109` | +0.0228 → +0.0228 | +0.0094 → +0.0094 | +0.0038 → +0.0038 | +0.0158 → +0.0158 |
| `E1E2_f121` | +0.0296 → +0.0296 | +0.0072 → +0.0072 | +0.0066 → +0.0066 | +0.0076 → +0.0076 |
| `N1N2_f113` | +0.1739 → +0.1739 | +0.0137 → +0.0137 | +0.0114 → +0.0114 | +0.0160 → +0.0160 |
| `T6T4_f121` | +0.0431 → +0.0431 | +0.0158 → +0.0158 | +0.0269 → +0.0269 | +0.0100 → +0.0100 |
| **`HGD569_f125`** | **+35.0533 → +0.0829** | **+0.0982 → +0.0262** | **+0.1056 → +0.0249** | **+0.0493 → +0.0239** |

**HGD569 는 이제 다른 4개 cell 과 같은 자릿수의 baseline 을 가진다** — cyclen 오프셋은 `N1N2_f113` 의 +0.174 보다도 작다. §7 이 "중립 개입이 cycle length 를 35 EFPD 늘릴 수는 없다" 고 적은 그 이상은 **소멸했다.** §7 의 진단 방향(cell 국소 자산 경로 문제)은 옳았고, AMENDMENT 1 의 원인 확정(alias no-op → 전 노심 `FA_P6`)이 **측정으로 확증**되었다.

`effects_by_cell` 의 HGD569 행, raw / **adj**:

| move_class · dir | n | d_cyclen r1 | **d_cyclen v2** | d_F_r v2 | d_F_xy v2 | d_node_peak v2 |
|---|---:|---|---|---|---|---|
| `fresh_relocate`·inward | 36 | +34.998 / −0.056 | **−0.003 / −0.086** | +0.0728 / +0.0466 | +0.1045 / +0.0796 | +0.0420 / +0.0180 |
| `fresh_relocate`·outward | 37 | +33.872 / −1.181 | **−0.798 / −0.881** | +0.0635 / +0.0373 | +0.1015 / +0.0766 | +0.0110 / −0.0129 |
| `batch_swap`·inward | 19 | +34.942 / −0.112 | **−0.385 / −0.468** | +0.0291 / +0.0029 | +0.0403 / +0.0153 | +0.0209 / −0.0031 |
| `batch_swap`·outward | 20 | +34.942 / −0.112 | **+0.158 / +0.075** | +0.0149 / −0.0113 | +0.0187 / −0.0062 | +0.0124 / −0.0115 |
| `batch_flip`·inward | 10 | +35.253 / +0.200 | **−1.432 / −1.515** | +0.0064 / −0.0198 | +0.0087 / −0.0163 | +0.0001 / −0.0239 |
| `batch_flip`·outward | 10 | +34.630 / −0.423 | **+0.965 / +0.882** | −0.0051 / −0.0313 | −0.0036 / −0.0285 | −0.0024 / −0.0263 |
| `rewire_swap`·neutral | 20 | +35.053 / 0 | **+0.083 / 0** | +0.0262 / 0 | +0.0249 / 0 | +0.0239 / 0 |

r1 에서 `batch_swap` 의 inward 행과 outward 행이 **소수점 이하 전 자리까지 동일**(둘 다 +34.9417)했던 것이 §5.1 축퇴의 표 차원 지문이다. v2 에서는 갈라진다(−0.385 vs +0.158).

---

### C. §5.1 · §11-1 · §15 정오 — 축퇴는 결함의 산물이었다

같은 20개 parent, 같은 `batch_swap` 형제쌍, 같은 restart·deck 에서:

| 검정 | r1 (결함) | **v2 (수정)** | 대조군 (T6T4, r1) |
|---|---|---|---|
| 4축 스칼라(`f_r`,`cyclen`,`f_xy`,`node_peak`) 비트 동일 쌍 | **20/20** | **0/19** | 0/19 |
| \|Δ f_r\| min / med / max | 0 / 0 / 0 | **0.0040 / 0.0223 / 0.1069** | — |
| \|Δ cyclen\| | 0 / 0 / 0 | **0.027 / 0.274 / 2.319** | — |
| \|Δ f_xy\| | 0 / 0 / 0 | **0.0038 / 0.0409 / 0.1585** | — |
| power map max\|Δ\| (유한 노드) | **0 — 20/20** | **2.295 / 2.611 / 3.365 — zero 0/19** | 2.84–4.61 |
| 참고: 같은 cell `fresh_relocate` map max\|Δ\| | 15.3–35.1 | 12.40 / 27.49 / 32.74 | — |

**§5.1 이 "prereg §2.5 census 가 놓친 두 번째 종류의 방향 축퇴(슬롯 기하)" 로 적은 현상은 존재하지 않는다.** 두 형제의 pattern 과 core digest 는 r1 에서도 **달랐고**, 동일해진 것은 MASTER 가 읽은 노심이었다. 따라서 §11-1 이 제안하고 그 뒤 구현된 **core-degeneracy guard 는 이 사고를 잡을 수 없었다**(AMENDMENT 1 R6 와 같은 결론, 이제 측정으로 확정). 축퇴를 막는 것은 digest 대조가 아니라 AMENDMENT 1 B 의 `%LPD_SHF` roster gate 다.

**예산 회수**: §10 이 "정보량 0" 으로 잃었다고 적은 40 chain(cell 예산 25%, wave 5.0%)은 **전부 유효 라벨로 회수되었다.** 실효 signed `batch_swap` 대조는 **T6T4 19쌍 단독 → T6T4 19쌍 + HGD569 19쌍 = 38쌍**으로 두 배가 된다.

---

### D. 갱신된 1차 통계량

> CI 는 전부 `ablation_analyze._bootstrap_ci`(쌍 재표집 20,000회, `SEED=20260829`), 부호검정은 `_sign_test`(정확 이항, 0 제외), dose slope 는 `_fe_slope`(parent-FE + parent 클러스터 부트스트랩 2,000회) — **§3 과 같은 함수·같은 시드**다. 재계산한 CI 는 §3.1 발행값과 **소수 3째 자리에서만** 다르다(부트스트랩 draw 차이). 평균·부호검정·p 는 §3.1 과 **비트 일치**한다.

#### D.1 §3.1 `fresh_relocate` 쌍 대조 — HGD569 행만 이동한다

| cell | n쌍 | d_cyclen (EFPD) | d_F_r | d_F_xy | d_node_peak | 매개 부호 |
|---|---:|---|---|---|---|---|
| `E1E2_f109` | 40 | −3.1616 [−4.261,−2.086] · 6/40 · **8.4e-6** | −0.1061 · 10/40 · **0.0022** | −0.1283 · 11/40 · **0.0064** | −0.0250 · 11/40 · **0.0064** | 정합 (불변) |
| `E1E2_f121` | 40 | −0.2825 [−0.601,+0.016] · 13/40 · **0.038** | −0.0674 · 4/40 · **1.9e-7** | −0.0677 · 12/40 · **0.017** | −0.0378 · 13/40 · **0.038** | 정합 (불변) |
| `N1N2_f113` | 39 | −1.0122 [−1.682,−0.355] · 11/39 · **0.0095** | −0.0512 · 17/39 · 0.522 | −0.0771 · 14/39 · 0.108 | −0.0186 · 18/39 · 0.749 | 정합 (불변) |
| **`HGD569_f125`** | 36 | **−0.8119** [−1.446,−0.262] · 14/36 · 0.243 | **−0.0103** [−0.034,+0.011] · 16/36 · 0.618 | **−0.0053** [−0.037,+0.026] · 17/36 · 0.868 | **−0.0312** [−0.048,−0.015] · 11/36 · **0.0288** | **정합 — 4/4 음수** (r1 은 "탈동조") |
| `T6T4_f121` | 40 | −0.5171 [−1.012,−0.026] · 18/40 · 0.636 | +0.0133 · 21/40 · 0.875 | +0.0112 · 20/36 · 0.618 | +0.0155 · 23/40 · 0.430 | 역전 (불변) |
| **POOLED** | **191/195** | **−1.1649** [−1.511,−0.836] · 62/195 · **4.0e-7** | **−0.0450** · 68/195 · **2.9e-5** | **−0.0557** · 74/191 · **0.0023** | **−0.0192** · 76/195 · **0.0025** | — |

r1 POOLED 는 −1.2353 (p=8.1e-8) / −0.0443 / −0.0546 / −0.0149 였다. **pooled 는 5% 이내로만 움직인다** — §7 이 "쌍 내부 대조는 공통모드에 면역" 이라고 적은 것이 사후적으로 확인된다.

바뀐 것은 HGD569 의 **개별 판독**이다. d_F_xy 가 +0.0003 → **−0.0053** 으로 나머지 세 축과 부호를 맞추고, `d_node_peak` 이 −0.0082(p=0.132) → **−0.0312(p=0.0288)** 로 **유의해진다.** 즉 이 cell 도 "outward → 누설↑ → cycle↓ ∧ 평탄도↑" 매개를 따르며, **§3.1 이 매개 부호가 역전된 유일한 cell 로 지목한 것은 이제 `T6T4_f121` 하나뿐**이다.

#### D.2 §3.2 dose-response (parent-FE, `d_cyclen ~ d_fresh_enr_r_center`)

| cell | slope r1 | **slope v2** | 95% CI (v2) | n |
|---|---:|---:|---|---:|
| `E1E2_f109` | −30.285 | −30.285 | [−33.98, −26.80] | 80 |
| `N1N2_f113` | −9.518 | −9.518 | [−14.40, −4.08] | 79 |
| **`HGD569_f125`** | **−9.967** | **−7.094** | [−12.40, **−2.24**] | 73 |
| `E1E2_f121` | −2.542 | −2.542 | [−5.36, −0.05] | 80 |
| `T6T4_f121` | −1.031 | −1.031 | [−3.88, +1.68] | 80 |
| **POOLED** | −8.842 | **−8.231** | [−10.77, −6.02] | 392 |

부호 **5/5 음수 유지**. HGD569 는 29% 작아지되 CI 는 여전히 0 을 포함하지 않는다. cell 간 산포는 29배 → **29배**로 실질 불변이며, §3.2 의 "dose 상수를 보고하지 않는다" 는 결론은 그대로다.

#### D.3 §3.3 `batch_swap` signed — 이제 두 cell 이 모두 판독된다

| cell | n쌍 | d_F_r | d_F_xy | d_cyclen | d_node_peak |
|---|---:|---|---|---|---|
| `T6T4_f121` | 19 | −0.0179 [−0.053,+0.016] · 8/19 · 0.648 | −0.0519 [−0.096,−0.009] · 6/17 · 0.332 | −0.0757 · 9/19 · 1.000 | −0.0001 · 10/19 · 1.000 |
| **`HGD569_f125`** | **19** | **−0.0177** [−0.036,−0.001] · 7/19 · 0.359 | **−0.0285** [−0.058,−0.001] · 9/19 · 1.000 | **+0.5726** [+0.215,+0.970] · 11/19 · 0.648 | **−0.0079** [−0.021,+0.005] · 9/19 · 1.000 |
| **POOLED** | **38** | **−0.0178** [−0.037,+0.001] · 15/38 · 0.256 | **−0.0395** [−0.066,−0.014] · 15/36 · 0.405 | **+0.2484** [+0.035,+0.492] · 20/38 · 0.871 | −0.0040 · 19/38 · 1.000 |
| branch split (T6T4) ~618 | 11 | −0.0349 · 4/11 · 0.549 | −0.0780 · 3/11 · 0.227 | −0.2206 · 4/11 · 0.549 | — |
| branch split (T6T4) ~625 | 8 | **+0.0055** · 4/8 · 1.000 | −0.0039 · 3/6 · 1.000 | **+0.1235** · 5/8 · 0.727 | — |

**두 signed cell 의 `d_F_r` out−in 이 −0.0179 와 −0.0177 로 사실상 일치한다.** 둘 다 유의하지 않다.

#### D.4 §3.4 vs-parent 한계효과 pooled — raw 열이 되살아났다

| move_class · dir | d_cyclen raw r1 / **raw v2** / adj v2 | d_F_xy raw v2 / adj v2 | d_F_r raw v2 / adj v2 | d_node_peak raw v2 / adj v2 |
|---|---|---|---|---|
| `fresh_relocate` · inward | +6.267 / **−0.195** / −0.265 | +0.1736 / +0.1593 | +0.1324 / +0.1181 | +0.0593 / +0.0449 |
| `fresh_relocate` · outward | +5.143 / **−1.369** / −1.439 | +0.1192 / +0.1049 | +0.0882 / +0.0739 | +0.0407 / +0.0262 |
| `batch_swap` · inward | +17.97 / **−0.144** / −0.207 | +0.0813 / +0.0554 | +0.0526 / +0.0316 | +0.0298 / +0.0128 |
| `batch_swap` · outward | +17.48 / **+0.089** / +0.026 | +0.0450 / +0.0192 | +0.0367 / +0.0157 | +0.0258 / +0.0089 |
| `batch_swap` · neutral | −0.208 / −0.208 / −0.284 | +0.0563 / +0.0490 | +0.0771 / +0.0670 | +0.0767 / +0.0636 |
| `batch_flip` · inward | +15.72 / **−1.751** / −1.813 | +0.0123 / −0.0135 | +0.0087 / −0.0121 | +0.0082 / −0.0084 |
| `batch_flip` · outward | +18.95 / **+1.227** / +1.163 | +0.0291 / +0.0032 | +0.0198 / −0.0015 | +0.0177 / +0.0004 |
| `batch_flip` · neutral | −0.009 / −0.009 / −0.084 | +0.0086 / +0.0013 | +0.0125 / +0.0024 | +0.0060 / −0.0071 |
| `rewire_swap` · neutral | +7.065 / **+0.070** / 0 (기준) | +0.0145 / 0 | +0.0145 / 0 | +0.0147 / 0 |

**§3.4 의 "raw 열은 그대로 읽으면 안 된다" 는 경고를 철회한다.** raw 와 adj 의 최대 격차가 cyclen 기준 **18 EFPD → 0.07 EFPD** 로 줄었다. adj 열은 계속 발행하지만 이제 두 열이 같은 이야기를 한다. (`batch_swap`·`batch_flip` 의 signed stratum 이 여전히 paramA 2 cell 에서만 나온다는 구성 조건은 §13.1 대로 유지된다.)

#### D.5 §3.5 burn-state — `once` 층이 0 근방으로 내려앉는다

| burn_state | outward d_cyclen r1 → **v2** | inward r1 → **v2** | out−in r1 → **v2** |
|---|---:|---:|---:|
| `once` (n=313) | +7.512 → **−0.658** | +7.825 → **−0.252** | −0.313 → **−0.406** |
| `twice_plus` (n=79) | −4.157 → −4.157 | +0.033 → +0.033 | −4.190 → −4.190 |

`once` 층의 +7.8 EFPD 는 전량 HGD569 공통모드였다(HGD569 의 `fresh_relocate` 는 전부 `once` 층이고, `twice_plus` 79행은 f109/f113 전용이라 **불변**). **`twice_plus` 의 cycle 손실이 `once` 의 약 10배** 라는 신호는 유지되며(−4.19 vs −0.41), burn-state × feed 교락(§14-2)도 그대로 남는다.

---

### E. 가설 판정 갱신

| 마크 | r1 판정 | **v2 판정** | 사유 |
|---|---|---|---|
| **H1** | 부분 성립, cell-conditional 로 강등 | **변경 없음 — 부분 성립, cell-conditional** | 부호 **5/5 음수** 유지, 유의 **3/5** 유지(HGD569 는 p 0.065 → 0.243 으로 **덜** 유의해짐). pooled −1.1649, p=4.0e-7. 등록된 반증 조건(부호 뒤집힘 ∧ p<0.05)은 여전히 미발생 |
| **H1b** | 🔴 T6T4 자체 재현 실패 | **변경 없음** | T6T4 는 이번 정정과 무관(−0.5171 · 18/40 · p=0.636, dose −1.031 [−3.88,+1.68]). ablation 대비 3.37× 과대추정 철회는 유지 |
| **H2** | 판정 불가(구조적 공집합) + 일반성 부정 | **🔁 판정 가능해짐. 등록 바(부호) 성립, 유의성 없음. §5.2 일반성 부정은 유지** | 등록 바 "signed 2 cell 모두 `mean(out−in) d_F_r < 0`" : T6T4 **−0.0179**, HGD569 **−0.0177** → **2/2 음수 = 충족**. 등록된 반증 조건("두 cell 부호가 서로 반대")은 **미발생**. 단 두 cell 모두 p ≥ 0.26 이고 pooled CI [−0.037,+0.001] 이 0 을 포함하며, T6T4 를 618/625 가지로 쪼개면 부호가 반대(§5.2)이고 batchswap wave 의 −0.1442 [−0.1888,−0.1013] 는 **어느 cell·어느 가지에서도 재현되지 않는다**(HGD569 CI [−0.036,−0.001] 역시 그 구간을 포함하지 않는다). → **"부호는 재현, 크기는 재현 실패". policy feature 전역 사용 불가 결론은 그대로 유지** |
| **H3** | ✅ 성립 | **✅ 성립 — 강화됨** | `fresh_relocate` 두 축 부호 일치가 **5/5 cell** 로 올라간다(HGD569 가 +0.0003/−0.0065 → −0.0053/−0.0103 으로 부호를 맞춤). 불일치 사례(E1E2_f109 `batch_flip\|neutral`, d_F_xy −0.0027 p=0.041 vs d_F_r +0.0100)는 ga80 cell 이라 **불변**. 전달계수 분기도 불변 — 농축-반경 계열 **1.35 / 1.42**(`fresh_relocate` in/out) · **1.76 / 1.22**(`batch_swap` signed in/out), Gd·격자 계열 **0.73**(`batch_swap\|neutral`) · **0.55**(`batch_flip\|neutral`). 신규: `batch_swap` out−in 전달비 T6T4 **2.90**, HGD569 **1.61** |
| **H4** | ✅ 2/3 cell 성립 | **✅ 변경 없음 (비트 불변)** | H4 는 축퇴 3 cell(E1E2_f121·E1E2_f109·N1N2_f113, 전부 ga80)만 쓴다. HGD569 는 H4 표에 등장하지 않으므로 +0.0712(20/20, p=1.9e-6) · +0.0689(17/20, p=0.0026) · +0.0071(12/20, p=0.503) 그대로 |
| **P1–P6** | ✅ | **✅** | §A. P3 만 HGD569 4.4% → **5.0%(임계 정확히 충족)** |
| **G1–G5** | ✅ | **✅** | G3 `single_move` **1.000**(152/152), G2 schema drift 0 · 중복 0 (§G) |
| **🔴 §7 공통모드** | cell 격리 필요 | **✅ 해소** | 중립 대조군 오프셋 +35.053 → **+0.083 EFPD** |
| **🔴 §5.1 축퇴** | 구조적 공집합 | **✅ 결함 산물로 확정, 40 chain 회수** | 4축 비트 동일 20/20 → **0/19** |

---

### F. §8 정오 — 블라인드 s1i 채점 재계산

예측 CSV 는 **개봉 전 등록본 그대로**(`5135B59E…F3A0`, 무편집)이고 진리값만 v2 로 바뀐다.

| 절대 수준 (child) | r1 (n=791) | **v2 (n=790)** |
|---|---|---|
| cyclen — bias / MAE / RMSE / ρ | +7.321 / 8.008 / 16.804 / 0.989 | **+0.559 / 1.251 / 2.061 / 0.990** |
| F_r | +0.0295 / 0.0521 / 0.0686 / 0.808 | **+0.0160 / 0.0466 / 0.0619 / 0.761** |
| node_peak | −0.0084 / 0.0355 / 0.0481 / 0.730 | **−0.0146 / 0.0332 / 0.0445 / 0.778** |
| F_xy (proxy) | +0.0321 / 0.0641 / 0.0877 / 0.768 | **+0.0172 / 0.0665 / 0.0879 / 0.706** |

| delta (parent→child) | r1 | **v2** |
|---|---|---|
| d_cyclen — bias / MAE / ρ / 부호일치 | +6.979 / 7.330 / 0.325 / 0.650 | **+0.218 / 0.619 / 0.634 / 0.752** |
| d_cyclen — **HGD569 만** | **+35.682 / 35.682** (n=153) | **+0.733 / 0.992** (n=152) |
| d_cyclen — HGD569 제외 | +0.095 / 0.530 (n=638) | +0.095 / 0.530 (n=638, **불변**) |
| d_F_r | +0.0453 / 0.0539 / 0.676 / 0.848 | **+0.0318 / 0.0432 / 0.777 / 0.835** |
| d_node_peak | +0.0147 / 0.0309 / 0.657 / 0.784 | **+0.0085 / 0.0277 / 0.679 / 0.771** |
| d_F_xy (proxy) | +0.0547 / 0.0663 / 0.656 / 0.830 | **+0.0393 / 0.0544 / 0.755 / 0.804** |

**이것이 §7·AMENDMENT 1 진단의 가장 강한 독립 확증이다.** wave 개봉 전에 등록·동결된 모델이 HGD569 자식 152개를 **MAE 0.99 EFPD** 로 맞힌다. r1 에서 이 cell 하나가 만든 35.68 EFPD 계통 편차는 물리도 모델 실패도 아니었다. 전체 delta ρ 는 0.325 → **0.634**, 부호 일치는 0.650 → **0.752** 로 올라간다 — r1 의 채점은 **모델을 과소평가**하고 있었다.

**등록된 관전 포인트는 뒤집힌다.** prereg §5 는 "s1i 가 `batch_swap` 의 방향 부호를 **틀릴** 것" 이라고 적었고 §8 은 이를 적중으로 채점했다:

| class | 반응 | 예측 out−in | 측정 r1 | **측정 v2** | |
|---|---|---:|---:|---:|---|
| `fresh_relocate` | cyclen | −1.175 | −1.235 | **−1.165** | ✅ 부호·크기 (오차 5% → **1%**) |
| `fresh_relocate` | F_r | −0.0132 | −0.0443 | −0.0450 | ✅ 부호 |
| `fresh_relocate` | F_xy | −0.0161 | −0.0546 | −0.0557 | ✅ 부호 |
| `batch_swap` | **cyclen** | **+0.211 / +0.220\*** | −0.037 | **+0.248** | **✅ 부호 일치 — §8 의 "예고 적중" 판정을 철회한다** |
| `batch_swap` | F_r | −0.0001 / −0.0002\* | −0.0087 | −0.0178 | ✅ (사실상 0 예측) |

\* v2 의 쌍 집합(38쌍)에서 재계산한 예측 평균. r1 의 `batch_swap` 측정 −0.037 은 **T6T4 19쌍 단독**에서 나온 값이고 HGD569 20쌍은 정확히 0 을 기여했다(축퇴). 그 쌍들이 실제 노심으로 되살아나자 부호가 예측과 일치하고 크기도 13% 이내다. **s1i 의 `batch_swap` 방향 실패는 관측되지 않았다** — 다만 측정 자체가 여전히 유의하지 않으므로(p=0.871) "모델이 맞혔다" 로도 읽지 않는다. 이 stratum 은 **아직 측정되지 않았다** 는 §8 의 두 번째 문장만 살아남는다.

---

### G. §10 정오 — corpus delta

```
python intervention_wave.py corpus --plan data/design/intervention_wave_r1.json \
    --cells HGD569_f125 --campaign-suffix _v2 --store data/store/records.parquet \
    --steps data/policy/steps.parquet --wave hgd569_v2
```

| | before (AMENDMENT 1 C 격리 후) | **after** |
|---|---:|---:|
| rows | **28,737** | **28,889** (**+152**) |
| cols | 85 | **85** (schema drift 0) |
| sha256 | `5FA791B8…8C37` | `2A34F7B3…F3E2` |
| bytes | 9,499,252 | 9,556,122 |

**백업 (append 전, 요청 경로)**: `E:/lpopt_data/5_RL/backups/steps.parquet.bak_pre_hgd569_v2_20260831` — sha256 `5FA791B8B31971793D44ECF3CF8568F516F4242A7338904DD3929DDD86CA8C37`, 9,499,252 bytes, **원본과 비트 동일**. (`corpus` 자신도 `data/policy/steps.parquet.bak_pre_hgd569_v2` 를 남겼다 — 같은 sha.)

**+160 이 아니라 +152 다.** 8건은 v2 에서 `non_finite_flux` 라 라벨이 없다. r1 은 error chain 을 target-null 행으로 코퍼스에 남겼지만(§10 무결성 표의 9행), 이번 append 는 store 를 원천으로 하고 격리된 8행이 store 에서 `converged=False` 이므로 `build_steps` 가 edge 를 만들지 않는다. 결과적으로 wave 전체의 target-null 행은 9 → **2**(T6T4 1 · N1N2 1)로 줄고 HGD569 의 target-null 은 **0** 이다.

| lineage_source | rows | Δ |
|---|---:|---|
| `intervention_T6T4_f121` | 160 | — |
| **`intervention_HGD569_f125`** | **152** | −160 (격리) → **+152** (v2) |
| `intervention_E1E2_f121` | 160 | — |
| `intervention_E1E2_f109` | 160 | — |
| `intervention_N1N2_f113` | 160 | — |
| **intervention 합** | **792** | (r1 800) |

`lineage_source` 는 접미사 없이 `intervention_HGD569_f125` 로 기록된다 — `--campaign-suffix` 는 store 의 `campaign` 열에만 붙으므로 나머지 4개 cell 과 **조인 계약이 동일**하게 유지된다.

#### 무결성

| 항목 | 값 |
|---|---|
| `(parent_record_id, child_record_id)` 중복 — 신규 152행 내부 | **0** |
| 동일 — 파일 전체 28,889행 | **0** |
| 신규 child 가 다른 lineage 의 child 로도 존재 | **0** |
| `single_move` | **152 / 152 = 1.000** |
| `cross_cell` | **False 152/152** |
| `both_converged` | **True 152/152** |
| target null (`d_f_r`/`d_cyclen`/`d_cbc_max`/`d_node_peak`) | **0행** |
| **`parent_f_xy` / `child_f_xy` / `d_f_xy` 채움** | **152 / 152 / 152 = 100%** |
| `d_f_xy` 범위 | −0.1432 ~ +0.4315 |
| `d_cyclen` 범위 / 평균 | −2.926 ~ +5.796 / **−0.242** (r1 의 160행은 평균 **+34.7**) |

**§11-3 의 도구 소견은 해소되었다.** 이 wave 의 1차 반응 F_xy 가 이제 코퍼스에 **직접** 들어 있다 — HGD569 152/152, E1E2 두 cell 160/160, N1N2 159/160, T6T4 143/160.

#### move_class × direction 균형 (intervention 792행)

| move_class | outward | inward | neutral | 합 |
|---|---:|---:|---:|---:|
| `fresh_relocate` | 197 | 196 | 0 | 393 |
| `batch_swap` | 40 | 39 | 120 | 199 |
| `rewire_swap` | 0 | 0 | 100 | 100 |
| `batch_flip` | 19 | 21 | 60 | 100 |
| **합** | **256** | **256** | **280** | **792** |

**outward/inward 가 256/256 으로 정확히 균형** 잡힌다(r1 은 259/261). 계획 대비 shortfall 은 HGD569 의 8 chain 뿐이다.

---

### H. §13 · §14 처방 갱신

1. **§13.1 `batch_swap` (signed) — "실효 1 cell" → "2 cell, 38쌍".** 다만 두 cell 모두 유의하지 않고 T6T4 의 branch split 부호 반전(§5.2)이 살아 있으므로 **pooled 금지 · cell/branch-conditioned 진단용** 이라는 등급 자체는 유지한다. 바뀐 것은 "정보량 0" 이라는 사유가 없어졌다는 점이다.
2. **§13.1 `fresh_relocate` — HGD569 의 "delta 절대값 제외" 조항을 해제한다.** v2 의 vs-parent 한계효과는 다른 4개 cell 과 같은 baseline 위에 있다(§B). 191쌍 382행 전부 pooled 사용 가능하며, cell 이질성 보고 의무(G5)만 남는다.
3. **§13.2-5 정오 — `HGD569_f125` 를 "영구 prospective holdout" 으로 고정할 사유가 사라졌다.** AMENDMENT 1 E 가 예고한 재판단이다. 남는 사유는 원래의 하나뿐이다 — **이 cell 은 프로그램 실현가능 행이 0인 OOD arm**(prereg §7). 그것은 "라벨을 못 믿는다" 가 아니라 "분포가 다르다" 는 뜻이므로 **pooled 회귀 fold 에서의 배제가 아니라 cell-holdout 축의 한 칸**으로 쓰는 것이 맞다. 정책 v3 는 이 cell 을 **일반화 시험용 holdout cell** 로 유지하되 "진단 슬라이스" 강등은 철회한다.
4. **§14-5 종결.** AMENDMENT 1 D 로 대체되었던 항목이 실행·완료되었다. `HGD569_f125` 는 r2 에 **정상 cell 로** 재투입 가능하다.
5. **§14-1 은 유효하되 사유가 바뀐다.** round 1 은 signed cell 2개를 샀고 **실효도 2개**였다. 필요한 것은 축퇴 회피 기준이 아니라 **38쌍으로 −0.018 크기의 효과를 가르지 못한다**는 검정력이다.
6. **§15 에서 2건 삭제.** "HGD569 +34.71 EFPD 오프셋의 원인"(AMENDMENT 1 A 에서 확정, 본 절에서 측정 확증)과 "HGD569 `batch_swap` 형제가 동일 노심을 내는 기전"(존재하지 않는 현상)은 더 이상 미해결이 아니다. `twice_plus` 교락과 "`batch_swap` 의 참값" 2건은 남는다 — 후자는 세 wave 가 −0.1711 → −0.1442 → **−0.0178**(2 cell 38쌍)로 단조 감쇠를 이어갔다.

---

### I. 격리 잔여 8행 (stale label)

store 에는 `campaign == intervention_HGD569_f125` 인 **8행이 남아 있고**, AMENDMENT 1 C 가 붙인 **`failure = "alias_noop_P6_20260830"` 라벨을 그대로 이고 있다.** v2 가 라벨을 만들지 못한 record 이므로 upgrade 대상이 아니었다.

| record_id (앞 16) | r1 status | v2 status | store `converged` / `valid` | FOM |
|---|---|---|---|---|
| **`4cbfb4f682229f97`** | **converged** | error | False / False | **cyclen 763.934 · F_r 1.6983 · F_xy 1.7719 · node_peak 1.3818 — r1 의 결함값(P6 단일 노심)이 그대로 남아 있다** |
| `95f65b853b27f091` 외 6건 | error | error | False / False | 전부 NaN |

세 가지를 명시한다.

1. **failure 라벨이 사실과 다르다.** 8행 중 7건은 r1 에서도 `non_finite_flux` 였고 v2 에서도 그렇다 — 이들의 실패 원인은 alias no-op 이 아니라 **수렴 실패**다. `alias_noop_P6_20260830` 은 "이 행은 결함 wave 의 산물" 이라는 **격리 표식**으로만 읽어야 하며 실패 사유로 읽으면 안 된다.
2. **`4cbfb4f682229f97` 한 행만이 실질 위험이다.** 결함 FOM 을 값으로 들고 있는 유일한 잔여 행이다. 현재 `converged=False ∧ valid=False` 이므로 이 트리의 표준 필터(`converged == True`)에 걸러진다 — AMENDMENT 1 C 의 ⚠️ 경고("`valid=False` 만으로는 elite/학습에서 빠지지 않는다")가 지적한 구멍은 **이 행에 대해서는 닫혀 있다**. 다만 AMENDMENT 1 C 의 상태 기술("153행은 여전히 `converged=True` 다")은 **현재 store 에 대해 더 이상 성립하지 않는다**(152행은 v2 로 upgrade, 남은 1행은 `converged=False`). 어느 단계가 이 열을 내렸는지는 본 작업에서 확정하지 않았고, 관측 상태만 기록한다.
3. **corpus 에는 이 8행이 없다**(§G). `steps.parquet` 의 HGD569 152행은 전부 v2 라벨이다.

**권고**: 잔여 8행의 `failure` 를 사후 정정하지 않는다. 라벨을 고치면 격리 이력이 사라진다. 본 절을 참조하는 것으로 충분하며, r2 이전에 이 8 chain 을 재시도할 경우 `--campaign-suffix _v3` 로 같은 upgrade 경로를 쓴다.

---

### J. Provenance

| artefact | sha256 | bytes | 비고 |
|---|---|---:|---|
| `runs/intervention_wave_r1_hgd569_fix/intervention_HGD569_f125/ablation_results.jsonl` | `FC73CAB6BED9B201FEA5C197E9BECD525FF59FC51E1ED76896E8137D276244A0` | 302,984 | 160행 / converged 152 |
| `…/fxy_sidecar.jsonl` | `5C24536EB58131DC004E31B079F2DD12625AE1962FB51BBEE85FB77D4BA2E907` | 34,980 | 160행 / sane 152 |
| `…/subplan.json` | `1DDAF1AC9DA953A880E75A0EE852190B03A2A7E0CADADB99571391275CD4F23E` | 590,612 | 후보 record_id 집합·pattern **r1 subplan 과 동일** |
| `data/reports/fxy_backfill_199_intervention_hgd569_fix_20260831.csv` | `580617D397954068A4C6EAF7C6846BDA2E8E219A015978F977D71B956578874A` | 37,417 | 161행 / sane 152 |
| `data/store/records.parquet` (v2 merge 후) | `747D37AE2A50CC25742F98CC4770694354032EC6E060EEB34D54B75EB830CB50` | 22,782,850 | **76,693행 불변**; v2 152행 제자리 upgrade, 격리 8행 잔존(§I) |
| `data/policy/steps.parquet` (append 전) | `5FA791B8B31971793D44ECF3CF8568F516F4242A7338904DD3929DDD86CA8C37` | 9,499,252 | AMENDMENT 1 C 격리 후 값과 **일치** |
| `data/policy/steps.parquet` (append 후) | `2A34F7B38D83390F73E07FCC0A660516BEC91B92CE66103111AB577D911AF3E2` | 9,556,122 | 28,889행 × 85열 |
| `E:/lpopt_data/5_RL/backups/steps.parquet.bak_pre_hgd569_v2_20260831` | `5FA791B8…8C37` | 9,499,252 | append 전 백업, 비트 동일 |
| `intervention_wave.py` | `F402EEFED4BE60FF15E20EABA63F278A6EEFDE6C90497AD1302AF46FD253AD41` | 88,206 | **AMENDMENT 1 B 등록값 일치 — 본 작업 무편집** |
| `ablation_wave.py` | `1B94C7128F41685B6B3852527AD8FF6625414F781009AD1ED9F17CAE5F9280C1` | 40,237 | **prereg §8 고정값 일치, 무편집** |
| `ablation_analyze.py` | `E5BC29F90EE8749FD4B593A8A7783AB58776E313B7B9DD864ABE930B291880A2` | 19,452 | AMENDMENT(2026-08-30 23:xx) 값 일치, 무편집 |
| `mine_policy_corpus.py` | `5377151AAEC14F2F0ED0E582BB327A0BD8AC58D1EBBCDD575A818AED2528E577` | 95,333 | 동일, 무편집 |
| `data/design/intervention_wave_r1.json` | `F82CE02943893D5132FFEC9321ADFA1757C3CF6DC30624CF392E93C2D86FE20D` | 2,660,532 | **등록값 그대로, 무편집** |
| `data/design/intervention_wave_r1_s1i_pred.csv` | `5135B59E75F176C39F450FD05C8E489A1F5B100AA307A146687AA69FB4D8F3A0` | 417,800 | **개봉 전 등록본, 무편집** — §F 재채점의 예측 원천 |

#### 이 정오가 만든 분석 산출물 (`data/reports/`)

| file | sha256 (앞 16) | bytes |
|---|---|---:|
| `intervention_wave_r1_effects_by_cell_v2.csv` | `67B3F6B1E35FD1FB` | 7,191 |
| `intervention_wave_r1_effects_by_cell_adj_v2.csv` | `766AF661A9485275` | 7,253 |
| `intervention_wave_r1_effects_pooled_v2.csv` | `0FDB260AAEC86B8A` | 2,574 |
| `intervention_wave_r1_effects_pooled_adj_v2.csv` | `99F6ABC59D9D61A8` | 2,625 |
| `intervention_wave_r1_effects_by_burn_state_v2.csv` | `CFDF32CF4123A422` | 3,139 |
| `intervention_wave_r1_effects_by_burn_state_adj_v2.csv` | `A897C2F9C8B2B407` | 3,170 |
| `intervention_wave_r1_neutral_control_offset_v2.csv` | `43590E9F8B20D8D0` | 1,103 |
| `intervention_wave_r1_paired_by_cell_v2.csv` | `96E106D34B7AAAE1` | 2,863 |
| `intervention_wave_r1_paired_pooled_v2.csv` | `0201C129B9EA7D2E` | 822 |
| `intervention_wave_r1_parent_blocked_signs_v2.csv` | `A89F599AA4583CAF` | 9,959 |
| `intervention_wave_r1_rows_v2.csv` | `EB88EC25A99F9405` | 450,641 |

**r1 CSV 7종은 무편집이다** — §12 의 등록 sha(`C6E5860FF1938D50` · `295B86F2F3CA8483` · `06786076E5937C64` · `60135A0D136B3DFE` · `71A6D783FDB9407A` · `74796C9DDE66E4AB` · `4A34610803079670`)가 현재 파일과 전부 일치한다.

#### 대조 무결성

원 5개 cell dir 로 **현재 도구**를 돌리면 §3.1·§3.2·§3.4 의 발행값이 전부 재현된다(예: pooled `fresh_relocate` d_cyclen **−1.2353**, HGD569 dose **−9.967** [−14.78,−5.47], `rewire_swap|neutral` raw cyclen **+7.065**, HGD569 중립 오프셋 **+35.0533**). 따라서 r1 ↔ v2 의 모든 차이는 **도구 변경이 아니라 HGD569 라벨 변경 단독**에서 온다.

`analyze` 는 run dir 의 jsonl + sidecar 만 읽으므로 이 view 는 store 상태와 독립이다. junction 은 읽기 경로일 뿐이며 `runs/intervention_wave_r1/` 는 읽기·쓰기 어느 쪽으로도 변경되지 않았다.
