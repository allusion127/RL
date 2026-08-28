# APR1400 SDM / MTC 인허가 제한치 확정 보고서

- 작성일: 2026-07-26
- 목적: KNF LEU+ 시범연료봉 장전 인허가 과제의 LP 후보 판정 기준(SDM/MTC) 확정
- 조사 범위: 프로젝트 내부 문서·코드(우선순위 1), 공개 문헌(우선순위 2)
- 제약 준수: MASTER 미실행, 리포 코드 미수정, 원격 박스 미접속

---

## 0. 결론 요약 — 바로 설정에 넣을 값

```
# MTC (감속재온도계수) — 단위: pcm/°C, 부호는 물리부호 그대로
mtc_min_pcm_per_c = -54.0      # 최대 음(-) 허용
mtc_max_pcm_per_c =  +9.0      # 최대 양(+) 허용 (HZP 기준)
# (권고) HFP 상태에 대해서는 상한을 0.0 pcm/°C 로 별도 적용 — 4.2절 참조

# SDM (정지여유도) — 단위: pcm
sdm_required_pcm           = 10870.0   # = 10180 + 690
cea_allowance_pcm          = 10180.0   # DCD Table 4.3-8 Total reactivity allowance 10.18 %Δρ
net_worth_uncertainty_pcm  =   690.0   # DCD Table 4.3-9 Uncertainty in net rod worth 0.69 %Δρ
# 판정: margin = (W_ARI - worst_stuck_worth) - 10870 >= 0

# 참고(비교용, 판정치 아님) — DCD Table 4.3-9 1주기 EOC 기준
dcd_all_cea_worth_pcm = 16700.0   # PSCEA 제외 전 제어봉 삽입 worth
dcd_stuck_worth_pcm   =  5690.0   # 최대반응도 단일 CEA (고착 가정)
dcd_excess_pcm        =   140.0   # DCD 자체 수지식의 잉여 = 16700-5690-690-10180

# SDM 분기 뱅크 (반드시 정지뱅크 A, B 포함)
scram_banks            = [R1, R2, R3, R4, R5, B, A]
stuck_candidate_banks  = [R1, R2, R3, R4, R5, B, A]
```

**신뢰도**: MTC 창(窓) [-54, +9] pcm/°C 와 SDM 요구 10,870 pcm 은 로컬 DCD PDF 원문에서
**직접 확인 완료(높음)**. 기술지침서(Tech Spec) LCO 3.1.1 / 3.1.3 의 조항 인용값은
**확보 실패(5절 참조)** — 위 값은 "DCD 설계기준 유도값"이지 "TS LCO 인용값"이 아니다.

---

## 1. 출처 계층과 확인 방법

| 순위 | 출처 | 상태 |
|---|---|---|
| 1 | `2_LP/References/ARP1400 Design Control Document TIER 2.pdf` (20.2 MB, **로컬 읽기 성공**) | APR1400 DCD Tier 2 **Chapter 4 "Reactor" 단독본**, Rev. 3, 2018-08, 276쪽. 문서번호 APR1400-K-X-FS-14002-NP |
| 1 | `2_LP/MOCHA/config_apr1400.yaml` `sdm_mtc.licensing` 블록 (프로젝트가 실제 사용해 온 값) | 확인 완료 |
| 1 | `2_LP/MOCHA/sdm_mtc*.py` (4개 파일) + `2_LP/.claude/skills/master-sdm-mtc/references/` | 확인 완료 |
| 1 | `5_RL/lpopt/config.py` `SdmMtcConfig`, `5_RL/lpopt/search/sdm_mtc.py` (MOCHA 포팅본) | 확인 완료 |
| 1 | 실제 생산 산출물 `2_LP/0_Case/runs/20260705-190653_random_s116968488/_sdm_mtc/results.json` | 확인 완료 (`limits_source: "dcd_with_config_mtc"` — DCD PDF 파싱 성공 이력) |
| 2 | NRC 공개 DCD Ch.16 (Tech Spec, ML15006A055), Ch.4/Ch.16 SER | **접근 실패** — nrc.gov 가 403(Akamai Access Denied), web.archive.org 연결 불가 |

DCD 원문 확인은 로컬 PDF를 PyMuPDF로 텍스트 추출하여 해당 표를 직접 판독하는 방식으로 수행했다.
(계산 부하 최소, MASTER 미실행)

---

## 2. MTC (감속재온도계수) 제한치

### 2.1 판정 제한치 — 사고해석 포락값 (DCD Table 4.3-3)

DCD Tier 2, 4.3-46쪽, **Table 4.3-3 "Comparison of Core Reactivity Coefficients with Those
Used in Various Accident Analyses"** 의 "Coefficients Used in Accident Analyses" 절 원문:

| 사고 | DCD 원표기 (Δρ/°C ×10⁻⁴) | DCD 원표기 (Δρ/°F ×10⁻⁴) | pcm/°C 환산 |
|---|---:|---:|---:|
| CEA withdrawal, full/zero power | 0 / **+0.9** | (+0.5) | 0 / **+9.0** |
| CEA ejection, BOC full/zero power | 0 / **+0.9** | (+0.5) | 0 / **+9.0** |
| LOCA large break | **+0.9** | (+0.5) | **+9.0** |
| LOCA small break | 0 | — | 0 |
| Loss of flow | 0.0 | — | 0.0 |
| CEA misoperation, **Dropped CEA** | **−5.4** | (−3.0) | **−54.0** |

⇒ **판정 창: −54.0 ≤ MTC ≤ +9.0 [pcm/°C]**

| 항목 | 값 | 단위 | 조건 | 출처 | 신뢰도 |
|---|---:|---|---|---|---|
| MTC 상한 (most positive) | **+9.0** | pcm/°C | **HZP(zero power)**. DCD 각주(3): nominal Tavg = 308.9 °C 기준값 | DCD Tier 2 Table 4.3-3 | 높음 (원문 확인) |
| MTC 상한 (most positive) | **0.0** | pcm/°C | **HFP(full power)** — 사고해석은 전출력에서 MTC ≤ 0 을 가정 | DCD Tier 2 Table 4.3-3 | 높음 (원문 확인) / **적용은 사용자 확인 필요** |
| MTC 하한 (most negative) | **−54.0** | pcm/°C | Dropped CEA 해석 가정, EOC 최대 음값 포락 | DCD Tier 2 Table 4.3-3 | 높음 (원문 확인) |

단위 환산 검증: `+0.9×10⁻⁴ Δρ/°C = 9.0×10⁻⁵ = 9.0 pcm/°C`,
`−5.4×10⁻⁴ Δρ/°C = −54 pcm/°C`. °F 표기와도 정합 (`×9/5`: 0.5→0.9, 3.0→5.4).

> **주의**: 이 값들은 DCD가 **사고해석에 가정한 포락 계수**이다. 노심설계 후보가 이 창을
> 벗어나면 Chapter 15 사고해석의 초기조건 가정을 깨뜨리므로 인허가 판정 기준으로 타당하다.
> 다만 형식상 "Technical Specification LCO 3.1.3 의 MTC 한계"와 동일한 조항은 아니다(5절).

### 2.2 참고값 — 설계 예측치 (DCD Table 4.3-4), 판정치 아님

DCD Tier 2, 4.3-47쪽 **Table 4.3-4 "Reactivity Coefficients"** (1주기 기준). 환산 = 원값 ×10.

| 상태 | DCD (Δρ/°C ×10⁻⁴) | pcm/°C | 프로젝트 키 |
|---|---:|---:|---|
| BOC, Cold 20 °C, clean, 1238 ppm | +0.23 | +2.3 | — |
| BOC, HZP 291.3 °C, no CEA, clean, 1187 ppm | −0.28 | **−2.8** | `boc_hzp_aro` |
| BOC, HFP 308.9 °C, no CEA, clean, 1067 ppm | −1.01 | −10.1 | — |
| BOC, HFP 308.9 °C, no CEA, eq Xe, 817 ppm | −1.71 | **−17.1** | `boc_hfp_aro_eq_xe` |
| BOC, HZP, CEA bank 5/4/3 삽입, 817 ppm | −1.59 | **−15.9** | `boc_hzp_rodded` |
| MOC(3,000 MWD/MTU), HZP, no CEA, clean | +0.25 | +2.5 | — |
| MOC, HFP, eq Xe | −0.84 | −8.4 | — |
| EOC(10 ppm, 17,571 MWD/MTU), Cold 20 °C | +0.22 | +2.2 | — |
| EOC, HZP 291.3 °C, no CEA, HFP eq Xe | −3.18 | **−31.8** | `eoc_hzp_aro` |
| EOC, HFP, eq Xe, no CEA | −4.34 | **−43.4** | `eoc_hfp_aro_eq_xe` |
| EOC, HZP, rodded (bank 5/4/3) | −3.73 | **−37.3** | `eoc_hzp_rodded` |

프로젝트 `mtc_reference_pcm_per_c` 6개 값이 **DCD Table 4.3-4 와 소수점까지 일치**함을 확인했다.

---

## 3. SDM (정지여유도) 제한치

### 3.1 DCD 반응도 수지식

**Table 4.3-8 "CEA Reactivity Allowances (%Δρ)"** (DCD 4.3-52쪽):

| 반응도 성분 | %Δρ |
|---|---:|
| Fuel temperature variation | 1.18 |
| Moderator temperature variation | 2.46 |
| Moderator voids | 0.10 |
| CEA bite and part-strength CEA insertion | 0.25 |
| Accident analysis allowance | 6.19 |
| **Total reactivity allowance** | **10.18** |

**Table 4.3-9 "Comparison of Available CEA Worths and Allowances"** (DCD 4.3-53쪽):

| 조건 | %Δρ |
|---|---:|
| All CEAs inserted, hot, 308.9 °C (588 °F) | 16.70 |
| Total reactivity allowance, full power (Table 4.3-8) | 10.18 |
| **Stuck rod worth** | **5.69** |
| **Uncertainty in net rod worth** | **0.69** |
| **Excess reactivity** | **+0.14** |

DCD 자체 수지식: `16.70 − 5.69 − 0.69 − 10.18 = +0.14 %Δρ` (표의 Excess reactivity 와 정확히 일치).

### 3.2 확정 제한치

| 항목 | 값 | 단위 | 조건 | 출처 | 신뢰도 |
|---|---:|---|---|---|---|
| **SDM 요구 최소치** | **10,870** | pcm (=10.87 %Δρ) | 최대반응도 CEA 1개 고착 가정 후의 **정미(net) 가용 제어봉가**가 이 값 이상이어야 함 | DCD Table 4.3-8 (10.18) + Table 4.3-9 (0.69) 합산 유도 | **높음** (구성 성분은 원문 확인, 합산식은 DCD 수지식에서 유도) |
| CEA 총 반응도 여유 | 10,180 | pcm | full power 기준 | DCD Table 4.3-8 | 높음 |
| 정미 제어봉가 불확실도 | 690 | pcm | — | DCD Table 4.3-9 | 높음 |
| (비교) 전 CEA 삽입 worth | 16,700 | pcm | **1주기 EOC(17,571 MWD/MTU), hot 308.9 °C, PSCEA 제외** | DCD Table 4.3-9 / Table 4.3-6 | 높음 |
| (비교) 고착봉 worth | 5,690 | pcm | 완전 인출 위치의 단일 최대반응도 CEA | DCD Table 4.3-9 | 높음 |
| (비교) DCD 잉여 반응도 | +140 | pcm | 위 수지식의 결과 = 우리 코드의 `margin` 과 동일 정의 | DCD Table 4.3-9 | 높음 |

### 3.3 적용 조건 (중요)

- **고착봉 가정**: **적용함**. DCD 4.3-19쪽 4.3.2.4.3.6 원문 — *"Table 4.3-9 shows the reactivity
  worths of all CEAs and the highest reactivity worth of a single CEA in the fully withdrawn
  position for the end of the first cycle."* → 최대반응도 제어봉 1개가 완전 인출 상태로 고착.
- **사이클 시점**: DCD 수지식은 **1주기 EOC (17,571 MWD/MTU)** 기준. 우리 하네스는
  BOC·EOC 양쪽 HZP 상태에서 평가하므로 DCD보다 넓게(보수적으로) 본다.
- **온도/출력 조건**: DCD 표는 *hot, 308.9 °C* (= HFP Tavg) 표기. 우리 하네스는
  `*_hzp` 상태(`pload = 1e-6`)에서만 SDM을 계산하며, T/H 경계조건은 restart에서 승계된다
  (분기 덱에 `%GEN_THD` 없음; 프로젝트 core 설정 `tin = 290.6 °C`, `tavg = 307.25 °C`).
  → 저출력에서 Tavg → Tin ≈ 290.6 °C 로 수렴하므로 사실상 HZP. DCD의 291.3 °C 와 0.7 °C 차이.
- **뱅크 범위**: DCD **Table 4.3-6** "Total (all CEAs) **Without PSCEA** = −16.70 %Δρ (EOC)"
  → 16.70 은 **PSCEA(P뱅크) 제외** 값. 따라서 SDM 덱은 **A, B, R1~R5 (총 81개, P 제외)** 로
  구성해야 DCD와 정합한다. MOCHA(`2_LP`)는 이 범위를 쓰고 있다.
- **참고(별개 기준)**: DCD 4.3-20쪽 — *"only the minimum shutdown worth of 8.0 percent Δρ is
  assumed to be available at hot full power"* → Chapter 15 스크램 반응도 삽입에 쓰는
  **8.0 %Δρ (8,000 pcm)** 가정. SDM 판정치가 **아니며** 10,870 pcm 보다 느슨하다. 혼동 금지.

### 3.4 DCD Table 4.3-6 뱅크별 worth (%Δρ, 참고)

| 뱅크 | 0 MWD/MTU | 7,018 | 13,992 | 17,571 (EOC) |
|---|---:|---:|---:|---:|
| Shutdown CEAs (A+B) | −10.63 | −11.33 | −11.89 | **−12.32** |
| Regulating Group 1 | −1.33 | −1.58 | −1.49 | −1.49 |
| Group 2 | −1.03 | −1.03 | −1.11 | −1.16 |
| Group 3 | −0.77 | −0.87 | −0.86 | −0.88 |
| Group 4 | −0.43 | −0.42 | −0.48 | −0.49 |
| Group 5 | −0.30 | −0.36 | −0.36 | −0.36 |
| Part-Strength (P) | −0.19 | −0.25 | −0.31 | −0.33 |
| **Total without PSCEA** | **−14.49** | **−15.59** | **−16.19** | **−16.70** |
| Total with PSCEA | −14.68 | −15.85 | −16.50 | −17.03 |

**정지뱅크 A+B 가 전체 제어봉가의 약 74 % 를 차지**한다 (EOC 12.32 / 16.70). → 5절 확인사항 (1) 참조.

CEA 뱅크별 rod 수 (DCD Figure 4.3-36, 코드 `EXPECTED_DCD_ROD_COUNTS`):
`A=16, B=20, R1=8, R2=12, R3=12, R4=8, R5=5, P=12, S=8` (합 101; S=예비, ROD_MAP 제외)

---

## 4. 프로젝트 코드의 단위·부호 규약 (저장/비교 시 필수 준수)

근거 파일:
- `2_LP/MOCHA/sdm_mtc_parse.py`, `sdm_mtc.py`, `sdm_mtc_types.py`, `sdm_mtc_io.py`
- `5_RL/lpopt/search/sdm_mtc.py` (MOCHA 포팅본, 2026-07-17), `5_RL/lpopt/config.py`
- `2_LP/.claude/skills/master-sdm-mtc/references/sign-units.md`

### 4.1 반응도

```python
rho_pcm(k) = (k - 1) / k * 1e5        # 단위 pcm
```

### 4.2 MTC

| 항목 | 규약 |
|---|---|
| 단위 | **pcm/°C** (섭씨! °F 아님). 저장 필드명 `value_pcm_per_c` |
| 부호 | **물리부호 그대로**. 음수 = 음의 MTC(정상), 양수 = 양의 MTC |
| 판정 | `mtc_min_pcm_per_c <= value <= mtc_max_pcm_per_c` → `pass_limit` |
| `mtc_max` 의미 | **최대 허용 양(+)값** (most-positive allowed) |
| `mtc_min` 의미 | **최대 허용 음(−)값** (most-negative allowed) |
| MASTER 출력 | `%EXE_RHO mtc 1` + 명시 `dtm`. 결과행 예: `... 1.001165  MTC  -23.12 PCM/C` |
| 파서 | `PCM/C` **단위 토큰 필수**. 없으면 `None` (입력 에코 `mtc 0` 오독 방지 — 감사 T7-3) |
| `mtc_output_units` | `pcm_per_c` → ×1 / `drho_per_c_1e-4` → **×10**. **`pcm_per_c` 고정 권장** |
| 2점 폴백 | rows `[base, T+Δ, T−Δ]` **정확히 3행**일 때만: `(rho(k[1]) − rho(k[2])) / (2Δ)`. 아니면 `None` |
| Δ (`mtc_delta_c`) | **5.0 °C**. 덱의 `%EXE_RHO ... dtm` 에 그대로 기입되어 폴백 스케일과 결합 |
| 증거 없음 처리 | `pass_limit is None` → **케이스 FAIL** (감사 T7-1, 공백 통과 방지) |
| 기하 | `mtc_geometry: quarter` (1/4 노심). SDM은 항상 full-core |

> ⚠️ **최악의 단위 버그(기지의 사고 사례)**: 2026-06 MOCHA 적대적 감사 F1 —
> 크기 기반 자동 재스케일이 `+5.0 pcm/°C` 를 `+50 pcm/°C` 로 바꿔 **허위 인허가 FAIL** 을
> 냈다. 현재는 `mtc_text_to_pcm_per_c(value, units)` 로 **명시적·결정적** 환산만 한다.
> `mtc_output_units` 를 `drho_per_c_1e-4` 로 두면 10× 오류가 재발한다.

### 4.3 SDM

| 항목 | 규약 |
|---|---|
| 단위 | **pcm** (전 필드) |
| 부호 | **삽입이 반응도를 낮추면 worth 는 양(+)** |
| ARO 기준 | 분기 1은 `%EXE_STD boron tr tr` (임계붕산 탐색) → **`rho_ARO ≡ 0`** (실측 확인) |
| 계산식 | `W_ARI = rho_ARO − rho_ARI` <br> `worth_i = rho_stuck_i − rho_ARI` <br> `available = W_ARI − max(worth_i)` <br> `margin = available − sdm_required_pcm` |
| 판정 | **`margin >= 0` 이면 pass** |
| 상태 | `*_hzp` 상태에서만 계산 (`boc_hzp`, `eoc_hzp`). HZP 상태가 없으면 에러(감사 T7-11) |
| 건전성 체크 | 모든 고착봉 분기에 대해 `rho_ARO >= rho_stuck_i >= rho_ARI` 단조성 검사(감사 T7-9) |
| MASTER 제약 | `%ROD_CFG` 그룹 수 ≤ **92** (93에서 크래시). A+B+R1~R5 = 81 → OK, P(12) 추가 시 93 → 크래시 |

`margin` 의 정의는 DCD Table 4.3-9 의 **"Excess reactivity"** 행과 정확히 동일하다
(DCD 값 = +140 pcm).

### 4.4 실측 생산 산출물 예시 (규약 검증)

`2_LP/0_Case/runs/20260705-190653_random_s116968488/_sdm_mtc/results.json`:

```
limits_source : "dcd_with_config_mtc"   ← DCD PDF 파싱 성공 (SDM 상수=DCD, MTC 창=YAML)
MTC boc_hzp (quarter) : +1.21 pcm/°C   → pass (source: EXE_RHO_pcm_per_c)
SDM boc_hzp : rho_ARO=0.0, rho_ARI=-9684.8, W_ARI=9684.8 pcm,
              worst_stuck=B14 (1606.9 pcm), available=8078.0 pcm,
              required=10870.0, margin=-2792.0 → FAIL
```

---

## 5. 불확실 항목 및 사용자 확인 필요 사항

### (1) 🔴 **`5_RL/lpopt` 의 SDM 뱅크 기본값이 정지뱅크 A·B 를 누락** — 즉시 확인 필요

`5_RL/lpopt/config.py` `SdmMtcConfig`:
```python
scram_banks           = ["R1","R2","R3","R4","R5"]   # ← A, B 없음
stuck_candidate_banks = ["R1","R2","R3","R4","R5"]   # ← A, B 없음
```
`5_RL/lpopt/search/sdm_mtc.py` `BranchSpec` 기본값도 동일.
반면 `2_LP/MOCHA/config_apr1400.yaml` 는 `[R1,R2,R3,R4,R5,B,A]` (81 그룹).

참조 설정 `5_RL/config/user_criteria_ref.inp` 의 `[sdm_mtc]` 절도 `mtc_max_pcm_per_c`,
`sdm_required_pcm`, `top_k` 만 지정하고 **뱅크 목록을 지정하지 않아 R1~R5 기본값으로 떨어진다.**

DCD Table 4.3-6 기준 R1~R5 합계는 EOC에서 **4.38 %Δρ (4,380 pcm)** 에 불과하고,
정지뱅크 A+B 가 **12.32 %Δρ** 를 담당한다. A·B 를 뺀 상태로 10,870 pcm 요구를 걸면
**물리적으로 통과 불가능**하며 모든 후보가 FAIL 한다.
→ **`5_RL` 설정을 `[R1,R2,R3,R4,R5,B,A]` 로 맞출 것** (또는 의도된 축소 범위인지 확인).
(본 보고서는 조사 전용이므로 코드는 수정하지 않았다.)

### (2) 🟡 HFP 상태의 MTC 상한 — 현재 +9 를 전 상태에 일괄 적용

DCD Table 4.3-3 는 전출력에서 MTC = 0 을, 영출력에서 +0.9×10⁻⁴ Δρ/°C 를 가정한다.
현재 코드는 `boc_hzp / boc_hfp / eoc_hzp / eoc_hfp` **네 상태 모두에 [-54, +9]** 를 적용한다.
엄격 적용 시 **HFP 상한은 0.0 pcm/°C** 여야 한다.
→ 상태별 창을 둘지, 보수적으로 전 상태 +9 를 유지할지 **사용자 판정 필요**.
(참고: DCD Table 4.3-4 설계 예측치는 HFP에서 −10.1 ~ −43.4 pcm/°C 로 여유가 크다.)

### (3) 🟡 `criteria.mtc_limit` 은 상한만 덮어씀

`5_RL/lpopt/cli.py` L686-697 / `LicensingLimits.with_overrides` — `mtc_limit` → `mtc_max_pcm_per_c`,
`sdm_limit` → `sdm_required_pcm` 만 적용된다. **`mtc_min` 을 바꾸는 CLI/설정 경로가 없다.**
음(−) 측 한계를 조정하려면 `[sdm_mtc] mtc_min_pcm_per_c` 를 직접 써야 한다.

### (4) 🟡 기술지침서(Tech Spec) LCO 3.1.1 / 3.1.3 인용값 — **확보 실패**

- 로컬 DCD PDF 는 **Tier 2 Chapter 4 단독본**(276쪽)이며 Chapter 16(Technical Specifications)
  및 Chapter 15 는 프로젝트 트리에 없다.
- NRC 공개본(ML15006A055 = DCD Tier 2 Ch.16, ML18227A061 = Ch.16 SER, ML18067A292 = Ch.4 SER)은
  **nrc.gov 가 403 Access Denied(Akamai) 로 차단**, web.archive.org 도 연결 불가하여
  본 조사 환경에서 원문 확보에 실패했다.
- 일반적으로 CE 계열 표준기술지침서(NUREG-1432)는 **SDM 값을 TS 본문이 아니라 COLR
  (Core Operating Limits Report)에 위임**한다. 따라서 "APR1400 TS LCO 3.1.1 의 SDM 숫자"는
  플랜트/사이클별 COLR 값일 가능성이 높다. **이 값은 본 보고서에서 제시하지 않는다(추정 금지).**
- **잠정 결론**: 본 과제의 판정치는 **DCD Tier 2 Ch.4 Table 4.3-3 / 4.3-8 / 4.3-9 에서
  유도한 값**을 사용한다. 이는 프로젝트가 이미 검증·사용해 온 값이며 원문 대조를 마쳤다.
  최종 인허가 문서화 시에는 KHNP/KNF 로부터 해당 사이클 **COLR 및 TS 3.1.1/3.1.3** 값을
  받아 재확인할 것을 권고한다.

### (5) 🟡 평형주기 노심의 제어봉가가 DCD 대비 크게 낮음 (물리/덱 검토 필요)

실측: BOC HZP 에서 `W_ARI = 9,685 pcm`. DCD 1주기 BOC(PSCEA 제외) = **14,490 pcm**, EOC = 16,700 pcm.
33 % 이상 낮다. 10,870 pcm 요구를 걸면 구조적으로 FAIL 한다(실제 −2,792 pcm).
제한치 문제가 아니라 **평형주기 노심 / rod 모델 / 분기 덱**의 문제일 수 있으므로,
SDM 을 게이트로 쓰기 전에 별도 검토가 필요하다. (LEU+ 고농축 평형노심은 DCD 1주기 대비
붕소가·중성자 스펙트럼이 달라 제어봉가가 낮아지는 경향이 있으나, 33 %는 과도해 보인다.)

### (6) 🟢 경미 — 노심 온도 조건 차이

프로젝트 `core.tin = 290.6 °C`, `tavg = 307.25 °C` vs DCD HZP 291.3 °C / HFP Tavg 308.9 °C.
차이 0.7 ~ 1.65 °C. MTC/SDM 판정에 유의한 영향은 없을 것으로 판단되나 문서화 시 명기 권장.

---

## 6. 부록 — DCD 원문 확인 이력

| 표/절 | DCD 쪽 | PDF 쪽(0-base) | 확인 내용 |
|---|---|---|---|
| Table 4.3-2 | 4.3-45 | 135 | Keff / 반응도 데이터 (1주기) |
| Table 4.3-3 | 4.3-46 | 136 | **MTC 사고해석 포락값 → [-54, +9] pcm/°C 확정** |
| Table 4.3-4 (1/2) | 4.3-47 | 137 | **MTC 설계 예측치 6종 → 프로젝트 참조값과 일치 확인** |
| Table 4.3-4 (2/2) | 4.3-48 | 138 | 밀도/출력/보이드/압력 계수 |
| Table 4.3-6 | 4.3-50 | 140 | **뱅크별 worth, Total without PSCEA = 16.70 %Δρ (EOC)** |
| Table 4.3-8 | 4.3-52 | 142 | **Total reactivity allowance = 10.18 %Δρ** |
| Table 4.3-9 | 4.3-53 | 143 | **Stuck 5.69 / Uncertainty 0.69 / Excess +0.14 %Δρ** |
| Table 4.3-10 | 4.3-54 | 144 | 봉삽입 형상별 Fr (참고) |
| 4.3.2.4.3.6 | 4.3-19 | 109 | 고착봉 정의 — "highest reactivity worth of a single CEA ... end of the first cycle" |
| 본문 | 4.3-20 | 110 | "minimum shutdown worth of **8.0 percent Δρ** ... at hot full power" (Ch.15 스크램 가정) |
| Figure 4.3-36 | — | — | CEA 그룹 배치 (코드가 241 박스 파싱, 뱅크별 rod 수 검증) |

문서 식별: `APR1400 DESIGN CONTROL DOCUMENT TIER 2 / CHAPTER 4 REACTOR /
APR1400-K-X-FS-14002-NP / REVISION 3 / AUGUST 2018`
