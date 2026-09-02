# Pre-registration — Campaign A round 1: Causal Move Atlas (`intervention_wave_r1`)

- 작성일 2026-08-29 · 대상 박스 HOST_199 · 상태 **등록 완료, 미발사(nothing launched)**
- 근거 검토서: `../RL_core_loading_engineer_AI_review_2026-08-29.md` §7.2(D1) · §7.4(Campaign A) · §7.6(표집 비율) · §7.7(split 원칙)
- 도구: `intervention_wave.py` (신규) — `ablation_wave.py`의 enumerator/annotator/runner/kit builder를 **import 재사용**, `ablation_wave.py`는 **수정하지 않음** (그 sha256은 `ablation_wave_prereg_20260815.md` §8/Amendment 2에 고정되어 있고 199에서 실제로 돈 artefact다)
- 선행 결과: `ablation_wave_results_20260815.md`, `batchswap_wave_results_20260815.md`, `policy_v2_results_20260817.md` §8–§9, `produce_fxyera_r1_prereg_20260829.md`

---

## 1. 이 wave가 존재하는 이유

`policy_v2_results_20260817.md` §1의 감사 결론이 출발점이다: 현재 era에서 도달 가능한
lineage edge 6,318개 중 **6,297개가 이미 채굴되었고, 남은 것은 f113의 21개가 전부였다.**
> "un-mined current-era data는 없다; 다음 증분은 harvest가 아니라 **측정**해야 한다."

동시에 `ablation_wave_results_20260815.md`는 **한 셀에서** 인과적 답을 냈다.
`T6_T4/f121/paramA`에서 outward fresh loading은 cycle length를 **깎는다**
(fresh_relocate out−in `d_cyclen` **−1.7415** [−2.1336, −1.3342], sign 0/10, p=0.002;
dose slope −8.37 EFPD/unit). 관측 코퍼스의 `+0.093`은 감쇠가 아니라 **부호가 틀렸다**
(wave 재계산 −0.635). 그러나 이 값이 **한 셀·한 library·한 restart의 사실인지, 장전
물리의 사실인지**는 지금 알 수 없다.

또 `batchswap_wave_results_20260815.md`는 같은 셀 안에서조차 일반화가 깨지는 것을 보였다:
ablation의 parent당 n=4 추정치는 213-chain 심층 표본에 대해 **5.8배 과대추정**이었고
(`1165441c31ea` improving 0.500 → 0.086), 방향 순서(inward > outward)는 **철회**되었으며,
213개 chain 중 **158개가 ~618 EFPD 가지**에 몰려 있어 ~625 가지에서의 batch_swap 거동은
사실상 미측정이다.

Campaign A round 1은 이 셋을 동시에 친다: **여러 셀에서**, **parent 안에서 대칭쌍으로**,
**F_xy를 1차 반응으로** 단일 move의 인과 효과를 측정한다.

---

## 2. 등록 설계

### 2.1 셀 (5개, F_xy frontier)

`data/store/records.parquet`의 f_xy 라벨 6,236개는 소수의 셀에 몰려 있다. 그중 상위
5개를 그대로 쓴다 — 예측이 아니라 **측정된 F_xy**가 parent 순위의 근거여야 하기 때문이다.

| cell | key | kit | 측정 f_xy (store) | 비고 |
|---|---|---|---:|---|
| `T6T4_f121` | `T6_T4/f121/paramA` | paramA | 746 | ablation/batchswap wave가 돈 셀. 직접 비교 가능 |
| `HGD569_f125` | `P6253Z1G06N24_P6253Z2G10N24/f125/paramA` | paramA | 104 | high-Gd 5.694 w/o. **프로그램 실현가능 행이 0개** (§2.4) |
| `E1E2_f121` | `E1_E2/f121/ga80` | ga80 | 654 | ga80 anchor |
| `E1E2_f109` | `E1_E2/f109/ga80` | ga80 | 156 | 같은 pair, feed 레버(잔류시간) |
| `N1N2_f113` | `N1_N2/f113/ga80` | ga80 | 172 | pin-burnup 캠페인 셀 |

검토서 §7.4는 "100개 이상의 context"를 권고한다. round 1은 **5개**다. 이것은 축소가
아니라 순서다: 5개 셀 × 20 parent × 8 move = 800 chain ≈ **14.4 h**이고, 100 context는
같은 도구로 round 2 이후에 쌓는다. `intervention_wave.py`는 셀 목록을 상수
(`CELLS_R1`)로만 들고 있고 `plan/score/run/analyze/corpus` 전 경로가 N-셀이므로,
round 2는 코드 변경 없이 `CELLS_R1`에 줄을 더하는 일이다.

### 2.2 라이브러리 라우팅 — **kit는 둘로 쪼갠다**

`ProduceDriver._run_library_id` / `resolver.build_case_resolver`는 자산을 **라이브러리별로**
라우팅한다: paramA 셀은 design package + registry alias bridge + package 자신의 `%GEN_DIM`
dims `(40, 42)`, ga80 셀은 `FEASIBLE_PACKAGE` + `LIBRARY_DIMS` `(83, 85)`.
ablation runner는 **하나의 `--package`에서 하나의 resolver**를 만들므로 같은 제약을 공유한다.

**따라서 run은 kit별로 나눈다** — `intervention_wave.py run`은 혼합 선택을 거부한다
(`refusing to run cells of ['ga80','paramA'] in one invocation ...`), 그리고 `--package`의
library dims가 요청된 kit와 맞지 않으면 역시 거부한다.

- kit **paramA** — `T6T4_f121`, `HGD569_f125` → `data/design/package`, dims (40, 42)
- kit **ga80** — `E1E2_f121`, `E1E2_f109`, `N1N2_f113` → `FEASIBLE_PACKAGE`, dims (83, 85)

배포는 하나다: `run_intervention_wave_r1_199.bat`이 두 번 **순차 호출**한다. 두 개의
resolver kit, 하나의 launcher family, 하나의 MASTER 큐.

### 2.3 parent 선정 (셀당 20)

**joint-clean** = converged **AND** `f_r`/`node_peak`/`cyclen`/`cbc_max` 4축이 모두 라벨됨.
delta를 낼 수 없는 행은 D1 행이 될 수 없으므로 라벨 결측은 **hard gate**다.
프로그램 실현가능성(F_r≤1.55, F_q≤2.41, CBC≤1600, |AO|≤0.30)은 **hard gate가 아니라 tier**다 (§2.4).

큐는 등록 순서대로:

1. `fxy_rank` — 측정된 `f_xy` 오름차순 (frontier. 예측값이 아니라 **측정값**)
2. `fr_rank` — 아직 f_xy가 없는 행의 `f_r` 오름차순 (등록된 fallback)
3. `random` — joint-clean 풀에서 seed 고정 균등 추출, **20개 중 5개 = 25%**
   (검토서 §7.6의 "25% 무작위/space-filling". optimizer가 이미 사는 곳에서만
   atlas를 재지 않기 위한 것이고, §6.9의 편중 지적에 대한 직접 대응이다)

모든 채택은 ablation wave와 동일한 **쌍별 69-slot Hamming ≥ 12** 다양성 게이트를 통과한다.

### 2.4 실현가능 행이 0인 셀 — HGD569

`P6253Z1G06N24_P6253Z2G10N24/f125`는 converged 121행 중 **프로그램 실현가능 0행**이다
(최량 F_r 1.6036 vs 한계 1.55). 실현가능성을 hard gate로 두면 이 셀은 통째로 빠지는데,
그것은 검토서 §6.9가 지적한 "데이터가 optimizer가 이미 사는 곳에만 있다"를 그대로 재생산한다.
등록 결정: **feasible tier를 먼저 소진하고, 부족하면 converged+labelled tier로 확장**하며
parent의 `family`에 `_infeasible` 접미사와 `program_feasible: false`를 남긴다.
HGD569의 20 parent는 전원 이 tier에서 나왔고, 이는 매니페스트에 명시된다
(`n_feasible_parents: 0`). round 1의 **adversarial/OOD arm**(검토서 §7.6의 10%)이 이 셀이다.

### 2.5 parent당 8 move — 대칭쌍 · burn-state 균형

| arm | class | 수량 | 추출 규칙 |
|---|---|---:|---|
| paired | `fresh_relocate` | outward 2 + inward 2 | 같은 parent·같은 burn-state의 **dose-matched 형제** |
| paired | `batch_swap` | outward 1 + inward 1 | 동일 |
| neutral control | `rewire_swap` | 1 | `swap_span` 기준 분산. 구조적으로 방향이 없음 |
| randomized | `batch_flip` | 1 | 방향을 **parent마다 무작위**(seed 고정) |
| | | **8** | |

- `rewire_swap`은 fresh 집합도 batch label도 건드리지 않으므로 `d_fresh_enr_r_center`가
  **항등적으로 0**이다(ablation prereg §3). 방향 stratum이 아니라 **중립 대조군**이다.
- `batch_flip`은 fresh batch multiset을 바꾸므로 **총 fresh 농축량이 변한다**
  (`|d_fresh_enr_mass|` 최대 T6T4 **1.2017**, HGD569 **0.7350**). 반응성 보존이 깨지므로
  outward/inward 쌍을 leakage 대조로 읽으면 안 된다 (ablation prereg §3b: batch_flip은
  instrument가 **아니다**). 그래서 쌍이 아니라 **parent 간 무작위화**로 균형을 맞춘다.
- `fresh_relocate` / `batch_swap` / `rewire_swap`은 이번 계획에서 `|d_fresh_enr_mass|`
  최대 **0.000000** — 반응성 보존이 측정으로 확인된 instrument다.

**burn-state 축** (검토서 §7.4의 "burn state" 열, §7.2의 "once/twice-burnt swap"):
move가 건드린 orbit unit의 parent 내 **잔류 깊이** 최댓값으로 `fresh`(0) / `once`(1) /
`twice_plus`(≥2) / `center`를 정한다. 깊이는 `GeneralOrbitGenome._depths()` —
`mine_policy_corpus.residence_profile`이 쓰는 바로 그 resolver — 에서 온다.
각 stratum의 슬롯은 burn-state 그룹에 **round-robin으로 먼저 배분**되고, 그 다음
그룹별 dose 순서의 **분위 중점**에서 뽑힌다.

> **등록된 예외 — 방향 축퇴(direction degeneracy).** ga80 두 pair는 fresh 타입의
> `u_avg_enrichment`가 동일하다(`E1`=`E2`=5.000, `N1`=`N2`=5.400). 그러면 `batch_swap`과
> `batch_flip`은 농축 질량을 옮기지 않으므로 `fresh_enr_r_center`를 **정확히 0만큼**
> 바꾼다 — `rewire_swap`과 같은 의미로 방향이 **구조적으로 없다**. 이런 클래스는
> shortfall로 보고하지 않고 **같은 크기의 중립 arm**(paired 클래스면 2k개)으로 뽑으며
> `pair_role='neutral_degenerate'`로 표시한다. Gd 배치를 바꾸는 move는 실제 물리이고,
> 다만 **농축 가중 반경 기술자에 보이지 않을 뿐**이므로 측정 대상에서 빼면 안 된다.
> 이 규칙이 없었다면 ga80 3개 셀에서 parent당 8 chain 중 3개가 구조적 공집합에 배정되어
> **ga80 예산의 3/8이 빈 stratum에 쓰였을 것**이다.
> 매니페스트의 셀별 `direction_regime`이 열거된 census에서 이 판정을 기록한다.

**교차 stratum backfill 없음.** 대칭 형제를 못 만들면 그 쌍은 그냥 뽑지 않고 shortfall로
기록한다. 쌍 예산으로 산 짝 없는 outward child는 within-parent 대조를 다시 between-parent
비교로 되돌려 놓는다 — 그것이 이 캠페인이 대체하려는 바로 그것이다.
(round 1 실측: **shortfall 0**.)

### 2.6 dedup

각 자식의 `record_id = compute_record_id(pattern, library, pair, PRODUCE_DECK_KNOBS)`를
store에 대해 조회한다. 이미 라벨된 자식은 **free 라벨**로 매니페스트에 실리되 MASTER 슬롯을
쓰지 않는다(`source='free'`, 총 26개). paid 800개는 **record_id가 전부 유일**하다
— batchswap wave의 §5 결함(parent별로만 dedup해 7개 충돌, 예산 3.2% 낭비)을 재현하지 않는다.

---

## 3. 등록된 계획 (실측치, `data/design/intervention_wave_r1.json`)

`written_utc 2026-08-29T08:13:02Z` · **paid 800 · free 26 · shortfall 0 · parent 100**

| cell | kit | parents | feasible | parent F_xy | parent F_r | Hamming min/med/max | 이웃/parent | free | restart |
|---|---|---:|---:|---|---|---|---:|---:|---|
| T6T4_f121 | paramA | 20 | 20 | 1.5491–1.6766 | 1.4812–1.5463 | 12/36/47 | 1,589 | 22 | `native:MAS_RST.APRQ_10_0615.11` |
| HGD569_f125 | paramA | 20 | **0** | 1.6421–1.8335 | 1.6036–1.7361 | 12/33/49 | 1,576 | 1 | `pair_ecore:MAS_RST.APRQ_11_0705.02` |
| E1E2_f121 | ga80 | 20 | 20 | 1.5295–1.6629 | 1.4636–1.5407 | 12/32/52 | 1,589 | 1 | `native:MAS_RST.APRQ_11_0635.19` |
| E1E2_f109 | ga80 | 20 | 20 | 1.5923–1.6670 | 1.4787–1.5450 | 12/24/32 | 1,461 | 0 | `pair_feed:MAS_RST.APRQ_11_0615.88` |
| N1N2_f113 | ga80 | 20 | 20 | 1.5457–1.6486 | 1.4908–1.5456 | 12/35/50 | 1,487 | 2 | `pair_feed:MAS_RST.APRQ_11_0677.23` |

**stratum 배분 (paid 800)**

| move_class | outward | inward | neutral | 합 |
|---|---:|---:|---:|---:|
| `fresh_relocate` | 200 | 200 | 0 | 400 |
| `batch_swap` | 40 | 40 | 120 | 200 |
| `rewire_swap` | 0 | 0 | 100 | 100 |
| `batch_flip` | 19 | 21 | 60 | 100 |

- 대칭쌍 **240쌍** (T6T4 60, HGD569 60, E1E2_f121 40, E1E2_f109 40, N1N2_f113 40).
  paramA 셀은 `batch_swap`이 signed이므로 parent당 3쌍, ga80 셀은 축퇴이므로 2쌍이다.
- burn-state: `once` 420 · `fresh` 300 · `twice_plus` 80. `twice_plus`는 depth-2 edge가
  존재하는 feed에서만 나온다 (f121은 `depth2_edges = 0`이라 구조적으로 0; f109/f113이
  40개씩 공급). 이것은 편향이 아니라 feed 산술의 결과이며, 분석은 셀 내부에서 층화한다.

**dose (paid `fresh_relocate`의 `|d_fresh_enr_r_center|` 중앙값 / 최대)**
T6T4 0.0607/0.0996 · HGD569 0.0627/0.1028 · E1E2_f121 0.0576/0.0993 ·
E1E2_f109 0.0555/0.0588 · N1N2_f113 0.0622/0.0854.

> dose 척도는 **셀 간 비교 불가**다. `fresh_enr_r_center`는 농축 가중 1차 모멘트라
> 농축 대비가 큰 paramA 셀과 축퇴 ga80 셀에서 스케일이 다르다. batchswap wave가
> ablation의 dose slope −9.41을 −21.19로 갱신하며 CI가 겹치지 않았던 것과 같은 이유로,
> **부호와 셀 내 순위를 보고하고 교정된 상수는 보고하지 않는다.**

**표집 규칙에서 ablation과 의도적으로 다른 한 곳.** ablation의 `_pick`은 랭크를
`[0, n-1]` **양끝 포함**으로 배치하므로 k=1이면 stratum의 **최소 dose**를, k=2면 최소와
최대를 집는다. quota가 3이던 ablation에서는 합리적이었지만 parent당 1–2개인 이 wave에서는
가장 작은 개입만 사게 된다 — 개발 중 `E1_E2/f109`에서 paid dose 중앙값 **0.0068**
(이웃 중앙값 0.0757의 1/11)로 실측되었고, 이는 실제 효과가 노이즈 바닥 아래로 숨는 영역이다.
그래서 `intervention_wave.quantile_ranks`는 **등확률 구간의 중점**`((i+0.5)/k)`을 쓴다
(k=1 → 중앙값, k=2 → 사분위). 위 표의 0.0555–0.0627이 그 결과다.
`ablation_wave._pick`은 **건드리지 않았다.**

---

## 4. F_xy 획득 — 그리고 `ablation_wave`를 고치지 않고 메운 두 구멍

F_xy(MASTER FXYP, pin planar, 하드 한계 **1.65**)는 **`MAS_OUT`에만** 인쇄된다.
`MAS_SUM`에는 FXYP 열이 없고 정규 파서는 `MAS_SUM`만 읽는다
(`produce_fxyera_r1_prereg_20260829.md` §2). runner는 `harvest_maps=True`로 만들어지고
이는 `keep_success`를 강제하므로(`verify.py:837`) 수렴한 chain의 **최종 평형 사이클 디렉터리가
MAS_OUT과 함께 살아남는다**. 추가 MASTER 사이클 비용은 0이다.

그런데 `ablation_wave`의 `run`은 f_xy 열보다 먼저 쓰였다: `WaveOutcome`을 jsonl로 직렬화할 때
`fxy`를 빼고, kit builder는 그 jsonl에서 outcome을 재구성하므로 **verifier가 파싱한 F_xy가
바닥에 떨어진다**. `ablation_wave.py`는 수정 금지이므로 두 개의 **작고 문서화된 rebinding**을
`lpopt.search.verify` 쪽에, `try/finally` 스코프 안에서 건다:

1. `WaveVerifier` → `evaluate_wave`가 각 outcome의 `FxyResult`를 셀별
   `fxy_sidecar.jsonl`에 덧붙이는 서브클래스. 파싱 실패는 wave를 죽이지 않는다
   (`fxy_from_work_dir`의 계약과 동일).
2. `outcome_to_record` → jsonl에서 재구성된 record에 sidecar의 `f_xy`/`f_xya`를 채우는 래퍼.

세 번째 rebinding은 자산 라우팅용이다: `CaseAssetResolver`에 paramA용
`registry_aliases` / `fuel_library` / `synth_root`를 주입한다 (ga80은 `library_id`만,
즉 ablation의 호출과 바이트 동일). `synth_root`가 없으면 HGD569는 packaged template deck이
없어 모든 chain이 `MissingCaseAssetError`로 죽는다 — dry-run으로 확인했고, 지금은
`synth_decks/P6253Z1G06N24_P6253Z2G10N24/MAS_INP_cy12.inp`(sha 앞 16 `cddba86904810b7c`)로 해소된다.

**restart provenance 게이트.** ablation runner의 자체 가드는 `fallback_level != 0` 거부인데,
그것은 parent가 native restart에서 나온 셀에는 맞고 **native restart가 아예 없는 셀에는 틀리다.**
HGD569 f125의 store 행 144개 중 120개가 `pair_ecore:MAS_RST.APRQ_11_0705.02`에서 라벨되었으므로
거기서의 level-3 해석은 drift가 아니라 **parent가 지고 있는 바로 그 연소 이력**이다.
그래서 `intervention_wave.check_restart`가 **run의 restart == parent의 restart**를 검사하고,
`--allow-fallback`은 그 게이트를 통과시키는 열쇠가 아니라 ablation 가드를 여는 열쇠일 뿐이다.
dry-run 확인:
`[run] restart pair_ecore:MAS_RST.APRQ_11_0705.02  fallback_level=3  parents label(s) ['pair_ecore:MAS_RST.APRQ_11_0705.02']`.

---

## 5. 가설 — 사전 등록

각 가설은 **셀별로** 판정하고, 그 다음 pooled로 판정한다. 1차 통계량은 **쌍 내부
(outward − inward) 차이의 정확 이항 부호검정**이다(zeros drop, 양측). 2차는 parent 평균에
대한 부호검정(분석 단위 = parent)과 stratum별 평균/중앙값 delta다.

### H1 — leakage 매개의 부호는 셀마다 같은가

> **H1.** 모든 5개 셀에서 `fresh_relocate`의 쌍 대조 `mean(out − in) d_cyclen < 0`이고
> 각 셀의 부호검정이 p < 0.05다.

- 기준점: ablation `T6_T4/f121/paramA` −1.7415 [−2.1336, −1.3342], sign 0/10, p=0.002.
- **반증 조건**: 어느 셀에서든 부호가 뒤집히고 그 셀의 부호검정이 p<0.05이면 H1은 기각되고,
  "outward fresh loading은 cycle length를 깎는다"는 **셀 조건부 규칙**으로 강등된다.
- 이 항목은 `T6T4_f121`에서 **재현 검정**이기도 하다: parent 20 / 쌍 60으로 ablation의
  parent 10 / 쌍 20을 다시 잰다. batchswap wave가 ablation의 n=4 stratum 추정치를 5.8배
  과대추정으로 판정한 전례가 있으므로, **자체 재현이 실패할 가능성을 명시적으로 등록한다.**

### H2 — `batch_swap` 일반성: 618 가지 밖에서도 성립하는가

> **H2.** `batch_swap`의 쌍 대조 `mean(out − in) d_f_r < 0`이 `T6T4_f121`과
> `HGD569_f125` 두 signed 셀 모두에서 성립한다.

- 기준점: batchswap wave 213 chain에서 **−0.1442** [−0.1888, −0.1013]
  (ablation 40 chain의 −0.1711을 갱신). 다만 그 213개 중 **158개가 ~618 EFPD parent**에
  몰려 있어, ~625 가지와 다른 셀에서의 거동은 미측정이다.
- 이번 wave의 `T6T4_f121` parent 20개는 F_xy 순위로 뽑혀 618/625 두 가지에 걸쳐 있고,
  `HGD569_f125`는 완전히 다른 pair·feed·e_core다.
- **반증 조건**: 두 셀의 부호가 서로 반대이면 batch_swap 방향 효과는 **branch/cell 특이적**이며
  policy feature로 전역 사용 불가로 기록한다.
- ga80 3개 셀은 이 가설에 **기여할 수 없다** — 방향이 구조적으로 존재하지 않는다(§2.5).
  대신 그 120개 `batch_swap|neutral` chain은 H4의 축퇴 대조군이 된다.

### H3 — F_xy와 F_r의 반응은 move class별로 갈라지는가

> **H3.** `fresh_relocate`의 쌍 대조에서 `mean(out − in) d_F_xy`와
> `mean(out − in) d_F_r`의 **부호가 같다**. 그리고 적어도 하나의 move class에서
> **부호가 다르다**.

- 근거: `produce_fxyera_r1_prereg_20260829.md` §2의 541개 retained dir에서 F_xy/F_r 비는
  F_r 1.50–1.55에서 1.0711, F_r ≥2.50에서 1.3754로 **단조 증가**한다 — 두 축은 상관은 있으나
  같은 양이 아니다. `node_peak`과 `f_xy`의 상관도 0.735–0.854에 그친다
  (`lpopt/data/flatness.py` 모듈 docstring).
- 실제 의미: F_xy가 하드 한계 1.65를 가진 새 목적함수인데, 지금까지의 모든 move 규칙은
  **F_r로 학습되었다.** 어떤 class에서 두 축이 갈라지는지가 정책 v3의 feature 부호를 정한다.
- **반증 조건**: 4개 class 전부에서 두 부호가 일치하면 "F_xy는 F_r의 단조 변환으로 충분"이
  round 1 범위에서 기각되지 않고, `predict_fxy`의 proxy 경로가 잠정적으로 정당화된다.

### H4 — 축퇴 셀의 `batch_swap`은 F_xy를 움직이는가

> **H4.** 등농축 pair(`E1_E2`, `N1_N2`)에서 `batch_swap|neutral` 120 chain의
> `d_F_xy` 분포는 `rewire_swap|neutral` 60 chain의 분포와 **다르다**
> (parent 평균에 대한 부호검정, 셀별 p<0.05).

- 두 class 모두 `d_fresh_enr_r_center ≡ 0`이지만, `rewire_swap`은 연소 이력만 재배선하고
  `batch_swap`은 **Gd 배치**를 바꾼다. F_xy는 pin planar peaking이므로 Gd 배치에 민감할
  이유가 있고, 농축 가중 반경 기술자는 그것을 **볼 수 없다.**
- 이 가설이 서면, 현재 feature set에 **Gd/lattice 기술자가 빠져 있다**는 것이 측정으로
  확립된다 — 관측 코퍼스로는 만들 수 없는 결론이다.

### 사전 등록된 블라인드 예측

`data/design/intervention_wave_r1_s1i_pred.csv` (sha256 §9)는 라벨이 생기기 **전에**
챔피언 `data/models/s1i`로 계산한 800개 자식 + 그 parent의 예측이며, 등록량은 **예측 delta**다.
`acquisition.has_fxy_head`로 확인한 결과 **이 체크포인트에는 f_xy head가 없어** F_xy 예측은
`fxy_proxy` 경로(F_r 열의 아핀 변환)에서 나왔고 CSV의 `pred_f_xy_source` 열에 `proxy`로 기록된다.
즉 이 예측은 **F_r에 대한 회귀를 F_xy 예측으로 제시하지 않는다.**

| move_class · dir | `d_pred_f_r` | `d_pred_f_xy` (proxy) | `d_pred_cyclen` | `d_pred_node_peak` |
|---|---:|---:|---:|---:|
| batch_flip · inward | +0.0130 | +0.0158 | −1.7338 | +0.0070 |
| batch_flip · neutral | +0.0105 | +0.0128 | −0.1962 | +0.0061 |
| batch_flip · outward | +0.0104 | +0.0127 | −1.6788 | +0.0099 |
| batch_swap · inward | +0.0248 | +0.0302 | −0.0832 | +0.0081 |
| batch_swap · neutral | +0.0294 | +0.0358 | −0.0564 | +0.0709 |
| batch_swap · outward | +0.0269 | +0.0327 | +0.1265 | +0.0053 |
| fresh_relocate · inward | +0.0951 | +0.1158 | −0.5317 | +0.0649 |
| fresh_relocate · outward | +0.0667 | +0.0812 | −1.6739 | +0.0337 |
| rewire_swap · neutral | +0.0047 | +0.0058 | −0.0263 | +0.0075 |

> 등록된 관전 포인트: proxy는 `fresh_relocate` outward의 `d_cyclen`을 **−1.67**로
> 내놓는데 이는 ablation의 측정 out−in −1.74와 부호가 맞다. 반대로 `batch_swap` outward의
> `d_cyclen`을 **+0.1265**(inward보다 큼)로 내놓는데, 측정된 out−in은 **−0.1893**이었다.
> 즉 s1i는 `fresh_relocate`의 방향 부호는 맞히고 `batch_swap`의 방향 부호는 **틀릴 것**으로
> 지금 예측된다. 이 예측 자체가 라벨 도착 시 검정 대상이다.

---

## 6. 성공 기준 · 예산 · 정지 규칙

| # | 기준 | 임계 |
|---|---|---|
| P1 | 수렴 chain 중 F_xy가 파싱된 비율 (셀별 `fxy_sidecar.jsonl` / converged) | **≥ 95%** — 미달이면 MAS_OUT 보존 사슬이 끊긴 것이므로 **즉시 중단** |
| P2 | 전체 수렴률 | **≥ 90%** (ablation 97.3%, batchswap 100%) |
| P3 | harness error (staging/disk/exit status) | 셀당 **≤ 5%**. 초과 시 그 셀 중단, 다른 셀은 계속 |
| P4 | 쌍 완결성: 두 멤버가 모두 수렴한 쌍 | **≥ 200 / 240** |
| P5 | `restart_provenance`가 parent와 불일치한 chain | **0** (게이트가 사전에 막음) |
| P6 | 벽시계 | **≤ 20 h** (예산 14.4 h + 여유) |

**예산.** 800 chain. HOST_199 실측 cadence(24 worker, wave 중앙값 24.32 min = 지속
**55.6 chains/h**, `produce_fxyera_r1_prereg` §5)에서 **14.4 h**. 보수적 p90(45.9 chains/h)로
17.4 h. `--max-chains` 펜스는 두지 않는다 — **계획 자체가 펜스**다(열거된 800개 record_id,
셀당 상한 200). 디스크: 보존 MAS_OUT 9.5 MB/chain × 800 ≈ **7.6 GB**, launcher는 25 GB 미만이면 거부.

**정지 규칙.** P1 위반 → 전면 중단. P3 위반 → 해당 셀만 중단(셀별 run dir이 독립이라
나머지 4개는 영향 없음). 박스가 조기 회수되면 이미 끝난 셀은 **그 자체로 완결된 sub-wave**다
— 5개 셀이 독립 resume 단위라는 것이 이 설계의 부수 효과다.

---

## 7. 분석 (사후, `intervention_wave.py analyze`)

입력은 **run 자신의 산출물**(셀별 `ablation_results.jsonl` + `fxy_sidecar.jsonl` + 매니페스트)이며
store를 읽지 않는다. 그래서 `merge-store` 전에 박스에서 바로 계산할 수 있고, 이후 store가
움직여도 발표된 효과크기가 바뀌지 않는다. F_xy는 sidecar에서, 즉 **`MAS_OUT`에서
`lpopt.data.fxy`를 통해서만** 온다.

산출 테이블 (`data/reports/intervention_wave_r1_*.csv`):

- `effects_by_cell` — (cell × move_class × direction)의 n / mean / median / improving 비율,
  4개 반응축 `d_F_xy` `d_F_r` `d_node_peak` `d_cyclen`
- `effects_pooled` — 같은 표의 pooled 판
- `effects_by_burn_state` — 위에 burn-state를 추가한 층
- `paired_by_cell`, `paired_pooled` — **1차 통계량**. 쌍 내부 (out − in), 정확 이항 부호검정
- `parent_blocked_signs` — parent 평균에 대한 부호검정 (분석 단위 = parent)
- `intervention_wave_r1_rows.csv` — 행 수준 원자료

부호검정은 `ablation_analyze._sign_test`를 그대로 import한다 (재구현 아님).

**split 원칙 (검토서 §7.7).** 이 wave가 만드는 행은 `lineage_source='intervention_<cell>'`로
`data/policy/steps.parquet`에 들어간다(`corpus` 서브커맨드, `mine_policy_corpus.build_steps`
자신이 행을 만들므로 80개 열이 채굴 코퍼스와 스키마 동일). 학습 시 등록된 split:
**parent-blocked**(같은 parent의 자식은 절대 train/test로 갈라지지 않음) **+ cell holdout**
(`(feed, e_core)` hidden cell 조항). record random split은 금지된다.
round 1의 `HGD569_f125`는 **영구 prospective holdout 후보**로 표시해 둔다 — 실현가능 행이
0인 유일한 셀이고, 거기서의 일반화가 검토서 §6.10(적용영역이 좁다)의 실질 시험이다.

---

## 8. policy-v3 코퍼스 게이트

이 wave의 행이 정책 v3 학습 코퍼스에 편입되기 위한 사전 등록 조건:

| gate | 조건 |
|---|---|
| G1 | P1–P5 충족 (§6) |
| G2 | `corpus`가 스키마 drift 없이 append되고, `(parent_record_id, child_record_id)` 중복이 0 |
| G3 | 각 셀의 `single_move` 비율 **≥ 0.98** — 열거자는 정확하지만 **분류기가 권위**이고, 순 diff가 의도한 class로 되읽히지 않는 후보는 재라벨이 아니라 **탈락**이다 |
| G4 | 최소 3개 셀에서 `fresh_relocate` 쌍 대조가 같은 부호 — H1이 셀 조건부로 강등되면 그 feature는 **cell-conditioned로만** 학습에 들어간다 |
| G5 | pooled 학습 전에 `effects_by_cell`의 셀 간 이질성이 보고되어야 한다. batchswap wave가 ablation의 stratum 추정치를 5.8배 과대추정으로 판정한 전례가 있으므로, **셀을 섞기 전에 셀 간 차이를 먼저 본다** |

G3–G5 중 하나라도 미달이면 행은 store와 `steps.parquet`에 **남되**(실패 레코드를 지우지 않는다,
검토서 §7.8-6), 정책 v3의 pooled 학습 fold에서는 제외되고 진단 슬라이스로만 쓰인다.

---

## 9. 동결 artefact (sha256)

launcher가 게이트하는 값이다. 이 표의 어떤 값이든 바꾸려면 **재등록**해야 한다.
특히 `plan`을 다시 돌리면 `written_utc`가 바뀌어 sha가 달라진다 — **등록 후 재계획 금지.**

| artefact | sha256 | bytes |
|---|---|---:|
| `intervention_wave.py` | `4D545E814A050953703769A684019F855D9C1944F16EE5D782E5B4B0AA88FDC6` | 75,031 |
| `tests/test_intervention_wave.py` | `D984FAB050B976DF2CEA21BEBC5A8F119E41B3901471BA1EBA483B3E65C3382F` | 26,893 |
| `data/design/intervention_wave_r1.json` | `F82CE02943893D5132FFEC9321ADFA1757C3CF6DC30624CF392E93C2D86FE20D` | 2,660,532 |
| `data/design/intervention_wave_r1_s1i_pred.csv` | `5135B59E75F176C39F450FD05C8E489A1F5B100AA307A146687AA69FB4D8F3A0` | 417,800 |
| `launch_intervention_wave_r1_199.ps1` | `508FA4DC2DD9FB85A0361DC153E9C87555C5A4CE705232C031CD9F83972A7E02` | 8,949 |
| `run_intervention_wave_r1_199.bat` | `A5F9CBF20F1F98AC7B55DD5F82CBC3C5B394DFCE4ED5381EE779FA984D86CE29` | 4,035 |
| `status_intervention_wave_r1_199.ps1` | `BB5DEB0F2A2C180006FE47C38DD45DA60AE4AB8130739B695559B77E2A7733AC` | 4,788 |

**의존 artefact — 수정 금지, 존재/무결성만 확인**

| artefact | sha256 | bytes |
|---|---|---:|
| `ablation_wave.py` (Amendment 2 버전) | `1B94C7128F41685B6B3852527AD8FF6625414F781009AD1ED9F17CAE5F9280C1` | 40,237 |
| `ablation_analyze.py` | `58D6F779D2F35A4CE24FFE489B96CEC52E2A8939EEC3214CA24A3B67C285E95E` | 19,216 |
| `mine_policy_corpus.py` | `78B798FA3537744F5E0B51026DA6E4530A7B1F7C12A9BAB511B23DF549A7C9AB` | 91,468 |
| `data/store/records.parquet` | `4CFF270B1020C87A2EC41BE3FE9595C481970197D01FC5AB58A174B194225057` | 22,144,665 |
| `data/store/fuel_types.parquet` | `FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137` | 64,343 |
| `data/policy/steps.parquet` (append 전) | `F6B877BBD8C71705FC41A11BB36C764E1613E25790930F91E308BEB4FB71BCF8` | 9,263,631 |

`intervention_wave.py`는 **비-ASCII 문자 0개**다 (ablation Amendment 1의 cp949
`UnicodeEncodeError` 재발 방지).

---

## 10. 배포 kit 및 실행 순서 (아직 실행하지 않음)

`C:\Users\USER\lpopt_work\kit_frontier`로 **바이너리 전송**한다. 원격에서 편집하면
PowerShell이 UTF-8 BOM을 붙여 해시 게이트가 거부한다 (2026-08-12 교훈).

```
intervention_wave.py
launch_intervention_wave_r1_199.ps1
run_intervention_wave_r1_199.bat
status_intervention_wave_r1_199.ps1
data/design/intervention_wave_r1.json
data/store/records.parquet          (해시가 이미 맞으면 생략)
data/store/fuel_types.parquet       (동일)
```

이미 kit에 있어야 하는 것: `ablation_wave.py`, `mine_policy_corpus.py`,
`data/design/package/**` (registry + bases + lib/MAS_XSL), `data/design/synth_decks/**`,
`FEASIBLE_PACKAGE/**`, `data/models/s1i/**`.

```bash
# 1) 사전 상태 확인 (읽기 전용)
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_intervention_wave_r1_199.ps1"

# 2) 발사 — MASTER를 시작하는 유일한 명령
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_intervention_wave_r1_199.ps1"

# 3) 감시 (읽기 전용, 반복 가능).  CELL/FXY/MASOUT 블록을 본다
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_intervention_wave_r1_199.ps1"
```

**전제 조건: `produce_fxyera_r1`이 끝나 있어야 한다.** launcher의 busy 게이트가
`produce_fxyera_r1` python 프로세스와 `master4.0m4_r1`를 모두 확인하고, 살아 있으면 거부한다.
같은 박스·같은 store를 공유하므로 두 MASTER 큐를 겹치면 두 wave의 cadence가 모두 무너진다.

**중단 후 재개는 launcher가 아니라 `.bat`을 직접 돌린다.** launcher는 무장 전에
`runs\intervention_wave_r1`을 지우고, `.bat`은 셀별 jsonl에서 resume한다
(`ablation_wave._done`: 물리적 답은 settled, harness 실패는 재실행 —
`tests/test_ablation_resume.py`가 지키는 계약).

수확:

```bash
# 셀별 merge-store kit 재생성 (run 끝에 자동으로도 돌지만, 재수확이 안전하다)
python intervention_wave.py kit --run-dir runs/intervention_wave_r1 --package data/design/package --kit paramA
python intervention_wave.py kit --run-dir runs/intervention_wave_r1 --package FEASIBLE_PACKAGE     --kit ga80
# 분석 (store 불필요)
python intervention_wave.py analyze --run-dir runs/intervention_wave_r1 --out-dir data/reports
# merge-store 이후
python intervention_wave.py corpus --dry-run
```

---

## 11. 테스트

`tests/test_intervention_wave.py` — 18 케이스, 합성 store(`random_genome`) 위에서 hermetic.
signed 셀과 축퇴 셀을 **둘 다** 돌린다.

- `plan` 균형/쌍: parent당 정확히 8개, `fresh_relocate` 2+2, `batch_swap` 1+1,
  `rewire_swap` 1, `batch_flip` 1; 모든 쌍이 2멤버·1 outward·1 inward·같은 parent·
  같은 class·같은 burn-state
- 축퇴 셀: `batch_swap`이 중립 2개, `batch_flip`이 중립 1개, shortfall 0, 여전히 8개
- burn-state 분류가 `GeneralOrbitGenome._depths()`와 일치, `pick_burnstate_balanced` 결정성
- 매니페스트 스키마 (최상위 키, 셀 키, 후보 열, `record_id == compute_record_id(...)`,
  paid record_id 유일)
- 실현가능 행 0인 셀에서도 parent 20개 착석 (HGD569 케이스)
- resume: 셀별 jsonl에서 harness 실패만 재실행, sub-plan이 그 셀만 담음
- run이 라이브러리 혼합을 거부
- analyze: 주입한 효과를 쌍 대조가 정확히 복원(`mean(out−in) = −3.0`, sign 0/10, p<0.01),
  parent-blocked 부호검정의 분석 단위가 parent, 미수렴/ F_xy 결측 행이 0이 아니라 **제외**됨
- F_xy sidecar 왕복(재실행이 이전 값을 대체, 깨진 줄 무시), `_fxy_row`가 `WaveOutcome.fxy`를 읽음
- corpus: append 멱등(재실행해도 행이 늘지 않음), 스키마 drift 거부, dry-run 무기록, 백업 생성

```
python -m pytest tests/test_intervention_wave.py -q   ->  18 passed
python -m pytest tests/test_ablation_resume.py -q     ->  10 passed
```

---

## 12. 알려진 한계 (사전 공개)

1. **5 context ≠ 100 context.** 검토서 §7.4의 규모는 round 1이 아니라 프로그램 목표다.
   round 1은 "N-셀 도구 + 첫 5셀"이고, 셀 추가는 `CELLS_R1`에 줄을 더하는 일이다.
2. **ga80 3개 셀은 H2에 기여하지 못한다.** 등농축 pair에서 `batch_swap`의 방향은 구조적으로
   없다. signed batch_swap 증거는 paramA 2개 셀(쌍 120개)에서만 나온다.
3. **`twice_plus` burn-state는 f121에서 구조적으로 0이다** (`depth2_edges = 0`).
   pooled 80개는 전부 f109/f113에서 온다 — burn-state 효과와 feed 효과는 round 1에서
   완전히 분리되지 않는다.
4. **dose 척도는 셀 간 비교 불가** (§3). 부호와 셀 내 순위만 보고한다.
5. **s1i에는 f_xy head가 없다.** 등록된 블라인드 F_xy 예측은 proxy이며, 그 사실이 CSV의
   `pred_f_xy_source` 열과 §5에 명시되어 있다. head가 실린 체크포인트가 나오면 같은 계획에
   대해 재채점할 수 있으나, **등록된 예측은 이 파일이고 그 sha256이 사전 약속이다.**
6. **replicate/QC arm 없음** (검토서 §7.6의 5%). `produce_fxyera_r1`과 같은 이유로
   dedup이 동일 `record_id`를 거르므로 이 경로로는 표현되지 않는다. MASTER 결정성은
   batchswap wave의 우발적 7쌍 충돌에서 **7/7 비트 동일 F_r**로 이미 확인되었다.
7. **`E1E2_f121`과 `E1E2_f109`는 같은 pair다.** 두 셀의 결과는 독립 관측이 아니며,
   pooled 부호검정에서 5개 셀을 독립으로 세지 않는다 (§7의 cell holdout 조항).

**STAMP 2026-08-30 12:0x:** launcher store pin re-stamped `4CFF270B…` → `255F0E41707CB4EF64D843FD19DB81531C12AB3A969F6F8F06C87E0AF5561A51` (75,893행; e_core backfill·pinbu patch·min_fxy r1 병합 반영). plan/code/fuel pins 불변. 블라인드 예측은 s1i 기준(계획 시점)으로 유지.

**DEVIATION 2026-08-30 17:0x:** `run_intervention_wave_r1_199.bat`의 ga80 호출에 `--allow-fallback`이 빠져 E1E2_f109(등록 restart `pair_feed:MAS_RST.APRQ_11_0615.88`, 부모 277행 전부 동일 restart)가 도구 가드에 의해 거부·프로세스 종료(rc 미기록). T6T4_f121·HGD569_f125·E1E2_f121(480체인) 완료. `resume_intervention_wave_r1_ga80_199.bat`로 E1E2_f109·N1N2_f113을 `--allow-fallback` + `check_restart`(run==parent restart) 게이트 하에 재개. 분석은 5셀 합산으로 진행.
