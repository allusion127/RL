# 상용 PWR 재장전 노심의 연료집합체 배치 규칙  
## Neutronics 중심의 엔지니어링 규칙, 적용 조건, 상충관계 및 검증 절차

- **작성일:** 2026-07-29
- **대상:** 상용 가압경수로(PWR) 재장전 노심 설계, 연료집합체 셔플링 및 loading pattern 최적화
- **중점:** 중성자물리·출력분포·반응도·제어봉·노심 수명·누설·용기 조사량·핀 출력/연소도
- **보조 범위:** 열수력, 연료성능, CRUD/CIPS, 집합체 변형, 연료취급 및 품질보증
- **문서 성격:** 공개된 규제문서, 국제기구 보고서, 원전·연료공급사 기술자료, 국립연구소 보고서 및 학술논문을 종합한 엔지니어링 보고서

> **중요한 전제**  
> 실제 상용 PWR의 최종 loading pattern 설계 규칙, 제한값, 금지 패턴, 보정계수 및 허용 여유는 연료공급사·원전·노형·주기별로 상당 부분 독점 정보이다. 따라서 본 문서는 공개자료에서 확인되는 공통 원리와 대표 사례를 정리한 것이며, 특정 플랜트의 기술사양서, Core Operating Limits Report(COLR), 안전해석 방법론, 연료설계 기준 및 승인된 설계절차를 대체하지 않는다.

---

## 초록

상용 PWR의 재장전 노심설계는 “반응도가 높은 집합체를 어디에 둘 것인가”로 환원되는 문제가 아니다. 설계자는 한정된 신연료와 조사연료 재고를 사용하여 목표 주기길이와 경제성을 달성하는 동시에, 주기 전 구간과 여러 운전상태에서 출력첨두, DNBR 관련 지표, 선출력률, 핀·집합체 연소도, 임계붕소농도, 감속재온도계수, 정지여유도, 제어봉가, 출력경사, 축방향 출력형상, 원자로용기 조사량 및 연료성능 제한을 만족해야 한다. 이 문제는 조합수가 매우 크고, 이웃 집합체와 운전상태에 따라 응답이 비선형적으로 바뀌며, 현 주기의 선택이 다음 2–3개 주기의 재고와 가능영역을 바꾸는 다목적·다주기 최적화 문제이다.[R01–R04, R08, R12, R15, R18]

본 보고서의 핵심 결론은 다음과 같다.

1. **OUT–IN, IN–OUT 및 저누설 loading은 서로 다른 목적을 가진 조건부 전략**이다. OUT–IN은 신연료의 높은 반응도를 외곽 누설로 억제하고 이후 안쪽으로 이동시키는 역사적 전략인 반면, 현대의 저누설 전략은 대체로 고반응도 신연료를 내부에 두고 고연소도·저반응도 연료를 외곽 완충층에 배치하여 중성자 경제성과 용기 조사량을 개선한다.[R06, R07, R08, R10]
2. **“BOC `k∞`가 높은 집합체는 외곽에 둔다”는 보편 법칙이 아니다.** 이는 OUT–IN 또는 특정 출력평탄화 상황에서는 성립할 수 있지만, 저누설 IN–OUT 계열에서는 대체로 반대이다. 또한 `k∞`는 무한격자 지표이므로 위치 중요도, 누설, 이웃 효과, 제논·붕산·온도 및 잔류 가연성흡수체를 반영하지 못한다.
3. **1/4 대칭은 설계·탐색·검증을 단순화하는 강력한 실무 규칙이지만 항상 강제되는 물리 법칙은 아니다.** 반사경계가 가능한 거울대칭과 단순 90° 회전대칭을 구분해야 하며, 실제 조사이력·LTA·계측기·집합체 상태가 비대칭이면 최종 전노심 검증이 필요하다.
4. **집합체 회전은 ‘물리적 90°/180°/270° 회전’, ‘반대편 위치로의 cross-core 이동’, ‘대칭 위치로의 수학적 매핑’을 구분해야 한다.** 핀 또는 사분면별 동위원소 이력과 방향성이 모델에 없으면 회전 효과는 계산상 사라진다.[R15–R17]
5. **신연료·고반응도 연료의 인접 금지, Gd 집합체 인접 제한, checkerboard 배치, Ring-of-Fire 억제 등은 실제 사용되는 규칙이지만 보편적인 절대법칙은 아니다.** 플랜트별로 금지, 완화 가능한 제약, 또는 단순 탐색 휴리스틱으로 구현 방식이 다르다.[R08, R11, R12]
6. **최종 배치는 assembly-average 출력만으로 승인할 수 없다.** 핀 출력복원, 3차원 축방향 형상, 열수력, 제어봉 사고, 연료성능, CIPS/CRUD, 용기 조사량 및 불확실도까지 연결해야 한다.[R02–R05, R09, R22]
7. **현 주기 최적점과 다주기 최적점은 다를 수 있다.** 지나친 반응도 추출, 특정 연료군의 조기 방출, 과도한 외곽 저연소, 또는 신연료군의 비효율적 소모는 다음 주기 가능영역을 악화시킬 수 있다.[R03, R12, R13]
8. **엔지니어링 규칙은 해석을 대체하지 않는다.** 규칙은 후보생성, 조기 배제, repair operator, 탐색공간 축소에 유용하지만 최종 수락은 승인된 3-D 노심해석과 안전·연료성능 평가로 이루어져야 한다.

---

# 1. 적용범위와 증거 수준

## 1.1 본 문서에서 말하는 “배치 규칙”

여기서 배치 규칙은 다음 네 종류를 모두 포함한다.

| 구분 | 의미 | 최적화 코드에서의 권장 구현 |
|---|---|---|
| **H: 경성 제약(Hard constraint)** | 위반 시 안전해석·기술사양·연료설계 기준을 만족하지 못하는 제약 | 후보 즉시 탈락 또는 보수적 screening |
| **P: 플랜트/공급사 규칙(Plant-specific rule)** | 승인된 설계절차, 경험, 기계적 호환성 또는 운전정책에 따른 규칙 | 명시적 금지 또는 repair; 예외는 공식 설계검토 필요 |
| **N: 물리 휴리스틱(Neutronic heuristic)** | 좋은 후보를 빠르게 만드는 경험칙이나 우선순위 | seed 생성, mutation bias, soft penalty |
| **O: 목적함수/선호(Objective)** | 만족 여부가 아니라 경제성·여유·운전유연성을 개선하는 지표 | 다목적 Pareto 최적화 |
| **Q: 검증/품질보증 규칙(QA)** | 계산·연료이력·현장 loading 오류를 방지하는 절차 | 독립 확인, 데이터 추적성, 전노심 재계산 |

실제 설계에서 “신연료 두 개를 붙이지 않는다”와 같은 문장은 H일 수도 있고 P 또는 N일 수도 있다. 해당 플랜트 안전해석 방법론에 명시되어 있지 않은 경험칙을 임의로 H로 승격하면 탐색공간을 불필요하게 잘라낼 수 있고, 반대로 진짜 경성 제약을 penalty로만 처리하면 최적화 과정에서 허용 불가능한 후보가 남을 수 있다. KNFC의 McFLOP 사례는 Ring-of-Fire와 같은 forbidden pattern을 완전 금지에서 가중 penalty로 바꾸면 국소최적점 문제를 완화할 수 있음을 보여준다. 이는 **규칙의 물리적 성격과 수치적 구현을 구분해야 한다**는 좋은 사례이다.[R11, R18]

## 1.2 근거 수준

- **A급:** IAEA 안전기준, NRC Standard Review Plan, OECD/NEA 기술검토 등 규제·국제기관 자료
- **B급:** 연료공급사, 원전 UFSAR/COLR 계열 자료, DOE/INL·ORNL 등 국립연구소 보고서
- **C급:** 동료평가 학술논문
- **D급:** 학회논문, 강의자료, 공개 최적화 demonstration
- **S급:** 위 자료를 결합한 본 보고서의 물리적·실무적 종합 판단

A급 자료는 주로 **무엇을 검증해야 하는지**를 규정하고, B–D급 자료는 **어떤 배치 휴리스틱과 최적화 방법을 사용하는지**를 더 구체적으로 보여준다.

---

# 2. 재장전 노심설계 문제의 구조

## 2.1 설계변수

상용 PWR의 loading pattern은 최소한 다음 변수를 포함한다.

1. 각 노심 위치에 배치할 **고유 연료집합체 ID**
2. 신연료의 농축도, Gd/IFBA/WABA 등 **가연성흡수체 사양**
3. 조사연료의 집합체 평균 연소도뿐 아니라 **축·핀·사분면별 조사이력**
4. 집합체의 **물리적 방향(orientation)**
5. 제어봉, 계측기, LTA, 누설감시, 수로·baffle 인접 등 **위치 속성**
6. 주기별 출력, 붕산, 온도, 제논, 제어봉 상태를 포함한 **평가 상태점**
7. 현 주기 후 예상되는 방출·잔류 재고와 다음 주기 투입 전략

따라서 같은 “3회차 연료”라도 실제로는 반응도와 출력이 크게 다를 수 있다. 집합체 평균 연소도만 같은 두 연료가 이전 위치, 출력이력, Gd 잔류량, 축방향 연소도, 핀별 연소도 및 냉각재 밀도 이력 때문에 서로 다른 거동을 보일 수 있다.

## 2.2 대표 목적함수

설계 목적은 플랜트와 연료공급 계약에 따라 달라지지만 공개자료에서 반복적으로 나타나는 항목은 다음과 같다.[R03, R08, R11, R12, R15, R18]

- 목표 주기길이 또는 EOC 반응도 확보
- 신연료 장전량·농축도·연료주기비용 최소화
- 평균 방출연소도 향상
- 최대 radial/3-D/pin peaking 최소화
- 열적여유 및 사고해석 여유 최대화
- 용기 fast-neutron fluence 및 노심 누설 최소화
- 최대 붕산농도, MTC, SDM, 제어봉가 등의 운전여유 확보
- CIPS/CRUD·연료부식·고출력 연료봉 duty 저감
- 다음 주기 재고의 유연성 확보

개념적 다목적 함수는 다음처럼 쓸 수 있다.

\[
\min_{\mathbf{x}}
\left[
C_{\mathrm{fuel}},
- L_{\mathrm{cycle}},
F_{q,\max},
F_{\Delta H,\max},
\Phi_{\mathrm{vessel}},
R_{\mathrm{CIPS}},
R_{\mathrm{fuel}}
\right]
\]

subject to

\[
g_j(\mathbf{x},s)\le 0
\quad
\text{for all required state points }s
\]

여기서 \(\mathbf{x}\)는 위치·집합체 ID·방향·BA 선택을 포함하고, \(s\)는 BOC/MOC/EOC, HFP/HZP, 제논 상태, 제어봉 상태 등의 조합이다. **한 상태점의 우수한 결과가 다른 상태점의 수락을 보장하지 않는다.**

## 2.3 핵심 수락지표

플랜트별 정의가 다를 수 있으므로 아래 기호는 일반적 의미로만 사용한다.

| 지표 | 일반적 역할 | 배치와의 관련성 |
|---|---|---|
| \(k_{\mathrm{eff}}\), EOC reactivity | 주기길이·임계성 | 고반응도 연료의 위치 중요도와 누설에 민감 |
| Critical Boron Concentration, CBC | 과잉반응도·붕산 운전범위 | 신연료·BA 개수와 위치에 민감 |
| MTC | 온도피드백 안전성 | 농축도, 붕산, BA, 스펙트럼 및 위치에 민감 |
| \(F_q\) | 3-D 국부 출력 또는 열유속 첨두 | 축·반경 결합, 제논, BA 소진에 민감 |
| \(F_{\Delta H}\) | 채널 엔탈피상승/핫채널 관련 첨두 | radial pin/assembly power와 열수력에 민감 |
| \(F_{xy}\), radial pin peak | 반경방향 핀 출력 첨두 | 신연료 인접, 반응도 불연속, reflector 경계에 민감 |
| LHGR/linear heat rate | 연료봉 선출력 | 고연소도 연료 duty와 출력이력 제한 |
| DNBR/MDNBR | DNB 열적여유 | 핀 출력, 유량, 혼합, 축형상과 연계 |
| SDM | 가장 반응도 높은 제어봉 고착 시 정지여유도 | 제어봉 주변의 연료반응도와 위치에 민감 |
| Rod worth/ejected-rod worth | 제어능·사고해석 | 고반응도 집합체와 제어봉 위치의 조합에 민감 |
| AO/ASI/radial tilt | 축·반경 출력균형 | 제논, BA, 제어봉 및 비대칭 loading에 민감 |
| Assembly/pin burnup | 연료 허용한계·방출전략 | shuffling·회전·국부 출력이력의 직접 결과 |
| Vessel fast fluence | 원자로용기 취성화 관리 | 외곽 출력과 누설 스펙트럼에 민감 |
| CIPS/CRUD indicators | 축방향 출력이동·부식 | 고출력/비등 duty와 3-D 분포에 민감 |

---

# 3. 배치규칙의 중성자물리적 기반

## 3.1 `k∞`와 실제 노심 내 반응도·출력은 다르다

집합체 무한격자 증배계수 \(k_\infty\)는 누설이 없는 반복 격자에서의 특성이다. 반면 실제 위치 \(i\)의 출력은 개략적으로

\[
P_i \propto \int_{V_i}\Sigma_{f,i}(\mathbf{r},E)\,
\phi(\mathbf{r},E)\,dE\,dV
\]

로 결정되며, \(\phi\)는 주변 연료·반사체·제어봉·온도·붕산·제논에 의해 바뀐다. 위치 교환의 전노심 반응도 효과는 단순한 \(k_\infty\) 차이가 아니라 adjoint 중요도를 포함하는 섭동량에 가깝다.

\[
\Delta \rho_i \sim
-\frac{\langle \phi^\dagger,\Delta A_i\phi\rangle}
{\langle \phi^\dagger,F\phi\rangle}
\]

따라서 중앙의 작은 반응도 차이가 외곽의 큰 \(k_\infty\) 차이보다 전노심 \(k_{\mathrm{eff}}\)에 더 크게 작용할 수 있다. 반대로 외곽에서는 높은 반응도가 누설로 소모되어 전노심 경제성이 낮아질 수 있다.

### 실무 권고

- 집합체를 한 개의 “BOC \(k_\infty\)”로 영구 서열화하지 않는다.
- 적어도 HFP 기준 붕산·온도·제논 조건에서 **상태점별 reactivity index**를 만든다.
- Gd 잔류, burnup, moderator density, Xe/Sm, 농축도 및 스펙트럼 이력을 포함한다.
- 최종 순위는 위치별 \(\Delta k_{\mathrm{eff}}\), power response, pin peak 및 safety parameter로 재평가한다.
- 무한격자 지표는 seed 생성용이고 최종 수락지표가 아니다.

## 3.2 위치 중요도와 누설

대체로 노심 내부는 중성자 중요도가 높고 누설이 작으며, 외곽·코너는 누설이 크다. 같은 연료를 중앙으로 옮기면 전노심 반응도와 출력이 증가하는 경향이 있고 외곽으로 옮기면 감소하는 경향이 있다. 그러나 reflector와 water gap, baffle 구조 때문에 외곽 집합체 내부의 핀 출력은 단순한 단조감소가 아니며, 경계면 핀에 국부적 왜곡이 발생할 수 있다.

## 3.3 이웃 효과와 반응도 불연속

고반응도 신연료를 서로 붙이면 assembly-average 출력뿐 아니라 맞닿은 면의 핀 출력이 증가할 수 있다. 반대로 강한 Gd 집합체를 뭉치면 저출력 영역이 생기고 Gd 소진 후 power rebound가 발생할 수 있다. 이 때문에 실제 설계에서는 다음과 같은 규칙이 쓰인다.

- 신연료 또는 무BA 신연료의 face-adjacency 제한
- 경우에 따라 diagonal adjacency 제한
- Gd 집합체의 수평·수직 인접 제한
- 고·저반응도 연료의 분산·checkerboard 배치
- 특정 fresh-fuel annulus, 즉 Ring-of-Fire 형태의 금지 또는 억제

INL의 PWR demonstration은 “특정 Gd 연료가 수평·수직으로 인접하지 않도록 한다”는 예를 제시하고, Yamamoto의 상용노심 강의는 inboard 영역에서 무BA 신연료의 side-by-side 또는 diagonal adjacency를 제한하는 예를 제시한다. KNFC의 APR1400 최적화 사례에서는 inner core의 인접 신연료와 Ring-of-Fire를 억제하여 checkerboard에 가까운 패턴을 얻었다.[R08, R11, R12]

## 3.4 연소이력은 평균 연소도 하나로 표현되지 않는다

다음 두 집합체가 평균 30 GWd/tU로 같더라도 등가라고 가정할 수 없다.

- 한 집합체는 중앙 고출력 위치에서 1주기 조사
- 다른 집합체는 외곽 저출력 위치에서 2주기 조사
- Gd 소진 이력과 잔류 흡수량이 다름
- 축방향 연소도 분포와 제논 이력이 다름
- 반경방향 pin/face burnup gradient가 다름
- 부식·growth·bow·grid-to-rod fretting 이력이 다름

따라서 commercial reload database는 최소한 assembly ID, burnup, exposure history, axial nodes, pin/quadrant history, BA history, orientation 및 mechanical disposition을 추적해야 한다.

## 3.5 현재 주기와 다주기 최적점의 차이

현 주기 EOC \(k_{\mathrm{eff}}\)를 최대화하면 고반응도 연료를 중요도가 높은 위치에 과도하게 집중할 수 있다. 이 경우 현 주기는 길어지더라도 다음 주기에 필요한 중간연소도 재고가 부족하거나, 특정 고연소도 연료가 duty 제한 때문에 배치 불가능해질 수 있다. Yamamoto 등의 연속 2주기 최적화 연구와 상용노심 설계 강의는 단일주기 최적화의 결과가 다주기 관점에서 불리할 수 있음을 강조한다.[R12, R13]

---

# 4. 대표 shuffling 및 radial zoning 전략

## 4.1 전략 비교

| 전략 | 전형적 이동 개념 | 장점 | 주요 약점 | 현대적 적용 |
|---|---|---|---|---|
| **OUT–IN** | 신연료를 외곽 또는 비교적 바깥쪽에 넣고, 다음 주기에 안쪽으로 이동 | 신연료 출력을 누설로 억제; 단순한 출력평탄화 | 중성자 경제성 저하, 용기 fluence 증가, 외곽 신연료 사용 비효율 | 특정 노형·과도기 주기·출력평탄화 목적에 조건부 사용 |
| **IN–OUT** | 신연료를 내부에 넣고 조사연료를 바깥으로 이동 | 중성자 경제성, 잔류반응도 활용, 저누설 구성 가능 | 내부 fresh/burned interface peaking, CBC·MTC·SDM 관리 필요 | 현대 저누설 loading의 기본 철학과 부합 |
| **IN–OUT–OUT / low-leakage** | 신연료는 내부, 고연소도 연료는 한 개 이상의 외곽층에 장기간 배치 | 외곽 fast flux 및 용기 조사량 저감 | 외곽 연료의 낮은 출력·stranded reactivity, 내부 peaking 증가 가능 | 다수 상용 PWR에서 대표적 접근 |
| **Scatter / checkerboard** | 신연료 또는 고반응도 연료를 조사연료/BA 연료 사이에 분산 | radial/pin peaking 억제, 금지 패턴 만족 | 반응도 불연속 면의 pin peak, 지나친 분산 시 경제성 손실 | 현대 loading의 일반적 구성 요소 |
| **Ring-of-Fire 계열** | 신연료 또는 고반응도 연료가 특정 반경대에 연속적 띠를 형성 | 특정 반경 출력형상·rod worth·주기길이에 유리할 수 있음 | 연속 고반응도 띠의 peaking, 탐색 편향, 플랜트별 금지 가능 | 어떤 설계에서는 억제, 다른 설계에서는 의도적 사용 |
| **Hybrid** | 내부 scatter + 조사연료 외곽 buffer + 일부 위치 특화 | 경제성·누설·peaking·제어봉 제약의 동시 절충 | 규칙이 복잡하고 3-D 검증 필요 | 실제 상용노심에 가장 가까운 형태 |

## 4.2 OUT–IN을 적용할 수 있는 상황

OUT–IN은 “잘못된 옛 방식”이 아니라 목적함수가 다른 전략이다. 다음 상황에서는 여전히 일부 논리가 유효할 수 있다.

- 신연료의 높은 반응도를 큰 누설 위치에서 억제해야 할 때
- 내부 fresh-fuel clustering이 강한 radial peak를 만들 때
- 과도기 주기에서 조사연료 재고 구성이 저누설 패턴에 적합하지 않을 때
- 특정 제어봉가, detector response 또는 출력평탄화 목표가 있을 때
- 외곽 fresh 위치가 제한적으로 허용되고 용기 fluence 여유가 충분할 때

그러나 용기 fluence, neutron economy, reflector-interface pin peak 및 향후 주기 연료활용을 함께 평가해야 한다.

## 4.3 저누설 loading의 의미

저누설 loading은 단순히 “가장 탄 연료를 무조건 외곽에 놓는 것”이 아니다. 실제 설계는 다음을 동시에 결정한다.

- 외곽 몇 개 층을 저반응도 연료로 구성할지
- edge와 corner에 같은 연료군을 쓸지
- 외곽 연료가 너무 저출력화되어 방출목표를 못 채우지 않는지
- 내부 신연료와 외곽 고연소도 연료의 interface pin peak가 허용되는지
- 고연소도 연료의 추가 고출력 duty가 연료성능상 허용되는지
- 용기 방위각별 fast flux가 목표를 만족하는지

OECD/NEA 검토는 전통적 저누설 패턴이 고연소도 연료를 외곽에 배치한다고 설명하는 한편, 고연소도 운전에서는 최대연소도 연료가 항상 외곽에만 존재하지 않고 잔류반응도 활용을 위해 신연료 인접 내부 위치가 필요할 수 있음을 지적한다.[R06, R07]

---

# 5. 상세 엔지니어링 배치 규칙

아래 규칙은 “항상 지켜야 하는 보편 법칙”이 아니라 **분류와 예외까지 포함한 rule card**이다. 실제 적용 전에는 각 규칙을 해당 플랜트 설계절차의 H/P/N/O/Q 중 어느 등급으로 둘지 결정해야 한다.

---

## 5.1 대칭, 재고 및 탐색공간 규칙

| ID | 분류 | 규칙 | 물리·실무 근거 | 예외 및 필수 검증 |
|---|---|---|---|---|
| **S-01** | P/N | 가능하면 1/4 거울대칭 loading을 seed로 사용한다. | 재고관리, 탐색공간 축소, radial tilt 억제, QA 단순화 | LTA, 고유 조사이력, 계측기, 비대칭 baffle/loop, 결함연료가 있으면 전노심 |
| **S-02** | H/Q | **거울대칭과 90° 회전대칭을 구분한다.** | quarter-core 반사경계는 x/y 축에 대한 mirror symmetry를 요구 | chiral한 4-fold 패턴은 반사 quarter model로 표현 불가 |
| **S-03** | H/Q | 대칭 위치의 재고 multiplicity를 지킨다. | 일반 위치 4개, 축 위치 2개, 중앙 1개가 한 묶음 | 고유 집합체 ID·방향이 다르면 type 대칭만으로 부족 |
| **S-04** | P/N | 1/8 대칭은 대각선 거울대칭까지 성립할 때만 사용한다. | 계산 및 최적화 변수 감소 | 단순 1/4 대칭 패턴을 임의로 1/8로 축소하지 않음 |
| **S-05** | H/Q | 대칭 제약은 **재고를 정확히 보존**해야 한다. | 신연료군·BA군·조사연료군 수량은 고정 | GA crossover/mutation 후 repair 필요 |
| **S-06** | Q | 대칭 map과 별도로 assembly serial ID map을 유지한다. | 같은 연료 type도 실제 burnup/orientation/mechanical history가 다름 | 최종 loading instruction은 ID 기준 |
| **S-07** | Q | 대칭 search를 사용해도 최종 후보는 전노심 3-D로 재계산한다. | 실제 연소이력·계측·제어봉 상태가 완전대칭이 아닐 수 있음 | 특히 높은 radial tilt 여유가 작을 때 필수 |
| **S-08** | P | 대칭성을 위해 부적합한 orientation이나 mechanical restriction을 강제하지 않는다. | 방향키, nozzle, bow, instrumentation 제한 우선 | 대칭보다 연료취급·기계적 적합성이 상위 |

**해석:** 1/4 대칭은 상용 설계에서 매우 흔하지만, 그 이유는 “노심은 반드시 1/4 대칭이어야 안전하다”가 아니라 설계의 견고성, 연료재고 구성, 출력경사 억제 및 계산효율 때문이다. INL의 최적화 demonstration도 1/4 또는 1/8 representation을 사용하지만, 이는 특정 모델과 탐색공간의 선택이지 모든 상용노심에 대한 규정이 아니다.[R08, R09, R12]

---

## 5.2 반응도 zoning, 신연료 및 인접 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **R-01** | N/Q | `k∞` 단독 대신 상태점별 reactivity index를 사용한다. | 무한격자는 누설·이웃·adjoint 중요도를 무시 | 최종은 전노심 \(\Delta k\), power response로 검증 |
| **R-02** | N | 고반응도 연료를 중앙에 둘 때 저반응도/BA 이웃으로 완충한다. | 중앙 중요도와 출력이 높음 | 너무 강한 완충은 저출력 hole과 EOC rebound 유발 |
| **R-03** | P/N | inboard에서 무BA 신연료의 face-adjacency를 제한한다. | 높은 local power와 pin interface peak 억제 | BA가 있거나 충분한 해석여유가 있으면 허용 가능 |
| **R-04** | P/N | 필요하면 diagonal fresh adjacency도 제한한다. | 대각선 결합도 2-D flux와 pin peak에 영향 | face 인접보다 영향이 작아 플랜트별 선택 |
| **R-05** | P/N | 고반응도 연료가 연속적인 cluster/annulus를 형성하지 않도록 한다. | radial peak와 국부 반응도 집중 억제 | 의도적 Ring-of-Fire 전략은 별도 safety basis 필요 |
| **R-06** | N | 고·저반응도 연료를 교차 배치해 출력분포를 평탄화한다. | neighbor power sharing | 반응도 mismatch가 지나치면 interface pin peak 증가 |
| **R-07** | N/H | 농축도·burnup·BA 차이가 큰 경계는 pin-level로 평가한다. | assembly-average가 국부 핀 첨두를 숨길 수 있음 | pin reconstruction 또는 pin-by-pin 해석 |
| **R-08** | N | edge와 corner 위치는 별도 반응도 등급으로 취급한다. | corner는 두 방향 누설, flat edge는 한 방향 누설 | reflector/baffle 세부구조에 따라 순위 변동 |
| **R-09** | N | 같은 연소도군을 단순히 한 덩어리로 배치하지 않는다. | 출력 island와 spectral island 방지 | 특정 제어봉가·주기길이 목적의 zoning은 가능 |
| **R-10** | H/Q | 고반응도 위치교환 후 MTC, CBC, SDM, rod worth를 함께 재평가한다. | `k`와 power만 좋아져도 제어·피드백이 악화될 수 있음 | BOC HZP/HFP 모두 고려 |

**공개된 실제 규칙의 예:** Yamamoto는 inboard에서 무BA 신연료를 side-by-side로 두지 않는 제한과 diagonal adjacency 금지 가능성을 제시한다. INL 보고서는 Gd 연료의 수평·수직 인접 금지 예를 사용한다. KNFC APR1400 사례는 inner-core 인접 fresh와 Ring-of-Fire를 억제해 checkerboard형 패턴을 유도한다.[R08, R11, R12]

---

## 5.3 외곽, 저누설 및 원자로용기 fluence 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **L-01** | N/O | 저누설 목표 시 조사연료를 최외곽 buffer로 우선 검토한다. | 외곽 fission source와 fast leakage 저감 | 지나친 저반응도는 연료활용 저하 |
| **L-02** | N | 가장 고연소도 연료를 무조건 전부 외곽에 보내지 않는다. | 잔류반응도 회수와 discharge target 필요 | 내부 배치 시 고연소도 duty·LHGR를 엄격 확인 |
| **L-03** | H/O | 외곽 배치는 vessel fast fluence의 방위각별 peak로 평가한다. | 평균 leakage만으로 취약 weld/capsule 위치를 대표하지 못함 | vessel surveillance 위치와 연계 |
| **L-04** | N | edge와 corner에 동일한 연료를 자동 배치하지 않는다. | corner 누설이 더 크고 power가 더 낮아질 수 있음 | reflector 구조가 비균일하면 더 세분화 |
| **L-05** | N | 외곽 신연료 사용 시 누설·용기 fluence penalty를 명시한다. | 신연료의 잉여중성자가 노심 밖으로 소실 | OUT–IN의 출력평탄화 이익과 비교 |
| **L-06** | H | reflector-facing face의 pin power를 별도 확인한다. | baffle/water gap/reflector 효과로 국부 왜곡 | assembly-average 외곽 power가 낮아도 면제되지 않음 |
| **L-07** | N | 외곽 BA는 내부와 동일한 worth로 가정하지 않는다. | 위치 중요도와 스펙트럼이 다름 | 과도한 외곽 poison은 neutron economy 손실 |
| **L-08** | O | 저누설 층 수와 연료 burnup을 Pareto 변수로 취급한다. | L3P/L4P 등 여러 강도의 저누설 전략 가능 | 플랜트별 용기 fluence·peaking 여유가 결정 |
| **L-09** | H/Q | baffle heating, ex-core detector response 및 shielding response를 함께 점검한다. | 외곽 source 변화가 노심 외 계측·구조물 응답에 영향 | 해당 플랜트 방법론 적용 |
| **L-10** | N | 외곽 연료가 지나치게 저출력화되지 않도록 최소 duty를 관리한다. | stranded reactivity와 미달 방출연소도 방지 | 최소 출력 자체가 안전제약은 아니며 경제성 판단 |

MHI의 advanced PWR core 사례는 조사연료를 외곽에 두는 저누설 개념을 명시하고, OECD/NEA 자료는 저누설 loading이 연료이용과 용기 fluence에 주는 이점과 고연소도 운전에서의 새로운 peaking 문제를 함께 논의한다.[R06, R07, R10]

---

## 5.4 Shuffling, 연소도 및 다주기 재고 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **B-01** | N | OUT–IN/IN–OUT을 이름만으로 선택하지 말고 목적함수를 명시한다. | 두 전략은 leakage·fluence·power tradeoff가 다름 | 과도기 주기는 hybrid 가능 |
| **B-02** | H | 집합체 평균과 최대 pin burnup을 모두 제한한다. | pin hot spot은 평균 burnup으로 숨겨질 수 있음 | 허용값은 연료설계·라이선스별 |
| **B-03** | H/P | 고연소도 연료에 반복적인 high-duty 위치를 부여하지 않는다. | corrosion, PCI, rod internal pressure, fragmentation 관련 여유 | 승인된 연료성능 분석이 허용하면 가능 |
| **B-04** | N | 이전에 외곽이었던 연료를 무조건 중앙으로 보내지 않는다. | face burnup gradient와 residual BA가 다름 | pin/quadrant history 기반 판단 |
| **B-05** | N | 이전 중앙 고출력 연료를 외곽에 보내 저누설 buffer로 활용한다. | 높은 burnup·낮은 reactivity와 외곽 저중요도 조합 | 너무 저반응도이면 underpower/stranding |
| **B-06** | N | residual reactivity가 큰 조사연료는 내부 fresh 주변의 완충·power sharing에 활용한다. | 신연료 출력 억제와 잔류반응도 회수 | 고연소도 duty 및 interface peak 확인 |
| **B-07** | O | 방출 후보와 잔류 후보를 현 주기만이 아니라 다음 2–3주기 관점에서 정한다. | single-cycle optimum이 inventory를 악화시킬 수 있음 | 연료조달·정비계획과 연계 |
| **B-08** | H/Q | burnup uncertainty와 operating history uncertainty를 고려한다. | 예측 burnup 오차가 다음 loading의 반응도·한계를 변경 | robust margin 또는 bounding cases |
| **B-09** | N | 같은 batch 번호보다 실제 reactivity/history를 우선한다. | 같은 batch 안에서도 위치·power history가 다름 | coarse search에서 batch grouping은 가능 |
| **B-10** | O | 일부 연료의 과도한 미연소 잔류와 일부 연료의 과도한 고연소를 동시에 피한다. | 연료이용 균형과 향후 배치가능성 | 경제성·방출전략별 가중치 |
| **B-11** | Q | 집합체별 연소도·방향·위치 이력을 serial ID로 추적한다. | 잘못된 history assignment는 계산과 현장 loading 모두 오류 | 독립 데이터베이스 reconciliation |
| **B-12** | H | 연료형식별 residence limit와 검사결과를 반영한다. | bow, wear, damage, oxide, growth 등 비중성자 제약 | 부적합 연료는 후보집합에서 제외 |
| **B-13** | N/O | EOC \(k\) 최대화와 평균 discharge burnup 최대화를 동일시하지 않는다. | 서로 다른 residual inventory와 peaking을 생성 | Pareto 또는 다주기 목적함수 |
| **B-14** | Q | outage 직전 실제 운전이력으로 reload calculation을 갱신한다. | 예상 cycle exposure와 실제 exposure 차이 | IAEA core management 절차와 부합 |

OECD/NEA는 현대 고연소도 loading에서 최대 burnup 연료가 단순히 외곽에만 남지 않을 수 있음을 설명한다. Yamamoto의 강의 및 연속주기 연구는 현 주기 최적화가 다음 주기에서 불리할 수 있으므로 successive-cycle feasibility를 미리 검토할 것을 권고한다.[R07, R12, R13]

---

## 5.5 농축도 및 가연성흡수체 배치 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **A-01** | H/O | BA 총량은 BOC excess reactivity뿐 아니라 CBC·MTC·SDM를 만족하도록 정한다. | poison은 power와 feedback, control margin을 함께 변경 | EOC residual penalty 고려 |
| **A-02** | N | 강한 BA 연료는 높은 중요도·고출력 예상 위치에 우선 배치한다. | 반응도와 출력첨두 억제 | 중앙 과독성으로 저출력 hole 가능 |
| **A-03** | P/N | 강한 Gd 집합체의 face-adjacency를 제한한다. | 저출력 cluster와 Gd 소진 후 rebound 방지 | INL 사례의 구체 규칙이며 보편 절대법칙 아님 |
| **A-04** | P/N | 무BA 신연료끼리의 inboard 인접을 제한한다. | BOC pin/radial peak 억제 | enrichment가 낮거나 주변이 충분히 burnt면 허용 가능 |
| **A-05** | H | BA 효과를 BOC 하나가 아니라 소진 전후 전주기로 확인한다. | 최대 peaking 시점이 Gd burnout 이후일 수 있음 | depletion step을 BA 변화구간에 조밀화 |
| **A-06** | N/H | Gd pin map이 비대칭이면 assembly orientation을 설계변수로 포함한다. | 방향에 따라 이웃면 pin power가 달라짐 | 대칭 pin map이면 orientation 영향 감소 |
| **A-07** | N | WABA, IFBA, integral Gd를 동일한 scalar poison worth로 취급하지 않는다. | 공간분포·잔류흡수·스펙트럼·depletion이 다름 | 승인된 lattice library 사용 |
| **A-08** | H/N | 농축도 경계와 BA 경계가 겹치는 위치를 pin-level로 검사한다. | 큰 반응도 불연속이 국부 첨두를 유발 | checkerboard라도 자동 안전 아님 |
| **A-09** | H | 최대 soluble boron과 음의 MTC 요구를 동시에 확인한다. | BA 부족은 CBC 증가, 과도한 BA/스펙트럼 변화는 다른 여유 감소 | HZP와 HFP branch 모두 |
| **A-10** | N | 외곽의 strong BA 사용은 최소화 후보로 검토한다. | 낮은 위치 중요도에서 poison 가치가 낭비될 수 있음 | 외곽 fresh 또는 pin peak 억제 필요 시 사용 |
| **A-11** | H | partial-length Gd/axial BA가 있으면 3-D 축형상과 결합한다. | radial 배치가 axial peak와 AO를 변화 | 2-D radial 최적화로 승인 불가 |
| **A-12** | O | BA 개수·종류·위치와 신연료 농축도를 공동 최적화한다. | 서로 대체·상충하는 변수 | 제조 가능 recipe와 재고 제한 포함 |

BA는 단순한 BOC 반응도 억제재가 아니라 시간에 따라 사라지는 공간적 제어변수이다. MHI 자료는 partial-length Gd를 이용한 축방향 출력형상 제어 사례를 제시하고, BA 관련 review와 다수 loading optimization 연구는 BA 배치를 연료위치·방향과 동시에 최적화한다.[R10, R15, R17, R23]

---

## 5.6 제어봉, 정지여유도 및 사고해석 관련 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **C-01** | H | 가장 반응도 높은 제어봉 고착 조건의 SDM을 만족한다. | 배치가 control worth와 residual reactivity를 변경 | HZP·EOC 등 limiting state 확인 |
| **C-02** | N/H | 고반응도 연료를 제어봉 주변에 둘 때 SDM 이득과 rod-ejection penalty를 동시에 본다. | rod worth 향상과 사고국부출력 증가가 상충 | 단일 지표 최적화 금지 |
| **C-03** | H | bank worth와 differential/integral rod worth를 허용범위에서 확인한다. | radial spectrum과 fuel worth가 rod worth에 영향 | 운영 제어전략과 연계 |
| **C-04** | H | dropped-rod/misaligned-rod 상태의 \(F_{\Delta H}\), tilt 및 회복가능성을 평가한다. | 비대칭 rod 상태에서 local peak 가능 | 전노심 또는 적절한 비대칭 모델 |
| **C-05** | H | rod ejection 위치별 worth와 국부 에너지침적을 검사한다. | 고반응도 fuel–rod 조합이 limiting case를 바꿀 수 있음 | 승인된 사고해석 방법론 |
| **C-06** | P/N | 정상운전 bank 이동경로 아래에 과도한 high-duty cluster를 만들지 않는다. | 삽입 이력, AO, power maneuver margin | 플랜트 rod-control strategy별 |
| **C-07** | H | MTC, Doppler coefficient, boron worth를 배치 변경 후 재계산한다. | spectrum과 leakage 변화 | BOC/MOC/EOC 및 HZP/HFP |
| **C-08** | H/O | radial tilt와 quadrant power imbalance 여유를 확보한다. | 제조오차·burnup오차·rod misalignment에 대한 견고성 | 완전대칭 계산값 0만으로 충분하지 않음 |
| **C-09** | Q | ex-core/in-core detector response가 power reconstruction에 충분한지 확인한다. | 저누설 배치가 detector sensitivity를 바꿀 수 있음 | startup flux map과 비교 |
| **C-10** | Q/H | final loading은 startup physics test acceptance와 연결한다. | 계산된 CBC, rod worth, flux symmetry의 현장 확인 | IAEA core management 및 플랜트 절차 |

NRC SRP 4.3은 출력분포·제어·정지여유도·반응도계수·계측 및 안정성을 연계하여 심사한다. 따라서 “출력첨두가 낮은 패턴”이 자동으로 안전한 패턴이 아니며, 제어봉가와 사고해석의 limiting location이 바뀔 수 있다.[R01, R02, R04]

---

## 5.7 집합체 회전, cross-core 이동 및 핀 연소도 균등화 규칙

### 먼저 구분해야 할 세 가지

1. **In-place physical rotation:** 같은 위치에서 집합체를 90°/180°/270° 물리 회전
2. **Cross-core relocation:** 집합체를 노심 중심 반대편 또는 다른 사분면으로 이동
3. **Mathematical symmetry transform:** 계산모델에서 type map을 회전·반사하여 대칭 패턴 생성

세 동작은 동일하지 않다. cross-core 이동 시 집합체를 plant coordinate에 대해 같은 방향으로 유지할지, inward face가 다시 inward를 보도록 회전할지에 따라 핀별 이력이 달라진다.

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **O-01** | Q | 모든 집합체의 physical orientation을 명시적 상태변수로 기록한다. | face/pin burnup과 BA map, hardware 방향 추적 | 방향대칭 연료라도 기록은 유지 |
| **O-02** | N | 외곽에서 생긴 face burnup gradient를 반전시키는 cross-core/rotation을 검토한다. | 장기적으로 pin exposure를 균등화할 수 있음 | 새 위치의 출력·mechanical 조건 우선 |
| **O-03** | H/Q | 회전 효과를 평가하려면 pin 또는 최소 quadrant별 isotopic history가 필요하다. | homogeneous assembly model에서는 회전 전후가 동일 | 단순 nodal XS만으로 회전 최적화 금지 |
| **O-04** | H | 회전 후 새 이웃면의 pin power를 재구성한다. | high/low reactivity interface 방향이 바뀜 | assembly-average power로 대체 불가 |
| **O-05** | P/H | nozzle key, guide thimble, instrumentation, mixing-vane 방향 등 허용 orientation을 확인한다. | 모든 PWR 연료가 임의회전을 허용하지 않음 | 연료설계·취급절차가 최우선 |
| **O-06** | P/H | bow·growth·wear 이력이 특정 방향을 요구하거나 금지하는지 확인한다. | 기계적 변형과 이웃 간 간극·삽입성이 방향 의존 | 검사결과 반영 |
| **O-07** | N | 비대칭 Gd/BA pin map을 가진 집합체는 방향을 neutronic 변수로 최적화한다. | poison-rich face와 fresh neighbor 조합 영향 | pin map이 완전대칭이면 영향 제한 |
| **O-08** | N | 중앙 대칭위치로 이동했다고 해서 자동으로 “균일 연소”가 된다고 가정하지 않는다. | 축형상·이웃·power history가 다름 | 누적 pin burnup histogram으로 확인 |
| **O-09** | Q | core-map rotation과 physical assembly rotation을 입력자료에서 별도 필드로 둔다. | 모델 변환과 현장 동작 혼동 방지 | loading instruction에 방향 화살표/키 포함 |
| **O-10** | H/Q | reconstituted fuel, failed-rod repair, LTA는 별도 orientation 제한을 적용한다. | 구조·계측·시험목적이 일반연료와 다름 | 공급사 승인 필요 |
| **O-11** | O | 회전의 목적함수를 max pin burnup, radial pin peak, duty 균등화로 명시한다. | “균일화”의 정의가 여러 개 | 한 지표 개선이 bow/CRUD를 악화할 수 있음 |
| **O-12** | Q | 회전 전후 isotopic remapping을 단위시험한다. | 90° index 오류는 계산상 그럴듯하지만 잘못된 결과 생성 | pin ID permutation 검증 |

Robinson 등은 PWR shuffling 최적화에서 assembly exchange와 rotation을 함께 사용해 power peak를 낮추는 절차를 제시했고, Kropaczek–Turinsky와 Wu의 최적화 연구도 조사연료 orientation을 설계변수에 포함했다.[R15–R17] 다만 이는 **회전이 언제나 유익하다**는 뜻이 아니라, 방향성 이력이 해석모델과 연료취급절차에 정확히 반영될 때 유효한 자유도라는 뜻이다.

---

## 5.8 3-D, 상태점 및 시간의존 검증 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **T-01** | H | 최소 BOC/MOC/EOC HFP 상태를 평가한다. | peak와 계수의 limiting burnup이 다름 | BA 소진구간 추가 |
| **T-02** | H | HZP/CZP 계열에서 CBC, MTC, SDM, rod worth를 확인한다. | hot-power optimum이 zero-power에서 limiting일 수 있음 | 플랜트 승인 state matrix 적용 |
| **T-03** | H | equilibrium xenon과 xenon-free/변동 상태를 구분한다. | 제논이 power shape와 reactivity를 크게 변경 | startup·shutdown·maneuver 조건 |
| **T-04** | H | samarium 및 장기 poison history를 depletion에 일관되게 반영한다. | burnup-equivalent라도 poison state가 다를 수 있음 | restart history 갱신 |
| **T-05** | H | axial peaking, AO 및 part-length BA 효과는 3-D로 평가한다. | 2-D radial flatness가 \(F_q\)를 보장하지 않음 | full-core 3-D final |
| **T-06** | H | Gd burnout, bank movement, power maneuver 등 형상변화 구간에 세분 burnup step을 둔다. | peak가 coarse depletion point 사이에서 발생 가능 | interpolation 검증 |
| **T-07** | H | radial–axial coupling을 포함해 pin power를 복원한다. | radial peak와 axial peak의 위치가 결합 | 보수적 peaking factor 조합 검토 |
| **T-08** | H/Q | branch conditions의 cross-section 범위를 벗어나지 않는지 확인한다. | 비정상 boron/temperature extrapolation 오류 | lattice library QA |
| **T-09** | H | 불확실도·bias를 적용한 제한값으로 평가한다. | nominal 계산의 작은 여유는 수락여유가 아님 | 방법론별 statistical allowance |
| **T-10** | Q | 최종 승인 패턴의 critical boron curve와 predicted flux maps를 보존한다. | startup 및 core-follow 비교 기준 | configuration control |

---

## 5.9 열수력, 연료성능, CRUD/CIPS 및 기계적 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **M-01** | H | \(F_q\), \(F_{\Delta H}\), LHGR, DNBR 관련 지표를 주기 전 구간에서 만족한다. | 핵설계의 1차 안전제약 | 정의·한계는 노형별 |
| **M-02** | H | assembly power가 아니라 pin/subchannel 조건으로 hot spot을 확인한다. | 동일 assembly power에서도 pin map·flow가 다름 | validated reconstruction 필요 |
| **M-03** | H | 고연소도 연료의 재상승 power ramp와 선출력 이력을 제한한다. | PCI, cladding strain, corrosion 관련 | fuel performance analysis로 판단 |
| **M-04** | H/O | high-duty 신연료 cluster가 subcooled boiling/CRUD 위험을 높이는지 평가한다. | CRUD/CIPS는 3-D pin/subchannel duty와 연계 | chemistry와 운전이력도 필요 |
| **M-05** | O | CIPS 위험지표를 loading objective 또는 constraint에 포함한다. | Yamamoto와 ORNL/NCSU 연구에서 명시적 최적화 대상 | surrogate는 고충실도 검증 필요 |
| **M-06** | H/P | 연료형식·spacer grid·압력강하가 다른 transition core의 hydraulic compatibility를 확인한다. | 유량 재분배가 DNBR/진동에 영향 | neutronic flatness보다 우선 |
| **M-07** | P/H | LTA는 시험목적과 instrumentability를 만족하는 허용 위치에 둔다. | 대표성, 보수성, 회수·감시 | 단순 최저출력 위치가 항상 최선 아님 |
| **M-08** | P/H | 검사에서 bow/growth/wear 이상이 있는 집합체는 허용 위치·방향을 제한한다. | 삽입성, control-rod drag, grid interaction | 연료공급사 disposition |
| **M-09** | N/P | 큰 이웃 burnup gradient가 assembly bow/growth 비대칭을 악화할 가능성을 검토한다. | 공개 KNS 연구에서 주변 burnup 분포 영향 논의 | 정량 평가는 구조해석/경험자료 |
| **M-10** | H | failed fuel, repaired fuel, debris-filter status 등 운전상태를 위치 제약으로 반영한다. | 누설관리·재고 건전성 | 플랜트 fuel reliability policy |
| **M-11** | O | 용기 fluence, baffle heating 및 fuel duty를 함께 Pareto 평가한다. | 저누설이 내부 power/duty를 높일 수 있음 | 단일 leakage 최소화 금지 |
| **M-12** | H/Q | coupled neutronic–thermal-hydraulic–fuel-performance 결과의 데이터 일관성을 검증한다. | 서로 다른 mesh/state mapping 오류 가능 | 독립 interface QA |
| **M-13** | N | high-power location을 동일 집합체에 반복 부여하지 않도록 cumulative duty 지표를 사용한다. | peak burnup 외에 corrosion/CRUD history 반영 | 모든 고출력이 금지되는 것은 아님 |
| **M-14** | H | mixed-fuel transition에서 fuel temperature coefficient와 thermal limits의 방법론 적용범위를 확인한다. | 새 연료형식의 extrapolation 위험 | licensing basis 확인 |

Andersen 등은 loading pattern과 관련된 CRUD/boron deposition 분포를 최적화할 수 있음을 보였고, INL의 2023 platform은 노심물리뿐 아니라 시스템·연료성능 피드백을 결합하려는 방향을 제시한다.[R09, R22] 이는 현대 reload design이 neutronics 단독 최적화에서 multiphysics·risk-informed 최적화로 확장되고 있음을 보여준다.

---

## 5.10 운전, 연료취급 및 품질보증 규칙

| ID | 분류 | 규칙 | 근거 | 예외·상충관계 |
|---|---|---|---|---|
| **Q-01** | Q | 최종 pattern은 fuel type map이 아니라 serial-ID + orientation map으로 발행한다. | 잘못된 개체·방향 loading 방지 | 현장 표준형식 준수 |
| **Q-02** | Q/P | in-core shuffle과 full-core offload/reload의 실제 이동순서를 검토한다. | 취급장비·임시저장·충돌·시간 제약 | 최적 물리패턴이 취급 불가능할 수 있음 |
| **Q-03** | Q | 핵설계 데이터베이스와 현장 연료관리시스템의 burnup/history를 reconcile한다. | 데이터 불일치가 가장 위험한 오류원 중 하나 | outage 전후 독립확인 |
| **Q-04** | Q | loading instruction에 위치좌표, ID, 방향, 연료형식, 검사상태를 명시한다. | human-factor 오류 저감 | barcode/vision verification 가능 |
| **Q-05** | Q | 독립 계산 또는 독립 검토로 핵설계 결과를 확인한다. | 모델·입력·postprocessing 공통원인 오류 방지 | 플랜트 QA program |
| **Q-06** | Q/H | post-loading verification과 startup physics test를 수행한다. | 실제 loading, CBC, rod worth, flux symmetry 확인 | 불일치 시 원인분석 |
| **Q-07** | Q | final full-core model에서 대칭 복제 오류와 unique history mapping을 검사한다. | type map은 맞아도 ID/orientation이 틀릴 수 있음 | checksum·automated audit |
| **Q-08** | Q/O | 예측오차에 대한 robust margin을 확보한다. | depletion, XS, manufacturing, operation uncertainty | 최소한 bounding perturbation |
| **Q-09** | Q | damaged/unavailable assembly 발생 시 contingency pattern을 준비한다. | outage 중 검사결과로 재고가 변할 수 있음 | 빠른 재해석 workflow |
| **Q-10** | Q | 모든 forbidden pattern과 예외승인을 configuration-controlled rule set으로 관리한다. | 엔지니어 개인 기억에 의존하지 않음 | 버전·근거·승인자 기록 |
| **Q-11** | Q | optimizer가 낸 후보의 규칙 위반 사유를 설명 가능한 형태로 출력한다. | QA와 설계검토 효율 | black-box score만 제공하지 않음 |
| **Q-12** | Q | startup 및 core-follow 측정으로 모델 bias를 갱신하되 임의 보정하지 않는다. | 다음 주기 예측개선 | 승인된 methodology/change control |
| **Q-13** | Q | 설계변경 시 영향을 받는 안전해석 set을 추적한다. | 작은 LP 변경도 limiting accident를 바꿀 수 있음 | impact matrix 사용 |
| **Q-14** | Q | 최종 후보의 모든 입력·코드버전·XS library·script hash를 보존한다. | 재현성과 규제추적성 | 장기 기록관리 |

IAEA SSG-73은 loading pattern 결정, 연료취급, post-loading verification 및 startup/core monitoring을 하나의 core-management 프로그램으로 다룬다. 즉, 좋은 loading pattern은 계산파일에서 끝나는 것이 아니라 **정확히 장전되고 측정으로 확인될 때** 완성된다.[R01, R03]

---

# 6. 사용자가 제시한 예시 규칙에 대한 판정

| 예시 | 판정 | 정확한 해석 |
|---|---|---|
| **IN–OUT / OUT–IN shuffling** | **둘 다 유효하나 목적과 시대·노형이 다름** | OUT–IN은 외곽 누설로 신연료 출력을 억제하지만 neutron economy와 vessel fluence에 불리할 수 있다. IN–OUT/저누설은 신연료를 내부에 두고 조사연료를 외곽에 두는 경향이며 현대 상용 PWR에서 널리 쓰이는 철학이다. 실제 패턴은 hybrid이다. |
| **1/4 대칭 준수** | **흔한 P/Q 규칙이지 보편적 물리법칙은 아님** | inventory multiplicity와 radial tilt, 탐색효율에 유리하다. 다만 mirror quarter symmetry와 C4 rotational symmetry를 구분하고, 실제 고유 연료이력이 비대칭이면 전노심 검증해야 한다. |
| **노심 중앙 기준으로 회전시켜 핀 연소 균등화** | **조건부로 타당** | physical rotation, cross-core 이동, 수학적 회전을 구분해야 한다. pin/quadrant isotopic history가 있고 연료 hardware가 회전을 허용할 때만 의미가 있다. 새 위치의 pin peak·bow·BA 방향성을 함께 검증해야 한다. |
| **BOC K값이 높은 집합체를 바깥에 배치** | **보편규칙으로는 부정확** | OUT–IN 또는 특정 flattening에서는 사용할 수 있으나 저누설 IN–OUT에서는 대개 높은 reactivity 연료를 내부에 둔다. 또한 “K값”이 \(k_\infty\)인지 location worth인지, HFP/HZP와 붕산·제논 조건이 무엇인지 명시해야 한다. |
| **고연소도 집합체는 바깥에 배치** | **저누설의 대표 규칙이나 절대법칙은 아님** | vessel fluence를 줄이는 데 유리하지만, 매우 고연소도 연료의 residual reactivity 회수나 discharge target 때문에 내부 fresh 인접 위치에 둘 수 있다. high-burnup duty 제한이 핵심이다. |
| **신연료는 checkerboard로 배치** | **강력한 휴리스틱이지만 자동 수락규칙은 아님** | fresh adjacency와 radial peak를 줄이지만 BA·enrichment mismatch와 reflector interface pin peak가 남을 수 있다. |
| **Gd 집합체는 붙이지 않는다** | **공개된 실제 플랜트/모델 규칙의 예이나 보편적 절대법칙은 아님** | Gd cluster의 저출력 hole과 later rebound를 피하려는 목적이다. Gd 농도·rod 수·위치·소진 이력에 따라 허용 가능하다. |

---

# 7. 규칙 간 대표 상충관계

| 설계 선택 | 개선되는 항목 | 악화될 수 있는 항목 | 엔지니어링 대응 |
|---|---|---|---|
| 신연료를 내부로 이동 | neutron economy, cycle length | central peak, CBC, MTC, SDM/rod-ejection tradeoff | burnt/BA buffer, 3-D pin check |
| 고연소도 연료를 외곽으로 이동 | vessel fluence, leakage | stranded reactivity, 낮은 discharge increment | 외곽층 강도 최적화, 일부 내부 재사용 |
| checkerboard 강화 | radial peak, fresh adjacency | interface pin peak, 복잡한 BA interaction | pin reconstruction, mismatch metric |
| strong BA 증가 | BOC CBC, peak, MTC 관리 | EOC residual absorption, power rebound | depletion-wide optimization |
| 제어봉 주변 reactivity 증가 | rod worth, SDM | rod-ejection worth/local energy | accident-specific constraint |
| 대칭성 강화 | tilt, QA, 계산효율 | 실제 inventory 활용성, LTA flexibility | symmetric seed + full-core refinement |
| assembly rotation | pin burnup 균등화 | 새 interface peak, bow/handling 문제 | pin history + mechanical screening |
| 저누설 강화 | vessel fluence, neutron economy | 내부 duty, CIPS/CRUD, \(F_{\Delta H}\) | multiphysics Pareto optimization |
| EOC \(k\) 최대화 | 현 주기길이 | 다음 주기 inventory, peak burnup | 2–3 cycle look-ahead |
| radial peak 최소화 | thermal margin | enrichment/BA 비용, cycle length | Pareto front와 비용평가 |
| 고출력 연료 분산 | assembly peak | 더 많은 high-low interfaces | face pin metric과 diagonal check |
| 외곽 fresh 사용 | radial flattening | leakage·vessel fluence | 제한된 위치, fluence budget |

---

# 8. 권장 엔지니어링 설계 절차

## 단계 1. 입력 재고와 위치속성 정리

### 집합체 레코드

```text
Assembly_ID
Fuel_design / lattice / nozzle / grid type
Fresh or irradiated
Enrichment
BA type, concentration, rod map, axial zoning
Assembly-average burnup
Axial-node burnup
Pin/quadrant burnup and isotopics
Previous core positions and orientations
Power/LHGR/CRUD/corrosion history
Inspection disposition: bow, growth, wear, damage
Allowed orientations
Allowed / prohibited location classes
Future-cycle status and discharge candidate flag
```

### 위치 레코드

```text
Core coordinate
Center / inboard / edge / corner / reflector adjacency
Symmetry class and multiplicity
Control-rod location and bank
In-core detector / instrumentation
Baffle, water-gap and vessel azimuth
Hydraulic / flow characteristics
Allowed fuel designs and orientations
LTA or surveillance restrictions
Handling restrictions
```

## 단계 2. Hard filter

다음은 optimizer가 물리계산 전에 배제할 수 있는 대표 항목이다.

- 재고 수량 불일치
- symmetry multiplicity 위반
- 위치·연료형식 비호환
- 허용되지 않은 orientation
- 검사 부적합 연료 사용
- LTA/계측 위치 위반
- 명백한 plant-specific forbidden pattern
- residence/burnup limit 초과가 확정된 연료
- handling 불가능한 shuffle

## 단계 3. Physics-informed seed 생성

1. 내부에 신연료·잔류반응도 높은 조사연료를 배치하되 분산
2. 외곽에 저누설 buffer 후보 배치
3. inboard 무BA fresh adjacency 최소화
4. strong BA cluster 억제
5. center/edge/corner의 서로 다른 importance 고려
6. control-rod/SDM 및 detector 제약 고려
7. 기존 검증 pattern과 가까운 보수적 seed도 병행

## 단계 4. 저비용 screening

- 2-D radial nodal 또는 surrogate 계산
- BOC reactivity, coarse radial peaking, CBC proxy
- forbidden pattern 및 simple mismatch metrics
- vessel leakage proxy
- inventory/future-cycle score

단, Park 등의 연구가 보여주듯 2-D screening은 3-D 계산을 줄이는 도구이지 최종 수락을 대체하지 않는다.[R18]

## 단계 5. 3-D depletion 및 pin reconstruction

- BOC/MOC/EOC와 BA burnout 구간
- equilibrium/non-equilibrium xenon 요구상태
- HFP/HZP branch
- \(F_q\), \(F_{\Delta H}\), radial pin peak
- CBC, MTC, boron worth
- assembly/pin burnup
- AO, radial tilt
- vessel fast flux

## 단계 6. 제어봉·사고·열수력·연료성능 평가

- SDM with most reactive rod stuck
- rod worth, ejected/dropped rod
- DNBR/LHGR/fuel temperature
- high-burnup duty, PCI/corrosion
- CIPS/CRUD risk
- bow/growth 및 hydraulic compatibility
- limiting accident set의 변화

## 단계 7. 다주기 및 불확실도 평가

- 다음 2–3주기 예상 feed 수량과 enrichment
- 잔류 재고의 배치가능성
- burnup/operating-history perturbation
- cross-section/model bias
- manufacturing tolerance
- outage contingency

## 단계 8. 독립검토, 현장 loading 및 startup 검증

- 독립 계산 또는 설계검토
- serial ID/orientation loading map
- post-loading verification
- predicted CBC와 actual critical boron 비교
- rod worth 및 flux symmetry 확인
- in-core/ex-core flux map과 predicted distribution 비교
- 차이를 다음 주기 모델 bias 관리에 반영

---

# 9. 최적화 코드에 규칙을 구현하는 권장 방식

## 9.1 경성 제약과 penalty를 분리한다

```text
if violates_inventory(candidate):
    reject

if violates_hardware_or_inspection(candidate):
    reject

if violates_licensed_or_plant_hard_rule(candidate):
    reject

score = physics_objectives(candidate)
score += soft_penalty_for_adjacency(candidate)
score += soft_penalty_for_leakage(candidate)
score += soft_penalty_for_future_inventory(candidate)
```

안전상 진짜 경성인 항목을 penalty로만 처리하면 “큰 경제적 보상으로 안전위반을 상쇄”하는 비물리적 후보가 생긴다. 반대로 Ring-of-Fire와 같은 휴리스틱을 무조건 reject하면 좋은 해를 잘라낼 수 있다.[R11]

## 9.2 권장 decision variables

- assembly-to-position permutation
- physical orientation \(0^\circ,90^\circ,180^\circ,270^\circ\)
- fresh fuel recipe/enrichment
- BA type/count/map
- discharge/retain decision
- 다음 주기용 inventory reserve
- 필요 시 control strategy 또는 operating constraints

## 9.3 유용한 물리기반 feature

```text
location_importance
edge/corner leakage class
assembly reactivity index at multiple states
residual BA worth
face-wise burnup
neighbor reactivity mismatch
fresh-fresh face/diagonal count
Gd-Gd adjacency count
distance to control rod
distance to reflector
predicted vessel leakage contribution
cumulative high-duty index
future-cycle scarcity score
```

## 9.4 adjacency를 단순 0/1보다 정교하게 표현

단순 금지 수 외에 다음 metric을 사용할 수 있다.

\[
M_{\mathrm{face}} =
\sum_{\langle i,j\rangle}
w_{ij}\,
|RI_i-RI_j|
\]

\[
M_{\mathrm{diag}} =
\sum_{\langle\langle i,j\rangle\rangle}
\alpha w_{ij}\,
|RI_i-RI_j|,
\quad 0<\alpha<1
\]

단, mismatch를 무조건 최소화하면 고·저반응도 혼합에 의한 power sharing 이점을 잃는다. 따라서 실제 pin peak surrogate 또는 nodal response와 함께 사용해야 한다.

## 9.5 symmetry-preserving operator

- general quarter position은 4개를 한 gene으로 처리
- axis position은 2개, center는 1개
- crossover 후 각 inventory class 수량을 repair
- orientation은 symmetry transform에 맞게 mapping
- physical orientation과 core-map transform을 별도 처리
- 비대칭 refinement 단계에서 unique assembly ID를 다시 할당

## 9.6 다단계 fidelity

| 단계 | 모델 | 역할 |
|---|---|---|
| 0 | combinatorial rules | 명백한 불가능 후보 제거 |
| 1 | response matrix / surrogate / 2-D nodal | 대량 후보 screening |
| 2 | 3-D nodal depletion | 주요 neutronic constraints |
| 3 | pin reconstruction / subchannel | local thermal margin |
| 4 | accident, fuel performance, vessel fluence | 최종 safety/multiphysics 확인 |
| 5 | independent production code | release candidate 승인 |

---

# 10. 권장 상태점 검증행렬

아래는 일반적인 검증 프레임이며 실제 플랜트 설계절차가 우선한다.

| 상태 | 핵심 확인항목 |
|---|---|
| BOC HFP, equilibrium Xe | CBC, MTC, radial/3-D peaking, DNBR proxy, rod worth |
| BOC HFP, Xe-free 또는 startup condition | excess reactivity, power shape, control margin |
| BOC HZP/CZP | SDM, MTC, boron worth, rod worth, ejection limiting cases |
| Early-cycle Gd depletion points | power rebound, local pin peak, AO |
| MOC HFP | peak shift, \(F_q\), \(F_{\Delta H}\), CIPS/CRUD duty |
| EOC HFP | cycle length, residual reactivity, maximum burnup, power shape |
| EOC HZP | SDM 및 limiting rod worth |
| Bank insertion/maneuver states | AO, radial tilt, local peaking |
| Dropped/misaligned rod | asymmetric power, \(F_{\Delta H}\), detector response |
| Rod ejection set | worth, local energy deposition, high-reactivity neighbor |
| Bounding temperature/boron states | XS branch validity, feedback coefficients |
| Uncertainty cases | burnup bias, power history, manufacturing, model bias |
| Full-core unique-history case | symmetry-breaking effects, LTA/instrumentation, actual IDs |

---

# 11. 최종 loading pattern release checklist

## 11.1 재고와 지도

- [ ] 모든 assembly serial ID가 한 번만 사용되었는가?
- [ ] 신연료 recipe, BA, enrichment 수량이 조달 재고와 일치하는가?
- [ ] burnup, axial history, pin/quadrant history가 올바른 ID에 연결되었는가?
- [ ] orientation이 계산모델과 loading instruction에서 동일한가?
- [ ] 대칭 multiplicity와 unique-ID mapping이 일치하는가?
- [ ] LTA, detector, repaired/damaged fuel 제한이 반영되었는가?

## 11.2 Neutronics

- [ ] BOC/MOC/EOC 및 BA burnout 구간의 \(k\), CBC가 만족되는가?
- [ ] \(F_q\), \(F_{\Delta H}\), radial pin peak가 uncertainty 포함 제한 내인가?
- [ ] assembly/pin burnup 한계를 만족하는가?
- [ ] MTC, boron worth, Doppler coefficient가 수락되는가?
- [ ] SDM, bank worth, ejected/dropped rod cases가 수락되는가?
- [ ] AO, radial tilt 및 xenon stability 여유가 충분한가?
- [ ] vessel fast fluence 및 외곽 pin power가 수락되는가?

## 11.3 Multiphysics와 연료성능

- [ ] DNBR/LHGR/fuel-temperature 관련 제한을 만족하는가?
- [ ] 고연소도 연료의 power history와 ramp가 허용되는가?
- [ ] CIPS/CRUD 및 subcooled boiling high-duty 영역이 수락되는가?
- [ ] fuel design transition과 hydraulic compatibility가 확인되었는가?
- [ ] bow/growth/wear와 allowed orientation이 확인되었는가?

## 11.4 다주기·운전·QA

- [ ] 다음 2–3주기 feed와 residual inventory가 가능한가?
- [ ] outage 중 연료 불용 시 contingency가 있는가?
- [ ] shuffle sequence가 실제 취급절차로 수행 가능한가?
- [ ] independent calculation/review가 완료되었는가?
- [ ] 코드, XS library, input, script 버전이 보존되었는가?
- [ ] startup test 예측값과 acceptance band가 준비되었는가?

---

# 12. 핵심 오해와 교정

## 오해 1. “반응도가 높은 연료를 외곽에 놓으면 출력이 낮아지므로 항상 안전하다.”

외곽 출력은 낮아질 수 있지만 다음 문제가 생긴다.

- 신연료의 반응도를 누설로 잃어 neutron economy가 나빠짐
- vessel fast fluence가 증가할 수 있음
- reflector/baffle 경계 pin peak가 남음
- 향후 주기 재고효율이 나빠질 수 있음
- ex-core detector와 shielding response가 변함

따라서 “외곽 배치로 출력이 낮아진다”는 사실만으로 수락할 수 없다.

## 오해 2. “1/4 map이 대칭이면 실제 연소이력도 대칭이다.”

연료 type map은 대칭이어도 다음은 비대칭일 수 있다.

- 고유 집합체의 이전 위치
- pin/face burnup
- physical orientation
- Gd residual distribution
- LTA·repaired fuel
- bow·growth·wear
- detector·baffle·운전이력

따라서 최종 전노심 unique-history model이 필요하다.

## 오해 3. “집합체를 180° 돌리면 핀 연소도가 자동으로 균일해진다.”

회전은 기존 gradient의 방향을 바꾸지만 새 위치의 flux gradient와 이웃이 다르다. 균일화 여부는

- 회전 전 pin burnup
- 새 위치 pin power
- 잔여 residence time
- BA map
- mechanical restrictions

를 적분한 결과로 판단해야 한다.

## 오해 4. “checkerboard면 pin peaking도 자동으로 낮다.”

checkerboard는 assembly-level cluster를 줄이지만 고반응도–저반응도 경계면 핀은 오히려 높은 국부 gradient를 가질 수 있다. pin reconstruction이 필수이다.

## 오해 5. “EOC \(k_{\mathrm{eff}}\)가 가장 큰 패턴이 가장 경제적이다.”

EOC \(k\) 최대화는 현 주기의 길이를 늘릴 수 있으나, fresh enrichment, maximum burnup, 향후 재고, BA 비용, vessel fluence 및 thermal margin을 포함한 총연료주기비용과 같지 않다.[R15]

---

# 13. 문헌에서 직접 확인되는 대표 규칙과 본 문서의 해석

| 공개자료 | 확인되는 내용 | 본 문서에서의 적용 |
|---|---|---|
| IAEA SSG-73 [R01] | loading pattern, 연료취급, post-loading verification, startup/core monitoring을 일관된 관리프로그램으로 요구 | Q-01–Q-14 및 전체 release workflow |
| IAEA SSG-52 [R02] | neutronic, thermal-hydraulic, thermomechanical, structural aspects의 통합 | M-01–M-14, 최종 multiphysics 검증 |
| IAEA TECDOC-1898 [R03] | reload design의 power distribution, safety, 경험·교훈 및 다부서 연계 | 단순 반응도 최적화를 넘어선 rule hierarchy |
| NRC SRP 4.3 [R04] | power distribution, reactivity coefficients, control, SDM, stability, instrumentation 등 핵설계 심사항목 | C/T 계열 규칙과 state matrix |
| OECD/NEA high-burnup reviews [R06, R07] | 저누설 loading, 외곽 고연소도 연료, 현대 고연소도에서의 peaking·duty 변화 | L-01–L-10 및 B-03/B-06 |
| INL 2021 [R08] | operating-PWR reload optimization demonstration, low-leakage형 외곽 조사연료, Gd 인접 제한 예, full-core evaluation | 실제 plant-rule 예와 optimizer 구현 |
| MHI 2009 [R10] | 조사연료 외곽 저누설 core, partial-length Gd에 의한 axial shape control | L 계열과 A-11 |
| KNFC 2016 [R11] | APR1400 LP에서 cycle length, \(F_{xy}\), pin burnup, MTC, Ring-of-Fire/adjacent fresh 처리 | R-03–R-06, hard vs penalty 구분 |
| Yamamoto lecture [R12] | PWR constraints/objectives, fresh/BA 위치 제한, side/diagonal adjacency, 2–3주기 feasibility | 상세 rule catalog의 핵심 실무 근거 |
| Kropaczek–Turinsky [R15] | feed/exposed fuel, orientation, BA를 함께 최적화; 경제성과 peaking의 명시적 tradeoff | O 계열 및 다목적 설계 |
| Robinson et al. [R16] | 교환과 회전을 포함한 shuffling으로 power peak 저감 | physical rotation rule |
| Wu [R17] | assembly, BA, burnt-fuel orientation 동시 GA 최적화 | orientation을 독립 설계변수로 취급 |
| Park et al. [R18] | discontinuous penalty와 2-D screening을 사용한 다목적 PWR LP 최적화 | 다단계 fidelity와 penalty 설계 |
| Andersen et al. [R22] | loading/assembly 설계 최적화를 통한 CRUD 관련 분포 제어 가능성 | CIPS/CRUD를 목적함수에 포함 |

---

# 14. 결론

상용 PWR의 연료집합체 배치는 다음 한 문장으로 요약할 수 있다.

> **“고반응도 연료를 분산하고, 외곽 누설과 내부 출력첨두를 절충하며, 조사이력과 방향성을 보존한 상태에서, 주기 전 구간의 제어·열적·연료성능 제한과 다음 주기 재고까지 만족시키는 전노심 3-D 다목적 설계”**

실무에서 가장 중요한 것은 특정 경험칙을 암기하는 것이 아니라 **그 규칙이 무엇을 방지하려고 만들어졌는지**, **어떤 상태점에서 유효한지**, **어떤 다른 제한과 상충하는지**를 이해하는 것이다.

- OUT–IN은 high-reactivity fuel을 외곽 누설로 억제하는 전략이고,
- IN–OUT/저누설은 high-reactivity fuel을 내부 중요도에 활용하고 burnt fuel로 외곽을 완충하는 전략이며,
- checkerboard와 인접 금지는 국부 power cluster를 줄이는 휴리스틱이고,
- 1/4 대칭은 search·tilt·QA를 위한 실무 제약이며,
- assembly rotation은 pin-level history와 hardware constraint가 있을 때만 의미가 있고,
- `k∞` 순위는 위치별 worth와 power response를 대신하지 못한다.

최종적으로 엔지니어는 규칙을 다음과 같이 사용해야 한다.

1. **경성 안전·기계 제약은 즉시 배제**
2. **플랜트 금지 패턴은 명시적으로 관리**
3. **물리 휴리스틱은 seed와 탐색편향에 사용**
4. **경제성·여유는 다목적 Pareto 문제로 처리**
5. **최종 후보는 unique-history 전노심 3-D 및 multiphysics로 검증**
6. **현장 장전과 startup measurement로 닫힌 검증루프 완성**

---

# 부록 A. 용어

| 용어 | 의미 |
|---|---|
| FA | Fuel Assembly, 연료집합체 |
| LP | Loading Pattern |
| BOC/MOC/EOC | Beginning/Middle/End of Cycle |
| HFP/HZP/CZP | Hot Full/Zero Power, Cold Zero Power |
| CBC | Critical Boron Concentration |
| BA | Burnable Absorber |
| Gd | Gadolinia-bearing fuel |
| IFBA | Integral Fuel Burnable Absorber |
| WABA | Wet Annular Burnable Absorber |
| MTC | Moderator Temperature Coefficient |
| SDM | Shutdown Margin |
| AO/ASI | Axial Offset/Axial Shape Index |
| \(F_q\) | 3-D nuclear enthalpy/heat-flux peaking 관련 지표; 정확한 정의는 공급사별 |
| \(F_{\Delta H}\) | Nuclear enthalpy rise hot-channel factor |
| \(F_{xy}\) | Radial pin power peaking factor의 한 표현 |
| LHGR | Linear Heat Generation Rate |
| DNBR | Departure from Nucleate Boiling Ratio |
| CIPS | Crud-Induced Power Shift |
| LTA | Lead Test Assembly |
| Low-leakage | 외곽 fission source/fast leakage를 낮추는 loading 철학 |
| OUT–IN | 신연료를 바깥쪽에서 시작해 이후 안쪽으로 이동하는 계열 |
| IN–OUT | 신연료를 안쪽에서 시작해 조사 후 바깥쪽으로 이동하는 계열 |
| Cross-core shuffle | 노심 중심을 가로질러 다른 사분면/반대편으로 이동 |
| Orientation | 집합체의 물리적 방위 |

---

# 부록 B. 참고문헌

## 국제기구·규제·공공기관

**[R01]** International Atomic Energy Agency, *Core Management and Fuel Handling for Nuclear Power Plants*, IAEA Safety Standards Series No. SSG-73, Vienna, 2022.  
Official page: https://www.iaea.org/publications/14904/core-management-and-fuel-handling-for-nuclear-power-plants

**[R02]** International Atomic Energy Agency, *Design of the Reactor Core for Nuclear Power Plants*, IAEA Safety Standards Series No. SSG-52, Vienna, 2019.  
Official page: https://www.iaea.org/publications/13382/design-of-the-reactor-core-for-nuclear-power-plants

**[R03]** International Atomic Energy Agency, *Reload Design and Core Management in Operating Nuclear Power Plants: Experiences and Lessons Learned*, IAEA-TECDOC-1898, Vienna, 2020.  
Official page: https://www.iaea.org/publications/13585/reload-design-and-core-management-in-operating-nuclear-power-plants

**[R04]** U.S. Nuclear Regulatory Commission, *Standard Review Plan for the Review of Safety Analysis Reports for Nuclear Power Plants: LWR Edition—Section 4.3, Nuclear Design*, NUREG-0800, Rev. 3, 2007.  
PDF: https://www.nrc.gov/docs/ML0703/ML070380179.pdf

**[R05]** U.S. Nuclear Regulatory Commission, *NUREG-0800, Section 4.4, Thermal and Hydraulic Design*.  
PDF: https://www.nrc.gov/docs/ML0523/ML052340664.pdf

**[R06]** OECD Nuclear Energy Agency, *Very High Burn-ups in Light Water Reactors*, NEA No. 6224, Paris, 2006.  
PDF: https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nea6224-burn-up.pdf

**[R07]** OECD Nuclear Energy Agency, *Nuclear Fuel Safety Criteria Technical Review*, 2nd ed., NEA No. 7072, Paris, 2012.  
PDF: https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nea7072-fuel-safety-criteria.pdf

## 상용·국립연구소·연료공급사 자료

**[R08]** Y.-J. Choi, M. Abdo, D. Mandelli, A. Epiney, J. Valeri, C. Gosdin, C. Frepoli, and A. Alfonsi, *Demonstration of the Plant Fuel Reload Process Optimization for an Operating PWR*, INL/EXT-21-64549, Idaho National Laboratory, 2021.  
PDF: https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_53142.pdf

**[R09]** J. Kim, M. Abdo, Y.-J. Choi, J. C. Luque Gutierrez, J. Hou, C. Gosdin, and J. Valeri, *Pressurized-Water Reactor Core Design Demonstration with Genetic Algorithm Based Multi-Objective Plant Fuel Reload Optimization Platform*, INL/RPT-23-74498, 2023.  
PDF: https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_67483.pdf  
DOI record: https://doi.org/10.2172/2006437

**[R10]** E. Saji et al., “Development of Advanced PWR Fuel and Core for High Reliability and Performance,” *Mitsubishi Heavy Industries Technical Review*, Vol. 46, No. 4, pp. 29–34, 2009.  
PDF: https://www.mhi.com/technology/review/sites/g/files/jwhtju2326/files/tr/pdf/e464/e464029.pdf

**[R11]** Y. D. Nam, H. C. Lee, and C. H. Im, “Parameters and Constraints Optimization of McFLOP for APR1400 Type Plant,” Transactions of the Korean Nuclear Society, 2016.  
PDF: https://www.kns.org/files/pre_paper/35/16S-638%EB%82%A8%EC%9C%A4%EB%8D%95.pdf

**[R12]** A. Yamamoto, *Loading Pattern Optimization for Commercial Reactors*, Reactor Physics lecture, Nagoya University/Hokkaido University OCW, 2021.  
PDF: https://ocw.hokudai.ac.jp/wp-content/uploads/2022/04/Lecture15_Reactor-Physics-Nagoya-University-Akio-YAMAMOTO.pdf

## 학술논문·학위논문

**[R13]** A. Yamamoto, E. Sugimura, Y. Kitamura, and Y. Yamane, “Simultaneous Loading Patterns Optimization for Two Successive Cycles of Pressurized Water Reactors,” *Journal of Nuclear Science and Technology*, Vol. 41, pp. 1065–1074, 2004.  
DOI: https://doi.org/10.3327/jnst.41.1065

**[R14]** A. Yamamoto, *Study on Advanced In-Core Fuel Management for Pressurized Water Reactors Using Loading Pattern Optimization Methods*, doctoral thesis, Kyoto University.  
Repository: https://repository.kulib.kyoto-u.ac.jp/bitstream/2433/156982/2/D_Yamamoto_Akio.pdf

**[R15]** D. J. Kropaczek and P. J. Turinsky, “In-Core Nuclear Fuel Management Optimization for Pressurized Water Reactors Utilizing Simulated Annealing,” *Nuclear Technology*, Vol. 95, No. 1, pp. 9–32, 1991.  
DOI: https://doi.org/10.13182/NT95-1-9

**[R16]** A. H. Robinson, J. O. Heaberlin, and G. L. Wang, “An Automated Search Procedure for Fuel Shuffling in PWRs Including Rotation Effects,” in *Artificial Intelligence and Other Innovative Computer Applications in the Nuclear Industry*, pp. 645–651, 1988.  
DOI: https://doi.org/10.1007/978-1-4613-1009-9_78

**[R17]** H. Wu, “Pressurized Water Reactor Reloading Optimization Using Genetic Algorithms,” *Annals of Nuclear Energy*, Vol. 28, No. 13, pp. 1329–1341, 2001.  
Publisher record: https://www.sciencedirect.com/science/article/abs/pii/S0306454900001225  
Open copy: https://necp.xjtu.edu.cn/__local/A/61/EB/F050AA42C0F08526F80D2D60133_73B2DC4A_5BCA9.pdf

**[R18]** T. K. Park, H. C. Lee, H. G. Joo, and C. H. Kim, “Multiobjective Loading Pattern Optimization by Simulated Annealing Employing Discontinuous Penalty Function and Screening Technique,” *Nuclear Science and Engineering*, Vol. 162, pp. 134–147, 2009.  
DOI: https://doi.org/10.13182/NSE162-134

**[R19]** E. Israeli and E. Gilad, “Novel Genetic Algorithm for Loading Pattern Optimization Based on Core Physics Heuristics,” *Annals of Nuclear Energy*, 2018.  
DOI: https://doi.org/10.1016/j.anucene.2018.03.042

**[R20]** S. Ishiguro, T. Endo, and A. Yamamoto, “Loading Pattern Optimization for a PWR Using Multi-Swarm Moth Flame Optimization Method with Predator,” *Journal of Nuclear Science and Technology*, Vol. 57, pp. 523–536, 2020.  
DOI: https://doi.org/10.1080/00223131.2019.1700844

**[R21]** W. Kubiński et al., “Optimization of the Loading Pattern of the PWR Core Using Genetic Algorithms and Multi-Purpose Fitness Function,” *Nukleonika*, Vol. 66, No. 3, 2021.  
DOI: https://doi.org/10.2478/nuka-2021-0022

**[R22]** B. Andersen, J. Hou, J. Godfrey, and D. Kropaczek, “A Novel Method for Controlling Crud Deposition in Nuclear Reactors Using Optimization Algorithms and Deep Neural Network Based Surrogate Models,” *Eng*, Vol. 3, No. 4, pp. 504–522, 2022.  
DOI: https://doi.org/10.3390/eng3040036

**[R23]** T. Evans et al., *Burnable Absorbers in Nuclear Reactors—A Review*, 2022.  
OSTI copy: https://www.osti.gov/servlets/purl/1908247

**[R24]** S. Jeon et al., “An Investigation on the Structural Behavior of the PWR Fuel Assembly,” Transactions of the Korean Nuclear Society, 2007.  
PDF: https://www.kns.org/files/pre_paper/14/322%EC%A0%84%EC%83%81%EC%9C%A4.pdf

**[R25]** S. Jeon et al., “The Effects of Fuel Design on the Fuel Assembly Bow,” Transactions of the Korean Nuclear Society.  
PDF: https://www.kns.org/files/pre_paper/4/391%EC%A0%84%EC%83%81%EC%9C%A4.pdf

**[R26]** A. Zameer, S. M. Mirza, and N. M. Mirza, “Core Loading Pattern Optimization of a Typical Two-Loop 300 MWe PWR Using Simulated Annealing, Novel Crossover Genetic Algorithms and Hybrid GA(SA) Schemes,” *Annals of Nuclear Energy*, Vol. 65, pp. 122–131, 2014.  
DOI: https://doi.org/10.1016/j.anucene.2013.10.024

**[R27]** J. Seurin and K. Shirvan, “Surpassing Legacy Approaches to PWR Core Reload Optimization with Single-Objective Reinforcement Learning,” 2024.  
Preprint: https://arxiv.org/abs/2402.11040

---

# 부록 C. 문헌 활용 시 주의사항

1. **연구용 benchmark의 규칙을 상용 플랜트 규칙으로 오인하지 않는다.**  
   1/8 symmetry, 특정 fresh count, 특정 peaking limit 등은 demonstration model의 설정일 수 있다.

2. **수치 제한은 출처별 사례값이지 보편값이 아니다.**  
   예를 들어 KNFC McFLOP 논문에 나타난 pin burnup·MTC 목표는 해당 APR1400 사례의 최적화 설정이다.[R11]

3. **“Ring of Fire”의 정의와 허용성은 플랜트·문헌마다 다를 수 있다.**  
   어떤 최적화에서는 forbidden pattern이고, 어떤 상용 loading에서는 특정 반경대를 의도적으로 구성하는 설계개념일 수 있다.

4. **최적화 논문은 좋은 탐색법의 근거이지 licensing acceptance의 근거가 아니다.**  
   최종 수락은 승인된 production code와 safety analysis process가 결정한다.

5. **회전 연구는 연료 hardware가 물리적 회전을 허용한다는 보장이 아니다.**  
   orientation freedom은 반드시 연료설계 및 현장취급 규정과 대조해야 한다.

6. **저누설의 효과는 단순히 외곽 assembly power가 낮다는 것으로 입증되지 않는다.**  
   vessel 위치의 energy-dependent fast flux와 fluence를 계산해야 한다.

---

*End of report.*
