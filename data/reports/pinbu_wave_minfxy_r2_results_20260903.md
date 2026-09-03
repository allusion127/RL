# Phase-2 pin-burnup + `F_xy` 결정성 wave — RESULTS (min_fxy `E1_E2`/f121/**ga80** r2, 2026-09-03)

**사전등록** `data/reports/pinbu_wave_minfxy_r2_prereg_20260831.md` (구속력 있음, §10 step 0 STAMP 포함)
**Manifest** `data/reports/pinbu_wave_minfxy_r2_manifest_20260831.json` (`3362F718…02CE`, 30 targets / 25 core)
**Run dir** `D:\lpopt_archive_199\runs\pinbu_wave_minfxy_r2` (199) · **Deck** `pinbu_wave_minfxy_r2_199.inp` · **Box** HOST_199
**Harness** `pinbu_wave.py run` / `patch` (sha256 `5B3688CF…1047` — r1 launcher 가 gate 한 것과 **동일**)
**오프라인 `F_xy` scan** `data/reports/fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv` (30행, 30 sane, `cycle_evidence = final` 30/30)
**상위 사전등록** `data/reports/minfxy_E1E2_f121_r2_prereg_20260831.md` §9.2 (pin 64–72 GWd/tU 예측과 반증 규칙) · §9.3 (3-set 구성)
**사전등록된 대로 채점했다.** 이 판독을 위해 deck·harness·분석 코드를 고치지 않았다. MASTER 추가 호출 없음.

> **한 줄.** 30 chain 전부 수렴·결정적이고, 25개 구별 core **전부**가 이제 `is_deliverable = True` 다.
> 그리고 사전등록 M4 가 **예측한 그대로** — 이 셀의 delivery-grade `F_xy` 기록은 r2 의 라운드 기록
> `6c2243ff`(1.5437)가 아니라 **`F_r` 시대 core `a785eded`(1.5295)** 이다. r2 의 NULL 판정은
> phase-2 로 **DELIVERY 등급에서 확정**되었다.

> **정정 있음 (2026-09-03 발행 후).** §6.3 · §8 · §9 · §10.3 · §10.4 에 7건을 고쳐 적었다.
> **마크 판정(M1–M9)·해시·store 수치는 하나도 바뀌지 않았다.** → §11.

---

## 0. 마크 — 사전등록 §4 대조

| mark | 등록된 예측 | 측정 | 판정 |
|---|---|---|---|
| **M1** `minfxy_r2_top20` 의 pin | **20/20 ≤ 80 AND 20/20 ≤ 78**, 개별값 전부 64–72 | **20/20 ≤ 80, 20/20 ≤ 78**, span **65.104 – 66.894**, 20/20 이 64–72 안 | **PASS** |
| **M2a** between-core \|ΔF_xy\| | 25/25 에서 **≤ 0.002** | **0.000000** (25/25 core, 30/30 행 정확) | **PASS (정확)** |
| **M2b** within-core spread | `6c2243ff` 6회 **spread = 0.000000 (정확)** | `F_xy` 1.5437 ×6, **spread 0.000000**; `F_xya`·pin·cyclen 도 0.000000 | **PASS (정확)** |
| **M3** chain 무결성 / provenance | **30/30 converged, 30/30 `provenance_ok`** | **30/30 converged, 30/30 `provenance_ok`, 30/30 `determinism_ok`**; restart 1종 `native:MAS_RST.APRQ_11_0635.19` | **PASS** |
| **M4** `F_r` frontier vs `F_xy` frontier DELIVERY | **5/5 ≤ 78 · 5/5 deliverable**, 그리고 셀 최선 deliverable `F_xy` 는 **`a785eded` 1.5295** (r2 기록 1.5437 아님) | **5/5 ≤ 78** (64.924 – 65.697) · **5/5 deliverable**; 셀 최선 deliverable `F_xy` = **1.5295 (`a785eded`)** | **PASS (조항 전부)** |
| **M5** ga80/f121 pin-head bias (최초 측정) | bias(pred−meas) **+2.3**, 구간 **[+1.0, +4.0]** | **+2.391**, MAE 2.391, sd 0.796, n = 25, 95% CI **[+2.08, +2.70]** — 구간 안, 점추정과 **0.09 차이** | 보고 — **예측 적중** |
| **M6** scalar 결정성 | 여섯 축 전부 **0.000000 (30/30)** | **0.000000 on 6/6 축, 30/30** | 보고 — 예측대로 |
| **M7** pin/assembly 비율 | ga80 범위 **[1.1297, 1.3256]** 안, 점추정 **1.14–1.17** | **1.14923 – 1.17747** (중앙 1.17130, n=25) — 범위 **안**, 점추정 상단을 **+0.0075 초과** | 보고 — **반증자 미발동**, 점추정 소폭 초과 |
| **M8** `f_xy` head level skill | top20 bias **+0.0128** / MAE 0.0128 / span +0.0024…+0.0309 · backfill bias **+0.0155** / MAE 0.0408 / span −0.0316…+0.0810 | **+0.012784 / 0.012784 / +0.002448…+0.030885** · **+0.015487 / 0.040773 / −0.031560…+0.080988** — **소수 4자리까지 일치** | 보고 — `G2′ MAE < 0.0767` 통과 |
| **M9** `F_xy`/`F_r` 비율 | top20 **1.0207–1.0560 (평균 1.0458)** · backfill **1.0409–1.0961 (평균 1.0713)** | **동일값** (측정−저장 최대 \|Δ\| = 0.000000) | 보고 — M2 의 독립 교차검증 통과 |
| **§9.2 pin 예보** | 개별 pin 전부 **64–72 GWd/tU**; **어느 한 core 라도 > 78 이면 §9.2 는 틀린 것으로 기록** | **25/25 (행 기준 30/30) 이 64–72 안**, 최대 66.894 — **78 초과 0건** | **반증 실패 → §9.2 유지** |

Wave wall **1,823 s (0.51 h)** = pass 1 1,536 s (25 chain) + pass 2 287 s (5 replicate). 사전등록 §5 의
예상 ~0.55 h 안, **등록 상한 1.5 h 의 34%**. per-chain wall 중앙 411 s / 최대 681 s / 최소 182 s
(r1: 중앙 328 s / 최대 596 s) — 이 셀의 cyclen 이 633–639 EFPD 로 r1 의 618–625 보다 길다는 사전등록
§5 의 주석과 방향이 일치한다.

**헤드라인.** r2 최적점 `6c2243ff` 는 pin **66.770 GWd/tU** 로 LEU+ 80 한계에서 **13.23 GWd/tU** 아래에
있고 `unknown_axes = ()` 가 되었다. 이 셀은 phase-2 이전에 `library_id = ga80 & feed = 121` 측정 pin
**0건**이었고 지금 **25건**이다 — **이 wave 가 이 셀의 deliverable 인구 전체를 만들었다** (0 → 25).

**카운터-헤드라인 (M4, 결정 등급).** 셀의 delivery-grade `F_xy` 순위는 **① `a785eded` 1.5295
(`fpcamp_minfr_199`, `F_r` 시대) → ② `deb058c0` 1.5407 (`fpcamp_199`) → ③ `6c2243ff` 1.5437 (r2)** 다.
r2 의 라운드 기록은 자기 셀의 delivery frontier 에서 **3위**이며, 앞의 둘은 **둘 다 `min_fxy` 가 아닌
목적함수가 만든 core** 다. r1 에서 `F_r` 시대 최선(1.5402)이 `min_fxy` 라운드 기록(1.5322)에
0.0080 뒤졌던 구도가, 이 셀에서는 **부호가 뒤집혀** `F_r` 시대가 0.0142 **앞선다**. §5.

**부수적이지만 프로그램 등급.** store 전체의 최선 deliverable `F_xy` 가 **1.5322 (`bf3a70b2`,
`T6_T4`/f121/paramA, r1 최적점) → 1.5295 (`a785eded`, `E1_E2`/f121/ga80)** 으로 움직였다. 프로그램
기록은 갱신되었지만 **그것을 만든 것은 r2 의 탐색이 아니라 이 wave 의 backfill 측정**이다. §6.2.

---

## 1. 실행 무결성 (M3) — **PASS**, 그리고 scalar 결정성 (M6)

| chain | converged | determinism ok | provenance ok | pin 산출 | **usable** |
|---:|---:|---:|---:|---:|---:|
| 30/30 | 30 | 30 | **30** | 30 | **30** (구별 core **25**) |

group 구성은 등록된 대로 **20 / 5 / 5** 이고 `record_id` 구별 수는 **25** 다 (NOTE (rep) 준수).
restart provenance 는 **단일** `native:MAS_RST.APRQ_11_0635.19` 30/30 — 등록된 mix
(`native` 30 / `promoted` 0 / `pair_ecore` 0 / `pair_feed` 0, 구별 restart 파일 1개)와 정확히 일치한다.

사전등록 §7 이 주장한 대로 이것은 **운이 아니라 구조**다: 30 target 전부가 plan time 에
`native:` 로 해소되었고 그 restart 는 `data\produce\promoted` 에서 resolve 되지 않는다. 선례
(`pinbu_wave_fxyera_r1`, 32/40)의 promoted-cache drift 는 **완화된 것이 아니라 부재**하며,
`$wantPromCells = 8` cell-count gate 는 통과했다(사전등록 STAMP, 2026-09-03).

**M6 — scalar 결정성 (측정 − 저장, 30 chain 전부):**

| 축 | n | max \|Δ\| | 허용오차 | r1 선례 |
|---|---:|---:|---:|---:|
| `f_r` | 30 | **0.000000** | 0.002 | 0.000000 |
| `cyclen` | 30 | **0.000000** | 0.5 | 0.000000 |
| `cbc_max` | 30 | **0.000000** | 2.0 | 0.000000 |
| `f_q` | 30 | **0.000000** | — | 0.000000 |
| `ao_abs` | 30 | **0.000000** | — | 0.000000 |
| `max_assembly_burnup` | 30 | **0.000000** | — | 0.000000 |

**여섯 축 전부 bit-exact, 30/30.** M6 의 반증자는 "어느 축이든 non-zero 이면 r1 의 결정성 결론이
paramA/단일 restart **조건부**였다는 뜻"이었다. 발동하지 않았다 — r1 이 paramA/`T6_T4` 에서 얻은
결과가 **ga80/`E1_E2`/f121 로, 다른 라이브러리·다른 셀·다른 restart 파일에서 그대로 재현**된다.
이로써 "MASTER 평형은 인쇄 정밀도까지 재현 가능"이라는 명제에서 라이브러리 조건부 caveat 이 제거된다.

---

## 2. M1 — `F_xy` frontier 20건의 pin 축 → **PASS**

**20건 중 20건이 ≤ 80.0 GWd/tU (100%)**, **20건 중 20건이 ≤ 78.0 (100%)** — 등록된 그대로다.
측정 span **65.104 – 66.894**, 중앙 66.459, 평균 66.248, sd 0.571. 최악 core 에서도 licensing
한계까지 **13.11 GWd/tU**, 획득 gate 78 까지 **11.11 GWd/tU** 여유다.

> **이 wave 를 돌린 이유가 된 숫자: rank 1 `6c2243ff` 의 측정 `max_pin_burnup` = 66.770 GWd/tU.**
> `unknown_axes = ("max_pin_burnup",)` 이 닫혔다. §6.

| rank | record | 저장 `F_xy` | `F_r` | assy BU | pred pin | **meas pin** | pred−meas | ≤80 | ≤78 |
|---:|---|---:|---:|---:|---:|---:|---:|:-:|:-:|
| 1 | `6c2243ffee29` | **1.5437** | 1.4721 | 56.765 | 68.280 | **66.770** | +1.510 | ✔ | ✔ |
| 2 | `2fd846a5f49a` | 1.5505 | 1.4717 | 56.727 | 68.325 | **66.642** | +1.683 | ✔ | ✔ |
| 3 | `0c92d2d714c7` | 1.5520 | 1.4764 | 56.418 | 68.168 | **66.357** | +1.811 | ✔ | ✔ |
| 4 | `9f8e2b278f02` | 1.5546 | 1.4975 | 56.161 | 68.623 | **66.030** | +2.593 | ✔ | ✔ |
| 5 | `8c6efc7ea8b1` | 1.5549 | 1.4931 | 56.944 | 68.318 | **65.696** | +2.622 | ✔ | ✔ |
| 6 | `0766f97c7a5c` | 1.5554 | 1.4781 | 56.211 | 68.629 | **66.157** | +2.472 | ✔ | ✔ |
| 7 | `42ec8c0bc5a8` | 1.5559 | 1.4810 | 56.737 | 68.212 | **66.694** | +1.518 | ✔ | ✔ |
| 8 | `0c5d2aa43600` | 1.5582 | 1.5112 | 56.301 | 68.384 | **65.756** | +2.628 | ✔ | ✔ |
| 9 | `8c4a23c47130` | 1.5588 | 1.4931 | 56.771 | 68.239 | **66.846** | +1.393 | ✔ | ✔ |
| 10 | `624f8d1efcc1` | 1.5603 | 1.4867 | 56.378 | 68.554 | **66.329** | +2.225 | ✔ | ✔ |
| 11 | `b8fa51d8979b` | 1.5619 | 1.5227 | 56.869 | 68.117 | **66.752** | +1.365 | ✔ | ✔ |
| 12 | `0f8ab1ae7336` | 1.5625 | 1.5234 | 56.834 | 68.234 | **66.737** | +1.497 | ✔ | ✔ |
| 13 | `7fc80a514cf8` | 1.5627 | 1.4821 | 56.566 | 68.667 | **65.412** | +3.255 | ✔ | ✔ |
| 14 | `b5941ef7c3d8` | 1.5640 | 1.4833 | 56.570 | 68.209 | **66.609** | +1.600 | ✔ | ✔ |
| 15 | `47713a26ef82` | 1.5649 | 1.4824 | 56.590 | 68.701 | **65.427** | +3.274 | ✔ | ✔ |
| 16 | `c5077f8c2d6b` | 1.5654 | 1.4855 | **57.621** | 68.566 | **66.561** | +2.005 | ✔ | ✔ |
| 17 | `d315aa29113a` | 1.5657 | 1.4834 | 56.643 | 68.663 | **65.493** | +3.170 | ✔ | ✔ |
| 18 | `1a4e702d1b2c` | 1.5661 | 1.5344 | 56.936 | 68.206 | **66.701** | +1.505 | ✔ | ✔ |
| 19 | `00b3d5b07597` | 1.5664 | 1.4834 | 56.325 | 68.656 | **65.104** | +3.552 | ✔ | ✔ |
| 20 | `b080c743a489` | 1.5669 | 1.4861 | 57.111 | 68.662 | **66.894** | +1.768 | ✔ | ✔ |

등록된 근거는 모든 절에서 성립했다. 사전등록 M1 은 **모델이 아니라 비율 추정자**에 근거를 걸었고
(중앙 비율 1.1694 → 65.68–67.38), 측정값 65.104–66.894 는 그 구간과 **거의 그대로 겹친다**.
s1j head 의 68.12–68.82 예측은 전건 과대(잔차 0/20 이 음수)였다. 즉 **이 셀에서도 pin 은 살아 있는
제약이 아니라 자문 열**이며, `minfxy_pin_bu_limit = 78` 획득 gate 는 이 셀에서 아무 비용도 물리지
않는다 — r1 §2.1 의 feed-121 결론이 **라이브러리를 바꿔도** 유지된다.

### 2.1 이 셀에서 pin 은 무엇과 함께 움직이는가

측정된 25 core 기준 상관: **assembly burnup +0.572**, `F_r` **+0.462**, cyclen +0.201,
`F_xy` **−0.098**. top20 안에서만 보면 assembly +0.459, `F_xy` −0.229.

r1(paramA/`T6_T4`)은 assembly +0.752, `F_xy` −0.606, `F_r` **−0.375** 였다. 여기서 달라진 것은
**`F_r` 상관의 부호**다(−0.375 → +0.462). `F_xy` 와 pin 사이에 유의미한 관계가 없다는 결론
("낮은 `F_xy` 는 pin 여유를 사지도 팔지도 않는다")은 두 셀에서 공통이지만, **pin 을 `F_r` 로
예보하려는 시도는 셀을 넘어 전이되지 않는다.** 어느 쪽이든 25 core 의 pin 전 범위가 1.97 GWd/tU
(64.924–66.894)뿐이라 이 상관들은 **좁은 창 안의 기울기**이고, 판정에는 쓰이지 않았다.

---

## 3. M2 — `F_xy` 결정성 → **PASS (두 층 모두 정확)**

30 chain 전부가 저장 `f_r`/`cyclen`/`cbc_max` 를 허용오차(0.002 / 0.5 / 2.0) 안에서 — 실제로는
0.000000 으로 — 재현하고 `native:` provenance 를 유지하므로 **30행 전부가 M2 의 범위 안**이다.
보존된 최종 cycle `MAS_OUT` 에서 `FXYP` 를 읽고 `digest16 = sha256(pack_pattern)[:16]` 로 join:

**(a) between-core: `|F_xy_replay − F_xy_stored| = 0.000000` — 구별 core 25/25, 행 30/30.**
등록 기준 ≤ 0.002 대비 최대값 **0.000000**. join 은 30/30 매칭 / 0 unmatched / 25 distinct digest.
`F_xya` 동반 축도 30/30 정확 (1.3730 – 1.4300).

**(b) within-core: `6c2243ff` 6회 측정의 spread = 0.000000 (정확).**

| 반복 측정량 | 6회 값 | spread (max−min) |
|---|---|---:|
| `F_xy` | 1.5437 ×6 | **0.000000** |
| `F_xya` | 1.3883 ×6 | **0.000000** |
| `max_pin_burnup` | 66.770 ×6 | **0.000000** |
| `cyclen` (`efpd_max`) | 637.277 ×6 | **0.000000** |
| `max_assembly_burnup` | 56.765 ×6 | **0.000000** |
| `f_r` | 1.4721 ×6 | **0.000000** |

이것이 D1 이 막은 `post_verify_top_k` 재검증의 **등록된 대체물**이며, 등록된 대로 top 5 가 아니라
**구별 core 25건 + 기록 core 6회**를 덮는다. **r2 결과 문서의 모든 `F_xy` 숫자는 replay-exact 이고,
1.5437 → 1.65 여유(0.1063)는 replay noise 로 넓힐 필요가 없다.** 선례 누계는 이제
`fxyera_r1` 40 + `minfxy_r1` 20 + 이번 30 = **90/90 정확**이다.

**독립 교차검증 두 가지.** (i) scan 의 `efpd_max` vs 저장 `cyclen` 최대 \|Δ\| = **0.000000**
(30행) — `backfill_fxy` 의 cycle-tolerance guard 가 잡을 것이 없었고, join 이 chain 을 섞지
않았음을 보인다. (ii) **M9** — 저장 기준 `F_xy`/`F_r` 비율(top20 1.0207–1.0560 평균 1.0458,
backfill 1.0409–1.0961 평균 1.0713)이 **측정 기준과 최대 \|Δ\| 0.000000 으로 동일**하다.
M9 는 새 측정이 아니라 M2 의 독립 교차검증이며, 어긋났다면 오프라인 `FXYP` scanner 의 join 이
틀렸다는 신호였을 것이다. 어긋나지 않았다.

또한 `backfill_fxy apply` 는 등록된 대로 **no-op** 이었다: `already filled 25 / populated 0`
(dry-run only, §7). `populated > 0` 이 join 오류의 신호였고 나오지 않았다.

---

## 4. M5 / M7 / M8 — head skill 과 비율 추정자

### 4.1 M5 — ga80/f121 pin-head bias, **이 셀 최초 측정** → 등록 구간 적중

| slice | n | bias (pred−meas) | MAE | sd | 95% CI |
|---|---:|---:|---:|---:|---|
| **pooled (구별 core)** | **25** | **+2.391** | 2.391 | 0.796 | **[+2.08, +2.70]** |
| pooled (전 30행) | 30 | +2.245 | 2.245 | 0.797 | [+1.96, +2.53] |
| `minfxy_r2_top20` (delivery) | 20 | +2.172 | 2.172 | 0.721 | [+1.86, +2.49] |
| `frontier_fr_backfill5` (calibration) | 5 | +3.267 | 3.267 | 0.361 | [+2.95, +3.58] |
| `record_replicate5` (동일 core 5회) | 5 | +1.510 | 1.510 | **0.000** | — |
| *등록된 예측* | — | ***+2.3*** | — | — | ***[+1.0, +4.0]*** |
| *r1, paramA/`T6_T4`/f121* | *25* | *+3.420* | *3.42* | *0.485* | *[+3.23, +3.61]* |
| *선례, feed 109* | — | *−2.65* | — | — | — |

**등록 구간 [+1.0, +4.0] 안이고, 점추정 +2.3 과 0.09 차이다.** 사전등록 §4 가 요구한 필수 판독
항목 — "측정값이 r1 의 paramA/f121 `+3.42` 와 선례의 f109 `−2.65` 중 어느 쪽에 가까운지" — 의 답:
**+3.42 쪽이 압도적으로 가깝다** (거리 1.03 vs 5.04). **부호는 r1 과 같고 크기는 30% 작다.**

여기서 결정적인 사실 하나. **이 셀의 CI [+2.08, +2.70] 과 r1 의 CI [+3.23, +3.61] 은 서로소다.**
즉 r2 사전등록 §9.2 가 "r1 의 `+3.42` 를 이 셀로 전이하지 말라"고 금지한 것은 **사후적으로도
옳았다** — 전이했다면 1.0 GWd/tU 만큼 틀렸을 것이다. 반면 **셀 자신의 assembly BU 와 ga80 비율
분포만으로 유도한 등록 예측 +2.3 은 맞았다.** 전이되는 것은 bias 가 아니라 **비율**이라는 §9.2 의
등록 명제가 이 라운드에서 처음으로 실측 지지를 얻었다.

`MAE = |bias|` 가 여기서도 정확히 성립한다: **25건 중 음수 잔차 0건**. head 는 이 셀에서도 노이즈가
아니라 **일률적으로 보수적**이다. 잔차 범위 +1.365 … +3.805.

> **등록된 대로 무시한다:** `frontier_fr_backfill5` 의 캘리브레이션 회귀. 예측 pin span 이
> 0.68 GWd/tU (68.14–68.82)뿐이라 fit 은 **degenerate** 이며, 사전등록 §2.2 와 manifest 의
> `degenerate_curve_warning` 이 사전에 그렇게 지정했다. r1 의 대응 fit 도 같은 이유로 폐기되었다.

### 4.2 모델-무관 비율 추정자가 head 를 다시 이겼다

ga80 중앙 비율 1.1694 × 측정 assembly BU 를 pin 예측으로 쓰면:

| 추정자 | n | bias | MAE |
|---|---:|---:|---:|
| **비율 (1.1694 × assy)** | 30 | **+0.044** | **0.465** |
| s1j head (원본) | 30 | +2.245 | 2.245 |
| **비율 (구별 core)** | 25 | — | **0.480** |
| s1j head (구별 core) | 25 | — | 2.391 |

**비율 추정자가 학습된 head 를 4.8–5.0배 이겼고, 실질적으로 무편향이다.** r1 §5.1 이 `F_xy`
축에서 발견한 것과 **같은 결론이 pin 축에서 재현**된다. 사전등록은 비율을 M1 의 근거로만 썼는데,
측정 결과 그것이 **더 나은 예측기**였다.

### 4.3 M7 — pin / assembly-burnup 비율, 이 셀 최초 측정

| slice | n | 측정 비율 |
|---|---:|---|
| 전 30행 | 30 | 1.14923 – 1.17747 (중앙 1.17401) |
| 구별 core | 25 | 1.14923 – 1.17747 (중앙 **1.17130**, 평균 1.16710) |
| `minfxy_r2_top20` | 20 | 1.15369 – 1.17747 |
| `frontier_fr_backfill5` | 5 | 1.14923 – 1.17126 |
| *등록: ga80 store 범위* | *1,799* | *[1.1297, 1.3256]* |
| *등록: 점추정* | — | *1.14 – 1.17* |

**반증자(범위 밖)는 발동하지 않았다** — 25/25 가 [1.1297, 1.3256] 안에 있고, 실제로는 ga80
중앙값 1.1694 를 **감싸는** 매우 좁은 띠(폭 0.028)다. 다만 **점추정 상단 1.17 을 최대값
1.17747 이 +0.0075 초과**한다. 이것은 등록된 반증 조건이 아니므로 M7 은 유지되지만, 이 초과는
정직하게 기록한다: r1 이 paramA/`T6_T4` 에서 측정한 1.1412–1.1549 보다 **이 셀의 비율이 계통적으로
높다**(중앙 1.1713 vs 1.148). 비율은 전이되지만 **셀마다 ±0.02 수준으로 이동**한다는 뜻이며,
M1 의 근거가 무효가 되지는 않는다(1.3256 최악 비율을 써도 이 셀 assembly BU 로는 74.4–76.4 이므로
78 조차 넘지 않는다는 사전등록의 여유 논증이 그대로 남는다).

### 4.4 M8 — `f_xy` head level skill (이 셀)

M2 가 통과했으므로 측정 = 저장이고, manifest 에 고정된 잔차(pred − stored)가 **그대로 성립**한다:

| group | n | bias | MAE | median | sd | span | 양수 |
|---|---:|---:|---:|---:|---:|---|---:|
| `minfxy_r2_top20` | 20 | **+0.012784** | 0.012784 | +0.010783 | 0.007944 | +0.002448 … +0.030885 | **20/20** |
| `frontier_fr_backfill5` | 5 | **+0.015487** | 0.040773 | −0.001514 | 0.052025 | −0.031560 … +0.080988 | 2/5 |

**사전등록 §4 의 M8 표(+0.0128 / 0.0128 / +0.0024…+0.0309 · +0.0155 / 0.0408 / −0.0316…+0.0810)와
소수 4자리까지 일치한다.** 두 group 모두 s1j 승격 bar `G2′ MAE < 0.0767` 를 통과한다 —
level skill 은 실재한다.

**ranking skill 은 없다.** top20 20건에 대한 Spearman ρ(예측 `F_xy`, 측정 `F_xy`) = **+0.113**
(p = 0.636). r1 은 같은 통계가 **−0.114** (p = 0.63)이었다. **두 셀·두 라이브러리에서 부호만 뒤집힌
0 근방**이라는 것이 결론이다 — head 는 이 셀이 *어디쯤*인지는 알지만 *셀 안에서 어느 core 가 더
나은지*는 모른다. 이것은 r2 캠페인 결과 §6.4 의 RANK gate 3/3 FAIL 과 **독립 경로로 같은 결론**이며,
"r3 의 랭커는 s1i(=`F_r`/비율 랭킹)로 돌린다"는 그 문서의 처분을 뒷받침한다.

또한 r1 에서는 `F_xy`-era group 이 −0.0075, `F_r`-era group 이 +0.0420 으로 **부호가 뒤집혔던**
반면, 여기서는 두 group 다 양수(+0.0128 / +0.0155)이고 head 는 균일하게 과대예측한다 —
사전등록 M8 이 미리 지목한 대조가 그대로 확인되었다.

---

## 5. M4 — `F_r` frontier vs `F_xy` frontier 의 DELIVERY 비교 → **PASS (모든 조항)**

`frontier_fr_backfill5` 5건의 측정 pin 과 6축 `is_deliverable`:

| # | record | campaign | `F_r` | `F_xy` | assy BU | pred pin | **meas pin** | pred−meas | ≤78 | deliverable |
|---:|---|---|---:|---:|---:|---:|---:|---:|:-:|:-:|
| 1 | `deb058c00433` | `fpcamp_199` | **1.4636** | 1.5407 | 56.097 | 68.141 | **64.924** | +3.217 | ✔ | ✔ |
| 2 | `6d1081b285be` | `fpcamp_minfr_199` | 1.4648 | 1.6056 | 56.091 | 68.543 | **65.697** | +2.846 | ✔ | ✔ |
| 3 | `0a4f76b51578` | `fpcamp_minfr2_199` | 1.4670 | 1.5763 | 56.369 | 68.576 | **65.185** | +3.391 | ✔ | ✔ |
| 4 | `11f18f3d18ce` | `fpcamp_minfr_199` | 1.4681 | 1.6039 | 56.121 | 68.418 | **65.341** | +3.077 | ✔ | ✔ |
| 5 | `a785ededf928` | `fpcamp_minfr_199` | 1.4694 | **1.5295** | 56.572 | 68.819 | **65.014** | +3.805 | ✔ | ✔ |

**5/5 가 ≤ 78 이고 5/5 가 deliverable** — 등록된 대로다. 그리고 등록된 **결정적 조항**:

> **셀의 최선 deliverable `F_xy` 는 r2 의 기록 `6c2243ff`(1.5437)가 아니라 backfill rank 5 인
> `a785eded`(1.5295)이다.** → **측정으로 확인됨.**

사전등록 M4 의 반증 경로 — "`a785eded` 가 pin > 80 이고 `6c2243ff` 가 통과하는 경우" — 는
**발동하지 않았다.** `a785eded` 는 pin 65.014 로 licensing 한계에서 **14.99 GWd/tU**,
획득 gate 78 에서 **12.99 GWd/tU** 떨어져 있다. 즉 **r2 가 이 셀의 기록을 가져갈 유일한 경로는
닫혔다.**

### 5.1 셀의 delivery-grade `F_xy` 순위 — r2 기록은 3위다

phase-2 이후 이 셀(`E1_E2`/f121/ga80)의 deliverable core 는 **25건**이고, `F_xy` 오름차순 상위:

| 순위 | record | `F_xy` | `F_r` | pin | campaign | 목적함수 시대 |
|---:|---|---:|---:|---:|---|---|
| **1** | `a785ededf928` | **1.5295** | 1.4694 | 65.014 | `fpcamp_minfr_199` | **`F_r`** |
| **2** | `deb058c00433` | **1.5407** | 1.4636 | 64.924 | `fpcamp_199` | **`F_r`** |
| **3** | `6c2243ffee29` | 1.5437 | 1.4721 | 66.770 | `fpcamp_minfxy_e1e2_f121_r2` | `F_xy` (r2 기록) |
| 4 | `2fd846a5f49a` | 1.5505 | 1.4717 | 66.642 | 〃 | `F_xy` |
| 5 | `0c92d2d714c7` | 1.5520 | 1.4764 | 66.357 | 〃 | `F_xy` |

**r2 의 100 call 은 이 셀의 delivery frontier 를 1 tick 도 움직이지 못했다.** 앞선 두 core 는
**둘 다 `min_fxy` 가 아닌 목적함수**가 만들었고, 둘 다 pin 도 더 좋다(64.92 / 65.01 vs 66.77).

r1 과의 대조가 이 라운드의 핵심 학습이다:

| 셀 | `min_fxy` 라운드 기록 | 그 셀 `F_r` 시대 최선 `F_xy` | 차이 |
|---|---:|---:|---:|
| `T6_T4`/f121/paramA (r1) | **1.5322** | 1.5402 (`4d70ab6f`) | `min_fxy` **+0.0080 우세** |
| `E1_E2`/f121/ga80 (r2) | 1.5437 | **1.5295** (`a785eded`) | `min_fxy` **−0.0142 열세** |

**`min_fxy` 목적함수의 한계 가치는 셀 의존적이고, 이 셀에서는 음수다.** r1 결과 §8 이 "열린 것"
1번으로 남긴 질문("`min_fr` 이 이미 dedicated `min_fxy` 라운드의 0.008 안에 드는 셀은 r2 를 쓸 곳으로
약하다")에 대해, 이 셀은 **그보다 강한 답**을 준다 — `min_fr` 이 이미 **앞서 있었다.**

### 5.2 프로그램 전체 delivery 기록이 움직였다 (그러나 탐색이 만든 것이 아니다)

| 시점 | store 전체 최선 deliverable `F_xy` | core | 셀 |
|---|---:|---|---|
| phase-2 직전 | 1.5322 | `bf3a70b20e50` | `T6_T4`/f121/paramA (r1 최적점) |
| **phase-2 이후** | **1.5295** | **`a785ededf928`** | **`E1_E2`/f121/ga80** |

**−0.0027.** 이 갱신은 **새 core 를 만들어서가 아니라 이미 store 에 있던 `F_r` 시대 core 의 pin 을
측정해서** 일어났다. 사전등록 §12 가 "이 라운드는 새 core 를 만들지 않는다"고 못 박은 그 성질이,
역설적으로 프로그램 기록을 갱신한 경로다. **탐색 예산 0 call 로 얻은 기록이다.**

---

## 6. DELIVERY 판정 — 25 core 전부 **DELIVERABLE**

`is_deliverable`(`lpopt/search/campaign.py:567`)은 gate 된 licensing 축 **전부가 측정**되고 한계
안일 것을 요구한다. 판정에 쓴 limits 는 r2 캠페인 deck 자신의 것을 lpopt 코드로 해소한 값이다
(`load_config('fpcamp_minfxy_E1E2_f121_r2_199.inp')` → `feasibility_limits_for(..., 'min_fxy')`
→ `deliverable_limits`): `f_xy ≤ 1.65`, `f_r ≤ 1.55`, `cbc_max ≤ 1600`, `f_q ≤ 2.41`,
`|AO| ≤ 0.30`, `max_pin_burnup ≤ 80.0` (licensing 80, 획득용 78 haircut 아님),
**`cyclen_lo`/`cyclen_hi` = None → cyclen 은 gate 되지 않는다** (r1 결과 §6 FOOTNOTE 과 동일한 구조;
아래 표의 cyclen 행은 **"기록"이지 "통과"가 아니다**).

### 6.1 r2 최적점 `6c2243ffee29…`

| 축 | 한계 | 측정 | 여유 | 판정 |
|---|---:|---:|---:|:-:|
| `f_xy` | ≤ 1.65 | **1.5437** | 0.1063 | ✔ |
| `f_r` | ≤ 1.55 | **1.4721** | 0.0779 | ✔ |
| `cbc_max` | ≤ 1600 | **1322.90** | 277.10 | ✔ |
| `f_q` | ≤ 2.41 | **1.8350** | 0.5750 | ✔ |
| \|AO\| | ≤ 0.30 | **0.0405** | 0.2595 | ✔ |
| `max_pin_burnup` | ≤ 80.0 | **66.770** | **13.230** | ✔ |
| `cyclen` | (gate 없음) | 637.277 | — | 기록 |

**`unknown_axes = ()` · `is_deliverable = True`.** r2 캠페인 결과 §0 의
"PIN / PRIMARY (delivery): 미판정" 행이 이로써 **판정됨**으로 바뀐다.

### 6.2 셀 incumbent `a785ededf928…` — 이 셀의 delivery 상한

| 축 | 한계 | 측정 | 여유 | 판정 |
|---|---:|---:|---:|:-:|
| `f_xy` | ≤ 1.65 | **1.5295** | 0.1205 | ✔ |
| `f_r` | ≤ 1.55 | **1.4694** | 0.0806 | ✔ |
| `cbc_max` | ≤ 1600 | **1330.81** | 269.19 | ✔ |
| `f_q` | ≤ 2.41 | **1.8422** | 0.5678 | ✔ |
| \|AO\| | ≤ 0.30 | **0.0404** | 0.2596 | ✔ |
| `max_pin_burnup` | ≤ 80.0 | **65.014** | **14.986** | ✔ |
| `cyclen` | (gate 없음) | 638.639 | — | 기록 |

**`unknown_axes = ()` · `is_deliverable = True`** — M4 가 예측한 대로 pin gate 에서 떨어지지 않는다.

### 6.3 인구 조사 (25 core, HOST_238 에서 lpopt 코드로 산출)

| set | n | deliverable | `unknown_axes` 보유 | 구속 축 |
|---|---:|---:|---:|---|
| `minfxy_r2_top20` | 20 | **20/20 (100%)** | 0 | `F_r` (최악 1.5344, 여유 0.0156) |
| `frontier_fr_backfill5` | 5 | **5/5 (100%)** | 0 | `F_xy` (최악 1.6056, 여유 0.0444) · `F_r` 0.0806 |  <!-- 정정 2026-09-03 -->
| **wave 합계 (구별 core)** | **25** | **25/25** | **0** | — |

25 core 전체의 축별 최악 여유:

| 축 | 최악값 | 여유 |
|---|---:|---:|
| **`f_r`** | **1.5344** | **0.0156** ← 최소 |
| `f_xy` | 1.6056 | **0.0444** ← 0.1 미만 (2위) |
| `f_q` | 1.9188 | 0.4912 |
| \|AO\| | 0.0405 | 0.2595 |
| `cbc_max` | 1336.55 | 263.45 |
| `max_pin_burnup` | 66.894 | **13.106** |

**pin·`F_q`·CBC·AO 로 제한되는 core 는 25건 중 0건이다.** 0.1 미만 여유를 갖는 축은 **둘**이다:
`F_r`(top20 0.0156 · backfill5 0.0806)과 `F_xy`(top20 0.0831 · backfill5 0.0444). 가장 좁은 것은
`F_r` 0.0156(`1a4e702d`, top20)이고, `F_xy` 최악 0.0444 는 `min_fxy` 가 만들지 않은 backfill core
`6d1081b2`(1.6056)다. 선례 r1 결과 §6 의 "0.1 미만은 `F_r` 뿐" 서술도 같은 이유로 부정확하다
(그 wave 의 deliverable `F_xy` 최악 여유 0.061). — *정정 2026-09-03, §11 참조*

### 6.4 phase-2 이후의 delivery-grade frontier

| 범위 | 병합 전 | **병합 후** |
|---|---:|---:|
| `E1_E2`/f121/ga80 (이 셀) — 이중 측정행 | 0 | **25** |
| `E1_E2`/f121/ga80 — deliverable | **0** | **25** |
| 셀 최선 deliverable `F_xy` | — (없음) | **1.5295** (`a785eded`) |
| 셀 deliverable `F_xy` span | — | 1.5295 – 1.6056 |
| 셀 deliverable pin span | — | 64.924 – 66.894 |
| store 전체 — 이중 측정행 | 106 | **131** |
| **store 전체 — deliverable** | **25** | **50** |
| store 최선 deliverable `F_xy` | 1.5322 (`bf3a70b2`) | **1.5295** (`a785eded`) |
| store deliverable 이 존재하는 셀 | 1 (`T6_T4`/f121/paramA) | **2** (+ `E1_E2`/f121/ga80) |

store 전체 deliverable 50행의 구성: `T6_T4`/f121/paramA **25** (r1 wave 가 만든 것) +
`E1_E2`/f121/ga80 **25** (이 wave). campaign 별로는 `fpcamp_minfxy_t6t4_f121_r1` 20 ·
`fpcamp_minfxy_e1e2_f121_r2` 20 · `batchswap_enum_T6T4` 4 · `fpcamp_minfr_199` 3 ·
`fpcamp_minfr2_199` 1 · `fpcamp_minfr_T6T4_r8` 1 · `fpcamp_199` 1.
**프로그램의 deliverable 인구는 전부 두 번의 phase-2 wave 가 만들었다.**

---

## 7. store 병합

**쓰기 전 백업.** `E:/lpopt_data/5_RL/backups/records.parquet.bak_pre_pinbu_minfxy_r2_20260903`
(22,814,014 B, sha256 `22854B72…C329D` — 병합 전 store 와 byte 동일) 및
`…/ledger.jsonl.bak_pre_pinbu_minfxy_r2_20260903` (19,623,488 B, `28675801…D8AE`).
`maps.npz` 는 patch 가 건드리지 않으므로 백업하지 않았다(사전등록 §3.1 의 "정직한 한계").
patch 자신이 쓴 in-store 백업 `data/store/records.parquet.bak_pre_pinbu_minfxy_r2_20260903` 도 동일 sha.

**Step 1 — `pinbu_wave.py patch` (`max_pin_burnup`, tag `pinbu_minfxy_r2_20260903`)**

```
[patch] 30 result row(s): 25 accepted, 0 refused
[patch] would write max_pin_burnup on 25 row(s) (25 also carry max_assembly_burnup)
[patch] wrote 25 measured pin value(s); store now 76793 row(s)
```

dry-run 과 실제 실행이 정확히 일치. **30행 → 25 write, +0 store 행.**

> **NOTE (rep) 의 등록된 위험은 무해하게 해소되었다.** 사전등록 §10 은 "`patch` 는
> `accept[record_id]` 에 **마지막 값**을 쓰므로 `6c2243ff` 의 6행 중 마지막 행의 pin 이 들어간다;
> 6개가 서로 다르면 병합 전에 멈추라"고 등록했다. `cmd_patch` 는 store 를 건드리기 전에
> `accept[record_id]` 로 접으므로(`pinbu_wave.py:654`) 6행은 **1개 항목**이 되고 write 는
> `record_id` 로 in-place 라 **append 가 불가능**하다. 그리고 M2b 가 보인 대로 6개 측정이 전부
> **66.770 으로 동일**하므로 "마지막 행"이 무엇이든 쓰이는 값은 유일하다. 멈출 이유가 없었다.

**Step 2 — `python -m lpopt.tools.backfill_fxy apply` (등록된 no-op, dry-run 만 수행)**

```
final & sane 30 -> 25 digest(s) | dup 0 | no store row 0 | ambiguous 0
cycle mismatch 0 | F_r<=F_xy<=F_q violations 0 | already filled 25
populated 0  -> nothing to populate; store untouched
```

사전등록 §10 의 기대(`already filled 25 / populated 0`)와 정확히 일치. 30 target 전부가 이미
`f_xy` 라벨을 갖고 있으므로 이 단계는 **join 검증용**이며, 실제 쓰기는 하지 않았다.
**이 harvest 가 store 에 가한 쓰기는 pin patch 단 하나다.**

**병합 전후 store**

| 항목 | before | **after** |
|---|---:|---:|
| 행 | 76,793 | **76,793** |
| `f_xy` non-null | 7,766 | **7,766** (+0) |
| `max_pin_burnup` non-null | 40,870 | **40,895** (+25) |
| 두 축 모두 측정 | 106 | **131** (+25) |
| `library_id = ga80 & feed = 121` 측정 pin | **0** | **25** |
| `E1_E2`/f121/ga80 측정 pin | **0** | **25** |
| **joint-clean + deliverable (6축)** | **25** | **50** |
| bytes | 22,814,014 | **22,810,322** (**−3,692**) |
| sha256 | `22854B72A4966935550FD322DA29FCAB58FBFA19FDBB84124DF444791B9C329D` | **`16E311AF4465E735B38DAF7ABF999268FAC27946C1C5CC279114607D9EE917BA`** |

전후 parquet 을 `record_id` 로 join 한 직접 대조: **null → 값 전환 25행, 기존 값 덮어쓰기 0행,
소실 행 0.** 바이트 수가 **줄면서** 값이 늘어난 것은 parquet 재인코딩 효과이며 행 손실이 아니다
(§9-1).

---

## 8. 이 wave 가 확정한 것, 그리고 연 것

**확정.**
1. **r2 최적점은 deliverable 이다** — pin 66.770, 여유 13.23 GWd/tU, `unknown_axes = ()`.
   `lpopt/search/verify.py:949` 의 `enable_pin_burnup=False` 사각지대가 이 라운드 frontier 에 대해 닫혔다.
2. **r2 의 `F_xy` 라벨은 replay noise 가 0 이다** (25 core / 30행 정확). r2 결과 문서의 어떤
   `F_xy` 여유도 넓힐 필요가 없다. 누계 90/90.
3. **MASTER 평형의 bit-exact 재현성은 라이브러리 조건부가 아니다** — paramA/`T6_T4` 에서 얻은
   여섯 축 0.000000 이 ga80/`E1_E2` 에서 그대로 재현되었고, within-core 6회 반복도 정확히 0 이다.
4. **이 셀 pin 은 살아 있는 제약이 아니다.** 25 core 가 64.92–66.89 로 78 gate 에서 11 GWd/tU 이상
   떨어져 있다. `minfxy_pin_bu_limit = 78` 은 이 셀에서 비용 0 이다.
5. **`min_fxy` r2 는 이 셀에서 아무 기록도 만들지 못했다 — DELIVERY 등급에서 확정.** 셀의 delivery
   frontier 1·2위가 모두 `F_r` 시대 core 다. r2 캠페인 결과 §8 의 NULL 귀속("셀 바닥")은
   phase-2 로 반박되지 않았고 오히려 강화되었다.
6. **비율 추정자 > 학습 head**, pin 축에서도. 무편향(+0.044) MAE 0.465 vs head 2.245.

**열린 것.**
1. **`min_fxy` 의 한계 가치는 셀 의존적이며 부호가 바뀐다** (`T6_T4` +0.0080, `E1_E2` −0.0142).
   r3 의 셀 선택은 "`min_fr` 이 이미 얼마나 좋은 `F_xy` 를 갖고 있는가"를 **먼저 측정한 뒤**
   결정해야 한다. 이 측정은 라벨만 있으면 되고 MASTER call 이 필요 없다.
2. **pin-head bias 는 셀마다 다르다 — CI 가 서로소다** (paramA/`T6_T4` [+3.23,+3.61] vs
   ga80/`E1_E2` [+2.08,+2.70], f109 −2.65). `pinbu_physics`(`lpopt/model/pinbu_physics.py::
   fit_pinbu_physics`)는 여전히 어떤 champion 에도 적합되어 있지 않다. **이제 두 라이브러리·세 셀의
   라벨이 있으므로 library 별 적합이 가능하다** — 다음 champion 전에 권고.
3. **pin/assembly 비율도 셀마다 ~0.02 이동한다** (1.148 → 1.171). 비율 추정자는 여전히 최선이지만,
   "ga80 중앙값 하나"가 아니라 **셀 조건부 비율**로 쓰는 편이 낫다. M7 의 점추정 초과가 그 신호다.
4. **head 의 ranking skill 은 두 셀 모두 0 근방이다** (ρ −0.114 / +0.113). r2 캠페인 §6.4 의
   RANK gate 3/3 FAIL 과 합쳐, r3 랭커를 s1i 로 되돌리는 결정의 근거가 **두 개의 독립 경로**로
   확보되었다.
5. **이 인구에서 0.1 미만 여유를 갖는 축은 `F_r`(최악 0.0156)과 `F_xy`(최악 0.0444) 둘이다.**
   다만 `min_fxy` 가 만든 top20 안에서는 `F_r`(0.0156)이 `F_xy`(0.0831)보다 5배 좁다. `min_fxy` 를 계속 밀 경우
   `F_r ≤ 1.55` 가 실제 binding constraint 가 될 가능성이 `F_xy` 나 pin 보다 높다.
6. **kit store 동기화 부채** — §9-3, §10.3.

---

## 9. 편차 (deviations) — 사전등록에 없던 것

1. **store 가 3,692 B 줄면서 값 25개가 늘었다** (22,814,014 → 22,810,322). 행 수는 76,793 로
   불변이고 25 target 전부 pin 을 되읽으므로 **parquet 재인코딩**이지 손실이 아니다
   (`max_pin_burnup` 이 해당 row group 에서 조밀해졌다). r1 병합은 크기가 **+382 B** 로 움직였기
   때문에, 크기만 보는 점검은 이것을 회귀로 오독할 수 있다. 전후 join 으로 직접 확인했다:
   null→값 25행 / 덮어쓰기 0 / 소실 0.
2. **harvest 와 archive 가 같은 run tree 를 이중 보관한다.** 과제가 두 경로를 모두 목적지로
   지정했으므로 `E:/lpopt_data/5_RL/harvest/pinbu_minfxy_r2/` 와
   `E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/pinbu_wave_minfxy_r2/` 가 각각
   archive 사본은 **356 파일 / 2,665,574,497 B**(199 원격 `Get-ChildItem -Recurse -File` 계측과
   일치)이고, harvest 사본은 그 **상위집합인 360 파일 / 2,665,589,389 B** 다 — 같은 run tree 356 파일에
   더해 최상위에 `fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv`(5,835 B) ·
   `pinbu_wave_minfxy_r2_out.log`(3,743 B) · `pinbu_wave_minfxy_r2_rc.txt`(6 B) ·
   `status_pinbu_wave_minfxy_r2_199_20260903.txt`(5,308 B) 4개 / 14,892 B 가 함께 있다
   (**byte 동일 아님**; 총 5.3 GB, 볼륨 여유 5.9 TB). 그 4개는 각각 `data/reports/` 와
   `runs/pinbu_wave_minfxy_r2/` 에도 있으므로 harvest 사본은 그대로 prunable 이다. 네트워크 pull 은
   1회였고 harvest 사본은 archive 에서 로컬 복사했다. r1 선례는 archive 사본만 두었다 —
   **prunable.**
3. **199 의 kit store 가 stale 해졌다.** arming 시점에 성립하던 "로컬 store == kit store"
   불변식(둘 다 `22854B72…` / 22,814,014 B)이 이 병합으로 깨졌다. patch 가 로컬에서 돌았기
   때문이며, 다음 199 launcher 는 `store sha256 mismatch` 로 **거부**한다. 이것은 결함이 아니라
   의도된 fail-closed 동작이다. 조치: 패치된 parquet 을
   `C:\Users\USER\lpopt_work\kit_frontier\data\store\` 로 재 scp + §10.3 의 `$wantSha`/`$wantLen`
   재stamp.
4. **199 에 남긴 쓰기 1건.** `F_xy` scan 출력
   `C:/Users/USER/lpopt_work/kit_frontier/data/reports/fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv`
   (5,835 B). 그 외 199 에서 삭제·수정된 것은 없다(상시 규칙). r1 과 같은 형태다.
5. **pass-2 replicate 위험은 현실화되지 않았다.** 사전등록 §3.2 는 `.bat` 재호출 시 pass 2 가
   5건을 **다시** 돌아 replicate 행이 5 초과가 될 수 있다고 등록했다. JSONL 은 정확히 30행 /
   25 distinct / 20+5+5 이므로 run 은 한 번에 깨끗이 끝났고 제거할 행이 없었다.
6. **`backfill_fxy apply` 는 dry-run 만 수행했다.** 등록된 기대가 no-op(`populated 0`)이었고
   실제로 no-op 이었으므로, store 에 불필요한 재작성을 남기지 않기 위해 실 적용은 하지 않았다.
   판독에 필요한 정보(join 검증)는 dry-run 출력에서 전부 얻었다.
7. **M7 점추정 상단 초과** (+0.0075). 등록된 반증 조건이 아니므로 마크 판정에는 영향이 없으나
   §4.3 에 명시했다.
8. **per-chain wall 이 r1 대비 25% 길다** (중앙 411 s vs 328 s). 사전등록 §5 가 "이 셀의 cycle 이
   길어 선형 스케일링이 낙관적일 수 있다"고 미리 적었고, 그래서 상한을 1.0 h 가 아니라 1.5 h 로
   등록했다. 실측 wave wall 1,823 s 는 예상 ~1,684 s 대비 **+8%**, 상한 대비 **34%** 다.
9. **결과 문서 생성 시각이 사전등록 §10 의 tag 와 다르다.** 사전등록 §10 의 예시 명령은
   `--tag pinbu_minfxy_r2_20260831` 이었으나 실제 실행은 **2026-09-03** 이므로 tag 는
   `pinbu_minfxy_r2_20260903` 을 썼다. 날짜 외 차이는 없다.
10. **arming 시점 kit store 교체.** *(2026-09-03 발행 후 추가 — §11)* 199 kit 의
   `data\store\records.parquet` 는 arming 시점에 pinned 값이 아니었다(22,813,908 B / `3298E661…`).
   §9.2 항목 4 의 등록된 처방은 "재 stamp 하지 말고 조정 box 에서 재 plan" 이었으나, 실제로는
   로컬 pinned store 를 kit 에 scp 하고 기존 사본을 `records.parquet.bak_pre_pinbu_r2_20260903`
   로 백업한 뒤 항목 4·5 를 통과시켰다. 실행된 것은 §8 이 고정한 바로 그 snapshot 이므로
   결정성 기준은 유지되지만, 이 사실은 사전등록 §10 step 0 STAMP 에만 기록되어 있었다.

---

## 10. Provenance / stamps

### 10.1 산출물 해시

| artifact | bytes | sha256 |
|---|---:|---|
| `runs/pinbu_wave_minfxy_r2/pinbu_wave_results.jsonl` (30행) | 44,239 | `53A42A73EB3F6368ED3BBCB5F2403AFB1A7FCF76254A780A402E41E85320E8B6` |
| `runs/pinbu_wave_minfxy_r2/pinbu_wave_minfxy_r2_out.log` | 3,743 | `5354CEED7B26581F104E4297E18B45882376A32A21493A1EDBA00015CD2825FF` |
| `runs/pinbu_wave_minfxy_r2/pinbu_wave_minfxy_r2_rc.txt` (`0/0`) | 6 | `4624804AEA3793B5940F27F4580AE6E601D9E31B8AEB5ACB1A6284D4E3BC5EEE` |
| `data/reports/fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv` (30행+header) | 5,835 | `66534BD4F588890E9B67C900F2D10A8A0E94E66E96FB14BA7121BC09C76208D0` |
| `data/reports/pinbu_wave_minfxy_r2_manifest_20260831.json` (사전등록 §8 대조 **일치**) | 72,937 | `3362F7188365F463DFA99742C725F19C1193636CAD5E2231F5BFAD7DF11902CE` |
| `fpcamp_minfxy_E1E2_f121_r2_199.inp` (limits 해소에 사용, r2 결과 §10.1 대조 **일치**) | 21,493 | `81AC7A406403B3C4BA5BB1772A59935F106281A1455945DBB671B4744FF086E8` |
| `pinbu_wave.py` (r1 launcher gate 값과 **동일**) | 36,222 | `5B3688CFAD684E9E837910F8842F68A2F2C21F931052DD52DE98262BA3581047` |
| `launch_pinbu_wave_minfxy_r2_199.ps1` (고정 후, 사전등록 §8) | 16,320 | `F549C5C7809E91964B45F9851C58949D64CBE13B2DE102B5848031313F7F3B8A` |

### 10.2 store sha (병합 전 → 후)

| 시점 | bytes | sha256 |
|---|---:|---|
| 병합 전 (사전등록 §8 이 gate 한 값) | 22,814,014 | `22854B72A4966935550FD322DA29FCAB58FBFA19FDBB84124DF444791B9C329D` |
| **병합 후 (현행)** | **22,810,322** | **`16E311AF4465E735B38DAF7ABF999268FAC27946C1C5CC279114607D9EE917BA`** |

백업: `E:/lpopt_data/5_RL/backups/records.parquet.bak_pre_pinbu_minfxy_r2_20260903`
(22,814,014 B, `22854B72…C329D`) · `…/ledger.jsonl.bak_pre_pinbu_minfxy_r2_20260903`
(19,623,488 B, `28675801A521CD1C7C97CC49018358C9FD3ECB454B8FDCFAEEDCDB469BF1D8AE`).

### 10.3 재stamp 대상 (사전등록 §8 D5 의 집행)

| 파일 | 줄 | 현재 값 | 필요한 값 |
|---|---:|---|---|
| `launch_pinbu_wave_minfxy_r2_199.ps1` | 114–115 | `$wantLen = 22814014`, `$wantSha = '22854B72…C329D'` | `22810322` / `16E311AF…17BA` |
| `launch_fpcamp_minfxy_E1E2_f121_r2_199.ps1` | 126 | `$wantStoreLen = 22782850` (76,693행 시대, 이미 stale) | 동상 |
| `launch_pinbu_wave_minfxy_r1_199.ps1` | 91 | `$wantSha = '73701E33…0C85F'` (r1 병합 이후 stale) | 동상 |
| `launch_pinbu_wave_fxyera_r1_199.ps1` | 70 | `$wantSha = 'F38666E9…A6EA'` (stale) | 동상 |
| 199 kit store `C:\Users\USER\lpopt_work\kit_frontier\data\store\records.parquet` | — | `22854B72…` / 22,814,014 B | 패치본 재 scp 필요 (§9-3) |

`22854B72…` 를 인용하는 문서 행: `minfxy_E1E2_f121_r2_prereg_20260831.md:719`,
`minfxy_E1E2_f121_r2_results_20260831.md:15 / 673 / 701 / 744`,
`pinbu_wave_minfxy_r2_prereg_20260831.md` §8,
`policy_v2_serving_ab_prereg_20260829_DRAFT.md:495`.

### 10.4 run dir 원본 / archive

| 항목 | 값 |
|---|---|
| 199 run dir | `D:\lpopt_archive_199\runs\pinbu_wave_minfxy_r2` — rc `0/0`, `NRESULTS 30/30` |
| 크기 | **356 파일 / 2,665,574,497 B**, 보존된 최종-cycle chain dir 30, `MAS_OUT` 30, `MAS_PPI` 30 |
| archive | `E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/pinbu_wave_minfxy_r2/` (356 / 2,665,574,497 — 원격 계측과 일치), TRANSFER_MANIFEST **row 28** |
| harvest 사본 | `E:/lpopt_data/5_RL/harvest/pinbu_minfxy_r2/` (**360 파일 / 2,665,589,389 B** = archive 의 356 파일 tree + 최상위 4파일 / 14,892 B; byte 동일 **아님**, 상위집합 — prunable, §9-2) |
| 199 잔여 쓰기 | scan CSV 1건 (§9-4). 삭제·수정 **없음** |

### 10.5 이 문서의 수치를 만든 명령 (재현)

```
# 0) 199 -> 로컬/아카이브 (harvest 단계, 이미 수행됨)
scp -qr USER@HOST_199:"D:/lpopt_archive_199/runs/pinbu_wave_minfxy_r2/." \
        E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/pinbu_wave_minfxy_r2/

# 1) F_xy scan (199, kit lpopt, r1 과 동일 scanner)
kit_pc2\venv\Scripts\python.exe -u -m lpopt.tools.backfill_fxy scan \
  --root D:/lpopt_archive_199/runs/pinbu_wave_minfxy_r2 \
  --out data/reports/fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv

# 2) pin 병합 (로컬, 경량 store I/O — 정본 절차)
python pinbu_wave.py patch --results runs/pinbu_wave_minfxy_r2/pinbu_wave_results.jsonl \
  --tag pinbu_minfxy_r2_20260903 [--dry-run]
python -m lpopt.tools.backfill_fxy apply \
  --csv data/reports/fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv --dry-run   # 등록된 no-op

# 3) 판독 (HOST_238, lpopt 코드 + 패치된 store)
scp -P 8022 data/store/records.parquet USER@HOST_238:~/lpopt_ws/scratch/records_pinbu_r2_76793.parquet
scp -P 8022 <backup>                    USER@HOST_238:~/lpopt_ws/scratch/records_pre_pinbu_r2_76793.parquet
scp -P 8022 runs/pinbu_wave_minfxy_r2/pinbu_wave_results.jsonl \
            data/reports/fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv \
            data/reports/pinbu_wave_minfxy_r2_manifest_20260831.json \
            fpcamp_minfxy_E1E2_f121_r2_199.inp  USER@HOST_238:~/lpopt_ws/scratch/
ssh -p 8022 USER@HOST_238 "cd ~/lpopt_ws && venv/bin/python scratch/adjudicate_r2.py"
#   -> scratch/adjudicate_r2_out.json  (M1–M9 · census · frontier · store 전후)
#   limits 는 lpopt.config.load_config -> feasibility_limits_for(..., 'min_fxy')
#            -> deliverable_limits 로 해소 (하드코딩하지 않았다)
```

---

## 11. 정정 이력 (2026-09-03, 발행 후 독립 재검증)

발행 후 두 건의 독립 반증(refutation)을 받아 HOST_238 에서 재계산했고, 아래 7건을 본문에
반영했다. **마크 M1–M9 의 판정, §10.1/§10.2 의 해시·바이트 수, §7 의 store 전후 수치,
§4·§5 의 통계는 전부 그대로다** — 반증 두 건 모두 "모든 mark-bearing 수치는 정확히
재현된다"를 먼저 확인했다. 고쳐진 것은 전부 비-mark 서술·인용·보관 메타데이터다.

| # | 위치 | 발행판 (틀림) | 정정 (2026-09-03) | 근거 |
|---:|---|---|---|---|
| 1 | §6.3 인구조사 표, `frontier_fr_backfill5` 행 | 구속 축 = `F_r` (1.4694, 여유 0.0806) | 구속 축 = **`F_xy`** (1.6056, 여유 **0.0444**), `F_r` 은 0.0806 으로 2위 | 238 재계산 |
| 2 | §6.3 축별 최악 여유 표 | `f_r` 행에 "← 유일하게 0.1 미만" 주석 (바로 아래 `f_xy` 0.0444 와 자기모순) | `f_r` = "← 최소", `f_xy` 0.0444 = "← 0.1 미만 (2위)" | 동상 |
| 3 | §6.3 맺음말 | "0.1 미만 여유를 가진 축은 `F_r` 뿐이고 top20 안에서만" | 0.1 미만 축은 **둘** (`F_r` top20 0.0156 / bf5 0.0806, `F_xy` top20 0.0831 / bf5 0.0444). 선례 r1 결과 §6 의 같은 서술도 부정확(그 wave deliverable `F_xy` 최악 여유 0.061) | 동상 + r1 결과:245 |
| 4 | §8 "열린 것" 5 | "`F_r` 여유가 이 인구의 **유일한** tight 축" | tight 축은 `F_r`(0.0156)과 `F_xy`(0.0444) 둘. 다만 top20 안에서만은 `F_r` 이 `F_xy`(0.0831)보다 5배 좁다 | 동상 |
| 5 | §8 "확정" 1 | `verify.py:851` | **`lpopt/search/verify.py:949`** (파일 sha256 `D1825085…084F`, 로컬·238 byte 동일; :851 은 resolver 주석. 사전등록 `minfxy_E1E2_f121_r2_prereg_20260831.md:492` 의 :851 인용도 stale) | grep |
| 6 | §9-2 / §10.4 | harvest 사본과 archive 사본이 "356 파일 / 2,665,574,497 B 로 byte 동일" | harvest 는 **상위집합**: **360 파일 / 2,665,589,389 B** (= archive 356 파일 tree + 최상위 4파일 / 14,892 B). prunable 결론은 유지(4파일 전부 다른 곳에 존재) | 로컬 `find` 계측 |
| 7 | §9 편차 목록 / §10.3 인용 행 | arming 시점 kit store 교체가 빠짐; 인용 행 목록에 `minfxy_E1E2_f121_r2_results_20260831.md:673` 누락 | 편차 **10번** 추가, 인용 행 목록을 `:15 / 673 / 701 / 744` 로 수정 | 사전등록 §10 step 0 STAMP, grep |

**정정된 수치의 재검증 (2026-09-03).**

상자 (1)(2)(3)(4) — HOST_238, 병합 후 store `records_pinbu_r2_76793.parquet`(22,810,322 B,
`16E311AF…17BA`)의 25 core 에 대해 `deliverable_limits`(`f_r` 1.55 / `f_xy` 1.65 / `f_q` 2.41 /
`|AO|` 0.30 / `cbc_max` 1600 / pin 80)까지의 축별 최악 여유를 그룹별로 재산:

```
ssh -p 8022 USER@HOST_238 "cd ~/lpopt_ws && venv/bin/python scratch/verify_r2_margins.py"
```

| set | 1위 tight | 2위 tight |
|---|---|---|
| pooled 25 | `f_r` 1.5344 / **0.0156** (`1a4e702d`) | `f_xy` 1.6056 / **0.0444** (`6d1081b2`) |
| `minfxy_r2_top20` (20) | `f_r` 1.5344 / **0.0156** (`1a4e702d`) | `f_xy` 1.5669 / **0.0831** (`b080c743`) |
| `frontier_fr_backfill5` (5) | `f_xy` 1.6056 / **0.0444** (`6d1081b2`) | `f_r` 1.4694 / **0.0806** (`a785eded`) |

나머지 축은 세 집합 모두 여유 0.25 이상(`|AO|` 0.2595, `f_q` >= 0.4912, pin >= 13.106,
`cbc_max` >= 263.45) — §6.3 의 "pin·`F_q`·CBC·AO 로 제한되는 core 0건" 은 그대로 맞다.

상자 (5) — `grep -n enable_pin_burnup lpopt/search/verify.py` -> 단 1회, **949행**.
파일 sha256 `D18250855D3660465123D0CF52570BFD504DD0CBADD17F1EF632F7DCB3FC084F`.

상자 (6) — 경량 로컬 파일시스템 계측(계산 없음):

```
find "E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/pinbu_wave_minfxy_r2" -type f | wc -l   # 356
find "E:/lpopt_archive/199_runs_20260830/kit_frontier_runs/pinbu_wave_minfxy_r2" -type f -printf "%s\n" | awk '{s+=$1} END {print s}'   # 2665574497
find "E:/lpopt_data/5_RL/harvest/pinbu_minfxy_r2" -type f | wc -l                                   # 360
find "E:/lpopt_data/5_RL/harvest/pinbu_minfxy_r2" -type f -printf "%s\n" | awk '{s+=$1} END {print s}'  # 2665589389
```
차이 4 파일 / 14,892 B = 5,835 + 3,743 + 6 + 5,308 (정확히 일치).

상자 (7) — `grep -rn 22854B72 data/reports/*.md` -> `minfxy_E1E2_f121_r2_results_20260831.md`
의 15·**673**·701·744 네 행에서 인용된다.

**반증이 확인해준 것 (변경 없음).** 두 반증자 모두 `adjudicate_r2.py` 와 독립한 스크립트로
M1–M9 전부, §2 20행 표 전체, §2.1 상관수, §4.2 비율 추정자, §6.1–§6.2 여유, §7 store 전후,
§10.1 해시 전부, 백업 3건, §10.3 재-stamp 표의 네 행번호·값, 199 run dir 원격 계측(356 /
2,665,574,497 B), 교차 문서 인용을 재현했다. 사전등록 준수 판정(post-hoc mark 없음, bar
이동 없음, §12 금지 4건 준수)도 두 번 독립적으로 확인됐다.

---

*생성 2026-09-03. 출처: `runs/pinbu_wave_minfxy_r2/pinbu_wave_results.jsonl` (30행, rc `0/0`),
`…_out.log` (wave wall 1,536 s + 287 s), `data/reports/fxy_backfill_199_pinbu_wave_minfxy_r2_20260903.csv`
(30 sane, `digest16 = sha256(pack_pattern)[:16]` 로 join, 30/30 매칭),
`data/reports/pinbu_wave_minfxy_r2_manifest_20260831.json`, 병합 후 `data/store/records.parquet`
(`16E311AF…17BA`) 및 병합 전 백업(`22854B72…C329D`). deliverable 판정·frontier·store 전후 대조는
HOST_238 에서 lpopt 코드(`lpopt.search.campaign.is_deliverable` / `unknown_axes` /
`deliverable_limits`)로 산출했다.*
