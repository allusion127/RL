# 방출연소도 `B_d` 백필 계획 (PLAN)

**작성** 2026-09-03 · **성격** 사양(spec)만. **코드 0줄 · DeCART 0 · MASTER 0 · 로컬 연산 0.**
**동기** `assembly_slice_Z_prereg_20260903_DRAFT.md` §7.2.2–§7.2.3 (MODE-P/MODE-T 의 `B_d` 축) ·
`fr135_feasibility_scoping_20260903.md` §6.1 "★ 방출연소도 축 — 선행 데이터 작업 1건".
**수신** 프로그램 오너.

---

## 1. 문제

`data/store/records.parquet` 의 `discharge_burnup` 컬럼은 **Tier-1 2,028행 전부 NaN** 이다
(스토어 전체로도 38,851/67,880 행만 채워짐). 저-`F_r` 상위 2,000행에서 채워진 burnup 축은
`max_assembly_burnup` (n=1,746, 중앙 56.38, min 53.14, max 77.08 MWd/kgHM) 뿐이다.
→ **`B_d` 를 Pareto 축이나 target 으로 쓰려면 백필이 선행**한다.

---

## 2. MASTER 출력 조사 — **진짜 배치평균 방출연소도 파서는 어디에도 없다**

| 파일:라인 | 무엇을 파싱하는가 | `B_d` 인가 |
|---|---|---|
| `lpopt/data/edit5.py:1-13, 246` | `MAS_SUM` EDIT2(반응도) / EDIT3(AO·peak) / EDIT5(**per-step 집합체 2-D 맵**: batch / power / **burnup** / k-inf) / EDIT6(축방향) | **아니다** — EDIT5 는 *스텝별 집합체 연소도 맵*이지 방출 시점의 배치평균이 아니다. `extract_maps` / `stack_step_maps` 는 그 맵을 그대로 넘긴다 |
| `lpopt/vendor/masterrl/burnup.py:70` `parse_summary_max_assembly_burnup` | `MAS_SUM` 에서 **집합체 연소도 최대**(단일 집합체 peak) | **아니다** — 최대이지 평균. 이것이 `max_assembly_burnup` 의 유일한 원천 (`master.py:404-416`) |
| `lpopt/vendor/masterrl/burnup.py` `parse_ppi_max_pin_burnup` (`lpopt/data/pinppi.py:3,23`) | `MAS_PPI` `BPIN` 의 **핀 노드 최대** | **아니다** — `max_pin_burnup`, 인허가 80 GWd/tU 축 |
| `lpopt/data/extract_a.py:374` `map_metrics` | `rec.metrics["discharge_burnup"]` 을 **그대로 통과**시킬 뿐 | **생산자가 없다** — 그래서 컬럼이 NaN 이다 |
| `lpopt/data/extract_b.py:494` | `discharge_burnup=None` **명시** | — |
| `lpopt/data/schema.py:115, 209` | `discharge_burnup: float \| None`, `pa.float64()` — **컬럼은 이미 존재** | 스키마 슬롯만 있다 |
| **`lpopt/design/bootstrap.py:104-117`** `estimate_discharge_burnup` | **에너지 밸런스 추정** — 유일한 생산자. `bootstrap.py:238` 에서 부트스트랩 결과에만 붙는다 | **프록시** (아래 §3 과 동일 식) |

> **결정적 문장 (인용).** `bootstrap.py:109-111` — *"Reported as an estimate — the vendor FOM
> carries no native discharge_burnup (plan 12.4 promotes it to a first-class MASTER target)."*
> → **MASTER FOM 에 네이티브 필드가 없다.** 파서를 새로 쓸 대상 자체가 없으므로,
> 백필은 **프록시 정의의 등록**이지 파싱 작업이 아니다.

---

## 3. 등록 프록시 정의 (registered proxy)

```
B_d  =  P_th · cyclen / M_HM(core)  ×  (N_FA / feed) / 1000        [MWd/kgHM]

     P_th   = 3983 MW          (config.py:1165  [criteria] power_mw)
     N_FA   = 241
     M_HM   = Σ_241 u_mass_g   (fuel_types.u_mass_g, FA_*.out MASS(g);
                                fuel_types.py:407-414, 1473)
     폴백    M_HM = 104.8 MTU  (config.py:1166  [criteria] hm_mtu)
```

**이 식은 `design/bootstrap.py:104` `estimate_discharge_burnup` 과 동일하다** —
새 물리를 도입하지 않고 저장소가 이미 쓰고 있는 추정기를 스토어 전체로 확장하는 것이다.

**등록된 가정 (전부 명시, 결과 문서가 반드시 라벨한다):**

1. **평형 배치 · 배치평균.** 모든 집합체가 동일 잔류시간 `241/feed` 사이클을 산다고 본다.
   실제 평형 배치 센서스는 그렇지 않다 (`opscreen_chain.batch_weights`: feed 121 →
   121 @ Bc + 120 @ 2Bc; feed 117 → 117, 117, 7). **`241/feed` 는 그 근사다.**
2. **`M_HM` 은 우라늄 질량**이다 (`u_mass_g` = U-235 + U-238). Gd 캐리어 핀의 U 포함 여부와
   중금속(HM) 정의 차이가 **1 % 오더의 계통 편의**를 남긴다.
3. **ga80 라이브러리는 `u_mass_g` 가 전부 NaN** (fuel_types.py:2066-2067) → 그 행은 폴백
   `hm_mtu = 104.8` 로 계산하고 **`bd_source = "proxy_hmfallback"`** 로 라벨한다.
4. **★ `B_d` 는 LP 위치에 무감이다.** cell·feed 를 고정하면 `B_d` 는 `cyclen` 의 **아핀 함수**다
   → 같은 셀 안에서 `B_d` 순위 = `cyclen` 순위. **그래서 `B_d` 는 축이지 마크가 될 수 없다**
   (사전등록 §7.5). 셀·feed **사이**에서만 새 정보를 준다 (잔류시간 `241/feed` 항).

**정합성 점검 (계산 아님, 산술 확인).** feed 121 · cyclen 620 → `3983 × 620 / 104.8 × 1.9917 / 1000`
≈ **46.9 MWd/kgHM**; cyclen 633.4(Tier-1 중앙) → ≈ **47.9**. 같은 행들의 실측
`max_assembly_burnup` 중앙 56.38 보다 **낮다** — 배치평균 < 집합체 최대이므로 **부호가 옳다.**

---

## 4. 추가할 도구 — `lpopt/tools/backfill_bd.py` (사양만)

**선례.** `lpopt/tools/backfill_flatness.py`, `lpopt/tools/backfill_fxy.py` 와 같은 자리·같은 규약.
단, `backfill_fxy` 의 `scan`/`apply` 2단계는 **필요 없다** — 프록시는 `MAS_OUT` 을 읽지 않고
스토어의 `cyclen` · `feed` · `library_id` 와 `fuel_types.parquet` 만으로 계산된다.

### 4.1 ★ 스키마 — **`discharge_burnup` 에 직접 쓰지 않는다**

```
schema.py 에 2 컬럼 추가 (nullable):
    ("discharge_burnup_proxy", pa.float64())     # §3 의 값
    ("bd_source",              pa.string())      # "proxy_utype" | "proxy_hmfallback" | "native"
`discharge_burnup` (네이티브 슬롯) 은 NaN 그대로 둔다.
```

**이유 (차단성).** `discharge_burnup` 은 **모델 타깃**이다
(`model/dataset_torch.py:76-81` 의 9-타깃, `model/model_api.py:115` `_EXTRA_TARGET_NAMES`).
프록시로 채우면 다음 재학습이 **`cyclen` 의 아핀 함수를 방출연소도 헤드로 학습**하게 되어
정보 없는 타깃을 만들고, `model/conformal.py:75-76` 이 *"discharge is all-NaN there"* 를 근거로
그 축을 홀드아웃에서 제외해 둔 전제도 조용히 깨진다. → **별도 컬럼 + 라벨.**

### 4.2 CLI

```bash
python -m lpopt.tools.backfill_bd apply \
    --store-dir data/store \
    [--fuel-types data/store/fuel_types.parquet] \
    [--power-mw 3983.0] [--n-fa 241] [--hm-mtu 104.8] \
    [--only-null] [--dry-run]
```

* **읽기:** `records.parquet`(`cyclen`, `feed`/`batch_feed`, `library_id`, `converged`),
  `fuel_types.parquet`(`u_mass_g`).
* **쓰기:** 수렴 행에 한해 두 컬럼. **`--dry-run` 이 기본 리허설**이며, 쓰기 전
  `.bak` 스냅샷 1 세대를 남긴다(백필 선례 규약).
* **보고(stdout):** 채운 행 수 / `bd_source` 별 카운트 / `B_d` 의 min·p05·중앙·p95·max /
  `cyclen` 과의 상관(셀별) — **1.000 에 가까워야 정상**이다(§3-가정 4).
* **거부:** `cyclen` 이 비었거나 ≤ 0, 미수렴, `feed ∉ [80, 241]` → 건드리지 않고 사유 카운트.

### 4.3 작성할 테스트 — `tests/test_backfill_bd.py`

1. **식 일치.** `backfill_bd` 의 단위 함수 == `design.bootstrap.estimate_discharge_burnup`
   (동일 입력 → 비트 동일). *이것이 가장 중요한 테스트다 — 두 번째 정의를 만들지 않는다.*
2. **단위.** feed 121 · cyclen 620 · hm 104.8 → 46.9 ± 0.1 MWd/kgHM (MWd/MTU → MWd/kgHM 의
   `/1000` 이 정확히 한 번 적용됨).
3. **폴백 라벨.** `u_mass_g` 가 NaN 인 라이브러리 행 → `bd_source == "proxy_hmfallback"`,
   값은 `hm_mtu` 기반.
4. **비파괴.** 실행 후 `discharge_burnup` 컬럼이 **입력과 바이트 동일**(NaN 유지).
5. **멱등.** 두 번 돌려도 결과 동일; `--only-null` 이 이미 채워진 행을 덮지 않음.
6. **거부 경로.** 미수렴 / `cyclen` NaN / feed 이상치 행은 NaN 으로 남고 사유 카운트가 맞음.
7. **스키마 왕복.** 추가된 두 컬럼이 `schema.py` 라운드트립(write→read)에서 보존됨.
8. **드라이런.** `--dry-run` 이 파일 mtime 을 바꾸지 않음.

---

## 5. `B_d` 가 들어가는 곳

### 5.1 Pareto readout (사전등록 §7.2.2)

* **MODE-P.** arm 의 클린 행에 대해 **`(F_r, cyclen, B_d)` 3-축 비지배 집합**
  (`F_r` 최소화 / `cyclen`·`B_d` 최대화)을 산출하고, 각 정점에 `node_peak` 을 병기한다.
  가정 4 때문에 **같은 셀 안에서는 `(F_r, cyclen)` 2-축 프론티어와 일치**하고,
  **셀·feed 를 섞을 때만** 3-축이 새 정점을 만든다 — 결과 문서는 이 사실을 명시한다.
* **MODE-T.** 사후 밴드 `cyclen ∈ [620, 645]` ∧ `B_d ≥ 45.0` 를 적용한 뒤의 최소 측정 `F_r`.
* **네이티브 MODE-T 는 라운드 2.** cyclen 과 방출연소도를 동시에 게이트하며 `F_r` 을
  최소화하는 코드경로는 `[criteria]` user_criteria 캠페인
  (`config.py:1113-1116` `cyclen_target`/`cyclen_tol`/`discharge_target`/`discharge_tol`,
  `search/acquisition.py:3171` `score_user_criteria`) 뿐이고, 그것은
  **`discharge_burnup`(네이티브 슬롯)을 읽는다.** §4.1 이 그 슬롯을 비워 두므로
  **user_criteria 의 `discharge_target` 은 백필 후에도 여전히 비어 있다** —
  라운드 2 에서 (a) 프록시 컬럼을 읽도록 배선하거나 (b) 진짜 배치평균을 EDIT5 맵에서
  유도하는 파서를 신설하는 것 중 하나를 **명시적으로 선택**해야 한다.

### 5.2 mesh (`scoping_mesh.py` / `mesh_multitype.py`)

현재 셀 readout 은 `(cyclen, f_r)` 2-축 Pareto 다 (`mesh_multitype.py:477-496`:
`pareto_min_f_r`, `pareto_cyclen_span`, `pareto_f_r_span`).
`B_d` 는 **예측 대상이 아니라 예측 `cyclen` 의 파생**이므로 모델 변경이 필요 없다:

```
pred_bd    = B_d(pred_cyclen, feed, M_HM)                # §3 식
sigma_bd   = (∂B_d/∂cyclen) · sigma_cyclen               # 아핀 → 정확한 전파
             ∂B_d/∂cyclen = P_th · (N_FA/feed) / M_HM / 1000
```

* 셀 CSV(`mesh_v4_cells.csv` 계열)에 `pred_bd` · `sigma_bd` · `pareto_bd_span` 3열 추가.
* **feed 가 mesh 격자의 축이므로** `B_d` 는 mesh 에서 실제로 정보를 준다 — 같은 `cyclen` 이라도
  `feed 109` 와 `feed 129` 의 `B_d` 는 잔류시간 항으로 **18 % 갈린다**.
* 스토어 오버레이(측정 행)는 `discharge_burnup_proxy` 를 그대로 쓰고 **`bd_source` 를 범례에
  라벨**한다. 프록시 값을 실측처럼 그리지 않는다.

---

## 6. 순서 · 비용 · 차단성

| 단계 | 위치 | 비용 | 차단성 |
|---|---|---:|---|
| 1. `schema.py` 2컬럼 추가 + `backfill_bd.py` + 테스트 8종 | 238 | ~2 h | — |
| 2. `--dry-run` 리허설 + 분포/상관 점검 | 238 | ~수 분 | — |
| 3. `.bak` 스냅샷 후 apply (수렴 67,880행) | 캠페인 호스트 | ~수 분 | — |
| 4. mesh 3열 추가 + readout 재생성 | 198 | ~1 h | — |
| — | **합계** | **반나절 미만**, **MASTER 콜 0 · DeCART 0** | **슬라이스 Z 를 차단하지 않는다** |

**슬라이스 Z 와의 관계 (등록).** 백필은 **비차단**이다. 백필 전에는 사전등록 §7.2.3 의 프록시를
결과 문서가 **행 단위로 직접 계산**해서 쓰고, `max_assembly_burnup` 과 **측정**
`max_pin_burnup ≤ 80` 을 **대리축으로 라벨**한다. 백필이 끝나면 같은 수가 스토어 컬럼으로
재현되어야 하며, **그 일치(1e-6 이내)가 백필의 수용 판정**이다.
