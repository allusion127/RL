# Serve-path featurization fix — per-cell calibration refit (2026-08-29 / 30)

**상태: 완료.** `featurize.serve_provenance()` / `model_api._record_inputs` 수정
(prereg `fxy_head_prereg_20260829.md` **Amendment C.3**, arm-2 결과보고서 §6)이
`backend.predict` 를 경유해 적합된 **모든 per-cell 보정**을 무효화했다. 본 문서는 그
재적합의 실행 기록이다.

**본 문서는 어떤 승격·개명·deck 수정도 하지 않았다.** 쓴 파일은 세 model dir 안의
보정 아티팩트(+백업)와 `data/reports/` 아래 산출물뿐이다.

---

## 1. 무엇이 무효화되었고 무엇이 아니었나

| 아티팩트 | 적합 경로 | 수정의 영향 | 처분 |
|---|---|---|---|
| `cell_calibration.json` (cyclen) | `cell_calibrate._fit_cell_affine_target` → **`backend.predict`** | **무효** | 재적합 |
| `f_r_calibration.json` | 〃 | **무효** | 재적합 |
| `cbc_calibration.json` | 〃 | **무효** | 재적합 |
| `f_q_calibration.json` | 〃 | **무효** | 재적합 |
| `ao_abs_calibration.json` | 〃 | **무효** | 재적합 |
| `flatness_calibration.json` (node_peak·map_cov) | 〃 | **무효** | 재적합 |
| `calibration.json` (σ isotonic + Platt) | `calibrate.fit_calibration` → **`LPDataset` / `predict_dataset`** (= **학습** featurization) | **영향 없음** | **재적합 불필요** |
| `cyclen_physics_prior.json` · `power_prior.json` · `ensemble.json` | 학습 산출물 | 영향 없음 | 그대로 |

`calibration.json` 이 제외되는 근거는 코드다 — `calibrate.py` 는 `from .dataset_torch import
LPDataset` 로 val fold 텐서를 만들고 `train.predict_dataset` 로 forward 한다. 서빙 요청
featurization(`_record_inputs`)을 **한 번도 타지 않는다.** 그래서 이 파일에는
`.bak_pre_servefix_20260829` 백업이 없고, 있어서도 안 된다.

---

## 2. 적합 절차 — 계보 재현 (새 로직 없음)

`train.fit_cell_calibrations` 가 학습 종료 후 호출하는 것과 **문자 그대로 같은 진입점·같은
기본 knob** 을 호출했다 (`lpopt/model/cell_calibrate.py`):

```
fit_cell_affine          -> cell_calibration.json      (cyclen)
fit_cell_affine_fr       -> f_r_calibration.json
fit_cell_affine_cbc      -> cbc_calibration.json
fit_cell_affine_fq       -> f_q_calibration.json
fit_cell_affine_ao       -> ao_abs_calibration.json
fit_flatness_calibration -> flatness_calibration.json
```

- `library_id = "ga80"` (= `TrainConfig.calibration_library_id` 기본값), `device="cpu"`,
  `min_rows`/`slope_min_rows`/`bin_width`/가중치/offset 전부 **기본값**.
- `split` 은 **각 아티팩트 자신의 메타가 적은 split** 을 그대로 썼다 — `s1i` → **S1i**,
  두 후보 → **S1j**. 재적합본의 `split` 필드가 원본과 동일함을 확인했다.
- 누수 가드(`_fit_cell_affine_target` 내부의 "적합 행은 holdout id 가 될 수 없다")는
  그대로 작동했다.
- **cyclen 은 freeze-parity 규칙을 보존했다.** 두 후보는 `--init-from data/models/s1i
  --freeze-trunk-cyclen` 으로 학습되어 cyclen head 가 챔피언과 byte-identical 이고,
  `train.fit_cell_calibrations` 는 이 모드에서 `cell_calibration.json` 을 **재적합하지 않고
  챔피언 것을 복사**한다. 따라서 `s1i` 만 재적합하고 그 결과를 두 후보에 **복사**했다
  (세 dir 의 `cell_calibration.json` sha256 이 동일한 이유).

실행 스크립트: `<scratch>/refit_one.py` (이전 세션) · `<scratch>/refit_one3.py`
(= 같은 파일에 `arm3` 항목 추가 + 프로세스당 스레드 3 고정. 적합 로직은 무변경).
아티팩트별 원장은 `<scratch>/refit/<model>_<target>.json`.

### 2.1 적합 입력이 실행 도중 바뀌지 않았음을 검증

재적합이 도는 동안 `merge-store`(`produce_fxyera_r1` 수확)와 f_xy backfill 이 store 를
건드렸다. **적합 입력은 불변임을 실측으로 확인했다** —
`records.parquet.bak_pre_fxyera_r1_20260830`(74,717행) vs 현재(75,793행):

| 점검 | 결과 |
|---|---|
| 새 record_id | **1,076** (전부 신규) |
| 기존 행의 `f_xy`·`f_r`·`cyclen`·`cbc_max`·`f_q`·`ao_abs`·`node_peak`·`map_cov`·`e_core`·`converged` 변경 | **0** (NaN 상태 변경도 0) |
| 새 id 중 S1i·S1j **train** fold 에 든 것 | **0** |
| 새 id 중 S1i·S1j **val** fold 에 든 것 | **0** |
| S1j val 라벨 f_xy 행 | **793 → 793** (동일 집합) |

교차 확인: 모든 재적합본의 `n_train_labelled` 가 수정 전 아티팩트와 **완전히 일치**한다
(s1i 23,367 / ao_abs 22,865 / flatness 13,904; 두 후보 23,498 / ao_abs 22,996 / flatness 14,035).
즉 **store 변동은 적합에 들어가지 않았다.**

---

## 3. 재적합 목록 — 18건 전부 성공, 실패 0

| model dir | 아티팩트 | 조치 | 초 | `created_at` |
|---|---|---|---:|---|
| `data/models/s1i` | `cell_calibration.json` | refit | 10,374.3 | 2026-08-30T00:44:30 |
| 〃 | `f_r_calibration.json` | refit | 10,375.9 | 00:45:16 |
| 〃 | `cbc_calibration.json` | refit | 10,461.9 | 00:41:57 |
| 〃 | `f_q_calibration.json` | refit | 10,500.3 | 00:42:36 |
| 〃 | `ao_abs_calibration.json` | refit | 6,402.6 | 02:31:37 |
| 〃 | `flatness_calibration.json` | refit | 2,208.6 | 01:21:43 |
| `data/models/20260829_163820` (arm 2) | `cell_calibration.json` | **copy** (freeze-parity) | — | — |
| 〃 | `f_r_calibration.json` | refit | 10,550.5 | 00:43:26 |
| 〃 | `cbc_calibration.json` | refit | 10,505.5 | 00:42:41 |
| 〃 | `f_q_calibration.json` | refit | 10,483.6 | 00:42:19 |
| 〃 | `ao_abs_calibration.json` | refit | 10,185.1 | 00:43:36 |
| 〃 | `flatness_calibration.json` | refit | 3,350.5 | 2026-08-29T22:43:27 |
| `data/models/20260829_194532` (arm 3) | `cell_calibration.json` | **copy** (freeze-parity) | — | — |
| 〃 | `f_r_calibration.json` | refit | 6,519.9 | 2026-08-30T02:33:34 |
| 〃 | `cbc_calibration.json` | refit | 6,560.6 | 02:34:15 |
| 〃 | `f_q_calibration.json` | refit | 6,557.7 | 02:34:12 |
| 〃 | `ao_abs_calibration.json` | refit | 6,431.4 | 02:32:05 |
| 〃 | `flatness_calibration.json` | refit | 2,225.5 | 01:22:00 |

모든 스칼라 적합이 **102 cell fitted / 65 skipped**, flatness 가 **83 cell / 69 skipped**
(node_peak·map_cov 각각)로 수정 전과 동일한 형상을 냈다. `dropped_serve_parity` 도 동일
(`260624` 25,657 + `5.8_5.1` 1,295 또는 4,902 — 타깃별 라벨 유무 차이).

> **이전 세션의 중단 지점 (기록).** 직전 에이전트가 12건을 투입했으나 프로세스 2건
> (`s1i` 의 `ao_abs`·`flatness`)이 산출 없이 죽었고(0-byte 로그), arm 3 은 아예 착수되지
> 않았다. 본 세션은 살아 있던 4건을 **죽이지 않고** 완주시킨 뒤 죽은 2건을 재투입하고
> arm 3 5건 + cyclen 복사 2건을 추가했다. 적합 로직·knob·split 은 승계했다.

---

## 4. Before → After (S1j val, **라벨 793행**, FIXED serve path)

"before" = 수정된 경로 + **재적합 전** 보정. "after" = 수정된 경로 + **재적합 후** 보정.
`data/models/20260829_194532` 의 before 는 백업에서 복원한 shadow dir 로 측정했다.
(참고 좌표: Amendment C.3 이 기록한 **ORIGINAL** path 의 s1i 값 — F_r 0.052678 /
cyclen 1.852122 / CBC 14.210126.)

### 4.1 `data/models/s1i` (챔피언)

| 축 | MAE before | **MAE after** | bias before | **bias after** |
|---|---:|---:|---:|---:|
| **F_r** | 0.102616 | **0.049914** | +0.084694 | **−0.000239** |
| **cyclen** (EFPD) | 2.400414 | **1.621244** | +0.840174 | **+0.120223** |
| **CBC_max** (ppm) | 16.607514 | **12.479430** | +2.463482 | **−0.505266** |
| F_q | 0.146613 | **0.075211** | +0.122644 | −0.001259 |
| AO_abs | 0.003345 | **0.002795** | +0.001848 | −0.000068 |
| node_peak | 0.049653 | **0.046648** | −0.002835 | +0.002262 |
| map_cov | 0.010955 | **0.010791** | −0.001459 | −0.001221 |
| f_xy proxy on served F_r | 0.132102 | **0.073173** | +0.105287 | +0.001872 |

**검수 기준 충족.** 요구된 복귀선은 F_r ≈ 0.05 / cyclen ≈ 1.9 EFPD / CBC ≈ 14 ppm 이었고
실측은 **0.0499 / 1.62 / 12.48** 로 세 축 모두 도달했으며 ORIGINAL path 값보다도 낫다.
그 이유는 구조적이다 — 보정이 이제 **자기가 서빙하는 바로 그 featurization 위에서** 적합되었고,
`bias` 가 모든 축에서 사실상 0으로 떨어진 것이 그 증거다.

### 4.2 `data/models/20260829_163820` (arm 2)

| 축 | MAE before | **MAE after** | bias before | **bias after** |
|---|---:|---:|---:|---:|
| **F_r** | 0.099989 | **0.049232** | +0.086272 | **−0.000019** |
| **cyclen** | 2.409075 | **1.619684** | +0.888242 | +0.168291 |
| **CBC_max** | 15.535530 | **11.929909** | +3.201234 | +0.097939 |
| F_q | 0.144793 | **0.074183** | +0.125133 | −0.000184 |
| AO_abs | 0.003335 | **0.002755** | +0.001650 | −0.000113 |
| node_peak | 0.047090 | **0.044998** | +0.003628 | +0.001534 |
| map_cov | 0.010027 | 0.010031 | −0.000731 | −0.001050 |
| f_xy proxy on served F_r | 0.129197 | **0.072330** | +0.107208 | +0.002141 |

### 4.3 `data/models/20260829_194532` (arm 3)

| 축 | MAE before | **MAE after** | bias before | **bias after** |
|---|---:|---:|---:|---:|
| **F_r** | 0.099669 | **0.049132** | +0.085554 | **−0.000401** |
| **cyclen** | 2.409075 | **1.619684** | +0.888242 | +0.168291 |
| **CBC_max** | 15.694640 | **11.994568** | +3.536277 | +0.130383 |
| F_q | 0.144116 | **0.074112** | +0.123764 | −0.000458 |
| AO_abs | 0.003332 | **0.002749** | +0.001623 | −0.000122 |
| node_peak | 0.046699 | **0.044112** | +0.003506 | +0.002661 |
| map_cov | 0.010149 | 0.010186 | −0.000952 | −0.000927 |
| f_xy proxy on served F_r | 0.128808 | **0.072149** | +0.106334 | +0.001676 |

> **정합성 점검 두 가지.**
> (i) arm 2 와 arm 3 의 **cyclen 이 소수점까지 동일**하다 (MAE 1.619684, bias +0.168291).
> 두 후보는 frozen cyclen head 가 byte-identical 이고 같은 복사본 보정을 쓰므로 **그래야 한다.**
> (ii) `map_cov` 만 MAE 가 사실상 제자리(±0.00004)다. `map_cov` 는 수정된 두 global 채널에
> 거의 반응하지 않는 축이라 재적합의 이동폭이 노이즈 수준인 것이며, 나머지 일곱 축의
> 큰 개선과 모순되지 않는다.

---

## 5. sha256 — 재적합 후 (전부)

| model dir | 파일 | sha256 |
|---|---|---|
| `s1i` | `cell_calibration.json` | `1f4b741583d3977ed92d7eb859ecdf7128f62982a85323de53df5f21b441e90c` |
| 〃 | `f_r_calibration.json` | `b7576cd1117eb0d27663da1241253907560ee8bd2800b83b44db49d5024cc7a1` |
| 〃 | `cbc_calibration.json` | `979190abe1d3653e48e76c39413d9e8b3328ad0396362747eba9a6bb34d2833b` |
| 〃 | `f_q_calibration.json` | `22084abfadb92a0c16a667d9f853d45a3d24a266eb250d4934504b7dfb289f09` |
| 〃 | `ao_abs_calibration.json` | `64116d98d880d1b3ce75f7ca1c76553dfe72b3536f9e2b0f829732808ba43c5f` |
| 〃 | `flatness_calibration.json` | `a05eb712cac8b510b4b33abb3a73db399bbdec5e84ab79a169278ec5ef3931c7` |
| `20260829_163820` | `cell_calibration.json` | `1f4b741583d3977ed92d7eb859ecdf7128f62982a85323de53df5f21b441e90c` |
| 〃 | `f_r_calibration.json` | `d1f733af5cac3f497a5baff8bbf5a5ee5cdda2538e96689de89be4204126bd41` |
| 〃 | `cbc_calibration.json` | `fca0ce148bab7caa89871ecc6b2d9940323af6e4426dd7cc77ddd5278743fea4` |
| 〃 | `f_q_calibration.json` | `78431999102064f2c0999e17c900b54a3a3829d56e182cb06036e504a24a40c1` |
| 〃 | `ao_abs_calibration.json` | `c6a3071a7cf4264c58948f9526f401ff5667f531af78da4315bb2b6057fe746e` |
| 〃 | `flatness_calibration.json` | `db4e5bf174fb54c493ca4ab29aa54f640fe34dd79cd361fbd5ff7301bb9a18a7` |
| `20260829_194532` | `cell_calibration.json` | `1f4b741583d3977ed92d7eb859ecdf7128f62982a85323de53df5f21b441e90c` |
| 〃 | `f_r_calibration.json` | `d4e963304dc3dcf85d662c73a34d29229b3cf28bd660bbd6c545f67c68f2c9b7` |
| 〃 | `cbc_calibration.json` | `9152620559d7aaa4274b0e30c24fb1330843fdf514b047f786a1cf40e081d20d` |
| 〃 | `f_q_calibration.json` | `babd655d2163019cc0a60b6a1369f6905aa94629f3425f696c10f8667d85be50` |
| 〃 | `ao_abs_calibration.json` | `b3991af39a3d7866a04193dd424e8531feb0ffb0db40cac90e6414e69ae18ce9` |
| 〃 | `flatness_calibration.json` | `c91870dd4aa54ac05f0e9a2d8a347d32f7860380dd1f85ed2c82c18bcd68bed1` |

수정 전 판은 같은 dir 에 `<name>.bak_pre_servefix_20260829` 로 **전부 보존**되어 있다
(s1i 6, arm 2 6, arm 3 6 — 총 18). 백업은 덮어쓰지 않는다(`refit_one*.py` 가 존재 시 skip).

서빙 경로 코드 (Amendment C.3 과 일치 확인):

| 파일 | sha256 | bytes |
|---|---|---:|
| `lpopt/model/model_api.py` | `94229de9e332c7faa66529f51b03d107f20098b1758f999b9c73ec8cfb21e6a2` | 110,976 |
| `lpopt/model/featurize.py` | `6977344dafbd770c9b1bc40e370db6c189320e301f8fa49570a25f927b575e36` | 86,245 |

parity gate `tests/test_model_api.py::test_serve_row_featurization_parity` — **1 passed**
(2026-08-30 재실행).

---

## 6. 운영 규칙 (등록)

> ### R1. 수정된 serve path 위에서 서빙되는 모든 model dir 은 **그 경로에서 재적합된 per-cell 보정 6종**을 반드시 지녀야 한다.
> 미재적합 보정은 이미 무편향인 raw 값에 **+0.073 (F_r 기준)** 을 더한다. 실측은 §4 다 —
> F_r MAE ×2.06, cyclen +0.78 EFPD, CBC +4.1 ppm.

- **순위만 쓰는 경로는 안전하다** (아핀 보정은 단조라 순위를 못 바꾼다). 그러나
  `F_r ≤ 1.55`, `F_xy ≤ 1.65`, pin-BU 임계 같은 **수준(level) 임계를 쓰는 어떤 경로도**
  재적합 전에는 신뢰할 수 없다. `acquisition` 은 `backend.predict` 의 **보정된** 열을 읽는다.
- **코드와 보정은 한 세트로 움직인다.** 수정된 `featurize.py`/`model_api.py` 를 배포하면서
  미재적합 보정을 함께 보내는 것, 그리고 그 반대 — **둘 다 금지**한다.
- 재적합 비용은 아티팩트당 CPU 1.8~2.9 시간(24-core 박스에서 5~8개 동시)이다.
  캠페인 중단 창을 잡아야 한다.

### 6.1 HOST_238 (GPU 학습 서버) — 조용한 실패 지점 하나

`lpopt/remote.py` 의 `ensure_checkpoint` 는 `checkpoint_fingerprint` 로만 재전송을 판단하고,
그 지문은 **`member_*/meta.json` + `backend.json` + `calibration.json` + `ensemble.json`**
만 해싱한다. **per-cell 보정 6종은 지문에 들어가지 않는다.**

> **귀결: 보정만 재적합하면 지문이 그대로이므로 `ensure_checkpoint` 는 push 를 건너뛴다.
> HOST_238 의 미러는 낡은 보정을 조용히 유지한다.** 원격 스크리닝을 켜기 전에
> 반드시 강제 재전송해야 한다 (원격 `…/FINGERPRINT` 를 지우거나 미러 dir 을 삭제).

배포할 파일 (model dir 당 — 승격 시엔 승격된 dir):

```
<model_dir>/cell_calibration.json
<model_dir>/f_r_calibration.json
<model_dir>/cbc_calibration.json
<model_dir>/f_q_calibration.json
<model_dir>/ao_abs_calibration.json
<model_dir>/flatness_calibration.json
lpopt/model/featurize.py            # sha 6977344d…  (serve_provenance 도입)
lpopt/model/model_api.py            # sha 94229de9…  (_record_inputs 가 그것을 호출)
```

`calibration.json`·`cyclen_physics_prior.json`·`power_prior.json`·`member_*/` 는 **변경 없음**
— 다시 보낼 필요가 없다(보내도 무해).

> **미배포 상태의 실측 영향 (Amendment C.3 #4).** 원격이 수정 전 코드인 동안
> `tests/test_remote_infer.py::test_remote_gpu_matches_local_cpu_determinism` 의 `mu_z`
> 최대 차이가 **4.282e-03 → 1.209e+00** 으로 벌어진다. 코드를 push 하기 전까지
> **원격 스크리닝을 켜면 안 된다.**

### 6.2 HOST_199 kit

`multi_pc.export_frontier_kit` / `export_produce_kit` 는 kit 을 만들 때
`shutil.copytree(model_dir, out/"data/models/champion")` 로 **model dir 전체**를 복사하고,
`lpopt` 패키지 소스도 **통째로** 복사한다. 따라서 **재적합·수정 이후에 kit 을 새로 export
하면 kit 은 자기완결적으로 정합**하다.

> **금지: 기존 kit 에 보정 파일만 손으로 갈아끼우는 것.** 그 kit 의 `lpopt/` 는 수정 전
> `featurize.py`/`model_api.py` 이므로, 재적합된 보정과 결합하면 **§4 의 before 보다도 나쁜**
> 조합(수정 전 featurization + 수정 후 보정)이 된다. **kit 은 통째로 다시 export 한다.**

새 kit 이 반드시 실어야 하는 것:

```
data/models/champion/**       # 재적합 보정 6종을 포함한 model dir 전체
lpopt/**                      # featurize.py 6977344d… / model_api.py 94229de9…
data/store/records.parquet    # 현재 f38666e9f1508d35d33e0c22f583c5479c6f09cac748201b494b47c8cfeca6ea (75,793행)
data/store/maps.npz, data/store/fuel_types.parquet
```

- kit 의 `_COPY_IGNORE` 는 `*.bak_pre_servefix_20260829` 를 거르지 않는다 — 약 240 KB 가
  함께 실린다. **무해하나** 원치 않으면 export 전에 옮겨 두면 된다.
- **launcher 선행 차단 (지금 유효).** `launch_fpcamp_minfxy_T6T4_f121_r1_199.ps1` 의
  `$wantStore = '0334E2D2…'` 는 e_core backfill **이전** 값이다. 현재 store 는
  `F38666E9…` 이므로 지금 돌리면 `MINFXY1 REFUSED: store sha256 mismatch` 로 거부된다.
  minfxy prereg §9.1 의 store 행과 launcher `$wantStore` 를 함께 갱신해야 출발한다.
  (본 문서는 두 파일 중 **어느 것도 수정하지 않았다.**)

### 6.3 미결로 남기는 것

`lpopt/policy/scorer.py` 는 여전히 `featurize.library_provenance` 를 쓴다 — 같은 결함을
갖지만 모델 서빙 경로 밖이므로 고치지 않았다 (Amendment C.3 #5 그대로 승계).

---

## 7. 이 재적합이 지지하는 판독

`data/reports/fxy_gate_eval_arm2c_20260829.json` (arm-2c, **정보 제공용**) 및
`data/reports/fxy_gate_eval_arm3_20260829.json` (arm-3, **구속력**)은 **양쪽 다 재적합된**
보정 위에서 산출되었다. 판정은 `fxy_head_results_arm3_20260829.md` 를 보라.

핵심 부수 결과 하나: FIXED path + 재적합에서 **서빙 proxy 자신의 값**은
MAE **0.073173** · ρ̄ **0.715696** 다 (ORIGINAL path 등록값 0.076721 / 0.726296).
즉 재적합은 proxy 도 같이 개선했고, **바(0.0767 / 0.7263)는 이동하지 않았다** (C.3 #2).
