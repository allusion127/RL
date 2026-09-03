# 슬라이스 Z 파이프라인 런북 (S4 → S9c) · 2026-09-03

**성격** 집행용 명령 대본. 사전등록 `assembly_slice_Z_prereg_20260903_DRAFT.md` 의 S4 이후를
호스트·경로·플래그까지 확정한 것. **이 문서는 아무것도 실행하지 않았다** (로컬 연산 0 / DeCART 0 /
MASTER 0 / 181·199 쓰기 0).
**전제** 부록 L 의 S3 발사 (2026-09-03 13:21:20 KST, T7 PID 13500 / T8 PID 30292, 예상 종료 14:12).
**중단 규약** 각 단계의 게이트가 FAIL 이면 **그 자리에서 멈춘다**. 사전등록 §10.1 이 정본.
**§0 의 결정 4건은 S6 이전에 오너가 처분해야 한다.** 미처분 상태로 S5 를 시작하지 않는다.

---

## 0. 사전 확인 (읽기 전용, 2026-09-03 측정)

| 항목 | 측정값 | 비고 |
|---|---|---|
| 정본 패키지 위치 | **`USER@HOST_199 : C:\Users\USER\lpopt_work\kit_frontier\data\design\package`** | E:/lpopt_data 도 181 도 아니다. `lib/` 에 `prolog41m4.exe`(1,451,520 B)·`TotalBatcher4.exe`(300,544 B) 동봉 확인 |
| 현행 라이브러리 | `MAS_XSL` **14,278,423** / `MAS_HFF` **14,979,709** / `FA_*.HGC` **37** | N=37 등식 일치 |
| `.bak` 세대 | `MAS_XSL.bak` 12,735,027 (N=33) / `MAS_HFF.bak` 13,360,281 | **한 세대뿐** — 재빌드 1회로 소멸 |
| `bases/` | **31 pair** | 사전등록의 "8 pair" 는 **로컬 체크아웃 수**. 결정 D-3 |
| `cores/` | 10 folder | 사전등록 §4.4 목록과 일치 |
| `synth_decks/` | 존재 | S5b 에서 purge |
| 199 드라이브 | C: 42.2 GB / D: **6,214 GB**. **E: 없음** | 사전등록 §4.1 의 `E:\lpopt_archive\...` 는 199 에 존재하지 않는다. 결정 D-2 |
| 199 유휴 | master 0 / python 0 | |
| 199 모델 | `data\models\` 에 s1i·s1j 존재 | |
| 238 | `~/lpopt_ws/{src,venv}`, `src/data/store/records.parquet` **22,810,322 B** (로컬과 동일), `data/models/{s1i,s1j}` 존재. **`src/data/design/package` 없음** | HGC 게이트는 여기서, 라이브러리 재빌드는 불가(Linux) |
| 로컬 스토어 sha256 | `16E311AF4465E735B38DAF7ABF999268FAC27946C1C5CC279114607D9EE917BA` / 22,810,322 B | 런처 RESTAMP 기준선 |

**호스트 배분 (동결)**

| 단계 | 호스트 | 이유 |
|---|---|---|
| S4 회수 | 181 → 로컬 → 238 (파일 전송만) | |
| S4 HGC 게이트 | **238** | `lpopt/design/hgc_gates.py` 는 순수 텍스트 파서. **CLI 진입점이 없다** (`main()` 부재) → `python -c` 로 API 호출. 스크린 대조 곡선(S1 산출)이 238 에 있으므로 G-H4 도 여기 |
| S5 스냅샷·등록·재빌드 | **199** | `build_master_library` 가 `TotalBatcher4.exe`/`prolog41m4.exe` 를 `subprocess` 로 실행 (library.py:85-) → **Windows 필수. 238 에서는 원리적으로 불가** |
| S5b cores 재생성 | 199 | 같은 패키지 |
| S6 부트스트랩 | 199 | MASTER |
| S7/S8/S9 | 199 | MASTER |
| 로컬 PC | 연산 금지. Read/Grep/전송/`Get-FileHash` 만 | 상시 규칙 |

---

## 1. S4 — 181 산출물 회수 · HGC 게이트

### 1.1 181 완주 확인 (읽기 전용)

```powershell
# 1
ssh USER@HOST_181 "powershell -NoProfile -Command \"Get-Content C:\lpopt_decart\slice_Z\manifest.json -Raw\""
# 2  완주 판정: manifest.json 의 두 케이스 모두 JOB FINISHED 마커 + rc 0
ssh USER@HOST_181 "powershell -NoProfile -Command \"Get-ChildItem C:\lpopt_decart\slice_Z\T7,C:\lpopt_decart\slice_Z\T8 | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize\""
# 3  프로세스 소멸 확인 (0 이어야 한다)
ssh USER@HOST_181 "powershell -NoProfile -Command \"@(Get-Process decart2d1.1m5 -EA SilentlyContinue).Count\""
```

**S4 선행 게이트 — 크기 등식 (사전등록 §3.5):**
`FA_T7.HGC` = `FA_T8.HGC` = **정확히 7,395,955 B**. 어긋나면 `.out` 을 먼저 읽고 **회수하지 않는다**.

### 1.2 회수 (파일 전송만)

```powershell
# 4  181 -> 로컬 (경유). 각 케이스 .HGC / .out / .sum / stdout.txt / process_result.json
scp -r USER@HOST_181:C:/lpopt_decart/slice_Z/T7 "<LOCAL>/slice_Z_products/T7"
scp -r USER@HOST_181:C:/lpopt_decart/slice_Z/T8 "<LOCAL>/slice_Z_products/T8"
scp    USER@HOST_181:C:/lpopt_decart/slice_Z/manifest.json "<LOCAL>/slice_Z_products/"
# 5  해시 대조 (로컬, Get-FileHash 만 허용)
Get-FileHash -Algorithm SHA256 "<LOCAL>/slice_Z_products/T7/FA_T7.HGC","<LOCAL>/slice_Z_products/T8/FA_T8.HGC"
#    manifest.json 의 sha256 과 바이트 일치해야 한다. 불일치 -> 재전송, 게이트 진입 금지.
```

`.sum` 은 사전등록 §3.5 의 A11 처분대로 **작업 디렉터리가 아니라 러너가 남긴
`FA_<alias>.sum`** 에서 온다. `dec_FA_<type_id>.inp` 는 **`templates_lat1600/…` 또는 238 의 저작본**에서
복사한다 (`lattice.harvest` 가 작업 디렉터리의 `decart.inp` 를 지운다, lattice.py:346).

### 1.3 238 로 올리고 HGC 게이트 실행

```bash
# 6
scp -P 8022 -r "<LOCAL>/slice_Z_products" USER@HOST_238:~/lpopt_ws/scratch/slice_Z_products
# 7  G-H1 / G-H1b / G-H1c / G-H2  (+ 스크린 곡선이 있으면 G-H4)
ssh -p 8022 USER@HOST_238 'cd ~/lpopt_ws/src && ../venv/bin/python - <<PY
import json
from lpopt.design.hgc_gates import run_gates_for_file, verdict
root = "../scratch/slice_Z_products"
# G-H4 용 스크린 곡선: S1 이 저장한 (A) 예측 kinf/FF 시리즈. 없으면 None -> G-H4 는
# 공허하게 PASS 하지 않고 SKIP 된다 (run_gates 독스트링).
screen = json.load(open("../scratch/slice_Z/screen_curves.json")) if __import__("os").path.exists("../scratch/slice_Z/screen_curves.json") else {}
for alias in ("T7","T8"):
    res = run_gates_for_file(f"{root}/{alias}/FA_{alias}.HGC", n_gd=20,
                             screen_kinf=screen.get(alias,{}).get("kinf"),
                             screen_ff=screen.get(alias,{}).get("ff"))
    print(alias, verdict(res))
    for r in res:
        print("   ", r.name, r.status, r.detail)
PY'
```

**판정 (사전등록 §4.2)**

| 게이트 | 기대 |
|---|---|
| G-H1 구조 | `%TITL` 334 = DEPL 62 + BRANCH 16×17; `%DIST`/`%MACX`/`%MICX`/`%ADFT` 각 334; 말미 `%FINE` 1 |
| G-H1b 크기 | 정확히 **7,395,955 B** |
| G-H1c 유효성 | `_hgc_looks_valid` + stdout `JOB FINISHED` |
| G-H2 Gd 인구조사 | **20 / 20** |
| G-H4 회귀검사 | BU ≥ 0.2 전 구간 \|Δk\| ≤ **100 pcm**, \|ΔFF\| ≤ **0.0021** |

> **G-H4 FAIL 이면 여기서 멈춘다** (§10.1). 라이브러리를 건드리지 않는다. 실패 설계는 능동학습
> 포인트로 기록만 한다.
> **G-H4 를 돌릴 스크린 곡선이 없으면 G-H4 는 SKIP 되고, 그 사실을 결과 문서에 명기한다.**
> SKIP 을 PASS 로 적지 않는다.

---

## 2. S5 — 스냅샷 → 등록 → 라이브러리 재빌드 (**전부 199**)

### 2.1 S0 스냅샷 (재빌드 전, 되돌릴 수 있는 유일한 지점)

```powershell
# 8  199 로 산출물 전송 (hgc/ 스테이징 원본)
scp -r "<LOCAL>/slice_Z_products" USER@HOST_199:C:/Users/USER/lpopt_work/kit_frontier/data/design/incoming_slice_Z
# 9  스냅샷.  ※ 199 에 E: 는 없다 -> D:\lpopt_archive_199 를 쓴다 (결정 D-2)
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt.design.library snapshot data/design/package D:/lpopt_archive_199/pkg_snapshots --tag slice_Z_20260903"
# 10  즉시 검증 (스냅샷이 유효하지 않으면 재빌드 금지)
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt.design.library verify data/design/package D:/lpopt_archive_199/pkg_snapshots/slice_Z_20260903"
#     기대 출력: "OK"
```

`snapshot` 은 `lib/ + bases/ + cores/ + registry.json + designs.json` 을 아카이브하고 매니페스트를
남긴다 (library.py `SNAPSHOT_MEMBERS`). **이 스냅샷이 `.bak` 한 세대를 대체하는 유일한 롤백이다.**

### 2.2 T7/T8 등록 + hgc 스테이징 + TotalBatcher 재빌드 (한 호출)

`assemble_package(pkg, sources, registry, apr1400_root, snapshot_dir=…)` 이
① `designs.json` 재작성 → ② `hgc/` 스테이징(`.HGC`/`.out`/`.sum`/`dec_FA_*.inp`) →
③ `build_library_from_sources` → `build_master_library` (TotalBatcher) 를 한 번에 한다.
`snapshot_dir` 은 **매니페스트를 다시 쓰기 전에** 검사되므로 반드시 넘긴다.

```powershell
# 11  199 에서. 스크립트 본문은 아래 그대로 (파일로 저장 후 실행 권장:
#     C:\Users\USER\lpopt_work\kit_frontier\s5_assemble_slice_Z.py)
```

```python
# s5_assemble_slice_Z.py  --  199 에서 실행. 인자 없음.
from pathlib import Path
from lpopt.design.spec import DesignRegistry, FuelDesign
from lpopt.design.package import DesignSource, assemble_package

PKG  = Path("data/design/package")
INC  = Path("data/design/incoming_slice_Z")
APR  = Path("../0_APR1400")

# 사전등록 §1.1 의 동결 튜플. e2 는 정확값(4.70/4.25)으로 기록한다 -- type_id 는
# 0.1 w/o 로 양자화되므로(spec.py:78) designs.json 만이 4.6750 변종과 구별한다.
Z1 = FuelDesign(e1=5.50, e2=4.70, zoning_variant="z1", gd_wt=8,  n_gd=20, gd_u_enr=4.0)
Z2 = FuelDesign(e1=5.00, e2=4.25, zoning_variant="z1", gd_wt=10, n_gd=20, gd_u_enr=4.0)
assert Z1.type_id == "P5547Z1G08N20", Z1.type_id
assert Z2.type_id == "P5042Z1G10N20", Z2.type_id

reg = DesignRegistry.load(PKG / "registry.json")      # designs.json 자동 hydrate (R23 가드)
a1, a2 = reg.alias(Z1), reg.alias(Z2)
assert (a1, a2) == ("T7", "T8"), f"alias pool broke: {a1}/{a2}"   # 사전등록 §1.1 중단점

EXTRA = {
  "T7": dict(gd_positions="1:1;4:1;6:4", layout="1:1;4:1;6:4",
             base_template="0_APR1400/5.8_5.1/FA/IGD_20/8_20_z1/dec_FA_B03.inp",
             xenon_mode="TR", density=9.95, provenance="on_demand_slice_Z",
             screen_pattern="PB"),
  "T8": dict(gd_positions="1:1;4:1;6:4", layout="1:1;4:1;6:4",
             base_template="0_APR1400/5.8_5.1/FA/IGD_20/10_20_z1/dec_FA_B05.inp",
             xenon_mode="TR", density=9.88, provenance="on_demand_slice_Z",
             screen_pattern="PB"),
}
# screen_ff / screen_k0 / screen_crossing_bu / screen_model_sha / decart_wall_s /
# hgc_sha256 / deck_sha256 는 S1 산출과 manifest.json 에서 채운다 (빈 값 금지).

srcs = []
for des, al in ((Z1, "T7"), (Z2, "T8")):
    d = INC / al
    srcs.append(DesignSource(design=des, alias=al,
                             hgc_path=d / f"FA_{al}.HGC",
                             out_path=d / f"FA_{al}.out",
                             sum_path=d / f"FA_{al}.sum",
                             deck_path=d / f"dec_FA_{al}.inp"))

build = assemble_package(PKG, srcs, reg, APR,
                         snapshot_dir="D:/lpopt_archive_199/pkg_snapshots/slice_Z_20260903",
                         require_gd_positions=True)
reg.save(PKG / "registry.json")
print("COMP", build.comp_count, "REFL", build.refl_count, "ncomp", build.ncomp)
print("sets", build.set_names)
```

> **주의 — 위 스크립트의 `EXTRA` 는 아직 `assemble_package` 에 배선되어 있지 않다.**
> `design_record(source, extra=…)` 가 extra 를 받지만 `assemble_package` 는 그것을 노출하지 않는다
> (package.py:616-635). 두 선택지 중 하나를 오너가 고른다:
> (a) `write_designs_manifest` 를 직접 호출해 extra 를 넘기고, 그 다음 `build_library_from_sources`
>     를 별도 호출한다 (코드 변경 0, 호출 2회) — **권장**;
> (b) `assemble_package` 에 `extra` 인자를 추가한다 (코드 변경 1줄).
> **(a) 를 쓰면 `snapshot_dir` 가드를 `write_designs_manifest` 앞에 직접 `require_snapshot(PKG, snap)`
> 로 재현해야 한다.** 결정 D-6.

### 2.3 라이브러리 게이트 G-H3 계열 (재빌드 직후, 캠페인 전)

```powershell
# 12
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -c \"
from pathlib import Path
from lpopt.design.library import gate_library_sizes, gate_comp_rosters, gate_comp_order, expected_library_sizes
lib=Path('data/design/package/lib')
xsl=(lib/'MAS_XSL'); hff=(lib/'MAS_HFF')
print('expected', expected_library_sizes(39))
print('actual  ', xsl.stat().st_size, hff.stat().st_size)
gate_library_sizes(xsl.stat().st_size, hff.stat().st_size, 39)   # G-H3, 허용오차 없음
t=xsl.read_text(errors='replace')
gate_comp_rosters(t, ['FA_T7','FA_T8'])                          # G-H3b
print('G-H3 / G-H3b OK')
\""
```

| 게이트 | 기대 (등식, 허용오차 없음) |
|---|---|
| **G-H3** | `MAS_XSL` = 2,010 + 385,849 × 39 = **15,050,121 B** · `MAS_HFF` = 404,857 × 39 = **15,789,423 B** |
| **G-H3b** | 새 `COMP FA_T7`/`FA_T8` 의 핵종 로스터가 기존 블록과 정확히 일치 (`BP01*`/`SB10*`/`MACX*`/`CRD1*` 포함) ∧ COMP 헤더 `BURN VAR DMOD ADF DUM = 62 17 6 0 0` |
| **G-H3c** | 기존 37 prefix 순서 불변 + `T7`/`T8` 이 뒤에 append (`gate_comp_order(before, after)`; `before` 는 **재빌드 전에** `comp_blocks(old_xsl).keys()` 로 떠 둔다 — 재빌드 후에는 복원 불가) |

> **G-H3c 는 재빌드 전 준비가 필요하다.** 11번 실행 전에
> `python -c "from lpopt.design.library import comp_blocks; print(list(comp_blocks(open('data/design/package/lib/MAS_XSL',errors='replace').read()).keys()))"`
> 의 출력을 파일로 남긴다. 남기지 않으면 G-H3c 는 영구히 검사 불가다.

**FAIL 시:** 중단 + 스냅샷 롤백(`library.py` 아카이브 복원). `.bak` 은 이미 소모되었을 수 있다.

---

## 3. S5b — 코어 템플릿 재생성 · synth_decks purge (199)

```powershell
# 13  DRY RUN 먼저. 절대 생략하지 않는다.
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt.design.library regen data/design/package --dry-run --synth-root data/design/synth_decks"
# 14  실행 (dry-run 출력이 기대와 일치할 때만)
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt.design.library regen data/design/package --synth-root data/design/synth_decks"
```

- `regen` 은 `bootstrap.library_aliases(pkg)` 로 **39-alias 로스터**를 읽고 `cores/` 10개 폴더의
  템플릿을 `write_core_template` 로 다시 쓴다.
- `%GEN_DIM` : `10 10 27 40 42` → **`10 10 27 42 44`** (nbatch +2, ncomp +2).
- `synth_decks/` 캐시는 `--synth-root` 를 주면 purge 된다. `--no-purge-synth` 는 **쓰지 않는다** —
  캐시된 40/42 덱이 남으면 `_resolve_template`(assets.py:718-760)이 차원 검사 없이 그것을 먼저 쓴다.
- **`T7_T8` 는 `cores/` 에 아직 없다.** `regen` 은 기존 폴더만 갱신하므로 새 pair 는 별도로 만든다:

```powershell
# 15  cores/T7_T8 생성 (S6 부트스트랩이 restart 를 쓰기 전이므로 restart_basename 은
#     부트스트랩이 만들 이름을 넣거나, 부트스트랩 직후에 다시 write_core_template 한다)
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -c \"
from lpopt.design.package import write_core_template
from lpopt.design.bootstrap import library_aliases
al = library_aliases('data/design/package')
assert len(al)==39, al
p = write_core_template('data/design/package', 'T7_T8', 121, al, 'MAS_RST.APRQ_XX_XXXX.XX')
print(p)
\""
```

**S5b 통과 판정 (사전등록 §4.4):** 재생성 후 **기존 paramA 쌍 10개 전부**가 `validate_reload_deck`
을 통과한다. 하나라도 FAIL 이면 **캠페인 전에 멈춘다**.

```powershell
# 16  검증.  시그니처: validate_reload_deck(deck_text, restart_basename, *, expected_dims=…)
#     expected_dims 의 모듈 기본값 LIBRARY_DIMS 는 (83, 85) -- **ga80 값이다.**
#     paramA 는 반드시 39-타입에서 유도한 (42, 44) 를 명시적으로 넘긴다.
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -c \"
import glob, re
from lpopt.search.assets import validate_reload_deck
from lpopt.design.coredeck import library_dims
from lpopt.design.bootstrap import library_aliases
n = len(library_aliases('data/design/package')); dims = library_dims(n)
print('n_aliases', n, 'expected_dims', dims)      # 39 -> (42, 44)
assert n == 39 and dims == (42, 44), (n, dims)
bad = []
decks = sorted(glob.glob('data/design/package/cores/*/*/MAS_INP_cy*.inp'))
for f in decks:
    t = open(f, encoding='utf-8').read()
    m = re.search(r'MAS_RST\.[^\s#]+', t)
    try: validate_reload_deck(t, m.group(0) if m else '', expected_dims=dims)
    except Exception as e: bad.append((f, str(e)))
print('n_decks', len(decks)); print('FAIL', bad if bad else 'none')
\""
```

> **등록된 코드 사실 (확인 완료).** `resolver.paramA_library_dims` 는
> `len(library_aliases(pkg))` 로 타입 수를 세어 `coredeck.library_dims(n)` 을 돌려주므로
> (resolver.py:80-86), **런타임 검증기는 39-타입 패키지에서 자동으로 (42, 44) 를 쓴다 —
> 코드 변경이 필요 없다.** 반면 `assets.LIBRARY_DIMS = (83, 85)` 는 ga80 기본값이므로,
> 위처럼 손으로 부를 때는 `expected_dims` 를 반드시 넘긴다. 넘기지 않으면 정상 덱이 FAIL 로 보인다.

---

## 4. S6 — MASTER 스모크 + 부트스트랩 (199)

덱: **`produce_sliceZ_bootstrap_199.inp`** (신규, 리포 루트).

```powershell
# 17  전송
scp produce_sliceZ_bootstrap_199.inp USER@HOST_199:C:/Users/USER/lpopt_work/kit_frontier/
# 18  (1) 스모크 — T3_T4 부터. 사전등록 §5: 39-COMP MASTER 스모크의 최소 단위.
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt design bootstrap --input produce_sliceZ_bootstrap_199.inp --pair T3_T4 --feed 121"
# 19  (2) 아암 A 셀
... --pair T7_T8 --feed 121
# 20  (3) 아암 B 셀
... --pair T6_T4 --feed 121
# 21  (4) regen 이 stale 로 지목한 pair
... --pair T5_T6 --feed 121
```

**18번을 끝내고 멈춘다.** G-H5a/b/c·G-H6 을 읽기 전에 19번으로 넘어가지 않는다.

| 게이트 | 검사 | 도구 |
|---|---|---|
| **G-H5a** cy1 덱 | `%GEN_DIM` 차원 ∧ `%LPD_BCH` 로스터 ∧ `%LPD_C&X`/`%LPD_HFF` 이름이 `MAS_XSL`/`MAS_HFF` 에 존재 | `library.gate_cycle1_deck(deck, aliases)`. **`validate_reload_deck` 금지** — `%LPD_BCH` 덱을 구조적으로 거부한다 (assets.py:325-327) |
| **G-H5b** reload 덱 | cy ≥ 2 덱 | `library.gate_reload_deck(deck, restart_basename)` |
| **G-H5c** 수렴 | `bootstrap_max_cycles = 16` 안에서 5-FOM 이 `consecutive = 2` 회 연속 안정 | `library.gate_convergence(converged=…, n_cycles=…)` · T6_T4 선례 11 사이클 |
| **G-H6** 덱 에코 | MASTER 출력이 `T7`/`T8` 을 실제 set 이름으로 부른다 | `master_work/*/MASTER.stdout` grep. **19번(T7_T8)에서 가장 중요** |

> **부트스트랩 범위는 결정 D-3 에 걸려 있다.** 라이브러리 재빌드는 **모든 paramA restart** 를
> 무효화하므로 원칙적으로 `bases/` 31 pair + `T7_T8` = **32 회**다. 슬라이스가 실제로 필요로 하는
> 것은 위 4 pair 뿐이다. 32 회를 돌리면 일정이 **일 단위**로 늘어난다(단일 실패 관측 8,744 s).
> **오너가 4 인지 32 인지 정한다.** 4 를 고르면, 나머지 27 pair 를 쓰는 다른 캠페인은 이 슬라이스
> 이후 **restart 무효 상태**임을 워크스페이스에 기록해야 한다.

### 4.1 S6b — 재-ingest · 동기화

```powershell
# 22  fuel_types 재빌드 (paramA 행 37 -> 39)
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -c \"
from lpopt.design.package import ingest_fuel_types
print(ingest_fuel_types('data/design/package'))
\""
# 23  통과 판정: data/store/fuel_types.parquet 의 paramA 행이 39
```

`paramA_rows`(fuel_types.py:1625-1690)는 `designs.json` **과** `hgc/FA_*.out` 둘 다 요구한다 —
S5 의 `stage_hgc` 가 `.out` 을 스테이징했는지 먼저 확인한다.

### 4.2 S8 — 콜드스타트 해소 (`T7_T8` 전이 스윕)

```powershell
# 24
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe fr_transfer.py --target-pair T7_T8 --k 32"
# 25  머지 후 영수증을 남긴다. 아암 A 런처가 이 파일의 존재를 게이트한다.
#     data\reports\fr_transfer_T7_T8_merged.json
#     {"pair":"T7_T8","feed":121,"k":32,"n_rows_merged":<N>,"store_sha256":"<...>","ts":"<...>"}
```

**자산 사다리는 레벨 0(native) 고정** — 부트스트랩이 만든 자기 restart를 쓴다. `pair_feed:` /
`promoted:` provenance 는 폴백이 아니라 **결함**이다 (r1 ga80 레벨-3 사고).

---

## 5. S7 — 캠페인 2 arm (199)

> ### ★ 5.0 개명 (2026-09-03 집행) — `Z1_Z2` → **`T7_T8`**
>
> 사전등록 **부록 P.6** 의 미처분 항목을 해소했다. `Z1`/`Z2` 는 사전등록 §1.1 스크립트의
> **FuelDesign 변수명**일 뿐이고 레지스트리 alias 가 아니다 (`BootstrapError: pair type 'Z1'
> not in library aliases`). 아암 A 의 정본 셀은 **`T7_T8`** (feed 121, paramA 39-type) 이다.
> 이 런북의 모든 `Z1_Z2` / `Z1Z2` / `z1z2` 표기는 `T7_T8` / `T7T8` / `t7t8` 로 교체되었다.
>
> | 이전 (SUPERSEDED, 삭제하지 않음) | 정본 |
> |---|---|
> | `fpcamp_minfr_Z1Z2_f121_sliceZ_199.inp` | `fpcamp_minfr_T7T8_f121_sliceZ_199.inp` |
> | `launch_fpcamp_minfr_Z1Z2_f121_sliceZ_199.ps1` | `launch_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1` |
> | `run_fpcamp_minfr_Z1Z2_f121_sliceZ_199.bat` | `run_fpcamp_minfr_T7T8_f121_sliceZ_199.bat` |
> | `status_fpcamp_minfr_Z1Z2_f121_sliceZ_199.ps1` | `status_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1` |
> | `pinbu_wave_sliceZ_Z1Z2_199.inp` | `pinbu_wave_sliceZ_T7T8_199.inp` |
> | 런 디렉터리 `…_z1z2_…` | `D:\lpopt_archive_199\runs\fpcamp_minfr_t7t8_f121_slicez` |
> | restart base `bases\Z1_Z2` | `bases\T7_T8` |
> | 전이 영수증 `fr_transfer_Z1_Z2_merged.json` | `fr_transfer_T7_T8_merged.json` |
>
> 옛 5종 파일은 **삭제하지 않고** 첫 줄에 `SUPERSEDED by T7T8 (alias registry, 부록 P)` 주석을
> 붙여 두었다. **199 로 전송하지 않는다.**
>
> 아암 B 런처(`launch_fpcamp_minfr_T6T4_…ps1`)의 아암-A 로그/rc 게이트 경로도 함께
> `fpcamp_minfr_t7t8_f121_slicez_{out.log,rc.txt}` 로 고쳤다 — 고치지 않으면 그 게이트가
> 조용히 발화하지 않는다. **아암 B 덱 본문은 무변경**(sha 핀 유지, 아래 표); 덱 헤더 주석에
> 남은 `Z1_Z2` 상호참조는 **미처분**으로 남긴다(아래 §5.5).

파일 6종을 kit 루트로 보낸다.

```powershell
# 26
scp fpcamp_minfr_T7T8_f121_sliceZ_199.inp fpcamp_minfr_T6T4_f121_sliceZ_199.inp `
    run_fpcamp_minfr_T7T8_f121_sliceZ_199.bat run_fpcamp_minfr_T6T4_f121_sliceZ_199.bat `
    launch_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1 launch_fpcamp_minfr_T6T4_f121_sliceZ_199.ps1 `
    status_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1 status_fpcamp_minfr_T6T4_f121_sliceZ_199.ps1 `
    USER@HOST_199:C:/Users/USER/lpopt_work/kit_frontier/
```

### 5.1 RESTAMP (발사 전 필수)

두 런처의 스토어 핀은 **슬라이스 이전** 값(`16E311AF…917BA` / 22,810,322 B)이 박혀 있고,
그 자리에 `# RESTAMP` 주석이 붙어 있다. S6b·S8 머지 후 값이 반드시 바뀌므로 **그대로 두면 REFUSED 된다
(설계된 동작)**.

```powershell
# 27  재해시
ssh USER@HOST_199 "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 C:\Users\USER\lpopt_work\kit_frontier\data\store\records.parquet).Hash; (Get-Item C:\Users\USER\lpopt_work\kit_frontier\data\store\records.parquet).Length\""
# 28  launch_..._T7T8_....ps1 의 $wantStore / $wantStoreLen 두 줄을 교체하고,
#     같은 편집에서 사전등록 부록 L 에도 기록한다.
# 29  아암 B 의 핀은 "아암 A 머지 후" 의 값이다. 아암 A 와 같은 값을 넣으면 REFUSED 된다.
```

덱 sha256/길이 핀 (저작본 기준, 이진 전송 전제):

| 파일 | sha256 | 바이트 |
|---|---|---|
| `fpcamp_minfr_T7T8_f121_sliceZ_199.inp` | `57E2B8291E6E457D2D09C6902FD7739AC45EBB490E2DA807F7457C343E5FDAD0` | 23,020 |
| `fpcamp_minfr_T6T4_f121_sliceZ_199.inp` | `1C38D5E51169DEAD29EC0C20DAF031C3DE67A0B1498981B16652E4A5F812FFAC` | 9,241 |

전송으로 CRLF 가 붙으면 해시가 바뀌고 게이트가 걸린다 — **게이트를 푸는 게 아니라 이진 전송으로 다시 보낸다.**

### 5.2 발사 (아암 A 먼저, 순차)

```powershell
# 30  아암 A
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1"
# 31  감시 (읽기 전용, 임의 횟수)
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_fpcamp_minfr_T7T8_f121_sliceZ_199.ps1"
# 32  A 종료(rc 파일 생성) 후 스토어 머지 -> 재해시 -> 아암 B 런처 RESTAMP -> 아암 B
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_minfr_T6T4_f121_sliceZ_199.ps1"
# 33  감시
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_fpcamp_minfr_T6T4_f121_sliceZ_199.ps1"
```

런 디렉터리는 **`D:\lpopt_archive_199\runs\fpcamp_minfr_{t7t8,t6t4}_f121_slicez`** (C: 42 GB 는
`harvest_maps` 보존분 200 체인을 감당하지 못한다).

**두 런처의 busy 게이트는 다른 lpopt 프로세스가 살아 있는 동안 무조건 거부한다.** 아암 B 런처는
추가로 "아암 A 가 rc 파일을 썼는가" 를 본다.

### 5.3 캠페인 중 기대/금지 배너

- 기대: `min_fr` objective 배너, `HARD F_r <= 1.550`, 예측 pin 게이트 78.
- **정상**: `post_verify_done = false`, `post_verify_calls = 0` (campaign.py:533-534 — flat_power 밖에서는
  전달 페이로드가 `None`). 크래시가 아니다.
- **정상**: 모든 행이 `deliverable = false`, `unknown_axes = ("max_pin_burnup",)` (verify.py:851).
- **금지**: `min_fxy objective` 또는 `[optimize][F_xy PROXY]` — 다른 실험이다. 즉시 중단.

### 5.4 ★ 등록된 코드 결함 D-FXY — 덱의 `f_xy_limit = 1.65` 는 **이 objective 에서 작동하지 않는다**

`campaign.feasibility_limits_for` 는 `limits["f_xy"]` 를 **`min_fxy` 와 `flat_power` 에서만** 채운다
(campaign.py:330-334, 352-359). `min_fr_max_cycle` 에서는 `None` 으로 남아 **F_xy 는 보고 열일 뿐
어떤 후보도 기각하지 않는다**. 사전등록 §7.2.1 은 이것을 "하드 제약" 이라고 적었다 — **코드와 불일치다.**

**처분 (이 런북의 등록):**
1. 덱은 `f_xy_limit = 1.65` 를 **그대로 적되** 헤더에 무력함을 명기한다 (덱의 D-FXY 블록).
2. `harvest_maps = true` 가 전 수렴 행의 **측정** F_xy 를 남기므로, **결과 문서가 1.65 스크린을
   사후 적용**한다. 예측 게이트보다 강한 판정이다.
3. `status_*.ps1` 은 `f_xy` 가 없는 수렴 행을 **Tier-1 판정 불가**로 경고한다.
4. 네이티브 게이트가 필요하면 그것은 **코드 변경**이며 이 슬라이스의 범위 밖이다 (결정 D-5).

### 5.5 개명 후 남은 RESTAMP · 미처분 항목 (발사 전 확인)

| # | 항목 | 상태 | 처분 |
|---|---|---|---|
| R-1 | `launch_..._T7T8_...ps1` 의 `$wantStore` / `$wantStoreLen` | **RESTAMP 플레이스홀더**(`16E311AF…917BA` / 22,810,322 B) | S6/S7 부트스트랩·S8 전이 머지 후 §5.1 의 27번 명령으로 재해시 → 두 줄 교체 + 사전등록 부록 L 기록 |
| R-2 | `launch_..._T6T4_...ps1` 의 스토어 핀 | 동일 플레이스홀더 | **아암 A 머지 후** 값. 아암 A 와 같은 값을 넣으면 REFUSED |
| R-3 | 아암 B 덱 `fpcamp_minfr_T6T4_f121_sliceZ_199.inp` 헤더 주석의 `Z1_Z2` 상호참조 (12·17행) | **미수정** — 덱 본문을 고치면 sha 핀(`1C38D5E5…FFAC` / 9,241 B)이 깨진다 | 주석만의 결함. 고치려면 덱·핀·런북 표를 **한 편집에서** 함께 갱신할 것 |
| R-4 | `produce_sliceZ_bootstrap_199.inp` 의 `[case] pair = "Z1_Z2"` (145행) | **미수정** — 이 경로에서 INERT(`--pair` CLI 가 정본)이고, S6 부트스트랩이 **현재 199 에서 이 덱으로 진행 중**(sha `7FB7594B…265F` 고정) | S6 종료 후에 고친다. 진행 중 편집 금지 |
| R-5 | `fr_transfer_T7_T8_merged.json` 영수증 | 미생성 (S8 미실행) | §4.2 의 24·25번 |
| R-6 | `pinbu_wave_sliceZ_T7T8_manifest_<date>.{json,csv}` | 미생성 (캠페인 종료 후에만 가능) | §7 의 36·37번 |

---

## 6. S9b — 독립 MTC 단계 (캠페인 종료 후, arm 마다 1회)

```powershell
# 34  아암 A
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt sdm-mtc --run D:/lpopt_archive_199/runs/fpcamp_minfr_t7t8_f121_slicez --input fpcamp_minfr_T7T8_f121_sliceZ_199.inp --top-k 5"
# 35  아암 B
ssh USER@HOST_199 "cd C:\Users\USER\lpopt_work\kit_frontier && C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt sdm-mtc --run D:/lpopt_archive_199/runs/fpcamp_minfr_t6t4_f121_slicez --input fpcamp_minfr_T6T4_f121_sliceZ_199.inp --top-k 5"
```

- 이 경로(cli.py:680)는 `candidates_from_delivery` 를 먼저 시도하고 **비면
  `select_topk_feasible` 로 폴백**한다(cli.py:741-746) → `min_fr_max_cycle` 에서도 **실제로 돈다**.
- 덱의 `[constraints]` 양 edge 가 설정되어 있으므로 `limits.mtc_gated() == True` → **PASS/FAIL 판정**.
- 산출: `<run>/sdm_mtc_report.md` (+`.csv`), `data/sdm_mtc/results.jsonl`.
- 예산: 축 1개 × 후보 5 ≈ **arm 당 5 콜**, 2 arm = **~10 콜**. 탐색 예산과 분리 보고.
- SDM 은 `campaign._rod_model()` 이 설계상 `None` → **INCONCLUSIVE**. 덱은 `sdm_enable = false`.

> **★ 선택 순서 함정 (사전등록 §7.3).** `select_topk_feasible` 은 후보를 **`|cyclen − 625|` 근접순**
> 으로 정렬한다(sdm_mtc.py:1259-1261) — `F_r` 순도 `F_xy` 순도 아니다.
> **34/35 실행 후 PRIMARY 후보(최소 측정 `F_r`)가 top-5 안에 있는지 반드시 확인**하고, 없으면
> 포함될 때까지 `--top-k` 를 올려 재실행한다 (+후보당 1 MASTER 콜). **중단 사유는 아니다.**
> `[sdm_mtc]` 의 `scram_banks` 기본값(R1–R5 + B + A)을 쓴다. 프로그램적 훅
> `post_verify_topk` 의 R-only 기본값은 config 자신이 결함이라고 적은 설정이다 (config.py:1295-1303).

---

## 7. S9c — phase-2 측정 pin 웨이브 (arm 당 top-20)

덱 스텁: `pinbu_wave_sliceZ_T7T8_199.inp` / `pinbu_wave_sliceZ_T6T4_199.inp` (신규, 리포 루트).
**둘 다 매니페스트 없이는 돌지 않는다.**

```bash
# 36  매니페스트 작성 (238, MASTER 0). 정렬키 = 측정 F_r 오름차순 (사전등록 §7.6)
#     joint-clean 행 중 top 20 + 1위 코어 5회 복제 = 25 엔트리
scp USER@HOST_199:D:/lpopt_archive_199/runs/fpcamp_minfr_t7t8_f121_slicez/labels.jsonl ./
#     -> data/reports/pinbu_wave_sliceZ_T7T8_manifest_<YYYYMMDD>.{json,csv}
# 37  record_id 재현 검증 20/20 (`record_id_minted_ok`) -- 지출 전에. r2 선례.
```

```bat
REM 38  199. 두 패스. 패스 2 는 --force 필요 (resume 집합이 record_id 키)
"%PY%" -u pinbu_wave.py run --plan data/reports/pinbu_wave_sliceZ_T7T8_manifest_<YYYYMMDD>.json ^
   --deck pinbu_wave_sliceZ_T7T8_199.inp ^
   --run-dir D:/lpopt_archive_199/runs/pinbu_wave_slicez_t7t8 ^
   --group slicez_A_top20
"%PY%" -u pinbu_wave.py run --plan ... --deck ... --run-dir ... --group slicez_A_replicate5 --force
REM 39  아암 B 동일 (…_T6T4_…, 그룹 slicez_B_top20 / slicez_B_replicate5)
REM 40  머지는 별도 단계: pinbu_wave.py patch --dry-run  ->  patch
```

- **PRIMARY 의 pin 축은 예측 78 이 아니라 측정 ≤ 80 으로 판정한다** (사전등록 §7.6).
- 아암 A 의 pin 예측은 T7/T8 학습행 0 개 위에 서 있다 — 2.0 마진이 **무보정**이다
  (감사의 무이웃 사례는 −5.93 저추정, pinbu_audit_20260820.md §4.3-4.4). 측정이 필수인 이유.
- 아암 B 에도 웨이브를 도는 이유: PRIMARY 판정축을 대조군과 **같은 방식으로** 재야 하고,
  T6_T4/f121 은 유일하게 측정 pin 편향(+3.42)이 관측된 셀이라 그 편향이 셀의 성질인지
  옛 restart 의 성질인지가 여기서 처음 갈린다.

---

## 8. 산출 · 결과 문서로 넘길 것

사전등록 §10.2 의 7항목 + 이 런북이 추가로 요구하는 것:

- **G-H4 가 SKIP 이었으면 SKIP 이라고 적는다** (PASS 아님).
- **G-H3c 의 before-roster 를 재빌드 전에 떴는지** 여부.
- **부트스트랩을 4 pair 로 했는지 32 pair 로 했는지**, 4 라면 나머지 27 pair 의 restart 가 무효라는 문장.
- **D-FXY**: F_xy 1.65 는 엔진이 아니라 결과 문서가 적용했다는 문장과, 그 스크린에서 탈락한 행 수.
- **cy1 캡 프로토콜**이 기존 `bases/` 와 같은지 다른지 (다르면 `R_Fr` 이 프로토콜 변화를 흡수한다).
- s1j 그림자 F_xy 예측은 **238 오프라인 산출** (덱 노브 없음, §9-D5).

---

## 9. ★ S6 이전에 오너가 결정할 것 (전부 미해결)

| # | 결정 | 왜 지금 | 결정 안 하면 |
|---|---|---|---|
| **D-1** | **T7 UO2G 캐리어 밀도 9.95 vs 형제 9.88** | 부록 L.1a: `IGD_20` 6개 덱 전부가 `UO2G 9.88`인데 사전등록은 T7 을 9.95 로 동결했고, 저작 시 B03 의 밀도 한 줄을 패치했다. 기존 `gd_wt 8 × n_gd 20` 형제(`P2 = P6253Z1G08N20`, `P3 = P5853Z2G08N20`)는 **9.88 로 실현되어 있다**. → **T7 은 라이브러리 형제와 캐리어 밀도가 다르다.** 사전등록은 따랐고 선례는 어겼다 | 이미 DeCART 를 돌렸으므로 되돌리려면 **S3 재실행**(0.86 h)이다. S5 전에 결정하지 않으면 라이브러리에 들어간 뒤에야 논쟁이 된다. **비차단이지만 S5 전에 서면 확인 필요** |
| **D-2** | **스냅샷 목적지** — 사전등록 §4.1 은 `E:\lpopt_archive\slice_Z_20260903\` 인데 **199 에 E: 드라이브가 없다** (C: 42 GB / D: 6.2 TB만) | 스냅샷은 재빌드의 **유일한 롤백**이다 (`.bak` 한 세대는 재빌드가 소모) | 명령 9번이 실패하거나, 더 나쁘게 C: 에 떨어져 42 GB 를 먹는다. 런북은 `D:\lpopt_archive_199\pkg_snapshots` 를 제안 — **승인 필요** |
| **D-3** | **부트스트랩 범위 4 pair 인가 32 pair 인가** — 사전등록은 "8+1 = 9" 인데 그것은 **로컬 체크아웃의 `bases/` 수**다. 199 정본은 **31 pair** | 재빌드는 **모든** paramA restart 를 무효화한다. 32 회면 §9 예산(2–5 h)이 **일 단위**로 틀린다 | 일정이 근본적으로 틀린 채 시작한다. 4 를 고르면 나머지 27 pair 를 쓰는 캠페인이 조용히 무효 restart 위에서 돈다 — 그 사실을 워크스페이스에 기록해야 한다 |
| **D-4** | **`cy1_cap_efpd` 를 쓸 것인가** (전 pair 통일 필수) | 아암 B 의 `R_Fr` 은 **재빌드 효과**여야 한다. T6_T4 만 캡을 새로 걸면 `R_Fr` 이 프로토콜 변화를 흡수해 사전등록이 정의한 수가 아니게 된다 | 대조군이 무의미해진다. 덱은 **미설정**이 기본 |
| **D-5** | **D-FXY 처분** — `f_xy_limit` 를 `min_fr_max_cycle` 에서 살릴 것인가 | 사전등록 §7.2.1 은 하드 제약이라 적었으나 `feasibility_limits_for` 는 그 축을 채우지 않는다 | 런북 §5.4 의 사후 스크린으로 간다(코드 변경 0). 네이티브 게이트를 원하면 **코드 변경**이고 슬라이스 범위 밖 |
| **D-6** | **`designs.json` extra 필드 주입 경로** — `assemble_package` 는 `extra` 를 노출하지 않는다 (package.py:616-635) | `provenance` / `e2` 정확값 / `screen_*` / `hgc_sha256` / `deck_sha256` 는 사전등록 §6.1 의 필수 기록이고, `gd_positions` 없이는 4.6750 변종이 미래에 T7 을 덮어쓴다 | (a) `write_designs_manifest` + `build_library_from_sources` 2-call (코드 0줄) 또는 (b) `assemble_package` 에 인자 1개 추가. **(a) 권장** |
| **D-7** | **181 fleet 정책 예외의 사후 기록** — `autoeng.toml:59-66` 은 여전히 *"181 is NEVER used"* / `forbidden` 에 181 | **S3 는 이미 181 에서 돌았다** (부록 L, `-PolicyExceptionRef OWNER-TASK-20260903-sliceZ-S3`). 예외가 워크스페이스에 기록되지 않은 채 집행된 상태 | 같은 워크스페이스에 상반된 상시 정책이 남는다. autoeng 자동 엔진이 이 슬라이스를 구동하지는 않지만, 사전등록 A10 이 요구한 기록이 아직 없다 |
| **D-8** | **s1j 그림자 F_xy readout 을 실제로 산출할 것인가** | **덱 노브가 없다** (config 전수 확인: `shadow_v2/v3` 는 정책 풀, `promote_fxy` 는 학습 플래그). 산출하려면 238 에서 `waves/*/selection.json` 을 s1j 로 사후 채점해야 한다 — 별도 작업 | 사전등록 §7.2.4 의 "둘 다 `wave_prereg.json` 에 그림자로 기록" 이 이행되지 않는다. **어떤 게이트도 이것을 잡지 않으므로 슬라이스는 진행 가능** |
| **D-9** | **(A) 서로게이트 접근 (#4a)** 이 아직 미해결 | 구현 보고 §4-1: 238 스테이징이 권한 분류기에 막혔다 → `--self-test` 로그도 체크포인트 SHA 매니페스트도 없다 | **G-H4 의 대조 곡선이 없다** → G-H4 는 SKIP 된다. 사전등록 §10.1 은 G-H4 를 중단점으로 두었으므로, SKIP 을 허용할지 명시 결정 필요 |

---

## 부록 A — 이 런북이 실행하지 않은 것

로컬 연산 0 · DeCART 0 · MASTER 0 · 181 쓰기 0 · 199 쓰기 0.
238 에서 한 것은 **덱 5종의 로더 파싱 + `optimize --dry-run --budget 8` (StubEvaluator, MASTER 0)** 뿐이다.
검증 결과(2026-09-03):

| 덱 | 로더 | dry-run |
|---|---|---|
| `fpcamp_minfr_T7T8_f121_sliceZ_199.inp` | OK · objective `min_fr_max_cycle` · budget 100 · pair T7_T8 · paramA · mtc True/−54/9 · top_k 5 · f_xy 1.65 · pin 78 · λ 1000 · harvest True | **구성 성공**, `status.json` = `{objective: min_fr_max_cycle, case: "T7_T8/feed-121", dry_run: true}` |
| `fpcamp_minfr_T6T4_f121_sliceZ_199.inp` | OK (동일, pair T6_T4) | **구성 성공**, `case: "T6_T4/feed-121"` |
| `produce_sliceZ_bootstrap_199.inp` | OK (`design bootstrap` 경로 — objective 는 미사용 기본값) | 해당 없음 |
| `pinbu_wave_sliceZ_{T7T8,T6T4}_199.inp` | OK | 해당 없음 (매니페스트 필요) |

238 에 `data/design/package` 가 없어도 dry-run 은 StubEvaluator 로 케이스를 구성한다 —
**즉 이 검증은 문법·설정·objective·라우팅 키를 확인한 것이지 자산 해석을 확인한 것이 아니다.**
실 자산 해석은 199 의 런처 게이트(LIBRARY/CORES/BASES)가 본다.
임시 런 디렉터리(`/tmp/sliceZ_dry_*`)와 `src/` 사본은 삭제했고, 프로세스도 종료했다.
덱 5종의 사본이 `238:~/lpopt_ws/scratch/` 에 남아 있다 (읽기 전용 스크래치, 무해).
`data/design/package` · 스토어 parquet · 모델 디렉터리 · sha-pin 하네스 무변경.

### A.1 개명 후 재검증 (238, 2026-09-03 · StubEvaluator, MASTER 0)

개명된 아암 A 덱과 **무변경** 아암 B 덱을 다시 `optimize --dry-run --budget 8` 으로 돌렸다.
전송본 sha256 은 저작본과 일치(이진 전송 확인).

| 덱 | sha256 / 바이트 | 결과 |
|---|---|---|
| `fpcamp_minfr_T7T8_f121_sliceZ_199.inp` | `57E2B829…FDAD0` / 23,020 | **RC 0** · `[optimize] campaign sliceZ_dry_t7t8 case=T7_T8/feed-121 budget=8 spent=0 dry_run=True` · `wave 0 size=8 spent=8/8 conv=8 feas=1 on_target=1 gate=explore+ tau=0.30` · `RESULT: complete — 1 waves, budget 8/8, 1 feasible / 1 on-target; best FEASIBLE F_r 1.463 (<= 1.55) @ cyclen 628.1 EFPD` |
| `fpcamp_minfr_T6T4_f121_sliceZ_199.inp` | `1C38D5E5…FFAC` / 9,241 (무변경) | **RC 0** · `case=T6_T4/feed-121` · `gate=objective-` · `best FEASIBLE F_r 1.434 @ cyclen 633.3 EFPD` |

- `status.json.case` 가 **`T7_T8/feed-121`** 로 찍힌다 — 개명의 핵심 확인점.
- 아암 A 의 dry-run 패턴이 `F:T7:0` / `F:T8:0` 로 신연료를 배치한다 — alias 해석이 정상.
- 양 arm 모두 `[optimize] SDM/MTC gate configured but not run here (dry-run / no [master].executable); top_k=5 carried for the live run` — §6 의 독립 단계 전제와 일치.
- **dry-run 의 F_r 값(1.463 / 1.434)은 StubEvaluator 산출이며 물리적 의미가 없다.** 마크 판정에 쓰지 않는다.

> **★ 새로 관측된 배너 (등록).** 두 arm 모두 첫 줄에
> `[optimize][DEPRECATED] objective='min_fr_max_cycle' is a RETIRED production mode
> (flatness-first program §10 STOP): it steers the search by F_r. Kept runnable for
> reproduction / A-B baselines only — use objective='flat_power' for production.`
> 를 찍는다. **금지 배너가 아니다** — 런북 §5.3 의 금지 목록(`min_fxy objective`,
> `[optimize][F_xy PROXY]`)에 해당하지 않고 실행도 정상 완료한다. 다만 사전등록이 고정한
> objective 가 코드에서 RETIRED 로 표시되어 있다는 사실은 **결과 문서에 기재해야 한다**
> (사용자 목표 진술 2026-09-03 이 `min F_r` 을 지정했으므로 objective 자체는 변경하지 않는다).

임시 런 디렉터리 `/tmp/sliceZ_dry_{t7t8,t6t4}` 는 삭제했다.
