"""Multi-PC produce kit: export assigned curriculum cells to a portable folder a
SECOND Windows PC can run ``lpopt produce`` on, then merge the results back.

Two operations, exposed as the ``lpopt export-produce-kit`` / ``lpopt merge-store``
CLI commands:

* :func:`export_produce_kit` — build a self-contained kit directory holding a
  generated ``lpopt_kit.inp`` produce deck (one ``[[produce.strata]]`` per assigned
  cell, ``campaign`` / ``name`` == the cell id, pairs auto-selected from the band
  with the *same* :func:`lpopt.curriculum.select_cell_pairs` logic the curriculum
  uses), a copy of ``data/store/fuel_types.parquet``, the shipped ``lpopt`` source
  tree + ``pyproject.toml`` (so PC2 can ``pip install -e .``), and a Korean
  ``KIT_README.md`` with the exact setup / run / return-shipping steps.  The kit
  carries **no** ``records.parquet`` — PC2 starts from a fresh empty store (the
  produce path tolerates a missing store; see :func:`lpopt.search.produce`).

* :func:`merge_store` — read a returned kit ``data/`` folder's
  ``store/records.parquet`` (+ ``maps.npz`` when present) and merge it into the
  main store through the quality-ranked UPSERT writer
  (:meth:`lpopt.data.store.StoreWriter.write_records`), then merge the kit ledger
  into the main ``data/produce/ledger.jsonl`` with ``(record_id, status)`` dedup so
  the main producer's resume counter stays consistent.  Idempotent (re-running is a
  no-op); prints a report of new / upgraded / duplicate rows and per-campaign
  converged counts before/after, flagging any campaign that is not a recognizable
  curriculum cell id.

Generator degradation (the ``elite_perturb`` constraint): the curriculum generator
mix is typically ``{random, heuristic, elite_perturb}``; ``elite_perturb`` needs
converged store elites, which a fresh PC2 store has none of.  The produce path
already degrades ``elite_perturb`` to ``random`` per-draw when no elites exist, but
the kit deck ALSO drops ``elite_perturb`` up front and renormalizes the remaining
weights (:func:`_degrade_generators`), so the exported mix is honest about what PC2
will actually run.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import LpoptConfig
from .curriculum import band_label, cell_id, select_cell_pairs
from .search.genome import fresh_units_from_feed
from .search.resolver import is_paramA_library, paramA_package_root
from .vendor.masterrl.ga import GenomeError

#: Kit-relative dir the bundled paramA design package is copied to (deck
#: ``[design].package_root`` points here so a paramA kit is self-contained).
KIT_DESIGN_PACKAGE_REL = "data/design/package"

#: Kit-relative dir the bundled ga80 MASTER package is copied to (deck
#: ``[verify].package_root`` points here so a ga80 kit is self-contained too).
KIT_GA80_PACKAGE_REL = "FEASIBLE_PACKAGE"

#: The ga80 package subdirs the PRODUCE path actually reads, and which therefore
#: MUST ship whole:
#:
#: * ``lib``   — ``MAS_XSL`` / ``MAS_HFF``; :class:`~lpopt.vendor.masterrl.master.MasterRunner`
#:              stages both into every case work dir and hard-fails without them.
#: * ``bases`` — the restart catalog.  :meth:`CaseAssetResolver._resolve_restart`
#:              scans **every** folder under ``bases/`` and picks by a five-level
#:              ladder (native exact ``<pair>_f<feed>`` -> promoted -> same-pair
#:              nearest feed -> nearest-e_core pair -> neutral).  Shipping a
#:              SUBSET does not merely remove choices, it silently *changes* the
#:              answer: drop ``bases/E1_E2_f117`` and case ``(E1_E2, 117)`` demotes
#:              from a level-0 feed-117 restart to a level-2 feed-121 one, so the
#:              chain restarts a feed-117 core deck from a feed-121 equilibrium and
#:              diverges (``non_finite_flux``).  There is no way to know from the
#:              assigned cells alone which folders the ladder will reach, so the
#:              whole catalog ships.
#: * ``cores`` — the template decks (exact-case and same-pair fallbacks).
#:
#: ``hgc/`` is deliberately NOT shipped: ``FA_*.HGC`` are lattice inputs consumed
#: only when *building* a MASTER library (``lpopt design build-library``); no
#: produce-path code reads them and no packaged deck references them (decks cite
#: only ``MAS_RST.*`` / ``MAS_XSL`` / ``MAS_HFF``).  Skipping it roughly halves
#: the kit.
GA80_PACKAGE_SUBDIRS = ("lib", "bases", "cores")

#: A recognizable curriculum cell id, e.g. ``5.25-5.5_f117`` / ``5-5.25_f125``.
CELL_ID_RE = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?_f\d+$")

KIT_DECK_NAME = "lpopt_kit.inp"


class KitError(ValueError):
    """Raised for an unusable export request (bad cell id, no in-band pairs, ...)."""


# --------------------------------------------------------------------------- #
# cell id <-> (band, feed)
# --------------------------------------------------------------------------- #
def parse_cell_id(cid: str) -> tuple[tuple[float, float], int]:
    """``"5.25-5.5_f101"`` -> ``((5.25, 5.5), 101)`` (inverse of ``cell_id``).

    Rejects a malformed id or an off-grid feed (via :func:`fresh_units_from_feed`).
    """
    text = str(cid).strip()
    m = re.match(r"^(?P<band>.+)_f(?P<feed>\d+)$", text)
    if not m:
        raise KitError(f"unparseable cell id {cid!r} (want '<lo>-<hi>_f<feed>')")
    band_str = m.group("band")
    parts = band_str.split("-")
    if len(parts) != 2:
        raise KitError(f"unparseable band in cell id {cid!r} (want '<lo>-<hi>')")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise KitError(f"non-numeric band in cell id {cid!r}") from exc
    feed = int(m.group("feed"))
    # normalize back through the canonical formatter and re-check round-trip so a
    # sloppy id (e.g. '5.0-5.25_f101') is caught rather than silently reshaped.
    canon = cell_id((lo, hi), feed)
    if canon != text:
        raise KitError(
            f"cell id {cid!r} is not canonical; expected {canon!r} "
            f"(use the curriculum's exact naming)"
        )
    try:
        fresh_units_from_feed(feed)  # raises on an off-1+4N-grid feed
    except GenomeError as exc:
        raise KitError(f"cell id {cid!r}: {exc}") from exc
    return (lo, hi), feed


def is_recognized_cell(campaign: str) -> bool:
    """True if ``campaign`` is a canonical curriculum cell id."""
    if not CELL_ID_RE.match(str(campaign)):
        return False
    try:
        parse_cell_id(str(campaign))
    except KitError:
        return False
    return True


# --------------------------------------------------------------------------- #
# generator degradation
# --------------------------------------------------------------------------- #
def _degrade_generators(gens: dict[str, float]) -> dict[str, float]:
    """Drop ``elite_perturb`` (no elites on a fresh PC2 store) and renormalize.

    Redistributing ``elite_perturb``'s weight proportionally over the survivors is
    exactly a renormalization of the remaining positive weights.  Falls back to
    ``{"random": 1.0}`` when nothing else is configured.
    """
    kept = {
        k: float(v)
        for k, v in (gens or {}).items()
        if k != "elite_perturb" and float(v) > 0.0
    }
    if not kept:
        return {"random": 1.0}
    total = sum(kept.values())
    return {k: round(v / total, 6) for k, v in kept.items()}


# --------------------------------------------------------------------------- #
# per-band library resolution (mirrors CurriculumDriver._band_library)
# --------------------------------------------------------------------------- #
def band_library(cfg: LpoptConfig, band: Sequence[float]) -> str:
    """The effective fuel library for a band (ga80 vs paramA), matching the
    curriculum's :meth:`CurriculumDriver._band_library`.

    An explicit ``[curriculum] band_libraries`` entry (keyed by the canonical band
    label) wins; else bands whose lower edge is >= ``paramA_band_lo`` resolve to
    ``paramA_library`` (the ga80 letter roster has no full-physics types there);
    else ``[curriculum] library`` (ga80).
    """
    curr = cfg.curriculum
    lo, hi = float(band[0]), float(band[1])
    override = dict(getattr(curr, "band_libraries", {}) or {})
    key = band_label(lo, hi)
    if override.get(key):
        return str(override[key])
    if lo >= float(getattr(curr, "paramA_band_lo", 5.75)):
        return str(getattr(curr, "paramA_library", "paramA") or curr.library)
    return curr.library


# --------------------------------------------------------------------------- #
# stratum construction (mirrors curriculum._stratum_for_cell)
# --------------------------------------------------------------------------- #
def build_cell_stratum(
    cfg: LpoptConfig,
    cid: str,
    band: Sequence[float],
    feed: int,
    fuel_library: Any,
    *,
    n_target: int | None = None,
) -> dict[str, Any]:
    """Build the ``[[produce.strata]]`` dict for one assigned cell.

    The cell's library follows the curriculum band rule (:func:`band_library`):
    high bands (>= ``paramA_band_lo``) resolve to ``paramA`` so the kit produces
    against the design package, low bands stay ga80.  Pairs are auto-selected from
    the band with the same :func:`lpopt.curriculum.select_cell_pairs` logic the
    curriculum uses; ``allow_single_cycle_discharge`` follows the feed>121 rule
    (``n_fresh > 30``); generators come from ``[curriculum].generators`` after
    :func:`_degrade_generators`.
    """
    curr = cfg.curriculum
    library_id = band_library(cfg, band)
    pairs = select_cell_pairs(curr, cid, list(band), int(feed), fuel_library, library_id)
    if not pairs:
        raise KitError(
            f"cell {cid!r}: no in-band pairs auto-selected for band "
            f"{band_label(band[0], band[1])} (library {library_id}); "
            "check the fuel table / band coverage"
        )
    n_fresh = fresh_units_from_feed(int(feed))
    target = int(n_target if n_target is not None else curr.n_target)
    return {
        "name": cid,
        "campaign": cid,
        "library": library_id,
        "pairs": list(pairs),
        "feed": int(feed),
        "split_w1": [float(x) for x in (curr.split_w1 or [0.5])],
        "generators": _degrade_generators(curr.generators),
        "n_target": target,
        "priority": 100,
        "allow_single_cycle_discharge": bool(n_fresh > 30),
        "max_shuffle_depth": 2,
        "notes": f"multi-PC kit cell {cid}",
    }


# --------------------------------------------------------------------------- #
# minimal TOML emitter (no third-party writer dependency)
# --------------------------------------------------------------------------- #
def _toml_str(value: str) -> str:
    """A TOML basic string (double-quoted, backslash/quote/control escaped)."""
    out = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{out}"'


def _toml_num(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    f = float(value)
    if f.is_integer():
        # keep an explicit '.0' so a TOML float stays a float, never an int
        return f"{f:.1f}"
    return repr(f)


def _toml_str_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _toml_num_list(values: Sequence[Any]) -> str:
    return "[" + ", ".join(_toml_num(v) for v in values) + "]"


def _toml_generators(gens: dict[str, float]) -> str:
    inner = ", ".join(f"{k} = {_toml_num(v)}" for k, v in gens.items())
    return "{ " + inner + " }"


def _kit_is_paramA(cfg: LpoptConfig, strata: Sequence[dict[str, Any]]) -> bool:
    """True when any assigned stratum resolves to the paramA library (design
    package), so the kit bundles the package instead of FEASIBLE_PACKAGE."""
    return any(is_paramA_library(cfg, s.get("library")) for s in strata)


def ga80_package_root(cfg: LpoptConfig) -> Path:
    """The ga80 MASTER package (FEASIBLE_PACKAGE) this deck produces against.

    ``[verify].package_root`` resolved against the deck's own directory — the
    same resolution :func:`lpopt.search.resolver.build_case_resolver` performs
    for a non-paramA library.
    """
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    root = cfg.verify.package_root
    if not root:
        raise KitError(
            "ga80 cell(s) requested but the main deck has no [verify].package_root "
            "(the FEASIBLE_PACKAGE with lib/ bases/ cores/); set it and re-export"
        )
    p = Path(root)
    return p if p.is_absolute() else (base / p)


def render_kit_deck(cfg: LpoptConfig, strata: Sequence[dict[str, Any]]) -> str:
    """Render the portable ``lpopt_kit.inp`` produce deck text."""
    p = cfg.produce
    m = cfg.master
    paramA = _kit_is_paramA(cfg, strata)
    lines: list[str] = []
    a = lines.append
    a("# lpopt multi-PC PRODUCE KIT deck (auto-generated by `lpopt export-produce-kit`).")
    a("# Run on the SECOND PC with:  python -m lpopt produce --input lpopt_kit.inp")
    a("# EDIT the path(s) below for this machine before running (see KIT_README.md):")
    a("#   [master].executable   -> the local MASTER .exe")
    if paramA:
        a("#   [design].package_root -> the bundled paramA design package")
        a("#                            (data/design/package; ships inside this kit).")
    else:
        a("#   [verify].package_root -> the bundled ga80 MASTER package")
        a(f"#                            ({KIT_GA80_PACKAGE_REL}; ships inside this kit).")
    a("")
    a("[flow]")
    a(f"title = {_toml_str('multi-PC produce kit')}")
    a(f'output_root = {_toml_str("runs")}')
    a(f"random_seed = {_toml_num(cfg.flow.random_seed)}")
    a("")
    a("[master]")
    a(f"executable = {_toml_str(m.executable or 'D:/DeCART_MASTER/BIN/master4.0m4_r1.exe')}")
    a(f"workers = {_toml_num(m.workers)}")
    a(f"timeout = {_toml_num(m.timeout)}")
    a(f"max_cycles = {_toml_num(m.max_cycles)}")
    a(f"consecutive = {_toml_num(m.consecutive)}")
    a("")
    if paramA:
        # paramA strata resolve against the bundled design package (its own
        # bases/cores/lib + registry alias bridge); FEASIBLE_PACKAGE is NOT used.
        a("[design]")
        a(f"package_root = {_toml_str(KIT_DESIGN_PACKAGE_REL)}")
    else:
        a("[verify]")
        a(f"package_root = {_toml_str(KIT_GA80_PACKAGE_REL)}")
        # propagate harvest_maps (ga80 [verify] only) so produce kits can also
        # harvest EDIT5 node-peak maps.
        if bool(getattr(cfg.verify, "harvest_maps", False)):
            a("harvest_maps = true")
    a("")
    a("[produce]")
    a(f"campaign = {_toml_str('multipc_kit')}")
    a(f'ledger = {_toml_str("data/produce/ledger.jsonl")}')
    a(f'store_dir = {_toml_str("data/store")}')
    a(f"workers = {_toml_num(p.workers)}")
    a(f"use_all_cores = {_toml_num(bool(p.use_all_cores))}")
    a(f"host_reserve = {_toml_num(p.host_reserve)}")
    a(f"chain_timeout = {_toml_num(p.chain_timeout)}")
    a(f"max_cycles = {_toml_num(p.max_cycles)}")
    a(f"consecutive = {_toml_num(p.consecutive)}")
    a(f"resume = {_toml_num(bool(p.resume))}")
    a(f"purge_case_dirs = {_toml_num(bool(p.purge_case_dirs))}")
    a(f"purge_intermediate = {_toml_num(bool(p.purge_intermediate))}")
    a(f'promoted_root = {_toml_str("data/produce/promoted")}')
    a(f'synth_decks_root = {_toml_str("data/design/synth_decks")}')
    a("# No runs_flow deck tree here; the CaseAssetResolver synth tier + the")
    a("# package's own decks cover every pair, so no template_fallbacks are needed.")
    a("template_fallbacks = []")
    a("")
    for s in strata:
        a("[[produce.strata]]")
        a(f"name = {_toml_str(s['name'])}")
        a(f"campaign = {_toml_str(s['campaign'])}")
        a(f"library = {_toml_str(s['library'])}")
        a(f"pairs = {_toml_str_list(s['pairs'])}")
        a(f"feed = {_toml_num(s['feed'])}")
        a(f"split_w1 = {_toml_num_list(s['split_w1'])}")
        a(f"generators = {_toml_generators(s['generators'])}")
        a(f"n_target = {_toml_num(s['n_target'])}")
        a(f"priority = {_toml_num(s['priority'])}")
        a(f"allow_single_cycle_discharge = {_toml_num(bool(s['allow_single_cycle_discharge']))}")
        a(f"max_shuffle_depth = {_toml_num(s['max_shuffle_depth'])}")
        a(f"notes = {_toml_str(s['notes'])}")
        a("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# README (Korean)
# --------------------------------------------------------------------------- #
def render_kit_readme(
    cfg: LpoptConfig, strata: Sequence[dict[str, Any]], *, n_target: int
) -> str:
    total = sum(int(s["n_target"]) for s in strata)
    per_cell_hours = 2.5  # ~2.5 h / 150 converged @ 23 workers
    est_hours = per_cell_hours * (total / 150.0) if total else 0.0
    cell_rows = "\n".join(
        f"| `{s['name']}` | {s['feed']} | {', '.join(s['pairs'])} | "
        f"{s['n_target']} | {'예' if s['allow_single_cycle_discharge'] else '아니오'} |"
        for s in strata
    )
    exe = cfg.master.executable or "D:/DeCART_MASTER/BIN/master4.0m4_r1.exe"
    gen_txt = ", ".join(f"{k}={v}" for k, v in strata[0]["generators"].items()) if strata else ""
    paramA = _kit_is_paramA(cfg, strata)
    if paramA:
        included_extra = (
            f"  `{KIT_DESIGN_PACKAGE_REL}`(paramA 설계 패키지: bases/ cores/ lib/ hgc/\n"
            "  designs.json/registry.json — 이 킷 안에 이미 들어 있음),"
        )
        pkg_bullet = (
            "  2. (해당 없음) 이 킷은 **paramA 설계 패키지**를 이미 포함하고 있어\n"
            "     FEASIBLE_PACKAGE 를 따로 복사할 필요가 없습니다."
        )
        edit_block = f"""```
[master]
executable = "{exe}"          # 이 PC의 MASTER .exe 경로
```

`[design].package_root` 는 킷 안 `{KIT_DESIGN_PACKAGE_REL}` 를 가리키도록 이미
설정돼 있으니 그대로 두면 됩니다(설계 패키지가 킷에 동봉되어 있음)."""
        edit_count = "한 줄"
    else:
        included_extra = (
            f"  `{KIT_GA80_PACKAGE_REL}`(ga80 MASTER 패키지: lib/ bases/ cores/ 전체 —\n"
            "  이 킷 안에 이미 들어 있음),"
        )
        pkg_bullet = (
            "  2. (해당 없음) 이 킷은 **ga80 MASTER 패키지**(`lib/` `bases/` `cores/`)를\n"
            "     이미 포함하고 있어 FEASIBLE_PACKAGE 를 따로 복사할 필요가 없습니다.\n"
            "     **일부만 골라 복사하지 마세요** — restart 탐색 사다리가 `bases/` 전체를\n"
            "     훑어 최적 restart 를 고르므로, 폴더를 빼면 실패가 아니라 *더 나쁜*\n"
            "     restart 로 조용히 강등되어(예: feed-117 → feed-121) 발산합니다."
        )
        edit_block = f"""```
[master]
executable = "{exe}"          # 이 PC의 MASTER .exe 경로
```

`[verify].package_root` 는 킷 안 `{KIT_GA80_PACKAGE_REL}` 를 가리키도록 이미
설정돼 있으니 그대로 두면 됩니다(MASTER 패키지가 킷에 동봉되어 있음)."""
        edit_count = "한 줄"
    return f"""# LEU+ 학습데이터 생산 킷 (2번 PC용)

이 폴더는 **두 번째 Windows PC**에서 `lpopt produce` 로 지정된 커리큘럼 셀들의
MASTER 평형-주기 학습 데이터를 생산하기 위한 **자립형(portable) 킷**입니다.
생산이 끝나면 `data/` 폴더를 압축해 1번 PC로 보내면 `lpopt merge-store` 로 본
저장소에 병합됩니다.

---

## 0. 킷에 들어있는 것 / 없는 것

- 포함: `{KIT_DECK_NAME}`(생산 deck), `lpopt/`(소스 트리), `pyproject.toml`,
  `data/store/fuel_types.parquet`(물리 연료 테이블),
{included_extra}  이 README.
- **미포함(용량 때문에 따로 복사해야 함)**:
  1. **MASTER 실행 파일 폴더** — `D:/DeCART_MASTER/BIN` (특히 MASTER `.exe`).
     1번 PC의 `{exe}` 와 같은 경로에 두면 편집 없이 동작합니다.
{pkg_bullet}
- **records.parquet 는 일부러 넣지 않았습니다.** PC2는 빈 저장소에서 시작하고
  (생산 경로가 빈/없는 `records.parquet` 를 허용), 생산하면서 새로 채웁니다.

---

## 1. 경로 수정 (`{KIT_DECK_NAME}`)

deck 의 아래 {edit_count}만 이 PC 환경에 맞게 고칩니다. 나머지 경로는 킷 폴더 기준
상대경로라 그대로 두면 됩니다.

{edit_block}

---

## 2. 파이썬 환경 (torch 불필요)

생산 경로는 **torch 가 필요 없습니다**(`pyproject.toml` 에 torch 는 의존성에서
제외되어 있고, produce 경로 import 에도 torch 가 없음을 확인했습니다). 아래 한 줄:

```
python -m venv .venv && .venv\\Scripts\\activate && pip install -e .
```

(설치되는 것: numpy, pandas, pyarrow, scikit-learn, scipy, pyyaml, matplotlib,
joblib — 모두 CPU 전용, torch 없음.)

---

## 3. 실행

```
python -m lpopt produce --input {KIT_DECK_NAME}
```

- 워커 수는 자동(`workers = 0`, `use_all_cores = true`): 이 PC의 논리 코어 수에서
  1개(host_reserve)를 뺀 만큼 MASTER 체인을 동시에 돌립니다(23코어 기준 ~23워커).
- **예상 소요시간**: 23워커 기준 셀당 150 수렴 ≈ 약 2.5시간.
  이 킷 합계 {total} 수렴 → 약 {est_hours:.1f}시간.
- 중단되어도 안전합니다(crash-safe resume): 다시 같은 명령을 실행하면 이미 만든
  라벨은 건너뛰고 목표(n_target)까지 이어서 생산합니다.

### 생성기(generators) 참고
이 킷의 생성기 혼합은 `{gen_txt}` 입니다. 커리큘럼 기본 혼합에 있던
`elite_perturb` 는 **PC2의 빈 저장소에는 elite 가 없어** 제거하고 나머지 가중치로
정규화했습니다(생산 품질에는 영향 없음 — 남은 random/heuristic 로 대체).

---

## 4. 지정된 셀

| 셀(campaign) | feed | pairs | n_target | 단일주기 방출 |
|---|---|---|---|---|
{cell_rows}

각 행은 deck 의 `[[produce.strata]]` 하나이며 `campaign`/`name` 은 커리큘럼
셀 id 와 **정확히 일치**합니다(병합 시 셀별로 집계됨).

---

## 5. 결과 돌려보내기

생산이 끝나면 이 킷의 **`data/` 폴더만** 압축해서 1번 PC로 보내세요:

```
# 예: PowerShell
Compress-Archive -Path data -DestinationPath produce_kit_result.zip
```

`data/` 안에는 `store/records.parquet`(+생겼다면 `maps.npz`) 와
`produce/ledger.jsonl` 이 들어 있습니다. 1번 PC에서:

```
python -m lpopt merge-store --from <압축을-푼-data-폴더>
```

- 본 저장소에 **품질우선(UPSERT) 병합**됩니다(수렴 라벨이 비수렴/실패를 덮어씀,
  반대로는 절대 덮지 않음). **재실행해도 안전(idempotent)** — 중복은 무시됩니다.
- ledger 도 `(record_id, status)` 기준으로 중복 없이 병합되어 본 생산기의
  resume 카운터가 어긋나지 않습니다.
"""


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", "*.egg-info", ".pytest_cache", ".mypy_cache"
)


def _carry_external_templates(
    cfg: LpoptConfig,
    strata: Sequence[dict[str, Any]],
    fuel_library: Any,
    src_pkg: Path,
    dst_pkg: Path,
) -> list[str]:
    """Copy template decks the SOURCE resolver reaches OUTSIDE the package.

    ``[produce].template_fallbacks`` lets the main deck resolve a template from a
    tree that is NOT part of FEASIBLE_PACKAGE (here: the hydrated ``runs_flow`` GA
    candidate decks, used because some packaged case decks are OneDrive
    placeholders).  A kit ships no such tree and sets ``template_fallbacks = []``,
    so without this step an assigned case that resolves to a fallback deck at
    home would silently drop to the *synthesis* tier on the kit PC — a different
    deck for the same case, which is exactly the class of divergence this module
    must not introduce.

    Each such deck is copied to ``cores/<case.folder>/carried/`` in the kit, where
    the resolver finds it as the case's own *exact* deck.  Returns one human note
    per carried deck.  Cases that already resolve inside the package are
    untouched, so a deck-complete package produces no copies at all.
    """
    from .search.assets import CaseAssetResolver
    from .vendor.masterrl.domain import CaseKey

    base = cfg.source_path.parent if cfg.source_path else Path.cwd()

    def _rel(p: str | Path) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (base / pp)

    fallbacks = [str(_rel(g)) for g in cfg.produce.template_fallbacks]
    if not fallbacks:
        return []
    # synth_root=None: probe the HONEST package/fallback answer and never write a
    # synthesized deck into the source repo as a side effect of exporting.
    probe = CaseAssetResolver(
        src_pkg,
        _rel(cfg.produce.promoted_root),
        template_fallbacks=fallbacks,
        fuel_library=fuel_library,
        library_id=cfg.curriculum.library,
        synth_root=None,
    )
    pkg_resolved = src_pkg.resolve()
    notes: list[str] = []
    for spec in strata:
        for pair in spec["pairs"]:
            case = CaseKey(str(pair), int(spec["feed"]))
            deck = probe.resolve(case).template_deck_path
            if deck is None or not deck.is_file():
                continue
            try:
                deck.resolve().relative_to(pkg_resolved)
                continue                      # already inside the package -> ships
            except ValueError:
                pass
            dest = dst_pkg / "cores" / case.folder / "carried" / deck.name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(deck, dest)
            notes.append(f"{case.folder} <- {deck.name} (external fallback)")
    return notes


@dataclass
class KitExport:
    out_dir: Path
    deck_path: Path
    readme_path: Path
    cells: list[dict[str, Any]]
    n_target: int


# --------------------------------------------------------------------------- #
# frontier kit (F_r=1.55 boundary training campaign — PC2 worker kit)
# --------------------------------------------------------------------------- #
#: kit-relative dir the frontier deck's [model].model_dir points at.
KIT_MODEL_REL = "data/models/champion"


@dataclass
class FrontierKitExport:
    out_dir: Path
    deck_path: Path
    bat_path: Path
    schtasks_path: Path
    model_dir: Path
    roster_pairs: list[str]


def render_frontier_kit_deck(cfg: LpoptConfig, *, round_budget: int = 276) -> str:
    """Render the portable ``lpopt_kit.inp`` deck for the fr_boundary campaign.

    ``objective = "fr_boundary"``, ``[verify].package_root = FEASIBLE_PACKAGE``,
    ``[model].inference = "local_cpu"`` (the kit runs the champion on PC2 CPU — no
    remote inference), and ``[model].model_dir`` points at the bundled champion.
    """
    m = cfg.master
    acq = cfg.acquisition
    lines: list[str] = []
    a = lines.append
    a("# lpopt FRONTIER KIT deck (auto-generated by `lpopt export-frontier-kit`).")
    a("# Run ONE round on PC2 with the bundled run_frontier.bat (sets LPOPT_WORKER=1).")
    a("# EDIT [master].executable for this machine before running (see below).")
    a("#   [verify].package_root -> the bundled ga80 MASTER package")
    a(f"#                            ({KIT_GA80_PACKAGE_REL}; ships WHOLE inside this kit).")
    a("")
    a("[flow]")
    a(f"title = {_toml_str('fr_boundary frontier kit')}")
    a(f'output_root = {_toml_str("runs")}')
    a(f"random_seed = {_toml_num(cfg.flow.random_seed)}")
    a("")
    a("[master]")
    a(f"executable = {_toml_str(m.executable or 'D:/DeCART_MASTER/BIN/master4.0m4_r1.exe')}")
    a(f"workers = {_toml_num(m.workers)}")
    a(f"timeout = {_toml_num(m.timeout)}")
    a(f"max_cycles = {_toml_num(m.max_cycles)}")
    a(f"consecutive = {_toml_num(m.consecutive)}")
    a("")
    a("[verify]")
    a(f"package_root = {_toml_str(KIT_GA80_PACKAGE_REL)}")
    # Harvest converged EDIT5 assembly maps into the kit store's maps.npz so the
    # node-power model gets BOUNDARY-region maps (forensic 20260723: the only maps
    # were Dataset-A f121).  Forces keep_success (final MAS_SUM survives to harvest).
    a("harvest_maps = true")
    a("")
    # CPU-sizing block — MUST be emitted (forensic 20260723): a local_cpu worker with
    # the [search] defaults (pool 20000 + [search.local_search] max_predictions 40000)
    # floods the per-wave screen (a 198 smoke was stuck scoring 187 min+).  These are
    # the round1c-proven lean values; the per-cell _cell_cfg re-clamps them too.
    s = cfg.search
    ls = s.local_search
    a("[search]")
    a(f"pool_size = {_toml_num(min(s.pool_size, 2000))}")
    a(f"pool_cap = {_toml_num(s.pool_cap)}")
    a(f"elite_frac = {_toml_num(0.70)}")
    a(f"guided_frac = {_toml_num(0.10)}")
    a(f"diversity_frac = {_toml_num(0.20)}")
    a(f"beam_width = {_toml_num(s.beam_width)}")
    a(f"completions_per_prefix = {_toml_num(s.completions_per_prefix)}")
    a(f"n_moves_early = {_toml_num(s.n_moves_early)}")
    a(f"n_moves_late = {_toml_num(s.n_moves_late)}")
    a(f"elite_top_k = {_toml_num(s.elite_top_k)}")
    a(f"dry_run_pool_size = {_toml_num(s.dry_run_pool_size)}")
    a("")
    a("[search.trust_region]")
    a(f"enabled = {'true' if s.trust_region.enabled else 'false'}")
    a(f"feed_step = {_toml_num(s.trust_region.feed_step)}")
    a(f"e_core_band = {_toml_num(s.trust_region.e_core_band)}")
    a(f"n_min = {_toml_num(s.trust_region.n_min)}")
    a(f"promote_after = {_toml_num(s.trust_region.promote_after)}")
    a(f"frontier_sigma_inflation = {_toml_num(s.trust_region.frontier_sigma_inflation)}")
    a(f"frontier_slots_per_wave = {_toml_num(s.trust_region.frontier_slots_per_wave)}")
    a("")
    a("[search.local_search]")
    a(f"top_m = {_toml_num(min(ls.top_m, 32))}")
    a(f"neighbors = {_toml_num(min(ls.neighbors, 48))}")
    a(f"depth = {_toml_num(min(ls.depth, 2))}")
    a(f"max_predictions = {_toml_num(min(ls.max_predictions, 1500))}")
    a(f"n_moves = {_toml_num(ls.n_moves)}")
    a("")
    a("[model]")
    a(f"model_dir = {_toml_str(KIT_MODEL_REL)}")
    a(f"store_dir = {_toml_str('data/store')}")
    a(f"library_id = {_toml_str(cfg.model.library_id)}")
    a(f"device = {_toml_str('cpu')}")
    a(f"inference = {_toml_str('local_cpu')}")
    a("")
    a("[produce]")
    a(f"store_dir = {_toml_str('data/store')}")
    a(f'ledger = {_toml_str("data/produce/ledger.jsonl")}')
    a(f'promoted_root = {_toml_str("data/produce/promoted")}')
    a("template_fallbacks = []")
    a("")
    a("[acquisition]")
    a(f"objective = {_toml_str('fr_boundary')}")
    a(f"budget = {_toml_num(round_budget)}")
    a(f"risk_z = {_toml_num(acq.risk_z)}")
    a(f"cbc_limit = {_toml_num(acq.cbc_limit)}")
    a(f"f_q_limit = {_toml_num(acq.f_q_limit)}")
    a(f"ao_abs_limit = {_toml_num(acq.ao_abs_limit)}")
    a(f"fr_boundary_band_lo = {_toml_num(getattr(acq, 'fr_boundary_band_lo', 1.45))}")
    a(f"fr_boundary_band_hi = {_toml_num(getattr(acq, 'fr_boundary_band_hi', 1.70))}")
    a(f"fr_boundary_pin_bu_limit = {_toml_num(getattr(acq, 'fr_boundary_pin_bu_limit', 80.0))}")
    a("")
    return "\n".join(lines) + "\n"


def render_run_frontier_bat(*, round_budget: int = 276,
                            run_root: str = "runs/frontier") -> str:
    """The PC2 ``run_frontier.bat``: sets LPOPT_WORKER=1 + PYTHONUTF8=1, writes the
    ROUND_RUNNING/ROUND_DONE/ROUND_FAILED lifecycle markers around exactly ONE round.

    The markers (not frontier_round.json) are what the supervising session polls, so
    a crash mid-round is observable (ROUND_FAILED carries %ERRORLEVEL%; ROUND_RUNNING
    is always deleted on exit)."""
    return (
        "@echo off\r\n"
        "REM ---- lpopt frontier round worker (PC2). ONE round then exit. ----\r\n"
        "setlocal\r\n"
        "set LPOPT_WORKER=1\r\n"
        "set PYTHONUTF8=1\r\n"
        "cd /d \"%~dp0\"\r\n"
        "del /q ROUND_DONE ROUND_FAILED 2>nul\r\n"
        "echo %DATE% %TIME% > ROUND_RUNNING\r\n"
        f"python -m lpopt frontier-produce --input lpopt_kit.inp "
        f"--round-budget {round_budget} --run-root {run_root}\r\n"
        "set RC=%ERRORLEVEL%\r\n"
        "if \"%RC%\"==\"0\" (\r\n"
        "  echo %DATE% %TIME% > ROUND_DONE\r\n"
        ") else (\r\n"
        "  echo %RC% > ROUND_FAILED\r\n"
        ")\r\n"
        "del /q ROUND_RUNNING 2>nul\r\n"
        "endlocal\r\n"
        "exit /b %RC%\r\n"
    )


def render_frontier_schtasks_xml(*, task_cmd: str = "run_frontier.bat") -> str:
    """A minimal Windows Task Scheduler XML that launches one frontier round on
    demand (registered/kicked by the supervising session between rounds)."""
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\r\n'
        '<Task version="1.2" '
        'xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\r\n'
        "  <RegistrationInfo>\r\n"
        "    <Description>lpopt fr_boundary frontier round (one-shot worker)</Description>\r\n"
        "  </RegistrationInfo>\r\n"
        "  <Triggers />\r\n"
        "  <Principals>\r\n"
        '    <Principal id="Author">\r\n'
        "      <LogonType>InteractiveToken</LogonType>\r\n"
        "      <RunLevel>LeastPrivilege</RunLevel>\r\n"
        "    </Principal>\r\n"
        "  </Principals>\r\n"
        "  <Settings>\r\n"
        "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\r\n"
        "    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\r\n"
        "    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\r\n"
        "    <ExecutionTimeLimit>PT12H</ExecutionTimeLimit>\r\n"
        "    <Enabled>true</Enabled>\r\n"
        "  </Settings>\r\n"
        "  <Actions Context=\"Author\">\r\n"
        "    <Exec>\r\n"
        f"      <Command>{task_cmd}</Command>\r\n"
        "    </Exec>\r\n"
        "  </Actions>\r\n"
        "</Task>\r\n"
    )


def frontier_roster_pairs() -> list[str]:
    """The 6 mono-anchor family pairs of the fr_boundary roster (Decision 1)."""
    from .search.frontier_search import build_roster
    return sorted({c.pair for c in build_roster()})


def export_frontier_kit(
    cfg: LpoptConfig,
    out_dir: str | Path,
    *,
    round_budget: int = 276,
    fuel_library: Any = None,
    last_round_marker: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> FrontierKitExport:
    """Build the self-contained PC2 fr_boundary frontier kit under ``out_dir``.

    Bundles: ``lpopt/`` source + pyproject.toml; ``FEASIBLE_PACKAGE`` lib/bases/cores
    WHOLE (hard-fail on any missing GA80_PACKAGE_SUBDIR — a partial bases/ silently
    demotes restart levels); the current champion model dir; ``data/store`` WITH
    records.parquet + maps.npz + fuel_types.parquet (without them elite sourcing,
    replay/holdout gating, and backend calibration all silently degrade); the
    ``lpopt_kit.inp`` fr_boundary deck; ``run_frontier.bat`` (LPOPT_WORKER=1,
    PYTHONUTF8=1, ROUND markers) + the schtasks XML.

    R1: the roster is all mono-anchor pairs; :func:`assert_mono_anchor` hard-fails on
    any cross-anchor pair.  Store guard: the build ASSERTS the store has converged
    rows for EVERY roster pair, and — when ``last_round_marker`` is given — REFUSES to
    build if the home store predates the last pulled round marker (enforcing
    pull+merge BEFORE re-ship, so the shipped store is a strict superset of PC2's)."""
    from .data.compliance import assert_mono_anchor
    from .data.store import MAPS_NAME, RECORDS_NAME, StoreReader

    log = log or (lambda m: print(m))
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()

    roster_pairs = frontier_roster_pairs()
    assert_mono_anchor(roster_pairs)                  # R1 structural guard

    store_dir = Path(cfg.model.store_dir)
    store_dir = store_dir if store_dir.is_absolute() else (base / store_dir)
    records_path = store_dir / RECORDS_NAME
    maps_path = store_dir / MAPS_NAME
    fuel_parquet = store_dir / "fuel_types.parquet"
    if not records_path.exists():
        raise KitError(f"frontier kit needs a populated store; {records_path} missing")

    # -- store guard: rows for every roster pair + freshness vs last pull ------ #
    df = StoreReader(store_dir).records
    have = set(df["case_pair"].dropna().unique()) if "case_pair" in df.columns else set()
    missing_pairs = [p for p in roster_pairs if p not in have]
    if missing_pairs:
        raise KitError(
            f"store has no rows for roster pair(s) {missing_pairs}; the shipped kit "
            "would start those cells with an EMPTY elite basin (check the home store)")
    if last_round_marker is not None:
        marker = Path(last_round_marker)
        if marker.exists() and records_path.stat().st_mtime < marker.stat().st_mtime:
            raise KitError(
                "home store predates the last pulled round marker "
                f"({records_path} older than {marker}); pull+merge the last round "
                "BEFORE re-shipping so the kit store is a superset of PC2's")

    # -- FEASIBLE_PACKAGE whole (lib/bases/cores) ------------------------------ #
    src_pkg = ga80_package_root(cfg)
    missing = [d for d in GA80_PACKAGE_SUBDIRS if not (src_pkg / d).is_dir()]
    if not src_pkg.is_dir() or missing:
        raise KitError(
            f"the MASTER package at {src_pkg} is unusable: "
            f"{'does not exist' if not src_pkg.is_dir() else 'missing ' + '/'.join(missing)} "
            "(the whole-bundle rule ships lib/bases/cores WHOLE; a partial bases/ "
            "silently demotes restart levels)")

    # -- champion model dir ---------------------------------------------------- #
    model_dir = Path(cfg.model.model_dir)
    model_dir = model_dir if model_dir.is_absolute() else (base / model_dir)
    if not model_dir.is_dir():
        raise KitError(f"champion model dir not found: {model_dir}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- source tree + pyproject ---------------------------------------------- #
    pkg_src = Path(__file__).resolve().parent
    shutil.copytree(pkg_src, out / "lpopt", ignore=_COPY_IGNORE, dirs_exist_ok=True)
    pyproject = pkg_src.parent / "pyproject.toml"
    if pyproject.exists():
        shutil.copy2(pyproject, out / "pyproject.toml")

    # -- store WITH records + maps + fuel table -------------------------------- #
    dst_store = out / "data" / "store"
    dst_store.mkdir(parents=True, exist_ok=True)
    shutil.copy2(records_path, dst_store / RECORDS_NAME)
    if maps_path.exists():
        shutil.copy2(maps_path, dst_store / MAPS_NAME)
    if fuel_parquet.exists():
        shutil.copy2(fuel_parquet, dst_store / "fuel_types.parquet")
    elif fuel_library is not None:
        fuel_library.frame.to_parquet(dst_store / "fuel_types.parquet")

    # -- FEASIBLE_PACKAGE copy (whole) ---------------------------------------- #
    dst_pkg = out / KIT_GA80_PACKAGE_REL
    for sub in GA80_PACKAGE_SUBDIRS:
        shutil.copytree(src_pkg / sub, dst_pkg / sub, ignore=_COPY_IGNORE,
                        dirs_exist_ok=True)

    # -- champion model copy --------------------------------------------------- #
    dst_model = out / KIT_MODEL_REL
    shutil.copytree(model_dir, dst_model, ignore=_COPY_IGNORE, dirs_exist_ok=True)

    # -- deck + bat + schtasks ------------------------------------------------- #
    deck_path = out / KIT_DECK_NAME
    deck_path.write_text(render_frontier_kit_deck(cfg, round_budget=round_budget),
                         encoding="utf-8")
    bat_path = out / "run_frontier.bat"
    bat_path.write_text(render_run_frontier_bat(round_budget=round_budget),
                        encoding="utf-8")
    schtasks_path = out / "frontier_task.xml"
    schtasks_path.write_text(render_frontier_schtasks_xml(), encoding="utf-8")

    # validate the generated deck parses back with the fr_boundary objective.
    from .config import load_config
    parsed = load_config(deck_path)
    if str(getattr(parsed.acquisition, "objective", "")) != "fr_boundary":
        raise KitError("generated frontier deck did not round-trip objective=fr_boundary")

    log(f"[frontier-kit] wrote deck    -> {deck_path}")
    log(f"[frontier-kit] wrote source  -> {out / 'lpopt'}")
    log(f"[frontier-kit] wrote store   -> {dst_store} (records + maps + fuel table)")
    log(f"[frontier-kit] wrote package -> {dst_pkg} (lib/bases/cores WHOLE)")
    log(f"[frontier-kit] wrote model   -> {dst_model}")
    log(f"[frontier-kit] wrote bat     -> {bat_path} (LPOPT_WORKER=1)")
    return FrontierKitExport(
        out_dir=out, deck_path=deck_path, bat_path=bat_path,
        schtasks_path=schtasks_path, model_dir=dst_model, roster_pairs=roster_pairs,
    )


def export_produce_kit(
    cfg: LpoptConfig,
    cells: Sequence[str],
    out_dir: str | Path,
    *,
    n_target: int | None = None,
    fuel_library: Any = None,
    log: Callable[[str], None] | None = None,
) -> KitExport:
    """Build a portable produce kit for ``cells`` under ``out_dir``.

    ``cfg`` is the MAIN campaign deck (for ``[curriculum]`` config + the fuel store
    location + master/produce defaults).  ``fuel_library`` is loaded from the
    store's ``fuel_types.parquet`` when not supplied.
    """
    log = log or (lambda m: print(m))
    if not cells:
        raise KitError("no cells given (use --cells a,b,c)")

    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    store_dir = Path(cfg.model.store_dir)
    store_dir = store_dir if store_dir.is_absolute() else (base / store_dir)
    fuel_parquet = store_dir / "fuel_types.parquet"

    if fuel_library is None:
        from .data.fuel_types import FuelLibrary
        if fuel_parquet.exists():
            fuel_library = FuelLibrary.from_parquet(fuel_parquet)
        else:
            fuel_library = FuelLibrary.build(cfg, persist=False)

    # -- build strata (one per cell) -------------------------------------- #
    strata: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cells:
        cid = str(raw).strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        band, feed = parse_cell_id(cid)
        strat = build_cell_stratum(cfg, cid, band, feed, fuel_library, n_target=n_target)
        strata.append(strat)
        log(f"[kit] cell {cid}: feed={feed} pairs={strat['pairs']} "
            f"n_target={strat['n_target']} "
            f"single_cycle_discharge={strat['allow_single_cycle_discharge']}")

    if not strata:
        raise KitError("no usable cells after parsing")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- deck -------------------------------------------------------------- #
    deck_text = render_kit_deck(cfg, strata)
    deck_path = out / KIT_DECK_NAME
    deck_path.write_text(deck_text, encoding="utf-8")

    # validate the generated deck parses AND its strata match the cells
    from .config import load_config
    parsed = load_config(deck_path)
    got = [s.name for s in parsed.produce.strata]
    want = [s["name"] for s in strata]
    if got != want:
        raise KitError(f"generated deck strata {got} != requested cells {want}")
    for pstrat, spec in zip(parsed.produce.strata, strata):
        if (pstrat.campaign != spec["campaign"] or list(pstrat.pairs) != spec["pairs"]
                or int(pstrat.feed) != int(spec["feed"])):
            raise KitError(f"generated deck stratum {pstrat.name!r} did not round-trip")

    # -- source tree + pyproject ------------------------------------------ #
    pkg_src = Path(__file__).resolve().parent          # .../lpopt
    shutil.copytree(pkg_src, out / "lpopt", ignore=_COPY_IGNORE, dirs_exist_ok=True)
    pyproject = pkg_src.parent / "pyproject.toml"
    if pyproject.exists():
        shutil.copy2(pyproject, out / "pyproject.toml")

    # -- fuel table copy (fresh EMPTY store: no records.parquet) ---------- #
    (out / "data" / "store").mkdir(parents=True, exist_ok=True)
    if fuel_parquet.exists():
        shutil.copy2(fuel_parquet, out / "data" / "store" / "fuel_types.parquet")
    else:
        fuel_library.frame.to_parquet(out / "data" / "store" / "fuel_types.parquet")

    # -- paramA design package copy (self-contained kit) ------------------ #
    # A paramA stratum resolves against the DESIGN PACKAGE (its own bases/ cores/
    # lib/ + registry alias bridge), NOT FEASIBLE_PACKAGE.  Bundle it under the
    # kit-relative dir the generated deck's [design].package_root points at so the
    # kit is self-contained (no 456MB FEASIBLE_PACKAGE push needed for paramA).
    if _kit_is_paramA(cfg, strata):
        src_pkg = paramA_package_root(cfg)
        if not src_pkg.is_dir():
            raise KitError(
                f"paramA cell(s) requested but the design package is missing: "
                f"{src_pkg} (build/assemble the paramA package first, or check "
                "[design].package_root / [design].store_dir on the main deck)"
            )
        dst_pkg = out / KIT_DESIGN_PACKAGE_REL
        shutil.copytree(src_pkg, dst_pkg, ignore=_COPY_IGNORE, dirs_exist_ok=True)
        pkg_bytes = sum(f.stat().st_size for f in dst_pkg.rglob("*") if f.is_file())
        log(f"[kit] wrote design  -> {dst_pkg}  ({pkg_bytes / 1e6:.1f} MB)")
    else:
        # -- ga80 MASTER package copy (self-contained kit) ----------------- #
        # A ga80 stratum resolves against FEASIBLE_PACKAGE.  Bundle lib/ bases/
        # cores/ WHOLE under the kit-relative dir the generated deck's
        # [verify].package_root points at.  Hand-copying a subset of bases/ is
        # exactly the failure this replaces: the restart ladder scans the entire
        # catalog, so a missing sibling folder silently DEMOTES a case to a
        # worse-matched restart instead of failing loudly (see
        # GA80_PACKAGE_SUBDIRS).
        src_pkg = ga80_package_root(cfg)
        missing = [d for d in GA80_PACKAGE_SUBDIRS if not (src_pkg / d).is_dir()]
        if not src_pkg.is_dir() or missing:
            raise KitError(
                f"ga80 cell(s) requested but the MASTER package at {src_pkg} is "
                f"unusable: {'does not exist' if not src_pkg.is_dir() else 'missing ' + '/'.join(missing)} "
                "(check [verify].package_root on the main deck)"
            )
        dst_pkg = out / KIT_GA80_PACKAGE_REL
        for sub in GA80_PACKAGE_SUBDIRS:
            shutil.copytree(
                src_pkg / sub, dst_pkg / sub, ignore=_COPY_IGNORE, dirs_exist_ok=True
            )
        carried = _carry_external_templates(cfg, strata, fuel_library, src_pkg, dst_pkg)
        pkg_bytes = sum(f.stat().st_size for f in dst_pkg.rglob("*") if f.is_file())
        n_bases = sum(1 for p in (dst_pkg / "bases").iterdir() if p.is_dir())
        n_decks = sum(1 for _ in (dst_pkg / "cores").glob("*/*/MAS_INP_cy*.inp"))
        log(f"[kit] wrote ga80pkg -> {dst_pkg}  ({pkg_bytes / 1e6:.1f} MB, "
            f"{n_bases} base restarts, {n_decks} template decks)")
        for note in carried:
            log(f"[kit]   carried deck: {note}")

    # -- README ------------------------------------------------------------ #
    total_target = strata[0]["n_target"] if strata else (n_target or cfg.curriculum.n_target)
    readme = render_kit_readme(cfg, strata, n_target=int(total_target))
    readme_path = out / "KIT_README.md"
    readme_path.write_text(readme, encoding="utf-8")

    log(f"[kit] wrote deck    -> {deck_path}")
    log(f"[kit] wrote source  -> {out / 'lpopt'}")
    log(f"[kit] wrote fueltab -> {out / 'data' / 'store' / 'fuel_types.parquet'}")
    log(f"[kit] wrote readme  -> {readme_path}")
    return KitExport(
        out_dir=out, deck_path=deck_path, readme_path=readme_path,
        cells=strata, n_target=int(total_target),
    )


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def _has_flatness(node_peak: Any = None, map_cov: Any = None) -> bool:
    """True when a row carries at least one populated flatness column.

    Pandas-free on purpose (this module defers its pandas import): ``float()``
    rejects ``None`` / ``pd.NA`` / ``NaT`` and the ``v == v`` test rejects NaN,
    which covers every null shape the store writes.
    """
    for value in (node_peak, map_cov):
        try:
            fv = float(value)               # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if fv == fv:                        # not NaN
            return True
    return False


def _quality_rank_row(converged: Any, valid: Any, node_peak: Any = None,
                      map_cov: Any = None) -> int:
    """Row information quality for the merge's upgrade decision (higher = better).

    ``converged*4 + valid*2 + has_flatness`` — the same ordering as the store's
    :func:`..data.store._quality_rank`, which this must agree with or the merge
    reports one thing and persists another.

    The flatness bit is the low-order term and it MATTERS here: ``merge_store``
    only rewrites the store when it classified at least one incoming row as new
    or upgraded, so a kit row that differs from the stored one ONLY by carrying
    ``node_peak`` / ``map_cov`` used to rank equal, be counted a duplicate, and
    have the whole merge skipped — the harvested flatness labels were silently
    discarded at the door of the flatness-first program.
    """
    return ((4 if bool(converged) else 0) + (2 if bool(valid) else 0)
            + (1 if _has_flatness(node_peak, map_cov) else 0))


def _resolve_kit_paths(from_dir: Path) -> tuple[Path, Path, Path]:
    """Return ``(kit_records, kit_maps, kit_ledger)`` for a returned kit ``data/``
    dir.  Tolerates ``from_dir`` pointing straight at the ``store/`` folder."""
    from_dir = Path(from_dir)
    if (from_dir / "store" / "records.parquet").exists() or (from_dir / "store").is_dir():
        store = from_dir / "store"
        ledger = from_dir / "produce" / "ledger.jsonl"
    elif (from_dir / "records.parquet").exists():
        store = from_dir
        ledger = from_dir.parent / "produce" / "ledger.jsonl"
    else:
        store = from_dir / "store"
        ledger = from_dir / "produce" / "ledger.jsonl"
    return store / "records.parquet", store / "maps.npz", ledger


def _merge_ledger(
    main_ledger: Path, kit_ledger: Path, *, dry_run: bool
) -> dict[str, int]:
    """Append kit ledger lines new to the main ledger, dedup by ``(record_id,
    status)`` (record_id-less cosmetic 'dup' lines are skipped)."""
    from .search.produce import Ledger
    result = {"kit_lines": 0, "appended": 0, "skipped_existing": 0, "skipped_ridless": 0}
    if not Path(kit_ledger).exists():
        return result
    seen: set[tuple[str, str]] = set()
    for row in Ledger.replay(main_ledger):
        rid = str(row.get("record_id", "") or "")
        if rid:
            seen.add((rid, str(row.get("status", ""))))
    to_append: list[dict[str, Any]] = []
    for row in Ledger.replay(kit_ledger):
        result["kit_lines"] += 1
        rid = str(row.get("record_id", "") or "")
        if not rid:
            result["skipped_ridless"] += 1
            continue
        key = (rid, str(row.get("status", "")))
        if key in seen:
            result["skipped_existing"] += 1
            continue
        seen.add(key)
        to_append.append(row)
    result["appended"] = len(to_append)
    if to_append and not dry_run:
        Path(main_ledger).parent.mkdir(parents=True, exist_ok=True)
        with open(main_ledger, "a", encoding="utf-8") as handle:
            for row in to_append:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    return result


@dataclass
class MergeReport:
    from_dir: str
    store_dir: str
    kit_rows: int
    new_rows: int
    upgraded_rows: int
    duplicate_rows: int
    total_before: int
    total_after: int
    maps_new: int
    ledger: dict[str, int]
    per_campaign: list[dict[str, Any]]
    flagged_campaigns: list[dict[str, str]]
    dry_run: bool

    def text(self) -> str:
        lines: list[str] = []
        a = lines.append
        a(f"merge-store  (from: {self.from_dir})")
        a(f"main store : {self.store_dir}   dry_run={self.dry_run}")
        a("")
        a(f"kit rows           : {self.kit_rows}")
        a(f"  new              : {self.new_rows}")
        a(f"  upgraded         : {self.upgraded_rows}")
        a(f"  duplicates (kept): {self.duplicate_rows}")
        a(f"store total        : {self.total_before} -> {self.total_after}")
        if self.maps_new:
            a(f"maps merged        : {self.maps_new}")
        a(f"ledger             : +{self.ledger.get('appended', 0)} lines "
          f"({self.ledger.get('kit_lines', 0)} kit / "
          f"{self.ledger.get('skipped_existing', 0)} already present / "
          f"{self.ledger.get('skipped_ridless', 0)} rid-less skipped)")
        a("")
        a("per-campaign converged (P, converged==True):")
        a(f"  {'CAMPAIGN':22s} {'BEFORE':>7s} {'AFTER':>7s} {'FLAG':>6s}")
        a("  " + "-" * 46)
        flagged = {f["campaign"] for f in self.flagged_campaigns}
        for row in self.per_campaign:
            flag = "!" if row["campaign"] in flagged else ""
            a(f"  {str(row['campaign']):22s} {row['before']:>7d} "
              f"{row['after']:>7d} {flag:>6s}")
        if self.flagged_campaigns:
            a("")
            a("flagged campaigns (merged, but NOT a recognized curriculum cell id):")
            for f in self.flagged_campaigns:
                a(f"  {f['campaign']} — {f['reason']}")
        return "\n".join(lines)


def _known_cells(cfg: LpoptConfig) -> set[str] | None:
    """Best-effort read-only load of the curriculum's known cell ids from
    ``state.json`` (never written).  Returns None when unavailable."""
    try:
        base = cfg.source_path.parent if cfg.source_path else Path.cwd()
        state_dir = Path(cfg.curriculum.state_dir)
        state_dir = state_dir if state_dir.is_absolute() else (base / state_dir)
        state_path = state_dir / "state.json"
        if not state_path.exists():
            return None
        state = json.loads(state_path.read_text(encoding="utf-8"))
        cells = state.get("cells")
        if isinstance(cells, dict):
            return set(cells.keys())
    except (OSError, ValueError):
        return None
    return None


def merge_store(
    cfg: LpoptConfig,
    from_dir: str | Path,
    *,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
    store_dir: str | Path | None = None,
    ledger: str | Path | None = None,
) -> MergeReport:
    """Merge a returned kit ``data/`` folder into the main store + ledger.

    ``store_dir`` / ``ledger`` override the deck's ``[model].store_dir`` /
    ``[produce].ledger`` targets (CLI: ``--store-dir`` / ``--ledger``).  They let
    a merge run against a SCRATCH COPY of the store+ledger — e.g. to rehearse a
    kit merge while the main producer is live — without editing the deck.  A
    relative override resolves against the CURRENT WORKING DIRECTORY (the natural
    CLI expectation); a deck-config value keeps resolving against the deck's
    parent as before.
    """
    import pandas as pd

    from .data.schema import SCHEMA_COLUMNS
    from .data.store import (
        RECORDS_NAME, StoreWriter, dedup_upsert, ensure_schema_columns,
    )

    log = log or (lambda m: print(m))
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    if store_dir is not None:
        store_dir = Path(store_dir)
        store_dir = store_dir if store_dir.is_absolute() else (Path.cwd() / store_dir)
    else:
        store_dir = Path(cfg.model.store_dir)
        store_dir = store_dir if store_dir.is_absolute() else (base / store_dir)
    if ledger is not None:
        main_ledger = Path(ledger)
        main_ledger = main_ledger if main_ledger.is_absolute() else (Path.cwd() / main_ledger)
    else:
        main_ledger = Path(cfg.produce.ledger)
        main_ledger = main_ledger if main_ledger.is_absolute() else (base / main_ledger)

    kit_records, kit_maps, kit_ledger = _resolve_kit_paths(Path(from_dir))
    if not kit_records.exists():
        raise KitError(f"kit records not found: {kit_records}")

    # A kit produced by an older PC predates the tail columns; back-fill them with
    # nulls so the merge never dies on a missing append-only column.
    incoming = ensure_schema_columns(pd.read_parquet(kit_records))
    if len(incoming):
        incoming = dedup_upsert(incoming[SCHEMA_COLUMNS])
    kit_rows = int(len(incoming))

    main_path = store_dir / RECORDS_NAME
    existing = (ensure_schema_columns(pd.read_parquet(main_path))
                if main_path.exists() else None)
    total_before = int(len(existing)) if existing is not None else 0

    # classify incoming vs existing by record_id + quality rank.  The rank
    # includes the flatness columns, so a kit row that upgrades a stored row ONLY
    # by carrying node_peak / map_cov counts as an upgrade and the merge persists
    # (``changed`` below is what decides whether the store is rewritten at all).
    def _rank_cols(df: "pd.DataFrame") -> tuple:
        peak = df["node_peak"] if "node_peak" in df.columns else [None] * len(df)
        cov = df["map_cov"] if "map_cov" in df.columns else [None] * len(df)
        return peak, cov

    existing_rank: dict[str, int] = {}
    if existing is not None and len(existing):
        e_peak, e_cov = _rank_cols(existing)
        for rid, cv, vl, pk, mc in zip(
            existing["record_id"].astype(str),
            existing["converged"], existing["valid"], e_peak, e_cov,
        ):
            existing_rank[rid] = _quality_rank_row(cv, vl, pk, mc)

    i_peak, i_cov = _rank_cols(incoming) if kit_rows else ([], [])
    new_rows = upgraded = duplicate = 0
    for rid, cv, vl, pk, mc in zip(
        incoming["record_id"].astype(str) if kit_rows else [],
        incoming["converged"] if kit_rows else [],
        incoming["valid"] if kit_rows else [],
        i_peak, i_cov,
    ):
        rank = _quality_rank_row(cv, vl, pk, mc)
        if rid not in existing_rank:
            new_rows += 1
        elif rank > existing_rank[rid]:
            upgraded += 1
        else:
            duplicate += 1

    # per-campaign converged BEFORE (P & converged)
    def _conv_by_campaign(df: "pd.DataFrame | None") -> dict[str, int]:
        if df is None or not len(df):
            return {}
        p = df[(df["dataset"] == "P") & (df["converged"] == True)]  # noqa: E712
        if not len(p):
            return {}
        return {str(k): int(v) for k, v in p.groupby("campaign")["record_id"].count().items()}

    before_conv = _conv_by_campaign(existing)

    # projected merged frame (also what we persist when not dry-run)
    if kit_rows:
        if existing is not None and len(existing):
            combined = pd.concat([existing, incoming[SCHEMA_COLUMNS]], ignore_index=True)
            merged = dedup_upsert(combined)
        else:
            merged = dedup_upsert(incoming[SCHEMA_COLUMNS])
    else:
        merged = existing if existing is not None else incoming
    after_conv = _conv_by_campaign(merged)
    total_after = int(len(merged)) if merged is not None else 0

    # flag campaigns that are not recognizable curriculum cell ids
    known = _known_cells(cfg)
    campaigns_incoming = (
        sorted({str(c) for c in incoming["campaign"].dropna().unique()}) if kit_rows else []
    )
    flagged: list[dict[str, str]] = []
    for camp in campaigns_incoming:
        if not is_recognized_cell(camp):
            flagged.append({"campaign": camp, "reason": "not a canonical cell id"})
        elif known is not None and camp not in known:
            flagged.append({"campaign": camp, "reason": "cell id not in this PC's curriculum grid"})

    per_campaign_names = sorted(set(before_conv) | set(after_conv))
    per_campaign = [
        {"campaign": c, "before": before_conv.get(c, 0), "after": after_conv.get(c, 0)}
        for c in per_campaign_names
    ]

    # -- persist (skip a pure no-op so we never rewrite the store needlessly) -- #
    changed = (new_rows > 0 or upgraded > 0)
    maps_new = 0
    if not dry_run and changed:
        StoreWriter(store_dir).write_records(incoming, append=True)
    if kit_maps.exists():
        maps_new = _merge_maps(store_dir, kit_maps, dry_run=dry_run)

    ledger_stats = _merge_ledger(main_ledger, kit_ledger, dry_run=dry_run)

    report = MergeReport(
        from_dir=str(from_dir), store_dir=str(store_dir), kit_rows=kit_rows,
        new_rows=new_rows, upgraded_rows=upgraded, duplicate_rows=duplicate,
        total_before=total_before, total_after=total_after, maps_new=maps_new,
        ledger=ledger_stats, per_campaign=per_campaign, flagged_campaigns=flagged,
        dry_run=dry_run,
    )
    return report


def _merge_maps(store_dir: Path, kit_maps: Path, *, dry_run: bool) -> int:
    """Merge kit EDIT5 map stacks into the store's ``maps.npz`` (keyed by
    record_id).  Returns the number of NEW keys added."""
    import numpy as np

    from .data.store import MAPS_NAME, StoreReader, StoreWriter

    with np.load(kit_maps) as npz:
        kit_keys = list(npz.files)
        payload = {k: npz[k] for k in kit_keys}
    if not payload:
        return 0
    existing_keys = set()
    main_maps = store_dir / MAPS_NAME
    if main_maps.exists():
        existing_keys = StoreReader(store_dir).maps_keys()
    new_keys = [k for k in kit_keys if k not in existing_keys]
    if new_keys and not dry_run:
        StoreWriter(store_dir).write_maps(payload, append=True)
    return len(new_keys)


__all__ = [
    "CELL_ID_RE",
    "GA80_PACKAGE_SUBDIRS",
    "KIT_DESIGN_PACKAGE_REL",
    "KIT_GA80_PACKAGE_REL",
    "KitError",
    "KitExport",
    "MergeReport",
    "FrontierKitExport",
    "KIT_MODEL_REL",
    "build_cell_stratum",
    "export_produce_kit",
    "export_frontier_kit",
    "frontier_roster_pairs",
    "ga80_package_root",
    "is_recognized_cell",
    "merge_store",
    "parse_cell_id",
    "render_kit_deck",
    "render_kit_readme",
    "render_frontier_kit_deck",
    "render_run_frontier_bat",
    "render_frontier_schtasks_xml",
]
