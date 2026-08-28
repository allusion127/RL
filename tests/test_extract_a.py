"""Dataset A extraction: fingerprint header, golden records, record_id
collision, EDIT5 harvest parse, case-dir join, and an end-to-end smoke run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from lpopt.config import load_config
from lpopt.data.edit5 import cbc_boc, cbc_max, extract_maps, parse_mas_sum, stack_maps
from lpopt.data.extract_a import (
    _normalize_spec,
    _unpad_fresh_batch,
    dedup_key_of,
    harvest,
    map_metrics,
    run_extract_a,
)
from lpopt.data.geometry import to_cache_key, to_canonical_from_cache_key, to_canonical_from_shf
from lpopt.data.schema import MOCHA_DECK_KNOBS, compute_record_id, pack_pattern, unpack_pattern
from lpopt.vendor.masterrl.domain import Pattern

REPO_ROOT = Path(__file__).resolve().parents[1]                 # 5_RL
DECK = REPO_ROOT / "lpopt.inp"
CASE_ROOT = (REPO_ROOT / ".." / "2_LP" / "0_Case").resolve()
MAIN_CACHE = CASE_ROOT / "sa_2b_cache.jsonl"
FUEL_PARQUET = REPO_ROOT / "data" / "store" / "fuel_types.parquet"


def _iter_records(path: Path, n: int) -> list[dict]:
    """First ``n`` record objects (skipping the line-1 fingerprint header)."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "key" not in obj:            # fingerprint header
                continue
            out.append(obj)
            if len(out) >= n:
                break
    return out


def _find_mas_sum() -> Path | None:
    for root in (CASE_ROOT / "runs", Path("D:/eqlp_ws/runs")):
        if not root.is_dir():
            continue
        for p in root.glob("*/cases/*/cy*/MAS_SUM"):
            if p.is_file():
                return p
    return None


# --------------------------------------------------------------------------- #
# fingerprint header handling
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MAIN_CACHE.exists(), reason="main cache not present")
def test_first_line_is_header_not_record() -> None:
    with open(MAIN_CACHE, "r", encoding="utf-8", errors="replace") as fh:
        first = json.loads(fh.readline())
    assert "key" not in first
    assert "fingerprint" in first


# --------------------------------------------------------------------------- #
# golden: first 3 records
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MAIN_CACHE.exists(), reason="main cache not present")
def test_golden_first3_records() -> None:
    for obj in _iter_records(MAIN_CACHE, 3):
        pattern = to_canonical_from_cache_key(obj["key"])       # validates conventions
        assert pattern.feed == 121
        assert pattern.items[0].is_fresh                        # centre fresh

        # metric mapping matches the raw JSON exactly.
        metrics = obj["rec"]["metrics"]
        mapped = map_metrics(metrics)
        assert mapped["f_r"] == metrics["max_frp"]
        assert mapped["f_q"] == metrics["max_fqp"]
        assert mapped["cbc_boc"] == metrics["boc_ppm"]
        assert mapped["cyclen"] == metrics["cycle_length_efpd"]
        assert mapped["ao_abs"] == metrics["max_abs_ao"]
        assert mapped["n_cycles"] == metrics["n_cycles"]


@pytest.mark.skipif(not MAIN_CACHE.exists(), reason="main cache not present")
def test_pattern_pack_unpack_roundtrip() -> None:
    obj = _iter_records(MAIN_CACHE, 1)[0]
    pattern = to_canonical_from_cache_key(obj["key"])
    packed = pack_pattern(pattern)
    assert unpack_pattern(packed).canonical() == pattern.canonical()


# --------------------------------------------------------------------------- #
# record_id collision (same pattern, two libraries -> distinct id)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MAIN_CACHE.exists(), reason="main cache not present")
def test_record_id_separates_libraries() -> None:
    obj = _iter_records(MAIN_CACHE, 1)[0]
    pattern = to_canonical_from_cache_key(obj["key"])
    canon = pattern.canonical()
    pair = "_".join(sorted(pattern.batch_feed()))

    rid_260624 = compute_record_id(canon, "260624", pair, MOCHA_DECK_KNOBS)
    rid_5851 = compute_record_id(canon, "5.8_5.1", pair, MOCHA_DECK_KNOBS)

    assert len(rid_260624) == 64 and len(rid_5851) == 64
    assert rid_260624 != rid_5851                    # library is part of identity

    # The vendor pattern-only digest CANNOT separate them: same pattern -> same
    # digest under both libraries (why record_id includes library_id).
    same_pattern = Pattern(pattern.items)
    assert pattern.digest == same_pattern.digest
    assert len(pattern.digest) == 16


# --------------------------------------------------------------------------- #
# EDIT5 harvest parse on a real MAS_SUM
# --------------------------------------------------------------------------- #
def test_edit5_real_mas_sum() -> None:
    mas_sum = _find_mas_sum()
    if mas_sum is None:
        pytest.skip("no real MAS_SUM found under the runs trees")
    summary = parse_mas_sum(mas_sum)

    boc = cbc_boc(summary)
    cmax = cbc_max(summary)
    assert cmax >= boc                               # max PPM >= BOC PPM
    assert boc > 0

    maps = extract_maps(summary)
    assert set(maps) == {"boc_power", "eoc_power", "eoc_burnup", "eoc_kinf"}
    for arr in maps.values():
        assert arr.shape == (9, 9)
        assert arr.dtype == np.float32
    # EOC burnup exceeds BOC assembly power scale and BOC burnup (depletion).
    assert np.nanmax(maps["eoc_burnup"]) > np.nanmax(maps["boc_power"])

    stack = stack_maps(summary)
    assert stack.shape == (4, 9, 9)


# --------------------------------------------------------------------------- #
# fresh-name padding normalization (5.8_5.1 re-harvest join fix, 20260720)
# --------------------------------------------------------------------------- #
def test_unpad_fresh_batch() -> None:
    assert _unpad_fresh_batch("A04") == "A4"          # single zero-pad stripped
    assert _unpad_fresh_batch("B01") == "B1"
    assert _unpad_fresh_batch("C01") == "C1"
    assert _unpad_fresh_batch("A4") == "A4"           # already bare -> no-op
    assert _unpad_fresh_batch("A10") == "A10"         # multi-digit -> untouched
    assert _unpad_fresh_batch("A0") == "A0"           # nothing after the 0 -> no-op


def test_normalize_spec_padding_symmetry() -> None:
    # padded (cache) and bare (MAS_INP deck) fresh specs collapse to one token …
    assert _normalize_spec("F:A04") == _normalize_spec("F:A4") == "F:A4r0"
    assert _normalize_spec("F:B05r0") == _normalize_spec("F:B5") == "F:B5r0"
    assert _normalize_spec("F:C01") == _normalize_spec("F:C1") == "F:C1r0"  # 260624
    # … while burnt-position (B:) specs are passed through verbatim (not fresh).
    assert _normalize_spec("B:(6,3)r2") == "B:(6,3)r2"


def test_dedup_key_padded_bare_join_and_no_false_merge() -> None:
    padded = [[1, 1, "F:A04"], [2, 1, "B:(6,3)r2"], [3, 1, "B:(6,7)r2"], [4, 1, "F:A04"]]
    bare = [[1, 1, "F:A4"], [2, 1, "B:(6,3)r2"], [3, 1, "B:(6,7)r2"], [4, 1, "F:A4"]]
    # the padded cache key and the bare deck key now yield the SAME dedup key …
    assert dedup_key_of(padded) == dedup_key_of(bare)
    # … but a genuinely different fresh type at a position still differs (no merge).
    other = [[1, 1, "F:A4"], [2, 1, "B:(6,3)r2"], [3, 1, "B:(6,7)r2"], [4, 1, "F:B4"]]
    assert dedup_key_of(bare) != dedup_key_of(other)


@pytest.mark.skipif(not MAIN_CACHE.exists(), reason="main cache not present")
def test_260624_join_rate_unchanged_by_padding_fix() -> None:
    """Regression: the 260624 case-dir join stays healthy under the padding fix.

    260624 decks label fresh types consistently on both sides, so the symmetric
    normalization is idempotent for them — a sample of a known 260624 run must
    still join its main-cache keys at a high rate (the full extract measured 87.7%).
    """
    target: set[tuple] = set()
    with open(MAIN_CACHE, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            obj = json.loads(line)
            if "key" in obj:
                target.add(dedup_key_of(obj["key"]))
    runs = sorted((CASE_ROOT / "runs").glob("20260705-*/cases"))
    if not runs:
        pytest.skip("no 20260705 260624 run present")
    case_dirs = sorted(p for p in runs[0].glob("*") if p.is_dir())[:300]
    if len(case_dirs) < 50:
        pytest.skip("too few case dirs to measure a rate")
    harvested, counts = harvest(case_dirs, target, workers=1, progress=False)
    parseable = len(case_dirs) - counts.get("no_cy", 0) - counts.get("inp_err", 0)
    rate = len(harvested) / max(1, parseable)
    assert rate >= 0.7, f"260624 join rate regressed to {rate:.2f}"


@pytest.mark.skipif(not MAIN_CACHE.exists(), reason="cache not present")
def test_5851_join_recovered_by_padding_fix() -> None:
    """The 5.8_5.1 join (0 matches before the fix) now recovers real case dirs."""
    stale = CASE_ROOT / "sa_2b_cache.stale-2c04d78c.jsonl"      # a 5.8_5.1 stale cache
    run = CASE_ROOT / "runs" / "20260623-032634_random_s1504408480" / "cases"
    if not (stale.exists() and run.is_dir()):
        pytest.skip("5.8_5.1 stale cache / run not present")
    target: set[tuple] = set()
    with open(stale, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            obj = json.loads(line)
            if "key" in obj:
                target.add(dedup_key_of(obj["key"]))
    case_dirs = sorted(p for p in run.glob("*") if p.is_dir())[:400]
    harvested, _ = harvest(case_dirs, target, workers=1, progress=False)
    assert len(harvested) > 0                          # was exactly 0 pre-fix
    cmax, stack = next(iter(harvested.values()))
    assert cmax > 0 and stack.shape == (4, 9, 9)


# --------------------------------------------------------------------------- #
# case-dir join: a known 260624 run's decks match cached records
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MAIN_CACHE.exists(), reason="main cache not present")
def test_harvest_join_matches_known_run() -> None:
    # target keys = every pattern in the main (260624) cache.
    target: set[tuple] = set()
    with open(MAIN_CACHE, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            obj = json.loads(line)
            if "key" in obj:
                target.add(dedup_key_of(obj["key"]))

    runs = sorted((CASE_ROOT / "runs").glob("20260703-*/cases"))
    if not runs:
        pytest.skip("no 20260703 260624 run present")
    case_dirs = sorted(p for p in runs[0].glob("*") if p.is_dir())[:60]
    if not case_dirs:
        pytest.skip("no case dirs in the known run")

    harvested, counts = harvest(case_dirs, target, workers=1, progress=False)
    assert len(harvested) > 0                         # the join actually matches
    cmax, stack = next(iter(harvested.values()))
    assert cmax > 0
    assert stack.shape == (4, 9, 9)
    assert stack.dtype == np.float16


# --------------------------------------------------------------------------- #
# end-to-end smoke
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (MAIN_CACHE.exists() and FUEL_PARQUET.exists()),
    reason="cache or fuel table not present",
)
def test_run_extract_a_smoke(tmp_path: Path) -> None:
    store = tmp_path / "store"
    reports = tmp_path / "reports"
    store.mkdir(parents=True)
    shutil.copy(FUEL_PARQUET, store / "fuel_types.parquet")

    cfg = load_config(DECK)
    cfg.extract.store_dir = str(store)
    cfg.extract.reports_dir = str(reports)

    result = run_extract_a(
        cfg, limit=200, workers=1, harvest_limit=200, progress=False
    )

    assert result["n_records"] == 200
    assert (store / "records.parquet").exists()
    assert (reports / "extract_report.md").exists()

    import pandas as pd

    df = pd.read_parquet(store / "records.parquet")
    assert len(df) == 200
    assert (df["feed"] == 121).all()
    assert (df["dataset"] == "A").all()
    assert df["record_id"].nunique() == 200
    report_text = (reports / "extract_report.md").read_text(encoding="utf-8")
    assert "Per-library record counts" in report_text


# --------------------------------------------------------------------------- #
# High-resolution harvest: EDIT5 full burnup trajectory + EDIT6 axial shape.
# Both were already parsed for the (4,9,9) stack and then discarded; harvesting
# them is not retroactive, so a record produced without them loses the
# resolution permanently (roadmap 20260725 items 1-2).
# --------------------------------------------------------------------------- #
_SYNTH_MAS_SUM = """
 SUMMARY EDIT 2 : REACTIVITY
    NO.     DAY    EFPD  CYC-BU  TOT-BU   P(%)     PPM    K-EFF  ERRFLX  REACT.
     1   0.000   0.000    0.00    0.00  100.0  1200.0  1.00000  1.0E-6   0.000
     2   5.000   5.000    1.00    1.00  100.0  1150.0  1.00000  1.0E-6   0.000
 SUMMARY EDIT 6 : AXIAL POWER
                     POWER(BOTTOM  --->  TOP)
       NO.     DAY     EFPD     2      3      4
        1   0.000   0.000  0.5000 1.2000 0.6000
        2   5.000   5.000  0.6000 1.1000 0.7000
"""


def test_edit6_axial_parsed_bottom_to_top() -> None:
    from lpopt.data.edit5 import stack_axial

    summary = parse_mas_sum(_SYNTH_MAS_SUM)
    assert len(summary.axial_rows) == 2
    axial = stack_axial(summary)
    assert axial.shape == (2, 3)                    # (n_steps, n_planes)
    assert axial.dtype == np.float32
    # row order follows (efpd, no); plane order is BOTTOM -> TOP as printed
    assert np.allclose(axial[0], [0.5, 1.2, 0.6])
    assert np.allclose(axial[1], [0.6, 1.1, 0.7])


def test_edit6_absent_is_not_fatal() -> None:
    """A MAS_SUM without EDIT 6 still parses; only stack_axial complains."""
    from lpopt.data.edit5 import stack_axial

    text = _SYNTH_MAS_SUM.split(" SUMMARY EDIT 6")[0]
    summary = parse_mas_sum(text)
    assert summary.axial_rows == []
    with pytest.raises(ValueError):
        stack_axial(summary)


def test_edit5_step_maps_superset_of_legacy_stack() -> None:
    """The full trajectory must REPRODUCE the legacy (4,9,9) BOC/EOC planes."""
    from lpopt.data.edit5 import stack_step_maps

    mas_sum = _find_mas_sum()
    if mas_sum is None:
        pytest.skip("no real MAS_SUM found under the runs trees")
    summary = parse_mas_sum(mas_sum)
    traj = stack_step_maps(summary)
    legacy = stack_maps(summary)

    n_steps = len(summary.edit5_maps)
    assert traj.shape == (n_steps, 3, 9, 9)
    assert traj.dtype == np.float32
    assert n_steps > 2                               # the whole point: >BOC/EOC
    # planes are (power, burnup, kinf); first step = BOC, last = EOC
    assert np.allclose(traj[0, 0], legacy[0], equal_nan=True)    # boc_power
    assert np.allclose(traj[-1, 0], legacy[1], equal_nan=True)   # eoc_power
    assert np.allclose(traj[-1, 1], legacy[2], equal_nan=True)   # eoc_burnup
    assert np.allclose(traj[-1, 2], legacy[3], equal_nan=True)   # eoc_kinf


def test_hires_harvest_never_raises_on_bad_result() -> None:
    """The harvest is best-effort: a junk result yields None, never an exception."""
    from lpopt.search.verify import _hires_from_equilibrium_result

    class _Junk:
        cycles = None
        retained_work_dirs = ("/nonexistent/dir/xyz",)

    assert _hires_from_equilibrium_result(_Junk()) is None
    assert _hires_from_equilibrium_result(object()) is None
