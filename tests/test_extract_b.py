"""Dataset B extraction: event-log parse golden, digest join round-trip,
synthetic manifest fixture, dehydrated-skip behaviour, and an idempotent
end-to-end smoke run (plan 4.2, milestone M2-B)."""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import pytest

from lpopt.config import load_config
from lpopt.data import extract_b
from lpopt.data.extract_b import (
    _folder_of,
    build_pattern_index,
    map_fom,
    parse_case,
    run_extract_b,
    scan_event_logs,
    scan_manifest,
)
from lpopt.data.geometry import to_canonical_from_shf
from lpopt.data.schema import compute_record_id
from lpopt.search.genome import random_genome

REPO_ROOT = Path(__file__).resolve().parents[1]                 # 5_RL
DECK = REPO_ROOT / "lpopt.inp"
GA_ROOT = (REPO_ROOT / ".." / "3_GA_Surrogate").resolve()
RUNS_FLOW = GA_ROOT / "runs_flow"
AUDIT_LOG = RUNS_FLOW / "20260713_061541" / "stages" / "ga_generations_K1_K2.jsonl"
FUEL_PARQUET = REPO_ROOT / "data" / "store" / "fuel_types.parquet"

_RANK_DIR_RE = re.compile(r"^rank_\d+_(?P<digest>[0-9a-f]{16})$")


def _a_candidate_shf() -> Path | None:
    """A real ``rank_*_<digest>/loading_shf.txt`` under runs_flow, if any."""
    if not RUNS_FLOW.is_dir():
        return None
    for shf in RUNS_FLOW.rglob("loading_shf.txt"):
        if _RANK_DIR_RE.match(shf.parent.name):
            return shf
    return None


# --------------------------------------------------------------------------- #
# pure-function units (no data dependency)
# --------------------------------------------------------------------------- #
def test_parse_case() -> None:
    assert parse_case("K1_K2/feed-121") == ("K1_K2", 121)
    assert parse_case("J1_J2/feed-117") == ("J1_J2", 117)
    assert parse_case("K5_K6") == ("K5_K6", None)


def test_map_fom_ao_abs() -> None:
    fom = {"F_r": 1.55, "CBC_max": 1490.0, "F_q": 2.4, "cyclen": 651.0,
           "AO_min": -0.31, "AO_max": 0.12, "max_assembly_burnup": 60.5}
    m = map_fom(fom)
    assert m["f_r"] == 1.55
    assert m["cbc_max"] == 1490.0            # 3_GA CBC_max is the EDIT2 max
    assert m["f_q"] == 2.4
    assert m["cyclen"] == 651.0
    assert m["ao_abs"] == pytest.approx(0.31)   # max(|-0.31|, |0.12|)
    assert m["max_assembly_burnup"] == 60.5
    # string inputs (manifest CSV cells) coerce; missing AO -> None.
    assert map_fom({"F_r": "1.60"})["f_r"] == 1.60
    assert map_fom({"F_r": "1.60"})["ao_abs"] is None


def test_folder_of() -> None:
    assert _folder_of("K1_K2", 121) == "K1_K2"
    assert _folder_of("J1_J2", 117) == "J1_J2_f117"


# --------------------------------------------------------------------------- #
# event-log parse golden (real audit file, first entries)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not AUDIT_LOG.exists(), reason="audit event log not present")
def test_event_log_first_entries_golden() -> None:
    with open(AUDIT_LOG, "r", encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    assert first["case"] == "K1_K2/feed-121"
    batch = first["batch"]
    assert len(batch) == 16
    e0 = batch[0]
    # documented schema of a candidate entry.
    for key in ("digest", "fom", "eq_ok", "feasible", "parent_digest", "selected"):
        assert key in e0
    assert re.fullmatch(r"[0-9a-f]{16}", e0["digest"])
    assert e0["digest"] == "7b6ecfbb7e725884"          # golden first digest
    for key in ("F_r", "CBC_max", "F_q", "cyclen"):
        assert key in e0["fom"]


@pytest.mark.skipif(not AUDIT_LOG.exists(), reason="audit event log not present")
def test_audit_run_counts_600_70_17() -> None:
    """Plan M2 acceptance: 600 labels / 70 feasible / 17 errors."""
    _labels, run_stats, _total = scan_event_logs(RUNS_FLOW, progress=False)
    audit = [s for s in run_stats
             if s.campaign == "20260713_061541" and s.case == "K1_K2/feed-121"]
    assert len(audit) == 1
    s = audit[0]
    assert (s.entries, s.feasible, s.errors) == (600, 70, 17)


# --------------------------------------------------------------------------- #
# digest join round-trip (real candidate deck -> vendor digest == dir name)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_a_candidate_shf() is None, reason="no candidate decks present")
def test_digest_join_roundtrip() -> None:
    shf = _a_candidate_shf()
    assert shf is not None
    digest_from_dir = _RANK_DIR_RE.match(shf.parent.name).group("digest")
    pattern = to_canonical_from_shf(shf.read_text(encoding="utf-8"))
    assert pattern.digest == digest_from_dir       # the join key round-trips


@pytest.mark.skipif(_a_candidate_shf() is None, reason="no candidate decks present")
def test_build_pattern_index_recovers_candidate() -> None:
    shf = _a_candidate_shf()
    digest_from_dir = _RANK_DIR_RE.match(shf.parent.name).group("digest")
    index, stat = build_pattern_index(RUNS_FLOW, [], progress=False)
    assert digest_from_dir in index
    assert stat.unique_digests == len(index) > 0


# --------------------------------------------------------------------------- #
# synthetic manifest fixture (proves the manifest path while real ones are
# dehydrated)
# --------------------------------------------------------------------------- #
def _write_synthetic_manifest(root: Path) -> str:
    """Build a FEASIBLE_PACKAGE-shaped root with one readable seed; return pair."""
    genome = random_genome(random.Random(7), "K1_K2", 30)      # feed 121
    pattern = genome.to_pattern()
    pair = "_".join(sorted(pattern.batch_feed()))
    feed = pattern.feed
    seed_dir = root / "cores" / _folder_of(pair, feed) / "seed01"
    seed_dir.mkdir(parents=True)
    (seed_dir / "loading_shf.txt").write_text(pattern.to_shf(), encoding="utf-8")
    (seed_dir / "MAS_INP_cy12.inp").write_text("stub\n", encoding="utf-8")
    header = ("id,pair,feed,cell,F_r,CBC_max,F_q,cyclen,ncyc,"
              "AO_min,AO_max,max_assembly_burnup,max_pin_burnup,eq_ok")
    rowline = (f"seed01,{pair},{feed},5.2,1.512,1488.0,2.301,651.2,12,"
               "-0.09,0.11,60.4,70.1,True")
    (root / "manifest.csv").write_text(header + "\n" + rowline + "\n", encoding="utf-8")
    return pair


def test_synthetic_manifest_join(tmp_path: Path) -> None:
    root = tmp_path / "FEASIBLE_PACKAGE"
    pair = _write_synthetic_manifest(root)
    labels, stat = scan_manifest(root)
    assert stat.status == "OK"
    assert (stat.rows, stat.joined, stat.row_errors) == (1, 1, 0)
    assert len(labels) == 1
    ml = labels[0]
    assert ml.case_pair == pair
    assert ml.feed == 121
    assert ml.eq_ok is True
    assert ml.n_cycles == 12.0
    assert map_fom(ml.fom)["cbc_max"] == 1488.0


def test_manifest_missing_is_no_manifest(tmp_path: Path) -> None:
    labels, stat = scan_manifest(tmp_path / "does_not_exist")
    assert labels == []
    assert stat.status == "NO_MANIFEST"


# --------------------------------------------------------------------------- #
# dehydrated-skip behaviour (manifest exists but read raises OSError)
# --------------------------------------------------------------------------- #
def test_manifest_dehydrated_skip(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "ga_campaign_X"
    _write_synthetic_manifest(root)          # a real, readable manifest on disk

    def _boom(path):
        raise OSError(22, "cloud file provider is not running")

    # Simulate the OneDrive dehydrated placeholder: exists() passes, read fails.
    monkeypatch.setattr(extract_b, "_read_text_flex", _boom)
    labels, stat = scan_manifest(root)
    assert labels == []
    assert stat.status == "DEHYDRATED"
    assert stat.rows == 0                     # nothing ingested; run continues


def test_pattern_index_tolerates_dehydrated_cores(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "FEASIBLE_PACKAGE"
    _write_synthetic_manifest(root)          # creates cores/<pair>/seed01/loading_shf.txt
    empty_flow = tmp_path / "runs_flow_empty"
    empty_flow.mkdir()

    def _boom(path):
        raise OSError(22, "cloud file provider is not running")

    monkeypatch.setattr(extract_b, "_read_text_flex", _boom)
    index, stat = build_pattern_index(empty_flow, [root], progress=False)
    assert index == {}
    assert stat.read_errors >= 1              # the dehydrated seed was attempted
    assert root.name in stat.dehydrated_roots


# --------------------------------------------------------------------------- #
# end-to-end smoke + idempotent re-run
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (AUDIT_LOG.exists() and FUEL_PARQUET.exists()),
    reason="ga runs_flow or fuel table not present",
)
def test_run_extract_b_smoke_and_idempotent(tmp_path: Path) -> None:
    import pandas as pd

    store = tmp_path / "store"
    reports = tmp_path / "reports"
    store.mkdir(parents=True)
    shutil.copy(FUEL_PARQUET, store / "fuel_types.parquet")

    cfg = load_config(DECK)
    cfg.extract.store_dir = str(store)
    cfg.extract.reports_dir = str(reports)

    result = run_extract_b(cfg, progress=False)

    # audit ground truth surfaced by the run.
    assert result["audit_K1_K2"] == {"entries": 600, "feasible": 70, "errors": 17}
    # only pattern-recovered labels become rows; the rest are counted + dropped.
    assert result["n_records"] == result["event_recovered"] + result["manifest_records"]
    assert result["event_recovered"] > 0
    assert result["event_recovered"] + result["event_unrecovered"] == result["n_unique_labels"]

    records = store / "records.parquet"
    assert records.exists()
    df = pd.read_parquet(records)
    b = df[df["dataset"] == "B"]
    assert len(b) == result["n_records"]
    assert (b["library_id"] == "ga80").all()
    assert (b["sym_class"] == "free69").all()
    # Dataset B feeds are dominated by 121; feed-117 records (plan sec. 1-3:
    # "feed 117은 135건이 전부") surface as rows once the *_f117 candidate decks
    # hydrate from OneDrive, so accept the documented {117, 121} B feed set
    # rather than assuming the dehydrated-only feed-121 state.
    assert set(int(f) for f in b["feed"].unique()) <= {121, 117}
    assert (b["feed"] == 121).mean() > 0.5
    assert (b["restart_provenance"] == "ga_native").all()
    assert b["record_id"].nunique() == len(b)          # no intra-B dup

    # record_id is library-scoped: the same canonical under ga80 vs 260624 differ.
    row = b.iloc[0]
    rid_260624 = compute_record_id(row["pattern"], "260624", row["case_pair"],
                                   extract_b.GA_DECK_KNOBS)
    assert row["record_id"] != rid_260624

    report_text = (reports / "extract_report.md").read_text(encoding="utf-8")
    assert "Dataset B extraction report" in report_text
    assert extract_b._B_MARKER in report_text
    assert "MATCH" in report_text                      # audit call-out matched

    # idempotent re-run: append + dedup by record_id -> identical B row count.
    n_before = len(pd.read_parquet(records))
    run_extract_b(cfg, progress=False)
    df2 = pd.read_parquet(records)
    assert len(df2) == n_before
    assert (df2["dataset"] == "B").sum() == len(b)


@pytest.mark.skipif(
    not (AUDIT_LOG.exists() and FUEL_PARQUET.exists()),
    reason="ga runs_flow or fuel table not present",
)
def test_report_preserves_dataset_a_section(tmp_path: Path) -> None:
    store = tmp_path / "store"
    reports = tmp_path / "reports"
    store.mkdir(parents=True)
    reports.mkdir(parents=True)
    shutil.copy(FUEL_PARQUET, store / "fuel_types.parquet")
    # a pre-existing Dataset A report must survive a B run (and a B re-run).
    (reports / "extract_report.md").write_text(
        "# Dataset A extraction report\n\n- sentinel-A-line\n", encoding="utf-8"
    )

    cfg = load_config(DECK)
    cfg.extract.store_dir = str(store)
    cfg.extract.reports_dir = str(reports)

    run_extract_b(cfg, progress=False)
    run_extract_b(cfg, progress=False)           # twice: B section replaced, not duped
    text = (reports / "extract_report.md").read_text(encoding="utf-8")
    assert "sentinel-A-line" in text             # A section preserved
    assert text.count(extract_b._B_MARKER) == 1  # exactly one B section
    assert "Dataset B extraction report" in text
