"""P0-1 — per-call wall-clock + MASTER/surrogate accounting in a real campaign.

``active_frontier_loop_spec_20260903.md`` §4d closes on the observation that
"lpopt 캠페인은 콜별 wall 을 기록하지 않는다 (`labels.jsonl`/`status.json` 모두
결측)" and makes fixing it P0's first task.  This runs the guided driver end to
end against the deterministic ``StubEvaluator`` and pins the three fields on
every label, the wave rows of ``events.jsonl``, and the launch/harvest KPI
artefacts the same run now produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.search.campaign import run_campaign
from tests.test_campaign_stub import _STORE, FakeModel, _cfg, _factory, _labels

pytestmark = pytest.mark.skipif(
    not (_STORE / "records.parquet").exists(), reason="no store present")


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "logs" / "events.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    tmp_path = tmp_path_factory.mktemp("kpi_campaign")
    out = tmp_path / "run"
    result = run_campaign(
        _cfg(tmp_path, budget=16), FakeModel(), _factory(), dry_run=True,
        run_dir=out, backend_factory=lambda ckpt: FakeModel(),
        early_stop=False, progress=False,
    )
    assert result.budget_spent == 16
    return out


def test_every_label_carries_the_three_accounting_fields(run_dir):
    labels = _labels(run_dir)
    assert len(labels) == 16
    for row in labels:
        assert "wall_s" in row
        assert "cumulative_master_calls" in row
        assert "cumulative_surrogate_evals" in row


def test_cumulative_master_calls_is_the_call_index(run_dir):
    calls = [row["cumulative_master_calls"] for row in _labels(run_dir)]
    assert calls == list(range(1, 17))


def test_cumulative_surrogate_evals_is_monotone_and_dominates_the_calls(run_dir):
    evals = [row["cumulative_surrogate_evals"] for row in _labels(run_dir)]
    assert all(evals[i] <= evals[i + 1] for i in range(len(evals) - 1))
    # screen_ratio is the point of the counter: many surrogate screens per call.
    assert evals[-1] > len(evals)


def test_wall_s_is_a_real_per_call_measurement(run_dir):
    walls = [row["wall_s"] for row in _labels(run_dir)]
    assert all(w is not None for w in walls)
    assert all(float(w) >= 0.0 for w in walls)
    assert any(float(w) > 0.0 for w in walls)


def test_events_wave_rows_carry_the_same_accounting(run_dir):
    waves = [e for e in _events(run_dir) if e.get("type") == "wave"]
    assert waves
    for event in waves:
        assert event["cumulative_master_calls"] == event["budget_spent"]
        assert event["cumulative_surrogate_evals"] >= event["budget_spent"]
        assert event["wall_s"] is not None and float(event["wall_s"]) >= 0.0
    # per-wave walls sum to the labels' total (a wave's cost, not its elapsed).
    total = sum(float(e["wall_s"]) for e in waves)
    label_total = sum(float(r["wall_s"]) for r in _labels(run_dir))
    assert total == pytest.approx(label_total, abs=1e-6)


def test_launch_freezes_the_baseline_and_writes_the_pre_snapshot(run_dir):
    base = json.loads((run_dir / "kpi_baseline.json").read_text(encoding="utf-8"))
    assert base["frozen"] is True
    assert base["pair"] == "K1_K2" and base["feed"] == 121
    assert base["prior_rows"] is not None
    pre = json.loads((run_dir / "ood_snapshot_pre.json").read_text(encoding="utf-8"))
    assert pre["kind"] == "pre"
    assert pre["coverage"]["in_distribution"] is not None


def test_harvest_writes_kpi_json_and_the_post_snapshot(run_dir):
    kpi = json.loads((run_dir / "kpi.json").read_text(encoding="utf-8"))
    assert kpi["A1_calls_to_frontier"]["valid"] is True
    assert kpi["K4"]["master_calls"] == 16
    assert kpi["K4"]["screen_ratio"] is not None
    assert kpi["K4"]["wall_hours"] is not None
    # the harvest runs AFTER the final status write, so kpi.json records the
    # run's terminal status rather than "running".
    assert kpi["status"] == "complete"
    post = json.loads((run_dir / "ood_snapshot_post.json").read_text(encoding="utf-8"))
    assert post["kind"] == "post"


def test_status_json_and_report_md_print_a1_a4(run_dir):
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert "kpi" in status
    assert "A1_calls_to_frontier" in status["kpi"]
    assert "A4_ood_frontier_updates" in status["kpi"]
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Sample-efficiency KPI (A1–A4)" in text
    assert "screen_ratio" in text
