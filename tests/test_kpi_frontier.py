"""P0 of ``active_frontier_loop_spec_20260903.md`` §5 — the KPI instrument.

What is pinned here:

* **P0-1** every ``labels.jsonl`` / ``events.jsonl`` row carries ``wall_s`` +
  ``cumulative_master_calls`` + ``cumulative_surrogate_evals``, and the
  cumulative counters are monotone and consistent with the budget.
* **P0-2** A1/K2/K3/K4 computed from a synthetic run dir + synthetic store agree
  with the K1–K4 definitions in ``sample_efficiency_kpi_20260903.md`` §5, and a
  run with NO frozen baseline reports ``valid=false`` rather than a silent
  post-hoc A1.
* **P0-3** the A2 pre/post snapshot pair makes ``ΔMAE / n_new`` computable.
* **P0-4** the support-bin coverage replacement is a real predicate (and the
  mesh keeps ``feed == 121`` unless the flag is passed).
* **P0-5** ``report.md`` / ``status.json`` carry A1–A4 exactly when ``kpi.json``
  exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lpopt.data.store import StoreWriter
from lpopt.search import coverage
from lpopt.tools import kpi_calls_to_frontier as K

PAIR = "A1_B2"
FEED = 121
LIB = "paramA"


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
def _record(idx: int, *, f_r: float, campaign: str, feed: int = FEED,
            e_core: float = 4.90):
    from lpopt.data.schema import CanonicalRecord

    return CanonicalRecord(
        record_id=f"{idx:064x}", dataset="P", campaign=campaign, stratum=None,
        generator="test", parent_record_id=None,
        case_pair=PAIR, feed=feed, n_batches=40, depth2_edges=0,
        e_core=e_core, e_split=0.1, library_id=LIB, sym_class="C1",
        pattern="F:A:0", f_r=f_r, f_q=2.0, cbc_max=1200.0, cbc_boc=1100.0,
        cbc_kind="max", cyclen=620.0, ao_abs=0.1, cycle_burnup=None,
        discharge_burnup=None, max_assembly_burnup=None, max_pin_burnup=None,
        eoc_ppm=None, delta_efpd=None, n_cycles=12.0, converged=True,
        converged_at_cap=False, tolerance_margin=0.1,
        restart_provenance="pair_ecore:MAS_RST",
        valid=True, failure="", maps_key=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    """20 prior rows in the cell, record F_r = 1.500 (an OTHER campaign's)."""

    store_dir = tmp_path / "store"
    rows = [_record(i, f_r=1.50 + 0.01 * i, campaign="prior") for i in range(20)]
    StoreWriter(store_dir).write_records(rows, derive_enrichment_columns=False)
    return store_dir


def _run_dir(tmp_path: Path, best_by_call: list[float], *,
             name: str = "20260903_run", wave_size: int = 4,
             pred_offset: float = 0.05) -> Path:
    """A run dir whose N labels descend through ``best_by_call``.

    ``selection.json`` is written wave-by-wave with a ``pred_mean`` that is the
    true value plus ``pred_offset`` on the F_r column, so the A2 post MAE is a
    known constant.
    """

    run_dir = tmp_path / "runs" / name
    (run_dir / "waves").mkdir(parents=True, exist_ok=True)
    labels = []
    surrogate = 0
    per_wave: dict[int, list[dict]] = {}
    for i, f_r in enumerate(best_by_call):
        wave = i // wave_size
        if i % wave_size == 0:
            surrogate += 100
        labels.append({
            "wave": wave, "slot": "exploit", "origin": "test",
            "record_id": f"{i:064x}", "status": "converged",
            "feasible": f_r <= 1.55, "on_target": True,
            "wall_s": 10.0,
            "cumulative_master_calls": i + 1,
            "cumulative_surrogate_evals": surrogate,
            "record": {"record_id": f"{i:064x}", "case_pair": PAIR, "feed": FEED,
                       "library_id": LIB, "converged": True, "f_r": f_r,
                       "cbc_max": 1200.0, "f_q": 2.0, "cyclen": 620.0,
                       "ao_abs": 0.1},
        })
        per_wave.setdefault(wave, []).append({
            "slot": "exploit", "origin": "test", "record_id": f"c{i}",
            "pred_mean": [f_r + pred_offset, 1200.0, 2.0, 620.0, 0.1, None, None],
            "ood_flag": (i % 5 == 0),
        })
    (run_dir / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    for wave, entries in per_wave.items():
        wdir = run_dir / "waves" / f"wave_{wave:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "selection.json").write_text(
            json.dumps({"wave": wave, "tau": 1.0, "selection": entries}),
            encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "complete", "objective": "min_fr_max_cycle",
                    "case": f"{PAIR}-f{FEED}"}), encoding="utf-8")
    return run_dir


def _freeze(run_dir: Path, store_dir: Path) -> dict:
    payload = K.freeze_baseline(store_dir, pair=PAIR, feed=FEED, library_id=LIB,
                                campaign=run_dir.name)
    (run_dir / K.BASELINE_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# P0-2 — A1 / K2 / K3 / K4
# --------------------------------------------------------------------------- #
def test_freeze_baseline_is_the_cell_record_excluding_this_campaign(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.40] * 4)
    base = _freeze(run_dir, store)
    assert base["R_cell"] == pytest.approx(1.50)
    assert base["incumbent"] == pytest.approx(1.50)
    assert base["prior_rows"] == 20
    assert base["frozen"] is True


def test_a1_calls_to_frontier_at_epsilon(store, tmp_path):
    # best descends 1.60, 1.58, ..., crossing R_cell + 0.005 = 1.505 at call 6.
    curve = [1.60, 1.58, 1.56, 1.54, 1.52, 1.50, 1.49, 1.49]
    run_dir = _run_dir(tmp_path, curve)
    _freeze(run_dir, store)
    kpi = K.compute_kpi(run_dir, store)
    a1 = kpi["A1_calls_to_frontier"]
    assert a1["R_cell"] == pytest.approx(1.50)
    assert a1["epsilon"] == 0.005
    assert a1["calls"] == 6            # first call whose running best <= 1.505
    assert a1["reached"] is True
    assert a1["valid"] is True
    assert a1["final_gap"] == pytest.approx(-0.01)


def test_a1_not_reached_reports_gt_budget_and_final_gap(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.59, 1.58, 1.57])
    _freeze(run_dir, store)
    a1 = K.compute_kpi(run_dir, store)["A1_calls_to_frontier"]
    assert a1["calls"] is None and a1["reached"] is False
    assert a1["report"] == ">4"
    assert a1["final_gap"] == pytest.approx(0.07)


def test_post_hoc_a1_is_flagged_invalid(store, tmp_path):
    """§5 K1: an A1 computed without a pre-frozen record is INVALID."""

    run_dir = _run_dir(tmp_path, [1.60, 1.50, 1.49, 1.49])   # no kpi_baseline.json
    kpi = K.compute_kpi(run_dir, store)
    a1 = kpi["A1_calls_to_frontier"]
    assert a1["valid"] is False
    assert a1["frozen"] is False
    assert "post-hoc" in a1["note"]
    # the number is still produced (orientation), and the store rows of THIS
    # campaign are excluded so R_cell is the prior record, not this run's best.
    assert a1["R_cell"] == pytest.approx(1.50)


def test_k2_calls_to_incumbent_and_no_new_information(store, tmp_path):
    beat = _run_dir(tmp_path, [1.60, 1.55, 1.49, 1.49], name="beat")
    _freeze(beat, store)
    assert K.compute_kpi(beat, store)["K2_calls_to_incumbent"]["calls"] == 3

    flat = _run_dir(tmp_path, [1.60, 1.58, 1.57, 1.56], name="flat")
    _freeze(flat, store)
    k2 = K.compute_kpi(flat, store)["K2_calls_to_incumbent"]
    assert k2["calls"] is None
    assert "no new information" in k2["verdict"]


def test_k3_auf_only_at_horizons_the_run_actually_reached(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60] * 4)
    _freeze(run_dir, store)
    k3 = K.compute_kpi(run_dir, store)["K3_AUF"]
    # 4 calls: AUF@100 / AUF@300 are NOT comparable and must be absent.
    assert "AUF@100" not in k3 and "AUF@300" not in k3
    assert k3["AUF@4"] == pytest.approx(0.10)   # every call sits 0.10 above 1.50


def test_k4_companions_are_all_reported(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.54, 1.52, 1.50, 1.50, 1.50, 1.50, 1.50])
    _freeze(run_dir, store)
    k4 = K.compute_kpi(run_dir, store)["K4"]
    assert k4["prior_rows"] == 20
    assert k4["master_calls"] == 8
    assert k4["surrogate_evals"] == 200
    assert k4["screen_ratio"] == pytest.approx(25.0)
    assert k4["first_feasible_call"] == 2         # first f_r <= 1.55
    assert k4["n_feasible"] == 7
    assert k4["wall_s_total"] == pytest.approx(80.0)
    assert k4["wall_hours"] == pytest.approx(80.0 / 3600.0, abs=1e-3)
    assert k4["delta_per_100calls"] is not None


def test_delta_per_100calls_floor_verdict():
    flat = [{"best": 1.5} for _ in range(50)]
    assert K.delta_per_100calls(flat) == pytest.approx(0.0)
    assert K._floor_verdict(K.delta_per_100calls(flat)) == "floor"
    gaining = [{"best": 1.5 - 0.001 * i} for i in range(50)]
    assert K._floor_verdict(K.delta_per_100calls(gaining)) == "gaining"


def test_a4_counts_only_flagged_candidates_that_moved_the_best(store, tmp_path):
    # ood_flag fires on i % 5 == 0 -> calls 1, 6; the best moves on 1..4 only.
    run_dir = _run_dir(tmp_path, [1.60, 1.58, 1.56, 1.54, 1.54, 1.54, 1.54, 1.54])
    _freeze(run_dir, store)
    a4 = K.compute_kpi(run_dir, store)["A4_ood_frontier"]
    assert a4["n_calls"] == 8
    assert a4["n_ood_flagged"] == 2
    assert a4["n_frontier_updates"] == 4
    assert a4["n_ood_frontier_updates"] == 1       # call 1 only
    assert a4["n_ood_unknown"] == 0


def test_a4_none_flag_is_unknown_not_clean(tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.55])
    for path in (run_dir / "waves").glob("wave_*/selection.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload["selection"]:
            entry["ood_flag"] = None
        path.write_text(json.dumps(payload), encoding="utf-8")
    kpi = K.compute_kpi(run_dir, None)
    a4 = kpi["A4_ood_frontier"]
    assert a4["n_ood_unknown"] == 2 and a4["n_ood_flagged"] == 0


def test_write_kpi_lands_in_the_run_dir(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.50])
    _freeze(run_dir, store)
    path = K.write_kpi(run_dir, store)
    assert path == run_dir / "kpi.json"
    assert json.loads(path.read_text(encoding="utf-8"))["run"] == run_dir.name


def test_cli_writes_kpi_and_post_snapshot(store, tmp_path, capsys):
    run_dir = _run_dir(tmp_path, [1.60, 1.55, 1.50, 1.49])
    (run_dir / K.PRE_NAME).write_text(json.dumps({
        "kind": "pre", "pair": PAIR, "feed": FEED, "e_core": 4.90,
        "store_dir": str(store), "n": 20,
        "summary": {"f_r_ALL": 0.10, "f_r_n": 20.0},
    }), encoding="utf-8")
    rc = K.main(["--run", str(run_dir), "--store", str(store), "--post"])
    assert rc == 0
    assert (run_dir / "kpi.json").exists()
    assert (run_dir / "ood_snapshot_post.json").exists()
    out = capsys.readouterr().out
    assert "A1 calls-to-frontier" in out


def test_cli_freeze_never_overwrites_an_existing_baseline(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.20])
    _freeze(run_dir, store)
    before = (run_dir / K.BASELINE_NAME).read_text(encoding="utf-8")
    assert K.main(["--run", str(run_dir), "--store", str(store), "--freeze"]) == 0
    assert (run_dir / K.BASELINE_NAME).read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- #
# P0-3 — A2 before/after
# --------------------------------------------------------------------------- #
def test_snapshot_pre_uses_the_serving_path_and_records_coverage(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60])

    class _Pred:
        def __init__(self, mean):
            self.mean = mean

    def predict(patterns):
        # constant +0.10 on the F_r column against every prior row.
        return _Pred([[1.50 + 0.10 + 0.01 * i, 1200.0, 2.0, 620.0, 0.1, 0.0, 0.0]
                      for i in range(len(patterns))])

    pre = K.snapshot_pre(
        run_dir, store, pair=PAIR, feed=FEED, e_core=4.90, library_id=LIB,
        predict=predict, patterns_of=lambda row: row["pattern"],
        model_dir="s1g", promote_after=16,
    )
    assert pre["n"] == 20
    assert pre["summary"]["f_r_ALL"] == pytest.approx(0.10)
    cov = pre["coverage"]
    assert cov["in_distribution"] is True          # 20 labels >= promote_after 16
    assert cov["bin_labels"] == 20


def test_snapshot_pre_degrades_honestly_without_a_model(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60])
    pre = K.snapshot_pre(run_dir, store, pair=PAIR, feed=FEED, e_core=4.90,
                         library_id=LIB)
    assert pre["summary"] is None and pre["n"] == 0
    assert pre["coverage"]["in_distribution"] is True   # coverage still computed


def test_snapshot_post_makes_error_reduction_per_label_measurable(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.58, 1.56, 1.54], pred_offset=0.02)
    (run_dir / K.PRE_NAME).write_text(json.dumps({
        "kind": "pre", "pair": PAIR, "feed": FEED, "e_core": 4.90,
        "store_dir": str(store), "n": 20,
        "summary": {"f_r_ALL": 0.10, "f_r_n": 20.0},
    }), encoding="utf-8")
    post = K.snapshot_post(run_dir)
    assert post["n_new_labels"] == 4
    assert post["summary"]["f_r_ALL"] == pytest.approx(0.02)
    assert post["delta_mae"]["f_r_ALL"] == pytest.approx(0.08)
    assert post["error_reduction_per_label"]["f_r_ALL"] == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# P0-4 — support-bin coverage replaces the feed==121 constant
# --------------------------------------------------------------------------- #
def test_support_bins_need_promote_after_labels(tmp_path):
    store_dir = tmp_path / "s"
    rows = ([_record(i, f_r=1.5, campaign="c", feed=121, e_core=4.90)
             for i in range(16)]
            + [_record(100 + i, f_r=1.5, campaign="c", feed=109, e_core=4.90)
               for i in range(3)])
    StoreWriter(store_dir).write_records(rows, derive_enrichment_columns=False)
    bins, counts = coverage.support_bins(store_dir, promote_after=16)
    b121 = (121, coverage.e_bin(4.90))
    b109 = (109, coverage.e_bin(4.90))
    assert counts[b121] == 16 and counts[b109] == 3
    assert b121 in bins and b109 not in bins
    assert coverage.in_distribution(121, 4.90, bins) is True
    assert coverage.in_distribution(109, 4.90, bins) is False
    # the replaced constant would have said True for 121 at ANY label count,
    # and False for 109 even with 10,000 labels -- that is what A3 could not use.
    assert coverage.in_distribution(121, 9.99, bins) is False


def test_unknown_e_core_admits_on_the_feed_alone(tmp_path):
    bins = {(121, 98)}
    assert coverage.in_distribution(121, None, bins) is True
    assert coverage.in_distribution(109, None, bins) is False
    assert coverage.in_distribution(None, 4.9, bins) is False


def test_scoping_mesh_coverage_flag_is_off_by_default():
    """The mesh keeps ``feed == 121`` unless --coverage-in-distribution is passed."""

    src = (Path(__file__).resolve().parents[1] / "scoping_mesh.py").read_text(
        encoding="utf-8")
    assert "--coverage-in-distribution" in src
    assert 'action="store_true"' in src
    # the default branch of the helper is still the historic constant
    assert "return bool(feed_ == 121)" in src
    # and the two write sites now route through the helper, not the constant
    assert "in_distribution=bool(feed == 121)" not in src


# --------------------------------------------------------------------------- #
# P0-5 — report.md / status.json surface A1-A4 iff kpi.json exists
# --------------------------------------------------------------------------- #
def test_kpi_markdown_renders_all_four(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.55, 1.50, 1.49])
    _freeze(run_dir, store)
    text = "\n".join(K.kpi_markdown(K.compute_kpi(run_dir, store)))
    for token in ("A1 calls-to-frontier", "K2 calls-to-incumbent", "K3 AUF",
                  "A2 ΔMAE/label", "A3 coverage", "A4 OOD frontier updates",
                  "screen_ratio", "prior_rows", "Δ/100calls"):
        assert token in text


def test_kpi_markdown_marks_a_post_hoc_a1(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.49])       # unfrozen
    text = "\n".join(K.kpi_markdown(K.compute_kpi(run_dir, store)))
    assert "INVALID as A1" in text


def test_status_block_absent_without_kpi_json(tmp_path):
    run_dir = _run_dir(tmp_path, [1.60])
    assert K.kpi_status_block(run_dir) is None


def test_status_block_present_with_kpi_json(store, tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.49])
    _freeze(run_dir, store)
    K.write_kpi(run_dir, store)
    block = K.kpi_status_block(run_dir)
    assert block is not None
    for key in ("A1_calls_to_frontier", "A1_valid", "A4_ood_frontier_updates",
                "K2_calls_to_incumbent", "K3_AUF", "screen_ratio", "prior_rows"):
        assert key in block


def test_report_section_is_empty_without_kpi_json(tmp_path):
    from lpopt.report.report import _kpi_section

    run_dir = _run_dir(tmp_path, [1.60])
    assert _kpi_section(run_dir) == []


def test_report_section_renders_with_kpi_json(store, tmp_path):
    from lpopt.report.report import _kpi_section

    run_dir = _run_dir(tmp_path, [1.60, 1.49])
    _freeze(run_dir, store)
    K.write_kpi(run_dir, store)
    assert "## Sample-efficiency KPI (A1–A4)" in _kpi_section(run_dir)[0]


def test_cli_exposes_kpi_subcommand():
    from lpopt.cli import build_parser

    args = build_parser().parse_args(["kpi", "--run", "runs/x"])
    assert args.func.__name__ == "cmd_kpi"
    assert args.epsilon == 0.005 and args.metric == "f_r"
    assert args.post is False and args.freeze is False


# --------------------------------------------------------------------------- #
# trajectory invariants
# --------------------------------------------------------------------------- #
def test_trajectory_running_best_is_monotone_and_call_indexed(tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.70, 1.55, 1.58])
    traj = K.trajectory(K.read_labels(run_dir))
    assert [p["call"] for p in traj] == [1, 2, 3, 4]
    bests = [p["best"] for p in traj]
    assert bests == [1.60, 1.60, 1.55, 1.55]
    assert all(bests[i] >= bests[i + 1] for i in range(len(bests) - 1))
    assert traj[-1]["cumulative_wall_s"] == pytest.approx(40.0)


def test_labels_without_wall_are_not_charged_zero(tmp_path):
    run_dir = _run_dir(tmp_path, [1.60, 1.55])
    rows = K.read_labels(run_dir)
    for row in rows:
        row.pop("wall_s")
    (run_dir / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    traj = K.trajectory(K.read_labels(run_dir))
    assert all(p["wall_s"] is None for p in traj)
    assert traj[-1]["cumulative_wall_s"] == 0.0
