"""``lpopt sdm-mtc`` as a standalone step: top-K plumbing (#21) + the REGISTERED
candidate ordering of ``select_topk_feasible`` (#21b).

No MASTER: ``run_post_verification`` / ``write_verdict_table`` are stubbed, so
what is under test is *which* candidates the CLI selects and *how many* — the two
decisions that decide where the licensing MASTER calls are spent.

#21b is a **confirmation** test, not a change: the selector orders by
``|cyclen - 625|`` and NOT by F_xy, which on a ``min_fxy`` arm can push the
PRIMARY candidate (minimum measured F_xy) out of the top-K.  The registered
response is to raise ``--top-k`` until the PRIMARY is inside (one extra MASTER
call per extra candidate) — never to re-rank the selector.  ``primary_rank`` below
is the verification helper that computes the K at which the PRIMARY enters; it
lives in the test (and in the runbook step it mirrors), deliberately NOT in the
production selector.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lpopt import cli
from lpopt.search import sdm_mtc

# --------------------------------------------------------------------------- #
# synthetic run dir
# --------------------------------------------------------------------------- #
def _candidate(run_dir: Path, pair: str, digest: str, *, record_id: str,
               cyclen: float | None, feasible: bool = True,
               restart: bool = True, cbc: float = 1300.0) -> Path:
    folder = run_dir / "candidates" / pair / digest
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"feasible": feasible, "cyclen": cyclen, "CBC_max": cbc,
            "F_r": 1.5, "F_q": 1.9, "AO_min": -0.02, "AO_max": 0.01,
            "extras": {"record_id": record_id}}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "MAS_INP1.inp").write_text(f"deck {record_id}\n", encoding="utf-8")
    if restart:
        (folder / "MAS_RST.001").write_bytes(b"restart")
    return folder


def _labels(run_dir: Path, rows: dict[str, float], *, nested: bool = True) -> None:
    lines = []
    for rid, fxy in rows.items():
        if nested:
            lines.append(json.dumps({"record_id": rid, "feasible": True,
                                     "record": {"record_id": rid, "f_xy": fxy}}))
        else:
            lines.append(json.dumps({"record_id": rid, "f_xy": fxy}))
    (run_dir / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status(run_dir: Path, objective: str) -> None:
    (run_dir / "status.json").write_text(
        json.dumps({"status": "complete", "objective": objective}), encoding="utf-8")


@pytest.fixture()
def minfxy_run(tmp_path: Path) -> Path:
    """A ``min_fxy`` run whose F_xy order DISAGREES with the cyclen order."""
    run = tmp_path / "runs" / "arm_a"
    run.mkdir(parents=True)
    # cyclen proximity to 625 ranks: near (2) < mid (10) < far (40)
    _candidate(run, "T6_T4", "aaaaaaaa", record_id="rid_far", cyclen=585.0)
    _candidate(run, "T6_T4", "bbbbbbbb", record_id="rid_mid", cyclen=615.0)
    _candidate(run, "T6_T4", "cccccccc", record_id="rid_near", cyclen=623.0)
    # ... the objective axis (min F_xy) ranks them the other way round.
    _labels(run, {"rid_near": 1.62, "rid_mid": 1.55, "rid_far": 1.49})
    _status(run, "min_fxy")
    return run


def _ids(candidates) -> list[str]:
    return [c.record_id for c in candidates]


# --------------------------------------------------------------------------- #
# the registered verification step (runbook-side, NOT production code)
# --------------------------------------------------------------------------- #
def _measured_fxy(run_dir: Path) -> dict[str, float]:
    """``record_id -> measured F_xy`` from ``labels.jsonl`` (last row wins)."""
    out: dict[str, float] = {}
    path = run_dir / "labels.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue                      # torn tail line: ignore, never crash
        rec = row.get("record") if isinstance(row.get("record"), dict) else {}
        rid = row.get("record_id") or rec.get("record_id")
        fxy = row.get("f_xy", rec.get("f_xy"))
        if rid is None or fxy is None:
            continue
        out[str(rid)] = float(fxy)
    return out


def primary_rank(run_dir: Path, max_k: int = 32) -> tuple[str | None, int | None]:
    """``(PRIMARY record_id, smallest K that includes it)``.

    PRIMARY = the feasible candidate with the minimum measured F_xy.  This is the
    preregistered pre-flight check for a ``min_fxy`` arm: if the returned K is
    larger than the deck's ``post_verify_top_k``, the registered response is to
    re-run with ``--top-k`` >= K, at the cost of one extra MASTER call per extra
    candidate.
    """
    fxy = _measured_fxy(run_dir)
    if not fxy:
        return None, None
    primary = min(fxy, key=lambda rid: fxy[rid])
    for k in range(1, max_k + 1):
        picked = _ids(sdm_mtc.select_topk_feasible(run_dir, k))
        if primary in picked:
            return primary, k
        if len(picked) < k:               # the archive is exhausted
            break
    return primary, None


# --------------------------------------------------------------------------- #
# #21b — the sort key is cyclen proximity, NOT F_xy
# --------------------------------------------------------------------------- #
def test_sort_key_is_cyclen_proximity_not_fxy(minfxy_run: Path) -> None:
    """REGISTERED property: even on a ``min_fxy`` run with a labels sidecar whose
    F_xy order is the exact reverse, the selector orders by ``|cyclen - 625|``."""
    assert _ids(sdm_mtc.select_topk_feasible(minfxy_run, 3)) == [
        "rid_near", "rid_mid", "rid_far"]


def test_primary_candidate_falls_outside_a_tight_top_k(minfxy_run: Path) -> None:
    """The registered trap itself: the PRIMARY (min measured F_xy = rid_far) is
    NOT in the top-1; the check DETECTS that and names the K that admits it."""
    assert _ids(sdm_mtc.select_topk_feasible(minfxy_run, 1)) == ["rid_near"]
    assert primary_rank(minfxy_run) == ("rid_far", 3)


def test_primary_rank_reports_the_k_to_re_run_with(tmp_path: Path) -> None:
    """PRIMARY already inside top-1 -> no re-run is needed (K = 1)."""
    run = tmp_path / "runs" / "agree"
    run.mkdir(parents=True)
    _candidate(run, "T6_T4", "aaaaaaaa", record_id="best", cyclen=624.0)
    _candidate(run, "T6_T4", "bbbbbbbb", record_id="other", cyclen=590.0)
    _labels(run, {"best": 1.40, "other": 1.80})
    _status(run, "min_fxy")
    assert primary_rank(run) == ("best", 1)


def test_labels_never_influence_the_selector(tmp_path: Path) -> None:
    """Regression guard against re-ranking: adding/removing a labels sidecar and
    changing the recorded objective must not move a single candidate."""
    run = tmp_path / "runs" / "labels_noop"
    run.mkdir(parents=True)
    _candidate(run, "T6_T4", "aaaaaaaa", record_id="far", cyclen=585.0)
    _candidate(run, "T6_T4", "bbbbbbbb", record_id="near", cyclen=623.0)

    assert _ids(sdm_mtc.select_topk_feasible(run, 2)) == ["near", "far"]
    _labels(run, {"near": 1.90, "far": 1.10})
    assert _ids(sdm_mtc.select_topk_feasible(run, 2)) == ["near", "far"]
    _status(run, "min_fxy")
    assert _ids(sdm_mtc.select_topk_feasible(run, 2)) == ["near", "far"]
    _status(run, "flat_power")
    assert _ids(sdm_mtc.select_topk_feasible(run, 2)) == ["near", "far"]


def test_infeasible_and_restartless_candidates_are_skipped(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "skips"
    run.mkdir(parents=True)
    _candidate(run, "T6_T4", "aaaaaaaa", record_id="ok", cyclen=620.0)
    _candidate(run, "T6_T4", "bbbbbbbb", record_id="no_restart", cyclen=624.0,
               restart=False)
    _candidate(run, "T6_T4", "cccccccc", record_id="infeasible", cyclen=625.0,
               feasible=False)
    # no cross-candidate restart substitution: the restart-less candidate is
    # dropped even though its cyclen key is better.
    assert _ids(sdm_mtc.select_topk_feasible(run, 5)) == ["ok"]


def test_missing_cyclen_falls_back_to_cbc(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "nocyclen"
    run.mkdir(parents=True)
    _candidate(run, "T6_T4", "aaaaaaaa", record_id="hi_cbc", cyclen=None, cbc=1400.0)
    _candidate(run, "T6_T4", "bbbbbbbb", record_id="lo_cbc", cyclen=None, cbc=1200.0)
    assert _ids(sdm_mtc.select_topk_feasible(run, 2)) == ["lo_cbc", "hi_cbc"]


def test_torn_labels_line_does_not_break_selection(minfxy_run: Path) -> None:
    with open(minfxy_run / "labels.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"record_id": "torn", "record": {"f_xy":')
    assert _ids(sdm_mtc.select_topk_feasible(minfxy_run, 3)) == [
        "rid_near", "rid_mid", "rid_far"]
    assert primary_rank(minfxy_run) == ("rid_far", 3)


# --------------------------------------------------------------------------- #
# #21 — the standalone CLI step and its top-K knob
# --------------------------------------------------------------------------- #
DECK = """\
[master]
executable = "C:/fake/master.exe"

[verify]
package_root = "pkg"
harvest_maps = true

[constraints]
mtc_enable = true
mtc_min_pcm_per_c = -54.0
mtc_max_pcm_per_c = 9.0
{extra}

[sdm_mtc]
top_k = 7
mtc_delta_c = 5.0
branch_timeout_s = 300.0
"""


@pytest.fixture()
def stub_master(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stand in for the MASTER branch runs; record what the CLI asked for."""
    seen: dict = {}
    real_select = sdm_mtc.select_topk_feasible

    def fake_run_post_verification(candidates, limits, master_cfg, out_dir, **kw):
        seen["n_candidates"] = len(candidates)
        seen["record_ids"] = [c.record_id for c in candidates]
        seen["limits"] = limits
        seen["master_cfg"] = master_cfg
        return [SimpleNamespace(verdict="PASS", violates=False, master_calls=1)
                for _ in candidates]

    def fake_select(run_dir, top_k):
        seen["top_k"] = top_k
        return real_select(run_dir, top_k)

    monkeypatch.setattr(sdm_mtc, "run_post_verification", fake_run_post_verification)
    monkeypatch.setattr(sdm_mtc, "select_topk_feasible", fake_select)
    monkeypatch.setattr(sdm_mtc, "write_verdict_table",
                        lambda results, run_dir, limits: run_dir / "verdict.md")
    return seen


def _run_cli(tmp_path: Path, run_dir: Path, extra: str, *argv: str) -> int:
    deck = tmp_path / "arm.inp"
    deck.write_text(DECK.format(extra=extra), encoding="utf-8")
    args = ["sdm-mtc", "--run", str(run_dir), "--input", str(deck), *argv]
    return cli.main(args)


def test_constraints_post_verify_top_k_is_the_knob_that_is_read(
    tmp_path: Path, minfxy_run: Path, stub_master: dict
) -> None:
    """``[constraints] post_verify_top_k`` wins over ``[sdm_mtc] top_k = 7``."""
    rc = _run_cli(tmp_path, minfxy_run, "post_verify_top_k = 5")
    assert rc == 0
    assert stub_master["top_k"] == 5
    assert stub_master["n_candidates"] == 3        # only 3 exist in the archive
    assert stub_master["record_ids"] == ["rid_near", "rid_mid", "rid_far"]
    assert stub_master["limits"].mtc_gated is True
    assert stub_master["master_cfg"]["timeout"] == 300.0


def test_cli_flag_overrides_the_deck(tmp_path: Path, minfxy_run: Path,
                                     stub_master: dict) -> None:
    assert _run_cli(tmp_path, minfxy_run, "post_verify_top_k = 5",
                    "--top-k", "1") == 0
    assert stub_master["top_k"] == 1
    assert stub_master["record_ids"] == ["rid_near"]


def test_sdm_mtc_top_k_is_plumbed_but_constraints_default_governs(
    tmp_path: Path, minfxy_run: Path, stub_master: dict
) -> None:
    """With no ``post_verify_top_k`` the loader's own default (3) applies — the
    ``[sdm_mtc] top_k = 7`` fallback only serves a deck that failed to load."""
    assert _run_cli(tmp_path, minfxy_run, "") == 0
    assert stub_master["top_k"] == 3


def test_top_k_zero_means_off_and_spends_no_master_call(
    tmp_path: Path, minfxy_run: Path, stub_master: dict
) -> None:
    """``post_verify_top_k = 0`` is documented as OFF (config.py:1240); it must
    not silently become 5 and buy ~5 unrequested MASTER calls.  DEVIATION from
    task #21's "0 lines of code": recorded in cmd_sdm_mtc's comment."""
    assert _run_cli(tmp_path, minfxy_run, "post_verify_top_k = 0") == 0
    assert "top_k" not in stub_master and "n_candidates" not in stub_master


def test_missing_run_dir_is_an_error(tmp_path: Path, stub_master: dict) -> None:
    assert _run_cli(tmp_path, tmp_path / "nope", "post_verify_top_k = 5") == 1
