"""Guard: a resume must re-run harness failures, not treat them as answers.

Regression test for the 2026-08-15 625-branch wave, which hit ``[Errno 28] No
space left on device`` partway through.  The original ``_done`` keyed off the
presence of a ``record_id`` in the results jsonl regardless of status, so a
re-launch would have skipped the 18 disk-failed chains and produced a wave that
looked complete but was 18 labels short.

The distinction under test is the project's own
(``verify.classify_outcome`` / ``PHYSICS_KILL_FAILURES``): a physics kill is an
honest negative about the PATTERN and is settled; a staging / disk / exit-status
failure is a fact about the BOX and is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation_wave import _done, is_settled                    # noqa: E402

DISK_FULL = (
    "MasterRunError: unexpected MASTER integration failure: "
    "[Errno 28] No space left on device (work directory retained at ...)"
)
EXIT_38 = "MasterRunError: MASTER exited with status 38 (work directory ...)"


@pytest.mark.parametrize(
    "status, failure, settled",
    [
        # settled -- a final answer about the pattern
        ("converged", "", True),
        ("nonconverged", "", True),
        ("error", "non_finite_flux", True),
        # NOT settled -- a fact about the box, must be re-run
        ("error", DISK_FULL, False),
        ("error", EXIT_38, False),
        ("error", "staging failed: template deck missing", False),
        ("error", "timeout", False),
        ("error", "", False),
    ],
)
def test_is_settled(status: str, failure: str, settled: bool) -> None:
    assert is_settled(status, failure) is settled


def test_done_excludes_harness_failures_only(tmp_path: Path) -> None:
    """``_done`` keeps physics answers and drops harness faults."""
    rows = [
        {"record_id": "aaa", "status": "converged", "failure": ""},
        {"record_id": "bbb", "status": "error", "failure": "non_finite_flux"},
        {"record_id": "ccc", "status": "error", "failure": DISK_FULL},
        {"record_id": "ddd", "status": "error", "failure": EXIT_38},
        {"record_id": "eee", "status": "nonconverged", "failure": ""},
    ]
    path = tmp_path / "ablation_results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    done = _done(path)

    assert done == {"aaa", "bbb", "eee"}
    # The two harness faults are the ones a resume must pick back up.
    assert {"ccc", "ddd"}.isdisjoint(done)


def test_done_tolerates_missing_file_and_junk(tmp_path: Path) -> None:
    assert _done(tmp_path / "nope.jsonl") == set()
    path = tmp_path / "ablation_results.jsonl"
    path.write_text(
        '{"record_id": "aaa", "status": "converged", "failure": ""}\n'
        "not json at all\n"
        "\n"
        '{"record_id": "bbb", "status": "error", "failure": "' + DISK_FULL.replace('"', '') + '"}\n',
        encoding="utf-8",
    )
    assert _done(path) == {"aaa"}
