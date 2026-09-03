"""Defect E.7-(a): the SCORED board and the VERIFIED/STORED board must be one board.

``selection.json`` records a prediction for the candidate the acquisition ranked;
``labels.jsonl`` / the store record the board MASTER actually ran.  Nothing used to
tie the two together, so any legalisation / canonicalisation / repair inserted
between scoring and verification would have detached every served prediction from
its label silently — and, in an offline rescore, would have been indistinguishable
from the campaign's own per-wave checkpoint swap (the misdiagnosis this test's fix
closes; see ``data/reports/fxy_head_prereg_20260829.md`` §E.7-(a)).

Two guards:

1. **Campaign-level (2 waves, StubEvaluator).**  Every selected candidate's
   ``pattern_digest`` in ``selection.json`` equals the digest of the pattern
   recorded for the same ``record_id`` in ``labels.jsonl``, AND the recorded
   pattern re-derives the served ``record_id``.  Wave >= 1 candidates are the
   local-search / exploit children — the population the defect was reported on —
   so the assertion is made per wave, not only in aggregate.
2. **Unit-level.**  :func:`~lpopt.search.verify.assert_scored_pattern_parity`
   passes on an untouched board, raises on a mutated one, and is inert when no
   digest was stamped (test doubles / resumed legacy entries).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.data.schema import compute_record_id, unpack_pattern
from lpopt.search.construct import CAMPAIGN_DECK_KNOBS
from lpopt.search.verify import (
    SCORED_DIGEST_KEY, ScoredPatternMismatch, assert_scored_pattern_parity)

from test_campaign_stub import FakeModel, _cfg, _factory, _labels  # noqa: F401

_STORE = Path(__file__).resolve().parents[1] / "data" / "store"


# --------------------------------------------------------------------------- #
# 1. campaign-level digest parity (2 waves)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_selected_candidates_are_stored_byte_identical(tmp_path):
    from lpopt.search.campaign import run_campaign

    cfg = _cfg(tmp_path, budget=16)                      # 2 waves x 8
    run_dir = tmp_path / "run"
    result = run_campaign(
        cfg, FakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        backend_factory=lambda ckpt: FakeModel(), early_stop=False, progress=False,
    )
    assert result.waves == 2
    assert result.budget_spent == 16

    stored = {l["record_id"]: l["record"]["pattern"] for l in _labels(run_dir)}
    assert len(stored) == 16

    seen_waves = 0
    for wdir in sorted((run_dir / "waves").glob("wave_*")):
        payload = json.loads((wdir / "selection.json").read_text(encoding="utf-8"))
        seen_waves += 1
        # the checkpoint that SERVED this wave is named, so a rescore is well-posed
        # even though the campaign re-serves a new champion every wave.
        assert payload["served_checkpoint"]
        assert payload["selection"]
        for row in payload["selection"]:
            rid = row["record_id"]
            assert rid in stored, f"{wdir.name}: selected row absent from labels"
            packed = stored[rid]
            pattern = unpack_pattern(packed)
            # (a) the board scored IS the board stored, byte for byte.
            assert row["pattern_digest"] == pattern.digest, (
                f"{wdir.name}/{rid}: scored digest {row['pattern_digest']} != "
                f"stored digest {pattern.digest}")
            # (b) and that board re-derives the record_id the prediction was
            #     served under (no silent re-keying between the two writes).
            assert compute_record_id(
                packed, cfg.model.library_id, cfg.case.pair, CAMPAIGN_DECK_KNOBS
            ) == rid
    assert seen_waves == 2


# --------------------------------------------------------------------------- #
# 2. the legalisation/parity guard itself
# --------------------------------------------------------------------------- #
def _a_pattern():
    import random

    from lpopt.search.construct import CaseContext, _heuristic_genome

    ctx = CaseContext(pair="K1_K2", feed=121, e_core=5.0)
    return _heuristic_genome(ctx, random.Random(0)).to_pattern()


def test_parity_guard_accepts_the_scored_board():
    pattern = _a_pattern()
    assert_scored_pattern_parity(
        pattern, {SCORED_DIGEST_KEY: pattern.digest, "record_id": "r0"}, stage="store"
    )


def test_parity_guard_rejects_a_modified_board():
    pattern = _a_pattern()
    # a single legal orbit swap — the smallest edit a legalisation/repair step
    # could make — must be caught, not absorbed.
    n = len(pattern.items)
    moved = None
    for i in range(n):
        for j in range(i + 1, n):
            try:
                cand = pattern.swap(i, j)      # orbit-legal swaps only
            except (ValueError, IndexError):
                continue
            if cand.digest != pattern.digest:
                moved = cand
                break
        if moved is not None:
            break
    assert moved is not None, "no legal single swap available on the seed board"
    with pytest.raises(ScoredPatternMismatch):
        assert_scored_pattern_parity(
            moved, {SCORED_DIGEST_KEY: pattern.digest, "record_id": "r0"}, stage="verify"
        )


def test_parity_guard_is_inert_without_a_stamp():
    pattern = _a_pattern()
    assert_scored_pattern_parity(pattern, {}, stage="store")
    assert_scored_pattern_parity(pattern, None, stage="verify")
