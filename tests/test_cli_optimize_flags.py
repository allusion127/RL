"""`lpopt optimize` flag wiring — the C2-4 `--allow-uncalibrated` escape hatch.

The hatch existed only as a `CampaignDriver` keyword and the
`LPOPT_ALLOW_UNCALIBRATED` env var, so an operator could not reach it from the
command line at all.  These tests pin BOTH halves of the wiring: the flag is
declared on the `optimize` sub-parser (default off), and `cmd_optimize` forwards
its value into `run_campaign` — deleting either half fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _deck(tmp_path: Path) -> Path:
    """A minimal `optimize`-able deck: fixed mode, a store dir and a model dir."""
    store = tmp_path / "store"
    model_dir = tmp_path / "model"
    store.mkdir()
    model_dir.mkdir()
    deck = tmp_path / "lpopt.inp"
    deck.write_text(
        '[case]\nmode = "fixed"\npair = "B1_C2"\nfeed = 121\n'
        f"[model]\nstore_dir = {json.dumps(str(store))}\n"
        f"model_dir = {json.dumps(str(model_dir))}\n",
        encoding="utf-8",
    )
    return deck


class _Result:
    """The proposals-only shape of `CampaignResult` that `cmd_optimize` prints."""

    status = "proposals_only"
    proposals: list = []
    run_dir = "runs/stub"


def _capture_run_campaign(monkeypatch) -> dict:
    """Replace `run_campaign` and `PosValCnnBackend.from_dir` with recorders."""
    seen: dict = {}

    from lpopt.model import model_api
    from lpopt.search import campaign as campaign_mod

    monkeypatch.setattr(model_api.PosValCnnBackend, "from_dir",
                        staticmethod(lambda *a, **k: object()))

    def _fake_run_campaign(cfg, model, evaluator_factory=None, **kwargs):
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(campaign_mod, "run_campaign", _fake_run_campaign)
    return seen


def test_optimize_declares_the_allow_uncalibrated_flag() -> None:
    """The flag exists on `optimize` and is OFF unless asked for."""
    from lpopt.cli import build_parser

    parser = build_parser()
    off = parser.parse_args(["optimize", "--input", "lpopt.inp"])
    assert off.allow_uncalibrated is False
    on = parser.parse_args(["optimize", "--input", "lpopt.inp",
                            "--allow-uncalibrated"])
    assert on.allow_uncalibrated is True


@pytest.mark.parametrize("flag, expected", [([], False), (["--allow-uncalibrated"], True)])
def test_cmd_optimize_forwards_the_hatch_to_run_campaign(
        tmp_path: Path, monkeypatch, capsys, flag: list[str], expected: bool) -> None:
    """THE WIRING TEST: the parsed flag must reach the driver.

    Dropping `allow_uncalibrated=` from the `run_campaign` call in `cmd_optimize`
    leaves the flag parseable and inert — exactly the state this change fixes.
    """
    from lpopt.cli import main

    seen = _capture_run_campaign(monkeypatch)
    rc = main(["optimize", "--input", str(_deck(tmp_path)), "--dry-run", *flag])
    capsys.readouterr()

    assert rc == 0
    assert seen["allow_uncalibrated"] is expected
