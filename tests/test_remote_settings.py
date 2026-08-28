"""remote.py [remote] TOML reader + defaults (M3b, plan 4.7).

No network: exercises only the settings parser (which must NOT import
lpopt.config — owned by a sibling agent)."""

from __future__ import annotations

import sys

from lpopt.remote import RemoteSettings


def test_defaults_match_plan() -> None:
    s = RemoteSettings.from_input(None)
    assert s.host == "HOST_238"
    assert s.user == "USER"
    assert s.port == 8022
    assert s.gpu == "auto"
    assert s.venv_python.endswith("/bin/python")
    assert s.home_rel("runs", "x") == "lpopt_ws/runs/x"


def test_reads_remote_table(tmp_path) -> None:
    deck = tmp_path / "deck.inp"
    deck.write_text(
        "[remote]\n"
        'host = "10.0.0.9"\n'
        'user = "bob"\n'
        "port = 2222\n"
        'gpu = 0\n'
        'workdir = "~/ws"\n'
        'env = "~/ws/venv"\n'
        "[flow]\ntitle = \"x\"\n",
        encoding="utf-8",
    )
    s = RemoteSettings.from_input(deck)
    assert (s.host, s.user, s.port, s.gpu) == ("10.0.0.9", "bob", 2222, 0)
    assert s.workdir == "~/ws"


def test_reader_does_not_import_lpopt_config() -> None:
    # remote.py must parse the deck without pulling in lpopt.config.
    import lpopt.remote as remote
    src = remote.__file__
    text = open(src, encoding="utf-8").read()
    assert "import lpopt.config" not in text
    assert "from lpopt.config" not in text
    assert "from .config" not in text
