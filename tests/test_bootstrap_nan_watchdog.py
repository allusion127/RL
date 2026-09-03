"""NaN watchdog on the `design bootstrap` MASTER path (S6 diagnosis P2).

MASTER V4.00 MOD3 has no NaN termination guard, so a divergent core loops on
``MGOUTER .. NaN NaN NaN`` until the wall timeout (3,600 s in the 2026-09-03 S6
failure).  `run_cycle1` and the equilibrium-chain runner now poll the growing
``MAS_OUT`` and kill MASTER with an explicit ``MasterDivergenceError``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from lpopt.design.bootstrap import (
    DEFAULT_BOOTSTRAP_TIMEOUT_S,
    MasterDivergenceError,
    _BootstrapMasterRunner,
    run_cycle1,
)
from lpopt.vendor.masterrl.master import MasterRunError

# A fake MASTER that diverges exactly the way the real one did: it floods
# MAS_OUT with non-finite outer-iteration lines and never exits.
_FAKE_NAN_MASTER = """\
import time
with open("MAS_OUT", "a", encoding="ascii") as fh:
    while True:
        fh.write("MGOUTER   11   20       NaN          NaN         NaN\\n")
        fh.write("****   0.000      NaN      NaN         NaN\\n")
        fh.flush()
        time.sleep(0.01)
"""

# A fake MASTER that behaves: a converging tail and one MAS_RST.
_FAKE_OK_MASTER = """\
with open("MAS_OUT", "a", encoding="ascii") as fh:
    fh.write("MGOUTER   11   20   1.0E-06   1.0E-06   1.0E-06\\n")
    fh.write("JOB FINISHED\\n")
open("MAS_RST.APRQ_01_0597.70", "w").write("rst")
"""


def _fake(tmp_path: Path, name: str, body: str) -> list[str]:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def _libs(tmp_path: Path) -> tuple[Path, Path]:
    xsl, hff = tmp_path / "MAS_XSL", tmp_path / "MAS_HFF"
    xsl.write_text("COMP FA_P0\n", encoding="utf-8")
    hff.write_text("hff\n", encoding="utf-8")
    return xsl, hff


def test_run_cycle1_kills_a_diverging_master(tmp_path: Path) -> None:
    """cy1 divergence raises MasterDivergenceError in seconds, not at timeout."""
    xsl, hff = _libs(tmp_path)
    exe = _fake(tmp_path, "fake_nan.py", _FAKE_NAN_MASTER)
    work = tmp_path / "cy1"

    start = time.perf_counter()
    with pytest.raises(MasterDivergenceError) as excinfo:
        run_cycle1("%JOB_TYP\n", xsl, hff, exe, work,
                   timeout_s=120.0,          # generous: only the watchdog can be this fast
                   nan_poll_s=0.3, nan_streak=6)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"watchdog did not kill cy1 promptly ({elapsed:.1f}s)"
    # work dir retained, with the sentinel + the NaN tail the diagnosis reads
    assert work.is_dir() and (work / "MAS_OUT").is_file()
    assert (work / "NONFINITE_FLUX").is_file()
    assert excinfo.value.work_dir == work
    assert "NaN" in str(excinfo.value)
    # a divergence is a MasterRunError, so the purging chain still trims/retains it
    assert isinstance(excinfo.value, MasterRunError)


def test_run_cycle1_healthy_master_is_untouched(tmp_path: Path) -> None:
    """A converging cy1 returns its restart; the watchdog never fires."""
    xsl, hff = _libs(tmp_path)
    exe = _fake(tmp_path, "fake_ok.py", _FAKE_OK_MASTER)
    work = tmp_path / "cy1"

    rst = run_cycle1("%JOB_TYP\n", xsl, hff, exe, work,
                     timeout_s=120.0, nan_poll_s=0.3, nan_streak=6)
    assert rst.name == "MAS_RST.APRQ_01_0597.70"
    assert not (work / "NONFINITE_FLUX").exists()


def test_run_cycle1_hang_still_times_out(tmp_path: Path) -> None:
    """A silent hang (no MAS_OUT at all) is still bounded by timeout_s."""
    from lpopt.design.bootstrap import BootstrapError

    xsl, hff = _libs(tmp_path)
    exe = _fake(tmp_path, "fake_hang.py", "import time\ntime.sleep(300)\n")
    work = tmp_path / "cy1"
    with pytest.raises(BootstrapError, match="timed out"):
        run_cycle1("%JOB_TYP\n", xsl, hff, exe, work,
                   timeout_s=2.0, nan_poll_s=0.3, nan_streak=6)
    assert work.is_dir()          # retained for diagnosis


def test_chain_runner_relabels_the_watchdog_kill(tmp_path: Path) -> None:
    """The equilibrium-chain runner reports divergence, not `exited with status`."""
    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "MAS_XSL").write_bytes(b"xsl")
    (pkg / "lib" / "MAS_HFF").write_bytes(b"hff")
    restart = tmp_path / "MAS_RST.FAKE.01"
    restart.write_bytes(b"r")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    exe = _fake(tmp_path, "fake_nan.py", _FAKE_NAN_MASTER)
    deck = (
        "%JOB_TYP\n        1       stead\n        MAS_RST.FAKE.01\n"
        "%LPD_SHF\n        F K1  0,\n%END\n"
    )

    runner = _BootstrapMasterRunner(
        pkg, exe, work_root=tmp_path / "work", timeout=120.0,
        keep_success=True, nan_poll_s=0.3, nan_streak=6,
    )
    start = time.perf_counter()
    with pytest.raises(MasterDivergenceError):
        runner.run(case_dir, deck_text=deck, restart_path=restart, use_cache=False)
    assert time.perf_counter() - start < 30.0
    assert list((tmp_path / "work").glob("*/NONFINITE_FLUX"))


def test_bootstrap_timeout_is_configurable_and_defaults_to_900(tmp_path: Path) -> None:
    """`[master].bootstrap_timeout_s` exists, defaults to 900, and is separate
    from the campaign `[master].timeout`."""
    from lpopt.config import MasterConfig, load_config

    assert MasterConfig().bootstrap_timeout_s == 900.0
    assert DEFAULT_BOOTSTRAP_TIMEOUT_S == 900.0

    deck = tmp_path / "d.inp"
    deck.write_text(
        '[flow]\ntitle = "t"\noutput_root = "runs/x"\n\n'
        '[master]\ntimeout = 3600\nbootstrap_timeout_s = 900\n',
        encoding="utf-8")
    cfg = load_config(deck)
    assert cfg.master.bootstrap_timeout_s == 900.0
    assert cfg.master.timeout == 3600          # campaign cap unchanged
