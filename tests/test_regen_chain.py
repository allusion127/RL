"""``regen_chain`` deck fidelity: the regenerated chain must be the SAME chain.

The whole point of ``regen_chain.py`` is that a re-run reproduces a stored
record's equilibrium.  That only holds if the two decks it synthesizes are the
decks the original bootstrap ran, so this checks both against the ONE retained
historical bootstrap on disk (``package/bootstrap_work/T5_T6_f101``, one of the
two surviving full cy1 chains in the whole program):

  * ``build_cycle1_deck(aliases, pair, cap_efpd=...)``  == the historical cy1 deck
  * ``build_reload_deck`` + ``replace_lpd_shf(pattern.to_shf())``
                                                        == the historical cy02 deck

and that ``historical_cy1_cap`` recovers the cap from the cell's own artifact.

Skipped when the package/retained work dirs are absent (fleet PCs ship no
``data/design/package``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

PKG = BASE / "data" / "design" / "package"
HIST = PKG / "bootstrap_work" / "T5_T6_f101"

pytestmark = pytest.mark.skipif(
    not (HIST / "cy1" / "MAS_INP").is_file(),
    reason="retained historical bootstrap (T5_T6_f101) not present",
)


def _hist_cy02() -> Path:
    return next((HIST / "master").glob("*/MAS_INP"))


def test_cy1_and_reload_decks_reproduce_the_historical_bootstrap():
    from lpopt.design.bootstrap import library_aliases
    from lpopt.design.coredeck import build_cycle1_deck, build_reload_deck
    from lpopt.vendor.masterrl.domain import Pattern
    from lpopt.vendor.masterrl.master import extract_lpd_shf, replace_lpd_shf

    aliases = library_aliases(PKG)
    hist_cy1 = (HIST / "cy1" / "MAS_INP").read_text(encoding="utf-8", errors="replace")
    # 579.4 EFPD = 2*B1/(241/101 + 1) with B1 = 981, the cap the run recorded.
    assert build_cycle1_deck(aliases, ("T5", "T6"), cap_efpd=579.4) == hist_cy1

    hist_cy02 = _hist_cy02().read_text(encoding="utf-8", errors="replace")
    pattern = Pattern.parse(extract_lpd_shf(hist_cy02))
    assert pattern.feed == 101
    rebuilt = replace_lpd_shf(
        build_reload_deck(aliases, "MAS_RST.APRQ_01_0579.40", 2), pattern.to_shf()
    )
    assert rebuilt == hist_cy02


def test_historical_cy1_cap_reads_the_cells_own_bootstrap_template():
    from regen_chain import historical_cy1_cap

    # T6_T4's cy02 template reads MAS_RST.APRQ_01_0620.00 -> a 620 EFPD cy1.
    assert historical_cy1_cap(PKG, "T6_T4") == pytest.approx(620.0)
    assert historical_cy1_cap(PKG, "NO_SUCH_CELL") is None
