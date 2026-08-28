"""Tests for ``lpopt.data.pinppi`` -- the HZ-weighted rod-average pin burnup
reduction (M2 close, 2026-08-20).

``tests/data/mas_ppi_k10_fixture.txt`` is a REAL-BYTE fixture: it is the
NPIN header line plus the K10 assembly block's FANAME line and its
"PIN 3-D BURNUP DISTRIBUTION" section, copied VERBATIM (byte-for-byte) out of
``runs/pinbu_wave_keep_f113pin5/ga80/master_work/worker_00/MAS_PPI.APRQ_22_0646.02``
-- the MAS_PPI produced by a real box-199 MASTER run (record_id
``2ad9de110b1d...``, ``[master] keep_success = true``). Only the intervening
records (2-8: fuel composition / number density / axial burnup / flux / pin
power / power coupling) are cut, since neither parser reads them -- the
FANAME line and the PIN 3-D BURNUP DISTRIBUTION rows are untouched, in their
original column layout, decimal formatting and whitespace.

The fixture exists to keep gate (a) -- this module's raw 3-D max must equal
``burnup.parse_ppi_max_pin_burnup``'s max exactly -- pinned in the repo
without committing an 11 MB full-core MAS_PPI file (69 assemblies; five of
them, one per f113-PIN core, were pulled from box 199 to validate this module
during development; the K10 one is the fixture's source).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lpopt.data.pinppi import (
    parse_ppi_assembly_rod_average,
    parse_ppi_core_rod_average_peak,
)
from lpopt.vendor.masterrl.burnup import _PPI_BLOCK, parse_ppi_max_pin_burnup

FIXTURE = Path(__file__).parent / "data" / "mas_ppi_k10_fixture.txt"

# Values independently confirmed on box 199, 2026-08-20 (full 11 MB file,
# 69 assemblies, record_id 2ad9de110b1d...): the deck's own SUMMARY EDIT5
# named K10 the max-assembly-burnup location, and
# burnup.parse_ppi_max_pin_burnup(text, "K", 10) on the untrimmed file
# returned exactly this value -- the harvested store row's max_pin_burnup.
EXPECTED_RAW_MAX = 84.202
EXPECTED_RAW_LOCATION = "K10/z8/i3/j3"
# lpopt.data.pinppi's own reduction on the SAME untrimmed file (69
# assemblies): the HZ-weighted rod-average peak, core-wide.
EXPECTED_ROD_AVG_MAX = 77.43448
EXPECTED_ROD_AVG_LOCATION = "K10/i3/j3"


@pytest.fixture(scope="module")
def fixture_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_fixture_is_one_real_assembly_block(fixture_text: str) -> None:
    blocks = list(_PPI_BLOCK.finditer(fixture_text))
    assert len(blocks) == 1
    assert blocks[0].group("x") == "K"
    assert blocks[0].group("y") == "10"
    assert blocks[0].group("nzc") == "25"


def test_gate_a_reproduces_node_peak(fixture_text: str) -> None:
    """PARSER CORRECTNESS GATE.  This module's raw (unweighted) 3-D max must
    equal ``burnup.parse_ppi_max_pin_burnup``'s max EXACTLY on the same real
    bytes -- if it ever does not, the two parsers have diverged on which
    numbers are pin burnups and neither's rod-average output can be trusted.
    """
    trusted = parse_ppi_max_pin_burnup(fixture_text, "K", 10)
    assert trusted.value == EXPECTED_RAW_MAX

    peak = parse_ppi_core_rod_average_peak(fixture_text)
    assert peak.raw_max == trusted.value
    assert peak.raw_max == EXPECTED_RAW_MAX
    assert peak.raw_max_location == EXPECTED_RAW_LOCATION


def test_rod_average_peak_on_real_bytes(fixture_text: str) -> None:
    peak = parse_ppi_core_rod_average_peak(fixture_text)
    assert peak.n_assemblies == 1
    assert peak.rod_avg_max == pytest.approx(EXPECTED_ROD_AVG_MAX, abs=1e-5)
    assert peak.rod_avg_max_location == EXPECTED_ROD_AVG_LOCATION
    # The physical relationship this module exists to quantify: the
    # rod-average (axially-smeared) peak is below the node (single-layer)
    # peak, by a factor in the DB's node/rod-average band (1.0886 +/- 0.0089,
    # data/reports/pinbu_audit_20260820.md).
    ratio = peak.raw_max / peak.rod_avg_max
    assert 1.06 < ratio < 1.12


def test_rod_average_uses_hz_weights_not_plain_mean(fixture_text: str) -> None:
    """HZ is uniform (15.24 cm) in this fixture, so a plain arithmetic mean
    would coincidentally equal the HZ-weighted mean here -- this test instead
    checks the weighting is actually applied (not silently dropped) by
    perturbing HZ and confirming the reduction moves.
    """
    match = next(_PPI_BLOCK.finditer(fixture_text))
    end = len(fixture_text)
    block = fixture_text[match.start() : end]
    baseline = parse_ppi_assembly_rod_average(fixture_text, block, match)

    # Reweight: give the bottom fuel plane (layer 0) 10x its HZ, leave the
    # rest untouched, by patching only the in-memory hz tuple used by the
    # reducer -- exercised via the private reducer to isolate the weighting
    # step from the text-parsing step.
    from lpopt.data.pinppi import _pin_burnup_grid, _reduce_grid

    npin, numeric_lines = _pin_burnup_grid(fixture_text, block, match)
    hz = list(baseline.hz)
    reweighted_hz = [hz[0] * 10.0] + hz[1:]
    reweighted = _reduce_grid(numeric_lines, npin, reweighted_hz)
    rod_avg_max_reweighted = reweighted[4]

    assert rod_avg_max_reweighted != pytest.approx(baseline.rod_avg_max, abs=1e-9)


def test_hz_line_rejects_wrong_count() -> None:
    from lpopt.vendor.masterrl.master import SummaryParseError
    import re

    bad_text = "  NPIN       :  16\n  FANAME  K   10  3    15.24  15.24\n"
    match = re.search(r"^\s*FANAME\s+(?P<x>[A-Z]+)\s+(?P<y>\d+)\s+(?P<nzc>\d+)\s+",
                       bad_text, re.MULTILINE)
    assert match is not None
    from lpopt.data.pinppi import _parse_hz_line

    with pytest.raises(SummaryParseError):
        _parse_hz_line(match)
