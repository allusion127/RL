"""``lpopt.data.fxy`` — the MAS_OUT pin/assembly PLANAR peaking parser.

The fixture below is copied VERBATIM (spacing included) out of a retained final
cycle under ``runs/fpcamp_minfr_T6T4_r1/...``: the run of spaces inside
``PIN     PLANAR`` is the whole reason the regexes use ``\\s+``, so a fixture that
normalised it would test nothing.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lpopt.data.fxy import (
    FXY_GARBAGE_CEILING,
    FXY_GARBAGE_FLOOR,
    NONFINITE_SENTINEL,
    fxy_from_work_dir,
    parse_mas_out_fxy,
)

# Three real steps of a real MAS_OUT (FXYP 1.8186 / 1.8247 / 1.8110; the cycle
# max is therefore the SECOND step, not the first or the last -- a "take the
# last block" parser would silently pass a max-of-EOC test).
FIXTURE = """\
$P2D_1          0.000 DAY          0.000 EFPD

          MAXIMUM PIN     PLANAR POWER (FXYP)=      1.8186  AT (M ,14, 4, 6,11)
          MAXIMUM ASSMBLY  (3-D) POWER (FQA) =      1.8813  AT (M ,14,12)
          MAXIMUM ASSMBLY  (2-D) POWER (FRA) =      1.5046  AT (M ,14)
          MAXIMUM ASSMBLY PLANAR POWER (FXYA)=      1.6210  AT (M ,14, 3)
          MAXIMUM AXIAL          POWER (FZ)  =      1.2466  AT (12)
$P2D_2          5.000 DAY          5.000 EFPD

          MAXIMUM PIN     PLANAR POWER (FXYP)=      1.8247  AT (M ,14, 4, 6,11)
          MAXIMUM ASSMBLY PLANAR POWER (FXYA)=      1.6253  AT (M ,14, 3)
$P2D_3         15.000 DAY         15.000 EFPD

          MAXIMUM PIN     PLANAR POWER (FXYP)=      1.8110  AT (M ,14, 3, 6,11)
          MAXIMUM ASSMBLY PLANAR POWER (FXYA)=      1.6176  AT (M ,14, 3)
"""


def test_parses_cycle_max_over_every_step() -> None:
    res = parse_mas_out_fxy(FIXTURE)
    assert res.f_xy == pytest.approx(1.8247)     # max, NOT first and NOT last
    assert res.f_xya == pytest.approx(1.6253)
    assert res.n_steps == 3 and len(res.steps) == 3
    assert res.sane is True and res.reason == ""
    # per-step rows carry the EFPD; the BOC step's 0.0 must survive as 0.0.
    assert [s.efpd for s in res.steps] == [0.0, 5.0, 15.0]
    assert res.steps[0].fxyp == pytest.approx(1.8186)
    assert res.efpd_max == pytest.approx(15.0)


def test_variable_spacing_is_required_to_match() -> None:
    # a single-space rendering must ALSO match (the regex must not depend on the
    # exact run length), while a differently-named factor must not.
    single = "$P2D_1 0.0 DAY 0.0 EFPD\nMAXIMUM PIN PLANAR POWER (FXYP)= 1.5\n"
    assert parse_mas_out_fxy(single).f_xy == pytest.approx(1.5)
    frp_only = "$P2D_1 0.0 DAY 0.0 EFPD\nMAXIMUM PIN (2-D) POWER (FRP) = 1.5\n"
    assert parse_mas_out_fxy(frp_only).f_xy is None


def test_missing_field_returns_none_and_never_raises() -> None:
    for text in ("", "no P2D blocks here at all\n",
                 "$P2D_1 0.0 DAY 0.0 EFPD\nnothing useful\n"):
        res = parse_mas_out_fxy(text)
        assert res.f_xy is None and res.f_xya is None
        assert res.n_steps == 0 and res.sane is False and res.reason == "no_fxyp"
    # FRP is never substituted for a missing FXYP.
    assert parse_mas_out_fxy("MAXIMUM PIN (2-D) POWER (FRP) = 1.55").f_xy is None


def test_non_finite_and_starred_values_are_dropped_not_maxed() -> None:
    text = FIXTURE + (
        "$P2D_4         30.000 DAY         30.000 EFPD\n"
        "          MAXIMUM PIN     PLANAR POWER (FXYP)=         NaN  AT (M ,1)\n"
        "$P2D_5         60.000 DAY         60.000 EFPD\n"
        "          MAXIMUM PIN     PLANAR POWER (FXYP)=      ******  AT (M ,1)\n"
    )
    res = parse_mas_out_fxy(text)
    assert res.f_xy == pytest.approx(1.8247)   # the garbage steps contribute none
    assert res.n_steps == 3                    # ... and are not counted as steps
    assert len(res.steps) == 5                 # ... but the steps themselves are kept


def test_garbage_guard_flags_diverged_and_subunity_values() -> None:
    over = FIXTURE.replace("1.8247", "5.1656")   # a real measured divergence
    res = parse_mas_out_fxy(over)
    assert res.f_xy == pytest.approx(5.1656)
    assert res.sane is False and res.reason == "above_ceiling"

    under = FIXTURE.replace("1.8186", "0.5").replace("1.8247", "0.5") \
                   .replace("1.8110", "0.5")
    res = parse_mas_out_fxy(under)
    assert res.sane is False and res.reason == "below_floor"

    # the thresholds bracket every physical core (measured population max 2.12)
    assert FXY_GARBAGE_FLOOR == 1.0 and FXY_GARBAGE_CEILING == 3.0
    edge = FIXTURE.replace("1.8247", f"{FXY_GARBAGE_CEILING:.4f}")
    assert parse_mas_out_fxy(edge).sane is True     # inclusive bound


def test_fxy_from_work_dir_refuses_nonfinite_and_missing(tmp_path: Path) -> None:
    work = tmp_path / "wd"
    work.mkdir()
    assert fxy_from_work_dir(work) is None                    # no MAS_OUT
    (work / "MAS_OUT").write_text(FIXTURE, encoding="ascii")
    assert fxy_from_work_dir(work).f_xy == pytest.approx(1.8247)
    (work / NONFINITE_SENTINEL).write_text("", encoding="ascii")
    assert fxy_from_work_dir(work) is None                    # physics kill
    assert fxy_from_work_dir(tmp_path / "does-not-exist") is None


def test_path_and_text_inputs_agree(tmp_path: Path) -> None:
    path = tmp_path / "MAS_OUT"
    path.write_text(FIXTURE, encoding="ascii")
    assert parse_mas_out_fxy(path).f_xy == parse_mas_out_fxy(FIXTURE).f_xy
    assert parse_mas_out_fxy(str(path)).f_xy == pytest.approx(1.8247)


def test_cp949_bytes_decode_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "MAS_OUT"
    # cp949-only bytes (Hangul) that are NOT valid utf-8: the decode ladder must
    # fall through to cp949 rather than raise.
    path.write_bytes(FIXTURE.encode("ascii") + "  BY 김 ON MK\n".encode("cp949"))
    assert parse_mas_out_fxy(path).f_xy == pytest.approx(1.8247)


def _real_mas_out() -> Path | None:
    hits = list(Path("runs").glob("*/master/master_work/*/*/MAS_OUT"))
    return hits[0] if hits else None


def test_against_a_real_retained_mas_out() -> None:
    path = _real_mas_out()
    if path is None:
        pytest.skip("no MAS_OUT fixture under runs/")
    res = parse_mas_out_fxy(path)
    if res.f_xy is None:
        pytest.skip("retained MAS_OUT carries no P2D edit")
    assert res.n_steps >= 2 and len(res.steps) == res.n_steps
    assert 1.0 <= res.f_xy <= 3.0 and res.sane is True
    # FXYA <= FXYP: the pin planar peak bounds the assembly planar peak.
    assert res.f_xya is not None and res.f_xya <= res.f_xy + 1e-9
    assert res.efpd_max is not None and math.isfinite(res.efpd_max)
