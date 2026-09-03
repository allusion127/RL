"""Library / deck gates G-H3, G-H3b, G-H3c, G-H5a, G-H5b, G-H5c (task #15) and
the ga80 curve-coverage audit (task #17).

The size arithmetic of G-H3 is frozen from six measured libraries (prereg sec.
4.3): the ``MAS_XSL`` intercept is **2,010 B** (the 2,008-byte ``MAS_REF`` plus
its CRLF), a gadolinia set costs 385,849 B, a set with ``n_gd == 0`` costs
377,461 B (the ``seq_canary`` V01/V02 precedent), and ``MAS_HFF`` is a flat
404,857 B per set.  It is an EQUALITY, no tolerance.

G-H5a exists because ``validate_reload_deck`` refuses any deck carrying
``%LPD_BCH`` — i.e. structurally refuses every cy1 fresh-core deck — so the cy1
deck needs its own gate; using the reload validator there is a defect, pinned as
a regression below.
"""

from __future__ import annotations

import pytest

from lpopt.design.coredeck import (
    REFLECTOR_COMPS,
    build_cycle1_deck,
    build_reload_deck,
    library_dims,
)
from lpopt.design.fuel_types import (
    CURVE_WITNESSES,
    audit_curve_coverage,
    coverage_deltas,
    format_curve_coverage,
)
from lpopt.design.library import (
    COMP_HEADER,
    LibraryGateError,
    comp_blocks,
    comp_header_counts,
    comp_nuclide_roster,
    expected_library_sizes,
    gate_comp_order,
    gate_comp_rosters,
    gate_convergence,
    gate_cycle1_deck,
    gate_library_sizes,
    gate_reload_deck,
)

# --------------------------------------------------------------------------- #
# G-H3 — the size equalities on six measured libraries
# --------------------------------------------------------------------------- #
#: ``(n_gd_sets, n_nogd_sets, MAS_XSL, MAS_HFF or None)`` — prereg sec. 4.3.
MEASURED = [
    (11, 0, 4_246_349, 4_453_427),      # lib.snap_20260811/MAS_XSL.bak
    (12, 0, 4_632_198, None),           # 0_APR1400/*/hgc
    (14, 2, 6_158_818, 6_477_712),      # seq_canary (V01/V02 have n_gd == 0)
    (33, 0, 12_735_027, 13_360_281),    # lib/MAS_XSL.bak
    (37, 0, 14_278_423, 14_979_709),    # current lib/MAS_XSL
    (80, 0, 30_869_930, None),          # 3_GA_Surrogate/FEASIBLE_PACKAGE
]


@pytest.mark.parametrize("n_gd,n_nogd,xsl,hff", MEASURED)
def test_size_equalities_reproduce_every_measured_library(n_gd, n_nogd, xsl, hff):
    exp_xsl, exp_hff = expected_library_sizes(n_gd, n_nogd)
    assert exp_xsl == xsl
    if hff is not None:
        assert exp_hff == hff
        gate_library_sizes(xsl, hff, n_gd, n_nogd)          # no raise


def test_slice_z_expectation_is_frozen():
    """N = 37 + 2 = 39, all ``n_gd > 0`` — the registered slice-Z numbers."""
    assert expected_library_sizes(39) == (15_050_121, 15_789_423)


def test_size_gate_has_no_tolerance():
    xsl, hff = expected_library_sizes(39)
    gate_library_sizes(xsl, hff, 39)
    with pytest.raises(LibraryGateError, match=r"MAS_XSL .*delta \+1"):
        gate_library_sizes(xsl + 1, hff, 39)
    with pytest.raises(LibraryGateError, match=r"MAS_HFF .*delta -1"):
        gate_library_sizes(xsl, hff - 1, 39)


def test_a_zero_gd_set_is_cheaper_by_a_known_amount():
    all_gd, _ = expected_library_sizes(16, 0)
    two_free, _ = expected_library_sizes(14, 2)
    assert all_gd - two_free == 2 * (385_849 - 377_461)


def test_size_gate_refuses_a_nonsense_roster():
    with pytest.raises(ValueError):
        expected_library_sizes(0)
    with pytest.raises(ValueError):
        expected_library_sizes(-1, 2)


# --------------------------------------------------------------------------- #
# G-H3b / G-H3c — COMP roster + order
# --------------------------------------------------------------------------- #
ROSTER = ("U235", "U238", "PU49", "BP01", "SB10", "FISP", "H2O", "RESI",
          "MACX", "CRD1")


def _comp_block(name: str, roster=ROSTER, header=COMP_HEADER) -> str:
    lines = [f"COMP {name}" + " " * 58 + "4.0",
             "-" * 72,
             " PROLOG 4.1 MOD 3              + XS GENERATION DATE: 2026-09-03",
             " BURN  VAR DMOD  ADF  DUM",
             "-" * 72,
             "   " + "".join(f"{v:>5}" for v in header)]
    for nuclide in roster:
        lines.append(f"{nuclide:<4}*                    1.00000E+00   0   0   0")
    return "\n".join(lines) + "\n"


def _xsl(names, *, roster_by_name=None, header_by_name=None) -> str:
    roster_by_name = roster_by_name or {}
    header_by_name = header_by_name or {}
    text = "REFL REF_R1\nRADIAL REFL   1 0 0 0 0\n 1.0\n"
    for name in names:
        text += _comp_block(name, roster_by_name.get(name, ROSTER),
                            header_by_name.get(name, COMP_HEADER))
    return text


def test_comp_blocks_and_roster_are_parsed():
    text = _xsl(["FA_T5", "FA_T6"])
    blocks = comp_blocks(text)
    assert list(blocks) == ["FA_T5", "FA_T6"]
    assert comp_nuclide_roster(blocks["FA_T5"]) == ROSTER
    assert comp_header_counts(blocks["FA_T6"]) == COMP_HEADER


def test_new_comp_blocks_must_match_the_incumbent_roster_exactly():
    incumbent = ["FA_T5", "FA_T6"]
    gate_comp_rosters(_xsl(incumbent + ["FA_T7", "FA_T8"]), ["FA_T7", "FA_T8"])


@pytest.mark.parametrize("bad_roster,needle", [
    (tuple(n for n in ROSTER if n != "BP01"), "BP01"),
    (tuple(n for n in ROSTER if n != "CRD1"), "CRD1"),
    (ROSTER + ("XX99",), "XX99"),
])
def test_a_dropped_or_added_nuclide_fails_g_h3b(bad_roster, needle):
    """A COUNT would not see this: ``BP01*``/``CRD1*`` are fields an ``irod=2``
    deck feeds, so a roster that differs by one entry is a silent core change."""
    text = _xsl(["FA_T5", "FA_T6", "FA_T7"],
                roster_by_name={"FA_T7": bad_roster})
    with pytest.raises(LibraryGateError, match="G-H3b"):
        gate_comp_rosters(text, ["FA_T7"])
    # the failing entry is named, not just counted
    with pytest.raises(LibraryGateError, match=needle):
        gate_comp_rosters(text, ["FA_T7"])


def test_a_wrong_comp_header_fails_g_h3b():
    text = _xsl(["FA_T5", "FA_T6"], header_by_name={"FA_T6": (62, 17, 5, 0, 0)})
    with pytest.raises(LibraryGateError, match="header"):
        gate_comp_rosters(text, ["FA_T6"])


def test_a_missing_new_set_fails_g_h3b():
    with pytest.raises(LibraryGateError, match="absent from MAS_XSL"):
        gate_comp_rosters(_xsl(["FA_T5"]), ["FA_T9"])


def test_comp_order_must_preserve_the_incumbent_prefix():
    before = [f"FA_T{i}" for i in range(6)]
    gate_comp_order(before, before + ["FA_T7", "FA_T8"], ["FA_T7", "FA_T8"])

    # R21: an alias sorted AHEAD of the incumbents rekeys every MAS_RST.*
    reordered = ["FA_A1"] + before + ["FA_T8"]
    with pytest.raises(LibraryGateError, match="COMP order changed at index 0"):
        gate_comp_order(before, reordered, ["FA_A1", "FA_T8"])

    with pytest.raises(LibraryGateError, match="appended sets"):
        gate_comp_order(before, before + ["FA_T8", "FA_T7"], ["FA_T7", "FA_T8"])

    with pytest.raises(LibraryGateError, match="fewer than"):
        gate_comp_order(before, before[:-1], [])


# --------------------------------------------------------------------------- #
# G-H5a / G-H5b — deck gates
# --------------------------------------------------------------------------- #
ALIASES = ["P0", "P1", "Z1", "Z2"]
DIMS = library_dims(len(ALIASES))


def _products(aliases=ALIASES) -> tuple[str, str]:
    """A miniature ``(MAS_XSL, MAS_HFF)``: the 5 reflector sets + the FA sets."""
    xsl = ("".join(f"REFL {name}\n" for _, name in REFLECTOR_COMPS)
           + "".join(f"COMP FA_{a}\n" for a in aliases))
    return xsl, "".join(f"FA_{a}\n" for a in aliases)


def test_cy1_deck_passes_its_own_gate():
    deck = build_cycle1_deck(ALIASES, ["Z1", "Z2"])
    xsl, hff = _products()
    gate_cycle1_deck(deck, ALIASES, expected_dims=DIMS, xsl_text=xsl, hff_text=hff)


def test_using_the_reload_validator_on_a_cy1_deck_is_a_defect():
    """THE G-H5a regression: ``validate_reload_deck`` structurally refuses a
    ``%LPD_BCH`` deck, so a cy1 deck run through it always fails — the gate must
    not be reused there."""
    from lpopt.search.assets import DeckValidationError, validate_reload_deck

    deck = build_cycle1_deck(ALIASES, ["Z1", "Z2"])
    with pytest.raises(DeckValidationError, match="LPD_BCH"):
        validate_reload_deck(deck, "MAS_RST.SEED.01", expected_dims=DIMS)


def test_cy1_gate_catches_a_stale_gen_dim_and_an_off_roster_batch():
    deck = build_cycle1_deck(ALIASES, ["Z1", "Z2"])
    with pytest.raises(LibraryGateError, match="GEN_DIM"):
        gate_cycle1_deck(deck, ALIASES, expected_dims=(40, 42))

    off = deck.replace("Z1", "Q9")
    with pytest.raises(LibraryGateError, match="outside the roster"):
        gate_cycle1_deck(off, ALIASES, expected_dims=DIMS)


def test_cy1_gate_checks_the_set_names_exist_in_the_products():
    deck = build_cycle1_deck(ALIASES, ["Z1", "Z2"])
    partial_xsl = "".join(f"COMP FA_{a}\n" for a in ALIASES[:-1])
    with pytest.raises(LibraryGateError, match=r"not in MAS_XSL.*FA_Z2"):
        gate_cycle1_deck(deck, ALIASES, expected_dims=DIMS, xsl_text=partial_xsl)


def test_reload_gate_is_the_reload_validator():
    deck = build_reload_deck(ALIASES, "MAS_RST.SEED.01", 12)
    gate_reload_deck(deck, "MAS_RST.SEED.01", expected_dims=DIMS)
    with pytest.raises(LibraryGateError, match="G-H5b"):
        gate_reload_deck(deck, "MAS_RST.SEED.01", expected_dims=(40, 42))


# --------------------------------------------------------------------------- #
# G-H5c — convergence
# --------------------------------------------------------------------------- #
def test_convergence_gate():
    gate_convergence(converged=True, n_cycles=11)          # T6_T4 precedent
    with pytest.raises(LibraryGateError, match="did not settle"):
        gate_convergence(converged=False, n_cycles=16)
    with pytest.raises(LibraryGateError, match="AT the cycle cap"):
        gate_convergence(converged=True, n_cycles=16, converged_at_cap=True)
    with pytest.raises(LibraryGateError, match="consecutive"):
        gate_convergence(converged=True, n_cycles=1)
    with pytest.raises(LibraryGateError, match="exceeds max_cycles"):
        gate_convergence(converged=True, n_cycles=17)


def test_convergence_at_the_cap_is_derived_not_trusted():
    """THE F9 regression: prereg line 339 excludes a chain that settled only AT
    the cap, and the gate must see it from ``n_cycles`` alone — the old
    ``converged_at_cap`` default (False) let the caller silence it by omission."""
    with pytest.raises(LibraryGateError, match="AT the cycle cap"):
        gate_convergence(converged=True, n_cycles=16, max_cycles=16)
    with pytest.raises(LibraryGateError, match="AT the cycle cap"):
        gate_convergence(converged=True, n_cycles=8, max_cycles=8)
    gate_convergence(converged=True, n_cycles=15, max_cycles=16)   # inside: fine


# --------------------------------------------------------------------------- #
# #17 — ga80 curve-coverage audit
# --------------------------------------------------------------------------- #
def _row(library_id: str, type_id: str, *, curves: bool) -> dict:
    row = {"library_id": library_id, "type_id": type_id}
    for cols in CURVE_WITNESSES.values():
        for col in cols:
            row[col] = 1.0 if curves else None
    return row


def _frame(rows):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(rows)


def test_curve_coverage_counts_the_sentinel_slots():
    """ga80: 70 rows carry a curve channel, 36 have the HGC — 34 on the 0.0
    sentinel.  The audit must produce that number, not assert 'no loss'."""
    rows = ([_row("ga80", f"G{i:02d}", curves=True) for i in range(36)]
            + [_row("ga80", f"G{i:02d}", curves=False) for i in range(36, 70)]
            + [_row("paramA", f"P{i:02d}", curves=True) for i in range(37)])
    report = audit_curve_coverage(_frame(rows), "ga80")

    assert report.n_rows == 70
    for group in CURVE_WITNESSES:
        assert report.present[group] == 36
        assert report.missing(group) == 34
    assert report.fraction("kinf_curve") == pytest.approx(36 / 70)
    assert report.missing_type_ids["kinf_curve"][0] == "G36"
    assert len(report.missing_type_ids["ff_curve"]) == 34

    # the paramA rows are fully covered — the shortfall is a ga80 property
    paramA = audit_curve_coverage(_frame(rows), "paramA")
    assert paramA.n_rows == 37 and paramA.missing("kinf_curve") == 0

    text = format_curve_coverage(report)
    assert "36/70" in text and "34 on the 0.0 sentinel" in text


def test_a_partly_harvested_row_does_not_count_as_covered():
    rows = [_row("ga80", "G00", curves=True), _row("ga80", "G01", curves=True)]
    rows[1]["kinf30"] = None                 # one witness column missing
    report = audit_curve_coverage(_frame(rows), "ga80")
    assert report.present["kinf_curve"] == 1
    assert report.missing_type_ids["kinf_curve"] == ["G01"]
    assert report.present["ff_curve"] == 2   # other groups unaffected


def test_a_column_absent_from_the_frame_reads_as_zero_not_full():
    rows = [_row("ga80", "G00", curves=True)]
    for row in rows:
        row.pop("pin_bu_r_inf")
    report = audit_curve_coverage(_frame(rows), "ga80")
    assert report.present["pin_bu_curve"] == 0
    assert report.absent_columns["pin_bu_curve"] == ["pin_bu_r_inf"]
    assert report.present["kinf_curve"] == 1


def test_the_increment_of_a_new_channel_set_is_measured_not_claimed():
    rows = ([_row("ga80", f"G{i:02d}", curves=True) for i in range(36)]
            + [_row("ga80", f"G{i:02d}", curves=False) for i in range(36, 70)])
    frame = _frame(rows)
    before = audit_curve_coverage(frame, "ga80",
                                  groups={"kinf_curve": CURVE_WITNESSES["kinf_curve"]})
    after = audit_curve_coverage(frame, "ga80",
                                 groups={"kinf_curve": CURVE_WITNESSES["ff_curve"]})
    # an HGC-derived channel is missing on exactly the rows the existing curve
    # channels already miss: increment 0, on a base of 34/70 (never "0 loss").
    assert coverage_deltas(before, after) == {"kinf_curve": 0}
    assert before.missing("kinf_curve") == 34


def test_audit_accepts_a_plain_list_of_mappings():
    rows = [_row("ga80", "G00", curves=True), _row("ga80", "G01", curves=False)]
    report = audit_curve_coverage(rows, "ga80")
    assert report.n_rows == 2 and report.present["kinf_curve"] == 1
    assert report.as_dict()["missing"]["kinf_curve"] == 1


def test_every_curve_witness_is_a_real_fuel_table_column():
    """The audit's own failure mode is silence: a witness column that does not
    exist in the frame reads as 0/N, so a rename in ``lpopt/data/fuel_types.py``
    would degrade the audit to 'nothing is covered' instead of erroring."""
    from lpopt.data.fuel_types import SCHEMA_COLUMNS

    declared = set(SCHEMA_COLUMNS)
    for group, cols in CURVE_WITNESSES.items():
        missing = [c for c in cols if c not in declared]
        assert not missing, f"{group}: {missing} are not fuel_types columns"
