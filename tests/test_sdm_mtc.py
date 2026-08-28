"""SDM/MTC post-verification harness (plan 12.5) — synthesis, units, verdicts.

All MASTER interaction is faked (``FakeExecutor``); no executable is invoked.
Covered: branch-deck synthesis round-trip (byte-diff limited to the swapped
sections, incl. a REAL ga80 deck when present), the audit-corrected MTC unit
conversion, keff two-point sign convention, SDM reactivity arithmetic vs hand
values, pass/fail verdicts, sidecar append/dedup, and the run_post_verification
end-to-end orchestration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lpopt.search import sdm_mtc as S
from lpopt.search.sdm_mtc import (
    BranchOutputs,
    BranchParams,
    CandidateRef,
    LicensingLimits,
    RodGroup,
    RodModel,
    build_branch_decks,
    load_sidecar,
    mtc_text_to_pcm_per_c,
    mtc_two_point_from_rows,
    parse_branch_keffs,
    parse_mtc_from_out,
    rho_pcm,
    run_post_verification,
    sdm_branch_labels,
    write_verdict_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
SYNTH_DECK = """%JOB_TYP
        1       stead                                                   # irrst, jobtyp
        MAS_RST.TEST_00_0100.00
        xsl     MAS_XSL
        out     MAS_OUT
        sum     MAS_SUM
%JOB_IDE
        APRQ    12
%JOB_TIT
        APR1400 Quarter reload CY12 test
%GEN_DIM
        10      10      27      83      85
%LPD_SHF
        1 K 10 2, 1 T 12 2, 1 K 17 2, F K2  0, F K2  0, F K2  0, F K2  0, F K2  0, F K2  0,
%LPD_HFF
        1       FA_A1
#################################################################################
%EXE_STD
        boron   eq      tr      1.0
/
%EXE_DEP
        0       0
/
%EDT_OPT
        1       0       0       0
/
END
"""

CONV_RESTART = "MAS_RST.CONV_12_0655.00"


def _small_rod_model() -> RodModel:
    return RodModel(
        groups=[
            RodGroup("R101", "R1", 1, 12),
            RodGroup("R201", "R2", 2, 12),
            RodGroup("R301", "R3", 3, 12),
        ],
        rod_map=[["R101", "o", "R201"], ["o", "R301", "o"]],
    )


def _sdm_params() -> BranchParams:
    return BranchParams(
        scram_banks=("R1", "R2", "R3"),
        stuck_candidate_banks=("R1", "R2", "R3"),
        rod_model=_small_rod_model(),
    )


def _exe_start(deck: str) -> int:
    return re.search(r"^[ \t]*%(?:EXE_|ROD_|EDT_OPT)", deck, re.M).start()


# --------------------------------------------------------------------------- #
# a fake MASTER runner: no executable, deterministic branch outputs
# --------------------------------------------------------------------------- #
class FakeExecutor(S.BranchExecutor):
    """Returns canned MAS_SUM/MAS_OUT keyed on branch kind (mtc/sdm)."""

    def __init__(self, mtc_keffs=(1.00000, 0.99900, 1.00100),
                 mtc_text="MTC   -20.00 PCM/C",
                 sdm_keffs=(1.00000, 0.90000, 0.93000, 0.95000, 0.94000)):
        self.mtc_keffs = mtc_keffs
        self.mtc_text = mtc_text
        self.sdm_keffs = sdm_keffs
        self.calls: list[str] = []

    @staticmethod
    def _sum(keffs) -> str:
        rows = ["", " SUMMARY EDIT 2 : REACTIVITY",
                " NO. DAY EFPD CYC-BU TOT-BU P PPM K-EFF ERRFLX REACT"]
        for i, k in enumerate(keffs, start=1):
            rows.append(f" {i} 0 0 0 10 0 900 {k:.6f} 1e-7 0")
        rows.append(" SUMMARY EDIT 3 : AO")
        return "\n".join(rows) + "\n"

    def run_branch(self, name, deck_text, restart_path) -> BranchOutputs:
        self.calls.append(name)
        if name.startswith("mtc"):
            return BranchOutputs(self._sum(self.mtc_keffs), self.mtc_text, 0.1)
        return BranchOutputs(self._sum(self.sdm_keffs), "", 0.1)


# --------------------------------------------------------------------------- #
# 1) MTC unit conversion — audit finding F1 (no magnitude ×10 heuristic)
# --------------------------------------------------------------------------- #
def test_mtc_unit_conversion_is_deterministic_and_audit_corrected():
    # A physical +5 pcm/°C must stay +5 (the pre-audit magnitude rescale turned
    # it into +50 and produced a false FAIL against the +9 window).
    assert mtc_text_to_pcm_per_c(5.0, "pcm_per_c") == 5.0
    assert mtc_text_to_pcm_per_c(-23.12, "pcm_per_c") == -23.12
    # 1e-4 Δρ/°C -> pcm/°C is exactly ×10.
    assert mtc_text_to_pcm_per_c(0.5, "drho_per_c_1e-4") == 5.0
    assert mtc_text_to_pcm_per_c(-2.312, "drho_per_c_1e-4") == pytest.approx(-23.12)
    with pytest.raises(ValueError):
        mtc_text_to_pcm_per_c(5.0, "bogus_units")


def test_rho_pcm_hand_value():
    assert rho_pcm(1.0) == 0.0
    assert rho_pcm(0.9) == pytest.approx((0.9 - 1.0) / 0.9 * 1e5, rel=1e-9)


def test_mtc_two_point_sign_and_arity():
    # base, T+Δ (k drops), T-Δ (k rises) -> negative MTC (correct physics).
    val = mtc_two_point_from_rows([1.00000, 0.99900, 1.00100], 5.0)
    assert val == pytest.approx(-20.0, abs=0.2)
    assert val < 0
    # anything but exactly 3 rows is ambiguous -> None (caller warns).
    assert mtc_two_point_from_rows([1.0, 0.999], 5.0) is None
    assert mtc_two_point_from_rows([1.0, 0.999, 1.001, 1.0], 5.0) is None
    with pytest.raises(ValueError):
        mtc_two_point_from_rows([1.0, 0.999, 1.001], 0.0)


# --------------------------------------------------------------------------- #
# 2) branch summary + MTC text parsers
# --------------------------------------------------------------------------- #
def test_parse_branch_keffs():
    txt = FakeExecutor._sum([1.0, 0.999, 1.001])
    assert parse_branch_keffs(txt) == [1.0, 0.999, 1.001]
    with pytest.raises(ValueError):
        parse_branch_keffs("no edit 2 here")


def test_parse_mtc_from_out_requires_unit_tag():
    # the echoed input card "mtc   0" must NOT be captured (audit T7-3).
    assert parse_mtc_from_out("        mtc     0\n") is None
    assert parse_mtc_from_out(" 00:10.61 1.001165  MTC   -23.12 PCM/C") == -23.12


# --------------------------------------------------------------------------- #
# 3) deck synthesis round-trip — byte-diff limited to swapped sections
# --------------------------------------------------------------------------- #
def test_mtc_deck_roundtrip_synthetic_prefix_byte_identical():
    prefix = SYNTH_DECK[: _exe_start(SYNTH_DECK)]
    # same restart -> the whole preserved prefix is byte-identical.
    decks = build_branch_decks(SYNTH_DECK, "MAS_RST.TEST_00_0100.00", "mtc", BranchParams())
    assert len(decks) == 1
    name, deck = decks[0]
    assert name == "mtc_hfp"
    assert deck.startswith(prefix)
    tail = deck[len(prefix):]
    assert "%EXE_STD" in tail and "%EXE_RHO" in tail and "mtc     1" in tail
    # the depletion chain + EOC restart write are GONE (branch is steady-state).
    assert "%EXE_DEP" not in tail and "%EDT_OPT" not in tail
    assert tail.rstrip().endswith("END")


def test_restart_reference_is_rewritten_only_line():
    decks = build_branch_decks(SYNTH_DECK, CONV_RESTART, "mtc", BranchParams())
    _name, deck = decks[0]
    prefix_new = deck[: _exe_start(deck)]
    prefix_old = SYNTH_DECK[: _exe_start(SYNTH_DECK)]
    # only the restart line differs; everything else in the prefix is preserved.
    assert CONV_RESTART in prefix_new
    assert "MAS_RST.TEST_00_0100.00" not in prefix_new
    diff_old = [l for l in prefix_old.splitlines() if "MAS_RST" in l]
    diff_new = [l for l in prefix_new.splitlines() if "MAS_RST" in l]
    # the non-restart lines are identical.
    rest_old = [l for l in prefix_old.splitlines() if "MAS_RST" not in l]
    rest_new = [l for l in prefix_new.splitlines() if "MAS_RST" not in l]
    assert rest_old == rest_new
    assert diff_old != diff_new


def test_sdm_deck_synthesis_cards_and_labels():
    params = _sdm_params()
    prefix = SYNTH_DECK[: _exe_start(SYNTH_DECK)]
    decks = build_branch_decks(SYNTH_DECK, "MAS_RST.TEST_00_0100.00", "sdm", params)
    name, deck = decks[0]
    assert name == "sdm_hzp"
    assert deck.startswith(prefix)
    tail = deck[len(prefix):]
    assert "%ROD_CFG" in tail and "%ROD_MAP" in tail
    # ARO (empty EXE_ROD) + ARI (scram banks) + one stuck case per candidate rod.
    assert tail.count("%EXE_STD") == 2 + 3          # aro, ari, + 3 stuck
    assert "R101  381." in tail and "R301  381." in tail
    assert sdm_branch_labels(params) == [
        "aro_critical", "ari", "stuck:R101:R1", "stuck:R201:R2", "stuck:R301:R3"
    ]


def test_build_branch_decks_validation():
    with pytest.raises(ValueError):
        build_branch_decks(SYNTH_DECK, "MAS_RST.X_0_0.0", "bogus", BranchParams())
    # SDM without a rod model must raise (never silently skip in the synth path).
    with pytest.raises(ValueError):
        build_branch_decks(SYNTH_DECK, "MAS_RST.X_0_0.0", "sdm", BranchParams())


def _find_real_ga80_deck() -> Path | None:
    for pat in (
        "runs/*/master/produce_cases/*/*/MAS_INP_cy*.inp",
        "data/**/MAS_INP_cy*.inp",
    ):
        for p in sorted(REPO_ROOT.glob(pat)):
            try:
                txt = p.read_text(encoding="latin-1")
            except OSError:
                continue
            if "%LPD_HFF" in txt and "%JOB_TYP" in txt and "MAS_RST." in txt and "%EXE" in txt:
                return p
    return None


def test_mtc_roundtrip_on_real_ga80_deck():
    deck_path = _find_real_ga80_deck()
    if deck_path is None:
        pytest.skip("no real ga80 MAS_INP deck present in runs/ or data/")
    base = deck_path.read_text(encoding="latin-1")
    restart = re.search(r"^\s*(MAS_RST\.\S+)", base, re.M).group(1)
    decks = build_branch_decks(base, restart, "mtc", BranchParams())
    _name, branch = decks[0]
    # same restart -> the whole geometry/composition/pattern prefix is byte-identical.
    prefix = base[: _exe_start(base)]
    assert branch.startswith(prefix), "prefix (geometry/composition) not preserved byte-for-byte"
    # and the ONLY new bytes are the swapped execution tail.
    assert branch[len(prefix):].lstrip().startswith("%EXE_STD")
    assert "%EXE_RHO" in branch and branch.rstrip().endswith("END")
    # the branch deck still validly references the converged restart by bare name.
    assert re.search(rf"^\s*{re.escape(restart)}\s*$", branch, re.M)


# --------------------------------------------------------------------------- #
# 4) SDM reactivity arithmetic vs hand values
# --------------------------------------------------------------------------- #
def test_sdm_arithmetic_and_verdict():
    # ARO k=1.0 -> rho 0 ; ARI k=0.90 -> -11111.1 ; stuck 0.93/0.95/0.94
    params = _sdm_params()
    out = BranchOutputs(FakeExecutor._sum([1.0, 0.90, 0.93, 0.95, 0.94]), "", 0.1)
    res = S._evaluate_sdm(out, "hzp", LicensingLimits(), params)
    assert res.rho_aro_pcm == pytest.approx(0.0, abs=1e-6)
    assert res.rho_ari_pcm == pytest.approx(-11111.11, abs=0.5)
    assert res.w_ari_pcm == pytest.approx(11111.11, abs=0.5)
    # worst stuck = the highest-rho rod = k=0.95 (R201).
    assert res.worst_stuck.rod_id == "R201"
    assert res.available_pcm == pytest.approx(5263.16, abs=1.0)
    # required 10870 -> negative margin -> FAIL.
    assert res.margin_pcm == pytest.approx(5263.16 - 10870.0, abs=1.0)
    assert res.pass_limit is False
    assert res.warning == ""


def test_sdm_monotonicity_warning():
    # a stuck rod more reactive than ARO (k above ARO) breaks ARO>=stuck>=ARI.
    params = _sdm_params()
    out = BranchOutputs(FakeExecutor._sum([1.0, 0.90, 1.05, 0.95, 0.94]), "", 0.1)
    res = S._evaluate_sdm(out, "hzp", LicensingLimits(), params)
    assert "monotonicity failed" in res.warning


def test_sdm_pass_when_margin_positive():
    # relax the requirement so the same net worth passes.
    params = _sdm_params()
    limits = LicensingLimits(sdm_required_pcm=5000.0)
    out = BranchOutputs(FakeExecutor._sum([1.0, 0.90, 0.93, 0.95, 0.94]), "", 0.1)
    res = S._evaluate_sdm(out, "hzp", limits, params)
    assert res.margin_pcm > 0
    assert res.pass_limit is True


# --------------------------------------------------------------------------- #
# 5) MTC verdicts (window enforcement)
# --------------------------------------------------------------------------- #
def test_mtc_verdict_in_and_out_of_window():
    p = BranchParams()
    limits = LicensingLimits()  # [-54, +9]
    # negative MTC at power -> PASS.
    neg = S._evaluate_mtc(BranchOutputs(FakeExecutor._sum([1.0, 0.999, 1.001]), "", 0), "hfp", limits, p)
    assert neg.value_pcm_per_c == pytest.approx(-20.0, abs=0.2)
    assert neg.pass_limit is True
    assert neg.source == "two_point_keff"
    # positive MTC above +9 -> FAIL (this is exactly the false-FAIL the audit fixed;
    # +5 must be read as +5 and pass, +12 must fail).
    pos_fail = S._evaluate_mtc(BranchOutputs(FakeExecutor._sum([1.0, 1.0012, 0.9988]), "", 0), "hfp", limits, p)
    assert pos_fail.value_pcm_per_c > 9.0
    assert pos_fail.pass_limit is False


def test_mtc_text_fallback_when_no_three_row_pattern():
    p = BranchParams(mtc_output_units="pcm_per_c")
    limits = LicensingLimits()
    # only 1 keff row -> two-point unavailable -> fall back to the PCM/C text.
    out = BranchOutputs(FakeExecutor._sum([1.001165]), "MTC   -23.12 PCM/C", 0)
    res = S._evaluate_mtc(out, "hfp", limits, p)
    assert res.value_pcm_per_c == pytest.approx(-23.12)
    assert res.source.startswith("EXE_RHO_text")
    assert res.pass_limit is True


def test_mtc_none_when_no_evidence():
    p = BranchParams()
    out = BranchOutputs(FakeExecutor._sum([1.001165]), "no mtc line here", 0)
    res = S._evaluate_mtc(out, "hfp", LicensingLimits(), p)
    assert res.value_pcm_per_c is None
    assert res.pass_limit is None
    assert "no MTC evidence" in res.warning


# --------------------------------------------------------------------------- #
# 6) run_post_verification end-to-end (fake MASTER) + verdict semantics
# --------------------------------------------------------------------------- #
def test_run_post_verification_mtc_only(tmp_path):
    cand = CandidateRef(
        record_id="rid_abc123", tag="K1_K2_deadbeef",
        deck_text=SYNTH_DECK, restart_path=tmp_path / CONV_RESTART,
        metrics={"cyclen": 654.6},
    )
    sidecar = tmp_path / "results.jsonl"
    results = run_post_verification(
        [cand], LicensingLimits(), master_cfg=None, work_root=tmp_path / "work",
        executor=FakeExecutor(), sidecar_path=sidecar,
    )
    assert len(results) == 1
    r = results[0]
    assert r.status == "ok"
    assert r.mtc.value_pcm_per_c == pytest.approx(-20.0, abs=0.2)
    assert r.pass_mtc is True
    # SDM deferred (no rod model) -> INCONCLUSIVE, not FAIL or PASS.
    assert r.pass_sdm is None
    assert r.verdict == "INCONCLUSIVE"
    assert "SDM skipped" in r.failure
    # sidecar written.
    rows = load_sidecar(sidecar)
    assert len(rows) == 1 and rows[0]["record_id"] == "rid_abc123"


def test_run_post_verification_with_sdm(tmp_path):
    cand = CandidateRef("rid_sdm", "tagX", SYNTH_DECK, tmp_path / CONV_RESTART)
    results = run_post_verification(
        [cand], LicensingLimits(sdm_required_pcm=5000.0), master_cfg=None,
        work_root=tmp_path / "work", sdm_params=_sdm_params(),
        executor=FakeExecutor(), sidecar_path=None,
    )
    r = results[0]
    assert r.pass_mtc is True
    assert r.pass_sdm is True
    assert r.verdict == "PASS"
    assert r.sdm.worst_stuck.rod_id == "R201"


def test_run_post_verification_requires_master_cfg_without_executor(tmp_path):
    cand = CandidateRef("rid", "tag", SYNTH_DECK, tmp_path / CONV_RESTART)
    with pytest.raises(ValueError):
        run_post_verification([cand], LicensingLimits(), master_cfg={}, work_root=tmp_path)


# --------------------------------------------------------------------------- #
# 7) sidecar append / dedup
# --------------------------------------------------------------------------- #
def test_sidecar_append_and_dedup(tmp_path):
    sidecar = tmp_path / "sdm_mtc" / "results.jsonl"
    r1 = S.SdmMtcResult(record_id="rid1", tag="a", status="ok", pass_mtc=True, pass_sdm=None)
    r2 = S.SdmMtcResult(record_id="rid2", tag="b", status="ok", pass_mtc=False, pass_sdm=None)
    S.append_sidecar(sidecar, r1)
    S.append_sidecar(sidecar, r2)
    assert len(load_sidecar(sidecar)) == 2
    # re-verify rid1 (last write wins) -> still exactly ONE row for rid1.
    r1b = S.SdmMtcResult(record_id="rid1", tag="a", status="ok", pass_mtc=False, pass_sdm=None)
    S.append_sidecar(sidecar, r1b)
    rows = load_sidecar(sidecar)
    assert len(rows) == 2
    by_id = {row["record_id"]: row for row in rows}
    assert by_id["rid1"]["pass_mtc"] is False       # updated, not duplicated
    # raw file has no duplicate record_id lines.
    raw = [l for l in sidecar.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(raw) == 2


# --------------------------------------------------------------------------- #
# 8) verdict table output
# --------------------------------------------------------------------------- #
def test_write_verdict_table(tmp_path):
    results = [
        S.SdmMtcResult(
            record_id="rid_pass", tag="cand1", status="ok", pass_mtc=True, pass_sdm=True,
            mtc=S.MtcResult(state="hfp", value_pcm_per_c=-20.0, pass_limit=True),
            sdm=S.SdmResult("hzp", 0.0, -11111.0, 11111.0, None, 5000.0, 6000.0, 1000.0, True),
        ),
    ]
    md = write_verdict_table(results, tmp_path, LicensingLimits())
    text = md.read_text(encoding="utf-8")
    assert "SDM / MTC post-verification" in text
    assert "cand1" in text and "-20.00" in text and "PASS" in text
    assert (tmp_path / "sdm_mtc_summary.csv").is_file()


# --------------------------------------------------------------------------- #
# 9) config section defaults
# --------------------------------------------------------------------------- #
def test_config_sdm_mtc_defaults():
    from lpopt.config import load_config
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    assert cfg.sdm_mtc.mtc_max_pcm_per_c == 9.0
    assert cfg.sdm_mtc.mtc_min_pcm_per_c == -54.0
    assert cfg.sdm_mtc.sdm_required_pcm == 10870.0
    assert cfg.sdm_mtc.top_k == 5
    # The scram / stuck scope MUST carry the shutdown banks A and B.  They hold
    # ~74 % of total CEA worth (A+B = 12.32 of 16.70 %drho, DCD Table 4.3-6), so
    # an R-only scope cannot reach the 10,870 pcm requirement for ANY pattern —
    # every candidate would FAIL on a config defect rather than on physics.
    # PSCEA ``P`` stays excluded (DCD quotes "Total without PSCEA").
    expected_banks = ["R1", "R2", "R3", "R4", "R5", "B", "A"]
    assert cfg.sdm_mtc.scram_banks == expected_banks
    assert cfg.sdm_mtc.stuck_candidate_banks == expected_banks
    assert "P" not in cfg.sdm_mtc.scram_banks

    # config and the branch spec must not drift apart
    from lpopt.search.sdm_mtc import BranchParams
    assert list(BranchParams().scram_banks) == expected_banks
    assert list(BranchParams().stuck_candidate_banks) == expected_banks
