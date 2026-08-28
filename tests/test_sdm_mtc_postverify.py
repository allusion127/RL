"""D9 SDM/MTC pre-delivery gate — selection, report-only semantics, accounting.

Decision D9 (2026-07-25): *"F_r 제외 feasible 노심에 대해서 (평탄도 높음) SDM, MTC
검증 실시"*.  The flatness objective dropped F_r, which is what lets it flatten —
and flattening monotonically degrades rod worth and raises leakage, which no
search axis measures.  This file covers the gate that closes that hole:

* the target set is the DELIVERY ranking (flat band, feasible EXCLUDING F_r);
* a candidate is verified against ITS OWN converged restart or skipped with a
  reason — never against a borrowed one;
* an enabled axis with no user limit is REPORT-ONLY: measured, never a violator;
* the extra MASTER calls are counted, including the ones that failed.

Every MASTER interaction is faked.  No executable is invoked anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.config import ConstraintsConfig, ConfigError, SdmMtcConfig
from lpopt.search import sdm_mtc as S

REPO_ROOT = Path(__file__).resolve().parents[1]

DECK = """%JOB_TYP
        1       stead
        MAS_RST.BASE_11_0000.00
        xsl     MAS_XSL
        out     MAS_OUT
        sum     MAS_SUM
%JOB_IDE
        APRQ    12
%GEN_DIM
        10      10      27      83      85
%LPD_SHF
        1 K 10 2, F K2  0,
%LPD_HFF
        1       FA_A1
%EXE_STD
        boron   eq      tr      1.0
/
END
"""


def _sum(keffs) -> str:
    rows = ["", " SUMMARY EDIT 2 : REACTIVITY",
            " NO. DAY EFPD CYC-BU TOT-BU P PPM K-EFF ERRFLX REACT"]
    for i, k in enumerate(keffs, start=1):
        rows.append(f" {i} 0 0 0 10 0 900 {k:.6f} 1e-7 0")
    return "\n".join(rows) + "\n"


class FakeExecutor(S.BranchExecutor):
    """Canned branch outputs; records which decks it was asked to run."""

    def __init__(self, mtc_keffs=(1.00000, 0.99900, 1.00100),
                 # aro, ari, stuck R101, stuck R201 — one row per branch label.
                 sdm_keffs=(1.00000, 0.90000, 0.93000, 0.95000), fail_on: str = ""):
        self.mtc_keffs = mtc_keffs
        self.sdm_keffs = sdm_keffs
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.restarts: list[str] = []

    def run_branch(self, name, deck_text, restart_path) -> S.BranchOutputs:
        self.calls.append(name)
        self.restarts.append(Path(restart_path).name)
        if self.fail_on and name.startswith(self.fail_on):
            raise RuntimeError("simulated MASTER branch failure")
        if name.startswith("mtc"):
            return S.BranchOutputs(_sum(self.mtc_keffs), "MTC   -20.00 PCM/C", 0.1)
        return S.BranchOutputs(_sum(self.sdm_keffs), "", 0.1)


def _rod_model() -> S.RodModel:
    return S.RodModel(
        groups=[S.RodGroup("R101", "R1", 1, 12), S.RodGroup("R201", "R2", 2, 12)],
        rod_map=[["R101", "o"], ["o", "R201"]],
    )


def _run_with_assets(tmp_path: Path, record_ids, *, write_restart=True) -> Path:
    """A run dir with a delivery ranking and a matching provenance index."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked = []
    for i, rid in enumerate(record_ids):
        cand_dir = run_dir / "cand" / rid
        cand_dir.mkdir(parents=True, exist_ok=True)
        deck = cand_dir / "MAS_INP"
        deck.write_text(DECK, encoding="ascii")
        restart = cand_dir / f"MAS_RST.CONV_{i:02d}_0655.00"
        if write_restart:
            restart.write_bytes(b"restart-bytes")
        S.record_target(run_dir, rid, deck_path=deck,
                        restart_path=restart if write_restart else None,
                        tag=f"cell_{rid}")
        ranked.append({"record_id": rid, "node_peak": 1.3 + 0.01 * i,
                       "f_r": 1.52, "compliance_margin": 0.03,
                       "peak_percentile": 0.2, "in_band": True})
    (run_dir / "delivery.json").write_text(
        json.dumps({"ranked": ranked, "excluded": []}), encoding="utf-8")
    return run_dir


# --------------------------------------------------------------------------- #
# 1) [constraints] — enable / limit are independent, default is report-only
# --------------------------------------------------------------------------- #
def test_constraints_defaults_are_off_and_report_only():
    c = ConstraintsConfig()
    # Extra MASTER calls are never implicit.
    assert c.mtc_enable is False and c.sdm_enable is False
    # And even once enabled, no limit is asserted on the user's behalf.
    assert c.mtc_max_pcm_per_c is None and c.sdm_required_pcm is None
    assert c.mtc_gated() is False and c.sdm_gated() is False


def test_constraints_enable_and_limit_are_independent():
    enabled_unlimited = ConstraintsConfig(mtc_enable=True, sdm_enable=True)
    assert enabled_unlimited.any_enabled() is True
    assert enabled_unlimited.mtc_gated() is False   # runs, but report-only
    assert enabled_unlimited.sdm_gated() is False

    limited_disabled = ConstraintsConfig(mtc_max_pcm_per_c=9.0, sdm_required_pcm=10870.0)
    assert limited_disabled.any_enabled() is False
    assert limited_disabled.mtc_gated() is False    # a limit alone never runs a branch
    assert limited_disabled.sdm_gated() is False

    gated = ConstraintsConfig(mtc_enable=True, mtc_max_pcm_per_c=9.0)
    assert gated.mtc_gated() is True and gated.sdm_gated() is False


def test_constraints_validate_rejects_inverted_window_and_negative_sdm():
    with pytest.raises(ConfigError):
        ConstraintsConfig(mtc_min_pcm_per_c=9.0, mtc_max_pcm_per_c=-54.0).validate()
    with pytest.raises(ConfigError):
        ConstraintsConfig(sdm_required_pcm=-1.0).validate()
    with pytest.raises(ConfigError):
        ConstraintsConfig(post_verify_top_k=-1).validate()
    ConstraintsConfig().validate()          # defaults are valid


def test_deck_without_constraints_section_loads_with_report_only_defaults():
    from lpopt.config import load_config
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    assert cfg.constraints.mtc_enable is False
    assert cfg.constraints.sdm_enable is False
    assert cfg.constraints.mtc_gated() is False


def test_limits_from_constraints_keeps_dcd_numbers_but_ungated():
    limits = S.LicensingLimits.from_constraints(ConstraintsConfig(mtc_enable=True),
                                                SdmMtcConfig())
    # DCD constants remain visible for CONTEXT in the report ...
    assert limits.mtc_max_pcm_per_c == 9.0
    assert limits.sdm_required_pcm == 10870.0
    # ... but a suggestion is never silently promoted to a verdict.
    assert limits.mtc_gated is False and limits.sdm_gated is False
    assert "report_only" in limits.limits_source

    user = S.LicensingLimits.from_constraints(
        ConstraintsConfig(mtc_enable=True, mtc_max_pcm_per_c=5.0), SdmMtcConfig())
    assert user.mtc_max_pcm_per_c == 5.0 and user.mtc_gated is True
    assert "user[constraints]" in user.limits_source


# --------------------------------------------------------------------------- #
# 2) report-only semantics — measured, recorded, never a violator
# --------------------------------------------------------------------------- #
def test_mtc_report_only_records_value_without_verdict():
    limits = S.LicensingLimits(mtc_gated=False)
    out = S.BranchOutputs(_sum([1.0, 0.999, 1.001]), "", 0)
    res = S._evaluate_mtc(out, "hfp", limits, S.BranchParams())
    assert res.value_pcm_per_c == pytest.approx(-20.0, abs=0.2)
    assert res.pass_limit is None                    # measured, not judged
    assert "REPORT-ONLY" in res.warning


def test_report_only_mtc_outside_dcd_window_is_still_not_a_violator():
    # +12 pcm/°C would FAIL the DCD-standard window — but the user set no limit,
    # so the run must report the number and mark nothing.
    limits = S.LicensingLimits(mtc_gated=False)
    out = S.BranchOutputs(_sum([1.0, 1.0012, 0.9988]), "", 0)
    res = S._evaluate_mtc(out, "hfp", limits, S.BranchParams())
    assert res.value_pcm_per_c > 9.0
    assert res.pass_limit is None


def test_sdm_report_only_records_margin_without_verdict():
    limits = S.LicensingLimits(sdm_gated=False)
    params = S.BranchParams(scram_banks=("R1", "R2"),
                            stuck_candidate_banks=("R1", "R2"),
                            rod_model=_rod_model())
    out = S.BranchOutputs(_sum([1.0, 0.90, 0.93, 0.95]), "", 0)
    res = S._evaluate_sdm(out, "hzp", limits, params)
    assert res.margin_pcm == pytest.approx(res.available_pcm - res.required_pcm)
    assert res.pass_limit is None
    assert "REPORT-ONLY" in res.warning


def test_report_only_result_verdict_is_REPORT_not_PASS(tmp_path):
    cand = S.CandidateRef("rid", "tag", DECK, tmp_path / "MAS_RST.CONV_00_0655.00")
    results = S.run_post_verification(
        [cand], S.LicensingLimits(mtc_gated=False, sdm_gated=False),
        master_cfg=None, work_root=tmp_path / "w",
        executor=FakeExecutor(), sidecar_path=None,
    )
    r = results[0]
    assert r.report_only is True
    assert r.verdict == "REPORT"
    assert r.passed is False        # a report is NEVER a clearance
    assert r.violates is False      # and NEVER a violation


# --------------------------------------------------------------------------- #
# 3) MASTER-call accounting — counted, including failures
# --------------------------------------------------------------------------- #
def test_master_calls_counted_per_axis(tmp_path):
    cand = S.CandidateRef("rid", "tag", DECK, tmp_path / "MAS_RST.CONV_00_0655.00")
    sdm_params = S.BranchParams(scram_banks=("R1", "R2"),
                                stuck_candidate_banks=("R1", "R2"),
                                rod_model=_rod_model())
    mtc_only = S.run_post_verification(
        [cand], S.LicensingLimits(), None, tmp_path / "w1",
        executor=FakeExecutor(), sidecar_path=None)
    assert mtc_only[0].master_calls == 1

    both = S.run_post_verification(
        [cand], S.LicensingLimits(), None, tmp_path / "w2",
        sdm_params=sdm_params, executor=FakeExecutor(), sidecar_path=None)
    assert both[0].master_calls == 2

    # Disabling MTC must not merely skip the verdict — it must not LAUNCH.
    ex = FakeExecutor()
    sdm_only = S.run_post_verification(
        [cand], S.LicensingLimits(), None, tmp_path / "w3",
        sdm_params=sdm_params, executor=ex, sidecar_path=None, run_mtc=False)
    assert sdm_only[0].master_calls == 1
    assert not any(name.startswith("mtc") for name in ex.calls)


def test_failed_branch_still_charges_its_master_call(tmp_path):
    cand = S.CandidateRef("rid", "tag", DECK, tmp_path / "MAS_RST.CONV_00_0655.00")
    results = S.run_post_verification(
        [cand], S.LicensingLimits(), None, tmp_path / "w",
        executor=FakeExecutor(fail_on="mtc"), sidecar_path=None)
    r = results[0]
    assert r.status == "failed"
    # The process ran and consumed wall time; under-reporting it would make the
    # licensing budget look cheapest exactly when it is most expensive.
    assert r.master_calls == 1
    assert r.verdict == "ERR"


# --------------------------------------------------------------------------- #
# 4) target resolution — own restart or skip, never a borrowed one
# --------------------------------------------------------------------------- #
def test_candidates_resolve_to_their_own_restart(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a", "rid_b"])
    cands, skipped = S.candidates_from_delivery(run_dir, None, top_k=2)
    assert skipped == []
    assert [c.record_id for c in cands] == ["rid_a", "rid_b"]
    # each candidate carries the restart from ITS OWN folder
    assert cands[0].restart_path.parent.name == "rid_a"
    assert cands[1].restart_path.parent.name == "rid_b"
    assert cands[0].restart_path != cands[1].restart_path


def test_missing_restart_is_skipped_not_substituted(tmp_path):
    """The regression that matters: candidate B's restart must never stand in.

    ``rid_a`` has no restart; ``rid_b`` does.  The old fallback took the first
    ``MAS_RST.*`` found anywhere under the run dir, which would have handed
    ``rid_a``'s deck ``rid_b``'s equilibrium state — a branch that completes
    normally and reports a licensing number for a core that was never evaluated.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for rid, with_restart in (("rid_a", False), ("rid_b", True)):
        cand_dir = run_dir / "cand" / rid
        cand_dir.mkdir(parents=True)
        deck = cand_dir / "MAS_INP"
        deck.write_text(DECK, encoding="ascii")
        restart = cand_dir / f"MAS_RST.CONV_{rid}.00"
        if with_restart:
            restart.write_bytes(b"x")
        S.record_target(run_dir, rid, deck_path=deck,
                        restart_path=restart if with_restart else None)
    (run_dir / "delivery.json").write_text(
        json.dumps({"ranked": [{"record_id": "rid_a"}, {"record_id": "rid_b"}]}),
        encoding="utf-8")

    cands, skipped = S.candidates_from_delivery(run_dir, None, top_k=2)
    assert [c.record_id for c in cands] == ["rid_b"]
    assert [s["record_id"] for s in skipped] == ["rid_a"]
    assert "restart" in skipped[0]["reason"]
    # and nothing points rid_a at rid_b's restart
    assert all(c.record_id == "rid_b" for c in cands)


def test_candidate_absent_from_index_is_skipped_with_reason(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "delivery.json").write_text(
        json.dumps({"ranked": [{"record_id": "ghost"}]}), encoding="utf-8")
    cands, skipped = S.candidates_from_delivery(run_dir, None, top_k=3)
    assert cands == []
    assert skipped[0]["record_id"] == "ghost"
    assert S.TARGETS_NAME in skipped[0]["reason"]


def test_purged_assets_are_skipped(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    for path in (run_dir / "cand" / "rid_a").glob("MAS_RST.*"):
        path.unlink()
    cands, skipped = S.candidates_from_delivery(run_dir, None, top_k=1)
    assert cands == []
    assert "no longer on disk" in skipped[0]["reason"]


def test_delivery_top_k_truncates_the_ranking(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["r0", "r1", "r2", "r3"])
    cands, _skipped = S.candidates_from_delivery(run_dir, None, top_k=2)
    assert [c.record_id for c in cands] == ["r0", "r1"]


def test_targets_index_last_write_wins(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    S.record_target(run_dir, "rid", deck_path=None, restart_path=None, note="first")
    S.record_target(run_dir, "rid", deck_path="d", restart_path="r", note="second")
    targets = S.load_targets(run_dir)
    assert len(targets) == 1
    assert targets["rid"]["note"] == "second"
    assert targets["rid"]["restart"] == "r"


# --------------------------------------------------------------------------- #
# 5) post_verify_delivery — the whole gate, stubbed
# --------------------------------------------------------------------------- #
def test_gate_disabled_by_default_runs_nothing(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    ex = FakeExecutor()
    summary = S.post_verify_delivery(run_dir, None, ConstraintsConfig(),
                                     executor=ex, sidecar_path=None)
    assert summary.results == [] and summary.master_calls == 0
    assert ex.calls == []                       # not one MASTER call implicitly


def test_gate_report_only_marks_no_violators(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a", "rid_b"])
    c = ConstraintsConfig(mtc_enable=True, post_verify_top_k=2)
    summary = S.post_verify_delivery(
        run_dir, None, c, sdm_mtc_cfg=SdmMtcConfig(),
        executor=FakeExecutor(), sidecar_path=None)
    assert summary.n_selected == 2 and len(summary.results) == 2
    assert summary.master_calls == 2            # 1 MTC branch per candidate
    assert summary.report_only is True
    assert summary.violators == []
    assert all(r.verdict == "REPORT" for r in summary.results)
    assert (run_dir / "sdm_mtc.json").is_file()


def test_gate_with_user_limit_marks_violators(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    # measured MTC is ~-20 pcm/°C; a user window of [-10, +9] makes it a violator.
    c = ConstraintsConfig(mtc_enable=True, mtc_min_pcm_per_c=-10.0,
                          mtc_max_pcm_per_c=9.0, post_verify_top_k=1)
    summary = S.post_verify_delivery(
        run_dir, None, c, sdm_mtc_cfg=SdmMtcConfig(),
        executor=FakeExecutor(), sidecar_path=None)
    assert summary.report_only is False
    assert summary.violators == ["rid_a"]
    assert summary.results[0].verdict == "FAIL"


def test_gate_with_user_limit_passes_a_compliant_candidate(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    c = ConstraintsConfig(mtc_enable=True, mtc_min_pcm_per_c=-54.0,
                          mtc_max_pcm_per_c=9.0, post_verify_top_k=1)
    summary = S.post_verify_delivery(
        run_dir, None, c, sdm_mtc_cfg=SdmMtcConfig(),
        executor=FakeExecutor(), sidecar_path=None)
    assert summary.violators == []
    # MTC gated + passing, SDM never run -> not a blanket PASS.
    assert summary.results[0].pass_mtc is True
    assert summary.results[0].pass_sdm is None


def test_gate_sdm_enabled_without_rod_model_is_inconclusive_not_pass(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    c = ConstraintsConfig(sdm_enable=True, sdm_required_pcm=10870.0,
                          post_verify_top_k=1)
    summary = S.post_verify_delivery(
        run_dir, None, c, sdm_mtc_cfg=SdmMtcConfig(), rod_model=None,
        executor=FakeExecutor(), sidecar_path=None)
    # No full-core rod model exists -> the absence of an SDM FAIL is not a pass.
    assert summary.violators == []
    assert all(r.pass_sdm is None for r in summary.results)
    assert any("rod model" in s["reason"] for s in summary.skipped)


def test_gate_sdm_with_rod_model_runs_both_axes(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    c = ConstraintsConfig(mtc_enable=True, sdm_enable=True,
                          sdm_required_pcm=1000.0, post_verify_top_k=1)
    ex = FakeExecutor()
    summary = S.post_verify_delivery(
        run_dir, None, c, sdm_mtc_cfg=SdmMtcConfig(
            scram_banks=["R1", "R2"], stuck_candidate_banks=["R1", "R2"]),
        rod_model=_rod_model(), executor=ex, sidecar_path=None)
    assert summary.master_calls == 2
    assert sorted(name.split("_")[0] for name in ex.calls) == ["mtc", "sdm"]
    assert summary.results[0].sdm is not None


def test_gate_writes_sidecar_and_verdict_table(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    sidecar = tmp_path / "sidecar.jsonl"
    c = ConstraintsConfig(mtc_enable=True, post_verify_top_k=1)
    summary = S.post_verify_delivery(
        run_dir, None, c, sdm_mtc_cfg=SdmMtcConfig(),
        executor=FakeExecutor(), sidecar_path=sidecar)
    rows = S.load_sidecar(sidecar)
    assert [r["record_id"] for r in rows] == ["rid_a"]
    assert rows[0]["master_calls"] == 1
    table = Path(summary.table_path)
    assert table.is_file()
    text = table.read_text(encoding="utf-8")
    assert "REPORT-ONLY" in text and "MASTER branch calls spent here" in text


def test_gate_never_raises_on_a_broken_run_dir(tmp_path):
    # A licensing-stage failure must not destroy a finished campaign's results.
    c = ConstraintsConfig(mtc_enable=True, post_verify_top_k=2)
    summary = S.post_verify_delivery(tmp_path / "does_not_exist", None, c,
                                     executor=FakeExecutor(), sidecar_path=None)
    assert summary.results == [] and summary.master_calls == 0


def test_gate_isolates_one_candidates_failure(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a", "rid_b"])
    c = ConstraintsConfig(mtc_enable=True, post_verify_top_k=2)

    class OneBadExecutor(FakeExecutor):
        def run_branch(self, name, deck_text, restart_path):
            if "rid_a" in str(deck_text) or len(self.calls) == 0:
                self.calls.append(name)
                raise RuntimeError("boom")
            return super().run_branch(name, deck_text, restart_path)

    summary = S.post_verify_delivery(run_dir, None, c, sdm_mtc_cfg=SdmMtcConfig(),
                                     executor=OneBadExecutor(), sidecar_path=None)
    assert len(summary.results) == 2
    assert summary.results[0].status == "failed"
    assert summary.results[1].status == "ok"
    assert summary.master_calls == 2       # both calls charged


# --------------------------------------------------------------------------- #
# 6) campaign wiring — delivery marking + two-budget accounting
# --------------------------------------------------------------------------- #
class _StubDriver:
    """The minimum of CampaignDriver the gate methods actually touch.

    The methods themselves are the REAL ones, bound off ``CampaignDriver`` — only
    the driver's heavyweight collaborators (store, fuel library, verifier) are
    replaced, so the wiring under test is production code, not a re-implementation.
    """

    from lpopt.search.campaign import CampaignDriver as _CD

    _mark_delivery_violators = _CD._mark_delivery_violators
    _maybe_post_verify = _CD._maybe_post_verify
    _rod_model = _CD._rod_model
    _record_sdm_mtc_target = _CD._record_sdm_mtc_target

    def __init__(self, run_dir: Path, constraints=None, executor=None):
        from lpopt.config import MasterConfig, SdmMtcConfig as _SM, VerifyConfig

        class _Cfg:
            pass

        self.run_dir = run_dir
        self.cfg = _Cfg()
        self.cfg.constraints = constraints or ConstraintsConfig()
        self.cfg.sdm_mtc = _SM(sidecar_path=str(run_dir / "sidecar.jsonl"))
        self.cfg.master = MasterConfig()
        self.cfg.verify = VerifyConfig()
        self.dry_run = True
        self.objective = "flat_power"
        self.post_verify_executor = executor
        self.post_verify_calls = 0
        self.post_verify_violators = []
        self.post_verify_summary = None
        # D9 licensing accounting is PERSISTED (the gate must never re-spend);
        # the stub records the save so the test can assert it happened.
        self.post_verify_done = False
        self.state_path = run_dir / "state.json"
        self.saves = 0
        self.logs: list[str] = []
        self.ctx = type("C", (), {"case_key": type("K", (), {"folder": "cell"})()})()

    def _log(self, msg):
        self.logs.append(msg)

    def _save_state(self):
        self.saves += 1

    def _resolve(self, p):
        return Path(p)


def test_driver_gate_runs_on_delivery_and_accounts_calls(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a", "rid_b"])
    delivery = json.loads((run_dir / "delivery.json").read_text(encoding="utf-8"))
    c = ConstraintsConfig(mtc_enable=True, mtc_min_pcm_per_c=-10.0,
                          mtc_max_pcm_per_c=9.0, post_verify_top_k=2)
    driver = _StubDriver(run_dir, c, executor=FakeExecutor())
    driver._maybe_post_verify(delivery)

    # both candidates verified, both violate the tight window, 1 call each
    assert driver.post_verify_calls == 2
    assert sorted(driver.post_verify_violators) == ["rid_a", "rid_b"]
    assert driver.post_verify_summary["master_calls"] == 2
    # and the delivery entries were marked in place
    written = json.loads((run_dir / "delivery.json").read_text(encoding="utf-8"))
    assert all(e["sdm_mtc_violation"] for e in written["ranked"])


def test_driver_gate_is_skipped_when_disabled(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a"])
    delivery = json.loads((run_dir / "delivery.json").read_text(encoding="utf-8"))
    ex = FakeExecutor()
    driver = _StubDriver(run_dir, ConstraintsConfig(), executor=ex)
    driver._maybe_post_verify(delivery)
    assert driver.post_verify_calls == 0 and ex.calls == []


def test_driver_gate_says_so_when_there_is_no_delivery_ranking(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    c = ConstraintsConfig(mtc_enable=True, post_verify_top_k=3)
    driver = _StubDriver(run_dir, c, executor=FakeExecutor())
    driver._maybe_post_verify(None)
    assert driver.post_verify_calls == 0
    assert any("no delivery ranking" in m for m in driver.logs)


def test_driver_records_target_with_note_when_provenance_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    driver = _StubDriver(run_dir)
    outcome = type("O", (), {"eq_provenance": None})()
    driver._record_sdm_mtc_target("rid_x", outcome)
    targets = S.load_targets(run_dir)
    assert targets["rid_x"]["restart"] is None
    assert "not licence-verifiable" in targets["rid_x"]["note"]

    outcome2 = type("O", (), {"eq_provenance": {"deck": "d", "restart": "r"}})()
    driver._record_sdm_mtc_target("rid_y", outcome2)
    assert S.load_targets(run_dir)["rid_y"]["restart"] == "r"


def test_delivery_entries_are_marked_not_deleted(tmp_path):
    run_dir = _run_with_assets(tmp_path, ["rid_a", "rid_b"])
    delivery = json.loads((run_dir / "delivery.json").read_text(encoding="utf-8"))
    c = ConstraintsConfig(mtc_enable=True, mtc_min_pcm_per_c=-10.0,
                          mtc_max_pcm_per_c=9.0, post_verify_top_k=1)
    summary = S.post_verify_delivery(run_dir, delivery, c, sdm_mtc_cfg=SdmMtcConfig(),
                                     executor=FakeExecutor(), sidecar_path=None)
    _StubDriver(run_dir)._mark_delivery_violators(delivery, summary)

    written = json.loads((run_dir / "delivery.json").read_text(encoding="utf-8"))
    # BOTH entries survive: the violator is marked, never silently removed.
    assert [e["record_id"] for e in written["ranked"]] == ["rid_a", "rid_b"]
    marked = {e["record_id"]: e for e in written["ranked"]}
    assert marked["rid_a"]["sdm_mtc_violation"] is True
    assert marked["rid_a"]["sdm_mtc_verdict"] == "FAIL"
    assert marked["rid_a"]["mtc_pcm_per_c"] == pytest.approx(-20.0, abs=0.2)
    # rid_b was outside top_k=1, so it is honestly "not verified", not "passed".
    assert marked["rid_b"]["sdm_mtc_verdict"] == "NOT_VERIFIED"
    assert marked["rid_b"]["sdm_mtc_violation"] is False
    assert written["sdm_mtc"]["master_calls"] == 1


def test_campaign_result_keeps_search_and_licensing_budgets_separate():
    from lpopt.search.campaign import CampaignResult

    r = CampaignResult(run_dir="x", status="complete", waves=3, budget=100,
                       budget_spent=96, n_feasible=4, on_target=2, best=None,
                       post_verify_master_calls=6)
    # The gate's calls are NOT drawn from the search budget ...
    assert r.budget_spent == 96
    assert r.post_verify_master_calls == 6
    # ... and the true MASTER cost is the sum, which the status file states.
    assert r.budget_spent + r.post_verify_master_calls == 102


def test_eq_provenance_requires_both_files_on_disk(tmp_path):
    """``verify._eq_provenance`` never half-resolves a candidate's assets."""
    from lpopt.search.verify import _eq_provenance

    work = tmp_path / "cy05"
    work.mkdir()
    restart = work / "MAS_RST.OUT_05_0655.00"

    class _Cycle:
        def __init__(self, wd, rst):
            self.work_dir = wd
            self.restart_path = rst

    class _Result:
        def __init__(self, cycles):
            self.cycles = cycles

    assert _eq_provenance(_Result([])) is None
    assert _eq_provenance(_Result([_Cycle(work, restart)])) is None   # nothing written
    (work / "MAS_INP").write_text(DECK, encoding="ascii")
    assert _eq_provenance(_Result([_Cycle(work, restart)])) is None   # restart missing
    restart.write_bytes(b"x")
    prov = _eq_provenance(_Result([_Cycle(work, restart)]))
    assert prov is not None
    assert Path(prov["deck"]).name == "MAS_INP"
    assert Path(prov["restart"]).name == restart.name
