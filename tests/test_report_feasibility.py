"""Report feasibility is OBJECTIVE-aware (flatness-first program 20260725 §10).

The STOP item this file pins: ``report.py`` defined campaign feasibility with
``_LIMITS["f_r"] = 1.55``, unconditionally.  A ``flat_power`` run retires F_r from
the objective and screens it at its own SAFETY gate (1.70, decision D1), so every
row the campaign itself had accepted between 1.55 and 1.70 was reported
**infeasible** — emptying the best-LP table, the budget curve and the GA overlay
of a run that had in fact succeeded.  1.55 stays, as the licensing margin column.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.report.report import (
    _LIMITS, _feasible, _report_objective, build_report,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _record(rid: str, *, f_r: float, node_peak: float | None = None,
            cyclen: float = 625.0, max_pin_burnup: float | None = 68.0,
            f_xy: float | None = 1.60) -> dict:
    """A converged verified row.

    ``max_pin_burnup`` / ``f_xy`` are MEASURED by default (2026-08-29): the
    delivered top-k table is now the DELIVERABLE set, which refuses any row whose
    gated licensing axes were never measured (review §6.4 three-state rule).  A
    test that means "this axis was not measured" passes ``None`` explicitly.
    """
    return {
        "record_id": rid, "converged": True, "valid": True,
        "f_r": f_r, "cbc_max": 1400.0, "f_q": 2.30, "ao_abs": 0.20,
        "cyclen": cyclen, "n_cycles": 11.0, "pattern": "F:K1:0|F:K2:0",
        "feed": 121, "node_peak": node_peak, "map_cov": 0.30,
        "max_pin_burnup": max_pin_burnup, "f_xy": f_xy,
    }


def _run_dir(tmp_path: Path, records: list[dict], objective: str) -> Path:
    run = tmp_path / "run"
    (run / "waves").mkdir(parents=True, exist_ok=True)
    with open(run / "labels.jsonl", "w", encoding="utf-8") as handle:
        for i, rec in enumerate(records):
            handle.write(json.dumps({
                "wave": 0, "slot": "exploit", "origin": "elite",
                "record_id": rec["record_id"], "status": "converged",
                "record": rec,
            }) + "\n")
    (run / "status.json").write_text(json.dumps({
        "status": "complete", "objective": objective, "budget": len(records),
        "budget_spent": len(records), "case": "K1_K2-121", "dry_run": True,
    }), encoding="utf-8")
    return run


def _report_text(tmp_path: Path, records: list[dict], objective: str,
                 **kw) -> str:
    run = _run_dir(tmp_path, records, objective)
    path = build_report(run, pair="K1_K2", **kw)
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# the feasibility predicate
# --------------------------------------------------------------------------- #
def test_feasible_still_gates_f_r_when_a_limit_is_given():
    fom = {"f_r": 1.62, "cbc_max": 1400.0, "f_q": 2.3, "ao_abs": 0.2}
    assert _feasible(fom, _LIMITS) is False
    assert _feasible(fom, {**_LIMITS, "f_r": 1.70}) is True


def test_feasible_ungates_f_r_when_the_limit_is_none():
    fom = {"f_r": 3.10, "cbc_max": 1400.0, "f_q": 2.3, "ao_abs": 0.2}
    assert _feasible(fom, {**_LIMITS, "f_r": None}) is True
    # the other three axes still gate.
    assert _feasible({**fom, "f_q": 9.9}, {**_LIMITS, "f_r": None}) is False


def test_feasible_ungated_tolerates_a_missing_f_r():
    fom = {"cbc_max": 1400.0, "f_q": 2.3, "ao_abs": 0.2}
    assert _feasible(fom, {**_LIMITS, "f_r": None}) is True
    assert _feasible(fom, _LIMITS) is False


# --------------------------------------------------------------------------- #
# the ranking scalar
# --------------------------------------------------------------------------- #
def test_report_objective_is_flatness_under_flat_power():
    fom = {"node_peak": 1.45, "cyclen": 600.0}
    assert _report_objective(fom, "flat_power", 625.0) == pytest.approx(-1.45)
    # flatter ranks HIGHER, and cyclen is irrelevant.
    flatter = {"node_peak": 1.30, "cyclen": 500.0}
    assert (_report_objective(flatter, "flat_power", 625.0)
            > _report_objective(fom, "flat_power", 625.0))
    # a row with no flatness label is unscorable, not "worst".
    assert _report_objective({"cyclen": 625.0}, "flat_power", 625.0) is None


def test_report_objective_default_is_unchanged():
    assert _report_objective({"cyclen": 620.0}, "target_cycle", 625.0
                             ) == pytest.approx(-5.0)
    assert _report_objective({"f_r": 1.48}, "fr_boundary", 625.0
                             ) == pytest.approx(-1.48)


# --------------------------------------------------------------------------- #
# end-to-end report
# --------------------------------------------------------------------------- #
def test_flat_power_report_keeps_the_rows_the_campaign_accepted(tmp_path):
    """THE bug: F_r 1.62 is campaign-feasible (gate 1.70) but reported infeasible."""
    records = [_record("rid_a", f_r=1.62, node_peak=1.44),
               _record("rid_b", f_r=1.58, node_peak=1.51)]
    text = _report_text(tmp_path / "flat", records, "flat_power")
    assert "verified feasible LPs: **2** / 2" in text
    assert "_No verified feasible LP" not in text
    assert "F_r **not gated**" in text
    # the licensing number is still on every row, as a margin.
    assert "lic -0.070" in text        # 1.55 - 1.62
    # and the flatness columns are in the table.
    assert "node_peak" in text


def test_the_same_rows_stay_infeasible_for_target_cycle(tmp_path):
    records = [_record("rid_a", f_r=1.62, node_peak=1.44),
               _record("rid_b", f_r=1.58, node_peak=1.51)]
    text = _report_text(tmp_path / "tc", records, "target_cycle")
    assert "verified feasible LPs: **0** / 2" in text
    assert "F_r ≤ 1.55" in text


def test_flat_power_report_ranks_by_flatness_not_by_cycle_distance(tmp_path):
    records = [
        _record("far_flat", f_r=1.60, node_peak=1.30, cyclen=560.0),
        _record("near_peaky", f_r=1.50, node_peak=1.68, cyclen=625.0),
    ]
    text = _report_text(tmp_path / "rank", records, "flat_power")
    body = text.split("## Best verified loading patterns")[1]
    assert body.index("far_flat"[:16]) < body.index("near_peaky"[:16])


def test_flat_power_report_drops_rows_with_no_flatness_label(tmp_path):
    records = [_record("labelled", f_r=1.60, node_peak=1.40),
               _record("unlabelled", f_r=1.50, node_peak=None)]
    text = _report_text(tmp_path / "nolabel", records, "flat_power")
    assert "verified feasible LPs: **1** / 2" in text


def test_explicit_limits_from_the_caller_win(tmp_path):
    """``write_campaign_report`` passes the mode's own 1.70 safety gate."""
    records = [_record("over", f_r=1.80, node_peak=1.40),
               _record("under", f_r=1.65, node_peak=1.45)]
    limits = {**_LIMITS, "f_r": 1.70}
    text = _report_text(tmp_path / "explicit", records, "flat_power", limits=limits)
    assert "verified feasible LPs: **1** / 2" in text
    assert "F_r ≤ 1.70" in text


def test_objective_falls_back_to_status_json(tmp_path):
    """``lpopt report <run_dir>`` has no deck; status.json carries the objective."""
    records = [_record("rid_a", f_r=1.62, node_peak=1.44)]
    text = _report_text(tmp_path / "status", records, "flat_power")
    assert "flat_power" in text
    assert "verified feasible LPs: **1** / 1" in text


def test_the_ga_600_overlay_is_not_mixed_into_a_flatness_curve(tmp_path):
    """Two different scalars must not share one axis / one table column."""
    records = [_record("rid_a", f_r=1.62, node_peak=1.44)]
    run = _run_dir(tmp_path / "ga", records, "flat_power")
    ga_log = tmp_path / "ga_generations_K1_K2.jsonl"
    ga_log.write_text(json.dumps({"batch": [
        {"fom": {"cyclen": 624.0}, "feasible": True, "eq_ok": True},
    ]}) + "\n", encoding="utf-8")
    text = build_report(run, pair="K1_K2", ga_log=ga_log).read_text(encoding="utf-8")
    assert "not comparable" in text
    assert "−node_peak" in text


# --------------------------------------------------------------------------- #
# the report's feasible set IS the campaign's (one predicate, not two)
# --------------------------------------------------------------------------- #
def test_the_report_applies_the_pin_burnup_gate_the_campaign_applies(tmp_path):
    """THE bug: ``report._feasible`` restated the rule and dropped a gate.

    ``campaign._is_feasible`` screens ``max_pin_burnup`` in ``flat_power`` /
    ``fr_boundary`` / ``min_fuel_cost``; the report judged CBC / F_q / |AO| / F_r
    only, so a row the campaign had REJECTED was listed as a verified feasible LP
    (and counted in the budget curve, and ranked as the run's best).
    """
    over = _record("pin_over", f_r=1.62, node_peak=1.44)
    over["max_pin_burnup"] = 85.0                 # the campaign rejects this row
    under = _record("pin_under", f_r=1.62, node_peak=1.45)
    under["max_pin_burnup"] = 78.0
    text = _report_text(tmp_path / "pin", [over, under], "flat_power")
    assert "verified feasible LPs: **1** / 2" in text
    assert "pin_under" in text and "pin_over" not in text
    # and the gate it applied is STATED, not silently in force.
    assert "max pin BU ≤ 80" in text


# --------------------------------------------------------------------------- #
# ``lpopt report`` judges the run at the F_r gate the RUN applied
# --------------------------------------------------------------------------- #
def _cfg_for_report(tmp_path: Path, objective: str):
    """The attributes ``regenerate_report`` reads off a deck (nothing more)."""
    from types import SimpleNamespace

    from lpopt.config import AcquisitionConfig

    deck = tmp_path / "lpopt.inp"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text("# fake deck\n", encoding="utf-8")
    return SimpleNamespace(
        case=SimpleNamespace(pair="K1_K2", feed=121),
        acquisition=AcquisitionConfig(objective=objective),
        model=SimpleNamespace(library_id="ga80"),
        extract=SimpleNamespace(ga_root="3_GA_Surrogate", ga_runs_flow="runs_flow"),
        source_path=deck,
    )


def test_regenerate_report_judges_flat_power_at_the_gate_the_run_applied(tmp_path):
    """THE bug: ``regenerate_report`` resolved the limits WITHOUT ``fr_gate``.

    ``feasibility_limits_for`` then returned flat_power's DECK gate (1.70) while
    the run had judged every row at its bias-corrected D1 gate (here 1.62, the
    value the campaign stamped into every row as ``f_r_limit_applied``).  So
    ``lpopt report`` re-admitted exactly the rows the safety gate had rejected —
    the defect the last round fixed in ``write_campaign_report`` (which asks the
    live driver) and left standing here.
    """
    from lpopt.report.report import regenerate_report

    records = [_record("under_gate", f_r=1.58, node_peak=1.44),
               _record("over_gate", f_r=1.66, node_peak=1.40)]   # 1.62 < 1.66 < 1.70
    run = _run_dir(tmp_path / "regen", records, "flat_power")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    status["best"] = {"record_id": "under_gate", "f_r_limit_applied": 1.62}
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")

    text = regenerate_report(run, _cfg_for_report(tmp_path / "deck", "flat_power")
                             ).read_text(encoding="utf-8")
    assert "F_r ≤ 1.62" in text
    assert "verified feasible LPs: **1** / 2" in text
    assert "under_gate" in text and "over_gate" not in text


def test_regenerate_report_holds_the_deck_gate_when_the_run_recorded_none(tmp_path):
    """No recorded gate (no best row) -> the deck value HOLDS, unchanged."""
    from lpopt.report.report import regenerate_report

    records = [_record("mid", f_r=1.66, node_peak=1.40)]
    run = _run_dir(tmp_path / "regen_hold", records, "flat_power")
    text = regenerate_report(run, _cfg_for_report(tmp_path / "deck2", "flat_power")
                             ).read_text(encoding="utf-8")
    assert "F_r ≤ 1.70" in text
    assert "verified feasible LPs: **1** / 1" in text


def test_recorded_gate_is_read_only_for_flat_power(tmp_path):
    """Other modes gate F_r from the deck (or not at all) — a stray
    ``f_r_limit_applied`` must not become their limit."""
    from lpopt.report.report import _recorded_fr_gate

    status = {"best": {"f_r_limit_applied": 1.62}}
    assert _recorded_fr_gate(status, "flat_power") == pytest.approx(1.62)
    assert _recorded_fr_gate(status, "target_cycle") is None
    assert _recorded_fr_gate({}, "flat_power") is None
    # a null / non-finite record is not a gate.
    assert _recorded_fr_gate({"best": {"f_r_limit_applied": None}}, "flat_power") is None
    assert _recorded_fr_gate({"best": {}, "best_overall": {"f_r_limit_applied": 1.55}},
                             "flat_power") == pytest.approx(1.55)


def test_report_and_campaign_agree_row_by_row(tmp_path):
    """Same rows, same limits, same verdict — because it is the same function."""
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import feasibility_limits_for, is_feasible

    acq = AcquisitionConfig(objective="flat_power")
    for objective in ("target_cycle", "flat_power", "fr_boundary", "min_fuel_cost"):
        limits = feasibility_limits_for(acq, objective)
        for pin_bu in (None, 78.0, 85.0):
            for f_r in (1.50, 1.62, 1.80):
                row = _record("r", f_r=f_r, node_peak=1.44)
                row["max_pin_burnup"] = pin_bu
                assert _feasible(row, limits) is is_feasible(row, limits)


def test_a_missing_pin_burnup_still_passes(tmp_path):
    """None-TOLERANT, exactly as the campaign is: MASTER adjudicates the field,
    and a strict None-reject would empty the table of every row lacking it."""
    rec = _record("no_pin", f_r=1.62, node_peak=1.44, max_pin_burnup=None)
    text = _report_text(tmp_path / "nopin", [rec], "flat_power")
    assert "verified feasible LPs: **1** / 1" in text
    # …and it is NOT deliverable: an unmeasured licensing axis is UNKNOWN, not
    # satisfied (review §6.4).  The two verdicts are reported side by side.
    assert "DELIVERABLE (every gated axis measured & within limit): **0**" in text
    assert "`max_pin_burnup` 1" in text


# --------------------------------------------------------------------------- #
# NaN == missing == None (decision 2026-07-31)
# --------------------------------------------------------------------------- #
#: The one fact, spelled three ways: absent key, in-memory ``None``, and the
#: ``NaN`` a parquet round-trip produces for the same absent float.
_MISSING_FORMS = ("absent", None, float("nan"))


def _with_pin(row: dict, form) -> dict:
    out = dict(row)
    if form == "absent":
        out.pop("max_pin_burnup", None)
    else:
        out["max_pin_burnup"] = form
    return out


@pytest.mark.parametrize("form", _MISSING_FORMS)
def test_missing_pin_burnup_passes_in_every_form(form):
    """THE fix: ``NaN`` pin BU is MISSING, not "infeasible".

    ``is_feasible``'s guard tested ``is not None``, so the identical row passed in
    memory and failed after a ``records.parquet`` round-trip (``nan <= 80.0`` is
    ``False``) — which collapsed the feasible-first elite tier and under-reported
    ``n_feasible`` under every objective that gates pin burnup.
    """
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import feasibility_limits_for, is_feasible

    for objective in ("flat_power", "fr_boundary", "min_fuel_cost"):
        limits = feasibility_limits_for(AcquisitionConfig(), objective)
        assert limits["max_pin_burnup"] is not None, objective
        row = _record("r", f_r=1.50, node_peak=1.44, cyclen=625.0)
        assert is_feasible(_with_pin(row, form), limits) is True, (objective, form)
        # the report agrees, because it is the same function.
        assert _feasible(_with_pin(row, form), limits) is True


def test_a_finite_violating_pin_burnup_is_still_infeasible():
    """No behaviour change for a MEASURED violation — only "missing" moved."""
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import feasibility_limits_for, is_feasible

    limits = feasibility_limits_for(AcquisitionConfig(), "flat_power")
    row = _record("r", f_r=1.50, node_peak=1.44)
    assert is_feasible({**row, "max_pin_burnup": 85.0}, limits) is False
    assert is_feasible({**row, "max_pin_burnup": 80.0}, limits) is True   # on the limit
    assert is_feasible({**row, "max_pin_burnup": 78.0}, limits) is True


@pytest.mark.parametrize("axis", ["cbc_max", "f_q", "ao_abs", "f_r"])
@pytest.mark.parametrize("form", [None, float("nan")])
def test_missing_primary_constraints_still_reject_in_both_forms(axis, form):
    """The REJECT axes keep rejecting — and reject ``None`` and ``NaN`` alike.

    Their behaviour is unchanged; the point is that the contract is now the same
    one everywhere: missing is missing, however it is spelled.
    """
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import feasibility_limits_for, is_feasible

    limits = feasibility_limits_for(AcquisitionConfig(), "target_cycle")
    row = _record("r", f_r=1.50, node_peak=1.44)
    row["max_pin_burnup"] = 40.0
    assert is_feasible(row, limits) is True
    assert is_feasible({**row, axis: form}, limits) is False


@pytest.mark.parametrize("form", [None, float("nan")])
def test_missing_cyclen_rejects_in_both_forms_when_the_band_is_gated(form):
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import feasibility_limits_for, is_feasible

    limits = feasibility_limits_for(AcquisitionConfig(), "min_fuel_cost")
    row = _record("r", f_r=1.50, cyclen=625.0)
    assert is_feasible(row, limits) is True
    assert is_feasible({**row, "cyclen": form}, limits) is False


def test_is_missing_is_the_one_contract():
    """``_is_missing`` is what makes NaN and None one fact (2026-07-31)."""
    from lpopt.search.campaign import _is_missing

    assert _is_missing(None) is True
    assert _is_missing(float("nan")) is True
    assert _is_missing(0.0) is False
    assert _is_missing(80.0) is False
    # +-inf is a MEASUREMENT (an absurd one) — the gate rejects it, it is not
    # "missing"; and a non-numeric value is invalid, not missing.
    assert _is_missing(float("inf")) is False
    assert _is_missing("n/a") is False


def test_the_wave_table_applies_the_pin_burnup_gate_too(tmp_path):
    """``_norm_fom`` dropped ``max_pin_burnup``, so the per-wave FEASIBLE count
    could not see the gate the campaign applies.

    ``FOM.as_dict()`` publishes the key under its own name; projecting the wave
    ``results.json`` FOM onto {f_r, cbc_max, f_q, cyclen, ao_abs} discarded it, so
    :func:`report._feasible` was handed ``None`` for every wave-sourced row and the
    pin-BU gate — None-tolerant by design — passed all of them.  The per-wave table
    then counted rows the campaign itself had rejected.
    """
    from lpopt.report.report import _norm_fom

    def _fom(pin_bu: float | None) -> dict:
        return {"F_r": 1.62, "CBC_max": 1400.0, "F_q": 2.30, "cyclen": 625.0,
                "AO_min": -0.20, "AO_max": 0.18, "max_pin_burnup": pin_bu}

    # the projection itself carries the column…
    assert _norm_fom(_fom(85.0))["max_pin_burnup"] == 85.0

    # …and the per-wave count therefore applies the gate.
    run = _run_dir(tmp_path / "wavepin",
                   [_record("rid_a", f_r=1.62, node_peak=1.44)], "flat_power")
    wave = run / "waves" / "wave_00"
    wave.mkdir(parents=True, exist_ok=True)
    (wave / "results.json").write_text(json.dumps({"wave": 0, "results": [
        {"slot": "exploit", "status": "converged", "fom": _fom(85.0)},   # rejected
        {"slot": "exploit", "status": "converged", "fom": _fom(78.0)},   # accepted
        {"slot": "explore", "status": "converged", "fom": _fom(None)},   # tolerated
    ]}), encoding="utf-8")
    text = build_report(run, pair="K1_K2").read_text(encoding="utf-8")
    wave_row = [ln for ln in text.splitlines() if ln.startswith("| 0 |")]
    assert wave_row, text
    # | wave | slots | converged | feasible | gate | tau |
    cells = wave_row[0].split("|")
    assert cells[2].strip() == "2/1/0"
    assert cells[3].strip() == "3"                # converged
    assert cells[4].strip() == "2"                # feasible — the 85.0 row is out


def test_a_deckless_min_fuel_cost_report_applies_the_cyclen_band(tmp_path):
    """The deck-less path learned the pin-BU gate but not the cyclen band.

    ``min_fuel_cost``'s feasible set gates BOTH edges of the cyclen band
    (``campaign.feasibility_limits_for``), so a deck-less ``lpopt report`` that
    only restored the pin-BU limit still listed out-of-band rows the campaign had
    rejected — the same hole, one axis over.
    """
    records = [_record("in_band", f_r=1.50, cyclen=625.0),
               _record("below_band", f_r=1.50, cyclen=600.0),
               _record("above_band", f_r=1.50, cyclen=650.0)]
    text = _report_text(tmp_path / "band", records, "min_fuel_cost")
    assert "verified feasible LPs: **1** / 3" in text
    assert "in_band" in text
    assert "below_band" not in text and "above_band" not in text
    # …and the band it applied is STATED, not silently in force.
    assert "cyclen ∈ [615, 635] EFPD" in text


def test_the_cyclen_band_is_min_fuel_cost_only(tmp_path):
    """No other mode gates cyclen — flat_power records it and never screens it."""
    records = [_record("far_off", f_r=1.62, node_peak=1.44, cyclen=520.0)]
    text = _report_text(tmp_path / "noband", records, "flat_power")
    assert "verified feasible LPs: **1** / 1" in text
    assert "cyclen ∈" not in text


def test_a_run_without_a_recorded_objective_is_unchanged(tmp_path):
    run = _run_dir(tmp_path / "legacy",
                   [_record("rid_a", f_r=1.62, node_peak=1.44)], "target_cycle")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    del status["objective"]
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    text = build_report(run, pair="K1_K2").read_text(encoding="utf-8")
    assert "F_r ≤ 1.55" in text
    assert "verified feasible LPs: **0** / 1" in text


def test_the_report_states_the_single_gate_limitation_when_the_run_used_two(tmp_path):
    """THE defect: ``_recorded_fr_gate`` reads ONE ``f_r_limit_applied`` off one
    row and the report applies it to EVERY row, though the D1 gate is rebuilt per
    driver — a resume under a re-fitted ``map_calibration.json``, or another cell
    of an outer cell race, writes rows judged at a DIFFERENT limit into the same
    run dir.  The per-row gate is not recoverable (``labels.jsonl`` carries the
    MASTER record, not the campaign row), so the limitation is STATED instead of
    silently judging the whole run at whichever value came first.
    """
    from lpopt.report.report import regenerate_report

    records = [_record("under_gate", f_r=1.58, node_peak=1.44),
               _record("over_gate", f_r=1.66, node_peak=1.40)]
    run = _run_dir(tmp_path / "twogates", records, "flat_power")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    status["best"] = {"record_id": "under_gate", "f_r_limit_applied": 1.62}
    status["best_overall"] = {"record_id": "over_gate", "f_r_limit_applied": 1.68}
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")

    text = regenerate_report(run, _cfg_for_report(tmp_path / "deck3", "flat_power")
                             ).read_text(encoding="utf-8")
    # the first recorded gate still decides the table (unchanged behaviour)…
    assert "F_r ≤ 1.62" in text
    # …and the report now SAYS that one number was applied to every row.
    assert "F_r gate limitation" in text
    assert "1.62" in text and "1.68" in text
    assert "every row below is judged at" in text


def test_a_single_recorded_gate_adds_no_limitation_note(tmp_path):
    """One gate throughout is the ordinary case: the footer is unchanged."""
    from lpopt.report.report import regenerate_report

    records = [_record("under_gate", f_r=1.58, node_peak=1.44)]
    run = _run_dir(tmp_path / "onegate", records, "flat_power")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    status["best"] = {"record_id": "under_gate", "f_r_limit_applied": 1.62}
    status["best_overall"] = {"record_id": "under_gate", "f_r_limit_applied": 1.62}
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    text = regenerate_report(run, _cfg_for_report(tmp_path / "deck4", "flat_power")
                             ).read_text(encoding="utf-8")
    assert "F_r ≤ 1.62" in text
    assert "F_r gate limitation" not in text


def test_recorded_fr_gates_collects_every_distinct_recorded_gate():
    """The helper the note is built from: distinct, in precedence order, and
    flat_power only."""
    from lpopt.report.report import _recorded_fr_gate, _recorded_fr_gates

    status = {"best": {"f_r_limit_applied": 1.62},
              "best_overall": {"f_r_limit_applied": 1.68}}
    assert _recorded_fr_gates(status, "flat_power") == [1.62, 1.68]
    assert _recorded_fr_gate(status, "flat_power") == pytest.approx(1.62)
    assert _recorded_fr_gates(status, "target_cycle") == []
    same = {"best": {"f_r_limit_applied": 1.62},
            "best_overall": {"f_r_limit_applied": 1.62}}
    assert _recorded_fr_gates(same, "flat_power") == [1.62]


# --------------------------------------------------------------------------- #
# search vs delivery: the THREE-STATE feasibility split (review 2026-08-29 6.4)
# --------------------------------------------------------------------------- #
def test_is_feasible_is_the_search_predicate_and_is_deliverable_is_not():
    """Two predicates, one row: search TOLERATES an unmeasured axis, delivery
    REFUSES it.  ``is_feasible`` stays an alias of the search one."""
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import (
        feasibility_limits_for, is_deliverable, is_feasible, is_feasible_search,
        unknown_axes,
    )

    assert is_feasible is is_feasible_search

    limits = feasibility_limits_for(AcquisitionConfig(), "min_fxy")
    assert limits["f_xy"] == pytest.approx(1.65)

    measured = _record("m", f_r=1.50, max_pin_burnup=70.0, f_xy=1.60)
    assert is_feasible_search(measured, limits) is True
    assert is_deliverable(measured, limits) is True
    assert unknown_axes(measured, limits) == ()

    for axis in ("f_xy", "max_pin_burnup"):
        row = dict(measured, **{axis: None})
        assert is_feasible_search(row, limits) is True, axis      # search PASSES
        assert is_deliverable(row, limits) is False, axis         # delivery REFUSES
        assert unknown_axes(row, limits) == (axis,), axis
        # NaN is the same fact as None (2026-07-31 contract).
        nan_row = dict(measured, **{axis: float("nan")})
        assert is_feasible_search(nan_row, limits) is True
        assert is_deliverable(nan_row, limits) is False

    # a MEASURED violation is not "unknown" — it is a plain FAIL on both.
    over = dict(measured, f_xy=1.70)
    assert is_feasible_search(over, limits) is False
    assert is_deliverable(over, limits) is False
    assert unknown_axes(over, limits) == ()


def test_deliverable_pin_burnup_uses_the_licensing_80_not_the_search_78():
    """The 2.0 GWd/tU haircut is MODEL margin on a PREDICTION; a MEASURED 79.4
    is deliverable, so re-using the search gate as a licensing verdict would
    reject deliverable cores."""
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import (
        DELIVERABLE_PIN_BU_LIMIT, deliverable_limits, feasibility_limits_for,
        is_deliverable, is_feasible_search,
    )

    assert DELIVERABLE_PIN_BU_LIMIT == pytest.approx(80.0)
    limits = feasibility_limits_for(AcquisitionConfig(), "min_fxy")
    assert limits["max_pin_burnup"] == pytest.approx(78.0)
    assert deliverable_limits(limits)["max_pin_burnup"] == pytest.approx(80.0)

    row = _record("p", f_r=1.50, max_pin_burnup=79.0, f_xy=1.60)
    assert is_feasible_search(row, limits) is False      # over the 78 model gate
    assert is_deliverable(row, limits) is True           # under the licensing 80


def test_unknown_axes_names_the_cyclen_column_once_for_a_two_edge_band():
    from lpopt.config import AcquisitionConfig
    from lpopt.search.campaign import feasibility_limits_for, unknown_axes

    limits = feasibility_limits_for(AcquisitionConfig(), "min_fuel_cost")
    row = _record("c", f_r=1.50, max_pin_burnup=70.0)
    row["cyclen"] = None
    assert unknown_axes(row, limits) == ("cyclen",)     # not ("cyclen", "cyclen")


def test_the_report_labels_the_search_set_and_the_deliverable_set_apart(tmp_path):
    """A row can be SEARCH-feasible and UNDELIVERABLE; calling the search set
    "feasible" in a delivery table is the claim the split exists to stop."""
    rows = [_record("measured", f_r=1.50, f_xy=1.60, max_pin_burnup=70.0),
            _record("no_pin", f_r=1.50, f_xy=1.58, max_pin_burnup=None)]
    text = _report_text(tmp_path / "split", rows, "min_fxy")
    assert "verified feasible LPs: **2** / 2" in text
    assert "DELIVERABLE (every gated axis measured & within limit): **1**" in text
    assert "UNKNOWN axes among the search-feasible rows: `max_pin_burnup` 1" in text
    body = text.split("## Best verified loading patterns")[1]
    # "no_pin" is the FLATTER row, so it would head a search-feasible table; the
    # delivered table drops it because its pin burnup was never measured.
    assert "measured" in body and "no_pin" not in body
    assert "F_xy (margin)" in body


def test_a_row_with_no_measured_fxy_is_unscorable_under_min_fxy(tmp_path):
    """Under ``min_fxy`` an unmeasured F_xy is not "worst", it is UNSCORABLE —
    the row cannot enter the ranking at all, so it never reaches either table."""
    rows = [_record("measured", f_r=1.50, f_xy=1.60, max_pin_burnup=70.0),
            _record("no_fxy", f_r=1.50, f_xy=None, max_pin_burnup=70.0)]
    text = _report_text(tmp_path / "unscorable", rows, "min_fxy")
    assert "verified feasible LPs: **1** / 2" in text
    assert "DELIVERABLE (every gated axis measured & within limit): **1**" in text
    assert "no_fxy" not in text.split("## Best verified loading patterns")[1]


def test_a_deckless_min_fxy_report_applies_the_1_65_gate(tmp_path):
    """Same class of hole as the pin-BU gate and the min_fuel_cost cyclen band:
    a deck-less report that forgot the gate would call rows feasible at the ONE
    axis this mode exists to optimise."""
    from lpopt.report.report import _DEFAULT_FXY_LIMIT

    assert _DEFAULT_FXY_LIMIT == pytest.approx(1.65)
    rows = [_record("under", f_r=1.50, f_xy=1.60, max_pin_burnup=70.0),
            _record("over", f_r=1.50, f_xy=1.70, max_pin_burnup=70.0)]
    text = _report_text(tmp_path / "deckless", rows, "min_fxy")
    assert "verified feasible LPs: **1** / 2" in text
    assert "F_xy \u2264 1.65" in text
    assert "under" in text


def test_the_min_fxy_report_objective_is_fxy_not_cycle_distance(tmp_path):
    """The lambda-objective check is mandatory on every frontier readout, and
    under this mode its axis is F_xy (design 3.5.5)."""
    from lpopt.report.report import _objective_axis_label, _report_objective

    lo = _report_objective({"f_xy": 1.55, "cyclen": 560.0}, "min_fxy", 625.0)
    hi = _report_objective({"f_xy": 1.60, "cyclen": 625.0}, "min_fxy", 625.0)
    assert lo > hi                                  # F_xy dominates cyclen
    # equal F_xy -> the longer cycle wins.
    a = _report_objective({"f_xy": 1.55, "cyclen": 630.0}, "min_fxy", 625.0)
    assert a > lo
    # no MEASURED f_xy -> unscorable, never "worst".
    assert _report_objective({"f_xy": None, "cyclen": 625.0}, "min_fxy", 625.0) is None
    assert "F_xy" in _objective_axis_label("min_fxy", 625.0)


def test_the_report_states_whether_fxy_came_from_a_head_or_the_proxy(tmp_path):
    """A proxy run may NOT be described as having optimised F_xy (design 3.6),
    so the distinction is in the report, not only in a log line."""
    rows = [_record("m", f_r=1.50, f_xy=1.60, max_pin_burnup=70.0)]
    run = _run_dir(tmp_path / "prov", rows, "min_fxy")
    wave = run / "waves" / "wave_00"
    wave.mkdir(parents=True, exist_ok=True)
    (wave / "selection.json").write_text(
        json.dumps({"wave": 0, "tau": 0.3, "selection": [], "fxy_source": "proxy"}),
        encoding="utf-8")
    text = build_report(run, pair="K1_K2").read_text(encoding="utf-8")
    assert "interim PROXY" in text
    assert "did NOT optimise a predicted F_xy" in text

    (wave / "selection.json").write_text(
        json.dumps({"wave": 0, "tau": 0.3, "selection": [], "fxy_source": "head"}),
        encoding="utf-8")
    text = build_report(run, pair="K1_K2").read_text(encoding="utf-8")
    assert "dedicated **head**" in text
