"""Phase A-2 safety shield — OOD policy + conformal chance constraint.

The regression these tests exist for (external review §6.5 P0-04): the OOD guard
and the split-conformal intervals were computed, printed, and then IGNORED.  A
candidate whose fuel population sat off the training manifold could win an exploit
slot on a surrogate score the ensemble's own sigma cannot doubt, and a conformal
interval that crossed a licensing limit changed nothing.  Here the guard's verdict
reaches the RANKING (escalate) or the POOL (reject), the conformal upper bound
becomes a hard ``U_c(x) <= L_c`` screen, and both states reach the delivery
dossier.

Every default is asserted to be the SHIPPED behaviour: ``ood_policy = "warn"`` +
``conformal_gate = false`` must leave the wave exactly as it was.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from lpopt.config import AcquisitionConfig, ConfigError, load_config
from lpopt.search import acquisition as acq
from lpopt.search.construct import Candidate, CaseContext, candidate_record_id
from lpopt.search.genome import random_genome

from test_campaign_stub import FakeModel, _STORE, _cfg, _factory


# --------------------------------------------------------------------------- #
# fixtures: a synthetic pool + stub guards / interval sources
# --------------------------------------------------------------------------- #
def _ctx() -> CaseContext:
    return CaseContext(pair="K1_K2", feed=121, library_id="ga80", e_core=5.2)


def _pool(n: int, ctx: CaseContext, seed: int = 0) -> acq.ScoredPool:
    """``n`` distinct in-region candidates with strictly increasing exploit score."""
    rng = random.Random(seed)
    cands: list[Candidate] = []
    for _ in range(n):
        g = random_genome(rng, "K1_K2", 30)
        pat = g.to_pattern()
        cands.append(Candidate(pat, g, "random", None,
                               candidate_record_id(pat, ctx), 5.2))
    mean = np.tile([1.52, 1500.0, 2.30, 620.0, 0.20, np.nan, np.nan], (n, 1))
    std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (n, 1))
    return acq.ScoredPool(
        candidates=cands, mean=mean, epistemic=std.copy(), calibrated=std.copy(),
        conv=np.ones(n), p_feas=np.full(n, 0.80), acq=np.full(n, 0.50),
        raw_epi=np.arange(n, dtype=float), in_region=np.ones(n, dtype=bool),
        exploit=np.arange(n, dtype=float),
    )


#: The four-axis gated limit set a ``min_fr_max_cycle``-style mode carries.
LIMITS = {"f_r": 1.55, "cbc_max": 1550.0, "f_q": 2.41, "ao_abs": 0.30,
          "max_pin_burnup": 78.0, "cyclen_lo": None, "cyclen_hi": None,
          "f_xy": None}


class _Guard:
    """Stub OOD guard: flags exactly the patterns it was built with."""

    def __init__(self, bad_patterns=(), *, raises=False):
        self._bad = {p.canonical() for p in bad_patterns}
        self._raises = bool(raises)

    def feature_ood_types(self, pattern, *, margin=None):
        if self._raises:
            raise RuntimeError("envelope unavailable")
        if pattern.canonical() in self._bad:
            return {"T9": [("pin_pitch", 2.4)]}
        return {}


@dataclass
class _Interval:
    """Minimal ``IntervalPrediction`` stand-in (only what the gate reads)."""

    upper: np.ndarray
    available: bool = True


class _IntervalModel(_Guard):
    """Guard + a ``predict_interval`` returning a caller-supplied upper bound."""

    def __init__(self, upper, *, available=True, bad_patterns=()):
        super().__init__(bad_patterns)
        self._upper = np.asarray(upper, dtype=float)
        self._available = bool(available)

    def predict_interval(self, patterns, case, cell=0.0, *, alpha=0.10):
        self.last_alpha = float(alpha)
        return _Interval(self._upper[: len(list(patterns))], self._available)


# --------------------------------------------------------------------------- #
# 1. config: accept / reject
# --------------------------------------------------------------------------- #
def test_defaults_are_todays_behaviour() -> None:
    cfg = AcquisitionConfig()
    assert cfg.ood_policy == "warn"
    assert cfg.conformal_gate is False
    assert cfg.conformal_alpha == pytest.approx(0.10)
    # and the shield built from those defaults cannot change a wave at all.
    assert acq.SafetyShield.from_config(cfg).active is False


def test_config_accepts_every_ood_policy(tmp_path) -> None:
    for policy in ("warn", "escalate", "reject"):
        deck = tmp_path / f"{policy}.inp"
        deck.write_text(f'[acquisition]\nood_policy = "{policy}"\n', encoding="utf-8")
        assert load_config(deck).acquisition.ood_policy == policy


def test_config_accepts_the_conformal_gate(tmp_path) -> None:
    deck = tmp_path / "gate.inp"
    deck.write_text("[acquisition]\nconformal_gate = true\nconformal_alpha = 0.32\n",
                    encoding="utf-8")
    cfg = load_config(deck).acquisition
    assert cfg.conformal_gate is True
    assert cfg.conformal_alpha == pytest.approx(0.32)
    assert acq.SafetyShield.from_config(cfg).active is True


def test_config_rejects_an_unknown_ood_policy(tmp_path) -> None:
    deck = tmp_path / "bad_policy.inp"
    deck.write_text('[acquisition]\nood_policy = "block"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "ood_policy" in str(exc.value)


def test_config_rejects_an_unfitted_conformal_alpha(tmp_path) -> None:
    """0.05 is not a level any artifact is fit at -> the bound would be vacuous."""
    deck = tmp_path / "bad_alpha.inp"
    deck.write_text("[acquisition]\nconformal_alpha = 0.05\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "conformal_alpha" in str(exc.value)


def test_config_alpha_tracks_the_fitted_levels() -> None:
    """The deck knob's valid set IS the fitter's, not a second hard-coded copy."""
    from lpopt.config import _valid_conformal_alphas
    from lpopt.model.conformal import DEFAULT_ALPHAS

    assert _valid_conformal_alphas() == {float(a) for a in DEFAULT_ALPHAS}


def test_shield_policy_set_matches_the_config_validator() -> None:
    from lpopt.config import _VALID_OOD_POLICIES

    assert set(acq.OOD_POLICIES) == _VALID_OOD_POLICIES


# --------------------------------------------------------------------------- #
# 2a. OOD policy — warn (the shipped no-op)
# --------------------------------------------------------------------------- #
def test_warn_flags_but_changes_nothing() -> None:
    ctx = _ctx()
    pool = _pool(4, ctx)
    before = pool.exploit.copy()
    model = _Guard([pool.candidates[1].pattern, pool.candidates[3].pattern])
    out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(ood_policy="warn"), LIMITS)
    assert len(out.candidates) == 4
    assert np.array_equal(out.exploit, before)          # no score moved
    assert report["ood_flagged"] == 2                   # but the guard was READ
    assert report["ood_escalated"] == 0 and report["ood_rejected"] == 0
    assert list(out.ood_flag) == [False, True, False, True]


# --------------------------------------------------------------------------- #
# 2b. OOD policy — escalate
# --------------------------------------------------------------------------- #
def test_escalate_demotes_flagged_candidates_out_of_the_exploit_tier() -> None:
    ctx = _ctx()
    pool = _pool(4, ctx)
    flagged = [pool.candidates[3].pattern]              # the TOP exploit score
    model = _Guard(flagged)
    out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(ood_policy="escalate"), LIMITS)

    assert len(out.candidates) == 4                     # nothing dropped
    assert out.exploit[3] == -np.inf and out.rank[3] == -np.inf
    assert np.isfinite(out.exploit[:3]).all()           # the rest untouched
    assert report["ood_escalated"] == 1 and report["ood_rejected"] == 0
    assert report["n_remaining"] == 4
    # the demoted candidate keeps its explore currency: p_feas and raw_epi (the
    # ONLY key the explore slot ranks on) are deliberately not touched.
    assert out.raw_epi[3] == pytest.approx(3.0)
    assert out.p_feas[3] == pytest.approx(0.80)


def test_escalated_candidate_loses_the_exploit_slot_in_compose_wave() -> None:
    ctx = _ctx()
    pool = _pool(4, ctx)
    model = _Guard([pool.candidates[3].pattern])
    out, _ = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(ood_policy="escalate"), LIMITS)
    slots = acq.compose_wave(out, [], random.Random(0), size=2, n_exploit=2,
                             n_explore=0, n_control=0, tau=0.5, hamming_min=0)
    picked = {s.index for s in slots if s.slot == "exploit"}
    assert 3 not in picked                              # the OOD top pick is out
    assert picked == {2, 1}                             # next best two win


def test_escalated_candidate_cannot_seed_the_next_wave_elites() -> None:
    """``_run_wave``'s elite carry-over filters on ``isfinite(exploit)``."""
    ctx = _ctx()
    pool = _pool(4, ctx)
    out, _ = acq.apply_safety_shield(
        _Guard([pool.candidates[3].pattern]), ctx, pool,
        acq.SafetyShield(ood_policy="escalate"), LIMITS)
    order = np.argsort(-out.rank)[:2]
    kept = [int(i) for i in order if np.isfinite(out.exploit[int(i)])]
    assert 3 not in kept


# --------------------------------------------------------------------------- #
# 2c. OOD policy — reject
# --------------------------------------------------------------------------- #
def test_reject_drops_flagged_candidates_from_the_pool() -> None:
    ctx = _ctx()
    pool = _pool(5, ctx)
    bad = [pool.candidates[0].pattern, pool.candidates[4].pattern]
    survivors = [c.pattern.canonical() for i, c in enumerate(pool.candidates)
                 if i not in (0, 4)]
    out, report = acq.apply_safety_shield(
        _Guard(bad), ctx, pool, acq.SafetyShield(ood_policy="reject"), LIMITS)

    assert report["ood_rejected"] == 2
    assert report["n_candidates"] == 5 and report["n_remaining"] == 3
    assert [c.pattern.canonical() for c in out.candidates] == survivors
    # every array is subset consistently, not just the candidate list.
    assert out.mean.shape == (3, 7) and out.exploit.shape == (3,)
    assert list(out.raw_epi) == [1.0, 2.0, 3.0]
    assert not out.ood_flag.any()


def test_reject_can_empty_the_pool_without_raising() -> None:
    ctx = _ctx()
    pool = _pool(2, ctx)
    out, report = acq.apply_safety_shield(
        _Guard([c.pattern for c in pool.candidates]), ctx, pool,
        acq.SafetyShield(ood_policy="reject"), LIMITS)
    assert len(out.candidates) == 0 and report["n_remaining"] == 0
    # an empty pool composes an empty wave rather than crashing the campaign.
    assert acq.compose_wave(out, [], random.Random(0), size=4, n_exploit=4,
                            n_explore=0, n_control=0, tau=0.5, hamming_min=0) == []


# --------------------------------------------------------------------------- #
# 2d. guard availability — "absent" is not "clean", "errored" is not "pass"
# --------------------------------------------------------------------------- #
def test_a_backend_without_a_guard_flags_nothing_and_says_so() -> None:
    ctx = _ctx()
    pool = _pool(3, ctx)
    out, report = acq.apply_safety_shield(
        object(), ctx, pool, acq.SafetyShield(ood_policy="reject"), LIMITS)
    assert report["ood_guard"] == "absent"
    assert report["ood_flagged"] == 0 and len(out.candidates) == 3


def test_a_guard_that_raises_fails_closed() -> None:
    """Review §8.5: ``unknown`` is a state needing more compute, not a pass."""
    ctx = _ctx()
    pool = _pool(3, ctx)
    out, report = acq.apply_safety_shield(
        _Guard(raises=True), ctx, pool,
        acq.SafetyShield(ood_policy="reject"), LIMITS)
    assert report["ood_guard"] == "available"
    assert report["ood_guard_errors"] == 3
    assert report["ood_rejected"] == 3 and len(out.candidates) == 0


# --------------------------------------------------------------------------- #
# 3. conformal chance constraint
# --------------------------------------------------------------------------- #
def _upper(rows):
    """``[N, 7]`` upper bounds from ``(f_r, cbc, f_q, ao)`` tuples."""
    out = np.full((len(rows), 7), np.nan)
    for i, (fr, cbc, fq, ao) in enumerate(rows):
        out[i, 0], out[i, 1], out[i, 2], out[i, 4] = fr, cbc, fq, ao
    return out


def test_conformal_gate_drops_candidates_whose_upper_bound_crosses_a_limit() -> None:
    ctx = _ctx()
    pool = _pool(3, ctx)
    # row 1's F_r upper bound is over 1.55; row 2's CBC bound is over 1550.
    model = _IntervalModel(_upper([
        (1.54, 1500.0, 2.30, 0.20),
        (1.58, 1500.0, 2.30, 0.20),
        (1.54, 1600.0, 2.30, 0.20),
    ]))
    keep = pool.candidates[0].pattern.canonical()
    out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(conformal_gate=True), LIMITS)

    assert report["conformal_available"] is True
    assert report["conformal_rejected"] == 2
    assert report["conformal_rejected_by_axis"] == {"f_r": 1, "cbc_max": 1}
    assert [c.pattern.canonical() for c in out.candidates] == [keep]
    assert out.conformal_unfit == ((),)          # every gated axis was bounded


def test_conformal_gate_reads_the_configured_alpha() -> None:
    ctx = _ctx()
    pool = _pool(1, ctx)
    model = _IntervalModel(_upper([(1.50, 1500.0, 2.30, 0.20)]))
    acq.apply_safety_shield(
        model, ctx, pool,
        acq.SafetyShield(conformal_gate=True, conformal_alpha=0.32), LIMITS)
    assert model.last_alpha == pytest.approx(0.32)


def test_conformal_gate_only_screens_axes_the_mode_actually_gates() -> None:
    """``flat_power`` leaves F_r ungated; the gate must not invent a limit."""
    ctx = _ctx()
    pool = _pool(2, ctx)
    limits = dict(LIMITS, f_r=None)
    assert [k for k, _ in acq.conformal_gate_axes(limits)] == [
        "cbc_max", "f_q", "ao_abs"]
    model = _IntervalModel(_upper([
        (9.99, 1500.0, 2.30, 0.20),              # an F_r bound far over 1.55
        (1.50, 1500.0, 2.30, 0.20),
    ]))
    out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(conformal_gate=True), limits)
    assert report["conformal_rejected"] == 0 and len(out.candidates) == 2


def test_no_conformal_artifact_screens_nothing_and_reports_every_axis_unfit() -> None:
    """An absent interval leaves the mu+kappa*sigma screen standing — and SAYS so."""
    ctx = _ctx()
    pool = _pool(2, ctx)
    model = _IntervalModel(np.full((2, 7), np.nan), available=False)
    out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(conformal_gate=True), LIMITS)
    assert report["conformal_available"] is False
    assert report["conformal_rejected"] == 0 and len(out.candidates) == 2
    assert out.conformal_unfit[0] == ("f_r", "cbc_max", "f_q", "ao_abs")
    assert report["conformal_unfit_by_axis"] == {
        "f_r": 2, "cbc_max": 2, "f_q": 2, "ao_abs": 2}


def test_a_partially_fitted_axis_is_unfit_not_a_violation() -> None:
    ctx = _ctx()
    pool = _pool(1, ctx)
    rows = _upper([(1.50, 1500.0, 2.30, 0.20)])
    rows[0, 2] = np.nan                          # F_q carries no fitted interval
    out, report = acq.apply_safety_shield(
        _IntervalModel(rows), ctx, pool,
        acq.SafetyShield(conformal_gate=True), LIMITS)
    assert report["conformal_rejected"] == 0
    assert out.conformal_unfit[0] == ("f_q",)


def test_ood_reject_runs_before_the_conformal_gate() -> None:
    """An off-manifold board's interval is calibrated on the manifold it left."""
    ctx = _ctx()
    pool = _pool(2, ctx)
    model = _IntervalModel(_upper([
        (1.50, 1500.0, 2.30, 0.20),
        (1.50, 1500.0, 2.30, 0.20),
    ]), bad_patterns=[pool.candidates[0].pattern])
    out, report = acq.apply_safety_shield(
        model, ctx, pool,
        acq.SafetyShield(ood_policy="reject", conformal_gate=True), LIMITS)
    assert report["ood_rejected"] == 1
    assert len(out.candidates) == 1
    # the conformal pass saw ONE candidate, not two.
    assert len(out.conformal_unfit) == 1


# --------------------------------------------------------------------------- #
# 3b. F_xy: head + fitted interval, else keep the proxy sigma
# --------------------------------------------------------------------------- #
class _FxyModel(_IntervalModel):
    """A backend with an f_xy head and (optionally) a fitted f_xy conformal entry."""

    def __init__(self, upper, fxy_mean, *, with_fit=True, halfwidth=0.02):
        super().__init__(upper)
        self._fxy = np.asarray(fxy_mean, dtype=float)
        self.conformal = ({"per_target": {"f_xy": {
            "score_type": "abs",
            "cells": {},
            "global": {"0.1": float(halfwidth), "0.32": float(halfwidth) / 2.0},
        }}} if with_fit else {"per_target": {}})

    def predict_fxy(self, patterns, case, cell=0.0):
        patterns = list(patterns)
        if not patterns:                          # ``has_fxy_head``'s free probe
            return np.zeros(0), np.zeros(0), "head"
        n = len(patterns)
        return self._fxy[:n], np.full(n, 0.01), "head"

    def conformal_cell_keys(self, patterns, case, cell=0.0):
        return ["f121_e5.25"] * len(list(patterns))


def test_fxy_joins_the_gate_with_a_head_and_a_fitted_interval() -> None:
    ctx = _ctx()
    pool = _pool(2, ctx)
    limits = dict(LIMITS, f_xy=1.65)
    # halfwidth 0.02: row 0 -> 1.66 (over), row 1 -> 1.62 (under).
    model = _FxyModel(_upper([(1.50, 1500.0, 2.30, 0.20)] * 2),
                      [1.64, 1.60], halfwidth=0.02)
    keep = pool.candidates[1].pattern.canonical()
    out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(conformal_gate=True), limits)
    assert report["conformal_fxy"] == "head"
    assert report["conformal_rejected_by_axis"] == {"f_xy": 1}
    assert [c.pattern.canonical() for c in out.candidates] == [keep]


def test_fxy_without_a_conformal_fit_keeps_the_proxy_screen() -> None:
    ctx = _ctx()
    pool = _pool(2, ctx)
    limits = dict(LIMITS, f_xy=1.65)
    model = _FxyModel(_upper([(1.50, 1500.0, 2.30, 0.20)] * 2),
                      [1.90, 1.90], with_fit=False)
    out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(conformal_gate=True), limits)
    assert report["conformal_fxy"] == "proxy"
    assert report["conformal_rejected"] == 0 and len(out.candidates) == 2
    # not "unfit" either: the axis is still screened, just by mu + kappa*sigma.
    assert "f_xy" not in out.conformal_unfit[0]


def test_fxy_without_a_head_keeps_the_proxy_screen() -> None:
    ctx = _ctx()
    pool = _pool(1, ctx)
    limits = dict(LIMITS, f_xy=1.65)
    # a plain interval model: no ``predict_fxy`` at all -> proxy, per design.
    model = _IntervalModel(_upper([(1.50, 1500.0, 2.30, 0.20)]))
    assert acq.has_fxy_head(model, ctx) is False
    _out, report = acq.apply_safety_shield(
        model, ctx, pool, acq.SafetyShield(conformal_gate=True), limits)
    assert report["conformal_fxy"] == "proxy"


# --------------------------------------------------------------------------- #
# 4. delivery dossier — the flags reach the hand-off, the predicate does not move
# --------------------------------------------------------------------------- #
class _GuardedFakeModel(FakeModel):
    """The stub campaign model, plus an OOD guard that flags every board."""

    def feature_ood_types(self, pattern, *, margin=None):
        return {"T9": [("pin_pitch", 2.4)]}


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_stub_campaign_wave_carries_the_dossier_flags(tmp_path) -> None:
    cfg = _cfg(tmp_path, budget=8)
    run_dir = tmp_path / "run"
    from lpopt.search.campaign import run_campaign

    result = run_campaign(
        cfg, _GuardedFakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        backend_factory=lambda ckpt: _GuardedFakeModel(), early_stop=False,
        progress=False,
    )
    assert result.best is not None
    # the guard flagged every board, so the campaign's best must SAY it is OOD ...
    assert result.best["ood_flag"] is True
    # ... and, with no conformal.json on a FakeModel, name every gated axis as
    # uncalibrated rather than quietly presenting the row as certified.
    assert result.best["conformal_unfit_axes"] == [
        "f_r", "cbc_max", "f_q", "ao_abs"]
    sel = json.loads(
        (run_dir / "waves" / "wave_00" / "selection.json").read_text("utf-8"))
    assert all(s["ood_flag"] is True for s in sel["selection"])
    # warn is the default, so the shield never ran and writes no accounting block.
    assert "shield" not in sel


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_stub_campaign_reject_policy_records_the_wave_accounting(tmp_path) -> None:
    cfg = _cfg(tmp_path, budget=8)
    cfg.acquisition.ood_policy = "escalate"
    run_dir = tmp_path / "run_esc"
    from lpopt.search.campaign import run_campaign

    result = run_campaign(
        cfg, _GuardedFakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        backend_factory=lambda ckpt: _GuardedFakeModel(), early_stop=False,
        progress=False,
    )
    assert result.wave_reports, "a wave must have run"
    wave0 = result.wave_reports[0]
    assert wave0["ood_flagged"] > 0
    assert wave0["ood_escalated"] == wave0["ood_flagged"]
    assert wave0["ood_rejected"] == 0 and wave0["conformal_rejected"] == 0
    sel = json.loads(
        (run_dir / "waves" / "wave_00" / "selection.json").read_text("utf-8"))
    assert sel["shield"]["ood_policy"] == "escalate"
    assert sel["shield"]["ood_guard"] == "available"


@pytest.mark.skipif(not (_STORE / "records.parquet").exists(), reason="no store present")
def test_unguarded_backend_reports_unknown_not_clean(tmp_path) -> None:
    """A model with no guard must yield ``ood_flag = None``, never ``False``."""
    cfg = _cfg(tmp_path, budget=8)
    run_dir = tmp_path / "run_noguard"
    from lpopt.search.campaign import run_campaign

    result = run_campaign(
        cfg, FakeModel(), _factory(), dry_run=True, run_dir=run_dir,
        backend_factory=lambda ckpt: FakeModel(), early_stop=False, progress=False,
    )
    assert result.best is not None
    assert result.best["ood_flag"] is None


def test_delivery_dossier_entries_carry_the_flags(tmp_path) -> None:
    """``delivery.json`` (flat_power) stamps both fields onto every entry."""
    from test_flatness_campaign import _drv, _row

    drv = _drv(tmp_path / "deliver")
    peaks = np.linspace(1.30, 1.68, 12)
    for i, p in enumerate(peaks):
        drv.campaign_rows.append(_row(peak=float(p), cov=0.30,
                                      f_r=1.50 + 0.01 * i, record_id=f"d{i}",
                                      max_pin_burnup=70.0, f_xy=1.60))
    # what ``_run_wave`` stamps for a verified candidate.
    drv._row_safety["d3"] = {"ood_flag": True, "conformal_unfit_axes": ["f_q"]}

    payload = drv._write_delivery()
    entries = {c["record_id"]: c for c in payload["ranked"] + payload["excluded"]}
    assert entries["d3"]["ood_flag"] is True
    assert entries["d3"]["conformal_unfit_axes"] == ["f_q"]
    # a row this session never evaluated is UNKNOWN, not clean.
    assert entries["d0"]["ood_flag"] is None
    assert entries["d0"]["conformal_unfit_axes"] is None
    # the flag does not remove the candidate from the ranking — it annotates it.
    assert "d3" in {c["record_id"] for c in payload["ranked"]}
    written = json.loads((drv.run_dir / "delivery.json").read_text("utf-8"))
    assert {c["record_id"] for c in written["ranked"]} == {
        c["record_id"] for c in payload["ranked"]}


def test_selected_safety_reports_absent_guard_as_unknown(tmp_path) -> None:
    from test_flatness_campaign import _drv

    drv = _drv(tmp_path / "sel")
    rng = random.Random(3)
    pats = [random_genome(rng, "K1_K2", 30).to_pattern() for _ in range(2)]
    flags, unfit = drv._selected_safety(pats)
    assert flags == [None, None]                       # FakeModel has no guard
    assert unfit == [["f_r", "cbc_max", "f_q", "ao_abs"]] * 2
