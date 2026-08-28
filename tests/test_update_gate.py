"""Wave online-update two-panel gate (plan sec. 4.6): challenger vs champion."""

from __future__ import annotations

import pytest

from lpopt.search.update import (
    Panel, accept_targets, evaluate_panel, gate, halt_primaries, panel_targets,
)


def _panel(skill: dict[str, float], n: int = 40) -> Panel:
    return Panel(skill=dict(skill), mae={k: 0.0 for k in skill}, n=n)


def test_gate_rejects_regressing_challenger():
    champion = _panel({"f_r": 0.50, "cyclen": 0.90, "cbc_max": 0.6, "f_q": 0.5, "ao_abs": 0.4})
    # challenger regresses F_r well past epsilon.
    challenger = _panel({"f_r": 0.20, "cyclen": 0.90, "cbc_max": 0.6, "f_q": 0.5, "ao_abs": 0.4})
    accepted, mode, reasons = gate(champion, challenger, champion, challenger, epsilon=0.02)
    assert accepted is False
    assert any("f_r" in r for r in reasons)
    # the retained model is the (skillful) champion -> mode objective.
    assert mode == "objective"


def test_gate_accepts_non_regressing_challenger():
    champion = _panel({"f_r": 0.50, "cyclen": 0.90, "cbc_max": 0.6, "f_q": 0.5, "ao_abs": 0.4})
    challenger = _panel({"f_r": 0.55, "cyclen": 0.92, "cbc_max": 0.62, "f_q": 0.5, "ao_abs": 0.4})
    accepted, mode, reasons = gate(champion, challenger, champion, challenger, epsilon=0.02)
    assert accepted is True
    assert mode == "objective"
    assert reasons == []


def test_gate_cumulative_regression_rejects():
    champion_h = _panel({"f_r": 0.50, "cyclen": 0.90})
    challenger_h = _panel({"f_r": 0.51, "cyclen": 0.91})  # holdout fine
    champion_c = _panel({"f_r": 0.60, "cyclen": 0.80})
    challenger_c = _panel({"f_r": 0.30, "cyclen": 0.50})  # cumulative much worse
    accepted, _, reasons = gate(champion_h, challenger_h, champion_c, challenger_c, epsilon=0.02)
    assert accepted is False
    assert any("cumulative" in r for r in reasons)


def test_gate_halt_on_no_skill_primaries():
    champion = _panel({"f_r": -0.05, "cyclen": -0.10, "cbc_max": 0.1, "f_q": 0.0, "ao_abs": 0.0})
    challenger = champion
    accepted, mode, _ = gate(champion, challenger, champion, challenger,
                             epsilon=0.02, skill_halt=0.0, skill_objective=0.10)
    assert mode == "halt"


def test_gate_explore_between_thresholds():
    champion = _panel({"f_r": 0.05, "cyclen": 0.90})   # f_r skillful-but-weak (< objective 0.10)
    accepted, mode, _ = gate(champion, champion, champion, champion,
                             epsilon=0.02, skill_halt=0.0, skill_objective=0.10)
    assert mode == "explore"


# --------------------------------------------------------------------------- #
# objective-aware panels (flatness-first program 20260725 §1.2 / §10)
#
# The regression: the panels were hard-wired to F_r/CBC/F_q/cyclen/|AO| skill and
# halted on F_r+cyclen, so a ``flat_power`` campaign accepted and halted its own
# model on the axes it had just retired — node_peak / map_cov were absent.
# --------------------------------------------------------------------------- #
def test_flat_power_panel_adds_the_flatness_targets():
    assert panel_targets("flat_power")[:2] == ("node_peak", "map_cov")
    # F_r is still REPORTED (program §10 KEEP), just no longer a veto.
    assert "f_r" in panel_targets("flat_power")
    assert panel_targets("target_cycle") == ("f_r", "cbc_max", "f_q", "cyclen", "ao_abs")


def test_flat_power_halts_on_flatness_not_on_f_r():
    assert halt_primaries("flat_power") == ("node_peak", "map_cov")
    for objective in ("target_cycle", "fr_boundary", "min_fuel_cost"):
        assert halt_primaries(objective) == ("f_r", "cyclen")


def test_flat_power_accept_set_drops_f_r_and_keeps_the_constraints():
    accept = accept_targets("flat_power")
    assert "f_r" not in accept                      # THE leak
    assert set(accept) == {"node_peak", "map_cov", "cbc_max", "f_q", "cyclen", "ao_abs"}
    assert accept_targets("target_cycle") == panel_targets("target_cycle")


def test_flat_power_f_r_regression_no_longer_vetoes_the_challenger():
    """THE bug: a challenger better at flatness was rejected for F_r skill."""
    champion = _panel({"node_peak": 0.40, "map_cov": 0.30, "f_r": 0.80,
                       "cbc_max": 0.6, "f_q": 0.5, "cyclen": 0.7, "ao_abs": 0.4})
    challenger = _panel({"node_peak": 0.75, "map_cov": 0.55, "f_r": 0.10,
                         "cbc_max": 0.6, "f_q": 0.5, "cyclen": 0.7, "ao_abs": 0.4})
    accepted, mode, reasons = gate(champion, challenger, champion, challenger,
                                   epsilon=0.02, objective="flat_power")
    assert accepted is True
    assert not any("f_r" in r for r in reasons)
    assert mode == "objective"
    # and the SAME panels under the default objective still reject it.
    rejected, _, why = gate(champion, challenger, champion, challenger, epsilon=0.02)
    assert rejected is False and any("f_r" in r for r in why)


def test_flat_power_cumulative_panel_does_not_vote_with_the_retired_f_r():
    """THE second leak: panel 1 dropped F_r, panel 2 kept averaging it.

    The campaign-cumulative panel is the OTHER acceptance panel, and averaging
    every REPORTED target handed the retired F_r skill a second, hidden vote: a
    challenger that gained on ``node_peak`` / ``map_cov`` and lost its F_r rank
    was rejected by the mean, for the axis the objective retired.
    """
    # panel 1 is deliberately inert (identical holdout panels), so ONLY the
    # cumulative panel can decide this gate.
    holdout = _panel({"node_peak": 0.40, "map_cov": 0.30, "f_r": 0.60,
                      "cbc_max": 0.5, "f_q": 0.5, "cyclen": 0.5, "ao_abs": 0.5})
    champ_cum = _panel({"node_peak": 0.40, "map_cov": 0.30, "f_r": 1.00,
                        "cbc_max": 0.5, "f_q": 0.5, "cyclen": 0.5, "ao_abs": 0.5})
    chal_cum = _panel({"node_peak": 0.50, "map_cov": 0.40, "f_r": 0.00,
                       "cbc_max": 0.5, "f_q": 0.5, "cyclen": 0.5, "ao_abs": 0.5})
    # the mean over EVERY reported target says "reject" (0.343 < 0.457 - ε) …
    assert chal_cum.mean_skill < champ_cum.mean_skill - 0.02
    # … while the mean over the targets ALLOWED to veto says the challenger won.
    veto = accept_targets("flat_power")
    assert chal_cum.mean_skill_over(veto) > champ_cum.mean_skill_over(veto)

    accepted, mode, reasons = gate(holdout, holdout, champ_cum, chal_cum,
                                   epsilon=0.02, objective="flat_power")
    assert accepted is True
    assert not any("cumulative" in r for r in reasons)
    assert mode == "objective"

    # the SAME panels under the default objective still reject on the mean (the
    # veto set IS the panel set there, so that behaviour is unchanged).
    rejected, _, why = gate(holdout, holdout, champ_cum, chal_cum, epsilon=0.02)
    assert rejected is False and any("cumulative" in r for r in why)


def test_flat_power_wave_artifact_records_the_cumulative_scalar_that_decided():
    """The recorded number must be the one panel 2 used, not a different mean."""
    from lpopt.search.update import GateResult, Panel

    champ_cum = _panel({"node_peak": 0.40, "map_cov": 0.30, "f_r": 1.00})
    chal_cum = _panel({"node_peak": 0.50, "map_cov": 0.40, "f_r": 0.00})
    result = GateResult(
        accepted=True, mode="objective", reasons=[],
        champion_holdout=Panel(), challenger_holdout=Panel(),
        champion_cumulative=champ_cum, challenger_cumulative=chal_cum,
        control_spearman=None, objective="flat_power")
    doc = result.as_dict()
    assert "f_r" not in doc["accept_targets"]
    assert doc["champion_cumulative_skill"] == pytest.approx(0.35)   # (0.40+0.30)/2
    assert doc["challenger_cumulative_skill"] == pytest.approx(0.45)  # (0.50+0.40)/2


def test_flat_power_still_vetoes_a_flatness_regression():
    champion = _panel({"node_peak": 0.70, "map_cov": 0.50, "f_r": 0.10})
    challenger = _panel({"node_peak": 0.20, "map_cov": 0.50, "f_r": 0.90})
    accepted, _, reasons = gate(champion, challenger, champion, challenger,
                                epsilon=0.02, objective="flat_power")
    assert accepted is False
    assert any("node_peak" in r for r in reasons)


def test_flat_power_halt_ignores_a_strong_f_r_skill():
    """No flatness skill == no steering, however well the model ranks F_r."""
    panel = _panel({"node_peak": -0.05, "map_cov": -0.10, "f_r": 0.95, "cyclen": 0.95})
    _, mode, _ = gate(panel, panel, panel, panel, epsilon=0.02,
                      skill_halt=0.0, skill_objective=0.10, objective="flat_power")
    assert mode == "halt"
    # the very same panel is "objective" for the F_r/cyclen default objective.
    _, default_mode, _ = gate(panel, panel, panel, panel, epsilon=0.02,
                              skill_halt=0.0, skill_objective=0.10)
    assert default_mode == "objective"


def test_no_map_head_cannot_halt_a_flat_power_campaign():
    """All-NaN flatness skill must fall to explore, never to a halt verdict."""
    panel = _panel({"f_r": -0.9, "cyclen": -0.9})     # no flatness keys at all
    _, mode, _ = gate(panel, panel, panel, panel, epsilon=0.02,
                      objective="flat_power")
    assert mode == "explore"


class _RankModel:
    """Predicts a monotone function of a hidden per-row target for a clean Spearman."""

    def __init__(self, rows, noise=0.0):
        self._map = {}
        for i, r in enumerate(rows):
            self._map[str(r["pattern"])] = i

    def predict(self, patterns, case, cell=0.0):
        import numpy as np
        from lpopt.vendor.masterrl.surrogate import SurrogatePrediction
        n = len(patterns)
        mean = np.zeros((n, 7))
        for j, p in enumerate(patterns):
            rank = self._map.get(p.canonical(), 0)
            mean[j] = [1.5 + 0.001 * rank, 1500 + rank, 2.3, 600 + rank, 0.2, np.nan, np.nan]
        std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (n, 1))
        return SurrogatePrediction(mean, std.copy(), std.copy())


def test_evaluate_panel_spearman_orders():
    import random
    from lpopt.search.genome import random_genome
    from lpopt.vendor.masterrl.domain import CaseKey
    rng = random.Random(4)
    rows = []
    for i in range(20):
        pat = random_genome(rng, "K1_K2", 30).to_pattern()
        rows.append({"pattern": pat.canonical(), "f_r": 1.5 + 0.001 * i,
                     "cbc_max": 1500 + i, "f_q": 2.3, "cyclen": 600 + i, "ao_abs": 0.2})
    model = _RankModel(rows)
    panel = evaluate_panel(model, rows, CaseKey("K1_K2", 121), 5.2)
    # perfectly monotone predictions -> Spearman ~1 on every target with variance.
    assert panel.skill["cyclen"] > 0.99
    assert panel.skill["f_r"] > 0.99


class _FlatRankModel(_RankModel):
    """A :class:`_RankModel` that also serves the map head (``node_peak``/CoV)."""

    def predict_map_flatness(self, patterns, case, cell=0.0):
        import numpy as np
        rank = np.array([self._map.get(p.canonical(), 0) for p in patterns],
                        dtype=float)
        # monotone in the hidden rank, like the surrogate head above.
        peak = 1.40 + 0.01 * rank
        cov = 0.10 + 0.001 * rank
        z = np.zeros_like(rank)
        return peak, z, cov, z.copy()


def _flat_rows(n: int = 20):
    import random
    from lpopt.search.genome import random_genome
    rng = random.Random(11)
    rows = []
    for i in range(n):
        pat = random_genome(rng, "K1_K2", 30).to_pattern()
        rows.append({"pattern": pat.canonical(), "f_r": 1.5 + 0.001 * i,
                     "cbc_max": 1500 + i, "f_q": 2.3, "cyclen": 600 + i,
                     "ao_abs": 0.2, "node_peak": 1.40 + 0.01 * i,
                     "map_cov": 0.10 + 0.001 * i})
    return rows


def test_evaluate_panel_scores_the_flatness_targets_from_the_map_head():
    from lpopt.vendor.masterrl.domain import CaseKey

    rows = _flat_rows()
    panel = evaluate_panel(_FlatRankModel(rows), rows, CaseKey("K1_K2", 121), 5.2,
                           objective="flat_power")
    assert panel.skill["node_peak"] > 0.99
    assert panel.skill["map_cov"] > 0.99
    assert panel.skill["f_r"] > 0.99          # still reported


def test_evaluate_panel_without_a_map_head_yields_nan_flatness_skill():
    import math
    from lpopt.vendor.masterrl.domain import CaseKey

    rows = _flat_rows()
    panel = evaluate_panel(_RankModel(rows), rows, CaseKey("K1_K2", 121), 5.2,
                           objective="flat_power")
    assert math.isnan(panel.skill["node_peak"])
    assert math.isnan(panel.skill["map_cov"])


class _RecordingModel:
    """Records the (new, replay) row counts a fine-tune call received."""

    def __init__(self):
        self.calls = []

    def predict(self, patterns, case, cell=0.0):
        import numpy as np
        from lpopt.vendor.masterrl.surrogate import SurrogatePrediction
        n = len(patterns)
        mean = np.tile([1.5, 1500.0, 2.3, 600.0, 0.2, np.nan, np.nan], (n, 1)).astype(float)
        std = np.tile([0.02, 15.0, 0.05, 3.0, 0.02, np.nan, np.nan], (n, 1))
        return SurrogatePrediction(mean, std.copy(), std.copy())

    def finetune(self, new, replay, epochs, seed):
        self.calls.append((len(new), len(replay)))
        return {"wall_seconds": 0.0, "n_new": len(new), "n_replay": len(replay), "epochs": epochs}


def test_finetune_oversamples_new_labels_for_boundary_emphasis():
    """Refinement 4: this-campaign wave labels are oversampled ``new_weight``× so
    the few fresh boundary labels move the discriminator; the reported n_new stays
    the honest unweighted count."""

    from lpopt.search.update import WaveUpdater
    from lpopt.vendor.masterrl.domain import CaseKey

    new = [{"pattern": f"p{i}", "f_r": 1.5, "cyclen": 600} for i in range(2)]
    replay = [{"pattern": f"r{i}", "f_r": 1.6, "cyclen": 620} for i in range(10)]

    model = _RecordingModel()
    updater = WaveUpdater(CaseKey("K1_K2", 121), 5.2, holdout_rows=[], new_weight=4)
    result = updater.update(model, new, replay, cumulative_rows=[], control_rows=[])

    seen_new, seen_replay = model.calls[0]
    assert seen_new == 4 * len(new)                # backend saw 4x the fresh labels
    assert seen_replay == len(replay)              # replay untouched
    assert result.finetune_stats["n_new"] == len(new)      # honest reported count
    assert result.finetune_stats["new_weight"] == 4


def test_finetune_new_weight_one_is_passthrough():
    from lpopt.search.update import WaveUpdater
    from lpopt.vendor.masterrl.domain import CaseKey

    new = [{"pattern": f"p{i}"} for i in range(3)]
    model = _RecordingModel()
    updater = WaveUpdater(CaseKey("K1_K2", 121), 5.2, holdout_rows=[], new_weight=1)
    updater.update(model, new, [], cumulative_rows=[], control_rows=[])
    assert model.calls[0][0] == len(new)           # no oversampling at weight 1


# --------------------------------------------------------------------------- #
# rollback honesty: a REJECTED challenger is only undone where it CAN be
#
# ``_restore`` rolls back member state.  A stateless-refit backend has none, so
# its fine-tune survives the rejection: the champion pointer says "nothing
# happened" while the served weights have moved — and under MODEL_HALT those are
# the weights the run keeps serving.  ``GateResult.weights_rolled_back`` is what
# lets the campaign tell the two apart (it reports the map-calibration drift).
# --------------------------------------------------------------------------- #
class _StatelessBackend:
    """A refit backend (e.g. sklearn): no ``members``, nothing to snapshot."""

    def __init__(self) -> None:
        self.refits = 0

    def finetune(self, new, replay, epochs=3, seed=0):
        self.refits += 1
        return {"refit": True}


class _FakeMember:
    def __init__(self) -> None:
        self.state = {"w": 0}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


class _EnsembleBackend(_StatelessBackend):
    """A CNN-style backend: member state exists, so a rejection is undoable."""

    def __init__(self) -> None:
        super().__init__()
        self.members = [_FakeMember(), _FakeMember()]

    def finetune(self, new, replay, epochs=3, seed=0):
        for m in self.members:
            m.state["w"] += 1
        return super().finetune(new, replay, epochs=epochs, seed=seed)


def _rejecting(monkeypatch):
    from lpopt.search import update as U

    monkeypatch.setattr(
        U, "gate", lambda *a, **kw: (False, "halt", ["forced rejection"]))


def test_a_rejected_stateless_refit_reports_that_it_was_not_rolled_back(monkeypatch):
    """THE gap: the rejection branch was read as 'the weights did not move'."""
    from lpopt.search.update import WaveUpdater
    from lpopt.vendor.masterrl.domain import CaseKey

    _rejecting(monkeypatch)
    model = _StatelessBackend()
    result = WaveUpdater(CaseKey("K1_K2", 121), 5.2, holdout_rows=[]).update(
        model, [], [], cumulative_rows=[], control_rows=[])
    assert result.accepted is False
    assert model.refits == 1                       # the refit HAPPENED…
    assert result.weights_rolled_back is False     # …and was never undone


def test_a_rejected_ensemble_is_rolled_back_and_says_so(monkeypatch):
    from lpopt.search.update import WaveUpdater
    from lpopt.vendor.masterrl.domain import CaseKey

    _rejecting(monkeypatch)
    model = _EnsembleBackend()
    result = WaveUpdater(CaseKey("K1_K2", 121), 5.2, holdout_rows=[]).update(
        model, [], [], cumulative_rows=[], control_rows=[])
    assert result.accepted is False
    assert result.weights_rolled_back is True
    assert [m.state["w"] for m in model.members] == [0, 0]   # champion restored


def test_an_accepted_gate_is_not_a_rollback(monkeypatch):
    """``weights_rolled_back`` is about the REJECT path; acceptance keeps the
    challenger, which is a weight change by design."""
    from lpopt.search.update import WaveUpdater
    from lpopt.vendor.masterrl.domain import CaseKey

    from lpopt.search import update as U
    monkeypatch.setattr(U, "gate", lambda *a, **kw: (True, "objective", []))
    model = _EnsembleBackend()
    result = WaveUpdater(CaseKey("K1_K2", 121), 5.2, holdout_rows=[]).update(
        model, [], [], cumulative_rows=[], control_rows=[])
    assert result.accepted is True
    assert result.weights_rolled_back is False
    assert [m.state["w"] for m in model.members] == [1, 1]   # challenger kept
