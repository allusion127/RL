"""Online wave update: fine-tune + challenger/champion two-panel gate (plan 4.6).

Each wave the campaign fine-tunes the backend on the 8 fresh labels plus a
replay sample (local CPU, plan sec. 4.7), producing a **challenger**.  The
challenger replaces the **champion** only when both panels agree:

1. **frozen (feed, e_core)-stratified holdout skill** — the primary quality
   statistic is the *within-case Spearman* (plan sec. 4.4: rank correlation is
   the主지표, robust to the ultra-tight feasible clusters that make an absolute
   MAE/baseline ratio pathological).  The challenger must not regress on *any*
   target beyond ``ε`` (default 0.02).
2. **campaign-cumulative Spearman** — the same statistic on every verified label
   so far, averaged over the SAME veto set as panel 1; the challenger must be
   not-worse (within ``ε``).

A pooled control-slot Spearman audit (predicted vs verified) is computed for the
record but is informational (one control label per wave has no power — the audit
is meaningful only pooled across the campaign, plan sec. 4.6).

The gate ``mode`` (``objective`` / ``explore`` / ``halt``) is read from the
retained model's holdout skill on the PRIMARY targets; two consecutive ``halt``
verdicts trigger the campaign's ``MODEL_HALT`` fallback path.

Which targets are primary is a property of the CAMPAIGN OBJECTIVE, not a
constant (flatness-first program 20260725 §1.2/§10).  A ``flat_power`` campaign
optimizes ``node_peak`` (primary) + ``map_cov`` (secondary) and has retired F_r
to a safety gate, so accepting or halting its own model on F_r + cyclen skill
judged the surrogate on axes the search does not use — a model that got sharply
better at flatness and slightly worse at F_r was rejected, and a model with no
flatness skill at all could not halt.  :func:`panel_targets` /
:func:`accept_targets` / :func:`halt_primaries` make the two panels
objective-aware; every other objective keeps the historical F_r + cyclen
behaviour byte-for-byte.

BOTH acceptance panels read the veto set.  Dropping ``f_r`` from panel 1 alone
left it deciding panel 2 through the MEAN of every reported target: a flat_power
challenger that gained on ``node_peak`` and lost its F_r rank was still rejected,
by the cumulative average, for the axis the objective retired.  Panel 2 therefore
averages :func:`accept_targets` only — F_r stays a reported secondary in
``Panel.skill`` (program §10 KEEP) with no vote on either panel.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from ..data.schema import unpack_pattern
from ..vendor.masterrl.domain import CaseKey, Pattern

#: dataset target name -> surrogate prediction column (plan sec. 4.5 layout).
_TARGET_TO_SURR: dict[str, int] = {
    "f_r": 0, "cbc_max": 1, "f_q": 2, "cyclen": 3, "ao_abs": 4,
}
#: The 7-column FOM targets (the surrogate head).  Panel set for every objective
#: except ``flat_power``; ALSO the historical value of :data:`_PANEL_TARGETS`.
_FOM_TARGETS = ("f_r", "cbc_max", "f_q", "cyclen", "ao_abs")
#: The flatness targets (program §1.1).  They are served by the MAP HEAD, not by
#: the 7-column surrogate, so :func:`evaluate_panel` scores them from
#: ``predict_map_flatness`` — see :func:`_flatness_predictions`.
_FLAT_TARGETS = ("node_peak", "map_cov")
#: Objective whose primaries are the flatness columns (program §1.2).
_FLAT_OBJECTIVE = "flat_power"
#: Default (non-flat) panel / halt sets — kept as module constants because they
#: are the historical contract and several callers import them.
_PANEL_TARGETS = _FOM_TARGETS
_HALT_PRIMARIES = ("f_r", "cyclen")


def panel_targets(objective: str = "target_cycle") -> tuple[str, ...]:
    """Targets the panel EVALUATES and REPORTS for ``objective``.

    ``flat_power`` reports the flatness primaries first and keeps every FOM
    target (F_r included) as a reported secondary — retiring F_r from the
    objective never meant hiding it from the record (program §10 KEEP).
    """
    if str(objective) == _FLAT_OBJECTIVE:
        return _FLAT_TARGETS + _FOM_TARGETS
    return _FOM_TARGETS


def accept_targets(objective: str = "target_cycle") -> tuple[str, ...]:
    """Targets whose holdout regression can VETO the challenger (panel 1).

    Under ``flat_power`` F_r is a safety GATE, not an objective, so its skill is
    reported but may not reject a model: leaving it in the veto set let F_r steer
    the campaign's own model acceptance, which is exactly the leak program §10
    STOPs.  ``cbc_max`` / ``f_q`` / ``ao_abs`` / ``cyclen`` stay in the veto set —
    they remain hard constraints (and cyclen a recorded axis) in that mode.
    """
    if str(objective) == _FLAT_OBJECTIVE:
        return _FLAT_TARGETS + tuple(t for t in _FOM_TARGETS if t != "f_r")
    return _FOM_TARGETS


def halt_primaries(objective: str = "target_cycle") -> tuple[str, ...]:
    """Targets whose no-skill verdict HALTS the campaign (mode selection).

    ``flat_power`` halts on the objective it actually optimizes — a surrogate
    with no ``node_peak`` / ``map_cov`` skill cannot steer a flatness search no
    matter how well it ranks F_r.
    """
    if str(objective) == _FLAT_OBJECTIVE:
        return _FLAT_TARGETS
    return _HALT_PRIMARIES


# --------------------------------------------------------------------------- #
# panels + gate result
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    #: primary per-target quality statistic (within-case Spearman, plan sec. 4.4).
    skill: dict[str, float] = field(default_factory=dict)
    #: secondary per-target MAE (recorded for the report; not the gate statistic).
    mae: dict[str, float] = field(default_factory=dict)
    n: int = 0

    def mean_skill_over(self, names: Sequence[str]) -> float:
        """Mean skill over ``names`` only (``nan`` when none of them is finite).

        The gate's panel-2 statistic: it must average the targets that are
        ALLOWED to veto, never every reported target, or a retired axis votes
        through the mean (see the module header).
        """
        vals = [self.skill[n] for n in names
                if n in self.skill and math.isfinite(self.skill[n])]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_skill(self) -> float:
        """Mean over EVERY scored target — a reported summary, not a verdict."""
        return self.mean_skill_over(tuple(self.skill))


@dataclass
class GateResult:
    accepted: bool
    mode: str                       # objective | explore | halt
    reasons: list[str]
    champion_holdout: Panel
    challenger_holdout: Panel
    champion_cumulative: Panel
    challenger_cumulative: Panel
    control_spearman: float | None
    finetune_stats: dict[str, Any] = field(default_factory=dict)
    #: the campaign objective the panels were computed for (which targets could
    #: veto, which could halt) — recorded so a wave artefact is self-describing.
    objective: str = "target_cycle"
    #: Was a REJECTED challenger actually rolled back?  ``False`` on an accepted
    #: gate (nothing to roll back) and — the case that matters — ``False`` on a
    #: rejection the backend could not undo: a stateless-refit backend has no
    #: member state to snapshot, so its fine-tune survives the rejection and the
    #: SERVED weights have moved even though the champion pointer has not.
    #: ``accepted or weights_rolled_back`` is therefore the only honest reading of
    #: "the weights this wave serves are still the ones it was fitted on".
    weights_rolled_back: bool = False

    def as_dict(self) -> dict[str, Any]:
        # the cumulative scalars are the ones that DECIDED panel 2, i.e. the mean
        # over the veto set — not the mean over every reported target.
        veto = accept_targets(self.objective)
        champ_cum = self.champion_cumulative.mean_skill_over(veto)
        chal_cum = self.challenger_cumulative.mean_skill_over(veto)
        return {
            "accepted": self.accepted,
            "mode": self.mode,
            "reasons": list(self.reasons),
            "objective": self.objective,
            "accept_targets": list(veto),
            "halt_primaries": list(halt_primaries(self.objective)),
            "champion_holdout_skill": self.champion_holdout.skill,
            "challenger_holdout_skill": self.challenger_holdout.skill,
            "champion_cumulative_skill": round(champ_cum, 4)
            if math.isfinite(champ_cum) else None,
            "challenger_cumulative_skill": round(chal_cum, 4)
            if math.isfinite(chal_cum) else None,
            "control_spearman": self.control_spearman,
            "finetune": self.finetune_stats,
        }


# --------------------------------------------------------------------------- #
# panel evaluation
# --------------------------------------------------------------------------- #
def _row_get(row: Any, key: str, default: Any = None) -> Any:
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _pattern_of(row: Any) -> Pattern | None:
    try:
        return unpack_pattern(str(_row_get(row, "pattern")))
    except (ValueError, KeyError, TypeError):
        return None


def _flatness_predictions(model: Any, patterns: Sequence[Pattern], case: CaseKey,
                          cell: float) -> dict[str, np.ndarray]:
    """``{"node_peak": [...], "map_cov": [...]}`` from whatever head the model has.

    Mirrors :func:`..search.acquisition.predict_flatness` so the panel scores the
    SAME quantity the acquisition ranks by.  A backend with no map head yields
    all-NaN, which makes the flatness skills NaN — the gate then has no finite
    primary and falls to ``explore`` rather than inventing a verdict.
    """
    n = len(patterns)
    nan = np.full(n, np.nan)
    if not n:
        return {"node_peak": nan, "map_cov": nan.copy()}
    fn = getattr(model, "predict_map_flatness", None)
    if callable(fn):
        pk_m, _pk_s, cv_m, _cv_s = fn(patterns, case, cell)
        return {"node_peak": np.asarray(pk_m, dtype=float),
                "map_cov": np.asarray(cv_m, dtype=float)}
    fn = getattr(model, "predict_map_peak", None)
    if callable(fn):
        pk_m, _pk_s = fn(patterns, case, cell)
        return {"node_peak": np.asarray(pk_m, dtype=float), "map_cov": nan}
    return {"node_peak": nan, "map_cov": nan.copy()}


def evaluate_panel(model: Any, rows: Sequence[Any], case: CaseKey, cell: float,
                   *, objective: str = "target_cycle") -> Panel:
    """Per-target within-case Spearman (primary) + MAE of ``model`` on ``rows``.

    ``objective`` selects the target set (:func:`panel_targets`); under
    ``flat_power`` that adds ``node_peak`` / ``map_cov``, scored from the map head.
    """

    from ..vendor.masterrl.surrogate import _spearman

    panel = Panel()
    parsed = [(r, _pattern_of(r)) for r in rows]
    parsed = [(r, p) for r, p in parsed if p is not None]
    if not parsed:
        return panel
    patterns = [p for _, p in parsed]
    names = panel_targets(objective)
    prediction = model.predict(patterns, case, cell)
    mean = np.asarray(prediction.mean, dtype=float)
    flat = (_flatness_predictions(model, patterns, case, cell)
            if any(n in _FLAT_TARGETS for n in names) else {})
    panel.n = len(patterns)
    for name in names:
        truth = np.asarray(
            [_finite(_row_get(r, name)) for r, _ in parsed], dtype=float
        )
        if name in flat:
            pred = np.asarray(flat[name], dtype=float)
        else:
            pred = mean[:, _TARGET_TO_SURR[name]]
        mask = np.isfinite(truth) & np.isfinite(pred)
        if mask.sum() < 2:
            panel.skill[name] = float("nan")
            panel.mae[name] = float("nan")
            continue
        t = truth[mask]
        p = pred[mask]
        panel.mae[name] = float(np.mean(np.abs(p - t)))
        rho = _spearman(t, p)
        panel.skill[name] = float("nan") if math.isnan(rho) else float(rho)
    return panel


def _finite(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return fv


def control_spearman(model: Any, rows: Sequence[Any], case: CaseKey, cell: float) -> float | None:
    """Pooled control-slot Spearman: predicted vs verified cyclen (audit only)."""

    parsed = [(r, _pattern_of(r)) for r in rows]
    parsed = [(r, p) for r, p in parsed if p is not None]
    if len(parsed) < 2:
        return None
    pred = model.predict([p for _, p in parsed], case, cell)
    predicted = np.asarray(pred.mean, dtype=float)[:, _TARGET_TO_SURR["cyclen"]]
    actual = np.asarray([_finite(_row_get(r, "cyclen")) for r, _ in parsed], dtype=float)
    mask = np.isfinite(predicted) & np.isfinite(actual)
    if mask.sum() < 2:
        return None
    from ..vendor.masterrl.surrogate import _spearman

    value = _spearman(actual[mask], predicted[mask])
    return None if math.isnan(value) else float(value)


# --------------------------------------------------------------------------- #
# gate (pure decision)
# --------------------------------------------------------------------------- #
def gate(
    champion_holdout: Panel,
    challenger_holdout: Panel,
    champion_cumulative: Panel,
    challenger_cumulative: Panel,
    *,
    epsilon: float = 0.02,
    skill_objective: float = 0.10,
    skill_halt: float = 0.0,
    objective: str = "target_cycle",
) -> tuple[bool, str, list[str]]:
    """Pure two-panel decision → ``(accepted, mode, reasons)`` (plan sec. 4.6).

    ``objective`` picks the veto set (:func:`accept_targets`) and the halt
    primaries (:func:`halt_primaries`); the default reproduces the historical
    F_r/CBC/F_q/cyclen/|AO| + F_r/cyclen behaviour exactly.
    """

    reasons: list[str] = []
    accepted = True
    veto = accept_targets(objective)

    # Panel 1: no per-target in-dist regression on the frozen holdout.
    for name in veto:
        champ = champion_holdout.skill.get(name, float("nan"))
        chal = challenger_holdout.skill.get(name, float("nan"))
        if math.isfinite(champ) and math.isfinite(chal) and chal < champ - epsilon:
            accepted = False
            reasons.append(f"{name}: holdout skill regressed {chal:.3f} < {champ:.3f}-ε")

    # Panel 2: campaign-cumulative skill not-worse — over the SAME veto set.
    # Averaging every REPORTED target instead handed the retired F_r skill a
    # second, hidden vote on acceptance (program §10 STOP); with the default
    # objective the veto set IS the panel set, so this is unchanged there.
    champ_cum = champion_cumulative.mean_skill_over(veto)
    chal_cum = challenger_cumulative.mean_skill_over(veto)
    if math.isfinite(champ_cum) and math.isfinite(chal_cum) and chal_cum < champ_cum - epsilon:
        accepted = False
        reasons.append(f"cumulative skill worse {chal_cum:.3f} < {champ_cum:.3f}-ε")

    # Mode from the retained model's primary skill.
    retained = challenger_holdout if accepted else champion_holdout
    names = halt_primaries(objective)
    primaries = [retained.skill.get(n, float("nan")) for n in names]
    finite = [v for v in primaries if math.isfinite(v)]
    if finite and all(v <= skill_halt for v in finite):
        mode = "halt"
        reasons.append(
            f"primary targets {list(names)} proven no-skill (<= halt threshold)")
    elif finite and all(v >= skill_objective for v in finite):
        mode = "objective"
    else:
        mode = "explore"
    return accepted, mode, reasons


# --------------------------------------------------------------------------- #
# snapshot / restore (CNN member state; no-op for stateless refit backends)
# --------------------------------------------------------------------------- #
def _snapshot(model: Any) -> Any:
    members = getattr(model, "members", None)
    if members is not None:
        return [copy.deepcopy(m.state_dict()) for m in members]
    return None


def _restore(model: Any, snapshot: Any) -> bool:
    """Put the champion weights back; ``False`` when the backend cannot.

    The return value is load-bearing, not decoration: a stateless-refit backend
    has no ``members`` to snapshot, so a REJECTED challenger is not rolled back —
    the served weights moved anyway.  The caller must be able to tell that apart
    from a real rollback (campaign: the map calibration was fitted on the
    pre-refit head, and a rejection is otherwise read as "nothing changed").
    """
    members = getattr(model, "members", None)
    if members is None or snapshot is None:
        return False
    for member, state in zip(members, snapshot):
        member.load_state_dict(state)
    return True


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
class WaveUpdater:
    """Fine-tune + gate one wave; keeps the champion unless the gate accepts."""

    def __init__(
        self,
        case: CaseKey,
        cell: float,
        holdout_rows: Sequence[Any],
        *,
        epsilon: float = 0.02,
        skill_objective: float = 0.10,
        skill_halt: float = 0.0,
        finetune_epochs: int = 3,
        new_weight: int = 1,
        objective: str = "target_cycle",
    ) -> None:
        self.case = case
        self.cell = float(cell)
        self.holdout_rows = list(holdout_rows)
        #: campaign objective — selects the panel / veto / halt target sets so the
        #: model is accepted and halted on the axes the search actually optimizes.
        self.objective = str(objective)
        self.epsilon = float(epsilon)
        self.skill_objective = float(skill_objective)
        self.skill_halt = float(skill_halt)
        self.finetune_epochs = int(finetune_epochs)
        #: boundary emphasis — this-campaign wave labels are oversampled this many
        #: times against the replay pool during fine-tune so the few fresh labels
        #: actually move the discriminator (plan sec. 4.6).  1 = no emphasis.
        self.new_weight = max(1, int(new_weight))

    def update(
        self,
        model: Any,
        new_rows: Sequence[Any],
        replay_rows: Sequence[Any],
        cumulative_rows: Sequence[Any],
        control_rows: Sequence[Any] = (),
        *,
        seed: int = 0,
    ) -> GateResult:
        champ_holdout = evaluate_panel(model, self.holdout_rows, self.case,
                                       self.cell, objective=self.objective)
        champ_cum = evaluate_panel(model, cumulative_rows, self.case, self.cell,
                                   objective=self.objective)

        # Boundary emphasis: oversample the fresh wave labels ``new_weight`` times
        # so each fresh boundary sample carries ~``new_weight``× the gradient of a
        # replay row (the 8 new labels/wave move the discriminator instead of
        # being swamped by the 512-row replay pool).  Replay is untouched.
        new_list = list(new_rows)
        weighted_new = new_list * self.new_weight if new_list else new_list
        snapshot = _snapshot(model)
        stats = dict(model.finetune(
            weighted_new, list(replay_rows), epochs=self.finetune_epochs, seed=seed
        ))
        # Report the true (unweighted) fresh-label count + the emphasis factor,
        # not the inflated oversampled length the backend saw.
        stats["n_new"] = len(new_list)
        stats["new_weight"] = self.new_weight

        chal_holdout = evaluate_panel(model, self.holdout_rows, self.case,
                                      self.cell, objective=self.objective)
        chal_cum = evaluate_panel(model, cumulative_rows, self.case, self.cell,
                                  objective=self.objective)
        spearman = control_spearman(model, control_rows, self.case, self.cell)

        accepted, mode, reasons = gate(
            champ_holdout, chal_holdout, champ_cum, chal_cum,
            epsilon=self.epsilon,
            skill_objective=self.skill_objective,
            skill_halt=self.skill_halt,
            objective=self.objective,
        )
        # A rejected challenger is rolled back — WHEN the backend can be rolled
        # back.  ``weights_rolled_back=False`` on a rejection means the fine-tune
        # survived the rejection (stateless refit), i.e. the served weights are
        # NOT the champion the run started the wave with.
        rolled_back = _restore(model, snapshot) if not accepted else False

        return GateResult(
            accepted=accepted,
            weights_rolled_back=rolled_back,
            mode=mode,
            reasons=reasons,
            champion_holdout=champ_holdout,
            challenger_holdout=chal_holdout,
            champion_cumulative=champ_cum,
            challenger_cumulative=chal_cum,
            control_spearman=spearman,
            finetune_stats=dict(stats),
            objective=self.objective,
        )


__all__ = [
    "GateResult", "Panel", "WaveUpdater", "accept_targets", "control_spearman",
    "evaluate_panel", "gate", "halt_primaries", "panel_targets",
]
