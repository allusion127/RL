"""The flatness A/B judging apparatus — program sections 8.2 through 8.6.

This is the thing that decides.  It takes arms that were all predicted on the
**same C2 rows**, forms paired cell-clustered BCa intervals on every
pre-registered metric, applies section 8.5's rule, and returns one of four
verdicts.  Three of them are not promotions.

Three structural guarantees, each enforced in code rather than in a docstring:

**1. A verdict without a control is impossible.**  Every judgement is produced
from a :class:`FlatArena`, and :class:`FlatArena` refuses to exist without the
control arm's predictions (:class:`ControlMissingError`).  Section 8.4's reason:
A7 differs from the incumbent in three ways at once -- init/schedule, the S2
training set, and the three new loss terms -- and section 5's P-1 diagnosis makes
the *training set* the more probable source of any gain.  A two-way
arm-vs-incumbent comparison therefore cannot attribute what it measures, and an
unattributable measurement is not evidence for the loss workstream.  The control
holds init, data, schedule and seed fixed and zeroes only
``map_cov_weight`` / ``map_peak_soft_weight`` / ``map_cov_rank_weight``, so
``arm - control`` is the loss effect and ``control - incumbent`` is the data
effect.  Both are reported; neither is inferred from the other.

**2. Nothing promotes on a point estimate.**  Every condition in section 8.5 is
phrased on an interval: a gain must have ``ci_lo > 0``, a harm must have
``-ci_lo < margin``.  An arm that leads on every point estimate and whose
intervals all straddle the null is routed to :data:`ESCALATE`, never to
:data:`PROMOTE`.  This is not conservatism for its own sake -- section 8.1
measured the old primary statistic to take six distinct values across the whole
slate, so "A leads B" was regularly a rounding artifact.

**3. A near-tie goes to the user.**  Section 8.5's rule was fixed before the arms
ran; resolving an ambiguity with an unregistered tiebreak converts the whole
pre-registration into a post-hoc analysis, which is precisely the failure the
program was written to stop.  So the apparatus reports *which* comparison is too
close and *what* a human has to decide, and stops.

The falsification condition (section 8.6) is recorded on every judgement whether
or not it fires, because a pre-registered falsification that is only reported
when it fails is not pre-registered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import (
    FR_GUARD_KNOB as _FR_GUARD_KNOB, fr_guard_enforced as _fr_guard_enforced)
from . import ab_eval as E
from . import flat_metrics as FM
from .ab_paired import (
    AGGREGATES, DEFAULT_ALPHA, DEFAULT_REPS, PairedDiff, paired_cell_bootstrap)
from .c2_slice import C2Slice

#: Verdicts.  Exactly one of these is returned per arm; only PROMOTE promotes.
PROMOTE = "promote"
HOLD = "hold"
ESCALATE = "escalate"
REJECT = "reject"
BLOCKED = "blocked"

#: Section 8.5 condition 3 -- a support metric may not be established worse than
#: this.  A loss of 0.01 of within-cell rho / normalized P@8.
RANK_HARM_MARGIN = 0.01
#: Section 8.5 condition 5 -- secondary targets (f_r, f_q, cyclen).
SECONDARY_HARM_MARGIN = 0.02
#: Section 8.5 condition 4 -- the acquisition consumes LEVELS, so MAE and |bias|
#: get their own non-inferiority margins, in the physical units of node_peak.
LEVEL_HARM_MARGIN = 0.005

#: Per-metric non-inferiority margins (frozen with the rule).
HARM_MARGINS: dict[str, float] = {
    "M2_flat_tercile_rho_node_peak": RANK_HARM_MARGIN,
    "M2_flat_tercile_rho_map_cov": RANK_HARM_MARGIN,
    "M3_norm_p_at_8_node_peak": RANK_HARM_MARGIN,
    "M3_norm_p_at_8_map_cov": RANK_HARM_MARGIN,
    "M5_cell_rho_f_r": SECONDARY_HARM_MARGIN,
    "M5_cell_rho_f_q": SECONDARY_HARM_MARGIN,
    "M5_cell_rho_cyclen": SECONDARY_HARM_MARGIN,
    "M7_cell_mae_node_peak": LEVEL_HARM_MARGIN,
    "M7_cell_mae_map_cov": LEVEL_HARM_MARGIN,
    "M7_abs_bias_node_peak": LEVEL_HARM_MARGIN,
}

#: The go/no-go axis (section 8.2 M0) and its operating-scale support (M1).
PRIMARY_METRICS: tuple[str, ...] = (
    "M0_regret8_node_peak", "M1_signhit_0.02_node_peak")
#: Extended no-regression gate targets (section 8.2 M6) -- the draft's gate looked
#: at cyclen and f_r only, i.e. it could not see a map collapse at all.
M6_TARGETS: tuple[str, ...] = ("cyclen", "f_r", "map_cov", "node_peak")

#: The condition-5 metric held DORMANT by the F_r deferral (user decision
#: 2026-07-26).  Scored on every judgement, reported with its harm bound, and
#: excluded from the verdict until ``[curriculum] gate_noreg_fr_guard_enabled``
#: is set -- the same one switch that re-arms the curriculum's promotion gates.
#: ``M5_cell_rho_f_q`` and ``M5_cell_rho_cyclen`` are untouched.
FR_HARM_METRIC = "M5_cell_rho_f_r"

#: Section 8.5, transcribed.  Stamped into every judgement so an artifact carries
#: the rule it was judged under.
PROMOTION_RULE: tuple[str, ...] = (
    "1. M0: node_peak top-8 regret improves, paired CI lower bound > 0",
    "2. M1: sign-hit in the |d| <= 0.02 band improves on node_peak, CI lower bound > 0",
    f"3. M2/M3: neither node_peak nor map_cov is harmed, harm upper bound < {RANK_HARM_MARGIN}",
    f"4. M6/M7: extended gate passes; node_peak MAE and |bias| non-inferior at {LEVEL_HARM_MARGIN}",
    f"5. M5: f_r / f_q / cyclen within-cell rho harm upper bound < {SECONDARY_HARM_MARGIN}",
    "6. section 8.3 provenance: condition 1 also holds on the production-only stratum",
    "Delta75/SD is reported and decides nothing.",
)

#: Section 8.6, transcribed.
FALSIFICATION = {
    "condition": ("M0 (regret) does not improve in the |d| <= 0.02 operating band"),
    "consequence": ("the loss workstream is the wrong lever: do NOT raise "
                    "map_cov_weight"),
    "switch_to": ("a different mechanism -- verify-many-then-select, a cheap "
                  "physics screen, or a repeated-sampling bandit"),
}


class ControlMissingError(ValueError):
    """Raised when a judgement is attempted without the section 8.4 control."""


# --------------------------------------------------------------------------- #
# the arena: the same rows, for every arm
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlatArena:
    """Predictions from every arm on ONE fixed row set, plus the truth.

    The row alignment is the pairing.  It is validated here, once, so no metric
    or bootstrap downstream has to re-check it, and so an arm scored on a
    different (even a merely reordered) row set cannot enter a comparison.
    """

    cells: np.ndarray
    truth: dict[str, np.ndarray]
    preds: dict[str, dict[str, np.ndarray]]
    control: str
    incumbent: str | None = None
    record_ids: tuple[str, ...] = ()
    frozen_cells: tuple[str, ...] = ()
    production: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.control:
            raise ControlMissingError(
                "the flatness A/B has no control arm.  Section 8.4 makes the "
                "control mandatory: without it, arm - incumbent confounds the "
                "loss change with the S2 training-set change, and section 5's "
                "P-1 diagnosis says the data is the more probable source.")
        if self.control not in self.preds:
            raise ControlMissingError(
                f"control arm {self.control!r} has no predictions in this arena "
                f"(have: {sorted(self.preds)}).  Section 8.4 makes the control "
                "mandatory: a verdict without a control is not a verdict, "
                "because arm - incumbent cannot separate the loss change from "
                "the S2 training-set change.")
        n = len(self.cells)
        for tgt, arr in self.truth.items():
            if len(arr) != n:
                raise ValueError(f"truth[{tgt!r}] has {len(arr)} rows, cells has {n}")
        for arm, by_tgt in self.preds.items():
            for tgt, arr in by_tgt.items():
                if len(arr) != n:
                    raise ValueError(
                        f"arm {arm!r} target {tgt!r} has {len(arr)} rows but the "
                        f"arena has {n}: arms must be scored on the SAME rows in "
                        "the SAME order -- that alignment is what makes the "
                        "comparison paired")
        if self.production is not None and len(self.production) != n:
            raise ValueError("production mask is not aligned with the arena rows")

    @property
    def arms(self) -> tuple[str, ...]:
        """Every arm that is not the control (the incumbent included: it is
        compared, never promoted)."""
        return tuple(a for a in sorted(self.preds) if a != self.control)

    @property
    def challengers(self) -> tuple[str, ...]:
        return tuple(a for a in self.arms if a != self.incumbent)

    def available_targets(self, arm: str) -> tuple[str, ...]:
        have = set(self.truth) & set(self.preds.get(arm, {})) \
            & set(self.preds[self.control])
        return tuple(sorted(have))

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_c2(cls, slice_: C2Slice, preds: Mapping[str, Mapping[str, np.ndarray]],
                *, control: str, incumbent: str | None = None) -> "FlatArena":
        targets = sorted({t for by in preds.values() for t in by})
        truth = {t: slice_.truth(t) for t in targets}
        return cls(
            cells=np.asarray(slice_.cells),
            truth={t: np.asarray(v, dtype=float) for t, v in truth.items()},
            preds={a: {t: np.asarray(v, dtype=float) for t, v in by.items()}
                   for a, by in preds.items()},
            control=control, incumbent=incumbent,
            record_ids=slice_.record_ids,
            frozen_cells=slice_.frozen_cells,
            production=slice_.production_mask(),
            provenance=dict(slice_.provenance),
        )


# --------------------------------------------------------------------------- #
# paired differences
# --------------------------------------------------------------------------- #
def _restrict(per_cell: Mapping[str, float],
              frozen: Sequence[str]) -> dict[str, float]:
    """Keep only the frozen cell list (section 8.2: the cell list is frozen)."""
    if not frozen:
        return dict(per_cell)
    keep = set(frozen)
    return {k: v for k, v in per_cell.items() if k in keep}


def paired_metric(arena: FlatArena, arm: str, spec: FM.MetricSpec, *,
                  control: str | None = None, rows: np.ndarray | None = None,
                  reps: int = DEFAULT_REPS, seed: int = 0,
                  alpha: float = DEFAULT_ALPHA) -> PairedDiff:
    """One metric's paired CI for ``arm`` against ``control`` on the same rows."""
    ctl = control or arena.control
    sel = slice(None) if rows is None else np.asarray(rows, dtype=bool)
    cells = arena.cells[sel]
    true = arena.truth[spec.target][sel]
    a = arena.preds[arm][spec.target][sel]
    c = arena.preds[ctl][spec.target][sel]
    a_cells = _restrict(spec.per_cell(a, true, cells), arena.frozen_cells)
    c_cells = _restrict(spec.per_cell(c, true, cells), arena.frozen_cells)
    return paired_cell_bootstrap(
        a_cells, c_cells, metric=spec.key, arm=arm, control=ctl,
        higher_is_better=spec.higher_is_better, reps=reps, seed=seed, alpha=alpha,
        aggregate=spec.aggregate)


def paired_between(arena: FlatArena, a: str, b: str, metric_key: str, *,
                   reps: int = DEFAULT_REPS, seed: int = 0) -> PairedDiff:
    """Head-to-head paired CI between two arms (used to test a near-tie)."""
    return paired_metric(arena, a, FM.METRICS_BY_KEY[metric_key], control=b,
                         reps=reps, seed=seed)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def _gain_status(d: PairedDiff) -> str:
    # ``not d.measured`` covers BOTH no-evidence methods: too few clusters to form
    # an interval, and a point-mass resample whose zero-width interval would
    # otherwise read as the most decisive result in the slate.
    if not d.measured:
        return "unresolved"
    if d.establishes_gain():
        return "established"
    if np.isfinite(d.ci_hi) and d.ci_hi < 0.0:
        return "established_worse"
    return "straddles_null"


def _harm_status(d: PairedDiff, margin: float) -> str:
    if not d.measured:
        return "unresolved"
    if d.bounds_harm(margin):
        return "bounded"
    if np.isfinite(d.ci_hi) and d.ci_hi < -float(margin):
        return "violated"
    return "unresolved"


def judge_arm(arena: FlatArena, arm: str, *, reps: int = DEFAULT_REPS,
              seed: int = 0, alpha: float = DEFAULT_ALPHA,
              gate: Mapping[str, Any] | None = None,
              reported: Mapping[str, Any] | None = None,
              fr_guarded: bool | None = None) -> dict[str, Any]:
    """Apply section 8.5 to one arm.  Pure; the arena has already been validated.

    F_r DEFERRAL (user decision 2026-07-26).  This is a PROMOTION surface, so the
    deferral applies here exactly as it does to the curriculum's retrain gate:
    ``M5_cell_rho_f_r`` (condition 5) is measured and reported but cannot withhold
    a promotion, and the M6 extended gate handed in via ``gate`` is expected to
    have been computed with the same setting (``ab_eval.no_regression_gate``
    scores ``f_r`` without enforcing it).  ``fr_guarded`` resolves through
    :func:`..config.fr_guard_enforced` — the ONE switch, ``[curriculum]
    gate_noreg_fr_guard_enabled`` — so the FA-optimized phase re-arms this
    condition and the in-loop gates together.  ``f_q`` and ``cyclen`` keep their
    M5 teeth: the deferral is one axis wide.
    """
    if arm == arena.control:
        raise ValueError("the control cannot be judged against itself")
    if arm not in arena.preds:
        raise KeyError(f"arm {arm!r} is not in the arena")

    fr_enforced = _fr_guard_enforced(fr_guarded)
    m5_report_only = () if fr_enforced else (FR_HARM_METRIC,)
    targets = set(arena.available_targets(arm))
    specs = [m for m in FM.PRE_REGISTERED_METRICS if m.target in targets]
    diffs: dict[str, PairedDiff] = {}
    for i, spec in enumerate(specs):
        diffs[spec.key] = paired_metric(arena, arm, spec, reps=reps,
                                        seed=seed + i, alpha=alpha)

    # section 8.3: the primary must also hold on the production-only stratum
    prod: dict[str, PairedDiff] = {}
    if arena.production is not None and arena.production.any():
        for i, key in enumerate(PRIMARY_METRICS):
            spec = FM.METRICS_BY_KEY.get(key)
            if spec is not None and spec.target in targets:
                prod[key] = paired_metric(arena, arm, spec,
                                          rows=arena.production,
                                          reps=reps, seed=seed + 500 + i,
                                          alpha=alpha)

    conditions: list[dict[str, Any]] = []

    # -- 1 / 2: the primaries ------------------------------------------------ #
    for n, key in enumerate(PRIMARY_METRICS, start=1):
        d = diffs.get(key)
        if d is None:
            conditions.append({"id": f"{n}", "rule": PROMOTION_RULE[n - 1],
                               "metric": key, "status": "unresolved",
                               "detail": "metric not computable on this slice "
                                         "(target absent)"})
            continue
        st = _gain_status(d)
        conditions.append({
            "id": f"{n}", "rule": PROMOTION_RULE[n - 1], "metric": key,
            "status": {"established": "pass", "established_worse": "fail",
                       "straddles_null": "unresolved",
                       "unresolved": "unresolved"}[st],
            "evidence": st, "paired": d.to_dict(),
            "detail": _explain(d, st)})

    # -- 3: rank harm -------------------------------------------------------- #
    conditions.append(_harm_condition(
        "3", PROMOTION_RULE[2], diffs,
        ("M2_flat_tercile_rho_node_peak", "M2_flat_tercile_rho_map_cov",
         "M3_norm_p_at_8_node_peak", "M3_norm_p_at_8_map_cov")))

    # -- 4: extended gate + level accuracy ----------------------------------- #
    level = _harm_condition(
        "4", PROMOTION_RULE[3], diffs,
        ("M7_cell_mae_node_peak", "M7_cell_mae_map_cov", "M7_abs_bias_node_peak"))
    if gate is not None:
        level["m6_gate"] = dict(gate)
        # The M6 gate's own F_r checks are report-only under the deferral
        # (``ab_eval.no_regression_gate`` excludes them from ``worst_drop`` and
        # from ``pass``), so this branch inherits the demotion rather than
        # re-deciding it.  A gate computed at the OTHER setting is a mismatched
        # input, and saying so beats silently mixing the two.
        if bool(gate.get("fr_guard", {}).get("enforced", fr_enforced)) != fr_enforced:
            level["status"] = "unresolved"
            level["detail"] = (
                "the M6 extended gate was computed with the F_r guard "
                f"{'ENFORCED' if not fr_enforced else 'DEFERRED'} while this "
                f"judgement runs with it {'DEFERRED' if not fr_enforced else 'ENFORCED'}"
                f" ({_FR_GUARD_KNOB}); re-run the gate at one setting; "
                + str(level.get("detail", "")))
        elif not gate.get("pass"):
            level["status"] = "fail"
            level["detail"] = (
                f"extended no-regression gate FAILED (worst per-cell ENFORCED drop "
                f"{gate.get('worst_drop')}, epsilon {gate.get('epsilon')}); "
                + str(level.get("detail", "")))
        if gate.get("note"):
            level["m6_note"] = str(gate["note"])
    else:
        level["m6_gate"] = None
        if level["status"] == "pass":
            level["status"] = "unresolved"
            level["detail"] = ("the M6 extended gate was not supplied, so "
                               "condition 4 cannot pass; " + str(level.get("detail", "")))
    conditions.append(level)

    # -- 5: secondary non-regression ----------------------------------------- #
    conditions.append(_harm_condition(
        "5", PROMOTION_RULE[4], diffs,
        ("M5_cell_rho_f_r", "M5_cell_rho_f_q", "M5_cell_rho_cyclen"),
        report_only=m5_report_only))

    # -- 6: provenance stratum ----------------------------------------------- #
    key = PRIMARY_METRICS[0]
    dp = prod.get(key)
    if dp is None:
        conditions.append({"id": "6", "rule": PROMOTION_RULE[5], "metric": key,
                           "status": "unresolved",
                           "detail": "no production-only stratum in this slice; "
                                     "section 8.3 forbids deciding on the "
                                     "proposed stratum alone"})
    else:
        st = _gain_status(dp)
        conditions.append({
            "id": "6", "rule": PROMOTION_RULE[5], "metric": key,
            "status": "pass" if st == "established" else
                      ("fail" if st == "established_worse" else "unresolved"),
            "evidence": st, "paired": dp.to_dict(),
            "detail": ("production-only stratum: " + _explain(dp, st))})

    verdict, reason = _verdict(conditions, diffs, arm, arena.control)

    m0 = diffs.get("M0_regret8_node_peak")
    m1 = diffs.get("M1_signhit_0.02_node_peak")
    falsified = bool(m0 is not None and m1 is not None
                     and not m0.establishes_gain()
                     and not m1.establishes_gain())
    out: dict[str, Any] = {
        "schema": "flat_ab_judgement_v1",
        "arm": arm,
        "control": arena.control,
        "incumbent": arena.incumbent,
        "verdict": verdict,
        "reason": reason,
        "rule": list(PROMOTION_RULE),
        "conditions": conditions,
        "paired": {k: d.to_dict() for k, d in diffs.items()},
        "paired_production": {k: d.to_dict() for k, d in prod.items()},
        "falsification": {**FALSIFICATION, "triggered": falsified,
                          "note": ("pre-registered before the arms ran; reported "
                                   "on every judgement, fired or not")},
        "n_cells_frozen": len(arena.frozen_cells),
        "slice": dict(arena.provenance),
        "reps": int(reps), "seed": int(seed), "alpha": float(alpha),
        "fr_guard": {"target": "f_r", "enforced": fr_enforced,
                     "knob": _FR_GUARD_KNOB,
                     "report_only_metrics": list(m5_report_only)},
    }
    if reported:
        out["reported_not_deciding"] = dict(reported)
    return out


def _explain(d: PairedDiff, status: str) -> str:
    lo = "-inf" if not np.isfinite(d.ci_lo) else f"{d.ci_lo:+.4f}"
    hi = "+inf" if not np.isfinite(d.ci_hi) else f"{d.ci_hi:+.4f}"
    head = (f"paired {d.method} CI on {d.arm} - {d.control} = "
            f"{d.point:+.4f} [{lo}, {hi}] over {d.n_cells} cells")
    if status == "established":
        return head + " -- gain established (lower bound above the null)"
    if status == "established_worse":
        return head + " -- established WORSE than the control"
    if status == "unresolved":
        if d.degenerate:
            return (head + " -- every paired cell moved by exactly the same "
                           "amount, so the resample is a point mass: the "
                           "zero-width interval expresses no uncertainty and "
                           "cannot establish anything")
        return head + " -- too few paired cells to judge"
    if d.favours_arm_on_points_only():
        return (head + " -- the POINT estimate favours the arm but the interval "
                       "does not exclude the null; section 8.5 forbids promoting "
                       "on this")
    return head + " -- indistinguishable from the control"


def _harm_condition(cid: str, rule: str, diffs: Mapping[str, PairedDiff],
                    keys: Sequence[str],
                    report_only: Sequence[str] = ()) -> dict[str, Any]:
    """One section-8.5 non-inferiority condition.

    ``report_only`` metrics are measured, reported and shown with their status,
    but excluded from the condition's verdict — the F_r deferral, expressed at
    the level the rule is evaluated rather than by deleting the metric.  Every
    entry carries ``enforced`` so an artifact says which ones had teeth.
    """
    ro = set(report_only)
    per: dict[str, Any] = {}
    worst = "pass"
    for k in keys:
        d = diffs.get(k)
        if d is None:
            continue
        st = _harm_status(d, HARM_MARGINS[k])
        enforced = k not in ro
        per[k] = {"status": st, "enforced": enforced, "margin": HARM_MARGINS[k],
                  "harm_upper": (None if not math.isfinite(d.harm_upper)
                                 else round(d.harm_upper, 6)),
                  "paired": d.to_dict()}
        if not enforced:
            continue
        if st == "violated":
            worst = "fail"
        elif st == "unresolved" and worst != "fail":
            worst = "unresolved"
    if not per:
        return {"id": cid, "rule": rule, "status": "unresolved", "metrics": {},
                "report_only": [], "detail":
                "none of this condition's metrics are computable here"}
    bad = [k for k, v in per.items() if v["status"] != "bounded" and v["enforced"]]
    detail = ("every enforced harm bound holds" if not bad else
              "harm not bounded for: " + ", ".join(sorted(bad)))
    ro_present = sorted(k for k in per if not per[k]["enforced"])
    if ro_present:
        ro_bad = [k for k in ro_present if per[k]["status"] != "bounded"]
        detail += ("; REPORT-ONLY (scored, not enforced): "
                   + ", ".join(ro_present)
                   + (f" — harm NOT bounded for {', '.join(ro_bad)}, and admitted "
                      "anyway" if ro_bad else "")
                   + f". Set {_FR_GUARD_KNOB} = true to enforce.")
    return {"id": cid, "rule": rule, "status": worst, "metrics": per,
            "report_only": ro_present, "detail": detail}


def _verdict(conditions: Sequence[Mapping[str, Any]],
             diffs: Mapping[str, PairedDiff], arm: str,
             control: str) -> tuple[str, str]:
    statuses = [c["status"] for c in conditions]
    if "fail" in statuses:
        failed = [c["id"] for c in conditions if c["status"] == "fail"]
        return REJECT, (f"{arm}: pre-registered condition(s) {', '.join(failed)} "
                        f"FAILED against control {control}")
    if all(s == "pass" for s in statuses):
        return PROMOTE, (f"{arm}: every section 8.5 condition holds against "
                         f"control {control}, each on a paired CI that excludes "
                         "the null")
    unresolved = [c["id"] for c in conditions if c["status"] == "unresolved"]
    near = [k for k in PRIMARY_METRICS
            if k in diffs and diffs[k].favours_arm_on_points_only()]
    if near:
        pretty = "; ".join(
            f"{k}: point {diffs[k].point:+.4f}, CI "
            f"[{diffs[k].ci_lo:+.4f}, {diffs[k].ci_hi:+.4f}]" for k in near)
        return ESCALATE, (
            f"{arm}: the point estimate favours the arm on {', '.join(near)} but "
            f"the paired CI does not exclude the null ({pretty}). The "
            "pre-registered rule cannot separate this from the control, and an "
            "unregistered tiebreak would void the pre-registration -- a human "
            "must decide whether to spend more labels or drop the arm. "
            f"Unresolved conditions: {', '.join(unresolved)}")
    return HOLD, (f"{arm}: condition(s) {', '.join(unresolved)} unresolved and no "
                  "primary point estimate favours the arm; nothing to escalate "
                  "and nothing to promote")


# --------------------------------------------------------------------------- #
# the whole slate
# --------------------------------------------------------------------------- #
def judge_all(arena: FlatArena, *, reps: int = DEFAULT_REPS, seed: int = 0,
              alpha: float = DEFAULT_ALPHA,
              gates: Mapping[str, Mapping[str, Any]] | None = None,
              reported: Mapping[str, Mapping[str, Any]] | None = None,
              fr_guarded: bool | None = None) -> dict[str, Any]:
    """Judge every challenger, plus the section 8.4 three-way attribution.

    ``fr_guarded`` is threaded to every arm so a slate is judged at ONE setting
    of the F_r guard; it is also stamped on the slate.
    """
    gates = dict(gates or {})
    reported = dict(reported or {})
    per_arm = {a: judge_arm(arena, a, reps=reps, seed=seed, alpha=alpha,
                            gate=gates.get(a), reported=reported.get(a),
                            fr_guarded=fr_guarded)
               for a in arena.challengers}

    out: dict[str, Any] = {
        "schema": "flat_ab_slate_v1",
        "control": arena.control,
        "incumbent": arena.incumbent,
        "arms": sorted(per_arm),
        "judgements": per_arm,
        "rule": list(PROMOTION_RULE),
        "falsification": dict(FALSIFICATION),
        "slice": dict(arena.provenance),
        "n_rows": int(len(arena.cells)),
        "n_cells_frozen": len(arena.frozen_cells),
        "reps": int(reps), "seed": int(seed), "alpha": float(alpha),
        "fr_guard": {"target": "f_r",
                     "enforced": _fr_guard_enforced(fr_guarded),
                     "knob": _FR_GUARD_KNOB},
    }

    # -- section 8.4: separate the data effect from the loss effect ---------- #
    if arena.incumbent and arena.incumbent in arena.preds:
        three: dict[str, Any] = {}
        for i, key in enumerate(PRIMARY_METRICS):
            spec = FM.METRICS_BY_KEY.get(key)
            if spec is None or spec.target not in arena.truth:
                continue
            d = paired_metric(arena, arena.control, spec,
                              control=arena.incumbent, reps=reps, seed=seed + 900 + i)
            three[key] = {
                "control_minus_incumbent": d.to_dict(),
                "reading": ("this is the DATA/init effect (S2 training set + "
                            "schedule); arm - control is the LOSS effect. "
                            "Section 8.4 requires both to be reported before "
                            "either is interpreted."),
                "mde80_from_this_se": (None if not math.isfinite(d.mde())
                                       else round(d.mde(), 6)),
            }
        out["three_way"] = {
            "incumbent": arena.incumbent, "control": arena.control,
            "metrics": three,
            "note": ("power pre-disclosure (section 8.3): the MDE above comes "
                     "from the control-vs-incumbent paired SE measured on this "
                     "very slice."),
        }
    else:
        out["three_way"] = {
            "incumbent": arena.incumbent, "control": arena.control,
            "metrics": {},
            "note": ("no incumbent in the arena: the data effect and the loss "
                     "effect cannot be separated, so a promotion here attributes "
                     "nothing (section 8.4)."),
        }

    promote = sorted(a for a, j in per_arm.items() if j["verdict"] == PROMOTE)
    escalate = sorted(a for a, j in per_arm.items() if j["verdict"] == ESCALATE)
    if len(promote) > 1:
        pair = _closest_pair(arena, promote, reps=reps, seed=seed)
        if pair["separated"]:
            out["verdict"] = PROMOTE
            out["winner"] = pair["better"]
            out["reason"] = (
                f"{pair['better']} promotes: it clears every condition and its "
                f"head-to-head paired CI against {pair['other']} excludes the "
                "null")
        else:
            out["verdict"] = ESCALATE
            out["candidates"] = promote
            out["reason"] = (
                f"{len(promote)} arms clear every pre-registered condition "
                f"({', '.join(promote)}) and the head-to-head paired CI between "
                f"the top two ({pair['better']} vs {pair['other']}: "
                f"{pair['detail']}) does not separate them. The rule does not "
                "rank co-qualifiers, and inventing a tiebreak now would void the "
                "pre-registration -- a human must choose.")
        out["head_to_head"] = pair
    elif promote:
        out["verdict"] = PROMOTE
        out["winner"] = promote[0]
        out["reason"] = per_arm[promote[0]]["reason"]
    elif escalate:
        out["verdict"] = ESCALATE
        out["candidates"] = escalate
        out["reason"] = "; ".join(per_arm[a]["reason"] for a in escalate)
    else:
        out["verdict"] = HOLD
        out["reason"] = ("no arm satisfies the pre-registered promotion rule on "
                         "paired CIs")
    out["falsified_arms"] = sorted(
        a for a, j in per_arm.items() if j["falsification"]["triggered"])
    return out


def _closest_pair(arena: FlatArena, labels: Sequence[str], *, reps: int,
                  seed: int) -> dict[str, Any]:
    key = PRIMARY_METRICS[0]
    pts = {a: arena_point(arena, a, key, reps=reps, seed=seed) for a in labels}
    ranked = sorted(labels, key=lambda a: -(pts[a] if math.isfinite(pts[a]) else -1e18))
    a, b = ranked[0], ranked[1]
    d = paired_between(arena, a, b, key, reps=reps, seed=seed + 700)
    return {"better": a, "other": b, "metric": key,
            "separated": bool(d.establishes_gain()),
            "paired": d.to_dict(),
            "detail": _explain(d, _gain_status(d))}


def arena_point(arena: FlatArena, arm: str, metric_key: str, *, reps: int = 0,
                seed: int = 0) -> float:
    """Point estimate of the paired gain (no bootstrap) -- for ORDERING only.

    Used to pick which two co-qualifiers to test head-to-head.  It never decides
    anything by itself: the pair it selects is then judged on a paired CI.
    """
    spec = FM.METRICS_BY_KEY[metric_key]
    if spec.target not in arena.available_targets(arm):
        return float("nan")
    true = arena.truth[spec.target]
    a = _restrict(spec.per_cell(arena.preds[arm][spec.target], true, arena.cells),
                  arena.frozen_cells)
    c = _restrict(spec.per_cell(arena.preds[arena.control][spec.target], true,
                                arena.cells), arena.frozen_cells)
    sign = 1.0 if spec.higher_is_better else -1.0
    gains = [sign * (a[k] - c[k]) for k in sorted(set(a) & set(c))
             if math.isfinite(a[k]) and math.isfinite(c[k])]
    if not gains:
        return float("nan")
    return float(AGGREGATES[spec.aggregate](gains))


def reported_effective_resolution(arena: FlatArena, arm: str) -> dict[str, Any]:
    """``Delta75/SD`` for the record.  Section 8.5: reported, decides nothing.

    Kept because it is the statistic every previous report quoted, and dropping
    it silently would make this apparatus's numbers look incomparable with the
    history rather than deliberately different from it.
    """
    out: dict[str, Any] = {"decides": False,
                           "note": ("section 8.5: Delta75/SD is reported and "
                                    "decides nothing -- it takes six distinct "
                                    "values across the slate (section 8.1)")}
    for tgt in arena.available_targets(arm):
        if tgt not in E.DELTA_BINS:
            continue
        r = E.effective_resolution(arena.preds[arm][tgt], arena.truth[tgt],
                                   arena.cells, tgt)
        out[tgt] = {"delta75": r.get("delta75"),
                    "delta75_over_sd": r.get("delta75_over_sd"),
                    "within_cell_sd": r.get("within_cell_sd")}
    return out


# --------------------------------------------------------------------------- #
# the block ab_decide reads
# --------------------------------------------------------------------------- #
def paired_block(judgement: Mapping[str, Any]) -> dict[str, Any]:
    """The compact record :mod:`.ab_decide` consumes from a results document.

    Carrying the control label INSIDE the block is deliberate: the decision
    module re-checks it, so paired evidence computed against the wrong control
    (or against the incumbent) cannot be silently accepted as if it were the
    section 8.4 comparison.
    """
    return {
        "control": judgement.get("control"),
        "verdict": judgement.get("verdict"),
        "reason": judgement.get("reason"),
        "metrics": dict(judgement.get("paired") or {}),
        "production": dict(judgement.get("paired_production") or {}),
        "conditions": [
            {"id": c.get("id"), "status": c.get("status"), "metric": c.get("metric"),
             "detail": c.get("detail")}
            for c in (judgement.get("conditions") or [])],
        "falsification": dict(judgement.get("falsification") or {}),
        "slice": dict(judgement.get("slice") or {}),
    }


def render_slate(slate: Mapping[str, Any]) -> str:
    L = [f"verdict : {slate.get('verdict')}",
         f"control : {slate.get('control')}  incumbent: {slate.get('incumbent')}",
         f"slice   : {slate.get('n_rows')} rows / "
         f"{slate.get('n_cells_frozen')} frozen cells",
         f"reason  : {slate.get('reason')}", ""]
    L.append(f"{'arm':6s} {'verdict':9s} {'M0 regret gain (CI)':>34s}  conditions")
    for a in slate.get("arms", []):
        j = slate["judgements"][a]
        d = (j.get("paired") or {}).get("M0_regret8_node_peak") or {}
        pt, lo, hi = d.get("point"), d.get("ci_lo"), d.get("ci_hi")
        cell = ("—" if pt is None else
                f"{pt:+.4f} [{_f(lo)}, {_f(hi)}]")
        conds = " ".join(f"{c['id']}:{c['status'][:4]}" for c in j["conditions"])
        L.append(f"{a:6s} {j['verdict']:9s} {cell:>34s}  {conds}")
    if slate.get("falsified_arms"):
        L.append("")
        L.append("section 8.6 FALSIFICATION triggered for: "
                 + ", ".join(slate["falsified_arms"]))
        L.append(f"  -> {FALSIFICATION['consequence']}; {FALSIFICATION['switch_to']}")
    tw = slate.get("three_way") or {}
    if tw.get("metrics"):
        L.append("")
        L.append(f"section 8.4 three-way ({tw['incumbent']} -> {tw['control']} -> arm):")
        for k, v in tw["metrics"].items():
            d = v["control_minus_incumbent"]
            L.append(f"  {k}: data effect {_f(d.get('point'))} "
                     f"[{_f(d.get('ci_lo'))}, {_f(d.get('ci_hi'))}]  "
                     f"MDE80 {_f(v.get('mde80_from_this_se'))}")
    elif tw:
        L.append("")
        L.append(f"section 8.4: {tw.get('note')}")
    return "\n".join(L)


def _f(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not math.isfinite(f) else f"{f:+.{nd}f}"


__all__ = [
    "BLOCKED", "ESCALATE", "FALSIFICATION", "HARM_MARGINS", "HOLD",
    "LEVEL_HARM_MARGIN", "M6_TARGETS", "PRIMARY_METRICS", "PROMOTE",
    "PROMOTION_RULE", "RANK_HARM_MARGIN", "REJECT", "SECONDARY_HARM_MARGIN",
    "ControlMissingError", "FlatArena", "arena_point", "judge_all", "judge_arm",
    "paired_between", "paired_block", "paired_metric", "render_slate",
    "reported_effective_resolution",
]
