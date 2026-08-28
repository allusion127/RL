"""Apply the pre-registered decision rule to the scored arms and pick a winner.

``python -m lpopt.model.ab_decide``            -> verdict + promotion command
``python -m lpopt.model.ab_decide --promote``  -> also runs ``lpopt gate-promote``

The rule (design doc ``hires_model_ab_design_20260725.md`` section 7) was fixed
before the arms launched and is transcribed here verbatim:

1. **Primary — effective resolution.**  ``Delta75 / within-cell SD`` on fold C
   must improve over the B1 control by **>= 0.15** on node_peak OR map_CoV, with
   **no target worse by > 0.05**.
2. **Tertiary — honest gate.**  No within-cell regression vs the incumbent.
3. **Secondary — fold C within-cell rho.**  cyclen and F_r must not fall below
   the incumbent B0 (paired CI lower bound >= -0.02).

Interval evidence is now mandatory (flatness program section 8.3/8.5)
--------------------------------------------------------------------
Rules 1-3 above are all POINT comparisons, and section 8.1 measured why that is
not enough: ``Delta75`` is the lower edge of the first bin reaching 75%, so it
takes six distinct values over the whole slate -- three arms scored exactly
1.409 and three scored exactly 0.705, and the promotion that came out of it was
a tiebreak in metric's clothing.  So a **paired, cell-clustered BCa CI on
``arm - control``** (:mod:`.ab_paired`, produced by :mod:`.flat_ab`) is required
in addition, and it must EXCLUDE THE NULL:

* an arm with no paired record is not eligible, whatever its point estimates;
* a paired record computed against some arm other than :data:`CONTROL` is not
  the section 8.4 comparison and is rejected by label;
* when no arm carries paired evidence at all, the verdict is ``blocked`` --
  never ``no_winner``, because "nobody won" and "nothing was measured properly"
  are different findings;
* two eligible arms whose paired intervals OVERLAP are a tie, even when their
  point estimates differ;
* a DEGENERATE paired record -- every cell moved by exactly the same amount, so
  the resample is a point mass and ``ci_lo == ci_hi == point`` with ``se = 0`` --
  is not eligible either.  Its collapsed interval passes a naive ``ci_lo > 0``
  test more decisively than any real one, which is precisely why it must be
  named rather than read.

The point rules are retained as a **veto only**.  They were pre-registered
before these arms ran, so deleting them now would itself be a post-hoc edit; but
a veto can only ever withhold a promotion, and nothing here can promote without
an interval that excludes the null.  The flatness program's own rule -- where
``Delta75/SD`` is demoted to reporting entirely (section 8.5) -- lives in
:func:`lpopt.model.flat_ab.judge_all`, and :func:`promotion_allowed` requires the
two to agree before anything is swapped.

Ties and near-ties are NOT resolved silently.  When two arms are within
``AMBIGUITY_MARGIN`` on the primary metric, or their paired intervals overlap,
the verdict is ``escalate`` and the caller is told exactly which comparison is
too close to call -- picking one by an unregistered tiebreak would convert a
pre-registered test into a post-hoc one.

The A5 clause is reported explicitly whether or not A5 wins: A5 is the pure
capacity arm, pre-registered as the NULL hypothesis, and its result is the
answer to "does simply making the model bigger work?"

F_r DEFERRAL (user decision 2026-07-26)
---------------------------------------
:func:`evaluate_arms` is a PROMOTION surface, and it withholds promotion on an
F_r drop in TWO independent places:

* the PRIMARY rule's regression veto ranges over :data:`ab_eval.PRIMARY_TARGETS`,
  which contains ``f_r`` -- an F_r ``Delta75/SD`` regression fails
  ``passes_primary``;
* the SECONDARY rule ranges over ``("cyclen", "f_r")`` -- an F_r within-cell rho
  drop fails ``passes_secondary``.

Both were hardcoded, so the switch every other surface honours could not reach
them.  They now resolve through :func:`lpopt.config.fr_guard_enforced` -- THE one
switch, ``[curriculum] gate_noreg_fr_guard_enabled``.  ``f_r`` is still scored,
still reported (``worst_regression_any_axis``, per-arm notes, the ``fr_guard``
block), and still cannot promote anything on its own; it simply cannot WITHHOLD a
promotion while the guard is deferred.  ``cyclen`` keeps both of its vetoes: the
deferral is one axis wide.

There is a THIRD leg this module does not compute but does consume:
``passes_gate`` is read verbatim off ``folds[<fold>].gate``, the no-regression
gate :func:`lpopt.model.ab_score.score_arm` wrote.  That gate carries the setting
it was BUILT at, and a gate built at the other setting is a mismatched input, not
evidence — :func:`_gate_setting_mismatch` catches it, fails the leg, and names it
in the artifact (``fr_guard.gate_setting_mismatches``, rendered ``MIX``).  Without
that check a slate scored under one policy could be judged under another and the
artifact would show nothing at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._proc import no_window_flags
from . import ab_eval as M
from . import flat_ab as F

#: Improvement in Delta75/SD required on node_peak or map_CoV.
PRIMARY_IMPROVEMENT = 0.15
#: Any target may not worsen by more than this.
MAX_REGRESSION = 0.05
#: Two arms closer than this on the primary metric are a tie -> escalate.
AMBIGUITY_MARGIN = 0.05
#: The pure-capacity arm and its pre-registered null threshold.
CAPACITY_ARM = "A5"
CAPACITY_NULL_THRESHOLD = 0.02
#: Control and incumbent labels.
CONTROL = "B1"
INCUMBENT = "B0"
#: Arms whose cond_schema needs the serving power-prior wiring before promotion.
V6_SCHEMAS = ("v6", "v6_prior", "v6_contrast")
#: The paired metric whose CI must exclude the null before anything promotes.
PAIRED_PRIMARY = F.PRIMARY_METRICS[0]
#: Reported alongside it (section 8.5 condition 2).
PAIRED_SUPPORT = F.PRIMARY_METRICS[1]
#: Slice key the paired evidence is stored under (the refined C2, section 7.1).
PAIRED_FOLD = "C2"
#: The one axis the 2026-07-26 deferral demotes on this surface.
FR_TARGET = "f_r"
#: Secondary (within-cell rho) rule targets and its drop bound.
SECONDARY_TARGETS: tuple[str, ...] = ("cyclen", FR_TARGET)
SECONDARY_RHO_DROP = -0.02


def _res(arm: dict[str, Any], target: str, fold: str = "C") -> float | None:
    r = ((arm.get("folds") or {}).get(fold) or {}).get("resolution") or {}
    return (r.get(target) or {}).get("delta75_over_sd")


def _rho(arm: dict[str, Any], target: str, fold: str = "C") -> float | None:
    a = ((arm.get("folds") or {}).get(fold) or {}).get("accuracy") or {}
    return (a.get(target) or {}).get("median_rho")


def _gate(arm: dict[str, Any], fold: str = "C") -> dict[str, Any] | None:
    return ((arm.get("folds") or {}).get(fold) or {}).get("gate")


def _gate_setting_mismatch(gate: Mapping[str, Any], fr_enforced: bool) -> bool:
    """True when ``gate`` was computed at the OTHER F_r-guard setting.

    ``ab_score.score_arm`` stamps ``gate['fr_guard']['enforced']`` with the
    setting the gate was actually built at, so a judgement can tell a gate that
    agrees with it from one that does not.  A gate with NO ``fr_guard`` block
    predates the stamp: absence is not evidence of conflict, so it is read as
    agreeing (the same reading :func:`flat_ab.judge_arm` gives it).
    """
    fg = gate.get("fr_guard")
    if not isinstance(fg, Mapping) or "enforced" not in fg:
        return False
    return bool(fg["enforced"]) != bool(fr_enforced)


def _paired_block(arm: dict[str, Any], fold: str = "C") -> dict[str, Any] | None:
    """The :func:`lpopt.model.flat_ab.paired_block` record for this arm/fold.

    Falls back to the ``C2`` key because the paired evidence is produced on the
    refined C2 slice (section 7.1), not on raw fold C, while the point-statistic
    tables above are still fold-C.  The two live under their own keys rather than
    being merged, so a reader can always tell which slice a number came from.
    """
    paired = arm.get("paired") or {}
    b = paired.get(fold)
    if not isinstance(b, dict):
        b = paired.get(PAIRED_FOLD)
    return b if isinstance(b, dict) else None


def _paired_metric(arm: dict[str, Any], metric: str,
                   fold: str = "C") -> dict[str, Any] | None:
    b = _paired_block(arm, fold)
    if not b:
        return None
    m = (b.get("metrics") or {}).get(metric)
    return m if isinstance(m, dict) else None


@dataclass
class ArmVerdict:
    label: str
    primary_gain: float | None = None          # best improvement over control
    worst_regression: float | None = None      # ENFORCED axes only
    #: Worst regression over EVERY scored axis, including report-only ones.  The
    #: enforced number is what decides; this one is what a reader needs in order
    #: to know a deferred axis moved at all.
    worst_regression_any_axis: float | None = None
    passes_primary: bool = False
    passes_gate: bool = False
    #: The fold gate was computed at the OTHER F_r-guard setting, so it is a
    #: mismatched input rather than evidence either way.  Reported, never
    #: silently absorbed into ``passes_gate``.
    gate_fr_mismatch: bool = False
    passes_secondary: bool = False
    passes_paired: bool = False
    paired_point: float | None = None
    paired_lo: float | None = None
    paired_hi: float | None = None
    paired_method: str | None = None
    paired_control: str | None = None
    eligible: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def intervals_overlap(self, other: "ArmVerdict") -> bool:
        """Two paired intervals that overlap cannot be ordered by this rule."""
        if None in (self.paired_lo, self.paired_hi,
                    other.paired_lo, other.paired_hi):
            return True
        return not (self.paired_lo > other.paired_hi
                    or other.paired_lo > self.paired_hi)


def _read_paired(v: ArmVerdict, arm: dict[str, Any], fold: str) -> None:
    """Fill ``v``'s paired fields, or record exactly why there are none.

    Every branch that leaves ``passes_paired`` false also leaves a note: an arm
    silently missing its interval would otherwise be indistinguishable from an
    arm whose interval straddled the null, and those call for different actions
    (score it, versus stop).
    """
    block = _paired_block(arm, fold)
    if block is None:
        v.notes.append(f"no paired inference on fold {fold} (section 8.3): "
                       "promotion on point estimates alone is not permitted")
        return
    v.paired_control = block.get("control")
    if v.paired_control != CONTROL:
        v.notes.append(
            f"paired evidence was computed against {v.paired_control!r}, not the "
            f"section 8.4 control {CONTROL!r} -- that comparison confounds the "
            "loss change with the training-set change and cannot promote")
        return
    m = _paired_metric(arm, PAIRED_PRIMARY, fold)
    if m is None:
        v.notes.append(f"paired block carries no {PAIRED_PRIMARY}")
        return
    v.paired_point = m.get("point")
    v.paired_lo = m.get("ci_lo")
    v.paired_hi = m.get("ci_hi")
    v.paired_method = m.get("method")
    if v.paired_method == "insufficient":
        v.notes.append(f"{PAIRED_PRIMARY}: too few paired cells to form an "
                       "interval; nothing is established either way")
        return
    if v.paired_method == "degenerate":
        # ci_lo == ci_hi == point with se = 0.  Arithmetically that is the most
        # decisive interval expressible, and the ``ci_lo > 0`` test below could
        # not tell it apart from one -- so a resample that saw a single distinct
        # value would have promoted.  It establishes nothing (ab_paired.
        # NO_EVIDENCE_METHODS) and is reported as such.
        v.notes.append(
            f"{PAIRED_PRIMARY}: every paired cell moved by exactly the same "
            f"amount ({v.paired_point}), so the resampling distribution is a "
            "point mass; the zero-width interval expresses no uncertainty and "
            "cannot promote")
        return
    v.passes_paired = bool(v.paired_lo is not None and v.paired_lo > 0.0)
    if not v.passes_paired:
        if v.paired_point is not None and v.paired_point > 0:
            v.notes.append(
                f"{PAIRED_PRIMARY}: point estimate {v.paired_point:+.4f} favours "
                f"the arm but the paired CI [{v.paired_lo}, {v.paired_hi}] does "
                "not exclude the null")
        else:
            v.notes.append(f"{PAIRED_PRIMARY}: paired CI does not establish a gain")


def evaluate_arms(doc: dict[str, Any], *, fold: str = "C",
                  fr_guarded: bool | None = None) -> dict[str, Any]:
    """Score every arm against the pre-registered rule.  Pure function.

    ``fr_guarded`` resolves through :func:`lpopt.config.fr_guard_enforced` — THE
    one switch (:data:`lpopt.config.FR_GUARD_KNOB`).  Deferred (the default), the
    two F_r vetoes above are scored and REPORT-ONLY; armed, they bite again.  Pure
    either way: pass the value in, never read a file from here.
    """
    from ..config import FR_GUARD_KNOB, fr_guard_enforced

    fr_enforced = fr_guard_enforced(fr_guarded)
    #: Axes that may be scored but may NOT withhold a promotion this run.
    report_only: tuple[str, ...] = () if fr_enforced else (FR_TARGET,)
    arms = doc.get("arms") or {}
    control = arms.get(CONTROL)
    out: dict[str, Any] = {
        "fold": fold, "control": CONTROL, "incumbent": INCUMBENT,
        "n_arms_scored": len(arms), "verdicts": {}, "missing": [],
        "paired_primary": PAIRED_PRIMARY,
        "rule_note": ("Delta75/SD is a VETO only; promotion additionally requires "
                      "a paired cell-clustered CI on arm - control that excludes "
                      "the null (sections 8.3/8.5)."),
        "fr_guard": {
            "target": FR_TARGET, "enforced": bool(fr_enforced),
            "knob": FR_GUARD_KNOB,
            "report_only_targets": list(report_only),
            "vetoes": ["primary Delta75/SD regression", "secondary within-cell rho"],
            "note": ("f_r is SCORED and REPORTED on both point rules but cannot "
                     "withhold a promotion; a verdict here does NOT mean f_r was "
                     f"verified regression-free.  Set {FR_GUARD_KNOB} = true to "
                     "enforce." if not fr_enforced else
                     "f_r vetoes are ARMED on both point rules."),
        },
    }
    expected = [CONTROL, "A1", "A2", "A3", "A4", "A5", "A6"]
    out["missing"] = [a for a in expected if a not in arms]
    if control is None:
        # Section 8.4: no control, no verdict.  Not "no winner" -- there is
        # nothing to be a winner against.
        out["verdict"] = "blocked"
        out["reason"] = (f"control {CONTROL} not scored yet; without it every "
                         "comparison confounds the loss change with the "
                         "training-set change (section 8.4)")
        return out

    verdicts: dict[str, ArmVerdict] = {}
    for label, arm in arms.items():
        if label in (CONTROL, INCUMBENT):
            continue
        v = ArmVerdict(label=label)
        gains, regressions, regressions_any = [], [], []
        for tgt in M.PRIMARY_TARGETS:
            a, b = _res(arm, tgt, fold), _res(control, tgt, fold)
            if a is None or b is None:
                v.notes.append(f"{tgt}: Delta75 not reached by one of the pair")
                continue
            delta = b - a                       # positive == arm is FINER
            if tgt in ("node_peak", "map_cov"):
                gains.append(delta)
            if delta < 0:
                regressions_any.append(-delta)
                # VETO 1, on the switch.  A deferred axis is still measured and
                # still named below; it just does not enter ``worst_regression``,
                # which is the number ``passes_primary`` reads.
                if tgt in report_only:
                    v.notes.append(
                        f"{tgt}: Delta75/SD regression {-delta:+.3f} — "
                        "REPORT-ONLY (scored, NOT enforced), it cannot fail the "
                        f"primary rule.  Set {FR_GUARD_KNOB} = true to enforce.")
                else:
                    regressions.append(-delta)
        v.primary_gain = max(gains) if gains else None
        v.worst_regression = max(regressions) if regressions else 0.0
        v.worst_regression_any_axis = (max(regressions_any)
                                       if regressions_any else 0.0)
        v.passes_primary = bool(
            v.primary_gain is not None
            and v.primary_gain >= PRIMARY_IMPROVEMENT
            and (v.worst_regression or 0.0) <= MAX_REGRESSION)

        g = _gate(arm, fold)
        v.passes_gate = bool(g and g.get("pass"))
        if g is None:
            v.notes.append("no gate result on this fold")
        elif _gate_setting_mismatch(g, fr_enforced):
            # SPLIT-BRAIN.  ``flat_ab.judge_arm`` already refuses to mix a gate
            # computed at one F_r setting with a judgement made at the other;
            # this leg consumed ``gate['pass']`` verbatim, so a stale-setting
            # gate would be laundered into ``passes_gate`` with nothing in the
            # artifact to show it.  A mismatched input is not evidence of a
            # pass, and saying so beats silently mixing the two.
            v.passes_gate = False
            v.gate_fr_mismatch = True
            gate_was = "ENFORCED" if not fr_enforced else "DEFERRED"
            judged_at = "DEFERRED" if not fr_enforced else "ENFORCED"
            v.notes.append(
                f"gate on fold {fold} was computed with the F_r guard "
                f"{gate_was} while this judgement runs with it {judged_at} "
                f"({FR_GUARD_KNOB}); re-score the arm at one setting — a "
                "mismatched gate cannot support a promotion")

        inc = arms.get(INCUMBENT)
        sec_ok = True
        if inc is not None:
            for tgt in SECONDARY_TARGETS:
                new, old = _rho(arm, tgt, fold), _rho(inc, tgt, fold)
                if new is None or old is None:
                    continue
                if new - old < SECONDARY_RHO_DROP:
                    # VETO 2, on the same switch.
                    if tgt in report_only:
                        v.notes.append(
                            f"{tgt} within-cell rho {new:.3f} vs incumbent "
                            f"{old:.3f} — REPORT-ONLY (scored, NOT enforced), it "
                            "cannot fail the secondary rule.  Set "
                            f"{FR_GUARD_KNOB} = true to enforce.")
                        continue
                    sec_ok = False
                    v.notes.append(
                        f"{tgt} within-cell rho {new:.3f} vs incumbent {old:.3f}")
        v.passes_secondary = sec_ok
        _read_paired(v, arm, fold)
        v.eligible = (v.passes_primary and v.passes_gate and v.passes_secondary
                      and v.passes_paired)
        verdicts[label] = v

    out["verdicts"] = {k: v.to_dict() for k, v in verdicts.items()}
    out["n_with_paired"] = sum(1 for v in verdicts.values()
                               if v.paired_method is not None)
    # A split-brain is a HEADLINE, not a per-arm footnote: it means the slate was
    # scored under one policy and judged under another.
    mism = sorted(k for k, v in verdicts.items() if v.gate_fr_mismatch)
    out["fr_guard"]["gate_setting_mismatches"] = mism
    if mism:
        out["fr_guard"]["gate_setting_mismatch_note"] = (
            f"the fold-{fold} no-regression gate of {', '.join(mism)} was "
            f"computed at the OTHER {FR_GUARD_KNOB} setting; those arms cannot "
            "pass this leg until they are re-scored "
            "(python -m lpopt.model.ab_score --arm ... --deck <deck>)")

    # --- no interval evidence anywhere: measured nothing, decided nothing --- #
    if verdicts and not any(_paired_block(arms[k], fold) for k in verdicts):
        out["verdict"] = "blocked"
        out["reason"] = (
            "no arm carries a paired cell-clustered CI against the control on "
            f"fold {fold}.  Section 8.3 forbids promoting on point estimates, and "
            "section 8.1 measured why: Delta75/SD takes six distinct values over "
            "the whole slate, so a point lead is routinely a binning artifact.  "
            "Run the flatness A/B apparatus (lpopt.model.flat_ab) and record its "
            "paired_block() output under arms[<label>]['paired'][fold].")
        out["falsification"] = dict(F.FALSIFICATION)
        return out

    # --- the capacity clause, reported regardless of who wins -------------- #
    cap = verdicts.get(CAPACITY_ARM)
    if cap is not None and cap.primary_gain is not None:
        beat = cap.primary_gain > CAPACITY_NULL_THRESHOLD
        out["capacity_clause"] = {
            "arm": CAPACITY_ARM, "primary_gain": cap.primary_gain,
            "threshold": CAPACITY_NULL_THRESHOLD,
            "null_rejected": bool(beat),
            "conclusion": (
                "순수 용량 확대가 실효 분해능을 개선했다 — 사전등록 귀무가설 기각"
                if beat else
                "순수 용량 확대는 실효 분해능을 개선하지 못했다 — "
                "\"단순 폭 확대 무효\"가 세 번째로 확정됨"),
        }

    # --- winner ------------------------------------------------------------ #
    eligible = [v for v in verdicts.values() if v.eligible]
    if out["missing"]:
        out["verdict"] = "incomplete"
        out["reason"] = f"not scored yet: {', '.join(out['missing'])}"
        out["leader"] = (max(eligible, key=lambda v: v.primary_gain).label
                         if eligible else None)
        return out
    if not eligible:
        near = sorted((v for v in verdicts.values()
                       if v.paired_point is not None and v.paired_point > 0
                       and not v.passes_paired),
                      key=lambda v: -(v.paired_point or 0.0))
        if near:
            # The exact trap this apparatus exists for: a point lead with an
            # interval that includes the null.  Escalating (rather than promoting
            # the leader, or silently reporting "no winner") is the pre-registered
            # response -- the finding is "underpowered", not "no effect".
            out["verdict"] = "escalate"
            out["candidates"] = [v.label for v in near[:2]]
            out["reason"] = (
                f"{near[0].label}의 {PAIRED_PRIMARY} 점추정은 "
                f"{near[0].paired_point:+.4f}로 arm에 유리하나 paired CI "
                f"[{near[0].paired_lo}, {near[0].paired_hi}]가 귀무를 배제하지 "
                "못한다 — 사전등록 규칙(§8.5)은 점추정 승격을 금지하며, 이는 "
                "'효과 없음'이 아니라 '검출력 부족'이다. 라벨을 더 쓸지 arm을 "
                "접을지는 사람이 결정한다.")
            return out
        out["verdict"] = "no_winner"
        out["reason"] = ("사전등록 승격 조건을 만족한 arm이 없다 "
                         "(1차 ≥0.15 개선 + 게이트 PASS + 2차 무하락 + "
                         f"{PAIRED_PRIMARY} paired CI 하한 > 0)")
        return out

    ranked = sorted(eligible, key=lambda v: -(v.paired_point if
                                              v.paired_point is not None
                                              else (v.primary_gain or 0.0)))
    best = ranked[0]
    if len(ranked) > 1 and (
            ranked[0].intervals_overlap(ranked[1])
            or abs((ranked[0].primary_gain or 0) -
                   (ranked[1].primary_gain or 0)) < AMBIGUITY_MARGIN):
        out["verdict"] = "escalate"
        out["reason"] = (
            f"{ranked[0].label}(paired {ranked[0].paired_point}, CI "
            f"[{ranked[0].paired_lo}, {ranked[0].paired_hi}])와 "
            f"{ranked[1].label}(paired {ranked[1].paired_point}, CI "
            f"[{ranked[1].paired_lo}, {ranked[1].paired_hi}])를 사전등록 규칙으로 "
            "가를 수 없다 — paired 구간이 겹치거나 1차 지표 차이가 "
            f"{AMBIGUITY_MARGIN} 미만이다. 사후 타이브레이크는 사전등록을 "
            "무효화하므로 사람 판단이 필요하다.")
        out["candidates"] = [r.label for r in ranked[:2]]
        return out
    out["verdict"] = "winner"
    out["winner"] = best.label
    out["winner_dir"] = arms[best.label].get("model_dir")
    out["winner_schema"] = arms[best.label].get("cond_schema")
    out["reason"] = (f"{best.label}: {PAIRED_PRIMARY} paired CI 하한 "
                     f"{best.paired_lo} > 0 (대조군 {CONTROL} 기준, "
                     f"{best.paired_method}), 1차 Δ₇₅/SD 거부권 통과 "
                     f"{best.primary_gain:.3f} (≥{PRIMARY_IMPROVEMENT}), 최대 악화 "
                     f"{best.worst_regression:.3f} (≤{MAX_REGRESSION}), 게이트 PASS")
    out["falsification"] = dict(F.FALSIFICATION)
    if str(out["winner_schema"]) in V6_SCHEMAS:
        out["serving_blocker"] = (
            "승자가 cond_v6 계열이다. 승격 전에 model_api.py가 "
            "meta['power_prior']의 (M², extrap)로 서빙 인코더를 재구성하도록 "
            "고쳐야 학습/서빙이 일치한다 (설계서 §11.4).")
    return out


def promote(winner_dir: str, *, prev_dir: str, deck: str = "lpopt.inp",
            out_json: str = "gate_hires_ab.json",
            dry_run: bool = False) -> dict[str, Any]:
    """Run the committed ``lpopt gate-promote`` path (both gates + atomic swap).

    Deliberately shells out to the existing, tested command rather than
    re-implementing promotion: that command already runs the honest per-cell
    no-regression gate AND the legacy high-cyclen tail gate, and only swaps the
    deck + curriculum pointers when both pass.
    """
    # WARNING: ``gate-promote`` takes NO --promote flag.  It swaps the deck's
    # [model].model_dir and the curriculum state's champion_model_dir
    # UNCONDITIONALLY as soon as both gates pass.  There is no read-only
    # invocation -- NOT CALLING IT is the only dry run.  (Learned the hard way on
    # 2026-07-25: a "just check the gate" invocation promoted A6 on the spot.)
    cmd = [sys.executable, "-m", "lpopt.cli", "gate-promote",
           "--input", deck, "--prev", prev_dir, "--new", winner_dir,
           "--out", out_json]
    if dry_run:
        return {"cmd": " ".join(cmd), "returncode": None, "dry_run": True,
                "stdout": "(not executed: gate-promote promotes on PASS)",
                "stderr": "", "gate_json": out_json}
    # **no_window_flags(): repo convention -- no console window pops up on
    # Windows when a long-running promotion is driven from a background watcher.
    proc = subprocess.run(cmd, capture_output=True, text=True, **no_window_flags())
    return {"cmd": " ".join(cmd), "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:],
            "gate_json": out_json}


def promotion_allowed(v: dict[str, Any], doc: dict[str, Any] | None = None
                      ) -> tuple[bool, str]:
    """May ``--promote`` fire?  Both pre-registrations must agree.

    The legacy hires rule (this module) and the flatness rule
    (:func:`lpopt.model.flat_ab.judge_all`, stored in the results document under
    ``flat_slate``) are different pre-registrations answering different
    questions.  Where both exist, a disagreement is exactly the situation a human
    has to look at: promoting on whichever one happens to say yes is
    post-hoc selection between rules.
    """
    if v.get("verdict") != "winner":
        return False, f"verdict is {v.get('verdict')!r}, not 'winner'"
    if v.get("serving_blocker"):
        return False, str(v["serving_blocker"])
    slate = (doc or {}).get("flat_slate")
    if isinstance(slate, dict):
        if slate.get("verdict") != F.PROMOTE:
            return False, (
                f"the flatness apparatus returned {slate.get('verdict')!r} "
                f"({slate.get('reason')}) -- the two pre-registered rules "
                "disagree, which is a decision for a human, not for whichever "
                "rule happens to say yes")
        if slate.get("winner") and slate.get("winner") != v.get("winner"):
            return False, (
                f"the flatness apparatus promotes {slate.get('winner')!r} but "
                f"this rule promotes {v.get('winner')!r}")
    return True, "both pre-registered rules promote the same arm"


def render_verdict(v: dict[str, Any]) -> str:
    L = [f"verdict: {v.get('verdict')}"]
    if v.get("reason"):
        L.append(f"reason : {v['reason']}")
    if v.get("missing"):
        L.append(f"missing: {', '.join(v['missing'])}")
    L.append("")
    L.append(f"{'arm':4s} {'1차 개선':>9s} {'최대악화':>9s} {'1차':>4s} {'게이트':>6s} "
             f"{'2차':>4s} {'paired 하한':>12s} {'paired':>7s} {'승격가능':>8s}")
    for label in sorted(v.get("verdicts", {})):
        d = v["verdicts"][label]
        g = d.get("primary_gain")
        w = d.get("worst_regression")
        lo = d.get("paired_lo")
        L.append(f"{label:4s} {('—' if g is None else f'{g:+.3f}'):>9s} "
                 f"{('—' if w is None else f'{w:.3f}'):>9s} "
                 f"{'O' if d['passes_primary'] else 'X':>4s} "
                 # a MISMATCHED gate is not a failed gate: the first says the
                 # evidence was never gathered at this setting, the second says
                 # it was gathered and the arm lost.
                 f"{('MIX' if d.get('gate_fr_mismatch') else
                     ('PASS' if d['passes_gate'] else 'FAIL')):>6s} "
                 f"{'O' if d['passes_secondary'] else 'X':>4s} "
                 f"{('—' if lo is None else f'{lo:+.4f}'):>12s} "
                 f"{'O' if d.get('passes_paired') else 'X':>7s} "
                 f"{'YES' if d['eligible'] else 'no':>8s}")
    if v.get("verdicts"):
        L.append("")
        L.append(f"paired 1차 지표: {v.get('paired_primary')} "
                 f"(대조군 {v.get('control')}, 셀 클러스터 BCa; "
                 "하한 > 0 이 아니면 승격 불가)")
    fg = v.get("fr_guard")
    if isinstance(fg, dict):
        L.append("")
        L.append(f"F_r 거부권: {'ENFORCED' if fg.get('enforced') else 'DEFERRED'} "
                 f"({fg.get('knob')}) — {fg.get('note')}")
        if fg.get("gate_setting_mismatch_note"):
            L.append(f"게이트 설정 불일치(MIX): {fg['gate_setting_mismatch_note']}")
    if v.get("capacity_clause"):
        c = v["capacity_clause"]
        L.append("")
        L.append(f"A5 용량 조항: 개선 {c['primary_gain']:+.3f} "
                 f"(임계 {c['threshold']}) -> {c['conclusion']}")
    if v.get("serving_blocker"):
        L.append("")
        L.append(f"** 승격 차단 항목: {v['serving_blocker']}")
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lpopt.model.ab_decide")
    ap.add_argument("--results", default=M.DEFAULT_RESULTS)
    ap.add_argument("--fold", default="C")
    ap.add_argument("--out", default="data/reports/hires_ab_verdict.json")
    ap.add_argument("--promote", action="store_true",
                    help="run lpopt gate-promote on the winner (only fires on a "
                         "clean 'winner' verdict with no serving blocker)")
    ap.add_argument("--deck", default="lpopt.inp",
                    help="deck the F_r-guard switch is resolved from (and the "
                         "deck gate-promote swaps on --promote)")
    ap.add_argument("--fr-guarded", dest="fr_guarded",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="override the deck's [curriculum] "
                         "gate_noreg_fr_guard_enabled for this run")
    ap.add_argument("--prev", default=M.DEFAULT_CHAMPION)
    args = ap.parse_args(argv)

    p = Path(args.results)
    if not p.exists():
        print(f"no results yet at {args.results}")
        return 1
    doc = json.loads(p.read_text(encoding="utf-8"))
    # THE one switch: the same deck field every in-loop gate reads, resolved here
    # rather than defaulted, so flipping it re-arms this surface too.
    from ..config import fr_guard_from_deck
    policy = fr_guard_from_deck(args.deck, fr_guarded=args.fr_guarded)
    v = evaluate_arms(doc, fold=args.fold, fr_guarded=policy["enforced"])
    v["fr_guard_policy"] = policy
    print(render_verdict(v))
    if isinstance(doc.get("flat_slate"), dict):
        print("")
        print(F.render_slate(doc["flat_slate"]))
        v["flat_slate_verdict"] = doc["flat_slate"].get("verdict")

    ok, why = promotion_allowed(v, doc)
    v["promotion_allowed"] = ok
    v["promotion_reason"] = why
    if args.promote:
        if not ok:
            print(f"\nNOT promoting: {why}")
        else:
            print(f"\nrunning gate-promote for {v['winner']} ...")
            v["promotion"] = promote(v["winner_dir"], prev_dir=args.prev,
                                     deck=args.deck)
            print(v["promotion"]["stdout"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(v, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
