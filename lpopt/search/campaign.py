"""Guided-search campaign driver (``lpopt optimize``) — plan sec. 4.6 / 7.

:func:`run_campaign` runs the wave loop: budget ``100 = 12 waves × 8 + 4 reserve``.
Per wave it

1. builds a candidate pool (:func:`lpopt.search.construct.build_pool`),
2. surrogate-scores it and applies the trust-region gate
   (:mod:`lpopt.search.acquisition`),
3. refines the top candidates by local search,
4. composes an 8-slot wave (5 exploit / 2 explore / 1 control),
5. verifies it with :class:`~lpopt.search.verify.WaveVerifier`
   (``StubEvaluator`` under ``--dry-run``; live MASTER is a flag-flip),
6. appends store rows (``dataset="P"``, ``campaign=run id``) + the append-only
   campaign ledger, archives feasible verified LPs (vendor ``archive_candidate``),
7. fine-tunes + gates the model online (:mod:`lpopt.search.update`),
8. writes the wave report artefacts and a checkpoint, and
9. checks the stopping rule.

State is fully resumable: ``state.json`` (wave index, remaining budget, RNG
state, champion checkpoint, halt counter) + an append-only ``labels.jsonl`` under
``runs/<ts>/`` reconstruct the verified set with no budget double-spend.  Two
consecutive gate ``halt`` verdicts trigger the ``MODEL_HALT`` path (sklearn
fallback refit, then stop with status).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _replace
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..config import LpoptConfig
from ..safelog import safe_logger, safe_print
from ..data import flat_scale as _FS
from ..data.flat_scale import FlatScale
from ..data.map_calibration import (
    MapCalibration, ModelMismatchError, model_fingerprint)
from ..data.schema import CanonicalRecord, unpack_pattern
from ..data.store import StoreReader, StoreWriter
from ..vendor.masterrl.domain import CaseKey, Pattern
from ..vendor.masterrl.ga import GAEvaluation, archive_candidate
from ..vendor.masterrl.reward import is_fom_feasible
from ..model.cell_calibrate import CampaignBiasCorrector, cyclen_cell_key
from .delivery import (
    LICENSING_FR_LIMIT, LICENSING_FXY_LIMIT, compliance_margin,
    compliance_margin_fxy, select_delivery)
from .assets import CaseAssetResolver
from .construct import (
    Candidate, CaseContext, PairCell, build_pair_universe, build_pool,
    candidate_record_id, e_core_in_band, predicted_e_core,
)
from .construct import _parent_to_genome as _pattern_to_case_genome
from .genome import GenomeError, mutate
from .verify import (
    PRODUCE_DECK_KNOBS, WaveEntry, WaveVerifier, lineage_anchor, outcome_to_record)
from . import acquisition as acq
from . import rule_metrics as acq_rules
from .update import WaveUpdater

STATE_NAME = "state.json"
LABELS_NAME = "labels.jsonl"

#: Objectives the flatness-first program RETIRED as production modes (§10 STOP).
#: They still run — reproducing an existing result and the A/B baselines need
#: them — but constructing one logs a DEPRECATED banner so no fleet drifts back
#: onto an F_r-steered search without someone having read it.
_RETIRED_PRODUCTION_OBJECTIVES = ("max_cycle_min_fr", "min_fr_max_cycle")

#: Name :meth:`CampaignDriver._save_champion` writes a wave champion under.  Kept
#: next to it so the two never drift: :meth:`CampaignDriver._is_wave_champion`
#: uses it to recognise THIS run's own fine-tuned descendants.
_WAVE_CHAMPION_RE = re.compile(r"champion_wave_\d+")


class FxySigmaBarLost(RuntimeError):
    """A checkpoint's f_xy serve-sigma bar (G4) did not survive a save/reload.

    The bar is stamped on the artifact (``ensemble.json`` ->
    ``fxy_head.serve_sigma = "barred"``) and decides the WIDTH every ``min_fxy``
    UCB is built from.  Losing it does not crash anything and does not look
    wrong in any artifact — it just silently swaps the served sigma for the
    over-wide one the G4 verdict refused (defect D3,
    ``data/reports/minfxy_T6T4_f121_r1_results_20260830.md`` §9).  So it is
    raised, loudly, rather than warned.
    """


def checkpoint_fxy_serve_sigma(ckpt: str | Path) -> str:
    """The f_xy serve-sigma a checkpoint DIRECTORY declares on disk (``""`` = none).

    Deliberately reads the files rather than a loaded backend: the question here
    is what a future ``--resume`` will find, and only the bytes answer that.
    Mirrors :meth:`..model.model_api.PosValCnnBackend.from_dir`'s resolution —
    ``ensemble.json`` first, then any member meta (any member asserting the bar
    bars the ensemble).
    """
    d = Path(str(ckpt))
    ens = d / "ensemble.json"
    if ens.is_file():
        try:
            meta = json.loads(ens.read_text(encoding="utf-8"))
            v = (meta.get("fxy_head") or {}).get("serve_sigma")
            if v:
                return str(v).strip().lower()
        except (OSError, ValueError, AttributeError):
            pass
    for md in sorted(d.glob("member_*")):
        try:
            meta = json.loads((md / "meta.json").read_text(encoding="utf-8"))
            v = (meta.get("fxy_head") or {}).get("serve_sigma")
        except (OSError, ValueError, AttributeError):
            continue
        if v and str(v).strip().lower() == "barred":
            return "barred"
    return ""


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #
@dataclass
class WaveReport:
    wave: int
    reserve: bool
    size: int
    budget_spent: int
    slots: dict[str, int]
    converged: int
    feasible: int
    on_target: int
    best_objective: float | None
    best_cyclen: float | None
    gate_mode: str
    gate_accepted: bool
    tau: float
    #: converged rows carrying a flatness label / converged rows (program §1.3).
    #: ``None`` when nothing converged.  Recorded for EVERY objective (it is free
    #: and it is the metric that would have caught a silent harvest outage), but
    #: only ``flat_power`` aborts on it.
    map_harvest: float | None = None
    #: outcomes whose status was ``error`` this wave.  Defaulted so an older
    #: state.json (written before this field existed) still loads.  Without it a
    #: wave where every chain died at staging is indistinguishable from a wave
    #: that merely failed to converge — both report conv=0 (ECC audit 2026-08-12).
    errors: int = 0
    #: SAFETY SHIELD accounting (review §6.5 item 5 — report the gates separately).
    #: ``ood_flagged`` is what the guard SAW, ``ood_escalated`` what lost its
    #: exploit tier, ``ood_rejected`` / ``conformal_rejected`` what each gate
    #: REMOVED from the pool.  All 0 at the shipped defaults, and defaulted so an
    #: older ``state.json`` still loads.
    ood_flagged: int = 0
    ood_escalated: int = 0
    ood_rejected: int = 0
    conformal_rejected: int = 0


class MapHarvestAbort(RuntimeError):
    """A ``flat_power`` wave's map harvest fell below the configured floor.

    Program §1.3: the flatness objective is defined on the harvested map columns,
    so a harvest outage makes the campaign unable to score what it evaluates —
    and its only other symptom would be a stalled improvement counter, i.e. it
    would masquerade as convergence.  Raised by :meth:`CampaignDriver._check_map_harvest`
    and turned into a ``map_harvest_abort`` result by :meth:`CampaignDriver.run`.
    """


@dataclass
class CampaignResult:
    run_dir: str
    status: str
    waves: int
    budget: int
    budget_spent: int
    n_feasible: int
    on_target: int
    best: dict[str, Any] | None
    best_overall: dict[str, Any] | None = None
    wave_reports: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    #: per-wave map harvest rate (program §1.3) — the diagnostic that tells a
    #: harvest outage apart from a converged search.
    map_harvest_rates: list[float] = field(default_factory=list)
    #: MASTER calls spent by the D9 SDM/MTC pre-delivery gate, kept OUT of
    #: ``budget_spent`` (which is the search budget) and reported alongside it.
    #: ``budget_spent + post_verify_master_calls`` is the run's true MASTER cost.
    post_verify_master_calls: int = 0
    #: record_ids the gate marked as violating a user-set SDM/MTC limit.
    post_verify_violators: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small IO helpers
# --------------------------------------------------------------------------- #
def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()


def _opt_float(value: Any) -> float | None:
    """``float(value)`` or ``None`` for an absent / zero / non-finite knob.

    The deck spells "this optional limit is off" as an absent key OR as ``0.0``
    (the convention ``flatpower_peak_scale`` already uses), and both must reach
    the spec as ``None`` rather than as a limit of zero that vetoes everything.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out == 0.0:
        return None
    return out


def _objective(cyclen: float | None, target: float) -> float:
    """Campaign objective for a feasible label: higher is better (−|cyclen−target|)."""

    if cyclen is None:
        return float("-inf")
    return -abs(float(cyclen) - float(target))


# --------------------------------------------------------------------------- #
# constraint feasibility — ONE definition, shared with the report
# --------------------------------------------------------------------------- #
#: Keys of a feasibility-limit mapping.  A missing key, or an explicit ``None``,
#: means that axis is UNGATED for the objective (reported, never a rejection).
FEASIBILITY_LIMIT_KEYS: tuple[str, ...] = (
    "f_r", "cbc_max", "f_q", "ao_abs", "max_pin_burnup", "cyclen_lo", "cyclen_hi",
    "f_xy",
)

#: The LICENSING pin-burnup limit :func:`is_deliverable` judges against — the
#: LEU+ 80 GWd/tU, deliberately NOT the 78.0 the search-time gates use.  That 2.0
#: is MODEL margin on a PREDICTION; a MEASURED 79.4 is deliverable and a search
#: gate must not be re-used as a licensing verdict.
DELIVERABLE_PIN_BU_LIMIT = 80.0


def feasibility_limits_for(acq_cfg: Any, objective: str, *,
                           fr_gate: float | None = None) -> dict[str, float | None]:
    """The limits ``objective`` judges feasibility at (``None`` = that axis is ungated).

    Pure function of the ``[acquisition]`` block, so ``lpopt report`` on a finished
    run resolves the SAME feasible set from the deck that the run applied.
    ``max_cycle_min_fr`` / ``fr_boundary`` make F_r a pure OBJECTIVE with no gate;
    ``flat_power`` gates it at its own safety limit (decision D1) — pass the live
    ``fr_gate`` when a per-cell map-head bias correction has been applied, else the
    deck value HOLDS.
    """
    limits: dict[str, float | None] = {
        "cbc_max": float(acq_cfg.cbc_limit),
        "f_q": float(acq_cfg.f_q_limit),
        "ao_abs": float(acq_cfg.ao_abs_limit),
        "f_r": None,
        "max_pin_burnup": None,
        "cyclen_lo": None,
        "cyclen_hi": None,
        # F_xy (MASTER FXYP) gates only where the objective screens it: the
        # min_fxy mode it IS the objective of, and flat_power's safety gate.
        # Everywhere else it is a REPORTED column, never a rejection.
        "f_xy": None,
    }
    if objective not in ("max_cycle_min_fr", "fr_boundary", "flat_power"):
        limits["f_r"] = float(acq_cfg.f_r_limit)
    if objective == "min_fuel_cost":
        # cyclen band (both edges) + pin BU are HARD constraints too (the
        # six-constraint feasibility set).
        limits["cyclen_lo"] = float(getattr(acq_cfg, "fuelcost_cyclen_lo", 615.0))
        limits["cyclen_hi"] = float(getattr(acq_cfg, "fuelcost_cyclen_hi", 635.0))
        limits["max_pin_burnup"] = float(
            getattr(acq_cfg, "fuelcost_pin_bu_limit", 80.0))
    elif objective == "min_fr_max_cycle":
        # Pin BU is a HARD constraint here too (added 2026-08-17).  It gates at
        # ``minfr_pin_bu_limit`` (default 78.0, a model-margin haircut off the LEU+
        # 80 the other modes use — see config).  Until this landed, min_fr was the
        # ONE pin-unscreened objective and both of its campaigns reported cores
        # over the limit as feasible.
        limits["max_pin_burnup"] = float(
            getattr(acq_cfg, "minfr_pin_bu_limit", 78.0))
    elif objective == "min_fxy":
        # F_xy is BOTH the objective and a hard limit (user decision 2026-08-29),
        # and F_r STAYS a constraint at f_r_limit (set above) — design §3.5.2:
        # the two axes disagree on real cores, so neither implies the other.
        # Pin BU gates at the same model-margin haircut min_fr uses.
        limits["f_xy"] = float(getattr(acq_cfg, "f_xy_limit", 1.65))
        limits["max_pin_burnup"] = float(
            getattr(acq_cfg, "minfxy_pin_bu_limit", 78.0))
        lo = getattr(acq_cfg, "minfxy_cyclen_lo", None)
        hi = getattr(acq_cfg, "minfxy_cyclen_hi", None)
        limits["cyclen_lo"] = None if lo is None else float(lo)
        limits["cyclen_hi"] = None if hi is None else float(hi)
    elif objective == "fr_boundary":
        # NO F_r gate, NO cyclen band (cyclen recorded but never gated).
        limits["max_pin_burnup"] = float(
            getattr(acq_cfg, "fr_boundary_pin_bu_limit", 80.0))
    elif objective == "flat_power":
        # F_r SAFETY GATE (decision D1) — the ONE place F_r still screens anything
        # in this mode.  It is kept (a flat pattern that violates 1.70 is not a
        # usable result), but it is a gate: it never orders two passing rows, and
        # it is not the objective.
        limits["f_r"] = (float(fr_gate) if fr_gate is not None
                         else float(getattr(acq_cfg, "flatpower_fr_limit", 1.7)))
        limits["max_pin_burnup"] = float(
            getattr(acq_cfg, "flatpower_pin_bu_limit", 80.0))
        # F_xy SAFETY GATE (design §3.5.3) — node_peak is NOT F_xy (corr 0.74-0.85),
        # so a flat pattern carries no F_xy guarantee.  0 / non-finite disables it.
        fxy_gate = float(getattr(acq_cfg, "flatpower_fxy_limit", 0.0) or 0.0)
        limits["f_xy"] = fxy_gate if math.isfinite(fxy_gate) and fxy_gate > 0.0 else None
    return limits


def _is_missing(value: Any) -> bool:
    """``True`` when ``value`` carries NO measurement — i.e. ``None`` **or** ``NaN``.

    **NaN == missing == None is the DELIBERATE contract (decision 2026-07-31).**
    A row that lives in memory carries an absent float as ``None``; the SAME row
    read back from ``records.parquet`` carries it as ``NaN``.  They are one fact —
    "MASTER did not report this axis" — so every feasibility axis must judge them
    identically, whichever way that axis judges a missing value.

    The bug this closes: :func:`is_feasible`'s ``max_pin_burnup`` guard tested
    ``pin_bu is not None``, so a missing pin burnup PASSED in memory and, as
    ``NaN``, FAILED after a parquet round-trip (``nan <= 80.0`` is ``False``).
    Since almost no store row carries pin BU, every parquet-sourced row was judged
    infeasible under the three objectives that gate it (``flat_power`` /
    ``fr_boundary`` / ``min_fuel_cost``) — which silently collapsed the
    feasible-first tier of :meth:`CampaignDriver._store_elites` into a plain
    objective sort, and under-reported ``n_feasible`` everywhere.

    A non-numeric value is NOT missing (it is invalid); ``is_feasible``'s own
    ``float()`` still rejects it.
    """

    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def is_feasible_search(row: Mapping[str, Any], limits: Mapping[str, Any]) -> bool:
    """SEARCH-TIME constraint feasibility of a verified row under ``limits``.

    One of the TWO predicates (review 2026-08-29 §6.4 / P0-03, design §3.5.4).
    This is the SEARCH contract — candidate ranking, elite pools, best-tracking,
    the outer weights, ``n_feasible`` — and it is DELIBERATELY tolerant of an
    axis MASTER did not report: a strict reject would zero the feasible set and
    starve the search before the first label of a new axis ever arrives.  For the
    DELIVERY verdict use :func:`is_deliverable`, which refuses exactly what this
    one tolerates.

    **The search-time contract, stated once:** ``max_pin_burnup`` and ``f_xy``
    PASS when missing; every other gated axis REJECTS when missing.

    :meth:`CampaignDriver._is_feasible` and the report's search statistics are
    the same function of the same limits, because a report that restates the
    campaign's rule drifts from it — it did, by omitting the pin-BU gate the
    campaign applies, and then called rows feasible that the campaign had
    rejected.

    **Missing values (2026-07-31 contract, see :func:`_is_missing`): ``NaN`` is
    treated exactly like ``None``.**  Two groups, each internally consistent:

    * ``cbc_max`` / ``f_q`` / ``ao_abs`` / ``f_r`` / ``cyclen`` — a missing value
      REJECTS.  These are the primary constraints and an unmeasured one cannot be
      called satisfied.  (Unchanged: ``None`` and ``NaN`` both already rejected;
      the checks are now spelled out with :func:`_is_missing` so that is auditable
      rather than a side effect of ``nan <= x`` being ``False``.)
    * ``max_pin_burnup`` / ``f_xy`` — a missing value PASSES.  MASTER adjudicates
      both, and a strict reject would zero feasibility on every row that lacks
      the field, starving elites / best-tracking / the outer weights.  ``f_xy``
      joins this group for the same reason and one sharper one: at the moment the
      objective switched, 98.2% of the store had no F_xy label at all, so a
      strict reject would have made the first ``min_fxy`` campaign unable to
      start.  ``max_pin_burnup`` is the axis the NaN hole broke, and folding NaN
      into "missing" here CHANGES the reported feasible set: parquet-sourced rows
      with no pin-BU label now count as feasible, exactly as this docstring has
      always claimed they did.
    """
    try:
        ok = bool(
            not _is_missing(row.get("cbc_max"))
            and float(row["cbc_max"]) <= float(limits["cbc_max"])
            and not _is_missing(row.get("f_q"))
            and float(row["f_q"]) <= float(limits["f_q"])
            and not _is_missing(row.get("ao_abs"))
            and float(row["ao_abs"]) <= float(limits["ao_abs"])
        )
        fr_limit = limits.get("f_r")
        if fr_limit is not None:
            ok = ok and not _is_missing(row.get("f_r")) and float(row["f_r"]) <= float(fr_limit)
        lo, hi = limits.get("cyclen_lo"), limits.get("cyclen_hi")
        if lo is not None or hi is not None:
            cyclen = row.get("cyclen")
            ok = ok and not _is_missing(cyclen)
            if ok and lo is not None:
                ok = ok and float(cyclen) >= float(lo)
            if ok and hi is not None:
                ok = ok and float(cyclen) <= float(hi)
        pin_limit = limits.get("max_pin_burnup")
        if pin_limit is not None:
            pin_bu = row.get("max_pin_burnup")
            # MISSING (None OR NaN) PASSES — see the docstring above.  Testing
            # ``is not None`` here was the whole defect.
            if not _is_missing(pin_bu):
                ok = ok and float(pin_bu) <= float(pin_limit)
        fxy_limit = limits.get("f_xy")
        if fxy_limit is not None:
            f_xy = row.get("f_xy")
            # MISSING PASSES at SEARCH time (design §3.5.4) — and REJECTS in
            # :func:`is_deliverable`.  That split is the whole point.
            if not _is_missing(f_xy):
                ok = ok and float(f_xy) <= float(fxy_limit)
        return ok
    except (TypeError, ValueError, KeyError):
        return False


#: Back-compat alias.  Every historical caller means the SEARCH predicate, and
#: they were all audited when the split landed; new delivery-side code must name
#: :func:`is_deliverable` explicitly rather than inherit a tolerant default.
is_feasible = is_feasible_search


def deliverable_limits(limits: Mapping[str, Any]) -> dict[str, float | None]:
    """The limits :func:`is_deliverable` judges at, from a search-limit mapping.

    ONE axis is re-resolved: ``max_pin_burnup`` gates at
    :data:`DELIVERABLE_PIN_BU_LIMIT` (LEU+ 80), never at the 78.0 model-margin
    haircut the search applies.  That 2.0 GWd/tU is head margin on a PREDICTION
    and has no meaning against a MEASURED burnup, so re-using it as a licensing
    verdict would reject deliverable cores.

    Everything else (``f_r`` / ``cbc_max`` / ``f_q`` / ``ao_abs`` / the cyclen
    band / ``f_xy``) is exactly the run's own gate, and an axis the objective
    leaves UNGATED stays ungated here too.  In particular this does NOT narrow
    ``flat_power``'s 1.70 F_r safety gate to the licensing 1.55: that would be a
    new rejection rule, and program §2.2 already answers "how compliant is this
    row on F_r" with ``compliance_margin``, which every report row prints.  What
    the delivery predicate adds is the MEASUREMENT requirement, not new limits.
    """
    return {
        "cbc_max": limits.get("cbc_max"),
        "f_q": limits.get("f_q"),
        "ao_abs": limits.get("ao_abs"),
        "f_r": limits.get("f_r"),
        "max_pin_burnup": DELIVERABLE_PIN_BU_LIMIT,
        "cyclen_lo": limits.get("cyclen_lo"),
        "cyclen_hi": limits.get("cyclen_hi"),
        "f_xy": limits.get("f_xy"),
    }


#: Gated-axis name -> the row column that must carry its measurement.  The cyclen
#: band's two limit keys share one column, which is why this is a table and not
#: ``limits.keys()``.
_DELIVERABLE_AXIS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cbc_max", "cbc_max"),
    ("f_q", "f_q"),
    ("ao_abs", "ao_abs"),
    ("f_r", "f_r"),
    ("max_pin_burnup", "max_pin_burnup"),
    ("cyclen_lo", "cyclen"),
    ("cyclen_hi", "cyclen"),
    ("f_xy", "f_xy"),
)


def unknown_axes(row: Mapping[str, Any], limits: Mapping[str, Any]) -> tuple[str, ...]:
    """Gated licensing axes of ``row`` that carry NO measurement (UNKNOWN state).

    The third state of the review's §6.4 table (measured PASS / measured FAIL /
    UNKNOWN), surfaced as data so the report can say WHICH axis is unmeasured
    instead of only that the row is undeliverable.  Column names, de-duplicated,
    in the fixed order of :data:`_DELIVERABLE_AXIS_COLUMNS`.
    """
    resolved = deliverable_limits(limits)
    out: list[str] = []
    for axis, column in _DELIVERABLE_AXIS_COLUMNS:
        if resolved.get(axis) is None:
            continue
        if column in out:
            continue
        if _is_missing(row.get(column)):
            out.append(column)
    return tuple(out)


def build_delivery_payload(
    rows: Iterable[Mapping[str, Any]],
    *,
    objective: str,
    limits: Mapping[str, Any],
    cell: str | None,
    safety: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The ``delivery.json`` payload for a finished campaign, or ``None``.

    Single source of truth for the §2.2 / D2 delivery dossier: shared by the live
    driver (:meth:`CampaignDriver._write_delivery`) and by the offline
    :func:`rerender_run_artifacts` path, so a regenerated ``delivery.json`` can
    never drift from the one the run itself would have written.

    ``None`` when ``objective`` is not ``flat_power`` — only that mode defines a
    delivery ranking.  ``safety`` supplies the §8.5 uncertainty fields per entry;
    omitted (the offline path) they are ``None`` = NOT EVALUATED, which is what
    the review's rule requires — never a pass.
    """

    if objective != "flat_power":
        return None
    # DELIVERY, so the DELIVERY predicate: a row whose F_xy or pin burnup was
    # never measured is UNKNOWN and must not be handed over as a candidate
    # (review §6.4; this used the tolerant search predicate before the split).
    kept = [r for r in rows if r.get("converged") and is_deliverable(r, limits)]
    dossier = select_delivery(kept).as_dict()
    # Stamp the §8.5 uncertainty fields onto every dossier entry.  The RANKING
    # and the deliverable predicate are untouched — this only makes each entry
    # SAY whether its fuel population was OOD and which gated axes its
    # conformal calibration does not cover, so a hand-off cannot read a
    # flagged or uncalibrated candidate as clean.
    for group in ("ranked", "excluded"):
        for entry in dossier.get(group, []):
            entry.update(
                safety(entry) if safety is not None
                else {"ood_flag": None, "conformal_unfit_axes": None}
            )
    return {
        "rule": "flatness-first program 2.2 (decision D2)",
        "note": ("search objective = flatness (no F_r); this ranking is a "
                 "downstream LICENSING filter only, and the flattest "
                 "candidate is excluded by the band floor on purpose"),
        "cell": cell,
        "compliance_limit": LICENSING_FR_LIMIT,
        "next_step": "SDM / MTC / axial confirmation on the top candidates "
                     "(outside the search loop, program 2.2 step 3)",
        "sdm_mtc_gate": ("decision D9 — the top-K entries below are the SDM/MTC "
                         "pre-delivery gate's targets; see the sdm_mtc block "
                         "and sdm_mtc.json when [constraints] enables it"),
        **dossier,
    }


def is_deliverable(row: Mapping[str, Any], limits: Mapping[str, Any]) -> bool:
    """DELIVERY-TIME verdict: every gated axis MEASURED and inside its limit.

    The strict twin of :func:`is_feasible_search` (review §6.4 / P0-03, design
    §3.5.4).  Where the search predicate lets an unmeasured ``max_pin_burnup`` or
    ``f_xy`` pass so the search can run at all, this one REJECTS them: an
    unmeasured licensing axis cannot be called satisfied, and the review's Phase-A
    exit criterion is "zero UNKNOWN deliveries".

    Judged against :func:`deliverable_limits` — the LICENSING numbers, not the
    search-time model-margin gates.  ``unknown_axes`` names what is missing.
    """
    if unknown_axes(row, limits):
        return False
    # every gated axis is now MEASURED, so the tolerant branches of the search
    # predicate are unreachable and the two agree by construction.
    return is_feasible_search(row, deliverable_limits(limits))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
class CampaignDriver:
    def __init__(
        self,
        cfg: LpoptConfig,
        model: Any,
        evaluator_factory: Callable | None = None,
        *,
        dry_run: bool = False,
        run_dir: str | Path | None = None,
        budget: int | None = None,
        max_waves: int | None = None,
        resume: bool = False,
        backend_factory: Callable[[str], Any] | None = None,
        fuel_library: Any = None,
        progress: bool = True,
        log: Callable[[str], None] | None = None,
        seed: int | None = None,
        early_stop: bool = True,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.evaluator_factory = evaluator_factory
        self.dry_run = bool(dry_run)
        self.max_waves = max_waves
        self.resume = bool(resume)
        self.early_stop_enabled = bool(early_stop)
        self.backend_factory = backend_factory
        self.progress = progress
        # Encoding-safe: a redirected Windows stdout is cp949, and a single
        # em-dash in a log line used to raise UnicodeEncodeError and sink a
        # finished 100-call run (incident 2026-08-30).  Wraps a supplied
        # ``log`` too -- that stream belongs to the caller, not to lpopt.
        self._log = safe_logger(log)
        self.acq = cfg.acquisition
        self.search = cfg.search

        base = cfg.source_path.parent if cfg.source_path else Path.cwd()
        self._base = base
        self.main_store_dir = self._resolve(cfg.model.store_dir)

        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = self._resolve(cfg.flow.output_root) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("logs", "waves", "candidates", "figures", "models"):
            (self.run_dir / sub).mkdir(exist_ok=True)

        self.state_path = self.run_dir / STATE_NAME
        self.labels_path = self.run_dir / LABELS_NAME
        self.events_path = self.run_dir / "logs" / "events.jsonl"

        # dry-run writes to a run-scoped store (no main-store pollution).
        self.campaign_store_dir = (
            self.run_dir / "store" if self.dry_run else self.main_store_dir
        )
        self.store = StoreWriter(self.campaign_store_dir)

        self.budget = int(budget if budget is not None else self.acq.budget)
        self.rng = random.Random(
            seed if seed is not None else cfg.flow.random_seed
        )
        self.fuel = fuel_library or self._load_fuel()
        self.ctx = self._case_context()
        self.constraints = acq.make_constraints(self.acq)
        self.library_id = cfg.model.library_id
        # Objective selector (plan sec. 1-4 + user directive 2026-07-21).  Default
        # "target_cycle" is byte-identical to the pre-change campaign; the additive
        # "max_cycle_min_fr" mode maximizes cyclen + minimizes F_r under the F_q /
        # CBC / |AO| constraints (F_r ungated) via ``acq.score_pool_max_cycle``.
        self.objective = str(getattr(self.acq, "objective", "target_cycle"))
        #: The f_xy serve-sigma mode the deck LAUNCHED with — ``"barred"`` when the
        #: constructed champion stamps ``fxy_head.serve_sigma = "barred"`` (G4),
        #: else ``"head"``.  Frozen here, before any wave checkpoint can replace the
        #: served weights, so :meth:`_load_state` has a launch-time reference to
        #: assert the resumed checkpoint against (defect D3,
        #: ``data/reports/minfxy_T6T4_f121_r1_results_20260830.md`` §9).
        self._fxy_sigma_mode_launch = self._fxy_sigma_mode()
        self.max_cycle_spec: acq.MaxCycleSpec | None = None
        self.min_fr_spec: acq.MinFrSpec | None = None
        self.min_fxy_spec: acq.MinFxySpec | None = None
        self.min_fuel_cost_spec: acq.MinFuelCostSpec | None = None
        self.fr_boundary_spec: acq.MinFrBoundarySpec | None = None
        self.flat_power_spec: acq.FlatPowerSpec | None = None
        # flat_power normalization (program §1.2/D4) — resolved below for that
        # objective only; the defaults keep every other mode untouched.
        self.flat_scale: FlatScale | None = None
        self.flat_cell_key: str | None = None
        #: map-head level calibration (program §2.1); inert outside flat_power.
        self.map_calibration: MapCalibration = MapCalibration.empty()
        #: the champion the calibration has been PROVEN to belong to (re-proven at
        #: every swap of the served weights, not once at construction).
        self._calibrated_model_dir: str | None = None
        #: one-shot: the served weights have been fine-tuned away from the
        #: checkpoint the calibration was fitted on (a level correction is a
        #: property of one checkpoint's head, so this is reported, never assumed).
        self.map_calibration_stale = False
        self.flat_w_cov = float(getattr(self.acq, "flatpower_w_cov", 0.5))
        self.flat_peak_scale = float(_FS.DEFAULT_PEAK_SCALE)
        self.flat_cov_scale = float(_FS.DEFAULT_COV_SCALE)
        #: the RESOLVED normalizer actually applied (per-cell lookup + any deck
        #: override already folded in), so the verified-row objective divides by
        #: exactly what the acquisition spec divides by.
        self._flat_eff_scale = FlatScale(peak_scale=self.flat_peak_scale,
                                         cov_scale=self.flat_cov_scale,
                                         per_cell=False)
        #: IDENTITY of the normalizer a persisted ``best["objective"]`` is in
        #: (``None`` outside flat_power).  Resolved after the spec is built.
        self.flat_scale_id: dict[str, Any] | None = None
        #: per-wave map harvest rate (converged rows carrying node_peak / converged)
        self.map_harvest_rates: list[float] = []
        #: one-shot: the frozen holdout could not be restricted to mapped rows.
        self._warned_unmapped_holdout = False
        #: D9 SDM/MTC pre-delivery gate accounting (licensing budget, NOT search
        #: budget — the two are reported separately and never merged).
        self.post_verify_calls = 0
        self.post_verify_violators: list[str] = []
        self.post_verify_summary: dict[str, Any] | None = None
        #: True once the D9 gate has ACTUALLY run for this run dir.  Persisted, so
        #: a resumed / re-reported run never re-spends the licensing budget.
        self.post_verify_done = False
        #: Test seam: an injected :class:`sdm_mtc.BranchExecutor` runs the gate
        #: without MASTER.  ``None`` (production) builds the real branch runner.
        self.post_verify_executor: Any = None
        if self.objective in _RETIRED_PRODUCTION_OBJECTIVES:
            # program §10 STOP: these two minimize/trade F_r as the search axis,
            # which is exactly the steering the flatness-first switch retired.
            # They still RUN (reproducibility, A/B baselines) but a production
            # campaign must not reach one silently.
            self._log(
                f"[optimize][DEPRECATED] objective={self.objective!r} is a RETIRED "
                "production mode (flatness-first program §10 STOP): it steers the "
                "search by F_r. Kept runnable for reproduction / A-B baselines "
                "only — use objective='flat_power' for production."
            )
        if self.objective == "max_cycle_min_fr":
            self.max_cycle_spec = acq.MaxCycleSpec(
                lam=float(getattr(self.acq, "mcmf_lambda", 100.0)),
                risk_z=float(self.acq.risk_z),
                cbc_limit=float(self.acq.cbc_limit),
                f_q_limit=float(self.acq.f_q_limit),
                ao_abs_limit=float(self.acq.ao_abs_limit),
            )
        elif self.objective == "min_fr_max_cycle":
            self.min_fr_spec = acq.MinFrSpec(
                lam_fr=float(getattr(self.acq, "minfr_lambda", 1000.0)),
                risk_z=float(self.acq.risk_z),
                f_r_limit=float(self.acq.f_r_limit),
                cbc_limit=float(self.acq.cbc_limit),
                f_q_limit=float(self.acq.f_q_limit),
                ao_abs_limit=float(self.acq.ao_abs_limit),
                pin_bu_limit=float(getattr(self.acq, "minfr_pin_bu_limit", 78.0)),
            )
        elif self.objective == "min_fxy":
            # F_xy PRIMARY (user decision 2026-08-29) — the same lambda structure
            # min_fr_max_cycle uses, with F_xy in F_r's place and F_r rejoining
            # the hard set as a pure constraint (design §3.5.2).
            self.min_fxy_spec = acq.MinFxySpec(
                lam_fxy=float(getattr(self.acq, "minfxy_lambda", 1000.0)),
                risk_z=float(self.acq.risk_z),
                f_xy_limit=float(getattr(self.acq, "f_xy_limit", 1.65)),
                f_r_limit=float(self.acq.f_r_limit),
                cbc_limit=float(self.acq.cbc_limit),
                f_q_limit=float(self.acq.f_q_limit),
                ao_abs_limit=float(self.acq.ao_abs_limit),
                pin_bu_limit=float(getattr(self.acq, "minfxy_pin_bu_limit", 78.0)),
                cyclen_lo=_opt_float(getattr(self.acq, "minfxy_cyclen_lo", None)),
                cyclen_hi=_opt_float(getattr(self.acq, "minfxy_cyclen_hi", None)),
                cyclen_width=float(getattr(self.acq, "minfxy_cyclen_width", 10.0)),
            )
            # The objective is DEFINED on the harvested MAS_OUT column ``f_xy``,
            # which only exists when the final work dir survives — and
            # ``harvest_maps`` is what forces ``keep_success``
            # (verify.WaveVerifier).  Without it EVERY verified row is unscorable
            # (``_campaign_objective`` -> -inf) and the run would rank on nothing,
            # which is precisely the flat_power §1.3 failure one axis over.
            if not bool(getattr(cfg.verify, "harvest_maps", False)):
                raise ValueError(
                    "objective='min_fxy' requires [verify] harvest_maps = true: "
                    "F_xy is parsed from the final cycle's MAS_OUT, which only "
                    "survives because harvest_maps forces keep_success. Without "
                    "it no verified row carries f_xy and the objective has "
                    "nothing to rank (design fxy_switch_20260829 §3.2)."
                )
            has_head = acq.has_fxy_head(self.model, self.ctx)
            self._log(
                f"[optimize] min_fxy objective = cyclen_LCB - "
                f"{self.min_fxy_spec.lam_fxy:g} * F_xy_UCB | HARD F_xy <= "
                f"{self.min_fxy_spec.f_xy_limit:.3f} | F_r stays a CONSTRAINT at "
                f"{self.min_fxy_spec.f_r_limit:.3f} | pin BU <= "
                f"{self.min_fxy_spec.pin_bu_limit:.1f}"
            )
            if not has_head:
                self._log(
                    "[optimize][F_xy PROXY] the served model exposes NO "
                    "`predict_fxy` head: candidates are ranked on the INTERIM "
                    f"proxy F_xy ~ {acq.FXY_PROXY_SLOPE:g}*F_r "
                    f"{acq.FXY_PROXY_INTERCEPT:+g} with an inflated sigma "
                    "(design §1.2). Every wave records fxy_source='proxy'. This "
                    "is an F_r-surrogate search with F_xy MEASURED after the "
                    "fact — it is NOT 'F_xy was optimized' (design §3.6)."
                )
            elif acq.fxy_sigma_barred(self.model):
                # A head whose MEAN promoted but whose WIDTH did not (G4).  The
                # readout must not let "fxy_source='head'" be read as covering the
                # UCB too — the UCB half is still the interim proxy convention.
                self._log(
                    "[optimize][F_xy SIGMA BARRED] the served checkpoint has a "
                    "`predict_fxy` head and its MEAN ranks the candidates "
                    "(fxy_source='head'), but the checkpoint stamps "
                    "fxy_head.serve_sigma='barred': its own 68% coverage failed "
                    "the registered [0.55, 0.80] gate (over-wide). F_xy_UCB "
                    f"therefore uses the INTERIM proxy sigma (resid_sd "
                    f"{acq.FXY_PROXY_RESID_SD:g} x K {acq.FXY_PROXY_SIGMA_K:g}), "
                    "not the head's, and no f_xy conformal bound is served."
                )
        elif self.objective == "min_fuel_cost":
            self.min_fuel_cost_spec = acq.MinFuelCostSpec(
                lam_fr=float(getattr(self.acq, "fuelcost_lambda_fr", 20.0)),
                risk_z=float(self.acq.risk_z),
                cyclen_lo=float(getattr(self.acq, "fuelcost_cyclen_lo", 615.0)),
                cyclen_hi=float(getattr(self.acq, "fuelcost_cyclen_hi", 635.0)),
                cyclen_width=float(getattr(self.acq, "fuelcost_cyclen_width", 10.0)),
                f_r_limit=float(self.acq.f_r_limit),
                cbc_limit=float(self.acq.cbc_limit),
                f_q_limit=float(self.acq.f_q_limit),
                ao_abs_limit=float(self.acq.ao_abs_limit),
                pin_bu_limit=float(getattr(self.acq, "fuelcost_pin_bu_limit", 80.0)),
            )
        elif self.objective == "fr_boundary":
            # F_r=1.55 boundary campaign: F_r is a PURE objective (no f_r_limit /
            # cyclen / λ fields), CBC/F_q/|AO| gate, pin-BU screened, cyclen recorded
            # but NEVER gated.  Band shaping toward the boundary is in the scorer.
            self.fr_boundary_spec = acq.MinFrBoundarySpec(
                risk_z=float(self.acq.risk_z),
                cbc_limit=float(self.acq.cbc_limit),
                f_q_limit=float(self.acq.f_q_limit),
                ao_abs_limit=float(self.acq.ao_abs_limit),
                pin_bu_limit=float(getattr(self.acq, "fr_boundary_pin_bu_limit", 80.0)),
                band_lo=float(getattr(self.acq, "fr_boundary_band_lo", 1.45)),
                band_hi=float(getattr(self.acq, "fr_boundary_band_hi", 1.70)),
            )
        elif self.objective == "flat_power":
            # FLATNESS-NATIVE (program §1.2): minimize the weighted
            # node_peak (PRIMARY) + map_cov (SECONDARY) UCB sum.  F_r is a pure
            # SAFETY GATE at flatpower_fr_limit (1.70, D1) — it is not in the
            # objective, not in the graded penalty tiers, and not in the
            # p_feasible / margin tie-break.  cyclen stays record-only.
            #
            # The normalizers are resolved for THIS campaign's cell so the
            # declared 1 : w_cov ratio is what the cell actually realizes (D4);
            # a cell the artifact never fitted falls back to the global
            # constants and says so in the log.
            self.flat_scale = FlatScale.from_store(
                self.main_store_dir,
                per_cell=bool(getattr(self.acq, "flatpower_per_cell_scale", True)),
            )
            self.flat_cell_key = cyclen_cell_key(self.ctx.feed, self.ctx.e_core)
            peak_scale, cov_scale = self.flat_scale.scales_for(self.flat_cell_key)
            ov_peak = float(getattr(self.acq, "flatpower_peak_scale", 0.0) or 0.0)
            ov_cov = float(getattr(self.acq, "flatpower_cov_scale", 0.0) or 0.0)
            if ov_peak > 0.0:
                peak_scale = ov_peak
            if ov_cov > 0.0:
                cov_scale = ov_cov
            self.flat_w_cov = float(getattr(self.acq, "flatpower_w_cov", 0.5))
            self.flat_peak_scale = float(peak_scale)
            self.flat_cov_scale = float(cov_scale)
            self._flat_eff_scale = FlatScale(peak_scale=self.flat_peak_scale,
                                             cov_scale=self.flat_cov_scale,
                                             per_cell=False,
                                             source=self.flat_scale.source)
            # What a persisted objective value MEANS.  Recorded next to every
            # stored best (and in state.json) so a refit of flat_scale.json —
            # which silently changes the units of the scalar — is detected on
            # resume instead of comparing two incomparable numbers.
            self.flat_scale_id = self._flat_eff_scale.identity(
                w_cov=self.flat_w_cov, cell=self.flat_cell_key)
            # program §2.1 (precondition): the map head's LEVEL calibration.
            # It carries (a) the D1 F_r safety-gate bias correction and (b) the
            # per-cell node_peak / map_cov bias + the dispersion the ensemble
            # spread does not have.  Absent -> every correction is inert and the
            # gate HOLDS at 1.70; present but fitted on a DIFFERENT champion ->
            # a loud refusal, never a silent misapplication (a level calibration
            # is a property of one checkpoint's head).
            self.map_calibration = MapCalibration.from_store(self.main_store_dir)
            self._require_calibration_model(
                self._resolve(cfg.model.model_dir), context="construction")
            fr_bias, fr_sigma = self.map_calibration.gate_for(self.flat_cell_key)
            peak_cal = self.map_calibration.resolve("node_peak", self.flat_cell_key)
            cov_cal = self.map_calibration.resolve("map_cov", self.flat_cell_key)
            self.flat_power_spec = acq.FlatPowerSpec(
                risk_z=float(self.acq.risk_z),
                w_cov=self.flat_w_cov,
                peak_scale=self.flat_peak_scale,
                cov_scale=self.flat_cov_scale,
                fr_limit=float(getattr(self.acq, "flatpower_fr_limit", 1.7)),
                fr_bias=fr_bias, fr_sigma=fr_sigma,
                peak_bias=peak_cal.bias if peak_cal.available else None,
                peak_sigma_extra=(peak_cal.sigma_extra if peak_cal.available
                                  else None),
                cov_bias=cov_cal.bias if cov_cal.available else None,
                cov_sigma_extra=(cov_cal.sigma_extra if cov_cal.available
                                 else None),
                cbc_limit=float(self.acq.cbc_limit),
                f_q_limit=float(self.acq.f_q_limit),
                ao_abs_limit=float(self.acq.ao_abs_limit),
                pin_bu_limit=float(getattr(self.acq, "flatpower_pin_bu_limit", 80.0)),
                fxy_limit=_opt_float(
                    getattr(self.acq, "flatpower_fxy_limit", 0.0)),
                rule_penalty_weights=self._rule_penalty_weights(),
            )
            # program §1.3: the objective is DEFINED on the harvested map columns.
            # A flat_power run without map harvesting produces no objective at all
            # and would silently rank on nothing — fail at construction, loudly.
            if not bool(getattr(cfg.verify, "harvest_maps", False)):
                raise ValueError(
                    "objective='flat_power' requires [verify] harvest_maps = true: "
                    "the objective is -(node_peak/PEAK_SCALE + w_cov*map_cov/"
                    "COV_SCALE) computed from the harvested EDIT5 map, and without "
                    "maps every verified row scores -inf (program §1.3)."
                )
            desc = self.flat_scale.describe(self.flat_cell_key)
            cal = self.map_calibration.describe(self.flat_cell_key)
            gate_note = ("bias-corrected per cell" if fr_bias is not None
                         else "HELD — no map-head bias correction available")
            fxy_gate_val = self.flat_power_spec.fxy_gate
            if fxy_gate_val is None:
                self._log("[optimize] flat_power F_xy safety gate OFF "
                          "(flatpower_fxy_limit unset/0)")
            elif acq.has_fxy_head(self.model, self.ctx):
                self._log(f"[optimize] flat_power F_xy SAFETY GATE at "
                          f"{fxy_gate_val:.3f} on the predict_fxy head")
            else:
                self._log(
                    f"[optimize] flat_power F_xy safety gate {fxy_gate_val:.3f} is "
                    "INERT in the acquisition (the served model has no predict_fxy "
                    "head, and the interim F_r proxy would re-impose an F_r screen "
                    "at ~1.53 — program §10). It STILL applies to every verified "
                    "row's MEASURED f_xy, so rows over it are reported infeasible."
                )
            self._log(
                f"[optimize] flat_power objective = -(node_peak/{peak_scale:.4f} "
                f"+ {self.flat_w_cov} * map_cov/{cov_scale:.4f}) | cell "
                f"{self.flat_cell_key} scales {'FITTED' if desc['fitted'] else 'GLOBAL fallback'} "
                f"({desc['source']}) | F_r SAFETY GATE at "
                f"{self.flat_power_spec.fr_gate:.3f} ({gate_note}, not an objective)"
            )
            if self.map_calibration.present:
                self._log(
                    f"[optimize] map_calibration ({cal['artifact']}, model "
                    f"{cal['fit_model_id'] or 'UNDECLARED'}): node_peak bias "
                    f"{cal['node_peak']['bias']} sigma_extra "
                    f"{cal['node_peak']['sigma_extra']} [{cal['node_peak']['source']}]"
                    f" | map_cov bias {cal['map_cov']['bias']} sigma_extra "
                    f"{cal['map_cov']['sigma_extra']} [{cal['map_cov']['source']}]"
                )
            else:
                self._log(
                    "[optimize] map_calibration ABSENT — the acquisition consumes "
                    "RAW map-head levels (fold C optimism node_peak −0.147 / "
                    "map_cov −0.058) and the F_r gate holds flat. Program §2.1 "
                    "makes this artifact a precondition: run "
                    "`python -m lpopt.tools.fit_map_calibration`."
                )

        # dry-run lightens the heavy CPU knobs (StubEvaluator acceptance path).
        self.pool_size = (
            self.search.dry_run_pool_size if self.dry_run else self.search.pool_size
        )
        self.replay_size = (
            self.acq.dry_run_replay_size if self.dry_run else self.acq.replay_size
        )
        self.finetune_epochs = (
            self.acq.dry_run_finetune_epochs if self.dry_run else self.acq.finetune_epochs
        )
        # dry-run lightens the local-search prediction budget (CPU-friendly).
        self.local_search_cfg = self.search.local_search
        if self.dry_run:
            self.local_search_cfg = _replace(
                self.search.local_search,
                top_m=16, neighbors=8, depth=2, max_predictions=600,
            )

        self.resolver = self._resolver()
        self.verifier = self._verifier()
        #: Maps that could not be written (store locked by a concurrent reader).
        #: Labels are unaffected; counted so a map gap is never silent.
        self.maps_skipped_waves = 0
        self.maps_skipped_records = 0

        # runtime state (restored on resume).
        self.wave_index = 0
        self.budget_spent = 0
        self.consecutive_halts = 0
        self.no_improve = 0
        self.best: dict[str, Any] | None = None
        #: Best-by-objective among ALL converged rows regardless of constraint
        #: feasibility (the honesty channel: min_fr_max_cycle may end with NO
        #: feasible label, so status/report expose ``best`` (feasible, may be null)
        #: and ``best_overall`` separately — never promoting a violator into ``best``).
        self.best_overall: dict[str, Any] | None = None
        self.champion_ckpt = str(self._resolve(cfg.model.model_dir))
        self.ledger_ids: set[str] = set()
        self.verified_patterns: list[Pattern] = []
        self.campaign_rows: list[dict[str, Any]] = []
        self.control_rows: list[dict[str, Any]] = []
        self.wave_reports: list[WaveReport] = []
        #: Previous wave's top exploit-ranked pool candidates, kept as whole
        #: Candidates (not (id, Pattern) pairs) so their lineage anchor is
        #: resolved at USE time -- by then this wave's rows are in the ledger, so
        #: a top candidate that actually got verified anchors on itself instead of
        #: on its grandparent.  See :func:`~.verify.lineage_anchor`.
        self.prev_top: list[Candidate] = []
        self.reward_model: Any = None
        self.trust_region = acq.TrustRegion.from_store(
            self.main_store_dir, self.search.trust_region, self.ctx
        )
        #: OOD policy + conformal chance constraint (review §6.5 / §8.5).  Inert
        #: at the defaults (``ood_policy = "warn"``, ``conformal_gate = false``).
        self.safety_shield = acq.SafetyShield.from_config(self.acq)
        #: ``{record_id: {"ood_flag": bool|None, "conformal_unfit_axes": [...]}}``
        #: for the candidates THIS session verified — the delivery dossier's
        #: uncertainty fields.  Rows carried over by a resume are absent from the
        #: map and report ``None`` (= "not evaluated"), never a clean ``False``.
        self._row_safety: dict[str, dict[str, Any]] = {}
        #: Last wave's :func:`acquisition.apply_safety_shield` accounting.
        self._shield_report: dict[str, Any] = {}

    # -- construction helpers ---------------------------------------------- #
    def _resolve(self, path_str: str | Path) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (self._base / p)

    # -- engineering-rule soft penalty ------------------------------------- #
    def _rule_penalty_weights(self) -> dict[str, float] | None:
        """``{metric: weight}`` from ``[acquisition] flatpower_rule_penalty_*``.

        ``None`` when every weight is 0.0 (the default), which is what makes the
        objective byte-identical to the pre-adoption campaign.  Only the metrics
        VALIDATED by the 218-cell within-cell study have a knob at all
        (:data:`..search.rule_metrics.VALIDATED_PENALTY_METRICS`); the soft
        penalty is a preference term outside the constraint tier and can never
        veto a candidate.
        """
        weights = {
            name: float(getattr(self.acq, f"flatpower_rule_penalty_{name}", 0.0) or 0.0)
            for name in acq_rules.VALIDATED_PENALTY_METRICS
        }
        active = {k: v for k, v in weights.items() if v != 0.0}
        if not active:
            return None
        self._log(
            "[optimize] engineering-rule SOFT penalty active (report R-03/R-04): "
            + ", ".join(f"{k}={v:g}" for k, v in sorted(active.items()))
            + " — subtracted from the exploit score only; NOT a constraint, "
              "never a veto"
        )
        return active

    # -- map-calibration provenance ---------------------------------------- #
    def _require_calibration_model(self, model_dir: str | Path | None, *,
                                   context: str) -> bool:
        """Refuse a map calibration not fitted on the champion NOW being served.

        Identity is re-proven at EVERY point the campaign replaces the served
        weights — construction, a resume that reloads a persisted champion, and
        the per-wave champion swap — because a check that only ran at
        construction says nothing about the checkpoint serving three waves later:
        the calibration would keep applying, silently, to a model it was never
        fitted on (program §2.1 — a level correction mis-states the D1 F_r safety
        gate and the UCB pessimism in the direction nobody notices).

        ``context`` names the moment, so a refusal points at the swap that caused
        it rather than at the artifact alone.  Returns ``True`` when identity is
        proven, ``False`` when there is nothing to prove it with (absent /
        unverifiable artifact — warned by
        :meth:`..data.map_calibration.MapCalibration.require_model`), and raises
        :class:`ModelMismatchError` when the artifact provably belongs to another
        champion.

        Three cases, and the middle one is the whole point:

        (a) **this run's own wave champion** (``<run_dir>/models/champion_wave_NN``
            — :meth:`_is_wave_champion`) whose CONTENT no longer matches the
            fitted champion.  Since the fingerprint began covering ``model.pt``
            weights, a wave fine-tune makes this the NORMAL outcome, not an
            anomaly: the fingerprint is correctly reporting "these are not the
            fitted weights".  Aborting on it is the wrong response — the
            checkpoint is a DESCENDANT of the champion the calibration was fitted
            on (only this run writes there, and only by fine-tuning the model it
            was constructed with), so the level correction stays APPROXIMATELY
            valid.  The run CONTINUES, the drift is warned, and the calibration is
            marked stale (:meth:`_note_finetuned_weights`, persisted in
            ``state.json``) so no corrected level is ever read as a proven one.
            This does not loosen D1: the gate correction it feeds is unchanged and
            still never loosens past 1.70 — what changes is that an approximate
            correction is now LABELLED approximate instead of killing the run.
        (b) **any other champion** — a different fitted model dir, or anything
            under ``models/`` that is not a wave champion of this run.  There is
            no descent to argue from, so this still ABORTS, exactly as before.
        (c) **no fingerprint on one side** — the :meth:`_content_decidable` path,
            unchanged: nothing can decide on content, and a run-scoped NAME is not
            evidence, so there is nothing to prove either way.
        """
        if self.objective != "flat_power":
            return False
        if self._is_run_scoped(model_dir) and not self._content_decidable(model_dir):
            # A run-scoped ``models/champion_wave_NN`` is a fresh checkpoint of the
            # lineage already being served, and its NAME is never the fitted
            # champion's — so the name is not evidence here, and with no
            # fingerprint to decide on CONTENT there is nothing to prove either
            # way.  Manufacturing a refusal out of a naming convention would
            # repeat, at the swap, exactly the name-first inversion the artifact's
            # fingerprint exists to prevent.  The drift is reported instead (see
            # :meth:`_note_finetuned_weights`).
            return False
        try:
            ok = self.map_calibration.require_model(model_dir, log=self._log)
        except ModelMismatchError as exc:
            if self._is_wave_champion(model_dir):
                # (a) a fine-tuned DESCENDANT, not another champion.  Report and
                # continue; never silently — the stale flag is what stops the
                # approximate correction being read as a proven one.
                self._note_finetuned_weights(
                    detail=f"[{context}] the fingerprint now covers the weights, "
                           f"so it correctly reports a difference: {exc}")
                return False
            raise ModelMismatchError(f"[{context}] {exc}") from None
        if ok:
            self._calibrated_model_dir = str(model_dir)
        return ok

    def _content_decidable(self, model_dir: str | Path | None) -> bool:
        """Can :meth:`MapCalibration.require_model` decide this on CONTENT?

        Only when BOTH sides carry a fingerprint — that is the exact condition
        under which ``require_model`` lets content decide; with either side
        missing one it falls through to the NAME, which for a run-scoped
        checkpoint is not evidence at all (``champion_wave_03`` is never the
        fitted champion's name, so the name branch is a guaranteed refusal).

        The escape hatch previously tested only the SERVING side, which is the
        wrong half of the pair: the common case is a *legacy / hand-written*
        artifact that records no fingerprint while the run-scoped checkpoint has
        one, and there the hatch did not fire — ``require_model`` reached the name
        branch and aborted a legitimate run at its own champion swap.

        It is not a hole: when both fingerprints exist this returns ``True`` and
        the full content check runs, so a genuinely different model is still
        refused.  Cheap side first — the artifact's recorded string — so the
        serving side's (weight-hashing) fingerprint is not computed when the
        artifact could never have settled it anyway.
        """
        if not self.map_calibration.fit_model_fingerprint:
            return False
        return bool(model_fingerprint(model_dir))

    def _is_run_scoped(self, model_dir: str | Path | None) -> bool:
        """Is ``model_dir`` a checkpoint THIS run wrote (``<run_dir>/models/...``)?"""
        if model_dir is None:
            return False
        try:
            return Path(str(model_dir)).resolve().is_relative_to(
                (self.run_dir / "models").resolve())
        except (OSError, ValueError):        # pragma: no cover - exotic path
            return False

    def _is_wave_champion(self, model_dir: str | Path | None) -> bool:
        """Is ``model_dir`` a checkpoint THIS run's wave fine-tune wrote?

        Deliberately narrower than :meth:`_is_run_scoped`: run-scoped *and* named
        by :meth:`_save_champion`'s own convention (``champion_wave_NN``).  That
        pair is what makes the descent argument in
        :meth:`_require_calibration_model` case (a) sound — the file was written
        by ``self.model.save()`` from the model this run was constructed with, so
        it can only be a fine-tuned descendant of the constructed champion.  A
        foreign checkpoint someone drops into ``<run_dir>/models/`` under any
        other name carries no such argument and is still refused.
        """
        if not self._is_run_scoped(model_dir):
            return False
        return _WAVE_CHAMPION_RE.fullmatch(Path(str(model_dir)).name) is not None

    def _note_gate_weight_drift(self, gate: Any) -> None:
        """Report the fine-tune drift whenever the SERVED weights actually moved.

        ACCEPTED — the challenger is the champion now, so they moved by design.

        REJECTED — normally the snapshot is restored and nothing moved, which is
        why this branch used to be silent.  But ``update._restore`` can only roll
        back a backend with member state: a **stateless-refit** backend is not
        snapshotted, so its refit SURVIVES the rejection and the served weights
        moved anyway — while the champion pointer, the member metas and therefore
        the fingerprint all say nothing happened.  Under ``MODEL_HALT`` that is
        precisely the model the run keeps serving for the rest of the campaign.

        ``gate.weights_rolled_back`` is the backend's own answer, so this does not
        have to guess which backend is in play.  Nothing here loosens the gate —
        the drift is reported exactly as for an accepted swap.
        """
        if gate.accepted or not getattr(gate, "weights_rolled_back", False):
            self._note_finetuned_weights()

    def _note_finetuned_weights(self, detail: str | None = None) -> None:
        """The wave fine-tune moved the served weights off the fitted checkpoint.

        The champion pointer still names a checkpoint of the same lineage, but the
        map head that produced the fitted bias/sigma is no longer the map head
        being served.

        VISIBILITY (corrected — the v2 fingerprint changed this).  This used to
        say the drift was invisible because the member ``meta.json`` files are
        unchanged and the fingerprint therefore could not see it.  That is no
        longer true: :func:`..data.map_calibration.model_fingerprint` now hashes
        the members' ``model.pt`` WEIGHTS as well as their metas, so a fine-tune
        does change the fingerprint and ``require_model`` does see it — it
        correctly reports "not the fitted champion".  What it cannot see is the
        DIFFERENCE between the two ways that can happen: a fine-tuned descendant
        of the fitted champion (approximation) and a genuinely different champion
        (misapplication).  Only the caller knows which, so
        :meth:`_require_calibration_model` case (a) routes the descendant here and
        raises on everything else.

        Two paths still reach this method with NO fingerprint evidence at all —
        an accepted gate (:meth:`_note_gate_weight_drift`) and a stateless-refit
        backend whose refit survived a rejection — because there the champion
        pointer, the metas and the fingerprint can all agree while the served
        weights moved.

        Nothing here loosens the gate (that would be the one unsafe direction,
        D1); the drift is REPORTED once and recorded in the run state so a
        corrected level is never mistaken for a proven one.  ``detail``, when
        given, names the moment and the evidence.
        """
        if self.objective != "flat_power" or not self.map_calibration.present:
            return
        if self.map_calibration_stale:
            return
        self.map_calibration_stale = True
        self._log(
            "[optimize][WARNING] the wave fine-tune replaced the served weights; "
            f"map_calibration ({self.map_calibration.source}) was fitted on "
            f"{self.map_calibration.fit_model_id or 'an undeclared champion'} and "
            "its node_peak / map_cov bias + the D1 gate correction are now "
            "APPROXIMATE for the serving head (refit with "
            "`python -m lpopt.tools.fit_map_calibration` against the promoted "
            "champion to restore a proven correction)"
            + (f"; {detail}" if detail else "")
        )

    def _load_fuel(self) -> Any:
        from ..data.fuel_types import FuelLibrary

        path = self.main_store_dir / "fuel_types.parquet"
        try:
            return FuelLibrary.from_parquet(path)
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _case_context(self) -> CaseContext:
        case = self.cfg.case
        e_core = self._case_e_core()
        return CaseContext(
            pair=str(case.pair),
            feed=int(case.feed),
            library_id=self.cfg.model.library_id,
            e_core=e_core,
            require_all_batches=bool(
                getattr(self.search, "require_all_fresh_types", False)),
        )

    def _case_e_core(self) -> float | None:
        from ..data.fuel_types import case_e_core as _case_e_core

        case = self.cfg.case
        if self.fuel is not None and case.pair:
            try:
                # 2 members -> the unchanged pair_e_core(a, b, 0.5); 3 members ->
                # the composition mean of the graded case (fuel_types.case_e_core).
                members = [p for p in str(case.pair).split("_") if p]
                return float(_case_e_core(
                    self.fuel, members, self.cfg.model.library_id))
            except (KeyError, ValueError, ZeroDivisionError, TypeError):
                pass
        # fall back to the store's median e_core for the case.
        try:
            df = StoreReader(self.main_store_dir).records
            sub = df[df["case_pair"] == case.pair]["e_core"].dropna()
            if len(sub):
                return float(sub.median())
        except (FileNotFoundError, OSError, KeyError):
            pass
        return None

    def _resolver(self) -> CaseAssetResolver:
        strict = (self.objective in ("min_fuel_cost", "fr_boundary", "flat_power")
                  and self.evaluator_factory is None)
        # A paramA campaign must take the SAME per-library routing the produce
        # path uses (registry alias bridge + the package's own %GEN_DIM dims +
        # no ga80 template fallbacks).  The plain construction below predates
        # that routing: on the first-ever paramA `optimize` run (T6_T4/f121,
        # 2026-08-11) it resolved restarts fine (the package root holds the
        # bases) but kept the ga80 LIBRARY_DIMS default, so the verifier's
        # deck sanity gate rejected all 100 prepared decks with
        # "(40, 42) != library (83, 85)" -- 0 MASTER launches in 100 calls.
        from .resolver import build_case_resolver, is_paramA_library
        lib_id = getattr(self.cfg.model, "library_id", None)
        if is_paramA_library(self.cfg, lib_id):
            r = build_case_resolver(self.cfg, self.fuel, library_id=lib_id)
            r.strict_restart = strict
            return r
        package_root = (
            self._resolve(self.cfg.verify.package_root)
            if self.cfg.verify.package_root
            else self.run_dir
        )
        fallbacks = [
            str(self._resolve(g)) for g in self.cfg.produce.template_fallbacks
        ]
        return CaseAssetResolver(
            package_root,
            self._resolve(self.cfg.produce.promoted_root),
            template_fallbacks=fallbacks,
            fuel_library=self.fuel,
            synth_root=self._resolve(self.cfg.produce.synth_decks_root),
            # min_fuel_cost outer search races many (pair, feed) cells; a cell whose
            # pair has no pair-matched restart must fail HARD here (not silently burn
            # a MASTER wave on an incompatible cross-pair fallback) so the outer
            # orchestrator can skip it — forensic 20260721.  Live path only; stub
            # dry-runs and every other objective keep the graceful fallback.
            strict_restart=strict,
        )

    def _verifier(self) -> WaveVerifier:
        package_root = (
            self._resolve(self.cfg.verify.package_root)
            if (self.evaluator_factory is None and self.cfg.verify.package_root)
            else None
        )
        stub = self.evaluator_factory is not None
        return WaveVerifier(
            run_dir=self.run_dir / "master",
            package_root=package_root,
            executable=self.cfg.master.executable if self.evaluator_factory is None else None,
            # OPTIMIZE default = 8 P-cores ([master] workers=8); the [master]
            # use_all_cores knob (default off) spreads onto every logical core.
            # A stub run has no cores to fill -> keep the legacy 8-wide wave.
            workers=(self.cfg.master.workers or 8) if stub else self.cfg.master.workers,
            use_all_cores=self.cfg.master.use_all_cores,
            host_reserve=self.cfg.master.host_reserve,
            timeout=self.cfg.master.timeout,
            max_cycles=self.cfg.master.max_cycles,
            consecutive=self.cfg.master.consecutive,
            tolerances=self.cfg.master.tolerances,
            evaluator_factory=self.evaluator_factory,
            resolver=self.resolver,
            # The verifier's deck sanity gate must judge %GEN_DIM against the
            # dims of the library THIS campaign actually runs on.  The resolver
            # already carries the per-library value (paramA reads its package's
            # MAS_XSL; ga80 keeps the (83, 85) constant) but the verifier used
            # to fall back to the ga80 default -- on the first-ever paramA
            # `optimize` run (T6_T4/f121, 2026-08-11) that rejected every
            # prepared deck with "(40, 42) != library (83, 85)": 32/32 chains
            # errored before MASTER launched.
            library_dims=self.resolver.library_dims,
            purge_intermediate=self.cfg.produce.purge_intermediate,
            harvest_maps=bool(getattr(self.cfg.verify, "harvest_maps", False)),
        )

    # -- store reads ------------------------------------------------------- #
    def _store_df(self):
        try:
            return StoreReader(self.main_store_dir).records
        except (FileNotFoundError, OSError):
            return None

    def _case_store_rows(self, converged: bool = True) -> list[dict[str, Any]]:
        df = self._store_df()
        if df is None or not len(df):
            return []
        sub = df[df["case_pair"] == self.ctx.pair]
        if converged:
            sub = sub[sub["converged"] == True]  # noqa: E712
        return [row.to_dict() for _, row in sub.iterrows()]

    def _elite_seed_rows(self) -> list[dict[str, Any]]:
        """Converged store rows of the DONOR cases named by ``[search]
        elite_seed_cases`` — mutation parents only.

        A graded 3-type case (``A_B_C``) matches no store row by ``case_pair``,
        so its elite pool is empty and ``graded_morph`` — which re-labels a slice
        of a 2-type parent onto the third type — has nothing to work from.
        Naming the parent PAIR as a donor hands those optimized boards to
        :meth:`_store_elites`; ``build_pool`` then mutates them under
        ``ctx.batches`` (the 3-type alphabet), which is where the morph happens.

        Deliberately NOT routed through :meth:`_case_store_rows`: that method also
        feeds :meth:`_holdout_rows` (the wave fine-tune gate), and a donor row is
        not a label of this case.  Empty list when the knob is unset, so every
        pre-existing deck keeps its exact parent set.
        """

        cases = [str(c) for c in (getattr(self.search, "elite_seed_cases", ()) or ())
                 if str(c) and str(c) != self.ctx.pair]
        if not cases:
            return []
        df = self._store_df()
        if df is None or not len(df):
            return []
        sub = df[df["case_pair"].isin(cases)]
        sub = sub[sub["converged"] == True]  # noqa: E712
        return [row.to_dict() for _, row in sub.iterrows()]

    def _store_elites(self) -> list[tuple[str | None, Pattern]]:
        """Top verified store rows to seed elite-mutation parents (plan sec. 4.6).

        **Feasibility-first ordering.**  The elite parents are the neighbourhood
        the pool mutates for the exploit slots, so verified-*feasible* rows must
        fill the elite slots before any infeasible row.  Ranking purely by
        cycle-distance-to-target (the previous behaviour) let converged-but-
        infeasible patterns whose cyclen happened to sit near the target crowd
        out the genuinely feasible basin — in the M5 pilot 0 of the 32 K1_K2
        "elites" were feasible (F_r 1.78–3.52) even though the store held 70
        verified-feasible rows (F_r 1.54), so no feasible-neighbourhood child was
        ever generated.  Feasible rows come first (best objective first), then
        the best infeasible converged rows only as backfill when short.
        """

        rows = self._case_store_rows(converged=True) + self._elite_seed_rows()
        feasible_scored: list[tuple[float, str, Pattern]] = []
        other_scored: list[tuple[float, str, Pattern]] = []
        for row in rows:
            try:
                pat = unpack_pattern(str(row["pattern"]))
            except (ValueError, KeyError):
                continue
            obj = self._campaign_objective(row)
            entry = (obj, str(row["record_id"]), pat)
            (feasible_scored if self._is_feasible(row) else other_scored).append(entry)
        feasible_scored.sort(key=lambda t: t[0], reverse=True)
        other_scored.sort(key=lambda t: t[0], reverse=True)
        ranked = feasible_scored + other_scored
        elites = [(rid, pat) for _, rid, pat in ranked[: self.search.elite_top_k]]
        # add the campaign's own verified feasible/best patterns.
        for row in self.campaign_rows:
            if self._is_feasible(row):
                try:
                    elites.append((str(row["record_id"]), unpack_pattern(str(row["pattern"]))))
                except (ValueError, KeyError):
                    continue
        return elites

    def _near_miss_parents(self) -> list[tuple[str | None, Pattern]]:
        """Parents that seed the tight (n_moves=1) elite-mutation arm (plan 4.6).

        These almost-good boundary samples seed the elite-mutation arm with a
        small-move trust-region bias, so the pool concentrates fresh 1-move probes
        around the best verified patterns even before a fully-feasible label
        exists — directly attacking the flat per-wave min the pilot showed.

        **Objective-aware** (program §10 STOP).  "Almost good" is a property of the
        objective, not of F_r: seeding by ``f_r <= near_miss_f_r`` in EVERY mode
        pointed the tightest local search a ``flat_power`` campaign performs at
        low-F_r parents — the one steering the flatness switch was supposed to
        retire, and the most damaging place to leave it because these parents get
        the smallest, most exploitative moves.  ``flat_power`` therefore seeds
        from its own scalar (:meth:`_flat_near_miss_parents`); every other
        objective keeps the F_r bound unchanged.
        """

        if float(self.search.near_miss_f_r) <= 0.0:
            return []                        # the arm's off switch, all modes
        if self.objective == "flat_power":
            return self._flat_near_miss_parents()
        bound = float(self.search.near_miss_f_r)
        out: list[tuple[str | None, Pattern]] = []
        seen: set[str] = set()
        for row in self.campaign_rows:
            if not row.get("converged"):
                continue
            f_r = row.get("f_r")
            try:
                if f_r is None or float(f_r) > bound:
                    continue
            except (TypeError, ValueError):
                continue
            rid = str(row.get("record_id", ""))
            if rid in seen:
                continue
            try:
                pat = unpack_pattern(str(row["pattern"]))
            except (ValueError, KeyError):
                continue
            seen.add(rid)
            out.append((rid or None, pat))
        return out

    def _flat_near_miss_parents(self) -> list[tuple[str | None, Pattern]]:
        """The K flattest verified this-campaign boards (``flat_power``, §10).

        Ranked by the SAME scalar the acquisition and best-tracking use, so the
        tight arm probes around the flattest labelled patterns.  Rows with no
        ``node_peak`` are excluded outright: they have NO objective value (§1.3),
        and admitting them would fill the arm with unranked parents.  ``F_r`` is
        not consulted — it is a safety gate in this mode and gates elsewhere.
        """

        top_k = int(getattr(self.search, "near_miss_top_k", 8))
        if top_k <= 0:
            return []
        scored: list[tuple[float, str, Pattern]] = []
        seen: set[str] = set()
        for row in self.campaign_rows:
            if not row.get("converged") or not self.has_flat_label(row):
                continue
            rid = str(row.get("record_id", ""))
            if rid in seen:
                continue
            obj = self._flat_objective(row)
            if not math.isfinite(obj):
                continue
            try:
                pat = unpack_pattern(str(row["pattern"]))
            except (ValueError, KeyError):
                continue
            seen.add(rid)
            scored.append((obj, rid, pat))
        # flattest first (higher scalar == flatter), stable on ties by record_id.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(rid or None, pat) for _obj, rid, pat in scored[:top_k]]

    def _replay_rows(self) -> list[dict[str, Any]]:
        df = self._store_df()
        if df is None or not len(df):
            return []
        sub = df[(df["converged"] == True) & (df["valid"] == True)]  # noqa: E712
        sub = sub[sub["f_r"].notna() & sub["cyclen"].notna()]
        if not len(sub):
            return []
        n = min(self.replay_size, len(sub))
        picked = sub.sample(n=n, random_state=self.rng.randint(0, 1_000_000))
        return [row.to_dict() for _, row in picked.iterrows()]

    def _holdout_rows(self) -> list[dict[str, Any]]:
        rows = self._case_store_rows(converged=True)
        if self.objective == "flat_power":
            # The panel's PRIMARY targets are node_peak / map_cov (update.py), so a
            # holdout of unmapped rows leaves both skills NaN and the gate blind on
            # the only axes this mode optimizes.  Require the flatness label.
            mapped = [r for r in rows if self.has_flat_label(r)]
            if len(mapped) >= 2:               # Spearman needs 2 points
                rows = mapped
            else:
                if not self._warned_unmapped_holdout:
                    self._warned_unmapped_holdout = True
                    self._log(
                        "[optimize][WARNING] this case has < 2 store rows carrying "
                        "a node_peak label; the frozen holdout falls back to the "
                        "unmapped rows and the flatness panel will report NaN "
                        "skill (the gate can then neither veto nor halt on "
                        "flatness — it will read 'explore')"
                    )
                rows = [r for r in rows
                        if r.get("f_r") is not None and r.get("cyclen") is not None]
        else:
            rows = [r for r in rows
                    if r.get("f_r") is not None and r.get("cyclen") is not None]
        # frozen: deterministic order, capped.
        rows.sort(key=lambda r: str(r["record_id"]))
        return rows[: self.acq.holdout_size]

    # -- f_xy serve-sigma mode (defect D3) ---------------------------------- #
    def _fxy_sigma_mode(self) -> str:
        """``"barred"`` / ``"head"`` — which f_xy SIGMA the served model gives out.

        The G4 bar is a property of the CHECKPOINT (``ensemble.json`` ->
        ``fxy_head.serve_sigma``), so it moves whenever the served weights are
        replaced — construction, a resume that reloads a persisted champion, a
        wave champion swap.  A plain attribute read, never a model call, so it is
        free to take at every one of those moments.
        """
        return ("barred" if acq.fxy_sigma_barred(self.model) else "head")

    def _assert_fxy_sigma_mode(self, context: str) -> None:
        """Log the sigma mode ACTUALLY in effect and refuse a silent un-barring.

        Defect D3 (``minfxy_T6T4_f121_r1`` §9): ``_save_champion`` wrote per-wave
        checkpoints without ``ensemble.json``, so a ``--resume`` reloading one
        served an `s1j` descendant whose bar had evaporated — the last 12 calls
        ranked on the head's own over-wide sigma, silently, under a pre-registered
        STAMP that said otherwise.  The checkpoint now carries the block
        (:meth:`PosValCnnBackend._save_ensemble_meta`); this is the assertion that
        the carry actually worked, because a bar that fails quietly is the whole
        defect.  Loosening (barred -> head) ABORTS; the harmless direction
        (head -> barred, a strictly narrower UCB) is reported, not fatal.
        """
        mode = self._fxy_sigma_mode()
        launch = getattr(self, "_fxy_sigma_mode_launch", mode)
        self._log(
            f"[optimize][F_xy SIGMA] {context}: serving mode={mode!r} "
            f"(launch={launch!r}) from champion {self.champion_ckpt}"
        )
        if launch == "barred" and mode != "barred":
            raise FxySigmaBarLost(
                f"[{context}] the deck launched on a champion stamping "
                f"fxy_head.serve_sigma='barred' (G4), but the checkpoint now "
                f"served ({self.champion_ckpt}) reports mode={mode!r}: the head's "
                "own sigma would be served un-barred and every F_xy UCB would be "
                "inflated against the registered convention. This is defect D3 of "
                "data/reports/minfxy_T6T4_f121_r1_results_20260830.md: the "
                "checkpoint was written without its ensemble.json fxy_head block. "
                "Re-save it from the source champion (or point model_dir at the "
                "source) rather than resuming un-barred."
            )
        if launch != "barred" and mode == "barred":
            self._log(
                "[optimize][F_xy SIGMA] the resumed checkpoint asserts a bar the "
                "launch champion did not; the narrower (proxy) width is served."
            )

    # -- resume ------------------------------------------------------------ #
    def _load_state(self) -> bool:
        if not (self.resume and self.state_path.exists()):
            return False
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.wave_index = int(state.get("wave_index", 0))
        self.budget_spent = int(state.get("budget_spent", 0))
        self.consecutive_halts = int(state.get("consecutive_halts", 0))
        self.no_improve = int(state.get("no_improve", 0))
        self.best = state.get("best")
        self.best_overall = state.get("best_overall")
        # D9 licensing accounting is part of the run's state: without it a resumed
        # run reported +0 MASTER calls for a gate that had already spent them, and
        # ``_maybe_post_verify`` re-ran (and re-spent) on every completed run().
        self.post_verify_calls = int(state.get("post_verify_calls", 0) or 0)
        self.post_verify_violators = list(state.get("post_verify_violators") or [])
        self.post_verify_done = bool(state.get("post_verify_done", False))
        # flat_power: the persisted objective values are in the units of the
        # normalizer recorded here.  Migrate them to the current one, or refuse.
        self._reconcile_flat_scale(state.get("flat_scale"))
        self.champion_ckpt = str(state.get("champion_ckpt", self.champion_ckpt))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            version, internal, gauss = rng_state
            self.rng.setstate((version, tuple(internal), gauss))
        # reconstruct verified set + accumulated rows from labels.jsonl.
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._absorb_label(row)
        # A resumed run serves the PERSISTED champion, which need not be the one
        # construction proved the calibration against (state.json can name a
        # champion written by a later wave, or by another run entirely).  Re-prove
        # it BEFORE the reload, so a mismatched pair never gets served at all.
        # Restore the persisted staleness FIRST: the re-proof below can itself
        # discover a fine-tuned descendant and set the flag, and loading after it
        # would overwrite that discovery with the older persisted value.  Staleness
        # is one-way (it is only ever cleared by a refit), so restore-then-detect
        # is the correct order.
        self.map_calibration_stale = bool(state.get("map_calibration_stale", False))
        self._require_calibration_model(self.champion_ckpt, context="resume")
        # reload the champion checkpoint if one was persisted mid-run.
        if self.backend_factory is not None and Path(self.champion_ckpt).exists():
            try:
                self.model = self.backend_factory(self.champion_ckpt)
            except Exception as exc:  # noqa: BLE001
                # Falling through here resumes on the CONSTRUCTION-TIME model, not
                # the checkpoint the campaign spent its budget producing — the run
                # continues and looks healthy while silently discarding every
                # wave's fine-tuning.
                # ponytail: log-only — abort-on-resume is the upgrade path.
                self._log(f"[optimize][WARNING] champion reload FAILED -> "
                          f"{self.champion_ckpt}: {type(exc).__name__}: {exc}; "
                          f"resuming on the construction-time model")
        # State the sigma mode the resumed run is ACTUALLY serving, and refuse a
        # bar that the reload dropped (defect D3).
        self._assert_fxy_sigma_mode("resume")
        return True

    def _reconcile_flat_scale(self, stored: Any) -> None:
        """Make a resumed ``flat_power`` best comparable with this run's scalar.

        A ``flat_power`` ``best["objective"]`` is a number in the units of the
        cell's normalizer, so a re-fit ``flat_scale.json`` (or a changed
        ``flatpower_*_scale`` / ``w_cov`` in the deck) silently redefines it: the
        resumed run would compare fresh values against a stale-unit incumbent and
        either freeze (stale looks unbeatable) or reset (stale looks terrible).

        MIGRATE when the stored best carries its ``node_peak`` label — the scalar
        is then recomputable EXACTLY under the new normalizer.  REFUSE, loudly,
        when it does not: silently trusting an un-migratable number is precisely
        the failure this guard exists for.
        """

        if self.objective != "flat_power":
            return
        if _FS.identity_matches(stored, self.flat_scale_id):
            return
        migrated: list[str] = []
        for name in ("best", "best_overall"):
            entry = getattr(self, name)
            if not isinstance(entry, dict) or entry.get("objective") is None:
                continue
            if not self.has_flat_label(entry):
                raise ValueError(
                    f"resume refused: {self.state_path} holds a flat_power "
                    f"{name!r} whose objective was computed under a DIFFERENT "
                    f"normalizer ({stored!r} != {self.flat_scale_id!r}) and which "
                    "carries no node_peak label to recompute it from. Re-fit "
                    "provenance cannot be reconstructed — start a fresh run dir "
                    "or restore the matching flat_scale.json."
                )
            entry["objective"] = self._flat_objective(entry)
            entry["objective_scale"] = self.flat_scale_id
            migrated.append(name)
        if migrated:
            self._log(
                "[optimize][WARNING] flat_power normalizer changed since this run "
                f"was written ({stored!r} -> {self.flat_scale_id!r}); recomputed "
                f"{', '.join(migrated)} objective(s) from the stored node_peak / "
                "map_cov labels so the resumed comparison is in ONE unit"
            )

    def _absorb_label(self, row: dict[str, Any]) -> None:
        rid = str(row.get("record_id", ""))
        if not rid:
            return
        self.ledger_ids.add(rid)
        record = row.get("record")
        if isinstance(record, dict):
            self.campaign_rows.append(record)
            if row.get("slot") == "control":
                self.control_rows.append(record)
            try:
                self.verified_patterns.append(unpack_pattern(str(record["pattern"])))
            except (ValueError, KeyError):
                pass

    def _save_state(self) -> None:
        version, internal, gauss = self.rng.getstate()
        _atomic_json(
            self.state_path,
            {
                "wave_index": self.wave_index,
                "budget": self.budget,
                "budget_spent": self.budget_spent,
                "consecutive_halts": self.consecutive_halts,
                "no_improve": self.no_improve,
                "best": self.best,
                "best_overall": self.best_overall,
                # licensing (D9) accounting — resumed, never re-spent.
                "post_verify_calls": self.post_verify_calls,
                "post_verify_violators": list(self.post_verify_violators),
                "post_verify_done": self.post_verify_done,
                # the normalizer ``best["objective"]`` is expressed in (flat_power
                # only; ``None`` elsewhere).  See FlatScale.identity.
                "flat_scale": self.flat_scale_id,
                "champion_ckpt": self.champion_ckpt,
                # the served weights have drifted off the calibrated checkpoint
                # (fine-tune); resumed so the warning is not re-issued and the
                # run artefacts stay honest about what the correction covers.
                "map_calibration_stale": self.map_calibration_stale,
                "rng_state": [version, list(internal), gauss],
                "case": {"pair": self.ctx.pair, "feed": self.ctx.feed,
                         "e_core": self.ctx.e_core},
            },
        )

    def _write_status(self, status: str) -> None:
        _atomic_json(
            self.run_dir / "status.json",
            {
                "status": status,
                # the objective is what defines this run's FEASIBILITY set and
                # ranking scalar; ``lpopt report <run_dir>`` has no deck, so it
                # reads it from here (report.build_report).
                "objective": self.objective,
                "wave_index": self.wave_index,
                "budget": self.budget,
                "budget_spent": self.budget_spent,
                # Licensing (D9 SDM/MTC gate) MASTER calls are reported SEPARATELY
                # from the search budget — they are not drawn from it, and the sum
                # is the run's true MASTER cost.  Never merged into budget_spent:
                # that number is what the acquisition loop is allowed to spend.
                "post_verify_master_calls": self.post_verify_calls,
                "master_calls_total": self.budget_spent + self.post_verify_calls,
                "post_verify_violators": list(self.post_verify_violators),
                "n_feasible": sum(1 for r in self.campaign_rows if self._is_feasible(r)),
                # ``best`` == best_feasible (null when NO constraint-feasible label
                # exists — e.g. a min_fr run that never reaches F_r<=1.55).
                "best": self.best,
                "best_feasible": self.best,
                # best-by-objective over ALL converged rows (feasible or not): the
                # honest "closest attempt" so a no-feasible run is reported truthfully.
                "best_overall": self.best_overall,
                "dry_run": self.dry_run,
                "case": self.ctx.case_key.label,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    # -- feasibility ------------------------------------------------------- #
    def feasibility_limits(self) -> dict[str, float | None]:
        """The limits THIS run judges feasibility at (:func:`feasibility_limits_for`).

        The single source of truth for the campaign's feasible set — and for the
        report's, which reads it here instead of restating the rule
        (``report._feasible``).  The live ``flat_power`` spec is preferred: it is
        the only place the D1 per-cell map-head bias correction has been applied.
        """
        return feasibility_limits_for(
            self.acq, self.objective,
            fr_gate=(self.flat_power_spec.fr_gate
                     if self.flat_power_spec is not None else None),
        )

    def _is_feasible(self, row: dict[str, Any]) -> bool:
        """SEARCH feasibility of a verified row (:func:`is_feasible_search`).

        Every search-side consumer (elites, best-tracking, ``n_feasible``, the
        outer weights, the wave log) goes through here.  The DELIVERY verdict is
        :meth:`_is_deliverable` and is deliberately a different question.
        """

        if not row.get("converged"):
            return False
        return is_feasible_search(row, self.feasibility_limits())

    def _is_deliverable(self, row: dict[str, Any]) -> bool:
        """DELIVERY verdict of a verified row (:func:`is_deliverable`).

        Strictly narrower than :meth:`_is_feasible`: every gated licensing axis
        must be MEASURED and inside its limit, so an unmeasured ``f_xy`` (or pin
        burnup) is UNKNOWN, not satisfied (review 2026-08-29 §6.4 / P0-03).
        """

        if not row.get("converged"):
            return False
        return is_deliverable(row, self.feasibility_limits())

    def _fuel_charge(self, row: dict[str, Any]) -> float | None:
        """Fresh fuel-economics metric FE of a verified row (min_fuel_cost).

        Computed from the row's canonical fresh composition via
        :func:`lpopt.data.fuel_types.fresh_fuel_charge`; ``None`` when the pattern
        or fuel library cannot resolve it (lower FE is more fuel-economical)."""

        from ..data.fuel_types import fresh_fuel_charge

        pat = row.get("pattern")
        if pat is None or self.fuel is None:
            return None
        try:
            charge, _ = fresh_fuel_charge(
                self.fuel, self.library_id, unpack_pattern(str(pat)).batch_feed()
            )
        except Exception:  # noqa: BLE001 — never let FE lookup abort bookkeeping
            return None
        return charge

    # -- flat_power (flatness-native) helpers ------------------------------- #
    @staticmethod
    def _flat_columns(row: dict[str, Any]) -> tuple[float | None, float | None]:
        """``(node_peak, map_cov)`` of a verified row — the harvested labels."""
        out: list[float | None] = []
        for key in ("node_peak", "map_cov"):
            v = row.get(key)
            try:
                f = float(v) if v is not None else None
            except (TypeError, ValueError):
                f = None
            out.append(f if f is not None and math.isfinite(f) else None)
        return out[0], out[1]

    def has_flat_label(self, row: dict[str, Any]) -> bool:
        """True when the row carries the PRIMARY flatness label (``node_peak``).

        The distinction program §1.3 demands: a row without this has NO objective
        value, which is a different fact from having a bad one.
        """
        peak, _cov = self._flat_columns(row)
        return peak is not None

    def _flat_objective(self, row: dict[str, Any]) -> float:
        """``-( node_peak/PEAK_SCALE + w_cov * map_cov/COV_SCALE )`` (program §1.3).

        Uses this campaign cell's resolved scales, so the verified-row objective
        and the acquisition scalar are the SAME function of the same quantities —
        best-tracking and candidate ranking cannot disagree.
        """
        peak, cov = self._flat_columns(row)
        if peak is None:
            return float("-inf")           # NO LABEL (not "worse"); see §1.3
        return self._flat_eff_scale.scalar_one(peak, cov, w_cov=self.flat_w_cov)

    def _map_harvest_rate(self, rows: Sequence[dict[str, Any]]) -> float | None:
        """Fraction of CONVERGED rows carrying a flatness label (``None`` if n=0)."""
        conv = [r for r in rows if r.get("converged")]
        if not conv:
            return None
        return sum(1 for r in conv if self.has_flat_label(r)) / float(len(conv))

    def _campaign_objective(self, row: dict[str, Any]) -> float:
        """Higher-is-better campaign objective for a verified row.

        target_cycle: ``−|cyclen − target|`` (closeness to the target window).
        max_cycle_min_fr: ``cyclen − λ·F_r`` (maximize cyclen, minimize F_r).
        min_fr_max_cycle: strict lexicographic ``(F_r asc, cyclen desc)`` encoded as
        ``−F_r·1e6 + cyclen`` (F_r dominates for any realistic cyclen < 1e6, cyclen
        is the tie-break) — minimize F_r first, then maximize cyclen.
        """

        # fr_boundary: F_r is the SOLE objective (minimize) and cyclen is recorded
        # but NEVER gated — placed BEFORE the cyclen-None guard so a converged row
        # missing cyclen stays visible to best-tracking and elites.
        if self.objective == "fr_boundary":
            f_r = row.get("f_r")
            if f_r is None:
                return float("-inf")
            return -float(f_r)
        if self.objective == "flat_power":
            # FLATNESS-NATIVE (program §1.3): the SAME scalar the acquisition
            # ranks by, applied to the record's harvested flatness columns.
            #
            #   -( node_peak / PEAK_SCALE + w_cov * map_cov / COV_SCALE )
            #
            # This replaces `-f_r*1e3 - f_q`, which was a fallback from before the
            # columns existed and meant the node-peak campaign was ranking its own
            # MASTER labels by F_r — the exact scalar the user retired.  A row with
            # no map returns -inf as "NO LABEL"; `_wave_improved` distinguishes
            # that from "worse", and `_map_harvest_rate` hard-aborts a campaign
            # whose maps stop arriving.  Placed before the cyclen-None guard so a
            # converged row missing cyclen stays visible.
            return self._flat_objective(row)
        cyclen = row.get("cyclen")
        if cyclen is None:
            return float("-inf")
        if self.objective == "max_cycle_min_fr":
            f_r = row.get("f_r")
            if f_r is None:
                return float("-inf")
            lam = self.max_cycle_spec.lam if self.max_cycle_spec else 100.0
            return float(cyclen) - float(lam) * float(f_r)
        if self.objective == "min_fr_max_cycle":
            f_r = row.get("f_r")
            if f_r is None:
                return float("-inf")
            return -float(f_r) * 1.0e6 + float(cyclen)
        if self.objective == "min_fxy":
            # The SAME strict lexicographic encoding as min_fr_max_cycle with F_xy
            # in F_r's place: (F_xy asc, cyclen desc).  A converged row with NO
            # MEASURED f_xy is NOT "worse" — it is UNSCORABLE on this objective's
            # axis, exactly as a flat_power row with no map is (§1.3), so it
            # returns -inf and best-tracking simply cannot see it.
            f_xy = row.get("f_xy")
            if _is_missing(f_xy):
                return float("-inf")
            return -float(f_xy) * 1.0e6 + float(cyclen)
        if self.objective == "min_fuel_cost":
            # minimize FE (primary) with F_r the subordinate tie-break — the exact
            # exploit scalar (−FE − λ_Fr·F_r) so verified best-tracking agrees with
            # acquisition ranking.  FE is position-invariant (≈ constant within a
            # cell), so F_r orders the within-cell ties; the outer cell-race moves FE.
            f_r = row.get("f_r")
            fe = self._fuel_charge(row)
            if f_r is None or fe is None:
                return float("-inf")
            lam = self.min_fuel_cost_spec.lam_fr if self.min_fuel_cost_spec else 20.0
            return -float(fe) - float(lam) * float(f_r)
        return _objective(cyclen, self.acq.cycle_target_efpd)

    def _is_on_target(self, row: dict[str, Any]) -> bool:
        if not self._is_feasible(row):
            return False
        # max_cycle_min_fr / min_fr_max_cycle have no target window: any
        # constraint-feasible label is a usable result (drives the early-stop
        # "have a keeper" condition; for min_fr, feasible == F_r<=1.55 too).
        if self.objective in ("max_cycle_min_fr", "min_fr_max_cycle", "min_fxy",
                              "min_fuel_cost", "fr_boundary", "flat_power"):
            return True
        cyclen = row.get("cyclen")
        return cyclen is not None and abs(
            float(cyclen) - self.acq.cycle_target_efpd
        ) <= self.acq.cycle_tolerance_efpd

    # -- main run ---------------------------------------------------------- #
    def run(self) -> CampaignResult:
        # deck copy (input provenance, plan sec. 7).
        if self.cfg.source_path and Path(self.cfg.source_path).exists():
            try:
                shutil.copy2(self.cfg.source_path, self.run_dir / "input_deck.inp")
            except OSError:
                pass

        self._load_state()

        if self.budget <= 0:
            return self._proposals_only()

        self._log(
            f"[optimize] campaign {self.run_dir.name} case={self.ctx.case_key.label} "
            f"budget={self.budget} spent={self.budget_spent} dry_run={self.dry_run}"
        )
        self._write_status("running")

        early_stop = False
        while self.budget_spent < self.budget:
            if self.max_waves is not None and self.wave_index >= self.max_waves:
                self._log(f"[optimize] reached max_waves={self.max_waves}; pausing (resumable)")
                self._write_status("paused")
                return self._result("paused")

            remaining = self.budget - self.budget_spent
            size = min(self.acq.wave_size, remaining)
            reserve = early_stop or remaining < self.acq.wave_size

            before = self.budget_spent
            try:
                report = self._run_wave(size, reserve=reserve)
            except MapHarvestAbort as exc:
                # The wave's labels are already committed; stop with the cause
                # named rather than spending the rest of the budget on rows the
                # objective cannot score (program §1.3).
                self.wave_index += 1
                self._save_state()
                self._log(f"[optimize][MAP_HARVEST_ABORT] {exc}")
                self._write_status("map_harvest_abort")
                result = self._result("map_harvest_abort")
                self._render_report(result)
                return result
            self.wave_reports.append(report)
            self.wave_index += 1
            self._save_state()
            self._write_status("running")

            # safety: a wave that evaluated nothing cannot make progress (the pool
            # yielded no in-region candidates); stop rather than spin.
            if self.budget_spent == before:
                self._log("[optimize][WARNING] wave produced no evaluations; stopping")
                self._write_status("stalled")
                result = self._result("stalled")
                self._render_report(result)
                return result

            if report.gate_mode == "halt":
                self.consecutive_halts += 1
            else:
                self.consecutive_halts = 0
            if self.consecutive_halts >= 2:
                return self._model_halt()

            if reserve:
                break
            # early-stop rule (plan sec. 4.6): >= min_waves, an on-target
            # feasible exists, and no best-objective improvement for N waves.
            have_on_target = any(self._is_on_target(r) for r in self.campaign_rows)
            if (
                self.early_stop_enabled
                and self.wave_index >= self.acq.min_waves_before_stop
                and have_on_target
                and self.no_improve >= self.acq.no_improve_waves
            ):
                early_stop = True

        self._write_status("complete")
        result = self._result("complete")
        self._render_report(result)
        return result

    def _prev_top_seeds(
        self, claimed: set[str]
    ) -> list[tuple[str | None, Pattern]]:
        """``self.prev_top`` as ``(lineage anchor, Pattern)`` build_pool seeds.

        Each carried-forward candidate is resolved to the nearest ancestor that is
        a REAL store row (:func:`~.verify.lineage_anchor`); by now the ledger holds
        last wave's verified rows, so a top candidate that actually reached MASTER
        anchors on itself rather than on its parent.

        ``claimed`` is the set of parent ids the earlier seed lists (near-miss,
        store elites) already occupy.  ``build_pool`` deduplicates its parent set
        by that id -- a rule written when the id WAS board identity -- so once
        several distinct boards legitimately share one anchor, passing the anchor
        on each would make build_pool silently drop all but the first and shrink
        the elite arm.  The duplicates therefore travel with ``None``: they stay
        parents exactly as they did before (``None`` is never deduplicated), and
        the lineage is recorded on the one seed that can carry it unambiguously.
        Omitting a true edge costs the corpus one step; inventing a phantom one
        cost it every step, which is the defect this replaces.
        """

        seeds: list[tuple[str | None, Pattern]] = []
        seen = set(claimed)
        for cand in self.prev_top:
            rid = lineage_anchor(cand, self.ledger_ids)
            if rid is not None and rid in seen:
                rid = None
            elif rid is not None:
                seen.add(rid)
            seeds.append((rid, cand.pattern))
        return seeds

    def _run_wave(self, size: int, *, reserve: bool) -> WaveReport:
        # 1. pool
        store_elites = self._store_elites()
        near_miss = self._near_miss_parents()
        claimed = {rid for rid, _ in (*near_miss, *store_elites) if rid is not None}
        # What the policy prior actually did this wave (mode / version / fallback,
        # plus shadow scores under shadow_v2).  Filled by build_pool and written
        # into selection.json, so a readout can never mistake a WARNING fallback
        # for a policy-on wave (external review section 6.12).
        self._pool_meta: dict[str, Any] = {}
        pool = build_pool(
            self.ctx, self.model, store_elites, self.ledger_ids,
            self.rng, self.cfg, wave_index=self.wave_index,
            prev_top=self._prev_top_seeds(claimed),
            near_miss_parents=near_miss,
            size=self.pool_size,
            meta=self._pool_meta,
        )
        # 2. score + gate.  ``have_feasible`` selects the τ schedule (feasibility-
        # first until a verified constraint-feasible label exists); it also selects
        # the target-cycle exploit-score stage.  ``max_cycle_min_fr`` scores with a
        # dedicated cyclen/F_r pool scorer (F_r ungated) and hill-climbs on it.
        have_feasible = any(self._is_feasible(r) for r in self.campaign_rows)
        incumbent = None if self.best is None else self.best.get("distance")
        tie_epsilon = float(self.acq.tie_epsilon)
        if self.objective == "max_cycle_min_fr":
            spec = self.max_cycle_spec
            scored = acq.score_pool_max_cycle(
                self.model, self.ctx, pool, spec, self.trust_region,
                tie_epsilon=tie_epsilon,
            )
            scored = acq.local_search(
                self.model, self.ctx, scored, None, None,
                self.trust_region, self.local_search_cfg, self.rng, self.ledger_ids,
                tie_epsilon=tie_epsilon,
                score_fn=lambda nb: acq.score_pool_max_cycle(
                    self.model, self.ctx, nb, spec, self.trust_region,
                    tie_epsilon=tie_epsilon,
                ),
            )
        elif self.objective == "min_fr_max_cycle":
            spec = self.min_fr_spec
            scored = acq.score_pool_min_fr(
                self.model, self.ctx, pool, spec, self.trust_region,
                tie_epsilon=tie_epsilon,
            )
            scored = acq.local_search(
                self.model, self.ctx, scored, None, None,
                self.trust_region, self.local_search_cfg, self.rng, self.ledger_ids,
                tie_epsilon=tie_epsilon,
                score_fn=lambda nb: acq.score_pool_min_fr(
                    self.model, self.ctx, nb, spec, self.trust_region,
                    tie_epsilon=tie_epsilon,
                ),
            )
        elif self.objective == "min_fxy":
            spec = self.min_fxy_spec
            scored = acq.score_pool_min_fxy(
                self.model, self.ctx, pool, spec, self.trust_region,
                tie_epsilon=tie_epsilon,
            )
            scored = acq.local_search(
                self.model, self.ctx, scored, None, None,
                self.trust_region, self.local_search_cfg, self.rng, self.ledger_ids,
                tie_epsilon=tie_epsilon,
                score_fn=lambda nb: acq.score_pool_min_fxy(
                    self.model, self.ctx, nb, spec, self.trust_region,
                    tie_epsilon=tie_epsilon,
                ),
            )
        elif self.objective == "min_fuel_cost":
            spec = self.min_fuel_cost_spec
            scored = acq.score_pool_min_fuel_cost(
                self.model, self.ctx, pool, spec, self.trust_region,
                tie_epsilon=tie_epsilon, fuel=self.fuel, library_id=self.library_id,
            )
            scored = acq.local_search(
                self.model, self.ctx, scored, None, None,
                self.trust_region, self.local_search_cfg, self.rng, self.ledger_ids,
                tie_epsilon=tie_epsilon,
                score_fn=lambda nb: acq.score_pool_min_fuel_cost(
                    self.model, self.ctx, nb, spec, self.trust_region,
                    tie_epsilon=tie_epsilon, fuel=self.fuel, library_id=self.library_id,
                ),
            )
        elif self.objective == "fr_boundary":
            spec = self.fr_boundary_spec
            scored = acq.score_pool_fr_boundary(
                self.model, self.ctx, pool, spec, self.trust_region,
                tie_epsilon=tie_epsilon,
            )
            scored = acq.local_search(
                self.model, self.ctx, scored, None, None,
                self.trust_region, self.local_search_cfg, self.rng, self.ledger_ids,
                tie_epsilon=tie_epsilon,
                score_fn=lambda nb: acq.score_pool_fr_boundary(
                    self.model, self.ctx, nb, spec, self.trust_region,
                    tie_epsilon=tie_epsilon,
                ),
            )
        elif self.objective == "flat_power":
            spec = self.flat_power_spec
            scored = acq.score_pool_flat_power(
                self.model, self.ctx, pool, spec, self.trust_region,
                tie_epsilon=tie_epsilon,
            )
            scored = acq.local_search(
                self.model, self.ctx, scored, None, None,
                self.trust_region, self.local_search_cfg, self.rng, self.ledger_ids,
                tie_epsilon=tie_epsilon,
                score_fn=lambda nb: acq.score_pool_flat_power(
                    self.model, self.ctx, nb, spec, self.trust_region,
                    tie_epsilon=tie_epsilon,
                ),
            )
        else:
            if self.reward_model is None:
                boot = self.model.predict([c.pattern for c in pool], self.ctx.case_key, self.ctx.e_core or 0.0)
                self.reward_model = acq.build_reward_model(
                    self.ctx, [c.pattern for c in pool], boot, self.constraints
                )
            scored = acq.score_pool(
                self.model, self.ctx, pool, self.reward_model, self.constraints,
                self.trust_region, incumbent_distance=incumbent, have_feasible=have_feasible,
                tie_epsilon=tie_epsilon,
            )
            # 3. local-search refinement (hill-climbs on the exploit score, not acq).
            scored = acq.local_search(
                self.model, self.ctx, scored, self.reward_model, self.constraints,
                self.trust_region, self.local_search_cfg, self.rng, self.ledger_ids,
                incumbent_distance=incumbent, have_feasible=have_feasible,
                tie_epsilon=tie_epsilon,
            )
        # 3b. SAFETY SHIELD (review §6.5 P0-04 / §8.5).  Runs on the FINISHED pool
        # — after local search, before the elite carry-over and the wave
        # composition — so an escalated candidate loses its exploit tier in BOTH
        # places at once and a rejected one is gone from both.  Inert (and skipped
        # entirely) at the shipped defaults.
        self._shield_report = {}
        if self.safety_shield.active:
            scored, self._shield_report = acq.apply_safety_shield(
                self.model, self.ctx, scored, self.safety_shield,
                self.feasibility_limits(),
            )
            r = self._shield_report
            if r.get("ood_rejected") or r.get("conformal_rejected") or r.get("ood_escalated"):
                self._log(
                    f"[optimize][shield] wave {self.wave_index}: "
                    f"ood({r['ood_policy']}) flagged={r['ood_flagged']} "
                    f"escalated={r['ood_escalated']} rejected={r['ood_rejected']}; "
                    f"conformal rejected={r['conformal_rejected']} "
                    f"{r.get('conformal_rejected_by_axis') or ''}; "
                    f"pool {r['n_candidates']} -> {r['n_remaining']}")

        # keep the top exploit-ranked patterns for the next wave's elite parents
        # (ranking on acq would seed next wave from high-σ OOD candidates).
        # The CANDIDATES are kept, not their record_ids: most of them are never
        # verified, so their ids name no store row and stamping one on a child
        # minted a dangling parent_record_id.  _run_wave resolves each to a real
        # lineage anchor when it consumes them next wave.
        top_order = np.argsort(-scored.rank)[: self.search.elite_top_k]
        self.prev_top = [
            scored.candidates[int(i)]
            for i in top_order
            if np.isfinite(scored.exploit[int(i)])
        ]

        # 4. compose
        tau = acq.tau_schedule(scored, self.acq.tau0, have_feasible=have_feasible)
        if reserve:
            n_exploit, n_explore, n_control = size, 0, 0
        else:
            n_exploit, n_explore, n_control = self.acq.exploit, self.acq.explore, self.acq.control
        slots = acq.compose_wave(
            scored, self.verified_patterns, self.rng, size=size,
            n_exploit=n_exploit, n_explore=n_explore, n_control=n_control,
            tau=tau, hamming_min=self.acq.hamming_min,
            exploit_verified_hamming=self.acq.exploit_verified_hamming,
        )

        # 5. verify
        resolved = self.resolver.resolve(self.ctx.case_key)
        entries: list[tuple[WaveEntry, acq.WaveSlot]] = []
        for wslot in slots:
            cand = scored.candidates[wslot.index]
            meta = {
                "e_core": self.ctx.e_core, "generator": cand.origin, "slot": wslot.slot,
                "parent_record_id": cand.parent_record_id, "record_id": cand.record_id,
            }
            entries.append((WaveEntry(cand.pattern, self.ctx.case_key, resolved, meta), wslot))
        # DELIVERY dossier uncertainty fields (review §8.5), evaluated for the
        # SELECTED candidates in EVERY mode — including the default report-only
        # one.  The shield decides what the search DOES about OOD/conformal; the
        # dossier must state it either way, so a hand-off can never present an
        # unscreened row as a clean one.  Cheap: one guard probe + one interval
        # call over at most ``wave_size`` patterns.
        sel_flags, sel_unfit = self._selected_safety(
            [scored.candidates[w.index].pattern for _, w in entries])
        #: the guard verdict per SELECTED slot, in ``entries`` order — read back by
        #: :meth:`_write_wave_artifacts` so ``selection.json`` and the delivery
        #: dossier can never disagree about the same board.
        self._sel_flags: list[bool | None] = list(sel_flags)
        outcomes = self.verifier.evaluate_wave([e for e, _ in entries])

        # 6. store rows + ledger + archive
        records: list[CanonicalRecord] = []
        wave_rows: list[dict[str, Any]] = []
        wave_maps: dict[str, Any] = {}          # {record_id: EDIT5 (4,9,9)} if harvested
        converged = feasible = on_target = errors = 0
        # first DISTINCT failure strings this wave — 100 chains that all die at
        # staging otherwise surface only as conv=0, which reads as "the search is
        # hard" instead of "nothing ran" (ECC audit 2026-08-12).
        wave_failures: list[str] = []
        for _i, ((entry, wslot), outcome) in enumerate(
                zip(entries, outcomes, strict=True)):
            cand = scored.candidates[wslot.index]
            record = outcome_to_record(
                outcome, dataset="P", library_id=self.library_id,
                stratum=f"campaign_{wslot.slot}", generator=cand.origin,
                parent_record_id=cand.parent_record_id, campaign=self.run_dir.name,
                e_core=self.ctx.e_core, e_split=None, deck_knobs=PRODUCE_DECK_KNOBS,
            )
            records.append(record)
            self._row_safety[str(record.record_id)] = {
                "ood_flag": sel_flags[_i],
                "conformal_unfit_axes": list(sel_unfit[_i]),
            }
            if getattr(outcome, "maps", None) is not None:
                wave_maps[record.record_id] = outcome.maps
            # High-res siblings under suffixed keys (legacy (4,9,9) key untouched).
            _hires = getattr(outcome, "maps_hires", None) or {}
            for _sfx in ("traj", "axial"):
                _arr = _hires.get(_sfx)
                if _arr is not None:
                    wave_maps[f"{record.record_id}__{_sfx}"] = _arr
            row = record.to_record()
            wave_rows.append(row)
            self.ledger_ids.add(record.record_id)
            self.campaign_rows.append(row)
            self.verified_patterns.append(cand.pattern)
            if wslot.slot == "control":
                self.control_rows.append(row)
            self.budget_spent += 1
            if outcome.status == "converged":
                converged += 1
                self._maybe_update_overall(row)   # honesty channel (all converged)
                # D9: capture this candidate's branch assets while they exist.
                self._record_sdm_mtc_target(record.record_id, outcome)
            elif outcome.status == "error":
                errors += 1
                fail = str(getattr(outcome, "failure", "") or "unknown")
                if fail not in wave_failures:
                    wave_failures.append(fail)
            if self._is_feasible(row):
                feasible += 1
                self._maybe_update_best(row, cand)
                self._archive(cand, outcome)
            if self._is_on_target(row):
                on_target += 1
            _append_jsonl(
                self.labels_path,
                {
                    "wave": self.wave_index, "slot": wslot.slot, "origin": cand.origin,
                    "record_id": record.record_id, "status": outcome.status,
                    "feasible": self._is_feasible(row), "on_target": self._is_on_target(row),
                    "record": row,
                },
            )
        self.store.write_records(records)
        if wave_maps:
            # EDIT5 assembly maps of this wave's converged candidates -> maps.npz
            # (keyed by record_id); pull+merge_store folds them into the home store.
            # A concurrent READER can hold maps.npz open on Windows and refuse the
            # atomic rename even after ``_atomic_write``'s ~135 s of retries; skip
            # this wave's maps rather than abort the campaign (labels are already
            # committed to records.parquet).
            try:
                self.store.write_maps(wave_maps, append=True)
            except (PermissionError, OSError) as exc:
                self.maps_skipped_waves += 1
                self.maps_skipped_records += len(wave_maps)
                safe_print(f"[optimize] WARNING maps write failed ({type(exc).__name__}); "
                      f"SKIPPED {len(wave_maps)} map(s) this wave — labels unaffected. "
                      f"cumulative: {self.maps_skipped_records} map(s) in "
                      f"{self.maps_skipped_waves} wave(s)")

        # 7. online update (fine-tune + gate)
        updater = WaveUpdater(
            self.ctx.case_key, self.ctx.e_core or 0.0, self._holdout_rows(),
            epsilon=self.acq.gate_epsilon,
            skill_objective=self.acq.gate_skill_objective,
            skill_halt=self.acq.gate_skill_halt,
            finetune_epochs=self.finetune_epochs,
            new_weight=self.acq.finetune_new_weight,
            # the panels judge the model on the axes THIS objective optimizes
            # (program §10: flat_power accepts/halts on node_peak + map_cov, not
            # on the F_r skill it retired) — see update.panel_targets.
            objective=self.objective,
        )
        gate = updater.update(
            self.model, wave_rows, self._replay_rows(),
            [r for r in self.campaign_rows if r.get("converged")],
            self.control_rows, seed=self.rng.randint(0, 1_000_000),
        )
        for feed, ec in [(self.ctx.feed, self.ctx.e_core)] * converged:
            self.trust_region.observe(feed, ec)
        if gate.accepted:
            # The served weights have just been REPLACED (the challenger was
            # retained).  Re-prove the calibration against the champion that is
            # now served, and report the fine-tune drift the fingerprint cannot
            # see — construction-time proof does not cover this checkpoint.
            self.champion_ckpt = str(self._save_champion())
            self._require_calibration_model(
                self.champion_ckpt, context=f"wave {self.wave_index} champion swap")
        self._note_gate_weight_drift(gate)

        # 8. wave artefacts
        self._write_wave_artifacts(self.wave_index, entries, outcomes, scored, gate, tau)
        self._log_event(self.wave_index, wave_rows, gate, converged, feasible, on_target)

        # 9. stopping bookkeeping
        best_obj = self.best["objective"] if self.best else None
        improved = self._wave_improved(best_obj)
        # program §1.3: "no label" is NOT "no improvement".  A flat_power wave
        # whose maps never arrived cannot be judged, so it must not tick the
        # early-stop counter — otherwise a harvest outage reads as convergence.
        if improved:
            self.no_improve = 0
        elif self._wave_judgeable(wave_rows):
            self.no_improve += 1
        else:
            self._log("[optimize][WARNING] wave produced converged rows but NO "
                      "flatness label; not counted as a no-improvement wave "
                      "(program §1.3: 'no label' != 'worse')")

        # per-wave MAP HARVEST RATE + hard abort (program §1.3).
        harvest = self._map_harvest_rate(wave_rows)
        if harvest is not None:
            self.map_harvest_rates.append(round(float(harvest), 4))
        self._check_map_harvest(harvest)

        if self.progress:
            # errors>0 appends ` err=<n> [<first distinct failure>]`; a clean wave's
            # line stays byte-identical to the pre-2026-08-12 format.
            err_txt = (f" err={errors} [{wave_failures[0][:80]}]"
                       if errors else "")
            self._log(
                f"[optimize] wave {self.wave_index:>2}{'*' if reserve else ' '} "
                f"size={size} spent={self.budget_spent}/{self.budget} | "
                f"conv={converged} feas={feasible} on_target={on_target} | "
                f"gate={gate.mode}{'+' if gate.accepted else '-'} tau={tau:.2f} | "
                f"best={self.best['cyclen'] if self.best else None}{err_txt}"
            )

        return WaveReport(
            wave=self.wave_index, reserve=reserve, size=size,
            budget_spent=self.budget_spent,
            slots={s: sum(1 for _, w in entries if w.slot == s) for s in ("exploit", "explore", "control")},
            converged=converged, feasible=feasible, on_target=on_target,
            best_objective=self.best["objective"] if self.best else None,
            best_cyclen=self.best["cyclen"] if self.best else None,
            gate_mode=gate.mode, gate_accepted=gate.accepted, tau=tau,
            map_harvest=harvest, errors=errors,
            ood_flagged=int(self._shield_report.get("ood_flagged", 0)),
            ood_escalated=int(self._shield_report.get("ood_escalated", 0)),
            ood_rejected=int(self._shield_report.get("ood_rejected", 0)),
            conformal_rejected=int(self._shield_report.get("conformal_rejected", 0)),
        )

    # -- safety shield / delivery uncertainty fields ------------------------ #
    def _selected_safety(
        self, patterns: Sequence[Pattern]
    ) -> tuple[list[bool | None], list[list[str]]]:
        """``(ood_flags, conformal_unfit_axes)`` for the wave's SELECTED patterns.

        ``ood_flags[i]`` is ``None`` when the backend exposes NO feature-OOD guard
        (there is nothing to report, which is not the same as a clean verdict) and
        a bool otherwise.  ``conformal_unfit_axes[i]`` names the GATED licensing
        axes for which this candidate has no finite conformal interval — the
        review's "calibration cell unsupported" state (§8.5).  A champion with no
        ``conformal.json`` therefore lists every gated axis, which is the honest
        report, not a defect.

        Best-effort by construction: this feeds a dossier, so any failure degrades
        to "unknown" rather than aborting a verified wave.
        """
        patterns = list(patterns)
        n = len(patterns)
        flags: list[bool | None] = [None] * n
        unfit: list[list[str]] = [[] for _ in range(n)]
        if n == 0:
            return flags, unfit
        try:
            raw, guard_state, _errors = acq.ood_flags(self.model, patterns)
            if guard_state != "absent":
                flags = [bool(v) for v in raw]
        except Exception:  # noqa: BLE001 — a dossier field never fails a wave
            pass
        try:
            limits = self.feasibility_limits()
            axes = acq.conformal_gate_axes(limits)
            upper = acq.conformal_upper(
                self.model, self.ctx, patterns,
                alpha=float(getattr(self.acq, "conformal_alpha", 0.10)))
            for i in range(n):
                for key, col in axes:
                    bound = (float("nan") if upper is None or upper.shape[0] <= i
                             else float(upper[i, col]))
                    if not math.isfinite(bound):
                        unfit[i].append(key)
        except Exception:  # noqa: BLE001
            pass
        return flags, unfit

    def _row_safety_fields(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """The dossier's OOD / conformal fields for one verified row.

        ``{"ood_flag": bool|None, "conformal_unfit_axes": [...]|None}`` — ``None``
        means NOT EVALUATED (a row carried over by a resume), which the review's
        §8.5 rule treats as a state needing more computation, never as a pass.
        """
        entry = self._row_safety.get(str(row.get("record_id")))
        if entry is None:
            return {"ood_flag": None, "conformal_unfit_axes": None}
        return {"ood_flag": entry.get("ood_flag"),
                "conformal_unfit_axes": list(entry.get("conformal_unfit_axes") or [])}

    # -- best / archive / helpers ------------------------------------------ #
    def _best_dict(self, row: dict[str, Any], obj: float) -> dict[str, Any]:
        distance = (
            None if self.objective in
            ("max_cycle_min_fr", "min_fr_max_cycle", "min_fxy", "min_fuel_cost",
             "fr_boundary", "flat_power")
            else abs(float(row["cyclen"]) - self.acq.cycle_target_efpd)
        )
        # Margin to the F_r limit THIS MODE actually gates at (>=0 feasible, <0 the
        # amount over) — the honest report field for a no-feasible min_fr run.
        #
        # flat_power gates F_r at its own SAFETY GATE (1.70, D1), NOT at
        # acq.f_r_limit (1.55): reporting a flat_power run's margin against 1.55
        # described a limit the mode never applied, so a row at F_r 1.62 was
        # reported as -0.07 "over" while the mode had accepted it.  The licensing
        # number lives in ``compliance_margin`` instead (below), which is what
        # program §2.2 delivery ranking reads.
        f_r = row.get("f_r")
        gate_limit = (
            float(self.flat_power_spec.fr_gate)
            if (self.objective == "flat_power" and self.flat_power_spec is not None)
            else float(self.acq.f_r_limit)
        )
        fr_margin = None if f_r is None else round(gate_limit - float(f_r), 4)
        # program §2.2 (decision D2): headroom to the LICENSING limit 1.55.  This
        # is the delivery-ranking key; it is deliberately NOT in any objective and
        # NOT a feasibility definition.  Verified F_r needs no bias correction.
        c_margin = compliance_margin(f_r)
        if c_margin is not None:
            c_margin = round(c_margin, 4)
        # fresh fuel-economics metric (min_fuel_cost primary objective) — reported
        # so the outer cell-race can compare cells on the verified FE of their best.
        fe = self._fuel_charge(row) if self.objective == "min_fuel_cost" else None
        peak, cov = self._flat_columns(row)
        # MEASURED F_xy (MASTER FXYP) and its headroom.  ``f_xy_limit_applied`` is
        # the gate THIS mode judged at (None where F_xy is a reported column only),
        # while ``compliance_margin_fxy`` is always against the LICENSING 1.65 —
        # the same two-number split ``f_r_margin_to_limit`` / ``compliance_margin``
        # already draws for F_r.
        f_xy = row.get("f_xy")
        fxy_gate = self.feasibility_limits().get("f_xy")
        fxy_margin = (None if (_is_missing(f_xy) or fxy_gate is None)
                      else round(float(fxy_gate) - float(f_xy), 4))
        cfxy = None if _is_missing(f_xy) else compliance_margin_fxy(f_xy)
        if cfxy is not None:
            cfxy = round(cfxy, 4)
        return {
            "record_id": row.get("record_id"),
            "objective": obj,
            # units of ``objective`` (flat_power only; None elsewhere, where the
            # scalar is in physical EFPD / F_r units that cannot be re-fitted).
            "objective_scale": self.flat_scale_id,
            "distance": distance,
            "fuel_cost": fe,
            "cyclen": row.get("cyclen"),
            "f_r": f_r, "f_r_margin_to_limit": fr_margin,
            "f_r_limit_applied": round(gate_limit, 4),
            "compliance_margin": c_margin,
            "compliance_limit": LICENSING_FR_LIMIT,
            "node_peak": peak, "map_cov": cov,
            "cbc_max": row.get("cbc_max"), "f_q": row.get("f_q"),
            "ao_abs": row.get("ao_abs"), "n_cycles": row.get("n_cycles"),
            "max_pin_burnup": row.get("max_pin_burnup"),
            "f_xy": (None if _is_missing(f_xy) else float(f_xy)),
            "f_xya": (None if _is_missing(row.get("f_xya"))
                      else float(row.get("f_xya"))),
            "f_xy_margin_to_limit": fxy_margin,
            "f_xy_limit_applied": (None if fxy_gate is None
                                   else round(float(fxy_gate), 4)),
            "compliance_margin_fxy": cfxy,
            "compliance_limit_fxy": LICENSING_FXY_LIMIT,
            "pattern": row.get("pattern"), "wave": self.wave_index,
            "feasible": self._is_feasible(row),
            # the DELIVERY verdict alongside the SEARCH one, never instead of it
            # (review §6.4): a row can be search-feasible and undeliverable, and
            # ``unknown_axes`` names exactly which measurement is missing.
            "deliverable": self._is_deliverable(row),
            "unknown_axes": list(unknown_axes(row, self.feasibility_limits())),
            # UNCERTAINTY provenance (review §8.5): the OOD verdict on this row's
            # fuel population and the gated axes whose conformal calibration does
            # not cover it.  Reported ALONGSIDE ``deliverable`` and deliberately
            # NOT folded into it: the boolean stays exactly the measured-and-inside
            # -limits predicate, while these two fields make an OOD-flagged or
            # uncalibrated candidate impossible to read as a clean deliverable.
            **self._row_safety_fields(row),
        }

    def _maybe_update_overall(self, row: dict[str, Any]) -> None:
        """Track the best-by-objective row regardless of constraint feasibility.

        This is the honesty channel for min_fr_max_cycle (and harmless bookkeeping
        for the other modes): if no row is ever feasible, ``best`` stays null while
        ``best_overall`` records the closest attempt (lowest F_r + its margin to the
        1.55 limit) — never promoting a violator into the feasible ``best``.
        """

        if not row.get("converged"):
            return
        obj = self._campaign_objective(row)
        if not math.isfinite(obj):
            return
        if self.best_overall is None or obj > self.best_overall["objective"]:
            self.best_overall = self._best_dict(row, obj)

    def _maybe_update_best(self, row: dict[str, Any], cand: Any) -> None:
        obj = self._campaign_objective(row)
        if self.best is None or obj > self.best["objective"]:
            if self.objective in ("fr_boundary", "flat_power", "min_fxy"):
                # Route through _best_dict so the outer race reads only keys that
                # EXIST (max_pin_burnup, feasible, f_r_margin_to_limit,
                # compliance_margin, node_peak/map_cov, distance=None) and a
                # converged cyclen=None row never hits |cyclen−target|.
                self.best = self._best_dict(row, obj)
                return
            distance = (
                None if self.objective == "max_cycle_min_fr"
                else abs(float(row["cyclen"]) - self.acq.cycle_target_efpd)
            )
            self.best = {
                "record_id": row["record_id"],
                "objective": obj,
                "distance": distance,
                "cyclen": row.get("cyclen"),
                "f_r": row.get("f_r"), "cbc_max": row.get("cbc_max"),
                "f_q": row.get("f_q"), "ao_abs": row.get("ao_abs"),
                "n_cycles": row.get("n_cycles"),
                "pattern": row.get("pattern"),
                "wave": self.wave_index,
                # UNCERTAINTY provenance (review §8.5) — the same two fields
                # ``_best_dict`` carries, so EVERY objective's ``best`` states the
                # OOD verdict and the uncalibrated axes, not just the modes that
                # route through the richer dossier.
                **self._row_safety_fields(row),
            }

    def _wave_improved(self, prior_best_obj: float | None) -> bool:
        cur = self.best["objective"] if self.best else None
        if cur is None:
            return False
        if prior_best_obj is None:
            return True
        return cur > prior_best_obj + 1.0e-9

    def _check_map_harvest(self, harvest: float | None) -> None:
        """HARD ABORT a flat_power wave whose map harvest collapsed (program §1.3).

        The store-wide map coverage is 58.8%, so a harvest outage is not
        hypothetical.  Without this check the campaign keeps spending MASTER calls
        producing rows the objective cannot score, and the only symptom is a
        stalled ``no_improve`` counter — i.e. it looks like convergence.  Raising
        is the point: the wave's labels are already committed to the store, so the
        run stops with its data intact and a status that names the cause.
        """
        if self.objective != "flat_power" or harvest is None:
            return
        floor = float(getattr(self.acq, "flatpower_min_map_harvest", 0.5))
        if floor <= 0.0 or harvest >= floor:
            return
        raise MapHarvestAbort(
            f"wave {self.wave_index} map harvest {harvest:.2f} < required "
            f"{floor:.2f}: the flat_power objective is defined on the harvested "
            f"map columns, so this campaign can no longer score what it evaluates. "
            f"Check [verify] harvest_maps and the EDIT5 parse before resuming."
        )

    def _wave_judgeable(self, wave_rows: Sequence[dict[str, Any]]) -> bool:
        """Can this wave's outcome be judged as better/worse at all? (§1.3)

        For ``flat_power`` the objective is defined on the harvested map columns,
        and for ``min_fxy`` on the harvested MAS_OUT ``f_xy`` column, so a wave
        that converged but harvested NEITHER produced no comparable value.
        Counting that as "no improvement" would walk the campaign into the
        early-stop path on a HARVEST failure — the label pipeline breaking would
        look exactly like the search converging.  Every other objective is always
        judgeable (its inputs are unconditional record columns).
        """
        if self.objective == "min_fxy":
            conv = [r for r in wave_rows if r.get("converged")]
            if not conv:
                return True                # nothing converged: a real non-result
            return any(not _is_missing(r.get("f_xy")) for r in conv)
        if self.objective != "flat_power":
            return True
        conv = [r for r in wave_rows if r.get("converged")]
        if not conv:
            return True                    # nothing converged: a real non-result
        return any(self.has_flat_label(r) for r in conv)

    def _archive(self, cand: Any, outcome: Any) -> None:
        fom = outcome.fom
        if fom is None:
            return
        try:
            evaluation = GAEvaluation(
                fom=fom, ncyc=int(outcome.n_cycles or 1), eq_ok=True, feasible=True,
                fitness=self._campaign_objective(
                    {"cyclen": fom.cyclen, "f_r": fom.f_r}
                ), penalty=0.0,
            )
            archive_candidate(
                self.run_dir / "candidates", self.ctx.case_key.folder, cand.pattern,
                evaluation, cand.pattern.to_shf(), "campaign",
                {"origin": cand.origin, "wave": self.wave_index, "record_id": cand.record_id},
            )
        except Exception:  # noqa: BLE001 — archiving must never abort a wave
            pass

    def _save_champion(self) -> Path:
        out = self.run_dir / "models" / f"champion_wave_{self.wave_index:02d}"
        try:
            self.model.save(out)
        except Exception as exc:  # noqa: BLE001
            # The gate ACCEPTED the challenger, so self.model is already the new
            # champion in memory while the served pointer stays on the previous
            # checkpoint — a disk/permission failure here silently un-does the
            # swap for every downstream reader (resume, delivery, calibration).
            # ponytail: log-only — state-flag + resume-refusal is the upgrade path.
            self._log(f"[optimize][WARNING] champion save FAILED -> {out}: "
                      f"{type(exc).__name__}: {exc}; keeping stale pointer "
                      f"{self.champion_ckpt}")
            return Path(self.champion_ckpt)
        # Defect D3: the bar has to survive the WRITE, not only the read.  A wave
        # checkpoint is exactly what a later ``--resume`` reloads, so a missing
        # ``ensemble.json`` fxy_head block here IS the silent un-barring — caught
        # at the moment it is created, naming the wave, instead of 12 calls later
        # in a results readout.
        if self._fxy_sigma_mode_launch == "barred":
            declared = checkpoint_fxy_serve_sigma(out)
            if declared != "barred":
                raise FxySigmaBarLost(
                    f"wave {self.wave_index} champion {out} was written WITHOUT "
                    f"the f_xy serve-sigma bar (declared={declared!r}) while the "
                    "launch champion stamps fxy_head.serve_sigma='barred' (G4). "
                    "A --resume would reload it and serve the head's own "
                    "over-wide sigma. See defect D3, "
                    "data/reports/minfxy_T6T4_f121_r1_results_20260830.md §9."
                )
            self._log(f"[optimize][F_xy SIGMA] wave {self.wave_index} checkpoint "
                      f"{out.name} carries serve_sigma='barred'")
        return out

    def _write_wave_artifacts(self, wave, entries, outcomes, scored, gate, tau) -> None:
        wdir = self.run_dir / "waves" / f"wave_{wave:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        sel_flags = list(getattr(self, "_sel_flags", []))
        fxy_src = str(getattr(scored, "fxy_source", "") or "")

        def _num(arr, i):
            """A JSON-safe scalar from a pool column (``None`` for NaN/absent)."""
            try:
                v = float(np.asarray(arr, dtype=float)[i])
            except (IndexError, TypeError, ValueError):
                return None
            return None if not np.isfinite(v) else round(v, 6)

        selection = [
            {
                "slot": w.slot, "origin": scored.candidates[w.index].origin,
                "record_id": scored.candidates[w.index].record_id,
                "parent_record_id": scored.candidates[w.index].parent_record_id,
                "p_feas": round(float(scored.p_feas[w.index]), 4),
                "acq": round(float(scored.acq[w.index]), 4),
                "exploit": round(float(scored.exploit[w.index]), 4),
                "margin": round(float(scored.margin[w.index]), 4),
                "raw_epi": round(float(scored.raw_epi[w.index]), 4),
                "pred_mean": [round(float(x), 4) for x in scored.mean[w.index]],
                # The OBJECTIVE-AXIS prediction for min_fxy, per candidate, as
                # predict_fxy SERVED it: the mean the exploit scalar is built from
                # and the width its UCB is built from -- under the G4 bar the
                # PROXY width, not the head's, which is exactly the distinction a
                # readout must be able to make without re-loading the wave's
                # checkpoint (defect D-LOG, minfxy_T6T4_f121_r1 section 9).
                # ``None`` (not 0) for every mode that does not predict F_xy.
                "fxy_mean": _num(scored.fxy_mean, w.index),
                "fxy_sigma": _num(scored.fxy_sigma, w.index),
                "fxy_source": (fxy_src or None),
                # SAFETY SHIELD provenance (review §6.5): the serve-time OOD
                # verdict for this board, read from the SAME evaluation the
                # delivery dossier uses so the two can never disagree.  ``None``
                # means the backend exposes no guard — not "clean".  Reported in
                # every mode, including the report-only default; what the POLICY
                # then did about it is the wave's ``shield`` block below.
                "ood_flag": (sel_flags[i] if i < len(sel_flags) else None),
            }
            for i, (_, w) in enumerate(entries)
        ]
        payload: dict[str, Any] = {"wave": wave, "tau": tau, "selection": selection}
        # Per-wave gate accounting (review §6.5 item 5).  Absent when the shield is
        # inert, so a default-deck selection.json is unchanged apart from the
        # per-candidate ``ood_flag``.
        if getattr(self, "_shield_report", None):
            payload["shield"] = dict(self._shield_report)
        # Where the F_xy numbers that ranked this wave came from — "head" (a real
        # predict_fxy) or "proxy" (the interim F_r regression).  Absent for every
        # mode that does not read F_xy, so old readers are unaffected.
        if getattr(scored, "fxy_source", ""):
            payload["fxy_source"] = str(scored.fxy_source)
        payload.update(getattr(self, "_pool_meta", {}))
        _atomic_json(wdir / "selection.json", payload)
        results = [
            {
                "slot": w.slot, "record_id": scored.candidates[w.index].record_id,
                "status": o.status, "n_cycles": o.n_cycles,
                "fom": (o.fom.as_dict() if o.fom is not None else None),
            }
            for (_, w), o in zip(entries, outcomes, strict=True)
        ]
        _atomic_json(wdir / "results.json", {"wave": wave, "gate": gate.as_dict(), "results": results})

    def _log_event(self, wave, wave_rows, gate, converged, feasible, on_target) -> None:
        _append_jsonl(
            self.events_path,
            {
                "type": "wave", "wave": wave, "budget_spent": self.budget_spent,
                "converged": converged, "feasible": feasible, "on_target": on_target,
                "gate_mode": gate.mode, "gate_accepted": gate.accepted,
                "control_spearman": gate.control_spearman,
                "best_cyclen": self.best["cyclen"] if self.best else None,
            },
        )

    def _model_halt(self) -> CampaignResult:
        self._log("[optimize][MODEL_HALT] two consecutive gate halts; attempting sklearn fallback refit")
        fallback_ok = False
        try:
            from ..model.model_sklearn import SklearnBackend

            SklearnBackend.fit_from_store(
                self.main_store_dir, [self.ctx.case_key],
                library_id=self.library_id, extra_rows=self.campaign_rows, max_rows=4000,
            )
            fallback_ok = True
        except Exception as exc:  # noqa: BLE001
            self._log(f"[optimize][MODEL_HALT] fallback refit failed: {exc}")
        status = "MODEL_HALT_fallback_refit" if fallback_ok else "MODEL_HALT_no_fallback"
        self._write_status(status)
        result = self._result(status)
        self._render_report(result)
        return result

    def _proposals_only(self) -> CampaignResult:
        self._log("[optimize] budget=0 -> proposals-only (top-16 candidates, no evaluation)")
        pool = build_pool(
            self.ctx, self.model, self._store_elites(), self.ledger_ids,
            self.rng, self.cfg, wave_index=0, size=self.pool_size,
        )
        if self.objective == "max_cycle_min_fr":
            scored = acq.score_pool_max_cycle(
                self.model, self.ctx, pool, self.max_cycle_spec, self.trust_region,
                tie_epsilon=float(self.acq.tie_epsilon),
            )
        elif self.objective == "min_fr_max_cycle":
            scored = acq.score_pool_min_fr(
                self.model, self.ctx, pool, self.min_fr_spec, self.trust_region,
                tie_epsilon=float(self.acq.tie_epsilon),
            )
        elif self.objective == "min_fxy":
            scored = acq.score_pool_min_fxy(
                self.model, self.ctx, pool, self.min_fxy_spec, self.trust_region,
                tie_epsilon=float(self.acq.tie_epsilon),
            )
        elif self.objective == "min_fuel_cost":
            scored = acq.score_pool_min_fuel_cost(
                self.model, self.ctx, pool, self.min_fuel_cost_spec, self.trust_region,
                tie_epsilon=float(self.acq.tie_epsilon),
                fuel=self.fuel, library_id=self.library_id,
            )
        elif self.objective == "fr_boundary":
            scored = acq.score_pool_fr_boundary(
                self.model, self.ctx, pool, self.fr_boundary_spec, self.trust_region,
                tie_epsilon=float(self.acq.tie_epsilon),
            )
        elif self.objective == "flat_power":
            scored = acq.score_pool_flat_power(
                self.model, self.ctx, pool, self.flat_power_spec, self.trust_region,
                tie_epsilon=float(self.acq.tie_epsilon),
            )
        else:
            boot = self.model.predict(
                [c.pattern for c in pool], self.ctx.case_key, self.ctx.e_core or 0.0
            )
            reward_model = acq.build_reward_model(
                self.ctx, [c.pattern for c in pool], boot, self.constraints
            )
            scored = acq.score_pool(
                self.model, self.ctx, pool, reward_model, self.constraints, self.trust_region,
                tie_epsilon=float(self.acq.tie_epsilon),
            )
        # rank proposals by the exploit rank (constrained objective LCB with the
        # margin tie-break), not the exploration-weighted acquisition.
        order = np.argsort(-scored.rank)[:16]
        proposals = []
        self._log(f"{'#':>2} {'origin':9s} {'p_feas':>7s} {'acq':>7s} "
                  f"{'F_r':>6s} {'F_q':>6s} {'CBC':>7s} {'cyclen':>7s} record_id")
        for rank, i in enumerate(order):
            i = int(i)
            m = scored.mean[i]
            cand = scored.candidates[i]
            self._log(
                f"{rank + 1:>2} {cand.origin:9s} {scored.p_feas[i]:7.3f} {scored.acq[i]:7.3f} "
                f"{m[0]:6.3f} {m[2]:6.3f} {m[1]:7.1f} {m[3]:7.1f} {cand.record_id[:12]}"
            )
            proposals.append({
                "rank": rank + 1, "origin": cand.origin, "record_id": cand.record_id,
                "p_feas": float(scored.p_feas[i]), "acq": float(scored.acq[i]),
                "pred": {"f_r": float(m[0]), "cbc_max": float(m[1]), "f_q": float(m[2]),
                         "cyclen": float(m[3]), "ao_abs": float(m[4])},
                "shf": cand.pattern.to_shf(),
            })
        _atomic_json(self.run_dir / "proposals.json", {"case": self.ctx.case_key.label, "proposals": proposals})
        self._write_status("proposals_only")
        return CampaignResult(
            run_dir=str(self.run_dir), status="proposals_only", waves=0,
            budget=0, budget_spent=0, n_feasible=0, on_target=0, best=None,
            proposals=proposals,
        )

    def _result(self, status: str) -> CampaignResult:
        return CampaignResult(
            run_dir=str(self.run_dir), status=status, waves=self.wave_index,
            budget=self.budget, budget_spent=self.budget_spent,
            n_feasible=sum(1 for r in self.campaign_rows if self._is_feasible(r)),
            on_target=sum(1 for r in self.campaign_rows if self._is_on_target(r)),
            best=self.best,
            best_overall=self.best_overall,
            wave_reports=[vars(w) for w in self.wave_reports],
            map_harvest_rates=list(self.map_harvest_rates),
            post_verify_master_calls=self.post_verify_calls,
            post_verify_violators=list(self.post_verify_violators),
        )

    def _write_delivery(self) -> dict[str, Any] | None:
        """Write ``delivery.json`` — the §2.2 / D2 delivery ranking (flat_power).

        Kept strictly downstream of the search: it reads the campaign's finished
        rows and applies the delivery rule (flat BAND 0.10-0.40 within-cell
        ``node_peak`` percentile, then rank by ``compliance_margin`` to 1.55).
        The flattest point is deliberately NOT a delivery candidate.  Nothing here
        feeds back into the objective, the elites or the pool.
        """
        payload = build_delivery_payload(
            self.campaign_rows,
            objective=self.objective,
            limits=self.feasibility_limits(),
            cell=self.flat_cell_key,
            safety=self._row_safety_fields,
        )
        if payload is None:
            return None
        try:
            _atomic_json(self.run_dir / "delivery.json", payload)
        except OSError:                     # a report must never fail a run
            return payload
        return payload

    # -- D9 SDM / MTC pre-delivery gate ------------------------------------- #
    def _record_sdm_mtc_target(self, record_id: str, outcome: Any) -> None:
        """Index one converged candidate's branch assets (decision D9).

        Best-effort and never fatal: a campaign must not die because a licensing
        convenience index could not be appended.  A candidate whose provenance is
        missing (``keep_success=False`` deleted the final work dir) is recorded
        with null assets and an explicit note, so the gate can report *why* it
        could not verify that candidate instead of quietly skipping it.
        """
        try:
            from . import sdm_mtc as _sm

            prov = getattr(outcome, "eq_provenance", None) or {}
            _sm.record_target(
                self.run_dir, record_id,
                deck_path=prov.get("deck"), restart_path=prov.get("restart"),
                tag=f"{self.ctx.case_key.folder}_{record_id[:8]}",
                note=("" if prov else
                      "no retained converged work dir (verify.keep_success / "
                      "harvest_maps off?) — candidate is not licence-verifiable"),
            )
        except Exception:  # noqa: BLE001 — index writing never fails a wave
            pass

    def _rod_model(self) -> Any | None:
        """Full-core rod model for the SDM branch, or ``None`` (SDM stays open).

        The campaign deck is QUARTER-core and carries no ``%ROD_CFG`` /
        ``%ROD_MAP``, and MOCHA builds its APR1400 model from a full-core seed
        (``sdm_mtc_io.build_apr1400_rod_model`` + a full-core equilibrium chain).
        lpopt has no full-core asset package, so this returns ``None`` until one
        exists — and the gate then reports SDM as INCONCLUSIVE rather than
        inventing a rod map.  See the workstream notes for what is needed.
        """
        return None

    def _maybe_post_verify(self, delivery: dict[str, Any] | None) -> None:
        """Run the SDM/MTC gate on the delivery-ranked flat feasible top-K.

        Order matters: this runs AFTER ``_write_delivery`` because D9's target set
        is defined by the delivery ranking (flat band, feasible excluding F_r), not
        by the search's best-objective row.
        """
        c = getattr(self.cfg, "constraints", None)
        if c is None or not c.any_enabled() or int(c.post_verify_top_k) <= 0:
            return
        if self.post_verify_done:
            # The gate spends MASTER calls and is re-entered by every completed
            # run() (resume, re-report).  Its accounting is persisted, so a second
            # pass would double-spend the licensing budget for the same candidates
            # while the first pass's calls are already in ``post_verify_calls``.
            self._log(
                "[optimize] SDM/MTC gate already ran for this run "
                f"({self.post_verify_calls} MASTER call(s) recorded in "
                f"{self.state_path.name}); not re-run"
            )
            return
        if self.post_verify_executor is None and (
            self.dry_run or not self.cfg.master.executable
        ):
            self._log(
                "[optimize] SDM/MTC gate configured but not run here "
                "(dry-run / no [master].executable); "
                f"top_k={c.post_verify_top_k} carried for the live run"
            )
            return
        if not (delivery or {}).get("ranked"):
            # D9's target set is the DELIVERY ranking (flat band, feasible
            # excluding F_r).  Only flat_power produces one, so say plainly that
            # there was nothing to verify instead of reporting a vacuous 0/0 pass.
            self._log(
                "[optimize] SDM/MTC gate: no delivery ranking to verify "
                f"(objective={self.objective!r}; the gate targets the flat_power "
                "delivery candidates, decision D9) — gate not run"
            )
            return
        try:
            from . import sdm_mtc as _sm

            summary = _sm.post_verify_delivery(
                self.run_dir, delivery, c,
                sdm_mtc_cfg=getattr(self.cfg, "sdm_mtc", None),
                master_cfg={
                    "executable": self.cfg.master.executable,
                    "package_root": (
                        str(self._resolve(self.cfg.verify.package_root))
                        if self.cfg.verify.package_root else None
                    ),
                    "timeout": float(getattr(self.cfg.sdm_mtc, "branch_timeout_s", 300.0)),
                },
                rod_model=self._rod_model(),
                executor=self.post_verify_executor,
                sidecar_path=self._resolve(self.cfg.sdm_mtc.sidecar_path),
            )
        except Exception as exc:  # noqa: BLE001 — the gate never sinks a finished run
            self._log(f"[optimize][WARNING] SDM/MTC gate failed: {exc}")
            return
        self.post_verify_calls += int(summary.master_calls)
        self.post_verify_violators = list(summary.violators)
        self.post_verify_summary = summary.as_dict()
        self.post_verify_done = True
        # persist the licensing spend IMMEDIATELY: these MASTER calls are already
        # gone, and a crash between here and the next _save_state would otherwise
        # lose the only record that they were made.
        self._save_state()
        self._mark_delivery_violators(delivery, summary)
        self._log(
            f"[optimize] SDM/MTC gate: {len(summary.results)}/{summary.n_selected} "
            f"candidate(s) verified, {len(summary.violators)} violator(s), "
            f"{summary.master_calls} extra MASTER call(s) "
            f"({'REPORT-ONLY — no user limit set' if summary.report_only else 'GATED'})"
        )
        for entry in summary.skipped:
            self._log(f"[optimize] SDM/MTC gate SKIPPED {entry.get('record_id')}: "
                      f"{entry.get('reason')}")

    def _mark_delivery_violators(self, delivery: dict[str, Any] | None,
                                 summary: Any) -> None:
        """Stamp each ranked delivery candidate with its SDM/MTC verdict.

        Violators are MARKED, not deleted: the delivery ranking is the record of
        what the campaign produced, and a candidate silently vanishing between
        ``delivery.json`` and hand-off is exactly the failure this stage exists to
        prevent.  ``sdm_mtc_verdict`` / ``sdm_mtc_violation`` ride alongside each
        entry, and the top-level ``sdm_mtc`` block carries the run summary.
        """
        if not delivery:
            return
        by_id = {r.record_id: r for r in summary.results}
        for entry in delivery.get("ranked") or []:
            res = by_id.get(str(entry.get("record_id") or ""))
            if res is None:
                entry["sdm_mtc_verdict"] = "NOT_VERIFIED"
                entry["sdm_mtc_violation"] = False
                continue
            entry["sdm_mtc_verdict"] = res.verdict
            entry["sdm_mtc_violation"] = res.violates
            entry["mtc_pcm_per_c"] = (
                res.mtc.value_pcm_per_c if res.mtc is not None else None
            )
            entry["sdm_margin_pcm"] = (
                res.sdm.margin_pcm if res.sdm is not None else None
            )
        delivery["sdm_mtc"] = summary.as_dict()
        try:
            _atomic_json(self.run_dir / "delivery.json", delivery)
        except OSError:
            pass

    def _render_report(self, result: CampaignResult) -> None:
        delivery: dict[str, Any] | None = None
        try:
            delivery = self._write_delivery()
        except Exception as exc:  # noqa: BLE001 — never fails the run
            self._log(f"[optimize][WARNING] delivery ranking failed: {exc}")
        # D9 gate runs on the delivery ranking, then folds its accounting into the
        # result BEFORE the report renders, so the report can state both budgets.
        #
        # FULLY CONTAINED (incident 2026-08-30): the gate is the LAST thing that
        # may fail, and everything below it -- status.json, report.md and the
        # delivery.json re-stamp -- is the run's deliverable.  An unguarded gate
        # call let a UnicodeEncodeError raised by one of its own log lines
        # (cp949 stdout, em-dash) propagate out of _render_report and skip every
        # artefact of a completed 100-call campaign, while status.json already
        # said "complete".  The gate's internals already swallow their own
        # errors; this catches what escapes them, logging included.
        try:
            self._maybe_post_verify(delivery)
        except Exception as exc:  # noqa: BLE001 -- a gate failure never sinks the artefacts
            self._log(f"[optimize][WARNING] SDM/MTC gate aborted: "
                      f"{type(exc).__name__}: {exc}; report + delivery still written")
        result.post_verify_master_calls = self.post_verify_calls
        result.post_verify_violators = list(self.post_verify_violators)
        try:
            self._write_status(result.status)
        except OSError as exc:  # noqa: BLE001 -- one artefact never blocks the next
            self._log(f"[optimize][WARNING] status.json write failed: {exc}")
        try:
            from ..report.report import write_campaign_report

            write_campaign_report(self, result)
        except Exception as exc:  # noqa: BLE001 — a report failure never fails the run
            self._log(f"[optimize][WARNING] report generation failed: {exc}")


# --------------------------------------------------------------------------- #
# user_criteria FREE-SEARCH driver (plan sec. 6.2 / 12.5)
# --------------------------------------------------------------------------- #
@dataclass
class _CellRun:
    """Per-cell (pair) accounting for the outer allocation report."""

    cell: PairCell
    stat: acq.OuterCellStat
    calls: int = 0
    best_row: dict[str, Any] | None = None
    best_fr: float | None = None
    #: patterns already verified for this cell (compose_wave diversity floor).
    seen_patterns: list[Pattern] = field(default_factory=list)


@dataclass
class _LeanCand:
    """One surrogate-scored candidate in the lean global registry (plan 12.5)."""

    pair: str
    candidate: Candidate
    ctx: CaseContext
    score: float                 # score_user_criteria hierarchical total (higher better)
    p_feas: float
    mean: np.ndarray             # 7-column surrogate mean prediction
    e_core: float | None
    #: True for a STORE-verified elite injected as a PREDICTION only (Bug B): it
    #: seeds the screen value + registry (and its mutation children are new
    #: verifiable candidates) but is itself never re-verified — it is already a
    #: converged store row (ledger dedup).
    verified: bool = False


@dataclass
class _Pending:
    """A prepared-but-not-yet-verified wave slot (batched full-width dispatch)."""

    entry: WaveEntry
    cand: Candidate
    total: float
    cell: PairCell
    phase: str
    wave: int
    e_core: float | None


class UserCriteriaDriver:
    """FREE-SEARCH over the pair universe with an outer racing allocation.

    The outer decision variable is the fuel PAIR (chosen from the e_core-reachable
    universe, :func:`build_pair_universe`); the inner search fixes the LP + batch
    split for a chosen pair.  Allocation (plan sec. 6.2, adapted from the feed_range
    design):

    * **wave 0** — surrogate-only virtual screening of every universe cell; the top
      ``outer_max_cells`` activate;
    * **waves 1..3** — ≤ ``outer_verify_per_wave`` verify slots per active cell
      (total ``outer_screen_budget``), with UCB/LCB racing elimination targeting
      ≤ ``outer_target_cells`` survivors;
    * **exploit** — softmax allocation of the remaining budget over survivors with
      an ``outer_exploit_floor`` on the best cell.

    Exploit ranking INSIDE a cell is :func:`score_user_criteria` (cyclen band ->
    discharge band -> min F_r UCB); the p_feas gate uses the CriteriaSpec SET
    limits.  Per-pair assets resolve through the proven ``CaseAssetResolver``
    fallback ladder (cross-anchor pairs typically resolve level 2-3; the fixed-
    point identity result justifies the restart-reference rewrite).
    """

    RACE_WAVE_CAP = 4

    def __init__(
        self,
        cfg: LpoptConfig,
        model: Any,
        evaluator_factory: Callable | None = None,
        *,
        dry_run: bool = False,
        run_dir: str | Path | None = None,
        budget: int | None = None,
        fuel_library: Any = None,
        progress: bool = True,
        log: Callable[[str], None] | None = None,
        seed: int | None = None,
        **_ignored: Any,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.evaluator_factory = evaluator_factory
        self.dry_run = bool(dry_run)
        self.progress = progress
        # Encoding-safe: a redirected Windows stdout is cp949, and a single
        # em-dash in a log line used to raise UnicodeEncodeError and sink a
        # finished 100-call run (incident 2026-08-30).  Wraps a supplied
        # ``log`` too -- that stream belongs to the caller, not to lpopt.
        self._log = safe_logger(log)
        self.criteria = cfg.criteria
        self.search = cfg.search

        base = cfg.source_path.parent if cfg.source_path else Path.cwd()
        self._base = base
        self.main_store_dir = self._resolve(cfg.model.store_dir)

        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = self._resolve(cfg.flow.output_root) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("logs", "waves", "candidates", "master", "models"):
            (self.run_dir / sub).mkdir(exist_ok=True)

        self.labels_path = self.run_dir / LABELS_NAME
        self.events_path = self.run_dir / "logs" / "events.jsonl"

        self.campaign_store_dir = (
            self.run_dir / "store" if self.dry_run else self.main_store_dir
        )
        self.store = StoreWriter(self.campaign_store_dir)

        self.budget = int(budget if budget is not None else cfg.acquisition.budget)
        self.library_id = cfg.model.library_id
        self.feed = int(cfg.case.feed)
        self.rng = random.Random(
            seed if seed is not None else cfg.flow.random_seed
        )
        self.fuel = fuel_library or self._load_fuel()

        self.spec = self._build_spec()
        # dry-run lightens the per-cell pool size (StubEvaluator path).
        self.pool_size = (
            self.search.dry_run_pool_size if self.dry_run else self.search.pool_size
        )
        self.resolver = self._resolver()
        # fallback level >= 3 needs a deep chain; keep >= 16 cycles (plan 12.5 item 4).
        _stub_verifier = evaluator_factory is not None
        self.verifier = WaveVerifier(
            run_dir=self.run_dir / "master",
            package_root=(
                self._resolve(cfg.verify.package_root)
                if (evaluator_factory is None and cfg.verify.package_root) else None
            ),
            executable=cfg.master.executable if evaluator_factory is None else None,
            # user_criteria inherits the [master] worker/core policy: 8 P-cores by
            # default, [master] use_all_cores spreads onto every logical core.
            workers=(cfg.master.workers or 8) if _stub_verifier else cfg.master.workers,
            use_all_cores=cfg.master.use_all_cores,
            host_reserve=cfg.master.host_reserve,
            timeout=cfg.master.timeout,
            max_cycles=max(int(cfg.master.max_cycles), 16),
            # deck sanity gate at the campaign library's real dims (ECC review
            # 2026-08-12 — the default is the ga80 constant (83, 85))
            library_dims=self.resolver.library_dims,
            consecutive=cfg.master.consecutive,
            tolerances=cfg.master.tolerances,
            evaluator_factory=evaluator_factory,
            resolver=self.resolver,
            purge_intermediate=cfg.produce.purge_intermediate,
        )

        # runtime state
        self.ledger_ids: set[str] = set()
        #: True while a remote GPU screener is attached to ``self.model`` (plan 4.7).
        self._remote_active = False
        #: Lazy store index (Bug B): pair -> converged, feed+e_core-band verified
        #: rows, built once from the MAIN store and reused across all 800+ cells.
        self._verified_idx: dict[str, list[dict[str, Any]]] | None = None
        #: Per-pair (record_id, Pattern) store-elite seeds, cached (registers each
        #: elite's record_id into ``ledger_ids`` on first compute so build_pool
        #: never re-generates — nor re-verifies — an already-verified store LP).
        self._elite_seed_cache: dict[str, list[tuple[str | None, Pattern]]] = {}
        self.budget_spent = 0
        self.cell_runs: dict[str, _CellRun] = {}
        self.best: dict[str, Any] | None = None
        self.first_feasible_call: int | None = None
        self.fr_trajectory: list[dict[str, Any]] = []   # {call, best_fr}
        self.race_timeline: list[dict[str, Any]] = []
        self.universe: list[PairCell] = []
        # lean (predict-then-verify) accounting.
        self.lean_rows: list[dict[str, Any]] = []       # predicted-vs-actual top-K
        self.screen_seconds: float = 0.0
        self.verify_seconds: float = 0.0
        # dry-run lightens the deepen pool + local-search compute (StubEvaluator path).
        self.lean_deep_pool = (
            min(self.search.dry_run_pool_size, int(self.criteria.lean_pool_per_cell))
            if self.dry_run else int(self.criteria.lean_pool_per_cell)
        )

        # -- Stage-2 running cyclen bias corrector (cell_calibrate) ---------- #
        # Accumulates a per-(feed,e_core-bin) cyclen bias from THIS campaign's
        # MASTER-verified chains and subtracts it from subsequent screening/deepen
        # cyclen predictions — but ONLY for cells the serve-side Stage-1 affine
        # calibration does NOT already cover (fitted cells are seeded here so they
        # are never double-corrected).  Persisted to ``cyclen_bias.json`` in the
        # run dir for resume; inert (bias 0) until a verified label lands.
        self.bias_path = self.run_dir / "cyclen_bias.json"
        fitted = set()
        fitted_getter = getattr(self.model, "fitted_cyclen_cells", None)
        if callable(fitted_getter):
            try:
                fitted = set(fitted_getter())
            except Exception:  # noqa: BLE001 — never fail construction on this
                fitted = set()
        if self.bias_path.is_file():
            self.bias_corrector = CampaignBiasCorrector.load(self.bias_path)
            self.bias_corrector.fitted_cells |= fitted
        else:
            self.bias_corrector = CampaignBiasCorrector(fitted_cells=fitted)

    # -- setup helpers ----------------------------------------------------- #
    def _resolve(self, path_str: str | Path) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (self._base / p)

    def _load_fuel(self) -> Any:
        from ..data.fuel_types import FuelLibrary

        path = self.main_store_dir / "fuel_types.parquet"
        try:
            return FuelLibrary.from_parquet(path)
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _build_spec(self) -> acq.CriteriaSpec:
        c = self.criteria
        return acq.CriteriaSpec(
            target_cyclen=float(c.cyclen_target),
            target_discharge_burnup=(
                None if c.discharge_target is None else float(c.discharge_target)
            ),
            cyclen_tolerance=float(c.cyclen_tol),
            discharge_tolerance=float(c.discharge_tol),
            f_r_limit=c.f_r_limit, cbc_limit=c.cbc_limit, f_q_limit=c.f_q_limit,
            asi_abs_limit=c.asi_abs_limit, pin_bu_limit=c.pin_bu_limit,
            mtc_limit=c.mtc_limit, sdm_limit=c.sdm_limit, risk_z=float(c.risk_z),
        )

    def _resolver(self) -> CaseAssetResolver:
        # Same per-library routing as CampaignDriver._resolver (ECC review
        # 2026-08-12): a paramA user_criteria campaign must NOT resolve against
        # the ga80 package/fallbacks nor judge decks at LIBRARY_DIMS (83, 85).
        from .resolver import build_case_resolver, is_paramA_library
        if is_paramA_library(self.cfg, self.library_id):
            return build_case_resolver(self.cfg, self.fuel,
                                       library_id=self.library_id)
        package_root = (
            self._resolve(self.cfg.verify.package_root)
            if self.cfg.verify.package_root else self.run_dir
        )
        fallbacks = [str(self._resolve(g)) for g in self.cfg.produce.template_fallbacks]
        return CaseAssetResolver(
            package_root,
            self._resolve(self.cfg.produce.promoted_root),
            template_fallbacks=fallbacks,
            fuel_library=self.fuel,
            synth_root=self._resolve(self.cfg.produce.synth_decks_root),
        )

    def _cell_context(self, pair: str, e_core: float | None) -> CaseContext:
        return CaseContext(
            pair=pair, feed=self.feed, library_id=self.library_id, e_core=e_core,
        )

    # -- criteria feasibility (used for best-tracking + first-feasible) ---- #
    def _is_criteria_feasible(self, row: dict[str, Any]) -> tuple[bool, float | None]:
        """(feasible, F_r).  Feasible == converged & every SET gate ok & cyclen in band.

        Missing values follow the 2026-07-31 contract (:func:`_is_missing`): NaN
        is treated exactly like None.  This predicate REJECTS a missing gate value
        — every gate the user SET must be shown satisfied — so NaN rejects too.
        That is the INVERTED twin of the :func:`is_feasible` hole: here
        ``float(nan) > limit`` is ``False``, so a parquet-sourced NaN silently
        PASSED a hard user gate that an in-memory ``None`` rejected.
        """

        if not row.get("converged"):
            return False, None
        s = self.spec
        try:
            f_r = float(row["f_r"])
        except (TypeError, ValueError, KeyError):
            return False, None
        checks = [
            (s.f_r_limit, row.get("f_r")),
            (s.cbc_limit, row.get("cbc_max")),
            (s.f_q_limit, row.get("f_q")),
            (s.asi_abs_limit, row.get("ao_abs")),
            (s.pin_bu_limit, row.get("max_pin_burnup")),
        ]
        for limit, value in checks:
            if limit is None:
                continue
            if _is_missing(value):
                return False, f_r
            try:
                if float(value) > float(limit):
                    return False, f_r
            except (TypeError, ValueError):
                return False, f_r
        cyclen = row.get("cyclen")
        if _is_missing(cyclen):
            return False, f_r
        if abs(float(cyclen) - s.target_cyclen) > s.cyclen_tolerance:
            return False, f_r
        return True, f_r

    # -- prepare + record (batched full-width verification) ---------------- #
    # A user_criteria wave is dispatched as ONE heterogeneous WaveVerifier wave
    # across every active cell (the WaveVerifier supports mixed pair/case entries),
    # never a per-cell 2-entry mini-wave: ``_prepare_cell_slots`` scores a cell and
    # returns its verify entries (reserving their record_ids so a sibling cell can
    # never re-draw them), then the caller concatenates every cell's entries into a
    # single ``evaluate_wave`` and ``_record_outcomes`` writes the labels back.
    def _prepare_cell_slots(
        self, cell: PairCell, n_slots: int, *, wave: int, phase: str
    ) -> list[_Pending]:
        """Score ``cell`` and build ≤ ``n_slots`` in-band verify entries (no verify)."""

        n_slots = int(n_slots)
        if n_slots <= 0:
            return []
        ctx = self._cell_context(cell.pair, cell_target_e_core(cell, self.criteria))
        pool = build_pool(
            ctx, self.model, self._cell_elites(cell.pair), self.ledger_ids,
            self.rng, self.cfg, wave_index=wave, size=self.pool_size,
        )
        if not pool:
            return []
        scored = acq.score_pool_user_criteria(
            self.model, ctx, pool, self.spec,
            fuel=self.fuel, library_id=self.library_id,
            e_core_target=self.criteria.e_core_target, e_core_tol=self.criteria.e_core_tol,
        )
        if not np.any(scored.in_region):
            return []
        # Exploit slots rank on score_user_criteria (scored.rank); explore/control
        # slots keep their usual behaviour (max epistemic / random).  _slot_counts
        # scales the configured ratio down for the small racing waves (plan 12.5
        # item 3: explore/control unchanged).
        a = self.cfg.acquisition
        tau = acq.tau_schedule(scored, a.tau0, have_feasible=False)
        slots = acq.compose_wave(
            scored, list(self.cell_runs[cell.pair].seen_patterns), self.rng,
            size=n_slots, n_exploit=a.exploit, n_explore=a.explore, n_control=a.control,
            tau=tau, hamming_min=a.hamming_min,
        )
        if not slots:
            return []
        resolved = self.resolver.resolve(ctx.case_key)
        run = self.cell_runs[cell.pair]
        pending: list[_Pending] = []
        for w in slots:
            cand = scored.candidates[w.index]
            ec = predicted_e_core(cand.pattern, self.fuel, self.library_id)
            entry = WaveEntry(
                cand.pattern, ctx.case_key, resolved,
                {"pair": cell.pair, "slot": w.slot, "phase": phase,
                 "record_id": cand.record_id},
            )
            pending.append(_Pending(
                entry=entry, cand=cand, total=float(scored.exploit[w.index]),
                cell=cell, phase=phase, wave=wave, e_core=ec,
            ))
            # reserve within-wave: dedup so a sibling cell/round never re-draws it,
            # and the cell's diversity floor already excludes it next time.
            self.ledger_ids.add(cand.record_id)
            run.seen_patterns.append(cand.pattern)
        return pending

    def _record_outcomes(self, pending: Sequence[_Pending], outcomes: Sequence[Any]) -> None:
        """Write store rows + labels for a verified batch; update racing stats/best."""

        records: list[CanonicalRecord] = []
        for pend, outcome in zip(pending, outcomes, strict=True):
            cand = pend.cand
            record = outcome_to_record(
                outcome, dataset="P", library_id=self.library_id,
                stratum=f"user_criteria_{pend.phase}", generator=cand.origin,
                parent_record_id=cand.parent_record_id, campaign=self.run_dir.name,
                e_core=pend.e_core, e_split=None, deck_knobs=PRODUCE_DECK_KNOBS,
            )
            records.append(record)
            row = record.to_record()
            self.budget_spent += 1
            run = self.cell_runs[pend.cell.pair]
            run.calls += 1
            run.stat.n_verify += 1
            run.stat.samples.append(pend.total)
            feasible, f_r = self._is_criteria_feasible(row)
            if feasible and f_r is not None:
                if run.best_fr is None or f_r < run.best_fr:
                    run.best_fr, run.best_row = f_r, row
                    run.stat.best_fr_ucb = min(run.stat.best_fr_ucb, f_r)
                    run.stat.best_feasible = True
                self._update_global_best(row, f_r, pend.cell)
            self.fr_trajectory.append(
                {"call": self.budget_spent,
                 "best_fr": (self.best["f_r"] if self.best else None)}
            )
            _append_jsonl(self.labels_path, {
                "wave": pend.wave, "phase": pend.phase, "pair": pend.cell.pair,
                "record_id": record.record_id, "status": outcome.status,
                "criteria_feasible": feasible, "criteria_total": pend.total,
                "e_core": pend.e_core, "record": row,
            })
        self.store.write_records(records)

    def _cell_elites(self, pair: str) -> list[tuple[str | None, Pattern]]:
        """This-run verified feasible board(s) of ``pair`` as elite parents."""

        run = self.cell_runs.get(pair)
        if run and run.best_row is not None:
            try:
                return [(run.best_row.get("record_id"),
                         unpack_pattern(str(run.best_row["pattern"])))]
            except (ValueError, KeyError):
                return []
        return []

    # -- store-verified elite injection (Bug B) ---------------------------- #
    def _verified_store_index(self) -> dict[str, list[dict[str, Any]]]:
        """Pair -> converged, feed-matched, e_core-in-band store rows (built once).

        The MAIN store already holds STORE-VERIFIED feasible LPs for reachable
        pairs (e.g. K1_K2 from an earlier campaign).  The lean screen used to draw
        only random/heuristic candidates, so a known-good basin was never deepened
        and its verified elites never seeded any pool (Bug B).  This index makes
        those rows available to every cell's screen/deepen pool at O(1)/pair.
        """

        if self._verified_idx is not None:
            return self._verified_idx
        idx: dict[str, list[dict[str, Any]]] = {}
        try:
            df = StoreReader(self.main_store_dir).records
        except (FileNotFoundError, OSError):
            df = None
        if df is not None and len(df):
            sub = df[df["converged"] == True]  # noqa: E712
            if "feed" in df.columns:
                sub = sub[sub["feed"] == int(self.feed)]
            tgt = float(self.criteria.e_core_target)
            tol = float(self.criteria.e_core_tol)
            # Vectorise the e_core band before iterrows so the per-row loop runs
            # over only the in-band rows (NaN e_core compares False -> excluded),
            # not every converged row in a 40k-row store.
            if "e_core" in sub.columns:
                ec = sub["e_core"]
                sub = sub[(ec >= tgt - tol) & (ec <= tgt + tol)]
            for _, r in sub.iterrows():
                d = r.to_dict()
                if not e_core_in_band(d.get("e_core"), tgt, tol):
                    continue
                idx.setdefault(str(d.get("case_pair")), []).append(d)
        self._verified_idx = idx
        return idx

    def _store_elite_seeds(self, pair: str) -> list[tuple[str | None, Pattern]]:
        """``(record_id, Pattern)`` store elites for ``pair`` (feasible-first, capped).

        Feeds ``build_pool`` as ``store_elites`` so its small-move mutation
        children are NEW verifiable candidates near a known optimum.  Registers each
        elite's record_id into the ledger (dedup) so build_pool never regenerates —
        nor the wave re-verifies — an already-verified store LP.
        """

        cached = self._elite_seed_cache.get(pair)
        if cached is not None:
            return cached
        cap = int(getattr(self.criteria, "lean_store_elites_per_cell", 8))
        seeds: list[tuple[str | None, Pattern]] = []
        if cap > 0:
            feasible: list[tuple[float, str | None, Pattern]] = []
            other: list[tuple[float, str | None, Pattern]] = []
            for row in self._verified_store_index().get(pair, []):
                try:
                    pat = unpack_pattern(str(row["pattern"]))
                except (ValueError, KeyError):
                    continue
                rid = row.get("record_id")
                rid = str(rid) if rid is not None else None
                feas, f_r = self._is_criteria_feasible(row)
                key = float(f_r) if f_r is not None else float("inf")
                (feasible if feas else other).append((key, rid, pat))
            feasible.sort(key=lambda t: t[0])
            other.sort(key=lambda t: t[0])
            seeds = [(rid, pat) for _, rid, pat in (feasible + other)[:cap]]
            for rid, _ in seeds:
                if rid:
                    self.ledger_ids.add(rid)
        self._elite_seed_cache[pair] = seeds
        return seeds

    def _inject_store_elites(
        self,
        cell: PairCell,
        ctx: CaseContext,
        seeds: Sequence[tuple[str | None, Pattern]],
    ) -> list[_LeanCand]:
        """Surrogate-score the store elites themselves as PREDICTION-only lean
        candidates (``verified=True``): they seed the cell screen value + registry
        (and the report) but are never re-verified — they are converged store rows.
        """
        return self._score_inject_candidates(
            cell, ctx, self._build_inject_candidates(ctx, seeds)
        )

    def _build_inject_candidates(
        self, ctx: CaseContext, seeds: Sequence[tuple[str | None, Pattern]],
    ) -> list[Candidate]:
        """Construct the store-elite injection candidates (the rng-drawing half).

        Kept separate from scoring so the batched remote screen can generate all
        cells' candidates in the exact same order (identical rng draws) before the
        single prewarm predict — see :meth:`_run_lean`.
        """
        cands: list[Candidate] = []
        for rid, pat in seeds:
            genome = _pattern_to_case_genome(pat, ctx, self.rng)
            if genome is None:
                continue
            cands.append(
                Candidate(
                    pat, genome, "store_elite", None,
                    str(rid) if rid is not None else candidate_record_id(pat, ctx),
                    ctx.e_core,
                )
            )
        return cands

    def _score_inject_candidates(
        self, cell: PairCell, ctx: CaseContext, cands: Sequence[Candidate],
    ) -> list[_LeanCand]:
        """Score prebuilt store-elite candidates (the pure, rng-free half)."""
        cands = list(cands)
        if not cands:
            return []
        scored = acq.score_pool_user_criteria(
            self.model, ctx, cands, self.spec,
            fuel=self.fuel, library_id=self.library_id,
            e_core_target=self.criteria.e_core_target, e_core_tol=self.criteria.e_core_tol,
        )
        out: list[_LeanCand] = []
        for i, cand in enumerate(cands):
            out.append(_LeanCand(
                pair=cell.pair, candidate=cand, ctx=ctx,
                score=float(scored.exploit[i]), p_feas=float(scored.p_feas[i]),
                mean=np.asarray(scored.mean[i], dtype=float),
                e_core=predicted_e_core(cand.pattern, self.fuel, self.library_id),
                verified=True,
            ))
        return out

    def _known_verified_store_lps(self) -> list[dict[str, Any]]:
        """Store rows matching the criteria (report table, Bug B item 3).

        A row qualifies on the SAME feasibility the search declares —
        ``_is_criteria_feasible``: converged + feed + e_core band (index) + every
        gated SET limit + the cyclen band.  The discharge band is a POST-HOC
        estimate (the verify FOM carries no discharge axis) and, like the search's
        own soft objective, is NOT a hard gate — it is annotated per row
        (``disch_est`` / ``disch_in_band``) for transparency, never used to hide a
        genuinely verified LP.  Sorted best-F_r first so the user sees existing
        solutions even when the new wave finds none.
        """

        out: list[dict[str, Any]] = []
        tgt = self.criteria.discharge_target
        tol = float(self.criteria.discharge_tol)
        for rows in self._verified_store_index().values():
            for row in rows:
                feas, _ = self._is_criteria_feasible(row)
                if not feas:
                    continue
                annotated = dict(row)
                disch = self._discharge_estimate(row.get("cyclen"))
                annotated["disch_est"] = disch
                annotated["disch_in_band"] = (
                    None if tgt is None or disch is None
                    else abs(float(disch) - float(tgt)) <= tol
                )
                out.append(annotated)
        out.sort(
            key=lambda r: float(r["f_r"]) if r.get("f_r") is not None else float("inf")
        )
        return out

    def _update_global_best(self, row: dict[str, Any], f_r: float, cell: PairCell) -> None:
        if self.first_feasible_call is None:
            self.first_feasible_call = self.budget_spent
        if self.best is None or f_r < self.best["f_r"]:
            self.best = {
                "pair": cell.pair, "record_id": row.get("record_id"),
                "f_r": f_r, "cbc_max": row.get("cbc_max"), "f_q": row.get("f_q"),
                "ao_abs": row.get("ao_abs"), "cyclen": row.get("cyclen"),
                "max_pin_burnup": row.get("max_pin_burnup"),
                "e_core": row.get("e_core"), "n_cycles": row.get("n_cycles"),
                "pattern": row.get("pattern"),
                "distance": abs(float(row["cyclen"]) - self.spec.target_cyclen)
                if row.get("cyclen") is not None else None,
            }

    # -- wave 0 surrogate screen ------------------------------------------- #
    def _screen_cell(self, cell: PairCell) -> float:
        """Surrogate-only virtual-screen value for a cell (best in-band total)."""

        ctx = self._cell_context(cell.pair, cell_target_e_core(cell, self.criteria))
        pool = build_pool(
            ctx, None, [], self.ledger_ids, self.rng, self.cfg,
            wave_index=0, size=int(self.criteria.screen_pool_per_cell),
        )
        if not pool:
            return float("-inf")
        scored = acq.score_pool_user_criteria(
            self.model, ctx, pool, self.spec,
            fuel=self.fuel, library_id=self.library_id,
            e_core_target=self.criteria.e_core_target, e_core_tol=self.criteria.e_core_tol,
        )
        vals = scored.exploit[np.isfinite(scored.exploit)]
        return float(np.max(vals)) if vals.size else float("-inf")

    # -- main run ---------------------------------------------------------- #
    def run(self) -> CampaignResult:
        """Build the pair universe, then dispatch on ``[criteria] search_mode``.

        ``lean`` (default) — one-shot screen + deepen, then ONE batched top-K
        verification wave (predict-then-verify, answers in minutes).  ``active`` —
        the outer racing/waves allocation over the pair universe.
        """

        if self.cfg.source_path and Path(self.cfg.source_path).exists():
            try:
                shutil.copy2(self.cfg.source_path, self.run_dir / "input_deck.inp")
            except OSError:
                pass

        # 0. build the pair universe (e_core-reachable cells only).
        all_cells = build_pair_universe(
            self.fuel, self.library_id,
            self.criteria.e_core_target, self.criteria.e_core_tol,
            split_range=self.criteria.split_range, allow_mono=self.criteria.allow_mono,
        )
        self.universe = all_cells
        included = [c for c in all_cells if c.included]
        self._log(
            f"[user_criteria] universe at e_core {self.criteria.e_core_target}"
            f"+/-{self.criteria.e_core_tol}: {len(included)} reachable "
            f"of {len(all_cells)} enumerated pairs (feed {self.feed}, lib {self.library_id})"
        )
        if not included:
            self._write_status("no_universe")
            result = self._result("no_universe")
            self._render_report(result)
            return result

        if self.criteria.search_mode == "lean":
            return self._run_lean(included)
        return self._run_active(included)

    # -- ACTIVE path: outer racing/waves allocation ------------------------ #
    def _run_active(self, included: list[PairCell]) -> CampaignResult:
        # 1. wave 0 — surrogate-only screen of every reachable cell.
        for cell in included:
            sv = self._screen_cell(cell)
            stat = acq.OuterCellStat(cell_id=cell.pair, screen_value=sv)
            self.cell_runs[cell.pair] = _CellRun(cell=cell, stat=stat)
        stats = [r.stat for r in self.cell_runs.values()]
        activated = acq.outer_activate(stats, self.criteria.outer_max_cells)
        self._log(
            f"[user_criteria] wave 0 screen done; activated {len(activated)} cells: "
            + ", ".join(activated[:8])
        )
        self.race_timeline.append({
            "wave": 0, "phase": "screen",
            "active": list(activated), "eliminated": [],
        })

        prior_sigma = self._prior_sigma(stats)

        # 2. racing waves — ONE batched full-width wave across all active cells,
        #    then eliminate (never per-cell 2-entry mini-waves).
        wave = 1
        while (
            self.budget_spent < min(self.criteria.outer_screen_budget, self.budget)
            and wave <= self.RACE_WAVE_CAP
        ):
            active = [r for r in self.cell_runs.values()
                      if r.stat.active and not r.stat.eliminated]
            if len(active) <= self.criteria.outer_target_cells:
                break
            limit = min(self.criteria.outer_screen_budget, self.budget)
            pending: list[_Pending] = []
            for run in sorted(active, key=lambda r: -r.stat.screen_value):
                remaining = limit - self.budget_spent - len(pending)
                if remaining <= 0:
                    break
                n = min(self.criteria.outer_verify_per_wave, remaining)
                pending.extend(
                    self._prepare_cell_slots(run.cell, n, wave=wave, phase="race")
                )
            if pending:
                outcomes = self.verifier.evaluate_wave([p.entry for p in pending])
                self._record_outcomes(pending, outcomes)
            eliminated = acq.outer_race(
                stats, z=self.criteria.outer_race_z, prior_sigma=prior_sigma,
                min_keep=self.criteria.outer_target_cells,
            )
            self.race_timeline.append({
                "wave": wave, "phase": "race",
                "active": [r.cell.pair for r in self.cell_runs.values()
                           if r.stat.active and not r.stat.eliminated],
                "eliminated": eliminated,
            })
            self._log(
                f"[user_criteria] race wave {wave}: spent {self.budget_spent}/{self.budget}"
                f" | survivors {sum(1 for s in stats if s.active and not s.eliminated)}"
                + (f" | eliminated {eliminated}" if eliminated else "")
            )
            wave += 1

        # 3. exploit — softmax allocation of the remaining budget over survivors,
        #    dispatched as ONE batched full-width wave across the allocated cells.
        remaining = self.budget - self.budget_spent
        alloc = acq.outer_softmax_alloc(
            stats, remaining, temperature=self.criteria.outer_softmax_temp,
            exploit_floor=self.criteria.outer_exploit_floor,
        )
        self.race_timeline.append({
            "wave": wave, "phase": "exploit", "alloc": dict(alloc), "eliminated": [],
        })
        self._log(f"[user_criteria] exploit alloc ({remaining} slots): {alloc}")
        pending = []
        for pair, n in sorted(alloc.items(), key=lambda kv: -kv[1]):
            if n <= 0:
                continue
            avail = self.budget - self.budget_spent - len(pending)
            if avail <= 0:
                break
            pending.extend(
                self._prepare_cell_slots(
                    self.cell_runs[pair].cell, min(n, avail), wave=wave, phase="exploit"
                )
            )
        if pending:
            outcomes = self.verifier.evaluate_wave([p.entry for p in pending])
            self._record_outcomes(pending, outcomes)

        return self._finish("active")

    def _finish(self, mode: str) -> CampaignResult:
        status = "complete" if self.best else "no_feasible"
        self._write_status(status)
        self._maybe_post_verify()
        result = self._result(status)
        self._render_report(result)
        self._log(
            f"[user_criteria][{mode}] done: spent {self.budget_spent}/{self.budget}"
            + (f" | best pair {self.best['pair']} F_r {self.best['f_r']:.3f} "
               f"cyclen {self.best['cyclen']:.1f}" if self.best else " | no feasible LP")
        )
        return result

    # -- remote GPU screening (plan 4.7) ----------------------------------- #
    def _enable_remote_screening(self) -> None:
        """Attach the gpu2-6000 batch-inference screener to ``self.model``.

        The effective mode comes from :func:`_resolve_inference_mode`: the explicit
        ``[model] inference`` selector (``"local_cpu"`` / ``"remote_gpu"``) wins, and
        an unset ``inference`` defers to the legacy ``[model] remote_screening``
        (``"auto"`` / ``true`` / ``false``).  ``auto`` (and ``remote_gpu``) probe the
        server (5 s) and stay local if unreachable; ``on`` attaches unconditionally
        (still falling back per-batch on any transport failure).  A backend without
        the hook (stub / sklearn) silently stays local.  NEVER raises — the screen
        must run, on GPU or CPU.
        """
        self._remote_active = False
        mode = _resolve_inference_mode(self.cfg.model)
        selector = str(getattr(self.cfg.model, "inference", "") or "").strip().lower()
        if mode == "off" or not hasattr(self.model, "enable_remote_screening"):
            return
        try:
            from ..remote import RemoteSettings, make_remote_screener, probe

            s = RemoteSettings.from_input(self.cfg.source_path)
            if mode == "auto" and not probe(s, timeout=5):
                # Loud on an explicit remote_gpu request; the campaign continues.
                how = ("inference=remote_gpu requested but "
                       if selector in ("remote_gpu", "remote", "gpu") else "")
                self._log(
                    f"[remote_screening] {how}{s.user}@{s.host}:{s.port} unreachable "
                    "(5s probe); FALLING BACK to local CPU for this campaign")
                return
            min_pred = int(getattr(self.cfg.model, "remote_screening_min", 5000))
            ckpt = str(self._resolve(self.cfg.model.model_dir))
            screener = make_remote_screener(
                s, ckpt, self.library_id, device="cuda", log=self._log)
            self.model.enable_remote_screening(
                screener, min_predictions=min_pred, log=self._log)
            self._remote_active = True
            label = f"inference={selector}" if selector else f"mode={mode}"
            self._log(
                f"[remote_screening] enabled ({label}); batches >= {min_pred} "
                f"predictions route to {s.user}@{s.host} gpu={s.gpu}")
        except Exception as exc:  # noqa: BLE001 — remote is best-effort, never fatal
            self._log(f"[remote_screening] setup failed ({exc}); local CPU")
            self._remote_active = False

    def _disable_remote_screening(self) -> None:
        if self._remote_active and hasattr(self.model, "disable_remote_screening"):
            self.model.disable_remote_screening()
        self._remote_active = False

    def _prewarm_screen(
        self,
        staged: Sequence[tuple[PairCell, CaseContext, Sequence[Candidate],
                               Sequence[Candidate]]],
    ) -> None:
        """One batched GPU inference over every staged screen pattern → fills memo.

        Collects the (pattern, case) of every cell's pool + injection candidates
        and prewarms the backend so the per-cell scoring below is served from the
        session cache instead of 800+ tiny local forward passes.
        """
        if not hasattr(self.model, "prewarm"):
            return
        pats: list[Pattern] = []
        cases: list[CaseKey] = []
        for _cell, ctx, pool, inj_cands in staged:
            for cand in list(pool) + list(inj_cands):
                pats.append(cand.pattern)
                cases.append(ctx.case_key)
        if pats:
            self._log(
                f"[remote_screening] prewarming {len(pats)} screen predictions "
                "in one GPU batch")
            self.model.prewarm(pats, cases)

    # -- LEAN path: one-shot predict-then-verify --------------------------- #
    def _run_lean(self, included: list[PairCell]) -> CampaignResult:
        """Screen the full universe + deepen top cells, then verify the global
        top-K predicted candidates in ONE batched wave (plan 12.5 addendum)."""

        t0 = time.perf_counter()
        registry: list[_LeanCand] = []

        # 1. screen — surrogate-score every reachable cell; keep its best cands.
        #    Bug B: inject the store's verified elites for the pair — as elite
        #    parents (their small-move children are NEW verifiable candidates near a
        #    known optimum) AND as prediction-only registry candidates — so a
        #    known-good basin (e.g. K1_K2) is deepened and its screen value reflects
        #    the best of injected + generated.
        # Remote screening (plan 4.7): attach the GPU screener (if configured).
        # The generation pass below advances rng + ledger in the EXACT order of the
        # sequential path, so results are byte-identical whether the bulk
        # predictions run locally or on the GPU — only WHERE they run changes.
        # Detached in the finally so the verification path never routes.
        self._enable_remote_screening()
        try:
            n_injected = 0
            # Pass 1 (generation): build every cell's pool + injection candidates
            # in cell order — advancing rng + ledger exactly as the local path —
            # then prewarm the GPU with ONE batched inference over all of them.
            staged: list[
                tuple[PairCell, CaseContext, list[Candidate], list[Candidate]]
            ] = []
            for cell in included:
                seeds = self._store_elite_seeds(cell.pair)
                ctx, pool = self._build_cell_pool(
                    cell, int(self.criteria.screen_pool_per_cell),
                    wave_index=0, elites=seeds,
                )
                inj_cands = self._build_inject_candidates(ctx, seeds)
                staged.append((cell, ctx, pool, inj_cands))
            if self._remote_active:
                self._prewarm_screen(staged)

            # Pass 2 (scoring): pure + rng-free; served from the session memo.
            for cell, ctx, pool, inj_cands in staged:
                scored = self._score_pool(ctx, pool)
                sv, cands = self._harvest(cell, ctx, scored, keep=int(self.criteria.lean_top_k))
                injected = self._score_inject_candidates(cell, ctx, inj_cands)
                if injected:
                    n_injected += len(injected)
                    inj_best = max(
                        (lc.score for lc in injected if math.isfinite(lc.score)),
                        default=float("-inf"),
                    )
                    sv = max(sv, inj_best)          # screen value = best of both
                self.cell_runs[cell.pair] = _CellRun(
                    cell=cell, stat=acq.OuterCellStat(cell_id=cell.pair, screen_value=sv)
                )
                registry.extend(cands)
                registry.extend(injected)
            if n_injected:
                self._log(
                    f"[user_criteria][lean] injected {n_injected} store-verified elite "
                    "prediction(s) + their mutation children into the screen"
                )
            stats = [r.stat for r in self.cell_runs.values()]
            deep = acq.outer_activate(stats, self.criteria.lean_deep_cells)
            self.race_timeline.append({
                "wave": 0, "phase": "screen", "active": list(deep), "eliminated": [],
            })
            self._log(
                f"[user_criteria][lean] screened {len(included)} cells; deepening "
                f"{len(deep)}: " + ", ".join(deep[:8])
            )

            # 2. deepen — larger surrogate pool + local search on the top cells.
            #    Each cell's pool prediction routes through the memo/threshold
            #    (>= remote_screening_min); the adaptive local search stays local.
            for pair in deep:
                registry.extend(self._deepen_cell(self.cell_runs[pair].cell))
        finally:
            self._disable_remote_screening()
        self.screen_seconds = time.perf_counter() - t0
        self._log(
            f"[user_criteria][lean] screen+deepen {self.screen_seconds:.1f}s; "
            f"{len(registry)} surrogate-scored candidates"
        )

        # 3. select the global top-K (diverse, per-pair-capped) + log predictions.
        selected = self._select_top_k(registry)
        self._log_predicted_table(selected)

        # 4. verify ONCE — a single batched full-width wave of all K entries.
        self.verify_seconds = self._verify_batch(selected, round_tag="r1")

        # 5. optional second round (default OFF): only if nothing met the bands.
        if self.criteria.lean_second_round and self.best is None:
            chosen = {lc.candidate.record_id for lc in selected}
            second = self._select_top_k(
                [lc for lc in registry if lc.candidate.record_id not in chosen]
            )
            if second:
                self._log(f"[user_criteria][lean] second round: {len(second)} candidates")
                self.verify_seconds += self._verify_batch(second, round_tag="r2")

        return self._finish("lean")

    # -- lean helpers ------------------------------------------------------ #
    def _build_cell_pool(
        self, cell: PairCell, size: int, *, wave_index: int,
        elites: Sequence[tuple[str | None, Pattern]] = (),
    ) -> tuple[CaseContext, list[Candidate]]:
        """Build a cell's surrogate pool (the rng/ledger-advancing half).

        Split out from :meth:`_score_cell` so the batched remote screen can
        generate every cell's pool first (preserving the exact rng + ledger draw
        order), prewarm the GPU once, then score — see :meth:`_run_lean`.
        """
        ctx = self._cell_context(cell.pair, cell_target_e_core(cell, self.criteria))
        pool = build_pool(
            ctx, self.model, list(elites), self.ledger_ids, self.rng, self.cfg,
            wave_index=wave_index, size=int(size),
        )
        return ctx, pool

    def _score_pool(self, ctx: CaseContext, pool: Sequence[Candidate]
                    ) -> acq.ScoredPool | None:
        """Score a prebuilt pool (the pure, rng-free half of :meth:`_score_cell`)."""
        if not pool:
            return None
        pool = list(pool)
        return acq.score_pool_user_criteria(
            self.model, ctx, pool, self.spec,
            fuel=self.fuel, library_id=self.library_id,
            e_core_target=self.criteria.e_core_target, e_core_tol=self.criteria.e_core_tol,
            cyclen_bias=self._cyclen_bias_for(pool, ctx),
        )

    def _cyclen_bias_for(self, candidates: Sequence[Candidate], ctx: CaseContext
                         ) -> np.ndarray | None:
        """Per-candidate Stage-2 running cyclen bias to subtract at scoring.

        ``None`` (the fast path) whenever the corrector has accumulated nothing —
        so screening is byte-identical until a verified label lands.  A candidate
        whose (feed, e_core-bin) cell is covered by the serve-side Stage-1
        calibration always contributes 0 here (the corrector never observes or
        biases a fitted cell), so the two stages never stack on one cell.
        """
        corr = self.bias_corrector
        if not corr.active:
            return None
        biases = np.empty(len(candidates), dtype=float)
        nonzero = False
        for i, cand in enumerate(candidates):
            ec = predicted_e_core(cand.pattern, self.fuel, self.library_id)
            b = corr.bias(corr.key(ctx.feed, ec))
            biases[i] = b
            nonzero = nonzero or (b != 0.0)
        return biases if nonzero else None

    def _score_cell(
        self, cell: PairCell, size: int, *, wave_index: int,
        elites: Sequence[tuple[str | None, Pattern]] = (),
    ) -> tuple[CaseContext, acq.ScoredPool | None]:
        """Build + user_criteria-score a surrogate pool for a cell (no verify)."""

        ctx, pool = self._build_cell_pool(cell, size, wave_index=wave_index,
                                          elites=elites)
        return ctx, self._score_pool(ctx, pool)

    def _harvest(
        self, cell: PairCell, ctx: CaseContext, scored: acq.ScoredPool | None, *, keep: int
    ) -> tuple[float, list[_LeanCand]]:
        """Screen value (best in-band total) + top-``keep`` in-band lean cands."""

        if scored is None:
            return float("-inf"), []
        region = np.flatnonzero(scored.in_region)
        if region.size == 0:
            return float("-inf"), []
        order = region[np.argsort(-scored.exploit[region])]
        best_val = float(scored.exploit[int(order[0])])
        out: list[_LeanCand] = []
        for i in order[: max(1, int(keep))]:
            i = int(i)
            cand = scored.candidates[i]
            out.append(_LeanCand(
                pair=cell.pair, candidate=cand, ctx=ctx,
                score=float(scored.exploit[i]), p_feas=float(scored.p_feas[i]),
                mean=np.asarray(scored.mean[i], dtype=float),
                e_core=predicted_e_core(cand.pattern, self.fuel, self.library_id),
            ))
        return best_val, out

    def _deepen_cell(self, cell: PairCell) -> list[_LeanCand]:
        """Larger surrogate pool + a criteria-score hill-climb for one top cell.

        Bug B: the elite parents are this-run verified rows PLUS the store's
        verified elites, so the deepened pool's mutation children + local-search
        seeds live near a KNOWN optimum (not just random draws)."""

        seeds = list(self._cell_elites(cell.pair)) + self._store_elite_seeds(cell.pair)
        ctx, scored = self._score_cell(
            cell, self.lean_deep_pool, wave_index=1, elites=seeds
        )
        if scored is None:
            return []
        _, cands = self._harvest(cell, ctx, scored, keep=int(self.criteria.lean_top_k))
        cands.extend(self._lean_local_search(cell, ctx, scored))
        return cands

    def _lean_local_search(
        self, cell: PairCell, ctx: CaseContext, scored: acq.ScoredPool
    ) -> list[_LeanCand]:
        """Surrogate-only first-improvement hill-climb on the criteria score."""

        region = np.flatnonzero(scored.in_region)
        if region.size == 0:
            return []
        ls = self.search.local_search
        top_m = min(int(ls.top_m), region.size)
        neighbors = max(1, int(ls.neighbors) // (8 if self.dry_run else 1))
        depth = max(1, int(ls.depth) // (2 if self.dry_run else 1))
        budget = int(ls.max_predictions) // (4 if self.dry_run else 1)
        seeds = [int(i) for i in region[np.argsort(-scored.exploit[region])][:top_m]]
        seen = {c.record_id for c in scored.candidates} | set(self.ledger_ids)
        out: list[_LeanCand] = []
        spent = 0
        for seed in seeds:
            if spent >= budget:
                break
            current = scored.candidates[seed]
            current_score = float(scored.exploit[seed])
            for _ in range(depth):
                if spent >= budget:
                    break
                neighbours: list[Candidate] = []
                for _ in range(neighbors):
                    try:
                        child = mutate(
                            current.genome, self.rng, 1,
                            feed_move_prob=0.0, batches=ctx.batches,
                        )
                        pattern = child.to_pattern()
                    except GenomeError:
                        continue
                    rid = candidate_record_id(pattern, ctx)
                    if rid in seen:
                        continue
                    seen.add(rid)
                    # Surrogate-only chain: anchor on the nearest ancestor that is
                    # a real store row, never on the unverified ``current``.
                    neighbours.append(
                        Candidate(pattern, child, "local",
                                  lineage_anchor(current, self.ledger_ids),
                                  rid, ctx.e_core)
                    )
                if not neighbours:
                    break
                sub = acq.score_pool_user_criteria(
                    self.model, ctx, neighbours, self.spec,
                    fuel=self.fuel, library_id=self.library_id,
                    e_core_target=self.criteria.e_core_target,
                    e_core_tol=self.criteria.e_core_tol,
                    cyclen_bias=self._cyclen_bias_for(neighbours, ctx),
                )
                spent += len(neighbours)
                reg = np.flatnonzero(sub.in_region)
                if reg.size == 0:
                    break
                best = int(reg[np.argmax(sub.exploit[reg])])
                if float(sub.exploit[best]) > current_score + 1.0e-9:
                    current = neighbours[best]
                    current_score = float(sub.exploit[best])
                    out.append(_LeanCand(
                        pair=cell.pair, candidate=current, ctx=ctx,
                        score=current_score, p_feas=float(sub.p_feas[best]),
                        mean=np.asarray(sub.mean[best], dtype=float),
                        e_core=predicted_e_core(current.pattern, self.fuel, self.library_id),
                    ))
                else:
                    break
        return out

    def _select_top_k(self, registry: Sequence[_LeanCand]) -> list[_LeanCand]:
        """Global top-K by criteria score with a Hamming floor + per-pair cap.

        Pass 1 enforces both the pairwise Hamming diversity floor and the per-pair
        cap.  Pass 2 (only if short of K) relaxes the Hamming floor but KEEPS the
        per-pair cap — "don't put all eggs in one pair" is a firm promise.
        """

        best_by_rid: dict[str, _LeanCand] = {}
        for lc in registry:
            # Bug B: store-verified elites are predictions only — they inform the
            # screen/registry but must NEVER consume a MASTER verification slot
            # (ledger dedup); only NEW candidates (incl. their mutation children)
            # are eligible for the top-K wave.
            if lc.verified:
                continue
            rid = lc.candidate.record_id
            cur = best_by_rid.get(rid)
            if cur is None or lc.score > cur.score:
                best_by_rid[rid] = lc
        ranked = sorted(best_by_rid.values(), key=lambda lc: (-lc.score, lc.candidate.record_id))

        k = int(self.criteria.lean_top_k)
        if self.budget > 0:
            k = min(k, max(0, self.budget - self.budget_spent))
        cap = int(self.criteria.lean_per_pair_cap)
        hmin = int(self.criteria.lean_hamming_min)

        selected: list[_LeanCand] = []
        per_pair: dict[str, int] = {}

        def _try_add(lc: _LeanCand, *, enforce_hamming: bool) -> bool:
            if len(selected) >= k:
                return False
            if per_pair.get(lc.pair, 0) >= cap:
                return False
            if enforce_hamming and any(
                lc.candidate.pattern.hamming(s.candidate.pattern) < hmin for s in selected
            ):
                return False
            selected.append(lc)
            per_pair[lc.pair] = per_pair.get(lc.pair, 0) + 1
            return True

        for lc in ranked:
            _try_add(lc, enforce_hamming=True)
        if len(selected) < k:
            chosen = {s.candidate.record_id for s in selected}
            for lc in ranked:
                if lc.candidate.record_id in chosen:
                    continue
                if _try_add(lc, enforce_hamming=False):
                    chosen.add(lc.candidate.record_id)
        return selected

    def _log_predicted_table(self, selected: Sequence[_LeanCand]) -> None:
        self._log(
            f"[user_criteria][lean] predicted top-{len(selected)} "
            "(pair | score | p_feas | F_r F_q CBC cyclen |ASI|):"
        )
        for rank, lc in enumerate(selected, 1):
            m = lc.mean
            self._log(
                f"  {rank:>2} {lc.pair:>8} | {lc.score:10.2f} | {lc.p_feas:5.2f} | "
                f"{m[0]:5.3f} {m[2]:5.3f} {m[1]:7.1f} {m[3]:6.1f} {m[4]:5.3f}"
            )

    def _verify_batch(self, selected: Sequence[_LeanCand], *, round_tag: str) -> float:
        """Verify the selected top-K in ONE batched full-width wave; record labels.

        Returns the wall-clock seconds of the ``evaluate_wave`` dispatch."""

        if not selected:
            return 0.0
        resolved_cache: dict[str, Any] = {}
        pending: list[_Pending] = []
        for lc in selected:
            ck = lc.ctx.case_key
            if lc.pair not in resolved_cache:
                resolved_cache[lc.pair] = self.resolver.resolve(ck)
            entry = WaveEntry(
                lc.candidate.pattern, ck, resolved_cache[lc.pair],
                {"pair": lc.pair, "slot": "lean", "phase": f"lean_{round_tag}",
                 "record_id": lc.candidate.record_id},
            )
            pending.append(_Pending(
                entry=entry, cand=lc.candidate, total=lc.score,
                cell=self.cell_runs[lc.pair].cell, phase=f"lean_{round_tag}",
                wave=1, e_core=lc.e_core,
            ))
            self.ledger_ids.add(lc.candidate.record_id)

        t0 = time.perf_counter()
        outcomes = self.verifier.evaluate_wave([p.entry for p in pending])   # ONE wave
        dt = time.perf_counter() - t0

        by_rid = {lc.candidate.record_id: lc for lc in selected}
        self._record_outcomes(pending, outcomes)
        # honest precision measurement: predicted vs verified-actual per candidate.
        for pend, outcome in zip(pending, outcomes, strict=True):
            lc = by_rid[pend.cand.record_id]
            row = outcome_to_record(
                outcome, dataset="P", library_id=self.library_id,
                deck_knobs=PRODUCE_DECK_KNOBS,
            ).to_record()
            feasible, _ = self._is_criteria_feasible(row)
            # Stage-2 running corrector: record the RAW cyclen over-prediction for
            # this cell (no-op on a Stage-1-fitted cell or a non-finite label).
            if row.get("converged"):
                self.bias_corrector.observe(
                    self.bias_corrector.key(lc.ctx.feed, lc.e_core),
                    float(lc.mean[3]), row.get("cyclen"),
                )
            self.lean_rows.append({
                "pair": lc.pair, "record_id": lc.candidate.record_id,
                "round": round_tag, "score": lc.score, "p_feas": lc.p_feas,
                "status": outcome.status, "feasible": feasible,
                "pred": {"f_r": float(lc.mean[0]), "cbc_max": float(lc.mean[1]),
                         "f_q": float(lc.mean[2]), "cyclen": float(lc.mean[3]),
                         "ao_abs": float(lc.mean[4])},
                "actual": {"f_r": row.get("f_r"), "cbc_max": row.get("cbc_max"),
                           "f_q": row.get("f_q"), "cyclen": row.get("cyclen"),
                           "ao_abs": row.get("ao_abs")},
            })
        self._persist_bias()
        return dt

    def _persist_bias(self) -> None:
        """Persist the running cyclen bias for resume (best-effort, never fatal)."""
        try:
            self.bias_corrector.save(self.bias_path)
        except OSError:
            pass

    def _prior_sigma(self, stats: Sequence[acq.OuterCellStat]) -> float:
        vals = [s.screen_value for s in stats if math.isfinite(s.screen_value)]
        if len(vals) >= 2:
            return max(float(np.std(vals)), 1.0e-6)
        return 1.0

    # -- post-verification hook (SDM/MTC) ---------------------------------- #
    def _maybe_post_verify(self) -> None:
        top_k = int(self.criteria.post_verify_topk)
        if top_k <= 0:
            return
        try:
            from . import sdm_mtc
        except Exception as exc:  # noqa: BLE001
            self._log(f"[user_criteria] sdm_mtc unavailable; post-verify skipped ({exc})")
            return
        hook = getattr(sdm_mtc, "post_verify_topk", None)
        if hook is None:
            self._log("[user_criteria] sdm_mtc.post_verify_topk absent; post-verify skipped")
            return
        if self.dry_run or not self.cfg.master.executable or self.best is None:
            self._log(
                "[user_criteria] SDM/MTC post-verify is a no-op here "
                "(needs live [master].executable + a feasible top-K); "
                f"top_k={top_k} carried for the live run"
            )
            return
        try:
            spec = {
                "top_k": top_k,
                "mtc_limit": self.spec.mtc_limit, "sdm_limit": self.spec.sdm_limit,
                "mtc_delta_c": self.cfg.sdm_mtc.mtc_delta_c,
                "mtc_output_units": self.cfg.sdm_mtc.mtc_output_units,
            }
            # run_post_verification's real-MASTER executor needs BOTH executable
            # and package_root (to stage MAS_XSL/MAS_HFF for the %EXE_RHO branch);
            # omitting package_root raised "needs master_cfg {executable,
            # package_root}" and skipped the whole post-verify.
            master_cfg = {
                "executable": self.cfg.master.executable,
                "package_root": (
                    str(self._resolve(self.cfg.verify.package_root))
                    if self.cfg.verify.package_root else None
                ),
                "timeout": self.cfg.master.timeout,
            }
            results = hook(self.run_dir, spec, master_cfg,
                           sidecar_path=self.cfg.sdm_mtc.sidecar_path)
            self._log(f"[user_criteria] SDM/MTC post-verify: {len(results)} candidate(s) verified")
        except Exception as exc:  # noqa: BLE001 — post-verify must never fail the run
            self._log(f"[user_criteria][WARNING] SDM/MTC post-verify failed: {exc}")

    # -- results + IO ------------------------------------------------------ #
    def _write_status(self, status: str) -> None:
        _atomic_json(self.run_dir / "status.json", {
            "status": status, "mode": "user_criteria",
            "search_mode": self.criteria.search_mode,
            "budget": self.budget, "budget_spent": self.budget_spent,
            "universe": sum(1 for c in self.universe if c.included),
            "best": self.best, "dry_run": self.dry_run,
            "screen_seconds": round(self.screen_seconds, 2),
            "verify_seconds": round(self.verify_seconds, 2),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    def _result(self, status: str) -> CampaignResult:
        n_feasible = sum(1 for r in self.cell_runs.values() if r.best_row is not None)
        return CampaignResult(
            run_dir=str(self.run_dir), status=status, waves=len(self.race_timeline),
            budget=self.budget, budget_spent=self.budget_spent,
            n_feasible=n_feasible,
            on_target=n_feasible, best=self.best,
            wave_reports=list(self.race_timeline),
        )

    def _render_report(self, result: CampaignResult) -> None:
        try:
            text = self._build_report_md(result)
            (self.run_dir / "report.md").write_text(text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — a report failure never fails the run
            self._log(f"[user_criteria][WARNING] report generation failed: {exc}")

    def _build_report_md(self, result: CampaignResult) -> str:
        c = self.criteria
        included = [x for x in self.universe if x.included]
        excluded = [x for x in self.universe if not x.included]
        L: list[str] = []
        L.append(f"# user_criteria FREE-SEARCH — {self.run_dir.name}\n")
        L.append(f"- mode: `user_criteria` / `{c.search_mode}` (plan sec. 12.5), "
                 f"library `{self.library_id}`, feed `{self.feed}`")
        L.append(f"- e_core target: **{c.e_core_target} +/- {c.e_core_tol}** "
                 f"(split in [{c.split_range[0]}, {c.split_range[1]}])")
        L.append(f"- cyclen target: **{c.cyclen_target} +/- {c.cyclen_tol}** EFPD")
        dt = "off" if c.discharge_target is None else f"{c.discharge_target} +/- {c.discharge_tol} MWd/kgU"
        L.append(f"- discharge_burnup criterion: {dt}")
        L.append(f"- budget: {self.budget_spent}/{self.budget} MASTER calls "
                 f"({'dry-run/stub' if self.dry_run else 'live MASTER'})")
        gates = ", ".join(self.spec.gated_axes()) or "(none)"
        L.append(f"- gated constraint axes (SET): {gates}")
        L.append("")

        # universe
        L.append("## Pair universe\n")
        L.append(f"- enumerated pairs: **{len(self.universe)}**")
        L.append(f"- e_core-reachable (denominator): **{len(included)}**")
        L.append(f"- excluded (unreachable): **{len(excluded)}**")
        cross = [x for x in included if not x.mono and x.type_a[0] != x.type_b[0]]
        L.append(f"- notable cross-family reachable pairs: "
                 + (", ".join(x.pair for x in cross[:10]) if cross else "(none)"))
        reasons: dict[str, int] = {}
        for x in excluded:
            key = ("above band" if "above" in x.reason
                   else "below band" if "below" in x.reason else x.reason)
            reasons[key] = reasons.get(key, 0) + 1
        L.append("- excluded-pair reasons: "
                 + ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())))
        L.append("")

        # activated cells table
        L.append("## Activated cells (post wave-0 screen)\n")
        L.append("| pair | e_core reach | screen value | calls | best F_r | best cyclen | feasible |")
        L.append("|------|--------------|-------------|-------|----------|-------------|----------|")
        runs = sorted(self.cell_runs.values(),
                      key=lambda r: (-(r.stat.value() if math.isfinite(r.stat.value()) else -1e18)))
        for r in runs:
            if not r.stat.active and r.calls == 0 and not r.stat.eliminated:
                continue
            reach = f"[{r.cell.e_lo:.3f}, {r.cell.e_hi:.3f}]"
            sv = "n/a" if not math.isfinite(r.stat.screen_value) else f"{r.stat.screen_value:.2f}"
            bf = "-" if r.best_fr is None else f"{r.best_fr:.3f}"
            bc = "-" if (r.best_row is None or r.best_row.get("cyclen") is None) else f"{float(r.best_row['cyclen']):.1f}"
            feas = "yes" if r.best_row is not None else "no"
            elim = " (elim)" if r.stat.eliminated else ""
            L.append(f"| {r.cell.pair}{elim} | {reach} | {sv} | {r.calls} | {bf} | {bc} | {feas} |")
        L.append("")

        # racing timeline
        L.append("## Racing timeline\n")
        for ev in self.race_timeline:
            if ev.get("phase") == "exploit":
                L.append(f"- wave {ev['wave']} exploit alloc: {ev.get('alloc', {})}")
            else:
                el = ev.get("eliminated") or []
                L.append(f"- wave {ev['wave']} ({ev['phase']}): "
                         f"{len(ev.get('active', []))} active"
                         + (f", eliminated {el}" if el else ""))
        L.append("")

        # best LP with all 7 axes + criteria bands
        L.append("## Best LP (all 7 constraint axes + criteria bands)\n")
        if self.best is None:
            L.append("No criteria-feasible LP was found within budget.")
        else:
            b = self.best
            disch = self._discharge_estimate(b.get("cyclen"))
            L.append(f"- pair: **{b['pair']}**, record `{str(b.get('record_id'))[:16]}`")
            L.append(f"- e_core (fresh feed): {_fmt_num(b.get('e_core'))}")
            L.append("")
            L.append("| axis | value | limit | status |")
            L.append("|------|-------|-------|--------|")
            L.append(_axis_row("F_r (objective)", b.get("f_r"), self.spec.f_r_limit))
            L.append(_axis_row("F_q", b.get("f_q"), self.spec.f_q_limit))
            L.append(_axis_row("CBC_max", b.get("cbc_max"), self.spec.cbc_limit))
            L.append(_axis_row("|ASI| (==|AO|)", b.get("ao_abs"), self.spec.asi_abs_limit))
            L.append(_axis_row("pin burnup", b.get("max_pin_burnup"), self.spec.pin_bu_limit))
            L.append(f"| MTC | post-verify | {_fmt_num(self.spec.mtc_limit)} | SDM/MTC stage |")
            L.append(f"| SDM | post-verify | {_fmt_num(self.spec.sdm_limit)} | SDM/MTC stage |")
            L.append("")
            L.append("| criterion band | value | target | tol | in band |")
            L.append("|----------------|-------|--------|-----|---------|")
            L.append(_band_row("cyclen [EFPD]", b.get("cyclen"), c.cyclen_target, c.cyclen_tol))
            if c.discharge_target is not None:
                L.append(_band_row("discharge_burnup [MWd/kgU] (est.)", disch,
                                   c.discharge_target, c.discharge_tol))
            L.append(_band_row("e_core", b.get("e_core"), c.e_core_target, c.e_core_tol))
            L.append("")
            L.append(f"- discharge_burnup energy-balance estimate: **{_fmt_num(disch)}** MWd/kgU "
                     f"(post-hoc; P={c.power_mw} MW, HM={c.hm_mtu} MTU, residence 241/{self.feed}; "
                     "approximation — the verify FOM carries no discharge axis).")
        L.append("")

        # known verified LPs already in the store matching ALL criteria (Bug B):
        # the user sees existing solutions even when this run's new wave finds none.
        try:
            L.extend(known_verified_lps_table_md(self._known_verified_store_lps()))
        except Exception as exc:  # noqa: BLE001 — a table failure never fails the report
            L.append("## Known verified LPs in store matching the criteria\n")
            L.append(f"_(table unavailable: {exc})_")
        L.append("")

        # lean: predicted-vs-actual precision table + wall-time breakdown.
        if self.lean_rows:
            L.append("## Predicted vs actual (lean top-K)\n")
            L.append("The honest precision measurement of the lean promise: the "
                     "surrogate prediction that ranked each verified candidate vs "
                     "its MASTER-verified value.\n")
            L.append("| pair | status | feas | F_r pred/act | F_q pred/act | "
                     "CBC pred/act | cyclen pred/act | score |")
            L.append("|------|--------|------|-------------|-------------|"
                     "-------------|-----------------|-------|")
            n_feas = 0
            for r in self.lean_rows:
                p, a = r["pred"], r["actual"]
                if r["feasible"]:
                    n_feas += 1
                L.append(
                    f"| {r['pair']} | {r['status']} | "
                    f"{'yes' if r['feasible'] else 'no'} | "
                    f"{_fmt_num(p['f_r'])}/{_fmt_num(a['f_r'])} | "
                    f"{_fmt_num(p['f_q'])}/{_fmt_num(a['f_q'])} | "
                    f"{_fmt_num(p['cbc_max'], 0)}/{_fmt_num(a['cbc_max'], 0)} | "
                    f"{_fmt_num(p['cyclen'], 1)}/{_fmt_num(a['cyclen'], 1)} | "
                    f"{_fmt_num(r['score'], 1)} |"
                )
            L.append("")
            L.append(f"- lean precision: **{n_feas}/{len(self.lean_rows)}** verified "
                     "candidates met all criteria bands "
                     f"({'converged' if any(r['status'] == 'converged' for r in self.lean_rows) else 'none converged'}).")
            L.append("")
            L.append("## Wall-time breakdown\n")
            L.append(f"- screen + deepen (surrogate CPU): **{self.screen_seconds:.1f} s**")
            L.append(f"- verification wave (MASTER): **{self.verify_seconds:.1f} s**")
            L.append(f"- verified candidates: **{len(self.lean_rows)}** "
                     f"in {'2 batched waves' if any(r['round'] == 'r2' for r in self.lean_rows) else '1 batched wave'} "
                     f"(top-K={self.criteria.lean_top_k}, per-pair cap {self.criteria.lean_per_pair_cap}, "
                     f"Hamming >= {self.criteria.lean_hamming_min})")
            L.append("")

        # trajectory
        L.append("## Convergence to feasibility\n")
        cf = self.first_feasible_call
        L.append(f"- calls-to-first-feasible: {cf if cf is not None else 'not reached'}")
        L.append(f"- criteria-feasible cells: {result.n_feasible}")
        if self.fr_trajectory:
            pts = [f"({p['call']}, {_fmt_num(p['best_fr'])})" for p in self.fr_trajectory
                   if p['best_fr'] is not None]
            L.append("- best-F_r trajectory (call, F_r): "
                     + (" ".join(pts) if pts else "no feasible point yet"))
        L.append("")
        L.append("*(GA-600 overlay omitted — different mode; the calls-to-first-feasible "
                 "and best-F_r trajectory replace it, per plan sec. 12.5 item 7.)*")
        return "\n".join(L) + "\n"

    def _discharge_estimate(self, cyclen: Any) -> float | None:
        if cyclen is None:
            return None
        try:
            from ..design.bootstrap import estimate_discharge_burnup

            return estimate_discharge_burnup(
                float(cyclen), self.feed,
                power_mw=self.criteria.power_mw, hm_mtu=self.criteria.hm_mtu,
            )
        except Exception:  # noqa: BLE001
            return None


def _normalize_remote_screening(value: Any) -> str:
    """Coerce ``[model] remote_screening`` to ``"auto"`` / ``"on"`` / ``"off"``.

    Accepts the TOML bool ``true``/``false`` or the string ``"auto"`` (plus the
    usual truthy/falsey spellings); anything unrecognised is ``"off"`` (local).
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    s = str(value).strip().lower()
    if s == "auto":
        return "auto"
    if s in ("true", "1", "yes", "on"):
        return "on"
    return "off"


def _resolve_inference_mode(model_cfg: Any) -> str:
    """Resolve the effective screening mode from ``[model] inference`` (+ legacy).

    The explicit ``inference`` selector wins when set:

    * ``"local_cpu"`` (aliases ``local`` / ``cpu``) -> ``"off"`` (never route).
    * ``"remote_gpu"`` (aliases ``remote`` / ``gpu``) -> ``"auto"``: probe the
      server once (5 s), attach the GPU screener if it answers, and fall back to
      local CPU — loudly — on an unreachable server or any per-batch error.  This
      is what lets an explicitly remote campaign survive a server outage.

    Unset (empty ``""``) defers to the legacy ``remote_screening`` key so a deck
    that never mentions ``inference`` behaves EXACTLY as it did before this key
    existed.  An unrecognised ``inference`` string is treated as unset (defer).
    """
    inference = str(getattr(model_cfg, "inference", "") or "").strip().lower()
    if inference in ("local_cpu", "local", "cpu"):
        return "off"
    if inference in ("remote_gpu", "remote", "gpu"):
        return "auto"
    # Unset / unrecognised: preserve today's behaviour (remote_screening governs).
    return _normalize_remote_screening(getattr(model_cfg, "remote_screening", False))


def cell_target_e_core(cell: PairCell, criteria: Any) -> float:
    """Nominal in-band e_core for a cell (target clamped into its reach interval)."""

    t = float(criteria.e_core_target)
    if math.isfinite(cell.e_lo) and math.isfinite(cell.e_hi):
        return min(max(t, cell.e_lo), cell.e_hi)
    return t


def _fmt_num(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _axis_row(name: str, value: Any, limit: Any) -> str:
    if limit is None:
        return f"| {name} | {_fmt_num(value)} | report-only | - |"
    try:
        ok = value is not None and float(value) <= float(limit)
        status = "ok" if ok else "OVER"
    except (TypeError, ValueError):
        status = "?"
    return f"| {name} | {_fmt_num(value)} | {_fmt_num(limit)} | {status} |"


def _band_row(name: str, value: Any, target: Any, tol: Any) -> str:
    try:
        ib = value is not None and abs(float(value) - float(target)) <= float(tol)
        status = "yes" if ib else "no"
    except (TypeError, ValueError):
        status = "?"
    return f"| {name} | {_fmt_num(value)} | {_fmt_num(target)} | {_fmt_num(tol)} | {status} |"


def known_verified_lps_table_md(rows: Sequence[dict[str, Any]], *, top: int = 10) -> list[str]:
    """Markdown lines for the "known verified LPs in store matching the criteria"
    table (Bug B item 3).  ``rows`` are store rows already filtered to the search's
    own feasibility (converged + feed + e_core band + every gated SET limit + the
    cyclen band), sorted best-F_r first, each optionally annotated with a post-hoc
    discharge estimate (``disch_est`` / ``disch_in_band``).  Always emits a heading
    so the user sees whether existing solutions exist even when the new wave finds
    none.
    """

    L: list[str] = ["## Known verified LPs in store matching the criteria\n"]
    if not rows:
        L.append(
            "_No converged store LP matches the criteria (feed, e_core band, all "
            "gated SET limits, and the cyclen band). The store holds no ready-made "
            "solution for these criteria._"
        )
        return L
    L.append(
        f"**{len(rows)}** converged store LP(s) already satisfy the criteria "
        "(feed, e_core band, all gated SET limits, and the cyclen band) — existing "
        "solutions independent of this run's new wave. The discharge column is a "
        "post-hoc estimate (the verify FOM carries no discharge axis), shown for "
        "reference, not a hard gate. Best by F_r:\n"
    )
    L.append("| # | pair | record | F_r | F_q | CBC | cyclen | e_core | disch(est) | n_cyc |")
    L.append("|---|------|--------|-----|-----|-----|--------|--------|-----------|-------|")
    for i, row in enumerate(rows[: max(1, int(top))], 1):
        disch = row.get("disch_est")
        flag = row.get("disch_in_band")
        disch_cell = _fmt_num(disch, 1)
        if disch is not None and flag is not None:
            disch_cell += " (in-band)" if flag else " (est. oob)"
        L.append(
            f"| {i} | {row.get('case_pair')} | "
            f"`{str(row.get('record_id'))[:16]}` | "
            f"{_fmt_num(row.get('f_r'))} | {_fmt_num(row.get('f_q'))} | "
            f"{_fmt_num(row.get('cbc_max'), 0)} | {_fmt_num(row.get('cyclen'), 1)} | "
            f"{_fmt_num(row.get('e_core'))} | {disch_cell} | "
            f"{_fmt_num(row.get('n_cycles'), 0)} |"
        )
    if len(rows) > top:
        L.append(f"\n_(+{len(rows) - top} more not shown)_")
    return L


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def run_campaign(
    cfg: LpoptConfig,
    model_backend: Any,
    evaluator_factory: Callable | None = None,
    **kwargs: Any,
) -> CampaignResult:
    """Build the campaign driver for ``cfg.case.mode`` and run it.

    ``fixed`` runs the guided-search :class:`CampaignDriver`; ``user_criteria``
    (plan sec. 12.5) runs the FREE-SEARCH :class:`UserCriteriaDriver` (outer racing
    over the pair universe).  ``feed_range`` / ``free`` parse and validate but
    raise here with a clear deferred-milestone message.
    """

    cfg.case.validate()
    # [constraints] drives the D9 SDM/MTC pre-delivery gate for EVERY mode, so it
    # is validated before either driver is built (a bad limit must fail the deck,
    # not the post-verification stage 3 hours into the run).
    cfg.constraints.validate()
    if cfg.case.mode == "user_criteria":
        cfg.criteria.validate()
        driver = UserCriteriaDriver(cfg, model_backend, evaluator_factory, **kwargs)
        return driver.run()
    if cfg.case.mode != "fixed":
        raise NotImplementedError(
            f"[case] mode {cfg.case.mode!r} is parsed and validated but not yet "
            "implemented; implemented modes are 'fixed' (guided search) and "
            "'user_criteria' (plan sec. 6.2 / 12.5). "
            "Use mode = \"fixed\" or \"user_criteria\"."
        )
    driver = CampaignDriver(cfg, model_backend, evaluator_factory, **kwargs)
    return driver.run()


__all__ = [
    "CampaignDriver", "CampaignResult", "FxySigmaBarLost", "UserCriteriaDriver",
    "WaveReport", "checkpoint_fxy_serve_sigma", "run_campaign",
]
