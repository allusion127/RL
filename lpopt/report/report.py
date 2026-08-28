"""Campaign report assembly (``report.md`` + figures) — plan sec. 7.

The report is built entirely from the persisted run artefacts
(``labels.jsonl`` + ``waves/wave_XX/{selection,results}.json`` + ``status.json``),
so ``lpopt report`` regenerates it for any ``runs/<ts>`` without re-running the
campaign.  It carries:

* the best verified LPs (full FOM + margins to every limit + n_cycles + digest +
  candidates path),
* a per-wave table + parity / budget-curve / p_feasible-reliability figures,
* the **GA-600 overlay** — the K1_K2 event log parsed into a best-feasible
  objective-vs-#chains curve for the GA vs this campaign, plus a
  chains-to-first-feasible table, and
* an honest-scope footer (StubEvaluator dry-run caveat when applicable).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from . import figures as F

# The report's feasible set IS the campaign's feasible set.  It is imported, not
# restated: the restatement had already drifted (it omitted the pin-BU gate the
# campaign applies, so a report could call a row feasible that the campaign had
# rejected).  ``campaign`` imports this module lazily, so there is no cycle.
from ..search.campaign import (
    feasibility_limits_for as _campaign_feasibility_limits,
    is_feasible as _campaign_is_feasible,
)

# Minimal SDM/MTC post-verification integration hook (plan 12.5, additive).
# Re-exported so campaign/report can licence-verify a run's top-K feasible
# candidates *after* the fact — NOT wired into build_report / the live campaign
# loop (the curriculum agent owns campaign.py).
from ..search.sdm_mtc import post_verify_topk, write_verdict_table  # noqa: F401

_LIMITS = {"f_r": 1.55, "cbc_max": 1550.0, "f_q": 2.41, "ao_abs": 0.30}

#: Objectives for which F_r is NOT a feasibility criterion (program §10 STOP:
#: ``report.py``'s ``_LIMITS["f_r"] = 1.55`` is DEMOTED to a reported column).
#:
#: ``flat_power`` retired F_r from the objective and gates it at its own safety
#: limit (1.70, decision D1); ``fr_boundary`` / ``max_cycle_min_fr`` make F_r a
#: pure objective with no limit at all.  Judging their rows against 1.55 marked
#: campaign-valid results infeasible, which emptied the "Best verified loading
#: patterns" table, the budget curve and the GA overlay of a run whose every
#: label the campaign itself had accepted.  The licensing number is still
#: printed for every row — as a margin, which is what 1.55 is.
_FR_UNGATED_OBJECTIVES = frozenset(
    {"flat_power", "fr_boundary", "max_cycle_min_fr"}
)

#: Objectives whose campaign feasibility set carries a max-pin-burnup gate, and
#: the deck default it gates at.  Restating F_r but SILENTLY DROPPING this gate is
#: how the report came to call rows feasible that the campaign had rejected; the
#: predicate is now shared (:func:`campaign.is_feasible`), and these are the
#: values used when ``lpopt report`` runs with no deck to resolve them from.
_PIN_BU_GATED_OBJECTIVES = frozenset({"flat_power", "fr_boundary", "min_fuel_cost",
                                      "min_fr_max_cycle"})
_DEFAULT_PIN_BU_LIMIT = 80.0
#: ``min_fr_max_cycle`` gates TIGHTER than the LEU+ 80 the others use — the 2.0
#: haircut is model margin on the predicted pin head (``config.minfr_pin_bu_limit``).
_MINFR_PIN_BU_LIMIT = 78.0

#: ``min_fuel_cost`` is the one mode whose feasible set has a CYCLEN BAND (both
#: edges are hard constraints — ``campaign.feasibility_limits_for``), and these
#: are its deck defaults, used when ``lpopt report`` runs with no deck.  The
#: deck-less path learned the pin-BU gate but not this one, so a deck-less report
#: of a min_fuel_cost run listed out-of-band rows the campaign had rejected.
_DEFAULT_FUELCOST_CYCLEN_LO = 615.0
_DEFAULT_FUELCOST_CYCLEN_HI = 635.0


# --------------------------------------------------------------------------- #
# report-only engineering-rule axis (L-03 peripheral power share)
# --------------------------------------------------------------------------- #
def _peripheral_shares(store_dir: Any, record_ids: Sequence[str]
                       ) -> dict[str, float]:
    """``{record_id: RM5}`` for the rows whose harvested map is reachable.

    RM5 (:func:`..search.rule_metrics.rm_peripheral_power_share`) is the
    outer-ring share of total core power — the vessel-fluence proxy of
    low-leakage rule **L-03**.  It is REPORT-ONLY, and deliberately computed here
    rather than stored: it is derived from the same harvested BOC map as
    ``node_peak`` / ``map_cov``, so it carries no independent label, and the
    store's canonical schema is frozen.  Every failure mode (no store dir, no
    ``maps.npz``, an unreadable or odd map) yields a missing entry and the report
    prints ``n/a`` — a diagnostic column must never be able to break a report.
    """
    if not store_dir or not record_ids:
        return {}
    try:
        from ..data.store import StoreReader
        from ..search.rule_metrics import rm_peripheral_power_share
        reader = StoreReader(store_dir)
        if not reader.has_maps:
            return {}
        out: dict[str, float] = {}
        for rid in record_ids:
            maps = reader.maps(str(rid))
            if maps is None:
                continue
            share = rm_peripheral_power_share(maps)
            if share == share:                        # not NaN
                out[str(rid)] = float(share)
        return out
    except Exception:  # noqa: BLE001 - a report diagnostic never breaks a report
        return {}


# --------------------------------------------------------------------------- #
# artefact readers
# --------------------------------------------------------------------------- #
def _read_labels(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "labels.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_selection(run_dir: Path) -> dict[str, dict[str, Any]]:
    by_rid: dict[str, dict[str, Any]] = {}
    for sel_path in sorted((run_dir / "waves").glob("wave_*/selection.json")):
        try:
            data = json.loads(sel_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for item in data.get("selection", []):
            by_rid[str(item.get("record_id"))] = item
    return by_rid


def _fr_limit(limits: dict[str, Any]) -> float | None:
    """The F_r limit feasibility is judged at, or ``None`` when F_r is ungated."""
    value = limits.get("f_r")
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _feasible(fom: dict[str, Any], limits: dict[str, Any]) -> bool:
    """The CAMPAIGN's feasibility predicate (:func:`campaign.is_feasible`).

    Deliberately a one-line delegation.  Restating the rule here is what let the
    two definitions drift: the report judged CBC / F_q / |AO| / F_r and silently
    dropped the ``max_pin_burnup`` gate the campaign applies in ``flat_power`` /
    ``fr_boundary`` / ``min_fuel_cost``, so a row the campaign had REJECTED could
    be listed as a verified feasible LP.

    ``limits`` follows :data:`campaign.FEASIBILITY_LIMIT_KEYS`; a missing key or a
    non-finite ``f_r`` means that axis is ungated for the objective (reported,
    never a rejection).
    """
    resolved = dict(limits)
    resolved["f_r"] = _fr_limit(limits)      # non-finite / absent -> ungated
    return _campaign_is_feasible(fom, resolved)


def _record_fom(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record.get(k) for k in ("f_r", "cbc_max", "f_q", "cyclen", "ao_abs",
                                       "n_cycles", "node_peak", "map_cov",
                                       # the campaign gates on it; a report that
                                       # never carried the column could not.
                                       "max_pin_burnup")}


def _report_objective(fom: dict[str, Any], objective: str, target: float
                      ) -> float | None:
    """Higher-is-better report objective of one verified row (``None`` = unscorable).

    Mirrors ``campaign._campaign_objective`` in SHAPE, not in units: the report
    reads only the persisted FOM columns, so ``flat_power`` ranks on the primary
    flatness label ``node_peak`` (lower is flatter) rather than on the per-cell
    normalized scalar, whose normalizer lives in the run's ``state.json``.  What
    matters is that it is not ``−|cyclen − 625|``, which for a flatness campaign
    ordered the table by an axis the search never optimized.
    """
    if objective == "flat_power":
        peak = fom.get("node_peak")
        if peak is None:
            return None
        try:
            value = -float(peak)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    if objective == "fr_boundary":
        f_r = fom.get("f_r")
        if f_r is None:
            return None
        try:
            return -float(f_r)
        except (TypeError, ValueError):
            return None
    cyclen = fom.get("cyclen")
    if cyclen is None:
        return None
    try:
        return -abs(float(cyclen) - float(target))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# trajectories
# --------------------------------------------------------------------------- #
def _campaign_trajectory(labels: Sequence[dict[str, Any]], target: float,
                         limits: dict[str, Any],
                         objective: str = "target_cycle",
                         ) -> tuple[list[int], list[float], int | None]:
    chains, best, first = [], [], None
    running = float("-inf")
    n = 0
    for row in labels:
        record = row.get("record") or {}
        n += 1
        fom = _record_fom(record)
        obj = (_report_objective(fom, objective, target)
               if record.get("converged") and _feasible(fom, limits) else None)
        if obj is not None:
            if obj > running:
                running = obj
            if first is None:
                first = n
        chains.append(n)
        best.append(running if math.isfinite(running) else float("nan"))
    return chains, best, first


def read_ga_600(log_path: Path, target: float,
                limits: dict[str, float]) -> tuple[list[int], list[float], int | None, int, int]:
    """Parse ``ga_generations_*.jsonl`` -> (chains, best_obj, first_feasible, n, n_feasible)."""

    chains, best, first = [], [], None
    running = float("-inf")
    n = n_feasible = 0
    if not log_path.exists():
        return chains, best, first, 0, 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return chains, best, first, 0, 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for entry in obj.get("batch", []) or []:
            fom = entry.get("fom")
            if not isinstance(fom, dict):
                # an error/failed chain still consumes a MASTER call.
                n += 1
                chains.append(n)
                best.append(running if math.isfinite(running) else float("nan"))
                continue
            n += 1
            feasible = bool(entry.get("feasible")) and bool(entry.get("eq_ok", True))
            cyclen = fom.get("cyclen")
            if feasible and cyclen is not None:
                n_feasible += 1
                cand = -abs(float(cyclen) - target)
                if cand > running:
                    running = cand
                if first is None:
                    first = n
            chains.append(n)
            best.append(running if math.isfinite(running) else float("nan"))
    return chains, best, first, n, n_feasible


#: The fully-hydrated GA-600 snapshot (plan sec. 1-9); other runs_flow logs are
#: partial/earlier runs, so this baseline is preferred for the overlay.
_GA600_SNAPSHOT = "20260713_061541"


def _find_ga_log(base: Path, ga_root: str, runs_flow: str, pair: str) -> Path | None:
    root = base / ga_root if not Path(ga_root).is_absolute() else Path(ga_root)
    if not (root / runs_flow).exists():
        return None
    hits = list((root / runs_flow).glob(f"*/stages/ga_generations_{pair}.jsonl"))
    if not hits:
        return None

    def _size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return -1

    # Prefer the named GA-600 baseline snapshot; else the fullest (largest) log.
    hits.sort(key=lambda p: (_GA600_SNAPSHOT in str(p), _size(p)), reverse=True)
    return hits[0]


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        fv = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if not math.isfinite(fv) else f"{fv:.{nd}f}"


def build_report(
    run_dir: str | Path,
    *,
    pair: str,
    target_efpd: float = 625.0,
    cycle_tolerance: float = 2.0,
    limits: dict[str, Any] | None = None,
    dry_run: bool = True,
    ga_log: Path | None = None,
    library_id: str = "ga80",
    ood_warnings: Sequence[str] | None = None,
    objective: str | None = None,
    store_dir: str | Path | None = None,
    log=None,
) -> Path:
    """Build ``report.md`` + figures for one ``runs/<ts>`` from its artefacts.

    ``store_dir`` — OPTIONAL record store.  When given (and it carries a
    ``maps.npz``), the ``flat_power`` best-LP table gains the REPORT-ONLY
    engineering-rule axis **RM5** (peripheral power share, low-leakage rule
    L-03: the outer-ring share of core power, a vessel fast-fluence proxy that
    radial flattening RAISES).  Absent, unreadable, or map-less, the column is
    simply not emitted and the report is unchanged — the axis is a diagnostic,
    never a criterion, and nothing about feasibility or ranking reads it.

    ``ood_warnings`` — optional serve-time feature/geometry OOD lines (from
    ``PosValCnnBackend.feature_ood_warning``, review sec. 4b) — are rendered as a
    warning block so a campaign that scored a pin-pitch/pin-radius geometry variant
    flags "prediction unvalidated" alongside the verified-LP table.

    ``objective`` — the campaign objective.  It selects the FEASIBILITY definition
    (F_r is a criterion only where the mode gates on it, program §10 STOP) and the
    ranking scalar.  ``None`` falls back to ``status.json``'s recorded objective,
    then to ``target_cycle``, so ``lpopt report`` on an old run is unchanged.

    Feasibility itself is :func:`campaign.is_feasible` — the run's own predicate,
    applied to the run's own limits (:func:`campaign.feasibility_limits_for`).
    The report does not restate it: when it did, it dropped the ``max_pin_burnup``
    gate and reported rows the campaign had rejected.
    """

    run_dir = Path(run_dir)
    log = log or (lambda m: None)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels = _read_labels(run_dir)
    selection = _read_selection(run_dir)
    status = _read_json(run_dir / "status.json")

    objective = str(objective or status.get("objective") or "target_cycle")
    supplied = limits is not None
    limits = dict(limits) if supplied else dict(_LIMITS)
    if not supplied:
        if objective in _FR_UNGATED_OBJECTIVES:
            # DEMOTE F_r to a reported column (program §10 STOP): a mode that
            # retired F_r from the objective must not have its rows judged
            # infeasible by 1.55.  A caller that KNOWS the mode resolves the right
            # value itself and passes it (``write_campaign_report`` hands
            # flat_power its 1.70 safety gate, and ``None`` for the modes with no
            # F_r gate at all) — that always wins.
            limits["f_r"] = None
        # …and pick up the pin-BU gate those modes DO apply, so a deck-less
        # ``lpopt report`` still reproduces the campaign's feasible set.
        if objective not in _PIN_BU_GATED_OBJECTIVES:
            limits["max_pin_burnup"] = None
        elif objective == "min_fr_max_cycle":
            limits["max_pin_burnup"] = _MINFR_PIN_BU_LIMIT
        else:
            limits["max_pin_burnup"] = _DEFAULT_PIN_BU_LIMIT
        # …and min_fuel_cost's CYCLEN BAND, the other axis of its six-constraint
        # feasible set.  Learning the pin-BU gate but not this one left the same
        # hole one axis over: a deck-less report of a min_fuel_cost run called
        # out-of-band rows feasible that the campaign had rejected.
        if objective == "min_fuel_cost":
            limits["cyclen_lo"] = _DEFAULT_FUELCOST_CYCLEN_LO
            limits["cyclen_hi"] = _DEFAULT_FUELCOST_CYCLEN_HI
        else:
            limits["cyclen_lo"] = limits["cyclen_hi"] = None

    # -- parity + reliability points ---------------------------------------- #
    points, pf_list, feas_list = [], [], []
    for row in labels:
        record = row.get("record") or {}
        rid = str(row.get("record_id"))
        sel = selection.get(rid)
        fom = _record_fom(record)
        is_feas = bool(record.get("converged")) and _feasible(fom, limits)
        if sel is not None and record.get("converged"):
            points.append({
                "origin": row.get("origin", sel.get("origin", "")),
                "pred": sel.get("pred_mean"),
                "actual": fom,
            })
            pf_list.append(float(sel.get("p_feas", 0.0)))
            feas_list.append(is_feas)

    # -- trajectories ------------------------------------------------------- #
    c_chains, c_best, c_first = _campaign_trajectory(labels, target_efpd, limits,
                                                     objective)
    # The GA-600 baseline is a TARGET-CYCLE trajectory (−|cyclen−625| of its own
    # feasible rows).  Overlaying it on a flatness / F_r-boundary curve would put
    # two different scalars on one axis, so it is read only when comparable.
    ga_comparable = objective not in _FR_UNGATED_OBJECTIVES
    ga_chains, ga_best, ga_first, ga_n, ga_feas = ([], [], None, 0, 0)
    if ga_log is not None and ga_comparable:
        ga_chains, ga_best, ga_first, ga_n, ga_feas = read_ga_600(ga_log, target_efpd, limits)

    # -- figures ------------------------------------------------------------ #
    obj_label = _objective_axis_label(objective, target_efpd)
    fig_paths: dict[str, Path | None] = {}
    fig_paths["parity"] = F.parity_figure(points, figures_dir / "parity.png")
    fig_paths["budget"] = F.budget_curve_figure(
        c_chains, c_best, figures_dir / "budget_curve.png",
        target_efpd=target_efpd, objective_label=obj_label)
    fig_paths["reliability"] = F.p_feas_reliability_figure(pf_list, feas_list, figures_dir / "p_feas_reliability.png")
    fig_paths["ga_overlay"] = F.ga_overlay_figure(
        c_chains, c_best, ga_chains, ga_best, figures_dir / "ga600_overlay.png",
        target_efpd=target_efpd, objective_label=obj_label,
    )

    # -- best verified LPs -------------------------------------------------- #
    verified = []
    for row in labels:
        record = row.get("record") or {}
        fom = _record_fom(record)
        if not (record.get("converged") and _feasible(fom, limits)):
            continue
        obj = _report_objective(fom, objective, target_efpd)
        if obj is None:                       # unscorable on THIS objective's axis
            continue
        verified.append((row, record, fom, obj))
    verified.sort(key=lambda t: t[3], reverse=True)

    if verified:
        best_row, best_record, best_fom, _ = verified[0]
        try:
            from ..data.schema import unpack_pattern
            best_pat = unpack_pattern(str(best_record["pattern"]))
            fig_paths["quarter"] = F.quarter_core_figure(
                best_pat, pair, figures_dir / "best_lp_quarter.png",
                title=f"Best verified LP — {pair} feed {best_record.get('feed')}",
            )
        except Exception:  # noqa: BLE001
            fig_paths["quarter"] = None

    # -- markdown ----------------------------------------------------------- #
    lines: list[str] = []
    lines.append(f"# lpopt campaign report — {run_dir.name}")
    lines.append("")
    lines.append(f"- case: **{pair} / feed {status.get('case', '').split('-')[-1] or '?'}**  ")
    lines.append(f"- status: **{status.get('status', '?')}**  budget "
                 f"{status.get('budget_spent', len(labels))}/{status.get('budget', '?')}  ")
    lines.append(f"- objective: {_objective_line(objective, target_efpd, cycle_tolerance)}  ")
    lines.append(f"- feasibility: {_feasibility_line(limits)}  ")
    lines.append(f"- verified feasible LPs: **{len(verified)}** / {len(labels)} evaluations  ")
    lines.extend(_sdm_mtc_section(run_dir, status))
    lines.append("")

    lines.append("## Best verified loading patterns")
    lines.append("")
    if not verified:
        lines.append("_No verified feasible LP within limits was found this campaign._")
        lines.append("")
    else:
        flat = objective == "flat_power"
        # REPORT-ONLY L-03 axis; emitted only when the maps are actually there.
        top_rows = verified[:5]
        rm5 = _peripheral_shares(
            store_dir, [str(rec.get("record_id")) for _r, rec, _f, _o in top_rows]
        ) if flat else {}
        head = ["Rank", "cyclen", "\\|Δ625\\|"]
        if flat:
            head += ["node_peak", "map_cov"]
        if rm5:
            head += ["periph share (L-03)"]
        head += ["F_r (margin)", "CBC (margin)", "F_q (margin)",
                 "\\|AO\\| (margin)", "n_cyc", "record_id"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for rank, (row, record, fom, _) in enumerate(verified[:5], start=1):
            def _m(name: str) -> str:
                """``value (+margin)`` against whatever limit applies to ``name``.

                When the mode ungates F_r (``limits["f_r"] is None``) the margin
                is still printed — against the LICENSING limit 1.55, tagged
                ``lic``, because 1.55 remains the licensing number even where it
                is no longer a feasibility criterion (program §10 KEEP)."""
                v = fom.get(name)
                if v is None:
                    return "n/a"
                limit = limits.get(name)
                if limit is None or not math.isfinite(float(limit)):
                    if name != "f_r":
                        return _fmt(v)
                    margin = _LIMITS["f_r"] - float(v)
                    return f"{_fmt(v)} (lic {'+' if margin >= 0 else ''}{_fmt(margin)})"
                return f"{_fmt(v)} (+{_fmt(float(limit) - float(v))})"
            cyclen = fom.get("cyclen")
            dist = _fmt(abs(float(cyclen) - target_efpd)) if cyclen is not None else "n/a"
            cells = [str(rank), _fmt(cyclen, 1), dist]
            if flat:
                cells += [_fmt(fom.get("node_peak")), _fmt(fom.get("map_cov"), 4)]
            if rm5:
                cells += [_fmt(rm5.get(str(record.get("record_id"))), 4)]
            cells += [
                _m("f_r"),
                f"{_fmt(fom.get('cbc_max'), 1)} "
                f"(+{_fmt(limits['cbc_max'] - float(fom['cbc_max']), 1)})",
                _m("f_q"), _m("ao_abs"), _fmt(fom.get("n_cycles"), 0),
                f"`{str(record.get('record_id'))[:16]}…`",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if rm5:
            lines.append(
                "> **periph share (L-03)** — multiplicity-weighted share of core "
                "power in the OUTERMOST assembly ring, from the same harvested BOC "
                "map as `node_peak` / `map_cov`. A REPORT AXIS ONLY: it is the "
                "vessel fast-fluence / baffle-heating proxy that radial flattening "
                "RAISES, so it is what a flatness gain is spent on. It is NOT a "
                "criterion, NOT part of feasibility, and NOT a predictor — being "
                "same-map derived, its correlation with the flatness columns is an "
                "identity, not skill."
            )
            lines.append("")
        lines.append(f"Feasible verified LPs are archived (3_GA-compatible) under "
                     f"`{run_dir.name}/candidates/`.")
        lines.append("")

    # serve-time feature / geometry OOD guard (review sec. 4b) — advisory only.
    ood_lines = [str(w) for w in (ood_warnings or []) if str(w).strip()]
    if ood_lines:
        lines.append("## Feature / geometry OOD warnings")
        lines.append("")
        lines.append("> These candidates touched a fuel type whose harvested features "
                     "fall outside the training population envelope (e.g. a pin-pitch / "
                     "pin-radius geometry variant). The prediction is **unvalidated** — "
                     "gate it through the DeCART blind-probe transfer test (review sec. 4c) "
                     "before trusting it. This is a warning only; predictions are unchanged.")
        lines.append("")
        for w in ood_lines:
            lines.append(f"- {w}")
        lines.append("")

    # figures
    lines.append("## Figures")
    lines.append("")
    for key, caption in (
        ("quarter", "Best verified LP — quarter-core batch/age map"),
        ("budget", "Best feasible objective vs budget (step curve)"),
        ("ga_overlay", "Guided search vs GA-600 baseline"),
        ("parity", "Per-target parity — surrogate vs verified (origin-coloured)"),
        ("reliability", "p_feasible reliability (selected candidates)"),
    ):
        p = fig_paths.get(key)
        if p is not None:
            lines.append(f"**{caption}**")
            lines.append("")
            lines.append(f"![{key}](figures/{Path(p).name})")
            lines.append("")

    # GA-600 comparison table
    lines.append("## GA-600 baseline comparison")
    lines.append("")
    lines.append(f"| Method | MASTER calls | feasible | chains-to-first-feasible | "
                 f"best objective ({obj_label}) |")
    lines.append("|---|---|---|---|---|")
    c_bestval = next((b for b in reversed(c_best) if math.isfinite(b)), None)
    lines.append(
        f"| lpopt campaign | {len(labels)} | {len(verified)} | "
        f"{c_first if c_first is not None else '—'} | {_fmt(c_bestval, 2)} |"
    )
    if not ga_comparable:
        lines.append(
            f"| GA-600 (K1_K2) | _not comparable — the GA baseline optimises "
            f"−\\|cyclen−{target_efpd:.0f}\\|, this run optimises {obj_label}_ "
            f"| — | — | — |"
        )
    elif ga_n:
        ga_bestval = next((b for b in reversed(ga_best) if math.isfinite(b)), None)
        lines.append(
            f"| GA-600 (K1_K2) | {ga_n} | {ga_feas} | "
            f"{ga_first if ga_first is not None else '—'} | {_fmt(ga_bestval, 2)} |"
        )
    else:
        lines.append("| GA-600 (K1_K2) | _event log not found_ | — | — | — |")
    lines.append("")

    # per-wave table
    lines.append("## Per-wave summary")
    lines.append("")
    lines.append("| Wave | slots (E/X/C) | converged | feasible | gate | τ |")
    lines.append("|---|---|---|---|---|---|")
    for wave, data in _wave_summaries(run_dir, limits):
        lines.append(
            f"| {wave} | {data['slots']} | {data['converged']} | {data['feasible']} | "
            f"{data['gate']} | {data['tau']} |"
        )
    lines.append("")

    # honest footer
    lines.append("## Scope & honesty")
    lines.append("")
    if dry_run:
        lines.append(
            "> **Dry-run (StubEvaluator).** These figures exercise the full guided-search "
            "machinery — pool construction, surrogate scoring, trust-region gate, wave "
            "composition, online update — against a **deterministic fake MASTER**, because "
            "the FEASIBLE_PACKAGE assets are OneDrive-dehydrated (no live MASTER run is "
            "possible today). The verified FOMs are synthetic; the numbers are not "
            "engineering results. Flip to a live MASTER evaluator (drop `--dry-run`) with a "
            "hydrated package to obtain real labels — the code path is otherwise identical."
        )
    else:
        lines.append(
            f"> Live MASTER labels. Feasibility is judged against "
            f"{_feasibility_line(limits)} with the "
            f"{_objective_line(objective, target_efpd, cycle_tolerance)}."
        )
    if _fr_limit(limits) is None:
        lines.append("")
        lines.append(
            f"> **F_r is NOT a feasibility criterion for `{objective}`** (program §10). "
            f"It is reported per row with its margin to the LICENSING limit "
            f"{_LIMITS['f_r']:.2f}, which is where that number belongs — judging this "
            f"mode's rows by it marked campaign-valid results infeasible."
        )
    # The single-gate limitation, stated where the feasibility claim is made
    # (:func:`_recorded_fr_gate`).  Rendered only when the run demonstrably used
    # more than one gate, so an ordinary run's footer is unchanged.
    note = _recorded_fr_gate_note(status, objective)
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[report] wrote {report_path}")
    return report_path


#: One-line descriptions of what each objective optimizes (report header).
_OBJECTIVE_TEXT = {
    "flat_power": "flat_power — minimise node_peak (primary) + map_cov "
                  "(secondary); F_r is a safety gate, cyclen is record-only",
    "fr_boundary": "fr_boundary — minimise F_r as a pure objective; "
                   "cyclen recorded, never gated",
    "max_cycle_min_fr": "max_cycle_min_fr — maximise cyclen, minimise F_r "
                        "(F_r ungated)",
    "min_fr_max_cycle": "min_fr_max_cycle — minimise F_r, then maximise cyclen",
    "min_fuel_cost": "min_fuel_cost — minimise fresh-fuel charge within the "
                     "cyclen band",
}


def _objective_axis_label(objective: str, target_efpd: float) -> str:
    """Short name of the scalar :func:`_report_objective` returns (figure axes)."""
    if objective == "flat_power":
        return "−node_peak"
    if objective == "fr_boundary":
        return "−F_r"
    return f"−|cyclen−{target_efpd:.0f}|"


def _objective_line(objective: str, target_efpd: float, tolerance: float) -> str:
    text = _OBJECTIVE_TEXT.get(objective)
    if text is not None:
        return text
    return (f"target_cycle {target_efpd:.0f} EFPD (±{tolerance:.0f}), "
            f"minimise Max CBC within window")


def _feasibility_line(limits: dict[str, Any]) -> str:
    """The feasibility set actually applied, F_r included only where it gates."""
    parts = []
    fr = _fr_limit(limits)
    if fr is not None:
        parts.append(f"F_r ≤ {fr:.2f}")
    else:
        parts.append("F_r **not gated** (reported only)")
    parts.append(f"CBC_max ≤ {float(limits['cbc_max']):.0f} ppm")
    parts.append(f"F_q ≤ {float(limits['f_q']):.2f}")
    parts.append(f"|AO| ≤ {float(limits['ao_abs']):.2f}")
    # printed only where the mode gates on it — the axis the report used to apply
    # nowhere while the campaign applied it in three modes.
    pin_bu = limits.get("max_pin_burnup")
    if pin_bu is not None:
        parts.append(f"max pin BU ≤ {float(pin_bu):.0f} MWd/kgU")
    lo, hi = limits.get("cyclen_lo"), limits.get("cyclen_hi")
    if lo is not None and hi is not None:
        parts.append(f"cyclen ∈ [{float(lo):.0f}, {float(hi):.0f}] EFPD")
    return ", ".join(parts)


def _wave_summaries(run_dir: Path, limits: dict[str, Any] | None = None
                    ) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for res_path in sorted((run_dir / "waves").glob("wave_*/results.json")):
        sel_path = res_path.with_name("selection.json")
        try:
            res = json.loads(res_path.read_text(encoding="utf-8"))
            sel = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
        except (json.JSONDecodeError, OSError):
            continue
        wave = int(res.get("wave", 0))
        results = res.get("results", [])
        slots = {"exploit": 0, "explore": 0, "control": 0}
        converged = feasible = on_target = 0
        for r in results:
            slots[r.get("slot", "exploit")] = slots.get(r.get("slot", "exploit"), 0) + 1
            fom = r.get("fom")
            if r.get("status") == "converged":
                converged += 1
            if fom and _feasible(
                {k.lower() if k != "CBC_max" else "cbc_max": v
                 for k, v in _norm_fom(fom).items()},
                limits or _LIMITS,
            ):
                feasible += 1
        gate = res.get("gate", {})
        out.append((wave, {
            "slots": f"{slots['exploit']}/{slots['explore']}/{slots['control']}",
            "converged": converged, "feasible": feasible, "on_target": on_target,
            "gate": f"{gate.get('mode', '?')}{'+' if gate.get('accepted') else '-'}",
            "tau": _fmt(sel.get("tau"), 2),
        }))
    return out


def _norm_fom(fom: dict[str, Any]) -> dict[str, Any]:
    """A wave ``results.json`` FOM (``FOM.as_dict`` keys) in feasibility keys.

    ``max_pin_burnup`` is carried through: the campaign GATES on it in
    ``flat_power`` / ``fr_boundary`` / ``min_fuel_cost``, and dropping it here
    fed :func:`_feasible` a ``None`` for every wave-sourced row — the pin-BU gate
    could then never reject one, so the per-wave "feasible" count silently
    counted rows the campaign itself had rejected.  ``FOM.as_dict`` provides the
    key under its own name, so there is nothing to translate.
    """
    return {
        "f_r": fom.get("F_r"), "cbc_max": fom.get("CBC_max"), "f_q": fom.get("F_q"),
        "cyclen": fom.get("cyclen"),
        "ao_abs": (max(abs(fom.get("AO_min") or 0.0), abs(fom.get("AO_max") or 0.0))
                   if (fom.get("AO_min") is not None or fom.get("AO_max") is not None) else None),
        "max_pin_burnup": fom.get("max_pin_burnup"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def _sdm_mtc_section(run_dir: Path, status: dict) -> list[str]:
    """Report lines for the D9 SDM/MTC pre-delivery gate (empty when it never ran).

    Two budgets are always printed as two numbers.  Folding the gate's MASTER
    calls into ``budget_spent`` would make the search look more expensive than it
    was and the licensing stage free; keeping them apart is what makes the
    ``+N licensing`` line auditable against ``sdm_mtc.json``.
    """
    calls = int(status.get("post_verify_master_calls", 0) or 0)
    summary = _read_json(run_dir / "sdm_mtc.json")
    if not calls and not summary:
        return []
    out = [
        f"- SDM/MTC pre-delivery gate (D9): **{len(summary.get('results') or [])}** "
        f"candidate(s) verified, **{len(summary.get('violators') or [])}** violator(s), "
        f"**+{calls}** MASTER call(s) on top of the search budget  "
    ]
    if summary.get("report_only", True):
        out.append("  - **REPORT-ONLY** — no user limit is set in `[constraints]`, so "
                   "no candidate can be marked a violator; the numbers are measured, "
                   "not cleared  ")
    for entry in (summary.get("skipped") or []):
        out.append(f"  - skipped `{entry.get('record_id')}`: {entry.get('reason')}  ")
    return out


def write_campaign_report(driver: Any, result: Any) -> Path:
    """Called by the campaign driver at completion (has full config context)."""

    cfg = driver.cfg
    base = driver._base
    ga_log = _find_ga_log(base, cfg.extract.ga_root, cfg.extract.ga_runs_flow, driver.ctx.pair)
    objective = str(getattr(driver, "objective", "target_cycle"))
    # Ask the driver what IT gated at (``feasibility_limits`` is the campaign's
    # own resolver, so the report's feasible set cannot differ from the run's).
    resolver = getattr(driver, "feasibility_limits", None)
    limits = dict(resolver()) if callable(resolver) else {
        "f_r": _driver_fr_limit(driver, objective), "cbc_max": driver.acq.cbc_limit,
        "f_q": driver.acq.f_q_limit, "ao_abs": driver.acq.ao_abs_limit,
    }
    return build_report(
        driver.run_dir, pair=driver.ctx.pair, target_efpd=driver.acq.cycle_target_efpd,
        cycle_tolerance=driver.acq.cycle_tolerance_efpd, limits=limits,
        dry_run=driver.dry_run, ga_log=ga_log, library_id=driver.library_id,
        objective=objective,
        store_dir=getattr(driver, "campaign_store_dir", None),
        log=driver._log,
    )


def _acq_fr_limit(acq: Any, objective: str) -> float | None:
    """The F_r limit ``objective`` gates at per the deck (``None`` = no gate).

    ``flat_power`` uses its own SAFETY gate (``flatpower_fr_limit``, 1.70 —
    decision D1) and NOT ``f_r_limit`` (1.55); ``fr_boundary`` /
    ``max_cycle_min_fr`` make F_r a pure objective with no gate at all.
    """
    if objective == "flat_power":
        return float(getattr(acq, "flatpower_fr_limit", 1.7))
    if objective in _FR_UNGATED_OBJECTIVES:
        return None
    return float(acq.f_r_limit)


def _driver_fr_limit(driver: Any, objective: str) -> float | None:
    """The F_r limit the DRIVER actually gated at (``None`` when it gated none).

    The report's feasible set must be the campaign's feasible set
    (``campaign._is_feasible``), so this prefers the live ``flat_power`` spec —
    the only place the D1 per-cell bias correction has been applied — and falls
    back to the deck value when there is no spec.
    """
    if objective == "flat_power":
        spec = getattr(driver, "flat_power_spec", None)
        if spec is not None:
            return float(spec.fr_gate)
    return _acq_fr_limit(driver.acq, objective)


def _recorded_fr_gates(status: dict[str, Any], objective: str) -> list[float]:
    """Every DISTINCT F_r gate recorded in ``status.json``, in precedence order.

    ``status.json`` carries the campaign row dicts for ``best`` and
    ``best_overall`` only, and each carries the gate ITS row was judged at as
    ``f_r_limit_applied``.  Two entries can disagree — see
    :func:`_recorded_fr_gate` — so both are read rather than the first.
    """
    if objective != "flat_power":
        return []
    seen: list[float] = []
    for key in ("best", "best_overall"):
        entry = status.get(key)
        if not isinstance(entry, dict):
            continue
        try:
            value = float(entry["f_r_limit_applied"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value not in seen:
            seen.append(value)
    return seen


def _recorded_fr_gate(status: dict[str, Any], objective: str) -> float | None:
    """The F_r gate the RUN applied, read back from ``status.json``.

    ``flat_power`` gates F_r at its own safety limit (D1) AFTER the per-cell
    map-head bias correction, so the deck's ``flatpower_fr_limit`` (1.70) is an
    UPPER bound on what the run really enforced — every campaign row carries the
    gate it was judged at as ``f_r_limit_applied``.  ``lpopt report`` has no live
    driver to ask, so it reads that number back; ``None`` (no such row, or another
    objective) means the deck value HOLDS, which is
    :func:`campaign.feasibility_limits_for`'s documented fallback.

    Judging a flat_power run at 1.70 when it ran at, say, 1.62 is not a cosmetic
    difference: it re-admits exactly the rows the campaign's safety gate rejected.

    KNOWN LIMITATION (stated, not hidden).  This returns ONE number and
    :func:`build_report` applies it to EVERY row, while the gate is a per-row
    property: it is ``flat_power_spec.fr_gate``, which is fixed for the lifetime
    of one :class:`..search.campaign.CampaignDriver` but is rebuilt whenever a
    driver is — a resumed run under a re-fitted ``map_calibration.json``, or a
    different cell of an outer cell race, writes rows judged at a DIFFERENT gate
    into the same run dir.  The per-row gate cannot be recovered here:
    ``labels.jsonl`` carries the MASTER record, not the campaign row, so
    ``f_r_limit_applied`` survives only on the two rows ``status.json`` keeps.
    When those two disagree the report says so —
    :func:`_recorded_fr_gate_note` renders the limitation into ``report.md``
    instead of silently judging the whole run at one of the two.
    """
    gates = _recorded_fr_gates(status, objective)
    return gates[0] if gates else None


def _recorded_fr_gate_note(status: dict[str, Any], objective: str) -> str | None:
    """Report text for the limitation in :func:`_recorded_fr_gate`, or ``None``.

    Emitted only when ``status.json``'s recorded gates actually DISAGREE — that is
    the observable proof the run did not use a single gate throughout, and the one
    case where judging every row at the first value is known to mis-classify rows.
    """
    gates = _recorded_fr_gates(status, objective)
    if len(gates) < 2:
        return None
    applied = gates[0]
    others = ", ".join(f"{g:.4f}" for g in gates[1:])
    return (
        f"> **F_r gate limitation.** This run recorded MORE THAN ONE applied F_r "
        f"safety gate (`f_r_limit_applied` = {applied:.4f} and {others}) — the D1 "
        f"gate is rebuilt per driver, so a resume under a re-fitted "
        f"`map_calibration.json`, or a different cell of an outer cell race, "
        f"judged part of this run at a different limit. `lpopt report` cannot "
        f"recover the per-row gate (`labels.jsonl` carries the MASTER record, not "
        f"the campaign row), so **every row below is judged at "
        f"{applied:.4f}**. Rows originally judged at a looser gate may be listed "
        f"infeasible here, and rows judged at a tighter one may be listed "
        f"feasible; `status.json`'s `best` / `best_overall` carry the gate each "
        f"was actually judged at."
    )


def regenerate_report(run_dir: str | Path, cfg: Any = None, *, log=None) -> Path:
    """``lpopt report`` entry: rebuild ``report.md`` for an existing run dir."""

    run_dir = Path(run_dir)
    pair = "K1_K2"
    target = 625.0
    tol = 2.0
    limits: dict[str, Any] | None = None
    dry_run = True
    ga_log = None
    library_id = "ga80"
    objective = None
    status = _read_json(run_dir / "status.json")
    if cfg is not None:
        pair = str(cfg.case.pair or pair)
        target = float(cfg.acquisition.cycle_target_efpd)
        tol = float(cfg.acquisition.cycle_tolerance_efpd)
        objective = str(getattr(cfg.acquisition, "objective", "target_cycle"))
        # the campaign's own resolver — including the pin-BU gate a re-generated
        # report used to drop (and then report rows the run had rejected), and
        # the F_r gate the run ACTUALLY applied: without ``fr_gate`` this resolved
        # flat_power's deck 1.70 while the run had run at its bias-corrected
        # (tighter) D1 gate, so ``lpopt report`` judged the run at a limit it
        # never used — the defect ``write_campaign_report`` already avoids by
        # asking the live driver.
        limits = dict(_campaign_feasibility_limits(
            cfg.acquisition, objective,
            fr_gate=_recorded_fr_gate(status, objective)))
        library_id = cfg.model.library_id
        base = cfg.source_path.parent if cfg.source_path else Path.cwd()
        ga_log = _find_ga_log(base, cfg.extract.ga_root, cfg.extract.ga_runs_flow, pair)
    if "dry_run" in status:
        dry_run = bool(status["dry_run"])
    return build_report(
        run_dir, pair=pair, target_efpd=target, cycle_tolerance=tol, limits=limits,
        dry_run=dry_run, ga_log=ga_log, library_id=library_id,
        objective=objective, log=log,
    )


__all__ = [
    "build_report", "post_verify_topk", "read_ga_600", "regenerate_report",
    "write_campaign_report", "write_verdict_table",
]
