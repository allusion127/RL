"""Sample-efficiency KPI instrument (A1–A4) — read-only, MASTER 0 calls.

Implements phase **P0-2 / P0-3** of
``data/reports/active_frontier_loop_spec_20260903.md`` §5, using the K1–K4
definitions registered in ``data/reports/sample_efficiency_kpi_20260903.md`` §5
verbatim:

* **K1 / A1 ``calls_to_frontier``** — the smallest cumulative MASTER call ``n``
  at which the run's running best on the objective axis reaches
  ``R_cell + epsilon`` (``epsilon = 0.005`` on ``f_r``).  ``R_cell`` is the
  cell record **frozen at campaign launch** (``kpi_baseline.json``, written by
  :func:`freeze_baseline` from the store before the first wave).  §5 K1 declares
  a post-hoc computation INVALID, so when no frozen baseline exists this module
  still reports a number but stamps ``frozen=false`` / ``valid=false`` on it and
  never silently presents it as A1.
* **K2 ``calls_to_incumbent``** — the smallest ``n`` beating the incumbent (the
  cell's best from OTHER campaigns) frozen at the same moment.  ``0`` reachable
  at call 1 means "no new information".
* **K3 ``AUF@N``** — ``(1/N) Σ_{n=1..N} (best_n − R_cell)``, lower is better,
  reported at ``N = 100`` and ``N = 300`` (and at the run's own budget).
* **K4 companions (mandatory)** — ``screen_ratio`` (surrogate evaluations per
  MASTER call), ``prior_rows`` (the cell's store rows at launch — the seeding
  cost proxy), ``first_feasible_call``, ``n_feasible/N``, ``delta_per_100calls``
  and wall-hours.

**A2** (error reduction per label) is the ``ood_snapshot_pre.json`` /
``ood_snapshot_post.json`` pair: the pre snapshot is the surrogate's error on
the cell's EXISTING labelled rows at launch (plus the coverage/OOD flags of the
cell), the post snapshot is the same model's error on the labels the campaign
then bought — a genuine holdout for the launch champion, computed from the
``pred_mean`` already recorded per selected candidate in
``waves/*/selection.json`` so ``--post`` needs no model and no GPU.

**A4** (frontier updates from extrapolation) is counted here from the
per-candidate ``ood_flag`` in ``selection.json`` joined to the labels that
improved the running best.

Nothing in this module calls MASTER, loads a checkpoint (except through the
``predict`` callable a live campaign hands :func:`snapshot_pre`), or writes
outside the run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

#: ``surrogate.TARGET_NAMES`` column order of ``selection.json``'s ``pred_mean``
#: mapped onto the store/record column names the labels carry.
PRED_COLUMNS: tuple[str, ...] = (
    "f_r", "cbc_max", "f_q", "cyclen", "ao_abs",
    "max_assembly_burnup", "max_pin_burnup",
)

#: A1's registered tolerance on the F_r axis (spec §4d).
DEFAULT_EPSILON = 0.005

#: The K3 budgets reported for every run (spec §5 K3).
AUF_BUDGETS: tuple[int, ...] = (100, 300)

BASELINE_NAME = "kpi_baseline.json"
KPI_NAME = "kpi.json"
PRE_NAME = "ood_snapshot_pre.json"
POST_NAME = "ood_snapshot_post.json"
LABELS_NAME = "labels.jsonl"


# --------------------------------------------------------------------------- #
# small IO helpers (self-contained: this module is also run stand-alone)
# --------------------------------------------------------------------------- #
def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(value: Any) -> float | None:
    """``float(value)`` or ``None`` for absent / non-finite / unparseable."""

    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_labels(run_dir: str | Path) -> list[dict[str, Any]]:
    """Every ``labels.jsonl`` row of a run, in write order (== call order)."""

    path = Path(run_dir) / LABELS_NAME
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


# --------------------------------------------------------------------------- #
# cell identity + the frozen record (A1's denominator)
# --------------------------------------------------------------------------- #
def cell_rows(
    records: Any,
    *,
    pair: str | None,
    feed: int | None,
    library_id: str | None = None,
    exclude_campaign: str | None = None,
):
    """The store rows of one cell (``pair × feed [× library]``), converged only.

    ``exclude_campaign`` drops the run's OWN rows, which is what makes a
    recomputed ``R_cell`` mean "the record BEFORE this campaign" rather than
    "the record this campaign just set" (the post-hoc trap §5 K1 names).
    """

    df = records
    if "converged" in df.columns:
        df = df[df["converged"].fillna(False).astype(bool)]
    if pair is not None and "case_pair" in df.columns:
        df = df[df["case_pair"] == pair]
    if feed is not None and "feed" in df.columns:
        df = df[df["feed"] == int(feed)]
    if library_id is not None and "library_id" in df.columns:
        df = df[df["library_id"] == library_id]
    if exclude_campaign and "campaign" in df.columns:
        df = df[df["campaign"] != exclude_campaign]
    return df


def freeze_baseline(
    store_dir: str | Path,
    *,
    pair: str | None,
    feed: int | None,
    library_id: str | None = None,
    metric: str = "f_r",
    campaign: str | None = None,
) -> dict[str, Any]:
    """The launch-time cell fingerprint: ``R_cell``, incumbent and ``prior_rows``.

    ``R_cell`` (K1) and the incumbent (K2) are the SAME number for a cold cell;
    they diverge only once a campaign is mid-flight, which is exactly why both
    are frozen at launch and never recomputed afterwards.
    """

    out: dict[str, Any] = {
        "pair": pair, "feed": (int(feed) if feed is not None else None),
        "library_id": library_id, "metric": metric, "campaign": campaign,
        "R_cell": None, "incumbent": None, "prior_rows": 0, "frozen": True,
        "source": str(store_dir),
    }
    try:
        from ..data.store import StoreReader

        records = StoreReader(store_dir).records
    except Exception:  # noqa: BLE001 — an absent/unreadable store is honest "unknown"
        out["frozen"] = False
        out["error"] = "store unreadable"
        return out
    rows = cell_rows(records, pair=pair, feed=feed, library_id=library_id,
                     exclude_campaign=campaign)
    out["prior_rows"] = int(len(rows))
    if len(rows) and metric in rows.columns:
        series = rows[metric].dropna()
        if len(series):
            out["R_cell"] = float(series.min())
            out["incumbent"] = float(series.min())
    return out


def baseline_for(
    run_dir: str | Path,
    store_dir: str | Path | None,
    *,
    metric: str = "f_r",
) -> dict[str, Any]:
    """The frozen baseline if the run has one, else a flagged post-hoc rebuild."""

    run_dir = Path(run_dir)
    frozen = _read_json(run_dir / BASELINE_NAME)
    if isinstance(frozen, dict) and frozen.get("R_cell") is not None:
        out = dict(frozen)
        out["frozen"] = True
        return out
    if store_dir is None:
        return {"R_cell": None, "incumbent": None, "prior_rows": None,
                "frozen": False, "metric": metric,
                "note": "no kpi_baseline.json and no --store; A1 not computable"}
    pair, feed, library_id = _cell_of_run(run_dir)
    out = freeze_baseline(store_dir, pair=pair, feed=feed, library_id=library_id,
                          metric=metric, campaign=run_dir.name)
    out["frozen"] = False
    out["note"] = ("recomputed post-hoc from the store (this campaign's own rows "
                   "excluded); sample_efficiency_kpi_20260903.md §5 K1 declares a "
                   "post-hoc A1 INVALID — reported for orientation only")
    return out


def _cell_of_run(run_dir: Path) -> tuple[str | None, int | None, str | None]:
    """``(pair, feed, library_id)`` of a run from its own artefacts."""

    status = _read_json(run_dir / "status.json") or {}
    baseline = _read_json(run_dir / BASELINE_NAME) or {}
    pair = baseline.get("pair")
    feed = baseline.get("feed")
    library_id = baseline.get("library_id")
    if pair is None or feed is None:
        for row in read_labels(run_dir):
            rec = row.get("record") or {}
            pair = pair or rec.get("case_pair") or row.get("pair")
            if feed is None and rec.get("feed") is not None:
                feed = rec.get("feed")
            library_id = library_id or rec.get("library_id")
            if pair is not None and feed is not None:
                break
    if pair is None:
        # ``status.json`` carries the case label ("<pair>-e<...>-f<feed>"-ish).
        label = str(status.get("case") or "")
        if label:
            pair = label.split("-")[0] or None
    return (pair, (int(feed) if feed is not None else None), library_id)


# --------------------------------------------------------------------------- #
# the running-best trajectory (the object every KPI is a functional of)
# --------------------------------------------------------------------------- #
def trajectory(
    labels: Sequence[Mapping[str, Any]],
    *,
    metric: str = "f_r",
    feasible_only: bool = False,
) -> list[dict[str, Any]]:
    """Per-call running best over the labels, in call order.

    One entry per MASTER call.  ``best`` is the running minimum of ``metric``
    over CONVERGED rows (the §1-2 convention: a per-campaign feasibility gate
    differs cell to cell, so the lineage curves are drawn on ``converged``);
    ``feasible_only=True`` switches to the run's own feasible flag.
    """

    out: list[dict[str, Any]] = []
    best: float | None = None
    calls = 0
    wall = 0.0
    for row in labels:
        calls += 1
        rec = row.get("record") or {}
        w = _num(row.get("wall_s"))
        if w is not None:
            wall += w
        converged = bool(rec.get("converged")) if "converged" in rec else (
            str(row.get("status", "")) == "converged")
        feasible = bool(row.get("feasible") or row.get("criteria_feasible"))
        use = feasible if feasible_only else converged
        value = _num(rec.get(metric))
        if use and value is not None and (best is None or value < best):
            best = value
        out.append({
            "call": (int(row.get("cumulative_master_calls"))
                     if row.get("cumulative_master_calls") is not None else calls),
            "best": best,
            "value": value,
            "converged": converged,
            "feasible": feasible,
            "wall_s": w,
            "cumulative_wall_s": round(wall, 3),
            "surrogate_evals": row.get("cumulative_surrogate_evals"),
            "record_id": rec.get("record_id") or row.get("record_id"),
            "wave": row.get("wave"),
        })
    return out


def _first_call_at_or_below(traj: Sequence[Mapping[str, Any]],
                            threshold: float) -> int | None:
    for point in traj:
        best = point.get("best")
        if best is not None and float(best) <= threshold:
            return int(point["call"])
    return None


def auf(traj: Sequence[Mapping[str, Any]], r_cell: float, budget: int) -> float | None:
    """K3 ``AUF@budget`` — mean gap to the frozen record over calls 1..budget.

    ``None`` when the run is shorter than ``budget`` (an AUF over a truncated
    horizon is not comparable to one over the full horizon, and §5 K3 requires
    an identical ``N`` for cross-cell comparison).  Calls before the first
    converged label carry no ``best``; they are charged the gap of the first
    ``best`` the run ever reaches, so an unlucky opening is not free.
    """

    if budget <= 0 or len(traj) < budget:
        return None
    firsts = [p["best"] for p in traj if p.get("best") is not None]
    if not firsts:
        return None
    seed = float(firsts[0])
    total = 0.0
    for point in traj[:budget]:
        best = point.get("best")
        total += (float(best) if best is not None else seed) - r_cell
    return total / budget


def delta_per_100calls(traj: Sequence[Mapping[str, Any]], window: int = 100) -> float | None:
    """K4 marginal gain — improvement over the last ``window`` calls, per 100.

    ``< 0.002`` is the registered "floor" verdict (spec §4b) that moves budget
    to another cell or another move set.
    """

    pts = [p for p in traj if p.get("best") is not None]
    if len(pts) < 2:
        return None
    span = min(window, len(pts) - 1)
    start = float(pts[-1 - span]["best"])
    end = float(pts[-1]["best"])
    return (start - end) * (100.0 / span)


# --------------------------------------------------------------------------- #
# A4 — frontier updates that came out of the extrapolation region
# --------------------------------------------------------------------------- #
def selection_index(run_dir: str | Path) -> dict[int, list[dict[str, Any]]]:
    """``{wave: [selection entry, ...]}`` from ``waves/wave_NN/selection.json``."""

    out: dict[int, list[dict[str, Any]]] = {}
    wdir = Path(run_dir) / "waves"
    if not wdir.is_dir():
        return out
    for path in sorted(wdir.glob("wave_*/selection.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        try:
            wave = int(payload.get("wave"))
        except (TypeError, ValueError):
            continue
        entries = payload.get("selection")
        if isinstance(entries, list):
            out[wave] = [e for e in entries if isinstance(e, dict)]
    return out


def a4_ood_frontier_updates(
    labels: Sequence[Mapping[str, Any]],
    traj: Sequence[Mapping[str, Any]],
    selections: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """A4 — labels that BOTH came from a flagged candidate AND moved the best.

    The join is positional within a wave: ``selection.json`` and ``labels.jsonl``
    are written from the same ``entries`` list in the same order, and the
    verified ``record_id`` is minted from the outcome so it does not match the
    candidate's.  ``ood_flag=None`` means "the backend exposes no guard" — it is
    NOT ``False`` (spec §4e rule 5), so it is counted separately as ``unknown``.
    """

    per_wave_pos: dict[int, int] = {}
    flagged = improved = flagged_improved = unknown = 0
    prev_best: float | None = None
    for row, point in zip(labels, traj):
        wave = row.get("wave")
        try:
            wave_i = int(wave)
        except (TypeError, ValueError):
            wave_i = -1
        pos = per_wave_pos.get(wave_i, 0)
        per_wave_pos[wave_i] = pos + 1
        entries = selections.get(wave_i) or []
        entry = entries[pos] if pos < len(entries) else {}
        flag = entry.get("ood_flag", None)
        best = point.get("best")
        moved = (best is not None
                 and (prev_best is None or float(best) < float(prev_best)))
        if moved:
            improved += 1
        if flag is None:
            unknown += 1
        elif bool(flag):
            flagged += 1
            if moved:
                flagged_improved += 1
        prev_best = best if best is not None else prev_best
    calls = len(labels)
    return {
        "n_calls": calls,
        "n_ood_flagged": flagged,
        "n_ood_unknown": unknown,
        "n_frontier_updates": improved,
        "n_ood_frontier_updates": flagged_improved,
        "rate_per_call": (flagged_improved / calls) if calls else None,
        "note": ("ood_flag=None is 'no guard exposed', counted as unknown and "
                 "never as clean (spec §4e rule 5)"),
    }


# --------------------------------------------------------------------------- #
# A1–A4 assembly
# --------------------------------------------------------------------------- #
def compute_kpi(
    run_dir: str | Path,
    store_dir: str | Path | None = None,
    *,
    metric: str = "f_r",
    epsilon: float = DEFAULT_EPSILON,
    feasible_only: bool = False,
) -> dict[str, Any]:
    """Every KPI of one run as a JSON-ready dict (no side effects)."""

    run_dir = Path(run_dir)
    labels = read_labels(run_dir)
    status = _read_json(run_dir / "status.json") or {}
    base = baseline_for(run_dir, store_dir, metric=metric)
    traj = trajectory(labels, metric=metric, feasible_only=feasible_only)
    r_cell = _num(base.get("R_cell"))
    incumbent = _num(base.get("incumbent"))
    calls = len(traj)
    final_best = next((p["best"] for p in reversed(traj) if p.get("best") is not None),
                      None)

    a1: dict[str, Any] = {
        "epsilon": epsilon, "metric": metric, "R_cell": r_cell,
        "frozen": bool(base.get("frozen")), "budget": calls,
        "calls": None, "reached": False, "final_gap": None,
        "valid": bool(base.get("frozen")),
    }
    if r_cell is not None:
        n = _first_call_at_or_below(traj, r_cell + epsilon)
        a1["calls"] = n
        a1["reached"] = n is not None
        if final_best is not None:
            a1["final_gap"] = round(float(final_best) - r_cell, 6)
        if n is None:
            a1["report"] = f">{calls}"
    if not a1["valid"]:
        a1["note"] = base.get("note") or "no frozen kpi_baseline.json — A1 is post-hoc"

    k2: dict[str, Any] = {"incumbent": incumbent, "calls": None, "reached": False}
    if incumbent is not None:
        n = _first_call_at_or_below(traj, incumbent - 1e-12)
        k2["calls"] = n
        k2["reached"] = n is not None
        if n is None:
            k2["verdict"] = "no new information (incumbent never beaten)"

    k3: dict[str, Any] = {}
    if r_cell is not None:
        for budget in (*AUF_BUDGETS, calls):
            value = auf(traj, r_cell, int(budget))
            if value is not None:
                k3[f"AUF@{int(budget)}"] = round(value, 6)

    surrogate = None
    for point in reversed(traj):
        if point.get("surrogate_evals") is not None:
            surrogate = int(point["surrogate_evals"])
            break
    wall_total = traj[-1]["cumulative_wall_s"] if traj else 0.0
    n_feasible = sum(1 for p in traj if p.get("feasible"))
    first_feasible = next((int(p["call"]) for p in traj if p.get("feasible")), None)

    a3 = _read_json(run_dir / PRE_NAME) or {}
    a3_post = _read_json(run_dir / POST_NAME) or {}

    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "status": status.get("status"),
        "objective": status.get("objective"),
        "cell": {"pair": base.get("pair"), "feed": base.get("feed"),
                 "library_id": base.get("library_id")},
        "A1_calls_to_frontier": a1,
        "K2_calls_to_incumbent": k2,
        "K3_AUF": k3,
        "A2_error_reduction": {
            "pre": a3.get("summary"),
            "post": a3_post.get("summary"),
            "per_label": a3_post.get("error_reduction_per_label"),
        },
        "A3_coverage": {
            "pre": a3.get("coverage"),
            "post": a3_post.get("coverage"),
        },
        "A4_ood_frontier": a4_ood_frontier_updates(labels, traj,
                                                   selection_index(run_dir)),
        "K4": {
            "prior_rows": base.get("prior_rows"),
            "screen_ratio": (round(surrogate / calls, 1)
                             if (surrogate and calls) else None),
            "surrogate_evals": surrogate,
            "master_calls": calls,
            "post_verify_master_calls": status.get("post_verify_master_calls"),
            "first_feasible_call": first_feasible,
            "n_feasible": n_feasible,
            "n_converged": sum(1 for p in traj if p.get("converged")),
            "delta_per_100calls": (None if delta_per_100calls(traj) is None
                                   else round(float(delta_per_100calls(traj)), 6)),
            "floor_verdict": _floor_verdict(delta_per_100calls(traj)),
            "wall_s_total": round(float(wall_total or 0.0), 1),
            "wall_hours": round(float(wall_total or 0.0) / 3600.0, 3),
            "best": final_best,
        },
        "trajectory_len": calls,
    }


def _floor_verdict(delta: float | None) -> str | None:
    """Spec §4b / K4: ``Δ/100calls < 0.002`` declares the cell a floor."""

    if delta is None:
        return None
    return "floor" if delta < 0.002 else "gaining"


def write_kpi(
    run_dir: str | Path,
    store_dir: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """:func:`compute_kpi` -> ``<run_dir>/kpi.json``; returns the path."""

    payload = compute_kpi(run_dir, store_dir, **kwargs)
    path = Path(run_dir) / KPI_NAME
    _atomic_json(path, payload)
    return path


# --------------------------------------------------------------------------- #
# A2 / A3 — before / after snapshots
# --------------------------------------------------------------------------- #
def coverage_flags(
    store_dir: str | Path | None,
    *,
    feed: int | None,
    e_core: float | None,
    e_core_band: float = 0.05,
    promote_after: int = 16,
) -> dict[str, Any]:
    """A3 coverage of one cell: is its ``(feed, e-bin)`` a SUPPORTED bin?

    The support set ``S`` is the spec's own definition (§4d A3): bins carrying at
    least ``promote_after`` labels, i.e. the bins ``TrustRegion`` would have
    promoted.  ``in_distribution`` here therefore MEANS something, unlike the
    ``feed == 121`` constant it replaces.
    """

    out: dict[str, Any] = {
        "feed": feed, "e_core": e_core, "e_core_band": e_core_band,
        "promote_after": promote_after, "n_support_bins": None,
        "in_distribution": None, "bin_labels": None,
    }
    if store_dir is None:
        return out
    try:
        from ..search.coverage import support_bins, e_bin

        bins, counts = support_bins(store_dir, e_core_band=e_core_band,
                                    promote_after=promote_after)
    except Exception:  # noqa: BLE001 — coverage is a diagnostic, never a gate
        return out
    out["n_support_bins"] = len(bins)
    if feed is None:
        return out
    key = (int(feed), e_bin(e_core, e_core_band))
    out["bin"] = [key[0], key[1]]
    out["bin_labels"] = int(counts.get(key, 0))
    out["in_distribution"] = bool(key in bins)
    return out


def snapshot_pre(
    run_dir: str | Path,
    store_dir: str | Path | None,
    *,
    pair: str | None,
    feed: int | None,
    e_core: float | None,
    library_id: str | None = None,
    predict: Callable[[Sequence[Any]], Any] | None = None,
    patterns_of: Callable[[Any], Any] | None = None,
    model_dir: str | None = None,
    max_rows: int = 256,
    e_core_band: float = 0.05,
    promote_after: int = 16,
) -> dict[str, Any]:
    """A2 "before": the surrogate's error on the cell's EXISTING labelled rows.

    ``predict`` is the campaign's own serving path (patterns -> object with a
    ``mean`` ``[N, 7]`` in ``TARGET_NAMES`` order).  It is optional: without it
    the snapshot still records the cell fingerprint and the A3 coverage flags,
    which is what a resumed/dry run can honestly say.  Bounded to ``max_rows``
    rows so a launch hook can never become the campaign's cost centre.
    """

    run_dir = Path(run_dir)
    out: dict[str, Any] = {
        "kind": "pre",
        "run": run_dir.name,
        "cell": f"{pair}/f{feed}",
        "pair": pair, "feed": feed, "e_core": e_core, "library_id": library_id,
        "model_dir": model_dir,
        "n": 0,
        "summary": None,
        "coverage": coverage_flags(store_dir, feed=feed, e_core=e_core,
                                   e_core_band=e_core_band,
                                   promote_after=promote_after),
    }
    if store_dir is None or predict is None or patterns_of is None:
        out["note"] = "no store and/or no serving path; error block not computed"
        return out
    try:
        from ..data.store import StoreReader

        records = StoreReader(store_dir).records
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"store unreadable: {type(exc).__name__}"
        return out
    rows = cell_rows(records, pair=pair, feed=feed, library_id=library_id,
                     exclude_campaign=run_dir.name)
    if not len(rows):
        out["note"] = "no prior labelled rows in this cell"
        return out
    rows = rows.tail(int(max_rows))
    patterns, actual = [], []
    for rec in rows.to_dict("records"):
        try:
            patterns.append(patterns_of(rec))
        except Exception:  # noqa: BLE001 — one unparseable row never sinks the hook
            continue
        actual.append(rec)
    if not patterns:
        out["note"] = "no decodable patterns among the prior rows"
        return out
    try:
        pred = predict(patterns)
        mean = [list(map(float, r)) for r in getattr(pred, "mean", pred)]
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"prediction failed: {type(exc).__name__}: {exc}"
        return out
    out["n"] = len(actual)
    out["summary"] = _mae_block(mean, actual)
    return out


def snapshot_post(
    run_dir: str | Path,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """A2 "after": the LAUNCH model's error on the labels this campaign bought.

    Uses the ``pred_mean`` recorded per selected candidate in
    ``waves/*/selection.json`` against the verified record in ``labels.jsonl``,
    so no checkpoint is loaded and the number is exactly what ranked the wave.
    Relative to the pre snapshot (rows the model was trained on) these are a
    genuine holdout, which is what makes ``ΔMAE / n_new`` measurable at all.
    """

    run_dir = Path(run_dir)
    labels = read_labels(run_dir)
    selections = selection_index(run_dir)
    per_wave_pos: dict[int, int] = {}
    mean: list[list[float | None]] = []
    actual: list[Mapping[str, Any]] = []
    for row in labels:
        try:
            wave_i = int(row.get("wave"))
        except (TypeError, ValueError):
            continue
        pos = per_wave_pos.get(wave_i, 0)
        per_wave_pos[wave_i] = pos + 1
        entries = selections.get(wave_i) or []
        if pos >= len(entries):
            continue
        pm = entries[pos].get("pred_mean")
        rec = row.get("record") or {}
        if not isinstance(pm, list) or not rec.get("converged"):
            continue
        # ``pred_mean`` carries ``None`` for a target the head does not serve
        # (max_assembly_burnup / max_pin_burnup on most decks); it must survive
        # to :func:`_mae_block`, which drops it per column, not crash here.
        mean.append([_num(x) for x in pm])
        actual.append(rec)
        if max_rows is not None and len(actual) >= int(max_rows):
            break

    pre = _read_json(run_dir / PRE_NAME) or {}
    out: dict[str, Any] = {
        "kind": "post",
        "run": run_dir.name,
        "cell": pre.get("cell"),
        "pair": pre.get("pair"), "feed": pre.get("feed"),
        "model_dir": pre.get("model_dir"),
        "n": len(actual),
        "n_new_labels": len(actual),
        "summary": _mae_block(mean, actual) if actual else None,
        "coverage": None,
        "error_reduction_per_label": None,
        "source": "waves/*/selection.json pred_mean vs labels.jsonl record",
    }
    store_dir = pre.get("store_dir")
    out["coverage"] = coverage_flags(
        store_dir, feed=pre.get("feed"), e_core=pre.get("e_core"),
        e_core_band=float((pre.get("coverage") or {}).get("e_core_band") or 0.05),
        promote_after=int((pre.get("coverage") or {}).get("promote_after") or 16),
    )
    pre_summary = pre.get("summary")
    if isinstance(pre_summary, dict) and out["summary"] and out["n"]:
        per_label: dict[str, float] = {}
        delta: dict[str, float] = {}
        for key, before in pre_summary.items():
            after = out["summary"].get(key)
            b, a = _num(before), _num(after)
            if b is None or a is None:
                continue
            delta[key] = round(b - a, 8)
            per_label[key] = round((b - a) / out["n"], 8)
        out["delta_mae"] = delta
        out["error_reduction_per_label"] = per_label
    return out


def _mae_block(mean: Sequence[Sequence[float | None]],
               actual: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-target MAE in the ``kpi_probe_out.json`` ``<target>_ALL`` layout."""

    out: dict[str, float] = {}
    for col, name in enumerate(PRED_COLUMNS):
        errs = []
        for pred_row, rec in zip(mean, actual):
            if col >= len(pred_row):
                continue
            truth = _num(rec.get(name))
            p = _num(pred_row[col])
            if truth is None or p is None:
                continue
            errs.append(abs(p - truth))
        if errs:
            out[f"{name}_ALL"] = sum(errs) / len(errs)
            out[f"{name}_n"] = float(len(errs))
    return out


def write_snapshot_pre(run_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / PRE_NAME
    _atomic_json(path, dict(payload))
    return path


def write_snapshot_post(run_dir: str | Path, **kwargs: Any) -> Path:
    payload = snapshot_post(run_dir, **kwargs)
    path = Path(run_dir) / POST_NAME
    _atomic_json(path, payload)
    return path


# --------------------------------------------------------------------------- #
# rendering (report.md / status.json share this)
# --------------------------------------------------------------------------- #
def kpi_markdown(kpi: Mapping[str, Any]) -> list[str]:
    """The A1–A4 block of ``report.md`` (spec §5 P0-5)."""

    a1 = kpi.get("A1_calls_to_frontier") or {}
    k2 = kpi.get("K2_calls_to_incumbent") or {}
    k3 = kpi.get("K3_AUF") or {}
    k4 = kpi.get("K4") or {}
    a2 = kpi.get("A2_error_reduction") or {}
    a3 = kpi.get("A3_coverage") or {}
    a4 = kpi.get("A4_ood_frontier") or {}

    def _s(v: Any, dash: str = "—") -> str:
        return dash if v is None else str(v)

    lines = ["## Sample-efficiency KPI (A1–A4)", ""]
    lines.append("> Counts **MASTER verification calls only**; surrogate "
                 "evaluations are the separate `screen_ratio` line "
                 "(`sample_efficiency_kpi_20260903.md` §2). "
                 "Source: `kpi.json` (`lpopt kpi --run <dir>`).")
    lines.append("")
    lines.append("| KPI | value | note |")
    lines.append("|---|---|---|")
    a1_val = (_s(a1.get("calls")) if a1.get("reached")
              else f"> {_s(a1.get('budget'))}")
    a1_note = (f"R_cell {_s(a1.get('R_cell'))}, ε {_s(a1.get('epsilon'))}, "
               f"final gap {_s(a1.get('final_gap'))}"
               + ("" if a1.get("valid") else " — **post-hoc, INVALID as A1**"))
    lines.append(f"| A1 calls-to-frontier@ε | {a1_val} | {a1_note} |")
    k2_val = _s(k2.get("calls")) if k2.get("reached") else "not reached"
    lines.append(f"| K2 calls-to-incumbent | {k2_val} | "
                 f"incumbent {_s(k2.get('incumbent'))} |")
    auf_cells = ", ".join(f"{k} = {v}" for k, v in sorted(k3.items())) or "—"
    lines.append(f"| K3 AUF | {auf_cells} | lower is better |")
    lines.append(f"| A2 ΔMAE/label | {_s(a2.get('per_label'))} | "
                 f"pre/post snapshots {'present' if a2.get('pre') else 'absent'} |")
    a3_pre = (a3.get("pre") or {})
    lines.append(f"| A3 coverage | in_distribution "
                 f"{_s(a3_pre.get('in_distribution'))}, support bins "
                 f"{_s(a3_pre.get('n_support_bins'))} | "
                 f"bin labels {_s(a3_pre.get('bin_labels'))} |")
    lines.append(f"| A4 OOD frontier updates | "
                 f"{_s(a4.get('n_ood_frontier_updates'))} / "
                 f"{_s(a4.get('n_calls'))} calls | "
                 f"flagged {_s(a4.get('n_ood_flagged'))}, "
                 f"unknown {_s(a4.get('n_ood_unknown'))} |")
    lines.append("")
    lines.append(f"- K4: prior_rows **{_s(k4.get('prior_rows'))}**, "
                 f"screen_ratio **{_s(k4.get('screen_ratio'))} : 1**, "
                 f"first_feasible_call **{_s(k4.get('first_feasible_call'))}**, "
                 f"n_feasible **{_s(k4.get('n_feasible'))}/{_s(k4.get('master_calls'))}**, "
                 f"Δ/100calls **{_s(k4.get('delta_per_100calls'))}** "
                 f"({_s(k4.get('floor_verdict'))}), "
                 f"wall **{_s(k4.get('wall_hours'))} h**")
    lines.append("")
    return lines


def kpi_status_block(run_dir: str | Path) -> dict[str, Any] | None:
    """The compact ``kpi`` block ``status.json`` carries when ``kpi.json`` exists."""

    kpi = _read_json(Path(run_dir) / KPI_NAME)
    if not isinstance(kpi, dict):
        return None
    a1 = kpi.get("A1_calls_to_frontier") or {}
    k4 = kpi.get("K4") or {}
    a4 = kpi.get("A4_ood_frontier") or {}
    return {
        "A1_calls_to_frontier": a1.get("calls"),
        "A1_reached": a1.get("reached"),
        "A1_valid": a1.get("valid"),
        "A1_final_gap": a1.get("final_gap"),
        "A2_error_reduction_per_label": (kpi.get("A2_error_reduction") or {}).get("per_label"),
        "A3_in_distribution": ((kpi.get("A3_coverage") or {}).get("pre") or {}).get("in_distribution"),
        "A4_ood_frontier_updates": a4.get("n_ood_frontier_updates"),
        "K2_calls_to_incumbent": (kpi.get("K2_calls_to_incumbent") or {}).get("calls"),
        "K3_AUF": kpi.get("K3_AUF"),
        "screen_ratio": k4.get("screen_ratio"),
        "prior_rows": k4.get("prior_rows"),
        "delta_per_100calls": k4.get("delta_per_100calls"),
        "wall_hours": k4.get("wall_hours"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lpopt kpi",
        description="sample-efficiency KPI (A1-A4) of a campaign run (MASTER 0 calls)",
    )
    p.add_argument("--run", required=True, help="the runs/<ts> directory")
    p.add_argument("--store", default=None,
                   help="record store dir (default data/store next to the run's root)")
    p.add_argument("--metric", default="f_r", help="objective axis (default f_r)")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help=f"A1 tolerance (default {DEFAULT_EPSILON})")
    p.add_argument("--feasible-only", action="store_true",
                   help="draw the trajectory on the run's feasible flag "
                        "instead of on `converged`")
    p.add_argument("--post", action="store_true",
                   help="also recompute the A2 'after' snapshot "
                        "(ood_snapshot_post.json) from this run's new labels")
    p.add_argument("--freeze", action="store_true",
                   help="(re)write kpi_baseline.json from --store; refuses to "
                        "overwrite an existing frozen baseline")
    return p


def _default_store(run_dir: Path) -> Path | None:
    for base in (run_dir.parent.parent, run_dir.parent, Path.cwd()):
        cand = base / "data" / "store"
        if (cand / "records.parquet").exists():
            return cand
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"[ERROR] run dir not found: {run_dir}")
        return 1
    store_dir = Path(args.store) if args.store else _default_store(run_dir)

    if args.freeze:
        path = run_dir / BASELINE_NAME
        if path.exists():
            print(f"[NOTE] {path.name} already frozen; not overwritten "
                  "(§5 K1: the frozen record is the KPI's denominator)")
        elif store_dir is None:
            print("[ERROR] --freeze needs --store")
            return 1
        else:
            pair, feed, library_id = _cell_of_run(run_dir)
            _atomic_json(path, freeze_baseline(
                store_dir, pair=pair, feed=feed, library_id=library_id,
                metric=args.metric, campaign=run_dir.name))
            print(f"RESULT: OK — wrote {path}")

    if args.post:
        post = write_snapshot_post(run_dir)
        print(f"RESULT: OK — wrote {post}")

    path = write_kpi(run_dir, store_dir, metric=args.metric,
                     epsilon=args.epsilon, feasible_only=args.feasible_only)
    print(f"RESULT: OK — wrote {path}")
    for line in kpi_markdown(_read_json(path) or {}):
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
