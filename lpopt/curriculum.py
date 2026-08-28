"""Cell-sequential curriculum driver (plan section 12.2 / 12.3).

The curriculum walks a ``(e_core band x feed)`` cell grid **outward** from a
support anchor and, at *every* cell, runs a crash-safe per-cell state machine::

    ensure_types -> blind_probe -> produce_cell -> retrain -> validate_gate -> done

The point (2026-07-17 user directive: no bulk production) is that the transfer
methodology is *measured before* each increment of learning (``blind_probe``:
the current champion predicts a fresh cell, then live MASTER labels the same
patterns, and per-target prediction-vs-truth error / rank Spearman / calibration
are recorded) and *re-validated after* (``validate_gate``: new-cell holdout,
no-regression on previous cells, the transfer-error-vs-distance curve, and a
mini user_criteria spot campaign).  Only when a cell passes does the cursor
advance to the next ring cell.

Design decisions (documented in the driver docstring and the run report):

* **Library growth** — ga80-backed bands (<= ~5.5 w/o core avg) reuse the
  harness-native 80-type library + FEASIBLE_PACKAGE restarts, so no lattice
  generation is needed there.  Bands lacking >= ``min_band_types`` full-physics
  types trigger on-demand Phase-A design generation that **extends ONE growing
  ``paramA`` library** (a TotalBatcher rebuild over the union of old+new HGCs)
  keyed by a persisted :class:`DesignRegistry` so ``FA_<alias>`` COMP names stay
  stable across rebuilds; band seed restarts are regenerated (feed 121 first,
  cross-feed borrow) after any rebuild.

* **Pin-burnup labels** — production and the blind probe both run through a
  WaveVerifier whose evaluator factory sets ``enable_pin_burnup=True`` (the
  default produce verifier does not), so ``max_pin_burnup`` labels flow.

* **New-cell holdout** — the blind-probe chains (a fixed set the production never
  generates) are reused as the post-train eval set: this gives a clean pre/post
  comparison on identical patterns with no extra MASTER budget.

* **Gates** — hard gates are (a) new-cell holdout finite + mean within-case
  Spearman >= threshold and (b) no previous-cell within-case Spearman drop >
  ``gate_noreg_epsilon``.  The transfer curve is always updated; the mini
  user_criteria campaign is a feasible-or-progress advisory (reported, warns).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._proc import no_window_flags
from .config import (
    FR_GUARD_KNOB as _FR_GUARD_KNOB, LpoptConfig, StratumConfig,
    fr_guard_enforced, load_config)
from .search.genome import fresh_units_from_feed, random_genome
from .vendor.masterrl.domain import CaseKey

PHASES = ("ensure_types", "blind_probe", "produce_cell", "retrain", "validate_gate", "done")

#: (name, surrogate predict column, vendor FOM attribute) for the targets a live
#: WaveVerifier can score (discharge_burnup is not on the FOM, so it is probe-
#: predicted only; max_assembly_burnup column 5 is always NaN in this model).
PROBE_TARGETS: tuple[tuple[str, int, str], ...] = (
    ("f_r", 0, "f_r"),
    ("cbc_max", 1, "cbc_max"),
    ("f_q", 2, "f_q"),
    ("cyclen", 3, "cyclen"),
    ("ao_abs", 4, "ao_abs"),
    ("max_pin_burnup", 6, "max_pin_burnup"),
)


# --------------------------------------------------------------------------- #
# pure helpers (grid, ring order, cell ids) — unit-tested directly
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return f"{float(x):g}"


def band_label(lo: float, hi: float) -> str:
    """Canonical band label, e.g. ``band_label(5.25, 5.5) == '5.25-5.5'``."""
    return f"{_fmt(lo)}-{_fmt(hi)}"


def cell_id(band: Sequence[float], feed: int) -> str:
    """Canonical cell id, e.g. ``'5.25-5.5_f117'``."""
    return f"{band_label(band[0], band[1])}_f{int(feed)}"


def _band_index(bands: Sequence[Sequence[float]], anchor_band: Sequence[float]) -> int:
    """Index of the band matching ``anchor_band`` (exact, else the one containing
    its midpoint, else nearest by midpoint distance)."""
    for i, b in enumerate(bands):
        if abs(b[0] - anchor_band[0]) < 1e-9 and abs(b[1] - anchor_band[1]) < 1e-9:
            return i
    mid = 0.5 * (anchor_band[0] + anchor_band[1])
    for i, b in enumerate(bands):
        if b[0] - 1e-9 <= mid < b[1] + 1e-9:
            return i
    return min(
        range(len(bands)),
        key=lambda i: abs(0.5 * (bands[i][0] + bands[i][1]) - mid),
    )


def ring_order(
    bands: Sequence[Sequence[float]],
    feeds: Sequence[int],
    anchor_band: Sequence[float],
    anchor_feed: int,
) -> list[tuple[tuple[float, float], int, int]]:
    """Deterministic expanding-ring order over the ``(band x feed)`` grid.

    Returns a list of ``((lo, hi), feed, ring)`` where ``ring`` is the Chebyshev
    ring distance from the anchor.  Within a ring, cells that stay in the anchor
    band (feed moves) come before cells that change band (per plan 12.2 example
    ``(5.25-5.5,117) -> (5.25-5.5,109/125) -> (5.0-5.25,117) -> ...``).  Ties
    break toward the lower feed index then lower band index (fully deterministic).
    """
    ai = _band_index(bands, anchor_band)
    if anchor_feed in feeds:
        af = list(feeds).index(anchor_feed)
    else:
        af = min(range(len(feeds)), key=lambda i: abs(feeds[i] - anchor_feed))

    cells = []
    for bi, b in enumerate(bands):
        for fi, feed in enumerate(feeds):
            bd = abs(bi - ai)
            fd = abs(fi - af)
            ring = max(bd, fd)
            key = (ring, bd, fd, fi, bi)
            cells.append((key, (float(b[0]), float(b[1])), int(feed), ring))
    cells.sort(key=lambda c: c[0])
    return [(band, feed, ring) for _key, band, feed, ring in cells]


def _now() -> float:
    return time.time()


def _stable_hash(text: str) -> int:
    """Process-stable hash (builtin ``hash`` is salted per run)."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default),
                   encoding="utf-8")
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return str(obj)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _spearman(truth: Sequence[float], pred: Sequence[float]) -> float | None:
    """Rank Spearman via the vendor helper (no torch); guards constant/short."""
    t = [v for v in truth]
    p = [v for v in pred]
    if len(t) < 3 or len(set(t)) < 2 or len(set(p)) < 2:
        return None
    from .vendor.masterrl.surrogate import _spearman as vendor_spearman
    val = vendor_spearman(t, p)
    return _finite(val)


#: primary no-regression targets: ``(name, surrogate column, store truth column)``.
# Per-check null SD of the no-regression drop statistic at n=30 holdouts
# (data/reports/gate_noise_analysis.md) — drives the family-wise adaptive epsilon.
_NOREG_SIGMA0 = 0.042

#: ``col`` sentinel for a target the surrogate ``predict`` head does NOT carry:
#: ``node_peak`` / ``map_cov`` are scalars of the MAP head (``predict_map_flatness``),
#: not columns of :data:`..model.dataset_torch.TARGETS`.  Kept as a sentinel in the
#: existing ``(name, col, truth_col)`` shape so a caller passing its own ``targets``
#: tuple is unaffected.
MAP_HEAD_COL = -1

# Retrain-promotion no-regression targets.  Under the flatness-first directive
# (program §1.2/§10) the campaign OBJECTIVE is node_peak + map_cov and F_r was
# retired from it — but this gate is NOT the campaign objective, and the two must
# not be conflated:
#
#   * ``node_peak`` / ``map_cov`` are here because a model promoted to champion
#     STEERS a flat_power campaign.  Judging promotion on cyclen and F_r alone let
#     a candidate that had lost map-head skill be promoted and then drive the
#     search one level down — F_r selecting the model that selects the loading
#     patterns.  These are the axes the campaign actually optimises, so they are
#     guarded axes here.
#
#   * ``f_r`` is SCORED but REPORT-ONLY by default — see the demotion note under
#     :data:`NOREG_ENFORCED_DEFAULT` below.  It stays in this tuple because the
#     axis is dormant, not deleted: every check row is still computed and printed.
#
#   * ``cyclen`` stays for the same reason it always was here: the cycle-length
#     band is a hard constraint in several modes and a record-only column in
#     flat_power, and a collapse there is a regression whatever the objective.
#
# A guarded axis this slice or this model cannot SCORE is reported unavailable
# (``score_no_regression_cell`` emits a check with ``drop=None`` and a reason,
# which ``gate_no_regression`` surfaces) rather than dropped in silence — an
# unscored guard must never read as a passed guard.
NOREG_TARGETS: tuple[tuple[str, int, str], ...] = (
    ("cyclen", 3, "cyclen"),
    ("f_r", 0, "f_r"),
    ("node_peak", MAP_HEAD_COL, "node_peak"),
    ("map_cov", MAP_HEAD_COL, "map_cov"),
)

# --------------------------------------------------------------------------- #
# F_r: SCORED, and by default NOT ENFORCED (user decision 2026-07-26)
#
# The previous round made ``f_r`` a guarded axis on the argument that it guards
# the SAFETY path (flat_power gates every row on the D1 F_r limit through this
# model's F_r prediction).  That argument is about the axis's IMPORTANCE.  A
# promotion gate is not a statement about importance — it is a REGRESSION
# detector, and it may only veto a model on an axis where a regression is (i)
# measurable on the gate's own slice and (ii) recoverable by the candidate.  On
# F_r today, neither holds, and the veto therefore punishes a DATA limitation
# rather than a model regression:
#
#   (1) The gate's own slice contains ZERO decision-band labels.  Measured
#       2026-07-26 over all 36 done curriculum cells' val holdouts (1,592 rows):
#       *not one* row carries F_r < 1.55.  The lowest val F_r in ANY cell is
#       1.5974 (cell 5-5.25_f125); 33 of 36 cells bottom out above 1.68.  The
#       whole store holds 744 sub-1.55 converged rows (1.46% of 50,786) and they
#       are Dataset-B/P fill rows at feed 117/121/125 — 742 of 744 sit in the
#       TRAIN fold and none in any per-cell val holdout.  So the "safety guard"
#       was never scoring boundary skill: it scored bulk F_r rank two-tenths
#       above the licensing limit and called that a licensing guard.
#
#   (2) Where it WOULD matter, the labels cannot carry the measurement.  The
#       transpose-pair reproducibility experiment (22 pairs, 48 MASTER calls;
#       data/reports/transpose_noise_measured_20260725.md §2.2) puts the label
#       noise at sigma=0.00595 against an in-band signal SD of 0.00917, i.e. a
#       label ceiling of rho_max = 0.839 for F_r < 1.55.  A perfect physics model
#       scores 0.84 there.  Gating on an axis whose attainable rank skill is
#       capped near the noise floor rejects models for label resolution.
#
#   (3) Core F_r is set by the single hottest ASSEMBLY, so the axis acquires
#       learnable signal only once FA-optimized assemblies enter the corpus.
#       Until then the program's sequencing is: learn node-power FLATTENING rules
#       now -> apply them to FA-optimized assemblies -> THEN review this gate.
#
# Nothing here is deleted.  ``f_r`` is still scored on every cell, still emitted
# in ``checks``, and the gate REPORTS in ``note`` that it was scored and not
# enforced — so a pass can never be read as "F_r verified regression-free".
# Flipping ``[curriculum] gate_noreg_fr_guard_enabled`` to true promotes it back
# to a guarded axis with no code change.
# --------------------------------------------------------------------------- #

#: Axes whose measured drop can VETO promotion when the F_r guard is off (default).
NOREG_ENFORCED_DEFAULT: tuple[str, ...] = ("cyclen", "node_peak", "map_cov")

#: The axis held dormant, and the knob that wakes it.  The knob name is imported
#: from :mod:`.config` so the ONE switch has ONE spelling across every surface.
FR_GUARD_TARGET = "f_r"
FR_GUARD_KNOB = _FR_GUARD_KNOB

#: Upper edge of the F_r DECISION band — measured where the model's F_r
#: prediction actually adjudicates, which is the **D1 in-loop safety gate 1.70**
#: (:attr:`..search.acquisition.FlatPowerSpec.fr_limit`), not the D2 licensing
#: constant 1.55.
#:
#: Why 1.70 and not 1.55 (this was the open question; it is now decided).  The
#: criteria below ask "can a drop on this axis be MEASURED and RECOVERED where
#: the axis is used to decide?".  The prediction decides in exactly one place: a
#: ``flat_power`` candidate is VETOED when its F_r UCB exceeds ``fr_gate``, whose
#: unmodified value is 1.70 and which a fitted bias correction may only TIGHTEN
#: (:func:`..search.acquisition.flatpower_fr_gate` clamps at ``fr_limit``).  1.55
#: is where a DELIVERED pattern is judged compliant — the D2 licensing limit and
#: a report column — and no in-loop decision is taken there.  Measuring the
#: guard's own preconditions at 1.55 therefore measured the wrong surface: it
#: scored label supply for a decision the model never makes, and reported "0 of
#: 36 cells" for a band that is not the band the veto acts on.
#:
#: Measured 2026-07-26 on the real gate slice (``data/splits/S1.json``
#: ``curriculum_val_by_cell`` x ``records.parquet``, converged rows only): moving
#: the band to the decision surface does NOT rescue the criterion — 11 of 1,494
#: rows sit below 1.70, spread over 6 of 36 cells, and the best-supplied cell has
#: THREE such rows against a bar of 30.  The band is now honest AND the answer is
#: unchanged, which is the strongest form the deferral could take.
FR_GUARD_BAND_HI = 1.70

#: The D2 LICENSING limit.  Not a decision surface for the model's prediction —
#: reported alongside the decision band so the older measurement stays visible
#: and so a reader can see the two are different questions.
FR_GUARD_LICENSING_LIMIT = 1.55

#: (b) Minimum decision-band labels PER GATED CELL before the F_r guard means
#: anything.  N = 30 is not a round number: sigma0 = 0.042, the per-check null SD
#: the whole epsilon calibration rests on, was measured at n=30 per-cell holdout
#: rows (data/reports/gate_noise_analysis.md §Task 4a), and ``gate_noreg_epsilon``
#: = 0.10 is that sigma0 turned into a 5% family-wise bar.  Score the F_r axis on
#: fewer than 30 band rows and the drop statistic's null SD exceeds the value
#: epsilon was derived from, so the configured epsilon silently stops being a 5%
#: bar and the guard false-rejects healthy candidates.  30 is therefore the point
#: at which the EXISTING calibration transfers to the band, not a new one.
FR_GUARD_MIN_BAND_LABELS_PER_CELL = 30

#: (c) Minimum label ceiling rho_max = s / sqrt(s^2 + sigma^2) in the decision
#: band.  Today: 0.839 (s=0.00917, sigma=0.00595).  0.95 is the bar because at
#: rho_max = 0.95 the epsilon=0.10 veto is a fifth of the attainable rank range
#: (1 - 0.95 = 0.05 is the part no model can win), whereas at 0.839 the veto
#: threshold is smaller than the label-noise-imposed deficit itself — the gate
#: would be adjudicating differences it cannot resolve.  Two routes reach 0.95,
#: and the FA-optimized corpus is expected to supply the first: (i) more in-band
#: DESIGN diversity raises the signal SD s — s must reach 0.0181 at today's
#: sigma; (ii) k-fold repeat-and-average of the MASTER label shrinks sigma by
#: 1/sqrt(k), needing k >= 4 at today's s.  Re-measure with the transpose-pair
#: protocol; do not assume.
FR_GUARD_MIN_LABEL_CEILING = 0.95

#: Measured 2026-07-26 (transpose_noise_measured_20260725.md §2.2) — the value
#: criterion (c) is currently compared against.  Update when re-measured.
FR_GUARD_MEASURED_LABEL_CEILING = 0.839

#: ACTIVATION CRITERIA for ``[curriculum] gate_noreg_fr_guard_enabled = true``.
#: All three must hold; they are recorded here so nobody has to re-derive them.
FR_GUARD_ACTIVATION_CRITERIA: tuple[str, ...] = (
    "(a) FA-optimized assemblies are present in the training corpus — core F_r is "
    "set by the hottest ASSEMBLY, so until FA optimization lands the F_r axis has "
    "no learnable signal for a candidate to win or lose.",
    f"(b) each gated cell's val holdout carries >= {FR_GUARD_MIN_BAND_LABELS_PER_CELL} "
    f"labels in the DECISION band f_r < {FR_GUARD_BAND_HI} — the D1 in-loop safety "
    "gate, i.e. where the model's F_r prediction actually adjudicates (measured "
    "2026-07-26: 11 of 1,494 val rows, over 6 of 36 cells, best cell 3; at the D2 "
    f"licensing constant {FR_GUARD_LICENSING_LIMIT} it is 0 of 1,494) — below this "
    "the drop statistic is noisier than the sigma0=0.042 the epsilon was "
    "calibrated at.",
    f"(c) the re-measured label ceiling in the decision band f_r < {FR_GUARD_BAND_HI} "
    f"is >= {FR_GUARD_MIN_LABEL_CEILING} (measured 2026-07-25 at the D2 constant "
    f"{FR_GUARD_LICENSING_LIMIT}: {FR_GUARD_MEASURED_LABEL_CEILING}; the D1-band "
    "value is not yet measured and the transpose-pair protocol must supply it "
    "before the switch is flipped) — a gate on an axis capped at 0.84 by label "
    "noise rejects models for label resolution, not for regression.",
)


def enforced_noreg_targets(*, fr_guarded: bool = False) -> tuple[str, ...]:
    """Names of the axes whose drop may VETO promotion.

    ``fr_guarded`` is the one-setting activation switch
    (``[curriculum] gate_noreg_fr_guard_enabled``): false (default) keeps ``f_r``
    scored-but-report-only, true promotes it back to a guarded axis.  See
    :data:`FR_GUARD_ACTIVATION_CRITERIA` for when flipping it is justified.
    """
    if fr_guarded:
        return tuple(t[0] for t in NOREG_TARGETS)
    return tuple(t[0] for t in NOREG_TARGETS if t[0] in NOREG_ENFORCED_DEFAULT)


def fr_guard_block(*, fr_guarded: bool,
                   band_labels: dict[str, int] | None = None,
                   licensing_band_labels: dict[str, int] | None = None,
                   ) -> dict[str, Any]:
    """The ``fr_guard`` report block, identical on every promotion surface.

    One shape means an operator reads the same five fields whether they opened a
    curriculum ``gate.json``, a new-cell block or an offline A/B judgement, and
    :data:`FR_GUARD_ACTIVATION_CRITERIA` travels with each of them.  ``band_labels``
    (criterion (b), counted in the D1 decision band) and ``licensing_band_labels``
    (the D2 reference) are supplied only by surfaces that hold a labelled slice;
    the rest report empty maps rather than fabricate a measurement.
    """
    band = dict(band_labels or {})
    return {
        "target": FR_GUARD_TARGET,
        "enforced": bool(fr_guarded),
        "knob": FR_GUARD_KNOB,
        "band_hi": FR_GUARD_BAND_HI,
        "licensing_limit": FR_GUARD_LICENSING_LIMIT,
        "min_band_labels_per_cell": FR_GUARD_MIN_BAND_LABELS_PER_CELL,
        "band_labels_by_cell": band,
        "licensing_band_labels_by_cell": dict(licensing_band_labels or {}),
        "cells_meeting_label_criterion": sum(
            1 for v in band.values() if v >= FR_GUARD_MIN_BAND_LABELS_PER_CELL),
        "cells_scored": len(band),
        "measured_label_ceiling": FR_GUARD_MEASURED_LABEL_CEILING,
        "min_label_ceiling": FR_GUARD_MIN_LABEL_CEILING,
        "activation_criteria": list(FR_GUARD_ACTIVATION_CRITERIA),
    }


def fr_report_only_note(axes: Sequence[str], *, measured: str = "",
                        exceeded: Sequence[str] = (),
                        verb: str = "block promotion") -> str:
    """The one sentence every in-loop report-only surface says, in the same words.

    A pass on a surface that scored ``f_r`` without teeth must never be readable
    as "F_r verified regression-free", so this text — not a per-callsite
    paraphrase — is what the no-regression gate and the new-cell skill gate both
    emit.  The offline A/B (:mod:`.model.ab_eval`) says the same thing in its own
    module: it must stay importable without pulling in pandas and the search
    stack, so it carries the wording rather than this function.
    """
    axes = list(axes)
    msg = ("REPORT-ONLY axes (scored, NOT enforced): " + ", ".join(axes)
           + f" — their drop cannot {verb}"
           + (f" ({measured})" if measured else "")
           + ". A pass does NOT mean " + "/".join(axes)
           + " was verified regression-free. "
           + f"Set {FR_GUARD_KNOB} = true to enforce; see "
             "FR_GUARD_ACTIVATION_CRITERIA for when that is justified.")
    if exceeded:
        msg += (" THIS RUN: " + ", ".join(exceeded)
                + " exceeded epsilon and was ADMITTED anyway.")
    return msg


def _map_head_flatness_ucb(model: Any, patterns: Sequence[Any],
                           cases: Sequence[Any]
                           ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray] | None, str | None]:
    """``((peak_mean, peak_std, cov_mean, cov_std), None)`` or ``(None, reason)``.

    The SPREAD-carrying sibling of :func:`_map_head_flatness`: the no-regression
    gate ranks and therefore needs only the means, while an ACQUISITION scalar is
    a UCB and needs the per-candidate ensemble spread as well.  Same never-raises
    contract, same fallback order, so both surfaces agree on what "this model has
    no map head" means.  A ``predict_map_peak``-only backend yields NaN CoV, which
    :func:`..search.acquisition.score_flat_power` reads as "drop the secondary
    term" rather than as a zero.
    """
    n = len(list(patterns))
    fn = getattr(model, "predict_map_flatness", None)
    if callable(fn):
        try:
            pk_m, pk_s, cv_m, cv_s = fn(list(patterns), list(cases))
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError,
                ValueError) as exc:
            return None, f"map head failed ({type(exc).__name__})"
        return (np.asarray(pk_m, dtype=float), np.asarray(pk_s, dtype=float),
                np.asarray(cv_m, dtype=float), np.asarray(cv_s, dtype=float)), None
    fn = getattr(model, "predict_map_peak", None)
    if callable(fn):
        try:
            pk_m, pk_s = fn(list(patterns), list(cases))
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError,
                ValueError) as exc:
            return None, f"map head failed ({type(exc).__name__})"
        nan = np.full(n, np.nan)
        return (np.asarray(pk_m, dtype=float), np.asarray(pk_s, dtype=float),
                nan, nan.copy()), None
    return None, "model exposes no map head (predict_map_flatness/predict_map_peak)"


def _map_head_flatness(model: Any, patterns: Sequence[Any], cases: Sequence[Any]
                       ) -> tuple[dict[str, Any] | None, str | None]:
    """``({"node_peak": [...], "map_cov": [...]}, None)`` or ``(None, reason)``.

    Mirrors :func:`..search.update._flatness_predictions` so the promotion gate
    scores the SAME quantity the campaign's wave gate and acquisition rank by.
    Never raises: a gate that dies on a backend without a map head is a gate that
    cannot be run at all, so the inability is returned as a REASON and reported.

    The ranking gate needs only the means; :func:`_map_head_flatness_ucb` is the
    single implementation of "get the map head out of this backend", so the two
    surfaces can never disagree about what a missing map head is.
    """
    out, reason = _map_head_flatness_ucb(model, patterns, cases)
    if out is None:
        return None, reason
    pk_m, _pk_s, cv_m, _cv_s = out
    return {"node_peak": pk_m, "map_cov": cv_m}, None


def score_no_regression_cell(
    old_model: Any,
    new_model: Any,
    sub_df: Any,
    *,
    targets: Sequence[tuple[str, int, str]] = NOREG_TARGETS,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Score the previous champion AND the candidate LIVE on the *same* held-out
    store rows (``sub_df``), predicting each row and comparing to its STORED truth.

    Both models are handed an identical ``(patterns, cases)`` list built once from
    ``sub_df`` — so the per-target Spearman comparison is honest out-of-sample vs
    out-of-sample (no in-sample-vs-out-of-sample phantom drop).  Returns
    ``(scored_record_ids, per_target_rows)`` where each row carries
    ``old_spearman`` / ``new_spearman`` / ``drop = old - new`` / ``n``.

    A MAP-HEAD target (``col == MAP_HEAD_COL``: ``node_peak`` / ``map_cov``) is
    predicted through :func:`_map_head_flatness` instead of the surrogate columns.
    When it cannot be scored — the model has no map head, the slice has no
    ``node_peak`` column, or fewer than three rows carry the label — a check row
    is still emitted, with ``drop=None`` and an ``unavailable`` reason, so an
    unscored guard is REPORTED rather than silently absent from the family.
    """
    from .data.schema import unpack_pattern
    rids = [str(r) for r in sub_df["record_id"].tolist()]
    pats = [unpack_pattern(str(p)) for p in sub_df["pattern"].tolist()]
    cases = [CaseKey(str(pr), int(fd))
             for pr, fd in zip(sub_df["case_pair"], sub_df["feed"])]
    old_pred = old_model.predict(pats, cases)
    new_pred = new_model.predict(pats, cases)

    # Map-head predictions are computed ONCE per model (a full forward), and only
    # when a map-head target is actually requested.
    want_map = any(col == MAP_HEAD_COL for _n, col, _t in targets)
    old_flat = new_flat = None
    map_reason: str | None = None
    if want_map:
        old_flat, r_old = _map_head_flatness(old_model, pats, cases)
        new_flat, r_new = _map_head_flatness(new_model, pats, cases)
        map_reason = r_old or r_new

    rows: list[dict[str, Any]] = []
    for name, col, truth_col in targets:
        map_head = (col == MAP_HEAD_COL)

        def _unavailable(reason: str, n: int = 0) -> None:
            rows.append({"target": name, "old_spearman": None,
                         "new_spearman": None, "drop": None, "n": int(n),
                         "unavailable": reason})

        # Emitted for EVERY axis, not just the map-head ones.  Guarding these two
        # branches with ``if map_head`` meant an unscoreable SCALAR axis produced no
        # row at all: absent from ``checks``, from ``unavailable``, from ``note`` and
        # from the console — the one genuinely silent path in the gate.  It is
        # unreached today only because cyclen carries a label on 100% of rows; the
        # moment a slice arrives without one, the axis would vanish rather than
        # report itself.  (2026-07-26 gate-blindness audit, defect D1.)
        if truth_col not in getattr(sub_df, "columns", ()):
            _unavailable(f"slice carries no {truth_col!r} column")
            continue
        truth = [_finite(v) for v in sub_df[truth_col].tolist()]
        idx = [i for i, t in enumerate(truth) if t is not None]
        if len(idx) < 3:
            _unavailable(f"fewer than 3 rows carry a {truth_col} label",
                         len(idx))
            continue
        tv = [truth[i] for i in idx]
        if map_head:
            if old_flat is None or new_flat is None:
                _unavailable(map_reason or "map head unavailable", len(idx))
                continue
            o_all, n_all = old_flat.get(name), new_flat.get(name)
            if o_all is None or n_all is None:
                _unavailable(f"map head predicts no {name}", len(idx))
                continue
            o_vals = [float(o_all[i]) for i in idx]
            n_vals = [float(n_all[i]) for i in idx]
            keep = [k for k, i in enumerate(idx)
                    if math.isfinite(o_vals[k]) and math.isfinite(n_vals[k])]
            if len(keep) < 3:
                _unavailable(f"fewer than 3 finite {name} predictions", len(keep))
                continue
            tv = [tv[k] for k in keep]
            o_vals = [o_vals[k] for k in keep]
            n_vals = [n_vals[k] for k in keep]
            n_used = len(keep)
        else:
            o_vals = [float(old_pred.mean[i, col]) for i in idx]
            n_vals = [float(new_pred.mean[i, col]) for i in idx]
            n_used = len(idx)
        osp = _spearman(tv, o_vals)
        nsp = _spearman(tv, n_vals)
        drop = (osp - nsp) if (osp is not None and nsp is not None) else None
        row = {"target": name, "old_spearman": osp, "new_spearman": nsp,
               "drop": drop, "n": n_used}
        if map_head and drop is None:
            row["unavailable"] = f"{name} Spearman undefined (constant ranks)"
        rows.append(row)
    return rids, rows


def certify_gate_coverage(
    checks: Sequence[dict[str, Any]],
    enforced: Sequence[str],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """How much of the ENFORCING family the gate actually managed to measure.

    ``pass`` is a reduction over the checks that produced a ``drop``; an enforced
    axis that produced none contributes nothing and therefore cannot fail.  That
    is correct arithmetic and a misleading report: ``guarded_targets`` names the
    axes the run INTENDED to enforce, so a reader sees three guards where one was
    measured.  This returns the measured truth — per target, the number of cells
    that yielded a scoreable check and the total rows behind them — plus the
    enforced axes that were never measured anywhere.

    Deliberately NOT wired into ``pass`` (user decision 2026-07-26: warn, do not
    block).  A coverage bar that vetoes would deadlock the curriculum on its
    second cell, where ``done_cells`` holds one entry and no axis can clear a
    two-cell requirement — and no code change can conjure a label.  The remedy is
    to produce map labels in curriculum cells; this function makes the deficit
    impossible to miss while that happens, and reports real judgment the moment
    the labels arrive, with no further change.
    """
    measured: dict[str, dict[str, int]] = {}
    for c in checks:
        if c.get("drop") is None or not c.get("enforced"):
            continue
        m = measured.setdefault(str(c["target"]), {"cells": 0, "rows": 0})
        m["cells"] += 1                      # one check == one cell
        m["rows"] += int(c.get("n") or 0)
    blind = sorted(str(t) for t in enforced if str(t) not in measured)
    return measured, blind


# --------------------------------------------------------------------------- #
# legacy-corpus high-cyclen tail no-regression guard (forensic 20260719)
# --------------------------------------------------------------------------- #
def sample_legacy_tail_rows(
    records_df: Any,
    *,
    bands: Sequence[Sequence[float]],
    feed: int = 121,
    sample_per_band: int = 150,
    seed: int = 0,
) -> dict[tuple[float, float], Any]:
    """Deterministic per-band sample of converged Dataset-A rows for the tail guard.

    Selection is a pure function of ``(seed, record_id)`` via :func:`_stable_hash`
    (SHA-1, process-stable) so the SAME rows are scored for every champion and
    across runs — the "fixed random sample" the guard needs to compare two models
    on identical held-out truth.  Growth-invariant: adding rows never changes which
    of the pre-existing rows are selected below the sample cap for a band whose
    membership they do not enter.
    """
    df = records_df
    conv = df["converged"].astype(bool) if "converged" in df.columns else True
    a = df[(df["dataset"].astype(str) == "A") & conv
           & (pd.to_numeric(df["feed"], errors="coerce") == int(feed))]
    cy = pd.to_numeric(a["cyclen"], errors="coerce")
    out: dict[tuple[float, float], Any] = {}
    for band in bands:
        lo, hi = float(band[0]), float(band[1])
        sub = a[(cy >= lo) & (cy < hi)]
        if len(sub):
            order = sub["record_id"].astype(str).map(
                lambda r: _stable_hash(f"tail:{seed}:{r}"))
            sub = sub.assign(_tail_h=order.to_numpy()).sort_values(
                "_tail_h").head(int(sample_per_band))
        out[(lo, hi)] = sub
    return out


def score_legacy_tail_no_regression(
    old_model: Any,
    new_model: Any,
    records_df: Any,
    *,
    bands: Sequence[Sequence[float]],
    feed: int = 121,
    sample_per_band: int = 150,
    seed: int = 0,
    epsilon: float = 2.0,
) -> dict[str, Any]:
    """HONEST tail no-regression: score BOTH champions' cyclen on the SAME fixed
    stable-hash sample of S1-val Dataset-A rows per cyclen band and fail when the
    candidate's per-band MAE degrades by more than ``epsilon`` EFPD.

    Both models are scored through :meth:`PosValCnnBackend.predict_rows_raw`, which
    featurizes each row from its OWN ``library_id`` provenance (train/serve parity),
    so the metric is the model's true tail skill — not the provenance-less serve
    path whose cross-library type-resolution break is what manifested the collapse.
    Returns a gate dict with per-band ``old_mae`` / ``new_mae`` / ``mae_increase``.
    """
    samples = sample_legacy_tail_rows(
        records_df, bands=bands, feed=feed, sample_per_band=sample_per_band, seed=seed)
    per_band: list[dict[str, Any]] = []
    worst = 0.0
    ok = True
    for (lo, hi), rows in samples.items():
        entry: dict[str, Any] = {"band": [lo, hi], "n": int(len(rows))}
        if len(rows) < 3:
            entry["note"] = "insufficient rows"
            per_band.append(entry)
            continue
        truth = pd.to_numeric(rows["cyclen"], errors="coerce").to_numpy(dtype=float)
        old_cy = np.asarray(old_model.predict_rows_raw(rows))[:, 3]
        new_cy = np.asarray(new_model.predict_rows_raw(rows))[:, 3]
        m_old = float(np.mean(np.abs(old_cy - truth)))
        m_new = float(np.mean(np.abs(new_cy - truth)))
        inc = m_new - m_old
        worst = max(worst, inc)
        if inc > epsilon:
            ok = False
        entry.update({"old_mae": m_old, "new_mae": m_new, "mae_increase": inc})
        per_band.append(entry)
    return {"pass": ok, "epsilon": float(epsilon),
            "worst_mae_increase": worst, "bands": per_band, "feed": int(feed)}


# --------------------------------------------------------------------------- #
# module-level promotion gates (refactored out of CurriculumDriver so the
# ``lpopt gate-promote`` CLI can run BOTH gates from a bare repo checkout — no
# curriculum "done" state required; the CurriculumDriver methods delegate here).
# --------------------------------------------------------------------------- #
def gate_no_regression(
    old_model: Any,
    new_model: Any,
    records_df: Any,
    val_by_cell: dict[str, Sequence[str]],
    done_cells: Sequence[str],
    *,
    epsilon: float,
    sigma0: float = _NOREG_SIGMA0,
    fr_guarded: bool = False,
) -> dict[str, Any]:
    """HONEST no-regression gate (reusable core of ``_gate_no_regression``).

    For every cell in ``done_cells`` score BOTH ``old_model`` and ``new_model`` LIVE
    on the SAME per-cell curriculum-val holdout (``val_by_cell[cell]`` record ids,
    all held out of both trainings) and require the candidate's within-case Spearman
    not to drop by more than the family-wise-adaptive epsilon on the primaries.

    The SCORED family is :data:`NOREG_TARGETS` — ``cyclen``, ``f_r`` and the
    flatness axes ``node_peak`` / ``map_cov`` the promoted champion will steer a
    ``flat_power`` campaign on.  The ENFORCING family is narrower: by default
    ``f_r`` is scored and REPORT-ONLY and cannot veto promotion (see the demotion
    note at :data:`NOREG_ENFORCED_DEFAULT`; ``fr_guarded=True``, wired to
    ``[curriculum] gate_noreg_fr_guard_enabled``, restores it).  ``guarded_targets``
    therefore lists only what can actually fail this run, ``report_only_targets``
    lists what was scored without teeth, and ``note`` says so in words — so a pass
    is never readable as "F_r verified regression-free".

    Axes that could not be scored come back in ``unavailable`` with a reason and
    are named in ``note``; ``pass`` reflects only the enforcing axes actually
    measured, so an unjudged guard is visible instead of being read as a pass.

    The configured ``epsilon`` was calibrated as a 5% family-wise bar for the
    original max-of-6 gate (sigma0=0.042); the check count grows ~2 per done cell,
    so the same calibration requires ``eps_N = sigma0·Phi^-1(0.95^(1/N))`` once
    N>6 — the configured value stays a floor.  Takes explicit inputs (no driver
    state), so ``gate-promote`` derives ``done_cells`` / ``val_by_cell`` directly
    from a store-built curriculum split.
    """
    enforced = enforced_noreg_targets(fr_guarded=bool(fr_guarded))
    report_only = [t[0] for t in NOREG_TARGETS if t[0] not in enforced]
    if not done_cells or old_model is None or new_model is None:
        # The trivial path still carries the WHOLE contract.  A consumer reading
        # ``guarded_targets`` / ``fr_guard`` off a gate.json used to hit a
        # KeyError on the very first cell, and — worse — that run left no record
        # that the F_r axis had been deferred at all.  Nothing was scored, so the
        # band counts are empty maps rather than an invented zero-measurement.
        return {"pass": True, "epsilon": float(epsilon), "worst_drop": 0.0,
                "worst_drop_any_axis": 0.0,
                "checks": [], "scored_record_ids": {},
                "guarded_targets": list(enforced),
                # Nothing was measured, but nothing was OWED either — there is no
                # previous cell to regress against.  ``blind_targets`` means "should
                # have been measured and was not", so it stays empty here; claiming
                # three blind axes on the first cell would cry wolf on every run.
                "guarded_measured": {},
                "blind_targets": [],
                "cells_scored": 0,
                "report_only_targets": report_only,
                "scored_targets": [t[0] for t in NOREG_TARGETS],
                "unavailable": [],
                "fr_guard": fr_guard_block(fr_guarded=bool(fr_guarded)),
                "note": "no previous cells"}
    indexed = records_df.drop_duplicates("record_id").set_index("record_id")
    checks: list[dict[str, Any]] = []
    scored: dict[str, list[str]] = {}
    band_labels: dict[str, int] = {}
    lic_labels: dict[str, int] = {}
    worst_drop = 0.0
    worst_any = 0.0
    cells_entered = 0
    for pc in done_cells:
        val_ids = [rid for rid in val_by_cell.get(pc, []) if rid in indexed.index]
        if len(val_ids) < 3:
            continue
        sub = indexed.loc[val_ids].reset_index()
        sub = sub[sub["converged"] == True]  # noqa: E712 — truth must be present
        if len(sub) < 3:
            continue
        cells_entered += 1
        # Criterion (b) of FR_GUARD_ACTIVATION_CRITERIA, measured on the gate's OWN
        # slice rather than asserted: how many of this cell's holdout rows sit in
        # the F_r DECISION band (the D1 in-loop safety gate, where the model's F_r
        # prediction actually adjudicates).  The D2 licensing count is carried
        # alongside so the older 1.55 measurement stays visible and the two
        # questions cannot be confused.  Free — ``sub`` is already in hand.
        if "f_r" in getattr(sub, "columns", ()):
            _fr = pd.to_numeric(sub["f_r"], errors="coerce")
            band_labels[pc] = int((_fr < FR_GUARD_BAND_HI).sum())
            lic_labels[pc] = int((_fr < FR_GUARD_LICENSING_LIMIT).sum())
        rids, rows = score_no_regression_cell(old_model, new_model, sub)
        scored[pc] = rids
        for r in rows:
            drop = r.get("drop")
            enf = r["target"] in enforced
            if drop is not None:
                worst_any = max(worst_any, drop)
                if enf:
                    # ``worst_drop`` stays the worst ENFORCED drop so the invariant
                    # ``pass == (worst_drop <= epsilon)`` survives the demotion; a
                    # report-only axis cannot silently look like the cause of a
                    # fail (or, worse, make a pass look self-contradictory).
                    worst_drop = max(worst_drop, drop)
            checks.append({"cell": pc, "enforced": enf, **r})
    # Only ENFORCING checks can veto, so only they carry family-wise multiplicity:
    # counting report-only checks in N would inflate eps for axes that cannot fail.
    n_checks = sum(1 for c in checks
                   if c.get("drop") is not None and c.get("enforced"))
    eps = float(epsilon)
    if n_checks > 6:
        from scipy.stats import norm as _norm
        eps = max(eps, float(sigma0 * _norm.ppf(0.95 ** (1.0 / n_checks))))
    ok = all(c["drop"] <= eps for c in checks
             if c.get("drop") is not None and c.get("enforced"))
    # What the enforcing family ACTUALLY measured, as opposed to what it declares.
    # ``pass`` is deliberately left alone (see certify_gate_coverage): this is the
    # reading instrument, not a new veto.
    guarded_measured, blind_targets = certify_gate_coverage(checks, enforced)
    # Guarded axes this run could NOT judge.  Reported, never dropped: a gate that
    # returns ``pass`` while its flatness guards never ran is a gate that says
    # "no regression" about axes it did not look at.  ``pass`` still reflects only
    # what was measured — the caller decides what an unjudged guard is worth — but
    # it can no longer be mistaken for a full family having been checked.
    unavailable = [c for c in checks if c.get("unavailable")]
    # The SAME honesty rule applied to a demoted axis: a report-only axis was
    # scored but has no teeth, so it is listed separately from ``guarded_targets``
    # and spelled out in ``note``.  An axis missing from ``guarded_targets`` is the
    # machine-readable form of "this run did not enforce it".
    ro_scored = sorted({str(c["target"]) for c in checks
                        if not c.get("enforced") and c.get("drop") is not None})
    ro_exceeded = sorted({str(c["target"]) for c in checks
                          if not c.get("enforced") and c.get("drop") is not None
                          and c["drop"] > eps})
    ro_worst = max((float(c["drop"]) for c in checks
                    if not c.get("enforced") and c.get("drop") is not None),
                   default=None)
    result: dict[str, Any] = {
        "pass": ok, "epsilon": eps, "worst_drop": worst_drop,
        "worst_drop_any_axis": worst_any,
        "checks": checks, "scored_record_ids": scored,
        "guarded_targets": list(enforced),
        # ``guarded_targets`` is the DECLARED contract and stays that way; these two
        # are the measured truth beside it, so "three guards" can no longer be read
        # off a run that judged one.  Absence of these keys on an older artifact
        # means "predates the 2026-07-26 coverage stamp", never "fully covered".
        "guarded_measured": guarded_measured,
        "blind_targets": blind_targets,
        "cells_scored": int(cells_entered),
        "report_only_targets": report_only,
        "scored_targets": [t[0] for t in NOREG_TARGETS],
        "unavailable": unavailable,
        "fr_guard": fr_guard_block(fr_guarded=bool(fr_guarded),
                                   band_labels=band_labels,
                                   licensing_band_labels=lic_labels),
    }
    notes: list[str] = []
    if unavailable:
        # Enforced and report-only axes are named separately: calling a demoted
        # axis an "UNJUDGED guarded axis" would overstate what the run promised.
        un_enf = sorted({str(c["target"]) for c in unavailable if c.get("enforced")})
        un_ro = sorted({str(c["target"]) for c in unavailable
                        if not c.get("enforced")})
        reasons = sorted({str(c["unavailable"]) for c in unavailable})
        if un_enf:
            notes.append(
                "UNJUDGED guarded axes: " + ", ".join(un_enf) + " — "
                + "; ".join(reasons)
                + ". These axes were NOT verified regression-free."
            )
        if un_ro:
            notes.append("UNJUDGED report-only axes: " + ", ".join(un_ro)
                         + " (no teeth either way).")
    # The coverage line is the point of the whole stamp: "fewer than 3 rows carry a
    # map_cov label" reads like a couple of thin cells, when the real shape can be
    # 1 cell of 36.  Always emitted when an enforced axis is short, PASS or FAIL.
    if cells_entered:
        short = [t for t in enforced
                 if guarded_measured.get(t, {}).get("cells", 0) < cells_entered]
        if short:
            spans = ", ".join(
                f"{t} {guarded_measured.get(t, {}).get('cells', 0)}/{cells_entered}"
                f" cells ({guarded_measured.get(t, {}).get('rows', 0)} rows)"
                for t in enforced)
            notes.append(f"GUARD COVERAGE: {spans}.")
        if blind_targets:
            notes.append(
                "BLIND guarded axes: " + ", ".join(blind_targets)
                + " — measured in NO cell; `pass` says nothing whatever about them."
            )
    if ro_scored:
        notes.append(fr_report_only_note(
            ro_scored, measured=f"worst {ro_worst:.4f} vs eps {eps:.4f}",
            exceeded=ro_exceeded))
    if notes:
        result["note"] = " | ".join(notes)
    return result


def gate_legacy_tail(
    old_model: Any,
    new_model: Any,
    records_df: Any,
    *,
    bands: Sequence[Sequence[float]],
    feed: int,
    sample_per_band: int,
    seed: int,
    epsilon: float,
    enabled: bool = True,
) -> dict[str, Any]:
    """Legacy-corpus high-cyclen tail no-regression gate (reusable core of
    ``_gate_legacy_tail``).  Scores BOTH champions on a fixed stable-hash sample of
    Dataset-A rows per band via ``predict_rows_raw`` and fails when the candidate's
    cyclen MAE degrades by more than ``epsilon`` EFPD.  Explicit inputs only."""
    if not enabled or old_model is None or new_model is None:
        return {"pass": True, "note": "disabled or no previous champion"}
    if not (hasattr(old_model, "predict_rows_raw")
            and hasattr(new_model, "predict_rows_raw")):
        return {"pass": True, "note": "model lacks predict_rows_raw"}
    return score_legacy_tail_no_regression(
        old_model, new_model, records_df,
        bands=bands, feed=feed, sample_per_band=sample_per_band,
        seed=seed, epsilon=epsilon)


# --------------------------------------------------------------------------- #
# fuel / pair selection helpers
# --------------------------------------------------------------------------- #
def select_band_types(fuel_library: Any, lo: float, hi: float, library_id: str
                      ) -> tuple[list[str], list[str]]:
    """Return ``(full_physics_types, feature_poor_types)`` with ``u_avg`` in band."""
    df = fuel_library.frame
    sub = df[(df["library_id"] == library_id)
             & (df["u_avg_enrichment"] >= lo)
             & (df["u_avg_enrichment"] < hi)]
    full = sub[~sub["feature_poor"].astype(bool)]["type_id"].astype(str).tolist()
    poor = sub[sub["feature_poor"].astype(bool)]["type_id"].astype(str).tolist()
    return sorted(full), sorted(poor)


def select_cell_pairs(
    cfg_curr: Any,
    the_cell_id: str,
    band: Sequence[float],
    feed: int,
    fuel_library: Any,
    library_id: str,
) -> list[str]:
    """Choose the in-band pairs for a cell.

    Priority: explicit ``[curriculum.cell_pairs]`` override -> global
    ``[curriculum].pairs`` (filtered to in-band) -> auto pairs formed from the
    band's full-physics (else feature-poor) types whose ``pair_e_core`` lands in
    the band.
    """
    lo, hi = float(band[0]), float(band[1])
    override = dict(getattr(cfg_curr, "cell_pairs", {}) or {})
    if the_cell_id in override and override[the_cell_id]:
        return list(override[the_cell_id])

    def _ec(a: str, b: str) -> float | None:
        try:
            return float(fuel_library.pair_e_core(a, b, 0.5, library_id))
        except Exception:  # noqa: BLE001 — unknown type / missing mass
            return None

    if getattr(cfg_curr, "pairs", None):
        keep = []
        for pr in cfg_curr.pairs:
            if "_" not in pr:
                continue
            a, b = pr.split("_", 1)
            ec = _ec(a, b)
            if ec is not None and lo <= ec < hi:
                keep.append(pr)
        if keep:
            return keep[: cfg_curr.max_pairs]

    full, poor = select_band_types(fuel_library, lo, hi, library_id)
    types = full or poor
    pairs: list[str] = []
    for i in range(0, len(types) - 1, 2):
        a, b = types[i], types[i + 1]
        ec = _ec(a, b)
        if ec is not None and lo <= ec < hi:
            pairs.append(f"{a}_{b}")
    if not pairs:  # same-type fallback
        for t in types:
            ec = _ec(t, t)
            if ec is not None and lo <= ec < hi:
                pairs.append(f"{t}_{t}")
    return pairs[: cfg_curr.max_pairs]


# --------------------------------------------------------------------------- #
# pin-burnup verifier factory (the default produce verifier has it OFF)
# --------------------------------------------------------------------------- #
def make_pin_burnup_verifier(
    cfg: LpoptConfig,
    run_dir: Path,
    resolver: Any,
    *,
    dry_run: bool = False,
) -> Any:
    """A :class:`WaveVerifier` whose equilibrium runner enables pin burnup.

    Dry runs use :class:`StubEvaluator` (which already emits a non-None
    ``max_pin_burnup``).  Live runs inject an evaluator factory that constructs
    ``EquilibriumRunner(..., enable_pin_burnup=True)`` and force ``stage_decks``
    so fallback restarts get the reload-deck rewrite + validation.
    """
    from .search.verify import (
        HarvestingEquilibriumEvaluator,
        PurgingEquilibriumRunner,
        WaveVerifier,
        WatchdogMasterRunner,
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        from .search.stub import StubEvaluator
        stub = StubEvaluator()
        return WaveVerifier(
            run_dir=run_dir,
            workers=cfg.produce.workers or 8,   # dry-run: legacy 8-wide stub wave
            max_cycles=cfg.produce.max_cycles,
            consecutive=cfg.produce.consecutive,
            evaluator_factory=lambda worker_id, cpu_core: stub,
            resolver=resolver,
            purge_intermediate=cfg.produce.purge_intermediate,
        )

    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    # PER-LIBRARY ROUTING: the MASTER package_root (lib/MAS_XSL + MAS_HFF staging)
    # and the reload-deck %GEN_DIM sanity gate must follow the resolver's library.
    # The resolver already carries both (ga80 -> [verify].package_root + (83,85);
    # paramA -> the design package + its own dims), so derive them from it and keep
    # ga80 byte-identical.
    package_root = getattr(resolver, "package_root", None)
    if package_root is None and cfg.verify.package_root:
        p = Path(cfg.verify.package_root)
        package_root = p if p.is_absolute() else (base / p)
    from .search.assets import LIBRARY_DIMS
    library_dims = tuple(getattr(resolver, "library_dims", LIBRARY_DIMS))
    executable = cfg.master.executable
    work_root = run_dir / "master_work"
    cache_dir = run_dir / "master_cache"
    harvest = bool(getattr(cfg.verify, "harvest_maps", False))
    # ``WaveVerifier`` widens ``keep_success`` the same way for its own default
    # factory: the EDIT5 read happens AFTER the chain converges, so the final
    # cycle's MAS_SUM has to survive or the harvest finds an empty work dir and
    # reports "no readable map" for a purge.  Mirrored here because an injected
    # factory builds its own runner and inherits nothing.
    keep_success = bool(cfg.master.keep_success) or harvest

    def _factory(worker_id: int, cpu_core: int | None) -> Any:
        from .vendor.masterrl.search import EquilibriumEvaluator
        master = WatchdogMasterRunner(
            package_root,
            executable,
            work_root=work_root / f"worker_{worker_id:02d}",
            cache_dir=cache_dir / f"worker_{worker_id:02d}",
            timeout=cfg.produce.chain_timeout,
            keep_success=keep_success,
            cpu_core=cpu_core,
        )
        # PurgingEquilibriumRunner: intermediate cycles are deleted per-cycle so
        # the injected (real pin-burnup) factory honours the [produce] directive
        # exactly like the default produce verifier (USER DIRECTIVE).
        runner = PurgingEquilibriumRunner(
            master,
            max_cycles=cfg.produce.max_cycles,
            consecutive=cfg.produce.consecutive,
            tolerances=cfg.master.tolerances,
            keep_success=keep_success,
            enable_pin_burnup=True,
            purge_intermediate=cfg.produce.purge_intermediate,
        )
        # THE fix (2026-07-26).  ``harvest_maps`` on the WaveVerifier only decides
        # whether ``_result_to_outcome`` READS ``metadata["maps"]``; the only place
        # that WRITES that key is HarvestingEquilibriumEvaluator, installed by
        # ``WaveVerifier._default_factory`` — which an injected factory bypasses
        # entirely.  So the curriculum path set the flag, changed nothing that
        # could produce a map, and left ``WaveOutcome.maps`` None forever: the
        # flat_power mini campaign RANKED on flatness and could never MEASURE it.
        if harvest:
            return HarvestingEquilibriumEvaluator(runner)
        return EquilibriumEvaluator(runner)

    return WaveVerifier(
        run_dir=run_dir,
        package_root=package_root,
        executable=executable,
        workers=cfg.produce.workers,           # 0 = auto (fill the core pool)
        # curriculum IS data production -> inherit the [produce] all-cores policy.
        use_all_cores=cfg.produce.use_all_cores,
        host_reserve=cfg.produce.host_reserve,
        # The injected _factory is a REAL pin-burnup MASTER runner (not a stub):
        # force P/E core detection + 1:1 pinning so it fills every core just like
        # the default produce verifier (which does so because it injects nothing).
        assign_cores=True,
        timeout=cfg.produce.chain_timeout,
        max_cycles=cfg.produce.max_cycles,
        consecutive=cfg.produce.consecutive,
        tolerances=cfg.master.tolerances,
        keep_success=cfg.master.keep_success,
        evaluator_factory=_factory,
        resolver=resolver,
        stage_decks=True,
        library_dims=library_dims,
        purge_intermediate=cfg.produce.purge_intermediate,
        # Honour ``[verify] harvest_maps`` exactly as ``search.produce`` and
        # ``search.campaign`` do.  Without it the validate_gate mini campaign can
        # RANK on the flatness objective but cannot MEASURE what its own MASTER
        # calls achieved (``WaveOutcome.maps`` stays None), which is how a gate
        # ends up reporting ``best_node_peak: null`` forever.  Default False, so a
        # deck that never asked for maps is byte-identical.  This flag is only the
        # READ side — ``_factory`` above installs the evaluator that WRITES the
        # map, and both must be driven by the same ``harvest`` value.
        harvest_maps=harvest,
    )


def _short_run_root(cfg: LpoptConfig, cid: str, kind: str) -> Path:
    """A SHORT MASTER work root (Windows MAX_PATH=260 guard).

    The deep ``data/curriculum/cells/<cid>/...`` tree plus the verifier's own
    ``master_work/worker_NN/<digest>__<restart_tag>/MAS_RST.*`` staging overruns
    260 chars on this (already deep, non-ASCII) project path, and MASTER then
    reports ``FILE DOES NOT EXIST`` for a restart that is physically staged.
    Routing live MASTER work to ``<output_root>/cur<hash><kind>`` keeps the full
    path well under the limit while artifacts stay under the cell dir.
    """
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    out = Path(cfg.flow.output_root)
    out = out if out.is_absolute() else (base / out)
    tag = hashlib.sha1(cid.encode("utf-8")).hexdigest()[:4]
    return out / f"cur{tag}{kind}"


# Per-library resolver routing (ga80 vs paramA) lives in the shared
# lpopt.search.resolver factory so the curriculum AND the plain produce / kit path
# stay in lock-step (a paramA stratum on a kit PC resolves exactly as a paramA
# curriculum cell does).  Keep the historical private names as thin aliases — the
# curriculum body + tests reference them.
from .search.resolver import (  # noqa: E402
    build_case_resolver as _build_resolver,
    is_paramA_library as _is_paramA_library,
    paramA_library_dims as _paramA_library_dims,
    paramA_package_root as _paramA_package_root,
    paramA_registry_aliases as _paramA_registry_aliases,
)


# --------------------------------------------------------------------------- #
# candidate generation for the blind probe / mini campaign
# --------------------------------------------------------------------------- #
def _gen_candidates(
    pairs: Sequence[str],
    feed: int,
    n: int,
    rng: random.Random,
) -> list[tuple[str, Any]]:
    """Diverse ``(pair, Pattern)`` candidates: round-robin over pairs, mixing
    plain random and heuristic-overlap genomes; deduped by canonical pattern."""
    from .search.produce import heuristic_fresh_set
    n_fresh = fresh_units_from_feed(feed)
    allow_single = n_fresh > 30
    out: list[tuple[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        pair = pairs[len(out) % len(pairs)]
        if rng.random() < 0.5:  # heuristic-overlap
            rule = rng.choice(("ring", "checker", "radial"))
            target = heuristic_fresh_set(rule, n_fresh)
            best = None
            best_ov = -1
            for _ in range(16):
                g = random_genome(rng, pair, n_fresh,
                                  max_shuffle_depth=2,
                                  allow_single_cycle_discharge=allow_single)
                ov = len(g.fresh_units & target)
                if ov > best_ov:
                    best_ov, best = ov, g
            genome = best
        else:
            genome = random_genome(rng, pair, n_fresh,
                                   max_shuffle_depth=2,
                                   allow_single_cycle_discharge=allow_single)
        pat = genome.to_pattern()
        key = f"{pair}|{pat.canonical()}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pair, pat))
    return out


# --------------------------------------------------------------------------- #
# curriculum driver
# --------------------------------------------------------------------------- #
class CurriculumDriver:
    """Crash-safe, resumable cell-sequential curriculum driver."""

    def __init__(
        self,
        cfg: LpoptConfig,
        *,
        dry_run: bool = False,
        state_dir: str | Path | None = None,
        progress: bool = True,
        log: Callable[[str], None] | None = None,
        model_loader: Callable[[str | Path], Any] | None = None,
        make_verifier: Callable[[Path, bool], Any] | None = None,
        retrain_hook: Callable[[str, str | None], str] | None = None,
        produce_hook: Callable[[str, dict], Any] | None = None,
        fuel_library: Any = None,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.curr = cfg.curriculum
        self.dry_run = bool(dry_run)
        self.progress = progress
        self._log = log or (lambda m: print(m))
        self.rng = rng or random.Random(cfg.flow.random_seed)

        self._base = cfg.source_path.parent if cfg.source_path else Path.cwd()
        self.state_dir = Path(state_dir) if state_dir else self._resolve(self.curr.state_dir)
        self.cells_dir = self.state_dir / "cells"
        self.state_path = self.state_dir / "state.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cells_dir.mkdir(parents=True, exist_ok=True)

        self.store_dir = self._resolve(self.cfg.model.store_dir)
        self.splits_dir = self._resolve("data/splits")

        self._model_loader = model_loader
        self._make_verifier = make_verifier
        self._retrain_hook = retrain_hook
        self._produce_hook = produce_hook
        self._fuel_library = fuel_library

        self.state: dict[str, Any] = {}

    # -- paths ------------------------------------------------------------- #
    def _resolve(self, p: str | Path) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (self._base / pp)

    def cell_dir(self, cid: str) -> Path:
        d = self.cells_dir / cid
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- per-band library resolution -------------------------------------- #
    def _band_library(self, band: Sequence[float]) -> str:
        """Resolve the effective fuel library for a band's ``ensure_types`` gate,
        pair selection, and per-cell produce stratum.

        Bands at/above ``[curriculum] paramA_band_lo`` (default 5.75 w/o) use the
        paramA parametric library — ga80's letter roster has no full-physics types
        there — while lower bands keep ``[curriculum] library`` (ga80).  An explicit
        ``[curriculum] band_libraries`` entry (keyed by the canonical band label,
        e.g. ``"5.75-6"``) overrides both.
        """
        lo, hi = float(band[0]), float(band[1])
        override = dict(getattr(self.curr, "band_libraries", {}) or {})
        key = band_label(lo, hi)
        if override.get(key):
            return str(override[key])
        if lo >= float(getattr(self.curr, "paramA_band_lo", 5.75)):
            return str(getattr(self.curr, "paramA_library", "paramA") or self.curr.library)
        return self.curr.library

    # -- fuel library ------------------------------------------------------ #
    def fuel_library(self) -> Any:
        if self._fuel_library is None:
            from .data.fuel_types import FuelLibrary
            fpath = self.store_dir / "fuel_types.parquet"
            if fpath.exists():
                self._fuel_library = FuelLibrary.from_parquet(fpath)
            else:
                self._fuel_library = FuelLibrary.build(self.cfg, persist=True)
        return self._fuel_library

    # -- model ------------------------------------------------------------- #
    def load_model(self, model_dir: str | Path) -> Any:
        if self._model_loader is not None:
            return self._model_loader(model_dir)
        from .model.model_api import PosValCnnBackend
        return PosValCnnBackend.from_dir(
            model_dir, store_dir=self.store_dir,
            library_id=self.cfg.model.library_id, device=self.cfg.model.device,
        )

    # -- verifier ---------------------------------------------------------- #
    def verifier(self, run_dir: Path, resolver: Any) -> Any:
        if self._make_verifier is not None:
            return self._make_verifier(run_dir, self.dry_run)
        return make_pin_burnup_verifier(self.cfg, run_dir, resolver, dry_run=self.dry_run)

    # -- state ------------------------------------------------------------- #
    def _init_state(self) -> None:
        bands = [tuple(b) for b in self.curr.e_core_bands]
        feeds = list(self.curr.feeds)
        order_spec = ring_order(bands, feeds, tuple(self.curr.anchor_band), self.curr.anchor_feed)
        if self.curr.cell_order:
            order = list(self.curr.cell_order)
            # map explicit ids back to (band, feed, ring)
            by_id = {cell_id(b, f): (b, f, r) for (b, f, r) in order_spec}
            meta = {cid: by_id.get(cid) for cid in order}
        else:
            order = [cell_id(b, f) for (b, f, r) in order_spec]
            meta = {cell_id(b, f): (b, f, r) for (b, f, r) in order_spec}

        cells: dict[str, Any] = {}
        for cid in order:
            m = meta.get(cid)
            if m is None:
                continue
            b, f, r = m
            cells[cid] = {
                "band": [b[0], b[1]],
                "feed": int(f),
                "ring": int(r),
                "phase": "ensure_types",
                "pairs": [],
                "budget": {"master_calls": 0, "wall_s": 0.0},
                "created_at": _now(),
                "updated_at": _now(),
            }
        self.state = {
            "anchor_band": list(self.curr.anchor_band),
            "anchor_feed": int(self.curr.anchor_feed),
            "order": order,
            "cursor": 0,
            "champion_model_dir": str(self._resolve(self.cfg.model.model_dir)),
            "cells": cells,
            "created_at": _now(),
        }
        self._save_state()

    def _load_state(self) -> None:
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self._init_state()

    def _save_state(self) -> None:
        _atomic_write_json(self.state_path, self.state)

    def _touch(self, cid: str) -> None:
        self.state["cells"][cid]["updated_at"] = _now()

    # ------------------------------------------------------------------ #
    # public entry
    # ------------------------------------------------------------------ #
    def run(self, *, max_cells: int | None = None, resume: bool = True) -> dict[str, Any]:
        if resume and self.state_path.exists():
            self._load_state()
        else:
            self._init_state()

        processed = 0
        summary_cells: list[dict[str, Any]] = []
        while self.state["cursor"] < len(self.state["order"]):
            if max_cells is not None and processed >= max_cells:
                break
            cid = self.state["order"][self.state["cursor"]]
            if cid not in self.state["cells"]:
                self.state["cursor"] += 1
                continue
            self._log(f"[curriculum] === cell {cid} "
                      f"(ring {self.state['cells'][cid].get('ring')}) ===")
            outcome = self._run_cell(cid)
            summary_cells.append({"cell": cid, "outcome": outcome,
                                  "phase": self.state["cells"][cid]["phase"]})
            if outcome in ("pending", "fail"):
                self._save_state()
                return {"status": outcome, "cell": cid,
                        "phase": self.state["cells"][cid]["phase"],
                        "cursor": self.state["cursor"], "cells": summary_cells,
                        "resume_cmd": self._resume_cmd()}
            # outcome == "done"
            self.state["cursor"] += 1
            processed += 1
            self._save_state()

        done = self.state["cursor"] >= len(self.state["order"])
        return {"status": "complete" if done else "paused",
                "cursor": self.state["cursor"], "cells": summary_cells,
                "resume_cmd": self._resume_cmd()}

    def _resume_cmd(self) -> str:
        deck = self.cfg.source_path.name if self.cfg.source_path else "lpopt.inp"
        return f"lpopt curriculum --input {deck} --resume"

    # ------------------------------------------------------------------ #
    # per-cell state machine
    # ------------------------------------------------------------------ #
    def _run_cell(self, cid: str) -> str:
        handlers = {
            "ensure_types": self._phase_ensure_types,
            "blind_probe": self._phase_blind_probe,
            "produce_cell": self._phase_produce_cell,
            "retrain": self._phase_retrain,
            "validate_gate": self._phase_validate_gate,
        }
        while True:
            phase = self.state["cells"][cid]["phase"]
            if phase == "done":
                return "done"
            handler = handlers[phase]
            status = handler(cid)
            self._touch(cid)
            self._save_state()
            if status == "advance":
                self._advance_phase(cid)
                continue
            return status  # "pending" or "fail"

    def _advance_phase(self, cid: str) -> None:
        phase = self.state["cells"][cid]["phase"]
        nxt = PHASES[PHASES.index(phase) + 1]
        self.state["cells"][cid]["phase"] = nxt

    # ---- phase 1: ensure_types ------------------------------------------ #
    def _phase_ensure_types(self, cid: str) -> str:
        cell = self.state["cells"][cid]
        band, feed = cell["band"], cell["feed"]
        lib = self.fuel_library()
        # Per-band library: bands above ga80's roster ceiling resolve to paramA so
        # the gate/pair selection see the (pre-)generated parametric types.
        lib_id = self._band_library(band)
        full, poor = select_band_types(lib, band[0], band[1], lib_id)
        n_usable = len(full) if len(full) >= self.curr.min_band_types else len(full) + len(poor)

        if n_usable < self.curr.min_band_types:
            # On-demand Phase-A design generation for this band (bands with too
            # few library types, typically > 5.5 w/o where ga80 is feature-poor).
            if not self.curr.allow_design or self.dry_run:
                cell["ensure_types"] = {
                    "status": "insufficient", "n_full": len(full), "n_poor": len(poor),
                    "note": "band lacks types and design generation disabled/dry",
                }
                self._log(f"[curriculum][{cid}] ensure_types: insufficient types "
                          f"({len(full)} full, {len(poor)} poor); design disabled -> halt")
                return "fail"
            gen = self._generate_band_designs(cid, band, feed)
            cell["ensure_types"] = gen
            if not gen.get("ok"):
                return "fail"
            lib = self.fuel_library()  # rebuilt with paramA rows
            lib_id = gen.get("library_id", lib_id)

        pairs = select_cell_pairs(self.curr, cid, band, feed, lib, lib_id)
        if not pairs:
            cell["ensure_types"] = {"status": "no_pairs", "n_full": len(full)}
            self._log(f"[curriculum][{cid}] ensure_types: no in-band pairs -> halt")
            return "fail"
        cell["pairs"] = pairs
        cell["library_id"] = lib_id
        cell.setdefault("ensure_types", {})
        cell["ensure_types"].update({
            "status": "ok", "n_full": len(full), "n_poor": len(poor),
            "pairs": pairs, "library_id": lib_id,
        })
        self._log(f"[curriculum][{cid}] ensure_types: OK "
                  f"({len(full)} full-physics types, pairs={pairs})")
        return "advance"

    def _generate_band_designs(self, cid: str, band: Sequence[float], feed: int) -> dict:
        """On-demand Phase-A chain: LHS designs in-band -> DeCART -> extend the ONE
        growing paramA library -> assemble -> bootstrap feed-121 band restart.

        Reuses lpopt.design.* building blocks.  Extends a single paramA library
        (union of old+new HGCs) keyed by a persisted DesignRegistry so aliases /
        COMP names stay stable; band restarts are (re)generated feed-121 first.
        """
        from .design.spec import DesignRegistry, DESIGN_GRID, FuelDesign
        from .design.lattice import run_batch
        from .design.spec import lhs_grid  # noqa: F401 (kept for parity/future)
        from .design.package import (DesignSource, assemble_package,
                                     ingest_fuel_types)
        from .design.bootstrap import make_band_restart

        d = self.cfg.design
        apr = self._resolve(d.apr1400_root)
        pkg = self._resolve(d.package_root) if d.package_root else (self._resolve(d.store_dir) / "package")
        pkg.mkdir(parents=True, exist_ok=True)
        registry = DesignRegistry.load(pkg / "registry.json")

        lo, hi = float(band[0]), float(band[1])
        # target e1 near band centre so the single-assembly average lands in band
        centre = 0.5 * (lo + hi)
        e1_opts = [e for e in DESIGN_GRID["e1"] if lo - 0.15 <= e <= hi + 0.3] or [
            min(DESIGN_GRID["e1"], key=lambda e: abs(e - centre))]
        designs: list[FuelDesign] = []
        rng = random.Random(_stable_hash(cid) & 0xFFFF)
        for _ in range(self.curr.design_n_types):
            e1 = rng.choice(e1_opts)
            ratio = rng.choice(DESIGN_GRID["ratio"])
            zv = rng.choice(DESIGN_GRID["zoning_variant"])
            gd = rng.choice(DESIGN_GRID["gd_wt"])
            ng = rng.choice(DESIGN_GRID["n_gd"])
            designs.append(FuelDesign(e1, round(e1 * ratio, 3), zv, gd, ng))
        # dedup
        uniq: dict[Any, FuelDesign] = {}
        for de in designs:
            uniq.setdefault(de.key, de)
        designs = list(uniq.values())
        registry.register_all(designs)
        registry.save(pkg / "registry.json")

        work = self._resolve(d.store_dir) / "curriculum_work" / cid
        runs = run_batch(designs, work, registry, apr,
                         exe=d.decart_exe, max_parallel=d.max_parallel,
                         timeout_s=d.decart_timeout)
        new_sources: list[DesignSource] = []
        for r in runs:
            if r.hgc_path is not None:
                new_sources.append(DesignSource(design=r.design, alias=r.alias,
                                                hgc_path=r.hgc_path, out_path=r.out_path))
        if len(new_sources) < self.curr.min_band_types:
            return {"ok": False, "status": "decart_short",
                    "produced": len(new_sources), "note": "too few HGCs for band"}
        # Extend the ONE growing paramA library: assemble over the UNION of the
        # already-packaged designs and this band's new designs.  build_master_library
        # rejects any assemble whose request omits an HGC already staged in lib/ (its
        # stale-HGC guard), so passing only the new band's sources would clobber
        # designs.json and then crash every band generated after the first.  Existing
        # products live at <pkg>/hgc/FA_<alias>.HGC (+ .out); merge them by alias
        # (this band's new products win on collision).
        by_alias: dict[str, DesignSource] = {s.alias: s for s in new_sources}
        try:
            packaged = json.loads(
                (pkg / "designs.json").read_text(encoding="utf-8")).get("designs", [])
        except (OSError, ValueError):
            packaged = []
        for rec in packaged:
            al = rec.get("alias")
            if not al or al in by_alias:
                continue
            hgc = pkg / "hgc" / f"FA_{al}.HGC"
            if not hgc.is_file():
                continue
            try:
                de_prev = FuelDesign.from_dict(rec)
            except (KeyError, ValueError, TypeError):
                continue
            out = pkg / "hgc" / f"FA_{al}.out"
            by_alias[al] = DesignSource(design=de_prev, alias=al, hgc_path=hgc,
                                        out_path=out if out.is_file() else None)
        sources = list(by_alias.values())
        assemble_package(pkg, sources, registry, apr)
        ingest_fuel_types(pkg, cfg=self.cfg, store_path=self.store_dir / "fuel_types.parquet")
        # refresh fuel library so paramA rows are visible
        self._fuel_library = None
        self.cfg.design.paramA_root = str(pkg)
        # bootstrap a feed-121 band restart with an in-band pair (first two aliases)
        pair121 = "?"
        try:
            aliases = [registry.alias(de) for de in designs[:2]]
            pair121 = f"{aliases[0]}_{aliases[1]}"
            exe = d.master_exe or self.cfg.master.executable
            res = make_band_restart(pkg, pair121, 121, random.Random(d.seed),
                                    exe=exe, max_cycles=d.bootstrap_max_cycles,
                                    enable_pin_burnup=d.enable_pin_burnup,
                                    purge_intermediate=self.cfg.produce.purge_intermediate)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[curriculum][{cid}] band restart bootstrap warning: {exc}")
            res = None
        # make_band_restart funnels failures into res.error / res.converged=False
        # rather than raising, so the try/except above sees a clean return on the
        # exact failure that matters: no band seed in bases/, every later cell of
        # this band then resolving at a fallback restart.  Check the RESULT the
        # way cli.py (cmd_design_bootstrap) and pathfinder.py already do.
        if res is None or res.error or not res.converged:
            err = (res.error or "not converged") if res is not None else "raised"
            self._log(f"[curriculum][{cid}] band restart bootstrap FAILED "
                      f"({pair121}/feed-121): {err}")
            return {"ok": False, "status": "bootstrap_failed", "error": err,
                    "n_designs": len(designs), "n_hgc": len(new_sources),
                    "n_library": len(sources), "library_id": "paramA",
                    "package": str(pkg)}
        return {"ok": True, "status": "generated", "n_designs": len(designs),
                "n_hgc": len(new_sources), "n_library": len(sources),
                "library_id": "paramA", "package": str(pkg)}

    # ---- phase 2: blind_probe (THE methodology measurement) ------------- #
    def _phase_blind_probe(self, cid: str) -> str:
        cell = self.state["cells"][cid]
        cdir = self.cell_dir(cid)
        out_path = cdir / "blind_probe.json"
        if out_path.exists() and cell.get("blind_probe", {}).get("status") == "ok":
            return "advance"

        band, feed, pairs = cell["band"], cell["feed"], cell["pairs"]
        champ = self.state["champion_model_dir"]
        t0 = _now()
        self._log(f"[curriculum][{cid}] blind_probe: champion={Path(champ).name} "
                  f"probe_size={self.curr.probe_size}")

        lib = self.fuel_library()
        rng = random.Random(self.cfg.flow.random_seed + _stable_hash(cid) % 9973)
        cands = _gen_candidates(pairs, feed, self.curr.probe_size, rng)
        patterns = [p for _pair, p in cands]
        cases = [CaseKey(pair, int(feed)) for pair, _p in cands]

        model = self.load_model(champ)
        pred = model.predict(patterns, cases)
        try:
            extra = model.predict_extra(patterns, cases)
            disch = extra.value("discharge_burnup")
        except Exception:  # noqa: BLE001
            disch = np.full(len(patterns), np.nan)
        conv = model.predict_convergence(patterns, cases)
        pfeas = self._p_feasible(pred, conv)

        # live evaluation (pin burnup ON); short work root (Windows MAX_PATH)
        cell_lib = cell.get("library_id") or self._band_library(band)
        resolver = _build_resolver(self.cfg, lib, cell_lib)
        verifier = self.verifier(_short_run_root(self.cfg, cid, "p"), resolver)
        entries = self._wave_entries(cands, feed, resolver, lib, cell_lib)
        outcomes = verifier.evaluate_wave(entries)

        rows, actuals_by_target = self._collect_probe(cands, pred, disch, conv, pfeas, outcomes)
        per_target = self._transfer_stats(rows)
        n_conv = sum(1 for r in rows if r["status"] == "converged")
        n_nonconv = sum(1 for r in rows if r["status"] == "nonconverged")
        n_err = sum(1 for r in rows if r["status"] == "error")

        payload = {
            "cell": cid, "band": band, "feed": feed, "pairs": pairs,
            "ring": cell.get("ring"),
            "model_dir": str(champ),
            "n_probe": len(rows), "n_converged": n_conv,
            "n_nonconverged": n_nonconv, "n_error": n_err,
            "master_calls": len(rows),
            "per_target": per_target,
            "candidates": rows,
            "wall_s": _now() - t0,
        }
        _atomic_write_json(out_path, payload)
        cell["blind_probe"] = {"status": "ok", "n_converged": n_conv,
                               "per_target": per_target, "path": str(out_path)}
        cell["budget"]["master_calls"] += len(rows)
        cell["budget"]["wall_s"] += payload["wall_s"]
        self._log_transfer_table(cid, per_target, n_conv, len(rows))
        return "advance"

    def _p_feasible(self, pred: Any, conv: Any) -> Any:
        from .search import acquisition as acq
        cons = acq.make_constraints(self.cfg.acquisition)
        try:
            return acq.p_feasible(pred, cons, convergence=conv)
        except Exception:  # noqa: BLE001
            return np.full(pred.mean.shape[0], np.nan)

    def _wave_entries(self, cands, feed, resolver, lib, library_id=None) -> list:
        from .search.verify import WaveEntry
        split = float(self.curr.split_w1[0]) if self.curr.split_w1 else 0.5
        library_id = library_id or self.curr.library
        entries = []
        for pair, pat in cands:
            ck = CaseKey(pair, int(feed))
            assets = resolver.resolve(ck)
            a, b = (pair.split("_", 1) + [pair])[:2]
            try:
                ec = float(lib.pair_e_core(a, b, split, library_id))
            except Exception:  # noqa: BLE001
                ec = 0.0
            entries.append(WaveEntry(pattern=pat, case_key=ck,
                                     resolved_assets=assets, meta={"e_core": ec}))
        return entries

    def _collect_probe(self, cands, pred, disch, conv, pfeas, outcomes):
        from .data.schema import pack_pattern
        rows = []
        actuals_by_target: dict[str, list[tuple[float, float, float]]] = {
            name: [] for name, _c, _f in PROBE_TARGETS}
        for i, (pair, pat) in enumerate(cands):
            oc = outcomes[i]
            status = oc.status
            pred_row = {}
            for name, col, _fattr in PROBE_TARGETS:
                pred_row[name] = _finite(pred.mean[i, col])
            pred_row["discharge_burnup"] = _finite(disch[i]) if disch is not None else None
            calib_row = {name: _finite(pred.calibrated_std[i, col])
                         for name, col, _f in PROBE_TARGETS}
            actual_row: dict[str, float | None] = {}
            if status == "converged" and oc.fom is not None:
                fom = oc.fom
                for name, _col, fattr in PROBE_TARGETS:
                    actual_row[name] = _finite(getattr(fom, fattr, None))
                for name, col, _f in PROBE_TARGETS:
                    pv, av, sv = pred_row[name], actual_row[name], calib_row[name]
                    if pv is not None and av is not None:
                        actuals_by_target[name].append((pv, av, sv if sv else float("nan")))
            rows.append({
                "pair": pair,
                "pattern": pack_pattern(pat),
                "digest": pat.digest,
                "status": status,
                "n_cycles": oc.n_cycles,
                "pred": pred_row,
                "calib_std": calib_row,
                "actual": actual_row,
                "p_feas": _finite(pfeas[i]) if pfeas is not None else None,
                "conv_pred": _finite(conv[i]) if conv is not None else None,
            })
        self._actuals_cache = actuals_by_target
        return rows, actuals_by_target

    def _transfer_stats(self, rows) -> dict:
        by_target = getattr(self, "_actuals_cache", None)
        if by_target is None:
            by_target = {name: [] for name, _c, _f in PROBE_TARGETS}
            for r in rows:
                if r["status"] != "converged":
                    continue
                for name, _col, _f in PROBE_TARGETS:
                    pv = r["pred"].get(name)
                    av = r["actual"].get(name)
                    sv = r["calib_std"].get(name)
                    if pv is not None and av is not None:
                        by_target[name].append((pv, av, sv if sv else float("nan")))
        stats = {}
        for name, triples in by_target.items():
            if not triples:
                stats[name] = {"n": 0}
                continue
            preds = np.array([t[0] for t in triples], float)
            acts = np.array([t[1] for t in triples], float)
            sig = np.array([t[2] for t in triples], float)
            err = preds - acts
            z = err / sig
            zf = z[np.isfinite(z)]
            stats[name] = {
                "n": int(len(triples)),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "spearman": _spearman(acts.tolist(), preds.tolist()),
                "mean_abs_z": float(np.mean(np.abs(zf))) if len(zf) else None,
                "cov1": float(np.mean(np.abs(zf) <= 1.0)) if len(zf) else None,
                "cov2": float(np.mean(np.abs(zf) <= 2.0)) if len(zf) else None,
            }
        return stats

    def _log_transfer_table(self, cid, per_target, n_conv, n_total) -> None:
        self._log(f"[curriculum][{cid}] blind transfer ({n_conv}/{n_total} converged):")
        self._log(f"    {'target':16s} {'n':>3s} {'MAE':>9s} {'bias':>9s} "
                  f"{'spearman':>9s} {'cov1':>5s}")
        for name, _c, _f in PROBE_TARGETS:
            s = per_target.get(name, {})
            if not s.get("n"):
                self._log(f"    {name:16s} {'0':>3s}  (no converged labels)")
                continue
            sp = s.get("spearman")
            cov1 = s.get("cov1")
            self._log(f"    {name:16s} {s['n']:>3d} {s['mae']:>9.3f} {s['bias']:>9.3f} "
                      f"{(sp if sp is not None else float('nan')):>9.3f} "
                      f"{(cov1 if cov1 is not None else float('nan')):>5.2f}")

    # ---- phase 3: produce_cell ------------------------------------------ #
    def _phase_produce_cell(self, cid: str) -> str:
        cell = self.state["cells"][cid]
        cdir = self.cell_dir(cid)
        # per-cell override so a gate-failed cell can be re-entered with an
        # enlarged production target without touching the global config
        n_target = int(cell.get("n_target_override") or self.curr.n_target)

        # Injected produce hook (tests / custom orchestration) takes precedence.
        if self._produce_hook is not None:
            result = self._produce_hook(cid, cell)
            if isinstance(result, dict) and result.get("pending"):
                cell["produce"] = {"status": "pending",
                                   **{k: v for k, v in result.items() if k != "pending"}}
                self._log(f"[curriculum][{cid}] produce_cell: pending (hook)")
                return "pending"
            cell["produce"] = {"status": "done_hook", **(result or {})}
            return "advance"

        n_conv = self._converged_count(cid)
        if n_conv >= n_target:
            cell["produce"] = {"status": "done", "converged": n_conv}
            self._log(f"[curriculum][{cid}] produce_cell: already {n_conv}/{n_target} -> advance")
            return "advance"

        if self.dry_run:
            summ = run_cell_produce(self.cfg, cid, cell, self.store_dir,
                                    dry_run=True, log=self._log,
                                    fuel_library=self.fuel_library())
            cell["produce"] = {"status": "done_dry", **summ}
            return "advance"

        # live: manage a detached produce process
        done_marker = cdir / "produce.done"
        pid_file = cdir / "produce.pid"
        if done_marker.exists():
            n_conv = self._converged_count(cid)
            if n_conv < n_target:
                # marker from an earlier, smaller target (reopened cell) —
                # stale; production must continue
                done_marker.unlink()
                self._log(f"[curriculum][{cid}] produce_cell: stale done marker "
                          f"({n_conv}/{n_target}) -> removed, relaunching")
            else:
                cell["produce"] = {"status": "done", "converged": n_conv,
                                   "marker": str(done_marker)}
                self._log(f"[curriculum][{cid}] produce_cell: done marker present "
                          f"({n_conv} converged) -> advance")
                return "advance"
        if _pid_alive(pid_file):
            self._log(f"[curriculum][{cid}] produce_cell: still running "
                      f"(pid {pid_file.read_text().strip()}, {n_conv}/{n_target} converged)")
            return "pending"
        # launch (or relaunch) detached
        pid = self._launch_produce(cid, cdir)
        cell["produce"] = {"status": "launched", "pid": pid,
                           "log": str(cdir / 'produce.log'),
                           "converged_at_launch": n_conv}
        self._log(f"[curriculum][{cid}] produce_cell: launched detached produce "
                  f"(pid {pid}); log {cdir / 'produce.log'}")
        return "pending"

    def _converged_count(self, cid: str) -> int:
        from .data.store import StoreReader
        try:
            df = StoreReader(self.store_dir).records
        except (FileNotFoundError, OSError):
            return 0
        if df is None or not len(df):
            return 0
        sub = df[(df["dataset"] == "P") & (df["campaign"] == cid)
                 & (df["converged"] == True)]  # noqa: E712
        return int(len(sub))

    def _launch_produce(self, cid: str, cdir: Path) -> int:
        deck = str(self.cfg.source_path) if self.cfg.source_path else "lpopt.inp"
        log_path = cdir / "produce.log"
        cmd = [sys.executable, "-m", "lpopt", "curriculum-produce",
               "--input", deck, "--cell", cid]
        logf = open(log_path, "a", encoding="utf-8")
        # Detached background relaunch: DETACHED_PROCESS gives the child no
        # console at all (so no window pops), which is why this call does NOT use
        # no_window_flags() — CREATE_NO_WINDOW is mutually exclusive with
        # DETACHED_PROCESS.  (Allowlisted in tests/test_no_window.py.)
        create_new = 0
        if os.name == "nt":
            create_new = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) \
                | getattr(subprocess, "DETACHED_PROCESS", 0)
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(self._base), creationflags=create_new,
            close_fds=True,
        )
        (cdir / "produce.pid").write_text(str(proc.pid), encoding="utf-8")
        return proc.pid

    # ---- phase 4: retrain ----------------------------------------------- #
    def _phase_retrain(self, cid: str) -> str:
        cell = self.state["cells"][cid]
        prev_champ = self.state["champion_model_dir"]
        cell["champion_before"] = prev_champ
        t0 = _now()
        self._log(f"[curriculum][{cid}] retrain: mode="
                  f"{'local_finetune' if self.dry_run else self.curr.retrain_mode}")
        try:
            if self._retrain_hook is not None:
                new_dir = self._retrain_hook(cid, prev_champ)
            else:
                new_dir = self._default_retrain(cid, prev_champ)
        except Exception as exc:  # noqa: BLE001
            cell["retrain"] = {"status": "error", "error": str(exc)}
            self._log(f"[curriculum][{cid}] retrain FAILED: {exc}")
            return "fail"
        # local round-trip check
        rt = self._roundtrip_check(new_dir, cid)
        self.state["champion_model_dir"] = str(new_dir)
        cell["retrain"] = {"status": "ok", "model_dir": str(new_dir),
                           "roundtrip": rt, "wall_s": _now() - t0}
        self._log(f"[curriculum][{cid}] retrain OK -> {new_dir} "
                  f"(round-trip finite={rt.get('finite')})")
        return "advance"

    def _default_retrain(self, cid: str, prev_dir: str) -> str:
        mode = "local_finetune" if self.dry_run else self.curr.retrain_mode
        if mode == "local_finetune":
            from .model.model_api import EncoderChannelMismatch
            try:
                return self._retrain_local_finetune(cid, prev_dir)
            except EncoderChannelMismatch as exc:
                # Feature-schema change (e.g. a v3 champion but a v4 store): the
                # champion's stem width is fixed, so a fine-tune cannot bridge it.
                # Fall back to a from-scratch remote full retrain rather than fail.
                self._log(f"[curriculum][{cid}] retrain: feature-schema change "
                          f"({exc}); fine-tune impossible -> remote_full")
                return self._retrain_remote_full(cid)
        if mode == "local_full":
            return self._retrain_local_full(cid)
        return self._retrain_remote_full(cid)

    def _cell_rows(self, cid: str):
        from .data.store import StoreReader
        df = StoreReader(self.store_dir).records
        return df[(df["dataset"] == "P") & (df["campaign"] == cid)
                  & (df["converged"] == True)]  # noqa: E712

    # ---- curriculum retrain split -------------------------------------- #
    def _known_cell_ids(self) -> list[str]:
        """Every cell id the driver knows about (the campaign tags produce uses)."""
        return list(self.state.get("cells", {}).keys())

    def _reached_cell_ids(self) -> list[str]:
        """Cells the driver has STARTED — the only known cells whose (possibly
        pre-merged) rows may train the champion or be scored by the honest gate.

        Reached is defined so a cell becomes reached exactly when the driver begins
        it, i.e. at its ``ensure_types``/``blind_probe`` step, which is strictly
        BEFORE the ``retrain`` that could leak it — so including a just-reached cell
        never retroactively contaminates a blind probe that already ran.

        Derivation from the state machine (``_init_state`` + ``run``): every cell is
        created at phase ``ensure_types`` and ``cursor`` advances past a cell only
        once it reaches ``done``.  Therefore a cell is reached iff

          * its order-index is ``<= cursor`` — every ``done`` cell sits strictly
            below the cursor and the cell AT the cursor is the one in progress; OR
          * its phase has already advanced past the initial ``ensure_types`` (a
            belt-and-braces signal the driver touched it).

        A not-yet-started future cell has index ``> cursor`` AND phase
        ``ensure_types``, so neither clause fires and it stays quarantined until the
        driver reaches it.  Every unknown/holey order entry is skipped, mirroring
        ``run``'s own ``cid not in self.state['cells']`` guard.
        """
        order = list(self.state.get("order", []))
        cursor = int(self.state.get("cursor", 0))
        cells = self.state.get("cells", {})
        reached: list[str] = []
        for i, cid in enumerate(order):
            if cid not in cells:
                continue
            phase = cells.get(cid, {}).get("phase", "ensure_types")
            if i <= cursor or phase != "ensure_types":
                reached.append(cid)
        return reached

    def _blind_probe_ids_by_cell(self, cells: Sequence[str]) -> dict[str, list[str]]:
        """Map ``cell id -> blind-probe record_ids`` read from each cell's
        ``blind_probe.json``.  Candidates store no ``record_id``, so it is derived
        from ``(pattern, library_id, pair)`` the same way the store computes it;
        the split pins only the ones that actually exist as store rows.

        The record_id MUST be derived with ``PRODUCE_DECK_KNOBS`` (the deck-knob
        signature the curriculum produce path stamps on every stored P row), NOT
        the Dataset-A ``mocha_default`` — otherwise a blind-probe pattern that
        WAS produced would hash to a different id and silently escape the val pin.
        """
        from .data.schema import compute_record_id
        from .search.verify import PRODUCE_DECK_KNOBS
        lib = self.cfg.model.library_id or self.curr.library
        out: dict[str, list[str]] = {}
        for cid in cells:
            p = self.cells_dir / cid / "blind_probe.json"
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            ids: list[str] = []
            for c in data.get("candidates", []):
                rid = c.get("record_id")
                if rid is None:
                    pat, pair = c.get("pattern"), c.get("pair")
                    if pat and pair:
                        rid = compute_record_id(str(pat), lib, str(pair),
                                                PRODUCE_DECK_KNOBS)
                if rid:
                    ids.append(str(rid))
            if ids:
                out[cid] = ids
        return out

    def _curriculum_split_manifest(self, records=None):
        """Build the retrain split (``make_curriculum_split``) from the current
        store + known cell ids + per-cell blind-probe pins.  Deterministic under
        ``[flow] random_seed``, so the honest gate rebuilds an identical holdout.

        ``reached_cells`` (``_reached_cell_ids``) is threaded so any known cell the
        driver has not yet started has its pre-merged rows quarantined out of both
        folds — keeping that cell's future blind-probe transfer measurement blind."""
        from .model.splits import make_curriculum_split
        from .data.store import StoreReader
        if records is None:
            records = StoreReader(self.store_dir).records
        cells = self._known_cell_ids()
        bp = self._blind_probe_ids_by_cell(cells)
        reached = self._reached_cell_ids()
        return make_curriculum_split(
            records, cells=cells, blind_probe_ids_by_cell=bp,
            reached_cells=reached,
            seed=self.cfg.flow.random_seed, name=self.curr.retrain_split,
            cell_cap=self.curr.cell_weight_cap,
        )

    def _write_curriculum_split(self):
        """Persist the curriculum split as the ``data/splits/<retrain_split>.json``
        artifact the trainer consumes (unchanged filename/format) and return it."""
        manifest = self._curriculum_split_manifest()
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(self.splits_dir / f"{self.curr.retrain_split}.json")
        self._log(f"[curriculum] curriculum split -> {self.curr.retrain_split}.json "
                  f"(train={manifest.n_train} val={manifest.n_val}; "
                  f"cells={manifest.groups.get('cells')})")
        return manifest

    def _retrain_local_finetune(self, cid: str, prev_dir: str) -> str:
        from .data.store import StoreReader
        model = self.load_model(prev_dir)
        new = self._cell_rows(cid)
        allconv = StoreReader(self.store_dir).records
        allconv = allconv[allconv["converged"] == True]  # noqa: E712
        replay = allconv.sample(min(self.curr.replay_size, len(allconv)),
                                 random_state=self.cfg.flow.random_seed) \
            if len(allconv) else allconv
        model.finetune(new, replay, epochs=self.curr.finetune_epochs,
                       seed=self.cfg.flow.random_seed)
        out = self.state_dir / "models" / f"{cid}_{time.strftime('%Y%m%d_%H%M%S')}"
        model.save(out)
        # The weights just moved, so the inherited per-cell calibrations are
        # stale; refit them into the new dir (train rows only, leakage-asserted)
        # so this champion serves calibrated like the one it will be gated against.
        self._fit_cell_calibrations(out)
        return str(out)

    def _v5_train_config(self):
        """A :class:`TrainConfig` carrying the deck's v5 knobs.

        All four default OFF/TRUE-as-today, so a deck that does not mention them
        yields exactly the config the retrain used before the v5 bundle landed.
        """
        from .model.train import TrainConfig
        m = self.cfg.model
        cfg = TrainConfig()
        cfg.cell_weight_cap = float(getattr(self.curr, "cell_weight_cap",
                                            cfg.cell_weight_cap))
        cfg.cyclen_physics_prior = bool(getattr(m, "cyclen_physics_prior", False))
        cfg.quantile_heads = bool(getattr(m, "quantile_heads", False))
        cfg.promote_max_asm_bu = bool(getattr(m, "promote_max_asm_bu", False))
        cfg.auto_fit_cell_calibration = bool(
            getattr(m, "auto_fit_cell_calibration", True))
        # hires map-path structure — must travel WITH cond_schema (see the
        # ModelSection docstring): a v6 schema on a linear map head pays for the
        # extra channels and cannot use them.
        cfg.map_head_mode = str(getattr(m, "map_head_mode", "linear"))
        cfg.map_prior_residual = bool(getattr(m, "map_prior_residual", False))
        cfg.map_spectral_weight = float(getattr(m, "map_spectral_weight", 0.0))
        return cfg

    def _v5_train_flags(self) -> list[str]:
        """The deck's v5 knobs as ``lpopt.model.train`` CLI flags (remote path)."""
        m = self.cfg.model
        flags: list[str] = []
        if getattr(m, "cyclen_physics_prior", False):
            flags.append("--cyclen-physics-prior")
        if getattr(m, "quantile_heads", False):
            flags.append("--quantile-heads")
        if getattr(m, "promote_max_asm_bu", False):
            flags.append("--promote-max-asm-bu")
        if not getattr(m, "auto_fit_cell_calibration", True):
            flags.append("--no-auto-cell-calibration")
        # hires map-path structure (arm A6).  Mirrors ``_v5_train_config`` exactly
        # so the remote retrain and the local retrain build the SAME network.
        mode = str(getattr(m, "map_head_mode", "linear"))
        if mode != "linear":
            flags += ["--map-decoder", mode]
        if getattr(m, "map_prior_residual", False):
            flags.append("--map-prior-residual")
        w = float(getattr(m, "map_spectral_weight", 0.0))
        if w > 0.0:
            flags += ["--map-spectral-weight", str(w)]
        return flags

    def _fit_cell_calibrations(self, out_dir) -> None:
        """Fit the per-cell cyclen + F_r calibrations into a finished model dir.

        Used by the paths that do NOT go through ``train_ensemble`` (the local
        fine-tune).  Never raises — a missing calibration must not lose a retrain.
        """
        if not getattr(self.cfg.model, "auto_fit_cell_calibration", True):
            return
        try:
            from .model.train import fit_cell_calibrations
            fit_cell_calibrations(
                out_dir, store_dir=self.store_dir, splits_dir=self.splits_dir,
                split=self.curr.retrain_split, cfg=self._v5_train_config())
        except Exception as exc:      # noqa: BLE001
            self._log(f"[curriculum] per-cell calibration refit failed: {exc}")

    def _retrain_local_full(self, cid: str) -> str:
        from .model.train import train_ensemble
        # curriculum-safe split (never ejects a non-121 band; per-cell holdout)
        self._write_curriculum_split()
        out = self.state_dir / "models" / f"{cid}_{time.strftime('%Y%m%d_%H%M%S')}"
        train_ensemble(self.curr.retrain_ensemble, split=self.curr.retrain_split,
                       device="cpu", out_dir=out, store_dir=self.store_dir,
                       splits_dir=self.splits_dir,
                       cond_schema=self.cfg.model.cond_schema,
                       censor_dataset_a_pin_labels=
                       self.cfg.model.censor_dataset_a_pin_labels,
                       config=self._v5_train_config())
        return str(out)

    def _retrain_remote_full(self, cid: str) -> str:
        from . import remote as remote_mod
        # curriculum-safe split written to the data/splits manifest the remote
        # trainer consumes (unchanged filename/format), then shipped by push().
        self._write_curriculum_split()
        s = remote_mod.RemoteSettings(**dataclasses.asdict(self.cfg.remote))
        self._log(f"[curriculum][{cid}] remote push -> {s.host}")
        remote_mod.push(s, install=True)   # ships the optimized train.py source
        # --parallel-members activates member-parallel training (batch-1024 +
        # device-resident auto-activate on cuda); ~50min -> ~10-14min retrain.
        # thread [model] censor_dataset_a_pin_labels as the explicit CLI flag so
        # the remote trainer's behavior is deterministic regardless of its default.
        censor_flag = ("--censor-a-pin-labels"
                       if self.cfg.model.censor_dataset_a_pin_labels
                       else "--no-censor-a-pin-labels")
        # The v5 knobs (physics prior / quantile heads / promoted asm-BU / auto
        # calibration) are threaded as explicit flags for the same reason
        # ``censor_flag`` is: the remote trainer's behaviour must be a function of
        # the deck, not of whatever the remote's defaults happen to be.
        tr = remote_mod.train(s, ["--ensemble", str(self.curr.retrain_ensemble),
                                  "--parallel-members", str(self.curr.retrain_ensemble),
                                  "--split", self.curr.retrain_split,
                                  "--cond-schema", str(self.cfg.model.cond_schema),
                                  censor_flag,
                                  *self._v5_train_flags(),
                                  "--device", "auto", "--num-workers", "8"])
        ts = tr["ts"]
        self._log(f"[curriculum][{cid}] remote train launched ts={ts} gpu={tr.get('gpu')}")
        poll = max(15, int(self.curr.remote_poll_s))
        while True:
            st = remote_mod.status(s, ts)
            if st["state"] in ("done", "failed"):
                break
            time.sleep(poll)
        if st["state"] != "done":
            raise RuntimeError(f"remote train {ts} state={st['state']}")
        pl = remote_mod.pull(s, ts)
        return str(pl["dest"])

    def _roundtrip_check(self, model_dir: str, cid: str) -> dict:
        try:
            model = self.load_model(model_dir)
            cell = self.state["cells"][cid]
            pairs = cell.get("pairs") or ["K1_K2"]
            rng = random.Random(7)
            cands = _gen_candidates(pairs, cell["feed"], 4, rng)
            pats = [p for _pair, p in cands]
            cases = [CaseKey(pr, cell["feed"]) for pr, _p in cands]
            pred = model.predict(pats, cases)
            finite = bool(np.isfinite(pred.mean[:, [0, 3]]).all())
            return {"finite": finite,
                    "n_targets": len(getattr(model, "target_names", []) or []),
                    "shape": list(pred.mean.shape)}
        except Exception as exc:  # noqa: BLE001
            return {"finite": False, "error": str(exc)}

    # ---- phase 5: validate_gate ----------------------------------------- #
    def _phase_validate_gate(self, cid: str) -> str:
        cell = self.state["cells"][cid]
        cdir = self.cell_dir(cid)
        new_dir = self.state["champion_model_dir"]
        prev_dir = cell.get("champion_before")

        gate: dict[str, Any] = {"cell": cid, "model_dir": str(new_dir)}
        # (a) new-cell holdout: reuse blind-probe chains as the eval set
        newcell = self._gate_newcell(cid, new_dir)
        gate["new_cell"] = newcell
        # (b) no-regression on previous done cells
        noreg = self._gate_no_regression(cid, prev_dir, new_dir)
        gate["no_regression"] = noreg
        # (b2) legacy-corpus high-cyclen tail no-regression (forensic 20260719):
        # the per-cell gate + global zMAE are tail-insensitive, so a collapse on the
        # 700-720 EFPD Dataset-A tail escapes them; score both champions there.
        tail = self._gate_legacy_tail(prev_dir, new_dir)
        gate["legacy_tail"] = tail
        # (c) transfer curve artifact
        self._update_transfer_curve(cid, newcell)
        gate["transfer_curve"] = str(self.state_dir / "transfer_curve.json")
        # (d) mini user_criteria spot campaign (advisory)
        mini = None
        if self.curr.gate_mini_campaign and not self.dry_run:
            try:
                mini = self._gate_mini_campaign(cid, new_dir, cdir)
            except Exception as exc:  # noqa: BLE001
                mini = {"status": "error", "error": str(exc)}
        elif self.curr.gate_mini_campaign and self.dry_run:
            mini = self._gate_mini_campaign(cid, new_dir, cdir)
        gate["mini_campaign"] = mini

        passed = (bool(newcell.get("pass")) and bool(noreg.get("pass"))
                  and bool(tail.get("pass", True)))
        gate["pass"] = passed
        _atomic_write_json(cdir / "gate.json", gate)
        cell["gate"] = gate
        # A verdict line that stops at PASS/FAIL reads as "the whole guarded
        # family was checked".  Axes that were scored-but-not-enforced (``f_r`` by
        # default) or could not be judged at all belong on the SAME transcript, or
        # the operator learns of them only by opening gate.json.  Emitted on BOTH
        # outcomes: a FAIL that hides the deferral is the worse case of the two —
        # it invites "so F_r was checked and something else broke".
        for _label, _block in (("new-cell", newcell), ("no-regression", noreg)):
            if _block.get("note"):
                self._log(f"[curriculum][{cid}] {_label}: {_block['note']}")
        if passed:
            cell["done_at"] = _now()
            # A PASS is exactly the moment an operator stops reading, so the
            # coverage caveat rides on the PASS line itself and not only in the
            # note above it.  Warn, never block (user decision 2026-07-26).
            _blind = list(noreg.get("blind_targets") or [])
            _cov = f", GUARDS NOT MEASURED: {', '.join(_blind)}" if _blind else ""
            self._log(f"[curriculum][{cid}] validate_gate PASS "
                      f"(new-cell skill={newcell.get('mean_spearman')}, "
                      f"no-regression={noreg.get('pass')}"
                      f"{_cov})")
            return "advance"  # _advance_phase moves validate_gate -> done
        # the retrain phase bumped the champion pointer before the gate ran;
        # a rejected model must not stay operative
        if prev_dir and str(prev_dir) != str(new_dir):
            self.state["champion_model_dir"] = str(prev_dir)
            self._log(f"[curriculum][{cid}] champion reverted -> {prev_dir}")
        self._log(f"[curriculum][{cid}] validate_gate FAIL: "
                  f"new_cell={newcell.get('pass')} no_regression={noreg.get('pass')} "
                  f"legacy_tail={tail.get('pass')} -> halt for user decision")
        return "fail"

    def _gate_newcell(self, cid: str, model_dir: str) -> dict:
        """Post-train performance on the new cell's eval holdout — the blind-probe
        chains (a fixed set production never generates, so genuinely out-of-sample;
        their ids are pinned into the split's per-cell val by construction): per-
        target MAE + Spearman vs the recorded live-MASTER actuals.  The scored
        ``record_id``s are persisted for auditability.

        F_r DEFERRAL (user decision 2026-07-26).  This gate ANDs into the SAME
        ``validate_gate`` verdict as the no-regression gate, so an ``f_r`` vote in
        its mean-Spearman is an F_r VETO on the retrain by the back door — the
        exact thing the deferral removed one function over.  ``f_r`` is therefore
        scored, reported per-target and named in ``note``, but excluded from the
        thresholded mean, through the SAME switch
        (``[curriculum] gate_noreg_fr_guard_enabled``) that re-arms every other
        surface.  This is a different mechanism from ``gate_advisory_targets``
        (a permanent forensic verdict about ``max_pin_burnup``); the F_r demotion
        is a dormant axis with a documented activation checklist, so the two are
        reported separately even though both are excluded from the mean.
        """
        from .data.schema import unpack_pattern, compute_record_id
        from .search.verify import PRODUCE_DECK_KNOBS
        fr_guarded = fr_guard_enforced(curriculum=self.curr)
        report_only = () if fr_guarded else (FR_GUARD_TARGET,)
        advisory = {str(t) for t in getattr(self.curr, "gate_advisory_targets", ())}
        # What actually votes in the thresholded mean: everything that is neither
        # permanently advisory nor deferred by the F_r switch.
        guarded = tuple(n for n, _c, _f in PROBE_TARGETS
                        if n not in report_only and n not in advisory)
        _base = {"guarded_targets": list(guarded),
                 "report_only_targets": list(report_only),
                 "scored_targets": [n for n, _c, _f in PROBE_TARGETS],
                 "fr_guard": fr_guard_block(fr_guarded=fr_guarded)}
        probe_path = self.cell_dir(cid) / "blind_probe.json"
        if not probe_path.exists():
            return {"pass": True, "note": "no probe file", "mean_spearman": None,
                    **_base}
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        conv = [r for r in probe["candidates"] if r["status"] == "converged"]
        if not conv:
            return {"pass": True, "note": "no converged probe chains",
                    "mean_spearman": None, **_base}
        model = self.load_model(model_dir)
        pats = [unpack_pattern(r["pattern"]) for r in conv]
        cases = [CaseKey(r["pair"], self.state["cells"][cid]["feed"]) for r in conv]
        _lib = self.cfg.model.library_id or self.curr.library
        scored_ids = [str(r.get("record_id")
                          or compute_record_id(str(r["pattern"]), _lib, str(r["pair"]),
                                               PRODUCE_DECK_KNOBS))
                      for r in conv]
        pred = model.predict(pats, cases)
        per_target = {}
        sp_vals = []
        for name, col, _f in PROBE_TARGETS:
            preds, acts = [], []
            for i, r in enumerate(conv):
                av = r["actual"].get(name)
                pv = _finite(pred.mean[i, col])
                if av is not None and pv is not None:
                    preds.append(pv)
                    acts.append(av)
            if not preds:
                per_target[name] = {"n": 0, "advisory": name in advisory,
                                    "report_only": name in report_only}
                continue
            err = np.array(preds) - np.array(acts)
            sp = _spearman(acts, preds)
            is_advisory = name in advisory
            is_report_only = name in report_only
            # Advisory targets (e.g. max_pin_burnup) are REPORTED but excluded from
            # the gate's mean-Spearman: their within-cell OOS rank skill is ~0 and,
            # at n~11, noise-dominated — see data/reports/pinbu_forensics.md.
            # ``f_r`` is excluded for the DIFFERENT reason recorded at
            # NOREG_ENFORCED_DEFAULT — the deferral — and is flagged separately.
            if sp is not None and not is_advisory and not is_report_only:
                sp_vals.append(sp)
            blind = probe["per_target"].get(name, {})
            per_target[name] = {
                "n": len(preds), "mae": float(np.mean(np.abs(err))),
                "spearman": sp, "blind_mae": blind.get("mae"),
                "improvement": (blind.get("mae") - float(np.mean(np.abs(err))))
                if blind.get("mae") is not None else None,
                "advisory": is_advisory,
                "report_only": is_report_only,
            }
        mean_sp = float(np.mean(sp_vals)) if sp_vals else None
        passed = (mean_sp is not None and mean_sp >= self.curr.gate_new_cell_min_spearman)
        out = {"pass": bool(passed), "mean_spearman": mean_sp,
               "per_target": per_target, "n_eval": len(conv),
               "scored_record_ids": scored_ids, **_base}
        ro_scored = [n for n in report_only
                     if per_target.get(n, {}).get("spearman") is not None]
        if ro_scored:
            worst = min(float(per_target[n]["spearman"]) for n in ro_scored)
            out["note"] = fr_report_only_note(
                ro_scored,
                measured=f"lowest report-only spearman {worst:.4f}; gate bar "
                         f"{float(self.curr.gate_new_cell_min_spearman):.4f} applies "
                         f"to the mean of {', '.join(guarded)}",
                verb="block this cell")
        return out

    def _gate_no_regression(self, cid: str, prev_dir: str | None, new_dir: str) -> dict:
        """HONEST no-regression: for every previously-done cell, score BOTH the
        previous champion and the candidate LIVE on the SAME per-cell eval holdout
        — the split's curriculum val rows for that cell — and require the
        candidate's within-case Spearman not to drop by more than
        ``gate_noreg_epsilon`` on the primaries.

        These val rows were held out of BOTH trainings (the split's per-cell
        holdout is stable-hash invariant to adding new cells), so the comparison
        is out-of-sample vs out-of-sample — no ``head(200)``-of-raw-store rows that
        were 100% in the champion's train set (the phantom-drop contamination the
        forensic audit isolated).  The scored ``record_id``s are persisted for
        auditability.
        """
        done_cells = [c for c in self.state["order"]
                      if c != cid and self.state["cells"].get(c, {}).get("phase") == "done"]
        fr_guarded = bool(getattr(self.curr, "gate_noreg_fr_guard_enabled", False))
        if not done_cells or prev_dir is None:
            # DELEGATE, do not re-hand-roll.  The module gate's own trivial path
            # already carries the WHOLE contract (``guarded_targets``,
            # ``report_only_targets``, ``unavailable``, ``fr_guard``, …), and this
            # driver-side early return used to return a four-key dict instead —
            # so on the very first cell a consumer reading ``fr_guard`` off the
            # gate result hit a KeyError, and the artifact carried no record that
            # the F_r axis had been deferred at all.  Passing ``None`` models is
            # exactly the trivial branch, so there is one shape and one place.
            return gate_no_regression(
                None, None, None, {}, [],
                epsilon=self.curr.gate_noreg_epsilon, fr_guarded=fr_guarded)
        from .data.store import StoreReader
        df = StoreReader(self.store_dir).records
        manifest = self._curriculum_split_manifest(records=df)
        val_by_cell = manifest.groups.get("curriculum_val_by_cell", {})
        old_model = self.load_model(prev_dir)
        new_model = self.load_model(new_dir)
        # Delegate to the module-level reusable gate (shared with ``gate-promote``).
        return gate_no_regression(
            old_model, new_model, df, val_by_cell, done_cells,
            epsilon=self.curr.gate_noreg_epsilon, fr_guarded=fr_guarded)

    def _gate_legacy_tail(self, prev_dir: str | None, new_dir: str) -> dict:
        """Legacy-corpus high-cyclen tail no-regression (forensic 20260719).

        The honest per-cell gate scores only ga80 curriculum cells and the global
        val zMAE-cyclen is tail-insensitive, so a collapse concentrated in the
        high-cyclen Dataset-A tail (the 700-720 EFPD band, entirely the 5.8_5.1
        library) slipped through every gate.  This scores BOTH champions on a fixed
        stable-hash sample of S1-val Dataset-A rows per band via the parity-correct
        :meth:`PosValCnnBackend.predict_rows_raw`, failing when the candidate's
        cyclen MAE degrades by more than ``[curriculum] gate_tail_epsilon`` EFPD.
        """
        if not getattr(self.curr, "gate_tail_enabled", True) or prev_dir is None:
            return {"pass": True, "note": "disabled or no previous champion"}
        old_model = self.load_model(prev_dir)
        new_model = self.load_model(new_dir)
        from .data.store import StoreReader
        df = StoreReader(self.store_dir).records
        # Delegate to the module-level reusable gate (shared with ``gate-promote``).
        res = gate_legacy_tail(
            old_model, new_model, df,
            bands=self.curr.gate_tail_bands, feed=self.curr.gate_tail_feed,
            sample_per_band=self.curr.gate_tail_sample,
            seed=self.cfg.flow.random_seed, epsilon=self.curr.gate_tail_epsilon,
            enabled=getattr(self.curr, "gate_tail_enabled", True))
        if res.get("note"):
            return res
        self._log(f"[curriculum] legacy-tail gate: pass={res['pass']} "
                  f"worst_mae_increase={res['worst_mae_increase']:.2f} EFPD "
                  f"(eps={res['epsilon']})")
        return res

    def _update_transfer_curve(self, cid: str, newcell: dict) -> None:
        path = self.state_dir / "transfer_curve.json"
        data = {"cells": []}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                data = {"cells": []}
        cell = self.state["cells"][cid]
        probe = cell.get("blind_probe", {}).get("per_target", {})
        entry = {
            "cell": cid, "ring": cell.get("ring"), "feed": cell["feed"],
            "band": cell["band"],
            "blind_mae": {k: v.get("mae") for k, v in probe.items()},
            "blind_spearman": {k: v.get("spearman") for k, v in probe.items()},
            "post_mae": {k: v.get("mae") for k, v in (newcell.get("per_target") or {}).items()},
            "post_mean_spearman": newcell.get("mean_spearman"),
        }
        data["cells"] = [c for c in data.get("cells", []) if c.get("cell") != cid]
        data["cells"].append(entry)
        data["cells"].sort(key=lambda c: (c.get("ring", 0), c.get("cell", "")))
        _atomic_write_json(path, data)
        self._render_transfer_png(data, self.state_dir / "transfer_curve.png")

    def _render_transfer_png(self, data: dict, png_path: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return
        cells = data.get("cells", [])
        if not cells:
            return
        rings = [c.get("ring", 0) for c in cells]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        n_lines = 0
        for name, _c, _f in PROBE_TARGETS:
            ys = [(_finite((c.get("blind_mae") or {}).get(name))) for c in cells]
            xs = [r for r, y in zip(rings, ys) if y is not None]
            yv = [y for y in ys if y is not None]
            if yv:
                ax.plot(xs, yv, marker="o", label=name)
                n_lines += 1
        ax.set_xlabel("cell ring distance from anchor")
        ax.set_ylabel("blind transfer MAE (pred vs live MASTER)")
        ax.set_title("Curriculum blind transfer error vs cell distance")
        if n_lines:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        try:
            fig.tight_layout()
            fig.savefig(png_path, dpi=110)
        finally:
            plt.close(fig)

    #: What the mini campaign is asked to demonstrate, in one sentence, stamped
    #: into every ``mini_campaign.json`` so the artifact says what it means.
    MINI_DEMONSTRATES = (
        "the promoted champion can STEER a flat_power campaign: its own ranking "
        "of a fresh cell pool by the node_peak (primary) + map_cov (secondary) "
        "objective, verified live, produces measurably flatter cores than the "
        "cell's blind probe. F_r is a licensing SAFETY GATE on the candidates, "
        "not a target and not a ranking key."
    )

    def _gate_mini_campaign(self, cid: str, model_dir: str, cdir: Path) -> dict:
        """Budget-``gate_mini_budget`` live spot campaign on the FLATNESS objective.

        WHAT THIS DEMONSTRATES (changed 2026-07-26; see :data:`MINI_DEMONSTRATES`).
        This is the only part of ``validate_gate`` that spends REAL MASTER calls —
        ``gate_mini_budget`` (16) chains per cell — so what it spends them on is a
        statement about what a promoted champion is expected to be good at.  It
        used to rank the pool with ``score_user_criteria`` at
        ``f_r_limit = gate_min_f_r = 1.55`` (a knob RETIRED with that move — see
        :data:`..config.RETIRED_KEYS`) and count "feasible" rows against that
        same 1.55, i.e. it spent every one of those calls hunting an F_r target the
        corpus cannot supply: measured 2026-07-26, not one of the 1,494 converged
        rows in the 36 done cells' val holdouts is below 1.55 (lowest 1.5974).  A
        budget spent proving an unreachable target proves nothing about the model.

        Under the flatness-first program (§1.2/§10) the champion will steer a
        ``flat_power`` campaign, so the mini campaign now ranks with the SAME
        apparatus that campaign uses — :func:`..search.acquisition.score_flat_power`
        over the model's map head — and F_r keeps exactly the role it has there: a
        BINARY SAFETY VETO at the D1 in-loop gate
        (:attr:`..search.acquisition.FlatPowerSpec.fr_limit`, 1.70, never loosened),
        applied to candidates before they can be picked.  The D2 licensing limit
        1.55 survives as a REPORTED per-row margin, which is what a licensing
        constant is: a compliance column, not an objective.

        A model with no map head cannot demonstrate any of this, so it reports
        ``status = "no_map_head"`` and spends ZERO MASTER calls rather than
        burning 16 chains on an all-``-inf`` ranking that is really a random draw.
        """
        from .search import acquisition as acq
        from .data.flat_scale import FlatScale
        from .data.flatness import record_flatness
        cell = self.state["cells"][cid]
        band, feed, pairs = cell["band"], cell["feed"], cell["pairs"]
        budget = int(self.curr.gate_mini_budget)

        # cyclen stays RECORD-ONLY here, exactly as in flat_power: reported next
        # to every verified row, never ranked on, never a feasibility criterion.
        probe = json.loads((cdir / "blind_probe.json").read_text(encoding="utf-8"))
        conv = [r for r in probe["candidates"] if r["status"] == "converged"]
        cy_actual = [r["actual"].get("cyclen") for r in conv if r["actual"].get("cyclen")]
        target_cyclen = self.curr.gate_cyclen_target
        if target_cyclen is None:
            target_cyclen = float(np.median(cy_actual)) if cy_actual else 625.0

        model = self.load_model(model_dir)
        rng = random.Random(self.cfg.flow.random_seed + 4242)
        pool = _gen_candidates(pairs, feed, max(budget * 40, 400), rng)
        pats = [p for _pair, p in pool]
        cases = [CaseKey(pr, int(feed)) for pr, _p in pool]

        flat, flat_reason = _map_head_flatness_ucb(model, pats, cases)
        if flat is None:
            out = {"status": "no_map_head", "objective": "flat_power",
                   "demonstrates": self.MINI_DEMONSTRATES,
                   "reason": flat_reason, "budget": budget, "master_calls": 0,
                   "results": []}
            _atomic_write_json(cdir / "mini_campaign.json", out)
            self._log(f"[curriculum][{cid}] mini campaign SKIPPED (0 MASTER "
                      f"calls): {flat_reason} — the flatness objective cannot be "
                      "demonstrated by a model with no map head")
            return out

        pred = model.predict(pats, cases)
        pk_m, pk_s, cv_m, cv_s = flat
        # Same normalizers the campaign would use for this cell (module defaults
        # when the artifact has not fitted it — reported either way).  No bias
        # correction is applied: ``flatpower_fr_gate`` may only TIGHTEN the safety
        # gate, so without a fitted correction it HOLDS at the unmodified 1.70.
        fscale = FlatScale.from_store(self.cfg.model.store_dir)
        peak_scale, cov_scale = fscale.scales_for(cid)
        spec = acq.FlatPowerSpec(
            peak_scale=peak_scale, cov_scale=cov_scale,
            cbc_limit=self.cfg.acquisition.cbc_limit,
            f_q_limit=self.cfg.acquisition.f_q_limit,
            ao_abs_limit=self.cfg.acquisition.ao_abs_limit)
        score = acq.score_flat_power(pred, pk_m, pk_s, spec,
                                     cov_mean=cv_m, cov_std=cv_s)
        fr_gate = float(acq.flatpower_fr_gate(spec))
        order = np.argsort(-np.asarray(score.total))
        picks, seen = [], set()
        for i in order:
            key = f"{pool[i][0]}|{pool[i][1].canonical()}"
            if key in seen:
                continue
            seen.add(key)
            picks.append(int(i))
            if len(picks) >= budget:
                break
        pick_cands = [pool[i] for i in picks]

        lib = self.fuel_library()
        cell_lib = cell.get("library_id") or self._band_library(band)
        resolver = _build_resolver(self.cfg, lib, cell_lib)
        verifier = self.verifier(_short_run_root(self.cfg, cid, "m"), resolver)
        entries = self._wave_entries(pick_cands, feed, resolver, lib, cell_lib)
        outcomes = verifier.evaluate_wave(entries)

        results = []
        n_fr_safe = 0
        n_fr_violations = 0
        best_peak = None
        # Counters that separate the innocent causes of a null measured flatness.
        # Without them the report can only say "no readable map", which reads as a
        # physics/parse problem even when the real cause is that the verifier was
        # never wired to harvest one — the exact misattribution this gate produced
        # for every cell before 2026-07-26.
        # ``n_converged`` counts CONVERGENCE and nothing else.  It used to be
        # incremented only for picks that ALSO carried a FOM, so a wave that
        # converged everything and lost every FOM reported ``n_converged == 0``
        # and the cause ``no_convergence`` — blaming the physics for a harness
        # fault, the same dishonest-cause reporting the four-way taxonomy below
        # exists to end.  The FOM is now its own counter and its own cause.
        n_converged = 0
        n_with_fom = 0
        n_with_maps = 0
        n_map_scored = 0
        for j, (pair, pat) in enumerate(pick_cands):
            i = picks[j]
            oc = outcomes[j]
            row = {"pair": pair, "pattern": pat.canonical(), "status": oc.status,
                   "pred_node_peak": _finite(pk_m[i]),
                   "pred_map_cov": _finite(cv_m[i]),
                   "pred_fr_ucb": _finite(score.fr_ucb[i]),
                   "fr_gate_violated": bool(score.fr_gate_violated[i])}
            n_converged += int(oc.status == "converged")
            if oc.status == "converged" and oc.fom is not None:
                fom = oc.fom
                fr = _finite(fom.f_r)
                n_with_fom += 1
                # Measured flatness from the candidate's OWN harvested map, via
                # the one canonical definition the store labels with.
                raw_maps = getattr(oc, "maps", None)
                n_with_maps += int(raw_maps is not None)
                peak, cov = record_flatness(raw_maps)
                n_map_scored += int(peak is not None)
                safe = fr is not None and fr <= fr_gate
                n_fr_safe += int(safe)
                n_fr_violations += int(fr is not None and not safe)
                if peak is not None and (best_peak is None or peak < best_peak):
                    best_peak = float(peak)
                row.update({
                    "node_peak": peak, "map_cov": cov,
                    "f_r": fr, "fr_safe": safe,
                    # D2 licensing MARGIN — a compliance column, not a criterion.
                    "fr_margin": (None if fr is None
                                  else float(FR_GUARD_LICENSING_LIMIT - fr)),
                    "cyclen": _finite(fom.cyclen),        # record-only
                    "cbc_max": _finite(fom.cbc_max), "f_q": _finite(fom.f_q),
                    "ao_abs": _finite(fom.ao_abs),
                })
            results.append(row)

        cell["budget"]["master_calls"] += len(pick_cands)
        # Flatness baseline: the blind probe's own labelled peak when the probe
        # recorded one.  Older probes predate the node_peak label, so this is
        # ``None`` rather than a fabricated comparison, and ``flatness_progress``
        # is then ``None`` (unknown), never a default True.
        blind_peaks = [r["actual"].get("node_peak") for r in conv
                       if r["actual"].get("node_peak") is not None]
        baseline_peak = float(min(blind_peaks)) if blind_peaks else None
        if best_peak is None or baseline_peak is None:
            progress = None
        else:
            progress = bool(best_peak <= baseline_peak + 1e-9)
        pred_peaks = [r["pred_node_peak"] for r in results
                      if r["pred_node_peak"] is not None]
        # A null ``best_node_peak`` has FIVE distinguishable causes, and the report
        # must name the one that actually fired.  Collapsing them (as the previous
        # two-branch text did) makes a harness fault read as a physics result:
        # "all nonconverged, or the EDIT5 parse came back odd" was printed for
        # cells whose every pick converged and whose verifier simply never
        # harvested anything.  Each cause implies a different action, so each gets
        # its own machine-readable tag and its own sentence.  The fifth,
        # ``no_fom``, was hiding INSIDE ``no_convergence`` until the converged
        # count stopped requiring a FOM: converged-then-dropped is a harness
        # fault, never-converged is physics, and they call for opposite actions.
        harvest = bool(getattr(self.cfg.verify, "harvest_maps", False))
        measured_note = None
        measured_cause = None
        if best_peak is None:
            if not harvest:
                measured_cause = "harvest_disabled"
                measured_note = (
                    "measured flatness unavailable: [verify] harvest_maps is "
                    "false, so these MASTER calls kept no map to score")
            elif n_converged == 0:
                measured_cause = "no_convergence"
                measured_note = (
                    f"measured flatness unavailable: none of the {len(pick_cands)} "
                    "picked candidates converged, so there was no core to score")
            elif n_with_fom == 0:
                measured_cause = "no_fom"
                measured_note = (
                    f"measured flatness unavailable: {n_converged} of "
                    f"{len(pick_cands)} picks CONVERGED but none carried a FOM, "
                    "so nothing downstream of the equilibrium solve was read — "
                    "this is a HARNESS fault (the verifier converged and then "
                    "dropped the result), not a convergence or physics failure")
            elif n_with_maps == 0:
                measured_cause = "no_maps_harvested"
                measured_note = (
                    f"measured flatness unavailable: {n_converged} of "
                    f"{len(pick_cands)} picks CONVERGED but the verifier handed "
                    "back no harvested map for any of them — [verify] "
                    "harvest_maps is true, so this is a HARNESS fault (the "
                    "evaluator must be a HarvestingEquilibriumEvaluator and the "
                    "converged work dir must survive), not a physics or parse "
                    "result")
            else:
                measured_cause = "unreadable_maps"
                measured_note = (
                    f"measured flatness unavailable: {n_with_maps} harvested map"
                    f"{'s' if n_with_maps != 1 else ''} came back but none scored "
                    "a finite node_peak — the EDIT5 stack parsed odd")
        elif harvest and n_map_scored < n_converged:
            # Partial measurement is not a null, but it is not "all measured"
            # either, and a silent partial is how an average drifts.
            measured_cause = "partial"
            measured_note = (
                f"measured flatness is PARTIAL: {n_map_scored} of {n_converged} "
                "converged picks scored a map; the rest carried none"
                + ("" if n_with_fom == n_converged else
                   f" ({n_converged - n_with_fom} of them carried no FOM at all)"))
        out = {
            "status": "ok",
            "objective": "flat_power",
            "demonstrates": self.MINI_DEMONSTRATES,
            "harvest_maps": harvest,
            "measured_flatness_note": measured_note,
            "measured_flatness_cause": measured_cause,
            "n_picks_converged": n_converged,
            "n_picks_with_fom": n_with_fom,
            "n_picks_with_maps": n_with_maps,
            "n_picks_map_scored": n_map_scored,
            "fr_role": "safety_gate",
            "fr_safety_gate": fr_gate,
            "licensing_limit": FR_GUARD_LICENSING_LIMIT,
            "budget": budget,
            "scales": fscale.describe(cid),
            "n_fr_safe": n_fr_safe,
            "n_fr_violations": n_fr_violations,
            "best_node_peak": best_peak,
            "best_pred_node_peak": (min(pred_peaks) if pred_peaks else None),
            "baseline_node_peak": baseline_peak,
            "flatness_progress": progress,
            "target_cyclen": target_cyclen,        # reported, ranked on by nothing
            "results": results,
        }
        _atomic_write_json(cdir / "mini_campaign.json", out)
        self._log(f"[curriculum][{cid}] mini campaign (flat_power): best "
                  f"node_peak={best_peak} vs blind baseline={baseline_peak} "
                  f"(progress={progress}); F_r safety gate {fr_gate:.3f}: "
                  f"{n_fr_safe} safe / {n_fr_violations} violations")
        if measured_note:
            self._log(f"[curriculum][{cid}] {measured_cause}: {measured_note}")
        return out


# --------------------------------------------------------------------------- #
# detached produce entry (invoked as `lpopt curriculum-produce`)
# --------------------------------------------------------------------------- #
def _pid_alive(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=15,
                                 **no_window_flags())
            return str(pid) in (out.stdout or "")
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _stratum_for_cell(cfg: LpoptConfig, cid: str, cell: dict) -> StratumConfig:
    feed = int(cell["feed"])
    n_fresh = fresh_units_from_feed(feed)
    return StratumConfig(
        name=cid,
        library=cell.get("library_id", cfg.curriculum.library),
        pairs=list(cell["pairs"]),
        feed=feed,
        split_w1=list(cfg.curriculum.split_w1),
        generators=dict(cfg.curriculum.generators),
        n_target=int(cell.get("n_target_override") or cfg.curriculum.n_target),
        priority=100,
        allow_single_cycle_discharge=(n_fresh > 30),
        max_shuffle_depth=2,
        notes=f"curriculum cell {cid}",
    )


def run_cell_produce(
    cfg: LpoptConfig,
    cid: str,
    cell: dict,
    store_dir: Path,
    *,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
    fuel_library: Any = None,
    run_dir: Path | None = None,
) -> dict:
    """Run the cell's stratified production with pin-burnup labels ON.

    Builds a ProduceDriver whose verifier enables ``enable_pin_burnup`` (the
    default produce verifier does not), tagging labels ``dataset='P'``,
    ``campaign=<cid>`` into the main store.
    """
    from .search.produce import ProduceDriver
    log = log or (lambda m: print(m))

    # tag the produce config for this cell
    cfg.produce.campaign = cid
    cfg.produce.strata = [_stratum_for_cell(cfg, cid, cell)]
    cfg.produce.store_dir = str(store_dir)
    cfg.produce.resume = True

    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    if run_dir is None:
        # short MASTER work root (Windows MAX_PATH guard, see _short_run_root)
        run_dir = _short_run_root(cfg, cid, "q")
    run_dir.mkdir(parents=True, exist_ok=True)

    # per-cell library (paramA for high bands) so the resolver's fuel lookups and
    # asset resolution match the stratum library _stratum_for_cell tags.
    resolver = _build_resolver(cfg, fuel_library,
                               cell.get("library_id") or cfg.curriculum.library)
    verifier = make_pin_burnup_verifier(cfg, run_dir, resolver, dry_run=dry_run)
    driver = ProduceDriver(
        cfg, dry_run=dry_run, run_dir=run_dir,
        store_dir=(run_dir / "store" if dry_run else store_dir),
        verifier=verifier, resolver=resolver, fuel_library=fuel_library,
        progress=True, log=log,
    )
    summary = driver.run()
    return {"chains": summary.chains, "converged": summary.converged,
            "nonconverged": summary.nonconverged, "errors": summary.errors,
            "duplicates": summary.duplicates}


def cmd_curriculum_produce(args) -> int:
    """Hidden CLI entry: run one cell's production (spawned detached by the driver)."""
    cfg = load_config(Path(args.input))
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    state_dir = Path(cfg.curriculum.state_dir)
    state_dir = state_dir if state_dir.is_absolute() else (base / state_dir)
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    cid = args.cell
    cell = state["cells"][cid]
    store_dir = Path(cfg.model.store_dir)
    store_dir = store_dir if store_dir.is_absolute() else (base / store_dir)

    from .data.fuel_types import FuelLibrary
    fpath = store_dir / "fuel_types.parquet"
    fuel = FuelLibrary.from_parquet(fpath) if fpath.exists() else FuelLibrary.build(cfg, persist=False)

    cdir = state_dir / "cells" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    try:
        summ = run_cell_produce(cfg, cid, cell, store_dir, dry_run=False, fuel_library=fuel)
        (cdir / "produce.done").write_text(
            json.dumps(summ, indent=2), encoding="utf-8")
        print(f"[curriculum-produce] cell {cid} done: {summ}")
        return 0
    except Exception as exc:  # noqa: BLE001
        (cdir / "produce.error").write_text(str(exc), encoding="utf-8")
        print(f"[curriculum-produce] cell {cid} FAILED: {exc}")
        return 1


def run_curriculum(
    cfg: LpoptConfig,
    *,
    dry_run: bool = False,
    max_cells: int | None = None,
    resume: bool = True,
    **kwargs: Any,
) -> dict:
    """Convenience entry mirroring run_produce/run_campaign."""
    driver = CurriculumDriver(cfg, dry_run=dry_run, **kwargs)
    return driver.run(max_cells=max_cells, resume=resume)
