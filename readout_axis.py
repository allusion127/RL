"""Objective-aware axis resolution for the top-level λ-objective / frontier readouts.

Design reference: ``data/reports/fxy_switch_design_20260829.md`` §3.5.5 —

> **λ-목적 검사 의무 이관.**  등록된 규칙 "λ-목적 검사는 모든 프런티어 판독에
> 의무"는 그대로 유효하되 **판정 축이 F_xy로 바뀐다.**  프런티어 판독은
> ``min(F_xy)`` 단독이 아니라 밴드 내 ``cyclen_LCB − λ_fxy·F_xy``로 읽는다.
> ``anchor_readout.py`` / ``autoeng.py`` 등 판독 스크립트의 목적 계산을 F_xy로
> 교체한다.

Every top-level readout used to hard-code F_r: the store column ``f_r``, the
licensing limit ``1.55``, the headline word ``F_r``.  This module makes those
three a single resolved object so the SAME readout can be read on either axis,
and so a headline can never name an axis it did not actually compute.

THE DEFAULT IS UNCHANGED.  With no ``--deck`` / ``--objective`` — and for every
``min_fr*`` deck — :func:`resolve_axis` returns :data:`F_R_AXIS`, whose ``label``
is the literal ``"F_r"`` and whose ``limit`` is the literal ``1.55`` the callers
used to inline.  Substituting ``axis.label`` / ``axis.limit`` into the existing
format strings therefore reproduces today's output BYTE FOR BYTE; only a
``min_fxy`` deck moves the axis.

THE MISSING-LABEL RULE.  F_xy is absent from ~92 % of the store (it is parsed out
of ``MAS_OUT``, not ``MAS_SUM``, so only harvested runs carry it — design §0.1).
A frontier readout must therefore NEVER silently rank a population whose axis is
mostly null: :func:`split_labelled` drops the unlabelled rows and returns their
count, and :func:`unlabelled_note` renders the count into the headline block.
Dropping them quietly would report a "frontier" of whichever 8 % happened to be
harvested; reporting the count is the honest form of the same table.  On the F_r
axis nothing is ever dropped in practice and the note is suppressed entirely,
which is what keeps the F_r output identical.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "Axis", "F_R_AXIS", "FXY_OBJECTIVES",
    "LICENSING_FR_LIMIT", "LICENSING_FXY_LIMIT",
    "resolve_axis", "deck_objective", "add_axis_args", "axis_from_args",
    "axis_values", "split_labelled", "unlabelled_note", "lambda_objective",
]

#: The two licensing limits.  Kept as literals rather than imported from
#: :mod:`lpopt.search.delivery` so a readout still runs in an environment where
#: only the store and pandas are installed; the values are asserted against
#: ``delivery.LICENSING_*`` by :func:`_check_against_lpopt` when lpopt imports.
LICENSING_FR_LIMIT = 1.55
LICENSING_FXY_LIMIT = 1.65

#: λ [EFPD per unit axis] for the scalarised objective ``cyclen − λ·axis``.
#: Both are 1000.0: ``minfr_lambda`` has always been, and design §3.5.1 carries it
#: over unchanged for ``minfxy_lambda`` because the measured within-cell F_xy
#: spread is the same order as F_r's.
FR_LAMBDA = 1000.0
FXY_LAMBDA = 1000.0

#: Deck objectives whose PRIMARY axis is F_xy.  ``flat_power`` is deliberately NOT
#: here: it carries an F_xy *gate* (design §3.5.3) but its objective is flatness,
#: so its frontier readout is not an F_xy frontier.
FXY_OBJECTIVES = frozenset({"min_fxy"})


@dataclass(frozen=True)
class Axis:
    """The one axis a readout is allowed to call its objective.

    ``key``       store / steps column name (``"f_r"`` | ``"f_xy"``)
    ``label``     the word a headline prints (``"F_r"`` | ``"F_xy"``)
    ``limit``     the hard licensing limit on that axis
    ``lam``       λ [EFPD per unit axis] of ``cyclen − λ·axis``
    ``objective`` the deck objective it was resolved from
    ``source``    where the resolution came from, for the provenance line
    """

    key: str
    label: str
    limit: float
    lam: float
    objective: str
    source: str = "default"

    @property
    def is_fxy(self) -> bool:
        return self.key == "f_xy"

    @property
    def gate(self) -> str:
        """``"F_xy <= 1.65"`` — the gate clause, for a headline."""
        return f"{self.label} <= {self.limit:g}"

    def provenance(self) -> str:
        """One line naming the axis AND where it came from.

        Every headline this module feeds must be accompanied by this, so a table
        can never be read as F_r when it was computed on F_xy or the reverse.
        """
        return (f"objective {self.objective} -> axis {self.label} "
                f"(store column '{self.key}', limit {self.limit:g}, "
                f"lambda {self.lam:g} EFPD/unit; from {self.source})")


#: The default axis.  Its three fields ARE the literals every readout inlined
#: before the switch, which is what makes the min_fr path byte-identical.
F_R_AXIS = Axis("f_r", "F_r", LICENSING_FR_LIMIT, FR_LAMBDA,
                "min_fr_max_cycle", "default")


def _axis_for(objective: str, limits: dict[str, Any] | None,
              source: str) -> Axis:
    limits = limits or {}
    if objective in FXY_OBJECTIVES:
        return Axis("f_xy", "F_xy",
                    float(limits.get("f_xy_limit", LICENSING_FXY_LIMIT)),
                    float(limits.get("minfxy_lambda", FXY_LAMBDA)),
                    objective, source)
    return Axis("f_r", "F_r",
                float(limits.get("f_r_limit", LICENSING_FR_LIMIT)),
                float(limits.get("minfr_lambda", FR_LAMBDA)),
                objective or F_R_AXIS.objective, source)


def deck_objective(deck: str | Path) -> tuple[str, dict[str, Any]]:
    """``(objective, [acquisition] table)`` read straight off a campaign deck.

    Parsed with :mod:`tomllib` rather than ``lpopt.config.load_config`` on
    purpose: a readout must be able to name the axis of a deck it did not itself
    validate (an archived deck, a generated per-cell deck, a deck whose model dir
    no longer exists), and the strict loader refuses those for reasons that have
    nothing to do with which axis the campaign optimised.
    """

    path = Path(deck)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    acq = dict(raw.get("acquisition", {}))
    objective = str(acq.get("objective", "")).strip()
    if not objective:
        raise ValueError(f"{path}: [acquisition] names no objective — cannot "
                         f"resolve the readout axis from this deck")
    return objective, acq


def resolve_axis(*, objective: str | None = None, deck: str | Path | None = None,
                 limits: dict[str, Any] | None = None,
                 source: str | None = None) -> Axis:
    """The axis this readout is entitled to call its objective.

    Precedence: an explicit ``objective`` wins (a caller that already knows),
    then ``deck``, then the F_r default.  ``limits`` overrides the deck's own
    ``f_r_limit`` / ``f_xy_limit`` / ``minfr_lambda`` / ``minfxy_lambda``.
    ``source`` names WHERE the caller got the objective, for
    :meth:`Axis.provenance`; it defaults to the mechanism used here.

    Resolving to the F_r default returns the shared :data:`F_R_AXIS` OBJECT, so
    ``axis is F_R_AXIS`` is a usable "did nothing move?" assertion.
    """

    if objective:
        objective = str(objective).strip()
        if deck is not None and limits is None:
            _, acq = deck_objective(deck)
            limits = acq
        axis = _axis_for(objective, limits, source or "--objective")
    elif deck is not None:
        obj, acq = deck_objective(deck)
        axis = _axis_for(obj, {**acq, **(limits or {})},
                         source or f"deck {Path(deck).name}")
    elif limits:
        axis = _axis_for(F_R_AXIS.objective, limits, source or "limits")
    else:
        return F_R_AXIS
    # Collapse an UNMOVED axis back onto the shared default object.  Only the
    # provenance string can differ at this point, and identity is worth more
    # than that string: `axis is F_R_AXIS` is then a one-token assertion that
    # this readout is doing exactly what it did before the switch.
    unmoved = (axis.key, axis.limit, axis.lam, axis.objective) == (
        F_R_AXIS.key, F_R_AXIS.limit, F_R_AXIS.lam, F_R_AXIS.objective)
    return F_R_AXIS if unmoved else axis


# --------------------------------------------------------------------------- #
# argparse plumbing — one pair of flags, identical in every readout
# --------------------------------------------------------------------------- #
def add_axis_args(parser: Any) -> Any:
    """Add ``--deck`` / ``--objective`` to a readout's parser.

    Neither has a default, so a readout invoked exactly as it is today resolves
    to :data:`F_R_AXIS` and prints exactly what it printed before.
    """

    g = parser.add_argument_group(
        "objective axis",
        "which axis the frontier / lambda-objective readout is computed on "
        "(design fxy_switch_design_20260829 sec. 3.5.5).  Default: F_r.")
    g.add_argument("--deck", default=None,
                   help="campaign deck (.inp) to read [acquisition] objective "
                        "and its limits from; 'min_fxy' switches the readout to "
                        "the store's f_xy column and f_xy_limit")
    g.add_argument("--objective", default=None,
                   help="name the deck objective directly instead of reading a "
                        "deck (e.g. min_fxy)")
    return parser


def axis_from_args(args: Any) -> Axis:
    """:func:`resolve_axis` from a namespace produced by :func:`add_axis_args`."""

    return resolve_axis(objective=getattr(args, "objective", None),
                        deck=getattr(args, "deck", None))


# --------------------------------------------------------------------------- #
# frame helpers
# --------------------------------------------------------------------------- #
def axis_values(frame: Any, axis: Axis, *, prefix: str = "") -> Any:
    """``frame[prefix + axis.key]`` as float, or an all-NaN column if absent.

    An ABSENT column is not an error here — it is the ordinary state of ``f_xy``
    in a corpus assembled before the switch — but it must become NaN rather than
    a KeyError so :func:`split_labelled` can count it as unlabelled and the
    readout can say so out loud.
    """

    import numpy as np
    import pandas as pd

    col = f"{prefix}{axis.key}"
    if col not in getattr(frame, "columns", ()):
        return pd.Series(np.nan, index=frame.index, dtype=float, name=col)
    return pd.to_numeric(frame[col], errors="coerce")


def split_labelled(frame: Any, axis: Axis, *, prefix: str = "") -> tuple[Any, int]:
    """``(rows whose axis is finite, count of the rest)``.

    On the F_r axis this is a no-op in every population the programme holds
    (``f_r`` is written for every converged row), which is why the F_r readouts
    are unchanged; on F_xy it is the whole point.
    """

    import numpy as np

    v = axis_values(frame, axis, prefix=prefix)
    keep = np.isfinite(v.to_numpy(dtype=float, na_value=np.nan))
    return frame[keep], int((~keep).sum())


def unlabelled_note(axis: Axis, n_unlabelled: int, n_total: int,
                    *, what: str = "rows") -> str:
    """The EXPLICIT unlabelled line, or ``""`` on the F_r axis.

    Returning ``""`` for F_r is deliberate: it is what keeps a min_fr readout
    byte-identical to its pre-switch self.  On F_xy the line is emitted even when
    the count is zero — "0 excluded" is a measurement, and its absence would be
    indistinguishable from a readout that never checked.
    """

    if not axis.is_fxy:
        return ""
    frac = (100.0 * n_unlabelled / n_total) if n_total else 0.0
    return (f"    {axis.label} unlabelled: {n_unlabelled}/{n_total} {what} "
            f"({frac:.1f} %) EXCLUDED from the frontier — no measured "
            f"{axis.label} (MAS_OUT not harvested)")


def lambda_objective(cyclen: Any, axis_value: Any, axis: Axis) -> Any:
    """``cyclen − λ·axis`` — the scalarised objective, higher is better.

    The registered rule is that a frontier is read on THIS, not on ``min(axis)``
    alone: the F_r-only headline was overturned twice by the λ reading, which is
    why the check is mandatory.  Design §3.5.5 moves the axis, not the rule.
    """

    import numpy as np

    return np.asarray(cyclen, dtype=float) - axis.lam * np.asarray(
        axis_value, dtype=float)


def best_by_lambda(frame: Any, axis: Axis, *, cyclen_col: str = "cyclen",
                   prefix: str = "") -> tuple[Any, int]:
    """``(row maximising cyclen − λ·axis, n_unlabelled)`` over a labelled frame.

    Returns ``(None, n_unlabelled)`` when nothing is labelled — the honest answer
    when a cell's F_xy has not been harvested yet, and NOT a fallback to F_r.
    """

    import numpy as np

    labelled, n_unlabelled = split_labelled(frame, axis, prefix=prefix)
    if not len(labelled):
        return None, n_unlabelled
    score = lambda_objective(labelled[cyclen_col],
                             axis_values(labelled, axis, prefix=prefix), axis)
    return labelled.iloc[int(np.nanargmax(score))], n_unlabelled


def _check_against_lpopt() -> None:
    """Assert the inlined limits still equal lpopt's, when lpopt is importable."""

    try:
        from lpopt.search.delivery import (LICENSING_FR_LIMIT as _FR,
                                           LICENSING_FXY_LIMIT as _FXY)
    except Exception:                                        # noqa: BLE001
        return
    if (_FR, _FXY) != (LICENSING_FR_LIMIT, LICENSING_FXY_LIMIT):
        raise AssertionError(
            f"readout_axis limits {(LICENSING_FR_LIMIT, LICENSING_FXY_LIMIT)} "
            f"disagree with lpopt.search.delivery {(_FR, _FXY)} — the readouts "
            f"would gate on a different number than the campaign did")
