"""MASTER4 ``MAS_OUT`` PLANAR peaking parser: ``FXYP`` (pin) / ``FXYA`` (assembly).

Third non-vendor MASTER-output parser, after :mod:`.edit5` (``MAS_SUM``) and
:mod:`.pinppi` (``MAS_PPI``).  The regexes and the "cycle maximum over every
depletion step" convention are ported from ``2_LP/MOCHA/master_sum.py``
(``_FXYP_RE`` / ``parse_mas_out_max_fxyp`` / ``parse_mas_out_peaking_features``);
the upstream file is NOT edited and NOT imported.

Why this file exists at all
---------------------------
``F_xy`` is the optimisation target of the 2026-08-29 objective switch
(``data/reports/fxy_switch_design_20260829.md``) and it is **not in MAS_SUM**:
EDIT3 carries ``FQN FRN FQP FRP`` only.  MASTER emits one ``$P2D_n`` block per
depletion step and prints the planar factors exactly once inside it::

    $P2D_1          0.000 DAY          0.000 EFPD
              MAXIMUM PIN     PLANAR POWER (FXYP)=      1.8186  AT (M ,14, 4, 6,11)
              MAXIMUM ASSMBLY PLANAR POWER (FXYA)=      1.6210  AT (M ,14, 3)

Note the RUN of spaces inside ``PIN     PLANAR``: a single-space literal grep
finds nothing (an actual trap hit during the design survey), so every gap is
``\\s+``.

What ``f_xy`` means here
------------------------
``f_xy`` is the **maximum over all depletion steps of the final equilibrium
cycle** — the design limit applies throughout the cycle, and the ``max_frp`` /
``max_fqp`` columns already in the store use the same convention, so a different
one would break axis-to-axis comparison.  Picking the final cycle is the
CALLER's job (:mod:`..search.verify` reads the retained final work dir;
:mod:`..tools.backfill_fxy` adjudicates a work dir on disk).

Contract
--------
:func:`parse_mas_out_fxy` is **tolerant**: a file with no FXYP line yields
``f_xy=None`` and ``sane=False``, never an exception.  Nothing in lpopt may
silently substitute ``FRP`` for a missing ``FXYP`` (they are different physical
quantities), and a ``None`` that flows into a nullable store column is the
honest way to say "not measured".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

#: Above this, a "converged" FXYP is a diverged-flux artefact, not a core.
#:
#: Measured population maxima are 2.12 (regen T6_T4, n=690) and 5.17 for a run
#: that MASTER still called converged (design §5.4).  The design draft named 4.0;
#: the P1 task tightened it to 3.0, which is still ~40% above anything physical.
FXY_GARBAGE_CEILING: float = 3.0
#: Below this it is not a peaking factor at all (``F_xy >= 1`` by definition:
#: it is a maximum of values normalised to a mean of 1).
FXY_GARBAGE_FLOOR: float = 1.0

#: Sentinel file a physics-killed MASTER work dir carries; its ``MAS_OUT`` holds
#: the FXYP of a diverging flux solve and must never be harvested as a label.
NONFINITE_SENTINEL = "NONFINITE_FLUX"

_NUM = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"

#: ``$P2D_<n>   <day> DAY   <efpd> EFPD`` — one per depletion step.
P2D_STEP_RE = re.compile(
    r"^\s*\$P2D_\d+\s+([-+0-9.Ee]+)\s+DAY\s+([-+0-9.Ee]+)\s+EFPD",
    re.IGNORECASE,
)
#: pin planar peak (the target).
FXYP_RE = re.compile(
    r"MAXIMUM\s+PIN\s+PLANAR\s+POWER\s*\(FXYP\)\s*=\s*" + _NUM, re.IGNORECASE
)
#: assembly planar peak (report-only companion; free in the same scan).
FXYA_RE = re.compile(
    r"MAXIMUM\s+ASSMBLY\s+PLANAR\s+POWER\s*\(FXYA\)\s*=\s*" + _NUM, re.IGNORECASE
)


@dataclass(frozen=True)
class FxyStep:
    """One ``$P2D`` depletion step's planar peaks (``None`` = not printed)."""

    efpd: float
    fxyp: float | None
    fxya: float | None


@dataclass(frozen=True)
class FxyResult:
    """Cycle-level planar peaking harvested from one ``MAS_OUT``."""

    #: max finite FXYP over every step, or ``None`` when the file has none.
    f_xy: float | None
    #: max finite FXYA over every step (report-only), or ``None``.
    f_xya: float | None
    #: per-step rows, file order.
    steps: tuple[FxyStep, ...]
    #: number of steps that carried a finite FXYP.
    n_steps: int
    #: ``f_xy`` present and inside ``[FXY_GARBAGE_FLOOR, FXY_GARBAGE_CEILING]``.
    sane: bool
    #: why ``sane`` is False (``""`` when sane) — carried into the backfill CSV.
    reason: str

    @property
    def efpd_max(self) -> float | None:
        """Largest step EFPD == the cycle length (verified equal to the dir's
        ``MAS_SUM`` EDIT2 EOC EFPD on 89/89 retained final cycles).  Lets a
        consumer that knows the record's ``cyclen`` check that this MAS_OUT is
        really the final cycle of its chain."""
        efpds = [s.efpd for s in self.steps if math.isfinite(s.efpd)]
        return max(efpds) if efpds else None


#: The "nothing measured" result — never raises, never lies.
EMPTY = FxyResult(f_xy=None, f_xya=None, steps=(), n_steps=0,
                  sane=False, reason="no_fxyp")


def _read_text_flex(path: Path) -> str:
    """Decode a MASTER output on a Windows/Korean box (same ladder as
    :func:`.edit5._read_text_flex`; copied, not imported, so this module stays
    stdlib-only and can be shipped to a campaign box on its own)."""
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("ascii", errors="replace")


def _as_text(text_or_path: str | Path) -> str:
    if isinstance(text_or_path, Path):
        return _read_text_flex(text_or_path)
    text = str(text_or_path)
    # A MAS_OUT is ~1 MB; a path never is.  The length guard keeps a long text
    # blob from being fed to the filesystem (MOCHA uses the same test).
    if len(text) < 500:
        candidate = Path(text)
        try:
            if candidate.is_file():
                return _read_text_flex(candidate)
        except OSError:                       # pragma: no cover - odd path chars
            pass
    return text


def _finite(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:                        # pragma: no cover - regex guards it
        return None
    return value if math.isfinite(value) else None


def parse_mas_out_fxy(text_or_path: str | Path) -> FxyResult:
    """Cycle FXYP/FXYA peaks from a ``MAS_OUT`` text or path.  NEVER raises on a
    missing field — a file with no ``$P2D`` block yields :data:`EMPTY`."""
    text = _as_text(text_or_path)

    steps: list[FxyStep] = []
    efpd = float("nan")
    fxyp: float | None = None
    fxya: float | None = None
    seen_step = False

    def _flush() -> None:
        if seen_step:
            steps.append(FxyStep(efpd=efpd, fxyp=fxyp, fxya=fxya))

    for line in text.splitlines():
        head = P2D_STEP_RE.match(line)
        if head:
            _flush()
            seen_step = True
            parsed = _finite(head.group(2))
            # NOT ``or float('nan')``: the BOC step's EFPD is a legitimate 0.0.
            efpd = float("nan") if parsed is None else parsed
            fxyp = fxya = None
            continue
        if not seen_step:
            continue
        match = FXYP_RE.search(line)
        if match:
            fxyp = _finite(match.group(1))
            continue
        match = FXYA_RE.search(line)
        if match:
            fxya = _finite(match.group(1))
    _flush()

    pins = [s.fxyp for s in steps if s.fxyp is not None]
    asms = [s.fxya for s in steps if s.fxya is not None]
    f_xy = max(pins) if pins else None
    f_xya = max(asms) if asms else None

    if f_xy is None:
        reason = "no_fxyp"
    elif f_xy > FXY_GARBAGE_CEILING:
        reason = "above_ceiling"
    elif f_xy < FXY_GARBAGE_FLOOR:
        reason = "below_floor"
    else:
        reason = ""
    return FxyResult(f_xy=f_xy, f_xya=f_xya, steps=tuple(steps),
                     n_steps=len(pins), sane=(reason == ""), reason=reason)


def fxy_from_work_dir(work_dir: str | Path) -> FxyResult | None:
    """Planar peaks of a MASTER work dir, or ``None`` when it must not be used.

    ``None`` (never an exception) for: a physics-killed dir (the
    :data:`NONFINITE_SENTINEL` is present — its FXYP describes a diverging solve,
    not an equilibrium core), a dir with no ``MAS_OUT``, and any read error.  The
    caller decides what an un-``sane`` result is worth; the harvest path refuses
    it, exactly as ``_maps_from_equilibrium_result`` refuses an unparseable map,
    because a label harvest failure must never abort a wave.
    """
    try:
        wd = Path(work_dir)
        if (wd / NONFINITE_SENTINEL).exists():
            return None
        mas_out = wd / "MAS_OUT"
        if not mas_out.is_file():
            return None
        return parse_mas_out_fxy(mas_out)
    except OSError:
        return None


__all__ = [
    "EMPTY",
    "FXYA_RE",
    "FXYP_RE",
    "FXY_GARBAGE_CEILING",
    "FXY_GARBAGE_FLOOR",
    "NONFINITE_SENTINEL",
    "P2D_STEP_RE",
    "FxyResult",
    "FxyStep",
    "fxy_from_work_dir",
    "parse_mas_out_fxy",
]
