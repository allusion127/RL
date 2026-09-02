"""Encoding-safe console logging.

Incident 2026-08-30 (``fpcamp_minfxy_t6t4_f121_r1`` on HOST_199): a finished
100-call campaign died in ``CampaignDriver._render_report`` with

    UnicodeEncodeError: 'cp949' codec can't encode character '\u2014'

because the launcher redirects stdout to a log file, Windows picks the ANSI
codepage (cp949) for that file, and one gate message carries an em-dash.  The
run had already spent its whole MASTER budget; ``report.md`` and
``delivery.json`` were never written.

The rule this module encodes: **a log line must never be able to sink a run.**
Two entry points:

* :func:`configure_stdio` — called once at the CLI entry, so every bare
  ``print()`` in every sub-command survives a narrow console encoding.
* :func:`safe_logger` / :func:`safe_print` — used by the library drivers, which
  are also imported by scripts and tests that supply their own ``log=``
  callable and therefore never pass through the CLI entry.

Folding is transliterating-first (``—`` -> ``-``, ``≤`` -> ``<=``, ``σ`` ->
``sigma``) so a degraded log line stays readable, with ``errors="replace"`` as
the last resort.
"""

from __future__ import annotations

import sys
from typing import Callable, TextIO

__all__ = ["configure_stdio", "fold_to_encoding", "safe_logger", "safe_print"]

#: Non-ASCII characters that actually occur in lpopt's user-facing strings,
#: mapped to an ASCII spelling that survives cp949/cp1252/ascii streams.
_TRANSLIT: dict[str, str] = {
    "\u2014": "-",      # — em dash          (the 2026-08-30 incident character)
    "\u2013": "-",      # – en dash
    "\u2212": "-",      # − minus
    "\u2264": "<=",     # ≤
    "\u2265": ">=",     # ≥
    "\u2260": "!=",     # ≠
    "\u2248": "~=",     # ≈
    "\u00b1": "+/-",    # ±
    "\u00d7": "x",      # ×
    "\u00b7": "*",      # ·
    "\u2192": "->",     # →
    "\u2190": "<-",     # ←
    "\u03c3": "sigma",  # σ
    "\u03c1": "rho",    # ρ
    "\u0394": "delta",  # Δ
    "\u03b4": "delta",  # δ
    "\u03bb": "lambda", # λ
    "\u03bc": "mu",     # μ
    "\u00b0": "deg",    # °
    "\u00a7": "sec.",   # §
    "\u221a": "sqrt",   # √
    "\u221e": "inf",    # ∞
    "\u2026": "...",    # …
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
}
_TABLE = str.maketrans(_TRANSLIT)


def fold_to_encoding(msg: str, encoding: str | None) -> str:
    """``msg`` rendered so ``msg.encode(encoding)`` cannot raise.

    Returns ``msg`` untouched when it already encodes.  ``encoding=None`` folds
    all the way to ASCII, which every text stream accepts.
    """

    enc = encoding or "ascii"
    for candidate in (msg, msg.translate(_TABLE)):
        try:
            candidate.encode(enc)
            return candidate
        except (UnicodeEncodeError, LookupError):
            continue
    folded = msg.translate(_TABLE)
    try:
        return folded.encode(enc, "replace").decode(enc, "replace")
    except (UnicodeError, LookupError):
        return folded.encode("ascii", "replace").decode("ascii")


def safe_print(msg: str, stream: TextIO | None = None) -> None:
    """``print(msg)`` that degrades the text instead of raising."""

    out = sys.stdout if stream is None else stream
    try:
        print(msg, file=out)
        return
    except UnicodeEncodeError:
        pass
    try:
        print(fold_to_encoding(msg, getattr(out, "encoding", None)), file=out)
    except UnicodeEncodeError:
        print(fold_to_encoding(msg, None), file=out)


def safe_logger(log: Callable[[str], None] | None = None) -> Callable[[str], None]:
    """Wrap a ``log`` callable (or build the default) so it cannot raise on encoding.

    A caller-supplied logger is wrapped too: the incident's stream was owned by
    the launcher, not by lpopt, and the same trap applies to any sink a script
    hands in.
    """

    if log is None:
        return safe_print

    def _log(msg: str) -> None:
        try:
            log(msg)
        except UnicodeEncodeError:
            # Retry once, ASCII-folded: readable, and accepted by every codec.
            log(fold_to_encoding(msg, None))

    return _log


def configure_stdio() -> None:
    """Make ``sys.stdout``/``sys.stderr`` lossy rather than fatal on encode.

    Called at the CLI entry so every bare ``print()`` in every sub-command is
    covered without touching ~60 call sites.  Best-effort: a stream that is not
    a reconfigurable :class:`io.TextIOWrapper` (pytest capture, a pipe wrapper)
    is left alone and the per-call guards above still apply.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError, TypeError, AttributeError):
            pass
