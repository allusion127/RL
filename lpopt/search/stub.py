"""Deterministic fake MASTER evaluator for tests and ``produce --dry-run``.

:class:`StubEvaluator` satisfies the vendor
:class:`~lpopt.vendor.masterrl.search.PatternEvaluator` protocol
(``evaluate(case, pattern) -> EvaluationResult``) with a **deterministic**,
network- and executable-free figure of merit.  It exists so the whole produce /
verify harness — wave dispatch, ledger, resume, store rows, QC counters — can be
exercised end to end without a live MASTER run (the FEASIBLE_PACKAGE assets are
OneDrive-dehydrated, so no live run is possible in this task).

The FOM is a smooth function of ``(pattern.digest, feed)`` with physically
plausible ranges (plan M2.5 spec):

* ``F_r``     in 1.4 .. 2.5
* ``cyclen``  in 550 .. 720 EFPD, scaled by ``feed / 121`` (fewer fresh
  assemblies -> shorter cycle)
* ``cbc_max`` in 900 .. 1600 ppm

Failure injection is by ``digest`` prefix so a test can force the full outcome
taxonomy (converged / nonconverged / error) from patterns it generated:

* ``fail_prefixes``        -> the evaluator raises (verifier reports ``error``)
* ``nonconverge_prefixes`` -> a non-converged FOM (verifier reports ``nonconverged``)

A converged stub outcome also carries a deterministic ``metadata["fxy"]``, the
same channel the live ``HarvestingEquilibriumEvaluator`` uses for the MAS_OUT
planar peaks, so the ``min_fxy`` objective can be exercised end to end without a
MASTER run.  It respects the MEASURED physics ordering ``F_r <= F_xy <= F_q``.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Sequence

from ..data.fxy import FxyResult, FxyStep
from ..vendor.masterrl.dataset import CaseData
from ..vendor.masterrl.domain import FOM, Pattern
from ..vendor.masterrl.search import EvaluationResult

#: Stub F_xy generator (design fxy_switch_20260829 §1.2 pooled fit, plus a
#: deterministic per-digest spread of the measured residual scale).  It is a
#: FIXTURE, not a model: it exists so a dry run produces an f_xy column at all.
_FXY_SLOPE = 1.1221
_FXY_INTERCEPT = -0.0831
_FXY_SPREAD = 0.06


def _unit(digest: str, salt: str) -> float:
    """Deterministic value in ``[0, 1)`` from ``digest`` and a named ``salt``.

    Uses SHA-256 (not the salted builtin ``hash``) so results are identical
    across processes and runs — a hard requirement for resume/idempotency tests.
    """

    raw = sha256(f"{digest}:{salt}".encode("ascii")).hexdigest()
    return int(raw[:12], 16) / float(1 << 48)


class StubEvaluator:
    """Deterministic ``PatternEvaluator`` for dry runs and tests."""

    def __init__(
        self,
        *,
        fail_prefixes: Sequence[str] = (),
        nonconverge_prefixes: Sequence[str] = (),
        wall_s: float = 0.01,
    ) -> None:
        self.fail_prefixes = tuple(fail_prefixes)
        self.nonconverge_prefixes = tuple(nonconverge_prefixes)
        self.wall_s = float(wall_s)

    # -- FOM ---------------------------------------------------------------- #
    def fom_for(self, digest: str, feed: int) -> FOM:
        """The deterministic FOM for one ``(digest, feed)`` (converged=True)."""

        feed_scale = float(feed) / 121.0
        f_r = 1.4 + 1.1 * _unit(digest, "f_r")
        f_q = f_r * (1.15 + 0.35 * _unit(digest, "f_q"))
        cbc_max = 900.0 + 700.0 * _unit(digest, "cbc")
        cyclen = (550.0 + 170.0 * _unit(digest, "cyclen")) * feed_scale
        ao = -0.30 + 0.60 * _unit(digest, "ao")
        max_burnup = 40.0 + 15.0 * _unit(digest, "burnup")
        return FOM(
            f_r=f_r,
            cbc_max=cbc_max,
            f_q=f_q,
            cyclen=cyclen,
            ao_min=min(ao, -abs(ao) * 0.1) if ao > 0 else ao,
            ao_max=max(ao, abs(ao) * 0.1) if ao < 0 else ao,
            max_burnup=max_burnup,
            max_pin_burnup=max_burnup * 1.15,
            converged=True,
        )

    def fxy_for(self, digest: str, feed: int) -> FxyResult:
        """Deterministic planar peaks for one ``(digest, feed)``.

        Built from the SAME ``fom_for`` F_r/F_q so the stub honours the measured
        inequality ``F_r <= F_xy <= F_q`` (192/192 real cores): a test that
        asserts on the F_xy gate must not be fed a physically impossible row.
        """
        fom = self.fom_for(digest, feed)
        f_r = float(fom.f_r)
        f_q = float(fom.f_q)
        raw = (_FXY_SLOPE * f_r + _FXY_INTERCEPT
               + _FXY_SPREAD * (_unit(digest, "fxy") - 0.5))
        f_xy = min(max(raw, f_r), f_q)
        f_xya = f_xy / (1.05 + 0.10 * _unit(digest, "fxya"))
        step = FxyStep(efpd=float(fom.cyclen), fxyp=f_xy, fxya=f_xya)
        return FxyResult(f_xy=f_xy, f_xya=f_xya, steps=(step,), n_steps=1,
                         sane=True, reason="")

    def n_cycles_for(self, digest: str, feed: int) -> int:
        """Deterministic chain length: 3..8 (feed != 121 stays comfortably < 14)."""

        return 3 + int(6 * _unit(digest, "ncyc"))

    # -- PatternEvaluator protocol ----------------------------------------- #
    def evaluate(self, case: CaseData, pattern: Pattern) -> EvaluationResult:
        digest = pattern.digest
        if any(digest.startswith(prefix) for prefix in self.fail_prefixes):
            raise RuntimeError(f"stub injected failure for digest {digest}")

        feed = case.key.feed
        converged = not any(
            digest.startswith(prefix) for prefix in self.nonconverge_prefixes
        )
        fom = replace(self.fom_for(digest, feed), converged=converged)
        n_cycles = self.n_cycles_for(digest, feed)
        # A converged chain has generous slack; a nonconverged one exhausted a
        # small nominal cap.  These flow into WaveOutcome via metadata.
        tol_margin = 0.05 + 0.9 * _unit(digest, "tol") if converged else 1.0
        return EvaluationResult(
            fom=fom,
            raw_master_calls=n_cycles,
            metadata={
                "mode": "stub",
                "converged": converged,
                "n_cycles": n_cycles,
                "tolerance_margin": tol_margin,
                "wall_s": self.wall_s,
                # same channel HarvestingEquilibriumEvaluator uses; the verifier
                # only reads it for a CONVERGED outcome, so a nonconverged stub
                # row keeps a null f_xy exactly as a real one does.
                "fxy": self.fxy_for(digest, feed) if converged else None,
            },
        )

    # Convenience so a produce wave can go through ``ParallelPatternEvaluator``
    # semantics if ever wired that way; the verifier drives per-entry instead.
    evaluate_one = evaluate


__all__ = ["StubEvaluator"]
