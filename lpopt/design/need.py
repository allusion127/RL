"""Need signal ``N`` on the ``FuelDesign`` axis -- **skeleton only** (task #7).

Nothing here is wired into any production path, and nothing may be until the
prerequisites below are met.  This file exists so the interfaces the launch
rules F1-F5 (task #9) will need are named and typed in one place, and so the
one hard prerequisite -- a *measured* ``sigma_chain,paired`` -- is enforced in
code rather than remembered.

**Why this is last in the dependency order.**  Every lpopt objective ranks
patterns with the fuel case held fixed (``search/acquisition.py:841`` ``score_min_fxy``,
``:489`` ``score_min_fr_max_cycle``, ``:1054`` ``MinFuelCostSpec``, ``:1799``
``score_flat_power``).  There is no objective term on the ``FuelDesign`` axis
at all, and "expected improvement per design dollar" has never been defined.
The hole to plug it into does exist -- ``search/construct.build_pair_universe(types=...)``
(``construct.py:770``), ``achievable_e_core_interval`` (``:748``),
``screen_e_core_band`` (``:847``) -- which is what task #8 will use.

**The slice does not use this module** (assumption A6: slice first).  Round 1
is a paired physical experiment; this is the round-2 automatic-launch machinery.

**Registered constraints on the eventual implementation.**

* ``geometry`` descriptor channels are excluded from the perturbation study:
  the ``ood_guard`` population envelope for those channels is the degenerate
  interval ``[0, 0]``, so a finite difference along them is meaningless.
* The neural head is **shadow only**.  Its within-cell ``F_xy`` rank fidelity is
  a measured failure (r2 RANK 3/3 FAIL; pinbu r1 within-cell rho -0.11; arm 4
  G5 R1 rho 0.535 with an n=95 CI containing 0), and the claim that the OPSCREEN
  chain beats the head is *also* unsupported -- in the only like-for-like
  comparison the head was better.  Both stay out of the gate.
* The chain's own paired error must be *measured*, not assumed: see
  :func:`launch_allowed` / :data:`SIGMA_CHAIN_REQUIRED_REPORT`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "EXCLUDED_CHANNELS",
    "SIGMA_CHAIN_REQUIRED_REPORT",
    "SigmaVerdict",
    "sigma_ladder_verdict",
    "LaunchRules",
    "launch_allowed",
    "need_signal",
]

#: Descriptor channels that must not be perturbed (degenerate ``ood_guard``
#: envelope).  Extend only with evidence.
EXCLUDED_CHANNELS: tuple[str, ...] = ("geometry",)

#: F1 has no default: the launch rules are inert until this report exists and
#: names a measured number.
SIGMA_CHAIN_REQUIRED_REPORT: str = (
    "data/reports/assembly_sigma_chain_retrodiction_20260903.md"
)


@dataclass(frozen=True)
class SigmaVerdict:
    """Where a measured ``sigma_chain,paired`` lands on the pre-registered ladder."""

    sigma: float
    band: str            # "resolving" | "order-only" | "unresolving"
    #: The F2 multiplier ``k`` in ``bar = k * sigma`` -- a dimensionless 2.0 in
    #: both live bands, ``None`` when the triggers are void.  The bar itself is
    #: :attr:`bar`; the two were previously conflated in one expression.
    k_bar: float | None
    triggers_valid: bool
    note: str

    @property
    def bar(self) -> float | None:
        """``k_bar * sigma`` -- the acceptance bar in F_xy units, or ``None``."""
        return None if self.k_bar is None else self.k_bar * self.sigma


def sigma_ladder_verdict(sigma: float) -> SigmaVerdict:
    """The pre-registered acceptance ladder (slice-Z pre-registration section 2.3).

    ``< 0.005``    -> the chain resolves the decision threshold; F2 keeps k = 2.
    ``0.005-0.020`` -> ordering key only; the bar becomes ``2*sigma`` and every
                       candidate must be re-checked against it.
    ``> 0.020``    -> magnitude undecidable; **F1-F5 are discarded** and round 2
                       is run as a paired physical experiment.

    The registered base-case expectation is the third band.
    """
    s = float(sigma)
    if s < 0.005:
        return SigmaVerdict(s, "resolving", 2.0, True,
                            "chain resolves the decision threshold; k=2 bar stands")
    if s <= 0.020:
        return SigmaVerdict(s, "order-only", 2.0, True,
                            "ordering key only; bar = 2*sigma, recompute every "
                            "candidate against it")
    return SigmaVerdict(s, "unresolving", None, False,
                        "magnitude undecidable; F1-F5 discarded, round 2 is a "
                        "paired physical experiment")


@dataclass(frozen=True)
class LaunchRules:
    """F1-F5.  Every field is a *registered* threshold, not a tunable."""

    #: F1 -- task #0 complete and sigma measured.  No default.
    sigma_chain_paired: float | None = None
    #: F2 -- paired predicted improvement must clear ``k * sigma``.
    k: float = 2.0
    #: F4 -- role contrast, pair-wise
    contrast_min: float = 0.026
    #: F5 -- operating window
    cyclen_window: tuple[float, float] = (620.0, 645.0)
    cbc_max: float = 1500.0


def launch_allowed(rules: LaunchRules) -> tuple[bool, str]:
    """F1 alone, evaluated: no measured sigma -> no launch, ever.

    This is the whole of the skeleton's enforcement surface.  F2-F5 need the
    task #4 screener, the task #1 layout catalog and a candidate roster, none
    of which exist yet.
    """
    if rules.sigma_chain_paired is None:
        return False, (
            "F1 unmet: sigma_chain,paired has not been measured; see "
            f"{SIGMA_CHAIN_REQUIRED_REPORT}")
    v = sigma_ladder_verdict(rules.sigma_chain_paired)
    if not v.triggers_valid:
        return False, f"F1 met but ladder verdict '{v.band}': {v.note}"
    return True, f"F1 met; ladder verdict '{v.band}'"


def need_signal(*_args, **_kwargs) -> None:
    """Finite-difference need signal over ``FuelVec`` perturbations.

    **Not implemented** (task #7).  When it is, it must: exclude
    :data:`EXCLUDED_CHANNELS`, be invariant to channel permutation, and report
    per-channel sensitivity of the *objective*, not of the head's prediction.
    """
    raise NotImplementedError(
        "need.need_signal is a task #7 skeleton; the FuelVec finite-difference "
        "study is not implemented and must not be faked")
