"""The OPSCREEN model-free **FLOOR chain**, ported as pure functions (task #5).

    FF_hot  ->  node_peak regression  ->  F_r = A * node_peak * FF_hot  ->  F_xy = r * F_r

Every constant below is transcribed from ``5_RL/opmodel/`` -- the fitted
coefficients from ``frmodel2.npz`` (written by ``s09b_contrast.py``), the
reported spreads from ``opmodel/OPSCREEN.md:241-252``, and the floor form from
``opmodel/s16_final_table.py:48``.  Nothing here re-fits anything: this module
is a *transcription* so the screen can be run without importing the analysis
scripts (which are Windows-path-bound and load 119 kB of cached curves).

**This chain is a FLOOR, not a prediction.**  Verbatim, ``OPSCREEN.md:250-252``:

    **Consequence for the screen**: ``contrast >= 0.026`` is a hard gate, and
    ``F_r_floor`` is reported as what the LP could reach *if* it restores
    node_peak to the best value ever measured -- not as a prediction of what the
    current patterns will give.

**The B3 counterexample** (``OPSCREEN.md:233-241``, ``opmodel/measured.py``).
Arm ``B3`` = ``paramA T5_T6 @ f121`` is the flattest of the four paramA arms --
``FF_hot`` 1.1020, within 0.001 of the flattest arm measured anywhere in the
table (``C5`` = E3 x121, 1.1010, ``OPSCREEN.md:222``) -- and it
recorded the **worst** measured ``F_r`` of the four paramA arms, **1.5795**
(``measured.py:FR_FLAT["B3"]``), with measured ``node_peak`` 1.3906.  Its role
contrast is **-0.0016**: T5 and T6 differ by 0.05 w/o U-235, so the pattern
degenerates into a single-type load.  A screen that ranks on ``FF_hot`` alone
picks B3 first and is wrong by +0.05 in ``F_r``; that is precisely why
:data:`CONTRAST_MIN` is a *hard* gate and not a tie-breaker.  See
:func:`b3_counterexample`.

**``d_fresh`` is NOT ``hump``.**  ``d_fresh`` is
``rho_mix(0.5) - rho_op(rho_mix, feed, bc, 0.0)`` (``s09b_contrast.py:27``) --
the gap between the *fresh* assembly reactivity and the *equilibrium core
average* at BOC.  ``hump`` is ``max(rho_mix(BU in [0.5, 12])) - rho_mix(0.5)``
(``s16_final_table.py:123``) -- the Gd burn-out overshoot of the mixture alone.
They are different quantities with different signs and different magnitudes;
the v2 design draft conflated them.  :func:`d_fresh` and :func:`mixture_hump`
are kept as separate functions so the difference is testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# equilibrium bookkeeping (ported from opmodel/opmodel.py:29-30, 68-80, 122-124)
# --------------------------------------------------------------------------- #
#: MWd/kgHM burnt per EFPD at rated power (``opmodel.py:29``).
RATE: float = 3983.0 / 104.8 / 1000.0

#: Full-core assembly count (``opmodel.py:30``).
NSLOT: int = 241

#: The Xe-free BU = 0 lattice point is dropped everywhere: MASTER runs
#: equilibrium Xe, so BU >= 0.2 is the physically comparable state
#: (``opmodel.py:23-25``).
BU_MIN: float = 0.2


def batch_weights(feed: int) -> list[tuple[float, int]]:
    """Equilibrium batch census ``[(n_assemblies, batch_index), ...]``.

    Port of ``opmodel.batch_weights`` (``opmodel.py:68-80``).  Batch index ``j``
    means "this batch sits at burnup ``t + j*Bc``"; ``j = 0`` is the fresh feed.
    Handles the non-integer residence of the ``1 + 4N`` feed grid exactly
    (``241 = k*feed + remainder``).
    """
    out: list[tuple[float, int]] = []
    left, j = NSLOT, 0
    while left > 0:
        take = min(int(feed), left)
        out.append((float(take), j))
        left -= take
        j += 1
    return out


def rho_op(rho_mix: Callable[[np.ndarray], np.ndarray],
           feed: int, bc: float, t: float) -> float:
    """Equilibrium core-average k-inf reactivity at burnup-time ``t`` into the
    cycle.  Port of ``opmodel.rho_op`` (``opmodel.py:122-124``)."""
    return float(
        sum(n * float(rho_mix(t + j * bc)) for n, j in batch_weights(feed)) / NSLOT
    )


def mix_rho(curves: Sequence[Callable[[np.ndarray], np.ndarray]],
            weights: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    """Slot-weighted average ``rho(BU)`` of a fuel mixture.

    Port of ``opmodel.mix`` (``opmodel.py:53-64``).  For a two-role pair at
    feed 121 the weights are the slot census ``(68, 53)``; a single-type load
    uses ``[1.0]`` with one curve.
    """
    w = np.asarray(weights, float)
    w = w / w.sum()

    def f(x):
        x = np.asarray(x, float)
        out = np.zeros(x.shape if x.shape else ())
        for c, wi in zip(curves, w):
            out = out + wi * np.asarray(c(x), float)
        return out

    return f


# --------------------------------------------------------------------------- #
# chain constants -- transcribed, with their sigma
# --------------------------------------------------------------------------- #
#: ``node_peak = c0 + c1*contrast + c2*d_fresh``, least squares over the 15
#: ``ARMS_FLAT`` arms that carry a measured ``node_peak``
#: (``s09b_contrast.py:40-42`` -> ``frmodel2.npz["c"]``).  ``OPSCREEN.md:241``
#: quotes these rounded to ``1.4210 - 4.1725*contrast - 3.4862*d_fresh``.
NODE_PEAK_COEFFS: tuple[float, float, float] = (
    1.4209643291368383,     # intercept
    -4.172540456817703,     # * contrast
    -3.4861743750806697,    # * d_fresh
)

#: In-sample spread of that fit (``OPSCREEN.md:241``): rms 0.036, R2 0.866,
#: n = 15.  **This is an in-sample number** -- the 15 arms ARE the fit
#: population, so it under-states the held-out error.  The leave-one-arm-out
#: retrodiction (task #0,
#: ``data/reports/assembly_sigma_chain_retrodiction_20260903.md``) measures
#: node_peak MAE 0.036 / p95 0.113 and F_r MAE 0.054 / p95 0.113 out of sample.
NODE_PEAK_RMS: float = 0.036
NODE_PEAK_R2: float = 0.866
NODE_PEAK_N: int = 15

#: Fusion scale ``A = sum(F_r*p*FF) / sum((p*FF)**2)`` with ``p`` the
#: **predicted** node_peak (``s09b_contrast.py:49`` -> ``frmodel2.npz["A"]``).
#: This is the value ``s16_final_table.py:20,47`` actually uses for ``Fr_fix``.
A_FUSION: float = 1.0319443879524353

#: ``OPSCREEN.md:242`` reports ``A = F_r/(node_peak*FF_hot) = 1.035 +- 0.031``.
#: That 1.035 is the *ratio-mean* estimator kept in ``frmodel.npz["A"]``
#: (1.0353784398445056), NOT the least-squares :data:`A_FUSION` -- the two
#: differ by 0.0034.  The +-0.031 is the sigma of A and is the dominant
#: multiplicative term in the chain's error budget.
A_RATIO_MEAN: float = 1.0353784398445056
A_SIGMA: float = 0.031

#: ``Fr_flr = 1.03 * 1.2085 * FF_hot`` (``OPSCREEN.md:260``,
#: ``s16_final_table.py:48``).  1.2085 is the **best node_peak ever measured**
#: (arm A0 = ga80 E1_E2 @ f121, ``measured.py:FR_FLAT["A0"]``), so the floor
#: answers "what could the LP reach if it restored that node_peak", not "what
#: will this pattern give".  1.03 is the rounded A of the older ``s09``
#: fusion law (``frmodel.npz``), kept literal so the published tables reproduce.
FR_FLOOR_A: float = 1.03
FR_FLOOR_NODE_PEAK: float = 1.2085

#: Hard role-contrast gate (``OPSCREEN.md:250``).  Drawn from the 15
#: fixed-pattern arms: contrast >= 0.043 -> node_peak 1.209-1.260;
#: 0.026-0.028 -> 1.274-1.327; ~0 -> 1.387-1.551 (``OPSCREEN.md:235-239``).
CONTRAST_MIN: float = 0.026

#: The contrast window used by ``s09b_contrast.py:30`` is ``BU 0.5 .. bc/3``
#: (20 points); ``s16_final_table.py:115`` uses the fixed ``0.5 .. 8.0`` (12
#: points) for realized pairs.  Both are "the first third of a cycle".
CONTRAST_BU_LO: float = 0.5
CONTRAST_BU_HI_FIXED: float = 8.0
CONTRAST_NPTS: int = 20
#: The point count that goes with ``CONTRAST_BU_HI_FIXED``: ``s16_final_table.py:115``
#: samples the fixed 0.5..8.0 window with **12** points, not 20.  Pass both
#: together (``bu_hi=CONTRAST_BU_HI_FIXED, npts=CONTRAST_NPTS_FIXED``) when
#: reproducing an s16 row; mixing the s16 window with the s09b point count is a
#: silent convention error.
CONTRAST_NPTS_FIXED: int = 12

#: The hump correction was fitted over mixture humps 0.0000-0.0148; beyond that
#: it extrapolates and the honest cycle length is the interval ``[raw, cyc]``
#: (``s16_final_table.py:1-6,21``).
HUMP_CAL_MAX: float = 0.0148
HUMP_BU_HI: float = 12.0
HUMP_NPTS: int = 24

#: ``F_xy = r * F_r``.  Measured on the store (76,793 rows; 7,758 carry both
#: ``f_xy`` and ``f_r``): r mean 1.0780, sd 0.0299 overall; at feed 121 alone
#: mean 1.0693, sd 0.0250.  Within a single (library, case_pair, feed) cell the
#: sd is 0.006-0.033, so r does **not** cancel in a paired difference taken
#: inside one cell.  See the task-#0 retrodiction report.
F_XY_RATIO_MEAN: float = 1.0780
F_XY_RATIO_SIGMA: float = 0.0299
F_XY_RATIO_MEAN_F121: float = 1.0693
F_XY_RATIO_SIGMA_F121: float = 0.0250


# --------------------------------------------------------------------------- #
# the chain, as pure functions
# --------------------------------------------------------------------------- #
def role_contrast(curve_hi: Callable[[np.ndarray], np.ndarray],
                  curve_lo: Callable[[np.ndarray], np.ndarray],
                  *, bu_hi: float, bu_lo: float = CONTRAST_BU_LO,
                  npts: int = CONTRAST_NPTS) -> float:
    """Mean ``rho`` difference between the two fresh roles over the zoning window.

    ``contrast = mean(rho_68slot(BU) - rho_53slot(BU))`` for ``BU`` on
    ``linspace(bu_lo, bu_hi, npts)`` (``s09b_contrast.py:29-31``).  ``bu_hi``
    is ``bc/3`` in ``s09b`` and the fixed 8.0 in ``s16``.  ``npts`` must follow
    the window: 20 with ``bc/3`` (s09b), :data:`CONTRAST_NPTS_FIXED` = 12 with
    :data:`CONTRAST_BU_HI_FIXED` (s16).  The default is the s09b pair.
    """
    ts = np.linspace(float(bu_lo), float(bu_hi), int(npts))
    return float(np.mean(np.asarray(curve_hi(ts), float)
                         - np.asarray(curve_lo(ts), float)))


def d_fresh(rho_mix: Callable[[np.ndarray], np.ndarray],
            feed: int, bc: float) -> float:
    """``rho_mix(0.5) - rho_op(rho_mix, feed, bc, 0.0)`` (``s09b_contrast.py:27``).

    The fresh-vs-equilibrium reactivity gap.  **Not** :func:`mixture_hump`.
    """
    return float(rho_mix(CONTRAST_BU_LO)) - rho_op(rho_mix, feed, bc, 0.0)


def mixture_hump(rho_mix: Callable[[np.ndarray], np.ndarray],
                 *, bu_hi: float = HUMP_BU_HI,
                 npts: int = HUMP_NPTS) -> float:
    """``max(rho_mix(BU in [0.5, 12])) - rho_mix(0.5)`` (``s16_final_table.py:123``).

    The Gd burn-out overshoot of the mixture alone -- a property of the fuel
    curves, with no core, no feed and no batch census in it.  **Not**
    :func:`d_fresh`.
    """
    ts = np.linspace(CONTRAST_BU_LO, float(bu_hi), int(npts))
    return float(np.max(np.asarray(rho_mix(ts), float))
                 - float(rho_mix(CONTRAST_BU_LO)))


def hump_extrapolates(hump: float) -> bool:
    """True when the hump correction is outside its calibration range."""
    return float(hump) > HUMP_CAL_MAX


def node_peak(contrast: float, dfresh: float,
              coeffs: Sequence[float] = NODE_PEAK_COEFFS) -> float:
    """``node_peak = c0 + c1*contrast + c2*d_fresh``.

    Coefficients and form from ``s09b_contrast.py:27,40-42`` -- the fit that
    produced ``frmodel2.npz``, and the only place the third regressor is
    ``d_fresh``.  ``s16_final_table.py:46`` evaluates the *same* regression at
    screening time with a different third input: ``rf - rboc``, the mixture
    reactivity at the ``BU = 0.2`` grid point minus the CBC-calibrated BOC core
    reactivity ``RS + cbc*WB``.  That proxy is not ``d_fresh()`` and the two are
    not interchangeable, so this module ports the s09b form.
    """
    c0, c1, c2 = (float(v) for v in coeffs)
    return c0 + c1 * float(contrast) + c2 * float(dfresh)


def f_r_fixed(npk: float, ff_hot: float, a: float = A_FUSION) -> float:
    """``F_r = A * node_peak * FF_hot`` (``s16_final_table.py:47``).

    The *modelled* F_r: it uses the node_peak this fuel pair is predicted to
    give.  Its held-out error is large (task #0: MAE 0.054, p95 0.113), so it
    is reported as the upper end of an interval, never alone.
    """
    return float(a) * float(npk) * float(ff_hot)


def f_r_floor(ff_hot: float) -> float:
    """``Fr_flr = 1.03 * 1.2085 * FF_hot`` (``s16_final_table.py:48``).

    What the LP could reach **if** it restored node_peak to the best value ever
    measured -- not a prediction (``OPSCREEN.md:250-252``).
    """
    return FR_FLOOR_A * FR_FLOOR_NODE_PEAK * float(ff_hot)


def f_r_interval(npk: float, ff_hot: float,
                 a: float = A_FUSION) -> tuple[float, float]:
    """Return ``(F_r_floor, F_r_fixed)`` -- the caller gets an interval, never a
    single number.  The floor can exceed the fixed value when the pair's
    predicted node_peak is *better* than 1.2085; that ordering is information,
    so the tuple is returned unsorted and labelled."""
    return f_r_floor(ff_hot), f_r_fixed(npk, ff_hot, a)


def f_xy(fr: float, r: float = F_XY_RATIO_MEAN) -> float:
    """``F_xy = r * F_r``.  ``r`` is measured per cell; the module default is the
    store-wide mean (:data:`F_XY_RATIO_MEAN`)."""
    return float(r) * float(fr)


def contrast_gate(contrast: float,
                  contrast_min: float = CONTRAST_MIN) -> bool:
    """The hard gate of ``OPSCREEN.md:250``.  ``True`` == the pair may proceed."""
    return float(contrast) >= float(contrast_min)


@dataclass(frozen=True)
class ChainResult:
    """One evaluation of the floor chain for a (68-role, 53-role, feed) pair."""

    contrast: float
    d_fresh: float
    node_peak: float
    ff_hot: float
    f_r_floor: float
    f_r_fixed: float
    f_xy_floor: float
    f_xy_fixed: float
    contrast_ok: bool

    @property
    def f_r_bounds(self) -> tuple[float, float]:
        return (self.f_r_floor, self.f_r_fixed)


def evaluate(*, contrast: float, dfresh: float, ff_hot: float,
             a: float = A_FUSION, r: float = F_XY_RATIO_MEAN,
             contrast_min: float = CONTRAST_MIN) -> ChainResult:
    """Run the whole floor chain on already-computed descriptors.

    Pure: no file IO, no surrogate, no store.  ``contrast`` and ``dfresh`` come
    from :func:`role_contrast` / :func:`d_fresh`; ``ff_hot`` from the DeCART
    ``%DIST`` max for a realized type or from the surrogate ensemble for a new
    design (which runs 0.0014 low, ``OPSCREEN.md:261``).
    """
    npk = node_peak(contrast, dfresh)
    flr, fix = f_r_interval(npk, ff_hot, a)
    return ChainResult(
        contrast=float(contrast),
        d_fresh=float(dfresh),
        node_peak=npk,
        ff_hot=float(ff_hot),
        f_r_floor=flr,
        f_r_fixed=fix,
        f_xy_floor=f_xy(flr, r),
        f_xy_fixed=f_xy(fix, r),
        contrast_ok=contrast_gate(contrast, contrast_min),
    )


# --------------------------------------------------------------------------- #
# the documented counterexample, as data
# --------------------------------------------------------------------------- #
#: ``arm -> (FF_hot, measured F_r, measured node_peak, contrast, d_fresh)``.
#: ``F_r``/``node_peak``/``FF_hot`` are ``opmodel/measured.py:FR_FLAT`` +
#: ``hgc_curves.npz``; ``contrast`` and ``d_fresh`` are ``s09b_contrast.py``'s
#: output, reproduced on HOST_238 2026-09-03.
_ARMS = {
    #  arm:  (FF_hot, F_r,    node_peak, contrast,  d_fresh)
    "A0": (1.1520, 1.5207, 1.2085, +0.04899, -0.00417),  # ga80 E1_E2 (best npk)
    "B2": (1.1430, 1.5329, 1.2145, +0.04385, -0.00655),  # paramA T3_T4
    "B3": (1.1020, 1.5795, 1.3906, -0.00156, +0.01729),  # paramA T5_T6 (counterexample)
}

#: Published mixture ``hump`` for the same two paramA pairs
#: (``OPSCREEN.md:275-276``, the ``hump`` column).  Compare against the
#: ``d_fresh`` entries of :data:`_ARMS`: for ``T3_T4`` the two even have
#: **opposite signs** (hump +0.0128, d_fresh -0.00655).  They are not the same
#: quantity and must never be substituted for one another.
PUBLISHED_HUMP = {
    "T3_T4@121": 0.0128,
    "T5_T6@121": 0.0148,
}


def measured_arm(arm: str) -> dict:
    """One measured arm as a labelled dict (``A0`` / ``B2`` / ``B3``)."""
    ff, fr, npk, con, dfr = _ARMS[arm]
    return {"ff_hot": ff, "f_r": fr, "node_peak": npk,
            "contrast": con, "d_fresh": dfr}


def b3_counterexample() -> dict:
    """The B3 counterexample as machine-readable data.

    ``B3`` (paramA ``T5_T6`` @ f121) has the **lowest** ``FF_hot`` of the
    measured arms and the **worst** ``F_r`` of the paramA arms.  Ranking on
    ``FF_hot`` alone inverts the true order; the contrast gate is what fixes it.
    """
    ff_b3, fr_b3, npk_b3, con_b3, _ = _ARMS["B3"]
    ff_b2, fr_b2, npk_b2, con_b2, _ = _ARMS["B2"]
    return {
        "B3": {"arm": "paramA T5_T6 @ f121", "ff_hot": ff_b3, "f_r": fr_b3,
               "node_peak": npk_b3, "contrast": con_b3,
               "contrast_ok": contrast_gate(con_b3)},
        "B2": {"arm": "paramA T3_T4 @ f121", "ff_hot": ff_b2, "f_r": fr_b2,
               "node_peak": npk_b2, "contrast": con_b2,
               "contrast_ok": contrast_gate(con_b2)},
        "flatter_is_B3": ff_b3 < ff_b2,
        "worse_f_r_is_B3": fr_b3 > fr_b2,
        "source": "opmodel/measured.py:FR_FLAT; opmodel/OPSCREEN.md:233-241",
    }


def error_budget(ff_hot: float) -> dict:
    """First-order propagation of the chain's published sigmas onto ``F_r``.

    ``F_r = A * npk * FF_hot`` so
    ``sigma_Fr^2 = (npk*FF*sigma_A)^2 + (A*FF*sigma_npk)^2`` with
    ``sigma_A = 0.031`` (:data:`A_SIGMA`) and ``sigma_npk = 0.036``
    (:data:`NODE_PEAK_RMS`, in-sample).  This is the *analytic* budget only --
    the measured out-of-sample number lives in the task-#0 report and is
    larger.
    """
    npk = FR_FLOOR_NODE_PEAK
    ff = float(ff_hot)
    from_a = npk * ff * A_SIGMA
    from_npk = A_FUSION * ff * NODE_PEAK_RMS
    return {
        "sigma_from_A": from_a,
        "sigma_from_node_peak": from_npk,
        "sigma_f_r": float(np.hypot(from_a, from_npk)),
        "note": "analytic, in-sample sigmas; see the task-#0 retrodiction for "
                "the measured leave-one-arm-out and paired-difference numbers",
    }


def as_table(results: Iterable[ChainResult]) -> str:
    """Render results in the ``s16_final_table.py:55-64`` column order."""
    head = (f"{'FFhot':>7}{'contr':>8}{'d_fresh':>9}{'npk':>6}"
            f"{'Fr_fix':>7}{'Fr_flr':>7}{'gate':>6}")
    lines = [head]
    for r in results:
        lines.append(f"{r.ff_hot:>7.4f}{r.contrast:>+8.4f}{r.d_fresh:>9.5f}"
                     f"{r.node_peak:>6.3f}{r.f_r_fixed:>7.3f}"
                     f"{r.f_r_floor:>7.3f}{'OK' if r.contrast_ok else 'LOW':>6}")
    return "\n".join(lines)
