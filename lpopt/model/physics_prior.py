"""Leading-order reactivity-balance physics prior for cyclen (v5 bundle).

Instead of regressing ``cyclen`` directly, the v5 arms may regress the RESIDUAL
against a closed-form physics estimate built from the assembly lattice results
already harvested into ``fuel_types`` (the reference k-inf(BU) curve shape:
``kinf_peak`` / ``bu_peak_gwd`` / ``reactivity_swing_pcm`` /
``depletion_slope_pcm_per_gwd``).  The network then only has to learn what the
zero-dimensional balance cannot express — spatial power shape, leakage
redistribution, shuffle geometry — instead of re-deriving the reactivity
bookkeeping that the lattice code already computed.

The formula
-----------
For each of the 69 quarter slots ``j``, trace the shuffle chain to its FRESH
origin (exactly as :class:`~.featurize.FeatureEncoder` does) and read that
origin's harvested curve.  With

* ``w_j``       the slot's orbit multiplicity (1, 2 or 4) — so a plain weighted
                mean over the quarter equals the full-core mean,
* ``bu_j0``     the a-priori accumulated burnup
                ``(age_j - 1) * NOMINAL_CYCLE_BURNUP_MWD_KG`` [GWd/tU],
* ``bu_j*``     ``bu_peak_gwd`` (burnout-hump burnup),
* ``rho_j*``    ``(kinf_peak - 1) / kinf_peak * 1e5`` [pcm] (hump reactivity),
* ``s_j``       ``depletion_slope_pcm_per_gwd`` (< 0),
* ``S_j``       ``reactivity_swing_pcm`` (holddown-release magnitude, >= 0),

the assembly reactivity at burnup ``b`` is the two-segment curve

    rho_j(b) = rho_j* + s_j * (b - bu_j*)                for b >= bu_j*
    rho_j(b) = rho_j* - S_j * (1 - b / bu_j*)            for b <  bu_j*

i.e. a linear holddown RELEASE from the suppression trough up to the burnout
hump, then a linear burnout DECAY at the harvested slope.  Evaluated at the
a-priori mid-cycle point ``b_j = bu_j0 + NOMINAL_CYCLE_BURNUP_MWD_KG / 2`` this
gives the cycle-averaged core reactivity and the core-average decay rate

    rho_bar = sum_j w_j rho_j(b_j) / sum_j w_j              [pcm]
    D       = - sum_j w_j s_j      / sum_j w_j              [pcm per GWd/tU] > 0

and the leading-order cycle burnup is excess reactivity over loss rate:

    B_cycle = (rho_bar - RHO_LEAK) / D                      [GWd/tU]

Finally the (single, global) unit conversion to EFPD is a two-parameter affine
map fitted by least squares on **train-split rows only**:

    cyclen_prior = alpha * B_cycle + beta                   [EFPD]

Assumptions (all deliberate, all leading-order)
-----------------------------------------------
1. **Zero-dimensional balance.** The core is treated as a single node: the
   multiplicity-weighted mean reactivity must overcome one lumped leakage +
   control term.  Spatial power shape, radial leakage gradients and the
   burnup/flux feedback that redistributes them are exactly what the residual
   network is left to learn.
2. **Piecewise-linear depletion.** The harvested least-squares
   ``depletion_slope_pcm_per_gwd`` is taken as constant over the burnout region,
   and the holddown release from dip to peak as linear in burnup.
3. **A-priori burn state.** ``bu_j0`` uses the nominal per-cycle burnup constant
   (:data:`~.featurize.NOMINAL_CYCLE_BURNUP_MWD_KG`), never the record's own
   ``cycle_burnup`` label — the prior obeys the same leakage rule as the
   featurizer, so it can be computed for an unlabelled served pattern.
4. **Constant lumped leakage.** ``RHO_LEAK`` is one fixed pcm constant for the
   APR1400 geometry.  Because ``D`` varies little across cores (every harvested
   slope sits in -650..-340 pcm/GWd), an error in ``RHO_LEAK`` is very nearly a
   constant shift in ``B_cycle`` and is therefore absorbed by ``beta``.
5. **Global unit conversion.** ``(alpha, beta)`` are two scalars fit ONCE on the
   train split.  They carry the tU-per-MWth conversion and the systematic offset
   of the zero-dimensional balance; they are NOT per-cell, so the prior cannot
   launder per-cell label information into the model.
6. **Graceful degradation.** A slot whose origin type resolves to no harvested
   curve contributes no weight.  A pattern where NO slot resolves has an
   undefined ``B_cycle`` and falls back to the fitted train-mean cyclen, so the
   prior is always finite and the residual round-trip is always exact.

The prior is a pure function of ``(pattern, feed, library)`` + the static fuel
table, so ``predict() = prior + residual`` reconstructs absolute cyclen at serve
time with no label access.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.fuel_types import FuelLibrary, FuelVec
from ..data.schema import unpack_pattern
from ..vendor.masterrl.domain import SLOTS
from .featurize import NOMINAL_CYCLE_BURNUP_MWD_KG, FeatureEncoder, RecordInputs

#: filename of the persisted prior artifact inside a model dir.
PRIOR_NAME = "cyclen_physics_prior.json"
#: schema tag stamped into the artifact (bump on a breaking format change).
PRIOR_SCHEMA = "cyclen_physics_prior_v1"

#: Lumped leakage + control reactivity the core-average must overcome [pcm].
#: A leading-order APR1400 value; see assumption 4 — ``beta`` absorbs the bulk of
#: any error in it, so it is a *scale setter*, not a tuned parameter.
RHO_LEAK_PCM: float = 3500.0
#: Fallback burnout slope for an origin whose curve carries no harvested slope
#: [pcm per GWd/tU] — the harvested population median (see kinf_shape_features.md).
DEFAULT_SLOPE_PCM_PER_GWD: float = -600.0
#: Plausible cycle-burnup band; a solve outside it is clamped (a degenerate core
#: must never emit a wild prior that the residual head then has to undo).
_B_LO, _B_HI = 0.0, 60.0
#: The a-priori mid-cycle evaluation point [GWd/tU] (assumption 3).
_BU_MID_OFFSET: float = NOMINAL_CYCLE_BURNUP_MWD_KG / 2.0


def _f(value: Any) -> float:
    """``float(value)`` with None / non-numeric / NaN collapsing to NaN."""
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def rho_pcm(kinf: float) -> float:
    """``(k - 1) / k * 1e5`` [pcm]; NaN for a non-positive / absent k."""
    k = _f(kinf)
    if not math.isfinite(k) or k <= 0.0:
        return float("nan")
    return (k - 1.0) / k * 1.0e5


def assembly_rho(vec: FuelVec | None, burnup_gwd: float) -> tuple[float, float]:
    """``(rho(burnup), slope)`` [pcm, pcm/GWd] for one assembly's origin type.

    Implements the two-segment curve of the module docstring.  Returns
    ``(nan, nan)`` when the origin carries no harvested reference curve at all
    (``kinf_peak`` and ``kinf0`` both absent) — the caller drops that slot's
    weight rather than inventing a reactivity for it.
    """
    if vec is None:
        return float("nan"), float("nan")
    peak_k = _f(getattr(vec, "kinf_peak", None))
    if not math.isfinite(peak_k):
        # A row with no hump analysis but a harvested BOC point still pins the
        # curve: for a monotone (weak-absorber) design the peak IS kinf0.
        peak_k = _f(getattr(vec, "kinf0", None))
    rho_peak = rho_pcm(peak_k)
    if not math.isfinite(rho_peak):
        return float("nan"), float("nan")

    slope = _f(getattr(vec, "depletion_slope_pcm_per_gwd", None))
    if not math.isfinite(slope):
        slope = DEFAULT_SLOPE_PCM_PER_GWD

    bu_peak = _f(getattr(vec, "bu_peak_gwd", None))
    if not math.isfinite(bu_peak) or bu_peak < 0.0:
        bu_peak = 0.0
    swing = _f(getattr(vec, "reactivity_swing_pcm", None))
    if not math.isfinite(swing) or swing < 0.0:
        swing = 0.0                     # monotone curve: nothing held down

    b = float(burnup_gwd)
    if b >= bu_peak or bu_peak <= 0.0:
        rho = rho_peak + slope * (b - bu_peak)
    else:
        rho = rho_peak - swing * (1.0 - b / bu_peak)
    return rho, slope


def cycle_burnup_estimate(inputs: RecordInputs, fuel: FuelLibrary,
                          *, rho_leak: float = RHO_LEAK_PCM,
                          encoder: FeatureEncoder | None = None,
                          vec_cache: dict | None = None) -> float:
    """``B_cycle`` [GWd/tU] for one pattern by the leading-order balance.

    Pure function of ``(pattern, library)`` + the static fuel table + the
    a-priori residence-age burnup — the same leakage-safe input surface the
    featurizer uses, so this is computable for an unlabelled served pattern.
    Returns NaN when no slot resolves to a harvested curve.
    """
    enc = encoder or _shared_encoder()
    items = unpack_pattern(inputs.pattern).items
    age, origin, _ = enc._trace_chain(items)

    num_rho = 0.0
    num_slope = 0.0
    wsum = 0.0
    for slot in SLOTS:
        j = slot.index
        vec = _resolve(fuel, items[origin[j]].batch, inputs.library_id, vec_cache)
        bu = (age[j] - 1) * NOMINAL_CYCLE_BURNUP_MWD_KG + _BU_MID_OFFSET
        rho, slope = assembly_rho(vec, bu)
        if not (math.isfinite(rho) and math.isfinite(slope)):
            continue
        w = float(slot.multiplicity)
        num_rho += w * rho
        num_slope += w * slope
        wsum += w
    if wsum <= 0.0:
        return float("nan")
    rho_bar = num_rho / wsum
    decay = -(num_slope / wsum)                       # > 0 pcm per GWd/tU
    if not (math.isfinite(rho_bar) and math.isfinite(decay)) or decay <= 1.0e-6:
        return float("nan")
    b_cycle = (rho_bar - float(rho_leak)) / decay
    return float(min(max(b_cycle, _B_LO), _B_HI))


def _shared_encoder() -> FeatureEncoder:
    """A module-level encoder used only for its (stateless) chain tracing."""
    global _ENCODER
    try:
        return _ENCODER
    except NameError:
        _ENCODER = FeatureEncoder()
        return _ENCODER


def _resolve(fuel: FuelLibrary, type_id: str, library_id: str,
             cache: dict | None) -> FuelVec | None:
    key = (library_id, type_id)
    if cache is not None and key in cache:
        return cache[key]
    from .featurize import _depad

    vec: FuelVec | None
    try:
        vec = fuel.get(type_id, library_id)
    except KeyError:
        alt = _depad(type_id)
        vec = None
        if alt != type_id:
            try:
                vec = fuel.get(alt, library_id)
            except KeyError:
                vec = None
    if cache is not None:
        cache[key] = vec
    return vec


def cycle_burnup_batch(rows: Any, fuel: FuelLibrary, *,
                       rho_leak: float = RHO_LEAK_PCM) -> np.ndarray:
    """``B_cycle`` for every row of a store frame / RecordInputs sequence."""
    enc = _shared_encoder()
    cache: dict = {}
    out: list[float] = []
    it = rows.iterrows() if hasattr(rows, "iterrows") else enumerate(rows)
    for _, row in it:
        inp = RecordInputs.coerce(row)
        out.append(cycle_burnup_estimate(inp, fuel, rho_leak=rho_leak,
                                         encoder=enc, vec_cache=cache))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# fitted artifact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CyclenPhysicsPrior:
    """The fitted ``cyclen_prior = alpha * B_cycle + beta`` unit conversion.

    ``fallback_cyclen`` is the train-split mean cyclen, returned whenever
    ``B_cycle`` is undefined (assumption 6) so the prior is ALWAYS finite and
    ``prior + residual`` is an exact round-trip for every row.
    """

    alpha: float
    beta: float
    rho_leak: float = RHO_LEAK_PCM
    fallback_cyclen: float = 0.0
    n_fit: int = 0
    pearson: float = float("nan")
    spearman: float = float("nan")
    split: str | None = None
    schema: str = PRIOR_SCHEMA

    # -- evaluation ------------------------------------------------------- #
    def from_b(self, b_cycle: np.ndarray | float) -> np.ndarray:
        """Map ``B_cycle`` [GWd/tU] -> prior cyclen [EFPD] (NaN -> fallback)."""
        b = np.asarray(b_cycle, dtype=float)
        out = self.alpha * b + self.beta
        return np.where(np.isfinite(out), out, float(self.fallback_cyclen))

    def for_rows(self, rows: Any, fuel: FuelLibrary) -> np.ndarray:
        """Prior cyclen [EFPD] for every row of a store frame / inputs sequence."""
        return self.from_b(cycle_burnup_batch(rows, fuel, rho_leak=self.rho_leak))

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CyclenPhysicsPrior":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in known})

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CyclenPhysicsPrior":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).correlation
        return float(r) if r is not None else float("nan")
    except Exception:                                   # pragma: no cover
        return float("nan")


def fit_cyclen_prior(df: Any, fuel: FuelLibrary, *,
                     rho_leak: float = RHO_LEAK_PCM,
                     split: str | None = None,
                     b_cycle: np.ndarray | None = None) -> CyclenPhysicsPrior:
    """Fit ``(alpha, beta)`` on **train-split rows only** (leakage rule).

    ``df`` must already be restricted to the train fold — this function does not
    (and cannot) know the split; the caller in ``train.py`` passes the train
    frame, and the leakage guard is asserted there.  Only converged rows with a
    finite ``cyclen`` label and a finite ``B_cycle`` enter the least squares.
    A degenerate fit (no usable rows, or a constant ``B_cycle``) falls back to
    ``alpha = 0`` — the prior then degenerates to the train-mean constant, which
    keeps the residual round-trip exact and simply reduces the prior to a shift.
    """
    import pandas as pd

    if b_cycle is None:
        b_cycle = cycle_burnup_batch(df, fuel, rho_leak=rho_leak)
    b = np.asarray(b_cycle, dtype=float)
    y = pd.to_numeric(df["cyclen"], errors="coerce").to_numpy(dtype=float)
    conv = (df["converged"].astype(bool).to_numpy()
            if "converged" in df.columns else np.ones(len(y), dtype=bool))
    ok = conv & np.isfinite(y) & np.isfinite(b)
    n = int(ok.sum())
    mean_y = float(y[ok].mean()) if n else 0.0
    if n < 2 or float(np.ptp(b[ok])) < 1e-9:
        return CyclenPhysicsPrior(
            alpha=0.0, beta=mean_y, rho_leak=float(rho_leak),
            fallback_cyclen=mean_y, n_fit=n, split=split)
    alpha, beta = np.polyfit(b[ok], y[ok], 1)
    r = float(np.corrcoef(b[ok], y[ok])[0, 1])
    return CyclenPhysicsPrior(
        alpha=float(alpha), beta=float(beta), rho_leak=float(rho_leak),
        fallback_cyclen=mean_y, n_fit=n, pearson=r,
        spearman=_spearman(b[ok], y[ok]), split=split)


def prior_correlation(prior: CyclenPhysicsPrior, df: Any,
                      fuel: FuelLibrary) -> dict[str, float]:
    """Pearson / Spearman of the prior against ACTUAL cyclen on a held-out frame.

    The honest read on "does the physics prior carry real signal": call it with
    the val fold, never the fold the prior was fitted on.
    """
    import pandas as pd

    p = prior.for_rows(df, fuel)
    y = pd.to_numeric(df["cyclen"], errors="coerce").to_numpy(dtype=float)
    conv = (df["converged"].astype(bool).to_numpy()
            if "converged" in df.columns else np.ones(len(y), dtype=bool))
    ok = conv & np.isfinite(y) & np.isfinite(p)
    if int(ok.sum()) < 2:
        return {"n": 0, "pearson": float("nan"), "spearman": float("nan"),
                "mae": float("nan")}
    return {
        "n": int(ok.sum()),
        "pearson": float(np.corrcoef(p[ok], y[ok])[0, 1]),
        "spearman": _spearman(p[ok], y[ok]),
        "mae": float(np.mean(np.abs(p[ok] - y[ok]))),
    }


__all__ = [
    "PRIOR_NAME",
    "PRIOR_SCHEMA",
    "RHO_LEAK_PCM",
    "CyclenPhysicsPrior",
    "assembly_rho",
    "cycle_burnup_batch",
    "cycle_burnup_estimate",
    "fit_cyclen_prior",
    "prior_correlation",
    "rho_pcm",
]
