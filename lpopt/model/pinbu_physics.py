"""Physics pin-burnup estimator (serve-side, no retrain — plan 12.4 addendum).

The champion's raw ``max_pin_burnup`` head does not generalize within-cell
out-of-sample (data/reports/pinbu_forensics.md): pin burnup is a single-assembly
discharge extreme, ~93 % rank-tied to the peak assembly's EOL assembly-average
burnup, and the head's dominant teacher (Dataset A) carries a *fake* surrogate
label ``~1.08 x assembly`` whereas the real MAS_PPI labels (Dataset P) peak at
``~1.18 x assembly`` — a different physical definition.  The head therefore
collapses toward the mean and *under-predicts the magnitude* of the licensing
pin-burnup constraint axis.

This module reconstructs the pin-burnup **magnitude** from physics instead of
from the weak head, keeping the head's weights untouched (the serve hook only
overrides the served column-6 mean when enabled):

    pin_bu  =  a * [ ratio_type(B_asm) * B_asm ]  +  b
               \\_______ per-type lattice physics _______/   \\__ P-fit __/

* ``ratio_type(B_asm)`` is the per-fuel-type peak-pin-to-assembly burnup ratio
  curve (BRP/BU), harvested into ``fuel_types`` from DeCART ``.sum`` EDIT3 (with
  an HGC ``%DIST`` map7 fallback for ga80) — the physics backbone that
  generalizes to arbitrary burnable-poison / zoning designs.
* ``B_asm`` (the peak assembly's EOL assembly-average burnup) is reconstructed at
  serve time from the **strong** cyclen prediction via the energy-balance
  discharge estimate and a per-feed peak-assembly burnup factor ``k_peak`` fit on
  Dataset P.
* the affine ``(a, b)`` absorbs the residual 2-D-lattice -> 3-D-MAS_PPI definition
  gap; it is fit on the champion split's **train-fold** Dataset-P rows only (the
  honest holdout never enters the fit — same leakage guarantee as
  ``cell_calibrate``) and persisted as ``pinbu_physics.json`` next to the
  champion, exactly like ``cell_calibration.json``.

Honesty note (per the forensics): a per-type feature is *constant within a fixed
cell* (the fuel pair is fixed), so this estimator recalibrates the pin-burnup
**scale / cross-cell magnitude** — it does not, and cannot, improve the
within-cell *ranking* of the raw head.  It is enable-flagged and defaults to a
no-op when no artifact is present, so the deployed champion is unaffected.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.fuel_types import (
    FuelLibrary, FuelVec, PIN_BU_COLUMNS, resolve_type_id,
)

#: filename of the persisted artifact inside a model dir (mirrors CELL_CALIB_NAME).
PINBU_PHYSICS_NAME = "pinbu_physics.json"
#: schema tag stamped into the artifact (bump on a breaking format change).
PINBU_PHYSICS_SCHEMA = "pinbu_physics_affine_v1"
#: surrogate/target column the estimate overrides (== max_pin_burnup, model_api).
PINBU_SURROGATE_COL = 6
#: APR1400 defaults (config CriteriaConfig) used by the energy-balance estimate.
DEFAULT_POWER_MW = 3983.0
DEFAULT_HM_MTU = 104.8
#: minimum labelled train rows (converged P w/ pin + assembly + cyclen) to fit.
DEFAULT_MIN_ROWS = 30
#: fallback peak-pin-to-assembly ratio for a type with no harvested curve.
DEFAULT_RATIO = 1.05
#: plausible affine-slope range; a fit outside is rejected for the ratio fallback.
_SLOPE_LO, _SLOPE_HI = 0.5, 2.5


def _finite(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


# --------------------------------------------------------------------------- #
# per-type peaking-ratio curve
# --------------------------------------------------------------------------- #
class PinBuRatioCurve:
    """Reconstructed ``ratio(BU) = BRP/BU`` curve for one fuel type.

    Built from the harvested :data:`fuel_types.PIN_BU_COLUMNS` summary: the
    discharge-tail fit ``r_inf + paramA/BU`` is the primary model, clamped to the
    directly-observed plateau ``ratio_asym`` as a floor (the 1/BU extrapolation
    keeps declining past the last harvested state, which is unphysical) and to the
    BOC power form factor ``ff_pin_max`` as an upper bound.  Degrades to a constant
    (plateau, else :data:`DEFAULT_RATIO`) when the fit columns are absent.
    """

    __slots__ = ("r_inf", "paramA", "ratio_asym", "bu_max", "ff_pin_max")

    def __init__(self, r_inf: float | None, paramA: float | None,
                 ratio_asym: float | None, bu_max: float | None,
                 ff_pin_max: float | None) -> None:
        self.r_inf = _opt(r_inf)
        self.paramA = _opt(paramA)
        self.ratio_asym = _opt(ratio_asym)
        self.bu_max = _opt(bu_max)
        self.ff_pin_max = _opt(ff_pin_max)

    @classmethod
    def from_vec(cls, vec: FuelVec | None) -> "PinBuRatioCurve":
        if vec is None:
            return cls(None, None, None, None, None)
        return cls(
            getattr(vec, "pin_bu_r_inf", None),
            getattr(vec, "pin_bu_paramA", None),
            getattr(vec, "pin_bu_ratio_asym", None),
            getattr(vec, "pin_bu_bu_max", None),
            getattr(vec, "ff_pin_max", None),
        )

    @property
    def harvested(self) -> bool:
        """True when at least the plateau or the tail fit is available."""
        return self.ratio_asym is not None or (
            self.r_inf is not None and self.paramA is not None
        )

    def ratio_at(self, bu: float) -> float:
        """Peak-pin-to-assembly burnup ratio at assembly-average burnup ``bu``."""
        b = float(bu)
        if not math.isfinite(b) or b <= 0.0:
            return self.ratio_asym or self.r_inf or DEFAULT_RATIO
        if self.r_inf is not None and self.paramA is not None:
            r = self.r_inf + self.paramA / b
            if self.ratio_asym is not None:
                r = max(r, self.ratio_asym)          # plateau is the physical floor
            if self.ff_pin_max is not None:
                r = min(r, self.ff_pin_max)          # never exceed the BOC peaking
            return r
        if self.ratio_asym is not None:
            return self.ratio_asym
        return DEFAULT_RATIO

    def pin_bu(self, bu: float) -> float:
        """Physics peak-pin burnup at assembly-average burnup ``bu``."""
        return self.ratio_at(bu) * float(bu)


def _opt(v: Any) -> float | None:
    f = _finite(v)
    return None if math.isnan(f) else f


# --------------------------------------------------------------------------- #
# peak-assembly type resolution
# --------------------------------------------------------------------------- #
def resolve_peak_curve(fuel: FuelLibrary, library_id: str,
                       batch_feed: Mapping[str, int]) -> PinBuRatioCurve:
    """Curve of the pattern's most-peaking fresh type (conservative peak choice).

    The peak-burnup assembly is the pin-burnup-limiting one; among the pattern's
    fresh types we pick the harvested curve with the largest plateau ratio
    (``ratio_asym``), which is the physically conservative upper bound for a
    licensing extreme and generalizes across designs.  Falls back to a
    default-ratio curve when no fed type resolves to a harvested row.
    """
    best: PinBuRatioCurve | None = None
    best_key = -1.0
    for raw in batch_feed:
        tid = resolve_type_id(fuel, library_id, str(raw))
        if tid is None:
            continue
        try:
            vec = fuel.get(tid, library_id)
        except KeyError:
            continue
        curve = PinBuRatioCurve.from_vec(vec)
        if not curve.harvested:
            continue
        rank = curve.ratio_asym if curve.ratio_asym is not None else curve.ratio_at(70.0)
        if rank > best_key:
            best_key = rank
            best = curve
    return best if best is not None else PinBuRatioCurve(None, None, None, None, None)


# --------------------------------------------------------------------------- #
# energy-balance assembly-burnup reconstruction
# --------------------------------------------------------------------------- #
def core_discharge_estimate(cyclen: float, feed: int, *, power_mw: float,
                            hm_mtu: float) -> float:
    """Energy-balance core-average equilibrium discharge burnup [GWd/tU].

    Thin wrapper over :func:`design.bootstrap.estimate_discharge_burnup` (lazy
    import) so the estimator does not pull the bootstrap subprocess machinery at
    module import time.
    """
    from ..design.bootstrap import estimate_discharge_burnup

    return estimate_discharge_burnup(
        float(cyclen), int(feed), power_mw=float(power_mw), hm_mtu=float(hm_mtu)
    )


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
class PinBuPhysicsEstimator:
    """Serve-side pin-burnup estimator built from a fitted artifact + fuel table."""

    def __init__(self, fuel: FuelLibrary, *, library_id: str,
                 a: float, b: float, global_k_peak: float,
                 k_peak_by_feed: Mapping[Any, float] | None = None,
                 power_mw: float = DEFAULT_POWER_MW,
                 hm_mtu: float = DEFAULT_HM_MTU) -> None:
        self.fuel = fuel
        self.library_id = str(library_id)
        self.a = float(a)
        self.b = float(b)
        self.global_k_peak = float(global_k_peak)
        self.k_peak_by_feed: dict[int, float] = {
            int(k): float(v) for k, v in (k_peak_by_feed or {}).items()
        }
        self.power_mw = float(power_mw)
        self.hm_mtu = float(hm_mtu)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any],
                      fuel: FuelLibrary) -> "PinBuPhysicsEstimator":
        by_feed = {
            int(k): float(v["k_peak"])
            for k, v in (artifact.get("k_peak_by_feed") or {}).items()
        }
        return cls(
            fuel,
            library_id=str(artifact.get("library_id", "ga80")),
            a=float(artifact["a"]),
            b=float(artifact["b"]),
            global_k_peak=float(artifact["global_k_peak"]),
            k_peak_by_feed=by_feed,
            power_mw=float(artifact.get("power_mw", DEFAULT_POWER_MW)),
            hm_mtu=float(artifact.get("hm_mtu", DEFAULT_HM_MTU)),
        )

    # -- physics chain ------------------------------------------------------ #
    def k_peak(self, feed: int) -> float:
        return self.k_peak_by_feed.get(int(feed), self.global_k_peak)

    def assembly_burnup(self, cyclen: float, feed: int) -> float:
        """Peak-assembly EOL assembly-average burnup from the cyclen prediction."""
        b_core = core_discharge_estimate(
            cyclen, feed, power_mw=self.power_mw, hm_mtu=self.hm_mtu)
        return self.k_peak(feed) * b_core

    def apply_affine(self, physics_pin: float) -> float:
        return self.a * float(physics_pin) + self.b

    def estimate(self, batch_feed: Mapping[str, int], feed: int,
                 cyclen: float) -> float:
        """Physics pin-burnup for a candidate from its cyclen + fresh types.

        Returns ``NaN`` when ``cyclen`` is non-finite so the caller keeps the raw
        head value.
        """
        c = _finite(cyclen)
        if math.isnan(c):
            return float("nan")
        curve = resolve_peak_curve(self.fuel, self.library_id, batch_feed)
        b_asm = self.assembly_burnup(c, feed)
        if not math.isfinite(b_asm) or b_asm <= 0.0:
            return float("nan")
        return self.apply_affine(curve.pin_bu(b_asm))

    def estimate_from_assembly(self, batch_feed: Mapping[str, int],
                               assembly_burnup: float) -> float:
        """Physics pin-burnup from a KNOWN assembly-average burnup (fit path)."""
        curve = resolve_peak_curve(self.fuel, self.library_id, batch_feed)
        return self.apply_affine(curve.pin_bu(assembly_burnup))


# --------------------------------------------------------------------------- #
# robust affine
# --------------------------------------------------------------------------- #
def _robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    """``y ~ a*x + b``: robust Huber slope if stable, else a ratio-only fit.

    Falls back to ``a = median(y/x), b = 0`` (a pure scale correction) when the
    slope is degenerate or outside :data:`_SLOPE_LO`/:data:`_SLOPE_HI` — the
    definition gap is fundamentally a scale factor, so a pinned-through-origin
    ratio is the safe default.
    """
    if x.size >= 2 and float(np.ptp(x)) > 1e-9:
        try:
            from sklearn.linear_model import HuberRegressor

            h = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=400)
            h.fit(x.reshape(-1, 1), y)
            a, b = float(h.coef_[0]), float(h.intercept_)
            if math.isfinite(a) and math.isfinite(b) and _SLOPE_LO <= a <= _SLOPE_HI:
                return a, b, "affine"
        except Exception:      # pragma: no cover - sklearn edge / non-convergence
            pass
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = y / x
    ratios = ratios[np.isfinite(ratios)]
    a = float(np.median(ratios)) if ratios.size else 1.0
    return a, 0.0, "ratio"


# --------------------------------------------------------------------------- #
# fit (train-fold Dataset-P only) + persist
# --------------------------------------------------------------------------- #
def fit_pinbu_physics(
    model_dir: str | Path,
    store_dir: str | Path,
    splits_dir: str | Path,
    *,
    split: str | None = None,
    library_id: str = "ga80",
    power_mw: float = DEFAULT_POWER_MW,
    hm_mtu: float = DEFAULT_HM_MTU,
    min_rows: int = DEFAULT_MIN_ROWS,
    write: bool = True,
    out_name: str = PINBU_PHYSICS_NAME,
) -> dict:
    """Fit + persist the physics pin-burnup correction for a champion.

    Uses the champion split's **train-fold** Dataset-P rows only (converged, with
    finite ``max_pin_burnup`` + ``max_assembly_burnup`` + ``cyclen``, produced under
    the serve ``library_id``) — never the honest holdout (``val``) rows.  Two
    correction pieces are fit and stored in ``model_dir/out_name``:

    * ``k_peak`` per feed = ``median(max_assembly_burnup / core_discharge_estimate)``
      — reconstructs the peak-assembly burnup from the served cyclen at serve time.
    * ``(a, b)`` global affine on ``physics_pin(actual B_asm) -> actual pin`` — the
      residual 2-D-lattice -> 3-D-MAS_PPI definition gap.

    The fit needs no model forward pass (it is a physics calibration on labelled
    store rows), which also keeps it independent of the running champion.
    """
    from ..data.schema import unpack_pattern
    from ..data.store import StoreReader
    from .splits import SplitManifest

    model_dir = Path(model_dir)
    reader = StoreReader(store_dir)
    records = reader.records

    # -- resolve the split the champion trained on ------------------------- #
    if split is None:
        metas = sorted(model_dir.glob("member_*/meta.json"))
        split = "S1"
        if metas:
            split = str(json.loads(metas[0].read_text(encoding="utf-8"))
                        .get("split", "S1"))
    manifest = SplitManifest.from_json(Path(splits_dir) / f"{split}.json")
    train_ids = set(manifest.record_ids("train"))
    val_ids = set(manifest.record_ids("val"))

    fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")

    # -- eligible train-fold Dataset-P rows -------------------------------- #
    mask = fit_row_mask(records, train_ids, library_id=library_id)
    sub = records[mask].copy()

    physics_pin: list[float] = []
    pin_actual: list[float] = []
    kpeak_rows: list[tuple[int, float]] = []
    used_rids: list[str] = []
    for _, row in sub.iterrows():
        cyclen = _finite(row["cyclen"])
        b_asm = _finite(row["max_assembly_burnup"])
        pin = _finite(row["max_pin_burnup"])
        feed = int(row["feed"])
        if math.isnan(cyclen) or math.isnan(b_asm) or math.isnan(pin) or b_asm <= 0:
            continue
        b_core = core_discharge_estimate(cyclen, feed, power_mw=power_mw, hm_mtu=hm_mtu)
        if not math.isfinite(b_core) or b_core <= 0:
            continue
        try:
            bf = unpack_pattern(str(row["pattern"])).batch_feed()
        except Exception:      # noqa: BLE001 - a malformed pattern just drops the row
            continue
        curve = resolve_peak_curve(fuel, library_id, bf)
        physics_pin.append(curve.pin_bu(b_asm))
        pin_actual.append(pin)
        kpeak_rows.append((feed, b_asm / b_core))
        used_rids.append(str(row["record_id"]))

    n = len(pin_actual)
    if n < int(min_rows):
        raise ValueError(
            f"pinbu physics fit: only {n} eligible train-fold P rows "
            f"(need >= {min_rows}); is fuel_types harvested + dataset P present?"
        )

    x = np.asarray(physics_pin, dtype=float)
    y = np.asarray(pin_actual, dtype=float)
    a, b, estimator = _robust_affine(x, y)

    # -- k_peak per feed + global ------------------------------------------ #
    all_k = np.asarray([k for _, k in kpeak_rows], dtype=float)
    global_k = float(np.median(all_k))
    by_feed: dict[str, dict] = {}
    feeds = sorted({f for f, _ in kpeak_rows})
    for f in feeds:
        ks = np.asarray([k for ff, k in kpeak_rows if ff == f], dtype=float)
        by_feed[str(f)] = {"k_peak": float(np.median(ks)), "n": int(ks.size)}

    pred = a * x + b
    mae_after = float(np.mean(np.abs(pred - y)))
    mae_before = float(np.mean(np.abs(x - y)))          # raw physics, no affine

    artifact = {
        "schema": PINBU_PHYSICS_SCHEMA,
        "target": "max_pin_burnup",
        "surrogate_col": PINBU_SURROGATE_COL,
        "model_dir": str(model_dir),
        "split": split,
        "library_id": library_id,
        "power_mw": float(power_mw),
        "hm_mtu": float(hm_mtu),
        "a": float(a),
        "b": float(b),
        "estimator": estimator,
        "global_k_peak": global_k,
        "k_peak_by_feed": by_feed,
        "n_fit": int(n),
        "mae_before_gwd": round(mae_before, 4),
        "mae_after_gwd": round(mae_after, 4),
        # mean actual-pin / physics-pin ratio == the 2-D->3-D definition scale gap.
        "definition_scale_gap": round(float(np.mean(y) / np.mean(x)), 4),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # leakage guard: the fit must never have touched a holdout (val) row.
    assert not (set(used_rids) & val_ids), \
        "pinbu physics fit set intersects the honest holdout (val) fold"

    if write:
        _atomic_write_json(model_dir / out_name, artifact)
    return artifact


def fit_row_mask(records: "Any", train_ids: set[str],
                 library_id: str | None = None) -> np.ndarray:
    """Boolean mask of rows eligible for the pin-burnup physics fit.

    A row enters iff its ``record_id`` is a **train** id (never a holdout — the
    leakage guarantee), its ``dataset`` is ``P`` (the only fidelity-consistent
    MAS_PPI pin labels), it converged, it carries finite ``max_pin_burnup`` +
    ``max_assembly_burnup`` + ``cyclen``, and (when given) it was produced under the
    serve ``library_id``.  A small pure function so the invariant is unit-testable
    without loading a store.
    """
    import pandas as pd

    rid = records["record_id"].astype(str)
    mask = rid.isin(train_ids).to_numpy()
    if "dataset" in records.columns:
        mask = mask & (records["dataset"].astype(str) == "P").to_numpy()
    if "converged" in records.columns:
        mask = mask & records["converged"].astype(bool).to_numpy()
    for col in ("max_pin_burnup", "max_assembly_burnup", "cyclen"):
        mask = mask & pd.to_numeric(records[col], errors="coerce").notna().to_numpy()
    if library_id is not None and "library_id" in records.columns:
        mask = mask & (records["library_id"].astype(str) == str(library_id)).to_numpy()
    return mask


# --------------------------------------------------------------------------- #
# load / apply
# --------------------------------------------------------------------------- #
def load_pinbu_physics(path: str | Path) -> dict:
    """Load a ``pinbu_physics.json`` artifact (full dict, incl. metadata)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path: str | Path, obj: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


__all__ = [
    "DEFAULT_HM_MTU",
    "DEFAULT_POWER_MW",
    "DEFAULT_RATIO",
    "PINBU_PHYSICS_NAME",
    "PINBU_PHYSICS_SCHEMA",
    "PINBU_SURROGATE_COL",
    "PinBuPhysicsEstimator",
    "PinBuRatioCurve",
    "core_discharge_estimate",
    "fit_pinbu_physics",
    "fit_row_mask",
    "load_pinbu_physics",
    "resolve_peak_curve",
]
