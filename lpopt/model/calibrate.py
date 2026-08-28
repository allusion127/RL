"""Uncertainty calibration for the PosValNet ensemble (plan sec. 4.4).

Per-target **isotonic** regression is fit ONLY on the S1-val ensemble residuals
— mapping the predicted *total* σ (``sqrt(mean aleatoric² + var of member μ)``)
to the observed error magnitude — and a **Platt** logistic maps the mean
convergence logit to a calibrated probability.  The fit is persisted alongside
the checkpoints as plain arrays (``calibration.json`` — no pickled sklearn
objects, matching the state_dict+meta portability rule), and
:func:`apply_calibration` turns a raw predicted σ into the calibrated σ.

A Gaussian's mean absolute error is ``σ·sqrt(2/π)``, so the isotonic target is
``|residual| / sqrt(2/π)`` — the fitted curve then reads directly as a
calibrated standard deviation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .dataset_torch import LPDataset, TARGETS
from .featurize import DEFAULT_COND_SCHEMA, FeatureEncoder
from .train import (
    DEFAULT_SPLITS,
    DEFAULT_STORE,
    denormalize,
    load_member,
    norm_from_meta,
    predict_dataset,
    _load_split,
)

_MAE_TO_STD = math.sqrt(2.0 / math.pi)      # E[|N(0,σ)|] = σ·sqrt(2/π)
CALIB_NAME = "calibration.json"


# --------------------------------------------------------------------------- #
# total-σ assembly from ensemble outputs
# --------------------------------------------------------------------------- #
def ensemble_stats(pred: dict[str, np.ndarray], tmean: np.ndarray, tstd: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw-space ``(mean[N,T], epistemic[N,T], total_sigma[N,T])``.

    ``T`` is the target count (5 for cond_v2, 7 for cond_v3); this function is
    target-count agnostic (it operates on the ensemble member arrays as-is).

    * ``mean``       — mean of member μ.
    * ``epistemic``  — std of member μ across the ensemble.
    * ``total``      — ``sqrt(mean aleatoric² + var of member μ)``.
    """
    mu_z = pred["mu_z_members"]                          # [M,N,5]
    mean_raw = denormalize(mu_z.mean(axis=0), tmean, tstd)
    # member means in raw units for the epistemic spread
    members_raw = mu_z * tstd[None, None, :] + tmean[None, None, :]
    epistemic = members_raw.std(axis=0)                  # [N,5]
    # aleatoric σ (raw) from log_sigma (z-space) -> * tstd
    alea_raw = np.exp(pred["log_sigma_members"]) * tstd[None, None, :]
    mean_alea_sq = (alea_raw ** 2).mean(axis=0)          # [N,5]
    total = np.sqrt(mean_alea_sq + epistemic ** 2)
    return mean_raw, epistemic, total


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def _fit_isotonic(sigma: np.ndarray, abs_err: np.ndarray) -> dict[str, list[float]]:
    """Monotone σ_pred -> σ_cal via isotonic regression (identity on degenerate)."""
    y = abs_err / _MAE_TO_STD
    finite = np.isfinite(sigma) & np.isfinite(y)
    sigma, y = sigma[finite], y[finite]
    if sigma.size < 10 or float(np.ptp(sigma)) < 1e-9:
        hi = float(max(1.0, np.nanmax(sigma) if sigma.size else 1.0))
        return {"x": [0.0, hi], "y": [0.0, hi]}          # identity passthrough
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(sigma, y)
        x = np.asarray(iso.X_thresholds_, dtype=float)
        yy = np.asarray(iso.y_thresholds_, dtype=float)
    except Exception:      # pragma: no cover
        order = np.argsort(sigma)
        x = sigma[order]
        yy = np.maximum.accumulate(y[order])
    # ensure strictly usable interp grid
    return {"x": [float(v) for v in x], "y": [float(v) for v in yy]}


def _fit_platt(mean_logit: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """Platt logistic: p_cal = sigmoid(a·logit + b)."""
    finite = np.isfinite(mean_logit) & np.isfinite(label)
    x, y = mean_logit[finite], label[finite]
    if x.size < 10 or len(np.unique(y)) < 2:
        return {"coef": 1.0, "intercept": 0.0, "degenerate": True}
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x.reshape(-1, 1), y.astype(int))
        return {"coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
                "degenerate": False}
    except Exception:      # pragma: no cover
        return {"coef": 1.0, "intercept": 0.0, "degenerate": True}


def fit_calibration(
    ckpt_dirs: Sequence[str | Path],
    *,
    split: str = "S1",
    device: str | torch.device = "cpu",
    out_dir: str | Path,
    store_dir: str | Path = DEFAULT_STORE,
    splits_dir: str | Path = DEFAULT_SPLITS,
) -> Path:
    """Fit isotonic (per target) + Platt (conv) on the ``split`` val fold."""
    from ..data.fuel_types import FuelLibrary
    from ..data.store import StoreReader

    device = torch.device(device)
    members = []
    metas = []
    for d in ckpt_dirs:
        m, meta = load_member(d, device)
        members.append(m)
        metas.append(meta)
    tmean, tstd = norm_from_meta(metas[0])

    reader = StoreReader(store_dir)
    fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
    manifest = _load_split(reader, split, splits_dir, seed=0)
    # the members' schema decides the channel inventory (v4 = 43ch); a default
    # v3 encoder here would shape-mismatch the stem on a v4 ensemble
    encoder = FeatureEncoder(
        cond_schema=str(metas[0].get("cond_schema", DEFAULT_COND_SCHEMA)))
    # calibration fits on the FULL label set (censoring is a training-loss
    # concern only); Dataset-A pin residuals stay in the isotonic fit as before.
    val_ds = LPDataset(reader, manifest, fuel, augment=False, fold="val",
                       encoder=encoder, censor_dataset_a_pin_labels=False)

    pred = predict_dataset(members, val_ds, device)
    mean_raw, _epi, total = ensemble_stats(pred, tmean, tstd)
    true = pred["targets"]
    tmask = pred["target_mask"]

    isotonic: dict[str, dict[str, list[float]]] = {}
    for k, name in enumerate(TARGETS):
        sel = tmask[:, k] > 0
        abs_err = np.abs(mean_raw[sel, k] - true[sel, k])
        isotonic[name] = _fit_isotonic(total[sel, k], abs_err)

    # Platt on the mean ensemble conv logit (recovered from mean prob).
    conv_prob = pred["conv_prob_members"].mean(axis=0).clip(1e-6, 1 - 1e-6)
    mean_logit = np.log(conv_prob / (1 - conv_prob))
    cmask = pred["conv_mask"] > 0
    platt = _fit_platt(mean_logit[cmask], pred["conv_label"][cmask])

    calib = {
        "targets": list(TARGETS),
        "split": split,
        "n_members": len(members),
        "isotonic": isotonic,
        "platt": platt,
        "n_val_used": int(tmask.any(axis=1).sum()),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / CALIB_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(calib, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def load_calibration(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_calibration(total_sigma: np.ndarray, calib: dict[str, Any],
                      target_names: Sequence[str] | None = None) -> np.ndarray:
    """Map raw total σ ``[N,T]`` to calibrated σ via the per-target isotonic curve.

    A target with NO fitted curve keeps its RAW σ (the array is seeded from
    ``total_sigma``, never from uninitialized memory).  This matters whenever the
    ensemble predicts more targets than the calibration artifact covers — e.g. a
    ``promote_max_asm_bu`` checkpoint has 8 model targets while a calibration
    fitted before the promotion (or copied verbatim by the freeze-finetune
    recipe) lists only 7.  The old ``np.empty_like`` + "loop over the artifact's
    targets" served *uninitialized memory* as ``max_assembly_burnup`` σ.

    ``target_names`` (the ensemble's model-target order) makes the mapping
    name-based, so a calibration whose target list is a subset — or in a
    different order — lands on the right columns.  Without it the legacy
    positional mapping is kept (correct when the artifact is a prefix of the
    model targets, which is the historical case).
    """
    total_sigma = np.asarray(total_sigma, dtype=float)
    out = total_sigma.copy()          # unfitted targets pass through RAW
    names = list(calib["targets"])
    if target_names is not None:
        index = {n: i for i, n in enumerate(target_names)}
        pairs = [(index[n], n) for n in names if n in index]
    else:
        pairs = [(k, n) for k, n in enumerate(names) if k < out.shape[1]]
    for k, name in pairs:
        curve = calib["isotonic"][name]
        x = np.asarray(curve["x"], dtype=float)
        y = np.asarray(curve["y"], dtype=float)
        out[:, k] = np.interp(total_sigma[:, k], x, y)
    return out


def apply_platt(mean_logit: np.ndarray, calib: dict[str, Any]) -> np.ndarray:
    """Calibrated convergence probability from the mean ensemble logit."""
    p = calib.get("platt", {"coef": 1.0, "intercept": 0.0})
    z = p["coef"] * np.asarray(mean_logit, dtype=float) + p["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


__all__ = [
    "fit_calibration", "load_calibration", "apply_calibration", "apply_platt",
    "ensemble_stats", "CALIB_NAME",
]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble_dir")
    ap.add_argument("--split", default="S1")
    args = ap.parse_args()
    dirs = sorted(Path(args.ensemble_dir).glob("member_*"))
    path = fit_calibration(dirs, split=args.split, out_dir=args.ensemble_dir)
    print("wrote", path)
