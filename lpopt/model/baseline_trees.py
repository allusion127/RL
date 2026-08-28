"""ExtraTrees regression baseline (plan sec. 4.4).

One :class:`~sklearn.ensemble.ExtraTreesRegressor` per target (mirroring the
vendor surrogate hyperparameters: ``n_estimators=256``, ``min_samples_leaf=2``,
``max_features=0.7``) on a **flattened, cheaper** physics feature vector: the
per-slot channel matrix ``(C, 69)`` (the leakage-safe pre-mirror-expansion
representation from :meth:`FeatureEncoder.encode_slot_matrix`) concatenated with
the ``G`` FiLM globals — ``C*69 + G = 1804`` features.  This deliberately avoids
the 4x mirror redundancy of the ``26x19x19`` CNN grid (documented cheaper
subset).  Same splits and same per-target masks as the CNN.

``run_baseline(split)`` fits on the train fold, scores the val fold, and persists
a metrics JSON.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.fuel_types import FuelLibrary
from ..data.store import StoreReader
from .dataset_torch import TARGETS
from .featurize import FeatureEncoder, RecordInputs
from .splits import SplitManifest
from .train import _load_split, within_case_spearman

DEFAULT_STORE = "data/store"
DEFAULT_SPLITS = "data/splits"
DEFAULT_REPORTS = "data/reports"


# --------------------------------------------------------------------------- #
# feature builder
# --------------------------------------------------------------------------- #
def build_flat_features(df: pd.DataFrame, fuel: FuelLibrary,
                        encoder: FeatureEncoder | None = None) -> np.ndarray:
    """Flatten ``(C,69)`` slot matrix + ``G`` globals for every row -> ``[N, F]``."""
    enc = encoder or FeatureEncoder()
    rows: list[np.ndarray] = []
    for _, row in df.iterrows():
        inp = RecordInputs.coerce(row)
        slot_vals = enc.encode_slot_matrix(inp, fuel)          # [C,69]
        g = enc._encode_globals(inp, fuel, slot_vals)          # [G]
        rows.append(np.concatenate([slot_vals.reshape(-1), g]))
    return np.asarray(rows, dtype=np.float32)


def _target_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw target matrix ``[N,5]`` and validity mask ``[N,5]`` (same rule as CNN)."""
    n = len(df)
    y = np.full((n, len(TARGETS)), np.nan, dtype=np.float64)
    mask = np.zeros((n, len(TARGETS)), dtype=bool)
    converged = df["converged"].astype(bool).to_numpy()
    boc = (df["cbc_kind"].astype(str).to_numpy() if "cbc_kind" in df.columns
           else np.array([""] * n))
    for k, name in enumerate(TARGETS):
        vals = pd.to_numeric(df[name], errors="coerce").to_numpy()
        y[:, k] = vals
        ok = converged & np.isfinite(vals)
        if name == "cbc_max":
            ok = ok & (boc != "boc_only")
        mask[:, k] = ok
    return y, mask


# --------------------------------------------------------------------------- #
# fit / score
# --------------------------------------------------------------------------- #
def run_baseline(
    split: str = "S1",
    *,
    store_dir: str | Path = DEFAULT_STORE,
    splits_dir: str | Path = DEFAULT_SPLITS,
    reports_dir: str | Path = DEFAULT_REPORTS,
    n_estimators: int = 256,
    min_samples_leaf: int = 2,
    max_features: float = 0.7,
    max_train_rows: int | None = None,
    min_case_val: int = 30,
    persist: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit per-target ExtraTrees on the train fold, score the val fold."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    reader = StoreReader(store_dir)
    fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
    manifest = _load_split(reader, split, splits_dir, seed=seed)

    df = reader.records.drop_duplicates("record_id").set_index("record_id")
    train_ids = [i for i in manifest.record_ids("train") if i in df.index]
    val_ids = [i for i in manifest.record_ids("val") if i in df.index]
    if not val_ids:
        result = {"split": split, "status": manifest.status,
                  "note": "empty val fold — no baseline", "per_target": {}}
        if persist:
            _persist(result, reports_dir, split)
        return result

    train_df = df.loc[train_ids].reset_index()
    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=seed).reset_index(drop=True)
    val_df = df.loc[val_ids].reset_index()

    enc = FeatureEncoder()
    t0 = time.time()
    x_train = build_flat_features(train_df, fuel, enc)
    x_val = build_flat_features(val_df, fuel, enc)
    feat_secs = time.time() - t0

    y_train, m_train = _target_arrays(train_df)
    y_val, m_val = _target_arrays(val_df)
    val_cases = val_df["case_pair"].astype(str).to_numpy()

    per_target: dict[str, Any] = {}
    t0 = time.time()
    for k, name in enumerate(TARGETS):
        tr = m_train[:, k]
        va = m_val[:, k]
        if tr.sum() < 8 or va.sum() < 2:
            per_target[name] = {"status": "insufficient_labels",
                                "n_train": int(tr.sum()), "n_val": int(va.sum())}
            continue
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=False, n_jobs=-1,
            random_state=seed + 1009 * k,
        )
        model.fit(x_train[tr], y_train[tr, k])
        pred = model.predict(x_val[va])
        truth = y_val[va, k]
        sp_mean, sp_sd, n_cases = within_case_spearman(
            model.predict(x_val), y_val[:, k], m_val[:, k].astype(float),
            val_cases, min_case_val,
        )
        per_target[name] = {
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "mae": float(mean_absolute_error(truth, pred)),
            "rmse": float(math.sqrt(mean_squared_error(truth, pred))),
            "r2": float(r2_score(truth, pred)) if va.sum() >= 2 else float("nan"),
            "within_case_spearman": sp_mean,
            "within_case_spearman_sd": sp_sd,
            "n_cases": int(n_cases),
        }
    fit_secs = time.time() - t0

    result = {
        "split": split,
        "status": manifest.status,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_features": int(x_train.shape[1]),
        "hyperparams": {"n_estimators": n_estimators,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features},
        "feature_seconds": round(feat_secs, 1),
        "fit_seconds": round(fit_secs, 1),
        "per_target": per_target,
    }
    if persist:
        _persist(result, reports_dir, split)
    return result


def _persist(result: dict[str, Any], reports_dir: str | Path, split: str) -> Path:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"baseline_{split}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = ["run_baseline", "build_flat_features"]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="S1")
    ap.add_argument("--max-train-rows", type=int, default=None)
    args = ap.parse_args()
    res = run_baseline(args.split, max_train_rows=args.max_train_rows)
    print(json.dumps(res, indent=2)[:2000])
