"""Measure ``PEAK_SCALE`` / ``COV_SCALE`` on the evaluation slice (program §1.2).

``python -m lpopt.tools.fit_flat_scale [--store-dir data/store] [--dry-run]``

The flatness objective divides ``node_peak`` and ``map_cov`` by a scale before
weighting them 1 : 0.5.  Those scales must be the **within-cell** spread of the
two scalars under the MULTIPLICITY-WEIGHTED definition (:mod:`..data.flatness`),
measured on the labelled slice — not the draft's 0.23 / 0.065, which came from
the unweighted definition on a corpus that was 87% two mega-cells.

What this writes
----------------
``<store-dir>/flat_scale.json``:

* ``global`` — the MEDIAN over cells of each within-cell SD.  Median over CELLS,
  not over rows: a row-weighted statistic would be the two mega-cells and nothing
  else.  These are the fallback for a cell the artifact never fitted.
* ``cells`` — every cell with at least ``--min-rows`` mapped converged rows, with
  its own two SDs.  Per-cell normalization is the DEFAULT (D4).
* ``realized_w_cov`` — the realized secondary weight distribution the GLOBAL
  constants would produce, so the honesty cost of the fallback is written down
  rather than assumed.  Program §1.2 requires this number to exist.

The evaluation slice is converged + valid rows carrying BOTH flatness columns,
which is exactly the population the objective is computed on.  Nothing here reads
a model, so it is reproducible from ``records.parquet`` alone.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.flat_scale import (
    ARTIFACT_NAME, ARTIFACT_SCHEMA, CellScale, DEFAULT_COV_SCALE,
    DEFAULT_PEAK_SCALE, DEFAULT_W_COV, FlatScale, MIN_CELL_ROWS,
)
from ..data.store import StoreReader
from ..model.cell_calibrate import DEFAULT_BIN_WIDTH, cyclen_cell_key


@dataclass
class FitReport:
    """What one :func:`fit_flat_scale` pass measured."""

    n_rows: int = 0                  # rows in the evaluation slice
    n_cells_seen: int = 0            # distinct cells in the slice
    n_cells_fitted: int = 0          # cells with >= min_rows
    peak_scale: float = DEFAULT_PEAK_SCALE
    cov_scale: float = DEFAULT_COV_SCALE
    realized_w_cov: dict[str, Any] = field(default_factory=dict)
    sd_ratio: dict[str, Any] = field(default_factory=dict)
    wrote: bool = False
    path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows, "n_cells_seen": self.n_cells_seen,
            "n_cells_fitted": self.n_cells_fitted,
            "peak_scale": self.peak_scale, "cov_scale": self.cov_scale,
            "realized_w_cov": self.realized_w_cov, "sd_ratio": self.sd_ratio,
            "wrote": self.wrote, "path": self.path,
        }


def evaluation_slice(df: pd.DataFrame, *, bin_width: float = DEFAULT_BIN_WIDTH
                     ) -> pd.DataFrame:
    """Converged + valid rows carrying BOTH flatness columns, with a cell key.

    The cell key is :func:`..model.cell_calibrate.cyclen_cell_key` — the
    ``(feed, e_core-bin)`` key that is derivable at SERVE time from the campaign's
    own case, which is what lets the acquisition scalar look up the same cell the
    fit measured.
    """
    if df is None or not len(df):
        return pd.DataFrame(columns=["node_peak", "map_cov", "cell"])
    sub = df
    for col, want in (("converged", True), ("valid", True)):
        if col in sub.columns:
            sub = sub[sub[col] == want]  # noqa: E712
    for col in ("node_peak", "map_cov"):
        if col not in sub.columns:
            return pd.DataFrame(columns=["node_peak", "map_cov", "cell"])
        sub = sub[sub[col].notna()]
    if not len(sub):
        return pd.DataFrame(columns=["node_peak", "map_cov", "cell"])
    feed = pd.to_numeric(sub["feed"], errors="coerce")
    ecore = pd.to_numeric(sub["e_core"], errors="coerce")
    keys = [
        cyclen_cell_key(int(f), (float(e) if e is not None and math.isfinite(float(e))
                                 else None), bin_width)
        if f is not None and math.isfinite(float(f)) else ""
        for f, e in zip(feed.to_numpy(dtype=float), ecore.to_numpy(dtype=float))
    ]
    out = pd.DataFrame({
        "cell": keys,
        "node_peak": pd.to_numeric(sub["node_peak"], errors="coerce").to_numpy(float),
        "map_cov": pd.to_numeric(sub["map_cov"], errors="coerce").to_numpy(float),
    })
    return out[(out["cell"] != "") & np.isfinite(out["node_peak"])
               & np.isfinite(out["map_cov"])]


def fit_cells(slice_df: pd.DataFrame, *, min_rows: int = MIN_CELL_ROWS
              ) -> tuple[list[CellScale], int]:
    """``([CellScale], n_cells_seen)`` — within-cell SDs of the two scalars."""
    if not len(slice_df):
        return [], 0
    out: list[CellScale] = []
    seen = 0
    for cell, d in slice_df.groupby("cell", sort=True):
        seen += 1
        if len(d) < int(min_rows):
            continue
        peak = d["node_peak"].to_numpy(dtype=float)
        cov = d["map_cov"].to_numpy(dtype=float)
        sd_p = float(np.std(peak, ddof=1))
        sd_c = float(np.std(cov, ddof=1))
        if not (math.isfinite(sd_p) and math.isfinite(sd_c)) or sd_p <= 0.0 or sd_c <= 0.0:
            continue
        out.append(CellScale(str(cell), int(len(d)), sd_p, sd_c))
    return out, seen


def fit_flat_scale(store_dir: str | Path = "data/store", *,
                   min_rows: int = MIN_CELL_ROWS,
                   w_cov: float = DEFAULT_W_COV,
                   bin_width: float = DEFAULT_BIN_WIDTH,
                   dry_run: bool = False,
                   log: Any = print) -> FitReport:
    """Measure the scales and (unless ``dry_run``) write the artifact."""
    store = Path(store_dir)
    df = StoreReader(store).records
    sl = evaluation_slice(df, bin_width=bin_width)
    cells, seen = fit_cells(sl, min_rows=min_rows)
    rep = FitReport(n_rows=int(len(sl)), n_cells_seen=int(seen),
                    n_cells_fitted=len(cells), path=str(store / ARTIFACT_NAME))
    if not cells:
        log("[fit_flat_scale] no cell reached the row floor; keeping module defaults")
        return rep

    rep.peak_scale = round(float(np.median([c.peak_scale for c in cells])), 6)
    rep.cov_scale = round(float(np.median([c.cov_scale for c in cells])), 6)

    ratio = np.array([c.cov_scale / c.peak_scale for c in cells], dtype=float)
    rep.sd_ratio = {
        "min": float(ratio.min()), "p25": float(np.percentile(ratio, 25)),
        "median": float(np.median(ratio)), "p75": float(np.percentile(ratio, 75)),
        "max": float(ratio.max()),
        "spread": float(ratio.max() / ratio.min()) if ratio.min() > 0 else None,
    }
    # The honesty number: what the GLOBAL constants would realize per cell.
    globalish = FlatScale(peak_scale=rep.peak_scale, cov_scale=rep.cov_scale,
                          cells={c.cell: c for c in cells}, per_cell=False,
                          source="fit")
    rep.realized_w_cov = globalish.realized_w_cov(w_cov=w_cov)

    doc = {
        "schema": ARTIFACT_SCHEMA,
        "fit_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "definition": "multiplicity-weighted (lpopt.data.flatness)",
        "slice": "converged & valid rows carrying node_peak AND map_cov",
        "cell_key": f"cyclen_cell_key(feed, e_core, bin_width={bin_width})",
        "min_cell_rows": int(min_rows),
        "n_rows": rep.n_rows,
        "n_cells_seen": rep.n_cells_seen,
        "n_cells_fitted": rep.n_cells_fitted,
        "w_cov": float(w_cov),
        "global": {"peak_scale": rep.peak_scale, "cov_scale": rep.cov_scale,
                   "how": "median over fitted cells of the within-cell SD"},
        "sd_ratio_cov_over_peak": rep.sd_ratio,
        "realized_w_cov_with_global_scales": rep.realized_w_cov,
        "cells": {c.cell: c.as_dict() for c in sorted(cells, key=lambda x: x.cell)},
    }
    log(f"[fit_flat_scale] {rep.n_rows} rows / {rep.n_cells_fitted} cells "
        f"(of {rep.n_cells_seen} seen, floor {min_rows})")
    log(f"[fit_flat_scale] PEAK_SCALE={rep.peak_scale:.6f} "
        f"COV_SCALE={rep.cov_scale:.6f}")
    log(f"[fit_flat_scale] within-cell SD_cov/SD_peak: "
        f"min {rep.sd_ratio['min']:.4f} median {rep.sd_ratio['median']:.4f} "
        f"max {rep.sd_ratio['max']:.4f} (spread {rep.sd_ratio['spread']:.2f}x)")
    r = rep.realized_w_cov
    log(f"[fit_flat_scale] realized w_cov with GLOBAL constants: "
        f"min {r['min']:.3f} median {r['median']:.3f} max {r['max']:.3f} "
        f"(declared {w_cov}) -> per-cell normalization is the default")
    if dry_run:
        log("[fit_flat_scale] --dry-run: nothing written")
        return rep
    store.mkdir(parents=True, exist_ok=True)
    tmp = store / f"{ARTIFACT_NAME}.tmp"
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(store / ARTIFACT_NAME)
    rep.wrote = True
    log(f"[fit_flat_scale] -> {rep.path}")
    return rep


def main(argv: Any = None) -> int:  # pragma: no cover - CLI wrapper
    ap = argparse.ArgumentParser(prog="python -m lpopt.tools.fit_flat_scale")
    ap.add_argument("--store-dir", default="data/store")
    ap.add_argument("--min-rows", type=int, default=MIN_CELL_ROWS)
    ap.add_argument("--w-cov", type=float, default=DEFAULT_W_COV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    fit_flat_scale(args.store_dir, min_rows=args.min_rows, w_cov=args.w_cov,
                   dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
