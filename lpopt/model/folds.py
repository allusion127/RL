"""Evaluation folds A / B / C — the honest-scoring partition, codified.

Until now these lived only in analysis scratch, so every report re-derived them by
hand.  They are load-bearing (the A/B decision rule and the promotion gate both
read fold C), so they belong in the library with tests.

The partition
-------------
Given a frozen :class:`~.splits.SplitManifest` and the CURRENT store frame:

===== =============== ==================================================
fold  name            definition
===== =============== ==================================================
A     ``curr_val``    the union of ``groups.curriculum_val_by_cell`` --
                      the per-cell curriculum validation rows
B     ``legacy_val``  ``val_ids`` minus fold A, restricted to the legacy
                      regime (``dataset == "A"`` and ``feed == 121``)
C     ``new_unseen``  every converged store row in NEITHER ``train_ids``
                      NOR ``val_ids``
===== =============== ==================================================

**Why fold C needs no timestamp.**  The manifest is frozen: ``train_ids`` and
``val_ids`` are literal record-id lists written when the split was made.  A row
in the store but in neither list therefore *cannot* have existed at split time —
set difference against a frozen manifest IS the "produced afterwards" predicate,
and it is exact, reproducible and immune to clock skew or missing timestamps.
This is why no ``created_at`` column is consulted.

**Honesty ordering.**  A and B were both consumed by best-epoch selection and
sigma fitting, so they carry an optimism bias and are reported for reference
only.  **Fold C is the sole uncontaminated slice** and is the only fold the
pre-registered decision rule reads.

**Post-selection caveat (carried, not hidden).**  A large share of fold C is
model-PROPOSED (``generator``/``campaign`` starting ``alsearch``), i.e. the
incumbent model chose to have those points computed.  Metrics conditioned on a
model's own proposals are not the same estimand as metrics on independent
production, so :func:`proposal_mask` splits them and callers report both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .splits import SplitManifest

#: Fold ids and their report names.
FOLD_NAMES: dict[str, str] = {
    "A": "curr_val",
    "B": "legacy_val",
    "C": "new_unseen",
}
#: The only fold the pre-registered A/B decision rule may read.
UNCONTAMINATED_FOLD = "C"
#: Legacy-regime predicate for fold B.
_LEGACY_DATASET = "A"
_LEGACY_FEED = 121
#: e_core bin width for the non-curriculum cell key.
CELL_BIN_WIDTH = 0.05
#: Minimum rows for a cell to enter within-cell aggregates.
MIN_CELL_ROWS = 8


def _campaign(df: pd.DataFrame) -> pd.Series:
    if "campaign" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df["campaign"].map(
        lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))


def cell_key(df: pd.DataFrame, *, bin_width: float = CELL_BIN_WIDTH) -> pd.Series:
    """Per-row evaluation cell: the campaign id, else ``ebin{e}_f{feed}``.

    A "cell" is the iso-design group inside which candidates actually compete, so
    within-cell metrics measure the ranking skill the search consumes -- unlike a
    global correlation, which is dominated by between-cell scale and looks far
    better than the model is (report 20260725 section 1.1: global rho 0.99 vs
    within-cell 0.73 for the same predictions).

    Curriculum rows key on ``campaign``; everything else keys on the
    (e_core bin, feed) pair, matching :func:`~.dataset_torch.cyclen_cell_codes`.
    """
    if not len(df):
        return pd.Series([], dtype=object)
    camp = _campaign(df)
    e = pd.to_numeric(df["e_core"], errors="coerce").to_numpy(dtype=float)
    feed = pd.to_numeric(df["feed"], errors="coerce").to_numpy(dtype=float)
    out = []
    for i, c in enumerate(camp.to_numpy()):
        if c:
            out.append(str(c))
            continue
        if not (math.isfinite(e[i]) and math.isfinite(feed[i])):
            out.append("")
            continue
        eb = math.floor(e[i] / bin_width) * bin_width
        out.append(f"ebin{eb:.2f}_f{int(feed[i])}")
    return pd.Series(out, index=df.index, dtype=object)


def fold_a_ids(manifest: SplitManifest) -> set[str]:
    """Union of the per-cell curriculum validation ids."""
    by_cell = (manifest.groups or {}).get("curriculum_val_by_cell", {}) or {}
    out: set[str] = set()
    for ids in by_cell.values():
        out.update(str(i) for i in ids)
    return out


def assign_folds(df: pd.DataFrame, manifest: SplitManifest) -> pd.Series:
    """Per-row fold label: ``"A"``, ``"B"``, ``"C"``, ``"train"`` or ``""``.

    ``""`` marks a validation row that is neither fold A nor in the legacy regime
    (so it belongs to no reported fold) -- it is dropped, never silently merged.
    """
    if not len(df):
        return pd.Series([], dtype=object)
    rid = df["record_id"].astype(str)
    train_ids = set(manifest.record_ids("train"))
    val_ids = set(manifest.record_ids("val"))
    a_ids = fold_a_ids(manifest)

    dataset = (df["dataset"].astype(str) if "dataset" in df.columns
               else pd.Series([""] * len(df), index=df.index))
    feed = pd.to_numeric(df.get("feed"), errors="coerce")

    labels = []
    for i, r in enumerate(rid.to_numpy()):
        if r in a_ids:
            labels.append("A")
        elif r in val_ids:
            legacy = (dataset.iloc[i] == _LEGACY_DATASET
                      and float(feed.iloc[i] or np.nan) == _LEGACY_FEED)
            labels.append("B" if legacy else "")
        elif r in train_ids:
            labels.append("train")
        else:
            # In the store but in NEITHER frozen id list => produced after the
            # split was written.  This is the fold C predicate.
            labels.append("C")
    return pd.Series(labels, index=df.index, dtype=object)


def proposal_mask(df: pd.DataFrame) -> np.ndarray:
    """``True`` where the row was proposed by a model (``alsearch*``).

    Fold C mixes model-proposed candidates with independent production; the two
    are different estimands (report 20260725 section 2.9 measured peak within-cell
    rho 0.646 vs 0.802 across that split), so they are reported separately rather
    than pooled.
    """
    if not len(df):
        return np.zeros(0, dtype=bool)
    out = np.zeros(len(df), dtype=bool)
    for col in ("generator", "campaign"):
        if col in df.columns:
            s = df[col].astype(str).str.lower()
            out |= s.str.startswith("alsearch").to_numpy()
    return out


@dataclass(frozen=True)
class FoldFrame:
    """One fold's rows plus its per-row cell keys and proposal flags."""

    fold: str
    name: str
    df: pd.DataFrame
    cells: np.ndarray
    is_proposal: np.ndarray

    def __len__(self) -> int:
        return len(self.df)

    @property
    def n_cells(self) -> int:
        return int(len(set(self.cells.tolist())))


def fold_frame(df: pd.DataFrame, manifest: SplitManifest, fold: str, *,
               converged_only: bool = True,
               labels: pd.Series | None = None) -> FoldFrame:
    """Extract one fold as a :class:`FoldFrame` (row order preserved).

    ``converged_only`` keeps only rows whose MASTER run converged -- a
    non-converged row's target values are not a physical answer, so scoring
    against them measures nothing.
    """
    if fold not in FOLD_NAMES:
        raise ValueError(f"unknown fold {fold!r}; have {sorted(FOLD_NAMES)}")
    lab = assign_folds(df, manifest) if labels is None else labels
    keep = (lab == fold).to_numpy()
    if converged_only and "converged" in df.columns:
        keep &= df["converged"].fillna(False).astype(bool).to_numpy()
    sub = df.loc[keep].reset_index(drop=True)
    return FoldFrame(fold=fold, name=FOLD_NAMES[fold], df=sub,
                     cells=cell_key(sub).to_numpy(),
                     is_proposal=proposal_mask(sub))


def summarize_folds(df: pd.DataFrame, manifest: SplitManifest) -> dict[str, Any]:
    """Row/cell/map counts per fold — the provenance block every report opens with."""
    lab = assign_folds(df, manifest)
    out: dict[str, Any] = {"n_store_rows": int(len(df)),
                           "n_train_ids": manifest.n_train,
                           "n_val_ids": manifest.n_val}
    for fold in FOLD_NAMES:
        ff = fold_frame(df, manifest, fold, labels=lab)
        has_map = (int(ff.df["maps_key"].notna().sum())
                   if "maps_key" in ff.df.columns else 0)
        out[fold] = {
            "name": ff.name,
            "n_converged": len(ff),
            "n_cells": ff.n_cells,
            "n_with_map": has_map,
            "n_proposal": int(ff.is_proposal.sum()),
            "uncontaminated": fold == UNCONTAMINATED_FOLD,
        }
    out["n_unlabelled_val"] = int((lab == "").sum())
    return out


__all__ = [
    "CELL_BIN_WIDTH",
    "FOLD_NAMES",
    "MIN_CELL_ROWS",
    "UNCONTAMINATED_FOLD",
    "FoldFrame",
    "assign_folds",
    "cell_key",
    "fold_a_ids",
    "fold_frame",
    "proposal_mask",
    "summarize_folds",
]
