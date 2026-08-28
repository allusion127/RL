"""Torch dataset over the unified store (plan sec. 4.4).

:class:`LPDataset` featurizes store records on the fly (leakage-safe, via
:class:`lpopt.model.featurize.FeatureEncoder`) and returns **raw** targets — z
scoring belongs to the training loop, so the dataset stays a pure feature/label
source.  Each item carries:

* ``cells``       ``float32[C,19,19]`` physics channels,
* ``globals``     ``float32[G]`` FiLM conditioning vector,
* ``targets``     ``float32[T]`` raw ``[f_r, f_q, cbc_max, cyclen, ao_abs,
  discharge_burnup, max_pin_burnup]`` (Phase D promoted the two burnup axes to
  first-class targets, plan sec. 12.4),
* ``target_mask`` ``float32[T]`` — 0 for a non-converged chain, a NaN target,
  ``cbc_max`` on a ``cbc_kind=="boc_only"`` record, or (when
  ``censor_dataset_a_pin_labels``, the default) ``max_pin_burnup`` on a
  ``dataset=="A"`` row — A's pin label is a MOCHA-cache surrogate, not a
  fidelity-consistent pin calc (data/reports/pinbu_forensics.md).  The mask is
  **per target**, so a Dataset B/P row that never carried ``discharge_burnup`` /
  ``max_pin_burnup`` (they are NaN) is censored on *those two axes only* while
  its other five labels still train,
* ``conv_label``  ``float32`` converged flag, with
* ``conv_mask``   ``float32`` — 0 when ``converged_at_cap`` (label is *unknown*,
  not a physical non-convergence),
* ``maps``        ``float32[4,9,9]`` EDIT5 stack (NaN-filled when absent) and
* ``maps_mask``   ``float32[4,9,9]`` — 1 only where a present map cell is finite.

With ``include_axial=True`` (default **off**) two more keys appear:

* ``axial``       ``float32[A,25]`` EDIT6 axial power profile at the
  :data:`lpopt.data.axial.ANCHORS` burnup anchors (NaN-filled when absent),
* ``axial_mask``  ``float32[A]`` — 1 per anchor with a present label.

With ``include_traj=True`` (default **off**) three more appear:

* ``traj``        ``float32[T,3,9,9]`` EDIT5 ``(power, burnup, kinf)`` planes at
  the requested cycle-burnup fractions (NaN-filled when absent; the 12 non-slot
  cells of each quarter are NaN exactly as in ``maps``),
* ``traj_frac``   ``float32[T]`` the ACHIEVED cycle-burnup fraction of each
  selected step (the model's conditioning input — never the requested value),
* ``traj_mask``   ``float32[T]`` — 1 per anchor the record can honestly support.

They are absent from the returned dict when the flag is off, so the flag-off
item is key-for-key the pre-axial/pre-traj item.

``augment=True`` applies a 50%-probability diagonal-mirror (transpose) on
``__getitem__``.  ``compute_cell_weights`` implements the plan's
inverse-sqrt (feed, e_core-bin, dataset) frequency weighting with a cap.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data.axial import ANCHORS as AXIAL_ANCHORS
from ..data.axial import N_PLANES as AXIAL_PLANES
from ..data.axial import anchor_profiles, load_axial
from ..data.fuel_types import FuelLibrary
from ..data.store import StoreReader
from ..data.traj import DEFAULT_ANCHORS as TRAJ_ANCHORS
from ..data.traj import N_PLANES as TRAJ_PLANES
from ..data.traj import QUARTER as TRAJ_QUARTER
from ..data.traj import anchor_planes as traj_anchor_planes
from ..data.traj import load_traj
from .featurize import FeatureEncoder

#: Raw regression targets returned by the dataset, in order.  Phase D
#: (plan sec. 12.4) appended ``discharge_burnup`` and ``max_pin_burnup`` as
#: first-class targets; NaN labels on either (Dataset B/P rows) are censored
#: per-target by :meth:`LPDataset._targets` via the standard ``isnan`` rule.
TARGETS: tuple[str, ...] = (
    "f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
    "discharge_burnup", "max_pin_burnup",
)

#: The v5 ``promote_max_asm_bu`` target order — :data:`TARGETS` with
#: ``max_assembly_burnup`` APPENDED (never inserted), so every existing index
#: (notably ``cyclen`` == 3, which the rank loss and the cell calibration key on)
#: is unchanged and a 7-target checkpoint stays readable.  ``max_assembly_burnup``
#: is the vendor's MAX-assembly-burnup CONSTRAINT axis (surrogate column 5) — a
#: different physical quantity from our ``discharge_burnup`` (core AVERAGE), which
#: is why it needs its own head output rather than reusing that column.  The store
#: carries it for ~94% of rows; the remainder are masked by the standard NaN rule.
TARGETS_WITH_ASM_BU: tuple[str, ...] = TARGETS + ("max_assembly_burnup",)


def targets_for(promote_max_asm_bu: bool = False) -> tuple[str, ...]:
    """The active dataset target tuple for a ``promote_max_asm_bu`` setting."""
    return TARGETS_WITH_ASM_BU if promote_max_asm_bu else TARGETS


def _resolve_ids(split_manifest: Any, df: pd.DataFrame, fold: str) -> list[str]:
    """Record-id membership for a dataset from a manifest / id list / None."""
    if split_manifest is None:
        return df["record_id"].astype(str).tolist()
    if hasattr(split_manifest, "record_ids"):
        return list(split_manifest.record_ids(fold))
    if isinstance(split_manifest, (list, tuple, set)):
        return [str(x) for x in split_manifest]
    raise TypeError(
        "split_manifest must be a SplitManifest, an id iterable, or None; "
        f"got {type(split_manifest).__name__}"
    )


class LPDataset(Dataset):
    """Map-style dataset yielding featurized records with raw targets + masks."""

    def __init__(
        self,
        store_reader: StoreReader,
        split_manifest: Any,
        fuel_library: FuelLibrary,
        augment: bool = False,
        include_maps: bool = True,
        *,
        fold: str = "train",
        encoder: FeatureEncoder | None = None,
        seed: int = 0,
        censor_dataset_a_pin_labels: bool = True,
        promote_max_asm_bu: bool = False,
        include_axial: bool = False,
        axial_anchors: Sequence[str] = AXIAL_ANCHORS,
        include_traj: bool = False,
        traj_anchors: Sequence[float] = TRAJ_ANCHORS,
    ):
        self.reader = store_reader
        self.fuel = fuel_library
        self.encoder = encoder or FeatureEncoder()
        self.augment = bool(augment)
        self.include_maps = bool(include_maps)
        #: Axial (EDIT6) profile labels.  Default OFF: with the flag off the item
        #: dict has exactly the pre-axial keys.
        self.include_axial = bool(include_axial)
        self.axial_anchors: tuple[str, ...] = tuple(axial_anchors)
        #: EDIT5 burnup-trajectory (EDIT5 per-step) labels.  Default OFF: with the
        #: flag off the item dict has exactly the pre-traj keys.
        self.include_traj = bool(include_traj)
        self.traj_anchors: tuple[float, ...] = tuple(float(f) for f in traj_anchors)
        self.censor_dataset_a_pin_labels = bool(censor_dataset_a_pin_labels)
        #: Active target inventory.  Default (flag off) is exactly :data:`TARGETS`,
        #: so every tensor width and index is unchanged.
        self.promote_max_asm_bu = bool(promote_max_asm_bu)
        self.targets: tuple[str, ...] = targets_for(self.promote_max_asm_bu)
        self._rng = np.random.default_rng(seed)

        df = store_reader.records
        ids = _resolve_ids(split_manifest, df, fold)
        indexed = df.drop_duplicates("record_id").set_index("record_id")
        present = [i for i in ids if i in indexed.index]
        self.df = indexed.loc[present].reset_index()
        self.record_ids: list[str] = present

    # ---------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.df)

    def _targets(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        converged = bool(row["converged"])
        names = self.targets
        vals = np.zeros(len(names), dtype=np.float32)
        mask = np.zeros(len(names), dtype=np.float32)
        boc_only = str(row.get("cbc_kind", "")) == "boc_only"
        # Dataset A's max_pin_burnup label is a MOCHA-cache surrogate (near-constant
        # ~1.08x assembly burnup), not a fidelity-consistent pin calc; censor it so
        # the pin-burnup head trains only on the real MAS_PPI (Dataset P) pin labels
        # (data/reports/pinbu_forensics.md).  Labels only — serving is unaffected.
        censor_a_pin = (self.censor_dataset_a_pin_labels
                        and str(row.get("dataset", "")) == "A")
        for k, name in enumerate(names):
            v = row.get(name)
            fv = float(v) if v is not None else float("nan")
            vals[k] = fv
            # max_assembly_burnup is masked wherever the label is absent by this
            # SAME standard NaN rule — no special case needed, and a row that
            # never carried it simply trains on its other targets.
            valid = converged and not math.isnan(fv)
            if name == "cbc_max" and boc_only:
                valid = False               # cbc_max unreliable on boc_only rows
            if name == "max_pin_burnup" and censor_a_pin:
                valid = False               # A pin label is a cache surrogate
            mask[k] = 1.0 if valid else 0.0
        return vals, mask

    def _maps(self, record_id: str, row: pd.Series
              ) -> tuple[np.ndarray, np.ndarray]:
        shape = (4, 9, 9)
        maps_key = row.get("maps_key")
        arr = None
        if self.include_maps and maps_key is not None and not (
            isinstance(maps_key, float) and math.isnan(maps_key)
        ):
            arr = self.reader.maps(str(maps_key))
        if arr is None:
            maps = np.full(shape, np.nan, dtype=np.float32)     # NaN-filled: absent
            mask = np.zeros(shape, dtype=np.float32)
            return maps, mask
        maps = np.asarray(arr, dtype=np.float32).reshape(shape)
        mask = np.isfinite(maps).astype(np.float32)
        return maps, mask

    def _axial(self, record_id: str, row: pd.Series
               ) -> tuple[np.ndarray, np.ndarray]:
        """``((A, 25) profiles, (A,) mask)`` — NaN/0 when the label is absent.

        The axial stack is keyed ``<record_id>__axial`` (not ``maps_key``): the
        high-resolution harvest writes it per record id, and a record can carry a
        map without an axial stack (every pre-EDIT6-harvest row does).  A
        non-converged chain is masked out by the same rule the scalar targets
        use — its EOC step is not a converged equilibrium state.
        """
        a_n = len(self.axial_anchors)
        blank = (np.full((a_n, AXIAL_PLANES), np.nan, dtype=np.float32),
                 np.zeros(a_n, dtype=np.float32))
        if not bool(row["converged"]):
            return blank
        stack = load_axial(self.reader, record_id)
        if stack is None:
            return blank
        prof = anchor_profiles(stack, self.axial_anchors).astype(np.float32)
        return prof, np.ones(a_n, dtype=np.float32)

    def _traj(self, record_id: str, row: pd.Series
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``((T,3,9,9) planes, (T,) achieved fractions, (T,) mask)``.

        NaN planes / zero mask when the record carries no EDIT5 trajectory, the
        trajectory has no usable burnup coordinate, or the chain did not converge
        — the same three rules :meth:`_axial` and :meth:`_maps` obey.  The
        returned FRACTION is the achieved one (see :mod:`lpopt.data.traj`); for a
        masked-out row it is the requested one, which is never read.
        """
        t_n = len(self.traj_anchors)
        want = np.asarray(self.traj_anchors, dtype=np.float32)
        blank = (
            np.full((t_n, TRAJ_PLANES, TRAJ_QUARTER, TRAJ_QUARTER), np.nan,
                    dtype=np.float32),
            want,
            np.zeros(t_n, dtype=np.float32),
        )
        if not bool(row["converged"]):
            return blank
        stack = load_traj(self.reader, record_id)
        if stack is None:
            return blank
        got = traj_anchor_planes(stack, self.traj_anchors)
        if got is None:
            return blank
        planes, frac, mask = got
        return (planes.astype(np.float32), frac.astype(np.float32),
                mask.astype(np.float32))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        record_id = str(row["record_id"])
        cells, globals_ = self.encoder.encode(row, self.fuel)
        if self.augment and self._rng.random() < 0.5:
            cells, globals_ = self.encoder.augment_transpose(
                cells, globals_, row, self.fuel
            )

        targets, target_mask = self._targets(row)
        conv_label = 1.0 if bool(row["converged"]) else 0.0
        conv_mask = 0.0 if bool(row.get("converged_at_cap", False)) else 1.0
        maps, maps_mask = self._maps(record_id, row)

        item = {
            "record_id": record_id,
            "cells": torch.from_numpy(np.ascontiguousarray(cells)),
            "globals": torch.from_numpy(np.ascontiguousarray(globals_)),
            "targets": torch.from_numpy(targets),
            "target_mask": torch.from_numpy(target_mask),
            "conv_label": torch.tensor(conv_label, dtype=torch.float32),
            "conv_mask": torch.tensor(conv_mask, dtype=torch.float32),
            "maps": torch.from_numpy(maps),
            "maps_mask": torch.from_numpy(maps_mask),
        }
        if self.include_axial:
            # The diagonal-transpose augmentation is a RADIAL relabelling; the
            # axial profile is invariant under it, so the (base-row) label is
            # correct for the transposed variant too — no augmented counterpart
            # is needed, exactly as for ``targets`` and ``maps``.
            axial, axial_mask = self._axial(record_id, row)
            item["axial"] = torch.from_numpy(axial)
            item["axial_mask"] = torch.from_numpy(axial_mask)
        if self.include_traj:
            # Same argument as the axial block above, with one difference worth
            # stating: the trajectory planes ARE radial maps, so like ``maps``
            # they inherit the pre-existing convention that the diagonal
            # transpose relabels ``cells`` only and the (base-row) map label is
            # reused.  That convention is the champion's; changing it would be a
            # SECOND change and does not belong in a one-change-per-arm A/B.
            traj, traj_frac, traj_mask = self._traj(record_id, row)
            item["traj"] = torch.from_numpy(traj)
            item["traj_frac"] = torch.from_numpy(traj_frac)
            item["traj_mask"] = torch.from_numpy(traj_mask)
        return item


# --------------------------------------------------------------------------- #
# inverse-sqrt frequency sampling weights (plan sec. 4.4 weighting policy)
# --------------------------------------------------------------------------- #
def compute_cell_weights(
    df: pd.DataFrame,
    *,
    cap: float = 8.0,
    e_core_bin_width: float = 0.05,
    curriculum_campaigns: Sequence[str] | None = None,
    curriculum_cap: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-row inverse-sqrt sampling weight over ``(feed, e_core-bin, dataset)``.

    A row's cell is ``(feed, floor(e_core / w) * w, dataset)``.  The raw weight
    ``1/sqrt(n_cell)`` is normalized so the *most common* cell weighs 1.0 and the
    rarer cells weigh up to ``cap`` (default 8.0).  Returns ``(weights, summary)``
    where ``summary`` reports per-cell counts and effective (post-cap) weight
    mass for the training report.

    ``curriculum_cap`` (with ``curriculum_campaigns``) applies a HIGHER per-row cap
    ONLY to curriculum-cell rows — ``dataset == 'P'`` rows whose ``campaign`` is one
    of ``curriculum_campaigns`` — while the legacy A/B/P0 corpus keeps ``cap``.
    This lets previously-learned cells un-cap their inverse-sqrt weight so their
    ranking holds against a new cell's gradient pressure (negative-transfer
    mitigation, plan 12.3).  The curriculum campaign ids + cap are threaded from
    the retrain split manifest, so the trainer needs no fragile CLI list.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float32), {"n_cells": 0, "cells": {}}

    feed = df["feed"].astype(int).to_numpy()
    dataset = df["dataset"].astype(str).to_numpy()
    e = pd.to_numeric(df["e_core"], errors="coerce").to_numpy()
    ebin = np.where(
        np.isfinite(e),
        np.floor(e / e_core_bin_width) * e_core_bin_width,
        np.nan,
    )

    keys = [
        (int(feed[i]), (round(float(ebin[i]), 4) if np.isfinite(ebin[i]) else None),
         str(dataset[i]))
        for i in range(n)
    ]
    counts: dict[tuple, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1

    raw = np.asarray([1.0 / math.sqrt(counts[key]) for key in keys], dtype=np.float64)
    # most common cell -> smallest raw weight; normalize it to 1.0.
    w_min = raw.min()
    norm = raw / w_min

    # per-row cap ceiling: the global ``cap`` everywhere, overridden by the higher
    # ``curriculum_cap`` on curriculum-cell rows (dataset==P & campaign in cells).
    cap_vec = np.full(n, float(cap), dtype=np.float64)
    curr_set = {str(c) for c in (curriculum_campaigns or [])}
    curr_mask = np.zeros(n, dtype=bool)
    if curr_set and curriculum_cap is not None:
        campaign = (df["campaign"].astype(str).to_numpy()
                    if "campaign" in df.columns else np.array(["None"] * n))
        curr_mask = (dataset == "P") & np.isin(campaign, list(curr_set))
        cap_vec[curr_mask] = float(curriculum_cap)

    weights = np.minimum(norm, cap_vec).astype(np.float32)

    # per-cell reporting: count + effective (post-cap) weight mass, computed from
    # the ACTUAL weights so a per-row curriculum cap is reflected correctly.
    cell_mass: dict[tuple, float] = {}
    for i, key in enumerate(keys):
        cell_mass[key] = cell_mass.get(key, 0.0) + float(weights[i])
    per_cell: dict[str, dict[str, float]] = {}
    for key, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        eff_mass = cell_mass[key]
        label = f"feed={key[0]}|ebin={key[1]}|ds={key[2]}"
        per_cell[label] = {
            "count": int(cnt),
            "row_weight": round(eff_mass / cnt, 4),
            "effective_mass": round(eff_mass, 2),
        }
    summary = {
        "n_rows": int(n),
        "n_cells": len(counts),
        "cap": float(cap),
        "e_core_bin_width": float(e_core_bin_width),
        # legacy rows clipped at the global ``cap`` (curriculum rows, which are
        # clipped at their own raised cap, are reported by ``curriculum_cap_hits``).
        "cap_hits": int(((norm > float(cap)) & ~curr_mask).sum()),
        "weight_min": round(float(weights.min()), 4),
        "weight_max": round(float(weights.max()), 4),
        "curriculum_cap": (float(curriculum_cap)
                           if (curr_set and curriculum_cap is not None) else None),
        "n_curriculum_rows": int(curr_mask.sum()),
        "curriculum_cap_hits": int((curr_mask & (norm > float(curriculum_cap))).sum())
        if (curr_set and curriculum_cap is not None) else 0,
        "cells": per_cell,
    }
    return weights, summary


def cyclen_cell_codes(
    df: pd.DataFrame, *, e_core_bin_width: float = 0.05
) -> np.ndarray:
    """Per-row integer ``(feed, e_core-bin, dataset)`` cell code for the cyclen
    within-cell rank loss (plan 12.5 addendum / RL forensic 20260720).

    Codes mirror the sampler's weighting cells (:func:`compute_cell_weights`) so a
    "cell" is the same iso-design group the honest gate scores its within-cell
    cyclen Spearman on.  A row with a non-finite ``e_core`` (unresolvable fed
    types) gets code ``-1`` and is excluded from pairing.  The absolute code
    values are arbitrary — only equality matters — so the mapping is a plain
    dense factorization, deterministic under a fixed row order.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    feed = df["feed"].astype(int).to_numpy()
    dataset = df["dataset"].astype(str).to_numpy()
    e = pd.to_numeric(df["e_core"], errors="coerce").to_numpy()
    ebin = np.where(np.isfinite(e), np.floor(e / e_core_bin_width) * e_core_bin_width, np.nan)
    codes = np.full(n, -1, dtype=np.int64)
    lut: dict[tuple, int] = {}
    for i in range(n):
        if not np.isfinite(ebin[i]):
            continue
        key = (int(feed[i]), round(float(ebin[i]), 4), str(dataset[i]))
        c = lut.get(key)
        if c is None:
            c = len(lut)
            lut[key] = c
        codes[i] = c
    return codes


# --------------------------------------------------------------------------- #
# CBC label-convention provenance (A/B round-2 arm A2)
# --------------------------------------------------------------------------- #
#: CBC label-convention provenance groups, in FIXED code order.  **Index 0 is the
#: reference** — the MASTER-native convention the model serves — and every other
#: group is a convention that may sit at a constant offset from it.
#:
#: Forensic (2026-07-29): Dataset A rows (``restart_provenance == "mocha_native"``,
#: 59% of the ``cbc_max``-labelled corpus) sit +100..410 ppm ABOVE MASTER-native
#: labels at matched ``(feed, e_core)``.  That is a label-convention gap, not
#: physics — the same core cannot have two critical boron concentrations — so
#: pooling the two conventions into one regression target injects a bimodal
#: label error of exactly that size into every cell that mixes them.
#:
#: The order is frozen because it is the code the learned offsets are indexed by;
#: appending a group is safe, reordering is not.
CBC_PROVENANCE_GROUPS: tuple[str, ...] = (
    "master_native",      # 0 — REFERENCE; the serve convention, offset pinned 0
    "mocha_native",       # 1 — Dataset A (MOCHA cache surrogate labels)
    "ga_native",          # 2 — Dataset B (GA-native harness labels)
)

#: The reference group's name and code (offset structurally zero — never learned).
CBC_PROVENANCE_REFERENCE = CBC_PROVENANCE_GROUPS[0]
CBC_PROVENANCE_REFERENCE_CODE = 0


def cbc_provenance_labels(df: pd.DataFrame) -> np.ndarray:
    """Per-row CBC label-convention group NAME (see :data:`CBC_PROVENANCE_GROUPS`).

    Resolution order — ``restart_provenance`` first because it is the column that
    literally names the harness that produced the boron label, then ``dataset``
    as the fallback for a row written before that column existed:

    * ``restart_provenance == "mocha_native"`` or ``dataset == "A"`` -> ``mocha_native``
    * ``restart_provenance == "ga_native"``    or ``dataset == "B"`` -> ``ga_native``
    * everything else (Dataset P's ``pair_ecore:`` / ``pair_feed:`` MASTER restarts,
      and any unknown value) -> ``master_native``, the REFERENCE.

    Falling back to the reference is the conservative direction: an unrecognised
    provenance gets NO offset, so a new data source can never be silently shifted.
    """
    n = len(df)
    if n == 0:
        return np.empty(0, dtype=object)
    prov = (df["restart_provenance"].astype(str).to_numpy()
            if "restart_provenance" in df.columns else np.array(["?"] * n))
    dataset = (df["dataset"].astype(str).to_numpy()
               if "dataset" in df.columns else np.array(["?"] * n))
    out = np.full(n, CBC_PROVENANCE_REFERENCE, dtype=object)
    out[(prov == "ga_native") | (dataset == "B")] = "ga_native"
    out[(prov == "mocha_native") | (dataset == "A")] = "mocha_native"
    return out


def cbc_provenance_codes(df: pd.DataFrame) -> np.ndarray:
    """Per-row integer CBC-provenance code into :data:`CBC_PROVENANCE_GROUPS`."""
    lut = {name: i for i, name in enumerate(CBC_PROVENANCE_GROUPS)}
    labels = cbc_provenance_labels(df)
    return np.asarray([lut.get(str(x), CBC_PROVENANCE_REFERENCE_CODE)
                       for x in labels], dtype=np.int64)


__all__ = ["LPDataset", "TARGETS", "TARGETS_WITH_ASM_BU", "targets_for",
           "compute_cell_weights", "cyclen_cell_codes",
           "CBC_PROVENANCE_GROUPS", "CBC_PROVENANCE_REFERENCE",
           "CBC_PROVENANCE_REFERENCE_CODE",
           "cbc_provenance_labels", "cbc_provenance_codes"]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    from .splits import make_splits

    reader = StoreReader("data/store")
    fl = FuelLibrary.from_parquet("data/store/fuel_types.parquet")
    manifests = make_splits(reader.records, seed=0, persist=False)
    ds = LPDataset(reader, manifests["S0"], fl, augment=True, fold="train")
    item = ds[0]
    print("LPDataset train size:", len(ds))
    print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in item.items()})
    weights, summary = compute_cell_weights(reader.records)
    print("cell weights:", weights.shape, "n_cells", summary["n_cells"],
          "cap_hits", summary["cap_hits"])
