"""Training loop for the PosValNet deep ensemble (plan sec. 4.4 / 4.7).

Runnable as ``python -m lpopt.model.train --ensemble 5 --split S1 ...``.

Key contracts:

* **Target normalization** — per-target z-score constants computed from the
  *train* split's converged/valid rows only, stored in the checkpoint meta.  The
  network predicts z-scored ``mu`` / ``log_sigma``; de-normalization happens at
  inference (``model_api``).
* **Losses** — z-scored Huber(δ=1) with a heteroscedastic β-NLL wrapper
  (``warmup_epochs=20`` train μ only with plain Huber, then enable NLL; β=0.5
  ``(σ²)^β`` weighting to prevent variance inflation), per-target ``target_mask``
  honored; convergence-head BCE masked by ``conv_mask``; map head masked Huber
  (λ=0.3) on z-scored EDIT5 maps.
* **Sampler** — ``WeightedRandomSampler`` from ``compute_cell_weights``;
  transpose augmentation 50% via the dataset's ``augment`` flag.
* **Optimizer** — AdamW 3e-4 → cosine to 3e-5, wd 1e-4, batch 256, max 150
  epochs, early-stop patience 15 on the S1-val composite
  = within-case Spearman(F_r) − z-MAE(cyclen) (both logged).
* **Fine-tune modes (both need ``--init-from``)** — ``--freeze-trunk-cyclen``
  bundles TWO things: the shared trunk is frozen (``requires_grad=False`` +
  excluded from the optimizer) AND the cyclen rows of the output heads are
  gradient-masked (so served cyclen stays byte-identical to the champion, which
  is also why the champion's cyclen physics prior + per-cell cyclen calibration
  are copied verbatim instead of re-fit).  ``--trunk-finetune-lr-mult M``
  (mutually exclusive with it) keeps **only the cyclen half**: the trunk is NOT
  frozen but trains in its own optimizer group at ``base_lr * M`` while every
  cyclen protection above still applies.  Both default OFF, so the flag-off path
  is the legacy one, module-for-module and byte-for-byte.
* **Checkpoint (plan sec. 4.7, hard)** — ``model.pt`` (``state_dict`` only) +
  ``meta.json`` (cond_schema, channel/global lists, z-score constants, target
  names, torch/python versions, vendor manifest hash, seed, config, best
  epoch/metrics).  No pickled custom classes.  ``load_member`` rebuilds the
  network from meta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.fuel_types import FuelLibrary
from ..safelog import configure_stdio
from ..data.store import StoreReader, trustworthy
from ..data.traj import DEFAULT_ANCHORS as TRAJ_ANCHORS
from ..data.traj import N_PLANES as TRAJ_PLANES
from ..data.traj import STEP_PLANES as TRAJ_STEP_PLANES
from .dataset_torch import (
    CBC_PROVENANCE_GROUPS, LPDataset, TARGETS, TARGETS_WITH_ASM_BU,
    cbc_provenance_codes, compute_cell_weights, cyclen_cell_codes,
    fxy_cell_codes, fxy_cell_key, targets_for,
)
from .featurize import (
    CHANNELS, CHANNELS_BY_SCHEMA, DEFAULT_COND_SCHEMA, FeatureEncoder)
from .net import TRAJ_MAP_CHANNELS, PosValNet, PosValNetConfig, count_parameters
from .physics_prior import (
    MIN_FXY_LABELS, FxyFrPrior, fit_fxy_prior, fxy_prior_z)
from .splits import SplitManifest, make_splits

COND_SCHEMA = "v3"          # Phase D expanded envelope (plan sec. 12.4)
DEFAULT_STORE = "data/store"
DEFAULT_SPLITS = "data/splits"
_VENDOR_MANIFEST = Path(__file__).resolve().parents[1] / "vendor" / "masterrl" / "VENDOR_MANIFEST.json"
_MAP_KEYS = ("boc_power", "eoc_power", "eoc_burnup", "eoc_kinf")
_CYCLEN_IDX = TARGETS.index("cyclen")
_FR_IDX = TARGETS.index("f_r")
#: boc_power is the map channel whose spatial max is the radial power peak the F_r
#: head scores — used by the (default-off) map/F_r consistency loss.
_BOC_POWER_MAP_IDX = _MAP_KEYS.index("boc_power")
#: Side of the 9x9 quarter map plane (``edit5._quadrant`` layout).
_MAP_SIDE = 9


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyperparameters for one member's training run."""

    epochs: int = 150
    warmup_epochs: int = 20
    batch_size: int = 256
    lr: float = 3.0e-4
    lr_final: float = 3.0e-5
    weight_decay: float = 1.0e-4
    beta_nll: float = 0.5
    huber_delta: float = 1.0
    map_lambda: float = 0.3
    conv_weight: float = 1.0
    patience: int = 15
    cell_weight_cap: float = 8.0
    # --- within-cell cyclen rank loss (RL forensic 20260720) ------------------
    # A margin-ranking auxiliary on same-(feed,e_core-bin,dataset)-cell cyclen
    # pairs in-batch: it enforces the correct WITHIN-cell cyclen ordering the
    # honest gate scores, decoupled from the global z-scale that compresses the
    # within-cell target spread to ~0.36 z (deep in Huber's quadratic regime,
    # where the ranking gradient is weakest).  Scale-invariant (uses only the
    # sign of the RAW cyclen gap), so it is unaffected by tstd; normalization,
    # meta, and the serve/gate path are untouched.  weight 0 == legacy loss.
    cyclen_rank_weight: float = 0.1
    cyclen_rank_margin_z: float = 0.1     # required z-space pred separation per pair
    cyclen_rank_min_gap_efpd: float = 2.0  # ignore pairs below MASTER conv noise
    # --- within-cell F_r elite rank loss (parity_round1c_20260722) ------------
    # The champion's f_r WITHIN-cell Spearman is only ~0.13 in the boundary band
    # (candidates crowd 1.6-1.8; the linear scale is right but the fine ordering
    # that the 1.55 search needs collapses).  A margin-ranking hinge on same-cell
    # F_r pairs restores that ordering, and low-F_r pairs (``min(f_r_i, f_r_j) <=
    # low_thresh``) are up-weighted so the gradient concentrates where the search
    # actually operates.  Scale-invariant (sign of the RAW f_r gap only).  Default
    # ON (weight 0.1); serve/gate/normalization untouched.  weight 0 == legacy.
    map_peak_weight: float = 0.0          # up-weight map_loss at hot (peak) nodes (node_peak-B); 0 = OFF
    # --- A/B round-2 arm A3: top-K per-slot map peak focus (default OFF) -------
    # ``map_peak_weight`` is a CONTINUOUS re-weighting by ``relu(map_z)``: every
    # above-average node gets some extra weight, in proportion to how hot it is.
    # It therefore spends most of its extra gradient on the broad warm region,
    # not on the peak — at map_peak_weight=2.0 the ~30 above-average nodes of a
    # plane collectively outweigh its single hottest node by an order of
    # magnitude.  ``map_peak_topk_weight`` is the RANK-based complement: the K
    # hottest nodes OF EACH PLANE (by the LABEL, so the selection is noise-free —
    # peak location reproduces 22/22 under the transpose check) get a further
    # ``1 + w`` factor.  ``F_q`` and ``node_peak`` are order statistics of exactly
    # those nodes.  ``0.0`` is byte-identical to the pre-existing map loss.
    map_peak_topk: int = 5
    map_peak_topk_weight: float = 0.0
    f_r_rank_weight: float = 0.1
    f_r_rank_margin_z: float = 0.1
    f_r_rank_min_gap: float = 0.01        # ignore f_r pairs below MASTER conv noise
    f_r_rank_low_thresh: float = 1.7      # up-weight pairs in the boundary band
    f_r_rank_low_weight: float = 3.0
    # --- map / F_r consistency loss (parity_round1c_20260722, default OFF) -----
    # A small within-batch-standardized agreement between the boc_power map's
    # spatial MAX (the predicted radial power peak) and the F_r head — both encode
    # peaking, so tying them regularizes the crowded-band f_r ordering with the
    # spatial map's signal.  Default 0.0 (no-op); recommended small (0.1).
    map_fr_consistency_weight: float = 0.0
    # --- v5 bundle: three independent, default-OFF additive knobs -------------
    # Each is a pure no-op at its default, so a run with all three off is
    # byte-identical to the pre-v5 training path (regression-tested).
    #
    # (1) Physics-prior residual learning for cyclen (physics_prior.py).  The
    #     network regresses ``cyclen - prior`` instead of ``cyclen``; the prior is
    #     a leading-order reactivity balance over the loaded assembly mix, fit
    #     (2 scalars) on TRAIN rows only and stamped into the checkpoint meta so
    #     serving adds it back and ``predict()`` still returns absolute cyclen.
    cyclen_physics_prior: bool = False
    # (2) Pinball-loss quantile heads alongside the mean/log_sigma heads.
    quantile_heads: bool = False
    quantile_levels: tuple[float, ...] = (0.10, 0.50, 0.90)
    quantile_targets: tuple[str, ...] = ("f_r", "cyclen")
    quantile_weight: float = 0.2
    # (3) Promote max_assembly_burnup from advisory (NaN surrogate column 5) to a
    #     first-class regression target; the global head grows by one output and
    #     the label is masked wherever absent.
    promote_max_asm_bu: bool = False
    # (3b) Promote f_xy (MASTER's FXYP, pin planar peaking) to a first-class
    #      regression target — the 8th/9th head row, APPENDED exactly as
    #      max_assembly_burnup was.  The label exists on ~2% of store rows, so the
    #      row trains on the masked subset only, and it regresses the RESIDUAL
    #      against the measured ``F_xy ~ F_r`` affine rather than the absolute
    #      value (net.PosValNet._compose_fxy; design 20260829 §3.4).  Training
    #      REFUSES if the train fold carries fewer than
    #      ``physics_prior.MIN_FXY_LABELS`` labelled rows.
    promote_fxy: bool = False
    # (3b-i) Compose the f_xy row against the ``a*f_r + b`` prior (the default,
    #      and what the 20260829 arm-1 run shipped) or predict f_xy DIRECTLY.
    #      With ``--freeze-trunk-cyclen`` the mu head is a LINEAR probe on a
    #      frozen embedding and ``mu[f_r]`` is another linear probe on that same
    #      embedding, so "prior + linear residual" and "direct" span the SAME
    #      function class — measured offline on S1j (see Amendment B): 0.0701 vs
    #      0.0708 MAE, 0.761 vs 0.766 rho.  The prior is therefore an
    #      INITIALIZATION and a graceful-degradation floor, not extra capacity,
    #      and the direct mode exists to measure that claim rather than assume it.
    fxy_prior_residual: bool = True
    # (3b-ii) Fit that prior on the model's OWN predicted F_r instead of on the
    #      MEASURED F_r.  The composition (net._compose_fxy) reads the raw
    #      ``mu[f_r]`` row, which carries the F_r head's uniform under-prediction
    #      (bias -0.0655 on s1i) that ``f_r_calibration.json`` exists to remove --
    #      and that artifact is applied in ``predict``, i.e. AFTER the composition
    #      has already happened inside the net.  A measured-F_r-fitted prior
    #      therefore inherits ``a * bias`` = -0.081, exactly the arm-1 f_xy bias.
    #      Fitting on the model's own predicted F_r absorbs it and keeps train and
    #      serve reading the SAME row (offline: floor MAE 0.1056 -> 0.0894, bias
    #      -0.077 -> +0.014).  Costs one eval-mode pass over the labelled train
    #      rows per member, after ``--init-from``.
    fxy_prior_on_predicted: bool = False
    # (3b-iii) Weight of the f_xy val score in the best-epoch / early-stop
    #      criterion.  0.0 (default) reproduces the legacy selection EXACTLY.
    #      ``fxy_metrics`` deliberately stays out of ``composite_metric`` -- a
    #      2%-labelled axis must not silently move a general-purpose retrain's
    #      checkpoint -- but a run whose ONLY purpose is the f_xy head must be
    #      able to select on it: arm 1 picked epochs 4-37 on a composite blind to
    #      f_xy, before its own LR warmup ended, and shipped an untrained residual.
    fxy_select_weight: float = 0.0
    # (3b-iv) arm 5 (prereg Amendment E.3): a within-cell pairwise margin-rank
    #      hinge on the COMPOSED f_xy row.  The r2 RANK gate demoted the arm-4
    #      head to a LEVEL estimator because a level objective has no reason to
    #      order rows INSIDE a cell — measured on the exploit slot, the
    #      estimator's own level error (0.0117) is the size of the whole spread
    #      it must order (0.0114) — while E.1.2 showed the ordering axis IS in
    #      the trunk embedding (leave-one-wave-out ridge +0.4688 vs a serving
    #      head +0.2858).  So the term that was missing is an objective, not a
    #      feature.  ``fxy_rank_weight = 0.0`` (the default) never builds the
    #      term: every existing training path is byte-identical.
    fxy_rank_weight: float = 0.0
    fxy_rank_margin_z: float = 0.1
    fxy_rank_min_gap: float = 0.005       # 3 orders above MASTER FXYP repeat noise
    fxy_rank_low_thresh: float = 1.60     # up-weight the low-f_xy boundary band
    fxy_rank_low_weight: float = 3.0
    #      "gate" == (case_pair, feed), the partition every f_xy gate scores on;
    #      "legacy" == the cyclen cell (feed, e_core-bin, dataset).  The two do
    #      not nest (E.1.4), and the registered choice is the gate's.
    fxy_rank_cell: str = "gate"
    # (3b-v) Restrict the f_xy SELECTION metric to the elite band the ranking
    #      clause (G6a) is scored on: within each GATE cell, the rows whose
    #      MEASURED f_xy is at or below that cell's ``band`` quantile.  1.0 (the
    #      default) is the legacy metric exactly — every labelled row, grouped by
    #      ``case_pair`` alone.  Note the registered mismatch this closes: the
    #      legacy metric groups by ``case_pair`` while every gate groups by
    #      ``(case_pair, feed)``, so best-epoch selection was reading a coarser
    #      partition than the bar it is judged against (prereg E.8-⑥).
    fxy_select_band: float = 1.0
    # --- post-train artifacts -------------------------------------------------
    # Fit the per-cell cyclen + F_r affine calibrations into the new model dir at
    # the end of a retrain (train-split rows only, leakage-asserted).  This writes
    # ADDITIONAL sidecar files; ``model.pt`` / ``meta.json`` are untouched, so the
    # trained weights are unaffected either way.
    auto_fit_cell_calibration: bool = True
    calibration_library_id: str = "ga80"
    # --- (4) soft-target distillation from per-cell historical teachers -------
    # ``distill_targets`` is a path to a prebuilt cache (see distill.py); with
    # None the term is absent and training is byte-identical.  The soft target is
    # a z-scored Huber pull toward the teacher's mean, masked to the distillable
    # targets (pin/assembly burnup excluded) — the FULL corpus is used and the
    # 5-member ensemble is preserved, per the recorded decision.
    distill_targets: str | None = None
    distill_weight: float = 0.3
    #: arm1(b) cyclen protection: multiply the cyclen-column distill loss by this
    #: factor for rows whose e_core falls in ``distill_cyclen_boost_bands`` — pins
    #: the champion teacher's cyclen ranking on the gate-failing bands.  1.0 = OFF.
    distill_cyclen_boost_factor: float = 1.0
    distill_cyclen_boost_bands: str = ""   # e.g. "5.0-5.25,6.0-6.25" (e_core lo-hi, hi exclusive)
    #: minimum fraction of the cache's built soft-target rows that must survive the
    #: record_id join, else training hard-errors (a <50% match means the arm would
    #: silently duplicate the baseline).  0 disables the guard.
    distill_min_match_frac: float = 0.5
    num_workers: int = 0
    augment: bool = True
    min_case_val: int = 10           # min val patterns/case for within-case Spearman
    map_norm_subset: int = 5000      # rows sampled for the map z-score constants
    round_trip_rows: int = 64        # val rows saved for the load round-trip
    # --- GPU-throughput knobs (defaults preserve the CPU/256-batch semantics) --
    batch_size_cuda: int = 1024      # effective batch on CUDA (CPU keeps batch_size)
    base_batch: int = 256            # reference batch for linear LR / warmup scaling
    batch_size_explicit: bool = False  # user pinned --batch-size (skip auto-pick)
    lr_scaling: bool = True          # linear LR scale with effective/base batch ratio
    warmup_step_scaling: bool = True   # keep the μ-only warmup *step* count constant
    device_resident: bool = True     # hold the precomputed tensors on-GPU (CUDA only)
    max_resident_gib: float = 40.0   # safety cap for the device-resident dataset
    parallel_members: int = 1        # ensemble members trained jointly in one process
    torch_compile: bool = False      # wrap member forward in torch.compile (opt-in)
    # --- network-shape knobs (defaults == PosValNetConfig defaults) ------------
    # These size the member CNN.  At their defaults the constructed
    # PosValNetConfig is field-for-field identical to the pre-flag one, so the
    # init (under the per-member manual_seed) is byte-identical — the flag-off
    # regression path is unchanged (proven by test).  A capacity sweep (e.g.
    # width 112 -> 160, ~1.6M -> ~3.15M params/member) flips only these.  They
    # are stamped into meta.net_config, and the serving path rebuilds the member
    # from that meta, so model_api needs no change (verified by a load test).
    width: int = 112
    n_blocks: int = 6
    head_hidden: int = 256
    # --- freeze-and-finetune (default OFF == the legacy from-scratch path) ------
    # ``init_from`` is a champion model dir: each member's PosValNet is loaded from
    # the champion's corresponding ``member_*`` state_dict (strict) BEFORE training,
    # turning the run into a fine-tune.  ``freeze_trunk_cyclen`` (requires
    # ``init_from``) freezes the shared trunk + the cyclen output rows so CYCLEN
    # predictions stay byte-identical to the champion while F_r + node_peak/map
    # heads adapt.  Both default None/False, so the model init, optimizer and step
    # are field-for-field the legacy path when they are off (state_dict-identical).
    init_from: str | None = None
    freeze_trunk_cyclen: bool = False
    # ``trunk_finetune_lr_mult`` (prereg Amendment D-③, arm 4) splits that bundle:
    # > 0 keeps the CYCLEN half of ``freeze_trunk_cyclen`` (row-masked cyclen
    # gradients, weight-decay-free masked heads, champion cyclen prior + per-cell
    # calibration COPIED not re-fit) but does NOT freeze the trunk — it trains in
    # its own optimizer group at ``lr * mult``.  Requires ``init_from`` (fine-
    # tuning a random trunk at 5% LR is not a fine-tune) and is mutually
    # exclusive with ``freeze_trunk_cyclen``.  0.0 == OFF == byte-identical to
    # the pre-flag path (no extra param group, the legacy scheduler object).
    trunk_finetune_lr_mult: float = 0.0
    # --- hires bundle (design doc data/reports/hires_model_ab_design_20260725.md)
    # Three independent, default-OFF knobs.  All three off == the pre-hires path,
    # module-for-module and byte-for-byte (regression-tested).
    #
    # (A1) ``map_head_mode="multiscale"`` swaps the 644-parameter 1x1 map head for
    #      a decoder that also reads the stem output and two intermediate block
    #      taps, giving the map a path that has NOT been through the full residual
    #      low-pass cascade.
    # (A2) ``map_prior_residual`` wires the cond_v6 ``prior_power`` channel into
    #      the map head as an additive, learnably-scaled skip, so the head learns
    #      the residual against the leading-order diffusion solve.  Requires a
    #      cond_schema that carries the channel (``v6`` / ``v6_prior``).
    # (A3) ``map_spectral_weight`` adds the band-weighted FFT term
    #      (:func:`spectral_map_loss`); 0.0 is a pure no-op.
    map_head_mode: str = "linear"
    map_prior_residual: bool = False
    map_spectral_weight: float = 0.0
    # --- axial bundle (decision D10; contract in lpopt/data/axial.py) ---------
    # ``axial_head`` adds a head predicting the EDIT6 axial power PROFILE at the
    # BOC and EOC burnup anchors, as ``axial_rank`` per-anchor coefficients in a
    # train-fold-fit PCA shape basis (:mod:`lpopt.data.axial`).  F_z, AO and ASI
    # are then exact analytic functions of the emitted profile — which matters
    # because |AO| is a FIRST MOMENT and cannot distinguish a saddle from a
    # double hump (measured: EOC saddle depth correlates -0.58 with |AO| and
    # +0.985 with F_z).  Default OFF: with the flag off no module is registered,
    # no label is read, no loss term is added and no meta key changes, so the
    # champion path is byte-identical (regression-tested).
    axial_head: bool = False
    axial_rank: int = 6
    axial_weight: float = 0.2
    # --- A/B round-2 arm A1: burnup-TRAJECTORY supervision (default OFF) -------
    # ``traj_weight > 0`` turns on auxiliary supervision of the EXISTING map head
    # at intermediate burnup steps: the ``<rid>__traj`` labels
    # (:mod:`lpopt.data.traj`) are read at ``traj_anchors`` cycle-burnup fractions
    # and the map decoder is re-run on a trunk feature FiLM-modulated by
    # ``[globals, burnup_fraction]``, reading planes
    # :data:`lpopt.model.net.TRAJ_MAP_CHANNELS`.  No new decoder, no new
    # normalisation: at fraction 1 the readout's labels ARE the eoc_* map planes
    # (verified bit-for-bit on the store), so the term is a strict extension of
    # the supervision the champion already has.
    #
    # Target: cyclen / cbc_max variance.  ``cyclen`` is the ENDPOINT of the boron
    # let-down trajectory; a model supervised only at BOC and EOC never sees the
    # path whose end it has to predict.
    #
    # ``0.0`` (default) reads no label, registers no module, adds no loss term and
    # writes no meta key — the champion-identical path (regression-tested).
    traj_weight: float = 0.0
    traj_anchors: tuple[float, ...] = TRAJ_ANCHORS
    # --- A/B round-2 arm A2: per-provenance CBC label offsets (default OFF) ----
    # A learned scalar per CBC label-convention provenance group
    # (:data:`lpopt.model.dataset_torch.CBC_PROVENANCE_GROUPS`), added to the
    # model's cbc_max prediction *inside the cbc regression loss only* so the
    # residual is taken in the row's OWN label convention.  Group 0
    # (``master_native``) has no parameter, so the served prediction is the
    # MASTER-native convention by construction.  Nothing else in the loss — rank
    # hinges, distillation, quantiles, map terms — sees the offset.
    #
    # Target: cbc_max variance.  Dataset A (59% of the cbc-labelled corpus) sits
    # +100..410 ppm above MASTER-native at matched (feed, e_core); pooling the two
    # conventions injects exactly that bimodal error into every mixed cell.
    cbc_provenance_offset: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def _valid_target_values(df: pd.DataFrame, name: str) -> np.ndarray:
    """Trustworthy, finite target values (cbc_max additionally drops boc_only).

    ``trustworthy`` (converged AND not quarantined), not ``converged`` alone: a
    ``valid=False`` row's labels describe a core that was never loaded, and the
    z-scoring statistics they would shift reach EVERY head.
    """
    converged = trustworthy(df).to_numpy()
    vals = pd.to_numeric(df[name], errors="coerce").to_numpy()
    ok = converged & np.isfinite(vals)
    if name == "cbc_max" and "cbc_kind" in df.columns:
        ok = ok & (df["cbc_kind"].astype(str).to_numpy() != "boc_only")
    return vals[ok]


def compute_target_norm(df: pd.DataFrame,
                        targets: Sequence[str] = TARGETS
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-target (mean, std) over the train split's converged/valid rows.

    ``targets`` defaults to :data:`TARGETS`, so the legacy call is unchanged.
    Under ``cyclen_physics_prior`` the caller passes a frame whose ``cyclen``
    column has already been replaced by the RESIDUAL (see
    :func:`residual_target_frame`), so the z-score constants are the residual's.
    """
    mean = np.zeros(len(targets), dtype=np.float64)
    std = np.ones(len(targets), dtype=np.float64)
    for k, name in enumerate(targets):
        vals = _valid_target_values(df, name)
        if vals.size:
            mean[k] = float(vals.mean())
            s = float(vals.std())
            std[k] = s if s > 1e-9 else 1.0
    return mean, std


def compute_map_norm(reader: StoreReader, df: pd.DataFrame, subset: int,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) of the EDIT5 maps over a train-row subset."""
    keys = df.loc[df["maps_key"].notna(), "maps_key"].astype(str).tolist()
    if not keys:
        return np.zeros(4, dtype=np.float64), np.ones(4, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(keys) > subset:
        keys = [keys[i] for i in rng.choice(len(keys), subset, replace=False)]
    acc = [[] for _ in range(4)]
    for key in keys:
        arr = reader.maps(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32).reshape(4, 9, 9)
        for c in range(4):
            plane = arr[c]
            acc[c].append(plane[np.isfinite(plane)])
    mean = np.zeros(4, dtype=np.float64)
    std = np.ones(4, dtype=np.float64)
    for c in range(4):
        if acc[c]:
            flat = np.concatenate(acc[c])
            if flat.size:
                mean[c] = float(flat.mean())
                s = float(flat.std())
                std[c] = s if s > 1e-9 else 1.0
    return mean, std


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def _masked_mean(per_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom <= 0:
        return per_elem.new_zeros(())
    return (per_elem * mask).sum() / denom


def regression_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target_z: torch.Tensor,
    mask: torch.Tensor,
    *,
    use_nll: bool,
    beta: float,
    delta: float,
) -> torch.Tensor:
    """z-scored Huber, wrapped by β-NLL after warmup (μ-only during warmup)."""
    # Masked positions carry NaN raw targets; zero them so ``NaN*0`` never
    # poisons the masked mean (they are excluded by ``mask`` regardless).
    target_z = torch.nan_to_num(target_z, nan=0.0)
    huber = F.smooth_l1_loss(mu, target_z, beta=delta, reduction="none")
    if not use_nll:
        return _masked_mean(huber, mask)
    log_sigma = log_sigma.clamp(-7.0, 7.0)
    precision = torch.exp(-2.0 * log_sigma)              # 1 / sigma^2
    # Huberized Gaussian NLL: 0.5 * huber/σ² + log σ  (const dropped).
    nll = 0.5 * huber * precision + log_sigma
    # β-NLL weight (σ²)^β, detached, prevents variance inflation (Seitzer 2022).
    weight = torch.exp(2.0 * log_sigma * beta).detach()
    return _masked_mean(weight * nll, mask)


def top_k_slot_weight(map_z: torch.Tensor, mask: torch.Tensor, k: int,
                      weight: float) -> torch.Tensor:
    """``1 + weight`` on the K hottest VALID slots of each (row, channel) plane.

    Selection is on the LABEL, never the prediction: the peak's *location* is the
    noise-free part of a map label (22/22 exact reproduction under the diagonal
    transpose check, ``transpose_noise_measured_20260725``), whereas its *value*
    carries the harvest's float16 quantum.  Selecting on the label therefore
    puts the extra gradient on a set that does not move with the model, which is
    what makes this a supervision knob rather than a self-confirming one.

    Invalid slots are pushed to ``-inf`` before the ``topk`` so a masked cell can
    never be selected; ``k`` is clamped to the number of slots present.  Returns
    an all-ones tensor (an exact no-op) when ``weight <= 0`` or ``k <= 0``.
    """
    if weight <= 0.0 or int(k) <= 0:
        return map_z.new_ones(())
    flat_t = torch.nan_to_num(map_z, nan=float("-inf")).flatten(2)   # [B,C,S]
    flat_m = mask.flatten(2) > 0
    scored = torch.where(flat_m, flat_t, flat_t.new_full((), float("-inf")))
    kk = min(int(k), scored.shape[-1])
    idx = scored.topk(kk, dim=-1).indices
    hot = torch.zeros_like(scored, dtype=map_z.dtype)
    hot.scatter_(-1, idx, 1.0)
    hot = hot * flat_m.to(map_z.dtype)          # never weight a masked slot
    return (1.0 + float(weight) * hot).view_as(map_z)


def map_loss(map_pred: torch.Tensor, map_z: torch.Tensor,
             mask: torch.Tensor, delta: float = 1.0,
             peak_weight: float = 0.0,
             peak_topk: int = 0, peak_topk_weight: float = 0.0) -> torch.Tensor:
    """Masked Huber on the (B,4,9,9) z-space map stack.

    ``peak_weight`` (node_peak-B, forensic 20260724): when > 0, up-weight the
    per-node loss where the z-space TARGET is above its per-channel mean (hot /
    peak nodes, ``relu(map_z)``) by ``1 + peak_weight*relu(map_z)``.  This
    sharpens the map head exactly at the node-peaking the ``predict_map_peak``
    derives node_peak from, without a records-schema column (Plan B).  ``0`` is
    byte-identical to the pre-existing uniform map loss.

    ``peak_topk_weight`` (A/B round-2 arm A3): a finer-grained RANK-based
    complement — a further ``1 + peak_topk_weight`` on the ``peak_topk`` hottest
    valid slots of each plane (:func:`top_k_slot_weight`).  The two knobs
    MULTIPLY, so the continuous term still shapes the warm region while the
    rank term concentrates on the order statistics ``F_q`` / ``node_peak``
    actually are.  ``0`` is again byte-identical.
    """
    tgt = torch.nan_to_num(map_z)
    per_elem = F.smooth_l1_loss(map_pred, tgt, beta=delta, reduction="none")
    if peak_weight > 0.0:
        per_elem = per_elem * (1.0 + float(peak_weight) * F.relu(tgt))
    if peak_topk_weight > 0.0 and int(peak_topk) > 0:
        per_elem = per_elem * top_k_slot_weight(
            map_z, mask, int(peak_topk), float(peak_topk_weight))
    return _masked_mean(per_elem, mask)


def traj_loss(traj_pred: torch.Tensor, traj_z: torch.Tensor,
              mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Masked Huber on the ``[B, A, P, 9, 9]`` z-space burnup-trajectory planes.

    ``mask`` is ``[B, A]`` (one flag per burnup anchor) and broadcasts over the
    plane/row/col axes, exactly as :func:`axial_loss`'s does over its mode axis.
    Per-SLOT validity comes from ``isfinite(traj_z)`` — the 12 non-slot cells of
    each 9x9 quarter are NaN by construction (:mod:`lpopt.data.traj`), the same
    rule ``LPDataset._maps`` uses to build ``maps_mask`` — so no separate
    ``[B,A,P,9,9]`` mask tensor has to be carried.

    A record with no trajectory, an anchor the record cannot honestly support,
    and a batch where no row carries one, all contribute exactly zero.
    """
    finite = torch.isfinite(traj_z).to(traj_pred.dtype)
    m = mask.to(traj_pred.dtype).view(*mask.shape, 1, 1, 1) * finite
    tgt = torch.nan_to_num(traj_z)
    per_elem = F.smooth_l1_loss(traj_pred, tgt, beta=delta, reduction="none")
    return _masked_mean(per_elem, m)


#: Radial-wavenumber band edges for :func:`spectral_map_loss`, and the weight each
#: band carries.  Identical binning to the 2-D FFT analysis in
#: ``data/reports/cyclen_nodepeak_resolution_20260725.md`` §3.6, which measured
#: the predicted/actual power ratio decaying 1.00 -> 0.71 from the lowest band to
#: Nyquist.  Weights rise with wavenumber because the ATTENUATION rises with
#: wavenumber while the raw power (and hence the plain Huber gradient) falls: the
#: lowest two bands already hold 71-74% of the map power, so an unweighted loss
#: is dominated by exactly the component that is not broken.
_SPECTRAL_BANDS: tuple[tuple[float, float, float], ...] = (
    (0.00, 0.13, 1.0),
    (0.13, 0.25, 1.0),
    (0.25, 0.36, 2.0),
    (0.36, 0.47, 4.0),
    (0.47, 1.00, 4.0),
)
_SPECTRAL_CACHE: dict[tuple[Any, Any], torch.Tensor] = {}


def _spectral_band_weights(device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
    """``[9, 9]`` per-mode weight map (DC zeroed) for the 9x9 quarter FFT."""
    key = (device, dtype)
    hit = _SPECTRAL_CACHE.get(key)
    if hit is not None:
        return hit
    k = torch.fft.fftfreq(_MAP_SIDE, device=device, dtype=torch.float32)
    kr = torch.sqrt(k.view(-1, 1) ** 2 + k.view(1, -1) ** 2)
    w = torch.zeros_like(kr)
    for lo, hi, weight in _SPECTRAL_BANDS:
        w = torch.where((kr >= lo) & (kr < hi), kr.new_tensor(weight), w)
    w[0, 0] = 0.0                       # DC carries no shape information
    out = w.to(dtype)
    _SPECTRAL_CACHE[key] = out
    return out


def spectral_map_loss(map_pred: torch.Tensor, map_z: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Band-weighted 2-D FFT amplitude loss on the map fluctuation field.

    Both fields are masked to the 69 valid slots, have their slot-mean removed
    (so this term says nothing about the level — that is the Huber term's job)
    and are transformed with the SAME window, exactly as the report's spectrum
    analysis does.  The loss is the band-weighted mean squared difference of the
    complex Fourier coefficients, so it penalizes getting a high-wavenumber mode's
    amplitude OR phase wrong.

    Rows with no map label contribute nothing (their mask is all zero).
    """
    tgt = torch.nan_to_num(map_z)
    valid = mask.to(map_pred.dtype)
    n_valid = valid.sum(dim=(2, 3), keepdim=True)
    if float(n_valid.sum()) <= 0.0:
        return map_pred.sum() * 0.0
    denom = n_valid.clamp_min(1.0)
    pred_c = (map_pred * valid) - (map_pred * valid).sum(dim=(2, 3), keepdim=True) / denom
    tgt_c = (tgt * valid) - (tgt * valid).sum(dim=(2, 3), keepdim=True) / denom
    pred_c = pred_c * valid
    tgt_c = tgt_c * valid
    # bf16 has no FFT kernel and the transform is tiny; do it in fp32.
    diff = (torch.fft.fft2(pred_c.float(), dim=(-2, -1))
            - torch.fft.fft2(tgt_c.float(), dim=(-2, -1)))
    w = _spectral_band_weights(map_pred.device, torch.float32)
    per_row = (diff.real ** 2 + diff.imag ** 2) * w
    # Normalize by the mode count and by the number of labelled planes so the
    # term's scale is comparable to the Huber map loss.
    row_has_label = (n_valid.squeeze(-1).squeeze(-1) > 0).float()
    scale = float(_MAP_SIDE * _MAP_SIDE) * float(w.sum())
    num = (per_row.sum(dim=(2, 3)) * row_has_label).sum()
    den = row_has_label.sum().clamp_min(1.0) * scale
    return (num / den).to(map_pred.dtype)


def axial_loss(axial_pred: torch.Tensor, axial_coeff_z: torch.Tensor,
               mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Masked Huber on the ``[B, A, K]`` standardised axial shape coefficients.

    ``mask`` is ``[B, A]`` (one flag per burnup anchor) and broadcasts over the
    mode axis: a record with no axial label contributes nothing, and the term is
    a no-op on a batch where none of the rows carries one.

    The coefficients are standardised PER MODE by
    :meth:`~lpopt.data.axial.AxialBasis.z_encode`, which is the same argument
    :func:`spectral_map_loss` makes in the radial plane: mode 0 holds ~80% of the
    axial shape variance, so an unstandardised loss would leave the higher modes
    — the saddle/double-hump structure — with essentially no gradient.
    """
    tgt = torch.nan_to_num(axial_coeff_z)
    per_elem = F.smooth_l1_loss(axial_pred, tgt, beta=delta, reduction="none")
    return _masked_mean(per_elem, mask.unsqueeze(-1).expand_as(per_elem))


def convergence_loss(logit: torch.Tensor, label: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    per_elem = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    return _masked_mean(per_elem, mask)


def _parse_ecore_bands(spec: str) -> list[tuple[float, float]]:
    """``"5.0-5.25,6.0-6.25"`` -> ``[(5.0,5.25),(6.0,6.25)]`` (hi exclusive at use)."""
    bands: list[tuple[float, float]] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            lo, hi = tok.split("-")
            bands.append((float(lo), float(hi)))
        except ValueError:
            continue
    return bands


def _parse_traj_anchors(spec: str) -> tuple[float, ...]:
    """``"0,0.25,0.5,0.75,1"`` -> ``(0.0, 0.25, 0.5, 0.75, 1.0)``.

    Values outside ``[0, 1]`` are dropped (a cycle-burnup FRACTION cannot be
    either), duplicates are collapsed, and the result is sorted — so the anchor
    order is a deterministic function of the set, which is what the head's
    ``[B, A, ...]`` axis and the stamped meta depend on.
    """
    vals: list[float] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            vals.append(round(v, 6))
    return tuple(sorted(set(vals)))


def cyclen_rank_loss(
    mu_cyclen_z: torch.Tensor,
    raw_cyclen: torch.Tensor,
    valid: torch.Tensor,
    cell_code: torch.Tensor,
    *,
    margin: float,
    min_gap_efpd: float,
) -> torch.Tensor:
    """Within-cell margin-ranking hinge on cyclen (RL forensic 20260720).

    For every ORDERED same-cell pair ``(i, j)`` whose RAW cyclen gap
    ``raw_i - raw_j > min_gap_efpd`` (so the ordering is real, not MASTER
    convergence noise), penalize the model unless its z-space cyclen prediction
    ranks ``i`` above ``j`` by at least ``margin``: ``relu(margin - (mu_i - mu_j))``.
    ``raw_i - raw_j > min_gap`` is strictly positive, so each real ordering is
    counted once (no double count).  Only rows with ``valid`` (converged + finite
    cyclen) and a resolved cell (``cell_code >= 0``) participate; a batch with no
    qualifying pair contributes zero.  Scale-free in the target (uses only the
    sign of the raw gap) — the whole point is independence from the global z-scale.
    """
    v = valid.bool() & (cell_code >= 0)
    if int(v.sum()) < 2:
        return mu_cyclen_z.new_zeros(())
    mu = mu_cyclen_z[v].float()
    raw = raw_cyclen[v].float()
    code = cell_code[v]
    same = code.unsqueeze(0) == code.unsqueeze(1)          # [K, K]
    d_raw = raw.unsqueeze(1) - raw.unsqueeze(0)            # i - j
    pair = same & (d_raw > float(min_gap_efpd))            # i genuinely above j
    npair = int(pair.sum())
    if npair == 0:
        return mu_cyclen_z.new_zeros(())
    d_mu = mu.unsqueeze(1) - mu.unsqueeze(0)               # want > 0 when pair
    hinge = torch.relu(float(margin) - d_mu)
    return (hinge * pair).sum() / npair


def f_r_rank_loss(
    mu_fr_z: torch.Tensor,
    raw_fr: torch.Tensor,
    valid: torch.Tensor,
    cell_code: torch.Tensor,
    *,
    margin: float,
    min_gap: float,
    low_thresh: float,
    low_weight: float,
) -> torch.Tensor:
    """Within-cell margin-ranking hinge on F_r with low-F_r pair up-weighting.

    Structurally identical to :func:`cyclen_rank_loss` (same-cell ordered pairs
    whose RAW F_r gap ``raw_i - raw_j > min_gap`` are penalized unless the model
    ranks ``i`` above ``j`` by ``margin`` in z-space), but each pair is WEIGHTED by
    ``low_weight`` when ``min(raw_i, raw_j) <= low_thresh`` (else 1.0) so the
    gradient concentrates on the low-F_r boundary the 1.55 search queries — where
    the champion's within-cell ordering is weakest (parity_round1c_20260722, ρ≈0.13
    for F_r<1.65).  Scale-free in the target (sign of the raw gap only).  A batch
    with no qualifying pair contributes zero.
    """
    v = valid.bool() & (cell_code >= 0)
    if int(v.sum()) < 2:
        return mu_fr_z.new_zeros(())
    mu = mu_fr_z[v].float()
    raw = raw_fr[v].float()
    code = cell_code[v]
    same = code.unsqueeze(0) == code.unsqueeze(1)          # [K, K]
    d_raw = raw.unsqueeze(1) - raw.unsqueeze(0)            # i - j
    pair = same & (d_raw > float(min_gap))                 # i genuinely above j
    if int(pair.sum()) == 0:
        return mu_fr_z.new_zeros(())
    # per-pair weight: boundary pairs (min raw f_r in the pair <= low_thresh) up.
    pair_min = torch.minimum(raw.unsqueeze(1), raw.unsqueeze(0))
    w = torch.where(pair_min <= float(low_thresh),
                    mu.new_tensor(float(low_weight)), mu.new_tensor(1.0))
    w = w * pair.float()
    d_mu = mu.unsqueeze(1) - mu.unsqueeze(0)               # want > 0 when pair
    hinge = torch.relu(float(margin) - d_mu)
    denom = w.sum()
    if float(denom) <= 0.0:
        return mu_fr_z.new_zeros(())
    return (hinge * w).sum() / denom


def f_xy_rank_loss(
    mu_fxy_z: torch.Tensor,
    raw_fxy: torch.Tensor,
    valid: torch.Tensor,
    cell_code: torch.Tensor,
    *,
    margin: float,
    min_gap: float,
    low_thresh: float,
    low_weight: float,
    stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """Within-cell margin-ranking hinge on the COMPOSED ``f_xy`` row (prereg E.8-①).

    The arithmetic is exactly :func:`f_r_rank_loss`'s — same-cell ordered pairs
    whose RAW gap exceeds ``min_gap`` are penalized unless the model ranks them
    apart by ``margin`` in z-space, with pairs whose ``min(raw_i, raw_j) <=
    low_thresh`` up-weighted — so it is CALLED rather than copied; what is new is
    everything around it, and that is why this wrapper exists as a named term:

    * it is attached to ``out["mu"][:, fxy_idx]``, the row
      :meth:`~lpopt.model.net.PosValNet._compose_fxy` has already ADDED the
      ``a*mu[f_r]+b`` prior onto, so the hinge ranks the served quantity.  No
      prior is added back here (unlike :func:`cyclen_rank_loss`, whose prior
      lives outside the net) — doing so would double-count it;
    * ``mu[f_r]`` inside that composition is ``detach()``-ed, so this term
      structurally cannot reach the F_r head, which is what keeps a ~2%-labelled
      axis from perturbing the seven dense targets;
    * the cell is the GATE cell ``(case_pair, feed)``
      (:func:`~lpopt.model.dataset_torch.fxy_cell_codes`), not the cyclen cell;
    * the target is *minimized*, so ``low_thresh`` (1.60) selects the low-``f_xy``
      boundary the F_xy <= 1.65 search actually queries.

    ``stats`` (optional) is filled in-place with the batch's ``n_pairs`` /
    ``n_cells`` census.  A term that silently sees no pair is the failure mode
    this arm cannot detect from the loss value alone (it is 0 either way), so the
    trainer always asks for it (prereg E.8-⑦).  Passing ``None`` costs nothing.
    """
    if stats is not None:
        with torch.no_grad():
            v = valid.bool() & (cell_code >= 0)
            n_pairs = 0
            n_cells = 0
            if int(v.sum()) >= 2:
                raw = raw_fxy[v].float()
                code = cell_code[v]
                pair = ((code.unsqueeze(0) == code.unsqueeze(1))
                        & ((raw.unsqueeze(1) - raw.unsqueeze(0)) > float(min_gap)))
                n_pairs = int(pair.sum())
                if n_pairs:
                    contrib = pair.any(dim=0) | pair.any(dim=1)
                    n_cells = int(torch.unique(code[contrib]).numel())
            stats["n_pairs"] = float(n_pairs)
            stats["n_cells"] = float(n_cells)
    return f_r_rank_loss(mu_fxy_z, raw_fxy, valid, cell_code, margin=margin,
                         min_gap=min_gap, low_thresh=low_thresh,
                         low_weight=low_weight)


def map_fr_consistency_loss(
    map_pred: torch.Tensor,
    mu_fr_z: torch.Tensor,
    fr_mask: torch.Tensor,
    map_mask: torch.Tensor,
    boc_power_ch: int,
) -> torch.Tensor:
    """Within-batch agreement between the boc_power map peak and the F_r head.

    Both the spatial MAX of the predicted ``boc_power`` map channel and the F_r
    head encode radial power peaking, so tying them regularizes the crowded-band
    F_r ordering with the map's spatial signal.  To be scale/units-free the two
    quantities are STANDARDIZED within the batch (subtract mean, divide by std)
    over the rows valid for BOTH, then a Huber pulls them together — so the loss
    rewards CO-MOVEMENT (a sample with a higher predicted peak should have a higher
    predicted F_r), not a spurious absolute match between a z-map and a z-target.
    Returns 0 for a batch with < 2 jointly-valid rows.
    """
    m = fr_mask.bool() & map_mask.bool()
    if int(m.sum()) < 2:
        return mu_fr_z.new_zeros(())
    peak = map_pred[:, boc_power_ch].flatten(1).amax(dim=1)[m].float()   # [K]
    fr = mu_fr_z[m].float()
    def _std(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / (x.std(unbiased=False) + 1.0e-6)
    return F.smooth_l1_loss(_std(peak), _std(fr), beta=1.0)


def pinball_loss(q_pred: torch.Tensor, target_z: torch.Tensor,
                 mask: torch.Tensor, levels: Sequence[float]) -> torch.Tensor:
    """Masked multi-quantile pinball (check) loss.

    ``q_pred`` is ``[B, K, Q]`` z-space quantile predictions for ``K`` targets at
    ``Q`` levels; ``target_z`` / ``mask`` are ``[B, K]``.  For each level ``tau``
    the check function ``max(tau*e, (tau-1)*e)`` with ``e = y - q`` is minimized
    by the ``tau``-quantile of the conditional distribution, which is what makes
    the q10/q90 pair an honest (distribution-free) interval rather than a
    Gaussian-sigma restatement.  Masked entries contribute nothing; a batch with
    no valid entry returns 0.
    """
    if q_pred.numel() == 0 or not len(levels):
        return q_pred.new_zeros(())
    y = torch.nan_to_num(target_z, nan=0.0).unsqueeze(-1)      # [B,K,1]
    m = mask.unsqueeze(-1)                                     # [B,K,1]
    tau = torch.as_tensor(list(levels), dtype=q_pred.dtype,
                          device=q_pred.device).view(1, 1, -1)
    err = y - q_pred
    per = torch.maximum(tau * err, (tau - 1.0) * err)
    denom = m.sum() * q_pred.shape[-1]
    if denom <= 0:
        return q_pred.new_zeros(())
    return (per * m).sum() / denom


def residual_target_frame(df: pd.DataFrame, cyclen_prior: np.ndarray
                          ) -> pd.DataFrame:
    """A shallow copy of ``df`` with ``cyclen`` replaced by ``cyclen - prior``.

    Used only to derive the z-score constants for physics-prior residual
    learning; the dataset tensors keep the ABSOLUTE cyclen label so the rank loss
    and the reported metrics stay in physical EFPD.
    """
    out = df.copy()
    y = pd.to_numeric(out["cyclen"], errors="coerce").to_numpy(dtype=float)
    out["cyclen"] = y - np.asarray(cyclen_prior, dtype=float)
    return out


def resolve_fxy_prior(train_df: pd.DataFrame, norm_df: pd.DataFrame,
                      target_names: Sequence[str], *, verbose: bool = False,
                      prior_residual: bool = True,
                      ) -> tuple[int, int, Any | None, tuple[float, float]]:
    """Resolve the F_xy prior-residual head from the TRAIN frame.

    Returns ``(fxy_idx, ref_idx, prior, (A, B))``.  Without ``f_xy`` in
    ``target_names`` this is ``(-1, -1, None, (0.0, 0.0))``, i.e. every
    ``net_config`` field stays at its default and the built network is
    byte-identical to the pre-F_xy one.

    ``prior_residual=False`` (``--fxy-direct``) keeps the label guard and the
    fitted prior — which is still stamped into the meta as the REPORTED baseline
    — but returns ``ref_idx = -1``, so ``net.PosValNet`` builds with the
    composition OFF and the f_xy row predicts the ABSOLUTE value.  The mu head is
    linear, so under a frozen trunk this is the same hypothesis class as
    prior + linear residual; the flag exists so the two can be measured against
    each other instead of assumed equivalent.

    The prior is fitted ONCE, on the TRAIN frame only — the same leakage rule the
    cyclen physics prior and the diffusion power prior obey — and converted into
    the z space the ``mu`` head lives in (see
    :func:`~.physics_prior.fxy_prior_z`); ``norm_df`` is the frame the z-score
    constants come from, so under cyclen residual learning it is the residual
    frame, exactly as for every other target.

    **Guard.** Fewer than :data:`~.physics_prior.MIN_FXY_LABELS` labelled train
    rows raises.  A head fitted on a handful of labels would still produce a
    number for every served core, and that number would be indistinguishable at
    the API from a trained one — the design's F_xy gate sits at 1.65 with a prior
    residual sd of 0.029, so a head trained on noise is not a weak signal but a
    wrong feasibility claim.  Refusing loudly is the only honest option.
    """
    names = tuple(target_names)
    if "f_xy" not in names:
        return -1, -1, None, (0.0, 0.0)
    if "f_r" not in names:
        raise ValueError("promote_fxy needs an 'f_r' target to build the F_xy "
                         "prior from; target_names=%r" % (names,))
    fxy_idx = names.index("f_xy")
    ref_idx = names.index("f_r")
    prior = fit_fxy_prior(train_df, split="train")
    if prior.n_fit < MIN_FXY_LABELS:
        raise ValueError(
            f"promote_fxy: only {prior.n_fit} labelled f_xy train rows "
            f"(need >= {MIN_FXY_LABELS}); refusing to train an f_xy head on a "
            "label set too small to fit its prior or to learn a residual against "
            "it.  Backfill f_xy labels before promoting.")
    if not prior_residual:
        if verbose:
            print(f"=== f_xy head: DIRECT mode — no prior composition; the row "
                  f"regresses ABSOLUTE f_xy.  (Reported baseline prior f_xy = "
                  f"{prior.a:.4f}*f_r {prior.b:+.4f} on {prior.n_fit} labelled "
                  f"train rows, resid sd={prior.resid_sd:.4f}) ===", flush=True)
        return fxy_idx, -1, prior, (0.0, 0.0)
    t_mean, t_std = compute_target_norm(norm_df, names)
    ab = fxy_prior_z(prior, t_mean, t_std, fxy_idx, ref_idx)
    if verbose:
        print(f"=== f_xy head: prior f_xy = {prior.a:.4f}*f_r {prior.b:+.4f} on "
              f"{prior.n_fit} labelled train rows (r={prior.pearson:.4f}, "
              f"resid sd={prior.resid_sd:.4f}); head regresses the RESIDUAL, "
              f"z prior (A={ab[0]:.4f}, B={ab[1]:.4f}) ===", flush=True)
    return fxy_idx, ref_idx, prior, ab


def refit_fxy_prior_on_predicted(
    model: PosValNet, tensors: dict[str, torch.Tensor], *,
    fxy_idx: int, ref_idx: int, tmean: np.ndarray, tstd: np.ndarray,
    device: torch.device, batch_size: int = 512,
) -> tuple[Any, tuple[float, float]] | None:
    """Refit ``f_xy = a*f_r + b`` on the MODEL'S OWN predicted F_r, in eval mode.

    ``net.PosValNet._compose_fxy`` composes on the RAW ``mu[f_r]`` row — never on
    the ``f_r_calibration.json``-corrected column that ``predict`` serves, because
    that correction is applied outside the network, after the composition.  A
    prior fitted on the MEASURED F_r therefore hands the f_xy row the F_r head's
    whole systematic offset multiplied by ``a`` (measured: 1.2161 x -0.0655 =
    -0.0797 against an observed f_xy bias of -0.0809).  Fitting the same two
    scalars against the row the composition actually reads absorbs that offset
    and — unlike applying the calibration at serve time only — leaves train and
    serve reading the IDENTICAL quantity.

    Runs after ``--init-from`` has loaded the champion weights, on the labelled
    f_xy train rows only (the same rows :func:`~.physics_prior.fit_fxy_prior`
    uses, so the leakage rule is unchanged: no val row is touched).  Returns
    ``None`` — leaving the measured-F_r prior in place — when the labelled subset
    is degenerate; the caller has already enforced ``MIN_FXY_LABELS``.
    """
    mask = tensors["target_mask"][:, fxy_idx].detach().cpu() > 0
    rows = torch.nonzero(mask, as_tuple=True)[0]
    if int(rows.numel()) < 2:
        return None
    y = tensors["targets"].detach().cpu()[rows, fxy_idx].numpy().astype(float)
    cells = tensors["cells"]
    globals_ = tensors["globals"]
    was_training = model.training
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(rows.numel()), batch_size):
            sel = rows[start:start + batch_size]
            c = cells.index_select(0, sel.to(cells.device)).to(device)
            g = globals_.index_select(0, sel.to(globals_.device)).to(device)
            chunks.append(model(c, g)["mu"][:, ref_idx].float().cpu().numpy())
    model.train(was_training)
    x = np.concatenate(chunks) * float(tstd[ref_idx]) + float(tmean[ref_idx])
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 2 or float(np.ptp(x[ok])) < 1e-9:
        return None
    a, b = np.polyfit(x[ok], y[ok], 1)
    resid = y[ok] - (a * x[ok] + b)
    prior = FxyFrPrior(
        a=float(a), b=float(b), n_fit=int(ok.sum()),
        pearson=float(np.corrcoef(x[ok], y[ok])[0, 1]),
        resid_sd=float(resid.std()), split="train:predicted_f_r")
    return prior, fxy_prior_z(prior, tmean, tstd, fxy_idx, ref_idx)


# --------------------------------------------------------------------------- #
# normalized batch helper
# --------------------------------------------------------------------------- #
class _Norm:
    def __init__(self, tmean, tstd, mmean, mstd, device, cyclen_idx: int = _CYCLEN_IDX):
        self.tmean = torch.as_tensor(tmean, dtype=torch.float32, device=device)
        self.tstd = torch.as_tensor(tstd, dtype=torch.float32, device=device)
        self.mmean = torch.as_tensor(mmean, dtype=torch.float32, device=device).view(1, 4, 1, 1)
        self.mstd = torch.as_tensor(mstd, dtype=torch.float32, device=device).view(1, 4, 1, 1)
        self.cyclen_idx = int(cyclen_idx)

    def z_targets(self, raw: torch.Tensor,
                  cyclen_prior: torch.Tensor | None = None) -> torch.Tensor:
        """z-score the raw targets, optionally against the cyclen physics prior.

        With a prior the cyclen column becomes the RESIDUAL ``cyclen - prior``
        before z-scoring (``tmean``/``tstd`` for that column were computed on the
        residual, see :func:`residual_target_frame`).  ``cyclen_prior=None`` is
        the legacy path and is bit-identical to the previous implementation.
        """
        if cyclen_prior is None:
            return (raw - self.tmean) / self.tstd
        raw = raw.clone()
        raw[:, self.cyclen_idx] = raw[:, self.cyclen_idx] - cyclen_prior
        return (raw - self.tmean) / self.tstd

    def z_maps(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.mmean) / self.mstd

    def z_traj(self, raw: torch.Tensor) -> torch.Tensor:
        """z-score ``[B, A, P, 9, 9]`` trajectory planes with the MAP constants.

        The trajectory readout reuses map-head channels
        :data:`lpopt.model.net.TRAJ_MAP_CHANNELS`, and at burnup fraction 1 its
        labels ARE those channels' labels, so it must live in the same z-space —
        using separate constants would make the shared projection weights mean
        two different things.
        """
        ch = list(TRAJ_MAP_CHANNELS)[:raw.shape[2]]
        mean = self.mmean.view(-1)[ch].view(1, 1, -1, 1, 1)
        std = self.mstd.view(-1)[ch].view(1, 1, -1, 1, 1)
        return (raw - mean) / std


# --------------------------------------------------------------------------- #
# precomputed in-memory dataset
# --------------------------------------------------------------------------- #
class PrecomputedDataset(torch.utils.data.Dataset):
    """In-memory featurized rows (base + transposed) for fast, worker-free epochs.

    Featurization is done ONCE in the main process; each epoch is then pure
    tensor indexing.  This is both fast on GPU and sidesteps the shared-``maps.npz``
    file-handle corruption that ``num_workers>0`` triggers under a forked
    DataLoader (Python 3.11 zipfile overlap check).  ``augment`` picks the
    diagonal-transposed variant with probability 0.5 per ``__getitem__``.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], record_ids: list[str],
                 df: pd.DataFrame, *, augment: bool = False, seed: int = 0):
        self._t = tensors
        self.record_ids = record_ids
        self.df = df
        self.augment = bool(augment) and "cells_t" in tensors
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._t["cells"].shape[0]

    def __getitem__(self, i: int) -> dict[str, Any]:
        use_t = self.augment and self._rng.random() < 0.5
        cells = self._t["cells_t"][i] if use_t else self._t["cells"][i]
        item = {
            "record_id": self.record_ids[i],
            "cells": cells,
            "globals": self._t["globals"][i],
            "targets": self._t["targets"][i],
            "target_mask": self._t["target_mask"][i],
            "conv_label": self._t["conv_label"][i],
            "conv_mask": self._t["conv_mask"][i],
            "maps": self._t["maps"][i],
            "maps_mask": self._t["maps_mask"][i],
        }
        # Optional axial / trajectory keys, present only when the tensors were
        # built with that flag on (so the flag-off item dict is key-for-key the
        # legacy one).
        for opt in ("axial", "axial_mask", "axial_coeff",
                    "traj", "traj_frac", "traj_mask"):
            if opt in self._t:
                item[opt] = self._t[opt][i]
        return item


#: Cap on rows entering the power-prior fit.  The fit has TWO free scalars, so a
#: few thousand mapped rows already pin them; the cap keeps the (single-threaded)
#: shuffle-chain tracing off the critical path of every run.
_POWER_PRIOR_FIT_ROWS = 4000


def _fit_power_prior_for_split(reader: StoreReader, manifest: SplitManifest,
                               fuel: FuelLibrary, split: str, cond_schema: str,
                               *, verbose: bool = True) -> Any | None:
    """Fit the diffusion power-map prior on TRAIN rows only, or ``None``.

    Returns ``None`` for every cond_schema that does not carry the
    ``prior_power`` channel, so the v2..v5 path never pays for this and never
    changes.  The leakage rule is the same one
    :func:`~.physics_prior.fit_cyclen_prior` obeys: only ``manifest.train_ids``
    rows enter, and the two fitted scalars are global (never per-cell), so no
    per-cell label information can reach the model through them.
    """
    from .featurize import CHANNELS_BY_SCHEMA

    if "prior_power" not in CHANNELS_BY_SCHEMA.get(cond_schema, ()):
        return None
    from .power_prior import (_QCOL, _QROW, PowerPrior, fit_power_prior,
                              kinf_quarter_batch)

    df = reader.records
    train_ids = set(manifest.record_ids("train"))
    df = df[df["record_id"].astype(str).isin(train_ids)]
    if "converged" in df.columns:
        df = df[trustworthy(df)]     # quarantined maps must not fit the prior
    df = df[df["maps_key"].notna()]
    if len(df) > _POWER_PRIOR_FIT_ROWS:
        df = df.sample(n=_POWER_PRIOR_FIT_ROWS, random_state=0)
    maps, keep = [], []
    for pos, (_, row) in enumerate(df.iterrows()):
        arr = reader.maps(str(row["maps_key"]))
        if arr is None:
            continue
        plane = np.asarray(arr, dtype=np.float64)[_BOC_POWER_MAP_IDX]
        maps.append(plane[_QROW, _QCOL])
        keep.append(pos)
    if not maps:
        if verbose:
            print("=== power prior: no mapped train rows; using defaults ===",
                  flush=True)
        return PowerPrior(split=split)
    sub = df.iloc[keep]
    # cond_v6b fits on the REGIME burn state, so the two scalars are fit on the
    # same k-inf field the encoder will serve on.  Fitting on the flat-22.0 field
    # and then serving on the regime one would put the arm on inputs it never
    # trained under -- the same class of mismatch ab_score's docstring warns about.
    from .featurize import schema_uses_regime_burnup
    kinf = kinf_quarter_batch(
        sub, fuel, regime_burnup=schema_uses_regime_burnup(cond_schema))
    prior = fit_power_prior(kinf, np.stack(maps),
                            np.asarray(cyclen_cell_codes(sub)), split=split)
    if verbose:
        print(f"=== power prior: M2={prior.m2_cm2:.0f} cm^2 extrap={prior.extrap} "
              f"n_fit={prior.n_fit} within_cell_rho={prior.within_cell_rho:.4f} "
              f"(train rows only) ===", flush=True)
    return prior


def build_precomputed(
    reader: StoreReader,
    manifest: SplitManifest,
    fuel: FuelLibrary,
    *,
    fold: str,
    augment: bool,
    encoder: FeatureEncoder,
    seed: int = 0,
    subset_rows: int | None = None,
    censor_dataset_a_pin_labels: bool = True,
    promote_max_asm_bu: bool = False,
    promote_fxy: bool = False,
    include_axial: bool = False,
    include_traj: bool = False,
    traj_anchors: Sequence[float] = TRAJ_ANCHORS,
) -> PrecomputedDataset:
    """Featurize a split fold once into stacked tensors (base + optional transpose)."""
    base = LPDataset(reader, manifest, fuel, augment=False, fold=fold,
                     encoder=encoder, seed=seed,
                     censor_dataset_a_pin_labels=censor_dataset_a_pin_labels,
                     promote_max_asm_bu=promote_max_asm_bu,
                     promote_fxy=promote_fxy,
                     include_axial=include_axial,
                     include_traj=include_traj, traj_anchors=traj_anchors)
    if subset_rows is not None and subset_rows < len(base):
        base.df = base.df.iloc[:subset_rows].reset_index(drop=True)
        base.record_ids = base.df["record_id"].astype(str).tolist()
    n = len(base)
    # width follows the encoder's schema-selected inventory (26 for v2/v3, more
    # for v4), NOT the module-level base CHANNELS.
    cells = torch.empty((n, len(encoder.channels), 19, 19), dtype=torch.float32)
    cells_t = torch.empty_like(cells) if augment else None
    keys = ("globals", "targets", "target_mask", "conv_label", "conv_mask",
            "maps", "maps_mask")
    if include_axial:
        keys = keys + ("axial", "axial_mask")
    if include_traj:
        keys = keys + ("traj", "traj_frac", "traj_mask")
    buffers: dict[str, list] = {k: [] for k in keys}
    for i in range(n):
        item = base[i]
        cells[i] = item["cells"]
        for k in keys:
            buffers[k].append(item[k])
        if augment:
            row = base.df.iloc[i]
            tc, _tg = encoder.augment_transpose(None, None, row, fuel)
            cells_t[i] = torch.from_numpy(np.ascontiguousarray(tc))
    tensors: dict[str, torch.Tensor] = {"cells": cells}
    if augment:
        tensors["cells_t"] = cells_t
    tensors["globals"] = torch.stack(buffers["globals"])
    tensors["targets"] = torch.stack(buffers["targets"])
    tensors["target_mask"] = torch.stack(buffers["target_mask"])
    tensors["conv_label"] = torch.stack(buffers["conv_label"])
    tensors["conv_mask"] = torch.stack(buffers["conv_mask"])
    tensors["maps"] = torch.stack(buffers["maps"])
    tensors["maps_mask"] = torch.stack(buffers["maps_mask"])
    if include_axial:
        # Raw PROFILES here, not coefficients: the shape basis is fit on the
        # TRAIN fold only and does not exist yet when the val fold is built.
        # ``attach_axial_coeffs`` projects both folds once the basis exists.
        tensors["axial"] = torch.stack(buffers["axial"])
        tensors["axial_mask"] = torch.stack(buffers["axial_mask"])
    if include_traj:
        # Absolute EDIT5 planes (z-scored per step with the MAP constants at loss
        # time — see ``_Norm.z_traj``), the ACHIEVED burnup fractions the model is
        # conditioned on, and the per-anchor presence mask.
        tensors["traj"] = torch.stack(buffers["traj"])
        tensors["traj_frac"] = torch.stack(buffers["traj_frac"])
        tensors["traj_mask"] = torch.stack(buffers["traj_mask"])
    # per-row CBC label-convention provenance code (arm A2).  Always attached —
    # it is an int64 column of the frame, costs 8 bytes/row, and is read ONLY
    # when ``cbc_provenance_offset`` is on, exactly like ``e_core`` below.
    tensors["cbc_prov"] = torch.as_tensor(
        cbc_provenance_codes(base.df), dtype=torch.long)
    # per-row (feed, e_core-bin, dataset) cell code for the within-cell cyclen
    # rank loss — aligned to the base rows (transpose leaves target/cell intact).
    tensors["cyclen_cell"] = torch.as_tensor(
        cyclen_cell_codes(base.df), dtype=torch.long)
    # per-row (case_pair, feed) GATE cell code for the within-cell f_xy rank
    # loss (prereg E.8-②/③).  Attached unconditionally for the same reason
    # ``cyclen_cell`` is — an int64 column costs 8 bytes/row and is read ONLY
    # when ``fxy_rank_weight > 0`` — and it is a DIFFERENT partition from
    # ``cyclen_cell`` on purpose (the two do not nest; see ``fxy_cell_codes``).
    tensors["fxy_cell"] = torch.as_tensor(
        fxy_cell_codes(base.df), dtype=torch.long)
    # per-row e_core (for arm1 cyclen distill boost); NaN -> -1 (matches no band)
    _ec = base.df["e_core"].to_numpy(dtype="float64") if "e_core" in base.df.columns \
        else np.full(len(base.df), np.nan)
    tensors["e_core"] = torch.as_tensor(np.nan_to_num(_ec, nan=-1.0), dtype=torch.float32)
    return PrecomputedDataset(tensors, base.record_ids, base.df,
                              augment=augment, seed=seed)


def attach_cyclen_prior(ds: PrecomputedDataset, prior: Any,
                        fuel: FuelLibrary) -> np.ndarray:
    """Compute + attach the per-row cyclen physics prior to a built dataset.

    Kept separate from :func:`build_precomputed` because the prior's two scalars
    are FIT on the train fold, which only exists once that fold has been built:
    build train -> fit prior on ``train.df`` -> attach to train AND val with the
    identical (train-fitted) parameters.  Returns the prior array in dataset row
    order.  The prior depends only on ``(pattern, library)`` + the static fuel
    table + the a-priori residence age, so attaching it to the val fold leaks
    nothing.
    """
    vals = prior.for_rows(ds.df, fuel)
    ds._t["cyclen_prior"] = torch.as_tensor(vals, dtype=torch.float32)
    return np.asarray(vals, dtype=float)


# --------------------------------------------------------------------------- #
# axial shape basis (decision D10) — fit on TRAIN rows, attached to both folds
# --------------------------------------------------------------------------- #
def fit_axial_basis_for_dataset(ds: PrecomputedDataset, *, rank: int,
                                anchors: Sequence[str] | None = None):
    """Fit the axial PCA shape basis from a built TRAIN dataset's own labels.

    Same leakage contract as the cyclen physics prior and the diffusion power
    prior: the basis is a LABEL-derived artifact, so it may only see the fold the
    network trains on.  Callers pass the train fold; :func:`attach_axial_coeffs`
    then projects both folds with the frozen basis.

    Returns ``None`` when the dataset carries no axial tensor or no labelled row
    at all — a graceful "this arm has no axial supervision", not a crash.
    """
    from ..data.axial import ANCHORS as _AX_ANCHORS
    from ..data.axial import fit_axial_basis

    if "axial" not in ds._t:
        return None
    prof = ds._t["axial"].numpy().astype(np.float64)
    mask = ds._t["axial_mask"].numpy() > 0.5
    if not mask.any():
        return None
    return fit_axial_basis(np.nan_to_num(prof, nan=1.0), rank=int(rank),
                           anchors=tuple(anchors or _AX_ANCHORS), mask=mask)


def attach_axial_coeffs(ds: PrecomputedDataset, basis) -> int:
    """Project a fold's axial profiles onto the (train-fitted) basis.

    Writes ``axial_coeff`` ``[N, A, K]`` in the per-mode standardised space the
    head predicts.  Unlabelled rows get zeros and keep their zero ``axial_mask``,
    so they contribute nothing to :func:`axial_loss`.  Returns the number of rows
    carrying at least one anchor label.
    """
    if basis is None or "axial" not in ds._t:
        return 0
    prof = ds._t["axial"].numpy().astype(np.float64)
    mask = ds._t["axial_mask"].numpy()
    z = basis.z_encode(np.nan_to_num(prof, nan=1.0))
    z = np.where(mask[..., None] > 0.5, z, 0.0)
    ds._t["axial_coeff"] = torch.as_tensor(z, dtype=torch.float32)
    return int((mask.max(axis=1) > 0.5).sum())


# --------------------------------------------------------------------------- #
# inference over a dataset
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_dataset(
    members: Sequence[PosValNet],
    dataset: LPDataset,
    device: torch.device,
    *,
    batch_size: int = 512,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Ensemble inference (fp32, no autocast) over a dataset in stored order.

    Returns raw-space arrays keyed by ``mu`` (mean of member μ, per target),
    ``mu_members`` (M,N,5), ``sigma_alea`` (mean aleatoric σ), ``conv_prob``,
    plus the true ``targets`` / ``target_mask`` / ``conv_label`` / ``conv_mask``
    and the ``record_ids``.  All in *raw* (de-normalized) units except conv.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    for m in members:
        m.eval()
    mu_all: list[np.ndarray] = []          # per member list of [N,5] z-space
    logs_all: list[np.ndarray] = []
    conv_all: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    tmask: list[np.ndarray] = []
    clabel: list[np.ndarray] = []
    cmask: list[np.ndarray] = []
    rids: list[str] = []
    for batch in loader:
        cells = batch["cells"].to(device)
        g = batch["globals"].to(device)
        b_mu, b_logs, b_conv = [], [], []
        for m in members:
            out = m(cells, g)
            b_mu.append(out["mu"].float().cpu().numpy())
            b_logs.append(out["log_sigma"].float().cpu().numpy())
            b_conv.append(torch.sigmoid(out["conv_logit"]).float().cpu().numpy())
        mu_all.append(np.stack(b_mu, axis=0))       # [M, B, 5]
        logs_all.append(np.stack(b_logs, axis=0))
        conv_all.append(np.stack(b_conv, axis=0))
        targets.append(batch["targets"].numpy())
        tmask.append(batch["target_mask"].numpy())
        clabel.append(batch["conv_label"].numpy())
        cmask.append(batch["conv_mask"].numpy())
        rids.extend(batch["record_id"])
    mu_z = np.concatenate(mu_all, axis=1)            # [M, N, 5]
    logs = np.concatenate(logs_all, axis=1)
    conv = np.concatenate(conv_all, axis=1)          # [M, N]
    return {
        "mu_z_members": mu_z,
        "log_sigma_members": logs,
        "conv_prob_members": conv,
        "targets": np.concatenate(targets, axis=0),
        "target_mask": np.concatenate(tmask, axis=0),
        "conv_label": np.concatenate(clabel, axis=0),
        "conv_mask": np.concatenate(cmask, axis=0),
        "record_ids": np.asarray(rids, dtype=object),
    }


def denormalize(mu_z: np.ndarray, tmean: np.ndarray, tstd: np.ndarray) -> np.ndarray:
    return mu_z * tstd + tmean


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).correlation
        return float(r) if r is not None else float("nan")
    except Exception:      # pragma: no cover
        ar = pd.Series(a).rank().to_numpy()
        br = pd.Series(b).rank().to_numpy()
        if ar.std() < 1e-12 or br.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(ar, br)[0, 1])


def within_case_spearman(pred: np.ndarray, true: np.ndarray, mask: np.ndarray,
                         case_pairs: np.ndarray, min_case: int
                         ) -> tuple[float, float, int]:
    """Mean±sd within-case Spearman over cases with >= ``min_case`` valid rows."""
    rs: list[float] = []
    for cp in np.unique(case_pairs):
        sel = (case_pairs == cp) & (mask > 0)
        if int(sel.sum()) < min_case:
            continue
        r = _spearman(pred[sel], true[sel])
        if math.isfinite(r):
            rs.append(r)
    if not rs:
        # fall back to a single global Spearman on the valid rows
        sel = mask > 0
        r = _spearman(pred[sel], true[sel])
        return (r, float("nan"), 0)
    return (float(np.mean(rs)), float(np.std(rs)), len(rs))


def composite_metric(pred: dict[str, np.ndarray], df: pd.DataFrame,
                     tmean: np.ndarray, tstd: np.ndarray, min_case: int,
                     cyclen_prior: np.ndarray | None = None,
                     ) -> dict[str, float]:
    """S1-val composite = within-case Spearman(F_r) − z-MAE(cyclen).

    ``cyclen_prior`` (row-aligned to ``pred``) is added back to the de-normalized
    cyclen column under physics-prior residual learning, so the composite is
    always computed against ABSOLUTE cyclen and stays comparable across arms.
    """
    mu_z = pred["mu_z_members"].mean(axis=0)                 # [N,5]
    mean_raw = denormalize(mu_z, tmean, tstd)
    if cyclen_prior is not None:
        cy_i0 = TARGETS.index("cyclen")
        mean_raw = mean_raw.copy()
        mean_raw[:, cy_i0] = mean_raw[:, cy_i0] + np.asarray(cyclen_prior, dtype=float)
    tmask = pred["target_mask"]
    rid_to_cp = dict(zip(df["record_id"].astype(str), df["case_pair"].astype(str)))
    case_pairs = np.asarray([rid_to_cp.get(str(r), "?") for r in pred["record_ids"]])

    fr_i = TARGETS.index("f_r")
    cy_i = TARGETS.index("cyclen")
    fr_sp, fr_sd, n_cases = within_case_spearman(
        mean_raw[:, fr_i], pred["targets"][:, fr_i], tmask[:, fr_i],
        case_pairs, min_case,
    )
    cy_mask = tmask[:, cy_i] > 0
    if cy_mask.any():
        z_mae = float(np.mean(np.abs(
            mean_raw[cy_mask, cy_i] - pred["targets"][cy_mask, cy_i]
        )) / tstd[cy_i])
    else:
        z_mae = float("nan")
    composite = (fr_sp if math.isfinite(fr_sp) else 0.0) - (
        z_mae if math.isfinite(z_mae) else 0.0)
    return {
        "composite": composite,
        "within_case_spearman_f_r": fr_sp,
        "within_case_spearman_f_r_sd": fr_sd,
        "n_cases": float(n_cases),
        "z_mae_cyclen": z_mae,
    }


def _band_cell_spearman(pred_vals: np.ndarray, true_vals: np.ndarray,
                        mask: np.ndarray, cells: np.ndarray,
                        band: float, min_case: int
                        ) -> tuple[float, int, int]:
    """``(mean rho, n_rows, n_cells)`` over each cell's low-``band`` quantile.

    Per cell: keep the labelled rows whose MEASURED value is ``<= quantile(band)``
    of that cell's labelled values, Spearman them, and average the cells
    UNWEIGHTED.  Cells left with fewer than ``min_case`` banded rows are dropped,
    NOT pooled — pooling elite rows across cells measures the between-cell level
    spread, not the within-cell ordering (prereg E.2.2).  There is deliberately
    no global fallback: with no qualifying cell the answer is NaN (unknown), and
    a NaN selection term falls back to the plain composite.
    """
    rs: list[float] = []
    n_rows = 0
    for c in np.unique(cells):
        sel = (cells == c) & (mask > 0)
        if not sel.any():
            continue
        vals = true_vals[sel]
        keep = sel.copy()
        keep[sel] = vals <= float(np.quantile(vals, band))
        n_keep = int(keep.sum())
        if n_keep < min_case:
            continue
        r = _spearman(pred_vals[keep], true_vals[keep])
        if math.isfinite(r):
            rs.append(r)
            n_rows += n_keep
    if not rs:
        return float("nan"), 0, 0
    return float(np.mean(rs)), n_rows, len(rs)


def fxy_metrics(pred: dict[str, np.ndarray], df: pd.DataFrame,
                tmean: np.ndarray, tstd: np.ndarray,
                target_names: Sequence[str], min_case: int,
                band: float = 1.0) -> dict[str, float]:
    """Val-fold ``f_xy`` metrics on the LABELLED SUBSET only (``{}`` when off).

    Deliberately NOT folded into :func:`composite_metric`: the composite selects
    the best epoch for EVERY run, and an axis labelled on ~2% of rows must not be
    able to move that selection in a general-purpose retrain (a noisy 20-row
    Spearman would otherwise pick the checkpoint).

    ``fxy_select`` is the same shape as ``composite`` (``within-cell Spearman
    minus z-space MAE``) so a run that explicitly opts in with
    ``TrainConfig.fxy_select_weight > 0`` can ADD it to the composite instead of
    replacing it — the legacy axes keep their veto, and the f_xy head stops being
    selected by a criterion that cannot see it.  It is NaN whenever the val fold
    carries no labelled row, and the weighted selection then falls back to the
    plain composite.

    ``within_cell_spearman_f_xy`` reuses :func:`within_case_spearman` on the same
    ``case_pair`` grouping the F_r metric uses, so "within cell" means exactly
    what it already means everywhere else in this file.  Labelled rows are sparse,
    so it usually falls back to that helper's global-Spearman branch — the
    ``n_fxy_cells`` count says which happened (0 == the global fallback).

    ``band`` (prereg E.8-⑥, ``--fxy-select-band``) < 1.0 ADDS the elite-band
    ranking axis the promotion clause G6a is actually scored on: within each
    GATE cell ``(case_pair, feed)``, only the rows whose MEASURED ``f_xy`` is at
    or below that cell's ``band`` quantile are ranked, and the per-cell Spearmans
    are averaged UNWEIGHTED (never row-pooled — a pooled elite statistic is a
    between-cell level artifact, prereg E.2.2).  ``fxy_select`` then selects on
    that number instead of the whole-cell one.  Two things move together here on
    purpose: the band, and the switch from ``case_pair`` to ``(case_pair, feed)``
    — the legacy metric's grouping is COARSER than every gate's, a registered
    mismatch.  ``band >= 1.0`` (the default) emits the legacy keys only, computed
    exactly as before.
    """
    names = list(target_names)
    if "f_xy" not in names:
        return {}
    k = names.index("f_xy")
    mask = pred["target_mask"][:, k]
    true = pred["targets"][:, k]
    sel = mask > 0
    n = int(sel.sum())
    banded = 0.0 < float(band) < 1.0
    if n == 0:
        out = {"n_fxy_val": 0.0, "mae_f_xy": float("nan"),
               "z_mae_f_xy": float("nan"), "fxy_select": float("nan"),
               "within_cell_spearman_f_xy": float("nan"), "n_fxy_cells": 0.0}
        if banded:
            out.update({"within_cell_spearman_f_xy_band": float("nan"),
                        "n_fxy_band_rows": 0.0, "n_fxy_band_cells": 0.0})
        return out
    mean_raw = denormalize(pred["mu_z_members"].mean(axis=0), tmean, tstd)
    rid_to_cp = dict(zip(df["record_id"].astype(str), df["case_pair"].astype(str)))
    case_pairs = np.asarray([rid_to_cp.get(str(r), "?") for r in pred["record_ids"]])
    sp, _sd, n_cells = within_case_spearman(
        mean_raw[:, k], true, mask, case_pairs, min_case)
    mae = float(np.mean(np.abs(mean_raw[sel, k] - true[sel])))
    scale = float(tstd[k])
    z_mae = mae / scale if math.isfinite(scale) and scale > 0 else float("nan")
    rank_sp = sp
    out = {
        "n_fxy_val": float(n),
        "mae_f_xy": mae,
        "z_mae_f_xy": z_mae,
        "fxy_select": float("nan"),
        "within_cell_spearman_f_xy": sp,
        "n_fxy_cells": float(n_cells),
    }
    if banded:
        # GATE cell, not case_pair: the band exists to make selection read the
        # same partition the bar is measured on.
        rid_to_gate = {str(r): fxy_cell_key(cp, fd) for r, cp, fd
                       in zip(df["record_id"].astype(str), df["case_pair"],
                              df["feed"])}
        gate_cells = np.asarray(
            [rid_to_gate.get(str(r), "") for r in pred["record_ids"]])
        rank_sp, band_rows, band_cells = _band_cell_spearman(
            mean_raw[:, k], true, np.where(gate_cells == "", 0.0, mask),
            gate_cells, float(band), min_case)
        out.update({"within_cell_spearman_f_xy_band": rank_sp,
                    "n_fxy_band_rows": float(band_rows),
                    "n_fxy_band_cells": float(band_cells)})
    out["fxy_select"] = (rank_sp - z_mae
                         if math.isfinite(rank_sp) and math.isfinite(z_mae)
                         else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# checkpoint I/O (state_dict + meta.json — no pickled classes)
# --------------------------------------------------------------------------- #
def _vendor_manifest_hash() -> str | None:
    if not _VENDOR_MANIFEST.is_file():
        return None
    return hashlib.sha256(_VENDOR_MANIFEST.read_bytes()).hexdigest()


def save_member(out_dir: str | Path, model: PosValNet, meta: dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tmp = out / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out / "meta.json")
    return out


def load_member(member_dir: str | Path, device: str | torch.device = "cpu"
                ) -> tuple[PosValNet, dict[str, Any]]:
    """Rebuild a member from ``meta.json`` + ``model.pt`` (plan sec. 4.7)."""
    d = Path(member_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    cfg = PosValNetConfig(**meta["net_config"])
    model = PosValNet(cfg)
    state = torch.load(d / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, meta


def norm_from_meta(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    z = meta["target_zscore"]
    return np.asarray(z["mean"], dtype=np.float64), np.asarray(z["std"], dtype=np.float64)


# --------------------------------------------------------------------------- #
# freeze-and-finetune: champion init (--init-from) + trunk/cyclen freeze
# --------------------------------------------------------------------------- #
#: Shared-trunk + convergence modules frozen WHOLE under ``--freeze-trunk-cyclen``
#: (``requires_grad=False`` on every param, and excluded from the optimizer so
#: decoupled weight decay can never move them).  The map head and the non-cyclen
#: rows of the output heads stay trainable.
_FREEZE_WHOLE_MODULES = ("stem", "blocks", "films", "head_trunk", "conv_head")


def _load_champion_member_states(champion_dir: str | Path,
                                 device: str | torch.device = "cpu"
                                 ) -> list[dict[str, torch.Tensor]]:
    """Load every ``member_*/model.pt`` state_dict from a champion checkpoint.

    Ordered by the champion's ``ensemble.json`` member list when present (else a
    lexical ``member_*`` glob, which is the same seed-ascending order).  The
    ``--init-from`` path indexes this list by ensemble position, reusing member 0
    when the champion carries fewer members than the run being trained.
    """
    d = Path(champion_dir)
    order: list[Path] = []
    ens = d / "ensemble.json"
    if ens.is_file():
        names = json.loads(ens.read_text(encoding="utf-8")).get("members", [])
        order = [d / n for n in names if (d / n / "model.pt").is_file()]
    if not order:
        order = sorted(p.parent for p in d.glob("member_*/model.pt"))
    if not order:
        raise FileNotFoundError(
            f"--init-from {str(champion_dir)!r} has no member_*/model.pt checkpoints")
    return [torch.load(md / "model.pt", map_location=device, weights_only=True)
            for md in order]


#: Head tensors whose dim 0 is the TARGET axis, i.e. the ones a ``promote_*``
#: flag makes taller.  ``conv_head`` / ``map_head`` / ``quantile_head`` are not
#: here: their widths are set by other knobs and a mismatch there is a genuine
#: recipe error that must stay loud.
_TARGET_ROW_TENSORS = ("mu_head.weight", "mu_head.bias",
                       "log_sigma_head.weight", "log_sigma_head.bias")


def _graft_appended_target_rows(state: dict[str, torch.Tensor],
                                model_state: dict[str, torch.Tensor],
                                ) -> dict[str, torch.Tensor]:
    """Pad a champion's head rows into a head that grew by an APPENDED target.

    The AL recipe for a new target is ``--init-from <champion>
    --freeze-trunk-cyclen --promote-fxy``: a new head row on a frozen trunk.  But
    the champion's ``mu_head`` is one row shorter than the model being trained, so
    the strict ``load_state_dict`` that guards every other mismatch would reject
    it.  Because :func:`~.dataset_torch.targets_for` only ever APPENDS, the
    champion's rows are a PREFIX of the new head's: copy them in and keep this
    run's freshly-initialized rows for the appended target(s).

    Exactly the same shape of surgery the axial head already does for a champion
    that predates it — and equally narrow: only a pure row-append is grafted
    (same trailing shape, strictly fewer rows).  Anything else is left untouched
    so the strict load still fails loudly.
    """
    out = dict(state)
    for key in _TARGET_ROW_TENSORS:
        src, dst = state.get(key), model_state.get(key)
        if src is None or dst is None or src.shape == dst.shape:
            continue
        if (src.ndim != dst.ndim or src.shape[0] >= dst.shape[0]
                or src.shape[1:] != dst.shape[1:]):
            continue
        grafted = dst.clone()
        grafted[:src.shape[0]] = src.to(dtype=dst.dtype, device=dst.device)
        out[key] = grafted
    return out


def _register_row_zero_hook(param: torch.Tensor, rows: Sequence[int]) -> Any:
    """Register a backward hook zeroing ``param.grad`` on the given dim-0 rows.

    Keeps the CYCLEN rows of the (otherwise trainable) mu / log_sigma / quantile
    heads at zero gradient, so the optimizer never updates them and cyclen stays
    byte-identical to the champion.
    """
    idx = torch.as_tensor(list(rows), dtype=torch.long, device=param.device)

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[idx] = 0.0
        return grad

    return param.register_hook(_hook)


def _cyclen_quantile_rows(model: PosValNet, q_names: Sequence[str]) -> list[int]:
    """Row indices of the CYCLEN block in ``quantile_head`` (``[c*nq:(c+1)*nq]``).

    ``q_names`` is the RESOLVED quantile-target order the head was built with
    (``cfg.quantile_targets`` filtered to the present targets); empty when there
    is no quantile head or cyclen is not among its targets.
    """
    q_names = list(q_names)
    if not getattr(model, "has_quantiles", False) or "cyclen" not in q_names:
        return []
    c = q_names.index("cyclen")
    nq = int(model.n_quantiles)
    return list(range(c * nq, (c + 1) * nq))


def _cyclen_masked_mode(cfg: TrainConfig) -> bool:
    """Whether this run masks the cyclen rows (freeze mode OR trunk fine-tune).

    Both modes promise the SAME cyclen contract — gradient-masked cyclen rows,
    champion physics prior and per-cell cyclen calibration copied verbatim — and
    both therefore require ``init_from``.  They differ only in whether the trunk
    is frozen (see :func:`_apply_freeze_trunk_cyclen`).
    """
    return bool(cfg.freeze_trunk_cyclen) or float(cfg.trunk_finetune_lr_mult) > 0.0


def _apply_freeze_trunk_cyclen(model: PosValNet, q_names: Sequence[str],
                               *, freeze_trunk: bool = True) -> list[Any]:
    """Freeze trunk + cyclen output so cyclen stays byte-identical to the champion.

    * ``requires_grad=False`` on every param of ``stem`` / ``blocks`` / ``films`` /
      ``head_trunk`` / ``conv_head`` (the whole shared trunk + convergence head).
      ``freeze_trunk=False`` (the ``--trunk-finetune-lr-mult`` mode) SKIPS this
      loop only — every cyclen protection below is applied identically.
    * ``mu_head`` / ``log_sigma_head`` stay trainable, but the CYCLEN weight row
      (``_CYCLEN_IDX``) and bias element get a backward hook zeroing their gradient.
    * ``quantile_head`` (when present) gets the same treatment on its CYCLEN block.
    * ``map_head`` and the non-cyclen (f_r, ...) rows stay fully trainable.

    Returns the hook handles (held on the member for the run).  The row-masked
    heads must ALSO sit in a ``weight_decay=0`` optimizer group (see
    :func:`_build_member_optim`): the zero-grad hook stops the Adam update but not
    AdamW's decoupled weight-decay term, which would otherwise shrink the frozen
    cyclen rows.
    """
    if freeze_trunk:
        for name in _FREEZE_WHOLE_MODULES:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad_(False)
    handles: list[Any] = []
    for head_name in ("mu_head", "log_sigma_head"):
        head = getattr(model, head_name)
        handles.append(_register_row_zero_hook(head.weight, [_CYCLEN_IDX]))
        handles.append(_register_row_zero_hook(head.bias, [_CYCLEN_IDX]))
    q_rows = _cyclen_quantile_rows(model, q_names)
    if q_rows:
        handles.append(_register_row_zero_hook(model.quantile_head.weight, q_rows))
        handles.append(_register_row_zero_hook(model.quantile_head.bias, q_rows))
    return handles


def _masked_head_param_ids(model: PosValNet) -> set[int]:
    """id()s of the row-masked head params (mu / log_sigma / quantile)."""
    ids: set[int] = set()
    for hn in ("mu_head", "log_sigma_head", "quantile_head"):
        h = getattr(model, hn, None)
        if h is not None:
            ids.update(id(p) for p in h.parameters())
    return ids


def _trunk_param_ids(model: PosValNet) -> set[int]:
    """id()s of the shared-trunk params (the modules freeze mode freezes whole)."""
    ids: set[int] = set()
    for name in _FREEZE_WHOLE_MODULES:
        mod = getattr(model, name, None)
        if mod is not None:
            ids.update(id(p) for p in mod.parameters())
    return ids


def _build_member_optim(model: PosValNet, *, lr: float, weight_decay: float,
                        freeze: bool, trunk_lr_mult: float = 0.0
                        ) -> torch.optim.Optimizer:
    """AdamW over a member's params (freeze-aware).

    Legacy (``freeze`` off): one group over ``model.parameters()`` with
    ``weight_decay`` — byte-identical to the pre-flag construction.

    ``freeze`` on: only ``requires_grad`` params enter the optimizer (the frozen
    trunk / conv modules are excluded so weight decay can never move them), and
    the row-masked heads go in a ``weight_decay=0`` group so the zero-grad CYCLEN
    rows also see no decay.  (The non-cyclen f_r rows of those heads therefore
    train without weight decay — a negligible, intended trade-off for cyclen
    byte-identity, since per-tensor WD cannot spare only some rows.)

    ``trunk_lr_mult > 0`` (the ``--trunk-finetune-lr-mult`` mode, which sets
    ``freeze`` too): the trunk is NOT frozen, so its params enter a THIRD group
    at ``lr * trunk_lr_mult`` with the normal weight decay.  Group 0 stays the
    head/decay group, so the logged ``metrics["lr"]`` keeps meaning the head LR.
    """
    if not freeze:
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    masked = _masked_head_param_ids(model)
    trunk = _trunk_param_ids(model) if float(trunk_lr_mult) > 0.0 else set()
    decay = [p for p in model.parameters()
             if p.requires_grad and id(p) not in masked and id(p) not in trunk]
    no_decay = [p for p in model.parameters()
                if p.requires_grad and id(p) in masked]
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if trunk:
        trunk_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) in trunk]
        groups.append({"params": trunk_params, "weight_decay": weight_decay,
                       "lr": lr * float(trunk_lr_mult)})
    return torch.optim.AdamW(groups, lr=lr)


class _PerGroupCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Cosine schedule whose floor scales with each group's OWN base LR.

    ``CosineAnnealingLR`` anneals every param group toward one shared
    ``eta_min``.  With the champion schedule (3e-4 -> 3e-5) and a trunk group at
    ``lr * 0.05`` = 1.5e-5, that shared floor is ABOVE the trunk's starting LR,
    so the trunk LR would RISE to 2x its registered multiple over the run.  Each
    group's floor is instead ``eta_min * base_lr / base_lrs[0]``, which keeps the
    trunk at exactly ``mult x`` the head LR at every epoch.  Used ONLY when
    ``trunk_finetune_lr_mult > 0``; every other run builds the stock scheduler.
    """

    def get_lr(self) -> list[float]:            # noqa: D102 - see class docstring
        ref = self.base_lrs[0] if self.base_lrs else 0.0
        cos = 1.0 + math.cos(math.pi * self.last_epoch / self.T_max)
        out = []
        for base in self.base_lrs:
            eta = self.eta_min * (base / ref) if ref > 0 else self.eta_min
            out.append(eta + (base - eta) * cos / 2.0)
        return out


# --------------------------------------------------------------------------- #
# split loading
# --------------------------------------------------------------------------- #
def _load_split(reader: StoreReader, split: str, splits_dir: str | Path,
                seed: int) -> SplitManifest:
    p = Path(splits_dir) / f"{split}.json"
    if p.is_file():
        return SplitManifest.from_json(p)
    manifests = make_splits(reader.records, seed=seed, out_dir=splits_dir, persist=True)
    if split not in manifests:
        raise KeyError(f"unknown split {split!r}; have {sorted(manifests)}")
    return manifests[split]


# --------------------------------------------------------------------------- #
# member training
# --------------------------------------------------------------------------- #
def _resolve_schedule(cfg: TrainConfig, device: torch.device
                      ) -> tuple[int, float, float, int, dict[str, Any]]:
    """Pick the effective batch and linearly-scaled LR / warmup for ``device``.

    CPU keeps ``batch_size`` (256); CUDA defaults to ``batch_size_cuda`` (1024)
    unless the user pinned ``--batch-size``.  LR scales linearly with the batch
    ratio (Goyal 2017) and the μ-only warmup is rescaled so the number of warmup
    *steps* is preserved across batch sizes (fewer steps/epoch at large batch ->
    more warmup epochs).  All three defaults collapse to the legacy values when
    the effective batch equals ``base_batch`` (the CPU path), so CPU is untouched.
    """
    base = max(1, int(cfg.base_batch))
    if cfg.batch_size_explicit:
        eff = int(cfg.batch_size)
    elif device.type == "cuda":
        eff = int(cfg.batch_size_cuda)
    else:
        eff = int(cfg.batch_size)
    ratio = eff / base
    lr = cfg.lr * ratio if cfg.lr_scaling else cfg.lr
    lr_final = cfg.lr_final * ratio if cfg.lr_scaling else cfg.lr_final
    if cfg.warmup_step_scaling and ratio != 1.0:
        warm = int(round(cfg.warmup_epochs * ratio))
        warm = max(1, min(warm, max(1, cfg.epochs - 1)))
    else:
        warm = int(cfg.warmup_epochs)
    meta = {
        "base_batch": base,
        "effective_batch": eff,
        "batch_scale": round(ratio, 6),
        "lr": lr,
        "lr_final": lr_final,
        "lr_scaling": bool(cfg.lr_scaling),
        "warmup_epochs_base": int(cfg.warmup_epochs),
        "warmup_epochs_effective": int(warm),
        "warmup_step_scaling": bool(cfg.warmup_step_scaling),
    }
    return eff, lr, lr_final, warm, meta


def _resident_fits(train_ds: PrecomputedDataset, val_ds: PrecomputedDataset,
                   device: torch.device, cfg: TrainConfig) -> bool:
    """Whether the precomputed tensors should live on-device (CUDA only).

    Returns ``False`` on CPU, when disabled, or when the estimated footprint
    (base + transposed cells + globals/targets/maps for train *and* val, plus a
    gather margin) exceeds ``max_resident_gib`` or half of free VRAM.
    """
    if device.type != "cuda" or not cfg.device_resident:
        return False
    need = 0
    for ds in (train_ds, val_ds):
        for t in ds._t.values():
            need += t.numel() * t.element_size()
    need = int(need * 1.15)                       # gather/headroom margin
    if need > cfg.max_resident_gib * (1024 ** 3):
        return False
    try:
        free, _total = torch.cuda.mem_get_info(device)
        if need > 0.5 * free:
            return False
    except Exception:                             # pragma: no cover - no CUDA here
        pass
    return True


class _MemberState:
    """Mutable per-member state carried through the joint training loop."""


def _gather_train_batch(host_t: dict[str, torch.Tensor],
                        dev_t: dict[str, torch.Tensor] | None,
                        sel: torch.Tensor, uset: torch.Tensor | None,
                        augment: bool, device: torch.device, resident: bool
                        ) -> dict[str, torch.Tensor]:
    """Index a training batch out of the precomputed tensors (index_select).

    Preserves the exact ``PrecomputedDataset.__getitem__`` semantics: the diagonal
    transpose selects ``cells_t`` per-row (globals/targets/maps/masks always come
    from the base rows — the transposed variant only differs in ``cells``).  When
    ``resident`` the source already lives on ``device`` (zero H2D per step); else
    the assembled batch is copied to ``device`` once.
    """
    src = dev_t if resident else host_t
    cells = src["cells"].index_select(0, sel)
    if augment and uset is not None:
        cells_t = src["cells_t"].index_select(0, sel)
        cells = torch.where(uset.view(-1, 1, 1, 1), cells_t, cells)
    batch = {
        "cells": cells,
        "globals": src["globals"].index_select(0, sel),
        "targets": src["targets"].index_select(0, sel),
        "target_mask": src["target_mask"].index_select(0, sel),
        "conv_label": src["conv_label"].index_select(0, sel),
        "conv_mask": src["conv_mask"].index_select(0, sel),
        "maps": src["maps"].index_select(0, sel),
        "maps_mask": src["maps_mask"].index_select(0, sel),
    }
    for opt in ("cyclen_cell", "fxy_cell", "cyclen_prior", "e_core",  # optional (absent in unit stubs)
                "distill_soft", "distill_mask", "axial_coeff", "axial_mask",
                "traj", "traj_frac", "traj_mask", "cbc_prov"):
        if opt in src:
            batch[opt] = src[opt].index_select(0, sel)
    if not resident:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return batch


def _step_member(m: _MemberState, batch: dict[str, torch.Tensor],
                 cfg: TrainConfig, use_amp: bool, device: torch.device) -> None:
    """One optimizer step for a member (loss identical to the legacy loop)."""
    m.optim.zero_grad(set_to_none=True)
    # Physics-prior residual learning: present only when the flag is on AND the
    # dataset carries the prior tensor, so the legacy path is untouched.
    prior = batch.get("cyclen_prior") if getattr(m, "use_prior", False) else None
    cy = m.norm.cyclen_idx
    # Trajectory supervision is active only when the head exists AND the fold
    # carries the labels, so a flag-off run calls ``m.fwd`` with exactly the two
    # legacy positional arguments (the byte-identity contract).
    traj_on = bool(getattr(m, "use_traj", False)) and "traj" in batch
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=use_amp):
        if traj_on:
            out = m.fwd(batch["cells"], batch["globals"], batch["traj_frac"])
        else:
            out = m.fwd(batch["cells"], batch["globals"])
        z_t = m.norm.z_targets(batch["targets"], prior)
        z_m = m.norm.z_maps(batch["maps"])
        # arm A2: map the model's (MASTER-native) cbc prediction into the row's
        # OWN label convention before the residual is taken.  Applied to a CLONE
        # and to the cbc column only, so every other term below — the rank
        # hinges, the distillation pull, the quantile pinball, the map terms —
        # still sees the unshifted ``out["mu"]``.  ``mu_reg is out["mu"]`` when
        # the flag is off, which is the byte-identical legacy call.
        mu_reg = out["mu"]
        cbc_off = getattr(m, "cbc_offset_param", None)
        if cbc_off is not None and "cbc_prov" in batch:
            cbc_i = m.cbc_idx
            # index 0 == the reference (MASTER-native) group: a structural zero,
            # never a learned parameter, so serving cannot drift off-convention.
            #
            # The parameter lives in Z UNITS, not ppm.  That is not cosmetic:
            # Adam's per-step displacement is bounded by ~lr, so a ppm-scale
            # parameter (the gap is 100-410 ppm) could move at most ~9 ppm over a
            # whole 150-epoch run at the scaled lr — the arm would silently do
            # nothing.  In z units the same gap is 0.26-1.06 (cbc tstd ~386 ppm)
            # and is reached in a few hundred steps.  ``_finalize_member``
            # multiplies by tstd to report the offsets in ppm.
            off_z = torch.cat([cbc_off.new_zeros(1), cbc_off])
            delta = off_z[batch["cbc_prov"]]
            mu_reg = out["mu"].clone()
            mu_reg[:, cbc_i] = mu_reg[:, cbc_i] + delta.to(mu_reg.dtype)
        loss_reg = regression_loss(
            mu_reg, out["log_sigma"], z_t, batch["target_mask"],
            use_nll=m.use_nll, beta=cfg.beta_nll, delta=cfg.huber_delta)
        loss_map = map_loss(out["map"], z_m, batch["maps_mask"], cfg.huber_delta,
                            peak_weight=cfg.map_peak_weight,
                            peak_topk=cfg.map_peak_topk,
                            peak_topk_weight=cfg.map_peak_topk_weight)
        loss_conv = convergence_loss(out["conv_logit"], batch["conv_label"],
                                     batch["conv_mask"])
        loss = loss_reg + cfg.map_lambda * loss_map + cfg.conv_weight * loss_conv
        if "traj" in out and traj_on and cfg.traj_weight > 0.0:
            # Auxiliary map-head supervision at intermediate burnup fractions.
            # Same decoder, same z-space as the EOC planes it extends.
            loss = loss + cfg.traj_weight * traj_loss(
                out["traj"], m.norm.z_traj(batch["traj"]), batch["traj_mask"],
                cfg.huber_delta)
        if cfg.map_spectral_weight > 0.0:
            # Band-weighted FFT term: targets the measured high-wavenumber
            # attenuation directly, where the Huber term's gradient is swamped by
            # the low-wavenumber radial tilt that holds ~74% of the map power.
            loss = loss + cfg.map_spectral_weight * spectral_map_loss(
                out["map"], z_m, batch["maps_mask"])
        if cfg.cyclen_rank_weight > 0.0 and "cyclen_cell" in batch:
            # The hinge must rank ABSOLUTE cyclen.  Under residual learning
            # ``mu`` is the z-residual, so the prior (in z units) is added back
            # before ranking; the constant tmean cancels in the pairwise
            # difference, so only the prior/tstd term is needed.  With no prior
            # this is exactly the legacy call.
            mu_cy = out["mu"][:, cy]
            if prior is not None:
                mu_cy = mu_cy + prior / m.norm.tstd[cy]
            loss_rank = cyclen_rank_loss(
                mu_cy, batch["targets"][:, cy],
                batch["target_mask"][:, cy], batch["cyclen_cell"],
                margin=cfg.cyclen_rank_margin_z,
                min_gap_efpd=cfg.cyclen_rank_min_gap_efpd)
            loss = loss + cfg.cyclen_rank_weight * loss_rank
        if cfg.f_r_rank_weight > 0.0 and "cyclen_cell" in batch:
            # F_r is NOT residual-learned, so ``mu[:, _FR_IDX]`` is the z-scored F_r
            # prediction directly; the same (feed, e_core-bin, dataset) cell code the
            # cyclen rank uses defines the within-cell F_r pairs.
            loss_fr_rank = f_r_rank_loss(
                out["mu"][:, _FR_IDX], batch["targets"][:, _FR_IDX],
                batch["target_mask"][:, _FR_IDX], batch["cyclen_cell"],
                margin=cfg.f_r_rank_margin_z, min_gap=cfg.f_r_rank_min_gap,
                low_thresh=cfg.f_r_rank_low_thresh, low_weight=cfg.f_r_rank_low_weight)
            loss = loss + cfg.f_r_rank_weight * loss_fr_rank
        fxy_i = int(getattr(m, "fxy_idx", -1))
        rank_cell = "fxy_cell" if cfg.fxy_rank_cell == "gate" else "cyclen_cell"
        if cfg.fxy_rank_weight > 0.0 and fxy_i >= 0 and rank_cell in batch:
            # arm 5 (prereg E.3): rank the COMPOSED f_xy row within its GATE
            # cell.  ``out["mu"][:, fxy_i]`` already carries the a*mu[f_r]+b
            # prior (net._compose_fxy), so — unlike the cyclen hinge above —
            # nothing is added back here; and mu[f_r] is detached inside that
            # composition, so this term cannot reach the F_r head.
            stats: dict[str, float] = {}
            loss_fxy_rank = f_xy_rank_loss(
                out["mu"][:, fxy_i], batch["targets"][:, fxy_i],
                batch["target_mask"][:, fxy_i], batch[rank_cell],
                margin=cfg.fxy_rank_margin_z, min_gap=cfg.fxy_rank_min_gap,
                low_thresh=cfg.fxy_rank_low_thresh,
                low_weight=cfg.fxy_rank_low_weight, stats=stats)
            loss = loss + cfg.fxy_rank_weight * loss_fxy_rank
            # A hinge that sees no pair contributes exactly 0 and is
            # indistinguishable from a satisfied one in the loss value; the
            # census is the only instrument that catches it (prereg E.8-⑦).
            m.fxy_rank_pairs = getattr(m, "fxy_rank_pairs", 0.0) + stats["n_pairs"]
            m.fxy_rank_cells = getattr(m, "fxy_rank_cells", 0.0) + stats["n_cells"]
            m.fxy_rank_batches = getattr(m, "fxy_rank_batches", 0) + 1
        if cfg.map_fr_consistency_weight > 0.0:
            map_present = (batch["maps_mask"][:, _BOC_POWER_MAP_IDX]
                           .flatten(1).amax(dim=1) > 0)
            loss_cons = map_fr_consistency_loss(
                out["map"], out["mu"][:, _FR_IDX],
                batch["target_mask"][:, _FR_IDX], map_present, _BOC_POWER_MAP_IDX)
            loss = loss + cfg.map_fr_consistency_weight * loss_cons
        if "distill_soft" in batch and cfg.distill_weight > 0.0:
            # z-scored Huber pull toward the per-cell teacher's mean, on the
            # SAME normalization the hard loss uses (so under residual learning
            # the teacher's cyclen — which was already made absolute — is
            # converted to the residual by the same z_targets call).
            z_soft = m.norm.z_targets(batch["distill_soft"], prior)
            per_kd = F.smooth_l1_loss(out["mu"], z_soft, beta=cfg.huber_delta,
                                      reduction="none")
            # arm1(b): boost the cyclen-column distill loss on the gate-failing
            # e_core bands, pinning the champion teacher's cyclen ranking there.
            if (cfg.distill_cyclen_boost_factor > 1.0 and "e_core" in batch
                    and cfg.distill_cyclen_boost_bands):
                bands = _parse_ecore_bands(cfg.distill_cyclen_boost_bands)
                if bands:
                    ec = batch["e_core"]
                    inband = torch.zeros_like(ec, dtype=torch.bool)
                    for lo, hi in bands:
                        inband = inband | ((ec >= lo) & (ec < hi))
                    factor = torch.where(
                        inband, ec.new_full((), float(cfg.distill_cyclen_boost_factor)),
                        ec.new_ones(()))
                    per_kd = per_kd.clone()
                    per_kd[:, cy] = per_kd[:, cy] * factor
            loss_kd = _masked_mean(per_kd, batch["distill_mask"])
            loss = loss + cfg.distill_weight * loss_kd
        if "axial" in out and "axial_coeff" in batch and cfg.axial_weight > 0.0:
            # Present only when the axial head is built AND the fold carries
            # projected labels, so a flag-off run never reaches this branch.
            loss = loss + cfg.axial_weight * axial_loss(
                out["axial"], batch["axial_coeff"], batch["axial_mask"],
                cfg.huber_delta)
        if "quantiles" in out and m.q_idx is not None:
            # Pinball loss on the SAME z-targets the mean head sees (so under
            # residual learning the quantiles are residual quantiles too, and
            # serving adds the prior back to all of them identically).
            loss_q = pinball_loss(
                out["quantiles"], z_t[:, m.q_idx],
                batch["target_mask"][:, m.q_idx], cfg.quantile_levels)
            loss = loss + cfg.quantile_weight * loss_q
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.model.parameters(), 5.0)
    m.optim.step()
    m.running += float(loss.detach())
    m.n_batches += 1


@torch.no_grad()
def _predict_member_resident(model: PosValNet, val_dev: dict[str, torch.Tensor],
                             val_ds: PrecomputedDataset, device: torch.device,
                             batch_size: int = 512) -> dict[str, np.ndarray]:
    """Device-resident single-member val inference (fp32, no autocast).

    Byte-for-byte equivalent to ``predict_dataset([model], val_ds, device)`` in
    stored order — same fp32 forward, same batching-invariant per-row outputs —
    but the inputs are already on-device so there is no per-step H2D copy.
    """
    model.eval()
    n = val_dev["cells"].shape[0]
    mu: list[np.ndarray] = []
    logs: list[np.ndarray] = []
    conv: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sel = torch.arange(start, end, device=device)
        out = model(val_dev["cells"].index_select(0, sel),
                    val_dev["globals"].index_select(0, sel))
        mu.append(out["mu"].float().cpu().numpy())
        logs.append(out["log_sigma"].float().cpu().numpy())
        conv.append(torch.sigmoid(out["conv_logit"]).float().cpu().numpy())
    host = val_ds._t
    return {
        "mu_z_members": np.stack([np.concatenate(mu, axis=0)], axis=0),
        "log_sigma_members": np.stack([np.concatenate(logs, axis=0)], axis=0),
        "conv_prob_members": np.stack([np.concatenate(conv, axis=0)], axis=0),
        "targets": host["targets"].numpy(),
        "target_mask": host["target_mask"].numpy(),
        "conv_label": host["conv_label"].numpy(),
        "conv_mask": host["conv_mask"].numpy(),
        "record_ids": np.asarray(val_ds.record_ids, dtype=object),
    }


def _train_members(
    seeds: Sequence[int],
    *,
    train_ds: PrecomputedDataset,
    val_ds: PrecomputedDataset,
    cfg: TrainConfig,
    device: str | torch.device,
    globals_names: Sequence[str],
    reader: StoreReader,
    eff_batch: int,
    lr: float,
    lr_final: float,
    warm: int,
    resident: bool,
    compile_flag: bool,
    n_channels: int = len(CHANNELS),
    channel_names: Sequence[str] = CHANNELS,
    verbose: bool = True,
    log_every: int = 1,
    manifest: SplitManifest | None = None,
    target_names: Sequence[str] = TARGETS,
    init_states: Sequence[dict[str, torch.Tensor] | None] | None = None,
) -> list[_MemberState]:
    """Train ``len(seeds)`` members jointly in one process over shared tensors.

    Each member owns an independent model init (``manual_seed(seed)``), an
    independent bootstrap permutation (its own ``torch.Generator`` feeding a
    ``WeightedRandomSampler``-equivalent ``multinomial``), and an independent
    transpose-augmentation stream (its own ``numpy`` ``default_rng``).  No global
    RNG is consumed inside the loop and members share no mutable state, so a
    member's trajectory is bit-identical whether trained alone (``len(seeds)==1``)
    or jointly, and is independent of the order members are stepped — this is what
    the ``parallel-members==1 == sequential`` exactness test verifies.  Members
    that trip early-stop drop out of the live set; survivors keep training.
    """
    # freeze-and-finetune guard (checked before any dataset access): freezing
    # random cyclen rows would silently ship a garbage cyclen head.
    if _cyclen_masked_mode(cfg) and init_states is None:
        raise ValueError(
            "freeze_trunk_cyclen / trunk_finetune_lr_mult requires init_from "
            "(champion weights to freeze); refusing to freeze randomly-"
            "initialized cyclen rows")
    device = torch.device(device)
    n_train = len(train_ds)
    train_df = train_ds.df
    val_df = val_ds.df
    augment = bool(cfg.augment) and ("cells_t" in train_ds._t)

    # curriculum-cell rows may carry a higher sampling-weight cap, threaded from
    # the retrain split manifest (groups: 'cells' + 'curriculum_cell_cap').
    curr_campaigns = manifest.groups.get("cells") if manifest is not None else None
    curr_cap = manifest.groups.get("curriculum_cell_cap") if manifest is not None else None
    weights, weight_summary = compute_cell_weights(
        train_df, cap=cfg.cell_weight_cap,
        curriculum_campaigns=curr_campaigns, curriculum_cap=curr_cap)
    weights_t = torch.as_tensor(weights, dtype=torch.double)   # CPU (multinomial)

    train_dev = val_dev = None
    if resident:
        train_dev = {k: v.to(device) for k, v in train_ds._t.items()}
        val_dev = {k: v.to(device) for k, v in val_ds._t.items()}

    # --- v5 knobs, all inert at their defaults ------------------------------ #
    target_names = tuple(target_names)
    cyclen_idx = target_names.index("cyclen")
    # Physics prior is active only when the flag is on AND the datasets actually
    # carry the attached prior tensor.
    use_prior = bool(cfg.cyclen_physics_prior) and "cyclen_prior" in train_ds._t
    train_prior = (train_ds._t["cyclen_prior"].numpy().astype(float)
                   if use_prior else None)
    val_prior = (val_ds._t["cyclen_prior"].numpy().astype(float)
                 if (use_prior and "cyclen_prior" in val_ds._t) else None)
    # Quantile heads: resolve the target names to column indices ONCE.
    q_names = tuple(n for n in cfg.quantile_targets if n in target_names)
    q_on = bool(cfg.quantile_heads) and bool(q_names) and bool(cfg.quantile_levels)
    q_cols = [target_names.index(n) for n in q_names] if q_on else []
    q_idx_t = (torch.as_tensor(q_cols, dtype=torch.long, device=device)
               if q_on else None)
    # The z-score constants: under residual learning cyclen's are the RESIDUAL's.
    norm_df = (residual_target_frame(train_df, train_prior)
               if use_prior else train_df)
    fxy_idx, fxy_ref_idx, fxy_prior, fxy_prior_z_ab = resolve_fxy_prior(
        train_df, norm_df, target_names, verbose=verbose,
        prior_residual=bool(cfg.fxy_prior_residual))
    fxy_select_w = float(cfg.fxy_select_weight)
    if fxy_idx >= 0 and verbose:
        if fxy_select_w <= 0.0:
            print("=== f_xy head: best-epoch selection does NOT see f_xy "
                  "(fxy_select_weight=0) — set --fxy-select-weight for a "
                  "head-focused run ===", flush=True)
        if warm >= cfg.epochs // 2:
            print(f"WARNING: f_xy head with warmup_epochs_effective={warm} of "
                  f"{cfg.epochs} epochs.  log_sigma receives NO gradient during "
                  "the mu-only warmup, so any checkpoint selected before epoch "
                  f"{warm} serves the INITIAL sigma (the 20260829 arm-1 failure: "
                  "best epochs 4-37, warm 80, 68% coverage 0.99).  Lower "
                  "--warmup-epochs for a head-only finetune.", flush=True)
        if cfg.fxy_rank_weight > 0.0:
            print(f"=== f_xy rank hinge: w={cfg.fxy_rank_weight} "
                  f"cell={cfg.fxy_rank_cell} margin={cfg.fxy_rank_margin_z}z "
                  f"min_gap={cfg.fxy_rank_min_gap} "
                  f"low<={cfg.fxy_rank_low_thresh} x{cfg.fxy_rank_low_weight} "
                  f"(watch rkPairs: a 0 there means the term is inert) ===",
                  flush=True)
        if 0.0 < cfg.fxy_select_band < 1.0:
            print(f"=== f_xy selection band: q={cfg.fxy_select_band} on GATE "
                  f"cells (case_pair, feed) ===", flush=True)

    # The z-score constants are identical for every member (same norm frame,
    # same target order); resolved once so the pre-training f_xy prior refit can
    # de-normalize the F_r row before the per-member block recomputes them.
    tmean0, tstd0 = compute_target_norm(norm_df, target_names)
    refit_fxy = bool(cfg.fxy_prior_on_predicted) and fxy_idx >= 0 and fxy_ref_idx >= 0
    if refit_fxy and init_states is None:
        raise ValueError(
            "fxy_prior_on_predicted needs --init-from: fitting the f_xy prior "
            "against a RANDOMLY initialized F_r head would fit noise.")

    freeze = bool(cfg.freeze_trunk_cyclen)   # guarded above (needs init_states)
    trunk_mult = float(cfg.trunk_finetune_lr_mult)
    if trunk_mult > 0.0 and freeze:
        raise ValueError(
            "trunk_finetune_lr_mult and freeze_trunk_cyclen are mutually "
            "exclusive: the first exists to keep the cyclen half of the second "
            "WITHOUT freezing the trunk")
    # cyclen protection (row-masked gradients + no weight decay on those heads)
    # is applied in BOTH modes; only the trunk freeze differs.
    cyclen_mask = freeze or trunk_mult > 0.0
    # hires A2: resolve the power-prior input channel ONCE.  ``-1`` (the default,
    # and the only possibility for a cond_schema without the channel) builds a
    # net whose module set is byte-identical to the pre-hires network.
    map_prior_channel = -1
    if bool(cfg.map_prior_residual):
        if "prior_power" not in channel_names:
            raise ValueError(
                "map_prior_residual requires a cond_schema carrying the "
                "'prior_power' channel (v6 or v6_prior); "
                f"cond_schema={cfg.cond_schema!r} has none")
        map_prior_channel = channel_names.index("prior_power")
    # Axial head: the width comes from the FITTED basis (which the caller has
    # already fit on the train fold and attached), never from the config alone —
    # so a run whose train fold turned out to carry no axial label builds the
    # legacy net instead of an unsupervised head.  0/0 == no module registered.
    axial_basis = getattr(train_ds, "axial_basis", None)
    n_ax_anchors = axial_basis.n_anchors if axial_basis is not None else 0
    n_ax_modes = axial_basis.n_modes if axial_basis is not None else 0
    # arm A1: the trajectory head's width comes from the ATTACHED labels, never
    # from the config alone — a run whose train fold turned out to carry no
    # trajectory builds the legacy net instead of an unsupervised head, exactly
    # as the axial head does.  0/0 == no module registered.
    use_traj = bool(cfg.traj_weight > 0.0) and "traj" in train_ds._t
    if use_traj and float(train_ds._t["traj_mask"].sum()) <= 0.0:
        use_traj = False
        if verbose:
            print("=== traj head: no labelled train rows; head DISABLED ===",
                  flush=True)
    n_traj_anchors = int(train_ds._t["traj"].shape[1]) if use_traj else 0
    n_traj_planes = int(train_ds._t["traj"].shape[2]) if use_traj else 0
    if use_traj and verbose:
        n_rows = int((train_ds._t["traj_mask"].amax(dim=1) > 0).sum())
        print(f"=== traj head: {n_traj_anchors} burnup anchors x "
              f"{n_traj_planes} planes {TRAJ_STEP_PLANES[:n_traj_planes]} -> map "
              f"channels {TRAJ_MAP_CHANNELS[:n_traj_planes]}, labels "
              f"{n_rows}/{len(train_ds)} train rows (weight {cfg.traj_weight}) ===",
              flush=True)
    # arm A2: one learned scalar per NON-reference CBC provenance group.
    n_cbc_prov = len(CBC_PROVENANCE_GROUPS) if cfg.cbc_provenance_offset else 0
    cbc_idx = target_names.index("cbc_max") if "cbc_max" in target_names else -1
    if n_cbc_prov and cbc_idx < 0:
        raise ValueError("cbc_provenance_offset needs a 'cbc_max' target column")
    if n_cbc_prov and verbose:
        codes = train_ds._t.get("cbc_prov")
        counts = ({g: int((codes == i).sum()) for i, g in
                   enumerate(CBC_PROVENANCE_GROUPS)} if codes is not None else {})
        print(f"=== cbc provenance offsets: groups {CBC_PROVENANCE_GROUPS} "
              f"(reference={CBC_PROVENANCE_GROUPS[0]}, offset pinned 0) "
              f"train rows {counts} ===", flush=True)
    members: list[_MemberState] = []
    for i, seed in enumerate(seeds):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        model = PosValNet(PosValNetConfig(
            in_channels=n_channels, n_globals=len(globals_names),
            width=cfg.width, n_blocks=cfg.n_blocks, head_hidden=cfg.head_hidden,
            n_targets=len(target_names),
            n_quantile_targets=len(q_cols) if q_on else 0,
            n_quantiles=len(cfg.quantile_levels) if q_on else 0,
            map_head_mode=cfg.map_head_mode,
            map_prior_channel=map_prior_channel,
            n_axial_anchors=n_ax_anchors,
            n_axial_modes=n_ax_modes,
            n_traj_anchors=n_traj_anchors,
            n_traj_planes=n_traj_planes,
            n_cbc_provenance_groups=n_cbc_prov,
            fxy_target_idx=fxy_idx,
            fxy_ref_idx=fxy_ref_idx,
            fxy_prior_a=fxy_prior_z_ab[0],
            fxy_prior_b=fxy_prior_z_ab[1])).to(device)
        # --- freeze-and-finetune: initialize from the champion member --------- #
        # The per-member RNG streams (sampler_gen / rng_aug, seeded below) are
        # independent of the global RNG the init above consumes, so loading the
        # champion weights here does NOT perturb sampling/augmentation — the
        # training trajectory's stochastics match a from-scratch run's.
        if init_states is not None and init_states[i] is not None:
            # strict load: a net-config mismatch (width / targets / quantile
            # widths) is a hard, loud failure, never a silent partial init.
            state = init_states[i]
            if model.has_axial and not any(k.startswith("axial_head.")
                                           for k in state):
                # The champion predates the axial head, and the standard AL
                # recipe is exactly `--init-from <champion> --freeze-trunk-cyclen`
                # (a NEW head on a frozen trunk).  Seed the champion's tensors and
                # keep this run's freshly-initialized axial rows — still strict on
                # every OTHER key, so a genuine width/target mismatch stays loud.
                state = {**state,
                         **{k: v for k, v in model.state_dict().items()
                            if k.startswith("axial_head.")}}
                if verbose:
                    print("=== axial head: champion has none; initializing it "
                          "fresh on the loaded trunk ===", flush=True)
            if fxy_idx >= 0:
                # The champion predates the promoted target, so its mu /
                # log_sigma heads are one row short.  Graft its rows in (they
                # are a strict prefix) and keep this run's fresh f_xy row —
                # still strict on every other key.
                grafted = _graft_appended_target_rows(state, model.state_dict())
                if grafted is not state and verbose and i == 0:
                    print("=== f_xy head: champion has none; grafting its "
                          f"{len(state.get('mu_head.bias', ()))}-target head "
                          "rows and initializing the f_xy row fresh ===",
                          flush=True)
                state = grafted
            model.load_state_dict(state, strict=True)
        member_prior, member_ab = fxy_prior, fxy_prior_z_ab
        if refit_fxy and init_states is not None and init_states[i] is not None:
            got = refit_fxy_prior_on_predicted(
                model, train_ds._t, fxy_idx=fxy_idx, ref_idx=fxy_ref_idx,
                tmean=tmean0, tstd=tstd0, device=device)
            if got is not None:
                member_prior, member_ab = got
                # The composition scalars live on the module AND on its config
                # (which ``_finalize_member`` reads back for ``net_config``), so
                # both must move or the checkpoint's recorded prior would not be
                # the one its weights were trained against.
                model.fxy_prior_a, model.fxy_prior_b = member_ab
                model.config.fxy_prior_a, model.config.fxy_prior_b = member_ab
                if verbose:
                    print(f"  [seed {seed}] f_xy prior refit on PREDICTED f_r: "
                          f"{member_prior.a:.4f}*f_r {member_prior.b:+.4f} "
                          f"(n={member_prior.n_fit}, r={member_prior.pearson:.4f}, "
                          f"resid sd={member_prior.resid_sd:.4f}); measured-f_r "
                          f"fit was {fxy_prior.a:.4f}/{fxy_prior.b:+.4f}",
                          flush=True)
            elif verbose:
                print(f"  [seed {seed}] f_xy prior refit on predicted f_r "
                      "DEGENERATE; keeping the measured-f_r fit", flush=True)
        freeze_handles = (_apply_freeze_trunk_cyclen(model, q_names,
                                                     freeze_trunk=freeze)
                          if cyclen_mask else [])
        fwd = torch.compile(model) if compile_flag else model
        optim = _build_member_optim(model, lr=lr,
                                    weight_decay=cfg.weight_decay,
                                    freeze=cyclen_mask, trunk_lr_mult=trunk_mult)
        sched_cls = (_PerGroupCosineAnnealingLR if trunk_mult > 0.0
                     else torch.optim.lr_scheduler.CosineAnnealingLR)
        sched = sched_cls(optim, T_max=cfg.epochs, eta_min=lr_final)
        if trunk_mult > 0.0 and verbose and i == 0:
            print(f"=== trunk fine-tune: lr x {trunk_mult:g} "
                  f"(trunk group lr={lr * trunk_mult:.3g}, head lr={lr:.3g}); "
                  f"trainable {count_parameters(model):,} of "
                  f"{sum(p.numel() for p in model.parameters()):,} params; "
                  "cyclen rows still gradient-masked ===", flush=True)
        tmean, tstd = compute_target_norm(norm_df, target_names)
        mmean, mstd = compute_map_norm(reader, train_df, cfg.map_norm_subset, seed=seed)
        m = _MemberState()
        m.seed = int(seed)
        m.model = model
        m.fwd = fwd
        m.optim = optim
        m.sched = sched
        m.freeze_handles = freeze_handles     # kept alive for the run (freeze mode)
        m.norm = _Norm(tmean, tstd, mmean, mstd, device, cyclen_idx)
        m.use_prior = use_prior
        m.q_idx = q_idx_t
        m.q_names = q_names if q_on else ()
        m.axial_basis = axial_basis
        m.use_traj = bool(use_traj and model.has_traj)
        m.traj_anchors = (tuple(cfg.traj_anchors[:n_traj_anchors])
                          if m.use_traj else ())
        m.fxy_idx = fxy_idx
        m.fxy_ref_idx = fxy_ref_idx
        m.fxy_prior = member_prior
        m.fxy_prior_z = member_ab
        m.cbc_idx = cbc_idx
        m.cbc_offset_param = (model.cbc_provenance_offset
                              if model.has_cbc_provenance else None)
        m.tmean, m.tstd, m.mmean, m.mstd = tmean, tstd, mmean, mstd
        m.sampler_gen = torch.Generator().manual_seed(int(seed))
        m.rng_aug = np.random.default_rng(seed)
        m.best = {"composite": -1e18, "select_score": -1e18, "epoch": -1,
                  "state": None, "metrics": {}}
        m.since_best = 0
        m.live = True
        m.history = []
        m.weight_summary = weight_summary
        members.append(m)

    use_amp = device.type == "cuda"
    for epoch in range(cfg.epochs):
        live = [m for m in members if m.live]
        if not live:
            break
        # per-member epoch permutation + transpose draws (independent streams).
        for m in live:
            idx = torch.multinomial(weights_t, n_train, True, generator=m.sampler_gen)
            m.epoch_idx = idx.to(device) if resident else idx
            if augment:
                uset = torch.from_numpy(m.rng_aug.random(n_train) < 0.5)
                m.epoch_uset = uset.to(device) if resident else uset
            else:
                m.epoch_uset = None
            m.model.train()
            m.use_nll = epoch >= warm
            m.running = 0.0
            m.n_batches = 0
            if cfg.fxy_rank_weight > 0.0:
                # Only the arm-5 path carries this bookkeeping, so a flag-off
                # member's state is untouched down to its attribute set.
                m.fxy_rank_pairs = 0.0
                m.fxy_rank_cells = 0.0
                m.fxy_rank_batches = 0
        for start in range(0, n_train, eff_batch):
            end = min(start + eff_batch, n_train)
            for m in live:
                sel = m.epoch_idx[start:end]
                uset = None if m.epoch_uset is None else m.epoch_uset[start:end]
                batch = _gather_train_batch(train_ds._t, train_dev, sel, uset,
                                            augment, device, resident)
                _step_member(m, batch, cfg, use_amp, device)
        for m in live:
            m.sched.step()
            if resident:
                val_pred = _predict_member_resident(m.model, val_dev, val_ds, device)
            else:
                val_pred = predict_dataset([m.model], val_ds, device, num_workers=0)
            metrics = composite_metric(val_pred, val_df, m.tmean, m.tstd,
                                       cfg.min_case_val, val_prior)
            metrics.update(fxy_metrics(val_pred, val_df, m.tmean, m.tstd,
                                       target_names, cfg.min_case_val,
                                       band=cfg.fxy_select_band))
            metrics["train_loss"] = m.running / max(1, m.n_batches)
            if getattr(m, "fxy_rank_batches", 0):
                # Mean EFFECTIVE pairs / contributing cells per batch (E.8-⑦).
                nb = float(m.fxy_rank_batches)
                metrics["fxy_rank_pairs"] = m.fxy_rank_pairs / nb
                metrics["fxy_rank_cells"] = m.fxy_rank_cells / nb
            metrics["epoch"] = epoch
            metrics["lr"] = m.optim.param_groups[0]["lr"]
            # Selection score: the composite, PLUS the opted-in f_xy term.  With
            # ``fxy_select_weight == 0`` (the default) this is the composite
            # itself, so every legacy run selects exactly the epoch it always did.
            score = metrics["composite"]
            fxy_sel = metrics.get("fxy_select")
            if fxy_select_w > 0.0 and fxy_sel is not None and math.isfinite(fxy_sel):
                score = score + fxy_select_w * fxy_sel
            metrics["select_score"] = score
            m.history.append(metrics)
            if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
                line = (f"  [seed {m.seed}] epoch {epoch:3d} "
                        f"loss={metrics['train_loss']:.4f} "
                        f"comp={metrics['composite']:.4f} "
                        f"spF_r={metrics['within_case_spearman_f_r']:.3f} "
                        f"zMAEcy={metrics['z_mae_cyclen']:.3f}")
                if "mae_f_xy" in metrics:
                    # Per-epoch f_xy trace: arm 1 shipped an inert residual and a
                    # never-trained log_sigma, and train.log carried no f_xy
                    # number at all to show it while the run was still alive.
                    line += (f" fxyMAE={metrics['mae_f_xy']:.4f}"
                             f" fxyRho={metrics['within_cell_spearman_f_xy']:.3f}"
                             f" n={int(metrics['n_fxy_val'])}")
                    if "within_cell_spearman_f_xy_band" in metrics:
                        line += (f" fxyRhoBand="
                                 f"{metrics['within_cell_spearman_f_xy_band']:.3f}"
                                 f"/{int(metrics['n_fxy_band_cells'])}c")
                    if "fxy_rank_pairs" in metrics:
                        # A silently pair-starved hinge is the one failure this
                        # arm cannot read off the loss; print it every epoch.
                        line += (f" rkPairs={metrics['fxy_rank_pairs']:.0f}"
                                 f" rkCells={metrics['fxy_rank_cells']:.1f}")
                    if fxy_select_w > 0.0:
                        line += f" sel={score:.4f}"
                print(line, flush=True)
            if score > m.best["select_score"]:
                m.best = {
                    "composite": metrics["composite"],
                    "select_score": score,
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in m.model.state_dict().items()},
                    "metrics": metrics,
                }
                m.since_best = 0
            else:
                m.since_best += 1
                if m.since_best >= cfg.patience:
                    m.live = False
                    if verbose:
                        print(f"  [seed {m.seed}] early stop at epoch {epoch} "
                              f"(best {m.best['epoch']})", flush=True)

    for m in members:
        if m.best["state"] is not None:
            m.model.load_state_dict(m.best["state"])
        m.model.eval()
    return members


def _last_history_value(m: Any, key: str) -> float:
    """The last epoch's value of ``key`` in a member's history (NaN if absent).

    Used for the meta census of terms that are per-epoch training statistics
    rather than checkpoint state; NaN means "the term never ran".
    """
    for h in reversed(list(getattr(m, "history", []) or [])):
        if key in h:
            return float(h[key])
    return float("nan")


def _finalize_member(
    out_dir: str | Path,
    m: _MemberState,
    *,
    cfg: TrainConfig,
    split: str,
    globals_names: Sequence[str],
    encoder: FeatureEncoder,
    train_ds: PrecomputedDataset,
    val_ds: PrecomputedDataset,
    device: str | torch.device,
    sched_meta: dict[str, Any],
    resident: bool,
    target_names: Sequence[str] = TARGETS,
    cyclen_prior: Any | None = None,
    val_prior_values: np.ndarray | None = None,
) -> Path:
    """Write a member's ``state_dict`` + ``meta.json`` checkpoint (+ round-trip)."""
    device = torch.device(device)
    meta = {
        "cond_schema": encoder.cond_schema,
        "cond_norm": encoder.cond_norm,
        "channels": list(encoder.channels),
        "globals": list(globals_names),
        "target_names": list(target_names),
        "map_keys": list(_MAP_KEYS),
        "target_zscore": {"mean": m.tmean.tolist(), "std": m.tstd.tolist()},
        "map_zscore": {"mean": m.mmean.tolist(), "std": m.mstd.tolist()},
        # Read the config off the BUILT model so the meta can never drift from
        # the weights (quantile widths / target count included).
        "net_config": dict(m.model.config.__dict__),
        # ``n_params`` counts TRAINABLE parameters only, so a --freeze-trunk-cyclen
        # run reports the head size (e.g. 6,298), not the model size.  The total
        # is stamped alongside it so the two can never be confused again.
        "n_params": count_parameters(m.model),
        "n_params_total": int(sum(p.numel() for p in m.model.parameters())),
        # hires bundle: the serve-side contract for the cond_v6 power prior.  A
        # v6/v6_prior model MUST rebuild its FeatureEncoder with these scalars.
        "power_prior": (encoder.power_prior.to_dict()
                        if getattr(encoder, "power_prior", None) is not None
                        else {"enabled": False}),
        # --- v5 serve-side contract (all null/false on the legacy path) ------
        # ``cyclen_physics_prior`` is what tells the serving backend to ADD the
        # prior back to the residual head, so predict() returns absolute cyclen.
        "cyclen_physics_prior": (
            {"enabled": True, **cyclen_prior.to_dict()}
            if (getattr(m, "use_prior", False) and cyclen_prior is not None)
            else {"enabled": False}
        ),
        "quantile_heads": {
            "enabled": bool(getattr(m, "q_names", ())),
            "targets": list(getattr(m, "q_names", ())),
            "levels": list(cfg.quantile_levels) if getattr(m, "q_names", ()) else [],
        },
        "promote_max_asm_bu": bool(cfg.promote_max_asm_bu),
        "promote_fxy": bool(cfg.promote_fxy),
        "seed": m.seed,
        "split": split,
        "train_config": cfg.to_dict(),
        "best_epoch": m.best["epoch"],
        "best_metrics": m.best["metrics"],
        "history": m.history,
        "cell_weight_summary": {k: m.weight_summary.get(k) for k in
                                ("n_rows", "n_cells", "cap", "cap_hits",
                                 "weight_min", "weight_max", "curriculum_cap",
                                 "n_curriculum_rows", "curriculum_cap_hits")},
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "schedule": {**sched_meta,
                     "device_resident": bool(resident),
                     "parallel_members": int(cfg.parallel_members),
                     "torch_compile": bool(cfg.torch_compile)},
        "versions": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "vendor_manifest_sha256": _vendor_manifest_hash(),
        "device": device.type,
    }
    # Axial head serve-side contract.  Added ONLY when the head exists, so a
    # flag-off meta.json is key-for-key the legacy one; a reader resolves it with
    # ``meta.get("axial_head", {"enabled": False})``, which is also correct for
    # every pre-axial checkpoint.  It carries the FITTED basis because the head
    # emits STANDARDISED coefficients and serving needs the basis to turn them
    # back into a profile (and then into F_z / AO / ASI).
    # Trajectory-supervision record (arm A1).  Added ONLY when the head exists,
    # so a flag-off meta.json is key-for-key the legacy one.  There is no
    # serve-side contract to carry — the head is auxiliary supervision and
    # ``forward`` never emits it unless a caller asks — so this block is pure
    # provenance: what was supervised, where, and how hard.
    if getattr(m, "use_traj", False):
        meta["traj_head"] = {
            "enabled": True,
            "anchors": [float(f) for f in m.traj_anchors],
            "planes": list(TRAJ_STEP_PLANES[:m.model.n_traj_planes]),
            "map_channels": list(TRAJ_MAP_CHANNELS[:m.model.n_traj_planes]),
            "weight": float(cfg.traj_weight),
            "serve_affecting": False,
        }
    # Per-provenance CBC offsets (arm A2).  The fitted offsets are recorded in
    # RAW ppm — they are the measured label-convention gap, the single most
    # useful number this arm produces — together with the explicit statement that
    # the SERVE convention is the reference group's (offset 0).
    if getattr(m, "cbc_offset_param", None) is not None:
        learned_z = [0.0] + [float(v) for v in
                             m.cbc_offset_param.detach().cpu().tolist()]
        cbc_std = float(m.tstd[m.cbc_idx])
        meta["cbc_provenance_offset"] = {
            "enabled": True,
            "groups": list(CBC_PROVENANCE_GROUPS),
            "reference": CBC_PROVENANCE_GROUPS[0],
            "offsets_z": learned_z,
            "offsets_ppm": [v * cbc_std for v in learned_z],
            "cbc_tstd_ppm": cbc_std,
            "serve_convention": CBC_PROVENANCE_GROUPS[0],
            "serve_affecting": False,
        }
    # Trunk fine-tune (prereg Amendment D-3, arm 4).  NOT serve-affecting -- it
    # is provenance the gate reads to tell an arm-4 checkpoint from an arm-3 one:
    # the trunk moved, the cyclen rows did not.  Added ONLY when the mode is on,
    # so a flag-off meta.json is key-for-key the legacy one (the multiplier is
    # also in ``train_config.trunk_finetune_lr_mult``, which records the CLI).
    if float(cfg.trunk_finetune_lr_mult) > 0.0:
        meta["trunk_finetune"] = {
            "enabled": True,
            "trunk_lr_mult": float(cfg.trunk_finetune_lr_mult),
            "trunk_frozen": False,
            "cyclen_masked": True,
            "init_from": str(cfg.init_from) if cfg.init_from else None,
        }
    # F_xy prior-residual head.  SERVE-AFFECTING: ``predict_fxy`` reads
    # ``enabled`` to decide whether this checkpoint can answer at all, and the
    # prior is what the served f_xy is a residual against — so the fitted
    # coefficients and the label count that justified them are recorded here, not
    # only in ``train_config``.  Added ONLY when the head exists, so a flag-off
    # meta.json is key-for-key the legacy one and every pre-F_xy checkpoint
    # resolves correctly with ``meta.get("fxy_head", {"enabled": False})``.
    if getattr(m, "fxy_idx", -1) >= 0 and getattr(m, "fxy_prior", None) is not None:
        a_z, b_z = getattr(m, "fxy_prior_z", (0.0, 0.0))
        meta["fxy_head"] = {
            "enabled": True,
            "target": "f_xy",
            "target_idx": int(m.fxy_idx),
            # "prior_residual" == the composition is live inside the net;
            # "direct" == the row predicts the absolute f_xy and ``prior`` below
            # is the REPORTED baseline only (prior_z is (0, 0) and composes
            # nothing).  ``prior_source`` says which F_r the two scalars were fit
            # against — "measured" (the store label) or "predicted" (the model's
            # own raw ``mu[f_r]`` row, i.e. the row the composition reads).
            "mode": ("prior_residual" if getattr(m, "fxy_ref_idx", -1) >= 0
                     else "direct"),
            "prior_source": ("predicted"
                             if str(getattr(m.fxy_prior, "split", "")).endswith(
                                 "predicted_f_r") else "measured"),
            "prior_ref": "f_r",
            "prior": m.fxy_prior.to_dict(),
            "prior_z": {"a": float(a_z), "b": float(b_z)},
            "n_labelled_train": int(m.fxy_prior.n_fit),
            "min_labels": int(MIN_FXY_LABELS),
            "select_weight": float(cfg.fxy_select_weight),
            "select_band": float(cfg.fxy_select_band),
            # arm 5 (prereg E.8-⑦).  NOT serve-affecting — a ranking term
            # changes the weights, not how they are read — but it is stamped
            # here so a checkpoint can always say whether the hinge was on, on
            # which partition, and whether it actually saw pairs: a term that
            # silently found none trains to exactly the same loss as one that
            # was satisfied, and only this census tells them apart.
            "rank": {
                "enabled": bool(cfg.fxy_rank_weight > 0.0),
                "weight": float(cfg.fxy_rank_weight),
                "cell": str(cfg.fxy_rank_cell),
                "margin_z": float(cfg.fxy_rank_margin_z),
                "min_gap": float(cfg.fxy_rank_min_gap),
                "low_thresh": float(cfg.fxy_rank_low_thresh),
                "low_weight": float(cfg.fxy_rank_low_weight),
                "mean_pairs_per_batch": _last_history_value(m, "fxy_rank_pairs"),
                "mean_cells_per_batch": _last_history_value(m, "fxy_rank_cells"),
            },
            "serve_affecting": True,
        }
    if getattr(m, "axial_basis", None) is not None:
        meta["axial_head"] = {
            "enabled": True,
            "anchors": list(m.axial_basis.anchors),
            "rank": int(m.axial_basis.n_modes),
            "weight": float(cfg.axial_weight),
            "n_fit": int(m.axial_basis.n_fit),
            "basis": m.axial_basis.to_dict(),
        }
    member_dir = save_member(out_dir, m.model, meta)
    # save val-head predictions for the local load round-trip (fp32, first K)
    _save_round_trip(member_dir, m.model, val_ds, device, m.tmean, m.tstd,
                     cfg.round_trip_rows, val_prior_values)
    return member_dir


def attach_distill_targets(ds: PrecomputedDataset, cache_path: str | Path,
                           target_names: Sequence[str],
                           min_match_frac: float = 0.5) -> int:
    """Attach a prebuilt soft-target cache to a dataset (train fold only).

    Returns the number of rows carrying at least one distillable soft target.
    Rows the cache does not cover get a zero mask and train on the hard label
    alone — the graceful-degradation contract of :mod:`.distill`.

    **Fail loud (min_match_frac):** the cache records how many rows the BUILD
    matched (``n_soft``); this is the record_id join onto the ACTUAL training
    store, which may be a different snapshot (the remote is pushed at a point in
    time while the local store keeps growing).  If fewer than ``min_match_frac``
    of the built soft-target rows survive the join, the distillation term would be
    a near-no-op and the arm a silent duplicate of the baseline — so we raise
    instead of proceeding.  A 0-match (the decoy that motivated this) is the
    limiting case.  ``min_match_frac <= 0`` disables the guard.
    """
    from .distill import align_soft_targets, load_soft_targets

    cache = load_soft_targets(cache_path)
    soft, mask = align_soft_targets(cache, ds.record_ids, target_names)
    ds._t["distill_soft"] = torch.as_tensor(soft, dtype=torch.float32)
    ds._t["distill_mask"] = torch.as_tensor(mask, dtype=torch.float32)
    n_attached = int((mask.sum(axis=1) > 0).sum())
    n_built = int(cache.get("n_soft", n_attached))
    if min_match_frac > 0.0:
        if n_built <= 0:
            raise ValueError(
                f"distillation cache {cache_path} carries no soft-target rows; "
                "it was built empty (broken teacher/cell-key join). Rebuild it.")
        frac = n_attached / n_built
        if frac < float(min_match_frac):
            raise ValueError(
                f"distillation join matched {n_attached}/{n_built} built "
                f"soft-target rows ({frac:.1%} < {min_match_frac:.0%}) against "
                f"{len(ds.record_ids)} train rows: the cache's record_ids do not "
                f"align with THIS store snapshot (version drift or a stale cache). "
                f"Rebuild the cache against the training store, do not proceed.")
    return n_attached


def _prepare_axial_basis(cfg: TrainConfig,
                         train_pre: PrecomputedDataset,
                         val_pre: PrecomputedDataset,
                         *, verbose: bool = True):
    """Fit the axial shape basis on TRAIN rows and attach coefficients to both folds.

    Returns the fitted basis, or ``None`` when the flag is off or the train fold
    carries no axial label at all.  ``None`` is the byte-identical legacy path:
    no ``axial_coeff`` tensor is attached, so ``_train_members`` builds a net with
    no axial module and ``_step_member`` never adds the term.

    **Leakage guard (asserted):** the basis is a label-derived artifact, so it may
    only see the fold the network trains on — the same rule
    :func:`_prepare_cyclen_prior` and :func:`_fit_power_prior_for_split` obey.
    The val fold is PROJECTED with the frozen basis, never re-fit.
    """
    if not cfg.axial_head:
        return None
    train_ids = set(train_pre.df["record_id"].astype(str))
    val_ids = set(val_pre.df["record_id"].astype(str))
    assert not (train_ids & val_ids), (
        "axial basis fit frame intersects the val fold; the basis must be fit "
        "on train rows only"
    )
    basis = fit_axial_basis_for_dataset(train_pre, rank=cfg.axial_rank)
    if basis is None:
        if verbose:
            print("=== axial head: no labelled train rows; head DISABLED ===",
                  flush=True)
        return None
    n_tr = attach_axial_coeffs(train_pre, basis)
    n_va = attach_axial_coeffs(val_pre, basis)
    train_pre.axial_basis = basis
    val_pre.axial_basis = basis
    if verbose:
        print(f"=== axial head: rank {basis.n_modes} x {basis.n_anchors} anchors "
              f"{basis.anchors}, labels train {n_tr}/{len(train_pre)} "
              f"val {n_va}/{len(val_pre)} (weight {cfg.axial_weight}) ===",
              flush=True)
    return basis


def _prepare_cyclen_prior(cfg: TrainConfig,
                          train_pre: PrecomputedDataset,
                          val_pre: PrecomputedDataset,
                          fuel: FuelLibrary | None,
                          store_dir: str | Path,
                          split: str | None):
    """Fit the cyclen physics prior on TRAIN rows and attach it to both folds.

    Returns ``(prior, val_prior_values)`` — ``(None, None)`` when the flag is off,
    which is the byte-identical legacy path (no tensor is attached, so
    ``_train_members`` never activates the residual branch).

    **Leakage guard (asserted):** the two fitted scalars see only ``train_pre.df``
    — the same fold the network trains on — and the val fold's prior is EVALUATED
    with those frozen parameters, never re-fit.  The prior itself is a pure
    function of the pattern + the static fuel table + the a-priori residence age,
    so it carries no label information in either fold.
    """
    if not cfg.cyclen_physics_prior:
        return None, None
    from .physics_prior import CyclenPhysicsPrior, PRIOR_NAME, fit_cyclen_prior

    if fuel is None:
        fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
    train_ids = set(train_pre.df["record_id"].astype(str))
    val_ids = set(val_pre.df["record_id"].astype(str))
    assert not (train_ids & val_ids), (
        "cyclen physics-prior fit frame intersects the val fold; the prior must "
        "be fit on train rows only"
    )
    # Freeze-and-finetune (and the trunk fine-tune mode, whose cyclen rows are
    # equally gradient-masked): the cyclen residual head is byte-identical to the
    # champion, and that head was trained against the CHAMPION's prior. Re-fitting
    # alpha/beta on the (now larger) store would shift the served cyclen (=
    # residual + prior) and change within-cell cyclen RANKING, so the honest gate's
    # cyclen worst-drop would NOT be ~=0. LOAD the champion's prior VERBATIM instead
    # of re-fitting it — this is the serve-read value (it is embedded per-member in
    # meta.json). The from-scratch (flag-off) path below still fits, unchanged.
    if _cyclen_masked_mode(cfg) and cfg.init_from:
        src = Path(cfg.init_from) / PRIOR_NAME
        prior = CyclenPhysicsPrior.load(src)
        print(f"=== cyclen physics prior: LOADED from champion {src} "
              f"(frozen cyclen head; not re-fit) alpha={prior.alpha:.4f} "
              f"beta={prior.beta:.2f} ===", flush=True)
    else:
        prior = fit_cyclen_prior(train_pre.df, fuel, split=split)
        print(f"=== cyclen physics prior: alpha={prior.alpha:.4f} beta={prior.beta:.2f} "
              f"n_fit={prior.n_fit} train_pearson={prior.pearson:.4f} ===", flush=True)
    attach_cyclen_prior(train_pre, prior, fuel)
    val_vals = attach_cyclen_prior(val_pre, prior, fuel)
    return prior, val_vals


def train_member(
    seed: int,
    *,
    split: str = "S1",
    device: str | torch.device = "cpu",
    epochs: int | None = None,
    out_dir: str | Path,
    store_dir: str | Path = DEFAULT_STORE,
    splits_dir: str | Path = DEFAULT_SPLITS,
    config: TrainConfig | None = None,
    subset_rows: int | None = None,
    train_pre: PrecomputedDataset | None = None,
    val_pre: PrecomputedDataset | None = None,
    globals_names: Sequence[str] | None = None,
    cond_schema: str = DEFAULT_COND_SCHEMA,
    censor_dataset_a_pin_labels: bool = True,
    log_every: int = 1,
    verbose: bool = True,
) -> Path:
    """Train one ensemble member; write a ``state_dict`` + ``meta.json`` checkpoint.

    ``train_pre`` / ``val_pre`` are optional precomputed datasets (shared across
    an ensemble to featurize once); when absent they are built here.  This is a
    thin wrapper over :func:`_train_members` with a single seed, so a solo run is
    bit-identical to the same seed trained inside a ``parallel-members`` chunk.
    """
    cfg = config or TrainConfig()
    if epochs is not None:
        cfg.epochs = epochs
    device = torch.device(device)

    reader = StoreReader(store_dir)
    # NOTE: the single-member helper does not fit the cond_v6 power prior (it may
    # be handed pre-built tensors and so has no manifest to obey the train-only
    # leakage rule with).  A v6/v6_prior encoder here uses the module DEFAULT
    # (M^2, extrap); ``train_ensemble`` — the path every A/B arm uses — fits them.
    encoder = FeatureEncoder(cond_schema=cond_schema)
    target_names = targets_for(cfg.promote_max_asm_bu, cfg.promote_fxy)
    manifest = None
    fuel = None
    if train_pre is None or val_pre is None:
        fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
        manifest = _load_split(reader, split, splits_dir, seed)
        train_pre = build_precomputed(
            reader, manifest, fuel, fold="train", augment=cfg.augment,
            encoder=encoder, seed=seed, subset_rows=subset_rows,
            censor_dataset_a_pin_labels=censor_dataset_a_pin_labels,
            promote_max_asm_bu=cfg.promote_max_asm_bu,
            promote_fxy=cfg.promote_fxy,
            include_axial=cfg.axial_head,
            include_traj=cfg.traj_weight > 0.0,
            traj_anchors=cfg.traj_anchors)
        val_n = None if subset_rows is None else max(64, subset_rows // 8)
        val_pre = build_precomputed(
            reader, manifest, fuel, fold="val", augment=False, encoder=encoder,
            seed=seed, subset_rows=val_n,
            censor_dataset_a_pin_labels=censor_dataset_a_pin_labels,
            promote_max_asm_bu=cfg.promote_max_asm_bu,
            promote_fxy=cfg.promote_fxy,
            include_axial=cfg.axial_head,
            include_traj=cfg.traj_weight > 0.0,
            traj_anchors=cfg.traj_anchors)
    if globals_names is None:
        globals_names = encoder.globals_names
    _prepare_axial_basis(cfg, train_pre, val_pre, verbose=verbose)
    prior, val_prior_vals = _prepare_cyclen_prior(
        cfg, train_pre, val_pre, fuel, store_dir, split)
    if cfg.distill_targets:
        # TRAIN fold only — a soft target on a holdout row would leak a teacher's
        # (label-trained) opinion into the honest eval fold.  Raises when the
        # record_id join matches too few rows (see attach_distill_targets).
        n_kd = attach_distill_targets(train_pre, cfg.distill_targets, target_names,
                                      min_match_frac=cfg.distill_min_match_frac)
        print(f"=== distillation: soft targets on {n_kd}/{len(train_pre)} train "
              f"rows (weight {cfg.distill_weight}) ===", flush=True)

    eff_batch, lr, lr_final, warm, sched_meta = _resolve_schedule(cfg, device)
    resident = _resident_fits(train_pre, val_pre, device, cfg)
    # freeze-and-finetune: a single member initializes from champion member 0.
    if _cyclen_masked_mode(cfg) and not cfg.init_from:
        raise ValueError(
            "freeze_trunk_cyclen / trunk_finetune_lr_mult requires init_from")
    init_states = ([_load_champion_member_states(cfg.init_from)[0]]
                   if cfg.init_from else None)
    members = _train_members(
        [seed], train_ds=train_pre, val_ds=val_pre, cfg=cfg, device=device,
        globals_names=globals_names, reader=reader, eff_batch=eff_batch,
        lr=lr, lr_final=lr_final, warm=warm, resident=resident,
        compile_flag=cfg.torch_compile, n_channels=len(encoder.channels),
        channel_names=tuple(encoder.channels),
        verbose=verbose, log_every=log_every, manifest=manifest,
        target_names=target_names, init_states=init_states)
    return _finalize_member(
        out_dir, members[0], cfg=cfg, split=split, globals_names=globals_names,
        encoder=encoder, train_ds=train_pre, val_ds=val_pre, device=device,
        sched_meta=sched_meta, resident=resident, target_names=target_names,
        cyclen_prior=prior, val_prior_values=val_prior_vals)


@torch.no_grad()
def _save_round_trip(member_dir: Path, model: PosValNet, val_ds: PrecomputedDataset,
                     device: torch.device, tmean, tstd, k: int,
                     cyclen_prior: np.ndarray | None = None) -> None:
    n = min(k, len(val_ds))
    if n == 0:
        return
    sub_tensors = {key: val_ds._t[key][:n] for key in val_ds._t}
    sub = PrecomputedDataset(sub_tensors, val_ds.record_ids[:n],
                             val_ds.df.iloc[:n].reset_index(drop=True), augment=False)
    pred = predict_dataset([model], sub, device, num_workers=0)
    mu_z = pred["mu_z_members"][0]                      # [n,5]
    mean_raw = denormalize(mu_z, tmean, tstd)
    if cyclen_prior is not None:
        # store the ABSOLUTE cyclen so the round-trip file means the same thing
        # for every arm (residual + prior == what predict() will serve).
        cy = TARGETS.index("cyclen")
        mean_raw = mean_raw.copy()
        mean_raw[:, cy] = mean_raw[:, cy] + np.asarray(cyclen_prior[:n], dtype=float)
    np.savez(
        member_dir / "val_pred.npz",
        record_ids=pred["record_ids"].astype(str),
        mu_z=mu_z.astype(np.float32),
        mean_raw=mean_raw.astype(np.float32),
        log_sigma=pred["log_sigma_members"][0].astype(np.float32),
        conv_prob=pred["conv_prob_members"][0].astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #
def train_ensemble(
    n: int = 5,
    *,
    split: str = "S1",
    device: str | torch.device = "cpu",
    epochs: int | None = None,
    out_dir: str | Path,
    base_seed: int = 20260716,
    parallel_members: int | None = None,
    cond_schema: str = DEFAULT_COND_SCHEMA,
    censor_dataset_a_pin_labels: bool = True,
    **kwargs: Any,
) -> list[Path]:
    """Train an ``n``-member deep ensemble into ``out_dir/member_<seed>``.

    Members are trained in chunks of ``parallel_members`` (default 1 = the legacy
    sequential behavior) jointly in one process, sharing the once-featurized,
    optionally device-resident tensors.  Per-member seeds/inits/permutations stay
    independent, so the ensemble is statistically identical to sequential training
    (see :func:`_train_members`).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = kwargs.get("config") or TrainConfig()
    if epochs is not None:
        cfg.epochs = epochs
    if parallel_members is not None:
        cfg.parallel_members = int(parallel_members)
    pm = max(1, int(cfg.parallel_members))
    # freeze-and-finetune: load the champion members ONCE and index them by
    # ensemble position (reusing member 0 when the champion has fewer members).
    if _cyclen_masked_mode(cfg) and not cfg.init_from:
        raise ValueError(
            "freeze_trunk_cyclen / trunk_finetune_lr_mult requires init_from")
    champ_states = (_load_champion_member_states(cfg.init_from)
                    if cfg.init_from else None)
    if champ_states is not None:
        print(f"=== init-from {cfg.init_from}: {len(champ_states)} champion "
              f"member(s), freeze_trunk_cyclen={cfg.freeze_trunk_cyclen}, "
              f"trunk_finetune_lr_mult={cfg.trunk_finetune_lr_mult} ===",
              flush=True)
    store_dir = kwargs.get("store_dir", DEFAULT_STORE)
    splits_dir = kwargs.get("splits_dir", DEFAULT_SPLITS)
    subset_rows = kwargs.get("subset_rows")
    log_every = kwargs.get("log_every", 1)
    verbose = kwargs.get("verbose", True)
    device_t = torch.device(device)

    # featurize the split ONCE and share the tensors across all members.
    print("=== featurizing split (shared across members) ===", flush=True)
    t_feat = time.time()
    reader = StoreReader(store_dir)
    fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
    manifest = _load_split(reader, split, splits_dir, base_seed)
    # hires A2: the cond_v6 ``prior_power`` channels need the diffusion prior's
    # two global scalars, and those must be fit on TRAIN ROWS ONLY (leakage rule,
    # exactly as the cyclen physics prior is).  Fitting reads only k-inf curves +
    # the map labels, so it runs BEFORE featurization and has no circularity.
    power_prior = _fit_power_prior_for_split(reader, manifest, fuel, split,
                                             cond_schema, verbose=verbose)
    encoder = FeatureEncoder(cond_schema=cond_schema, power_prior=power_prior)
    target_names = targets_for(cfg.promote_max_asm_bu, cfg.promote_fxy)
    train_pre = build_precomputed(
        reader, manifest, fuel, fold="train", augment=cfg.augment,
        encoder=encoder, seed=base_seed, subset_rows=subset_rows,
        censor_dataset_a_pin_labels=censor_dataset_a_pin_labels,
        promote_max_asm_bu=cfg.promote_max_asm_bu,
        promote_fxy=cfg.promote_fxy,
        include_axial=cfg.axial_head,
        include_traj=cfg.traj_weight > 0.0,
        traj_anchors=cfg.traj_anchors)
    val_n = None if subset_rows is None else max(64, subset_rows // 8)
    val_pre = build_precomputed(
        reader, manifest, fuel, fold="val", augment=False, encoder=encoder,
        seed=base_seed, subset_rows=val_n,
        censor_dataset_a_pin_labels=censor_dataset_a_pin_labels,
        promote_max_asm_bu=cfg.promote_max_asm_bu,
        promote_fxy=cfg.promote_fxy,
        include_axial=cfg.axial_head,
        include_traj=cfg.traj_weight > 0.0,
        traj_anchors=cfg.traj_anchors)
    print(f"=== featurized train={len(train_pre)} val={len(val_pre)} in "
          f"{time.time() - t_feat:.1f}s ===", flush=True)
    globals_names = encoder.globals_names
    _prepare_axial_basis(cfg, train_pre, val_pre, verbose=verbose)
    prior, val_prior_vals = _prepare_cyclen_prior(
        cfg, train_pre, val_pre, fuel, store_dir, split)
    if cfg.distill_targets:
        # TRAIN fold only — a soft target on a holdout row would leak a teacher's
        # (label-trained) opinion into the honest eval fold.  Raises when the
        # record_id join matches too few rows (see attach_distill_targets).
        n_kd = attach_distill_targets(train_pre, cfg.distill_targets, target_names,
                                      min_match_frac=cfg.distill_min_match_frac)
        print(f"=== distillation: soft targets on {n_kd}/{len(train_pre)} train "
              f"rows (weight {cfg.distill_weight}) ===", flush=True)

    eff_batch, lr, lr_final, warm, sched_meta = _resolve_schedule(cfg, device_t)
    resident = _resident_fits(train_pre, val_pre, device_t, cfg)
    print(f"=== schedule: effective_batch={eff_batch} lr={lr:.2e} "
          f"warmup_epochs={warm} device_resident={resident} "
          f"parallel_members={pm} torch_compile={cfg.torch_compile} ===",
          flush=True)

    seeds = [base_seed + i for i in range(n)]
    dirs: list[Path] = []
    for cstart in range(0, n, pm):
        chunk = seeds[cstart:cstart + pm]
        labels = ", ".join(str(s) for s in chunk)
        print(f"=== training members {cstart + 1}-{cstart + len(chunk)}/{n} "
              f"(seeds {labels}) ===", flush=True)
        t0 = time.time()
        init_states = None
        if champ_states is not None:
            init_states = [champ_states[min(cstart + j, len(champ_states) - 1)]
                           for j in range(len(chunk))]
        members = _train_members(
            chunk, train_ds=train_pre, val_ds=val_pre, cfg=cfg, device=device_t,
            globals_names=globals_names, reader=reader, eff_batch=eff_batch,
            lr=lr, lr_final=lr_final, warm=warm, resident=resident,
            compile_flag=cfg.torch_compile, n_channels=len(encoder.channels),
            channel_names=tuple(encoder.channels),
            verbose=verbose, log_every=log_every, manifest=manifest,
            target_names=target_names, init_states=init_states)
        for seed, m in zip(chunk, members):
            member_dir = _finalize_member(
                out / f"member_{seed}", m, cfg=cfg, split=split,
                globals_names=globals_names, encoder=encoder,
                train_ds=train_pre, val_ds=val_pre, device=device_t,
                sched_meta=sched_meta, resident=resident,
                target_names=target_names, cyclen_prior=prior,
                val_prior_values=val_prior_vals)
            dirs.append(member_dir)
        print(f"=== chunk of {len(chunk)} done in {time.time() - t0:.1f}s ===",
              flush=True)
    manifest_out = {
        "members": [str(d.name) for d in dirs],
        "n_members": n,
        "split": split,
        "base_seed": base_seed,
        "parallel_members": pm,
    }
    (out / "ensemble.json").write_text(
        json.dumps(manifest_out, indent=2, sort_keys=True), encoding="utf-8")
    if prior is not None:
        from .physics_prior import PRIOR_NAME
        prior.save(out / PRIOR_NAME)
    if power_prior is not None:
        # The fitted (M^2, extrap) belong next to the ensemble: they define the
        # cond_v6 ``prior_power`` channel, so a serving encoder MUST rebuild with
        # these values, not the module defaults.
        from .power_prior import POWER_PRIOR_NAME
        power_prior.write(out / POWER_PRIOR_NAME)

    # fit calibration on the S1-val ensemble residuals (plan sec. 4.4)
    try:
        from .calibrate import fit_calibration
        fit_calibration(dirs, split=split, device=device, out_dir=out,
                        store_dir=store_dir, splits_dir=splits_dir)
        print("=== calibration fitted ===", flush=True)
    except Exception as exc:      # pragma: no cover
        print(f"WARNING: calibration failed: {exc}", flush=True)

    # fit the per-cell cyclen + F_r affine calibrations INTO the new model dir.
    # Runs on the TRAINING device: the fit is a serve-path forward over every
    # labelled ga80 train row, which is minutes on a GPU and tens of minutes on
    # CPU — worth threading rather than defaulting.
    fit_cell_calibrations(out, store_dir=store_dir, splits_dir=splits_dir,
                          split=split, cfg=cfg, device=str(device_t))
    return dirs


def _report_calibration_failure(label: str, exc: BaseException,
                                strict: bool) -> None:
    """Report a per-cell calibration failure LOUDLY (or re-raise under ``strict``).

    The old behaviour — one ``WARNING:`` line among hundreds of lines of training
    log — is what let a ``TypeError`` on a ``str`` ``model_dir`` drop calibration
    artifacts unnoticed.  A skip must look like a failure, not like progress.
    """
    if strict:
        raise RuntimeError(
            f"per-cell {label} calibration failed and strict=True") from exc
    import traceback

    print("=" * 78, flush=True)
    print(f"ERROR: per-cell {label} calibration FAILED and was SKIPPED "
          f"({type(exc).__name__}: {exc})", flush=True)
    print(f"ERROR: the model dir will have NO {label} calibration artifact.",
          flush=True)
    # stdout, not stderr: the training log is what a human reads afterwards, and
    # a traceback that lands in a different stream than its banner is a traceback
    # nobody connects to the failure.
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stdout)
    print("=" * 78, flush=True)


def fit_cell_calibrations(model_dir: str | Path, *,
                          store_dir: str | Path = DEFAULT_STORE,
                          splits_dir: str | Path = DEFAULT_SPLITS,
                          split: str | None = None,
                          cfg: TrainConfig | None = None,
                          device: str = "cpu",
                          strict: bool = False) -> dict[str, Any]:
    """Fit ALL THREE per-cell affine calibration artifacts into a finished model dir.

    Historically ``cell_calibration.json`` (cyclen) and ``f_r_calibration.json``
    (F_r) had to be produced by hand after every retrain — nothing in the
    training or curriculum-retrain path wrote them.  A retrained champion
    therefore served UNCALIBRATED while the champion it was gated against served
    calibrated, which is both an unfair comparison and a silent screening-recall
    loss.  Fitting them here makes a retrain self-contained.

    Both fits run through the exact serve path with the calibrations disabled and
    use the champion's own split-manifest **train** rows only; the leakage guard
    (`no fitted row may be a holdout id`) is asserted inside
    :func:`cell_calibrate._fit_cell_affine_target`.

    **A skip is LOUD.**  Failures are still swallowed by default — a missing
    calibration must never lose a completed multi-hour training run — but a
    swallowed failure now prints a banner with the exception TYPE and a traceback
    instead of one ``WARNING:`` line, and is recorded in the returned dict under
    ``"failed"`` so a caller can assert on it.  ``strict=True`` re-raises instead,
    for tests and for any caller that would rather fail than ship a half-calibrated
    model dir.  The bug this replaces: ``model_dir`` arriving as a ``str`` made
    ``model_dir / out_name`` raise ``TypeError`` at the very end of each fit, and
    the old one-line warning meant a model dir could silently lose its ``f_q`` /
    ``ao_abs`` (indeed all five scalar) calibrations with nothing in the log that
    read as an error.  ``model_dir`` is coerced to ``Path`` here AND at the root
    (:func:`cell_calibrate._fit_cell_affine_target`).
    """
    cfg = cfg or TrainConfig()
    model_dir = Path(model_dir)
    out: dict[str, Any] = {"cyclen": None, "f_r": None, "cbc_max": None,
                           "f_q": None, "ao_abs": None, "flatness": None,
                           "cyclen_physics_prior": None, "failed": {}}
    if not cfg.auto_fit_cell_calibration:
        return out
    from .cell_calibrate import (
        CELL_CALIB_NAME, fit_cell_affine, fit_cell_affine_ao, fit_cell_affine_cbc,
        fit_cell_affine_fq, fit_cell_affine_fr, fit_flatness_calibration,
    )

    # Under freeze-and-finetune -- and under --trunk-finetune-lr-mult, which keeps
    # the same cyclen row mask -- the cyclen head is byte-identical to the
    # champion, so its per-cell calibration must be the champion's VERBATIM — re-fitting it
    # against a (possibly grown) store would silently change served cyclen and
    # break the "cyclen == champion" guarantee.  The F_r calibration is still
    # freshly fit.  Copy (never re-fit) the cyclen artifact in this mode.
    freeze_copy = _cyclen_masked_mode(cfg) and bool(cfg.init_from)

    # cbc_max joined 2026-07-29 (debug-panel); f_q, ao_abs and the flatness pair
    # joined the same day once the panel showed every target carries a per-cell
    # shift (bias share: f_q 71%, cbc_max 70%, f_r 46%, node_peak 38%, cyclen 21%,
    # ao_abs 17%, map_cov 84%).  One loop, so a new target can never be the one
    # nobody remembered to automate.
    for label, fn in (("cyclen", fit_cell_affine), ("f_r", fit_cell_affine_fr),
                      ("cbc_max", fit_cell_affine_cbc), ("f_q", fit_cell_affine_fq),
                      ("ao_abs", fit_cell_affine_ao),
                      ("flatness", fit_flatness_calibration)):
        if label == "cyclen" and freeze_copy:
            src = Path(cfg.init_from) / CELL_CALIB_NAME
            dst = Path(model_dir) / CELL_CALIB_NAME
            try:
                if not src.is_file():
                    raise FileNotFoundError(src)
                shutil.copyfile(src, dst)
                out[label] = {"copied_from_champion": str(src)}
                print(f"=== per-cell cyclen calibration COPIED from champion "
                      f"{src} (frozen cyclen head; not re-fit) ===", flush=True)
            except Exception as exc:      # pragma: no cover - never lose a retrain
                _report_calibration_failure(label, exc, strict)
                out["failed"][label] = f"{type(exc).__name__}: {exc}"
            continue
        try:
            art = fn(model_dir, store_dir, splits_dir, split=split,
                     device=device, library_id=cfg.calibration_library_id)
            out[label] = {"n_cells_fitted": art.get("n_cells_fitted"),
                          "n_cells_skipped": art.get("n_cells_skipped"),
                          "n_train_labelled": art.get("n_train_labelled")}
            print(f"=== per-cell {label} calibration fitted: "
                  f"{art.get('n_cells_fitted')} cells ===", flush=True)
        except Exception as exc:      # pragma: no cover - never lose a retrain
            _report_calibration_failure(label, exc, strict)
            out["failed"][label] = f"{type(exc).__name__}: {exc}"

    # Freeze-and-finetune with the cyclen PHYSICS PRIOR on: the served cyclen is
    # residual + prior and the frozen residual head was trained against the
    # CHAMPION's prior. Copy the champion's prior artifact VERBATIM (byte-identical
    # — a re-serialize would also drift line endings) so the served prior can never
    # be a store-refit. Mirrors the per-cell cyclen calibration copy above; only in
    # freeze mode AND with the prior flag on (from-scratch path leaves it re-fit).
    if freeze_copy and cfg.cyclen_physics_prior:
        from .physics_prior import PRIOR_NAME
        src = Path(cfg.init_from) / PRIOR_NAME
        dst = Path(model_dir) / PRIOR_NAME
        try:
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copyfile(src, dst)
            out["cyclen_physics_prior"] = {"copied_from_champion": str(src)}
            print(f"=== cyclen physics prior COPIED from champion {src} "
                  f"(frozen cyclen head; not re-fit) ===", flush=True)
        except Exception as exc:      # pragma: no cover - never lose a retrain
            _report_calibration_failure("cyclen_physics_prior", exc, strict)
            out["failed"]["cyclen_physics_prior"] = f"{type(exc).__name__}: {exc}"
    if out["failed"]:
        print("!" * 78, flush=True)
        print(f"!!! {len(out['failed'])} PER-CELL CALIBRATION(S) MISSING FROM "
              f"{model_dir}: {sorted(out['failed'])}", flush=True)
        print("!!! This model dir will SERVE UNCALIBRATED on those axes and is "
              "NOT comparable to a fully-calibrated champion.", flush=True)
        print("!" * 78, flush=True)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def main(argv: Sequence[str] | None = None) -> int:
    # stdio is REDIRECTED to a log file by every launcher, so Windows gives it
    # the ANSI codepage; a non-ASCII banner then raises UnicodeEncodeError and
    # kills the run (incident 2026-08-30).  This module has its own __main__,
    # so it cannot rely on lpopt.cli.main's call.
    configure_stdio()
    ap = argparse.ArgumentParser(description="Train the PosValNet ensemble")
    ap.add_argument("--ensemble", type=int, default=1, help="number of members")
    ap.add_argument("--split", default="S1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out-dir", default="data/models/local")
    ap.add_argument("--store-dir", default=DEFAULT_STORE)
    ap.add_argument("--splits-dir", default=DEFAULT_SPLITS)
    ap.add_argument("--base-seed", type=int, default=20260716)
    ap.add_argument("--cond-schema", default=DEFAULT_COND_SCHEMA,
                    help="conditioning/feature schema: v2 | v3 (default) | v4 | "
                         "v5 (poison-agnostic) | v5_noshape (v5 ablation) | "
                         "v6 (hires bundle) | v6b (v6 + regime burnup table + "
                         "source-chain channels)")
    # --- v5 bundle knobs (all default OFF == the legacy training path) --------
    ap.add_argument("--cyclen-physics-prior", action="store_true",
                    help="regress the cyclen RESIDUAL against the leading-order "
                         "reactivity-balance prior (added back at serve time)")
    ap.add_argument("--f-r-rank-weight", type=float, default=None,
                    help="within-cell F_r elite-rank hinge weight (default 0.1, ON; "
                         "0 disables — low-F_r pairs are up-weighted)")
    ap.add_argument("--map-fr-consistency-weight", type=float, default=None,
                    help="map(boc_power)-peak vs F_r-head consistency weight "
                         "(default 0.0 = OFF; recommended small, e.g. 0.1)")
    ap.add_argument("--map-peak-weight", type=float, default=None,
                    help="node_peak-B: up-weight map_loss at hot/peak nodes by "
                         "1+w*relu(map_z) (default 0.0 = OFF; A/B, e.g. 2.0)")
    ap.add_argument("--cyclen-rank-weight", type=float, default=None,
                    help="within-cell cyclen elite-rank hinge weight (default 0.1; "
                         "raise to protect cyclen ranking, e.g. 0.25)")
    ap.add_argument("--quantile-heads", action="store_true",
                    help="add pinball-loss q10/q50/q90 heads for f_r + cyclen")
    ap.add_argument("--quantile-weight", type=float, default=None,
                    help="loss weight on the pinball term (default 0.2)")
    ap.add_argument("--promote-max-asm-bu", action="store_true",
                    help="promote max_assembly_burnup to a first-class target "
                         "(global head grows by one output, masked where absent)")
    ap.add_argument("--promote-fxy", action="store_true",
                    help="promote f_xy (MASTER FXYP, pin planar peaking) to a "
                         "first-class target: one more head row, masked wherever "
                         "the label is absent (nearly all rows today), "
                         "regressing the residual against the fitted F_xy~F_r "
                         f"prior; refuses to train below {MIN_FXY_LABELS} "
                         "labelled train rows")
    ap.add_argument("--fxy-direct", action="store_true",
                    help="f_xy head predicts the ABSOLUTE value instead of a "
                         "residual against the F_xy~F_r prior (the prior is "
                         "still fitted and reported, but composes nothing)")
    ap.add_argument("--fxy-prior-on-predicted", action="store_true",
                    help="fit the F_xy~F_r prior on the model's OWN predicted "
                         "F_r (the raw mu row the composition reads) instead of "
                         "the measured F_r, so the F_r head's uncalibrated bias "
                         "is absorbed instead of inherited; needs --init-from")
    ap.add_argument("--fxy-select-weight", type=float, default=None,
                    help="weight of the f_xy val score (within-cell Spearman "
                         "minus z-MAE) added to the composite for best-epoch / "
                         "early-stop selection; 0 (default) = legacy selection")
    ap.add_argument("--fxy-select-band", type=float, default=None,
                    help="restrict the f_xy SELECTION Spearman to each GATE "
                         "cell's low-f_xy band (e.g. 0.50 = the cell median and "
                         "below), the axis the ranking clause is scored on; "
                         "1.0 (default) = the legacy whole-cell metric")
    # --- arm 5: within-cell pairwise rank hinge on the composed f_xy row -----
    ap.add_argument("--fxy-rank-weight", type=float, default=None,
                    help="within-cell f_xy pairwise margin-rank hinge weight "
                         "(default 0.0 = OFF and byte-identical; prereg arm 5 "
                         "uses 3.0); requires --promote-fxy")
    ap.add_argument("--fxy-rank-cell", choices=("gate", "legacy"), default=None,
                    help="cell the f_xy hinge pairs WITHIN: 'gate' (default) = "
                         "(case_pair, feed), the partition every f_xy gate "
                         "scores on; 'legacy' = the cyclen cell "
                         "(feed, e_core-bin, dataset)")
    ap.add_argument("--fxy-rank-margin-z", type=float, default=None,
                    help="z-space margin the f_xy hinge demands per pair (0.1)")
    ap.add_argument("--fxy-rank-min-gap", type=float, default=None,
                    help="ignore f_xy pairs whose RAW gap is below this "
                         "(default 0.005; MASTER FXYP repeat noise is 0.000000)")
    ap.add_argument("--fxy-rank-low-thresh", type=float, default=None,
                    help="up-weight pairs whose min RAW f_xy is <= this (1.60)")
    ap.add_argument("--fxy-rank-low-weight", type=float, default=None,
                    help="weight multiple for those boundary pairs (3.0)")
    ap.add_argument("--distill-targets", default=None,
                    help="path to a prebuilt soft-target cache (lpopt.model.distill); "
                         "enables per-cell teacher distillation on the FULL corpus")
    ap.add_argument("--distill-weight", type=float, default=None,
                    help="loss weight on the distillation term (default 0.3)")
    ap.add_argument("--distill-cyclen-boost-factor", type=float, default=None,
                    help="arm1(b) cyclen protection: multiply the cyclen-column distill "
                         "loss by this on --distill-cyclen-boost-bands (default 1.0=OFF)")
    ap.add_argument("--distill-cyclen-boost-bands", default=None,
                    help="e_core bands 'lo-hi,...' for the cyclen distill boost, "
                         "e.g. 5.0-5.25,6.0-6.25")
    ap.add_argument("--distill-min-match-frac", type=float, default=None,
                    help="hard-error if the cache join matches < this fraction of "
                         "its built soft-target rows (default 0.5; 0 disables)")
    ap.add_argument("--no-auto-cell-calibration", dest="auto_cell_calibration",
                    action="store_false", default=True,
                    help="skip the automatic post-train per-cell cyclen/F_r "
                         "calibration fit (default: fit both)")
    ap.add_argument("--subset-rows", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="pin the effective batch (else 1024 on cuda, 256 on cpu)")
    ap.add_argument("--parallel-members", type=int, default=None,
                    help="members trained jointly per process (default 1)")
    ap.add_argument("--device-resident", dest="device_resident",
                    action="store_true", default=None,
                    help="hold the dataset on-GPU (default on for cuda)")
    ap.add_argument("--no-device-resident", dest="device_resident",
                    action="store_false",
                    help="disable the device-resident dataset")
    ap.add_argument("--torch-compile", action="store_true",
                    help="wrap member forward in torch.compile (opt-in)")
    # --- network-shape knobs (default == the current 112/6/256 architecture) ---
    ap.add_argument("--width", type=int, default=None,
                    help="member CNN trunk width (default 112; 160 ~= 2x params)")
    ap.add_argument("--n-blocks", type=int, default=None,
                    help="number of residual blocks (default 6)")
    ap.add_argument("--head-hidden", type=int, default=None,
                    help="global/conv head MLP width (default 256)")
    ap.add_argument("--censor-a-pin-labels", dest="censor_a_pin_labels",
                    action="store_true", default=True,
                    help="mask Dataset-A max_pin_burnup labels in training "
                         "(default; A's pin label is a cache surrogate)")
    ap.add_argument("--no-censor-a-pin-labels", dest="censor_a_pin_labels",
                    action="store_false",
                    help="train on Dataset-A max_pin_burnup labels (legacy behavior)")
    # --- freeze-and-finetune (default OFF == from-scratch training) ------------
    ap.add_argument("--init-from", default=None,
                    help="champion model dir; initialize each member's PosValNet "
                         "from its matching member_*/model.pt (strict) before "
                         "training (fine-tune instead of from-scratch)")
    ap.add_argument("--freeze-trunk-cyclen", action="store_true",
                    help="(requires --init-from) freeze the shared trunk + cyclen "
                         "output rows so CYCLEN stays byte-identical to the "
                         "champion while F_r + node_peak/map heads adapt; the "
                         "champion's per-cell cyclen calibration is copied verbatim")
    ap.add_argument("--trunk-finetune-lr-mult", type=float, default=None,
                    metavar="MULT",
                    help="(requires --init-from; MUTUALLY EXCLUSIVE with "
                         "--freeze-trunk-cyclen) keep the CYCLEN half of "
                         "--freeze-trunk-cyclen but NOT the trunk freeze: the "
                         "cyclen rows of mu/log_sigma/quantile stay "
                         "gradient-masked and weight-decay-free, and the "
                         "champion's cyclen physics prior + per-cell cyclen "
                         "calibration are still copied verbatim, while the shared "
                         "trunk TRAINS in its own optimizer group at "
                         "base_lr*MULT (0.0 = OFF = the legacy path)")
    # --- hires bundle (hires_model_ab_design_20260725.md) ----------------------
    ap.add_argument("--map-decoder", choices=("linear", "multiscale"), default=None,
                    help="map head architecture (default linear = the single 1x1 "
                         "conv); 'multiscale' adds a stem + intermediate-tap skip "
                         "path so the map is not read off the fully low-pass "
                         "filtered trunk feature alone [arm A1]")
    ap.add_argument("--map-prior-residual", action="store_true",
                    help="wire the cond_v6 'prior_power' diffusion power-map prior "
                         "into the map head as an additive learnable skip, so the "
                         "head learns the RESIDUAL against the leading-order "
                         "physics solve (requires --cond-schema v6 or v6_prior) "
                         "[arm A2]")
    ap.add_argument("--map-spectral-weight", type=float, default=None,
                    help="weight of the band-weighted FFT map loss that penalizes "
                         "high-wavenumber amplitude/phase error directly "
                         "(default 0.0 = OFF; recommended 0.3) [arm A3]")
    # --- axial bundle (decision D10) ------------------------------------------
    ap.add_argument("--axial-head", action="store_true",
                    help="predict the EDIT6 AXIAL power profile at BOC + EOC as "
                         "shape-basis coefficients (F_z / AO / ASI then follow "
                         "analytically from the profile). Default OFF")
    ap.add_argument("--axial-rank", type=int, default=None,
                    help="axial shape-basis rank (default 6; measured "
                         "leave-one-campaign-out F_z reconstruction error 9e-3, "
                         "half the within-cell F_z spread)")
    ap.add_argument("--axial-weight", type=float, default=None,
                    help="loss weight on the axial coefficient term (default 0.2)")
    # --- A/B round-2 variance arms (all default OFF == the champion recipe) ----
    ap.add_argument("--traj-weight", type=float, default=None,
                    help="[arm A1] weight of the EDIT5 burnup-TRAJECTORY "
                         "supervision: re-run the map decoder on a trunk feature "
                         "FiLM-conditioned by cycle-burnup fraction and supervise "
                         "it against the '<rid>__traj' planes at --traj-anchors "
                         "(default 0.0 = OFF; recommended 0.3)")
    ap.add_argument("--traj-anchors", default=None,
                    help="cycle-burnup fractions supervised by --traj-weight, "
                         "comma-separated in [0,1] (default 0,0.25,0.5,0.75,1)")
    ap.add_argument("--cbc-provenance-offset", action="store_true",
                    help="[arm A2] learn one cbc_max label-convention offset per "
                         "provenance group (mocha_native / ga_native), applied "
                         "ONLY inside the cbc regression loss; the reference "
                         "group (master_native) is pinned to 0 so serving stays "
                         "MASTER-native. Default OFF")
    ap.add_argument("--map-peak-topk-weight", type=float, default=None,
                    help="[arm A3] extra map-loss weight (1+w) on the --map-peak-topk "
                         "hottest LABEL slots of each map plane (default 0.0 = OFF; "
                         "recommended 2.0). Finer-grained than --map-peak-weight, "
                         "which re-weights every above-average node continuously")
    ap.add_argument("--map-peak-topk", type=int, default=None,
                    help="K for --map-peak-topk-weight (default 5 of 69 slots)")
    args = ap.parse_args(argv)

    cfg = TrainConfig(num_workers=args.num_workers)
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        cfg.batch_size_explicit = True
    if args.parallel_members is not None:
        cfg.parallel_members = args.parallel_members
    if args.device_resident is not None:
        cfg.device_resident = args.device_resident
    if args.torch_compile:
        cfg.torch_compile = True
    if args.width is not None:
        cfg.width = int(args.width)
    if args.n_blocks is not None:
        cfg.n_blocks = int(args.n_blocks)
    if args.head_hidden is not None:
        cfg.head_hidden = int(args.head_hidden)
    cfg.cyclen_physics_prior = bool(args.cyclen_physics_prior)
    cfg.quantile_heads = bool(args.quantile_heads)
    if args.quantile_weight is not None:
        cfg.quantile_weight = float(args.quantile_weight)
    if args.f_r_rank_weight is not None:
        cfg.f_r_rank_weight = float(args.f_r_rank_weight)
    if args.map_peak_weight is not None:
        cfg.map_peak_weight = float(args.map_peak_weight)
    if args.cyclen_rank_weight is not None:
        cfg.cyclen_rank_weight = float(args.cyclen_rank_weight)
    if args.distill_cyclen_boost_factor is not None:
        cfg.distill_cyclen_boost_factor = float(args.distill_cyclen_boost_factor)
    if args.distill_cyclen_boost_bands is not None:
        cfg.distill_cyclen_boost_bands = str(args.distill_cyclen_boost_bands)
    if args.map_fr_consistency_weight is not None:
        cfg.map_fr_consistency_weight = float(args.map_fr_consistency_weight)
    cfg.promote_max_asm_bu = bool(args.promote_max_asm_bu)
    cfg.promote_fxy = bool(args.promote_fxy)
    cfg.fxy_prior_residual = not bool(args.fxy_direct)
    cfg.fxy_prior_on_predicted = bool(args.fxy_prior_on_predicted)
    if args.fxy_select_weight is not None:
        cfg.fxy_select_weight = float(args.fxy_select_weight)
    if args.fxy_select_band is not None:
        cfg.fxy_select_band = float(args.fxy_select_band)
        if not 0.0 < cfg.fxy_select_band <= 1.0:
            ap.error("--fxy-select-band must be in (0, 1] (1.0 = legacy metric)")
    if args.fxy_rank_weight is not None:
        cfg.fxy_rank_weight = float(args.fxy_rank_weight)
    if args.fxy_rank_cell is not None:
        cfg.fxy_rank_cell = str(args.fxy_rank_cell)
    if args.fxy_rank_margin_z is not None:
        cfg.fxy_rank_margin_z = float(args.fxy_rank_margin_z)
    if args.fxy_rank_min_gap is not None:
        cfg.fxy_rank_min_gap = float(args.fxy_rank_min_gap)
    if args.fxy_rank_low_thresh is not None:
        cfg.fxy_rank_low_thresh = float(args.fxy_rank_low_thresh)
    if args.fxy_rank_low_weight is not None:
        cfg.fxy_rank_low_weight = float(args.fxy_rank_low_weight)
    if cfg.fxy_rank_weight < 0.0:
        ap.error("--fxy-rank-weight must be >= 0 (0 = OFF)")
    if cfg.fxy_rank_weight > 0.0 and not cfg.promote_fxy:
        # There is no f_xy row to rank without the head; silently training the
        # seven legacy targets under an "arm 5" command line is the drift the
        # cond_schema/head_hidden guards exist to stop.
        ap.error("--fxy-rank-weight > 0 requires --promote-fxy "
                 "(there is no f_xy head row to rank without it)")
    cfg.auto_fit_cell_calibration = bool(args.auto_cell_calibration)
    cfg.distill_targets = args.distill_targets
    if args.distill_weight is not None:
        cfg.distill_weight = float(args.distill_weight)
    if args.distill_min_match_frac is not None:
        cfg.distill_min_match_frac = float(args.distill_min_match_frac)
    cfg.init_from = args.init_from
    cfg.freeze_trunk_cyclen = bool(args.freeze_trunk_cyclen)
    if cfg.freeze_trunk_cyclen and not cfg.init_from:
        ap.error("--freeze-trunk-cyclen requires --init-from")
    if args.trunk_finetune_lr_mult is not None:
        cfg.trunk_finetune_lr_mult = float(args.trunk_finetune_lr_mult)
    if cfg.trunk_finetune_lr_mult < 0.0:
        ap.error("--trunk-finetune-lr-mult must be >= 0 "
                 "(0 = OFF = freeze/finetune flags off)")
    if cfg.trunk_finetune_lr_mult > 0.0:
        if cfg.freeze_trunk_cyclen:
            ap.error("--trunk-finetune-lr-mult and --freeze-trunk-cyclen are "
                     "mutually exclusive: the first keeps the cyclen row mask "
                     "of the second WITHOUT freezing the trunk")
        if not cfg.init_from:
            ap.error("--trunk-finetune-lr-mult requires --init-from "
                     "(there is no champion trunk to fine-tune otherwise)")
    if args.map_decoder is not None:
        cfg.map_head_mode = str(args.map_decoder)
    cfg.map_prior_residual = bool(args.map_prior_residual)
    if args.map_spectral_weight is not None:
        cfg.map_spectral_weight = float(args.map_spectral_weight)
    cfg.axial_head = bool(args.axial_head)
    if args.axial_rank is not None:
        cfg.axial_rank = int(args.axial_rank)
    if args.axial_weight is not None:
        cfg.axial_weight = float(args.axial_weight)
    # --- A/B round-2 variance arms ------------------------------------------- #
    if args.traj_weight is not None:
        cfg.traj_weight = float(args.traj_weight)
    if args.traj_anchors is not None:
        cfg.traj_anchors = _parse_traj_anchors(args.traj_anchors)
        if not cfg.traj_anchors:
            ap.error("--traj-anchors parsed to an empty list; expected "
                     "comma-separated fractions in [0,1], e.g. 0,0.25,0.5,0.75,1")
    cfg.cbc_provenance_offset = bool(args.cbc_provenance_offset)
    if args.map_peak_topk_weight is not None:
        cfg.map_peak_topk_weight = float(args.map_peak_topk_weight)
    if args.map_peak_topk is not None:
        cfg.map_peak_topk = int(args.map_peak_topk)
    # Gate on the CHANNEL INVENTORY, not a hard-coded schema list: the residual
    # skip needs the 'prior_power' plane and nothing else, so every schema that
    # carries it qualifies (v6 / v6_prior / v6b) and a future one cannot be
    # silently locked out by a stale tuple.
    if cfg.map_prior_residual and "prior_power" not in CHANNELS_BY_SCHEMA.get(
            args.cond_schema, ()):
        ap.error("--map-prior-residual requires a cond-schema carrying the "
                 "'prior_power' channel (v6 / v6_prior / v6b); got "
                 f"{args.cond_schema!r}, which has none")

    device = _resolve_device(args.device)
    print(f"device={device} torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()}", flush=True)
    dirs = train_ensemble(
        args.ensemble, split=args.split, device=device, epochs=args.epochs,
        out_dir=args.out_dir, base_seed=args.base_seed, config=cfg,
        store_dir=args.store_dir, splits_dir=args.splits_dir,
        subset_rows=args.subset_rows, cond_schema=args.cond_schema,
        censor_dataset_a_pin_labels=args.censor_a_pin_labels,
    )
    print(f"trained {len(dirs)} member(s) into {args.out_dir}", flush=True)
    return 0


__all__ = [
    "TrainConfig", "train_member", "train_ensemble", "predict_dataset",
    "save_member", "load_member", "norm_from_meta", "denormalize",
    "compute_target_norm", "compute_map_norm", "composite_metric",
    "within_case_spearman", "regression_loss", "cyclen_rank_loss",
    "f_r_rank_loss", "f_xy_rank_loss", "map_fr_consistency_loss",
    "pinball_loss", "residual_target_frame", "attach_cyclen_prior",
    "fit_cell_calibrations", "build_precomputed",
    "map_loss", "traj_loss", "top_k_slot_weight",
]


if __name__ == "__main__":
    raise SystemExit(main())
