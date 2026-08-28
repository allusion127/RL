"""PosValNet — the position-value CNN (plan sec. 4.4).

A single ensemble member maps one featurized loading pattern

* ``cells``   ``float32[B, C, 19, 19]``  physics channels (``featurize.CHANNELS``)
* ``globals`` ``float32[B, G]``          FiLM conditioning vector (``cond_v2``)

to three heads:

* **map head**    ``[B, 4, 9, 9]`` per-slot EDIT5 normalizer (BOC/EOC power,
  EOC burnup/kinf), masked to the 69 quarter slots — a multitask spatial
  regularizer (plan sec. 4.4 "map head as normalizer").
* **global head** per-target ``mu`` and ``log_sigma_alea`` for the seven targets
  ``[f_r, f_q, cbc_max, cyclen, ao_abs, discharge_burnup, max_pin_burnup]``
  (dataset ``TARGETS`` order; ``discharge_burnup`` / ``max_pin_burnup`` promoted
  to first-class targets in Phase D, plan sec. 12.4), from a
  multiplicity-weighted masked mean+max pool → MLP.  Older cond_v2 checkpoints
  carry only the first five targets (``net_config.n_targets == 5``) and rebuild
  at that width; the 7-target head is the cond_v3 default.
* **conv head**   a single convergence logit.

Trunk: Conv stem ``C→W`` (GroupNorm + SiLU) → 6 residual blocks (GN/SiLU) with a
FiLM(globals) modulation injected every two blocks.  The 19x19 grid is the
mirror-expanded full core, so a plain masked mean over the fuel cells is already
the orbit-multiplicity-weighted mean (an interior slot occupies 4 grid cells, an
axis slot 2, the centre 1).

The default width (112) is chosen so a member lands at ~1.5M parameters, inside
the plan's 1.0M–2.5M acceptance band (a pure-64ch trunk underfills it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .featurize import C as N_CHANNELS
from ..vendor.masterrl.domain import SLOTS

#: Number of regression targets predicted by the global head (dataset order).
#: Phase D (plan sec. 12.4) promoted ``discharge_burnup`` and ``max_pin_burnup``
#: to first-class targets, taking the head from 5 to 7.  A cond_v2 checkpoint
#: stores ``n_targets == 5`` in its ``net_config`` and rebuilds at that width.
N_TARGETS = 7
#: Number of EDIT5 map channels (map head output planes).
N_MAP_CHANNELS = 4
#: Padded grid side and quarter side.
_GRID = 19
_QUARTER = 9
_GRID_CENTER = 9          # centre index of the 19x19 grid (see featurize)

#: Map-head output channels the (optional) burnup-trajectory readout reuses, in
#: :data:`lpopt.data.traj.STEP_PLANES` order ``(power, burnup, kinf)``.
#:
#: These are ``(eoc_power, eoc_burnup, eoc_kinf)`` — the map head's channels 1..3.
#: The choice is not free: those three planes are EXACTLY what the trajectory
#: holds at its last step (verified bit-for-bit on the stored labels, see
#: :mod:`lpopt.data.traj`).  Reusing them makes the trajectory readout a
#: *generalisation of the EOC readout across burnup* rather than a second,
#: independently-scaled head — same weights, same z-score constants, and at
#: burnup fraction 1 the two must agree.
TRAJ_MAP_CHANNELS: tuple[int, ...] = (1, 2, 3)


def _tap_indices(n_blocks: int) -> tuple[int, ...]:
    """Three evenly-spaced block indices for the multiscale map decoder.

    Always ends at the final block, so the decoder's input strictly CONTAINS the
    legacy map head's input (the last trunk feature).  ``n_blocks=6 -> (1, 3, 5)``.
    """
    n = max(int(n_blocks), 1)
    picks = sorted({max(0, round(n * f) - 1) for f in (1 / 3, 2 / 3, 1.0)})
    return tuple(picks)


def _slot_indices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather/scatter indices linking 19x19 SE cells and the 9x9 quarter map.

    For each of the 69 quarter slots: ``se_r/se_c`` are its SE-mirror position in
    the 19x19 grid ``(9+row, 9+col)``; ``q_r/q_c`` are its ``(row, col)`` in the
    9x9 quarter — the exact layout ``edit5._quadrant`` writes the target maps in.
    """
    se_r, se_c, q_r, q_c = [], [], [], []
    for slot in SLOTS:
        se_r.append(_GRID_CENTER + slot.row)
        se_c.append(_GRID_CENTER + slot.col)
        q_r.append(slot.row)
        q_c.append(slot.col)
    return (
        torch.tensor(se_r, dtype=torch.long),
        torch.tensor(se_c, dtype=torch.long),
        torch.tensor(q_r, dtype=torch.long),
        torch.tensor(q_c, dtype=torch.long),
    )


class FiLM(nn.Module):
    """Feature-wise linear modulation from the global conditioning vector."""

    def __init__(self, n_globals: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.Linear(n_globals, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        self.channels = channels

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(g)                 # [B, 2C]
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        # (1 + scale) keeps the block near-identity at init.
        return x * (1.0 + scale) + shift


class ResidualBlock(nn.Module):
    """Pre-activation-ish residual block: (GN,SiLU,Conv)x2 + skip."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


@dataclass
class PosValNetConfig:
    """Shape/width knobs for :class:`PosValNet`."""

    in_channels: int = N_CHANNELS
    n_globals: int = 10
    width: int = 112
    n_blocks: int = 6
    groups: int = 8
    film_every: int = 2
    head_hidden: int = 256
    n_targets: int = N_TARGETS
    n_map_channels: int = N_MAP_CHANNELS
    #: --- optional pinball-loss quantile head (v5 bundle) -------------------
    #: ``n_quantile_targets`` targets x ``n_quantiles`` levels of DIRECT quantile
    #: regression, alongside (never replacing) the mean/log_sigma heads.  BOTH
    #: must be > 0 for the head to exist; the default 0/0 builds a network whose
    #: module set, parameter count and ``forward`` output keys are byte-identical
    #: to the pre-v5 net, so an existing checkpoint loads and serves unchanged
    #: (a legacy ``meta.json`` simply lacks these keys and gets the defaults).
    n_quantile_targets: int = 0
    n_quantiles: int = 0
    #: --- optional multi-scale map decoder (hires bundle) -------------------
    #: ``"linear"`` (default) keeps the single 1x1 ``map_head`` conv — the module
    #: set, parameter count and ``state_dict`` keys are then byte-identical to the
    #: pre-hires net, so an existing checkpoint loads and serves unchanged.
    #: ``"multiscale"`` replaces it with a decoder that reads the STEM output plus
    #: three intermediate block taps (see :class:`MultiScaleMapDecoder`), giving
    #: the map head a high-frequency path that has not been through the full
    #: residual-conv low-pass cascade (design doc 20260725 arm A1).
    map_head_mode: str = "linear"
    #: --- optional physics-prior residual map head (hires bundle) ------------
    #: Index of the input channel holding the diffusion power-map prior
    #: (``featurize`` channel ``"prior_power"``, cond_schema v6/v6_prior).  When
    #: ``>= 0`` the map head's output is ADDED to a learnable per-plane affine of
    #: that channel, so the head learns the RESIDUAL against the leading-order
    #: physics solve while ``out["map"]`` stays ABSOLUTE — the serving path needs
    #: no change and no round-trip bookkeeping.  ``-1`` (default) registers no
    #: parameters at all, keeping the flag-off byte-identity contract.
    map_prior_channel: int = -1
    #: --- optional axial profile head (axial bundle, decision D10) -----------
    #: ``n_axial_anchors`` burnup anchors x ``n_axial_modes`` shape-basis
    #: coefficients, emitted as ``out["axial"]`` ``[B, A, K]`` in the per-mode
    #: STANDARDISED space of :class:`lpopt.data.axial.AxialBasis`.  The profile
    #: itself is ``basis.z_decode(coeff)`` — the basis (mean profile + zero-sum
    #: shape components) lives in the checkpoint meta, not in the weights, so the
    #: head never has to learn the core-average-1 normalisation the label obeys,
    #: and F_z / AO / ASI are then exact analytic functions of the emitted shape.
    #: BOTH must be > 0 for the head to exist; the default 0/0 registers no
    #: module at all, so the parameter count, ``state_dict`` keys and ``forward``
    #: output keys are identical to the pre-axial net and an existing checkpoint
    #: loads and serves unchanged (the flag-off byte-identity contract).
    n_axial_anchors: int = 0
    n_axial_modes: int = 0
    #: --- optional burnup-TRAJECTORY supervision (A/B round-2 arm A1) ---------
    #: ``n_traj_anchors`` cycle-burnup fractions x ``n_traj_planes`` EDIT5 planes,
    #: emitted as ``out["traj"]`` ``[B, A, P, 9, 9]`` — but ONLY when ``forward``
    #: is given a ``traj_frac`` argument, so nothing on the serving path changes
    #: even for a model that HAS the head.
    #:
    #: The head is not a new decoder: it re-runs the EXISTING map head /
    #: multiscale decoder on a trunk feature that has been FiLM-modulated by
    #: ``[globals, burnup_fraction]``, and reads planes
    #: :data:`TRAJ_MAP_CHANNELS` off the result.  The only new parameters are that
    #: one FiLM (``traj_film``).  BOTH must be > 0 for the head to exist; the
    #: default 0/0 registers no module at all, so the parameter count,
    #: ``state_dict`` keys and ``forward`` output keys are identical to the
    #: pre-traj net and an existing checkpoint loads and serves unchanged.
    n_traj_anchors: int = 0
    n_traj_planes: int = 0
    #: --- optional per-provenance CBC label offsets (A/B round-2 arm A2) ------
    #: Number of CBC label-convention provenance groups
    #: (:data:`lpopt.model.dataset_torch.CBC_PROVENANCE_GROUPS`).  When ``> 1`` a
    #: ``cbc_provenance_offset`` parameter of ``n - 1`` scalars is registered —
    #: group 0 (MASTER-native, the SERVE convention) has no parameter at all, so
    #: its offset is structurally, not merely numerically, zero.
    #:
    #: The parameter is deliberately NOT used in ``forward``: it exists only to be
    #: added to the cbc column of the *training* residual (see
    #: ``lpopt.model.train._step_member``).  A served prediction therefore carries
    #: the reference convention by construction — there is no serve-side branch to
    #: get wrong.  The stored values are in Z units (see ``_step_member`` for why
    #: ppm would be unlearnable under Adam); ``meta.json`` reports both.  ``0``
    #: (default) registers nothing.
    n_cbc_provenance_groups: int = 0


class MultiScaleMapDecoder(nn.Module):
    """Map head with a high-frequency skip path (design doc 20260725, arm A1).

    The baseline map head is a single 1x1 conv on the LAST trunk feature, so every
    map prediction has passed through ``2 * n_blocks`` 3x3 convolutions — a
    low-pass cascade.  ``data/reports/cyclen_nodepeak_resolution_20260725.md``
    section 3.6 measured the consequence directly: the predicted/actual map power
    ratio decays monotonically 1.00 -> 0.71 from the lowest wavenumber to Nyquist,
    and the per-mode amplitude correlation falls 0.897 -> 0.593.

    This decoder concatenates the stem output (zero smoothing passes) with three
    evenly-spaced block taps, then mixes them with a dilation-1 conv (local
    contrast) followed by a dilation-2 conv (assembly-pair scale) before the 1x1
    projection.  The stem tap is the load-bearing part: it is the only route by
    which un-smoothed spatial detail can reach the map output.
    """

    def __init__(self, width: int, n_taps: int, n_map_channels: int,
                 groups: int = 8):
        super().__init__()
        self.mix1 = nn.Conv2d(n_taps * width, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, width)
        self.mix2 = nn.Conv2d(width, width, 3, padding=2, dilation=2)
        self.norm2 = nn.GroupNorm(groups, width)
        self.proj = nn.Conv2d(width, n_map_channels, 1)
        self.act = nn.SiLU()

    def forward(self, taps: list[torch.Tensor]) -> torch.Tensor:
        h = torch.cat(taps, dim=1)
        h = self.act(self.norm1(self.mix1(h)))
        h = self.act(self.norm2(self.mix2(h)))
        return self.proj(h)


class PosValNet(nn.Module):
    """One deep-ensemble member (plan sec. 4.4)."""

    def __init__(self, config: PosValNetConfig | None = None, **overrides: Any):
        super().__init__()
        cfg = config or PosValNetConfig()
        if overrides:
            cfg = PosValNetConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg
        W = cfg.width

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, W, 3, padding=1),
            nn.GroupNorm(cfg.groups, W),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(W, cfg.groups) for _ in range(cfg.n_blocks)
        )
        # A FiLM injection after every ``film_every`` blocks.
        self.films = nn.ModuleDict()
        for b in range(cfg.n_blocks):
            if (b + 1) % cfg.film_every == 0:
                self.films[str(b)] = FiLM(cfg.n_globals, W)

        # map head: 1x1 conv over the trunk -> 4 planes, gathered to the quarter.
        # ``map_head_mode="multiscale"`` swaps in the skip-path decoder instead;
        # the default registers EXACTLY the legacy module (flag-off byte identity).
        self.map_head_mode = str(cfg.map_head_mode)
        if self.map_head_mode not in ("linear", "multiscale"):
            raise ValueError(
                f"unknown map_head_mode {cfg.map_head_mode!r}; "
                "have ('linear', 'multiscale')")
        if self.map_head_mode == "multiscale":
            # Stem tap + three evenly-spaced block taps (the last is always the
            # final block, so the decoder strictly extends the legacy input).
            self._map_taps = _tap_indices(cfg.n_blocks)
            self.map_decoder = MultiScaleMapDecoder(
                W, 1 + len(self._map_taps), cfg.n_map_channels, cfg.groups)
        else:
            self._map_taps = ()
            self.map_head = nn.Conv2d(W, cfg.n_map_channels, 1)

        # Physics-prior residual map head.  Gain is initialized to "pass the prior
        # straight through on the boc_power plane, ignore it elsewhere", so at
        # step 0 the map prediction IS the diffusion prior and the head starts
        # from a solution that already carries the high-wavenumber content the
        # trunk attenuates (design doc 20260725 §3.3).
        self.map_prior_channel = int(cfg.map_prior_channel)
        if self.map_prior_channel >= 0:
            if self.map_prior_channel >= cfg.in_channels:
                raise ValueError(
                    f"map_prior_channel {self.map_prior_channel} is outside the "
                    f"{cfg.in_channels}-channel input")
            gain = torch.zeros(cfg.n_map_channels)
            gain[0] = 1.0
            self.map_prior_gain = nn.Parameter(gain)
            self.map_prior_bias = nn.Parameter(torch.zeros(cfg.n_map_channels))

        # global + conv heads read the multiplicity-weighted mean+max pool.
        pooled = 2 * W + cfg.n_globals
        self.head_trunk = nn.Sequential(
            nn.Linear(pooled, cfg.head_hidden),
            nn.SiLU(),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(cfg.head_hidden, cfg.n_targets)
        self.log_sigma_head = nn.Linear(cfg.head_hidden, cfg.n_targets)
        self.conv_head = nn.Linear(cfg.head_hidden, 1)
        # Optional pinball-loss quantile head.  Registered ONLY when enabled, so
        # a disabled net has an identical module set / state_dict to the pre-v5
        # network (the flag-off byte-identity contract).
        self.n_quantile_targets = int(cfg.n_quantile_targets)
        self.n_quantiles = int(cfg.n_quantiles)
        self.has_quantiles = self.n_quantile_targets > 0 and self.n_quantiles > 0
        if self.has_quantiles:
            self.quantile_head = nn.Linear(
                cfg.head_hidden, self.n_quantile_targets * self.n_quantiles)
        # Optional axial profile head — registered ONLY when enabled, exactly
        # like the quantile head above, so a disabled net has an identical module
        # set / state_dict / output-key set to the pre-axial network.
        self.n_axial_anchors = int(cfg.n_axial_anchors)
        self.n_axial_modes = int(cfg.n_axial_modes)
        self.has_axial = self.n_axial_anchors > 0 and self.n_axial_modes > 0
        if self.has_axial:
            self.axial_head = nn.Linear(
                cfg.head_hidden, self.n_axial_anchors * self.n_axial_modes)
        # Optional burnup-trajectory conditioning — one FiLM over
        # ``[globals, burnup_fraction]``.  Registered ONLY when enabled, exactly
        # like the quantile / axial heads above.  ``(1 + scale)`` in
        # :class:`FiLM` keeps it near-identity at init, so at step 0 the
        # trajectory readout IS the EOC readout at every burnup fraction — a
        # sane, physically-meaningful starting point rather than noise.
        self.n_traj_anchors = int(cfg.n_traj_anchors)
        self.n_traj_planes = int(cfg.n_traj_planes)
        self.has_traj = self.n_traj_anchors > 0 and self.n_traj_planes > 0
        if self.has_traj:
            if self.n_traj_planes > len(TRAJ_MAP_CHANNELS):
                raise ValueError(
                    f"n_traj_planes {self.n_traj_planes} exceeds the "
                    f"{len(TRAJ_MAP_CHANNELS)} reusable map-head planes "
                    f"{TRAJ_MAP_CHANNELS}")
            if max(TRAJ_MAP_CHANNELS[:self.n_traj_planes]) >= cfg.n_map_channels:
                raise ValueError(
                    f"trajectory planes {TRAJ_MAP_CHANNELS[:self.n_traj_planes]} "
                    f"are outside the {cfg.n_map_channels}-channel map head")
            self.traj_film = FiLM(cfg.n_globals + 1, W)
            self._traj_channels = TRAJ_MAP_CHANNELS[:self.n_traj_planes]
        else:
            self._traj_channels = ()
        # Optional per-provenance CBC label-convention offsets.  LOSS-ONLY: never
        # read by ``forward`` (see the config docstring).  ``n - 1`` scalars, so
        # the reference group's offset is structurally zero.
        self.n_cbc_provenance_groups = int(cfg.n_cbc_provenance_groups)
        self.has_cbc_provenance = self.n_cbc_provenance_groups > 1
        if self.has_cbc_provenance:
            self.cbc_provenance_offset = nn.Parameter(
                torch.zeros(self.n_cbc_provenance_groups - 1))

        se_r, se_c, q_r, q_c = _slot_indices()
        self.register_buffer("_se_r", se_r, persistent=False)
        self.register_buffer("_se_c", se_c, persistent=False)
        self.register_buffer("_q_r", q_r, persistent=False)
        self.register_buffer("_q_c", q_c, persistent=False)

    # ------------------------------------------------------------------ #
    def _map_quarter(self, taps: list[torch.Tensor], h: torch.Tensor,
                     cells: torch.Tensor) -> torch.Tensor:
        """Map head -> physics prior -> gather SE cells into the 9x9 quarter.

        Extracted verbatim from ``forward`` so the (optional) burnup-trajectory
        readout runs the SAME decoder, the SAME prior wiring and the SAME gather
        on a FiLM-modulated feature — the shared-decoder contract of arm A1.  The
        operation order is unchanged, so the legacy path is bit-identical.
        """
        if self._map_taps:
            map_feat = self.map_decoder(taps)            # [B, 4, 19, 19]
        else:
            map_feat = self.map_head(h)                  # [B, 4, 19, 19]
        if self.map_prior_channel >= 0:
            prior_plane = cells[:, self.map_prior_channel:self.map_prior_channel + 1]
            map_feat = (map_feat
                        + self.map_prior_gain.view(1, -1, 1, 1) * prior_plane
                        + self.map_prior_bias.view(1, -1, 1, 1))
        gathered = map_feat[:, :, self._se_r, self._se_c]  # [B, 4, 69]
        b_n = gathered.shape[0]
        quarter = map_feat.new_zeros((b_n, self.config.n_map_channels, _QUARTER, _QUARTER))
        quarter[:, :, self._q_r, self._q_c] = gathered
        return quarter

    def _traj_quarter(self, taps: list[torch.Tensor], h: torch.Tensor,
                      cells: torch.Tensor, globals_: torch.Tensor,
                      traj_frac: torch.Tensor) -> torch.Tensor:
        """``[B, A, P, 9, 9]`` trajectory planes at the given burnup fractions.

        Rows with NO finite requested fraction (i.e. no trajectory label) are
        skipped entirely and returned as zeros: the label mask zeroes them in the
        loss anyway, and skipping keeps the cost proportional to the ~21% of the
        corpus that actually carries a trajectory instead of to the whole batch.
        """
        b_n, a_n = traj_frac.shape
        out = h.new_zeros((b_n, a_n, len(self._traj_channels), _QUARTER, _QUARTER))
        rows = torch.isfinite(traj_frac).any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return out
        h_s = h.index_select(0, rows)
        taps_s = [t.index_select(0, rows) for t in taps]
        cells_s = cells.index_select(0, rows)
        g_s = globals_.index_select(0, rows)
        frac_s = torch.nan_to_num(traj_frac.index_select(0, rows), nan=0.0)
        chans = list(self._traj_channels)
        for a in range(a_n):
            cond = torch.cat([g_s, frac_s[:, a:a + 1].to(g_s.dtype)], dim=1)
            h_a = self.traj_film(h_s, cond)
            taps_a = [self.traj_film(t, cond) for t in taps_s]
            quarter = self._map_quarter(taps_a, h_a, cells_s)
            out[rows, a] = quarter[:, chans].to(out.dtype)
        return out

    def forward(self, cells: torch.Tensor, globals_: torch.Tensor,
                traj_frac: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        fuel_mask = cells[:, 0:1]                        # channel 0 == fuel_mask
        h = self.stem(cells)
        # Only collected when the multiscale decoder is on; the default path
        # allocates nothing and runs the legacy loop unchanged.
        taps: list[torch.Tensor] = [h] if self._map_taps else []
        for b, block in enumerate(self.blocks):
            h = block(h)
            if str(b) in self.films:
                h = self.films[str(b)](h, globals_)
            if self._map_taps and b in self._map_taps:
                taps.append(h)

        # --- map head (gather SE cells -> 9x9 quarter) ---
        quarter = self._map_quarter(taps, h, cells)

        # --- pooled global features (multiplicity-weighted via mirror grid) ---
        denom = fuel_mask.sum(dim=(2, 3)).clamp_min(1.0)   # [B, 1]
        masked_mean = (h * fuel_mask).sum(dim=(2, 3)) / denom
        neg_inf = torch.finfo(h.dtype).min
        masked = h.masked_fill(fuel_mask == 0, neg_inf)
        masked_max = masked.amax(dim=(2, 3))
        pooled = torch.cat([masked_mean, masked_max, globals_], dim=1)

        feat = self.head_trunk(pooled)
        out = {
            "mu": self.mu_head(feat),
            "log_sigma": self.log_sigma_head(feat),
            "map": quarter,
            "conv_logit": self.conv_head(feat).squeeze(-1),
        }
        if self.has_quantiles:
            # [B, n_quantile_targets, n_quantiles] in z-space.  Emitted as an
            # ADDITIONAL key: every existing consumer reads out["mu"] / ["map"] /
            # ["log_sigma"] / ["conv_logit"] by name and is unaffected.
            out["quantiles"] = self.quantile_head(feat).view(
                -1, self.n_quantile_targets, self.n_quantiles)
        if self.has_axial:
            # [B, n_anchors, n_modes] standardised shape-basis coefficients.
            # Emitted as an ADDITIONAL key, so every existing consumer (which
            # reads out["mu"] / ["map"] / ["log_sigma"] / ["conv_logit"] by name)
            # is unaffected.
            out["axial"] = self.axial_head(feat).view(
                -1, self.n_axial_anchors, self.n_axial_modes)
        if self.has_traj and traj_frac is not None:
            # [B, A, P, 9, 9] EDIT5 planes at the requested burnup fractions.
            # Emitted ONLY when a caller passes ``traj_frac`` — the serving path
            # never does, so a traj-trained model serves exactly like any other.
            out["traj"] = self._traj_quarter(taps, h, cells, globals_, traj_frac)
        return out


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_member(config: PosValNetConfig | None = None, **overrides: Any) -> PosValNet:
    """Construct a member and assert it lands in the 1.0M–2.5M param band."""
    model = PosValNet(config, **overrides)
    n = count_parameters(model)
    if not (1_000_000 <= n <= 2_500_000):
        raise AssertionError(
            f"PosValNet has {n:,} params, outside the 1.0M-2.5M band"
        )
    return model


__all__ = [
    "PosValNet",
    "PosValNetConfig",
    "FiLM",
    "MultiScaleMapDecoder",
    "ResidualBlock",
    "build_member",
    "count_parameters",
    "N_TARGETS",
    "N_MAP_CHANNELS",
    "TRAJ_MAP_CHANNELS",
]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    net = PosValNet()
    n = count_parameters(net)
    print(f"PosValNet width={net.config.width} params={n:,}")
    cells = torch.randn(4, net.config.in_channels, 19, 19)
    cells[:, 0] = (torch.rand(4, 19, 19) > 0.3).float()      # fake fuel mask
    g = torch.randn(4, net.config.n_globals)
    out = net(cells, g)
    for k, v in out.items():
        print(f"  {k:10s} {tuple(v.shape)}")
    assert 1_000_000 <= n <= 2_500_000, "param count outside band"
    print("OK: param count in band")
