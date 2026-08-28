"""The move scorer.  Two arms, one pre-registered comparison between them.

**Arm ``cnn``** is the PosValNet trunk with the heads swapped: conv stem ->
residual blocks -> FiLM(cond) every two blocks -> masked mean+max pool -> MLP ->
two sigmoid logits ``(P_improve_fr, P_improve_flat)``.  :class:`~lpopt.model.net.FiLM`
and :class:`~lpopt.model.net.ResidualBlock` are IMPORTED, not re-implemented.

**Arm ``mlp``** drops the board tensor entirely and reads only the conditioning
vector (13 board globals + the move descriptors).  It is the control: the whole
convolutional apparatus has to earn its place against a scalar model that costs
a tenth of the parameters.  If the two tie, ship the MLP.

The s1e champion encoder was considered as a transfer initialisation and
rejected — see ``data/reports/policy_v1_prereg_20260815.md`` section 6.  Two
independent reasons: its trunk is width 224 / 8 blocks (10.4M params, 5-7x this
budget for 13k labelled rows) so no matched-geometry transfer exists, and it was
trained with direct supervision on the ``f_r`` and ``node_peak`` of the very
store records that are this corpus's parents and children, so an s1e-warmed
policy would carry indirect access to held-out outcome labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..model.net import FiLM, ResidualBlock

ARMS: tuple[str, ...] = ("cnn", "mlp")


@dataclass
class PolicyNetConfig:
    """Shape knobs.  ``in_channels`` / ``n_cond`` come from the dataset."""

    arm: str = "cnn"
    in_channels: int = 0
    n_cond: int = 0
    width: int = 112
    n_blocks: int = 6
    groups: int = 8
    film_every: int = 2
    head_hidden: int = 256
    mlp_hidden: int = 256
    mlp_layers: int = 3
    dropout: float = 0.1
    n_heads: int = 2

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; have {ARMS}")
        if self.n_cond <= 0:
            raise ValueError("n_cond must be set from the dataset")
        if self.arm == "cnn" and self.in_channels <= 0:
            raise ValueError("in_channels must be set from the dataset")


class PolicyNet(nn.Module):
    """Scores one (parent board, move, cell) triple into two improvement logits."""

    def __init__(self, config: PolicyNetConfig):
        super().__init__()
        self.config = cfg = config
        self.arm = cfg.arm

        if cfg.arm == "cnn":
            W = cfg.width
            self.stem = nn.Sequential(
                nn.Conv2d(cfg.in_channels, W, 3, padding=1),
                nn.GroupNorm(cfg.groups, W),
                nn.SiLU(),
            )
            self.blocks = nn.ModuleList(
                ResidualBlock(W, cfg.groups) for _ in range(cfg.n_blocks))
            self.films = nn.ModuleDict({
                str(b): FiLM(cfg.n_cond, W)
                for b in range(cfg.n_blocks) if (b + 1) % cfg.film_every == 0
            })
            head_in = 2 * W + cfg.n_cond
            self.head = nn.Sequential(
                nn.Linear(head_in, cfg.head_hidden), nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, cfg.head_hidden), nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, cfg.n_heads),
            )
        else:
            layers: list[nn.Module] = []
            d = cfg.n_cond
            for _ in range(cfg.mlp_layers):
                layers += [nn.Linear(d, cfg.mlp_hidden), nn.SiLU(),
                           nn.Dropout(cfg.dropout)]
                d = cfg.mlp_hidden
            layers.append(nn.Linear(d, cfg.n_heads))
            self.head = nn.Sequential(*layers)

    def forward(self, cells: torch.Tensor | None,
                cond: torch.Tensor) -> torch.Tensor:
        """``[B, n_heads]`` logits.  ``cells`` is ignored by the ``mlp`` arm."""
        if self.arm == "mlp":
            return self.head(cond)

        # channel 0 of the parent block is the v6b fuel mask; the 19x19 grid is
        # the mirror-expanded core, so a masked mean over it is already the
        # orbit-multiplicity-weighted mean (interior slot = 4 cells, axis = 2,
        # centre = 1) — the same argument PosValNet's pooling rests on.
        fuel_mask = cells[:, 0:1]
        h = self.stem(cells)
        for b, block in enumerate(self.blocks):
            h = block(h)
            if str(b) in self.films:
                h = self.films[str(b)](h, cond)

        denom = fuel_mask.sum(dim=(2, 3)).clamp_min(1.0)
        mean = (h * fuel_mask).sum(dim=(2, 3)) / denom
        masked = h.masked_fill(fuel_mask == 0, torch.finfo(h.dtype).min)
        peak = masked.amax(dim=(2, 3))
        return self.head(torch.cat([mean, peak, cond], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
