"""Campaign figures (matplotlib, Agg, 150 dpi) — plan sec. 7.

Every function is defensive: missing/empty data yields ``None`` (a skipped
figure) rather than an exception, so report generation never fails a run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DPI = 150
_ORIGIN_COLORS = {
    "elite": "#1f77b4", "guided": "#ff7f0e", "heuristic": "#2ca02c",
    "random": "#9467bd", "local": "#d62728", "control": "#7f7f7f",
}
# surrogate prediction column per dataset target.
_TARGET_COL = {"f_r": 0, "cbc_max": 1, "f_q": 2, "cyclen": 3, "ao_abs": 4}
_TARGET_LABEL = {"f_r": "F_r", "cbc_max": "CBC_max [ppm]", "f_q": "F_q",
                 "cyclen": "cyclen [EFPD]", "ao_abs": "|AO|"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# parity scatter (pred vs actual, origin-coloured)
# --------------------------------------------------------------------------- #
def parity_figure(points: Sequence[dict[str, Any]], path: Path,
                  targets=("f_r", "cbc_max", "f_q", "cyclen")) -> Path | None:
    """One parity panel per target; a point = one verified candidate.

    ``points`` items: ``{"origin", "pred": [7], "actual": {target: value}}``.
    """

    usable = [p for p in points if p.get("pred") is not None and p.get("actual")]
    if not usable:
        return None
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.3))
    if n == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        col = _TARGET_COL[target]
        xs, ys, cs = [], [], []
        for p in usable:
            actual = p["actual"].get(target)
            pred = p["pred"][col] if col < len(p["pred"]) else None
            if actual is None or pred is None or not math.isfinite(float(pred)):
                continue
            xs.append(float(actual))
            ys.append(float(pred))
            cs.append(_ORIGIN_COLORS.get(p.get("origin", ""), "#333333"))
        if xs:
            ax.scatter(xs, ys, c=cs, s=28, alpha=0.8, edgecolors="none")
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_title(_TARGET_LABEL[target])
        ax.set_xlabel("MASTER (actual)")
        ax.set_ylabel("surrogate (pred)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=o)
               for o, c in _ORIGIN_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Wave parity — surrogate vs verified", y=1.02)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# budget vs best objective (step curve)
# --------------------------------------------------------------------------- #
def budget_curve_figure(chains: Sequence[int], best_obj: Sequence[float],
                        path: Path, *, target_efpd: float = 625.0,
                        objective_label: str | None = None) -> Path | None:
    """``objective_label`` names the plotted scalar; ``None`` = target-cycle."""
    if not chains:
        return None
    label = objective_label or f"−|cyclen−{target_efpd:.0f}|"
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.step(chains, best_obj, where="post", color="#1f77b4", lw=1.8)
    ax.set_xlabel("MASTER evaluations (budget)")
    ax.set_ylabel(f"best feasible objective ({label})")
    ax.set_title("Campaign best-objective vs budget")
    ax.grid(alpha=0.3)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# GA-600 overlay
# --------------------------------------------------------------------------- #
def ga_overlay_figure(
    campaign_chains: Sequence[int], campaign_best: Sequence[float],
    ga_chains: Sequence[int], ga_best: Sequence[float],
    path: Path, *, target_efpd: float = 625.0,
    objective_label: str | None = None,
) -> Path | None:
    if not campaign_chains and not ga_chains:
        return None
    label = objective_label or f"−|cyclen−{target_efpd:.0f}|"
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if ga_chains:
        ax.step(ga_chains, ga_best, where="post", color="#7f7f7f", lw=1.6,
                label=f"GA-600 ({ga_chains[-1]} chains)")
    if campaign_chains:
        ax.step(campaign_chains, campaign_best, where="post", color="#d62728", lw=2.0,
                label=f"lpopt campaign ({campaign_chains[-1]} chains)")
    ax.set_xlabel("MASTER evaluations (#chains)")
    ax.set_ylabel(f"best feasible objective ({label})")
    ax.set_title("Guided search vs GA-600 baseline")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# p_feasible reliability
# --------------------------------------------------------------------------- #
def p_feas_reliability_figure(p_feas: Sequence[float], feasible: Sequence[bool],
                              path: Path, *, n_bins: int = 5) -> Path | None:
    pf = np.asarray(p_feas, dtype=float)
    fe = np.asarray(feasible, dtype=float)
    if pf.size < 2:
        return None
    edges = np.linspace(0.0, max(1e-6, float(pf.max())), n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pf >= lo) & (pf < hi) if hi < edges[-1] else (pf >= lo) & (pf <= hi)
        if mask.sum() == 0:
            continue
        xs.append(float(pf[mask].mean()))
        ys.append(float(fe[mask].mean()))
        ns.append(int(mask.sum()))
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="ideal")
    ax.scatter(xs, ys, s=[20 + 12 * n for n in ns], color="#1f77b4", alpha=0.8)
    ax.plot(xs, ys, color="#1f77b4", lw=1.2)
    ax.set_xlabel("predicted p_feasible (bin mean)")
    ax.set_ylabel("empirical feasible fraction")
    ax.set_title("p_feasible reliability (selected candidates)")
    ax.set_xlim(-0.02, max(0.2, max(xs) * 1.1))
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# quarter-core LP figure (17x17 batch/age map)
# --------------------------------------------------------------------------- #
def quarter_core_figure(pattern: Any, pair: str, path: Path,
                        *, title: str = "") -> Path | None:
    """Full-core 17x17 batch/age map from a mirror-69 :class:`Pattern`."""

    from ..vendor.masterrl.domain import SLOTS

    try:
        batches = pair.split("_")
        batch_a = batches[0]
    except Exception:  # noqa: BLE001
        batch_a = None
    grid = np.full((17, 17), np.nan)
    labels = np.empty((17, 17), dtype=object)
    centre = 8
    for slot, item in zip(SLOTS, pattern.items):
        if item.is_fresh:
            if slot.orbit_class == "center":
                code, txt = 3.0, item.batch
            elif item.batch == batch_a:
                code, txt = 1.0, item.batch
            else:
                code, txt = 2.0, item.batch
        else:
            code, txt = 0.0, "s"
        for dr in ({0} if slot.row == 0 else {slot.row, -slot.row}):
            for dc in ({0} if slot.col == 0 else {slot.col, -slot.col}):
                grid[centre + dr, centre + dc] = code
                labels[centre + dr, centre + dc] = txt
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    cmap = matplotlib.colors.ListedColormap(["#bdbdbd", "#4e79a7", "#f28e2b", "#59a14f"])
    masked = np.ma.masked_invalid(grid)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=3)
    for r in range(17):
        for c in range(17):
            if labels[r, c] is not None and not (isinstance(grid[r, c], float) and math.isnan(grid[r, c])):
                ax.text(c, r, str(labels[r, c]), ha="center", va="center", fontsize=5.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"Best verified LP — {pair} (fresh=batch, s=shuffled)")
    handles = [
        plt.Line2D([], [], marker="s", ls="", color="#4e79a7", label=f"fresh {batches[0]}"),
        plt.Line2D([], [], marker="s", ls="", color="#f28e2b", label=f"fresh {batches[-1]}"),
        plt.Line2D([], [], marker="s", ls="", color="#59a14f", label="centre"),
        plt.Line2D([], [], marker="s", ls="", color="#bdbdbd", label="shuffled/burned"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=4, fontsize=7, frameon=False)
    return _save(fig, path)


__all__ = [
    "budget_curve_figure", "ga_overlay_figure", "parity_figure",
    "p_feas_reliability_figure", "quarter_core_figure",
]
