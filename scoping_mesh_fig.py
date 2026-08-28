"""Renderer for the (e_core x feed) scoping mesh PNG (see scoping_mesh.py).

Rev 2026-08-16c — rebuilt as a CLASSIC RELOAD-DESIGN MAP (Driscoll-style).

The 2026-08-15b version drew only the gate-passing cells, so the picture was a
handful of disconnected dots and the reader could not see the design space at
all.  A reload map is a COMPLETE curvilinear quadrilateral mesh: every (e_core,
feed) cell is a node, both line families run through every node, and there are
no gaps.  Feasibility is an OVERLAY on that map, not the reason a node is
missing — limits close part of a design space, they do not delete it.

  * node        = the gate-free representative: the max predicted cyclen over the
                  in-band candidate pool, NO F_r/F_q/CBC/AO filter
                  (``max_pred_cyclen_any``).  B_d is the usual mass balance.
  * family 1    = 11 iso-enrichment curves (5.0 … 6.0), each through 5 feeds
  * family 2    = 5 iso-feed curves (105 … 121), each through 11 enrichments
  * overlay     = cells with a gate-passing candidate: filled node, heavier mesh
                  segments, and a soft hull labelled 안전인자 만족 영역
  * labels live at the LINE ENDS (feed along the top edge, enrichment along the
    right edge) with the family names as plain text — there is no legend box.

The per-cell Pareto fronts (multi-objective view) are deliberately NOT in this
figure; they live in mesh_pareto.csv and in mesh_vs_database.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Sequential single-hue blue ramp, 11 steps.  Retained because mesh_vs_db.py
#: imports it for the DB segment colours; the reload map itself is monochrome.
BLUE = {"5.0": "#AECEEA", "5.1": "#9BC2E4", "5.2": "#88B6DE", "5.3": "#75AAD7",
        "5.4": "#639DCE", "5.5": "#5290C4", "5.6": "#4381B6", "5.7": "#3771A4",
        "5.8": "#2E608F", "5.9": "#26507A", "6.0": "#1F3864"}
GRAY = "#8C8C8C"
INK = "#1A1A1A"
MUTED = "#8A8A8A"

#: display label + promotion date for each champion ensemble the mesh has been run
#: with (the figure has to say WHICH model drew it — the 2026-08-15 s1e run is
#: preserved as ``mesh_nodes_s1e.csv`` / ``scoping_mesh_s1e.png``).
MODEL_LABEL = {"s1g": "s1g (8대 챔피언), 2026-08-16",
               "s1f": "s1f (7대 챔피언), 2026-08-16",
               "s1e": "s1e (6대 챔피언), 2026-08-15"}

MESH_C = "#A5ABB3"          # the design map itself
MESH_HI = "#2E608F"         # feasible sub-region
HULL_C = "#DCE8F4"
P_TH_MW, N_FA = 3983.0, 241


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain).  Used only to shade the feasible island."""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return np.asarray(p)

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (q[1] - y1) - (y2 - y1) * (q[0] - x1) > 0:
                    break
                out.pop()
            out.append(q)
        return out

    return np.asarray(half(p)[:-1] + half(reversed(p))[:-1])


def render(df: pd.DataFrame, path: Path, model: str = "s1f") -> Path:
    _setup()
    import matplotlib.pyplot as plt

    d = df.copy()
    # gate-free node.  Recomputed here when the column is absent so an older
    # mesh_nodes.csv still renders.
    if "ceil_cyclen" not in d.columns:
        d["ceil_cyclen"] = d.max_pred_cyclen_any
        d["ceil_B_d"] = (d.ceil_cyclen * P_TH_MW / d.M_HM_tU / 1000.0) * N_FA / d.feed
    d["ok"] = d.n_feasible > 0
    feeds = sorted(d.feed.unique())
    targets = sorted(d.e_target.unique())
    grid = {(round(r.e_target, 2), int(r.feed)): r for r in d.itertuples()}

    fig, ax = plt.subplots(figsize=(11.5, 8.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # -- 0. feasible island, shaded under everything ------------------------- #
    fp = d[d.ok]
    if len(fp) >= 3:
        h = _hull(np.column_stack([fp.ceil_cyclen, fp.ceil_B_d]))
        c = h.mean(axis=0)
        h = c + (h - c) * 1.22               # breathe, so nodes sit inside
        ax.fill(h[:, 0], h[:, 1], color=HULL_C, zorder=1, lw=0)

    # -- 1. the complete mesh: both families through every node -------------- #
    def draw(seq_keys):
        """One mesh line; segments with BOTH ends feasible are drawn heavy."""
        pts = [grid[k] for k in seq_keys]
        x = [p.ceil_cyclen for p in pts]
        y = [p.ceil_B_d for p in pts]
        ax.plot(x, y, "-", color=MESH_C, lw=1.15, zorder=3,
                solid_capstyle="round")
        for a, b in zip(range(len(pts) - 1), range(1, len(pts))):
            if pts[a].ok and pts[b].ok:
                ax.plot(x[a:b + 1], y[a:b + 1], "-", color=MESH_HI, lw=2.6,
                        zorder=5, solid_capstyle="round")

    for e in targets:                        # iso-enrichment (across feeds)
        draw([(round(e, 2), f) for f in feeds])
    for f in feeds:                          # iso-feed (across enrichments)
        draw([(round(e, 2), f) for e in targets])

    # -- 2. nodes: feasible filled, the rest a faint tick --------------------- #
    ax.plot(d.loc[~d.ok, "ceil_cyclen"], d.loc[~d.ok, "ceil_B_d"], "o", ms=3.4,
            color="white", mec=MESH_C, mew=1.0, ls="none", zorder=6)
    ax.plot(fp.ceil_cyclen, fp.ceil_B_d, "o", ms=9.5, color=MESH_HI, mec="white",
            mew=1.5, ls="none", zorder=8)

    # -- 3. edge labels at the LINE ENDS (no legend box) ---------------------- #
    top = [grid[(round(targets[-1], 2), f)] for f in feeds]        # e = 6.0 rung
    for p in top:
        ax.annotate(f"{int(p.feed)}", (p.ceil_cyclen, p.ceil_B_d),
                    textcoords="offset points", xytext=(1, 13), ha="center",
                    color=INK, fontsize=12.5, zorder=10)
    ax.annotate("신연료 장전수", (top[0].ceil_cyclen, top[0].ceil_B_d),
                textcoords="offset points", xytext=(-16, 30), ha="right",
                color=INK, fontsize=13.5, fontweight="bold", zorder=10)

    right = [grid[(round(e, 2), feeds[-1])] for e in targets]      # f121 rung
    for p in right:
        ax.annotate(f"{p.e_target:.1f}", (p.ceil_cyclen, p.ceil_B_d),
                    textcoords="offset points", xytext=(13, -1), ha="left",
                    va="center", color=INK, fontsize=12.5, zorder=10)
    # Both family names go in the wedge below/right of the f121 rung, which is
    # the only large empty area on this plane — anchoring the enrichment name to
    # its own line end put it on top of the "117" feed label.
    ax.text(0.885, 0.455, "노심 평균 농축도, w/o", transform=ax.transAxes,
            ha="center", va="center", color=INK, fontsize=13.5,
            fontweight="bold", zorder=10)

    # -- 4. name the overlay in place, again instead of a legend -------------- #
    if len(fp):
        lo = fp.loc[fp.ceil_B_d.idxmin()]
        ax.annotate("안전인자 만족 영역\n(F_r ≤ 1.55, 예측)",
                    xy=(lo.ceil_cyclen, lo.ceil_B_d), xycoords="data",
                    xytext=(0.60, 0.085), textcoords="axes fraction",
                    ha="center", va="center", color=MESH_HI, fontsize=12.5,
                    fontweight="bold", linespacing=1.5, zorder=10,
                    arrowprops=dict(arrowstyle="-", color=MESH_HI, lw=1.2,
                                    shrinkA=6, shrinkB=8))

    # -- chrome --------------------------------------------------------------- #
    ax.grid(True, color="#F4F6F8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CFD3D8")
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors="#3C3C3C", labelsize=12, length=4, width=0.9)
    ax.set_xlabel("예측 주기길이 (EFPD)", fontsize=14, color=INK, labelpad=10)
    ax.set_ylabel("평균 방출연소도 (GWd/tU, 질량수지)", fontsize=14, color=INK,
                  labelpad=10)
    ax.set_xlim(d.ceil_cyclen.min() - 22, d.ceil_cyclen.max() + 46)
    ax.set_ylim(d.ceil_B_d.min() - 2.4, d.ceil_B_d.max() + 2.1)

    fig.tight_layout(rect=(0, 0.085, 1, 0.905))
    n_ok, n_all = int(d.ok.sum()), len(d)
    tag = MODEL_LABEL.get(model, model)
    fig.text(0.055, 0.975, f"APR1400 LEU+ 재장전 설계지도 (e_core × feed) — {tag} 모델 예측",
             fontsize=16, color=INK, ha="left", va="top")
    fig.text(0.055, 0.933,
             f"{len(targets)}×{len(feeds)} = {n_all}개 격자점 전부 표시 · "
             f"굵은 파란 격자 = 안전인자 만족 {n_ok}셀 · 격자점은 안전인자 미적용 대표점",
             fontsize=12, color="#3C3C3C", ha="left", va="top")
    fig.text(0.055, 0.022,
             "주기길이 목표 없음 — 격자점은 각 셀에서 예측 주기길이가 가장 긴 후보이고, "
             "안전인자는 지도를 지우지 않고 그 위에 겹쳐 읽는다.\n"
             "안전인자 만족 셀 안에서는 단일 최적 대신 (주기길이, F_r) Pareto 대표점 집합을 "
             "쓴다 — mesh_pareto.csv · mesh_vs_database.png · README §10.\n"
             "모델 예측 전용 — MASTER 미검증. 방출연소도는 평형 질량수지 "
             "B_d = cyclen·P_th/M_HM·241/feed.",
             fontsize=10.5, color=MUTED, ha="left", va="bottom", linespacing=1.75)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data/reports/scoping_mesh_20260815"
    render(pd.read_csv(out / "mesh_nodes.csv"), out / "scoping_mesh.png", model="s1f")
