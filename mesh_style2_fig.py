"""Style2 mesh renderer — the format the user pinned to a reference figure drawn
from ``feasible_database.xlsx`` (see ``data/reports/mesh_style_spec_20260817.md``,
which this module implements and must stay in sync with).

Additive: this does NOT replace ``mesh_v3_fig.py`` / ``scoping_mesh_fig.py``'s
existing renders — it writes a second, differently-named PNG next to them.

    python mesh_style2_fig.py --domain v3
    python mesh_style2_fig.py --domain v2
    python mesh_style2_fig.py --domain both
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
STORE = BASE / "data" / "store" / "records.parquet"
P_TH_MW, N_FA = 3983.0, 241
M_HM_GA80 = 102.031495          # tU — the ga80/paramA mesh's approx constant

INK, MUTED = "#1A1A1A", "#8A8A8A"
ISO_E_C = "#A5ABB3"

# --------------------------------------------------------------------------- #
# palette — mesh_style_spec_20260817.md is the source of truth; keep in sync
# --------------------------------------------------------------------------- #
FEED_C = {101: "#F781BF", 105: "#A65628", 109: "#984EA3", 113: "#4DAF4A",
          117: "#E41A1C", 121: "#377EB8", 125: "#FF7F00", 129: "#1B9E77"}

ENRICH_C = {
    5.0: "#C8843C", 5.1: "#BC763A", 5.2: "#AD6836", 5.3: "#9A5A30",
    5.4: "#82492A", 5.5: "#1A1A1A", 5.6: "#4B1F1F", 5.7: "#3D2A4B",
    5.8: "#1F3864", 5.9: "#1F4B3D", 6.0: "#4B3D1F", 6.1: "#2F2F2F",
    6.2: "#5C1F3D", 6.3: "#1F2F5C", 6.4: "#3D1F5C", 6.5: "#000000",
}

RING_ORDER = ["green", "yellow", "orange", "red", "미산출"]
RING_C = {"green": "#2E9E4F", "yellow": "#E8B923", "orange": "#E07B39",
          "red": "#C0392B", "미산출": "#9AA5B5"}
RING_LABEL = {"green": "< 62 GWd/tU", "yellow": "62 – 68", "orange": "68 – 75",
              "red": "≥ 75", "미산출": "예측값 없음"}

#: cell -> (pin_bu, source note).  Cells the campaign reports pinned a real
#: predicted/measured max-pin-BU to but that ``mesh_nodes.csv``'s own
#: gate-free node representative does not carry (NaN there).  Approximation:
#: the campaign's frontier core, not necessarily the node's own candidate —
#: flagged in the caption per the spec's honesty-over-completeness rule.
PIN_OVERRIDE = {
    (5.0, 109): (83.16, "E1_E2/f109 F_r-승자 예측 (fpcamp_E1E2_f109_results)"),
    (5.7, 109): (81.13, "hgd569/f109 프런티어 예측 (fpcamp_HGD569_f125_results §8)"),
    (5.7, 125): (76.96, "hgd569/f125 4/5-게이트 프런티어 예측 (fpcamp_HGD569_f125_results)"),
}


def pin_class(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "미산출"
    if v < 62:
        return "green"
    if v < 68:
        return "yellow"
    if v < 75:
        return "orange"
    return "red"


def nearest_e_color(e: float) -> str:
    key = min(ENRICH_C, key=lambda k: abs(k - e))
    return ENRICH_C[key]


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def _chrome(ax, xlab, ylab):
    ax.grid(True, color="#F4F6F8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CFD3D8")
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors="#3C3C3C", labelsize=11, length=4, width=0.9)
    ax.set_xlabel(xlab, fontsize=13, color=INK, labelpad=9)
    ax.set_ylabel(ylab, fontsize=13, color=INK, labelpad=9)


def background_cloud(feed_lo: int, feed_hi: int, e_lo: float, e_hi: float) -> pd.DataFrame:
    """Converged store rows, mass-balance B_d, inside a generous domain pad —
    axis limits (set later from the node grid) do the final clipping."""

    s = pd.read_parquet(STORE, columns=["e_core", "feed", "cyclen", "valid",
                                        "converged", "library_id"])
    s = s[(s.valid == True) & (s.converged == True)                # noqa: E712
          & s.library_id.isin(["ga80", "paramA"])
          & s.feed.between(feed_lo - 4, feed_hi + 4)
          & s.e_core.between(e_lo - 0.3, e_hi + 0.3)]
    s = s.copy()
    s["B_d"] = (s.cyclen * P_TH_MW / M_HM_GA80 / 1000.0) * N_FA / s.feed
    s["color"] = [nearest_e_color(e) for e in s.e_core]
    return s


def render(nodes: pd.DataFrame, out_path: Path, title: str, subtitle: str,
           override: dict, figsize=(13.2, 9.8)) -> tuple[Path, dict]:
    _setup()
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    HALO = [pe.withStroke(linewidth=2.6, foreground="white", alpha=0.88)]

    d = nodes.copy()
    d["x"] = d.max_pred_cyclen_any
    d["y"] = (d.x * P_TH_MW / d.M_HM_tU / 1000.0) * N_FA / d.feed
    d["e_target"] = d.e_target.round(1)
    d["feed"] = d.feed.astype(int)

    pin_val, pin_src = {}, {}
    for r in d.itertuples():
        key = (r.e_target, r.feed)
        v = r.pred_max_pin_bu
        if np.isfinite(v):
            pin_val[key] = float(v)
            pin_src[key] = "mesh_nodes.csv (셀 대표 예측)"
        elif key in override:
            pin_val[key], pin_src[key] = override[key]
        else:
            pin_val[key] = np.nan
            pin_src[key] = ""
    d["pin_bu"] = [pin_val[(r.e_target, r.feed)] for r in d.itertuples()]
    d["pin_class"] = [pin_class(v) for v in d.pin_bu]
    d["fill"] = [nearest_e_color(e) for e in d.e_target]

    feeds = sorted(d.feed.unique())
    targets = sorted(d.e_target.unique())
    grid = {(r.e_target, r.feed): r for r in d.itertuples()}

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # -- 0. background cloud -------------------------------------------------- #
    bg = background_cloud(min(feeds), max(feeds), min(targets), max(targets))
    ax.scatter(bg.cyclen, bg.B_d, s=8, c=bg.color, alpha=0.10, linewidths=0,
              zorder=1)

    # -- 1. iso-enrichment connectors (thin gray) ----------------------------- #
    for e in targets:
        pts = [grid[(e, f)] for f in feeds if (e, f) in grid]
        if len(pts) < 2:
            continue
        ax.plot([p.x for p in pts], [p.y for p in pts], "-", color=ISO_E_C,
                lw=1.0, zorder=3, solid_capstyle="round")

    # -- 2. iso-feed lines (colored, ~2pt) ------------------------------------ #
    for f in feeds:
        pts = [grid[(e, f)] for e in targets if (e, f) in grid]
        if len(pts) < 2:
            continue
        c = FEED_C.get(f, "#555555")
        ax.plot([p.x for p in pts], [p.y for p in pts], "-", color=c, lw=2.0,
                zorder=4, solid_capstyle="round")

    # -- 3. nodes: fill = enrichment, ring = pin-BU class --------------------- #
    for cls in RING_ORDER:
        m = d.pin_class == cls
        if not m.any():
            continue
        sub = d[m]
        ax.scatter(sub.x, sub.y, s=105, c=list(sub.fill), edgecolors=RING_C[cls],
                  linewidths=2.5, zorder=8)

    # -- 4. feed labels at line starts (left/min-x end), staggered ------------ #
    # mirror of §5 below: sort by y, enforce a minimum gap, draw a hairline back
    # to the node whenever a label had to move, so low-e feed starts (which
    # cluster tightly in B_d) never print on top of each other or the node ring.
    span_y = (d.y.max() - d.y.min()) or 1.0
    gap = 0.060 * span_y
    # a single far-left column, clear of EVERY node (not just the labelled
    # feed's own start) — anchoring each label to its own start.x - dx let
    # a label drift into a neighbouring feed's early nodes wherever the fan
    # of lines converges near e=5.0, so all labels share one x instead.
    xa = d.x.min() - 0.115 * (d.x.max() - d.x.min())
    starts = []
    for f in feeds:
        pts = [grid[(e, f)] for e in targets if (e, f) in grid]
        if not pts:
            continue
        starts.append((f, min(pts, key=lambda p: p.x)))
    ysL: list[float] = []
    orderedL = sorted(starts, key=lambda t: t[1].y)
    for f, p0 in orderedL:
        y = p0.y if not ysL else max(p0.y, ysL[-1] + gap)
        ysL.append(y)
    for (f, p0), y in zip(orderedL, ysL):
        c = FEED_C.get(f, "#555555")
        ax.plot([p0.x, xa], [p0.y, y], "-", color="#C9CDD3", lw=0.8,
                zorder=9, clip_on=False)
        ax.annotate(f"feed {f}", (xa, y), textcoords="offset points",
                    xytext=(-4, 0), ha="right", va="center", color=c,
                    fontsize=10.5, fontweight="bold", zorder=11, clip_on=False,
                    path_effects=HALO)

    # -- 5. enrichment labels at line ends (right/max-x end), staggered ------ #
    ends = []
    for e in targets:
        pts = [grid[(e, f)] for f in feeds if (e, f) in grid]
        if not pts:
            continue
        p1 = max(pts, key=lambda p: p.x)
        ends.append((e, p1))
    dx = 0.022 * (d.x.max() - d.x.min())
    ys: list[float] = []
    ordered = sorted(ends, key=lambda t: t[1].y)
    for e, p1 in ordered:
        y = p1.y if not ys else max(p1.y, ys[-1] + gap)
        ys.append(y)
    for (e, p1), y in zip(ordered, ys):
        c = nearest_e_color(e)
        if abs(y - p1.y) > 1e-9:
            ax.plot([p1.x, p1.x + dx], [p1.y, y], "-", color="#C9CDD3", lw=0.8,
                    zorder=9, clip_on=False)
        ax.annotate(f"{e:.1f} w/o", (p1.x + dx, y), textcoords="offset points",
                    xytext=(5, 0), ha="left", va="center", color=c,
                    fontsize=10.5, fontweight="bold", zorder=11, clip_on=False,
                    path_effects=HALO)

    _chrome(ax, "주기길이 (EFPD)", "방출연소도 (GWd/tU, 질량수지)")
    pad_x = (d.x.max() - d.x.min())
    ax.set_xlim(d.x.min() - pad_x * 0.20, d.x.max() + pad_x * 0.30)
    ax.set_ylim(d.y.min() - 3.4, d.y.max() + 3.4)

    # -- 6. two-column legend, no box ------------------------------------------ #
    fig.tight_layout(rect=(0, 0.30, 1, 0.905))
    used_e = [e for e in sorted(ENRICH_C) if e in set(targets)]
    lx0, ly0, lstep = 0.045, 0.270, 0.0225
    ncol_e = 6
    ax.text(lx0, ly0 + lstep, "농축도 (노드 채움)", transform=fig.transFigure,
            ha="left", va="top", fontsize=10.5, color=INK, fontweight="bold")
    for i, e in enumerate(used_e):
        col, row = divmod(i, ncol_e)
        x = lx0 + col * 0.078
        y = ly0 - (row + 1) * lstep
        ax.plot([x], [y], "o", ms=9, color=ENRICH_C[e], mec="white", mew=0.6,
                transform=fig.transFigure, clip_on=False)
        ax.text(x + 0.011, y, f"{e:.1f}", transform=fig.transFigure, ha="left",
                va="center", fontsize=9.3, color=INK)
    rx0 = lx0 + ncol_e * 0.078 + 0.028
    ax.text(rx0, ly0 + lstep, "봉최대(핀) 연소도 등급 (테두리)", transform=fig.transFigure,
            ha="left", va="top", fontsize=10.5, color=INK, fontweight="bold")
    for i, cls in enumerate(RING_ORDER):
        y = ly0 - (i + 1) * lstep
        ax.plot([rx0], [y], "o", ms=9, color="white", mec=RING_C[cls], mew=2.2,
                transform=fig.transFigure, clip_on=False)
        ax.text(rx0 + 0.014, y, f"{cls}  ({RING_LABEL[cls]})",
                transform=fig.transFigure, ha="left", va="center", fontsize=9.3,
                color=INK)
    fig.text(0.045, 0.978, title, fontsize=15.5, color=INK, ha="left", va="top")
    fig.text(0.045, 0.943, subtitle, fontsize=11.2, color="#3C3C3C", ha="left",
            va="top")
    n_real = int((d.pin_class != "미산출").sum())
    n_over = sum(1 for k in override if k in grid)
    over_txt = "; ".join(f"e{e:.1f}/f{f} {pin_src.get((e, f), '')}"
                         for (e, f) in override if (e, f) in grid)
    lines = [
        "양식: data/reports/mesh_style_spec_20260817.md (이후 표준 양식) · "
        "style2 추가본 — 기존 그림을 대체하지 않는다.",
        f"핀-연소도 등급 유효 노드 {n_real}/{len(d)} — 나머지는 예측값이 없어 "
        "미산출(회색)로 둔다(완전성보다 정직성).",
    ]
    if over_txt:
        lines += textwrap.wrap(f"캠페인 값을 노드에 근사시킨 {n_over}곳: " + over_txt,
                               width=108)
    lines.append("배경 점 = 저장소 수렴 전 행 (cyclen, 질량수지 B_d), 농축도색 · "
                 "alpha 0.10 — 노드와 같은 좌표계.")
    fig.text(0.045, 0.012, "\n".join(lines), fontsize=9.4, color=MUTED,
             ha="left", va="bottom", linespacing=1.65)

    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)

    coverage = dict(n_nodes=len(d), n_real_pred=int((d.pin_class != "미산출").sum())
                     - n_over, n_override=n_over,
                     n_missing=int((d.pin_class == "미산출").sum()),
                     by_class=d.pin_class.value_counts().to_dict())
    return out_path, coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["v3", "v2", "both"], default="both")
    args = ap.parse_args()

    results = []
    if args.domain in ("v3", "both"):
        out_dir = BASE / "data" / "reports" / "mesh_v3_20260817"
        d = pd.read_csv(out_dir / "mesh_nodes.csv")
        path, cov = render(
            d, out_dir / "mesh_v3_map_style2.png",
            "APR1400 LEU+ 고농축 확장 재장전 설계지도 v3 (e_core 5.0-6.4 x feed 109-129) "
            "— 표준 양식(style2)",
            "90개 격자점 · s1g 예측 · 저장소 74,477행(수렴 65,922) · "
            "채움=농축도, 테두리=예측 봉최대 연소도 등급, 옅은 점=배경 모집단",
            PIN_OVERRIDE)
        print("v3:", path, cov)
        results.append(("v3", path, cov))

    if args.domain in ("v2", "both"):
        out_dir = BASE / "data" / "reports" / "scoping_mesh_20260815"
        d = pd.read_csv(out_dir / "mesh_nodes.csv")
        v2_override = {k: v for k, v in PIN_OVERRIDE.items() if k[1] <= 121}
        path, cov = render(
            d, out_dir / "scoping_mesh_style2.png",
            "APR1400 LEU+ 재장전 설계지도 v2 (e_core 5.0-6.0 x feed 105-121) "
            "— 표준 양식(style2)",
            "55개 격자점 · s1g 예측 · 저장소 74,477행(수렴 65,922) · "
            "채움=농축도, 테두리=예측 봉최대 연소도 등급, 옅은 점=배경 모집단",
            v2_override, figsize=(12.4, 9.4))
        print("v2:", path, cov)
        results.append(("v2", path, cov))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
