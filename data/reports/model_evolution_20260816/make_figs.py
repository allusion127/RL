"""Figures for the champion-lineage accuracy evolution (2026-08-16).

Reads evolution_metrics.csv (written by serve_evolution.py) and renders
  evolution_spearman.png   headline: iso-group Spearman rho per target x generation
  evolution_mae.png        MAE per target x generation (small multiples)
  evolution_frontier.png   F_r iso-rho over the search frontier it guided

Palette: dataviz reference instance, categorical slots 1-6 in fixed order.
Validated (adjacent pairlist, light surface #fcfcfb): worst normal-vision
dE 19.6 (>=15), worst CVD dE 9.1 (>=8).  Slots 3/4/5 sit below 3:1 contrast on
the light surface -- the documented relief is applied: every series carries a
direct label, so identity is never colour-alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ tokens
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

TARGETS = [("f_r", "F_r", SLOT[0]), ("node_peak", "node_peak", SLOT[1]),
           ("map_cov", "map_cov", SLOT[2]), ("cyclen", "cyclen", SLOT[3]),
           ("cbc_max", "CBC_max", SLOT[4]), ("f_q", "F_q", SLOT[5])]
NOISE = 0.018        # single-seed sd, scaling_results_20260815.md L181-185

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
})


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def end_label(ax, x, y, text, color, dy=0.0, fs=9.5, weight="normal"):
    """Direct label: a colour chip carries identity, the text stays in ink."""
    ax.plot([x + 0.055], [y + dy], marker="s", ms=5.5, color=color,
            clip_on=False, zorder=6, mec=SURFACE, mew=0.8)
    ax.text(x + 0.115, y + dy, text, va="center", ha="left", fontsize=fs,
            color=INK, weight=weight, clip_on=False, zorder=6)


def gen_ticks(df):
    short = {"20260729_054749": "20260729", "20260810_bu_T": "bu_T",
             "split_S1b": "S1b", "s1c": "s1c", "s1d": "s1d",
             "s1e": "s1e", "s1f": "s1f"}
    return [f"{short.get(g, g)}\n{d[5:].replace('-', '/')}"
            for g, d in zip(df["gen"], df["date"])]


# ============================================================ FIG 1 headline
def fig_spearman(df):
    fig = plt.figure(figsize=(11.8, 9.2), dpi=170)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.50,
                          left=0.078, right=0.775, top=0.855, bottom=0.115)
    ax = fig.add_subplot(gs[0])
    style(ax)
    x = np.arange(len(df))
    ax.set_ylim(0.748, 0.976)
    ax.set_xlim(-0.35, len(df) - 0.62)

    ends = []
    for key, lab, col in TARGETS:
        y = df[f"rho_iso_{key}"].to_numpy(float)
        ax.plot(x, y, color=col, lw=2.0, marker="o", ms=5.0, zorder=4,
                mec=SURFACE, mew=1.0, solid_capstyle="round")
        ends.append([y[-1], f"{lab}  {y[-1]:.3f}", col])

    # Direct labels at the line ends.  These carry identity (five of the six
    # series sit inside 0.016 of each other, so colour alone would not do it)
    # and they are also the documented relief for the three slots below 3:1.
    ends.sort(key=lambda r: r[0])
    minsep = 0.0088
    pos = [e[0] for e in ends]
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < minsep:
            pos[i] = pos[i - 1] + minsep
    shift = max(0.0, pos[-1] - 0.972)
    pos = [p - shift for p in pos]
    for (yv, txt, col), py in zip(ends, pos):
        ax.plot([x[-1], x[-1] + 0.045], [yv, py], color=col, lw=0.9,
                alpha=0.6, clip_on=False, zorder=5)
        end_label(ax, x[-1], py, txt, col)

    # ---- noise floor, drawn ONCE as a shaded height gauge -------------------
    # A per-series +/-0.018 corridor was tried and rejected: five of the six
    # series' corridors coincide and the translucent fills stack into one solid
    # block that reads as a single band.  A constant tolerance is shown once.
    gx0, gx1 = 3.30, 4.10
    gc = 0.818
    ax.fill_between([gx0, gx1], gc - NOISE, gc + NOISE, color="#0b0b0b",
                    alpha=0.10, lw=0, zorder=2)
    for yy in (gc - NOISE, gc + NOISE):
        ax.plot([gx0, gx1], [yy, yy], color=MUTED, lw=0.9, zorder=3)
    ax.annotate("", xy=(gx0 - 0.10, gc - NOISE), xytext=(gx0 - 0.10, gc + NOISE),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9,
                                shrinkA=0, shrinkB=0))
    ax.text(gx1 + 0.10, gc, f"시드 노이즈 바닥  ±{NOISE:g}\n"
            "이 높이보다 작은 오르내림은\n잡음과 구분되지 않는다",
            fontsize=9, color=INK2, va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(gen_ticks(df), fontsize=9)
    ax.set_ylabel("iso군 내 Spearman ρ  (iso군 중앙값)", fontsize=10.5)
    ax.set_title(f"세대별 예측 순위정확도 — {len(df)}세대 전부를 같은 고정 평가면에 다시 서빙",
                 fontsize=14.5, color=INK, weight="bold", loc="left", pad=44)
    ax.text(0, 1.072, "동결 검증면 3,207행 · 36셀 · iso군 118개 (campaign×pair×feed, "
            f"맵 지표는 100개) · {len(df)}세대 모두 학습에서 제외됨을 확인",
            transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.text(0, 1.023, "읽히는 것: 잡음 바닥을 넘는 개선은 1세대 → 2세대"
            f"(cond_schema v6 → v6b) 한 번뿐이다. 이후 {len(df) - 2}개 세대에서 바닥을 "
            "넘는 개선은 여섯 지표 어디에도 없다 (최대 +0.002).",
            transform=ax.transAxes, fontsize=9.5, color=INK, weight="bold")

    # ---------------- lower panel: RECORDED history, not recomputed ---------
    ax2 = fig.add_subplot(gs[1])
    style(ax2)
    rec_a = [(2, 0.077), (3, 0.513)]        # surface A: 44 rows / 1 cell
    rec_b = [(4, 0.834), (5, 0.788)]        # surface B: 144 rows / 5 cells
    for seg in (rec_a, rec_b):
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        ax2.plot(xs, ys, color=MUTED, lw=1.5, ls=(0, (4, 2)), zorder=3)
        ax2.plot(xs, ys, marker="o", ms=6.5, ls="none", mfc=SURFACE,
                 mec=INK2, mew=1.6, zorder=4)
        for xx, yy in seg:
            ax2.text(xx, yy + 0.10, f"{yy:.3f}", ha="center", fontsize=9,
                     color=INK)
    ax2.plot([5], [0.649], marker="D", ms=5.5, mfc=SURFACE, mec=MUTED,
             mew=1.3, ls="none", zorder=4)
    ax2.annotate("같은 144행에 대한 scaling §4의 다른 기록값 0.649",
                 xy=(5.06, 0.649), xytext=(3.32, 0.30), fontsize=8.4,
                 color=INK2, va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                 shrinkA=2, shrinkB=3))
    ax2.text(2.5, 0.72, "평가면 A\n44행 · 1셀", ha="center", fontsize=8.6,
             color=INK2)
    ax2.text(4.5, 1.10, "평가면 B\n144행 · 5셀", ha="center", fontsize=8.6,
             color=INK2)
    ax2.text(6.0, 0.62, "s1f\n기록 없음", ha="center", fontsize=8.6, color=MUTED)
    ax2.set_xticks(np.arange(len(df)))
    ax2.set_xticklabels(gen_ticks(df), fontsize=9)
    ax2.set_xlim(-0.35, len(df) - 0.62)
    ax2.set_ylim(-0.10, 1.42)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_ylabel("T6_T4 내 ρ (기록값)", fontsize=10)
    ax2.set_title("참고 — 문서에 기록된 T6_T4 2차 판독값 (재계산 아님 · 평가셋이 서로 다름)",
                  fontsize=10.5, color=INK, loc="left", pad=12)
    ax2.text(0, -0.46, "두 구간을 잇지 않은 이유: A와 B는 행 수·셀 수·참값 분산이 모두 달라 "
             "직접 비교할 수 없다 (0.513 → 0.834에는 세대 효과와 평가면 교체가 섞여 있다).\n"
             "위 패널과도 다른 평가셋이다 — 위는 고정면 재계산, 아래는 문서 기록값 인용이다.",
             transform=ax2.transAxes, fontsize=8.6, color=INK2, va="top")

    p = OUT / "evolution_spearman.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ================================================================ FIG 2 MAE
def fig_mae(df):
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.0), dpi=170)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.815, bottom=0.105,
                        hspace=0.78, wspace=0.30)
    x = np.arange(len(df))
    ticks = gen_ticks(df)
    for ax, (key, lab, col) in zip(axes.ravel(), TARGETS):
        style(ax)
        y = df[f"mae_{key}"].to_numpy(float)
        ax.plot(x, y, color=col, lw=2.0, marker="o", ms=4.6, mec=SURFACE,
                mew=1.0, zorder=4)
        lo, hi = np.nanmin(y), np.nanmax(y)
        pad = (hi - lo) * 0.34 or abs(hi) * 0.06 or 1e-6
        ax.set_ylim(lo - pad, hi + pad * 1.5)
        best = int(np.nanargmin(y))
        ax.set_title(f"{lab}", fontsize=11.5, color=INK, loc="left",
                     weight="bold", pad=30)
        ax.text(0, 1.145, f"1세대 {y[0]:.4g}  →  2세대 {y[1]:.4g}  →  "
                f"{len(df)}세대 {y[-1]:.4g}", transform=ax.transAxes, fontsize=8.8,
                color=INK2)
        ax.text(0, 1.055, f"최저: {gen_ticks(df)[best].split(chr(10))[0]}"
                f" ({y[best]:.4g})", transform=ax.transAxes, fontsize=8.8,
                color=INK, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([t.split("\n")[0] for t in ticks], fontsize=7.6,
                           rotation=45, ha="right")
        ax.set_xlim(-0.4, len(df) - 0.6)
        ax.tick_params(axis="y", labelsize=8.4)
        ax.set_ylabel("MAE", fontsize=9)
        n = int(df[f"n_{key}"].iloc[0])
        ax.text(0.99, 0.04, f"n={n:,}", transform=ax.transAxes, fontsize=8,
                color=MUTED, ha="right")
    fig.suptitle("세대별 예측 오차 (MAE) — 동일 고정 평가면 · 지표별 축 분리",
                 fontsize=14, color=INK, weight="bold", x=0.065, ha="left",
                 y=0.965)
    fig.text(0.065, 0.917, "순위정확도(ρ)가 1차 지표이고 MAE는 보조다. 척도가 서로 "
             "달라 하나의 축에 겹치지 않고 지표마다 별도 축을 썼다 (이중축 없음).",
             fontsize=9.5, color=INK2)
    p = OUT / "evolution_mae.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ============================================================ FIG 3 frontier
# in-band feasible best F_r each campaign round delivered, placed at the
# generation whose model was in the loop for that round.
ROUNDS = [("r2", 1.5082, "split_S1b"),   # split_S1b was in the loop, T-blind
          ("r3", 1.5018, "s1c"),
          ("r4", 1.4866, "s1c"),
          ("r6", 1.4797, "s1d"),
          ("r8", 1.4749, "s1e")]
BATCHSWAP = 1.4605
#: The 2026-08-16 low-feed OPENING campaign (`fpcamp_minfr_N1N2_f113`, champion
#: s1f in the loop).  It is NOT a point on the T6_T4 series above: different cell
#: (N1_N2 / feed 113 / e_core 5.4), so its F_r is not commensurable with the
#: T6_T4 rounds' F_r and the two must never be joined by a line.  Drawn as its
#: own series precisely so the incommensurability is visible.
F113 = {"label": "N1_N2/f113 개통", "f_r": 1.4961, "efpd": 641.6, "guide": "s1f"}


def fig_frontier(df):
    gens = list(df["gen"])
    gi = {g: i for i, g in enumerate(gens)}
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.0, 8.6), dpi=170, sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.22], hspace=0.14,
                         left=0.088, right=0.815, top=0.885, bottom=0.175))
    x = np.arange(len(df))

    # -- top: F_r iso-rho
    style(ax)
    y = df["rho_iso_f_r"].to_numpy(float)
    ax.set_ylim(0.899, 0.959)
    ax.axhspan(y[0] - NOISE, y[0] + NOISE, color="#0b0b0b", alpha=0.055, lw=0,
               zorder=1)
    ax.plot(x, y, color=SLOT[0], lw=2.2, marker="o", ms=5.4, mec=SURFACE,
            mew=1.0, zorder=4)
    end_label(ax, x[-1], y[-1], f"F_r iso-ρ  {y[-1]:.3f}", SLOT[0])
    ax.text(0.02, y[0] - NOISE - 0.003,
            f"시드 노이즈 바닥 ±{NOISE:g} (1세대 기준) — 뚜렷이 벗어난 세대는 "
            "bu_T(+0.025)뿐이고, s1d·s1f는 경계선상(+0.0001·+0.0020)이다",
            fontsize=8.8, color=INK2, va="top")
    ax.set_ylabel("F_r iso군 내 Spearman ρ", fontsize=10.5)
    ax.set_title("정확도와 성과 — 위: 고정 평가면의 F_r 순위정확도 / "
                 "아래: 그 세대가 이끈 탐색 최전선",
                 fontsize=13.5, color=INK, weight="bold", loc="left", pad=26)
    ax.text(0, 1.055, "두 패널은 x축(세대)을 공유한다. 아래 점은 각 라운드를 "
            "'루프에 들어가 있던 모델'의 세대 위치에 놓은 것이다.",
            transform=ax.transAxes, fontsize=9.3, color=INK2)

    # -- bottom: the search frontier
    style(ax2)
    pos, seen = [], {}
    for rid, val, guide in ROUNDS:
        i = gi[guide]
        k = seen.get(i, 0)
        seen[i] = k + 1
        pos.append((i, val, rid, k))
    nrep = {i: c for i, c in seen.items()}
    xs, ys = [], []
    for i, val, rid, k in pos:
        off = 0.0 if nrep[i] == 1 else (-0.13 + 0.26 * k)
        xs.append(i + off)
        ys.append(val)
    ax2.plot(xs, ys, color=SLOT[1], lw=2.0, marker="o", ms=6.0, mec=SURFACE,
             mew=1.1, zorder=4)
    for (xx, yy), (_i, _v, rid, _k) in zip(zip(xs, ys), pos):
        ax2.text(xx, yy + 0.0032, f"{rid}  {yy:.4f}", ha="center", fontsize=9,
                 color=INK)
    # -- the f113 opening campaign: its OWN series, deliberately unconnected ---
    # Same x (the champion that guided it) but a different CELL, so it gets a
    # different colour, a different marker and no connecting line.  Reading it as
    # "the frontier improved to 1.4961" would be wrong: it is a first entry into a
    # cell that had no feasible core at all, not a step down the T6_T4 ladder.
    if F113["guide"] in gi:
        fx, fy = gi[F113["guide"]], F113["f_r"]
        ax2.plot([fx], [fy], marker="D", ms=7.5, color=SLOT[2], mec=SURFACE,
                 mew=1.2, ls="none", zorder=5)
        ax2.text(fx, fy + 0.0034, f"{F113['label']}  {fy:.4f}", ha="center",
                 fontsize=9, color=INK)
        ax2.text(fx, fy - 0.0030, f"({F113['efpd']:.1f} EFPD · 다른 셀 — T6_T4 계열과 "
                 "높이를 견주지 말 것)", ha="center", fontsize=8.2, color=INK2,
                 va="top")

    # out-of-band record, annotated OUTSIDE the campaign series
    ax2.axhline(BATCHSWAP, color=MUTED, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax2.text(-0.22, BATCHSWAP + 0.0016, f"batch_swap {BATCHSWAP:.4f}",
             fontsize=9.2, color=INK, va="bottom", ha="left")
    ax2.text(-0.22, BATCHSWAP - 0.0016,
             "캠페인 계열 밖 · 밴드 밖(618.0 EFPD) · 어떤 모델도 이끌지 않은 열거형 탐색",
             fontsize=8.4, color=INK2, va="top", ha="left")
    ax2.set_ylabel("밴드내 최우수 F_r  (↓ 좋음)", fontsize=10.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gen_ticks(df), fontsize=9)
    ax2.set_xlim(-0.35, len(df) - 0.62)
    ax2.set_ylim(BATCHSWAP - 0.0125, 1.5155)
    ax2.text(0, -0.30,
             "정직한 단서:  r2는 저장소 기준 430콜 블록 전체의 재계산값이다 "
             "(과제 표의 1.5440은 원격 풀 100행 기준이고 밴드 밖 619.93 EFPD).\n"
             "나머지 라운드는 각 100 MASTER 콜이다.  r1·r2를 이끈 split_S1b는 "
             "'모델 없음'이 아니라 T-셀 라벨을 한 줄도 못 본 T-blind 챔피언이다.\n"
             "1·2세대와 s1f가 이끈 T6_T4 캠페인 라운드는 없다 — 그래서 주황 계열에 "
             "해당 점이 없다.  r5·r7은 누적 최우수를 갱신하지 못해 최전선에 나타나지 않는다.\n"
             "초록 마름모는 T6_T4가 아니라 N1_N2/f113 셀의 첫 진입이다 (가용노심 0기 → 41기). "
             "셀이 다르므로 주황 계열과 잇지 않았고, 높이 비교도 성립하지 않는다.",
             transform=ax2.transAxes, fontsize=8.5, color=INK2, va="top",
             linespacing=1.6)
    p = OUT / "evolution_frontier.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    df = pd.read_csv(OUT / "evolution_metrics.csv")
    print(df[["gen", "date"] + [f"rho_iso_{k}" for k, _, _ in TARGETS]]
          .to_string(index=False))
    for f in (fig_spearman(df), fig_mae(df), fig_frontier(df)):
        print("wrote", f)


if __name__ == "__main__":
    main()
