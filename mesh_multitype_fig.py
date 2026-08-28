"""Style2 renderers for the MULTI-TYPE mesh (P4) — revision 2.

Implements ``data/reports/mesh_style_spec_20260817.md`` (axes, feed colours,
enrichment fills, pin-BU rings, boxless legend) with four operator-directed
changes over revision 1:

1. **The two phantom "unlabeled feed lines" are gone.**  They were never lines:
   ``mesh_style2_fig.background_cloud`` pads the feed window by +/-4, which is
   exactly one lattice step, so it dragged in **feed 105 (261 rows) and feed 133
   (2 813 rows)** — both outside the 109-129 mesh domain.  Since
   ``B_d = cyclen * const * 241 / feed``, a fixed feed is a straight ray through
   the origin, so each stray feed drew a dense unlabeled ray parallel to the
   labelled ones.  The operator's guess of "f105" was literally right; the other
   was f133.  All six real iso-feed lines were and are labelled.

2. **Declutter.**  Background cloud removed from the design maps entirely (it is
   what produced the phantom rays, and the store population now has its own
   figure, §4).  Iso-enrichment connectors reduced from full 6-node polylines to
   hairline segments between ADJACENT feeds, drawn only where both ends are
   real nodes.  Cells with no constrained content are small grey ticks, not
   full nodes.

3. **The node semantics are split into two aligned panels sharing both axes**,
   because one node cannot honestly carry both.  Revision 1 positioned nodes at
   the GATE-FREE maximum cyclen while colouring the ring from the CONSTRAINED
   representative — which is exactly why higher enrichment appeared to buy less
   cycle.  Top panel = unconstrained ceiling (reactivity physics, monotone).
   Bottom panel = constrained representative (where the high-e squeeze lives).

4. **A DB comparison figure**, ``mesh_multitype_vs_db_style2.png``: the
   MASTER-verified feasible-core database's per-cell best against ours,
   like-for-like on min-F_r, only where the DB actually covers.

    python mesh_multitype_fig.py --anchors 5.0:113
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "reports" / "mesh_multitype_20260818"
DB_XLSX = BASE / "data" / "reports" / "scoping_mesh_20260815" / "feasible_database.xlsx"
STORE = BASE / "data" / "store" / "records.parquet"

from mesh_style2_fig import (ENRICH_C, FEED_C, INK, ISO_E_C, MUTED,  # noqa: E402
                             N_FA, P_TH_MW, RING_C as _RING_C_BASE,
                             RING_LABEL as _RING_LABEL_BASE, _chrome, _setup,
                             nearest_e_color)

# --------------------------------------------------------------------------- #
# 핀 축 = ADVISORY (2026-08-20 사용자 판정)
# --------------------------------------------------------------------------- #
#: 공식 관측량은 **핀 axial peak** (= our ``max_pin_burnup``, the 3-D pin nodal
#: peak) and the licensing limit is **80 GWd/tU** —
#: ``data/reports/pinbu_definition_20260820.md``.
#:
#: MESH POLICY (user ruling, verbatim): "농축도가 올라가고 feed 수가 적어져서 핀
#: 조건을 만족하지 못하면 그물망 그림을 위해서는 해당 제약을 무시하고 나중에
#: 결과에 표기만".  So on THIS map the pin axis never closes a cell: tiers are
#: derived from the FOUR real axes (F_r / CBC / F_q / |AO|) and the pin is drawn
#: as the node RING — the user's own DB-figure convention.  It remains a real
#: gate for campaign delivery verdicts.
PIN_LIMIT_OFFICIAL = 80.0
#: the sweep's own predicted-pin gate (80 − 2.0 model margin); recorded because
#: the CSV's ``n_feasible`` / ``clean_*`` columns were produced under it.
PIN_GATE_SWEEP = 78.0

#: ring classes — the DB figure's 62/68/75 ladder plus the official 80 rung, so
#: an over-limit cell is VISIBLE rather than deleted.
RING_ORDER = ["green", "yellow", "orange", "red", "초과", "미산출"]
RING_C = dict(_RING_C_BASE, **{"초과": "#7B1E1E"})
RING_LABEL = dict(_RING_LABEL_BASE,
                  **{"red": "75 – 80", "초과": "≥ 80 (한도 초과 · 표기용)"})


def pin_class(v: float | None) -> str:
    """DB-convention pin class, extended with the official 80 GWd/tU rung."""

    if v is None or not np.isfinite(v):
        return "미산출"
    if v < 62:
        return "green"
    if v < 68:
        return "yellow"
    if v < 75:
        return "orange"
    if v < PIN_LIMIT_OFFICIAL:
        return "red"
    return "초과"

#: |delta| below this is not drawn — the sweep's pool-to-pool reproducibility is
#: not finer than this, and a dot for noise is a lie with a legend.
DEAD_BAND = 0.005
DELTA_FULL = 0.06
#: a cell is worth a full node when its constrained side is feasible, is within
#: reach of the gate, or has been MASTER-measured.  1.65 is not a new number:
#: it is tier-2 of the registered F_r ladder (scoping_mesh.TIERS).
NEAR_GATE = 1.65
FR_GATE = 1.55
TICK_C = "#B9BFC7"
#: the DB's own coverage.  Outside it the comparison is not "we win", it is
#: "the database was never asked".
DB_FEEDS = (109, 113, 117, 121)
DB_E_MAX = 5.5


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #
def _bd(cyclen, m_hm, feed):
    return (np.asarray(cyclen, float) * P_TH_MW / np.asarray(m_hm, float)
            / 1000.0) * N_FA / np.asarray(feed, float)


def prepare(nodes: pd.DataFrame) -> pd.DataFrame:
    """Split every cell into its CEILING and its CONSTRAINED representative.

    ``ceil_*``   the largest in-band predicted cyclen with NO gate applied, over
                 the trained (k=2) and sparse (k=3) cases.  k=4 is excluded:
                 the training store holds zero 4-type cores, so admitting it
                 into a panel whose whole claim is "this is the physics" would
                 let an extrapolation set the ceiling.
    ``con_*``    the joint-clean representative — the flattest core that passes
                 CBC, F_q, |AO| and pin — and ITS OWN cyclen, not the ceiling's.
                 Revision 1 mixed these two and that is the defect being fixed.
    """

    d = nodes.copy()
    d["e_target"] = d.e_target.round(1)
    d["feed"] = d.feed.astype(int)

    ceil = pd.DataFrame({k: d.get(f"max_pred_cyclen_any_{k}",
                                  pd.Series(np.nan, index=d.index)).astype(float)
                         for k in (2, 3)})
    allna = ceil.isna().all(axis=1)
    d["ceil_cyclen"] = ceil.max(axis=1)
    d["ceil_k"] = ceil[~allna].idxmax(axis=1).reindex(d.index).fillna(2).astype(int)

    fr = pd.DataFrame({k: d.get(f"min_f_r_clean_{k}",
                                pd.Series(np.nan, index=d.index)).astype(float)
                       for k in (2, 3, 4)})
    allna = fr.isna().all(axis=1)
    d["con_k"] = fr[~allna].idxmin(axis=1).reindex(d.index).fillna(0).astype(int)
    for col in ("clean_cyclen", "min_f_r_clean", "clean_pin", "clean_cbc",
                "n_feasible", "case"):
        d[f"con_{col}"] = [d.at[i, f"{col}_{k}"] if k and f"{col}_{k}" in d else np.nan
                           for i, k in zip(d.index, d.con_k)]
    # the strict like-for-like column: our 2-type constrained representative
    d["con2_f_r"] = d.get("min_f_r_clean_2", np.nan).astype(float)
    d["con2_cyclen"] = d.get("clean_cyclen_2", np.nan).astype(float)
    d["delta32"] = d.get("d_min_f_r_clean_3v2",
                         pd.Series(np.nan, index=d.index)).astype(float)

    d["ceil_x"] = d.ceil_cyclen
    d["ceil_y"] = _bd(d.ceil_cyclen, d.M_HM_tU, d.feed)
    d["con_x"] = d.con_clean_cyclen.astype(float)
    d["con_y"] = _bd(d.con_x, d.M_HM_tU, d.feed)
    d["pin_class"] = [pin_class(v) for v in d.con_clean_pin.astype(float)]
    d["fill"] = [nearest_e_color(e) for e in d.e_target]

    # ---- ADVISORY re-derivation: the pin is not a closing axis here -------- #
    # ``n_feasible``     = the sweep's FIVE-axis count (F_r/CBC/F_q/|AO| + pin)
    # ``n_feasible_4ax`` = the same four REAL axes with the pin term dropped —
    #                      exactly the per-axis pass data this policy needs.
    f5 = pd.DataFrame({k: d.get(f"n_feasible_{k}", pd.Series(np.nan, index=d.index)
                                ).astype(float) for k in (2, 3)}).fillna(0)
    f4 = pd.DataFrame({k: d.get(f"n_feasible_4ax_{k}",
                                pd.Series(np.nan, index=d.index)).astype(float)
                       for k in (2, 3)}).fillna(0)
    d["feas_pin"] = f5.max(axis=1) > 0            # old, pin-closed
    d["feas_adv"] = f4.max(axis=1) > 0            # new, pin advisory
    d["reopen"] = d.feas_adv & ~d.feas_pin
    #: the cell's BEST achievable predicted pin over the in-band pool — the
    #: honest per-cell annotation value, defined on all 90 cells.
    d["band_pin"] = pd.DataFrame(
        {k: d.get(f"min_pred_pin_{k}", pd.Series(np.nan, index=d.index)).astype(float)
         for k in (2, 3)}).min(axis=1)
    d["band_pin_class"] = [pin_class(v) for v in d.band_pin]
    #: a cell with NO pin-inclusive joint-clean representative but a non-empty
    #: four-axis feasible set PROVABLY gains one once the pin term is dropped.
    #: Its coordinates are not recoverable from the aggregate CSV (the sweep
    #: stored the representative, not the candidate matrix), so the map draws a
    #: bounded placeholder at the ceiling and says so.
    d["reopen_noclean"] = (~np.isfinite(d.con_x)) & d.feas_adv

    # measured = this cell's own 2-type pair has converged MASTER rows here
    s = pd.read_parquet(STORE, columns=["library_id", "case_pair", "feed",
                                        "valid", "converged"])
    s = s[(s.valid == True) & (s.converged == True)]              # noqa: E712
    n = s.groupby(["library_id", "case_pair", "feed"]).size()
    d["n_measured"] = [int(n.get((r.library_id, r.pair, r.feed), 0))
                       for r in d.itertuples()]

    near = d.con_min_f_r_clean.astype(float) <= NEAR_GATE
    d["content"] = (d.feas_adv | near | (d.n_measured > 0)).fillna(False)
    return d


# --------------------------------------------------------------------------- #
# shared chrome
# --------------------------------------------------------------------------- #
def _iso_lines(ax, pts: dict, feeds, targets, lw=2.0, adjacent_hairlines=True):
    """The 6 coloured iso-feed lines + short iso-enrichment hairlines.

    The revision-1 map drew a full 6-node polyline per enrichment level, i.e. 15
    grey zigzags across the whole field.  Here the enrichment connector is a
    segment between ADJACENT feeds only, and only when both ends exist — so it
    ties neighbours together without re-drawing a second full mesh on top of the
    feed lines.
    """

    for f in feeds:
        line = [pts[(e, f)] for e in targets if (e, f) in pts]
        if len(line) >= 2:
            ax.plot([p[0] for p in line], [p[1] for p in line], "-",
                    color=FEED_C.get(f, "#555555"), lw=lw, zorder=4,
                    solid_capstyle="round")
    if not adjacent_hairlines:
        return
    for e in targets:
        for a, b in zip(feeds, feeds[1:]):
            if (e, a) in pts and (e, b) in pts:
                ax.plot([pts[(e, a)][0], pts[(e, b)][0]],
                        [pts[(e, a)][1], pts[(e, b)][1]], "-", color=ISO_E_C,
                        lw=0.7, zorder=3, alpha=0.75, solid_capstyle="round")


def _feed_labels(ax, pts, feeds, targets, xa, gap, HALO):
    starts = []
    for f in feeds:
        cand = [pts[(e, f)] for e in targets if (e, f) in pts]
        if cand:
            starts.append((f, min(cand, key=lambda p: p[0])))
    ys: list[float] = []
    ordered = sorted(starts, key=lambda t: t[1][1])
    for _f, p0 in ordered:
        ys.append(p0[1] if not ys else max(p0[1], ys[-1] + gap))
    for (f, p0), y in zip(ordered, ys):
        ax.plot([p0[0], xa], [p0[1], y], "-", color="#C9CDD3", lw=0.8, zorder=9,
                clip_on=False)
        ax.annotate(f"feed {f}", (xa, y), textcoords="offset points",
                    xytext=(-4, 0), ha="right", va="center",
                    color=FEED_C.get(f, "#555555"), fontsize=10.5,
                    fontweight="bold", zorder=11, clip_on=False,
                    path_effects=HALO)
    return {f for f, _ in starts}


def _enrich_labels(ax, pts, feeds, targets, xb, gap, HALO):
    ends = []
    for e in targets:
        cand = [pts[(e, f)] for f in feeds if (e, f) in pts]
        if cand:
            ends.append((e, max(cand, key=lambda p: p[0])))
    ys: list[float] = []
    ordered = sorted(ends, key=lambda t: t[1][1])
    for _e, p1 in ordered:
        ys.append(p1[1] if not ys else max(p1[1], ys[-1] + gap))
    for (e, p1), y in zip(ordered, ys):
        ax.plot([p1[0], xb], [p1[1], y], "-", color="#C9CDD3", lw=0.8, zorder=9,
                clip_on=False)
        ax.annotate(f"{e:.1f} w/o", (xb, y), textcoords="offset points",
                    xytext=(5, 0), ha="left", va="center", color=nearest_e_color(e),
                    fontsize=10.0, fontweight="bold", zorder=11, clip_on=False,
                    path_effects=HALO)


# --------------------------------------------------------------------------- #
# 1+2+3. the two-panel design map
# --------------------------------------------------------------------------- #
def render_two_panel(d: pd.DataFrame, out_path: Path, anchors: set) -> tuple[Path, dict]:
    _setup()
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    HALO = [pe.withStroke(linewidth=2.6, foreground="white", alpha=0.88)]

    feeds = sorted(d.feed.unique())
    targets = sorted(d.e_target.unique())
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(13.4, 14.4), dpi=150,
                                   sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in (axT, axB):
        ax.set_facecolor("white")

    # ---------------- top: unconstrained ceiling --------------------------- #
    ptsT = {(r.e_target, r.feed): (r.ceil_x, r.ceil_y) for r in d.itertuples()
            if np.isfinite(r.ceil_x)}
    _iso_lines(axT, ptsT, feeds, targets, lw=2.0)
    # ticks keep their SHAPE (= no constrained content) but take the pin class
    # as their COLOUR: under the advisory policy the pin cliff must stay visible
    # in exactly the cells it used to delete, and those are all tick cells.
    tick = d[~d.content.astype(bool)]
    axT.scatter(tick.ceil_x, tick.ceil_y, s=26, marker="|",
                c=[RING_C[c] for c in tick.band_pin_class], linewidths=1.8,
                zorder=6)
    node = d[d.content.astype(bool)]
    # ring = ADVISORY pin class of the cell's BEST in-band predicted pin.  The
    # pin closes nothing here; this is the annotation the ruling asks for, and
    # it is drawn on the ceiling panel because that is where the whole grid —
    # including every cell the pin used to delete — is present.
    for cls in RING_ORDER:
        m = node.band_pin_class == cls
        if m.any():
            s_ = node[m]
            axT.scatter(s_.ceil_x, s_.ceil_y, s=100, c=list(s_.fill),
                        edgecolors=RING_C[cls], linewidths=2.2, zorder=8)

    # ---------------- bottom: constrained representative ------------------- #
    con = d[np.isfinite(d.con_x)]
    ptsB = {(r.e_target, r.feed): (r.con_x, r.con_y) for r in con.itertuples()}
    _iso_lines(axB, ptsB, feeds, targets, lw=2.0)
    for cls in RING_ORDER:
        m = con.pin_class == cls
        if m.any():
            s = con[m]
            axB.scatter(s.con_x, s.con_y, s=115, c=list(s.fill),
                        edgecolors=RING_C[cls], linewidths=2.5, zorder=8)
    # cells the PIN alone had closed: they re-open under the advisory policy.
    # Their representative provably exists (>=1 candidate clears F_r/CBC/F_q/|AO|)
    # but the aggregate CSV kept only the pin-inclusive representative, so the
    # node is drawn OPEN at the cell's ceiling — an upper bound on its cyclen,
    # with a caret pointing the way the true node lies.
    reop = d[d.reopen_noclean.astype(bool)]
    for r in reop.itertuples():
        axB.scatter([r.ceil_x], [r.ceil_y], s=150, facecolors="white",
                    edgecolors=RING_C[pin_class(r.band_pin)], linewidths=2.2,
                    linestyle=(0, (2, 1.4)), zorder=9)
        axB.annotate("", xy=(r.ceil_x - 0.052 * (d.ceil_x.max() - d.ceil_x.min()),
                             r.ceil_y), xytext=(r.ceil_x, r.ceil_y),
                     arrowprops=dict(arrowstyle="-|>", color="#5A6068", lw=1.1,
                                     shrinkA=7, shrinkB=0), zorder=9)
        axB.annotate(f"{r.e_target:.1f}/{r.feed} 핀 재개방", (r.ceil_x, r.ceil_y),
                     textcoords="offset points", xytext=(0, 13), ha="center",
                     fontsize=8.2, color="#7B1E1E", fontweight="bold",
                     path_effects=HALO, zorder=12)

    for r in con.itertuples():
        if (r.e_target, r.feed) in anchors:
            axB.scatter([r.con_x], [r.con_y], s=430, facecolors="none",
                        edgecolors=INK, linewidths=2.2, zorder=6)
        v = r.delta32
        if np.isfinite(v) and abs(v) >= DEAD_BAND:
            area = 12 + 40 * min(1.0, abs(v) / DELTA_FULL)
            if v < 0:
                axB.scatter([r.con_x], [r.con_y], s=area, c="white", linewidths=0,
                            zorder=10)
            else:
                axB.scatter([r.con_x], [r.con_y], s=area, facecolors="none",
                            edgecolors="white", linewidths=1.3, zorder=10)

    # ---------------- shared limits and labels ----------------------------- #
    xs = np.concatenate([d.ceil_x.dropna().to_numpy(), con.con_x.to_numpy()])
    ys = np.concatenate([d.ceil_y.dropna().to_numpy(), con.con_y.to_numpy()])
    span_x, span_y = xs.max() - xs.min(), ys.max() - ys.min()
    axT.set_xlim(xs.min() - 0.20 * span_x, xs.max() + 0.19 * span_x)
    axT.set_ylim(ys.min() - 0.06 * span_y, ys.max() + 0.06 * span_y)
    xa = xs.min() - 0.115 * span_x
    xb = xs.max() + 0.035 * span_x
    gapT, gapB = 0.052 * span_y, 0.052 * span_y
    labelled = _feed_labels(axT, ptsT, feeds, targets, xa, gapT, HALO)
    _enrich_labels(axT, ptsT, feeds, targets, xb, gapT, HALO)
    _feed_labels(axB, ptsB, feeds, targets, xa, gapB, HALO)
    _enrich_labels(axB, ptsB, feeds, targets, xb, gapB, HALO)

    for ax in (axT, axB):
        _chrome(ax, "", "방출연소도 (GWd/tU, 질량수지)")
    axB.set_xlabel("주기길이 (EFPD)", fontsize=13, color=INK, labelpad=9)

    axT.set_title("① 무제약 천장 — 게이트 없는 밴드내 최대 예측 주기길이(2·3종 중 큰 쪽).  "
                  "테두리·눈금 색 = 그 셀의 최저 예측 핀연소도 등급(표기 전용).",
                  fontsize=11.4, color=INK, loc="left", pad=10)
    axB.set_title("② 제약 통과 대표핵 — CBC·F_q·|AO| 통과 후보 중 최저 F_r 노심과 그 노심의 "
                  "주기길이.  핀은 판정에 넣지 않는다(표기만).",
                  fontsize=11.4, color=INK, loc="left", pad=10)

    # ---------------- legend ------------------------------------------------ #
    fig.tight_layout(rect=(0, 0.235, 1, 0.945))
    used_e = [e for e in sorted(ENRICH_C) if e in set(targets)]
    # 4 x 4, not 2 x 8: at eight rows the last swatch landed on the footnote.
    lx0, ly0, lstep, ncol_e = 0.040, 0.206, 0.0158, 4
    axT.text(lx0, ly0 + lstep, "농축도 (노드 채움)", transform=fig.transFigure,
             ha="left", va="top", fontsize=10.5, color=INK, fontweight="bold")
    for i, e in enumerate(used_e):
        col, row = divmod(i, ncol_e)
        x, y = lx0 + col * 0.052, ly0 - (row + 1) * lstep
        axT.plot([x], [y], "o", ms=8.5, color=ENRICH_C[e], mec="white", mew=0.6,
                 transform=fig.transFigure, clip_on=False)
        axT.text(x + 0.009, y, f"{e:.1f}", transform=fig.transFigure, ha="left",
                 va="center", fontsize=9.0, color=INK)
    rx0 = lx0 + 4 * 0.052 + 0.037
    axT.text(rx0, ly0 + lstep, "핀 axial peak 연소도 등급  ·  표기용",
             transform=fig.transFigure, ha="left", va="top", fontsize=10.5,
             color=INK, fontweight="bold")
    for i, cls in enumerate(RING_ORDER):
        y = ly0 - (i + 1) * lstep
        axT.plot([rx0], [y], "o", ms=8.5, color="white", mec=RING_C[cls], mew=2.2,
                 transform=fig.transFigure, clip_on=False)
        axT.text(rx0 + 0.012, y, f"{cls}  ({RING_LABEL[cls]})",
                 transform=fig.transFigure, ha="left", va="center", fontsize=9.0,
                 color=INK)
    tx0 = rx0 + 0.205
    axT.text(tx0, ly0 + lstep, "다종 층 · 마커 규칙",
             transform=fig.transFigure, ha="left", va="top", fontsize=10.5,
             color=INK, fontweight="bold")
    demo = [("●  흰 점 = 3종이 더 평평 (Δ<0), 면적 ∝ |Δ|", "fill"),
            ("○  흰 테두리 = 3종이 더 나쁨 (Δ>0)", "open"),
            ("◯  굵은 먹색 후광 = MASTER 앵커 셀", "anchor"),
            ("|   눈금(상단) = 제약측 내용이 없는 셀 · 색은 핀 등급", "tick"),
            ("⊘  점선 테두리(하단) = 핀 표기 정책으로 재개방된 셀 — 대표핵 좌표는", "reopen"),
            ("     스윕 집계 CSV로 복원 불가라 천장(상한) 위치에 그리고 화살표로 방향만 표시", "none"),
            ]
    for i, (txt, kind) in enumerate(demo):
        y = ly0 - (i + 1) * lstep
        if kind == "fill":
            axT.plot([tx0 + 0.004], [y], "o", ms=7.2, color="#4B4B4B", mec="none",
                     transform=fig.transFigure, clip_on=False)
            axT.plot([tx0 + 0.004], [y], "o", ms=4.2, color="white", mec="none",
                     transform=fig.transFigure, clip_on=False)
        elif kind == "open":
            axT.plot([tx0 + 0.004], [y], "o", ms=7.2, color="#4B4B4B", mec="none",
                     transform=fig.transFigure, clip_on=False)
            axT.plot([tx0 + 0.004], [y], "o", ms=4.4, color="none", mec="white",
                     mew=1.2, transform=fig.transFigure, clip_on=False)
        elif kind == "anchor":
            axT.plot([tx0 + 0.004], [y], "o", ms=9.0, color="none", mec=INK,
                     mew=2.0, transform=fig.transFigure, clip_on=False)
        elif kind == "reopen":
            axT.plot([tx0 + 0.004], [y], "o", ms=8.6, color="white",
                     mec=RING_C["초과"], mew=2.0, ls=(0, (2, 1.4)),
                     transform=fig.transFigure, clip_on=False)
        elif kind == "tick":
            axT.plot([tx0 + 0.004], [y], marker="|", ms=8, color=TICK_C, mew=1.6,
                     transform=fig.transFigure, clip_on=False)
        axT.text(tx0 + 0.015, y, txt, transform=fig.transFigure, ha="left",
                 va="center", fontsize=9.0, color=INK)

    fig.text(0.040, 0.986, "APR1400 LEU+ 다종(2/3종) 재장전 설계지도 — 천장 / 제약 2단",
             fontsize=15.5, color=INK, ha="left", va="top")
    fig.text(0.040, 0.958,
             "90개 격자점 (e_core 5.0-6.4 × feed 109-129) · s1i(cond_v8) 예측 · "
             "판정 게이트 4축(F_r·CBC·F_q·|AO|) · 핀은 표기용, 지도 판정 미반영",
             fontsize=11.0, color="#3C3C3C", ha="left", va="top")
    n_con = int(np.isfinite(d.con_x).sum())
    n_reop = int(d.reopen.sum())
    n_reop_nc = int(d.reopen_noclean.sum())
    lines = [
        "핀 정책 (사용자 판정 2026-08-20, data/reports/pinbu_definition_20260820.md): 관측량은 "
        "핀 axial peak(= max_pin_burnup, 3-D 핀 노드 첨두), 한도는 80 GWd/tU.  설계지도에서 핀은 "
        f"ADVISORY — 셀을 닫지 않고 링 색으로 표기만 한다.  이 정책으로 tier-1 실행가능 셀이 "
        f"{int(d.feas_pin.sum())} → {int(d.feas_adv.sum())} 로 늘었다(핀 단독 폐쇄 {n_reop}셀 재개방).",
        "양식: mesh_style_spec_20260817.md.  §6 배경 구름은 운영자 지시로 제거 — feed 창을 "
        "±4 패딩하던 탓에 격자 밖 feed 105·133 이 라벨 없는 직선 광선으로 찍혀 있었다"
        "(그것이 '이름 없는 두 선'의 정체다).  저장소 모집단은 vs-DB 그림이 담당한다.",
        f"상단 {len(d)}셀 전부 · 하단은 joint-clean 대표핵이 있는 {n_con}셀 + 재개방 {n_reop_nc}셀"
        f"(점선, 천장 위치는 상한).  두 패널의 x 차이가 곧 '게이트가 가져간 주기길이'다.  "
        f"3종 학습지지 39행·4종 0행이라 천장 패널에서 4종은 제외했다.",
    ]
    fig.text(0.040, 0.010, "\n".join(textwrap.fill(t, 128) for t in lines),
             fontsize=9.2, color=MUTED, ha="left", va="bottom", linespacing=1.6)

    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path, dict(n_nodes=len(d), n_content=int(d.content.sum()),
                          n_tick=int((~d.content.astype(bool)).sum()),
                          n_constrained=n_con, feeds_labelled=sorted(labelled),
                          feas_pin=int(d.feas_pin.sum()),
                          feas_adv=int(d.feas_adv.sum()), n_reopen=n_reop,
                          reopened=sorted(d.loc[d.reopen, "cell"].tolist()),
                          reopen_noclean=sorted(d.loc[d.reopen_noclean, "cell"].tolist()),
                          ring_top=d[d.content.astype(bool)].band_pin_class.value_counts().to_dict(),
                          con_k=d[np.isfinite(d.con_x)].con_k.value_counts().to_dict())


# --------------------------------------------------------------------------- #
# delta companion panel (unchanged semantics)
# --------------------------------------------------------------------------- #
def render_delta_panel(d: pd.DataFrame, out_path: Path, anchors: set,
                       title: str, subtitle: str) -> Path:
    _setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    feeds = sorted(d.feed.astype(int).unique())
    targets = sorted(d.e_target.round(1).unique())
    M = np.full((len(targets), len(feeds)), np.nan)
    state = np.empty((len(targets), len(feeds)), dtype=object)
    state[:] = ""
    for r in d.itertuples():
        i, j = targets.index(round(r.e_target, 1)), feeds.index(int(r.feed))
        case3 = getattr(r, "case_3", "")
        if not isinstance(case3, str) or not case3:
            state[i, j] = "no3"
            continue
        state[i, j] = "cross" if not bool(getattr(r, "mono_anchor_3", False)) else "mono"
        M[i, j] = r.delta32

    lim = max(0.02, float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.02)
    fig, ax = plt.subplots(figsize=(9.6, 8.0), dpi=150)
    fig.patch.set_facecolor("white")
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(feeds)), [str(f) for f in feeds])
    ax.set_yticks(range(len(targets)), [f"{t:.1f}" for t in targets])
    for i in range(len(targets)):
        for j in range(len(feeds)):
            st = state[i, j]
            if st == "no3":
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#EDEFF2",
                                           edgecolor="white", lw=1.0, hatch="///"))
                continue
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8.6,
                        color=INK if abs(v) < 0.55 * lim else "white")
            else:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="#F7F8FA", edgecolor="white",
                                           lw=1.0))
                ax.text(j, i, "–", ha="center", va="center", fontsize=11,
                        color="#B4BAC2")
            if st == "cross":
                ax.plot([j - .40], [i + .36], "s", ms=4.2, color="#7A4FA3",
                        mec="white", mew=0.6)
            if (round(targets[i], 1), feeds[j]) in anchors:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="none",
                                           edgecolor=INK, lw=2.4))
    ax.set_xlabel("feed (신연료 집합체 수)", fontsize=12, color=INK, labelpad=8)
    ax.set_ylabel("목표 노심평균 농축도 (w/o)", fontsize=12, color=INK, labelpad=8)
    ax.tick_params(colors="#3C3C3C", labelsize=10, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label("Δ(3종 − 2종)  ·  음수 = 계단화 이득", fontsize=9.5, color=INK)
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=14.5, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.938, subtitle, fontsize=10.4, color="#3C3C3C", ha="left",
             va="top")
    fig.text(0.02, 0.012,
             "빗금 = 로스터에 조성일치 3종 사다리가 아예 없는 셀.\n"
             "옅은 회색 「–」 = 3종은 있으나 2종·3종 중 한쪽에 joint-clean 노심이 없어 "
             "Δ가 정의되지 않는 셀 — 흰색(=0)과 구별해 표시한다.\n"
             "보라 사각 = 그 셀의 3종이 R1 cross-spec(앵커 불가, PREREG §3.1).   "
             "굵은 먹색 테두리 = MASTER 앵커로 선정된 셀.\n"
             "핀 정책 각주(2026-08-20): 이 Δ는 스윕이 산출한 핀 포함 joint-clean 바닥 "
             "차이 그대로다 — 핀을 뺀 Δ는 후보 행렬이 저장되지 않아 집계 CSV로 재도출할 "
             "수 없다.  핀이 관여한 셀은 지도(map_style2)의 재개방 표기를 함께 볼 것.",
             fontsize=9.2, color=MUTED, ha="left", va="bottom", linespacing=1.7)
    fig.tight_layout(rect=(0, 0.10, 1, 0.925))
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 4. vs the MASTER-verified feasible-core database
# --------------------------------------------------------------------------- #
def load_db_best() -> pd.DataFrame:
    """Per (enrichment segment, feed) the DB's MIN-F_r feasible core.

    ``feasible`` is the database's own verdict (``eq_ok & feasible_at_metrics``),
    which INCLUDES F_r — so its per-cell minimum is a fully legal core, while
    our joint-clean floor is legal on the other four axes with F_r free.  The
    comparison below is therefore min-F_r against min-F_r, and the caption says
    which side is already inside the gate.

    Every one of the 4 569 feasible DB cores is **2-type**; there is no graded
    core anywhere in the database.  That is why the like-for-like partner is our
    k=2 representative and the 3-type mark is drawn separately.
    """

    c = pd.read_excel(DB_XLSX, "cores")
    c = c[c.eq_ok & c.feasible_at_metrics].copy()
    c["e_target"] = c.enrichment_segment.round(1)
    c["feed"] = c.feed.astype(int)
    i = c.groupby(["e_target", "feed"]).F_r.idxmin()
    # BOTH pin columns: ``pinmax_node_GWd`` is the one that shares our
    # observable's tier (pin axial peak); ``pinmax_rodavg_GWd`` belongs with our
    # SECONDARY ``max_rod_avg_burnup``.  Carrying only rodavg is what produced
    # the withdrawn "our pin prediction is 9 GWd/tU pessimistic" reading.
    b = c.loc[i, ["e_target", "feed", "F_r", "EFPD", "CBC_max", "F_q",
                  "core_mean_discharge_GWd", "pinmax_rodavg_GWd",
                  "pinmax_node_GWd", "n_types", "pair"]].copy()
    b["B_d"] = _bd(b.EFPD, 102.031495, b.feed)
    return b.set_index(["e_target", "feed"])


def render_vs_db(d: pd.DataFrame, out_path: Path) -> tuple[Path, dict]:
    _setup()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec, patheffects as pe
    from matplotlib.colors import TwoSlopeNorm
    HALO = [pe.withStroke(linewidth=2.6, foreground="white", alpha=0.88)]

    db = load_db_best()
    cov = d[(d.e_target <= DB_E_MAX) & (d.feed.isin(DB_FEEDS))].copy()
    cov["db_f_r"] = [float(db.F_r.get((r.e_target, r.feed), np.nan))
                     for r in cov.itertuples()]
    cov["db_x"] = [float(db.EFPD.get((r.e_target, r.feed), np.nan))
                   for r in cov.itertuples()]
    cov["db_y"] = [float(db.B_d.get((r.e_target, r.feed), np.nan))
                   for r in cov.itertuples()]
    cov["db_node"] = [float(db.pinmax_node_GWd.get((r.e_target, r.feed), np.nan))
                      for r in cov.itertuples()]
    cov["db_rodavg"] = [float(db.pinmax_rodavg_GWd.get((r.e_target, r.feed), np.nan))
                        for r in cov.itertuples()]
    cov["ours_f_r"] = cov.con2_f_r
    cov["ours_x"] = cov.con2_cyclen
    cov["ours_y"] = _bd(cov.con2_cyclen, cov.M_HM_tU, cov.feed)
    cov["d_f_r"] = cov.ours_f_r - cov.db_f_r

    fig = plt.figure(figsize=(15.6, 8.6), dpi=150)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.62, 1.0], wspace=0.20,
                           left=0.055, right=0.975, top=0.845, bottom=0.205)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("white")

    # our full constrained field, faded, so the covered block has context
    out = d[np.isfinite(d.con_x) & ~(
        (d.e_target <= DB_E_MAX) & (d.feed.isin(DB_FEEDS)))]
    ax.scatter(out.con_x, out.con_y, s=58, c=list(out.fill), alpha=0.22,
               edgecolors="none", zorder=5)

    ok = cov[np.isfinite(cov.db_x) & np.isfinite(cov.ours_x)]
    for r in ok.itertuples():
        ax.plot([r.db_x, r.ours_x], [r.db_y, r.ours_y], "-", color="#9AA5B5",
                lw=1.0, zorder=6)
    ax.scatter(ok.db_x, ok.db_y, s=118, marker="D", facecolors="none",
               edgecolors=[nearest_e_color(e) for e in ok.e_target],
               linewidths=2.0, zorder=9)
    ax.scatter(ok.ours_x, ok.ours_y, s=112, c=list(ok.fill),
               edgecolors="white", linewidths=1.2, zorder=10)

    miss = cov[np.isfinite(cov.db_x) & ~np.isfinite(cov.ours_x)]
    ax.scatter(miss.db_x, miss.db_y, s=118, marker="D", facecolors="none",
               edgecolors="#C0392B", linewidths=2.0, zorder=9)

    _chrome(ax, "주기길이 (EFPD)", "방출연소도 (GWd/tU, 질량수지)")
    ax.set_title("DB 실측 최적핵(◇) ↔ 우리 2종 제약 대표핵(●), 같은 셀끼리 연결",
                 fontsize=11.6, color=INK, loc="left", pad=10)
    # alternate the label above/below by feed index so the f117/f121 clusters,
    # which sit almost on top of each other, do not print into one another.
    for r in pd.concat([ok, miss]).itertuples():
        up = (list(DB_FEEDS).index(int(r.feed)) % 2 == 0)
        ax.annotate(f"{r.e_target:.1f}/{r.feed}", (r.db_x, r.db_y),
                    textcoords="offset points", xytext=(0, 11 if up else -15),
                    ha="center", fontsize=7.6,
                    color=nearest_e_color(r.e_target) if np.isfinite(r.ours_x)
                    else "#C0392B", path_effects=HALO, zorder=12)

    # ---- right: the like-for-like number ---------------------------------- #
    ax2 = fig.add_subplot(gs[0, 1])
    feeds = list(DB_FEEDS)
    targets = sorted(cov.e_target.unique())
    M = np.full((len(targets), len(feeds)), np.nan)
    for r in cov.itertuples():
        M[targets.index(r.e_target), feeds.index(r.feed)] = r.d_f_r
    lim = max(0.02, float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.02)
    im = ax2.imshow(M, cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                    aspect="auto", origin="lower")
    ax2.set_xticks(range(len(feeds)), [str(f) for f in feeds])
    ax2.set_yticks(range(len(targets)), [f"{t:.1f}" for t in targets])
    for i, e in enumerate(targets):
        for j, f in enumerate(feeds):
            row = cov[(cov.e_target == e) & (cov.feed == f)]
            if not len(row):
                continue
            r = row.iloc[0]
            if not np.isfinite(r.ours_f_r):
                ax2.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                            facecolor="#F7F8FA", edgecolor="white",
                                            lw=1.0))
                ax2.text(j, i, f"DB {r.db_f_r:.3f}\n우리 없음", ha="center",
                         va="center", fontsize=7.4, color="#B0392B")
                continue
            v = float(r.d_f_r)
            mark = " ✔" if r.ours_f_r <= FR_GATE else ""
            ax2.text(j, i, f"DB {r.db_f_r:.3f}\n우리 {r.ours_f_r:.3f}{mark}\n{v:+.3f}",
                     ha="center", va="center", fontsize=7.4,
                     color=INK if abs(v) < 0.55 * lim else "white")
    ax2.set_xlabel("feed", fontsize=11, color=INK, labelpad=7)
    ax2.set_ylabel("농축도 (w/o)", fontsize=11, color=INK, labelpad=7)
    ax2.tick_params(colors="#3C3C3C", labelsize=9.5, length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.set_title("min-F_r 대 min-F_r  ·  음수 = 우리가 더 평평",
                  fontsize=11.6, color=INK, loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax2, fraction=0.040, pad=0.02)
    cb.set_label("F_r(우리 2종) − F_r(DB)", fontsize=9.2, color=INK)
    cb.outline.set_visible(False)

    fig.text(0.055, 0.975,
             "다종 메쉬 대 MASTER 실측 가용노심 DB — 같은 셀, 같은 지표(min-F_r)",
             fontsize=15.0, color=INK, ha="left", va="top")
    fig.text(0.055, 0.930,
             f"DB 커버리지 안({len(DB_FEEDS)} feed × {len(targets)} 농축도 = "
             f"{len(cov)}셀)에서만 비교한다.  DB 가용핵 4,569기는 전부 2종이므로 "
             "짝은 우리 2종 대표핵이다.",
             fontsize=10.6, color="#3C3C3C", ha="left", va="top")
    node = cov.db_node.astype(float).dropna()
    fig.text(0.055, 0.012,
             "◇ = DB 가용핵(=F_r 포함 전 제약 통과) 중 최저 F_r.  ● = 우리 2종 joint-clean "
             "대표핵(CBC·F_q·|AO| 통과, F_r 자유) 중 최저 F_r — 우리 값이 1.55 를 넘으면 "
             "아직 가용이 아니다(통과분은 ✔).\n"
             "잣대 정정(2026-08-20): 우리 관측량과 같은 계층인 DB 열은 pinmax_node_GWd 이다"
             f"(rodavg 가 아니다).  그 축에서 DB 기준핵 {len(node)}기는 {node.min():.1f}-"
             f"{node.max():.1f} GWd/tU 이고 78 이하는 {int((node <= 78).sum())}/{len(node)}, "
             f"80 이하는 {int((node <= PIN_LIMIT_OFFICIAL).sum())}/{len(node)} — "
             "즉 DB 기준핵도 전부가 우리 핀 게이트를 통과하지는 않는다.\n"
             "옅은 점 = DB 미포괄 셀(feed 125·129 전부, e ≥ 5.6 전부) — 비교 대상이 아니다.  "
             "빨간 ◇ = DB 에는 실측 가용핵이 있는데 우리 쪽엔 joint-clean 후보가 없는 7셀.\n"
             "'우리 핀 예측이 최대 9 GWd/tU 비관적'이라는 종전 읽기는 철회한다 — 그 9 는 "
             "node↔rodavg 축 혼용의 산물이었다(README §5.1).",
             fontsize=9.0, color=MUTED, ha="left", va="bottom", linespacing=1.7)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    stat = dict(db_node_span=(float(node.min()), float(node.max())),
                db_node_le78=int((node <= PIN_GATE_SWEEP).sum()),
                db_node_le80=int((node <= PIN_LIMIT_OFFICIAL).sum()),
                db_rodavg_span=(float(cov.db_rodavg.min()), float(cov.db_rodavg.max())),
                n_covered=len(cov), n_paired=len(ok), n_db_only=len(miss),
                n_uncovered=int(np.isfinite(d.con_x).sum() - len(ok) - len(miss)),
                mean_d_f_r=float(np.nanmean(cov.d_f_r)),
                median_d_f_r=float(np.nanmedian(cov.d_f_r)),
                ours_better=int((cov.d_f_r < 0).sum()),
                db_better=int((cov.d_f_r > 0).sum()))
    return out_path, stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=str(OUT / "mesh_multitype.csv"))
    ap.add_argument("--anchors", default="")
    args = ap.parse_args()

    d = prepare(pd.read_csv(args.nodes))
    anchors = set()
    for tok in filter(None, args.anchors.split(",")):
        e, f = tok.split(":")
        anchors.add((round(float(e), 1), int(f)))

    p1, cov = render_two_panel(d, OUT / "mesh_multitype_map_style2.png", anchors)
    print("map:", p1, cov)
    p2 = render_delta_panel(
        d, OUT / "mesh_multitype_delta_panel.png", anchors,
        "계단화 이득 지도 — Δ(3종 − 2종) joint-clean F_r 바닥",
        "예측 전용(MASTER 아님) · s1i · 3종 학습지지 39행 → 예측이지 검증이 아니다")
    print("panel:", p2)
    p3, st = render_vs_db(d, OUT / "mesh_multitype_vs_db_style2.png")
    print("vs_db:", p3, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
