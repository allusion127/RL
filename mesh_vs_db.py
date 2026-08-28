"""Compare the MODEL-ONLY scoping mesh against the MASTER-verified feasible-core
database (``feasible_database.xlsx``).

The database carries no loading pattern and its ``cid`` resolves to nothing in
``data/store`` (0 / 6113 on record_id, record_id[:16] and the pattern digest), so
the model CANNOT be run on those exact cores.  The model-vs-truth question is
therefore answered on the store's OWN MASTER labels — the same (pair, feed) cells,
scored through the same ``predict()`` path the mesh used — which yields a per-cell
F_r bias.  That bias decomposes the frontier gap:

    mesh_min_pred_F_r  -  db_min_true_F_r  =  model_bias  +  pool_gap

    python mesh_vs_db.py             # full run (model inference ~5 min on CPU)
    python mesh_vs_db.py --no-model  # skip inference, reuse model_bias.csv
    python mesh_vs_db.py --model s1e # score the bias with an older champion

The bias MUST be measured with the same ensemble that drew the mesh, otherwise
``corrected_floor`` mixes two models — ``--model`` defaults to ``s1f``, matching
``scoping_mesh.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
OUT = BASE / "data" / "reports" / "scoping_mesh_20260815"
XLSX = OUT / "feasible_database.xlsx"

FR_LIMIT, FQ_LIMIT, CBC_LIMIT, AO_LIMIT = 1.55, 2.41, 1600.0, 0.30
P_TH_MW, N_FA = 3983.0, 241
M_HM_GA80 = 102.031495          # tU, the mesh's ga80 value

#: per-cell store sample for the bias measurement
N_TAIL, N_RAND = 40, 60


def load_db() -> pd.DataFrame:
    c = pd.read_excel(XLSX, "cores")
    c["ok"] = c.eq_ok & c.feasible_at_metrics
    c["segment"] = c.enrichment_segment.round(1)
    bd = (c.EFPD * P_TH_MW / M_HM_GA80 / 1000.0) * N_FA / c.feed
    c["bd_massbalance"] = bd
    c["mb_rel"] = 100.0 * (bd - c.core_mean_discharge_GWd) / c.core_mean_discharge_GWd
    return c


def measure_model_bias(db_cells: set, log, model_name: str = "s1f") -> pd.DataFrame:
    """Champion predicted-vs-true F_r / cyclen on the store's ga80 MASTER rows.

    Scored through ``predict()`` with a ``CaseKey(pair, feed)`` — byte-identical
    to the path the mesh used, so the bias is directly subtractable from the
    mesh's floor.  Two samples per cell: the 40 lowest-F_r rows (the FRONTIER,
    which is what a min-F_r comparison actually rides on) and 60 random rows
    (an unbiased cell estimate)."""

    import torch
    torch.set_num_threads(10)
    from lpopt.model.model_api import PosValCnnBackend
    from lpopt.data.schema import unpack_pattern
    from lpopt.vendor.masterrl.domain import CaseKey

    model = PosValCnnBackend.from_dir(BASE / "data/models" / model_name,
                                      store_dir=BASE / "data/store",
                                      library_id="ga80", device="cpu")
    model.quantile_targets = ()          # same as the mesh run
    st = pd.read_parquet(BASE / "data/store/records.parquet")
    st = st[(st.valid == True) & (st.converged == True)      # noqa: E712
            & (st.library_id == "ga80") & st.f_r.notna()]
    st = st.assign(segment=st.e_core.round(1))
    rows = []
    rng = np.random.default_rng(20260815)
    for (seg, feed), g in st.groupby(["segment", "feed"]):
        if (seg, feed) not in db_cells:
            continue
        tail = g.nsmallest(N_TAIL, "f_r")
        rest = g.drop(index=tail.index)
        rand = rest.sample(min(N_RAND, len(rest)), random_state=7) if len(rest) else rest
        sample = pd.concat([tail.assign(grp="tail"), rand.assign(grp="rand")])
        for pair, gp in sample.groupby("case_pair"):
            try:
                pats = [unpack_pattern(str(p)) for p in gp["pattern"]]
            except Exception:                        # noqa: BLE001
                continue
            pred = model.predict(pats, CaseKey(str(pair), int(feed)), float(seg))
            m = pred.mean
            for i, (_, r) in enumerate(gp.iterrows()):
                rows.append(dict(segment=seg, feed=int(feed), pair=str(pair),
                                 grp=r.grp, record_id=str(r.record_id),
                                 f_r_true=float(r.f_r), f_r_pred=float(m[i, 0]),
                                 cyclen_true=float(r.cyclen), cyclen_pred=float(m[i, 3]),
                                 cbc_true=float(r.cbc_max), cbc_pred=float(m[i, 1])))
        log(f"  bias sample seg {seg} feed {feed}: {len(sample)} rows")
    return pd.DataFrame(rows)


def build_verdicts(db: pd.DataFrame, mesh: pd.DataFrame, bias: pd.DataFrame
                   ) -> pd.DataFrame:
    """Per (segment x feed) cell: truth frontier, mesh floor, bias-corrected floor,
    and the verdict that separates model pessimism from pool starvation."""

    ok = db[db.ok]
    out = []
    for seg in sorted(ok.segment.unique()):
        for feed in sorted(ok.feed.unique()):
            d = ok[(ok.segment == seg) & (ok.feed == feed)]
            if not len(d):
                continue
            b = d.loc[d.F_r.idxmin()]
            m = mesh[(mesh.e_target.round(1) == seg) & (mesh.feed == feed)]
            bs = bias[(bias.segment == seg) & (bias.feed == feed)]
            bs_tail = bs[bs.grp == "tail"]
            row = dict(
                segment=seg, feed=int(feed), db_n=len(d),
                db_min_f_r=float(d.F_r.min()),
                db_best_efpd=float(b.EFPD), db_best_bu=float(b.core_mean_discharge_GWd),
                db_max_efpd=float(d.EFPD.max()),
                db_max_bu=float(d.core_mean_discharge_GWd.max()),
                db_best_pair=str(b.pair), db_best_split=str(b.assembly_composition),
                bias_n=len(bs), bias_n_tail=len(bs_tail),
                f_r_bias_tail=float((bs_tail.f_r_pred - bs_tail.f_r_true).mean())
                if len(bs_tail) else np.nan,
                f_r_bias_all=float((bs.f_r_pred - bs.f_r_true).mean()) if len(bs) else np.nan,
                f_r_mae=float((bs.f_r_pred - bs.f_r_true).abs().mean()) if len(bs) else np.nan,
                cyclen_bias=float((bs.cyclen_pred - bs.cyclen_true).mean()) if len(bs) else np.nan,
                store_min_f_r_true=float(bs.f_r_true.min()) if len(bs) else np.nan,
            )
            if len(m):
                mm = m.iloc[0]
                row.update(mesh_min_pred_f_r=float(mm.min_pred_f_r),
                           mesh_n_feasible=int(mm.n_feasible),
                           mesh_pred_cyclen=float(mm.pred_cyclen),
                           mesh_ceiling_cyclen=float(mm.max_pred_cyclen_any),
                           mesh_pair=str(mm.pair))
            else:
                row.update(mesh_min_pred_f_r=np.nan, mesh_n_feasible=-1,
                           mesh_pred_cyclen=np.nan, mesh_ceiling_cyclen=np.nan,
                           mesh_pair="")
            out.append(row)
    v = pd.DataFrame(out)
    v["corrected_floor"] = v.mesh_min_pred_f_r - v.f_r_bias_tail
    v["gap_total"] = v.mesh_min_pred_f_r - v.db_min_f_r
    v["gap_pool"] = v.corrected_floor - v.db_min_f_r
    v["gap_data"] = v.store_min_f_r_true - v.db_min_f_r      # training data blind spot
    v["gap_search"] = v.corrected_floor - v.store_min_f_r_true

    def verdict(r):
        if np.isnan(r.mesh_min_pred_f_r):
            return "not-in-mesh"
        if r.db_min_f_r > FR_LIMIT:                 # truth agrees the cell is shut
            return "model-right"
        if r.mesh_n_feasible > 0:
            return "model-right"                    # mesh already found a feasible LP
        if not np.isnan(r.corrected_floor) and r.corrected_floor <= FR_LIMIT:
            return "model-biased"                   # pessimism alone explains the miss
        return "pool-starved"

    v["verdict"] = v.apply(verdict, axis=1)
    return v


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
BLUE6 = {5.0: "#AECEEA", 5.1: "#9BC2E4", 5.2: "#88B6DE",
         5.3: "#75AAD7", 5.4: "#639DCE", 5.5: "#5290C4"}
INK, MUTED, GRAY = "#1A1A1A", "#8A8A8A", "#8C8C8C"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    return plt


def fig_mesh_vs_db(db, mesh, verdicts, pareto, path, model="s1f"):
    """Model nodes vs MASTER truth — "how far up the truth ladder did the mesh get".

    Rev 2026-08-16 (readability rebuild).  The 2026-08-15b version carried FIVE
    mark types on one axes (cloud + cell-mean diamond + best-core ring + model
    node + 미발견 square) and could not be read at a glance.  This version keeps
    the minimum vocabulary that still tells the story and splits the two questions
    onto two panels:

      left  — WHERE on the (EFPD x discharge BU) plane the model landed relative
              to the MASTER truth.  Three marks: DB best core (dark), model node
              (blue), and the connector between them.  The cell-MEAN diamonds are
              gone (one representative per cell, not two) and the 6 113-core cloud
              is a single neutral gray at alpha 0.08 so it never competes.
      right — WHICH of the 30 grid cells the mesh reached, as a 6x5 status matrix
              carrying the per-cell F_r gap.  This is where the 미발견 cells live,
              so the plane keeps no squares at all.

    feed 101 (6 DB cells) is outside the mesh grid and is excluded from both
    panels; the cloud is likewise restricted to the grid feeds so the background
    and the marks describe the same population.
    """
    plt = _mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from scoping_mesh_fig import MODEL_LABEL
    tag = MODEL_LABEL.get(model, model)

    NODE_C = "#3F73AE"          # model
    TRUTH_C = "#2B3137"         # MASTER truth
    LINK_C = "#8E949B"          # model -> truth connector
    MISS_C = "#E9ECF0"          # 미발견 cell fill
    CLOUD_C = "#C9CED4"
    LADDER_C = "#DCE0E5"

    feeds = sorted(mesh.feed.unique())
    grid = db[db.feed.isin(feeds)]
    ok = grid[grid.ok & (grid.mb_rel.abs() <= 2.0)]     # self-consistent feasible

    best = (ok.loc[ok.groupby(["segment", "feed"]).F_r.idxmin()]
              .rename(columns={"core_mean_discharge_GWd": "bu"})
              [["segment", "feed", "EFPD", "bu", "F_r"]])
    node = mesh[mesh.n_feasible > 0].copy()
    node["segment"] = node.e_target.round(1)
    found = set(zip(node.segment, node.feed))
    segs = sorted(best.segment.unique())

    fig = plt.figure(figsize=(12.5, 7.4), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.62, 1.0), wspace=0.20,
                          left=0.062, right=0.975, top=0.775, bottom=0.245)
    ax = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    for a in (ax, axm):
        a.set_facecolor("white")

    # ---------------- left: the (EFPD x discharge BU) plane ----------------- #
    ax.plot(ok.EFPD, ok.core_mean_discharge_GWd, "o", ms=2.6, alpha=0.12,
            color=CLOUD_C, mew=0, ls="none", zorder=1)

    for seg in segs:                      # hairline ladder: one rung per segment
        g = best[best.segment == seg].sort_values("feed")
        ax.plot(g.EFPD, g.bu, "-", color=LADDER_C, lw=1.4, zorder=2)
        # label above the f109 rung point: the f105 ends of 5.0 and 5.1 are 0.7
        # EFPD apart (collide) and the f121 ends sit right on top of the model
        # nodes; the f109 points are 24-38 EFPD apart and carry no node.
        r = g[g.feed == 109].iloc[0] if (g.feed == 109).any() else g.iloc[0]
        ax.annotate(f"{seg:.1f}", (r.EFPD, r.bu), textcoords="offset points",
                    xytext=(-11, 9), ha="right", va="bottom", color="#4A4A4A",
                    fontsize=12.5, fontweight="bold", zorder=9)

    # -- per-cell Pareto front, and the LIKE-WITH-LIKE connector -------------- #
    # The DB representative is that cell's LOWEST-F_r core, so the honest model
    # counterpart is our lowest-F_r FEASIBLE candidate — not the max-cyclen node.
    # Both ends of the front are drawn; the connector starts at the min-F_r end.
    for r in node.itertuples():
        g = (pareto[(pareto.e_target.round(1) == r.segment) & (pareto.feed == r.feed)]
             .sort_values("rank")) if pareto is not None else None
        if g is None or not len(g):
            src = (r.pred_cyclen, r.B_d)
        else:
            ax.plot(g.pred_cyclen, g.B_d, "-", color=NODE_C, lw=2.6, alpha=0.35,
                    zorder=5, solid_capstyle="round")
            lo = g.iloc[0]
            ax.plot(r.pred_cyclen, r.B_d, "o", ms=7.5, mfc="white", mec=NODE_C,
                    mew=2.0, zorder=8)                    # 최장주기 대표
            src = (float(lo.pred_cyclen), float(lo.B_d))
        b = best[(best.segment == r.segment) & (best.feed == r.feed)]
        if len(b):
            b = b.iloc[0]
            ax.annotate("", xy=(b.EFPD, b.bu), xytext=src,
                        arrowprops=dict(arrowstyle="-|>", color=LINK_C, lw=1.5,
                                        shrinkA=7, shrinkB=6.5, mutation_scale=12),
                        zorder=4)

    ax.plot(best.EFPD, best.bu, "o", ms=9, color=TRUTH_C, mec="white", mew=1.1,
            ls="none", zorder=6)
    if pareto is not None and len(pareto):
        mn = pareto[pareto.rep == "min_f_r"]
        ax.plot(mn.pred_cyclen, mn.B_d, "o", ms=12, color=NODE_C, mec="white",
                mew=1.6, ls="none", zorder=9)             # 최저 F_r 대표
    else:
        ax.plot(node.pred_cyclen, node.B_d, "o", ms=12, color=NODE_C, mec="white",
                mew=1.6, ls="none", zorder=9)

    ax.grid(True, color="#F2F4F6", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#CFD3D8")
        ax.spines[s_].set_linewidth(0.9)
    ax.tick_params(colors="#3C3C3C", labelsize=12, length=4, width=0.9)
    ax.set_xlabel("주기길이 (EFPD)", fontsize=13.5, color=INK, labelpad=9)
    ax.set_ylabel("평균 방출연소도 (GWd/tU)", fontsize=13.5, color=INK, labelpad=9)
    # the marks, not the cloud, set the frame — a handful of low-burnup DB cores
    # would otherwise stretch the y axis by 7 GWd/tU and flatten the whole lattice
    my = list(node.B_d) + (list(pareto.B_d) if pareto is not None else [])
    mx = list(node.pred_cyclen) + (list(pareto.pred_cyclen) if pareto is not None else [])
    ylo, yhi = min(best.bu.min(), *my), max(best.bu.max(), *my)
    xlo, xhi = min(best.EFPD.min(), *mx), max(best.EFPD.max(), *mx)
    ax.set_ylim(ylo - 0.9, yhi + 0.9)
    ax.set_xlim(xlo - 14, xhi + 14)

    # ---------------- right: 6x5 cell status matrix -------------------------- #
    v = verdicts.set_index(["segment", "feed"])
    axm.set_xlim(-0.5, len(feeds) - 0.5)
    axm.set_ylim(len(segs) - 0.5, -0.5)            # 5.0 on top
    for yi, seg in enumerate(segs):
        for xi, feed in enumerate(feeds):
            hit = (seg, feed) in found
            axm.add_patch(Rectangle((xi - 0.46, yi - 0.44), 0.92, 0.88,
                                    facecolor=NODE_C if hit else MISS_C,
                                    edgecolor="white", lw=1.4, zorder=2))
            try:
                gap = float(v.loc[(seg, feed)].mesh_min_pred_f_r
                            - v.loc[(seg, feed)].db_min_f_r)
                txt = f"{gap:+.3f}"
            except KeyError:
                txt = "—"
            axm.text(xi, yi, txt, ha="center", va="center", fontsize=11.5,
                     color="white" if hit else "#5B6169",
                     fontweight="bold" if hit else "normal", zorder=3)
    axm.set_xticks(range(len(feeds)))
    axm.set_xticklabels([f"f{f}" for f in feeds], fontsize=12, color="#3C3C3C")
    axm.set_yticks(range(len(segs)))
    axm.set_yticklabels([f"{s_:.1f}" for s_ in segs], fontsize=12, color="#3C3C3C")
    axm.xaxis.set_ticks_position("top")
    axm.tick_params(length=0)
    for s_ in axm.spines.values():
        s_.set_visible(False)
    axm.set_xlabel("셀별 F_r 격차  (모델 예측 최저 − 실측 최저)", fontsize=12.5,
                   color=INK, labelpad=13)
    axm.set_ylabel("농축 세그먼트", fontsize=12.5, color=INK, labelpad=9)

    # ---------------- chrome: title, one legend, caption --------------------- #
    n_hit, n_cell = len(node), len(segs) * len(feeds)
    fig.text(0.062, 0.965, "모델 스코핑 그물망 vs MASTER 가용노심 DB", fontsize=16.5,
             color=INK, ha="left", va="top")
    fig.text(0.062, 0.915,
             f"{tag} · 그물망 격자 {n_cell}셀 중 {n_hit}셀 도달, {n_cell - n_hit}셀 미발견 "
             f"(불가능 아님) · 실측 최량 노심은 {n_cell}셀 전부에 존재",
             fontsize=12.5, color="#3C3C3C", ha="left", va="top")
    fig.legend(handles=[
        Line2D([], [], color=NODE_C, marker="o", ms=11, lw=0, mec="white", mew=1.6,
               label=f"모델 최저 F_r 대표 ({model})"),
        Line2D([], [], color="white", marker="o", ms=8, lw=0, mec=NODE_C, mew=2.0,
               label="모델 최장주기 대표"),
        Line2D([], [], color=NODE_C, lw=2.6, alpha=0.35, label="셀 Pareto 전선"),
        Line2D([], [], color=TRUTH_C, marker="o", ms=9, lw=0, mec="white", mew=1.1,
               label="실측 최량 노심 (최저 F_r)"),
        Line2D([], [], color=LINK_C, lw=1.8, label="모델−실측 격차 (최저 F_r 기준)"),
        Line2D([], [], color=MISS_C, marker="s", ms=13, lw=0, mec="#D3D7DC",
               mew=1.0, label="실측 존재 · 모델 미발견"),
    ], loc="upper left", bbox_to_anchor=(0.058, 0.892), ncol=3, frameon=False,
        fontsize=12.0, labelcolor=INK, handletextpad=0.7, columnspacing=2.0)
    fig.text(0.062, 0.026,
             "주기길이 목표 없음 · 단일 최적 대신 안전인자 만족 집합의 Pareto 대표점 — "
             "목적함수 정의는 바뀔 수 있다 (전선 전체는 mesh_pareto.csv)\n"
             "격차는 같은 기준끼리 잰다 — DB 셀의 최저 F_r 노심 ↔ 모델의 최저 F_r 대표 "
             "(이전 판은 최장주기 노드를 이었다) · 같은 feed 셀들은 B_d ∝ cyclen/feed 이라 "
             "한 직선 위에 놓인다\n"
             "가는 회색선 = 같은 농축세그먼트의 실측 사다리 (왼쪽 = f105 → 오른쪽 = f121) · "
             "DB 게이트는 CBC ≤ 1400 ppm 으로 더 엄격 · feed 101은 그물망 격자 밖이라 제외",
             fontsize=10.5, color=MUTED, ha="left", va="bottom", linespacing=1.8)

    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_fr_model_vs_truth(bias, path, model="s1f"):
    plt = _mpl()
    from scoping_mesh_fig import MODEL_LABEL
    tag = MODEL_LABEL.get(model, model)
    feeds = sorted(bias.feed.unique())
    n = len(feeds)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(10, 7), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes = np.ravel(axes)
    for ax, feed in zip(axes, feeds):
        g = bias[bias.feed == feed]
        ax.set_facecolor("white")
        lo, hi = 1.42, 2.35
        ax.plot([lo, hi], [lo, hi], color="#C6CAD0", lw=1.0, zorder=1)
        ax.axhline(FR_LIMIT, color="#D9DDE2", lw=0.9, zorder=1)
        ax.axvline(FR_LIMIT, color="#D9DDE2", lw=0.9, zorder=1)
        for grp, col, ms in (("rand", "#BFD4E8", 3.0), ("tail", "#2E608F", 3.6)):
            s = g[g.grp == grp]
            ax.plot(s.f_r_true, s.f_r_pred, "o", ms=ms, alpha=0.6, mew=0,
                    color=col, ls="none", zorder=3)
        b = (g.f_r_pred - g.f_r_true)
        bt = (g[g.grp == "tail"].f_r_pred - g[g.grp == "tail"].f_r_true)
        ax.set_title(f"feed {feed}   n={len(g)}", fontsize=10, color=INK, loc="left", pad=6)
        ax.text(0.04, 0.95, f"bias {b.mean():+.3f}\nMAE {b.abs().mean():.3f}\n"
                            f"프런티어 bias {bt.mean():+.3f}" if len(bt) else "",
                transform=ax.transAxes, fontsize=8.2, color=MUTED, va="top")
        ax.grid(True, color="#F3F5F7", lw=0.6, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D5D8DC"); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors="#5A5A5A", labelsize=8, length=3, width=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for ax in axes[len(feeds):]:
        ax.set_visible(False)
    fig.supxlabel("MASTER 실측 F_r", fontsize=11, color=INK, y=0.055)
    fig.supylabel(f"{model} 예측 F_r", fontsize=11, color=INK)
    fig.suptitle(f"{tag} F_r 예측 정확도 — 스토어 ga80 MASTER 라벨 "
                 "(DB cid 조인 실패로 대체 측정)",
                 fontsize=12.5, color=INK, x=0.01, ha="left", y=0.975)
    fig.text(0.99, 0.010, "진한 점 = 각 셀의 최저 F_r 40기(프런티어) · 옅은 점 = 무작위 60기 · "
             "회색 선 = y=x, 회색 눈금 = F_r 1.55",
             fontsize=8.2, color=MUTED, ha="right")
    fig.tight_layout(rect=(0.03, 0.085, 1, 0.905))
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def fig_frontier_gap(v, path, model="s1f"):
    plt = _mpl()
    from scoping_mesh_fig import MODEL_LABEL
    tag = MODEL_LABEL.get(model, model)
    segs = sorted(v.segment.unique())
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes = np.ravel(axes)
    for ax, seg in zip(axes, segs):
        g = v[v.segment == seg].sort_values("feed")
        c = BLUE6.get(round(seg, 1), "#1F3864")
        ax.set_facecolor("white")
        ax.axhline(FR_LIMIT, color="#B0B5BB", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot(g.feed, g.db_min_f_r, "o-", ms=5, lw=1.8, color="#1A1A1A", zorder=4)
        ax.plot(g.feed, g.store_min_f_r_true, ":", lw=1.4, color="#7A7F86", zorder=2)
        ax.plot(g.feed, g.mesh_min_pred_f_r, "s-", ms=5, lw=1.8, color=c, zorder=3)
        ax.plot(g.feed, g.corrected_floor, "^--", ms=5, lw=1.4, color=c, alpha=0.75, zorder=3)
        ax.set_title(f"세그먼트 {seg:.1f}", fontsize=10, color=INK, loc="left", pad=6)
        ax.grid(True, color="#F3F5F7", lw=0.6, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D5D8DC"); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors="#5A5A5A", labelsize=8, length=3, width=0.8)
        ax.set_xticks(sorted(v.feed.unique()))
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color="#1A1A1A", marker="o", ms=5, lw=1.8,
               label="MASTER 실측 최저 F_r (DB)"),
        Line2D([], [], color="#3F73AE", marker="s", ms=5, lw=1.8,
               label="모델 그물망 최저 예측 F_r"),
        Line2D([], [], color="#3F73AE", marker="^", ms=5, lw=1.4, ls="--",
               alpha=0.75, label="편향 보정 후"),
        Line2D([], [], color="#7A7F86", lw=1.4, ls=":", label="스토어 실측 최저 F_r (학습자료)"),
    ], loc="upper right", ncol=4, frameon=False, fontsize=8.2,
        bbox_to_anchor=(0.995, 0.965), labelcolor=INK)
    fig.supxlabel("장전량 feed", fontsize=11, color=INK, y=0.055)
    fig.supylabel("F_r", fontsize=11, color=INK)
    fig.suptitle(f"F_r 프런티어 격차 — 실측 vs 모델 그물망 vs 편향 보정 ({tag})",
                 fontsize=12.5, color=INK, x=0.01, ha="left", y=0.975)
    fig.text(0.99, 0.010, "회색 파선 = 게이트 F_r 1.55 · 보정량 = 해당 셀 스토어 프런티어 40기의 "
             "예측−실측 평균 편향 · feed 101은 그물망 격자 밖",
             fontsize=8.2, color=MUTED, ha="right")
    fig.tight_layout(rect=(0.03, 0.085, 1, 0.905))
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--model", default="s1f", help="ensemble under data/models — "
                    "MUST match the one scoping_mesh.py used")
    args = ap.parse_args()
    logf = OUT / "comparison.log"

    def log(msg):
        print(msg, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(str(msg) + "\n")

    db = load_db()
    mesh = pd.read_csv(OUT / "mesh_nodes.csv")
    ppath = OUT / "mesh_pareto.csv"
    pareto = pd.read_csv(ppath) if ppath.exists() else None
    log(f"DB {len(db)} cores, {int(db.ok.sum())} eq_ok & feasible; "
        f"mesh {len(mesh)} cells, {int((mesh.n_feasible>0).sum())} feasible")

    # mass-balance cross-validation (no join needed)
    bd = (db.EFPD * P_TH_MW / M_HM_GA80 / 1000.0) * N_FA / db.feed
    rel = 100 * (bd - db.core_mean_discharge_GWd) / db.core_mean_discharge_GWd
    clean = rel[rel.abs() <= 2]
    log(f"mass-balance B_d vs DB discharge: {clean.mean():+.3f}% +/- {clean.std():.3f}% "
        f"(n={len(clean)}, {len(rel)-len(clean)} outliers dropped)")

    cells = set(zip(db.segment, db.feed))
    bpath = OUT / "model_bias.csv"
    if args.no_model and bpath.exists():
        bias = pd.read_csv(bpath)
    else:
        t0 = time.time()
        bias = measure_model_bias(cells, log, args.model)
        bias.to_csv(bpath, index=False)
        log(f"model bias sample: {len(bias)} store rows in {time.time()-t0:.0f}s")

    v = build_verdicts(db, mesh, bias)
    v.to_csv(OUT / "cell_verdicts.csv", index=False)
    log("\n=== verdicts ===")
    log(v.verdict.value_counts().to_string())
    log(v[["segment", "feed", "db_min_f_r", "mesh_min_pred_f_r", "f_r_bias_tail",
           "corrected_floor", "gap_total", "gap_pool", "verdict"]].to_string(index=False))

    fig_mesh_vs_db(db, mesh, v, pareto, OUT / "mesh_vs_database.png", args.model)
    fig_fr_model_vs_truth(bias, OUT / "fr_model_vs_truth.png", args.model)
    fig_frontier_gap(v, OUT / "frontier_gap.png", args.model)
    log(f"\nwrote 3 figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
