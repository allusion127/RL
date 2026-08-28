"""Mesh-v3 deliverables: the tiered reload map, the LRM three-way comparison,
and the DB-gap scoreboard.

House style follows ``scoping_mesh_fig.py`` — a complete Driscoll-style
curvilinear mesh where the safety factors are an OVERLAY, never a reason a node
is missing.  v3 adds the tier ladder: the gate is no longer one line but three
nested level sets of the same ``min_pred_f_r`` surface, so the map is read as
contours and the three tiers are named cuts through them.

    python mesh_v3_fig.py

Reads only ``data/reports/mesh_v3_20260817/``; writes only PNGs and the
scoreboard CSV there.  No MASTER, no fleet, no store writes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
OUT = BASE / "data" / "reports" / "mesh_v3_20260817"
V2 = BASE / "data" / "reports" / "scoping_mesh_20260815"

INK, MUTED, GRAY = "#1A1A1A", "#8A8A8A", "#8C8C8C"
MESH_C = "#A5ABB3"
#: one hue, three values — the tiers are nested level sets, not categories.
TIER_C = {"tier1": "#1F3864", "tier2": "#3771A4", "tier3": "#88B6DE"}
#: JOINT tiers — F_r AND CBC relax together, because relaxing F_r alone opens
#: 1 core in 11 244 (README §2b).  PREREG2 §2.
TIER_LABEL = {"tier1": "Tier-1 표준 (F_r ≤ 1.55 ∧ CBC ≤ 1600)",
              "tier2": "Tier-2 완화 (≤ 1.65 ∧ ≤ 1800)",
              "tier3": "Tier-3 관찰 (≤ 1.80 ∧ ≤ 2200)"}
#: which constraint closes a cell — the engineering content of the map
BIND_C = {"cbc_max": "#B03A2E", "f_r": "#1F3864", "f_q": "#7D6608",
          "ao_abs": "#4A4A4A"}
BIND_LABEL = {"cbc_max": "붕소 CBC", "f_r": "출력첨두 F_r",
              "f_q": "F_q", "ao_abs": "AO"}
LRM_C, SUR_C, DB_C, MAS_C = "#B8763E", "#2E608F", "#4A4A4A", "#B03A2E"
#: the anchors this study bought — drawn apart from the inherited MASTER cloud
FRESH_C = "#0F7B6C"
#: Cells whose f109-campaign verdict is 핀연소도-제한: every feasible core there
#: carries predicted max pin burnup over the 80 GWd/tU limit, so the cell is
#: "open" on F_r ∧ CBC yet undeliverable.  The tier ladder gates F_r and CBC
#: only — pin burnup is a PENDING FOURTH AXIS — so these cells must be annotated
#: or the map oversells them.  Keyed (e_target, feed) via the cell's anchor pair:
#: e5.0 -> E1_E2, e5.4 -> N1_N2.
PINBU_C = "#B9770E"
PINBU_LIMITED = {(5.0, 109): "E1_E2", (5.4, 113): "N1_N2"}
#: the hgd569 pair (`fpcamp_minfr_hgd569_f109`/`f125`, README §5e-4) is not
#: itself a mesh anchor pair, so truth_grid() carries no row for it and the
#: star logic below would never draw it.  Its f125 joint-clean frontier (all
#: gates but F_r) is the programme's closest approach yet to 5-gate
#: feasibility (F_r . CBC . F_q . AO . predicted pin BU) — worth its own
#: marker on the nearest buildable rung rather than staying invisible.
#: fpcamp_HGD569_f125_results_20260817.md §3/§6, record 8b9acbcd....
HGD569_PAIR = "P6253Z1G06N24_P6253Z2G10N24"
HGD569_CELL = (5.7, 125)          # nearest buildable rung to e_core 5.694
P_TH_MW, N_FA = 3983.0, 241
#: normally the canonical store; ``MESHV3_RECORDS`` points it at a read-only kit
#: copy so the anchors can be drawn BEFORE the merge without touching the store.
STORE = Path(os.environ.get("MESHV3_RECORDS")
             or BASE / "data" / "store" / "records.parquet")
#: ``produce`` stamps each row's ``campaign`` with its STRATUM name, not with the
#: deck-level tag in the run header, so the anchors are selected by the prefix
#: the PREREG-2 deck gives its strata.  Matching the header tag returns nothing.
CAMPAIGN_PREFIX = "mv3_"
FR_LIMIT, CBC_LIMIT = 1.55, 1600.0
#: PREREG §D — the registered history.  Derived, not hard-coded, from the v2
#: cell_verdicts so it is auditable; the three feeds the readout registered
#: (f105 +0.1663, f109 +0.0776, f113 +0.0394) must reproduce exactly, and
#: ``_baseline`` asserts they do before anything is plotted against them.
GAP_HISTORY_REGISTERED = {105: 0.1663, 109: 0.0776, 113: 0.0394}


def _baseline() -> pd.Series:
    v = pd.read_csv(V2 / "cell_verdicts.csv")
    b = v.groupby("feed").gap_total.mean()
    for f, want in GAP_HISTORY_REGISTERED.items():
        got = float(b.get(f, np.nan))
        if not np.isfinite(got) or abs(got - want) > 5e-4:
            raise SystemExit(
                f"baseline integrity check FAILED at f{f}: cell_verdicts gives "
                f"{got:.4f}, comparison_readout.md §10.3 registered {want:+.4f}. "
                "The v2 study moved under this scoreboard — stop and re-derive.")
    return b


def truth_grid(lrm: pd.DataFrame) -> pd.DataFrame:
    """Per (e_target, feed): what real MASTER cores actually reached.

    The cell's representative is its ANCHOR pair — the runnable pair closest to
    the target enrichment, exactly the one README §2 tabulates — so the truth
    panel and the §2 table are the same numbers, and the map's predicted
    contours have an honest denominator drawn on the identical lattice.

    ``n_fresh`` counts rows from the PREREG-2 deck, so a cell that this study
    actually bought can be drawn differently from one inherited from earlier
    campaigns.  Read ``fr_true_min`` as an UPPER BOUND on the achievable floor:
    none of the inherited rows came from a campaign minimising F_r.
    """

    s = pd.read_parquet(STORE)
    s = s[(s.valid == True) & (s.converged == True)]            # noqa: E712
    key = ["library_id", "case_pair", "feed"]
    g = {k: v for k, v in s.groupby(key)}
    a = lrm[lrm.is_primary][["e_target", "feed", "anchor_library_id",
                             "anchor_pair"]].drop_duplicates()
    out = []
    for r in a.itertuples():
        q = g.get((r.anchor_library_id, r.anchor_pair, int(r.feed)))
        if q is None or not len(q):
            out.append(dict(e_target=r.e_target, feed=int(r.feed),
                            anchor_pair=r.anchor_pair, n_true=0, n_fresh=0,
                            fr_true_min=np.nan, cbc_true_min=np.nan,
                            cyclen_q95=np.nan))
            continue
        fresh = int(q.campaign.astype(str).str.startswith(CAMPAIGN_PREFIX).sum())
        out.append(dict(
            e_target=r.e_target, feed=int(r.feed), anchor_pair=r.anchor_pair,
            n_true=int(len(q)), n_fresh=fresh,
            fr_true_min=float(q.f_r.min()), cbc_true_min=float(q.cbc_max.min()),
            cyclen_q95=float(q.cyclen.quantile(0.95))))
    t = pd.DataFrame(out)
    t["joint_tier"] = [
        "tier1" if (fr <= 1.55 and cb <= 1600) else
        "tier2" if (fr <= 1.65 and cb <= 1800) else
        "tier3" if (fr <= 1.80 and cb <= 2200) else "none"
        if np.isfinite(fr) and np.isfinite(cb) else ""
        for fr, cb in zip(t.fr_true_min, t.cbc_true_min)]
    return t


def fresh_anchor_rows() -> pd.DataFrame:
    """The MASTER rows this study bought, or an empty frame before the merge."""

    if not STORE.exists():
        return pd.DataFrame()
    s = pd.read_parquet(STORE)
    if "campaign" not in s.columns:
        return pd.DataFrame()
    s = s[s.campaign.astype(str).str.startswith(CAMPAIGN_PREFIX)]
    return s[(s.valid == True) & (s.converged == True)]         # noqa: E712


def hgd569_closest() -> dict | None:
    """The joint-clean (CBC . F_q . AO all pass) core with the lowest F_r for
    the hgd569 pair at f125 — read live from the store so the number moves if
    a later campaign beats it, rather than freezing this round's result."""

    if not STORE.exists():
        return None
    s = pd.read_parquet(STORE)
    q = s[(s.valid == True) & (s.converged == True)               # noqa: E712
          & (s.case_pair == HGD569_PAIR) & (s.feed == HGD569_CELL[1])]
    p = q[(q.cbc_max <= CBC_LIMIT) & (q.f_q <= 2.41) & (q.ao_abs.abs() <= 0.30)]
    if not len(p):
        return None
    r = p.loc[p.f_r.idxmin()]
    return dict(f_r=float(r.f_r), cbc=float(r.cbc_max), f_q=float(r.f_q),
                ao=float(abs(r.ao_abs)), over=float(r.f_r - FR_LIMIT))


def db_cells() -> pd.DataFrame:
    """The real MASTER-verified core database, per (enrichment segment, feed).

    ``q95(EFPD)`` matches the ceiling statistic the LRM hold-out was judged on
    (``lrm_validation_cells.csv``), so the DB series is directly comparable with
    both backbones rather than being a differently-defined maximum.  Cores are
    filtered on ``eq_ok`` only — ``feasible_at_metrics`` would pre-select on the
    very constraints under study and make the DB look better than it is.
    """

    x = pd.read_excel(V2 / "feasible_database.xlsx", "cores")
    x = x[x.eq_ok.astype(bool)]
    x["segment"] = x.enrichment_segment.round(1)
    return x.groupby(["segment", "feed"]).agg(
        n=("EFPD", "size"), db_q95=("EFPD", lambda v: float(v.quantile(0.95))),
        db_min_f_r=("F_r", "min"), db_min_cbc=("CBC_max", "min")).reset_index()


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


# --------------------------------------------------------------------------- #
# 1. the tiered reload map
# --------------------------------------------------------------------------- #
def fig_map(d: pd.DataFrame, path: Path, model="s1g",
            fresh: pd.DataFrame | None = None) -> Path:
    _setup()
    import matplotlib.pyplot as plt

    d = d.copy()
    d["x"] = d.max_pred_cyclen_any
    d["y"] = (d.x * P_TH_MW / d.M_HM_tU / 1000.0) * N_FA / d.feed
    feeds = sorted(d.feed.unique())
    targets = sorted(d.e_target.unique())
    grid = {(round(r.e_target, 2), int(r.feed)): r for r in d.itertuples()}

    fig, ax = plt.subplots(figsize=(12.4, 9.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def draw(keys):
        pts = [grid[k] for k in keys if k in grid]
        if len(pts) < 2:
            return
        ax.plot([p.x for p in pts], [p.y for p in pts], "-", color=MESH_C,
                lw=1.05, zorder=3, solid_capstyle="round")
        # a segment is drawn in the HEAVIEST tier both of its ends reach
        for a in range(len(pts) - 1):
            p, q = pts[a], pts[a + 1]
            for t in ("tier1", "tier2", "tier3"):
                if p._asdict()[f"n_feasible_{t}"] > 0 and q._asdict()[f"n_feasible_{t}"] > 0:
                    ax.plot([p.x, q.x], [p.y, q.y], "-", color=TIER_C[t],
                            lw=3.0 if t == "tier1" else 2.2 if t == "tier2" else 1.7,
                            zorder=6 if t == "tier1" else 5 if t == "tier2" else 4,
                            solid_capstyle="round")
                    break

    for e in targets:
        draw([(round(e, 2), f) for f in feeds])
    for f in feeds:
        draw([(round(e, 2), f) for e in targets])

    # nodes, coloured by the deepest tier reached
    for t in ("tier3", "tier2", "tier1"):
        m = d[f"n_feasible_{t}"] > 0
        if t != "tier1":
            for u in ("tier1", "tier2"):
                if u != t:
                    m &= ~(d[f"n_feasible_{u}"] > 0)
        if m.any():
            ax.plot(d.loc[m, "x"], d.loc[m, "y"], "o",
                    ms=10 if t == "tier1" else 8 if t == "tier2" else 6.5,
                    color=TIER_C[t], mec="white", mew=1.4, ls="none", zorder=9)
    # Cells no tier reaches are not merely "closed" — the map must say WHAT
    # closes them.  A ring in the binding constraint's colour does that without
    # adding a legend box or a second figure.
    none = d.n_feasible_tier3 <= 0
    if "binding_constraint" in d.columns:
        for bc, col in BIND_C.items():
            m = none & (d.binding_constraint == bc)
            if m.any():
                ax.plot(d.loc[m, "x"], d.loc[m, "y"], "o", ms=6.0,
                        color="white", mec=col, mew=1.8, ls="none", zorder=7)
    else:
        ax.plot(d.loc[none, "x"], d.loc[none, "y"], "o", ms=3.4, color="white",
                mec=MESH_C, mew=1.0, ls="none", zorder=7)

    # pin-burnup-limited cells: open on the two gated axes, undeliverable on the
    # ungated third.  Drawn as an amber ring + ⚠ so a Tier-1 node cannot be read
    # as "deliverable" without seeing the caveat attached to it.
    pinbu_pts = [(grid[k], k) for k in
                 ((round(e, 2), f) for e, f in PINBU_LIMITED)
                 if k in grid]
    for p, _k in pinbu_pts:
        ax.plot([p.x], [p.y], "o", ms=17, color="none", mec=PINBU_C, mew=2.4,
                ls="none", zorder=11)
        ax.annotate("⚠", (p.x, p.y), textcoords="offset points", xytext=(0, -20),
                    ha="center", va="center", color=PINBU_C, fontsize=13,
                    fontweight="bold", zorder=12)

    # the programme's closest approach yet to full 5-gate feasibility, on a
    # pair the anchor-pair grid above cannot otherwise show (HGD569_PAIR is
    # not a mesh anchor pair — see the constant's comment).
    hc = hgd569_closest()
    hk = (round(HGD569_CELL[0], 2), HGD569_CELL[1])
    if hc is not None and hk in grid:
        hp = grid[hk]
        ax.plot([hp.x], [hp.y], "D", ms=15, color="none", mec=FRESH_C, mew=2.2,
                ls="none", zorder=11)
        ax.annotate(f"4/5 게이트 · F_r +{hc['over']:.3f}", (hp.x, hp.y),
                    textcoords="offset points", xytext=(8, -26), ha="left",
                    va="top", color=FRESH_C, fontsize=10, fontweight="bold",
                    zorder=12)

    # cells this study actually bought MASTER labels for.  The PREREG-2 anchor
    # pairs are NEW pairs — they are deliberately not the mesh's anchor pair for
    # any row — so a star is placed on the rung whose e_target is nearest the
    # anchor's realised e_core, at the anchor's own feed.  Keying on pair
    # identity instead would silently drop every star.
    n_anchor = 0
    if fresh is not None and len(fresh):
        fg = fresh.groupby(["case_pair", "feed"]).e_core.mean().reset_index()
        seen = set()
        for r in fg.itertuples():
            e = min(targets, key=lambda t: abs(t - r.e_core))
            k = (round(e, 2), int(r.feed))
            if k in grid:
                seen.add(k)
        n_anchor = len(seen)
        if seen:
            pts = [grid[k] for k in seen]
            ax.plot([p.x for p in pts], [p.y for p in pts], "*", ms=22,
                    color=FRESH_C, mec="white", mew=1.5, ls="none", zorder=10)

    # min-F_r contour annotation on the iso-feed rung at the top of the grid
    hi = [grid[(round(targets[-1], 2), f)] for f in feeds if (round(targets[-1], 2), f) in grid]
    for p in hi:
        ax.annotate(f"{int(p.feed)}", (p.x, p.y), textcoords="offset points",
                    xytext=(1, 13), ha="center", color=INK, fontsize=12, zorder=11)
    if hi:
        ax.annotate("신연료 장전수", (hi[0].x, hi[0].y), textcoords="offset points",
                    xytext=(-14, 30), ha="right", color=INK, fontsize=13,
                    fontweight="bold", zorder=11)
    right = [grid[(round(e, 2), feeds[-1])] for e in targets
             if (round(e, 2), feeds[-1]) in grid]
    # The e-rungs crowd together at the top of the lattice, so the row labels
    # would overprint.  Push each label just far enough up to clear the one
    # below it and draw a hairline back to its own node, so the mesh keeps every
    # rung named without the numbers colliding.
    span = (d.y.max() - d.y.min()) or 1.0
    gap, dx = 0.030 * span, 0.012 * (d.x.max() - d.x.min())
    ys: list[float] = []
    for p in sorted(right, key=lambda q: q.y):
        y = p.y if not ys else max(p.y, ys[-1] + gap)
        ys.append(y)
    for p, y in zip(sorted(right, key=lambda q: q.y), ys):
        if abs(y - p.y) > 1e-9:
            ax.plot([p.x, p.x + dx], [p.y, y], "-", color="#C9CDD3", lw=0.8,
                    zorder=10, clip_on=False)
        ax.annotate(f"{p.e_target:.1f}", (p.x + dx, y), textcoords="offset points",
                    xytext=(4, 0), ha="left", va="center", color=INK,
                    fontsize=11.5, zorder=11)
    ax.text(0.905, 0.40, "노심 평균 농축도, w/o", transform=ax.transAxes,
            ha="center", va="center", color=INK, fontsize=13,
            fontweight="bold", zorder=11)

    # the tier ladder, named in place (house style: no legend box)
    ys = 0.965
    for t in ("tier1", "tier2", "tier3"):
        n = int((d[f"n_feasible_{t}"] > 0).sum())
        ax.plot([0.030, 0.068], [ys, ys], "-", transform=ax.transAxes,
                color=TIER_C[t], lw=3.0 if t == "tier1" else 2.2 if t == "tier2" else 1.7,
                clip_on=False, zorder=12)
        ax.text(0.078, ys, f"{TIER_LABEL[t]} — {n}셀", transform=ax.transAxes,
                ha="left", va="center", color=TIER_C[t], fontsize=11.5,
                fontweight="bold" if t == "tier1" else "normal", zorder=12)
        ys -= 0.042
    ax.text(0.078, ys, f"미달 — {int(none.sum())}셀", transform=ax.transAxes,
            ha="left", va="center", color=MUTED, fontsize=11.5, zorder=12)
    # The e6.5 rung has no coordinates to plot — the fuel lattice cannot build
    # it at all — so it is named in the ladder rather than silently absent.  A
    # missing row and an unreachable row must not look the same.
    ys -= 0.042
    ax.plot([0.030, 0.068], [ys, ys], "-", transform=ax.transAxes, color="#C9CDD3",
            lw=1.2, dashes=(3, 3), clip_on=False, zorder=12)
    ax.text(0.078, ys, "실현 불가(격자) — e6.5 행 6셀 (격자 최대 e_core 6.3645)",
            transform=ax.transAxes, ha="left", va="center", color=MUTED,
            fontsize=11.5, zorder=12)
    if pinbu_pts:
        ys -= 0.044
        ax.plot([0.046], [ys], "o", ms=14, color="none", mec=PINBU_C, mew=2.2,
                transform=ax.transAxes, clip_on=False, zorder=12)
        ax.text(0.078, ys, f"⚠ 실현 확인 · 핀연소도 제한 — {len(pinbu_pts)}셀",
                transform=ax.transAxes, ha="left", va="center", color=PINBU_C,
                fontsize=11.5, fontweight="bold", zorder=12)
    if hc is not None and hk in grid:
        ys -= 0.044
        ax.plot([0.046], [ys], "D", ms=12, color="none", mec=FRESH_C, mew=2.0,
                transform=ax.transAxes, clip_on=False, zorder=12)
        ax.text(0.078, ys, f"4/5 게이트 (F_r +{hc['over']:.3f}) — hgd569/f125 최근접",
                transform=ax.transAxes, ha="left", va="center", color=FRESH_C,
                fontsize=11.5, fontweight="bold", zorder=12)
    if n_anchor:
        ys -= 0.044
        ax.plot([0.046], [ys], "*", ms=19, color=FRESH_C, mec="white", mew=1.3,
                transform=ax.transAxes, clip_on=False, zorder=12)
        ax.text(0.078, ys, f"실측 앵커 (MASTER 신규) — {n_anchor}셀",
                transform=ax.transAxes, ha="left", va="center", color=FRESH_C,
                fontsize=11.5, fontweight="bold", zorder=12)
    if "binding_constraint" in d.columns and none.any():
        ys -= 0.050
        ax.text(0.030, ys, "닫는 제약 (테두리 색):", transform=ax.transAxes,
                ha="left", va="center", color=INK, fontsize=11, zorder=12,
                fontweight="bold")
        for bc, n in d.loc[none, "binding_constraint"].value_counts().items():
            ys -= 0.040
            ax.plot([0.042], [ys], "o", ms=6.0, color="white",
                    mec=BIND_C.get(bc, MESH_C), mew=1.8, transform=ax.transAxes,
                    clip_on=False, zorder=12)
            ax.text(0.078, ys, f"{BIND_LABEL.get(bc, bc)} — {int(n)}셀",
                    transform=ax.transAxes, ha="left", va="center",
                    color=BIND_C.get(bc, MUTED), fontsize=11.5, zorder=12,
                    fontweight="bold" if bc == "cbc_max" else "normal")

    _chrome(ax, "예측 주기길이 (EFPD)", "평균 방출연소도 (GWd/tU, 질량수지)")
    ax.set_xlim(d.x.min() - 22, d.x.max() + 52)
    ax.set_ylim(d.y.min() - 2.4, d.y.max() + 2.4)
    fig.tight_layout(rect=(0, 0.185, 1, 0.915))
    fig.text(0.052, 0.980,
             f"APR1400 LEU+ 고농축 확장 재장전 설계지도 v3 (e_core 5.0–6.5 × feed 109–129) — {model} 예측",
             fontsize=15.5, color=INK, ha="left", va="top")
    fig.text(0.052, 0.944,
             f"{len(targets)}×{len(feeds)} = {len(d)}개 격자점 전부 표시 · "
             "안전인자는 지도를 지우지 않고 그 위에 겹쳐 읽는다 · "
             "격자점은 안전인자 미적용 대표점(셀 내 최대 예측 주기길이)",
             fontsize=11.5, color="#3C3C3C", ha="left", va="top")
    fig.text(0.052, 0.020,
             "계층은 F_r 과 CBC 를 «함께» 완화한다 — F_r 만 풀면 실측 11,244 고농축 노심 중 1개만 열리기 "
             "때문이다(README §2b). F_q ≤ 2.41 · |AO| ≤ 0.30 은 전 계층 불변.\n"
             "Tier-3 은 관찰 전용이며 인허가 주장의 근거가 아니다.\n"
             "어느 계층에도 못 드는 셀은 그 셀을 닫는 제약의 색으로 테두리를 그렸다 — "
             "붉은 테두리는 «이 셀을 닫는 것은 붕소다» 를 뜻한다.\n"
             "계층은 F_r ∧ CBC 만 건다 — 핀연소도(max pin BU ≤ 80 GWd/tU)는 아직 게이트가 아닌 «네 번째 축»이며, "
             "저feed 셀 2곳은 그 축에서 납품 불가로 판정됐다(⚠).\n"
             "격자점은 모델 예측 — 별표 셀만 MASTER 실측 앵커가 붙어 있다. "
             "e6.5 행은 연료 격자상 실현 불가(50/50 pair 최대 e_core = 6.3645).",
             fontsize=10.2, color=MUTED, ha="left", va="bottom", linespacing=1.75)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 2. LRM vs surrogate vs DB (vs MASTER anchors when they land)
# --------------------------------------------------------------------------- #
def fig_lrm(d: pd.DataFrame, lrm: pd.DataFrame, verdict: dict, path: Path,
            db: pd.DataFrame | None = None, val: pd.DataFrame | None = None,
            fresh: pd.DataFrame | None = None) -> Path:
    _setup()
    import matplotlib.pyplot as plt

    m = lrm.merge(d[["cell", "max_pred_cyclen_any", "min_pred_f_r"]], on="cell",
                  how="left")
    m = m[m.is_primary]
    feeds = sorted(m.feed.unique())
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.4), dpi=150,
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    fig.patch.set_facecolor("white")

    # -- panel A: the two backbones against enrichment, per feed ------------- #
    ax = axes[0]
    for f in feeds:
        g = m[m.feed == f].sort_values("e_core")
        ax.plot(g.e_core, g.lrm_ceil_efpd, "-", color=LRM_C, lw=1.5, alpha=0.85,
                zorder=3)
        ax.plot(g.e_core, g.max_pred_cyclen_any, "--", color=SUR_C, lw=1.4,
                alpha=0.85, zorder=4)
        if len(g):
            ax.annotate(f"{int(f)}", (g.e_core.iloc[-1], g.lrm_ceil_efpd.iloc[-1]),
                        textcoords="offset points", xytext=(6, 0), ha="left",
                        va="center", fontsize=9.5, color=GRAY)
    # the real feasible-core DB — the population the LRM was fitted on, shown as
    # itself rather than only through the fit, so "the LRM is accurate" can be
    # checked against the data it came from as well as against the hold-out.
    if db is not None and len(db):
        q = db[db.feed.isin(feeds)]
        if len(q):
            ax.plot(q.segment, q.db_q95, "s", ms=5.0, color="white", mec=DB_C,
                    mew=1.3, ls="none", zorder=5)
    if val is not None and len(val):
        ax.plot(val.e, val.q95, "o", ms=4.5, color=MAS_C, mec="white", mew=0.7,
                ls="none", zorder=6)
    # the anchors this study bought, drawn apart from the inherited cloud
    if fresh is not None and len(fresh):
        fg = fresh.groupby(["case_pair", "feed"]).agg(
            e=("e_core", "mean"), q95=("cyclen", lambda v: float(v.quantile(0.95))))
        ax.plot(fg.e, fg.q95, "D", ms=7.0, color=FRESH_C, mec="white", mew=1.2,
                ls="none", zorder=8)
    ys = 0.965
    for c, lab, on in ((LRM_C, "LRM ceiling 백본 (실선)", True),
                       (SUR_C, "대리모델 상한 (파선)", True),
                       (DB_C, "실제 DB q95 (□, 6,113 노심)",
                        db is not None and len(db) > 0),
                       (MAS_C, "저장소 MASTER q95 (●)", True),
                       (FRESH_C, "신규 앵커 MASTER q95 (◆)",
                        fresh is not None and len(fresh) > 0)):
        if not on:
            continue
        ax.text(0.035, ys, lab, transform=ax.transAxes, color=c, fontsize=11.0,
                fontweight="bold", va="top")
        ys -= 0.047
    ax.axvline(5.5, color=GRAY, lw=0.9, ls=":", zorder=2)
    ax.text(5.51, ax.get_ylim()[0] + 6, "  DB 적합 상한 →  외삽 구간",
            fontsize=10, color=GRAY, va="bottom")
    _chrome(ax, "노심 평균 농축도 (w/o)", "주기길이 (EFPD)")

    # -- panel B: hold-out residuals ----------------------------------------- #
    ax = axes[1]
    if val is not None and len(val):
        for held, c, lab in ((False, "#9AA5B1", "적합 범위 (e ≤ 5.5)"),
                             (True, MAS_C, "홀드아웃 (e > 5.55)")):
            g = val[val.held_out == held]
            ax.plot(g.e, 100 * g.resid_frac, "o", ms=5, color=c, mec="white",
                    mew=0.6, ls="none", zorder=4 if held else 3, label=lab)
        ax.axhline(0, color=INK, lw=1.0, zorder=2)
        for s in (-2, 2):
            ax.axhline(s, color=GRAY, lw=0.8, ls="--", zorder=2)
        # the fresh anchors are a SECOND hold-out, drawn from pairs no earlier
        # campaign ever ran — the strictest test the backbone has faced.
        fr_txt = ""
        if fresh is not None and len(fresh):
            co = verdict["coefficients"]
            ac = {int(k): v for k, v in co["alpha_ceil"].items()}
            fg = fresh.groupby(["case_pair", "feed"]).agg(
                e=("e_core", "mean"),
                q95=("cyclen", lambda v: float(v.quantile(0.95)))).reset_index()
            fg["lrm"] = [ac.get(int(f), np.nan)
                         * (co["a_ceil"] + co["b_ceil"] * e)
                         for f, e in zip(fg.feed, fg.e)]
            fg["rf"] = (fg.q95 - fg.lrm) / fg.lrm
            ax.plot(fg.e, 100 * fg.rf, "D", ms=7.0, color=FRESH_C, mec="white",
                    mew=1.2, ls="none", zorder=6, label="신규 앵커 (2차 홀드아웃)")
            fr_txt = (f"\n신규 앵커 {len(fg)}셀: 편향 {100*fg.rf.mean():+.2f}% · "
                      f"rms {100*np.sqrt((fg.rf ** 2).mean()):.2f}%")
        h = verdict["holdout"]
        ax.text(0.035, 0.055,
                f"홀드아웃 편향 {100*h['bias_frac']:+.2f}% · rms {100*h['rms_frac']:.2f}%\n"
                f"n = {h['n_cells']}셀, e {h['e_lo']:.2f}–{h['e_hi']:.2f}\n"
                f"판정: {h['verdict']}" + fr_txt,
                transform=ax.transAxes, fontsize=11, color=INK, va="bottom",
                linespacing=1.6, fontweight="bold")
        ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    _chrome(ax, "노심 평균 농축도 (w/o)", "(MASTER q95 − LRM ceiling) / LRM  [%]")

    # -- panel C: surrogate minus LRM, the anchoring signal ------------------ #
    ax = axes[2]
    for f in feeds:
        g = m[m.feed == f].sort_values("e_core")
        ax.plot(g.e_core, g.max_pred_cyclen_any - g.lrm_ceil_efpd, "o-", ms=4,
                lw=1.3, alpha=0.9, zorder=3, label=f"f{int(f)}")
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    ax.legend(loc="best", frameon=False, fontsize=10, ncol=2)
    _chrome(ax, "노심 평균 농축도 (w/o)", "대리모델 − LRM  (EFPD)")

    fig.tight_layout(rect=(0, 0.075, 1, 0.90))
    fig.text(0.035, 0.975,
             "LRM 백본 · 대리모델 · MASTER 실측 비교 — 사용자 가설 «LRM이 꽤나 정확할 것» 의 정량 심판",
             fontsize=15, color=INK, ha="left", va="top")
    fig.text(0.035, 0.933,
             f"LRM 은 DB 6,113 노심(e 5.0–5.5, feed 101–121)에만 적합됐고, "
             f"검정은 그 밖의 e {verdict['holdout']['e_lo']:.2f}–{verdict['holdout']['e_hi']:.2f} "
             f"저장소 MASTER 셀에서 이뤄진 홀드아웃이다 · "
             "네 계열을 한 축에 겹쳐 읽는다: LRM · 대리모델 · 실제 DB · MASTER(기존+신규 앵커)",
             fontsize=11.5, color="#3C3C3C", ha="left", va="top")
    fig.text(0.035, 0.018,
             "정확도 언급은 Spearman 우선 어법을 따른다 — 여기 수치는 절대오차이며, 순위정확도가 아니라 "
             "«수준»의 재현도를 보고한다.\n"
             f"농축도 기울기 오차 {verdict['slope_error_pct']:+.1f}% "
             f"(95% CI {verdict['slope_ci_pct'][0]:+.1f}…{verdict['slope_ci_pct'][1]:+.1f}%) — "
             "양수는 LRM 이 고농축에서 이득을 과소평가한다는 뜻, 즉 백본이 보수적이다.",
             fontsize=10.2, color=MUTED, ha="left", va="bottom", linespacing=1.75)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 3. DB-gap scoreboard
# --------------------------------------------------------------------------- #
def scoreboard(d: pd.DataFrame, path_csv: Path, path_png: Path) -> pd.DataFrame:
    """``gap_total = mesh_min_pred_f_r - db_min_f_r`` per feed, exactly the
    definition ``mesh_vs_db.py:149`` uses, so the numbers are comparable with
    the registered history.

    comparison_readout.md §10.3 recorded that a ``--model`` switch alone cannot
    move this number because it is computed from the ON-DISK mesh — the mesh
    must be regenerated first, and it left that instruction for the next
    pre-registration.  v3 regenerates the mesh on s1g, so this scoreboard is a
    real re-measurement rather than a replay of the baseline."""

    _setup()
    import matplotlib.pyplot as plt

    # --- reproduction check ------------------------------------------------ #
    # The v2 baseline mesh was ALSO drawn with s1g, on the same pairs, the same
    # per-cell RNG seed and the same store (the f109 campaign has not merged).
    # So the 24 overlapping cells must come back bit-for-bit.  If they do not,
    # something moved underneath this study and the gap numbers are not
    # comparable with the registered history — say so loudly rather than
    # quietly plotting a difference that is really a drift.
    v2 = pd.read_csv(V2 / "mesh_nodes.csv")[["cell", "min_pred_f_r"]]
    rep = d[["cell", "min_pred_f_r"]].merge(v2, on="cell", suffixes=("_v3", "_v2"))
    if len(rep):
        dd = (rep.min_pred_f_r_v3 - rep.min_pred_f_r_v2).abs()
        print(f"[reproduction] {len(rep)} overlapping cells vs the v2 s1g mesh: "
              f"max |Δ min_pred_f_r| = {dd.max():.6f}, "
              f"n(|Δ| > 1e-6) = {int((dd > 1e-6).sum())}")

    v = pd.read_csv(V2 / "cell_verdicts.csv")[["segment", "feed", "db_min_f_r"]]
    m = d.merge(v, left_on=["e_target", "feed"], right_on=["segment", "feed"],
                how="inner")
    m["gap_total"] = m.min_pred_f_r - m.db_min_f_r
    tab = m.groupby("feed").agg(n_cells=("gap_total", "size"),
                                gap_mean=("gap_total", "mean"),
                                gap_min=("gap_total", "min"),
                                gap_max=("gap_total", "max")).reset_index()
    # every v3 column gets a row, including the two the DB never covered.  A
    # column dropped for lack of a denominator is a fact about the DB, not a
    # result, and silently omitting it would read as "no gap there".
    allf = pd.DataFrame({"feed": sorted(d.feed.unique())})
    tab = allf.merge(tab, on="feed", how="left")
    tab["n_cells"] = tab.n_cells.fillna(0).astype(int)
    tab["history_s1g_v2mesh"] = tab.feed.map(_baseline())
    tab["delta_vs_history"] = tab.gap_mean - tab.history_s1g_v2mesh
    tab["db_coverage"] = np.where(tab.n_cells > 0, "DB 포괄", "DB 미포괄")
    tab.round(4).to_csv(path_csv, index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10.6, 7.0), dpi=150)
    fig.patch.set_facecolor("white")
    x = np.arange(len(tab))
    ax.bar(x - 0.19, tab.history_s1g_v2mesh.fillna(0.0), 0.36, color="#C9D3DE",
           label="기준선 (s1g × v2 그물망)", zorder=3)
    ax.bar(x + 0.19, tab.gap_mean.fillna(0.0), 0.36, color=SUR_C,
           label="v3 (s1g × 재생성 그물망)", zorder=3)
    for i, r in enumerate(tab.itertuples()):
        if np.isfinite(r.gap_mean):
            ax.annotate(f"{r.gap_mean:+.3f}", (i + 0.19, r.gap_mean),
                        textcoords="offset points",
                        xytext=(0, 4 if r.gap_mean >= 0 else -13), ha="center",
                        fontsize=10, color=SUR_C, fontweight="bold", zorder=5)
        else:
            ax.annotate("DB 미포괄\n(비교 불가)", (i, 0), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, color=MUTED,
                        zorder=5, linespacing=1.5)
    ax.axhline(0, color=INK, lw=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{int(f)}" for f in tab.feed], fontsize=12)
    ax.legend(loc="best", frameon=False, fontsize=11)
    _chrome(ax, "", "gap_total  (예측 F_r 바닥 − DB 실측 바닥)")
    dl = tab.delta_vs_history.dropna()
    if len(dl):
        ax.text(0.985, 0.06,
                f"재생성 후 기준선 대비 최대 |Δ| = {dl.abs().max():.4f}\n"
                "→ 두 막대가 같은 것이 결과다",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                color=INK, fontweight="bold", linespacing=1.6, zorder=6)
    fig.tight_layout(rect=(0, 0.235, 1, 0.905))
    fig.text(0.035, 0.978, "DB 갭 스코어보드 — 열별 평균 (낮을수록 좋다)",
             fontsize=15, color=INK, ha="left", va="top")
    fig.text(0.035, 0.943,
             "정의는 mesh_vs_db.py:149 와 동일 · 음수는 모델이 DB 실측보다 낙관적이라는 뜻",
             fontsize=11, color="#3C3C3C", ha="left", va="top")
    fig.text(0.035, 0.016,
             "기준선은 v2 그물망 위의 s1g 값(comparison_readout.md §10.3). §10.3 은 «--model 만 바꾸면\n"
             "갭이 안 움직이니 다음 프리레지는 그물망 재생성을 명시하라» 고 남겼고, v3 는 s1g 로 그물망을\n"
             "재생성했다 — 그런데 겹치는 44셀이 bit 단위로 같아 갭도 그대로다. 갭은 (모델 × 후보풀)의 성질이지\n"
             "그물망 재실행의 성질이 아니다. f105 는 v3 격자에서 빠졌다. f109 캠페인(E1_E2 opening +100·\n"
             "hgd569 +120, §5e-4)은 이제 병합됐지만 gap_total 은 그대로다 — 병합은 후보풀 자체를 바꾸지\n"
             "않는다는 §6a 결론이 재확인된 것이지, 갱신이 빠진 것이 아니다.\n"
             "f125·f129 는 DB(feed 101–121)가 포괄하지 않아 분모가 없다 — 갭 0 이 아니라 «비교할 실측 프런티어가 없다».",
             fontsize=10.0, color=MUTED, ha="left", va="bottom", linespacing=1.7)
    fig.savefig(path_png, dpi=150, facecolor="white")
    plt.close(fig)
    return tab


def fig_contour(d: pd.DataFrame, path: Path, truth: pd.DataFrame | None = None) -> Path:
    """min-achievable-F_r as an actual CONTOUR field over (e_core, feed).

    The tier ladder on the reload map is three named level sets of this same
    surface; this panel shows the surface itself, which is what makes the map
    readable as a design gradient rather than a pass/fail stencil.  The right
    panel is the MASTER truth for the cells that have labels — the honest
    denominator for every predicted contour on the left."""

    _setup()
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    # every panel is reindexed onto the SAME lattice as the mesh.  Without this
    # a truth row that has no labels anywhere (e5.6 / e5.7 / e6.0) silently
    # vanishes, the two rows of panels stop lining up, and the reader compares
    # a predicted e6.1 against a measured e6.2.
    rows = sorted(d.e_target.unique())
    cols = sorted(d.feed.unique())

    def piv(frame, col):
        return (frame.pivot_table(index="e_target", columns="feed", values=col)
                .reindex(index=rows, columns=cols))

    panels = [("대리모델 예측 최소 F_r", piv(d, "min_pred_f_r"), "fr")]
    if "min_cbc_max" in d.columns:
        panels.append(("대리모델 예측 최소 CBC_max [ppm]", piv(d, "min_cbc_max"), "cbc"))
    if truth is not None and len(truth):
        panels.append(("MASTER 실측 최소 F_r (앵커 pair)", piv(truth, "fr_true_min"), "fr"))
        if "cbc_true_min" in truth.columns:
            panels.append(("MASTER 실측 최소 CBC_max [ppm] (앵커 pair)",
                           piv(truth, "cbc_true_min"), "cbc"))
    # 2 columns keeps the predicted row directly above the measured row, so the
    # eye compares F_r with F_r and boron with boron rather than left-to-right.
    ncol = 2 if len(panels) > 2 else len(panels)
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.6 * ncol, 6.6 * nrow),
                             dpi=150, squeeze=False)
    flat = [a for row in axes for a in row]
    for a in flat[len(panels):]:
        a.set_visible(False)
    fig.patch.set_facecolor("white")
    # one normalisation per quantity across BOTH rows, so a colour means the
    # same thing in the predicted panel and in the measured panel below it.
    # ``YlGnBu_r`` runs DARK at the low end, so the cell that needs white text is
    # the LOW one.  ``lo_dark`` is the value below which the fill is dark enough
    # to swallow ink — getting this backwards makes exactly the frontier cells
    # (the lowest F_r, the lowest boron) the unreadable ones.
    SPEC = {"fr": dict(norm=mcolors.Normalize(1.45, 2.20),
                       levels=[1.55, 1.65, 1.80],
                       fmt={1.55: "T1 1.55", 1.65: "T2 1.65", 1.80: "T3 1.80"},
                       txt="{:.2f}", lo_dark=1.80, cbar="최소 F_r"),
            "cbc": dict(norm=mcolors.Normalize(1000, 2800),
                        levels=[1600.0, 1800.0, 2200.0],
                        fmt={1600.0: "T1 1600", 1800.0: "T2 1800",
                             2200.0: "T3 2200"},
                        txt="{:.0f}", lo_dark=1850.0, cbar="최소 CBC_max [ppm]")}
    from matplotlib import patheffects as pe
    halo = [pe.withStroke(linewidth=2.4, foreground="#1A1A1A", alpha=0.55)]
    for ax, (title, P, kind) in zip(flat, panels):
        sp = SPEC[kind]
        Z = P.to_numpy(float)
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="YlGnBu_r",
                       norm=sp["norm"],
                       extent=(-0.5, Z.shape[1] - 0.5, -0.5, Z.shape[0] - 0.5))
        # the three registered tier limits as real contours.  White with a dark
        # halo is the only ink that survives both ends of this colormap.
        M = np.ma.masked_invalid(Z)
        if M.count() > 4:
            cs = ax.contour(M, levels=sp["levels"], colors="white",
                            linewidths=[2.6, 2.0, 1.5],
                            linestyles=["solid", "dashed", "dotted"], zorder=4)
            # matplotlib >= 3.8 makes ContourSet a single artist; older versions
            # expose the per-level collections instead.
            for art in getattr(cs, "collections", [cs]):
                art.set_path_effects(halo)
            lb = ax.clabel(cs, fmt=sp["fmt"], fontsize=9.5, inline=True,
                           colors="white")
            for t in lb:
                t.set_path_effects(halo)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, sp["txt"].format(Z[i, j]), ha="center",
                            va="center", fontsize=8.0, zorder=6,
                            color="white" if Z[i, j] < sp["lo_dark"] else INK)
        ax.set_xticks(range(Z.shape[1]))
        ax.set_xticklabels([str(int(c)) for c in P.columns], fontsize=11)
        ax.set_yticks(range(Z.shape[0]))
        ax.set_yticklabels([f"{v:.1f}" for v in P.index], fontsize=11)
        ax.set_title(title, fontsize=13, color=INK, pad=10)
        ax.set_xlabel("신연료 장전수 (feed)", fontsize=12.5, color=INK, labelpad=8)
        ax.set_ylabel("노심 평균 농축도 (w/o)", fontsize=12.5, color=INK, labelpad=8)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(
            sp["cbar"], fontsize=11.5, color=INK)
    H = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0.86 / H, 1, 1 - 1.05 / H), w_pad=3.2, h_pad=2.4)
    fig.text(0.035, 1 - 0.24 / H,
             "달성 가능 최소 F_r · 최소 CBC 등고선 — 계층은 이 두 곡면의 이름 붙은 등위집합이다",
             fontsize=15, color=INK, ha="left", va="top")
    fig.text(0.035, 1 - 0.62 / H,
             "위 행 = 대리모델 예측 · 아래 행 = MASTER 실측(앵커 pair) · "
             "지도는 이진 판정이 아니라 등고선으로 읽는다 · 빈 칸 = 그 (pair, feed) 에 실측 라벨 없음",
             fontsize=11.5, color="#3C3C3C", ha="left", va="top")
    fig.text(0.035, 0.16 / H,
             "⚠ F_r 은 고농축의 병목이 아니다 — CBC ≤ 1600 ppm 이 먼저 닫는다 (README §2b). "
             "두 축을 함께 읽어야 «열림»을 말할 수 있다 — 어느 한 판만으로는 아니다.\n"
             "실측 판의 값은 «바닥»이 아니라 «바닥의 상한»이다: 그 행들은 F_r 을 최소화하려던 캠페인의 "
             "산물이 아니라 다른 목적의 탐색이 남긴 부산물이다.",
             fontsize=10.4, color=MUTED, ha="left", va="bottom", linespacing=1.7)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    import json

    d = pd.read_csv(OUT / "mesh_nodes.csv")
    lrm = pd.read_csv(OUT / "lrm_backbone.csv")
    verdict = json.load(open(OUT / "lrm_verdict.json", encoding="utf-8"))
    val = pd.read_csv(OUT / "lrm_validation_cells.csv")

    truth = truth_grid(lrm)
    truth.round(4).to_csv(OUT / "master_truth_grid.csv", index=False,
                          encoding="utf-8")
    fresh = fresh_anchor_rows()
    db = db_cells()
    print(f"[truth] {int((truth.n_true > 0).sum())}/{len(truth)} cells have "
          f"MASTER labels; joint-tier counts "
          f"{truth.joint_tier.value_counts().to_dict()}")
    print(f"[fresh] {len(fresh)} converged rows from strata '{CAMPAIGN_PREFIX}*'"
          + (f" over {fresh.groupby(['case_pair','feed']).ngroups} cells"
             if len(fresh) else " — anchors not merged yet"))
    print(f"[db]    {int(db.n.sum())} eq_ok cores over {len(db)} (segment, feed) cells")

    print(fig_map(d, OUT / "mesh_v3_map.png", fresh=fresh))
    print(fig_contour(d, OUT / "min_fr_contour.png", truth=truth))
    print(fig_lrm(d, lrm, verdict, OUT / "lrm_vs_surrogate.png", db=db, val=val,
                  fresh=fresh))
    tab = scoreboard(d, OUT / "db_gap_scoreboard.csv", OUT / "db_gap_scoreboard.png")
    print(tab.round(4).to_string(index=False))
    print(OUT / "db_gap_scoreboard.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
