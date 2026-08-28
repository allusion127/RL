"""LRM backbone mesh over the v3 grid — the ZERO-MASTER comparison layer.

The calibrated Linear-Reactivity-Model fitted in ``dbx/make_lrm.py`` to 6 113
real database cores is evaluated on the SAME (e_core x feed) grid that
``scoping_mesh.py`` searches with the CNN ensemble, so the two can be laid on
top of each other cell by cell.  Nothing here runs MASTER and nothing under
``data/store`` is written.

Three things happen, in this order (the order is the point):

1. **alpha extrapolation, adjudicated against real MASTER.**  The DB fit only
   covers feeds 101-121; the v3 grid needs 125 and 129.  Five candidate
   functional forms are extrapolated and then judged against the store's own
   converged rows via a within-pair ratio that cancels enrichment, library and
   level effects.  The pre-registered rule (PREREG §1b) is that the form with
   the smallest deviation from the measured f125/f129 ratio wins, and that
   leave-one-out is reported but NOT used to choose — LOO measures
   interpolation, and this is an extrapolation.

2. **the backbone itself** — mean and ceiling EFPD per cell, then the same
   equilibrium mass balance ``scoping_mesh`` uses, so B_d is comparable.

3. **the hold-out verdict on the user's hypothesis.**  The DB fit never saw
   e > 5.55 or feed > 121.  The store holds ~10 000 converged paramA rows at
   e 5.79-6.36 and feeds up to 125.  Predicting those is a genuine hold-out
   over an 0.86 wt% enrichment extrapolation, and PREREG §1d fixed the
   pass/fail bands before the number was computed.

    python lrm_mesh.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from scoping_mesh import (E_TARGETS, FEEDS, N_FA, P_TH_MW,  # noqa: E402
                          pair_hm_tu, select_pairs)

OUT = BASE / "data" / "reports" / "mesh_v3_20260817"
FIT = BASE / "data" / "reports" / "dbx_lrm_fit.csv"

#: the DB fit's coverage — everything outside this is extrapolation and is
#: flagged as such in the output.
DB_E_MAX, DB_FEED_MAX = 5.5, 121
#: store-cell admission for the adjudication and hold-out: enough rows that the
#: 95th percentile is a ceiling and not one lucky draw.
MIN_ROWS = 25
#: PREREG §1d verdict bands, |bias| and rms as a FRACTION of predicted EFPD.
VERDICT_BANDS = ((0.02, 0.03, "확증 (confirmed)"),
                 (0.05, 0.06, "부분 확증 (partially confirmed)"))


# --------------------------------------------------------------------------- #
# 1. alpha extrapolation
# --------------------------------------------------------------------------- #
def _designs(feed: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate design matrices for alpha(feed).  ``n_b = 241/feed`` is the
    batch count, so ``1/(n_b+1)`` is the analytic linear-reactivity shape."""

    f = np.asarray(feed, float)
    inv = 1.0 / (N_FA / f + 1.0)
    one = np.ones_like(f)
    return {"lin_feed": np.c_[one, f],
            "quad_feed": np.c_[one, f, f ** 2],
            "inv_nb1": np.c_[one, inv],
            "inv_nb1_quad": np.c_[one, inv, inv ** 2],
            "log_feed": np.c_[one, np.log(f)]}


def store_feed_ratio(store: pd.DataFrame, feeds=FEEDS, ref: int = 117):
    """Measured feed response: per pair, ``q95(cyclen)`` at each feed divided by
    the same pair's value at ``ref``.

    Holding the pair fixed cancels enrichment, library, u_mass and the absolute
    level, leaving only what the feed does — which is exactly what alpha is.
    Returned as (mean, sd, n_pairs) per feed."""

    g = (store[store.feed.isin(feeds)]
         .groupby(["library_id", "case_pair", "feed"])
         .agg(n=("cyclen", "size"),
              q95=("cyclen", lambda v: float(np.percentile(v, 95)))).reset_index())
    g = g[g.n >= MIN_ROWS]
    rows = []
    for (lib, pair), h in g.groupby(["library_id", "case_pair"]):
        h = h.set_index("feed")
        if ref not in h.index or len(h) < 4:
            continue
        for f in h.index:
            rows.append(dict(lib=lib, pair=pair, feed=int(f),
                             ratio=float(h.q95[f] / h.q95[ref])))
    R = pd.DataFrame(rows)
    return R.groupby("feed").ratio.agg(["mean", "std", "count"])


def fit_alpha(alpha: dict[int, float], measured: pd.DataFrame, want=(125, 129),
              ref: int = 117, log=print) -> tuple[dict[int, float], str, pd.DataFrame]:
    """Extrapolate ``alpha`` to ``want``, choosing the form by PREREG §1b."""

    F = np.array(sorted(alpha), float)
    y = np.array([alpha[int(f)] for f in F])
    X = _designs(F)
    Xt = _designs(np.array(want, float))
    rows = []
    for name in X:
        c = np.linalg.lstsq(X[name], y, rcond=None)[0]
        loo = []
        for i in range(len(F)):                 # reported, deliberately not used
            m = np.ones(len(F), bool)
            m[i] = False
            if m.sum() <= X[name].shape[1]:
                continue
            ci = np.linalg.lstsq(X[name][m], y[m], rcond=None)[0]
            loo.append(X[name][i] @ ci - y[i])
        ext = dict(alpha) | {int(f): float(v) for f, v in zip(want, Xt[name] @ c)}
        # deviation from the MEASURED ratio at the extrapolated feeds
        dev = [abs(ext[int(f)] / ext[ref] / measured["mean"][int(f)] - 1.0)
               for f in want if int(f) in measured.index]
        rows.append(dict(form=name, n_param=X[name].shape[1],
                         loo_rms=float(np.sqrt(np.mean(np.square(loo)))) if loo else np.nan,
                         **{f"alpha_{int(f)}": ext[int(f)] for f in want},
                         max_dev_vs_store=float(np.max(dev)) if dev else np.nan,
                         mean_dev_vs_store=float(np.mean(dev)) if dev else np.nan))
    T = pd.DataFrame(rows).sort_values("max_dev_vs_store")
    best = str(T.iloc[0].form)
    c = np.linalg.lstsq(X[best], y, rcond=None)[0]
    ext = dict(alpha) | {int(f): float(v) for f, v in zip(want, Xt[best] @ c)}
    log("\n--- alpha extrapolation adjudication (PREREG §1b) ---")
    log(T.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    log(f"CHOSEN: {best}  (smallest max deviation vs the store's measured "
        f"feed ratio; LOO would have chosen "
        f"{T.sort_values('loo_rms').iloc[0].form})")
    return ext, best, T


# --------------------------------------------------------------------------- #
# 2/3. backbone + hold-out verdict
# --------------------------------------------------------------------------- #
def verdict(bias_frac: float, rms_frac: float) -> str:
    for b, r, name in VERDICT_BANDS:
        if abs(bias_frac) < b and rms_frac < r:
            return name
    return "기각 (rejected)"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / "lrm_mesh.log"
    logf.write_text("", encoding="utf-8")

    def log(msg="") -> None:
        print(msg, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(str(msg) + "\n")

    from lpopt.data.fuel_types import FuelLibrary

    fit = pd.read_csv(FIT)
    aB, bB = float(fit.lrmB_a.iloc[0]), float(fit.lrmB_b.iloc[0])
    aC, bC = float(fit.lrmCeil_a.iloc[0]), float(fit.lrmCeil_b.iloc[0])
    alphaB = {int(k): float(v) for k, v in fit.groupby("feed").lrmB_alpha.first().items()}
    alphaC = {int(k): float(v) for k, v in fit.groupby("feed").lrmCeil_alpha.first().items()}
    log(f"LRM coefficients from {FIT.name} (dbx/make_lrm.py, 6113-core fit)")
    log(f"  mean    EFPD = alphaB(F) * ({aB:+.6f} {bB:+.6f}*e)")
    log(f"  ceiling EFPD = alphaC(F) * ({aC:+.6f} {bC:+.6f}*e)")
    log(f"  fitted feeds {sorted(alphaB)}, fitted e range <= {DB_E_MAX}")

    fuel_df = pd.read_parquet(BASE / "data/store/fuel_types.parquet")
    fuel = FuelLibrary.from_parquet(BASE / "data/store/fuel_types.parquet")
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    store = store[(store.valid == True) & (store.converged == True)]   # noqa: E712

    measured = store_feed_ratio(store)
    log("\n--- measured feed response from the store (within-pair q95 ratio, ref f117) ---")
    log(measured.round(4).to_string())
    alphaB_x, formB, tabB = fit_alpha(alphaB, measured, log=log)
    alphaC_x, formC, tabC = fit_alpha(alphaC, measured, log=log)

    # ---------------------------------------------------------------- grid ---
    picks = select_pairs(fuel, fuel_df, store, lambda _m: None)
    picks["per_fa_tU"], picks["M_HM_tU"] = zip(*[
        pair_hm_tu(fuel_df, r.library_id, r.type_a, r.type_b) for r in picks.itertuples()])
    key = picks.library_id + "|" + picks.pair
    picks["is_primary"] = ~key.duplicated()

    pair_feed = store.groupby(["library_id", "case_pair", "feed"]).size()
    pair_any = store.groupby(["library_id", "case_pair"]).size()

    def label(lib: str, pair: str, feed: int, primary: bool) -> str:
        """실현 가능성 라벨 (PREREG §2c).

        ``격자-우회(fallback)`` is the label the pathfinder earned: the pair has
        no restart asset AT THIS FEED but does at another, and level-2
        ``pair_feed`` resolution reaches it from the nearest one.  That is a
        real, demonstrated path (12 converged rows each at f113/f129), not an
        impossibility — which is why it is not folded into 미발견."""

        if not primary:
            return "실현 불가(격자)"
        if int(pair_feed.get((lib, pair, feed), 0)) > 0:
            return "실현 확인"
        if int(pair_any.get((lib, pair), 0)) > 0:
            return "격자-우회(fallback)"
        return "미발견"

    rows = []
    for p in picks.itertuples():
        for feed in FEEDS:
            e = p.e_core
            ef_m = alphaB_x[feed] * (aB + bB * e)
            ef_c = alphaC_x[feed] * (aC + bC * e)
            n_here = int(pair_feed.get((p.library_id, p.pair, feed), 0))
            realiz = label(p.library_id, p.pair, feed, bool(p.is_primary))
            # the mesh pair is chosen for e_core accuracy; the ANCHOR pair is
            # chosen for assets, so it is the anchor label that says whether
            # Phase C can actually put MASTER into this cell.
            realiz_a = label(p.anchor_library_id, p.anchor_pair, feed,
                             bool(p.is_primary))
            rows.append(dict(
                cell=f"e{p.e_target:.1f}_f{feed}", e_target=p.e_target,
                library_id=p.library_id, pair=p.pair, e_core=e, feed=feed,
                is_primary=bool(p.is_primary), realizability=realiz,
                anchor_realizability=realiz_a,
                M_HM_tU=round(p.M_HM_tU, 3), n_store_pair_feed=n_here,
                anchor_library_id=p.anchor_library_id,
                anchor_pair=p.anchor_pair, anchor_e_core=p.anchor_e_core,
                anchor_n_seeded_feeds=int(p.anchor_n_seeded_feeds),
                anchor_n_store_pair_feed=int(
                    pair_feed.get((p.anchor_library_id, p.anchor_pair, feed), 0)),
                alpha_B=alphaB_x[feed], alpha_C=alphaC_x[feed],
                alpha_source=("fitted" if feed in alphaB else f"extrapolated_{formB}"),
                e_extrapolated=bool(e > DB_E_MAX),
                feed_extrapolated=bool(feed > DB_FEED_MAX),
                lrm_mean_efpd=ef_m, lrm_ceil_efpd=ef_c,
                lrm_mean_B_cycle=ef_m * P_TH_MW / p.M_HM_tU / 1000.0,
                lrm_ceil_B_cycle=ef_c * P_TH_MW / p.M_HM_tU / 1000.0))
    B = pd.DataFrame(rows)
    B["lrm_mean_B_d"] = B.lrm_mean_B_cycle * N_FA / B.feed
    B["lrm_ceil_B_d"] = B.lrm_ceil_B_cycle * N_FA / B.feed
    B.round(6).to_csv(OUT / "lrm_backbone.csv", index=False, encoding="utf-8")
    log(f"\nwrote {OUT/'lrm_backbone.csv'}  ({len(B)} cells, "
        f"{int(B.is_primary.sum())} buildable / {int((~B.is_primary).sum())} 실현 불가(격자))")
    log("\n실현 가능성 — 메쉬 pair (e_core 정확도 우선):")
    log(B.realizability.value_counts().to_string())
    log("\n실현 가능성 — 앵커 pair (자산 우선; Phase C 가 실제로 던지는 pair):")
    log(B.anchor_realizability.value_counts().to_string())
    log("\n앵커 실현성 x e_target:")
    log(pd.crosstab(B.e_target, B.anchor_realizability).to_string())

    # ------------------------------------------------------- hold-out test ---
    g = (store[store.feed.isin(FEEDS)]
         .groupby(["library_id", "case_pair", "feed"])
         .agg(n=("cyclen", "size"), e=("e_core", "mean"),
              q95=("cyclen", lambda v: float(np.percentile(v, 95))),
              fr_min=("f_r", "min")).reset_index())
    g = g[(g.n >= MIN_ROWS) & (g.e >= 4.95)].copy()
    g["lrm_ceil"] = [alphaC_x[int(f)] * (aC + bC * e) for f, e in zip(g.feed, g.e)]
    g["resid"] = g.q95 - g.lrm_ceil
    g["resid_frac"] = g.resid / g.lrm_ceil
    g["held_out"] = g.e > DB_E_MAX + 0.05
    g.round(6).to_csv(OUT / "lrm_validation_cells.csv", index=False, encoding="utf-8")

    log("\n--- HOLD-OUT: LRM ceiling vs store q95(cyclen)  (PREREG §1d) ---")
    log(g.groupby("held_out").agg(
        n_cells=("resid", "size"), e_lo=("e", "min"), e_hi=("e", "max"),
        bias_efpd=("resid", "mean"), bias_pct=("resid_frac", lambda s: 100 * s.mean()),
        rms_efpd=("resid", lambda s: float(np.sqrt((s ** 2).mean()))),
        rms_pct=("resid_frac", lambda s: 100 * float(np.sqrt((s ** 2).mean())))
    ).round(3).to_string())

    h = g[g.held_out]
    bias_f, rms_f = float(h.resid_frac.mean()), float(np.sqrt((h.resid_frac ** 2).mean()))
    vd = verdict(bias_f, rms_f)
    log(f"\nVERDICT on '아마 LRM이 꽤나 정확할거야': **{vd}**  "
        f"(bias {100*bias_f:+.2f}%, rms {100*rms_f:.2f}%, n={len(h)} held-out cells)")

    # enrichment-slope error, feed fixed effects removed
    feeds_u = sorted(g.feed.unique())
    D = np.array([[1.0 * (f == q) for q in feeds_u] for f in g.feed])
    X = np.c_[D, g.e.to_numpy() - 5.25]
    yv = g.resid.to_numpy()
    coef = np.linalg.lstsq(X, yv, rcond=None)[0]
    rng = np.random.default_rng(20260817)
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(g), len(g))
        try:
            bs.append(np.linalg.lstsq(X[i], yv[i], rcond=None)[0][-1])
        except Exception:                            # noqa: BLE001
            continue
    lo, hi = np.percentile(bs, [2.5, 97.5])
    log(f"\nenrichment-slope error (feed fixed effects removed): "
        f"{coef[-1]:+.2f} EFPD/wt% = {100*coef[-1]/bC:+.2f}% of b_ceil, "
        f"95% CI [{100*lo/bC:+.2f}%, {100*hi/bC:+.2f}%]")
    log("  feed fixed effects [EFPD]: "
        + str({int(f): round(float(v), 1) for f, v in zip(feeds_u, coef[:len(feeds_u)])}))
    log("  (positive = the LRM UNDER-predicts the gain from enrichment, i.e. the "
        "backbone is conservative at the top of the grid)")

    json.dump(dict(
        coefficients=dict(a_mean=aB, b_mean=bB, a_ceil=aC, b_ceil=bC,
                          alpha_mean=alphaB_x, alpha_ceil=alphaC_x,
                          alpha_form_mean=formB, alpha_form_ceil=formC),
        alpha_adjudication=dict(mean=tabB.to_dict("records"),
                                ceiling=tabC.to_dict("records")),
        measured_feed_ratio=measured.to_dict("index"),
        holdout=dict(n_cells=len(h), bias_frac=bias_f, rms_frac=rms_f, verdict=vd,
                     e_lo=float(h.e.min()), e_hi=float(h.e.max())),
        slope_error_pct=float(100 * coef[-1] / bC),
        slope_ci_pct=[float(100 * lo / bC), float(100 * hi / bC)],
        grid=dict(e_targets=list(E_TARGETS), feeds=list(FEEDS),
                  n_cells=len(B), n_buildable=int(B.is_primary.sum())),
    ), open(OUT / "lrm_verdict.json", "w", encoding="utf-8"), indent=1, default=float)
    log(f"\nwrote {OUT/'lrm_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
