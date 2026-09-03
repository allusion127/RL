"""MODEL-ONLY (e_core x feed) scoping mesh — "reload map" for APR1400 LEU+.

For every (core-average enrichment target, feed) cell the champion ensemble
(``--model``, default ``s1f``)
scores a candidate pool built by the SAME machinery the campaign optimizer uses
(:func:`lpopt.search.construct.build_pool`), the pool is gated on the PREDICTED
constraint set (F_r <= 1.55, F_q <= 2.41, CBC_max <= 1600 ppm, |AO| <= 0.30) and
the node is the max-predicted-cyclen feasible candidate.  Discharge burnup is an
equilibrium mass balance, NOT a model head:

    B_cycle = cyclen * P_th / M_HM          [GWd/tU]
    B_d     = B_cycle * 241 / feed

NO MASTER is executed and nothing under data/store is written.

    python scoping_mesh.py                  # full 6x5 mesh
    python scoping_mesh.py --pairs-only     # just print the pair->e_core table
    python scoping_mesh.py --figure-only    # re-render the PNG from mesh_nodes.csv

THE OBJECTIVE AXIS (design ``data/reports/fxy_switch_design_20260829.md`` §3.5.5).
This file is a PRODUCER, not a readout: every node it writes is a PREDICTION, and
its whole frontier — the feasibility mask, the (cyclen, F_r) Pareto front, its
knee, ``min_pred_f_r``, the tier columns — is F_r by construction, and
``mesh_nodes.csv``'s column list is consumed verbatim by ``mesh_vs_db.py``,
``scoping_mesh_fig.py`` and ``autoeng.py``'s marks.  Moving that axis is a
schema change on four consumers AND needs a champion that carries an f_xy head
(``PosValCnnBackend.predict_fxy`` returns ``None`` without one), which no shipped
champion does yet.  So this file does NOT silently switch: ``--objective
min_fxy`` is REFUSED with the two prerequisites named, because the failure mode
worth preventing here is a mesh whose F_r numbers get read under an F_xy heading
(design §3.6: "F_xy를 최적화했다" is exactly the claim that must not be made by
accident).  The measured-side readouts that CAN switch already do —
``anchor_readout.py``, ``autoeng.py``, ``batchswap*_analyze.py``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import readout_axis as RA                                     # noqa: E402

OUT = BASE / "data" / "reports" / "scoping_mesh_20260815"

# -- the scoping grid -------------------------------------------------------- #
#: v3 (2026-08-17): the high-enrichment extension.  0.1 steps over 5.0-6.5 (16
#: levels) x the six-feed production lattice = 96 cells.  Two targets can
#: resolve to the SAME nearest achievable pair — that cell is computed ONCE and
#: drawn as ONE line (see ``is_primary`` / ``e_targets`` in pair_selection.csv).
#: e6.5 is BEYOND THE FUEL GRID: under MAX_PAIR_SPREAD the richest 50/50 pair
#: reaches e_core 6.3645, so 6.4 and 6.5 both collapse onto it and the 6.5 row
#: is reported as 실현 불가(격자) rather than computed twice.
#: v2 was E_TARGETS 5.0-6.0 x FEEDS (105,109,113,117,121); the 5.0-6.0 x
#: 109-121 block is common to both and stays comparable.
E_TARGETS = tuple(round(5.0 + 0.1 * i, 1) for i in range(16))
FEEDS = (109, 113, 117, 121, 125, 129)

# -- predicted-feasibility gate (project constraint set; lpopt.config defaults
#    with the lat1600 CBC gate 1600) ----------------------------------------- #
#: Tier-1 limits == the project constraint set.  F_q and AO are invariant across
#: tiers; F_r and CBC relax together (see TIERS).  Registered ladder:
#: data/reports/mesh_v3_20260817/PREREG2_anchor_redirect_20260817.md §2.
FR_LIMIT, FQ_LIMIT, CBC_LIMIT, AO_LIMIT = 1.55, 2.41, 1600.0, 0.30
#: (label, F_r cap, CBC cap).  JOINT tiers — v3b (2026-08-17).
#:
#: The first cut of this ladder relaxed F_r alone, because that is the knob the
#: directive named.  Reading the 11 244 converged cores that already exist at
#: e_core >= 5.6 shows that knob is not the one holding the door: relaxing F_r
#: alone to 1.80 opens ONE core, relaxing CBC alone to 2600 ppm opens NONE, and
#: only a joint relaxation moves anything (see cbc_wall.py and README §2b).  So
#: the tiers relax BOTH, and each cell also records WHICH constraint binds it —
#: a map that says only "closed" hides the engineering content.
#:
#: Tier-1 IS the project constraint set, so tier1 columns reproduce the v2 gate.
#: Tier-3 is OBSERVATION ONLY and carries no licensing claim.
TIERS = (("tier1", 1.55, 1600.0),
         ("tier2", 1.65, 1800.0),
         ("tier3", 1.80, 2200.0))
#: constraint order used by the binding-constraint diagnostic: (name, column in
#: the prediction matrix, Tier-1 limit, take absolute value first).
CONSTRAINTS = (("f_r", 0, FR_LIMIT, False), ("cbc_max", 1, CBC_LIMIT, False),
               ("f_q", 2, FQ_LIMIT, False), ("ao_abs", 4, AO_LIMIT, True))

# -- equilibrium mass balance ------------------------------------------------ #
P_TH_MW = 3983.0            # %GEN_THD full-core rated thermal power
N_FA = 241
#: grams of U per FA per (g/cm) of the DeCART lattice ``u_mass_g``.  The lattice
#: MASS(g) TOTAL is a per-unit-length 1/8-sector inventory; 8 sectors x 381 cm of
#: active height = 3048.  VERIFIED against the store: over the 38 854 MASTER rows
#: that carry a parsed ``cycle_burnup``, P*cyclen/(1000*cycle_burnup) divided by
#: (241 * pair-mean u_mass_g) is 3048.6 +/- 1 (0.02% off 8*381).
G_PER_FA_PER_UMASS = 8.0 * 381.0
#: population per-FA U mass [tU] for a library whose types carry no ``u_mass_g``
#: (the ga80 letter library is all-NaN); the type-to-type spread is +/-0.5%.
FALLBACK_U_MASS_G = 138.9

#: enrichment spread cap on a candidate pair — the store's paramA 99th percentile
#: ``e_split`` is 0.166, so a wider pair is an out-of-distribution fuel set.
MAX_PAIR_SPREAD = 0.25
#: half-width of the e_core acceptance band around the pair's 50/50 value (the
#: genome's batch-flip moves drift the split; this keeps the cell at ~50/50).
E_CORE_TOL = 0.02

POOL_W0, POOL_W1 = 800, 400
N_ELITE_BAND, N_ELITE_DONOR = 32, 12
SEED = 20260815


# --------------------------------------------------------------------------- #
# fuel-pair selection
# --------------------------------------------------------------------------- #
#: paramA ships every lattice TWICE — a descriptive id and a short package-registry
#: alias (``S7`` == ``P6257Z1G10N12``).  Keeping both would enumerate the same
#: physical pair under four spellings and invent zero-spread "pairs" out of a type
#: and its own alias, so only the descriptive form is enumerated.
_PARAMA_CANON = r"^P\d{4}Z\d"


#: how far from the e_core target the ANCHOR search may stray.  Wider than the
#: 0.02 near-tie window of the mesh pick because an anchor must actually RUN —
#: a pair with no store rows at any mesh feed has no restart asset to morph
#: from, so a slightly worse e_core with real assets beats an exact e_core with
#: none.  Anchors are what Phase C spends MASTER on; the mesh pick is unchanged.
ANCHOR_DE = 0.06
#: an anchor has to be a library the fleet actually RUNS.  ga80 and paramA are
#: the two full-physics production libraries every current deck names; 260624 /
#: 5.8_5.1 / CPHA are SA-era libraries that survive in the store but have no
#: current design package or routing, so an "anchor" in one of them is a deck
#: that cannot be launched.  The MESH pick is unrestricted (it is a model-only
#: representative and never becomes a MASTER call).
ANCHOR_LIBS = ("ga80", "paramA")


def select_pairs(fuel, fuel_df: pd.DataFrame, store: pd.DataFrame,
                 log=lambda _m: None, targets=E_TARGETS,
                 feeds=FEEDS) -> pd.DataFrame:
    """Nearest-achievable 50/50 pair per e_core target, over every full-physics
    library.  Rule: e_core matched to 0.01 (``round(|e_core - target|, 2)``), then
    the smallest enrichment spread (an in-distribution fuel set), then the most
    store rows carrying that exact pair.

    v3 additionally reports, per target, the best ANCHOR pair — the same
    enumeration re-ranked by how many of the mesh feeds the pair is actually
    seeded at.  The mesh pick (the ``pair`` columns) is UNCHANGED from v2 so the
    common block stays comparable; the anchor pick lives in ``anchor_*`` and is
    what the Phase-C MASTER decks address."""

    pair_rows = store.groupby(["library_id", "case_pair"]).size()
    pair_feed = store.groupby(["library_id", "case_pair", "feed"]).size()
    cand = []
    for lib in ("ga80", "paramA", "260624", "5.8_5.1", "CPHA"):
        g = fuel_df[(fuel_df.library_id == lib) & (~fuel_df.feature_poor.astype(bool))]
        if lib == "paramA":
            g = g[g.type_id.str.match(_PARAMA_CANON)]
        enr = dict(zip(g.type_id, g.u_avg_enrichment))
        for a, b in itertools.combinations(sorted(g.type_id), 2):
            spread = abs(enr[a] - enr[b])
            if spread > MAX_PAIR_SPREAD:
                continue
            try:
                ec = float(fuel.pair_e_core(a, b, 0.5, lib))
            except Exception:                        # noqa: BLE001 — skip unresolvable
                continue
            if math.isfinite(ec):
                cand.append((lib, f"{a}_{b}", a, b, ec, spread))
    U = pd.DataFrame(cand, columns=["lib", "pair", "a", "b", "e_core", "spread"])
    U["sup_pair"] = [int(pair_rows.get((r.lib, r.pair), 0)) for r in U.itertuples()]
    U["n_seeded_feeds"] = [
        sum(int(pair_feed.get((r.lib, r.pair, f), 0)) > 0 for f in feeds)
        for r in U.itertuples()]
    log(f"pair enumeration: {len(U)} pairs, e_core {U.e_core.min():.4f}-"
        f"{U.e_core.max():.4f} (GRID CEILING — targets above it cannot be built)")

    picks = []
    for t in targets:
        c = U.assign(d=(U.e_core - t).abs())
        c = c[c.d <= c.d.min() + 0.02].copy()
        c["dbin"] = c.d.round(2)
        c = c.sort_values(["dbin", "spread", "sup_pair"], ascending=[True, True, False])
        log(f"-- target {t}: {len(c)} near-ties; top 4")
        log(c.head(4)[["lib", "pair", "e_core", "spread", "d", "sup_pair"]]
            .to_string(index=False))
        r = c.iloc[0]
        # anchor: restrict to runnable libraries, widen to ANCHOR_DE, and rank
        # seeded-feed coverage first — assets beat an exact e_core here.
        aw = U[U.lib.isin(ANCHOR_LIBS)].assign(d=(U.e_core - t).abs())
        aw = aw[aw.d <= max(ANCHOR_DE, aw.d.min())]
        aw = aw.sort_values(["n_seeded_feeds", "sup_pair", "spread"],
                            ascending=[False, False, True])
        ar = aw.iloc[0]
        picks.append(dict(e_target=t, library_id=r.lib, pair=r.pair, type_a=r.a,
                          type_b=r.b, e_core=float(r.e_core), spread=float(r.spread),
                          d_target=float(abs(r.e_core - t)), sup_pair=int(r.sup_pair),
                          n_seeded_feeds=int(r.n_seeded_feeds),
                          anchor_library_id=ar.lib, anchor_pair=ar.pair,
                          anchor_type_a=ar.a, anchor_type_b=ar.b,
                          anchor_e_core=float(ar.e_core),
                          anchor_d_target=float(abs(ar.e_core - t)),
                          anchor_sup_pair=int(ar.sup_pair),
                          anchor_n_seeded_feeds=int(ar.n_seeded_feeds),
                          anchor_is_mesh_pair=bool(ar.pair == r.pair
                                                   and ar.lib == r.lib),
                          **{f"anchor_n_f{f}": int(pair_feed.get((ar.lib, ar.pair, f), 0))
                             for f in feeds}))
    return pd.DataFrame(picks)


def pair_hm_tu(fuel_df: pd.DataFrame, lib: str, a: str, b: str) -> tuple[float, float]:
    """``(per-FA U mass [tU], M_HM [tU])`` for a 50/50 pair core in equilibrium."""

    m = fuel_df.set_index(["library_id", "type_id"]).u_mass_g
    vals = [m.get((lib, t), np.nan) for t in (a, b)]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    u_mass_g = float(np.mean(vals)) if vals else FALLBACK_U_MASS_G
    per_fa = u_mass_g * G_PER_FA_PER_UMASS / 1e6
    return per_fa, per_fa * N_FA


# --------------------------------------------------------------------------- #
# per-cell model search
# --------------------------------------------------------------------------- #
def _feasible_mask(m: np.ndarray, fr_cap: float = FR_LIMIT,
                   cbc_cap: float = CBC_LIMIT) -> np.ndarray:
    """Predicted-feasibility gate.  F_r and CBC are the tier-varying limits —
    they are the two that actually bind (README §2b).  F_q and AO stay at the
    project constraint set in every tier: AO is never violated in the store
    (0 % of 11 244 high-e cores) and F_q follows F_r rather than leading it."""

    return ((m[:, 0] <= fr_cap) & (m[:, 2] <= FQ_LIMIT)
            & (m[:, 1] <= cbc_cap) & (np.abs(m[:, 4]) <= AO_LIMIT))


def binding_constraint(m: np.ndarray, band: np.ndarray) -> dict:
    """Which constraint closes this cell, and by how much.

    For every in-band candidate each constraint is expressed as a fraction of
    its Tier-1 limit, so the four become comparable.  The candidate that
    minimises its own WORST fraction is the most nearly feasible core the pool
    contains; the constraint that is worst *at that candidate* is the one
    genuinely holding the cell shut.  Reporting the per-constraint minima
    separately would name a different core for each constraint and could
    declare a cell "closed by F_r" when no single core is close on both.
    """

    out = {"binding_constraint": "", "binding_excess_pct": np.nan}
    if not band.any():
        return out
    idx = np.flatnonzero(band)
    rel = np.column_stack([
        (np.abs(m[idx, col]) if absval else m[idx, col]) / lim
        for _n, col, lim, absval in CONSTRAINTS])
    worst = rel.max(axis=1)
    k = int(np.argmin(worst))
    j = int(np.argmax(rel[k]))
    out["binding_constraint"] = CONSTRAINTS[j][0]
    out["binding_excess_pct"] = float(100.0 * (worst[k] - 1.0))
    for n, col, lim, absval in CONSTRAINTS:      # honest per-constraint floors
        v = np.abs(m[idx, col]) if absval else m[idx, col]
        out[f"min_{n}"] = float(np.nanmin(v))
    return out


def pareto_front(m: np.ndarray, feas: np.ndarray) -> np.ndarray:
    """Indices of the feasible non-dominated set in (cyclen MAX, F_r MIN).

    A feasible candidate is dominated when some other feasible candidate is at
    least as long AND at least as flat, strictly better on one of the two.  The
    safety gates are unchanged — this only stops collapsing the surviving set to
    a single argmax, because "best" is a choice the study should not hard-code.

    Sweep: sort by F_r ascending (cyclen descending inside an F_r tie) and keep a
    point only when it is strictly longer than every flatter point seen so far.
    Returned in ascending F_r, which on a true front is also ascending cyclen —
    so ``front[0]`` is the flattest representative and ``front[-1]`` the longest.
    """

    idx = np.flatnonzero(feas)
    if idx.size == 0:
        return idx
    cy, fr = m[idx, 3], m[idx, 0]
    keep, best_cy = [], -np.inf
    for k in np.lexsort((-cy, fr)):
        if cy[k] > best_cy + 1e-12:
            keep.append(k)
            best_cy = cy[k]
    return idx[np.asarray(keep, dtype=int)]


def knee_index(cyclen: np.ndarray, f_r: np.ndarray) -> int:
    """Index of the front's knee — max perpendicular distance from the chord
    joining the two endpoints, both axes min-max normalised first so EFPD and
    F_r contribute on the same scale.  Returns -1 when the front is degenerate
    (< 3 points, or all points on one line)."""

    n = len(cyclen)
    if n < 3:
        return -1
    x = np.ptp(cyclen), np.ptp(f_r)
    if x[0] <= 0 or x[1] <= 0:
        return -1
    u = (cyclen - cyclen.min()) / x[0]
    v = (f_r - f_r.min()) / x[1]
    dx, dy = u[-1] - u[0], v[-1] - v[0]
    norm = math.hypot(dx, dy)
    if norm <= 0:
        return -1
    d = np.abs(dy * (u - u[0]) - dx * (v - v[0])) / norm
    i = int(np.argmax(d))
    return i if (0 < i < n - 1 and d[i] > 0.05) else -1


def build_elites(store: pd.DataFrame, donors: pd.DataFrame, lib: str, pair: str,
                 e_core: float, feed: int):
    """Elite parents: feasibility-first band rows (nearest feed) + the globally
    flattest verified cores, batch-remapped onto this cell's pair."""

    from fr_transfer import pair_mapping, substitute
    from lpopt.data.schema import unpack_pattern

    def _take(df):
        out = []
        for _, r in df.iterrows():
            try:
                pat = unpack_pattern(str(r["pattern"]))
                mp = pair_mapping(str(r.case_pair), pair)
                out.append((str(r.record_id), substitute(pat, mp)))
            except Exception:                        # noqa: BLE001 — a parent is optional
                continue
        return out

    band = store[(store.library_id == lib) & (store.e_core.sub(e_core).abs() <= 0.05)]
    elites = []
    if len(band):
        b = band.assign(dfeed=(band.feed - feed).abs(), feas=_store_feasible(band))
        # half the slots on the flattest cores, half on the longest feasible ones
        elites += _take(b.sort_values(["dfeed", "f_r"]).head(N_ELITE_BAND // 2))
        best = b[b.feas].sort_values(["dfeed", "cyclen"], ascending=[True, False])
        if not len(best):
            best = b.sort_values(["dfeed", "cyclen"], ascending=[True, False])
        elites += _take(best.head(N_ELITE_BAND // 2))
    elites += _take(donors.head(N_ELITE_DONOR))
    seen, uniq = set(), []
    for rid, pat in elites:
        if pat.canonical() in seen:
            continue
        seen.add(pat.canonical())
        uniq.append((rid, pat))
    return uniq


def _store_feasible(df: pd.DataFrame) -> pd.Series:
    return ((df.f_r <= FR_LIMIT) & (df.f_q <= FQ_LIMIT)
            & (df.cbc_max <= CBC_LIMIT) & (df.ao_abs.abs() <= AO_LIMIT))


def run_cell(model, cfg, store, donors, lib, pair, ta, tb, e_core, feed, log) -> dict:
    """Two-wave model-only search of one (pair, feed) cell."""

    from lpopt.search.construct import CaseContext, build_pool, screen_e_core_band

    ctx = CaseContext(pair=pair, feed=feed, library_id=lib, e_core=e_core)
    rng = random.Random(SEED + feed * 13 + int(round(e_core * 100)))
    elites = build_elites(store, donors, lib, pair, e_core, feed)

    t0 = time.time()
    pool = build_pool(ctx, None, elites, set(), rng, cfg, wave_index=0, size=POOL_W0)
    pats = [c.pattern for c in pool]
    pred = model.predict(pats, ctx.case_key, e_core)
    m, sig = pred.mean, pred.epistemic_std

    # wave 1: exploit the wave-0 top + probe the feasibility boundary with the
    # near-miss (n_moves=1) trust region the campaign uses.
    band = screen_e_core_band(pats, model.fuel, lib, e_core, E_CORE_TOL)
    feas = _feasible_mask(m) & band
    order = np.argsort(-np.where(feas, m[:, 3], -np.inf))[:24]
    prev_top = [(pool[i].record_id, pats[i]) for i in order if feas[i]]
    near = band & ~feas & (m[:, 0] <= FR_LIMIT + 0.10) & (m[:, 2] <= FQ_LIMIT + 0.15) \
        & (m[:, 1] <= CBC_LIMIT + 120) & (np.abs(m[:, 4]) <= AO_LIMIT)
    nm_order = np.argsort(-np.where(near, m[:, 3], -np.inf))[:24]
    near_miss = [(pool[i].record_id, pats[i]) for i in nm_order if near[i]]

    ledger = {c.record_id for c in pool}
    pool1 = build_pool(ctx, None, elites, ledger, rng, cfg, wave_index=0,
                       prev_top=prev_top, near_miss_parents=near_miss, size=POOL_W1)
    if pool1:
        p1 = [c.pattern for c in pool1]
        pr1 = model.predict(p1, ctx.case_key, e_core)
        pool = pool + pool1
        pats = pats + p1
        m = np.vstack([m, pr1.mean])
        sig = np.vstack([sig, pr1.epistemic_std])
        band = np.concatenate([band, screen_e_core_band(p1, model.fuel, lib, e_core,
                                                        E_CORE_TOL)])
    feas = _feasible_mask(m) & band

    row = dict(n_candidates=len(pats), n_in_band=int(band.sum()),
               n_feasible=int(feas.sum()),
               min_pred_f_r=float(np.nanmin(m[band, 0])) if band.any() else np.nan,
               max_pred_cyclen_any=float(np.nanmax(m[band, 3])) if band.any() else np.nan,
               seconds=round(time.time() - t0, 1))

    # -- TIERED gates.  Tier-1 == the v2 gate, so the untagged columns above and
    #    the *_tier1 columns agree by construction.  F_r AND CBC relax together
    #    (see TIERS); the map is read as contours of both, and
    #    ``binding_constraint`` names the one that actually closes the cell.
    row.update(binding_constraint(m, band))
    for tname, cap, cbc in TIERS:
        tf = _feasible_mask(m, cap, cbc) & band
        row[f"n_feasible_{tname}"] = int(tf.sum())
        if tf.any():
            k = int(np.argmax(np.where(tf, m[:, 3], -np.inf)))
            row[f"cyclen_{tname}"] = float(m[k, 3])
            row[f"f_r_{tname}"] = float(m[k, 0])
            row[f"sigma_cyclen_{tname}"] = float(sig[k, 3])
            row[f"record_id_{tname}"] = pool[k].record_id
        else:
            row[f"cyclen_{tname}"] = np.nan
            row[f"f_r_{tname}"] = np.nan
            row[f"sigma_cyclen_{tname}"] = np.nan
            row[f"record_id_{tname}"] = ""
    row["tier_reached"] = next(
        (t for t, _c, _b in TIERS if row[f"n_feasible_{t}"] > 0), "none")
    front_rows: list[dict] = []
    if feas.any():
        i = int(np.argmax(np.where(feas, m[:, 3], -np.inf)))
        row.update(pred_cyclen=float(m[i, 3]), sigma_cyclen=float(sig[i, 3]),
                   pred_f_r=float(m[i, 0]), pred_f_q=float(m[i, 2]),
                   pred_cbc_max=float(m[i, 1]), pred_ao_abs=float(m[i, 4]),
                   pred_max_pin_bu=float(m[i, 6]),
                   node_e_core=float(_pat_e_core(model, pats[i], lib)),
                   node_record_id=pool[i].record_id, node_origin=pool[i].origin)
        # MULTI-OBJECTIVE: the gate says which cores are ALLOWED; "best" among them
        # is a choice this study must not hard-code, so the whole (cyclen, F_r)
        # Pareto front is kept and three representatives are tagged on it.
        front = pareto_front(m, feas)
        row["n_pareto"] = int(front.size)
        k_knee = knee_index(m[front, 3], m[front, 0])
        for rank, idx in enumerate(front):
            rep = ("min_f_r" if rank == 0 else
                   "max_cyclen" if rank == len(front) - 1 else
                   "knee" if rank == k_knee else "")
            front_rows.append(dict(
                rank=rank, rep=rep, pred_cyclen=float(m[idx, 3]),
                sigma_cyclen=float(sig[idx, 3]), pred_f_r=float(m[idx, 0]),
                pred_f_q=float(m[idx, 2]), pred_cbc_max=float(m[idx, 1]),
                pred_ao_abs=float(m[idx, 4]), pred_max_pin_bu=float(m[idx, 6]),
                cand_e_core=float(_pat_e_core(model, pats[idx], lib)),
                record_id=pool[idx].record_id, origin=pool[idx].origin))
        row["pareto_min_f_r"] = float(m[front[0], 0])
        row["pareto_min_f_r_cyclen"] = float(m[front[0], 3])
        row["pareto_cyclen_span"] = float(m[front[-1], 3] - m[front[0], 3])
        row["pareto_f_r_span"] = float(m[front[-1], 0] - m[front[0], 0])
    else:                       # honest gap: report the best in-band candidate
        j = int(np.argmax(np.where(band, m[:, 3], -np.inf))) if band.any() else 0
        row.update(pred_cyclen=np.nan, sigma_cyclen=np.nan, pred_f_r=np.nan,
                   pred_f_q=np.nan, pred_cbc_max=np.nan, pred_ao_abs=np.nan,
                   pred_max_pin_bu=np.nan, node_e_core=np.nan,
                   node_record_id="", node_origin="",
                   ref_cyclen=float(m[j, 3]), ref_f_r=float(m[j, 0]))
        row["n_pareto"] = 0
    log(f"  {pair:32s} f{feed} e={e_core:.4f} | pool {row['n_candidates']:5d} "
        f"band {row['n_in_band']:5d} feas {row['n_feasible']:4d} "
        f"pareto {row['n_pareto']:3d} | "
        f"minF_r {row['min_pred_f_r']:.3f} | "
        f"cyclen {row['pred_cyclen'] if row['n_feasible'] else float('nan'):8.2f} "
        f"| tiers "
        + "/".join(f"{row['n_feasible_' + t]}" for t, _c, _b in TIERS)
        + f" ->{row['tier_reached']:5s} | {row['seconds']:.0f}s")
    return row, front_rows


def _pat_e_core(model, pat, lib):
    from lpopt.search.construct import predicted_e_core
    try:
        return predicted_e_core(pat, model.fuel, lib)
    except Exception:                                # noqa: BLE001
        return np.nan


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--e-targets", default="", help="comma-separated subset of the "
                    "e_core grid (shards the 30 cells over several CPU processes)")
    ap.add_argument("--feeds", default="", help="comma-separated subset of the feed "
                    "grid; with --model this re-runs a column under a different "
                    "ensemble as a controlled A/B (always use --tag)")
    ap.add_argument("--tag", default="", help="suffix for this shard's CSV/log")
    ap.add_argument("--model", default="s1g", help="ensemble directory under "
                    "data/models.  s1g is the 8th champion (gate_s1g.json "
                    "pass=true, promoted 2026-08-16); s1f/s1e runs are "
                    "preserved as mesh_nodes_s1f.csv / mesh_nodes_s1e.csv")
    ap.add_argument("--coverage-in-distribution", action="store_true",
                    help="compute `in_distribution` from the TrustRegion SUPPORT "
                         "BIN set (KPI A3, active_frontier_loop_spec_20260903.md "
                         "sec.4d) instead of the shipped `feed == 121` constant.  "
                         "DEFAULT OFF: mesh_nodes.csv keeps its historic column "
                         "so mesh_vs_db.py / autoeng.py readouts are unchanged.")
    ap.add_argument("--coverage-e-core-band", type=float, default=0.05,
                    help="e_core bin width for --coverage-in-distribution "
                         "([trust_region].e_core_band default 0.05)")
    ap.add_argument("--coverage-promote-after", type=int, default=16,
                    help="labels that make a bin SUPPORTED for "
                         "--coverage-in-distribution ([trust_region]."
                         "promote_after default 16)")
    ap.add_argument("--pairs-only", action="store_true")
    ap.add_argument("--figure-only", action="store_true")
    ap.add_argument("--out-dir", default="", help="report directory for this "
                    "sweep (default: the v2 dir).  v3 writes to "
                    "data/reports/mesh_v3_20260817 so the v2 study it is "
                    "compared against stays untouched.")
    RA.add_axis_args(ap)
    args = ap.parse_args()

    # See the module docstring: this producer is F_r by construction and refuses
    # to be READ as anything else.  The refusal names both prerequisites so the
    # next person does not have to rediscover them.
    axis = RA.axis_from_args(args)
    if axis.is_fxy:
        raise SystemExit(
            f"scoping_mesh is an F_r producer and cannot yet be swept on "
            f"{axis.label}.  Two prerequisites, both open:\n"
            f"  1. a champion carrying an f_xy head — PosValCnnBackend.predict_fxy "
            f"returns None on every model currently under data/models, so there "
            f"is no {axis.label} to gate or rank on (design 20260829 sec. 3.6 "
            f"forbids inventing one);\n"
            f"  2. mesh_nodes.csv's column list is F_r-named (min_pred_f_r, "
            f"pareto_min_f_r, f_r_tier*) and is consumed verbatim by "
            f"mesh_vs_db.py, scoping_mesh_fig.py and autoeng.py.\n"
            f"Refusing rather than relabelling F_r columns as {axis.label}.")

    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir)
        if not OUT.is_absolute():
            OUT = BASE / OUT
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / f"run{args.tag}.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    if args.figure_only:
        from scoping_mesh_fig import render
        render(pd.read_csv(OUT / "mesh_nodes.csv"), OUT / "scoping_mesh.png",
               model=args.model)
        return 0

    import torch
    torch.set_num_threads(int(args.threads))
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.model.model_api import PosValCnnBackend

    cfg = load_config(str(BASE / "lpopt.inp"))
    fuel_df = pd.read_parquet(BASE / "data/store/fuel_types.parquet")
    fuel = FuelLibrary.from_parquet(BASE / "data/store/fuel_types.parquet")
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    store = store[(store.valid == True) & (store.converged == True)]   # noqa: E712

    picks = select_pairs(fuel, fuel_df, store, log)
    picks["per_fa_tU"], picks["M_HM_tU"] = zip(*[
        pair_hm_tu(fuel_df, r.library_id, r.type_a, r.type_b) for r in picks.itertuples()])
    # COLLAPSE: several e_core targets can resolve to the same nearest achievable
    # pair, i.e. to the SAME physical cell.  It is computed once (``is_primary``)
    # and drawn as one line; ``e_targets`` records every target that landed on it.
    key = picks.library_id + "|" + picks.pair
    grp = picks.groupby(key, sort=False).e_target
    picks["e_targets"] = key.map(grp.apply(lambda s: ",".join(f"{v:.1f}" for v in s)))
    picks["n_targets"] = key.map(grp.size())
    picks["is_primary"] = ~key.duplicated()
    log("=== pair -> e_core mapping (50/50 split) ===")
    log(picks.to_string(index=False))
    n_col = int(len(picks) - picks.is_primary.sum())
    log(f"{len(picks)} targets -> {int(picks.is_primary.sum())} distinct cells "
        f"({n_col} target(s) collapsed onto an already-selected pair)")
    picks.to_csv(OUT / "pair_selection.csv", index=False)
    if args.pairs_only:
        return 0
    picks = picks[picks.is_primary]
    if args.e_targets:
        keep = {float(x) for x in args.e_targets.split(",")}
        picks = picks[picks.e_target.isin(keep)]

    model = PosValCnnBackend.from_dir(BASE / "data/models" / args.model,
                                      store_dir=BASE / "data/store",
                                      library_id="ga80", device="cpu")
    # The pinball quantile heads are a SECOND encode+forward over the whole batch
    # (~50% of predict() wall time) and this study reads only the ensemble mean and
    # the epistemic spread, so they are switched off.  The quantile block in
    # predict() is purely additive — mean / epistemic_std are byte-identical.
    model.quantile_targets = ()
    log(f"model: data/models/{args.model}, {len(model.members)} members, cpu, "
        f"{args.threads} torch threads, quantile heads off")
    # KPI A3 (spec sec.4d): the shipped `in_distribution` is the CONSTANT
    # `feed == 121`, which answers "is this the feed the study was centred on",
    # not "has the model seen this bin" -- so the coverage-expansion KPI is not
    # measurable from it.  Behind --coverage-in-distribution the column becomes
    # the TrustRegion support-bin membership of the node's own (feed, e-bin).
    # Default off: every existing readout of mesh_nodes.csv is byte-identical.
    _cov_bins: set = set()
    if args.coverage_in_distribution:
        from lpopt.search.coverage import bin_counts, in_distribution as _cov_in
        _counts = bin_counts(store, e_core_band=args.coverage_e_core_band)
        _cov_bins = {k for k, n in _counts.items()
                     if n >= int(args.coverage_promote_after)}
        log(f"coverage: {len(_cov_bins)} SUPPORTED (feed, e-bin) bins at "
            f">= {args.coverage_promote_after} labels, band "
            f"{args.coverage_e_core_band} (KPI A3)")

    def _in_dist(feed_: int, e_core_: float) -> bool:
        if not args.coverage_in_distribution:
            return bool(feed_ == 121)
        return _cov_in(feed_, e_core_, _cov_bins,
                       e_core_band=args.coverage_e_core_band)

    donors = store[_store_feasible(store)].nsmallest(N_ELITE_DONOR * 2, "f_r")
    log(f"flat donors: {donors.case_pair.value_counts().to_dict()} "
        f"F_r {donors.f_r.min():.3f}-{donors.f_r.max():.3f}")

    feeds = tuple(int(x) for x in args.feeds.split(",")) if args.feeds else FEEDS
    rows, fronts, t0 = [], [], time.time()
    for p in picks.itertuples():
        for feed in feeds:
            r, front = run_cell(model, cfg, store, donors, p.library_id, p.pair,
                                p.type_a, p.type_b, p.e_core, feed, log)
            for fr in front:                       # same mass balance as the node
                bc = fr["pred_cyclen"] * P_TH_MW / p.M_HM_tU / 1000.0
                fr.update(cell=f"e{p.e_target:.1f}_f{feed}", e_target=p.e_target,
                          library_id=p.library_id, pair=p.pair, feed=feed,
                          M_HM_tU=round(p.M_HM_tU, 3), B_cycle=bc,
                          B_d=bc * N_FA / feed,
                          in_distribution=_in_dist(feed, fr.get("cand_e_core",
                                                                p.e_core)))
                fronts.append(fr)
            B_cycle = (r["pred_cyclen"] * P_TH_MW / p.M_HM_tU / 1000.0
                       if r["n_feasible"] else np.nan)
            for tname, _cap, _cbc in TIERS:   # same mass balance, per tier
                bc = r[f"cyclen_{tname}"] * P_TH_MW / p.M_HM_tU / 1000.0
                r[f"B_cycle_{tname}"] = bc
                r[f"B_d_{tname}"] = bc * N_FA / feed
            r.update(cell=f"e{p.e_target:.1f}_f{feed}", e_target=p.e_target,
                     e_targets=p.e_targets, n_targets=p.n_targets,
                     library_id=p.library_id, pair=p.pair, e_core_pair=p.e_core,
                     pair_spread=p.spread, feed=feed,
                     per_fa_tU=round(p.per_fa_tU, 5), M_HM_tU=round(p.M_HM_tU, 3),
                     B_cycle=B_cycle, B_d=B_cycle * N_FA / feed,
                     n_store_feed=int((store.feed == feed).sum()),
                     n_store_feed_ecore=int(((store.feed == feed) &
                                             (store.e_core.sub(p.e_core).abs() <= 0.05)).sum()),
                     n_store_feed_lib=int(((store.feed == feed) &
                                           (store.library_id == p.library_id)).sum()),
                     n_store_pair_feed=int(((store.feed == feed) &
                                            (store.case_pair == p.pair)).sum()),
                     in_distribution=_in_dist(feed, p.e_core))
            rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["cell", "e_target", "e_targets", "n_targets",
            "library_id", "pair", "e_core_pair", "pair_spread",
            "node_e_core", "feed", "per_fa_tU", "M_HM_tU", "n_candidates",
            "n_in_band", "n_feasible", "pred_cyclen", "sigma_cyclen", "pred_f_r",
            "pred_f_q", "pred_cbc_max", "pred_ao_abs", "pred_max_pin_bu",
            "B_cycle", "B_d", "min_pred_f_r", "max_pred_cyclen_any", "ref_cyclen",
            "ref_f_r", "n_store_feed", "n_store_feed_ecore", "n_store_feed_lib",
            "n_store_pair_feed", "in_distribution", "node_record_id", "node_origin",
            "n_pareto", "pareto_min_f_r", "pareto_min_f_r_cyclen",
            "pareto_cyclen_span", "pareto_f_r_span", "seconds", "tier_reached",
            "binding_constraint", "binding_excess_pct",
            "min_f_r", "min_cbc_max", "min_f_q", "min_ao_abs"]
    cols += [f"{stem}_{t}" for t, _c, _b in TIERS
             for stem in ("n_feasible", "cyclen", "f_r", "sigma_cyclen",
                          "B_cycle", "B_d", "record_id")]
    df = df.reindex(columns=cols)
    out_csv = OUT / f"mesh_nodes{args.tag}.csv"
    df.to_csv(out_csv, index=False)

    pf = pd.DataFrame(fronts).reindex(columns=[
        "cell", "e_target", "feed", "library_id", "pair", "rank", "rep",
        "pred_cyclen", "sigma_cyclen", "pred_f_r", "pred_f_q", "pred_cbc_max",
        "pred_ao_abs", "pred_max_pin_bu", "B_cycle", "B_d", "M_HM_tU",
        "cand_e_core", "in_distribution", "record_id", "origin"])
    pf.to_csv(OUT / f"mesh_pareto{args.tag}.csv", index=False)
    log(f"\nwrote {out_csv}  ({time.time()-t0:.0f}s total)")
    log(f"feasible cells: {int((df.n_feasible>0).sum())}/{len(df)}  "
        f"pareto points: {len(pf)}")
    if not args.tag:
        from scoping_mesh_fig import render
        render(df, OUT / "scoping_mesh.png", model=args.model)
        log(f"wrote {OUT/'scoping_mesh.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
