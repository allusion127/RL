"""MULTI-TYPE (e_core x feed) scoping mesh — P3a.  MODEL-ONLY, NO MASTER.

The v3 sweep (``scoping_mesh.py``) asked one question per cell: *what is the best
2-fresh-type reload this cell admits?*  This sweep asks three, on the SAME 90-cell
grid, with the SAME machinery, under the SAME gate — plus the pin axis, now
first-class:

    k = 2   the v3 pair                       (74 537 training rows)
    k = 3   a composition-matched TRIPLE      (60 rows / 39 in train — SPARSE)
    k = 4   a composition-matched QUAD        (0 rows — PURE EXTRAPOLATION)

and reports, per cell, the best of each and the predicted grading gain
``delta(3-type - 2-type)``.

Honesty rails, wired into the output rather than the prose:

* every k>=3 column carries a ``support_class`` word.  ``cond_v8``'s
  ``g_type_frac_4/5`` are identically zero in every training row (S1i addendum
  §3.2), so a 4-type prediction is the model evaluating a global it has never
  seen move.  That is extrapolation, and the CSV says so on every row.
* the 3-type budget is SMALLER than the 2-type budget by construction (the
  operator's 50/40/10 mixture).  A 3-type pool that still finds a flatter core
  than a 50 %-larger 2-type pool is evidence; the asymmetry can only understate
  the gain, never manufacture it.  Registered here, before the sweep ran.
* the 2-type sub-pool keeps v3's budget EXACTLY (800 + 400), so the k=2 columns
  are directly comparable to ``mesh_v3_20260817/mesh_nodes.csv`` cell by cell.
  Any 2-type movement is the s1g -> s1i model change, not a budget change.

Nothing under ``data/store`` is written and no fleet call is made.

    python mesh_multitype.py --cases-only          # print the case table only
    python mesh_multitype.py --tag _A --e-targets 5.0,5.1,5.2
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from scoping_mesh import (AO_LIMIT, CBC_LIMIT, E_CORE_TOL, E_TARGETS,  # noqa: E402
                          FEEDS, FQ_LIMIT, FR_LIMIT, MAX_PAIR_SPREAD, N_FA,
                          N_ELITE_BAND, N_ELITE_DONOR, P_TH_MW, SEED,
                          _PARAMA_CANON, _pat_e_core, _store_feasible,
                          binding_constraint, knee_index, pair_hm_tu,
                          pareto_front, select_pairs)

OUT = BASE / "data" / "reports" / "mesh_multitype_20260818"

# --------------------------------------------------------------------------- #
# PRE-REGISTERED CONSTANTS  (fixed before the first cell was scored)
# --------------------------------------------------------------------------- #
#: the pin-burnup gate, now a FIRST-CLASS axis of the feasibility test rather
#: than a post-hoc check.  Same number the campaigns carry.
PIN_LIMIT = 78.0

#: candidate-pool budget per fresh-type count.  50 / 40 / 10 of 2 400, with the
#: k=2 share pinned to v3's 1 200 so the 2-type column stays comparable.
#: (wave-0, wave-1) — the same 2:1 split v3 used.
BUDGET = {2: (800, 400), 3: (640, 320), 4: (160, 80)}

#: how far a k-type case's equal-share e_core may sit from the cell's 2-type
#: e_core and still count as COMPOSITION-MATCHED.  Same number as the band
#: screen, so a matched case's boards can actually land in the cell's band.
MATCH_TOL = E_CORE_TOL          # 0.02

#: the training support behind each fresh-type count, measured on the canonical
#: store (74 597 rows) and printed into every output row.
SUPPORT_CLASS = {
    2: "trained (74,537 rows)",
    3: "SPARSE (60 rows / 39 in train)",
    4: "EXTRAPOLATION (0 rows; g_type_frac_4 inert at train time)",
    5: "EXTRAPOLATION (0 rows)",
}

#: ladder tie-break: two types whose enrichments differ by less than this are
#: treated as enrichment-degenerate, and the grading ladder is read off n_gd
#: instead (the ga80 letter library grades by gadolinia at fixed enrichment).
E_TIE = 1e-6


# --------------------------------------------------------------------------- #
# 1. case selection — the composition-matching rule
# --------------------------------------------------------------------------- #
def _ladder(rows: pd.DataFrame) -> tuple[list[float], str]:
    """Reactivity-ladder coordinate for a candidate type set, hot -> cold.

    Enrichment first (the precedent rule, ``tripletype_design_20260817.md``
    §3.1: hot 5.7861 / mid 5.6685 / cold 5.6023).  When every member shares an
    enrichment — the whole ga80 letter library does — enrichment cannot express
    a ladder and the grading knob is the gadolinia loading, so the coordinate
    falls back to ``-n_gd`` (fewer Gd rods = hotter).  BOC ``kinf0`` was
    considered and REJECTED: it puts ``P6253Z2G10N20`` (the mid the first real
    3-type campaign actually used) ABOVE its own hot member, because Gd
    suppression at BOC is not the cycle-average reactivity that grading targets.
    """

    e = rows.u_avg_enrichment.astype(float).tolist()
    if max(e) - min(e) > E_TIE:
        return e, "enrichment"
    ng = rows.n_gd.astype(float).tolist()
    return [-v for v in ng], "n_gd"


def anchor_key(row) -> tuple:
    """The R1 "enrichment spec" a fresh type belongs to.

    R1 bans mixing enrichment specs across feed assemblies.  The repo's
    :func:`lpopt.data.compliance.is_cross_anchor` reads that off the leading
    LETTER run of the type id, which is exactly right for the ga80 letter
    library (``E1``/``E2`` share family ``E``) and **degenerates to a single
    family ``P`` for the whole paramA library**, where every descriptive id
    starts with one letter.  Under that reading ``P5853…`` and ``P6253…`` — two
    genuinely different enrichment specs (5.80/5.336 vs 6.20/5.270) — would
    count as mono-anchor, which is not what R1 means.

    So the spec is read off the DATA where the data carries it:
    ``(enr_main, enr_zone)`` for paramA, falling back to the compliance
    module's own ``(family letter, enrichment)`` where those columns are NaN
    (the whole ga80 library).  The first real 3-type campaign's case,
    ``P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24``, is mono-spec under this key
    (all 6.20/5.270, axial zones mixed) — as it was under the campaign's own
    reading, so nothing already run is reclassified.
    """

    from lpopt.data.compliance import family_anchor

    em, ez = row.get("enr_main"), row.get("enr_zone")
    if em is not None and ez is not None and np.isfinite(em) and np.isfinite(ez):
        return (round(float(em), 4), round(float(ez), 4))
    return (family_anchor(str(row.name)), round(float(row.u_avg_enrichment), 4))


def _mix_e_core(fuel, lib: str, types: list[str], weights: list[float]) -> float:
    """Composition-weighted core-average enrichment (the SHARED recipe).

    ``FuelLibrary.case_e_core`` is the same entry point the campaign, the
    resolver and ``produce`` use, so a case's nominal e_core here is the number
    those paths will compute for it later — not a second implementation that can
    drift.
    """

    return float(fuel.case_e_core(types, lib, weights))


def select_multitype(fuel, fuel_df: pd.DataFrame, store: pd.DataFrame,
                     picks: pd.DataFrame, log=lambda _m: None) -> pd.DataFrame:
    """For every primary v3 cell, the composition-matched TRIPLE and QUAD.

    **Registered rule (P3a).**  A k-type case is admissible for a cell when

    1. every member comes from the cell's OWN library (a core cannot mix design
       packages), is not ``feature_poor``, and — in paramA — is the descriptive
       spelling, not the package alias;
    2. the members form a STRICT ladder on the coordinate of :func:`_ladder`:
       ``lam_1 > lam_2 > ... > lam_k``.  k distinct rungs, no plateau — a
       "3-type" core with two equal rungs is a 2-type core wearing a third
       label;
    3. the enrichment spread ``max - min`` is within :data:`MAX_PAIR_SPREAD`,
       the same in-distribution cap the pair enumeration uses;
    4. **composition match**: the equal-share (1/k each) core-average
       enrichment is within :data:`MATCH_TOL` of the cell's 2-type e_core.  This
       is what makes ``delta(3-2)`` a controlled comparison — same e_core, same
       feed, same gate, finer ladder — instead of a comparison between two
       different cores.

    Among the admissible sets the pick is, in order:

    a. **R1 mono-spec** — every member shares one :func:`anchor_key`.  A
       cross-spec case mixes enrichment specifications across feed assemblies,
       which R1 bans.  The ban is enforced only in the roster/frontier export
       today (the campaign path does not check it) and how R1 should read a
       GRADED case was explicitly deferred as a policy call
       (``tripletype_design_20260817.md`` §A.3), so a cross-spec case is
       **selected and scored when nothing mono-spec exists** — and flagged
       ``mono_anchor_k = False``.  Nothing that carries that flag is allowed to
       become a MASTER anchor.
    b. **nestedness** — a case that CONTAINS both members of the cell's 2-type
       pair is preferred over one containing a single member, over one
       containing neither.  A nested triple is the pair plus a rung, so the
       delta is attributable to grading and not to a fuel change.
    c. **ladder symmetry** — the smallest normalised rung imbalance,
       ``std(gaps)/mean(gaps)``.  Even steps are what "graded" means.
    d. **|delta e_core|** — the tightest composition match.
    e. **store support** — the most existing rows carrying the new member(s), so
       the level-3 ``pair_ecore`` restart ladder has something to resolve to.
    """

    sup_type: dict[tuple[str, str], int] = {}
    for (lib, cp), n in store.groupby(["library_id", "case_pair"]).size().items():
        for t in str(cp).split("_"):
            sup_type[(lib, t)] = sup_type.get((lib, t), 0) + int(n)

    roster: dict[str, pd.DataFrame] = {}
    for lib in sorted(picks.library_id.unique()):
        g = fuel_df[(fuel_df.library_id == lib)
                    & (~fuel_df.feature_poor.astype(bool))]
        if lib == "paramA":
            g = g[g.type_id.str.match(_PARAMA_CANON)]
        roster[lib] = g.set_index("type_id")

    out = []
    for p in picks.itertuples():
        lib = p.library_id
        g = roster[lib]
        pair_members = {p.type_a, p.type_b}
        akey = {t: anchor_key(g.loc[t]) for t in g.index}
        rec: dict = dict(e_target=p.e_target, library_id=lib, e_core_cell=p.e_core,
                         mono_anchor_2=bool(akey.get(p.type_a) == akey.get(p.type_b)))
        for k in (3, 4):
            best = None
            for combo in itertools.combinations(sorted(g.index), k):
                rows = g.loc[list(combo)]
                enr = rows.u_avg_enrichment.astype(float)
                if not np.isfinite(enr).all():
                    continue
                if float(enr.max() - enr.min()) > MAX_PAIR_SPREAD:
                    continue
                lam, metric = _ladder(rows)
                order = np.argsort(-np.asarray(lam))          # hot -> cold
                lam_s = [lam[i] for i in order]
                if any(lam_s[i] - lam_s[i + 1] <= 0 for i in range(k - 1)):
                    continue                                   # not a strict ladder
                types = [combo[i] for i in order]
                try:
                    ec = _mix_e_core(fuel, lib, types, [1.0 / k] * k)
                except Exception:                              # noqa: BLE001
                    continue
                if not math.isfinite(ec) or abs(ec - p.e_core) > MATCH_TOL:
                    continue
                gaps = np.diff(-np.asarray(lam_s))
                asym = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 9.9
                nested = len(pair_members & set(combo))
                sup = sum(sup_type.get((lib, t), 0) for t in combo if t not in pair_members)
                mono = len({akey[t] for t in combo}) == 1
                key = (not mono, -nested, asym, abs(ec - p.e_core), -sup,
                       "_".join(types))
                if best is None or key < best[0]:
                    best = (key, types, ec, asym, nested, sup, metric, mono)
            if best is None:
                rec.update({f"case_{k}": "", f"e_core_{k}": np.nan,
                            f"asym_{k}": np.nan, f"nested_{k}": 0,
                            f"support_{k}": 0, f"ladder_{k}": "",
                            f"mono_anchor_{k}": False})
                continue
            _key, types, ec, asym, nested, sup, metric, mono = best
            rec.update({f"case_{k}": "_".join(types), f"e_core_{k}": ec,
                        f"asym_{k}": asym, f"nested_{k}": nested,
                        f"support_{k}": sup, f"ladder_{k}": metric,
                        f"mono_anchor_{k}": bool(mono)})
        out.append(rec)
        log(f"  e{p.e_target:.1f} {lib:7s} pair {p.pair} (e {p.e_core:.4f})")
        for k in (3, 4):
            c = out[-1][f"case_{k}"]
            if c:
                log(f"      k={k}: {c}  e {out[-1][f'e_core_{k}']:.4f} "
                    f"(de {out[-1][f'e_core_{k}'] - p.e_core:+.4f}) "
                    f"asym {out[-1][f'asym_{k}']:.3f} nested {out[-1][f'nested_{k}']}/2 "
                    f"ladder-by-{out[-1][f'ladder_{k}']} support {out[-1][f'support_{k}']} "
                    f"{'R1-mono' if out[-1][f'mono_anchor_{k}'] else 'R1-CROSS-SPEC (no anchor)'}")
            else:
                log(f"      k={k}: NONE — roster admits no composition-matched "
                    f"{k}-type ladder in this cell")
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 2. per-case model search
# --------------------------------------------------------------------------- #
def feasible5(m: np.ndarray) -> np.ndarray:
    """The FIVE-axis predicted-feasibility gate.

    v3 gated on four axes and checked the pin margin afterwards.  The 3-type
    campaign showed the pin axis actually moving (76.96 -> 75.53 measured), so
    it is promoted to the gate here.  ``n_feasible_4ax`` is also reported so the
    k=2 column can still be read against v3's four-axis count.
    """

    return ((m[:, 0] <= FR_LIMIT) & (m[:, 2] <= FQ_LIMIT) & (m[:, 1] <= CBC_LIMIT)
            & (np.abs(m[:, 4]) <= AO_LIMIT) & (m[:, 6] <= PIN_LIMIT))


def feasible4(m: np.ndarray) -> np.ndarray:
    """v3's four-axis gate, kept for cell-by-cell comparability."""

    return ((m[:, 0] <= FR_LIMIT) & (m[:, 2] <= FQ_LIMIT) & (m[:, 1] <= CBC_LIMIT)
            & (np.abs(m[:, 4]) <= AO_LIMIT))


def clean_but_fr(m: np.ndarray) -> np.ndarray:
    """Passes every axis EXCEPT F_r — the "joint-clean" set.

    This is the set the 3-type campaign's headline was measured on (F_r 1.5993
    at CBC 1597.33): the flattest core that is legal on everything else.  In a
    grid where almost no cell is F_r-feasible, the joint-clean F_r FLOOR is the
    decision-relevant number, not the count of fully feasible cores.
    """

    return ((m[:, 2] <= FQ_LIMIT) & (m[:, 1] <= CBC_LIMIT)
            & (np.abs(m[:, 4]) <= AO_LIMIT) & (m[:, 6] <= PIN_LIMIT))


def build_elites_multi(store: pd.DataFrame, donors: pd.DataFrame, lib: str,
                       types: list[str], e_core: float, feed: int,
                       rng: random.Random):
    """Elite parents for a k-type case, graded up from 2-type store elites.

    The 2-type parents are chosen exactly as v3 chooses them (feasibility-first
    band rows at the nearest feed + the globally flattest verified cores), then
    remapped onto the case's HOT and COLD rungs and walked up the alphabet with
    :func:`graded_morph`, which converts a radially contiguous slice of the most
    populated batch onto the least populated member.  Feed, wiring and
    depth-2 count are preserved exactly (the operator only relabels), so the
    seed keeps the elite's structure and differs from it only by the new rung —
    which is precisely the intervention under test.

    For k = 2 this is byte-identical to :func:`scoping_mesh.build_elites`.
    """

    from fr_transfer import pair_mapping, substitute
    from lpopt.data.schema import unpack_pattern
    from lpopt.search.genome import GeneralOrbitGenome, GenomeError, graded_morph

    hot, cold = types[0], types[-1]

    def _grade(pat):
        if len(types) < 3:
            return pat
        try:
            gen = GeneralOrbitGenome.from_pattern(
                pat, max_shuffle_depth=2, allow_single_cycle_discharge=True)
        except GenomeError:
            return None
        for _ in range(len(types) - 2):
            gen = graded_morph(gen, rng, types)
        if any(gen.batch_counts.get(t, 0) <= 0 for t in types):
            return None                      # the morph could not seat every rung
        try:
            return gen.to_pattern()
        except GenomeError:
            return None

    def _take(df):
        out = []
        for _, r in df.iterrows():
            try:
                pat = unpack_pattern(str(r["pattern"]))
                mp = pair_mapping(str(r.case_pair), f"{hot}_{cold}")
                pat = substitute(pat, mp)
            except Exception:                          # noqa: BLE001 — a parent is optional
                continue
            pat = _grade(pat)
            if pat is not None:
                out.append((str(r.record_id), pat))
        return out

    band = store[(store.library_id == lib) & (store.e_core.sub(e_core).abs() <= 0.05)
                 & (store.case_pair.str.count("_") == 1)]
    elites = []
    if len(band):
        b = band.assign(dfeed=(band.feed - feed).abs(), feas=_store_feasible(band))
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


def run_case(model, cfg, store, donors, lib, case_id, types, e_core, feed,
             k, log) -> tuple[dict, list[dict]]:
    """Two-wave model-only search of one (case, feed) cell at fresh-type count k."""

    from lpopt.search.construct import CaseContext, build_pool, screen_e_core_band

    w0, w1 = BUDGET[k]
    # require_all_batches: a k-type board that dropped a rung is PHYSICALLY a
    # (k-1)-type board.  Admitting it would let the k=3 pool win by quietly
    # becoming the k=2 pool, which is exactly the delta this sweep measures.
    ctx = CaseContext(pair=case_id, feed=feed, library_id=lib, e_core=e_core,
                      require_all_batches=(k > 2))
    rng = random.Random(SEED + feed * 13 + int(round(e_core * 100)) + 1000 * k)
    elites = build_elites_multi(store, donors, lib, types, e_core, feed,
                                random.Random(SEED + 7 * k))

    t0 = time.time()
    pool = build_pool(ctx, None, elites, set(), rng, cfg, wave_index=0, size=w0)
    pats = [c.pattern for c in pool]
    if not pats:
        return (dict(n_candidates=0, n_in_band=0, n_feasible=0, n_feasible_4ax=0,
                     seconds=round(time.time() - t0, 1), n_pareto=0), [])
    pred = model.predict(pats, ctx.case_key, e_core)
    m, sig = pred.mean, pred.epistemic_std

    band = screen_e_core_band(pats, model.fuel, lib, e_core, E_CORE_TOL)
    feas = feasible5(m) & band
    order = np.argsort(-np.where(feas, m[:, 3], -np.inf))[:24]
    prev_top = [(pool[i].record_id, pats[i]) for i in order if feas[i]]
    near = band & ~feas & (m[:, 0] <= FR_LIMIT + 0.10) & (m[:, 2] <= FQ_LIMIT + 0.15) \
        & (m[:, 1] <= CBC_LIMIT + 120) & (np.abs(m[:, 4]) <= AO_LIMIT) \
        & (m[:, 6] <= PIN_LIMIT + 2.0)
    nm_order = np.argsort(-np.where(near, m[:, 3], -np.inf))[:24]
    near_miss = [(pool[i].record_id, pats[i]) for i in nm_order if near[i]]

    ledger = {c.record_id for c in pool}
    pool1 = build_pool(ctx, None, elites, ledger, rng, cfg, wave_index=0,
                       prev_top=prev_top, near_miss_parents=near_miss, size=w1)
    if pool1:
        p1 = [c.pattern for c in pool1]
        pr1 = model.predict(p1, ctx.case_key, e_core)
        pool = pool + pool1
        pats = pats + p1
        m = np.vstack([m, pr1.mean])
        sig = np.vstack([sig, pr1.epistemic_std])
        band = np.concatenate([band, screen_e_core_band(p1, model.fuel, lib,
                                                        e_core, E_CORE_TOL)])
    feas = feasible5(m) & band
    clean = clean_but_fr(m) & band

    row: dict = dict(
        n_candidates=len(pats), n_in_band=int(band.sum()),
        n_feasible=int(feas.sum()),
        n_feasible_4ax=int((feasible4(m) & band).sum()),
        n_clean_but_fr=int(clean.sum()),
        min_pred_f_r=float(np.nanmin(m[band, 0])) if band.any() else np.nan,
        min_f_r_clean=float(np.nanmin(m[clean, 0])) if clean.any() else np.nan,
        min_pred_pin=float(np.nanmin(m[band, 6])) if band.any() else np.nan,
        max_pred_cyclen_any=float(np.nanmax(m[band, 3])) if band.any() else np.nan,
        seconds=round(time.time() - t0, 1))
    row.update(binding_constraint(m, band))
    # the joint-clean representative: flattest core that is legal on the other
    # four axes.  Defined in cells where nothing is F_r-feasible, which is most
    # of this grid — so this, not the feasible count, carries the delta.
    if clean.any():
        j = int(np.argmin(np.where(clean, m[:, 0], np.inf)))
        row.update(clean_f_r=float(m[j, 0]), clean_cyclen=float(m[j, 3]),
                   clean_cbc=float(m[j, 1]), clean_f_q=float(m[j, 2]),
                   clean_ao=float(m[j, 4]), clean_pin=float(m[j, 6]),
                   clean_record_id=pool[j].record_id,
                   clean_e_core=float(_pat_e_core(model, pats[j], lib)))
    else:
        row.update(clean_f_r=np.nan, clean_cyclen=np.nan, clean_cbc=np.nan,
                   clean_f_q=np.nan, clean_ao=np.nan, clean_pin=np.nan,
                   clean_record_id="", clean_e_core=np.nan)

    front_rows: list[dict] = []
    if feas.any():
        i = int(np.argmax(np.where(feas, m[:, 3], -np.inf)))
        row.update(pred_cyclen=float(m[i, 3]), sigma_cyclen=float(sig[i, 3]),
                   pred_f_r=float(m[i, 0]), pred_f_q=float(m[i, 2]),
                   pred_cbc_max=float(m[i, 1]), pred_ao_abs=float(m[i, 4]),
                   pred_max_pin_bu=float(m[i, 6]),
                   node_e_core=float(_pat_e_core(model, pats[i], lib)),
                   node_record_id=pool[i].record_id, node_origin=pool[i].origin)
        front = pareto_front(m, feas)
        row["n_pareto"] = int(front.size)
        k_knee = knee_index(m[front, 3], m[front, 0])
        for rank, idx in enumerate(front):
            rep = ("min_f_r" if rank == 0 else
                   "max_cyclen" if rank == len(front) - 1 else
                   "knee" if rank == k_knee else "")
            front_rows.append(dict(
                n_types=k, rank=rank, rep=rep, pred_cyclen=float(m[idx, 3]),
                sigma_cyclen=float(sig[idx, 3]), pred_f_r=float(m[idx, 0]),
                pred_f_q=float(m[idx, 2]), pred_cbc_max=float(m[idx, 1]),
                pred_ao_abs=float(m[idx, 4]), pred_max_pin_bu=float(m[idx, 6]),
                cand_e_core=float(_pat_e_core(model, pats[idx], lib)),
                record_id=pool[idx].record_id, origin=pool[idx].origin))
        row["pareto_min_f_r"] = float(m[front[0], 0])
        row["pareto_cyclen_span"] = float(m[front[-1], 3] - m[front[0], 3])
        row["pareto_f_r_span"] = float(m[front[-1], 0] - m[front[0], 0])
    else:
        row.update(pred_cyclen=np.nan, sigma_cyclen=np.nan, pred_f_r=np.nan,
                   pred_f_q=np.nan, pred_cbc_max=np.nan, pred_ao_abs=np.nan,
                   pred_max_pin_bu=np.nan, node_e_core=np.nan,
                   node_record_id="", node_origin="", n_pareto=0,
                   pareto_min_f_r=np.nan, pareto_cyclen_span=np.nan,
                   pareto_f_r_span=np.nan)
    log(f"    k={k} {case_id[:44]:44s} f{feed} | pool {row['n_candidates']:5d} "
        f"band {row['n_in_band']:5d} feas5 {row['n_feasible']:4d} "
        f"clean {row['n_clean_but_fr']:4d} | minF_r {row['min_pred_f_r']:.4f} "
        f"cleanF_r {row['min_f_r_clean']:.4f} | pin {row['clean_pin']:.2f} "
        f"| {row['seconds']:.0f}s")
    return row, front_rows


# --------------------------------------------------------------------------- #
# 3. main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--e-targets", default="")
    ap.add_argument("--feeds", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="s1i")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--cases-only", action="store_true")
    ap.add_argument("--cases-from", default="", help="re-use a case_selection.csv "
                    "written by an earlier --cases-only run.  The selection is "
                    "deterministic and enumerates C(37,4) sets per cell, so the "
                    "shards read it rather than each recomputing the same table.")
    ap.add_argument("--max-k", type=int, default=4)
    args = ap.parse_args()

    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir)
        if not OUT.is_absolute():
            OUT = BASE / OUT
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / f"run{args.tag}.log"

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(str(msg) + "\n")

    import torch
    torch.set_num_threads(int(args.threads))
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelLibrary

    cfg = load_config(str(BASE / "lpopt.inp"))
    fuel_df = pd.read_parquet(BASE / "data/store/fuel_types.parquet")
    fuel = FuelLibrary.from_parquet(BASE / "data/store/fuel_types.parquet")
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    store = store[(store.valid == True) & (store.converged == True)]   # noqa: E712

    picks = select_pairs(fuel, fuel_df, store, lambda _m: None)
    picks["per_fa_tU"], picks["M_HM_tU"] = zip(*[
        pair_hm_tu(fuel_df, r.library_id, r.type_a, r.type_b) for r in picks.itertuples()])
    key = picks.library_id + "|" + picks.pair
    picks["is_primary"] = ~key.duplicated()
    picks = picks[picks.is_primary]

    if args.cases_from:
        src = Path(args.cases_from)
        if not src.is_absolute():
            src = BASE / src
        sel = pd.read_csv(src)
        log(f"=== case selection READ from {src} ({len(sel)} cells) ===")
    else:
        log("=== composition-matched multi-type case selection (P3a rule) ===")
        cases = select_multitype(fuel, fuel_df, store, picks, log)
        sel = picks.merge(cases, on=["e_target", "library_id"], how="left")
        sel.to_csv(OUT / f"case_selection{args.tag}.csv", index=False,
                   encoding="utf-8")
        log(f"\nwrote {OUT / f'case_selection{args.tag}.csv'}")
    log(f"  triples found: {int((sel.case_3.fillna('') != '').sum())}/{len(sel)} cells")
    log(f"  quads   found: {int((sel.case_4.fillna('') != '').sum())}/{len(sel)} cells")
    if args.cases_only:
        return 0

    if args.e_targets:
        keep = {float(x) for x in args.e_targets.split(",")}
        sel = sel[sel.e_target.isin(keep)]

    from lpopt.model.model_api import PosValCnnBackend
    model = PosValCnnBackend.from_dir(BASE / "data/models" / args.model,
                                      store_dir=BASE / "data/store",
                                      library_id="ga80", device="cpu")
    model.quantile_targets = ()
    log(f"\nmodel: data/models/{args.model}, {len(model.members)} members, cpu, "
        f"{args.threads} torch threads, quantile heads off")
    donors = store[_store_feasible(store)].nsmallest(N_ELITE_DONOR * 2, "f_r")

    feeds = tuple(int(x) for x in args.feeds.split(",")) if args.feeds else FEEDS
    rows, fronts, t0 = [], [], time.time()
    for p in sel.itertuples():
        for feed in feeds:
            log(f"  cell e{p.e_target:.1f}_f{feed}  ({p.library_id})")
            cell = dict(cell=f"e{p.e_target:.1f}_f{feed}", e_target=p.e_target,
                        feed=feed, library_id=p.library_id,
                        e_core_cell=p.e_core, M_HM_tU=round(p.M_HM_tU, 3),
                        pair=p.pair)
            for k in (2, 3, 4):
                if k > args.max_k:
                    continue
                case_id = p.pair if k == 2 else getattr(p, f"case_{k}")
                if not isinstance(case_id, str) or not case_id:
                    cell[f"case_{k}"] = ""
                    cell[f"support_class_{k}"] = SUPPORT_CLASS[k]
                    continue
                types = case_id.split("_")
                ec = p.e_core if k == 2 else float(getattr(p, f"e_core_{k}"))
                # the CELL's e_core is the band centre for every k — that is what
                # makes the comparison composition-matched.  A case whose own
                # equal-share e_core sits up to MATCH_TOL away still has boards in
                # this band, because the split is an inner variable.
                r, front = run_case(model, cfg, store, donors, p.library_id,
                                    case_id, types, p.e_core, feed, k, log)
                bc = r.get("pred_cyclen", np.nan) * P_TH_MW / p.M_HM_tU / 1000.0
                r["B_cycle"] = bc
                r["B_d"] = bc * N_FA / feed
                bcc = r.get("clean_cyclen", np.nan) * P_TH_MW / p.M_HM_tU / 1000.0
                r["B_cycle_clean"] = bcc
                r["B_d_clean"] = bcc * N_FA / feed
                cell[f"case_{k}"] = case_id
                cell[f"e_core_{k}"] = ec
                cell[f"support_class_{k}"] = SUPPORT_CLASS[k]
                for kk, vv in r.items():
                    cell[f"{kk}_{k}"] = vv
                for fr in front:
                    bcf = fr["pred_cyclen"] * P_TH_MW / p.M_HM_tU / 1000.0
                    fr.update(cell=cell["cell"], e_target=p.e_target, feed=feed,
                              library_id=p.library_id, case_id=case_id,
                              support_class=SUPPORT_CLASS[k],
                              M_HM_tU=round(p.M_HM_tU, 3), B_cycle=bcf,
                              B_d=bcf * N_FA / feed)
                    fronts.append(fr)
            # ---- the deltas.  Negative F_r delta = grading helps. ---------- #
            for k in (3, 4):
                for stem in ("min_pred_f_r", "min_f_r_clean", "clean_cyclen",
                             "clean_pin", "clean_cbc", "n_feasible",
                             "n_clean_but_fr", "pred_cyclen"):
                    a, b = cell.get(f"{stem}_{k}"), cell.get(f"{stem}_2")
                    cell[f"d_{stem}_{k}v2"] = (
                        float(a) - float(b)
                        if a is not None and b is not None
                        and np.isfinite(np.float64(a)) and np.isfinite(np.float64(b))
                        else np.nan)
            fr_by_k = {k: cell.get(f"min_f_r_clean_{k}", np.nan) for k in (2, 3, 4)}
            live = {k: v for k, v in fr_by_k.items()
                    if v is not None and np.isfinite(np.float64(v))}
            cell["best_ntypes_by_clean_f_r"] = int(min(live, key=live.get)) if live else 0
            rows.append(cell)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"mesh_multitype{args.tag}.csv", index=False, encoding="utf-8")
    pf = pd.DataFrame(fronts)
    pf.to_csv(OUT / f"mesh_multitype_pareto{args.tag}.csv", index=False,
              encoding="utf-8")
    log(f"\nwrote {OUT / f'mesh_multitype{args.tag}.csv'}  "
        f"({time.time() - t0:.0f}s, {len(df)} cells, {len(pf)} pareto points)")
    if "d_min_f_r_clean_3v2" in df:
        d = df.d_min_f_r_clean_3v2.astype(float)
        log(f"delta(3-2) joint-clean F_r floor: n={int(d.notna().sum())} "
            f"mean {d.mean():+.4f} median {d.median():+.4f} "
            f"best {d.min():+.4f}  (negative = grading helps)")
        log(f"  cells with a gain > 0.005: {int((d < -0.005).sum())}")
    json.dump(dict(model=args.model, pin_limit=PIN_LIMIT, budget=BUDGET,
                   match_tol=MATCH_TOL, feeds=list(feeds),
                   support_class=SUPPORT_CLASS,
                   n_cells=len(df)),
              open(OUT / f"run_meta{args.tag}.json", "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
