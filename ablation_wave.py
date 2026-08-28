"""1-move ablation wave - interventional single-move labels on ONE cell.

WHY THIS EXISTS (pre-registration: ``data/reports/ablation_wave_prereg_20260815.md``)

The observational policy corpus (``data/reports/policy_corpus_20260815.md`` s4b/4g)
cannot separate a MOVE-CLASS effect from a RADIAL-DIRECTION effect, because no
campaign ever sampled direction at fixed class: ``rewire_swap`` is 100% radially
neutral BY CONSTRUCTION, so the corpus' "neutral" row IS its rewire row.  The two
eras then disagree on the SIGN of the cycle-length response to outward fresh
loading (``lpopt_genome`` d_cyclen -2.21 outward vs ``sa_mocha`` +0.14), and
nothing observational can arbitrate.  This module runs the intervention: from a
fixed set of parent boards it enumerates the COMPLETE verified single-move
neighbourhood, samples it balanced across (move_class x radial direction), and
sends every child to MASTER as a full equilibrium chain.

It also produces current-era (paramA / ga80-family) single-move labels, which is
what ``policy_v1`` lacks: that model is 93% SA-era 260624 and fails its baselines
on the ``heldout_era`` fold (``policy_v1_results_20260815.md`` s2).

DESIGN INVARIANT — every descriptor is computed by ``mine_policy_corpus``'s own
functions (``classify_move``, ``board_physics``, ``_direction``,
``SINGLE_MOVE_MAX_EDITS``).  Nothing is re-derived here, so a row this module
emits and a row ``build_steps`` mines for the same (parent, child) pair are the
same row.

Subcommands
-----------
``plan``     select parents, enumerate + classify the full single-move
             neighbourhood, dedup against the store, draw the stratified sample,
             write the manifest.  LOCAL, read-only, no MASTER.
``score``    blind policy-v1 CNN-ensemble predictions for every planned child,
             written BEFORE any label exists.  LOCAL, read-only.
``run``      evaluate the manifest's children as MASTER equilibrium chains and
             write a merge-ready kit.  REMOTE (box 199), self-contained: needs
             only the manifest + the paramA package + the MASTER exe.
``analyze``  the registered analyses over the returned labels.  LOCAL.
``corpus``   append the new lineage edges to ``data/policy/steps.parquet`` with
             ``lineage_source='ablation_paramA'``.  LOCAL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# --------------------------------------------------------------------------- #
# the cell, and everything that identifies its rows
# --------------------------------------------------------------------------- #
PAIR = "T6_T4"
FEED = 121
LIBRARY = "paramA"
CELL = f"{PAIR}/f{FEED}/{LIBRARY}"
CAMPAIGN = "ablation_1move_T6T4"
GENERATOR = "ablation_1move"
LINEAGE_SOURCE = "ablation_paramA"

#: Pre-registered seed.  Every random draw in this module is derived from it;
#: the sampler is in fact rank-deterministic (see ``_pick``), so the seed only
#: breaks ties.
SEED = 20260815

#: Parents.  Registered composition: top by F_r, top by node_peak, mid-band.
N_PARENTS = 10
N_TOP_FR = 4
N_TOP_FLAT = 3
N_MID = 3
#: Minimum pairwise 69-slot Hamming distance between accepted parents.
HAMMING_MIN = 12

#: PER-PARENT quota over the STRUCTURALLY REACHABLE strata.  See the
#: pre-registration s3: of the 4 x 3 = 12 (class x direction) cells, four are
#: structurally EMPTY (``rewire_swap`` cannot have a radial direction: it touches
#: neither the fresh set nor the batch labels, so d_fresh_enr_r_center is
#: IDENTICALLY zero) and the ``fresh_relocate``/``batch_swap`` neutral cells are
#: measure-zero radius coincidences that are excluded by design.
QUOTA: dict[tuple[str, str], int] = {
    ("rewire_swap", "neutral"): 3,
    ("fresh_relocate", "outward"): 3,
    ("fresh_relocate", "inward"): 3,
    ("batch_swap", "outward"): 2,
    ("batch_swap", "inward"): 2,
    ("batch_flip", "outward"): 1,
    ("batch_flip", "inward"): 1,
}
CHAINS_PER_PARENT = sum(QUOTA.values())          # 15
BUDGET_CAP = 160                                 # registered hard cap incl. reruns


# --------------------------------------------------------------------------- #
# enumeration — the COMPLETE single-move neighbourhood of one board
# --------------------------------------------------------------------------- #
def _finalize(fresh, wiring, template):
    from lpopt.search.genome import GenomeError

    cand = replace(template, fresh=tuple(sorted(fresh.items())),
                   wiring=tuple(sorted(wiring.items())))
    try:
        cand.validate()
    except GenomeError:
        return None
    return cand


def _validated(cand, template):
    from lpopt.search.genome import GenomeError

    try:
        cand.validate()
    except GenomeError:
        return None
    return None if cand == template else cand


def enum_rewire_swap(g) -> list[tuple[str, Any, str]]:
    """Every source exchange between two burned units (mirrors ``_rewire_swap``)."""
    wiring = dict(g.wiring)
    units = sorted(wiring)
    out = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            b1, b2 = units[i], units[j]
            if wiring[b1] == wiring[b2]:
                continue
            edit = dict(wiring)
            edit[b1], edit[b2] = wiring[b2], wiring[b1]
            cand = _finalize(dict(g.fresh), edit, g)
            if cand is not None and cand != g:
                out.append(("rewire_swap", cand, f"rw:{b1}<->{b2}"))
    return out


def enum_fresh_relocate(g) -> list[tuple[str, Any, str]]:
    """Every fresh/burned role exchange (mirrors ``_fresh_relocate`` exactly)."""
    from lpopt.search.genome import _consumer_map

    fresh_batches = dict(g.fresh)
    source_of = dict(g.wiring)
    consumer = _consumer_map(g)
    out = []
    for a in sorted(fresh_batches):
        for b in sorted(source_of):
            source_b = source_of[b]
            new_fresh = dict(fresh_batches)
            batch_a = new_fresh.pop(a)
            new_fresh[b] = batch_a
            new_wiring = dict(g.wiring)
            del new_wiring[b]
            if source_b == a:
                new_wiring[a] = b
            else:
                c = consumer.get(a)
                d = consumer.get(b)
                new_wiring[a] = source_b
                if c is not None:
                    new_wiring[c] = b
                if d is not None:
                    new_wiring[d] = a
            cand = _finalize(new_fresh, new_wiring, g)
            if cand is not None and cand != g:
                out.append(("fresh_relocate", cand, f"fr:{a}->{b}"))
    return out


def enum_batch_flip(g, batches: Sequence[str]) -> list[tuple[str, Any, str]]:
    """Every single batch-label repaint, fresh units AND the centre."""
    out = []
    fresh = dict(g.fresh)
    for unit in sorted(fresh):
        for nb in batches:
            if nb == fresh[unit]:
                continue
            edit = dict(fresh)
            edit[unit] = nb
            cand = _validated(
                replace(g, fresh=tuple(sorted(edit.items()))), g)
            if cand is not None:
                out.append(("batch_flip", cand, f"bf:{unit}={nb}"))
    for nb in batches:
        if nb == g.center_batch:
            continue
        cand = _validated(replace(g, center_batch=nb), g)
        if cand is not None:
            out.append(("batch_flip", cand, f"bf:center={nb}"))
    return out


def enum_batch_swap(g) -> list[tuple[str, Any, str]]:
    """Every label exchange between two fresh units of different batches."""
    fresh = dict(g.fresh)
    units = sorted(fresh)
    out = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            u1, u2 = units[i], units[j]
            if fresh[u1] == fresh[u2]:
                continue
            edit = dict(fresh)
            edit[u1], edit[u2] = edit[u2], edit[u1]
            cand = _validated(
                replace(g, fresh=tuple(sorted(edit.items()))), g)
            if cand is not None:
                out.append(("batch_swap", cand, f"bs:{u1}<->{u2}"))
    return out


def enumerate_single_moves(g) -> list[tuple[str, Any, str]]:
    batches = sorted({b for _, b in g.fresh} | {g.center_batch})
    return (enum_rewire_swap(g) + enum_fresh_relocate(g)
            + enum_batch_flip(g, batches) + enum_batch_swap(g))


# --------------------------------------------------------------------------- #
# annotation — mine_policy_corpus's own descriptors, never re-derived
# --------------------------------------------------------------------------- #
def _fresh_enr_mass(packed: str, enr: dict[str, float] | None) -> float:
    """Multiplicity-weighted TOTAL fresh enrichment — the reactivity covariate.

    ``fresh_enr_r_center`` is a NORMALIZED first moment, so it is blind to a
    change in the total.  ``batch_flip`` changes the fresh batch multiset and
    therefore this total; ``batch_swap`` / ``fresh_relocate`` / ``rewire_swap``
    conserve it exactly.  Carrying it makes the confound measurable instead of
    assumed (pre-registration s4.2).
    """
    import mine_policy_corpus as M

    if enr is None:
        return float("nan")
    mask, batches = M.fresh_slots(packed)
    weight = M.SLOT_MULT * mask
    e = np.array([M._enrichment_of(enr, b) if m else 0.0
                  for b, m in zip(batches, mask)])
    return float(np.nansum(weight * e))


def annotate(parent_packed: str, parent_genome, children, enr):
    """One dict per child: class, direction, dose, and the corpus descriptors."""
    import mine_policy_corpus as M
    from lpopt.data.schema import pack_pattern, unpack_pattern

    p_phys = M.board_physics(parent_packed, parent_genome, enr)
    p_pattern = unpack_pattern(parent_packed)
    p_mass = _fresh_enr_mass(parent_packed, enr)

    rows: list[dict[str, Any]] = []
    for intended, cg, tag in children:
        packed = pack_pattern(cg.to_pattern())
        diff = M.classify_move(parent_genome, cg)
        # The enumerators are exact, but the CLASSIFIER is the authority: a
        # candidate whose net diff does not read back as the intended class (or
        # exceeds the single-move edit bound) is dropped, not relabelled.
        single = diff.n_unit_edits <= M.SINGLE_MOVE_MAX_EDITS.get(
            diff.move_class, -1)
        if diff.move_class != intended or not single:
            continue
        c_phys = M.board_physics(packed, cg, enr)
        d_center = c_phys["fresh_enr_r_center"] - p_phys["fresh_enr_r_center"]
        direction = ("outward" if d_center > M.RADIAL_EPS
                     else "inward" if d_center < -M.RADIAL_EPS else "neutral")
        if diff.swap_units is None:
            span = radius = float("nan")
        else:
            r1 = M.ORBIT_UNITS[diff.swap_units[0]].radius
            r2 = M.ORBIT_UNITS[diff.swap_units[1]].radius
            span, radius = abs(r1 - r2), 0.5 * (r1 + r2)
        row: dict[str, Any] = {
            "move_tag": tag,
            "move_class": diff.move_class,
            "fresh_radial_dir": direction,
            "d_fresh_enr_r_center": d_center,
            "dose": abs(d_center),
            "n_unit_edits": int(diff.n_unit_edits),
            "n_slots_changed": int(p_pattern.hamming(unpack_pattern(packed))),
            "swap_span": span,
            "swap_radius": radius,
            "single_move": True,
            "d_fresh_enr_mass": _fresh_enr_mass(packed, enr) - p_mass,
            "pattern": packed,
        }
        for name in M.PHYSICS:
            row[f"parent_{name}"] = p_phys[name]
            row[f"child_{name}"] = c_phys[name]
            row[f"d_{name}"] = c_phys[name] - p_phys[name]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# parent selection
# --------------------------------------------------------------------------- #
def select_parents(store, enr, *, n_parents: int = N_PARENTS,
                   hamming_min: int = HAMMING_MIN, log=print):
    """Diverse, feasible, converged parents of ``CELL``.

    Priority order interleaves the F_r tail, the node_peak tail and the mid-band
    so a Hamming rejection costs the *next* candidate of the same family, never
    the whole family.  Deterministic: no RNG.
    """
    import mine_policy_corpus as M
    import pandas as pd
    from lpopt.data.schema import unpack_pattern

    cell = store[(store["case_pair"] == PAIR) & (store["feed"] == FEED)
                 & (store["library_id"] == LIBRARY)].copy()
    cell["feasible"] = M.feasibility(cell)
    pool = cell[cell["feasible"].fillna(False).astype(bool).to_numpy()].copy()
    pool = pool[pool["f_r"].notna() & pool["node_peak"].notna()]
    log(f"[plan] cell {CELL}: {len(cell)} rows, {len(pool)} feasible+converged")

    by_fr = pool.sort_values("f_r", kind="mergesort")
    by_flat = pool.sort_values("node_peak", kind="mergesort")
    lo, hi = pool["f_r"].quantile([0.40, 0.60])
    mid = (pool[(pool["f_r"] >= lo) & (pool["f_r"] <= hi)]
           .sort_values("f_r", kind="mergesort"))

    queues = {
        "top_fr": list(by_fr["record_id"]),
        "top_flat": list(by_flat["record_id"]),
        "mid_band": list(mid["record_id"]),
    }
    want = {"top_fr": N_TOP_FR, "top_flat": N_TOP_FLAT, "mid_band": N_MID}
    order = (["top_fr", "top_flat", "mid_band"] * max(want.values()))

    indexed = pool.set_index("record_id", drop=False)
    patterns: dict[str, Any] = {}

    def pat(rid):
        if rid not in patterns:
            patterns[rid] = unpack_pattern(indexed.loc[rid, "pattern"])
        return patterns[rid]

    accepted: list[tuple[str, str]] = []
    taken: dict[str, int] = {k: 0 for k in want}
    seen: set[str] = set()
    for family in order:
        if len(accepted) >= n_parents or taken[family] >= want[family]:
            continue
        for rid in queues[family]:
            if rid in seen:
                continue
            p = pat(rid)
            if all(p.hamming(pat(a)) >= hamming_min for a, _ in accepted):
                accepted.append((rid, family))
                seen.add(rid)
                taken[family] += 1
                break
    # Top-up (relaxed): the quota shape is a preference, N_PARENTS is the contract.
    if len(accepted) < n_parents:
        for rid in queues["top_fr"]:
            if len(accepted) >= n_parents:
                break
            if rid in seen:
                continue
            p = pat(rid)
            if all(p.hamming(pat(a)) >= hamming_min for a, _ in accepted):
                accepted.append((rid, "topup_fr"))
                seen.add(rid)

    rows = []
    for rid, family in accepted:
        r = indexed.loc[rid]
        phys = M.board_physics(r["pattern"], M.genome_of(r["pattern"]), enr)
        rows.append({
            "record_id": rid, "family": family, "pattern": r["pattern"],
            "campaign": r["campaign"], "f_r": float(r["f_r"]),
            "node_peak": float(r["node_peak"]), "cyclen": float(r["cyclen"]),
            "cbc_max": float(r["cbc_max"]), "f_q": float(r["f_q"]),
            "e_core": (None if pd.isna(r["e_core"]) else float(r["e_core"])),
            **{k: phys[k] for k in M.PHYSICS},
        })
    ham = [[int(pat(a["record_id"]).hamming(pat(b["record_id"])))
            for b in rows] for a in rows]
    return rows, ham


# --------------------------------------------------------------------------- #
# stratified sampling
# --------------------------------------------------------------------------- #
def _pick(frame, k: int, key: str) -> list[int]:
    """``k`` evenly spaced RANKS of ``frame`` sorted by ``key`` — deterministic.

    Even ranks over the dose order (not a random draw) so each stratum spans its
    own dose range: the direction contrast is then also a dose-response readout,
    and outward / inward samples of the same class occupy the SAME quantile
    positions, which is what makes them dose-matched.
    """
    if len(frame) == 0 or k <= 0:
        return []
    order = frame.sort_values([key, "move_tag"], kind="mergesort").index.tolist()
    if len(order) <= k:
        return order
    pos = np.linspace(0, len(order) - 1, k)
    return [order[int(round(p))] for p in dict.fromkeys(pos.tolist()).keys()]


def stratified_sample(cand, quota: dict[tuple[str, str], int]):
    """Draw ``quota`` per (move_class, direction), dose-spread within stratum."""
    keep: list[int] = []
    shortfall: list[dict[str, Any]] = []
    for (cls, direction), k in quota.items():
        sub = cand[(cand["move_class"] == cls)
                   & (cand["fresh_radial_dir"] == direction)]
        key = "swap_span" if direction == "neutral" else "dose"
        if sub[key].isna().all():
            key = "n_slots_changed"
        idx = _pick(sub, k, key)
        keep.extend(idx)
        if len(idx) < k:
            shortfall.append({"move_class": cls, "fresh_radial_dir": direction,
                              "want": k, "got": len(idx), "available": len(sub)})
    return cand.loc[keep], shortfall


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def cmd_plan(args) -> int:
    import mine_policy_corpus as M
    import pandas as pd
    from lpopt.data.schema import compute_record_id, unpack_pattern
    from lpopt.search.verify import PRODUCE_DECK_KNOBS

    store = pd.read_parquet(BASE / "data/store/records.parquet")
    enr_all = M.load_enrichment(BASE / "data/store/fuel_types.parquet")
    enr = enr_all.get(LIBRARY)
    if enr is None:
        raise SystemExit(f"no enrichment rows for library {LIBRARY!r}")
    known = set(store["record_id"].astype(str))

    parents, ham = select_parents(store, enr)
    print(f"[plan] {len(parents)} parents, pairwise Hamming min "
          f"{min(h for row in ham for h in row if h) if len(parents) > 1 else 0}")

    fuel = None
    try:
        from lpopt.data.fuel_types import FuelLibrary
        fuel = FuelLibrary.from_parquet(BASE / "data/store/fuel_types.parquet")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[plan] WARNING: fuel table unreadable ({exc}); e_core left null")

    from lpopt.search.construct import predicted_e_core

    all_rows: list[dict[str, Any]] = []
    census: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    for p in parents:
        pg = M.genome_of(p["pattern"])
        children = enumerate_single_moves(pg)
        rows = annotate(p["pattern"], pg, children, enr)
        frame = pd.DataFrame(rows)
        frame = frame.drop_duplicates(subset=["pattern"])
        frame["parent_record_id"] = p["record_id"]
        frame["record_id"] = [
            compute_record_id(x, LIBRARY, PAIR, PRODUCE_DECK_KNOBS)
            for x in frame["pattern"]
        ]
        frame["already_labeled"] = frame["record_id"].isin(known)
        for (cls, direction), n in (
                frame.groupby(["move_class", "fresh_radial_dir"]).size().items()):
            census.append({"parent_record_id": p["record_id"], "move_class": cls,
                           "fresh_radial_dir": direction, "n_available": int(n),
                           "n_already_labeled": int(frame[
                               (frame["move_class"] == cls)
                               & (frame["fresh_radial_dir"] == direction)
                           ]["already_labeled"].sum())})
        # Dedup FIRST: an already-labelled child is a FREE label and must not
        # consume a MASTER slot.  It is still recorded (as ``free``) so the
        # stratum's n is the paid + free total.
        free = frame[frame["already_labeled"]].copy()
        paid_pool = frame[~frame["already_labeled"]].copy()
        picked, short = stratified_sample(paid_pool, QUOTA)
        for s in short:
            s["parent_record_id"] = p["record_id"]
        shortfalls.extend(short)
        picked = picked.copy()
        picked["source"] = "paid"
        free = free[free["move_class"].isin({c for c, _ in QUOTA})]
        free["source"] = "free"
        block = pd.concat([picked, free], ignore_index=True)
        block["parent_f_r"] = p["f_r"]
        block["parent_node_peak"] = p["node_peak"]
        block["parent_cyclen"] = p["cyclen"]
        block["parent_cbc_max"] = p["cbc_max"]
        block["e_core"] = [
            predicted_e_core(unpack_pattern(x), fuel, LIBRARY) for x in block["pattern"]
        ] if fuel is not None else None
        all_rows.append(block)
        print(f"[plan]   parent {p['record_id'][:12]} ({p['family']:<9}) "
              f"F_r {p['f_r']:.4f}  neighbourhood {len(frame):5d}  "
              f"free {int(frame['already_labeled'].sum()):3d}  paid {len(picked):3d}")

    cand = pd.concat(all_rows, ignore_index=True)
    paid = cand[cand["source"] == "paid"]
    if len(paid) > BUDGET_CAP:
        raise SystemExit(f"paid chains {len(paid)} exceed the registered cap "
                         f"{BUDGET_CAP}")

    manifest = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell": CELL, "pair": PAIR, "feed": FEED, "library_id": LIBRARY,
        "campaign": CAMPAIGN, "generator": GENERATOR, "seed": SEED,
        "deck_knobs": PRODUCE_DECK_KNOBS,
        "quota": {f"{c}|{d}": n for (c, d), n in QUOTA.items()},
        "budget_cap": BUDGET_CAP,
        "n_paid": int(len(paid)), "n_free": int((cand["source"] == "free").sum()),
        "parents": parents,
        "parent_hamming": ham,
        "shortfalls": shortfalls,
        "neighbourhood_census": census,
        "candidates": json.loads(cand.to_json(orient="records")),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[plan] paid {manifest['n_paid']}  free {manifest['n_free']}  -> {out}")
    if shortfalls:
        print(f"[plan] SHORTFALLS ({len(shortfalls)}):")
        for s in shortfalls:
            print(f"[plan]   {s['parent_record_id'][:12]} {s['move_class']}/"
                  f"{s['fresh_radial_dir']}: want {s['want']} got {s['got']} "
                  f"(available {s['available']})")
    return 0


# --------------------------------------------------------------------------- #
# blind policy-v1 scoring
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, pair, feed, library_id):
        self.pair, self.feed, self.library_id = pair, feed, library_id


def cmd_score(args) -> int:
    import mine_policy_corpus as M
    import pandas as pd
    from lpopt.data.schema import unpack_pattern
    from lpopt.policy.scorer import MoveScorer

    manifest = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cand = pd.DataFrame(manifest["candidates"])
    scorer = MoveScorer.load(args.model_dir,
                             fuel_types=BASE / "data/store/fuel_types.parquet",
                             device="cpu", n_threads=args.threads)
    print(f"[score] {len(scorer.members)} ensemble members from {args.model_dir}")
    ctx = _Ctx(PAIR, FEED, LIBRARY)
    parents = {p["record_id"]: p["pattern"] for p in manifest["parents"]}

    frames = []
    for prid, block in cand.groupby("parent_record_id", sort=False):
        p_packed = parents[prid]
        p_pair = (M.genome_of(p_packed), unpack_pattern(p_packed))
        kids = [(M.genome_of(x), unpack_pattern(x)) for x in block["pattern"]]
        probs = scorer.score(p_pair, kids, ctx)
        out = block[["record_id", "parent_record_id", "move_class",
                     "fresh_radial_dir", "dose", "source"]].copy()
        out["p_improve_fr"] = probs[:, 0]
        out["p_improve_flat"] = probs[:, 1]
        # The validated readout is PARENT-BLOCKED: scores from different parents
        # are not comparable (policy_v1_results s1).  The within-parent rank is
        # therefore carried explicitly so the prospective test uses it.
        out["rank_fr_in_parent"] = out["p_improve_fr"].rank(ascending=False)
        out["rank_flat_in_parent"] = out["p_improve_flat"].rank(ascending=False)
        frames.append(out)
        print(f"[score]   {prid[:12]}  n={len(out)}  "
              f"p_fr {probs[:, 0].min():.3f}..{probs[:, 0].max():.3f}")

    pred = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    print(f"[score] {len(pred)} predictions -> {out_path}")
    print(f"[score] REGISTERED BEFORE TRUTH: sha256 of the prediction file is "
          f"the pre-commitment; see the pre-registration s6.")
    return 0


# --------------------------------------------------------------------------- #
# run — MASTER equilibrium chains (executes on box 199)
# --------------------------------------------------------------------------- #
RESULTS_NAME = "ablation_results.jsonl"


def is_settled(status: str, failure: str) -> bool:
    """Is this jsonl row a FINAL answer about the pattern, or a harness fault?

    Settled — do not re-run:

    * ``converged``    — an equilibrium label;
    * ``nonconverged`` — an honest non-convergence (cap-exhausted); re-running
      it would just burn the same budget for the same answer;
    * ``error`` whose failure is a PHYSICS kill
      (:data:`lpopt.search.verify.PHYSICS_KILL_FAILURES`, e.g.
      ``non_finite_flux``) — an honest NEGATIVE label about the pattern.

    NOT settled — must be re-run:

    * any other ``error`` — staging / deck / resolver / timeout / **out of disk**
      / MASTER exit status.  These say nothing about the pattern, only about the
      box.

    This distinction is the project's own (``verify.classify_outcome``), reused
    rather than re-derived.  It exists because the 2026-08-15 625-branch wave hit
    ``[Errno 28] No space left on device`` mid-run: the old resume treated ANY
    row in the jsonl as done, so re-launching would silently have skipped the 18
    harness failures and produced a quietly short wave.  A crash must never
    become a truncated dataset that looks complete.
    """
    from lpopt.search.verify import PHYSICS_KILL_FAILURES

    if status in ("converged", "nonconverged"):
        return True
    if status == "error":
        return any(kill in (failure or "") for kill in PHYSICS_KILL_FAILURES)
    return False


def _done(results_path: Path) -> set[str]:
    """Record ids whose result is SETTLED (see :func:`is_settled`).

    Harness-failed rows are deliberately excluded so a resume re-runs them.
    """
    done: set[str] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:                                   # noqa: BLE001
                continue
            if is_settled(str(row.get("status", "")), str(row.get("failure", ""))):
                done.add(str(row["record_id"]))
    return done


def cmd_run(args) -> int:
    import numpy as np
    from lpopt.data.flatness import map_cov, node_peak, slot_values
    from lpopt.data.schema import unpack_pattern
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.search.verify import WaveEntry, WaveVerifier
    from lpopt.vendor.masterrl.domain import CaseKey

    manifest = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cand = [c for c in manifest["candidates"] if c["source"] == "paid"]
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / RESULTS_NAME
    done = _done(results_path)

    pkg = Path(args.package).resolve()
    if not pkg.is_dir():
        raise SystemExit(f"package not found: {pkg}")
    dims = _library_dims(pkg)
    key = CaseKey(PAIR, FEED)
    resolver = CaseAssetResolver(pkg, library_dims=dims)
    assets = resolver.resolve(key)

    print(f"[run] cell {CELL}  candidates {len(cand)}  done {len(done)}")
    print(f"[run] package {pkg}  library_dims={dims}")
    print(f"[run] assets  {key.label}  fallback_level={assets.fallback_level} "
          f"restart={assets.restart_provenance}")
    print(f"[run] deck    {assets.template_deck_path}")
    # Asset fingerprint: the restart carries the burnt-fuel history every chain
    # inherits, so a cross-box comparison is only meaningful if these hashes
    # match the box the parents were labelled on (v520_run.py precedent).
    for label, path in (("restart", getattr(assets, "restart_path", None)),
                        ("deck", assets.template_deck_path)):
        try:
            import hashlib
            h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
            print(f"[run] sha256 {label:<8} {h}  {Path(path).name}")
        except Exception as exc:                                # noqa: BLE001
            print(f"[run] sha256 {label:<8} unavailable ({exc})")
    if assets.fallback_level != 0:
        print("!" * 78)
        print(f"!! {PAIR}/f{FEED} resolves at fallback_level="
              f"{assets.fallback_level}: the restart is NOT this cell's own.")
        print("!! Every chain would carry a foreign burnt-fuel history and the")
        print("!! parent/child delta would mix the move with a restart change.")
        print("!" * 78)
        if not args.allow_fallback:
            raise SystemExit("refusing to run on a fallback restart "
                             "(--allow-fallback to override)")

    todo = [c for c in cand if c["record_id"] not in done]
    if args.max_chains:
        todo = todo[:int(args.max_chains)]
    print(f"[run] {len(todo)} chain(s) to evaluate this invocation")
    if args.dry_run:
        print("[run] DRY RUN - no MASTER launched")
        return 0
    if not todo:
        print("[run] nothing left to run")
        return 0

    verifier = WaveVerifier(
        run_dir=run_dir, package_root=pkg, executable=args.exe,
        workers=args.workers, timeout=args.timeout, max_cycles=args.max_cycles,
        consecutive=2, library_dims=dims, harvest_maps=True,
        use_all_cores=False, host_reserve=args.host_reserve,
    )
    t0 = time.time()
    # Fixed-size waves so a mid-run crash costs one wave, not the campaign, and
    # so the jsonl is flushed incrementally (resume reads it back).
    for start in range(0, len(todo), args.wave_size):
        chunk = todo[start:start + args.wave_size]
        wave = [
            WaveEntry(unpack_pattern(c["pattern"]), key, assets,
                      {"record_id": c["record_id"],
                       "parent_record_id": c["parent_record_id"],
                       "move_class": c["move_class"],
                       "fresh_radial_dir": c["fresh_radial_dir"],
                       "move_tag": c["move_tag"], "e_core": c.get("e_core")})
            for c in chunk
        ]
        outcomes = verifier.evaluate_wave(wave)
        with results_path.open("a", encoding="utf-8") as fh:
            for oc in outcomes:
                rid = str(oc.meta.get("record_id"))
                rec = {
                    "record_id": rid,
                    "parent_record_id": oc.meta.get("parent_record_id"),
                    "move_class": oc.meta.get("move_class"),
                    "fresh_radial_dir": oc.meta.get("fresh_radial_dir"),
                    "move_tag": oc.meta.get("move_tag"),
                    "e_core": oc.meta.get("e_core"),
                    "pair": oc.case_key.pair, "feed": oc.case_key.feed,
                    "pattern": oc.pattern.canonical(),
                    "status": oc.status, "n_cycles": oc.n_cycles,
                    "converged_at_cap": oc.converged_at_cap,
                    "tolerance_margin": oc.tolerance_margin,
                    "wall_s": oc.wall_s,
                    "restart_provenance": oc.restart_provenance,
                    "fallback_level": assets.fallback_level,
                    "failure": oc.failure,
                    "fom": oc.fom.as_dict() if oc.fom else None,
                }
                if oc.maps is not None:
                    arr = np.asarray(oc.maps, dtype=float)
                    sv = slot_values(arr)
                    rec["node_peak"] = float(node_peak(sv)[0])
                    rec["map_cov"] = float(map_cov(sv)[0])
                    np.save(run_dir / f"map_{rid[:12]}.npy", arr)
                else:
                    rec["node_peak"] = rec["map_cov"] = None
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        conv = sum(1 for oc in outcomes if oc.status == "converged")
        print(f"[run] wave {start // args.wave_size}: {conv}/{len(chunk)} "
              f"converged  elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"[run] total wall {time.time() - t0:.1f}s -> {results_path}")
    _build_kit(run_dir)
    return 0


def _library_dims(package_root: Path) -> tuple[int, int]:
    """``(nbatch, ncomp)`` read off the package's own ``lib/MAS_XSL``.

    Verbatim the recipe ``fr_arms.library_dims`` / ``fr_transfer.library_dims``
    use — the paramA package is (40, 42), not the ga80 default (83, 85), and a
    wrong dims pair fails staging rather than silently mis-running.
    """
    xsl = (package_root / "lib" / "MAS_XSL").read_text(errors="replace")
    comp = sum(1 for ln in xsl.splitlines() if ln.startswith("COMP "))
    refl = sum(1 for ln in xsl.splitlines() if ln.startswith("REFL "))
    return (comp + 3, comp + refl)


def _fom_from_dict(d: dict | None):
    """Rebuild the vendor :class:`FOM` from its ``as_dict()`` round-trip."""
    if not d:
        return None
    from lpopt.vendor.masterrl.domain import FOM

    return FOM(
        f_r=d.get("F_r"), cbc_max=d.get("CBC_max"), f_q=d.get("F_q"),
        cyclen=d.get("cyclen"), ao_min=d.get("AO_min"), ao_max=d.get("AO_max"),
        max_burnup=d.get("max_burnup"), max_pin_burnup=d.get("max_pin_burnup"),
        max_burnup_assembly=d.get("max_burnup_assembly"),
        max_burnup_pin=d.get("max_burnup_pin"),
        converged=bool(d.get("converged", True)),
    )


def _build_kit(run_dir: Path) -> Path:
    """Write a ``merge-store``-shaped kit holding ONLY this wave's rows.

    Scoped on purpose: shipping the box's whole 21 MB ``records.parquet`` + 196 MB
    ``maps.npz`` back is neither necessary nor safe (the kit copy has diverged
    from the canonical store).  This kit is ~150 rows and merges by ``record_id``.

    The rows are built by ``outcome_to_record`` from a REBUILT ``WaveOutcome``
    whose ``maps`` is the saved plane, so ``node_peak`` / ``map_cov`` / ``maps_key``
    are filled by the same code path a live campaign uses — no hand-set columns.
    """
    import numpy as np
    from lpopt.data.schema import unpack_pattern
    from lpopt.data.store import StoreWriter
    from lpopt.search.verify import (
        PRODUCE_DECK_KNOBS, WaveOutcome, outcome_to_record,
    )
    from lpopt.vendor.masterrl.domain import CaseKey

    results_path = run_dir / RESULTS_NAME
    if not results_path.exists():
        print("[kit] no results yet")
        return run_dir
    kit = run_dir / "kitdata"
    (kit / "store").mkdir(parents=True, exist_ok=True)
    (kit / "produce").mkdir(parents=True, exist_ok=True)

    records, maps, ledger_lines = [], {}, []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        map_path = run_dir / f"map_{r['record_id'][:12]}.npy"
        plane = np.load(map_path) if map_path.is_file() else None
        oc = WaveOutcome(
            status=r["status"], fom=_fom_from_dict(r.get("fom")),
            n_cycles=int(r.get("n_cycles") or 0),
            tolerance_margin=r.get("tolerance_margin"),
            wall_s=float(r.get("wall_s") or 0.0),
            restart_provenance=str(r.get("restart_provenance") or ""),
            failure=str(r.get("failure") or ""),
            converged_at_cap=bool(r.get("converged_at_cap")),
            case_key=CaseKey(r["pair"], int(r["feed"])),
            pattern=unpack_pattern(r["pattern"]), meta={}, maps=plane,
        )
        stratum = f"{r['move_class']}|{r['fresh_radial_dir']}"
        rec = outcome_to_record(
            oc, dataset="P", library_id=LIBRARY, stratum=stratum,
            generator=GENERATOR, parent_record_id=r["parent_record_id"],
            campaign=CAMPAIGN, e_core=r.get("e_core"), e_split=None,
            deck_knobs=PRODUCE_DECK_KNOBS)
        if rec.record_id != r["record_id"]:
            raise SystemExit(
                f"record_id drift: planned {r['record_id'][:12]} but "
                f"outcome_to_record minted {rec.record_id[:12]} - the deck-knob "
                f"or library/pair identity does not match the plan")
        if plane is not None:
            maps[rec.record_id] = np.asarray(plane, dtype=np.float16)
        records.append(rec)
        ledger_lines.append(json.dumps({
            "record_id": rec.record_id, "status": r["status"],
            "campaign": CAMPAIGN, "stratum": stratum,
            "generator": GENERATOR,
            "parent_record_id": r["parent_record_id"],
            "move_tag": r["move_tag"], "wall_s": r.get("wall_s"),
        }, sort_keys=True))

    writer = StoreWriter(kit / "store")
    info = writer.write_records(records, append=False)
    if maps:
        writer.write_maps(maps, append=False)
    (kit / "produce" / "ledger.jsonl").write_text(
        "\n".join(ledger_lines) + "\n", encoding="utf-8")
    print(f"[kit] {info['new']} records, {len(maps)} maps -> {kit}")
    return kit


def cmd_kit(args) -> int:
    _build_kit(Path(args.run_dir).resolve())
    return 0


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="select parents + draw the stratified sample")
    p.add_argument("--out", default="data/design/ablation_wave_20260815.json")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("score", help="blind policy-v1 predictions (BEFORE truth)")
    p.add_argument("--plan", default="data/design/ablation_wave_20260815.json")
    p.add_argument("--model-dir", default="data/models/policy_v1")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="data/design/ablation_wave_policy_v1_pred.csv")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("run", help="MASTER equilibrium chains (box 199)")
    p.add_argument("--plan", default="data/design/ablation_wave_20260815.json")
    p.add_argument("--package", default="data/design/package")
    p.add_argument("--exe", default="C:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    p.add_argument("--run-dir", default="runs/ablation_1move_T6T4")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--host-reserve", type=int, default=1)
    p.add_argument("--wave-size", type=int, default=8)
    p.add_argument("--max-cycles", type=int, default=16)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--max-chains", type=int, default=0)
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("kit", help="rebuild the merge-store kit from the jsonl")
    p.add_argument("--plan", default="data/design/ablation_wave_20260815.json")
    p.add_argument("--run-dir", default="runs/ablation_1move_T6T4")
    p.set_defaults(func=cmd_kit)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
