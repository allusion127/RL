"""Campaign A - Causal Move Atlas: paired single-move interventions on N cells.

WHY THIS EXISTS (review ``RL_core_loading_engineer_AI_review_2026-08-29.md``
s7.2 D1 / s7.4 Campaign A; pre-registration
``data/reports/intervention_wave_r1_prereg_20260829.md``)

The 2026-08-15 ablation wave answered the direction question on ONE cell
(``T6_T4/f121/paramA``).  Campaign A asks it on the F_xy frontier as a whole:
the review's D1 layer wants, from one parent, a move and its SYMMETRIC SIBLING
computed together, so parent difficulty differences out and the sign of a
feature is verified rather than inferred from an observational correlation.
``policy_v2_results_20260817.md`` s8/s9 names exactly what the corpus still
lacks - current-era single-move labels ACROSS CELLS, balanced on
(move_class x radial direction) - and that is what this module produces.

It is also the programme's first campaign whose primary response is **F_xy**
(MASTER FXYP, pin planar peaking, hard limit 1.65).  F_xy is printed ONLY in
``MAS_OUT``, so every chain here runs with the retention chain the F_xy-era
produce round uses (``harvest_maps=True`` forces ``keep_success``), and the
parsed value comes from :mod:`lpopt.data.fxy` - never re-derived here.

REUSE, NOT DUPLICATION
----------------------
The enumerator (:func:`ablation_wave.enumerate_single_moves`), the annotator
(:func:`ablation_wave.annotate`, which is itself a thin shell over
``mine_policy_corpus``'s ``classify_move`` / ``board_physics`` / ``_direction``
/ ``SINGLE_MOVE_MAX_EDITS``), the runner (:func:`ablation_wave.cmd_run`) and the
kit builder (:func:`ablation_wave._build_kit`) are ``ablation_wave``'s, imported
here.  ``ablation_wave.py`` is NOT edited: its sha256 is pinned in
``ablation_wave_prereg_20260815.md`` s8 and it is the artefact that ran on 199.
Per-cell provenance is REBOUND on the imported module, exactly as
``batchswap_wave.py`` does it - only here the rebinding happens once per cell
inside :func:`_cell_binding`, because this wave spans several cells in one plan.

THE ONE THING ``ablation_wave`` CANNOT DO, AND HOW IT IS ADDED WITHOUT EDITING IT
--------------------------------------------------------------------------------
``ablation_wave``'s ``run`` predates the f_xy column: it serializes
``WaveOutcome`` without :attr:`~lpopt.search.verify.WaveOutcome.fxy`, and its kit
builder rebuilds outcomes from that jsonl, so an F_xy label harvested by the
verifier would be dropped on the floor.  Two SMALL, DOCUMENTED rebindings close
that, both inside a ``try/finally`` scope and both on ``lpopt.search.verify``
rather than on ``ablation_wave``:

* :class:`WaveVerifier` -> a subclass whose ``evaluate_wave`` also appends the
  parsed :class:`~lpopt.data.fxy.FxyResult` of every outcome to a per-cell
  ``fxy_sidecar.jsonl``;
* :func:`outcome_to_record` -> a wrapper that fills ``f_xy`` / ``f_xya`` on the
  rebuilt record from that sidecar.

Nothing else about the run path changes, and the sidecar is the ONLY new
artefact, so a wave run with an unpatched ``ablation_wave`` is still readable.

Library routing (READ THIS BEFORE ADDING A CELL)
------------------------------------------------
``ProduceDriver`` routes assets PER LIBRARY (``produce.py`` s_run_library_id /
``resolver.build_case_resolver``): a paramA cell resolves against the design
package with the registry alias bridge and the package's own ``%GEN_DIM`` dims,
a ga80 cell against ``FEASIBLE_PACKAGE`` with ``LIBRARY_DIMS``.  The ablation
runner builds ONE resolver from ONE ``--package``, so it shares that limitation:
**a single ``run`` invocation may only carry cells of one library.**  ``run``
refuses a mixed selection rather than silently staging one library's boards
against the other's deck.

Subcommands
-----------
``plan``     select parents per cell, enumerate + classify each parent's complete
             verified single-move neighbourhood, dedup against the store, draw
             the balanced, direction-PAIRED sample, write ONE manifest with a
             ``cell`` column.  LOCAL, read-only, no MASTER.
``score``    blind s1i predictions (7-column surrogate + node_peak + F_xy via
             ``acquisition.predict_fxy``: the dedicated head when the checkpoint
             carries one, the F_r proxy otherwise, and the CSV says which) for
             every planned child AND its parent, written BEFORE any label exists.
``run``      evaluate one kit's cells as MASTER equilibrium chains.  Remote-safe,
             self-contained, resume-safe per cell, MAS_OUT retained.
``kit``      rebuild the per-cell merge-store kits from the jsonl + sidecar.
``analyze``  per-cell and pooled conditional-effect tables (d_cyclen / d_F_r /
             d_F_xy / d_node_peak by move_class x direction), the neutral
             control's per-cell baseline offset and the same tables with it
             removed (``effects_*_adj``), parent-blocked sign tests on the
             symmetric pairs.
``corpus``   append each cell's lineage edges to ``data/policy/steps.parquet``
             with ``lineage_source='intervention_<cell>'``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import ablation_wave as W                                      # noqa: E402

# --------------------------------------------------------------------------- #
# registered constants
# --------------------------------------------------------------------------- #
#: Pre-registered seed.  Only the RANDOM-feasible parent draw and the per-parent
#: ``batch_flip`` direction toss consume it; every other selection in this module
#: is rank-deterministic (``ablation_wave._pick``).
SEED = 20260829

GENERATOR = "intervention_1move"
RESULTS_NAME = W.RESULTS_NAME                 # "ablation_results.jsonl"
FXY_SIDECAR_NAME = "fxy_sidecar.jsonl"
SUBPLAN_NAME = "subplan.json"

N_PARENTS = 20
MOVES_PER_PARENT = 8
#: Review s7.6 wants 25% random / space-filling.  Five of twenty parents per cell
#: are drawn at random from the feasible pool instead of off the F_xy ranking, so
#: the atlas is not measured only where the optimizer already lives.
RANDOM_PARENT_FRAC = 0.25
#: Minimum pairwise 69-slot Hamming distance between accepted parents (the
#: ablation wave's diversity gate, unchanged).
HAMMING_MIN = 12
#: Hard per-cell cap on PAID chains, incl. reruns.
BUDGET_CAP_PER_CELL = 200

#: Classes drawn as SYMMETRIC PAIRS: k outward children and, for each, the
#: dose-matched inward sibling of the SAME parent (review s7.2 asks that the
#: direction be randomized per parent and that the symmetric action be
#: computed alongside it).  Both members are paid, so the pair difference is a
#: within-parent contrast with no imputation.
PAIRED_QUOTA: dict[str, int] = {"fresh_relocate": 2, "batch_swap": 1}
#: Structurally direction-free: ``rewire_swap`` touches neither the fresh set nor
#: the batch labels, so ``d_fresh_enr_r_center`` is IDENTICALLY zero (ablation
#: pre-registration s3).  It is the neutral control arm.
NEUTRAL_QUOTA: dict[str, int] = {"rewire_swap": 1}
#: Direction RANDOMIZED per parent rather than paired: ``batch_flip`` changes the
#: fresh batch multiset and therefore total fresh enrichment, so an
#: outward/inward pair of flips is not reactivity-matched and must not be read as
#: a leakage contrast (ablation pre-registration s3b - batch_flip is reported but
#: is NOT an instrument).  Randomizing balances it ACROSS parents instead.
RANDOMIZED_QUOTA: dict[str, int] = {"batch_flip": 1}

#: 2*2 (fresh_relocate) + 1*2 (batch_swap) + 1 (rewire) + 1 (flip) = 8.
QUOTA_TOTAL = (2 * sum(PAIRED_QUOTA.values())
               + sum(NEUTRAL_QUOTA.values()) + sum(RANDOMIZED_QUOTA.values()))

#: Burn-state classes the sample is spread over inside each stratum.  Derived
#: from the PARENT genome's own ``_depths()`` - the same source
#: ``mine_policy_corpus.residence_profile`` uses - so "once-burnt swap" here and
#: ``once_burnt_periph_share`` there mean the same thing.
BURN_STATES: tuple[str, ...] = ("fresh", "once", "twice_plus", "center")


@dataclass(frozen=True)
class Cell:
    """One ``(case_pair, feed, library)`` context of the atlas."""

    name: str
    pair: str
    feed: int
    library: str
    #: Asset-routing family: ``"paramA"`` (design package + registry aliases) or
    #: ``"ga80"`` (FEASIBLE_PACKAGE + LIBRARY_DIMS).  A ``run`` invocation may
    #: only carry one.
    kit: str
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.pair}/f{self.feed}/{self.library}"

    @property
    def campaign(self) -> str:
        return f"intervention_{self.name}"

    @property
    def lineage_source(self) -> str:
        return f"intervention_{self.name}"

    @property
    def run_subdir(self) -> str:
        return f"intervention_{self.name}"

    def rng(self, seed: int = SEED) -> random.Random:
        """Per-cell RNG whose stream does not move when another cell is added."""
        h = hashlib.sha256(f"{seed}|{self.key}".encode()).hexdigest()[:16]
        return random.Random(int(h, 16))


#: The registered round-1 cells: the five F_xy frontier cells of the 6,236-label
#: F_xy state (``data/store/records.parquet``).  Two libraries -> two kits.
CELLS_R1: tuple[Cell, ...] = (
    Cell("T6T4_f121", "T6_T4", 121, "paramA", "paramA",
         "746 measured f_xy - the densest cell and the ablation wave's cell"),
    Cell("HGD569_f125", "P6253Z1G06N24_P6253Z2G10N24", 125, "paramA", "paramA",
         "104 measured f_xy - the high-Gd 5.69 w/o pair (HGD569 deck family)"),
    Cell("E1E2_f121", "E1_E2", 121, "ga80", "ga80",
         "654 measured f_xy - the ga80 anchor cell"),
    Cell("E1E2_f109", "E1_E2", 109, "ga80", "ga80",
         "156 measured f_xy - same pair, low feed (residence lever)"),
    Cell("N1N2_f113", "N1_N2", 113, "ga80", "ga80",
         "172 measured f_xy - the pin-burnup campaign cell"),
)

CELLS_BY_NAME = {c.name: c for c in CELLS_R1}


def resolve_cells(names: Sequence[str] | None) -> list[Cell]:
    """``--cells a,b`` -> Cell objects; empty/None -> the whole registered set."""
    if not names:
        return list(CELLS_R1)
    out = []
    for n in names:
        if n not in CELLS_BY_NAME:
            raise SystemExit(f"unknown cell {n!r}; known: "
                             f"{', '.join(sorted(CELLS_BY_NAME))}")
        out.append(CELLS_BY_NAME[n])
    return out


# --------------------------------------------------------------------------- #
# burn-state class of a move
# --------------------------------------------------------------------------- #
def changed_units(parent_packed: str, child_packed: str) -> set[int]:
    """Orbit units whose slots differ between two packed boards.

    Thin re-export of :func:`mine_policy_corpus.changed_units`; the unit-level
    diff lives beside ``UNIT_SLOTS``, which defines it.
    """
    import mine_policy_corpus as M

    return M.changed_units(parent_packed, child_packed)


def burn_state_class(parent_genome, parent_packed: str, child_packed: str) -> str:
    """Deepest residence layer this move touches, in the PARENT board.

    ``fresh`` (depth 0 only), ``once`` (a depth-1 unit moved), ``twice_plus`` (a
    depth>=2 unit moved), ``center`` (only the centre cell changed - the centre
    is not an orbit unit).  This is the review's "once/twice-burnt swap" axis
    (s7.2) and the "burn state" column of the Campaign A effect table (s7.4).

    Thin re-export of :func:`mine_policy_corpus.move_burn_state`, which is also
    what ``build_steps`` now writes into the corpus' ``burn_state`` column - one
    implementation, so the wave's stratum and the corpus column are the same
    quantity by construction rather than by agreement.
    """
    import mine_policy_corpus as M

    return M.move_burn_state(parent_genome, parent_packed, child_packed)


# --------------------------------------------------------------------------- #
# core (board) identity
# --------------------------------------------------------------------------- #
def core_digest(packed: str) -> str:
    """Canonical 16-hex digest of the CORE a packed board realizes.

    ``Pattern.digest`` is an encoding digest: it separates two strings, not two
    reactors.  A board and its diagonal transpose are the SAME physical core in
    two encodings (``lpopt.data.geometry.transpose``; measured in
    ``transpose_noise_measured_20260725.md``), so the canonical representative is
    the lexicographically smaller of the two digests.  Two children with equal
    ``core_digest`` are one experiment bought twice.

    Memoized on the packed string: ``plan`` calls this once per pairing
    candidate, and a parent's neighbourhood revisits the same boards.
    """
    hit = _CORE_DIGEST_CACHE.get(packed)
    if hit is None:
        from lpopt.data.geometry import transpose
        from lpopt.data.schema import unpack_pattern

        pat = unpack_pattern(packed)
        try:
            hit = min(pat.digest, transpose(pat).digest)
        except Exception:                                       # noqa: BLE001
            # ``transpose`` re-asserts the quarter-core conventions.  A board it
            # refuses is still a board: fall back to its own digest, which is
            # strictly weaker (it separates two encodings of one core) but never
            # merges two DIFFERENT cores, so the guard stays sound either way.
            hit = pat.digest
        _CORE_DIGEST_CACHE[packed] = hit
    return hit


_CORE_DIGEST_CACHE: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# parent selection
# --------------------------------------------------------------------------- #
def joint_clean(frame, require_feasible: bool = True):
    """Rows usable as a parent: converged, FULLY LABELLED, and (tier 1) feasible.

    "Joint-clean" is stricter than ``mine_policy_corpus.feasibility`` alone on the
    label side: a parent whose ``node_peak`` or ``cbc_max`` is missing cannot
    supply a parent-blocked DELTA on that axis, and a Campaign A row exists to be
    differenced.  ``f_xy`` is deliberately NOT required - it is the ranking key
    when present, but a cell with thin F_xy coverage must still seat its parents.

    ``require_feasible`` is a TIER, not a gate (see :func:`select_parents`).  The
    F_xy frontier contains cells where NO board is program-feasible yet - the
    high-Gd 5.69 w/o pair's best F_r is 1.6036 against the 1.55 limit - and
    refusing to measure the move atlas exactly where the search is hardest would
    reproduce the review s6.9 complaint that the data sits only where the
    optimizer already lives.
    """
    import mine_policy_corpus as M
    import pandas as pd

    have = np.ones(len(frame), dtype=bool)
    for col in ("f_r", "node_peak", "cyclen", "cbc_max"):
        have &= frame[col].notna().to_numpy()
    conv = frame["converged"].fillna(False).astype(bool).to_numpy()
    ok = have & conv
    if require_feasible:
        ok = ok & M.feasibility(frame).fillna(False).astype(bool).to_numpy()
    return pd.Series(ok, index=frame.index)


def select_parents(store, enr, cell: Cell, *, n_parents: int = N_PARENTS,
                   random_frac: float = RANDOM_PARENT_FRAC,
                   hamming_min: int = HAMMING_MIN, seed: int = SEED, log=print):
    """Parents of ``cell``: the measured-F_xy ranking plus a random-feasible 25%.

    Two queues, in registered order:

    * ``fxy_rank`` - joint-clean rows that carry a MEASURED ``f_xy``, ascending
      (lower planar peaking is better).  This is the frontier the campaign is
      about, and it is a MEASURED ranking, never a predicted one.
    * ``fr_rank``  - joint-clean rows with no ``f_xy`` yet, ascending ``f_r``.
      The registered fallback for a cell whose F_xy coverage is thin.

    plus ``random`` - a uniform draw (seeded) over the whole joint-clean pool,
    ``round(random_frac * n_parents)`` of them, so review s7.6's 25%
    random/space-filling share is met at the PARENT level and the atlas is not
    measured only where the optimizer already lives.

    Deterministic given ``seed``.  Every acceptance passes the same pairwise
    Hamming gate the ablation wave used.
    """
    import mine_policy_corpus as M
    import pandas as pd
    from lpopt.data.schema import unpack_pattern

    rows = store[(store["case_pair"] == cell.pair)
                 & (store["feed"] == cell.feed)
                 & (store["library_id"] == cell.library)].copy()
    if rows.empty:
        raise SystemExit(f"no store rows for cell {cell.key}")
    labelled = rows[joint_clean(rows, require_feasible=False).to_numpy()].copy()
    feasible_ids = set(rows[joint_clean(rows).to_numpy()]["record_id"])
    n_fxy = int(labelled["f_xy"].notna().sum()) if "f_xy" in labelled else 0
    log(f"[plan] cell {cell.key}: {len(rows)} rows, {len(labelled)} converged+"
        f"labelled, {len(feasible_ids)} also program-feasible, "
        f"{n_fxy} with measured f_xy")
    if labelled.empty:
        raise SystemExit(f"cell {cell.key} has no converged, fully labelled rows")

    rng = random.Random(int(hashlib.sha256(
        f"{seed}|{cell.key}|parents".encode()).hexdigest()[:16], 16))
    indexed = labelled.set_index("record_id", drop=False)
    patterns: dict[str, Any] = {}

    def pat(rid):
        if rid not in patterns:
            patterns[rid] = unpack_pattern(indexed.loc[rid, "pattern"])
        return patterns[rid]

    # Two TIERS in registered order: program-feasible boards first, then the rest
    # of the converged+labelled pool.  Within a tier: the MEASURED-F_xy ranking,
    # then the F_r ranking of rows that have no f_xy yet, then a seeded uniform
    # draw for the review s7.6 random share.
    def queues(pool):
        has_fxy = (pool["f_xy"].notna() if "f_xy" in pool.columns
                   else pd.Series(False, index=pool.index))
        shuffled = sorted(pool["record_id"])
        rng.shuffle(shuffled)
        return (list(pool[has_fxy].sort_values(["f_xy", "record_id"],
                                               kind="mergesort")["record_id"]),
                list(pool[~has_fxy].sort_values(["f_r", "record_id"],
                                                kind="mergesort")["record_id"]),
                shuffled)

    is_feas = labelled["record_id"].isin(feasible_ids)
    tiers = [("", labelled[is_feas]), ("_infeasible", labelled[~is_feas])]

    n_random = int(round(random_frac * n_parents))
    n_ranked = n_parents - n_random
    accepted: list[tuple[str, str]] = []
    seen: set[str] = set()

    def take(queue, want, family):
        got = 0
        for rid in queue:
            if got >= want or len(accepted) >= n_parents:
                return got
            if rid in seen:
                continue
            if all(pat(rid).hamming(pat(a)) >= hamming_min for a, _ in accepted):
                accepted.append((rid, family))
                seen.add(rid)
                got += 1
        return got

    n_rank_left, n_rand_left = n_ranked, n_random
    for suffix, pool in tiers:
        if pool.empty:
            continue
        fxy_rank, fr_rank, shuffled = queues(pool)
        n_rank_left -= take(fxy_rank, n_rank_left, f"fxy_rank{suffix}")
        n_rank_left -= take(fr_rank, n_rank_left, f"fr_rank{suffix}")
        n_rand_left -= take(shuffled, n_rand_left, f"random{suffix}")
    # Top-up (relaxed): the tier/family SHAPE is a preference, n_parents is the
    # contract - the same rule the ablation wave registered.
    if len(accepted) < n_parents:
        for _suffix, pool in tiers:
            if pool.empty or len(accepted) >= n_parents:
                continue
            for rid in sorted(pool["record_id"]):
                if len(accepted) >= n_parents:
                    break
                if rid in seen:
                    continue
                if all(pat(rid).hamming(pat(a)) >= hamming_min
                       for a, _ in accepted):
                    accepted.append((rid, "topup"))
                    seen.add(rid)

    out = []
    for rid, family in accepted:
        r = indexed.loc[rid]
        phys = M.board_physics(r["pattern"], M.genome_of(r["pattern"]), enr)

        def _f(col):
            v = r[col] if col in r.index else None
            return None if v is None or pd.isna(v) else float(v)

        out.append({
            "record_id": rid, "family": family, "pattern": r["pattern"],
            "campaign": r["campaign"], "cell": cell.name,
            "program_feasible": bool(rid in feasible_ids),
            "f_r": _f("f_r"), "node_peak": _f("node_peak"),
            "cyclen": _f("cyclen"), "cbc_max": _f("cbc_max"), "f_q": _f("f_q"),
            "f_xy": _f("f_xy"), "f_xya": _f("f_xya"), "e_core": _f("e_core"),
            "restart_provenance": str(r.get("restart_provenance") or ""),
            **{k: phys[k] for k in M.PHYSICS},
        })
    ham = [[int(pat(a["record_id"]).hamming(pat(b["record_id"])))
            for b in out] for a in out]
    return out, ham


# --------------------------------------------------------------------------- #
# balanced, direction-paired sampling
# --------------------------------------------------------------------------- #
def quantile_ranks(frame, k: int, key: str) -> list[int]:
    """``k`` rows at the MIDPOINTS of ``k`` equal-probability bins of ``key``.

    Deliberately NOT ``ablation_wave._pick``.  That rule spaces ranks over
    ``[0, n-1]`` INCLUSIVE, so it always takes the extremes: at ``k = 1`` it
    takes the SMALLEST dose in the stratum and at ``k = 2`` the smallest and the
    largest.  With the ablation wave's quotas (3 per stratum) that was a
    reasonable spread; with this wave's per-parent quota of 1-2 per stratum it
    would systematically buy the tiniest available intervention - measured on
    ``E1_E2``/f109 as a paid median dose of 0.0068 against a neighbourhood median
    of 0.0757, i.e. an order of magnitude below the moves the cell actually
    offers, which is exactly the regime where a real effect hides under the noise
    floor.  Bin midpoints (``(i + 0.5) / k``) give the median at ``k = 1`` and the
    quartiles at ``k = 2``: the same "spans its own dose range" intent, without
    the extreme bias.  Deterministic; ties broken by ``move_tag``.
    """
    if len(frame) == 0 or k <= 0:
        return []
    order = frame.sort_values([key, "move_tag"], kind="mergesort").index.tolist()
    if len(order) <= k:
        return order
    n = len(order)
    pos = [min(n - 1, int((i + 0.5) / k * n)) for i in range(k)]
    return [order[p] for p in dict.fromkeys(pos)]


def pick_burnstate_balanced(frame, k: int, key: str) -> list[int]:
    """``k`` rows of ``frame``, round-robin over burn-state, dose-spread within.

    The ``k`` slots are ALLOCATED round-robin across the burn-state groups first
    (so the stratum spans its own residence range), and each group then draws its
    allocation at :func:`quantile_ranks` positions of its OWN dose order (so it
    also spans its own dose range).  Allocating first and drawing second is what
    keeps the dose spread: drawing ``k`` per group and interleaving would take
    every group's rank-0 row, which is every group's smallest dose.
    """
    if len(frame) == 0 or k <= 0:
        return []
    groups = {s: frame[frame["burn_state"] == s] for s in BURN_STATES}
    states = [s for s in BURN_STATES if len(groups[s])]
    counts = {s: 0 for s in states}
    remaining = k
    while remaining > 0:
        progressed = False
        for s in states:
            if remaining <= 0:
                break
            if counts[s] < len(groups[s]):
                counts[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    out: list[int] = []
    for s in states:
        out.extend(quantile_ranks(groups[s], counts[s], key))
    return out[:k]


def _sibling(pool, used: set[int], target_dose: float, target_state: str,
             forbid_core: str | None = None):
    """The dose-matched, burn-state-matched opposite-direction sibling.

    Deterministic tie-break chain: same burn state first (a fresh<->once move
    and a once<->twice move are different physics, so a pair that mixes them is
    not the symmetric action the review asks for), then the closest |dose|, then
    ``move_tag`` alphabetically.

    ``forbid_core`` is the outward child's :func:`core_digest`.  A candidate that
    realizes the SAME core is rejected however well it matches on dose: the pair
    would buy two MASTER chains for one experiment and contribute an identically
    zero out-minus-in difference to the campaign's primary statistic.  Returns
    ``(index_or_None, n_core_degenerate_rejected)``.
    """
    best = None
    best_key = None
    n_degenerate = 0
    for idx, row in pool.iterrows():
        if idx in used:
            continue
        if forbid_core is not None and core_digest(row["pattern"]) == forbid_core:
            n_degenerate += 1
            continue
        key = (0 if row["burn_state"] == target_state else 1,
               abs(float(row["dose"]) - float(target_dose)),
               str(row["move_tag"]))
        if best_key is None or key < best_key:
            best, best_key = idx, key
    return best, n_degenerate


def draw_parent_sample(frame, parent_record_id: str, rng: random.Random):
    """One parent's ``MOVES_PER_PARENT`` children: paired, balanced, deterministic.

    Returns ``(picked_index_list, meta_by_index, shortfalls)``.  ``meta_by_index``
    carries ``pair_id`` / ``pair_role`` / ``stratum``.

    NO CROSS-STRATUM BACKFILL.  If a parent's neighbourhood cannot supply a
    symmetric sibling, the pair is simply not drawn and a shortfall is recorded:
    an unpaired outward child bought with a pair's budget would silently turn a
    within-parent contrast back into the between-parent comparison this campaign
    exists to replace.

    ONE registered exception - **direction degeneracy**.  On a cell whose fresh
    alphabet is enrichment-degenerate (both ga80 pairs measured here are:
    ``E1``/``E2`` are both 5.000 w/o, ``N1``/``N2`` both 5.400), a ``batch_swap``
    or ``batch_flip`` moves no enrichment mass and therefore leaves
    ``fresh_enr_r_center`` EXACTLY unchanged: the class has no radial direction
    at all, in the same structural sense ``rewire_swap`` has none.  Such a class
    is then drawn as a NEUTRAL arm of the same size (``2k`` for a paired class),
    tagged ``neutral_degenerate``, rather than being reported as ``2k`` missing
    chains - a Gd-pattern move is still real physics, it is only invisible to an
    ENRICHMENT-weighted radial descriptor, and the wave should measure it.

    A SECOND degeneracy, added 2026-08-30 from
    ``intervention_wave_r1_results_20260830.md`` s5.1/s11-1: the enrichment
    census above is a property of the fresh ALPHABET, and it cannot see a pair
    whose outward and inward children happen to realize the SAME CORE.  Round 1
    bought 20 such ``batch_swap`` pairs on ``HGD569_f125`` - 40 chains, 5.0% of
    the paid budget, contributing an identically zero out-minus-in difference to
    the campaign's primary statistic.  Every pair is therefore checked with
    :func:`core_digest` BEFORE it is drawn; a core-degenerate sibling is skipped,
    and a candidate with no non-degenerate sibling yields to the next one.
    """
    picked: list[int] = []
    meta: dict[int, dict[str, Any]] = {}
    shortfalls: list[dict[str, Any]] = []
    used: set[int] = set()
    # The FULL parent id, not a 12-char prefix: ``pair_id`` is the join key of
    # the primary statistic, and a prefix collision would silently merge two
    # parents' pairs into one contrast.
    tag = parent_record_id

    def take_neutral(cls: str, k: int, role: str, key: str = "dose"):
        pool = frame[(frame["move_class"] == cls)
                     & (frame["fresh_radial_dir"] == "neutral")]
        if key not in pool.columns or not pool[key].notna().any():
            key = "n_slots_changed"
        idx = pick_burnstate_balanced(pool, k, key)
        for j in idx:
            picked.append(j)
            used.add(j)
            meta[j] = {"pair_id": None, "pair_role": role,
                       "stratum": f"{cls}|neutral"}
        return idx, pool

    # -- symmetric pairs ---------------------------------------------------- #
    for cls in sorted(PAIRED_QUOTA):
        k = PAIRED_QUOTA[cls]
        out_pool = frame[(frame["move_class"] == cls)
                         & (frame["fresh_radial_dir"] == "outward")]
        in_pool = frame[(frame["move_class"] == cls)
                        & (frame["fresh_radial_dir"] == "inward")]
        if len(out_pool) == 0 and len(in_pool) == 0:
            idx, pool = take_neutral(cls, 2 * k, "neutral_degenerate")
            if len(idx) < 2 * k:
                shortfalls.append({
                    "parent_record_id": parent_record_id, "move_class": cls,
                    "kind": "neutral_degenerate", "want": 2 * k,
                    "got": len(idx), "available": int(len(pool))})
            continue
        # The REGISTERED draw is unchanged; the ``2k`` draw only supplies a
        # RESERVE, consulted solely when a chosen candidate turns out to be
        # core-degenerate against every inward sibling.  With no degeneracy the
        # first ``k`` of the queue are the registered ``k`` and the sample is
        # bit-identical to the pre-2026-08-30 rule.
        chosen_out = pick_burnstate_balanced(out_pool, k, "dose")
        head = set(chosen_out)
        chosen_out = chosen_out + [j for j in pick_burnstate_balanced(
            out_pool, 2 * k, "dose") if j not in head]
        n_pairs = 0
        n_degenerate = 0
        for oi in chosen_out:
            if n_pairs >= k:
                break
            si, n_deg = _sibling(in_pool, used, frame.at[oi, "dose"],
                                 frame.at[oi, "burn_state"],
                                 forbid_core=core_digest(frame.at[oi, "pattern"]))
            if si is None:
                n_degenerate += n_deg
                if n_deg:
                    continue          # core-degenerate only - try the next one
                break                 # genuinely no sibling left
            n_degenerate += n_deg
            pair_id = f"{tag}:{cls}:{n_pairs}"
            for idx, role in ((oi, "outward"), (si, "inward")):
                picked.append(idx)
                used.add(idx)
                meta[idx] = {"pair_id": pair_id, "pair_role": role,
                             "stratum": f"{cls}|{role}"}
            n_pairs += 1
        if n_pairs < k:
            shortfalls.append({
                "parent_record_id": parent_record_id, "move_class": cls,
                "kind": "core_degenerate" if n_degenerate else "pair",
                "want": k, "got": n_pairs,
                "n_outward": int(len(out_pool)), "n_inward": int(len(in_pool)),
                "n_core_degenerate": int(n_degenerate)})

    # -- neutral control ---------------------------------------------------- #
    for cls in sorted(NEUTRAL_QUOTA):
        k = NEUTRAL_QUOTA[cls]
        # ``swap_span`` is the dose surrogate for a direction-free move: how far
        # apart the two rewired units sit, which is what a leakage-neutral
        # exchange actually varies.
        idx, pool = take_neutral(cls, k, "neutral", key="swap_span")
        if len(idx) < k:
            shortfalls.append({"parent_record_id": parent_record_id,
                               "move_class": cls, "kind": "neutral",
                               "want": k, "got": len(idx),
                               "available": int(len(pool))})

    # -- direction-randomized arm ------------------------------------------- #
    for cls in sorted(RANDOMIZED_QUOTA):
        k = RANDOMIZED_QUOTA[cls]
        sub = frame[frame["move_class"] == cls]
        avail = [d for d in ("outward", "inward")
                 if (sub["fresh_radial_dir"] == d).any()]
        if not avail:
            idx, pool = take_neutral(cls, k, "neutral_degenerate")
            if len(idx) < k:
                shortfalls.append({
                    "parent_record_id": parent_record_id, "move_class": cls,
                    "kind": "neutral_degenerate", "want": k, "got": len(idx),
                    "available": int(len(pool))})
            continue
        direction = avail[0] if len(avail) == 1 else rng.choice(sorted(avail))
        pool = sub[sub["fresh_radial_dir"] == direction]
        idx = pick_burnstate_balanced(pool, k, "dose")
        for j in idx:
            picked.append(j)
            used.add(j)
            meta[j] = {"pair_id": None, "pair_role": f"random_{direction}",
                       "stratum": f"{cls}|{direction}"}
        if len(idx) < k:
            shortfalls.append({"parent_record_id": parent_record_id,
                               "move_class": cls, "kind": "randomized",
                               "want": k, "got": len(idx),
                               "available": int(len(pool))})
    return picked, meta, shortfalls


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan_cell(cell: Cell, store, enr_all, fuel, known: set[str], *,
              n_parents: int, seed: int, log=print):
    """One cell's block of the manifest."""
    import mine_policy_corpus as M
    import pandas as pd
    from lpopt.data.schema import compute_record_id, unpack_pattern
    from lpopt.search.construct import predicted_e_core
    from lpopt.search.verify import PRODUCE_DECK_KNOBS

    enr = enr_all.get(cell.library)
    if enr is None:
        raise SystemExit(f"no enrichment rows for library {cell.library!r} "
                         f"(cell {cell.key})")
    parents, ham = select_parents(store, enr, cell, n_parents=n_parents,
                                  seed=seed, log=log)
    rng = cell.rng(seed)

    blocks, census, shortfalls = [], [], []
    for p in parents:
        pg = M.genome_of(p["pattern"])
        rows = W.annotate(p["pattern"], pg, W.enumerate_single_moves(pg), enr)
        frame = pd.DataFrame(rows)
        if frame.empty:
            shortfalls.append({"parent_record_id": p["record_id"],
                               "move_class": "*", "kind": "empty_neighbourhood",
                               "want": MOVES_PER_PARENT, "got": 0})
            continue
        frame = frame.drop_duplicates(subset=["pattern"]).reset_index(drop=True)
        frame["burn_state"] = [
            burn_state_class(pg, p["pattern"], x) for x in frame["pattern"]]
        frame["record_id"] = [
            compute_record_id(x, cell.library, cell.pair, PRODUCE_DECK_KNOBS)
            for x in frame["pattern"]]
        frame["already_labeled"] = frame["record_id"].isin(known)
        frame["parent_record_id"] = p["record_id"]

        for (cls, direction), n in (
                frame.groupby(["move_class", "fresh_radial_dir"]).size().items()):
            blk = frame[(frame["move_class"] == cls)
                        & (frame["fresh_radial_dir"] == direction)]
            census.append({
                "cell": cell.name, "parent_record_id": p["record_id"],
                "move_class": cls, "fresh_radial_dir": direction,
                "n_available": int(n),
                "n_already_labeled": int(blk["already_labeled"].sum()),
                **{f"n_{s}": int((blk["burn_state"] == s).sum())
                   for s in BURN_STATES}})

        # Dedup FIRST: an already-labelled child is a FREE label and must not
        # consume a MASTER slot.  It is still carried (source ``free``) so the
        # stratum's n is paid + free.
        paid_pool = frame[~frame["already_labeled"]]
        idx, meta, short = draw_parent_sample(paid_pool, p["record_id"], rng)
        shortfalls.extend(short)

        picked = frame.loc[idx].copy()
        picked["source"] = "paid"
        for col in ("pair_id", "pair_role", "stratum"):
            picked[col] = [meta[i][col] for i in idx]

        free = frame[frame["already_labeled"]].copy()
        free = free[free["move_class"].isin(
            set(PAIRED_QUOTA) | set(NEUTRAL_QUOTA) | set(RANDOMIZED_QUOTA))]
        free["source"] = "free"
        free["pair_id"] = None
        free["pair_role"] = "free"
        free["stratum"] = free["move_class"] + "|" + free["fresh_radial_dir"]

        block = pd.concat([picked, free], ignore_index=True)
        block["core_digest"] = [core_digest(x) for x in block["pattern"]]
        # Defence in depth: the draw already refuses a core-degenerate sibling,
        # so a pair that survives to here with one core is a REGRESSION, and it
        # must fail on this box rather than 8 hours into a MASTER run.
        paired_rows = block[block["pair_id"].notna()]
        for pid, pair in paired_rows.groupby("pair_id"):
            if pair["core_digest"].nunique() < len(pair):
                raise SystemExit(
                    f"cell {cell.name}: pair {pid} is core-degenerate "
                    f"(both children realize core {pair['core_digest'].iloc[0]})")
        block["cell"] = cell.name
        block["case_pair"] = cell.pair
        block["feed"] = cell.feed
        block["library_id"] = cell.library
        block["campaign"] = cell.campaign
        for col in ("f_r", "node_peak", "cyclen", "cbc_max", "f_q", "f_xy"):
            block[f"parent_{col}"] = p[col]
        block["e_core"] = ([predicted_e_core(unpack_pattern(x), fuel, cell.library)
                            for x in block["pattern"]]
                           if fuel is not None else None)
        blocks.append(block)
        log(f"[plan]   {p['record_id'][:12]} ({p['family']:<8}) "
            f"F_xy {('%.4f' % p['f_xy']) if p['f_xy'] is not None else '  n/a '} "
            f"F_r {p['f_r']:.4f}  nbhd {len(frame):5d}  "
            f"free {int(frame['already_labeled'].sum()):3d}  "
            f"paid {len(picked):2d}/{MOVES_PER_PARENT}")

    cand = (pd.concat(blocks, ignore_index=True) if blocks
            else pd.DataFrame(columns=["source"]))
    n_paid = int((cand["source"] == "paid").sum()) if len(cand) else 0
    if n_paid > BUDGET_CAP_PER_CELL:
        raise SystemExit(f"cell {cell.name}: paid {n_paid} exceeds the registered "
                         f"per-cell cap {BUDGET_CAP_PER_CELL}")
    # Direction regime per class, read off the ENUMERATED census (not asserted):
    # "signed" when the class reaches both radial directions on this cell,
    # "degenerate" when it reaches neither - the iso-enrichment case, where the
    # class is drawn as a neutral arm instead.
    regime: dict[str, str] = {}
    for cls in sorted(set(PAIRED_QUOTA) | set(NEUTRAL_QUOTA)
                      | set(RANDOMIZED_QUOTA)):
        dirs = {r["fresh_radial_dir"] for r in census
                if r["move_class"] == cls and r["n_available"] > 0}
        signed = dirs & {"outward", "inward"}
        regime[cls] = ("signed" if len(signed) == 2 else
                       "one_sided" if signed else "degenerate")

    block_meta = {
        "name": cell.name, "cell": cell.key, "pair": cell.pair,
        "feed": cell.feed, "library_id": cell.library, "kit": cell.kit,
        "note": cell.note, "campaign": cell.campaign,
        "lineage_source": cell.lineage_source, "run_subdir": cell.run_subdir,
        "direction_regime": regime,
        "n_feasible_parents": int(sum(1 for p in parents
                                      if p["program_feasible"])),
        # The restart every parent of this cell was LABELLED against.  ``run``
        # refuses to spend chains on a different one: a parent/child delta
        # measured across a restart change is a move plus a burnt-fuel history
        # change, and the wave cannot tell them apart.
        "parent_restart_provenance": sorted(
            {p["restart_provenance"] for p in parents if p["restart_provenance"]}),
        # Slot-geometry direction degeneracy, the census the enrichment regime
        # above cannot see (s11-1): how many inward candidates were rejected for
        # realizing the outward child's core, and how many pairs were lost to it.
        "n_core_degenerate_rejected": int(sum(
            int(s.get("n_core_degenerate", 0)) for s in shortfalls)),
        "n_core_degenerate_shortfalls": int(sum(
            1 for s in shortfalls if s["kind"] == "core_degenerate")),
        "n_parents": len(parents), "n_paid": n_paid,
        "n_free": int((cand["source"] == "free").sum()) if len(cand) else 0,
        "parents": parents, "parent_hamming": ham,
        "shortfalls": shortfalls, "neighbourhood_census": census,
    }
    return block_meta, cand


def cmd_plan(args) -> int:
    import pandas as pd
    import mine_policy_corpus as M
    from lpopt.search.verify import PRODUCE_DECK_KNOBS

    cells = resolve_cells(args.cells)
    store = pd.read_parquet(args.store)
    enr_all = M.load_enrichment(Path(args.fuel_types))
    known = set(store["record_id"].astype(str))
    try:
        from lpopt.data.fuel_types import FuelLibrary
        fuel = FuelLibrary.from_parquet(Path(args.fuel_types))
    except Exception as exc:                                    # noqa: BLE001
        print(f"[plan] WARNING: fuel table unreadable ({exc}); e_core left null")
        fuel = None

    metas, frames = [], []
    for cell in cells:
        meta, cand = plan_cell(cell, store, enr_all, fuel, known,
                               n_parents=args.parents, seed=args.seed)
        metas.append(meta)
        if len(cand):
            frames.append(cand)
    cand = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    kits: dict[str, list[str]] = {}
    for c in cells:
        kits.setdefault(c.kit, []).append(c.name)

    manifest = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wave": args.wave, "generator": GENERATOR, "seed": args.seed,
        "deck_knobs": PRODUCE_DECK_KNOBS,
        "n_parents_per_cell": args.parents,
        "moves_per_parent": MOVES_PER_PARENT,
        "quota": {"paired": PAIRED_QUOTA, "neutral": NEUTRAL_QUOTA,
                  "randomized": RANDOMIZED_QUOTA, "total": QUOTA_TOTAL},
        "random_parent_frac": RANDOM_PARENT_FRAC,
        "hamming_min": HAMMING_MIN,
        "budget_cap_per_cell": BUDGET_CAP_PER_CELL,
        "burn_states": list(BURN_STATES),
        "kits": kits,
        "cells": metas,
        "n_paid": int((cand["source"] == "paid").sum()) if len(cand) else 0,
        "n_free": int((cand["source"] == "free").sum()) if len(cand) else 0,
        "candidates": json.loads(cand.to_json(orient="records")) if len(cand) else [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"\n[plan] {len(cells)} cell(s)  paid {manifest['n_paid']}  "
          f"free {manifest['n_free']}  -> {out}")
    for m in metas:
        print(f"[plan]   {m['name']:<12} {m['cell']:<48} kit={m['kit']:<7} "
              f"parents {m['n_parents']:2d} ({m['n_feasible_parents']} feasible)  "
              f"paid {m['n_paid']:3d}  free {m['n_free']:3d}  "
              f"shortfalls {len(m['shortfalls'])}  regime {m['direction_regime']}")
        if m["n_core_degenerate_rejected"]:
            print(f"[plan]   {'':<12} core-degenerate siblings rejected "
                  f"{m['n_core_degenerate_rejected']} "
                  f"(pairs lost {m['n_core_degenerate_shortfalls']}) - "
                  f"outward/inward children realizing ONE core")
    if len(cand):
        mix = (cand[cand["source"] == "paid"]
               .groupby(["move_class", "fresh_radial_dir"]).size())
        print("[plan] paid stratum mix:")
        print(mix.to_string())
        bs = cand[cand["source"] == "paid"]["burn_state"].value_counts()
        print(f"[plan] burn-state mix: {bs.to_dict()}")
    print(f"[plan] manifest sha256 "
          f"{hashlib.sha256(out.read_bytes()).hexdigest().upper()}")
    return 0


# --------------------------------------------------------------------------- #
# blind s1i scoring
# --------------------------------------------------------------------------- #
def cmd_score(args) -> int:
    import pandas as pd
    from lpopt.data.schema import unpack_pattern
    from lpopt.model.model_api import PosValCnnBackend
    from lpopt.search import acquisition as ACQ
    from lpopt.search.construct import CaseContext

    manifest = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cand = pd.DataFrame(manifest["candidates"])
    cells = {c["name"]: c for c in manifest["cells"]}
    selected = [n for n in (args.cells or list(cells)) if n in cells]

    frames = []
    backends: dict[str, Any] = {}
    for name in selected:
        meta = cells[name]
        lib = meta["library_id"]
        if lib not in backends:
            backends[lib] = PosValCnnBackend.from_dir(
                args.model_dir, store_dir=args.store_dir, library_id=lib,
                device="cpu")
            print(f"[score] {args.model_dir} loaded for library {lib}: "
                  f"{len(backends[lib].members)} members, targets "
                  f"{backends[lib].target_names}")
        model = backends[lib]
        block = cand[(cand["cell"] == name) & (cand["source"] == "paid")].copy()
        if block.empty:
            continue
        parents = {p["record_id"]: p for p in meta["parents"]}
        ctx0 = CaseContext(meta["pair"], int(meta["feed"]), lib)
        has_head = ACQ.has_fxy_head(model, ctx0)
        print(f"[score] {name}: n={len(block)}  f_xy head "
              f"{'PRESENT' if has_head else 'ABSENT (F_r proxy)'}")

        # Parent AND child are scored, so the registered blind quantity is the
        # PREDICTED DELTA - which is what a D1 row is about - not a level whose
        # parent baseline was never predicted.
        rows: list[dict[str, Any]] = []
        for prid, blk in block.groupby("parent_record_id", sort=True):
            p = parents[prid]
            todo = [("__parent__", p["pattern"], p.get("e_core"))]
            todo += list(zip(blk["record_id"], blk["pattern"], blk["e_core"]))
            # ``predict`` broadcasts ONE e_core over the batch, so rows are
            # grouped by their own e_core: batch_flip changes the fresh batch
            # multiset and therefore e_core, and serving it under the parent's
            # e_core would be a different core than the one that will be run.
            by_ecore: dict[float, list[tuple[str, str]]] = {}
            for rid, packed, ec in todo:
                by_ecore.setdefault(round(float(ec or 0.0), 6), []).append(
                    (rid, packed))
            for ec, items in sorted(by_ecore.items()):
                ctx = CaseContext(meta["pair"], int(meta["feed"]), lib, e_core=ec)
                pats = [unpack_pattern(x) for _, x in items]
                pred = model.predict(pats, ctx.case_key, ec)
                pk_mu, pk_sd = ACQ.predict_flatness(model, pats, ctx)[:2]
                fxy_mu, fxy_sd, source = ACQ.predict_fxy(model, pats, ctx, pred)
                mean = np.asarray(pred.mean, dtype=float)
                for j, (rid, _packed) in enumerate(items):
                    rows.append({
                        "cell": name, "record_id": rid,
                        "parent_record_id": prid, "e_core": ec,
                        "pred_f_r": mean[j, 0], "pred_cbc_max": mean[j, 1],
                        "pred_f_q": mean[j, 2], "pred_cyclen": mean[j, 3],
                        "pred_ao_abs": mean[j, 4],
                        "pred_node_peak": float(pk_mu[j]),
                        "pred_node_peak_sd": float(pk_sd[j]),
                        "pred_f_xy": float(fxy_mu[j]),
                        "pred_f_xy_sd": float(fxy_sd[j]),
                        "pred_f_xy_source": source,
                    })
        pred_frame = pd.DataFrame(rows)
        par = pred_frame[pred_frame["record_id"] == "__parent__"].set_index(
            "parent_record_id")
        kid = pred_frame[pred_frame["record_id"] != "__parent__"].copy()
        for col in ("pred_f_r", "pred_cyclen", "pred_node_peak", "pred_f_xy"):
            kid[f"d_{col}"] = (kid[col].to_numpy()
                               - par.loc[kid["parent_record_id"], col].to_numpy())
        kid = kid.merge(
            block[["record_id", "move_class", "fresh_radial_dir", "burn_state",
                   "dose", "pair_id", "pair_role", "stratum"]],
            on="record_id", how="left")
        # The validated readout is PARENT-BLOCKED (policy_v1_results s1): scores
        # from different parents are not comparable, so the within-parent rank is
        # carried explicitly and the prospective test uses it.
        for col in ("d_pred_f_r", "d_pred_f_xy", "d_pred_node_peak"):
            kid[f"rank_{col}_in_parent"] = kid.groupby("parent_record_id")[
                col].rank(ascending=True)
        frames.append(kid)

    if not frames:
        raise SystemExit("nothing to score")
    pred = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"[score] {len(pred)} predictions -> {out}")
    print(f"[score] sha256 {hashlib.sha256(out.read_bytes()).hexdigest().upper()}")
    print("[score] REGISTERED BEFORE TRUTH: this file's sha256 is the "
          "pre-commitment (pre-registration s7).")
    return 0


# --------------------------------------------------------------------------- #
# run - per-cell delegation to ablation_wave's runner
# --------------------------------------------------------------------------- #
def _fxy_row(outcome) -> dict[str, Any]:
    peaks = getattr(outcome, "fxy", None)
    return {
        "record_id": str(outcome.meta.get("record_id") or ""),
        "status": outcome.status,
        "f_xy": getattr(peaks, "f_xy", None),
        "f_xya": getattr(peaks, "f_xya", None),
        "fxy_n_steps": getattr(peaks, "n_steps", None),
        "fxy_sane": getattr(peaks, "sane", None),
        "fxy_reason": getattr(peaks, "reason", "no_result" if peaks is None else ""),
        "fxy_efpd_max": getattr(peaks, "efpd_max", None),
    }


def load_fxy_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    """``record_id -> parsed FXYP row``; last write wins (a rerun supersedes)."""
    out: dict[str, dict[str, Any]] = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:                                       # noqa: BLE001
            continue
        rid = str(row.get("record_id") or "")
        if rid:
            out[rid] = row
    return out


def _campaign_suffix(args: Any) -> str:
    """``--campaign-suffix`` of an argparse namespace (``""`` when absent).

    Read defensively: several callers (tests, ``resume_*.bat`` shims) build a
    ``SimpleNamespace`` by hand, and a correction re-run's tag must never be the
    reason one of them stops working.
    """
    return str(getattr(args, "campaign_suffix", "") or "")


@contextmanager
def _cell_binding(cell: Cell, run_dir: Path, package: Path,
                  fuel_types: Path, campaign_suffix: str = "") -> Iterator[Path]:
    """Rebind ``ablation_wave``'s provenance + patch the two F_xy gaps.

    Everything is restored on exit, so a multi-cell ``run`` cannot leak one
    cell's identity into the next.  The three patches, all on
    ``lpopt.search.*`` and none on ``ablation_wave``:

    1. ``CaseAssetResolver`` - the ablation runner constructs it with the package
       and dims only.  A paramA cell additionally needs the fuel library and the
       synth-deck cache root (``resolver.build_case_resolver``), which the shim
       injects; a ga80 cell gets the library id and nothing else, i.e. the
       ablation call byte-for-byte.
    2. ``WaveVerifier`` -> the F_xy-capturing subclass (the sidecar) which ALSO
       hands the verifier this cell's resolver EXPLICITLY (see below).
    3. ``outcome_to_record`` -> fills ``f_xy`` / ``f_xya`` from that sidecar when
       the kit is rebuilt from the jsonl.

    THE ``resolver=`` INJECTION (defect 20260830 — read this before touching it)
    --------------------------------------------------------------------------
    Patching ``lpopt.search.assets.CaseAssetResolver`` reaches the resolver
    ``ablation_wave.cmd_run`` builds for *asset resolution* (it imports the name
    inside the function), but NOT the one that emits decks: ``verify.py`` bound
    ``CaseAssetResolver`` into its own namespace at import time and
    ``WaveVerifier`` built its own fallback instance from it.  On ``HGD569_f125``
    (13-character ``type_id``\\ s) that fallback had an empty alias bridge, so
    every ``%LPD_SHF`` fresh card kept its raw ``type_id``, MASTER absorbed it as
    an unrelated batch, and 160 chains were computed on a core nobody designed
    (``data/reports/hgd569_degeneracy_memo_20260830.md``).  The fix is not a
    wider patch: it is to PASS the resolver, exactly as ``campaign.py`` and
    ``produce.py`` do.  ``ablation_wave.py`` is sha-pinned, so the injection
    happens in the ``WaveVerifier`` subclass this module already owns.
    """
    import lpopt.search.assets as A
    import lpopt.search.verify as V

    sidecar = run_dir / FXY_SIDECAR_NAME
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    campaign = cell.campaign + campaign_suffix
    saved = {k: getattr(W, k) for k in
             ("PAIR", "FEED", "LIBRARY", "CELL", "CAMPAIGN", "GENERATOR")}
    W.PAIR, W.FEED, W.LIBRARY = cell.pair, cell.feed, cell.library
    W.CELL, W.CAMPAIGN, W.GENERATOR = cell.key, campaign, GENERATOR

    orig_resolver = A.CaseAssetResolver
    orig_verifier = V.WaveVerifier
    orig_otr = V.outcome_to_record

    extra: dict[str, Any] = {"library_id": cell.library}
    if cell.kit == "paramA":
        from lpopt.search.resolver import paramA_registry_aliases
        aliases = paramA_registry_aliases(package)
        if not aliases:
            raise SystemExit(
                f"{cell.name}: no type_id->alias bridge in {package/'registry.json'}; "
                f"a paramA cell's %LPD_SHF cards would carry raw type_ids that "
                f"MASTER absorbs silently (memo 20260830).  Refusing to run.")
        extra["registry_aliases"] = aliases
        # A paramA pair with no packaged template deck (the HGD569 family) gets a
        # SYNTHESIZED reload deck cached under synth_decks/<pair>/ - the same
        # ``[produce].synth_decks_root`` the produce driver passes.  Without it
        # every chain of such a cell dies as a MissingCaseAssetError.
        extra["synth_root"] = BASE / "data/design/synth_decks"
        try:
            from lpopt.data.fuel_types import FuelLibrary
            extra["fuel_library"] = FuelLibrary.from_parquet(fuel_types)
        except Exception as exc:                                # noqa: BLE001
            print(f"[run] WARNING: fuel table unreadable ({exc}); the level-3 "
                  f"pair_ecore restart fallback is disabled for {cell.name}")

    def resolver_shim(package_root, *a, **kw):
        for k, v in extra.items():
            kw.setdefault(k, v)
        return orig_resolver(package_root, *a, **kw)

    def cell_resolver(**kw: Any):
        """This cell's fully configured resolver (alias bridge included)."""
        kw.setdefault("package_root", package)
        return resolver_shim(kw.pop("package_root"), **kw)

    class FxyCapturingWaveVerifier(orig_verifier):              # type: ignore[misc,valid-type]
        """``WaveVerifier`` that also persists each outcome's parsed FXYP, and
        that is handed this cell's resolver EXPLICITLY (see :func:`_cell_binding`).
        """

        def __init__(self, *a: Any, **kw: Any) -> None:
            if kw.get("resolver") is None:
                kw["resolver"] = cell_resolver(
                    package_root=Path(kw.get("package_root") or package),
                    library_dims=tuple(kw.get("library_dims")
                                       or W._library_dims(package)),
                )
            super().__init__(*a, **kw)

        def evaluate_wave(self, entries):
            outcomes = super().evaluate_wave(entries)
            try:
                with sidecar.open("a", encoding="utf-8") as fh:
                    for oc in outcomes:
                        fh.write(json.dumps(_fxy_row(oc)) + "\n")
                    fh.flush()
            except OSError as exc:                              # noqa: BLE001
                # A label-harvest failure must never abort a wave (the
                # ``fxy_from_work_dir`` contract); the chain's scalars are
                # already safe in the results jsonl.
                print(f"[run] WARNING: F_xy sidecar write failed: {exc}")
            return outcomes

    def otr_shim(outcome, **kw):
        rec = orig_otr(outcome, **kw)
        if rec.f_xy is None:
            row = load_fxy_sidecar(sidecar).get(rec.record_id)
            if row is not None:
                rec.f_xy = row.get("f_xy")
                rec.f_xya = row.get("f_xya")
        return rec

    A.CaseAssetResolver = resolver_shim
    V.WaveVerifier = FxyCapturingWaveVerifier
    V.outcome_to_record = otr_shim
    try:
        yield sidecar
    finally:
        A.CaseAssetResolver = orig_resolver
        V.WaveVerifier = orig_verifier
        V.outcome_to_record = orig_otr
        for k, v in saved.items():
            setattr(W, k, v)


def write_subplan(manifest: dict, cell_meta: dict, path: Path) -> Path:
    """A single-cell manifest in ``ablation_wave``'s own shape.

    ``ablation_wave.cmd_run`` reads ``candidates`` (``source == 'paid'``) and
    ``parents``; giving it exactly that keeps the runner unmodified and keeps the
    per-cell run dir self-describing on the box.
    """
    name = cell_meta["name"]
    sub = dict(cell_meta)
    sub.update({
        "written_utc": manifest.get("written_utc"),
        "wave": manifest.get("wave"),
        "seed": manifest.get("seed"),
        "deck_knobs": manifest.get("deck_knobs"),
        "parent_manifest_sha256": manifest.get("_self_sha256"),
        "candidates": [c for c in manifest["candidates"] if c.get("cell") == name],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sub, indent=1), encoding="utf-8")
    return path


def _selected_cells(manifest: dict, names: Sequence[str] | None,
                    kit: str | None) -> list[Cell]:
    metas = {m["name"]: m for m in manifest["cells"]}
    chosen = [n for n in (names or list(metas)) if n in metas]
    cells = [Cell(m["name"], m["pair"], int(m["feed"]), m["library_id"],
                  m["kit"], m.get("note", ""))
             for m in (metas[n] for n in chosen)]
    if kit:
        cells = [c for c in cells if c.kit == kit]
    return cells


def check_restart(cell: Cell, package: Path, cell_meta: dict,
                  dims: tuple[int, int], *, allow_drift: bool = False) -> str:
    """Refuse a cell whose restart is not the one its PARENTS were labelled on.

    ``ablation_wave.cmd_run``'s own guard is ``fallback_level != 0``, which is the
    right guard for a cell whose parents came off a NATIVE restart.  It is the
    wrong guard for a cell that has no native restart at all: the HGD569 f125
    parents were themselves labelled at ``fallback_level = 3``
    (``pair_ecore:MAS_RST.APRQ_11_0705.02``, 120 of its 144 store rows), so a
    level-3 resolution there is not drift - it is the SAME burnt-fuel history the
    parents carry, which is the only thing a parent/child delta needs.  What must
    never happen is a resolution to a DIFFERENT restart than the parents'; that
    is what this checks, and ``--allow-fallback`` alone would not.

    Must be called INSIDE :func:`_cell_binding`, so the resolver it builds is the
    same shimmed one the runner will build.
    """
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.vendor.masterrl.domain import CaseKey

    expected = list(cell_meta.get("parent_restart_provenance") or [])
    assets = CaseAssetResolver(package, library_dims=dims).resolve(
        CaseKey(cell.pair, cell.feed))
    got = str(assets.restart_provenance or "")
    print(f"[run] restart {got}  fallback_level={assets.fallback_level}  "
          f"parents label(s) {expected or ['<unknown>']}")
    if expected and got not in expected:
        msg = (f"restart drift on {cell.name}: this run resolves {got!r} but its "
               f"parents were labelled on {expected!r}.  Every delta would mix "
               f"the move with a burnt-fuel history change.")
        if not allow_drift:
            raise SystemExit(msg + "  (--allow-restart-drift to override)")
        print("[run] WARNING (overridden): " + msg)
    return got


def cmd_run(args) -> int:
    manifest = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    manifest["_self_sha256"] = hashlib.sha256(
        Path(args.plan).read_bytes()).hexdigest().upper()
    cells = _selected_cells(manifest, args.cells, args.kit)
    if not cells:
        raise SystemExit("no cells selected")
    kits = sorted({c.kit for c in cells})
    if len(kits) > 1:
        raise SystemExit(
            f"refusing to run cells of {kits} in one invocation: asset routing is "
            f"PER LIBRARY (ProduceDriver._run_library_id / "
            f"resolver.build_case_resolver), and this runner builds ONE resolver "
            f"from ONE --package.  Run once per kit with its own --package.")
    kit = kits[0]

    pkg = Path(args.package).resolve()
    if not pkg.is_dir():
        raise SystemExit(f"package not found: {pkg}")
    dims = W._library_dims(pkg)
    from lpopt.search.assets import LIBRARY_DIMS
    if kit == "ga80" and tuple(dims) != tuple(LIBRARY_DIMS):
        raise SystemExit(f"kit ga80 but --package {pkg} reports dims {dims} != "
                         f"{tuple(LIBRARY_DIMS)}; this is the paramA package")
    if kit == "paramA" and tuple(dims) == tuple(LIBRARY_DIMS):
        raise SystemExit(f"kit paramA but --package {pkg} reports the ga80 dims "
                         f"{dims}; point --package at data/design/package")

    metas = {m["name"]: m for m in manifest["cells"]}
    root = Path(args.run_dir).resolve()
    print(f"[run] kit {kit}  package {pkg}  dims {dims}  "
          f"cells {[c.name for c in cells]}")
    rc = 0
    for cell in cells:
        cell_dir = root / cell.run_subdir
        cell_dir.mkdir(parents=True, exist_ok=True)
        sub = write_subplan(manifest, metas[cell.name], cell_dir / SUBPLAN_NAME)
        print(f"\n[run] ==== {cell.name} ({cell.key}) -> {cell_dir} ====")
        with _cell_binding(cell, cell_dir, pkg, Path(args.fuel_types),
                           campaign_suffix=_campaign_suffix(args)):
            check_restart(cell, pkg, metas[cell.name], dims,
                          allow_drift=args.allow_restart_drift)
            sub_args = SimpleNamespace(
                plan=str(sub), package=str(pkg), exe=args.exe,
                run_dir=str(cell_dir), workers=args.workers,
                host_reserve=args.host_reserve, wave_size=args.wave_size,
                max_cycles=args.max_cycles, timeout=args.timeout,
                max_chains=args.max_chains, allow_fallback=args.allow_fallback,
                dry_run=args.dry_run)
            rc = int(W.cmd_run(sub_args) or 0) or rc
    return rc


def cmd_kit(args) -> int:
    manifest = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    manifest["_self_sha256"] = hashlib.sha256(
        Path(args.plan).read_bytes()).hexdigest().upper()
    root = Path(args.run_dir).resolve()
    for cell in _selected_cells(manifest, args.cells, args.kit):
        cell_dir = root / cell.run_subdir
        if not (cell_dir / RESULTS_NAME).exists():
            print(f"[kit] {cell.name}: no results yet")
            continue
        with _cell_binding(cell, cell_dir, Path(args.package),
                           Path(args.fuel_types),
                           campaign_suffix=_campaign_suffix(args)):
            print(f"[kit] {cell.name} -> {cell_dir}")
            W._build_kit(cell_dir)
    return 0


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
#: Response axes of the conditional-effect table.  ``lower_is_better`` drives the
#: "improving" fraction only; the sign tests are on the RAW delta.
RESPONSES: tuple[tuple[str, str, bool], ...] = (
    ("d_f_xy", "F_xy", True),
    ("d_f_r", "F_r", True),
    ("d_node_peak", "node_peak", True),
    ("d_cyclen", "cyclen", False),
)

#: The structurally direction-free control arm (``NEUTRAL_QUOTA``).  A
#: ``rewire_swap`` moves neither fresh placement nor batch labels, so
#: ``d_fresh_enr_r_center`` is IDENTICALLY zero and its vs-parent delta measures
#: whatever is common to the cell rather than to the move: the parent labels'
#: own era, deck or restart.  Round 1 measured +34.71 EFPD / +0.098 F_r / +0.106
#: F_xy of it on ``HGD569_f125`` and the raw pooled tables carried it straight
#: into ``batch_swap``/``batch_flip`` (s7, s11-2).
NEUTRAL_CONTROL_CLASS = "rewire_swap"


def wave_frame(manifest: dict, run_dir: Path,
               cells: Sequence[Cell] | None = None):
    """Join the plan, the results jsonl and the F_xy sidecar into one frame.

    Self-contained on purpose: this reads the RUN's own artefacts, so the
    conditional-effect table is computable on the box before ``merge-store`` has
    ever seen the rows, and a store that has since moved on cannot change a
    published effect size.  ``f_xy`` comes from the sidecar, i.e. from
    ``MAS_OUT`` via :mod:`lpopt.data.fxy` - never from a proxy.
    """
    import pandas as pd

    plan = pd.DataFrame(manifest["candidates"])
    metas = {m["name"]: m for m in manifest["cells"]}
    chosen = ([c.name for c in cells] if cells is not None else list(metas))

    rows = []
    for name in chosen:
        meta = metas[name]
        cdir = Path(run_dir) / meta.get("run_subdir", f"intervention_{name}")
        res = cdir / RESULTS_NAME
        if not res.exists():
            continue
        side = load_fxy_sidecar(cdir / FXY_SIDECAR_NAME)
        for line in res.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                                   # noqa: BLE001
                continue
            fom = r.get("fom") or {}
            sc = side.get(str(r.get("record_id")), {})
            rows.append({
                "cell": name, "record_id": str(r.get("record_id")),
                "parent_record_id": r.get("parent_record_id"),
                "status": r.get("status"), "failure": r.get("failure"),
                "child_f_r": fom.get("F_r"), "child_cyclen": fom.get("cyclen"),
                "child_cbc_max": fom.get("CBC_max"), "child_f_q": fom.get("F_q"),
                "child_node_peak": r.get("node_peak"),
                "child_f_xy": sc.get("f_xy"), "child_f_xya": sc.get("f_xya"),
                "fxy_sane": sc.get("fxy_sane"), "fxy_reason": sc.get("fxy_reason"),
            })
    lab = pd.DataFrame(rows)
    if lab.empty:
        return lab
    keep = ["record_id", "cell", "parent_record_id", "move_class",
            "fresh_radial_dir", "burn_state", "dose", "pair_id", "pair_role",
            "stratum", "move_tag", "swap_span", "swap_radius", "source",
            "d_fresh_enr_r_center", "d_fresh_enr_mass",
            "parent_f_r", "parent_node_peak", "parent_cyclen", "parent_cbc_max",
            "parent_f_xy"]
    keep = [k for k in keep if k in plan.columns]
    frame = lab.merge(plan[keep].drop_duplicates(subset=["record_id", "cell"]),
                      on=["record_id", "cell"], how="left", suffixes=("", "_plan"))
    frame["converged"] = frame["status"] == "converged"
    for fom in ("f_r", "cyclen", "node_peak", "f_xy"):
        frame[f"d_{fom}"] = (pd.to_numeric(frame[f"child_{fom}"], errors="coerce")
                             - pd.to_numeric(frame[f"parent_{fom}"], errors="coerce"))
        frame.loc[~frame["converged"], f"d_{fom}"] = np.nan
    return frame


def effect_table(frame, by: Sequence[str]):
    """n, mean/median delta and improving fraction per group, per response."""
    import pandas as pd

    out = []
    for keys, blk in frame[frame["converged"]].groupby(list(by), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["n"] = int(len(blk))
        for col, label, lower in RESPONSES:
            v = pd.to_numeric(blk[col], errors="coerce").dropna()
            row[f"n_{label}"] = int(len(v))
            row[f"mean_{label}"] = float(v.mean()) if len(v) else np.nan
            row[f"median_{label}"] = float(v.median()) if len(v) else np.nan
            better = (v < 0) if lower else (v > 0)
            row[f"improving_{label}"] = float(better.mean()) if len(v) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def neutral_control_offset(frame, by: Sequence[str] = ("cell",)):
    """Common-mode vs-parent offset the NEUTRAL CONTROL arm exposes per group.

    A ``rewire_swap`` is registered as structurally direction-free, so under the
    campaign's own model its mean vs-parent delta is an estimate of ZERO plus
    whatever the cell adds to every child alike.  When that estimate is not zero
    the cell's parent labels and its child labels do not sit on the same
    baseline, and every vs-parent number in that cell - including the pooled ones
    the other cells share - is shifted by it.  The design already paid for this
    arm, so the diagnostic is free; round 1 had to reconstruct it by hand.

    One row per (group x response): ``n`` chains, ``n_parents`` blocks, and the
    ``offset`` to subtract.  ``delta_col`` names the column it applies to.
    """
    import pandas as pd

    by = list(by)
    ctrl = frame[frame["converged"].fillna(False).astype(bool)
                 & (frame["move_class"] == NEUTRAL_CONTROL_CLASS)]
    cols = [*by, "response", "delta_col", "n", "n_parents", "offset"]
    out = []
    for keys, blk in ctrl.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(by, keys))
        for col, label, _lower in RESPONSES:
            v = pd.to_numeric(blk[col], errors="coerce").dropna()
            out.append({
                **base, "response": label, "delta_col": col, "n": int(len(v)),
                "n_parents": int(blk.loc[v.index, "parent_record_id"].nunique()),
                "offset": float(v.mean()) if len(v) else np.nan})
    return pd.DataFrame(out, columns=cols)


def adjust_by_neutral_control(frame, offsets, by: Sequence[str] = ("cell",)):
    """``frame`` with each response delta reduced by its group's control offset.

    The RAW frame is never modified: the adjusted copy is what the ``*_adj``
    tables are built from, and both are published.  A group with no usable
    control row is left RAW (offset 0) rather than dropped - "not corrected" is
    a different statement from "corrected by nothing", and ``*_offset`` says
    which by carrying NaN.

    The paired out-minus-in contrasts need no adjusted twin: a common-mode
    offset is shared by both members of a pair and cancels in the difference,
    which is precisely why that statistic survived round 1's baseline shift.
    """
    import pandas as pd

    by = list(by)
    adj = frame.copy()
    for col, _label, _lower in RESPONSES:
        table = (offsets[offsets["delta_col"] == col][[*by, "offset"]]
                 if len(offsets) else None)
        if table is None or table.empty:
            adj[f"{col}_offset"] = np.nan
            continue
        off = adj[by].merge(table, on=by, how="left")["offset"].to_numpy(dtype=float)
        adj[f"{col}_offset"] = off
        adj[col] = (pd.to_numeric(adj[col], errors="coerce").to_numpy()
                    - np.nan_to_num(off))
    return adj


def paired_contrasts(frame, by: Sequence[str]):
    """Within-``pair_id`` (outward - inward) differences + an exact sign test.

    This is the campaign's primary statistic.  The two members of a pair share a
    parent, a move class and (by construction) a burn state, so the difference
    removes parent difficulty exactly - the review's stated reason for D1 - and
    no between-parent model is needed to read it.
    """
    import pandas as pd
    from ablation_analyze import _sign_test

    conv = frame[frame["converged"] & frame["pair_id"].notna()]
    out = []
    for keys, blk in conv.groupby(list(by), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(by, keys))
        for col, label, _lower in RESPONSES:
            diffs = []
            for _pid, pair in blk.groupby("pair_id"):
                o = pd.to_numeric(
                    pair[pair["pair_role"] == "outward"][col], errors="coerce")
                i = pd.to_numeric(
                    pair[pair["pair_role"] == "inward"][col], errors="coerce")
                if o.notna().any() and i.notna().any():
                    diffs.append(float(o.mean() - i.mean()))
            arr = np.asarray(diffs, dtype=float)
            pos, n, p = _sign_test(arr)
            out.append({**base, "response": label, "n_pairs": int(len(arr)),
                        "mean_out_minus_in": float(np.nanmean(arr)) if len(arr) else np.nan,
                        "median_out_minus_in": float(np.nanmedian(arr)) if len(arr) else np.nan,
                        "sign_pos": pos, "sign_n": n, "sign_p": p})
    return pd.DataFrame(out)


def parent_blocked_signs(frame, by: Sequence[str]):
    """Exact sign test on PARENT-MEAN deltas - one observation per parent.

    Children of one parent are not independent draws (they share a board, a
    restart and a difficulty), so the analysis unit is the parent, exactly as the
    ablation wave's bootstrap resamples parents rather than chains.
    """
    import pandas as pd
    from ablation_analyze import _sign_test

    conv = frame[frame["converged"]]
    out = []
    for keys, blk in conv.groupby(list(by), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(by, keys))
        for col, label, _lower in RESPONSES:
            means = (blk.assign(_v=pd.to_numeric(blk[col], errors="coerce"))
                     .dropna(subset=["_v"])
                     .groupby("parent_record_id")["_v"].mean().to_numpy())
            pos, n, p = _sign_test(means)
            out.append({**base, "response": label, "n_parents": int(len(means)),
                        "mean_of_parent_means": (float(np.mean(means))
                                                 if len(means) else np.nan),
                        "sign_pos": pos, "sign_n": n, "sign_p": p})
    return pd.DataFrame(out)


def cmd_analyze(args) -> int:
    import pandas as pd

    manifest = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cells = _selected_cells(manifest, args.cells, args.kit)
    frame = wave_frame(manifest, Path(args.run_dir), cells)
    if frame.empty:
        raise SystemExit("no results found under --run-dir")
    n_conv = int(frame["converged"].sum())
    n_fxy = int(pd.to_numeric(frame["child_f_xy"], errors="coerce").notna().sum())
    print(f"[analyze] {len(frame)} chains, {n_conv} converged, "
          f"{n_fxy} with a parsed F_xy")
    print(f"[analyze] status mix: {frame['status'].value_counts().to_dict()}")

    per_cell = effect_table(frame, ["cell", "move_class", "fresh_radial_dir"])
    pooled = effect_table(frame, ["move_class", "fresh_radial_dir"])
    by_burn = effect_table(frame, ["move_class", "fresh_radial_dir", "burn_state"])
    pairs_cell = paired_contrasts(frame, ["cell", "move_class"])
    pairs_pooled = paired_contrasts(frame, ["move_class"])
    signs = parent_blocked_signs(frame, ["cell", "move_class", "fresh_radial_dir"])

    # Cell-baseline diagnostic (s11-2).  The RAW tables above are published
    # unchanged; the ``_adj`` twins are the same tables computed on deltas with
    # the cell's own neutral-control offset removed.
    offsets = neutral_control_offset(frame, ["cell"])
    adj = adjust_by_neutral_control(frame, offsets, ["cell"])
    per_cell_adj = effect_table(adj, ["cell", "move_class", "fresh_radial_dir"])
    pooled_adj = effect_table(adj, ["move_class", "fresh_radial_dir"])
    by_burn_adj = effect_table(adj, ["move_class", "fresh_radial_dir",
                                     "burn_state"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in (("effects_by_cell", per_cell), ("effects_pooled", pooled),
                        ("effects_by_burn_state", by_burn),
                        ("neutral_control_offset", offsets),
                        ("effects_by_cell_adj", per_cell_adj),
                        ("effects_pooled_adj", pooled_adj),
                        ("effects_by_burn_state_adj", by_burn_adj),
                        ("paired_by_cell", pairs_cell),
                        ("paired_pooled", pairs_pooled),
                        ("parent_blocked_signs", signs)):
        path = out_dir / f"{args.wave}_{name}.csv"
        table.to_csv(path, index=False)
        written.append(path)
    frame.to_csv(out_dir / f"{args.wave}_rows.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print("\n[analyze] NEUTRAL-CONTROL cell offsets "
              f"({NEUTRAL_CONTROL_CLASS}; subtracted in the *_adj tables)")
        print(offsets.to_string(index=False))
        print("\n[analyze] POOLED conditional effects "
              "(move_class x radial direction)")
        print(pooled.to_string(index=False))
        print("\n[analyze] POOLED conditional effects, cell-baseline ADJUSTED")
        print(pooled_adj.to_string(index=False))
        print("\n[analyze] PAIRED within-parent contrasts, pooled "
              "(outward - inward)")
        print(pairs_pooled.to_string(index=False))
        print("\n[analyze] PER-CELL paired contrasts")
        print(pairs_cell.to_string(index=False))
    print("\n[analyze] wrote " + ", ".join(p.name for p in written)
          + f", {args.wave}_rows.csv -> {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# corpus append
# --------------------------------------------------------------------------- #
def cmd_corpus(args) -> int:
    """Append each cell's edges to ``steps.parquet`` as ``intervention_<cell>``.

    The rows are produced by ``mine_policy_corpus.build_steps`` ITSELF (through
    ``ablation_analyze.build_wave_steps``), so every column is schema-identical
    to the mined corpus and a row emitted here equals the row a full re-mine
    would produce for the same edge.  The only column this path sets is the
    provenance tag.
    """
    import pandas as pd
    import ablation_analyze as A
    import mine_policy_corpus as M

    manifest = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cells = _selected_cells(manifest, args.cells, args.kit)
    A.STORE = Path(args.store)
    A.STEPS = Path(args.steps)

    news = []
    for cell in cells:
        try:
            new = A.build_wave_steps(cell.campaign + _campaign_suffix(args),
                                     cell.lineage_source)
        except SystemExit as exc:
            print(f"[corpus] {cell.name}: SKIPPED ({exc})")
            continue
        n_single = (int(new["single_move"].sum())
                    if "single_move" in new.columns else -1)
        print(f"[corpus] {cell.name}: {len(new)} edges, single_move "
              f"{n_single}/{len(new)}")
        news.append(new)
    if not news:
        raise SystemExit("no intervention rows in the store - merge-store first")
    new = pd.concat(news, ignore_index=True)
    existing = pd.read_parquet(A.STEPS)
    print(f"[corpus] existing steps.parquet: {len(existing)} rows, "
          f"{len(existing.columns)} cols")
    missing = set(existing.columns) - set(new.columns)
    extra = set(new.columns) - set(existing.columns)
    if missing or extra:
        raise SystemExit(
            f"schema drift - missing {sorted(missing)} extra {sorted(extra)}"
            + ("  (the corpus predates the F_xy / burn_state columns; run "
               "`python policy_v2_corpus.py backfill-fxy --apply` first)"
               if extra & set(M.FXY_SCHEMA_COLUMNS) else ""))
    new = new[existing.columns]
    print(new.groupby(["lineage_source", "move_class", "fresh_radial_dir"])
          .size().to_string())
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["parent_record_id", "child_record_id"], keep="last")
    print(f"[corpus] combined {len(combined)} rows "
          f"(+{len(combined) - len(existing)})")
    if args.dry_run:
        print("[corpus] DRY RUN - steps.parquet not written")
        return 0
    backup = A.STEPS.with_suffix(f".parquet.bak_pre_{args.wave}")
    if not backup.exists():
        backup.write_bytes(A.STEPS.read_bytes())
        print(f"[corpus] backup -> {backup.name}")
    combined.to_parquet(A.STEPS, index=False)
    print(f"[corpus] wrote {A.STEPS}")
    return 0


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #
DEFAULT_PLAN = "data/design/intervention_wave_r1.json"
DEFAULT_RUN_DIR = "runs/intervention_wave_r1"
WAVE = "intervention_wave_r1"


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cells", type=_csv, default=None,
                       help="comma-separated cell names (default: all)")
        p.add_argument("--kit", choices=("paramA", "ga80"), default=None)

    p = sub.add_parser("plan", help="parents + balanced paired sample -> manifest")
    common(p)
    p.add_argument("--out", default=DEFAULT_PLAN)
    p.add_argument("--store", default="data/store/records.parquet")
    p.add_argument("--fuel-types", default="data/store/fuel_types.parquet")
    p.add_argument("--parents", type=int, default=N_PARENTS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--wave", default=WAVE)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("score", help="blind s1i predictions (BEFORE truth)")
    common(p)
    p.add_argument("--plan", default=DEFAULT_PLAN)
    p.add_argument("--model-dir", default="data/models/s1i")
    p.add_argument("--store-dir", default="data/store")
    p.add_argument("--out", default="data/design/intervention_wave_r1_s1i_pred.csv")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("run", help="MASTER equilibrium chains for ONE kit")
    common(p)
    p.add_argument("--plan", default=DEFAULT_PLAN)
    p.add_argument("--package", default="data/design/package")
    p.add_argument("--fuel-types", default="data/store/fuel_types.parquet")
    p.add_argument("--exe", default="C:/DeCART_MASTER/BIN/master4.0m4_r1.exe")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--host-reserve", type=int, default=1)
    p.add_argument("--wave-size", type=int, default=24)
    p.add_argument("--max-cycles", type=int, default=16)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--max-chains", type=int, default=0)
    p.add_argument("--allow-fallback", action="store_true",
                   help="permit a non-native restart (REQUIRED for a cell whose "
                        "parents were themselves labelled on a fallback restart; "
                        "the restart-provenance gate is the real check)")
    p.add_argument("--allow-restart-drift", action="store_true",
                   help="override the parents-vs-run restart equality gate")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--campaign-suffix", default="",
                   help="appended to the cell CAMPAIGN tag (and to the "
                        "store `campaign` column the kit stamps).  Use it "
                        "for a CORRECTION re-run, e.g. --campaign-suffix _v2: "
                        "the record_id is unchanged (it is a function of "
                        "pattern/library/pair/deck_knobs only), so the new "
                        "rows UPGRADE the quarantined ones on merge while the "
                        "tag says which pass produced the surviving label.")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("kit", help="rebuild the per-cell merge-store kits")
    common(p)
    p.add_argument("--plan", default=DEFAULT_PLAN)
    p.add_argument("--package", default="data/design/package")
    p.add_argument("--fuel-types", default="data/store/fuel_types.parquet")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--campaign-suffix", default="",
                   help="appended to the cell CAMPAIGN tag (and to the "
                        "store `campaign` column the kit stamps).  Use it "
                        "for a CORRECTION re-run, e.g. --campaign-suffix _v2: "
                        "the record_id is unchanged (it is a function of "
                        "pattern/library/pair/deck_knobs only), so the new "
                        "rows UPGRADE the quarantined ones on merge while the "
                        "tag says which pass produced the surviving label.")
    p.set_defaults(func=cmd_kit)

    p = sub.add_parser("analyze", help="conditional-effect tables + sign tests")
    common(p)
    p.add_argument("--plan", default=DEFAULT_PLAN)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--out-dir", default="data/reports")
    p.add_argument("--wave", default=WAVE)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("corpus", help="append edges to data/policy/steps.parquet")
    common(p)
    p.add_argument("--plan", default=DEFAULT_PLAN)
    p.add_argument("--store", default="data/store/records.parquet")
    p.add_argument("--steps", default="data/policy/steps.parquet")
    p.add_argument("--wave", default=WAVE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--campaign-suffix", default="",
                   help="appended to the cell CAMPAIGN tag (and to the "
                        "store `campaign` column the kit stamps).  Use it "
                        "for a CORRECTION re-run, e.g. --campaign-suffix _v2: "
                        "the record_id is unchanged (it is a function of "
                        "pattern/library/pair/deck_knobs only), so the new "
                        "rows UPGRADE the quarantined ones on merge while the "
                        "tag says which pass produced the surviving label.")
    p.set_defaults(func=cmd_corpus)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
