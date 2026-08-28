"""Candidate-pool construction for the guided-search campaign (plan sec. 4.6).

:func:`build_pool` assembles a deduplicated pool of valid genome candidates from
three sources, in the plan's proportions:

* **elite mutation (~60%)** — parents are the top verified feasible/objective
  store rows for the case (nearest-feed via feed-morph when the exact cell is
  empty) plus the top-predicted candidates from the previous wave's pool; each
  child is a small-``n_moves`` mutation (a trust-region bias that widens with the
  wave index).  ``[acquisition] policy_prior`` (default ``"off"``) optionally
  ranks each parent's proposed edits with the v1 learned move policy and
  softmax-samples one, keeping a floor of unscored random mutations
  (``data/reports/policy_v1_results_20260815.md`` section 7);
* **rollout-scored guided construction (~30%)** — a seed ("prefix") genome is
  greedily completed into ``completions_per_prefix`` *complete* patterns, the
  surrogate scores those complete boards (the CNN never scores a partial board,
  plan sec. 4.6), and the beam-best completions are kept;
* **diversity (~10%)** — heuristic (ring/checker/radial) and uniform-random
  genomes.

Every candidate carries its ``origin``, its ``parent_record_id`` (lineage for
the S1 closure split), and its ``record_id`` (the schema preimage), so the pool
is deduplicated against the campaign ledger and within itself by ``record_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

import numpy as np

from ..data.fuel_types import core_enrichment_split
from ..data.schema import compute_record_id
from ..vendor.masterrl.domain import CaseKey, Pattern
from .genome import (
    GeneralOrbitGenome,
    GenomeError,
    _add_fresh_unit,
    _remove_fresh_unit,
    case_batches,
    mutate,
    random_genome,
)
from .produce import heuristic_fresh_set
from .verify import PRODUCE_DECK_KNOBS

#: record_id deck-knob signature for campaign candidates — identical to the
#: produce harness so campaign labels dedup against produce/store rows (plan 4.2).
CAMPAIGN_DECK_KNOBS = PRODUCE_DECK_KNOBS


# --------------------------------------------------------------------------- #
# case context + candidate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseContext:
    """The fixed external decision variables of a campaign case (plan sec. 6.2)."""

    pair: str
    feed: int
    library_id: str = "ga80"
    e_core: float | None = None
    center_batch: str = "auto"
    max_shuffle_depth: int = 2
    #: Reject any candidate that does not FEED every member of :attr:`batches`.
    #: Off by default, so every existing deck is unaffected (a 2-type board
    #: always uses both members, making the check a no-op for a pair case).
    #:
    #: Exists for a GRADED case.  A 3-type alphabet admits boards that feed only
    #: two of its members, and such a board is not merely "less graded" -- it is
    #: PHYSICALLY THE SAME CORE as a 2-type board of that sub-alphabet, with an
    #: identical ``%LPD_SHF``.  Verifying one spends a MASTER call re-measuring a
    #: population that already carries labels under the pair's own case id, and
    #: it does so on the say-so of a model that has never seen a 3-type input.
    #: With the flag on, the budget buys genuinely graded cores only.
    require_all_batches: bool = False

    def uses_full_alphabet(self, genome: "GeneralOrbitGenome") -> bool:
        """True unless :attr:`require_all_batches` is set and a member is unfed.

        The search can still drive a member down to a SINGLE assembly, so "does
        the optimizer actually want the extra type?" stays answerable -- it just
        cannot be answered by deleting the type and reporting the resulting
        2-type core as a graded result.
        """

        if not self.require_all_batches:
            return True
        counts = genome.batch_counts
        return all(counts.get(b, 0) > 0 for b in self.batches)

    @property
    def batches(self) -> tuple[str, ...]:
        """The case's fresh-type alphabet — 2 types (pair) or 3..5 (graded)."""

        return case_batches(self.pair)

    @property
    def resolved_center(self) -> str:
        return self.batches[0] if self.center_batch == "auto" else self.center_batch

    @property
    def n_fresh(self) -> int:
        return (int(self.feed) - 1) // 4

    @property
    def case_key(self) -> CaseKey:
        return CaseKey(self.pair, int(self.feed))

    @property
    def allow_discharge(self) -> bool:
        return self.n_fresh > 30


@dataclass
class Candidate:
    """One pool candidate: a validated pattern + its genome + provenance."""

    pattern: Pattern
    genome: GeneralOrbitGenome
    origin: str                       # elite | guided | heuristic | random
    parent_record_id: str | None
    record_id: str
    e_core: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def candidate_record_id(pattern: Pattern, ctx: CaseContext) -> str:
    return compute_record_id(
        pattern.canonical(), ctx.library_id, ctx.pair, CAMPAIGN_DECK_KNOBS
    )


def _morph_feed(
    genome: GeneralOrbitGenome,
    rng: random.Random,
    target_n: int,
    batches: tuple[str, ...],
) -> GeneralOrbitGenome | None:
    """Directed feed-morph to ``target_n`` fresh units (add/remove moves)."""

    guard = 0
    current = genome
    while current.n_fresh != target_n and guard < 400:
        guard += 1
        if current.n_fresh < target_n:
            nxt = _add_fresh_unit(current, rng, batches)
        else:
            nxt = _remove_fresh_unit(current, rng)
        if nxt == current:
            continue
        current = nxt
    return current if current.n_fresh == target_n else None


def _heuristic_genome(ctx: CaseContext, rng: random.Random) -> GeneralOrbitGenome:
    """A ring/checker/radial-biased genome for the case feed (diversity slot)."""

    rule = rng.choice(("ring", "checker", "radial"))
    target = heuristic_fresh_set(rule, ctx.n_fresh)
    best: GeneralOrbitGenome | None = None
    best_overlap = -1
    for _ in range(20):
        g = random_genome(
            rng, ctx.pair, ctx.n_fresh,
            max_shuffle_depth=ctx.max_shuffle_depth,
            allow_single_cycle_discharge=ctx.allow_discharge,
        )
        overlap = len(g.fresh_units & target)
        if overlap > best_overlap:
            best_overlap, best = overlap, g
    assert best is not None
    return best


#: Printed once per process when a deck asks for the policy prior and the
#: checkpoints will not load.  Without it the campaign would run as a silent
#: control arm — the failure mode that makes an A/B measure nothing.
_POLICY_WARNED = False


def _policy_prior(cfg: Any) -> tuple[Any, str]:
    """``(scorer, mode)`` for the elite-mutation prior; ``(None, "off")`` if unused.

    ``mode`` is ``[acquisition] policy_prior``.  Loading is deferred to the first
    call that actually wants it, so a flag-off campaign never imports torch.
    """

    global _POLICY_WARNED
    acq = getattr(cfg, "acquisition", None)
    mode = str(getattr(acq, "policy_prior", "off") or "off").strip().lower()
    if mode == "off":
        return None, "off"
    from ..policy.scorer import DEFAULT_MODEL_DIR, get_scorer

    scorer = get_scorer(
        getattr(acq, "policy_prior_model_dir", DEFAULT_MODEL_DIR),
        device="cpu",
        n_threads=int(getattr(acq, "policy_prior_threads", 4)),
    )
    if scorer is None:
        if not _POLICY_WARNED:
            _POLICY_WARNED = True
            print(f"[construct] WARNING [acquisition] policy_prior={mode!r} but the "
                  f"policy checkpoints did not load; the elite arm falls back to "
                  f"UNSCORED random mutation for this whole run")
        return None, "off"
    return scorer, mode


def _policy_pick(
    scorer: Any,
    mode: str,
    ctx: CaseContext,
    parent: GeneralOrbitGenome,
    candidates: list[tuple[GeneralOrbitGenome, Pattern, str]],
    rng: random.Random,
    temperature: float,
) -> int:
    """Index of the candidate to admit: a SOFTMAX SAMPLE over policy score.

    Not ``argmax`` — the results report's first safety rail (section 7).  A hard
    argmax on a scorer with 0.650 era AUC collapses the neighbourhood diversity
    that stage 3's quota exists to protect, so the policy tilts the draw and does
    not own it.  A scoring failure falls back to a uniform draw for the same
    reason ``_score_completions`` swallows a surrogate failure: the prior must
    never abort construction.
    """

    try:
        probs = scorer.score(
            (parent, parent.to_pattern()),
            [(genome, pattern) for genome, pattern, _ in candidates],
            ctx,
        )
        from ..policy.scorer import HEAD_INDEX

        if mode == "both":
            score = np.asarray(probs, dtype=float).mean(axis=1)
        else:
            score = np.asarray(probs, dtype=float)[:, HEAD_INDEX[mode]]
        if not np.all(np.isfinite(score)):
            raise ValueError("policy returned a non-finite score")
    except Exception:  # noqa: BLE001 — the prior must never abort construction
        return rng.randrange(len(candidates))

    tau = max(float(temperature), 1e-6)
    weight = np.exp((score - score.max()) / tau)
    total = weight.sum()
    if not np.isfinite(total) or total <= 0.0:
        return rng.randrange(len(candidates))
    # Sampled off the SAME rng the rest of the pool draws from, so a seeded
    # campaign stays reproducible.
    draw = rng.random() * float(total)
    return min(int(np.searchsorted(np.cumsum(weight), draw)), len(candidates) - 1)


def _parent_to_genome(
    pattern: Pattern, ctx: CaseContext, rng: random.Random
) -> GeneralOrbitGenome | None:
    """Parse a parent pattern to a genome and feed-morph it onto the case feed."""

    try:
        genome = GeneralOrbitGenome.from_pattern(
            pattern,
            max_shuffle_depth=max(2, ctx.max_shuffle_depth),
            allow_single_cycle_discharge=True,
        )
    except GenomeError:
        return None
    if genome.n_fresh != ctx.n_fresh:
        genome = _morph_feed(genome, rng, ctx.n_fresh, ctx.batches)
    return genome


# --------------------------------------------------------------------------- #
# pool builder
# --------------------------------------------------------------------------- #
def build_pool(
    ctx: CaseContext,
    model: Any,
    store_elites: Sequence[tuple[str | None, Pattern]],
    ledger_ids: set[str],
    rng: random.Random,
    cfg: Any,
    *,
    wave_index: int = 0,
    prev_top: Sequence[tuple[str | None, Pattern]] = (),
    near_miss_parents: Sequence[tuple[str | None, Pattern]] = (),
    size: int | None = None,
) -> list[Candidate]:
    """Build a deduplicated candidate pool for one wave (plan sec. 4.6).

    ``store_elites`` / ``prev_top`` are ``(record_id, Pattern)`` parent seeds.
    ``near_miss_parents`` are ``(record_id, Pattern)`` verified almost-feasible
    boards this campaign (F_r just above the limit); they join the elite-mutation
    parent set with an ``n_moves = 1`` small-move (trust-region) bias so the pool
    tightly probes the feasibility boundary.  ``ledger_ids`` are the already-
    verified record_ids (hard dedup).  Elite children are generated **round-robin**
    across ALL parents so no single parent's neighbourhood dominates the pool
    (the pilot's parent-concentration failure).  ``size`` overrides the configured
    pool size (StubEvaluator tests pass ~500).
    """

    search = cfg.search
    target = int(size if size is not None else search.pool_size)
    target = min(max(1, target), int(search.pool_cap))

    n_elite = int(round(target * search.elite_frac))
    n_guided = int(round(target * search.guided_frac))
    n_div = max(0, target - n_elite - n_guided)

    n_moves = search.n_moves_early if wave_index < 3 else search.n_moves_late

    seen: set[str] = set()
    pool: list[Candidate] = []

    def _admit(cand: Candidate) -> bool:
        if len(pool) >= target:      # hard cap: never overshoot the pool size
            return False
        # Graded cases only (no-op unless ctx.require_all_batches): a board that
        # dropped a fresh type is the same physical core as a smaller-alphabet
        # board and must not consume this case's budget.
        if not ctx.uses_full_alphabet(cand.genome):
            return False
        rid = cand.record_id
        if rid in ledger_ids or rid in seen:
            return False
        seen.add(rid)
        pool.append(cand)
        return True

    # -- parents (near-miss + elite mutation + previous-wave top predictions) - #
    # Near-miss parents lead and carry an n_moves=1 small-move bias; the parent
    # set is deduplicated by record_id so a board that is both a near-miss and an
    # elite is mutated once, with the tighter (near-miss) move budget.
    near_miss_rids = {rid for rid, _ in near_miss_parents if rid is not None}
    parents: list[tuple[str | None, GeneralOrbitGenome, int]] = []
    seen_parent_rids: set[str] = set()
    for rid, pattern in [*near_miss_parents, *store_elites, *prev_top]:
        if rid is not None and rid in seen_parent_rids:
            continue
        genome = _parent_to_genome(pattern, ctx, rng)
        if genome is None:
            continue
        moves = 1 if rid in near_miss_rids else max(1, n_moves)
        parents.append((rid, genome, moves))
        if rid is not None:
            seen_parent_rids.add(rid)

    # -- 1. elite mutation (round-robin over ALL parents) ------------------- #
    # One child per parent per round: every parent is drawn before any parent
    # gets a second child, so the elite-mutation share spreads across the whole
    # elite/near-miss set instead of concentrating on a lucky few.
    #
    # ``[acquisition] policy_prior`` (default "off") inserts the learned move
    # policy HERE and nowhere else: a scored slot proposes ``policy_candidates``
    # edits FROM ONE PARENT, ranks them, and softmax-samples one.  With the flag
    # off ``scorer`` is None, no branch below is entered, and the rng is drawn in
    # exactly the sequence it was drawn before the knob existed.
    scorer, policy_mode = _policy_prior(cfg)
    acq = getattr(cfg, "acquisition", None)
    policy_floor = float(getattr(acq, "policy_prior_random_floor", 0.20))
    policy_n = max(1, int(getattr(acq, "policy_prior_candidates", 16)))
    policy_tau = float(getattr(acq, "policy_prior_temperature", 0.25))

    elite_made = 0
    attempts = 0
    max_attempts = max(50, n_elite * 40)
    per_parent_tries = max(4, int(round(1.0 / max(search.elite_frac, 0.05))) + 2)
    while parents and elite_made < n_elite and attempts < max_attempts:
        progressed = False
        for parent_rid, parent, moves in parents:
            if elite_made >= n_elite:
                break
            # POLICY-SCORED SLOT.  The floor draw comes first so a floor slot
            # costs nothing; scored slots then propose a batch, and if the batch
            # yields no novel board the code falls through to the unscored path
            # below rather than burning the parent's turn.
            if scorer is not None and rng.random() >= policy_floor:
                scored: list[tuple[GeneralOrbitGenome, Pattern, str]] = []
                for _ in range(policy_n):
                    attempts += 1
                    try:
                        child = mutate(
                            parent, rng, moves, feed_move_prob=0.0,
                            batches=ctx.batches,
                        )
                        pattern = child.to_pattern()
                    except GenomeError:
                        continue
                    rid = candidate_record_id(pattern, ctx)
                    if rid in ledger_ids or rid in seen:
                        continue
                    scored.append((child, pattern, rid))
                if scored:
                    child, pattern, rid = scored[_policy_pick(
                        scorer, policy_mode, ctx, parent, scored, rng, policy_tau)]
                    if _admit(Candidate(pattern, child, "elite", parent_rid, rid,
                                        ctx.e_core)):
                        elite_made += 1
                        progressed = True
                    continue
            # feed_move_prob=0: a fixed-case campaign never drifts off its feed
            # grid point (feed_range/free modes are deferred, plan sec. 6.2).
            for _ in range(per_parent_tries):
                attempts += 1
                try:
                    child = mutate(
                        parent, rng, moves, feed_move_prob=0.0, batches=ctx.batches
                    )
                    pattern = child.to_pattern()
                except GenomeError:
                    continue
                rid = candidate_record_id(pattern, ctx)
                if _admit(Candidate(pattern, child, "elite", parent_rid, rid, ctx.e_core)):
                    elite_made += 1
                    progressed = True
                    break
        if not progressed:                     # every parent exhausted its novel
            break                              # 1-move neighbourhood this round

    # -- 2. rollout-scored guided construction ------------------------------ #
    beam = max(1, int(search.beam_width))
    k = max(1, int(search.completions_per_prefix))
    attempts = 0
    max_attempts = max(50, n_guided * 20)
    while sum(c.origin == "guided" for c in pool) < n_guided and attempts < max_attempts:
        attempts += 1
        prefix = _guided_prefix(ctx, parents, rng)
        if prefix is None:
            break
        completions: list[tuple[GeneralOrbitGenome, Pattern, str]] = []
        for _ in range(k):
            try:
                child = mutate(
                    prefix, rng, rng.randint(1, 3), feed_move_prob=0.0,
                    batches=ctx.batches,
                )
                pattern = child.to_pattern()
            except GenomeError:
                continue
            rid = candidate_record_id(pattern, ctx)
            if rid in ledger_ids or rid in seen:
                continue
            completions.append((child, pattern, rid))
        if not completions:
            continue
        best = _score_completions(model, ctx, completions, beam)
        for child, pattern, rid in best:
            _admit(Candidate(pattern, child, "guided", None, rid, ctx.e_core))

    # -- 3. diversity (heuristic + random) ---------------------------------- #
    attempts = 0
    max_attempts = max(50, n_div * 40)
    while sum(c.origin in ("heuristic", "random") for c in pool) < n_div and attempts < max_attempts:
        attempts += 1
        if rng.random() < 0.5:
            try:
                genome = _heuristic_genome(ctx, rng)
                origin = "heuristic"
            except GenomeError:
                continue
        else:
            try:
                genome = random_genome(
                    rng, ctx.pair, ctx.n_fresh,
                    max_shuffle_depth=ctx.max_shuffle_depth,
                    allow_single_cycle_discharge=ctx.allow_discharge,
                )
                origin = "random"
            except GenomeError:
                continue
        pattern = genome.to_pattern()
        rid = candidate_record_id(pattern, ctx)
        _admit(Candidate(pattern, genome, origin, None, rid, ctx.e_core))

    # -- top-up: if elites/guided starved (wave 0, empty store), fill random. #
    attempts = 0
    max_attempts = max(200, target * 40)
    while len(pool) < target and attempts < max_attempts:
        attempts += 1
        try:
            genome = random_genome(
                rng, ctx.pair, ctx.n_fresh,
                max_shuffle_depth=ctx.max_shuffle_depth,
                allow_single_cycle_discharge=ctx.allow_discharge,
            )
        except GenomeError:
            break
        pattern = genome.to_pattern()
        rid = candidate_record_id(pattern, ctx)
        _admit(Candidate(pattern, genome, "random", None, rid, ctx.e_core))

    return pool


def _guided_prefix(
    ctx: CaseContext,
    parents: Sequence[tuple[str | None, GeneralOrbitGenome, int]],
    rng: random.Random,
) -> GeneralOrbitGenome | None:
    """A seed genome to greedily complete (an elite parent, else heuristic)."""

    if parents and rng.random() < 0.5:
        return rng.choice(parents)[1]
    try:
        return _heuristic_genome(ctx, rng)
    except GenomeError:
        try:
            return random_genome(
                rng, ctx.pair, ctx.n_fresh,
                max_shuffle_depth=ctx.max_shuffle_depth,
                allow_single_cycle_discharge=ctx.allow_discharge,
            )
        except GenomeError:
            return None


def _score_completions(
    model: Any,
    ctx: CaseContext,
    completions: list[tuple[GeneralOrbitGenome, Pattern, str]],
    beam: int,
) -> list[tuple[GeneralOrbitGenome, Pattern, str]]:
    """Score COMPLETE patterns with the surrogate; keep the beam-best.

    The score is a cheap feasibility proxy: predicted cyclen near the target and
    low F_r/F_q/CBC.  When no model is available every completion is kept up to
    the beam (guided degrades to structural diversity, plan sec. 4.6).
    """

    if model is None or len(completions) <= beam:
        return completions[:beam]
    patterns = [pat for _, pat, _ in completions]
    try:
        pred = model.predict(patterns, ctx.case_key, ctx.e_core or 0.0)
    except Exception:  # noqa: BLE001 — surrogate must never abort construction
        return completions[:beam]
    mean = np.asarray(pred.mean, dtype=float)
    # Lower is better: F_r + F_q margin proxies; keep beam with the lowest.
    score = np.nan_to_num(mean[:, 0], nan=5.0) + np.nan_to_num(mean[:, 2], nan=5.0)
    order = np.argsort(score)[:beam]
    return [completions[int(i)] for i in order]


# --------------------------------------------------------------------------- #
# user_criteria FREE-SEARCH: pair universe + e_core-band candidate screen
# (plan sec. 12.5).  The outer decision variable is the fuel PAIR (+ split); the
# universe is the set of pairs whose ACHIEVABLE core-average enrichment interval
# overlaps the target band — the 2_LP feasible-pair lesson (exclude unreachable
# pairs from the denominator so budget is never spent where e_core can't land).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairCell:
    """One candidate (pair, split) cell of the ``user_criteria`` universe.

    ``e_lo`` / ``e_hi`` bound the mass-weighted core-average enrichment reachable
    by varying the batch split over ``split_range`` (a monotone interval — the
    endpoints are the split extremes; a mono ``A==B`` pair collapses to a point).
    ``included`` is set iff that interval overlaps ``e_core_target +/- tol``;
    ``reason`` records WHY an excluded pair was dropped (report provenance).
    """

    pair: str
    type_a: str
    type_b: str
    e_lo: float
    e_hi: float
    mono: bool
    included: bool
    reason: str = ""


def achievable_e_core_interval(
    fuel: Any, library_id: str, type_a: str, type_b: str,
    split_range: Sequence[float] = (0.2, 0.8),
) -> tuple[float, float] | None:
    """Return ``(min, max)`` mass-weighted ``e_core`` over ``split in split_range``.

    ``pair_e_core`` is monotone in the split fraction, so the interval endpoints
    are exactly the ``e_core`` at the two split extremes.  Returns ``None`` when a
    type is unresolvable or lacks an enrichment (caller marks the pair excluded).
    """

    lo_s, hi_s = float(split_range[0]), float(split_range[1])
    try:
        v0 = float(fuel.pair_e_core(type_a, type_b, lo_s, library_id))
        v1 = float(fuel.pair_e_core(type_a, type_b, hi_s, library_id))
    except (KeyError, ValueError, ZeroDivisionError, TypeError):
        return None
    if not (math.isfinite(v0) and math.isfinite(v1)):
        return None
    return (min(v0, v1), max(v0, v1))


def build_pair_universe(
    fuel: Any,
    library_id: str,
    e_core_target: float,
    e_core_tol: float,
    *,
    split_range: Sequence[float] = (0.2, 0.8),
    allow_mono: bool = True,
    types: Sequence[str] | None = None,
) -> list[PairCell]:
    """Enumerate every candidate pair of ``library_id`` and mark reachability.

    Returns ALL enumerated cells (both included and excluded) so the caller can
    report the universe size and the excluded-pair reasons.  ``A==B`` mono pairs
    are enumerated only when ``allow_mono`` (a mono is included iff its single
    enrichment lands in band).  Cross-anchor pairs are enumerated the same as
    within-anchor pairs — the per-pair asset resolver's fallback ladder supplies
    a restart for any pair (plan sec. 12.5 fixed-point identity result).
    """

    if fuel is None:
        return []
    roster = list(types) if types is not None else list(fuel.types(library_id))
    roster = sorted(set(str(t) for t in roster))
    lo_band = float(e_core_target) - float(e_core_tol)
    hi_band = float(e_core_target) + float(e_core_tol)

    cells: list[PairCell] = []
    for i, a in enumerate(roster):
        for b in roster[i:]:
            mono = a == b
            if mono and not allow_mono:
                continue
            pair = f"{a}_{b}"
            iv = achievable_e_core_interval(fuel, library_id, a, b, split_range)
            if iv is None:
                cells.append(PairCell(pair, a, b, float("nan"), float("nan"),
                                      mono, False, "unresolved enrichment"))
                continue
            e_lo, e_hi = iv
            overlap = (e_lo <= hi_band) and (e_hi >= lo_band)
            if overlap:
                cells.append(PairCell(pair, a, b, e_lo, e_hi, mono, True, ""))
            else:
                where = "above" if e_lo > hi_band else "below"
                reason = (
                    f"reach [{e_lo:.3f},{e_hi:.3f}] {where} band "
                    f"[{lo_band:.3f},{hi_band:.3f}]"
                )
                cells.append(PairCell(pair, a, b, e_lo, e_hi, mono, False, reason))
    return cells


def predicted_e_core(pattern: Pattern, fuel: Any, library_id: str) -> float | None:
    """Core-average enrichment of a pattern's fresh feed (shared extraction recipe).

    Delegates to :func:`lpopt.data.fuel_types.core_enrichment_split` — the single
    recipe used at extraction and inference, so a screened prediction equals the
    stored/served ``e_core``.  ``None`` when the feed's types are unresolvable.
    """

    if fuel is None:
        return None
    e_core, _ = core_enrichment_split(fuel, library_id, pattern.batch_feed())
    return e_core


def e_core_in_band(e_core: float | None, target: float, tol: float) -> bool:
    """True iff ``e_core`` is finite and within ``[target-tol, target+tol]``."""

    return (
        e_core is not None
        and math.isfinite(e_core)
        and (float(target) - float(tol)) <= float(e_core) <= (float(target) + float(tol))
    )


def screen_e_core_band(
    patterns: Sequence[Pattern], fuel: Any, library_id: str,
    target: float, tol: float,
) -> np.ndarray:
    """Boolean mask: which patterns' predicted ``e_core`` lands in the target band.

    This is the split-as-inner-variable screen applied to candidate validation
    (plan sec. 12.5 item 2): the genome's batch moves move the split, and this
    screen keeps only patterns whose fresh-composition ``e_core`` (from the shared
    :func:`predicted_e_core` recipe) sits in band.  When ``fuel`` is unavailable
    the screen is permissive (all True) so a missing table never empties the pool.
    """

    patterns = list(patterns)
    if fuel is None:
        return np.ones(len(patterns), dtype=bool)
    mask = np.empty(len(patterns), dtype=bool)
    for i, pat in enumerate(patterns):
        mask[i] = e_core_in_band(predicted_e_core(pat, fuel, library_id), target, tol)
    return mask


__all__ = [
    "CAMPAIGN_DECK_KNOBS", "CaseContext", "Candidate", "PairCell",
    "achievable_e_core_interval", "build_pair_universe", "build_pool",
    "candidate_record_id", "e_core_in_band", "predicted_e_core",
    "screen_e_core_band",
]
