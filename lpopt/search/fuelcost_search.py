"""Outer cell-race search for the minimum-fuel-cost configuration (plan sec. 6.2
mode (c) "free", in miniature; user directive 2026-07-21).

GOAL: find the (e_core band, feed) cell + loading pattern that MINIMIZES the fresh
fuel-economics metric ``FE`` (total fresh U-235 charge) subject to the six hard
constraints (cyclen ∈ [615, 635] EFPD, F_r ≤ 1.55, F_q ≤ 2.41, CBC ≤ 1550 ppm,
|AO| ≤ 0.30, predicted max pin burnup ≤ 80 GWd/MTU).

WHY AN OUTER RACE.  ``FE`` is *position-invariant*: it depends only on a cell's
fresh-assembly composition, not the LP layout, so within one (pair, feed) cell
every LP shares the same FE and only F_r / the constraints vary.  For the ga80
count-weighted metric the per-cell prior is EXACT and cheap:

    FE_prior(cell) = feed × e_core(pair, split)          (grams-scale for a
                                                          u_mass-bearing library
                                                          is a constant factor)

So the outer problem is: **sweep cells in ascending FE_prior and stop at the
lowest-FE cell that yields a constraint-feasible LP** — every higher-FE cell is
then FE-dominated and eliminated without spending a MASTER call on it.  Racing
(mini-waves allocated UCB-style over the still-unproven lower-FE cells) only
decides how to *spend* verification budget among the cells that could still beat
the incumbent; the elimination itself is deterministic because FE is.

DESIGN: this is an ORCHESTRATOR over :class:`CampaignDriver` — each surviving cell
is a fixed-(pair, feed) ``min_fuel_cost`` campaign (the additive objective mode).
The heavy MASTER lifting, wave gate, store merge, and reporting are all the
audited CampaignDriver path; this module only enumerates cells, pre-ranks them
from the store, screens the top cells with the champion for free, and allocates
verification mini-waves with FE-dominance elimination.  Every pure helper is unit
tested; the live race is exercised on PC2/PC3 in Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _replace
import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..data.fuel_types import fresh_fuel_charge
from .construct import (
    CaseContext,
    build_pair_universe,
)


# --------------------------------------------------------------------------- #
# cell model
# --------------------------------------------------------------------------- #
@dataclass
class FuelCostCell:
    """One (pair, feed) cell of the min_fuel_cost outer search.

    ``fe_prior`` is the a-priori fuel-economics metric (``feed × e_core`` for the
    ga80 count-weighted proxy; ``feed × e_core × ū_mass/100`` grams-U-235 for a
    mass-bearing library).  ``band_ok`` is the store-empirical cyclen-band screen:
    True/False when the cell has converged store rows whose median cyclen lands
    in / out of [cyclen_lo, cyclen_hi], ``None`` when the store has no evidence
    (neutral — never eliminated on absence of data).  ``store_*`` summarize the
    matching converged store rows (the empirical prior that pre-ranks the cell).
    """

    pair: str
    type_a: str
    type_b: str
    feed: int
    e_core: float
    fe_prior: float
    store_n: int = 0
    store_cyclen_median: float | None = None
    store_fr_median: float | None = None
    band_ok: bool | None = None
    screen_value: float = float("-inf")     # wave-0 champion-prediction best total
    screen_feasible: bool = False           # any predicted-feasible LP in the screen
    eliminated: bool = False
    reason: str = ""

    @property
    def cell_id(self) -> str:
        return f"{self.pair}_f{self.feed}"


# --------------------------------------------------------------------------- #
# feed grid + FE prior
# --------------------------------------------------------------------------- #
def feed_grid(lo: int, hi: int) -> list[int]:
    """The ``1 + 4N`` feed grid within ``[lo, hi]`` (inclusive), ascending."""
    feeds = [f for f in range(int(lo), int(hi) + 1) if (f - 1) % 4 == 0]
    return feeds


def cell_fe_prior(feed: int, e_core: float, u_mass_mean: float | None = None) -> float:
    """A-priori fuel-economics metric of a cell.

    Count-weighted (ga80, ``u_mass_mean`` None): ``feed × e_core`` [pos·w/o] — the
    exact value :func:`lpopt.data.fuel_types.fresh_fuel_charge` returns for a feed
    whose fresh positions average ``e_core`` (Σ count·e = feed × mean-e).  When a
    representative per-assembly ``u_mass_mean`` [g] is supplied it becomes the true
    grams-U-235 charge ``feed × e_core × u_mass_mean / 100`` (one consistent scale
    only if EVERY cell in the run supplies u_mass — else pass None throughout).
    """
    base = float(feed) * float(e_core)
    if u_mass_mean is None:
        return base
    return base * float(u_mass_mean) / 100.0


def cell_fe_from_fuel(
    fuel: Any, library_id: str, batch_feed: dict[str, int]
) -> tuple[float | None, bool]:
    """Exact FE of a concrete representative composition (delegates to
    :func:`lpopt.data.fuel_types.fresh_fuel_charge`) — used when a cell has a real
    representative pattern rather than the (feed × e_core) prior."""
    return fresh_fuel_charge(fuel, library_id, batch_feed)


# --------------------------------------------------------------------------- #
# store-empirical cell statistics
# --------------------------------------------------------------------------- #
def store_cell_stats(
    store_df: Any,
    library_id: str,
    feed: int,
    e_lo: float,
    e_hi: float,
) -> tuple[int, float | None, float | None]:
    """``(n, cyclen_median, fr_median)`` over converged store rows in a cell.

    A row matches on ``library_id`` + exact ``feed`` + ``e_core ∈ [e_lo, e_hi]`` +
    ``converged``.  Returns ``(0, None, None)`` when the store has no matching
    evidence (the caller keeps the cell band-neutral).  ``store_df`` is the
    :attr:`lpopt.data.store.StoreReader.records` frame (or None).
    """
    if store_df is None or len(store_df) == 0:
        return 0, None, None
    df = store_df
    try:
        mask = (
            (df["library_id"] == library_id)
            & (df["feed"] == int(feed))
            & (df["converged"] == True)  # noqa: E712 — pandas boolean mask
            & (df["e_core"] >= float(e_lo))
            & (df["e_core"] <= float(e_hi))
        )
        sub = df[mask]
    except KeyError:
        return 0, None, None
    n = int(len(sub))
    if n == 0:
        return 0, None, None
    cy = sub["cyclen"].dropna()
    fr = sub["f_r"].dropna()
    cy_med = float(cy.median()) if len(cy) else None
    fr_med = float(fr.median()) if len(fr) else None
    return n, cy_med, fr_med


# --------------------------------------------------------------------------- #
# cell enumeration
# --------------------------------------------------------------------------- #
def restart_bearing_pairs(package_root: str | Path) -> set[str]:
    """The set of pairs that have a NATIVE MASTER restart base in the feasible
    package (``package_root/bases/<pair>[_f<feed>]/MAS_RST.*``).

    A cell whose pair is NOT in this set has no exact/same-pair restart, so the
    :class:`~lpopt.search.assets.CaseAssetResolver` would fall back to a nearest-
    e_core restart from a DIFFERENT pair — whose burnt-assembly types are absent
    from the target ga80 deck's ``%LPD_B&C`` — and MASTER dies at INITIALIZE
    ("MAS_SUM EDIT 2 header anchor not found"), silently burning the budget.  The
    outer search must therefore restrict itself to these restart-bearing pairs.
    Folder names carry an optional ``_f<feed>`` promotion suffix that is stripped
    to the base pair (``E1_E2_f117`` → ``E1_E2``).
    """
    root = Path(package_root)
    bases = root / "bases"
    pairs: set[str] = set()
    if not bases.is_dir():
        return pairs
    for d in bases.iterdir():
        if not d.is_dir():
            continue
        if not any(d.glob("MAS_RST.*")):
            continue
        name = d.name
        m = re.match(r"^(.*)_f\d+$", name)     # strip promotion feed suffix
        pairs.add(m.group(1) if m else name)
    return pairs


def enumerate_cells(
    fuel: Any,
    library_id: str,
    e_core_targets: Sequence[float],
    e_core_tol: float,
    feeds: Sequence[int],
    *,
    store_df: Any = None,
    cyclen_lo: float = 615.0,
    cyclen_hi: float = 635.0,
    split: float = 0.5,
    u_mass_mean: float | None = None,
    types: Sequence[str] | None = None,
    restart_pairs: set[str] | None = None,
) -> list[FuelCostCell]:
    """Enumerate every (pair, feed) cell of the search space.

    For each ``e_core_target`` the pair universe (:func:`build_pair_universe`)
    supplies the pairs whose achievable ``e_core`` reaches the band; each is
    crossed with the ``feeds`` grid to form cells.  A cell is annotated with its
    FE prior (``feed × e_core(pair, split)``) and the store-empirical band screen
    (median cyclen ∈ [cyclen_lo, cyclen_hi] over converged rows in its e_core
    band + feed).  Duplicate (pair, feed) cells across overlapping bands are kept
    once (lowest e_core wins — the more fuel-economical realization).

    ``restart_pairs`` (when given) is a HARD allowlist: only pairs with a native
    MASTER restart base (:func:`restart_bearing_pairs`) are enumerated, so the
    search never proposes a cell whose only restart is an incompatible cross-pair
    fallback.  ``None`` disables the filter (every in-band pair — for tests /
    dry-runs against a stub evaluator that needs no real restart).
    """
    seen: dict[str, FuelCostCell] = {}
    for tgt in e_core_targets:
        universe = build_pair_universe(
            fuel, library_id, float(tgt), float(e_core_tol),
            types=types,
        )
        for pc in universe:
            if not pc.included:
                continue
            if restart_pairs is not None and pc.pair not in restart_pairs:
                continue
            try:
                e_core = float(fuel.pair_e_core(pc.type_a, pc.type_b, split, library_id))
            except (KeyError, ValueError, ZeroDivisionError, TypeError):
                continue
            if not math.isfinite(e_core):
                continue
            for feed in feeds:
                fe = cell_fe_prior(feed, e_core, u_mass_mean)
                cid = f"{pc.pair}_f{int(feed)}"
                if cid in seen and seen[cid].e_core <= e_core:
                    continue
                n, cy_med, fr_med = store_cell_stats(
                    store_df, library_id, feed, float(tgt) - e_core_tol,
                    float(tgt) + e_core_tol,
                )
                band_ok: bool | None
                if cy_med is None:
                    band_ok = None
                else:
                    band_ok = (cyclen_lo <= cy_med <= cyclen_hi)
                seen[cid] = FuelCostCell(
                    pair=pc.pair, type_a=pc.type_a, type_b=pc.type_b,
                    feed=int(feed), e_core=e_core, fe_prior=fe,
                    store_n=n, store_cyclen_median=cy_med, store_fr_median=fr_med,
                    band_ok=band_ok,
                )
    return list(seen.values())


# --------------------------------------------------------------------------- #
# pre-ranking + FE-dominance elimination
# --------------------------------------------------------------------------- #
def dedup_by_composition(
    cells: Sequence[FuelCostCell], *, e_decimals: int = 3
) -> list[FuelCostCell]:
    """Collapse FE-equivalent pairs to one representative cell per (feed, e_core).

    Many ga80 pairs realize the *same* core-average enrichment (the letter anchors
    are per-family, so e.g. ``J1_L1 … J1_L6`` all land at one e_core) — they are
    FE-identical and behave the same in the outer race, so a "cell" in the mission
    sense is (band, feed), not (pair, feed).  The representative is the pair with
    the MOST converged store evidence (``store_n``), tie-broken by pair name, so
    the raced pair is the best-characterized restart for that (feed, e_core).
    """
    best: dict[tuple[int, float], FuelCostCell] = {}
    for c in cells:
        key = (c.feed, round(c.e_core, e_decimals))
        cur = best.get(key)
        if cur is None or (c.store_n, cur.pair) > (cur.store_n, c.pair):
            # prefer more store evidence; deterministic tie-break on pair name.
            if cur is None or c.store_n > cur.store_n or (
                c.store_n == cur.store_n and c.pair < cur.pair
            ):
                best[key] = c
    return list(best.values())


def prerank_cells(cells: Sequence[FuelCostCell]) -> list[FuelCostCell]:
    """Order cells for the race: best first.

    Key (all ascending after sign flips): (1) band feasibility class — cells whose
    store-median cyclen is IN band (band_ok True) first, then band-UNKNOWN cells
    (no store evidence, band_ok None), then band-OUT cells (band_ok False) last;
    (2) within a class, ascending FE prior (cheapest fuel first); (3) feed then
    pair for a stable deterministic tie-break.  Store-empirical cyclen feasibility
    therefore dominates the pre-rank exactly as the directive requires, with FE
    the ordering key inside each feasibility class.
    """
    def band_rank(c: FuelCostCell) -> int:
        if c.band_ok is True:
            return 0
        if c.band_ok is None:
            return 1
        return 2

    return sorted(
        cells,
        key=lambda c: (band_rank(c), c.fe_prior, c.feed, c.pair),
    )


def eliminate_dominated(
    cells: Sequence[FuelCostCell], proven_fe: float | None, *, atol: float = 1e-9
) -> int:
    """Mark every not-yet-proven cell whose FE prior is ≥ ``proven_fe`` eliminated.

    Deterministic FE-dominance: once a constraint-feasible LP is verified at fuel
    cost ``proven_fe``, no cell with an equal-or-higher FE prior can improve on it,
    so it is dropped from the race without a MASTER call.  ``proven_fe`` None (no
    feasible incumbent yet) eliminates nothing.  Returns the count newly
    eliminated.  A cell already ``eliminated`` is left untouched.
    """
    if proven_fe is None:
        return 0
    n = 0
    for c in cells:
        if c.eliminated:
            continue
        if c.fe_prior >= float(proven_fe) - float(atol):
            c.eliminated = True
            c.reason = f"FE-dominated (prior {c.fe_prior:.2f} ≥ proven {proven_fe:.2f})"
            n += 1
    return n


def race_allocation(
    cells: Sequence[FuelCostCell], slots: int, z: float = 1.0,
) -> dict[str, int]:
    """Allocate ``slots`` verification calls over the surviving cells, UCB-style.

    Lower FE is better, so the racing value of a cell is ``−FE_prior`` boosted by
    its wave-0 screen feasibility (a screen-feasible cell gets priority).  The best
    (lowest-FE) surviving cell always gets at least one slot (exploit floor); the
    remainder is spread over the other survivors by a softmax on ``−FE_prior`` so a
    close FE runner-up still gets probed before it is FE-eliminated.  Empty when no
    survivors or ``slots ≤ 0``.
    """
    survivors = [c for c in cells if not c.eliminated]
    if not survivors or slots <= 0:
        return {}
    survivors = sorted(survivors, key=lambda c: (c.fe_prior, c.feed, c.pair))
    alloc: dict[str, int] = {c.cell_id: 0 for c in survivors}
    alloc[survivors[0].cell_id] += 1
    remaining = int(slots) - 1
    if remaining > 0 and len(survivors) > 1:
        fe = np.array([c.fe_prior for c in survivors], dtype=float)
        # softmax on -FE (normalized) so magnitudes don't saturate.
        spread = float(np.std(fe)) or 1.0
        w = np.exp(-(fe - fe.min()) / spread)
        w = w / w.sum()
        counts = _largest_remainder(w, remaining)
        for c, extra in zip(survivors, counts):
            alloc[c.cell_id] += int(extra)
    elif remaining > 0:
        alloc[survivors[0].cell_id] += remaining
    return alloc


def _largest_remainder(weights: np.ndarray, total: int) -> list[int]:
    """Largest-remainder apportionment of ``total`` over ``weights`` (sum→total)."""
    if total <= 0 or weights.size == 0:
        return [0] * int(weights.size)
    exact = weights / weights.sum() * float(total)
    floor = np.floor(exact).astype(int)
    rem = total - int(floor.sum())
    if rem > 0:
        order = np.argsort(-(exact - floor))
        for i in order[:rem]:
            floor[i] += 1
    return floor.tolist()


# --------------------------------------------------------------------------- #
# outer orchestrator (reuses CampaignDriver per cell)
# --------------------------------------------------------------------------- #
@dataclass
class FuelCostSearchResult:
    """Outcome of the outer min_fuel_cost search."""

    best_cell: str | None
    best_fe: float | None
    best: dict[str, Any] | None            # the winning LP (CampaignResult.best)
    cells_screened: int
    cells_raced: int
    cells_eliminated: int
    budget_spent: int
    per_cell: list[dict[str, Any]] = field(default_factory=list)
    retrain_events: list[dict[str, Any]] = field(default_factory=list)


class FuelCostOuterSearch:
    """Drive the outer cell race for the minimum-fuel-cost configuration.

    Construction takes the same ``cfg`` / ``model`` a CampaignDriver takes plus the
    search space (``e_core_targets``, ``feeds``) and budgets.  ``driver_factory``
    builds a per-cell CampaignDriver — defaulting to the real
    :class:`~lpopt.search.campaign.CampaignDriver`; tests inject a stub so the pure
    racing logic runs without MASTER.  The cfg is cloned per cell with
    ``[case].pair`` / ``[case].feed`` set and ``[acquisition].objective =
    "min_fuel_cost"`` (existing modes untouched — additive).
    """

    def __init__(
        self,
        cfg: Any,
        model: Any,
        *,
        e_core_targets: Sequence[float],
        feeds: Sequence[int],
        e_core_tol: float = 0.125,
        screen_top_k: int = 8,
        mini_wave: int = 8,
        total_budget: int = 300,
        fuel_library: Any = None,
        store_df: Any = None,
        restart_pairs: set[str] | None = None,
        driver_factory: Callable[..., Any] | None = None,
        log: Callable[[str], None] | None = None,
        retrain_gate_callback: Callable[[int], dict | None] | None = None,
        retrain_label_threshold: int = 200,
        model_reload: Callable[[str], Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.e_core_targets = list(e_core_targets)
        self.feeds = list(feeds)
        self.e_core_tol = float(e_core_tol)
        self.screen_top_k = int(screen_top_k)
        self.mini_wave = int(mini_wave)
        self.total_budget = int(total_budget)
        self.library_id = cfg.model.library_id
        self.fuel = fuel_library
        self.store_df = store_df
        self.restart_pairs = restart_pairs
        self.driver_factory = driver_factory or self._default_driver_factory
        self._log = log or (lambda m: print(m))
        self.budget_spent = 0
        # -- active-learning retrain integration (plan 1c) ----------------- #
        # ``retrain_gate_callback(new_labels) -> gate_dict | None`` runs a FULL
        # remote retrain + the HONEST no-regression gate + gated champion
        # promotion (the audited curriculum ``remote_full`` path; champion advances
        # ONLY when the gate passes).  Fired when accumulated new labels reach
        # ``retrain_label_threshold`` OR a race round finishes.  Default None keeps
        # the search retrain-free (the labels still merge to the store for a later
        # explicit cycle — Stage 2).  ``model_reload(model_dir) -> model`` swaps in
        # a gate-promoted champion mid-search.
        self.retrain_gate_callback = retrain_gate_callback
        self.retrain_label_threshold = int(retrain_label_threshold)
        self.model_reload = model_reload
        self._new_labels_since_retrain = 0
        self.retrain_events: list[dict[str, Any]] = []

    # -- cell setup -------------------------------------------------------- #
    def build_cells(self) -> list[FuelCostCell]:
        cells = enumerate_cells(
            self.fuel, self.library_id, self.e_core_targets, self.e_core_tol,
            self.feeds, store_df=self.store_df,
            cyclen_lo=float(getattr(self.cfg.acquisition, "fuelcost_cyclen_lo", 615.0)),
            cyclen_hi=float(getattr(self.cfg.acquisition, "fuelcost_cyclen_hi", 635.0)),
            restart_pairs=self.restart_pairs,
        )
        return prerank_cells(dedup_by_composition(cells))

    def _cell_cfg(self, cell: FuelCostCell) -> Any:
        """Clone ``cfg`` for one cell: fixed (pair, feed) + min_fuel_cost objective."""
        case = _replace(self.cfg.case, mode="fixed", pair=cell.pair, feed=cell.feed)
        acq = _replace(self.cfg.acquisition, objective="min_fuel_cost")
        return _replace(self.cfg, case=case, acquisition=acq)

    def _default_driver_factory(self, cell: FuelCostCell, budget: int, run_dir: Path) -> Any:
        from .campaign import CampaignDriver
        return CampaignDriver(
            self._cell_cfg(cell), self.model,
            budget=budget, run_dir=run_dir, resume=True,
            fuel_library=self.fuel, progress=False, log=self._log,
        )

    # -- wave-0 champion-prediction screen (free) -------------------------- #
    def screen(self, cells: Sequence[FuelCostCell]) -> list[FuelCostCell]:
        """Score the top ``screen_top_k`` pre-ranked cells with the champion (no
        verify): record each cell's best predicted min_fuel_cost total + whether any
        LP is predicted feasible.  Free — surrogate predictions only."""
        from . import acquisition as acq
        from .campaign import build_pool
        import random as _random

        top = list(cells)[: self.screen_top_k]
        spec = acq.MinFuelCostSpec(
            lam_fr=float(getattr(self.cfg.acquisition, "fuelcost_lambda_fr", 20.0)),
            risk_z=float(self.cfg.acquisition.risk_z),
            cyclen_lo=float(getattr(self.cfg.acquisition, "fuelcost_cyclen_lo", 615.0)),
            cyclen_hi=float(getattr(self.cfg.acquisition, "fuelcost_cyclen_hi", 635.0)),
            f_r_limit=float(self.cfg.acquisition.f_r_limit),
            cbc_limit=float(self.cfg.acquisition.cbc_limit),
            f_q_limit=float(self.cfg.acquisition.f_q_limit),
            ao_abs_limit=float(self.cfg.acquisition.ao_abs_limit),
            pin_bu_limit=float(getattr(self.cfg.acquisition, "fuelcost_pin_bu_limit", 80.0)),
        )
        rng = _random.Random(int(getattr(self.cfg.flow, "random_seed", 0)))
        for cell in top:
            ctx = CaseContext(pair=cell.pair, feed=cell.feed,
                              library_id=self.library_id, e_core=cell.e_core)
            pool = build_pool(ctx, self.model, [], set(), rng, self._cell_cfg(cell),
                              wave_index=0, size=int(self.cfg.search.dry_run_pool_size))
            if not pool:
                continue
            # Champion predictions only (region gating is irrelevant to a per-cell
            # screen — the cell IS the campaign's fixed bin): score FE + all six
            # constraints directly.  Free.
            prediction = self.model.predict(
                [c.pattern for c in pool], ctx.case_key, ctx.e_core or 0.0)
            fe, _ = acq.fuel_charge_array(self.fuel, self.library_id, pool)
            fc = acq.score_min_fuel_cost(prediction, spec, fe)
            finite = fc.total[np.isfinite(fc.total)]
            cell.screen_value = float(np.max(finite)) if finite.size else float("-inf")
            cell.screen_feasible = bool(np.any(fc.constraint_ok))
        return top

    # -- the race ---------------------------------------------------------- #
    def run(self, run_root: str | Path) -> FuelCostSearchResult:
        """Race the cells: pre-rank → screen → verify mini-waves with FE-dominance
        elimination → converge on the lowest-FE feasible cell.  Reuses one
        resumable CampaignDriver per cell so budget accumulates across rounds."""
        run_root = Path(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        cells = self.build_cells()
        self._log(f"[fuelcost] enumerated {len(cells)} cells; "
                  f"FE prior range [{min((c.fe_prior for c in cells), default=float('nan')):.1f}, "
                  f"{max((c.fe_prior for c in cells), default=float('nan')):.1f}]")
        self.screen(cells)
        best_fe: float | None = None
        best_cell_id: str | None = None
        best: dict[str, Any] | None = None
        raced = 0
        #: cumulative budget GRANTED to a cell's resumable mini-campaign, and the
        #: cumulative REAL calls it has actually spent (``CampaignResult.budget_spent``
        #: is cumulative for a resumed run, so a round's real cost is the DELTA).
        cell_budget: dict[str, int] = {}
        cell_spent: dict[str, int] = {}

        # Round-robin deepening: each round gives the top-K surviving cells one more
        # wave of budget (EXTENDING their resumable campaign so the CampaignDriver
        # runs NEW exploit waves — driving F_r down within the cell), and charges the
        # global budget ONLY the real new labels (the delta).  A cell that adds no
        # new calls (early-stopped / no-improve) is exhausted → eliminated.  A whole
        # round with zero new calls ends the search (no phantom budget — forensic
        # 20260721: the old race charged allocated slots for no-op resumes).
        while self.budget_spent < self.total_budget:
            eliminate_dominated(cells, best_fe)
            survivors = [c for c in prerank_cells(cells) if not c.eliminated]
            if not survivors:
                break
            active = survivors[: max(1, self.screen_top_k)]
            progressed = False
            eliminated_this_round = 0
            for cell in active:
                if self.budget_spent >= self.total_budget:
                    break
                cid = cell.cell_id
                wave = min(self.mini_wave, self.total_budget - self.budget_spent)
                if wave <= 0:
                    break
                granted = cell_budget.get(cid, 0) + wave
                cell_budget[cid] = granted
                run_dir = run_root / f"alsearch_{cid}_minFE"
                try:
                    driver = self.driver_factory(cell, granted, run_dir)
                    result = driver.run()
                except Exception as exc:  # noqa: BLE001 — a cell that cannot resolve
                    # a compatible restart (AssetResolutionError) or otherwise fails
                    # to build must not crash the race — eliminate it and move on so
                    # no budget is spent on an unverifiable cell (forensic 20260721).
                    cell.eliminated = True
                    cell.reason = f"driver error: {type(exc).__name__}: {exc}"
                    self._log(f"[fuelcost] cell {cid} skipped ({cell.reason})")
                    continue
                cum = int(getattr(result, "budget_spent", 0) or 0)
                delta = max(0, cum - cell_spent.get(cid, 0))     # REAL new calls
                cell_spent[cid] = cum
                self.budget_spent += delta
                self._new_labels_since_retrain += delta
                if delta > 0:
                    progressed = True
                    raced += 1
                cell_best = getattr(result, "best", None)
                if cell_best is not None:
                    fe = cell_best.get("fuel_cost", cell.fe_prior)
                    fe = float(fe) if fe is not None else cell.fe_prior
                    if best_fe is None or fe < best_fe:
                        best_fe, best_cell_id, best = fe, cid, cell_best
                        self._log(f"[fuelcost] new incumbent: cell {cid} FE={fe:.2f} "
                                  f"F_r={cell_best.get('f_r')} cyclen={cell_best.get('cyclen')}")
                if delta == 0:
                    # The mini-campaign added no new labels for its extended budget —
                    # it has early-stopped / exhausted its within-cell exploit.  Drop
                    # it so the race deepens the NEXT-ranked cell (never re-charges a
                    # no-op) — the lower-ranked survivors move into ``active`` on the
                    # following round.
                    cell.eliminated = True
                    eliminated_this_round += 1
                    cell.reason = (f"exhausted at {cell_spent[cid]} calls "
                                   f"(early-stop / no F_r improvement)")
                    self._log(f"[fuelcost] cell {cid} {cell.reason}")
                eliminate_dominated(cells, best_fe)
            self._maybe_retrain(round_finished=True)
            # Terminate only when a full round neither spent a real call NOR removed a
            # cell — otherwise keep going so lower-ranked survivors get their waves
            # (each round strictly shrinks the problem: +budget or −a cell).
            if not progressed and eliminated_this_round == 0:
                break

        return self._result(cells, best_cell_id, best_fe, best, raced)

    def _maybe_retrain(self, *, round_finished: bool) -> None:
        """Fire the full-retrain + honest-gate cycle when the label budget or a
        round boundary is reached (plan 1c).  The gate table is always logged; the
        champion is swapped in ONLY when the callback reports a passing gate."""
        if self.retrain_gate_callback is None:
            return
        due = (self._new_labels_since_retrain >= self.retrain_label_threshold
               or round_finished)
        if not due or self._new_labels_since_retrain <= 0:
            return
        n = self._new_labels_since_retrain
        self._log(f"[fuelcost][AL] retrain trigger: {n} new labels "
                  f"(threshold {self.retrain_label_threshold}, round_finished={round_finished})")
        gate = self.retrain_gate_callback(n)
        self._new_labels_since_retrain = 0
        if not gate:
            return
        passed = bool(gate.get("pass"))
        self.retrain_events.append({"new_labels": n, **gate})
        self._log(f"[fuelcost][AL] gate: pass={passed} worst_drop={gate.get('worst_drop')} "
                  f"epsilon={gate.get('epsilon')} checks={len(gate.get('checks', []))}")
        if passed and gate.get("champion_model_dir") and self.model_reload is not None:
            self.model = self.model_reload(str(gate["champion_model_dir"]))
            self._log(f"[fuelcost][AL] champion promoted -> {gate['champion_model_dir']}")

    def _result(self, cells, best_cell_id, best_fe, best, raced) -> "FuelCostSearchResult":
        eliminated = sum(1 for c in cells if c.eliminated)
        return FuelCostSearchResult(
            best_cell=best_cell_id, best_fe=best_fe, best=best,
            cells_screened=min(self.screen_top_k, len(cells)),
            cells_raced=raced,
            cells_eliminated=eliminated,
            budget_spent=self.budget_spent,
            per_cell=[{
                "cell": c.cell_id, "feed": c.feed, "e_core": round(c.e_core, 4),
                "fe_prior": round(c.fe_prior, 2), "band_ok": c.band_ok,
                "store_n": c.store_n, "store_cyclen_median": c.store_cyclen_median,
                "screen_feasible": c.screen_feasible, "eliminated": c.eliminated,
                "reason": c.reason,
            } for c in prerank_cells(cells)],
            retrain_events=list(self.retrain_events),
        )


__all__ = [
    "FuelCostCell",
    "FuelCostSearchResult",
    "FuelCostOuterSearch",
    "feed_grid",
    "cell_fe_prior",
    "cell_fe_from_fuel",
    "store_cell_stats",
    "restart_bearing_pairs",
    "enumerate_cells",
    "dedup_by_composition",
    "prerank_cells",
    "eliminate_dominated",
    "race_allocation",
]
