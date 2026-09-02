"""Outer cell-race over the (pair, feed) design grid (user directive 2026-07-22,
rev 2; flatness-first program 20260725 §5 / §10 / decisions D5 + D7).

TWO MODES, TWO ALLOCATION RULES — the same orchestration skeleton, deliberately
different budget rules, because they are different problems:

``fr_boundary`` — **DESIGN-SPACE MAPPING, not a core-loading objective.**  It
answers "which design cells can reach F_r <= 1.55 at all", which is a question
about the design space, not about any particular loading pattern, and which no
surrogate improvement can answer.  RE-AIMED (decision D5) at the 16 cells program
§5 shows empirically hold compliant rows — ``e_core 5.0-5.5 x feed {117,121,125}``
minus 5.4_f117 / 5.5_f117.  The draft proposed deleting this mode on the grounds
that F_r <= 1.55 is unreachable; that was a POOLED-statistic error (pooled
P = 0.0147 hiding per-cell rates up to 0.722), so the mode is kept and retargeted
instead.  Budget is allocated by F_r proximity to 1.55, which is legitimate here
BECAUSE F_r is this mode's objective.

``flat_power`` — the flatness-first production race over the full 24-cell roster.
Budget is allocated by PURE MAP-COVERAGE DEFICIT (:func:`coverage_weights`,
decision D7).  The F_r proximity rule is RETIRED for this mode (program §10 STOP:
"the outer budget has been allocated by F_r independently of the inner objective —
this is the part that was genuinely broken").

GOAL (both): accumulate MASTER-labeled loading patterns across the roster,
biased toward constraint-feasible rows (F_q<=2.41, CBC<=1550, |AO|<=0.30,
pin BU<=80).  Cycle length is recorded but NEVER gated.

WHY A NEW RACE (not FuelCostOuterSearch reused verbatim).  The fuel-cost race is an
ELIMINATION tournament: FE is position-invariant, so once a feasible LP is proven
every higher-FE cell is dropped without a MASTER call.  The boundary campaign is the
OPPOSITE — every cell is a TARGET (coverage matters), nothing is eliminated on
domination, and the budget is spent to DEEPEN the cells nearest F_r=1.55.  So this
module keeps the fuel-cost orchestration skeleton (resumable per-cell CampaignDriver,
stable run_dir naming, retrain-gate / champion-swap hooks) but REWRITES the budget /
state model for a ONE-PROCESS-PER-ROUND execution model:

* **Persistent cross-invocation state.**  ``race_state.json`` (atomic tmp+os.replace)
  holds the round index, cumulative grants, per-cell verified best F_r, d8 probe
  strike counters, transient-failure counters and exclusion records.  INDEPENDENTLY
  — so a lost race_state cannot cause phantom charging — the race seeds each cell's
  spent from that cell's own ``state.json`` ``budget_spent`` at construction.
* **True budget EXTENSION.**  Grant = ``persisted_spent + round_weight`` (matching
  CampaignDriver's ``while budget_spent < budget`` resume semantics); the round
  charges only ``max(0, state.json_after - persisted_before)`` — new MASTER calls.
* **Exception handling.**  On a driver crash the cell's real pre-crash delta is still
  charged from state.json.  ``AssetResolutionError`` / ``RestartFallbackError`` are
  STRUCTURAL (permanent exclusion); any other exception is TRANSIENT (cell retained,
  excluded only after 2 consecutive transient-failure rounds).

Every pure helper (roster, seeds, weights, accounting) is unit tested; the live race
runs one round per invocation via the ``lpopt frontier-produce`` CLI on PC2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..data.compliance import assert_mono_anchor, is_cross_anchor
from ..safelog import safe_logger


# --------------------------------------------------------------------------- #
# roster
# --------------------------------------------------------------------------- #
#: e_core band per (mono-anchor) family pair — one pair per band (Decision 1).
_PAIR_ECORE: dict[str, float] = {
    "E1_E2": 5.0, "J1_J2": 5.1, "K1_K2": 5.2,
    "L1_L2": 5.3, "N1_N2": 5.4, "G3_G4": 5.5,
}
#: The low-e_core group (<=5.2) whose f113 resolves at level-2 delta-4 (native f117)
#: and whose f117 is a level-0 native; the high group (>=5.3) has no native f117 so
#: its f113 falls to a delta-8 PROBE and its f117 to a delta-4 restart.
_LOW_GROUP = frozenset({"E1_E2", "J1_J2", "K1_K2"})
_FEEDS = (113, 117, 121, 125)

#: Round-1 base-quality weights per cell class + the d8 probe wave size.
NATIVE_WEIGHT = 16
D4_WEIGHT = 10
PROBE_WEIGHT = 4
PROBE_SIZE = 4
ROUND_BUDGET = 276
FLOOR_WEIGHT = 4
#: proximity target and shaping constant for round>=2 weights.
#:
#: RETIRED for the ``flat_power`` objective (program §10 STOP / decision D7): the
#: outer budget must not be steered by F_r when the inner objective is flatness.
#: They remain live for ``fr_boundary``, whose objective IS the F_r boundary.
FR_TARGET = 1.55
PROX_EPS = 0.05
#: a d8 probe cell that produces no converged row for this many ROUNDS is excluded.
D8_STRIKE_LIMIT = 2
#: consecutive transient (non-structural) failing rounds before a cell is excluded.
TRANSIENT_LIMIT = 2
#: exception class names treated as STRUCTURAL (permanent exclusion).
_STRUCTURAL_EXC = frozenset({"AssetResolutionError", "RestartFallbackError"})

#: Mapped rows a cell needs before it stops being a coverage TARGET (decision D7).
#: Sized off the labelled corpus: a per-cell within-cell statistic needs ~8 rows to
#: exist at all (``flat_scale.MIN_CELL_ROWS``) and the licensing-corridor cells of
#: program §5 currently hold 9-40 maps each, so 60 keeps every corridor cell in
#: deficit while letting a genuinely covered cell fall back to the floor.
COVERAGE_TARGET = 60

#: e_core x feed cells that EMPIRICALLY contain F_r <= 1.55 rows (program §5, the
#: 16 cells with n >= 20).  ``fr_boundary`` is re-aimed at exactly these
#: (decision D5).  Keyed by (e_core, feed) because that is how §5 reports them;
#: :func:`build_roster` maps them onto the pair roster.
#:
#: The two (e_core, feed) combinations MISSING from this set inside the
#: 5.0-5.5 x {117,121,125} block are 5.4_f117 and 5.5_f117 — they hold no
#: compliant row in the store, so they are not design-space targets.
FR_COMPLIANT_CELLS: frozenset[tuple[float, int]] = frozenset({
    (5.0, 117), (5.0, 121), (5.0, 125),
    (5.1, 117), (5.1, 121), (5.1, 125),
    (5.2, 117), (5.2, 121), (5.2, 125),
    (5.3, 117), (5.3, 121), (5.3, 125),
    (5.4, 121), (5.4, 125),
    (5.5, 121), (5.5, 125),
})

#: Outer-race objectives this module understands.  Anything else is clamped to
#: ``fr_boundary`` by the CLI, so it is clamped here too rather than silently
#: allocating with the wrong rule.
_RACE_OBJECTIVES = ("fr_boundary", "flat_power", "min_fuel_cost")


def _cell_class(pair: str, feed: int) -> str:
    """The restart class of a (pair, feed) cell: ``native`` | ``d4`` | ``d8_probe``."""
    if feed == 121:
        return "native"
    if feed == 117:
        return "native" if pair in _LOW_GROUP else "d4"
    if feed == 113:
        return "d4" if pair in _LOW_GROUP else "d8_probe"
    # feed == 125 -> delta-4 from native f121 in every band.
    return "d4"


def _base_status(pair: str, feed: int, klass: str) -> str:
    if klass == "native":
        return f"level0 native f{feed}"
    if klass == "d8_probe":
        return ("level2 same-pair f121 (delta 8) - PROBE ONLY (3 exploit + 1 control), "
                "non_finite_flux risk")
    # d4: f113 restarts from f117 in the low group, everything else from f121.
    src = 117 if (feed == 113 and pair in _LOW_GROUP) else 121
    return f"level2 same-pair f{src} (delta 4)"


@dataclass
class FrontierCell:
    """One (pair, feed) roster cell of the fr_boundary campaign."""

    pair: str
    feed: int
    e_core: float
    klass: str                       # native | d4 | d8_probe
    base_status: str
    excluded: bool = False
    reason: str = ""

    @property
    def cell_id(self) -> str:
        return f"{self.pair}_f{self.feed}"

    @property
    def is_probe(self) -> bool:
        return self.klass == "d8_probe"


def is_compliant_cell(e_core: float, feed: int) -> bool:
    """True when (e_core, feed) empirically contains F_r <= 1.55 rows (§5)."""
    return (round(float(e_core), 1), int(feed)) in FR_COMPLIANT_CELLS


def build_roster(*, compliant_only: bool = False) -> list[FrontierCell]:
    """The FIXED 24-cell roster (Decision 1).  Hard-fails (R1) on any cross-anchor
    pair — every roster pair is a mono-anchor same-family pair by construction, so
    this is a structural guarantee, not a hope.

    ``compliant_only`` (decision D5) restricts the roster to the 16 cells of
    program §5 that empirically hold F_r <= 1.55 rows — the DESIGN-SPACE MAPPING
    roster for ``fr_boundary``.  See :class:`FrBoundaryOuterRace` for what that
    mode is and is not.
    """
    assert_mono_anchor(_PAIR_ECORE.keys())
    cells: list[FrontierCell] = []
    for pair, e_core in _PAIR_ECORE.items():
        for feed in _FEEDS:
            if compliant_only and not is_compliant_cell(e_core, feed):
                continue
            klass = _cell_class(pair, feed)
            cells.append(FrontierCell(
                pair=pair, feed=feed, e_core=e_core, klass=klass,
                base_status=_base_status(pair, feed, klass),
            ))
    return cells


def round1_weights(cells: Sequence[FrontierCell]) -> dict[str, int]:
    """Round-1 base-quality weights: native x16, d4 x10, d8 probe x4."""
    w = {"native": NATIVE_WEIGHT, "d4": D4_WEIGHT, "d8_probe": PROBE_WEIGHT}
    return {c.cell_id: w[c.klass] for c in cells if not c.excluded}


def _largest_remainder(weights: Sequence[float], total: int) -> list[int]:
    """Largest-remainder apportionment of ``total`` over ``weights`` (sum -> total)."""
    w = np.asarray(weights, dtype=float)
    if total <= 0 or w.size == 0 or w.sum() <= 0:
        return [0] * int(w.size)
    exact = w / w.sum() * float(total)
    floor = np.floor(exact).astype(int)
    rem = int(total) - int(floor.sum())
    if rem > 0:
        order = np.argsort(-(exact - floor))
        for i in order[:rem]:
            floor[i] += 1
    return floor.tolist()


def proximity_weights(
    cells: Sequence[FrontierCell],
    best_fr: dict[str, float | None],
    has_converged: dict[str, bool],
    *,
    budget: int = ROUND_BUDGET,
    floor: int = FLOOR_WEIGHT,
    probe_cap: int = PROBE_SIZE,
) -> dict[str, int]:
    """Round>=2 boundary-proximity weights (Decision 4), summing to ``budget``.

    ``w_cell ∝ 1/(PROX_EPS + |bestFr - 1.55|)``, floor ``floor`` per surviving cell.
    ``bestFr`` fallback chain (explicit — no None arithmetic): best verified FEASIBLE
    F_r, else best converged-any F_r, else the cell gets the floor weight.  A d8 probe
    cell that has NOT yet produced a converged row is capped at ``probe_cap`` until it
    does.  The floored, capped weights sum EXACTLY to ``budget`` (largest-remainder on
    the residual after floors and caps)."""
    survivors = [c for c in cells if not c.excluded]
    if not survivors:
        return {}
    capped: dict[str, int] = {}
    pool: list[FrontierCell] = []
    for c in survivors:
        if c.is_probe and not has_converged.get(c.cell_id, False):
            capped[c.cell_id] = min(probe_cap, floor)
        else:
            pool.append(c)
    remaining = int(budget) - sum(capped.values())
    if not pool:
        return capped
    base = floor * len(pool)
    extra = remaining - base
    if extra <= 0:
        # not enough budget to exceed the floors — every pool cell gets the floor.
        alloc = {c.cell_id: floor for c in pool}
        alloc.update(capped)
        return alloc
    raws: list[float] = []
    for c in pool:
        bf = best_fr.get(c.cell_id)
        if bf is None:
            raws.append(1.0 / (PROX_EPS + 1.0))          # floor-equivalent baseline
        else:
            raws.append(1.0 / (PROX_EPS + abs(float(bf) - FR_TARGET)))
    extras = _largest_remainder(raws, extra)
    alloc = {c.cell_id: floor + e for c, e in zip(pool, extras)}
    alloc.update(capped)
    return alloc


def coverage_weights(
    cells: Sequence[FrontierCell],
    map_counts: dict[str, int],
    *,
    budget: int = ROUND_BUDGET,
    floor: int = FLOOR_WEIGHT,
    probe_cap: int = PROBE_SIZE,
    target: int = COVERAGE_TARGET,
) -> dict[str, int]:
    """Round>=2 weights by PURE MAP-COVERAGE DEFICIT (decision D7), summing to
    ``budget``.

    ``w_cell ∝ max(0, target − n_mapped_rows(cell))``, floor ``floor`` per
    surviving cell, same probe cap as :func:`proximity_weights`.

    Why not "coverage + achieved flatness", and above all why not the F_r
    proximity rule this replaces:

    * ``|best_F_r − 1.55|`` steers the OUTER budget by F_r while the INNER
      objective is flatness.  The two are simply different problems, and the
      program's whole point is that F_r is retired from the objective.
    * Ranking cells by their achieved ``best map_cov`` / ``best node_peak`` is
      worse than useless: (a) the comparison is across cells, so it ranks by the
      model's per-cell CALIBRATION ERROR as much as by physics, and (b) it is a
      MIN statistic, so a cell that has already been sampled hard looks flatter
      purely because it was sampled more — a rich-get-richer loop that starves
      exactly the under-covered cells this race exists to fill.

    Coverage deficit has neither failure mode: it is a count of labels the cell
    does not have, it cannot be gamed by the model, and it monotonically
    de-prioritizes a cell as that cell fills up.  When EVERY surviving cell has
    met ``target`` the deficits are all zero and the residual budget is split
    evenly (no cell is preferred; the race has done its job).
    """
    survivors = [c for c in cells if not c.excluded]
    if not survivors:
        return {}
    capped: dict[str, int] = {}
    pool: list[FrontierCell] = []
    for c in survivors:
        # A probe cell with NO map yet is still capped: an unproven d8 restart must
        # not soak up budget just because its deficit is maximal.
        if c.is_probe and int(map_counts.get(c.cell_id, 0) or 0) <= 0:
            capped[c.cell_id] = min(probe_cap, floor)
        else:
            pool.append(c)
    remaining = int(budget) - sum(capped.values())
    if not pool:
        return capped
    base = floor * len(pool)
    extra = remaining - base
    if extra <= 0:
        alloc = {c.cell_id: floor for c in pool}
        alloc.update(capped)
        return alloc
    deficits = [max(0.0, float(target) - float(map_counts.get(c.cell_id, 0) or 0))
                for c in pool]
    if sum(deficits) <= 0.0:
        deficits = [1.0] * len(pool)          # fully covered -> split evenly
    extras = _largest_remainder(deficits, extra)
    alloc = {c.cell_id: floor + e for c, e in zip(pool, extras)}
    alloc.update(capped)
    return alloc


def store_map_counts(store_dir: str | Path | None,
                     cells: Sequence[FrontierCell]) -> dict[str, int]:
    """``{cell_id: mapped-row count}`` from the store (0 for every cell on error).

    "Mapped" means the row carries a ``node_peak`` label — the actual thing the
    flatness objective needs, not merely a ``maps_key`` (program §9 P-5 left 13
    rows carrying a key with no map behind it).
    """
    counts = {c.cell_id: 0 for c in cells}
    if store_dir is None:
        return counts
    try:
        from ..data.store import StoreReader

        df = StoreReader(Path(store_dir)).records
    except Exception:  # noqa: BLE001 — a missing store means "no coverage yet"
        return counts
    if df is None or not len(df) or "node_peak" not in df.columns:
        return counts
    try:
        import pandas as pd

        sub = df[df["node_peak"].notna()]
        feed = pd.to_numeric(sub["feed"], errors="coerce")
        for c in cells:
            counts[c.cell_id] = int(
                ((sub["case_pair"] == c.pair) & (feed == c.feed)).sum()
            )
    except Exception:  # noqa: BLE001
        return {c.cell_id: 0 for c in cells}
    return counts


# --------------------------------------------------------------------------- #
# per-cell deterministic seed
# --------------------------------------------------------------------------- #
def cell_seed(cell_id: str, base_seed: int) -> int:
    """Deterministic per-cell seed ``sha256("frB|{cell_id}|{base_seed}")[:8]``."""
    digest = hashlib.sha256(f"frB|{cell_id}|{int(base_seed)}".encode()).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------------------------------- #
# state IO
# --------------------------------------------------------------------------- #
def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}


def _state_spent(run_dir: Path) -> int:
    """The persisted ``budget_spent`` of a cell's resumable CampaignDriver state."""
    st = _read_json(run_dir / "state.json")
    try:
        return int(st.get("budget_spent", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _state_best_fr(run_dir: Path) -> tuple[float | None, float | None]:
    """``(feasible_best_fr, converged_any_best_fr)`` from a cell's state.json."""
    st = _read_json(run_dir / "state.json")

    def _fr(d: Any) -> float | None:
        if not isinstance(d, dict):
            return None
        v = d.get("f_r")
        try:
            return float(v) if v is not None and math.isfinite(float(v)) else None
        except (TypeError, ValueError):
            return None

    return _fr(st.get("best")), _fr(st.get("best_overall"))


# --------------------------------------------------------------------------- #
# round result
# --------------------------------------------------------------------------- #
@dataclass
class FrontierRoundResult:
    """Outcome of ONE frontier round."""

    round_index: int
    round_spent: int
    per_cell: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    retrain_events: list[dict[str, Any]] = field(default_factory=list)


#: round1c-proven lean CPU caps for a per-cell local_cpu frontier campaign — the
#: [search] defaults (pool 20000 / local_search max_predictions 40000) flood the
#: featurization-bound screen and stall a wave (forensic 20260723).
_LEAN_POOL_SIZE = 2000
_LEAN_LS_TOP_M = 32
_LEAN_LS_NEIGHBORS = 48
_LEAN_LS_DEPTH = 2
_LEAN_LS_MAX_PRED = 1500


def _lean_search(search: Any) -> Any:
    """Return ``search`` clamped to the lean CPU caps (only ever LOWERS a value, so
    a deck already at/below the caps is byte-identical)."""
    ls = search.local_search
    lean_ls = _replace(
        ls,
        top_m=min(ls.top_m, _LEAN_LS_TOP_M),
        neighbors=min(ls.neighbors, _LEAN_LS_NEIGHBORS),
        depth=min(ls.depth, _LEAN_LS_DEPTH),
        max_predictions=min(ls.max_predictions, _LEAN_LS_MAX_PRED),
    )
    return _replace(search, pool_size=min(search.pool_size, _LEAN_POOL_SIZE),
                    local_search=lean_ls)


class FrBoundaryOuterRace:
    """Drive the outer cell-race, ONE round per :meth:`run_round`.

    Two modes, and they allocate the outer budget by DIFFERENT rules because they
    are different problems (decision D7):

    ``fr_boundary`` — **a design-space mapping campaign, not a core-loading
    objective** (decision D5).  Its question is "which design cells can reach
    F_r <= 1.55 at all", which no surrogate answers; it is retained (the draft's
    plan to delete it rested on a POOLED statistic — pooled P(F_r<=1.55) = 0.0147
    hid 16 cells with per-cell rates up to 0.722) and RE-AIMED at exactly those 16
    cells.  Budget is allocated by F_r proximity, which is legitimate here because
    F_r IS this mode's objective.  Nothing it produces is a delivery candidate by
    virtue of being produced; §2.2 decides that separately.

    ``flat_power`` — the flatness-first production race.  Budget is allocated by
    PURE MAP-COVERAGE DEFICIT (:func:`coverage_weights`); the F_r proximity rule
    is retired here (program §10 STOP).

    Construction reads persistent state (``race_state.json`` + each cell's own
    ``state.json``) so a fresh process resumes exactly where the last one stopped.
    ``driver_factory(cell, budget, run_dir, seed) -> driver`` builds a per-cell
    CampaignDriver (default: the real one with ``early_stop=False`` and a fresh
    ``backend_factory`` per cell); tests inject a stub.
    """

    def __init__(
        self,
        cfg: Any,
        model: Any,
        *,
        run_root: str | Path,
        round_budget: int = ROUND_BUDGET,
        base_seed: int | None = None,
        driver_factory: Callable[..., Any] | None = None,
        backend_factory: Callable[[str], Any] | None = None,
        log: Callable[[str], None] | None = None,
        retrain_gate_callback: Callable[[int], dict | None] | None = None,
        model_reload: Callable[[str], Any] | None = None,
        exclude_cells: "set[str] | None" = None,
        objective: str | None = None,
        compliant_only: bool | None = None,
        map_counts: Callable[[Sequence[FrontierCell]], dict[str, int]] | None = None,
        coverage_target: int = COVERAGE_TARGET,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.round_budget = int(round_budget)
        self.base_seed = int(
            base_seed if base_seed is not None
            else getattr(getattr(cfg, "flow", None), "random_seed", 0) or 0
        )
        self.driver_factory = driver_factory or self._default_driver_factory
        self.backend_factory = backend_factory
        # Encoding-safe default logger: a redirected Windows stdout is cp949 and a
        # single em-dash used to raise UnicodeEncodeError mid-run (2026-08-30).
        self._log = safe_logger(log)
        self.retrain_gate_callback = retrain_gate_callback
        self.model_reload = model_reload

        # -- objective (decides the ALLOCATION RULE and the roster) -------------
        obj = str(objective or getattr(getattr(cfg, "acquisition", None),
                                       "objective", "") or "fr_boundary")
        # Same clamp the CLI applies: an unrecognised objective must never end up
        # silently allocating by a rule that does not match the inner campaign.
        self.objective = obj if obj in _RACE_OBJECTIVES else "fr_boundary"
        if compliant_only is None:
            compliant_only = (
                self.objective == "fr_boundary"
                and bool(getattr(getattr(cfg, "acquisition", None),
                                 "fr_boundary_compliant_only", True))
            )
        self.compliant_only = bool(compliant_only)
        self.coverage_target = int(coverage_target)
        self._map_counts_fn = map_counts or self._default_map_counts

        self.roster = build_roster(compliant_only=self.compliant_only)
        self._by_id = {c.cell_id: c for c in self.roster}
        self.race_state_path = self.run_root / "race_state.json"

        # -- persistent race state ---------------------------------------------
        self.round_index = 0
        self.cell_grant: dict[str, int] = {}
        self.best_fr_feasible: dict[str, float | None] = {}
        self.best_fr_converged: dict[str, float | None] = {}
        self.map_counts: dict[str, int] = {}
        self.probe_strikes: dict[str, int] = {}
        self.transient_strikes: dict[str, int] = {}
        self.exclusions: dict[str, str] = {}
        self.retrain_events: list[dict[str, Any]] = []
        self.migrations: list[dict[str, Any]] = []
        self._new_labels_since_retrain = 0
        self._load_race_state()

        # -- seed spent from EACH cell's own state.json (phantom-charge safe) ---
        self.cell_spent: dict[str, int] = {}
        for cell in self.roster:
            self.cell_spent[cell.cell_id] = _state_spent(self._cell_run_dir(cell))

        # Multi-PC DISJOINT split (user directive 2026-07-23): cells assigned to a
        # PEER PC are excluded here so two boxes run SEROSO (disjoint) rosters — the
        # curriculum-era "same deck + same seed -> 80% duplicate" failure cannot
        # recur (disjoint cells + per-PC seed).  Merged with any persisted exclusion
        # and PERSISTED itself, so a resumed round keeps the same half.
        if exclude_cells:
            for cid in exclude_cells:
                if cid in self._by_id:
                    self.exclusions.setdefault(cid, "assigned to peer PC (disjoint split)")

        # apply persisted exclusions to the live roster.
        for cell in self.roster:
            if cell.cell_id in self.exclusions:
                cell.excluded = True
                cell.reason = self.exclusions[cell.cell_id]

    # -- run dirs / seeds --------------------------------------------------- #
    def _cell_run_dir(self, cell: FrontierCell) -> Path:
        return self.run_root / f"alsearch_{cell.pair}_f{cell.feed}_frB"

    def cell_seed(self, cell_id: str) -> int:
        return cell_seed(cell_id, self.base_seed)

    # -- default (live) driver factory -------------------------------------- #
    def _cell_cfg(self, cell: FrontierCell) -> Any:
        case = _replace(self.cfg.case, mode="fixed", pair=cell.pair, feed=cell.feed)
        acq = _replace(self.cfg.acquisition, objective=self.objective)
        # Lean-search clamp (forensic 20260723): a per-cell local_cpu campaign must
        # NOT inherit the heavy [search] defaults (pool 20000 / local_search
        # max_predictions 40000) that flood the screen — clamp to the round1c-proven
        # lean caps even when a deck omitted the block.  A deck already at/below the
        # caps is unchanged.
        search = _lean_search(self.cfg.search)
        return _replace(self.cfg, case=case, acquisition=acq, search=search)

    def _default_driver_factory(
        self, cell: FrontierCell, budget: int, run_dir: Path, seed: int
    ) -> Any:
        from .campaign import CampaignDriver

        return CampaignDriver(
            self._cell_cfg(cell), self.model,
            budget=budget, run_dir=run_dir, resume=True,
            backend_factory=self.backend_factory, early_stop=False,
            seed=seed, progress=False, log=self._log,
        )

    # -- weights ------------------------------------------------------------ #
    def _default_map_counts(self, cells: Sequence[FrontierCell]) -> dict[str, int]:
        store = getattr(getattr(self.cfg, "model", None), "store_dir", None)
        return store_map_counts(store, cells)

    def refresh_map_counts(self) -> dict[str, int]:
        """Re-read the per-cell mapped-row counts (the coverage-deficit input)."""
        try:
            self.map_counts = {k: int(v) for k, v in
                               self._map_counts_fn(self.roster).items()}
        except Exception as exc:  # noqa: BLE001 — coverage is advisory, never fatal
            self._log(f"[frontier][WARNING] map-count read failed ({exc}); "
                      "treating every cell as uncovered")
            self.map_counts = {c.cell_id: 0 for c in self.roster}
        return self.map_counts

    def weights_for_round(self, round_index: int) -> dict[str, int]:
        survivors = [c for c in self.roster if not c.excluded]
        if round_index == 0:
            return round1_weights(survivors)
        if self.objective == "flat_power":
            # decision D7: PURE coverage deficit.  No F_r proximity, and NO
            # achieved-flatness term (which would rank cells by calibration error
            # and hand more budget to the already-best-sampled cell).
            counts = self.refresh_map_counts()
            deficit = sum(max(0, self.coverage_target - int(counts.get(c.cell_id, 0)))
                          for c in survivors)
            self._log(f"[frontier] round {round_index} allocation = map-coverage "
                      f"deficit (target {self.coverage_target}/cell, total deficit "
                      f"{deficit} over {len(survivors)} cells)")
            return coverage_weights(survivors, counts, budget=self.round_budget,
                                    target=self.coverage_target)
        best = {c.cell_id: self._best_fr(c.cell_id) for c in survivors}
        conv = {c.cell_id: self.best_fr_converged.get(c.cell_id) is not None
                for c in survivors}
        return proximity_weights(survivors, best, conv, budget=self.round_budget)

    def _best_fr(self, cell_id: str) -> float | None:
        """bestFr fallback chain: feasible best -> converged-any best -> None."""
        fb = self.best_fr_feasible.get(cell_id)
        if fb is not None:
            return fb
        return self.best_fr_converged.get(cell_id)

    # -- one round ---------------------------------------------------------- #
    def run_round(self) -> FrontierRoundResult:
        """Run EXACTLY one round over the surviving roster, then return + persist."""
        weights = self.weights_for_round(self.round_index)
        per_cell: list[dict[str, Any]] = []
        round_spent = 0
        for cell in self.roster:
            if cell.excluded:
                continue
            w = int(weights.get(cell.cell_id, 0))
            if w <= 0:
                continue
            row = self._run_cell(cell, w)
            round_spent += int(row["delta"])
            per_cell.append(row)
            self._save_race_state()               # after each cell completes

        self.round_index += 1
        self._new_labels_since_retrain += round_spent
        gate = self._maybe_retrain(round_spent)
        self._save_race_state()                    # at round end
        excluded = [{"cell": cid, "reason": r} for cid, r in sorted(self.exclusions.items())]
        return FrontierRoundResult(
            round_index=self.round_index - 1, round_spent=round_spent,
            per_cell=per_cell, excluded=excluded,
            retrain_events=[gate] if gate else [],
        )

    def _run_cell(self, cell: FrontierCell, weight: int) -> dict[str, Any]:
        cid = cell.cell_id
        run_dir = self._cell_run_dir(cell)
        persisted = self.cell_spent.get(cid, 0)
        grant = persisted + weight
        self.cell_grant[cid] = grant
        outcome = "ran"
        crashed = False
        try:
            driver = self.driver_factory(cell, grant, run_dir, self.cell_seed(cid))
            result = driver.run()
        except Exception as exc:  # noqa: BLE001 — a crash must still charge real spend
            crashed = True
            # Real pre-crash spend is in the cell's state.json (labels already saved).
            after = _state_spent(run_dir)
            delta = max(0, after - persisted)
            self.cell_spent[cid] = after
            self._classify_exception(cell, exc)
            outcome = f"crash:{type(exc).__name__}"
            self._log(f"[frontier] cell {cid} {outcome} charged delta={delta}")
            return {"cell": cid, "klass": cell.klass, "granted": grant,
                    "delta": delta, "excluded": cell.excluded, "reason": cell.reason,
                    "outcome": outcome, "best_fr": self._best_fr(cid)}

        after = int(getattr(result, "budget_spent", None) or _state_spent(run_dir))
        delta = max(0, after - persisted)
        self.cell_spent[cid] = after
        self._update_best(cid, result)
        converged_this_round = self._converged_this_round(result)
        # a successful (non-crashing) round clears the transient-failure streak.
        self.transient_strikes[cid] = 0
        if cell.is_probe:
            self._update_probe(cell, converged_this_round)
        return {"cell": cid, "klass": cell.klass, "granted": grant, "delta": delta,
                "excluded": cell.excluded, "reason": cell.reason, "outcome": outcome,
                "converged": converged_this_round, "best_fr": self._best_fr(cid),
                # the quantity that DROVE this cell's grant under flat_power
                # (decision D7); None under the F_r-proximity rule.
                "n_mapped": (self.map_counts.get(cid)
                             if self.objective == "flat_power" else None)}

    # -- best / convergence bookkeeping ------------------------------------- #
    def _update_best(self, cid: str, result: Any) -> None:
        feas, conv = self._result_best_fr(result)
        if feas is None and conv is None:
            feas, conv = _state_best_fr(self._cell_run_dir(self._by_id[cid]))
        if feas is not None:
            cur = self.best_fr_feasible.get(cid)
            self.best_fr_feasible[cid] = feas if cur is None else min(cur, feas)
        if conv is not None:
            cur = self.best_fr_converged.get(cid)
            self.best_fr_converged[cid] = conv if cur is None else min(cur, conv)

    @staticmethod
    def _result_best_fr(result: Any) -> tuple[float | None, float | None]:
        def _fr(d: Any) -> float | None:
            if not isinstance(d, dict):
                return None
            v = d.get("f_r")
            try:
                return float(v) if v is not None and math.isfinite(float(v)) else None
            except (TypeError, ValueError):
                return None
        return _fr(getattr(result, "best", None)), _fr(getattr(result, "best_overall", None))

    @staticmethod
    def _converged_this_round(result: Any) -> bool:
        """True when the cell has any converged row (feasible best or converged-any)."""
        if getattr(result, "best", None) is not None:
            return True
        if getattr(result, "best_overall", None) is not None:
            return True
        return int(getattr(result, "budget_spent", 0) or 0) > 0 and bool(
            getattr(result, "n_feasible", 0)
        )

    def _update_probe(self, cell: FrontierCell, converged_this_round: bool) -> None:
        cid = cell.cell_id
        # once a probe has EVER produced a converged row it is out of probe jeopardy.
        if self.best_fr_converged.get(cid) is not None or converged_this_round:
            self.probe_strikes[cid] = 0
            return
        self.probe_strikes[cid] = self.probe_strikes.get(cid, 0) + 1
        if self.probe_strikes[cid] >= D8_STRIKE_LIMIT:
            self._exclude(cell, "f121->f113 d8 restart diverges (non_finite_flux): "
                                f"{self.probe_strikes[cid]} zero-converged probe rounds")

    # -- exceptions --------------------------------------------------------- #
    def _classify_exception(self, cell: FrontierCell, exc: Exception) -> None:
        name = type(exc).__name__
        if name in _STRUCTURAL_EXC:
            self._exclude(cell, f"structural {name}: {exc}")
            return
        cid = cell.cell_id
        self.transient_strikes[cid] = self.transient_strikes.get(cid, 0) + 1
        if self.transient_strikes[cid] >= TRANSIENT_LIMIT:
            self._exclude(cell, f"transient {name} for {self.transient_strikes[cid]} "
                                f"consecutive rounds: {exc}")

    def _exclude(self, cell: FrontierCell, reason: str) -> None:
        cell.excluded = True
        cell.reason = reason
        self.exclusions[cell.cell_id] = reason
        self._log(f"[frontier] cell {cell.cell_id} EXCLUDED: {reason}")

    # -- retrain hook ------------------------------------------------------- #
    def _maybe_retrain(self, round_spent: int) -> dict | None:
        if self.retrain_gate_callback is None or round_spent <= 0:
            return None
        n = self._new_labels_since_retrain
        gate = self.retrain_gate_callback(n)
        self._new_labels_since_retrain = 0
        if not gate:
            return None
        passed = bool(gate.get("pass"))
        event = {"new_labels": n, **gate}
        self.retrain_events.append(event)
        if passed and gate.get("champion_model_dir") and self.model_reload is not None:
            self.model = self.model_reload(str(gate["champion_model_dir"]))
            self._log(f"[frontier][AL] champion promoted -> {gate['champion_model_dir']}")
        return event

    # -- persistence -------------------------------------------------------- #
    #: race_state keys that only mean something for ONE allocation rule.  Changing
    #: the objective must not carry them across (decision D7): they were recorded
    #: under a different rule and would silently steer the new one.
    _OBJECTIVE_SCOPED_KEYS = ("best_fr_feasible", "best_fr_converged", "map_counts")

    def _save_race_state(self) -> None:
        _atomic_json(self.race_state_path, {
            "schema": "race_state_v2",
            "objective": self.objective,
            "roster": "compliant16" if self.compliant_only else "full24",
            "coverage_target": self.coverage_target,
            "round_index": self.round_index,
            "round_budget": self.round_budget,
            "base_seed": self.base_seed,
            "cell_grant": self.cell_grant,
            "cell_spent": self.cell_spent,
            "best_fr_feasible": self.best_fr_feasible,
            "best_fr_converged": self.best_fr_converged,
            "map_counts": self.map_counts,
            "probe_strikes": self.probe_strikes,
            "transient_strikes": self.transient_strikes,
            "exclusions": self.exclusions,
            "retrain_events": self.retrain_events,
            "migrations": self.migrations,
        })

    def _load_race_state(self) -> None:
        st = _read_json(self.race_state_path)
        if not st:
            return
        self.round_index = int(st.get("round_index", 0) or 0)
        self.cell_grant = dict(st.get("cell_grant", {}) or {})
        self.best_fr_feasible = dict(st.get("best_fr_feasible", {}) or {})
        self.best_fr_converged = dict(st.get("best_fr_converged", {}) or {})
        self.map_counts = {k: int(v) for k, v in (st.get("map_counts", {}) or {}).items()}
        self.probe_strikes = dict(st.get("probe_strikes", {}) or {})
        self.transient_strikes = dict(st.get("transient_strikes", {}) or {})
        self.exclusions = dict(st.get("exclusions", {}) or {})
        self.retrain_events = list(st.get("retrain_events", []) or [])
        self.migrations = list(st.get("migrations", []) or [])
        self._migrate_race_state(st)

    def _migrate_race_state(self, st: dict[str, Any]) -> None:
        """Reconcile a race_state written under a DIFFERENT objective / roster.

        A pre-v2 state has no ``objective`` key at all; it can only have been an
        ``fr_boundary`` state, so it is labelled as one rather than guessed at.

        When the objective actually changed, the objective-scoped allocation keys
        are RESET (not silently reused) with a loud log and an audit entry in
        ``migrations``.  Everything objective-INDEPENDENT — cumulative spend, the
        round counter, exclusions, probe/transient strike counters, retrain
        history — is kept, because it describes what MASTER did, not how the
        budget was steered.  Losing that would re-charge real spend.
        """
        prev_obj = st.get("objective")
        if prev_obj is None:
            prev_obj = "fr_boundary"        # pre-v2 states are fr_boundary by origin
            self._log("[frontier][MIGRATION] race_state has no 'objective' key "
                      "(pre-v2); treating it as fr_boundary")
        prev_roster = st.get("roster")
        cur_roster = "compliant16" if self.compliant_only else "full24"

        if str(prev_obj) != self.objective:
            self._log(
                f"[frontier][MIGRATION] race_state objective {prev_obj!r} -> "
                f"{self.objective!r}: RESETTING the objective-scoped allocation "
                f"keys {list(self._OBJECTIVE_SCOPED_KEYS)} (they were recorded "
                f"under a different budget rule and would steer this one). "
                f"Spend / round_index / exclusions / strikes are KEPT.")
            self.best_fr_feasible = {}
            self.best_fr_converged = {}
            self.map_counts = {}
            self.migrations.append({
                "at": "load", "from_objective": str(prev_obj),
                "to_objective": self.objective,
                "reset": list(self._OBJECTIVE_SCOPED_KEYS),
                "round_index": self.round_index,
            })
        if prev_roster is not None and str(prev_roster) != cur_roster:
            # ``cell_spent`` is seeded from each cell's own state.json AFTER this
            # runs, so read the persisted map rather than the attribute.
            dropped = sorted(set(st.get("cell_spent", {}) or {}) - set(self._by_id))
            self._log(
                f"[frontier][MIGRATION] roster {prev_roster!r} -> {cur_roster!r}: "
                f"{len(dropped)} persisted cell(s) are no longer on the roster and "
                f"will not be run again (their spend is retained for the record): "
                f"{dropped}")
            self.migrations.append({
                "at": "load", "from_roster": str(prev_roster),
                "to_roster": cur_roster, "off_roster_cells": dropped,
            })

    # -- roster report ------------------------------------------------------ #
    def roster_report(self) -> list[dict[str, Any]]:
        """Every grid cell with base_status + exclusion reason (CLI roster JSON)."""
        return [{
            "cell": c.cell_id, "pair": c.pair, "feed": c.feed,
            "e_core": c.e_core, "class": c.klass, "base_status": c.base_status,
            "excluded": c.excluded, "reason": c.reason,
            "granted": self.cell_grant.get(c.cell_id, 0),
            "spent": self.cell_spent.get(c.cell_id, 0),
            "best_fr": self._best_fr(c.cell_id),
            "n_mapped": self.map_counts.get(c.cell_id),
            "objective": self.objective,
        } for c in self.roster]


__all__ = [
    "FrontierCell",
    "FrontierRoundResult",
    "FrBoundaryOuterRace",
    "build_roster",
    "round1_weights",
    "proximity_weights",
    "coverage_weights",
    "store_map_counts",
    "is_compliant_cell",
    "cell_seed",
    "is_cross_anchor",
    "COVERAGE_TARGET",
    "FR_COMPLIANT_CELLS",
    "NATIVE_WEIGHT",
    "D4_WEIGHT",
    "PROBE_WEIGHT",
    "PROBE_SIZE",
    "ROUND_BUDGET",
    "FLOOR_WEIGHT",
    "D8_STRIKE_LIMIT",
    "TRANSIENT_LIMIT",
]
