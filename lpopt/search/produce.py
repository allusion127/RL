"""Learning-data production campaign driver (``python -m lpopt produce``).

``produce`` is the no-acquisition wave판 of the :class:`WaveVerifier` harness
(plan sections 5 / 5.4): a stratified DoE sampler that fills ``[[produce.strata]]``
cells with MASTER equilibrium labels for new ``(pair, feed, split)`` regions the
existing ~43k records never covered.

Pieces:

* **Generators** — G1 ``random`` (feed-general :func:`random_genome`), G2
  ``heuristic`` (ring/checkerboard/radial priors generalized to N via
  overlap-biased random sampling), G3 ``elite_perturb`` (top converged store
  rows by ``[produce] elite_objective`` — ``cyclen`` (legacy default) or
  ``flat`` / ``flat_feasible`` for a flatness campaign — -> ``from_pattern`` ->
  feed-morph to the stratum N -> ``mutate``, recording ``parent_record_id``; a
  draw that finds no usable parent degrades to ``random`` and says so LOUDLY),
  G4 ``rule_biased`` (OPTIONAL, config-gated,
  default OFF — ``random`` draws with the worst RM1 decile rejected; engineering
  rule R-03, see :mod:`.rule_metrics`).
* **Ledger** — append-only JSONL (``{record_id, stratum, generator,
  parent_record_id, status, restart_provenance, ts}``), flushed per line.  On
  start the ledger + store record_ids are replayed into a dedup set and
  per-stratum counters, so ``kill -9`` resume is duplicate-free and
  loss-free (the store is authoritative for produced labels).
* **Loop** — fill waves of ``workers`` entries from the highest-priority unmet
  stratum (round-robin within a priority), resolve assets, verify, write store
  rows (``dataset="P"``) + ledger, run QC counters with the plan's rebalancing
  rules as WARNING logs, and print per-wave progress.
* ``--dry-run`` swaps in :class:`StubEvaluator` and a run-scoped temp store so no
  live MASTER run and no main-store pollution occur (the acceptance path today).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..config import LpoptConfig, ProduceConfig, StratumConfig
from ..safelog import safe_print
from ..data.fuel_types import case_e_core as _case_e_core
from ..data.schema import CanonicalRecord, compute_record_id, unpack_pattern
from ..data.store import StoreReader, StoreWriter
from ..vendor.masterrl.domain import CaseKey, Pattern
from ..vendor.masterrl.ga import ORBIT_UNITS, MOVABLE_UNIT_COUNT
from ..vendor.masterrl.domain import SLOTS
from .assets import CaseAssetResolver, LIBRARY_DIMS
from .resolver import build_case_resolver, is_paramA_library
from .rule_metrics import rm_fresh_face_adjacency
from .genome import (
    GeneralOrbitGenome,
    GenomeError,
    case_batches,
    mutate,
    random_genome,
    _add_fresh_unit,
    _remove_fresh_unit,
)
from .stub import StubEvaluator
from .verify import (
    PRODUCE_DECK_KNOBS,
    WaveEntry,
    WaveOutcome,
    WaveVerifier,
    classify_outcome,
    outcome_to_record,
)

_GEN_ATTEMPT_FACTOR = 40  # per-wave unique-candidate draw attempts / workers
_ELITE_POOL = 32

#: ``elite_perturb`` PARENT-selection objectives (``[produce] elite_objective``
#: and the per-stratum override).  See :attr:`lpopt.config.ProduceConfig.
#: elite_objective` for the measured E1_E2 justification of the non-legacy modes.
#:
#: ``cyclen`` is the LEGACY DEFAULT and its code path is byte-identical to the
#: pre-fix driver — an unset knob cannot change a single draw of a running
#: cycle-length campaign.
ELITE_OBJECTIVES: tuple[str, ...] = ("cyclen", "flat", "flat_feasible")

#: The ``flat`` / ``flat_feasible`` ordering keys: ``node_peak`` PRIMARY,
#: ``map_cov`` SECONDARY, both ASCENDING (lower is flatter) — the same
#: primary/secondary pair, in the same direction, as the ``flat_power``
#: objective ``-( node_peak/PEAK_SCALE + w_cov * map_cov/COV_SCALE )``.
_FLAT_ELITE_KEYS: tuple[str, ...] = ("node_peak", "map_cov")

#: Constraint axes a ``flat_feasible`` elite must pass, read from
#: ``campaign.feasibility_limits_for(acq, "flat_power")``.  F_r and pin burnup
#: are deliberately NOT applied here — see :meth:`ProduceDriver._flat_feasible_mask`.
_FLAT_ELITE_GATES: tuple[tuple[str, str], ...] = (
    ("cbc_max", "cbc_max"), ("f_q", "f_q"), ("ao_abs", "ao_abs"),
)

#: G4 ``rule_biased`` metric selector.  Only the two VALIDATED fresh-fresh
#: face-adjacency forms are offered (R-03): ``rm1`` whole core (the decile rule
#: as specified) and ``rm1i`` inboard-only pairs (measured stronger, rho +0.235
#: vs +0.085 against ``node_peak``).
_RULE_BIAS_METRICS = {
    "rm1": lambda p: rm_fresh_face_adjacency(p, inboard=False),
    "rm1i": lambda p: rm_fresh_face_adjacency(p, inboard=True),
}


def _rule_bias_metric(produce_cfg: ProduceConfig) -> Callable[[Pattern], float]:
    """The ``rule_biased`` generator's metric callable (raises on an unknown id)."""
    name = str(getattr(produce_cfg, "rule_bias_metric", "rm1") or "rm1").lower()
    if name not in _RULE_BIAS_METRICS:
        raise ValueError(
            f"[produce] rule_bias_metric={name!r} is not available; choose one of "
            f"{sorted(_RULE_BIAS_METRICS)} (the fresh-fresh face-adjacency forms "
            f"validated by the within-cell study)."
        )
    return _RULE_BIAS_METRICS[name]


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #
class Ledger:
    """Append-only JSONL ledger with per-line flush and tolerant replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, **fields: Any) -> None:
        fields.setdefault("ts", time.time())
        line = json.dumps(fields, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    @staticmethod
    def replay(path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            return rows
        with open(p, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A kill -9 can leave a torn final line; tolerate it.
                    continue
        return rows


# --------------------------------------------------------------------------- #
# per-stratum runtime state / QC
# --------------------------------------------------------------------------- #
@dataclass
class _StratumState:
    cfg: StratumConfig
    produced: int = 0                 # cumulative CONVERGED labels (store ground truth)
    converged: int = 0                # this-run converged
    nonconverged: int = 0             # this-run honest non-convergence
    nonfinite: int = 0                # this-run non_finite_flux watchdog kills
    harness_error: int = 0            # this-run harness/deck/resolver defects
    attempts: int = 0                 # this-run chains evaluated
    stalled: bool = False
    effective_n_target: int = 0
    effective_generators: dict[str, float] = field(default_factory=dict)
    warned_err: bool = False          # HALT (harness-error) warning emitted
    warned_rebalance: bool = False    # generator-mix rebalance advisory emitted
    #: Draws where ``elite_perturb`` found no usable parent and DEGRADED to a
    #: random draw.  The ledger's ``generator`` field stays truthful ("random"),
    #: which is exactly why this counter exists: without it the degradation is
    #: invisible — a stratum with a 60% elite_perturb weight can spend the whole
    #: campaign drawing random and nothing in the ledger says so.
    elite_fallback_random: int = 0
    warned_elite_fallback: bool = False   # loud-fallback warning emitted (once)

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def remaining(self) -> int:
        return max(0, self.effective_n_target - self.produced)


# --------------------------------------------------------------------------- #
# heuristic fresh-unit sets (ring / checkerboard / radial), generalized to N
# --------------------------------------------------------------------------- #
def _by_radius() -> list[int]:
    return sorted(range(MOVABLE_UNIT_COUNT), key=lambda u: (ORBIT_UNITS[u].radius, u))


def _take_n(primary: Sequence[int], fill_order: Sequence[int], n: int) -> set[int]:
    prim = sorted(primary, key=lambda u: (ORBIT_UNITS[u].radius, u))
    sel = list(prim[:n])
    if len(sel) < n:
        have = set(sel)
        sel += [u for u in fill_order if u not in have][: n - len(sel)]
    return set(sel[:n])


def heuristic_fresh_set(rule: str, n: int) -> set[int]:
    """Ring / checkerboard / radial fresh-unit set of size ``n`` (plan 5.3 / G2)."""

    by_radius = _by_radius()
    if rule == "ring":
        ring = [u for i, u in enumerate(by_radius) if i % 2 == 0]
        return _take_n(ring, by_radius, n)
    if rule == "checker":
        checker = [
            unit.unit
            for unit in ORBIT_UNITS
            if (SLOTS[unit.slots[0]].row + SLOTS[unit.slots[0]].col) % 2 == 0
        ]
        return _take_n(checker, by_radius, n)
    # radial: fresh fuel on the outermost N units (low-leakage-style periphery)
    return set(by_radius[-n:])


# --------------------------------------------------------------------------- #
# genome helpers
# --------------------------------------------------------------------------- #
def _apply_split(
    genome: GeneralOrbitGenome,
    rng: random.Random,
    batches: tuple[str, ...],
    w1: float,
    center_batch: str,
) -> GeneralOrbitGenome:
    """Relabel fresh units to a ~``w1`` fraction of the case's first type.

    A 2-type case keeps the historical single-draw Bernoulli relabel verbatim.  A
    3-type (graded) case puts ``w1`` on the first type and splits the remaining
    ``1 - w1`` evenly over the rest, so a triple stratum actually feeds three
    types — the alternative (falling through to ``batches[:2]``) would silently
    produce a 2-type core carrying a 3-type ``case_pair`` label.
    """

    from dataclasses import replace

    type_a = batches[0]
    if len(batches) <= 2:
        type_b = batches[1] if len(batches) > 1 else batches[0]
        new_fresh = tuple(
            (unit, type_a if rng.random() < w1 else type_b) for unit, _ in genome.fresh
        )
    else:
        rest = list(batches[1:])
        edges = [w1 + (1.0 - w1) * (i + 1) / len(rest) for i in range(len(rest))]

        def _draw() -> str:
            u = rng.random()
            if u < w1:
                return type_a
            for name, edge in zip(rest, edges, strict=True):
                if u < edge:
                    return name
            return rest[-1]

        new_fresh = tuple((unit, _draw()) for unit, _ in genome.fresh)
    candidate = replace(
        genome, fresh=tuple(sorted(new_fresh)), center_batch=center_batch
    )
    candidate.validate()
    return candidate


def _morph_feed(
    genome: GeneralOrbitGenome,
    rng: random.Random,
    target_n: int,
    batches: tuple[str, ...],
) -> GeneralOrbitGenome | None:
    """Directed feed-morph to ``target_n`` fresh units via add/remove moves."""

    guard = 0
    current = genome
    while current.n_fresh != target_n and guard < 400:
        guard += 1
        if current.n_fresh < target_n:
            nxt = _add_fresh_unit(current, rng, batches)
        else:
            nxt = _remove_fresh_unit(current, rng)
        if nxt == current:
            continue  # stuck this draw; retry within the guard budget
        current = nxt
    return current if current.n_fresh == target_n else None


def _pattern_split(pattern: Pattern, type_a: str) -> float:
    counts = pattern.batch_feed()
    total = sum(counts.values())
    return counts.get(type_a, 0) / total if total else 0.0


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
@dataclass
class ProduceSummary:
    campaign: str
    dry_run: bool
    waves: int
    chains: int
    converged: int
    nonconverged: int
    nonfinite: int
    harness_error: int
    errors: int                       # total error outcomes (= nonfinite + harness_error)
    duplicates: int
    store_dir: str
    ledger_path: str
    strata: list[dict[str, Any]]
    #: E-core awareness (accounting only): {"P"|"E"|"?": chains} and cumulative
    #: wall seconds per class, so wall-time stats can distinguish P vs E chains.
    core_class_chains: dict[str, int] = field(default_factory=dict)
    core_class_wall_s: dict[str, float] = field(default_factory=dict)
    #: Draws where ``elite_perturb`` degraded to ``random`` (all strata summed).
    #: A non-zero value means the exploit arm was partly a random arm.
    elite_fallback_random: int = 0


class ProduceDriver:
    """Stratified MASTER production campaign over the WaveVerifier harness."""

    def __init__(
        self,
        cfg: LpoptConfig,
        *,
        dry_run: bool = False,
        run_dir: str | Path | None = None,
        store_dir: str | Path | None = None,
        ledger_path: str | Path | None = None,
        verifier: WaveVerifier | None = None,
        resolver: CaseAssetResolver | None = None,
        fuel_library: Any = None,
        rng: random.Random | None = None,
        progress: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.produce: ProduceConfig = cfg.produce
        self.dry_run = bool(dry_run)
        self.rng = rng or random.Random(cfg.flow.random_seed)
        self.progress = progress
        self._log = log or (lambda msg: print(msg))

        base = cfg.source_path.parent if cfg.source_path else Path.cwd()
        self._base = base

        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            tag = "dryrun" if self.dry_run else "run"
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = self._resolve(cfg.flow.output_root) / f"produce_{tag}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Store / ledger: dry-run is run-scoped (no main-store pollution).
        if store_dir is not None:
            self.store_dir = Path(store_dir)
        elif self.dry_run:
            self.store_dir = self.run_dir / "store"
        else:
            self.store_dir = self._resolve(self.produce.store_dir)
        if ledger_path is not None:
            self.ledger_path = Path(ledger_path)
        elif self.dry_run:
            self.ledger_path = self.run_dir / "ledger.jsonl"
        else:
            self.ledger_path = self._resolve(self.produce.ledger)

        self.store = StoreWriter(self.store_dir)
        self.ledger = Ledger(self.ledger_path)
        self.fuel_library = fuel_library

        self.resolver = resolver or self._default_resolver()
        self.verifier = verifier or self._default_verifier()
        #: Waves / records whose maps could not be written (store file locked by a
        #: concurrent reader).  Labels are unaffected; reported in the summary so a
        #: map gap is never silent.
        self.maps_skipped_waves = 0
        self.maps_skipped_records = 0

        # Wave size follows the verifier's ACTUAL worker count, not the raw
        # ``[produce] workers`` knob: with ``use_all_cores`` the verifier builds
        # one worker per logical core (23 on this 8P+16E box, host_reserve=1), so
        # waves must fill to 23, not the legacy 8.  A fake/stub verifier without
        # ``n_workers`` (some tests) falls back to the configured knob.
        self.workers = (
            int(getattr(self.verifier, "n_workers", 0))
            or int(self.produce.workers)
            or 8
        )
        # Wall-time observability by CPU class (P|E) — E-core chains run ~30-40%
        # slower; accounting only, no behavioural effect (plan 5.4).
        self._core_class_chains: Counter[str] = Counter()
        self._core_class_wall_s: dict[str, float] = {}
        #: keyed ``"<pair>|<feed>|<elite_objective>"`` — the objective belongs in
        #: the key: two strata of the same cell may select their parents by
        #: different objectives and must not share a cached pool.
        self._elite_cache: dict[str, list[tuple[str, Pattern]]] = {}
        # Fail at CONSTRUCTION on a bad elite_objective, not mid-wave: a typo in
        # a frontier deck must not surface as a dead exploit arm hours in.
        self._elite_objective(None)
        for _s in self.produce.strata:
            self._elite_objective(_s)
        # rule_biased generator: per-(pair, N, depth) worst-decile RM1 cut.
        self._rule_bias_cut: dict[tuple, float] = {}
        self._wave_counter = 0
        #: Cells already seeded into the resolver's level-1 promoted cache this
        #: run (:meth:`_maybe_promote`).  Only guards WITHIN a wave — once the
        #: cache holds the cell, the next wave's resolve returns level 1 and the
        #: promote condition is false anyway.
        self._promoted_cells: set[CaseKey] = set()

    # -- construction helpers ---------------------------------------------- #
    def _resolve(self, path_str: str | Path) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (self._base / p)

    def _run_library_id(self) -> str:
        """The single fuel library the run's strata share (drives resolver routing).

        A produce run builds ONE resolver + ONE verifier, so a kit is
        single-library (the multi-PC kit assigns cells from one band -> one
        library).  When strata disagree (or none are configured) fall back to the
        deck's ``[model].library_id``.
        """
        libs = {s.library for s in self.produce.strata if getattr(s, "library", None)}
        if len(libs) == 1:
            return next(iter(libs))
        return self.cfg.model.library_id or "ga80"

    def _default_resolver(self) -> CaseAssetResolver:
        library_id = self._run_library_id()
        # PER-LIBRARY ROUTING: a paramA stratum (kit PC assigned a high band) must
        # resolve against the DESIGN PACKAGE — its own bases/cores/lib, the
        # registry type_id->alias bridge, the package %GEN_DIM dims, and NO ga80
        # template_fallbacks — exactly as a paramA curriculum cell does.  Reuses
        # the shared factory; ga80 keeps its byte-identical path below.
        if is_paramA_library(self.cfg, library_id):
            return build_case_resolver(self.cfg, self.fuel_library, library_id)
        package_root = (
            self._resolve(self.cfg.verify.package_root)
            if self.cfg.verify.package_root
            else self.run_dir
        )
        promoted = self._resolve(self.produce.promoted_root)
        neutral = self._resolve(self.produce.neutral_restart) if self.produce.neutral_restart else None
        fallbacks = [str(self._resolve(g)) for g in self.produce.template_fallbacks]
        return CaseAssetResolver(
            package_root,
            promoted,
            neutral_restart=neutral,
            template_fallbacks=fallbacks,
            fuel_library=self.fuel_library,
            synth_root=self._resolve(self.produce.synth_decks_root),
        )

    def _default_verifier(self) -> WaveVerifier:
        if self.dry_run:
            stub = StubEvaluator()
            return WaveVerifier(
                run_dir=self.run_dir,
                # dry-run has no cores to fill; keep the legacy 8-wide stub wave
                # (``workers or 8`` since produce.workers now defaults to 0=auto).
                workers=self.produce.workers or 8,
                max_cycles=self.produce.max_cycles,
                consecutive=self.produce.consecutive,
                evaluator_factory=lambda worker_id, cpu_core: stub,
                resolver=self.resolver,
                purge_intermediate=self.produce.purge_intermediate,
            )
        # PER-LIBRARY ROUTING: derive the live MASTER package_root + reload-deck
        # %GEN_DIM gate from the (already-built) resolver so a paramA run points at
        # the design package and validates the paramA dims; ga80 keeps
        # [verify].package_root + the default dims (byte-identical to before).
        if is_paramA_library(self.cfg, self._run_library_id()):
            package_root = getattr(self.resolver, "package_root", None)
            library_dims = tuple(getattr(self.resolver, "library_dims", LIBRARY_DIMS))
        else:
            package_root = (
                self._resolve(self.cfg.verify.package_root)
                if self.cfg.verify.package_root
                else None
            )
            library_dims = LIBRARY_DIMS
        return WaveVerifier(
            run_dir=self.run_dir,
            package_root=package_root,
            executable=self.cfg.master.executable,
            workers=self.produce.workers,          # 0 = auto (fill the core pool)
            use_all_cores=self.produce.use_all_cores,
            host_reserve=self.produce.host_reserve,
            timeout=self.produce.chain_timeout,
            max_cycles=self.produce.max_cycles,
            consecutive=self.produce.consecutive,
            tolerances=self.cfg.master.tolerances,
            keep_success=self.cfg.master.keep_success,
            resolver=self.resolver,
            library_dims=library_dims,
            purge_intermediate=self.produce.purge_intermediate,
            harvest_maps=bool(getattr(self.cfg.verify, "harvest_maps", False)),
        )

    # -- resume: replay ledger + store ------------------------------------- #
    def _reconstruct(self, states: dict[str, _StratumState]) -> tuple[set[str], int]:
        """Rebuild the dedup set + per-stratum produced counters (crash-safe).

        The per-stratum ``produced`` counter is the STORE's CONVERGED count — the
        same ground truth the curriculum gate reads (:meth:`Curriculum._converged_count`,
        ``dataset=='P' & converged==True``).  Counting converged labels only (not
        every valid label) keeps ``resumed N/target`` in lockstep with the store so
        a cell that has produced non-converged labels can never satisfy its own
        "done" check while the curriculum still sees it short of target — the
        off-by-one relaunch loop.  The dedup set still tracks EVERY evaluated
        ``record_id`` (converged, non-converged, or error) so nothing is re-run.
        """

        store_ids: set[str] = set()
        store_produced: Counter[str] = Counter()
        if self.produce.resume and self.store.records_path.exists():
            try:
                df = StoreReader(self.store_dir).records
            except (FileNotFoundError, OSError):
                df = None
            if df is not None and len(df):
                store_ids = set(df["record_id"].astype(str))
                p = df[(df["dataset"] == "P") & (df["converged"] == True)]  # noqa: E712
                store_produced = Counter(
                    s for s in p["stratum"].tolist() if isinstance(s, str)
                )

        ledger_done: dict[str, str] = {}
        ledger_error: set[str] = set()
        duplicates = 0
        if self.produce.resume:
            last_status: dict[str, str] = {}
            last_stratum: dict[str, str] = {}
            for row in Ledger.replay(self.ledger_path):
                rid = str(row.get("record_id", ""))
                if not rid:
                    if row.get("status") == "dup":
                        duplicates += 1
                    continue
                status = str(row.get("status", ""))
                last_status[rid] = status
                if row.get("stratum"):
                    last_stratum[rid] = str(row["stratum"])
            for rid, status in last_status.items():
                if status == "done":
                    ledger_done[rid] = last_stratum.get(rid, "")
                elif status == "error":
                    ledger_error.add(rid)

        dedup = store_ids | set(ledger_done) | ledger_error
        # ``produced`` is the STORE's converged count, full stop: the store row is
        # committed (atomically, per wave) BEFORE its ledger "done" line, so a
        # converged label is always already in ``store_produced``.  The ledger
        # cannot distinguish converged from non-converged "done" rows, so folding
        # ledger_done into a converged counter would miscount non-convergences as
        # produced — exactly the drift this fix removes.  ledger_done still feeds
        # ``dedup`` above so a completed chain is never re-evaluated.
        for name, state in states.items():
            state.produced = store_produced.get(name, 0)
        return dedup, duplicates

    # -- generators -------------------------------------------------------- #
    def _elite_objective(self, strat: StratumConfig | None) -> str:
        """The validated ``elite_perturb`` parent-selection objective for ``strat``.

        Resolution order: per-stratum ``elite_objective`` -> ``[produce]
        elite_objective`` -> ``"cyclen"`` (legacy).  Raises on an unknown id — a
        knob that silently means "legacy" when misspelled is how a flatness
        campaign ends up running the cycle-length elite rule for a week.
        """

        name = getattr(strat, "elite_objective", None) if strat is not None else None
        if not name:
            name = getattr(self.produce, "elite_objective", None)
        name = str(name or "cyclen").strip().lower()
        if name not in ELITE_OBJECTIVES:
            where = f"[[produce.strata]] {strat.name!r}" if strat is not None else "[produce]"
            raise ValueError(
                f"{where} elite_objective={name!r} is not available; choose one of "
                f"{list(ELITE_OBJECTIVES)} ('cyclen' = legacy top-cyclen parents, "
                f"'flat'/'flat_feasible' = flattest node_peak parents)."
            )
        return name

    def _elites_for(
        self, pair: str, feed: int, objective: str = "cyclen"
    ) -> list[tuple[str, Pattern]]:
        """``(record_id, Pattern)`` elite parents for ``elite_perturb``.

        ``objective`` picks WHICH converged store rows are "elite"; see
        :data:`ELITE_OBJECTIVES`.  The returned list is what
        :meth:`_elite_perturb` samples uniformly, so this ordering IS the exploit
        arm's search neighbourhood.
        """

        key = f"{pair}|{feed}|{objective}"
        if key in self._elite_cache:
            return self._elite_cache[key]
        elites: list[tuple[str, Pattern]] = []
        if self.store.records_path.exists():
            try:
                df = StoreReader(self.store_dir).records
            except (FileNotFoundError, OSError):
                df = None
            if df is not None and len(df):
                pool = (
                    self._cyclen_elite_pool(df, pair)
                    if objective == "cyclen"
                    else self._flat_elite_pool(df, pair, feed, objective)
                )
                if pool is not None and len(pool):
                    for _, row in pool.iterrows():
                        try:
                            pat = unpack_pattern(str(row["pattern"]))
                        except (ValueError, KeyError):
                            continue
                        elites.append((str(row["record_id"]), pat))
        self._elite_cache[key] = elites
        return elites

    @staticmethod
    def _cyclen_elite_pool(df: Any, pair: str) -> Any:
        """LEGACY pool: top-``_ELITE_POOL`` converged rows by ``cyclen`` DESCENDING.

        Byte-identical to the pre-fix selection, including the pair-wide fallback
        (``same if len(same) else conv``) when the pair has no converged row yet.
        Correct for a cycle-length campaign and WRONG for a flatness one — the
        reason ``elite_objective`` exists.
        """

        conv = df[df["converged"] == True]  # noqa: E712
        same = conv[conv["case_pair"] == pair]
        pool = same if len(same) else conv
        if not len(pool):
            return None
        return pool.sort_values("cyclen", ascending=False).head(_ELITE_POOL)

    def _flat_elite_pool(self, df: Any, pair: str, feed: int, objective: str) -> Any:
        """FLATNESS pool: converged, ``node_peak``-labelled rows, FLATTEST FIRST.

        Three deliberate departures from the legacy rule, each measured on the
        E1_E2 case (2026-07-31):

        1. **Ordering** — ``node_peak`` ascending, ``map_cov`` ascending as the
           tie-break.  Under the legacy ``cyclen``-descending rule the E1_E2
           elite set was 100% feed-133/141 high-cyclen rows and EVERY flat_power
           campaign winner (node_peak ~1.23-1.28 at cyclen ~632) fell below the
           cyclen cut, so ``elite_perturb`` never perturbed the flattest cores.
        2. **Label requirement** — a row with no ``node_peak`` has NO objective
           value (that is a different fact from having a bad one), so it is
           excluded rather than sorted to the end.
        3. **Feed preference, then pair-wide fallback within the pair** — the
           frontier best is ``E1_E2 / f121`` while the legacy elites were feed
           133/141.  A cross-feed parent must be feed-morphed by 3-5 fresh units
           (:func:`_morph_feed`) BEFORE ``mutate`` ever runs, which dismantles
           the fresh-placement structure that made the parent flat; the child is
           then barely a perturbation of its parent.  Same-feed parents are
           therefore preferred, and the fallback is only to other feeds OF THE
           SAME PAIR — never to another pair, whose batch types would not even
           survive ``validate_case``.  When nothing qualifies this returns
           ``None`` and the draw degrades LOUDLY to random.
        """

        conv = df[df["converged"] == True]  # noqa: E712
        pool = conv[conv["case_pair"] == pair]
        if not len(pool) or "node_peak" not in pool.columns:
            return None
        peak = pd.to_numeric(pool["node_peak"], errors="coerce").to_numpy(dtype=float)
        pool = pool[np.isfinite(peak)]
        if not len(pool):
            return None
        if objective == "flat_feasible":
            pool = pool[self._flat_feasible_mask(pool)]
            if not len(pool):
                return None
        if "feed" in pool.columns:
            same_feed = pool[
                pd.to_numeric(pool["feed"], errors="coerce").to_numpy(dtype=float)
                == float(feed)
            ]
            if len(same_feed):
                pool = same_feed
        keys = [c for c in _FLAT_ELITE_KEYS if c in pool.columns]
        return pool.sort_values(
            keys, ascending=True, kind="mergesort", na_position="last"
        ).head(_ELITE_POOL)

    def _flat_feasible_mask(self, pool: Any) -> Any:
        """Boolean mask: rows passing the ``flat_power`` CBC / F_q / |AO| gates.

        Limits come from the ONE definition
        (:func:`lpopt.search.campaign.feasibility_limits_for` at
        ``objective="flat_power"``), so a produce kit and the campaign that will
        later consume its labels agree on what "feasible" means.

        F_r and ``max_pin_burnup`` are NOT applied.  F_r under ``flat_power`` is a
        deck-level SAFETY gate whose live value is the per-cell map-head
        bias-corrected ``flat_power_spec.fr_gate``, which exists only inside a
        campaign — screening produce parents at the uncorrected deck number would
        apply a gate the campaign itself does not use.  ``max_pin_burnup`` is
        skipped because most produce rows simply do not carry it: MASTER
        adjudicates it, and gating on a value that is usually absent would reject
        nearly the whole parent pool (note that
        :func:`~lpopt.search.campaign.is_feasible`'s pin-BU guard is None-tolerant
        by intent but NOT NaN-tolerant, and a parquet round-trip turns a missing
        float into NaN).  These are PARENTS for mutation, not results:
        over-filtering them costs coverage and buys nothing, since every child is
        verified on its own merits anyway.
        """

        from .campaign import feasibility_limits_for

        limits = feasibility_limits_for(self.cfg.acquisition, "flat_power")
        mask = np.ones(len(pool), dtype=bool)
        for col, limit_key in _FLAT_ELITE_GATES:
            lim = limits.get(limit_key)
            if lim is None:
                continue
            if col not in pool.columns:
                return np.zeros(len(pool), dtype=bool)
            vals = pd.to_numeric(pool[col], errors="coerce").to_numpy(dtype=float)
            mask &= np.isfinite(vals) & (vals <= float(lim))
        return mask

    def _generate(
        self, state: _StratumState, rng: random.Random
    ) -> tuple[Pattern, str, str | None, float, str] | None:
        """Draw one candidate.

        Returns ``(pattern, generator, parent_id, split_a, pair)``.  ``pair`` is
        the stratum's decision variable (an external choice, never derived from
        the genome — plan 6.1), so record_id / CaseKey / enrichment all use it
        consistently even when a coincidentally single-type pattern is drawn.
        """

        strat = state.cfg
        if not strat.pairs:
            return None
        pair = rng.choice(list(strat.pairs))
        batches = case_batches(pair)
        feed = int(strat.feed)
        n_fresh = (feed - 1) // 4
        lo, hi = _split_range(strat.split_w1)
        w1 = rng.uniform(lo, hi)
        center = batches[0] if strat.center_batch == "auto" else strat.center_batch

        gen = _weighted_choice(state.effective_generators, rng)
        parent: str | None = None

        try:
            if gen == "elite_perturb":
                result = self._elite_perturb(rng, pair, feed, n_fresh, strat, batches)
                if result is None:
                    # LOUD FALLBACK: the ledger keeps generator='random' (it IS a
                    # random draw — truthful), but the operator must be told that
                    # the exploit arm silently became a second random arm.
                    self._note_elite_fallback(state, pair, feed, strat)
                    gen = "random"  # no elites yet -> honest random fallback
                    genome = random_genome(
                        rng, pair, n_fresh,
                        max_shuffle_depth=strat.max_shuffle_depth,
                        allow_single_cycle_discharge=strat.allow_single_cycle_discharge,
                    )
                else:
                    genome, parent = result
            elif gen == "heuristic":
                genome = self._heuristic_genome(rng, pair, n_fresh, strat, batches)
            elif gen == "rule_biased":
                genome = self._rule_biased_genome(rng, pair, n_fresh, strat)
            else:  # random
                gen = "random"
                genome = random_genome(
                    rng, pair, n_fresh,
                    max_shuffle_depth=strat.max_shuffle_depth,
                    allow_single_cycle_discharge=strat.allow_single_cycle_discharge,
                )
            # A stratum's feed is a fixed decision variable: every candidate must
            # carry exactly its N fresh units.  mutate()/elite feed-morph can drift
            # N by +-1 (a feed move), which would give the pattern a weighted feed
            # 1+4N' != the stratum feed and fail pattern.validate_case downstream.
            genome = self._pin_feed(genome, rng, pair, n_fresh, strat, batches)
            genome = _apply_split(genome, rng, batches, w1, center)
            pattern = genome.to_pattern()
        except GenomeError:
            return None

        split_a = _pattern_split(pattern, batches[0])
        return pattern, gen, parent, split_a, pair

    def _note_elite_fallback(
        self, state: _StratumState, pair: str, feed: int, strat: StratumConfig
    ) -> None:
        """Count — and ONCE per stratum, warn about — an elite_perturb degradation.

        The pre-fix driver degraded to ``random`` in total silence: the ledger's
        ``generator`` field said "random" (truthfully — it was a random draw),
        and nothing anywhere said that the stratum's ``elite_perturb`` WEIGHT had
        been spent on it.  A stratum could therefore run at 60% "exploit" and
        produce 100% random data with no signal at all.  The count lands in the
        summary line; the first occurrence logs the reason.
        """

        state.elite_fallback_random += 1
        if state.warned_elite_fallback:
            return
        state.warned_elite_fallback = True
        objective = self._elite_objective(strat)
        reason = (
            "no store row qualified as an elite parent"
            if not self._elites_for(pair, feed, objective)
            else "the chosen elite parent could not be feed-morphed/mutated to the "
                 "stratum N"
        )
        self._log(
            f"[produce][WARNING] stratum {state.name!r}: elite_perturb DEGRADED to "
            f"random for {pair}/feed{feed} — {reason} "
            f"(elite_objective={objective!r}). These draws are ledgered as "
            f"generator='random'; the count is reported as elite_fallback_random "
            f"in the summary."
        )

    def _pin_feed(
        self,
        genome: GeneralOrbitGenome,
        rng: random.Random,
        pair: str,
        n_fresh: int,
        strat: StratumConfig,
        batches: tuple[str, str],
    ) -> GeneralOrbitGenome:
        """Return a genome guaranteed to have exactly ``n_fresh`` fresh units.

        If a generator (mutate/elite-morph) drifted the fresh-unit count, re-morph
        back to the stratum's N with bounded retries; if that cannot land it,
        fall back to an honest ``random_genome`` at the target N (plan 6.1 — feed
        is an external decision variable, never derived from the drawn genome).
        """

        if genome.n_fresh == n_fresh:
            return genome
        for _ in range(4):
            morphed = _morph_feed(genome, rng, n_fresh, batches)
            if morphed is not None and morphed.n_fresh == n_fresh:
                return morphed
        return random_genome(
            rng, pair, n_fresh,
            max_shuffle_depth=strat.max_shuffle_depth,
            allow_single_cycle_discharge=strat.allow_single_cycle_discharge,
        )

    def _heuristic_genome(
        self,
        rng: random.Random,
        pair: str,
        n_fresh: int,
        strat: StratumConfig,
        batches: tuple[str, str],
    ) -> GeneralOrbitGenome:
        rule = rng.choice(("ring", "checker", "radial"))
        target = heuristic_fresh_set(rule, n_fresh)
        best: GeneralOrbitGenome | None = None
        best_overlap = -1
        for _ in range(24):
            genome = random_genome(
                rng, pair, n_fresh,
                max_shuffle_depth=strat.max_shuffle_depth,
                allow_single_cycle_discharge=strat.allow_single_cycle_discharge,
            )
            overlap = len(genome.fresh_units & target)
            if overlap > best_overlap:
                best_overlap = overlap
                best = genome
        assert best is not None
        if rng.random() < 0.5:
            try:
                best = mutate(best, rng, rng.randint(1, 3), batches=batches)
            except GenomeError:
                pass
        return best

    # -- G4 rule_biased (OPTIONAL, config-gated, default OFF) --------------- #
    def _rule_bias_threshold(self, rng: random.Random, pair: str, n_fresh: int,
                             strat: StratumConfig) -> float:
        """Worst-decile RM1 cut for ``(pair, n_fresh)``, from random draws.

        The cut is the ``rule_bias_percentile`` (default 90th) percentile of the
        metric over ``rule_bias_calib`` plain ``random_genome`` draws, so it is a
        decile of the sampler's OWN distribution in this stratum rather than a
        transplanted absolute count.  Cached per ``(pair, n_fresh, depth)``.
        """
        key = (pair, int(n_fresh), int(strat.max_shuffle_depth),
               bool(strat.allow_single_cycle_discharge))
        cached = self._rule_bias_cut.get(key)
        if cached is not None:
            return cached
        metric = _rule_bias_metric(self.produce)
        n_calib = max(8, int(getattr(self.produce, "rule_bias_calib", 128)))
        values: list[float] = []
        for _ in range(n_calib):
            try:
                genome = random_genome(
                    rng, pair, n_fresh,
                    max_shuffle_depth=strat.max_shuffle_depth,
                    allow_single_cycle_discharge=strat.allow_single_cycle_discharge,
                )
                values.append(metric(genome.to_pattern()))
            except GenomeError:
                continue
        pct = float(getattr(self.produce, "rule_bias_percentile", 90.0))
        pct = min(100.0, max(0.0, pct))
        cut = (float(np.percentile(values, pct)) if values else float("inf"))
        self._rule_bias_cut[key] = cut
        self._log(
            f"[produce] rule_biased generator: {getattr(self.produce, 'rule_bias_metric', 'rm1')} "
            f"p{pct:g} cut = {cut:.2f} for {pair} N={n_fresh} "
            f"(from {len(values)} random draws) — a BIAS (bounded retries), not a filter"
        )
        return cut

    def _rule_biased_genome(
        self,
        rng: random.Random,
        pair: str,
        n_fresh: int,
        strat: StratumConfig,
    ) -> GeneralOrbitGenome:
        """G4 — ``random`` draws with the worst RM1 decile rejected (rule R-03).

        Engineering rule R-03 ("fewer fresh face neighbours -> lower local
        peaking") measured within-cell rho ``+0.085`` against ``node_peak``
        (``+0.235`` for the inboard variant), so the tail of high fresh-fresh
        face adjacency is the part of the DoE least worth spending MASTER time
        on.  Rejection is BOUNDED (``rule_bias_tries``): after that many failures
        the last draw is returned anyway.  A heuristic must not be able to
        truncate the search space — that is the source report's own McFLOP /
        Ring-of-Fire lesson — so this generator only shifts sampling density.
        """
        metric = _rule_bias_metric(self.produce)
        cut = self._rule_bias_threshold(rng, pair, n_fresh, strat)
        tries = max(1, int(getattr(self.produce, "rule_bias_tries", 16)))
        genome: GeneralOrbitGenome | None = None
        for _ in range(tries):
            genome = random_genome(
                rng, pair, n_fresh,
                max_shuffle_depth=strat.max_shuffle_depth,
                allow_single_cycle_discharge=strat.allow_single_cycle_discharge,
            )
            if metric(genome.to_pattern()) <= cut:
                return genome
        assert genome is not None
        return genome                      # bounded retries: never starve

    def _elite_perturb(
        self,
        rng: random.Random,
        pair: str,
        feed: int,
        n_fresh: int,
        strat: StratumConfig,
        batches: tuple[str, str],
    ) -> tuple[GeneralOrbitGenome, str] | None:
        elites = self._elites_for(pair, feed, self._elite_objective(strat))
        if not elites:
            return None
        rid, pattern = rng.choice(elites)
        try:
            genome = GeneralOrbitGenome.from_pattern(
                pattern,
                max_shuffle_depth=max(2, strat.max_shuffle_depth),
                allow_single_cycle_discharge=True,
            )
        except GenomeError:
            return None
        morphed = _morph_feed(genome, rng, n_fresh, batches)
        if morphed is None:
            return None
        try:
            morphed = mutate(morphed, rng, rng.randint(2, 8), batches=batches)
        except GenomeError:
            pass
        return morphed, rid

    def _enrichment(
        self, pair: str, split_a: float, library_id: str
    ) -> tuple[float | None, float | None]:
        """Nominal ``(e_core, e_split)`` of a case at feed split ``split_a``.

        A 2-type case is the unchanged ``pair_e_core(a, b, split_a)`` /
        ``|e_a - e_b|``.  A 3-type graded case takes the composition mean at
        ``(split_a, rest split evenly)`` — the same composition
        :func:`_apply_split` draws — and the spread as ``max - min`` over all
        three members (which is exactly ``|e_a - e_b|`` when there are two).
        """
        if self.fuel_library is None:
            return None, None
        parts = [p for p in pair.split("_") if p]
        members = parts if len(parts) >= 2 else [parts[0], parts[0]]
        rest = len(members) - 1
        fracs = [split_a] + [(1.0 - split_a) / rest] * rest
        try:
            e_core = float(_case_e_core(
                self.fuel_library, members, library_id, fracs))
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            return None, None
        try:
            enrs = [self.fuel_library.get(m, library_id).u_avg_enrichment
                    for m in members]
            e_split = (max(enrs) - min(enrs)
                       if all(e is not None for e in enrs) else None)
        except (KeyError, ValueError, TypeError):
            e_split = None
        return e_core, e_split

    # -- QC ---------------------------------------------------------------- #
    def _qc(self, state: _StratumState) -> None:
        """Stratum QC (plan 5.4 / 4.6 정직 회계), emitted as WARNING logs.

        Only a genuine *harness* defect (staging / deck / resolver / timeout /
        genome error) halts a stratum.  A ``non_finite_flux`` watchdog kill is an
        HONEST NEGATIVE label — a property of the drawn pattern, like a
        non-convergence — so it never counts toward the halt rule and only feeds
        the (non-halting) rebalancing advisory below.  This is what keeps a
        legitimately NaN-heavy low-feed stratum (P1: 20-40% non-finite) alive
        instead of being crippled by the old "all errors are exceptions" rule.
        """

        n = state.attempts
        if n == 0:
            return
        harness_rate = state.harness_error / n
        honest_neg_rate = (state.nonfinite + state.nonconverged) / n

        # HALT rule: real harness faults only.
        if harness_rate > 0.10 and n >= 20 and not state.warned_err:
            state.warned_err = True
            state.stalled = True
            self._log(
                f"[produce][WARNING] stratum {state.name!r}: harness-error rate "
                f"{harness_rate:.0%} (>10%) after {n} chains -> HALT stratum "
                "(deck/asset inspection required)"
            )
        # Advisory (no halt): a divergent/nonconverging region -> shift the mix
        # toward the structured priors, but keep producing honest negatives.
        if n >= 100 and honest_neg_rate > 0.50 and not state.warned_rebalance:
            state.warned_rebalance = True
            state.effective_generators = {"heuristic": 0.7, "random": 0.2, "elite_perturb": 0.1}
            self._log(
                f"[produce][WARNING] stratum {state.name!r}: non-finite+nonconverged "
                f"rate {honest_neg_rate:.0%} (>50%) after {n} chains -> shifting to a "
                "heuristic-centred generator mix (rebalancing; stratum continues)"
            )

    # -- main loop --------------------------------------------------------- #
    def run(self, max_chains: int | None = None) -> ProduceSummary:
        states: dict[str, _StratumState] = {}
        for strat in self.produce.strata:
            state = _StratumState(cfg=strat)
            state.effective_n_target = int(strat.n_target)
            state.effective_generators = _normalize_mix(strat.generators)
            states[strat.name] = state

        if not states:
            self._log("[produce] no [[produce.strata]] configured; nothing to do")
            return self._summary(states, 0, 0, 0, 0, 0, 0)

        dedup, resumed_dups = self._reconstruct(states)
        total_chains = 0
        total_dups = resumed_dups
        conv_total = nonconv_total = nonfinite_total = harness_total = 0

        self._log(
            f"[produce] campaign={self.produce.campaign!r} dry_run={self.dry_run} "
            f"strata={len(states)} store={self.store_dir} ledger={self.ledger_path}"
        )
        for state in states.values():
            self._log(
                f"[produce]   stratum {state.name!r}: resumed {state.produced}/"
                f"{state.effective_n_target}"
            )

        inflight: set[str] = set()
        while True:
            if max_chains is not None and total_chains >= max_chains:
                break
            state = self._next_stratum(states)
            if state is None:
                break

            need = state.remaining
            wave_target = min(self.workers, need)
            if max_chains is not None:
                wave_target = min(wave_target, max_chains - total_chains)
            if wave_target <= 0:
                break

            wave: list[tuple[WaveEntry, str, str, str | None]] = []
            attempts = 0
            while len(wave) < wave_target and attempts < wave_target * _GEN_ATTEMPT_FACTOR:
                attempts += 1
                drawn = self._generate(state, self.rng)
                if drawn is None:
                    continue
                pattern, gen, parent, split_a, pair = drawn
                rid = compute_record_id(
                    pattern.canonical(), state.cfg.library, pair, PRODUCE_DECK_KNOBS
                )
                if rid in dedup or rid in inflight:
                    total_dups += 1
                    self.ledger.append(
                        record_id="", stratum=state.name, generator=gen,
                        parent_record_id=parent, status="dup", restart_provenance="",
                    )
                    continue
                inflight.add(rid)
                case_key = CaseKey(pair, int(state.cfg.feed))
                resolved = self.resolver.resolve(case_key)
                e_core, e_split = self._enrichment(pair, split_a, state.cfg.library)
                meta = {
                    "stratum": state.name,
                    "generator": gen,
                    "parent_record_id": parent,
                    "library_id": state.cfg.library,
                    "e_core": e_core,
                    "e_split": e_split,
                    "split_a": split_a,
                    "record_id": rid,
                }
                entry = WaveEntry(
                    pattern=pattern, case_key=case_key, resolved_assets=resolved, meta=meta
                )
                wave.append((entry, rid, gen, parent))

            if not wave:
                state.stalled = True
                self._log(
                    f"[produce][WARNING] stratum {state.name!r}: generators "
                    "exhausted the unique-candidate space -> marking stalled"
                )
                continue

            # ledger: running
            for entry, rid, gen, parent in wave:
                self.ledger.append(
                    record_id=rid, stratum=state.name, generator=gen,
                    parent_record_id=parent, status="running",
                    restart_provenance=entry.resolved_assets.restart_provenance,
                )

            outcomes = self.verifier.evaluate_wave([entry for entry, *_ in wave])

            records: list[CanonicalRecord] = []
            wave_maps: dict[str, Any] = {}
            for (entry, rid, gen, parent), outcome in zip(wave, outcomes, strict=True):
                record = outcome_to_record(
                    outcome,
                    dataset="P",
                    library_id=entry.meta["library_id"],
                    stratum=state.name,
                    generator=gen,
                    parent_record_id=parent,
                    # Per-stratum campaign override (multi-PC produce kits set one
                    # stratum per curriculum cell with campaign == cell id) falls
                    # back to the run-wide [produce] campaign when unset.
                    campaign=(state.cfg.campaign or self.produce.campaign),
                    e_core=entry.meta.get("e_core"),
                    e_split=entry.meta.get("e_split"),
                )
                records.append(record)
                # harvest_maps graft: keep the (4,9,9) EDIT5 stack keyed by record_id
                # (maps_key == record_id) so high-band produce runs also feed node_peak.
                om = getattr(outcome, "maps", None)
                if om is not None:
                    wave_maps[record.record_id] = om
                # High-resolution siblings under suffixed keys: the legacy
                # ``<record_id>`` -> (4,9,9) contract stays byte-identical, so every
                # existing maps reader is unaffected.  NOT retroactive — a record
                # produced without these loses the resolution permanently.
                oh = getattr(outcome, "maps_hires", None) or {}
                for _sfx in ("traj", "axial"):
                    _arr = oh.get(_sfx)
                    if _arr is not None:
                        wave_maps[f"{record.record_id}__{_sfx}"] = _arr

            self.store.write_records(records)  # atomic per wave
            if wave_maps:
                # LAST-RESORT GUARD (forensic 20260725): on Windows the maps.npz
                # rename is refused while ANY process holds the file open (an A/B
                # watcher polling it, an analysis script, the scanner).
                # ``_atomic_write`` already retries ~135 s; if even that loses we
                # SKIP this wave's maps rather than kill a multi-hour campaign —
                # the labels are already safe in records.parquet, so the cost is
                # one wave of maps, not the run.  Skips are counted and surfaced.
                try:
                    self.store.write_maps(wave_maps, append=True)
                except (PermissionError, OSError) as exc:
                    self.maps_skipped_waves += 1
                    self.maps_skipped_records += len(wave_maps)
                    safe_print(f"[produce] WARNING maps write failed ({type(exc).__name__}: "
                          f"{exc}); SKIPPED {len(wave_maps)} map(s) for this wave — "
                          f"labels are unaffected. cumulative skipped: "
                          f"{self.maps_skipped_records} map(s) in "
                          f"{self.maps_skipped_waves} wave(s)")

            for (entry, rid, gen, parent), outcome in zip(wave, outcomes, strict=True):
                # The ledger/store row keeps status="error" for a non_finite_flux
                # kill (it IS invalid as a converged label); the four-way class
                # below only drives QC counting, never the row's validity.
                status = "error" if outcome.status == "error" else "done"
                self.ledger.append(
                    record_id=rid, stratum=state.name, generator=gen,
                    parent_record_id=parent, status=status,
                    failure=outcome.failure or "",
                    restart_provenance=outcome.restart_provenance,
                    # E-core awareness (accounting only): the CPU class + wall time
                    # let post-hoc stats separate P- vs E-core chain durations.
                    core_class=outcome.core_class or "",
                    wall_s=round(float(outcome.wall_s), 3),
                )
                cls = outcome.core_class or "?"
                self._core_class_chains[cls] += 1
                self._core_class_wall_s[cls] = (
                    self._core_class_wall_s.get(cls, 0.0) + float(outcome.wall_s)
                )
                dedup.add(rid)
                inflight.discard(rid)
                state.attempts += 1
                total_chains += 1
                klass = classify_outcome(outcome)
                if klass == "converged":
                    state.converged += 1
                    conv_total += 1
                    state.produced += 1     # target is CONVERGED labels (store truth)
                    # Seed the level-1 cache BEFORE _maybe_purge() below.
                    self._maybe_promote(entry, outcome)
                elif klass == "nonconverged":
                    # An honest non-convergence is a stored, valid label but does
                    # NOT count toward the converged target the curriculum gates on
                    # (else produce would stop one-or-more converged short).
                    state.nonconverged += 1
                    nonconv_total += 1
                elif klass == "nonfinite":
                    # Honest negative label — counted, but not a produced label
                    # (the store row is valid=False) and not a harness fault.
                    state.nonfinite += 1
                    nonfinite_total += 1
                else:  # harness_error
                    state.harness_error += 1
                    harness_total += 1

            self._maybe_purge()
            self._qc(state)
            self._wave_counter += 1
            self._progress(state, total_chains, max_chains)

        summary = self._summary(
            states, self._wave_counter, total_chains,
            conv_total, nonconv_total, nonfinite_total, harness_total,
        )
        summary.duplicates = total_dups
        summary.core_class_chains = dict(self._core_class_chains)
        summary.core_class_wall_s = dict(self._core_class_wall_s)
        self._print_summary(summary)
        return summary

    # -- loop helpers ------------------------------------------------------ #
    def _next_stratum(self, states: dict[str, _StratumState]) -> _StratumState | None:
        unmet = [s for s in states.values() if not s.stalled and s.remaining > 0]
        if not unmet:
            return None
        top_priority = max(s.cfg.priority for s in unmet)
        top = [s for s in unmet if s.cfg.priority == top_priority]
        return top[self._wave_counter % len(top)]

    def _maybe_promote(self, entry: WaveEntry, outcome: WaveOutcome) -> None:
        """Seed the resolver's level-1 promoted cache from a converged chain.

        The ladder's self-improving tier was never populated —
        :meth:`CaseAssetResolver.promote` had no caller anywhere in ``lpopt`` — so
        every chain in a new ``(pair, feed)`` cell re-ran off the cross-feed
        level-2 restart forever.  At the high-feed edge that seed sits far enough
        from the cell's own equilibrium that a quarter of the chains die
        ``non_finite_flux`` while the SAME deck converges cleanly from a
        cell-native restart (feed-grid pathfinder 20260815 §4: 7 chains -> 4 for
        the same 4 labels, and 2 diverged patterns rescued; the promoted seed
        lands on the identical fixed point, |Δcyclen| <= 0.007 EFPD).

        Only a FALLBACK cell is promoted (``fallback_level >= 2``): levels 0/1
        already resolve cell-natively, so a copy there would buy nothing.

        MUST be called before :meth:`_maybe_purge` — the wave purge and the vendor
        chain lifecycle delete the final cycle's work dir, and with it the only
        restart there is to promote.  Best-effort and never fatal: a missed
        promotion costs chains, a raised copy error would cost the whole run.
        """

        if entry.resolved_assets.fallback_level < 2:
            return
        case = entry.case_key
        if case in self._promoted_cells:
            return
        # ``eq_provenance`` is the converged final cycle's {deck, restart, work_dir},
        # captured while the raw EquilibriumResult was still in scope and verified
        # on disk at capture time; absent when the verifier kept no work dir.
        restart = (outcome.eq_provenance or {}).get("restart")
        if not restart:
            return
        try:
            dest = self.resolver.promote(case, restart)
        except (OSError, ValueError) as exc:
            self._log(
                f"[produce][WARNING] level-1 promotion for {case.label} failed "
                f"({type(exc).__name__}: {exc}); the cell keeps resolving at "
                f"level {entry.resolved_assets.fallback_level}"
            )
            return
        self._promoted_cells.add(case)
        self._log(
            f"[produce] promoted {case.label} -> {dest} "
            f"(level-1 cache; was {entry.resolved_assets.restart_provenance})"
        )

    def _maybe_purge(self) -> None:
        if self.produce.purge_case_dirs and self.verifier.cases_dir.exists():
            shutil.rmtree(self.verifier.cases_dir, ignore_errors=True)

    def _progress(self, state: _StratumState, total: int, max_chains: int | None) -> None:
        if not self.progress:
            return
        cap = f"/{max_chains}" if max_chains else ""
        self._log(
            f"[produce] wave {self._wave_counter:>3} | stratum {state.name:<14} "
            f"produced {state.produced:>3}/{state.effective_n_target:<3} | "
            f"conv {state.converged} nonconv {state.nonconverged} "
            f"nonfin {state.nonfinite} herr {state.harness_error} | "
            f"chains {total}{cap}"
        )

    def _summary(
        self,
        states: dict[str, _StratumState],
        waves: int,
        chains: int,
        conv: int,
        nonconv: int,
        nonfinite: int,
        harness_error: int,
    ) -> ProduceSummary:
        strata_rows = [
            {
                "name": s.name,
                "produced": s.produced,
                "n_target": s.effective_n_target,
                "converged": s.converged,
                "nonconverged": s.nonconverged,
                "nonfinite": s.nonfinite,
                "harness_error": s.harness_error,
                "attempts": s.attempts,
                "conv_rate": (s.converged / s.attempts) if s.attempts else None,
                "harness_rate": (s.harness_error / s.attempts) if s.attempts else None,
                "stalled": s.stalled,
                "elite_fallback_random": s.elite_fallback_random,
            }
            for s in states.values()
        ]
        return ProduceSummary(
            campaign=self.produce.campaign,
            dry_run=self.dry_run,
            waves=waves,
            chains=chains,
            converged=conv,
            nonconverged=nonconv,
            nonfinite=nonfinite,
            harness_error=harness_error,
            errors=nonfinite + harness_error,
            duplicates=0,
            store_dir=str(self.store_dir),
            ledger_path=str(self.ledger_path),
            strata=strata_rows,
            elite_fallback_random=sum(s.elite_fallback_random for s in states.values()),
        )

    def _print_summary(self, summary: ProduceSummary) -> None:
        self._log("")
        self._log(f"produce summary  (campaign={summary.campaign}, dry_run={summary.dry_run})")
        self._log(
            f"waves={summary.waves} chains={summary.chains} "
            f"converged={summary.converged} nonconverged={summary.nonconverged} "
            f"nonfinite={summary.nonfinite} harness_error={summary.harness_error} "
            f"duplicates={summary.duplicates} "
            f"elite_fallback_random={summary.elite_fallback_random}"
        )
        header = (
            f"{'STRATUM':16s} {'PRODUCED':>10s} {'CONV':>5s} {'NONCONV':>8s} "
            f"{'NONFIN':>7s} {'HERR':>5s} {'HERR%':>6s} {'ELITE->RND':>11s} "
            f"{'STALLED':>8s}"
        )
        self._log(header)
        self._log("-" * len(header))
        for row in summary.strata:
            herr_pct = (
                f"{row['harness_rate']*100:.0f}" if row["harness_rate"] is not None else "-"
            )
            self._log(
                f"{row['name']:16s} {str(row['produced'])+'/'+str(row['n_target']):>10s} "
                f"{row['converged']:>5d} {row['nonconverged']:>8d} "
                f"{row['nonfinite']:>7d} {row['harness_error']:>5d} {herr_pct:>6s} "
                f"{row.get('elite_fallback_random', 0):>11d} "
                f"{str(row['stalled']):>8s}"
            )
        if summary.core_class_chains:
            def _label(k: str) -> str:
                return {"P": "P-core", "E": "E-core"}.get(k, "unpinned")
            parts = []
            for k in ("P", "E", "?"):
                n = summary.core_class_chains.get(k)
                if not n:
                    continue
                wall = summary.core_class_wall_s.get(k, 0.0)
                parts.append(f"{_label(k)} {n} ({wall / n:.1f}s/chain)")
            if parts:
                self._log("cpu   : " + " | ".join(parts))
        self._log(f"store : {summary.store_dir}")
        self._log(f"ledger: {summary.ledger_path}")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _split_range(split_w1: Sequence[float]) -> tuple[float, float]:
    vals = list(split_w1) if split_w1 else [0.5]
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    return float(min(vals)), float(max(vals))


def _normalize_mix(generators: dict[str, float]) -> dict[str, float]:
    mix = {k: float(v) for k, v in (generators or {}).items() if float(v) > 0.0}
    if not mix:
        return {"random": 1.0}
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def _weighted_choice(mix: dict[str, float], rng: random.Random) -> str:
    names = list(mix)
    weights = [mix[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def run_produce(
    cfg: LpoptConfig,
    *,
    dry_run: bool = False,
    max_chains: int | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ProduceSummary:
    """Build a :class:`ProduceDriver` from an :class:`LpoptConfig` and run it."""

    driver = ProduceDriver(cfg, dry_run=dry_run, progress=progress, **kwargs)
    return driver.run(max_chains=max_chains)


__all__ = [
    "ELITE_OBJECTIVES",
    "Ledger",
    "ProduceDriver",
    "ProduceSummary",
    "heuristic_fresh_set",
    "run_produce",
]
