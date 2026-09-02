"""F_r boundary micro-verification harness (plan Task B — build + smoke).

The curriculum honest holdouts contain ZERO converged rows near the ``F_r <= 1.55``
feasibility limit (min f_r ~1.656, and only outside the legacy feed-121 corpus), so
the champion has never been shown a near-threshold labelled pattern in the new
(pair, feed) cells.  This module manufactures elite-structured candidates that the
champion PREDICTS to be near the limit, ranks them by the champion's (calibrated)
F_r head, and — when ``--verify`` — runs the top-K through the exact produce/verify
machinery so MASTER returns real near-1.55 labels and parity statistics.

Per target cell it builds a candidate pool from three generators (all reuse the
production genome operators — no new genome logic):

* **legacy_transfer** — the lowest-F_r legacy patterns (store rows ``dataset in
  {A,B}``, ``feed == 121``, converged, ``f_r <= 1.75``, F_r-ascending) parsed to a
  :class:`GeneralOrbitGenome`, feed-morphed to the cell's N via the production
  ``add/remove_fresh_unit`` moves (structure preserved), then relabelled to the
  cell's ``(pair, split)``.  This transfers the *low-peaking structure* the legacy
  corpus discovered into a cell whose labels never reached the boundary.
* **heuristic (G2 checkerboard)** — the production ``heuristic_fresh_set('checker')``
  low-peaking prior, overlap-biased-sampled to the cell N.
* **elite_perturb (G3)** — the production ``mutate`` operator applied to the
  legacy_transfer morphs (a few closed moves), recording the legacy ``parent``.

Every candidate is ``validate()``-d (via ``to_pattern``) and de-duplicated by its
``record_id`` preimage against the store.  Ranking uses the champion backend
(batch CPU predict, F_r calibration ON — the deployed artifact): candidates are
SELECTED by predicted F_r **mean ascending** (we want the most-likely-low-F_r), and
BOTH the mean and the conservative ``mean + risk_z*sigma`` (risk_z=0.25) are
reported.  ``--verify`` harvests the top-K into the store with
``campaign=<cell_id>`` / ``generator="g3_elite_boundary"``; without it only the
candidate list + predictions are written to ``data/reports/boundary_probe_<cell>.json``.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..config import LpoptConfig, StratumConfig
from ..safelog import safe_logger
from ..data.schema import compute_record_id, pack_pattern, unpack_pattern
from ..data.store import StoreReader
from ..vendor.masterrl.domain import CaseKey, Pattern
from ..vendor.masterrl.ga import FRESH_UNIT_COUNT
from .genome import (
    GeneralOrbitGenome,
    GenomeError,
    case_batches,
    mutate,
    random_genome,
)
from .produce import (
    _apply_split,
    _morph_feed,
    _pattern_split,
    heuristic_fresh_set,
)
from .verify import PRODUCE_DECK_KNOBS, WaveEntry, classify_outcome, outcome_to_record

#: Provenance the harvested rows carry (checked by the curriculum's per-cell reads).
BOUNDARY_GENERATOR = "g3_elite_boundary"
#: Conservative risk multiplier applied to sigma for the reported UCB (mean+z*sigma).
DEFAULT_RISK_Z = 0.25
#: Default candidates to verify per cell (one MASTER wave on this box's core pool).
DEFAULT_TOP_K = 16
#: Default candidate-pool size (ranked down to top-K); split across the 3 generators.
DEFAULT_POOL_SIZE = 240
#: Legacy elite seed selection: dataset A/B, feed 121, converged, this F_r ceiling.
LEGACY_FEED = 121
LEGACY_FR_MAX = 1.75
#: Legacy seeds pulled (lowest-F_r first) to morph from.
LEGACY_SEED_POOL = 96
#: F_r lives in surrogate column 0 (see model_api).
_FR_COL = 0

#: Coordinator-invoked default cell list (outward from the support anchor).  The
#: coordinator drives one cell per invocation; this is the canonical roster.
DEFAULT_BOUNDARY_CELLS: tuple[str, ...] = (
    "5.5-5.75_f109",
    "5-5.25_f117",
    "5-5.25_f125",
    "5.25-5.5_f109",
    "5.25-5.5_f117",
    "5.75-6_f101",
    "6-6.25_f101",      # produced but NOT yet learned (quarantined)
    "6.25-6.5_f109",    # produced but NOT yet learned (quarantined)
)

_CELL_RE = re.compile(r"^(?P<lo>\d+(?:\.\d+)?)-(?P<hi>\d+(?:\.\d+)?)_f(?P<feed>\d+)$")


# --------------------------------------------------------------------------- #
# cell resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellSpec:
    """A curriculum cell's decision variables (band, feed, pairs, library)."""

    cell_id: str
    band: tuple[float, float]
    feed: int
    pairs: tuple[str, ...]
    library: str
    learned: bool

    @property
    def n_fresh(self) -> int:
        return (int(self.feed) - 1) // 4

    @property
    def allow_single_cycle_discharge(self) -> bool:
        return self.n_fresh > FRESH_UNIT_COUNT


def resolve_cell(cell_id: str, records: Any, *, state: dict | None = None,
                 splits: dict | None = None) -> CellSpec:
    """Resolve a cell's (band, feed, pairs, library, learned) from state + store.

    ``band`` / ``feed`` parse from the id (``<lo>-<hi>_f<feed>``).  ``pairs`` and
    ``library`` prefer the curriculum ``state.json`` cell record, falling back to
    the store (the cell's produced rows) — essential for the two *unlearned*
    (quarantined) cells whose ``state.json`` ``pairs`` is empty.  ``learned`` is
    False when the cell is in the split's ``quarantined_by_cell`` (produced but
    not yet trained on).
    """
    m = _CELL_RE.match(cell_id)
    if not m:
        raise ValueError(f"unparseable cell id {cell_id!r} (expected <lo>-<hi>_f<feed>)")
    band = (float(m.group("lo")), float(m.group("hi")))
    feed = int(m.group("feed"))

    cell_rec = ((state or {}).get("cells", {}) or {}).get(cell_id, {}) or {}
    pairs = list(cell_rec.get("pairs") or [])
    library = cell_rec.get("library_id")

    sub = records[records["campaign"].astype(str) == cell_id] if len(records) else records
    if not pairs and len(sub):
        pairs = sorted(sub["case_pair"].dropna().astype(str).unique().tolist())
    if not library:
        if len(sub) and sub["library_id"].notna().any():
            library = str(sub["library_id"].dropna().astype(str).mode().iloc[0])
        else:
            library = "ga80"

    quar = ((splits or {}).get("groups", {}) or {}).get("quarantined_by_cell", {})
    learned = cell_id not in set(quar)

    if not pairs:
        raise ValueError(
            f"cell {cell_id!r} has no pairs in state.json nor produced store rows; "
            "cannot generate candidates")
    return CellSpec(cell_id=cell_id, band=band, feed=feed,
                    pairs=tuple(pairs), library=str(library), learned=bool(learned))


# --------------------------------------------------------------------------- #
# candidate pool
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One pooled candidate + its provenance and (filled at rank time) predictions."""

    pattern: Pattern
    pair: str
    generator: str                    # legacy_transfer | heuristic | elite_perturb
    parent_id: str | None
    record_id: str
    split_a: float
    f_r_mean: float = float("nan")     # champion calibrated F_r mean
    f_r_sigma: float = float("nan")    # champion calibrated F_r sigma
    f_r_ucb: float = float("nan")      # mean + risk_z * sigma (conservative)


def legacy_low_fr_seeds(records: Any, *, feed: int = LEGACY_FEED,
                        fr_max: float = LEGACY_FR_MAX,
                        pool: int = LEGACY_SEED_POOL) -> list[tuple[str, Pattern, float]]:
    """The lowest-F_r legacy patterns to morph from (dataset A/B, feed 121, conv)."""
    if not len(records):
        return []
    df = records
    keep = (
        df["dataset"].astype(str).isin(["A", "B"])
        & (df["feed"] == feed)
        & (df["converged"] == True)                                  # noqa: E712
        & df["f_r"].notna()
        & (df["f_r"] <= float(fr_max))
    )
    sub = df[keep].sort_values("f_r", ascending=True).head(int(pool))
    seeds: list[tuple[str, Pattern, float]] = []
    for _, row in sub.iterrows():
        try:
            pat = unpack_pattern(str(row["pattern"]))
        except (ValueError, KeyError):
            continue
        seeds.append((str(row["record_id"]), pat, float(row["f_r"])))
    return seeds


class _PoolBuilder:
    """Accumulates validated, de-duplicated candidates for one cell."""

    def __init__(self, cell: CellSpec, store_ids: set[str],
                 rng: random.Random) -> None:
        self.cell = cell
        self.rng = rng
        self._seen: set[str] = set(store_ids)         # dedup vs store + within pool
        self.candidates: list[Candidate] = []

    def _batches(self, pair: str) -> tuple[str, ...]:
        return case_batches(pair)

    def _pin_feed(self, genome: GeneralOrbitGenome, batches: tuple[str, ...]
                  ) -> GeneralOrbitGenome | None:
        """Guarantee exactly ``cell.n_fresh`` fresh units (feed is a fixed decision
        variable): re-morph a genome a mutation drifted off N (mirrors
        ``ProduceDriver._pin_feed``).  ``None`` when it cannot be re-pinned."""
        if genome.n_fresh == self.cell.n_fresh:
            return genome
        for _ in range(4):
            m = _morph_feed(genome, self.rng, self.cell.n_fresh, batches)
            if m is not None and m.n_fresh == self.cell.n_fresh:
                return m
        return None

    def add(self, genome: GeneralOrbitGenome, pair: str, generator: str,
            parent_id: str | None) -> bool:
        """Feed-pin, split-relabel, compile, validate, dedup, and append one genome."""
        batches = self._batches(pair)
        pinned = self._pin_feed(genome, batches)      # a mutate/morph may drift N
        if pinned is None:
            return False
        w1 = self.rng.uniform(0.3, 0.7)
        try:
            g = _apply_split(pinned, self.rng, batches, w1, batches[0])
            pattern = g.to_pattern()                  # to_pattern() calls validate()
        except GenomeError:
            return False
        rid = compute_record_id(
            pattern.canonical(), self.cell.library, pair, PRODUCE_DECK_KNOBS)
        if rid in self._seen:
            return False
        self._seen.add(rid)
        split_a = _pattern_split(pattern, batches[0])
        self.candidates.append(Candidate(
            pattern=pattern, pair=pair, generator=generator,
            parent_id=parent_id, record_id=rid, split_a=split_a))
        return True

    # -- (a) legacy elite transfer ------------------------------------------ #
    def morph_legacy(self, seed_rid: str, seed_pat: Pattern, pair: str
                     ) -> GeneralOrbitGenome | None:
        """Parse a legacy pattern, feed-morph to the cell N (structure-preserving)."""
        batches = self._batches(pair)
        try:
            g = GeneralOrbitGenome.from_pattern(
                seed_pat, max_shuffle_depth=2, allow_single_cycle_discharge=True)
        except GenomeError:
            return None
        return _morph_feed(g, self.rng, self.cell.n_fresh, batches)

    # -- (b) heuristic checkerboard (G2) ------------------------------------ #
    def checker_genome(self, pair: str) -> GeneralOrbitGenome | None:
        """Overlap-biased random genome closest to the checkerboard fresh set."""
        target = heuristic_fresh_set("checker", self.cell.n_fresh)
        best: GeneralOrbitGenome | None = None
        best_overlap = -1
        for _ in range(24):
            try:
                g = random_genome(
                    self.rng, pair, self.cell.n_fresh, max_shuffle_depth=2,
                    allow_single_cycle_discharge=self.cell.allow_single_cycle_discharge)
            except GenomeError:
                continue
            overlap = len(g.fresh_units & target)
            if overlap > best_overlap:
                best_overlap = overlap
                best = g
        return best


def generate_pool(cell: CellSpec, records: Any, rng: random.Random, *,
                  pool_size: int = DEFAULT_POOL_SIZE,
                  log: Callable[[str], None] | None = None) -> list[Candidate]:
    """Build the de-duplicated candidate pool for a cell (3 generators)."""
    log = log or (lambda _m: None)
    store_ids = set(records["record_id"].astype(str)) if len(records) else set()
    seeds = legacy_low_fr_seeds(records)
    builder = _PoolBuilder(cell, store_ids, rng)

    n_legacy = int(round(pool_size * 0.40))
    n_heur = int(round(pool_size * 0.30))
    n_elite = pool_size - n_legacy - n_heur
    pairs = list(cell.pairs)

    # (a) legacy transfer — keep the morphed (pre-split) genomes to seed (c).
    legacy_morphs: list[tuple[GeneralOrbitGenome, str, str]] = []   # (genome, pair, rid)
    if seeds:
        max_att = max(1, n_legacy) * 40
        att = 0
        while sum(c.generator == "legacy_transfer" for c in builder.candidates) < n_legacy \
                and att < max_att:
            att += 1
            rid, seed_pat, _fr = rng.choice(seeds)
            pair = rng.choice(pairs)
            morph = builder.morph_legacy(rid, seed_pat, pair)
            if morph is None:
                continue
            if builder.add(morph, pair, "legacy_transfer", rid):
                legacy_morphs.append((morph, pair, rid))
    else:
        log("[boundary] no legacy low-F_r seeds found; skipping legacy_transfer")

    # (c) elite_perturb — mutate the legacy morphs (closed moves), record parent.
    if legacy_morphs:
        max_att = max(1, n_elite) * 40
        att = 0
        while sum(c.generator == "elite_perturb" for c in builder.candidates) < n_elite \
                and att < max_att:
            att += 1
            morph, pair, rid = rng.choice(legacy_morphs)
            batches = case_batches(pair)
            try:
                child = mutate(morph, rng, rng.randint(2, 8), batches=batches)
            except GenomeError:
                continue
            builder.add(child, pair, "elite_perturb", rid)

    # (b) heuristic checkerboard.
    max_att = max(1, n_heur) * 40
    att = 0
    while sum(c.generator == "heuristic" for c in builder.candidates) < n_heur \
            and att < max_att:
        att += 1
        pair = rng.choice(pairs)
        g = builder.checker_genome(pair)
        if g is None:
            continue
        builder.add(g, pair, "heuristic", None)

    log(f"[boundary] pool: {len(builder.candidates)} unique candidates "
        f"(legacy={sum(c.generator=='legacy_transfer' for c in builder.candidates)}, "
        f"heuristic={sum(c.generator=='heuristic' for c in builder.candidates)}, "
        f"elite_perturb={sum(c.generator=='elite_perturb' for c in builder.candidates)})")
    return builder.candidates


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def rank_pool(backend: Any, cell: CellSpec, candidates: list[Candidate], *,
              risk_z: float = DEFAULT_RISK_Z, batch_size: int = 512) -> list[Candidate]:
    """Fill each candidate's champion (calibrated) F_r mean/sigma/UCB; sort by mean.

    Selection is by predicted F_r **mean ascending** (most-likely-low-F_r); the
    conservative ``mean + risk_z*sigma`` is also recorded (never used to sort).
    The backend serves the deployed F_r calibration — the served, gated value.
    """
    if not candidates:
        return []
    for start in range(0, len(candidates), int(batch_size)):
        batch = candidates[start:start + int(batch_size)]
        pats = [c.pattern for c in batch]
        cases = [CaseKey(pair=c.pair, feed=int(cell.feed)) for c in batch]
        pred = backend.predict(pats, cases, 0.0)
        mean = np.asarray(pred.mean, dtype=float)[:, _FR_COL]
        sigma = np.asarray(pred.calibrated_std, dtype=float)[:, _FR_COL]
        for c, m, s in zip(batch, mean, sigma):
            c.f_r_mean = float(m)
            c.f_r_sigma = float(s)
            c.f_r_ucb = float(m + risk_z * s)
    ranked = sorted(candidates, key=lambda c: (c.f_r_mean, c.f_r_ucb))
    return ranked


def _distribution(vals: Sequence[float]) -> dict[str, float]:
    a = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if not a.size:
        return {}
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _candidate_row(c: Candidate) -> dict[str, Any]:
    return {
        "record_id": c.record_id,
        "pair": c.pair,
        "generator": c.generator,
        "parent_record_id": c.parent_id,
        "split_a": round(c.split_a, 4),
        "f_r_mean": round(c.f_r_mean, 4),
        "f_r_sigma": round(c.f_r_sigma, 4),
        "f_r_mean_plus_zsigma": round(c.f_r_ucb, 4),
        "pattern": pack_pattern(c.pattern),
    }


def build_report(cell: CellSpec, champion_dir: str, ranked: list[Candidate],
                 top: list[Candidate], *, risk_z: float, verified: dict | None) -> dict:
    gen_counts: dict[str, int] = {}
    for c in ranked:
        gen_counts[c.generator] = gen_counts.get(c.generator, 0) + 1
    report = {
        "schema": "boundary_probe_v1",
        "cell": cell.cell_id,
        "band": list(cell.band),
        "feed": cell.feed,
        "pairs": list(cell.pairs),
        "library": cell.library,
        "learned": cell.learned,
        "quarantined": not cell.learned,
        "champion_model_dir": str(champion_dir),
        "risk_z": risk_z,
        "generator": BOUNDARY_GENERATOR,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_candidates": len(ranked),
        "generator_counts": gen_counts,
        "pool_f_r_mean_distribution": _distribution([c.f_r_mean for c in ranked]),
        "top_k": len(top),
        "top_k_f_r_mean_distribution": _distribution([c.f_r_mean for c in top]),
        "top_k_f_r_ucb_distribution": _distribution([c.f_r_ucb for c in top]),
        "top_candidates": [_candidate_row(c) for c in top],
    }
    if verified is not None:
        report["verified"] = verified
    return report


# --------------------------------------------------------------------------- #
# champion resolution
# --------------------------------------------------------------------------- #
def resolve_champion_dir(cfg: LpoptConfig, base: Path) -> Path:
    """Resolve the champion ensemble dir from the curriculum ``state.json``.

    ``data/curriculum/state.json``'s ``champion_model_dir`` is authoritative (the
    curriculum advances it); falls back to ``[model].model_dir`` when the state
    file is absent.  Never writes it.
    """
    state_path = base / "data" / "curriculum" / "state.json"
    if state_path.is_file():
        try:
            champ = json.loads(state_path.read_text(encoding="utf-8")).get("champion_model_dir")
            if champ:
                p = Path(champ)
                return p if p.is_absolute() else (base / p)
        except (OSError, ValueError):
            pass
    p = Path(cfg.model.model_dir)
    return p if p.is_absolute() else (base / p)


def _load_state_and_splits(base: Path) -> tuple[dict, dict]:
    state_path = base / "data" / "curriculum" / "state.json"
    splits_path = base / "data" / "splits" / "S1.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.is_file() else {}
    return state, splits


# --------------------------------------------------------------------------- #
# verify (produce/verify machinery)
# --------------------------------------------------------------------------- #
def build_cell_driver(cfg: LpoptConfig, cell: CellSpec, n_chains: int, *,
                      fuel_library: Any = None,
                      log: Callable[[str], None] | None = None) -> Any:
    """Build the single-stratum (live) :class:`ProduceDriver` for one cell.

    The synthetic stratum tags the cell's produced rows with ``campaign ==
    <cell_id>`` (the same convention the curriculum / multi-PC kit use), so the
    driver's per-library resolver + core-pinned verifier + main store + ledger are
    reused verbatim.
    """
    import copy
    from .produce import ProduceDriver

    strat = StratumConfig(
        name=cell.cell_id, library=cell.library, campaign=cell.cell_id,
        pairs=list(cell.pairs), feed=int(cell.feed),
        generators={"random": 1.0}, n_target=int(n_chains),
        allow_single_cycle_discharge=cell.allow_single_cycle_discharge,
    )
    cfg_v = copy.copy(cfg)
    cfg_v.produce = copy.copy(cfg.produce)
    cfg_v.produce.strata = [strat]
    cfg_v.produce.campaign = cell.cell_id
    return ProduceDriver(cfg_v, dry_run=False, fuel_library=fuel_library,
                         progress=False, log=log)


def verify_candidates(cfg: LpoptConfig, cell: CellSpec, top: list[Candidate], *,
                      base: Path, fuel_library: Any = None,
                      max_chains: int | None = None, driver: Any = None,
                      log: Callable[[str], None] | None = None) -> dict:
    """Run the selected candidates through the produce/verify machinery.

    Builds (or reuses an injected) single-stratum :class:`ProduceDriver` for the
    cell — its per-library resolver + core-pinned :class:`WaveVerifier` + store +
    ledger — verifies the top-K as one or more MASTER waves, and writes harvested
    rows to the store with ``campaign=<cell_id>`` / ``generator="g3_elite_boundary"``
    (identical provenance to a produce cell).  Returns per-outcome counts + the
    verified F_r parity of the converged rows.  ``driver`` is injectable so the
    wiring is testable without MASTER.
    """
    log = log or (lambda _m: None)
    chains = top if max_chains is None else top[: int(max_chains)]
    if not chains:
        return {"chains": 0, "converged": 0, "nonconverged": 0, "error": 0, "labels": []}

    if driver is None:
        if not cfg.master.executable:
            raise RuntimeError("a --verify run needs [master].executable in the deck")
        driver = build_cell_driver(cfg, cell, len(chains),
                                   fuel_library=fuel_library, log=log)

    # build the wave entries from the selected candidates.
    entries: list[tuple[WaveEntry, Candidate]] = []
    for c in chains:
        case_key = CaseKey(c.pair, int(cell.feed))
        resolved = driver.resolver.resolve(case_key)
        e_core, e_split = driver._enrichment(c.pair, c.split_a, cell.library)
        meta = {
            "stratum": cell.cell_id, "generator": BOUNDARY_GENERATOR,
            "parent_record_id": c.parent_id, "library_id": cell.library,
            "e_core": e_core, "e_split": e_split, "split_a": c.split_a,
            "record_id": c.record_id,
        }
        entries.append((WaveEntry(pattern=c.pattern, case_key=case_key,
                                  resolved_assets=resolved, meta=meta), c))

    workers = max(1, int(getattr(driver.verifier, "n_workers", 1)))
    counts = {"converged": 0, "nonconverged": 0, "error": 0}
    labels: list[dict[str, Any]] = []
    for start in range(0, len(entries), workers):
        wave = entries[start:start + workers]
        for entry, c in wave:
            driver.ledger.append(
                record_id=c.record_id, stratum=cell.cell_id, generator=BOUNDARY_GENERATOR,
                parent_record_id=c.parent_id, status="running",
                restart_provenance=entry.resolved_assets.restart_provenance)
        outcomes = driver.verifier.evaluate_wave([e for e, _ in wave])
        records = []
        for (entry, c), outcome in zip(wave, outcomes):
            rec = outcome_to_record(
                outcome, dataset="P", library_id=cell.library, stratum=cell.cell_id,
                generator=BOUNDARY_GENERATOR, parent_record_id=c.parent_id,
                campaign=cell.cell_id, e_core=entry.meta.get("e_core"),
                e_split=entry.meta.get("e_split"))
            records.append(rec)
            klass = classify_outcome(outcome)
            key = "converged" if klass == "converged" else (
                "nonconverged" if klass == "nonconverged" else "error")
            counts[key] += 1
            actual_fr = None if outcome.fom is None else getattr(outcome.fom, "f_r", None)
            labels.append({
                "record_id": c.record_id, "pair": c.pair, "status": outcome.status,
                "pred_f_r_mean": round(c.f_r_mean, 4), "pred_f_r_ucb": round(c.f_r_ucb, 4),
                "actual_f_r": (round(float(actual_fr), 4) if actual_fr is not None else None),
            })
            status = "error" if outcome.status == "error" else "done"
            driver.ledger.append(
                record_id=c.record_id, stratum=cell.cell_id, generator=BOUNDARY_GENERATOR,
                parent_record_id=c.parent_id, status=status, failure=outcome.failure or "",
                restart_provenance=outcome.restart_provenance)
        driver.store.write_records(records)
    return {"chains": len(entries), **counts, "store_dir": str(driver.store_dir),
            "labels": labels}


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_boundary_probe(cfg: LpoptConfig, cell_id: str, *, top_k: int = DEFAULT_TOP_K,
                       verify: bool = False, pool_size: int = DEFAULT_POOL_SIZE,
                       risk_z: float = DEFAULT_RISK_Z, seed: int = 0,
                       reports_dir: str | Path | None = None,
                       fuel_library: Any = None,
                       log: Callable[[str], None] | None = None) -> dict:
    """Generate + rank (+ optionally verify) one cell's boundary candidates.

    Writes ``data/reports/boundary_probe_<cell>.json`` and returns the report dict.

    **DEPRECATED** (flatness-first program 20260725 §10 STOP): this harness
    MANUFACTURES labels by ranking candidates on the champion's F_r head, i.e. it
    spends MASTER calls to enrich the corpus along the axis the program retired
    from the objective.  It is kept runnable for reproducing the existing
    ``boundary_probe_*.json`` reports and is NOT carried into the flatness
    workstream; ``--verify`` in particular writes F_r-selected rows into the
    store.  Prefer the flat_power campaign for new labels.
    """
    from ..model.model_api import PosValCnnBackend

    # Encoding-safe default logger: a redirected Windows stdout is cp949 and a
    # single em-dash used to raise UnicodeEncodeError mid-run (2026-08-30).
    log = safe_logger(log)
    log("[boundary][DEPRECATED] F_r-ranked label manufacturing (program §10 STOP): "
        "this selects candidates by the champion's F_r head, the axis the "
        "flatness-first program retired. Kept for reproducing existing reports; "
        "use objective='flat_power' for new labels.")
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    store_dir = _resolve(cfg.model.store_dir, base)
    records = StoreReader(store_dir).records

    state, splits = _load_state_and_splits(base)
    cell = resolve_cell(cell_id, records, state=state, splits=splits)
    champ = resolve_champion_dir(cfg, base)
    log(f"[boundary] cell {cell.cell_id}  band={cell.band} feed={cell.feed} "
        f"library={cell.library}  {'LEARNED' if cell.learned else 'UNLEARNED (quarantined)'}")
    log(f"[boundary]   pairs={list(cell.pairs)}")
    log(f"[boundary]   champion={champ.name}")

    rng = random.Random(seed)
    candidates = generate_pool(cell, records, rng, pool_size=pool_size, log=log)
    if not candidates:
        raise RuntimeError(f"cell {cell_id!r}: generated an empty candidate pool")

    backend = PosValCnnBackend.from_dir(
        champ, store_dir=store_dir, library_id=cfg.model.library_id,
        device=cfg.model.device)
    log(f"[boundary]   F_r calibration on champion: "
        f"{'YES (' + str(len(backend.fitted_fr_cells())) + ' cells)' if backend.fitted_fr_cells() else 'none'}")
    ranked = rank_pool(backend, cell, candidates, risk_z=risk_z)
    top = ranked[: int(top_k)]
    dist = _distribution([c.f_r_mean for c in top])
    log(f"[boundary]   top-{len(top)} predicted F_r mean: "
        f"min={dist.get('min'):.4f} median={dist.get('median'):.4f} "
        f"max={dist.get('max'):.4f}" if dist else "[boundary]   (no ranked candidates)")

    verified = None
    if verify:
        verified = verify_candidates(cfg, cell, top, base=base,
                                     fuel_library=fuel_library, log=log)
        log(f"[boundary]   verify: {verified['converged']} converged / "
            f"{verified['nonconverged']} nonconv / {verified['error']} error")

    report = build_report(cell, str(champ), ranked, top, risk_z=risk_z, verified=verified)
    reports = _resolve(reports_dir or "data/reports", base)
    out = reports / f"boundary_probe_{cell.cell_id}.json"
    _atomic_write_json(out, report)
    report["_report_path"] = str(out)
    log(f"[boundary]   report -> {out}")
    return report


def _resolve(path_str: str | Path, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p)


__all__ = [
    "BOUNDARY_GENERATOR",
    "DEFAULT_BOUNDARY_CELLS",
    "DEFAULT_RISK_Z",
    "DEFAULT_TOP_K",
    "Candidate",
    "CellSpec",
    "build_cell_driver",
    "build_report",
    "generate_pool",
    "legacy_low_fr_seeds",
    "rank_pool",
    "resolve_cell",
    "resolve_champion_dir",
    "run_boundary_probe",
    "verify_candidates",
]
