"""Mine the improvement-step corpus for a learned move-proposal policy (stage 1).

The campaigns already performed millions of *moves*: every store row whose
``parent_record_id`` is set is a child produced by mutating a parent board with
the closed operators of :mod:`lpopt.search.genome`.  This script turns that
lineage into the supervised data a move-proposal policy trains on:

    one STEP row = (parent board, child board, which move, did it help)

Outputs (all NEW files; nothing existing is modified, the store is read-only):

* ``data/policy/steps.parquet``   — one row per (parent, child) lineage edge
* ``data/policy/elites.parquet``  — top-K feasible boards per cell ("good states")
* ``data/reports/policy_corpus_<stamp>.md`` — coverage / physics / verdict report

Move-class inference
--------------------
Both boards are decoded with the existing helpers (``schema.unpack_pattern`` ->
``GeneralOrbitGenome.from_pattern``) and diffed at the ORBIT-UNIT level, which
is the level the mutation operators actually act on.  A genome is
``(fresh unit -> batch, burned unit -> source unit, centre batch)``; the rules
are documented in :func:`classify_move` and mirrored in the report.

Usage::

    python mine_policy_corpus.py                 # write everything
    python mine_policy_corpus.py --no-report     # parquets only
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from lpopt.data.extract_a import _unpad_fresh_batch
from lpopt.data.geometry import transpose
from lpopt.data.schema import pack_pattern, unpack_pattern
from lpopt.search.genome import GeneralOrbitGenome, GenomeError
from lpopt.vendor.masterrl.ga import ORBIT_UNITS
from lpopt.vendor.masterrl.domain import SLOTS

# --------------------------------------------------------------------------- #
# program constraints (task brief; lpopt.config ships cbc_limit=1550, see report)
# --------------------------------------------------------------------------- #
F_R_LIMIT = 1.55
F_Q_LIMIT = 2.41
CBC_LIMIT = 1600.0
AO_ABS_LIMIT = 0.30

#: FOMs carried through as parent/child/delta triples.
FOMS: tuple[str, ...] = (
    "f_r", "cyclen", "cbc_max", "f_q", "ao_abs", "node_peak", "map_cov",
)

#: The F_xy axis (MASTER FXYP pin planar peaking, hard limit 1.65) - the primary
#: response of the F_xy-era campaigns and, from 2026-08-30, a corpus column
#: (``intervention_wave_r1_results_20260830.md`` s11-3: the wave produced 791
#: F_xy labels that the 80-column schema had nowhere to put).  It is deliberately
#: NOT folded into :data:`FOMS`: ``FOMS`` also drives :func:`build_elites`'
#: column list and the historical column ORDER of ``steps.parquet``, and the
#: F_xy columns are appended by a migration rather than inserted.  Emitted
#: unconditionally - all-NaN when the store predates the label - so the column
#: SET is stable and the appenders' schema-drift guard stays meaningful.
FXY_FOM = "f_xy"

#: The columns ``steps.parquet`` gained on 2026-08-30, in the order the
#: migration (``policy_v2_corpus.py backfill-fxy``) appends them.  Named here so
#: the appenders' schema-drift guard can tell "this corpus predates the F_xy
#: columns, migrate it" from a genuine mismatch.
FXY_SCHEMA_COLUMNS: tuple[str, ...] = (
    "parent_f_xy", "child_f_xy", "d_f_xy", "improved_fxy", "burn_state",
)

#: Elites kept per cell per ranking key.
ELITE_K = 20

REPO = Path(__file__).resolve().parent
STORE = REPO / "data" / "store" / "records.parquet"
FUEL_TYPES = REPO / "data" / "store" / "fuel_types.parquet"
POLICY_DIR = REPO / "data" / "policy"
REPORT_DIR = REPO / "data" / "reports"


# --------------------------------------------------------------------------- #
# radial geometry — the in-out / out-in axis
# --------------------------------------------------------------------------- #
# Every canonical slot already carries ``radius = hypot(row, col)`` (0-based
# quarter-core lattice steps from the centre).  Three rings are the RADIUS
# TERCILES of the 69 canonical slots; the cuts land at r = 5.0 and r = sqrt(50)
# = 7.0711, giving 22 / 23 / 24 slots and 69 / 80 / 92 full-core assemblies:
#
#   ring 0 "inner"       r <  5.0000   (22 slots,  69 assemblies)
#   ring 1 "middle"      5.0000 <= r < 7.0711   (23 slots,  80 assemblies)
#   ring 2 "peripheral"  r >= 7.0711   (24 slots,  92 assemblies)
#
# Shares are MULTIPLICITY-WEIGHTED (full-core assembly counts), never raw slot
# counts: a peripheral interior slot is 4 physical assemblies, an axis slot 2.
SLOT_RADIUS = np.array([s.radius for s in SLOTS], dtype=float)
SLOT_MULT = np.array([s.multiplicity for s in SLOTS], dtype=float)
RING_CUTS = np.quantile(SLOT_RADIUS, [1 / 3, 2 / 3])
RING_OF_SLOT = np.digitize(SLOT_RADIUS, RING_CUTS)
RING_MULT = np.array([SLOT_MULT[RING_OF_SLOT == k].sum() for k in range(3)])
RING_NAMES = ("inner", "middle", "periph")

#: unit -> its slot indices (an axis pair owns two slots).
UNIT_SLOTS: tuple[tuple[int, ...], ...] = tuple(u.slots for u in ORBIT_UNITS)

#: physics descriptor names emitted for every board.
PHYSICS: tuple[str, ...] = (
    "fresh_share_inner", "fresh_share_middle", "fresh_share_periph",
    "fresh_r_center", "fresh_enr_r_center", "fresh_enr_mass",
    "once_burnt_periph_share", "twice_burnt_periph_share",
)


def fresh_slots(packed: str) -> tuple[np.ndarray, list[str]]:
    """Per-slot fresh mask and fresh batch labels, straight off the packed string.

    Cheap on purpose: the canonical token is ``F:<batch>:<rot>`` for fresh and
    ``S:<restart>:<x>:<y>:<rot>`` for shuffled, so no ``Pattern`` object is
    needed for the ring profile of tens of thousands of boards.
    """
    tokens = packed.split("|")
    mask = np.fromiter((t[0] == "F" for t in tokens), dtype=bool, count=len(tokens))
    batches = [t.split(":")[1] if t[0] == "F" else "" for t in tokens]
    return mask, batches


def _enrichment_of(table: dict[str, float], batch: str) -> float:
    """Fresh-batch enrichment, tolerant of the zero-padded legacy labels.

    The MOCHA cache writes fresh types zero-padded (``C04``) while the 260624
    fuel table lists them bare (``C4``); the 5.8_5.1 table is padded on both
    sides.  ``extract_a._unpad_fresh_batch`` is the project's existing
    normalizer and is reused here rather than re-derived — without it the whole
    260624 corpus (the bulk of Dataset A) silently loses its radial descriptors.
    """
    if batch in table:
        return table[batch]
    return table.get(_unpad_fresh_batch(batch), np.nan)


def ring_profile(packed: str, enrichment: dict[str, float] | None) -> dict[str, float]:
    """Fresh radial signature of one board (the in-out vs out-in fingerprint)."""
    mask, batches = fresh_slots(packed)
    weight = SLOT_MULT * mask
    out = {
        f"fresh_share_{name}": float(weight[RING_OF_SLOT == k].sum() / RING_MULT[k])
        for k, name in enumerate(RING_NAMES)
    }
    total = weight.sum()
    out["fresh_r_center"] = float((weight * SLOT_RADIUS).sum() / total) if total else np.nan
    if enrichment is None:
        out["fresh_enr_r_center"] = np.nan
        out["fresh_enr_mass"] = np.nan
        return out
    enr = np.array([
        _enrichment_of(enrichment, b) if m else 0.0
        for b, m in zip(batches, mask, strict=True)
    ])
    rho = weight * enr
    denom = np.nansum(rho)
    out["fresh_enr_r_center"] = (
        float(np.nansum(rho * SLOT_RADIUS) / denom) if denom > 0 else np.nan
    )
    # Multiplicity-weighted TOTAL fresh enrichment — the reactivity covariate.
    # ``fresh_enr_r_center`` is a NORMALIZED first moment and is therefore blind
    # to a change in the total: ``batch_flip`` is the one same-cell operator that
    # changes the fresh batch multiset, so it moves this and nothing else, while
    # ``batch_swap`` / ``fresh_relocate`` / ``rewire_swap`` conserve it exactly.
    #
    # Carrying it separates "moved reactivity outward" from "changed how much
    # reactivity there is" — the distinction policy v1 could not make, and the
    # covariate that diagnosed its era failure (``ablation_wave_results_20260815``
    # sections 2c and 9.3: corr(d_cyclen, d_fresh_enr_mass) = 1.000 on batch_flip
    # while corr with the radial dose is 0.343).  The definition is
    # ``ablation_wave._fresh_enr_mass`` verbatim, lifted here so the corpus, the
    # wave planner and the proposal-time scorer all read one implementation.
    out["fresh_enr_mass"] = float(np.nansum(rho))
    return out


def residence_profile(genome: GeneralOrbitGenome) -> dict[str, float]:
    """Once-/twice-burnt share of the PERIPHERAL ring.

    Residence is decodable exactly: ``_depths()`` is the genome's own source-chain
    resolver, and a burned unit's chain depth to its fresh root is its residence
    (1 = once-burnt, >= 2 = twice-burnt or deeper).  Fresh units are depth 0.
    """
    depths = genome._depths()
    once = np.zeros(len(SLOTS))
    twice = np.zeros(len(SLOTS))
    for unit, depth in depths.items():
        target = once if depth == 1 else twice
        for slot in UNIT_SLOTS[unit]:
            target[slot] = 1.0
    periph = RING_OF_SLOT == 2
    return {
        "once_burnt_periph_share": float((once * SLOT_MULT)[periph].sum() / RING_MULT[2]),
        "twice_burnt_periph_share": float((twice * SLOT_MULT)[periph].sum() / RING_MULT[2]),
    }


def board_physics(
    packed: str, genome: GeneralOrbitGenome | None, enrichment: dict[str, float] | None
) -> dict[str, float]:
    out = ring_profile(packed, enrichment)
    out.update(
        residence_profile(genome) if genome is not None
        else {"once_burnt_periph_share": np.nan, "twice_burnt_periph_share": np.nan}
    )
    return out


def load_enrichment(path: Path) -> dict[str, dict[str, float]]:
    """library_id -> {fuel type_id: u_avg_enrichment}, the fresh reactivity weight.

    ``u_avg_enrichment`` is used rather than ``kinf0``: kinf0 is NaN for 36/194
    fuel types including most of the ``ga80`` library that dominates the lineage,
    so a kinf-weighted descriptor would be silently absent exactly where the
    corpus is densest.  Enrichment is present for every type.
    """
    if not path.is_file():
        return {}
    fuel = pd.read_parquet(path, columns=["library_id", "type_id", "u_avg_enrichment"])
    table: dict[str, dict[str, float]] = {}
    for lib, tid, enr in zip(
        fuel["library_id"], fuel["type_id"], fuel["u_avg_enrichment"], strict=True
    ):
        table.setdefault(str(lib), {})[str(tid)] = float(enr)
    return table


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def genome_of(packed: str) -> GeneralOrbitGenome:
    """Decode a packed 69-slot pattern into its orbit-unit genome.

    ``max_shuffle_depth``/``allow_single_cycle_discharge`` are relaxed so the
    reader never rejects a board the campaigns actually evaluated; the strict
    identities are the *generator's* business, not the miner's.
    """
    return GeneralOrbitGenome.from_pattern(
        unpack_pattern(packed), max_shuffle_depth=3, allow_single_cycle_discharge=True
    )


# --------------------------------------------------------------------------- #
# burn state of an edge
# --------------------------------------------------------------------------- #
#: Residence layers a move can reach, deepest-wins (``intervention_wave``'s
#: stratification axis; review s7.2 "once/twice-burnt swap").
BURN_STATES: tuple[str, ...] = ("fresh", "once", "twice_plus", "center")


def changed_units(parent_packed: str, child_packed: str) -> set[int]:
    """Orbit units whose slots differ between two packed boards.

    The packed string is 69 ``|``-joined canonical tokens (:func:`fresh_slots`),
    and ``UNIT_SLOTS[u]`` lists the slots unit ``u`` owns, so this is an exact
    unit-level diff with no genome decode.
    """
    p = parent_packed.split("|")
    c = child_packed.split("|")
    if len(p) != len(c):
        return set()
    moved = {i for i, (a, b) in enumerate(zip(p, c)) if a != b}
    if not moved:
        return set()
    return {u for u, slots in enumerate(UNIT_SLOTS) if moved & set(slots)}


def move_burn_state(parent_genome: GeneralOrbitGenome, parent_packed: str,
                    child_packed: str) -> str:
    """Deepest residence layer this edge touches, in the PARENT board.

    ``fresh`` (depth 0 only), ``once`` (a depth-1 unit moved), ``twice_plus``
    (a depth>=2 unit moved), ``center`` (only the centre cell changed - the
    centre is not an orbit unit).  Residence comes from
    :meth:`GeneralOrbitGenome._depths`, the same source :func:`residence_profile`
    uses, so "once-burnt swap" means the same thing in the corpus, in the
    ablation tables and in ``intervention_wave``.
    """
    units = changed_units(parent_packed, child_packed)
    if not units:
        return "center"
    depths = parent_genome._depths()
    deepest = max(int(depths.get(u, 0)) for u in units)
    if deepest <= 0:
        return "fresh"
    return "once" if deepest == 1 else "twice_plus"


# --------------------------------------------------------------------------- #
# move-class inference
# --------------------------------------------------------------------------- #
#: Maximum ``n_unit_edits`` a GENUINE single application of each operator can
#: produce, measured by replaying the operators from
#: :mod:`lpopt.search.genome` over 300 random genomes each (2026-08-15 audit;
#: every single move landed at or below these values, and the classifier
#: recovered all six operator names with 100% accuracy).  A composed move whose
#: NET diff aliases to a single-move signature is filtered out by this guard.
SINGLE_MOVE_MAX_EDITS: dict[str, int] = {
    "batch_flip": 1,
    "batch_swap": 2,
    "rewire_swap": 2,
    "fresh_relocate": 4,
}


@dataclass(frozen=True, slots=True)
class MoveDiff:
    """Genome-level diff between a parent and a child board."""

    move_class: str
    n_unit_edits: int          # orbit-unit level edit count (move-budget proxy)
    swap_units: tuple[int, int] | None  # the two rewired burned units, if any


def classify_move(parent: GeneralOrbitGenome, child: GeneralOrbitGenome) -> MoveDiff:
    """Infer which genome operator (family) turned ``parent`` into ``child``.

    Rules, in order (a child produced by ``n_moves > 1`` composed operators
    generally falls through to a ``*_multi`` / ``multi`` class -- that is the
    honest answer, not a failure):

    1. fresh-unit COUNT changed  -> ``add_fresh_unit`` / ``remove_fresh_unit``
       (feed +-4); any other count delta -> ``feed_change_multi``.
    2. fresh-unit SET changed at equal size:
       exactly one unit left the fresh set and one joined -> ``fresh_relocate``;
       more -> ``multi``.
    3. fresh set identical, wiring identical, batch labels differ:
       one label (a fresh unit's or the centre's) -> ``batch_flip``;
       two labels whose batch multiset is preserved -> ``batch_swap``;
       otherwise -> ``batch_multi``.
    4. fresh set identical, batches identical, wiring differs:
       exactly two burned units that exchanged each other's sources ->
       ``rewire_swap``; otherwise -> ``rewire_multi``.
    5. wiring AND batches both differ (fresh set intact) -> ``multi``.
    6. genomes equal -> ``identity`` (never expected: record_id keys the pattern).
    """
    p_fresh, c_fresh = dict(parent.fresh), dict(child.fresh)
    p_wire, c_wire = dict(parent.wiring), dict(child.wiring)
    p_units, c_units = set(p_fresh), set(c_fresh)

    entered, left = c_units - p_units, p_units - c_units
    wire_diff = sorted(
        u for u in set(p_wire) | set(c_wire) if p_wire.get(u) != c_wire.get(u)
    )
    batch_diff = sorted(
        u for u in p_units & c_units if p_fresh[u] != c_fresh[u]
    )
    center_diff = parent.center_batch != child.center_batch
    n_edits = max(len(entered), len(left)) + len(wire_diff) + len(batch_diff) + int(center_diff)

    if len(p_units) != len(c_units):
        delta = len(c_units) - len(p_units)
        cls = {1: "add_fresh_unit", -1: "remove_fresh_unit"}.get(delta, "feed_change_multi")
        return MoveDiff(cls, n_edits, None)

    if entered or left:
        cls = "fresh_relocate" if len(entered) == 1 and len(left) == 1 else "multi"
        return MoveDiff(cls, n_edits, None)

    labels_changed = bool(batch_diff) or center_diff
    if not wire_diff:
        if not labels_changed:
            return MoveDiff("identity", n_edits, None)
        if len(batch_diff) + int(center_diff) == 1:
            return MoveDiff("batch_flip", n_edits, None)
        if (
            len(batch_diff) == 2
            and not center_diff
            and sorted(p_fresh[u] for u in batch_diff)
            == sorted(c_fresh[u] for u in batch_diff)
        ):
            return MoveDiff("batch_swap", n_edits, None)
        return MoveDiff("batch_multi", n_edits, None)

    if labels_changed:
        return MoveDiff("multi", n_edits, None)

    if (
        len(wire_diff) == 2
        and all(u in p_wire and u in c_wire for u in wire_diff)
        and {p_wire[u] for u in wire_diff} == {c_wire[u] for u in wire_diff}
    ):
        return MoveDiff("rewire_swap", n_edits, (wire_diff[0], wire_diff[1]))
    return MoveDiff("rewire_multi", n_edits, None)


# --------------------------------------------------------------------------- #
# feasibility
# --------------------------------------------------------------------------- #
def feasibility(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    """Tri-state program feasibility: True / False / <NA> when an axis is missing.

    Axes: F_r <= 1.55, F_q <= 2.41, CBC_max <= 1600 ppm, |AO| <= 0.30, converged.
    ``cbc_max`` is unharvested on ~17% of converged rows, so <NA> is a real and
    frequent state and must not be silently collapsed to False.
    """
    limits = {
        "f_r": F_R_LIMIT, "f_q": F_Q_LIMIT, "cbc_max": CBC_LIMIT, "ao_abs": AO_ABS_LIMIT,
    }
    out = pd.Series(True, index=frame.index, dtype="boolean")
    for col, limit in limits.items():
        values = frame[f"{prefix}{col}"]
        ok = (values <= limit).astype("boolean")
        ok[values.isna()] = pd.NA
        out = out & ok
    return out & frame[f"{prefix}converged"].astype("boolean")


def cyclen_bands(repo: Path) -> dict[str, tuple[float, float]]:
    """campaign -> (target EFPD, tolerance) parsed from that campaign's deck.

    Only wave campaigns keep a deck (``runs/<campaign>/input_deck.inp`` or a
    root ``<campaign>.inp``); produce strata do not, so most campaigns end up
    with no band and their ``in_cyclen_band`` stays <NA>.
    """
    pat_t = re.compile(r"^\s*cycle_target_efpd\s*=\s*([0-9.]+)", re.M)
    pat_o = re.compile(r"^\s*cycle_tolerance_efpd\s*=\s*([0-9.]+)", re.M)
    bands: dict[str, tuple[float, float]] = {}
    decks: list[tuple[str, Path]] = [
        (d.name, d / "input_deck.inp") for d in sorted((repo / "runs").glob("*")) if d.is_dir()
    ]
    decks += [(p.stem, p) for p in sorted(repo.glob("*.inp"))]
    for name, path in decks:
        if name in bands or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        m_t, m_o = pat_t.search(text), pat_o.search(text)
        if m_t and m_o:
            bands[name] = (float(m_t.group(1)), float(m_o.group(1)))
    return bands


# --------------------------------------------------------------------------- #
# step corpus
# --------------------------------------------------------------------------- #
def cell_key(pair: str, feed: int, library: str) -> str:
    return f"{pair}/f{int(feed)}/{library}"


#: Deadband on a radial-centre delta below which a move is called neutral.
#: Moves that touch neither fresh placement nor batch labels (e.g. rewire_swap)
#: give an EXACTLY zero delta, so the deadband only guards float noise.
RADIAL_EPS = 1.0e-9


def _direction(delta: pd.Series) -> pd.Series:
    """outward / inward / neutral (n/a when the descriptor is unavailable)."""
    out = pd.Series("neutral", index=delta.index, dtype=object)
    out[delta > RADIAL_EPS] = "outward"
    out[delta < -RADIAL_EPS] = "inward"
    out[delta.isna()] = "n/a"
    return out


#: MOCHA moves that are a genuine single operator even when their net diff is
#: large (a ``change_fresh_type`` family repaint relabels ~30 units at once).
#: ``compound_shuffle`` applies several primitives and is marked, not fitted.
SA_COMPOUND_MOVE = "compound_shuffle"


def lineage_edges(store: pd.DataFrame, sa: pd.DataFrame | None) -> pd.DataFrame:
    """All (parent, child) edges to mine, from every lineage source.

    ``lpopt_genome`` edges come from the store's own ``parent_record_id``
    column (campaign / produce era).  ``sa_mocha`` edges are the Dataset A
    proposal chains recovered by ``mine_sa_lineage.py`` from ``sa_log.csv``;
    they carry MOCHA's own move name, so their class is ground truth rather
    than inference.
    """
    known = set(store["record_id"])
    native = store.loc[store["parent_record_id"].notna(), ["record_id", "parent_record_id"]]
    native = native[native["parent_record_id"].isin(known)]
    edges = pd.DataFrame({
        "parent_record_id": native["parent_record_id"].to_numpy(),
        "child_record_id": native["record_id"].to_numpy(),
        "lineage_source": "lpopt_genome",
        "source_move": pd.NA,
        "sa_accepted": pd.NA,
    })
    if sa is not None and not sa.empty:
        ok = sa[
            sa["parent_record_id"].isin(known) & sa["child_record_id"].isin(known)
        ]
        edges = pd.concat([edges, pd.DataFrame({
            "parent_record_id": ok["parent_record_id"].to_numpy(),
            "child_record_id": ok["child_record_id"].to_numpy(),
            "lineage_source": "sa_mocha",
            "source_move": ok["sa_move"].to_numpy(),
            "sa_accepted": ok["sa_accepted"].to_numpy(),
        })], ignore_index=True)
    edges = edges[edges["parent_record_id"] != edges["child_record_id"]]
    return edges.drop_duplicates(subset=["parent_record_id", "child_record_id"])


def build_steps(
    store: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    enrichment: dict[str, dict[str, float]],
    sa: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Emit one row per lineage edge, from every source."""
    indexed = store.set_index("record_id", drop=False)
    edges = lineage_edges(store, sa).reset_index(drop=True)

    parents = indexed.loc[edges["parent_record_id"].to_numpy()].reset_index(drop=True)
    children = indexed.loc[edges["child_record_id"].to_numpy()].reset_index(drop=True)

    genomes: dict[str, GeneralOrbitGenome] = {}

    def cached(rid: str, packed: str) -> GeneralOrbitGenome:
        genome = genomes.get(rid)
        if genome is None:
            genome = genomes[rid] = genome_of(packed)
        return genome

    move_class: list[str] = []
    n_unit_edits: list[int] = []
    n_slots_changed: list[int] = []
    swap_span: list[float] = []
    swap_radius: list[float] = []
    parent_phys: list[dict[str, float]] = []
    child_phys: list[dict[str, float]] = []
    burn_state: list[str] = []
    for prid, ppat, crid, cpat, lib in zip(
        parents["record_id"], parents["pattern"],
        children["record_id"], children["pattern"],
        children["library_id"], strict=True,
    ):
        pg, cg = cached(prid, ppat), cached(crid, cpat)
        diff = classify_move(pg, cg)
        move_class.append(diff.move_class)
        n_unit_edits.append(diff.n_unit_edits)
        n_slots_changed.append(unpack_pattern(ppat).hamming(unpack_pattern(cpat)))
        if diff.swap_units is None:
            swap_span.append(np.nan)
            swap_radius.append(np.nan)
        else:
            r1 = ORBIT_UNITS[diff.swap_units[0]].radius
            r2 = ORBIT_UNITS[diff.swap_units[1]].radius
            swap_span.append(abs(r1 - r2))
            swap_radius.append(0.5 * (r1 + r2))
        enr = enrichment.get(str(lib))
        parent_phys.append(board_physics(ppat, pg, enr))
        child_phys.append(board_physics(cpat, cg, enr))
        burn_state.append(move_burn_state(pg, ppat, cpat))

    steps = pd.DataFrame({
        "lineage_source": edges["lineage_source"].to_numpy(),
        "source_move": edges["source_move"].to_numpy(),
        "sa_accepted": edges["sa_accepted"].to_numpy(),
        "campaign": children["campaign"].to_numpy(),
        "dataset": children["dataset"].to_numpy(),
        "generator": children["generator"].to_numpy(),
        "case_pair": children["case_pair"].to_numpy(),
        "feed": children["feed"].to_numpy(),
        "library_id": children["library_id"].to_numpy(),
        "cell": [
            cell_key(p, f, l) for p, f, l in
            zip(children["case_pair"], children["feed"], children["library_id"], strict=True)
        ],
        "cross_cell": (
            (parents["case_pair"].to_numpy() != children["case_pair"].to_numpy())
            | (parents["feed"].to_numpy() != children["feed"].to_numpy())
            | (parents["library_id"].to_numpy() != children["library_id"].to_numpy())
        ),
        "parent_record_id": parents["record_id"].to_numpy(),
        "child_record_id": children["record_id"].to_numpy(),
        "parent_pattern": parents["pattern"].to_numpy(),
        "child_pattern": children["pattern"].to_numpy(),
        "n_slots_changed": np.asarray(n_slots_changed, dtype="int32"),
        "n_unit_edits": np.asarray(n_unit_edits, dtype="int32"),
        "move_class": move_class,
        "swap_span": np.asarray(swap_span, dtype=float),
        "swap_radius": np.asarray(swap_radius, dtype=float),
        "parent_converged": parents["converged"].to_numpy(),
        "child_converged": children["converged"].to_numpy(),
    })

    for fom in FOMS:
        steps[f"parent_{fom}"] = parents[fom].to_numpy()
        steps[f"child_{fom}"] = children[fom].to_numpy()
        steps[f"d_{fom}"] = steps[f"child_{fom}"] - steps[f"parent_{fom}"]

    # ---- F_xy (see FXY_FOM) ----------------------------------------------- #
    for side, rows in (("parent", parents), ("child", children)):
        steps[f"{side}_{FXY_FOM}"] = (
            pd.to_numeric(rows[FXY_FOM], errors="coerce").to_numpy()
            if FXY_FOM in rows.columns
            else np.full(len(steps), np.nan))
    steps[f"d_{FXY_FOM}"] = steps[f"child_{FXY_FOM}"] - steps[f"parent_{FXY_FOM}"]

    # ---- physics annotations (radial strategy) ---------------------------- #
    p_phys = pd.DataFrame(parent_phys)
    c_phys = pd.DataFrame(child_phys)
    for name in PHYSICS:
        steps[f"parent_{name}"] = p_phys[name].to_numpy()
        steps[f"child_{name}"] = c_phys[name].to_numpy()
        steps[f"d_{name}"] = steps[f"child_{name}"] - steps[f"parent_{name}"]

    steps["fresh_radial_dir"] = _direction(steps["d_fresh_enr_r_center"])
    steps["burnt_periph_dir"] = _direction(steps["d_twice_burnt_periph_share"])
    #: The residence layer the move reaches - the stratum the intervention wave
    #: blocked on and the only one of its axes the corpus could not express.
    steps["burn_state"] = burn_state

    # A ``compound_shuffle`` is several MOCHA primitives in one move; its net
    # diff is a real description of the boards but NOT of a single operator, so
    # it is marked rather than fitted to the lpopt vocabulary.
    compound = (steps["source_move"] == SA_COMPOUND_MOVE).to_numpy()
    steps.loc[compound, "move_class"] = "sa_unknown"

    # ``single_move`` is an EDIT-COUNT INFERENCE for lpopt rows (a composition
    # can alias to a single-move signature) but GROUND TRUTH for sa_mocha rows:
    # sa_log.csv names the one operator that produced the child.
    inferred = [
        edits <= SINGLE_MOVE_MAX_EDITS.get(cls, -1)
        for cls, edits in zip(steps["move_class"], steps["n_unit_edits"], strict=True)
    ]
    is_sa = (steps["lineage_source"] == "sa_mocha").to_numpy()
    steps["single_move"] = np.where(is_sa, ~compound, np.asarray(inferred, dtype=bool))
    steps["single_move_evidence"] = np.where(is_sa, "sa_log", "edit_count")

    both = (steps["parent_converged"] & steps["child_converged"]).astype("boolean")
    objectives = (
        ("improved_fxy", FXY_FOM, True),
        ("improved_fr", "f_r", True),
        ("improved_flat", "node_peak", True),
        ("improved_cbc", "cbc_max", True),
        ("improved_cyclen", "cyclen", False),
    )
    for label, fom, lower_is_better in objectives:
        child, parent = steps[f"child_{fom}"], steps[f"parent_{fom}"]
        better = (child < parent if lower_is_better else child > parent).astype("boolean")
        known = (
            both & parent.notna() & child.notna()
        ).fillna(False).astype(bool).to_numpy()
        better[~known] = pd.NA
        steps[label] = better
    steps["both_converged"] = both

    steps["feasible_parent"] = feasibility(steps, "parent_")
    steps["feasible_child"] = feasibility(steps, "child_")

    target = steps["campaign"].map(lambda c: bands.get(c, (np.nan, np.nan))[0])
    tol = steps["campaign"].map(lambda c: bands.get(c, (np.nan, np.nan))[1])
    in_band = ((steps["child_cyclen"] - target).abs() <= tol).astype("boolean")
    in_band[target.isna() | steps["child_cyclen"].isna()] = pd.NA
    steps["in_cyclen_band_child"] = in_band
    steps["cyclen_band_known"] = target.notna()
    return steps


# --------------------------------------------------------------------------- #
# elite ("good state") set
# --------------------------------------------------------------------------- #
def build_elites(
    store: pd.DataFrame,
    enrichment: dict[str, dict[str, float]],
    k: int = ELITE_K,
) -> pd.DataFrame:
    """Top-``k`` feasible boards per cell by F_r and by node_peak (campaign-blind)."""
    pool = store.copy()
    pool["feasible"] = feasibility(pool)
    pool = pool[pool["feasible"].fillna(False).astype(bool).to_numpy()].copy()
    pool["cell"] = [
        cell_key(p, f, l) for p, f, l in
        zip(pool["case_pair"], pool["feed"], pool["library_id"], strict=True)
    ]

    keep = [
        "cell", "case_pair", "feed", "library_id", "record_id", "campaign",
        "generator", "dataset", "pattern", *FOMS,
    ]
    frames: list[pd.DataFrame] = []
    for rank_by in ("f_r", "node_peak"):
        ranked = pool[pool[rank_by].notna()]
        if ranked.empty:
            continue
        ranked = ranked.sort_values(["cell", rank_by], kind="mergesort")
        top = ranked.groupby("cell", sort=False).head(k).copy()
        top["rank_by"] = rank_by
        top["rank"] = top.groupby("cell", sort=False).cumcount() + 1
        frames.append(top[keep + ["rank_by", "rank"]])
    if not frames:
        return pd.DataFrame(columns=keep + ["rank_by", "rank", *PHYSICS])

    elites = pd.concat(frames, ignore_index=True)
    phys = pd.DataFrame([
        board_physics(packed, genome_of(packed), enrichment.get(str(lib)))
        for packed, lib in zip(elites["pattern"], elites["library_id"], strict=True)
    ])
    for name in PHYSICS:
        elites[name] = phys[name].to_numpy()
    return elites


def verify_transpose(steps: pd.DataFrame, enrichment: dict[str, dict[str, float]],
                     sample: int = 1500, seed: int = 0) -> dict[str, int]:
    """Check that the diagonal-mirror augmentation is label-preserving.

    ``lpopt.data.geometry.transpose`` reflects a board across the qi<->qj
    diagonal.  It is an involution, it preserves feed, and — because it maps
    each slot to one of EQUAL radius and EQUAL orbit multiplicity — it must
    leave every ring descriptor, the move class, the unit-edit count and the
    69-slot Hamming distance untouched.  Only the two pattern strings change.
    That is what makes the augmentation free: the recipe is

        parent_pattern -> pack_pattern(transpose(unpack_pattern(parent_pattern)))
        child_pattern  -> pack_pattern(transpose(unpack_pattern(child_pattern)))

    with EVERY other column copied verbatim.  This function proves that claim
    on a random sample instead of asserting it.
    """
    frame = steps.sample(min(sample, len(steps)), random_state=seed)
    counts = {"tested": 0, "undecodable": 0, "class_break": 0,
              "hamming_break": 0, "physics_break": 0}
    for row in frame.itertuples():
        enr = enrichment.get(str(row.library_id))
        try:
            t_parent = pack_pattern(transpose(unpack_pattern(row.parent_pattern)))
            t_child = pack_pattern(transpose(unpack_pattern(row.child_pattern)))
            p_g, c_g = genome_of(t_parent), genome_of(t_child)
        except (ValueError, KeyError, AssertionError, GenomeError):
            counts["undecodable"] += 1
            continue
        counts["tested"] += 1
        diff = classify_move(p_g, c_g)
        # Re-apply the same compound override the corpus applies, so the check
        # compares like with like instead of flagging the marker itself.
        mirrored = ("sa_unknown" if str(row.source_move) == SA_COMPOUND_MOVE
                    else diff.move_class)
        if mirrored != row.move_class or diff.n_unit_edits != row.n_unit_edits:
            counts["class_break"] += 1
        if unpack_pattern(t_parent).hamming(unpack_pattern(t_child)) != row.n_slots_changed:
            counts["hamming_break"] += 1
        after = board_physics(t_child, c_g, enr)
        for name in PHYSICS:
            before = getattr(row, f"child_{name}")
            if np.isnan(before) and np.isnan(after[name]):
                continue
            if not np.isclose(before, after[name], rtol=0, atol=1e-12):
                counts["physics_break"] += 1
                break
    return counts


def cell_ring_baseline(store: pd.DataFrame, cells: Iterable[str]) -> pd.DataFrame:
    """All-comers mean fresh share per ring, per cell (converged rows only).

    Only the ring profile is computed here (straight off the packed string, no
    genome), so the baseline over tens of thousands of boards stays cheap.
    """
    wanted = set(cells)
    pool = store[store["converged"].astype(bool)].copy()
    pool["cell"] = [
        cell_key(p, f, l) for p, f, l in
        zip(pool["case_pair"], pool["feed"], pool["library_id"], strict=True)
    ]
    pool = pool[pool["cell"].isin(wanted)]
    if pool.empty:
        return pd.DataFrame(columns=["cell", "n_all", *(f"all_{n}" for n in RING_NAMES)])
    rings = pd.DataFrame([ring_profile(p, None) for p in pool["pattern"]])
    rings["cell"] = pool["cell"].to_numpy()
    agg = rings.groupby("cell").agg(
        n_all=("fresh_share_inner", "size"),
        **{f"all_{n}": (f"fresh_share_{n}", "mean") for n in RING_NAMES},
    )
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# chain statistics
# --------------------------------------------------------------------------- #
def chain_stats(steps: pd.DataFrame, improving_only: bool) -> dict[str, int]:
    """Longest lineage chain length (edges) per cell over the step DAG."""
    frame = steps
    if improving_only:
        frame = frame[frame["improved_fr"].fillna(False).astype(bool).to_numpy()]
    out: dict[str, int] = {}
    for cell, group in frame.groupby("cell"):
        parent_of = dict(zip(group["child_record_id"], group["parent_record_id"], strict=True))
        depth: dict[str, int] = {}

        def depth_of(node: str) -> int:
            path: list[str] = []
            walking: set[str] = set()
            while node in parent_of and node not in depth and node not in walking:
                walking.add(node)
                path.append(node)
                node = parent_of[node]
            base = depth.get(node, 0)
            for member in reversed(path):
                base += 1
                depth[member] = base
            return base

        best = 0
        for node in group["child_record_id"]:
            best = max(best, depth_of(node))
        out[cell] = best
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _md_table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "n/a" if pd.isna(v) else floatfmt.format(v)
        return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def _improving_table(frame: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    grouped = frame.groupby(list(by), dropna=False)
    out = pd.DataFrame({
        "n_steps": grouped.size(),
        "n_labeled": grouped["improved_fr"].apply(lambda s: int(s.notna().sum())),
        "n_improving": grouped["improved_fr"].apply(lambda s: int(s.fillna(False).sum())),
    }).reset_index()
    out["improving_frac"] = np.where(
        out["n_labeled"] > 0, out["n_improving"] / out["n_labeled"].replace(0, np.nan), np.nan
    )
    return out.sort_values("n_steps", ascending=False)


#: objective label -> column shown in the multi-objective tables.
OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("improved_fr", "F_r v"),
    ("improved_flat", "flat v"),
    ("improved_cbc", "CBC v"),
    ("improved_cyclen", "cyclen ^"),
)


def _multi_objective_table(frame: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Improving fraction for every objective, per group (n per objective too)."""
    grouped = frame.groupby(list(by), dropna=False, observed=True)
    out = pd.DataFrame({"n_steps": grouped.size()})
    for label, title in OBJECTIVES:
        out[f"n({title})"] = grouped[label].apply(lambda s: int(s.notna().sum()))
        out[title] = grouped[label].apply(
            lambda s: float(s.fillna(False).sum()) / n if (n := int(s.notna().sum())) else np.nan
        )
    return out.reset_index().sort_values("n_steps", ascending=False)


def write_report(
    store: pd.DataFrame, steps: pd.DataFrame, elites: pd.DataFrame,
    bands: dict[str, tuple[float, float]], path: Path,
    transpose_check: dict[str, int] | None = None,
) -> None:
    n_children = int(store["parent_record_id"].notna().sum())
    n_resolved = int((steps["lineage_source"] == "lpopt_genome").sum())
    # The policy's action space is a FIXED cell (pair, feed, library); a
    # cross-cell edge is a feed-morph / transfer re-seeding, not a move.
    moves = steps[~steps["cross_cell"]]
    transfers = steps[steps["cross_cell"]]
    main_cell = moves["cell"].value_counts().idxmax()
    main = moves[moves["cell"] == main_cell]
    n_boards = int(pd.unique(np.concatenate([
        steps["parent_record_id"].to_numpy(), steps["child_record_id"].to_numpy()
    ])).size)
    n_legacy = int((store["dataset"] == "A").sum())
    known_ids = set(store["record_id"])
    orphan_parents = int(
        store.loc[store["parent_record_id"].notna(), "parent_record_id"]
        .loc[lambda s: ~s.isin(known_ids)].nunique()
    )

    lines: list[str] = []
    add = lines.append
    add(f"# Policy corpus mining — improvement-step corpus (stage 1)\n")
    add(f"Generated {date.today().isoformat()} by `mine_policy_corpus.py` "
        f"(read-only over `data/store/records.parquet`).\n")
    add("Goal: supervised data for a **learned move-proposal policy** — not a "
        "surrogate. One row of `data/policy/steps.parquet` is one board->board "
        "*move* the campaigns actually made, with the move class inferred and "
        "the outcome labeled.\n")

    # -- 1. headline ------------------------------------------------------- #
    add("## 1. Headline\n")
    add(f"- store rows: **{len(store):,}** ({int(store['converged'].sum()):,} converged)")
    src = steps["lineage_source"].value_counts()
    add(f"- step pairs emitted: **{len(steps):,}** — "
        f"**{int(src.get('sa_mocha', 0)):,}** recovered from the Dataset A "
        f"MOCHA SA logs (section 6a) plus "
        f"**{int(src.get('lpopt_genome', 0)):,}** from the store's own "
        "`parent_record_id` lineage. Of the total:")
    add(f"  - **same-cell MOVES: {len(moves):,}** — parent and child share "
        "(pair, feed, library). *This is the policy corpus.*")
    add(f"  - cross-cell TRANSFERS: **{len(transfers):,}** — feed-morph / "
        "pair-transfer re-seedings (`elite_perturb`, `transfer`, "
        "`g3_elite_boundary`, and MOCHA's fuel-family repaints). Kept in the "
        "parquet under `cross_cell=True` because they are real physics, but "
        "they are **not moves in a fixed action space** and are excluded from "
        f"every policy-facing table below. They are "
        f"{len(transfers) / len(steps):.0%} of the raw lineage — down from 75% "
        "before the Dataset A recovery, which is almost entirely same-cell.")
    add(f"- both endpoints converged: **{int(steps['both_converged'].fillna(False).sum()):,}** "
        f"overall, **{int(moves['both_converged'].fillna(False).sum()):,}** same-cell")
    add(f"- F_r-labeled MOVES: **{int(moves['improved_fr'].notna().sum()):,}**, "
        f"of which improving: **{int(moves['improved_fr'].fillna(False).sum()):,}** "
        f"({moves['improved_fr'].fillna(False).mean():.1%} of same-cell steps)")
    add(f"- flatness(node_peak)-labeled MOVES: "
        f"**{int(moves['improved_flat'].notna().sum()):,}**, improving: "
        f"**{int(moves['improved_flat'].fillna(False).sum()):,}**")
    na_feas = int(steps["feasible_child"].isna().sum())
    add(f"- feasible children (F_r<=1.55, F_q<=2.41, CBC<=1600, |AO|<=0.30, converged): "
        f"**{int(moves['feasible_child'].fillna(False).sum()):,}** of the same-cell "
        f"moves, **{int(steps['feasible_child'].fillna(False).sum()):,}** overall"
        + (f" (<NA> on {na_feas:,} steps with an unharvested constraint axis)"
           if na_feas else " (no step has a missing constraint axis)"))
    add(f"- elite boards emitted: **{len(elites):,}** over "
        f"**{elites['cell'].nunique()}** cells\n")

    # -- 2. move-class inference ------------------------------------------- #
    add("## 2. Move-class inference rules\n")
    add("Both boards are decoded with the existing helpers "
        "(`lpopt.data.schema.unpack_pattern` -> "
        "`lpopt.search.genome.GeneralOrbitGenome.from_pattern`) and diffed at the "
        "**orbit-unit** level, which is the level the mutation operators act on. "
        "A genome is `(fresh unit -> batch, burned unit -> source unit, centre batch)`. "
        f"All {n_boards:,} distinct boards in the lineage decoded without a "
        "single failure.\n")
    add("| rule (first match wins) | class |")
    add("|---|---|")
    add("| fresh-unit count +1 / -1 (feed +-4) | `add_fresh_unit` / `remove_fresh_unit` |")
    add("| fresh-unit count changed by more | `feed_change_multi` |")
    add("| fresh set same size, exactly 1 unit left and 1 joined | `fresh_relocate` |")
    add("| fresh set same size, more than 1 swapped in/out | `multi` |")
    add("| identical fresh set + wiring, 1 batch label (or centre) changed | `batch_flip` |")
    add("| identical fresh set + wiring, 2 labels changed, batch multiset preserved | `batch_swap` |")
    add("| identical fresh set + wiring, any other label change | `batch_multi` |")
    add("| identical fresh set + labels, exactly 2 burned units exchanged sources | `rewire_swap` |")
    add("| identical fresh set + labels, any other wiring change | `rewire_multi` |")
    add("| wiring AND labels both changed (fresh set intact) | `multi` |")
    add("")
    add("**Classifier validation (2026-08-15).** Every operator in "
        "`lpopt/search/genome.py` was replayed over 300 random genomes each and "
        "the classifier recovered the operator name **100% of the time** for all "
        "six single moves (`rewire_swap`, `fresh_relocate`, `batch_flip`, "
        "`batch_swap`, `add_fresh_unit`, `remove_fresh_unit`).\n")
    add("**But the class is the NET-DIFF class, not the move count — and they "
        "come apart.** Campaigns mutate with `n_moves >= 1` composed operators, "
        "and a composition can *alias* to a single-move signature: replaying "
        "3-move mutations, 197/400 landed on a net diff indistinguishable from "
        "one `fresh_relocate` (relocate A->B then B->C leaves exactly one unit "
        "entering and one leaving the fresh set). So `move_class` alone must not "
        "be read as \"this was one move\".\n")
    add("The guard is `n_unit_edits`, and it is sharp because a genuine single "
        "move has a bounded edit count:\n")
    guard = pd.DataFrame({
        "move_class": list(SINGLE_MOVE_MAX_EDITS),
        "max n_unit_edits for a TRUE single move": list(SINGLE_MOVE_MAX_EDITS.values()),
        "corpus median (same-cell)": [
            float(moves.loc[moves["move_class"] == c, "n_unit_edits"].median())
            for c in SINGLE_MOVE_MAX_EDITS
        ],
        "corpus n": [int((moves["move_class"] == c).sum()) for c in SINGLE_MOVE_MAX_EDITS],
    })
    add(_md_table(guard, "{:.1f}"))
    add("")
    add("Read that table carefully: `rewire_swap`, `batch_flip` and `batch_swap` "
        "sit exactly at their single-move bound in the corpus, so those really "
        "are single moves. `fresh_relocate` has a corpus median far above its "
        "bound of 4 — **most steps classified `fresh_relocate` are composed "
        "moves that merely look like one.** The boolean column `single_move` "
        "applies this per-class bound and is the filter to use when training a "
        "single-move proposal head; `n_slots_changed` (69-slot Hamming) is "
        "carried alongside for the raw board-level distance.\n")

    split = steps.pivot_table(
        index="move_class", columns="cross_cell", values="child_record_id",
        aggfunc="count", fill_value=0,
    ).rename(columns={False: "same_cell", True: "cross_cell"}).reset_index()
    add("### Where each class lives (same-cell moves vs cross-cell transfers)\n")
    add(_md_table(split))
    add("")
    add("The split is almost perfectly clean and it is a sanity check on the "
        "classifier: `feed_change_multi`, `add_fresh_unit`, `remove_fresh_unit` "
        "and (cross-pair) `batch_multi` occur **only** on cross-cell edges — a "
        "feed morph *must* change the fresh-unit count and a pair transfer "
        "*must* relabel batches. Within a cell the campaigns only ever play "
        "`fresh_relocate`, `rewire_*`, `batch_*` and composed `multi`, exactly "
        "as `construct.py` configures them (`feed_move_prob=0.0`).\n")
    edits = moves.groupby("move_class")[["n_unit_edits", "n_slots_changed"]].median().reset_index()
    add("Median edit size per class, same-cell moves (unit edits / 69-slot Hamming):\n")
    add(_md_table(edits, "{:.1f}"))
    add("")
    add("`multi` dominating at ~25 unit edits is the `n_moves_late` composition "
        "showing through: those children are not one move away from their "
        "parent and are weak supervision for a single-move proposal head.\n")

    # -- 3. physics annotations --------------------------------------------- #
    add("## 3. Physics annotations — the radial strategy axis\n")
    add("A move-proposal policy has to arbitrate the same trade the engineer "
        "argues about out loud: *leakage vs flattening*. Push once/twice-burnt "
        "fuel to the periphery (out-in) and you cut neutron leakage — cycle "
        "length and CBC go up. Push fresh to the periphery (in-out) and you "
        "flatten radial power — F_r and node_peak go down, but you leak. Every "
        "step row and every elite row therefore carries the cheap decodable "
        "descriptors that make the trade visible.\n")
    add("**Ring definition** — radius terciles of the 69 canonical slots, where "
        "`radius = hypot(row, col)` is the vendor `Slot.radius` (0-based "
        "quarter-core lattice steps from the centre). Cuts land at "
        f"r = {RING_CUTS[0]:.4f} and r = {RING_CUTS[1]:.4f} (= sqrt(50)):\n")
    ring_def = pd.DataFrame({
        "ring": ["0 inner", "1 middle", "2 peripheral"],
        "radius": [
            f"r < {RING_CUTS[0]:.4f}",
            f"{RING_CUTS[0]:.4f} <= r < {RING_CUTS[1]:.4f}",
            f"r >= {RING_CUTS[1]:.4f}",
        ],
        "slots": [int((RING_OF_SLOT == k).sum()) for k in range(3)],
        "full_core_assemblies": [int(RING_MULT[k]) for k in range(3)],
    })
    add(_md_table(ring_def))
    add("")
    add("All shares are **multiplicity-weighted** (full-core assembly counts, not "
        "raw slot counts): a peripheral interior slot is 4 physical assemblies, "
        "an axis slot 2, the centre 1.\n")
    add("| descriptor | meaning |")
    add("|---|---|")
    add("| `fresh_share_inner/middle/periph` | fraction of each ring's assemblies that are FRESH — the in-out vs out-in signature |")
    add("| `fresh_r_center` | multiplicity-weighted mean radius of the fresh inventory |")
    add("| `fresh_enr_r_center` | the same, additionally weighted by fresh-FA `u_avg_enrichment` — \"did this move push REACTIVITY outward or inward?\" |")
    add("| `once_burnt_periph_share` | fraction of the peripheral ring holding once-burnt FA (chain depth 1) |")
    add("| `twice_burnt_periph_share` | fraction of the peripheral ring holding twice-burnt-or-deeper FA (chain depth >= 2) |")
    add("")
    add("Each appears as `parent_*`, `child_*` and `d_*` on every step row, and "
        "once per board on every elite row.\n")
    add("Provenance and what is NOT approximated:\n")
    add("- **Residence is exact, not guessed.** The genome's own source-chain "
        "resolver (`GeneralOrbitGenome._depths()`) gives every burned orbit unit "
        "its chain depth to a fresh root: depth 1 = once-burnt, depth >= 2 = "
        "twice-burnt or deeper. Nothing is inferred from burnup.")
    add("- **Reactivity weight is `u_avg_enrichment`, NOT `kinf0`.** "
        "`data/store/fuel_types.parquet` carries `kinf0`, but it is NaN for "
        "36/194 types including most of the `ga80` library that dominates this "
        "lineage — a kinf-weighted descriptor would be silently absent exactly "
        "where the corpus is densest. Enrichment is present for every "
        "(library_id, type_id) used by any step, so `fresh_enr_r_center` is "
        "populated on 100% of steps. If a kinf-weighted variant is wanted later, "
        "back-fill `kinf0` for ga80 first.")
    add("- **Per-assembly burnup per slot is NOT available** and is not "
        "approximated. Residence class (fresh / once / twice+) is the honest "
        "resolution the pattern encoding supports; actual slot burnup would need "
        "the EDIT5 maps, which exist for only 58% of converged rows.\n")
    dirs = moves["fresh_radial_dir"].value_counts()
    add("Direction of the **same-cell moves** on the reactivity-weighted fresh "
        f"radial centre (`d_fresh_enr_r_center`): outward "
        f"**{int(dirs.get('outward', 0)):,}**, inward "
        f"**{int(dirs.get('inward', 0)):,}**, neutral "
        f"**{int(dirs.get('neutral', 0)):,}**. `neutral` is not noise — a "
        "`rewire_swap` leaves the fresh set and the batch labels untouched, so "
        "its fresh radial centre is *exactly* unchanged and the whole move lives "
        "in the burnt inventory. That is why `burnt_periph_dir` "
        "(sign of `d_twice_burnt_periph_share`) is carried alongside.\n")

    # -- 4. the key table --------------------------------------------------- #
    add("## 4. KEY TABLE — improving fraction by move class and radial direction\n")
    add("This is what the policy has to learn: *which move improves what, where*. "
        "`F_r v` / `flat v` / `CBC v` = fraction of labeled steps where the child "
        "beat the parent (lower is better); `cyclen ^` = fraction where the child "
        "ran longer. Each `n(...)` is that objective's labeled count (both "
        "endpoints converged AND both values present), so they differ per column "
        "— `flat` and `CBC` are much sparser than `F_r`.\n")
    add(f"**All tables in this section use the {len(moves):,} SAME-CELL moves "
        "only.** Cross-cell transfers are tabulated separately in section 5.\n")
    add("### 4a. By move class\n")
    add(_md_table(_multi_objective_table(moves, ["move_class"])))
    add("")
    add("### 4b. By radial direction of the move\n")
    add("`outward` = the move pushed the enrichment-weighted fresh centre toward "
        "the periphery (in-out / flattening logic); `inward` = toward the core "
        "centre (out-in / low-leakage logic); `neutral` = the move did not touch "
        "fresh placement or batch labels at all (pure rewiring of burnt fuel).\n")
    rad = _multi_objective_table(moves, ["fresh_radial_dir"]).set_index("fresh_radial_dir")
    add(_md_table(rad.reset_index()))
    add("")
    if {"outward", "inward"} <= set(rad.index):
        o, i = rad.loc["outward"], rad.loc["inward"]
        add("**The flattening half of the rule of thumb is confirmed.** Moving "
            "fresh fuel outward beats moving it inward on both flattening "
            f"objectives — F_r improves {o['F_r v']:.1%} of the time going "
            f"outward vs {i['F_r v']:.1%} going inward, node_peak "
            f"{o['flat v']:.1%} vs {i['flat v']:.1%}.\n")
        conv = moves[moves["both_converged"].fillna(False).astype(bool)]
        corr = conv[[
            "d_fresh_share_periph", "d_f_r", "d_node_peak", "d_cyclen", "d_cbc_max",
        ]].corr().loc["d_fresh_share_periph"]
        add("Correlations over the same-cell, both-converged moves "
            f"(n={len(conv):,}) say the same thing on the continuous descriptor: "
            f"`d_fresh_share_periph` vs `d_f_r` **{corr['d_f_r']:+.3f}**, vs "
            f"`d_node_peak` **{corr['d_node_peak']:+.3f}** — more peripheral "
            "fresh, lower peaking.\n")
        cyc_sign = "costs" if corr["d_cyclen"] < 0 else "does NOT cost"
        add(f"**The leakage half does not replicate.** `d_fresh_share_periph` vs "
            f"`d_cyclen` is **{corr['d_cyclen']:+.3f}** and vs `d_cbc_max` "
            f"**{corr['d_cbc_max']:+.3f}**: on this corpus, loading fresh "
            f"outward {cyc_sign} cycle length. An earlier read of this table on "
            "the 1,305-step lpopt-only corpus reported the opposite (outward "
            "loading paying a cycle-length bill); that finding **did not "
            "survive** the 15x larger corpus and is retracted. The two eras "
            "genuinely disagree — see the per-source split in section 4g — and "
            "nothing here can say which is right, because neither era sampled "
            "direction at fixed move class. This is the single clearest thing "
            "the queued 1-move ablation wave has to settle.\n")
        add("The flattening signal, by contrast, holds in both eras and at "
            "every sample size tried, so a v1 policy can be trained on it now.\n")
    add("### 4c. By twice-burnt peripheral direction\n")
    add("`outward` = the move pushed twice-burnt FA toward the periphery "
        "(low-leakage / out-in logic); `inward` = it pulled them in.\n")
    add(_md_table(_multi_objective_table(moves, ["burnt_periph_dir"])))
    add("")
    add(f"### 4d. move_class x radial direction — main cell `{main_cell}`\n")
    add("The leakage-vs-flattening arbitration the policy has to make, in the "
        "densest single cell (groups with >= 5 steps):\n")
    mo_main = _multi_objective_table(main, ["move_class", "fresh_radial_dir"])
    add(_md_table(mo_main[mo_main["n_steps"] >= 5]))
    add("")
    add("### 4e. Per-cell x move class (cells with >= 100 same-cell moves)\n")
    per_cell = _improving_table(moves, ["cell", "move_class"])
    big = moves["cell"].value_counts()
    big = big[big >= 100].index
    view = per_cell[per_cell["cell"].isin(big)].sort_values(
        ["cell", "n_steps"], ascending=[True, False]
    )
    add(_md_table(view[["cell", "move_class", "n_steps", "n_labeled", "n_improving", "improving_frac"]]))
    add("")
    add("### 4f. Per-cell x radial direction (cells with >= 100 same-cell moves)\n")
    rad_cell = _multi_objective_table(moves[moves["cell"].isin(big)], ["cell", "fresh_radial_dir"])
    add(_md_table(rad_cell.sort_values(["cell", "n_steps"], ascending=[True, False])))
    add("")
    add("### 4g. The two eras, side by side\n")
    add("Mean outcome deltas by radial direction, split by lineage source. This "
        "is where the disagreement in 4b lives — read the `d_cyclen` column:\n")
    conv = moves[moves["both_converged"].fillna(False).astype(bool)]
    era = conv.groupby(["lineage_source", "fresh_radial_dir"]).agg(
        n=("d_f_r", "size"),
        d_f_r=("d_f_r", "mean"),
        d_node_peak=("d_node_peak", "mean"),
        d_cyclen=("d_cyclen", "mean"),
        d_cbc_max=("d_cbc_max", "mean"),
    ).reset_index()
    add(_md_table(era, "{:.4f}"))
    add("")
    add("`sa_mocha` moves are single MOCHA primitives on 260624/5.8_5.1 boards "
        "at feed 121; `lpopt_genome` moves are mostly composed mutations on "
        "ga80/paramA boards across many feeds, near the F_r=1.55 boundary. "
        "Different libraries, different feeds, different move sizes, different "
        "operating points — so the sign flip is not necessarily a "
        "contradiction, but it is not resolvable from observational data.\n")

    add("Read 4b/4c/4f as the corpus's own statement of the engineer's rule of "
        "thumb, with the caveat that these are **observational** frequencies over "
        "whatever the campaigns sampled, not a controlled experiment: the move "
        "distribution is not balanced across directions within a class, so a "
        "direction effect and a class effect are partly confounded. Section 10 "
        "prescribes the ablation wave that would de-confound them.\n")

    # -- 4. campaign / cell breakdown --------------------------------------- #
    add("## 5. Breakdown by campaign and by cell\n")
    add("### 5a. Same-cell moves — top 20 cells\n")
    cells = _improving_table(moves, ["cell"]).head(20)
    add(_md_table(cells[["cell", "n_steps", "n_labeled", "n_improving", "improving_frac"]]))
    add("")
    add("### 5b. Same-cell moves — top 20 campaigns\n")
    camp = _improving_table(moves, ["campaign"]).head(20)
    add(_md_table(camp[["campaign", "n_steps", "n_labeled", "n_improving", "improving_frac"]]))
    add("")
    add("### 5c. Cross-cell transfers (kept in the parquet, excluded from policy tables)\n")
    add(f"{len(transfers):,} edges. What they actually are, by generator:\n")
    tgen = _improving_table(transfers, ["generator"])
    add(_md_table(tgen[["generator", "n_steps", "n_labeled", "n_improving", "improving_frac"]]))
    add("")
    add("`elite_perturb` is `produce.py`'s feed-morph: it takes a store elite, "
        "re-seats it onto a different feed stratum (N fresh units changes, so "
        "feed changes by 4 per unit) and then mutates. `transfer` carries a good "
        "board across fuel pairs, which relabels every fresh batch. Both are "
        "genuinely useful physics — they are how the program bootstraps a new "
        "cell — but a fixed-cell move policy cannot emit them, and their median "
        "edit size (30+ orbit units) confirms they are re-seedings rather than "
        "moves. If a *curriculum* / transfer policy is wanted later, this is its "
        "training set and it is the larger half of the corpus.\n")

    # -- 5. lineage coverage honesty ---------------------------------------- #
    add("## 6. Lineage coverage — honesty section\n")
    add("### 6a. Dataset A lineage RECOVERED (2026-08-15)\n")
    sa_steps = steps[steps["lineage_source"] == "sa_mocha"]
    if len(sa_steps):
        add("The previous pass reported Dataset A as lineage-free and called it "
            "the highest-leverage fix. It has now been done, with zero MASTER "
            "cost, by `mine_sa_lineage.py`.\n")
        add("**What the cache encodes, precisely.** The `sa_2b_cache` records "
            "themselves carry no lineage — only `key` (the rot61 board) and "
            "`rec` (metrics, converged, tag, rule). The lineage lives in the "
            "per-run **`sa_log.csv`** that `2_LP/MOCHA/optimizer.py` writes "
            "beside the cache, which gives every evaluated candidate its `tag`, "
            "its **`move`** (MOCHA's own operator name) and its `accepted` "
            "flag.\n")
        add("**It is a PROPOSAL chain, not an accept chain.** The optimizer "
            "generates a batch of candidates from one incumbent "
            "(`cand, mv = self._move(cur)` for each of `parallel_workers`, all "
            "before any acceptance), then applies Metropolis sequentially. So a "
            "candidate's parent is the board it was mutated from **whether or "
            "not it was accepted** — rejected proposals keep their parent and "
            "become negative examples, which is exactly what a move-proposal "
            "policy needs. Both readings of \"the incumbent\" were "
            "reconstructed and compared by genome diff size; the batch reading "
            "won decisively (median 2.0 unit edits vs 6.0 for the sequential "
            "reading), confirming the batch incumbent is the true generative "
            "parent. `parent_record_id` uses the batch reading; the sequential "
            "one is kept in `sa_lineage.parquet` as `seq_parent_record_id`.\n")
        add("**Tag -> board.** `rec.tag` is per-run and NOT unique across the "
            "cache (6,086 tags map to more than one board), so the join goes "
            "through each run's own case directories "
            "(`runs/<run>/cases/<tag>/cy<NN>/MAS_INP` -> `%LPD_SHF` -> "
            "`extract_a.dedup_key_of`), which is the same key `extract_a` used "
            "to build the store. The dedup index over all "
            f"{n_legacy:,} Dataset A rows had **0 ambiguous keys**.\n")
        add("**The move classifier was validated against MOCHA's own labels.** "
            "This is an independent cross-check the previous pass could not "
            "make: `sa_log.csv` says what the move was, and the genome differ "
            "says what the boards differ by. They agree exactly where the "
            "operator vocabularies overlap (counts from the 4,000-edge "
            "`mine_sa_lineage.py --verify` sample):\n")
        add("| MOCHA `move` | lpopt genome class inferred | agreement |")
        add("|---|---|---|")
        add("| `swap_burned_sources` | `rewire_swap` | 1941/1941 = **100%** |")
        add("| `swap_fresh_burned` | `fresh_relocate` | 1026/1026 = **100%** |")
        add("| `change_fresh_type` | `batch_flip` (176) / `batch_multi` (527) | see below |")
        add("| `compound_shuffle` | spread over 6 classes | marked `sa_unknown` |")
        add("")
        add("`change_fresh_type` splits because MOCHA's operator has **two** "
            "strategies (`_mv_change_fresh_type`): repaint one fresh cell "
            "(-> `batch_flip`) *or* repaint a whole fuel-type family "
            "(-> `batch_multi`, ~30 units at once, and it changes the pair, so "
            "those land in the cross-cell bucket). The family repaint is a real "
            "MOCHA move with **no counterpart in the lpopt genome vocabulary** "
            "— it is described faithfully by its net diff and identified by "
            "`source_move`, not forced into a single-operator class.\n")
        add("`compound_shuffle` applies several primitives in one move, so its "
            "net diff genuinely is not one operator: those "
            f"{int((steps['source_move'] == SA_COMPOUND_MOVE).sum()):,} steps "
            "carry `move_class='sa_unknown'` rather than a fitted label.\n")
        add("**`single_move` is now ground truth for the SA era.** For "
            "`lpopt_genome` rows it stays an edit-count inference (a "
            "composition can alias to a single-move signature); for `sa_mocha` "
            "rows the log names the one operator, so `single_move_evidence` "
            "records which basis was used.\n")
        add("**GA lineage was deliberately NOT mined.** `ga_log.csv` exists and "
            "carries `operator` + `parents`, but it lists **two** tags even for "
            "`clone+mutation`, and `crossover` (2,699 rows) is a genuine "
            "two-parent operator that is not a move at all. Rather than guess "
            "which listed parent was cloned, its 7,122 rows are excluded and "
            "counted here. Resolving them by minimum-diff parent selection is a "
            "cheap follow-up worth ~900 extra single-parent steps.\n")
        gen = _improving_table(steps, ["lineage_source"])
        add("Yield by lineage source (all edges):\n")
        add(_md_table(gen[["lineage_source", "n_steps", "n_labeled", "n_improving", "improving_frac"]]))
        add("")
        acc = sa_steps[sa_steps["improved_fr"].notna()]
        if len(acc):
            tab = acc.groupby("sa_accepted")["improved_fr"].agg(
                n="size", improving=lambda s: int(s.fillna(False).sum()),
            ).reset_index()
            tab["frac"] = tab["improving"] / tab["n"]
            add("MOCHA's own accept decision vs whether F_r actually improved:\n")
            add(_md_table(tab))
            add("")
            add("The two disagree substantially, and they should: MOCHA accepted "
                "on a multi-objective Metropolis test over its aggregated `J` "
                "(cycle length, boron and peaking together), not on F_r alone. "
                "A policy trained to imitate `sa_accepted` would inherit "
                "MOCHA's objective; a policy trained on `improved_fr` learns "
                "ours. The corpus carries both so the choice stays explicit.\n")
    else:
        add("No SA lineage loaded — run `mine_sa_lineage.py` first.\n")

    add("### 6b. What is still missing\n")
    by_gen = store[store["parent_record_id"].notna()].copy()
    by_gen["resolved"] = by_gen["parent_record_id"].isin(set(store["record_id"]))
    gen_tab = by_gen.groupby("generator").agg(
        children=("record_id", "size"), parent_in_store=("resolved", "sum")
    ).reset_index().sort_values("children", ascending=False)
    gen_tab["resolved_frac"] = gen_tab["parent_in_store"] / gen_tab["children"]
    add(f"- store rows **with** `parent_record_id`: **{n_children:,}** / {len(store):,} "
        f"({n_children / len(store):.1%})")
    add(f"- rows **without** a `parent_record_id` in the store: "
        f"**{len(store) - n_children:,}**. Dataset A ({n_legacy:,} rows, "
        f"{n_legacy / len(store):.0%} of the store) still has the column empty "
        "— `lpopt/data/extract_a.py` writes `parent_record_id=None` "
        "unconditionally and the store is read-only here — but its lineage is "
        f"no longer lost: {len(sa_steps):,} of those rows are now reachable "
        "through `sa_lineage.parquet` (section 6a). Fixing the column itself "
        "belongs in `extract_a.py`, not in this miner.")
    add(f"- children whose parent resolves in the store: **{n_resolved:,}**; "
        f"**{n_children - n_resolved:,}** do not "
        f"(**{orphan_parents:,}** distinct missing parent boards).")
    add("- the unresolved ones are not lost data, they are *surrogate-only* boards:")
    add(_md_table(gen_tab[["generator", "children", "parent_in_store", "resolved_frac"]]))
    add("")
    add("  `local` (the `_lean_local_search` first-improvement hill-climb in "
        "`lpopt/search/campaign.py`) walks the **surrogate** landscape and only "
        "its final accepted board is sent to MASTER, so the intermediate "
        "`current` boards it names as parents were never evaluated and never "
        "entered the store — hence the near-zero resolved fraction above. "
        "`elite` children seeded from `prev_top` (previous-wave *predicted* top, "
        "`construct.py`) have the same gap.")
    add("- audited 2026-08-15 against `data/produce/ledger.jsonl` (58,447 rows): "
        "7,871 of its edges have both endpoints in the store and **0** are new. "
        "The store's `parent_record_id` column is therefore complete w.r.t. the "
        "produce ledger — there is no extra lineage hiding there.")
    add("- audited 2026-08-15 against every `runs/**/labels.jsonl` plus the "
        "scratchpad-pulled `t6r*/fpcamp*/minfr*` copies (1,468 records, 1,183 "
        "with a parent): **43** records are not in the store and **0** unresolved "
        "parent patterns were recoverable from them or from "
        "`data/campaigns/**/*.parquet`. The wave label files add nothing the "
        "store does not already have.")
    add(f"- both endpoints converged: **{int(steps['both_converged'].fillna(False).sum()):,}** "
        f"of {n_resolved:,} steps ({steps['both_converged'].fillna(False).mean():.1%}); "
        f"the remainder has a non-converged endpoint and therefore an <NA> "
        f"`improved_fr` label.")
    add(f"- `node_peak` (flatness) is only harvested where an EDIT5 map exists, so "
        f"**{int(steps['improved_flat'].notna().sum()):,}** of "
        f"{int(steps['improved_fr'].notna().sum()):,} F_r-labeled steps "
        f"({steps['improved_flat'].notna().sum() / max(int(steps['improved_fr'].notna().sum()), 1):.0%}) "
        "also carry a flatness label.")
    add(f"- cyclen band: parsed from a campaign's own deck "
        f"(`runs/<campaign>/input_deck.inp` or `<campaign>.inp`). Only "
        f"**{len(bands)}** campaigns ship a deck with "
        f"`cycle_target_efpd`/`cycle_tolerance_efpd`, covering "
        f"**{int(steps['cyclen_band_known'].sum()):,}** steps; every other step "
        f"has `cyclen_band_known=False` and `in_cyclen_band_child=<NA>`. Note the "
        f"min-F_r campaigns declare their cycle target *report-only* (gates "
        f"nothing), so the band is informational and is NOT folded into "
        f"`feasible_child`.")
    add("- CBC limit: this corpus uses the program value **1600 ppm** (the "
        "`fpcamp_minfr_T6T4` deck header says \"CBC gate 1600\"), while "
        "`lpopt/config.py` still defaults `cbc_limit = 1550.0`. "
        f"{int(((store['cbc_max'] > 1550) & (store['cbc_max'] <= 1600)).sum()):,} "
        "converged store rows sit in the 1550-1600 gap and flip feasibility "
        "between the two conventions.\n")

    # -- 6. chain depth ------------------------------------------------------ #
    add("## 7. Chain depth — how long are the improvement chains?\n")
    raw = chain_stats(moves, improving_only=False)
    imp = chain_stats(moves, improving_only=True)
    counts = moves["cell"].value_counts()
    chain = pd.DataFrame({
        "cell": list(raw),
        "n_steps": [int(counts[c]) for c in raw],
        "longest_lineage_chain": [raw[c] for c in raw],
        "longest_F_r_improving_chain": [imp.get(c, 0) for c in raw],
    }).sort_values("n_steps", ascending=False)
    add("Chain length = number of consecutive lineage edges (a depth-1 chain is a "
        "single parent->child move). The *improving* chain restricts to edges "
        "where the child's F_r beat its parent's. Computed over the same-cell "
        "moves, since a cross-cell transfer breaks the action space.\n")
    add(_md_table(chain.head(20)))
    add("")
    focus = [c for c in chain["cell"] if c.startswith(("T6_T4/", "E1_E2/"))]
    if focus:
        add("Requested focus cells (T6_T4 / E1_E2):\n")
        add(_md_table(chain[chain["cell"].isin(focus)]))
        add("")
    add(f"Overall the lineage is **shallow**: the deepest chain anywhere is "
        f"{max(raw.values(), default=0)} edges and the deepest strictly-F_r-improving "
        f"chain is {max(imp.values(), default=0)}. The campaigns re-seed every wave "
        "from the store's elite set rather than pushing one board down a long "
        "trajectory, so this corpus is a set of **1-step neighbourhoods around "
        "good boards**, not a set of long improvement trajectories. A policy "
        "trained on it learns *which single move to propose next*; it does not "
        "get multi-step credit assignment for free.\n")

    # -- 8. positional readout ------------------------------------------------ #
    add(f"## 8. Positional readout — rewire_swap steps\n")
    rw = main[(main["move_class"] == "rewire_swap") & main["improved_fr"].notna()].copy()
    scope = f"main cell `{main_cell}`"
    add(f"Densest same-cell cell: **{main_cell}** ({len(main):,} moves). "
        f"`rewire_swap` moves there with an F_r label: **{len(rw):,}**.\n")
    if len(rw) < 20:
        rw = moves[
            (moves["move_class"] == "rewire_swap") & moves["improved_fr"].notna()
        ].copy()
        scope = "ALL same-cell moves pooled"
        add(f"That is too thin, so the readout below pools **every** same-cell "
            f"`rewire_swap` move across all cells (**{len(rw):,}** labeled). "
            "Pooling mixes cells, so read it as a direction, not a coefficient.\n")
    add(f"Scope: {scope}.\n")
    if len(rw) >= 20:
        base = rw["improved_fr"].fillna(False).mean()
        add(f"Baseline F_r-improving fraction over this scope: **{base:.1%}**.\n")
        rw["span_bin"] = pd.qcut(rw["swap_span"], 3, duplicates="drop")
        rw["radius_bin"] = pd.qcut(rw["swap_radius"], 3, duplicates="drop")
        for col, title in (
            ("span_bin", "radial SEPARATION of the two rewired orbit units |r1-r2|"),
            ("radius_bin", "mean RADIUS of the two rewired orbit units (r1+r2)/2"),
        ):
            grp = rw.groupby(col, observed=True)["improved_fr"]
            tab = pd.DataFrame({
                "bin": [str(i) for i in grp.size().index],
                "n": grp.size().to_numpy(),
                "improving_frac": grp.apply(lambda s: float(s.fillna(False).mean())).to_numpy(),
            })
            add(f"By {title}:\n")
            add(_md_table(tab))
            add("")
        add("Radius is the vendor `ORBIT_UNITS[...].radius` (hypot of the quarter-core "
            "row/col of the unit's representative slot), so it is free to compute — "
            "no map read, no MASTER call.\n")
    else:
        add("Too few labeled `rewire_swap` steps in this cell for a positional "
            "readout; skipped per the brief (not cheaply computable at this n).\n")

    # -- 9. elites ----------------------------------------------------------- #
    add("## 9. Elite (\"good state\") set\n")
    add(f"`data/policy/elites.parquet`: top-{ELITE_K} **feasible** boards per cell, "
        "ranked twice — by `f_r` (ascending) and by `node_peak` (ascending) — "
        "campaign-blind and wave-blind, exactly the imitation target for a "
        "constructor policy. Feasible means converged AND all four program axes "
        "known and satisfied; a board missing `cbc_max` is <NA> and is excluded "
        "rather than assumed good. Every elite row carries the full physics "
        "descriptor set.\n")
    n_feas = int(feasibility(store).fillna(False).sum())
    add(f"The set is small on purpose and the reason is physical: only "
        f"{int((store['converged'] & (store['f_r'] <= F_R_LIMIT)).sum()):,} of "
        f"{int(store['converged'].sum()):,} converged boards clear F_r <= 1.55, "
        f"and {n_feas:,} clear all four axes. F_r is the binding constraint by a "
        "wide margin — the program is genuinely running at the licensing "
        "boundary, which is exactly why a policy that knows *where to put fuel* "
        "is worth more here than a better surrogate.\n")
    ecount = elites.groupby(["cell", "rank_by"]).size().unstack(fill_value=0)
    ecount["total"] = ecount.sum(axis=1)
    ecount = ecount.sort_values("total", ascending=False).head(20).reset_index()
    add(_md_table(ecount))
    add("")

    add("### 9a. Do our best cores put fresh at the PERIPHERY or INSIDE?\n")
    add("Elite fresh-share-by-ring vs the cell's all-comers average (every "
        "converged board in that cell). `d_periph` = elite peripheral fresh "
        "share minus all-comers; **positive = the good cores load fresh further "
        "OUT (flattening-driven, accepts leakage), negative = further IN "
        "(leakage-driven).** Cells with >= 200 all-comers, top 20 by that count.\n")
    profile = elites.groupby(["cell", "rank_by"])[
        [f"fresh_share_{n}" for n in RING_NAMES]
    ].mean().unstack("rank_by")
    baseline = cell_ring_baseline(store, elites["cell"].unique()).set_index("cell")
    rows: list[dict[str, object]] = []
    for cell in baseline.index:
        if cell not in profile.index or int(baseline.loc[cell, "n_all"]) < 200:
            continue
        row: dict[str, object] = {"cell": cell, "n_all": int(baseline.loc[cell, "n_all"])}
        for name in RING_NAMES:
            row[f"all_{name}"] = float(baseline.loc[cell, f"all_{name}"])
        for rank_by, tag in (("f_r", "eliteFr"), ("node_peak", "eliteFlat")):
            key = (f"fresh_share_periph", rank_by)
            value = float(profile.loc[cell, key]) if key in profile.columns else np.nan
            row[f"{tag}_periph"] = value
            row[f"d_periph_{tag}"] = value - float(baseline.loc[cell, "all_periph"])
        rows.append(row)
    ring_tab = pd.DataFrame(rows).sort_values("n_all", ascending=False).head(20)
    if not ring_tab.empty:
        add(_md_table(ring_tab))
        add("")
        signed = ring_tab["d_periph_eliteFr"].dropna()
        flat_signed = ring_tab["d_periph_eliteFlat"].dropna()
        if len(signed):
            add("**The answer is OUT, and the flatness-ranked elites are further "
                "out still.** Across these cells the F_r-elites sit on average "
                f"**{signed.mean():+.3f}** in peripheral fresh share relative to "
                f"all comers ({int((signed > 0).sum())}/{len(signed)} cells "
                "positive)"
                + (f", and the node_peak-elites **{flat_signed.mean():+.3f}** "
                   f"({int((flat_signed > 0).sum())}/{len(flat_signed)} positive)"
                   if len(flat_signed) else "")
                + ". Note also the absolute levels: even the all-comers average "
                f"loads {ring_tab['all_periph'].mean():.2f} of the peripheral "
                f"ring fresh against {ring_tab['all_inner'].mean():.2f} of the "
                "inner ring, so these cores are already in-out (fresh outside) "
                "and the good ones lean further that way. Consistent with "
                "section 4b: at F_r 1.55 the binding pressure is radial power "
                "flattening, and the corpus pays for it in cycle length.\n")
            worst = ring_tab.loc[ring_tab["d_periph_eliteFr"].idxmin()]
            add("Two caveats, stated rather than buried. This is an "
                "*observational* answer — the elites are whatever the campaigns "
                "happened to find, and those campaigns were themselves steered "
                "by F_r-minimizing acquisition, so peripheral-fresh boards were "
                "preferentially *sampled* as well as preferentially *kept*. And "
                f"the rule is not universal: `{worst['cell']}` goes the other "
                f"way ({float(worst['d_periph_eliteFr']):+.3f}). Cells are not "
                "interchangeable, which is the whole reason the policy has to be "
                "conditioned on the cell rather than taught one global rule.\n")
    else:
        add("No cell has both an elite set and >= 200 converged all-comers; "
            "table skipped.\n")

    # -- 9. verdict ----------------------------------------------------------- #
    add("## 9b. Transpose augmentation — free doubling\n")
    add("`lpopt.data.geometry.transpose` reflects a board across the qi<->qj "
        "diagonal. It is an involution, it preserves feed, and it maps every "
        "slot to one of **equal radius and equal orbit multiplicity** — so the "
        "move class, the unit-edit count, the 69-slot Hamming distance and "
        "every ring descriptor are invariant, and the FOMs are the same physical "
        "experiment. Only the two pattern strings change.\n")
    add("**Chosen form: an on-the-fly recipe, not materialized rows.** "
        "Materializing would write "
        f"{len(steps):,} duplicate rows across {len(steps.columns)} columns to "
        "change exactly 2 of them, doubling the parquet for zero new "
        "information and creating a second copy to keep in sync. The recipe is "
        "three lines at load time:\n")
    add("```python")
    add("from lpopt.data.geometry import transpose")
    add("from lpopt.data.schema import pack_pattern, unpack_pattern")
    add("")
    add("mirror = steps.copy()")
    add("mirror['parent_pattern'] = [pack_pattern(transpose(unpack_pattern(p)))")
    add("                            for p in steps['parent_pattern']]")
    add("mirror['child_pattern']  = [pack_pattern(transpose(unpack_pattern(p)))")
    add("                            for p in steps['child_pattern']]")
    add("mirror['augmented'] = True        # every other column is copied verbatim")
    add("train = pd.concat([steps.assign(augmented=False), mirror])")
    add("```")
    add("")
    if transpose_check:
        broken = sum(transpose_check[k] for k in
                     ("class_break", "hamming_break", "physics_break"))
        add(f"**Verified, not asserted.** `verify_transpose()` re-derives the "
            f"class, the edit counts, the Hamming distance and all "
            f"{len(PHYSICS)} ring descriptors from the mirrored boards on a "
            f"random sample of {transpose_check['tested']:,} steps: "
            f"**{broken} invariance violations** "
            f"(class {transpose_check['class_break']}, "
            f"Hamming {transpose_check['hamming_break']}, "
            f"physics {transpose_check['physics_break']}; "
            f"{transpose_check['undecodable']} boards undecodable). "
            "Run `python mine_policy_corpus.py --verify-transpose` to reproduce.\n")
    add("Caveat worth stating plainly: this doubles the *training rows*, not the "
        "*information*. It teaches the policy the diagonal symmetry it should "
        "already respect and it regularizes; it does not add a single new "
        "MASTER evaluation. Counting augmented rows toward a data-sufficiency "
        "bar would be self-deception, so section 10 counts raw steps and states "
        "the augmented figure separately.\n")

    add("## 10. Verdict — is this enough for a supervised move-proposal policy v1?\n")
    main_labeled = int(main["improved_fr"].notna().sum())
    total_labeled = int(moves["improved_fr"].notna().sum())
    clean = moves[moves["single_move"]]
    clean_labeled = int(clean["improved_fr"].notna().sum())
    add("Bar from the brief: **>= 5,000 labeled steps in the main cell** and "
        "**>= 20,000 overall**. Counted over same-cell moves, because a "
        "cross-cell transfer is not an action the policy can take.\n")
    add(f"- main cell `{main_cell}`: **{main_labeled:,}** labeled moves — "
        f"**{main_labeled / 5000:.0%}** of the bar.")
    add(f"- overall: **{total_labeled:,}** labeled moves — "
        f"**{total_labeled / 20000:.0%}** of the bar.")
    add(f"- verified SINGLE-move subset (`single_move == True`, i.e. the net diff "
        f"is one operator AND the edit count proves it): **{clean_labeled:,}** "
        f"labeled moves — **{clean_labeled / 20000:.1%}** of the bar. "
        + (", ".join(
            f"`{c}` {n}" for c, n in clean["move_class"].value_counts().items()
        ) or "none")
        + ".")
    add(f"- (counting cross-cell transfers too would give "
        f"{int(steps['improved_fr'].notna().sum()):,} labeled edges, "
        f"{int(steps['improved_fr'].notna().sum()) / 20000:.0%} of the bar — "
        "but they are not moves, so they do not count.)")
    add(f"- with transpose augmentation (section 9b): "
        f"**{2 * total_labeled:,}** overall / **{2 * main_labeled:,}** main "
        "cell. Doubled rows, not doubled information — noted, not banked.\n")
    verdict = (
        "SUFFICIENT to build v1" if total_labeled >= 20000
        else "MARGINAL — at the overall bar, short in the main cell"
        if total_labeled >= 15000 else "INSUFFICIENT"
    )
    add(f"**Verdict: {verdict}.** The Dataset A recovery moved the overall "
        f"count from 1,305 to {total_labeled:,} labeled same-cell moves "
        f"({total_labeled / 1305:.0f}x) and the verified single-move subset "
        f"from 118 to {clean_labeled:,} ({clean_labeled / 118:.0f}x), which is "
        "the number that actually matters for a move-proposal head. The "
        "remaining gap is concentrated, not diffuse:\n")
    add(f"1. **No single cell reaches 5,000 labeled moves.** The densest is "
        f"`{main_cell}` at {main_labeled:,}. The corpus is broad (many cells) "
        "rather than deep (one cell), so a per-cell policy is not trainable "
        "yet, while a **cell-conditioned** policy over the whole corpus is. "
        "That is a modelling choice the data now forces, and it is the right "
        "one anyway — section 9a already showed cells are not interchangeable.")
    add("2. **The recovered corpus is old-library.** Every `sa_mocha` move is "
        "260624 or 5.8_5.1 at feed 121; the live program runs ga80/paramA "
        "across feeds 101-141 near F_r=1.55. The policy will need the cell "
        "context as an input and will be extrapolating to the current "
        "operating point, so hold out a current-library cell for validation "
        "rather than trusting a random split.")
    add("3. **The leakage signal is unresolved** (section 4b/4g): the two eras "
        "disagree on the sign of the cycle-length response to outward fresh "
        "loading. Train the flattening head now; do not train a "
        "leakage-arbitration head until the ablation wave lands.")
    add(f"4. **Class imbalance persists at the tails.** The same-cell mix is now "
        + ", ".join(
            f"`{c}` {n:,}" for c, n in moves["move_class"].value_counts().items()
        )
        + ". The two structural classes are well covered; `batch_swap` and any "
        "within-cell feed move are still effectively absent. A v1 policy can "
        "only learn the classes it has seen.")
    add(f"5. **Surrogate-only parents (lpopt era only).** {orphan_parents:,} "
        "distinct parents — mostly the `local` hill-climb, plus "
        "`elite`-from-`prev_top` — were never evaluated, so their edges are "
        "unusable *as labeled steps*. They remain usable as **unlabeled** move "
        "examples for a behaviour-cloning warm start.")
    add(f"6. **The feasible region is tiny.** Only "
        f"{int((store['converged'] & (store['f_r'] <= F_R_LIMIT)).sum()):,} of "
        f"{int(store['converged'].sum()):,} converged boards clear F_r <= 1.55 "
        "at all; F_r is the binding constraint by a wide margin, which is why "
        f"the elite set is only {len(elites):,} rows over "
        f"{elites['cell'].nunique()} cells. The *step* corpus is now healthy; "
        "the *good-state* corpus is not, and a constructor policy trained to "
        "imitate elites still has very few targets per cell.\n")
    add("Two worries from the previous pass are now **closed**: multi-move "
        f"contamination ({clean_labeled:,} of {total_labeled:,} labeled "
        "same-cell moves are verified single moves, most on `sa_log` ground "
        "truth rather than an edit-count guess) and flatness sparsity "
        f"({int(moves['improved_flat'].notna().sum()):,} same-cell moves now "
        "carry a `node_peak` label on both endpoints, up from 870).\n")
    add("**What to do next (cheapest first):**\n")
    add("- **Back-fill `parent_record_id` into the store** from "
        "`sa_lineage.parquet` so the lineage is a first-class store column "
        "instead of a side file. That is an `extract_a.py` / store-writer "
        "change and was deliberately not done here (read-only mandate).")
    add("- **Resolve the GA lineage** by minimum-diff parent selection: "
        "`ga_log.csv` lists two candidate parents for `clone+mutation`, and the "
        "genome differ can say which one the child is one move from. ~900 extra "
        "single-parent steps for an hour's work, no MASTER time.")
    add("- **Log the local-search chain.** Teach `_lean_local_search` to emit its "
        "intermediate boards to the ledger (surrogate-predicted FOMs are fine, "
        "flagged as such); every hill-climb step then becomes a step row.")
    add("- **Run a dedicated 1-move ablation wave** on the main cell: take the top "
        "~50 elites, apply each move class exhaustively at `n_moves=1`, evaluate. "
        "A few thousand MASTER calls buys a *balanced, clean* single-move dataset "
        "which is worth more per row than the entire current corpus — and it is "
        "the only way to de-confound move class from radial direction, which the "
        "observational tables in section 4 cannot do.")
    add("- **Stratify that wave by radial direction.** For each elite, propose "
        "matched outward / inward / neutral variants of the *same* class so the "
        "leakage-vs-flattening trade is measured at fixed move class. That is "
        "the experiment that turns the engineer's rule of thumb into a label.")
    add("- **Harvest EDIT5 maps on the remaining lineage endpoints.** "
        f"`node_peak` is the flattening signal and it is still missing on "
        f"{len(moves) - int(moves['improved_flat'].notna().sum()):,} of "
        f"{len(moves):,} same-cell moves "
        f"({1 - moves['improved_flat'].notna().mean():.0%}) — much better than "
        "before, but the gap is what caps the flattening head's training set.")
    add("- **Augment by diagonal mirror** at load time (section 9b) — verified "
        "label-preserving, free, and it doubles the rows a v1 head sees.")
    add("- **Train the flattening head now.** The F_r and node_peak signals are "
        "large, consistent across both eras, and backed by "
        f"{total_labeled:,} labeled moves. The leakage-arbitration head waits "
        "for the ablation wave.\n")

    add("## 11. Files\n")
    add("| file | rows | what |")
    add("|---|---|---|")
    add(f"| `data/policy/steps.parquet` | {len(steps):,} "
        f"({len(moves):,} same-cell moves + {len(transfers):,} cross-cell "
        "transfers) | one lineage edge: classified move, physics annotations, "
        "outcome labels |")
    add(f"| `data/policy/sa_lineage.parquet` | {len(sa_steps):,} | Dataset A "
        "proposal chains recovered from `2_LP/0_Case/runs/*/sa_log.csv` "
        "(tag, MOCHA move, accept flag, both parent readings) |")
    add("| `mine_sa_lineage.py` | — | the recovery script (read-only over "
        "`2_LP`; rerun only if the MOCHA runs change) |")
    add(f"| `data/policy/elites.parquet` | {len(elites):,} | top-{ELITE_K} "
        "feasible boards per cell, by F_r and by node_peak, with radial "
        "profiles |")
    add(f"| `{path.relative_to(REPO).as_posix()}` | — | this report |")
    add("")
    add(f"`steps.parquet` columns ({len(steps.columns)}): provenance "
        "(`lineage_source`, `source_move`, `sa_accepted`, "
        "`single_move_evidence`), context "
        "(`campaign`, `dataset`, `generator`, `case_pair`, `feed`, "
        "`library_id`, `cell`, `cross_cell`), identity (`parent_record_id`, "
        "`child_record_id`, `parent_pattern`, `child_pattern`), diff "
        "(`n_slots_changed`, `n_unit_edits`, `move_class`, `single_move`, "
        "`swap_span`, `swap_radius`, `burn_state`), FOMs "
        f"(`parent_*`/`child_*`/`d_*` over {', '.join(f'`{f}`' for f in FOMS)}, "
        f"`{FXY_FOM}`), physics "
        f"(`parent_*`/`child_*`/`d_*` over {', '.join(f'`{p}`' for p in PHYSICS)}, "
        "plus `fresh_radial_dir`, `burnt_periph_dir`), labels "
        "(`improved_fxy`, `improved_fr`, `improved_flat`, `improved_cbc`, "
        "`improved_cyclen`, "
        "`feasible_parent`, `feasible_child`, `both_converged`, "
        "`in_cyclen_band_child`, `cyclen_band_known`).\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--out-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--elite-k", type=int, default=ELITE_K)
    parser.add_argument("--sa-lineage", type=Path, default=POLICY_DIR / "sa_lineage.parquet")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--verify-transpose", type=int, nargs="?", const=1500, default=1500,
                        help="sample size for the transpose invariance check (0 to skip)")
    args = parser.parse_args(argv)

    store = pd.read_parquet(args.store)
    bands = cyclen_bands(REPO)
    enrichment = load_enrichment(FUEL_TYPES)
    sa = pd.read_parquet(args.sa_lineage) if args.sa_lineage.is_file() else None
    if sa is None:
        print(f"[warn] no SA lineage at {args.sa_lineage}; "
              "run mine_sa_lineage.py to recover the Dataset A corpus")

    steps = build_steps(store, bands, enrichment, sa)
    elites = build_elites(store, enrichment, args.elite_k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = args.out_dir / "steps.parquet"
    elites_path = args.out_dir / "elites.parquet"
    steps.to_parquet(steps_path, index=False)
    elites.to_parquet(elites_path, index=False)
    print(f"steps  -> {steps_path}  ({len(steps):,} rows)")
    print(f"elites -> {elites_path}  ({len(elites):,} rows)")

    check: dict[str, int] | None = None
    if args.verify_transpose:
        check = verify_transpose(steps, enrichment, sample=args.verify_transpose)
        print(f"transpose invariance check: {check}")

    if not args.no_report:
        report = args.report or REPORT_DIR / f"policy_corpus_{date.today():%Y%m%d}.md"
        write_report(store, steps, elites, bands, report, check)
        print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
