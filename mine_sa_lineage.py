"""Recover Dataset A (MOCHA SA) lineage that ``extract_a.py`` discarded.

``lpopt/data/extract_a.py`` writes ``parent_record_id=None`` for every Dataset A
row, so the legacy MOCHA corpus — 53% of the store — contributes zero steps to
the policy corpus.  The lineage was never in the ``sa_2b_cache`` records to begin
with (they carry only ``key`` + ``rec``), but it *is* recoverable, exactly and
with zero MASTER cost, from the per-run **``sa_log.csv``** the optimizer writes
alongside the cache.

What the log actually encodes
-----------------------------
``2_LP/MOCHA/optimizer.py`` runs synchronous parallel SA::

    # batch of moves from the SAME incumbent, evaluated concurrently
    # (synchronous parallel SA); acceptance below stays strictly sequential
    for b in range(nbatch):
        cand, mv = self._move(cur)
        batch.append((cand, f"s{stage:02d}_{k + b:03d}", mv))

so ``sa_log.csv`` gives, per evaluated candidate: its ``tag``, the **``move``**
that produced it (MOCHA's own operator name — no inference needed), and whether
it was ``accepted``.  The parent link is therefore a **PROPOSAL chain, not an
accept chain**: the parent of a candidate is the incumbent it was mutated from,
whether or not the candidate was subsequently accepted.  Rejected proposals keep
their parent and become negative training examples, which is precisely what a
move-proposal policy needs.

Two readings of "the incumbent" are possible and they differ inside a batch:

``batch``       the incumbent at batch start — what ``_move`` was actually
                called on.  This is the physically correct parent.
``sequential``  the most recently accepted candidate — what the Metropolis test
                compared against (``J_cur``).

Both are reconstructed and their genome diff sizes compared; the script reports
the evidence rather than assuming (see ``--verify``).

Tag -> record_id
----------------
``rec.tag`` in the cache is NOT unique (6,086 tags map to more than one board —
tags are per-run and the cache is global), so the join goes through the run's own
case directories: ``runs/<run>/cases/<tag>/cy<NN>/MAS_INP`` -> ``%LPD_SHF`` ->
canonical pattern -> :func:`extract_a.dedup_key_of`, which is the same key
``extract_a`` used to build the store.

GA lineage is deliberately NOT mined: ``ga_log.csv`` records two tags in its
``parents`` column even for ``clone+mutation``, and ``crossover`` is a genuine
two-parent operator that is not a move at all.  7,122 GA rows are counted and
reported, not guessed at.

Usage::

    python mine_sa_lineage.py                  # write data/policy/sa_lineage.parquet
    python mine_sa_lineage.py --verify         # + parent-variant diff evidence
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from lpopt.data.extract_a import _CY_RE, _read_text_flex, dedup_key_of
from lpopt.data.geometry import to_cache_key, to_canonical_from_shf
from lpopt.data.schema import unpack_pattern

REPO = Path(__file__).resolve().parent
STORE = REPO / "data" / "store" / "records.parquet"
OUT = REPO / "data" / "policy" / "sa_lineage.parquet"

#: MOCHA workspace whose cache produced the store's Dataset A rows.  The
#: ``99_Archive`` caches were never ingested (their campaigns do not appear in
#: the store), so mining them would produce edges with no store endpoints.
DEFAULT_WORKSPACE = (
    REPO.parent / "2_LP" / "0_Case"
)

#: MOCHA SA operator -> the lpopt genome operator it corresponds to.  The two
#: eras share an operator basis, which is why the corpus is joinable at all.
#: ``compound_shuffle`` applies several primitives in one move and has no
#: single-operator counterpart; ``init`` is a seed, not a move.
MOCHA_MOVE_EQUIVALENT: dict[str, str] = {
    "swap_burned_sources": "rewire_swap",
    "change_fresh_type": "batch_flip",
    "swap_fresh_burned": "fresh_relocate",
    "compound_shuffle": "sa_unknown",
    "init": "sa_seed",
}


# --------------------------------------------------------------------------- #
# tag -> board
# --------------------------------------------------------------------------- #
def case_dedup_key(case_dir_str: str) -> tuple[str, tuple | None]:
    """Worker: ``cases/<tag>`` -> the extract_a dedup key of its final cycle.

    Mirrors :func:`extract_a._harvest_case_dir` minus the MAS_SUM harvest, so a
    board resolved here is keyed exactly as the store keyed it.
    """
    case_dir = Path(case_dir_str)
    try:
        cys = [
            (int(m.group(1)), d)
            for d in case_dir.iterdir()
            if d.is_dir() and (m := _CY_RE.match(d.name))
        ]
    except OSError:
        return (case_dir.name, None)
    if not cys:
        return (case_dir.name, None)
    final = max(cys, key=lambda t: t[0])[1]
    try:
        pattern = to_canonical_from_shf(_read_text_flex(final / "MAS_INP"))
        return (case_dir.name, dedup_key_of(to_cache_key(pattern)))
    except (OSError, ValueError, KeyError, AssertionError):
        return (case_dir.name, None)


def resolve_run_tags(run: Path, workers: int) -> dict[str, tuple]:
    """``tag -> dedup key`` for every case directory of one MOCHA run."""
    cases = run / "cases"
    if not cases.is_dir():
        return {}
    dirs = [str(d) for d in cases.iterdir() if d.is_dir()]
    if not dirs:
        return {}
    out: dict[str, tuple] = {}
    if workers <= 1:
        results: Iterable[tuple[str, tuple | None]] = map(case_dedup_key, dirs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(case_dedup_key, dirs, chunksize=32)
    for tag, key in results:
        if key is not None:
            out[tag] = key
    if workers > 1:
        pool.shutdown()
    return out


def store_key_index(store: pd.DataFrame) -> tuple[dict[tuple, str], int]:
    """``dedup key -> record_id`` over the Dataset A rows, plus a collision count.

    Dataset A is the MOCHA corpus; a dedup key that resolves to more than one
    record_id is dropped rather than guessed (it would mean the same board under
    two fuel libraries).
    """
    legacy = store[store["dataset"] == "A"]
    by_key: dict[tuple, set[str]] = {}
    for rid, packed in zip(legacy["record_id"], legacy["pattern"], strict=True):
        try:
            key = dedup_key_of(to_cache_key(unpack_pattern(packed)))
        except (ValueError, KeyError, AssertionError):
            continue
        by_key.setdefault(key, set()).add(rid)
    index = {k: next(iter(v)) for k, v in by_key.items() if len(v) == 1}
    return index, sum(1 for v in by_key.values() if len(v) > 1)


# --------------------------------------------------------------------------- #
# sa_log.csv -> proposal chain
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"^s(?P<stage>\d+)_(?P<offset>\d+)$")


@dataclass(frozen=True, slots=True)
class SaRow:
    tag: str
    stage: int
    offset: int
    move: str
    accepted: bool
    status: str


def read_sa_log(run: Path) -> list[SaRow]:
    path = run / "sa_log.csv"
    if not path.is_file():
        return []
    rows: list[SaRow] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for raw in csv.DictReader(handle):
            tag = (raw.get("tag") or "").strip()
            match = _TAG_RE.match(tag)
            rows.append(SaRow(
                tag=tag,
                stage=int(match.group("stage")) if match else -1,
                offset=int(match.group("offset")) if match else -1,
                move=(raw.get("move") or "").strip(),
                accepted=str(raw.get("accepted") or "").strip() == "1",
                status=(raw.get("status") or "").strip(),
            ))
    return rows


def parent_chain(rows: Sequence[SaRow], batch_size: int) -> dict[str, tuple[str, str]]:
    """``child tag -> (batch parent tag, sequential parent tag)``.

    ``batch_size`` is the run's ``sa.parallel_workers``: proposals sharing a
    ``(stage, offset // batch_size)`` bucket were all generated from the same
    incumbent, before any of them could be accepted.
    """
    out: dict[str, tuple[str, str]] = {}
    incumbent: str | None = None      # sequential (Metropolis) incumbent
    batch_incumbent: str | None = None  # incumbent at the current batch's start
    current_batch: tuple[int, int] | None = None
    for row in rows:
        if row.move == "init" or row.stage < 0:
            if row.accepted:
                incumbent = batch_incumbent = row.tag
            continue
        bucket = (row.stage, row.offset // max(batch_size, 1))
        if bucket != current_batch:
            current_batch = bucket
            batch_incumbent = incumbent
        if incumbent is not None and batch_incumbent is not None:
            out[row.tag] = (batch_incumbent, incumbent)
        if row.accepted:
            incumbent = row.tag
    return out


def run_batch_size(run: Path) -> int:
    meta = run / "run_meta.json"
    if not meta.is_file():
        return 1
    try:
        sa = json.loads(meta.read_text(encoding="utf-8", errors="replace")).get("sa") or {}
    except (OSError, ValueError):
        return 1
    return max(1, int(sa.get("parallel_workers") or 1))


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def mine(workspace: Path, store: pd.DataFrame, workers: int) -> pd.DataFrame:
    index, collisions = store_key_index(store)
    print(f"[sa-lineage] store dedup index: {len(index):,} keys "
          f"({collisions:,} ambiguous keys dropped)")

    runs = sorted(d for d in (workspace / "runs").glob("*") if d.is_dir())
    frames: list[dict[str, object]] = []
    stats = {"runs": 0, "sa_rows": 0, "chained": 0, "tags_resolved": 0,
             "child_in_store": 0, "both_in_store": 0}
    for run in runs:
        rows = read_sa_log(run)
        if not rows:
            continue
        stats["runs"] += 1
        stats["sa_rows"] += len(rows)
        chain = parent_chain(rows, run_batch_size(run))
        stats["chained"] += len(chain)
        tags = resolve_run_tags(run, workers)
        stats["tags_resolved"] += len(tags)
        by_tag = {r.tag: r for r in rows}
        for child_tag, (batch_parent, seq_parent) in chain.items():
            child_key = tags.get(child_tag)
            if child_key is None:
                continue
            child_rid = index.get(child_key)
            if child_rid is None:
                continue
            stats["child_in_store"] += 1
            batch_rid = index.get(tags.get(batch_parent) or ())
            seq_rid = index.get(tags.get(seq_parent) or ())
            if batch_rid is None:
                continue
            stats["both_in_store"] += 1
            row = by_tag[child_tag]
            frames.append({
                "run_id": run.name,
                "child_tag": child_tag,
                "parent_tag": batch_parent,
                "seq_parent_tag": seq_parent,
                "child_record_id": child_rid,
                "parent_record_id": batch_rid,
                "seq_parent_record_id": seq_rid,
                "sa_move": row.move,
                "sa_move_equivalent": MOCHA_MOVE_EQUIVALENT.get(row.move, "sa_unknown"),
                "sa_accepted": row.accepted,
                "sa_status": row.status,
                "stage": row.stage,
                "offset": row.offset,
            })
        print(f"[sa-lineage] {run.name}: {len(rows):,} log rows, "
              f"{len(tags):,} tags resolved, {len(frames):,} edges so far")
    print(f"[sa-lineage] {stats}")
    return pd.DataFrame(frames)


def verify(edges: pd.DataFrame, store: pd.DataFrame) -> None:
    """Compare the two parent readings by genome diff size (smaller = truer)."""
    from mine_policy_corpus import classify_move, genome_of

    patterns = dict(zip(store["record_id"], store["pattern"], strict=True))
    cache: dict[str, object] = {}

    def genome(rid: str):
        if rid not in cache:
            cache[rid] = genome_of(patterns[rid])
        return cache[rid]

    sample = edges[edges["seq_parent_record_id"].notna()].head(4000)
    rows = []
    for child, batch_p, seq_p, move in zip(
        sample["child_record_id"], sample["parent_record_id"],
        sample["seq_parent_record_id"], sample["sa_move"], strict=True,
    ):
        try:
            child_g = genome(child)
            rows.append({
                "sa_move": move,
                "batch": classify_move(genome(batch_p), child_g).n_unit_edits,
                "sequential": classify_move(genome(seq_p), child_g).n_unit_edits,
                "batch_class": classify_move(genome(batch_p), child_g).move_class,
            })
        except (KeyError, ValueError):
            continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("[verify] no comparable edges")
        return
    print("\n[verify] median genome edit size by parent reading "
          f"(n={len(frame):,}):")
    print(frame.groupby("sa_move")[["batch", "sequential"]].median().to_string())
    print(f"\n[verify] overall  batch={frame['batch'].median():.1f}  "
          f"sequential={frame['sequential'].median():.1f}")
    print("\n[verify] MOCHA move name vs lpopt genome class (batch parent):")
    print(pd.crosstab(frame["sa_move"], frame["batch_class"]).to_string())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    store = pd.read_parquet(args.store, columns=["record_id", "dataset", "pattern"])
    edges = mine(args.workspace, store, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(args.out, index=False)
    print(f"sa_lineage -> {args.out}  ({len(edges):,} edges)")
    if args.verify and not edges.empty:
        verify(edges, store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
