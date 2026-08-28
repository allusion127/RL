"""Deterministic train/val split manifests (plan sec. 4.4 splits).

``make_splits`` builds and persists the S0..S4 split family as JSON manifests
of ``record_id`` lists (plus, for S1, the ancestry groups):

* **S0** — random 90/10.
* **S1** — ancestry-closure *group* split.  Groups are the connected
  components of a union-find over the ``campaign`` tag of every record and the
  ``parent_record_id`` lineage edges (non-null).  Whole groups are held out so a
  G3 child never lands in val while its parent trains (no group straddles).
* **S2** — leave-pair-out: every record of the held-out pairs (default
  ``{C3_C6, A01_B05}``, extendable for Dataset-B pairs) is val.
* **S3a** — ``feed == 117`` evaluation set (sign-test report; empty in a
  feed-121-only store).
* **S3b** — ``feed in {105, 113}`` evaluation set — ``awaiting_production``
  (currently empty; emitted so downstream has a stable handle).
* **S4** — enrichment-band leave-out (``e_core in [lo, hi)``; parameterized).

Every manifest is deterministic under ``seed`` and round-trips through JSON
byte-for-byte (``SplitManifest.to_json`` / ``from_json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
@dataclass
class SplitManifest:
    """One named split: train / val ``record_id`` lists + provenance."""

    name: str
    kind: str                          # random|group|leave_pair|filter
    seed: int
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    status: str = "ok"                 # ok|empty|awaiting_production
    predicate: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, Any] = field(default_factory=dict)

    # -- sizes -------------------------------------------------------------- #
    @property
    def n_train(self) -> int:
        return len(self.train_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_ids)

    def record_ids(self, fold: str = "train") -> list[str]:
        if fold == "train":
            return list(self.train_ids)
        if fold == "val":
            return list(self.val_ids)
        if fold == "all":
            return list(self.train_ids) + list(self.val_ids)
        raise ValueError(f"unknown fold {fold!r}; use train|val|all")

    # -- (de)serialization -------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(p)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitManifest":
        return cls(
            name=d["name"],
            kind=d["kind"],
            seed=int(d["seed"]),
            train_ids=list(d.get("train_ids", [])),
            val_ids=list(d.get("val_ids", [])),
            status=d.get("status", "ok"),
            predicate=dict(d.get("predicate", {})),
            groups=dict(d.get("groups", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# union-find
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:      # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: smaller key wins
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for x in self._parent:
            out.setdefault(self.find(x), []).append(x)
        return out


# --------------------------------------------------------------------------- #
# individual splits
# --------------------------------------------------------------------------- #
def _sorted_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["record_id"].astype(str).tolist())


def make_s0(df: pd.DataFrame, seed: int, val_frac: float = 0.10) -> SplitManifest:
    ids = _sorted_ids(df)
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_frac))
    val = set(shuffled[:n_val])
    train_ids = [i for i in ids if i not in val]
    val_ids = [i for i in ids if i in val]
    return SplitManifest(
        name="S0", kind="random", seed=seed,
        train_ids=train_ids, val_ids=val_ids,
        predicate={"val_frac": val_frac},
    )


def make_s1(df: pd.DataFrame, seed: int, val_frac: float = 0.10,
            max_group_frac: float = 0.25) -> SplitManifest:
    """Ancestry-closure group split (campaign tags + parent_record_id edges)."""
    ids = _sorted_ids(df)
    id_set = set(ids)
    uf = _UnionFind(ids)

    # campaign tag: all records sharing a campaign are one lineage cluster.
    by_campaign: dict[str, list[str]] = {}
    for rid, camp in zip(df["record_id"].astype(str), df["campaign"]):
        key = "campaign::" + ("" if camp is None else str(camp))
        by_campaign.setdefault(key, []).append(rid)
    for members in by_campaign.values():
        first = members[0]
        for other in members[1:]:
            uf.union(first, other)

    # parent_record_id lineage edges (when the parent is in this store).
    if "parent_record_id" in df.columns:
        for rid, parent in zip(df["record_id"].astype(str), df["parent_record_id"]):
            if parent is not None and not pd.isna(parent):
                parent = str(parent)
                if parent in id_set:
                    uf.union(rid, parent)

    groups = uf.groups()
    group_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_total = len(ids)
    target = val_frac * n_total
    cap = max_group_frac * n_total
    held: list[str] = []
    val_ids: list[str] = []
    n_held = 0
    for gk in group_keys:
        members = groups[gk]
        if len(members) > cap:
            continue                       # never let one huge group dominate val
        held.append(gk)
        val_ids.extend(members)
        n_held += len(members)
        if n_held >= target:
            break
    if not held:
        # every group exceeds the cap: hold out the single smallest group.
        smallest = min(group_keys, key=lambda g: len(groups[g]))
        held = [smallest]
        val_ids = list(groups[smallest])
    val_set = set(val_ids)
    train_ids = [i for i in ids if i not in val_set]
    val_ids_sorted = [i for i in ids if i in val_set]
    return SplitManifest(
        name="S1", kind="group", seed=seed,
        train_ids=train_ids, val_ids=val_ids_sorted,
        predicate={"val_frac": val_frac, "max_group_frac": max_group_frac},
        groups={
            "n_groups": len(groups),
            "held_out_groups": sorted(held),
            "held_out_group_sizes": {gk: len(groups[gk]) for gk in sorted(held)},
        },
    )


def make_leave_pair(df: pd.DataFrame, seed: int, name: str = "S2",
                    holdout_pairs: Sequence[str] = ("C3_C6", "A01_B05")
                    ) -> SplitManifest:
    ids = _sorted_ids(df)
    pairs = set(holdout_pairs)
    is_val = df["case_pair"].astype(str).isin(pairs)
    val_ids = sorted(df.loc[is_val, "record_id"].astype(str).tolist())
    val_set = set(val_ids)
    train_ids = [i for i in ids if i not in val_set]
    status = "ok" if val_ids else "empty"
    return SplitManifest(
        name=name, kind="leave_pair", seed=seed,
        train_ids=train_ids, val_ids=val_ids, status=status,
        predicate={"holdout_pairs": sorted(pairs)},
    )


def make_feed_filter(df: pd.DataFrame, seed: int, name: str,
                     feeds: Sequence[int], status_when_empty: str = "empty"
                     ) -> SplitManifest:
    ids = _sorted_ids(df)
    feed_set = set(int(f) for f in feeds)
    is_val = df["feed"].astype(int).isin(feed_set)
    val_ids = sorted(df.loc[is_val, "record_id"].astype(str).tolist())
    val_set = set(val_ids)
    train_ids = [i for i in ids if i not in val_set]
    status = "ok" if val_ids else status_when_empty
    return SplitManifest(
        name=name, kind="filter", seed=seed,
        train_ids=train_ids, val_ids=val_ids, status=status,
        predicate={"feed_in": sorted(feed_set)},
    )


def make_e_core_band(df: pd.DataFrame, seed: int, name: str = "S4",
                     lo: float = 5.43, hi: float = 5.50) -> SplitManifest:
    ids = _sorted_ids(df)
    e = pd.to_numeric(df["e_core"], errors="coerce")
    is_val = (e >= lo) & (e < hi)
    val_ids = sorted(df.loc[is_val.fillna(False), "record_id"].astype(str).tolist())
    val_set = set(val_ids)
    train_ids = [i for i in ids if i not in val_set]
    status = "ok" if val_ids else "empty"
    return SplitManifest(
        name=name, kind="filter", seed=seed,
        train_ids=train_ids, val_ids=val_ids, status=status,
        predicate={"e_core_band": [lo, hi]},
    )


# --------------------------------------------------------------------------- #
# curriculum split (retrain-safe: never ejects a whole non-121 band, per-cell
# deterministic eval holdout that is invariant to adding new campaigns)
# --------------------------------------------------------------------------- #
def _hash01(text: str) -> float:
    """Deterministic, process-stable float in ``[0, 1)`` from a string.

    Uses SHA-1 (NOT the salted builtin ``hash``) so a record's fold membership is
    a pure function of its ``record_id`` — identical across runs and unaffected by
    which *other* rows/campaigns exist in the store.
    """
    h = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)
    return h / float(1 << 64)


def _campaign_series(df: pd.DataFrame) -> pd.Series:
    """``campaign`` as plain strings (``None``/NaN -> ``""``), index-aligned."""
    return df["campaign"].map(lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))


def make_curriculum_split(
    records: pd.DataFrame,
    *,
    cells: Sequence[str],
    blind_probe_ids_by_cell: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
    val_frac: float = 0.10,
    cell_val_frac: float = 0.20,
    max_group_frac: float = 0.25,
    protect_feed: int = 121,
    name: str = "S1",
    cell_cap: float | None = None,
    reached_cells: Sequence[str] | None = None,
) -> SplitManifest:
    """Retrain split for the cell-sequential curriculum (plan sec. 12.2/12.3).

    Two disjoint pools are split by *different* rules and then merged into one
    train/val manifest whose byte format is identical to :func:`make_s1` (so the
    trainer needs no change):

    * **Legacy pool** — Dataset A/B rows plus any Dataset-P rows whose ``campaign``
      is NOT one of ``cells``.  Keeps the S1 ancestry-closure *group* mechanics
      (campaign tags + ``parent_record_id`` lineage edges), but the val-carving
      candidate pool is RESTRICTED to groups every member of which is
      ``feed == protect_feed`` (default 121).  A group touching any non-121 row is
      train-only, so a whole non-121 evaluation band (feed-117 Dataset-B, the
      P0 pathfinder's cross-feed rows, ...) can NEVER be ejected into val.

    * **Curriculum-cell pool** — Dataset-P rows whose ``campaign`` is a known cell
      id.  Per cell, a DETERMINISTIC, model-independent eval holdout goes to val:
      (a) the cell's blind-probe ``record_id``s (when they exist as store rows) are
      always pinned into val, then (b) the holdout is topped up to ~``cell_val_frac``
      of the cell's *converged* rows by stable hash of ``record_id`` (NOT an rng),
      so a row's fold membership never changes as other cells/rows are added.
      Everything else in the cell (the >= ``1 - cell_val_frac`` converged remainder
      plus all non-converged rows) goes to TRAIN — the in-band training rows the
      forensic audit proved are irreplaceable.

    The manifest's ``groups`` dict additionally records ``curriculum_val_by_cell``
    (the per-cell val ``record_id`` lists) so the honest no-regression gate can
    score both champions on exactly these held-out rows.  When ``cell_cap`` is
    given it is recorded as ``groups['curriculum_cell_cap']`` and, together with
    ``groups['cells']`` (the curriculum campaign ids), is read by the trainer to
    apply a higher sampling-weight cap to the curriculum rows (negative-transfer
    mitigation) — the manifest is the trainer's only channel, so no CLI list.

    **Reached / unreached quarantine (methodology guard).**  ``reached_cells`` is
    the subset of ``cells`` the driver has already STARTED (see
    ``CurriculumDriver._reached_cell_ids``).  A known cell that is NOT reached is a
    *future* cell whose rows may already be pre-merged into the store; those rows
    are **quarantined** — dropped from BOTH the train fold AND the metric val fold
    (they appear only under ``groups['quarantined_by_cell']``, never in
    ``train_ids``/``val_ids``, so ``LPDataset`` never materializes them for either
    gradient steps or the per-epoch composite metric that drives champion best-epoch
    selection).  This keeps each cell's blind-probe transfer measurement genuinely
    blind: nothing about a future cell touches the champion until the driver reaches
    it.  A quarantined cell's stable-hash 80/20 holdout is deliberately NOT computed
    here, so when the cell is later reached the split it gets is byte-identical to
    what it would have had if never quarantined (fold membership is a pure function
    of ``record_id``, independent of quarantine history — growth invariance extends
    to quarantine release).  ``reached_cells=None`` disables the distinction: every
    known cell is treated as reached, byte-identical to the pre-quarantine behavior.
    """
    bp = {str(k): [str(x) for x in v] for k, v in (blind_probe_ids_by_cell or {}).items()}
    cellset = {str(c) for c in cells}
    # A known cell that is NOT reached is quarantined (see docstring); None means
    # "no distinction" -> all known cells reached (pre-quarantine behavior).
    reached_set = set(cellset) if reached_cells is None \
        else {str(c) for c in reached_cells} & cellset

    ids_all = _sorted_ids(records)
    id_set = set(ids_all)
    rid_col = records["record_id"].astype(str)
    ds_col = records["dataset"].astype(str)
    camp_col = _campaign_series(records)
    feed_col = pd.to_numeric(records["feed"], errors="coerce")
    conv_col = records["converged"].astype(bool) if "converged" in records.columns \
        else pd.Series(False, index=records.index)

    is_curr = (ds_col == "P") & camp_col.isin(cellset)
    legacy_df = records[~is_curr]
    curr_df = records[is_curr]

    # ---- legacy pool: S1 ancestry closure, val candidates restricted to 121 --- #
    legacy_ids = _sorted_ids(legacy_df) if len(legacy_df) else []
    legacy_val: list[str] = []
    held_groups: list[str] = []
    group_sizes: dict[str, int] = {}
    n_legacy_groups = 0
    if legacy_ids:
        legacy_id_set = set(legacy_ids)
        uf = _UnionFind(legacy_ids)
        by_campaign: dict[str, list[str]] = {}
        for rid, camp in zip(legacy_df["record_id"].astype(str), _campaign_series(legacy_df)):
            by_campaign.setdefault("campaign::" + camp, []).append(rid)
        for members in by_campaign.values():
            first = members[0]
            for other in members[1:]:
                uf.union(first, other)
        if "parent_record_id" in legacy_df.columns:
            for rid, parent in zip(legacy_df["record_id"].astype(str),
                                   legacy_df["parent_record_id"]):
                if parent is not None and not pd.isna(parent):
                    parent = str(parent)
                    if parent in legacy_id_set:
                        uf.union(rid, parent)

        feed_by_rid = {str(r): f for r, f in zip(legacy_df["record_id"].astype(str),
                                                 pd.to_numeric(legacy_df["feed"], errors="coerce"))}
        groups = uf.groups()
        n_legacy_groups = len(groups)
        # a group is a val candidate only if EVERY member is feed == protect_feed
        def _pure_protect(members: list[str]) -> bool:
            return all(feed_by_rid.get(m) == protect_feed for m in members)

        candidates = sorted(gk for gk, mem in groups.items() if _pure_protect(mem))
        rng = random.Random(seed)
        rng.shuffle(candidates)
        n_legacy = len(legacy_ids)
        target = val_frac * n_legacy
        cap = max_group_frac * n_legacy
        n_held = 0
        for gk in candidates:
            members = groups[gk]
            if len(members) > cap:
                continue
            held_groups.append(gk)
            legacy_val.extend(members)
            group_sizes[gk] = len(members)
            n_held += len(members)
            if n_held >= target:
                break
        if not held_groups and candidates:
            smallest = min(candidates, key=lambda g: len(groups[g]))
            held_groups = [smallest]
            legacy_val = list(groups[smallest])
            group_sizes[smallest] = len(groups[smallest])

    # ---- curriculum-cell pool: per-cell deterministic eval holdout ------------ #
    curriculum_val_by_cell: dict[str, list[str]] = {}
    curriculum_conv_counts: dict[str, int] = {}
    curriculum_train_conv_counts: dict[str, int] = {}
    blind_probe_pins: dict[str, list[str]] = {}
    quarantined_by_cell: dict[str, list[str]] = {}
    curr_val: list[str] = []
    quarantined: list[str] = []
    for cell in sorted(cellset):
        sub = curr_df[camp_col.reindex(curr_df.index).eq(cell)] if len(curr_df) else curr_df
        if not len(sub):
            continue
        sub_rid = sub["record_id"].astype(str)
        if cell not in reached_set:
            # QUARANTINE: the driver has not started this cell, so every one of its
            # (pre-merged) rows — converged or not — is dropped from both folds and
            # recorded only here.  No stable-hash holdout is computed, so a later
            # reached-time split for this cell is byte-identical to a never-
            # quarantined one.  Nothing about the cell touches the champion, so its
            # future blind-probe transfer measurement stays genuinely blind.
            q_ids = sorted(dict.fromkeys(sub_rid.tolist()))
            quarantined_by_cell[cell] = q_ids
            quarantined.extend(q_ids)
            continue
        sub_conv = sub["converged"].astype(bool) if "converged" in sub.columns \
            else pd.Series(False, index=sub.index)
        conv_ids = sorted(sub_rid[sub_conv].tolist())
        conv_set = set(conv_ids)
        n_conv = len(conv_ids)
        curriculum_conv_counts[cell] = n_conv

        pins_present = [rid for rid in bp.get(cell, []) if rid in conv_set]
        # dedup preserving determinism
        pins_present = sorted(dict.fromkeys(pins_present))
        blind_probe_pins[cell] = pins_present

        if n_conv >= 2:
            val_count = int(round(cell_val_frac * n_conv))
            val_count = max(1, val_count)
            val_count = min(val_count, n_conv - 1)  # keep >=1 converged in train
        else:
            # a lone converged in-band row is irreplaceable -> keep it in train
            val_count = 0

        val_ids_cell = set(pins_present)
        if len(val_ids_cell) < val_count:
            pool = [rid for rid in conv_ids if rid not in val_ids_cell]
            pool.sort(key=lambda r: (_hash01(r), r))
            for rid in pool:
                if len(val_ids_cell) >= val_count:
                    break
                val_ids_cell.add(rid)
        cell_val_sorted = sorted(val_ids_cell)
        curriculum_val_by_cell[cell] = cell_val_sorted
        curr_val.extend(cell_val_sorted)
        n_train_conv = n_conv - len(val_ids_cell)
        curriculum_train_conv_counts[cell] = n_train_conv

        # -- invariants (plan requirement: assert in code AND test) ------------- #
        assert n_conv == 0 or n_conv < 2 or cell_val_sorted, \
            f"curriculum cell {cell!r} has converged rows but an empty eval holdout"
        assert n_train_conv >= math.floor(0.70 * n_conv), \
            f"curriculum cell {cell!r} keeps < 70% of converged rows in train " \
            f"({n_train_conv}/{n_conv})"

    val_set = set(legacy_val) | set(curr_val)
    quarantine_set = set(quarantined)
    # Quarantined rows live in NEITHER fold (only in groups.quarantined_by_cell):
    # LPDataset resolves each fold purely from record_ids(fold), so a row absent
    # from both is never materialized -> inert for gradient steps AND for the
    # per-epoch composite val metric that drives champion best-epoch selection.
    train_ids = [i for i in ids_all if i not in val_set and i not in quarantine_set]
    val_ids = [i for i in ids_all if i in val_set]

    # -- quarantine invariants (methodology guard: asserted AND documented) ----- #
    train_id_set = set(train_ids)
    assert quarantine_set.isdisjoint(val_set), \
        "quarantined rows must not enter the metric val fold"
    assert quarantine_set.isdisjoint(train_id_set), \
        "quarantined rows must never train"
    _all_cell_val_ids = {rid for ids in curriculum_val_by_cell.values() for rid in ids}
    assert quarantine_set.isdisjoint(_all_cell_val_ids), \
        "quarantined rows must never be scored by the honest no-regression gate"

    return SplitManifest(
        name=name, kind="curriculum_group", seed=seed,
        train_ids=train_ids, val_ids=val_ids,
        predicate={
            "kind": "curriculum",
            "val_frac": val_frac,
            "cell_val_frac": cell_val_frac,
            "max_group_frac": max_group_frac,
            "protect_feed": protect_feed,
        },
        groups={
            "n_groups": n_legacy_groups,
            "held_out_groups": sorted(held_groups),
            "held_out_group_sizes": {gk: group_sizes[gk] for gk in sorted(held_groups)},
            "legacy_n_val": len(legacy_val),
            "protect_feed": protect_feed,
            "cells": sorted(c for c in cellset if curriculum_conv_counts.get(c)),
            "curriculum_cell_cap": (float(cell_cap) if cell_cap is not None else None),
            "curriculum_val_by_cell": curriculum_val_by_cell,
            "curriculum_conv_counts": curriculum_conv_counts,
            "curriculum_train_conv_counts": curriculum_train_conv_counts,
            "blind_probe_pins": blind_probe_pins,
            "quarantined_by_cell": quarantined_by_cell,
        },
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def make_splits(
    store_df: pd.DataFrame,
    seed: int,
    out_dir: str | Path = "data/splits",
    *,
    holdout_pairs: Sequence[str] = ("C3_C6", "A01_B05"),
    s4_band: tuple[float, float] = (5.43, 5.50),
    persist: bool = True,
) -> dict[str, SplitManifest]:
    """Build the S0..S4 manifest family and (optionally) write JSON to ``out_dir``.

    Returns ``{name: SplitManifest}`` for ``S0, S1, S2, S3a, S3b, S4``.
    """
    manifests: dict[str, SplitManifest] = {
        "S0": make_s0(store_df, seed),
        "S1": make_s1(store_df, seed),
        "S2": make_leave_pair(store_df, seed, "S2", holdout_pairs),
        "S3a": make_feed_filter(store_df, seed, "S3a", [117], "empty"),
        "S3b": make_feed_filter(
            store_df, seed, "S3b", [105, 113], "awaiting_production"
        ),
        "S4": make_e_core_band(store_df, seed, "S4", s4_band[0], s4_band[1]),
    }
    if persist:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, manifest in manifests.items():
            manifest.to_json(out / f"{name}.json")
    return manifests


__all__ = [
    "SplitManifest",
    "make_splits",
    "make_s0",
    "make_s1",
    "make_leave_pair",
    "make_feed_filter",
    "make_e_core_band",
    "make_curriculum_split",
]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    import pandas as pd

    df = pd.read_parquet("data/store/records.parquet")
    manifests = make_splits(df, seed=0, persist=False)
    for name, m in manifests.items():
        total = m.n_train + m.n_val
        pct = 100.0 * m.n_val / total if total else 0.0
        print(
            f"{name:4s} kind={m.kind:11s} status={m.status:18s} "
            f"train={m.n_train:6d} val={m.n_val:6d} ({pct:4.1f}%)"
        )
