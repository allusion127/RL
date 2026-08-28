"""Corpus, splits and features for the v1 move-proposal policy.

Reads ``data/policy/steps.parquet`` (built by ``mine_policy_corpus.py``) READ-ONLY.

Three things live here and nothing else:

* **the universe rule** — which rows are moves the policy could actually take;
* **the split** — grouped by lineage connected component, with three whole
  cell families held out for the transfer readouts;
* **the features** — the leakage-safe encoding of (parent board, move, cell).

Leakage discipline
------------------
The model may see anything computable from the PARENT board, the CANDIDATE
EDIT and the CELL, because all three are known before the child is evaluated.
It may never see a child OUTCOME.  :data:`FORBIDDEN_COLUMNS` names every column
that would break that and :func:`scalar_features` asserts none of them reached
the feature frame.  Two further exclusions are deliberate and are NOT leakage
in the strict sense — they are noted where they are made:

* ``lineage_source`` / ``campaign`` / ``generator`` / ``sa_accepted`` /
  ``source_move`` / ``single_move_evidence`` — provenance.  A move the policy
  proposes has no provenance, so a model leaning on it would not transfer.  And
  ``lineage_source`` is collinear with the era holdout, which would make that
  readout meaningless.
* ``parent_f_r`` / ``parent_node_peak`` / ... — the PARENT's own FOMs.  These
  are legitimately available at proposal time, but they let a scorer win on
  "this parent is bad, anything helps" instead of on move quality, which is the
  confound the pooled deployment metric is already exposed to.  v1 excludes
  them; a v2 may add the SURROGATE-predicted parent FOM (available for an
  unevaluated parent too).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

STEPS_PARQUET = "data/policy/steps.parquet"
FUEL_TYPES_PARQUET = "data/store/fuel_types.parquet"

#: v6b is the schema the s1e champion surrogate uses; the policy reuses it so a
#: later round can compare or share encoders without a re-featurization.
COND_SCHEMA = "v6b"
#: Power-prior constants the v6b ``prior_power*`` channels are evaluated with.
#: Taken from the s1e fit so the two models see the SAME leading-order physics.
POWER_PRIOR = {"m2_cm2": 150.0, "extrap": 2.0}

#: Same-cell move classes, in a fixed order (the one-hot layout).  ``sa_unknown``
#: is a FEATURE value, not a missing target: it marks a MOCHA compound primitive
#: whose net diff the classifier could not name.
MOVE_CLASSES: tuple[str, ...] = (
    "rewire_swap", "fresh_relocate", "batch_flip", "sa_unknown",
    "multi", "rewire_multi", "batch_multi", "batch_swap",
)
RADIAL_DIRS: tuple[str, ...] = ("outward", "inward", "neutral")

# --------------------------------------------------------------------------- #
# held-out families (pre-registered 2026-08-15, before any weight was trained)
# --------------------------------------------------------------------------- #
#: The whole lpopt_genome era: every ga80 / paramA cell.  This is the LIVE
#: operating point (feeds 101-141 near F_r=1.55) and the corpus's 1,399 rows
#: there are exactly the rows the SA recovery did not produce, so holding them
#: out costs 7% of training data and buys the only deployment-relevant readout.
HELDOUT_ERA_LIBRARIES: tuple[str, ...] = ("ga80", "paramA")
#: One full mid-size cell of the DOMINANT library — the brief's cell-transfer
#: readout.  785 steps, 43.9% improving on F_r, node_peak labelled throughout.
HELDOUT_CELL: str = "B1_C6/f121/260624"
#: A whole unseen library inside the SA era (772 steps over 5 cells).
HELDOUT_LIBRARY: str = "5.8_5.1"

#: Child OUTCOME columns.  Any of these in the feature frame is a bug.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset({
    "child_f_r", "child_cyclen", "child_cbc_max", "child_f_q", "child_ao_abs",
    "child_node_peak", "child_map_cov", "child_converged",
    "d_f_r", "d_cyclen", "d_cbc_max", "d_f_q", "d_ao_abs", "d_node_peak",
    "d_map_cov",
    "improved_fr", "improved_flat", "improved_cbc", "improved_cyclen",
    "feasible_child", "both_converged", "in_cyclen_band_child",
    # parent FOMs: available at proposal time but excluded by decision (above)
    "parent_f_r", "parent_cyclen", "parent_cbc_max", "parent_f_q",
    "parent_ao_abs", "parent_node_peak", "parent_map_cov", "parent_converged",
    "feasible_parent",
    # provenance: not available for a move the policy invents
    "lineage_source", "campaign", "generator", "sa_accepted", "source_move",
    "single_move_evidence", "dataset_split",
})

#: Move descriptors, all functions of (parent pattern, child pattern).
_MOVE_DELTAS: tuple[str, ...] = (
    "d_fresh_share_inner", "d_fresh_share_middle", "d_fresh_share_periph",
    "d_fresh_r_center", "d_fresh_enr_r_center",
    "d_once_burnt_periph_share", "d_twice_burnt_periph_share",
)
#: Parent-board descriptors (the state the move acts on).
_PARENT_RINGS: tuple[str, ...] = (
    "parent_fresh_share_inner", "parent_fresh_share_middle",
    "parent_fresh_share_periph", "parent_fresh_r_center",
    "parent_fresh_enr_r_center", "parent_once_burnt_periph_share",
    "parent_twice_burnt_periph_share",
)

HEADS: tuple[str, ...] = ("fr", "flat")
_LABEL_COL = {"fr": "improved_fr", "flat": "improved_flat"}


# --------------------------------------------------------------------------- #
# universe
# --------------------------------------------------------------------------- #
def load_universe(path: str | Path = STEPS_PARQUET) -> pd.DataFrame:
    """The rows a v1 policy is allowed to train and be judged on.

    Two filters, both from ``policy_corpus_20260815.md``:

    * ``cross_cell == False`` — a feed morph or a pair transfer is not an action
      in a fixed move space, so it is not a move the policy can propose.
    * at least one of the two heads is labelled — a row with neither label
      carries no supervision for either head.
    """
    steps = pd.read_parquet(path)
    same_cell = steps[~steps["cross_cell"].astype(bool)].copy()
    has_label = (same_cell["improved_fr"].notna()
                 | same_cell["improved_flat"].notna())
    out = same_cell[has_label].reset_index(drop=True)
    out["move_class"] = out["move_class"].astype(str)
    return out


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def _components(frame: pd.DataFrame) -> np.ndarray:
    """Connected-component id per row over the (parent, child) lineage graph.

    A random ROW split would put a board's siblings on both sides: the parent
    tensor is then memorisable and the held-out number is inflated.  Splitting
    on the connected components of the lineage DAG is the strongest available
    guard — no board reachable from a training board can appear in test.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    for p, c in zip(frame["parent_record_id"], frame["child_record_id"],
                    strict=True):
        ra, rb = find(str(p)), find(str(c))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(str(p)) for p in frame["parent_record_id"]])


def _greedy_group_split(groups: np.ndarray, frac: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Boolean mask selecting whole groups until ``frac`` of rows is covered.

    Groups are visited in random order and taken while the running total stays
    under target; a group larger than the whole target is skipped rather than
    blowing the split (documented in the results report as a realized-fraction
    deviation).
    """
    sizes = Counter(groups.tolist())
    order = list(sizes)
    rng.shuffle(order)
    target = frac * len(groups)
    picked: set[Any] = set()
    total = 0
    for key in order:
        if total + sizes[key] <= target:
            picked.add(key)
            total += sizes[key]
    return np.isin(groups, list(picked)) if picked else np.zeros(len(groups), bool)


def build_splits(steps: pd.DataFrame, *, seed: int = 20260815,
                 test_frac: float = 0.10,
                 val_frac: float = 0.10) -> pd.Series:
    """Assign every row to one of six folds.  Deterministic in ``seed``.

    ``heldout_era`` / ``heldout_cell`` / ``heldout_lib`` are whole cell families
    removed before any random split touches the data; ``test`` / ``val`` /
    ``train`` are lineage-component-grouped draws from what is left, stratified
    by cell (the draw runs inside each cell independently).
    """
    fold = pd.Series("train", index=steps.index, dtype=object)

    era = steps["library_id"].isin(HELDOUT_ERA_LIBRARIES)
    fold[era] = "heldout_era"
    lib = (~era) & (steps["library_id"] == HELDOUT_LIBRARY)
    fold[lib] = "heldout_lib"
    cell = (~era) & (~lib) & (steps["cell"] == HELDOUT_CELL)
    fold[cell] = "heldout_cell"

    pool = fold == "train"
    comp = pd.Series(index=steps.index, dtype=object)
    comp[pool] = _components(steps[pool])

    rng = np.random.default_rng(seed)
    for cell_key, group in steps[pool].groupby("cell", sort=True):
        idx = group.index.to_numpy()
        groups = comp.loc[idx].to_numpy()
        take = _greedy_group_split(groups, test_frac, rng)
        fold.loc[idx[take]] = "test"
        rest = idx[~take]
        if len(rest):
            take_v = _greedy_group_split(comp.loc[rest].to_numpy(), val_frac, rng)
            fold.loc[rest[take_v]] = "val"
    return fold


# --------------------------------------------------------------------------- #
# scalar features
# --------------------------------------------------------------------------- #
def scalar_features(steps: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """``(X[N, D] float32, names)`` — the move/context vector fed to FiLM.

    Every column is a function of (parent pattern, child pattern, cell).  The
    13 board globals are added later by :func:`build_pattern_cache`, which is
    where the featurizer lives.
    """
    cols: dict[str, np.ndarray] = {}

    klass = steps["move_class"].to_numpy()
    for name in MOVE_CLASSES:
        cols[f"cls_{name}"] = (klass == name).astype(np.float32)

    cols["n_unit_edits"] = steps["n_unit_edits"].to_numpy(np.float32) / 10.0
    cols["n_slots_changed"] = steps["n_slots_changed"].to_numpy(np.float32) / 10.0
    cols["single_move"] = steps["single_move"].to_numpy(np.float32)

    # swap_span / swap_radius exist only for the two swap classes; the presence
    # flag keeps "absent" distinguishable from "zero".
    for name in ("swap_span", "swap_radius"):
        raw = steps[name].to_numpy(np.float64)
        cols[f"{name}_present"] = (~np.isnan(raw)).astype(np.float32)
        cols[name] = np.nan_to_num(raw, nan=0.0).astype(np.float32) / 10.0

    for name in (*_MOVE_DELTAS, *_PARENT_RINGS):
        cols[name] = np.nan_to_num(
            steps[name].to_numpy(np.float64), nan=0.0).astype(np.float32)

    for field, values in (("fresh_radial_dir", RADIAL_DIRS),
                          ("burnt_periph_dir", RADIAL_DIRS)):
        raw = steps[field].astype(str).to_numpy()
        for v in values:
            cols[f"{field}_{v}"] = (raw == v).astype(np.float32)

    cols["feed_centered"] = (
        (steps["feed"].to_numpy(np.float32) - 121.0) / 20.0)

    bad = FORBIDDEN_COLUMNS & set(cols)
    if bad:                                   # structural, not a style check
        raise AssertionError(f"outcome/provenance leaked into features: {sorted(bad)}")

    names = sorted(cols)
    return np.stack([cols[n] for n in names], axis=1).astype(np.float32), names


# --------------------------------------------------------------------------- #
# board feature cache
# --------------------------------------------------------------------------- #
@dataclass
class PatternCache:
    """Per-pattern featurizer output, plus the transpose partner of each.

    ``slots[i]`` is the ``(C, 69)`` slot matrix of pattern ``i`` and
    ``globals_[i]`` its 13 conditioning globals.  ``mirror[i]`` is the index of
    the diagonal-mirror image, which is what makes the augmentation free at
    train time: no encoder call, just a different row.
    """

    index: dict[str, int]
    slots: np.ndarray                  # [P, C, 69] float16
    globals_: np.ndarray               # [P, G] float32
    mirror: np.ndarray                 # [P] int32
    channels: list[str]
    globals_names: list[str]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, slots=self.slots, globals_=self.globals_, mirror=self.mirror,
            patterns=np.array(sorted(self.index, key=self.index.get), dtype=object),
            channels=np.array(self.channels, dtype=object),
            globals_names=np.array(self.globals_names, dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "PatternCache":
        z = np.load(path, allow_pickle=True)
        pats = [str(p) for p in z["patterns"]]
        return cls(index={p: i for i, p in enumerate(pats)},
                   slots=z["slots"], globals_=z["globals_"],
                   mirror=z["mirror"],
                   channels=[str(c) for c in z["channels"]],
                   globals_names=[str(g) for g in z["globals_names"]])


def _sym_class(library_id: str) -> str:
    from ..model.featurize import library_provenance
    return library_provenance(str(library_id))[1]


def build_pattern_cache(steps: pd.DataFrame, *,
                        fuel_types: str | Path = FUEL_TYPES_PARQUET,
                        n_workers: int = 0,
                        progress: bool = True) -> PatternCache:
    """Featurize every distinct pattern in ``steps`` AND its diagonal mirror.

    ~21k distinct boards x 2 at ~19 ms each — minutes, once, then cached to
    disk.  The cache stores the ``(C, 69)`` slot matrix rather than the
    ``(C, 19, 19)`` grid: the grid is a static scatter of those 69 values plus a
    constant reflector mask, so keeping slots costs 5.2x less memory (338 MB vs
    1.77 GB at float16) and the scatter is a fancy-index in the collate.
    """
    from ..data.fuel_types import FuelLibrary
    from ..data.geometry import transpose
    from ..data.schema import pack_pattern, unpack_pattern
    from ..model.featurize import FeatureEncoder, RecordInputs
    from ..model.power_prior import PowerPrior

    # context per pattern: a pattern is (almost always) unique to one cell; if
    # it ever recurred across cells the first sighting wins and the second is
    # recorded, because the globals would differ.
    ctx: dict[str, tuple[int, str, str, str]] = {}
    for side in ("parent", "child"):
        for pat, feed, pair, lib, ds in zip(
                steps[f"{side}_pattern"], steps["feed"], steps["case_pair"],
                steps["library_id"], steps["dataset"], strict=True):
            ctx.setdefault(str(pat), (int(feed), str(pair), str(lib), str(ds)))

    # add the mirror of every pattern under the same context
    mirror_of: dict[str, str] = {}
    for pat in list(ctx):
        t = pack_pattern(transpose(unpack_pattern(pat)))
        mirror_of[pat] = t
        ctx.setdefault(t, ctx[pat])

    patterns = sorted(ctx)
    index = {p: i for i, p in enumerate(patterns)}
    mirror = np.array([index[mirror_of.get(p, p)] for p in patterns], np.int32)

    fuel = FuelLibrary.from_parquet(fuel_types)
    enc = FeatureEncoder(cond_schema=COND_SCHEMA,
                         power_prior=PowerPrior(**POWER_PRIOR))

    slots = np.zeros((len(patterns), enc.n_channels, 69), np.float16)
    gvecs = np.zeros((len(patterns), len(enc.globals_names)), np.float32)
    for i, pat in enumerate(patterns):
        feed, pair, lib, ds = ctx[pat]
        inp = RecordInputs(pattern=pat, feed=feed, case_pair=pair,
                           library_id=lib, sym_class=_sym_class(lib), dataset=ds)
        slot_vals = enc.encode_slot_matrix(inp, fuel)
        slots[i] = slot_vals.astype(np.float16)
        gvecs[i] = enc._encode_globals(inp, fuel, slot_vals)
        if progress and i % 2000 == 0:
            print(f"  [cache] {i}/{len(patterns)}", flush=True)
    return PatternCache(index=index, slots=slots, globals_=gvecs, mirror=mirror,
                        channels=list(enc.channels),
                        globals_names=list(enc.globals_names))


# --------------------------------------------------------------------------- #
# torch dataset
# --------------------------------------------------------------------------- #
def _grid_scatter() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices scattering the 69 slot values onto the 19x19 mirror grid."""
    from ..model.featurize import _POS_ROWS, _POS_COLS, _POS_SLOT
    return np.asarray(_POS_ROWS), np.asarray(_POS_COLS), np.asarray(_POS_SLOT)


class PolicySteps:
    """Map-style dataset over move rows; import-light so tests stay torch-free.

    ``__getitem__`` returns numpy; the caller wraps it in a torch ``Dataset``.
    Transpose augmentation is applied HERE and is a pure index swap: the parent
    and the child both move to their mirror rows and every scalar is copied
    verbatim, which is exactly the recipe ``mine_policy_corpus.verify_transpose``
    proved label-preserving (0 violations in 1,500).
    """

    def __init__(self, steps: pd.DataFrame, cache: PatternCache,
                 scalars: np.ndarray, *, delta_channels: Sequence[int],
                 augment: bool = False, seed: int = 0):
        self.parent = np.array([cache.index[p] for p in steps["parent_pattern"]],
                               np.int32)
        self.child = np.array([cache.index[p] for p in steps["child_pattern"]],
                              np.int32)
        self.cache = cache
        self.scalars = scalars.astype(np.float32)
        self.delta = np.asarray(delta_channels, np.int32)
        self.augment = bool(augment)
        self._rng = np.random.default_rng(seed)

        self.labels = np.zeros((len(steps), len(HEADS)), np.float32)
        self.mask = np.zeros((len(steps), len(HEADS)), np.float32)
        for h, head in enumerate(HEADS):
            raw = steps[_LABEL_COL[head]]
            self.mask[:, h] = raw.notna().to_numpy(np.float32)
            self.labels[:, h] = raw.fillna(False).astype(bool).to_numpy(np.float32)

        rows, cols, slot = _grid_scatter()
        self._rows, self._cols, self._slot = rows, cols, slot
        self._grid = (len(cache.channels) + len(self.delta) + 1, 19, 19)

    def __len__(self) -> int:
        return len(self.parent)

    @property
    def n_channels(self) -> int:
        return self._grid[0]

    @property
    def n_cond(self) -> int:
        return self.scalars.shape[1] + self.cache.globals_.shape[1]

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        p, c = int(self.parent[i]), int(self.child[i])
        if self.augment and self._rng.random() < 0.5:
            p, c = int(self.cache.mirror[p]), int(self.cache.mirror[c])

        ps = self.cache.slots[p].astype(np.float32)          # [C, 69]
        cs = self.cache.slots[c].astype(np.float32)
        d = cs[self.delta] - ps[self.delta]
        changed = (np.abs(cs - ps).max(axis=0) > 1e-6).astype(np.float32)[None]

        vals = np.concatenate([ps, d, changed], axis=0)      # [C', 69]
        grid = np.zeros(self._grid, np.float32)
        grid[:, self._rows, self._cols] = vals[:, self._slot]

        cond = np.concatenate([self.cache.globals_[p], self.scalars[i]])
        return {"cells": grid, "cond": cond.astype(np.float32),
                "y": self.labels[i], "m": self.mask[i]}


def pick_delta_channels(steps: pd.DataFrame, cache: PatternCache, *,
                        sample: int = 4000, seed: int = 0) -> list[int]:
    """Channels that ever differ between a parent and its child.

    Roughly half of the 58 v6b channels are pure geometry (radius, masks, orbit
    multiplicity) and are identical for every board, so their delta plane would
    be a hard zero.  Selecting on the TRAIN rows only keeps the choice honest.
    """
    rng = np.random.default_rng(seed)
    take = rng.choice(len(steps), size=min(sample, len(steps)), replace=False)
    keep = np.zeros(cache.slots.shape[1], bool)
    pp = steps["parent_pattern"].to_numpy()
    cc = steps["child_pattern"].to_numpy()
    for i in take:
        a = cache.slots[cache.index[pp[i]]].astype(np.float32)
        b = cache.slots[cache.index[cc[i]]].astype(np.float32)
        keep |= np.abs(b - a).max(axis=1) > 1e-6
    return [int(i) for i in np.flatnonzero(keep)]


def corpus_fingerprint(path: str | Path = STEPS_PARQUET) -> str:
    """SHA-256 of the steps parquet — stamped into every checkpoint's meta."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_summary(steps: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """Rows, cells and per-head base rate for each fold — the prereg table."""
    rows = []
    for name, group in steps.groupby(fold, sort=False):
        entry = {"fold": name, "n_steps": len(group),
                 "n_cells": group["cell"].nunique()}
        for head in HEADS:
            col = group[_LABEL_COL[head]]
            entry[f"n_{head}"] = int(col.notna().sum())
            entry[f"base_{head}"] = (float(col.mean())
                                     if col.notna().any() else float("nan"))
        rows.append(entry)
    order = ["train", "val", "test", "heldout_cell", "heldout_lib", "heldout_era"]
    out = pd.DataFrame(rows).set_index("fold")
    return out.reindex([o for o in order if o in out.index]).reset_index()
