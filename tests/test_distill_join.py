"""Distillation soft-target join: the 0-match defect and its guards.

The v5_distill arm produced a decoy: it trained "successfully" but its weights
were byte-identical to v5_full because the soft-target cache matched ZERO train
rows.  Root cause: the teacher map is keyed by curriculum cell-id
(``5-5.25_f101`` == a row's ``campaign``) while the cache keyed rows by
``cyclen_cell_key`` (``feed=121|ebin=5.4``) — disjoint namespaces, so every row
matched no teacher and the cache was silently all-zero.

These tests pin: (1) campaign-keying matches, (2) the mismatched-key build
hard-errors instead of emitting an all-zero cache, (3) the attach join hard-errors
when too few of the cache's built rows survive the record_id join against the
actual (possibly newer) training store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.model.cell_calibrate import cyclen_cell_key                 # noqa: E402
from lpopt.model.dataset_torch import targets_for                     # noqa: E402
from lpopt.model.distill import (                                     # noqa: E402
    build_soft_targets, load_soft_targets,
)
from lpopt.model.splits import SplitManifest                          # noqa: E402
from lpopt.model.train import attach_distill_targets                 # noqa: E402
from lpopt.config import load_config                                 # noqa: E402

STORE = "data/store"
TN = targets_for(True)


@pytest.fixture(scope="module")
def train_df():
    from lpopt.data.store import StoreReader
    recs = StoreReader(STORE).records
    man = SplitManifest.from_json("data/splits/S1.json")
    ids = set(man.record_ids("train"))
    df = recs[recs["record_id"].astype(str).isin(ids)].reset_index(drop=True)
    return df, man


@pytest.fixture(scope="module")
def champion():
    return load_config("lpopt.inp").model.model_dir


class _DS:
    """A minimal PrecomputedDataset stand-in (attach only touches these)."""

    def __init__(self, record_ids):
        self.record_ids = [str(r) for r in record_ids]
        self._t: dict = {}


# --------------------------------------------------------------------------- #
# root cause: campaign keying matches, cyclen_cell_key keying does not
# --------------------------------------------------------------------------- #
def test_campaign_keying_matches_the_teacher_cells(tmp_path, train_df, champion):
    df, man = train_df
    cells = sorted(man.groups["curriculum_val_by_cell"])[:3]
    teachers = {c: champion for c in cells}
    keys = df["campaign"].astype(str).tolist()          # THE FIX
    art = build_soft_targets(df, teachers, cell_keys=keys, target_names=TN,
                             store_dir=STORE, library_id="ga80", device="cpu",
                             out_path=tmp_path / "c.npz")
    # every intended row (campaign in a teacher cell) got a soft target
    n_intended = int(df["campaign"].astype(str).isin(cells).sum())
    assert n_intended > 0
    assert art["n_intended"] == n_intended
    assert art["n_soft"] == n_intended > 0


def test_cyclen_cell_key_keying_is_the_bug_and_now_hard_errors(train_df, champion):
    """The exact decoy: teacher cells (campaign) vs row keys (cyclen_cell_key)."""
    df, man = train_df
    cells = sorted(man.groups["curriculum_val_by_cell"])[:3]
    teachers = {c: champion for c in cells}
    bad_keys = [cyclen_cell_key(f, e) for f, e in zip(df["feed"], df["e_core"])]
    with pytest.raises(ValueError, match="no soft targets|do not match"):
        build_soft_targets(df, teachers, cell_keys=bad_keys, target_names=TN,
                           store_dir=STORE, library_id="ga80", device="cpu")


def test_build_cache_used_by_runner_keys_by_campaign():
    """Guard against a regression to cyclen_cell_key in the runner's builder."""
    import ast
    import inspect
    from lpopt.model import v5_experiment as V
    src = inspect.getsource(V._build_distill_cache)
    assert 'df["campaign"]' in src
    # strip comments + docstring, then assert the CODE never calls cyclen_cell_key
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "cyclen_cell_key" not in calls


# --------------------------------------------------------------------------- #
# fail-loud attach guard (version-robust record_id join)
# --------------------------------------------------------------------------- #
def _cache(tmp_path, record_ids, n_soft_rows):
    """A hand-built cache: the first ``n_soft_rows`` carry a real soft target."""
    n = len(record_ids)
    soft = np.zeros((n, len(TN)), dtype=np.float32)
    mask = np.zeros((n, len(TN)), dtype=np.float32)
    cy = TN.index("cyclen")
    soft[:n_soft_rows, cy] = 600.0
    mask[:n_soft_rows, cy] = 1.0
    p = tmp_path / "cache.npz"
    np.savez_compressed(
        p, record_ids=np.asarray([str(r) for r in record_ids], dtype=str),
        soft=soft, mask=mask, target_names=np.asarray(list(TN), dtype=str),
        schema="distill_soft_targets_v1",
        n_intended=np.asarray(n_soft_rows), n_soft=np.asarray(n_soft_rows))
    return p


def test_load_exposes_match_counts(tmp_path):
    p = _cache(tmp_path, [f"r{i}" for i in range(10)], 6)
    cache = load_soft_targets(p)
    assert cache["n_soft"] == 6
    assert cache["n_intended"] == 6


def test_load_recomputes_n_soft_for_a_legacy_cache(tmp_path):
    """An old cache without the scalar fields still yields a usable n_soft."""
    n = 8
    soft = np.zeros((n, len(TN)), dtype=np.float32)
    mask = np.zeros((n, len(TN)), dtype=np.float32)
    mask[:5, TN.index("cyclen")] = 1.0
    p = tmp_path / "legacy.npz"
    np.savez_compressed(p, record_ids=np.asarray([f"r{i}" for i in range(n)], dtype=str),
                        soft=soft, mask=mask,
                        target_names=np.asarray(list(TN), dtype=str),
                        schema="distill_soft_targets_v1")
    assert load_soft_targets(p)["n_soft"] == 5


def test_full_match_attaches(tmp_path):
    ids = [f"r{i}" for i in range(10)]
    p = _cache(tmp_path, ids, 6)
    ds = _DS(ids)
    n = attach_distill_targets(ds, p, TN, min_match_frac=0.5)
    assert n == 6
    assert "distill_soft" in ds._t and "distill_mask" in ds._t


def test_zero_match_hard_errors(tmp_path):
    """The decoy signature: the training store shares no record_id with the cache."""
    p = _cache(tmp_path, [f"cache{i}" for i in range(10)], 6)
    ds = _DS([f"train{i}" for i in range(10)])          # disjoint ids
    with pytest.raises(ValueError, match="matched 0/6|do not align"):
        attach_distill_targets(ds, p, TN, min_match_frac=0.5)


def test_below_threshold_match_hard_errors(tmp_path):
    ids = [f"r{i}" for i in range(10)]
    p = _cache(tmp_path, ids, 6)
    # only 2 of the 6 built soft-target rows exist in this train fold -> 33% < 50%
    ds = _DS(["r0", "r1", "x2", "x3", "x4"])
    with pytest.raises(ValueError, match="matched 2/6|do not align"):
        attach_distill_targets(ds, p, TN, min_match_frac=0.5)


def test_empty_cache_hard_errors(tmp_path):
    p = _cache(tmp_path, [f"r{i}" for i in range(10)], 0)
    ds = _DS([f"r{i}" for i in range(10)])
    with pytest.raises(ValueError, match="no soft-target rows|built empty"):
        attach_distill_targets(ds, p, TN, min_match_frac=0.5)


def test_guard_can_be_disabled(tmp_path):
    """min_match_frac<=0 restores the permissive graceful-degradation behaviour."""
    p = _cache(tmp_path, [f"cache{i}" for i in range(10)], 6)
    ds = _DS([f"train{i}" for i in range(10)])
    assert attach_distill_targets(ds, p, TN, min_match_frac=0.0) == 0


def test_partial_match_above_threshold_passes(tmp_path):
    ids = [f"r{i}" for i in range(10)]
    p = _cache(tmp_path, ids, 6)
    # 5 of 6 built rows present -> 83% >= 50%
    ds = _DS(["r0", "r1", "r2", "r3", "r4", "z"])
    assert attach_distill_targets(ds, p, TN, min_match_frac=0.5) == 5


def test_train_config_default_threshold_is_half():
    from lpopt.model.train import TrainConfig
    assert TrainConfig().distill_min_match_frac == 0.5


def test_trainer_cli_exposes_the_threshold():
    import argparse
    import contextlib
    import io
    from lpopt.model.train import main as train_main
    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
        train_main(["--help"])
    assert "--distill-min-match-frac" in buf.getvalue()
