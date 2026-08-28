"""F_r boundary micro-verification harness (lpopt.search.boundary_probe).

Covers cell resolution (learned vs quarantined), the legacy low-F_r seed pull, the
3-generator candidate pool (valid + de-duplicated + stratum-feed-pinned), the
mean-ascending ranking + reported UCB, the report schema, and the --verify
provenance wiring (rows tagged campaign=<cell> / generator="g3_elite_boundary")
exercised through an injected StubEvaluator-style driver (no MASTER).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.search.boundary_probe import (
    BOUNDARY_GENERATOR, DEFAULT_BOUNDARY_CELLS, Candidate, CellSpec, build_report,
    generate_pool, legacy_low_fr_seeds, rank_pool, resolve_cell, verify_candidates,
)
from lpopt.data.schema import compute_record_id, unpack_pattern

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"
STATE = REPO_ROOT / "data" / "curriculum" / "state.json"
SPLITS = REPO_ROOT / "data" / "splits" / "S1.json"

pytestmark = pytest.mark.skipif(
    not (STORE / "records.parquet").is_file(), reason="store not present")


def _records() -> pd.DataFrame:
    from lpopt.data.store import StoreReader
    return StoreReader(STORE).records


def _state_splits() -> tuple[dict, dict]:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    splits = json.loads(SPLITS.read_text(encoding="utf-8")) if SPLITS.is_file() else {}
    return state, splits


# --------------------------------------------------------------------------- #
# cell resolution
# --------------------------------------------------------------------------- #
def test_default_roster_learned_flags() -> None:
    # `learned` must mirror the LIVE curriculum state (phase == "done"), whatever
    # the curriculum's current progress is -- a snapshot assertion here rots as
    # cells complete (it did once, at 36/36).
    assert len(DEFAULT_BOUNDARY_CELLS) == 8
    records = _records()
    state, splits = _state_splits()
    quar = set((splits.get("groups", {}) or {}).get("quarantined_by_cell", {}))
    for c in DEFAULT_BOUNDARY_CELLS:
        expected = c not in quar
        got = resolve_cell(c, records, state=state, splits=splits).learned
        assert got is expected, f"{c}: learned={got}, quarantine says {expected}"


def test_resolve_cell_parses_band_feed_pairs() -> None:
    records = _records()
    state, splits = _state_splits()
    cell = resolve_cell("5-5.25_f117", records, state=state, splits=splits)
    assert cell.band == (5.0, 5.25)
    assert cell.feed == 117
    assert cell.n_fresh == (117 - 1) // 4        # 29
    assert cell.library == "ga80"
    assert cell.pairs and all("_" in p for p in cell.pairs)


def test_resolve_unlearned_cell_derives_pairs_from_store() -> None:
    # A quarantined cell has no usable state.json entry; pairs must come from the
    # store.  The live curriculum has since learned every cell, so SIMULATE the
    # quarantine by dropping the cell's state entry.
    records = _records()
    state, splits = _state_splits()
    state = dict(state)
    state["cells"] = {k: v for k, v in state.get("cells", {}).items()
                      if k != "6-6.25_f101"}
    splits = dict(splits)
    groups = dict(splits.get("groups", {}) or {})
    groups["quarantined_by_cell"] = {"6-6.25_f101": []}
    splits["groups"] = groups
    cell = resolve_cell("6-6.25_f101", records, state=state, splits=splits)
    assert cell.learned is False
    assert cell.feed == 101
    assert cell.library == "paramA"
    assert len(cell.pairs) >= 1


def test_resolve_cell_bad_id() -> None:
    with pytest.raises(ValueError):
        resolve_cell("not-a-cell", _records())


# --------------------------------------------------------------------------- #
# legacy seeds
# --------------------------------------------------------------------------- #
def test_legacy_low_fr_seeds_ascending_and_bounded() -> None:
    seeds = legacy_low_fr_seeds(_records(), pool=40)
    assert seeds, "no legacy low-F_r seeds found"
    frs = [fr for _rid, _pat, fr in seeds]
    assert frs == sorted(frs)                    # F_r-ascending
    assert max(frs) <= 1.75                       # ceiling honoured
    assert len(seeds) <= 40


# --------------------------------------------------------------------------- #
# pool generation
# --------------------------------------------------------------------------- #
def test_generate_pool_valid_deduped_feed_pinned() -> None:
    records = _records()
    state, splits = _state_splits()
    cell = resolve_cell("5-5.25_f117", records, state=state, splits=splits)
    rng = random.Random(0)
    pool = generate_pool(cell, records, rng, pool_size=80)
    assert 40 <= len(pool) <= 80

    store_ids = set(records["record_id"].astype(str))
    rids = [c.record_id for c in pool]
    assert len(rids) == len(set(rids))            # unique within pool
    assert not (set(rids) & store_ids)            # deduped vs the store
    gens = {c.generator for c in pool}
    assert "legacy_transfer" in gens              # all three generators present
    assert "heuristic" in gens
    assert "elite_perturb" in gens

    for c in pool:
        assert c.pair in cell.pairs               # pair is the cell's decision var
        pat = c.pattern
        assert pat.feed == cell.feed              # every candidate carries N=cell feed
        # record_id preimage is reproducible from the (canonical pattern, lib, pair)
        assert c.record_id == compute_record_id(
            pat.canonical(), cell.library, c.pair, "ga80_produce")
        # elite_perturb rows carry a legacy parent; heuristic rows do not
        if c.generator == "heuristic":
            assert c.parent_id is None
        if c.generator == "elite_perturb":
            assert c.parent_id is not None


# --------------------------------------------------------------------------- #
# ranking (fake backend — no torch / champion load)
# --------------------------------------------------------------------------- #
class _FakeSurrogate:
    def __init__(self, mean, sig):
        self.mean = mean
        self.epistemic_std = sig
        self.calibrated_std = sig


class _FakeBackend:
    """Returns F_r = descending-within-batch so ranking must reorder ascending."""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def fitted_fr_cells(self):
        return {"feed=117|ebin=5.0"}

    def predict(self, patterns, cases, cell):
        n = len(patterns)
        mean = np.zeros((n, 7))
        sig = np.zeros((n, 7))
        mean[:, 0] = np.linspace(2.5, 1.5, n)     # descending F_r within the batch
        sig[:, 0] = self.sigma
        return _FakeSurrogate(mean, sig)


def _fake_candidates(n=10):
    records = _records()
    cell = resolve_cell("5-5.25_f117", records, state=_state_splits()[0],
                        splits=_state_splits()[1])
    pool = generate_pool(cell, records, random.Random(1), pool_size=max(20, n))
    return cell, pool[:n]


def test_rank_pool_sorts_by_mean_and_fills_ucb() -> None:
    cell, cands = _fake_candidates(10)
    ranked = rank_pool(_FakeBackend(sigma=0.2), cell, cands, risk_z=0.25, batch_size=1000)
    means = [c.f_r_mean for c in ranked]
    assert means == sorted(means)                 # ascending by predicted mean
    for c in ranked:
        assert c.f_r_ucb == pytest.approx(c.f_r_mean + 0.25 * c.f_r_sigma)
        assert c.f_r_sigma == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_build_report_schema_and_distributions() -> None:
    cell, cands = _fake_candidates(12)
    ranked = rank_pool(_FakeBackend(), cell, cands, batch_size=1000)
    top = ranked[:8]
    rep = build_report(cell, "some/champ", ranked, top, risk_z=0.25, verified=None)
    assert rep["schema"] == "boundary_probe_v1"
    assert rep["cell"] == "5-5.25_f117"
    assert rep["generator"] == BOUNDARY_GENERATOR
    assert rep["n_candidates"] == len(ranked)
    assert rep["top_k"] == 8
    assert rep["top_k_f_r_mean_distribution"]["n"] == 8
    assert len(rep["top_candidates"]) == 8
    row = rep["top_candidates"][0]
    for k in ("record_id", "pair", "generator", "f_r_mean",
              "f_r_mean_plus_zsigma", "pattern"):
        assert k in row


# --------------------------------------------------------------------------- #
# --verify provenance wiring (injected StubEvaluator-style driver — no MASTER)
# --------------------------------------------------------------------------- #
class _FakeResolver:
    def resolve(self, case_key):
        from lpopt.search.assets import ResolvedAssets
        return ResolvedAssets(case_key, None, None, -1, "stub:none")


class _FakeVerifier:
    n_workers = 8

    def evaluate_wave(self, entries):
        from lpopt.search.verify import WaveOutcome
        from lpopt.vendor.masterrl.domain import FOM
        out = []
        for e in entries:
            fom = FOM(f_r=1.552, cbc_max=1400.0, f_q=2.1, cyclen=615.0,
                      ao_min=-0.05, ao_max=0.05, max_burnup=53.0, max_pin_burnup=70.0,
                      converged=True)
            out.append(WaveOutcome(
                status="converged", fom=fom, n_cycles=6, tolerance_margin=0.4,
                wall_s=1.0, restart_provenance="stub:none", failure="",
                converged_at_cap=False, case_key=e.case_key, pattern=e.pattern,
                meta=dict(e.meta)))
        return out


class _FakeDriver:
    def __init__(self, tmp: Path):
        from lpopt.data.store import StoreWriter
        from lpopt.search.produce import Ledger as PLedger
        self.resolver = _FakeResolver()
        self.verifier = _FakeVerifier()
        self.store = StoreWriter(tmp / "store")
        self.store_dir = tmp / "store"
        self.ledger = PLedger(tmp / "ledger.jsonl")

    def _enrichment(self, pair, split_a, library):
        return 5.1, 0.5


def test_verify_wiring_writes_boundary_provenance(tmp_path: Path) -> None:
    from lpopt.data.store import StoreReader
    records = _records()
    state, splits = _state_splits()
    cell = resolve_cell("5-5.25_f117", records, state=state, splits=splits)
    pool = generate_pool(cell, records, random.Random(2), pool_size=40)
    ranked = rank_pool(_FakeBackend(), cell, pool, batch_size=1000)
    top = ranked[:2]

    driver = _FakeDriver(tmp_path)
    result = verify_candidates(None, cell, top, base=REPO_ROOT, driver=driver)
    assert result["chains"] == 2
    assert result["converged"] == 2
    assert all(lbl["actual_f_r"] == pytest.approx(1.552) for lbl in result["labels"])

    df = StoreReader(tmp_path / "store").records
    assert len(df) == 2
    assert (df["campaign"].astype(str) == "5-5.25_f117").all()
    assert (df["generator"].astype(str) == BOUNDARY_GENERATOR).all()
    assert (df["dataset"].astype(str) == "P").all()
    assert (df["feed"].astype(int) == 117).all()
    # the harvested rows carry the produced record_id preimage of their pattern
    for _, row in df.iterrows():
        pat = unpack_pattern(str(row["pattern"]))
        assert str(row["record_id"]) == compute_record_id(
            pat.canonical(), cell.library, str(row["case_pair"]), "ga80_produce")


def test_verify_requires_master_without_driver() -> None:
    from lpopt.config import LpoptConfig, MasterConfig
    records = _records()
    state, splits = _state_splits()
    cell = resolve_cell("5-5.25_f117", records, state=state, splits=splits)
    cand = Candidate(pattern=unpack_pattern(str(records.iloc[0]["pattern"])),
                     pair=cell.pairs[0], generator="legacy_transfer",
                     parent_id=None, record_id="x", split_a=0.5)
    cfg = LpoptConfig.__new__(LpoptConfig)        # minimal shell w/ empty master exe
    cfg.master = MasterConfig()
    with pytest.raises(RuntimeError, match="executable"):
        verify_candidates(cfg, cell, [cand], base=REPO_ROOT)
