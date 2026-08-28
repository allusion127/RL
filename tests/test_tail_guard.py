"""Legacy-corpus high-cyclen tail no-regression guard (forensic 20260719).

The honest per-cell curriculum gate scores only ga80 curriculum cells and the
global val zMAE-cyclen is tail-insensitive, so a collapse concentrated in the
700-720 EFPD Dataset-A tail (entirely the 5.8_5.1 library) escaped every gate.
These cover the guard's scorer and its deterministic sampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.curriculum import (
    sample_legacy_tail_rows,
    score_legacy_tail_no_regression,
)

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data" / "store"
BANDS = [[660.0, 680.0], [680.0, 700.0], [700.0, 720.0]]


# --------------------------------------------------------------------------- #
# synthetic corpus + stub models
# --------------------------------------------------------------------------- #
def _corpus(n_per_band: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for lo, hi in BANDS:
        for _ in range(n_per_band):
            cy = float(rng.uniform(lo, hi))
            rows.append({"record_id": f"r{k}", "dataset": "A", "converged": True,
                         "feed": 121, "cyclen": cy, "library_id": "5.8_5.1",
                         "pattern": "p", "case_pair": "A01_A02"})
            k += 1
    # add non-A / non-121 / non-converged noise that must be excluded
    rows += [
        {"record_id": "bx", "dataset": "B", "converged": True, "feed": 121,
         "cyclen": 705.0, "library_id": "ga80", "pattern": "p", "case_pair": "K1_K2"},
        {"record_id": "fx", "dataset": "A", "converged": True, "feed": 117,
         "cyclen": 705.0, "library_id": "5.8_5.1", "pattern": "p", "case_pair": "A01_A02"},
        {"record_id": "nx", "dataset": "A", "converged": False, "feed": 121,
         "cyclen": 705.0, "library_id": "5.8_5.1", "pattern": "p", "case_pair": "A01_A02"},
    ]
    return pd.DataFrame(rows)


class _StubModel:
    """Predicts cyclen = truth + a per-band affine error (bias, then only in band)."""

    def __init__(self, bias_by_band: dict[tuple[float, float], float] | None = None,
                 global_bias: float = 0.0):
        self.bias_by_band = bias_by_band or {}
        self.global_bias = global_bias

    def predict_rows_raw(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.full((len(rows), 7), np.nan, dtype=float)
        cy = pd.to_numeric(rows["cyclen"], errors="coerce").to_numpy(float)
        pred = cy + self.global_bias
        for i, c in enumerate(cy):
            for (lo, hi), b in self.bias_by_band.items():
                if lo <= c < hi:
                    pred[i] = c + b
        out[:, 3] = pred
        return out


# --------------------------------------------------------------------------- #
# deterministic sampling
# --------------------------------------------------------------------------- #
def test_sample_is_deterministic_and_filtered():
    df = _corpus()
    s1 = sample_legacy_tail_rows(df, bands=BANDS, feed=121, sample_per_band=150, seed=0)
    s2 = sample_legacy_tail_rows(df, bands=BANDS, feed=121, sample_per_band=150, seed=0)
    for band in BANDS:
        key = (band[0], band[1])
        assert len(s1[key]) == 150
        # stable across calls (same record_ids, same order)
        assert list(s1[key]["record_id"]) == list(s2[key]["record_id"])
        # only converged Dataset-A feed-121 rows in-band
        assert set(s1[key]["dataset"]) == {"A"}
        assert set(s1[key]["feed"].astype(int)) == {121}
        assert bool(s1[key]["converged"].all())
        cy = s1[key]["cyclen"].to_numpy()
        assert cy.min() >= band[0] and cy.max() < band[1]


def test_sample_seed_changes_selection():
    df = _corpus()
    a = sample_legacy_tail_rows(df, bands=BANDS, sample_per_band=150, seed=0)
    b = sample_legacy_tail_rows(df, bands=BANDS, sample_per_band=150, seed=7)
    key = (700.0, 720.0)
    assert list(a[key]["record_id"]) != list(b[key]["record_id"])


def test_sample_growth_invariance():
    """Adding rows to a band must not change which pre-existing rows are picked
    when the pre-existing count already exceeds the sample cap."""
    df = _corpus(n_per_band=400)
    picked = set(sample_legacy_tail_rows(
        df, bands=BANDS, sample_per_band=150, seed=0)[(700.0, 720.0)]["record_id"])
    # append brand-new rows to the same band; their ids never collide.
    extra = pd.DataFrame([{
        "record_id": f"NEW{i}", "dataset": "A", "converged": True, "feed": 121,
        "cyclen": 710.0, "library_id": "5.8_5.1", "pattern": "p", "case_pair": "A01_A02"}
        for i in range(50)])
    grown = pd.concat([df, extra], ignore_index=True)
    picked2 = set(sample_legacy_tail_rows(
        grown, bands=BANDS, sample_per_band=150, seed=0)[(700.0, 720.0)]["record_id"])
    # every newly-added id that displaces an old one can only be one that hashes
    # below the cap; the pre-existing selected set is a superset-stable core.
    survivors = picked & picked2
    assert len(survivors) >= 150 - 50   # at most the 50 new rows can displace


# --------------------------------------------------------------------------- #
# scorer pass / fail
# --------------------------------------------------------------------------- #
def test_equal_models_pass():
    df = _corpus()
    old = _StubModel(global_bias=0.5)
    new = _StubModel(global_bias=0.5)     # identical skill
    res = score_legacy_tail_no_regression(old, new, df, bands=BANDS, epsilon=2.0)
    assert res["pass"] is True
    assert res["worst_mae_increase"] == pytest.approx(0.0, abs=1e-9)
    assert len(res["bands"]) == 3


def test_tail_collapse_is_caught():
    df = _corpus()
    old = _StubModel(global_bias=0.5)                      # ~0.5 MAE everywhere
    # candidate collapses ONLY on the 700-720 tail (the escaped-gate signature).
    new = _StubModel(bias_by_band={(700.0, 720.0): -40.0}, global_bias=0.5)
    res = score_legacy_tail_no_regression(old, new, df, bands=BANDS, epsilon=2.0)
    assert res["pass"] is False
    tail = [b for b in res["bands"] if b["band"] == [700.0, 720.0]][0]
    assert tail["mae_increase"] > 30.0
    # the fine mid bands do not trip the gate on their own
    mid = [b for b in res["bands"] if b["band"] == [660.0, 680.0]][0]
    assert mid["mae_increase"] == pytest.approx(0.0, abs=1e-9)


def test_small_degradation_within_epsilon_passes():
    df = _corpus()
    old = _StubModel(global_bias=0.0)
    new = _StubModel(global_bias=1.5)     # +1.5 EFPD MAE, under eps=2.0
    res = score_legacy_tail_no_regression(old, new, df, bands=BANDS, epsilon=2.0)
    assert res["pass"] is True
    assert res["worst_mae_increase"] == pytest.approx(1.5, abs=1e-6)


def test_insufficient_rows_skips_band():
    df = _corpus(n_per_band=2)             # < 3 per band
    old = _StubModel(); new = _StubModel()
    res = score_legacy_tail_no_regression(old, new, df, bands=BANDS, epsilon=2.0)
    assert res["pass"] is True
    assert all(b.get("note") == "insufficient rows" for b in res["bands"])


# --------------------------------------------------------------------------- #
# real-model parity check (guarded — proves the champion is sound on the tail)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (STORE / "records.parquet").exists(),
                    reason="store not present")
def test_real_v4_champion_sound_on_tail_via_provenance():
    """The v4 champion, scored through the parity-correct provenance path, keeps
    <5 EFPD MAE on every band including the 700-720 tail (the collapse is a serve-
    library artifact, not a model regression)."""
    from lpopt.model.model_api import PosValCnnBackend
    champ = REPO / "data" / "models" / "20260719_051103"
    if not (champ / "ensemble.json").is_file() and not list(champ.glob("member_*")):
        pytest.skip("v4 champion not present")
    be = PosValCnnBackend.from_dir(champ, store_dir=str(STORE),
                                   library_id="ga80", device="cpu")
    df = pd.read_parquet(STORE / "records.parquet")
    samples = sample_legacy_tail_rows(df, bands=BANDS, sample_per_band=80, seed=0)
    for band, rows in samples.items():
        if len(rows) < 3:
            continue
        pred = np.asarray(be.predict_rows_raw(rows))[:, 3]
        truth = rows["cyclen"].to_numpy(float)
        mae = float(np.mean(np.abs(pred - truth)))
        assert mae < 5.0, f"band {band} MAE {mae:.2f} EFPD >= 5"
