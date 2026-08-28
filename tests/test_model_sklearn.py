"""sklearn_fallback backend (plan sec. 4.5): Protocol conformance + case remap."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from lpopt.model.model_api import PositionValueModel
from lpopt.model.model_sklearn import SklearnBackend, row_to_record
from lpopt.vendor.masterrl.domain import CaseKey, FOM, PatternRecord
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction, TARGET_NAMES
from lpopt.search.genome import random_genome


def _records(n=16, pair="K1_K2", feed=121, seed=0):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        pat = random_genome(rng, pair, (feed - 1) // 4).to_pattern()
        f_r = 1.50 + 0.02 * (i % 5)
        recs.append(PatternRecord(
            case=CaseKey(pair, feed), cell=5.2, seed_id=f"s{i}", pattern=pat,
            fom=FOM(f_r=f_r, cbc_max=1500 + 5 * i, f_q=f_r * 1.4,
                    cyclen=610 + 3 * i, ao_min=-0.2, ao_max=0.2, converged=True),
            ncyc=7, deck_path=Path("."), shf_path=Path("."),
        ))
    return recs


def test_backend_satisfies_protocol_and_predict_shape():
    backend = SklearnBackend.fit(_records(), library_id="ga80")
    assert isinstance(backend, PositionValueModel)
    rng = random.Random(1)
    pats = [random_genome(rng, "K1_K2", 30).to_pattern() for _ in range(6)]
    pred = backend.predict(pats, CaseKey("K1_K2", 121), 5.2)
    assert isinstance(pred, SurrogatePrediction)
    assert pred.mean.shape == (6, len(TARGET_NAMES))
    conv = backend.predict_convergence(pats, CaseKey("K1_K2", 121), 5.2)
    assert conv.shape == (6,)
    assert backend.position_values(pats[0], CaseKey("K1_K2", 121), 5.2) is None


def test_unknown_case_remaps_without_keyerror():
    backend = SklearnBackend.fit(_records(), library_id="ga80")
    rng = random.Random(2)
    # An unfitted feed grid point of the SAME pair (plan sec. 4.5: the vendor
    # case one-hot is fit-frozen, so a feed-117 K1_K2 case was never seen).  The
    # fallback remaps to the fitted reference case (a warning, never a KeyError).
    pats = [random_genome(rng, "K1_K2", 29).to_pattern() for _ in range(4)]  # feed 117
    with pytest.warns(RuntimeWarning):
        pred = backend.predict(pats, CaseKey("K1_K2", 117), 5.2)
    assert pred.mean.shape == (4, len(TARGET_NAMES))


def test_row_to_record_drops_incomplete_rows():
    good = {"pattern": random_genome(random.Random(3), "K1_K2", 30).to_pattern().canonical(),
            "case_pair": "K1_K2", "feed": 121, "e_core": 5.2, "converged": True,
            "f_r": 1.52, "f_q": 2.1, "cbc_max": 1500.0, "cyclen": 620.0, "ao_abs": 0.2,
            "record_id": "abc", "n_cycles": 7, "max_assembly_burnup": None, "max_pin_burnup": None}
    assert row_to_record(good) is not None
    bad = dict(good, f_r=None)     # missing a primary FOM -> dropped
    assert row_to_record(bad) is None


def test_fit_from_store_reads_campaign_case():
    store_dir = Path(__file__).resolve().parents[1] / "data" / "store"
    if not (store_dir / "records.parquet").exists():
        pytest.skip("no store present")
    backend = SklearnBackend.fit_from_store(
        store_dir, [CaseKey("K1_K2", 121)], library_id="ga80", max_rows=200
    )
    assert CaseKey("K1_K2", 121) in backend.fitted_cases
    rng = random.Random(5)
    pats = [random_genome(rng, "K1_K2", 30).to_pattern() for _ in range(3)]
    pred = backend.predict(pats, CaseKey("K1_K2", 121), 5.2)
    assert np.isfinite(pred.mean[:, :5]).all()
