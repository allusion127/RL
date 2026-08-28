"""Per-target split-conformal prediction intervals (lpopt.model.conformal).

Pure layer (torch-free):
  * the finite-sample split-conformal quantile ``ceil((n+1)(1-a))``-th order stat,
    incl. the ``+inf`` (too-few-points) guard;
  * abs vs sigma-normalized nonconformity;
  * per-cell + global fit with the ``min_cell`` gate;
  * the distribution-free MARGINAL coverage guarantee on exchangeable data;
  * held-out-cell k-fold coverage + the abs/norm score-type selection;
  * ``interval_arrays`` layout (7-col surrogate, NaN off-target, from_cell flags).

Serve layer (real backend, tiny v4 ensemble):
  * ``predict_interval`` is REPORT-ONLY — predict()/extra/convergence are byte-
    identical whether or not a conformal artifact is installed;
  * intervals center on the served mean with the fitted half-width (abs + norm);
  * the ``available=False`` all-NaN path with no artifact;
  * ``from_dir`` load round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lpopt.model.conformal import (
    CONFORMAL_TARGETS, conformal_quantile, conformal_targets, coverage_table,
    fit_conformal, halfwidths, interval_arrays, kfold_cell_coverage,
    nonconformity, select_score_type, _fit_cell_qs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


# --------------------------------------------------------------------------- #
# pure: quantile
# --------------------------------------------------------------------------- #
def test_conformal_quantile_finite_sample() -> None:
    s = list(range(1, 31))                       # 1..30
    # k = ceil(31 * 0.9) = 28 -> 28th order statistic (1-based) = 28
    assert conformal_quantile(s, 0.10) == 28.0
    # k = ceil(31 * 0.68) = 22 -> 22
    assert conformal_quantile(s, 0.32) == 22.0


def test_conformal_quantile_too_few_is_inf() -> None:
    # n=5, alpha=0.10: ceil(6*0.9)=6 > 5 -> vacuous +inf bound
    assert conformal_quantile([1, 2, 3, 4, 5], 0.10) == np.inf
    assert conformal_quantile([], 0.10) == np.inf


def test_conformal_quantile_ignores_nonfinite() -> None:
    s = [1.0, 2.0, np.nan, np.inf, 3.0, 4.0]     # 4 finite
    # finite n=4, alpha=0.10: ceil(5*0.9)=5 > 4 -> inf
    assert conformal_quantile(s, 0.10) == np.inf
    # alpha=0.32: ceil(5*0.68)=4 -> 4th finite order stat = 4.0
    assert conformal_quantile(s, 0.32) == 4.0


# --------------------------------------------------------------------------- #
# pure: nonconformity
# --------------------------------------------------------------------------- #
def test_nonconformity_abs_and_norm() -> None:
    pred = np.array([10.0, 12.0, 8.0])
    actual = np.array([9.0, 15.0, 8.0])
    sig = np.array([2.0, 0.0, 4.0])
    ae = nonconformity(pred, actual, None, "abs")
    assert np.allclose(ae, [1.0, 3.0, 0.0])
    z = nonconformity(pred, actual, sig, "norm")
    assert z[0] == pytest.approx(0.5)
    assert np.isnan(z[1])                          # sigma <= 0 -> drop
    assert z[2] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        nonconformity(pred, actual, sig, "bogus")


# --------------------------------------------------------------------------- #
# pure: per-cell + global fit, min_cell gate
# --------------------------------------------------------------------------- #
def test_fit_cell_qs_min_cell_and_global() -> None:
    rng = np.random.default_rng(0)
    big = np.abs(rng.normal(0, 1, 40))
    small = np.abs(rng.normal(0, 1, 5))
    scores = np.concatenate([big, small])
    keys = ["cellA"] * 40 + ["cellB"] * 5
    cells, gq, n = _fit_cell_qs(scores, keys, [0.10, 0.32], min_cell=20)
    assert n == 45
    assert "cellA" in cells and cells["cellA"]["n"] == 40   # >= min_cell -> fitted
    assert "cellB" not in cells                             # < min_cell -> global only
    # global uses ALL finite scores
    assert gq["0.1"] == pytest.approx(conformal_quantile(scores, 0.10))


# --------------------------------------------------------------------------- #
# pure: the marginal coverage guarantee (the whole point)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.10, 0.32])
def test_marginal_coverage_guarantee(alpha) -> None:
    rng = np.random.default_rng(7)
    # exchangeable calibration + test errors from a heavy-ish distribution
    cal = np.abs(rng.standard_t(5, size=3000))
    test = np.abs(rng.standard_t(5, size=20000))
    q = conformal_quantile(cal, alpha)
    cov = float(np.mean(test <= q))
    # split-conformal marginal coverage is >= 1-alpha (finite-sample, slightly over)
    assert cov >= (1 - alpha) - 0.02
    assert cov <= (1 - alpha) + 0.06


# --------------------------------------------------------------------------- #
# pure: held-out-cell k-fold coverage + score-type selection
# --------------------------------------------------------------------------- #
def test_kfold_cell_coverage_near_nominal() -> None:
    rng = np.random.default_rng(3)
    keys, pred, actual, sig = [], [], [], []
    for c in range(8):                              # 8 cells x 40 rows
        scale = 1.0 + 0.5 * c                       # heteroscedastic across cells
        for _ in range(40):
            a = rng.normal(100.0, 10.0)
            e = rng.normal(0.0, scale)
            keys.append(f"cell{c}"); actual.append(a)
            pred.append(a + e); sig.append(scale)
    pred = np.array(pred); actual = np.array(actual); sig = np.array(sig)
    m = kfold_cell_coverage(pred, actual, sig, keys, "norm", [0.10, 0.32],
                            min_cell=20, k=2, seed=0)
    assert m is not None
    assert m["cov"]["0.1"] >= 0.85                  # held-out-cell coverage near 90%
    assert m["cov"]["0.32"] >= 0.60


def test_select_score_type_prefers_valid_then_tighter() -> None:
    alphas = [0.10, 0.32]
    valid_wide = {"cov": {"0.1": 0.93, "0.32": 0.70}, "width": {"0.1": 10.0, "0.32": 5.0}}
    valid_tight = {"cov": {"0.1": 0.90, "0.32": 0.68}, "width": {"0.1": 6.0, "0.32": 3.0}}
    invalid = {"cov": {"0.1": 0.80, "0.32": 0.55}, "width": {"0.1": 1.0, "0.32": 0.5}}
    # both valid -> tighter wins
    assert select_score_type(valid_wide, valid_tight, alphas) == "norm"
    # only abs valid -> abs, even though norm is tighter-but-invalid
    assert select_score_type(valid_wide, invalid, alphas) == "abs"
    # neither valid -> higher primary coverage
    assert select_score_type(invalid, {"cov": {"0.1": 0.86, "0.32": 0.6},
                                       "width": {"0.1": 9, "0.32": 4}}, alphas) == "norm"
    # norm unavailable -> abs
    assert select_score_type(valid_wide, None, alphas) == "abs"


# --------------------------------------------------------------------------- #
# pure: halfwidths + interval_arrays layout
# --------------------------------------------------------------------------- #
def test_halfwidths_abs_vs_norm_and_from_cell() -> None:
    entry_abs = {"score_type": "abs",
                 "cells": {"k1": {"n": 30, "q": {"0.1": 5.0}}},
                 "global": {"0.1": 9.0}}
    keys = ["k1", "k2"]                              # k2 -> global fallback
    hw, fc = halfwidths(entry_abs, keys, np.array([2.0, 2.0]), 0.10)
    assert np.allclose(hw, [5.0, 9.0])
    assert list(fc) == [True, False]

    entry_norm = {"score_type": "norm",
                  "cells": {"k1": {"n": 30, "q": {"0.1": 2.0}}},
                  "global": {"0.1": 3.0}}
    hw2, _ = halfwidths(entry_norm, keys, np.array([4.0, 4.0]), 0.10)
    assert np.allclose(hw2, [2.0 * 4.0, 3.0 * 4.0])  # q * sigma


def test_interval_arrays_layout_and_offtarget_nan() -> None:
    artifact = {"bin_width": 0.25, "per_target": {
        "cyclen": {"surrogate_col": 3, "score_type": "abs",
                   "cells": {"kA": {"n": 30, "q": {"0.1": 12.0}}},
                   "global": {"0.1": 20.0}},
    }}
    n = 3
    mean = np.zeros((n, 7)); mean[:] = 100.0
    sigma = np.full((n, 7), 5.0)
    keys = ["kA", "kB", "kA"]
    lo, hi, hw, fc = interval_arrays(mean, sigma, keys, artifact, 0.10)
    # cyclen col 3: kA -> 12 half-width, kB -> 20 global
    assert np.allclose(hw[:, 3], [12.0, 20.0, 12.0])
    assert np.allclose(lo[:, 3], mean[:, 3] - hw[:, 3])
    assert np.allclose(hi[:, 3], mean[:, 3] + hw[:, 3])
    assert list(fc[:, 3]) == [True, False, True]
    # every OTHER column is off-target -> NaN bounds
    for col in (0, 1, 2, 4, 5, 6):
        assert np.isnan(lo[:, col]).all() and np.isnan(hi[:, col]).all()
    # lower <= upper wherever finite
    finite = np.isfinite(lo)
    assert (lo[finite] <= hi[finite]).all()


# =========================================================================== #
# serve layer — real backend (tiny v4 ensemble)
# =========================================================================== #
pytest.importorskip("torch")
import torch                                                         # noqa: E402

from lpopt.model.featurize import CHANNELS_V4, FeatureEncoder        # noqa: E402
from lpopt.model.model_api import PosValCnnBackend, IntervalPrediction  # noqa: E402
from lpopt.model.net import PosValNet, PosValNetConfig               # noqa: E402
from lpopt.model.train import save_member                            # noqa: E402
from lpopt.data.schema import unpack_pattern                         # noqa: E402
from lpopt.data.store import StoreReader                             # noqa: E402
from lpopt.vendor.masterrl.domain import CaseKey                     # noqa: E402

_ZMEAN = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0]
_ZSTD = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0]
_TARGETS = ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
            "discharge_burnup", "max_pin_burnup")


def _make_v4_ensemble(tmp: Path, n: int = 2) -> Path:
    ens = tmp / "ens"
    globals_names = list(FeatureEncoder(cond_schema="v4").globals_names)
    cfg = PosValNetConfig(in_channels=len(CHANNELS_V4), n_globals=len(globals_names))
    for i in range(n):
        seed = 700 + i
        net = PosValNet(cfg)
        meta = {
            "net_config": cfg.__dict__, "cond_schema": "v4",
            "channels": list(CHANNELS_V4), "globals": globals_names,
            "target_names": list(_TARGETS),
            "target_zscore": {"mean": _ZMEAN, "std": _ZSTD},
            "seed": seed, "split": "S1", "versions": {"torch": torch.__version__},
        }
        save_member(ens / f"member_{seed}", net, meta)
    return ens


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    ens = _make_v4_ensemble(tmp_path_factory.mktemp("conformal"), n=2)
    return PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")


@pytest.fixture(scope="module")
def sample_case(backend):
    reader = StoreReader(STORE)
    row = reader.records[reader.records["library_id"].astype(str) == "ga80"].iloc[0]
    pat = unpack_pattern(str(row["pattern"]))
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    return pat, case


def _artifact_for(backend, pat, case, *, cyclen_kind="abs"):
    """Build a conformal artifact whose cyclen cell matches this request's key."""
    key = backend._conformal_cell_keys([pat], [case])[0]
    return {
        "schema": "split_conformal_v1", "bin_width": backend.conformal_bin_width,
        "alphas": [0.10, 0.32],
        "per_target": {
            "cyclen": {"surrogate_col": 3, "score_type": cyclen_kind,
                       "cells": {key: {"n": 30, "q": {"0.1": 12.0, "0.32": 6.0}}},
                       "global": {"0.1": 20.0, "0.32": 10.0}},
            "cbc_max": {"surrogate_col": 1, "score_type": "abs",
                        "cells": {}, "global": {"0.1": 50.0, "0.32": 25.0}},
        },
    }, key


def test_predict_interval_is_report_only(backend, sample_case) -> None:
    pat, case = sample_case
    backend.set_conformal(None)
    base = backend.predict([pat], case, 0.0)
    base_extra = backend.predict_extra([pat], case, 0.0)
    base_conv = backend.predict_convergence([pat], case, 0.0)

    art, _key = _artifact_for(backend, pat, case)
    backend.set_conformal(art)
    after = backend.predict([pat], case, 0.0)
    # predict() and every sibling accessor are byte-identical with conformal loaded
    assert np.array_equal(after.mean, base.mean, equal_nan=True)
    assert np.array_equal(after.epistemic_std, base.epistemic_std, equal_nan=True)
    assert np.array_equal(after.calibrated_std, base.calibrated_std, equal_nan=True)
    assert np.array_equal(backend.predict_extra([pat], case, 0.0).mean,
                          base_extra.mean, equal_nan=True)
    assert np.array_equal(backend.predict_convergence([pat], case, 0.0),
                          base_conv, equal_nan=True)
    backend.set_conformal(None)


def test_predict_interval_abs_centers_on_mean(backend, sample_case) -> None:
    pat, case = sample_case
    art, _key = _artifact_for(backend, pat, case, cyclen_kind="abs")
    backend.set_conformal(art)
    mean_cy = float(backend.predict([pat], case, 0.0).mean[0, 3])
    iv = backend.predict_interval([pat], case, alpha=0.10)
    assert isinstance(iv, IntervalPrediction) and iv.available
    assert iv.coverage == pytest.approx(0.90)
    # abs: half-width == fitted per-cell q, centered on the served mean
    assert iv.halfwidth[0, 3] == pytest.approx(12.0)
    assert iv.lower[0, 3] == pytest.approx(mean_cy - 12.0)
    assert iv.upper[0, 3] == pytest.approx(mean_cy + 12.0)
    assert bool(iv.from_cell[0, 3]) is True
    # cbc_max has no fitted cell -> global fallback (from_cell False), still finite
    assert iv.halfwidth[0, 1] == pytest.approx(50.0)
    assert bool(iv.from_cell[0, 1]) is False
    # off-target columns stay NaN
    for col in (0, 2, 4, 5, 6):
        assert np.isnan(iv.lower[0, col]) and np.isnan(iv.upper[0, col])
    backend.set_conformal(None)


def test_predict_interval_norm_scales_with_sigma(backend, sample_case) -> None:
    pat, case = sample_case
    art, _key = _artifact_for(backend, pat, case, cyclen_kind="norm")
    backend.set_conformal(art)
    pred = backend.predict([pat], case, 0.0)
    sig = float(pred.calibrated_std[0, 3])
    iv = backend.predict_interval([pat], case, alpha=0.10)
    assert iv.halfwidth[0, 3] == pytest.approx(12.0 * sig)   # q * sigma
    backend.set_conformal(None)


def test_predict_interval_unavailable_without_artifact(backend, sample_case) -> None:
    pat, case = sample_case
    backend.set_conformal(None)
    iv = backend.predict_interval([pat], case, alpha=0.10)
    assert iv.available is False
    assert np.isnan(iv.lower).all() and np.isnan(iv.upper).all()
    assert not backend.has_conformal()
    # empty batch is still well-formed
    empty = backend.predict_interval([], case, alpha=0.10)
    assert empty.lower.shape == (0, 7)


def test_from_dir_loads_conformal(tmp_path, sample_case) -> None:
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    pat, case = sample_case
    ens = _make_v4_ensemble(tmp_path, n=2)
    raw = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    base_mean = raw.predict([pat], case, 0.0).mean.copy()
    art, key = _artifact_for(raw, pat, case)
    (ens / "conformal.json").write_text(json.dumps(art), encoding="utf-8")

    loaded = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert loaded.has_conformal()
    assert "cyclen" in conformal_targets(loaded.conformal)
    # predict() unchanged by the presence of the artifact …
    assert np.array_equal(loaded.predict([pat], case, 0.0).mean, base_mean, equal_nan=True)
    # … and the interval is served from the loaded quantiles
    iv = loaded.predict_interval([pat], case, alpha=0.10)
    assert iv.available and iv.halfwidth[0, 3] == pytest.approx(12.0)


# --------------------------------------------------------------------------- #
# end-to-end fit on the real champion holdout (skipped without store/split)
# --------------------------------------------------------------------------- #
def test_fit_conformal_end_to_end_smoke() -> None:
    split = REPO_ROOT / "data" / "splits" / "S1.json"
    if not (STORE / "records.parquet").is_file() or not split.is_file():
        pytest.skip("store/split not present")
    import glob
    champs = sorted(glob.glob(str(REPO_ROOT / "data" / "models" / "*" / "calibration.json")))
    if not champs:
        pytest.skip("no champion ensemble present")
    model_dir = Path(champs[-1]).parent
    art = fit_conformal(model_dir, store_dir=STORE,
                        splits_dir=REPO_ROOT / "data" / "splits",
                        split="S1", write=False)
    assert art["schema"] == "split_conformal_v1"
    assert set(art["targets"]) == {n for n, _ in CONFORMAL_TARGETS}
    rows = coverage_table(art)
    assert rows and all(r["score_type"] in ("abs", "norm") for r in rows)
    # every fitted target achieves near-nominal held-out-cell coverage at 90%
    for r in rows:
        if r["cov@90"] is not None:
            assert r["cov@90"] >= 0.85
