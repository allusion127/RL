"""Physics-prior residual learning for cyclen.

Pins four things:

1. the two-segment assembly reactivity curve behaves the way the docstring
   claims (holddown release below the hump, linear burnout decay above it);
2. the prior is a leakage-safe function of ``(pattern, library)`` + the static
   fuel table — no label, no record-specific burnup;
3. ``prior + residual`` reconstructs cyclen EXACTLY (the serve-time round trip);
4. the prior ALONE correlates with actual cyclen on the honest holdout, which is
   the whole justification for regressing the residual instead of the target.

Plus the flag-off contract: with ``cyclen_physics_prior=False`` no prior tensor
is attached and the training path is the legacy one.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.data.fuel_types import FuelLibrary, FuelVec              # noqa: E402
from lpopt.model.physics_prior import (                             # noqa: E402
    DEFAULT_SLOPE_PCM_PER_GWD, RHO_LEAK_PCM, CyclenPhysicsPrior, assembly_rho,
    cycle_burnup_batch, cycle_burnup_estimate, fit_cyclen_prior,
    prior_correlation, rho_pcm,
)
from lpopt.model.splits import SplitManifest                        # noqa: E402

STORE = "data/store"
#: the prior's measured holdout Pearson is ~0.84; guard well below it so the
#: test pins "the physics carries real signal" without being brittle.
MIN_HOLDOUT_PEARSON = 0.55


@pytest.fixture(scope="module")
def fuel():
    return FuelLibrary.from_parquet(f"{STORE}/fuel_types.parquet")


@pytest.fixture(scope="module")
def folds():
    recs = (pd.read_parquet(f"{STORE}/records.parquet")
            .drop_duplicates("record_id").set_index("record_id"))
    man = SplitManifest.from_json("data/splits/S1.json")
    tr = [i for i in man.record_ids("train") if i in recs.index][:4000]
    va = [i for i in man.record_ids("val") if i in recs.index][:2000]
    return recs.loc[tr].reset_index(), recs.loc[va].reset_index()


# --------------------------------------------------------------------------- #
# the formula
# --------------------------------------------------------------------------- #
def test_rho_pcm_matches_the_definition():
    assert rho_pcm(1.0) == pytest.approx(0.0)
    assert rho_pcm(1.25) == pytest.approx((0.25 / 1.25) * 1e5)
    assert math.isnan(rho_pcm(None))
    assert math.isnan(rho_pcm(0.0))


def _vec(**kw):
    return FuelVec(library_id="L", type_id="T", **kw)


def test_burnout_segment_decays_linearly_at_the_harvested_slope():
    v = _vec(kinf_peak=1.15, bu_peak_gwd=20.0, reactivity_swing_pcm=800.0,
             depletion_slope_pcm_per_gwd=-600.0)
    r20, s = assembly_rho(v, 20.0)
    r30, _ = assembly_rho(v, 30.0)
    assert s == pytest.approx(-600.0)
    assert r20 == pytest.approx(rho_pcm(1.15))
    assert r30 - r20 == pytest.approx(-600.0 * 10.0)


def test_holddown_segment_releases_reactivity_up_to_the_hump():
    v = _vec(kinf_peak=1.15, bu_peak_gwd=20.0, reactivity_swing_pcm=800.0,
             depletion_slope_pcm_per_gwd=-600.0)
    r0, _ = assembly_rho(v, 0.0)
    r10, _ = assembly_rho(v, 10.0)
    r20, _ = assembly_rho(v, 20.0)
    # fresh (fully suppressed) -> hump: monotone rise, exactly one full swing
    assert r0 < r10 < r20
    assert r20 - r0 == pytest.approx(800.0)
    assert r10 - r0 == pytest.approx(400.0)          # linear release


def test_monotone_design_has_no_swing_and_still_depletes():
    v = _vec(kinf_peak=1.13, bu_peak_gwd=0.0,
             reactivity_swing_pcm=float("nan"),
             depletion_slope_pcm_per_gwd=-470.0)
    r0, s = assembly_rho(v, 0.0)
    r10, _ = assembly_rho(v, 10.0)
    assert s == pytest.approx(-470.0)
    assert r0 == pytest.approx(rho_pcm(1.13))
    assert r10 - r0 == pytest.approx(-4700.0)


def test_missing_slope_falls_back_to_the_population_median():
    _, s = assembly_rho(_vec(kinf_peak=1.15), 10.0)
    assert s == pytest.approx(DEFAULT_SLOPE_PCM_PER_GWD)


def test_unharvested_origin_contributes_nothing():
    rho, s = assembly_rho(_vec(), 10.0)
    assert math.isnan(rho) and math.isnan(s)
    assert all(math.isnan(x) for x in assembly_rho(None, 10.0))


def test_kinf0_stands_in_when_no_hump_analysis_exists():
    """A row with a BOC point but no hump still pins the curve."""
    rho, _ = assembly_rho(_vec(kinf0=1.12), 0.0)
    assert rho == pytest.approx(rho_pcm(1.12))


# --------------------------------------------------------------------------- #
# leakage safety
# --------------------------------------------------------------------------- #
def test_prior_reads_no_label_column(fuel, folds):
    """Stripping every target/metric column must not change the prior by a bit."""
    train, _ = folds
    sub = train.head(200)
    keep = ["record_id", "pattern", "feed", "e_core", "e_split", "case_pair",
            "library_id", "sym_class", "dataset"]
    stripped = sub[keep].copy()
    a = cycle_burnup_batch(sub, fuel)
    b = cycle_burnup_batch(stripped, fuel)
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-1),
                                  np.nan_to_num(b, nan=-1))


def test_prepare_prior_asserts_when_folds_overlap(fuel, folds):
    """The train/val disjointness guard must actually fire."""
    from lpopt.model.train import TrainConfig, _prepare_cyclen_prior

    train, _ = folds

    class _DS:
        def __init__(self, df):
            self.df = df
            self._t = {}
            self.record_ids = df["record_id"].astype(str).tolist()

    cfg = TrainConfig()
    cfg.cyclen_physics_prior = True
    overlapping = _DS(train.head(50))
    with pytest.raises(AssertionError, match="train rows only"):
        _prepare_cyclen_prior(cfg, overlapping, _DS(train.head(50)), fuel,
                              STORE, "S1")


def test_prepare_prior_is_a_noop_when_the_flag_is_off(fuel, folds):
    from lpopt.model.train import TrainConfig, _prepare_cyclen_prior

    train, val = folds

    class _DS:
        def __init__(self, df):
            self.df = df
            self._t = {}
            self.record_ids = df["record_id"].astype(str).tolist()

    tr, va = _DS(train.head(50)), _DS(val.head(50))
    prior, vals = _prepare_cyclen_prior(TrainConfig(), tr, va, fuel, STORE, "S1")
    assert prior is None and vals is None
    assert "cyclen_prior" not in tr._t and "cyclen_prior" not in va._t


# --------------------------------------------------------------------------- #
# fit, round-trip, correlation
# --------------------------------------------------------------------------- #
def test_prior_is_finite_for_every_row(fuel, folds):
    """A non-finite prior would poison the residual; the fallback must cover it."""
    train, _ = folds
    p = fit_cyclen_prior(train, fuel, split="S1")
    vals = p.for_rows(train, fuel)
    assert np.all(np.isfinite(vals))


def test_residual_round_trip_is_exact(fuel, folds):
    """prior + (cyclen - prior) == cyclen, to floating-point equality."""
    train, _ = folds
    p = fit_cyclen_prior(train, fuel, split="S1")
    prior = p.for_rows(train, fuel)
    y = pd.to_numeric(train["cyclen"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y)
    residual = y[ok] - prior[ok]
    np.testing.assert_allclose(prior[ok] + residual, y[ok], rtol=0, atol=1e-9)


def test_residual_frame_replaces_only_cyclen(fuel, folds):
    from lpopt.model.train import residual_target_frame

    train, _ = folds
    p = fit_cyclen_prior(train, fuel, split="S1")
    prior = p.for_rows(train, fuel)
    res = residual_target_frame(train, prior)
    for col in ("f_r", "f_q", "cbc_max", "ao_abs"):
        pd.testing.assert_series_equal(res[col], train[col])
    y = pd.to_numeric(train["cyclen"], errors="coerce").to_numpy(float)
    np.testing.assert_allclose(
        pd.to_numeric(res["cyclen"]).to_numpy(float), y - prior, atol=1e-9)


def test_residual_is_a_tighter_target_than_the_raw_label(fuel, folds):
    """The point of the prior: the residual must have a SMALLER spread than
    cyclen itself, or the network gains nothing from regressing it."""
    train, _ = folds
    p = fit_cyclen_prior(train, fuel, split="S1")
    prior = p.for_rows(train, fuel)
    y = pd.to_numeric(train["cyclen"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & train["converged"].astype(bool).to_numpy()
    assert np.std(y[ok] - prior[ok]) < np.std(y[ok])


def test_prior_correlates_with_actual_cyclen_on_holdout(fuel, folds):
    """The reported justification metric — fit on TRAIN, scored on VAL."""
    train, val = folds
    p = fit_cyclen_prior(train, fuel, split="S1")
    c = prior_correlation(p, val, fuel)
    assert c["n"] > 100
    assert c["pearson"] > MIN_HOLDOUT_PEARSON, c
    assert c["spearman"] > 0.4, c


def test_prior_is_in_physical_efpd_units(fuel, folds):
    train, val = folds
    p = fit_cyclen_prior(train, fuel, split="S1")
    vals = p.for_rows(val, fuel)
    # cyclen lives around 600-720 EFPD; the prior must land in that regime.
    assert 400.0 < float(np.median(vals)) < 900.0


def test_degenerate_fit_falls_back_to_a_constant(fuel):
    df = pd.DataFrame({
        "record_id": ["a", "b"], "cyclen": [600.0, 620.0],
        "converged": [True, True],
    })
    p = fit_cyclen_prior(df, fuel, b_cycle=np.array([5.0, 5.0]))
    assert p.alpha == 0.0
    assert p.beta == pytest.approx(610.0)
    assert p.from_b(np.array([np.nan])) == pytest.approx(610.0)


def test_artifact_round_trips_through_json(tmp_path, fuel, folds):
    train, _ = folds
    p = fit_cyclen_prior(train, fuel, split="S1")
    path = p.save(tmp_path / "prior.json")
    q = CyclenPhysicsPrior.load(path)
    assert q.alpha == pytest.approx(p.alpha)
    assert q.beta == pytest.approx(p.beta)
    assert q.rho_leak == pytest.approx(p.rho_leak)
    np.testing.assert_allclose(q.for_rows(train.head(20), fuel),
                               p.for_rows(train.head(20), fuel))


def test_rho_leak_error_is_absorbed_by_the_affine_fit(fuel, folds):
    """Assumption 4: because the core-average decay rate barely varies, a wrong
    RHO_LEAK is nearly a constant shift and must not degrade the correlation."""
    train, val = folds
    base = prior_correlation(fit_cyclen_prior(train, fuel), val, fuel)
    shifted = prior_correlation(
        fit_cyclen_prior(train, fuel, rho_leak=RHO_LEAK_PCM + 1500.0),
        val, fuel)
    assert abs(base["pearson"] - shifted["pearson"]) < 0.05
