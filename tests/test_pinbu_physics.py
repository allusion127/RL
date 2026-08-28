"""Physics pin-burnup estimator + harvest (plan 12.4 addendum).

Covers: the ``fuel_types`` BRP/BU curve harvest on real DeCART samples (.sum EDIT3
and HGC %DIST map7, with cross-parity); the additive parquet augment
(other columns byte-identical); the ``pinbu_physics`` curve/estimator math; the
no-leakage fit on train-fold Dataset-P rows; and the serve-side application /
disable in ``PosValCnnBackend.predict`` (raw head preserved, other columns
untouched).  Every check that needs the real store / champion / lattice tree
degrades to a skip when the artefact is absent, so the suite runs anywhere.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpopt.data.fuel_types import (
    PIN_BU_COLUMNS, FuelLibrary, augment_fuel_table_pin_bu,
    build_fuel_table, parse_fa_sum_pin_bu, parse_hgc_pin_bu,
    summarize_pin_bu_curve,
)
from lpopt.model.pinbu_physics import (
    DEFAULT_RATIO, PINBU_PHYSICS_NAME, PINBU_SURROGATE_COL,
    PinBuPhysicsEstimator, PinBuRatioCurve, core_discharge_estimate,
    fit_pinbu_physics, fit_row_mask, load_pinbu_physics, resolve_peak_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"
APR1400 = REPO_ROOT.parent / "0_APR1400"
GA80_HGC = REPO_ROOT.parent / "3_GA_Surrogate" / "FEASIBLE_PACKAGE" / "hgc"


def _lattice_with_both() -> Path | None:
    """A lattice ``FA_*.sum`` that has a sibling ``FA_*.HGC`` (both products)."""
    for s in sorted(glob.glob(str(APR1400 / "**" / "FA_*.sum"), recursive=True)):
        if Path(s).with_suffix(".HGC").is_file():
            return Path(s)
    return None


# --------------------------------------------------------------------------- #
# curve-summary math (pure)
# --------------------------------------------------------------------------- #
def test_summarize_pin_bu_curve_math() -> None:
    # ratio(BU) = 1.04 + 0.8/BU exactly -> the fit must recover it.
    pts = [(bu, 1.04 + 0.8 / bu) for bu in (10, 20, 30, 40, 50, 60, 70, 80)]
    s = summarize_pin_bu_curve(pts)
    assert s["pin_bu_n_pts"] == 8.0
    assert s["pin_bu_bu_max"] == 80.0
    assert s["pin_bu_ratio_asym"] == pytest.approx(1.04 + 0.8 / 80.0, abs=1e-9)
    assert s["pin_bu_r_inf"] == pytest.approx(1.04, abs=1e-6)
    assert s["pin_bu_paramA"] == pytest.approx(0.8, abs=1e-6)


def test_summarize_pin_bu_curve_empty_and_short() -> None:
    assert summarize_pin_bu_curve([]) == {}
    # no points in the discharge tail (all BU < 25) -> plateau only, no fit cols.
    s = summarize_pin_bu_curve([(5.0, 1.2), (10.0, 1.15)])
    assert s["pin_bu_ratio_asym"] == pytest.approx(1.15)
    assert "pin_bu_r_inf" not in s and "pin_bu_paramA" not in s


def test_summarize_drops_nonfinite_and_nonpositive() -> None:
    pts = [(0.0, 1.3), (-1.0, 1.2), (30.0, float("nan")), (40.0, 1.06), (50.0, 1.05)]
    s = summarize_pin_bu_curve(pts)
    assert s["pin_bu_n_pts"] == 2.0            # only the two finite BU>0 rows survive
    assert s["pin_bu_bu_max"] == 50.0


# --------------------------------------------------------------------------- #
# real-sample harvest + .sum <-> HGC parity
# --------------------------------------------------------------------------- #
def test_parse_fa_sum_pin_bu_real_sample() -> None:
    s = _lattice_with_both()
    if s is None:
        pytest.skip("no lattice FA_*.sum tree present")
    out = parse_fa_sum_pin_bu(s)
    # a real depletion curve: many points, plateau in the physical lattice range,
    # and a declining shape (asym < BOC power-peaking ~1.1-1.2).
    assert out["pin_bu_n_pts"] >= 20
    assert 1.02 <= out["pin_bu_ratio_asym"] <= 1.15
    assert out["pin_bu_bu_max"] >= 40.0
    assert 1.0 <= out["pin_bu_r_inf"] <= 1.10


def test_sum_hgc_pin_bu_parity() -> None:
    """A type carrying both products agrees on the harvested curve (map7 == BRP)."""
    s = _lattice_with_both()
    if s is None:
        pytest.skip("no lattice with both .sum and .HGC present")
    a = parse_fa_sum_pin_bu(s)
    b = parse_hgc_pin_bu(s.with_suffix(".HGC"))
    assert a["pin_bu_ratio_asym"] == pytest.approx(b["pin_bu_ratio_asym"], abs=5e-3)
    assert a["pin_bu_r_inf"] == pytest.approx(b["pin_bu_r_inf"], abs=5e-3)


def test_ga80_hgc_pin_bu_from_map7() -> None:
    hgcs = sorted(glob.glob(str(GA80_HGC / "FA_*.HGC")))
    if not hgcs:
        pytest.skip("ga80 HGC package not present")
    filled = 0
    for hp in hgcs:
        out = parse_hgc_pin_bu(hp)
        if not out:
            continue
        filled += 1
        # lattice peak-pin/assembly burnup ratio sits just above 1 at discharge.
        assert 1.02 <= out["pin_bu_ratio_asym"] <= 1.20
    assert filled >= 1


# --------------------------------------------------------------------------- #
# additive parquet augment: other columns byte-identical
# --------------------------------------------------------------------------- #
def test_augment_pin_bu_is_additive(tmp_path) -> None:
    if not (STORE / "fuel_types.parquet").is_file():
        pytest.skip("fuel store not present")
    # build a fresh (paramA-free) table into a scratch store, strip pin_bu, then
    # augment and confirm every pre-existing column is byte-identical.
    from lpopt.config import load_config
    from lpopt.data.fuel_types import fuel_paths_from_config

    cfg = load_config(str(REPO_ROOT / "lpopt.inp"))
    paths = fuel_paths_from_config(cfg)
    full = build_fuel_table(paths, persist=False)
    base = full.drop(columns=list(PIN_BU_COLUMNS))
    scratch = tmp_path / "fuel_types.parquet"
    base.to_parquet(scratch, index=False)

    paths.store = scratch
    merged = augment_fuel_table_pin_bu(paths, persist=True)

    assert len(merged) == len(base)
    for c in PIN_BU_COLUMNS:
        assert c in merged.columns
    reloaded = pd.read_parquet(scratch)
    for c in base.columns:
        left, right = base[c].reset_index(drop=True), reloaded[c].reset_index(drop=True)
        if c == "source_flags":
            assert all(list(x) == list(y) for x, y in zip(left, right))
        elif left.dtype == object:
            assert left.astype(str).equals(right.astype(str))
        else:
            assert np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True)
    # at least the lattice libraries are filled.
    assert reloaded["pin_bu_ratio_asym"].notna().sum() >= 40


# --------------------------------------------------------------------------- #
# curve reconstruction / estimator math
# --------------------------------------------------------------------------- #
def test_ratio_curve_reconstruction_and_clamps() -> None:
    c = PinBuRatioCurve(r_inf=1.04, paramA=0.8, ratio_asym=1.05, bu_max=80.0,
                        ff_pin_max=1.18)
    # interior: r_inf + A/BU
    assert c.ratio_at(40.0) == pytest.approx(1.04 + 0.8 / 40.0)
    assert c.pin_bu(40.0) == pytest.approx((1.04 + 0.8 / 40.0) * 40.0)
    # monotone decreasing toward the plateau, then floored at the plateau.
    assert c.ratio_at(50.0) < c.ratio_at(40.0)
    assert c.ratio_at(1e6) == pytest.approx(1.05)          # floored at ratio_asym
    # never exceeds the BOC power peaking upper bound.
    assert c.ratio_at(0.5) <= 1.18 + 1e-9


def test_ratio_curve_fallbacks() -> None:
    # plateau only -> constant plateau.
    assert PinBuRatioCurve(None, None, 1.06, 80.0, None).ratio_at(60.0) == 1.06
    # nothing harvested -> library default.
    bare = PinBuRatioCurve(None, None, None, None, None)
    assert not bare.harvested
    assert bare.ratio_at(60.0) == DEFAULT_RATIO


def test_core_discharge_estimate_energy_balance() -> None:
    # P*cyclen/HM * (241/feed) / 1000, in GWd/tU.
    got = core_discharge_estimate(650.0, 121, power_mw=3983.0, hm_mtu=104.8)
    exp = 3983.0 * 650.0 / 104.8 * (241.0 / 121.0) / 1000.0
    assert got == pytest.approx(exp)


def test_estimator_estimate_chain(fuel_lib) -> None:
    est = PinBuPhysicsEstimator(
        fuel_lib, library_id="ga80", a=1.12, b=-0.3, global_k_peak=1.45,
        k_peak_by_feed={117: 1.47}, power_mw=3983.0, hm_mtu=104.8,
    )
    bf = _some_ga80_feed(fuel_lib)
    val = est.estimate(bf, 117, 650.0)
    # recompute the chain independently.
    b_core = core_discharge_estimate(650.0, 117, power_mw=3983.0, hm_mtu=104.8)
    b_asm = 1.47 * b_core
    curve = resolve_peak_curve(fuel_lib, "ga80", bf)
    assert val == pytest.approx(1.12 * curve.pin_bu(b_asm) - 0.3)
    # feed with no fitted k_peak -> global fallback.
    assert est.k_peak(999) == 1.45
    # non-finite cyclen -> NaN (caller keeps raw head).
    assert math.isnan(est.estimate(bf, 117, float("nan")))


def test_resolve_peak_curve_picks_max_ratio() -> None:
    lib = _synthetic_two_type_library()
    curve = resolve_peak_curve(lib, "syn", {"HI": 30, "LO": 30})
    # HI has the larger plateau -> it is chosen (conservative peak).
    assert curve.ratio_asym == pytest.approx(1.09)


# --------------------------------------------------------------------------- #
# no-leakage fit
# --------------------------------------------------------------------------- #
def test_fit_row_mask_excludes_val_nonP_nonconverged() -> None:
    df = pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "e", "f"],
        "dataset":   ["P", "P", "P", "A", "P", "P"],
        "converged": [True, True, False, True, True, True],
        "library_id": ["ga80"] * 5 + ["260624"],
        "max_pin_burnup":      [80.0, 82.0, 81.0, 79.0, 83.0, 84.0],
        "max_assembly_burnup": [70.0, 71.0, 70.5, 69.0, 72.0, 73.0],
        "cyclen":              [640.0, 650.0, 645.0, 630.0, 655.0, 660.0],
    })
    train_ids = {"a", "b", "c", "d", "e"}       # 'f' is a holdout
    mask = fit_row_mask(df, train_ids, library_id="ga80")
    got = set(df["record_id"][mask])
    # 'a','b' pass; 'c' non-converged; 'd' dataset A; 'e' passes; 'f' not train.
    assert got == {"a", "b", "e"}


def test_fit_pinbu_physics_no_leakage_and_sane(tmp_path) -> None:
    champ = _real_champion()
    if champ is None:
        pytest.skip("champion + splits + store not present")
    model_dir, split = champ
    art = fit_pinbu_physics(model_dir, STORE, STORE.parent / "splits",
                            split=split, library_id="ga80", write=False)
    assert art["schema"].startswith("pinbu_physics")
    assert art["surrogate_col"] == PINBU_SURROGATE_COL
    assert art["n_fit"] >= 100
    # the affine recovers the ~2-D->3-D definition scale gap and cuts the error.
    assert 1.0 <= art["a"] <= 1.4
    assert art["mae_after_gwd"] < art["mae_before_gwd"]
    assert art["mae_after_gwd"] < 3.0
    assert 1.2 <= art["global_k_peak"] <= 1.7
    # explicit leakage assertion: recompute the fit row-ids and intersect val.
    from lpopt.model.splits import SplitManifest
    val_ids = set(SplitManifest.from_json(
        STORE.parent / "splits" / f"{split}.json").record_ids("val"))
    reader = _store_records()
    mask = fit_row_mask(reader, set(SplitManifest.from_json(
        STORE.parent / "splits" / f"{split}.json").record_ids("train")),
        library_id="ga80")
    used = set(reader["record_id"][mask].astype(str))
    assert not (used & val_ids)


def test_fit_pinbu_physics_json_roundtrip(tmp_path) -> None:
    champ = _real_champion()
    if champ is None:
        pytest.skip("champion + splits + store not present")
    _model_dir, split = champ
    # write into a SCRATCH dir (split is explicit, so model_dir is only the write
    # target) — never touch the live champion dir.
    scratch = tmp_path / "ens"
    scratch.mkdir()
    art = fit_pinbu_physics(scratch, STORE, STORE.parent / "splits",
                            split=split, library_id="ga80", write=True)
    p = scratch / PINBU_PHYSICS_NAME
    assert p.is_file()
    loaded = load_pinbu_physics(p)
    assert loaded["a"] == pytest.approx(art["a"])
    assert loaded["global_k_peak"] == pytest.approx(art["global_k_peak"])


# --------------------------------------------------------------------------- #
# serve-side application / disable (raw head preserved, other columns untouched)
# --------------------------------------------------------------------------- #
def _hand_artifact() -> dict:
    return {
        "schema": "pinbu_physics_affine_v1",
        "library_id": "ga80",
        "a": 1.12, "b": -0.3,
        "global_k_peak": 1.45,
        "k_peak_by_feed": {"117": {"k_peak": 1.47, "n": 100}},
        "power_mw": 3983.0, "hm_mtu": 104.8,
    }


def test_serve_pinbu_override_and_disable(tmp_path, backend_and_row) -> None:
    backend, pats, case = backend_and_row
    art = _hand_artifact()

    backend.set_pinbu_physics(None)              # ensure off
    raw = backend.predict(pats, case).mean.copy()

    backend.set_pinbu_physics(art, enabled=True)
    on = backend.predict(pats, case).mean.copy()

    backend.set_pinbu_physics(art, enabled=False)   # loaded but disabled
    off = backend.predict(pats, case).mean.copy()

    # (1) enabling changes the pin column; disabling restores the raw head exactly.
    assert not np.allclose(on[:, PINBU_SURROGATE_COL], raw[:, PINBU_SURROGATE_COL])
    assert np.array_equal(off[:, PINBU_SURROGATE_COL], raw[:, PINBU_SURROGATE_COL])

    # (2) EVERY other surrogate column is byte-identical whether enabled or not
    # (col 5 max_assembly_burnup stays NaN in both -> equal_nan).
    other = [c for c in range(on.shape[1]) if c != PINBU_SURROGATE_COL]
    assert np.array_equal(on[:, other], raw[:, other], equal_nan=True)

    # (3) the override equals the estimator's own value from the served cyclen.
    est = PinBuPhysicsEstimator.from_artifact(art, backend.fuel)
    for i, pat in enumerate(pats):
        want = est.estimate(pat.batch_feed(), int(case.feed),
                            float(on[i, 3]))       # served cyclen (col 3)
        if math.isfinite(want):
            assert on[i, PINBU_SURROGATE_COL] == pytest.approx(want)


def test_serve_no_artifact_is_noop(backend_and_row) -> None:
    backend, pats, case = backend_and_row
    backend.set_pinbu_physics(None)
    assert backend._pinbu is None
    a = backend.predict(pats, case).mean
    b = backend.predict(pats, case).mean
    assert np.array_equal(a, b, equal_nan=True)  # deterministic, no override


def test_from_dir_loads_pinbu_artifact(tmp_path, store_present) -> None:
    from lpopt.model.model_api import PosValCnnBackend

    ens = _make_min_ensemble(tmp_path)
    # drop a hand artifact into the ensemble dir; from_dir must pick it up + enable.
    import json
    (ens / PINBU_PHYSICS_NAME).write_text(json.dumps(_hand_artifact()))
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert backend._pinbu is not None
    assert backend.apply_pinbu_physics is True


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fuel_lib() -> FuelLibrary:
    p = STORE / "fuel_types.parquet"
    if not p.is_file():
        pytest.skip("fuel store not present")
    return FuelLibrary.from_parquet(p)


@pytest.fixture()
def store_present():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    return True


@pytest.fixture()
def backend_and_row(tmp_path):
    pytest.importorskip("torch")
    if not (STORE / "records.parquet").is_file() or \
            not (STORE / "fuel_types.parquet").is_file():
        pytest.skip("store not present")
    from lpopt.data.store import StoreReader
    from lpopt.data.schema import unpack_pattern
    from lpopt.model.model_api import PosValCnnBackend
    from lpopt.vendor.masterrl.domain import CaseKey

    ens = _make_min_ensemble(tmp_path)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    reader = StoreReader(STORE)
    df = reader.records
    sub = df[(df["dataset"] == "P") & (df["library_id"] == "ga80") &
             (df["feed"] == 117)]
    if sub.empty:
        sub = df[df["library_id"] == "ga80"]
    if sub.empty:
        pytest.skip("no ga80 rows to serve")
    rows = sub.head(5)
    pats = [unpack_pattern(str(p)) for p in rows["pattern"]]
    case = CaseKey(pair=str(rows.iloc[0]["case_pair"]), feed=int(rows.iloc[0]["feed"]))
    return backend, pats, case


def _make_min_ensemble(tmp: Path) -> Path:
    """Tiny cond_v3 7-target ensemble (predicts max_pin_burnup at col 6)."""
    import torch
    from lpopt.model.dataset_torch import TARGETS
    from lpopt.model.net import PosValNet, PosValNetConfig
    from lpopt.model.train import save_member

    ens = tmp / "ens_pinbu"
    cfg = PosValNetConfig()
    zmean = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0]
    zstd = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0]
    for i in range(2):
        seed = 700 + i
        net = PosValNet(cfg)
        meta = {
            "net_config": cfg.__dict__,
            "cond_schema": "v3",
            "target_names": list(TARGETS),
            "target_zscore": {"mean": zmean, "std": zstd},
            "seed": seed,
            "versions": {"torch": torch.__version__},
        }
        save_member(ens / f"member_{seed}", net, meta)
    return ens


def _some_ga80_feed(fuel: FuelLibrary) -> dict[str, int]:
    types = fuel.types("ga80")
    # two harvested ga80 types (finite pin_bu_ratio_asym) for a real curve.
    picked = [t for t in types
              if fuel.get(t, "ga80").pin_bu_ratio_asym is not None][:2]
    if not picked:
        picked = types[:2]
    return {picked[0]: 40, picked[-1]: 40}


def _synthetic_two_type_library() -> FuelLibrary:
    from lpopt.data.fuel_types import SCHEMA_COLUMNS, FuelVec, _rows_to_frame

    hi = FuelVec(library_id="syn", type_id="HI",
                 pin_bu_r_inf=1.05, pin_bu_paramA=0.7, pin_bu_ratio_asym=1.09,
                 pin_bu_bu_max=80.0, ff_pin_max=1.2)
    lo = FuelVec(library_id="syn", type_id="LO",
                 pin_bu_r_inf=1.03, pin_bu_paramA=0.6, pin_bu_ratio_asym=1.05,
                 pin_bu_bu_max=80.0, ff_pin_max=1.15)
    return FuelLibrary(_rows_to_frame([hi, lo]))


def _store_records():
    from lpopt.data.store import StoreReader
    return StoreReader(STORE).records


def _real_champion():
    """(model_dir, split) of a real champion with member metas + splits, or None."""
    models = STORE.parent / "models"
    splits = STORE.parent / "splits"
    if not models.is_dir() or not splits.is_dir() or \
            not (STORE / "records.parquet").is_file():
        return None
    for d in sorted(models.glob("*"), reverse=True):
        metas = sorted(d.glob("member_*/meta.json"))
        if not metas:
            continue
        import json
        split = str(json.loads(metas[0].read_text(encoding="utf-8")).get("split", "S1"))
        if (splits / f"{split}.json").is_file():
            return d, split
    return None
