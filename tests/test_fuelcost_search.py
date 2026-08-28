"""Outer min_fuel_cost cell-race: pure pre-ranking / FE-dominance + orchestrator."""

from __future__ import annotations

import os

import pytest

from lpopt.search import fuelcost_search as fcs
from lpopt.search.fuelcost_search import FuelCostCell


def _cell(pair, feed, e_core, *, band_ok=None, store_n=0, cy_med=None):
    return FuelCostCell(
        pair=pair, type_a=pair.split("_")[0], type_b=pair.split("_")[-1],
        feed=feed, e_core=e_core, fe_prior=fcs.cell_fe_prior(feed, e_core),
        store_n=store_n, store_cyclen_median=cy_med, band_ok=band_ok,
    )


# --------------------------------------------------------------------------- #
# feed grid + FE prior
# --------------------------------------------------------------------------- #
def test_feed_grid_is_1_plus_4n():
    assert fcs.feed_grid(101, 141) == [101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141]
    assert all((f - 1) % 4 == 0 for f in fcs.feed_grid(1, 200))
    assert fcs.feed_grid(102, 104) == []            # nothing on-grid


def test_cell_fe_prior_is_feed_times_e_core():
    assert fcs.cell_fe_prior(121, 5.2) == pytest.approx(629.2)
    assert fcs.cell_fe_prior(117, 5.0) == pytest.approx(585.0)
    # lower feed AND lower e_core -> strictly lower FE (the economics trade).
    assert fcs.cell_fe_prior(117, 5.0) < fcs.cell_fe_prior(121, 5.2)
    # mass-bearing scale is a constant factor (still monotone).
    assert fcs.cell_fe_prior(117, 5.0, 138.0) < fcs.cell_fe_prior(121, 5.2, 138.0)


# --------------------------------------------------------------------------- #
# pre-ranking: band feasibility dominates, FE orders within a class
# --------------------------------------------------------------------------- #
def test_prerank_band_feasible_first_then_fe():
    cells = [
        _cell("A_A", 121, 5.4, band_ok=True),       # FE 653.4, in-band
        _cell("B_B", 117, 5.0, band_ok=False),      # FE 585.0 but OUT of band
        _cell("C_C", 117, 5.2, band_ok=True),       # FE 608.4, in-band  <- best
        _cell("D_D", 121, 5.1, band_ok=None),       # FE 617.1, unknown band
    ]
    ranked = fcs.prerank_cells(cells)
    assert ranked[0].pair == "C_C"                  # in-band + lowest FE
    assert ranked[1].pair == "A_A"                  # in-band, higher FE
    assert ranked[2].pair == "D_D"                  # unknown band next
    assert ranked[-1].pair == "B_B"                 # out-of-band last despite low FE


def test_prerank_fe_orders_within_same_band_class():
    cells = [
        _cell("H_H", 125, 5.2, band_ok=True),       # FE 650
        _cell("L_L", 117, 5.2, band_ok=True),       # FE 608.4 <- lower
    ]
    ranked = fcs.prerank_cells(cells)
    assert ranked[0].pair == "L_L" and ranked[1].pair == "H_H"


# --------------------------------------------------------------------------- #
# FE-dominance elimination (deterministic)
# --------------------------------------------------------------------------- #
def test_eliminate_dominated_by_proven_fe():
    cells = [_cell("A_A", 117, 5.0), _cell("B_B", 117, 5.2), _cell("C_C", 121, 5.2)]
    # nothing proven yet -> nothing eliminated.
    assert fcs.eliminate_dominated(cells, None) == 0
    # prove feasible at FE 608.4 (117x5.2): the 629.2 cell is dominated, 585 is not.
    n = fcs.eliminate_dominated(cells, 608.4)
    assert n == 2                                    # B_B (==608.4) and C_C (629.2)
    assert cells[0].eliminated is False              # A_A 585 < 608.4 survives
    assert cells[1].eliminated and cells[2].eliminated
    # idempotent: re-running does not double count.
    assert fcs.eliminate_dominated(cells, 608.4) == 0


def test_dedup_by_composition_keeps_most_evidenced_representative():
    cells = [
        _cell("J1_L1", 117, 5.2, store_n=10),
        _cell("J1_L2", 117, 5.2, store_n=465),       # most evidence -> representative
        _cell("J1_L3", 117, 5.2, store_n=3),
        _cell("A1_A1", 117, 5.0, store_n=0),          # different e_core -> own cell
    ]
    dd = fcs.dedup_by_composition(cells)
    assert len(dd) == 2                               # (117,5.2) collapses to 1, (117,5.0) own
    rep = next(c for c in dd if abs(c.e_core - 5.2) < 1e-6)
    assert rep.pair == "J1_L2" and rep.store_n == 465


# --------------------------------------------------------------------------- #
# race allocation
# --------------------------------------------------------------------------- #
def test_race_allocation_floors_best_and_spreads_remainder():
    cells = [_cell("A_A", 117, 5.0), _cell("B_B", 117, 5.2), _cell("C_C", 121, 5.2)]
    alloc = fcs.race_allocation(cells, 8)
    assert sum(alloc.values()) == 8
    assert alloc[cells[0].cell_id] >= 1              # lowest FE gets an exploit floor
    # eliminated cells receive nothing.
    cells[1].eliminated = True
    alloc2 = fcs.race_allocation(cells, 8)
    assert cells[1].cell_id not in alloc2 or alloc2.get(cells[1].cell_id, 0) == 0
    assert fcs.race_allocation(cells, 0) == {}


# --------------------------------------------------------------------------- #
# enumeration against the real fuel table + store (integration)
# --------------------------------------------------------------------------- #
def test_enumerate_cells_surfaces_low_band_low_feed_region():
    fp = "data/store/fuel_types.parquet"
    if not os.path.exists(fp):
        pytest.skip("fuel_types.parquet not present")
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    fuel = FuelLibrary.from_parquet(fp)
    store = StoreReader("data/store").records
    cells = fcs.enumerate_cells(
        fuel, "ga80", [5.0, 5.125, 5.25], 0.125, [117, 121, 125], store_df=store)
    assert cells, "expected non-empty cell set"
    ranked = fcs.prerank_cells(fcs.dedup_by_composition(cells))
    top = ranked[0]
    # FE-optimal region is a low-feed, low-e_core, band-feasible cell.
    assert top.feed <= 121
    assert top.fe_prior == pytest.approx(top.feed * top.e_core, rel=1e-6)
    # the champion low-band f117 region should be present and band-feasible.
    assert any(c.feed == 117 and c.band_ok is True for c in ranked)


# --------------------------------------------------------------------------- #
# orchestrator convergence with a stub driver (no MASTER)
# --------------------------------------------------------------------------- #
def test_outer_search_converges_to_min_fe_feasible_cell(tmp_path):
    fp = "data/store/fuel_types.parquet"
    if not os.path.exists(fp):
        pytest.skip("fuel_types.parquet not present")
    import numpy as np
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    from lpopt.vendor.masterrl.surrogate import SurrogatePrediction

    fuel = FuelLibrary.from_parquet(fp)
    store = StoreReader("data/store").records
    cfg = load_config("lpopt.inp")

    class StubModel:
        def predict(self, pats, case, cell=0.0):
            n = len(list(pats))
            m = np.full((n, 7), np.nan)
            m[:, :5] = np.tile([1.50, 1400.0, 2.30, 625.0, 0.20], (n, 1))
            m[:, 6] = 68.0
            z = np.zeros((n, 7)) + 0.5
            return SurrogatePrediction(m, z.copy(), z.copy())

    class _Res:
        def __init__(self, best, spent):
            self.best = best
            self.budget_spent = spent

    def factory(cell, budget, run_dir):
        class D:
            def run(_s):
                # only the globally-cheapest cell (feed 117, e≈5.0) is feasible.
                if cell.feed == 117 and cell.e_core <= 5.201:
                    return _Res({"fuel_cost": cell.fe_prior, "f_r": 1.49,
                                 "cyclen": 620.0, "feasible": True}, budget)
                return _Res(None, budget)
        return D()

    search = fcs.FuelCostOuterSearch(
        cfg, StubModel(), e_core_targets=[5.0, 5.125, 5.25], feeds=[117, 121, 125],
        e_core_tol=0.125, screen_top_k=8, mini_wave=8, total_budget=40,
        fuel_library=fuel, store_df=store, driver_factory=factory, log=lambda m: None)
    res = search.run(tmp_path)
    assert res.best is not None
    assert res.best_cell.endswith("_f117")           # converged on a feed-117 cell
    assert res.best_fe == pytest.approx(res.best["fuel_cost"])
    # every cell is FE-eliminated once the global-min-FE feasible LP is proven.
    assert res.cells_eliminated >= res.cells_raced
    assert res.budget_spent <= 40


def test_race_no_phantom_budget_and_deepens(tmp_path):
    """The race must charge REAL calls only and terminate — never burn phantom
    budget on no-op resumes of a completed mini-campaign (forensic 20260721)."""
    import types
    import numpy as np
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.store import StoreReader
    from lpopt.vendor.masterrl.surrogate import SurrogatePrediction

    fp = "data/store/fuel_types.parquet"
    if not os.path.exists(fp):
        pytest.skip("fuel_types.parquet not present")
    fuel = FuelLibrary.from_parquet(fp)
    store = StoreReader("data/store").records
    cfg = load_config("lpopt.inp")

    class StubModel:
        def predict(self, pats, case, cell=0.0):
            n = len(list(pats))
            m = np.full((n, 7), np.nan)
            m[:, :5] = np.tile([1.60, 1400.0, 2.30, 625.0, 0.20], (n, 1))
            m[:, 6] = 68.0
            z = np.zeros((n, 7)) + 0.5
            return SurrogatePrediction(m, z.copy(), z.copy())

    # Each cell early-stops after 2 real calls: its resumable run's CUMULATIVE
    # budget_spent is capped at 2 no matter how much budget is granted (a no-op
    # resume once deepened past 2).  No cell is ever feasible (best None), so
    # FE-dominance never fires — only real-delta accounting + exhaustion can end it.
    CAP = 2
    grants: dict[str, list[int]] = {}

    def factory(cell, granted, run_dir):
        grants.setdefault(cell.cell_id, []).append(granted)

        class D:
            def run(_s):
                return types.SimpleNamespace(best=None, budget_spent=min(granted, CAP))
        return D()

    search = fcs.FuelCostOuterSearch(
        cfg, StubModel(), e_core_targets=[5.0], feeds=[117], e_core_tol=0.125,
        screen_top_k=4, mini_wave=8, total_budget=100,
        fuel_library=fuel, store_df=store, restart_pairs=None,
        driver_factory=factory, log=lambda m: None)
    n_cells = len(search.build_cells())
    res = search.run(tmp_path)
    # Real spend is bounded by CAP per cell — NOT the 100 budget (no phantom slots).
    assert res.budget_spent == n_cells * CAP
    assert res.budget_spent < 100                       # terminated, no phantom burn
    # Every cell was granted budget MORE THAN ONCE (deepening was attempted) and
    # then eliminated as exhausted (delta 0 on the extended grant).
    assert all(len(v) >= 2 for v in grants.values())
    assert res.cells_eliminated == n_cells


def test_retrain_trigger_fires_and_gated_promotion(tmp_path):
    """The AL retrain trigger fires on a round boundary; the champion is swapped in
    ONLY when the injected gate reports pass, and the gate table is recorded."""
    import types

    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(library_id="ga80"),
        acquisition=types.SimpleNamespace(
            fuelcost_cyclen_lo=615.0, fuelcost_cyclen_hi=635.0),
        flow=types.SimpleNamespace(random_seed=0),
    )
    search = fcs.FuelCostOuterSearch(
        cfg, model="M0", e_core_targets=[5.0], feeds=[117],
        fuel_library=None, store_df=None, driver_factory=lambda *a: None,
        log=lambda m: None)

    # -- gate FAILS: champion must NOT advance -------------------------------- #
    fails = {"pass": False, "worst_drop": 0.20, "epsilon": 0.10, "checks": [{"cell": "c"}],
             "champion_model_dir": "data/models/NEW"}
    search.retrain_gate_callback = lambda n: fails
    search.model_reload = lambda d: "M_NEW"
    search._new_labels_since_retrain = 250
    search._maybe_retrain(round_finished=True)
    assert search.model == "M0"                        # gate failed -> no promotion
    assert search.retrain_events[-1]["pass"] is False
    assert search._new_labels_since_retrain == 0       # counter reset after trigger

    # -- gate PASSES: champion promoted via the gate path --------------------- #
    passes = {"pass": True, "worst_drop": 0.01, "epsilon": 0.10, "checks": [{"cell": "c"}],
              "champion_model_dir": "data/models/NEW"}
    search.retrain_gate_callback = lambda n: passes
    search._new_labels_since_retrain = 250
    search._maybe_retrain(round_finished=True)
    assert search.model == "M_NEW"                     # gate passed -> promoted
    assert search.retrain_events[-1]["pass"] is True

    # -- no callback / no new labels: trigger is a no-op ---------------------- #
    search.retrain_gate_callback = None
    search._new_labels_since_retrain = 999
    search._maybe_retrain(round_finished=True)         # must not raise
    search.retrain_gate_callback = lambda n: passes
    search._new_labels_since_retrain = 0
    before = len(search.retrain_events)
    search._maybe_retrain(round_finished=True)         # zero labels -> skip
    assert len(search.retrain_events) == before
