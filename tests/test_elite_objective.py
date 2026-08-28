"""Objective-aware ELITE PARENT selection for the exploit arms.

The defect (measured 2026-07-31): ``ProduceDriver._elites_for`` selected the
``elite_perturb`` parents as the top-32 converged rows of the pair by ``cyclen``
DESCENDING — pair-scoped, but neither feed- nor objective-scoped.  For E1_E2 the
resulting elite set was 100% feed-133/141 high-cyclen rows, while EVERY
flat_power campaign winner (node_peak ~1.23-1.28 at cyclen ~632, feed 121) fell
BELOW the cyclen cut.  The generator that is supposed to exploit the frontier
therefore never once perturbed the flattest cores.  A second, quieter half of the
same defect: when no parent qualified, the draw degraded to ``random`` in total
silence (ledger ``generator='random'`` is truthful, but nothing said the
``elite_perturb`` WEIGHT had been spent on a random draw).

This module pins:

1. the LEGACY default (``elite_objective`` unset -> ``"cyclen"``) is byte-identical
   to the pre-fix selection — an existing cycle-length campaign cannot move;
2. ``"flat"`` orders by ``node_peak`` ascending with ``map_cov`` as the tie-break,
   excludes rows with NO ``node_peak`` label, prefers same-feed parents, and never
   crosses to another pair;
3. ``"flat_feasible"`` additionally applies the ``flat_power`` CBC / F_q / |AO|
   gates from ``campaign.feasibility_limits_for``;
4. the elite -> random degradation is LOUD (warned once per stratum) and COUNTED
   in the summary;
5. the campaign-side elite arm (``CampaignDriver._store_elites``, which feeds
   ``construct.build_pool``) ranks by the CAMPAIGN objective — flatness under
   ``flat_power`` — and not by cyclen;
6. that arm's FEASIBLE-FIRST tier survives a parquet round-trip: a missing
   ``max_pin_burnup`` arrives as ``NaN``, and ``NaN == missing == None`` is the
   deliberate contract (decision 2026-07-31) — before it, every store row was
   judged infeasible under ``flat_power`` and the tier silently vanished.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from lpopt.config import (
    AcquisitionConfig, CaseConfig, DataConfig, ExtractConfig, FlowConfig, FuelConfig,
    LpoptConfig, MasterConfig, ModelConfig, ProduceConfig, RemoteConfig, SearchConfig,
    StratumConfig, VerifyConfig,
)
from lpopt.data.schema import (
    SYM_CLASS, CanonicalRecord, compute_record_id, pack_pattern, unpack_pattern,
)
from lpopt.data.store import StoreReader, StoreWriter
from lpopt.search.genome import random_genome
from lpopt.search.produce import (
    ELITE_OBJECTIVES, ProduceDriver, _StratumState, _normalize_mix,
)
from lpopt.search.verify import PRODUCE_DECK_KNOBS

PAIR = "K1_K2"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_package(tmp_path: Path) -> Path:
    """A minimal MASTER package (same shape as tests/test_produce_ledger.py)."""

    pkg = tmp_path / "pkg"
    base = pkg / "bases" / PAIR
    base.mkdir(parents=True)
    (base / "MAS_RST.NATIVE.01").write_bytes(b"rst")
    core = pkg / "cores" / PAIR / "s1"
    core.mkdir(parents=True)
    (core / "MAS_INP_cy01.inp").write_text(
        "dummy\n%LPD_SHF\n F K1  0,\n%END\n", encoding="utf-8"
    )
    return pkg


def _record(
    seed: int,
    *,
    pair: str = PAIR,
    feed: int = 121,
    cyclen: float = 620.0,
    node_peak: float | None = None,
    map_cov: float | None = None,
    cbc: float = 1300.0,
    f_q: float = 2.10,
    ao: float = 0.12,
    #: ``None`` round-trips through parquet as ``NaN``; since the 2026-07-31
    #: NaN==missing==None decision that is MISSING (passes the pin-BU gate), which
    #: the tests at the bottom of this module pin explicitly.
    pin_bu: float | None = 40.0,
    converged: bool = True,
) -> CanonicalRecord:
    """One converged store row with controllable cyclen / flatness / gates."""

    n_fresh = (feed - 1) // 4
    pattern = random_genome(random.Random(seed), pair, n_fresh).to_pattern()
    rid = compute_record_id(pattern.canonical(), "ga80", pair, PRODUCE_DECK_KNOBS)
    return CanonicalRecord(
        record_id=rid, dataset="P", campaign="t", stratum="s", generator="random",
        parent_record_id=None, case_pair=pair, feed=feed, n_batches=2, depth2_edges=0,
        e_core=5.2, e_split=0.0, library_id="ga80", sym_class=SYM_CLASS,
        pattern=pack_pattern(pattern),
        f_r=1.60, f_q=f_q, cbc_max=cbc, cbc_boc=None, cbc_kind="max",
        cyclen=cyclen, ao_abs=ao, cycle_burnup=None, discharge_burnup=None,
        max_assembly_burnup=None, max_pin_burnup=pin_bu, eoc_ppm=None, delta_efpd=None,
        n_cycles=12.0, converged=converged, converged_at_cap=False,
        tolerance_margin=0.2, restart_provenance="native:MAS_RST.NATIVE.01",
        valid=True, failure="", maps_key=None,
        node_peak=node_peak, map_cov=map_cov,
    )


def _cfg(
    tmp_path: Path, pkg: Path, strata: list[StratumConfig], **produce_kw
) -> LpoptConfig:
    return LpoptConfig(
        flow=FlowConfig(random_seed=3), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(package_root=str(pkg)), data=DataConfig(),
        case=CaseConfig(), fuel=FuelConfig(), extract=ExtractConfig(),
        produce=ProduceConfig(
            campaign="test", workers=4, template_fallbacks=[], strata=strata,
            **produce_kw,
        ),
        source_path=tmp_path / "lpopt.inp",
    )


def _driver(tmp_path: Path, cfg: LpoptConfig, tag: str = "d", **kw) -> ProduceDriver:
    return ProduceDriver(
        cfg, dry_run=True, run_dir=tmp_path / f"run_{tag}",
        store_dir=tmp_path / "store", ledger_path=tmp_path / "ledger.jsonl",
        progress=False, **kw,
    )


def _seed_store(tmp_path: Path, records) -> None:
    StoreWriter(tmp_path / "store").write_records(list(records))


def _stratum(name: str = "s", **kw) -> StratumConfig:
    base = dict(pairs=[PAIR], feed=121, n_target=4, generators={"random": 1.0},
                priority=100)
    base.update(kw)
    return StratumConfig(name=name, **base)


def _ids(elites) -> list[str]:
    return [rid for rid, _ in elites]


# --------------------------------------------------------------------------- #
# 1) LEGACY DEFAULT: byte-identical to the pre-fix cyclen-descending selection
# --------------------------------------------------------------------------- #
def _legacy_reference(store_dir: Path, pair: str) -> list[str]:
    """The PRE-FIX ``_elites_for`` body, verbatim, as an oracle."""

    df = StoreReader(store_dir).records
    conv = df[df["converged"] == True]  # noqa: E712
    same = conv[conv["case_pair"] == pair]
    pool = same if len(same) else conv
    out: list[str] = []
    if len(pool):
        pool = pool.sort_values("cyclen", ascending=False).head(32)
        for _, row in pool.iterrows():
            try:
                unpack_pattern(str(row["pattern"]))
            except (ValueError, KeyError):
                continue
            out.append(str(row["record_id"]))
    return out


def test_default_elite_objective_is_legacy_cyclen(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    cfg = _cfg(tmp_path, pkg, [_stratum()])
    drv = _driver(tmp_path, cfg)
    # unset [produce] + unset per-stratum -> "cyclen"
    assert drv._elite_objective(None) == "cyclen"
    assert drv._elite_objective(cfg.produce.strata[0]) == "cyclen"
    assert ELITE_OBJECTIVES[0] == "cyclen"


def test_legacy_default_selection_is_byte_identical(tmp_path: Path) -> None:
    """With the knob unset, the elite ORDER is exactly the pre-fix order."""

    pkg = _make_package(tmp_path)
    # 40 rows (> the 32 cap) with interleaved cyclen and flatness so a flat-aware
    # rule would produce a visibly different order.
    records = [
        _record(i, cyclen=600.0 + (i * 7 % 40), node_peak=1.20 + (i % 11) * 0.01,
                map_cov=0.30 + (i % 5) * 0.01)
        for i in range(40)
    ]
    _seed_store(tmp_path, records)

    cfg = _cfg(tmp_path, pkg, [_stratum()])
    drv = _driver(tmp_path, cfg)
    got = _ids(drv._elites_for(PAIR, 121))
    assert got == _legacy_reference(tmp_path / "store", PAIR)
    assert len(got) == 32
    # and it really is cyclen-descending (the property that made it wrong here).
    df = StoreReader(tmp_path / "store").records.set_index("record_id")
    cyclens = [float(df.loc[r, "cyclen"]) for r in got]
    assert cyclens == sorted(cyclens, reverse=True)


def test_legacy_keeps_the_pair_wide_fallback(tmp_path: Path) -> None:
    """Legacy behaviour when the pair has no converged row: fall back pair-WIDE."""

    pkg = _make_package(tmp_path)
    _seed_store(tmp_path, [_record(i, pair="J1_N1", cyclen=610.0 + i) for i in range(3)])
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    assert len(drv._elites_for(PAIR, 121)) == 3        # other pair's rows (legacy)


# --------------------------------------------------------------------------- #
# 2) FLAT: node_peak ascending, map_cov tie-break, label required
# --------------------------------------------------------------------------- #
def test_flat_orders_flattest_first_with_map_cov_tiebreak(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    # The E1_E2 shape: the flattest rows have the LOWEST cyclen, so the legacy
    # rule ranks them last and the flat rule must rank them first.
    peaky_high_cyclen = _record(1, cyclen=690.0, node_peak=1.42, map_cov=0.40)
    flat_low_cyclen = _record(2, cyclen=632.0, node_peak=1.2285, map_cov=0.31)
    tie_worse_cov = _record(3, cyclen=650.0, node_peak=1.30, map_cov=0.36)
    tie_better_cov = _record(4, cyclen=655.0, node_peak=1.30, map_cov=0.33)
    _seed_store(tmp_path, [peaky_high_cyclen, flat_low_cyclen, tie_worse_cov,
                           tie_better_cov])

    cfg = _cfg(tmp_path, pkg, [_stratum(elite_objective="flat")])
    drv = _driver(tmp_path, cfg)
    assert drv._elite_objective(cfg.produce.strata[0]) == "flat"

    got = _ids(drv._elites_for(PAIR, 121, "flat"))
    assert got == [
        flat_low_cyclen.record_id,   # node_peak 1.2285 — THE frontier core
        tie_better_cov.record_id,    # 1.30, map_cov 0.33 wins the tie
        tie_worse_cov.record_id,     # 1.30, map_cov 0.36
        peaky_high_cyclen.record_id,  # 1.42 — top cyclen, last under flat
    ]
    # the legacy rule puts the frontier core LAST (this is the measured defect).
    assert _ids(drv._elites_for(PAIR, 121, "cyclen"))[-1] == flat_low_cyclen.record_id


def test_flat_excludes_rows_without_a_node_peak_label(tmp_path: Path) -> None:
    """No ``node_peak`` == NO objective value, which is not the same as a bad one."""

    pkg = _make_package(tmp_path)
    labelled = _record(1, cyclen=630.0, node_peak=1.31, map_cov=0.32)
    unlabelled = _record(2, cyclen=700.0, node_peak=None, map_cov=None)
    _seed_store(tmp_path, [labelled, unlabelled])

    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    assert _ids(drv._elites_for(PAIR, 121, "flat")) == [labelled.record_id]
    # ... and non-converged rows never qualify either.
    assert unlabelled.record_id in _ids(drv._elites_for(PAIR, 121, "cyclen"))


def test_flat_excludes_nonconverged(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    bad = _record(1, node_peak=1.10, map_cov=0.20, converged=False)
    good = _record(2, node_peak=1.35, map_cov=0.34)
    _seed_store(tmp_path, [bad, good])
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    assert _ids(drv._elites_for(PAIR, 121, "flat")) == [good.record_id]


# --------------------------------------------------------------------------- #
# 3) FLAT feed scoping + no cross-pair leakage
# --------------------------------------------------------------------------- #
def test_flat_prefers_same_feed_parents(tmp_path: Path) -> None:
    """The measured E1_E2 case: the flat winner is at f121, the legacy elites at
    feed 133/141.  A cross-feed parent is feed-morphed before ``mutate`` runs,
    which destroys the fresh placement that made it flat."""

    pkg = _make_package(tmp_path)
    flatter_wrong_feed = _record(1, feed=133, node_peak=1.20, map_cov=0.30)
    same_feed = _record(2, feed=121, node_peak=1.28, map_cov=0.33)
    _seed_store(tmp_path, [flatter_wrong_feed, same_feed])

    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    assert _ids(drv._elites_for(PAIR, 121, "flat")) == [same_feed.record_id]
    # the same store, asked for feed 133, selects the feed-133 row.
    assert _ids(drv._elites_for(PAIR, 133, "flat")) == [flatter_wrong_feed.record_id]


def test_flat_falls_back_pair_wide_across_feeds(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    other_feed = _record(1, feed=133, node_peak=1.24, map_cov=0.31)
    _seed_store(tmp_path, [other_feed])
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    # no feed-121 row exists -> the pair's other feeds are still usable parents.
    assert _ids(drv._elites_for(PAIR, 121, "flat")) == [other_feed.record_id]


def test_flat_never_crosses_to_another_pair(tmp_path: Path) -> None:
    """Legacy falls back to OTHER pairs; flat returns nothing (-> loud random)."""

    pkg = _make_package(tmp_path)
    _seed_store(tmp_path, [_record(i, pair="J1_N1", node_peak=1.2 + i * 0.01,
                                   map_cov=0.30) for i in range(3)])
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    assert drv._elites_for(PAIR, 121, "flat") == []
    assert len(drv._elites_for(PAIR, 121, "cyclen")) == 3     # legacy unchanged


# --------------------------------------------------------------------------- #
# 4) FLAT_FEASIBLE: the flat_power CBC / F_q / |AO| gates
# --------------------------------------------------------------------------- #
def test_flat_feasible_applies_the_flat_power_gates(tmp_path: Path) -> None:
    from lpopt.search.campaign import feasibility_limits_for

    pkg = _make_package(tmp_path)
    limits = feasibility_limits_for(AcquisitionConfig(), "flat_power")
    assert limits["cbc_max"] == 1550.0 and limits["f_q"] == 2.41
    assert limits["ao_abs"] == 0.30

    flattest_but_cbc = _record(1, node_peak=1.20, map_cov=0.30, cbc=1600.0)
    flat_but_fq = _record(2, node_peak=1.22, map_cov=0.30, f_q=2.50)
    flat_but_ao = _record(3, node_peak=1.23, map_cov=0.30, ao=0.35)
    feasible = _record(4, node_peak=1.29, map_cov=0.32)
    _seed_store(tmp_path, [flattest_but_cbc, flat_but_fq, flat_but_ao, feasible])

    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    # 'flat' keeps everything, flattest first ...
    assert _ids(drv._elites_for(PAIR, 121, "flat"))[0] == flattest_but_cbc.record_id
    # ... 'flat_feasible' keeps only the row that passes every gate.
    assert _ids(drv._elites_for(PAIR, 121, "flat_feasible")) == [feasible.record_id]


def test_flat_feasible_empty_when_nothing_passes(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    _seed_store(tmp_path, [_record(1, node_peak=1.20, map_cov=0.30, cbc=1600.0)])
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [_stratum()]))
    assert drv._elites_for(PAIR, 121, "flat_feasible") == []


# --------------------------------------------------------------------------- #
# 5) config knob resolution + validation
# --------------------------------------------------------------------------- #
def test_stratum_override_beats_the_produce_default(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    cfg = _cfg(
        tmp_path, pkg,
        [_stratum("inherits"), _stratum("overrides", elite_objective="cyclen")],
        elite_objective="flat_feasible",
    )
    drv = _driver(tmp_path, cfg)
    assert drv._elite_objective(cfg.produce.strata[0]) == "flat_feasible"
    assert drv._elite_objective(cfg.produce.strata[1]) == "cyclen"


def test_unknown_elite_objective_fails_at_construction(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    cfg = _cfg(tmp_path, pkg, [_stratum(elite_objective="flatt")])
    with pytest.raises(ValueError, match="elite_objective"):
        _driver(tmp_path, cfg)


def test_elite_objective_parses_from_a_deck(tmp_path: Path) -> None:
    """The exact deck lines a frontier campaign will use must parse cleanly."""

    from lpopt.config import load_config

    deck = tmp_path / "deck.inp"
    deck.write_text(
        "[produce]\n"
        'campaign = "frontier_flat"\n'
        'elite_objective = "flat_feasible"\n'
        "\n"
        "[[produce.strata]]\n"
        'name = "e1e2_f121_flat"\n'
        'pairs = ["E1_E2"]\n'
        "feed = 121\n"
        'elite_objective = "flat_feasible"\n'
        "generators = { elite_perturb = 0.6, heuristic = 0.25, random = 0.15 }\n"
        "n_target = 48\n",
        encoding="utf-8",
    )
    cfg = load_config(deck)
    assert cfg.produce.elite_objective == "flat_feasible"
    assert cfg.produce.strata[0].elite_objective == "flat_feasible"


# --------------------------------------------------------------------------- #
# 6) LOUD FALLBACK: elite_perturb -> random is warned once + counted
# --------------------------------------------------------------------------- #
def _elite_state(strat: StratumConfig) -> _StratumState:
    state = _StratumState(cfg=strat)
    state.effective_n_target = int(strat.n_target)
    state.effective_generators = _normalize_mix(strat.generators)
    return state


def test_elite_fallback_is_loud_and_counted(tmp_path: Path) -> None:
    """Empty store + an elite_perturb-only stratum: every draw degrades."""

    pkg = _make_package(tmp_path)
    strat = _stratum("elite_only", generators={"elite_perturb": 1.0}, n_target=8)
    logs: list[str] = []
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [strat]), log=logs.append)

    state = _elite_state(strat)
    rng = random.Random(11)
    gens = [drv._generate(state, rng) for _ in range(6)]
    assert all(g is not None for g in gens)
    # the ledger's generator field stays TRUTHFUL: these ARE random draws.
    assert {g[1] for g in gens} == {"random"}
    # ... and the degradation is now visible.
    assert state.elite_fallback_random == 6
    warnings = [m for m in logs if "elite_perturb DEGRADED to random" in m]
    assert len(warnings) == 1, "the warning must fire exactly ONCE per stratum"
    assert "no store row qualified as an elite parent" in warnings[0]
    assert "elite_objective='cyclen'" in warnings[0]


def test_elite_fallback_fires_for_a_flat_stratum_with_unlabelled_store(
    tmp_path: Path,
) -> None:
    """A store full of converged-but-unlabelled rows starves a FLAT elite arm —
    the case that used to look like a healthy elite_perturb campaign."""

    pkg = _make_package(tmp_path)
    _seed_store(tmp_path, [_record(i, cyclen=640.0 + i) for i in range(5)])
    strat = _stratum("flat_only", generators={"elite_perturb": 1.0},
                     elite_objective="flat")
    logs: list[str] = []
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [strat]), log=logs.append)

    state = _elite_state(strat)
    rng = random.Random(5)
    for _ in range(4):
        drv._generate(state, rng)
    assert state.elite_fallback_random == 4
    assert any("elite_objective='flat'" in m for m in logs)
    # the same store DOES feed the legacy arm — proof the starvation is a
    # property of the objective, not of the store being empty.
    assert len(drv._elites_for(PAIR, 121, "cyclen")) == 5


def test_no_fallback_warning_when_elites_exist(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    _seed_store(tmp_path, [_record(i, cyclen=640.0 + i) for i in range(6)])
    strat = _stratum("ok", generators={"elite_perturb": 1.0})
    logs: list[str] = []
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [strat]), log=logs.append)

    state = _elite_state(strat)
    rng = random.Random(7)
    gens = [drv._generate(state, rng) for _ in range(8)]
    assert any(g is not None and g[1] == "elite_perturb" for g in gens)
    assert not [m for m in logs if "DEGRADED" in m]


def test_flat_arm_draws_its_parents_from_the_flat_pool(tmp_path: Path) -> None:
    """End-to-end: with ``elite_objective="flat"`` every ``elite_perturb`` parent
    is a node_peak-labelled same-feed row — the high-cyclen unlabelled rows that
    monopolised the legacy elite set are never parents."""

    pkg = _make_package(tmp_path)
    flat_rows = [
        _record(10 + i, feed=121, cyclen=630.0 + i,
                node_peak=1.23 + i * 0.01, map_cov=0.31)
        for i in range(3)
    ]
    # the legacy elite set: top cyclen, no flatness label, wrong feed.
    legacy_rows = [_record(50 + i, feed=133, cyclen=700.0 + i) for i in range(6)]
    _seed_store(tmp_path, flat_rows + legacy_rows)

    strat = _stratum("flat_arm", generators={"elite_perturb": 1.0},
                     elite_objective="flat")
    logs: list[str] = []
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [strat]), log=logs.append)

    flat_ids = set(_ids(drv._elites_for(PAIR, 121, "flat")))
    assert flat_ids == {r.record_id for r in flat_rows}

    state = _elite_state(strat)
    rng = random.Random(3)
    parents = set()
    for _ in range(20):
        drawn = drv._generate(state, rng)
        if drawn is not None and drawn[1] == "elite_perturb":
            parents.add(drawn[2])
    assert parents, "the flat elite arm must actually produce elite children"
    assert parents <= flat_ids
    assert not parents & {r.record_id for r in legacy_rows}
    assert state.elite_fallback_random == 0
    assert not [m for m in logs if "DEGRADED" in m]


def test_summary_reports_elite_fallback_random(tmp_path: Path) -> None:
    pkg = _make_package(tmp_path)
    strat = _stratum("elite_only", generators={"elite_perturb": 1.0}, n_target=4)
    logs: list[str] = []
    drv = _driver(tmp_path, _cfg(tmp_path, pkg, [strat]), log=logs.append)
    summary = drv.run()

    assert summary.elite_fallback_random > 0
    assert summary.strata[0]["elite_fallback_random"] == summary.elite_fallback_random
    printed = "\n".join(logs)
    assert f"elite_fallback_random={summary.elite_fallback_random}" in printed
    assert "ELITE->RND" in printed


# --------------------------------------------------------------------------- #
# 7) CAMPAIGN side: the build_pool elite arm ranks by the CAMPAIGN objective
# --------------------------------------------------------------------------- #
def _campaign_cfg(tmp_path: Path, store_dir: Path, objective: str) -> LpoptConfig:
    deck = tmp_path / "lpopt.inp"
    deck.write_text("# fake deck\n", encoding="utf-8")
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(harvest_maps=True), data=DataConfig(),
        case=CaseConfig(pair=PAIR, feed=121), fuel=FuelConfig(),
        extract=ExtractConfig(), produce=ProduceConfig(), search=SearchConfig(),
        acquisition=AcquisitionConfig(budget=8, objective=objective,
                                      gate_skill_halt=-2.0),
        model=ModelConfig(store_dir=str(store_dir)), source_path=deck,
    )


def _campaign_driver(tmp_path: Path, records, objective: str):
    from lpopt.search.campaign import CampaignDriver
    from lpopt.search.stub import StubEvaluator
    from test_campaign_stub import FakeModel

    store_dir = tmp_path / "main_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    StoreWriter(store_dir).write_records(list(records))
    cfg = _campaign_cfg(tmp_path, store_dir, objective)
    stub = StubEvaluator()
    return CampaignDriver(cfg, FakeModel(), lambda w, c: stub, dry_run=True,
                          run_dir=tmp_path / "crun", progress=False,
                          log=lambda m: None)


def test_campaign_flat_power_elites_rank_by_flatness_not_cyclen(tmp_path: Path) -> None:
    """``CampaignDriver._store_elites`` (the ``build_pool`` ``store_elites`` arm)
    must key off the CAMPAIGN objective — the produce-side defect must not exist
    here."""

    peaky_high_cyclen = _record(1, cyclen=690.0, node_peak=1.44, map_cov=0.41)
    flat_low_cyclen = _record(2, cyclen=632.0, node_peak=1.2285, map_cov=0.31)
    mid = _record(3, cyclen=660.0, node_peak=1.33, map_cov=0.35)
    recs = [peaky_high_cyclen, flat_low_cyclen, mid]

    drv = _campaign_driver(tmp_path / "flat", recs, "flat_power")
    assert drv.objective == "flat_power"
    got = [rid for rid, _ in drv._store_elites()]
    assert got[0] == flat_low_cyclen.record_id, (
        "the flattest verified row must be the FIRST elite parent of a "
        "flat_power campaign"
    )
    assert got.index(mid.record_id) < got.index(peaky_high_cyclen.record_id)

    # ... while a cycle-length campaign over the SAME store ranks the other way.
    drv2 = _campaign_driver(tmp_path / "cyc", recs, "max_cycle_min_fr")
    got2 = [rid for rid, _ in drv2._store_elites()]
    assert got2[0] == peaky_high_cyclen.record_id


def test_campaign_flat_power_elites_prefer_feasible_rows(tmp_path: Path) -> None:
    """Feasibility-first still holds: an infeasible row cannot outrank a feasible
    one on flatness alone."""

    flat_but_infeasible = _record(1, cyclen=632.0, node_peak=1.20, map_cov=0.30,
                                  cbc=1900.0)
    feasible = _record(2, cyclen=640.0, node_peak=1.31, map_cov=0.33)
    drv = _campaign_driver(tmp_path / "feas", [flat_but_infeasible, feasible],
                           "flat_power")
    got = [rid for rid, _ in drv._store_elites()]
    assert got[0] == feasible.record_id


# --------------------------------------------------------------------------- #
# 8) NaN == missing == None (2026-07-31): the feasible-first elite tier works on
#    parquet-round-tripped rows, which carry a MISSING pin BU as NaN
# --------------------------------------------------------------------------- #
def test_store_elite_tier_survives_a_nan_pin_burnup(tmp_path: Path) -> None:
    """The regression this decision exists for.

    ``max_pin_burnup=None`` becomes ``NaN`` in ``records.parquet``, and
    ``is_feasible``'s pin-BU guard tested ``is not None`` — so EVERY store row
    was judged infeasible under ``flat_power`` (which gates pin BU at 80.0), the
    feasible tier of ``_store_elites`` went empty, and feasibility stopped
    influencing the elite parents at all.
    """

    # No pin-BU label anywhere: exactly what a produce store looks like.
    flat_but_over_cbc = _record(1, cyclen=632.0, node_peak=1.20, map_cov=0.30,
                                cbc=1900.0, pin_bu=None)
    feasible = _record(2, cyclen=640.0, node_peak=1.31, map_cov=0.33, pin_bu=None)
    drv = _campaign_driver(tmp_path / "nanpin", [flat_but_over_cbc, feasible],
                           "flat_power")

    rows = drv._case_store_rows(converged=True)
    assert len(rows) == 2
    # the frame really did round-trip the missing float to NaN ...
    import math
    assert all(math.isnan(float(r["max_pin_burnup"])) for r in rows)
    assert drv.feasibility_limits()["max_pin_burnup"] == 80.0

    # ... and the tier is intact: the CBC violator is still infeasible, the other
    # row is feasible DESPITE having no pin-BU measurement.
    verdicts = {r["record_id"]: drv._is_feasible(r) for r in rows}
    assert verdicts[feasible.record_id] is True
    assert verdicts[flat_but_over_cbc.record_id] is False

    got = [rid for rid, _ in drv._store_elites()]
    assert got[0] == feasible.record_id, (
        "the feasible row must lead the elite parents even though it is the LESS "
        "flat of the two — this is the tier that the NaN hole silently removed"
    )


def test_nan_pin_burnup_row_is_feasible_but_a_measured_violation_is_not(
    tmp_path: Path,
) -> None:
    """Only "missing" moved: a MEASURED over-limit pin BU is still infeasible."""

    no_label = _record(1, node_peak=1.30, map_cov=0.32, pin_bu=None)
    over = _record(2, node_peak=1.25, map_cov=0.31, pin_bu=85.0)
    under = _record(3, node_peak=1.27, map_cov=0.31, pin_bu=78.0)
    drv = _campaign_driver(tmp_path / "pin", [no_label, over, under], "flat_power")
    verdicts = {r["record_id"]: drv._is_feasible(r)
                for r in drv._case_store_rows(converged=True)}
    assert verdicts[no_label.record_id] is True
    assert verdicts[under.record_id] is True
    assert verdicts[over.record_id] is False
    # ordering: both feasible rows lead, flattest first; the violator is last.
    got = [rid for rid, _ in drv._store_elites()]
    assert got[:2] == [under.record_id, no_label.record_id]
    assert got[-1] == over.record_id
