"""multi-PC produce kit: export (deck generation, pair selection, generator
degradation, kit layout), empty-store produce tolerance + per-cell campaign
tagging, and merge-store round-trip (new/upgrade/dup, per-campaign counts,
idempotent re-run, unknown-campaign flag, ledger dedup).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lpopt.config import (
    CaseConfig,
    CurriculumConfig,
    DataConfig,
    ExtractConfig,
    FlowConfig,
    FuelConfig,
    LpoptConfig,
    MasterConfig,
    ModelConfig,
    ProduceConfig,
    RemoteConfig,
    StratumConfig,
    VerifyConfig,
    load_config,
)
from lpopt.data.schema import CanonicalRecord, SCHEMA_COLUMNS
from lpopt.data.store import StoreReader, StoreWriter
from lpopt.multi_pc import (
    KitError,
    _degrade_generators,
    build_cell_stratum,
    export_produce_kit,
    is_recognized_cell,
    merge_store,
    parse_cell_id,
)
from lpopt.search.produce import Ledger, ProduceDriver

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# cell-id parsing / recognition
# --------------------------------------------------------------------------- #
def test_parse_cell_id_roundtrip() -> None:
    assert parse_cell_id("5.25-5.5_f101") == ((5.25, 5.5), 101)
    assert parse_cell_id("5-5.25_f125") == ((5.0, 5.25), 125)
    assert parse_cell_id("6.25-6.5_f141") == ((6.25, 6.5), 141)


@pytest.mark.parametrize(
    "bad",
    [
        "5.25-5.5",          # no feed
        "5.25_f101",         # no band range
        "5.0-5.25_f101",     # non-canonical (5.0 should format as 5)
        "5.25-5.5_f100",     # off the 1+4N grid
        "garbage",
    ],
)
def test_parse_cell_id_rejects(bad: str) -> None:
    with pytest.raises(KitError):
        parse_cell_id(bad)


def test_is_recognized_cell() -> None:
    assert is_recognized_cell("5.25-5.5_f117")
    assert is_recognized_cell("5-5.25_f125")
    assert not is_recognized_cell("P0_pathfinder")
    assert not is_recognized_cell("5.0-5.25_f101")   # non-canonical
    assert not is_recognized_cell("5.25-5.5_f100")   # off-grid feed


# --------------------------------------------------------------------------- #
# generator degradation (elite_perturb dropped + renormalized)
# --------------------------------------------------------------------------- #
def test_degrade_generators_drops_elite() -> None:
    out = _degrade_generators({"random": 0.4, "heuristic": 0.4, "elite_perturb": 0.2})
    assert "elite_perturb" not in out
    assert set(out) == {"random", "heuristic"}
    assert out["random"] == pytest.approx(0.5)
    assert out["heuristic"] == pytest.approx(0.5)
    assert sum(out.values()) == pytest.approx(1.0)


def test_degrade_generators_only_elite_falls_back_to_random() -> None:
    assert _degrade_generators({"elite_perturb": 1.0}) == {"random": 1.0}
    assert _degrade_generators({}) == {"random": 1.0}


# --------------------------------------------------------------------------- #
# export: kit generation on the real deck (real fuel table)
# --------------------------------------------------------------------------- #
def _fake_ga80_package(base: Path) -> Path:
    """A tiny stand-in for FEASIBLE_PACKAGE (lib/ bases/ cores/).

    Keeps the export tests off the real 456 MB package while still exercising the
    bundling branch end-to-end.
    """
    pkg = base / "FEASIBLE_PACKAGE_src"
    (pkg / "lib").mkdir(parents=True, exist_ok=True)
    (pkg / "lib" / "MAS_XSL").write_text("COMP FA_K1 x\n", encoding="utf-8")
    (pkg / "lib" / "MAS_HFF").write_text("hff\n", encoding="utf-8")
    for folder in ("K1_K2", "K1_K2_f117", "L1_L2"):
        (pkg / "bases" / folder).mkdir(parents=True, exist_ok=True)
        (pkg / "bases" / folder / f"MAS_RST.{folder}.01").write_bytes(b"rst")
        (pkg / "cores" / folder / "seed").mkdir(parents=True, exist_ok=True)
        (pkg / "cores" / folder / "seed" / "MAS_INP_cy12.inp").write_text(
            "%LPD_SHF\n F K1  0,\n", encoding="utf-8")
    # not read by the produce path -> must NOT be bundled
    (pkg / "hgc").mkdir(parents=True, exist_ok=True)
    (pkg / "hgc" / "FA_K1.HGC").write_text("hgc\n", encoding="utf-8")
    return pkg


def test_export_kit_generation(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(_fake_ga80_package(tmp_path))
    out = tmp_path / "kit"
    cells = ["5.25-5.5_f101", "5-5.25_f125"]
    result = export_produce_kit(cfg, cells, out, n_target=150, log=lambda m: None)

    # the generated deck parses and its strata EXACTLY match the requested cells.
    parsed = load_config(result.deck_path)
    assert [s.name for s in parsed.produce.strata] == cells
    assert [s.campaign for s in parsed.produce.strata] == cells

    by_name = {s.name: s for s in parsed.produce.strata}
    # feed>121 rule for allow_single_cycle_discharge (f101 -> False, f125 -> True)
    assert by_name["5.25-5.5_f101"].allow_single_cycle_discharge is False
    assert by_name["5-5.25_f125"].allow_single_cycle_discharge is True
    # pair selection sane: non-empty and matches the curriculum's own selector
    from lpopt.curriculum import select_cell_pairs
    from lpopt.data.fuel_types import FuelLibrary
    lib = FuelLibrary.from_parquet(REPO_ROOT / "data/store/fuel_types.parquet")
    for cid, band in (("5.25-5.5_f101", [5.25, 5.5]), ("5-5.25_f125", [5.0, 5.25])):
        feed = int(cid.split("_f")[1])
        want = select_cell_pairs(cfg.curriculum, cid, band, feed, lib, cfg.curriculum.library)
        assert list(by_name[cid].pairs) == want
        assert by_name[cid].pairs, "no pairs selected"
    # generators degraded: no elite_perturb anywhere
    for s in parsed.produce.strata:
        assert "elite_perturb" not in s.generators
        assert sum(s.generators.values()) == pytest.approx(1.0)

    # kit layout
    assert (out / "lpopt" / "cli.py").is_file()          # shipped source tree
    assert (out / "pyproject.toml").is_file()
    assert (out / "data" / "store" / "fuel_types.parquet").is_file()
    # a FRESH empty store: NO records.parquet shipped
    assert not (out / "data" / "store" / "records.parquet").exists()
    # no __pycache__ copied
    assert not list((out / "lpopt").rglob("__pycache__"))

    # README (Korean) covers the required setup / run / return steps
    readme = (out / "KIT_README.md").read_text(encoding="utf-8")
    assert "python -m lpopt produce --input lpopt_kit.inp" in readme
    assert "pip install -e ." in readme
    assert "torch" in readme and "불필요" in readme
    assert "FEASIBLE_PACKAGE" in readme
    assert "package_root" in readme and "executable" in readme
    assert "data" in readme and "Compress-Archive" in readme  # how to ship results back


def test_export_kit_rejects_unknown_cell(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    with pytest.raises(KitError):
        export_produce_kit(cfg, ["not-a-cell"], tmp_path / "kit", log=lambda m: None)


# --------------------------------------------------------------------------- #
# export: a ga80 kit bundles the MASTER package (regression — hand-copying a
# SUBSET of bases/ silently demotes restart resolution instead of failing)
# --------------------------------------------------------------------------- #
def test_export_kit_ga80_bundles_master_package(tmp_path: Path) -> None:
    """A ga80 kit ships lib/ + the WHOLE bases/ catalog + cores/, and its deck
    points [verify].package_root at the bundled copy — so the kit is
    self-contained and no hand-copy step can subset the restart catalog."""
    from lpopt.multi_pc import KIT_GA80_PACKAGE_REL

    src = _fake_ga80_package(tmp_path)
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(src)
    out = tmp_path / "kit"
    export_produce_kit(cfg, ["5.25-5.5_f101"], out, n_target=150, log=lambda m: None)

    kit_pkg = out / KIT_GA80_PACKAGE_REL
    assert (kit_pkg / "lib" / "MAS_XSL").is_file()
    assert (kit_pkg / "lib" / "MAS_HFF").is_file()
    # the ENTIRE bases catalog ships, including exact-feed folders for pairs that
    # are NOT in the assigned cell — the restart ladder scans all of them.
    got_bases = sorted(p.name for p in (kit_pkg / "bases").iterdir() if p.is_dir())
    assert got_bases == ["K1_K2", "K1_K2_f117", "L1_L2"]
    assert (kit_pkg / "cores" / "K1_K2_f117" / "seed" / "MAS_INP_cy12.inp").is_file()
    # hgc/ is lattice-build input only -> not bundled
    assert not (kit_pkg / "hgc").exists()

    parsed = load_config(out / "lpopt_kit.inp")
    assert parsed.verify.package_root == KIT_GA80_PACKAGE_REL
    assert parsed.design.package_root is None
    # and the resolver built from the KIT deck sees the full catalog
    assert (out / KIT_GA80_PACKAGE_REL / "bases" / "K1_K2_f117").is_dir()

    readme = (out / "KIT_README.md").read_text(encoding="utf-8")
    assert KIT_GA80_PACKAGE_REL in readme


def test_export_kit_ga80_carries_external_fallback_deck(tmp_path: Path) -> None:
    """A case whose template resolves through [produce].template_fallbacks (a tree
    OUTSIDE the package) must have that deck carried INTO the kit, so the kit
    resolves the same deck instead of dropping to the synthesis tier."""
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.multi_pc import KIT_GA80_PACKAGE_REL
    from lpopt.search.resolver import build_case_resolver

    src = _fake_ga80_package(tmp_path)
    # a pair with a base restart but NO cores/ deck of its own -> the only
    # readable reload template lives in the external fallback tree
    (src / "bases" / "Z1_Z2").mkdir(parents=True, exist_ok=True)
    (src / "bases" / "Z1_Z2" / "MAS_RST.Z1_Z2.01").write_bytes(b"rst")
    ext = tmp_path / "runs_flow" / "cand"
    ext.mkdir(parents=True, exist_ok=True)
    ext_deck = ext / "MAS_INP_cy12.inp"
    ext_deck.write_text("%LPD_SHF\n F Z1  0,\n", encoding="utf-8")

    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(src)
    cfg.produce.template_fallbacks = [str(ext / "MAS_INP_cy*.inp")]
    cfg.curriculum.cell_pairs = {"5.25-5.5_f101": ["Z1_Z2"]}

    out = tmp_path / "kit"
    lib = FuelLibrary.from_parquet(REPO_ROOT / "data/store/fuel_types.parquet")
    export_produce_kit(cfg, ["5.25-5.5_f101"], out, n_target=8,
                       fuel_library=lib, log=lambda m: None)

    carried = out / KIT_GA80_PACKAGE_REL / "cores" / "Z1_Z2_f101" / "carried"
    assert (carried / "MAS_INP_cy12.inp").is_file()
    assert (carried / "MAS_INP_cy12.inp").read_bytes() == ext_deck.read_bytes()

    # and the KIT's own resolver now returns that deck for the case
    kit_cfg = load_config(out / "lpopt_kit.inp")
    got = build_case_resolver(kit_cfg, lib, "ga80").resolve(("Z1_Z2", 101))
    assert got.template_deck_path is not None
    assert got.template_deck_path.read_bytes() == ext_deck.read_bytes()


def test_export_kit_ga80_no_carry_when_package_is_deck_complete(tmp_path: Path) -> None:
    """No external carry happens when every assigned case resolves inside the
    package — the kit stays a plain package copy."""
    from lpopt.multi_pc import KIT_GA80_PACKAGE_REL

    src = _fake_ga80_package(tmp_path)
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(src)
    cfg.produce.template_fallbacks = [str(tmp_path / "nothing" / "*.inp")]
    cfg.curriculum.cell_pairs = {"5.25-5.5_f101": ["K1_K2"]}
    out = tmp_path / "kit"
    export_produce_kit(cfg, ["5.25-5.5_f101"], out, n_target=8, log=lambda m: None)
    assert not list((out / KIT_GA80_PACKAGE_REL / "cores").glob("*/carried"))


def test_export_kit_ga80_missing_package_raises(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(tmp_path / "nope")
    with pytest.raises(KitError):
        export_produce_kit(cfg, ["5.25-5.5_f101"], tmp_path / "kit", log=lambda m: None)


def test_export_kit_ga80_incomplete_package_raises(tmp_path: Path) -> None:
    """A package missing one of lib/ bases/ cores/ must fail the export, not ship
    a kit that dies at MASTER-staging time on the remote PC."""
    src = _fake_ga80_package(tmp_path)
    import shutil as _sh
    _sh.rmtree(src / "bases")
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(src)
    with pytest.raises(KitError, match="bases"):
        export_produce_kit(cfg, ["5.25-5.5_f101"], tmp_path / "kit", log=lambda m: None)


def test_build_cell_stratum_no_pairs_raises() -> None:
    """A band with no in-band types must fail loudly rather than ship an
    empty-pairs stratum that would stall on PC2."""
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    from lpopt.data.fuel_types import FuelLibrary
    lib = FuelLibrary.from_parquet(REPO_ROOT / "data/store/fuel_types.parquet")
    # a band far outside the ga80 enrichment coverage yields no pairs
    with pytest.raises(KitError):
        build_cell_stratum(cfg, "9-9.25_f101", (9.0, 9.25), 101, lib)


# --------------------------------------------------------------------------- #
# empty-store produce tolerance + per-cell campaign tagging
# --------------------------------------------------------------------------- #
def _make_package(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    base = pkg / "bases" / "K1_K2"
    base.mkdir(parents=True)
    (base / "MAS_RST.NATIVE.01").write_bytes(b"rst")
    core = pkg / "cores" / "K1_K2" / "s1"
    core.mkdir(parents=True)
    (core / "MAS_INP_cy01.inp").write_text(
        "dummy\n%LPD_SHF\n F K1  0,\n%END\n", encoding="utf-8"
    )
    return pkg


def _produce_cfg(tmp_path: Path, pkg: Path, strata: list[StratumConfig]) -> LpoptConfig:
    return LpoptConfig(
        flow=FlowConfig(random_seed=3),
        remote=RemoteConfig(),
        master=MasterConfig(),
        verify=VerifyConfig(package_root=str(pkg)),
        data=DataConfig(),
        case=CaseConfig(),
        fuel=FuelConfig(),
        extract=ExtractConfig(),
        produce=ProduceConfig(campaign="multipc_kit", workers=4,
                              template_fallbacks=[], strata=strata),
        source_path=tmp_path / "lpopt.inp",
    )


def test_empty_store_produce_tolerance_and_campaign_tag(tmp_path: Path) -> None:
    """The produce path tolerates a MISSING records.parquet (fresh PC2 store) and
    tags each stratum's rows with its per-stratum campaign == cell id."""
    pkg = _make_package(tmp_path)
    empty_store = tmp_path / "store"          # does NOT exist yet -> fresh
    assert not (empty_store / "records.parquet").exists()

    strat = StratumConfig(name="5.25-5.5_f101", campaign="5.25-5.5_f101",
                          pairs=["K1_K2"], feed=121, n_target=8,
                          generators={"random": 1.0}, priority=100)
    cfg = _produce_cfg(tmp_path, pkg, [strat])
    driver = ProduceDriver(
        cfg, dry_run=True, run_dir=tmp_path / "run",
        store_dir=empty_store, ledger_path=tmp_path / "ledger.jsonl",
        progress=False,
    )
    summary = driver.run()
    assert summary.converged == 8

    df = StoreReader(empty_store).records
    p = df[df["dataset"] == "P"]
    assert len(p) == 8
    # per-stratum campaign override reached the stored rows (curriculum reads this)
    assert set(p["campaign"]) == {"5.25-5.5_f101"}
    assert set(p["stratum"]) == {"5.25-5.5_f101"}


# --------------------------------------------------------------------------- #
# merge-store round-trip
# --------------------------------------------------------------------------- #
def _p_record(rid: str, campaign: str, *, converged: bool = True, valid: bool = True,
              cyclen: float = 620.0, dataset: str = "P", **late) -> CanonicalRecord:
    fields = dict(
        record_id=rid, dataset=dataset, campaign=campaign, stratum=campaign,
        generator="random", parent_record_id=None, case_pair="L1_L2", feed=101,
        n_batches=3, depth2_edges=0, e_core=5.4, e_split=0.1, library_id="ga80",
        sym_class="rot61", pattern="F:L1:0|F:L2:0",
        f_r=1.5, f_q=2.3, cbc_max=1400.0, cbc_boc=1380.0, cbc_kind="max",
        cyclen=cyclen, ao_abs=0.05, cycle_burnup=27.0, discharge_burnup=54.0,
        max_assembly_burnup=67.0, max_pin_burnup=71.0, eoc_ppm=10.0,
        delta_efpd=0.5, n_cycles=11.0, converged=converged, converged_at_cap=False,
        tolerance_margin=None, restart_provenance="native:MAS_RST.X",
        valid=valid, failure="" if valid else "non_finite_flux", maps_key=None,
    )
    fields.update(late)
    return CanonicalRecord(**fields)


def _merge_cfg(tmp_path: Path, main_store: Path, main_ledger: Path) -> LpoptConfig:
    return LpoptConfig(
        flow=FlowConfig(),
        remote=RemoteConfig(),
        master=MasterConfig(),
        verify=VerifyConfig(),
        data=DataConfig(),
        case=CaseConfig(),
        fuel=FuelConfig(),
        extract=ExtractConfig(),
        produce=ProduceConfig(ledger=str(main_ledger)),
        model=ModelConfig(store_dir=str(main_store)),
        curriculum=CurriculumConfig(state_dir=str(tmp_path / "no_such_curr")),
        source_path=tmp_path / "lpopt.inp",
    )


def _write_kit(kit_data: Path, records: list[CanonicalRecord],
               ledger_rows: list[dict] | None = None) -> None:
    StoreWriter(kit_data / "store").write_records(records, append=False)
    if ledger_rows is not None:
        led = Ledger(kit_data / "produce" / "ledger.jsonl")
        for row in ledger_rows:
            led.append(**row)


def test_merge_roundtrip_new_upgrade_dup(tmp_path: Path) -> None:
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, main_store, main_ledger)

    # main store: one converged, one NON-converged that the kit will UPGRADE.
    StoreWriter(main_store).write_records([
        _p_record("keep", "5.25-5.5_f101", converged=True),
        _p_record("upgrade_me", "5.25-5.5_f101", converged=False, valid=True),
    ], append=False)

    # kit: a brand-new row, a converged upgrade of 'upgrade_me', and a worse dup
    # of 'keep' (non-converged) that must NOT downgrade the stored converged row.
    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [
        _p_record("brand_new", "5.25-5.5_f101", converged=True),
        _p_record("upgrade_me", "5.25-5.5_f101", converged=True, valid=True),
        _p_record("keep", "5.25-5.5_f101", converged=False, valid=True),
    ])

    report = merge_store(cfg, kit_data, log=lambda m: None)
    assert report.kit_rows == 3
    assert report.new_rows == 1          # brand_new
    assert report.upgraded_rows == 1     # upgrade_me
    assert report.duplicate_rows == 1    # keep (worse -> ignored)
    assert report.total_before == 2 and report.total_after == 3

    # per-campaign converged before/after: 1 -> 3 (brand_new + upgraded)
    row = next(r for r in report.per_campaign if r["campaign"] == "5.25-5.5_f101")
    assert row["before"] == 1 and row["after"] == 3

    # the store on disk reflects the merge; the worse dup did not downgrade 'keep'
    df = StoreReader(main_store).records
    assert len(df) == 3
    assert bool(df[df["record_id"] == "keep"]["converged"].iloc[0]) is True
    assert bool(df[df["record_id"] == "upgrade_me"]["converged"].iloc[0]) is True

    # idempotent: a second merge is a no-op (all duplicates, store unchanged)
    before_bytes = (main_store / "records.parquet").read_bytes()
    report2 = merge_store(cfg, kit_data, log=lambda m: None)
    assert report2.new_rows == 0 and report2.upgraded_rows == 0
    assert report2.duplicate_rows == 3
    assert (main_store / "records.parquet").read_bytes() == before_bytes


# --------------------------------------------------------------------------- #
# the flatness columns are part of the UPGRADE decision
#
# The regression: the merge's quality rank was ``converged*2 + valid``, so a kit
# row identical to the stored one EXCEPT that it carried node_peak / map_cov
# ranked equal, was counted a duplicate, and — because ``changed`` is what
# decides whether the store is rewritten at all — the harvested flatness labels
# were dropped at the door.
# --------------------------------------------------------------------------- #
def test_merge_upgrades_a_row_that_only_gained_the_flatness_columns(
        tmp_path: Path) -> None:
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, main_store, main_ledger)

    StoreWriter(main_store).write_records(
        [_p_record("mapped", "5.25-5.5_f101", converged=True)], append=False)

    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [
        _p_record("mapped", "5.25-5.5_f101", converged=True,
                  node_peak=1.512, map_cov=0.0834, maps_key="mapped"),
    ])

    report = merge_store(cfg, kit_data, log=lambda m: None)
    assert report.upgraded_rows == 1
    assert report.duplicate_rows == 0

    df = StoreReader(main_store).records.set_index("record_id")
    assert float(df.loc["mapped", "node_peak"]) == pytest.approx(1.512)
    assert float(df.loc["mapped", "map_cov"]) == pytest.approx(0.0834)


def test_merge_never_nulls_out_an_already_harvested_flatness_row(
        tmp_path: Path) -> None:
    """The reverse direction: an unmapped kit row must not clobber a mapped one."""
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, main_store, main_ledger)

    StoreWriter(main_store).write_records([
        _p_record("mapped", "5.25-5.5_f101", converged=True,
                  node_peak=1.512, map_cov=0.0834, maps_key="mapped"),
    ], append=False)

    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [
        _p_record("mapped", "5.25-5.5_f101", converged=True),      # no map
        _p_record("brand_new", "5.25-5.5_f101", converged=True),   # forces a write
    ])

    report = merge_store(cfg, kit_data, log=lambda m: None)
    assert report.new_rows == 1 and report.duplicate_rows == 1

    df = StoreReader(main_store).records.set_index("record_id")
    assert float(df.loc["mapped", "node_peak"]) == pytest.approx(1.512)


def test_merge_into_empty_main_store(tmp_path: Path) -> None:
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, main_store, main_ledger)
    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [
        _p_record("a", "5-5.25_f125", converged=True),
        _p_record("b", "5-5.25_f125", converged=False, valid=True),
    ])
    report = merge_store(cfg, kit_data, log=lambda m: None)
    assert report.new_rows == 2 and report.total_before == 0 and report.total_after == 2
    row = next(r for r in report.per_campaign if r["campaign"] == "5-5.25_f125")
    assert row["before"] == 0 and row["after"] == 1     # only the converged one
    assert StoreReader(main_store).records.shape[0] == 2


def test_merge_flags_unknown_campaign(tmp_path: Path) -> None:
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, main_store, main_ledger)
    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [
        _p_record("x", "5.25-5.5_f101", converged=True),   # recognized cell
        _p_record("y", "P0_pathfinder", converged=True),   # NOT a cell id
    ])
    report = merge_store(cfg, kit_data, log=lambda m: None)
    flagged = {f["campaign"] for f in report.flagged_campaigns}
    assert flagged == {"P0_pathfinder"}
    # flagged campaigns are still MERGED (refuses nothing silently)
    assert StoreReader(main_store).records.shape[0] == 2


def test_merge_dry_run_writes_nothing(tmp_path: Path) -> None:
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, main_store, main_ledger)
    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [_p_record("a", "5-5.25_f125", converged=True)],
               ledger_rows=[{"record_id": "a", "stratum": "5-5.25_f125",
                             "generator": "random", "status": "done"}])
    report = merge_store(cfg, kit_data, dry_run=True, log=lambda m: None)
    assert report.new_rows == 1 and report.dry_run is True
    assert not (main_store / "records.parquet").exists()   # store not written
    assert not main_ledger.exists()                        # ledger not written


# --------------------------------------------------------------------------- #
# ledger merge dedup
# --------------------------------------------------------------------------- #
def test_merge_ledger_dedup_idempotent(tmp_path: Path) -> None:
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, main_store, main_ledger)

    # main ledger already has a 'running' line for rid 'a'
    Ledger(main_ledger).append(record_id="a", stratum="5-5.25_f125",
                               generator="random", status="running")

    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [
        _p_record("a", "5-5.25_f125", converged=True),
        _p_record("b", "5-5.25_f125", converged=True),
    ], ledger_rows=[
        {"record_id": "a", "stratum": "5-5.25_f125", "generator": "random", "status": "running"},
        {"record_id": "a", "stratum": "5-5.25_f125", "generator": "random", "status": "done"},
        {"record_id": "b", "stratum": "5-5.25_f125", "generator": "random", "status": "done"},
        {"record_id": "", "stratum": "5-5.25_f125", "generator": "random", "status": "dup"},
    ])

    report = merge_store(cfg, kit_data, log=lambda m: None)
    # ('a','running') already present -> skipped; ('a','done'),('b','done') new;
    # the rid-less 'dup' line is skipped.
    assert report.ledger["appended"] == 2
    assert report.ledger["skipped_existing"] == 1
    assert report.ledger["skipped_ridless"] == 1

    rows = Ledger.replay(main_ledger)
    keys = [(str(r.get("record_id", "")), r.get("status")) for r in rows]
    assert keys.count(("a", "running")) == 1      # not duplicated
    assert ("a", "done") in keys and ("b", "done") in keys

    # idempotent: re-merging appends nothing new
    report2 = merge_store(cfg, kit_data, log=lambda m: None)
    assert report2.ledger["appended"] == 0
    assert len(Ledger.replay(main_ledger)) == len(rows)


# --------------------------------------------------------------------------- #
# CLI end-to-end (export then merge)
# --------------------------------------------------------------------------- #
def test_cli_export_then_merge(tmp_path: Path, capsys) -> None:
    from lpopt.cli import main

    out = tmp_path / "kit"
    rc = main([
        "export-produce-kit", "--input", str(REPO_ROOT / "lpopt.inp"),
        "--cells", "5.25-5.5_f101", "--out", str(out), "--n-target", "30",
    ])
    assert rc == 0
    assert (out / "lpopt_kit.inp").is_file()
    kit_out = capsys.readouterr().out
    assert "RESULT: OK" in kit_out

    # build a synthetic returned kit data/ and merge it via the CLI
    main_store = tmp_path / "main" / "store"
    main_ledger = tmp_path / "main" / "produce" / "ledger.jsonl"
    deck = tmp_path / "lpopt.inp"
    deck.write_text(
        f"[model]\nstore_dir = {json.dumps(str(main_store))}\n"
        f"[produce]\nledger = {json.dumps(str(main_ledger))}\n",
        encoding="utf-8",
    )
    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [_p_record("z", "5.25-5.5_f101", converged=True)])

    rc2 = main(["merge-store", "--input", str(deck), "--from", str(kit_data)])
    assert rc2 == 0
    merge_out = capsys.readouterr().out
    assert "merge-store" in merge_out and "RESULT: OK" in merge_out
    assert StoreReader(main_store).records.shape[0] == 1


def test_merge_store_dir_ledger_override(tmp_path: Path) -> None:
    """``store_dir`` / ``ledger`` overrides retarget the merge at a SCRATCH copy so
    the deck's own store/ledger are NEVER touched (safe rehearsal while the main
    producer is live)."""
    deck_store = tmp_path / "deck_store"
    deck_ledger = tmp_path / "deck_produce" / "ledger.jsonl"
    cfg = _merge_cfg(tmp_path, deck_store, deck_ledger)

    # scratch copy of a store the override will merge INTO (starts non-empty)
    scratch_store = tmp_path / "scratch" / "store"
    scratch_ledger = tmp_path / "scratch" / "produce" / "ledger.jsonl"
    StoreWriter(scratch_store).write_records(
        [_p_record("existing", "5.25-5.5_f101", converged=True)], append=False)

    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [
        _p_record("brand_new", "5.25-5.5_f101", converged=True),
        _p_record("existing", "5.25-5.5_f101", converged=False, valid=True),  # worse dup
    ], ledger_rows=[{"record_id": "brand_new", "stratum": "5.25-5.5_f101",
                     "generator": "random", "status": "done"}])

    report = merge_store(cfg, kit_data, store_dir=scratch_store, ledger=scratch_ledger,
                         log=lambda m: None)
    assert report.new_rows == 1 and report.duplicate_rows == 1
    assert report.store_dir == str(scratch_store)

    # the OVERRIDE targets received the merge ...
    assert StoreReader(scratch_store).records.shape[0] == 2
    assert scratch_ledger.exists()
    # ... and the deck's own store/ledger were NEVER created/written.
    assert not (deck_store / "records.parquet").exists()
    assert not deck_ledger.exists()

    # idempotent under the override too
    report2 = merge_store(cfg, kit_data, store_dir=scratch_store, ledger=scratch_ledger,
                          log=lambda m: None)
    assert report2.new_rows == 0 and report2.upgraded_rows == 0
    assert StoreReader(scratch_store).records.shape[0] == 2


def test_cli_merge_store_dir_ledger_flags(tmp_path: Path, capsys) -> None:
    """The ``--store-dir`` / ``--ledger`` CLI flags thread through to the override."""
    from lpopt.cli import main

    deck_store = tmp_path / "deck_store"
    deck_ledger = tmp_path / "deck_produce" / "ledger.jsonl"
    deck = tmp_path / "lpopt.inp"
    deck.write_text(
        f"[model]\nstore_dir = {json.dumps(str(deck_store))}\n"
        f"[produce]\nledger = {json.dumps(str(deck_ledger))}\n",
        encoding="utf-8",
    )
    scratch_store = tmp_path / "scratch" / "store"
    scratch_ledger = tmp_path / "scratch" / "produce" / "ledger.jsonl"
    kit_data = tmp_path / "kit_data"
    _write_kit(kit_data, [_p_record("z", "5.25-5.5_f101", converged=True)])

    rc = main(["merge-store", "--input", str(deck), "--from", str(kit_data),
               "--store-dir", str(scratch_store), "--ledger", str(scratch_ledger)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RESULT: OK" in out
    assert StoreReader(scratch_store).records.shape[0] == 1
    assert not (deck_store / "records.parquet").exists()   # deck store untouched


# --------------------------------------------------------------------------- #
# export: fr_boundary frontier kit (PC2 worker — user directive 2026-07-22)
# --------------------------------------------------------------------------- #
def _frontier_record(pair: str, feed: int = 121) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=f"frB_{pair}_f{feed}", dataset="P", campaign="frB", stratum="frB",
        generator="random", parent_record_id=None, case_pair=pair, feed=feed,
        n_batches=2, depth2_edges=0, e_core=5.2, e_split=0.0, library_id="ga80",
        sym_class="rot61", pattern="F:X1:0|F:X2:0",
        f_r=1.56, f_q=2.30, cbc_max=1400.0, cbc_boc=1380.0, cbc_kind="max",
        cyclen=625.0, ao_abs=0.12, cycle_burnup=27.0, discharge_burnup=54.0,
        max_assembly_burnup=67.0, max_pin_burnup=70.0, eoc_ppm=10.0,
        delta_efpd=0.5, n_cycles=12.0, converged=True, converged_at_cap=False,
        tolerance_margin=0.2, restart_provenance="native:MAS_RST.X",
        valid=True, failure="", maps_key=None,
    )


def _frontier_cfg(tmp_path: Path) -> LpoptConfig:
    from lpopt.multi_pc import frontier_roster_pairs

    store = tmp_path / "store"
    StoreWriter(store).write_records([_frontier_record(p) for p in frontier_roster_pairs()])
    # ship a fuel table alongside (copy the repo one if present, else a stub row).
    src_fuel = REPO_ROOT / "data" / "store" / "fuel_types.parquet"
    if src_fuel.exists():
        import shutil
        shutil.copy2(src_fuel, store / "fuel_types.parquet")
    model = tmp_path / "champion"
    model.mkdir()
    (model / "ensemble.json").write_text("{}", encoding="utf-8")
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(_fake_ga80_package(tmp_path))
    cfg.model.store_dir = str(store)
    cfg.model.model_dir = str(model)
    return cfg


def test_export_frontier_kit_layout(tmp_path: Path) -> None:
    from lpopt.multi_pc import KIT_GA80_PACKAGE_REL, KIT_MODEL_REL, export_frontier_kit

    cfg = _frontier_cfg(tmp_path)
    out = tmp_path / "kit"
    res = export_frontier_kit(cfg, out, log=lambda m: None)

    # deck round-trips with objective=fr_boundary + local_cpu inference + kit model.
    parsed = load_config(res.deck_path)
    assert parsed.acquisition.objective == "fr_boundary"
    assert parsed.model.inference == "local_cpu"
    assert parsed.model.model_dir == KIT_MODEL_REL
    assert parsed.verify.package_root == KIT_GA80_PACKAGE_REL

    # store ships WITH records + maps + fuel table (not the fresh-empty produce kit).
    assert (out / "data" / "store" / "records.parquet").is_file()
    # FEASIBLE_PACKAGE ships WHOLE (lib/bases/cores).
    assert (out / KIT_GA80_PACKAGE_REL / "lib" / "MAS_XSL").is_file()
    assert (out / KIT_GA80_PACKAGE_REL / "bases" / "K1_K2").is_dir()
    # champion model bundled.
    assert (out / KIT_MODEL_REL / "ensemble.json").is_file()
    # source tree + no __pycache__.
    assert (out / "lpopt" / "cli.py").is_file()
    assert not list((out / "lpopt").rglob("__pycache__"))

    # run_frontier.bat sets the worker env + PYTHONUTF8 + the ROUND lifecycle markers.
    bat = res.bat_path.read_text(encoding="utf-8")
    assert "set LPOPT_WORKER=1" in bat and "set PYTHONUTF8=1" in bat
    assert "ROUND_RUNNING" in bat and "ROUND_DONE" in bat and "ROUND_FAILED" in bat
    assert "frontier-produce" in bat
    # schtasks XML shipped.
    assert res.schtasks_path.is_file()
    assert "Task" in res.schtasks_path.read_text(encoding="utf-8")
    # all six roster pairs are mono-anchor.
    assert res.roster_pairs == sorted(res.roster_pairs)
    assert len(res.roster_pairs) == 6


def test_export_frontier_kit_asserts_store_rows_per_pair(tmp_path: Path) -> None:
    from lpopt.multi_pc import export_frontier_kit, frontier_roster_pairs

    # a store MISSING one roster pair must hard-fail (empty elite basin).
    store = tmp_path / "store"
    pairs = frontier_roster_pairs()[:-1]                # drop the last roster pair
    StoreWriter(store).write_records([_frontier_record(p) for p in pairs])
    model = tmp_path / "champion"; model.mkdir()
    (model / "e.json").write_text("{}", encoding="utf-8")
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    cfg.verify.package_root = str(_fake_ga80_package(tmp_path))
    cfg.model.store_dir = str(store)
    cfg.model.model_dir = str(model)
    with pytest.raises(KitError, match="roster pair"):
        export_frontier_kit(cfg, tmp_path / "kit", log=lambda m: None)


def test_export_frontier_kit_refuses_stale_store(tmp_path: Path) -> None:
    import os
    import time as _time
    from lpopt.multi_pc import export_frontier_kit

    cfg = _frontier_cfg(tmp_path)
    # a round marker NEWER than the store records -> the home store predates the last
    # pull, so re-shipping would clobber PC2's newer rows: refuse.
    marker = tmp_path / "ROUND_DONE"
    marker.write_text("done", encoding="utf-8")
    future = _time.time() + 1000
    os.utime(marker, (future, future))
    with pytest.raises(KitError, match="predates the last pulled round"):
        export_frontier_kit(cfg, tmp_path / "kit", last_round_marker=marker,
                            log=lambda m: None)


def test_frontier_kit_deck_always_emits_lean_search_block():
    # forensic 20260723: the frontier deck MUST carry the CPU-sizing [search] block
    # (a local_cpu worker with the 20000/40000 defaults stalled a wave 187 min+).
    from lpopt.multi_pc import render_frontier_kit_deck
    cfg = load_config(REPO_ROOT / "lpopt.inp")
    deck = render_frontier_kit_deck(cfg, round_budget=276)
    assert "[search]" in deck
    assert "[search.local_search]" in deck and "[search.trust_region]" in deck
    import tempfile, os
    p = Path(tempfile.mkdtemp()) / "fr.inp"
    p.write_text(deck, encoding="utf-8")
    parsed = load_config(p)
    assert parsed.search.pool_size <= 2000
    assert parsed.search.local_search.max_predictions <= 1500
    assert parsed.search.local_search.top_m <= 32
    assert parsed.acquisition.objective == "fr_boundary"
