"""The 181 queue recipe as PROPERTIES of ``lattice.run_batch`` (task #10).

The 2_LP queue script was deliberately not adopted (task list §4: "러너 정본은
``lattice.run_batch`` 다 … 큐를 채택하지 않고, 검증된 성질을 가져온다").  What is
under test here is that each ported property actually reaches the runner of
record, that it is OFF by default (so every pre-existing caller launches exactly
what it launched before), and that the registered numbers live in ``[design]``
rather than in a literal at a call site.

No DeCART: ``launch_decart`` is stubbed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lpopt.design import lattice as L
from lpopt.design.spec import DesignRegistry, FuelDesign

# A structurally live one-state HGC (passes G-H1c; G-H1/G-H1b are delivery gates).
VALID_HGC = (
    "%TITL\n"
    " CASE :: REFERENCE CASE\n"
    " 1\n"
    " 1.0 0.0 1.35 1.30 0.0 900.0\n"
    " 305.0 700.0 0 155.0 0.74 1.0\n"
    "%DIST\n"
    + ("1.000 " * 16 + "\n") * 16
    + "% padding to clear the 256-byte floor " + "x" * 256 + "\n"
)


class _FakeProc:
    """A DeCART process stub that has already exited successfully."""

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture()
def stub_launch(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record every launch instead of starting DeCART; write plausible products."""
    seen: list[dict] = []

    _SENTINEL = object()

    def fake_launch(deck_path, work_dir, design, alias,
                    exe=L.DEFAULT_DECART_EXE, env=_SENTINEL):
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / f"FA_{alias}_0101.HGC").write_text(VALID_HGC, encoding="utf-8")
        (wd / f"FA_{alias}.out").write_text("inventory\n", encoding="utf-8")
        (wd / "decart.stdout").write_text("... JOB FINISHED\n", encoding="utf-8")
        seen.append({"alias": alias, "exe": str(exe),
                     "env": None if env is _SENTINEL else env,
                     "env_passed": env is not _SENTINEL,
                     "deck": Path(deck_path), "work_dir": wd})
        run = L.DecartRun(design=design, alias=alias, work_dir=wd,
                          caseid=f"FA_{alias}", fa_name=f"FA_{alias}",
                          deck_path=Path(deck_path))
        run.process = _FakeProc()
        run.started = time.monotonic()
        run.input_sha256 = "deadbeef"
        return run

    monkeypatch.setattr(L, "launch_decart", fake_launch)
    monkeypatch.setattr(L, "write_dec_deck",
                        lambda design, wd, registry, apr: _deck(Path(wd), registry.alias(design)))
    return seen


def _deck(wd: Path, alias: str) -> Path:
    wd.mkdir(parents=True, exist_ok=True)
    deck = wd / f"dec_FA_{alias}.inp"
    deck.write_text(f"CASEID FA_{alias}\nnxfile OLD.BIN\n", encoding="utf-8")
    return deck


def _designs() -> tuple[FuelDesign, FuelDesign]:
    return (FuelDesign(6.2, 5.70, "z2", 8.0, 16),
            FuelDesign(5.8, 4.93, "z1", 8.0, 16))


def _run(designs, tmp_path, **kw):
    reg = DesignRegistry()
    runs = L.run_batch(list(designs), tmp_path / "work", reg, tmp_path / "apr",
                       poll_s=0.001, **kw)
    return reg, runs


# --------------------------------------------------------------------------- #
# (2)(3) the registered numbers live in [design]
# --------------------------------------------------------------------------- #
def test_design_defaults_are_the_registered_181_recipe() -> None:
    """7200 s / 2-way — NOT 5400 / 4.  A slice-Z deck that overrides nothing must
    still get the registered recipe, so this cannot live in a deck."""
    from lpopt.config import DesignConfig

    d = DesignConfig()
    assert d.decart_timeout == 7200
    assert d.max_parallel == 2


def test_cli_design_run_passes_the_config_values_through() -> None:
    """``design run`` must hand [design] max_parallel / decart_timeout to
    run_batch verbatim — a literal at the call site would silently un-register
    the recipe."""
    import inspect

    from lpopt import cli

    src = inspect.getsource(cli.cmd_design_run)
    assert "max_parallel=d.max_parallel" in src
    assert "timeout_s=d.decart_timeout" in src


# --------------------------------------------------------------------------- #
# defaults are byte-identical (nothing below is on unless asked for)
# --------------------------------------------------------------------------- #
def test_no_options_means_no_preflight_no_gate_no_manifest(tmp_path, stub_launch) -> None:
    d0, d1 = _designs()
    reg, runs = _run((d0, d1), tmp_path, max_parallel=5, timeout_s=30.0)
    assert [r.alias for r in runs] == [reg.alias(d0), reg.alias(d1)]
    assert len(stub_launch) == 2
    # no env override, no manifest, deck left exactly as written
    # the historical call passes no ``env`` kwarg at all, so a pre-existing stub
    # with the old signature keeps working (tests/test_design.py)
    assert all(entry["env"] is None for entry in stub_launch)
    assert all(entry["env_passed"] is False for entry in stub_launch)
    assert not (tmp_path / "work" / L.BATCH_MANIFEST_NAME).exists()
    assert "nxfile OLD.BIN" in stub_launch[0]["deck"].read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# (1) design_wave.json as the case source + the manifest design block
# --------------------------------------------------------------------------- #
def _wave(tmp_path: Path) -> Path:
    payload = {
        "wave": "slice_Z",
        "prereg": "assembly_slice_Z_20260903",
        "cases": [
            {"e1": 5.5, "e2": 4.7, "zoning_variant": "z1", "gd_wt": 8.0,
             "n_gd": 20, "gd_positions": "1:1;4:1;6:4",
             "predicted": {"ff": 1.1208, "kinf_boc": 1.2345}},
            {"e1": 6.2, "e2": 5.7, "zoning_variant": "z2", "gd_wt": 8.0,
             "n_gd": 16},
        ],
    }
    p = tmp_path / L.WAVE_FILE_NAME
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_designs_from_wave_reads_cases_and_predictions(tmp_path) -> None:
    designs, meta = L.designs_from_wave(_wave(tmp_path))
    assert [d.type_id for d in designs] == ["P5547Z1G08N20", "P6257Z2G08N16"]
    assert designs[0].gd_layout == "1:1;4:1;6:4"
    assert designs[1].gd_positions is None
    assert meta["wave"] == "slice_Z"
    assert meta["predicted"][designs[0].type_id_tagged]["ff"] == pytest.approx(1.1208)


@pytest.mark.parametrize("payload, needle", [
    ('{"cases": []}', "no 'cases' list"),
    ('{"cases": [{"e1": 5.5}]}', "missing"),
    ('{"cases": [{"e1": 5.5, "e2": 4.7, "zoning_variant": "z1", '
     '"gd_wt": 8.0, "n_gd": 20, "gd_positions": "1:1"}]}', "realizes"),
    ("not json", "malformed"),
])
def test_malformed_wave_is_a_lattice_error(tmp_path, payload, needle) -> None:
    p = tmp_path / "bad.json"
    p.write_text(payload, encoding="utf-8")
    with pytest.raises(L.LatticeError, match=needle):
        L.designs_from_wave(p)


def test_manifest_records_the_design_tuple_positions_and_prediction(
        tmp_path, stub_launch) -> None:
    designs, meta = L.designs_from_wave(_wave(tmp_path))
    opts = L.BatchOptions(manifest=True, wave_meta=meta)
    reg, runs = _run(designs, tmp_path, max_parallel=5, timeout_s=30.0, options=opts)

    manifest = json.loads(
        (tmp_path / "work" / L.BATCH_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["wave"]["wave"] == "slice_Z"
    assert "predicted" not in manifest["wave"]           # per-case, not global
    rows = {r["design"]["type_id"]: r for r in manifest["rows"]}
    z = rows["P5547Z1G08N20"]["design"]
    assert (z["e1"], z["e2"], z["zoning_variant"], z["gd_wt"], z["n_gd"]) == (
        5.5, 4.7, "z1", 8.0, 20)
    assert z["gd_positions"] == "1:1;4:1;6:4"
    assert z["predicted"]["ff"] == pytest.approx(1.1208)
    assert rows["P6257Z2G08N16"]["design"]["gd_positions"] is None
    # the input receipt (queue recipe :63) and the products are recorded
    assert rows["P5547Z1G08N20"]["input_sha256"] == "deadbeef"
    assert rows["P5547Z1G08N20"]["hgc"].endswith(".HGC")


# --------------------------------------------------------------------------- #
# the host process gate (queue recipe :32) does not scale with max_parallel
# --------------------------------------------------------------------------- #
def test_host_process_gate_is_independent_of_max_parallel(tmp_path, stub_launch) -> None:
    """The ceiling is a property of the shared HOST.  Raising max_parallel to 4
    must NOT widen it to 4 — that was the defect in the ported gate."""
    counts = iter([2, 2])                  # host saturated, then it drains
    seen: list[int] = []

    def process_count() -> int:
        n = next(counts, 0)
        seen.append(n)
        return n

    opts = L.BatchOptions(process_count_fn=process_count, host_process_limit=2)
    d0, d1 = _designs()
    reg, runs = _run((d0, d1), tmp_path, max_parallel=4, timeout_s=30.0, options=opts)

    assert L.HOST_PROCESS_LIMIT == 2
    assert L.DECART_PROCESS_NAME == "decart2d1.1m5.exe"
    # both cases still complete, and the gate was consulted while work was live
    assert [r.alias for r in runs] == [reg.alias(d0), reg.alias(d1)]
    assert seen and seen[0] == 2                        # it really did hold once
    assert all(r.hgc_path is not None for r in runs)


def test_process_gate_never_deadlocks_an_idle_batch(tmp_path, stub_launch) -> None:
    """A permanently saturated host must not freeze a batch that has nothing in
    flight: the gate is only consulted once a launch is already active."""
    opts = L.BatchOptions(process_count_fn=lambda: 99, host_process_limit=2)
    d0, d1 = _designs()
    _reg, runs = _run((d0, d1), tmp_path, max_parallel=4, timeout_s=30.0, options=opts)
    assert len(runs) == 2


# --------------------------------------------------------------------------- #
# the JOB FINISHED completion marker (queue recipe :57)
# --------------------------------------------------------------------------- #
def test_missing_success_marker_refuses_to_harvest(tmp_path, monkeypatch) -> None:
    def fake_launch(deck_path, work_dir, design, alias,
                    exe=L.DEFAULT_DECART_EXE, env=None):
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / f"FA_{alias}_0101.HGC").write_text(VALID_HGC, encoding="utf-8")
        (wd / "decart.stdout").write_text("... segmentation fault\n", encoding="utf-8")
        run = L.DecartRun(design=design, alias=alias, work_dir=wd,
                          caseid=f"FA_{alias}", fa_name=f"FA_{alias}")
        run.process = _FakeProc()
        run.started = time.monotonic()
        return run

    monkeypatch.setattr(L, "launch_decart", fake_launch)
    monkeypatch.setattr(L, "write_dec_deck",
                        lambda design, wd, registry, apr: _deck(Path(wd), registry.alias(design)))
    d0, _ = _designs()
    opts = L.BatchOptions(require_success_marker=True)
    _reg, runs = _run((d0,), tmp_path, max_parallel=2, timeout_s=30.0, options=opts)

    assert runs[0].hgc_path is None                     # not renamed into delivery
    assert "JOB FINISHED" in (runs[0].error or "")
    # ... and without the option the same run harvests, as it always did.
    _reg2, runs2 = _run((d0,), tmp_path / "legacy", max_parallel=2, timeout_s=30.0)
    assert runs2[0].hgc_path is not None


# --------------------------------------------------------------------------- #
# preflight: nxfile rewrite + the serial environment reach the launch
# --------------------------------------------------------------------------- #
def test_preflight_rewrites_the_deck_and_forces_one_thread(
        tmp_path, monkeypatch, stub_launch) -> None:
    xs = tmp_path / "XS.BIN"
    xs.write_bytes(b"library")
    exe = tmp_path / "decart2d1.1m5.exe"
    exe.write_bytes(b"exe")

    opts = L.BatchOptions(preflight=True, xs_lib=xs, exe_sha256=None, xs_sha256=None)
    d0, _ = _designs()
    _reg, runs = _run((d0,), tmp_path, exe=exe, max_parallel=2, timeout_s=30.0,
                      options=opts)

    entry = stub_launch[0]
    assert entry["exe"] == str(exe)
    assert entry["env"]["OMP_NUM_THREADS"] == "1"
    assert str(xs) in entry["deck"].read_text(encoding="utf-8")
    assert "OLD.BIN" not in entry["deck"].read_text(encoding="utf-8")


def test_preflight_hash_mismatch_raises_before_any_launch(tmp_path, stub_launch) -> None:
    xs = tmp_path / "XS.BIN"
    xs.write_bytes(b"library")
    exe = tmp_path / "decart2d1.1m5.exe"
    exe.write_bytes(b"exe")
    opts = L.BatchOptions(preflight=True, xs_lib=xs, exe_sha256="00" * 32,
                          xs_sha256=None)
    d0, _ = _designs()
    with pytest.raises(L.LatticeError, match="SHA-256"):
        _run((d0,), tmp_path, exe=exe, max_parallel=2, timeout_s=30.0, options=opts)
    assert stub_launch == []


# --------------------------------------------------------------------------- #
# task #11 wired: the gates are consumed, not orphaned
# --------------------------------------------------------------------------- #
def test_gate_products_reports_the_delivery_verdict(tmp_path, stub_launch) -> None:
    d0, _ = _designs()
    _reg, runs = _run((d0,), tmp_path, max_parallel=2, timeout_s=30.0)
    report = L.gate_products(runs)
    entry = next(iter(report.values()))
    # the one-state fixture is live but is NOT a 334-state delivery product
    assert entry["verdict"] == "FAIL"
    assert {g["gate"] for g in entry["gates"]} >= {"G-H1", "G-H1b", "G-H1c", "G-H2"}


def test_skip_guard_rejects_a_truncated_product(tmp_path) -> None:
    hgc = tmp_path / "FA_A1.HGC"
    hgc.write_text("%TITL\n CASE :: REFERENCE CASE\n%DIST\n" + "x" * 300,
                   encoding="utf-8")
    assert L._hgc_looks_valid(hgc) is False
    good = tmp_path / "FA_A2.HGC"
    good.write_text(VALID_HGC, encoding="utf-8")
    assert L._hgc_looks_valid(good) is True


def test_authored_design_skip_guard_also_checks_the_gd_census(tmp_path) -> None:
    """An AUTHORED layout must not silently reuse a product whose Gd census
    disagrees; a frozen-template design keeps the historical liveness-only guard."""
    good = tmp_path / "FA_A2.HGC"
    good.write_text(VALID_HGC, encoding="utf-8")          # census 0 (no Gd pins)
    assert L._hgc_looks_valid(good, n_gd=None) is True
    assert L._hgc_looks_valid(good, n_gd=20) is False
