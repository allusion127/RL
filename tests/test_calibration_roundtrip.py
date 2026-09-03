"""Calibration round-trip through ``PosValCnnBackend.save`` / ``from_dir`` — C2-4.

Closes ``data/reports/fxy_era_adversarial_verification_20260831.md`` §4.4 D and
§9 rank 3 ("C2-4 per-cell 보정 라운드트립 누락 — ``--resume`` 이 수준 보정 없는
모델을 서빙"):

* ``save()`` used to write members + ``ensemble.json`` + ``calibration.json`` +
  ``feature_ood.json`` + ``backend.json`` and nothing else, while ``from_dir``
  reads six more per-cell files (``cell_``/``f_r_``/``cbc_``/``f_q_``/``ao_abs_``/
  ``flatness_calibration.json``) plus ``pinbu_physics.json`` / ``conformal.json``
  from the directory it is handed.
* ``campaign._save_champion`` writes ``<run_dir>/models/champion_wave_NN`` with
  that method and ``--resume`` reloads exactly it, so every resumed campaign
  served an uncalibrated descendant.  r1 was resumed three times.

The tests here are the two halves of the fix: the artefact set survives a
save→load round trip AND the served predictions are identical afterwards; and a
resume onto a checkpoint that lost the set is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.data.map_calibration import ARTIFACT_NAME as MAP_CALIB_NAME  # noqa: E402
from lpopt.data.schema import unpack_pattern                            # noqa: E402
from lpopt.data.store import StoreReader                                # noqa: E402
from lpopt.model.calibrate import CALIB_NAME                            # noqa: E402
from lpopt.model.cell_calibrate import (                                # noqa: E402
    AO_CALIB_NAME, CBC_CALIB_NAME, CELL_CALIB_NAME, FLAT_CALIB_NAME,
    FQ_CALIB_NAME, FR_CALIB_NAME,
)
from lpopt.model.conformal import CONFORMAL_NAME                        # noqa: E402
from lpopt.model.dataset_torch import TARGETS                           # noqa: E402
from lpopt.model.model_api import (                                     # noqa: E402
    CALIBRATION_ARTEFACT_NAMES, PosValCnnBackend,
)
from lpopt.model.net import PosValNet, PosValNetConfig                  # noqa: E402
from lpopt.model.pinbu_physics import PINBU_PHYSICS_NAME                # noqa: E402
from lpopt.model.train import save_member                               # noqa: E402
from lpopt.search.campaign import (                                     # noqa: E402
    CalibrationSetLost, checkpoint_calibration_set,
)
from lpopt.vendor.masterrl.domain import CaseKey                        # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"

_ZMEAN_V3 = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0]
_ZSTD_V3 = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0]

#: The six per-cell files the memo names, in its own order.
_PER_CELL = (CELL_CALIB_NAME, FR_CALIB_NAME, CBC_CALIB_NAME,
             FQ_CALIB_NAME, AO_CALIB_NAME, FLAT_CALIB_NAME)


def _make_ensemble(tmp: Path, name: str = "ens", n: int = 2) -> Path:
    """A tiny synthetic cond_v3 ensemble (same recipe as tests/test_model_api.py)."""
    ens = tmp / name
    cfg = PosValNetConfig()
    for i in range(n):
        seed = 700 + i
        torch.manual_seed(seed)
        net = PosValNet(cfg)
        meta = {
            "net_config": cfg.__dict__,
            "cond_schema": "v3",
            "target_names": list(TARGETS),
            "target_zscore": {"mean": _ZMEAN_V3, "std": _ZSTD_V3},
            "seed": seed,
            "versions": {"torch": torch.__version__},
        }
        save_member(ens / f"member_{seed}", net, meta)
    return ens


def _affine_artifact(schema: str, *, a: float, b: float) -> dict:
    """A minimal but REAL-shaped per-cell affine artifact (``{schema, cells, ...}``).

    Deliberately non-identity (``a != 1`` / ``b != 0``) and with a per-library
    global fallback, so a lost artifact CHANGES the served numbers rather than
    being invisible — that is what makes the prediction-equality assertion below
    a real test of the round trip.
    """
    return {
        "schema": schema,
        "bin_width": 0.05,
        "cells": {"f121_e5.40": {"a": a, "b": b, "n": 64}},
        "global_by_library": {"ga80": {"a": a, "b": b, "n": 512}},
    }


def _flatness_artifact() -> dict:
    return {
        "schema": "cell_flatness_intercept_v1",
        "bin_width": 0.05,
        "targets": {
            "node_peak": {"cells": {"f121_e5.40": {"b": -0.011, "n": 40}},
                          "global_by_library": {"ga80": {"b": -0.008, "n": 300}}},
            "map_cov": {"cells": {"f121_e5.40": {"b": 0.004, "n": 40}},
                        "global_by_library": {"ga80": {"b": 0.003, "n": 300}}},
        },
    }


def _write_calibration_set(d: Path) -> dict[str, dict]:
    """Write one of every calibration artifact into a checkpoint dir."""
    payloads = {
        CALIB_NAME: {
            "targets": list(TARGETS), "split": "S1", "n_members": 2,
            "isotonic": {n: {"x": [0.0, 1.0], "y": [0.0, 2.0]} for n in TARGETS},
            "platt": {"coef": 1.3, "intercept": -0.2, "degenerate": False},
            "n_val_used": 123,
        },
        CELL_CALIB_NAME: _affine_artifact("cell_cyclen_affine_v1", a=0.97, b=8.5),
        FR_CALIB_NAME: _affine_artifact("cell_f_r_affine_v1", a=0.98, b=0.021),
        CBC_CALIB_NAME: _affine_artifact("cell_cbc_max_affine_v1", a=1.02, b=-19.0),
        FQ_CALIB_NAME: _affine_artifact("cell_f_q_affine_v1", a=0.99, b=0.017),
        AO_CALIB_NAME: _affine_artifact("cell_ao_abs_affine_v1", a=1.01, b=-0.004),
        FLAT_CALIB_NAME: _flatness_artifact(),
        PINBU_PHYSICS_NAME: {
            "schema": "pinbu_physics_affine_v1", "library_id": "ga80",
            "a": 1.03, "b": -0.4, "global_k_peak": 1.18,
            "k_peak_by_feed": {"121": {"k_peak": 1.19, "n": 90}},
            "power_mw": 3983.0, "hm_mtu": 104.8, "n": 200,
        },
        CONFORMAL_NAME: {
            "schema": "split_conformal_v1", "bin_width": 0.25, "alpha": 0.1,
            "per_target": {"cyclen": {"global": 11.0, "cells": {}}},
        },
    }
    d.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (d / name).write_text(json.dumps(payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    return payloads


@pytest.fixture(scope="module")
def store_reader():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    return StoreReader(STORE)


def _serve_batch(reader, n: int = 12):
    """A handful of real patterns + their case key, for a served comparison."""
    df = reader.records
    k = df[df["pattern"].notna() & df["case_pair"].notna() & df["feed"].notna()]
    if k.empty:
        pytest.skip("no usable store rows")
    first = k.iloc[0]
    k = k[(k["case_pair"] == str(first["case_pair"]))
          & (k["feed"] == int(first["feed"]))].head(n)
    pats = [unpack_pattern(str(p)) for p in k["pattern"]]
    case = CaseKey(pair=str(first["case_pair"]), feed=int(first["feed"]))
    return pats, case


# --------------------------------------------------------------------------- #
# 1. the artefact set survives save() -> from_dir()  (memo §4.4 D)
# --------------------------------------------------------------------------- #
def test_campaign_artefact_names_are_pinned_to_model_api() -> None:
    """``campaign._CALIBRATION_ARTEFACTS`` is a literal (so campaign.py loads
    without torch); this is the pin that stops it drifting from the real names."""
    from lpopt.search import campaign as _c

    assert tuple(_c._CALIBRATION_ARTEFACTS) == tuple(CALIBRATION_ARTEFACT_NAMES)
    assert set(CALIBRATION_ARTEFACT_NAMES) == {
        CALIB_NAME, *_PER_CELL, PINBU_PHYSICS_NAME, CONFORMAL_NAME}


def test_save_round_trips_every_calibration_artefact(tmp_path, store_reader) -> None:
    """The C2-4 defect, asserted directly: every file ``from_dir`` reads must come
    back out of ``save()``.  Before the fix only ``calibration.json`` did."""
    src = _make_ensemble(tmp_path)
    payloads = _write_calibration_set(src)
    # ...and a map_calibration.json, which the backend holds NO in-memory copy of.
    (src / MAP_CALIB_NAME).write_text(
        json.dumps({"schema": "map_calibration_v1", "cells": {}}, indent=2),
        encoding="utf-8")

    backend = PosValCnnBackend.from_dir(src, store_dir=STORE, library_id="ga80")
    out = tmp_path / "derived"
    backend.save(out)

    for name in CALIBRATION_ARTEFACT_NAMES:
        assert (out / name).is_file(), f"{name} did NOT survive save() (defect C2-4)"
        assert json.loads((out / name).read_text(encoding="utf-8")) == payloads[name]
    # the copy-only artifact came across byte-for-byte
    assert (out / MAP_CALIB_NAME).read_bytes() == (src / MAP_CALIB_NAME).read_bytes()
    # and the on-disk view campaign.py uses agrees with both dirs
    assert checkpoint_calibration_set(out) == checkpoint_calibration_set(src)
    assert checkpoint_calibration_set(out) == set(CALIBRATION_ARTEFACT_NAMES)


def test_reloaded_checkpoint_serves_identical_predictions(tmp_path, store_reader) -> None:
    """save -> load -> IDENTICAL predictions on a fixture.

    The artefact-presence test above would pass on an empty file; this is the one
    that says the LEVEL correction actually still applies.  The control at the
    bottom shows the calibrations are non-trivial: a checkpoint saved without them
    (the pre-fix behaviour, reproduced by deleting the files) serves DIFFERENT
    numbers, which is exactly the silent gate shift the memo describes.
    """
    src = _make_ensemble(tmp_path)
    _write_calibration_set(src)
    pats, case = _serve_batch(store_reader)

    original = PosValCnnBackend.from_dir(src, store_dir=STORE, library_id="ga80")
    base = original.predict(pats, case, float(case.feed))

    out = tmp_path / "derived"
    original.save(out)
    reloaded = PosValCnnBackend.from_dir(out, store_dir=STORE, library_id="ga80")
    again = reloaded.predict(pats, case, float(case.feed))

    np.testing.assert_array_equal(base.mean, again.mean)
    np.testing.assert_array_equal(base.epistemic_std, again.epistemic_std)
    np.testing.assert_array_equal(base.calibrated_std, again.calibrated_std)
    # the Platt half of calibration.json rides on this one
    np.testing.assert_array_equal(
        original.predict_convergence(pats, case, float(case.feed)),
        reloaded.predict_convergence(pats, case, float(case.feed)))
    # every artifact is loaded, not merely present on disk
    for attr in ("calibration", "cell_calibration", "fr_calibration",
                 "cbc_calibration", "fq_calibration", "ao_calibration",
                 "flatness_calibration", "pinbu_physics", "conformal"):
        assert getattr(reloaded, attr) == getattr(original, attr)
        assert getattr(reloaded, attr) is not None

    # CONTROL — the pre-fix checkpoint (calibration files stripped) is NOT the
    # same model.  Without this the equality above could hold vacuously.
    stripped = tmp_path / "stripped"
    original.save(stripped)
    for name in _PER_CELL:
        (stripped / name).unlink()
    uncal = PosValCnnBackend.from_dir(stripped, store_dir=STORE, library_id="ga80")
    assert uncal.cell_calibration is None and uncal.fr_calibration is None
    assert not np.array_equal(base.mean, uncal.predict(pats, case, float(case.feed)).mean), \
        "the fixture calibrations are inert; the round-trip assertion proves nothing"


def test_backend_without_calibration_still_saves_nothing(tmp_path, store_reader) -> None:
    """Backward compat: COPIED, never synthesised.  A checkpoint that never had a
    calibration must not gain an (invented) one — same contract
    ``_save_ensemble_meta`` keeps for the f_xy bar."""
    src = _make_ensemble(tmp_path)
    backend = PosValCnnBackend.from_dir(src, store_dir=STORE, library_id="ga80")
    out = tmp_path / "derived"
    backend.save(out)
    assert checkpoint_calibration_set(out) == frozenset()
    assert (out / "backend.json").is_file()          # the rest of save() is unchanged


# --------------------------------------------------------------------------- #
# 2. resume refuses a checkpoint that lost the set  (memo §9 rank 3)
# --------------------------------------------------------------------------- #
class _Driver:
    """The two guard methods under test, bound to a stub driver.

    Constructing a real :class:`CampaignDriver` needs a deck, a store writer and a
    model; the guards themselves depend on exactly four attributes, so binding
    them here exercises the LOGIC cheaply and in isolation.

    This stub proves nothing about WIRING — section 3 below drives a real
    ``CampaignDriver`` into both call sites for that.
    """

    def __init__(self, launch: Path, *, allow_uncalibrated: bool = False) -> None:
        from lpopt.search.campaign import CampaignDriver

        self.champion_ckpt = str(launch)
        #: the driver reassigns ``champion_ckpt`` on resume; the refusal message
        #: quotes THIS one, so the reference checkpoint it names stays right.
        self._champion_ckpt_launch = str(launch)
        self._calibration_set_launch = checkpoint_calibration_set(launch)
        self.allow_uncalibrated = allow_uncalibrated
        self.logs: list[str] = []
        self._log = self.logs.append
        self._assert_calibration_set = CampaignDriver._assert_calibration_set.__get__(self)
        self._require_calibration_set = CampaignDriver._require_calibration_set.__get__(self)


def test_resume_refuses_a_checkpoint_missing_its_calibration(tmp_path) -> None:
    launch = tmp_path / "launch"
    _write_calibration_set(launch)
    wave = tmp_path / "champion_wave_11"
    _write_calibration_set(wave)
    for name in _PER_CELL:                       # the pre-fix save() outcome
        (wave / name).unlink()

    driver = _Driver(launch)
    with pytest.raises(CalibrationSetLost) as exc:
        driver._require_calibration_set(wave, context="resume")
    message = str(exc.value)
    assert "resume" in message
    for name in _PER_CELL:
        assert name in message
    assert "C2-4" in message


def test_resume_accepts_a_complete_checkpoint(tmp_path) -> None:
    launch = tmp_path / "launch"
    _write_calibration_set(launch)
    wave = tmp_path / "champion_wave_11"
    _write_calibration_set(wave)
    driver = _Driver(launch)
    driver._require_calibration_set(wave, context="resume")     # no raise
    assert any("CALIBRATION" in line for line in driver.logs)


def test_allow_uncalibrated_downgrades_the_refusal_to_a_warning(tmp_path) -> None:
    launch = tmp_path / "launch"
    _write_calibration_set(launch)
    wave = tmp_path / "champion_wave_11"
    _write_calibration_set(wave)
    (wave / CELL_CALIB_NAME).unlink()

    driver = _Driver(launch, allow_uncalibrated=True)
    driver._require_calibration_set(wave, context="resume")     # no raise
    assert any("WARNING" in line and CELL_CALIB_NAME in line for line in driver.logs)


def test_guards_are_inert_for_a_launch_champion_without_calibration(tmp_path) -> None:
    """Backward compat: a deck whose champion ships no calibration is unaffected —
    the same contract the ``serve_sigma`` guard keeps for a pre-`s1j` checkpoint."""
    launch = _make_ensemble(tmp_path, name="bare")
    wave = tmp_path / "champion_wave_01"
    wave.mkdir()
    driver = _Driver(launch)
    assert driver._calibration_set_launch == frozenset()
    driver._require_calibration_set(wave, context="resume")     # no raise
    driver._assert_calibration_set(wave, context="wave 1 champion")


def test_gaining_a_calibration_artefact_is_not_a_loss(tmp_path) -> None:
    """Only LOSS is fatal; a re-fit that ADDS an artifact must not abort the run."""
    launch = tmp_path / "launch"
    _write_calibration_set(launch)
    (launch / CONFORMAL_NAME).unlink()
    wave = tmp_path / "champion_wave_02"
    _write_calibration_set(wave)                  # carries conformal.json too
    _Driver(launch)._assert_calibration_set(wave, context="wave 2 champion")


def test_save_champion_assertion_catches_the_pre_fix_write(tmp_path, store_reader) -> None:
    """End to end on the write side: a checkpoint written the way the pre-fix
    ``save()`` wrote one trips :meth:`_assert_calibration_set` at the moment it is
    created, naming the wave — instead of surfacing 12 acquisition calls later."""
    src = _make_ensemble(tmp_path)
    _write_calibration_set(src)
    backend = PosValCnnBackend.from_dir(src, store_dir=STORE, library_id="ga80")
    driver = _Driver(src)

    good = tmp_path / "champion_wave_03"
    backend.save(good)
    driver._assert_calibration_set(good, context="wave 3 champion")   # no raise

    pre_fix = tmp_path / "champion_wave_04"
    backend.save(pre_fix)
    for name in _PER_CELL + (PINBU_PHYSICS_NAME, CONFORMAL_NAME):
        (pre_fix / name).unlink()
    with pytest.raises(CalibrationSetLost, match="wave 4 champion"):
        driver._assert_calibration_set(pre_fix, context="wave 4 champion")


# --------------------------------------------------------------------------- #
# 3. the guards are actually WIRED — real CampaignDriver, real call sites
#
# The stub above proves the guard LOGIC.  It cannot prove that anything calls
# it: with the calls in ``_save_champion`` and ``_load_state`` replaced by
# ``pass``, every test above still passes — and "the guard was not there" is this
# defect's own history.  These five drive a REAL driver into the two call sites,
# the same way ``tests/test_campaign_stub.py`` does for the D3 serve-sigma twin.
#
# The harness is local rather than imported from test_campaign_stub.py: that file
# belongs to another track, and a cross-test-module import would make this
# coverage hostage to its refactors.
# --------------------------------------------------------------------------- #
class _CalibModel:
    """A torch-free PositionValueModel double whose ``save()`` emits a CHOSEN
    calibration set — the knob the pre-fix ``save()`` bug is reproduced with."""

    def __init__(self, emit: tuple[str, ...] = CALIBRATION_ARTEFACT_NAMES) -> None:
        self.emit = tuple(emit)

    def predict(self, patterns, case, cell=0.0):
        from lpopt.vendor.masterrl.surrogate import SurrogatePrediction

        n = len(list(patterns))
        mean = np.tile([1.50, 2.10, 1400.0, 690.0, 0.10, np.nan, np.nan], (n, 1))
        std = np.tile([0.02, 0.05, 15.0, 3.0, 0.02, np.nan, np.nan], (n, 1))
        return SurrogatePrediction(mean, std.copy(), std.copy())

    def predict_convergence(self, patterns, case, cell=0.0):
        return np.ones(len(list(patterns)), dtype=float)

    def position_values(self, pattern, case, cell=0.0):
        return None

    def finetune(self, new, replay, epochs=3, seed=0):
        return {"refit": False, "n_new": len(list(new))}

    def save(self, path):
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "backend.json").write_text("{}", encoding="utf-8")
        for name in self.emit:
            (out / name).write_text("{}", encoding="utf-8")
        return out


def _campaign_cfg(tmp_path: Path, model_dir: Path):
    from lpopt.config import (
        AcquisitionConfig, CaseConfig, DataConfig, ExtractConfig, FlowConfig,
        FuelConfig, LpoptConfig, MasterConfig, ModelConfig, ProduceConfig,
        RemoteConfig, SearchConfig, VerifyConfig,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    deck = tmp_path / "lpopt.inp"
    deck.write_text("# fake deck\n", encoding="utf-8")
    return LpoptConfig(
        flow=FlowConfig(), remote=RemoteConfig(), master=MasterConfig(),
        verify=VerifyConfig(), data=DataConfig(),
        case=CaseConfig(pair="K1_K2", feed=121), fuel=FuelConfig(),
        extract=ExtractConfig(), produce=ProduceConfig(), search=SearchConfig(),
        acquisition=AcquisitionConfig(budget=8, gate_skill_halt=-2.0),
        model=ModelConfig(store_dir=str(STORE), model_dir=str(model_dir)),
        source_path=deck,
    )


def _real_driver(tmp_path: Path, launch: Path, *, emit=CALIBRATION_ARTEFACT_NAMES,
                 resume: bool = False, allow_uncalibrated: bool = False, log=None):
    from lpopt.search.campaign import CampaignDriver
    from lpopt.search.stub import StubEvaluator

    stub = StubEvaluator()
    return CampaignDriver(
        _campaign_cfg(tmp_path, launch), _CalibModel(emit),
        lambda worker_id, cpu_core: stub, dry_run=True,
        run_dir=tmp_path / "run", progress=False, resume=resume,
        allow_uncalibrated=allow_uncalibrated,
        backend_factory=lambda ckpt: _CalibModel(emit),
        log=(log if log is not None else (lambda m: None)),
    )


_NEEDS_STORE = pytest.mark.skipif(
    not (STORE / "records.parquet").is_file(), reason="no store present")


@_NEEDS_STORE
def test_driver_freezes_the_launch_calibration_set(tmp_path) -> None:
    """``__init__`` must capture the LAUNCH champion's set — the reference every
    later assertion is made against — and the launch PATH alongside it."""
    launch = tmp_path / "launch"
    _write_calibration_set(launch)
    drv = _real_driver(tmp_path / "freeze", launch)
    assert drv._calibration_set_launch == frozenset(CALIBRATION_ARTEFACT_NAMES)
    assert Path(drv._champion_ckpt_launch) == launch


@_NEEDS_STORE
def test_save_champion_call_site_refuses_a_stripped_checkpoint(tmp_path) -> None:
    """THE WIRING TEST for the write side: ``_save_champion`` itself must raise.

    The model saves the pre-fix subset (``calibration.json`` and nothing else),
    which is exactly what ``PosValCnnBackend.save`` used to write.  With the call
    in ``_save_champion`` deleted this test fails; every stub-driver test above
    still passes.
    """
    launch = tmp_path / "launch"
    _write_calibration_set(launch)
    drv = _real_driver(tmp_path / "strip", launch, emit=(CALIB_NAME,))
    drv.wave_index = 4
    with pytest.raises(CalibrationSetLost) as exc:
        drv._save_champion()
    message = str(exc.value)
    assert "wave 4 champion" in message and "C2-4" in message
    for name in _PER_CELL:
        assert name in message
    # the refusal names the LAUNCH champion as the reference checkpoint, which is
    # what an operator re-saves from.
    assert str(launch) in message


@_NEEDS_STORE
def test_save_champion_accepts_a_complete_checkpoint(tmp_path) -> None:
    """The control: with the round-trip fix in place the same call site is quiet."""
    launch = tmp_path / "launch"
    _write_calibration_set(launch)
    drv = _real_driver(tmp_path / "ok", launch)
    drv.wave_index = 2
    out = drv._save_champion()
    assert checkpoint_calibration_set(out) == frozenset(CALIBRATION_ARTEFACT_NAMES)


@_NEEDS_STORE
def test_load_state_call_site_refuses_a_stripped_checkpoint(tmp_path) -> None:
    """THE WIRING TEST for the read side: ``_load_state`` must raise BEFORE the
    reload, so the uncalibrated descendant is never served at all."""
    base = tmp_path / "resume"
    launch = tmp_path / "launch"
    _write_calibration_set(launch)

    drv = _real_driver(base, launch)
    drv.wave_index = 7
    ckpt = drv._save_champion()
    drv.champion_ckpt = str(ckpt)
    drv._save_state()
    for name in _PER_CELL:                       # a pre-fix checkpoint on disk
        (ckpt / name).unlink()

    resumed = _real_driver(base, launch, resume=True)
    with pytest.raises(CalibrationSetLost) as exc:
        resumed._load_state()
    assert "resume" in str(exc.value)


@_NEEDS_STORE
def test_allow_uncalibrated_lets_a_run_resume_and_keep_saving(tmp_path) -> None:
    """The hatch must be SYMMETRIC.

    A run resumed with ``--allow-uncalibrated`` serves an uncalibrated backend by
    construction, so every champion it writes is artefact-less too.  With the
    hatch on the read side only, the first accepted gate raised out of
    ``_run_wave`` — after that wave's MASTER budget was spent — while the hatch's
    own message promised "the run continues".
    """
    base = tmp_path / "hatch"
    launch = tmp_path / "launch"
    _write_calibration_set(launch)

    seed = _real_driver(base, launch)
    seed.wave_index = 3
    ckpt = seed._save_champion()
    seed.champion_ckpt = str(ckpt)
    seed._save_state()
    for name in _PER_CELL:
        (ckpt / name).unlink()

    lines: list[str] = []
    resumed = _real_driver(base, launch, emit=(CALIB_NAME,), resume=True,
                           allow_uncalibrated=True, log=lines.append)
    assert resumed._load_state() is True
    assert any("WARNING" in line and "CALIBRATION" in line for line in lines)
    # ...and the wave champion it then writes does NOT abort the run
    resumed.wave_index = 4
    out = resumed._save_champion()
    assert checkpoint_calibration_set(out) != frozenset(CALIBRATION_ARTEFACT_NAMES)
    assert any("WARNING" in line and "wave 4 champion" in line for line in lines)
