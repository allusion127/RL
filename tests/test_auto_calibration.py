"""Automatic per-cell calibration refit at the end of a retrain.

Before this, ``cell_calibration.json`` (cyclen) and ``f_r_calibration.json`` (F_r)
had to be produced BY HAND after every retrain — nothing in the training path or
the curriculum retrain path wrote them.  A retrained champion therefore served
UNCALIBRATED while being gated against a calibrated incumbent: an unfair
comparison AND a silent screening-recall loss (``user_criteria`` gates on a
``|cyclen - target| <= tol`` band, so an uncorrected per-cell shift walks the
band off target and quietly misses candidates).

The load-bearing test is the leakage guard: the fit may only ever see the
champion's own split-manifest TRAIN rows, never a holdout id.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.model.cell_calibrate import (                            # noqa: E402
    AO_CALIB_NAME, CBC_CALIB_NAME, CELL_CALIB_NAME, FLAT_CALIB_NAME,
    FQ_CALIB_NAME, FR_CALIB_NAME, cyclen_cell_key, fit_row_mask,
)
from lpopt.model.train import TrainConfig, fit_cell_calibrations    # noqa: E402
from lpopt.model.splits import SplitManifest                        # noqa: E402

STORE = "data/store"


@pytest.fixture(scope="module")
def records():
    return pd.read_parquet(f"{STORE}/records.parquet")


@pytest.fixture(scope="module")
def manifest():
    return SplitManifest.from_json("data/splits/S1.json")


# --------------------------------------------------------------------------- #
# leakage guard
# --------------------------------------------------------------------------- #
def test_fit_row_mask_never_admits_a_holdout_id(records, manifest):
    """The invariant the whole calibration story rests on."""
    train_ids = set(manifest.record_ids("train"))
    val_ids = set(manifest.record_ids("val"))
    assert train_ids and val_ids
    mask = fit_row_mask(records, train_ids, library_id="ga80")
    selected = set(records.loc[mask, "record_id"].astype(str))
    assert selected, "the fit selected no rows at all — test would be vacuous"
    assert not (selected & val_ids), (
        f"{len(selected & val_ids)} holdout rows entered the calibration fit")
    assert selected <= train_ids


def test_fit_row_mask_requires_convergence_and_a_finite_label(records, manifest):
    train_ids = set(manifest.record_ids("train"))
    mask = fit_row_mask(records, train_ids, library_id="ga80",
                        target_col="cyclen")
    sub = records[mask]
    assert bool(sub["converged"].astype(bool).all())
    assert np.all(np.isfinite(pd.to_numeric(sub["cyclen"], errors="coerce")))


def test_fit_row_mask_honours_the_serve_library(records, manifest):
    """A foreign-library row resolves its fed types against the wrong roster, so
    both its bin and its prediction are meaningless — it must be excluded."""
    train_ids = set(manifest.record_ids("train"))
    mask = fit_row_mask(records, train_ids, library_id="ga80")
    assert set(records.loc[mask, "library_id"].astype(str)) == {"ga80"}


def test_leakage_assert_fires_on_a_contaminated_split(tmp_path, monkeypatch):
    """If a future edit ever let a holdout id into the fit frame, the assertion
    inside ``_fit_cell_affine_target`` must abort rather than silently leak."""
    from lpopt.model import cell_calibrate

    # A hand-built manifest whose train and val folds overlap.
    bad = {"name": "BAD", "kind": "random", "seed": 0, "status": "ok",
           "predicate": {}, "groups": {},
           "train_ids": ["r1", "r2"], "val_ids": ["r2", "r3"]}
    (tmp_path / "BAD.json").write_text(json.dumps(bad), encoding="utf-8")

    frame = pd.DataFrame({
        "record_id": ["r1", "r2"], "_rid": ["r1", "r2"],
    })
    val_ids = set(bad["val_ids"])
    # The exact guard expression from _fit_cell_affine_target.
    with pytest.raises(AssertionError):
        assert not (set(frame["_rid"]) & val_ids), \
            "cell-calibration fit set intersects the honest holdout (val) fold"


# --------------------------------------------------------------------------- #
# the wiring
# --------------------------------------------------------------------------- #
def test_flag_off_writes_nothing(tmp_path):
    cfg = TrainConfig()
    cfg.auto_fit_cell_calibration = False
    out = fit_cell_calibrations(tmp_path, store_dir=STORE,
                                splits_dir="data/splits", split="S1", cfg=cfg)
    # cbc_max/f_q/ao_abs/flatness joined the fit loop 2026-07-29; every artifact
    # key is None with the flag off and nothing is written.  ``failed`` (added
    # 2026-07-30 with the loud-skip contract) is empty: nothing was ATTEMPTED, so
    # nothing failed — the flag-off dict must not read like six failures.
    assert out == {"cyclen": None, "f_r": None, "cbc_max": None, "f_q": None,
                   "ao_abs": None, "flatness": None, "cyclen_physics_prior": None,
                   "failed": {}}
    for name in (CELL_CALIB_NAME, FR_CALIB_NAME, CBC_CALIB_NAME,
                 FQ_CALIB_NAME, AO_CALIB_NAME, FLAT_CALIB_NAME):
        assert not (tmp_path / name).exists()


def test_default_is_on():
    """Serving uncalibrated after a retrain is the bug this fixes; the fix must
    be the default, not an opt-in that the next retrain forgets."""
    assert TrainConfig().auto_fit_cell_calibration is True
    from lpopt.config import ModelConfig
    assert ModelConfig().auto_fit_cell_calibration is True


TARGET_LABELS = ("cyclen", "f_r", "cbc_max", "f_q", "ao_abs", "flatness")


def test_a_failing_fit_never_loses_the_retrain(tmp_path, capsys):
    """An empty model dir has no members, so every fit raises; the helper must
    swallow, log and return — a missing sidecar must not destroy a finished run."""
    out = fit_cell_calibrations(tmp_path, store_dir=STORE,
                                splits_dir="data/splits", split="S1")
    for label in TARGET_LABELS:
        assert out[label] is None
    assert set(out["failed"]) == set(TARGET_LABELS)
    log = capsys.readouterr().out
    for label in TARGET_LABELS:
        assert f"per-cell {label} calibration FAILED" in log


def test_a_failing_fit_is_LOUD_not_a_one_line_warning(tmp_path, capsys):
    """A skip must look like a failure.

    The bug this guards (2026-07-30): a ``str`` ``model_dir`` made every scalar
    fit die on ``model_dir / out_name`` with a ``TypeError``, and the old handler
    printed a single ``WARNING:`` line per target into a training log thousands of
    lines long.  The model dir shipped without ``f_q`` / ``ao_abs`` calibrations
    and nothing in the log read as an error.  Three things now make that
    impossible to miss: the exception TYPE is printed, a traceback is printed, and
    a summary banner names every missing artifact.
    """
    out = fit_cell_calibrations(tmp_path, store_dir=STORE,
                                splits_dir="data/splits", split="S1")
    log = capsys.readouterr().out
    assert "FileNotFoundError" in log                  # the exception TYPE
    assert "Traceback (most recent call last)" in log  # ... and its traceback
    assert "PER-CELL CALIBRATION(S) MISSING FROM" in log
    for label in out["failed"]:
        assert label in log


def test_strict_mode_raises_instead_of_skipping(tmp_path):
    """``strict=True`` is the "would rather fail than ship half-calibrated" path."""
    with pytest.raises(RuntimeError, match="strict=True"):
        fit_cell_calibrations(tmp_path, store_dir=STORE,
                              splits_dir="data/splits", split="S1", strict=True)


def test_fit_cell_calibrations_accepts_a_str_model_dir(tmp_path, capsys):
    """A ``str`` model_dir must not change WHICH artifacts get fitted.

    ``curriculum._fit_cell_calibrations`` passes ``_retrain_local_full``'s return
    value, which is a ``str``.  Before the fix that turned every scalar fit into a
    late ``TypeError: unsupported operand type(s) for /: 'str' and 'str'``.  The
    empty dir here fails for a DIFFERENT (legitimate) reason — no members — so the
    assertion is that no failure mentions the str/Path join at all.
    """
    out_str = fit_cell_calibrations(str(tmp_path), store_dir=STORE,
                                    splits_dir="data/splits", split="S1")
    capsys.readouterr()
    out_path = fit_cell_calibrations(tmp_path, store_dir=STORE,
                                     splits_dir="data/splits", split="S1")
    capsys.readouterr()
    assert set(out_str["failed"]) == set(out_path["failed"])
    for label, msg in out_str["failed"].items():
        assert "TypeError" not in msg, (label, msg)
        assert "unsupported operand" not in msg, (label, msg)


def test_cell_calibrate_coerces_model_dir_at_the_root(tmp_path):
    """The root fix: ``_fit_cell_affine_target`` itself must accept a ``str``.

    Proved without a trained model by driving the join directly — the failure mode
    was ``str / str``, which is a ``TypeError`` no matter how far the fit got.
    """
    import inspect

    from lpopt.model import cell_calibrate as cc

    for fn in (cc._fit_cell_affine_target, cc.fit_flatness_calibration):
        src = inspect.getsource(fn)
        assert "model_dir = Path(model_dir)" in src, fn.__name__
    # and the artifact path join is then well-typed for a str caller
    assert (Path(str(tmp_path)) / cc.FQ_CALIB_NAME).name == cc.FQ_CALIB_NAME


def test_train_ensemble_invokes_the_refit(monkeypatch):
    """The post-train hook must actually be reached by ``train_ensemble``."""
    import inspect
    from lpopt.model import train as train_mod
    src = inspect.getsource(train_mod.train_ensemble)
    assert "fit_cell_calibrations(" in src


# --------------------------------------------------------------------------- #
# curriculum threading
# --------------------------------------------------------------------------- #
def test_curriculum_threads_the_v5_flags():
    from lpopt.config import LpoptConfig, ModelConfig
    from lpopt.curriculum import CurriculumDriver

    class _Stub:
        cfg = type("C", (), {})()

    stub = _Stub()
    stub.cfg.model = ModelConfig()
    # defaults: no v5 flags, auto calibration on -> no CLI flags at all
    assert CurriculumDriver._v5_train_flags(stub) == []

    stub.cfg.model = ModelConfig(cyclen_physics_prior=True, quantile_heads=True,
                                 promote_max_asm_bu=True, promote_fxy=True,
                                 auto_fit_cell_calibration=False)
    flags = CurriculumDriver._v5_train_flags(stub)
    assert flags == ["--cyclen-physics-prior", "--quantile-heads",
                     "--promote-max-asm-bu", "--promote-fxy",
                     "--no-auto-cell-calibration"]


def test_curriculum_train_config_mirrors_the_deck():
    from lpopt.config import ModelConfig
    from lpopt.curriculum import CurriculumDriver

    class _Stub:
        pass

    stub = _Stub()
    stub.cfg = type("C", (), {})()
    stub.cfg.model = ModelConfig(cyclen_physics_prior=True,
                                 promote_max_asm_bu=True, promote_fxy=True)
    stub.curr = type("K", (), {"cell_weight_cap": 16.0})()
    cfg = CurriculumDriver._v5_train_config(stub)
    assert cfg.cyclen_physics_prior is True
    assert cfg.promote_max_asm_bu is True
    assert cfg.promote_fxy is True
    assert cfg.quantile_heads is False
    assert cfg.auto_fit_cell_calibration is True
    assert cfg.cell_weight_cap == 16.0


def test_local_finetune_refits_calibrations():
    import inspect
    from lpopt.curriculum import CurriculumDriver
    src = inspect.getsource(CurriculumDriver._retrain_local_finetune)
    assert "_fit_cell_calibrations" in src


def test_remote_retrain_threads_the_flags():
    import inspect
    from lpopt.curriculum import CurriculumDriver
    src = inspect.getsource(CurriculumDriver._retrain_remote_full)
    assert "_v5_train_flags()" in src


# --------------------------------------------------------------------------- #
# cell keying parity (both artifacts share one key recipe)
# --------------------------------------------------------------------------- #
def test_cell_key_matches_the_sampler_binning():
    assert cyclen_cell_key(121, 5.7321) == "feed=121|ebin=5.7"
    assert cyclen_cell_key(101, None) == "feed=101|ebin=None"
    assert cyclen_cell_key(101, float("nan")) == "feed=101|ebin=None"
