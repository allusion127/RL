"""The v2 (weights-covering) model fingerprint vs the wave fine-tune.

THE REGRESSION pinned here.  Once ``model_fingerprint`` began hashing the members'
``model.pt`` as well as their ``meta.json``, a wave fine-tune changed the
fingerprint — so ``MapCalibration.require_model`` correctly reported "not the
fitted champion" at the campaign's OWN per-wave champion swap, and the uncaught
``ModelMismatchError`` ABORTED every live ``flat_power`` campaign at its first
accepted gate.

Aborting is the right response for a different champion and the wrong one for a
fine-tuned DESCENDANT, and the fingerprint alone cannot tell the two apart.  The
three cases ``CampaignDriver._require_calibration_model`` distinguishes:

(a) ``<run_dir>/models/champion_wave_NN`` — a descendant.  CONTINUE, warn, mark
    the calibration stale, persist it.
(b) any other champion — still a hard ABORT.
(c) no fingerprint on one side — the unchanged ``_content_decidable`` path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.data.map_calibration import (
    ModelMismatchError, model_fingerprint, model_id,
)

from test_map_calibration import (            # noqa: F401 — shared fixtures
    CELL, _doc, _drv, _entry, _fake_model_dir, _resumed, _write,
)


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class _FineTuningModel:
    """A ``FakeModel`` whose ``save()`` writes a REAL, fingerprintable checkpoint
    with MOVED weights — i.e. exactly what a wave fine-tune leaves behind.

    The stock ``FakeModel.save`` writes a single ``fake.json``, which fingerprints
    as the empty string and therefore lands in case (c); it cannot exercise the
    regression at all.
    """

    def __init__(self, weights: bytes = b"FINETUNED" * 32) -> None:
        from test_campaign_stub import FakeModel

        self._inner = FakeModel()
        self._weights = weights

    def __getattr__(self, name):                  # delegate the model protocol
        return getattr(self._inner, name)

    def save(self, path):
        p = Path(path)
        return _fake_model_dir(p.parent, p.name, weights=self._weights)


class _AcceptedGate:
    """The minimum a ``WaveUpdater`` result needs to look like an ACCEPTED gate."""

    accepted = True
    weights_rolled_back = False
    mode = "accept"
    control_spearman = 0.9

    def as_dict(self) -> dict:
        return {"mode": self.mode, "accepted": self.accepted}


class _AcceptingUpdater:
    """Forces the accepted-gate branch: the swap, not the gate, is the subject."""

    def __init__(self, *a, **kw) -> None:
        pass

    def update(self, *a, **kw):
        return _AcceptedGate()


def _fitted_store(tmp_path: Path, weights: bytes = b"FITTED" * 32):
    """A store whose ``map_calibration.json`` is fingerprint-BOUND to a champion.

    Both sides carrying a fingerprint is the exact precondition under which
    ``require_model`` lets content decide — i.e. the configuration in which the
    v2 fingerprint turned a fine-tune into an abort.
    """
    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    fitted_on = _fake_model_dir(tmp_path / "fitted", "20260725_063351",
                                weights=weights)
    _write(store, _doc(cells={CELL: {"f_r": _entry(-0.09),
                                     "node_peak": _entry(-0.147),
                                     "map_cov": _entry(-0.058)}},
                       fit={"model_dir": str(fitted_on),
                            "model_id": model_id(fitted_on),
                            "model_fingerprint": model_fingerprint(fitted_on)}))
    return store, fitted_on


# --------------------------------------------------------------------------- #
# (a) the descendant: CONTINUE, warn, mark stale
# --------------------------------------------------------------------------- #
def test_a_wave_finetune_champion_swap_completes_and_marks_the_calibration_stale(
        tmp_path, monkeypatch):
    """THE bug: a flat_power campaign aborted at its own wave champion swap."""
    from lpopt.search import campaign as C
    from lpopt.search.stub import StubEvaluator
    from test_flatness_campaign import _cfg

    store, fitted_on = _fitted_store(tmp_path)
    said: list[str] = []
    cfg = _cfg(tmp_path / "cfg", store_dir=store, model_dir=fitted_on,
               # the §1.3 map-harvest abort is a different guard; keep it out of
               # the way so a raise here can only be the calibration refusal.
               flatpower_min_map_harvest=0.0)
    drv = C.CampaignDriver(cfg, _FineTuningModel(), lambda w, c: StubEvaluator(),
                           dry_run=True, run_dir=tmp_path / "run", progress=False,
                           log=said.append)
    # construction PROVED the pair — this is a correctly-set-up, legitimate run.
    assert drv._calibrated_model_dir is not None
    assert drv.map_calibration_stale is False

    monkeypatch.setattr(C, "WaveUpdater", _AcceptingUpdater)
    report = drv._run_wave(4, reserve=False)      # the wave AND its champion swap

    # the wave COMPLETED — before the fix this raised ModelMismatchError.
    assert report.gate_accepted is True
    swapped = Path(drv.champion_ckpt)
    assert drv._is_wave_champion(swapped) and swapped.exists()
    # …the fine-tune really did move the weights off the fitted champion, so the
    # fingerprint really did have a difference to report.
    assert model_fingerprint(swapped) not in ("", model_fingerprint(fitted_on))
    # …and the calibration is marked stale, loudly — never silently kept "proven".
    assert drv.map_calibration_stale is True
    assert drv._calibrated_model_dir != str(swapped)
    assert any("WARNING" in m and "APPROXIMATE" in m for m in said)

    # persisted, so a resume of this run cannot read the correction as proven.
    drv._save_state()
    state = json.loads((drv.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["map_calibration_stale"] is True


def test_a_finetuned_descendant_does_not_loosen_the_d1_gate(tmp_path):
    """"Continue instead of abort" buys nothing from the safety gate.

    The D1 F_r correction that was applied stays applied and stays <= 1.70; the
    stale flag is what tells the reader it is now approximate.
    """
    store, fitted_on = _fitted_store(tmp_path)
    drv = _drv(tmp_path / "d1", store_dir=store, model_dir=fitted_on)
    before = float(drv.flat_power_spec.fr_gate)
    assert before <= 1.70

    own = _fake_model_dir(drv.run_dir / "models", "champion_wave_01",
                          weights=b"FINETUNED" * 32)
    assert drv._require_calibration_model(
        own, context="wave 1 champion swap") is False       # reported, not refused
    assert drv.map_calibration_stale is True
    assert float(drv.flat_power_spec.fr_gate) == pytest.approx(before)
    assert float(drv.flat_power_spec.fr_gate) <= 1.70


def test_a_resume_onto_a_finetuned_descendant_keeps_the_stale_flag(tmp_path):
    """(a) at the OTHER swap point — and the ordering trap it exposed.

    ``_load_state`` re-proved the persisted champion and THEN restored
    ``map_calibration_stale`` from the state file, so a resume that DISCOVERED the
    drift had its discovery overwritten by the older persisted value.
    """
    store, fitted_on = _fitted_store(tmp_path)
    own = _fake_model_dir(tmp_path / "resume_ft" / "run" / "models",
                          "champion_wave_02", weights=b"FINETUNED" * 32)
    drv = _resumed(tmp_path / "resume_ft",
                   {"wave_index": 3, "budget_spent": 16,
                    "champion_ckpt": str(own),
                    "map_calibration_stale": False},
                   store_dir=store, model_dir=fitted_on)
    assert drv._load_state() is True                # no refusal
    assert drv.map_calibration_stale is True        # …and not clobbered


# --------------------------------------------------------------------------- #
# (b) a genuinely different champion: still a hard ABORT
# --------------------------------------------------------------------------- #
def test_a_genuinely_different_champion_still_aborts_the_champion_swap(tmp_path):
    """The fix is not a blanket catch.  A champion with no descent argument would
    have the calibration MISAPPLIED, and that is still a refusal."""
    store, fitted_on = _fitted_store(tmp_path)
    drv = _drv(tmp_path / "different", store_dir=store, model_dir=fitted_on)
    other = _fake_model_dir(tmp_path / "other_run", "20260726_121500",
                            weights=b"A DIFFERENT CHAMPION" * 16)
    assert model_fingerprint(other) != model_fingerprint(fitted_on)
    with pytest.raises(ModelMismatchError) as e:
        drv._require_calibration_model(other, context="wave 1 champion swap")
    assert "wave 1 champion swap" in str(e.value)
    # …and it is an ABORT, not a downgrade to a warning.
    assert drv.map_calibration_stale is False


def test_a_foreign_checkpoint_under_the_run_models_dir_still_aborts(tmp_path):
    """Descent is argued from ``_save_champion``'s OWN naming, not from the path.

    Anything else under ``<run_dir>/models/`` could have been put there by hand,
    carries no descent argument, and is refused.
    """
    store, fitted_on = _fitted_store(tmp_path)
    drv = _drv(tmp_path / "intruder", store_dir=store, model_dir=fitted_on)
    intruder = _fake_model_dir(drv.run_dir / "models", "handcopied_champion",
                               weights=b"SOMETHING ELSE" * 32)
    assert drv._is_run_scoped(intruder) and not drv._is_wave_champion(intruder)
    with pytest.raises(ModelMismatchError):
        drv._require_calibration_model(intruder, context="wave 3 champion swap")
    assert drv.map_calibration_stale is False


# --------------------------------------------------------------------------- #
# the docstring the v2 fingerprint made false
# --------------------------------------------------------------------------- #
def test_note_finetuned_weights_docstring_no_longer_claims_the_drift_is_invisible():
    """``_note_finetuned_weights`` claimed the fine-tune was invisible "because
    the member metas are unchanged, so the fingerprint cannot see it".

    Since the fingerprint began covering ``model.pt`` that is simply false — and
    it is the sentence a reader would use to justify NOT routing the swap here,
    which is how the abort survived review.  The docstring must state what the
    fingerprint can and cannot see.
    """
    from lpopt.search.campaign import CampaignDriver

    doc = " ".join((CampaignDriver._note_finetuned_weights.__doc__ or "").split())
    assert "the fingerprint cannot see it" not in doc
    assert "no longer true" in doc                 # the correction is explicit
    assert "model.pt" in doc and "WEIGHTS" in doc  # …and says what it DOES cover
