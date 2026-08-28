"""Tests for the cell-sequential curriculum driver (plan section 12.2/12.3).

Fast + hermetic: a StubEvaluator-style FakeVerifier (deterministic FOM, no
MASTER) plus a FakeModel (no torch) drive the state machine, so the whole
``ensure_types -> blind_probe -> produce_cell -> retrain -> validate_gate``
cycle runs in milliseconds.  Ring-order determinism and pair selection are
unit-tested directly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lpopt.config import load_config
from lpopt import curriculum as C
from lpopt.vendor.masterrl.domain import FOM
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction
from lpopt.model.model_api import ExtraPrediction
from lpopt.search.verify import WaveOutcome

DECK = Path(__file__).resolve().parents[1] / "lpopt.inp"

TARGET_COLS = {"f_r": 0, "cbc_max": 1, "f_q": 2, "cyclen": 3, "ao_abs": 4, "max_pin_burnup": 6}


# --------------------------------------------------------------------------- #
# deterministic truth shared by the fake model + fake verifier
# --------------------------------------------------------------------------- #
def _truth(pat) -> dict:
    h = int(hashlib.sha1(pat.canonical().encode()).hexdigest(), 16)
    r = (h % 1000) / 1000.0
    r2 = ((h // 1000) % 1000) / 1000.0
    return {
        "f_r": 1.4 + 0.4 * r,
        "cbc_max": 1200.0 + 400.0 * r2,
        "f_q": 2.0 + 0.5 * r,
        "cyclen": 600.0 + 80.0 * r2,
        "ao_abs": 0.10 + 0.2 * r,
        "max_pin_burnup": 60.0 + 30.0 * r2,
        "discharge_burnup": 45.0 + 15.0 * r,
    }


class FakeModel:
    """Minimal PositionValueModel: ``perfect`` predicts the truth (Spearman 1),
    ``constant`` predicts a fixed vector (Spearman undefined -> gate fail)."""

    target_names = ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                    "discharge_burnup", "max_pin_burnup")

    def __init__(self, mode: str = "perfect") -> None:
        self.mode = mode

    def _mean_row(self, pat) -> np.ndarray:
        row = np.full(7, np.nan)
        if self.mode == "constant":
            vals = {"f_r": 1.5, "cbc_max": 1300.0, "f_q": 2.2,
                    "cyclen": 640.0, "ao_abs": 0.2, "max_pin_burnup": 70.0}
        else:
            vals = _truth(pat)
        for name, col in TARGET_COLS.items():
            row[col] = vals[name]
        return row

    def predict(self, patterns, case, cell=0.0) -> SurrogatePrediction:
        n = len(patterns)
        mean = np.stack([self._mean_row(p) for p in patterns]) if n else np.zeros((0, 7))
        calib = np.full((n, 7), 0.05)
        calib[:, 1] = 20.0  # cbc scale
        calib[:, 3] = 5.0   # cyclen scale
        calib[:, 6] = 1.0
        calib[:, 5] = np.nan
        return SurrogatePrediction(mean=mean, epistemic_std=calib.copy(),
                                   calibrated_std=calib)

    def predict_extra(self, patterns, case, cell=0.0) -> ExtraPrediction:
        n = len(patterns)
        if self.mode == "constant":
            mean = np.full((n, 1), 50.0)
        else:
            mean = np.array([[_truth(p)["discharge_burnup"]] for p in patterns]) \
                if n else np.zeros((0, 1))
        return ExtraPrediction(names=("discharge_burnup",), mean=mean,
                               epistemic_std=np.full((n, 1), 0.5),
                               calibrated_std=np.full((n, 1), 0.5))

    def predict_convergence(self, patterns, case, cell=0.0) -> np.ndarray:
        return np.full(len(patterns), 0.98)

    def finetune(self, new, replay, epochs=3, seed=0) -> dict:
        return {"n_new": 0, "n_replay": 0, "epochs": epochs}

    def save(self, path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "backend.json").write_text("{}", encoding="utf-8")
        return p


class FakeVerifier:
    """Deterministic verifier: every entry converges with a truth-derived FOM."""

    def __init__(self, fail_all: bool = False) -> None:
        self.fail_all = fail_all

    def evaluate_wave(self, entries):
        outs = []
        for e in entries:
            if self.fail_all:
                outs.append(WaveOutcome(
                    status="nonconverged", fom=None, n_cycles=16,
                    tolerance_margin=None, wall_s=0.0, restart_provenance="x",
                    failure="cap", converged_at_cap=True,
                    case_key=e.case_key, pattern=e.pattern, meta=e.meta))
                continue
            t = _truth(e.pattern)
            fom = FOM(f_r=t["f_r"], cbc_max=t["cbc_max"], f_q=t["f_q"],
                      cyclen=t["cyclen"], ao_min=-t["ao_abs"], ao_max=t["ao_abs"],
                      max_burnup=t["max_pin_burnup"] - 5.0,
                      max_pin_burnup=t["max_pin_burnup"], converged=True)
            outs.append(WaveOutcome(
                status="converged", fom=fom, n_cycles=11, tolerance_margin=0.5,
                wall_s=0.0, restart_provenance="native:MAS_RST", failure="",
                converged_at_cap=False, case_key=e.case_key, pattern=e.pattern,
                meta=e.meta))
        return outs


def _driver(tmp_path, *, model_mode="perfect", verifier_fail=False,
            produce_hook=None, retrain_dir="FAKE_GOOD", n_target=8,
            mini=False, models=None, log=None):
    cfg = load_config(DECK)
    cfg.curriculum.n_target = n_target
    cfg.curriculum.probe_size = 6
    cfg.curriculum.gate_mini_campaign = mini
    cfg.curriculum.gate_mini_budget = 3

    model_registry = models or {}

    def loader(d):
        return model_registry.get(str(d), FakeModel(model_mode))

    def make_verifier(run_dir, dry):
        return FakeVerifier(fail_all=verifier_fail)

    def retrain(cid, prev):
        return retrain_dir

    if produce_hook is None:
        produce_hook = lambda cid, cell: {"chains": n_target, "converged": n_target}

    return C.CurriculumDriver(
        cfg, dry_run=True, state_dir=tmp_path, progress=False,
        log=(log if log is not None else (lambda m: None)),
        model_loader=loader, make_verifier=make_verifier,
        retrain_hook=retrain, produce_hook=produce_hook,
    )


# --------------------------------------------------------------------------- #
# ring order + pure helpers
# --------------------------------------------------------------------------- #
def test_band_label_and_cell_id():
    assert C.band_label(5.25, 5.5) == "5.25-5.5"
    assert C.band_label(5.0, 5.25) == "5-5.25"
    assert C.cell_id((5.25, 5.5), 117) == "5.25-5.5_f117"


def test_ring_order_deterministic_and_anchor_first():
    bands = [[5.0, 5.25], [5.25, 5.5], [5.5, 5.75], [5.75, 6.0], [6.0, 6.25], [6.25, 6.5]]
    feeds = [101, 109, 117, 125, 133, 141]
    o1 = C.ring_order(bands, feeds, (5.25, 5.5), 117)
    o2 = C.ring_order(bands, feeds, (5.25, 5.5), 117)
    assert o1 == o2                             # deterministic
    assert len(o1) == 36                        # 6 bands x 6 feeds
    # anchor is first with ring 0
    assert C.cell_id(o1[0][0], o1[0][1]) == "5.25-5.5_f117"
    assert o1[0][2] == 0
    # rings are non-decreasing
    rings = [r for _b, _f, r in o1]
    assert rings == sorted(rings)
    # within ring 1, same-band feed-neighbours precede band-changes (plan 12.2)
    ids = [C.cell_id(b, f) for b, f, r in o1]
    i109 = ids.index("5.25-5.5_f109")
    i125 = ids.index("5.25-5.5_f125")
    i_band_neighbor = ids.index("5-5.25_f117")
    assert i109 < i_band_neighbor and i125 < i_band_neighbor


def test_ring_order_covers_all_cells_uniquely():
    bands = [[5.0, 5.25], [5.25, 5.5]]
    feeds = [109, 117, 125]
    o = C.ring_order(bands, feeds, (5.25, 5.5), 117)
    ids = {C.cell_id(b, f) for b, f, _r in o}
    assert len(ids) == 6


def test_select_cell_pairs_override_and_auto():
    cfg = load_config(DECK)
    from lpopt.data.fuel_types import FuelLibrary
    lib = FuelLibrary.from_parquet(
        Path(cfg.model.store_dir if Path(cfg.model.store_dir).is_absolute()
             else DECK.parent / cfg.model.store_dir) / "fuel_types.parquet")
    # explicit override from the deck
    pairs = C.select_cell_pairs(cfg.curriculum, "5.25-5.5_f117", (5.25, 5.5), 117,
                                lib, "ga80")
    assert pairs == ["L1_L2", "N1_N2"]
    # auto selection for a cell with no override -> in-band pairs
    auto = C.select_cell_pairs(cfg.curriculum, "5.25-5.5_f109", (5.25, 5.5), 109,
                               lib, "ga80")
    assert auto and all("_" in p for p in auto)
    full, _poor = C.select_band_types(lib, 5.25, 5.5, "ga80")
    assert len(full) >= 4                        # L + N series qualify


# --------------------------------------------------------------------------- #
# state machine
# --------------------------------------------------------------------------- #
def test_full_cell_passes_and_advances(tmp_path):
    d = _driver(tmp_path, model_mode="perfect", retrain_dir="FAKE_GOOD")
    res = d.run(max_cells=1)
    assert res["status"] in ("paused", "complete")
    cid = "5.25-5.5_f117"
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["cells"][cid]["phase"] == "done"
    assert st["cursor"] == 1
    assert st["cells"][cid]["gate"]["pass"] is True
    # champion re-pointed to the retrained dir
    assert st["champion_model_dir"] == "FAKE_GOOD"
    # blind probe artifact written with all 6 verifiable targets
    bp = json.loads((tmp_path / "cells" / cid / "blind_probe.json").read_text())
    assert bp["n_converged"] == bp["n_probe"] == 6
    for name in ("f_r", "cbc_max", "f_q", "cyclen", "ao_abs", "max_pin_burnup"):
        assert bp["per_target"][name]["n"] == 6
        assert bp["per_target"][name]["mae"] == pytest.approx(0.0, abs=1e-6)  # perfect model


def test_blind_probe_nonconverged_records_no_labels(tmp_path):
    d = _driver(tmp_path, verifier_fail=True)
    # verifier fails everything -> blind probe has no converged labels, but the
    # phase still advances (transfer simply has n=0 per target).
    d.run(max_cells=1)
    cid = "5.25-5.5_f117"
    bp = json.loads((tmp_path / "cells" / cid / "blind_probe.json").read_text())
    assert bp["n_converged"] == 0
    assert bp["per_target"]["f_r"]["n"] == 0


def test_gate_fail_halts_for_user(tmp_path):
    # retrained champion is a constant predictor -> new-cell Spearman undefined
    # -> new-cell gate fails -> driver halts without advancing the cursor.
    models = {"FAKE_BAD": FakeModel("constant")}
    d = _driver(tmp_path, model_mode="perfect", retrain_dir="FAKE_BAD", models=models)
    res = d.run(max_cells=1)
    assert res["status"] == "fail"
    cid = "5.25-5.5_f117"
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["cursor"] == 0                     # did NOT advance
    assert st["cells"][cid]["phase"] == "validate_gate"
    assert st["cells"][cid]["gate"]["pass"] is False
    assert "resume_cmd" in res


class PinInvertedModel(FakeModel):
    """Perfect on the five gate-driving targets, but ANTI-correlated on
    ``max_pin_burnup`` (predicts ``-truth``) — exercises the advisory-exclusion
    path so a structurally-unrankable target cannot drag the gate aggregate."""

    def _mean_row(self, pat) -> np.ndarray:
        row = super()._mean_row(pat)
        row[TARGET_COLS["max_pin_burnup"]] = -_truth(pat)["max_pin_burnup"]
        return row


def test_advisory_pin_burnup_excluded_from_newcell_gate(tmp_path):
    # Candidate is perfect on f_r/cbc_max/f_q/cyclen/ao_abs but perfectly
    # ANTI-correlated on max_pin_burnup.  Because max_pin_burnup is advisory
    # (config default), its -1.0 Spearman is REPORTED but excluded from the gate's
    # mean_spearman, so the aggregate stays 1.0 and the cell still passes.
    models = {"FAKE_PININV": PinInvertedModel("perfect")}
    d = _driver(tmp_path, model_mode="perfect", retrain_dir="FAKE_PININV",
                models=models)
    # sanity: the config demotes max_pin_burnup to advisory by default
    assert "max_pin_burnup" in d.curr.gate_advisory_targets
    res = d.run(max_cells=1)
    cid = "5.25-5.5_f117"
    st = json.loads((tmp_path / "state.json").read_text())
    nc = st["cells"][cid]["gate"]["new_cell"]
    pt = nc["per_target"]
    # pin is reported with its bad spearman AND flagged advisory
    assert pt["max_pin_burnup"]["advisory"] is True
    assert pt["max_pin_burnup"]["spearman"] == pytest.approx(-1.0)
    # the five gate-driving targets are reported as non-advisory
    for name in ("f_r", "cbc_max", "f_q", "cyclen", "ao_abs"):
        assert pt[name]["advisory"] is False
    # aggregate is the mean over the five NON-advisory targets only (== 1.0),
    # so the anti-correlated pin term neither drags it nor fails the cell
    assert nc["mean_spearman"] == pytest.approx(1.0)
    assert nc["pass"] is True
    assert st["cells"][cid]["gate"]["pass"] is True
    assert st["cursor"] == 1
    assert res["status"] in ("paused", "complete")


def test_transfer_curve_artifact_schema(tmp_path):
    d = _driver(tmp_path, model_mode="perfect")
    d.run(max_cells=1)
    tc = tmp_path / "transfer_curve.json"
    assert tc.exists()
    data = json.loads(tc.read_text())
    assert "cells" in data and len(data["cells"]) == 1
    entry = data["cells"][0]
    for key in ("cell", "ring", "feed", "band", "blind_mae", "post_mae",
                "post_mean_spearman"):
        assert key in entry
    assert entry["cell"] == "5.25-5.5_f117"
    assert set(entry["blind_mae"]).issuperset({"f_r", "cyclen", "max_pin_burnup"})
    # png rendered when matplotlib is available (a dependency of this project)
    assert (tmp_path / "transfer_curve.png").exists()


def test_resume_mid_cell_produce_pending(tmp_path):
    calls = {"n": 0}

    def hook(cid, cell):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"pending": True, "pid": 1234}
        return {"chains": 8, "converged": 8}

    # first pass: pends at produce_cell (resumable)
    d1 = _driver(tmp_path, produce_hook=hook)
    res1 = d1.run(max_cells=1)
    assert res1["status"] == "pending"
    cid = "5.25-5.5_f117"
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["cells"][cid]["phase"] == "produce_cell"
    assert st["cursor"] == 0

    # resume with a brand-new driver instance (same state dir + same hook) ->
    # produce completes and the cell advances to done.
    d2 = _driver(tmp_path, produce_hook=hook)
    res2 = d2.run(max_cells=1)
    assert res2["status"] in ("paused", "complete")
    st2 = json.loads((tmp_path / "state.json").read_text())
    assert st2["cells"][cid]["phase"] == "done"
    assert st2["cursor"] == 1
    assert calls["n"] == 2


def test_cell_order_override(tmp_path):
    cfg = load_config(DECK)
    cfg.curriculum.cell_order = ["5.25-5.5_f117", "5.25-5.5_f109"]
    d = C.CurriculumDriver(cfg, dry_run=True, state_dir=tmp_path, progress=False,
                           log=lambda m: None,
                           model_loader=lambda x: FakeModel("perfect"),
                           make_verifier=lambda r, dry: FakeVerifier(),
                           retrain_hook=lambda c, p: "FAKE",
                           produce_hook=lambda c, cell: {"converged": 8})
    d._init_state()
    assert d.state["order"] == ["5.25-5.5_f117", "5.25-5.5_f109"]


def test_mini_campaign_runs_and_reports(tmp_path):
    d = _driver(tmp_path, model_mode="perfect", mini=True,
                models={"FAKE_GOOD": FlatFakeModel()})
    d.run(max_cells=1)
    cid = "5.25-5.5_f117"
    mc = tmp_path / "cells" / cid / "mini_campaign.json"
    assert mc.exists()
    data = json.loads(mc.read_text())
    assert data["status"] == "ok" and data["objective"] == "flat_power"
    assert "n_fr_safe" in data and "flatness_progress" in data
    assert data["budget"] == 3


def test_a_model_without_a_map_head_spends_no_master_calls(tmp_path):
    """The mini campaign's whole point is demonstrating the FLATNESS objective.
    A backend with no map head cannot demonstrate it, and an all-``-inf`` ranking
    is a random draw — so it reports the inability and spends nothing, instead of
    burning ``gate_mini_budget`` live chains on noise."""
    d = _driver(tmp_path, model_mode="perfect", mini=True)   # plain FakeModel
    d.run(max_cells=1)
    cid = "5.25-5.5_f117"
    data = json.loads((tmp_path / "cells" / cid / "mini_campaign.json").read_text())
    assert data["status"] == "no_map_head"
    assert data["master_calls"] == 0
    assert data["results"] == []
    assert "map head" in data["reason"]


# --------------------------------------------------------------------------- #
# retrain wiring: curriculum split + cond_schema threading
# --------------------------------------------------------------------------- #
def test_retrain_remote_full_threads_split_and_cond_schema(tmp_path, monkeypatch):
    """The remote full retrain builds the curriculum split (never plain
    make_splits) and threads ``[model] cond_schema`` as ``--cond-schema`` — the
    remote-train arg contract the trainer consumes unchanged."""
    import lpopt.remote as remote_mod
    cfg = load_config(DECK)
    cfg.model.cond_schema = "v4"
    d = C.CurriculumDriver(cfg, dry_run=False, state_dir=tmp_path, progress=False,
                           log=lambda m: None)
    d._init_state()

    built = {"n": 0}
    d._write_curriculum_split = lambda: built.__setitem__("n", built["n"] + 1)

    captured = {}
    monkeypatch.setattr(remote_mod, "push", lambda s, **k: {"installed": True})

    def fake_train(s, args, **k):
        captured["args"] = list(args)
        return {"ts": "TS1", "gpu": 1}

    monkeypatch.setattr(remote_mod, "train", fake_train)
    monkeypatch.setattr(remote_mod, "status", lambda s, ts, **k: {"state": "done"})
    monkeypatch.setattr(remote_mod, "pull", lambda s, ts, **k: {"dest": "REMOTE_DEST"})

    out = d._retrain_remote_full("5.25-5.5_f117")
    assert out == "REMOTE_DEST"
    assert built["n"] == 1                          # curriculum split was built
    args = captured["args"]
    assert "--cond-schema" in args
    assert args[args.index("--cond-schema") + 1] == "v4"
    assert "--split" in args
    assert args[args.index("--split") + 1] == cfg.curriculum.retrain_split


@pytest.mark.parametrize("censor, flag", [
    (True, "--censor-a-pin-labels"),
    (False, "--no-censor-a-pin-labels"),
])
def test_retrain_remote_full_threads_censor_a_pin_labels(
        tmp_path, monkeypatch, censor, flag):
    """The remote full retrain threads ``[model] censor_dataset_a_pin_labels`` as
    the explicit ``--censor-a-pin-labels`` / ``--no-censor-a-pin-labels`` CLI flag
    so the remote trainer's pin-label masking is deterministic
    (data/reports/pinbu_forensics.md)."""
    import lpopt.remote as remote_mod
    cfg = load_config(DECK)
    cfg.model.censor_dataset_a_pin_labels = censor
    d = C.CurriculumDriver(cfg, dry_run=False, state_dir=tmp_path, progress=False,
                           log=lambda m: None)
    d._init_state()
    d._write_curriculum_split = lambda: None

    captured = {}
    monkeypatch.setattr(remote_mod, "push", lambda s, **k: {"installed": True})

    def fake_train(s, args, **k):
        captured["args"] = list(args)
        return {"ts": "TS1", "gpu": 1}

    monkeypatch.setattr(remote_mod, "train", fake_train)
    monkeypatch.setattr(remote_mod, "status", lambda s, ts, **k: {"state": "done"})
    monkeypatch.setattr(remote_mod, "pull", lambda s, ts, **k: {"dest": "REMOTE_DEST"})

    d._retrain_remote_full("5.25-5.5_f117")
    args = captured["args"]
    assert flag in args
    # exactly one of the pair is threaded (the one matching the config)
    other = ("--no-censor-a-pin-labels" if censor else "--censor-a-pin-labels")
    assert other not in args


def test_curriculum_split_threads_cell_weight_cap(tmp_path, monkeypatch):
    """``_curriculum_split_manifest`` threads ``[curriculum] cell_weight_cap`` into
    ``make_curriculum_split`` as ``cell_cap`` — the trainer's only channel for the
    negative-transfer mitigation (no fragile CLI list)."""
    import lpopt.model.splits as splits_mod
    cfg = load_config(DECK)
    cfg.curriculum.cell_weight_cap = 16.0
    d = C.CurriculumDriver(cfg, dry_run=True, state_dir=tmp_path, progress=False,
                           log=lambda m: None)
    d._init_state()

    captured = {}
    real = splits_mod.make_curriculum_split

    def spy(records, **kw):
        captured.update(kw)
        return real(records, **kw)

    monkeypatch.setattr(splits_mod, "make_curriculum_split", spy)
    d._curriculum_split_manifest()
    assert captured["cell_cap"] == pytest.approx(16.0)
    # the campaign ids the trainer will match against are the known cells
    assert "cells" in captured


def test_blind_probe_ids_by_cell_reads_probe_files(tmp_path):
    """The retrain split's blind-probe pins are derived from each cell's
    blind_probe.json candidates (record_id computed from pattern+lib+pair with the
    PRODUCE deck-knob signature the store stamps on a produced P row — NOT the
    Dataset-A mocha_default, or a produced blind-probe pattern could never match
    its store row and would silently escape the val pin)."""
    from lpopt.data.schema import compute_record_id
    from lpopt.search.verify import PRODUCE_DECK_KNOBS
    cfg = load_config(DECK)
    d = C.CurriculumDriver(cfg, dry_run=True, state_dir=tmp_path, progress=False,
                           log=lambda m: None)
    cid = "5.25-5.5_f117"
    cdir = tmp_path / "cells" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    packed = "F:L1:0|F:L2:0"
    probe = {"candidates": [
        {"pair": "L1_L2", "pattern": packed, "status": "converged"},
        {"pair": "L1_L2", "pattern": "F:L3:0", "status": "nonconverged"},
    ]}
    (cdir / "blind_probe.json").write_text(json.dumps(probe), encoding="utf-8")
    ids = d._blind_probe_ids_by_cell([cid])
    lib = cfg.model.library_id or cfg.curriculum.library
    assert ids[cid][0] == compute_record_id(packed, lib, "L1_L2", PRODUCE_DECK_KNOBS)
    # regression guard: the old Dataset-A default recipe must NOT be used.
    assert ids[cid][0] != compute_record_id(packed, lib, "L1_L2")


def test_reached_cell_ids_matches_cursor_and_phase(tmp_path):
    """``_reached_cell_ids`` = the cells the driver has STARTED, which gate the
    quarantine.  A cell is reached iff its order-index is ``<= cursor`` (every
    ``done`` cell below the cursor + the in-progress cell AT it) OR its phase has
    advanced past the initial ``ensure_types`` (a defensive signal the driver
    touched it).  A not-yet-started future cell (index > cursor, phase
    ``ensure_types``) is NOT reached, so its pre-merged rows stay quarantined —
    before any retrain could leak them into its future blind probe."""
    cfg = load_config(DECK)
    d = C.CurriculumDriver(cfg, dry_run=True, state_dir=tmp_path, progress=False,
                           log=lambda m: None)
    order = ["c0", "c1", "c2", "c3", "c4"]
    d.state = {"order": order, "cursor": 2, "cells": {
        "c0": {"phase": "done"}, "c1": {"phase": "done"},
        "c2": {"phase": "produce_cell"},      # in progress AT the cursor
        "c3": {"phase": "ensure_types"},      # future, untouched
        "c4": {"phase": "ensure_types"},      # future, untouched
    }}
    assert d._reached_cell_ids() == ["c0", "c1", "c2"]

    # the in-progress cell counts as reached even at its very first phase: with the
    # cursor at 0 and every cell freshly at ensure_types, only c0 is reached.
    d.state = {"order": order, "cursor": 0,
               "cells": {c: {"phase": "ensure_types"} for c in order}}
    assert d._reached_cell_ids() == ["c0"]

    # a completed curriculum (cursor past the end) reaches every cell.
    d.state["cursor"] = len(order)
    assert d._reached_cell_ids() == order

    # defensive OR: a future cell whose phase somehow advanced past ensure_types is
    # treated as reached (the driver touched it), never trained-on-blind by accident.
    d.state["cursor"] = 1
    for c in order:
        d.state["cells"][c]["phase"] = "ensure_types"
    d.state["cells"]["c0"]["phase"] = "done"
    d.state["cells"]["c1"]["phase"] = "blind_probe"
    d.state["cells"]["c3"]["phase"] = "produce_cell"     # advanced future cell
    assert d._reached_cell_ids() == ["c0", "c1", "c3"]

    # holey order entries (no cell record) are skipped, mirroring run()'s guard.
    d.state = {"order": ["c0", "gap", "c1"], "cursor": 2, "cells": {
        "c0": {"phase": "done"}, "c1": {"phase": "ensure_types"}}}
    assert d._reached_cell_ids() == ["c0", "c1"]


# --------------------------------------------------------------------------- #
# per-band library resolution (ga80 <= ~5.5 w/o, paramA above) — plan 12.2
# --------------------------------------------------------------------------- #
def test_band_library_threshold_and_override(tmp_path):
    d = _driver(tmp_path)
    # default threshold 5.75: the ga80-covered bands stay ga80 ...
    assert d._band_library([5.0, 5.25]) == "ga80"
    assert d._band_library([5.25, 5.5]) == "ga80"
    assert d._band_library([5.5, 5.75]) == "ga80"
    # ... and every band at/above 5.75 resolves to paramA.
    assert d._band_library([5.75, 6.0]) == "paramA"
    assert d._band_library([6.0, 6.25]) == "paramA"
    assert d._band_library([6.25, 6.5]) == "paramA"

    # explicit band_libraries entry (canonical label) overrides the threshold.
    d.curr.band_libraries = {"5.5-5.75": "paramA", "5.75-6": "ga80"}
    assert d._band_library([5.5, 5.75]) == "paramA"
    assert d._band_library([5.75, 6.0]) == "ga80"

    # a custom threshold is honoured too.
    d.curr.band_libraries = {}
    d.curr.paramA_band_lo = 6.0
    assert d._band_library([5.75, 6.0]) == "ga80"
    assert d._band_library([6.0, 6.25]) == "paramA"


def test_band_library_matches_curriculum_cell_boundary(tmp_path):
    """The live curriculum order's ga80 cells (12-14) keep ga80; the first paramA
    band (5.75-6, index 15) and everything above resolve to paramA."""
    d = _driver(tmp_path)
    d._init_state()
    ga80_cells = ["5.5-5.75_f101", "5-5.25_f133", "5.5-5.75_f133"]
    paramA_cells = ["5.75-6_f117", "5.75-6_f109", "6-6.25_f117", "6.25-6.5_f141"]
    for cid in ga80_cells:
        band = d.state["cells"][cid]["band"]
        assert d._band_library(band) == "ga80", cid
    for cid in paramA_cells:
        band = d.state["cells"][cid]["band"]
        assert d._band_library(band) == "paramA", cid


def test_ensure_types_uses_resolved_library(tmp_path, monkeypatch):
    """_phase_ensure_types gates on the per-band library: paramA for 5.75-6,
    ga80 for a lower band — and stores it on the cell for downstream produce."""
    d = _driver(tmp_path)
    d._init_state()
    d._fuel_library = object()          # unused: select_* are stubbed

    seen = {}

    def fake_select_band_types(lib, lo, hi, library_id):
        seen["lib"] = library_id
        return (["T1", "T2", "T3", "T4"], [])   # 4 full -> gate passes, no design gen

    def fake_select_cell_pairs(curr, cid, band, feed, lib, library_id):
        seen["pair_lib"] = library_id
        return ["T1_T2"]

    monkeypatch.setattr(C, "select_band_types", fake_select_band_types)
    monkeypatch.setattr(C, "select_cell_pairs", fake_select_cell_pairs)

    status = d._phase_ensure_types("5.75-6_f117")
    assert status == "advance"
    assert seen["lib"] == "paramA"
    assert seen["pair_lib"] == "paramA"
    assert d.state["cells"]["5.75-6_f117"]["library_id"] == "paramA"
    assert d.state["cells"]["5.75-6_f117"]["pairs"] == ["T1_T2"]

    status = d._phase_ensure_types("5.25-5.5_f117")
    assert status == "advance"
    assert seen["lib"] == "ga80"
    assert d.state["cells"]["5.25-5.5_f117"]["library_id"] == "ga80"


def test_band_design_bootstrap_failure_is_not_reported_as_generated(
        tmp_path, monkeypatch):
    """A band seed that never converged must fail the phase, not pass it.

    ``make_band_restart`` funnels failures into ``result.error`` /
    ``converged=False`` instead of raising, so the old ``try/except`` around it
    saw a clean return and reported ``status="generated"`` with NO seed restart
    in ``bases/`` — every later cell of the band then silently resolving at a
    fallback restart (ECC audit).
    """
    from types import SimpleNamespace

    import lpopt.design.bootstrap as B
    import lpopt.design.lattice as L
    import lpopt.design.package as P

    d = _driver(tmp_path)
    d._init_state()
    d.cfg.design.store_dir = str(tmp_path / "design")        # never touch the repo
    d.cfg.design.package_root = str(tmp_path / "design" / "package")

    monkeypatch.setattr(L, "run_batch", lambda designs, work, registry, apr, **kw: [
        SimpleNamespace(design=de, alias=registry.alias(de),
                        hgc_path=tmp_path / f"FA_{registry.alias(de)}.HGC",
                        out_path=None)
        for de in designs])
    monkeypatch.setattr(P, "assemble_package", lambda *a, **kw: None)
    monkeypatch.setattr(P, "ingest_fuel_types", lambda *a, **kw: None)

    seen = {}

    def _fake_bootstrap(pkg, pair, feed, rng, **kw):
        seen["pair"] = pair
        return B.BootstrapResult(
            pair=pair, feed=int(feed), folder="f", converged=False,
            converged_at_cap=False, n_cycles=16, cycles_needed=17,
            restart_path=None, tolerance_margin=None, wall_s=1.0)

    monkeypatch.setattr(B, "make_band_restart", _fake_bootstrap)

    out = d._generate_band_designs("5.75-6_f117", (5.75, 6.0), 117)
    assert seen["pair"], "the bootstrap really was attempted"
    assert out["ok"] is False
    assert out["status"] == "bootstrap_failed"
    assert "not converged" in out["error"]
    # the per-band counters survive, so the failure is still diagnosable
    assert out["n_designs"] and out["library_id"] == "paramA"


# --------------------------------------------------------------------------- #
# per-library asset routing (paramA -> design package; ga80 unchanged)
# --------------------------------------------------------------------------- #
def _make_paramA_package(base: Path) -> Path:
    """Minimal design package: registry + lib/MAS_XSL (COMP roster) + one seed
    deck (for the vendor %LPD_C&X synth-roster read)."""
    pkg = base / "data" / "design" / "package"
    (pkg / "bases" / "P0_P1").mkdir(parents=True, exist_ok=True)
    (pkg / "bases" / "P0_P1" / "MAS_RST.SEED.02").write_bytes(b"seed")
    (pkg / "registry.json").write_text(
        json.dumps({"aliases": {"P5849X": "P0", "P6257X": "P1", "P6253X": "P2"}}),
        encoding="utf-8",
    )
    lib = pkg / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    lib.joinpath("MAS_XSL").write_text(
        "COMP FA_P0  x\nCOMP FA_P1  x\nCOMP FA_P2  x\n", encoding="utf-8")
    lib.joinpath("MAS_HFF").write_text("hff\n", encoding="utf-8")
    return pkg


def test_build_resolver_routes_paramA_to_design_package(tmp_path):
    """A paramA cell's resolver points at the design package (its own bases/lib),
    loads the registry alias bridge, and carries the package's %GEN_DIM."""
    from lpopt.design.coredeck import library_dims

    cfg = load_config(DECK)
    cfg.source_path = tmp_path / "lpopt.inp"        # anchor _res() at tmp_path
    pkg = _make_paramA_package(tmp_path)

    r = C._build_resolver(cfg, fuel_library=object(), library_id="paramA")
    assert r.package_root == pkg
    assert r.library_id == "paramA"
    assert r.template_fallbacks == ()               # ga80 decks suppressed
    assert r.type_to_alias == {"P5849X": "P0", "P6257X": "P1", "P6253X": "P2"}
    assert r.alias_to_type["P0"] == "P5849X"
    assert r.library_dims == library_dims(3)        # 3 COMP FA_* sets -> (6, 8)


def test_build_resolver_ga80_unchanged(tmp_path):
    """ga80 cells keep [verify].package_root, the configured template_fallbacks,
    the default library dims, and no alias bridge."""
    from lpopt.search.assets import LIBRARY_DIMS

    cfg = load_config(DECK)
    r = C._build_resolver(cfg, fuel_library=object(), library_id="ga80")
    assert r.package_root.name == "FEASIBLE_PACKAGE"
    assert r.library_dims == LIBRARY_DIMS
    assert r.type_to_alias == {}
    assert r.template_fallbacks  # the configured ga80 fallbacks survive


def test_make_pin_burnup_verifier_inherits_dims_and_root_from_resolver(tmp_path):
    """The live verifier's MASTER package_root + reload-deck %GEN_DIM gate follow
    the resolver's library (so a paramA wave stages the paramA lib and validates
    against the paramA dims, while ga80 stays byte-identical)."""
    from lpopt.search.assets import CaseAssetResolver, LIBRARY_DIMS

    cfg = load_config(DECK)

    para = CaseAssetResolver(tmp_path / "pkgA", library_id="paramA", library_dims=(14, 16))
    v = C.make_pin_burnup_verifier(cfg, tmp_path / "runA", para, dry_run=False)
    assert v.package_root == (tmp_path / "pkgA")
    assert tuple(v.library_dims) == (14, 16)

    ga = CaseAssetResolver(tmp_path / "pkgG", library_id="ga80")
    vg = C.make_pin_burnup_verifier(cfg, tmp_path / "runG", ga, dry_run=False)
    assert vg.package_root == (tmp_path / "pkgG")
    assert tuple(vg.library_dims) == LIBRARY_DIMS


# --------------------------------------------------------------------------- #
# F_r DEFERRAL (user decision 2026-07-26) — every promotion surface, one switch
#
# The deferral is about PROMOTION/ACCEPTANCE semantics, not about one function.
# ``[curriculum] gate_noreg_fr_guard_enabled`` re-arms them all.
# --------------------------------------------------------------------------- #
class FrReversedFakeModel(FakeModel):
    """Perfect on every probe target EXCEPT ``f_r``, whose rank it inverts."""

    def _mean_row(self, pat) -> np.ndarray:
        row = super()._mean_row(pat)
        row[TARGET_COLS["f_r"]] = 3.0 - _truth(pat)["f_r"]      # rank reversed
        return row


def _flat_truth(pat) -> tuple[float, float]:
    """``(node_peak, map_cov)`` — deliberately INDEPENDENT of :func:`_truth`, so a
    ranking driven by F_r/cyclen criteria is a random draw in peak order."""
    h = int(hashlib.sha1(("peak" + pat.canonical()).encode()).hexdigest(), 16)
    return 1.30 + 0.40 * ((h % 9973) / 9973.0), 0.15


class FlatFakeModel(FakeModel):
    """A model with a MAP HEAD and constraint-clean surrogate rows.

    ``f_r`` still comes from the truth (so the F_r SAFETY gate has teeth), but
    ``cbc_max`` / ``f_q`` / ``ao_abs`` / ``max_pin_burnup`` are pinned well inside
    their limits, so the flat_power ranking is decided by ``node_peak`` alone —
    which is exactly the property the mini-campaign test needs to assert.
    """

    def _mean_row(self, pat) -> np.ndarray:
        row = np.full(7, np.nan)
        row[TARGET_COLS["f_r"]] = _truth(pat)["f_r"]
        row[TARGET_COLS["cbc_max"]] = 1300.0
        row[TARGET_COLS["f_q"]] = 2.20
        row[TARGET_COLS["cyclen"]] = _truth(pat)["cyclen"]
        row[TARGET_COLS["ao_abs"]] = 0.15
        row[TARGET_COLS["max_pin_burnup"]] = 70.0
        return row

    def predict_map_flatness(self, patterns, case, cell=0.0):
        vals = [_flat_truth(p) for p in patterns]
        pk = np.array([v[0] for v in vals], dtype=float)
        cv = np.array([v[1] for v in vals], dtype=float)
        return pk, np.zeros(len(pk)), cv, np.zeros(len(cv))


def _run_newcell(tmp_path, *, fr_guarded: bool, min_spearman: float = 0.9):
    d = _driver(tmp_path, model_mode="perfect")
    d.curr.gate_new_cell_min_spearman = min_spearman
    d.curr.gate_noreg_fr_guard_enabled = fr_guarded
    d.load_model = lambda _dir: FrReversedFakeModel("perfect")
    d.run(max_cells=1)
    cid = d.state["order"][0]
    gate = json.loads((tmp_path / "cells" / cid / "gate.json").read_text())
    return gate["new_cell"]


def test_new_cell_skill_gate_does_not_let_f_r_veto_a_retrain(tmp_path):
    """MEDIUM (1): the new-cell skill gate ANDs into the SAME ``validate_gate``
    verdict, so an F_r vote there is an F_r veto on the retrain by the back door.

    Under the deferral ``f_r`` is scored on the probe holdout and REPORTED, but
    it is excluded from the mean-Spearman the gate thresholds — the same
    treatment ``gate_no_regression`` already gives it, through the same switch.
    """
    nc = _run_newcell(tmp_path, fr_guarded=False)

    assert nc["per_target"]["f_r"]["spearman"] < 0        # scored, and badly
    assert nc["per_target"]["f_r"]["report_only"] is True
    assert nc["report_only_targets"] == ["f_r"]
    assert nc["guarded_targets"] and "f_r" not in nc["guarded_targets"]
    assert nc["mean_spearman"] == pytest.approx(1.0)      # f_r did not vote
    assert nc["pass"] is True
    assert "f_r" in nc["note"] and "REPORT-ONLY" in nc["note"]
    assert nc["fr_guard"]["enforced"] is False
    assert nc["fr_guard"]["knob"] == C.FR_GUARD_KNOB


def test_the_same_switch_re_arms_the_new_cell_skill_gate(tmp_path):
    """One setting, every surface: with the guard on, the identical candidate is
    rejected by the new-cell gate on its F_r collapse."""
    nc = _run_newcell(tmp_path, fr_guarded=True)

    assert nc["pass"] is False
    assert nc["mean_spearman"] < 0.9
    assert nc["per_target"]["f_r"]["report_only"] is False
    assert nc["report_only_targets"] == []
    assert "f_r" in nc["guarded_targets"]
    assert nc["fr_guard"]["enforced"] is True


def test_validate_gate_logs_the_report_only_note_on_FAIL_too(tmp_path, monkeypatch):
    """LOW (5): a FAIL line that omits the note hides the fact that F_r was
    SCORED but not enforced — exactly the reading the note exists to prevent."""
    msgs: list[str] = []
    d = _driver(tmp_path, log=msgs.append)
    d._init_state()
    cid = d.state["order"][0]
    d.state["cells"][cid]["champion_before"] = "PREV"
    d.state["champion_model_dir"] = "NEW"
    note = ("REPORT-ONLY axes (scored, NOT enforced): f_r — their drop cannot "
            "block promotion.")
    monkeypatch.setattr(d, "_gate_newcell",
                        lambda c, m: {"pass": False, "mean_spearman": -1.0,
                                      "per_target": {}, "note": "newcell note here"})
    monkeypatch.setattr(d, "_gate_no_regression",
                        lambda c, p, n: {"pass": True, "note": note})
    monkeypatch.setattr(d, "_gate_legacy_tail", lambda p, n: {"pass": True})

    assert d._phase_validate_gate(cid) == "fail"
    joined = "\n".join(msgs)
    assert "REPORT-ONLY axes (scored, NOT enforced): f_r" in joined
    assert "newcell note here" in joined


def _mini(tmp_path, budget=8):
    d = _driver(tmp_path, mini=True, models={"FAKE_GOOD": FlatFakeModel()})
    d.curr.gate_mini_budget = budget
    d.run(max_cells=1)
    cid = d.state["order"][0]
    data = json.loads((tmp_path / "cells" / cid / "mini_campaign.json").read_text())
    return d, cid, data


def test_mini_campaign_exercises_the_flatness_objective_not_an_f_r_target(tmp_path):
    """MEDIUM (2): the validate_gate mini campaign spends REAL MASTER calls, and
    it was spending them hunting an F_r target the corpus cannot supply.  Under
    the flatness-first program it must demonstrate what the promoted champion
    will actually steer: the node_peak/map_cov objective, with F_r kept only as
    the licensing SAFETY gate on candidate rows.
    """
    import random as _random

    d, cid, data = _mini(tmp_path, budget=8)

    assert data["objective"] == "flat_power"
    assert data["fr_role"] == "safety_gate"
    # F_r is NOT the selection target any more…
    assert "n_feasible" not in data
    # …but it still screens candidates at the D1 in-loop safety gate.
    assert data["fr_safety_gate"] == pytest.approx(1.70)

    # The picks are the FLATTEST safety-clean candidates in the pool the campaign
    # drew — reproduced here from the same seed, so this pins the objective the
    # ranking used rather than the shape of the output dict.
    cell = d.state["cells"][cid]
    budget = int(d.curr.gate_mini_budget)
    pool = C._gen_candidates(cell["pairs"], cell["feed"], max(budget * 40, 400),
                             _random.Random(d.cfg.flow.random_seed + 4242))
    safe = [(pat.canonical(), _flat_truth(pat)[0]) for _pair, pat in pool
            if _truth(pat)["f_r"] + 0.25 * 0.05 <= 1.70]
    expect = {c for c, _pk in sorted(safe, key=lambda t: t[1])[:budget]}
    got = {r["pattern"] for r in data["results"]}
    assert got == expect


def test_mini_campaign_reports_flatness_progress_and_licensing_margin(tmp_path):
    """What the campaign now DEMONSTRATES is stated in the artifact, and the
    licensing limit survives as a reported margin rather than as the objective."""
    _d, _cid, data = _mini(tmp_path, budget=4)

    assert data["demonstrates"]
    assert "flat" in data["demonstrates"].lower()
    assert data["best_pred_node_peak"] is not None
    assert data["n_fr_safe"] == len(data["results"])          # all picks screened
    assert data["licensing_limit"] == pytest.approx(1.55)
    for row in data["results"]:
        assert "pred_node_peak" in row and "pred_map_cov" in row
        assert "fr_margin" in row                             # 1.55 - measured F_r


def test_the_curriculum_verifier_honours_harvest_maps(tmp_path):
    """A mini campaign that RANKS on flatness must be able to MEASURE it.

    ``WaveOutcome.maps`` is None unless the verifier was built with
    ``harvest_maps``, so without this wiring the flat_power mini campaign reports
    ``best_node_peak: null`` on every cell forever.  Driven by the SAME
    ``[verify] harvest_maps`` knob ``search.produce`` and ``search.campaign``
    already read — not a second hardcoded policy.
    """
    from lpopt.search.assets import CaseAssetResolver

    cfg = load_config(DECK)
    res = CaseAssetResolver(tmp_path / "pkg", library_id="ga80")

    cfg.verify.harvest_maps = True
    assert C.make_pin_burnup_verifier(
        cfg, tmp_path / "on", res, dry_run=False).harvest_maps is True

    cfg.verify.harvest_maps = False
    assert C.make_pin_burnup_verifier(
        cfg, tmp_path / "off", res, dry_run=False).harvest_maps is False


def test_a_null_measured_flatness_is_explained_not_silent(tmp_path):
    """The mini campaign says WHY it has no measured peak (the fake verifier
    harvests no maps), so a null is never readable as "the model was flat"."""
    _d, _cid, data = _mini(tmp_path, budget=3)

    assert data["best_node_peak"] is None            # stub verifier: no maps
    assert data["flatness_progress"] is None         # unknown, not True
    assert "unavailable" in data["measured_flatness_note"]
    assert "harvest_maps" in data or "harvest_maps" in data["measured_flatness_note"]


# --------------------------------------------------------------------------- #
# HIGH-2: harvest_maps on the CURRICULUM path was cosmetic
#
# ``make_pin_burnup_verifier`` INJECTS an ``evaluator_factory``, and an injected
# factory bypasses ``WaveVerifier._default_factory`` entirely — which is the only
# place that wraps the runner in ``HarvestingEquilibriumEvaluator``.  So setting
# ``verifier.harvest_maps = True`` changed nothing that could produce a map:
# ``metadata["maps"]`` was never written, ``WaveOutcome.maps`` stayed None, the
# mini campaign still could not measure flatness — and the null-reason reporting
# then blamed "all nonconverged / EDIT5 parse odd", which is the WRONG cause.
# --------------------------------------------------------------------------- #
def test_the_injected_curriculum_factory_installs_the_harvesting_evaluator(tmp_path):
    """The switch must reach the EVALUATOR, not just the verifier attribute."""
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.search.verify import HarvestingEquilibriumEvaluator

    cfg = load_config(DECK)
    res = CaseAssetResolver(tmp_path / "pkg", library_id="ga80")

    cfg.verify.harvest_maps = True
    v_on = C.make_pin_burnup_verifier(cfg, tmp_path / "on", res, dry_run=False)
    ev_on = v_on._factory(0, None)
    assert isinstance(ev_on, HarvestingEquilibriumEvaluator), (
        "harvest_maps=True built a plain EquilibriumEvaluator, so metadata['maps'] "
        "is never written and WaveOutcome.maps is always None")
    # the final converged work dir must survive for the EDIT5 read
    assert bool(getattr(ev_on.runner, "keep_success", False)) is True

    cfg.verify.harvest_maps = False
    ev_off = C.make_pin_burnup_verifier(
        cfg, tmp_path / "off", res, dry_run=False)._factory(0, None)
    assert not isinstance(ev_off, HarvestingEquilibriumEvaluator)


def test_the_harvesting_evaluator_survives_the_pin_burnup_wiring(tmp_path):
    """The harvesting wrapper must keep the curriculum's pin-burnup + purging
    runner, or the fix would trade a measurable map for a lost label."""
    from lpopt.search.assets import CaseAssetResolver
    from lpopt.search.verify import PurgingEquilibriumRunner

    cfg = load_config(DECK)
    cfg.verify.harvest_maps = True
    res = CaseAssetResolver(tmp_path / "pkg", library_id="ga80")
    ev = C.make_pin_burnup_verifier(
        cfg, tmp_path / "on", res, dry_run=False)._factory(0, None)
    runner = ev.runner
    assert isinstance(runner, PurgingEquilibriumRunner)
    assert bool(getattr(runner, "enable_pin_burnup", False)) is True


def test_the_null_flatness_reason_names_the_cause_that_actually_fired(tmp_path):
    """A gate that reports "measured" while measuring nothing is worse than one
    that reports "unavailable" — so the null-reason must be the TRUE cause.

    The fixture's verifier CONVERGES every pick and carries no map, with
    ``[verify] harvest_maps = true`` in the deck.  The honest reading is "the
    verifier handed back no harvested map", NOT "all nonconverged, or the EDIT5
    parse came back odd" — the latter blames the physics for a wiring fault.
    """
    _d, _cid, data = _mini(tmp_path, budget=3)

    assert data["harvest_maps"] is True
    assert data["best_node_peak"] is None
    assert data["n_picks_converged"] == len(data["results"]) > 0
    assert data["n_picks_with_maps"] == 0
    assert data["measured_flatness_cause"] == "no_maps_harvested"
    note = data["measured_flatness_note"]
    assert "harvest" in note
    assert "nonconverged" not in note, "blamed convergence for a harvesting fault"


def test_a_measured_flatness_null_from_nonconvergence_says_so(tmp_path):
    """The other innocent cause keeps its own, distinct reason."""
    d = _driver(tmp_path, mini=True, models={"FAKE_GOOD": FlatFakeModel()},
                verifier_fail=True)
    d.curr.gate_mini_budget = 3
    d.run(max_cells=1)
    cid = d.state["order"][0]
    data = json.loads(
        (tmp_path / "cells" / cid / "mini_campaign.json").read_text())
    assert data["measured_flatness_cause"] == "no_convergence"
    assert data["n_picks_converged"] == 0
    assert "converge" in data["measured_flatness_note"]


def test_harvest_disabled_is_still_reported_as_its_own_cause(tmp_path):
    """And the deck-said-no cause is not folded into the others."""
    d = _driver(tmp_path, mini=True, models={"FAKE_GOOD": FlatFakeModel()})
    d.cfg.verify.harvest_maps = False
    d.curr.gate_mini_budget = 3
    d.run(max_cells=1)
    cid = d.state["order"][0]
    data = json.loads(
        (tmp_path / "cells" / cid / "mini_campaign.json").read_text())
    assert data["measured_flatness_cause"] == "harvest_disabled"
    assert "harvest_maps" in data["measured_flatness_note"]


def test_a_measured_flatness_that_worked_says_nothing_and_a_partial_says_partial(
        tmp_path, monkeypatch):
    """A run that DID measure carries no excuse; a run that measured only some of
    its converged picks is not allowed to look like a run that measured all."""
    import numpy as np

    from lpopt.data.flatness import BOC_CHANNEL, BOC_STEP  # noqa: F401

    def _maps(scale: float):
        m = np.full((4, 9, 9), np.nan, dtype=np.float32)
        rng = np.random.default_rng(0)
        m[0] = (1.0 + 0.05 * scale * rng.random((9, 9))).astype(np.float32)
        return m

    class _HarvestingFake(FakeVerifier):
        def __init__(self, n_with_maps):
            super().__init__()
            self.n_with_maps = n_with_maps

        def evaluate_wave(self, entries):
            outs = super().evaluate_wave(entries)
            return [oc if i >= self.n_with_maps
                    else dataclasses.replace(oc, maps=_maps(i + 1))
                    for i, oc in enumerate(outs)]

    def _run(n_with_maps):
        d = _driver(tmp_path / f"n{n_with_maps}", mini=True,
                    models={"FAKE_GOOD": FlatFakeModel()})
        d._make_verifier = lambda run_dir, dry: _HarvestingFake(n_with_maps)
        d.curr.gate_mini_budget = 3
        d.run(max_cells=1)
        cid = d.state["order"][0]
        return json.loads((tmp_path / f"n{n_with_maps}" / "cells" / cid /
                           "mini_campaign.json").read_text())

    full = _run(3)
    assert full["best_node_peak"] is not None
    assert full["n_picks_with_maps"] == full["n_picks_converged"] == 3
    assert full["measured_flatness_cause"] is None
    assert full["measured_flatness_note"] is None

    part = _run(1)
    assert part["best_node_peak"] is not None      # something WAS measured …
    assert part["measured_flatness_cause"] == "partial"   # … but not everything
    assert "PARTIAL" in part["measured_flatness_note"]


# --------------------------------------------------------------------------- #
# LOW: the DRIVER's own "no previous cells" early return
#
# The module-level ``gate_no_regression`` trivial path was given the full gate
# contract last round, but ``CurriculumDriver._gate_no_regression`` has a SECOND
# early return of its own that never got it — so on the very first cell a
# consumer reading ``guarded_targets`` / ``fr_guard`` off the gate result still
# hits a KeyError, and the artifact carries no record that F_r was deferred.
# --------------------------------------------------------------------------- #
def test_the_driver_early_return_carries_the_whole_gate_contract(tmp_path):
    d = _driver(tmp_path)
    d._init_state()
    cid = d.state["order"][0]

    first = d._gate_no_regression(cid, None, "NEW")          # prev_dir is None
    assert first["note"] == "no previous cells"
    assert first["pass"] is True
    for key in ("epsilon", "worst_drop", "worst_drop_any_axis", "checks",
                "scored_record_ids", "guarded_targets", "report_only_targets",
                "scored_targets", "unavailable", "fr_guard"):
        assert key in first, f"driver early return dropped {key!r}"
    assert first["report_only_targets"] == ["f_r"]
    assert "f_r" not in first["guarded_targets"]
    assert first["fr_guard"]["enforced"] is False
    assert first["fr_guard"]["knob"] == C.FR_GUARD_KNOB

    # …and the same early return honours the one switch.
    d.curr.gate_noreg_fr_guard_enabled = True
    armed = d._gate_no_regression(cid, None, "NEW")
    assert armed["fr_guard"]["enforced"] is True
    assert "f_r" in armed["guarded_targets"]
    assert armed["report_only_targets"] == []


def test_the_driver_early_return_matches_the_module_gate_contract(tmp_path):
    """One contract, one shape — the two trivial paths must not drift."""
    d = _driver(tmp_path)
    d._init_state()
    cid = d.state["order"][0]
    mine = d._gate_no_regression(cid, None, "NEW")
    theirs = C.gate_no_regression(None, None, None, {}, [],
                                  epsilon=d.curr.gate_noreg_epsilon)
    assert set(mine) == set(theirs)
    assert mine == theirs


# --------------------------------------------------------------------------- #
# LOW: "no_convergence" must mean nothing converged
#
# ``n_converged`` counted only the picks that ALSO carried a FOM, so a wave whose
# picks all CONVERGED but whose FOM came back empty was reported as
# ``no_convergence`` -- the same class of dishonest-cause reporting the four-way
# taxonomy one layer up was just built to end.  The two are different faults with
# different actions: a real non-convergence is physics, a missing FOM is the
# harness losing a result it already had.
# --------------------------------------------------------------------------- #
def test_converged_picks_without_a_fom_are_not_called_nonconvergence(tmp_path):
    """Every MINI pick converged and none carried a FOM: that is not "nothing
    converged", and the report must not say so."""
    import dataclasses as _dc

    class _FomlessFake(FakeVerifier):
        """Converges everything, then drops the FOM on the floor."""

        def evaluate_wave(self, entries):
            return [_dc.replace(oc, status="converged", fom=None)
                    for oc in super().evaluate_wave(entries)]

    d = _driver(tmp_path, mini=True, models={"FAKE_GOOD": FlatFakeModel()})
    # only the MINI wave (work root tag "m") loses its FOM; the blind probe
    # ("p") stays healthy, so the cell reaches the mini campaign normally.
    d._make_verifier = lambda run_dir, dry: (
        _FomlessFake() if str(run_dir).endswith("m") else FakeVerifier())
    d.curr.gate_mini_budget = 3
    d.run(max_cells=1)
    cid = d.state["order"][0]
    data = json.loads(
        (tmp_path / "cells" / cid / "mini_campaign.json").read_text())

    assert data["n_picks_converged"] == len(data["results"]) > 0, (
        "converged picks that carried no FOM were not counted as converged")
    assert data["n_picks_with_fom"] == 0
    assert data["measured_flatness_cause"] != "no_convergence", (
        "a FOM-less wave was misreported as a convergence failure")
    assert data["measured_flatness_cause"] == "no_fom"
    note = data["measured_flatness_note"]
    assert "FOM" in note
    assert "none of the 3 picked candidates converged" not in note


def test_a_real_nonconvergence_still_says_no_convergence(tmp_path):
    """The honest cause keeps its name: the counter change must not blur it."""
    d = _driver(tmp_path, mini=True, models={"FAKE_GOOD": FlatFakeModel()},
                verifier_fail=True)
    d.curr.gate_mini_budget = 3
    d.run(max_cells=1)
    cid = d.state["order"][0]
    data = json.loads(
        (tmp_path / "cells" / cid / "mini_campaign.json").read_text())
    assert data["measured_flatness_cause"] == "no_convergence"
    assert data["n_picks_converged"] == 0
    assert data["n_picks_with_fom"] == 0
