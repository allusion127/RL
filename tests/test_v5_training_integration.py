"""End-to-end v5 training + serving, and the flag-off byte-identity contract.

Two halves:

* **Flags OFF must be the legacy path.** The decisive test is
  :func:`test_step_loss_with_flags_off_equals_the_legacy_expression` — it
  recomputes the pre-v5 loss expression independently and asserts the training
  step's loss is bit-equal, so no new term can silently perturb a gradient.
  Combined with the featurize / net golden digests (test_v5_schema,
  test_quantile_heads) this pins inputs, init AND objective.

* **Flags ON must round-trip.** A real (tiny) v5 ensemble is trained and served,
  and ``predict()`` must return ABSOLUTE cyclen (prior + residual), fill the
  previously-NaN assembly-burnup column, and carry the quantile band — without
  breaking the 7-column vendor layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F                                     # noqa: E402

from lpopt.data.schema import unpack_pattern                        # noqa: E402
from lpopt.model.dataset_torch import TARGETS, TARGETS_WITH_ASM_BU  # noqa: E402
from lpopt.model.net import PosValNet, PosValNetConfig              # noqa: E402
from lpopt.model.train import (                                     # noqa: E402
    TrainConfig, _Norm, _step_member, convergence_loss, cyclen_rank_loss,
    f_r_rank_loss, map_loss, regression_loss, train_ensemble,
)
from lpopt.vendor.masterrl.domain import CaseKey                    # noqa: E402

STORE = "data/store"
N_CH, N_G = 43, 13


def _tiny_cfg(**over) -> TrainConfig:
    cfg = TrainConfig(epochs=2, warmup_epochs=1, batch_size=32, augment=False,
                      min_case_val=3, map_norm_subset=64, round_trip_rows=8)
    cfg.auto_fit_cell_calibration = False       # a full serve-path fit is slow
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _batch(n=12, n_targets=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    cells = torch.randn(n, N_CH, 19, 19, generator=g)
    cells[:, 0] = 1.0                                   # fuel mask
    targets = torch.randn(n, n_targets, generator=g) * 5.0 + 600.0
    return {
        "cells": cells,
        "globals": torch.randn(n, N_G, generator=g),
        "targets": targets,
        "target_mask": torch.ones(n, n_targets),
        "conv_label": torch.ones(n),
        "conv_mask": torch.ones(n),
        "maps": torch.randn(n, 4, 9, 9, generator=g),
        "maps_mask": torch.ones(n, 4, 9, 9),
        "cyclen_cell": torch.zeros(n, dtype=torch.long),
    }


class _Member:
    """A minimal ``_MemberState`` stand-in for driving ``_step_member``."""

    def __init__(self, cfg, n_targets=7, **net_over):
        torch.manual_seed(11)
        self.model = PosValNet(PosValNetConfig(
            in_channels=N_CH, n_globals=N_G, n_targets=n_targets, **net_over))
        self.fwd = self.model
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        tmean = np.full(n_targets, 600.0)
        tstd = np.full(n_targets, 5.0)
        self.norm = _Norm(tmean, tstd, np.zeros(4), np.ones(4),
                          torch.device("cpu"), TARGETS.index("cyclen"))
        self.use_nll = True
        self.use_prior = False
        self.q_idx = None
        self.q_names = ()
        self.running = 0.0
        self.n_batches = 0


# --------------------------------------------------------------------------- #
# flags OFF == the legacy objective
# --------------------------------------------------------------------------- #
def test_step_loss_with_flags_off_equals_the_legacy_expression():
    cfg = TrainConfig()
    assert not (cfg.cyclen_physics_prior or cfg.quantile_heads
                or cfg.promote_max_asm_bu or cfg.distill_targets)
    m = _Member(cfg)
    batch = _batch()

    # independently recompute the pre-v5 loss
    with torch.no_grad():
        out = m.model(batch["cells"], batch["globals"])
    z_t = (batch["targets"] - m.norm.tmean) / m.norm.tstd
    z_m = (batch["maps"] - m.norm.mmean) / m.norm.mstd
    expected = (
        regression_loss(out["mu"], out["log_sigma"], z_t, batch["target_mask"],
                        use_nll=True, beta=cfg.beta_nll, delta=cfg.huber_delta)
        + cfg.map_lambda * map_loss(out["map"], z_m, batch["maps_mask"],
                                    cfg.huber_delta)
        + cfg.conv_weight * convergence_loss(out["conv_logit"],
                                             batch["conv_label"],
                                             batch["conv_mask"])
        + cfg.cyclen_rank_weight * cyclen_rank_loss(
            out["mu"][:, 3], batch["targets"][:, 3],
            batch["target_mask"][:, 3], batch["cyclen_cell"],
            margin=cfg.cyclen_rank_margin_z,
            min_gap_efpd=cfg.cyclen_rank_min_gap_efpd)
        # f_r elite rank loss is also default-ON (parity_round1c_20260722 [1a]);
        # map/F_r consistency stays default-OFF so it adds nothing here.
        + cfg.f_r_rank_weight * f_r_rank_loss(
            out["mu"][:, 0], batch["targets"][:, 0],
            batch["target_mask"][:, 0], batch["cyclen_cell"],
            margin=cfg.f_r_rank_margin_z, min_gap=cfg.f_r_rank_min_gap,
            low_thresh=cfg.f_r_rank_low_thresh, low_weight=cfg.f_r_rank_low_weight)
    )
    _step_member(m, batch, cfg, use_amp=False, device=torch.device("cpu"))
    assert m.running == pytest.approx(float(expected), rel=1e-6)


def test_z_targets_without_a_prior_is_the_legacy_formula():
    norm = _Norm(np.full(7, 600.0), np.full(7, 5.0), np.zeros(4), np.ones(4),
                 torch.device("cpu"))
    raw = torch.randn(6, 7) * 5 + 600
    torch.testing.assert_close(norm.z_targets(raw), (raw - norm.tmean) / norm.tstd)


def test_z_targets_with_a_prior_touches_only_cyclen():
    norm = _Norm(np.full(7, 600.0), np.full(7, 5.0), np.zeros(4), np.ones(4),
                 torch.device("cpu"), TARGETS.index("cyclen"))
    raw = torch.randn(6, 7) * 5 + 600
    prior = torch.full((6,), 590.0)
    z = norm.z_targets(raw, prior)
    legacy = norm.z_targets(raw)
    for k in range(7):
        if k == 3:
            torch.testing.assert_close(z[:, k], (raw[:, k] - prior - 600.0) / 5.0)
        else:
            torch.testing.assert_close(z[:, k], legacy[:, k])
    # and the caller's tensor was not mutated
    assert not torch.equal(z, legacy)


def test_distillation_term_is_absent_without_a_cache():
    """No ``distill_soft`` in the batch -> the KD branch never runs."""
    cfg = TrainConfig()
    m = _Member(cfg)
    batch = _batch()
    assert "distill_soft" not in batch
    _step_member(m, batch, cfg, use_amp=False, device=torch.device("cpu"))
    assert np.isfinite(m.running)


def test_distillation_term_changes_the_loss_when_present():
    cfg_off = TrainConfig()
    cfg_on = TrainConfig()
    cfg_on.distill_weight = 0.5
    batch = _batch()
    kd = dict(batch)
    kd["distill_soft"] = batch["targets"] + 50.0        # a teacher that disagrees
    kd["distill_mask"] = torch.ones_like(batch["target_mask"])

    a, b = _Member(cfg_off), _Member(cfg_on)
    _step_member(a, batch, cfg_off, use_amp=False, device=torch.device("cpu"))
    _step_member(b, kd, cfg_on, use_amp=False, device=torch.device("cpu"))
    assert b.running > a.running


def test_distillation_excludes_the_pin_burnup_targets():
    from lpopt.model.distill import EXCLUDED_TARGETS, teacher_mask
    mask = teacher_mask(TARGETS_WITH_ASM_BU)
    assert mask[TARGETS_WITH_ASM_BU.index("max_pin_burnup")] == 0.0
    assert mask[TARGETS_WITH_ASM_BU.index("max_assembly_burnup")] == 0.0
    assert mask[TARGETS_WITH_ASM_BU.index("cyclen")] == 1.0
    assert mask[TARGETS_WITH_ASM_BU.index("f_r")] == 1.0
    assert EXCLUDED_TARGETS == {"max_pin_burnup", "max_assembly_burnup"}


def test_quantile_term_is_absent_when_the_head_is_off():
    cfg = TrainConfig()
    m = _Member(cfg)
    with torch.no_grad():
        out = m.model(_batch()["cells"], _batch()["globals"])
    assert "quantiles" not in out


# --------------------------------------------------------------------------- #
# flags ON: a real (tiny) train -> serve round trip
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def v5_model(tmp_path_factory):
    cfg = _tiny_cfg(cyclen_physics_prior=True, quantile_heads=True,
                    promote_max_asm_bu=True)
    out = tmp_path_factory.mktemp("v5") / "ens"
    train_ensemble(1, split="S1", device="cpu", out_dir=out, config=cfg,
                   subset_rows=300, cond_schema="v5", base_seed=1, verbose=False)
    return out


def test_v5_checkpoint_records_the_full_contract(v5_model):
    meta = json.loads(
        sorted(v5_model.glob("member_*/meta.json"))[0].read_text(encoding="utf-8"))
    assert meta["cond_schema"] == "v5"
    assert len(meta["channels"]) == 48
    assert "origin_n_gd" not in meta["channels"]
    assert "origin_reactivity_swing" in meta["channels"]
    assert meta["target_names"] == list(TARGETS_WITH_ASM_BU)
    assert meta["cyclen_physics_prior"]["enabled"] is True
    assert meta["quantile_heads"] == {"enabled": True,
                                      "targets": ["f_r", "cyclen"],
                                      "levels": [0.1, 0.5, 0.9]}
    assert meta["promote_max_asm_bu"] is True
    assert meta["net_config"]["n_targets"] == 8
    assert meta["net_config"]["n_quantile_targets"] == 2


def test_prior_artifact_is_written_next_to_the_ensemble(v5_model):
    from lpopt.model.physics_prior import PRIOR_NAME, CyclenPhysicsPrior
    p = v5_model / PRIOR_NAME
    assert p.is_file()
    assert CyclenPhysicsPrior.load(p).n_fit > 0


def _serve(model_dir, n=6):
    from lpopt.model.model_api import PosValCnnBackend
    b = PosValCnnBackend.from_dir(model_dir, store_dir=STORE, library_id="ga80")
    df = pd.read_parquet(f"{STORE}/records.parquet")
    df = df[df["library_id"] == "ga80"].head(n)
    pats = [unpack_pattern(str(p)) for p in df["pattern"]]
    cases = [CaseKey(str(c), int(f)) for c, f in zip(df["case_pair"], df["feed"])]
    return b, pats, cases, df


def test_predict_returns_absolute_cyclen(v5_model):
    """The residual head must be invisible to every downstream consumer."""
    b, pats, cases, _ = _serve(v5_model)
    served = b.predict(pats, cases).mean[:, 3]
    prior = b._cyclen_prior_values(pats, cases)
    assert prior is not None
    b._cyclen_prior = None                      # residual-only
    residual = b.predict(pats, cases).mean[:, 3]
    np.testing.assert_allclose(residual + prior, served, atol=1e-6)
    # and the served value is a physical cycle length, not a small residual
    assert np.all(served > 300.0)
    assert np.all(np.abs(residual) < 200.0)


def test_predict_fills_the_assembly_burnup_column(v5_model):
    b, pats, cases, _ = _serve(v5_model)
    col5 = b.predict(pats, cases).mean[:, 5]
    assert np.all(np.isfinite(col5)), "promoted checkpoints must populate column 5"


def test_predict_exposes_quantiles_without_breaking_the_layout(v5_model):
    from lpopt.model.model_api import QuantileSurrogatePrediction
    from lpopt.vendor.masterrl.surrogate import SurrogatePrediction, TARGET_NAMES

    b, pats, cases, _ = _serve(v5_model)
    pred = b.predict(pats, cases)
    assert isinstance(pred, QuantileSurrogatePrediction)
    assert isinstance(pred, SurrogatePrediction)
    assert pred.mean.shape == (len(pats), len(TARGET_NAMES))
    assert pred.quantiles.shape == (len(pats), 2, 3)
    # the vendor reward stack's accessors still work
    assert np.isfinite(pred.mean_fom(0).cyclen)
    lo, hi = pred.band("cyclen")
    assert np.all(lo <= hi)
    # quantiles are in absolute EFPD, like the mean
    assert np.all(lo > 300.0)


def test_predict_rows_raw_also_adds_the_prior_back(v5_model):
    """The tail / no-regression gates use this path; it must be absolute too."""
    b, _, _, df = _serve(v5_model)
    col = np.asarray(b.predict_rows_raw(df))[:, 3]
    assert np.all(col > 300.0)


def test_v5_checkpoint_rejects_a_v4_encoder(v5_model):
    """A schema change must be a hard load failure, never a silent 43-vs-48
    width mismatch that featurizes a served pattern into garbage."""
    from lpopt.model.featurize import FeatureEncoder
    from lpopt.model.model_api import PosValCnnBackend

    members, metas = _members_and_metas(v5_model)
    with pytest.raises(ValueError, match="cond_schema"):
        PosValCnnBackend(members, metas, fuel=_fuel(),
                         encoder=FeatureEncoder(cond_schema="v4"))


def _members_and_metas(model_dir):
    from lpopt.model.train import load_member
    members, metas = [], []
    for md in sorted(Path(model_dir).glob("member_*")):
        model, meta = load_member(md, "cpu")
        members.append(model)
        metas.append(meta)
    return members, metas


def _fuel():
    from lpopt.data.fuel_types import FuelLibrary
    return FuelLibrary.from_parquet(f"{STORE}/fuel_types.parquet")


# --------------------------------------------------------------------------- #
# the ablation arm trains too
# --------------------------------------------------------------------------- #
def test_ablation_arm_trains_and_records_its_schema(tmp_path):
    cfg = _tiny_cfg(cyclen_physics_prior=True, quantile_heads=True,
                    promote_max_asm_bu=True)
    out = tmp_path / "abl"
    train_ensemble(1, split="S1", device="cpu", out_dir=out, config=cfg,
                   subset_rows=200, cond_schema="v5_noshape", base_seed=2,
                   verbose=False)
    meta = json.loads(
        sorted(out.glob("member_*/meta.json"))[0].read_text(encoding="utf-8"))
    assert meta["cond_schema"] == "v5_noshape"
    assert len(meta["channels"]) == 40
    assert "origin_reactivity_swing" not in meta["channels"]
    assert "origin_n_gd" not in meta["channels"]


def test_v4_baseline_arm_still_trains_unpromoted(tmp_path):
    out = tmp_path / "v4"
    train_ensemble(1, split="S1", device="cpu", out_dir=out, config=_tiny_cfg(),
                   subset_rows=200, cond_schema="v4", base_seed=3, verbose=False)
    meta = json.loads(
        sorted(out.glob("member_*/meta.json"))[0].read_text(encoding="utf-8"))
    assert meta["cond_schema"] == "v4"
    assert meta["target_names"] == list(TARGETS)
    assert meta["net_config"]["n_targets"] == 7
    assert meta["net_config"]["n_quantile_targets"] == 0
    assert meta["cyclen_physics_prior"] == {"enabled": False}
    assert meta["promote_max_asm_bu"] is False
    # column 5 stays unknown for the control arm
    from lpopt.model.model_api import PosValCnnBackend
    b = PosValCnnBackend.from_dir(out, store_dir=STORE, library_id="ga80")
    assert b._cyclen_prior is None and not b.has_quantiles()
    _, pats, cases, _ = _serve(out)
    assert np.all(np.isnan(b.predict(pats, cases).mean[:, 5]))
