"""arm 5: the within-cell pairwise rank hinge on the composed ``f_xy`` row.

Prereg Amendment E (``fxy_head_prereg_20260829.md`` §E.3 / §E.8 ①–⑦).  The r2
RANK gate demoted the arm-4 head to a LEVEL estimator (R-a +0.2526 < +0.30, R-b
−0.1157 < 0, R-c 1/13) while its own model-free proxy cleared the same line, and
E.1 traced that to the OBJECTIVE rather than to the features: on the exploit slot
the estimator's own level error (0.0117) is the size of the entire spread it has
to order (0.0114), and a level loss has no reason to shrink within-cell spread
once the cell mean is right — yet a leave-one-WAVE-out ridge on the arm-4 trunk
embedding ranks those same rows at +0.4688.  arm 5 adds the term that was
missing and changes nothing else, so the comparison against arm 4 is paired.

What these tests pin, in the order prereg E.8's "required tests" lists them:

* (a) ``fxy_rank_weight = 0`` is byte-identical — same loss, same parameters;
* (b) pairs form only between LABELLED rows inside one ``(case_pair, feed)`` cell;
* (c) ``min_gap`` filters, and a batch with no qualifying pair contributes 0;
* (e) ``--fxy-rank-weight`` without ``--promote-fxy`` dies at argparse;
* (f) ``--fxy-rank-cell legacy`` selects the ``cyclen_cell`` grouping exactly;

plus the boundary (low-``f_xy``) up-weighting, the ``--fxy-select-band``
selection metric G6a is scored on, the ``--arm5`` recipe composition, and a
2-epoch end-to-end smoke that runs the composed recipe on a synthetic frame and
reads ``fxy_head.rank`` back out of the written ``meta.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.model.al_retrain import (                                 # noqa: E402
    ARM5_FXY_RANK_LOW_THRESH, ARM5_FXY_RANK_LOW_WEIGHT,
    ARM5_FXY_RANK_MARGIN_Z, ARM5_FXY_RANK_MIN_GAP, ARM5_FXY_RANK_WEIGHT,
    ARM5_FXY_SELECT_BAND, plan_al_retrain,
)
from lpopt.model.dataset_torch import (                              # noqa: E402
    TARGETS_WITH_FXY, cyclen_cell_codes, fxy_cell_codes,
)
from lpopt.model.net import PosValNet, PosValNetConfig               # noqa: E402
from lpopt.model.physics_prior import MIN_FXY_LABELS                 # noqa: E402
from lpopt.model.train import (                                      # noqa: E402
    TrainConfig, f_r_rank_loss, f_xy_rank_loss, fxy_metrics,
)

_FXY_IDX = TARGETS_WITH_FXY.index("f_xy")
_FR_IDX = TARGETS_WITH_FXY.index("f_r")


def _t(vals, dtype=torch.float32):
    return torch.tensor(vals, dtype=dtype)


def _cells(vals):
    return torch.tensor(vals, dtype=torch.long)


def _loss(mu, raw, valid, cell, *, margin=0.1, min_gap=0.005,
          low_thresh=1.60, low_weight=3.0, stats=None):
    return f_xy_rank_loss(_t(mu), _t(raw), _t(valid), _cells(cell),
                          margin=margin, min_gap=min_gap,
                          low_thresh=low_thresh, low_weight=low_weight,
                          stats=stats)


# --------------------------------------------------------------------------- #
# 1. the loss term itself (prereg E.8-①; required tests b + c)
# --------------------------------------------------------------------------- #
def test_the_hinge_penalizes_a_wrong_within_cell_order():
    """Two same-cell rows with a real 0.02 f_xy gap; only the ordering differs."""
    raw, valid, cell = [1.62, 1.60], [1.0, 1.0], [0, 0]
    wrong = _loss([0.0, 1.0], raw, valid, cell)     # ranks row 1 above row 0
    right = _loss([1.0, 0.0], raw, valid, cell)     # ranks them as measured
    assert float(wrong) > 0.0
    assert float(right) == pytest.approx(0.0)       # gap 1.0 clears margin 0.1


def test_pairs_form_only_inside_one_gate_cell():
    """(b) The same two rows in DIFFERENT cells are not comparable at all."""
    raw, mu, valid = [1.62, 1.60], [0.0, 1.0], [1.0, 1.0]
    assert float(_loss(mu, raw, valid, [0, 0])) > 0.0
    assert float(_loss(mu, raw, valid, [0, 1])) == 0.0
    # an unresolved cell (-1) is excluded on the same footing
    assert float(_loss(mu, raw, valid, [-1, -1])) == 0.0


def test_pairs_form_only_between_labelled_rows():
    """(b) A masked row carries no f_xy label, so it supplies no ordering."""
    raw, mu, cell = [1.62, 1.60], [0.0, 1.0], [0, 0]
    assert float(_loss(mu, raw, [1.0, 1.0], cell)) > 0.0
    assert float(_loss(mu, raw, [1.0, 0.0], cell)) == 0.0
    assert float(_loss(mu, raw, [0.0, 0.0], cell)) == 0.0


def test_min_gap_drops_orderings_inside_master_repeat_noise():
    """(c) 0.005 sits three orders above the measured FXYP repeat noise
    (0.000000 over 6 re-runs, prereg E.7-②), so a 0.001 "ordering" is not one."""
    mu, valid, cell = [0.0, 1.0], [1.0, 1.0], [0, 0]
    assert float(_loss(mu, [1.601, 1.600], valid, cell)) == 0.0    # gap 0.001
    assert float(_loss(mu, [1.610, 1.600], valid, cell)) > 0.0     # gap 0.010


def test_a_batch_with_no_qualifying_pair_contributes_exactly_zero():
    """(c) Not "small" — exactly 0.0, and detached from the graph entirely, so
    a pair-starved batch cannot even push a zero gradient through the head."""
    mu = torch.zeros(3, requires_grad=True)
    out = f_xy_rank_loss(mu, _t([1.60, 1.60, 1.60]), _t([1.0, 1.0, 1.0]),
                         _cells([0, 0, 0]), margin=0.1, min_gap=0.005,
                         low_thresh=1.60, low_weight=3.0)
    assert float(out) == 0.0
    assert out.grad_fn is None and out.requires_grad is False
    # ... and a single labelled row cannot form a pair either
    assert float(_loss([0.0], [1.60], [1.0], [0])) == 0.0


def test_boundary_pairs_are_up_weighted():
    """A pair whose lower row is in the 1.60 boundary band counts `low_weight`
    times a bulk pair — the band the F_xy <= 1.65 search actually queries."""
    mu4 = _t([0.0, 1.0, 0.0, 1.0])
    raw4 = _t([1.60, 1.55, 1.75, 1.70])       # cell 0 boundary, cell 1 bulk
    v4, c4 = _t([1.0] * 4), _cells([0, 0, 1, 1])
    kw = dict(margin=0.1, min_gap=0.005, low_thresh=1.60)
    # both pairs mis-ordered by the same amount -> the weighted mean is the same
    assert float(f_xy_rank_loss(mu4, raw4, v4, c4, low_weight=3.0, **kw)) == \
        pytest.approx(float(f_xy_rank_loss(mu4, raw4, v4, c4,
                                           low_weight=1.0, **kw)))
    # fix the BOUNDARY pair only: up-weighting must make that fix count more
    mu_fix_low = _t([1.0, 0.0, 0.0, 1.0])
    assert float(f_xy_rank_loss(mu_fix_low, raw4, v4, c4, low_weight=3.0, **kw)) \
        < float(f_xy_rank_loss(mu_fix_low, raw4, v4, c4, low_weight=1.0, **kw))
    # fix the BULK pair instead: up-weighting must make that fix count less
    mu_fix_bulk = _t([0.0, 1.0, 1.0, 0.0])
    assert float(f_xy_rank_loss(mu_fix_bulk, raw4, v4, c4, low_weight=3.0, **kw)) \
        > float(f_xy_rank_loss(mu_fix_bulk, raw4, v4, c4, low_weight=1.0, **kw))


def test_the_hinge_is_scale_free_in_the_target():
    """Only the SIGN of the raw gap enters, so the term cannot be tuned by the
    global f_xy z-scale — the property both existing hinges rely on."""
    mu, valid, cell = [0.0, 1.0], [1.0, 1.0], [0, 0]
    assert float(_loss(mu, [1.62, 1.60], valid, cell)) == pytest.approx(
        float(_loss(mu, [2.40, 1.60], valid, cell)))


def test_the_math_is_the_registered_f_r_hinge():
    """E.8-① registers this as a structural copy of `f_r_rank_loss`; it IS that
    function, so the two cannot drift apart under a later edit to either."""
    kw = dict(margin=0.1, min_gap=0.005, low_thresh=1.60, low_weight=3.0)
    mu, raw = _t([0.0, 1.0, 0.3]), _t([1.62, 1.60, 1.70])
    v, c = _t([1.0, 1.0, 1.0]), _cells([0, 0, 0])
    assert float(f_xy_rank_loss(mu, raw, v, c, **kw)) == float(
        f_r_rank_loss(mu, raw, v, c, **kw))


def test_the_pair_census_counts_pairs_and_contributing_cells():
    """E.8-⑦: a hinge that silently sees no pair trains to the same 0.0 as a
    satisfied one, so the census is the only instrument separating them."""
    stats: dict[str, float] = {}
    # cell 0: rows 0,1 gap 0.02 -> 1 pair.  cell 1: rows 2,3 gap 0.001 -> none.
    _loss([0.0, 1.0, 0.0, 1.0], [1.62, 1.60, 1.601, 1.600], [1.0] * 4,
          [0, 0, 1, 1], stats=stats)
    assert stats == {"n_pairs": 1.0, "n_cells": 1.0}
    starved: dict[str, float] = {}
    _loss([0.0, 1.0], [1.600, 1.600], [1.0, 1.0], [0, 0], stats=starved)
    assert starved == {"n_pairs": 0.0, "n_cells": 0.0}


# --------------------------------------------------------------------------- #
# 2. config + CLI plumbing (prereg E.8-④/⑤/⑥; required test e)
# --------------------------------------------------------------------------- #
def test_defaults_are_the_pre_arm5_training_path():
    """(a) Every new knob defaults to the value that makes the term not exist,
    and the five registered constants are pinned so a later edit is a failure."""
    cfg = TrainConfig()
    assert cfg.fxy_rank_weight == 0.0
    assert cfg.fxy_select_band == 1.0
    assert cfg.fxy_rank_margin_z == 0.1
    assert cfg.fxy_rank_min_gap == 0.005
    assert cfg.fxy_rank_low_thresh == 1.60
    assert cfg.fxy_rank_low_weight == 3.0
    assert cfg.fxy_rank_cell == "gate"


def test_cli_threads_the_seven_new_flags_into_the_config(monkeypatch):
    """The flags must reach `TrainConfig`, not merely parse."""
    from lpopt.model import train as T

    seen: dict[str, TrainConfig] = {}

    def _capture(*_a, **kw):
        seen["cfg"] = kw["config"]
        return []

    monkeypatch.setattr(T, "train_ensemble", _capture)
    T.main(["--promote-fxy",
            "--fxy-rank-weight", "3.0", "--fxy-rank-cell", "gate",
            "--fxy-rank-margin-z", "0.1", "--fxy-rank-min-gap", "0.005",
            "--fxy-rank-low-thresh", "1.60", "--fxy-rank-low-weight", "3.0",
            "--fxy-select-band", "0.50"])
    cfg = seen["cfg"]
    assert cfg.fxy_rank_weight == 3.0
    assert cfg.fxy_rank_cell == "gate"
    assert cfg.fxy_rank_margin_z == 0.1
    assert cfg.fxy_rank_min_gap == 0.005
    assert cfg.fxy_rank_low_thresh == 1.60
    assert cfg.fxy_rank_low_weight == 3.0
    assert cfg.fxy_select_band == 0.50


def test_an_unflagged_run_still_gets_the_legacy_config(monkeypatch):
    """(a) The same entry point with no arm-5 flag composes the old config."""
    from lpopt.model import train as T

    seen: dict[str, TrainConfig] = {}

    def _capture(*_a, **kw):
        seen["cfg"] = kw["config"]
        return []

    monkeypatch.setattr(T, "train_ensemble", _capture)
    T.main(["--promote-fxy"])
    assert seen["cfg"].fxy_rank_weight == 0.0
    assert seen["cfg"].fxy_select_band == 1.0


def test_the_rank_hinge_refuses_to_run_without_the_head():
    """(e) There is no f_xy row to rank without `--promote-fxy`; a run that
    quietly trained the seven legacy targets under an arm-5 command line would be
    the same silent-drift class the cond_schema/head_hidden guards closed."""
    from lpopt.model import train as T

    with pytest.raises(SystemExit):
        T.main(["--fxy-rank-weight", "3.0"])
    # a negative weight is a typo, not a request to ascend the loss
    with pytest.raises(SystemExit):
        T.main(["--promote-fxy", "--fxy-rank-weight", "-1.0"])
    # the band is a quantile, and the cell is one of two registered partitions
    with pytest.raises(SystemExit):
        T.main(["--promote-fxy", "--fxy-select-band", "0.0"])
    with pytest.raises(SystemExit):
        T.main(["--promote-fxy", "--fxy-select-band", "1.5"])
    with pytest.raises(SystemExit):
        T.main(["--promote-fxy", "--fxy-rank-cell", "elite"])


def test_legacy_cell_selects_the_cyclen_grouping_exactly():
    """(f) `--fxy-rank-cell legacy` is the escape hatch back to the OLD
    partition, and it must be that partition, not an approximation of it."""
    df = pd.DataFrame({
        "case_pair": ["E1_E2", "T6_T4", "J5_J6"],   # three distinct gate cells
        "feed": [121, 121, 121],
        "e_core": [5.12, 5.13, 5.62],               # rows 0,1 share an e_core bin
        "dataset": ["B", "B", "B"],
    })
    legacy = torch.as_tensor(cyclen_cell_codes(df), dtype=torch.long)
    gate = torch.as_tensor(fxy_cell_codes(df), dtype=torch.long)
    assert legacy.tolist() == [0, 0, 1] and gate.tolist() == [0, 1, 2]
    mu, raw, valid = _t([0.0, 1.0, 0.5]), _t([1.62, 1.60, 1.58]), _t([1.0] * 3)
    kw = dict(margin=0.1, min_gap=0.005, low_thresh=1.60, low_weight=3.0)
    # rows 0 and 1 share the legacy cell but NOT the gate cell, so the one
    # mis-ranked pair is visible under `legacy` and invisible under `gate`.
    assert float(f_xy_rank_loss(mu, raw, valid, legacy, **kw)) > 0.0
    assert float(f_xy_rank_loss(mu, raw, valid, gate, **kw)) == 0.0


# --------------------------------------------------------------------------- #
# 3. the selection band (prereg E.8-⑥, the axis clause G6a is scored on)
# --------------------------------------------------------------------------- #
def _band_frame() -> pd.DataFrame:
    """Two GATE cells that share ONE case_pair, so the grouping is observable."""
    rows = []
    for feed, base in ((121, 1.50), (109, 1.60)):
        for j in range(6):
            rows.append({"record_id": f"{feed}-{j}", "case_pair": "E1_E2",
                         "feed": feed, "f_xy": base + 0.01 * j})
    return pd.DataFrame(rows)


def _band_pred(df: pd.DataFrame, pred_fxy: np.ndarray) -> dict[str, np.ndarray]:
    n, t = len(df), len(TARGETS_WITH_FXY)
    mu_z = np.zeros((1, n, t))
    mu_z[0, :, _FXY_IDX] = pred_fxy                # tmean 0 / tstd 1 below
    targets = np.zeros((n, t))
    targets[:, _FXY_IDX] = df["f_xy"].to_numpy(float)
    mask = np.zeros((n, t))
    mask[:, _FXY_IDX] = 1.0
    return {"mu_z_members": mu_z, "targets": targets, "target_mask": mask,
            "record_ids": df["record_id"].to_numpy(dtype=object)}


def _metrics(df, pred, min_case, **kw):
    z = np.zeros(len(TARGETS_WITH_FXY))
    return fxy_metrics(pred, df, z, np.ones(len(TARGETS_WITH_FXY)),
                       TARGETS_WITH_FXY, min_case, **kw)


def test_band_one_is_the_legacy_metric_untouched():
    df = _band_frame()
    pred = _band_pred(df, df["f_xy"].to_numpy(float))
    legacy = _metrics(df, pred, 3)
    assert legacy == _metrics(df, pred, 3, band=1.0)
    assert "within_cell_spearman_f_xy_band" not in legacy


def test_the_band_scores_each_gate_cells_low_half_only():
    df = _band_frame()
    pred_vals = df["f_xy"].to_numpy(float).copy()
    for feed in (121, 109):                         # invert the UPPER half only
        upper = np.where(df["feed"].to_numpy() == feed)[0][3:]
        pred_vals[upper] = pred_vals[upper][::-1] + 10.0
    out = _metrics(df, _band_pred(df, pred_vals), 3, band=0.50)
    assert out["n_fxy_band_cells"] == 2.0           # 2 gate cells ...
    assert out["n_fxy_band_rows"] == 6.0            # ... x their 3 lowest rows
    assert out["within_cell_spearman_f_xy_band"] == pytest.approx(1.0)
    assert out["within_cell_spearman_f_xy"] < 1.0   # the whole cell is not
    # selection reads the BAND axis, not the whole-cell one
    assert out["fxy_select"] == pytest.approx(
        out["within_cell_spearman_f_xy_band"] - out["z_mae_f_xy"])


def test_the_band_splits_a_case_pair_by_feed():
    """The registered mismatch E.8-⑥ closes: the legacy metric groups on
    `case_pair` alone while every gate groups on `(case_pair, feed)`."""
    df = _band_frame()
    pred = _band_pred(df, df["f_xy"].to_numpy(float))
    assert _metrics(df, pred, 3)["n_fxy_cells"] == 1.0           # one case_pair
    assert _metrics(df, pred, 3, band=0.50)["n_fxy_band_cells"] == 2.0


def test_a_thin_cell_is_dropped_not_pooled():
    """Pooling elite rows across cells measures the between-cell LEVEL spread,
    which prereg E.2.2 forbids as a statistic."""
    df = _band_frame()
    out = _metrics(df, _band_pred(df, df["f_xy"].to_numpy(float)), 4, band=0.50)
    assert out["n_fxy_band_cells"] == 0.0           # 3 banded rows < min_case 4
    assert np.isnan(out["within_cell_spearman_f_xy_band"])
    assert np.isnan(out["fxy_select"])              # -> falls back to composite


# --------------------------------------------------------------------------- #
# 4. the arm-5 recipe (prereg E.3.2 [3])
# --------------------------------------------------------------------------- #
def _mock_champion(tmp_path):
    """A champion checkpoint carrying the arm-4 recipe (member meta only)."""
    d = tmp_path / "s1j"
    (d / "member_0").mkdir(parents=True)
    (d / "ensemble.json").write_text(json.dumps(
        {"members": ["member_0"], "n_members": 5, "split": "S1j",
         "base_seed": 0}), encoding="utf-8")
    (d / "member_0" / "meta.json").write_text(json.dumps({
        "cond_schema": "v8",
        "target_names": list(TARGETS_WITH_FXY),
        "net_config": {"width": 224, "n_blocks": 8, "head_hidden": 384,
                       "map_head_mode": "multiscale", "map_prior_channel": 0},
        "train_config": {"epochs": 150, "cyclen_physics_prior": True,
                         "quantile_heads": True, "quantile_weight": 0.2,
                         "distill_weight": 0.4, "distill_min_match_frac": 0.5,
                         "promote_max_asm_bu": True, "map_spectral_weight": 0.3,
                         "map_peak_weight": 2.0, "num_workers": 8},
    }), encoding="utf-8")
    return str(d)


def test_arm5_composes_the_prereg_command(tmp_path):
    args = plan_al_retrain(_mock_champion(tmp_path), arm5=True)["train_args"]
    # arm 4's flags, verbatim -- the comparison is paired or it is nothing
    assert args[args.index("--trunk-finetune-lr-mult") + 1] == "0.05"
    assert "--fxy-prior-on-predicted" in args
    assert args[args.index("--fxy-select-weight") + 1] == "0.5"
    assert args[args.index("--warmup-epochs") + 1] == "2"
    assert args[args.index("--f-r-rank-weight") + 1] == "0.1"
    assert "--promote-fxy" in args and "--promote-max-asm-bu" in args
    # ... plus exactly the seven new ones, at their REGISTERED values
    assert float(args[args.index("--fxy-rank-weight") + 1]) == ARM5_FXY_RANK_WEIGHT
    assert args[args.index("--fxy-rank-cell") + 1] == "gate"
    assert float(args[args.index("--fxy-rank-margin-z") + 1]) == ARM5_FXY_RANK_MARGIN_Z
    assert float(args[args.index("--fxy-rank-min-gap") + 1]) == ARM5_FXY_RANK_MIN_GAP
    assert float(args[args.index("--fxy-rank-low-thresh") + 1]) == ARM5_FXY_RANK_LOW_THRESH
    assert float(args[args.index("--fxy-rank-low-weight") + 1]) == ARM5_FXY_RANK_LOW_WEIGHT
    assert float(args[args.index("--fxy-select-band") + 1]) == ARM5_FXY_SELECT_BAND


def test_arm5_is_arm4_plus_a_suffix_and_the_same_teacher(tmp_path):
    """E.3: the distill teacher is arm 4's.  A teacher refreshed from anywhere
    else would move the gate independently of the objective under test."""
    champ = _mock_champion(tmp_path)
    a4, a5 = plan_al_retrain(champ, arm4=True), plan_al_retrain(champ, arm5=True)
    assert a5["distill_teacher_refresh"] == a4["distill_teacher_refresh"]
    assert a5["train_args"][:len(a4["train_args"])] == a4["train_args"]
    assert a5["train_args"][len(a4["train_args"]):] == [
        "--fxy-rank-weight", "3.0", "--fxy-rank-cell", "gate",
        "--fxy-rank-margin-z", "0.1", "--fxy-rank-min-gap", "0.005",
        "--fxy-rank-low-thresh", "1.6", "--fxy-rank-low-weight", "3.0",
        "--fxy-select-band", "0.5"]


def test_arms_1_to_4_are_untouched_by_arm5(tmp_path):
    """arm 1-4 verdicts are final; their composed commands must not move."""
    champ = _mock_champion(tmp_path)
    for kwargs in ({}, {"add_fxy_head": True}, {"arm4": True}):
        args = plan_al_retrain(champ, **kwargs)["train_args"]
        assert not any(a.startswith("--fxy-rank") for a in args)
        assert "--fxy-select-band" not in args


# --------------------------------------------------------------------------- #
# 5. byte-identity of the flag-off optimizer step (required test a)
# --------------------------------------------------------------------------- #
_N_CH, _N_G = 4, 3


def _net_cfg(**kw) -> PosValNetConfig:
    base = dict(in_channels=_N_CH, n_globals=_N_G, width=16, n_blocks=2,
                groups=4, head_hidden=16, n_targets=len(TARGETS_WITH_FXY),
                fxy_target_idx=_FXY_IDX, fxy_ref_idx=_FR_IDX,
                fxy_prior_a=0.9, fxy_prior_b=0.1)
    base.update(kw)
    return PosValNetConfig(**base)


@pytest.fixture(scope="module")
def step_batch():
    """One synthetic batch: 3 gate cells x 8 labelled rows, orderable."""
    rng = np.random.default_rng(0)
    n, n_t = 24, len(TARGETS_WITH_FXY)
    targets = rng.normal(size=(n, n_t)).astype(np.float32)
    targets[:, _FXY_IDX] = 1.50 + 0.01 * (np.arange(n) % 8)
    cells = torch.as_tensor(rng.normal(size=(n, _N_CH, 19, 19)),
                            dtype=torch.float32)
    cells[:, 0] = 1.0                                   # channel 0 == fuel_mask
    return {
        "cells": cells,
        "globals": torch.as_tensor(rng.normal(size=(n, _N_G)),
                                   dtype=torch.float32),
        "targets": torch.as_tensor(targets),
        "target_mask": torch.ones((n, n_t)),
        "conv_label": torch.ones(n),
        "conv_mask": torch.ones(n),
        "maps": torch.as_tensor(rng.normal(size=(n, 4, 9, 9)),
                                dtype=torch.float32),
        "maps_mask": torch.ones((n, 4, 9, 9)),
        "fxy_cell": torch.as_tensor(np.arange(n) // 8, dtype=torch.long),
        "cyclen_cell": torch.as_tensor(np.arange(n) // 12, dtype=torch.long),
    }


def _stepped(cfg: TrainConfig, batch):
    """One `_step_member` on a freshly seeded member; returns that member."""
    from lpopt.model import train as T

    n_t = len(TARGETS_WITH_FXY)
    torch.manual_seed(11)
    m = T._MemberState()
    m.model = PosValNet(_net_cfg())
    m.fwd = m.model
    m.optim = torch.optim.AdamW(m.model.parameters(), lr=1e-2)
    m.norm = T._Norm(np.zeros(n_t, dtype=np.float32),
                     np.ones(n_t, dtype=np.float32),
                     np.zeros(4, dtype=np.float32),
                     np.ones(4, dtype=np.float32), torch.device("cpu"))
    m.use_nll = False
    m.use_prior = False
    m.q_idx = None
    m.fxy_idx = _FXY_IDX
    m.running = 0.0
    m.n_batches = 0
    m.model.train()
    T._step_member(m, batch, cfg, False, torch.device("cpu"))
    return m


def test_weight_zero_is_the_same_optimizer_step(step_batch):
    """(a) Flag off -> the term is not merely small, it is not built: the loss
    and every resulting parameter are identical to the pre-arm-5 path."""
    off = _stepped(TrainConfig(promote_fxy=True), step_batch)
    zero = _stepped(TrainConfig(promote_fxy=True, fxy_rank_weight=0.0),
                    step_batch)
    assert zero.running == off.running
    for (k, a), (_, b) in zip(off.model.state_dict().items(),
                              zero.model.state_dict().items()):
        assert torch.equal(a, b), k
    assert getattr(zero, "fxy_rank_batches", 0) == 0    # the census never ran


def test_weight_on_changes_the_step_and_counts_pairs(step_batch):
    """The other half of (a): the flag is not a no-op when it IS set."""
    off = _stepped(TrainConfig(promote_fxy=True), step_batch)
    on = _stepped(TrainConfig(promote_fxy=True, fxy_rank_weight=3.0), step_batch)
    assert on.running != off.running
    assert on.fxy_rank_batches == 1
    assert on.fxy_rank_pairs > 0.0            # 3 cells x C(8,2) real orderings
    assert on.fxy_rank_cells == 3.0
    assert any(not torch.equal(a, b) for a, b in
               zip(off.model.state_dict().values(),
                   on.model.state_dict().values()))


def test_a_run_without_the_head_index_never_builds_the_term(step_batch):
    """`--promote-fxy` is enforced at the CLI, but the loop is also defensive:
    with no resolved f_xy row there is nothing to rank."""
    from lpopt.model import train as T

    cfg = TrainConfig(promote_fxy=True, fxy_rank_weight=3.0)
    m = _stepped(cfg, step_batch)
    assert m.fxy_rank_batches == 1
    m.fxy_idx = -1                                   # simulate "head absent"
    m.fxy_rank_batches = 0
    T._step_member(m, step_batch, cfg, False, torch.device("cpu"))
    assert m.fxy_rank_batches == 0
    # ... and so is a batch built before the cell column existed
    no_cell = {k: v for k, v in step_batch.items() if k != "fxy_cell"}
    m.fxy_idx = _FXY_IDX
    T._step_member(m, no_cell, cfg, False, torch.device("cpu"))
    assert m.fxy_rank_batches == 0


def test_the_hinge_cannot_move_the_f_r_head(step_batch):
    """`_compose_fxy` detaches `mu[f_r]`, so a ~2%-labelled ranking term is
    structurally unable to perturb the seven dense targets through the prior."""
    torch.manual_seed(11)
    model = PosValNet(_net_cfg())
    out = model(step_batch["cells"], step_batch["globals"])
    loss = f_xy_rank_loss(
        out["mu"][:, _FXY_IDX], step_batch["targets"][:, _FXY_IDX],
        step_batch["target_mask"][:, _FXY_IDX], step_batch["fxy_cell"],
        margin=0.1, min_gap=0.005, low_thresh=1.60, low_weight=3.0)
    assert float(loss.detach()) > 0.0
    loss.backward()
    grad = model.mu_head.weight.grad
    assert torch.all(grad[_FR_IDX] == 0.0), "the f_r row must not move"
    assert torch.any(grad[_FXY_IDX] != 0.0), "the f_xy row must"


# --------------------------------------------------------------------------- #
# 6. end-to-end smoke: the composed recipe runs and stamps the meta (E.8-⑦)
#
# Synthetic and 2 epochs on purpose — the point is that every wire is connected
# (dataset column -> batch key -> loss term -> census -> band selection ->
# meta.json), not that anything is fitted.  `--init-from` is the one arm-5 flag
# NOT exercised here: it is arm 4's, already covered by test_fxy_head.py §13,
# and there is no champion checkpoint to fine-tune from in a synthetic fixture.
# --------------------------------------------------------------------------- #
def _synthetic_fold(n_cells: int, per_cell: int, seed: int):
    """A PrecomputedDataset over a synthetic frame with dense f_xy labels."""
    from lpopt.model.train import PrecomputedDataset

    rng = np.random.default_rng(seed)
    n = n_cells * per_cell
    n_t = len(TARGETS_WITH_FXY)
    cell_of = np.arange(n) // per_cell
    within = np.arange(n) % per_cell
    f_r = 1.55 + 0.01 * within + 0.002 * cell_of
    # f_xy = 1.05*f_r + a within-cell residual the prior alone cannot express
    f_xy = 1.05 * f_r - 0.10 + 0.004 * ((within * 7) % per_cell)
    df = pd.DataFrame({
        "record_id": [f"r{seed}-{i}" for i in range(n)],
        "case_pair": [f"C{c}" for c in cell_of],
        "feed": 121,
        "e_core": 5.12 + 0.10 * cell_of,
        "dataset": "B",
        "converged": True,
        "valid": True,
        "maps_key": np.nan,
        "f_r": f_r, "f_xy": f_xy,
        "f_q": 2.3, "cbc_max": 1400.0, "cyclen": 690.0, "ao_abs": 0.1,
        "discharge_burnup": 53.0, "max_pin_burnup": 70.0,
    })
    targets = np.zeros((n, n_t), dtype=np.float32)
    for j, name in enumerate(TARGETS_WITH_FXY):
        targets[:, j] = df[name].to_numpy(float) + rng.normal(0, 1e-3, n)
    cells = torch.as_tensor(rng.normal(size=(n, _N_CH, 19, 19)),
                            dtype=torch.float32)
    cells[:, 0] = 1.0
    tensors = {
        "cells": cells,
        "globals": torch.as_tensor(rng.normal(size=(n, _N_G)),
                                   dtype=torch.float32),
        "targets": torch.as_tensor(targets),
        "target_mask": torch.ones((n, n_t)),
        "conv_label": torch.ones(n),
        "conv_mask": torch.ones(n),
        "maps": torch.full((n, 4, 9, 9), float("nan")),
        "maps_mask": torch.zeros((n, 4, 9, 9)),
        "cyclen_cell": torch.as_tensor(cyclen_cell_codes(df), dtype=torch.long),
        "fxy_cell": torch.as_tensor(fxy_cell_codes(df), dtype=torch.long),
    }
    return PrecomputedDataset(tensors, df["record_id"].tolist(), df)


def test_two_epoch_smoke_runs_the_arm5_recipe(tmp_path):
    """Runs on GPU 1 when CUDA is visible (`CUDA_VISIBLE_DEVICES=1`), else CPU."""
    from lpopt.model import train as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_cell = 24
    n_cells = -(-(MIN_FXY_LABELS + 40) // per_cell)     # clear the label guard
    train_ds = _synthetic_fold(n_cells, per_cell, seed=0)
    val_ds = _synthetic_fold(4, per_cell, seed=1)
    assert len(train_ds) >= MIN_FXY_LABELS

    cfg = TrainConfig(
        epochs=2, warmup_epochs=1, batch_size=64, augment=False,
        min_case_val=5, map_norm_subset=4, round_trip_rows=2,
        promote_fxy=True, fxy_select_weight=0.5, fxy_select_band=0.50,
        fxy_rank_weight=3.0, fxy_rank_cell="gate",
        width=16, n_blocks=2, head_hidden=16)
    cfg.auto_fit_cell_calibration = False
    enc = T.FeatureEncoder()
    eff, lr, lr_final, warm, sched_meta = T._resolve_schedule(cfg, device)

    members = T._train_members(
        [3], train_ds=train_ds, val_ds=val_ds, cfg=cfg, device=device,
        globals_names=[f"g{i}" for i in range(_N_G)], reader=None,
        eff_batch=eff, lr=lr, lr_final=lr_final, warm=warm, resident=False,
        compile_flag=False, n_channels=_N_CH,
        channel_names=tuple(f"c{i}" for i in range(_N_CH)), verbose=False,
        target_names=TARGETS_WITH_FXY)
    m = members[0]

    # the term ran, on real pairs, in every epoch
    assert len(m.history) == 2
    for h in m.history:
        assert h["fxy_rank_pairs"] > 0.0, "the hinge saw no pair"
        assert h["fxy_rank_cells"] > 0.0
        # selection read the BAND axis G6a is scored on
        assert "within_cell_spearman_f_xy_band" in h
        assert h["n_fxy_band_cells"] > 0.0

    member_dir = T._finalize_member(
        tmp_path, m, cfg=cfg, split="SYN",
        globals_names=[f"g{i}" for i in range(_N_G)], encoder=enc,
        train_ds=train_ds, val_ds=val_ds, device=device,
        sched_meta=sched_meta, resident=False, target_names=TARGETS_WITH_FXY)
    meta = json.loads((member_dir / "meta.json").read_text(encoding="utf-8"))

    rank = meta["fxy_head"]["rank"]
    assert rank["enabled"] is True
    assert rank["weight"] == 3.0
    assert rank["cell"] == "gate"
    assert rank["margin_z"] == 0.1
    assert rank["min_gap"] == 0.005
    assert rank["low_thresh"] == 1.60
    assert rank["low_weight"] == 3.0
    assert rank["mean_pairs_per_batch"] > 0.0
    assert rank["mean_cells_per_batch"] > 0.0
    assert meta["fxy_head"]["select_band"] == 0.50
    # the new knobs are also in the reproducible recipe
    assert meta["train_config"]["fxy_rank_weight"] == 3.0
    assert meta["train_config"]["fxy_select_band"] == 0.50
