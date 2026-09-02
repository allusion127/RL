"""The ``f_xy`` prior-residual head (F_xy switch, phase P4).

``f_xy`` is MASTER's FXYP — pin PLANAR peaking, the new optimization objective
(design 20260829, hard limit 1.65).  It is promoted exactly the way
``max_assembly_burnup`` was (APPENDED to the dataset target tuple, one more row
of the existing mu / log_sigma heads, masked wherever the label is absent), with
three things that are new and are what these tests pin down:

1. **The label is sparse** — ~2% of store rows carry it today, so every test
   below runs against a 95%-NaN label column and asserts the masked loss and the
   masked metrics ignore the rest.  Below ``MIN_FXY_LABELS`` the trainer refuses.
2. **The head regresses a RESIDUAL** against the measured ``F_xy ~ F_r`` affine,
   evaluated on the model's own (detached) F_r prediction, so the sparse head
   inherits the dense F_r head's accuracy and cannot perturb it.
3. **It is served OUTSIDE the frozen 7-column surrogate** via ``predict_fxy``,
   which returns ``None`` — not a NaN column — on a checkpoint with no head, so a
   caller can tell "cannot answer" from "answered NaN" (design §3.6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lpopt.data.fuel_types import FuelLibrary                        # noqa: E402
from lpopt.data.schema import unpack_pattern                         # noqa: E402
from lpopt.data.store import StoreReader                             # noqa: E402
from lpopt.model.al_retrain import (                                 # noqa: E402
    champion_recipe, plan_al_retrain, recipe_to_train_args,
)
from lpopt.model.calibrate import apply_calibration                  # noqa: E402
from lpopt.model.conformal import CONFORMAL_TARGETS                  # noqa: E402
from lpopt.model.dataset_torch import (                              # noqa: E402
    LPDataset, TARGETS, TARGETS_WITH_ASM_BU, TARGETS_WITH_FXY, targets_for,
)
from lpopt.model.model_api import (                                  # noqa: E402
    PosValCnnBackend, _TARGET_TO_SURROGATE_COL, _to_surrogate,
)
from lpopt.model.net import PosValNet, PosValNetConfig               # noqa: E402
from lpopt.model.physics_prior import (                              # noqa: E402
    MIN_FXY_LABELS, FxyFrPrior, fit_fxy_prior, fxy_prior_z,
)
from lpopt.model.train import (                                      # noqa: E402
    TrainConfig, _graft_appended_target_rows, compute_target_norm, fxy_metrics,
    refit_fxy_prior_on_predicted, regression_loss, resolve_fxy_prior,
    save_member,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"

# 8-target z-score (TARGETS + f_xy), physical units.
_ZMEAN_FXY = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0, 1.66]
_ZSTD_FXY = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0, 0.06]


# --------------------------------------------------------------------------- #
# synthetic sparse-label frame (95% NaN f_xy)
# --------------------------------------------------------------------------- #
def _sparse_frame(n: int = 800, labelled: int = 40, seed: int = 0
                  ) -> pd.DataFrame:
    """A store-shaped frame whose ``f_xy`` column is 95% NaN by construction."""
    rng = np.random.default_rng(seed)
    f_r = 1.45 + 0.25 * rng.random(n)
    f_xy = np.full(n, np.nan)
    idx = rng.choice(n, labelled, replace=False)
    # the measured pooled relation, plus its measured residual sd (0.029)
    f_xy[idx] = 1.1221 * f_r[idx] - 0.0831 + rng.normal(0.0, 0.029, labelled)
    return pd.DataFrame({
        "record_id": [f"r{i:04d}" for i in range(n)],
        "converged": np.ones(n, dtype=bool),
        "f_r": f_r,
        "f_q": 1.16 * f_r,
        "cbc_max": 1400.0 + rng.normal(0, 50, n),
        "cyclen": 690.0 + rng.normal(0, 15, n),
        "ao_abs": np.abs(rng.normal(0, 0.05, n)),
        "discharge_burnup": 53.0 + rng.normal(0, 1, n),
        "max_pin_burnup": 70.0 + rng.normal(0, 2, n),
        "f_xy": f_xy,
        "case_pair": ["A_B"] * n,
    })


# --------------------------------------------------------------------------- #
# 1. target inventory — APPEND only
# --------------------------------------------------------------------------- #
def test_promotion_appends_and_never_reorders():
    assert targets_for() == TARGETS
    assert targets_for(promote_fxy=True) == TARGETS_WITH_FXY
    assert TARGETS_WITH_FXY[:7] == TARGETS
    assert TARGETS_WITH_FXY[7] == "f_xy"
    # cyclen == 3 is keyed on by the rank loss and the cell calibration.
    assert TARGETS_WITH_FXY.index("cyclen") == 3
    assert TARGETS_WITH_FXY.index("f_r") == 0


def test_both_promotions_compose_as_strict_prefixes():
    both = targets_for(promote_max_asm_bu=True, promote_fxy=True)
    assert both == TARGETS_WITH_ASM_BU + ("f_xy",)
    assert len(both) == 9
    # the prefix property is what makes the --init-from graft legal
    assert both[:8] == TARGETS_WITH_ASM_BU
    assert both[:7] == TARGETS


def test_train_config_defaults_to_unpromoted():
    assert TrainConfig().promote_fxy is False


def test_f_xy_has_no_surrogate_column():
    """It is served OUTSIDE the frozen 7-column contract, never through one of
    the vendor's constraint axes (the discharge_burnup lesson)."""
    assert "f_xy" not in _TARGET_TO_SURROGATE_COL
    out = _to_surrogate(np.arange(8, dtype=float).reshape(1, 8),
                        list(TARGETS_WITH_FXY))
    assert out.shape == (1, 7)
    assert np.isnan(out[0, 5])          # max_assembly_burnup still unknown


# --------------------------------------------------------------------------- #
# 2. dataset masking on a 95%-NaN label
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def store_bits():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    reader = StoreReader(STORE)
    fuel = FuelLibrary.from_parquet(STORE / "fuel_types.parquet")
    ids = reader.records["record_id"].astype(str).tolist()[:200]
    return reader, fuel, ids


def test_dataset_width_and_prefix_are_unchanged_by_promotion(store_bits):
    reader, fuel, ids = store_bits
    off = LPDataset(reader, ids, fuel, fold="train")
    on = LPDataset(reader, ids, fuel, fold="train", promote_fxy=True)
    assert off[0]["targets"].shape == (7,)
    assert on[0]["targets"].shape == (8,)
    for i in range(min(30, len(off))):
        a, b = off[i], on[i]
        torch.testing.assert_close(a["targets"], b["targets"][:7], equal_nan=True)
        torch.testing.assert_close(a["target_mask"], b["target_mask"][:7])


def test_absent_f_xy_labels_are_masked(store_bits):
    """~98% of store rows carry no FXYP; those must be masked, not trained on."""
    reader, fuel, ids = store_bits
    ds = LPDataset(reader, ids, fuel, fold="train", promote_fxy=True)
    n_masked = 0
    for i in range(len(ds)):
        item, row = ds[i], ds.df.iloc[i]
        raw = row.get("f_xy")
        finite = raw is not None and np.isfinite(float(raw))
        expect = 1.0 if (finite and bool(row["converged"])) else 0.0
        assert float(item["target_mask"][7]) == expect
        n_masked += expect == 0.0
    assert n_masked > 0, "no unlabelled rows in the sample — test is vacuous"


def test_masked_rows_are_excluded_from_the_z_score():
    df = _sparse_frame()
    mean, std = compute_target_norm(df, TARGETS_WITH_FXY)
    vals = df["f_xy"].to_numpy(float)
    ok = np.isfinite(vals)
    assert mean[7] == pytest.approx(float(vals[ok].mean()))
    assert std[7] == pytest.approx(float(vals[ok].std()))
    m7, s7 = compute_target_norm(df, TARGETS)
    np.testing.assert_allclose(mean[:7], m7)
    np.testing.assert_allclose(std[:7], s7)


def test_masked_loss_ignores_the_nan_rows():
    """The NaN f_xy labels must contribute NOTHING — not a poisoned mean, and not
    a value that changes when the unlabelled entries change."""
    torch.manual_seed(0)
    n, t = 64, 8
    mu = torch.randn(n, t, requires_grad=True)
    log_sigma = torch.zeros(n, t)
    target = torch.randn(n, t)
    mask = torch.ones(n, t)
    target[:, 7] = float("nan")
    mask[:, 7] = 0.0
    mask[:3, 7] = 1.0                      # 3 labelled rows out of 64
    target[:3, 7] = torch.tensor([0.1, -0.2, 0.3])

    loss = regression_loss(mu, log_sigma, target, mask,
                           use_nll=True, beta=0.5, delta=1.0)
    assert torch.isfinite(loss)
    # perturbing an UNLABELLED f_xy label cannot move the loss
    other = target.clone()
    other[10:, 7] = 1e6
    loss2 = regression_loss(mu, log_sigma, other, mask,
                            use_nll=True, beta=0.5, delta=1.0)
    torch.testing.assert_close(loss, loss2)
    # and the gradient on the unlabelled f_xy rows is exactly zero
    loss.backward()
    assert torch.all(mu.grad[10:, 7] == 0.0)
    assert torch.any(mu.grad[:3, 7] != 0.0)


def test_fxy_metrics_score_only_the_labelled_subset():
    df = _sparse_frame(n=40, labelled=8)
    n, t = 40, 8
    rng = np.random.default_rng(1)
    pred = {
        "mu_z_members": rng.normal(size=(2, n, t)),
        "targets": np.zeros((n, t)),
        "target_mask": np.zeros((n, t)),
        "record_ids": df["record_id"].tolist(),
    }
    labelled = np.isfinite(df["f_xy"].to_numpy(float))
    pred["target_mask"][:, 7] = labelled.astype(float)
    pred["targets"][:, 7] = np.nan_to_num(df["f_xy"].to_numpy(float))
    out = fxy_metrics(pred, df, np.zeros(t), np.ones(t), TARGETS_WITH_FXY, 5)
    assert out["n_fxy_val"] == float(labelled.sum()) == 8.0
    assert np.isfinite(out["mae_f_xy"])
    # no f_xy target at all -> no keys, so a legacy checkpoint's metric dict is
    # key-for-key what it always was
    assert fxy_metrics(pred, df, np.zeros(t), np.ones(t), TARGETS, 5) == {}


# --------------------------------------------------------------------------- #
# 3. the prior fit
# --------------------------------------------------------------------------- #
def test_prior_recovers_the_planted_affine():
    df = _sparse_frame(n=2000, labelled=600, seed=3)
    prior = fit_fxy_prior(df)
    assert prior.n_fit == 600
    assert prior.a == pytest.approx(1.1221, abs=0.05)
    assert prior.b == pytest.approx(-0.0831, abs=0.08)
    assert prior.resid_sd == pytest.approx(0.029, abs=0.01)
    assert prior.pearson > 0.9
    assert prior.split is None and prior.schema == "fxy_fr_affine_v1"


def test_prior_fits_only_labelled_converged_rows():
    df = _sparse_frame(n=400, labelled=100, seed=4)
    df.loc[df.index[:200], "converged"] = False
    prior = fit_fxy_prior(df)
    labelled_converged = int((np.isfinite(df["f_xy"].to_numpy(float))
                              & df["converged"].to_numpy(bool)).sum())
    assert prior.n_fit == labelled_converged < 100


def test_prior_degenerates_gracefully_without_the_column():
    df = _sparse_frame(n=20, labelled=2).drop(columns=["f_xy"])
    prior = fit_fxy_prior(df)
    assert prior.n_fit == 0 and prior.a == 0.0


def test_prior_z_round_trip_is_exact():
    """A residual of zero must reproduce the PHYSICAL prior exactly, or "the head
    starts at the prior" is not true."""
    prior = FxyFrPrior(a=1.1221, b=-0.0831, n_fit=500)
    tmean, tstd = np.array(_ZMEAN_FXY), np.array(_ZSTD_FXY)
    A, B = fxy_prior_z(prior, tmean, tstd, 7, 0)
    f_r = np.array([1.45, 1.55, 1.70])
    z_fr = (f_r - tmean[0]) / tstd[0]
    z_fxy = A * z_fr + B                       # residual == 0
    np.testing.assert_allclose(z_fxy * tstd[7] + tmean[7], prior.evaluate(f_r),
                               rtol=0, atol=1e-12)


def test_prior_z_is_safe_on_a_degenerate_std():
    A, B = fxy_prior_z(FxyFrPrior(a=1.0, b=0.0), [0.0] * 8, [1.0] * 7 + [0.0], 7, 0)
    assert (A, B) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# 4. the MIN_FXY_LABELS guard
# --------------------------------------------------------------------------- #
def test_resolve_is_a_no_op_without_the_target():
    df = _sparse_frame()
    assert resolve_fxy_prior(df, df, TARGETS) == (-1, -1, None, (0.0, 0.0))


def test_resolve_refuses_too_few_labels():
    df = _sparse_frame(n=1000, labelled=MIN_FXY_LABELS - 1)
    with pytest.raises(ValueError, match=r"promote_fxy: only \d+ labelled"):
        resolve_fxy_prior(df, df, TARGETS_WITH_FXY)


def test_resolve_accepts_at_the_threshold():
    df = _sparse_frame(n=4000, labelled=MIN_FXY_LABELS)
    idx, ref, prior, (A, B) = resolve_fxy_prior(df, df, TARGETS_WITH_FXY)
    assert (idx, ref) == (7, 0)
    assert prior.n_fit == MIN_FXY_LABELS and prior.split == "train"
    assert np.isfinite(A) and np.isfinite(B) and A > 0


def test_resolve_needs_an_f_r_reference():
    df = _sparse_frame()
    with pytest.raises(ValueError, match="needs an 'f_r' target"):
        resolve_fxy_prior(df, df, ("cyclen", "f_xy"))


# --------------------------------------------------------------------------- #
# 5. the net-level composition
# --------------------------------------------------------------------------- #
def _tiny_cfg(**kw) -> PosValNetConfig:
    base = dict(in_channels=4, n_globals=3, width=16, n_blocks=2, groups=4,
                head_hidden=16, n_targets=8)
    base.update(kw)
    return PosValNetConfig(**base)


def _tiny_batch(n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    cells = torch.rand(n, 4, 19, 19, generator=g)
    cells[:, 0] = 1.0                            # channel 0 == fuel_mask
    globals_ = torch.rand(n, 3, generator=g)
    return cells, globals_


def test_composition_off_is_byte_identical():
    torch.manual_seed(0)
    off = PosValNet(_tiny_cfg())
    torch.manual_seed(0)
    on = PosValNet(_tiny_cfg(fxy_target_idx=7, fxy_ref_idx=0,
                             fxy_prior_a=1.0, fxy_prior_b=0.0))
    assert not off.has_fxy_prior and on.has_fxy_prior
    # no new parameters, no new state_dict keys — the residual head IS the mu row
    assert set(off.state_dict()) == set(on.state_dict())
    assert sum(p.numel() for p in off.parameters()) == \
        sum(p.numel() for p in on.parameters())
    cells, g = _tiny_batch()
    with torch.no_grad():
        a, b = off(cells, g), on(cells, g)
    torch.testing.assert_close(a["mu"][:, :7], b["mu"][:, :7])
    torch.testing.assert_close(a["log_sigma"], b["log_sigma"])
    torch.testing.assert_close(a["map"], b["map"])


def test_composed_row_is_prior_plus_residual():
    torch.manual_seed(1)
    A, B = 1.87, -0.21
    raw = PosValNet(_tiny_cfg())
    torch.manual_seed(1)
    comp = PosValNet(_tiny_cfg(fxy_target_idx=7, fxy_ref_idx=0,
                               fxy_prior_a=A, fxy_prior_b=B))
    cells, g = _tiny_batch(seed=2)
    with torch.no_grad():
        r, c = raw(cells, g)["mu"], comp(cells, g)["mu"]
    torch.testing.assert_close(c[:, 7], r[:, 7] + A * r[:, 0] + B)
    torch.testing.assert_close(c[:, :7], r[:, :7])


def test_zero_residual_head_serves_exactly_the_prior():
    A, B = 1.87, -0.21
    net = PosValNet(_tiny_cfg(fxy_target_idx=7, fxy_ref_idx=0,
                              fxy_prior_a=A, fxy_prior_b=B))
    with torch.no_grad():
        net.mu_head.weight[7].zero_()
        net.mu_head.bias[7].zero_()
    cells, g = _tiny_batch(seed=3)
    with torch.no_grad():
        mu = net(cells, g)["mu"]
    torch.testing.assert_close(mu[:, 7], A * mu[:, 0] + B)


def test_fxy_loss_cannot_move_the_f_r_head():
    """The reference row is detached: a target labelled on 2% of rows must not be
    able to perturb the seven dense targets the no-regression gate scores."""
    net = PosValNet(_tiny_cfg(fxy_target_idx=7, fxy_ref_idx=0,
                              fxy_prior_a=1.2, fxy_prior_b=0.0))
    cells, g = _tiny_batch(seed=4)
    out = net(cells, g)
    out["mu"][:, 7].sum().backward()
    grad = net.mu_head.weight.grad
    assert torch.any(grad[7] != 0.0), "the f_xy row must receive gradient"
    assert torch.all(grad[0] == 0.0), "the f_r row must not"


def test_bad_prior_indices_are_rejected():
    with pytest.raises(ValueError, match="outside the 8-target head"):
        PosValNet(_tiny_cfg(fxy_target_idx=9, fxy_ref_idx=0))
    with pytest.raises(ValueError, match="cannot be its own prior"):
        PosValNet(_tiny_cfg(fxy_target_idx=7, fxy_ref_idx=7))


def test_head_learns_the_residual_from_a_sparse_linear_relation():
    """95% of the f_xy labels are NaN, and the 5% that exist encode a residual the
    prior cannot express.  The head must learn it and GENERALIZE to the
    unlabelled rows — otherwise the promotion buys nothing over the prior alone.
    """
    torch.manual_seed(7)
    n, n_lab = 256, 13                       # ~5% labelled
    g = torch.Generator().manual_seed(7)
    a = torch.rand(n, generator=g) * 2 - 1   # drives f_r
    b = torch.rand(n, generator=g) * 2 - 1   # drives the residual only
    cells = torch.zeros(n, 4, 19, 19)
    cells[:, 0] = 1.0
    cells[:, 1] = a.view(-1, 1, 1)
    cells[:, 2] = b.view(-1, 1, 1)
    globals_ = torch.stack([a, b, torch.ones(n)], dim=1)

    A, B = 1.3, -0.2
    target = torch.zeros(n, 8)
    target[:, 0] = a                                     # f_r (dense label)
    target[:, 7] = A * a + B + 0.7 * b                   # f_xy = prior + residual
    mask = torch.zeros(n, 8)
    mask[:, 0] = 1.0
    lab = torch.arange(n_lab)                            # labelled f_xy rows
    mask[lab, 7] = 1.0
    held = torch.arange(n_lab, n)                        # never seen by the head
    target[held, 7] = target[held, 7]                    # (kept for scoring only)
    scored = target.clone()
    target[held, 7] = float("nan")                       # what the trainer sees

    net = PosValNet(_tiny_cfg(width=32, head_hidden=32, fxy_target_idx=7,
                              fxy_ref_idx=0, fxy_prior_a=A, fxy_prior_b=B))
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)

    def held_out_mae() -> float:
        with torch.no_grad():
            mu = net(cells, globals_)["mu"]
        return float((mu[held, 7] - scored[held, 7]).abs().mean())

    before = held_out_mae()
    for _ in range(400):
        opt.zero_grad()
        out = net(cells, globals_)
        loss = regression_loss(out["mu"], out["log_sigma"], target, mask,
                               use_nll=False, beta=0.5, delta=1.0)
        loss.backward()
        opt.step()
    after = held_out_mae()

    assert after < 0.5 * before, f"held-out f_xy MAE {before:.3f} -> {after:.3f}"
    # and it beats the prior-alone baseline (residual == 0 leaves 0.7*b, whose
    # mean absolute value is 0.35)
    assert after < 0.30, f"head did not beat the prior-only baseline ({after:.3f})"


# --------------------------------------------------------------------------- #
# 6. serving — predict_fxy
# --------------------------------------------------------------------------- #
def _make_ensemble(tmp: Path, *, with_fxy: bool, n: int = 2) -> Path:
    ens = tmp / ("ens_fxy" if with_fxy else "ens_plain")
    names = list(TARGETS_WITH_FXY) if with_fxy else list(TARGETS)
    cfg = (PosValNetConfig(n_targets=8, fxy_target_idx=7, fxy_ref_idx=0,
                           fxy_prior_a=0.67, fxy_prior_b=0.05)
           if with_fxy else PosValNetConfig())
    zmean = _ZMEAN_FXY if with_fxy else _ZMEAN_FXY[:7]
    zstd = _ZSTD_FXY if with_fxy else _ZSTD_FXY[:7]
    for i in range(n):
        seed = 500 + i
        meta = {
            "net_config": dict(cfg.__dict__),
            "cond_schema": "v3",
            "target_names": names,
            "target_zscore": {"mean": zmean, "std": zstd},
            "seed": seed,
            "versions": {"torch": torch.__version__},
        }
        if with_fxy:
            meta["fxy_head"] = {
                "enabled": True, "target": "f_xy", "target_idx": 7,
                "prior_ref": "f_r",
                "prior": FxyFrPrior(a=1.1221, b=-0.0831, n_fit=840).to_dict(),
                "prior_z": {"a": 0.67, "b": 0.05},
                "n_labelled_train": 840, "min_labels": MIN_FXY_LABELS,
                "serve_affecting": True,
            }
        save_member(ens / f"member_{seed}", PosValNet(cfg), meta)
    return ens


@pytest.fixture(scope="module")
def one_case(store_bits):
    reader, _fuel, _ids = store_bits
    from lpopt.vendor.masterrl.domain import CaseKey
    row = reader.records.iloc[0]
    return (unpack_pattern(str(row["pattern"])),
            CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"])),
            float(row["feed"]))


def test_predict_fxy_shape_sigma_and_source(tmp_path, one_case):
    pat, case, cell = one_case
    backend = PosValCnnBackend.from_dir(
        _make_ensemble(tmp_path, with_fxy=True), store_dir=STORE,
        library_id="ga80")
    out = backend.predict_fxy([pat] * 3, case, cell)
    assert out is not None
    mean, sigma, source = out
    assert source == "head"
    assert mean.shape == sigma.shape == (3,)
    assert np.isfinite(mean).all() and np.isfinite(sigma).all()
    assert (sigma > 0).all()
    # the 7-column contract is untouched by the extra head
    pred = backend.predict([pat] * 3, case, cell)
    assert pred.mean.shape == (3, 7)
    assert np.isnan(pred.mean[:, 5]).all()


def test_predict_fxy_is_none_without_a_head(tmp_path, one_case):
    pat, case, cell = one_case
    backend = PosValCnnBackend.from_dir(
        _make_ensemble(tmp_path, with_fxy=False), store_dir=STORE,
        library_id="ga80")
    assert backend.predict_fxy([pat], case, cell) is None
    assert "f_xy" not in backend.target_names


def test_predict_fxy_on_an_empty_batch(tmp_path, one_case):
    _pat, case, cell = one_case
    backend = PosValCnnBackend.from_dir(
        _make_ensemble(tmp_path, with_fxy=True), store_dir=STORE,
        library_id="ga80")
    mean, sigma, source = backend.predict_fxy([], case, cell)
    assert mean.shape == sigma.shape == (0,) and source == "head"


def test_served_fxy_is_the_prior_plus_the_residual(tmp_path, one_case):
    """Serving must reproduce the composition, or ``predict_fxy`` would report a
    bare residual as if it were an absolute F_xy."""
    pat, case, cell = one_case
    backend = PosValCnnBackend.from_dir(
        _make_ensemble(tmp_path, with_fxy=True), store_dir=STORE,
        library_id="ga80")
    mean, _sigma, _src = backend.predict_fxy([pat], case, cell)
    # F_xy must land near the plausible physical range the z constants encode,
    # NOT near zero (which is what a bare z-residual would de-normalize to only
    # if the composition were dropped)
    assert 1.0 < float(mean[0]) < 2.5


# --------------------------------------------------------------------------- #
# 7. calibration / conformal BY NAME
# --------------------------------------------------------------------------- #
def test_unfitted_fxy_sigma_passes_through_raw():
    """The σ-bug lesson: a champion's calibration.json predates f_xy and lists 7
    targets.  The 8th column must keep its RAW σ — never another target's curve,
    and never uninitialized memory."""
    calib = {
        "targets": list(TARGETS),
        "isotonic": {n: {"x": [0.0, 10.0], "y": [0.0, 20.0]} for n in TARGETS},
    }
    raw = np.full((4, 8), 1.5)
    out = apply_calibration(raw, calib, list(TARGETS_WITH_FXY))
    np.testing.assert_allclose(out[:, 7], raw[:, 7])       # untouched
    np.testing.assert_allclose(out[:, 0], 3.0)             # fitted -> doubled


def test_calibration_maps_by_name_not_position():
    calib = {
        "targets": ["f_xy"],
        "isotonic": {"f_xy": {"x": [0.0, 10.0], "y": [0.0, 30.0]}},
    }
    raw = np.full((2, 8), 1.0)
    out = apply_calibration(raw, calib, list(TARGETS_WITH_FXY))
    np.testing.assert_allclose(out[:, 7], 3.0)             # landed on column 7
    np.testing.assert_allclose(out[:, :7], raw[:, :7])     # not on column 0


def test_conformal_deliberately_excludes_fxy():
    """Phase-1 decision (design §3.4.3): CONFORMAL_TARGETS is keyed on 7-column
    surrogate indices, f_xy has none, and the per-cell label count is far below
    DEFAULT_MIN_CELL.  No f_xy interval is better than a vacuous one."""
    assert "f_xy" not in {name for name, _col in CONFORMAL_TARGETS}
    assert all(0 <= col < 7 for _name, col in CONFORMAL_TARGETS)


# --------------------------------------------------------------------------- #
# 8. freeze-finetune: grafting a new head row onto a champion
# --------------------------------------------------------------------------- #
def test_graft_pads_an_appended_target_row():
    champ = {"mu_head.weight": torch.arange(16, dtype=torch.float32).view(8, 2),
             "mu_head.bias": torch.arange(8, dtype=torch.float32),
             "log_sigma_head.weight": torch.zeros(8, 2),
             "log_sigma_head.bias": torch.zeros(8),
             "stem.0.weight": torch.ones(3)}
    model = {"mu_head.weight": torch.full((9, 2), -1.0),
             "mu_head.bias": torch.full((9,), -1.0),
             "log_sigma_head.weight": torch.full((9, 2), -1.0),
             "log_sigma_head.bias": torch.full((9,), -1.0),
             "stem.0.weight": torch.zeros(3)}
    out = _graft_appended_target_rows(champ, model)
    assert out["mu_head.weight"].shape == (9, 2)
    torch.testing.assert_close(out["mu_head.weight"][:8], champ["mu_head.weight"])
    assert float(out["mu_head.weight"][8, 0]) == -1.0     # fresh row kept
    torch.testing.assert_close(out["mu_head.bias"][:8], champ["mu_head.bias"])
    # untouched keys are passed through by identity
    assert out["stem.0.weight"] is champ["stem.0.weight"]


def test_graft_refuses_anything_that_is_not_a_row_append():
    """A width/feature mismatch is a genuine recipe error and must stay loud —
    the graft leaves it for the strict load to reject."""
    champ = {"mu_head.weight": torch.zeros(8, 4)}          # different fan-in
    model = {"mu_head.weight": torch.zeros(9, 2)}
    assert _graft_appended_target_rows(champ, model)["mu_head.weight"].shape == (8, 4)
    shrink = {"mu_head.weight": torch.zeros(9, 2)}         # champion is WIDER
    assert _graft_appended_target_rows(shrink, {"mu_head.weight": torch.zeros(8, 2)}
                                       )["mu_head.weight"].shape == (9, 2)


def test_grafted_champion_loads_strictly_into_the_wider_head():
    torch.manual_seed(11)
    champ_net = PosValNet(_tiny_cfg(n_targets=8))
    wide = PosValNet(_tiny_cfg(n_targets=9, fxy_target_idx=8, fxy_ref_idx=0,
                               fxy_prior_a=1.1, fxy_prior_b=0.0))
    state = _graft_appended_target_rows(champ_net.state_dict(), wide.state_dict())
    wide.load_state_dict(state, strict=True)               # must not raise
    torch.testing.assert_close(wide.mu_head.weight[:8],
                               champ_net.mu_head.weight)


# --------------------------------------------------------------------------- #
# 9. al_retrain plumbing (mock champion)
# --------------------------------------------------------------------------- #
def _mock_champion(tmp: Path, *, target_names, promote_fxy=False) -> Path:
    d = tmp / "champ"
    (d / "member_0").mkdir(parents=True)
    (d / "ensemble.json").write_text(json.dumps(
        {"members": ["member_0"], "n_members": 5, "split": "S1i",
         "base_seed": 0}), encoding="utf-8")
    (d / "member_0" / "meta.json").write_text(json.dumps({
        "cond_schema": "v8",
        "target_names": list(target_names),
        "net_config": {"width": 224, "n_blocks": 8, "head_hidden": 384},
        "train_config": {"epochs": 150, "cyclen_physics_prior": True,
                         "quantile_heads": True, "quantile_weight": 0.2,
                         "promote_max_asm_bu": True, "promote_fxy": promote_fxy,
                         "distill_weight": 0.0},
    }), encoding="utf-8")
    return d


def test_recipe_reads_promote_fxy_from_the_checkpoint_targets(tmp_path):
    """A champion carrying the head can never be 'reproduced' without it — the
    target names are what the WEIGHTS were built with."""
    plain = champion_recipe(_mock_champion(tmp_path / "a",
                                           target_names=TARGETS_WITH_ASM_BU))
    assert plain["promote_fxy"] is False
    withfxy = champion_recipe(_mock_champion(
        tmp_path / "b", target_names=TARGETS_WITH_ASM_BU + ("f_xy",)))
    assert withfxy["promote_fxy"] is True


def test_add_fxy_head_composes_the_freeze_finetune_recipe(tmp_path):
    champ = _mock_champion(tmp_path, target_names=TARGETS_WITH_ASM_BU)
    plan = plan_al_retrain(champ, add_fxy_head=True)
    args = plan["train_args"]
    assert "--promote-fxy" in args
    assert "--freeze-trunk-cyclen" in args
    assert args[args.index("--init-from") + 1] == str(champ)
    # champion switches survive
    assert "--promote-max-asm-bu" in args and "--cyclen-physics-prior" in args
    assert args[args.index("--cond-schema") + 1] == "v8"
    # and the whole thing rides through the remote wrapper after the -- token
    remote = plan["steps"][-1]
    assert "python -m lpopt.remote --input lpopt.inp train --" in remote
    assert "--promote-fxy" in remote and "--freeze-trunk-cyclen" in remote


def test_plain_retrain_is_unchanged_by_the_new_switch(tmp_path):
    champ = _mock_champion(tmp_path, target_names=TARGETS_WITH_ASM_BU)
    args = plan_al_retrain(champ)["train_args"]
    assert "--promote-fxy" not in args
    assert "--init-from" not in args and "--freeze-trunk-cyclen" not in args


def test_freeze_flag_is_never_emitted_without_init_from():
    """``--freeze-trunk-cyclen`` without ``--init-from`` is a hard error in
    train.py (it would freeze randomly-initialized cyclen rows)."""
    recipe = {"n_members": 5, "split": "S1", "cond_schema": "v8", "width": 224,
              "n_blocks": 8, "head_hidden": 384, "epochs": 150,
              "promote_fxy": True}
    args = recipe_to_train_args(recipe, distill_cache=None,
                                freeze_trunk_cyclen=True, init_from=None)
    assert "--freeze-trunk-cyclen" not in args


# --------------------------------------------------------------------------- #
# 10. arm-2 fixes (adjudication data/reports/fxy_head_results_20260829.md)
#
# Arm 1 FAILED G2/G3/G4.  The measured causes, and the test that pins each fix:
#
#   * the residual never trained -- best epochs 4-37 were picked by a composite
#     that ``fxy_metrics`` is deliberately kept out of  -> 10.3
#   * ``log_sigma`` never trained either: the mu-only warmup ran to epoch 80, so
#     the served sigma (0.56) was its INITIAL value, 1.7x the label spread, and
#     ``calibration.json`` carried no ``f_xy`` curve to rein it in  -> 10.4
#   * the -0.081 f_xy bias was 1.2161x s1i's UNCALIBRATED F_r bias (-0.0655),
#     because ``_compose_fxy`` reads the raw ``mu[f_r]`` row while
#     ``f_r_calibration.json`` is applied later, in ``predict``  -> 10.1
#   * and there was no way to ask whether the composition earns its keep  -> 10.2
# --------------------------------------------------------------------------- #

# 10.1 the prior refit on the model's OWN predicted F_r
def _fxy_refit_tensors(n: int = 400, *, offset: float, a: float = 1.2161,
                       b: float = -0.2488, seed: int = 0):
    """Tensors whose f_xy label is ``a*f_r_true + b`` for a net that MISREADS
    ``f_r`` by exactly ``offset``.

    The net comes first: ``f_r_true`` is defined as ``mu[f_r] - offset``, so the
    F_r head's systematic error is a number the test controls exactly, which is
    what turns "inherited vs absorbed bias" into a measurement.  Returns
    ``(net, tensors, f_r_pred)``.
    """
    torch.manual_seed(seed)
    net = PosValNet(_tiny_cfg(fxy_target_idx=7, fxy_ref_idx=0,
                              fxy_prior_a=0.0, fxy_prior_b=0.0))
    cells, globals_ = _tiny_batch(n=n, seed=seed)
    with torch.no_grad():
        f_r_pred = net(cells, globals_)["mu"][:, 0]
    f_r_true = f_r_pred - offset
    targets = torch.zeros(n, 8)
    targets[:, 0] = f_r_true
    targets[:, 7] = a * f_r_true + b
    mask = torch.zeros(n, 8)
    mask[:, 0] = 1.0
    mask[:, 7] = 1.0
    tensors = {"cells": cells, "globals": globals_, "targets": targets,
               "target_mask": mask}
    return net, tensors, f_r_pred.numpy()


def test_refit_on_predicted_absorbs_the_f_r_head_bias():
    """The arm-1 defect, measured: a prior fitted on the MEASURED F_r hands the
    f_xy row ``a x (F_r head bias)``; refitting on the row the composition
    actually reads removes it, and train and serve still read the SAME row."""
    offset = -0.0655                       # s1i's uncalibrated F_r bias
    net, tensors, fr_pred = _fxy_refit_tensors(offset=offset)
    got = refit_fxy_prior_on_predicted(
        net, tensors, fxy_idx=7, ref_idx=0, tmean=np.zeros(8), tstd=np.ones(8),
        device=torch.device("cpu"))
    assert got is not None
    prior, (A, B) = got
    assert prior.split == "train:predicted_f_r"
    assert prior.n_fit == 400

    truth = tensors["targets"][:, 7].numpy()
    measured_bias = float(np.mean((1.2161 * fr_pred - 0.2488) - truth))
    refit_bias = float(np.mean(prior.evaluate(fr_pred) - truth))
    assert measured_bias == pytest.approx(1.2161 * offset, abs=1e-6)
    assert abs(refit_bias) < 1e-6 < abs(measured_bias)
    # the refit recovers the label's own slope, only re-based on the biased row
    assert prior.a == pytest.approx(1.2161, abs=1e-4)
    assert prior.b == pytest.approx(-0.2488 - 1.2161 * offset, abs=1e-4)
    # and the z-space image composes to exactly the physical refit affine
    np.testing.assert_allclose(A * fr_pred + B, prior.evaluate(fr_pred),
                               rtol=0, atol=1e-6)


def test_refit_is_none_on_a_degenerate_reference():
    """A constant predicted F_r cannot pin a slope -- keep the measured fit
    rather than divide by zero and ship a wild prior."""
    net, tensors, _ = _fxy_refit_tensors(n=50, offset=0.0)
    with torch.no_grad():                       # mu[f_r] becomes a constant
        net.mu_head.weight[0].zero_()
        net.mu_head.bias[0].zero_()
    assert refit_fxy_prior_on_predicted(
        net, tensors, fxy_idx=7, ref_idx=0, tmean=np.zeros(8), tstd=np.ones(8),
        device=torch.device("cpu")) is None


def test_refit_uses_only_the_labelled_rows():
    net, tensors, _ = _fxy_refit_tensors(n=120, offset=-0.05)
    tensors["target_mask"][40:, 7] = 0.0             # 40 labelled rows left
    tensors["targets"][40:, 7] = float("nan")
    prior, _ab = refit_fxy_prior_on_predicted(
        net, tensors, fxy_idx=7, ref_idx=0, tmean=np.zeros(8), tstd=np.ones(8),
        device=torch.device("cpu"))
    assert prior.n_fit == 40


def test_refit_leaves_the_model_in_its_original_train_eval_mode():
    net, tensors, _ = _fxy_refit_tensors(n=60, offset=-0.05)
    net.train()
    refit_fxy_prior_on_predicted(net, tensors, fxy_idx=7, ref_idx=0,
                                 tmean=np.zeros(8), tstd=np.ones(8),
                                 device=torch.device("cpu"))
    assert net.training is True


# 10.2 direct mode
def test_direct_mode_keeps_the_guard_but_drops_the_reference_row():
    df = _sparse_frame(n=4000, labelled=MIN_FXY_LABELS)
    idx, ref, prior, ab = resolve_fxy_prior(df, df, TARGETS_WITH_FXY,
                                            prior_residual=False)
    assert (idx, ref, ab) == (7, -1, (0.0, 0.0))
    assert prior is not None and prior.n_fit == MIN_FXY_LABELS   # still reported
    # the label guard is NOT weakened by asking for the direct head
    thin = _sparse_frame(n=1000, labelled=MIN_FXY_LABELS - 1)
    with pytest.raises(ValueError, match=r"promote_fxy: only \d+ labelled"):
        resolve_fxy_prior(thin, thin, TARGETS_WITH_FXY, prior_residual=False)


def test_direct_mode_net_predicts_the_absolute_value():
    """``fxy_ref_idx = -1`` turns the composition off, so the f_xy row IS the
    prediction -- same shape, same state_dict, no prior arithmetic."""
    torch.manual_seed(5)
    plain = PosValNet(_tiny_cfg())
    torch.manual_seed(5)
    direct = PosValNet(_tiny_cfg(fxy_target_idx=7, fxy_ref_idx=-1))
    assert direct.has_fxy_prior is False
    assert set(plain.state_dict()) == set(direct.state_dict())
    cells, g = _tiny_batch(seed=6)
    with torch.no_grad():
        a, b = plain(cells, g)["mu"], direct(cells, g)["mu"]
    assert b.shape == (8, 8)
    torch.testing.assert_close(a, b)          # identical: nothing is composed


def test_train_config_defaults_keep_the_prior_composition():
    cfg = TrainConfig()
    assert cfg.fxy_prior_residual is True
    assert cfg.fxy_prior_on_predicted is False
    assert cfg.fxy_select_weight == 0.0


# 10.3 the best-epoch / early-stop criterion
def test_fxy_metrics_expose_the_selection_term():
    df = _sparse_frame(n=40, labelled=8)
    n, t = 40, 8
    rng = np.random.default_rng(2)
    pred = {
        "mu_z_members": rng.normal(size=(2, n, t)),
        "targets": np.zeros((n, t)),
        "target_mask": np.zeros((n, t)),
        "record_ids": df["record_id"].tolist(),
    }
    labelled = np.isfinite(df["f_xy"].to_numpy(float))
    pred["target_mask"][:, 7] = labelled.astype(float)
    pred["targets"][:, 7] = np.nan_to_num(df["f_xy"].to_numpy(float))
    tstd = np.ones(t)
    tstd[7] = 0.06
    out = fxy_metrics(pred, df, np.zeros(t), tstd, TARGETS_WITH_FXY, 5)
    assert out["z_mae_f_xy"] == pytest.approx(out["mae_f_xy"] / 0.06)
    assert out["fxy_select"] == pytest.approx(
        out["within_cell_spearman_f_xy"] - out["z_mae_f_xy"])
    # no labelled val row -> NaN, and the weighted selection falls back to the
    # plain composite rather than poisoning it
    empty = {**pred, "target_mask": np.zeros((n, t))}
    off = fxy_metrics(empty, df, np.zeros(t), tstd, TARGETS_WITH_FXY, 5)
    assert np.isnan(off["fxy_select"])


@pytest.fixture(scope="module")
def fxy_train_run(store_bits):
    """A 4-epoch, 1-member f_xy run on the real store's LABELLED rows.

    Small on purpose: the point is not the fit but the SELECTION bookkeeping the
    arm-1 failure turned on -- ``select_score`` per epoch, and which epoch the
    checkpoint is taken from.
    """
    from lpopt.model import train as T
    from lpopt.model.splits import SplitManifest

    reader, fuel, _ids = store_bits
    df = reader.records
    lab = df[(df["converged"] == True) & df["f_xy"].notna()]        # noqa: E712
    if len(lab) < MIN_FXY_LABELS + 40:
        pytest.skip("store carries too few f_xy labels")
    ids = lab["record_id"].astype(str).tolist()
    man = SplitManifest(name="FXY", kind="filter", seed=0,
                        train_ids=ids[:MIN_FXY_LABELS + 20], val_ids=ids[-40:])
    enc = T.FeatureEncoder()
    cfg = TrainConfig(epochs=4, warmup_epochs=1, batch_size=64, augment=False,
                      min_case_val=5, map_norm_subset=16, round_trip_rows=2,
                      promote_fxy=True, fxy_select_weight=0.5)
    cfg.auto_fit_cell_calibration = False
    tr = T.build_precomputed(reader, man, fuel, fold="train", augment=False,
                             encoder=enc, seed=0, promote_fxy=True)
    va = T.build_precomputed(reader, man, fuel, fold="val", augment=False,
                             encoder=enc, seed=0, promote_fxy=True)
    eff, lr, lr_final, warm, _m = T._resolve_schedule(cfg, torch.device("cpu"))
    members = T._train_members(
        [7], train_ds=tr, val_ds=va, cfg=cfg, device="cpu",
        globals_names=enc.globals_names, reader=reader, eff_batch=eff, lr=lr,
        lr_final=lr_final, warm=warm, resident=False, compile_flag=False,
        n_channels=len(enc.channels), channel_names=tuple(enc.channels),
        verbose=False, manifest=man,
        target_names=targets_for(promote_fxy=True))
    return members[0]


def test_every_epoch_logs_the_fxy_metrics(fxy_train_run):
    """arm 1 train.log carried no f_xy number at all while the run was alive."""
    hist = fxy_train_run.history
    assert len(hist) == 4
    for h in hist:
        for key in ("mae_f_xy", "within_cell_spearman_f_xy", "z_mae_f_xy",
                    "fxy_select", "n_fxy_val", "select_score"):
            assert key in h, key
        assert h["n_fxy_val"] > 0


def test_best_epoch_maximizes_the_fxy_weighted_score(fxy_train_run):
    hist = fxy_train_run.history
    w = 0.5
    for h in hist:
        expect = h["composite"] + (w * h["fxy_select"]
                                   if np.isfinite(h["fxy_select"]) else 0.0)
        assert h["select_score"] == pytest.approx(expect)
    best = max(hist, key=lambda h: h["select_score"])
    assert fxy_train_run.best["epoch"] == best["epoch"]
    assert fxy_train_run.best["select_score"] == pytest.approx(
        best["select_score"])
    # the f_xy term is what makes the two criteria differ; without that this
    # test could not tell the fixed selection from the arm-1 one
    assert any(abs(h["select_score"] - h["composite"]) > 1e-9 for h in hist)


# 10.4 sigma: fitted BY NAME so G4 can pass
def test_fit_calibration_fits_the_promoted_targets_by_name(tmp_path, store_bits):
    """arm 1 calibration.json listed 7 targets, so ``predict_fxy``'s
    "calibrated" sigma was the raw head sigma (coverage 0.99, sigma 1.7x the
    label spread).  The inventory must come from the CHECKPOINT."""
    from lpopt.model.calibrate import fit_calibration
    from lpopt.model.splits import SplitManifest

    reader, _fuel, _ids = store_bits
    ids = reader.records["record_id"].astype(str).tolist()[:24]
    splits = tmp_path / "splits"
    SplitManifest(name="TINY", kind="filter", seed=0,
                  train_ids=ids[:12], val_ids=ids[12:]).to_json(
                      splits / "TINY.json")
    ens = _make_ensemble(tmp_path, with_fxy=True)
    fit_calibration(sorted(ens.glob("member_*")), split="TINY", out_dir=ens,
                    store_dir=STORE, splits_dir=splits)
    calib = json.loads((ens / "calibration.json").read_text(encoding="utf-8"))
    assert calib["targets"] == list(TARGETS_WITH_FXY)
    assert "f_xy" in calib["isotonic"]


def test_fitted_fxy_curve_reaches_the_served_sigma(tmp_path, one_case):
    """G4 is only reachable if a fitted f_xy curve actually moves predict_fxy's
    sigma -- the calibration is applied by NAME inside ``_ensemble_raw``."""
    pat, case, cell = one_case
    ens = _make_ensemble(tmp_path / "sigma", with_fxy=True)
    raw = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    mu0, sig0, _ = raw.predict_fxy([pat] * 3, case, cell)
    (ens / "calibration.json").write_text(json.dumps({
        "targets": list(TARGETS_WITH_FXY),
        "isotonic": {n: ({"x": [0.0, 10.0], "y": [0.0, 5.0]} if n == "f_xy"
                         else {"x": [0.0, 10.0], "y": [0.0, 10.0]})
                     for n in TARGETS_WITH_FXY},
        "platt": {"coef": 1.0, "intercept": 0.0, "degenerate": False},
    }), encoding="utf-8")
    cal = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    mu1, sig1, _ = cal.predict_fxy([pat] * 3, case, cell)
    np.testing.assert_allclose(mu1, mu0)                  # mu untouched
    np.testing.assert_allclose(sig1, sig0 * 0.5, rtol=1e-6)


# --------------------------------------------------------------------------- #
# 12. G4 SIGMA BAR — the head's MEAN promoted, its WIDTH did not (`s1j`)
#
# `s1j` (11th champion, promoted 2026-08-30) passed G1/G2'/G3' and FAILED G4:
# 68% coverage 0.831021 against the registered [0.55, 0.80] band — over-wide.
# The registered disposition (C.4 / results arm 3 §5, §10.2) keeps the promotion
# but forbids SERVING that sigma, because ``min_fxy`` ranks on
# ``cyclen_LCB - lam*F_xy_UCB`` and an over-wide sigma inflates every UCB.
# The bar is stamped on the CHECKPOINT (``ensemble.json`` ->
# ``fxy_head.serve_sigma = "barred"``), not on a deck knob: it is a property of
# the measured artifact.
# --------------------------------------------------------------------------- #
def _barred_ensemble(tmp: Path, *, barred: bool) -> Path:
    ens = _make_ensemble(tmp, with_fxy=True)
    payload = {"members": [p.name for p in sorted(ens.glob("member_*"))],
               "n_members": 2, "split": "TEST"}
    if barred:
        payload["fxy_head"] = {"serve_sigma": "barred", "bar": "G4",
                               "measured_coverage_68": 0.831021}
    (ens / "ensemble.json").write_text(json.dumps(payload), encoding="utf-8")
    return ens


def test_ensemble_meta_declares_the_sigma_bar(tmp_path):
    """The flag travels WITH the checkpoint, and its absence changes nothing."""
    free = PosValCnnBackend.from_dir(
        _barred_ensemble(tmp_path / "free", barred=False), store_dir=STORE,
        library_id="ga80")
    assert free.fxy_sigma_barred is False
    barred = PosValCnnBackend.from_dir(
        _barred_ensemble(tmp_path / "barred", barred=True), store_dir=STORE,
        library_id="ga80")
    assert barred.fxy_sigma_barred is True
    # the bar touches the WIDTH ONLY at the serving boundary: the backend's own
    # predict_fxy still reports its calibrated sigma, so a gate/diagnostic can
    # still MEASURE the coverage that produced the bar.
    assert barred.predict_fxy([], "K1_K2-121", 5.0)[2] == "head"


def test_a_member_meta_can_also_assert_the_bar(tmp_path):
    """Any member asserting the bar bars the ensemble — a width objection is
    never outvoted, and a future trainer may stamp it per member."""
    ens = _make_ensemble(tmp_path / "member_bar", with_fxy=True)
    members = sorted(ens.glob("member_*"))
    meta_path = members[-1] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fxy_head"]["serve_sigma"] = "barred"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert backend.fxy_sigma_barred is True


def test_barred_head_sigma_serves_the_proxy_sigma():
    """The serving contract: head MEAN, head SOURCE, PROXY WIDTH."""
    from lpopt.search import acquisition as acq
    from lpopt.vendor.masterrl.surrogate import SurrogatePrediction

    class _Ctx:
        case_key = "K1_K2-121"
        e_core = 5.0

    class _Head:
        fxy_sigma_barred = False

        def predict_fxy(self, patterns, case, cell=0.0):
            return (np.full(len(patterns), 1.42), np.full(len(patterns), 0.99),
                    "head")

    class _BarredHead(_Head):
        fxy_sigma_barred = True

    mean = np.tile([1.50, 1400.0, 2.30, 625.0, 0.20, np.nan, 70.0], (3, 1))
    std = np.tile([0.01, 10.0, 0.02, 2.0, 0.01, np.nan, 0.5], (3, 1))
    pred = SurrogatePrediction(mean, std.copy(), std)
    patterns = [object(), object(), object()]

    mu, sd, src = acq.predict_fxy(_Head(), patterns, _Ctx(), pred)
    assert src == acq.FXY_SOURCE_HEAD and sd.tolist() == [0.99] * 3

    mu_b, sd_b, src_b = acq.predict_fxy(_BarredHead(), patterns, _Ctx(), pred)
    # the MEAN is the head's, unchanged, and it is still labelled "head": the
    # number being ranked really did come from the head.
    assert src_b == acq.FXY_SOURCE_HEAD
    np.testing.assert_allclose(mu_b, mu)
    # the WIDTH is the interim proxy convention (resid_sd 0.0476 x K 3.0
    # propagated through the F_r sigma) — never the barred head sigma.
    _, proxy_sd = acq.fxy_proxy(pred)
    np.testing.assert_allclose(sd_b, proxy_sd)
    assert not np.allclose(sd_b, 0.99)
    assert acq.fxy_sigma_barred(_BarredHead()) and not acq.fxy_sigma_barred(_Head())


def test_barred_head_serves_no_fxy_conformal_bound():
    """A conformal half-width is fitted ON the barred sigma, so serving that
    bound would re-admit exactly the width the bar refuses.  ``None`` is the
    documented 'keep the proxy screen' answer, so no caller needs a new branch."""
    from lpopt.search import acquisition as acq

    class _Ctx:
        case_key = "K1_K2-121"
        e_core = 5.0

    class _Backend:
        fxy_sigma_barred = True
        conformal = {"per_target": {"f_xy": {"global": {0.32: 0.05}}}}

        def predict_fxy(self, patterns, case, cell=0.0):
            return (np.full(len(patterns), 1.42), np.full(len(patterns), 0.99),
                    "head")

        def conformal_cell_keys(self, patterns, case, cell=0.0):
            return ["121|5.0"] * len(patterns)

    assert acq.fxy_conformal_upper(_Backend(), _Ctx(), [object()],
                                   alpha=0.32) is None


# --------------------------------------------------------------------------- #
# 12b. THE BAR MUST SURVIVE A SAVE (defect D3)
#
# ``minfxy_T6T4_f121_r1`` §9 D3: the campaign's per-wave fine-tune wrote
# ``runs/<run>/models/champion_wave_NN`` with members + ``backend.json`` but NO
# ``ensemble.json``.  ``fxy_sigma_barred`` is resolved from that file, so a
# ``--resume`` reloading a wave checkpoint got ``False`` and served the head's
# own (G4-failed, over-wide) sigma for the last 12 calls of the campaign — with
# nothing in any artifact saying so.
# --------------------------------------------------------------------------- #
def test_saved_checkpoint_round_trips_the_sigma_bar(tmp_path):
    """save() -> from_dir() must give back a BARRED backend, not a freed one."""
    src = _barred_ensemble(tmp_path / "src", barred=True)
    backend = PosValCnnBackend.from_dir(src, store_dir=STORE, library_id="ga80")
    assert backend.fxy_sigma_barred is True

    out = backend.save(tmp_path / "champion_wave_07")
    # the block is on disk, copied — not re-derived, not invented.
    written = json.loads((out / "ensemble.json").read_text(encoding="utf-8"))
    source = json.loads((src / "ensemble.json").read_text(encoding="utf-8"))
    assert written["fxy_head"] == source["fxy_head"]
    assert written["split"] == source["split"]
    # ... and the member list names what this save actually wrote.
    assert written["members"] == [d.name for d in sorted(out.glob("member_*"))]
    assert written["n_members"] == len(written["members"])

    reloaded = PosValCnnBackend.from_dir(out, store_dir=STORE, library_id="ga80")
    assert reloaded.fxy_sigma_barred is True
    # and the serving contract still holds on the reloaded checkpoint.
    assert reloaded.fxy_serve_sigma == "barred"


def test_saving_an_unbarred_checkpoint_is_unchanged(tmp_path):
    """Backward compat: a checkpoint with nothing to preserve writes nothing.

    Every pre-`s1j` champion has no ``fxy_head`` stamp, and several readers key
    off the mere PRESENCE of ``ensemble.json``, so save() must not start
    manufacturing one out of thin air."""
    plain = _make_ensemble(tmp_path / "plain", with_fxy=False)
    backend = PosValCnnBackend.from_dir(plain, store_dir=STORE, library_id="ga80")
    assert backend.fxy_sigma_barred is False
    out = backend.save(tmp_path / "champion_wave_00")
    assert not (out / "ensemble.json").exists()
    assert PosValCnnBackend.from_dir(
        out, store_dir=STORE, library_id="ga80").fxy_sigma_barred is False


def test_a_member_asserted_bar_also_survives_the_save(tmp_path):
    """The member-meta fallback is restated at ensemble level on the copy, so the
    derived checkpoint declares the bar the same way the source did in effect."""
    ens = _make_ensemble(tmp_path / "member_bar_save", with_fxy=True)
    meta_path = sorted(ens.glob("member_*"))[-1] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fxy_head"]["serve_sigma"] = "barred"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    out = backend.save(tmp_path / "champion_wave_01")
    assert json.loads((out / "ensemble.json").read_text(
        encoding="utf-8"))["fxy_head"]["serve_sigma"] == "barred"
    assert PosValCnnBackend.from_dir(
        out, store_dir=STORE, library_id="ga80").fxy_sigma_barred is True


def test_promoted_s1j_carries_the_bar():
    """Promotion regression (2026-08-30): the shipped champion must actually
    carry the stamp the verdict required.  Skipped where the dir is absent."""
    ens = Path(__file__).resolve().parents[1] / "data/models/s1j/ensemble.json"
    if not ens.is_file():
        pytest.skip("data/models/s1j not present in this checkout")
    meta = json.loads(ens.read_text(encoding="utf-8"))
    assert (meta.get("fxy_head") or {}).get("serve_sigma") == "barred"
