"""model_api SurrogatePrediction contract + reward consumption (M3b, plan 4.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.data.fuel_types import FuelLibrary          # noqa: E402
from lpopt.data.schema import unpack_pattern            # noqa: E402
from lpopt.data.store import StoreReader                # noqa: E402
from lpopt.model.featurize import RecordInputs           # noqa: E402
from lpopt.model.dataset_torch import TARGETS           # noqa: E402
from lpopt.model.model_api import (                     # noqa: E402
    EncoderChannelMismatch, ExtraPrediction, PosValCnnBackend,
    PositionValueModel, _to_surrogate,
)
from lpopt.model.featurize import CHANNELS_V4, FeatureEncoder  # noqa: E402
from lpopt.model.net import PosValNet, PosValNetConfig  # noqa: E402
from lpopt.model.train import save_member               # noqa: E402
from lpopt.vendor.masterrl.domain import CaseKey        # noqa: E402
from lpopt.vendor.masterrl.reward import (              # noqa: E402
    CaseRewardScale, ConstraintConfig, RewardModel,
)
from lpopt.vendor.masterrl.surrogate import TARGET_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


# cond_v3 7-target z-score (dataset order: f_r, f_q, cbc_max, cyclen, ao_abs,
# discharge_burnup, max_pin_burnup).
_ZMEAN_V3 = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0]
_ZSTD_V3 = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0]


def test_target_name_reorder_contract() -> None:
    # dataset order (f_r, f_q, cbc_max, cyclen, ao_abs) -> surrogate columns.
    assert TARGET_NAMES == ("F_r", "CBC_max", "F_q", "cyclen", "AO_abs",
                            "max_assembly_burnup", "max_pin_burnup")
    # bare 5-target call keeps the historical contract (burnups NaN).
    m = _to_surrogate(np.array([[10.0, 11.0, 12.0, 13.0, 14.0]]))
    # F_r<-0, CBC_max<-cbc_max(2), F_q<-f_q(1), cyclen<-3, AO_abs<-4
    assert m[0, 0] == 10.0          # F_r
    assert m[0, 1] == 12.0          # CBC_max (dataset cbc_max)
    assert m[0, 2] == 11.0          # F_q (dataset f_q)
    assert m[0, 3] == 13.0          # cyclen
    assert m[0, 4] == 14.0          # AO_abs
    assert np.isnan(m[0, 5]) and np.isnan(m[0, 6])   # burnup axes unknown


def test_7target_reorder_contract() -> None:
    # Phase D: max_pin_burnup exact-match into col 6; discharge_burnup is NOT
    # mapped (it is served via predict_extra), so col 5 stays NaN.
    m = _to_surrogate(
        np.array([[10.0, 11.0, 12.0, 13.0, 14.0, 55.0, 77.0]]), list(TARGETS)
    )
    assert m[0, 0] == 10.0 and m[0, 1] == 12.0 and m[0, 2] == 11.0
    assert m[0, 3] == 13.0 and m[0, 4] == 14.0
    assert np.isnan(m[0, 5])        # max_assembly_burnup: discharge_burnup NOT routed here
    assert m[0, 6] == 77.0          # max_pin_burnup exact-match


def _make_ensemble(tmp: Path, n: int = 2) -> Path:
    ens = tmp / "ens"
    cfg = PosValNetConfig()          # 7-target cond_v3 default
    for i in range(n):
        seed = 100 + i
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


def _make_v4_ensemble(tmp: Path, n: int = 2) -> Path:
    """Tiny cond_v4 synthetic ensemble (43-channel stem, 13-global FiLM)."""
    ens = tmp / "ensv4"
    globals_names = list(FeatureEncoder(cond_schema="v4").globals_names)
    cfg = PosValNetConfig(in_channels=len(CHANNELS_V4),
                          n_globals=len(globals_names))
    for i in range(n):
        seed = 300 + i
        net = PosValNet(cfg)
        meta = {
            "net_config": cfg.__dict__,
            "cond_schema": "v4",
            "channels": list(CHANNELS_V4),
            "globals": globals_names,
            "target_names": list(TARGETS),
            "target_zscore": {"mean": _ZMEAN_V3, "std": _ZSTD_V3},
            "seed": seed,
            "versions": {"torch": torch.__version__},
        }
        save_member(ens / f"member_{seed}", net, meta)
    return ens


@pytest.fixture(scope="module")
def store_fixtures():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    reader = StoreReader(STORE)
    return reader


def test_predict_contract_and_reward(tmp_path, store_fixtures) -> None:
    reader = store_fixtures
    ens = _make_ensemble(tmp_path, n=2)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert isinstance(backend, PositionValueModel)

    row = reader.records.iloc[0]
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    pats = [unpack_pattern(str(row["pattern"]))] * 3

    pred = backend.predict(pats, case, float(row["feed"]))
    # 7-column contract
    assert pred.mean.shape == (3, 7)
    assert pred.epistemic_std.shape == (3, 7)
    assert pred.calibrated_std.shape == (3, 7)
    # first five axes finite; col 6 (max_pin_burnup) now carries a real model
    # (Phase D exact-match); col 5 (max_assembly_burnup) stays NaN because
    # discharge_burnup is served via predict_extra, not through that axis.
    assert np.isfinite(pred.mean[:, :5]).all()
    assert np.isfinite(pred.mean[:, 6]).all()
    for band in (pred.mean, pred.epistemic_std, pred.calibrated_std):
        assert np.isnan(band[:, 5]).all()      # max_assembly_burnup unknown
        assert np.isfinite(band[:, 6]).all()   # max_pin_burnup predicted

    # predict_extra exposes discharge_burnup (finite), NOT in the 7-col surrogate
    extra = backend.predict_extra(pats, case)
    assert isinstance(extra, ExtraPrediction)
    assert extra.names == ("discharge_burnup",)
    assert extra.mean.shape == (3, 1) and np.isfinite(extra.mean).all()
    assert np.isfinite(extra.calibrated_std).all()

    # convergence probability in [0,1]
    cp = backend.predict_convergence(pats, case)
    assert cp.shape == (3,) and (cp >= 0).all() and (cp <= 1).all()
    # position_values is None in v1
    assert backend.position_values(pats[0], case) is None

    # reward.RewardModel consumes the prediction without error (plan 4.5)
    scale = CaseRewardScale(reference_cycle=690.0, cycle_scale=15.0,
                            reference_f_r=1.55, f_r_scale=0.1,
                            reference_cbc=1400.0, cbc_scale=60.0)
    reward = RewardModel({case: scale}, ConstraintConfig())
    batch = reward.score(pred, [case] * 3)
    assert batch.total.shape == (3,)
    assert np.isfinite(batch.total).all()
    pfeas = reward.feasible_probability(pred)
    assert pfeas.shape == (3,) and np.isfinite(pfeas).all()


def test_save_load_roundtrip(tmp_path, store_fixtures) -> None:
    ens = _make_ensemble(tmp_path, n=2)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE)
    out = backend.save(tmp_path / "saved")
    reloaded = PosValCnnBackend.load(out, store_dir=STORE)
    assert len(reloaded.members) == 2


def test_v3_meta_roundtrip_and_schema(tmp_path, store_fixtures) -> None:
    """cond_v3 schema + 7 target_names survive save/load and predictions match."""
    reader = store_fixtures
    ens = _make_ensemble(tmp_path, n=2)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert backend.cond_schema == "v3"
    assert backend.target_names == tuple(TARGETS)
    assert backend.encoder.cond_schema == "v3"

    row = reader.records.iloc[0]
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    pats = [unpack_pattern(str(row["pattern"]))] * 3
    ref = backend.predict(pats, case, float(row["feed"])).mean
    ref_extra = backend.predict_extra(pats, case).mean

    out = backend.save(tmp_path / "v3saved")
    reloaded = PosValCnnBackend.load(out, store_dir=STORE, library_id="ga80")
    assert reloaded.cond_schema == "v3"
    assert reloaded.target_names == tuple(TARGETS)
    got = reloaded.predict(pats, case, float(row["feed"])).mean
    got_extra = reloaded.predict_extra(pats, case).mean
    np.testing.assert_allclose(ref, got, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ref_extra, got_extra, atol=1e-6)


def test_v2_checkpoint_backward_compat(store_fixtures) -> None:
    """A pre-Phase-D cond_v2 (5-target) checkpoint still loads and predicts.

    The backend rebuilds a v2 encoder from the checkpoint meta; the 7-column
    surrogate has BOTH burnup axes NaN (v2 predicts neither) and predict_extra
    returns NaN discharge_burnup — the "no model for this axis" sentinel.
    """
    v2 = REPO_ROOT / "data" / "models" / "20260716_195130"
    if not (v2 / "ensemble.json").is_file():
        pytest.skip("v2 checkpoint not present")
    reader = store_fixtures
    backend = PosValCnnBackend.from_dir(v2, store_dir=STORE, library_id="ga80")
    assert backend.cond_schema == "v2"
    assert backend.encoder.cond_schema == "v2"
    assert len(backend.target_names) == 5 and "max_pin_burnup" not in backend.target_names

    row = reader.records.iloc[0]
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    pats = [unpack_pattern(str(row["pattern"]))] * 2
    pred = backend.predict(pats, case, float(row["feed"]))
    assert pred.mean.shape == (2, 7)
    assert np.isfinite(pred.mean[:, :5]).all()
    for band in (pred.mean, pred.epistemic_std, pred.calibrated_std):
        assert np.isnan(band[:, 5]).all() and np.isnan(band[:, 6]).all()
    extra = backend.predict_extra(pats, case)
    assert extra.names == ("discharge_burnup",) and np.isnan(extra.mean).all()


def test_mixed_schema_ensemble_rejected(tmp_path) -> None:
    """Loading members that disagree on cond_schema must fail (plan sec. 6.2)."""
    from lpopt.model.train import load_member
    ens = _make_ensemble(tmp_path, n=2)          # both v3
    member_dirs = sorted(ens.glob("member_*"))
    # corrupt one member's meta to a v2 schema
    meta_path = member_dirs[0] / "meta.json"
    import json
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cond_schema"] = "v2"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    members, metas = [], []
    for d in member_dirs:
        m, meta = load_member(d, "cpu")
        members.append(m); metas.append(meta)
    fuel = FuelLibrary.from_parquet(STORE / "fuel_types.parquet")
    with pytest.raises(ValueError, match="cond_schema"):
        PosValCnnBackend(members, metas, fuel=fuel)


def test_v4_roundtrip_and_backend_manifest(tmp_path, store_fixtures) -> None:
    """A cond_v4 (43ch/13-global) ensemble serves, round-trips through save/load,
    and the backend.json freezes the channel/global lists."""
    import json
    reader = store_fixtures
    ens = _make_v4_ensemble(tmp_path, n=2)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert backend.cond_schema == "v4"
    assert backend.encoder.cond_schema == "v4"
    assert backend.encoder.n_channels == len(CHANNELS_V4) == 43
    assert len(backend.encoder.globals_names) == 13

    row = reader.records.iloc[0]
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    pats = [unpack_pattern(str(row["pattern"]))] * 3
    ref = backend.predict(pats, case, float(row["feed"])).mean
    assert ref.shape == (3, 7)
    assert np.isfinite(ref[:, :5]).all() and np.isfinite(ref[:, 6]).all()

    out = backend.save(tmp_path / "v4saved")
    mani = json.loads((out / "backend.json").read_text(encoding="utf-8"))
    assert mani["cond_schema"] == "v4"
    assert mani["channels"] == list(CHANNELS_V4)
    assert len(mani["globals"]) == 13
    reloaded = PosValCnnBackend.load(out, store_dir=STORE, library_id="ga80")
    assert reloaded.cond_schema == "v4"
    got = reloaded.predict(pats, case, float(row["feed"])).mean
    np.testing.assert_allclose(ref, got, atol=1e-6, equal_nan=True)


def test_v3_checkpoint_serves_on_26_channels_under_v4_code(tmp_path, store_fixtures) -> None:
    """A v3 (26ch) checkpoint still loads + serves unchanged now that v4 exists —
    the encoder is rebuilt from the checkpoint's own cond_schema (26 channels)."""
    reader = store_fixtures
    ens = _make_ensemble(tmp_path, n=2)              # v3, 26ch
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    assert backend.cond_schema == "v3"
    assert backend.encoder.n_channels == 26
    assert backend.members[0].config.in_channels == 26
    row = reader.records.iloc[0]
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    pats = [unpack_pattern(str(row["pattern"]))] * 2
    assert backend.predict(pats, case, float(row["feed"])).mean.shape == (2, 7)


def test_channel_parity_guard_rejects_in_channels_mismatch(tmp_path) -> None:
    """The F2 fix: a checkpoint whose meta claims v4 but whose net stem is 26ch is
    rejected at load with a distinct EncoderChannelMismatch (a ValueError)."""
    import json
    from lpopt.model.train import load_member
    ens = _make_ensemble(tmp_path, n=1)              # v3 net (26ch)
    md = sorted(ens.glob("member_*"))[0]
    meta = json.loads((md / "meta.json").read_text(encoding="utf-8"))
    meta["cond_schema"] = "v4"                       # v4 encoder -> 43ch != 26
    (md / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    m, meta2 = load_member(md, "cpu")
    fuel = FuelLibrary.from_parquet(STORE / "fuel_types.parquet")
    with pytest.raises(EncoderChannelMismatch):
        PosValCnnBackend([m], [meta2], fuel=fuel)


def test_finetune_rejects_channel_mismatch(tmp_path, store_fixtures) -> None:
    """finetune refuses when the encoder no longer matches the members' stem width
    (a v3->v4 schema change forces a full retrain, not a fine-tune)."""
    reader = store_fixtures
    ens = _make_ensemble(tmp_path, n=1)              # v3 (26ch)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE)
    backend.encoder = FeatureEncoder(cond_schema="v4")   # 43ch, mismatched
    with pytest.raises(EncoderChannelMismatch):
        backend.finetune(reader.records.head(4), reader.records.iloc[4:8],
                         epochs=1, seed=0)


def test_curriculum_retrain_guard_falls_back_to_remote_full(tmp_path) -> None:
    """The curriculum schema-change guard: when a fine-tune hits an
    EncoderChannelMismatch, _default_retrain diverts to a from-scratch
    remote_full retrain instead of failing the cell."""
    from lpopt import curriculum as C
    from lpopt.config import load_config
    deck = REPO_ROOT / "lpopt.inp"
    if not deck.is_file():
        pytest.skip("campaign deck not present")
    cfg = load_config(deck)
    driver = C.CurriculumDriver(cfg, dry_run=True, state_dir=tmp_path,
                                progress=False, log=lambda m: None)

    def _boom(cid, prev):
        raise EncoderChannelMismatch("member in_channels 26 != encoder 43")

    driver._retrain_local_finetune = _boom          # dry_run -> local_finetune
    driver._retrain_remote_full = lambda cid: "REMOTE_FULL_DIR"
    assert driver._default_retrain("5.25-5.5_f117", "champ_v3") == "REMOTE_FULL_DIR"


def test_finetune_runs(tmp_path, store_fixtures) -> None:
    reader = store_fixtures
    ens = _make_ensemble(tmp_path, n=1)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE)
    new = reader.records.head(8)
    replay = reader.records.iloc[8:40]
    info = backend.finetune(new, replay, epochs=2, seed=0)
    assert info["n_new"] == 8 and info["n_replay"] == 32
    assert info["wall_seconds"] >= 0.0


# --------------------------------------------------------------------------- #
# inference-conditioning regression (ga80 / Dataset-B basin)
# --------------------------------------------------------------------------- #
def _ga80_feasible_rows(reader, n: int | None = None):
    """Feasible-basin K1_K2 Dataset-B store rows (converged, binding limits met)."""
    df = reader.records
    cc = ConstraintConfig()
    k = df[(df["dataset"] == "B") & (df["case_pair"] == "K1_K2")
           & (df["converged"] == True)                       # noqa: E712
           & df["f_r"].notna() & df["cyclen"].notna()
           & (df["f_r"] <= cc.f_r_limit) & (df["cbc_max"] <= cc.cbc_limit)]
    if k.empty:
        pytest.skip("no ga80 K1_K2 Dataset-B rows in store")
    return k if n is None else k.head(n)


def _train_side_mean(backend, rows) -> np.ndarray:
    """Predict directly from *store rows* through the training featurizer + ensemble.

    Mirrors ``PosValCnnBackend.predict`` exactly but featurizes each row via
    ``RecordInputs.coerce`` (the training path) instead of ``_record_inputs``
    (the serve path), so any train/serve featurization gap surfaces as a diff.
    """
    cells, gvecs = [], []
    for _, row in rows.iterrows():
        c, g = backend.encoder.encode(RecordInputs.coerce(row), backend.fuel)
        cells.append(c)
        gvecs.append(g)
    cells_t = torch.from_numpy(np.ascontiguousarray(np.stack(cells))).to(backend.device)
    g_t = torch.from_numpy(np.ascontiguousarray(np.stack(gvecs))).to(backend.device)
    mus = []
    with torch.no_grad():
        for m in backend.members:
            mus.append(m(cells_t, g_t)["mu"].float().cpu().numpy())
    members_raw = np.stack(mus) * backend.tstd[None, None, :] + backend.tmean[None, None, :]
    return _to_surrogate(members_raw.mean(axis=0), backend.target_names)


def _campaign_rows(reader, lib: str, n: int):
    """Converged CAMPAIGN-written rows (``dataset == "P"``) of one live library.

    This — not the 574-row historical ``extract_b`` ga80 harvest — is the
    population the campaign serves: 18,973 ga80 + 16,316 paramA of the 74,717-row
    store carry ``dataset="P"``, ``sym_class="rot61"``.
    """
    df = reader.records
    k = df[(df["library_id"] == lib) & (df["dataset"] == "P")
           & (df["converged"] == True)                       # noqa: E712
           & df["pattern"].notna() & df["e_core"].notna()]
    if k.empty:
        pytest.skip(f"no campaign (dataset=P) {lib} rows in store")
    return k.head(n)


def test_serve_conditioning_reconstructs_store_provenance(tmp_path, store_fixtures) -> None:
    """``_record_inputs`` must rebuild the store row's provenance, not the
    RecordInputs defaults (``dataset='A'``/``sym_class='rot61'``) that biased ga80
    inference onto the Dataset-A regime.  Encoded (cells, globals) must be identical
    to featurizing the store row itself.

    The rows are the campaign-written ones (``dataset="P"``), because that is what
    the serve path is a prediction OF — see ``featurize.serve_provenance``.  Before
    the 2026-08-29 fix ``library_provenance`` answered ``("B", "free69")`` here and
    flipped ``g_sym_class`` against all 18,973 ga80 training rows.
    """
    reader = store_fixtures
    ens = _make_ensemble(tmp_path, n=1)
    for lib in ("ga80", "paramA"):
        backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id=lib)
        for _, row in _campaign_rows(reader, lib, 12).iterrows():
            pat = unpack_pattern(str(row["pattern"]))
            case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
            inp = backend._record_inputs(pat, case)
            assert inp.dataset == str(row["dataset"]) == "P"
            assert inp.sym_class == str(row["sym_class"]) == "rot61"
            assert inp.e_core == pytest.approx(float(row["e_core"]), abs=1e-9)
            c_serve, g_serve = backend.encoder.encode(inp, backend.fuel)
            c_train, g_train = backend.encoder.encode(RecordInputs.coerce(row), backend.fuel)
            assert np.array_equal(g_serve, g_train)      # globals byte-identical
            assert np.array_equal(c_serve, c_train)      # cells byte-identical


def test_serve_row_featurization_parity(tmp_path, store_fixtures) -> None:
    """HARD GATE (2026-08-29 train/serve forensic): on 50 REAL store rows spanning
    BOTH live libraries, the serve path (``predict``, CaseKey-based featurization)
    and the train path (``predict_rows_raw``, each row's own provenance) must agree
    on EVERY cond global, every cell channel, and the raw ensemble mean to 1e-6.

    Draws the rows from the S1j honest holdout when that split file is present (the
    exact 793-row slice the arm-2 adjudication used), else from the store head.
    A synthetic ensemble is used so the gate does not depend on a champion being
    on disk and so no physics prior / per-cell calibration can mask a featurization
    gap — ``predict``'s raw means are then literally comparable to
    ``predict_rows_raw``'s.
    """
    import json

    import pandas as pd

    reader = store_fixtures
    df = reader.records
    split_path = REPO_ROOT / "data" / "splits" / "S1j.json"
    if split_path.is_file():
        spec = json.loads(split_path.read_text(encoding="utf-8"))
        val_ids = set(spec.get("val_ids") or spec.get("val") or ())
        if val_ids:
            df = df[df["record_id"].isin(val_ids)]
    frames = []
    for lib in ("ga80", "paramA"):
        k = df[(df["library_id"] == lib) & (df["converged"] == True)   # noqa: E712
               & df["pattern"].notna() & df["e_core"].notna()]
        if len(k) < 25:
            pytest.skip(f"fewer than 25 usable {lib} rows available")
        frames.append(k.head(25))
    rows = pd.concat(frames)
    assert len(rows) == 50

    ens = _make_ensemble(tmp_path, n=2)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    pats = [unpack_pattern(str(p)) for p in rows["pattern"]]
    cases = [CaseKey(pair=str(p), feed=int(f))
             for p, f in zip(rows["case_pair"], rows["feed"])]

    # (a) every cond global + every cell channel, exactly
    g_names = list(backend.encoder.globals_names)
    for pat, case, (_, row) in zip(pats, cases, rows.iterrows()):
        c_s, g_s = backend.encoder.encode(backend._record_inputs(pat, case), backend.fuel)
        c_t, g_t = backend.encoder.encode(RecordInputs.coerce(row), backend.fuel)
        bad = [g_names[j] for j in range(len(g_names))
               if not abs(float(g_s[j]) - float(g_t[j])) <= 1e-6]
        assert not bad, f"cond globals differ for {row['library_id']} row: {bad}"
        np.testing.assert_allclose(c_s, c_t, atol=1e-6)

    # (b) raw ensemble mean, serve path vs row path
    serve_raw = _to_surrogate(backend._ensemble_raw(pats, cases)[0], backend.target_names)
    row_raw = backend.predict_rows_raw(rows)
    np.testing.assert_allclose(serve_raw, row_raw, atol=1e-6, equal_nan=True)


def test_train_serve_predict_parity(tmp_path, store_fixtures) -> None:
    """HARD GATE: serving 20 campaign ga80 patterns through ``predict`` equals
    featurizing their store rows with the training featurizer + ensemble."""
    reader = store_fixtures
    ens = _make_ensemble(tmp_path, n=3)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")
    rows = _campaign_rows(reader, "ga80", 4000)
    first = rows.iloc[0]
    rows = rows[(rows["feed"] == int(first["feed"]))
                & (rows["case_pair"] == str(first["case_pair"]))].head(20)
    case = CaseKey(pair=str(first["case_pair"]), feed=int(first["feed"]))
    pats = [unpack_pattern(str(p)) for p in rows["pattern"]]

    serve_mean = backend.predict(pats, case, float(case.feed)).mean
    train_mean = _train_side_mean(backend, rows)
    np.testing.assert_allclose(serve_mean, train_mean, atol=1e-5, equal_nan=True)


def test_ga80_conditioning_shifts_prediction_toward_b_regime(store_fixtures) -> None:
    """With the real trained ensemble, correct ga80/Dataset-B conditioning pulls the
    feasible-basin prediction toward the measured B basin (lower cyclen, higher
    p_feas) versus the buggy Dataset-A default conditioning.  Documents the
    residual model-quality gap without hard-coding the (still-unmet) ±15/±0.12
    absolute gate."""
    real = REPO_ROOT / "data" / "models" / "20260716_195130"
    if not (real / "ensemble.json").is_file():
        pytest.skip("trained ensemble not present")
    reader = store_fixtures
    backend = PosValCnnBackend.from_dir(real, store_dir=STORE, library_id="ga80")
    rows = _ga80_feasible_rows(reader)
    cc = ConstraintConfig()

    fixed_cy, fixed_pf, buggy_cy, buggy_pf = [], [], [], []
    for fd, grp in rows.groupby("feed"):
        pats = [unpack_pattern(str(p)) for p in grp["pattern"]]
        case = CaseKey(pair="K1_K2", feed=int(fd))
        # correct (fixed) serve path
        pr = backend.predict(pats, case, float(fd))
        fixed_cy.append(pr.mean[:, 3])
        fixed_pf.append(RewardModel({}, cc).feasible_probability(pr))
        # buggy path: force the old RecordInputs defaults (dataset A / rot61)
        cells, gvecs = [], []
        for p in pats:
            inp = RecordInputs(pattern=p.canonical(), feed=int(fd), case_pair="K1_K2",
                               library_id="ga80", e_core=None, e_split=None)  # -> A/rot61
            c, g = backend.encoder.encode(inp, backend.fuel)
            cells.append(c); gvecs.append(g)
        ct = torch.from_numpy(np.ascontiguousarray(np.stack(cells))).to(backend.device)
        gt = torch.from_numpy(np.ascontiguousarray(np.stack(gvecs))).to(backend.device)
        with torch.no_grad():
            mus = [m(ct, gt)["mu"].float().cpu().numpy() for m in backend.members]
        raw = np.stack(mus) * backend.tstd[None, None, :] + backend.tmean[None, None, :]
        buggy_cy.append(_to_surrogate(raw.mean(axis=0))[:, 3])

    fixed_cy = np.concatenate(fixed_cy); fixed_pf = np.concatenate(fixed_pf)
    buggy_cy = np.concatenate(buggy_cy)
    measured = float(rows["cyclen"].mean())
    # the fix moves cyclen toward the measured basin and lifts p_feas off the floor.
    assert fixed_cy.mean() < buggy_cy.mean()
    assert abs(fixed_cy.mean() - measured) < abs(buggy_cy.mean() - measured)
    assert fixed_pf.mean() > 0.005


# --------------------------------------------------------------------------- #
# predict_map_flatness — PHYSICAL-unit node_peak / map_cov (program §13)
# --------------------------------------------------------------------------- #
_MAP_MEAN = [1.02, 1.00, 41.3, 1.04]
_MAP_STD = [0.40, 0.21, 14.9, 0.08]


def _make_map_ensemble(tmp: Path, n: int = 2, *, map_zscore: bool = True) -> Path:
    """Synthetic ensemble that stamps a per-member ``map_zscore`` like training."""
    ens = tmp / ("ensmap" if map_zscore else "ensmapraw")
    cfg = PosValNetConfig()
    for i in range(n):
        seed = 500 + i
        net = PosValNet(cfg)
        meta = {
            "net_config": cfg.__dict__,
            "cond_schema": "v3",
            "target_names": list(TARGETS),
            "target_zscore": {"mean": _ZMEAN_V3, "std": _ZSTD_V3},
            "map_keys": ["boc_power", "eoc_power", "eoc_burnup", "eoc_kinf"],
            "seed": seed,
            "versions": {"torch": torch.__version__},
        }
        if map_zscore:
            meta["map_zscore"] = {"mean": _MAP_MEAN, "std": _MAP_STD}
        save_member(ens / f"member_{seed}", net, meta)
    return ens


def _map_case(reader):
    row = reader.records.iloc[0]
    case = CaseKey(pair=str(row["case_pair"]), feed=int(row["feed"]))
    return case, [unpack_pattern(str(row["pattern"]))] * 3


def test_predict_map_flatness_returns_four_arrays_and_is_finite(tmp_path, store_fixtures):
    backend = PosValCnnBackend.from_dir(_make_map_ensemble(tmp_path),
                                        store_dir=STORE, library_id="ga80")
    case, pats = _map_case(store_fixtures)
    pk_m, pk_s, cv_m, cv_s = backend.predict_map_flatness(pats, case, 5.2)
    for arr in (pk_m, pk_s, cv_m, cv_s):
        assert arr.shape == (3,) and np.all(np.isfinite(arr))
    assert np.all(pk_s >= 0.0) and np.all(cv_s >= 0.0)
    # empty input keeps the shape contract.
    e = backend.predict_map_flatness([], case, 5.2)
    assert all(a.shape == (0,) for a in e)


def test_predict_map_flatness_matches_the_canonical_definition(tmp_path, store_fixtures):
    """The scalars must be data.flatness applied to the DE-NORMALIZED slots."""
    from lpopt.data.flatness import SLOT_COLS, SLOT_ROWS, flatness_pair

    backend = PosValCnnBackend.from_dir(_make_map_ensemble(tmp_path),
                                        store_dir=STORE, library_id="ga80")
    case, pats = _map_case(store_fixtures)
    pk_m, pk_s, cv_m, cv_s = backend.predict_map_flatness(pats, case, 5.2)

    cases, _ = backend._broadcast(pats, case, 5.2)
    cells_t, g_t = backend._encode_batch(pats, cases)
    peaks, covs = [], []
    with torch.no_grad():
        for m, meta in zip(backend.members, backend.metas):
            plane = m(cells_t, g_t)["map"][:, 0].float().cpu().numpy()
            mz = meta["map_zscore"]
            slots = plane[:, SLOT_ROWS, SLOT_COLS] * mz["std"][0] + mz["mean"][0]
            p, c = flatness_pair(slots)
            peaks.append(p)
            covs.append(c)
    assert pk_m == pytest.approx(np.mean(peaks, axis=0), rel=1e-6)
    assert cv_m == pytest.approx(np.mean(covs, axis=0), rel=1e-6)
    # The returned spread is the across-member SD floored at the per-model OOD
    # sigma floor (2026-07-29 debug-panel — the raw spread read 12x too small on a
    # blind OOD case).  Compare against the floored quantity, not the bare SD.
    floor = backend.flatness_sigma_floor
    assert pk_s == pytest.approx(np.maximum(np.std(peaks, axis=0),
                                            floor["node_peak"]), rel=1e-6, abs=1e-9)
    assert cv_s == pytest.approx(np.maximum(np.std(covs, axis=0),
                                            floor["map_cov"]), rel=1e-6, abs=1e-9)


def test_predict_map_flatness_denormalizes_per_member(tmp_path, store_fixtures):
    """A z-space output is NOT the physical peak; the map_zscore must be applied."""
    with_z = PosValCnnBackend.from_dir(_make_map_ensemble(tmp_path),
                                       store_dir=STORE, library_id="ga80")
    raw = PosValCnnBackend.from_dir(_make_map_ensemble(tmp_path, map_zscore=False),
                                    store_dir=STORE, library_id="ga80")
    case, pats = _map_case(store_fixtures)
    pk_z = with_z.predict_map_flatness(pats, case, 5.2)[0]
    pk_raw = raw.predict_map_flatness(pats, case, 5.2)[0]
    # different random weights, so compare the SCALE not the value: a de-normalized
    # peak lives near 1 (core-average-normalized power), the raw head output does not.
    assert not np.allclose(pk_z, pk_raw)
    # a checkpoint with no map_zscore is treated as already-physical (identity),
    # never silently multiplied by someone else's constants.
    assert raw._map_norm(raw.metas[0], 0) == (0.0, 1.0)
    assert with_z._map_norm(with_z.metas[0], 0) == (pytest.approx(_MAP_MEAN[0]),
                                                    pytest.approx(_MAP_STD[0]))


def test_predict_map_peak_is_the_first_half_of_predict_map_flatness(tmp_path, store_fixtures):
    backend = PosValCnnBackend.from_dir(_make_map_ensemble(tmp_path),
                                        store_dir=STORE, library_id="ga80")
    case, pats = _map_case(store_fixtures)
    pk_m, pk_s, _cv_m, _cv_s = backend.predict_map_flatness(pats, case, 5.2)
    m2, s2 = backend.predict_map_peak(pats, case, 5.2)
    assert m2 == pytest.approx(pk_m) and s2 == pytest.approx(pk_s)
