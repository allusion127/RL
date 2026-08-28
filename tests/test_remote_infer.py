"""Remote-GPU screening inference: payload round-trip, routing, fallback (plan 4.7).

NO live ssh in these tests — the transport (`run_ssh` / `_scp_to` / `_scp_from`)
is mocked so the whole client path runs in-process.  The determinism test proves
a serialized encode+infer on CPU is bit-identical to a direct local
`_raw_forward_local`, so the only thing a real GPU changes is float precision.

One OPTIONAL live smoke (`test_live_smoke_remote_screen`) runs only when
``LPOPT_LIVE_SMOKE=1`` is set AND the server answers a 5 s probe; it reports the
real remote screen timing.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.data.schema import unpack_pattern                # noqa: E402
from lpopt.data.store import StoreReader                    # noqa: E402
from lpopt.model.dataset_torch import TARGETS               # noqa: E402
from lpopt.model.model_api import PosValCnnBackend          # noqa: E402
from lpopt.model.net import PosValNet, PosValNetConfig      # noqa: E402
from lpopt.model.train import save_member                   # noqa: E402
from lpopt.model import remote_infer as ri                  # noqa: E402
from lpopt import remote as rem                             # noqa: E402
from lpopt.search.campaign import _normalize_remote_screening  # noqa: E402
from lpopt.vendor.masterrl.domain import CaseKey            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"
DECK = REPO_ROOT / "lpopt.inp"                    # [remote] gpu pinned (not auto)
CHAMPION = REPO_ROOT / "data" / "models" / "20260721_061913"

#: Cross-device float32 tolerances (measured on champion 20260721_061913 vs the
#: local CPU — see data/reports/inference_backend.md).  The regression heads (the
#: means/sigmas that feed ``predict``) agree to ~1.4e-3; the convergence head is
#: compared in PROBABILITY space (``sigmoid``) — the space the campaign actually
#: consumes — where it agrees to ~4e-5.  (Its RAW logit differs by ~1.6e-2 purely
#: because ``logit`` amplifies a 4e-5 sigmoid difference near saturation; that raw
#: number is reported but never gated, since nothing consumes the raw logit.)
_DETERMINISM_TOL_REG = 5e-3          # mu_z, log_sigma (z-space)
_DETERMINISM_TOL_CONVP = 1e-3        # convergence, probability space

_ZMEAN_V3 = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0]
_ZSTD_V3 = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0]


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


@pytest.fixture(scope="module")
def store_present():
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    return StoreReader(STORE)


@pytest.fixture(scope="module")
def backend(store_present, tmp_path_factory):
    ens = _make_ensemble(tmp_path_factory.mktemp("ens"))
    return PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")


@pytest.fixture(scope="module")
def sample(store_present):
    rows = store_present.records.iloc[:6]
    pats = [unpack_pattern(str(p)) for p in rows["pattern"]]
    cases = [CaseKey(pair=str(p), feed=int(f))
             for p, f in zip(rows["case_pair"], rows["feed"])]
    return pats, cases


# --------------------------------------------------------------------------- #
# payload serialization
# --------------------------------------------------------------------------- #
def test_request_roundtrip_preserves_patterns_and_cases(sample):
    pats, cases = sample
    blob = ri.pack_request(pats, cases, "ga80")
    got_pats, got_cases, lib = ri.unpack_request(blob)
    assert lib == "ga80"
    assert [p.canonical() for p in got_pats] == [p.canonical() for p in pats]
    assert [(c.pair, c.feed) for c in got_cases] == [(c.pair, c.feed) for c in cases]


def test_request_length_mismatch_rejected(sample):
    pats, cases = sample
    with pytest.raises(ValueError):
        ri.pack_request(pats, cases[:-1], "ga80")


def test_response_roundtrip():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal((2, 5, 7)).astype(np.float32)
    ls = rng.standard_normal((2, 5, 7)).astype(np.float32)
    cl = rng.standard_normal((2, 5)).astype(np.float32)
    a, b, c = ri.unpack_response(ri.pack_response(mu, ls, cl))
    np.testing.assert_array_equal(a, mu)
    np.testing.assert_array_equal(b, ls)
    np.testing.assert_array_equal(c, cl)


def test_schema_version_mismatch_raises(sample):
    pats, cases = sample
    blob = bytearray(ri.pack_request(pats, cases, "ga80"))
    # Rebuild a payload with a bad schema to trip the guard.
    import io
    buf = io.BytesIO()
    np.savez(buf, schema=np.int64(999),
             pat_blob=np.frombuffer(b"x", dtype=np.uint8),
             pair_blob=np.frombuffer(b"y", dtype=np.uint8),
             feeds=np.zeros(1, dtype=np.int64),
             lib=np.frombuffer(b"ga80", dtype=np.uint8))
    with pytest.raises(ValueError):
        ri.unpack_request(buf.getvalue())


# --------------------------------------------------------------------------- #
# determinism: serialized encode+infer on CPU == direct local forward
# --------------------------------------------------------------------------- #
def test_run_request_matches_local(backend, sample):
    pats, cases = sample
    local = backend._raw_forward_local(pats, cases)
    remote_style = ri.unpack_response(
        ri.run_request(backend, ri.pack_request(pats, cases, "ga80")))
    for a, b in zip(local, remote_style):
        np.testing.assert_array_equal(a.astype(np.float32), b)


def test_run_request_predict_equivalence(backend, sample):
    """A full round-trip feeding predict() yields identical surrogate columns."""
    pats, cases = sample
    ref = backend.predict(pats, cases[0]).mean
    # Simulate the remote by running run_request and stuffing the cache, then
    # predict from the memo — must match the direct local predict.
    calls = {"n": 0}

    def echo(_be, patterns, kases):
        calls["n"] += 1
        return ri.unpack_response(
            ri.run_request(_be, ri.pack_request(patterns, kases, "ga80")))

    backend.enable_remote_screening(echo, min_predictions=1, log=lambda m: None)
    try:
        got = backend.predict(pats, cases[0]).mean
    finally:
        backend.disable_remote_screening()
    np.testing.assert_allclose(ref, got, atol=1e-6, equal_nan=True)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# routing: threshold gating + local fallback
# --------------------------------------------------------------------------- #
def test_threshold_gates_routing(backend, sample):
    """The threshold gates the compute (miss) batch; a fresh cache each time makes
    the full request the miss-batch."""
    pats, cases = sample                      # N = 6
    calls = {"n": 0}

    def counting(_be, patterns, kases):
        calls["n"] += 1
        return _be._raw_forward_local(patterns, kases)

    # below threshold: fresh cache, single call of N=5 -> stays local.
    backend.enable_remote_screening(counting, min_predictions=6, log=lambda m: None)
    backend.predict(pats[:5], cases[0])
    backend.disable_remote_screening()
    assert calls["n"] == 0

    # at/above threshold: fresh cache, single call of N=6 -> routes.
    backend.enable_remote_screening(counting, min_predictions=6, log=lambda m: None)
    backend.predict(pats, cases[0])
    backend.disable_remote_screening()
    assert calls["n"] == 1


def test_local_fallback_on_remote_failure(backend, sample):
    """Unreachable host simulated: the screener raises; predict must NOT hard-fail."""
    pats, cases = sample
    ref = backend.predict(pats, cases[0]).mean
    logs = []

    def unreachable(_be, patterns, kases):
        raise OSError("Network is unreachable")

    backend.enable_remote_screening(unreachable, min_predictions=1, log=logs.append)
    try:
        got = backend.predict(pats, cases[0]).mean
    finally:
        backend.disable_remote_screening()
    np.testing.assert_allclose(ref, got, atol=1e-6, equal_nan=True)
    assert any("local CPU fallback" in m for m in logs)


def test_fallback_on_none_return(backend, sample):
    pats, cases = sample
    backend.enable_remote_screening(lambda *_: None, min_predictions=1,
                                    log=lambda m: None)
    try:
        out = backend.predict(pats, cases[0])
        assert out.mean.shape == (6, 7)
    finally:
        backend.disable_remote_screening()


# --------------------------------------------------------------------------- #
# session cache: prewarm + 3x->1x collapse
# --------------------------------------------------------------------------- #
def test_prewarm_collapses_repeat_passes(backend, sample):
    pats, cases = sample
    ref_mean = backend.predict(pats, cases[0]).mean
    ref_conv = backend.predict_convergence(pats, cases[0])
    calls = {"n": 0}

    def counting(_be, patterns, kases):
        calls["n"] += 1
        return _be._raw_forward_local(patterns, kases)

    backend.enable_remote_screening(counting, min_predictions=1, log=lambda m: None)
    try:
        backend.prewarm(pats, [cases[0]] * len(pats))
        assert calls["n"] == 1                 # one batched compute
        got_mean = backend.predict(pats, cases[0]).mean
        got_conv = backend.predict_convergence(pats, cases[0])
        backend.predict_extra(pats, cases[0])
        assert calls["n"] == 1                 # predict/conv/extra are memo hits
    finally:
        backend.disable_remote_screening()
    np.testing.assert_allclose(ref_mean, got_mean, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ref_conv, got_conv, atol=1e-6)


def test_prewarm_deduplicates(backend, sample):
    pats, cases = sample
    calls = {"n": 0, "sizes": []}

    def counting(_be, patterns, kases):
        calls["n"] += 1
        calls["sizes"].append(len(patterns))
        return _be._raw_forward_local(patterns, kases)

    backend.enable_remote_screening(counting, min_predictions=1, log=lambda m: None)
    try:
        # duplicate every pattern; the memo should compute each unique once.
        backend.prewarm(pats + pats, [cases[0]] * (2 * len(pats)))
    finally:
        backend.disable_remote_screening()
    assert calls["sizes"] == [len(pats)]       # 6 unique, not 12


def test_disable_clears_cache(backend, sample):
    pats, cases = sample
    backend.enable_remote_screening(lambda *_: None, min_predictions=1)
    assert backend._screen_cache is not None
    backend.disable_remote_screening()
    assert backend._screen_cache is None
    assert backend._remote is None


# --------------------------------------------------------------------------- #
# checkpoint freshness
# --------------------------------------------------------------------------- #
def test_checkpoint_fingerprint_stable_and_detects_change(tmp_path):
    ens = _make_ensemble(tmp_path, n=2)
    fp1 = rem.checkpoint_fingerprint(ens)
    assert rem.checkpoint_fingerprint(ens) == fp1        # stable
    meta = next(ens.glob("member_*/meta.json"))
    meta.write_text(meta.read_text(encoding="utf-8").replace(
        '"seed": 100', '"seed": 999'), encoding="utf-8")
    assert rem.checkpoint_fingerprint(ens) != fp1        # detects a retrain


def test_fingerprint_requires_members(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        rem.checkpoint_fingerprint(tmp_path / "empty")


# --------------------------------------------------------------------------- #
# full client transport (mocked ssh/scp) — end-to-end pack->wire->unpack
# --------------------------------------------------------------------------- #
class _FakeTransport:
    """In-memory stand-in for the remote filesystem + one-shot ssh runner."""

    def __init__(self, backend):
        self.backend = backend
        self.fs: dict[str, bytes] = {}
        self.commands: list[str] = []

    def scp_to(self, s, local, remote_rel, timeout=300):
        self.fs[Path(remote_rel).name] = Path(local).read_bytes()

    def scp_from(self, s, remote_rel, local, timeout=300):
        Path(local).write_bytes(self.fs[Path(remote_rel).name])

    def run_ssh(self, s, cmd, timeout=120, **kw):
        self.commands.append(cmd)
        if "remote_infer" in cmd:
            out = ri.run_request(self.backend, self.fs[rem._SCREEN_IN])
            self.fs[rem._SCREEN_OUT] = out
            return subprocess.CompletedProcess(cmd, 0, stdout=b"INFER_OK\n", stderr=b"")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"OK\n", stderr=b"")


def test_remote_infer_transport_roundtrip(backend, sample, monkeypatch):
    pats, cases = sample
    fake = _FakeTransport(backend)
    monkeypatch.setattr(rem, "ensure_checkpoint",
                        lambda s, d, **k: "~/lpopt_ws/_screen_ckpt")
    monkeypatch.setattr(rem, "pick_gpu", lambda s: "1")
    monkeypatch.setattr(rem, "run_ssh", fake.run_ssh)
    monkeypatch.setattr(rem, "_scp_to", fake.scp_to)
    monkeypatch.setattr(rem, "_scp_from", fake.scp_from)

    s = rem.RemoteSettings()
    mu, ls, cl = rem.remote_infer(s, "unused_ckpt", pats, cases, "ga80", device="cpu")
    local = backend._raw_forward_local(pats, cases)
    for a, b in zip(local, (mu, ls, cl)):
        np.testing.assert_array_equal(a.astype(np.float32), b)
    # the pinned GPU + one-shot module invocation are in the command
    infer_cmd = next(c for c in fake.commands if "remote_infer" in c)
    assert "CUDA_VISIBLE_DEVICES=1" in infer_cmd
    assert "-m lpopt.model.remote_infer" in infer_cmd


def test_make_remote_screener_backend_fallback(backend, sample, monkeypatch):
    """A screener whose transport fails is caught by the backend -> local, no raise."""
    pats, cases = sample

    def boom(s, d, *a, **k):
        raise subprocess.TimeoutExpired("ssh", 5)

    monkeypatch.setattr(rem, "remote_infer", boom)
    s = rem.RemoteSettings()
    screener = rem.make_remote_screener(s, "ckpt", "ga80", log=lambda m: None)
    backend.enable_remote_screening(screener, min_predictions=1, log=lambda m: None)
    try:
        out = backend.predict(pats, cases[0])   # must fall back, not raise
    finally:
        backend.disable_remote_screening()
    assert out.mean.shape == (6, 7)


# --------------------------------------------------------------------------- #
# probe + config coercion
# --------------------------------------------------------------------------- #
def test_probe_false_on_timeout(monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("ssh", 5)

    monkeypatch.setattr(rem, "run_ssh", timeout)
    assert rem.probe(rem.RemoteSettings(), timeout=1) is False


def test_probe_true_on_sentinel(monkeypatch):
    monkeypatch.setattr(
        rem, "run_ssh",
        lambda s, cmd, timeout=120, **k: subprocess.CompletedProcess(
            cmd, 0, stdout=b"LPOPT_OK\n", stderr=b""))
    assert rem.probe(rem.RemoteSettings(), timeout=1) is True


def test_normalize_remote_screening_coercion():
    assert _normalize_remote_screening(True) == "on"
    assert _normalize_remote_screening(False) == "off"
    assert _normalize_remote_screening("auto") == "auto"
    assert _normalize_remote_screening("AUTO") == "auto"
    assert _normalize_remote_screening("true") == "on"
    assert _normalize_remote_screening("false") == "off"
    assert _normalize_remote_screening("nonsense") == "off"


# --------------------------------------------------------------------------- #
# REAL remote-GPU vs local-CPU determinism (plan 4.7 req 2).  Runs whenever the
# gpu2-6000 server answers a 5 s probe; skips (never fails) when unreachable so a
# network-independent machine stays green.  Pins GPU 1 via the deck's [remote].
# --------------------------------------------------------------------------- #
def test_remote_gpu_matches_local_cpu_determinism():
    """Same champion + same inputs => remote GPU == local CPU to float tolerance.

    Loads :data:`RemoteSettings` from the deck so the deck-pinned GPU is honoured
    (as of 2026-07-24 that is GPU 0 — "GPU 1 사용 금지, GPU 0 전용").  The test's
    intent is cross-device determinism, so it reads whatever index the deck pins
    (only requiring it not be ``auto``) rather than hard-coding one.  Asserts the
    raw ensemble arrays agree and prints the measured max abs difference.
    """
    if not (STORE / "records.parquet").is_file():
        pytest.skip("store not present")
    if not CHAMPION.is_dir():
        pytest.skip(f"champion {CHAMPION.name} not present")
    s = rem.RemoteSettings.from_input(DECK)          # deck-pinned gpu (not auto)
    import tomllib
    deck_gpu = tomllib.loads(DECK.read_text(encoding="utf-8"))["remote"]["gpu"]
    assert str(s.gpu) == str(deck_gpu), "deck [remote].gpu must load verbatim"
    assert str(s.gpu) != "auto", "deck must pin a concrete GPU index (not auto)"
    if not rem.probe(s, timeout=5):
        pytest.skip(f"{s.user}@{s.host}:{s.port} did not answer a 5s probe")

    reader = StoreReader(STORE)
    rows = reader.records.iloc[:24]                  # tiny + short: good GPU neighbour
    pats = [unpack_pattern(str(p)) for p in rows["pattern"]]
    cases = [CaseKey(pair=str(p), feed=int(f))
             for p, f in zip(rows["case_pair"], rows["feed"])]

    local_be = PosValCnnBackend.from_dir(
        CHAMPION, store_dir=STORE, library_id="ga80", device="cpu")
    mu_l, ls_l, cl_l = local_be._raw_forward_local(pats, cases)
    mu_r, ls_r, cl_r = rem.remote_infer(s, CHAMPION, pats, cases, "ga80",
                                        device="cuda", timeout=600)

    assert mu_l.shape == mu_r.shape and ls_l.shape == ls_r.shape
    assert cl_l.shape == cl_r.shape
    d_mu = float(np.abs(mu_l.astype(np.float32) - mu_r).max())
    d_ls = float(np.abs(ls_l.astype(np.float32) - ls_r).max())
    d_cl_raw = float(np.abs(cl_l.astype(np.float32) - cl_r).max())
    sig = lambda x: 1.0 / (1.0 + np.exp(-x.astype(np.float32)))
    d_cl_prob = float(np.abs(sig(cl_l) - sig(cl_r)).max())

    print(f"\n[determinism] N={len(pats)} champion={CHAMPION.name} gpu={s.gpu}")
    print(f"[determinism] mu_z (regression mean, z-space): max|local-remote| = {d_mu:.3e}")
    print(f"[determinism] log_sigma (z-space):             max|local-remote| = {d_ls:.3e}")
    print(f"[determinism] conv (probability, sigmoid):     max|local-remote| = {d_cl_prob:.3e}")
    print(f"[determinism] conv (raw logit, informational): max|local-remote| = {d_cl_raw:.3e}")

    assert d_mu < _DETERMINISM_TOL_REG, (
        f"mu_z remote-vs-local {d_mu:.3e} > {_DETERMINISM_TOL_REG:.0e}")
    assert d_ls < _DETERMINISM_TOL_REG, (
        f"log_sigma remote-vs-local {d_ls:.3e} > {_DETERMINISM_TOL_REG:.0e}")
    assert d_cl_prob < _DETERMINISM_TOL_CONVP, (
        f"convergence-probability remote-vs-local {d_cl_prob:.3e} "
        f"> {_DETERMINISM_TOL_CONVP:.0e}")


# --------------------------------------------------------------------------- #
# OPTIONAL live smoke (never runs in CI) — reports real remote screen timing
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("LPOPT_LIVE_SMOKE") != "1",
                    reason="set LPOPT_LIVE_SMOKE=1 to run the live GPU smoke")
def test_live_smoke_remote_screen(sample):
    import time as _time
    s = rem.RemoteSettings.from_input(None)
    if not rem.probe(s, timeout=5):
        pytest.skip(f"{s.user}@{s.host}:{s.port} did not answer a 5s probe")
    pats, cases = sample
    ckpt = REPO_ROOT / "data" / "models" / "20260716_195130"
    if not ckpt.is_dir():
        pytest.skip("champion checkpoint not present for live smoke")
    t0 = _time.perf_counter()
    mu, ls, cl = rem.remote_infer(s, ckpt, pats, cases, "ga80", device="cuda")
    dt = _time.perf_counter() - t0
    print(f"\n[live smoke] remote screen of {len(pats)} patterns in {dt:.2f}s; "
          f"mu_z={mu.shape}")
    assert mu.shape[1] == len(pats)
    assert np.isfinite(mu).all()
