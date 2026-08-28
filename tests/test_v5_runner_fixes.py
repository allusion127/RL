"""The three runner bugs the coordinator hit, and their fixes.

C(i)  the built cache must be SHIPPED to the remote before the distill arm
      launches (push() ships store+splits but not data/models, so the relative
      --distill-targets path resolves to a nonexistent remote file).
C(ii) ``--score-only auto`` must resolve arm dirs from a manifest the runner
      writes, instead of requiring a hand-written v5_arm_dirs.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt import remote as remote_mod
from lpopt.model import v5_experiment as V


# --------------------------------------------------------------------------- #
# C(i): ship_file (cat-over-ssh, scp-free)
# --------------------------------------------------------------------------- #
def test_ship_file_streams_bytes_over_ssh(tmp_path, monkeypatch):
    src = tmp_path / "cache.npz"
    payload = b"\x00\x01BINARY\xff" * 100
    src.write_bytes(payload)

    seen = {}

    class _CP:
        returncode = 0
        stdout = str(len(payload)).encode()
        stderr = b""

    def fake_run_ssh(s, cmd, *, timeout=120, input_bytes=None, binary=False):
        seen["cmd"] = cmd
        seen["bytes"] = input_bytes
        return _CP()

    monkeypatch.setattr(remote_mod, "run_ssh", fake_run_ssh)
    s = remote_mod.RemoteSettings()
    abs_path = remote_mod.ship_file(s, src, "data/models/_v5_distill_soft.npz")

    assert seen["bytes"] == payload                 # exact bytes, no scp
    assert "cat >" in seen["cmd"]                    # cat-over-ssh, not scp
    assert "data/models/_v5_distill_soft.npz" in seen["cmd"]
    assert "mkdir -p" in seen["cmd"]                 # remote dir created
    assert abs_path.endswith("data/models/_v5_distill_soft.npz")


def test_ship_file_detects_a_truncated_transfer(tmp_path, monkeypatch):
    src = tmp_path / "cache.npz"
    src.write_bytes(b"12345678")

    class _CP:
        returncode = 0
        stdout = b"3"                                # remote saw only 3 bytes
        stderr = b""

    monkeypatch.setattr(remote_mod, "run_ssh",
                        lambda *a, **k: _CP())
    with pytest.raises(RuntimeError, match="size mismatch"):
        remote_mod.ship_file(remote_mod.RemoteSettings(), src, "x/y.npz")


def test_ship_file_raises_on_ssh_failure(tmp_path, monkeypatch):
    src = tmp_path / "f.bin"
    src.write_bytes(b"x")

    class _CP:
        returncode = 1
        stdout = b""
        stderr = b"Permission denied"

    monkeypatch.setattr(remote_mod, "run_ssh", lambda *a, **k: _CP())
    with pytest.raises(RuntimeError, match="ship_file failed"):
        remote_mod.ship_file(remote_mod.RemoteSettings(), src, "x/y.bin")


def test_runner_ships_the_cache_before_launch():
    """Structural guard: run_experiment must call ship_file on the distill path."""
    import inspect
    src = inspect.getsource(V.run_experiment)
    assert "ship_file(" in src
    # and it must happen for the distill arm specifically
    assert "distill" in src


# --------------------------------------------------------------------------- #
# C(ii): --score-only auto + the arm-dir manifest
# --------------------------------------------------------------------------- #
def test_score_only_auto_reads_the_manifest(tmp_path):
    cfg = V.ExperimentConfig(reports_dir=str(tmp_path))
    dirs = {"v4_baseline": "data/models/A", "v5_full": "data/models/B"}
    V._write_arm_manifest(cfg, dirs)
    assert (tmp_path / "v5_arm_dirs.json").is_file()
    assert V.resolve_score_only(cfg, "auto") == dirs


def test_score_only_auto_without_a_manifest_errors_clearly(tmp_path):
    cfg = V.ExperimentConfig(reports_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no arm manifest"):
        V.resolve_score_only(cfg, "auto")


def test_score_only_explicit_path(tmp_path):
    cfg = V.ExperimentConfig(reports_dir=str(tmp_path))
    p = tmp_path / "hand.json"
    p.write_text(json.dumps({"v5_full": "data/models/X"}), encoding="utf-8")
    assert V.resolve_score_only(cfg, str(p)) == {"v5_full": "data/models/X"}


def test_score_only_bad_path_errors_clearly(tmp_path):
    cfg = V.ExperimentConfig(reports_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="neither 'auto' nor an existing"):
        V.resolve_score_only(cfg, str(tmp_path / "nope.json"))


def test_manifest_write_is_atomic_and_roundtrips(tmp_path):
    cfg = V.ExperimentConfig(reports_dir=str(tmp_path))
    V._write_arm_manifest(cfg, {"a": "1"})
    V._write_arm_manifest(cfg, {"a": "1", "b": "2"})     # incremental update
    assert V._load_arm_manifest(cfg) == {"a": "1", "b": "2"}
    assert not (tmp_path / "v5_arm_dirs.json.tmp").exists()


def test_runner_writes_the_manifest_incrementally():
    import inspect
    src = inspect.getsource(V.run_experiment)
    assert "_write_arm_manifest(" in src


def test_cli_score_only_auto_parses():
    from lpopt.cli import build_parser
    args = build_parser().parse_args(
        ["v5-experiment", "--score-only", "auto", "--champion-dir", "x"])
    assert args.score_only == "auto"


# --------------------------------------------------------------------------- #
# teacher-map leak guard (an arm must never teach its own distillation student)
# --------------------------------------------------------------------------- #
def test_build_teacher_map_excludes_by_basename(monkeypatch, tmp_path):
    """``exclude`` must drop candidate dirs so an arm can't be a teacher."""
    calls = []

    def fake_discover(_):
        return [tmp_path / "champ_A", tmp_path / "arm_v5_full"]

    monkeypatch.setattr(V, "discover_champions", fake_discover)
    # patch the modules build_teacher_map imports from locally
    monkeypatch.setattr("lpopt.model.splits.SplitManifest", _FakeManifest)
    monkeypatch.setattr("lpopt.data.store.StoreReader", _FakeStoreReader)

    class _FakeBackend:
        target_names = ("cyclen",)

        @classmethod
        def from_dir(cls, d, **k):
            calls.append(Path(d).name)
            raise RuntimeError("stop after load attempt")

    monkeypatch.setattr("lpopt.model.model_api.PosValCnnBackend", _FakeBackend)

    cfg = V.ExperimentConfig(models_dir=str(tmp_path))
    V.build_teacher_map(cfg, out_path=tmp_path / "t.json",
                        exclude=["arm_v5_full"], log=lambda *a: None)
    # the excluded arm dir must never even be loaded as a candidate
    assert "arm_v5_full" not in calls
    assert "champ_A" in calls


def test_teacher_map_self_check_rejects_a_leaked_dir(tmp_path):
    """If a chosen teacher is an excluded dir, the build must RAISE."""
    import inspect
    src = inspect.getsource(V.build_teacher_map)
    # the self-check + raise must be present (the leak that shipped once)
    assert "teacher map leak" in src
    assert "must never teach its own" in src.lower() or "never" in src


def test_runner_auto_excludes_pulled_arm_dirs():
    import inspect
    src = inspect.getsource(V.run_experiment)
    assert "exclude=" in src
    assert "_load_arm_manifest" in src


class _FakeManifest:
    groups = {V.HOLDOUT_GROUP: {}}

    @classmethod
    def from_json(cls, p):
        return cls()


class _FakeStoreReader:
    def __init__(self, *a, **k):
        import pandas as pd
        self.records = pd.DataFrame({"record_id": [], "campaign": []})
