"""Remote GPU training infrastructure for gpu2-6000 (plan sec. 4.7).

``python -m lpopt.remote <cmd>`` sub-commands over SSH:

* ``env-check`` — GPU / driver / torch + sm_120 CUDA matmul smoke + disk.
* ``push``      — tar-over-scp the ``lpopt`` source + ``data/store`` +
  ``data/splits`` to ``~/lpopt_ws/{src,data}`` then ``pip install -e`` the source
  into the server venv.
* ``train``     — launch ``CUDA_VISIBLE_DEVICES=<gpu> python -m lpopt.model.train``
  in a ``lpopt_<ts>`` tmux session with a heartbeat + ``DONE``/``FAILED`` markers.
* ``status``    — marker state + heartbeat age + ``train.log`` tail + tmux list.
* ``pull``      — retrieve ``runs/<ts>`` (checkpoints + reports) into
  ``data/models/<ts>/``.

Connection settings default to the plan's ``[remote]`` values and are overridden
by a deck's ``[remote]`` table when ``--input`` is given — parsed with a tiny
local ``tomllib`` reader that deliberately does NOT import ``lpopt.config`` (owned
by a sibling agent).  The broken ``gpu2-6000(40.238)`` ssh alias is never used;
host/user/port are always explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ._proc import no_window_flags

# --- module defaults (plan sec. 4.7 [remote]) ------------------------------- #
DEFAULT_HOST = "HOST_238"
DEFAULT_USER = "USER"
DEFAULT_PORT = 8022
DEFAULT_WORKDIR = "~/lpopt_ws"
DEFAULT_ENV = "~/lpopt_ws/venv"
DEFAULT_GPU = "auto"
DEFAULT_TMUX_PREFIX = "lpopt"

_REPO_ROOT = Path(__file__).resolve().parents[1]     # 5_RL/
_NOISE_RE = re.compile(
    r"post-quantum|store now, decrypt later|openssh\.com|vars\.sh|"
    r"UMF before running|DPC\+\+|WARNING: connection|This session may be|"
    r"The server may need"
)


@dataclass
class RemoteSettings:
    host: str = DEFAULT_HOST
    user: str = DEFAULT_USER
    port: int = DEFAULT_PORT
    workdir: str = DEFAULT_WORKDIR
    env: str = DEFAULT_ENV
    gpu: str | int = DEFAULT_GPU
    tmux_prefix: str = DEFAULT_TMUX_PREFIX

    @classmethod
    def from_input(cls, path: str | Path | None) -> "RemoteSettings":
        """Read the ``[remote]`` table from a deck (tomllib) or use defaults."""
        if path is None:
            return cls()
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"deck not found: {p}")
        with open(p, "rb") as handle:
            raw = tomllib.load(handle)
        section = raw.get("remote", {}) or {}
        known = {f: section[f] for f in cls.__dataclass_fields__ if f in section}
        return cls(**known)

    # convenience
    @property
    def venv_python(self) -> str:
        return f"{self.env}/bin/python"

    @property
    def venv_pip(self) -> str:
        return f"{self.env}/bin/pip"

    def home_rel(self, *parts: str) -> str:
        """A path under workdir expressed relative to ``$HOME`` for scp."""
        base = self.workdir.replace("~/", "").rstrip("/")
        return "/".join([base, *parts])


# --------------------------------------------------------------------------- #
# ssh / scp primitives
# --------------------------------------------------------------------------- #
def _ssh_base(s: RemoteSettings) -> list[str]:
    return ["ssh", "-p", str(s.port), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
            f"{s.user}@{s.host}"]


def _scp_base(s: RemoteSettings) -> list[str]:
    return ["scp", "-P", str(s.port), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new"]


def _clean(text: str) -> str:
    """Drop server login / ssh-client noise lines from captured stdout."""
    return "\n".join(ln for ln in text.splitlines() if not _NOISE_RE.search(ln))


def run_ssh(s: RemoteSettings, remote_cmd: str, *, timeout: int = 120,
            input_bytes: bytes | None = None, binary: bool = False
            ) -> subprocess.CompletedProcess:
    cmd = _ssh_base(s) + [remote_cmd]
    return subprocess.run(cmd, input=input_bytes, capture_output=True,
                          timeout=timeout, **no_window_flags())


def _scp_to(s: RemoteSettings, local: Path, remote_rel: str, *, timeout: int = 300) -> None:
    dst = f"{s.user}@{s.host}:{remote_rel}"
    cp = subprocess.run(_scp_base(s) + [str(local), dst], capture_output=True,
                        timeout=timeout, **no_window_flags())
    if cp.returncode != 0:
        raise RuntimeError(f"scp push failed: {cp.stderr.decode(errors='replace')}")


def _scp_from(s: RemoteSettings, remote_rel: str, local: Path, *, timeout: int = 300) -> None:
    src = f"{s.user}@{s.host}:{remote_rel}"
    cp = subprocess.run(_scp_base(s) + [src, str(local)], capture_output=True,
                        timeout=timeout, **no_window_flags())
    if cp.returncode != 0:
        raise RuntimeError(f"scp pull failed: {cp.stderr.decode(errors='replace')}")


def ship_file(s: RemoteSettings, local: Path, remote_rel: str, *,
              timeout: int = 600) -> str:
    """Copy a single local file to ``$HOME/<remote_rel>`` via cat-over-ssh.

    Uses ``ssh ... 'cat > path'`` with the file streamed on stdin rather than
    ``scp``: scp mangles a source path that contains spaces / non-ASCII (this
    repo lives under a Korean path with a ``+`` and parentheses), whereas the
    local path never appears in the remote command here — only the bytes cross
    the wire.  Verifies the landed byte count so a truncated transfer fails loud.
    Returns the absolute remote path.
    """
    local = Path(local)
    data = local.read_bytes()
    home_rel = remote_rel.replace("~/", "").lstrip("/")
    remote_dir = home_rel.rsplit("/", 1)[0] if "/" in home_rel else "."
    abs_path = f"$HOME/{home_rel}"
    cp = run_ssh(s, f"mkdir -p $HOME/{remote_dir} && cat > {abs_path} && "
                    f"wc -c < {abs_path}",
                 input_bytes=data, timeout=timeout)
    if cp.returncode != 0:
        raise RuntimeError(
            f"ship_file failed: {cp.stderr.decode(errors='replace')}")
    landed = _clean(cp.stdout.decode(errors="replace")).strip().split()
    got = int(landed[-1]) if landed and landed[-1].isdigit() else -1
    if got != len(data):
        raise RuntimeError(
            f"ship_file size mismatch for {home_rel}: sent {len(data)} bytes, "
            f"remote reports {got}")
    return abs_path


# --------------------------------------------------------------------------- #
# gpu selection
# --------------------------------------------------------------------------- #
def pick_gpu(s: RemoteSettings) -> str:
    """Return the GPU index to pin: fixed if configured, else the idlest."""
    if str(s.gpu) != "auto":
        return str(s.gpu)
    cp = run_ssh(s, "nvidia-smi --query-gpu=index,memory.used "
                    "--format=csv,noheader,nounits", timeout=60)
    text = _clean(cp.stdout.decode(errors="replace"))
    best_idx, best_mem = "0", float("inf")
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*,\s*(\d+)", line)
        if m:
            idx, mem = m.group(1), int(m.group(2))
            if mem < best_mem:
                best_idx, best_mem = idx, mem
    return best_idx


# --------------------------------------------------------------------------- #
# env-check
# --------------------------------------------------------------------------- #
_ENV_CHECK_PY = (
    "import torch; "
    "d=torch.cuda.is_available(); "
    "name=torch.cuda.get_device_name(0) if d else 'cpu'; "
    "cap=torch.cuda.get_device_capability(0) if d else (0,0); "
    "a=torch.randn(512,512,device='cuda' if d else 'cpu'); "
    "b=torch.randn(512,512,device='cuda' if d else 'cpu'); "
    "c=(a@b); "
    "ok=bool(torch.isfinite(c).all()); "
    "print('TORCH', torch.__version__); "
    "print('CUDA_AVAIL', d); "
    "print('DEVICE', name); "
    "print('SM', 'sm_%d%d'%cap); "
    "print('MATMUL_OK', ok)"
)


def env_check(s: RemoteSettings) -> dict[str, Any]:
    print(f"[env-check] {s.user}@{s.host}:{s.port}  venv={s.env}", flush=True)
    driver = run_ssh(s, "nvidia-smi --query-gpu=index,name,driver_version,"
                        "memory.used,memory.total,utilization.gpu "
                        "--format=csv,noheader", timeout=60)
    gpus = _clean(driver.stdout.decode(errors="replace")).strip()
    print("GPUs:\n" + gpus, flush=True)
    smoke = run_ssh(s, f"{s.venv_python} -c \"{_ENV_CHECK_PY}\"", timeout=180)
    out = _clean(smoke.stdout.decode(errors="replace")).strip()
    err = _clean(smoke.stderr.decode(errors="replace")).strip()
    print("torch smoke:\n" + (out or "(no stdout)"), flush=True)
    if smoke.returncode != 0 and err:
        print("stderr:\n" + err, flush=True)
    disk = run_ssh(s, "df -h ~ | tail -1", timeout=60)
    disk_line = _clean(disk.stdout.decode(errors="replace")).strip()
    print("disk (~):", disk_line, flush=True)
    parsed = dict(
        ln.split(" ", 1) for ln in out.splitlines() if " " in ln
    ) if out else {}
    ok = parsed.get("MATMUL_OK") == "True" and smoke.returncode == 0
    print(f"[env-check] {'PASS' if ok else 'FAIL'}", flush=True)
    return {"gpus": gpus, "smoke": parsed, "disk": disk_line, "ok": ok}


# --------------------------------------------------------------------------- #
# push
# --------------------------------------------------------------------------- #
def _build_tar(members: list[tuple[Path, str]]) -> bytes:
    """Gzip tar of (path, arcname) pairs, in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, arc in members:
            if path.exists():
                tar.add(path, arcname=arc,
                        filter=lambda ti: None if "__pycache__" in ti.name
                        or ti.name.endswith(".pyc") else ti)
    return buf.getvalue()


def push(s: RemoteSettings, *, repo_root: Path | None = None,
         install: bool = True, data: bool = True) -> dict[str, Any]:
    """Ship the source (and, by default, ``data/store`` + ``data/splits``).

    ``data=False`` ships ONLY the source tree and leaves the remote's existing
    ``data/`` untouched — used when a run must train against a store snapshot the
    remote already holds (e.g. an A/B arm launched later must see the SAME data
    the earlier arms trained on, not a since-grown local store).
    """
    root = repo_root or _REPO_ROOT
    print(f"[push] {root} -> {s.user}@{s.host}:{s.workdir} (data={data})", flush=True)

    src_tar = _build_tar([(root / "lpopt", "lpopt"),
                          (root / "pyproject.toml", "pyproject.toml")])
    data_tar = b""
    if data:
        data_members = [(root / "data" / "store", "store"),
                        (root / "data" / "splits", "splits")]
        data_tar = _build_tar([(p, a) for p, a in data_members if p.exists()])
    print(f"[push] source tar {len(src_tar)/1e6:.1f} MB, "
          f"data tar {len(data_tar)/1e6:.1f} MB", flush=True)

    run_ssh(s, f"mkdir -p {s.workdir}/src {s.workdir}/data {s.workdir}/runs",
            timeout=60)
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "src.tgz"
        sp.write_bytes(src_tar)
        _scp_to(s, sp, s.home_rel("_src.tgz"))
        if data:
            dp = Path(td) / "data.tgz"
            dp.write_bytes(data_tar)
            _scp_to(s, dp, s.home_rel("_data.tgz"), timeout=600)
    extract = f"tar -xzf {s.workdir}/_src.tgz -C {s.workdir}/src && "
    if data:
        extract += f"tar -xzf {s.workdir}/_data.tgz -C {s.workdir}/data && "
    extract += (f"rm -f {s.workdir}/_src.tgz {s.workdir}/_data.tgz && echo EXTRACT_OK")
    cp = run_ssh(s, extract, timeout=180)
    ex = _clean(cp.stdout.decode(errors="replace"))
    if "EXTRACT_OK" not in ex:
        raise RuntimeError(f"remote extract failed: {cp.stderr.decode(errors='replace')}")
    print("[push] extracted", flush=True)

    result = {"src_bytes": len(src_tar), "data_bytes": len(data_tar), "installed": False}
    if install:
        print("[push] pip install -e (server venv)...", flush=True)
        cp = run_ssh(s, f"{s.venv_pip} install -e {s.workdir}/src -q && echo PIP_OK",
                     timeout=600)
        out = _clean(cp.stdout.decode(errors="replace"))
        result["installed"] = "PIP_OK" in out
        if not result["installed"]:
            print("[push] pip stderr:\n" +
                  _clean(cp.stderr.decode(errors="replace"))[-1500:], flush=True)
        else:
            print("[push] pip install OK", flush=True)
    return result


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
_RUN_SCRIPT = """#!/bin/bash
set -o pipefail
RUN="$HOME/{workdir_rel}/runs/{ts}"
mkdir -p "$RUN"
cd "$HOME/{workdir_rel}"
( while true; do date +%s > "$RUN/heartbeat"; sleep 15; done ) &
HB=$!
CUDA_VISIBLE_DEVICES={gpu} {venv_python} -m lpopt.model.train {train_args} \
    --out-dir "runs/{ts}" > "$RUN/train.log" 2>&1
RC=$?
kill $HB 2>/dev/null
echo $RC > "$RUN/rc"
if [ $RC -eq 0 ]; then touch "$RUN/DONE"; else touch "$RUN/FAILED"; fi
"""


def train(s: RemoteSettings, train_args: Sequence[str], *,
          ts: str | None = None) -> dict[str, Any]:
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    gpu = pick_gpu(s)
    session = f"{s.tmux_prefix}_{ts}"
    workdir_rel = s.workdir.replace("~/", "").rstrip("/")
    venv_python = s.venv_python.replace("~", "$HOME")
    args_str = " ".join(train_args)
    script = _RUN_SCRIPT.format(workdir_rel=workdir_rel, ts=ts, gpu=gpu,
                                venv_python=venv_python, train_args=args_str)
    print(f"[train] session={session} gpu={gpu} args='{args_str}'", flush=True)

    run_ssh(s, f"mkdir -p {s.workdir}/runs/{ts}", timeout=60)
    run_ssh(s, f"cat > {s.workdir}/runs/{ts}/run.sh",
            input_bytes=script.encode("utf-8"), timeout=60)
    launch = (f"tmux new-session -d -s {session} "
              f"'bash $HOME/{workdir_rel}/runs/{ts}/run.sh' && echo LAUNCHED")
    cp = run_ssh(s, launch, timeout=60)
    out = _clean(cp.stdout.decode(errors="replace"))
    launched = "LAUNCHED" in out
    if not launched:
        print("[train] launch stderr:\n" +
              _clean(cp.stderr.decode(errors="replace")), flush=True)
    print(f"[train] {'launched' if launched else 'FAILED to launch'} {session}",
          flush=True)
    return {"ts": ts, "session": session, "gpu": gpu, "launched": launched}


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def status(s: RemoteSettings, ts: str, *, tail: int = 20) -> dict[str, Any]:
    run = f"{s.workdir}/runs/{ts}"
    cmd = (
        f"echo '== markers =='; ls {run} 2>/dev/null | grep -E 'DONE|FAILED|rc' || echo none; "
        f"echo '== rc =='; cat {run}/rc 2>/dev/null || echo none; "
        f"echo '== heartbeat =='; if [ -f {run}/heartbeat ]; then "
        f"echo \"age_s=$(( $(date +%s) - $(cat {run}/heartbeat) ))\"; else echo none; fi; "
        f"echo '== tmux =='; tmux ls 2>/dev/null | grep {s.tmux_prefix}_{ts} || echo 'no session'; "
        f"echo '== log tail =='; tail -{tail} {run}/train.log 2>/dev/null || echo 'no log'"
    )
    cp = run_ssh(s, cmd, timeout=90)
    out = _clean(cp.stdout.decode(errors="replace"))
    print(out, flush=True)
    done = "DONE" in out.split("== rc ==")[0]
    failed = "FAILED" in out.split("== rc ==")[0]
    state = "done" if done else ("failed" if failed else "running")
    return {"ts": ts, "state": state, "raw": out}


# --------------------------------------------------------------------------- #
# pull
# --------------------------------------------------------------------------- #
def pull(s: RemoteSettings, ts: str, *, local_root: Path | None = None) -> dict[str, Any]:
    root = local_root or (_REPO_ROOT / "data" / "models")
    dest = root / ts
    dest.mkdir(parents=True, exist_ok=True)
    remote_tgz = f"{s.workdir}/_pull_{ts}.tgz"
    make = (f"tar -czf {remote_tgz} -C {s.workdir}/runs/{ts} . && echo TAR_OK")
    cp = run_ssh(s, make, timeout=180)
    if "TAR_OK" not in _clean(cp.stdout.decode(errors="replace")):
        raise RuntimeError(f"remote tar failed: {cp.stderr.decode(errors='replace')}")
    with tempfile.TemporaryDirectory() as td:
        local_tgz = Path(td) / f"pull_{ts}.tgz"
        _scp_from(s, s.home_rel(f"_pull_{ts}.tgz"), local_tgz, timeout=600)
        with tarfile.open(local_tgz, "r:gz") as tar:
            tar.extractall(dest)
    run_ssh(s, f"rm -f {remote_tgz}", timeout=60)
    members = sorted(p.name for p in dest.glob("member_*"))
    print(f"[pull] {ts} -> {dest}  ({len(members)} members: {members})", flush=True)
    return {"ts": ts, "dest": str(dest), "members": members}


# --------------------------------------------------------------------------- #
# screening inference (plan 4.7) — batch-offload the lean user_criteria screen
# --------------------------------------------------------------------------- #
DEFAULT_SCREEN_CKPT_REL = "_screen_ckpt"     # under workdir; champion mirror
_SCREEN_IN = "_screen_in.npz"
_SCREEN_OUT = "_screen_out.npz"


def probe(s: RemoteSettings, *, timeout: int = 5) -> bool:
    """Fast reachability check for ``remote_screening = "auto"``.

    Returns ``True`` iff a one-shot ssh completes and echoes the sentinel within
    ``timeout`` seconds.  Any timeout / transport error is a quiet ``False`` — the
    campaign silently stays on local CPU.
    """
    try:
        cp = run_ssh(s, "echo LPOPT_OK", timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return b"LPOPT_OK" in (cp.stdout or b"")


def checkpoint_fingerprint(local_dir: str | Path) -> str:
    """Fast content hash of a champion ensemble dir (meta.json compare).

    Hashes each ``member_*/meta.json`` (which carries seed, history, best_metrics
    and version stamps — unique per training run) plus ``calibration.json`` /
    ``backend.json`` / ``ensemble.json`` when present.  Deliberately does NOT read
    the large ``model.pt`` weight files: two checkpoints with identical metas +
    calibration are the same champion, so the meta hash is a sufficient and cheap
    freshness key.
    """
    d = Path(local_dir)
    members = sorted(p for p in d.glob("member_*") if p.is_dir())
    if not members:
        raise FileNotFoundError(f"no member_* checkpoints under {d}")
    h = hashlib.sha256()
    for md in members:
        h.update(md.name.encode("utf-8"))
        meta = md / "meta.json"
        h.update(meta.read_bytes() if meta.is_file() else b"")
    for extra in ("backend.json", "calibration.json", "ensemble.json"):
        p = d / extra
        if p.is_file():
            h.update(extra.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def ensure_checkpoint(s: RemoteSettings, local_dir: str | Path, *,
                      remote_rel: str = DEFAULT_SCREEN_CKPT_REL,
                      timeout: int = 600) -> str:
    """Guarantee the server carries the local champion; return its remote path.

    Compares the local :func:`checkpoint_fingerprint` against a ``FINGERPRINT``
    marker beside the remote mirror; pushes (tar-over-scp + extract) only when
    they differ, so a warm server skips the transfer entirely.
    """
    local_fp = checkpoint_fingerprint(local_dir)
    remote_ckpt = f"{s.workdir}/{remote_rel}"
    fp_file = f"{remote_ckpt}/FINGERPRINT"
    cp = run_ssh(s, f"cat {fp_file} 2>/dev/null || echo MISSING", timeout=60)
    remote_fp = _clean(cp.stdout.decode(errors="replace")).strip()
    if remote_fp == local_fp:
        return remote_ckpt

    print(f"[screen] champion fingerprint differs (remote={remote_fp[:12]}...); "
          f"pushing {Path(local_dir).name}", flush=True)
    tar = _build_tar([(Path(local_dir), remote_rel)])
    run_ssh(s, f"rm -rf {remote_ckpt} && mkdir -p {s.workdir}", timeout=60)
    with tempfile.TemporaryDirectory() as td:
        cp_path = Path(td) / "ckpt.tgz"
        cp_path.write_bytes(tar)
        _scp_to(s, cp_path, s.home_rel(f"{remote_rel}.tgz"), timeout=timeout)
    extract = (
        f"tar -xzf {s.workdir}/{remote_rel}.tgz -C {s.workdir} && "
        f"rm -f {s.workdir}/{remote_rel}.tgz && "
        f"printf '%s' '{local_fp}' > {fp_file} && echo CKPT_OK"
    )
    cp = run_ssh(s, extract, timeout=180)
    if "CKPT_OK" not in _clean(cp.stdout.decode(errors="replace")):
        raise RuntimeError(
            f"remote checkpoint sync failed: {cp.stderr.decode(errors='replace')}")
    print("[screen] champion synced to server", flush=True)
    return remote_ckpt


def remote_infer(s: RemoteSettings, local_ckpt_dir: str | Path,
                 patterns: Sequence[Any], cases: Sequence[Any], library_id: str,
                 *, device: str = "cuda", timeout: int = 300):
    """One-shot GPU batch inference; returns raw ``(mu_z, log_sigma, conv_logit)``.

    Freshness-syncs the champion, scp's a packed ``(pattern, case)`` request, runs
    ``python -m lpopt.model.remote_infer`` on the pinned GPU in a tmux-less ssh
    call (seconds-scale), scp's the response back, and unpacks it.  Raises on any
    failure so the caller (:meth:`PosValCnnBackend._compute_missing`) can fall
    back to local CPU.
    """
    from .model.remote_infer import pack_request, unpack_response

    remote_ckpt = ensure_checkpoint(s, local_ckpt_dir, timeout=timeout)
    gpu = pick_gpu(s)
    store_dir = f"{s.workdir}/data/store"
    blob = pack_request(patterns, cases, library_id)
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.npz"
        out_path = Path(td) / "out.npz"
        in_path.write_bytes(blob)
        _scp_to(s, in_path, s.home_rel(_SCREEN_IN), timeout=timeout)
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} {s.venv_python} -m lpopt.model.remote_infer "
            f"--ckpt {remote_ckpt} --in {s.workdir}/{_SCREEN_IN} "
            f"--out {s.workdir}/{_SCREEN_OUT} --store-dir {store_dir} "
            f"--library-id {library_id} --device {device} && echo INFER_OK"
        )
        cp = run_ssh(s, cmd, timeout=timeout)
        out = _clean(cp.stdout.decode(errors="replace"))
        if "INFER_OK" not in out:
            raise RuntimeError(
                "remote inference failed: "
                + _clean(cp.stderr.decode(errors="replace"))[-1500:])
        _scp_from(s, s.home_rel(_SCREEN_OUT), out_path, timeout=timeout)
        resp = out_path.read_bytes()
    run_ssh(s, f"rm -f {s.workdir}/{_SCREEN_IN} {s.workdir}/{_SCREEN_OUT}", timeout=60)
    return unpack_response(resp)


def make_remote_screener(s: RemoteSettings, local_ckpt_dir: str | Path,
                         library_id: str, *, device: str = "cuda",
                         timeout: int = 300, log=None):
    """Build the ``remote_fn(backend, patterns, cases)`` closure the backend calls.

    The ``backend`` argument is ignored — the server loads its own freshness-
    matched champion — so the same closure serves any backend instance.
    """
    def _screen(_backend, patterns, cases):
        if log is not None:
            log(f"[remote_screening] offloading {len(patterns)} predictions to "
                f"{s.user}@{s.host} (gpu={s.gpu})")
        return remote_infer(s, local_ckpt_dir, patterns, cases, library_id,
                            device=device, timeout=timeout)
    return _screen


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lpopt.remote")
    ap.add_argument("cmd", choices=["env-check", "env_check", "push", "train",
                                    "status", "pull"])
    ap.add_argument("--input", default=None, help="deck with a [remote] table")
    ap.add_argument("--ts", default=None, help="run timestamp (status/pull)")
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--split", default="S1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("train_args", nargs="*",
                    help="extra args after -- forwarded to lpopt.model.train")
    args = ap.parse_args(argv)

    s = RemoteSettings.from_input(args.input)
    cmd = args.cmd.replace("_", "-")

    if cmd == "env-check":
        return 0 if env_check(s)["ok"] else 1
    if cmd == "push":
        r = push(s, install=not args.no_install)
        return 0 if (r["installed"] or args.no_install) else 1
    if cmd == "train":
        targs = list(args.train_args) or [
            "--ensemble", str(args.ensemble), "--split", args.split,
            "--device", "auto", "--num-workers", str(args.num_workers),
        ]
        if args.epochs is not None and "--epochs" not in targs:
            targs += ["--epochs", str(args.epochs)]
        r = train(s, targs, ts=args.ts)
        print(f"launched ts={r['ts']} session={r['session']} gpu={r['gpu']}")
        return 0 if r["launched"] else 1
    if cmd == "status":
        if not args.ts:
            ap.error("status requires --ts")
        status(s, args.ts)
        return 0
    if cmd == "pull":
        if not args.ts:
            ap.error("pull requires --ts")
        pull(s, args.ts)
        return 0
    return 2


__all__ = ["RemoteSettings", "env_check", "push", "train", "status", "pull",
           "pick_gpu", "probe", "checkpoint_fingerprint", "ensure_checkpoint",
           "remote_infer", "make_remote_screener", "ship_file"]


if __name__ == "__main__":
    raise SystemExit(main())
