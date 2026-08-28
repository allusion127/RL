"""Remote batch-inference RPC for the lean ``user_criteria`` screen (plan 4.7).

The lean screen + deepen runs ~45k deep-ensemble predictions.  On the local ~20
core CPU that is ~7 minutes; routed to the gpu2-6000 RTX PRO 6000 it is seconds.
This module is the payload contract + the one-shot remote entry point.

**Payload choice — send packed PATTERNS, encode remotely (not features).**  The
encoder is deterministic and ``fuel_types.parquet`` is already on the server
(pushed by :func:`lpopt.remote.push`), so re-encoding a pattern remotely is
byte-identical to encoding it locally.  The arithmetic decides it:

* encoded features are ``float32[N, 26, 19, 19]`` (+ ``[N, 10]`` globals) — for
  ``N = 45_000`` that is ``45000 * 26 * 361 * 4 B ≈ 1.69 GB``;
* a packed ``Pattern.canonical()`` string is ~626 chars (~626 B utf-8); ``45_000``
  of them is ``≈ 28 MB`` uncompressed (~2 MB gzip-over-scp).

So the pattern payload is ~60x smaller than the feature payload, and re-encoding
on a GPU box costs nothing.  We ship ``(canonical pattern, case pair, feed)`` per
item plus the ``library_id``; the remote loads the freshness-matched champion,
rebuilds its own encoder from the checkpoint meta, and runs
:meth:`PosValCnnBackend._raw_forward_local`.

The response carries the *raw* ensemble arrays exactly as the local choke point
returns them — ``mu_z`` / ``log_sigma`` ``[M, N, T]`` and ``conv_logit`` ``[M, N]``
— so the campaign's :meth:`predict` / :meth:`predict_extra` /
:meth:`predict_convergence` denorm + calibration stack is reused unchanged.

The pack / unpack helpers are torch-free (numpy + domain only) so tests can
exercise the serialization round-trip without loading a model; :func:`run_request`
needs a backend and is the shared body of both the CLI entry and the
determinism test.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.schema import unpack_pattern
from ..vendor.masterrl.domain import CaseKey, Pattern

#: Payload schema version — bumped if the wire format changes so a stale remote
#: install (older ``lpopt`` src) fails loudly instead of mis-decoding.
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# request (patterns + case context) serialization  — torch-free
# --------------------------------------------------------------------------- #
def _join_utf8(items: Sequence[str]) -> np.ndarray:
    """Newline-join strings into a ``uint8`` array (compact, non-pickle)."""
    return np.frombuffer("\n".join(items).encode("utf-8"), dtype=np.uint8)


def _split_utf8(arr: np.ndarray) -> list[str]:
    text = bytes(arr.tobytes()).decode("utf-8")
    return text.split("\n") if text else []


def pack_request(patterns: Sequence[Pattern], cases: Sequence[CaseKey],
                 library_id: str) -> bytes:
    """Serialize a screening batch to an NPZ byte blob (patterns + case context).

    ``patterns[i]`` is paired with ``cases[i]`` (a per-item :class:`CaseKey`, so a
    single request can span many pair-cells — exactly what the batched screen
    needs).  Only ``pattern.canonical()`` + ``(pair, feed)`` + ``library_id`` cross
    the wire; the remote re-encodes.
    """
    patterns = list(patterns)
    cases = list(cases)
    if len(patterns) != len(cases):
        raise ValueError(
            f"patterns ({len(patterns)}) and cases ({len(cases)}) length mismatch"
        )
    canon = [p.canonical() for p in patterns]
    pairs = [c.pair for c in cases]
    feeds = np.asarray([int(c.feed) for c in cases], dtype=np.int64)
    buf = io.BytesIO()
    np.savez(
        buf,
        schema=np.int64(SCHEMA_VERSION),
        pat_blob=_join_utf8(canon),
        pair_blob=_join_utf8(pairs),
        feeds=feeds,
        lib=_join_utf8([library_id]),
    )
    return buf.getvalue()


def unpack_request(blob: bytes) -> tuple[list[Pattern], list[CaseKey], str]:
    """Inverse of :func:`pack_request` — ``(patterns, cases, library_id)``."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        schema = int(z["schema"])
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"payload schema {schema} != {SCHEMA_VERSION} (stale remote src?)"
            )
        feeds = z["feeds"].astype(np.int64)
        canon = _split_utf8(z["pat_blob"])
        pairs = _split_utf8(z["pair_blob"])
        library_id = (_split_utf8(z["lib"]) or [""])[0]
    n = int(feeds.shape[0])
    if not (len(canon) == len(pairs) == n):
        raise ValueError(
            f"payload item-count mismatch: patterns={len(canon)} "
            f"pairs={len(pairs)} feeds={n}"
        )
    patterns = [unpack_pattern(s) for s in canon]
    cases = [CaseKey(pair=pairs[i], feed=int(feeds[i])) for i in range(n)]
    return patterns, cases, library_id


# --------------------------------------------------------------------------- #
# response (raw ensemble arrays) serialization  — torch-free
# --------------------------------------------------------------------------- #
def pack_response(mu_z: np.ndarray, log_sigma: np.ndarray,
                  conv_logit: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(
        buf,
        schema=np.int64(SCHEMA_VERSION),
        mu_z=np.ascontiguousarray(mu_z, dtype=np.float32),
        log_sigma=np.ascontiguousarray(log_sigma, dtype=np.float32),
        conv_logit=np.ascontiguousarray(conv_logit, dtype=np.float32),
    )
    return buf.getvalue()


def unpack_response(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        schema = int(z["schema"])
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"response schema {schema} != {SCHEMA_VERSION} (stale remote src?)"
            )
        return (z["mu_z"].astype(np.float32),
                z["log_sigma"].astype(np.float32),
                z["conv_logit"].astype(np.float32))


# --------------------------------------------------------------------------- #
# shared inference body (used by the CLI entry AND the determinism test)
# --------------------------------------------------------------------------- #
def run_request(backend, blob: bytes) -> bytes:
    """Decode a request, run the ensemble choke point, encode the response.

    Device-agnostic: ``backend`` may sit on ``cuda`` (remote) or ``cpu`` (test).
    ``backend._raw_forward_local`` re-encodes each pattern with the checkpoint's
    own encoder and returns the identical raw arrays the local path produces, so a
    CPU round-trip here is bit-identical to a direct local ``_raw_forward_local``.
    """
    patterns, cases, _library_id = unpack_request(blob)
    mu_z, log_sigma, conv_logit = backend._raw_forward_local(patterns, cases)
    return pack_response(mu_z, log_sigma, conv_logit)


# --------------------------------------------------------------------------- #
# CLI — one-shot remote entry (invoked over ssh: seconds-scale, no tmux)
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lpopt.model.remote_infer")
    ap.add_argument("--ckpt", required=True, help="champion ensemble dir (member_*)")
    ap.add_argument("--in", dest="in_path", required=True, help="request .npz")
    ap.add_argument("--out", dest="out_path", required=True, help="response .npz")
    ap.add_argument("--store-dir", default="data/store",
                    help="dir holding fuel_types.parquet (remote: ~/lpopt_ws/data/store)")
    ap.add_argument("--library-id", default=None,
                    help="library id override (else the request's / checkpoint's)")
    ap.add_argument("--device", default="cuda",
                    help="torch device for inference (cuda on the GPU box)")
    args = ap.parse_args(argv)

    # Import the backend lazily so the pack/unpack helpers stay torch-free.
    from .model_api import PosValCnnBackend

    blob = Path(args.in_path).read_bytes()
    _pats, _cases, req_lib = unpack_request(blob)
    library_id = args.library_id or req_lib or "ga80"

    device = args.device
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[remote_infer] cuda unavailable; using cpu", file=sys.stderr,
                  flush=True)
            device = "cpu"
    except Exception:  # noqa: BLE001 — torch import failure surfaces below
        pass

    backend = PosValCnnBackend.from_dir(
        args.ckpt, store_dir=args.store_dir, library_id=library_id, device=device
    )
    out = run_request(backend, blob)
    Path(args.out_path).write_bytes(out)
    n = len(_cases)
    print(f"[remote_infer] OK device={device} n={n} members={len(backend.members)} "
          f"targets={len(backend.target_names)}", flush=True)
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "pack_request", "unpack_request",
    "pack_response", "unpack_response",
    "run_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
