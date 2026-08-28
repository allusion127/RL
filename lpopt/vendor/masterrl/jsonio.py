"""Shared strict-JSON sanitizer and crash-safe atomic writer (F-06/F-09).

Every JSON artifact the pipeline writes must be strictly parseable
(``allow_nan=False``): a bare ``NaN``/``Infinity`` token silently breaks any
standards-compliant reader.  ``sanitize_json`` maps non-finite floats to
``None`` and unwraps NumPy scalars, ``dumps_strict`` serializes the sanitized
payload, and ``write_json_atomic`` replaces the destination atomically —
never overwriting it non-atomically on failure.  When the destination stays
locked (Windows virus scanners/indexers), the payload is preserved in a
``*.recovery.<pid>*`` sidecar and a warning string is returned so callers can
surface the degradation (``io_warnings`` + ``COMPLETED_WITH_IO_WARNING``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

# Windows: virus scanners/indexers can hold the destination open for a
# moment, so an atomic replace can fail with a transient WinError 5.
_REPLACE_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)


def sanitize_json(value: Any) -> Any:
    """Recursively coerce a payload to strictly serializable JSON values.

    NumPy scalars are unwrapped via ``.item()``, arrays become (sanitized)
    lists, tuples become lists, and non-finite floats become ``None``.
    """

    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


def dumps_strict(payload: object, *, indent: int | None = 2) -> str:
    """Serialize with ``allow_nan=False`` after sanitizing; newline-terminated."""

    return (
        json.dumps(
            sanitize_json(payload),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )


def write_json_atomic(
    path: str | Path,
    payload: object,
    *,
    retry_delays: Sequence[float] = _REPLACE_RETRY_DELAYS,
) -> str | None:
    """Atomically write strict JSON; return a warning string on degradation.

    The text is written to a per-process temporary sibling (flushed and
    fsynced) and then moved over the destination with ``os.replace``,
    retrying transient Windows sharing violations.  If every replace fails,
    the destination is NEVER overwritten in place: the payload is preserved
    as ``<stem>.recovery.<pid><suffix>`` next to it and the returned warning
    describes the situation (``None`` means a clean atomic write).
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_strict(payload)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    for delay in retry_delays:
        try:
            os.replace(temporary, target)
            return None
        except PermissionError:
            time.sleep(delay)
    try:
        os.replace(temporary, target)
        return None
    except PermissionError as error:
        recovery = target.with_name(
            f"{target.stem}.recovery.{os.getpid()}{target.suffix}"
        )
        try:
            os.replace(temporary, recovery)
        except OSError:
            temporary.unlink(missing_ok=True)
            return (
                f"atomic replace failed for {target} and the recovery sidecar "
                f"could not be written ({type(error).__name__}: {error}); "
                "payload for this update was lost"
            )
        return (
            f"atomic replace failed for {target} "
            f"({type(error).__name__}: {error}); payload preserved at "
            f"{recovery}"
        )


__all__ = ["sanitize_json", "dumps_strict", "write_json_atomic"]
