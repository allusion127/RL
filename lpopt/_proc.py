"""Shared subprocess launch flags — suppress the Windows console window.

Every console EXE the package launches (MASTER, DeCART2D, TotalBatcher / prolog)
and the console CLI tools it shells out to (ssh / scp, ``tasklist``) allocate
their *own* console window on Windows when started via ``subprocess.Popen`` /
``subprocess.run`` — even when the launching Python process is itself hidden.
During a production wave that pops up to eight visible console windows on the
user's desktop at once.

Passing ``CREATE_NO_WINDOW`` in ``creationflags`` tells Windows not to allocate a
console for the child.  It does **not** touch stdout / stderr redirection: file
handles passed as ``stdout=`` / ``stderr=`` and ``capture_output=True`` pipes are
unaffected, so captured output keeps working exactly as before.

``no_window_flags()`` returns a kwargs dict to splat into the launch call —
``{"creationflags": subprocess.CREATE_NO_WINDOW}`` on Windows, ``{}`` on every
other platform (where ``CREATE_NO_WINDOW`` does not exist and no console is
allocated anyway).  Splatting the empty dict is a no-op, so the same call site is
correct cross-platform::

    subprocess.run(cmd, capture_output=True, **no_window_flags())

Note: ``CREATE_NO_WINDOW`` is mutually exclusive with ``DETACHED_PROCESS`` /
``CREATE_NEW_CONSOLE`` — a call that already detaches (its own console
suppression) must not also add these flags.  See ``curriculum._launch_produce``.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def no_window_flags() -> dict[str, Any]:
    """Popen/run kwargs suppressing the child console window on Windows.

    Returns ``{"creationflags": subprocess.CREATE_NO_WINDOW}`` on Windows and an
    empty dict elsewhere.  Splat with ``**no_window_flags()`` into any
    ``subprocess.Popen`` / ``subprocess.run`` call that launches a console tool.
    """

    if os.name == "nt":
        # CREATE_NO_WINDOW exists on the Windows build of CPython (>=3.7).
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


__all__ = ["no_window_flags"]
