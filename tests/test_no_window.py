"""No visible console window for launched console EXEs (Windows UX fix).

Two guarantees:

1. ``no_window_flags()`` suppresses the child console window on Windows without
   disturbing output redirection — captured pipes *and* file-handle redirects
   still receive the child's stdout/stderr (CREATE_NO_WINDOW only removes the
   console window, not the streams).
2. Every ``subprocess.Popen(`` / ``subprocess.run(`` launch in the ``lpopt``
   package (tests excluded) either splats ``**no_window_flags()`` or is
   explicitly allowlisted with a justification (a launch that suppresses its
   window another way, e.g. DETACHED_PROCESS, or the pinned vendored file).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from lpopt._proc import no_window_flags

LPOPT_DIR = Path(__file__).resolve().parents[1] / "lpopt"


# --------------------------------------------------------------------------- #
# 1. helper shape + output redirection is unaffected by CREATE_NO_WINDOW
# --------------------------------------------------------------------------- #
def test_no_window_flags_shape() -> None:
    flags = no_window_flags()
    if os.name == "nt":
        assert flags == {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        assert flags == {}


def test_captured_output_survives_no_window_flags() -> None:
    """``subprocess.run(..., capture_output=True, **no_window_flags())`` still
    captures both stdout and stderr."""
    code = (
        "import sys; sys.stdout.write('OUT_MARKER'); "
        "sys.stderr.write('ERR_MARKER')"
    )
    cp = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        **no_window_flags(),
    )
    assert cp.returncode == 0
    assert "OUT_MARKER" in cp.stdout
    assert "ERR_MARKER" in cp.stderr


def test_file_redirect_survives_no_window_flags(tmp_path: Path) -> None:
    """A file-handle ``stdout=`` redirect (how the MASTER/DeCART launchers run)
    still receives the child's output with the no-window flags applied."""
    log = tmp_path / "child.log"
    code = "import sys; sys.stdout.write('FILE_MARKER'); sys.stderr.write('E2')"
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=fh,
            stderr=subprocess.STDOUT,
            **no_window_flags(),
        )
        assert proc.wait(timeout=60) == 0
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "FILE_MARKER" in text
    assert "E2" in text


# --------------------------------------------------------------------------- #
# 2. every subprocess launch in the package suppresses its console window
# --------------------------------------------------------------------------- #
# (relative posix path, enclosing function) -> justification for NOT splatting
# **no_window_flags().  Each entry suppresses the window by another mechanism.
_ALLOWLIST: dict[tuple[str, str], str] = {
    ("vendor/masterrl/master.py", "run"): (
        "Pinned vendored file: patched with a self-contained, os.name-guarded "
        "CREATE_NO_WINDOW inline (dict-splat) rather than importing the helper, "
        "to keep the vendored module standalone. Delta is recorded in "
        "VENDOR_MANIFEST.json ('patched: CREATE_NO_WINDOW')."
    ),
    ("curriculum.py", "_launch_produce"): (
        "Detached background produce relaunch: uses DETACHED_PROCESS, which gives "
        "the child no console at all and is mutually exclusive with "
        "CREATE_NO_WINDOW."
    ),
}


class _LaunchFinder(ast.NodeVisitor):
    """Collect (call_node, enclosing_function_name) for subprocess.Popen/run."""

    def __init__(self) -> None:
        self._stack: list[str] = []
        self.hits: list[tuple[ast.Call, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("Popen", "run")
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        ):
            self.hits.append((node, self._stack[-1] if self._stack else "<module>"))
        self.generic_visit(node)


def _splats_no_window_flags(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Call):  # **something()
            f = kw.value.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name == "no_window_flags":
                return True
    return False


def test_every_subprocess_launch_suppresses_window() -> None:
    offenders: list[str] = []
    seen_allowlisted: set[tuple[str, str]] = set()

    for py in sorted(LPOPT_DIR.rglob("*.py")):
        rel = py.relative_to(LPOPT_DIR).as_posix()
        finder = _LaunchFinder()
        finder.visit(ast.parse(py.read_text(encoding="utf-8")))
        for call, func in finder.hits:
            if _splats_no_window_flags(call):
                continue
            key = (rel, func)
            if key in _ALLOWLIST:
                seen_allowlisted.add(key)
                continue
            offenders.append(f"{rel}:{call.lineno} (in {func}())")

    assert not offenders, (
        "subprocess launch(es) neither splat **no_window_flags() nor are "
        "allowlisted:\n  " + "\n  ".join(offenders)
    )

    # Keep the allowlist honest: every entry must correspond to a real call site
    # (a stale entry would silently mask a regression).
    stale = set(_ALLOWLIST) - seen_allowlisted
    assert not stale, f"stale allowlist entries (no matching call site): {stale}"
