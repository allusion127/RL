"""Vendored master_rl closure: torch-free imports + manifest integrity."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import lpopt.vendor.masterrl as masterrl

VENDOR_DIR = Path(masterrl.__file__).resolve().parent
MODULES = [
    "domain",
    "features",
    "dataset",
    "jsonio",
    "master",
    "burnup",
    "equilibrium",
    "parallel",
    "reward",
    "surrogate",
    "ga",
    "search",
]


def test_torch_free_import_closure() -> None:
    """Each vendored module imports in a fresh subprocess without pulling torch."""
    for mod in MODULES:
        code = (
            f"import lpopt.vendor.masterrl.{mod}\n"
            "import sys\n"
            "assert 'torch' not in sys.modules, 'torch imported by "
            f"{mod}'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{mod} import failed:\n{result.stderr}"


def test_manifest_integrity() -> None:
    """Every vendored file hashes to exactly what the manifest recorded."""
    manifest = json.loads((VENDOR_DIR / "VENDOR_MANIFEST.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    assert len(files) == 13, f"expected 13 vendored files, got {len(files)}"
    for name, meta in files.items():
        path = VENDOR_DIR / name
        assert path.exists(), f"vendored file missing: {name}"
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], f"hash mismatch: {name}"
        assert len(data) == meta["bytes"], f"byte-count mismatch: {name}"
