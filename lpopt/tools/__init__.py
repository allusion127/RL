"""One-off store maintenance utilities (backfills, hygiene passes).

Everything here is a *maintenance* entry point: it edits an existing store in
place rather than producing new physics.  Each module is runnable as
``python -m lpopt.tools.<name>`` and must be idempotent — re-running it on an
already-migrated store is a no-op that writes nothing.
"""

from __future__ import annotations

__all__: list[str] = []
