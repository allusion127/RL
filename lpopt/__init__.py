"""lpopt — APR1400 equilibrium-cycle loading-pattern optimization.

A supervised position-value model plus guided active search that reuses a
byte-pinned snapshot subset of the ``master_rl`` harness (see
``lpopt.vendor.masterrl``).  Milestone M0 provides the package scaffold, the
vendored snapshot, and the ``check`` / ``vendor-check`` CLI preflights.
"""

from __future__ import annotations

__version__ = "0.1.0"
