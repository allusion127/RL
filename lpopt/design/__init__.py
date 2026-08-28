"""Parametric fuel-design production chain (plan section 12, Phase A).

The ``lpopt.design`` subpackage turns a 5-axis assembly design spec into a MASTER
loading-pattern library, end to end:

    spec.FuelDesign            5-axis design + stable MASTER alias registry
    lattice.write_dec_deck     design -> dec_FA deck (template edit)
    lattice.run_decart         DeCART2D lattice run -> FA_<alias>.HGC / .out
    library.build_master_library  TotalBatcher4 -> paramA MAS_XSL / MAS_HFF
    coredeck                   synthesize cy1 fresh + reload MASTER decks
    bootstrap.make_band_restart   cy1 -> equilibrium chain -> bases/<folder>/MAS_RST.*
    package.assemble_package   FEASIBLE_PACKAGE layout + fuel_types ingest

Everything downstream (produce / verify harness) treats a ``paramA`` package
exactly like the native ga80 FEASIBLE_PACKAGE.
"""

from __future__ import annotations

from .spec import (
    ANCHOR_DESIGNS,
    DESIGN_GRID,
    DesignRegistry,
    FuelDesign,
    lhs_grid,
)

__all__ = [
    "ANCHOR_DESIGNS",
    "DESIGN_GRID",
    "DesignRegistry",
    "FuelDesign",
    "lhs_grid",
]
