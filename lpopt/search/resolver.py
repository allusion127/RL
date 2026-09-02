"""Shared config-driven :class:`CaseAssetResolver` factory (per-library routing).

Both the curriculum (:mod:`lpopt.curriculum`) and the plain production driver
(:mod:`lpopt.search.produce`) must build a resolver that routes assets by the
cell's effective fuel *library*:

* **ga80** (the harness-native FEASIBLE_PACKAGE) -> ``[verify].package_root`` +
  the configured ``[produce].template_fallbacks`` + the default MASTER dims.
* **paramA** (the on-demand parametric library) -> the assembled *design package*
  (its own ``bases/`` ``cores/`` ``lib/``), the registry ``type_id -> alias``
  bridge, the package's ``%GEN_DIM`` library dims, and **no** ga80
  ``template_fallbacks`` (a ga80 reload deck would otherwise win resolution and
  then fail the paramA ``%GEN_DIM`` sanity gate).  The reload template for a
  paramA case is synthesized from the package's own ``MAS_XSL`` roster.

This module is the single home of that routing so the curriculum and the produce
kit path stay in lock-step (a paramA stratum on a kit PC resolves *exactly* as a
paramA curriculum cell does).  ga80 callers get byte-identical behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .assets import (
    CaseAssetResolver,
    LIBRARY_DIMS,
    registry_aliases_from_package,
)

if TYPE_CHECKING:
    from ..config import LpoptConfig


def _base_dir(cfg: "LpoptConfig") -> Path:
    return cfg.source_path.parent if cfg.source_path else Path.cwd()


def _resolve_rel(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def paramA_package_root(cfg: "LpoptConfig") -> Path:
    """The assembled paramA design package dir (``[design]`` config).

    ``[design].package_root`` when set, else ``[design].store_dir / "package"``
    (mirrors ``CurriculumDriver._generate_band_designs``).
    """

    base = _base_dir(cfg)
    d = cfg.design
    return _resolve_rel(base, d.package_root) if d.package_root else (
        _resolve_rel(base, d.store_dir) / "package"
    )


def paramA_registry_aliases(pkg: Path) -> dict[str, str]:
    """``{type_id: alias}`` from the package ``registry.json`` (``{}`` if absent).

    Thin re-export of :func:`lpopt.search.assets.registry_aliases_from_package`,
    which is also what :class:`CaseAssetResolver` now calls for itself when a
    caller passes no ``registry_aliases``: ONE reader of the bridge, so a
    hand-wired caller and the resolver's own default can never disagree.
    """

    return registry_aliases_from_package(pkg)


def paramA_library_dims(pkg: Path, registry_aliases: Mapping[str, str]) -> tuple[int, int]:
    """MASTER ``(nbatch, ncomp)`` for the package's assembled library.

    Authoritative source is the ``lib/MAS_XSL`` COMP roster (what the coredeck's
    ``%GEN_DIM`` encodes); falls back to the registry alias count.
    """

    from ..design.coredeck import library_dims

    n = 0
    try:
        from ..design.bootstrap import library_aliases
        n = len(library_aliases(pkg))
    except Exception:  # noqa: BLE001 — MAS_XSL unreadable; fall back to registry
        n = len(registry_aliases)
    return library_dims(n) if n else LIBRARY_DIMS


def is_paramA_library(cfg: "LpoptConfig", library_id: str | None) -> bool:
    """A cell whose effective library is the on-demand paramA parametric library
    (routed to the design package) rather than the harness-native ga80 deck.

    The verdict is ``library_id == [curriculum].paramA_library`` and nothing
    else.  It used to also require ``lib_id != cfg.curriculum.library``, which is
    vacuously true for every shipped deck (``curriculum.library`` is ``ga80``)
    but would make this predicate ALWAYS false — silently routing every paramA
    cell back to the ga80 package — the moment an operator set
    ``curriculum.library = "paramA"`` (ECC audit 2026-08-12).
    """

    lib_id = library_id or cfg.curriculum.library
    paramA = getattr(cfg.curriculum, "paramA_library", "paramA")
    return bool(paramA) and lib_id == paramA


def build_case_resolver(
    cfg: "LpoptConfig", fuel_library: Any, library_id: str | None = None
) -> CaseAssetResolver:
    """Build the :class:`CaseAssetResolver` for ``library_id`` (per-library routing).

    ``library_id`` defaults to ``[curriculum].library`` (ga80).  A paramA library
    routes to the design package with the registry alias bridge + its own
    ``%GEN_DIM`` dims and no ga80 ``template_fallbacks``; any other library keeps
    the ga80 ``[verify].package_root`` + configured fallbacks path unchanged.
    """

    base = _base_dir(cfg)
    lib_id = library_id or cfg.curriculum.library
    promoted = _resolve_rel(base, cfg.produce.promoted_root)
    neutral = (
        _resolve_rel(base, cfg.produce.neutral_restart)
        if cfg.produce.neutral_restart
        else None
    )
    synth_root = _resolve_rel(base, cfg.produce.synth_decks_root)

    if is_paramA_library(cfg, lib_id):
        pkg = paramA_package_root(cfg)
        registry_aliases = paramA_registry_aliases(pkg)
        dims = paramA_library_dims(pkg, registry_aliases)
        return CaseAssetResolver(
            pkg, promoted,
            neutral_restart=neutral, template_fallbacks=[],
            fuel_library=fuel_library, library_id=lib_id,
            synth_root=synth_root,
            registry_aliases=registry_aliases, library_dims=dims,
        )

    package_root = _resolve_rel(base, cfg.verify.package_root) if cfg.verify.package_root else base
    fallbacks = [str(_resolve_rel(base, g)) for g in cfg.produce.template_fallbacks]
    return CaseAssetResolver(
        package_root, promoted,
        neutral_restart=neutral, template_fallbacks=fallbacks,
        fuel_library=fuel_library, library_id=lib_id,
        synth_root=synth_root,
    )


__all__ = [
    "build_case_resolver",
    "is_paramA_library",
    "paramA_library_dims",
    "paramA_package_root",
    "paramA_registry_aliases",
]
