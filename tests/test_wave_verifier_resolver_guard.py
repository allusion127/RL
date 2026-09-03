"""``WaveVerifier`` alias-bridge hygiene (task #14).

``resolver=None`` is no longer the HGD569 defect — the constructor DERIVES the
resolver from ``package_root``, whose ``registry.json`` is where the
``type_id -> %LPD_B&C alias`` bridge lives (``verify.py`` comment citing memo
20260830).  What remains is narrow and silent:

* a paramA (multi-char ``type_id``) package with **no ``registry.json``**, and
* an explicit ``registry_aliases={}`` on a resolver for such a package,

both of which leave the bridge empty on a library whose type_ids are NOT batch
ids.  The wave then dies one deck at a time inside ``validate_reload_deck``
instead of at construction.  This pins the invariant the task states —
*multi-char-type library ⇒ ``resolver.type_to_alias != {}``* — and pins that
ga80 (``type_id == alias``, 2 chars, no bridge needed) is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.search.assets import CaseAssetResolver
from lpopt.search.verify import WaveVerifier, _is_multi_char_type_package

TYPE_IDS = ("P6253Z1G06N24", "P6253Z2G10N24")
REGISTRY = {TYPE_IDS[0]: "S3", TYPE_IDS[1]: "S5"}


def _paramA_package(root: Path, *, with_registry: bool) -> Path:
    """A paramA package: ``designs.json`` is the marker a ga80 package lacks."""
    pkg = root / "paramA_package"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "designs.json").write_text(
        json.dumps({"library_id": "paramA", "designs": [
            {"type_id": t, "alias": a} for t, a in REGISTRY.items()]}),
        encoding="utf-8",
    )
    if with_registry:
        (pkg / "registry.json").write_text(
            json.dumps({"aliases": REGISTRY}), encoding="utf-8")
    return pkg


def _ga80_package(root: Path) -> Path:
    """A ga80 FEASIBLE_PACKAGE: lib/bases/cores, no designs.json, no registry."""
    pkg = root / "ga80_package"
    (pkg / "lib").mkdir(parents=True, exist_ok=True)
    (pkg / "lib" / "MAS_XSL").write_text("COMP FA_E1\n", encoding="utf-8")
    return pkg


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
def test_multi_char_type_detection(tmp_path: Path) -> None:
    assert _is_multi_char_type_package(_paramA_package(tmp_path, with_registry=True))
    assert _is_multi_char_type_package(_paramA_package(tmp_path, with_registry=False))
    assert not _is_multi_char_type_package(_ga80_package(tmp_path))
    assert not _is_multi_char_type_package(None)
    assert not _is_multi_char_type_package(tmp_path / "does-not-exist")


def test_a_registry_of_multi_char_types_is_enough(tmp_path: Path) -> None:
    """No ``designs.json``, but the registry maps 13-char types -> 2-char aliases."""
    pkg = tmp_path / "bare"
    pkg.mkdir()
    (pkg / "registry.json").write_text(json.dumps({"aliases": REGISTRY}),
                                       encoding="utf-8")
    assert _is_multi_char_type_package(pkg)


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #
def test_paramA_package_without_a_registry_refuses_to_construct(tmp_path: Path) -> None:
    """THE #14 guard: the bridge cannot be derived, so say so HERE."""
    pkg = _paramA_package(tmp_path, with_registry=False)
    with pytest.raises(ValueError, match="EMPTY type_id -> alias bridge"):
        WaveVerifier(run_dir=tmp_path / "run", package_root=pkg)


def test_an_explicit_empty_bridge_is_refused_too(tmp_path: Path) -> None:
    """The other half of the residual risk: ``registry_aliases={}`` passed by
    hand to a resolver for a package that DOES have a registry."""
    pkg = _paramA_package(tmp_path, with_registry=True)
    resolver = CaseAssetResolver(pkg, library_id="paramA", registry_aliases={})
    assert resolver.type_to_alias == {}
    with pytest.raises(ValueError, match="EMPTY type_id -> alias bridge"):
        WaveVerifier(run_dir=tmp_path / "run", package_root=pkg, resolver=resolver)


def test_the_allow_flag_is_the_only_way_through(tmp_path: Path) -> None:
    pkg = _paramA_package(tmp_path, with_registry=False)
    verifier = WaveVerifier(run_dir=tmp_path / "run", package_root=pkg,
                            allow_missing_alias_bridge=True)
    assert verifier.resolver.type_to_alias == {}


def test_a_registry_backed_package_keeps_deriving_its_bridge(tmp_path: Path) -> None:
    """The invariant: multi-char-type library => a non-empty bridge."""
    pkg = _paramA_package(tmp_path, with_registry=True)
    verifier = WaveVerifier(run_dir=tmp_path / "run", package_root=pkg)
    assert _is_multi_char_type_package(Path(pkg))
    assert verifier.resolver.type_to_alias == REGISTRY
    assert verifier.resolver.alias_to_type == {a: t for t, a in REGISTRY.items()}


def test_an_explicit_resolver_is_still_the_one_used(tmp_path: Path) -> None:
    pkg = _paramA_package(tmp_path, with_registry=True)
    resolver = CaseAssetResolver(pkg, library_id="paramA")
    verifier = WaveVerifier(run_dir=tmp_path / "run", package_root=pkg,
                            resolver=resolver)
    assert verifier.resolver is resolver


# --------------------------------------------------------------------------- #
# ga80 / 2-char behaviour is untouched
# --------------------------------------------------------------------------- #
def test_ga80_package_constructs_with_an_empty_bridge(tmp_path: Path) -> None:
    """ga80 ``type_id == alias`` (2 chars): no bridge is needed and none is
    demanded — the pre-#14 behaviour, byte for byte."""
    pkg = _ga80_package(tmp_path)
    verifier = WaveVerifier(run_dir=tmp_path / "run", package_root=pkg)
    assert verifier.resolver.type_to_alias == {}


def test_no_package_root_constructs(tmp_path: Path, monkeypatch) -> None:
    """``package_root=None`` still constructs — but only because the resolver's
    own root (the CWD fallback) is not a multi-char-type package."""
    monkeypatch.chdir(tmp_path)
    verifier = WaveVerifier(run_dir=tmp_path / "run")
    assert verifier.package_root is None
    assert verifier.resolver.type_to_alias == {}


def test_no_package_root_is_not_a_hole_in_the_guard(tmp_path: Path,
                                                    monkeypatch) -> None:
    """THE residual-hole regression: HGD569 lived exactly here — an alias-less
    resolver rooted at the CWD with ``package_root=None``.  The detector must
    look at the RESOLVER's root when the verifier has none of its own."""
    pkg = _paramA_package(tmp_path, with_registry=True)
    resolver = CaseAssetResolver(pkg, library_id="paramA", registry_aliases={})
    with pytest.raises(ValueError, match="EMPTY type_id -> alias bridge"):
        WaveVerifier(run_dir=tmp_path / "run", resolver=resolver)

    # the CWD fallback is covered too: with a registry-less paramA package AS
    # the working directory, the derived resolver's bridge is empty and the
    # verifier must refuse instead of running the wave with untranslated ids.
    cwd_pkg = _paramA_package(tmp_path / "cwd", with_registry=False)
    monkeypatch.chdir(cwd_pkg)
    with pytest.raises(ValueError, match="EMPTY type_id -> alias bridge"):
        WaveVerifier(run_dir=tmp_path / "run")

    # ... and the allow flag remains the way through
    assert WaveVerifier(run_dir=tmp_path / "run",
                        allow_missing_alias_bridge=True).package_root is None
